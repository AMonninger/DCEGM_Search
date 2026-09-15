% simple CT model of durables with fixed cost, income uncertainty, leverage
% non-uniform grid
 
clear all;
close all;



% Initial Model Parameters
parfcoc = setparametersall('fcoc');
parfcoctheta = setparametersall('fcoctheta');

parfcoc.rho = 0.076202394013524;
parfcoc.psid = 0.536972206377407;
parfcoc.fc = 0.010181530414015;

par = parfcoc;

% saving string
savehandle = 'allest';

% controls
saveplots = 1;  % save plots

%--------------------------------------------------------------------------
% Step 0: Parameters for loss calculation
%-------------------------------------------------------------------------- 

% grid of thetas we use
par.thetasets = [0:0.01:0.12,0.1225:0.0025:0.155,0.156,0.157,0.1575,0.158,0.1585,0.159];

% grid of measurement error we use
par.measurementerrorsets = 0:0.005:0.15;

% grids for loss calculations
par.nmerror=6;
par.nmerrorc=18;
par.gridgap = [-1.15:0.15:-0.7,-0.6:0.1:1.2,1.35:0.15:1.95];
par.itdisp=200;


%--------------------------------------------------------------------------
% Step 1: load data
%--------------------------------------------------------------------------


filename = ['../Data/estimationdata.csv'];
psidstruct = importdata(filename, ',', 1);

% vehicles
psiddatadur.d = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'JWdurREALscale')));
psiddatadur.a = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'JWaREALscale')));
psiddatadur.c = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'JWnondurREAL')));
psiddatadur.adj = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'adjustdur')));
psiddatadur.weight = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'weight')));
psiddatadur.id = psidstruct.data(:,find(strcmp(psidstruct.colheaders,'myID')));

% alldatafields=fieldnames(psiddatadur);
% alldatafields(end,:)=[];
% mat=[];
% for datafield=alldatafields'
%     mat = [mat,psiddatadur.(datafield{1})];
%     nansum(psiddatadur.weight.*psiddatadur.(datafield{1})/sum(psiddatadur.weight));
% end
% sum(psiddatadur.weight);




%--------------------------------------------------------------------------
% Step 2: Find best theta
%-------------------------------------------------------------------------- 



loss = zeros(length(par.thetasets),3);
expandgrid=1;
convcrit = 0.001;
parest = {'rho','fc','psid','theta'};


for kk=1:length(par.thetasets) 
    
    par.theta = par.thetasets(kk);
    
    % predict other parameters
    if kk==1
        par.targetpar.rho = parfcoc.rho;
        par.targetpar.psid = parfcoc.psid;
        par.targetpar.fc = parfcoc.fc;
        policy0 = [];
    elseif kk==length(par.thetasets) 
        par.targetpar.rho = 0.106892258342314;
        par.targetpar.psid = 0.595144171488674;
        par.targetpar.fc = 0.252346151384704;
        convcrit = 0.00125;
        policy0 = policyset{kk-1};    
    else
        par.targetpar.rho = parset{kk-1}.rho;
        par.targetpar.psid = parset{kk-1}.psid;
        par.targetpar.fc = parset{kk-1}.fc;
        policy0 = policyset{kk-1};
    end


    if par.theta>0.155
        par.hazdt = 1/960;
        par.hazT = 960;
        par.hazTlong = 960;
    end
 
    % find solution
    [parset{kk},policyset{kk}] = fsteadytargets(par,par.targets,par.targetpar,convcrit,policy0,expandgrid);

    parset{kk}.gridgap = par.gridgap;
%     parset{kk}.nmerrorc = par.nmerrorc;
    
    
    [loss(kk,:),outputset{kk}] = festimation(parset{kk},parest,[parset{kk}.rho,parset{kk}.fc,parset{kk}.psid,parset{kk}.theta],psiddatadur,policyset{kk});
    loss
    try
        save('Results/allestimation.mat','loss','outputset','parset','policyset','-append')
    catch
        save('Results/allestimation.mat','loss','outputset','parset','policyset')
    end
    
end


%--------------------------------------------------------------------------
% Step 3: Construct standard errors
%-------------------------------------------------------------------------- 

load('Results/allestimation.mat','parset','policyset','loss','outputset')

% dataset in matrix form
datamat = [psiddatadur.d,psiddatadur.a,psiddatadur.c,psiddatadur.adj,psiddatadur.weight,psiddatadur.id];

% ids
ids = unique([psiddatadur.id]);

% sort obs into cell by ids
for kk=1:length(ids)
    datacell{kk} = datamat(datamat(:,6)==ids(kk),:);
end

nsimul = 2000;
s = RandStream('mlfg6331_64');

datamatrepall = datasample(s,1:1:length(ids),length(ids)*nsimul);

% theta candidates
thetarep = NaN(nsimul,1);
fgapmodelhatrep = NaN(nsimul,length(par.gridgap));
fgapdatahatrep = NaN(nsimul,length(par.gridgap));
probgapmodelhatrep = NaN(nsimul,length(par.gridgap));
probgapdatahatrep = NaN(nsimul,length(par.gridgap));

parest = {'rho','fc','psid','theta'};

for ss=1:nsimul
    
    ss
    
    % initial loss
    lossrep = zeros(length(par.thetasets),3);
   
    % draw from dataset with replacement
    datamatrep = [];
    for ii=1:length(ids)
        datamatrep = [datamatrep;datacell{datamatrepall(ii+(ss-1)*length(ids))}];
    end

    % allocate to structure
    psiddatadurrep.d = datamatrep(:,1);
    psiddatadurrep.a = datamatrep(:,2);
    psiddatadurrep.c = datamatrep(:,3);
    psiddatadurrep.adj = datamatrep(:,4);
    psiddatadurrep.weight = datamatrep(:,5);

    for kk=1:length(par.thetasets)
        [lossrep(kk,:),outputsetrep{kk}] = festimation(parset{kk},parest,[parset{kk}.rho,parset{kk}.fc,parset{kk}.psid,parset{kk}.theta],psiddatadurrep,policyset{kk},outputset{kk});
    end
    
    % best estimate
    minlossloc = find(lossrep(:,1)==min(lossrep(:,1)));
    
    thetarep(ss) = par.thetasets(minlossloc);
    fgapmodelhatrep(ss,:) = outputsetrep{minlossloc}.fgapmodelhat(outputsetrep{minlossloc}.bestnm,:);
    fgapdatahatrep(ss,:) = outputsetrep{minlossloc}.fgapdatahat(1,:);
    probgapmodelhatrep(ss,:) = outputsetrep{minlossloc}.probgapmodelhat(outputsetrep{minlossloc}.bestnm,:);
    probgapdatahatrep(ss,:) = outputsetrep{minlossloc}.probgapdatahat(1,:);
end

% confidence band:
cil = 2.5;
ciu = 97.5;

% sort everything
thetarep = sort(thetarep);
fgapmodelhatrep = sort(fgapmodelhatrep);
fgapdatahatrep = sort(fgapdatahatrep);
probgapmodelhatrep = sort(probgapmodelhatrep);
probgapdatahatrep = sort(probgapdatahatrep);

thetarepcil = thetarep(floor(cil/100*nsimul),1);
thetarepciu = thetarep(ceil(ciu/100*nsimul),1);

fgapmodelhatrepcil = fgapmodelhatrep(floor(cil/100*nsimul),:);
fgapmodelhatrepciu = fgapmodelhatrep(ceil(ciu/100*nsimul),:);
fgapdatahatrepcil = fgapdatahatrep(floor(cil/100*nsimul),:);
fgapdatahatrepciu = fgapdatahatrep(ceil(ciu/100*nsimul),:);
probgapmodelhatrepcil = probgapmodelhatrep(floor(cil/100*nsimul),:);
probgapmodelhatrepciu = probgapmodelhatrep(ceil(ciu/100*nsimul),:);
probgapdatahatrepcil = probgapdatahatrep(floor(cil/100*nsimul),:);
probgapdatahatrepciu = probgapdatahatrep(ceil(ciu/100*nsimul),:);

save('Results/allestimationplot.mat','par','loss','outputset','fgapmodelhatrepcil','fgapmodelhatrepciu','fgapdatahatrepcil','fgapdatahatrepciu','probgapmodelhatrepcil','probgapmodelhatrepciu','probgapdatahatrepcil','probgapdatahatrepciu')

%--------------------------------------------------------------------------
% Step 4: Plots
%-------------------------------------------------------------------------- 

load('Results/allestimationplot.mat','par','loss','outputset','fgapmodelhatrepcil','fgapmodelhatrepciu','fgapdatahatrepcil','fgapdatahatrepciu','probgapmodelhatrepcil','probgapmodelhatrepciu','probgapdatahatrepcil','probgapdatahatrepciu')

gridgap = par.gridgap;
kk=find(loss(:,1)==min(loss(:,1)));

figure(1)
plot(gridgap,outputset{kk}.probgapdatahat,'b',gridgap,outputset{kk}.probgapmodelhat(outputset{kk}.bestnm,:),'r-','Linewidth',3)
hold on
xlen = gridgap';
hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [probgapdatahatrepcil'; probgapdatahatrepciu(end:-1:1)';probgapdatahatrepcil(1)],'k');
set(hp, 'facecolor', 'b', 'edgecolor', 'none');
set(hp,'facealpha',0.2)
hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [probgapmodelhatrepcil'; probgapmodelhatrepciu(end:-1:1)';probgapmodelhatrepcil(1)],'k');
set(hp, 'facecolor', 'r', 'edgecolor', 'none');
set(hp,'facealpha',0.2)
xlim([-1 1.5])
ylim([0,1]);
legend('Data','Model')
xlabel('Durable Gap, log points')
title('Adjustment Probability, next 2 Years')
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
if saveplots
    saveas(gcf,['../Figures/estimation-hazard.png'])
end

figure(2)
plot(gridgap,outputset{kk}.fgapdatahat,'b',gridgap,outputset{kk}.fgapmodelhat(outputset{kk}.bestnm,:),'r-','Linewidth',3)
hold on
xlen = gridgap';
hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [fgapdatahatrepcil'; fgapdatahatrepciu(end:-1:1)';fgapdatahatrepcil(1)],'k');
set(hp, 'facecolor', 'b', 'edgecolor', 'none');
set(hp,'facealpha',0.2)
hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [fgapmodelhatrepcil'; fgapmodelhatrepciu(end:-1:1)';fgapmodelhatrepcil(1)],'k');
set(hp, 'facecolor', 'r', 'edgecolor', 'none');
set(hp,'facealpha',0.2)
xlim([-1 1.5])
legend('Data','Model')
xlabel('Durable Gap, log points')
title('Density')
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
if saveplots
    saveas(gcf,['../Figures/estimation-density.png'])
end

%--------------------------------------------------------------------------
% Step 5: Statistics for text
%-------------------------------------------------------------------------- 

loc = 1:length(gridgap);(gridgap>=-1 & gridgap<=1.5);

mdl = fitlm(outputset{kk}.probgapmodelhat(outputset{kk}.bestnm,loc),outputset{kk}.probgapdatahat(1,loc),'linear')

mdlw = fitlm(outputset{kk}.probgapmodelhat(outputset{kk}.bestnm,loc),outputset{kk}.probgapdatahat(1,loc),'linear','Weight',outputset{kk}.weighthat(outputset{kk}.bestnm,loc))
