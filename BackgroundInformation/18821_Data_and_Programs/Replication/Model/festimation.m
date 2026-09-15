function [loss,output] = festimation(par,parest,parvalues,dataset,policy0,output)


if nargin<5
    policy0=[];
end
if nargin<6
    output = struct;
end
for kk=1:length(parest)
    par.(parest{kk})  = parvalues(kk);
end
if isfield(output,'probgapmodelhat')
    fgapmodelhat = output.fgapmodelhat;
    fgapmodellin = output.fgapmodellin;
    fgapmodeltrue = output.fgapmodeltrue;
    
    probgapmodelhat = output.probgapmodelhat;
    probgapmodellin = output.probgapmodellin;
    probgapmodeltrue = output.probgapmodeltrue;
    
    betaagg = output.betaagg;
    
    cmeanmodel = output.cmeanmodel;
    
    nmsets = size(probgapmodeltrue,1);
    
    gridgap = par.gridgap;
    gridint = griddedInterpolant(gridgap,transpose(1:1:length(gridgap)),'linear','nearest');
    
    measurementerrorset = par.measurementerrorsets;
else

%--------------------------------------------------------------------------
% Step 1: Solve model given estimated parameters
%--------------------------------------------------------------------------

% Nodes
[nodesgrids,par] = getnodes(par);

% Grids
nodesgrids = getgrids(par,nodesgrids);

% Drifts
nodesgrids = getdrifts(par,nodesgrids);
% tic
% Solve for policy function
if isempty(policy0)
    policy = fpolicy(par,nodesgrids);  
else
    policy = fpolicy(par,nodesgrids,policy0);
    if isfield(policy,'notconverged')
        policy = fpolicy(par,nodesgrids);  
    end
end
if isfield(policy,'notconverged')
    loss = NaN(1,6);
    return
end
output.policy = policy;
% toc

% Solve for steady state distribution
par.hazT=0;
steady = fsteady(par,nodesgrids,policy);
if isfield(steady,'noweight')
    loss = NaN(1,6);
    return
end
output.steady=steady;
psi0 = steady.psi0t./steady.ddist;

% distribution of durables
% psid = sum(sum(reshape(steady.psi0t,[par.Nz,par.Nd,par.Ny]),1),3)
% figure(9)
% plot(nodesgrids.nodesd,psid,'Linewidth',3)
%--------------------------------------------------------------------------
% Step 2: Conditional adjustment probabilities
%--------------------------------------------------------------------------
% tic
% only need to forecast for locations with positive weight
psipos = (steady.psi0t~=0);
psiposloc = find(psipos==1);

% These are the nodes for which we calculate expected consumption
probdist = 1;2;
psiposr = reshape(psipos,[par.Nz,par.Nnodesz/par.Nz]);
selection = [];
minloc = zeros(par.Nnodesz/par.Nz,1);
maxloc = zeros(par.Nnodesz/par.Nz,1);
for kk=1:par.Nnodesz/par.Nz
    if isempty(min(find(psiposr(:,kk)==1)))==0
        minloc(kk) = min(find(psiposr(:,kk)==1));
        maxloc(kk) = max(find(psiposr(:,kk)==1));
        selection = [selection;(kk-1)*par.Nz+minloc(kk);(kk-1)*par.Nz+transpose(minloc(kk)+1:probdist:maxloc(kk)-1);(kk-1)*par.Nz+maxloc(kk)];
    end
end
selection = intersect(selection,psiposloc);

% length of forecast
tf = 2;

% time interval of forecast (fine and corse)
probdt = 1/96;
probdt2= 1/12;

% forecast for prob of not adjusting
probnoadj2 = fforecast(par,nodesgrids,selection,steady.Cnoadj',1./steady.ddist,tf,tf,probdt,probdt2);
if min(probnoadj2)<-0.05 || max(probnoadj2)>1.1
    warning(['Probabilities outside boundaries: ' num2str(iter)])
    min(probnoadj2)
    max(probnoadj2)
end
probnoadj2 = max(min(probnoadj2,1),0);


% average quarterly consumption over next two years
if probdist>1
    % interpolate on nodes 
    probnoadj2r = reshape(probnoadj2,[par.Nz,par.Nnodesz/par.Nz]);
    for kk=1:par.Nnodesz/par.Nz
        if minloc(kk)>0 && maxloc(kk)>minloc(kk)
            selectionkk = [minloc(kk);transpose(minloc(kk)+1:probdist:maxloc(kk)-1);maxloc(kk)];
            selectionall = minloc(kk):1:maxloc(kk);
            probnoadj2r(selectionall,kk) = interp1(nodesgrids.nodesz(selectionkk),probnoadj2r(selectionkk,kk),nodesgrids.nodesz(selectionall));
        end
    end
    probnoadj2 = reshape(probnoadj2r,[par.Nnodesz,1]);
end
% max(abs(probnoadj2-probnoadj2base))
% nanmean(abs(probnoadj2-probnoadj2base))

% adjustment probability
probadj2 = 1-probnoadj2;
probadj2(steady.psi0t<=0)=1;


%--------------------------------------------------------------------------
% Step 3: Expected consumption 
%--------------------------------------------------------------------------

% operating cost
nu = par.nu;% - par.chi/(1-par.chi)*par.delta;

% distance between z nodes over which we interpolate
cdist = 4;6;

% These are the nodes for which we calculate expected consumption
% psiposr = reshape(psipos,[par.Nz,par.Nnodesz/par.Nz]);
% psiposr = reshape(psipos,[par.Nz,par.Nnodesz/par.Nz]);
selection = [];
for kk=1:par.Nnodesz/par.Nz
    if minloc(kk)>0
%         minloc(kk) = min(find(psiposr(:,kk)==1));
%         maxloc(kk) = max(find(psiposr(:,kk)==1));
        selection = [selection;(kk-1)*par.Nz+minloc(kk);(kk-1)*par.Nz+transpose(minloc(kk)+1:cdist:maxloc(kk)-1);(kk-1)*par.Nz+maxloc(kk)];
    end
end
selection = intersect(selection,psiposloc);

% time interval of forecast (fine and corse)
cdt = 1/96;1/60;
cdt2= 1/12;

% interval at which consumption is recorded
cdtm= 1/2;

% set of forecasts at each interval (annualized consumption)
cf2set = fforecast(par,nodesgrids,selection,steady.C',(policy.c+nu*nodesgrids.gridd)./steady.ddist,tf,cdtm,cdt,cdt2);
if min(min(cf2set))<-0.1 || max(max(cf2set))>10
    warning(['Consumption outside boundaries' num2str(iter)])
    min(min(cf2set))
    max(max(cf2set))
end
% min(min(cf2set))
%     max(max(cf2set))
% cf2set = max(min(cf2set,10),0);

% average quarterly consumption over next two years
cf2setmean = mean([(policy.c+nu*nodesgrids.gridd),cf2set],2);
if cdist>1
    % interpolate on nodes 
    cf2r = reshape(cf2setmean,[par.Nz,par.Nnodesz/par.Nz]);
    for kk=1:par.Nnodesz/par.Nz
        if minloc(kk)>0 && maxloc(kk)>minloc(kk)
            selectionkk = [minloc(kk);transpose(minloc(kk)+1:cdist:maxloc(kk)-1);maxloc(kk)];
            selectionall = minloc(kk):1:maxloc(kk);
            cf2r(minloc(kk):1:maxloc(kk),kk) = interp1(nodesgrids.nodesz(selectionkk),cf2r(selectionkk,kk),nodesgrids.nodesz(selectionall),'linear');
        end
    end
    cf2 = reshape(cf2r,[par.Nnodesz,1]);
else
    cf2 = cf2setmean;
end
% max(abs(cf2-cf2base)./cf2base)
% nanmean(abs(cf2-cf2base)./cf2base)
% toc
%--------------------------------------------------------------------------
% Step 4: Model regression
%--------------------------------------------------------------------------

% Mean nondurable (for scaling as in data)
cf2mean = cf2;
cf2mean(isnan(cf2mean))=0;
cmeanmodel = trapint(cf2mean.*psi0,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});

cf2mean = policy.c;
cf2mean(isnan(cf2mean))=0;
cmeanmodel = trapint(cf2mean.*psi0,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});

% Weighted regression:
yvar = policy.dprimeadj/cmeanmodel;
xvars = [nodesgrids.grida/cmeanmodel,nodesgrids.gridd/cmeanmodel,cf2/cmeanmodel,nodesgrids.gridd./cf2];
xvars = [xvars,(nodesgrids.grida/cmeanmodel).^2];
weights = max(steady.psi0t,0);
weights = weights./sum(weights);

% only use locations with positive weight
yvarreg = yvar(steady.psi0t>0,:);
xvarsreg = xvars(steady.psi0t>0,:);
weightsreg = weights(steady.psi0t>0,:);

mdl = fitlm(xvarsreg,yvarreg,'linear','Weights',weightsreg);

betaagg = mdl.Coefficients.Estimate;
output.betaagg=betaagg;
%--------------------------------------------------------------------------
% Step 5: Adding measurement error to x variables
%--------------------------------------------------------------------------

% Gaps space
gridgap = par.gridgap;%[par.gapmin:par.gapdelta:par.gapmax];

% mapping models to gap points
gridint = griddedInterpolant(gridgap,transpose(1:1:length(gridgap)),'linear','nearest');

% standard deviation
measurementerrorset = par.measurementerrorsets;
nmsets = length(measurementerrorset);

% distributions: density and conditional hazards
fgapmodelhat = zeros(nmsets,length(gridgap));
probgapmodelhat = ones(nmsets,length(gridgap));
fgapmodellin = zeros(nmsets,length(gridgap));
probgapmodellin = ones(nmsets,length(gridgap));
fgapmodeltrue = zeros(nmsets,length(gridgap));
probgapmodeltrue = ones(nmsets,length(gridgap));

% cashing densities
fgapdatatrue = zeros(1,length(gridgap));
probgapdatatrue = NaN(1,length(gridgap));
fgapdatalin = zeros(1,length(gridgap));
probgapdatalin = NaN(1,length(gridgap));
fgapdatahat = zeros(1,length(gridgap));
probgapdatahat = NaN(1,length(gridgap));
%     tic
for im=1:nmsets
    
    measurementerror=measurementerrorset(im);
    
    % tic
    % number of grid points
    nmerror = par.nmerror;
    nmerrorc = par.nmerrorc;
    [x, w] = GaussHermite(0,measurementerror,nmerror);
    [xc, wc] = GaussHermite(0,measurementerror*3,nmerrorc); % error of consumption
    dimnm = nmerror^2*nmerrorc;
    
    % modify model output with measurement error nodes
    ahat = ones(dimnm*par.Nnodesz,1);
    chat = ones(dimnm*par.Nnodesz,1);
    dhat = ones(dimnm*par.Nnodesz,1);
    psithat = zeros(dimnm*par.Nnodesz,1);
    probadj2hat = ones(dimnm*par.Nnodesz,1);
    for n1=1:nmerror
        for n2=1:nmerror
            for n3=1:nmerrorc
                loc = n1 + (n2-1)*nmerror + (n3-1)*nmerror*nmerror;

                % values with error
                ahat(1+(loc-1)*par.Nnodesz:loc*par.Nnodesz) = nodesgrids.grida.*exp(x(n1));
                dhat(1+(loc-1)*par.Nnodesz:loc*par.Nnodesz) = nodesgrids.gridd.*exp(x(n2));
                chat(1+(loc-1)*par.Nnodesz:loc*par.Nnodesz) = cf2.*exp(xc(n3));

                % weights
                psithat(1+(loc-1)*par.Nnodesz:loc*par.Nnodesz) = steady.psi0t.*w(n1).*w(n2).*wc(n3);

                % adjustments are correctly measured
                probadj2hat(1+(loc-1)*par.Nnodesz:loc*par.Nnodesz) = probadj2;
            end
        end
    end

    % set of x variables with measurement error
    xvarshat = [ahat/cmeanmodel,dhat/cmeanmodel,chat/cmeanmodel,dhat./chat,ahat/cmeanmodel.^2];
    % toc
    %--------------------------------------------------------------------------
    % Step 6: Model gaps
    %--------------------------------------------------------------------------

    

    

    % Three different predictions for dstar: true model, linear model, and
    % linear model with measurement error
    dprimeadjtrue = min(max(policy.dprimeadj,par.dmin),par.dmax)/cmeanmodel;
    dprimeadjlin = min(max(betaagg(1)+xvars*betaagg(2:end),par.dmin/cmeanmodel),par.dmax/cmeanmodel);
    dprimeadjhat = min(max(betaagg(1)+xvarshat*betaagg(2:end),par.dmin/cmeanmodel),par.dmax/cmeanmodel);

    % dist('try nonlinear policy function')
    % dprimeadjnonlinhat = 0;

    % Three different gaps:
    gapmodelhat = log(dprimeadjhat)-log(dhat/cmeanmodel);
    gapmodellin = log(dprimeadjlin)-log(nodesgrids.gridd/cmeanmodel);
    gapmodeltrue = log(dprimeadjtrue)-log(nodesgrids.gridd/cmeanmodel);

    % Mapping the three different gaps to the interval points in gaps space
    gapmodelinthat = max(min(gridint(gapmodelhat),length(gridgap)),1);
    gapmodelfloorhat = floor(gapmodelinthat);
    weightmodelhat = gapmodelinthat-gapmodelfloorhat; % weight on upper index 

    gapmodelintlin = max(min(gridint(gapmodellin),length(gridgap)),1);
    gapmodelfloorlin = floor(gapmodelintlin);
    weightmodellin = gapmodelintlin-gapmodelfloorlin; % weight on upper index 

    gapmodelinttrue = max(min(gridint(gapmodeltrue),length(gridgap)),1);
    gapmodelfloortrue = floor(gapmodelinttrue);
    weightmodeltrue = gapmodelinttrue-gapmodelfloortrue; % weight on upper index 

    

    for kk=1:length(gridgap)
        % measurement error
        fgapmodelhat(im,kk) = trapint(sum(reshape( ((1-weightmodelhat).*(gapmodelfloorhat==kk) + weightmodelhat.*(gapmodelfloorhat==kk-1) ).*psithat ,par.Nnodesz,dimnm),2)./steady.ddist,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
        probgapmodelhat(im,kk) = trapint(sum(reshape( ((1-weightmodelhat).*(gapmodelfloorhat==kk) + weightmodelhat.*(gapmodelfloorhat==kk-1) ).*probadj2hat.*psithat ,par.Nnodesz,dimnm),2)./steady.ddist,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy})/fgapmodelhat(im,kk);

        % linear model (no error)
        fgapmodellin(im,kk) = trapint( ((1-weightmodellin).*(gapmodelfloorlin==kk) + weightmodellin.*(gapmodelfloorlin==kk-1) ).*psi0 ,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
        probgapmodellin(im,kk) = trapint( ((1-weightmodellin).*(gapmodelfloorlin==kk) + weightmodellin.*(gapmodelfloorlin==kk-1) ).*probadj2.*psi0 ,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy})/fgapmodellin(im,kk);    

        % truth
        fgapmodeltrue(im,kk) = trapint( ((1-weightmodeltrue).*(gapmodelfloortrue==kk) + weightmodeltrue.*(gapmodelfloortrue==kk-1) ).*psi0,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
        probgapmodeltrue(im,kk) = trapint( ((1-weightmodeltrue).*(gapmodelfloortrue==kk) + weightmodeltrue.*(gapmodelfloortrue==kk-1) ).*probadj2.*psi0,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy})/fgapmodeltrue(im,kk);
    end
    

    % % plot
    % figure(1)
    % plot(gridgap,fgapmodeltrue,'b-',gridgap,fgapmodellin,'r--',gridgap,fgapmodelhat,'g-.','Linewidth',3)
    % legend('Truth','Lin','Hat')
    % 
    % % plot
    % figure(2)
    % plot(gridgap,probgapmodeltrue,'b-',gridgap,probgapmodellin,'r--',gridgap,probgapmodelhat,'g-.','Linewidth',3)
    % legend('Truth','Lin','Hat')
    
    
    
end

% cleanup
probgapmodelhat(isnan(probgapmodelhat))=1;
probgapmodellin(isnan(probgapmodellin))=1;
probgapmodeltrue(isnan(probgapmodeltrue))=1;

end

%--------------------------------------------------------------------------
% Step 7: Data gaps
%--------------------------------------------------------------------------

% importing data from dataset
adatahat = dataset.a;
ddatahat = dataset.d;
cdatahat = dataset.c;
adjdata = dataset.adj;
if isfield(dataset,'atrue')
    adatatrue = dataset.atrue;
    ddatatrue = dataset.dtrue;
    cdatatrue = dataset.ctrue;
else
    adatatrue = dataset.a;
    ddatatrue = dataset.d;
    cdatatrue = dataset.c;
end

% length of dataset
ndata = length(adatahat);
if isfield(dataset,'weight')
    datasetweight = reshape(dataset.weight,[1,ndata])./sum(dataset.weight);
else
    datasetweight = ones(1,ndata)/ndata;
end

% average nondurable consumption
cmeandatahat = datasetweight*cdatahat;
if isfield(dataset,'atrue')
    cmeandatatrue = datasetweight*cdatatrue;
else
    cmeandatatrue = cmeandatahat;
end

% predict dprimeadj
xvarsdatalin = [adatatrue/cmeandatatrue,ddatatrue/cmeandatatrue,cdatatrue/cmeandatatrue,ddatatrue./cdatatrue,(adatatrue/cmeandatatrue).^2];
xvarsdatahat = [adatahat/cmeandatahat,ddatahat/cmeandatahat,cdatahat/cmeandatahat,ddatahat./cdatahat,(adatahat/cmeandatahat).^2];

dprimeadjdatalin = betaagg(1)+xvarsdatalin*betaagg(2:end);
dprimeadjdatahat = betaagg(1)+xvarsdatahat*betaagg(2:end);

dprimeadjdatalin = min(max(dprimeadjdatalin,par.dmin/cmeanmodel),par.dmax/cmeanmodel);
dprimeadjdatahat = min(max(dprimeadjdatahat,par.dmin/cmeanmodel),par.dmax/cmeanmodel);

% implied gaps
gapdatalin = log(dprimeadjdatalin)-log(ddatatrue/cmeandatatrue);
gapdatahat = log(dprimeadjdatahat)-log(ddatahat/cmeandatahat);

if isfield(dataset,'dprimeadj')
    dprimeadjdata = dataset.dprimeadj/cmeandatatrue;
    gapdatatrue = log(dprimeadjdata)-log(ddatatrue/cmeandatatrue);
else
    gapdatatrue = gapdatalin;
end

output.gapdatahat = gapdatahat;
output.dprimeadjdatahat = dprimeadjdatahat;
output.ddata = ddatahat/cmeandatahat;

% density and hazard at gap points
gapdatainttrue = max(min(gridint(gapdatatrue),length(gridgap)),1);
gapdatafloortrue = floor(gapdatainttrue);
weightdatatrue = max(min(gapdatainttrue-gapdatafloortrue,1),0); % weight on upper index 

gapdataintlin = max(min(gridint(gapdatalin),length(gridgap)),1);
gapdatafloorlin = floor(gapdataintlin);
weightdatalin = max(min(gapdataintlin-gapdatafloorlin,1),0); % weight on upper index 

gapdatainthat = max(min(gridint(gapdatahat),length(gridgap)),1);
gapdatafloorhat = floor(gapdatainthat);
weightdatahat = max(min(gapdatainthat-gapdatafloorhat,1),0); % weight on upper index 

for kk=1:length(gridgap)
    % truth
    fgapdatatrue(kk) = datasetweight*( (1-weightdatatrue).*(gapdatafloortrue==kk) + weightdatatrue.*(gapdatafloortrue==kk-1) ); 
    probgapdatatrue(kk) = ( datasetweight*( ( (1-weightdatatrue).*(gapdatafloortrue==kk) + weightdatatrue.*(gapdatafloortrue==kk-1) ) .*adjdata ) ) / fgapdatatrue(kk);

    % linear model
    fgapdatalin(kk) = (datasetweight*( (1-weightdatalin).*(gapdatafloorlin==kk) + weightdatalin.*(gapdatafloorlin==kk-1) ));  
    probgapdatalin(kk) = ( datasetweight*( ( (1-weightdatalin).*(gapdatafloorlin==kk) + weightdatalin.*(gapdatafloorlin==kk-1) ) .*adjdata ) ) / fgapdatalin(kk);

    % measurement error
    fgapdatahat(kk) = (datasetweight*( (1-weightdatahat).*(gapdatafloorhat==kk) + weightdatahat.*(gapdatafloorhat==kk-1) ));  
    probgapdatahat(kk) = ( datasetweight*( ( (1-weightdatahat).*(gapdatafloorhat==kk) + weightdatahat.*(gapdatafloorhat==kk-1) ) .*adjdata ) ) / fgapdatahat(kk);

end



% % plot
% gridgapplot = gridgap;(gridgap(2:end)+gridgap(1:end-1))/2;
% 
% figure(3)
% plot(gridgapplot,fgapdatatrue,'b-',gridgapplot,fgapdatalin,'r--',gridgapplot,fgapdatahat,'g-.','Linewidth',3)
% 
% figure(4)
% plot(gridgapplot,probgapdatatrue,'b-',gridgapplot,probgapdatalin,'r--',gridgapplot,probgapdatahat,'g-.','Linewidth',3)
% 
% % figure(5)
% % plot(gridgapplot,fgapdatatrue,'b-',gridgap,fgapmodeltrue,'r--','Linewidth',3)
% % legend('Model','Data')
% 
% figure(5)
% plot(gridgapplot,fgapdatahat,'b-',gridgap,fgapmodelhat,'r--','Linewidth',3)
% legend('Data','Model')
% 
% figure(6)
% plot(gridgapplot,probgapdatahat,'b-',gridgap,probgapmodelhat,'r--','Linewidth',3)
% legend('Data','Model')


% toc

fgapdatalin(isnan(fgapdatalin))=0;
fgapdatahat(isnan(fgapdatahat))=0;
if isfield(dataset,'dprimeadj')
    fgapdatatrue(isnan(fgapdatatrue))=0;
else
    fgapdatatrue = NaN(1,length(gridgap));
    probgapdatatrue = NaN(1,length(gridgap));
end

probgapdatalin(isnan(probgapdatalin))=1;
probgapdatahat(isnan(probgapdatahat))=1;
if isfield(dataset,'dprimeadj')
    probgapdatatrue(isnan(probgapdatatrue))=1;
end


    

% %--------------------------------------------------------------------------
% % Step 8: Other statistics to match
% %--------------------------------------------------------------------------
% 
% % liquid assets
% amodel = steady.a/cmeanmodel;
% adata = par.targets.a/cmeanmodel;%(datasetweight*adatahat)/cmeandatahat;
% 
% % dc ratio
% dcmodel = steady.dc;
% dcdata = par.targets.dc;%(datasetweight*ddatahat)/(datasetweight*cdatahat);
% 
% % data adjustment frequency
% adjmodel = trapint(probadj2.*psi0,{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
% adjdata = datasetweight*adjdata;



%--------------------------------------------------------------------------
% Step 9: Loss function
%--------------------------------------------------------------------------

% criterion
for im=1:nmsets
    weighttrue = (0.5*probgapdatatrue+0.5*probgapmodeltrue(im,:)).*(0.5*fgapdatatrue+0.5*fgapmodeltrue(im,:));
    weighttrue = weighttrue/sum(weighttrue);
    losstrue(im,1) = trapz(gridgap,weighttrue.*abs(fgapmodeltrue(im,:)-fgapdatatrue)+weighttrue.*abs(probgapmodeltrue(im,:)-probgapdatatrue));
    
    weightlin = (0.5*probgapdatalin+0.5*probgapmodellin(im,:)).*(0.5*fgapdatalin+0.5*fgapmodellin(im,:));
    weightlin = weightlin/sum(weightlin);
    losslin(im,1) = trapz(gridgap,weightlin.*abs(fgapmodellin(im,:)-fgapdatalin)+weightlin.*abs(probgapmodellin(im,:)-probgapdatalin));
    
    weighthat = (0.5*probgapdatahat+0.5*probgapmodelhat(im,:)).*(0.5*fgapdatahat+0.5*fgapmodelhat(im,:));
    weighthat = weighthat/sum(weighthat);
    losshat(im,1) = trapz(gridgap,weighthat.*abs(fgapmodelhat(im,:)-fgapdatahat)+weighthat.*abs(probgapmodelhat(im,:)-probgapdatahat));
end

% find lowest loss:
bestnm = find(losshat==min(losshat));
% losshat = losshat(bestnm);
% losslin = losslin(bestnm);
% losstrue = losstrue(bestnm);

% store measurement error
output.measurementerror = measurementerrorset(bestnm);

% toc
weight = 10;
loss = weight*[losshat(bestnm),losslin(bestnm),losstrue(bestnm)];

% % adding other criteria
% loss2 = loss + abs(amodel-adata)./adata + abs(dcmodel-dcdata)./dcdata + abs(adjmodel-adjdata)./adjdata;
% loss = [loss,loss2];

%--------------------------------------------------------------------------
% Step 10: Put into structure
%--------------------------------------------------------------------------

output.fgapmodeltrue    = fgapmodeltrue;
output.fgapdatatrue     = fgapdatatrue;
output.fgapmodellin     = fgapmodellin;
output.fgapdatalin      = fgapdatalin;
output.fgapmodelhat     = fgapmodelhat;
output.fgapdatahat      = fgapdatahat;

output.probgapmodeltrue	= probgapmodeltrue;
output.probgapdatatrue	= probgapdatatrue;
output.probgapmodellin	= probgapmodellin;
output.probgapdatalin	= probgapdatalin;
output.probgapmodelhat	= probgapmodelhat;
output.probgapdatahat	= probgapdatahat;

output.bestnm = bestnm;

% output.adjmodel = adjmodel;
% output.adjdata = adjdata;
% 
% output.amodel = amodel;
% output.adata = adata;
% 
% output.dcmodel = dcmodel;
% output.dcdata = dcdata;

output.cmeanmodel = cmeanmodel;

output.weighthat = (0.5*probgapdatahat+0.5*probgapmodelhat).*(0.5*fgapdatahat+0.5*fgapmodelhat);
output.weighthat = output.weighthat./(sum(output.weighthat,2)*ones(1,length(gridgap)));

output.losshat = losshat;

% gridgapplot = gridgap;(gridgap(2:end)+gridgap(1:end-1))/2;
% figure(5)
% plot(gridgapplot,fgapdatahat,'b-',gridgap,fgapmodelhat(bestnm,:),'r--','Linewidth',3)
% legend('Data','Model')
% 
% figure(6)
% plot(gridgapplot,probgapdatahat,'b-',gridgap,probgapmodelhat(bestnm,:),'r--','Linewidth',3)
% legend('Data','Model')


% warning(['completed: ' num2str(iter)])


end
