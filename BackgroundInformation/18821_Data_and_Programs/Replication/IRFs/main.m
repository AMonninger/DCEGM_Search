addpath('VAR')
addpath('LP')

clear all;
close all

%--------------------------------------------------------------------------
% 0. Parameters for estimation
%--------------------------------------------------------------------------

% specifications
nlag = 16;          % number of lags in specification
irfh = 32;          % horizon for IRF
cons = 2;           % 1= constant only, 2 = linear trend
datstart = 1960;    % estimation start date (must be >=1960)
datend = 2006;      % estimation end date (must be <=2006)

%--------------------------------------------------------------------------
% 1. Loading data
%--------------------------------------------------------------------------

% Import data from fred
names = {'ymin','ymax','gdp','gdpnom','fedfundsrate','qdurnom','qresnom','qressitenom','qresothernom','qnondurnom','qservnom','qservhousenom','pdurnom','presnom','pressitenom','presothernom','pnondurnom','pservnom','pservhousenom','qcarnom','pcarnom','pcarnewnom','pcarusednom','bondyield10','mortgagerate','carloanrate','pcarnewblsnom','pcarusedblsnom','phousenom','pop','pmedhousenom'};
load('../Data/freddownload',names{:},'dv','tyear')

% Import romer shock
num = xlsread('../Data/romershocks_quarterly.xlsx');
sdromer = std(num(:,end));
romershock = [NaN((1969-ymin)*4,1);num(:,end)./sdromer];
romershock(isnan(romershock))=0;

% Import CEX vehicle data
num = dlmread('../Data/cexvehiclemargins.txt',',',1,0);
qcarcexext = [NaN((1980-ymin)*4,1);num(1:(2007-1980+1)*4,4)];
qcarcex = [NaN((1980-ymin)*4,1);num(1:(2007-1980+1)*4,5)]./pop;
qcarcexint = [NaN((1980-ymin)*4,1);num(1:(2007-1980+1)*4,6)]./pop;

%--------------------------------------------------------------------------
% 2. Create variables for estimation
%--------------------------------------------------------------------------

% normalize housing services to same units
qservhousenom = qservhousenom/1000;

% real quantities per capita
qset = {'dur','res','ressite','resother','nondur','serv','servhouse','car'};
for kk=1:length(qset)
	eval(['q' qset{kk} 'nom = q' qset{kk} 'nom./ pop;']);
end
for kk=1:length(qset)
	eval(['q' qset{kk} ' = q' qset{kk} 'nom./p' qset{kk} 'nom ;']);
end
gdp = gdp./pop;

% taking rent out of services expenditure
[trash,pservXrentnom,qservXrentnom] = deflate(qservhousenom,qservnom,pservhousenom,pservnom,-1);

% aggregating nondurable goods and services excluding rent
[qnonall,pnonallnom,qnonallnom,trash,dlpnonallnom] = deflate(qnondurnom,qservXrentnom,pnondurnom,pservXrentnom);

% alternative price index for durables: replace site price index with house
% price index
[trash,presXsitenom,qresXsitenom] = deflate(qressitenom,qresnom,pressitenom,presnom,-1);
[trash,preshousenom,testres,trash,dlpreshousenom] = deflate(qresXsitenom,qressitenom,presXsitenom,phousenom);

% incorporate house price index into overall durable price index
[trash,pallhousenom,trash,trash2,dlpallhousenom] = deflate(qdurnom,qresnom,pdurnom,preshousenom);

% constructing real broad durable spending
[qall,pallnom,qallnom,trash,dlpallnom] = deflate(qdurnom,qresnom,pdurnom,presnom);

% use durables price when house price not available
dlpallhousenom(isnan(dlpallhousenom)) = dlpallnom(isnan(dlpallhousenom)); 
dlpallhousenom(1) = 0;
pallhousenom = exp(cumsum(dlpallhousenom));
dlpallhousenom(1) = NaN;

% use durables price when house price not available
dlpreshousenom(isnan(dlpreshousenom)) = dlpallnom(isnan(dlpreshousenom)); 
dlpreshousenom(1) = 0;
preshousenom = exp(cumsum(dlpreshousenom));
dlpreshousenom(1) = NaN;

% aggregate total spending
[qallnonall,pallnonallnom] = deflate(qallnom,qnonallnom,pallnom,pnonallnom);

% prices relative to nondurable prices 
pset = {'allhouse'};
for kk=1:length(pset)
	eval(['p' pset{kk} 'rel = p' pset{kk} 'nom ./ pnonallnom;']); 
end

% real federal funds rate in terms of nondurables and durables
nondur4Qinfl = zeros(length(pnonallnom),1);
dur4Qinfl    = zeros(length(pnonallnom),1);
for kk=1:4
   nondur4Qinfl = nondur4Qinfl + lagmatrix(dlpnonallnom,-kk);
   dur4Qinfl    = dur4Qinfl + lagmatrix(dlpallhousenom,-kk);
end

realrate = fedfundsrate - nondur4Qinfl*100;


% growth rate of real variables
dlset = {'gdp','qall','qnonall','qres','pallhouserel','qcarcexext','qcarcexint','qcarcex'};
for kk=1:length(dlset)
	eval(['l' dlset{kk} ' = log(' dlset{kk} ')*100;'])
    eval(['dl' dlset{kk} ' = [NaN;l' dlset{kk} '(2:end,:)-l' dlset{kk} '(1:end-1,:)];'])
end



%--------------------------------------------------------------------------
% 3. Run VAR models
%--------------------------------------------------------------------------

% we run the VAR models first to normalize the LP IRFs. For each of the
% following var models we specify the variables included, their names, and
% whether the IRFs for the variable are aggregated.
clear regvars
regsets{1} = [romershock,realrate];
regsets{2} = [romershock,dlpallhouserel];

regsetsnames{1} = {'romershock','c1drealrate'};
regsetsnames{2} = {'romershock','c1dlpallhouserel'};

regsettitles{1} = {'romershock','Real Interest Rate'};
regsettitles{2} = {'romershock','Relative Durable Price'};

aggmat{1} = [0,0];
aggmat{2} = [0,1];

% the first model establishes the scaling factor the IRF
RRnorm=0;

for kk=1:length(regsets)
        
    datavec0 = regsets{kk};


    % eliminate missing data
    tstart = max(find(sum(isnan(datavec0),2)==0, 1 ),1+(datstart-1960)*4);
    tend = length(datavec0)-(ymax-datend)*4;
    tnan = find(sum(isnan(datavec0(tstart:tend,:)),2)==1 , 1 );
    if isempty(tnan)==0
        tend = tstart+tnan-2;
    end

    % data
    datavec = datavec0(tstart:tend,:);
    varcons = cons;
    

    % VAR estimation
    VARstruct.nlag = nlag;
    VARstruct.cons = varcons;
    VARstruct.irfh = irfh;
    VARstruct.aggr = aggmat{kk};
    VARstruct.choldc = 1;

    VARstruct = VARestimate(datavec,[],VARstruct);


    % IRF & confidence bands
    VARstruct.irfci = 1;
    VARstruct.nblock=4;
    VARstruct.ndraws=2000;
    VARstruct.sebias=1;

    VARstructIRF = VARirf(datavec,[],VARstruct);
    IRFRR = VARstructIRF.IRF;
    IRFRRcil = VARstructIRF.IRFcil;
    IRFRRciu = VARstructIRF.IRFciu;

    % determine normalization to get 25bp shock on impact
    if RRnorm==0
        rloc = find(strcmp(regsetsnames{kk},'c1drealrate'));
        if isempty(rloc)==0
            RRnorm = 0.25/IRFRR(1,rloc,1);
        else
            RRnorm = 1;
        end
    end

    % monetary shock IRF tp 25bp Romer shock
    IRFRR = (IRFRR(:,:,1))*RRnorm;

    % CIs
    IRFRRcil = IRFRRcil(:,:,1)*RRnorm;
    IRFRRciu = IRFRRciu(:,:,1)*RRnorm;


    % store results
    for mm=1:length(regsetsnames{kk})
        eval(['irf.' regsetsnames{kk}{mm} num2str(kk)  '.irfmean = transpose(IRFRR(:,mm));'])
        eval(['irf.' regsetsnames{kk}{mm} num2str(kk)  '.name = ''' regsettitles{kk}{mm} ''';'])

        eval(['irf.' regsetsnames{kk}{mm} num2str(kk)  '.irfcil = transpose(IRFRRcil(:,mm));'])
        eval(['irf.' regsetsnames{kk}{mm} num2str(kk)  '.irfciu = transpose(IRFRRciu(:,mm));'])
        
        eval(['irf.' regsetsnames{kk}{mm} num2str(kk)  '.RRnorm = RRnorm;'])
    end  
    
    % Plot results for variables in the text
    if sum(strcmp(regsetsnames{kk}{2},{'c1drealrate','c1dlpallhouserel'}))>0

        % Plot for expansionary shock
        irfmean = -IRFRR(:,2);
        irfcil = -IRFRRcil(:,2,1);
        irfciu = -IRFRRciu(:,2,1);

        figure
        xlen = [0:1:irfh-1]';
        plot(xlen,irfmean,'k','Linewidth',3)
        hold on
        hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
        set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
        set(hp,'facealpha',0.2)
        plot(0:1:length(irfmean)-1,zeros(1,length(irfmean)),'k:','Linewidth',1)
        hold off
        xlim([0 length(irfmean)-1])
        ylabel('%')
        xlabel('Quarters')
        title( [regsettitles{kk}{2}] )
        set(gcf,'color','w');
        set(gca,'FontSize',16,'fontWeight','normal')
        set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
        
        saveas(gcf,['../Figures/irfRR-' regsetsnames{kk}{2} '.png'])
    end


end


%--------------------------------------------------------------------------
% 4. Run LP models
%--------------------------------------------------------------------------


% variables for which we run LP regressions
regcvars={'lgdp','lqall','lqnonall'};
regctitles={'GDP','Durable Expenditure','Nondurable Expenditure'};


% trend variable
trend = [transpose(1:1:length(gdp))./100];
constrendset = ones(length(gdp),1);
for cc=1:cons-1
    constrendset = [constrendset,trend.^cc];
end

% looping over variables
for kk=1:length(regcvars)
    
    % determine current variable
    disp(regcvars{kk})
    eval(['currvar = ' regcvars{kk} ';']) 

    % set estimation window
    yvar = currvar(1+(datstart-ymin)*4:end-(ymax-datend)*4);
    rvar = romershock(1+(datstart-ymin)*4:end-(ymax-datend)*4);
    consvar = constrendset(1+(datstart-ymin)*4:end-(ymax-datend)*4,:);

    % lagged dependent variables and monetary shocks
    xvar = [lagmatrix(rvar,0:1:nlag),lagmatrix(yvar,1:nlag),consvar];
    
    % determines whether we estimate IRF or cumulative IRF
    for aggr=[0,1] 

        if aggr==0
            addletter = '';
            addtitle = '';
        elseif aggr==1
            addletter = 'c1';
            addtitle = 'Cumulative ';
        end

        % check for which t we have data
        isdata = min(not(isnan([xvar,yvar]))')';
        Yvar = yvar(isdata,:);
        Xvar = xvar(isdata,:);

        % shorten x by the horizon of the IRF => balanced sample
        Xvar = Xvar(1:size(Xvar,1) - irfh,:);

        % run local projection
        [B,V] = LPreg(Yvar,Xvar,aggr);

        % normalize
        B = B*RRnorm;
        V = V*RRnorm^2;
        
        % store results
        eval(['irf.' addletter  regcvars{kk} 'LP.irfmean = transpose(B);'])
        eval(['irf.' addletter  regcvars{kk} 'LP.irfvcv = V;'])
        eval(['irf.' addletter  regcvars{kk} 'LP.irfcil = transpose(B-tinv(0.975,size(Xvar,1)-size(Xvar,2))*sqrt(diag(V)));'])
        eval(['irf.' addletter  regcvars{kk} 'LP.irfciu = transpose(B+tinv(0.975,size(Xvar,1)-size(Xvar,2))*sqrt(diag(V)));'])
        eval(['irf.' addletter  regcvars{kk} 'LP.name = ''' [addtitle regctitles{kk}] ''';'])
        


        % Plot results for variables in the text
        if sum(strcmp(regcvars{kk},{'lgdp','lqall'}))>0
            
            % Plot for expansionary shock
            eval(['irfmean = -transpose(irf.' addletter  regcvars{kk} 'LP.irfmean);'])
            eval(['irfcil = -transpose(irf.' addletter  regcvars{kk} 'LP.irfcil);'])
            eval(['irfciu = -transpose(irf.' addletter  regcvars{kk} 'LP.irfciu);'])
            
            figure
            xlen = [0:1:irfh]';
            plot(xlen,irfmean,'k','Linewidth',3)
            hold on
            hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
            set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
            set(hp,'facealpha',0.2)
            plot(0:1:length(irfmean)-1,zeros(1,length(irfmean)),'k:','Linewidth',1)
            hold off
            xlim([0 length(irfmean)-1])
            ylabel('%')
            xlabel('Quarters')
            title( [addtitle regctitles{kk}] )
            set(gcf,'color','w');
            set(gca,'FontSize',16,'fontWeight','normal')
            set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
            
            saveas(gcf,['../Figures/irfRR-' addletter regcvars{kk} '.png'])
            
        end
    end
end


%--------------------------------------------------------------------------
% 5. Run LP model for CEX (unbalanced sample starting in 1980)
%--------------------------------------------------------------------------

% CEX variables for which we estimate LPs
regcvarscex={'lqcarcexext','lqcarcexint','lqcarcex'};
regccextitle = {'Car Acquisition Probability','Car Intensive Margin','Car Expenditure'};

% car acquisition probability has quadratic trend
constrendset = ones(length(gdp),1);
for cc=1:2
    constrendset = [constrendset,trend.^cc];
end

for kk=1:length(regcvarscex)
    
    % determine current variable
    disp(regcvarscex{kk})
    eval(['currvar = ' regcvarscex{kk} ';']) 

    % set estimation window
    currvar = currvar(1+(datstart-ymin)*4:end-(ymax-datend)*4);
    rvar = romershock(1+(datstart-ymin)*4:end-(ymax-datend)*4);
    consvar = constrendset(1+(datstart-ymin)*4:end-(ymax-datend)*4,:);

    % lagged dependent variables and monetary shocks
    xvar = [lagmatrix(rvar,0:1:nlag),consvar];

    % add quarterly dummies given seasonal CEX data
    quart = transpose(1+mod(0:1:length(xvar)-1,4));
    xvar = [xvar,(quart==1),(quart==2),(quart==3)];

   
    for aggr=[0,1]
        if aggr==0
            addletter = '';
            addtitle = '';
        elseif aggr==1
            addletter = 'c1';
            addtitle = 'Cumulative '; 
        end
        
        aggvar = currvar;
        rvar = romershock(1+(datstart-ymin)*4:end-(ymax-datend)*4,1);

        for hh=0:irfh
            
            if aggr==1
                % cumulative variable when we aggregate
                aggvar = currvar + lagmatrix(aggvar,1);     
            end

            yvar = lagmatrix(aggvar,-hh);

            isdata = min(not(isnan([xvar,yvar]))')';
            N = sum(isdata);
            Yvar = yvar(isdata,:);
            Xvar = xvar(isdata,:);
            
            % coefficients on all dependent variables
            Ball   = (Xvar'*Xvar)\(Xvar'*Yvar);

            % residuals
            R = Yvar - Xvar*Ball;

            % coefficient on IRF
            B = Ball(1);
            
            % calculate standard errors with NW weight matrix
            weight = eye(N);
            for mm=1:hh
                weight = weight+diag((1-mm/(hh+1))*ones(N-mm,1),mm)+diag((1-mm/(hh+1))*ones(N-mm,1),-mm);
            end
            meat = (Xvar'*diag(R.^2)*Xvar);
            for mm=1:hh
                xvarp = Xvar(1+mm:end,:);
                xvarm = Xvar(1:end-mm,:);
                Rp = R(1+mm:end,1);
                Rm = R(1:end-mm,1);
                meat = meat + (1-mm/(hh+1))* ( (xvarm'*diag(Rp.*Rm)*xvarp) + (xvarp'*diag(Rm.*Rp)*xvarm) );
            end
            V = (Xvar'*Xvar)\( meat .*(N/(N-length(B))) )/(Xvar'*Xvar) ;
            
            % normalize
            B = B*RRnorm;
            V = V*RRnorm^2;

                        
            eval(['irf.' addletter  regcvarscex{kk} 'LP.irfmean(hh+1) = B(1);'])
            eval(['irf.' addletter  regcvarscex{kk} 'LP.irfsd(hh+1) = sqrt(V(1));'])
            eval(['irf.' addletter  regcvarscex{kk} 'LP.irfcil(hh+1) = B(1)-tinv(0.975,N-length(B))*sqrt(V(1));'])
            eval(['irf.' addletter  regcvarscex{kk} 'LP.irfciu(hh+1) = B(1)+tinv(0.975,N-length(B))*sqrt(V(1));'])
            eval(['irf.' addletter  regcvarscex{kk} 'LP.name = ''' [addtitle regccextitle{kk}] ''';'])
            
        end
        
        % Plot results for variables in the text
        if sum(strcmp(regcvarscex{kk},{'lqcarcexext'}))>0

            % Plot for expansionary shock
            eval(['irfmean = -transpose(irf.' addletter  regcvarscex{kk} 'LP.irfmean);'])
            eval(['irfcil = -transpose(irf.' addletter  regcvarscex{kk} 'LP.irfcil);'])
            eval(['irfciu = -transpose(irf.' addletter  regcvarscex{kk} 'LP.irfciu);'])

            figure
            xlen = [0:1:irfh]';
            plot(xlen,irfmean,'k','Linewidth',3)
            hold on
            hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
            set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
            set(hp,'facealpha',0.2)
            plot(0:1:length(irfmean)-1,zeros(1,length(irfmean)),'k:','Linewidth',1)
            hold off
            xlim([0 length(irfmean)-1])
            ylabel('%')
            xlabel('Quarters')
            title( [addtitle regccextitle{kk}] )
            set(gcf,'color','w');
            set(gca,'FontSize',16,'fontWeight','normal')
            set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
            
            saveas(gcf,['../Figures/irfRR-' addletter regcvarscex{kk} '.png'])
        end
    end
end

datairf = irf;


%--------------------------------------------------------------------------
% 6. Run annual LP on moving probability
%--------------------------------------------------------------------------

datairfann=LPannual(irfh/4,2,RRnorm,ymin,ymax);

