% This file solves for the IRF to a government spending shock

clear all; close all;

% models
model = 'all'; 

% parameterizations
spec = 'fcoctheta';

savehandle = [model spec];

% load other structures we need for our analyis
load(['Results/' savehandle '.mat'],'steady','par');
steady.ypot=steady.AD;
steady.yaggnew2=steady.yaggnew;

% controls
saveplots = 1;  % save plots
savext = 'png';

% controls for computing IRFs
computeirf = 0;

shocksize = 0.01;

% Periods for which we set output gap to zero
Tgap = 100;

% autoregressive coefficient for government spending shock
ar1g = 0.9;

% autoregressive coefficient for z shock
rhoz = 0.8333;

% set of data from which we infer shocks
if par.psid==0
    filterseries = {'ygaphat','rhat','rbhat'}; %{'gydev','gxdev','sxdev','ygap'}

    % set of shocks which we infer from time series
    infershocks = {'g','r','rb'};%{'g','do'};
    
    shocknames = {'Non-Household Demand','Monetary Policy','Borrowing Spread'};
else
    filterseries = {'ygaphat','dsxhat','rhat','rbhat'}; %{'gydev','gxdev','sxdev','ygap'}

    % set of shocks which we infer from time series
    infershocks = {'pygr','g','r','rb'};%{'g','do'};
    
    shocknames = {'Productivity','Non-Household Demand','Monetary Policy','Borrowing Spread'};
end
plotstyle = {'b-','r--','k:','c-.'};

% lower bound for real rate
imin = 0.0015;  % minimum FFR
rbar = -steady.r-0.02658+imin; % mean(nondur4Qinfl(tquarter>=1991 & tquarter<=2007.75))*100
rloc = find(strcmp(filterseries,{'rhat'}));
if isempty(rloc)
    rloc = find(strcmp(filterseries,{'drhat'}));
end

% horizons for which we calculate rstar
hhset = [1,20];
hhmax = max(hhset);


% sticky information model
stinfo=1;





%--------------------------------------------------------------------------
% Step 1: Load data
%--------------------------------------------------------------------------

% Import data from fred
load ../Data/freddownloadfilter;
ymin = 1972;
ymax = 2019;


cd ../IRFs/
% constructing real broad durable spending
[data.qall,data.pall,data.qallnom,data.dlqall,data.dlpall] = deflate(data.qdurnom,data.qresnom,...
                                                                     data.pdurnom,data.presnom);
                                          
% constructing real nondurable spending
[data.qnonall,data.pnonall,data.qnonallnom,data.dlqnonall,data.dlpnonall] = deflate(data.qnondurnom,data.qservnom,...
                                                                                    data.pnondurnom,data.pservnom);


cd ../Model/



% share of nominal expenditure in durables
data.sx = data.qallnom./data.gdpnompot;
data.sxdev = data.sx - mean(data.sx);

% change in share of nominal expenditure in durables
data.dsx = data.sx - lagmatrix(data.sx,1);
data.dsx(1) = 0;
data.dsxhat = data.dsx;

% output gap
data.ygap = (data.gdp-data.gdppot)./data.gdppot;                                                                                
data.ygaphat = data.ygap;

% mortgage spread
data.rb     = data.mortgagerate - data.bondyield10;



%--------------------------------------------------------------------------
% Step 2: Construct time series models
%--------------------------------------------------------------------------
                                                                            
% regression parameters                                                                           
regstart = find(tquarter==1991.0);
regend = find(tquarter==2007.75);

% expected gdp deflator inflation
data.gdpdefl = data.gdpnom./data.gdp;
data.dlp = log(data.gdpdefl) - lagmatrix(log(data.gdpdefl),1);
                                                                                                                                                                
% nondurable inflation over next four quarters
nondur4Qinfl = zeros(length(data.pnonall),1);
nondur4Qlaginfl = zeros(length(data.pnonall),1);
gdp4Qlaginfl = zeros(length(data.pnonall),1);

for kk=1:4
   nondur4Qinfl = nondur4Qinfl + lagmatrix(data.dlpnonall,-kk);
end
for kk=1:4
   nondur4Qlaginfl = nondur4Qlaginfl + lagmatrix(data.dlpnonall,kk);
   gdp4Qlaginfl = gdp4Qlaginfl + lagmatrix(data.dlp,kk);
end

% ex post real rate
data.realrate = data.fedfundsrate - nondur4Qinfl*100;
                                                   
% average nondurable inflation
avgnondur4Qinfl = mean(nondur4Qinfl(regstart:regend));

% ex ante real rate
data.r = max(max(data.fedfundsrate,imin) - avgnondur4Qinfl*100,(rbar+steady.r)*100+0.01);
data.r(isnan(avgnondur4Qinfl*100)) = steady.r*100;

data.rhat = data.r/100-steady.r;
data.dr = data.r-[0;data.r(1:end-1)];
data.drhat = data.dr/100;
data.drhat(isnan(data.drhat)) = 0;
data.drhat(1) = data.rhat(1);

% borrowing spread
data.rbhat = (data.rb - nanmean(data.rb))/100;
data.rbhat(isnan(data.rbhat)) = 0;

% length of time series
tdata = length(data.r);

% term structure rates
data.rlong = NaN(tdata,40);
data.rlong(:,1) = data.r/100;


% time series model for spread and real rate
nlag = 1; hh=1; zz=1;
for var = {'rb'}
    
    Xvar = [ones(size(data.(var{1}))),lagmatrix(data.(var{1}),1:nlag)]; %,[1:1:length(nondur4Qinfl)]'
    bb    = regress(data.(var{1})(regstart:regend,hh),Xvar(regstart:regend,:));
    
    % shocks to time series relationship
    tsshocks.(var{1}) = data.(var{1}) - Xvar*bb ;%+ bb(1);
    
    % set shocks to zero at beginning of sample
    tsshocks.(var{1})(isnan(tsshocks.(var{1})),1) = 0;
    
    % autoregressive coefficient for model IRFs
    ar1.(var{1}) = bb(2:1+nlag);
    
    % save mean
    armean.(var{1}) = bb(1)/(1-sum(ar1.(var{1})));

end

% autoregressive coefficient for other AD
ar1.g = ar1g;

% estimate monetary policy rule    
Xvar = [ones(size(data.r)),lagmatrix(data.r,1:nlag),data.ygap*100,gdp4Qlaginfl*100]; %,[1:1:length(nondur4Qinfl)]'
[bb,tr1,res,tr3,STATS]    = regress(data.r(regstart:regend,1),Xvar(regstart:regend,:));
rpred = Xvar*bb;

mprule.rho = bb(2);
mprule.phiy = bb(3);
mprule.phipi = bb(4);


ar1.r = mprule.rho;

% autocorrelation of residuals
Xvar = [ones(size(res)),lagmatrix(res,1)];
bres = regress(res,Xvar);

ar1.rres = bres(2);




%--------------------------------------------------------------------------
% Step 3: Get IRFs for filtering
%--------------------------------------------------------------------------

% loop over use of M matrix
for nondurM = [1,0]


% set of shocks we consider
allshocks = [infershocks];
allshocksexr = allshocks;
allshocksexr(ismember(allshocks,'r')) = [];

% load jacobian
if stinfo==1
    load(['Results/' model spec 'jacob.mat'],'jacobifge');
    jacobge = jacobifge;
else
    load(['Results/' model spec 'jacob.mat'],'jacobge');
end


% variations on baseline jacobian
nksave='';
if nondurM==1
    if stinfo==1
        jacobifge0 = jacobge;
        load(['Results/' model 'nondurifyjacob.mat'],'jacobifge');
        jacobifge0.r = jacobifge.r;
        jacobge = jacobifge0;
        
    else
        jacobge0 = jacobge;
        load(['Results/' model 'nonduryjacob.mat'],'jacobge');
        jacobge0.r = jacobge.r;
        jacobge = jacobge0;
    end
    nksave = 'nondurM';
end


savestr = [nksave 'SI' num2str(stinfo) ];

% lower triangular part of monetary policy - AD matrix
jacobrADtril = tril(jacobge.r.AD,-1);


% growth in productivity
zval(1) = 1-rhoz;
zval(2) = zval(1)+rhoz*zval(1);
for kk=3:Tgap
    zval(kk) = zval(kk-1)+rhoz*(zval(kk-1)-zval(kk-2));
    zval(kk) = (1+rhoz)*zval(kk-1)-rhoz*zval(kk-2);
end
zvaltemp = 1-zval;

if isfield(jacobge,'py')
    for var = fieldnames(jacobge.py)'
        vartemp = zeros(Tgap,1);
        for kk=1:Tgap
            vartemp   = vartemp  +zvaltemp(kk)*  jacobge.zagg.(var{1})(:,kk);
        end
        jacobge.pygr.(var{1})   =   jacobge.py.(var{1}) + vartemp  ;
    end
end

% change in productivity
for shock=allshocks
   jacobge.(shock{1}).dz = (jacobge.(shock{1}).zagg-[zeros(1,size(jacobge.(shock{1}).zagg,2));jacobge.(shock{1}).zagg(1:end-1,:)]);
end
jacobge.pygr.dz = -zval'+[0;zval(1:end-1)'];
for shock=allshocks
   jacobge.(shock{1}).zlvl = cumsum(jacobge.(shock{1}).dz);
end
steady.dz = 0;
steady.zlvl = steady.zagg;

% construct IRF to persistent shocks based on jacobian
for shock = infershocks
    for irfvars = fieldnames(jacobge.(shock{1}))'
        if strcmp(irfvars{1},'muc') || strcmp(irfvars{1},'yaggsmooth') || strcmp(irfvars{1},'ygap') || strcmp(irfvars{1},'gy') || strcmp(irfvars{1},'gx')
            continue
        end
        irfFilter2.(shock{1}).(irfvars{1}) = steady.(irfvars{1});
        if strcmp(shock{1},'py') || strcmp(shock{1},'pygr')
            irfFilter2.(shock{1}).(irfvars{1}) = irfFilter2.(shock{1}).(irfvars{1}) + 0.01*jacobge.(shock{1}).(irfvars{1})(:,1);
            if strcmp(irfvars{1},'zlvl')==1
               irfFilter2.(shock{1}).(irfvars{1}) = irfFilter2.(shock{1}).(irfvars{1})./irfFilter2.(shock{1}).(irfvars{1})(end); 
            end
        else
            for tt=1:Tgap
                irfFilter2.(shock{1}).(irfvars{1}) = irfFilter2.(shock{1}).(irfvars{1}) + 0.01*ar1.(shock{1})(1)^(tt-1)*jacobge.(shock{1}).(irfvars{1})(:,tt);
            end
        end
        irfFilter.(shock{1}).(irfvars{1}) = irfFilter2.(shock{1}).(irfvars{1});
    end
    
end


% additional variables for filtering
for shock = allshocks 
    
    % output gap
    irfFilter.(shock{1}).ygap = irfFilter.(shock{1}).AD-irfFilter.(shock{1}).ypot;
    
    % impulse response functions for data objects
    irfFilter.(shock{1}).sxdev = ((irfFilter.(shock{1}).x+irfFilter.(shock{1}).fctot).*irfFilter.(shock{1}).p./irfFilter.(shock{1}).ypot) - (steady.x+steady.fctot) ;
    
    irfFilter.(shock{1}).dsx = irfFilter.(shock{1}).sxdev - [0;irfFilter.(shock{1}).sxdev(1:end-1)] ;
    
    
    % collect impact on term structure of r at horizon hh
    for hh = 1:1:hhmax
        for tt=1:Tgap
            irfFilter.(shock{1}).rlong(tt,hh) = sum(irfFilter.(shock{1}).r(tt:min(Tgap,tt+hh-1),1)-steady.r)/hh;
        end
    end
    
    % monetary policy shock
    irfFilter.(shock{1}).rshock = zeros(Tgap,1);
end



%--------------------------------------------------------------------------
% Step 3.5: Alow for endogenous real rate response
%--------------------------------------------------------------------------

% compute IRF of monetary policy shock
% irfFilter.r.rshock(1) = 0.01;%transpose(0.01*ar1.rres.^[0:1:Tgap-1]);
irfFilter.r.rshock = transpose(0.01*ar1.rres.^[0:1:Tgap-1]);

% System of equations:
% (1) r = rho r_-1 +  Rry y + Rrp p + epsr
% (2) y = Ayr r + epsy
% (3) p = Apr r + epsp
% Solve:
% r = ( Itilde - Rrp*Apr )^-1 ( Rry y + epsr + Rrp*epsp )
% y = Ayr *  ( Itilde - Rrp*Apr )^-1 * ( Rry y + epsr + Rrp*epsp ) + epsy

% matrix that maps output gap into inflation
ygaptoinf = zeros(Tgap,Tgap);
for kk=1:Tgap
    discount = par.kappa*par.irfdt/(1+par.rho*par.irfdt)^kk * ones(Tgap+1-kk,1);
    
    ygaptoinf = ygaptoinf + diag(discount,kk-1);
end

% endogenous price response
xshare = 0;(steady.x+steady.fctot);
Rrp = xshare*mprule.phipi*(eye(Tgap) - [zeros(1,Tgap);eye(Tgap-1),zeros(Tgap-1,1)]);

% matrix that handles lag operation and endogenous price response
eyepluslag = eye(Tgap) - mprule.rho * [zeros(1,Tgap);eye(Tgap-1),zeros(Tgap-1,1)] - Rrp * jacobge.r.p;

% matrix to be inverted
matendoygap =  eye(Tgap) - jacobge.r.AD*...
                    (eyepluslag\(mprule.phiy*eye(Tgap) + mprule.phipi*ygaptoinf));

% r = eyetilde\(phipi*M+phiy*I)*ygap
% ygap = A*r + eps
% ygap = A*(eyetilde\(phipi*M+phiy*I))*ygap + eps
% compute IRF with endogenous monetary policy response
for shock = allshocks
 
    % endogenous output response
    if strcmp(shock{1},'r')
        irfFilter.(shock{1}).ygaphat = matendoygap \ (jacobge.r.AD*( eyepluslag \ (irfFilter.(shock{1}).rshock + Rrp*(irfFilter.(shock{1}).p -steady.p ))  ) );
    else
        irfFilter.(shock{1}).ygaphat = matendoygap \ ( irfFilter.(shock{1}).ygap + jacobge.r.AD*( eyepluslag \ (irfFilter.(shock{1}).rshock + Rrp*(irfFilter.(shock{1}).p -steady.p )) ) );
    end
    
    % endogenous real rate response
    irfFilter.(shock{1}).rhat = steady.r + eyepluslag \ ...
                                ( (mprule.phiy*eye(Tgap) + mprule.phipi*ygaptoinf) ...
                                * irfFilter.(shock{1}).ygaphat + irfFilter.(shock{1}).rshock ...
                                + Rrp*(irfFilter.(shock{1}).p -steady.p ) );                             

    % check
    if strcmp(shock{1},'r')==0
        irfFilter.(shock{1}).ygaphattest = irfFilter.(shock{1}).ygap + jacobge.r.AD*(irfFilter.(shock{1}).rhat-steady.r);
    else
        irfFilter.(shock{1}).ygaphattest = jacobge.r.AD*(irfFilter.(shock{1}).rhat-steady.r);
    end
    if sum(abs(irfFilter.(shock{1}).ygaphat-irfFilter.(shock{1}).ygaphattest))>10^-4
        stop here
    end
   
    % endogenous inflation response
    irfFilter.(shock{1}).pihat = ygaptoinf*irfFilter.(shock{1}).ygaphat;
    
    % endogenous durable price response
    irfFilter.(shock{1}).phat = irfFilter.(shock{1}).p + jacobge.r.p*(irfFilter.(shock{1}).rhat-irfFilter.(shock{1}).r);
    
    
    % checking response
    irfFilter.(shock{1}).rhattest(1) = steady.r ...
                                          + mprule.phiy*irfFilter.(shock{1}).ygaphat(1) ...
                                          + mprule.phipi*irfFilter.(shock{1}).pihat(1) ...
                                          + xshare*mprule.phipi*(irfFilter.(shock{1}).phat(1)-steady.p);
    for tt=2:Tgap
        irfFilter.(shock{1}).rhattest(tt,1) = (1-mprule.rho)*steady.r ...
                                          + mprule.rho*irfFilter.(shock{1}).rhattest(tt-1,1) ...
                                          + mprule.phiy*irfFilter.(shock{1}).ygaphat(tt) ...
                                          + mprule.phipi*irfFilter.(shock{1}).pihat(tt) ...
                                          + xshare*mprule.phipi*(irfFilter.(shock{1}).phat(tt)-irfFilter.(shock{1}).phat(tt-1));
    end
    

    if sum(abs(irfFilter.(shock{1}).rhat-irfFilter.(shock{1}).rhattest))>10^-6 && strcmp(shock{1},'r')==0
        stop here
    end

    % endogenous durable response
    irfFilter.(shock{1}).xhat = irfFilter.(shock{1}).x + jacobge.r.x*(irfFilter.(shock{1}).rhat-irfFilter.(shock{1}).r);
    
    % endogenous FC response
    irfFilter.(shock{1}).fctothat = irfFilter.(shock{1}).fctot + jacobge.r.fctot*(irfFilter.(shock{1}).rhat-irfFilter.(shock{1}).r);
    
    % durable spending ratio
    irfFilter.(shock{1}).sxdevhat = ((irfFilter.(shock{1}).xhat+irfFilter.(shock{1}).fctothat).*irfFilter.(shock{1}).phat - (steady.x+steady.fctot)*steady.p)./irfFilter.(shock{1}).zlvl ;

    % change in durable spending ratio
    irfFilter.(shock{1}).dsxhat = irfFilter.(shock{1}).sxdevhat - [0;irfFilter.(shock{1}).sxdevhat(1:end-1)] ;
    
    % borrowing spread
    irfFilter.(shock{1}).rbhat = irfFilter.(shock{1}).rb - steady.rb;
    
    % collect impact on term structure of rstar at horizon hh
    for hh = 1:1:hhmax
        for tt=1:Tgap
            irfFilter.(shock{1}).rlonghat(tt,hh) = sum(irfFilter.(shock{1}).rhat(tt:min(Tgap,tt+hh-1))-steady.r)/hh;
        end
    end
    
    irfFilter.(shock{1}).rhat        = irfFilter.(shock{1}).rhat - steady.r;
    
    irfFilter.(shock{1}).drhat       = irfFilter.(shock{1}).rhat - [0;irfFilter.(shock{1}).rhat(1:end-1)];
    
    
end

%--------------------------------------------------------------------------
% Step 4: Calculate term structure of rstar given our shocks
%--------------------------------------------------------------------------



% compute natural rate forecast for each shock conditional on fixing the
% output gap to zero for T periods
for shock = allshocks
    
    % set of monetary shocks that generate a zero output gap for Tmax periods under a g shock
    % implied natural rate of interest:
    rstar =  - jacobge.r.AD\irfFilter.(shock{1}).ygap;
    
    % contribution to rstar from past interest rate changes
    rstarendomat = zeros(Tgap,Tgap);

    for tt=2:Tgap
        rstarendomat(tt:end,tt) = -jacobge.r.AD(tt:end,tt:end)\(jacobrADtril(tt:end,1:tt-1)*rstar(1:tt-1,1));
    end
    rstarendo = diag(rstarendomat);
    
    % collect impact on term structure of rstar at horizon hh
    for hh = 1:1:hhmax
        for tt=1:Tgap
            irfFilter.(shock{1}).rstar(tt,hh) = sum(rstar(tt:min(Tgap,tt+hh-1),1))/hh;

            irfFilter.(shock{1}).rstarendo(tt,hh) = sum(rstarendo(tt:min(Tgap,tt+hh-1),1))/hh;
        end
    end
    
    % implied output gap in percent: (testing purposes only)
    ygap = (irfFilter.(shock{1}).ygap + jacobge.r.AD*rstar);

end
eval(['irfFilterSI' num2str(stinfo) '=irfFilter;'])
save(['Results/' model spec 'jacob.mat'],['irfFilterSI' num2str(stinfo) ],'-append')



%--------------------------------------------------------------------------
% Step 5: Extract shocks from time series
%--------------------------------------------------------------------------

filterdata = [];
for var = filterseries
    netdata.(var{1}) = data.(var{1});% - predictdata.(var{1})(1:tdata,1);
    
    % collect data in a matrix for matrix manipulation in the next step
    filterdata = [filterdata,netdata.(var{1})];
end

% set of shocks which we infer from time series
nshocks = length(infershocks);

% set of data from which we infer the shocks
nseries = length(filterseries);

% matrix for inverting netted time series
irfcontemp = zeros(nseries,nshocks);
for kk = 1:length(filterseries)
    for mm = 1:nshocks
        irfcontemp(kk,mm) = irfFilter.(infershocks{mm}).(filterseries{kk})(1);
    end
end

% matrix for projecting monetary news
jacobge.r.rhat = jacobge.r.r;
jacobge.r.drhat = jacobge.r.r - [zeros(1,Tgap);jacobge.r.r(1:end-1,:)] ;
jacobge.r.ygaphat = jacobge.r.AD;
jacobge.r.sxdevhat = (jacobge.r.x+jacobge.r.fctot) + (steady.x+steady.fctot).*jacobge.r.p ;
jacobge.r.dsxhat = jacobge.r.sxdevhat - [zeros(1,Tgap);jacobge.r.sxdevhat(1:end-1,:)] ;
jacobge.r.rbhat = zeros(size(jacobge.r.r));

% cashing objects for filtering
filterpredict = zeros(tdata+Tgap,nseries);
filtershocks = zeros(tdata,nshocks);

% store real rate forecasts
results.actacc.rfc = zeros(tdata+Tgap,tdata,hhmax);

% monetary news shocks for ZLB
tsshocks.rzlb = zeros(tdata,Tgap);


% filtering starts here
for tt=1:tdata
    
    % update predictions for length of IRF
    updaterate = 0.25;
    rnews = 1;
    while max(rnews)>10^-6
        % current period data net of predicted effects
        filternetdata = filterdata(tt,:) - filterpredict(tt,:);
        
        % update estimate of shocks
        updateshocks = irfcontemp \ filternetdata';
        
        % implied set of shocks that match netdata
        filtershocks(tt,:) = filtershocks(tt,:) + updateshocks';
        
        for kk = 1:nseries
            for mm = 1:nshocks
                filterpredict(tt:tt-1+Tgap,kk) = filterpredict(tt:tt-1+Tgap,kk) ...
                             + irfFilter.(infershocks{mm}).(filterseries{kk}) * updateshocks(mm,1) ;
            end
        end

       % news shock to r that capture ZLB 
       if strcmp(filterseries{rloc},'rhat')==1
           % forecast for real rate deviation from steady state
           rfc   = filterpredict(tt:tt-1+Tgap,rloc);
       end
        
       rnews = -min(rfc - rbar,0);

       % gradual updating when shocks are large
       if max(rnews)==0
           % store forecast for the actual real rate
           results.actacc.rfc(tt:end,tt,1) = results.actacc.rfc(tt:end,tt,1) + filterpredict(tt:end,rloc);
           results.actacc.rfc(1:tt-1,tt,1) = NaN;
           
           % long-rate forecasts
           for hh=2:hhmax
               for ss=1:Tgap
                    results.actacc.rfc(tt+ss-1,tt,hh) = sum(results.actacc.rfc(tt+ss-1:min(tt+Tgap-1,tt+ss+hh-2),tt,1))/hh;
               end
           end

           
           continue
       elseif max(rnews)>0.0001
          % gradual updating if differences are large
          rnews = updaterate * rnews; 
       end
       
       % storing monetary news shocks 
       tsshocks.rzlb(tt,:) = tsshocks.rzlb(tt,:) + rnews';
       
       % effect of news shocks on predicted series
       filterpredict0 = filterpredict(tt:tt-1+Tgap,:);
       for kk = 1:nseries
            filterpredict(tt:tt-1+Tgap,kk) = filterpredict(tt:tt-1+Tgap,kk) ...
                        + jacobge.r.(filterseries{kk}) * rnews;
       end
       
       
       
    end
    
    % testing code
    testval = sum(abs(filterpredict(tt,:)-filterdata(tt,:)));
    if testval>10^-4
        stop here
    end

end

for mm = 1:nshocks
   tsshocks.(infershocks{mm}) = filtershocks(:,mm);
end

% Figure 8: plot shocks
if nondurM==0
figure
for kk=1:length(infershocks)
     if strcmp(infershocks{kk},'r') || strcmp(infershocks{kk},'rb')
        norm = -1;
    else
        norm = 1;
    end
    plot(tquarter,norm*tsshocks.(infershocks{kk})*irfFilter.(infershocks{kk}).ygaphat(1)*100,plotstyle{kk},'Linewidth',3)
    hold on
end
plot(tquarter,zeros(1,length(tquarter)),'k:','Linewidth',1)
hold off
title('Time Series of Filtered Shocks')
ylabel(['%'])
xlim([2005 2019.75])
ylim([-1.5 1])
yticks([-1.5:0.5:1])
legend(shocknames{:},'Location','SouthEast') 
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
if saveplots==1
export_fig('gcf','-transparent',['../Figures/filteredshocksnorm' savehandle savestr '.' savext])
end
end

%--------------------------------------------------------------------------
% Step 7: Compute time series of full accommodation natural real rate of interest
%--------------------------------------------------------------------------

% compute IRF for output gap following shocks
tstart  = find(tquarter==1991.0);
tendshock = find(tquarter==2012.75); % last period for which we use shocks to forecast
tend    = find(tquarter==2019.75);

% Counterfactuals: what if there had been no accommodation through date t?
noaccdates.fullacc = zeros(tdata,1);
noaccdates.actacc = zeros(tdata,1);
noaccdates.noacc = zeros(tdata,1);
noaccdates.noacc(tquarter>=2007 & tquarter<2013) = 1;
noaccdates.noacc2 = noaccdates.noacc;

% here we calculate the real rate dynamics from initial condition so we get
% the initial level of r right. if we start early enough, then this has no
% implication for our filtering for the great recession as the effect
% tempers out
initr = NaN(tdata+Tgap-1,hhmax);

% evoluation of the real rate based on initial condition
initrtemp = (data.r(tstart)-steady.r*100)*(irfFilter.r.r'-steady.r);

for hh = 1:1:hhmax
    for tt=tstart:tstart+Tgap-1
        initr(tt,hh) = sum(initrtemp(tt-tstart+1:min(Tgap,tt-tstart+hh)))/hh;
    end
    initr(tt+1:end,hh) = 0;
end


% calculate rstar given filtered shocks
for cfcase = {'fullacc'}
    results.(cfcase{1}).rstar = NaN(tdata+Tgap-1,hhmax);
    for shock = allshocksexr
        results.(cfcase{1}).contrib.(shock{1}) = NaN(tdata+Tgap-1,hhmax);
    end
end

for cfcase = {'fullacc'}
    for hh = hhset  % horizons in quarters for which we calculate rstar    
    
        results.(cfcase{1}).rstar(tstart:tend+Tgap-1,hh) = zeros(tend-tstart+Tgap,1);
        results.(cfcase{1}).rstarendo(tstart:tend+Tgap-1,hh) = zeros(tend-tstart+Tgap,1);
        for shock = allshocksexr 
            results.(cfcase{1}).contrib.(shock{1})(tstart:tend+Tgap-1,hh) = zeros(tend-tstart+Tgap,1);
        end
        
        % for the no accommodation case we must keep track of the output
        % gap
        ygap = zeros(tdata+Tgap-1,1); 

        % effect of shocks on path of rstar
        for tt=tstart:tendshock
                        
                
            % News to rstar = A^-1 * C * epsilon in note.
            rstarnews = 0;
            rstarendonews = 0;
            for shock = allshocksexr 
                % contribution of this shock
                rstarshock = tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).rstar(:,hh);

                results.(cfcase{1}).contrib.(shock{1})(tt:tt+Tgap-1,hh) = ...
                                results.(cfcase{1}).contrib.(shock{1})(tt:tt+Tgap-1,hh)...
                              + rstarshock;

                % total contribution
                rstarnews = rstarnews + rstarshock ;

                % contribution to rstar that is coming from past real rate
                % news on output gap
                rstarendoshock = tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).rstarendo(:,hh);
                

                rstarendonews = rstarendonews + rstarendoshock ;
            end

            % Add to existing forecast of rstar
            results.(cfcase{1}).rstar(tt:tt+Tgap-1,hh) = ...
                                results.(cfcase{1}).rstar(tt:tt+Tgap-1,hh)...
                          +     rstarnews ;

            % Add to existing forecast of rstar
            results.(cfcase{1}).rstarendo(tt:tt+Tgap-1,hh) = ...
                                results.(cfcase{1}).rstarendo(tt:tt+Tgap-1,hh)...
                          +     rstarendonews ;
                                              
            
            
        end

        % adding initial condition
        results.(cfcase{1}).rstar(:,hh) = results.(cfcase{1}).rstar(:,hh) + initr(1:tdata+Tgap-1,hh);

    end
    
end




%--------------------------------------------------------------------------
% Step 8: Compute time series for actual accommodation natural real rate of interest
%--------------------------------------------------------------------------

tendshock0 = tendshock;
tendshockset = tendshock0;


for tendshock = tendshockset
    results.actacc.rstar = NaN(tdata+Tgap-1,hhmax);
    results.actacc.rstarfc = NaN(tdata+Tgap-1,tdata+Tgap-1,hhmax);
    results.actacc.rstarendo = NaN(tdata+Tgap-1,hhmax);
    results.actacc.rstarendofc = NaN(tdata+Tgap-1,tdata+Tgap-1,hhmax);
    results.actacc.ygap = NaN(tdata+Tgap,1);
    results.actacc.dsxhat = NaN(tdata+Tgap,1);
    results.actacc.ygapavg = NaN(tdata,1);
    for shock = allshocks
        results.actacc.contribygap.(shock{1}) = NaN(tdata+Tgap,1);
    end


    % What to we want to capture here? That actual use of monetary policy was
    % less aggressive and therefore that we borrowed less from the future. 
    for cfcase = {'actacc'} 

        for hh=hhset
            results.(cfcase{1}).rstar(tstart:tend+Tgap-1,hh) = 0;
            results.(cfcase{1}).rstarendo(tstart:tend+Tgap-1,hh) = 0;
            for pp=1:tdata+Tgap-1
                results.(cfcase{1}).rstarfc(tstart:tend+Tgap-1,pp,hh) = results.(cfcase{1}).rfc(tstart:tend+Tgap-1,tstart-1,hh);%repmat(initr(tstart:tend+Tgap-1,hh),[1,tdata+Tgap-1]);
            end
            results.(cfcase{1}).rstarendofc(tstart:tend+Tgap-1,:,hh) = 0;
            results.(cfcase{1}).rendo(tstart:tend+Tgap-1,hh) = 0;
        end
        results.(cfcase{1}).rmisses(tstart:tend+Tgap,tstart:tend+Tgap,1) = 0;
        results.(cfcase{1}).ygap(tstart:tend+Tgap,1) = 0;
        results.(cfcase{1}).dsxhat(tstart:tend+Tgap,1) = 0;
        results.(cfcase{1}).sxdevhat(tstart:tend+Tgap,1) = 0;
        results.(cfcase{1}).rbhat(tstart:tend+Tgap,1) = 0;
        missedaccommodation = zeros(Tgap,1);
        for shock = allshocks
            results.(cfcase{1}).contribygap.(shock{1})(tstart:tend+Tgap-1,1) = zeros(tend-tstart+Tgap,1);
        end

        for tt=tstart:tend%min(tendshock,tend) % min(tendshock+1,tend)

            % Output gap news
            rstarnews = zeros(Tgap,1);
            ygapnews = zeros(Tgap,1); 
            ygaphatnews = zeros(Tgap,1); 
            dsxhatnews = zeros(Tgap,1); 
            rbhatnews = zeros(Tgap,1); 
            sxdevhatnews  = zeros(Tgap,1); 
            rnews = zeros(Tgap,1);
            if tt<=tendshock
                for shock = allshocksexr

                    % output gap path absent monetary policy response
                    ygapnews = ygapnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).ygap;
                    
                end
                
                % output gap path with monetary policy
                for shock = allshocks
                    ygaphatnews = ygaphatnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).ygaphat;
                    
                    results.(cfcase{1}).contribygap.(shock{1})(tt:tt+Tgap-1,1) = ...
                                results.(cfcase{1}).contribygap.(shock{1})(tt:tt+Tgap-1,1)...
                              + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).ygap;
                    results.(cfcase{1}).contribygap.r(tt:tt+Tgap-1,1) = ...
                                results.(cfcase{1}).contribygap.r(tt:tt+Tgap-1,1)...
                              + tsshocks.(shock{1})(tt)*(irfFilter.(shock{1}).ygaphat-irfFilter.(shock{1}).ygap);
                          
                    rbhatnews = rbhatnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).rbhat;
                    dsxhatnews = dsxhatnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).dsxhat;
                    sxdevhatnews = sxdevhatnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).sxdevhat;

                    rnews = rnews + tsshocks.(shock{1})(tt)*irfFilter.(shock{1}).rhat;
                end
                
                % add ZLB news shocks
                ygaphatnews = ygaphatnews + jacobge.r.AD*tsshocks.rzlb(tt,:)';
                
                dsxhatnews = dsxhatnews + jacobge.r.dsxhat*tsshocks.rzlb(tt,:)';
                sxdevhatnews = sxdevhatnews + jacobge.r.sxdevhat*tsshocks.rzlb(tt,:)';
                rbhatnews = rbhatnews + jacobge.r.rbhat*tsshocks.rzlb(tt,:)';

                rnews = rnews + tsshocks.rzlb(tt,:)';
                
                
                results.(cfcase{1}).contribygap.r(tt:tt+Tgap-1,1) = ...
                                results.(cfcase{1}).contribygap.r(tt:tt+Tgap-1,1)...
                              + jacobge.r.AD*tsshocks.rzlb(tt,:)';

            end
            
            % r-star news for closing future output gaps
            for ss=0:1:Tgap-1
                % s = time until we implement r-star policy
                
                % monetary news today
                rstarnewsfc = zeros(Tgap,1);
                
                rstarnewsfc(1:ss,1) = rnews(1:ss);
                
                rstarnewsfc(ss+1:end,1) = - jacobge.r.AD(1+ss:end,1+ss:end) \  ...
                      ( ygapnews(1+ss:end) + jacobge.r.AD(1+ss:end,1:ss)*rstarnewsfc(1:ss) );
                
                % contribution to path from past interest rates
                rstarendomat = zeros(Tgap,Tgap);  
                for zz=1+ss:Tgap
                    % only need this calculation for time ss onwards
                    rstarendomat(zz:end,zz) = -jacobge.r.AD(zz:end,zz:end)\(jacobrADtril(zz:end,1:zz-1)*rstarnewsfc(1:zz-1,1));
                end
                rstarendonewsfc = diag(rstarendomat);  
                  
                % aggregate to hh horizon
                rstarnewsfclong = NaN(Tgap,1,hhmax);
                rstarendonewsfclong = NaN(Tgap,1,hhmax);
                for hh = hhset  
                    for zz=1:Tgap
                        rstarnewsfclong(zz,1,hh) = sum(rstarnewsfc(zz:min(zz+hh-1,Tgap),1))/hh;
                        rstarendonewsfclong(zz,1,hh) = sum(rstarendonewsfc(zz:min(zz+hh-1,Tgap),1))/hh;
                    end
                end
                  
                % add to current forecast of output gap
                % first column is time horizon, second column is date at
                % which r-star policy is implemented, third column is
                % horizon                          
                results.(cfcase{1}).rstarfc(tt:tt+Tgap-1,tt+ss,:) = ...
                                    results.(cfcase{1}).rstarfc(tt:tt+Tgap-1,tt+ss,:)...       
                              +     rstarnewsfclong;
                          
                results.(cfcase{1}).rstarendofc(tt:tt+Tgap-1,tt+ss,:) = ...
                                    results.(cfcase{1}).rstarendofc(tt:tt+Tgap-1,tt+ss,:)...       
                              +     rstarendonewsfclong;
                    
                
            end
            
            % contemporaneous value of r-star
            results.(cfcase{1}).rstar(tt:tt+Tgap-1,:) = ...
                           results.(cfcase{1}).rstarfc(tt:tt+Tgap-1,tt,:);
            
          
            % past interest rate contribution
            results.(cfcase{1}).rstarendo(tt:tt+Tgap-1,:) = ...
                           results.(cfcase{1}).rstarendofc(tt:tt+Tgap-1,tt,:);           
                       
            % contemporaneous value of r
            results.(cfcase{1}).r(tt:tt+Tgap-1,:) = ...
                           results.(cfcase{1}).rfc(tt:tt+Tgap-1,min(tt,tendshock),:);           
            

            % output gap forecast
            results.(cfcase{1}).ygap(tt:tt+Tgap-1,1) = ...
                            results.(cfcase{1}).ygap(tt:tt+Tgap-1,1) ...
                        +   ygaphatnews; 
                    
            % output gap forecast
            results.(cfcase{1}).dsxhat(tt:tt+Tgap-1,1) = ...
                            results.(cfcase{1}).dsxhat(tt:tt+Tgap-1,1) ...
                        +   dsxhatnews; 
                    
             % output gap forecast
            results.(cfcase{1}).sxdevhat(tt:tt+Tgap-1,1) = ...
                            results.(cfcase{1}).sxdevhat(tt:tt+Tgap-1,1) ...
                        +   sxdevhatnews; 
                    
             % output gap forecast
            results.(cfcase{1}).rbhat(tt:tt+Tgap-1,1) = ...
                            results.(cfcase{1}).rbhat(tt:tt+Tgap-1,1) ...
                        +   rbhatnews; 

                     
            % here we find the smallest output gap forecasted
            results.(cfcase{1}).ygapavg(tt) = sqrt(mean((results.(cfcase{1}).ygap(tt:tt+hhmax/2-1)).^2));

        end
        


        % find quarter with smallest forecasted output gap before 2007 and set r = r^*
        % under actual accommodation
        ygapmin = find(results.actacc.ygapavg(tquarter>=2003.0 & tquarter<=2007.5)==min(results.actacc.ygapavg(tquarter>=2003.0 & tquarter<=2007.5)));
        ygapmin = ygapmin + find(tquarter==2003)-1;
        if nondurM==0
        
        % Figure 10: r-star and intertemporal shifting contribution
        for hh=hhset
            
            % rstar figure
            figure
            bl = plot(tquarter(1:tendshock),steady.r*100 + results.actacc.rstar(1:tendshock,hh)*100,'b-','Linewidth',3);
            hold on
            
            hpstart = find(tquarter==2005);
            xlen = tquarter(hpstart:tend);
            irfcil = steady.r*100 + results.actacc.rstarendo(hpstart:tend,hh)*100;
            irfciu = steady.r*100 + zeros(tend-hpstart+1,1);
            hp1 = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'m');
            set(hp1, 'facecolor', [0.75 0.75 0.75], 'edgecolor', 'none');
%             set(hp1,'facealpha',0.4)

            txt1 = text(2009.15,0.5,'Intertemporal Shifting Contribution','FontSize',14);
            
            
            
%             bl3 = plot(tquarter(1:tendshock),steady.r*100 + results.actacc.rstarendo(1:tendshock,hh)*100,'k-.','Linewidth',3);
            plot(tquarter(1:tend),steady.r*100 +zeros(tend,1),'k:','Linewidth',2)
            if hh==1
                ylabel(['Short-term r^*, percentage points'])
            else
                ylabel([num2str(hh/4) '-year r^*, percentage points'])
            end
            xlim([2005 tquarter(tend)])
            if tendshock<tend
                yl = ylim;
                
                newcol = [0.4940, 0.1840, 0.5560];%[1,0,0];%
                fc = plot(tquarter(tendshock:tend),steady.r*100 + results.actacc.rstar(tendshock:tend,hh)*100,'b-','Linewidth',3);
%                 fc3 = plot(tquarter(tendshock:tend),steady.r*100 + results.actacc.rstarendo(tendshock:tend,hh)*100,'k-.','Linewidth',3);

                
                fc4 = plot(mean([tquarter(tendshock),tquarter(tendshock+1)])*ones(1,2),yl,'Color',newcol,'Linewidth',2);
      
            end
            
            
            
            
            uistack(bl,'top')
            legendset = {'r^*','Intertemporal Shifting Contribution'};
            legend([bl,hp1],legendset)
            
            set(gcf,'color','w');
            set(gca,'FontSize',16,'fontWeight','normal')
            set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
            if tendshock<tend
                dist = -yl(1)+yl(2);
                txt = {['Forecasts for r^*'];['as of 2012Q4'];'\rightarrow'};

            end
            
            if saveplots
                saveas(gcf,['../Figures/rstar'  num2str(hh) savestr num2str(tquarter(tendshock)*100)  '.' savext])
            end
            

            
            

        end
        end
    end
    
    
end


save(['Results/' savehandle 'rstar' savestr '.mat'],'results')

end

%--------------------------------------------------------------------------
% Figure 9: 
%--------------------------------------------------------------------------

load(['Results/' savehandle 'rstarnondurMSI1.mat'],'results')
resultsnondur = results;

load(['Results/' savehandle 'rstarSI1.mat'],'results')

for hh=hhset(1)

    figure
    plot(tquarter(1:tend),data.rlong(1:tend,hh)*100,'k-','Linewidth',3)
    hold on

    fc0 = plot(tquarter(1:tendshock),steady.r*100 + results.actacc.r(1:tendshock,hh)*100,'r-.','Linewidth',3);
    fcNK0 = plot(tquarter(1:tendshock),steady.r*100 + resultsnondur.actacc.r(1:tendshock,hh)*100,'c-.','Linewidth',3);
    plot(tquarter(1:tend),data.rlong(1:tend,hh)*100,'k-','Linewidth',3)
    hold on
    plot(tquarter(1:tend),steady.r*100 + zeros(tend,1),'k:','Linewidth',2)
    if hh==1
        ylabel(['Short-term Real Rate, Percentage Points'])
    else
        ylabel([num2str(hh/4) '-year Real Rate, Percentage Points'])
    end
    xlim([2005 tquarter(tend)])
   

    if tendshock<tend
        yl = ylim;

        newcol = [1,0,0];
        fc = plot(tquarter(tendshock:tend),steady.r*100 + results.actacc.r(tendshock:tend,hh)*100,'r-.','Linewidth',3);

        fcNK = plot(tquarter(tendshock:tend),steady.r*100 + resultsnondur.actacc.r(tendshock:tend,hh)*100,'c-.','Linewidth',3);

        fcvert = plot((tquarter(tendshock)+tquarter(tendshock+1))/2*ones(1,2),yl,'Color',newcol,'Linewidth',2);
    end
    legtxt = 'r';
    if hh>1
        legendset = {['Data (TIPS)'],['Full Model'],['Nondurables M-Matrix']};
    else
        legendset = {['Data'],['Full Model'],['Nondurables M-Matrix']};
    end
    legend(legendset)
    hold off
    set(gcf,'color','w');
    set(gca,'FontSize',16,'fontWeight','normal')
    set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')

    if saveplots
        export_fig('gcf','-transparent',['../Figures/rdatapres' num2str(hh) savestr  num2str(tquarter(tendshock)*100) '.' savext])
    end

end

%--------------------------------------------------------------------------
% Figure 11: Contribution of shocks to r*
%--------------------------------------------------------------------------

contribshocks = infershocks;
contribshocks(strcmp(infershocks,'r')) = [];
contribshocknames = shocknames;
contribshocknames(strcmp(infershocks,'r')) = [];

lineset.pygr = 'r--';
lineset.g = 'c-.';
lineset.rb = 'k:';

for hh=hhset(1)

    % actual accommodation decomposition
    figure
    plot(tquarter(1:tend),steady.r*100 + results.actacc.rstar(1:tend,hh)*100,'b-','Linewidth',3)
    hold on
    for shock = contribshocks
        plot(tquarter(1:tend),steady.r*100 + results.fullacc.contrib.(shock{1})(1:tend,hh)*100,lineset.(shock{1}),'Linewidth',3) 
    end
    plot(tquarter(1:tend),steady.r*100 + (results.actacc.rstar(1:tend,hh)-results.fullacc.rstar(1:tend,hh))*100,'m-.','Linewidth',3) 
    
    plot(tquarter(1:tend),steady.r*100 + zeros(tend,1),'k:')
    
    
    if hh==1 && stinfo==1 && strcmp(spec,'fcoctheta')
        ylim([-9 6])
    end
    yl = ylim;
    fcvert = plot((tquarter(tendshock)+tquarter(tendshock+1))/2*ones(1,2),yl,'Color',newcol,'Linewidth',2);
    hold off

    xlim([2005 tquarter(tend)])
    if hh==1
        ylabel(['Short-term r^*, percentage points'])
    else
        ylabel([num2str(hh/4) '-year r^*, percentage points'])
    end
    legend(['r^*',contribshocknames,'Monetary Shock / ELB'],'Location','SouthEast')
    set(gcf,'color','w');
        set(gca,'FontSize',16,'fontWeight','normal')
        set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
        
    if saveplots
        export_fig('gcf','-transparent',['../Figures/rstarcontrib' num2str(hh) savestr  num2str(tquarter(tendshock)*100)  '.' savext])
    end

end



%--------------------------------------------------------------------------
% Appendix Figure: IRFs for all shocks
%--------------------------------------------------------------------------

plotmax = 8;
steady.rhat = 0;
steady.ygaphat = 0;
steady.dsxhat = 0;
steady.rbhat = par.rb;

lineset.ygaphat = 'b-';
lineset.dsxhat = 'r--';
lineset.rhat = 'c-.';
lineset.rbhat = 'k:';

for xvar = infershocks
    figure
    for yvar = filterseries
        plot(0:1:plotmax,(irfFilter.(xvar{1}).(yvar{1})(1:plotmax+1)-irfFilter.(xvar{1}).(yvar{1})(end))*100,lineset.(yvar{1}),'Linewidth',3)
        hold on
    end
    plot(0:1:plotmax,(irfFilter.(xvar{1}).r(1:plotmax+1,1) - irfFilter.(xvar{1}).r(end,1) + irfFilter.(xvar{1}).rstar(1:plotmax+1,1))*100,'m-','Linewidth',3)
    plot(0:1:plotmax,zeros(1,plotmax+1),'k:','Linewidth',1)
    hold off
    leg = legend({'$\hat{Y}$','$\Delta s^x$','r','$r^b$','$r^*$'});
    set(leg,'Interpreter','latex');
    ylabel('% deviation from steady state')
    xlabel('Quarter')
    set(gcf,'color','w');
    set(gca,'FontSize',16,'fontWeight','normal')
    set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
    if saveplots
        export_fig('gcf','-transparent',['../Figures/irfs' (xvar{1}) savestr '.' savext])
    end
end





