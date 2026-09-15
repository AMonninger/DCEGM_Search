function par = setparametershouse(spec)


%% Specific Model Parameters
if strcmp(spec,'base') || strcmp(spec,'nod')
    par.rho     = 0.015;                   % discount rate
    par.psid    = 0.15;                     % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = -log((1-.0058)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0;                        % cost of adjustment
elseif strcmp(spec,'fc')
    par.rho     = 0.078987944102781;                    % discount rate
    par.psid    = 0.291227083390619;        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = -log((1-.0058)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0.003226369197459;        % cost of adjustment
elseif strcmp(spec,'fctheta')
    par.rho     = 0.085259181918171;        % discount rate
    par.psid    = 0.327189043964579;        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = -log((1-.0058)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0.12;                     % exogenous adjustment probability
    par.fc      = 0.037807589945457;                     % cost of adjustment
    par.pdac    = 0.049;
elseif strcmp(spec,'fcoctheta')
    par.rho     = 0.084183098070221;        % discount rate
    par.psid    = 0.402601499138386;                     % weight durable
    par.chi     = 0.515;                     % maintenance cost
    par.delta   = -log((1-.0058)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0.041;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0.118;                                 % exogenous adjustment probability
    par.fc      = 0.038807943817199;                     % cost of adjustment
elseif strcmp(spec,'fcoc')
    par.rho     = 0.080037836467056;                      % discount rate
    par.psid    = 0.384597269312613;                      % weight durable
    par.chi     = 0.515;                     % maintenance cost
    par.delta   = -log((1-.0058)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0.041;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0;                                    % exogenous adjustment probability
    par.fc      = 0.004829861486318;                     % cost of adjustment
end


   
%% General Model Parameters
par.sigma   = 4;         % curvature: nondurable
par.xii     = 0.5;1;
par.r       = 0.015;     % real return on asset
par.rb      = 0.017;     % borrowing spread
par.p       = 1;        % relative durable price
par.yagg    = 1;        % aggregate income
par.zagg    = 1;
par.la      = 0;
par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized
par.lambda  = par.lambdatemp*(1-par.fc);
par.yrho    = 0.9136;                                               % income: log AR(1) coeff (annual)
par.yvar    = 0.0426*(2*-log(par.yrho)/(1-exp(2*log(par.yrho))));    % income: log AR(1) variance (annual)
par.g       = 0.2;        % SS govt spending
par.kappa   = 0.8;      % slope of PC
par.mprpi   = 0.5;      % MP reaction function: inflation
par.mpry    = 0;        % MP reaction function: output
par.stinf   = 1/6;      % fraction updating information

% order of variables
par.varsetall  = {'z','d','y','x','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc','dtries'};
par.varsetz  = {'z','d','y','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc'};
par.varsetx  = {'x','y','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc'};

%% targets and parameters
par.targets.a    = 0.92;         % asset to income ratio
par.targets.dc   = 1.9296;      % durable stock to nondurable cons ratio
par.targets.probadjy = 0.15;    % annual adj probability

par.targetpar.rho   = par.rho;    % discount rate
par.targetpar.psid  = par.psid;   % durable weight
par.targetpar.fc    = par.fc;     % fixed cost

%% IRF parameters
par.hazdt = 1/120;   % hazard function step size
par.hazT = 1/par.hazdt;   % final year for hazard function / probabilities
par.hazTlong = 10/par.hazdt;   % final year for hazard function / probabilities

par.irfdt = 1/4;           % IRF step size
par.irfagg= 1/4;            % aggregate to this frequency
par.irfT  = 5/par.irfdt;      % length of IRF 

par.irfvars = {'yagg','yaggnew','a','d','c','x','adj','xadj','AD','r','p','g','rb','zagg','fctot','octot','spreadtot','depr','muc'}; % variables to be stored

%% Numerical Paramters
par.Nz      = 125;200;75*1.0;300;75;100;75;80;      % grid length: asset - lambda*durable
par.Nd      = 60;120;80*1.0;100;2*40;3*40;      % grid length: durable
par.Nx      = 150;250;150;150;      % grid length: durable
par.Ny      = 5;15;      % grid length: income
par.Nexo    = 1;       % grid length: exogenous path
par.Nr      = 1;       % grid length: interest rate
par.Nrb     = 1;       % grid length: interest rate
par.Np      = 1;       % grid length: prices
par.Nyagg   = 1;       % grid length: agg income
par.Nzagg   = 1;       % grid length: agg productivity
par.Ntax    = 1;       % grid length: transfer
par.Ntxpr   = 1;       % grid length: tax prop
par.Ng      = 1;       % grid length: govt
par.Ncfc    = 1;       % grid length: cfc
par.Ndtries = 1000;      % for global search of optimal d



% curvature of nodes
par.phiz    = 1.1;1.0;
par.phid    = 1.05;1.0;
par.phix    = 1.1;1.0;
par.phiy    = 1.4;
par.phiexo  = 1.0;
par.phir    = 1.0;
par.phirb   = 1.0;
par.phip    = 1.0;
par.phiyagg = 1.0;
par.phizagg = 1.0;
par.phitax  = 1.0;
par.phitxpr = 1.0;
par.phig    = 1.0;
par.phicfc  = 1.0;
par.phidtries = 1.0;

% find mean stochstic income, such that aggregate income = yavg
if par.Ny>1
    if par.la==0
        yrel = 2;   % multiples of income
        [par.ymean,par.yvariance,par.yvarianceinf] = fymean(par.yagg,par.yagg,1/yrel,yrel,par.Ny,par.phiy,par.yrho,par.yvar);
    else
        yrel = 1.375;
        par.ymean = 1;
    end
else
    yrel = 1;
    par.ymean = 1; 
end

% boundaries of state space
par.zmax	= 15;                                   % grid max: asset - lambda*durable
par.dmax	= 4;                                    % grid max: durable
par.xmax    = par.zmax+(1-par.fc)*par.dmax;         % grid max: COH
par.ymax    = par.ymean*yrel;                           % grid max: income
par.exomax  = 0;                                    % grid max: exo path
par.rmax    = par.r;                                % grid max: interest rate
par.rbmax   = par.rb;                                % grid max: interest rate
par.pmax    = par.p;                                 % grid max: rel durable price
par.yaggmax = par.yagg;                                % grid max: agg income
par.zaggmax = par.zagg;                                % grid max: agg income
par.taxmax   = 0;                                    % grid max: transfer
par.txprmax = 1;                                    % grid max: tax proportional
par.gmax    = par.g;                                % grid max: transfer
par.cfcmax   = 0;                                    % grid max: cfc
par.zmin	= 0;                                    % grid min: asset - lambda*durable
par.dmin	= 0.05;                              % grid min: durable
par.xmin    = par.zmin + (1-par.lambda)*1.05*par.dmin ;  % grid min: COH
par.ymin    = par.ymean/yrel;                        % grid min: income
par.exomin  = 0;                                    % grid min: exo path
par.rmin    = par.r;                                % grid min: interest rate
par.rbmin   = par.rb;                                % grid min: interest rate
par.pmin    = par.p;                                 % grid min: rel durable price
par.yaggmin = par.yagg;                                % grid min: agg income
par.zaggmin = par.zagg;                                % grid min: agg income
par.taxmin   = 0;                                    % grid min: transfer
par.txprmin = 1;                                    % grid max: tax proportional
par.gmin    = par.g;                                     % grid min: transfer
par.cfcmin   = 0;                                    % grid min: cfc

par.dtriesmin = par.dmin;
par.dtriesmax = par.dmax;


% if par.ymin*par.yagg < (par.nu + par.lambda*(par.r+par.rspread+par.delta)).*par.pmax*par.dmax
%     warning('Code not set up to allow for future violation of borrowing constraint')
%     par.ymin*par.yagg - (par.nu + par.lambda*(par.r+par.rspread+par.delta)).*par.pmax*par.dmax
%     stop here
% end

% Additional nodes
par.zadd = [0,0.01]';[0,0.01,0.025,0.05]';[0,0.01]';[0,0.005,0.01,0.025,0.05,0.1]';[0,0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.2]';[0,0.001,0.005,0.01:0.01:0.1]';[0,0.01]';
par.dadd = 0;[0,0.005,0.01,0.025,0.05]';[0,0.001,0.005,0.01]';
par.xadd = 0;[0,0.01,0.025,0.05,0.1]';[0,0.01,0.05,0.1,0.2]';[0,0.05,0.1]';[0,0.025,0.05,0.1,0.15,0.2,0.25]';[0,0.025:0.025:0.1]';[0,0.05]';%[0,0.001,0.0025,0.005,0.01,0.05,0.1]';
par.yadd = 0;
par.exoadd = 0;
par.radd = 0;
par.rbadd = 0;
par.padd = 0;
par.yaggadd = 0;
par.zaggadd = 0;
par.taxadd = 0;
par.txpradd = 0;
par.gadd = 0;
par.cfcadd = 0;
par.dtriesadd = 0;

% expansion parameters for more detailed grid
par.Nzexp = par.Nz*2.0;
par.Ndexp = par.Nd*2.0;
par.dtexp = 1/200;

% Iteration set-up:
par.Delta     = 100000; % step size (for time) in the implicit method
par.maxit     = 200;
par.maxitpar  = 50;
par.maxitLCP  = 50;100;
par.tol       = 1e-7;
par.tolirf    = 1e-5;
par.iter      = 0;
par.eps       = 10^-10;
par.newton    = 5000;
par.minu      = -10^4;
par.cmin      = 10^-8;
par.steadyinit= 1;

par.interp = 'linear';

par.AssumeConcavity = 0;


end