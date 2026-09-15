function par = setparametersall(spec)

      
%% Specific Model Parameters
if strcmp(spec,'base') || strcmp(spec,'nod')
    par.sigma   = 4;         % curvature: nondurable
    par.rho     = 0.015;                   % discount rate
    par.psid    = 0.15;                     % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);    % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0;                        % cost of adjustment
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized    
elseif strcmp(spec,'fc')
    par.sigma   = 4;         % curvature: nondurable
    par.rho     = 0.074956948650480;                    % discount rate
    par.psid    = 0.450949476811724;        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);  % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0.016332260365444;        % cost of adjustment
    par.pdac    = 0.049;                   % durable price adjustment cost
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized    
elseif strcmp(spec,'fctheta')
    par.sigma   = 4;                 % curvature: nondurable
    par.rho     = 0.096339133735174;                 % discount rate
    par.psid    = 0.515;                 % weight durable
    par.chi     = 0;                     % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);  % depreciation post maintenance
    par.nu      = 0;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0.1575;                     % exogenous adjustment probability
    par.fc      = 0.225;            % cost of adjustment
    par.pdac    = 0.049;                     % durable price adjustment cost
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized 
elseif strcmp(spec,'fcoctheta')
    par.sigma   = 4;         % curvature: nondurable
    par.rho     = 0.095635391649500;            % discount rate
    par.psid    = 0.580723515971232;        % weight durable
    par.chi     = 0.35;                     % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);  % depreciation post maintenance
    par.nu      = 0.048;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0.1575;                     % exogenous adjustment probability
    par.fc      = 0.194430854033669;            % cost of adjustment
    par.pdac    = 0.049;                     % durable price adjustment cost
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized   
elseif strcmp(spec,'highmpc')
    par.sigma   = 4;         % curvature: nondurable
    par.rho     = 0.23;0.185;0.125;0.11;            % discount rate
    par.psid    = 0.685;0.655;0.595;        % weight durable
    par.chi     = 0.35;                     % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);  % depreciation post maintenance
    par.nu      = 0.048;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0.1575;                     % exogenous adjustment probability
    par.fc      = 0.45;0.4;0.25;            % cost of adjustment
    par.pdac    = 0.049;                     % durable price adjustment cost
    par.lambdatemp = 0;0.8;      % fraction of post-sale durable that can be collateralized 
elseif strcmp(spec,'fcoc')
    par.sigma   = 4;         % curvature: nondurable
    par.rho     = 0.076213985517715;                    % discount rate
    par.psid    = 0.536979021405425;                      % weight durable
    par.chi     = 0.35;                     % maintenance cost
    par.delta   = -log((1-.017)^4)*(1-par.chi);  % depreciation post maintenance
    par.nu      = 0.048;%+par.chi/(1-par.chi)*par.delta;   % operating cost
    par.theta   = 0;                     % exogenous adjustment probability
    par.fc      = 0.010198996926357;                     % cost of adjustment  
    par.pdac    = 0.049; 
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized    
elseif strcmp(spec,'nondur')
    par.sigma   = 0.33;0.335;                    % curvature: nondurable
    par.rho     = 0.020157701855274;0.021066784272554;        % discount rate
    par.psid    = 0;                        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = 0;                        % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0;                        % cost of adjustment    
    par.pdac    = 0;
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized  
elseif strcmp(spec,'nondurif')
    par.sigma   = 0.154;                     % curvature: nondurable
    par.rho     = 0.017800312201740;        % discount rate
    par.psid    = 0;                        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = 0;                        % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0;                        % cost of adjustment    
    par.pdac    = 0;
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized    
elseif strcmp(spec,'nondurify')
    par.sigma   = 0.8;1;0.5;0.66;1/0.75;                   % curvature: nondurable
    par.rho     = 0.025934712740200;0.028638251402823;0.022201985977830;0;0.033577330908247;        % discount rate
    par.psid    = 0;                        % weight durable
    par.chi     = 0;                        % maintenance cost
    par.delta   = 0;                        % depreciation post maintenance
    par.nu      = 0;                        % operating cost
    par.theta   = 0;                        % exogenous adjustment probability
    par.fc      = 0;                        % cost of adjustment    
    par.pdac    = 0;
    par.lambdatemp = 0.8;      % fraction of post-sale durable that can be collateralized        
end

% specific nondurable parameters
if par.psid>0
   dmax = 4;
   dmin = 0.05;
   Nd = 60;
   Nz = 125;
   Ny = 5;
   Nexpand = 4;
   phid = 1.05;
   zadd = [0,0.01]';
else
   dmax = 1.25;2.6388;1;
   dmin = 1.25;2.6388;1;
   Nd   = 1;
   phid = 1;
   Nz   = 250;
   Ny = 5;7;
   Nexpand = 1;
   zadd = 0;
end

   
%% General Model Parameters
par.xii     = 0.5;1;
par.r       = 0.015;     % real return on asset
par.rb      = 0.017;     % borrowing spread 
par.p       = 1;        % relative durable price
par.zagg    = 1;
par.yagg    = 1;        % aggregate income
par.la      = 0;

par.lambda  = par.lambdatemp*(1-par.fc);
par.yrho    = 0.9136;                                               % income: log AR(1) coeff (annual)
par.yvar    = 0.0426*(2*-log(par.yrho)/(1-exp(2*log(par.yrho))));    % income: log AR(1) variance (annual)
par.g       = 0.2;        % SS govt spending
par.kappa   = 0.03*16;       % slope of PC
par.etta    = 2;
par.lswealth= 0;
par.mprpi   = 0.5;      % MP reaction function: inflation
par.mpry    = 0;        % MP reaction function: output
par.stinf   = 1/6;      % fraction updating information

% order of variabl es
par.varsetall  = {'z','d','y','x','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc','dtries'};
par.varsetz  = {'z','d','y','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc'};
par.varsetx  = {'x','y','exo','r','p','yagg','zagg','txpr','tax','g','rb','cfc'};

%% targets and parameters
par.targets.a    = .873679;
par.targets.dc   = 2.6388;      % durable stock to nondurable cons ratio
par.targets.probadjy = 0.19;    % quarterly adj probability

par.targetpar.rho   = par.rho;    % discount rate
par.targetpar.psid  = par.psid;   % durable weight
par.targetpar.fc    = par.fc;     % fixed cost

%% IRF parameters
par.hazdt = 1/120;   % hazard function step size
par.hazT = 1/par.hazdt;   % final year for hazard function / probabilities
par.hazTlong = 1/par.hazdt;

par.irfdt = 1/4;           % IRF step size
par.irfagg= 1/4;            % aggregate to this frequency
par.irfT  = 5/par.irfdt;      % length of IRF 

par.irfvars = {'yagg','yaggnew','a','d','c','x','adj','xadj','AD','r','p','g','rb','zagg','fctot','octot','spreadtot','depr','muc'}; % variables to be stored

%% Numerical Paramters
par.Nz      = Nz;75*1.0;300;75;100;75;80;      % grid length: asset - lambda*durable
par.Nd      = Nd;80*1.0;100;2*40;3*40;      % grid length: durable
par.Nx      = 150;250;150;150;      % grid length: durable
par.Ny      = Ny;      % grid length: income
par.Nexo    = 1;       % grid length: exogenous path
par.Nr      = 1;       % grid length: interest rate
par.Nrb     = 1;       % grid length: interest rate spread
par.Np      = 1;       % grid length: prices
par.Nyagg   = 1;       % grid length: agg income
par.Nzagg   = 1;       % grid length: agg productivity
par.Ntax    = 1;       % grid length: transfer
par.Ntxpr   = 1;       % grid length: tax proportional to productivity
par.Ng      = 1;       % grid length: govt
par.Ncfc    = 1;       % grid length: cfc

par.Ndtries = 1000;      % for global search of optimal d



% curvature of nodes
par.phiz    = 1.1;1.1;1.0;
par.phid    = phid;1.0;
par.phix    = 1.1;1.0;
par.phiy    = 1.4;1.0;
% disp('update')
par.phiexo  = 1.0;
par.phir    = 1.0;
par.phirb   = 1.0;
par.phip    = 1.0;
par.phiyagg = 1.0;
par.phizagg = 1.0;
par.phitxpr = 1.0;
par.phitax  = 1.0;
par.phig    = 1.0;
par.phicfc  = 1.0;
par.phidtries = 1.0;

% find mean stochstic income, such that aggregate income = yavg
if par.Ny>1
    if par.la==0
        par.yrel = 2;   % multiples of income
%         disp('update')
        [par.ymean,par.yvariance,par.yvarianceinf,A,par.ggt] = fymean(par.yagg,par.yagg,1/par.yrel,par.yrel,par.Ny,par.phiy,par.yrho,par.yvar);
%         par.ymean
%         stop here;
    else
        par.yrel = 1.375;
        par.ymean = 1;
    end
else
    par.yrel = 1;
    par.ymean = 1; 
end

% boundaries of state space
par.zmax	= 15;                                   % grid max: asset - lambda*durable
par.dmax	= dmax;                                    % grid max: durable
par.xmax    = par.zmax+(1-par.fc)*par.dmax;         % grid max: COH
par.ymax    = par.ymean*par.yrel;                           % grid max: income
par.exomax  = 0;                                    % grid max: exo path
par.rmax    = par.r;                                % grid max: interest rate
par.rbmax   = par.rb;                                % grid max: interest rate
par.pmax    = par.p;                                 % grid max: rel durable price
par.yaggmax = par.yagg;                                % grid max: agg income
par.zaggmax = par.zagg;                                % grid max: agg income
par.taxmax   = 0;                                    % grid max: transfer
par.txprmax = 1;                                    % grid max: tax proportional
par.cfcmax   = 0;                                    % grid max: cfc
par.gmax    = par.g;                                % grid max: transfer
par.zmin	= 0;                                    % grid min: asset - lambda*durable
par.dmin	= dmin;                              % grid min: durable
par.xmin    = par.zmin + (1-par.lambda)*1.05*par.dmin ;  % grid min: COH
par.ymin    = par.ymean/par.yrel;                        % grid min: income
par.exomin  = 0;                                    % grid min: exo path
par.rmin    = par.r;                                % grid min: interest rate
par.rbmin   = par.rb;                                % grid min: interest rate
par.pmin    = par.p;                                 % grid min: rel durable price
par.yaggmin = par.yagg;                                % grid min: agg income
par.zaggmin = par.zagg;                                % grid min: agg income
par.taxmin   = 0;                                    % grid min: transfer
par.txprmin = 1;                                    % grid max: tax proportional
par.cfcmin   = 0;                                    % grid min: cfc
par.gmin    = par.g;                                     % grid min: transfer

par.dtriesmin = par.dmin;
par.dtriesmax = par.dmax;


% if par.ymin*par.yagg < (par.nu + par.lambda*(par.r+par.rspread+par.delta)).*par.pmax*par.dmax
%     warning('Code not set up to allow for future violation of borrowing constraint')
%     par.ymin*par.yagg - (par.nu + par.lambda*(par.r+par.rspread+par.delta)).*par.pmax*par.dmax
%     stop here
% end

% Additional nodes
par.zadd = zadd;[0,0.01]';[0,0.005,0.01,0.025,0.05,0.1]';[0,0.001,0.0025,0.005,0.01,0.025,0.05,0.1,0.2]';[0,0.001,0.005,0.01:0.01:0.1]';[0,0.01]';
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
par.Nzexp = par.Nz*Nexpand;
par.Ndexp = par.Nd*Nexpand;
par.dtexp = 1/200;

% Iteration set-up:
par.Delta     = 100000; % step size (for time) in the implicit method
par.maxit     = 200;
par.maxitpar  = 50;
par.maxitLCP  = 50;100;
par.lcpsteps  = 20;
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