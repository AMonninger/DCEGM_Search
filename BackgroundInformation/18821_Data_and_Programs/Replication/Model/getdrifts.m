function nodesgrids = getdrifts(par,nodesgrids)

%----------------------------------------------------------------------
% 1. Load from structure
%----------------------------------------------------------------------

% parameters
delta = par.delta;
lambda = par.lambda;
dmin = par.dmin;
yrho = par.yrho;
yvar = par.yvar;
ymean = par.ymean;

Nz = par.Nz;
Nnodesz = par.Nnodesz;

% grids
gridz = nodesgrids.gridz;
gridp = nodesgrids.gridp;
gridd = nodesgrids.gridd;
gridy = nodesgrids.gridy;
gridyx = nodesgrids.gridyx;
gridr = nodesgrids.gridr;
gridrb = nodesgrids.gridrb;
gridpdt = nodesgrids.gridpdt;

%----------------------------------------------------------------------
% 
% 2. Borrowing grid and drift at constraint
% 
%----------------------------------------------------------------------

% z = distance to borrowing constraint. Here we determine whether the
% borrowing constraint at the state (d,y) is the LTV constraint or the PTY
% constraint

% liquid assets and interest rate
nodesgrids.grida	=   gridz - lambda*gridp*gridd; 
nodesgrids.gridrspread	=   gridr + (nodesgrids.grida<0).*gridrb;
% nodesgrids = rmfield(nodesgrids,'gridrb');

% interest rate grid
nodesgrids.gridrspreadresh   =   reshape(nodesgrids.gridrspread,[Nz,Nnodesz/Nz,1]);
nodesgrids.gridrspreadminz   =   nodesgrids.gridrspreadresh(1,:);
nodesgrids.gridrspreadmaxz   =   nodesgrids.gridrspreadresh(Nz,:);

% % remove base interest rate grids to make sure they are not used later
% nodesgrids = rmfield(nodesgrids,'gridr');
% nodesgrids = rmfield(nodesgrids,'gridrresh');
% nodesgrids = rmfield(nodesgrids,'gridrminz');
% nodesgrids = rmfield(nodesgrids,'gridrmaxz');

% drift of borrowing constraint in z-component
nodesgrids.gridconstrdrift = - lambda*(nodesgrids.gridrspread+delta-gridpdt).*gridp.*gridd ;

nodesgrids.gridconstrdriftresh   =   reshape(nodesgrids.gridconstrdrift,[Nz,Nnodesz/Nz,1]);
nodesgrids.gridconstrdriftminz   =   nodesgrids.gridconstrdriftresh(1,:);
nodesgrids.gridconstrdriftmaxz   =   nodesgrids.gridconstrdriftresh(Nz,:);

% % drift of borrowing constraint in x-component
% nodesgrids.gridconstrdriftx = - lambda*(nodesgrids.gridrspread+delta).*gridd ;
% 
% nodesgrids.gridconstrdriftreshx   =   reshape(nodesgrids.gridconstrdriftx,[Nx,Nnodesz/Nx,1]);
% nodesgrids.gridconstrdriftminx   =   nodesgrids.gridconstrdriftreshx(1,:);
% nodesgrids.gridconstrdriftmaxx   =   nodesgrids.gridconstrdriftreshx(Nx,:);


%----------------------------------------------------------------------
% 
% 3. Drift of the durable stock
% 
%----------------------------------------------------------------------

% durable stock
nodesgrids.ddrift  = -delta*gridd;

% adjust drift at end of state space
nodesgrids.ddrift(gridd<=dmin) = 0;


%----------------------------------------------------------------------
% 
% 4. Drift and variance of idiosyncratic income
% 
%----------------------------------------------------------------------

% Income (in levels)
% process is
% $\ln y_{t+dt} = yrho^{dt} (\ln y_t - \ln ybar) + \ln ybar +  e_{t+dt}$ 
% $\ln y_{t+dt} - \ln y_t = (e^{\ln(yrho) dt}-1) (\ln y_t - \ln ybar) + \sigma dt w_{t}$ 
% $d \ln y_t = \ln(yrho) (\ln y_t - \ln ybar) + \sigma w_{t}$ 
% $d y_t = [\ln(yrho) (\ln y_t - \ln ybar)+ \sigma^2/2 ]y_t + \sigma w_{t}y_t^2$
% pick mean of this distribution, such that E(y)=ybar?
nodesgrids.ydrift = (log(yrho)*(log(gridy)-log(ymean)) + yvar/2).*gridy; % drift in levels
nodesgrids.ysigma = yvar.*gridy.^2; % variance in levels

nodesgrids.ydriftx = (log(yrho)*(log(gridyx)-log(ymean)) + yvar/2).*gridyx; % drift in levels
nodesgrids.ysigmax = yvar.*gridyx.^2; % variance in levels




end