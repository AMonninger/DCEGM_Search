function policy = foptd(par,nodesgrids,policy)

%% Unload parameters
lambda = par.lambda;
fc = par.fc;

Nz = par.Nz;
Nd = par.Nd;
Nexo=par.Nexo;

Ndtries = par.Ndtries;
nodesdtries = nodesgrids.nodesdtries;

ymin = par.ymin;
zmax = par.zmax;
xmax = par.xmax;
dmin = par.dmin;
dmax = par.dmax;

totnodesz = nodesgrids.totnodesz;
nodessetz = nodesgrids.nodessetz;

totnodesx = nodesgrids.totnodesx;
nodessetx = nodesgrids.nodessetx;

coh = nodesgrids.coh;
% exosetx = nodesgrids.exosetx;
dtries = nodesgrids.dtries;
yx = nodesgrids.yx;

grida = nodesgrids.grida;
gridd = nodesgrids.gridd;
gridy = nodesgrids.gridy;
gridp = nodesgrids.gridp;
gridcfc = nodesgrids.gridcfc;
gridxx = nodesgrids.gridxx;
gridyx = nodesgrids.gridyx;
gridpx = nodesgrids.gridpx;

if Nexo>1
    exox = nodesgrids.exox;
    gridexo = nodesgrids.gridexo;
end

interp = par.interp;

% initial condition
v = policy.v;

if isfield(policy,'adjtries')
    adjtries0 = policy.adjtries;
else
    adjtries0 = zeros(size(dtries));
end

% adjustment subsidy
if isfield(par,'cfcsize')
    subsidy = gridcfc(1)*(dtries>par.cfcsize);
else
    subsidy = 0;
end

%% Set of feasible durable choices
% relative price of durable
if par.Np>1
    pdtries = repmat(gridpx,[1,par.Ndtries]);
else
    pdtries = gridpx;
end

% maximum d from borrowing constraint
dtries(pdtries.*dtries>(coh+subsidy)/(1-lambda)) = NaN;


%% Adjustment policy function
% implied policy choice: adjust
policy.adjust = (policy.z<=0);

% sum(policy.adjust)

% never optimal to adjust at BC with minimum d stock due to fixed cost
policy.adjust(1:Nz*Nd:end) = 0; 

% convert to double
adjust = round(double(policy.adjust));


%% Solve for optimal durable stock

% set to NaN tries in adjustment region (which would imply we instantaneous
% adjust twice)
adjint = griddedInterpolant(nodessetz(totnodesz>1),reshape(adjust,totnodesz),'linear','linear');

if par.Ny>1 && par.Nexo>1
    adjtries = floor(adjint(min(coh-(1-lambda)*pdtries.*dtries,zmax),dtries,yx,exox));
elseif par.Ny>1
    adjtries = floor(adjint(min(coh-(1-lambda)*pdtries.*dtries,zmax),dtries,yx));
else
    adjtries = floor(adjint(min(coh-(1-lambda)*pdtries.*dtries,zmax),dtries));
end
policy.adjtries = adjtries;

dtries(adjtries==1) = NaN;

% if there are no admissible solutions based on current adjustment rule,
% use previous adjustment rule
if min(sum(~isnan(dtries),2))==0
%    warning('no trial solution') 
   % find locations
   nosol = find(sum(~isnan(dtries),2)==0);
    
   % set allowable d based on previous set of allowable d
   dtemp = nodesgrids.dtries(nosol,:);
   adj0temp = adjtries0(nosol,:);
   dtemp(adj0temp==1) = NaN;
   dtries(nosol,:) = dtemp;

   % update adjustment set
   policy.adjtries(nosol,:) = adj0temp;


else
    nosol = [];
end

% value function interpolant
Vint = griddedInterpolant(nodessetz(totnodesz>1),reshape(v,totnodesz),'linear','linear');

% pick the maximum among the options using the grid points as guesses
if par.Ny>1 && par.Nexo>1
    [vx,dindex] = max(Vint(min(coh-(1-lambda)*pdtries.*dtries + subsidy,zmax),dtries,yx,exox),[],2);
elseif par.Ny>1
    [vx,dindex] = max(Vint(min(coh-(1-lambda)*pdtries.*dtries + subsidy,zmax),dtries,yx),[],2);
else
    [vx,dindex] = max(Vint(min(coh-(1-lambda)*pdtries.*dtries + subsidy,zmax),dtries),[],2);
end
dprimeadjx = nodesdtries(dindex);


% interpolate value function cond on adjustment onto a-d grid
Vintx = griddedInterpolant(nodessetx(totnodesx>1),reshape(vx,totnodesx),'linear','linear');

if par.Ny>1 && par.Nexo>1
    policy.vadj = Vintx(grida + (1-fc).*gridp.*gridd,gridy,gridexo);
elseif par.Ny>1
    policy.vadj = Vintx(grida + (1-fc).*gridp.*gridd,gridy);
else
    policy.vadj = Vintx(grida + (1-fc).*gridp.*gridd);
end



% add to structure
policy.vx = vx;
policy.dprimeadjx = dprimeadjx;


end