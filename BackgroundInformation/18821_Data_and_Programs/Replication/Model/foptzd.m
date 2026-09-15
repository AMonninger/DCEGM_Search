function policy = foptzd(par,nodesgrids,policy)

%% Unload parameters
lambda = par.lambda;
fc = par.fc;%.*(nodesgrids.gridd>par.dmin)-par.fc/100.*(nodesgrids.gridd<=par.dmin);

dmax = par.dmax;
% dmin = par.dmin;
zmax = par.zmax;
% zmin = par.zmin;
% tol = par.tol;

totnodesx = nodesgrids.totnodesx;
nodessetx = nodesgrids.nodessetx;

grida = nodesgrids.grida;
gridd = nodesgrids.gridd;
gridp = nodesgrids.gridp;
gridy = nodesgrids.gridy;
gridcfc = nodesgrids.gridcfc;
gridexo = nodesgrids.gridexo;

% initial condition
v = policy.v;
vadj = policy.vadj;
dprimeadjx = policy.dprimeadjx;

%% Adjustment policy function
% implied policy choice: adjust
policy.adjust = (policy.z<=0);

% never optimal to adjust at BC with minimum d stock due to fixed cost
policy.adjust(1:par.Nz*par.Nd:end) = 0; 

% convert to double
% policy.adjust = round(double(policy.adjust));


%% Policy function

% implied policy choice: durable cond on adjustment
dprimadjxint = griddedInterpolant(nodessetx(totnodesx>1),reshape(dprimeadjx,totnodesx),'linear','linear');
if par.Ny>1 && par.Nexo>1
    dprimeadj = min(dprimadjxint(grida + (1-fc).*gridp.*gridd,gridy,gridexo),dmax); 
elseif par.Ny>1
    dprimeadj = min(dprimadjxint(grida + (1-fc).*gridp.*gridd,gridy),dmax);  
else
    dprimeadj = min(dprimadjxint(grida + (1-fc).*gridp.*gridd),dmax);  
end

% adjustment subsidy
if isfield(par,'cfcsize')
    subsidy = gridcfc(1)*(dprimeadj>par.cfcsize);
else
    subsidy = 0;
end

% BC is enforced at the nodes, but interpolation may violate it by small
% amounts at state space boundary
policy.dprimeadj = min(dprimeadj,  (grida + (1-fc).*gridp.*gridd + subsidy)/(1-lambda));

% implied policy choice: liquid asset cond on adjustment
policy.zprimeadj = min(grida + (1-fc).*gridp.*gridd + subsidy - (1-lambda)*gridp.*policy.dprimeadj ,zmax);


end