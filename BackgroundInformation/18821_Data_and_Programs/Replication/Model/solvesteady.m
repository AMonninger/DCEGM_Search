% This file solves for the steady state policy functions and distributions
% for a set of models and parameterizations

clear all; close all;


% models
modelset = {'car';'house';'all'};

% parameterizations
specset = {'nondurify';'fc';'fcoc';'fctheta';'fcoctheta'};

for mm=1:length(modelset)
    model = modelset{mm};
for ss=1:length(specset)  
    spec = specset{ss};
    
if strcmp(model,'all')==0 && strcmp(spec,'fcoctheta')==0 
    continue
end

%--------------------------------------------------------------------------
% Step 1: Initialization
%--------------------------------------------------------------------------

% Initial Model Parameters
eval(['par = setparameters' model '(''' spec ''');']);

% saving string
savehandle = [model spec];

%--------------------------------------------------------------------------
% Step 2: Solve model given estimated parameters
%--------------------------------------------------------------------------

% Nodes
[nodesgrids,par] = getnodes(par);

% Grids
nodesgrids = getgrids(par,nodesgrids);

% Drifts
nodesgrids = getdrifts(par,nodesgrids);

% Solve for policy function
policy = fpolicy(par,nodesgrids);   

% expand the grid for distribution
parexp = par;
parexp.Nz = par.Nzexp;
parexp.Nd = par.Ndexp;
[nodesgridsexp,parexp] = getnodes(parexp,nodesgrids);          

nodesgridsexp = getgrids(parexp,nodesgridsexp);
nodesgridsexp = getdrifts(parexp,nodesgridsexp);
policyexp = fpolexpand(par,parexp,nodesgrids,nodesgridsexp,policy);  

% Solve for steady state distribution
hazT = parexp.hazT;
parexp.hazT = parexp.hazTlong;
steady = fsteady(parexp,nodesgridsexp,policyexp);
parexp.hazT = hazT;



%--------------------------------------------------------------------------
% Step 3: Saving output
%--------------------------------------------------------------------------


% saving output
try
    save(['Results/' savehandle '.mat'],'par','policy','steady','-append')
catch
    save(['Results/' savehandle '.mat'],'par','policy','steady')
end


end
end