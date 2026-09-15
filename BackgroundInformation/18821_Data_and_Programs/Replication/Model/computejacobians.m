clear all; close all;

% models
modelset ={'house';'car';'all'}; 

% parameterizations %;'fc';'fcoc'
specset = {'nondurify';'fcoctheta';'fc';'fcoc';'fctheta'};

% frequency of simulations
irfdt = 1/4;

%--------------------------------------------------------------------------
% Step 1: Load steady state results
%--------------------------------------------------------------------------

for mm=1:length(modelset)
    model = modelset{mm};


for ss=1:length(specset)  
    spec = specset{ss};
    
% length of IRFs (dim of jacobian is irfT X irfT)
if strcmp(model,'all') && (strcmp(spec,'fcoctheta')==1 || strcmp(spec,'nondurify')==1)
    irfT = 100;    
else
    irfT = 32; 
end    
    
if strcmp(model,'all')==0 && strcmp(spec,'fcoctheta')==0 
    continue
end        
       

% saving/loading string
savehandle = [model spec];    

% loading saved policy function and steady state distribution
load(['Results/' savehandle '.mat'],'par','steady','policy')

% quarterly simulations
par.irfdt = irfdt;
par.irfT = irfT;

% Nodes
[nodesgrids,par] = getnodes(par);

% Grids
nodesgrids = getgrids(par,nodesgrids);

% Drifts
nodesgrids = getdrifts(par,nodesgrids);

% expand grid
parexp = par;
parexp.Nz = par.Nzexp;
parexp.Nd = par.Ndexp;
[nodesgridsexp,parexp] = getnodes(parexp,nodesgrids);          

nodesgridsexp = getgrids(parexp,nodesgridsexp);
nodesgridsexp = getdrifts(parexp,nodesgridsexp);
policyexp = fpolexpand(par,parexp,nodesgrids,nodesgridsexp,policy); 

%--------------------------------------------------------------------------
% Step 2: Compute Jacobians
%--------------------------------------------------------------------------

steady.yaggnew2 = steady.yaggnew;

% time intervals for aggregation
par.irfdt = irfdt;
parexp.irfdt = irfdt;

if strcmp(spec,'highmpc') 
    parexp.itdist = 120*min(parexp.irfdt);
    parexp.itvar  = parexp.itdist/5;
else
    parexp.itdist = 60*min(parexp.irfdt);
    parexp.itvar  = parexp.itdist/5;
end

% shocks
if strcmp(model,'all')==1 && par.psid>0
    shockvars = {'r','p','yagg','rb','g'}; 
elseif par.psid==0
    shockvars = {'r','yagg','rb','g'};
else
    shockvars = {'p','r','yagg'}; 
end
shocksize.p = 0.0025 ;
shocksize.r = 0.0025 ;
shocksize.yagg = 0.01;
shocksize.g = -0.01;
shocksize.rb = 0.01;
shocksize.py = 0.01;
parexp.irfT = par.irfT;

try
    load(['Results/' model spec 'jacob.mat'],'jacob');
catch
    jacob = struct;
    save(['Results/' model spec 'jacob.mat'],'jacob');
end
for shockvar = shockvars
    disp(shockvar)
    [jacob.(shockvar{1}),policyexpset] = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,shockvar{1},shocksize.(shockvar{1}));


    save(['Results/' model spec 'jacob.mat'],'jacob','-append'); %
end

%--------------------------------------------------------------------------
% Step 3: Compute impact of "unknown" r,p shock
%--------------------------------------------------------------------------

% r-shock and r^b shock only become known when budget constraint is violated
jacobnotknown.r = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,'r',shocksize.r,[par.irfT,1],1);
save(['Results/' model spec 'jacob.mat'],'jacobnotknown','-append');

jacobnotknown.rb = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,'rb',shocksize.rb,[par.irfT,1],1);
save(['Results/' model spec 'jacob.mat'],'jacobnotknown','-append');

if sum(strcmp(shockvars,'p'))>0
% for p-shock check BC and update intensive margin of adjustment
jacobnotknown.p = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,'p',shocksize.p,[par.irfT,1],1);
save(['Results/' model spec 'jacob.mat'],'jacobnotknown','-append');


%--------------------------------------------------------------------------
% Step 4: Compute durable overhang / productivity shock
%--------------------------------------------------------------------------

% all variables relative to productivity level z. Assume taxation such that
% aggregate liquid assets move immediately to new steady state level
ddist = 1;
for kk=1:length(par.varsetz)
    eval(['ddist = ddist .* nodesgridsexp.ddist' par.varsetz{kk} ';'])
end

% new grids
psi0t = steady.psi0t;
Nd = parexp.Nd;
Nz = parexp.Nz;
Ny = parexp.Ny;
Nnodesz = parexp.Nnodesz;
nodesz = nodesgridsexp.nodesz;
nodesd = nodesgridsexp.nodesd;
nodesy = nodesgridsexp.nodesy;
gridz = nodesgridsexp.gridz;
gridp = nodesgridsexp.gridp;
gridd0 = nodesgridsexp.gridd;
gridd1 = nodesgridsexp.gridd*(1+shocksize.py);
grida0 = gridz - par.lambda*gridp*gridd0;
grida1 = gridz - par.lambda*gridp*gridd1;

a0 = trapint(grida0.*steady.psi0t./ddist,{nodesz,nodesd,nodesy}) ;
d0 = trapint(gridd0.*steady.psi0t./ddist,{nodesz,nodesd,nodesy}) ;

% adjust asset dimensions, so that liquid assets are unchanged
% z' = a + lambda*p*d'
znew = nodesgridsexp.zint(gridz + par.lambda*gridp*(gridd1-gridd0));

znewindex = min(max(floor(znew),1),Nz-1);

weight_znew = znew-znewindex;

% position in distribution
znewindex = znewindex + Nz*floor(transpose(0:1:Nnodesz-1)/Nz);

% set of mass points
znewindex = [znewindex;znewindex+1];
probmass = [psi0t.*(1-weight_znew);psi0t.*weight_znew];

% add mass points to update distribution
psi0t = accumarray(znewindex,probmass,[Nnodesz, 1],[],0);

anew = trapint(grida0.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;

dnew = trapint(gridd0.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;

% now adjust durable dimension by the shock
% d' = d*(1+shock)
dnew = nodesgridsexp.dint(gridd1) ;

dnewindex = min(max(floor(dnew),1),Nd-1);

weight_dnew = dnew-dnewindex;

% position in distribution
dnewindex = (dnewindex-1)*Nz + repmat(transpose(1:1:Nz),[Nd*Ny,1]) + Nz*Nd*floor(transpose(0:1:Nnodesz-1)/(Nz*Nd));  % Nd*floor(transpose(0:1:Nnodesz-1)/Nd);

% set of mass points
dnewindex = [dnewindex;dnewindex+Nz];
probmass = [psi0t.*(1-weight_dnew);psi0t.*weight_dnew];

% add mass points to update distribution
psi0t = accumarray(dnewindex,probmass,[Nnodesz, 1],[],0);

dnew = trapint(gridd0.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;
% dnew/d0

anew = trapint(grida0.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;
% anew/a0

if sum(abs(anew-a0))>10^-6 && sum(abs(dnew-d0*(1+shocksize.py)))>10^-6
    error('Durable jump not calculated correctly')
end
steadypy = steady;
steadypy.psi0t = psi0t;
steadypy.psi1t = psi0t;

% Now solve for IRF of shock
jacobpy = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steadypy,'d',shocksize.py,[par.irfT,1]);

save(['Results/' model spec 'jacob.mat'],'jacobpy','-append');


else
    
for var =fieldnames(jacob.(shockvars{1}))'
    jacob.p.(var{1}) = zeros(size(jacob.(shockvars{1}).(var{1})));
    jacobpy.(var{1}) = zeros(size(jacob.(shockvars{1}).(var{1})));
    jacobpy2.(var{1}) = zeros(size(jacob.(shockvars{1}).(var{1})));
    jacobnotknown.p.(var{1}) = zeros(length(jacob.p.(var{1})),1);
end

save(['Results/' model spec 'jacob.mat'],'jacob','jacobnotknown','jacobpy','-append');

end
%--------------------------------------------------------------------------
% Step 5: Compute impact of monthly r
%--------------------------------------------------------------------------

parbase = par;
parexpbase = parexp;

% time intervals for aggregation
par.irfdt = 1/12;
parexp.irfdt = 1/12;

parexp.itdist = 60*min(parexp.irfdt);
parexp.itvar  = parexp.itdist/5;

% simulation length
par.irfT = 12;
parexp.irfT = par.irfT;

for shockvar = {'r'}
    jacobmon.(shockvar{1}) = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,shockvar{1},shocksize.(shockvar{1}));
    save(['Results/' model spec 'jacob.mat'],'jacobmon','-append');
end

jacobmonnotknown.r = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,'r',shocksize.r,[par.irfT,1],1);
save(['Results/' model spec 'jacob.mat'],'jacobmonnotknown','-append');

par = parbase;
parexp = parexpbase;

end
end