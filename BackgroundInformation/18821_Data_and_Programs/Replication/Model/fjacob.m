function [jacob,policyexpset] = fjacob(par,parexp,nodesgrids,nodesgridsexp,policy,steady,shockvar,shocksize,jacobT,shocknotknown)
% function that computes jacobians for news shocks

%--------------------------------------------------------------------------
% Step 1: path of exogenous variables
%--------------------------------------------------------------------------

% dimension of jacobian
if nargin<=8
    jacobT = [par.irfT,par.irfT];
end
if nargin<=9
    shocknotknown=0;
end

if shocknotknown==1
    jacobT(2) = 1;
end
jacobT(2) = jacobT(2)+1;

polT = jacobT(2);
irfT = jacobT(1);

% add extra period to capture steady state
% par.irfT = polT+1;
% parexp.irfT = polT+1;

% shocks and shock variables
shocks.nodesexo = ones(1,polT);
shocks.exodrift = zeros(1,polT);
shocks.exosigma = zeros(1,polT);

if strcmp(shockvar,'d')
    shockvar = 'r';
    shockvars.r = zeros(1,polT);
elseif strcmp(shockvar,'txpr')
    shockvars.(shockvar) = [ones(1,polT-1)*shocksize,0];
else
    shockvars.(shockvar) = [zeros(1,polT-2),shocksize,0]; % add extra 0 to get steady state
end

% frequency in distribution iterations
par.distdt = min(par.irfdt);
parexp.distdt = min(parexp.irfdt);
if length(par.irfdt)>1
    par.irfdt = [par.irfdt,par.distdt];
    parexp.irfdt = [parexp.irfdt,parexp.distdt];
end

% Number of shocks
par.Nexo = size(shocks.nodesexo,1);
parexp.Nexo = size(shocks.nodesexo,1);

% if shocks include thsoe that affect the tax, then include other variables
% as well
shockvarsset = fieldnames(shockvars);
taxset = {'r','yagg','g','zagg','txpr'};
if max(ismember(shockvar,taxset))==1
    notincl = find(ismember(taxset,shockvarsset)==0);
    for kk=notincl
        shockvars.(taxset{kk}) = zeros(1,polT);
    end
end

% if the price path is included, need to add future price
if strcmp(shockvar,'p')==1
    if shocknotknown==0
        shockvars.ptp1 = [shockvars.p(2:end),0];
    elseif shocknotknown==1
        shockvars.ptp1 = shockvars.p;
    end
    nodesgrids.nodesptp1 = nodesgrids.nodesp;
    nodesgridsexp.nodesptp1 = nodesgridsexp.nodesp;
end
if strcmp(shockvar,'txpr')==1
    if shocknotknown==0
        shockvars.txprtp1 = [shockvars.txpr(2:end),0];
    elseif shocknotknown==1
        shockvars.txprtp1 = shockvars.txpr;
    end
    nodesgrids.nodestxprtp1 = nodesgrids.nodestxpr;
    nodesgridsexp.nodestxprtp1 = nodesgridsexp.nodestxpr;
end

%--------------------------------------------------------------------------
% Step 2: solve policy function backwards once with expected shock at irfT
%--------------------------------------------------------------------------

disp('Solving for Policy Functions')
tic
if shocknotknown==0
    [policyexpset,newgridsexpset] = fpolicytransition(par,parexp,nodesgrids,nodesgridsexp,policy,shocks,shockvars,jacobT);

%--------------------------------------------------------------------------
% Step 2.5: adjust policy function if HH do not know about the shock
%--------------------------------------------------------------------------

elseif (strcmp(shockvar,'r')==1 || strcmp(shockvar,'rb')==1 || strcmp(shockvar,'p')==1) && shocknotknown==1
    
    % this way we update the boundary condition, durable choices conditional on adjustment, but not the optimal
    % choices elsewhere
    par.lcpsteps=0;
    par.maxitLCP=1;
    
    % only run once through the solver, so use initial policy function
    [policyexpset,newgridsexpset] = fpolicytransition(par,parexp,nodesgrids,nodesgridsexp,policy,shocks,shockvars,jacobT);
    
    
end

% if strcmp(shockvar,'p')==1 && polT==2
%     % HH in the adjustment region also learn of the shock
%     for pol = fieldnames(policyexpset{1})'
%         policyexpset{1}.(pol{1})(newgridsexpset{1}.gridz>0) = policyexpset{2}.(pol{1})(newgridsexpset{1}.gridz>0);
%     end
% end
% end
toc
%--------------------------------------------------------------------------
% Step 3: solve for set of matrices to iterate on distributions
%--------------------------------------------------------------------------

% break up forward problem into two parts:
% (1) Find matrix to iterate distribution forward
% (2) Iterate on distribution

disp('Solving for Distributional Dynamics')
tic
for tt=1:polT
    distrmatexpset{tt} = fdistrmat(parexp,newgridsexpset{tt},policyexpset{tt});
    % don't need this anymore so free up memory
    if polT>10
        policyexpset{tt} = [];
    end
end
toc

%--------------------------------------------------------------------------
% Step 4: solve distribution forward
%--------------------------------------------------------------------------
disp('Construcing IRFs')
tic
% par.irfT-1-ss is the period in which news of the shock arrives. 
% if ss=par.irfT-1 then the shock occurs in the current period,
% if ss=1 the shock occurs par.irfT-1 periods in the future.

% store impulse response columnwise to create jacobian
jacob = struct;
    
for ss=1:1:polT-1   
    disp(['IRF ' num2str(polT-ss)])
    % set initial distribution equal to steady state
    psi0t = steady.psi1t;
    
    % steps
    irfdt = par.irfdt(ss:end);
    irfdt = [irfdt,par.distdt*ones(1,par.irfT-1-sum(irfdt)/par.distdt)];
    dtpoints = 1+[0,cumsum(irfdt)/par.distdt];

    % solve distribution forward for irfT periods
    for tt=1:irfT
    
        % if we reach a period with a new policy function, then update the
        % distribution matrix unless we are already at steady state in the
        % policy function 
        if max(tt==dtpoints)==1 && tt+ss-1<=polT % check this last conditon
            
            polloc = find(tt==dtpoints)+ss-1;
            
            newgridsexp = newgridsexpset{polloc};
            if tt==1
                if shocknotknown==1 && strcmp(shockvar,'p')==1
                    newgridsexp.gridpt0 = steady.p+shockvars.p(1);
                    newgridsexp.gridptp1 = steady.p;
                else
                    newgridsexp.gridpt0 = steady.p;
                end
            end

            distrmatexp = distrmatexpset{polloc};
        end
        
        % iterate on distribution
        [psi0t,irfq] = fdistriter(parexp,newgridsexp,distrmatexp,psi0t);
        
        % append irf results
        for var=fieldnames(irfq)'
           jacob.(var{1})(tt,polT-ss) =  (irfq.(var{1})(1) -steady.(var{1}))/shocksize;
        end
    end

    % we do not need this transition matrix anymore, so free up memory
    distrmatexpset{ss} = [];
end
toc


end

