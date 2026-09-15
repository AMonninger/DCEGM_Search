function policy = fpolicy(par,nodesgrids,policy0)

if isfield(par,'itdisp')==0
    itdisp=1;
else
    itdisp=par.itdisp;
end

if nargin==2 || isfield(policy0,'v')==0 || par.Nd==1
    
% Construct B matrix (exgoenous drifts only)
policy = buildBexog(par,nodesgrids);

% Initial guess
policy.v  = futility(par.ymin + par.rho*nodesgrids.gridz,par.dmin,par)/par.rho;

% SOLVE WITHOUT ADJUSTMENT
% disp('Solving model without adjustment')

% Iteration
distv      = zeros(par.maxit,1);
for nn=1:par.maxit/5
    
    v0 = policy.v;
    
    % optimal consumption and savings choice
    policy = foptc(par,nodesgrids,policy);

    % instantaneous utility
    u = max(futility(policy.c,nodesgrids.gridd,par),par.minu);   % bound losses at boundary
%     min(u)
%     max(u)
%     min(policy.c)
%     max(policy.c)
%     min(utilityfunc(policy.c,nodesgrids.gridd,par))
%     max(utilityfunc(policy.c,nodesgrids.gridd,par))
    
    % payoffs
    uvec = u + v0/par.Delta  ;
    
    % endogenous drift
    policy = buildAendo(par,nodesgrids,policy);

    % B matrix for iterations
    B = (par.rho  + 1/par.Delta)*speye(par.Nnodesz) - policy.A ;% - policy.Aexog  - policy.Aendo;

    % iteration
    policy.v = B\uvec;

    distv(nn) = max(abs((policy.v-v0)./v0).*(abs(v0)>10^-2) + (policy.v-v0).*(abs(v0)<10^-2));
    if mod(nn,itdisp)==0
        disp(['Initial guess, no adjustment, iter ', int2str(nn), ', dist ' num2str(distv(nn)) ]);
    end
    if distv(nn)<par.tol
%         disp(['Value Function Converged, Iteration = ' int2str(nn)]);
        break
    end
    

end

% policy.dprimeadj = par.dmin*ones(par.Nnodesz,1);
% policy.zprimeadj = par.zmin*ones(par.Nnodesz,1);
% policy.adjust = logical(zeros(par.Nnodesz,1));
% steady = fsteady2(par,nodesgrids,policy);
% steady
if par.Nd==1
    policy.z=policy.v;
    policy.vadj=policy.v;
    policy.adjust=ones(size(policy.v));
    policy.dprimeadjx=ones(size(nodesgrids.gridxx));
    return
end

% initial guess
policy.vadj = min(policy.v)*ones(par.Nnodesz,1);
% policy.vnoadj = policy.v;
policy.z = max(v0 - policy.vadj,0);

else
    % otherwise use provided initial guess
    policy = policy0;
end


% SOLVE WITH ADJUSTMENT
% disp('Solving model with adjustment')

distv      = zeros(par.maxit,1);
distz      = zeros(par.maxit,1);
LCP_error  = zeros(par.maxit,1);
adj_dist   = zeros(par.maxit,1);

% boundaries for LCP solver
lbnd = zeros(par.Nnodesz,1);    % lower bound for z is 0, since then pick the max of v=vstar and adjust, so HJB equation is slack.
ubnd = Inf*ones(par.Nnodesz,1); % there is no upper bound for z.


for nn=1:par.maxit
    
    % initial value functions (for checking convergence)
    v0 = policy.v;
    z0 = policy.z;
    vadj0 = policy.vadj;
    adj0 = (policy.z==0);
    
    % optimal consumption and savings choice
    policy = foptc(par,nodesgrids,policy);

    % instantaneous utility when not adjusting
    u = max(futility(policy.c,nodesgrids.gridd,par),par.minu);   % bound losses at boundary
    
    % endogenous drift when not adjusting
    policy = buildAendo(par,nodesgrids,policy);
    
    % B matrix for iterations
    B = (par.rho + par.theta + 1/par.Delta)*speye(par.Nnodesz) - policy.A ;%- policy.Aexog  - policy.Aendo;

    % we need to cast the problem in the LCP solver form:
    % z' * (M * z + q) = 0
    % z >= 0
    % M * z + q >= 0
    % 
    % from the discretized HJBVI we have z = v(n+1) - Vstar, M=B so that the problem would be
    % z' * (B * v(n+1) - uvec ) = 0
    % z >= 0
    % B * v(n+1) - vec >= 0
    % but then we can just add and subtract B*Vstar to get the desired form, implying q = -vec + B*Vstar
%     z0 = max(v0 - policy.vadj,0);
    
    % constant in HJB
    uvec = u + v0/par.Delta + par.theta*policy.vadj;
    q = -uvec + B*policy.vadj;
    
    z = z0;
    for mm=1:20
        % run LCP solver
        z = LCP(B,q,lbnd,ubnd,z,0);

        % check the accuracy of the LCP solution (should all be 0)
        LCP_error(nn) = max(abs(z.*(B*z + q)));

        if LCP_error(nn)<0.1
%             [mm,LCP_error(nn)]
            break
        end
    end

    % store output
    policy.z = z;

    % convert back to value function
    policy.v = policy.z + policy.vadj;
    
    % optimal choice of d and corresponding vadj
    policy = foptd(par,nodesgrids,policy);
    
    

    distv(nn)=max(max(abs((policy.v-v0)./v0).*(abs(v0)>10^-2) + abs(policy.v-v0).*(abs(v0)<10^-2)),max(abs((policy.vadj-vadj0)./vadj0).*(abs(vadj0)>10^-2) + abs(policy.vadj-vadj0).*(abs(vadj0)<10^-2)));
    distz(nn)=max(abs((policy.z-z0)./z0).*(abs(z0)>1) + abs(policy.z-z0).*(abs(z0)<1)) ;
    adj_dist(nn) = max(abs((policy.z==0)-adj0));


    % check
%     policy.v = max(policy.v,policy.vadj);
    if mod(nn,itdisp)==0
        disp(['W/ adjustment, iter ', int2str(nn), ', Vdist ' num2str(distv(nn)) ', zdist ' num2str(distz(nn)) ', LCP ' num2str(LCP_error(nn)) ]);
    end

    if distv(nn)<par.tol && LCP_error(nn)<par.tol && distz(nn)<par.tol && adj_dist(nn)==0
        if nn>itdisp
            disp(['Value Function Converged, Iteration = ' int2str(nn)]);
        end
        break
    end
    % stops when progress no longer made
    if nn>1
        if max([distv(nn),distz(nn),LCP_error(nn)])<par.tol*100 && min([distv(nn)/distv(nn-1),distz(nn)/distz(nn-1),LCP_error(nn)/LCP_error(nn-1)])>0.97 
            disp('Asymptote, breaking loop')
            break
        end
    end

end
if distv(nn)>par.tol*100
    policy.notconverged=1;
end

% storing output
policy.B = B;

% policy function on (z,d) grid
policy = foptzd(par,nodesgrids,policy);


end