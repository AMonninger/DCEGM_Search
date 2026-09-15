function [policyexpset,newgridsexpset] = fpolicytransition(par,parexp,nodesgrids,nodesgridsexp,policy,shocks,shockvars,jacobT)

% length of problem
polT = jacobT(2);
if length(par.irfdt)==1
    dtset = par.irfdt*ones(1,polT+2);
else
    dtset = par.irfdt;
end

% possible realizations of aggregate variables
Nexo = par.Nexo;

% initial grid
nodesgrids0 = nodesgrids;
nodesgridsexp0 = nodesgridsexp;

policy0=policy;

%--------------------------------------------------------------------------
% Step 1: solve policy function backwards
%--------------------------------------------------------------------------
tic
for tt=polT:-1:1 
    
    par.irfdt = dtset(tt);
    parexp.irfdt = dtset(tt);
    
    %----------------------------------------------------------------------
    % 2.1: Exogenous variables at time t and t+1
    %----------------------------------------------------------------------
    
    newgrids = struct;
    newgridsexp = struct;
    
    % start with all initial nodes if we expand grid
    if Nexo>1 && tt==polT
        newgrids = getnodes(par);
        newgridsexp = getnodes(parexp);
    end

    
    % update node counts based on dimension of exogenous variables
    [newgrids,par] = updatenodes(par,newgrids,nodesgrids0,shocks,shockvars,tt);
    [newgridsexp,parexp] = updatenodes(parexp,newgridsexp,nodesgridsexp0,shocks,shockvars,tt);
    
    % create new grids
    if Nexo>1 && tt==polT
        % if we expand the grid update all variables
        newgrids = getgrids(par,newgrids); 
        newgridsexp = getgrids(parexp,newgridsexp); 

    else
        % if we do not expand grid, just update exogenous variables
        newgridset = fieldnames(shockvars)';
        newgridset{end+1} = 'exo';
%         newgridset
        newgrids = getgrids(par,newgrids,newgridset); 
        newgridsexp = getgrids(parexp,newgridsexp,newgridset); 
    end
    

    % updating nodes
    for fn = fieldnames(newgrids)'
       nodesgrids.(fn{1}) = newgrids.(fn{1});
       nodesgridsexp.(fn{1}) = newgridsexp.(fn{1});
    end
    newgrids = nodesgrids;
    newgridsexp = nodesgridsexp;
    
    
    % boundaries for LCP solver
    lbnd = zeros(prod(newgrids.totnodesz),1);    % lower bound for z is 0, since then pick the max of v=vstar and adjust, so HJB equation is slack.
    ubnd = Inf*ones(prod(newgrids.totnodesz),1); % there is no upper bound for z.
    

    %----------------------------------------------------------------------
    % 2.2: solve for policy function at t
    %----------------------------------------------------------------------

    % next periods value function
    vtp1 = policy.v;
    policy.vtp1 = vtp1;
    
    % drift of exogenous variables
    newgrids = getdrifts(par,newgrids);

    % Construct B matrix (exgoenous drifts only)
    policy = buildBexog(par,newgrids,policy);
    

    % iterating on policy function until convergence for this period only
    z0 = policy.z;
    v0 = policy.v;
    vadj0 = policy.vadj;
    adj0 = (z0==0);

    % iterating on today's value function until convergence
%     tic
    
    for kk=1:par.maxitLCP

        % optimal consumption and savings choice
        policy = foptc(par,newgrids,policy);
        
            
        % instantaneous utility when not adjusting
        u = max(futility(policy.c,newgrids.gridd,par),par.minu);   % bound losses at boundary
        
        % endogenous drift when not adjusting
        policy = buildAendo(par,newgrids,policy);
 
        % B matrix for iterations
        B = (par.rho + par.theta + 1/dtset(tt))*speye(par.Nnodesz) - policy.A;
        
        % constant in HJB.
        uvec = u + vtp1/dtset(tt) + par.theta*policy.vadj;
        q = -uvec + B*policy.vadj;
        
        % simple solution without durables
        if par.Nd==1
            policy.v = B\uvec;
            policy.z = policy.v;
            policy.vadj = policy.v;
            policy.adjust = ones(size(policy.v));
            
            v_dist = max(abs((policy.v-v0)./v0));
            if v_dist<par.tolirf
                break
            end
            if kk==par.maxitLCP
                warning(['Iteration Period ', int2str(tt), ' not solved, iter ', int2str(kk), ', Vdist ' num2str(v_dist) ])
            end
            
        % otherwise use solver
        else
            z = z0;
            if isfield(par,'lcpsteps')==0
                par.lcpsteps=20;
            end
            LCP_error = 1;
            for mm=1:par.lcpsteps
                % run LCP solver
                z = LCP(B,q,lbnd,ubnd,z0,0);

                % check the accuracy of the LCP solution (should all be 0)
                LCP_error = max(abs(z.*(B*z + q))); 
                if LCP_error<1
                    break
                end
            end
            % store output
            policy.z = z;
            
            % convert back to value function
            policy.v = policy.z + policy.vadj;

            % value of adjusting
            policy = foptd(par,newgrids,policy);
            
            % check convergence: vadj
            v_dist = max(abs((policy.v-v0)./v0));
            vadj_dist = max(abs((policy.vadj-vadj0)./vadj0));
            z_dist = max(abs(policy.z-z0));  
            adj_dist = max(abs((policy.z==0)-adj0));
%             [LCP_error,v_dist,z_dist,adj_dist]
%             [v_dist,z_dist,adj_dist]
%             stop here;
            if LCP_error<par.tolirf && v_dist<par.tolirf && vadj_dist<par.tolirf && adj_dist==0 %&& z_dist<par.tol
                policy = rmfield(policy,'adjtries');
                break
            end
            if kk==par.maxitLCP
                warning(['Iteration Period ', int2str(tt), ' not solved, iter ', int2str(kk), ', Vdist ' num2str(v_dist) ', Vadjdist ' num2str(vadj_dist) ', adjdist ' num2str(adj_dist) ', zdist ' num2str(z_dist) ', LCP ' num2str(LCP_error) ])
                % do not want to update extensive margin for unknown shocks
                if par.maxitLCP==1
                    policy.vadj = vadj0;
                end
            end
        end
        
        % update initial conditions
        z0 = policy.z;
        v0 = policy.v;
        vadj0 = policy.vadj;
        adj0 = (z0==0);
    end
    % value of not adjusting
    policy.vnoadj = 1/( par.theta + 1/dtset(tt)) *(uvec - par.rho *policy.v + policy.A*policy.v);


    % drifts on expanded grid
    newgridsexp = getdrifts(parexp,newgridsexp);
    
    % store expanded grid for later use in distribution
    newgridsexpset{tt} = newgridsexp;  

    % policy function on expanded grid
    if parexp.Nnodesz>par.Nnodesz
        policyexp = fpolexpand(par,parexp,newgrids,newgridsexp,policy);  
    else
        policyexp = policy;
    end

    % store results
    if par.Nd==1
        saveset = {'c','A','vnoadj','adjust','sb'};
    else
        saveset = {'zprimeadj','dprimeadj','adjust','c','A','vnoadj'};
    end
    for kk=1:length(saveset)
        policyexpset{tt}.(saveset{kk}) = policyexp.(saveset{kk});
    end
    
    clear policyexp;
end


end