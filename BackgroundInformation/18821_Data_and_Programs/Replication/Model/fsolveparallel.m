function [distance,policy] = fsolveparallel(par,x,targetparset,target,policy0,expandgrid)

    
    % parameters
    for kk=1:length(targetparset)
        par.(targetparset{kk})    = x(kk);
        
        % if we changex aggregate variables need to change boundaries as well
        if sum(ismember(par.varsetall,targetparset{kk}))>0
%             loc = find(ismember(par.varsetall,targetparset{kk}));
            eval(['par.' targetparset{kk} 'min = x(kk);'])
            eval(['par.' targetparset{kk} 'max = x(kk);'])
        end
    end
    
    
    % Nodes
    [nodesgrids,par] = getnodes(par);
    
    
    % Grids
    nodesgrids = getgrids(par,nodesgrids);
    
    % Drifts
    nodesgrids = getdrifts(par,nodesgrids);
    
    % calculating steady-state policy function
    policy = fpolicy(par,nodesgrids,policy0);

    if expandgrid==1
        % expand the grid for distribution
        parexp = par;
        parexp.Nz = par.Nzexp;
        parexp.Nd = par.Ndexp;

        [nodesgridsexp,parexp] = getnodes(parexp,nodesgrids);
        nodesgridsexp = getgrids(parexp,nodesgridsexp);
        nodesgridsexp = getdrifts(parexp,nodesgridsexp);

        % policy function on expanded grid
        policyexp = fpolexpand(par,parexp,nodesgrids,nodesgridsexp,policy);  

        % caluclating steady-state distribution on expanded grid
        steady = fsteady(parexp,nodesgridsexp,policyexp);
    else
        % caluclating steady-state distribution
        steady = fsteady(par,nodesgrids,policy);
    end
   
    % distance from target
    targetset = fieldnames(target);
    distance = zeros(length(targetset),1);
    for kk=1:length(targetset)
        distance(kk,1) =  (target.(targetset{kk})-steady.(targetset{kk}))./target.(targetset{kk});
    end
    
end