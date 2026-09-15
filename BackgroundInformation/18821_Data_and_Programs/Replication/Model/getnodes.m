function [nodesgrids,par] = getnodes(par,nodesgrids0)

varsetall  = par.varsetall;
varsetz  = par.varsetz;
varsetx  = par.varsetx;


%% Construct nodes
for kk=1:length(varsetall)
    
    if nargin==1
        % additional elements near constraints
        eval(['N' varsetall{kk} ' = par.N' varsetall{kk} '-length(par.' varsetall{kk} 'add)+1;'])
        eval([varsetall{kk} 'min = par.' varsetall{kk} 'min+par.' varsetall{kk} 'add(end);'])
        
        % build nodes
        eval(['nodesgrids.nodes' varsetall{kk} ' = nonlinspace(' varsetall{kk} 'min,par.' varsetall{kk} 'max,N' varsetall{kk} ',par.phi' varsetall{kk} ');'])
        eval(['nodesgrids.nodes' varsetall{kk} ' = [par.' varsetall{kk} 'min+par.' varsetall{kk} 'add;nodesgrids.nodes' varsetall{kk} '(2:end)];'])
    else
        % if we are expanding grids, then we add nodes between the existing
        % nodes
        eval(['Norg = length(nodesgrids0.nodes' varsetall{kk} ');']);
        eval(['nexp = par.N' varsetall{kk} '/ Norg;'])
        
        if nexp>1
            % existing length of addition
            eval(['len = length(par.' varsetall{kk} 'add);']) 
            
            % collect segments between original nodes
            eval(['allsegments = transpose(nodesgrids0.nodes' varsetall{kk} '(1:len));'])
            for nn=len:Norg-1
                eval(['segmentnn = linspace(nodesgrids0.nodes' varsetall{kk} '(nn),nodesgrids0.nodes' varsetall{kk} '(nn+1),nexp+1);'])
                allsegments=[allsegments,segmentnn(2:end)];
            end
            % new nodes and number of nodes
            eval(['nodesgrids.nodes' varsetall{kk} ' = transpose(allsegments);'])
            eval(['par.N' varsetall{kk} ' = length(nodesgrids.nodes' varsetall{kk} ');'])
        else
            eval(['nodesgrids.nodes' varsetall{kk} ' = nodesgrids0.nodes' varsetall{kk} ';'])
        end
    end
    
    

    % distance between nodes
    if eval(['par.N' varsetall{kk} '>1;'])
        eval(['nodesgrids.d' varsetall{kk} ' = nodesgrids.nodes' varsetall{kk} '(2:end)-nodesgrids.nodes' varsetall{kk} '(1:end-1);'])
        eval(['nodesgrids.d' varsetall{kk} '2 = nodesgrids.d' varsetall{kk} '(2:end).*nodesgrids.d' varsetall{kk} '(1:end-1);'])
    else
        eval(['nodesgrids.d' varsetall{kk} ' = 0;'])
        eval(['nodesgrids.d' varsetall{kk} '2 = 0;'])
    end
end

% assemble nodes on each grid
for kk=1:length(varsetz)
    eval(['nodesgrids.nodessetz{kk} = nodesgrids.nodes' varsetz{kk} ';'])
end
for kk=1:length(varsetx)
    eval(['nodesgrids.nodessetx{kk} = nodesgrids.nodes' varsetx{kk} ';'])
end

% mapping nodes to index points
nodesgrids.zint = griddedInterpolant(nodesgrids.nodesz,1:1:par.Nz);
if par.Nd>1
    nodesgrids.dint = griddedInterpolant(nodesgrids.nodesd,1:1:par.Nd);
end


%% length of grids
for kk=1:length(varsetz)
    eval(['nodesgrids.totnodesz(kk) = par.N' varsetz{kk} ';'])
end
for kk=1:length(varsetx)
    eval(['nodesgrids.totnodesx(kk) = par.N' varsetx{kk} ';'])
end

par.Nnodesz = prod(nodesgrids.totnodesz);
par.Nnodesx = prod(nodesgrids.totnodesx);


% total assets
% par.Abar = par.yagg * par.targets.a;
par.Abar = par.targets.a;


end