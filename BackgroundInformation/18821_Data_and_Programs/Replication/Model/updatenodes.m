function [newgrids,par] = updatenodes(par,newgrids,nodesgrids,shocks,shockvars,tt)

% update grid sizes
newgrids.totnodesz = nodesgrids.totnodesz;
newgrids.totnodesx = nodesgrids.totnodesx;

newgrids.totnodesz(strcmp(par.varsetz,'exo')) = par.Nexo;
newgrids.totnodesx(strcmp(par.varsetx,'exo')) = par.Nexo;

par.Nnodesz = prod(newgrids.totnodesz);
par.Nnodesx = prod(newgrids.totnodesx);

% adding new nodes of exogenous aggregate vars
newgrids.nodesexo = shocks.nodesexo(:,tt);
newgrids.dexo = newgrids.nodesexo(2:end,1)-newgrids.nodesexo(1:end-1,1);

newgrids.nodessetz = nodesgrids.nodessetz;
newgrids.nodessetx = nodesgrids.nodessetx;

newgrids.nodessetz{strcmp(par.varsetz,'exo')} = newgrids.nodesexo;
newgrids.nodessetx{strcmp(par.varsetx,'exo')} = newgrids.nodesexo;

% adding drift and variance of exogenous aggregate vars
newgrids.exodrift = shocks.exodrift(:,tt);
newgrids.exosigma = shocks.exosigma(:,tt);

% add shock variables to grid list
for fn=fieldnames(shockvars)'
    eval(['newgrids.nodes' fn{1} ' = nodesgrids.nodes' fn{1} ';'])
    newgrids.shockvars.(fn{1}) = shockvars.(fn{1})(1,tt);
    
    
end
    
end