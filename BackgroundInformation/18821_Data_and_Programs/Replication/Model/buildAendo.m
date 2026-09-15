function policy = buildAendo(par,nodesgrids,policy)

%% Unloading parameters
Nz = par.Nz;
Nnodesz = par.Nnodesz;

sb = policy.sb;
sf = policy.sf;
Ib = policy.Ib;
If = policy.If;

dzb = nodesgrids.dzb;
dzf = nodesgrids.dzf;

%% Building matrix components
% drift of z
%zdrift  = gridr.*gridz + gridy - c  + gridconstrdrift;
zLDiag   = -Ib.*sb./dzb;
zCDiag   = -If.*sf./dzf + Ib.*sb./dzb;
zUDiag   = If.*sf./dzf;

% reshape lower diagonal to conform with spdiag
zLDiagresh  = reshape(zLDiag,Nz,Nnodesz/Nz);
zLDiagresh(1:Nz-1,:) = zLDiagresh(2:Nz,:);
zLDiagresh(Nz,:) = zeros(1,Nnodesz/Nz);
zLDiag = reshape(zLDiagresh,Nnodesz,1);

zUDiagresh  = reshape(zUDiag,Nz,Nnodesz/Nz);
zUDiagresh(2:Nz,:) = zUDiagresh(1:Nz-1,:);
zUDiagresh(1,:) = zeros(1,Nnodesz/Nz);
zUDiag = reshape(zUDiagresh,Nnodesz,1);
    
% endogenous drifts
policy.Aendo = spdiags(zCDiag,0,Nnodesz,Nnodesz) ...
    +spdiags(zLDiag,-1,Nnodesz,Nnodesz) ...
    +spdiags(zUDiag,1,Nnodesz,Nnodesz);


policy.A = policy.Aendo+policy.Aexog;

end