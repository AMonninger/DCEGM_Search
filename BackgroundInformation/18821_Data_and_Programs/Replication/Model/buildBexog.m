function policy = buildBexog(par,nodesgrids,policy)

if nargin==2
    policy=struct;
end

%--------------------------------------------------------------------------
% Unload parameters
%--------------------------------------------------------------------------
Nz = par.Nz;
Nd = par.Nd;
Ny = par.Ny;
Nexo = par.Nexo;
Nnodesz = par.Nnodesz;

dmin = par.dmin;
ymin = par.ymin;
dmax = par.dmax;
ymax = par.ymax;

ddrift = nodesgrids.ddrift;
ydrift = nodesgrids.ydrift;
ysigma = nodesgrids.ysigma;

ddb = nodesgrids.ddb;
ddf = nodesgrids.ddf;
dyb = nodesgrids.dyb;
dyf = nodesgrids.dyf;

gridd = nodesgrids.gridd;
gridy = nodesgrids.gridy;

% la = par.la;
% rho = par.rho;
% Delta = par.Delta;
% theta = par.theta;
% 
% totnodesz  = nodesgrids.totnodesz;

%--------------------------------------------------------------------------
% 
% Building matrix components
% 
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Step 1: drift of d
%--------------------------------------------------------------------------
if Nd>1
    dLDiag   = -min(ddrift,0)./ddb;
    dCDiag   = -max(ddrift,0)./ddf + min(ddrift,0)./ddb;
    dUDiag   = max(ddrift,0)./ddf;

    % drift at boundaries
    dLDiag(gridd<=dmin) = 0;
    dUDiag(gridd>=dmax) = 0;

    % reshape diagonals to conform with spdiag
    dLDiagresh  = reshape(dLDiag,Nz,Nd,Nnodesz/(Nz*Nd));
    dLDiagresh(:,1:Nd-1,:) = dLDiagresh(:,2:Nd,:);
    dLDiagresh(:,Nd,:) = zeros(Nz,1,Nnodesz/(Nz*Nd));
    dLDiag = reshape(dLDiagresh,Nnodesz,1);

    dUDiagresh  = reshape(dUDiag,Nz,Nd,Nnodesz/(Nz*Nd));
    dUDiagresh(:,2:Nd,:) = dUDiagresh(:,1:Nd-1,:);
    dUDiagresh(:,1,:) = zeros(Nz,1,Nnodesz/(Nz*Nd));
    dUDiag = reshape(dUDiagresh,Nnodesz,1);

    Ad = spdiags(dCDiag ,0,Nnodesz,Nnodesz) ...
        +spdiags(dLDiag,-Nz,Nnodesz,Nnodesz) ...
        +spdiags(dUDiag,Nz,Nnodesz,Nnodesz);
else
    Ad=0;
end
%--------------------------------------------------------------------------
% Step 2: drift of y
%--------------------------------------------------------------------------

if Ny>1
    ydiffLDiag   = -min(ydrift,0)./dyb + ysigma ./(dyb.*(dyf+dyb));
    ydiffCDiag   = -max(ydrift,0)./dyf + min(ydrift,0)./dyb  - ysigma ./(dyf.*dyb);
    ydiffUDiag   = max(ydrift,0)./dyf + ysigma ./(dyf.*(dyf+dyb));

    % adjust boundaries
    ydiffCDiag(gridy<=ymin) = ydiffCDiag(gridy<=ymin) + ydiffLDiag(gridy<=ymin);
    ydiffCDiag(gridy>=ymax) = ydiffCDiag(gridy>=ymax) + ydiffUDiag(gridy>=ymax);


    % reshape lower diagonal to conform with spdiag
    ydiffLDiagresh  = reshape(ydiffLDiag,Nz*Nd,Ny,Nnodesz/(Nz*Nd*Ny));
    ydiffLDiagresh(:,1:Ny-1,:) = ydiffLDiagresh(:,2:Ny,:);
    ydiffLDiagresh(:,Ny,:) = zeros(Nz*Nd,1,Nnodesz/(Nz*Nd*Ny));
    ydiffLDiag = reshape(ydiffLDiagresh,Nnodesz,1);

    ydiffUDiagresh  = reshape(ydiffUDiag,Nz*Nd,Ny,Nnodesz/(Nz*Nd*Ny));
    ydiffUDiagresh(:,2:Ny,:) = ydiffUDiagresh(:,1:Ny-1,:);
    ydiffUDiagresh(:,1,:) = zeros(Nz*Nd,1,Nnodesz/(Nz*Nd*Ny));
    ydiffUDiag = reshape(ydiffUDiagresh,Nnodesz,1);

    Ay = spdiags(ydiffCDiag ,0,Nnodesz,Nnodesz) ...
        +spdiags(ydiffLDiag,-Nz*Nd,Nnodesz,Nnodesz) ...  
        +spdiags(ydiffUDiag,Nz*Nd,Nnodesz,Nnodesz);
else

    Ay = sparse([],[],[],Nnodesz,Nnodesz);

end


%--------------------------------------------------------------------------
% Step 3: drift of aggregate exogenous variables
%--------------------------------------------------------------------------

if Nexo>1
    
    gridexo = nodesgrids.gridexo;
    exodrift = nodesgrids.exodrift;
    exosigma = nodesgrids.exosigma;
    
    exomax = max(nodesgrids.nodesexo);
    exomin = min(nodesgrids.nodesexo);
    
    Nexo = length(nodesgrids.nodesexo);
    
    dexob = nodesgrids.dexob;
    dexof = nodesgrids.dexof;
    
    exodiffLDiag   = -min(exodrift,0)./dexob + exosigma ./(dexob.*(dexof+dexob));
    exodiffCDiag   = -max(exodrift,0)./dexof + min(exodrift,0)./dexob  - exosigma ./(dexof.*dexob);
    exodiffUDiag   = max(exodrift,0)./dexof + exosigma ./(dexof.*(dexof+dexob));

    % adjust boundaries
    exodiffCDiag(gridexo<=exomin) = exodiffCDiag(gridexo<=exomin) + exodiffLDiag(gridexo<=exomin);
    exodiffCDiag(gridexo>=exomax) = exodiffCDiag(gridexo>=exomax) + exodiffUDiag(gridexo>=exomax);

    % reshape lower diagonal to conform with spdiag
    exodiffLDiagresh  = reshape(exodiffLDiag,Nnodesz/Nexo,Nexo);
    exodiffLDiagresh(:,1:Nexo-1) = exodiffLDiagresh(:,2:Nexo);
    exodiffLDiagresh(:,Nexo) = zeros(Nnodesz/Nexo,1);
    exodiffLDiag = reshape(exodiffLDiagresh,Nnodesz,1);
    

    exodiffUDiagresh  = reshape(exodiffUDiag,Nnodesz/Nexo,Nexo);
    exodiffUDiagresh(:,2:Nexo) = exodiffUDiagresh(:,1:Nexo-1);
    exodiffUDiagresh(:,1) = zeros(Nnodesz/Nexo,1);
    exodiffUDiag = reshape(exodiffUDiagresh,Nnodesz,1);

    Aexo = spdiags(exodiffCDiag ,0,Nnodesz,Nnodesz) ...
        +spdiags(exodiffLDiag,-Nnodesz/Nexo,Nnodesz,Nnodesz) ...  
        +spdiags(exodiffUDiag,Nnodesz/Nexo,Nnodesz,Nnodesz);
    

    
else

    Aexo = sparse([],[],[],Nnodesz,Nnodesz);

end
policy.Aexo  = Aexo;

%--------------------------------------------------------------------------
% Step 4: combine matrices
%--------------------------------------------------------------------------

policy.Aexog = Ad + Ay + Aexo;


end