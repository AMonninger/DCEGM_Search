function [ymean,yvariance,yvarianceinf,A,ggt] = fymean(ymeantarget,ybarinit,yrmin,yrmax,Ny,phiy,yrho,yvar)
% function that calculates mean income on a grid and spits out average
% (non-stochastic) mean income such that the stochastic mean is 1

maxit = 200;
% % ymax = ymaxinit;

for kk=1:maxit

    % boundaries
    ymin = ybarinit*yrmin;
    ymax = ybarinit*yrmax;

    % nodes
    nodesy = nonlinspace(ymin,ymax,Ny,phiy);

    % drift and variance
    ydrift = (log(yrho)*(log(nodesy)-log(ybarinit)) + yvar/2).*nodesy; % drift in levels
    ysigma = yvar.*nodesy.^2; 

    % difference on grid
    dy = nodesy(2:end)-nodesy(1:end-1);
    
    dyb = [dy(1);dy];
    dyf = [dy;dy(end)];

    ddist = 0.5*[dy(1);dy(2:end)+dy(1:end-1);dy(end)];

    % normalize drifts in finite differences
    ydiffLDiag   = -min(ydrift,0)./dyb + ysigma ./(dyb.*(dyf+dyb));
    ydiffCDiag   = -max(ydrift,0)./dyf + min(ydrift,0)./dyb  - ysigma ./(dyf.*dyb);
    ydiffUDiag   = max(ydrift,0)./dyf + ysigma ./(dyf.*(dyf+dyb));

    % adjust boundaries
    ydiffCDiag(nodesy<=ymin) = ydiffCDiag(nodesy<=ymin) + ydiffLDiag(nodesy<=ymin);
    ydiffCDiag(nodesy>=ymax) = ydiffCDiag(nodesy>=ymax) + ydiffUDiag(nodesy>=ymax);

    % reshape lower diagonal to conform with spdiag
    ydiffLDiag(1:Ny-1) = ydiffLDiag(2:Ny);
    ydiffLDiag(Ny)     = 0;

    ydiffUDiag(2:Ny)   = ydiffUDiag(1:Ny-1);
    ydiffUDiag(1)      = 0;
    
    % matrices of drifts
    A = spdiags(ydiffCDiag ,0,Ny,Ny) ...
        +spdiags(ydiffLDiag,-1,Ny,Ny) ...  
        +spdiags(ydiffUDiag,1,Ny,Ny);

    % SS distribution
    AT = A';
    b = zeros(Ny,1);

    %need to fix one value, otherwise matrix is singular
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,Ny-i_fix)];
    AT(i_fix,:) = row;

    %Solve linear system
    ggt = AT\b;
    g_sum = sum(ggt);
    ggt = ggt./g_sum;
    gg = ggt ./ddist;

    % mean income
    ymean = trapz(nodesy,nodesy.*gg);
    
    % variance of income
    yvariance = trapz(nodesy,(log(nodesy)-log(ymean)).^2.*gg); 
    
    
    crit = abs(ymean - ymeantarget);
    if crit<10^-10
        break
    end

    ybarinit = ybarinit - 0.5*(ymean - ymeantarget);

end

ymean = ybarinit;

yvarianceinf = yvar/(-2*log(yrho));

end

