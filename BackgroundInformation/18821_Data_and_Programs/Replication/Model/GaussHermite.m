function [x, w] = GaussHermite(mu,sigma,n)
    
    % calculate n Gauss-Hermite nodes for normal with mean mu and std.
    % sigma
    
    i = 1:n-1;
    a = sqrt(i/2);
    CM = diag(a,1) + diag(a,-1);
    [V, L]   = eig(CM);
    [x, ind] = sort(diag(L));
    x = x*sqrt(2)*sigma + mu;
    V = V(:,ind)';
    w = (sqrt(pi) * V(:,1).^2)./sqrt(pi);
    
end