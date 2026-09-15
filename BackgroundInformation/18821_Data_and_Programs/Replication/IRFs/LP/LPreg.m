function [B,V]=LPreg(y,x,agg,nw)
% This function constructs an IRF using the Local projection method.
% 
% Inputs    Dimension   Description
%   y        T+H X 1        Dependent variable. H is the Horizon over which
%                           the IRF is constructed
% 
%   x        T X K          Independent variables. First column is the
%                           shock.
% 
%   agg      1 X 1          Set to 1 to construct cumulative IRF.
% 
% 
% Outputs   Dimension   Description
%  B          H X 1         IRF
%  V          H X H         Variance-Covariance matrix of the IRF
% 

% Input size
[T,K] = size(x);
H = size(y,1)-T+1;

% Checking inputs are consistent
if size(y,2)~=1
    disp('error')
end

% Default is to construct IRF in levels
if nargin==2
    agg = 0;
    nw = H;
elseif nargin==3
    nw = H;
end

% Construct inverse of x'x
% xxinv = inv(x'*x);

% beta coefficients and residuals
B = zeros(H,1);
R = zeros(T,H);
for hh=1:H
    % dependent variable: levels or cumulative if specified
    if agg==0 || (agg>0 && hh==1)
        ydep = y(hh:T+hh-1,1);
    elseif agg==1 && hh>1
        ydep = y(hh:T+hh-1,1) + ydep;
    elseif agg==2 && hh>1
        yadd = 0;
        for ll=1:hh
            yadd = yadd + y(ll:T+ll-1,1);
        end
        ydep = yadd + ydep;
    end
    
    % coefficients on all dependent variables
    Ball   = (x'*x)\(x'*ydep);

    % residuals
    R(:,hh) = ydep - x*Ball;
    
    % coefficient on IRF
    B(hh) = Ball(1);
end

% variance covariance matrix
V = zeros(H,H);
for hh=1:H
    
    
    for ii=hh:min(H,hh+nw)
        
        % sandwich estimator for element (hh,ii) of (X'OmegaX)
        Rhp = R(:,hh);
        Rhm = Rhp;
        Rim = R(:,ii);
        Rip = Rim;
        O = (x'*diag(Rhp.*Rim)*x);
        
        xp = x;
        xm = x;
        
        % Newey-West component: mm takes leads/lags of residuals of both hh
        % horizon and ii horizon.
        for mm=1:min(ii-1,nw)               % using ii>=hh
            xp = xp(2:end,:);
            xm = xm(1:end-1,:);

            % p = lead residuals
            % l = lag residuals
            % h = hh horizon
            % i = ii horizon
            Rip = Rip(2:end,1);
            Rhm = Rhm(1:end-1,1);

            Rhp = Rhp(2:end,1);
            Rim = Rim(1:end-1,1);

            % squared residuals: first term is lead in ii with lag in hh,
            % second term is lead in hh and lag in ii
            Rmat = diag(Rip.*Rhm + Rhp.*Rim); 

            % Add newey-West component
            O = O +  (1-mm/ii) ...
                * ( (xm' * Rmat) * xp );

        end
         
        
        % Adjust for degrees of freedom
        O = O.*(T/(T-K));
        
        % VCV of this block
        XOX = (x'*x)\O/(x'*x)  ;
        
        % IRF VCV is the first element
        V(ii,hh) = XOX(1,1);
        
        % using symmetry of the VCV matrix
        V(hh,ii) = V(ii,hh);
    end
    
    
end
    

end