function [VARstruct] = VARestimate(Data,Exog,VARstruct)
% Inputs:
%     Data    = matrix of endogenous variables
%     Exog    = matrix of exogenous variables
%     VARstruct = structure with VAR specification
% 
% Output
%       VAR = structure with outputs

%--------------------------------------------------------------------------
% Step 1: checking inputs
%--------------------------------------------------------------------------

nlag = VARstruct.nlag;
cons = VARstruct.cons;
if isfield(VARstruct,'choldc')==0
    VARstruct.choldc=0;
end
if isfield(VARstruct,'aggr')==0
    VARstruct.aggr=0;
end
if isfield(VARstruct,'irfh')==0
    VARstruct.irfh=0;
end
if isfield(VARstruct,'lambda')==0
    VARstruct.lambda=0;
end
lambda = VARstruct.lambda;
if isempty(Exog)
    nlagexo=0;
    VARstruct.nlagexo=nlagexo;
elseif isfield(VARstruct,'nlagexo')==0
    nlagexo=0;
    VARstruct.nlagexo=nlagexo;
end

% observations and variables
[nobs,nvar] = size(Data);

% number of exogenous variables
[~,nvarexo] = size(Exog);

% number of coefficients
ncoeff = nlag*nvar + cons + nvarexo*nlagexo;

% add to structure
VARstruct.nvarexo = nvarexo;
VARstruct.nobs = nobs;
VARstruct.nvar = nvar;
VARstruct.ncoeff = ncoeff;

%--------------------------------------------------------------------------
% Step 2: construct x,y matrix for estimation
%--------------------------------------------------------------------------

[Y,XLAG] = VARlags(Data,Exog,nlag,cons,0,lambda);


%--------------------------------------------------------------------------
% Step 3: estimate by OLS
%--------------------------------------------------------------------------

% all coefficients
B = (XLAG'*XLAG)\(XLAG'*Y);
BT = B';

% residuals
R = Y - XLAG*B;


% Covariance Matrix of the Residuals
Sigma = (1/(nobs-nlag-ncoeff))*(R'*R);




% add to structure
VARstruct.B = BT;
% VARstruct.Bcomp = Bcomp;
VARstruct.Sigma = Sigma;
VARstruct.R = R;

% %--------------------------------------------------------------------------
% % Step 4: construct impulse response
% %--------------------------------------------------------------------------
% 
% IRF = VARirf(VARstruct);
% 
% % add to structure
% VARstruct.IRF = IRF;

end