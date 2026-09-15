function Datadraw = VARsimul(Data,VARstruct)
% Inputs:
%     VARstruct =  structure of VAR
% 
% Output
%       IRF   = Structure with IRFs

% unload elements
nobs = VARstruct.nobs;
nlag = VARstruct.nlag;
nvar = VARstruct.nvar;
nblock = VARstruct.nblock;
cons = VARstruct.cons;
lambda = VARstruct.lambda;
B = VARstruct.B;
if isfield(VARstruct,'Bbc')
  B = VARstruct.Bbc;    % for this case use bias-corrected 
end
R = VARstruct.R;
if lambda~=0
   R = R(1:end-1,:); % we do not use the dummy residual for bootstrapping
end

% cashing data matrix
% Datadraw = zeros(nobs+nlag,nvar);
Datadraw = zeros(nobs,nvar);

% drawing initial starting point
u0 = 1;ceil(rand(1)*(nobs-nlag+1));

Datadraw(1:nlag,:) = Data(u0:u0+nlag-1,:);

% drawing residuals
R_draw = NaN(nobs,nvar);
for bb=1:floor(nobs/nblock)
    u1 = ceil((size(R,1)-nblock+1)*rand(nobs,1));
    R_draw(1+(bb-1)*nblock:bb*nblock,:) = R(u1:u1+nblock-1,:);
end
u1 = ceil((size(R,1)-nblock+1)*rand(nobs,1));
R_draw(bb*nblock+1:nobs,:) = R(u1:u1+(nobs-bb*nblock)-1,:);


% Iterate to compute time series
for tt = nlag+1:nobs %+nlag
    % lag matrix up to today
    [trash,XLAG] = VARlags(Datadraw(tt-nlag:tt,:),[],nlag,cons,u0-1+tt-nlag-1,0);

    % Compute the value for time=jj
    Datadraw(tt,:) = XLAG * B' + R_draw(tt-nlag,:);
end
% Datadraw = Datadraw(nlag+1:nlag+nobs,:);



end