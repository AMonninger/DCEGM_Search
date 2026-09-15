function [VARstruct] = VARirf(Data,Exog,VARstruct)
% Inputs:
%     data    = matrix of endogenous variables
%     Exog    = matrix of exogenous variables
%     VARstruct = structure with VAR specification
% 
% Output
%       VAR = structure with outputs

%--------------------------------------------------------------------------
% Step 1: checking inputs
%--------------------------------------------------------------------------

if isfield(VARstruct,'irfci')==0
    VARstruct.irfci=0;
end
if isfield(VARstruct,'nblock')==0
    VARstruct.nblock=1;
end
if isfield(VARstruct,'ndraws')==0
    VARstruct.ndraws=0;
end
if isfield(VARstruct,'sebias')==0
    VARstruct.sebias=0;
end
if isempty(Exog)==0
    errort('Exog variables not implemented here')
end

% elements we need from structure
irfci  = VARstruct.irfci;
sebias = VARstruct.sebias;
ndraws = VARstruct.ndraws;

% results from VAR:
B      = VARstruct.B;
nvar   = VARstruct.nvar;
nlag   = VARstruct.nlag;

%--------------------------------------------------------------------------
% Step 2: construct impulse response
%--------------------------------------------------------------------------


% IRFs
IRF = VARcomputeirf(VARstruct);

VARstruct.IRF = IRF;

if irfci==1
    
%--------------------------------------------------------------------------
% Step 3.1: construct standard errors by bootstrap
% 
% We first bias correct the OLS estimate
%--------------------------------------------------------------------------

% first construct bias estimate
if sebias==1
    
    B_drawset = NaN([size(B),ndraws]);
    
    for nn=1:ndraws
            
        % new simulated data
        Datadraw = VARsimul(Data,VARstruct);

        % estimate on artificial data
        VARstruct_draw = VARestimate(Datadraw,Exog,VARstruct);

        % store estimates of B
        B_drawset(:,:,nn) = VARstruct_draw.B;
            
    end
    
    % Bias
    Bavg_draw = mean(B_drawset,3);
    Bbias = Bavg_draw-B;
    
    % Add to structure
    VARstruct.Bbias = Bbias;
    
    % Ensure stationarity 
    Broot = 1.1;
    delta = 1;
    while Broot>1
       Bbc = B - Bbias*delta; 

       % companion form
       Bcomp = [Bbc(:,1:nvar*nlag); eye(nvar*(nlag-1)) zeros(nvar*(nlag-1),nvar)];

       % new root
       Broot = max(abs(real(eig(Bcomp))));

       if Broot>=1
           delta = delta-0.01;
       end
    end
    VARstruct.delta = delta;
    VARstruct.Bbc = Bbc;
    
end
  
%--------------------------------------------------------------------------
% Step 3.2: Bootstrap the IRF
%--------------------------------------------------------------------------

IRF_drawset = NaN([size(IRF),ndraws]);

for nn=1:ndraws

    % new simulated data
    Datadraw = VARsimul(Data,VARstruct);

    % estimate on artificial data
    VARstruct_draw = VARestimate(Datadraw,Exog,VARstruct);
    
    % if bias correction is available use it here
    if sebias==1
        VARstruct_draw.B = VARstruct_draw.B - Bbias.*delta ;
    end
    
    % IRF
    IRF_drawset(:,:,:,nn) = VARcomputeirf(VARstruct_draw);
    
end
        
% CIs
IRFcil = prctile(IRF_drawset,2.5,4);
IRFciu = prctile(IRF_drawset,97.5,4); 

% Add to structure
VARstruct.IRF_drawset = IRF_drawset;
VARstruct.IRFcil = IRFcil;
VARstruct.IRFciu = IRFciu;
    

end