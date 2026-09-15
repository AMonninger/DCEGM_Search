function IRF = VARcomputeirf(VARstruct)
% Inputs:
%     VARstruct =  structure of VAR
% 
% Output
%       IRF   = Structure with IRFs

% unload structure
irfh    = VARstruct.irfh;
B       = VARstruct.B;
Sigma   = VARstruct.Sigma;
aggr    = VARstruct.aggr;
choldc  = VARstruct.choldc;
nlag    = VARstruct.nlag;
nvar    = VARstruct.nvar;

% companion form
Bcomp = [B(:,1:nvar*nlag); eye(nvar*(nlag-1)) zeros(nvar*(nlag-1),nvar)];

% Compute IRFs
for ii=1:nvar

    % Initialize the impulse response vector
    response = zeros(nvar, irfh);
    
    % Create the impulse vector
    impulse = zeros(nvar,1); 
    
    % Get the matrix invA containing the structural impulses
    if choldc
        invAT = chol(Sigma);
        invA  = invAT';
    else
        invA  = eye(size(Sigma));
    end
    
    % Construct unitary IRF
    impulse(ii,1) = 1/invA(ii,ii); % unitary shock

    % First period impulse response (=impulse vector)
    response(:,1) = invA*impulse;

    % Make it comparable with companion
    impulse_comp  = [response(:,1)' zeros(1, nvar*(nlag-1))]';
    
    % Recursive computation of impulse response
    Bcomp_nn = eye(size(Bcomp,1)); 
    for nn = 2:irfh
        Bcomp_nn = Bcomp * Bcomp_nn; % this is the multiplier Bcomp^n
        response_comp  = Bcomp_nn * impulse_comp;
        response(:,nn) = response_comp(1:nvar);
    end
    IRF(:,:,ii) = response';
    
    % aggregate IRFs if specified
    for nn=1:nvar
        while aggr(nn)>0
            IRF(:,nn,ii) = cumsum(IRF(:,nn,ii));
            aggr(nn) = aggr(nn)-1;
        end
    end
end

