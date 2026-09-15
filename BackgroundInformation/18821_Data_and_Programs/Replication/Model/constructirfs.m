clear all; close all;

% models
modelset ={ 'all'; 'house';'car'};%'all'

% parameterizations
specset ={'nondurify';'fcoctheta';'fc';'fcoc';'fctheta'};

% frequency
irfdt = 1/4;

% tax duration
tmax=32;

%--------------------------------------------------------------------------
% Step 1: Load PE Jacobian
%--------------------------------------------------------------------------

for irfdt=[1/4,1/12]

for mm=1:length(modelset)
    model = modelset{mm};
    
for ss=1:length(specset)  
    spec = specset{ss};
    
if strcmp(model,'all')==0 && strcmp(spec,'fcoctheta')==0 
    continue
end

if irfdt==1/12 && strcmp(model,'car')==0 && strcmp(spec,'fcoctheta')==0
    continue
end
    
clear jacob jacobif;

% saving/loading string
savehandle = [model spec]; 

% loading steady state
load(['Results/' savehandle '.mat'],'par','steady')

% info rigidity
omstinf = 1-par.stinf*(4*irfdt); % fraction not updating every time period

% load jacobians
if irfdt==1/4
    load(['Results/' model spec 'jacob.mat'],'jacob','jacobpy','jacobnotknown');
    savefrq = '';
    if strcmp(spec,'nondur')==0 && strcmp(spec,'nondurif')==0
        jacob.py = jacobpy;
    end
    
    % add productivity matrix
    jacob.zagg = jacob.yagg;
    jacob.zagg.zagg = eye(size(jacob.zagg.AD));
    if isfield(jacob.zagg,'ypot')==0
        jacob.zagg.ypot = eye(size(jacob.zagg.AD));
    end
    for shock=fieldnames(jacob)'
        if isfield(jacob.(shock{1}),'ypot')==0
            jacob.(shock{1}).ypot = zeros(size(jacob.zagg.AD));
        end
    end
    for shock=fieldnames(jacobnotknown)'
        if isfield(jacobnotknown.(shock{1}),'ypot')==0
            jacobnotknown.(shock{1}).ypot = zeros(length(jacob.zagg.AD),1);
        end
    end

   
elseif  irfdt==1/12
    load(['Results/' model spec 'jacob.mat'],'jacobmon','jacobmonnotknown');
    jacob = jacobmon;
    jacobnotknown = jacobmonnotknown;
    savefrq = 'mon';
end

% length of IRFs (dim of jacobian is irfT X irfT)
irfT = size(jacob.r.r,1);



%--------------------------------------------------------------------------
% Step 2: Incorporate sluggish information updating
%--------------------------------------------------------------------------

% fraction of households that learn each period
Tinfomax = 400;
fracupdate = (1-omstinf)*omstinf.^[0:1:Tinfomax];

% with real rate:
% know p_t and p_{t+1}. assume p_{t+k}=p_{t+1} etc
% could split into r_{t-1}^d=-(p_t-p_{t-1}), r_{t+1}^d=-(p_t-p_{t-1})
% r_{t-1}^d is always zero, r_{t+1}^d captures change in p_t relative to
% p_t+1, r_{t+k}^d has low/high price p_t until time k when p_{t+k} returns to
% steady state
% note: the first column is r_{t+1}^d
steady.yaggnew2 = steady.yaggnew;
if isfield(jacob,'p')
    for shock = {'rd'}
        for var = fieldnames(jacob.p)'

            jacob.(shock{1}).(var{1}) = zeros(irfT,irfT);

            % ss is the time of the shock
            for ss=1:irfT
                jacob.(shock{1}).(var{1})(:,ss) = sum(jacob.p.(var{1})(:,1:ss),2);
            end

            jacobnotknown.rd.(var{1}) = jacobnotknown.p.(var{1});
        end

    end
end

% adjust IRFs for incomplete information
for shock = fieldnames(jacob)'
    for var = fieldnames(jacob.(shock{1}))'
        
        jacobif.(shock{1}).(var{1}) = zeros(irfT,irfT);
        
        if max(strcmp(var,{'y','r','rb','g','yagg','yaggnew','yaggnew2','zagg'}))==0
            for ss=1:irfT

                % we deal with PI/tax shock later
                if strcmp(shock{1},'py') && ss>1
                    continue
                end
                
                

                prob=0;
                % how many people know of the initial shock at t=0
                jacobif.(shock{1}).(var{1})(:,ss) = jacob.(shock{1}).(var{1})(:,ss)*fracupdate(1);

                % we cumulate probability here for testing
                prob = prob+fracupdate(1);

                % between t=1 and t=ss-1 a fraction of households learn about the shock
                for kk=2:ss
                    jacobif.(shock{1}).(var{1})(:,ss) = jacobif.(shock{1}).(var{1})(:,ss) + [zeros(kk-1,1);jacob.(shock{1}).(var{1})(1:irfT-(kk-1),ss-(kk-1))]*fracupdate(kk);
                    prob = prob+fracupdate(kk);
                end

                % at t=ss the remaining households learn about the shock, as it is
                % reflected in the budget constraint and there is no information at a
                % later date
                if max(strcmp(shock,{'yagg','g','py','zagg','txpr'}))==1
                    jacobif.(shock{1}).(var{1})(:,ss) = jacobif.(shock{1}).(var{1})(:,ss) + [zeros(ss-1,1);jacob.(shock{1}).(var{1})(1:irfT-(ss-1),1)]*sum(fracupdate(ss+1:end));
                    prob = prob + sum(fracupdate(ss+1:end));
                    
                elseif max(strcmp(shock,{'r','rb','p','rd'}))==1
                    % For HH that do not know about prices, we make sure
                    % they do not violate budget constraints
                    jacobif.(shock{1}).(var{1})(:,ss) = jacobif.(shock{1}).(var{1})(:,ss) + [zeros(ss-1,1);jacobnotknown.(shock{1}).(var{1})(1:irfT-(ss-1),1)]*sum(fracupdate(ss+1:end));
                    prob = prob + sum(fracupdate(ss+1:end));
                end
                


                if abs(prob-1)>10^-4
                    error('probabilities do not add')
                end

            end
        else
            % if the variable is exogenous, sticky info does not affect
            % jacobian
            for ss=1:irfT
                if strcmp(shock{1},'py') && ss>1
                    continue
                end
                
                jacobif.(shock{1}).(var{1})(:,ss) = jacob.(shock{1}).(var{1})(:,ss);
            end
        end
    end
end


% HH think PY shock is transitory until they learn new information
% if isfield(jacob,'py')
if isfield(jacob,'py')
for shock = {'py'}
    for var = fieldnames(jacob.(shock{1}))'
        
        jacobif.(shock{1}).(var{1}) = zeros(irfT,1);
        


        % HH gradually learn about permanent productivity shock
        jacobif.(shock{1}).(var{1})(:,1) = zeros(irfT,1);
        for kk=1:irfT
            jacobif.(shock{1}).(var{1})(kk:end,1) = jacobif.(shock{1}).(var{1})(kk:end,1) ...
                                  + jacob.(shock{1}).(var{1})(1:end-kk+1,1)*(1-omstinf)*omstinf^(kk-1) ...
                                  - jacob.zagg.(var{1})(1:end-kk+1,1)*omstinf^kk ;
        end
        

    end
end



end

% save
eval(['jacob' savefrq 'if = jacobif;'])

save(['Results/' model spec 'jacob.mat'],['jacob' savefrq 'if'],'-append');

%--------------------------------------------------------------------------
% Step 3: IRFs in GE
%--------------------------------------------------------------------------


if isfield(par,'pdac') && irfdt==1/4

for pdac = par.pdac 
    for jacobian = {'jacob','jacobif'}
        
        clear jacobge00

        eval(['jacob00 = ' jacobian{1} ';'])


        % jacobians for GE responses
        if pdac==0
            for varadd = fieldnames(jacob00.yagg)'
                jacob00.p.(varadd{1}) = zeros(irfT,irfT); 
            end
        end
        
        dA.p = jacob00.p.a;
        dXfc.p = jacob00.p.x+jacob00.p.fctot;
        
        dA.yagg = jacob00.yagg.a;
        dXfc.yagg = jacob00.yagg.x+jacob00.yagg.fctot;

        % endogenous price response
        % p_t/p = (x_t/x)^pdac => x_t \approx 1/pdac * p_t * (x/p)
        dXfcdPequ = 1/pdac*(steady.x+steady.fctot)*eye(irfT);
        if pdac==0
            dXfcdPequ = eye(irfT);
        end

        % variables for which we calculate output

        % solve for price and income paths such that liquid assets are
        % constant (no real rate feedback)
        for var = {'r','r2','rb','g','py','pytax','zagg'} %,'p'
            if isfield(jacob,var{1})==0
                continue
            end

            for ss=1:irfT
                if strcmp(var{1},'py') && ss>1
                    continue
                end
                dA0 = zeros(irfT,1);

                dA.(var{1}) = jacob00.(var{1}).a(:,ss);
                if pdac~=0
                    dXfc.(var{1}) = jacob00.(var{1}).x(:,ss)+jacob00.(var{1}).fctot(:,ss);
                else
                    dXfc.(var{1}) = zeros(irfT,1);
                end
                
                if strcmp(var{1},'p')==0
                    % endogenous income response
                    irfy = -( dA.yagg + dA.p* ( (dXfcdPequ-dXfc.p) \ dXfc.yagg) ) \ ...
                             (-dA0 + dA.(var{1})+ dA.p* ( (dXfcdPequ-dXfc.p) \ dXfc.(var{1}) ) ) ;

                    % endogenous price / real rate response
                    irfp = (dXfcdPequ-dXfc.p) \ (dXfc.yagg*irfy+dXfc.(var{1}));
                else
                    % here we calculate GE income response and fix the
                    % price change
                    % endogenous income response
                    irfy = -( dA.yagg ) \ ...
                             (-dA0 + dA.(var{1}) ) ;

                    % endogenous price / real rate response
                    irfp = zeros(irfT,1);
                end
                
                % checking results
                if strcmp(var{1},'p')==0
                    dAtot = dA.yagg*irfy+dA.(var{1})+dA.p*irfp;
                else
                    dAtot = dA.yagg*irfy+dA.(var{1});
                end
                if sum(abs(dAtot-dA0))>10^-8
                    error('Calculation incorrect. GE not achieved.')
                end 

                % compile in GE jacobian:
                for varadd = fieldnames(jacob00.(var{1}))'
                    jacobge00.(var{1}).(varadd{1})(:,ss) =  jacob00.(var{1}).(varadd{1})(:,ss) + ...
                                                    jacob00.yagg.(varadd{1})*irfy + ...
                                                    jacob00.p.(varadd{1})*irfp;    
                end
                
            end

        end

        eval([jacobian{1} 'ge  = jacobge00;'])

        save(['Results/' model spec 'jacob.mat'],[jacobian{1} 'ge'],'-append');
    end

    
end
end

end
end
end