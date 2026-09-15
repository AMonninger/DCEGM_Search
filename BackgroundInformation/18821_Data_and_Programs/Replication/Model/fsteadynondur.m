function distribution = fsteadynondur(par,nodesgrids,policy,distribution)

if nargin==3
    psi0t = [];
    dt = 1; 
    itz = 1;
else
    psi0t = distribution.psi1t;
    dt = par.irfdt;
    itz = max(1000*dt,250);
end

distribution = struct;

%__________________________________________________________________________
%
% Loading structures
%__________________________________________________________________________ 

Nz = par.Nz;
Nd = par.Nd;
Nnodesz = par.Nnodesz;


% policies
polc = policy.c;



% transition matrix (no adjustments)
A = policy.A;
if max(abs(sum(A, 2))) > 1e-10
    max(abs(sum(A(1:100:5010,:), 2)))
    max(abs(sum(A, 2)))
    stop here;
    warning('Some rows of A do not sum to 0')
end

% nodes
ddistz = nodesgrids.ddistz;
ddistd = nodesgrids.ddistd;
nodesz = nodesgrids.nodesz;
nodesd = nodesgrids.nodesd;
nodesy = nodesgrids.nodesy;

% grids
gridz = nodesgrids.gridz;
gridd = nodesgrids.gridd;
grida = nodesgrids.grida;

gridy = nodesgrids.gridy;
gridyagg = nodesgrids.gridyagg(1);
gridg = nodesgrids.gridg(1);
gridr = nodesgrids.gridr(1);
gridrb = nodesgrids.gridrb(1);
gridzagg = nodesgrids.gridzagg(1);

% product of distances for trapezoidal integration
ddist = 1;
for kk=1:length(par.varsetz)
    eval(['ddist = ddist .* nodesgrids.ddist' par.varsetz{kk} ';'])
end
distribution.ddist = ddist;


%__________________________________________________________________________
%
% Step 1 : compute steady-state distribuition
%__________________________________________________________________________ 
% if there is no initial distribution, then calculate SS distribution
if isempty(psi0t) == 1
    

AT = A';
distribution.AT = AT;

% Fix one value (not in the adjustment region) so matrix isn't singular:
vec = zeros(Nnodesz,1);
iFix=par.Nz;
vec(iFix)=.01;
AT(iFix,:) = [zeros(1,iFix-1),1,zeros(1,Nnodesz-iFix)];


% Solve system:
psi0t = AT\vec;

psi0_sum = sum(psi0t);
psi0t = psi0t./psi0_sum;
distribution.psi0t = psi0t;
distribution.psi1t = psi0t;

if max(abs(AT * psi0t)) > 1e-6
    max(abs(AT * psi0t))
    warning('Solution to KFE not accurate')
end
if sum(psi0t < -1e-8) ~= 0
    min(psi0t)
    warning('Solution to KFE has negative values')
end

% convert to distribution
psi0 = psi0t./ddist ;


% distribution next period
psi1t  = psi0t;
psi1   = psi0;


if abs(trapint(psi0,{nodesz,nodesy})-1)>10^-8
    warning('Distribution does not integrate to 1')
end


%__________________________________________________________________________
%
% Step 4 (alternative) : calculate distribution for next period
%__________________________________________________________________________ 
else
    
    % initial values for flow calculations
    psi1t = psi0t;
    psi1 = psi1t./ddist;
    psi0 = psi0t./ddist;
    c = 0;
    muc = 0;
    yint = 0;
    inc = 0;
    spreadtot = 0;
    
	% initial assets
    a0 = trapint(grida.*psi0t./ddist,{nodesz,nodesy}) ;
    a1 = a0;
    neta = 0;
    
    % borrowing region
    gridaneg = grida.*(grida<0);
    
    % changes in distribution for iterations
    AT = A'*dt/itz;

    % then change in distribution determined by drifts captured in C
    % g_{t+dt} - g_{t} = C' g_{t} dt
    % we also calculate flow quantities that change over the interval here.
    for zz=1:itz

        % nondurable consumption
        c = c + trapint(polc.*psi1,{nodesz,nodesy})*dt/itz;
        
        % marginal utility of consumption
        muc = muc + trapint(margutility(polc,gridd,par).*psi1,{nodesz,nodesy})*dt/itz; 

        % spread
        spreadtot = spreadtot - gridrb*trapint(gridaneg.*psi1,{nodesz,nodesy})*dt/itz;
        
        % interest income
        yint = yint + gridr*a1*dt/itz;
        
        % change in distribution
        psi1t = psi1t + AT*psi1t;    % need to break down iteration to avoid negative values  
        
        % normalized
        psi1 = psi1t./ddist;
        
        
        if sum(psi1t < -1e-6) ~= 0
            zz
            min(psi1t)
            find(psi1t==min(psi1t))
%             warning('Distribution has negative values')
        end
        
        % new asset holdings
        a1 = trapint(grida.*psi1,{nodesz,nodesy});
        
        % savings, income, expenditures
        neta = neta + a1-a0;
        a0 = a1;

        y = gridyagg*trapint(gridy.*psi1,{nodesz,nodesy});
        inc = inc + (y+gridr*a0)*dt/itz;
        
    end

    c = c/dt;
    muc = muc/dt;
    spreadtot = spreadtot/dt;
    yint = yint/dt;
    
    expen =  c + spreadtot;
      
    % store new distribution
    distribution.psi1t = psi1t;

    % convert to distribution
    psi0 = psi0t./ddist ;
    psi1 = psi1t./ddist ;
    
    % check integration constant
    if abs(trapint(psi1t./ddist,{nodesz,nodesy})-1)>10^-8
        warning('t+1 Distribution does not integrate to 1')
    end
    if sum(psi1t < -1e-6) ~= 0
        min(psi0t)
        warning('Distribution has negative values')
%         stop here;
    end
   
end

%__________________________________________________________________________
%
% Step 5 : calculate quantities
%__________________________________________________________________________ 



% asset holdings (end of period) --- need to fix grida
a0 = trapint(grida.*psi0,{nodesz,nodesy});
a = trapint(grida.*psi1,{nodesz,nodesy});
apos = trapint(grida.*(grida>0).*psi1,{nodesz,nodesy});

% fraction borrowing
anegind = trapint((grida<0).*psi1,{nodesz,nodesy});

% post-tax income
y = gridyagg*trapint(gridy.*psi1,{nodesz,nodesy});

% these are the simpler caluclations for the steady state
if itz==1
    
    % nondurable consumption
    c = trapint(polc.*psi1,{nodesz,nodesy});
    
    % marginal utility of consumption
    muc = trapint(fmargutility(polc,gridd,par).*psi1,{nodesz,nodesy}); 

    % spread cost
    spreadtot = gridrb*(apos-a);
    
    % interest income
    yint = gridr*a;

end

% aggregate demand (includes payments of spread)
% disp('update2')
AD = c + gridg + spreadtot; 

% aggregate income
yagg = y + yint  + gridg;

% exog variables
r = gridr;
g = gridg;
rb = gridrb;
% neta
% expen
% inc

% variables not part of this model
d=gridd;
dc=gridd/c;
x=0;
adj=0;
xadj=0;
p=1;
fctot=0;
ocmctot=0;
octot=0;
depr=0;
yaggnew=yagg;

% aggregate productivity
zagg = gridzagg;
ypot = zagg;

steadyvars = {'y','a','a0','apos','c','r','rb','g','AD','yagg','anegind','spreadtot','muc','d','x','dc','adj','xadj','p','fctot','ocmctot','octot','depr','yaggnew','zagg','ypot','dc'};  
for jj=1:length(steadyvars)
    eval(['distribution.' steadyvars{jj} ' = ' steadyvars{jj} ';' ]) 
end

end