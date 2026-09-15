function [psi1t,irf] = fdistriter(par,nodesgrids,distrmat,psi0t)

    
% elements of distrmat:
% polc, adjust
% CTdt
% CadjinstantT
% AendoadjT

%__________________________________________________________________________
%
% Loading structures
%__________________________________________________________________________ 

Nz = par.Nz;
Nd = par.Nd;
Nnodesz = par.Nnodesz;
theta = par.theta;
delta = par.delta;
fc = par.fc;
nu = par.nu;
chi = par.chi;

% policies
adjust = distrmat.adjust;
polc = distrmat.polc;

% nodes
nodesz = nodesgrids.nodesz;
nodesd = nodesgrids.nodesd;
nodesy = nodesgrids.nodesy;

% grids
gridz = nodesgrids.gridz;
gridd = nodesgrids.gridd;
gridp = nodesgrids.gridp(1);
grida = nodesgrids.grida;
if isfield(nodesgrids,'gridptp1')
    gridptp1 = nodesgrids.gridptp1(1);
    gridatp1 = gridz - par.lambda*gridptp1*gridd;
else
    gridptp1 = nodesgrids.gridp(1);
    gridatp1 = grida;
end
% gridy = nodesgrids.gridy;
gridyagg = nodesgrids.gridyagg(1);
gridzagg = nodesgrids.gridzagg(1);
gridtxpr = nodesgrids.gridtxpr(1);
gridg = nodesgrids.gridg(1);
gridr = nodesgrids.gridr(1);%par.r;
gridrb = min(nodesgrids.gridrb);%0;

% time change and iteration steps
dt = par.distdt;
itdist = par.itdist;
itvar = par.itvar;

% product of distances for trapezoidal integration
ddist = 1;
for kk=1:length(par.varsetz)
    eval(['ddist = ddist .* nodesgrids.ddist' par.varsetz{kk} ';'])
end


%__________________________________________________________________________
%
% Step 1: check for price jump, which requires converting initial
% distribution's z from last periods to this periods z
%__________________________________________________________________________ 

% if there is a jump in price, then there is a shift in z and we need to
% modify the distribution here
if isfield(nodesgrids,'gridpt0')
    
    gridpt0 = nodesgrids.gridpt0;
    grida0 = gridz - par.lambda*gridpt0*gridd;
    
    a0 = trapint(grida0.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;

    % get new node positions and weights
    % z' = a + lambda*p'*d
    % z  = a + lambda*p*d
    znew = nodesgrids.zint(gridz + par.lambda*(gridp-gridpt0)*gridd);

    znewindex = min(max(floor(znew),1),Nz-1);

    weight_znew = znew-znewindex;

    % position in distribution
    znewindex = znewindex + Nz*floor(transpose(0:1:Nnodesz-1)/Nz);
    
    % set of mass points
    znewindex = [znewindex;znewindex+1];
    probmass = [psi0t.*(1-weight_znew);psi0t.*weight_znew];
    
    % add mass points to update distribution
    psi0t = accumarray(znewindex,probmass,[Nnodesz, 1],[],0);
    
    anew = trapint(grida.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;
    
    if sum(abs(anew-a0))>10^-6
        warning('Price jump not calculated correctly')
    end
end

%__________________________________________________________________________
%
% Step 2: Iterate on the distribution using policy matrix
%__________________________________________________________________________ 

% instantaneous adjustments
psi1t = psi0t + distrmat.CadjinstantT*psi0t;
clear CadjinstantT;

% initial distrbutions
psi1 = psi1t./ddist;
psi0 = psi0t./ddist;

% initial values for flow calculations
adj =  trapint(adjust.*psi0,{nodesz,nodesd,nodesy}) ;
depr = 0;
c = 0;
x = trapint(gridd.*(psi1-psi0),{nodesz,nodesd,nodesy});
muc = 0;
fctot = trapint(fc.*gridd.*adjust.*psi0,{nodesz,nodesd,nodesy})  ;
octot=0;
spreadtot = 0;
yint = 0;
yint2 = 0;

% expenditure and savings on instantaneous adjustments
a0 = trapint(grida.*psi0,{nodesz,nodesd,nodesy}) ;
a1 = trapint(grida.*psi1,{nodesz,nodesd,nodesy});
a00= a0;
% a0 = a1;

% borrowing region
gridaneg = grida.*(grida<0);

% initial durables
dinit = trapint(gridd.*psi1t./ddist,{nodesz,nodesd,nodesy});

% changes in distribution for iterations
CT = distrmat.CTdt;
AendoadjT = distrmat.AendoadjTdt;


% then change in distribution determined by drifts captured in C
% g_{t+dt} - g_{t} = C' g_{t} dt
% we also calculate flow quantities that change over the interval here.
for zz=1:itdist
    
    % adjustments;
    adjendo = adjust.*(AendoadjT*psi1t)./ddist;%adjust.*(AendoadjT*psi1t)./ddist;

    % adjustment intensity
    adj =  adj+ trapint(adjendo,{nodesz,nodesd,nodesy}) + theta*dt/itdist ;

    % depreciation into next period
    depr = depr - delta*dinit*dt/itdist;

    % total fixed cost
%         fctot = fctot + fc*gridp*trapint(gridd.*((adjust.*(Aendoadj'*dt/itz*psi1t))./ddist + theta*dt/itz*psi1t./ddist),{nodesz,nodesd,nodesy}) ;
    fctot = fctot + fc*(  trapint(gridd.*adjendo,{nodesz,nodesd,nodesy}) + theta*dt/itdist*dinit  );
%         fctot = fctot + trapint(fc.*gridd.*(adjust.*(AendoadjT*psi1t)./ddist + (adjust.*psi1t)./ddist + theta*dt.*((1-adjust).*psi1t)./ddist),{nodesz,nodesd,nodesy})  / dt;

    if mod(zz,itdist/itvar)==0
        % operating cost
        octot = octot + nu*dinit*dt/itvar;

        % nondurable consumption
        c = c + trapint(polc.*psi1,{nodesz,nodesd,nodesy})*dt/itvar;
        
        % marginal utility of consumption
        muc = muc + trapint(fmargutility(polc,gridd,par).*psi1,{nodesz,nodesd,nodesy})*dt/itvar;

        % spread
        spreadtot = spreadtot - gridrb*trapint(gridaneg.*psi1,{nodesz,nodesd,nodesy})*dt/itvar;

        % interest income
        yint2 = yint2 + gridr*a00*dt/itvar;
    end

    % change in distribution
    psi1t = psi1t + CT*psi1t;    % need to break down iteration to avoid negative values  

    % normalized
    psi1 = psi1t./ddist;

    % current durable stock
    dnew = trapint(gridd.*psi1,{nodesz,nodesd,nodesy});

    % durable expenditure
    x = x + (dnew-dinit);
    
    % initial durable stock
    dinit = dnew;

    if sum(psi1t < -1e-6) ~= 0
        zz
        min(psi1t)
        find(psi1t==min(psi1t))
%             warning('Distribution has negative values')
    end

    
    if mod(zz,itdist/itvar)==0
        % new asset holdings
        a1 = trapint(grida.*psi1,{nodesz,nodesd,nodesy});
 
        % savings, income, expenditures
        yint = yint + gridr*a1*dt/itvar;
        
    end

end

% annualize flow expenditure
adj = adj/dt;
depr = depr/dt;
fctot = fctot/dt;
octot = octot/dt;
c = c/dt;
muc = muc/dt;
x = x/dt;
spreadtot = spreadtot/dt;
yint = yint/dt;
yint2 = yint2/dt;

% total expenditure
expen = c + octot + x - 1/(1-chi)*depr  + fctot + spreadtot;


% check integration constant
if abs(trapint(psi1t./ddist,{nodesz,nodesd,nodesy})-1)>10^-8
    warning('t+1 Distribution does not integrate to 1')
end
if sum(psi1t < -1e-6) ~= 0
    min(psi0t)
    warning('Distribution has negative values')
%         stop here;
end



%__________________________________________________________________________
%
% Step 5 : calculate quantities
%__________________________________________________________________________ 



% asset holdings (end of period) 
a0 = a00;%trapint(grida.*psi0,{nodesz,nodesd,nodesy});
a = trapint(gridatp1.*psi1,{nodesz,nodesd,nodesy});
% apos = trapint(gridatp1.*(gridatp1>0).*psi1,{nodesz,nodesd,nodesy});

% fraction borrowing
anegind = trapint((gridatp1<0).*psi1,{nodesz,nodesd,nodesy});

% durable holdings
d = trapint(gridd.*psi1,{nodesz,nodesd,nodesy});
% d0 = trapint(gridd.*psi0,{nodesz,nodesd,nodesy});

% post-tax income
y = gridyagg;%*trapint(gridy.*psi1,{nodesz,nodesd,nodesy}); %

% total durable expenditure
x = x - 1/(1-chi)*depr;

% average durable expenditure conditional on adjustment
xadj = x/adj;

% aggregate demand (includes payments of spread)
AD = c + gridp*(x+fctot) + octot + gridg + spreadtot; %

% interest income
% yint = gridr*par.targets.a;

% aggregate income in steady state
yagg = y + gridr*par.Abar/gridtxpr  + gridg;

% aggregate income
yaggnew = y + yint  + gridg;
yaggnew2 = y + yint2  + gridg;

% d/c ratio
dc = d/(c+octot+ spreadtot);

% exog variables
r = gridr;
rb = gridrb;
g = gridg;
p = gridp;

% aggregate productivity
zagg = gridzagg;
ypot = zagg;

steadyvars = {'y','a','a0','d','c','depr','x','r','rb','p','g','fctot','octot','xadj','adj','AD','yagg','yaggnew','yaggnew2','zagg','anegind','dc','spreadtot','muc','ypot'};  
for jj=1:length(steadyvars)
    eval(['irf.' steadyvars{jj} ' = ' steadyvars{jj} ';' ]) 
end

end