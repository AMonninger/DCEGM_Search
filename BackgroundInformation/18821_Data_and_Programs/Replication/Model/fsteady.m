function distribution = fsteady(par,nodesgrids,policy,distribution)

if par.Nd==1
    if nargin==3
        distribution = fsteadynondur(par,nodesgrids,policy);
    else
        distribution = fsteadynondur(par,nodesgrids,policy,distribution);
    end
    return
end
    
if nargin==3
    psi0t = [];
    dt = 1; 
    policy.adjusttp1 = policy.adjust;
    itz = 1;
else
    psi0t = distribution.psi1t;
    dt = par.irfdt;
    if isfield(policy,'adjusttp1')==0
        policy.adjusttp1 = policy.adjust;
    end
    itz = max(160*dt,40);
end

distribution = struct;

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
% rspread = par.rspread;

% policies
zprimeadj = policy.zprimeadj;
dprimeadj = policy.dprimeadj;
adjust = policy.adjust;
adjusttp1 = policy.adjusttp1;
polc = policy.c;

% transition matrix (no adjustments)
A = policy.A;
if max(abs(sum(A, 2))) > 1e-10
    error('Some rows of A do not sum to 0')
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
gridp = nodesgrids.gridp(1);
grida = nodesgrids.grida;
if isfield(nodesgrids,'gridptp1')
    gridptp1 = nodesgrids.gridptp1(1);
    gridatp1 = gridz - par.lambda*gridptp1*gridd;
else
    gridptp1 = nodesgrids.gridp(1);
    gridatp1 = grida;
end
gridy = nodesgrids.gridy;
gridyagg = nodesgrids.gridyagg(1);
gridzagg = nodesgrids.gridzagg(1);
gridtxpr = nodesgrids.gridtxpr(1);
gridg = nodesgrids.gridg(1);
gridr = nodesgrids.gridr(1);%par.r;
gridrb = min(nodesgrids.gridrb);%0;

% product of distances for trapezoidal integration
ddist = 1;
for kk=1:length(par.varsetz)
    eval(['ddist = ddist .* nodesgrids.ddist' par.varsetz{kk} ';'])
end
distribution.ddist = ddist;

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
% Step 1 : put decisions in index form
%__________________________________________________________________________ 

% adjustment choices in index points
zprimeadjindex0 = nodesgrids.zint(zprimeadj);
dprimeadjindex0 = nodesgrids.dint(dprimeadj);

% create weights on nodes that span the non-integer index we calculated
zprimeadjindex = min(max(floor(zprimeadjindex0),1),Nz-1);
weight_zadj = zprimeadjindex0-zprimeadjindex;

dprimeadjindex = min(max(floor(dprimeadjindex0),1),Nd-1);
weight_dadj = dprimeadjindex0-dprimeadjindex;

% new choice index in Nz*Nd space
zdprimeadjindex = zprimeadjindex + Nz*(dprimeadjindex-1) + Nz*Nd*floor(transpose(0:1:Nnodesz-1)/(Nz*Nd));


% combined weight
indexshift = [0,1,Nz,Nz+1];
weight_adadj(:,1) = (1-weight_zadj).*(1-weight_dadj);
weight_adadj(:,2) = weight_zadj.*(1-weight_dadj);
weight_adadj(:,3) = (1-weight_zadj).*weight_dadj;
weight_adadj(:,4) = weight_zadj.*weight_dadj;


%__________________________________________________________________________
%
% Step 2 : reshuffle mass at adjustment points
%__________________________________________________________________________ 


% subtracting intensity of exgeonous adjustments
Aendoadj = A - spdiags(theta*ones(Nnodesz,1),0,Nnodesz,Nnodesz);

% take vector representation of the sparse matrix A
[Arows, Acols, Avals] = find(Aendoadj);

% set to 0 the columns of A corresponding to endogenous adjustment points
pointslist = (1:length(Aendoadj))';
adjpoints  = pointslist(adjust);
Ak_adjpoints = ismember(Acols, adjpoints); % check if column numbers Acols are in the list of adjustment points

% set to zero mass at endogenous adjustment points
Cnoadjval = Avals;
Cnoadjval(Ak_adjpoints) = 0;

% mass at nonadjustment points
Cnoadj = sparse(Arows, Acols, Cnoadjval, Nnodesz, Nnodesz);

% point mass to be reallocated to adjustment points is the sum of
% endogenous adjustments (A0 - Cnoadj) and exogenous adjustments (theta)
Aadj = A - Cnoadj;

% points with mass to be reallocated
[Aadjrows, Aadjcols, rebmass] = find(Aadj);
  
% new index to which we move conditional on adjustment
zdindex = zdprimeadjindex(Aadjcols) + indexshift;%zdprimeadjindex(reblocs) + indexshift;


% weights for each index node
zdweightadj = weight_adadj(Aadjcols,:);%weight_adadj(reblocs,:);

% check if each point is not an adjustment point next period (same period
% in steady state)
adjpoints  = pointslist(adjust);
adjpointstp1  = pointslist(adjusttp1);
pointscheck = ismember(zdindex, adjpointstp1); % 0 = not an adjustment point, so keep it
pointscheck = 1 - pointscheck;

count_idwpoints = 0; % count number of times inverse distance weights are used to allocate mass to be rebalanced

% if all points are not an adjustment, use the initial set of
% weights. otherwise reshuffle the weight on adjustment points based on distance
if sum(sum(pointscheck,2)<4)>1

    % locations where we have to use inverse distance weights
    idwloc =find((sum(pointscheck,2)<4)==1);

    count_idwpoints = count_idwpoints + length(idwloc);

    % use weights proportional to area opposite the point
    for ii=1:length(idwloc)   

        % new node
        nn = Aadjcols(idwloc(ii));

        % Eucledian distance of each point from (a',d')
        dist =  sqrt([(weight_zadj(nn))^2,(1-weight_zadj(nn))^2,(weight_zadj(nn))^2,(1-weight_zadj(nn))^2]./ddistz(nn)^2 +...
                [(weight_dadj(nn))^2,(weight_dadj(nn))^2,(1-weight_dadj(nn))^2,(1-weight_dadj(nn))^2]./ddistd(nn)^2 );

        % inv distance at nonadj point, converted to weight
        invdist = dist.^-1.*pointscheck(idwloc(ii),:);
        invdist = invdist./sum(invdist);
        
        % if we are exactly at one point adjust weight here
        invdist(dist==0) = 1;
        invdist(isnan(invdist)) = 0;
        
        % weight at nonadjustment points
        weightinv = invdist.*pointscheck(idwloc(ii),:);
        weightinv = weightinv./sum(weightinv);

        % use these weights
        zdweightadj(idwloc(ii),:) = weightinv;

    end
end
if sum(sum(isnan(zdweightadj),2)>0)>0 || sum(abs(sum(zdweightadj,2)-1))>10^-10;
    disp('Weight undefined')
    distribution.noweight=1;
    return
%     isnanloc = find(sum(isnan(zdweightadj),2)>0);
%     length(isnanloc)
%     test = Aadjcols(isnanloc);
%     adjust(zdprimeadjindex(test(1:10)))
%     stop here;
%     temp=[zprimeadj(Aadjcols(isnanloc))-par.lambda*dprimeadj(Aadjcols(isnanloc)),dprimeadj(Aadjcols(isnanloc))];
%     temp2=[grida(Aadjcols(isnanloc)),gridd(Aadjcols(isnanloc)),gridy(Aadjcols(isnanloc))];
%     temp(1,:)
%     temp2(1,:)
% %     [policy.v(1128:1130),policy.vadj(1128:1130),adjust(1128:1130)]
%     stop here
end

mlen = length(rebmass);

% collect indeces and mass for different income levels
mindex = reshape(repmat(transpose(Aadjrows),4,1),4*mlen,1);

zdindexset = reshape(transpose(zdindex),4*mlen,1);

rebmassset = reshape(transpose(zdweightadj .* rebmass),4*mlen,1);

if count_idwpoints>0
%     display(['Number of points where IDWs were used = ', num2str(count_idwpoints)])
end

% add common elements into a sparse matrix
Cadj = accumarray([mindex,zdindexset],rebmassset,[Nnodesz, Nnodesz],[],0,true);

% transition matrix with adjustments
C = Cnoadj + Cadj;

distribution.C = C;
% distribution.Cadj = Cadj;
distribution.Cnoadj = Cnoadj;
distribution.Aendoadj = Aendoadj;

% checking output
if max(abs(sum(C, 2))) > 1e-10
    warning('Some rows of C do not sum to 0')
    stop here;
end

if max(abs(sum(C(:,adjpoints), 1))) > 1e-10
    warning('Some adjustment columns of C do not sum to 0')
%     stop here;
end




%__________________________________________________________________________
%
% Step 3 : compute steady-state distribuition
%__________________________________________________________________________ 
% if there is no initial distribution, then calculate SS distribution
if isempty(psi0t) == 1
    
% change diagonal of C at points in the adjustment region (otherwise C matrix is singular)
CDelta = C + sparse(adjpoints, adjpoints, - 1/dt * ones(length(adjpoints), 1), Nnodesz, Nnodesz);

sumabsCDelta_row = sum(abs(CDelta), 2);
[sumabsCDelta_row_i, ~, ~] = find(sumabsCDelta_row);
if (min(sumabsCDelta_row) < 1e-10) || length(sumabsCDelta_row_i) ~= Nnodesz
    warning('Some row of CDelta is 0')
end

CDeltaT = CDelta';
distribution.CDeltaT = CDeltaT;
clear CDelta C

% Fix one value (not in the adjustment region) so matrix isn't singular:
vec = zeros(Nnodesz,1);
inactpoints = pointslist(~adjust);
rng(10,'twister');
iFix = inactpoints(end);%inactpoints(randi(length(inactpoints),1));
vec(iFix)=.01;
CDeltaT(iFix,:) = [zeros(1,iFix-1),1,zeros(1,Nnodesz-iFix)];


% Solve system:
psi0t = CDeltaT\vec;

psi0_sum = sum(psi0t);
psi0t = psi0t./psi0_sum;
distribution.psi0t = psi0t;
distribution.psi1t = psi0t;



if max(abs(CDeltaT * psi0t)) > 1e-6
%     max(abs(CDeltaT * psi0t))
%     warning('Solution to KFE not accurate')
end
if sum(psi0t < -1e-8) ~= 0
    min(psi0t)
    warning('Solution to KFE has negative values')
end

% convert to distribution
psi0 = psi0t./ddist ;
% psi0adj = psi0 ;
% psi0tadj = psi0t;
% distribution.psi0 = psi0;

% distribution next period
psi1t  = psi0t;
psi1   = psi0;


if abs(trapint(psi0,{nodesz,nodesd,nodesy})-1)>10^-8
    warning('Distribution does not integrate to 1')
end


%__________________________________________________________________________
%
% Step 4 : calculate steady-state statistics: adjustment probabilities and
% hazard functions
%__________________________________________________________________________ 

if par.hazT>0

% try solving forward insteady
hazdt = par.hazdt;
hazT  = par.hazT;
hazmat = speye(Nnodesz) + Cnoadj'*hazdt;

% solving backwards from initial distribution. For adjustment probabilities
% initial distr is the SS. For hazard functions, the initial distr is the
% distribution of those who just adjusted
gfin = zeros(Nnodesz,hazT+1);
gfin(:,1) = psi0t;

gtadj = Cadj'*psi0t./sum(Cadj'*psi0t);
ghazfin = zeros(Nnodesz,hazT+1);
ghazfin(:,1) = gtadj;

% solve ODE backwards (solve both at the same time, so only invert once)
gtemp = [gfin(:,1),ghazfin(:,1)];

for tt=2:1:hazT+1
    gtemp = hazmat * gtemp;
    gfin(:,tt) = gtemp(:,1);
    ghazfin(:,tt) = gtemp(:,2);
end


% adjustment probabilities
for kk=1:1:hazT
    probadj(kk) = -trapint((gfin(:,kk+1)-gfin(:,1))./ddist,{nodesz,nodesd,nodesy});
end
distribution.probadj = probadj;

distribution.probadjy = probadj(1/hazdt);
distribution.probadjq = probadj(1/hazdt /4);
distribution.probadjm = probadj(1/hazdt /12);


if distribution.probadjy>1 || distribution.probadjy<0
    par.hazdt
    warning('Probabilities outside range')
end


% hazard rates
for kk=1:1:hazT
    probhazadj(kk) = -trapint((ghazfin(:,kk+1)-ghazfin(:,1))./ddist,{nodesz,nodesd,nodesy});
end

distribution.probhazadj = probhazadj;
distribution.probhazadjy = probhazadj(1/hazdt:1/hazdt:end);
distribution.probhazadjq = probhazadj(1/hazdt/4:1/hazdt/4:end);
distribution.probhazadjm = probhazadj(1/hazdt/12:1/hazdt/12:end);

distribution.hazadjy = (distribution.probhazadjy-[0,distribution.probhazadjy(1:end-1)])./(1-[0,distribution.probhazadjy(1:end-1)]);
distribution.hazadjq = (distribution.probhazadjq-[0,distribution.probhazadjq(1:end-1)])./(1-[0,distribution.probhazadjq(1:end-1)]);
distribution.hazadjm = (distribution.probhazadjm-[0,distribution.probhazadjm(1:end-1)])./(1-[0,distribution.probhazadjm(1:end-1)]);

end

%__________________________________________________________________________
%
% Step 4 (alternative) : calculate distribution for next period
%__________________________________________________________________________ 
else
    
    % set to -1 change in mass at immediate adjustment points
    Cadjinstantneg = spdiags(adjust,0,Nnodesz, Nnodesz);
    
    % points with mass to be reallocated instantaneously
    [Ainstantrows, Ainstantcols, instantmass] = find(Cadjinstantneg);
    mlen = length(instantmass);
    
    % new index to which we move conditional on adjustment
    zdinstantindex = zdprimeadjindex(Ainstantrows) + indexshift;%zdprimeadjindex(reblocs) + indexshift;

    % weights for each index node
    zdinstantweightadj = weight_adadj(Ainstantrows,:);
    
    % collect indeces and mass for different income levels
    minstantindex = reshape(repmat(transpose(Ainstantrows),4,1),4*mlen,1);

    zdinstantindexset = reshape(transpose(zdinstantindex),4*mlen,1);

    instantmassset = reshape(transpose(zdinstantweightadj .* instantmass),4*mlen,1);

    % add common elements into a sparse matrix
    Cadjinstantpos = accumarray([minstantindex,zdinstantindexset],instantmassset,[Nnodesz, Nnodesz],[],0,true);

    % add the two matrices, should sum to zero in all rows
    CadjinstantT = transpose(Cadjinstantpos - Cadjinstantneg);
    clear Cadjinstantpos Cadjinstantneg;

    if max(abs(sum(CadjinstantT, 1))) > 1e-10
        warning('Some rows of Cadjinstant do not sum to 0')
        stop here;
    end
     
    % instantaneous adjustments
    psi0tadj = psi0t + CadjinstantT*psi0t;
    clear CadjinstantT;

    % initial values for flow calculations
    psi1t = psi0tadj;
    psi1 = psi1t./ddist;
    psi0 = psi0t./ddist;
    adj =  trapint(adjust.*psi0,{nodesz,nodesd,nodesy}) ;
    depr = 0;
    c = 0;
    x = trapint(gridd.*(psi0tadj./ddist-psi0),{nodesz,nodesd,nodesy});
    muc = 0;
    fctot = trapint(fc.*gridd.*adjust.*psi0,{nodesz,nodesd,nodesy})  ;
%     ocmctot = 0;
    octot=0;
    spreadtot = 0;
    yint = 0;
    
%     % check expenditure and savings on instant adjustment
    a0 = trapint(grida.*psi0t./ddist,{nodesz,nodesd,nodesy}) ;
    a1 = trapint(grida.*psi0tadj./ddist,{nodesz,nodesd,nodesy});
    neta = a1-a0;
    expen = fctot + x;
    inc = 0;
    a0 = a1;
    
    % borrowing region
    gridaneg = grida.*(grida<0);
    
    % initial durables
    dinit = trapint(gridd.*psi0tadj./ddist,{nodesz,nodesd,nodesy});

    % changes in distribution for iterations
    CT = C'*dt/itz;
    CadjT = Cadj*dt/itz;
    AendoadjT = Aendoadj'*dt/itz;
    clear C Aendoadj
    
%     disp('update')
%     x = 0;
%     fctot = 0;
    
    % then change in distribution determined by drifts captured in C
    % g_{t+dt} - g_{t} = C' g_{t} dt
    % we also calculate flow quantities that change over the interval here.
    for zz=1:itz
        % current price
%         gridpt = gridp + zz/itz*(gridptp1-gridp);
        
        % borrowing region
%         gridaneg = (gridz - par.lambda*gridp*gridd).*(gridz - par.lambda*gridp*gridd<0);
%         gridaneg = grida.*(grida<0);
        
        % adjsutments;
        adjendo = adjust.*(AendoadjT*psi1t)./ddist;
        
        % adjustment intensity
%         adj =  adj+ trapint((Cadj'*dt/itz*psi1t)./ddist,{nodesz,nodesd,nodesy}) ;
        adj =  adj+ trapint(adjendo,{nodesz,nodesd,nodesy}) + theta*dt/itz ;
        
        % depreciation into next period
        depr = depr - delta*dinit*dt/itz;
 
        % total fixed cost
%         fctot = fctot + fc*gridp*trapint(gridd.*((adjust.*(Aendoadj'*dt/itz*psi1t))./ddist + theta*dt/itz*psi1t./ddist),{nodesz,nodesd,nodesy}) ;
        fctot = fctot + fc*(  trapint(gridd.*adjendo,{nodesz,nodesd,nodesy}) + theta*dt/itz*dinit  );
%         fctot = fctot + trapint(fc.*gridd.*(adjust.*(AendoadjT*psi1t)./ddist + (adjust.*psi1t)./ddist + theta*dt.*((1-adjust).*psi1t)./ddist),{nodesz,nodesd,nodesy})  / dt;
        
        
        % operating cost
        octot = octot + nu*dinit*dt/itz;

        % nondurable consumption
        c = c + trapint(polc.*psi1,{nodesz,nodesd,nodesy})*dt/itz;
        
        % marginal utility of consumption
        muc = muc + trapint(margutility(polc,gridd,par).*psi1,{nodesz,nodesd,nodesy})*dt/itz; 

        % spread
        spreadtot = spreadtot - gridrb*trapint(gridaneg.*psi1,{nodesz,nodesd,nodesy})*dt/itz;
        
        % interest income
        yint = yint + gridr*a1*dt/itz;
        
        % change in distribution
        psi1t = psi1t + CT*psi1t;    % need to break down iteration to avoid negative values  
        
        % normalized
        psi1 = psi1t./ddist;
        
        % current durable stock
        dnew = trapint(gridd.*psi1,{nodesz,nodesd,nodesy});
        
        % durable expenditure
        x = x + (dnew-dinit);
        
        if sum(psi1t < -1e-6) ~= 0
            zz
            min(psi1t)
            find(psi1t==min(psi1t))
%             warning('Distribution has negative values')
        end
        
        % initial durable stock
        dinit = dnew;
        
        % new asset holdings
        a1 = trapint(grida.*psi1,{nodesz,nodesd,nodesy});
        
        % savings, income, expenditures
        neta = neta + a1-a0;
        a0 = a1;
        

        y = gridyagg*gridzagg*trapint(gridy.*psi1,{nodesz,nodesd,nodesy});
        inc = inc + (y+gridr*a0)*dt/itz;
        
    end

    adj = adj/dt;
    depr = depr/dt;
    fctot = fctot/dt;
    octot = octot/dt;
%     ocmctot = ocmctot/dt;
%     octot = ocmctot + chi/(1-chi)*depr;    % we net out maintenance costs, since these are incorporated into durable expenditure
    c = c/dt;
    muc = muc/dt;
    x = x/dt;
    spreadtot = spreadtot/dt;
    yint = yint/dt;
    
    expen =  c + octot + x - 1/(1-chi)*depr  + fctot + spreadtot;
      
    % store new distribution
    distribution.psi1t = psi1t;

    % convert to distribution
    psi0 = psi0t./ddist ;
    psi1 = psi1t./ddist ;
    
    % check integration constant
    if abs(trapint(psi1t./ddist,{nodesz,nodesd,nodesy})-1)>10^-8
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
a0 = trapint(grida.*psi0,{nodesz,nodesd,nodesy});
a = trapint(gridatp1.*psi1,{nodesz,nodesd,nodesy});
apos = trapint(gridatp1.*(gridatp1>0).*psi1,{nodesz,nodesd,nodesy});

% fraction borrowing
anegind = trapint((gridatp1<0).*psi1,{nodesz,nodesd,nodesy});

% durable holdings
d = trapint(gridd.*psi1,{nodesz,nodesd,nodesy});
d0 = trapint(gridd.*psi0,{nodesz,nodesd,nodesy});

% post-tax income
y = gridyagg*gridzagg*trapint(gridy.*psi1,{nodesz,nodesd,nodesy});

% these are the simpler caluclations for the steady state
if itz==1
    
    % adjsutments;
    adjendo = adjust.*(Aendoadj'*psi1t)./ddist;
    
    % adjustment intensity
    % adj = trapint(adjust.*(Aendoadj'*psi0t)./ddist + theta*(psi0t./ddist),{nodesz,nodesd,nodesy});
%     adj = trapint((Cadj'*psi0t)./ddist,{nodesz,nodesd,nodesy});
    adj = trapint(adjendo,{nodesz,nodesd,nodesy}) + theta;

    % nondurable consumption
    c = trapint(polc.*psi1,{nodesz,nodesd,nodesy});
    
    % marginal utility of consumption
    muc = trapint(fmargutility(polc,gridd,par).*psi1,{nodesz,nodesd,nodesy}); 

    % depreciation into next period
    depr = trapint(gridd.*(A'*dt*psi0t)./ddist,{nodesz,nodesd,nodesy}) / dt;
%     depr = - delta*gridp*d;
    
    % total fixed cost
    % fctot = trapint(gridp.*(fc.*adjust.*gridd.*(Aendoadj'*dt*psi0t)./ddist + fc.*gridd.*(adjust.*psi0t)./ddist + fc.*theta*dt.*gridd.*psi0t./ddist),{nodesz,nodesd,nodesy})  / dt;
    fctot = trapint(fc.*gridd.*(adjust.*(A'*dt*((1-adjust).*psi0t))./ddist + (adjust.*psi0t)./ddist + theta*dt.*((1-adjust).*psi0t)./ddist),{nodesz,nodesd,nodesy})  / dt;
%     fctot = fc*gridp*(  trapint(gridd.*adjendo,{nodesz,nodesd,nodesy}) + theta*d  );
    
    % operating  cost
    octot = nu*d;

    % operating net of maintenance cost
%     octot = ocmctot;% + chi/(1-chi)*depr; 
   
    % spread cost
    spreadtot = gridrb*(apos-a);
    
    % durable expenditure
    x = d-d0;
    
    % interest income
    yint = gridr*a;

end

% total durable expenditure
x = x - 1/(1-chi)*depr;
% x = (d-d0)/dt - 1/(1-chi)*depr

% average durable expenditure conditional on adjustment
xadj = x/adj;

% aggregate demand (includes payments of spread)
% disp('update2')
AD = c + gridp*(x+fctot) + octot + gridg + spreadtot; %

% [y,yint,gridg,sum([y,yint,gridg])]
% [nodesgrids.gridyagg,gridr*par.targets.a,gridg,sum([nodesgrids.gridyagg,gridr*par.targets.a,gridg])]

% interest income
yint0 = gridr*par.targets.a/gridtxpr;

% aggregate income
yagg = y + yint0  + gridg;

% aggregate income
yaggnew = y + yint  + gridg;

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

% neta
% expen
% inc

steadyvars = {'y','a','a0','apos','d','c','depr','x','r','rb','p','g','fctot','octot','xadj','adj','AD','yagg','yaggnew','zagg','anegind','dc','spreadtot','muc','ypot'};  
for jj=1:length(steadyvars)
    eval(['distribution.' steadyvars{jj} ' = ' steadyvars{jj} ';' ]) 
end

end