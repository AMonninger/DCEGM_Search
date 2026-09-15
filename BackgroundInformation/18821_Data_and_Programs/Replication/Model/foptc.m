function policy = foptc(par,nodesgrids,policy)

%% Unload parameters
nu = par.nu;
chi = par.chi;
delta = par.delta;

AssumeConcavity = par.AssumeConcavity;

Nz = par.Nz;
Nnodesz = par.Nnodesz;

zmin = par.zmin;
zmax = par.zmax;

par.cmin = 10^-8;
cmin = par.cmin;

dz = nodesgrids.dz;

% grids
gridz = nodesgrids.gridz;
gridd = nodesgrids.gridd;
gridy = nodesgrids.gridy;
gridrspread = nodesgrids.gridrspread;
gridp = nodesgrids.gridp;
gridyagg = nodesgrids.gridyagg;
gridtax = nodesgrids.gridtax;
gridconstrdrift = nodesgrids.gridconstrdrift;

% grids at asset boundary
griddmaxz = nodesgrids.griddmaxz;
gridymaxz = nodesgrids.gridymaxz;
gridrspreadmaxz = nodesgrids.gridrspreadmaxz;
gridpmaxz = nodesgrids.gridpmaxz;
gridyaggmaxz = nodesgrids.gridyaggmaxz;
gridtaxmaxz = nodesgrids.gridtaxmaxz;
griddminz = nodesgrids.griddminz;
gridyminz = nodesgrids.gridyminz;
gridrspreadminz = nodesgrids.gridrspreadminz;
gridpminz = nodesgrids.gridpminz;
gridyaggminz = nodesgrids.gridyaggminz;
gridtaxminz = nodesgrids.gridtaxminz;

gridconstrdriftmaxz = nodesgrids.gridconstrdriftmaxz;
gridconstrdriftminz = nodesgrids.gridconstrdriftminz;

% initial condition
v = policy.v;

if isfield(policy,'cf')==0
    policy.cf = [];
    policy.cb = [];
end

%% Solve for optimal consumption


if isfield(policy,'vz')==0
    % reshape to create derivatives
    vresh = reshape(v,[Nz,Nnodesz/Nz]);

    % derivative
    vz = (vresh(2:Nz,:)-vresh(1:Nz-1,:))./dz;
else
    vz = reshape(policy.vz,[Nz,Nnodesz/Nz]);
    vz = vz(2:end,:);
    policy = rmfield(policy,'vz');
end

% forward difference for the derivative of v wrt b
vzf = NaN(Nz,Nnodesz/Nz);
vzf(1:Nz-1,:) = vz;
vzf(Nz,:)     = fmargutility(gridymaxz.*gridyaggmaxz + gridtaxmaxz -(nu+chi/(1-chi)*delta*gridpmaxz).*griddmaxz + gridrspreadmaxz.*zmax + gridconstrdriftmaxz,griddmaxz,par);  %will never be used, but impose state constraint a<=amax just in case

% backward difference
vzb = NaN(Nz,Nnodesz/Nz);
vzb(2:Nz,:) = vz;
vzb(1,:) = fmargutility(max(gridyminz.*gridyaggminz + gridtaxminz -(nu+chi/(1-chi)*delta*gridpminz).*griddminz + gridrspreadminz*zmin + gridconstrdriftminz,cmin),griddminz,par); %state constraint boundary condition: v'b(1,j,k)_B = u'(z(k) + rb(1) * b(1))


% we impose a very low consumption level when this is negative, so consumer
% will adjust and not reach positions that imply negative consumption at
% the boundary


% this is to avoid dealing with too large numbers in parts where v is
% very steep wrt z  
% this is to avoid dealing with too small numbers in parts where v is
% almost flat wrt z
% vzf(1,:)
vzf = reshape(max(vzf,10^(-4)),[Nnodesz,1]); 
vzb = reshape(max(vzb,10^(-4)),[Nnodesz,1]);

% need to fix this next time
% loc = find(policy.cb>19 & nodesgrids.gridz==0);   
% policy.cb(loc) = cmin;    
% vzf(loc)
% vzb(loc)
% 
% margutility(cmin,griddminz(1),par)
% 
% ctest = invmargutility(vzb,gridd,par,policy.cb);
% 
% stop here;

%consumption and savings with forward difference
cf = finvmargutility(vzf,gridd,par,policy.cf);
sf = gridy.*gridyagg + gridtax -(nu+chi/(1-chi)*delta*gridp).*gridd + gridrspread.*gridz - cf + gridconstrdrift;
sf = reshape(sf,[Nz,Nnodesz/Nz]);
sf(Nz,:) = 0;
sf = reshape(sf,[Nnodesz,1]);
Haf = futility(cf,gridd,par) + vzf.*sf; % hamiltonian component that is a function of a,c. Other components are common, so do not matter in comparison Hb <> Hf


%consumption and savings with backward difference
cb = finvmargutility(vzb,gridd,par,policy.cb);
sb = gridy.*gridyagg + gridtax -(nu+chi/(1-chi)*delta*gridp).*gridd  + gridrspread.*gridz - cb + gridconstrdrift;
sb = reshape(sb,[Nz,Nnodesz/Nz]);
sb(1,:) = 0;
sb = reshape(sb,[Nnodesz,1]);
sb(cb<=1.01*cmin) = 0;
Hab = futility(cb,gridd,par) + vzb.*sb; % hamiltonian component that is a function of a,c



%consumption and derivative of value function at steady state
c0 = max(gridy.*gridyagg + gridtax -(nu+chi/(1-chi)*delta*gridp).*gridd  + gridrspread.*gridz + gridconstrdrift,10^-8); % this is where s=0 i.e. dz = 0


%     % make a choice of forward or backward differences based on the sign of the drift (upwind scheme)
%     Ib = (sb < -10^-15); %negative drift --> backward difference. This is where we assume concavity, because we don't check sf <= 0 as well (see *** reference)
%     If = (sf > 10^-15).*(1-Ib); %positive drift --> forward difference. Note that the (1-Ib) makes sure we take only points where also sb>=0, so Ib and If are 
%                            % disjoint. Concavity of v wrt b would imply that, because with concavity sf <= sb, but this enforces it in a setting
%                            % where there could be some numerical instability
%     I0 = (1-If-Ib); %at steady state. These have to be points where sb>=0 & sf<=0. If we didn't impose disjointness in the previous step, this could = -1 where
%                     % (sb < 0 & sf > 0) and we would need to split these points (see *** reference). Instead we assume concavity i.e. these points don't arise 
if AssumeConcavity == 1 % same as above
    Ib = (sb < -10^-15); %negative drift --> backward difference
    If = (sf > 10^-15).*(1-Ib); %positive drift --> forward difference
    I0 = (1-If-Ib); %at steady state
else % (*** reference)
    I0 = (1-(sf>10^-15)) .* (1-(sb<-10^-15)); % points where sf <= 0 & sb >= 0
    Iunique = (sb<-10^-15).*(1-(sf>10^-15)) + (1-(sb<-10^-15)).*(sf>10^-15); % points where (sf <= 0 & sb < 0) U (sf > 0 & sb >= 0). "Unique" because we know here we can easily take sb
                                                       % over (sf <= 0 & sb < 0) and sf over (sf > 0 & sb >= 0)
    Iboth = (sb<-10^-15).*(sf>10^-15);
    Ib = Iunique.*(sb<-10^-15) + Iboth.*(Hab>Haf); % split the points where (sb < 0 & sf > 0) based on u(cb) + Vbb * sb </> u(cf) + Vbf * sf: if > then backward. Note
                                           % that the other elements in the HJB are the same between backward and forward
    If = Iunique.*(sf>10^-15) + Iboth.*(Hab<=Haf);
end

c = cf.*If + cb.*Ib + c0.*I0;


%% put elements into structure
policy.c  = c;
policy.cf = cf;
policy.cb = cb;
policy.sb = sb;
policy.sf = sf;
policy.Ib = Ib;
policy.If = If;
policy.I0 = I0;
policy.Haf = Haf;
policy.Hab = Hab;

end