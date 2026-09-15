function policy = fpolexpand(par,parexp,nodesgrids,nodesgridsexp,policy0)
% function to expand policy decisions to get a more accurate grid

%--------------------------------------------------------------------------
% Step 1: creating value of not adjusting on existing grid
%--------------------------------------------------------------------------

% time period over which we do not adjust: t to t+dt
dt = par.dtexp;

% time period until next value function is calculcate, t+dttp1
dttp1 = par.irfdt;

% instantaneous utility when not adjusting
u = max(futility(policy0.c,nodesgrids.gridd,par),par.minu);   % bound losses at boundary

% constant in HJB.
if isfield(policy0,'vtp1')
    % linearly interpolate between next periods value function and current
    % value function to determine value at t+dt
    weighttp1 = dt/dttp1;
    vtpdt = (1-weighttp1)*policy0.v + weighttp1*policy0.vtp1;
else
    vtpdt = policy0.v;
end

% utility
% uvec = u + vtp1/dt + par.theta*policy0.vadj;
uvec = u + vtpdt/dt + par.theta*policy0.vadj;

% value of not adjusting
% policy0.vnoadj = 1/(par.rho + par.theta + 1/dt) *(uvec + policy0.A*policy0.v);
policy0.vnoadj = 1/(par.theta + 1/dt) *(uvec - par.rho*policy0.v + policy0.A*policy0.v);

% reshape to create derivatives
vresh = reshape(policy0.v,[par.Nz,par.Nnodesz/par.Nz]);

% derivative
vz = (vresh(2:par.Nz,:)-vresh(1:par.Nz-1,:))./nodesgrids.dz;
policy0.vz = [vz(1,:);vz];

%--------------------------------------------------------------------------
% Step 2: interpolating to larger grid
%--------------------------------------------------------------------------


% interpolating policies
% cint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.c,nodesgrids.totnodesz),'spline','linear');
cbint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.cb,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'linear','linear');
cfint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.cf,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'linear','linear');
% sbint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.sb,nodesgrids.totnodesz),'spline','linear');
% sfint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.sf,nodesgrids.totnodesz),'spline','linear');
% Habint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.Hab,nodesgrids.totnodesz),'spline','linear');
% Hafint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.Haf,nodesgrids.totnodesz),'spline','linear');
adjint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(double(policy0.adjust),nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'linear','linear');
vint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.v,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'spline','linear');
vzint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.vz,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'spline','linear');
vadjint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.vadj,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'spline','linear');
vnoadjint = griddedInterpolant(nodesgrids.nodessetz(nodesgrids.totnodesz>1),reshape(policy0.vnoadj,nodesgrids.totnodesz(nodesgrids.totnodesz>1)),'spline','linear');


% policies on detailed grid
if par.Ny>1 && par.Nd>1 && par.Nexo>1
%     policy.c = cint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
%     policy.sb = sbint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
%     policy.sf = sfint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
%     policy.Hab = Habint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
%     policy.Haf = Hafint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.cb = cbint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.cf = cfint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.v = vint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.vz = vzint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.vadj = vadjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.vnoadj = vnoadjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
    policy.adjust = adjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy,nodesgridsexp.gridexo);
elseif par.Ny>1 && par.Nd>1
%     policy.c = cint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.sb = sbint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.sf = sfint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.Hab = Habint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.Haf = Hafint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.cb = cbint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.cf = cfint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.v = vint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.vz = vzint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.vadj = vadjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.vnoadj = vnoadjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.adjust = adjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
elseif par.Ny>1
%     policy.c = cint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.sb = sbint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.sf = sfint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.Hab = Habint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
%     policy.Haf = Hafint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
    policy.cb = cbint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.cf = cfint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.v = vint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.vz = vzint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.vadj = vadjint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.vnoadj = vnoadjint(nodesgridsexp.gridz,nodesgridsexp.gridy);
    policy.adjust = adjint(nodesgridsexp.gridz,nodesgridsexp.gridy);    
else
%     policy.c = cint(nodesgridsexp.gridz,nodesgridsexp.gridd);
%     policy.sb = sbint(nodesgridsexp.gridz,nodesgridsexp.gridd);
%     policy.sf = sfint(nodesgridsexp.gridz,nodesgridsexp.gridd);
%     policy.Hab = Habint(nodesgridsexp.gridz,nodesgridsexp.gridd);
%     policy.Haf = Hafint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.cb = cbint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.cf = cfint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.v = vint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.vz = vzint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.vadj = vadjint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.vnoadj = vnoadjint(nodesgridsexp.gridz,nodesgridsexp.gridd);
    policy.adjust = adjint(nodesgridsexp.gridz,nodesgridsexp.gridd,nodesgridsexp.gridy);
end
policy.dprimeadjx = policy0.dprimeadjx;

%--------------------------------------------------------------------------
% Step 2: determine value of not adjusting
%--------------------------------------------------------------------------


% Construct B matrix (exgoenous drifts only)
policy = buildBexog(parexp,nodesgridsexp,policy);

% optimal consumption and savings choice
policy = foptc(parexp,nodesgridsexp,policy);
% policy.I0 = (1-(policy.sf>10^-15)) .* (1-(policy.sb<-10^-15)); % points where sf <= 0 & sb >= 0
% Iunique = (policy.sb<-10^-15).*(1-(policy.sf>10^-15)) + (1-(policy.sb<-10^-15)).*(policy.sf>10^-15); % points where (sf <= 0 & sb < 0) U (sf > 0 & sb >= 0). "Unique" because we know here we can easily take sb
%                                                    % over (sf <= 0 & sb < 0) and sf over (sf > 0 & sb >= 0)
% Iboth = (policy.sb<-10^-15).*(policy.sf>10^-15);
% 
% policy.Ib = Iunique.*(policy.sb<-10^-15) + Iboth.*(policy.Hab>policy.Haf); % split the points where (sb < 0 & sf > 0) based on u(cb) + Vbb * sb </> u(cf) + Vbf * sf: if > then backward. Note
%                                        % that the other elements in the HJB are the same between backward and forward
% policy.If = Iunique.*(policy.sf>10^-15) + Iboth.*(policy.Hab<=policy.Haf);

% endogenous drift when not adjusting
policy = buildAendo(parexp,nodesgridsexp,policy);

% new adjustment policy
adjust = policy.adjust;
locinbetw = (adjust<1 & adjust>0);
adjust(locinbetw) = (policy.vadj(locinbetw)>=policy.vnoadj(locinbetw));

% sum(adjust(locinbetw))/sum(locinbetw)

% new "z"
policy.z = max(0,policy.vnoadj-policy.vadj);

% update decision rules for d',z'
policy = foptzd(parexp,nodesgridsexp,policy);

% implied policy choice: adjust
policy.adjust = logical(adjust);

% never optimal to adjust at BC with minimum d stock due to fixed cost
policy.adjust(1:parexp.Nz*parexp.Nd:end) = 0; 

end