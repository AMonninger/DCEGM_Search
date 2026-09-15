function distrmat = fdistrmat(par,nodesgrids,policy)

% if isfield(policy,'adjusttp1')==0
%     policy.adjusttp1 = policy.adjust;
% end


%__________________________________________________________________________
%
% Loading structures
%__________________________________________________________________________ 


Nz = par.Nz;
Nd = par.Nd;
Nnodesz = par.Nnodesz;
theta = par.theta;

% time change and iteration steps
dt = par.distdt;
itdist = par.itdist;
% itvar = par.itvar;

% simple solution without nondurables
if Nd==1
    C = policy.A;
    distrmat.C = C;
    distrmat.CTdt = C'*(dt/itdist);
    distrmat.AendoadjTdt = distrmat.CTdt;
    distrmat.polc = policy.c;
    distrmat.adjust = policy.adjust;
    distrmat.CadjinstantT = zeros(size(C));
    return
end

% policies
zprimeadj = policy.zprimeadj;
dprimeadj = policy.dprimeadj;
adjust = policy.adjust;
adjusttp1 = policy.adjust;%policy.adjusttp1;

% transition matrix (no adjustments)
A = policy.A;
if max(abs(sum(A, 2))) > 1e-10
    error('Some rows of A do not sum to 0')
end

% nodes
ddistz = nodesgrids.ddistz;
ddistd = nodesgrids.ddistd;



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

% check if each point is not an adjustment point next instant (same period
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
    return
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

distrmat.C = C;
distrmat.CTdt = C'*(dt/itdist);
distrmat.Cnoadj = Cnoadj;
distrmat.Cadj = Cadj;
distrmat.AendoadjTdt = Aendoadj'*(dt/itdist);
distrmat.adjpoints = adjpoints;
% distribution.weight_adadj = weight_adadj;
% distribution.indexshift = indexshift;
% distribution.indexshift = indexshift;

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
% Step 3: calculate instantaneous adjustment matrix for iteration
%__________________________________________________________________________ 


% set to -1 change in mass at immediate adjustment points
Cadjinstantneg = spdiags(adjust,0,Nnodesz, Nnodesz);

% points with mass to be reallocated instantaneously
[Ainstantrows, Ainstantcols, instantmass] = find(Cadjinstantneg);
mlen = length(instantmass);

% new index to which we move conditional on adjustment
zdinstantindex = zdprimeadjindex(Ainstantrows) + indexshift;

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

distrmat.CadjinstantT = CadjinstantT;

% these elements we need for the iteration and it is more efficient to
% store them here
distrmat.polc = policy.c;
distrmat.adjust = policy.adjust;

end