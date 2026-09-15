function nodesgrids = getgrids(par,nodesgrids,varsetall)

%----------------------------------------------------------------------
% 0. Variables for which we construct grids
%----------------------------------------------------------------------

if nargin==2
     % do all the interpolants when unspecified
     varsetall = par.varsetall;
%      varset{end+1}='exo';
end
% varexoset = setdiff(varset,{'a','m','x','exo','ypre'});

%----------------------------------------------------------------------
% 1. Load from structure
%----------------------------------------------------------------------

Nz = par.Nz;
Nx = par.Nx;
Nexo = par.Nexo;
Nnodesz = par.Nnodesz;
Nnodesx = par.Nnodesx;

varsetz  = par.varsetz;
varsetx  = par.varsetx;
totnodesz  = nodesgrids.totnodesz;
totnodesx  = nodesgrids.totnodesx;

%----------------------------------------------------------------------
% 
% 2. Grids of endogenous variables
% 
%----------------------------------------------------------------------

%% Construct grids
for kk=1:length(varsetz)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetz{kk}))==1
    
        if totnodesz(kk)>1
            eval(['grid' varsetz{kk} ' = reshape(repmat(transpose(repmat(nodesgrids.nodes' varsetz{kk} ',prod(totnodesz(kk+1:end)),1)),prod(totnodesz(1:kk-1)),1),[Nnodesz,1]);'])
        else
            eval(['grid' varsetz{kk} ' = nodesgrids.nodes' varsetz{kk} ';'])
        end
        eval(['nodesgrids.grid' varsetz{kk} '   =   grid' varsetz{kk} ';'])
        
    end
end

for kk=1:length(varsetx)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetx{kk}))==1

        if totnodesx(kk)>1
            eval(['grid' varsetx{kk} 'x = reshape(repmat(transpose(repmat(nodesgrids.nodes' varsetx{kk} ',prod(totnodesx(kk+1:end)),1)),prod(totnodesx(1:kk-1)),1),[Nnodesx,1]);'])
        else
            eval(['grid' varsetx{kk} 'x = nodesgrids.nodes' varsetx{kk} ';'])
        end
        eval(['nodesgrids.grid' varsetx{kk} 'x   =   grid' varsetx{kk} 'x;'])

    end
end


% ----------------------------------------------------------------------
% 
% 3. Exogenous grids
% 
% ----------------------------------------------------------------------

if sum(strcmp(varsetall,'exo'))>0 && isfield(nodesgrids,'shockvars')

    shockvars = nodesgrids.shockvars;

    % add next periods durable price
    if isfield(shockvars,'ptp1')
        gridptp1 = nodesgrids.gridp;
        nodesgrids.gridptp1 = gridptp1;
        
        gridptp1x = nodesgrids.gridpx;
        nodesgrids.gridptp1x = gridptp1x;
    end
    if isfield(shockvars,'txprtp1')
        gridtxprtp1 = nodesgrids.gridtxpr;
        nodesgrids.gridtxprtp1 = gridtxprtp1;
        
        gridtxprtp1x = nodesgrids.gridtxprx;
        nodesgrids.gridtxprtp1x = gridtxprtp1x;
    end

    % then update for each field for which I have a coefficient
    for fn=fieldnames(shockvars)'
        
        eval(['grid' fn{1} '= grid' fn{1} ' + shockvars.(fn{1})*gridexo;'])
        eval(['nodesgrids.grid' fn{1} '= grid' fn{1} ';'])

        eval(['grid' fn{1} 'x= grid' fn{1} 'x + shockvars.(fn{1})*gridexox;'])
        eval(['nodesgrids.grid' fn{1} 'x= grid' fn{1} 'x;'])

    end
    
    % these are the drifts and variance of the exogenous aggregate shocks
    if Nexo>1
        exoloc =  find(strcmp(par.varsetz,'exo')==1);

        nodesgrids.exodrift = reshape(repmat(transpose(repmat(nodesgrids.exodrift,prod(totnodesz(exoloc+1:end)),1)),prod(totnodesz(1:exoloc-1)),1),[Nnodesz,1]);
        nodesgrids.exosigma = reshape(repmat(transpose(repmat(nodesgrids.exosigma,prod(totnodesz(exoloc+1:end)),1)),prod(totnodesz(1:exoloc-1)),1),[Nnodesz,1]);

    end
   
    
end


% ----------------------------------------------------------------------
% 
% 3. Tax grids
% 
% ----------------------------------------------------------------------

%% tax nodes
if max(ismember(varsetall,{'r','yagg','g','zagg','txpr'}))==1
    
    if isfield(nodesgrids,'gridtxprtp1')==0
        nodesgrids.gridtxprtp1 = nodesgrids.gridtxpr;
    end
    
    % proportional tax rate
    gridtau = (nodesgrids.gridr * par.Abar/nodesgrids.gridtxpr + par.Abar*(1/nodesgrids.gridtxpr-1/nodesgrids.gridtxprtp1) + nodesgrids.gridg) ./ (nodesgrids.gridyagg.*nodesgrids.gridzagg);
%     gridtau = (par.r * par.Abar/nodesgrids.gridtxpr + par.Abar*(1/nodesgrids.gridtxpr-1/nodesgrids.gridtxprtp1) + nodesgrids.gridg) ./ (nodesgrids.gridyagg.*nodesgrids.gridzagg);
%     disp('update')
    nodesgrids.gridtau = gridtau;

    % add proportional tax
    gridyagg = nodesgrids.gridyagg.*nodesgrids.gridzagg.*(1-gridtau)  ;
    nodesgrids.gridyagg = gridyagg;
    
    % proportional tax rate
    gridtaux = (nodesgrids.gridrx * par.Abar/nodesgrids.gridtxpr  + par.Abar*(1/nodesgrids.gridtxpr-1/nodesgrids.gridtxprtp1) + nodesgrids.gridgx) ./ (nodesgrids.gridyaggx.*nodesgrids.gridzaggx);
%     gridtaux = (par.r * par.Abar/nodesgrids.gridtxpr  + par.Abar*(1/nodesgrids.gridtxpr-1/nodesgrids.gridtxprtp1) + nodesgrids.gridgx) ./ (nodesgrids.gridyaggx.*nodesgrids.gridzaggx);

    % add proportional tax
    gridyaggx = nodesgrids.gridyaggx.*nodesgrids.gridzagg.*(1-gridtaux)  ;
    nodesgrids.gridyaggx = gridyaggx;
    
end

%----------------------------------------------------------------------
% 
% 5. Grids at bottom and top of asset nodes
% 
%----------------------------------------------------------------------

% grids excluding asset nodes
for kk=2:length(varsetz)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetz{kk}))==1
        eval(['len = length(grid' varsetz{kk} ');'])
        if len>1
            eval(['nodesgrids.grid' varsetz{kk} 'resh   =   reshape(grid' varsetz{kk} ',[Nz,Nnodesz/Nz,1]);'])
            eval(['nodesgrids.grid' varsetz{kk} 'minz   =   nodesgrids.grid' varsetz{kk} 'resh(1,:);'])
            eval(['nodesgrids.grid' varsetz{kk} 'maxz   =   nodesgrids.grid' varsetz{kk} 'resh(Nz,:);'])
        else
            eval(['nodesgrids.grid' varsetz{kk} 'resh   =   grid' varsetz{kk} ';'])
            eval(['nodesgrids.grid' varsetz{kk} 'minz   =   grid' varsetz{kk} ';'])
            eval(['nodesgrids.grid' varsetz{kk} 'maxz   =   grid' varsetz{kk} ';'])
        end
    end
end
% grids excluding asset nodes
for kk=2:length(varsetx)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetx{kk}))==1
        eval(['len = length(grid' varsetx{kk} ');'])
        if len>1
            eval(['nodesgrids.grid' varsetx{kk} 'reshx  =   reshape(grid' varsetx{kk} 'x,[Nx,Nnodesx/Nx,1]);'])
            eval(['nodesgrids.grid' varsetx{kk} 'minx   =   nodesgrids.grid' varsetx{kk} 'reshx(1,:);'])
            eval(['nodesgrids.grid' varsetx{kk} 'maxx   =   nodesgrids.grid' varsetx{kk} 'reshx(Nx,:);'])
        else
            eval(['nodesgrids.grid' varsetx{kk} 'reshx  =   grid' varsetx{kk} 'x;'])
            eval(['nodesgrids.grid' varsetx{kk} 'minx   =   grid' varsetx{kk} 'x;'])
            eval(['nodesgrids.grid' varsetx{kk} 'maxx   =   grid' varsetx{kk} 'x;'])
        end
    end
end

% distances between nodes
for kk=1:length(varsetz)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetz{kk}))==1
        if totnodesz(kk)>1
            % forward and backward changes in nodes. Adding an extra node to make is
            % the same length as the nodes, but these extra elements will never be used
            % as the drift at the boundary of the state space points to the interior.
            eval(['nodesgrids.d' varsetz{kk} 'b = reshape(repmat(transpose(repmat([nodesgrids.d' varsetz{kk} '(1);nodesgrids.d' varsetz{kk} '],prod(totnodesz(kk+1:end)),1)),prod(totnodesz(1:kk-1)),1),[Nnodesz,1]);'])
            eval(['nodesgrids.d' varsetz{kk} 'f = reshape(repmat(transpose(repmat([nodesgrids.d' varsetz{kk} ';nodesgrids.d' varsetz{kk} '(end)],prod(totnodesz(kk+1:end)),1)),prod(totnodesz(1:kk-1)),1),[Nnodesz,1]);'])

            % central difference
            eval(['nodesgrids.ddist' varsetz{kk} ' = reshape(repmat(transpose(repmat(0.5*[nodesgrids.d' varsetz{kk} '(1);nodesgrids.d' varsetz{kk} '(2:end)+nodesgrids.d' varsetz{kk} '(1:end-1);nodesgrids.d' varsetz{kk} '(end)],prod(totnodesz(kk+1:end)),1)),prod(totnodesz(1:kk-1)),1),[Nnodesz,1]);'])
        else
            eval(['nodesgrids.d' varsetz{kk} 'b = 1;'])
            eval(['nodesgrids.d' varsetz{kk} 'f = 1;'])
            eval(['nodesgrids.ddist' varsetz{kk} ' = 1;'])
        end
%         eval(['nodesgrids.ddist = nodesgrids.ddist .* nodesgrids.ddist' varsetz{kk} ';'])
    end   
end

for kk=1:length(varsetx)
    % check if we are updating the grid for this variable
    if max(strcmp(varsetall,varsetx{kk}))==1
        if totnodesx(kk)>1
            % forward and backward changes in nodes. Adding an extra node to make is
            % the same length as the nodes, but these extra elements will never be used
            % as the drift at the boundary of the state space points to the interior.
            eval(['nodesgrids.d' varsetx{kk} 'bx = reshape(repmat(transpose(repmat([nodesgrids.d' varsetx{kk} '(1);nodesgrids.d' varsetx{kk} '],prod(totnodesx(kk+1:end)),1)),prod(totnodesx(1:kk-1)),1),[Nnodesx,1]);'])
            eval(['nodesgrids.d' varsetx{kk} 'fx = reshape(repmat(transpose(repmat([nodesgrids.d' varsetx{kk} ';nodesgrids.d' varsetx{kk} '(end)],prod(totnodesx(kk+1:end)),1)),prod(totnodesx(1:kk-1)),1),[Nnodesx,1]);'])

            % central difference
%             eval(['nodesgrids.ddist' varsetz{kk} ' = reshape(repmat(transpose(repmat(0.5*[nodesgrids.d' varsetz{kk} '(1);nodesgrids.d' varsetz{kk} '(2:end)+nodesgrids.d' varsetz{kk} '(1:end-1);nodesgrids.d' varsetz{kk} '(end)],prod(totnodesz(kk+1:end)),1)),prod(totnodesz(1:kk-1)),1),[Nnodesz,1]);'])
        else
            eval(['nodesgrids.d' varsetx{kk} 'bx = 1;'])
            eval(['nodesgrids.d' varsetx{kk} 'fx = 1;'])
%             eval(['nodesgrids.ddist' varsetz{kk} ' = 1;'])
        end
%         eval(['nodesgrids.ddist = nodesgrids.ddist .* nodesgrids.ddist' varsetz{kk} ';'])
    end   
end


%----------------------------------------------------------------------
% 
% 6. Grids for global search in optd
% 
%----------------------------------------------------------------------

% if Nexo>1
%     compset ={'x','y','exo','dtries'};
% else
%     compset ={'x','y','dtries'};
% end
compset ={'x','y','dtries'};

if max(ismember(varsetall,compset))==1  %
    
% candidate d choices
nodesgrids.dtries = repmat(nodesgrids.nodesdtries',prod(nodesgrids.totnodesx),1);

% available cash on hand
nodesgrids.coh = repmat(nodesgrids.gridxx,[1,par.Ndtries]);

% income
nodesgrids.yx = repmat(nodesgrids.gridyx,[1,par.Ndtries]);

nodesgrids.exox = repmat(nodesgrids.gridexox,[1,par.Ndtries]);

% % all exogenous variables for interpolation
% nodesgrids.exosetx = nodesgrids.yx;
% nodesgrids.exoset = nodesgrids.gridy;
% if Nexo>1
%     nodesgrids.exosetx = [nodesgrids.exosetx,nodesgrids.exox];
%     nodesgrids.exoset = [nodesgrids.exoset,nodesgrids.gridexo];
% end

% maximum feasible d
nodesgrids.dtries(nodesgrids.dtries>par.dmax) = NaN;
if min(sum(~isnan(nodesgrids.dtries),2))==0
   warning('no trial solution') 
    stop here;
end

end

% for more grid points, need to update this.
if Nexo>1
    nodesgrids.exox = repmat(nodesgrids.gridexox,[1,par.Ndtries]);
end


%----------------------------------------------------------------------
% 
% 7. Grids for change in durable price
% 
%----------------------------------------------------------------------

if max(ismember(varsetall,{'p'}))==1  %

% only need this field when there are shocks
if isfield(nodesgrids,'shockvars')

    shockvars = nodesgrids.shockvars;

    if isfield(nodesgrids,'gridptp1') 
        gridptp1 = nodesgrids.gridptp1;
        shockp   = shockvars.p;
    else
        gridptp1 = nodesgrids.gridp;
        shockp   = 0;
    end
    
    % change in next period price at current exo 
    gridpdt = (gridptp1-nodesgrids.gridp)/par.irfdt;

    % adding drift of exo at current price
    gridpdt = gridpdt + nodesgrids.exodrift*shockp;

    % normalize to %
    nodesgrids.gridpdt = gridpdt./nodesgrids.gridp ;

else
    nodesgrids.gridptp1 = nodesgrids.gridp;
    nodesgrids.gridpdt = 0;

end


% if isfield(nodesgrids,'nodesptp1')
%     ptp1 = nodesgrids.nodesptp1;
% else
%     ptp1 = nodesgrids.nodesp;
% end
% 
% ploc = find(strcmp(par.varsetz,'p')==1);
% 
% % next periods durable price
% if totnodesz(ploc)>1
%     nodesgrids.gridptp1 = reshape(repmat(transpose(repmat(ptp1,prod(totnodesz(ploc+1:end)),1)),prod(totnodesz(1:ploc-1)),1),[Nnodesz,1]);
% else
%     nodesgrids.gridptp1 = ptp1;
% end
% 
% % change in durable price
% nodesgrids.gridpdt = nodesgrids.gridptp1./nodesgrids.gridp - 1;

end




end