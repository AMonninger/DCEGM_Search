function [par,policy0] = fsteadytargets(par,targets,targetpar,convcrit,policy0,expandgrid)
% function to solve for steady state given targets and convergence
% criterion
if nargin<6
   expandgrid=1; 
end
% initial policy structure, if not specified
if nargin==4 || isempty(policy0)
    policy0 = struct;
end

% parmeters to be updated
targetparset = fieldnames(targetpar);
x = zeros(length(targetparset),1);
for kk=1:length(targetparset)
    x(kk,1)    = targetpar.(targetparset{kk});
end

if length(targetparset)>1
    updateval = 0.75;
    boundaryval = 3;1.5;
    minxdelta=min(convcrit/10,10^-5);
else
    updateval = 0.75;
    boundaryval = 3;
    minxdelta=min(convcrit/10,10^-5);
end

% initial change in parameters for derivatives (%)
xdelta = -0.001*ones(length(fieldnames(targetpar)),1);

% matrix for storing differences from target. first column are baseline
% paramteres, subsequent columns are for calculating derivatives
targetset = fieldnames(targets);
fvalset = zeros(length(targetset),length(targetparset)+1);



% run convergence loop
for ss=1:par.maxitpar

    % initial x
    x0 = x;

    % dx = level change in x for derivatives
    dx = x * ones(1,length(x)) .* diag(xdelta) ;
    
    % set of parameter values for which we solve the model
    xset = x * ones(1,length(x)+1) + [zeros(size(x)),dx];

%     parfor jj=1:length(x)+1
    for jj=1:length(x)+1 
       
        % function defined at end of file
        [fvalset(:,jj),policy0set{jj}] = fsolveparallel(par,xset(:,jj),targetparset,targets,policy0,expandgrid); % 
        
        % new initial condition
        if par.steadyinit==1
            policy0 = policy0set{1};
        end

    end
    
    jacob = diag(diag(dx\(fvalset(:,2:end)-fvalset(:,1))));%(dx\(fvalset(:,2:end)-fvalset(:,1)));%
    
    % impose limits on how much we change x in each dimension (up to the deviation we considered in the jacobian)
    xupdatenolim = -jacob\fvalset(:,1);
    
%     % how much we want to update x relative to how the jacobian was
%     % calculated
%     abs(diag(dx))
%     xdevjac = abs(xupdatenolim)./abs(diag(dx))
    
    % impose limits on how much we change x in each dimension (up to the deviation we considered in the jacobian)
    xupdate = max(min(xupdatenolim ,abs(diag(dx))*boundaryval),-abs(diag(dx))*boundaryval); %xupdatenolim*min(boundaryval/xdevjac,1) %
    
    % update derivative for next calculation
    xdelta = (sign(xupdate./x)==1).*max(xupdate./x,minxdelta)+(sign(xupdate./x)==-1).*min(xupdate./x,-minxdelta);
    
    % convergence criterion
    critpar = max(abs(fvalset(:,1)));
    
    

    % adjust parameters  
    x = x + updateval*xupdate;

    % if we hit a FC=1, then use only other two parameters to update two
    % remaining targets
    if max(strcmp(targetparset,'fc'))==1
        fcloc = find(strcmp(targetparset,'fc'));
        
        % FC has to be the minimum of the solution and 1
        x(fcloc) = min(1,x(fcloc));
	
        % need to reduce target dimension when we hit this constraint, so take out probability target
        if x(find(strcmp(targetparset,'fc')))==1
            
            probloc = find(strcmp(targetset,'probadjq'));
            fvalset2 = fvalset([1:probloc-1,probloc+1:end],[1:fcloc,fcloc+2:end]);  % column dim is shifted 1 to the right
            dx2 = dx([1:fcloc-1,fcloc+1:end],[1:fcloc-1,fcloc+1:end]);
            jacob2 = dx2\(fvalset2(:,2:end)-fvalset2(:,1));
            xupdate = ones(size(xupdate))*0.001;
            xupdate([1:fcloc-1,fcloc+1:length(x)]) = max(min(-jacob2\fvalset2(:,1) ,abs(diag(dx2))*3),-abs(diag(dx2))*3);
            xdelta = xupdate./x;
            x = x0 + updateval*xupdate;
            x(fcloc) = min(1,x(fcloc));
            critpar = max(abs(fvalset2(:,1)));

        end
    end
    
    

    [x';xupdate';fvalset(:,1)']
    
    if critpar<convcrit
        break
    end
    
end

% update parameter structure with solution
for kk=1:length(targetparset)
	par.(targetparset{kk}) = x(kk,1);
    par.targetpar.(targetparset{kk}) = x(kk,1);
    if sum(ismember(par.varsetall,targetparset{kk}))>0
        eval(['par.' targetparset{kk} 'min = x(kk);'])
        eval(['par.' targetparset{kk} 'max = x(kk);'])
    end
end

policy0 = policy0set{1};

end

