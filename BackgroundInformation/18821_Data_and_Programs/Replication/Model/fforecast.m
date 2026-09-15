function fv=fforecast(par,nodesgrids,selection,Cmat,polvec,tf,dtm,dt,dt2)

% forecast length
T  = round(tf/dt);

% separating points into sets
pointsall = intersect(transpose(1:1:par.Nnodesz),selection);

% matrices for forecast iteration
Fmat = speye(par.Nnodesz) + Cmat*dt;
Fmat2 = speye(par.Nnodesz) + Cmat*dt2;

% separating state space:
pointsslow = find(diag(Fmat2)<0.01);
points1 = intersect(pointsslow,selection);
points2 = setdiff(pointsall,points1);%intersect(setdiff(find(diag(Fmat3)<0.01),points1),selection);


% points10 = points1;


% vector of current probabilities
Fprob1 = sparse(points1,1:1:length(points1),ones(length(points1),1),par.Nnodesz,length(points1));
Fprob2 = sparse(points2,1:1:length(points2),ones(length(points2),1),par.Nnodesz,length(points2));


% vector of forecast
fv = NaN(par.Nnodesz,tf/dtm);

for kk=1:1:T 
%     kk
    Fprob1 = Fmat*Fprob1;
    if mod(kk,dt2/dt)==0
        Fprob2 = Fmat2*Fprob2;
    end
    
    
    % integrate at specific intervals
    if mod(kk,dtm/dt)==0
        for nn=1:1:length(points1)
            fv(points1(nn),kk/(dtm/dt)) = trapint(polvec.*Fprob1(:,nn),{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
        end
        for nn=1:1:length(points2)
            fv(points2(nn),kk/(dtm/dt)) = trapint(polvec.*Fprob2(:,nn),{nodesgrids.nodesz,nodesgrids.nodesd,nodesgrids.nodesy});
        end
        
    end
    
    
    
    % reassign points based on new mass
    if mod(kk,dt2/dt)==0 && isempty(points2)==0
        points1new = find(max(Fprob2(pointsslow,:),[],1)>0.001);
        length(points1new);
        points2old = setdiff(1:1:length(points2),points1new);
        points1 = [points1;points2(points1new,1)];
        points2 = points2(points2old);
        Fprob1 = sparse([Fprob1,Fprob2(:,points1new)]);
        Fprob2 = sparse(Fprob2(:,points2old));
    end

    
end

end