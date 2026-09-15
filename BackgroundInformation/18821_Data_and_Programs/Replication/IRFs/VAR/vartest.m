% vartest

% length of time series
tsimul = 50000;
burn = 2000;
nlagest = 2;
irfh = 12;

% VAR matrix
dynmat = [0.5,0.1,0.1,0.2;0.2,0.4,-0.1,-0.1];
sigma = [1,0.2;0.2,1];
csigma = chol(sigma)';

% simulate data
nx = size(dynmat,1);
nlag = size(dynmat,2)/nx;
T = nlag+burn+tsimul;
csigma = chol(sigma)';

data = zeros(nx,T);
for tt=nlag+1:T
    datalag = [];
    for ll=0:nlag-1
        datalag = [data(:,tt-nlag+ll); datalag];
    end
    
    data(:,tt) = dynmat*datalag + csigma*randn(2,1);
    
end
data = data(:,burn+nlag+1:end)';
size(data);


VARstruct.nlag = nlagest;
VARstruct.aggr = [0,0];
VARstruct.irfh = irfh;
VARstruct.cons = 1;
VARstruct.choldc = 1;

VARstruct = VARestimate(data,[],VARstruct);


IRF = VARstruct.IRF;
B = VARstruct.B;

% irftest
for mm=1:nx
    data = zeros(nx,irfh+nlag);
    shock = zeros(nx,1);
    shock(mm) = 1;
    data(:,nlag+1) = csigma*shock;
    for tt=nlag+2:irfh+nlag
        datalag = [];
        for ll=0:nlag-1
            datalag = [data(:,tt-nlag+ll); datalag];
        end

        data(:,tt) = dynmat*datalag;

    end

    IRFall(:,:,mm) = data(:,nlag+1:nlag+irfh)';
end

IRF
IRFall