function c = finvmargutility(muc,d,par,cguess)

% parameters;
psid = par.psid;
sigma = par.sigma;
xii = par.xii;

% invert utility
if xii==1 || psid==0
    c = (muc ./ ( (1-psid)*d.^(psid*(1-sigma)) ) ).^(1/( (1-psid)*(1-sigma)-1 ));

else

    % when xi!=1, we need to solve for c numerically. we use golden section
    % search and these are the input arguments
    a = par.cmin*ones(length(muc),1);    % lower bound on c
    b = 20*ones(length(muc),1);  % upper bound on c
    if nargin==3 || isempty(cguess)
        c0 = (a+b)./2;
    else
        c0 = cguess;
    end

    % function we minimize
    lfunc = @(c) finvmargutilityfunc(c,muc,d,psid,sigma,xii);

    for kk=1:20
       [lf,dlf] = lfunc(c0);
       if max(lf)<10^-6
           break
       end

       % updating c guess -- set update to zero when derivative is small
       update = dlf.\lf;
       update(abs(dlf)<10^-15) = 0;

       c0 = max(c0 - update,a);
    end

    c = min(c0,b);

end