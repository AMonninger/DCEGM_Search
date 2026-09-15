function u = futility(c,d,par)

% parameters;
psid = par.psid;
sigma = par.sigma;
xii = par.xii;

% utility
if xii==1
    if sigma==1
        u = (1-psid)*log(c) + psid*log(d);
    else
        u = ( c.^((1-psid)*(1-sigma)) .* d.^(psid*(1-sigma) ) - 1) / (1-sigma);
    end
else
    
    uint = ( (1-psid)^(1/xii) * c.^((xii-1)/xii) + psid^(1/xii) * d.^((xii-1)/xii) ).^(xii/(xii-1)) * ( psid^psid*(1-psid)^(1-psid));
    
    if sigma==1
        u = log( uint );
    else
        u = (uint.^(1-sigma) - 1) / (1-sigma);
    end
end

end