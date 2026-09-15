function muc = margutility(c,d,par)

% parameters;
psid = par.psid;
sigma = par.sigma;
xii = par.xii;

% marginal utility
if xii==1
    muc = (1-psid)* c.^((1-psid)*(1-sigma)-1) .* d.^(psid*(1-sigma) ) ;
else
    
    complementarity = ( (1-psid)^(1/xii) * c.^((xii-1)/xii) ...
                        + psid^(1/xii) * d.^((xii-1)/xii)  ...
                        ).^((1-xii*sigma)/(xii-1));
                    
    muc =  (1-psid)^(1/xii) * c.^(-1/xii) .* complementarity * ( psid^psid*(1-psid)^(1-psid))^(1-sigma);
    
end


end