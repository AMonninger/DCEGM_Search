function [f,df] = finvmargutilityfunc(c,muc,d,psid,sigma,xii)

cinv = c.^(-1);
cds = c.^((xii-1)/xii);


dsterm =  (1-psid)^(1/xii) * cds  + psid^(1/xii) * d.^((xii-1)/xii);

func = log(muc) - (1/xii)*log(1-psid) + (1/xii)*log(c)   ...
        - ((1-xii*sigma)/(xii-1))*log( dsterm )...
        - (1-sigma) * (psid*log(psid) + (1-psid)*log(1-psid));

f =  0.5*( func ).^2; 
    
df = cinv.*(  (1/xii)   ...
        - ((1-xii*sigma)/xii)*( (1-psid)^(1/xii) * cds )  ...
        ./dsterm ) ...
        .* func ; 

end