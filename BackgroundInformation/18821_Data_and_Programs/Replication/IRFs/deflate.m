function [q,p,pq,dlq,dlp] = deflate(pq1,pq2,p1,p2,sub)
% creates a combined real series based on nominal quantities pq1 and pq2 and
% price indeces p1 and p2. if sub=1 we subtract series 1 from 2 instead
if nargin==4
   sub = 1; 
end

% total nominal spending
pq = sub*pq1 + pq2;  

% weight of good 1
pq1share = sub* pq1 ./ pq;

% lagged weight
pq1sharelag = lagmatrix(pq1share,1);

% growth rate of nominal spending
dlpq	= log(pq) - lagmatrix(log(pq),1);

% growth rate of the individual price indeces
dlp1	= log(p1) - lagmatrix(log(p1),1); 
dlp2	= log(p2) - lagmatrix(log(p2),1); 

% growth rate of overall price index
dlp	= pq1sharelag.*dlp1 + (1-pq1sharelag).*dlp2;  % growth in broad durable price 

% find location at which data begins
lastnan = find(isnan(dlp)==1, 1, 'last' );

% price level
lp  = [NaN(lastnan-1,1);0;cumsum(dlp(lastnan+1:end))];
p   = exp(lp);

% real growth rate
dlq	= dlpq - dlp;

% quantity level
lq  = [NaN(lastnan-1,1);0;cumsum(dlq(lastnan+1:end))];
q   = exp(lq);


end