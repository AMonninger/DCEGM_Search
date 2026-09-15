function [Y,XLAG,ydum,xdum] = VARlags(Data,Exog,nlag,cons,t0,lambda)
% Inputs:
%     data    = matrix of endogenous variables
%     nlags   = number of lags
%     cons    = constant
% 
% Output
%       Y    = LHS adjusted for lags
%       XLAG = lagged endogenous variables

if nargin<=5
    lambda=0;
end

% Number of observations
[nobs,nvar] = size(Data);


% Create the lagged matrix
XLAG = [];
for jj=0:nlag-1
    XLAG = [Data(jj+1:nobs-nlag+jj,:), XLAG];
end

% LHS adjusted for lags
Y = Data(nlag+1:nobs,:);

% add constant
if cons==1
    XLAG = [XLAG ones(nobs-nlag,1)];
elseif cons==2
    trend = t0+transpose(1:1:nobs-nlag);
    XLAG = [XLAG ones(nobs-nlag,1) trend];
elseif cons==3
    trend = t0+transpose(1:1:nobs-nlag);
    XLAG = [XLAG ones(nobs-nlag,1) trend trend.^2];
elseif cons==4
    trend = t0+transpose(1:1:nobs-nlag);
    XLAG = [XLAG ones(nobs-nlag,1) trend trend.^2 trend.^3];    
end

% ------------------Form persistence dummies-------------------
if lambda~=0 
    
    if isempty(Exog)==0 
        xbar=mean(Exog(1:lags,:),1);
    else
        xbar=[];
    end
    if cons>=1
        xbar = [xbar 1];
    end
    % adjust for trend
%     if cons>=2        
%         % average trend in pre-conditions
%         trend = mean(Data(2:nlag+1,:)-Data(1:nlag,:));
%         
%         trendvals = [1-nlag:1:0]'*trend;
%         
%         % we do not incorporate trend for these observations
%         xbar = [xbar 0];
%         if cons==3
%             xbar = [xbar 0];
%         end
%     end
%         xbar = [xbar mean(ones(nobs-nlag,1)) 0];
%     elseif cons==3
%         trend = 1-nlag:1:0;
%         xbar = [xbar mean(ones(nobs-nlag,1)) 0 0];
%     end
    
    ybar=mean(Data(1:nlag,:),1);
   
    xdum=lambda*[repmat(ybar,1,nlag) xbar];

    ydum=zeros(1,nvar);
    ydum(1,:)=lambda*ybar;
    Y=[Y;ydum];
    XLAG=[XLAG;xdum];

end
    