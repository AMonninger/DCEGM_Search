clear all; close all;

% models
modelset ={'car'}; 

% parameterizations
specset = {'fcoctheta'};%

% legend
legendset = {'Data','Full Model'};

% plot ci
ciplot = 0; % paper does not report cumultive CI

%--------------------------------------------------------------------------
% Step 1: Load PE Jacobian
%--------------------------------------------------------------------------

for mm=1:length(modelset)
    model = modelset{mm};

    
for ss=1:length(specset)  
    spec = specset{ss};

% saving/loading string
savehandle = [model spec]; 

% loading steady state
load(['Results/' savehandle '.mat'],'par','steady')

% load jacobians
load(['Results/' model spec 'jacob.mat'],'jacobmon','jacobmonif');

%--------------------------------------------------------------------------
% Step 2: Tax shock
%--------------------------------------------------------------------------

Tant = 5;
irfT = size(jacobmon.r.r,1);

taxshock = [zeros(Tant-1,1);-0.01;zeros(irfT-Tant,1)];

% durable adjustments under sticky information
taxadjSI(:,ss) = cumsum(jacobmonif.r.adj*taxshock)./steady.adj*100;

%--------------------------------------------------------------------------
% Step 3: Data
%--------------------------------------------------------------------------

xlen = [-2:1:1]';

% taxdata = zeros(4,1);
taxdata(1,1) = 9.053;
taxdata(2,1)   = 3.695;
taxdata(3,1) = -8.986;
taxdata(4,1) = -4.759;

taxdata = cumsum(taxdata)./12;

% standard errors
stddata(1,1) = 2.039;
stddata(2,1) = 3.784;
stddata(3,1) = 2.391;
stddata(4,1) = 5.045;

% cumulative standard error
stddata = sqrt(cumsum(stddata.^2))./12;
taxdataciu = taxdata + 1.96*stddata;
taxdatacil = taxdata - 1.96*stddata;

%--------------------------------------------------------------------------
% Step 4: Plot against data
%--------------------------------------------------------------------------

end


figure
hset1=plot(xlen,taxdata,'k','Linewidth',3);
hold on
if ciplot==1
hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [taxdatacil; taxdataciu(end:-1:1);taxdatacil(1)],'k');
set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
set(hp,'facealpha',0.2)
end
hset2=plot(-Tant:1:irfT-Tant-1,taxadjSI,'Linewidth',3);
plot(-Tant:1:irfT-Tant-1,zeros(irfT,1),'k:','Linewidth',1)
xlim([-2 1])
ylim([-1 2])
xticks([-2:1:1])
legend([hset1,hset2],legendset)
title('Cumulative Car Purchases')
ylabel('% of steady state')
xlabel('Month Relative to Tax Change')
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')

export_fig('gcf','-transparent',['../Figures/taxshockstickyinfo' spec '.eps'])


end