% This file plots figure 7:

clear all; close all;

% models
modelset = {'all'}; 

% parameterizations
specset = {'fcoctheta','nondurify'};
legendset = {'Full Model';'Nondurables Model'};

% shocksize
shocksize = 0.01;

% horizons of interest rate cuts
FGset = [0,4,8];

% length of IRF
Tlen = max(FGset)+5;

%--------------------------------------------------------------------------
% Step 1: Load steady state results
%--------------------------------------------------------------------------

for mm=1:length(modelset)
    model = modelset{mm};

    
for ss=1:length(specset)  
    spec = specset{ss};
    
    
% saving/loading string
savehandle = [model spec];    

% loading steady state distribution
load(['Results/' savehandle '.mat'],'par','steady')

% loading policy function
load(['Results/' savehandle 'jacob.mat'],'jacobge','jacobifge')

%--------------------------------------------------------------------------
% Step 2.1: Intertemporal shifting graph
%--------------------------------------------------------------------------

irfcolorsFG={'b-','b--','b:'};
clear hset legend1

figure
count=1;
for horizon = FGset
    hset(count) = plot(0:1:Tlen-1,-[jacobifge.r.AD(1:Tlen,1+horizon)/steady.yagg(1)*shocksize*100],irfcolorsFG{count},'Linewidth',3);

    hold on
    legend1{count} = [num2str(shocksize*100) '% cut at t=' num2str(horizon) ];
    count=count+1;
end



plot(0:1:Tlen-1,zeros(1,Tlen),'k:','Linewidth',1)
hold off
legend(hset,legend1,'Location','NorthEast');
xlabel('Quarters')
xlim([0 Tlen-1])
ylim([-0.4 0.5])
ylabel('Output, % deviation from steady state')

set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')

export_fig('gcf','-transparent',['../Figures/intertemporalshifting' spec '.eps'])


end
end








  