% This file solves for the IRF to a government spending shock

clear all; close all;

% models
model = 'all'; 

% parameterizations
spec = 'fcoctheta';

savehandle = [model spec];

% load other structures we need for our analyis
load(['Results/' savehandle '.mat'],'steady','par');
steady.ypot=steady.AD;
steady.yaggnew2=steady.yaggnew;

% controls
saveplots = 1;  % save plots
savext = 'eps';

% Periods for which we set output gap to zero
Tgap = 100;

% sticky information model
stinfo=1;

% dates for volcker
volckerstart = 1979.5;
volckerend = 1982.75;1979.25;

%--------------------------------------------------------------------------
% Step 2: Load Romer Shocks and IRF
%--------------------------------------------------------------------------

% load ymin and ymax
names = {'ymin','ymax','gdpnom','gdp','fedfundsrate','pop','qnondurnom','qservhousenom','qservnom','pservhousenom','pservnom','pnondurnom'};
load('../Data/freddownload',names{:},'dv','tyear')
tquarter = dv(:,1)+(dv(:,2)-1)/12;

% Import romer shock
num = xlsread('../Data/romershocks_quarterly.xlsx');
sdromer = std(num(:,end));
romershock = [NaN((1969-ymin)*4,1);num(:,end)./sdromer];
romershock(isnan(romershock))=0;

figure
plot(tquarter,romershock)

% import estimated IRF
load('../Data/datairf.mat','datairf','datairfann','RRnorm')

% implied real rate IRF
realrateIRF = datairf.c1drealrate1.irfmean'/RRnorm;

% implied output 
gdpIRF = datairf.lgdpLP.irfmean'/RRnorm;


irflen = length(realrateIRF);

%--------------------------------------------------------------------------
% Step 3: Construct r* path implied by real rate IRF
%--------------------------------------------------------------------------

% load jacobian
if stinfo==1
    load(['Results/' model spec 'jacob.mat'],'jacobifge');
    jacobge = jacobifge;
else
    load(['Results/' model spec 'jacob.mat'],'jacobge');
end
% lower triangular part of monetary policy - AD matrix
jacobrADtril = tril(jacobge.r.AD,-1);

% compute natural rate forecast for each shock conditional on fixing the
% output gap to zero for T periods
ygapirf = jacobge.r.AD(:,1:irflen)*realrateIRF;

% set of monetary shocks that generate a zero output gap for Tmax periods under a g shock
% implied natural rate of interest:
rstar =  - jacobge.r.AD\ygapirf;

if sum(abs(rstar(1:irflen)+realrateIRF))>10^-10
    error('rstar calculation incorrect')
end

% contribution to rstar from past interest rate changes
rstarendomat = zeros(Tgap,irflen);

for tt=2:irflen
    rstarendomat(tt:end,tt) = -jacobge.r.AD(tt:end,tt:end)\(jacobrADtril(tt:end,1:tt-1)*realrateIRF(1:tt-1,1));
end
rstarendo = diag(rstarendomat);
% 
% % collect impact on term structure of rstar at horizon hh
% for hh = 1:1:hhmax
%     for tt=1:Tgap
%         irfFilter.(shock{1}).rstar(tt,hh) = sum(rstar(tt:min(Tgap,tt+hh-1),1))/hh;
% 
%         irfFilter.(shock{1}).rstarendo(tt,hh) = sum(rstarendo(tt:min(Tgap,tt+hh-1),1))/hh;
%     end
% end
% 
%--------------------------------------------------------------------------
% Step 4: Implied Real Rate Start
%--------------------------------------------------------------------------

% volckerdates
volckerdates = tquarter>=volckerstart & tquarter<=volckerend;

% volckershocks
volckershocks = zeros(size(tquarter));
volckershocks(volckerdates) = romershock(volckerdates);

figure
plot(tquarter,volckershocks)

% implied real rate path
gdppath = zeros(length(tquarter) + length(ygapirf)-1,1);
realratepath = zeros(length(tquarter) + length(realrateIRF)-1,1);
realratepathendo = zeros(length(tquarter) + length(realrateIRF)-1,1);

for tt=1:length(tquarter)
    
    gdppath(tt:tt+length(ygapirf)-1,1) = gdppath(tt:tt+length(ygapirf)-1,1) + ...
                                                  volckershocks(tt) * ygapirf;
    
    realratepath(tt:tt+length(realrateIRF)-1,1) = realratepath(tt:tt+length(realrateIRF)-1,1) + ...
                                                  volckershocks(tt) * realrateIRF;
    
    realratepathendo(tt:tt+length(realrateIRF)-1,1) = realratepathendo(tt:tt+length(realrateIRF)-1,1) + ...
                                                  volckershocks(tt) * rstarendo;
    
end
gdppath = gdppath(1:length(tquarter));
realratepath = realratepath(1:length(tquarter));
realratepathendo = realratepathendo(1:length(tquarter));


figure
plot(tquarter,realratepathendo)

figure
plot(tquarter,realratepath)

figure
plot(tquarter,gdppath)


stop erhe;
%--------------------------------------------------------------------------
% Step 5: Comparison vs data
%--------------------------------------------------------------------------

% normalize housing services to same units
qservhousenom = qservhousenom/1000;

% real quantities per capita
qset = {'nondur','serv','servhouse'};
for kk=1:length(qset)
	eval(['q' qset{kk} 'nom = q' qset{kk} 'nom./ pop;']);
end
for kk=1:length(qset)
	eval(['q' qset{kk} ' = q' qset{kk} 'nom./p' qset{kk} 'nom ;']);
end

cd ../IRFs

% taking rent out of services expenditure
[trash,pservXrentnom,qservXrentnom] = deflate(qservhousenom,qservnom,pservhousenom,pservnom,-1);

% aggregating nondurable goods and services excluding rent
[qnonall,pnonallnom,qnonallnom,trash,dlpnonallnom] = deflate(qnondurnom,qservXrentnom,pnondurnom,pservXrentnom);

cd ../Model

% real federal funds rate in terms of nondurables and durables
nondur4Qinfl = zeros(length(pnonallnom),1);
for kk=1:4
   nondur4Qinfl = nondur4Qinfl + lagmatrix(dlpnonallnom,-kk);
end

% implied nondurable real rate
realratedata = fedfundsrate - nondur4Qinfl*100;

% implied nondurable real rate
realratenoshock = realratedata - realratepath;

% without IS from monetary policy shocks
realratenoISMP = realratedata - realratepathendo;

figure
plot(tquarter, realratedata ,'Linewidth', 3)
hold on
plot(tquarter, realratenoISMP ,'Linewidth', 3)
plot(tquarter, realratenoshock ,'Linewidth', 3)
hold off
legend('Data','Without IS from Mon Shocks','Real Rate Wihtout MP Shocks')
xlim([1980,1984.75])
ylabel('Ex-Post Real Rate, %')
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
if saveplots
    savestr = savehandle;
    export_fig('gcf','-transparent',['Figures/irfsVolcker' savestr '.' savext])
end

figure
plot(tquarter, realratepathendo ,'Linewidth', 3)
hold off
legend('IS from Mon Shocks')
xlim([1980,1986.75])
ylabel('Ex-Post Real Rate, %')
set(gcf,'color','w');
set(gca,'FontSize',16,'fontWeight','normal')
set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
if saveplots
    savestr = savehandle;
    export_fig('gcf','-transparent',['Figures/irfsVolckerendo' savestr '.' savext])
end

disp('1983-84')
mean(realratedata(tquarter>=1983 & tquarter<=1984.75))
mean(realratenoISMP(tquarter>=1983 & tquarter<=1984.75))
mean(realratepathendo(tquarter>=1983 & tquarter<=1984.75))

disp('1983')
mean(realratedata(tquarter>=1983 & tquarter<=1983.75))
mean(realratenoISMP(tquarter>=1983 & tquarter<=1983.75))
mean(realratepathendo(tquarter>=1983 & tquarter<=1983.75))

disp('1983Q1')
mean(realratedata(tquarter==1983 & tquarter<=1983.75))
mean(realratenoISMP(tquarter==1983 & tquarter<=1983.75))
mean(realratepathendo(tquarter==1983 & tquarter<=1983.75))