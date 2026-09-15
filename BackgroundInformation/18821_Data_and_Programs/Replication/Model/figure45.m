% This file plots figures 4 and 5: IRF in fixed cost models to an identified 
% monetary policy shock

clear all; close all;

% models
modelset = {'all','house','car'}; 

% parameterizations
specset = {'fcoctheta';'fctheta';'fcoc';'fc'};
legendset = {'Full Model';'w/o Operating/Maintenance Cost';'w/o Match-quality';'w/o MQ & Operating/Maint. Cost'};

% for house and car only plot full model
specsetdetail = specset(1);
legendsetdetail = legendset(1);

% IRFs to plot
plotset = {'RRSI'};

% controls
onlyplot = 0;   % only plot variables using saved output
saveplots = 1;  % save plots
extend = 29;
savext = 'png';

%--------------------------------------------------------------------------
% Step 1: Load empirical IRFs
%--------------------------------------------------------------------------

% load IRF inputs
load('../Data/datairf.mat','datairf','datairfann');    




%--------------------------------------------------------------------------
% Step 2: Load steady state results
%--------------------------------------------------------------------------
if onlyplot~=1
for mm=1:length(modelset)
    model = modelset{mm};

    
for ss=1:length(specset)  
    spec = specset{ss};
    
if strcmp(model,'all')==0 && strcmp(spec,'fcoctheta')==0
    continue
end    
    
    
% saving/loading string
savehandle = [model spec];
    
% loading saved policy function and steady state distribution
load(['Results/' savehandle '.mat'],'par','steady') 

%--------------------------------------------------------------------------
% Step 3: Calculate IRFs for monetary shock
%--------------------------------------------------------------------------
            
maxlen = length(datairf.c1drealrate1.irfmean);

shockvarspos.r = [datairf.c1drealrate1.irfmean(1:maxlen)]./100 ;
if strcmp(model,'all')
    shockvarspos.p = [datairf.c1dlpallhouserel2.irfmean(1:maxlen)]./100  ;
elseif strcmp(model,'car')
    shockvarspos.r = [datairf.c1drealrate1.irfmean(1:maxlen)]./100 ;
    shockvarspos.p = [datairf.c1dlpcarrel4.irfmean(1:maxlen)]./100  ;
elseif strcmp(model,'house')
    shockvarspos.r = [datairf.c1drealrate1.irfmean(1:maxlen)]./100 ;
    shockvarspos.p = [datairf.c1dlphouserel3.irfmean(1:maxlen)]./100   ;
end
shockvarspos.yagg = [datairf.lgdpLP.irfmean(1:maxlen)]./100 ;
shockvarspos.g = 0;

shockvarsneg.r = -shockvarspos.r;
shockvarsneg.yagg = -shockvarspos.yagg;
shockvarsneg.p = -shockvarspos.p;

% impulse: monetary shock
shocks.nodesexo = ones(1,length(shockvarsneg.r));
shocks.exodrift = zeros(1,length(shockvarsneg.r));
shocks.exosigma = zeros(1,length(shockvarsneg.r));


% load jacobians
load(['Results/' model spec 'jacob.mat'],'jacob','jacobif');
steady.yaggnew2= steady.yagg;

% full-information IRF
for var = fieldnames(jacob.yagg)'
    irfdeviation = jacob.r.(var{1})(1:maxlen,1:maxlen)*shockvarspos.r' ...
                        + jacob.yagg.(var{1})(1:maxlen,1:maxlen)*shockvarspos.yagg' ...
                        + jacob.p.(var{1})(1:maxlen,1:maxlen)*shockvarspos.p' ;
    
    irfRRFI{1}.(var{1}) = steady.(var{1}) + irfdeviation;                
    irfRRFI{2}.(var{1}) = steady.(var{1}) - irfdeviation;
end

% sticky-information IRF
for var = fieldnames(jacob.yagg)'
    irfdeviation = jacobif.r.(var{1})(1:maxlen,1:maxlen)*shockvarspos.r' ...
                        + jacobif.yagg.(var{1})(1:maxlen,1:maxlen)*shockvarspos.yagg' ...
                        + jacobif.p.(var{1})(1:maxlen,1:maxlen)*shockvarspos.p' ;

    irfRRSI{1}.(var{1}) = steady.(var{1}) + irfdeviation;                
    irfRRSI{2}.(var{1}) = steady.(var{1}) - irfdeviation;
end

save(['Results/' savehandle '.mat'],'irfRRFI','irfRRSI','-append')


end



end
end


%--------------------------------------------------------------------------
% Step 4: Plot output
%--------------------------------------------------------------------------

count=1;
% irfcolors = {'m-','c-.','g:','r--','m-.'};
% irfcolors = {'-','--','-.','r--','m-.'};
irfcolors = {'-','--','-.',':','r--','m-.'};
legendset = {'Data',legendset{:}};
legendsetdetail = {'Data',legendsetdetail{:}};

% iterating over models
for mm=1:length(modelset)
    
    model = modelset{mm};
    
    % iterating over sets of shocks
    for jj=1:length(plotset)

        % negative and positive IRFs
        for mult = -1 %[-1,1]
            if mult==1
                multsave='pos';
                posneg = 1;
            elseif mult==-1
                multsave='neg';
                posneg = 2;
            end
            

            %--------------------------------------------------------------------------
            % Step 4.1: Plot empirical and model IRFs for overall response
            %--------------------------------------------------------------------------

            if strcmp(model,'all')
                titleset = {'Cumulative Real Durable Expenditure'};

                % empirical IRFs
                eval(['irfmean = mult*transpose(datairf.c1lq' model 'LP.irfmean); '])
                eval(['irfcil = mult*transpose(datairf.c1lq' model 'LP.irfcil); '])
                eval(['irfciu = mult*transpose(datairf.c1lq' model 'LP.irfciu); '])

            

                clear hset



                figure(count)
                xlen = [0:1:length(irfmean)-1]';
                hset = plot(0:1:length(irfmean)-1,irfmean,'k','Linewidth',3);
                hold on
                hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
                set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
                set(hp,'facealpha',0.2)

                % model IRFs
                for kk=1:length(specset)
                    load(['Results/' model specset{kk} '.mat'],['irf' plotset{jj}],'steady')
                    eval(['irf = irf' plotset{jj} '{1.5-mult/2};'])
                    Trans = length(irf.x);
                    hset(1+kk) = plot(0:1:Trans-1,cumsum(((irf.x+irf.fctot)/(steady.x(1)+steady.fctot(1))-1)*100),irfcolors{kk},'Linewidth',3);
                end

                plot(0:1:Trans-1,zeros(1,Trans),'k:','Linewidth',1);
                hold off
                legend(hset,legendset);
                xlabel('Quarters')
                maxlen = length(irf.x);
                xlim([0 maxlen-1])
                if strcmp(model,'all') 
                    ylim([[-20,110].*(mult==-1 | mult==0)+[-110,20].*(mult==1)])
                elseif strcmp(model,'car') 
                    ylim([[-10,50].*(mult==-1 | mult==0)+[-50,10].*(mult==1)])
                elseif strcmp(model,'house') 
                    ylim([[-25,400].*(mult==-1 | mult==0)+[-400,25].*(mult==1)])
                end
                ylabel('%')
                title(titleset{1})
                set(gcf,'color','w');
                set(gca,'FontSize',16,'fontWeight','normal')
                set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')

                if saveplots
                    saveas(gcf,['../Figures/' plotset{jj} model multsave '-sumx.' savext])
                end

                count=count+1;
            end
            %--------------------------------------------------------------------------
            % Step 4.2: extensive margin
            %--------------------------------------------------------------------------


            % empirical IRFs
            if strcmp(model,'car')     
                eval(['irfmean = mult*transpose(datairf.c1lqcarcexextLP.irfmean); '])
                eval(['irfcil = mult*transpose(datairf.c1lqcarcexextLP.irfcil); '])
                eval(['irfciu = mult*transpose(datairf.c1lqcarcexextLP.irfciu); '])
                xlen = [0:1:length(irfmean)-1]';
            elseif strcmp(model,'all')
                eval(['irfmean = NaN(length(irfmean),1); '])
                eval(['irfcil = NaN(length(irfmean),1); '])
                eval(['irfciu = NaN(length(irfmean),1); '])   
                xlen = [0:1:length(irfmean)-1]';
            elseif strcmp(model,'house')
                eval(['irfmean = mult*transpose(datairfann.c1lprobmovebcLP.irfmean*4); '])
                eval(['irfcil = mult*transpose(datairfann.c1lprobmovebcLP.irfcil*4); '])
                eval(['irfciu = mult*transpose(datairfann.c1lprobmovebcLP.irfciu*4); '])   
                xlen = [1.5:4:length(irfmean)*4-2.5]';
                xlen = [0:1:length(irfmean)-1]';
            end

            figure(count)
            hset = plot(xlen,irfmean,'k','Linewidth',3);
            hold on
            hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
            set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
            set(hp,'facealpha',0.2)

            % model IRFs
            tablemax = zeros(length(specset),1);
            for kk=1:length(specsetdetail)
                load(['Results/' model specsetdetail{kk} '.mat'],['irf' plotset{jj}],'steady')
                eval(['irf = irf' plotset{jj} '{1.5-mult/2};'])
                Trans = length(irf.adj);
                if strcmp(model,'house')
                    % aggregate to annual frequency
                    irfbase = (irf.adj/steady.adj(1)-1)*100;
                    irfadj=0;
                    for zz=0:3
                        % time aggregate
                        irfadj = irfadj + [zeros(zz,1);irfbase;zeros(4-zz,1)]/4;        
                    end
                    % cumulate
                    irfadjcum = cumsum(irfadj);
                    % average
                    cumext = zeros(Trans/4+1,1);
                    for zz=1:Trans/4+1
                        cumext(zz)=mean(irfadjcum(1+(zz-1)*4:zz*4,1));
                    end
                else
                    cumext = cumsum(irf.adj/steady.adj(1)-1)*100; 
                end
                xlen = [0:1:length(cumext)-1]';
                hset(1+kk) = plot(xlen,cumext,irfcolors{kk},'Linewidth',3);
                tablemax(kk) = max(cumext);
                tableend(kk) = cumext(end);
            end
            plot(xlen,zeros(1,length(cumext)),'k:','Linewidth',1);
            hold off
            xlim([0 min(maxlen,length(cumext))-1])
            legend(hset,legendsetdetail);
            if strcmp(model,'house')
                xlabel('Years')
            else
                xlabel('Quarters')
            end
            ylabel('%')
            if strcmp(model,'car')   
                ylim([[-10,15].*(mult==-1)+[-15,10].*(mult==1)])
                title('Cumulative Extensive Margin: Cars')
            elseif strcmp(model,'all')
                ylim([[-20,110].*(mult==-1)+[-110,20].*(mult==1)])
                title('Cumulative Extensive Margin: Durables')
            elseif strcmp(model,'house')
                ylim([[-10,25].*(mult==-1)+[-25,10].*(mult==1)])
                title('Cumulative Extensive Margin: Housing')
            end

            set(gcf,'color','w');
            set(gca,'FontSize',16,'fontWeight','normal')
            set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
            if saveplots
                saveas(gcf,['../Figures/' plotset{jj} model  multsave '-sumadj.' savext])
            end
            count=count+1;




            %--------------------------------------------------------------------------
            % Step 4.3: Plot empirical and model IRFs for overall nondurable response
            %--------------------------------------------------------------------------

            if strcmp(model,'all')
            titleset = {'Cumulative Real Nondurable Expenditure'};

            % empirical IRFs
            irfmean = mult*datairf.c1lqnonallLP.irfmean; 
            irfcil = mult*transpose(datairf.c1lqnonallLP.irfcil); 
            irfciu = mult*transpose(datairf.c1lqnonallLP.irfciu); 


            clear hset



            figure(count)
            xlen = [0:1:length(irfmean)-1]';
            hset = plot(0:1:length(irfmean)-1,irfmean,'k','Linewidth',3);
            hold on
            hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
            set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
            set(hp,'facealpha',0.2)

            % model IRFs
            for kk=1:length(specset)
                load(['Results/' model specset{kk}  '.mat'],['irf' plotset{jj}],'steady')
                eval(['irf = irf' plotset{jj} '{1.5-mult/2};'])

                Trans = length(irf.c);
                hset(1+kk) = plot(0:1:Trans-1,cumsum(((irf.c+irf.octot)/(steady.c(1)+steady.octot(1))-1)*100),irfcolors{kk},'Linewidth',3);
            end
            plot(0:1:Trans-1,zeros(1,Trans),'k:','Linewidth',1);
            hold off
            legend(hset,legendset);
            xlabel('Quarters')
            xlim([0 maxlen-1])

            ylabel('%')
            title(titleset{1})
            set(gcf,'color','w');
            set(gca,'FontSize',16,'fontWeight','normal')
            set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
            if saveplots
                saveas(gcf,['../Figures/' plotset{jj} model  multsave '-sumc.' savext])
            end



            count=count+1;
            end

            
        end

    end
end
