function irf=LPannual(irfh,lags,rnorm,ymin,ymax)
% norm scales the monetary shock if desired

%--------------------------------------------------------------------------
% 0. Specification
%--------------------------------------------------------------------------

datstart = ymin;

%--------------------------------------------------------------------------
% 1. Loading data
%--------------------------------------------------------------------------


% Import quarterly romer shock for normalization
num = xlsread('../Data/romershocks_quarterly.xlsx');
sdromer = std(num(:,4));

% Import romer shock
num = xlsread('../Data/romershocks_yearly.xlsx');
romershock = [NaN((1969-ymin),1);num(:,end)./sdromer];

% Import PSID moving data
% num = dlmread('../../Data/housing_uncon_adjust.csv',',',1,0);
num = xlsread('../Data/ts_CT.xlsx');

% Moving probability computed following Bachmann-Cooper
probmovebc = [NaN((1969-ymin),1);num(1:(1997-1969+1),6);NaN((2007-1997),1)];

%--------------------------------------------------------------------------
% 2. Creating Variables
%--------------------------------------------------------------------------

% variables used in regression:
regvarlvl={'probmovebc'};
regvarname={'Moving Probability'};

% if variable is in levels then convert to logs 
for kk=1:length(regvarlvl)
	eval(['l' regvarlvl{kk} ' = log(' regvarlvl{kk} ')*100;'])

	% add log to list 
    eval(['regvars{kk} = ''l' regvarlvl{kk} ''';']);
	
end

% moving probability has quadratic trend
trend = [transpose(1:1:length(probmovebc))./100];
constrendset = ones(length(probmovebc),1);
for cc=1:2
    constrendset = [constrendset,trend.^cc];
end


%--------------------------------------------------------------------------
% 3. Estimate LP
%--------------------------------------------------------------------------

% estimate impulse response functions
for kk=1:length(regvars)
    
    disp(regvars{kk})
    eval(['currvar = ' regvars{kk} ';']) 
       
    % data
    yvar = currvar(1+(datstart-ymin):end);
    rvar = romershock(1+(datstart-ymin):end);
    consvar = constrendset(1+(datstart-ymin):end,:);


    % baseline y and x variables
    xvar = [lagmatrix(rvar,0:1:lags),lagmatrix(yvar,1:lags),consvar];

    % check for which t we have data
    isdata = min(not(isnan([xvar,yvar]))')';
    Yvar = yvar(isdata,:);
    Xvar = xvar(isdata,:);

    % shorten x by the horizon of the IRF => balanced sample
    Xvar = Xvar(1:size(Xvar,1) - irfh,:);


    for aggr=0:1
        if aggr==0
            addletter = '';
            addtxt = '';
        elseif aggr==1
            addletter = 'c1';
            addtxt = 'Cumulative ';
        end        

        % run local projection
        [B,V] = LPreg(Yvar,Xvar,aggr);

        % normalize
        B=B*rnorm;
        V=V*rnorm^2;

        eval(['irf.' addletter  regvars{kk}  'LP.irfmean = transpose(B);'])
        eval(['irf.' addletter  regvars{kk}  'LP.irfvcv = V;'])
        eval(['irf.' addletter  regvars{kk}  'LP.irfcil = transpose(B-tinv(0.975,size(Xvar,1)-size(Xvar,2))*sqrt(diag(V)));'])
        eval(['irf.' addletter  regvars{kk}  'LP.irfciu = transpose(B+tinv(0.975,size(Xvar,1)-size(Xvar,2))*sqrt(diag(V)));'])
        eval(['irf.' addletter  regvars{kk}  'LP.name = ''' [addtxt regvarname{kk}] ''';'])
      
        % Plot for expansionary shock
        eval(['irfmean = -transpose(irf.' addletter  regvars{kk} 'LP.irfmean);'])
        eval(['irfcil = -transpose(irf.' addletter  regvars{kk} 'LP.irfcil);'])
        eval(['irfciu = -transpose(irf.' addletter  regvars{kk} 'LP.irfciu);'])

        figure
        xlen = [0:1:irfh]';
        plot(xlen,irfmean,'k','Linewidth',3)
        hold on
        hp = patch([xlen; xlen(end:-1:1) ; xlen(1)], [irfcil; irfciu(end:-1:1);irfcil(1)],'k');
        set(hp, 'facecolor', [0.5 0.5 0.5], 'edgecolor', 'none');
        set(hp,'facealpha',0.2)
        plot(0:1:length(irfmean)-1,zeros(1,length(irfmean)),'k:','Linewidth',1)
        hold off
        xlim([0 length(irfmean)-1])
        ylabel('%')
        xlabel('Years')
        title( [addtxt regvarname{kk}] )
        set(gcf,'color','w');
        set(gca,'FontSize',16,'fontWeight','normal')
        set(findall(gcf,'type','text'),'FontSize',18,'fontWeight','normal')
        
        saveas(gcf,['../Figures/irfRR-' addletter regvars{kk} '.png'])
    end
end

end