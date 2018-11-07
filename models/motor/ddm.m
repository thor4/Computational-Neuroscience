% drift diffusion model
% start with defining parameters to setup diffusion process
% response is positive or negative
starting_point = 0; %initial starting point/bias
% drift_rate = 0.002; % can be positive or negative, positive here, toward 'positive' resp
drift_rate = current_drift; %takes drift from batch script
upper_decision_boundary = 1; %arbitrary, could be .5 or 100, scaling issue
lower_decision_boundary = -1;
noise = 0.05; %std of a norm distrib (gaussian noise)

%don't know how long this will take (since it's random), in contrast with 
% td learning, when we knew exactly how many timesteps it took. so use a
% while loop

current_position=starting_point; %keep track of curr position
noResp=1; %indicates neither boundary has been hit yet, no resp made
upperN = 0; %how often it hits the upper boundary
lowerN=0; %how often it hits the lower boundary

lowerRT = []; %how long it takes to reach lower boundary
upperRT = []; %how long it takes to reach upper boundary
trialN = 10000; %run trialN instances of ddm: 100,000 for step 2 ddm, 10,000 for step 3 batch_ddm

for tN=1:trialN
    time=0; %var to keep track of time
    positions = []; % initialize positions for each trial
    noResp=1; current_position=0; % start at 0 for each trial
    while noResp==1 %haven't hit decision boundary yet

        time = time+1;
        delta_position = drift_rate + randn.*noise; %get change in position
        current_position=current_position+delta_position; % update the position

        %check if the boundaries are hit
        if current_position>=1 %upper decision boundary
            noResp=0; %gets us out of while loop
            upperN = upperN+1;
            upperRT= [upperRT time];
        end

        if current_position<=-1 %lower decision boundary
            noResp=0; %gets us out of while loop
            lowerN = lowerN+1;
            lowerRT= [lowerRT time];
        end

        positions=[positions current_position];

    %     plot(positions)
    %      xlim([0 100]);
    %       ylim([-1 1]);
    %        drawnow;
    %     pause
    end %end while loop, single run through drift diffusion model
end

% %visualize
% figure(1), clf
% histogram(lowerRT,100)
% xlabel('Response time (RT)','FontSize',30), ylabel('# of observations','FontSize',30)
% title('Distribution of RT for Lower Decision Boundary','FontSize',36)
% export_fig hist_lower.pdf % export to pdf
% 
% figure(2), clf
% histogram(upperRT,100)
% xlabel('Response time (RT)','FontSize',30), ylabel('# of observations','FontSize',30)
% title('Distribution of RT for Upper Decision Boundary','FontSize',36)
% export_fig hist_upper.pdf % export to pdf

