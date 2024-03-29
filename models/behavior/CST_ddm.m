% drift diffusion model
% start with defining parameters to setup diffusion process
% response is positive or negative
% starting_point = 0; %initial starting point/bias
starting_point = current_bias; %takes bias from grid search script
% drift_rate = 0.002; % can be positive or negative, positive here, toward 'positive' resp
drift_rate = current_drift; %takes drift from grid search script
upper_decision_boundary = 1; %arbitrary, could be .5 or 100, scaling issue
lower_decision_boundary = -1;
noise = 0.05; %std of a norm distrib (gaussian noise)

noResp=1; %indicates neither boundary has been hit yet, no resp made
upperN=0; lowerN=0; %how often it hits upper & lower boundaries

lowerRT=[]; upperRT=[]; %how long it takes to reach upper & lower boundary
trialN = 10000; %run trialN instances of ddm

for tN=1:trialN
    time=0; %var to keep track of time
%     positions = []; % initialize positions for each trial
    noResp=1; current_position=starting_point; %start at bias for ea trial
    while noResp==1 %haven't hit decision boundary yet

        time = time+1;
        delta_position = drift_rate + randn.*noise; %get change in position
        current_position=current_position+delta_position; % update the position

        %check if the boundaries are hit
        if current_position>=1 %upper decision boundary (correct resp)
            noResp=0; %gets us out of while loop
            upperN = upperN+1;
            upperRT= [upperRT time];
        end

        if current_position<=-1 %lower decision boundary (incorrect resp)
            noResp=0; %gets us out of while loop
            lowerN = lowerN+1;
            lowerRT= [lowerRT time];
        end
%         positions=[positions current_position];

    end %end while loop, single run through drift diffusion model
end

% accuracy is percentage of upper boundary hits
probability_upper=upperN./(upperN+lowerN);

