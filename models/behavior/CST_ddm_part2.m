% drift diffusion model (part 2 of hw 5&6)
% start with defining parameters to setup diffusion process
% response is correct (upper boundary) or incorrect (lower boundary)
% four conditions:
% condition 1 = high error likelihood go 
% condition 2 = high error likelihood change 
% condition 3 = low error likelihood go
% condition 4 = low error likelihood change
% drift range and bias range for ea condition computed in part 1
drift_range = [0.003 0.002 0.003 0.003];
bias_range = [0.085 -0.281 0.472 -0.094];
upper_decision_boundary = 1; %arbitrary, could be .5 or 100, scaling issue
lower_decision_boundary = -1;
noise = 0.075; %std of a norm distrib (gaussian noise)
subjectN = 10; condN = 4; %total subjects to simulate & total conditions

noResp=1; %indicates neither boundary has been hit yet, no resp made
trialN = 10000; %run trialN instances of ddm
subject_data = zeros(subjectN,2,condN); %all sub data (accuracy and RT) for ea cond

for cN=1:condN
    mean_accuracy=[]; %init vector to contain all subjects' accuracies 
    mean_rt=[]; %init vector to contain all subjects' response times
    for sN=1:subjectN %subject simulation
        upperN=0; lowerN=0; %how often it hits upper & lower boundaries
        lowerRT=[]; upperRT=[]; %how long it takes to reach boundaries
        for tN=1:trialN
            time=0; %var to keep track of time
            noResp=1; current_position=bias_range(cN); %start at bias for ea trial
            while noResp==1 %haven't hit decision boundary yet

                time = time+1;
                delta_position = drift_range(cN) + randn.*noise; %get change in position
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
            end %end while loop, single run through drift diffusion model
        end %run through all trials
        % accuracy= % of upper boundary hits, save ea subject
        mean_accuracy=[mean_accuracy upperN./(upperN+lowerN)];
        mean_rt=[mean_rt mean(upperRT)];
    end
    % save all subjects data for each condition
    subject_data(:,:,cN)=[mean_accuracy' mean_rt'];
end

% mean acc & RT for each condition
mean(subject_data,1)
% standard deviation acc & RT for each condition
std(subject_data,1)