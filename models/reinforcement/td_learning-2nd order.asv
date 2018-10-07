% temporal difference learning model
% extension of the rescorla-wagner model
alpha=.25; % learning rate
lambda=1; % reward magnitude (strength)
gamma = 0.9; % discount factor (rate)
lambda_duration = 1; % reward duration, length reward is active

trainN = 1000; % number of training examples

time_steps=5; % how long from stimulus to reward

time_representation = zeros(1,time_steps); %representation of current time for stimulus 1. vector to keep track of current time-step. which is active and which are inactive. represents current time
time_representation2 = zeros(1,time_steps); %representation of current time for stimulus 2. representation of time relative to onset of each individual stimulus
value_prediction = zeros(1,time_steps); %keeps track of values for each time step. prediction weights
value_prediction2=zeros(1,time_steps); % prediction weights - stimulus 2
%keep record of each value for all trials
value_prediction_mat = zeros(trainN,time_steps);
prediction_error_mat=zeros(trainN,time_steps);
for tN = 1:trainN % which trial
    last_prediction=0; %no prediction of reward when starting out
    for timeN = 1:time_steps % which time step of current trial
        
        if (timeN==45) || (timeN==35) %reward time (two rewards)
            current_reward = lambda;
        else % no reward
            current_reward = 0;
        end
        
        time_representation=0.*time_representation;
        if timeN>10
            time_representation(timeN)=1; %first stimulus
        end
        
        time_representation=0.*time_representation;
        if timeN>30
            time_representation2(timeN)=1; %second stimulus
        end
        % step 1 calculate predicted reward
        current_prediction=sum(value_prediction.*time_representation); %stimulus 1 but need 2: sum of both (stim 1 and 2) for that particular time step
        % step 2 calculate prediction error
        prediction_error = current_reward+gamma.*current_prediction-last_prediction; %should be the same
        % step 3 calculate weight update
        if (timeN>1)
            value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error; %update to include both stimuli
        end
        value_prediction_mat(tN,timeN) = last_prediction;
        prediction_error_mat(tN,timeN) = prediction_error;
        last_prediction=current_prediction; % update your last prediction for next timestep
    end
end
    
    
%     %visualize
%     mesh(prediction_error_mat);
%     %for first trial
%    plot(prediction_error_mat(1,:));
%     plot(prediction_error_mat(100,:)); %100th trial
% this should show the reward backing up to the first & second stimulus,
% ramp to second then back down and ramp again
% plot(value_prediction_mat(150,:),'LineWidth',5);