% temporal difference learning model
% extension of the rescorla-wagner model
alpha=.25; % learning rate
lambda=1; % reward magnitude (strength)
gamma = 0.9; % discount factor (rate)
lambda_duration = 1; % reward duration, length reward is active

trainN = 1000; % number of training examples

time_steps=5; % how long from stimulus to reward

time_representation = zeros(1,time_steps); %vector to keep track of current time-step. which is active and which are inactive. represents current time
value_prediction = zeros(1,time_steps); %keeps track of values for each time step. prediction weights
%keep record of each value for all trials
value_prediction_mat = zeros(trainN,time_steps);
prediction_error_mat=zeros(trainN,time_steps);
for tN = 1:trainN % which trial
    last_prediction=0; %no prediction of reward when starting out
    for timeN = 1:time_steps % which time step of current trial
       
        if (timeN==5) %reward time
            current_reward = lambda;
        else % no reward
            current_reward = 0;
        end
        time_representation=0.*time_representation;
        time_representation(timeN)=1;
        % step 1 calculate predicted reward
        current_prediction=sum(value_prediction.*time_representation);
        % step 2 calculate prediction error
        prediction_error = current_reward+gamma.*current_prediction-last_prediction;
        % step 3 calculate weight update
        if (timeN>1)
            value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error;
        end
        
        last_prediction=current_prediction; % update your last prediction for next timestep
        value_prediction_mat(tN,timeN) = last_prediction;
        prediction_error_mat(tN,timeN) = prediction_error;
    end
end
    
    
    