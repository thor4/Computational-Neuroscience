% temporal difference learning model
% extension of the rescorla-wagner model
alpha=.25; % learning rate
gamma = 0.9; % discount rate
lambda=1; % reward magnitude (strength)
lambda_duration = 1; % reward duration, length reward is active

trainN = 1000; % number of training examples

time_steps=50; % how long from stimulus to reward

time_representation = zeros(1,time_steps); %representation of current time for stimulus 1. 
time_representation2 = zeros(1,time_steps); %representation of current time for stimulus 2. 
value_prediction = zeros(1,time_steps); % prediction weights for stimulus 1
value_prediction2 = zeros(1,time_steps); % prediction weights for stimulus 2
%keep record of each value for all trials
value_prediction_mat = zeros(trainN,time_steps);
prediction_error_mat = zeros(trainN,time_steps);
for tN = 1:trainN %which trial
    last_prediction=0; %no prediction of reward when starting out
%     if (tN > 500)
%         lambda = 0; % extinction of reward after 500 trials
%     end
    if (tN > 145) % wait to introduce earlier stimulus until after stimulus at timeN=30 converges
        for timeN = 1:time_steps %which time step of current trial

            if (timeN==50) %reward time
                current_reward = lambda;
            else %no reward
                current_reward = 0;
            end

            time_representation=0.*time_representation; %reset to 0
            if timeN>10
                time_representation(timeN)=1; %first stimulus
            end

    %         time_representation2=0.*time_representation2; %reset to 0
    %         if timeN>30
    %             time_representation2(timeN)=1; %second stimulus
    %         end
            % step 1 calculate predicted reward
            current_prediction=sum(value_prediction.*time_representation + value_prediction2.*time_representation); %stimulus 1 and 2 from its previously established convergence values
            % step 2 calculate prediction error
            prediction_error = current_reward+gamma.*current_prediction-last_prediction; %should be the same
            % step 3 calculate weight update
            if (timeN>10) 
                value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error; %update to include both stimuli
            end
    %         if (timeN>30)
    %             value_prediction2(timeN-1)=value_prediction2(timeN-1)+alpha.*prediction_error; %update to include both stimuli
    %         end
            value_prediction_mat(tN,timeN) = last_prediction;
            prediction_error_mat(tN,timeN) = prediction_error;
            last_prediction=current_prediction; % update your last prediction for next timestep
        end
    else % start with timeN=30 converging
        for timeN = 1:time_steps %which time step of current trial

            if (timeN==50) %reward time
                current_reward = lambda;
            else %no reward
                current_reward = 0;
            end

%             time_representation=0.*time_representation; %reset to 0
%             if timeN>10
%                 time_representation(timeN)=1; %first stimulus
%             end

            time_representation2=0.*time_representation2; %reset to 0
            if timeN>30
                time_representation2(timeN)=1; %second stimulus
            end
            % step 1 calculate predicted reward
            current_prediction=sum(value_prediction2.*time_representation2); %stimulus 2 predicted values
            % step 2 calculate prediction error
            prediction_error = current_reward+gamma.*current_prediction-last_prediction; %should be the same
            % step 3 calculate weight update
%             if (timeN>10) 
%                 value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error; %update to include both stimuli
%             end
            if (timeN>30)
                value_prediction2(timeN-1)=value_prediction2(timeN-1)+alpha.*prediction_error; %update to include both stimuli
            end
            value_prediction_mat(tN,timeN) = last_prediction;
            prediction_error_mat(tN,timeN) = prediction_error;
            last_prediction=current_prediction; % update your last prediction for next timestep
        end
    end
end
    