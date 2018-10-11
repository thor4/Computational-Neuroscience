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
for tN = 145:trainN %which trial
    last_prediction=0; %no prediction of reward when starting out
%     if (tN > 500)
%         lambda = 0; % extinction of reward after 500 trials
%     end
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
        
        time_representation2=0.*time_representation2; %reset to 0
        if timeN>30
            time_representation2(timeN)=1; %second stimulus
        end
        % step 1 calculate predicted reward
        current_prediction=sum(value_prediction.*time_representation + value_prediction2.*time_representation2); %stimulus 1 but need 2: sum of both (stim 1 and 2) for that particular time step
        % step 2 calculate prediction error
        prediction_error = current_reward+gamma.*current_prediction-last_prediction; %should be the same
        % step 3 calculate weight update
        if (timeN>10) 
            value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error; %update to include both stimuli
        end
        if (timeN>30)
            value_prediction2(timeN-1)=value_prediction2(timeN-1)+alpha.*prediction_error; %update to include both stimuli
        end
        value_prediction_mat(tN,timeN) = last_prediction;
        prediction_error_mat(tN,timeN) = prediction_error;
        last_prediction=current_prediction; % update your last prediction for next timestep
    end
end

% plot 1: value function (weights) of 2nd order conditioning
figure(4), clf
plot(value_prediction_mat(10,:),'LineWidth',2); % trial 10 (beginning of learning) 
hold on
plot(value_prediction_mat(166,:),'LineWidth',2); % trial 166 (after learning converges) 
legend('trial 10 (beginning of learning)',...
    'trial 166 (after learning converges)','FontSize',12,'Location','best')
xlabel('time step','FontSize',14), ylabel('weight strength','FontSize',14)
title(sprintf('2nd Order Conditioning Using TD Learning\nlearning rate: %1.2f, discount rate: %1.2f, reward magnitude: %d, reward duration: %d, trials: %d, time steps: %d',alpha,gamma,lambda,lambda_duration,trainN,time_steps),'FontSize',18)
set(gca,'xlim',[1 50])

% % plot 2: value function (weights) of different scenarios
% figure(3), clf
% subplot(311)
% plot(value_prediction_mat_30_10(10,:),'LineWidth',2); % trial 10 (beginning of learning) 
% hold on
% plot(value_prediction_mat_30_10(145,:),'LineWidth',2); % trial 145 (after learning converges- stimulus1) 
% plot(value_prediction_mat_30_10(311,:),'LineWidth',2); % trial 311 (after learning converges- stimulus2) 
% legend('trial 10 (beginning of learning)',...
%     'trial 145 (after learning converges-stim 1)',...
%     'trial 311 (after learning converges-stim 2)','FontSize',12,'Location','best')
% xlabel('time step','FontSize',14), ylabel('weight strength','FontSize',14)
% title('2nd Order Conditioning, Stim 1 at 31s, Stim 2 at 11s','FontSize',16)
% set(gca,'xlim',[1 50])
% subplot(312)
% plot(value_prediction_mat_10(10,:),'LineWidth',2); % trial 10 (beginning of learning) 
% hold on
% plot(value_prediction_mat_10(218,:),'LineWidth',2); % trial 218 (after learning converges) 
% legend('trial 10 (beginning of learning)',...
%     'trial 218 (after learning converges)','FontSize',12,'Location','best')
% xlabel('time step','FontSize',14), ylabel('weight strength','FontSize',14)
% title('1st Order Conditioning, Stimulus at 11s','FontSize',16)
% set(gca,'xlim',[1 50])
% subplot(313)
% plot(value_prediction_mat_30__10(10,:),'LineWidth',2); % trial 10 (beginning of learning) 
% hold on
% plot(value_prediction_mat_30__10(166,:),'LineWidth',2); % trial 166 (after learning converges-stim 1 & 2) 
% legend('trial 10 (beginning of learning)',...
%     'trial 166 (after learning converges-stim 1 & 2)','FontSize',12,'Location','best')
% xlabel('time step','FontSize',14), ylabel('weight strength','FontSize',14)
% title('2nd Order Conditioning, Stim 1 at 11s, Stim 2 at 31s','FontSize',16)
% set(gca,'xlim',[1 50])
