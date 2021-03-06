% temporal difference learning model
% extension of the rescorla-wagner model
alpha=.25; % learning rate
gamma = 0.9; % discount rate
lambda=1; % reward magnitude (strength)
lambda_duration = 1; % reward duration, length reward is active

trainN = 1000; % number of training examples

time_steps=50; % how long from stimulus to reward

% vector to keep track of current time-step. which is active and which are 
% inactive. represents current time
time_representation = zeros(1,time_steps); 
% keep track of values for each time step. prediction weights
value_prediction = zeros(1,time_steps); 
% keep record of each value for all trials
value_prediction_mat = zeros(trainN,time_steps);
prediction_error_mat = zeros(trainN,time_steps);

for tN = 1:trainN % which trial
    last_prediction=0; % no prediction of reward when starting out
    if (tN > 500)
        lambda = 0; % extinction of reward after 500 trials
    end
    for timeN = 1:time_steps % which time step of current trial
        if (timeN==50) %reward time
            current_reward = lambda;
        else % no reward
            current_reward = 0;
        end
        time_representation = 0.*time_representation; % reset to 0
        if (timeN>10) % first stimulus
            time_representation(timeN) = 1; 
        end
        time_representation(timeN) = 1; % assign current time a value of 1
        % step 1 calculate predicted reward
        current_prediction = sum(value_prediction.*time_representation);
        % step 2 calculate prediction error, 2nd version diff from slides
        prediction_error = current_reward+gamma.*current_prediction-last_prediction;
        % step 3 calculate weight update
        if (timeN > 1)
            value_prediction(timeN-1)=value_prediction(timeN-1)+alpha.*prediction_error;
        end
        
        last_prediction=current_prediction; % update your last prediction for next timestep
        value_prediction_mat(tN,timeN) = last_prediction;
        prediction_error_mat(tN,timeN) = prediction_error;
    end
end
    
% plot 1: value function (weights)
figure(1), clf
plot(value_prediction_mat(10,:),'LineWidth',2); % trial 10 (beginning of learning) 
hold on
plot(value_prediction_mat(475,:),'LineWidth',2); % trial 475 (after learning converges) 
plot(value_prediction_mat(1000,:),'LineWidth',2); % trial 1000 (after extinction completes) 
xlabel('time step','FontSize',14), ylabel('weight strength','FontSize',14)
legend('trial 10 (beginning of learning)',...
    'trial 475 (after learning converges)','trial 1000 (after extinction completes','FontSize',12,'Location','best')
set(gca,'xlim',[1 50])
title(sprintf('1st Order Conditioning w/ Extinction Using TD Learning\nlearning rate: %1.2f, discount rate: %1.2f, reward magnitude: %d, reward duration: %d, trials: %d, time steps: %d',alpha,gamma,lambda+1,lambda_duration,trainN,time_steps),'FontSize',18)

% plot 2: prediction error
figure(2), clf
subplot(211)
plot(prediction_error_mat(2,:),'LineWidth',2); % trial 2 (first conditioning) 
hold on
plot(prediction_error_mat(475,:),'LineWidth',2); % trial 475 (after learning converges) 
legend('trial 2 (first conditioning)',...
    'trial 475 (after learning converges)','FontSize',12,'Location','best')
xlabel('time step','FontSize',14), ylabel('prediction error','FontSize',14)
set(gca,'xlim',[1 50],'ylim',[0,1])
subplot(212)
plot(prediction_error_mat(501,:),'LineWidth',2); % trial 501 (first extinction) 
hold on
plot(prediction_error_mat(1000,:),'LineWidth',2); % trial 1000 (after extinction completes) 
xlabel('time step','FontSize',14), ylabel('prediction error','FontSize',14)
set(gca,'xlim',[1 50],'ylim',[-1,0])
legend('trial 501 (first extinction)',...
    'trial 1000 (after extinction completes)','FontSize',12,'Location','best')
suptitle(sprintf('1st Order Conditioning w/ Extinction Using TD Learning\nlearning rate: %1.2f, discount rate: %1.2f, reward magnitude: %d, reward duration: %d, trials: %d, time steps: %d',alpha,gamma,lambda+1,lambda_duration,trainN,time_steps))

% mesh(prediction_error_mat);
% xlabel('time step'), ylabel('trial'), zlabel('error')    