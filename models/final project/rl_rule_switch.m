% model response time of monkeys in dmts working memory task
% condition 1: rule 1 - correct
% condition 2: rule 1 - incorrect
% condition 3: rule 2 - correct
% condition 4: rule 2 - incorrect
load('m2-rule_resp_rt.mat') %load rule, resp, rt variable 'behavior'
% determine avg RT per condition in actual data

all(behavior(:,1:2)==[1 1],2) %condition 1\

time=[1,605];

r1 = behavior(1:605,[2 3]); %rule 1, pull out cor/inc with rt
idx_cor = r1(:,1)==1;
idx_inc = r1(:,1)==0;

plot(time,r1(idx_cor,2),'b+',time,r1(idx_inc,2),'rx')
set(gca,'ylim',[-.5,1.5])


% delayed match-to-sample td-learning model

alpha=.25; % learning rate
gamma = 0.9; % discount rate
lambda=1; % reward magnitude (strength)
lambda_duration = 1; % reward duration, length reward is active
attn = .5; %strength of attention before experiment, [0,1]
conf = .5; %strength of confidence in underlying rule

trainN = 1000; % number of training examples
max_resp_time_steps = 500; %max rt

%low attn and/or low conf means longer rt
time_steps=((1-attn)+(1-conf))*max_resp_time_steps; % how long from stimulus to response

%vector to keep track of current time-step. which is active and which are 
% inactive. represents current time
time_representation = zeros(1,time_steps); 
% keep track of values for each time step. prediction weights
value_prediction = zeros(1,time_steps); 
% keep record of each value for all trials
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
        time_representation = 0.*time_representation; % reset to 0
        if (timeN>10)
            time_representation(timeN) = 1; %first stimulus
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
    
    
    