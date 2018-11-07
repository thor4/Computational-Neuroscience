% batch ddm
% automatically cycles through drift parameter
% try diff levels of this drift param and see what effect this has on
% choices to go to lower or upper boundary
drift_params = [-0.01:0.001:0.01];
% initialized for checking probability of upper decision boundary
probability_upper = [];
% initialized for checking probability of lower decision boundary
probability_lower = []; 

for driftN = 1:length(drift_params) %which drift param
    current_drift = drift_params(driftN); %assign current drift param
    ddm %run trialN trials
    % save probability of getting upper decision boundary for current drift
    % parameter
    probability_upper=[probability_upper upperN./(upperN+lowerN)];
end

%visualization
figure(3), clf
plot(drift_params,probability_upper,'linew',3)
xlabel('Drift parameter','FontSize',30), ylabel('Probability','FontSize',30)
title('Prob of DDM Reaching Upper Decision Boundary','FontSize',36)
export_fig prob_upper.pdf % export to pdf
% as noise increases, the drift rate gets more impacted and the probability
% of getting an upper decision gets lower
% as drift increases, the probability of getting an upper decision
% approaches 1