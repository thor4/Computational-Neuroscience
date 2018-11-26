%% Drift diffusion model for change signal task (CST)
% accuracy for the following 4 conditions of the CST:
% high error likelihood go condition
he_go_perf = .94; %accuracy
he_go_rt = 264.9; %response time
% high error likelihood change condition
he_change_perf = .72; %accuracy
he_change_rt = 387.8; %response time
%low error likelihood go condition
le_go_perf = .98; %accuracy
le_go_rt = 220.4; %response time
%low error likelihood change condition
le_change_perf = .90; %accuracy
le_change_rt = 293.8; %response time

%all condition accuracies
cond_perf = [he_go_perf he_change_perf le_go_perf le_change_perf];

% try diff levels of the drift & bias params to see what effect this has on
% accuracy
drift_range = -0.01:0.001:0.01;
bias_range = -0.7:0.1:0.7;

% % zoom in on drift_range(11:20)
% drift_range = 0:.0001:0.009;

squared_distance_matrix = zeros(length(drift_range),length(bias_range));
correlation_matrix = squared_distance_matrix;

tic
for dN=1:length(drift_range)
    for bN=1:length(bias_range)
        
        current_drift = drift_range(dN);
        current_bias = bias_range(bN);
        for condN=1:4
            CST_ddm;
            squared_distance = sum((cond_perf(condN) - probability_upper).^2);
            squared_distance_matrix(condN,dN,bN) = squared_distance;
%             [r,p] = corr(cond_perf(condN)',P_Outcome');
%         correlation_matrix(dN,bN) = r;
        end
    end
end
toc

%visualization
figure(2), clf
imagesc(squared_distance_matrix), colorbar
xlabel('bias'), ylabel('drift')
% figure(2), clf
% imagesc(correlation_matrix), colorbar
%both of these should point to the same area of graph with best parameters 
%lets you see where the minimum is most likely going to be
[x,y] = find((squared_distance_matrix) == min(min(squared_distance_matrix)));
drift_range(x); % best drift param
bias_range(y); % best bias param
