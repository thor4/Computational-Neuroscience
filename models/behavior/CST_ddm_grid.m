%% Drift diffusion model for change signal task (CST)
% accuracy for the following 4 conditions of the CST:
% high error likelihood go condition
he_go_perf = .94; he_go_rt = 264.9; %accuracy & response time
% high error likelihood change condition
he_change_perf = .72; he_change_rt = 387.8; %accuracy & response time
% low error likelihood go condition
le_go_perf = .98; le_go_rt = 220.4; %accuracy & response time
% low error likelihood change condition
le_change_perf = .90; le_change_rt = 293.8; %accuracy & response time

%all condition accuracies
cond_perf = [he_go_perf he_change_perf le_go_perf le_change_perf];

% try diff levels of the drift & bias params to see what effect this has on
% accuracy
% drift_range = -0.01:0.001:0.01; %coarse search
% bias_range = -0.7:0.1:0.7; %coarse search
drift_range = 0:0.001:0.005; %granular search

squared_distance_matrix = zeros(length(cond_perf),length(drift_range),length(bias_range));
% correlation_matrix = squared_distance_matrix;

tic
for condN=1:4
    if condN==1
        bias_range = 0.05:0.001:0.15; %granular search
    elseif condN==2
        bias_range = -0.35:0.001:-0.25; %granular search
    elseif condN==3
        bias_range = 0.45:0.001:0.55; %granular search
    else
        bias_range = -0.15:0.001:-0.05; %granular search
    end
    for dN=1:length(drift_range)
        for bN=1:length(bias_range)
            accuracy=[]; %init mat to hold predicted accuracy values
            current_drift = drift_range(dN); %set current drift
            current_bias = bias_range(bN); %set current bias
            for ddmN=1:4 %run ddm model to generate 4 predicted accuracies
                CST_ddm;
                accuracy=[accuracy probability_upper];
            end
            squared_distance = sum((cond_perf(condN) - accuracy).^2);
            squared_distance_matrix(condN,dN,bN) = squared_distance;
    %             [r,p] = corr(cond_perf(condN)',P_Outcome');
    %         correlation_matrix(dN,bN) = r;
        end
    end
end
toc

%% visualization
figure(5), clf
subplot(221), imagesc(-log(squeeze(squared_distance_matrix(1,:,:))))
set(gca,'clim',[10 13]), colorbar
xlabel('bias'), ylabel('drift'), title('high error likelihood go condition')
subplot(222), imagesc(-log(squeeze(squared_distance_matrix(2,:,:))))
set(gca,'clim',[10 13]), colorbar
xlabel('bias'), ylabel('drift'), title('high error likelihood change condition')
subplot(223), imagesc(-log(squeeze(squared_distance_matrix(3,:,:))))
set(gca,'clim',[10 13]), colorbar
xlabel('bias'), ylabel('drift'), title('low error likelihood go condition')
subplot(224), imagesc(-log(squeeze(squared_distance_matrix(4,:,:))))
set(gca,'clim',[10 13]), colorbar
xlabel('bias'), ylabel('drift'), title('low error likelihood change condition')
export_fig param_space_cond_neg_log_transform_granular.png -transparent % no background
export_fig param_space_cond_neg_log_transform_granular.pdf % export to pdf
% figure(2), clf
% imagesc(correlation_matrix), colorbar

x=zeros(1,4); y=zeros(1,4); %init x & y coord of param space per cond
for condN=1:4 %find best params where minimum is most likely to be
    [x(condN),y(condN)] = find((squeeze(squared_distance_matrix(condN,:,:))) == min(min(squeeze(squared_distance_matrix(condN,:,:)))));
end
drift_range(x) % best drift params
% best bias params (cycle through 1-4 per condition's bias_range)
bias_range = 0.05:0.001:0.15; bias_range(y(1)) %condition 1
bias_range = -0.35:0.001:-0.25; bias_range(y(2)) %condition 2
bias_range = 0.45:0.001:0.55; bias_range(y(3)) %condition 3
bias_range = -0.15:0.001:-0.05; bias_range(y(4)) %condition 4

