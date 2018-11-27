%% Drift diffusion model for change signal task (CST)
% accuracy for the following 4 conditions of the CST:
% high error likelihood go condition
he_go_perf = .94; %accuracy
he_go_rt = 264.9; %response time
% high error likelihood change condition
he_change_perf = .72; %accuracy
he_change_rt = 387.8; %response time
% low error likelihood go condition
le_go_perf = .98; %accuracy
le_go_rt = 220.4; %response time
% low error likelihood change condition
le_change_perf = .90; %accuracy
le_change_rt = 293.8; %response time

%all condition accuracies
cond_perf = [he_go_perf he_change_perf le_go_perf le_change_perf];

% try diff levels of the drift & bias params to see what effect this has on
% accuracy
% drift_range = -0.01:0.001:0.01;
% bias_range = -0.7:0.1:0.7;
drift_range = 0:0.001:0.005; %granular search

% % zoom in on drift_range(11:20)
% drift_range = 0:.0001:0.009;

squared_distance_matrix = zeros(length(cond_perf),length(drift_range),length(bias_range));
% correlation_matrix = squared_distance_matrix;

tic
for condN=1:4
    if condN==1
        bias_range = 0.05:0.001:0.15;
    elseif condN==2
        bias_range = -0.35:0.001:-0.25;
    elseif condN==3
        bias_range = 0.45:0.001:0.55;
    else
        bias_range = -0.15:0.001:-0.05; 
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
figure(3), clf
subplot(221), imagesc(-log(squeeze(squared_distance_matrix(1,:,:))))
set(gca,'clim',[0 20]), colorbar
xlabel('bias'), ylabel('drift'), title('high error likelihood go condition')
subplot(222), imagesc(-log(squeeze(squared_distance_matrix(2,:,:))))
set(gca,'clim',[0 20]), colorbar
xlabel('bias'), ylabel('drift'), title('high error likelihood change condition')
subplot(223), imagesc(-log(squeeze(squared_distance_matrix(3,:,:))))
set(gca,'clim',[0 20]), colorbar
xlabel('bias'), ylabel('drift'), title('low error likelihood go condition')
subplot(224), imagesc(-log(squeeze(squared_distance_matrix(4,:,:))))
set(gca,'clim',[0 20]), colorbar
xlabel('bias'), ylabel('drift'), title('low error likelihood change condition')
export_fig param_space_cond_neg_log_transform.png -transparent % no background
export_fig param_space_cond_neg_log_transform.pdf % export to pdf
% figure(2), clf
% imagesc(correlation_matrix), colorbar

x=zeros(1,4); y=zeros(1,4); %init x & y coord of param space per cond
for condN=1:4 %find best params where minimum is most likely to be
    [x(condN),y(condN)] = find((squeeze(squared_distance_matrix(condN,:,:))) == min(min(squeeze(squared_distance_matrix(condN,:,:)))));
end
drift_range(x); bias_range(y); % best drift & bias param

%% setup table
Condition = {'high error likelihood go'; 'high error likelihood change'; ...
    'low error likelihood go'; 'low error likelihood change'};
Drift = drift_range(x)'; Bias = bias_range(y)';
T = table(Condition,Drift,Bias);
writetable(T,'condition_params.txt');

%% granular bias_range analysis
[xcond1,ycond1] = find((squeeze(squared_distance_matrix(1,:,:))) == min(min(squeeze(squared_distance_matrix(1,:,:)))));
[xcond1_neglog,ycond1_neglog] = find(((squeeze(squared_distance_matrix(1,:,:))).^.5) == min(min((squeeze(squared_distance_matrix(1,:,:))).^.5)));
[xcond2,ycond2] = find((squeeze(squared_distance_matrix(2,:,:))) == min(min(squeeze(squared_distance_matrix(2,:,:)))));
[xcond3,ycond3] = find((squeeze(squared_distance_matrix(3,:,:))) == min(min(squeeze(squared_distance_matrix(3,:,:)))));
[xcond4,ycond4] = find((squeeze(squared_distance_matrix(4,:,:))) == min(min(squeeze(squared_distance_matrix(4,:,:)))));

%pull out values from sq dist mat
sq_cond1 = diag(squeeze(squared_distance_matrix(1,xcond1,ycond1)));
sq_cond2 = diag(squeeze(squared_distance_matrix(2,xcond2,ycond2)));
sq_cond3 = diag(squeeze(squared_distance_matrix(3,xcond3,ycond3)));
sq_cond4 = diag(squeeze(squared_distance_matrix(4,xcond4,ycond4)));
%confirmed- all minimums from each condition are 0

%identify the best-fit drift and bias parameters
d_cond1 = drift_range(xcond1); b_cond1 = bias_range(ycond1);
d_cond2 = drift_range(xcond2); b_cond2 = bias_range(ycond2);
d_cond3 = drift_range(xcond3); b_cond3 = bias_range(ycond3);
d_cond4 = drift_range(xcond4); b_cond4 = bias_range(ycond4);