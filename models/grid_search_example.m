%grid search space
subject_data=[.95 .05 .60 .40];

salience_range = [0:0.1:5]; % entire salience range at intervals of .1
beta_range = [-3:0.25:3]; %gain of softmax function parameter

% update to get better resolution of parameter space plot
% salience_range = [1:0.01:2]; % entire salience range at intervals of .1
% beta_range = [2:0.01:4]; %gain of softmax function parameter

squared_distance_matrix = zeros(length(salience_range),length(beta_range));
correlation_matrix = squared_distance_matrix;

for sN=1:length(salience_range)
    for bN=1:length(beta_range)
        
        salience = salience_range(sN);
        beta = beta_range(bN);
        
        CST_network_grid_search_version;
        
        squared_distance = sum((subject_data - P_Outcome).^2);
        squared_distance_matrix(sN,bN) = squared_distance;
        [r,p] = corr(subject_data',P_Outcome');
        correlation_matrix(sN,bN) = r;
    end
end

%visualization
% imagesc(squared_distance_matrix)
% imagesc(correlation_matrix)
%both of these point to the same area of graph with best parameters (2nd
%quadrant ish)
%lets you see where the minimum is most likely going to be
% [x,y] = find((squared_distance_matrix) == min(min(squared_distance_matrix)));
% salience_range(x); % best salience param
% beta_range(y): % best beta param