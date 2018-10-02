% Competitive Networks
% initialize random groupings and their spread
c1_center = -.5; c1_dist = .1;
c2_center = 2; c2_dist = .2;
c3_center = 2.5; c3_dist = .2;

cs = [c1_center c1_dist;
    c2_center c2_dist;
    c3_center c3_dist];

% setup network
inpN = 2; % number of input units
outN = 300; % number of output units
weights = randn(inpN,outN).*.1;
alpha = 0.1; % learning rate
trainingN = 10000; % number of training examples
training_hist=[]; % initialize array to hold history of all values
distance_vec = zeros(1,outN); % initialize array to hold history of all distance values

for tN = 1:trainingN
    % get training cluster
    curr_samp = randperm(3,1); % pick random number from 1-3
    
    % determine x,y coordinates
    x_coord = randn(1) .* cs(curr_samp,2) + cs(curr_samp,1);
    y_coord = randn(1) .* cs(curr_samp,2) + cs(curr_samp,1);
    inp = [x_coord y_coord]; % assign input
    training_hist = [training_hist; inp]; %keep track of all that have been tried
    
    % tells how close from input pattern that output unit's weights are
    % goes through each distance one at a time, but for iris can use 
    % @bsxfun instead
    for oN = 1:outN
        distance_vec(oN)=sum(abs(weights(:,oN)-inp'));
    end
    
    % now determine the winning weight vector connecting input to winning
    % output neuron
    winner = find(distance_vec == min(distance_vec));
    
    % now initiate learning for winning neuron by updating associated 
    % weight vector
    weights(:,winner) = weights(:,winner) + alpha.*(inp' - weights(:,winner));
    
    % now update weights for neighboring units to winner (neighborhood
    % function). update all the same for this example
    if winner > 1 % deal with edge case, no wrap-around
        weights(:,winner-1) = weights(:,winner-1) + alpha.*(inp' - weights(:,winner-1));
    end
    if winner < outN
        weights(:,winner+1) = weights(:,winner+1) + alpha.*(inp' - weights(:,winner+1));
    end
    
    % visualize how the weight vectors update according to the learning
    
    figure(1)
    clf
    scatter(training_hist(:,1),training_hist(:,2),'rX');
    hold on
%     scatter(weights(1,:),weights(1,:),'bO'); hold on
%     scatter(weights(2,:),weights(2,:),'gO');
    scatter(weights(1,:),weights(2,:),'bO');
    hold off
    drawnow
    pause(.1)
    % this gives you the main clusters, weights go to them
end