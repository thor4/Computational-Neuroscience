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
    winner=find(distance_vec==min(distance_vec));
    
    %now calulate diff between activity pattern and 
    weights(:,winner) = weights(:,winner) + alpha.*(inp' - weights(:,winner));
    
    % now update weights for neighboring units to winner
    % update all the same for this example
    if winner>1 %deal with edge case, no wrap-around
        weights(:,winner-1) = weights(:,winner-1) + alpha.*(inp' - weights(:,winner-1));
    end
    if winner<300
        weights(:,winner+1) = weights(:,winner+1) + alpha.*(inp' - weights(:,winner+1));
    end
    
    figure(1)
    clf
    scatter(training_hist(:,1),training_hist(:,2),'rX');
    hold on
    scatter(weights1,:),weights(1,:),'b0');
    hold off
    drawnow
    pause(.001)
    %this gives you the main clusters, weights go to them
    
    scatter(x_coord,y_coord)
    hold on
    drawnow
end


%need to extend this to the 2 dimensional edge case of the iris dataset
% use distance data to identify different regions in the 2-d map of iris
% dataset. distance data is the 4 features differenced from the actual. the
% dividing lines between the species determine the map. need to show region
% "boundaries" and the labels for each region

% for r,g,b randomly start with a number between 0,1 for red, 0,1 for 
% green, 0,1 for blue