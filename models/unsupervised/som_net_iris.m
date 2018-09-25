% Fisher Iris Data Self Organizing Map
% load iris data
data = csvread('iris_data.csv');
classes = csvread('iris_classes.csv');
% setup network
inpN = 4; % number of input units
outN = 30; % number of output units
weights = randn(inpN,outN).*.1;
alpha = 0.1; % learning rate
trainingN = 10000; % number of training examples
training_hist=[]; % initialize array to hold history of all values
% distance_vec = zeros(1,outN); % initialize array to hold history of all distance values

for tN = 1:trainingN
    % generate random iris training example
    curr_samp = randperm(150,1); % pick random number from 1-4
    inp = data(curr_samp,:); % assign input
    
    training_hist = [training_hist; inp]; %keep track of all that have been tried
    
    % calculate Euclidean distance between input and weight matrix. tells
    % how close input pattern is to each output unit's weight vector
    difference = bsxfun(@minus,inp',weights);
    difference = sum(abs(difference),1);
    
    % now determine the winning output neuron by identifying the most
    % similar unit coordinates
    winner = find(difference == min(difference));
        
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
    
    % visualize how the weights vectors update according to the learning
    if (mod(size(training_hist,1),5)) == 0
        figure(2)
        clf
        scatter(training_hist(:,4),training_hist(:,3),'rX');
        hold on
    %     scatter(weights(1,:),weights(1,:),'bO'); hold on
    %     scatter(weights(2,:),weights(2,:),'gO');
        scatter(weights(4,:),weights(3,:),'bO');
        xlabel('Petal.Width'), ylabel('Petal.Length')
        title('d=30')
        hold off
        drawnow
    end
%     pause(.1)
    
end


%need to extend this to the 2 dimensional edge case of the iris dataset
% use distance data to identify different regions in the 2-d map of iris
% dataset. distance data is the 4 features differenced from the actual. the
% dividing lines between the species determine the map. need to show region
% "boundaries" and the labels for each region