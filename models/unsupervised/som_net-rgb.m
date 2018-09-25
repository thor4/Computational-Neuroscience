% RGB Self Organizing Map
% setup network
dim = 20; % dimensions of the 2d map
net = rand(dim,dim,3); % setup 2d map of dim x dim units
figure(1)
imshow(net,'InitialMagnification','fit'); % visualize to ensure random rgb image
title('Random RGB');
alpha = 0.1; % learning rate
trainingN = 10000; % number of training examples
% training_hist=[]; % initialize array to hold history of all values
% distance_vec = zeros(1,outN); % initialize array to hold history of all distance values

for tN = 1:trainingN
    % generate random rgb training example
    inp = rand(1,1,3);
    
%     training_hist = [training_hist; inp]; %keep track of all that have been tried
    
    % calculate Euclidean distance between input and weight matrix. tells
    % how close input pattern is to each output unit's weight vector
    difference = bsxfun(@minus,inp,net);
    difference = sum(abs(difference),3);
    
    % now determine the winning output neuron by identifying the most
    % similar unit coordinates
    [x y] = find(difference == min(min(difference)));
    
    % now initiate learning for winning neuron by updating 2d map
    net(x,y,:) = net(x,y,:) + alpha.*(inp - net(x,y,:));
    
    % now update weights for neighboring units to winner (neighborhood
    % function). update all neighbors with 80% of alpha. need to deal with
    % edge cases of 2d map:
    disc = 0.80; % neighborhood discount
    % Top Left Corner
    if (x == 1) && (y == 1)
        net(x,y+1,:) = net(x,y+1,:) + disc.*alpha.*(inp - net(x,y+1,:)); %below
        net(x+1,y,:) = net(x+1,y,:) + disc.*alpha.*(inp - net(x+1,y,:)); %rt
        net(x+1,y+1,:) = net(x+1,y+1,:) + disc.*alpha.*(inp - net(x+1,y+1,:)); %diag rt dn
    % Top Edge
    elseif (x == 1) && (y > 1) && (y < dim)
        net(x,y-1,:) = net(x,y-1,:) + disc.*alpha.*(inp - net(x,y-1,:)); %lt
        net(x+1,y-1,:) = net(x+1,y-1,:) + disc.*alpha.*(inp - net(x+1,y-1,:)); %diag lt dn
        net(x+1,y,:) = net(x+1,y,:) + disc.*alpha.*(inp - net(x+1,y,:)); %below
        net(x,y+1,:) = net(x,y+1,:) + disc.*alpha.*(inp - net(x,y+1,:)); %rt
        net(x+1,y+1,:) = net(x+1,y+1,:) + disc.*alpha.*(inp - net(x+1,y+1,:)); %diag rt dn
    % Top Right Corner
    elseif (x == 1) && (y == dim)
        net(x,y-1,:) = net(x,y-1,:) + disc.*alpha.*(inp - net(x,y-1,:)); %lt
        net(x+1,y-1,:) = net(x+1,y-1,:) + disc.*alpha.*(inp - net(x+1,y-1,:)); %diag lt dn
        net(x+1,y,:) = net(x+1,y,:) + disc.*alpha.*(inp - net(x+1,y,:)); %below
    % Left Edge
    elseif (x > 1) && (x < dim) && (y == 1)
        net(x-1,y,:) = net(x-1,y,:) + disc.*alpha.*(inp - net(x-1,y,:)); %above
        net(x-1,y+1,:) = net(x-1,y+1,:) + disc.*alpha.*(inp - net(x-1,y+1,:)); %diag rt up
        net(x,y+1,:) = net(x,y+1,:) + disc.*alpha.*(inp - net(x,y+1,:)); %rt
        net(x+1,y+1,:) = net(x+1,y+1,:) + disc.*alpha.*(inp - net(x+1,y+1,:)); %diag rt dn
        net(x+1,y,:) = net(x+1,y,:) + disc.*alpha.*(inp - net(x+1,y,:)); %below
    % Bottom Left Corner
    elseif (x == dim) && (y == 1)
        net(x-1,y,:) = net(x-1,y,:) + disc.*alpha.*(inp - net(x-1,y,:)); %above
        net(x-1,y+1,:) = net(x-1,y+1,:) + disc.*alpha.*(inp - net(x-1,y+1,:)); %diag rt up
        net(x,y+1,:) = net(x,y+1,:) + disc.*alpha.*(inp - net(x,y+1,:)); %rt
    % Bottom Edge
    elseif (x == dim) && (y > 1) && (y < dim)
        net(x,y-1,:) = net(x,y-1,:) + disc.*alpha.*(inp - net(x,y-1,:)); %lt
        net(x-1,y-1,:) = net(x-1,y-1,:) + disc.*alpha.*(inp - net(x-1,y-1,:)); %diag lt up
        net(x-1,y,:) = net(x-1,y,:) + disc.*alpha.*(inp - net(x-1,y,:)); %above
        net(x-1,y+1,:) = net(x-1,y+1,:) + disc.*alpha.*(inp - net(x-1,y+1,:)); %diag rt up
        net(x,y+1,:) = net(x,y+1,:) + disc.*alpha.*(inp - net(x,y+1,:)); %rt
    % Bottom Right Corner
    elseif (x == dim) && (y == dim)
        net(x,y-1,:) = net(x,y-1,:) + disc.*alpha.*(inp - net(x,y-1,:)); %lt
        net(x-1,y-1,:) = net(x-1,y-1,:) + disc.*alpha.*(inp - net(x-1,y-1,:)); %diag lt up
        net(x-1,y,:) = net(x-1,y,:) + disc.*alpha.*(inp - net(x-1,y,:)); %above
    % Right Edge
    else
        net(x-1,y,:) = net(x-1,y,:) + disc.*alpha.*(inp - net(x-1,y,:)); %above
        net(x-1,y-1,:) = net(x-1,y-1,:) + disc.*alpha.*(inp - net(x-1,y-1,:)); %diag lt up
        net(x,y-1,:) = net(x,y-1,:) + disc.*alpha.*(inp - net(x,y-1,:)); %lt
        net(x+1,y-1,:) = net(x+1,y-1,:) + disc.*alpha.*(inp - net(x+1,y-1,:)); %diag lt dn
        net(x+1,y,:) = net(x+1,y,:) + disc.*alpha.*(inp - net(x+1,y,:)); %below
    end
    
    figure(2)
    imshow(net,'InitialMagnification','fit'); % visualize to ensure random rgb image
    drawnow
end

% visualize how the weights vectors update according to the learning
figure(3)
imshow(net,'InitialMagnification','fit')
title('d = 100');

%need to extend this to the 2 dimensional edge case of the iris dataset
% use distance data to identify different regions in the 2-d map of iris
% dataset. distance data is the 4 features differenced from the actual. the
% dividing lines between the species determine the map. need to show region
% "boundaries" and the labels for each region

% for r,g,b randomly start with a number between 0,1 for red, 0,1 for 
% green, 0,1 for blue