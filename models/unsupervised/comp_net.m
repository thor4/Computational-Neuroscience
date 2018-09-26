% Competitive Networks
c1_center = -.5;
c1_dist = .1;

c2_center=2;
c2_dist=.2;
c3_center=2.5;
c3_dist=.2;

cs=[c1_center c1_dist;
    c2_center c2_dist;
    c3_center c3_dist];

% setup network
inpN = 2;
outN = 2;
weights = randn(inpN,outN).*.1;
alpha=0.1;
trainingN=10000;
training_hist=[];

for tN = 1:trainingN
    %get training cluster
    curr_samp = randperm(3); curr_samp = curr_samp(1);
    
    %determine x,y coordinates
    x_coord = randn(1) .*cs(curr_samp,2) + cs(curr_samp,1);
    y_coord = randn(1) .*cs(curr_samp,2) + cs(curr_samp,1);
    inp=[x_coord y_coord];
    training_hist = [training_hist; inp]; %keep track of all that have been tried
    outAct = inp*weights; %get output
    
    %now determine the winning unit
    winner=find(outAct==max(outAct));
    
    %now calulate diff between activity pattern and 
    weights(:,winner) = weights(:,winner) + alpha.*(inp' - weights(:,winner));
    
    figure(1)
    clf
    scatter(training_hist(:,1),training_hist(:,2),
    hold on
    scatter(weights1,:),weights(1,:),'b0');
    hold off
    drawnow
    pause
    %this gives you the main clusters, weights go to them
    
    scatter(x_coord,y_coord)
    hold on
    drawnow
end
