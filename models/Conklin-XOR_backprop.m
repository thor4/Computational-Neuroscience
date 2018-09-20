% XOR network

training_data = [1 1; 0 1; 1 0; 0 0];
target = [0; 1; 1; 0];

% Class1 = NOT_XOR, Class2 = XOR, so this would allow for 2 output units
% target_dual = [1 0; 0 1; 0 1; 1 0];

% define network structure - 2 layers
inpN = 2; % input units
hidN = 2; % hidden units
outN = 1; % output units, binary classification scheme [0 1]
alpha = 0.25;
noise = 0.1;

% define the weight matrices 
% nodes in one layer * nodes in next layer = total number of weights
% between the two layers
weights1 = randn(inpN+1, hidN).* noise; % add one to get the bias units
% treat the bias unit like another input unit
weights2 = randn(hidN+1, outN).* noise;

% now train the network
exampN = 1000000; % training examples

% initialize input data categories
inp1 = 0; inp2 = 0; inp3 = 0; inp4 = 0;

for eN = 1:exampN
    id = randperm(4,1); % pick random number from 1-4
    temp_inp = training_data(id,:); % choose random training example
    temp_targ = target(id); % choose associated target for example
    % calculate activity of hidden layer based on input * weights
    hidInp = [temp_inp 1] * weights1; %the bias unit is 1
    % this is matrix multiplication by hand:
    % temp_inp(1).*weights1(1,1) + temp_inp(2).*weights1(2,1)+1.*weigths
    % now pass input through the activation (act) function (sigmoid)
    hidAct = 1./(1+exp(-hidInp)); % sigmoid activation function
    % now output activity
    outInp = [hidAct 1] * weights2;
    outAct = 1./(1+exp(-outInp)); % pass through sigmoid
    % this completes one pass
    % now find the difference between target and prediction for error
    error = temp_targ - outAct;
    % now time for backprop. find change in weights for hidden layer by
    % calculating the partial derivative (pd) of error w/r/t weights:
    % pd(error wrt act) * pd(act wrt input) * pd(input wrt weights)
    % (desired - act) * (act(1 - act)) * (act). add 1 for bias. repmat to
    % ensure all weights are accounted for. node weight updates from
    % hidden->output:
    delta_wj1k1 = error.*(outAct.*(1-outAct)).*hidAct(1);
    delta_wj2k1 = error.*(outAct.*(1-outAct)).*hidAct(2);
    delta_wj3k1 = error.*(outAct.*(1-outAct)); % bias weight update
    % now need the change in weights for the input layer:
    % pd(error) wrt weights1. first need pd(error) wrt hidAct
    h1err = error.*(outAct.*(1-outAct)).*weights2(1); % 1st hidden unit
    h2err = error.*(outAct.*(1-outAct)).*weights2(2); % 2nd hidden unit
    % now pd(error) wrt weights1. repmat to ensure all weights are
    % accounted for. node weight updates from input->hidden:
    delta_wi1j1 = h1err*(hidAct(1).*(1-hidAct(1)))*temp_inp(1);
    delta_wi1j2 = h2err*(hidAct(2).*(1-hidAct(2)))*temp_inp(1);
    delta_wi2j1 = h1err*(hidAct(1).*(1-hidAct(1)))*temp_inp(2);
    delta_wi2j2 = h2err*(hidAct(2).*(1-hidAct(2)))*temp_inp(2);
    delta_wi3j1 = h1err*(hidAct(1).*(1-hidAct(1))); % bias weight update 1
    delta_wi3j2 = h2err*(hidAct(2).*(1-hidAct(2))); % bias weight update 2
    % now put all the delta weights together
    delta_weight1 = ... 
        [delta_wi1j1 delta_wi1j2; ...
        delta_wi2j1 delta_wi2j2; ...
        delta_wi3j1 delta_wi3j2];
    delta_weight2 = ...
        [delta_wj1k1; delta_wj2k1; delta_wj3k1];
    % now that we have the weight changes for each layer, time to update
    % the weights accordingly:
    weights1 = weights1+alpha.*delta_weight1;
    weights2 = weights2+alpha.*delta_weight2;
    % build out data for learning curve plot
    % counter and build out each input for learning curve plot
    switch id
        case 1
            inp1 = inp1 + 1;
            inp1_learning(inp1) = outAct;
        case 2
            inp2 = inp2 + 1;
            inp2_learning(inp2) = outAct;
        case 3
            inp3 = inp3 + 1;
            inp3_learning(inp3) = outAct;
        otherwise
            inp4 = inp4 + 1;
            inp4_learning(inp4) = outAct;
    end
end


% now test the network
test_id = randperm(4,1); % pick random number from 1-4
test_inp = training_data(test_id,:); % choose random training example
test_targ = target(test_id); % choose associated target for example
% calculate activity of hidden layer based on input * weights
hidInp = [test_inp 1] * weights1; %the bias unit is 1
% this is matrix multiplication by hand:
% temp_inp(1).*weights1(1,1) + temp_inp(2).*weights1(2,1)+1.*weigths
% now pass input through the activation (act) function (sigmoid)
hidAct = 1./(1+exp(-hidInp)); % sigmoid activation function
% now output activity
outInp = [hidAct 1] * weights2;
outAct = 1./(1+exp(-outInp)); % pass through sigmoid
% now find the difference between target and prediction for error
error = test_targ - outAct;
% error should be close to 0 if network trained properly

% plot learning curves
figure
plot(inp1_learning,'m'), hold on
plot(inp2_learning,'c'), hold on
plot(inp3_learning,'y'), hold on
plot(inp4_learning,'g')
legend({'[1 1]';'[0 1]';'[1 0]';'[0 0]';},'FontSize', 16)
set(gca,'color','k','ylim',[0 1],'xlim',[0 size(inp1_learning,2)+0.05*size(inp1_learning,2)])
set(gcf,'color','w');
xlabel('# of Examples','FontSize', 20), ylabel('Prediction','FontSize', 20)
title(sprintf('XOR Learning Curve\n sigma=%.2f noise=%.2f',alpha,noise),'FontSize', 24);
