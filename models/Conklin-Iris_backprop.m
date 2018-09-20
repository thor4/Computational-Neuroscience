% Iris dataset neural network
data = csvread('iris_data.csv');
classes = csvread('iris_classes.csv');

% verify balanced # of classes
classN1 = sum(classes(:) == 1); classN2 = sum(classes(:) == 2);
classN3 = sum(classes(:) == 3);
% all = 50, balanced
tic
% define network structure - 2 layers
inpN = 4; % input units
hidN = 5; % hidden units
outN = 3; % output units, binary classification scheme [0 1]
alpha = 0.4;
noise = 0.1;

% define the weight matrices 
% nodes in one layer * nodes in next layer = total number of weights
% between the two layers
weights1 = randn(inpN+1, hidN).* noise; % add one to get the bias units
% treat the bias unit like another input unit
weights2 = randn(hidN+1, outN).* noise;

% now train the network
exampN = 1000000; % training examples

% initialize classifications for learning curve plot
class1 = 0; class2 = 0; class3 = 0;

for eN = 1:exampN
    id = randperm(length(data),1); % pick random index
    temp_inp = data(id,:); % choose random training example
    temp_targ_class = classes(id); % choose associated target for example
    % this is how you do 3 classes for the iris dataset
    target_tri = [1 0 0;...
        0 1 0; ...
        0 0 1];
    temp_targ = target_tri(temp_targ_class,:); % update to 3 to match output
    % calculate activity of hidden layer based on input * weights
    hidInp = [temp_inp 1] * weights1; %the bias unit is 1
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
    % hidden->output. add one for bias. find pd(error) wrt output:
    delta_weight2 = repmat(error,hidN+1,1).*repmat((outAct.*(1-outAct)),hidN+1,1).*repmat([hidAct 1]',1,outN);
%     delta_wj2k1 = error.*(outAct.*(1-outAct)).*hidAct(2);
%     delta_wj3k1 = error.*(outAct.*(1-outAct)); % bias weight update
    % now need the change in weights for the input layer:
    % pd(error) wrt weights1. first need pd(error) wrt hidAct, minus the
    % bias unit since it is not connected to previous layer- no backprop
    % then take the sum of all three errors for each hidden unit
    hiderr = sum(repmat(error,hidN,1).*repmat((outAct.*(1-outAct)),hidN,1).*weights2(1:hidN,:),2); 
%     h1err = error.*(outAct.*(1-outAct)).*weights2(1); % 1st hidden unit
    % now pd(error) wrt weights1. repmat to ensure all weights are
    % accounted for. node weight updates from input->hidden. add one for 
    % bias:
    delta_weight1 = repmat(hiderr',inpN+1,1).*repmat((hidAct.*(1-hidAct)),inpN+1,1).*repmat([temp_inp 1]',1,hidN);
%     delta_wi1j1 = h1err*(hidAct(1).*(1-hidAct(1)))*temp_inp(1);
%     delta_wi1j2 = h2err*(hidAct(2).*(1-hidAct(2)))*temp_inp(1);
%     delta_wi2j1 = h1err*(hidAct(1).*(1-hidAct(1)))*temp_inp(2);
%     delta_wi2j2 = h2err*(hidAct(2).*(1-hidAct(2)))*temp_inp(2);
%     delta_wi3j1 = h1err*(hidAct(1).*(1-hidAct(1))); % bias weight update 1
%     delta_wi3j2 = h2err*(hidAct(2).*(1-hidAct(2))); % bias weight update 2
    % now put all the delta weights together
%     delta_weight1 = ... 
%         [delta_wi1j1 delta_wi1j2; ...
%         delta_wi2j1 delta_wi2j2; ...
%         delta_wi3j1 delta_wi3j2];
%     delta_weight2 = ...
%         [delta_wj1k1; delta_wj2k1; delta_wj3k1];
    % now that we have the weight changes for each layer, time to update
    % the weights accordingly:
    weights1 = weights1+alpha.*delta_weight1;
    weights2 = weights2+alpha.*delta_weight2;
    % build out data for learning curve plot
    % counter and build out each input for learning curve plot
    switch temp_targ_class
        case 1
            class1 = class1 + 1;
            class1_accuracy(class1) = sum(error.^2); % l2
        case 2
            class2 = class2 + 1;
            class2_accuracy(class2) = sum(error.^2);
        otherwise
            class3 = class3 + 1;
            class3_accuracy(class3) = sum(error.^2);
    end
end

% plot accuracy curves
figure
subplot(3,1,1)
plot(smooth(class1_accuracy,101),'m'), set(gca,'XTick',[]);
set(gca,'color','k','xlim',[0 size(class1_accuracy,2)+0.05*size(class1_accuracy,2)])
title('Class 1','FontSize', 14);
ylabel('?(Error^2) MA 101 trials','FontSize', 12)
subplot(3,1,2)
plot(smooth(class2_accuracy,101),'c'), set(gca,'XTick',[]);
set(gca,'color','k','xlim',[0 size(class1_accuracy,2)+0.05*size(class1_accuracy,2)])
title('Class 2','FontSize', 14);
ylabel('?(Error^2) MA 101 trials','FontSize', 12)
subplot(3,1,3)
plot(smooth(class3_accuracy,101),'g')
set(gca,'color','k','xlim',[0 size(class1_accuracy,2)+0.05*size(class1_accuracy,2)])
set(gcf,'color','w');
xlabel('# of Examples','FontSize', 12), ylabel('?(Error^2) MA 101 trials','FontSize', 12)
title('Class 3','FontSize', 14);
suptitle(sprintf('Iris Network: Hidden Units=%d sigma=%.2f noise=%.2f',hidN,alpha,noise));
toc