% Fisher Iris Data Self Organizing Map
% load iris data
data = csvread('iris_data.csv');
classes = csvread('iris_classes.csv');
% setup network
tic
inpN = 4; % number of input units
dim = 100; % number of output units
net = randn(dim,dim,inpN).*.1;
alpha = 0.1; % learning rate
trainingN = 1000000; % number of training examples
% training_hist=[]; % initialize array to hold history of all values
% initialize class arrays to index difference values
class1 = 0; class2 = 0; class3 = 0;

for tN = 1:trainingN
    % generate random iris training example
    curr_samp = randperm(150,1); % pick random number from 1-4
    inp(1,1,:) = data(curr_samp,:); % assign input
    % identify class associated with training example
    class = classes(curr_samp);
    
    % calculate Euclidean distance between input and weight matrix. tells
    % how close input pattern is to each output unit's weight vector
    difference = bsxfun(@minus,inp,net);
    difference = sum(abs(difference),3);
    
    % build out average difference matrix per class
    switch class
        case 1
            class1 = class1 + 1;
            class1_diff(:,:,class1) = difference; 
        case 2
            class2 = class2 + 1;
            class2_diff(:,:,class2) = difference; 
        otherwise
            class3 = class3 + 1;
            class3_diff(:,:,class3) = difference; 
    end
    
    % now determine the winning output neuron by identifying the most
    % similar unit coordinates
    [x y] = find(difference == min(min(difference)));
    
    % now initiate learning for winning neuron by updating 2d map
    net(x,y,:) = net(x,y,:) + alpha.*(inp - net(x,y,:));
    
    % now update weights for neighboring units to winner (neighborhood
    % function). update all neighbors with 80% of alpha. need to deal with
    % edge cases of 2d map for each [x,y]:
    disc = 0.80; % neighborhood discount
    for dN = 1:length(x)
        % Top Left Corner
        if (x(dN) == 1) && (y(dN) == 1)
            net(x(dN),y(dN)+1,:) = net(x(dN),y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)+1,:)); %below
            net(x(dN)+1,y(dN),:) = net(x(dN)+1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN),:)); %rt
            net(x(dN)+1,y(dN)+1,:) = net(x(dN)+1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)+1,:)); %diag rt dn
        % Top Edge
        elseif (x(dN) == 1) && (y(dN) > 1) && (y(dN) < dim)
            net(x(dN),y(dN)-1,:) = net(x(dN),y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)-1,:)); %lt
            net(x(dN)+1,y(dN)-1,:) = net(x(dN)+1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)-1,:)); %diag lt dn
            net(x(dN)+1,y(dN),:) = net(x(dN)+1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN),:)); %below
            net(x(dN),y(dN)+1,:) = net(x(dN),y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)+1,:)); %rt
            net(x(dN)+1,y(dN)+1,:) = net(x(dN)+1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)+1,:)); %diag rt dn
        % Top Right Corner
        elseif (x(dN) == 1) && (y(dN) == dim)
            net(x(dN),y(dN)-1,:) = net(x(dN),y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)-1,:)); %lt
            net(x(dN)+1,y(dN)-1,:) = net(x(dN)+1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)-1,:)); %diag lt dn
            net(x(dN)+1,y(dN),:) = net(x(dN)+1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN),:)); %below
        % Left Edge
        elseif (x(dN) > 1) && (x(dN) < dim) && (y(dN) == 1)
            net(x(dN)-1,y(dN),:) = net(x(dN)-1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN),:)); %above
            net(x(dN)-1,y(dN)+1,:) = net(x(dN)-1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)+1,:)); %diag rt up
            net(x(dN),y(dN)+1,:) = net(x(dN),y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)+1,:)); %rt
            net(x(dN)+1,y(dN)+1,:) = net(x(dN)+1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)+1,:)); %diag rt dn
            net(x(dN)+1,y(dN),:) = net(x(dN)+1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN),:)); %below
        % Bottom Left Corner
        elseif (x(dN) == dim) && (y(dN) == 1)
            net(x(dN)-1,y(dN),:) = net(x(dN)-1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN),:)); %above
            net(x(dN)-1,y(dN)+1,:) = net(x(dN)-1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)+1,:)); %diag rt up
            net(x(dN),y(dN)+1,:) = net(x(dN),y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)+1,:)); %rt
        % Bottom Edge
        elseif (x(dN) == dim) && (y(dN) > 1) && (y(dN) < dim)
            net(x(dN),y(dN)-1,:) = net(x(dN),y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)-1,:)); %lt
            net(x(dN)-1,y(dN)-1,:) = net(x(dN)-1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)-1,:)); %diag lt up
            net(x(dN)-1,y(dN),:) = net(x(dN)-1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN),:)); %above
            net(x(dN)-1,y(dN)+1,:) = net(x(dN)-1,y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)+1,:)); %diag rt up
            net(x(dN),y(dN)+1,:) = net(x(dN),y(dN)+1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)+1,:)); %rt
        % Bottom Right Corner
        elseif (x(dN) == dim) && (y(dN) == dim)
            net(x(dN),y(dN)-1,:) = net(x(dN),y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)-1,:)); %lt
            net(x(dN)-1,y(dN)-1,:) = net(x(dN)-1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)-1,:)); %diag lt up
            net(x(dN)-1,y(dN),:) = net(x(dN)-1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN),:)); %above
        % Right Edge
        else
            net(x(dN)-1,y(dN),:) = net(x(dN)-1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN),:)); %above
            net(x(dN)-1,y(dN)-1,:) = net(x(dN)-1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)-1,y(dN)-1,:)); %diag lt up
            net(x(dN),y(dN)-1,:) = net(x(dN),y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN),y(dN)-1,:)); %lt
            net(x(dN)+1,y(dN)-1,:) = net(x(dN)+1,y(dN)-1,:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN)-1,:)); %diag lt dn
            net(x(dN)+1,y(dN),:) = net(x(dN)+1,y(dN),:) + disc.*alpha.*(inp - net(x(dN)+1,y(dN),:)); %below
        end
    end
end

% find average distance matrix for each class
class1_diff_avg = mean(class1_diff,3);
class2_diff_avg = mean(class2_diff,3);
class3_diff_avg = mean(class3_diff,3);
toc

% visualize average distance matrix for each class
% no transformation
figure(6), clf
subplot(131)
imagesc(class1_diff_avg)
% set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
set(gca,'xlim',[1 35],'ylim',[31,50],'ydir','norm')
title('class 1'), colorbar
% c = colorbar; set(get(c,'label'),'string','MI (baseline subtracted)');    
% set(gca,'FontName','Times New Roman','Fontsize', 14);
subplot(132)
imagesc(class2_diff_avg)
% set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
set(gca,'clim',[0,5],'xlim',[1 11],'ylim',[11,50],'ydir','norm')
title('class 2'), colorbar
subplot(133)
imagesc(class3_diff_avg)
% set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
set(gca,'clim',[0,5],'xlim',[1 11],'ylim',[11,50],'ydir','norm')
title('class 3'), colorbar
% colormap(gray)
suptitle(sprintf('no transform, %d iterations, %d dimensions',trainingN,dim));

% log10 transformation
figure(5), clf
subplot(131)
imagesc(log10(class1_diff_avg))
% set(gca,'clim',[-.075 .075],'yscale','log','ytick',round(logspace(log10(frex(1)),log10(frex(end)),6)))
set(gca,'clim',[0.1,0.75],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 1'), colorbar
% c = colorbar; set(get(c,'label'),'string','MI (baseline subtracted)');    
% set(gca,'FontName','Times New Roman','Fontsize', 14);
subplot(132)
imagesc(log10(class2_diff_avg))
set(gca,'clim',[0.1,0.75],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 2'), colorbar
subplot(133)
imagesc(log10(class3_diff_avg))
set(gca,'clim',[0.1,0.75],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 3'), colorbar
colormap(winter)
suptitle(sprintf('log(10) transform, %d iterations, %d dimensions',trainingN,dim));


% exp() transformation (terrible discrimination)
figure(6), clf
subplot(131)
imagesc(exp(class1_diff_avg))
% set(gca,'clim',[-.075 .075],'yscale','log','ytick',round(logspace(log10(frex(1)),log10(frex(end)),6)))
set(gca,'clim',[mean(exp(class1_diff_avg(1:20,15:30)),'all')-4*std(exp(class1_diff_avg(1:20,15:30)),0,'all'),mean(exp(class1_diff_avg(1:20,15:30)),'all')],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 1'), colorbar
% c = colorbar; set(get(c,'label'),'string','MI (baseline subtracted)');    
% set(gca,'FontName','Times New Roman','Fontsize', 14);
subplot(132)
imagesc(exp(class2_diff_avg))
set(gca,'clim',[mean(exp(class2_diff_avg(1:20,15:30)),'all')-4*std(exp(class2_diff_avg(1:20,15:30)),0,'all'),mean(exp(class2_diff_avg(1:20,15:30)),'all')],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 2'), colorbar
subplot(133)
imagesc(exp(class3_diff_avg))
set(gca,'clim',[mean(exp(class3_diff_avg(1:20,15:30)),'all')-4*std(exp(class3_diff_avg(1:20,15:30)),0,'all'),mean(exp(class3_diff_avg(1:20,15:30)),'all')],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
title('class 3'), colorbar
% colormap(winter)
suptitle(sprintf('exp() transform, %d iterations, %d dimensions',trainingN,dim));



xlabel('Time (ms)'), ylabel('imaginary axis')
title('Projection onto imaginary and time axes') 

%need to extend this to the 2 dimensional edge case of the iris dataset
% use distance data to identify different regions in the 2-d map of iris
% dataset. distance data is the 4 features differenced from the actual. the
% dividing lines between the species determine the map. need to show region
% "boundaries" and the labels for each region

% 	• Iris self-organizing map visualization
% 		? Visualize as the distance vector
% 			§ Absolute value of sum of all distances per class
% 			§ The distances produce a grid (ndim x ndim) which you can then visualize
% 			§ Use imagesc(distance)
% 			§ Colormap(gray)
% 			§ Colorbar
% 			§ Avg distance for each example, show average distance map for each class
% 			§ Try the log(distance) to see if that helps discriminate better
% 			§ Or also try exp(distance), any transformations to better classify
% So, show the distance visualizations for each class, be sure they show clear separation per class into different parts of the grid

% legacy visualization
%     % visualize how the weights vectors update according to the learning
%     if (mod(size(training_hist,1),5)) == 0
%         figure(2)
%         clf
%         scatter(training_hist(:,4),training_hist(:,3),'rX');
%         hold on
%     %     scatter(weights(1,:),weights(1,:),'bO'); hold on
%     %     scatter(weights(2,:),weights(2,:),'gO');
%         scatter(net(4,:),net(3,:),'bO');
%         xlabel('Petal.Width'), ylabel('Petal.Length')
%         title('d=30')
%         hold off
%         drawnow
%     end
