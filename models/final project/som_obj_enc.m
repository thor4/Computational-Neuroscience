%self organizing map to encode object identity during delay period of 
%delayed match-to-sample working memory task

%load delay-period lfp data and associated object classes (3)
%areas x features (samples) x instances (good trials)
clear
load('m2-day1_delay_obj.mat')

class_options = unique(obj); %list all possible objects
length(find(obj==(class_options(1)))) %total of class 1
length(find(obj==(class_options(2)))) %total of class 2
length(find(obj==(class_options(3)))) %total of class 3

% chanN=1;

%% transformation 1: try filtering (do raw first)

filtOrder = 3; % filter order, higher values = steeper roll off around the 
%cutoff but too high and you're gonna get phase distortions
x = 10;
y = 30;
sr = 1000;

%[b,a] = butter(filtOrder,[xx,yy]/(sr/2),'bandpass'); % construct filter; xx 
%and yy are the lower and upper components of the bandpass, sr is your sampling rate
[b,a] = butter(filtOrder,[x,y]/(sr/2),'bandpass');

% filter your data using the coefficients you computed above and save the 
% output. data should be an #samples x #trials matrix
delay_filt = filtfilt(b,a,squeeze(delay(chanN,:,:)));  

% %% transformation 2: try looking at power from fft of entire delay period
% 
% n = 2^nextpow2(size(delay,2)); %nframe in fftPow to pad signal X with 
% % trailing zeros to improve performance of fft
% sr = 1000; %sampling rate
% %transform to freq domain and extract power
% [fftPow, fax] = fftPow(sr,squeeze(delay(chanN,:,:))',n);


%% setup network
inpN = size(delay,2); % number of input units
dim = 50; % number of output units
alpha = 0.1; % learning rate
trainingN = 10000; % number of training examples
chan = size(delay,1);
% training_hist=[]; % initialize array to hold history of all values

tic
for chanN=1:chan
    net = randn(dim,dim,inpN).*.1; %init som
    % initialize class arrays to index difference values
    class1 = 0; class2 = 0; class3 = 0;
    for tN = 1:trainingN
        % generate random iris training example
        curr_samp = randperm(length(obj),1); % pick random number from 1-total instances
        inp(1,1,:) = delay(chanN,:,curr_samp); % assign input
%         inp(1,1,:) = delay_filt(:,curr_samp); % assign input
        % identify class associated with training example
        class = obj(curr_samp);
        
        % calculate Euclidean distance bet input and weight matrix. tells
        % how close input pattern is to each output unit's weight vector
        difference = bsxfun(@minus,inp,net);
        difference = sum(abs(difference),3);
        
        % build out average difference matrix per class
        switch class
            case class_options(1)
                class1 = class1 + 1;
                class1_diff(:,:,class1) = difference; 
            case class_options(2)
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
%     f1=figure('visible','off'), clf
    f1=figure(1), clf
    subplot(131)
    imagesc(class1_diff_avg)
    % set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
%     set(gca,'clim',[0,4],'xlim',[1 16],'ylim',[1,25],'ydir','norm')
    title('class 1'), colorbar
    subplot(132)
    imagesc(class2_diff_avg)
    % set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
%     set(gca,'clim',[0,4],'xlim',[1 16],'ylim',[1,25],'ydir','norm')
    title('class 2'), colorbar
    subplot(133)
    imagesc(class3_diff_avg)
    % set(gca,'clim',[0,6],'xlim',[0 20],'ylim',[15 30],'ydir','norm')
%     set(gca,'clim',[0,4],'xlim',[1 16],'ylim',[1,25],'ydir','norm')
    title('class 3'), colorbar
%     export_fig (sprintf('chan%d dim%d raw.pdf',chanN,dim)); % export to pdf
    saveas(f1,(sprintf('chan%d dim%d raw.pdf',chanN,dim)));
    
%     f2=figure('visible','off'), clf
    f2=figure(2), clf
    subplot(131)
    imagesc(log10(class1_diff_avg))
    % set(gca,'clim',[-.075 .075],'yscale','log','ytick',round(logspace(log10(frex(1)),log10(frex(end)),6)))
%     set(gca,'clim',[0.1,0.75],'xlim',[1 16],'ylim',[1 30],'ydir','norm')
    title('class 1'), colorbar
    % c = colorbar; set(get(c,'label'),'string','MI (baseline subtracted)');    
    % set(gca,'FontName','Times New Roman','Fontsize', 14);
    subplot(132)
    imagesc(log10(class2_diff_avg))
%     set(gca,'clim',[0.1,0.75],'xlim',[1 16],'ylim',[1 30],'ydir','norm')
    title('class 2'), colorbar
    subplot(133)
    imagesc(log10(class3_diff_avg))
%     set(gca,'clim',[0.1,0.75],'xlim',[1 16],'ylim',[1 30],'ydir','norm')
    title('class 3'), colorbar
    colormap(winter)
%     export_fig (sprintf('chan%d dim%d log transform.pdf',chanN,dim)); % export to pdf
    saveas(f2,(sprintf('chan%d dim%d log transform.pdf',chanN,dim)));
    clearvars class1_diff class2_diff class3_diff
end
toc
