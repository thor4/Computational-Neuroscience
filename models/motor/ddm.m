% drift diffusion model
% start with defining parameters to setup diffusion process
% response is positive or negative
starting_point = 0; %initial starting point
drift_rate = 0.002; % can be positive or negative, positive here, toward 'positive' resp
upper_decision_boundary = 1; %arbitrary, could be .5 or 100, scaling issue
lower_decision_boundary = -1;
% noise = 0.05; %std of a norm distrib (gaussian noise)
noise =current_noise; %takes noise from batch script

%don't know how long this will take (since it's random), in contrast with 
% td learning, when we knew exactly how many timesteps it took. so use a
% while loop

current_position=starting_point; %keep track of curr position
noResp=1; %indicates neither boundary has been hit yet, no resp made
upperN = 0; %how often it hits the upper boundary
lowerN=0; %how often it hits the lower boundary

lowerRT = []; %how long it takes to reach lower boundary
upperRT = []; %how long it takes to reach upper boundary
trialN = 10000; %run 50,000 instances of ddm

for tN=1:trialN
time=0; %var to keep track of time
positions = [];
noResp=1; current_position=0;
while noResp==1
    
    time = time+1;
    delta_position = drift_rate + randn.*noise; %get change in position
    current_position=current_position+delta_position; % update the position
    
    %check if the boundaries are hit
    if current_position>=1 %upper decision boundary
        noResp=0; %gets us out of while loop
        upperN = upperN+1;
        upperRT= [upperRT time];
    end
    
    if current_position<=-1 %lower decision boundary
        noResp=0; %gets us out of while loop
        lowerN = lowerN+1;
        lowerRT= [lowerRT time];
    end
    
    positions=[positions current_position];
    
%     plot(positions)
%      xlim([0 100]);
%       ylim([-1 1]);
%        drawnow;
%     pause
end %end while loop, single run through drfit diffusion model
end

%visualize
histogram(lowerRT)
hold on
histogram(upperRT)
