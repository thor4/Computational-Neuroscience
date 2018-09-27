% now we are moving into a real-time model. instead of a for-loop over
% training examples, loop over time steps. each iteration represents a
% sequential moment in time

% eligibility trace
time_steps = 100; %number of time points to simulate
% need to keep track of external stimulus and stimulus trace
% external stimulus
cs_strength = 1; % whether bell is soft or loud
cs_duration = 10; % duration of stimulus (in time steps)
% define when external stimulus appears
cs_time = 20;
% also determine what theta scaling factor is. so difference between
% external stimulus and internal representation will be discounted by
% theta percent
theta = 0.2;

x = 0; % current stimulus
x_vec = zeros(1,time_steps);
% internal representation
xbar = 0; % no trace at the beginning
xbar_vec = zeros(1,time_steps);
for tN = 1:time_steps % time loop
    if tN == cs_time % check to see if timestep is time at which ext stim occurs
        x = cs_strength;
    end
    if tN == cs_time+cs_duration-1 % check to see if it's when stimulus ends
        x = 0;
    end
    x_vec(tN)=x;
    
    % now update x bar (according to slide equation)
    xbar = xbar + theta.*(x - xbar);
    xbar_vec(tN) = xbar;
end

plot(x_vec) % shows stimulus trace
hold on
plot(xbar_vec)