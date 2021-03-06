% leaky accumulator model, no competition, single unit
% single output unit being driven by a constant input
I = 1; % input (sum(Ai*wij))
leakage = 0.1; % leakage, decay rate, amount output unit leaks
act = 0; %initial starting condition of output unit, not firing at all

time_steps=100;
activity_history=[]; % keep track of activity over time

for tN=1:time_steps
    if tN>50
        I = 0; %extinction
    end
    %change in activity = total input to unit - lambda*current activity to unit
    delta_activity = (I-leakage.*act);
    act=act+delta_activity; % update activity for next loop
    activity_history = [activity_history act]; % save activity for this loop
end

% visualize temporal activation profile
plot(activity_history)