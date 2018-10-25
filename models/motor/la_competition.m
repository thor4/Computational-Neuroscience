% leaky competing accumulator model 2 units
% 2 output units being driven by a constant input
I = 1; % input (sum(Ai*wij))
I2 = 0.9; % unit 2 input
leakage = 0.1; % leakage, decay rate, amount output unit leaks
beta=0.1; %mutual inhibition parameter between the two units
act=0; %initial activity unit 1, starting condition of output unit, not firing at all
act2=0; % initial activity unit 2

time_steps=100;
activity_history=[]; % keep track of activity over time

for tN=1:time_steps
%     if tN>50
%         I = 0; %extinction
%     end
    %change in activity = (total input to unit - lambda*current activity to
    %unit) - inhibition * activity from other unit
    delta_activity = (I-leakage.*act) - beta.*act2; %unit 1
    delta_activity2 = (I2-leakage.*act2) - beta.*act; %unit 2
    act=act+delta_activity+randn.*.5; % update activity for next loop
    act2=act2+delta_activity2+randn.*.5; % update activity for next loop
    if act<0; act=0; end
    if act2<0; act2=0; end
    activity_history = [activity_history [act act2]']; % save activity for this loop
end

% visualize temporal activation profile
plot(activity_history(:,1:20)')
plot(activity_history')