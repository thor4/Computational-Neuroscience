% Rescorla Wagner model
% initialize the values for the delta Va and delta Vx
% use same learning rate for everything (alpha), instead of different
% alphaa and alphax
alpha=.5; 
beta=1; %setting it to 1 means it doesn't have an impact on the model
lambda=1;

% weights=[0 0 0];


values = [0 0 0]; %A B X stimuli
trainN = 1000;
% vax=[];
% vbx=[];
value_vec = zeros(3,trainN); %store values for each of the stimuli

for tN=1:trainN
    if rand>.5 %flip a coin, input between 0,1
        inp=[1 0 1];
        reward = lambda; %Ax trial
    else
        inp = [0 1 1];
        reward = 0; %Bx trial
    end
    %calculate level of reward predicted
    predicted_value=sum(inp.*values);
    error = reward - predicted_value;
    
    values=values+alpha.*(error.*inp); %update values
    value_vec(:,tN)=values';
end

%gives us values vector that shows reward prediction of A, B and X with A
%being greatest, values(1), B negative (values(1) and X middle (values(3))

% visualize it
plot(value_vec(1,1:100)), hold on %A
plot(value_vec(3,1:100)), hold on %x
plot(value_vec(2,1:100)) %B

% do this many times to get the plots they got in the seminal paper shown
% in the slides for each letter: A, B and x
    
