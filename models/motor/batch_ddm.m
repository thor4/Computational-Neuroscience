% batch ddm
% automatically cycles through noise parameter
% try diff levels of this noise param and see what effect this has on
% choices to go to lower or upper boundary
noise_params = [0.01:0.01:0.1];

probability_upper = [];
probability_lower = [];

for noiseN = 1:length(noise_params)
    current_noise = noise_params(noiseN);
    ddm
    
    probability_upper=[probability_upper upperN./(upperN+lowerN)];
end

%visualization
figure(1)
plot(noise_params,probability_upper)
% as noise increases, the drift rate gets more impacted and the probability
% of getting an upper decision gets lower