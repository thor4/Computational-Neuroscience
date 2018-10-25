% stroop example in LCA
inpN=4; % number of inputs, 2 words, 2 colors
out=2; % red or green words
% input x output
weights = [.5 0; % font color of red, will weakly activate red output word
            0 .5; % font color of green, will weakly activate green output word
            .7 0; % word color of red strongly activates red output
            0 .7]; % word color green strongly activates green output
lambda = .01;
beta=.01;

act=[0 0];

