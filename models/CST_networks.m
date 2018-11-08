outN=2; %go/change
inpN=2;

weights=eye(2); %go/change connections

salience=1.5; %change signal input strength, lowering will make accuracy for change go down
%could change this to try to match behavior
beta=2; %softmax gain

trialN = 1000;

%keep track of the kind of trial 
Change_Correct = 0;
Change_Error=0;
Go_Correct=0;
Go_Error=0;
TrialSequence=[]; %sequence of trials

for tN=1:trialN
    
    trialtype=randperm(2); %flip a coin
    trialtype=trialtype(2);
    
    switch trialtype
        case 1 %go trial
            input=[1 0];
        case 2 %change trial
            input=[0 salience]; %low should be more of a problem, high easier
    end
    
    output = input*weights;
    response = exp(beta.*output)./(sum(exp(beta.*output)));
    
    if rand<response(1) %go response
        if trialtype==1 %response correct
            Go_Correct=[Go_Correct+1];
        else
            Change_Error=[Change_Error+1];
        end
    else
        if trialtype==1 %go trial network responded
            Go_Error=[Go_Error+1];
        else
            Change_Correct=[Change_Correct+1];
        end
    end
    TrialSequence = [TrialSequence trialtype];
end

Go_Correct./(Go_Correct+Go_Error) %accuracy go trials
Change_Correct./(Change_Correct+Change_Error) %accuracy change trials