%test the model with best-fit parameters, creates data that includes
%values derived from the model that change on a trial by trial basis

%load up best drift and bias parameters based on model fit

drift_parameters = [0.003 0.002 0.003 0.003];
bias_parameters = [0.085 -0.281 0.472 -0.094];

% conditions: hel go, hel change, lel go, lel change

trial_sequence=importdata('TrialLabels.txt'); %load up the sequences

%look for string matches
hel_go=ismember(trial_sequence,'High Error Likelihood/GO');
% plot(hel_go) %confirm 1 shows existence of that trial
hel_go_d=hel_go.*drift_parameters(1);
hel_go=hel_go.*bias_parameters(1);

hel_change=ismember(trial_sequence,'High Error Likelihood/CHANGE');
hel_change_d=hel_change.*drift_parameters(2);
hel_change=hel_change.*bias_parameters(2);

lel_go = ismember(trial_sequence,'Low Error Likelihood/GO');
lel_go_d=lel_go.*drift_parameters(3);
lel_go=lel_go.*bias_parameters(3);

lel_change = ismember(trial_sequence,'Low Error Likelihood/CHANGE');
lel_change_d=lel_change.*drift_parameters(4);
lel_change=lel_change.*bias_parameters(4);

%prediction of the trial by trial change in the bias parameter
activity_prediction_vector=hel_go+hel_change+lel_go+lel_change; %bias
activity_prediction_vector_d=hel_go_d+hel_change_d+lel_go_d+lel_change_d; %drift
% 
% figure(1), clf
% plot(activity_prediction_vector)
% %confirm that first trial is lel_go and that shows 0.472 bias
% plot(activity_prediction_vector_d)

%now correlate/regress these against brain activity
