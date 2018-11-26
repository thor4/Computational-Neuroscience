% can do coarse grid search to identify which area of parameter space to
% further explore

salience_range = 0:10;
beta_range = 0:10;

%idea of where minimum area will be,
salience_coord = 3.5; %rand.*10;
beta_coord 4; %rand.*10;

stepsize = .2;

epsilon_threshold = .01;
last_error = 10000;
constraints_not_met=1;
coord_history=[];
coord_history=[coord_history; salience_coord(1) bias_coord(1)];

%visualization
coord_history = [coord_history; [salience_coord(1) bias_coord(1)]];
imagesc(squared_distance_matrix);
hold on
plot(coord_history(:,1), coord_history(:,2),'color',[1 1 1]);
pause
hold off