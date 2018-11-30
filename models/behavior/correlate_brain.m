load('fmri_CST_data.mat') %load fmri synthetic "images"

imagesc(img_sequence(:,:,1)), colormap(gray) %check it

results_img=zeros(256,256); %store results, pixel-by-pixel
results_img_d=zeros(256,256); %store results, pixel-by-pixel for drift param

%get activity prediction vector
predict_activity

for i=1:256
    for j=1:256
        
        %get "brain activity" from experiment at each pixel
        curr_vec=squeeze(img_sequence(i,j,:));
        
        %correlate predicted activity to actual brain activity
        [r_d,p_d] = corr(curr_vec,activity_prediction_vector); %drift
        [r,p] = corr(curr_vec,activity_prediction_vector); %bias
                
        results_img(i,j)=p;
        results_img_d(i,j)=p_d;
        
    end
end

figure(1), clf
imagesc(1:256,1:256,1e10.*(squeeze(mean(img_sequence,3)))), hold on %scale up so it survives the contour
contour(1:256,1:256,results_img<.001,5,'linecolor','w') %show only the p<0.001 significant p values
title('regions of brain that correlate w/ best-fit drift rate and bias params, p<0.001')
export_fig bias_param_best_fit_p_001.pdf % export to pdf
figure(2), clf %show drift, same as bias
imagesc(results_img_d<.001) %show only the p<0.001 significant p values