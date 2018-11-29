% load('fmri_CST_data.mat')

imagesc(img_sequence(:,:,1)), colormap(gray) %check it

results_img=zeros(256,256); %store results

for i=1:256
    for j=1:256
        
        %get "brain activity" from experiment at each pixel
        curr_vec=squeeze(img_sequence(i,j,:));
        
        %correlate predicted activity to actual brain activity
        [r,p] = corr(curr_vec,activity_prediction_vector);
        
        results_img(i,j)=p;
        
    end
end

imagesc(results_img<.001) %show only the p<0.001 significant p values