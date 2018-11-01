%paramaters of gabor filter
gamma =.1; %aspect ratio, how wide or thin it will be
sigma=1; %size of receptive field, I think width of gaussian also
lambda = 2; %wavelength, controlling freq of cos
phi=.5; %offset - changes where cosine is in receptive field
theta=.2; %changing the orientation

%define cartesian coordinates in matrix form instead of vectors. this makes
%the math easier to do later 
x = -10:.1:10; y=x;
x=x'*ones(size(y));
y=x';
%visualize x and y: imagesc(x)

%translation of a grid coordinate system into radial coordinates (polar
%system)
x_bar=x.*cos(theta) + y.*sin(theta);
y_bar=-x.*sin(theta) + y.*cos(theta);

% x bar and y bar are coordinates in the visual field
normal_dist = exp(-(x_bar.^2+gamma.^2.*y_bar.^2)./(2.*sigma.^2)); %sigma is std dev around normal curve
cosine_component = cos(2.*pi.*x_bar./lambda+phi); % component of gabor filter
% visualize imagesc(normal_dist) and imagesc(cosine_component) and multiply
% them together to get a gabor filter
imagesc(normal_dist.*cosine_component)