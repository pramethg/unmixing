clear; close all; clc;

addpath(genpath('./data/'));
load('./data/3DPlot.mat');
load('./data/ABSCOEFFS.mat');

% [750, 760, 800, 850, 900, 925]
% [750, 760, 800, 850, 900, 910, 920, 930, 940, 950]
wavelist = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950];

%% Loading Simulation Data
PA_Images = zeros(size(wavelist, 2), 396, 101);
for widx = 1:size(wavelist, 2)
    PA_Images(widx, :, :) = load(strcat('./data/hb_hbo2_fat_11_15/PA_Image_', num2str(wavelist(widx)))).Image_PA;
end

%% Loading Experimental Data

%% Normalizing Data
for idx = 1:size(PA_Images, 1)
    minval = min(PA_Images(idx, :, :), [], 'all');
    maxval = max(PA_Images(idx, :, :), [], 'all');
    PA_Images(idx, :, :) = (PA_Images(idx, :, :) - minval) / (maxval - minval);
end
clear minval maxval idx widx

%% Linear Unmixing
NCOMP = 3;
PA_Sources = zeros(NCOMP, size(PA_Images, 2), size(PA_Images, 3));
for i = 1:size(PA_Sources, 2)
    for j = 1:size(PA_Sources, 3)
        PA_Sources(:, i, j) = lsqnonneg(EXP10, PA_Images(:, i, j));
    end
end

%% Plotting Sources
figure;
for fidx = 1:NCOMP
    subplot(1, NCOMP, fidx)
    imshow(squeeze(PA_Sources(fidx, :, :)), []);
    title(strcat('C-',num2str(fidx)));
    colorbar;
    colormap hot;
end