%% Initialization
clc; clear all; close all;

addpath('./pcommend/');
addpath(genpath('./data'));

wave_list = [750, 760, 800, 850, 900, 910, 920, 930, 940, 950];
for wave = 1:length(wave_list)
    Image_PA = load(strcat("./data/hb_hbo2_fat_29_15/PA_Image_", num2str(wave_list(wave)), ".mat")).Image_PA;
    Image_PA = (Image_PA - min(Image_PA(:))) / (max(Image_PA(:)) - min(Image_PA(:)));
    PA_Images(wave, :, :) = Image_PA;
end
[n, h, w] = size(PA_Images);
PA_Images = reshape(PA_Images, [length(wave_list), h*w])';
clear Image_PA wave
%% PCOMMEND Algorithm
% Input:
%   - X         : Data (N x D) matrix. N data points of dimensionality D.
%   - parameters: The parameters set by PCOMMEND_Parameters function.
%
% Output:
%   - E         : Cell of C endmembers matrices. One MxD matrix per cluster.
%   - P         : Cell of C abundance matrices. One NxM matrix per cluster.
%   - U         : Fuzzy membership matrix CxN.
%  Parameters - struct - The struct contains the following fields:
%                   1. alpha : Regularization Parameter to trade off
%                       between the RSS and V terms of ICE and PCOMMEND
%                       (0<alpha<1)
%                   2. changeThresh: Stopping Criteria, Change threshold
%                       for Objective Function.
%                   3. M: Number of endmembers per cluster.
%                   4. iterationCap: Maximum number of iterations.
%                   5. C: Number of clusters.
%                   6. m: Fuzzifier.
%                   7. EPS: small positive constant.
Parameters.alpha = 0.0001;
Parameters.changeThresh = 1e-6;
Parameters.M = 3;
Parameters.iterationCap = 1000;
Parameters.C = 3;
Parameters.m = 3;
Parameters.EPS = 0.0001;

[E, P, U] = PCOMMEND(PA_Images, Parameters);

%% Plotting End Members
for i = 1:1
    figure;
    plot(wave_list, E{i}');
    legend(arrayfun(@(x) sprintf('%d', x), 1:Parameters.M, 'UniformOutput', false));
end

%% Plotting Unmixed Sources
for i = 1:1
    figure;
    sgtitle(strcat('Unmixed Sources: ', num2str(i)));
    for j = 1:Parameters.M
        subplot(1, Parameters.M, j)
        imagesc(reshape(squeeze(P{i}(:, j)), [h, w]));
        colorbar;
        colormap hot;
    end
end
%% Plotting Fuzzy Maps
figure;
sgtitle('Fuzzy Maps');
for i = 1:Parameters.m
    subplot(1, Parameters.m, i)
    imagesc(reshape(squeeze(U(i, :)), [h, w]));
    colorbar;
    colormap hot;
end