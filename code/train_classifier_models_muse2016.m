%% Find best predictors from extracted features using Random forest (or use
% PCA dimension reduction instead). Then, using Matlab's Classification 
% Learner, train various classifiers (e.g. discriminant analysis, 
% decision trees, naive Bayes, KNN, logistic regression, SVM, ensemble 
% classifiers, neural network classifiers) and export the best as a compact 
% model. 
% 
% Validation accuracy is estimated on 75% of the sample, using 
% 5-fold cross-validation to protect against overfitting (5 divisions). 
% Average validation error over all folds is calculated to estimate
% predictive accuracy. Best for small datasets, otherwise, use the "Holdout
% validation" method to use a percentage of data for validation. 
% Accuracy is reported through confusion matrices and ROC curve plots.
% 
% Frontal and posterior channels are done separately. 
% 
% Cedric Cannard, Feb 2022

clear; close all; clc
dataPath = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal\outputs\muse_2016';

load(fullfile(dataPath, 'front_good_RMS'), 'front_good_RMS')
load(fullfile(dataPath, 'front_bad_RMS'), 'front_bad_RMS')
load(fullfile(dataPath, 'front_good_peak2RMS'), 'front_good_peak2RMS')
load(fullfile(dataPath, 'front_bad_peak2RMS'), 'front_bad_peak2RMS')
load(fullfile(dataPath, 'front_good_SKEW'), 'front_good_SKEW')
load(fullfile(dataPath, 'front_bad_SKEW'), 'front_bad_SKEW')
load(fullfile(dataPath, 'front_good_HF'), 'front_good_HF')
load(fullfile(dataPath, 'front_bad_HF'), 'front_bad_HF')
load(fullfile(dataPath, 'front_good_LF'), 'front_good_LF')
load(fullfile(dataPath, 'front_bad_LF'), 'front_bad_LF')
load(fullfile(dataPath, 'front_good_SNR'), 'front_good_SNR')
load(fullfile(dataPath, 'front_bad_SNR'), 'front_bad_SNR')
load(fullfile(dataPath, 'front_good_SAMP'), 'front_good_SAMP')
load(fullfile(dataPath, 'front_bad_SAMP'), 'front_bad_SAMP')
load(fullfile(dataPath, 'front_good_APP'), 'front_good_APP')
load(fullfile(dataPath, 'front_bad_APP'), 'front_bad_APP')
load(fullfile(dataPath, 'front_good_FUZ'), 'front_good_FUZ')
load(fullfile(dataPath, 'front_bad_FUZ'), 'front_bad_FUZ')
load(fullfile(dataPath, 'post_good_RMS'), 'post_good_RMS')
load(fullfile(dataPath, 'post_bad_RMS'), 'post_bad_RMS')
load(fullfile(dataPath, 'post_good_peak2RMS'), 'post_good_peak2RMS')
load(fullfile(dataPath, 'post_bad_peak2RMS'), 'post_bad_peak2RMS')
load(fullfile(dataPath, 'post_good_SKEW'), 'post_good_SKEW')
load(fullfile(dataPath, 'post_bad_SKEW'), 'post_bad_SKEW')
load(fullfile(dataPath, 'post_good_HF'), 'post_good_HF')
load(fullfile(dataPath, 'post_bad_HF'), 'post_bad_HF')
load(fullfile(dataPath, 'post_good_LF'), 'post_good_LF')
load(fullfile(dataPath, 'post_bad_LF'), 'post_bad_LF')
load(fullfile(dataPath, 'post_good_SNR'), 'post_good_SNR')
load(fullfile(dataPath, 'post_bad_SNR'), 'post_bad_SNR')
load(fullfile(dataPath, 'post_good_SAMP'), 'post_good_SAMP')
load(fullfile(dataPath, 'post_bad_SAMP'), 'post_bad_SAMP')
load(fullfile(dataPath, 'post_good_APP'), 'post_good_APP')
load(fullfile(dataPath, 'post_bad_APP'), 'post_bad_APP')
load(fullfile(dataPath, 'post_good_FUZ'), 'post_good_FUZ')
load(fullfile(dataPath, 'post_bad_FUZ'), 'post_bad_FUZ')

% Prep data in table format
rms = [front_good_RMS; front_bad_RMS];
peakrms = [front_good_peak2RMS; front_bad_peak2RMS];
skewness = [front_good_SKEW; front_bad_SKEW];
hf = [front_good_HF; front_bad_HF];
lf = [front_good_LF; front_bad_LF];
snr = [front_good_SNR; front_bad_SNR];
sampEn = [front_good_SAMP; front_bad_SAMP];
appEn = [front_good_APP; front_bad_APP];
fuzzEn = [front_good_FUZ; front_bad_FUZ];
labels(1:length(front_good_RMS),:) = categorical({'G'});
labels(length(labels)+1:length(rms),:) = categorical({'B'});
features_front = table(labels, lf, rms, peakrms, skewness, hf, snr, sampEn, appEn, fuzzEn);
writetable(features_front, fullfile(dataPath, 'features_front.csv'))

rms = [post_good_RMS; post_bad_RMS];
peakrms = [post_good_peak2RMS; post_bad_peak2RMS];
skewness = [post_good_SKEW; post_bad_SKEW];
hf = [post_good_HF; post_bad_HF];
lf = [post_good_LF; post_bad_LF];
snr = [post_good_SNR; post_bad_SNR];
sampEn = [post_good_SAMP; post_bad_SAMP];
appEn = [post_good_APP; post_bad_APP];
fuzzEn = [post_good_FUZ; post_bad_FUZ];
labels(1:length(post_good_RMS),:) = categorical({'G'});
labels(length(labels)+1:length(rms),:) = categorical({'B'});
features_post = table(labels, lf, rms, peakrms, skewness, hf, snr, sampEn, appEn, fuzzEn);
writetable(features_post, fullfile(dataPath, 'features_post.csv'))

%% Find predictors with highest importance using Random Forest
clear; close all; clc
dataPath = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal\outputs\muse_2016';
cd(dataPath)

features_front = readtable(fullfile(dataPath, 'features_front.csv'));

% Train Bagged Ensemble of 200 regression trees to estimate predictor importance
% use all predictors at each node
% use interaction-curvature test to select split predictors ('allsplits',
%       'curvature', 'interaction-curvature')
% use surrogate splits to increase accuracy (when dataset includes missing values)
t = templateTree('NumVariablesToSample','all','PredictorSelection','interaction-curvature','Surrogate','on');
y(strcmp(features_front{:,1},'G'),:) = 1;
y(strcmp(features_front{:,1},'B'),:) = 0;
Mdl = fitrensemble(features_front(:,2:end),y,'Method','Bag','NumLearningCycles',200,'Learners',t);
yHat = oobPredict(Mdl);
R2 = corr(Mdl.Y,yHat)^2;  %Mdl explains 82.7% of the variability around the mean
fprintf('Model R2 = %g \n', round(R2,2))

% Estimate predictor importance values by permuting out-of-bag observations among the trees.
impOOB = oobPermutedPredictorImportance(Mdl);
figure('color','w'); subplot(2,1,1); bar(impOOB);
% title('Unbiased Predictor Importance Estimates'); 
xlabel('Predictor variable'); ylabel('Importance'); 
h = gca; h.XTickLabel = Mdl.PredictorNames; h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Compare predictor importance estimates by permuting out-of-bag observations 
% and those estimates obtained by summing gains in the mean squared error
% (MSE) due to splits on each predictor. Also, obtain predictor association 
% measures estimated by surrogate splits. 
[impGain,predAssociation] = predictorImportance(Mdl);
hold on; 
plot(1:numel(Mdl.PredictorNames),impOOB,'linewidth',3)
plot(1:numel(Mdl.PredictorNames),impGain,'linewidth',3)
title('Predictor Importance Estimation Comparison')
legend('OOB permuted', 'OOB permuted','MSE improvement')
grid on

% Assess predictive measure association to indicate similarity between
% decision rules that split observations. The best surrogate decision split 
% yields the maximum predictive measure of association. You can infer the 
% strength of the relationship between pairs of predictors using the elements 
% of predAssociation. Larger values indicate more highly correlated pairs of 
% predictors.
subplot(2,1,2)
imagesc(predAssociation); title('Predictor Association Estimates')
colorbar; h = gca; h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45; h.TickLabelInterpreter = 'none';
h.YTickLabel = Mdl.PredictorNames;

% Input the strongest association here (yellow or green), if < .7 association
% is not high enough to indicate stron relationship between the 2
% predictors
predAssociation(2,1)  % row,column

% Run Random Forest again using selected predictors and caompare R2
t = templateTree('PredictorSelection','interaction-curvature','Surrogate','on'); % For reproducibility of random predictor selections
MdlReduced  = fitrensemble(features_front(:,{'lf' 'snr'}), y,'Method','Bag','NumLearningCycles',200,'Learners',t);
yHatReduced = oobPredict(MdlReduced);
r2Reduced = corr(Mdl.Y,yHatReduced)^2;
fprintf('Model R2 = %g \n', round(R2,2))
fprintf('Reduced Model R2 = %g \n', round(r2Reduced,2))

% SAVE TABLE WITH BEST PREDICTORS
features_front = features_front(:,{'labels' 'lf' 'snr'}); close(gcf);

% Launch APPS > Classification learner > All
% Save best as compact model as trainedModelFront
% Import ValidationSet and run test
% Save summary and plots 
% Exit and save session

%% Take first and last 50 observations for validation predictions and keep
% the rest for traning (i.e. 308). Requires large dataset. 

% features_front = readtable(fullfile(dataPath, 'features_front.csv'));

nVals = 50;
validationSetFront = features_front(1:nVals,:);
validationSetFront(nVals+1:nVals*2,:) = features_front(end-nVals+1:end,:);
summary(categorical(features_front.labels))
features_front(1:nVals,:) = [];
features_front(end-nVals+1:end,:) = [];

%% Prep data and find predictors with highest importance

features_post = readtable(fullfile(dataPath, 'features_post.csv'));

% Train Bagged Ensemble of regression trees to estimate predictor importance
t = templateTree('NumVariablesToSample','all','PredictorSelection','interaction-curvature','Surrogate','on');
y(strcmp(features_post{:,1},'G'),:) = 1;
Mdl = fitrensemble(features_post(:,2:end),y,'Method','Bag','NumLearningCycles',200,'Learners',t);
yHat = oobPredict(Mdl);
R2 = corr(Mdl.Y,yHat)^2;  %Mdl explains 82.7% of the variability around the mean
fprintf('Model R2 = %g \n', round(R2,2))

% Estimate predictor importance values by permuting out-of-bag observations among the trees.
impOOB = oobPermutedPredictorImportance(Mdl);
figure('color','w'); subplot(2,1,1); bar(impOOB);
xlabel('Predictor variable'); ylabel('Importance'); 
h = gca; h.XTickLabel = Mdl.PredictorNames; h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Compare predictor importance estimates and obtain predictor associations 
[impGain,predAssociation] = predictorImportance(Mdl);
hold on; 
plot(1:numel(Mdl.PredictorNames),impOOB,'linewidth',3)
plot(1:numel(Mdl.PredictorNames),impGain,'linewidth',3)
title('Predictor Importance Estimation Comparison')
legend('OOB permuted', 'OOB permuted','MSE improvement')
grid on

% Assess predictive measure association. Larger values indicate more 
% correlated pairs of predictors.
subplot(2,1,2)
imagesc(predAssociation); title('Predictor Association Estimates')
colorbar; h = gca; h.XTickLabel = Mdl.PredictorNames;
h.XTickLabelRotation = 45; h.TickLabelInterpreter = 'none';
h.YTickLabel = Mdl.PredictorNames;

% strongest association
predAssociation(7,8)  % row,column

% Run Random Forest again using selected predictors and caompare R2
t = templateTree('PredictorSelection','interaction-curvature','Surrogate','on'); % For reproducibility of random predictor selections
MdlReduced  = fitrensemble(features_post(:,{'lf' 'snr'}), ...
    y,'Method','Bag','NumLearningCycles',200,'Learners',t);
yHatReduced = oobPredict(MdlReduced);
r2Reduced = corr(Mdl.Y,yHatReduced)^2;
fprintf('Model R2 = %g \n', round(R2,2))
fprintf('Reduced Model R2 = %g \n', round(r2Reduced,2))

% SAVE TABLE WITH BEST PREDICTORS
features_post = features_post(:,{'labels' 'lf' 'snr'}); close(gcf);

%% Take first and last 50 observations for validation predictions and keep
% the rest for traning (i.e. 308). Requires large sample.

nVals = 50;
validationSetPost = features_post(1:nVals,:);
validationSetPost(nVals+1:nVals*2,:) = features_post(end-nVals+1:end,:);
summary(categorical(features_post.labels))
features_post(1:nVals,:) = [];
features_post(end-nVals+1:end,:) = [];

% Launch APPS > Classification learner > All
% Save best as compact model as trainedModelPost
% Import ValidationSet and run test
% Save summary and plots 
% Exit and save session
