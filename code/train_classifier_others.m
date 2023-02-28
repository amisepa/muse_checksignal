%% Other tests to find other methods to classify good/bad channels.

clear; close all; clc
mainDir = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal';
outDir = fullfile(mainDir, 'outputs', 'muse_2016');

load(fullfile(outDir, 'front_good_RMS'), 'front_good_RMS')
load(fullfile(outDir, 'front_bad_RMS'), 'front_bad_RMS')
load(fullfile(outDir, 'front_good_peak2RMS'), 'front_good_peak2RMS')
load(fullfile(outDir, 'front_bad_peak2RMS'), 'front_bad_peak2RMS')
load(fullfile(outDir, 'front_good_SKEW'), 'front_good_SKEW')
load(fullfile(outDir, 'front_bad_SKEW'), 'front_bad_SKEW')
load(fullfile(outDir, 'front_good_HF'), 'front_good_HF')
load(fullfile(outDir, 'front_bad_HF'), 'front_bad_HF')
load(fullfile(outDir, 'front_good_LF'), 'front_good_LF')
load(fullfile(outDir, 'front_bad_LF'), 'front_bad_LF')
load(fullfile(outDir, 'front_good_SNR'), 'front_good_SNR')
load(fullfile(outDir, 'front_bad_SNR'), 'front_bad_SNR')
load(fullfile(outDir, 'front_good_SAMP'), 'front_good_SAMP')
load(fullfile(outDir, 'front_bad_SAMP'), 'front_bad_SAMP')
load(fullfile(outDir, 'front_good_APP'), 'front_good_APP')
load(fullfile(outDir, 'front_bad_APP'), 'front_bad_APP')
load(fullfile(outDir, 'front_good_FUZ'), 'front_good_FUZ')
load(fullfile(outDir, 'front_bad_FUZ'), 'front_bad_FUZ')
load(fullfile(outDir, 'post_good_RMS'), 'post_good_RMS')
load(fullfile(outDir, 'post_bad_RMS'), 'post_bad_RMS')
load(fullfile(outDir, 'post_good_peak2RMS'), 'post_good_peak2RMS')
load(fullfile(outDir, 'post_bad_peak2RMS'), 'post_bad_peak2RMS')
load(fullfile(outDir, 'post_good_SKEW'), 'post_good_SKEW')
load(fullfile(outDir, 'post_bad_SKEW'), 'post_bad_SKEW')
load(fullfile(outDir, 'post_good_HF'), 'post_good_HF')
load(fullfile(outDir, 'post_bad_HF'), 'post_bad_HF')
load(fullfile(outDir, 'post_good_LF'), 'post_good_LF')
load(fullfile(outDir, 'post_bad_LF'), 'post_bad_LF')
load(fullfile(outDir, 'post_good_SNR'), 'post_good_SNR')
load(fullfile(outDir, 'post_bad_SNR'), 'post_bad_SNR')
load(fullfile(outDir, 'post_good_SAMP'), 'post_good_SAMP')
load(fullfile(outDir, 'post_bad_SAMP'), 'post_bad_SAMP')
load(fullfile(outDir, 'post_good_APP'), 'post_good_APP')
load(fullfile(outDir, 'post_bad_APP'), 'post_bad_APP')
load(fullfile(outDir, 'post_good_FUZ'), 'post_good_FUZ')
load(fullfile(outDir, 'post_bad_FUZ'), 'post_bad_FUZ')

%% Visualize decision surfaces using different classifiers
clear X; clear y;

X(:,1) = goodRMS(1:100);
X(:,2) = goodPowerLF(1:100);
% X(:,3) = goodPowerHF(1:100);
% X(:,4) = goodKurtosis(1:100);
% X(:,5) = goodEntropy(1:100);
X(101:200,1) = badRMS(1:100);
X(101:200,2) = badPowerLF(1:100);
% X(101:200,3) = badPowerHF(1:100);
% X(101:200,4) = badKurtosis(1:100);
% X(101:200,5) = badEntropy(1:100);

y(1:100,1) = "good";
y(101:200) = "bad";
labels = categories(y);

figure('color','w'); gscatter(X(:,1),X(:,2),y,'rgb','osd');
xlabel('RMS'); ylabel('Power LF');

classifier_name = {'Naive Bayes','Discriminant Analysis','Classification Tree','K-nearest Neighbor'};
classifier{1} = fitcnb(X,y);        % Naive Bayes
classifier{2} = fitcdiscr(X,y);     % Discriminant Analysis
classifier{3} = fitctree(X,y);      % Decision tree
classifier{4} = fitcknn(X,y);       % K-nearest Neighbor

% Create a grid of points spanning the entire space within some bounds of the actual data values.
x1range = min(X(:,1)):.01:max(X(:,1));
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

% Predict the good/bad category of each observation in XGrid using all classifiers.
% Plot scatter plots of the results.
figure('color', 'w');
for i = 1:numel(classifier)
    predictedspecies = predict(classifier{i},XGrid);

    subplot(2,2,i);
    gscatter(xx1(:), xx2(:), predictedspecies,'rgb');

    title(classifier_name{i})
    legend off, axis tight
end
legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal')

%% Select predictors for Random Forest (needs response variable to be double vector)

% X = table(goodRMS(1:100),goodPowerLF(1:100),goodPowerHF(1:100),goodKurtosis(1:100),goodEntropy(1:100));
X = table( [goodRMS(1:100); badRMS(1:100)], [goodPowerLF(1:100); badPowerLF(1:100)], ...
    [goodPowerHF(1:100); badPowerHF(1:100)], [goodKurtosis(1:100); badKurtosis(1:100)], ...
    [goodEntropy(1:100); badEntropy(1:100)] );
y(1:100,1) = "good";
y(101:200) = "bad";

% The standard CART algorithm tends to split predictors with many unique values (levels),
% e.g., continuous variables, over those with fewer levels, e.g., categorical variables.
% If your data is heterogeneous, or your predictor variables vary greatly in their number of levels,
% then consider using the curvature or interaction tests for split-predictor selection instead of standard CART.
% For each predictor, determine the number of levels in the data.
% One way to do this is define an anonymous function that:
%   - Converts all variables to the categorical data type using categorical
%   - Determines all unique categories while ignoring missing values using categories
%   - Counts the categories using numel
% Then, apply the function to each variable using varfun.
countLevels = @(x)numel(categories(categorical(x)));
numLevels = varfun(countLevels,X,'OutputFormat','uniform');

% Compare the number of levels among the predictor variables
figure('color','w')
bar(numLevels)
title('Number of Levels Among Predictors')
xlabel('Predictor variable')
ylabel('Number of levels')
h = gca;
h.XTickLabel = X.Properties.VariableNames(1:end-1);
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Continuous variables have many more levels than categorical variables.
% When there is lots of variability in the number of levels among the
% predictors, using standard CART to select split predictors at each node
% of the trees in a random forest can yield inaccurate predictor importance
% estimates. In this case, use the curvature test or interaction test.
% Specify the algorithm by using the 'PredictorSelection' name-value pair
% argument. For more details, see Choose Split Predictor Selection Technique.

% Train a bagged ensemble of 200 regression trees to estimate predictor importance values.
% Define a tree learner using these name-value pair arguments:
%     'NumVariablesToSample','all' — Use all predictor variables at each node to ensure that each tree uses all predictor variables.
%     'PredictorSelection','interaction-curvature' — Specify usage of the interaction test to select split predictors.
%     'Surrogate','on' — Specify usage of surrogate splits to increase accuracy because the data set includes missing values.
t = templateTree('NumVariablesToSample','all','PredictorSelection','interaction-curvature', ...
    'Surrogate','on');
mdl = fitrensemble(X,y,'Method','Bag','NumLearningCycles',200, ...
    'Learners',t);


%% Random forest (arno's codeto figure out)

X = double( [ [goodRMS(1:100); badRMS(1:100)], [goodPowerLF(1:100); badPowerLF(1:100)], ...
    [goodPowerHF(1:100); badPowerHF(1:100)], [goodKurtosis(1:100); badKurtosis(1:100)], ...
    [goodEntropy(1:100); badEntropy(1:100)] ] );
y(1:100,1) = {'good'};
y(101:200) = {'bad'};

modelRegress = fitensemble(X,y,'Bag',300,'Tree','type','regression','kfold',5);
% errTmp = oobError(modelRegress);
% labelPred = kfoldPredict(modelRegress);







%% Generalized additive model (GAM)

load(fullfile(outDir, 'goodSD'))
load(fullfile(outDir, 'goodRMS'))
load(fullfile(outDir, 'goodPeak2RMS'))
load(fullfile(outDir, 'goodKurtosis'))
load(fullfile(outDir, 'goodSkewness'))
load(fullfile(outDir, 'goodPowerHF'))
load(fullfile(outDir, 'goodPowerLF'))
load(fullfile(outDir, 'goodEntropy'))
load(fullfile(outDir, 'badSD'))
load(fullfile(outDir, 'badRMS'))
load(fullfile(outDir, 'badPeak2RMS'))
load(fullfile(outDir, 'badKurtosis'))
load(fullfile(outDir, 'badSkewness'))
load(fullfile(outDir, 'badPowerHF'))
load(fullfile(outDir, 'badPowerLF'))
load(fullfile(outDir, 'badEntropy'))

x = double( [ [goodRMS(1:200); badRMS(1:100)], [goodPowerLF(1:200); badPowerLF(1:100)], ...
    [goodPowerHF(1:200); badPowerHF(1:100)], [goodKurtosis(1:200); badKurtosis(1:100)], ...
    [goodEntropy(1:200); badEntropy(1:100)] ] );
y(1:200,1) = {'good'};
y(201:300) = {'bad'};

% load ionosphere
mdl = fitcgam(x,y)

% Classify the first observation of the training data
i = 199;
label = predict(mdl,x(i,:))

% Plot the local effects of the terms in mdl on the prediction
figure; plotLocalEffects(mdl,x(i,:))

% --> the 1st observation is labeled as good
% The plot shows the local effects of the 10 most important terms on the prediction.
% Each local effect value shows the contribution of each term to the classification score for 'g',
% which is the logit of the posterior probability that the classification is 'g' for the observation.
plotLocalEffects(mdl,x(i,:))


%% Linear discriminant analysis

% take randomly same number of obs for good and bad variables
X(:,1) = badRMS;
X(:,2) = badPowerLF;
y(1:length(X),:) = "bad";
idx = randsample(1:length(X),length(X));
X(end+1:end+length(idx),1) = goodRMS(idx);
X(end+1:end+length(idx),1) = goodPowerLF(idx);
y(end+1:end+length(idx)) = "good";

% scatter plot
h1 = gscatter(X(:,1),X(:,2),y,'krb','ov^','off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
hold on

