%% Extract a set of EEG features to train machine learning model to discriminate 
% between good and bad channels from Muse data. 
% Different methods are tested (random forest, LDA, GAM, confusion matrix).
% 
% Cedric Cannard, January 2023

clear; close all; clc
mainDir = 'C:\Users\Cedric\Documents\MATLAB\muse_checksignal';
outDir = fullfile(mainDir, 'outputs');
load(fullfile(mainDir, 'code', 'maar_sInfo.mat'))
% load(fullfile(mainDir, 'code', 'sInfo.mat'))

% check if filenames correspond
dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\data_raw';
cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'csv'),:) = [];
ids = extractBetween(filenames,'sub-','_ses','Boundaries','exclusive');
sessions = extractBetween(filenames,'ses-0','_task','Boundaries','exclusive');

fprintf( '%g matches found. \n', sum(contains(ids, {sInfo.id})) )

%% Extract features from good/bad channels on nFiles tagged manually

nFiles = 100;

eeglab; close;
count1 = 1;
count2 = 1;
for iFile = 1:nFiles

    disp('-------------------------------------------------------------------------------')
    fprintf('File %d \n', iFile)
    clear match
    match = find(contains(ids, sInfo(iFile).id));

    if ~exist('match','var'), warning('No match found for this file.'); continue; end
    % tmpsession = extractAfter(sInfo(iFile).filename, 'ses-0');
    % tmpsession = extractBefore(tmpsession, '_task');
    % sInfo(iFile).session = str2double(tmpsession);
%     if length(str2double(sessions(match))) > 1 
    if length(match) > 1
        disp('Several session for this subject, finding matching session.')
        match = match(str2double(sessions(match)) == sInfo(iFile).session);
    end

    % Index of bad/good channels from manual tagging
    if islogical(sInfo(iFile).badChan_manual)
        badChanMan = sInfo(iFile).badChan_manual;
    else
        badChanMan = contains({EEG.chanlocs.labels}, sInfo(iFile).badChan_manual);
    end

    % Import data
    if sum(badChanMan) == 4, fprintf('all channels are bad, skipping file. \n'); continue; end
    try
        EEG = import_muse(fullfile(dataDir, filenames{match}),'eeg');
    catch
        warning('Bad sample rate, skipping file!')
    end
%     if sum(isnan(EEG.data(:,1))) > 0, warning('This file contains NaNs, skipping it.'); continue; end
    EEG = pop_select(EEG, 'nochannel',{EEG.chanlocs(badChanMan).labels});
    EEG = pop_eegfiltnew(EEG,'hicutoff',30);
    badData = sInfo(iFile).badData_manual;
    if ~isempty(badData)
%         goodEEG = eeg_eegrej(EEG, badData);
        goodEEG = pop_select(EEG, 'nopoint', badData);
        badEEG = pop_select(EEG, 'point', badData);
%         pop_eegplot(goodEEG,1,1,1);
%         pop_eegplot(badEEG,1,1,1);
    else
        fprintf('No bad data for this file, skipping it. \n')
        continue
    end

    % Extract features: RMS raw signal, RMS high-freq power, and fuzzy entropy
    % and allocate them to either good channels or bad channely using the
    % badChan index (i.e., manual tagging)
    disp('Extracting channel features...')
    for iChan = 1:EEG.nbchan

        % Estimate power spectra of interest
        powerHF = get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[70 100],'psd');
        powerLF = get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[0.01 3],'psd');

        if badChanMan(iChan)

            % SD raw signal
            badSD(count1,:) = std(EEG.data(iChan,:));

            % RMS raw signal
            badRMS(count1,:) = rms(EEG.data(iChan,:));

            % Peak to RMS
            badPeak2RMS(count1,:) = peak2rms(EEG.data(iChan,:));

            % Kurtosis raw signal
            badKurtosis(count1,:) = kurtosis(EEG.data(iChan,:));

            % Skewness raw signal
            badSkewness(count1,:) = skewness(EEG.data(iChan,:));

            % RMS on high-frequency power
            badPowerHF(count1,:) = rms(powerHF);

            % RMS on low-frequency power
            badPowerLF(count1,:) = rms(powerLF);

            % Fuzzy entropy
            badEntropy(count1,:) = compute_fe(EEG.data(iChan,:));

            % EEG signal
            badSignal(count1,:) = { EEG.data(iChan,:) };

            count1 = count1 + 1;

            % send error when some features don't have the same number of values
            if length(badSD) ~= length(badSkewness)
                errordlg('check bad features data, different number of values when should be the same.')
            end

        else

            % SD raw signal
            goodSD(count2,:) = std(EEG.data(iChan,:));

            % RMS raw signal
            goodRMS(count2,:) = rms(EEG.data(iChan,:));

            % Peak to RMS
            goodPeak2RMS(count2,:) = peak2rms(EEG.data(iChan,:));

            % Kurtosis raw signal
            goodKurtosis(count2,:) = kurtosis(EEG.data(iChan,:));

            % Skewness raw signal
            goodSkewness(count2,:) = skewness(EEG.data(iChan,:));

            % RMS on high-freq power
            goodPowerHF(count2,:) = rms(powerHF);

            % RMS on low-frequency power
            goodPowerLF(count2,:) = rms(powerLF);

            % Fuzzy entropy
            goodEntropy(count2,:) = compute_fe(EEG.data(iChan,:));

            % EEG signal
            goodSignal(count2,:) = { EEG.data(iChan,:) };

            count2 = count2 + 1;

            % send error when some features don't have the same number of values
            if length(goodSD) ~= length(goodSkewness)
                errordlg('check good features data, different number of values when should be the same.'); 
            end

        end
    end
end

% Remove outliers to gain accuracy
outliers = isoutlier(goodSD,'gesd');
goodSD(outliers) = [];
outliers = isoutlier(goodRMS,'gesd');
goodRMS(outliers) = [];
outliers = isoutlier(goodPeak2RMS,'gesd');
goodPeak2RMS(outliers) = [];
outliers = isoutlier(goodKurtosis,'gesd');
goodKurtosis(outliers) = [];
outliers = isoutlier(goodSkewness,'gesd');
goodSkewness(outliers) = [];
outliers = isoutlier(goodPowerHF,'gesd');
goodPowerHF(outliers) = [];
outliers = isoutlier(goodPowerLF,'gesd');
goodPowerLF(outliers) = [];
outliers = isoutlier(goodEntropy,'gesd');
goodEntropy(outliers) = [];
outliers = isoutlier(badSD,'gesd');
badSD(outliers) = [];
outliers = isoutlier(badRMS,'gesd');
badRMS(outliers) = [];
outliers = isoutlier(badPeak2RMS,'gesd');
badPeak2RMS(outliers) = [];
outliers = isoutlier(badKurtosis,'gesd');
badKurtosis(outliers) = [];
outliers = isoutlier(badSkewness,'gesd');
badSkewness(outliers) = [];
outliers = isoutlier(badPowerHF,'gesd');
badPowerHF(outliers) = [];
outliers = isoutlier(badPowerLF,'gesd');
badPowerLF(outliers) = [];
outliers = isoutlier(badEntropy,'gesd');
badEntropy(outliers) = [];

save(fullfile(outDir, 'goodSD'), 'goodSD')
save(fullfile(outDir, 'goodRMS'), 'goodRMS')
save(fullfile(outDir, 'goodPeak2RMS'), 'goodPeak2RMS')
save(fullfile(outDir, 'goodKurtosis'), 'goodKurtosis')
save(fullfile(outDir, 'goodSkewness'), 'goodSkewness')
save(fullfile(outDir, 'goodPowerHF'), 'goodPowerHF')
save(fullfile(outDir, 'goodPowerLF'), 'goodPowerLF')
save(fullfile(outDir, 'goodEntropy'), 'goodEntropy')
save(fullfile(outDir, 'goodSignal'), 'goodSignal')
save(fullfile(outDir, 'badSD'), 'badSD')
save(fullfile(outDir, 'badRMS'), 'badRMS')
save(fullfile(outDir, 'badPeak2RMS'), 'badPeak2RMS')
save(fullfile(outDir, 'badKurtosis'), 'badKurtosis')
save(fullfile(outDir, 'badSkewness'), 'badSkewness')
save(fullfile(outDir, 'badPowerHF'), 'badPowerHF')
save(fullfile(outDir, 'badPowerLF'), 'badPowerLF')
save(fullfile(outDir, 'badEntropy'), 'badEntropy')
save(fullfile(outDir, 'badSignal'), 'badSignal')

disp('Done')

% Comapre visually with histograms
figure('color','w');
subplot(3,3,1)
histogram(goodSD); hold on; histogram(badSD); title('SD'); legend('good', 'bad')
subplot(3,3,2)
histogram(goodRMS); hold on; histogram(badRMS); title('RMS'); 
subplot(3,3,3)
histogram(goodPeak2RMS); hold on; histogram(badPeak2RMS); title('Peak-to-RMS');
subplot(3,3,4)
histogram(goodKurtosis); hold on; histogram(badKurtosis); title('Kurtosis');
subplot(3,3,5)
histogram(goodSkewness); hold on; histogram(badSkewness); title('Skewness');
subplot(3,3,6)
histogram(goodPowerHF); hold on; histogram(badPowerHF); title('HF power');
subplot(3,3,7)
histogram(goodPowerLF); hold on; histogram(badPowerLF); title('LF power');
subplot(3,3,8)
histogram(goodEntropy); hold on; histogram(badEntropy); title('Entropy');
print(gcf, fullfile(outDir, 'histo.png'),'-dpng','-r300');   % 300 dpi .png

gong

%% Look at quantiles to find potentially good features

fid = fopen(fullfile(outDir, 'quantiles.txt'),'w');

% SD
goodQ = quantile(goodSD,3);
badQ = quantile(badSD,3);
fprintf(fid, 'Good SD quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad SD quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    thresh = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using SD of raw signals was found! value = %g \n', round(thresh,1));
end

% RMS
[t,diff,CI,~,~,df,p] = yuen(goodRMS,badRMS,20,0.05)
goodQ = quantile(goodRMS,3);
badQ = quantile(badRMS,3);
fprintf(fid, 'Good RMS quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad RMS quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    thresh = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of raw signals was found! value = %g \n', round(thresh,1));
end

% Peak2RMS
[t,diff,CI,~,~,df,p] = yuen(goodPeak2RMS,badPeak2RMS,20,0.05)
goodQ = quantile(goodPeak2RMS,3);
badQ = quantile(badPeak2RMS,3);
fprintf(fid, 'Good peak2RMS quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad peak2RMS quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    thresh = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using peak-to-RMS of raw signals was found! value = %g \n', round(thresh,1));
end

% Kurtosis
[t,diff,CI,~,~,df,p] = yuen(goodKurtosis,badKurtosis,20,0.05)
goodQ = quantile(goodKurtosis,3);
badQ = quantile(badKurtosis,3);
fprintf(fid, 'Good kurtosis quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad kurtosis quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    thresh = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using Kurtosis of raw signals was found! value = %g \n', round(thresh,1));
end

% Skewness
[t,diff,CI,~,~,df,p] = yuen(goodSkewness,badSkewness,20,0.05)
goodQ = quantile(goodSkewness,3);
badQ = quantile(badSkewness,3);
fprintf(fid, 'Good skewness quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad skewness quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    thresh = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using skewness of raw signals was found! value = %g \n', round(thresh,1));
end

% High-freq power
[t,diff,CI,~,~,df,p] = yuen(goodPowerHF,badPowerHF,20,0.05)
goodQ = quantile(goodPowerHF,3);
badQ = quantile(badPowerHF,3);
fprintf(fid, 'Good HF power quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad HF power quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshPower = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of high-frequency power was found! value = %g \n', round(threshPower,1));
end

% Low-freq power
[t,diff,CI,~,~,df,p] = yuen(goodPowerLF,badPowerLF,20,0.05)
goodQ = quantile(goodPowerLF,3);
badQ = quantile(badPowerLF,3);
fprintf(fid, 'Good LF power quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad LF power quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshPower = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of low-frequency power was found! value = %g \n', round(threshPower,1));
end

% Fuzzy entropy
[t,diff,CI,~,~,df,p] = yuen(goodEntropy,badEntropy,20,0.05)
goodQ = quantile(goodEntropy,3);
badQ = quantile(badEntropy,3);
fprintf(fid, 'Good entropy quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad entropy quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshEntropy = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using fuzzy entropy was found! value = %g \n', round(threshEntropy,1));
end

fclose(fid);
disp(['Quantiles are saved in: ' fullfile(outDir, 'quantiles.txt') ])

%%  Visualize 95% CI of a feature of interest

x1 = goodEntropy;   % goodPowerLF goodEntropy goodRMS
x2 = badEntropy;    % badPowerLF badEntropy badRMS
binsize = 10;

figure('color','w');
goodCI(1) = prctile(x1, (100-95)/2);
goodCI(2) = prctile(x1, 100-(100-95)/2);
[y,x] = histcounts(x1,binsize);
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','blue','EdgeColor','b','facealpha',.8); hold on
patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'blue', 'facealpha',.3,'edgecolor','none')
plot([1 1]*mean(x1), [0 1.5],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(x2, (100-95)/2);
badCI(2) = prctile(x2, 100-(100-95)/2);
[y,x] = histcounts(x2,binsize);
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'red', 'facealpha',.3,'edgecolor','none')
plot([1 1]*mean(x2), [0 1.5],'r--', 'linew',2, 'linewidth', 3)
legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
% print(gcf, fullfile(outDir, 'ci-plot.png'),'-dpng','-r300');   % 300 dpi .png

%% Use the remaining files to validate with a confusion matrix

% Thresholds based on quantiles and histograms
threshSD = 13;
threshLF = 13;
threshRMS = 13;

true_positive = 0;
true_negative = 0;
false_positive = 0;
false_negative = 0;
for iFile = 76:150%length(sInfo)

    disp('-------------------------------------------------------------------------------')
    fprintf('File %d \n', iFile)
    clear match
    match = find(contains(ids, sInfo(iFile).id));

    if ~exist('match','var'), disp('No match found for this file.'); continue; end
    %     tmpsession = extractAfter(sInfo(iFile).filename, 'ses-0');
    %     tmpsession = extractBefore(tmpsession, '_task');
    %     sInfo(iFile).session = str2double(tmpsession);
    if str2double(sessions(match)) ~= sInfo(iFile).session
        disp('Several session for this subject, finding matching session.')
        match = match( str2double(sessions(match)) == sInfo(iFile).session );
    end

    % Import data
    try
        EEG = import_muse(fullfile(dataDir, filenames{match}),'eeg');
    catch
        disp('Erronous sample rate: skipping file.')
        continue
    end

    if sum(isnan(EEG.data(:,1))) > 0, warning('This file contains NaNs, skipping it.'); continue; end

    % Index of bad/good channels from manual tagging
    badChanMan = contains({EEG.chanlocs.labels}, sInfo(iFile).badChan_manual);

    disp('Scanning channels...')
    parfor iChan = 1:EEG.nbchan

        % Features to classify as good or bad
        sigSD = std(EEG.data(iChan,:));
        sigRMS = rms(EEG.data(iChan,:))
        sigLF = rms(get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[0.01 3],'psd'));
        if  sigSD > threshSD || sigRMS > threshRMS || sigLF > threshLF
            badChanAuto = true;
        else
            badChanAuto = false;
        end

        % Confusion matrix
        if badChanMan(iChan) == 1 && badChanAuto == 1
            true_positive = true_positive + 1;
        elseif badChanMan(iChan) == 1 && badChanAuto == 0
            false_negative = false_negative + 1;
        elseif badChanMan(iChan) == 0 && badChanAuto == 0
            true_negative = true_negative + 1;
        elseif badChanMan(iChan) == 0 && badChanAuto == 1
            false_positive = false_positive + 1;
        end
    end
end
disp('Done')

% True positive rate (sensitivity or hit rate)
TPR = true_positive / (true_positive + false_negative);
fprintf('True positive rate (sensitivity or hit rate): %g%% \n', round(TPR*100,1))

% True negative rate (specificity or selectivity)
TNR = true_negative / (true_negative + false_positive);
fprintf('True negative rate (specificity or selectivity): %g%% \n', round(TNR*100,1))

% Positive predictive value (precision)
PPV = true_positive / (true_positive + false_positive);
fprintf('Positive predictive value (precision): %g%% \n', round(PPV*100,1))

% False negative rate (miss rate)
FNR = false_negative / (false_negative + true_positive);
fprintf('False negative rate (miss rate): %g%% \n', round(FNR*100,1))

% False discovery rate
FDR = false_positive / (false_positive + true_positive);
fprintf('False discovery rate: %g%% \n', round(FDR*100,1))

% Accuracy
ACC = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative);
fprintf('Accuracy: %g%% \n', round(ACC*100,1))

% save in text file
fid = fopen(fullfile(outDir, 'performance_powerLF.txt'),'w');
fprintf(fid,['True positive rate (sensitivity or hit rate): %g \n' ...
    'True negative rate (specificity or selectivity): %g \n' ...
    'Positive predictive value (precision): %g \n' ...
    'False negative rate (miss rate): %g \n' ...
    'False discovery rate: %g \n' ...
    'Accuracy: %g'], ...
    round(TPR*100,1), round(TNR*100,1), round(PPV*100,1), round(FNR*100,1), ...
    round(FDR*100,1), round(ACC*100,1));
fclose(fid);

gong






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

%% Select predictors for Random Forest (needs response variable to be double vector

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

