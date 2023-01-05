%% Test different features extraction and methods to detect bad channels
% automatically from Muse data. Finds highest accuracy using true/false positive/negative rates.
% 
% Cedric Cannard, January 2023

clear; close all; clc
mainDir = 'G:\Shared drives\Science\IDL\6. ANALYSES\cedric\muse_checksignal';
outDir = fullfile(mainDir, 'outputs', 'channels');
load(fullfile(mainDir, 'code', 'maar_sInfo.mat'))

% check if filenames correspond
dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\data_raw';
cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'csv'),:) = [];
ids = extractBetween(filenames,'sub-','_ses','Boundaries','exclusive');
sessions = extractBetween(filenames,'ses-0','_task','Boundaries','exclusive');

fprintf( '%g matches found. \n', sum(contains(ids, {sInfo.id})) )

%% Extract features of good and bad channels using half the files tagged manually

count1 = 1;
count2 = 1;
for iFile = 1:75%length(sInfo)

    disp('-------------------------------------------------------------------------------')
    fprintf('File %d \n', iFile)
    clear match
    match = find(contains(ids, sInfo(iFile).id));

    if ~exist('match','var'), disp('No match found for this file.'); continue; end
    if str2double(sessions(match)) ~= sInfo(iFile).session
        disp('Several session for this subject, finding matching session.')
        match = match(str2double(sessions(match)) == sInfo(iFile).session);
    end

    % Import data
    try
        EEG = import_muse(fullfile(dataDir, filenames{match}),'eeg');
    catch
        disp('Erronous sample rate: skipping file.')
        continue
    end
    
    if sum(isnan(EEG.data(:,1))) > 0, warning('This file contains NaNs, skipping it.'); continue; end

    % Remove DC component
    EEG = rm_DC(EEG); 
%     pop_eegplot(EEG,1,1,1);
    
    % Index of bad/good channels from manual tagging
    badChan = contains({EEG.chanlocs.labels}, sInfo(iFile).badChan_manual);

    % Extract features: RMS raw signal, RMS high-freq power, and fuzzy entropy
    % and allocate them to either good channels or bad channely using the
    % badChan index (i.e., manual tagging)
    disp('Extracting channel features...')
    for iChan = 1:EEG.nbchan

        powerHF = get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[70 100],'psd');
        powerLF = get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[0.01 3],'psd');

        if badChan(iChan)
            % RMS raw signal
            badRMS(count1,:) = rms(EEG.data(iChan,:));

            % Kurtosis raw signal
            badKurtosis(count1,:) = kurtosis(EEG.data(iChan,:));

            % RMS on high-frequency power
            badPowerHF(count1,:) = rms(powerHF);

            % RMS on low-frequency power
            badPowerLF(count1,:) = rms(powerLF);

            % Fuzzy entropy
            badEntropy(count1,:) = compute_fe(EEG.data(iChan,:));

            count1 = count1 + 1;
        else
            % RMS raw signal
            goodRMS(count2,:) = rms(EEG.data(iChan,:));

            % Kurtosis raw signal
            goodKurtosis(count1,:) = kurtosis(EEG.data(iChan,:));

            % RMS on high-freq power
            goodPowerHF(count2,:) = rms(powerHF);

            % RMS on low-frequency power
            goodPowerLF(count1,:) = rms(powerLF);

            % Fuzzy entropy
            goodEntropy(count1,:) = compute_fe(EEG.data(iChan,:));

            count2 = count2 + 1;
        end
    end
end

% Remove outliers to gain accuracy
outliers = isoutlier(goodRMS,'gesd'); 
goodRMS(outliers) = [];
outliers = isoutlier(goodKurtosis,'gesd'); 
goodKurtosis(outliers) = [];
outliers = isoutlier(goodPowerHF,'gesd');
goodPowerHF(outliers) = [];
outliers = isoutlier(goodPowerLF,'gesd');
goodPowerLF(outliers) = [];
outliers = isoutlier(goodEntropy,'gesd');  
goodEntropy(outliers) = [];

outliers = isoutlier(badRMS,'gesd');
badRMS(outliers) = [];
outliers = isoutlier(badKurtosis,'gesd');
badKurtosis(outliers) = [];
outliers = isoutlier(badPowerHF,'gesd');  
badPowerHF(outliers) = [];
outliers = isoutlier(badPowerLF,'gesd');  
badPowerLF(outliers) = [];
outliers = isoutlier(badEntropy,'gesd');    
badEntropy(outliers) = [];

save(fullfile(outDir, 'goodRMS'), 'goodRMS')
save(fullfile(outDir, 'goodKurtosis'), 'goodKurtosis')
save(fullfile(outDir, 'goodPowerHF'), 'goodPowerHF')
save(fullfile(outDir, 'goodPowerLF'), 'goodPowerLF')
save(fullfile(outDir, 'goodEntropy'), 'goodEntropy')
save(fullfile(outDir, 'badRMS'), 'badRMS')
save(fullfile(outDir, 'badKurtosis'), 'badKurtosis')
save(fullfile(outDir, 'badPowerHF'), 'badPowerHF')
save(fullfile(outDir, 'badPowerLF'), 'badPowerLF')
save(fullfile(outDir, 'badEntropy'), 'badEntropy')

disp('Done')

% Comapre visually with histograms 
figure('color','w');
subplot(3,2,1)
histogram(goodRMS); hold on; histogram(badRMS); title('RMS'); legend('good', 'bad')
subplot(3,2,2)
histogram(goodKurtosis); hold on; histogram(badKurtosis); title('Kurtosis'); 
subplot(3,2,3)
histogram(goodPowerHF); hold on; histogram(badPowerHF); title('HF power'); 
subplot(3,2,4)
histogram(goodPowerLF); hold on; histogram(badPowerLF); title('LF power'); 
subplot(3,2,5)
histogram(goodEntropy); hold on; histogram(badEntropy); title('Entropy'); 
print(gcf, fullfile(outDir, 'histo.png'),'-dpng','-r300');   % 300 dpi .png

% % scaled histograms + 95% CI
% binsize = 10;
% figure('color','w');
% subplot(3,1,1)  % RMS raw signal
% goodCI(1) = prctile(goodRMS, (100-95)/2);     
% goodCI(2) = prctile(goodRMS, 100-(100-95)/2);
% [y,x] = histcounts(goodRMS,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','blue','EdgeColor','b','facealpha',.8); hold on
% patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'blue', 'facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(goodRMS), [0 1],'b--', 'linew',2, 'linewidth', 3)
% badCI(1) = prctile(badPowerHF, (100-95)/2);     
% badCI(2) = prctile(badPowerHF, 100-(100-95)/2);
% [y,x] = histcounts(badPowerHF,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
% patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'red', 'facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(badPower), [0 1],'r--', 'linew',2, 'linewidth', 3)
% % legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
% legend('good (data)', 'good (95% CI)', 'bad (data)', 'bad (95% CI)')
% title('RMS of raw signal ')
% 
% subplot(3,1,2)  % RMS power
% goodCI(1) = prctile(goodPowerHF, (100-95)/2);     
% goodCI(2) = prctile(goodPowerHF, 100-(100-95)/2);
% [y,x] = histcounts(goodPowerHF,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','b','EdgeColor','b','facealpha',.8); hold on
% patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'b','facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(goodPower), [0 1],'b--', 'linew',2, 'linewidth', 3)
% badCI(1) = prctile(badRMS, (100-95)/2);     
% badCI(2) = prctile(badRMS, 100-(100-95)/2);
% [y,x] = histcounts(badRMS,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
% patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'r','facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(badRMS), [0 1],'r--', 'linew',2, 'linewidth', 3)
% % legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
% title('RMS of high-frequency power')
% 
% subplot(3,1,3)  % RMS power
% goodCI(1) = prctile(goodEntropy, (100-95)/2);     
% goodCI(2) = prctile(goodEntropy, 100-(100-95)/2);
% [y,x] = histcounts(goodEntropy,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','b','EdgeColor','b','facealpha',.8); hold on
% patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'blue','facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(goodEntropy), [0 1],'b--', 'linew',2, 'linewidth', 3)
% badCI(1) = prctile(badEntropy, (100-95)/2);     
% badCI(2) = prctile(badEntropy, 100-(100-95)/2);
% [y,x] = histcounts(badEntropy,binsize);    
% y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
% bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
% patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'red', 'facealpha',.3,'edgecolor','none')
% % plot([1 1]*mean(badEntropy), [0 1],'r--', 'linew',2, 'linewidth', 3)
% % legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
% title('Fuzzy entropy')
% 
% print(gcf, fullfile(outDir, 'ci-plot.png'),'-dpng','-r300');   % 300 dpi .png

%% Find thresholds using quantiles

fid = fopen(fullfile(outDir, 'quantiles.txt'),'w');

% RMS
goodQ = quantile(goodRMS,3);
badQ = quantile(badRMS,3);
fprintf(fid, 'Good RMS quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad RMS quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshRMS = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of raw signals was found! RMS = %g \n', round(threshRMS,1));
end

% Kurtosis
goodQ = quantile(goodKurtosis,3);
badQ = quantile(badKurtosis,3);
fprintf(fid, 'Good kurtosis quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad kurtosis quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshRMS = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using Kurtosis of raw signals was found! kurtosis = %g \n', round(threshRMS,1));
end

% High-freq power
goodQ = quantile(goodPowerHF,3);
badQ = quantile(badPowerHF,3);
fprintf(fid, 'Good HF power quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad HF power quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshPower = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of high-frequency power was found! RMS = %g \n', round(threshPower,1));
end

% Low-freq power
goodQ = quantile(goodPowerLF,3);
badQ = quantile(badPowerLF,3);
fprintf(fid, 'Good LF power quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad LF power quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshPower = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using RMS of low-frequency power was found! RMS = %g \n', round(threshPower,1));
end

% Fuzzy entropy
goodQ = quantile(goodEntropy,3);
badQ = quantile(badEntropy,3);
fprintf(fid, 'Good entropy quantiles: %s \n', num2str(goodQ));
fprintf(fid, 'Bad entropy quantiles: %s \n', num2str(badQ));
if goodQ(3) < badQ(1)
    threshEntropy = badQ(1);
    fprintf(fid, '===>>> A good threshold for tagging bad channels using fuzzy entropy was found! Entropy = %g \n', round(threshEntropy,1));
end

fclose(fid);

%% Use the other half of files to validate 

% Thresholds based on quantiles and histograms
threshRMS = 14.9;
threshLF = 14.5;
% treshEntropy = 1; 

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
    if str2double(sessions(match)) ~= sInfo(iFile).session
        disp('Several session for this subject, finding matching session.')
        match = match(str2double(sessions(match)) == sInfo(iFile).session);
    end

    % Import data
    try
        EEG = import_muse(fullfile(dataDir, filenames{match}),'eeg');
    catch
        disp('Erronous sample rate: skipping file.')
        continue
    end
    
    if sum(isnan(EEG.data(:,1))) > 0, warning('This file contains NaNs, skipping it.'); continue; end

    % Remove DC component
%     EEG = rm_DC(EEG); 
%     pop_eegplot(EEG,1,1,1);
    
    % Index of bad/good channels from manual tagging
    badChanMan = contains({EEG.chanlocs.labels}, sInfo(iFile).badChan_manual);
    
    % Confusion matrix
    disp('Scanning channels...')
    parfor iChan = 1:EEG.nbchan
        % Using features to classify as good or bad
        sigRMS = rms(EEG.data(iChan,:))
        lf = rms(get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[0.01 3],'psd'));
        if  sigRMS > threshRMS || lf > threshLF
            badChanAuto = true;
        else
            badChanAuto = false;
        end
        
        % TO ADD: reject bnad data with ASr and if 50% is bad, consider bad channel
%         lineThresh = 5;         % line noise threshold (default = 4)
%         winLength = 10;          % length of windows (in s) to compute corrThresh (default = 5)
%         brokenTime = 0.15;      % max time (fraction of recording) of broken channel (0.1-0.6) 
%         nSamples = 100;         % ransac samples to generate random sampling consensus (in s; default = 50; higher is more robust but longer)
%         [EEG, rmchans] = clean_channels(EEG,corrThresh,lineThresh,winLength,brokenTime,nSamples); 

%         % Reduce false positives with entropy (no improvement)
%         if  compute_fe(EEG.data(iChan,:)) < 1
%             badChanAuto = false;
%         end
        

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
fid = fopen(fullfile(outDir, 'performance.txt'),'w');
fprintf(fid,['True positive rate (sensitivity or hit rate): %g \n' ...
    'True negative rate (specificity or selectivity): %g \n' ...
    'Positive predictive value (precision): %g \n' ...
    'False negative rate (miss rate): %g \n' ...
    'False discovery rate: %g \n' ...
    'Accuracy: %g'], ...
    round(TPR*100,1), round(TNR*100,1), round(PPV*100,1), round(FNR*100,1), ...
    round(FDR*100,1), round(ACC*100,1));
fclose(fid);


%% Random forest (to figure out)

% modelRegress = fitensemble(dataTmp,labelTmp,'Bag',300,'Tree','type','regression','kfold',5);
% errTmp = oobError(modelRegress);
% labelPred = kfoldPredict(modelRegress);
    