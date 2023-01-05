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

sum(contains(ids, {sInfo.id}))

%% Bad channels

count1 = 1;
count2 = 1;
% progressbar('Scanning files')
for iFile = 1:100%length(sInfo)

    disp('-------------------------------------------------------------------------------')
    fprintf('File %d \n', iFile)
    clear match
    match = find(contains(ids, sInfo(iFile).id));

    if ~exist('match','var'), disp('No match found for this file.'); continue; end
    if str2double(sessions(match)) ~= sInfo(iFile).session
        error('sessions don''t match')
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

    % RMS
    disp('Extracting channel features...')
    for iChan = 1:EEG.nbchan

        power = get_psd(EEG.data(iChan,:),EEG.srate,'hamming',50,[],EEG.srate,[70 100],'psd');

        if badChan(iChan)
            % RMS on raw signal
            badRMS(count1,:) = rms(EEG.data(iChan,:));

            % RMS on high-freq power
            badPower(count1,:) = rms(power);
            
            % Fuzzy entropy
            t = tic;
            badEntropy(count1,:) = compute_fe(EEG.data(iChan,:));
            toc(t)
            
            count1 = count1 + 1;
        else
            % RMS
            goodRMS(count2,:) = rms(EEG.data(iChan,:));

            % RMS on high-freq power
            goodPower(count2,:) = rms(power);

            % Fuzzy entropy
            goodEntropy(count1,:) = compute_fe(EEG.data(iChan,:));

            count2 = count2 + 1;
        end
    end

%     progressbar(iFile/150)

end

% Remove outliers (methods: 'mean' 'median' 'quartiles' 'grubbs' 'gesd')
outliers = isoutlier(goodRMS,'gesd'); 
% tmp = goodRMS; tmp(outliers) = []; histogram(tmp)
goodRMS(outliers) = [];
outliers = isoutlier(goodPower,'gesd');
goodPower(outliers) = [];
outliers = isoutlier(goodEntropy,'gesd');  
goodEntropy(outliers) = [];

outliers = isoutlier(badRMS,'gesd');
badRMS(outliers) = [];
outliers = isoutlier(badPower,'gesd');  
badPower(outliers) = [];
outliers = isoutlier(badEntropy,'gesd');    
badEntropy(outliers) = [];

save(fullfile(outDir, 'goodRMS'), 'goodRMS')
save(fullfile(outDir, 'goodPower'), 'goodPower')
save(fullfile(outDir, 'goodEntropy'), 'goodEntropy')
save(fullfile(outDir, 'badRMS'), 'badRMS')
save(fullfile(outDir, 'badPower'), 'badPower')
save(fullfile(outDir, 'badEntropy'), 'badEntropy')

disp('Done')

%% Plot histo and 95% CI

figure('color','w');
subplot(3,1,1)
histogram(goodRMS); hold on; histogram(badRMS); title('RMS raw signal'); legend('good', 'bad')
subplot(3,1,2)
histogram(goodPower); hold on; histogram(badPower); title('RMS HF power'); legend('good', 'bad')
subplot(3,1,3)
histogram(goodEntropy); hold on; histogram(badEntropy); title('Fuzzy entropy'); legend('good', 'bad')
print(gcf, fullfile(outDir, 'histo.png'),'-dpng','-r300');   % 300 dpi .png

binsize = 10;
figure('color','w');
subplot(3,1,1)  % RMS raw signal
goodCI(1) = prctile(goodRMS, (100-95)/2);     
goodCI(2) = prctile(goodRMS, 100-(100-95)/2);
[y,x] = histcounts(goodRMS,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','blue','EdgeColor','b','facealpha',.8); hold on
patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'blue', 'facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(goodRMS), [0 1],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(badPower, (100-95)/2);     
badCI(2) = prctile(badPower, 100-(100-95)/2);
[y,x] = histcounts(badPower,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'red', 'facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(badPower), [0 1],'r--', 'linew',2, 'linewidth', 3)
% legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
legend('good (data)', 'good (95% CI)', 'bad (data)', 'bad (95% CI)')
title('RMS of raw signal ')

subplot(3,1,2)  % RMS power
goodCI(1) = prctile(goodPower, (100-95)/2);     
goodCI(2) = prctile(goodPower, 100-(100-95)/2);
[y,x] = histcounts(goodPower,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','b','EdgeColor','b','facealpha',.8); hold on
patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'b','facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(goodPower), [0 1],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(badRMS, (100-95)/2);     
badCI(2) = prctile(badRMS, 100-(100-95)/2);
[y,x] = histcounts(badRMS,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'r','facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(badRMS), [0 1],'r--', 'linew',2, 'linewidth', 3)
% legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
title('RMS of high-frequency power')

subplot(3,1,3)  % RMS power
goodCI(1) = prctile(goodEntropy, (100-95)/2);     
goodCI(2) = prctile(goodEntropy, 100-(100-95)/2);
[y,x] = histcounts(goodEntropy,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','b','EdgeColor','b','facealpha',.8); hold on
patch(goodCI([1 1 2 2]), [0 1.5 1.5 0],'blue','facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(goodEntropy), [0 1],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(badEntropy, (100-95)/2);     
badCI(2) = prctile(badEntropy, 100-(100-95)/2);
[y,x] = histcounts(badEntropy,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','r','EdgeColor','r','facealpha',.8); hold on
patch(badCI([1 1 2 2]), [0 1.5 1.5 0],'red', 'facealpha',.3,'edgecolor','none')
% plot([1 1]*mean(badEntropy), [0 1],'r--', 'linew',2, 'linewidth', 3)
% legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
title('Fuzzy entropy')

print(gcf, fullfile(outDir, 'ci-plot.png'),'-dpng','-r300');   % 300 dpi .png
% print(gcf, fullfile(outpath, 'ci-plot.tiff'),'-dtiff','-r300');  % 300 dpi .tiff

%% Find thresholds using quantiles

% rms
goodQ = quantile(goodRMS,3);
badQ = quantile(badRMS,3);
fprintf('Good RMS quantiles: %s \n', num2str(goodQ))
fprintf('Bad RMS quantiles: %s \n', num2str(badQ))
if goodQ(3) < badQ(1)
    threshRMS = badQ(1);
    fprintf('===>>> A good threshold for tagging bad channels using RMS of raw signals was found! RMS = %g \n', round(threshRMS,1))
end

% High-freq power
goodQ = quantile(goodPower,3);
badQ = quantile(badPower,3);
fprintf('Good power quantiles: %s \n', num2str(goodQ))
fprintf('Bad power quantiles: %s \n', num2str(badQ))
if goodQ(3) < badQ(1)
    threshPower = badQ(1);
    fprintf('===>>> A good threshold for tagging bad channels using RMS of high-frequency power was found! RMS = %g \n', round(threshPower,1))
end

% Fuzzy entropy
goodQ = quantile(goodEntropy,3);
badQ = quantile(badEntropy,3);
fprintf('Good entropy quantiles: %s \n', num2str(goodQ))
fprintf('Bad entropy quantiles: %s \n', num2str(badQ))
if goodQ(3) < badQ(1)
    threshEntropy = badQ(1);
    fprintf('===>>> A good threshold for tagging bad channels using fuzzy entropy was found! Entropy = %g \n', round(threshEntropy,1))
end


%% find thresholds


