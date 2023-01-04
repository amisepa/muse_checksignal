%% Test different features extraction and methods to assess signal quality 
% automatically from Muse S data. Finds highest accuracy using true/false positive/negative rates.
% 
% Cedric Cannard, January 2023

clear; close all; clc
mainDir = 'G:\Shared drives\Science\IDL\6. ANALYSES\cedric\muse_checksignal';
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
progressbar('Scanning files')
for iFile = 1:150%length(sInfo)

    disp('-------------------------------------------------------------------------------')
    fprintf('File %d \n', iFile)
    clear match
    match = find(contains(ids, sInfo(iFile).id));

    if ~exist('match','var'), disp('No match found for this file.'); continue; end

    if str2double(sessions(match)) ~= sInfo(iFile).session
        error('sessions don''t match')
    end
    if length(match) > 1
        disp('check sessions')
        break
        %CHECK SESSION
    end

    % Import data
    try
        EEG = import_muse(fullfile(dataDir, filenames{match}),'eeg');
    catch
        disp('Erronous sample rate: skipping file.')
        continue
    end

    % Remove DC component
    EEG = rm_DC(EEG); 
%     pop_eegplot(EEG,1,1,1);
    
    % index of bad/good channels
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

            count1 = count1 + 1;
        else
            % RMS
            goodRMS(count2,:) = rms(EEG.data(iChan,:));

            % RMS on high-freq power
            goodPower(count2,:) = rms(power);

            count2 = count2 + 1;
        end
    end

    progressbar(iFile/10)

end

% figure('color','w');
% subplot(2,1,1)
% histogram(goodRMS); hold on; histogram(badRMS); title('RMS raw signal'); legend('good', 'bad')
% subplot(2,1,1)
% histogram(goodPower); hold on; histogram(badPower); title('RMS HF power'); legend('good', 'bad')

% Plot histo and 95% CI
binsize = 40;
figure('color','w');
subplot(2,1,1)  % RMS raw signal
goodCI(1) = prctile(goodRMS, (100-95)/2);     
goodCI(2) = prctile(goodRMS, 100-(100-95)/2);
[y,x] = histcounts(goodRMS,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','blue','EdgeColor', 'k'); hold on
patch(goodCI([1 1 2 2]), [0 1 1 0],'blue', 'facealpha',.5,'edgecolor','none')
plot([1 1]*mean(goodRMS), [0 1],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(badPower, (100-95)/2);     
badCI(2) = prctile(badPower, 100-(100-95)/2);
[y,x] = histcounts(badPower,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','red','EdgeColor', 'k'); hold on
patch(badCI([1 1 2 2]), [0 1 1 0],'red', 'facealpha',.5,'edgecolor','none')
plot([1 1]*mean(badPower), [0 1],'r--', 'linew',2, 'linewidth', 3)
legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
title('RMS of raw signal ')

subplot(2,1,2)  % RMS power
goodCI(1) = prctile(goodPower, (100-95)/2);     
goodCI(2) = prctile(goodPower, 100-(100-95)/2);
[y,x] = histcounts(goodPower,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','blue','EdgeColor', 'k'); hold on
patch(goodCI([1 1 2 2]), [0 1 1 0],'blue', 'facealpha',.5,'edgecolor','none')
plot([1 1]*mean(goodPower), [0 1],'b--', 'linew',2, 'linewidth', 3)
badCI(1) = prctile(badRMS, (100-95)/2);     
badCI(2) = prctile(badRMS, 100-(100-95)/2);
[y,x] = histcounts(badRMS,binsize);    
y = y./max(y); x = (x(1:end-1)+x(2:end))/2;
bar(x,y,'facecolor','red','EdgeColor', 'k'); hold on
patch(badCI([1 1 2 2]), [0 1 1 0],'red', 'facealpha',.5,'edgecolor','none')
plot([1 1]*mean(badRMS), [0 1],'r--', 'linew',2, 'linewidth', 3)
legend('good (data)', 'good (95% CI)', 'good (mean)', 'bad (data)', 'bad (95% CI)', 'bad (mean)')
title('RMS of high-frequency power')
% print(gcf, fullfile(outpath, 'ci-plot.png'),'-dpng','-r300');   % 300 dpi .png
% print(gcf, fullfile(outpath, 'ci-plot.tiff'),'-dtiff','-r300');  % 300 dpi .tiff

disp('Done')



