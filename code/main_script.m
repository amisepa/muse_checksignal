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
for iFile = 1:10%length(sInfo)

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

disp('Done')



