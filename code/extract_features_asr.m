%% Test if ASR code can be used to automatically detect mad channels from Muse data.
% Assess performance using a confusion matrix. 
%
% Cedric Cannard, January 2023

clear; close all; clc
mainDir = 'C:\Users\Cedric\Documents\MATLAB\muse_checksignal';
outDir = fullfile(mainDir, 'outputs', 'channels');
% load(fullfile(mainDir, 'code', 'sInfo.mat'))
load(fullfile(mainDir, 'code', 'maar_sInfo.mat'))
eeglab;close;

% Check if filenames correspond
dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\data_raw';
cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'csv'),:) = [];
ids = extractBetween(filenames,'sub-','_ses','Boundaries','exclusive');
sessions = extractBetween(filenames,'ses-0','_task','Boundaries','exclusive');
fprintf( '%g matches found. \n', sum(contains(ids, {sInfo.id})) )

% Parameters
nFiles = length(sInfo);
max_flat_time = 0.33;       % 0.33 means rejects channels if a third of the channel is flat
maxJitter = 15;             % max jitter tolerated during flatlines (as a multiple of epsilon; default = 20)
window_len = 3;             % window length (in s, default = 2)
ignored_quantile = 0.1;     % 0.05 - 0.2
linenoise_aware = 0;        % corr measure is affected by line noise (0) or not (1, default)
min_corr = .9;              % minimum correlation between channels (default = .85)
max_var = 15;               % ASR variance threshold (2-120)
max_bad_data = .5;          % max amount of data rejected by ASR to consider bad file

true_positive = 0;
false_negative = 0;
true_negative = 0; 
false_positive = 0;
for iFile = 1:nFiles

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

    % Index of bad/good channels from manual tagging
    if islogical(sInfo(iFile).badChan_manual)
        badChanMan = sInfo(iFile).badChan_manual;
    else
        if isempty(sInfo(iFile).badChan_manual)
            badChanMan = false(1,4);
        else
            badChanMan = contains({EEG.chanlocs.labels}, sInfo(iFile).badChan_manual);
        end
    end
    
    % Flat channels
    for iChan = 1:EEG.nbchan
        maxFlat = max_flat_time*EEG.xmax;    % consider bad if 1/3 of signal is flat
        zero_intervals = reshape(find(diff([false abs(diff(EEG.data(iChan,:)))<(maxJitter*eps) false])),2,[])';
        if max(zero_intervals(:,2) - zero_intervals(:,1)) > maxFlat*EEG.srate
            badChanAuto(iChan) = true;
        else
            badChanAuto(iChan) = false;
        end
    end
    
    %  clean_channels_nolocs (correlation to its robust random estimate)
    %  [cleanEEG,rmchans] = clean_channels_nolocs(EEG,minCorr,ignQuantile,winLength,lineNoiseAware);
    % EEG.data = double(EEG.data);
    [nchans,datasize] = size(EEG.data);
    winlen = window_len*EEG.srate;
    wnd = 0:winlen-1;
    offsets = 1:winlen:datasize-winlen;
    W = length(offsets);
    retained = 1:(nchans-ceil(nchans*ignored_quantile));
%     clear X
%     if linenoise_aware % ignore both 50 and 60 Hz spectral components
%         Bwnd = design_kaiser(2*45/EEG.srate,2*50/EEG.srate,60,true);
%         if EEG.srate <= 130
%             B = design_fir(length(Bwnd)-1,[2*[0 45 50 55]/EEG.srate 1],[1 1 0 1 1],[],Bwnd);
%         else
%             B = design_fir(length(Bwnd)-1,[2*[0 45 50 55 60 65]/EEG.srate 1],[1 1 0 1 0 1 1],[],Bwnd);
%         end
%         for iChan = EEG.nbchan:-1:1
%             X(:,iChan) = filtfilt_fast(B,1,EEG.data(iChan,:)');
%         end
%     else
%         X = EEG.data';
%     end
    % For each window, flag channels with too low correlation to any other
    % channel (outside the ignored quantile threshold)
    signal = EEG.data';
    flagged = zeros(nchans,W);
    for o = 1:W
        sortcc = sort(abs(corrcoef(signal(offsets(o)+wnd,:))));
        flagged(:,o) = all(sortcc(retained,:) < min_corr);
    end
    % Mark all channels for removal which have more flagged samples than the
    % maximum number of ignored samples
    badchans = sum(flagged,2)*window_len > size(EEG.data,2)*max_bad_data;
    for iChan = 1:EEG.nbchan
        if badChanAuto(iChan) == 0 && badchans(iChan) == 1
            badChanAuto(iChan) = true;
        end
    end
%     EEG = pop_select(EEG, 'nochannel', {EEG.chanlocs(badChanAuto).labels});

%     % ASR
%     reconstruct = false;
%     useriemannian = false;
%     m = memory;
%     maxmem = round(.85 * (m.MemAvailableAllArrays/1000000),1);  % use half of available memory in MB
% %     disp(['Using 85% of available memory (' num2str(round(maxmem/1000,1)) ' GB)'])
%     cleanEEG = clean_asr(EEG,max_var,[],[],[],[],[],[],[],useriemannian,maxmem);
%     mask = sum(abs(EEG.data-cleanEEG.data),1) > 1e-10;
%     badData = get_badSegments(mask, .1*EEG.srate);
%     rmData = (sum(badData(:,2) - badData(:,1))/EEG.srate) / EEG.xmax;  % bad data ratio
%     if rmData > max_bad_data
%         badChanAuto(1:EEG.nbchan) = true;
%     end
% %     vis_artifacts(cleanEEG,EEG);

    % Confusion matrix
    for iChan = 1:EEG.nbchan
        if badChanMan(iChan) == 1 && badChanAuto(iChan) == 1
            true_positive = true_positive + 1;
        elseif badChanMan(iChan) == 1 && badChanAuto(iChan) == 0 
            false_negative = false_negative + 1;
        elseif badChanMan(iChan) == 0 && badChanAuto(iChan) == 0 
            true_negative = true_negative + 1; 
        elseif badChanMan(iChan) == 0 && badChanAuto(iChan) == 1
            false_positive = false_positive + 1;
        end
    end
end

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
fid = fopen(fullfile(outDir, 'performance_ASR.txt'),'w');
fprintf(fid,['True positive rate (sensitivity or hit rate): %g \n' ...
    'True negative rate (specificity or selectivity): %g \n' ...
    'Positive predictive value (precision): %g \n' ...
    'False negative rate (miss rate): %g \n' ...
    'False discovery rate: %g \n' ...
    'Accuracy: %g'], ...
    round(TPR*100,1), round(TNR*100,1), round(PPV*100,1), round(FNR*100,1), ...
    round(FDR*100,1), round(ACC*100,1));
fclose(fid);

disp('Done')
gong
