%% Tag good/bad EEG data from Muse S files manually to then automatically
% classify new files.
%
% Cedric Cannard, Feb, 2022
clear; close all; clc
eeglab; close;

dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\edf_museS';
outDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\labeled_data\muse_s';

cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'eeg'),:) = [];

% Initialize outputs (only 1st time)
signals_post = cell.empty;
signals_front = cell.empty;
labels_post = categorical.empty;
labels_front = categorical.empty;
front_good = [];
front_bad = [];
post_good = [];
post_bad = [];
sInfo = [];

% load(fullfile(outDir, "sInfo.mat"));
% load(fullfile(outDir, "front_good.mat"));
% load(fullfile(outDir, "front_bad.mat"));
% load(fullfile(outDir, "post_good.mat"));
% load(fullfile(outDir, "post_bad.mat"));

%% Launch this at the end of each file

% for eegplot
mycommand = '[tmpgood, com] = eeg_eegrej(EEG,eegplot2event(TMPREJ,-1));';

% File to tag
iFile = length(sInfo) + 1;
if iFile == floor(length(filenames)/2)
    error('You did half the files. Done with labeling data.')
end
disp('---------------------------------------------------------------')
fprintf('File %d \n', iFile)

% Import file
sInfo(iFile).filename = filenames{iFile};
EEG = import_edf(fullfile(dataDir, filenames{iFile}),true);
EEG = pop_select(EEG, 'nochannel',{'CH5','CH6'});
% oriEEG = EEG;
EEG = pop_eegfiltnew(EEG,'locutoff',1);
EEG = pop_eegfiltnew(EEG,'hicutoff',50);

for iChan = 1:2%EEG.nbchan

    tmpeeg = pop_select(EEG,'channel',iChan);

    % Plot PSD
    f1 = figure; figpos(f1);
    [power, f] = get_psd(tmpeeg.data,tmpeeg.srate,'hamming',50,[],tmpeeg.srate,[1 50],'psd');
    fig = plot(f,power,'linewidth',2); hold on;
    legend({tmpeeg.chanlocs.labels});
    if iChan == 1 %|| iChan == 4
        title('Posterior channel','fontsize',12)
    else
        title('Frontal channel','fontsize',12)
    end
    
    % Reject bad segments manually
    eegplot(tmpeeg.data,'winlength',30,'srate',tmpeeg.srate,'spacing',200, ...
        'title','Frontal channels','command',mycommand);

    % Save good/bad signals separately
    reply = input("DONE CLEANING (y/n)? \n ", 's');
    if strcmpi(reply,'y')
        close(f1);

        % Extract latency bounds of bad segments
        if ~isempty(com)
            badData = extractBetween(com, ',',')');
            badData = cellfun(@str2num, badData, 'UniformOutput', false);
            badData = badData{:};
        else
            badData = [];
        end
        field = sprintf('channel%g_badData',iChan);
        sInfo(iFile).(field) = badData;

        % Export goog/bad signals
        tmpgood = pop_select(tmpeeg, 'nopoint', badData);
        % eegplot(tmpgood.data,'winlength',30,'srate',EEG.srate,'spacing',150);   % to check
        tmpbad = pop_select(tmpeeg, 'point', badData);
        % eegplot(tmpbad.data,'winlength',30,'srate',EEG.srate,'spacing',150);    % to check

        if iChan == 1
            % For LSTM deep learning
            signals_post(length(signals_post)+1,:) = { tmpgood.data };
            labels_post(length(labels_post)+1,:) = categorical({'G'});
            signals_post(length(signals_post)+1,:) = { tmpbad.data };
            labels_post(length(labels_post)+1,:) = categorical({'B'});
            
            % Save everything in 1 univariate signal
            post_good = [ post_good tmpgood.data ];
            if ~isempty(badData)
                post_bad = [ post_bad tmpbad.data ];
            else
                post_bad = [ post_bad nan(1,1) ];
            end
            % plot(post_good);hold on; plot(post_bad); legend('good','bad');

        elseif iChan == 2
            % For LSTM deep learning
            signals_front(length(signals_front)+1,:) = { tmpgood.data };
            labels_front(length(labels_front)+1,:) = categorical({'G'});
            signals_front(length(signals_front)+1,:) = { tmpbad.data };
            labels_front(length(labels_front)+1,:) = categorical({'B'});

            % Save everything in 1 univariate signal
            front_good = [ front_good tmpgood.data ];
            if ~isempty(badData)
                front_bad = [ front_bad tmpbad.data ];
            else
                front_bad = [ front_bad nan(1,1) ];
            end
            % plot(front_good);hold on; plot(front_bad); legend('good','bad');
        end
    end

    % save after each channel
    disp('Saving data...')
    save(fullfile(outDir,'labeled_data.mat'), 'signals_post', 'signals_front')
    save(fullfile(outDir,'labels.mat'), 'labels_post', 'labels_front')
    save(fullfile(outDir, 'labeled_data_merged.mat'),'front_good', ...
        'front_bad','post_good','post_bad');
    save(fullfile(outDir, 'sInfo.mat'),'sInfo');
end

disp('Done with this file.')
nans = cell2mat(cellfun(@(x) any(isnan(x)), {sInfo.channel1_badData}, 'UniformOutput',false));
fprintf('%g files completed. \n', length(sInfo)-sum(nans))
disp('Press CTRL + ENTER to launch the next one.')
