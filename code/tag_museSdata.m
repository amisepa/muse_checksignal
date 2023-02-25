%% Tag good/bad EEG data from Muse S files manually to then automatically
% classify new files.
%
% Cedric Cannard, Feb, 2022
clear; close all; clc
eeglab; close;

dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\edf_museS';
outDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\tagged_data\muse_s';

cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'eeg'),:) = [];

% % Initialize these only 1st time
% sInfo = [];
% front_good = [];
% front_bad = [];
% post_good = [];
% post_bad = [];

% % Load sInfo and tagged data to update them with new files
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
EEG = import_edf(fullfile(dataDir, filenames{iFile}),1);
EEG = pop_select(EEG, 'nochannel',{'CH5','CH6'});
% oriEEG = EEG;
EEG = pop_eegfiltnew(EEG,'locutoff',1);
EEG = pop_eegfiltnew(EEG,'hicutoff',50);

for iChan = 1:EEG.nbchan

    tmpeeg = pop_select(EEG,'channel',iChan); 

    % PSD
    f1 = figure;
    [power, f] = get_psd(tmpeeg.data,tmpeeg.srate,'hamming',50,[],tmpeeg.srate,[1 50],'psd');
    fig = plot(f,power,'linewidth',2); hold on;
    legend({tmpeeg.chanlocs.labels});

    % Reject bad segments manually
    eegplot(tmpeeg.data,'winlength',20,'srate',tmpeeg.srate,'spacing',150, ...
        'title','Frontal channels','command',mycommand);

    % Save good/bad signals separately
    reply = input("DONE CLEANING (y/n)?", 's');
    if strcmpi(reply,'y')
        close(f1);
        if ~isempty(com)
            badData = extractBetween(com, ',',')');
            badData = cellfun(@str2num, badData, 'UniformOutput', false);
            badData = badData{:};
        else
            badData = [];
        end

        field = sprintf('channel%g_badData',iChan);
        % if ~isfield(sInfo,'field'), sInfo(iFile).(field) = []; end
        sInfo(iFile).(field) = badData;

        if iChan == 2 || iChan == 3
            frontGood = pop_select(tmpeeg, 'nopoint', badData);
            frontBad = pop_select(tmpeeg, 'point', badData);
            % eegplot(frontGood.data,'winlength',30,'srate',EEG.srate,'spacing',100);
            % eegplot(frontBad.data,'winlength',30,'srate',EEG.srate,'spacing',100);
            % pop_saveset(frontGood, 'filename',['frontGood' num2str(iFile) '.set'],'filepath',outDir);
            % pop_saveset(frontBad, 'filename',['frontBad' num2str(iFile) '.set'],'filepath',outDir);
            front_good = [ front_good frontGood.data ];
            if ~isempty(badData)
                front_bad = [ front_bad frontBad.data ];
            else
                front_bad = [ front_bad nan(1,1) ];
            end

        elseif iChan == 1 || iChan == 4
            postGood = pop_select(tmpeeg, 'nopoint', badData);
            postBad = pop_select(tmpeeg, 'point', badData);
            % eegplot(postGood.data,'winlength',30,'srate',EEG.srate,'spacing',100);
            % eegplot(postBad.data,'winlength',30,'srate',EEG.srate,'spacing',100);
            post_good = [ post_good postGood.data ];
            if ~isempty(badData)
                post_bad = [ post_bad postBad.data ];
            else
                post_bad = [ post_bad nan(1,1) ];
            end
        end
    end
end

disp('Saving data...')
save(fullfile(outDir, 'sInfo.mat'),'sInfo');
save(fullfile(outDir, 'front_good.mat'),'front_good');
save(fullfile(outDir, 'front_bad.mat'),'front_bad');
save(fullfile(outDir, 'post_good.mat'),'post_good');
save(fullfile(outDir, 'post_bad.mat'),'post_bad');

disp('Done with this file.')
nans = cell2mat(cellfun(@(x) any(isnan(x)), {sInfo.channel1_badData}, 'UniformOutput',false));
fprintf('%g files completed. \n', length(sInfo)-sum(nans))
disp('Press CTRL + ENTER to launch the next one.')
