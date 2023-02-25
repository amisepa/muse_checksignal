%% Tag good/bad EEG data from Muse 2016 files manually to then automatically
% classify new files.
%
% Cedric Cannard, Feb, 2022
clear; close all; clc
eeglab; close;

dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\data_raw';
outDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\tagged_data\muse_2016';

cd(dataDir)
filenames = dir;
filenames = { filenames.name }';
filenames(~contains(filenames,'csv'),:) = [];
ids = extractBetween(filenames,'sub-','_ses','Boundaries','exclusive');
sessions = extractBetween(filenames,'ses-0','_task','Boundaries','exclusive');

% Initialize these only 1st time
% front_good = [];
% front_bad = [];
% post_good = [];
% post_bad = [];
load(fullfile(outDir, "sInfo.mat"));
load(fullfile(outDir, "front_good.mat"));
load(fullfile(outDir, "front_bad.mat"));
load(fullfile(outDir, "post_good.mat"));
load(fullfile(outDir, "post_bad.mat"));

figure; 
subplot(2,2,1)
plot(front_good); hold on; plot(front_bad); 
legend('front good', 'front bad'); title('Raw signal')
subplot(2,2,2)
histogram(front_good); hold on; histogram(front_bad); xlim([-150 150])
legend('front good', 'front bad'); title('Distribution of raw signal')
subplot(2,2,3)
plot(post_good); hold on; plot(post_bad); 
legend('posterior good', 'posterior bad'); title('Raw signal')
subplot(2,2,4)
histogram(post_good); hold on; histogram(post_bad); xlim([-150 150])
legend('posterior good', 'posterior bad'); title('Distribution of raw signal')

% for eegplot
mycommand = '[tmpgood, com] = eeg_eegrej(EEG,eegplot2event(TMPREJ,-1));';

%% Launch this at the end of each file

% File to tag
iFile = length(sInfo) + 1;
if iFile == 51
    error('You did 50 files. Remove NaN files and NaNs in signals.')
end
disp('---------------------------------------------------------------')
fprintf('File %d \n', iFile)

% Import file
sInfo(iFile).filename = filenames(iFile);
try
    EEG = import_muse(fullfile(dataDir, filenames{iFile}),'eeg');
    imported = true;
catch
    warning('Bad sample rate, skipping this file.');
    imported = false;
    for iChan = 1:4
        field = sprintf('channel%g_badData',iChan);
        sInfo(iFile).(field) = NaN;
    end
end

if imported
    for iChan = 1:EEG.nbchan

        tmpeeg = pop_select(EEG,'channel',iChan);

        % PSD
        f1 = figure;
        [power, f] = get_psd(tmpeeg.data,tmpeeg.srate,'hamming',50,[],tmpeeg.srate,[0.5 120],'psd');
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
end

disp('Done with this file.')
nans = cell2mat(cellfun(@(x) any(isnan(x)), {sInfo.channel1_badData}, 'UniformOutput',false));
fprintf('%g files completed. \n', length(sInfo)-sum(nans))
disp('Press CTRL + ENTER to launch the next one.')

