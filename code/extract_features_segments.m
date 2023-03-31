%% Extract a set of EEG features from Muse raw data labeled manually.
% Frontal and posterior channels are considered separately.
%
% Cedric Cannard, Feb 2023

clear; close all; clc
mainPath = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal';
% dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\labeled_data\muse_2016';
dataPath = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\labeled_data\muse_s';

minfreq = 0.2;          % minimum frequency for LF power
segSize = 1/minfreq     % in s (t = 1/f)
fs = 256;               % sample rate

% Load data
% load(fullfile(dataPath, 'front_good.mat'));
% load(fullfile(dataPath, 'front_bad.mat'));
% load(fullfile(dataPath, 'post_good.mat'));
% load(fullfile(dataPath, 'post_bad.mat'));
load(fullfile(dataPath, 'labeled_data_merged.mat'));

% Remove NaNs
front_good(isnan(front_good)) = [];
front_bad(isnan(front_bad)) = [];
post_good(isnan(post_good)) = [];
post_bad(isnan(post_bad)) = [];

% Take max number of samples from shortest file
minLength = min([ length(front_good)  length(front_bad) length(post_good) length(post_bad)]);
minLength = minLength / 2; % to divide in 2 halves
fprintf('labeled sets are %g minutes long. \n', round(minLength/256/60,1));

% 1st half is for training and 2nd half for validation
front_good2 = front_good(minLength+1:minLength*2);
front_bad2 = front_bad(minLength+1:minLength*2);
post_good2 = post_good(minLength+1:minLength*2);
post_bad2 = post_bad(minLength+1:minLength*2);
front_good = front_good(1:minLength);
front_bad = front_bad(1:minLength);
post_good = post_good(1:minLength);
post_bad = post_bad(1:minLength);

% for the EM SNR feature
b = design_fir(100,[2*[0 45 50]/fs 1],[1 1 0 0]);

% Extract features for each segment
segSize = segSize * fs;            % convert to samples
nSeg = floor(minLength/segSize);    % # of segments
progressbar('Extracing EEG features on sliding windows...');
for iSeg = 1:nSeg-1
    
    fprintf('   segment %g \n', iSeg);

    % Lower/upper bounds of this time segment
    if iSeg == 1
        tStart = 1; 
    else
        tStart = tEnd + 1;
    end
    tEnd = (tStart + segSize)-1;
    if tEnd > minLength
        d = tEnd - minLength;
        tEnd = minLength;
        warning('tEnd is %g s beyond the last sample. Replacing with last sample.', round(d*256,1))
    end
    tSeg = tStart:tEnd;  % time index in samples for this segment

    % Kurtosis
    front_good_KURT(iSeg,:) = kurtosis(front_good(tSeg));
    front_bad_KURT(iSeg,:) = kurtosis(front_bad(tSeg));
    post_good_KURT(iSeg,:) = kurtosis(post_good(tSeg));
    post_bad_KURT(iSeg,:) = kurtosis(post_bad(tSeg));

    % RMS raw signal
    front_good_RMS(iSeg,:) = rms(front_good(tSeg));
    front_bad_RMS(iSeg,:) = rms(front_bad(tSeg));
    post_good_RMS(iSeg,:) = rms(post_good(tSeg));
    post_bad_RMS(iSeg,:) = rms(post_bad(tSeg));

    % Peak to RMS
    front_good_peak2RMS(iSeg,:) = peak2rms(front_good(tSeg));
    front_bad_peak2RMS(iSeg,:) = peak2rms(front_bad(tSeg));
    post_good_peak2RMS(iSeg,:) = peak2rms(post_good(tSeg));
    post_bad_peak2RMS(iSeg,:) = peak2rms(post_bad(tSeg));

    % Skewness raw signal
    front_good_SKEW(iSeg,:) = skewness(front_good(tSeg));
    front_bad_SKEW(iSeg,:) = skewness(front_bad(tSeg));
    post_good_SKEW(iSeg,:) = skewness(post_good(tSeg));
    post_bad_SKEW(iSeg,:) = skewness(post_bad(tSeg));

    % Compute rms of HF power
    tmp = get_psd(front_good(tSeg),256,'hamming',50,[],256,[70 100],'psd');
    front_good_HF(iSeg,:) = rms(tmp);
    tmp = get_psd(front_bad(tSeg),256,'hamming',50,[],256,[50 100],'psd');
    front_bad_HF(iSeg,:) = rms(tmp);
    tmp = get_psd(post_good(tSeg),256,'hamming',50,[],256,[50 100],'psd');
    post_good_HF(iSeg,:) = rms(tmp);
    tmp = get_psd(post_bad(tSeg),256,'hamming',50,[],256,[50 100],'psd');
    post_bad_HF(iSeg,:) = rms(tmp);

    % Compute rms of LF power
    tmp = get_psd(front_good(tSeg),256,'hamming',50,[],256,[0.2 3],'psd');
    front_good_LF(iSeg,:) = rms(tmp);
    tmp = get_psd(front_bad(tSeg),256,'hamming',50,[],256,[0.2 3],'psd');
    front_bad_LF(iSeg,:) = rms(tmp);
    tmp = get_psd(post_good(tSeg),256,'hamming',50,[],256,[0.2 3],'psd');
    post_good_LF(iSeg,:) = rms(tmp);
    tmp = get_psd(post_bad(tSeg),256,'hamming',50,[],256,[0.2 3],'psd');
    post_bad_LF(iSeg,:) = rms(tmp);

    % High-frequency EM SNR
    tmp = filtfilt_fast(b,1, front_good(tSeg)');
    front_good_SNR(iSeg,:) = mad(front_good(tSeg) - tmp');
    tmp = filtfilt_fast(b,1, front_bad(tSeg)');
    front_bad_SNR(iSeg,:) = mad(front_bad(tSeg) - tmp');
    tmp = filtfilt_fast(b,1, post_good(tSeg)');
    post_good_SNR(iSeg,:) = mad(post_good(tSeg) - tmp');
    tmp = filtfilt_fast(b,1, post_bad(tSeg)');
    post_bad_SNR(iSeg,:) = mad(post_bad(tSeg) - tmp');
    
    % Sample entropy
    front_good_SAMP(iSeg,:) = compute_se(front_good(tSeg));
    front_bad_SAMP(iSeg,:) = compute_se(front_bad(tSeg));
    post_good_SAMP(iSeg,:) = compute_se(post_good(tSeg));
    post_bad_SAMP(iSeg,:) = compute_se(post_bad(tSeg));

    % Approximate entropy
    front_good_APP(iSeg,:) = compute_ae(front_good(tSeg));
    front_bad_APP(iSeg,:) = compute_ae(front_bad(tSeg));
    post_good_APP(iSeg,:) = compute_ae(post_good(tSeg));
    post_bad_APP(iSeg,:) = compute_ae(post_bad(tSeg));

    % Fuzzy entropy
    front_good_FUZ(iSeg,:) = compute_fe(front_good(tSeg));
    front_bad_FUZ(iSeg,:) = compute_fe(front_bad(tSeg));
    post_good_FUZ(iSeg,:) = compute_fe(post_good(tSeg));
    post_bad_FUZ(iSeg,:) = compute_fe(post_bad(tSeg));
    
    progressbar(iSeg/(nSeg-1));

end
fprintf('Done. \n'); gong

% Convert outliers to mean to increase accuracy and preserve data length
% (good for quantiles but not necessary for Yuen t-tests)
outliers = isoutlier(front_good_RMS,'gesd');
front_good_RMS(outliers) = mean(front_good_RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_RMS,'gesd'); 
front_bad_RMS(outliers) = mean(front_bad_RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_peak2RMS,'gesd'); 
front_good_peak2RMS(outliers) = mean(front_good_peak2RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_peak2RMS,'gesd'); 
front_bad_peak2RMS(outliers) = mean(front_bad_peak2RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_SKEW,'gesd'); 
front_good_SKEW(outliers) = mean(front_good_SKEW);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_SKEW,'gesd');
front_bad_SKEW(outliers) = mean(front_bad_SKEW);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_HF,'gesd');
front_good_HF(outliers) = mean(front_good_HF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_HF,'gesd');
front_bad_HF(outliers) = mean(front_bad_HF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_LF,'gesd');
front_good_LF(outliers) = mean(front_good_LF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_LF,'gesd');
front_bad_LF(outliers) = mean(front_bad_LF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_SNR,'gesd');
front_good_SNR(outliers) = mean(front_good_SNR);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_SNR,'gesd');
front_bad_SNR(outliers) = mean(front_bad_SNR);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_SAMP,'gesd');
front_good_SAMP(outliers) = mean(front_good_SAMP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_SAMP,'gesd');
front_bad_SAMP(outliers) = mean(front_bad_SAMP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_APP,'gesd');
front_good_APP(outliers) = mean(front_good_APP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_APP,'gesd');
front_bad_APP(outliers) = mean(front_bad_APP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_good_FUZ,'gesd');
front_good_FUZ(outliers) = mean(front_good_FUZ);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(front_bad_FUZ,'gesd');
front_bad_FUZ(outliers) = mean(front_bad_FUZ);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_RMS,'gesd');
post_good_RMS(outliers) = mean(post_good_RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_RMS,'gesd'); 
post_bad_RMS(outliers) = mean(post_bad_RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_peak2RMS,'gesd'); 
post_good_peak2RMS(outliers) = mean(post_good_peak2RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_peak2RMS,'gesd'); 
post_bad_peak2RMS(outliers) = mean(post_bad_peak2RMS);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_SKEW,'gesd'); 
post_good_SKEW(outliers) = mean(post_good_SKEW);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_SKEW,'gesd');
post_bad_SKEW(outliers) = mean(post_bad_SKEW);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_HF,'gesd');
post_good_HF(outliers) = mean(post_good_HF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_HF,'gesd');
post_bad_HF(outliers) = mean(post_bad_HF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_LF,'gesd');
post_good_LF(outliers) = mean(post_good_LF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_LF,'gesd');
post_bad_LF(outliers) = mean(post_bad_LF);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_SNR,'gesd');
post_good_SNR(outliers) = mean(post_good_SNR);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_SNR,'gesd');
post_bad_SNR(outliers) = mean(post_bad_SNR);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_SAMP,'gesd');
post_good_SAMP(outliers) = mean(post_good_SAMP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_SAMP,'gesd');
post_bad_SAMP(outliers) = mean(post_bad_SAMP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_APP,'gesd');
post_good_APP(outliers) = mean(post_good_APP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_bad_APP,'gesd');
post_bad_APP(outliers) = mean(post_bad_APP);
fprintf('Adjusting %g outliers. \n', sum(outliers));
outliers = isoutlier(post_good_FUZ,'gesd');
post_good_FUZ(outliers) = mean(post_good_FUZ);
outliers = isoutlier(post_bad_FUZ,'gesd');
fprintf('Adjusting %g outliers. \n', sum(outliers));
post_bad_FUZ(outliers) = mean(post_bad_FUZ);

save(fullfile(dataPath, 'front_good_RMS'), 'front_good_RMS')
save(fullfile(dataPath, 'front_bad_RMS'), 'front_bad_RMS')
save(fullfile(dataPath, 'front_good_peak2RMS'), 'front_good_peak2RMS')
save(fullfile(dataPath, 'front_bad_peak2RMS'), 'front_bad_peak2RMS')
save(fullfile(dataPath, 'front_good_SKEW'), 'front_good_SKEW')
save(fullfile(dataPath, 'front_bad_SKEW'), 'front_bad_SKEW')
save(fullfile(dataPath, 'front_good_HF'), 'front_good_HF')
save(fullfile(dataPath, 'front_bad_HF'), 'front_bad_HF')
save(fullfile(dataPath, 'front_good_LF'), 'front_good_LF')
save(fullfile(dataPath, 'front_bad_LF'), 'front_bad_LF')
save(fullfile(dataPath, 'front_good_SNR'), 'front_good_SNR')
save(fullfile(dataPath, 'front_bad_SNR'), 'front_bad_SNR')
save(fullfile(dataPath, 'front_good_SAMP'), 'front_good_SAMP')
save(fullfile(dataPath, 'front_bad_SAMP'), 'front_bad_SAMP')
save(fullfile(dataPath, 'front_good_APP'), 'front_good_APP')
save(fullfile(dataPath, 'front_bad_APP'), 'front_bad_APP')
save(fullfile(dataPath, 'front_good_FUZ'), 'front_good_FUZ')
save(fullfile(dataPath, 'front_bad_FUZ'), 'front_bad_FUZ')
save(fullfile(dataPath, 'post_good_RMS'), 'post_good_RMS')
save(fullfile(dataPath, 'post_bad_RMS'), 'post_bad_RMS')
save(fullfile(dataPath, 'post_good_peak2RMS'), 'post_good_peak2RMS')
save(fullfile(dataPath, 'post_bad_peak2RMS'), 'post_bad_peak2RMS')
save(fullfile(dataPath, 'post_good_SKEW'), 'post_good_SKEW')
save(fullfile(dataPath, 'post_bad_SKEW'), 'post_bad_SKEW')
save(fullfile(dataPath, 'post_good_HF'), 'post_good_HF')
save(fullfile(dataPath, 'post_bad_HF'), 'post_bad_HF')
save(fullfile(dataPath, 'post_good_LF'), 'post_good_LF')
save(fullfile(dataPath, 'post_bad_LF'), 'post_bad_LF')
save(fullfile(dataPath, 'post_good_SNR'), 'post_good_SNR')
save(fullfile(dataPath, 'post_bad_SNR'), 'post_bad_SNR')
save(fullfile(dataPath, 'post_good_SAMP'), 'post_good_SAMP')
save(fullfile(dataPath, 'post_bad_SAMP'), 'post_bad_SAMP')
save(fullfile(dataPath, 'post_good_APP'), 'post_good_APP')
save(fullfile(dataPath, 'post_bad_APP'), 'post_bad_APP')
save(fullfile(dataPath, 'post_good_FUZ'), 'post_good_FUZ')
save(fullfile(dataPath, 'post_bad_FUZ'), 'post_bad_FUZ')

% Histograms
figure('color','w');
subplot(3,3,1)
histogram(front_good_KURT); hold on; histogram(front_bad_KURT); title('Kurtosis (frontal)'); legend('good', 'bad')
subplot(3,3,2)
histogram(front_good_RMS); hold on; histogram(front_bad_RMS); title('RMS');
subplot(3,3,3)
histogram(front_good_peak2RMS); hold on; histogram(front_bad_peak2RMS); title('Peak-to-RMS');
subplot(3,3,4)
histogram(front_good_SKEW); hold on; histogram(front_bad_SKEW); title('Skewness');
subplot(3,3,5)
histogram(front_good_HF); hold on; histogram(front_bad_HF); title('High-frequency');
subplot(3,3,6)
histogram(front_good_LF); hold on; histogram(front_bad_LF); title('LF power');
subplot(3,3,7)
histogram(front_good_SAMP); hold on; histogram(front_bad_SAMP); title('Sample entropy');
subplot(3,3,8)
histogram(front_good_APP); hold on; histogram(front_bad_APP); title('Approximate entropy');
subplot(3,3,9)
histogram(front_good_FUZ); hold on; histogram(front_bad_FUZ); title('Fuzzy entropy');
print(gcf, fullfile(dataPath, 'histo_frontal.png'),'-dpng','-r300');   % 300 dpi .png

figure('color','w');
subplot(3,3,1)
histogram(post_good_KURT); hold on; histogram(post_bad_KURT); title('Kurtosis (posterior)'); legend('good', 'bad')
subplot(3,3,2)
histogram(post_good_RMS); hold on; histogram(post_bad_RMS); title('RMS');
subplot(3,3,3)
histogram(post_good_peak2RMS); hold on; histogram(post_bad_peak2RMS); title('Peak-to-RMS');
subplot(3,3,4)
histogram(post_good_SKEW); hold on; histogram(post_bad_SKEW); title('Skewness');
subplot(3,3,5)
histogram(post_good_HF); hold on; histogram(post_bad_HF); title('High-frequency');
subplot(3,3,6)
histogram(post_good_LF); hold on; histogram(post_bad_LF); title('LF power');
subplot(3,3,7)
histogram(post_good_SAMP); hold on; histogram(post_bad_SAMP); title('Sample entropy');
subplot(3,3,8)
histogram(post_good_APP); hold on; histogram(post_bad_APP); title('Approximate entropy');
subplot(3,3,9)
histogram(post_good_FUZ); hold on; histogram(post_bad_FUZ); title('Fuzzy entropy');
print(gcf, fullfile(dataPath, 'histo_posterior.png'),'-dpng','-r300');   % 300 dpi .png
