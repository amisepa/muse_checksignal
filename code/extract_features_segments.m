%% Extract a set of EEG features from Muse raw data labeled manually.
% Frontal and posterior channels are considered separately.
%
% Cedric Cannard, Feb 2023

clear; close all; clc
mainDir = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal';
dataDir = 'G:\Shared drives\Science\IDL\5. DATA\muse\eeg\tagged_data';
outDir = fullfile(mainDir, 'outputs');

minfreq = 0.2;          % minimum frequency for LF power
segSize = 1/minfreq     % in s (t = 1/f)
fs = 256;               % sample rate

% Load data
load(fullfile(dataDir, 'front_good.mat'));
load(fullfile(dataDir, 'front_bad.mat'));
load(fullfile(dataDir, 'post_good.mat'));
load(fullfile(dataDir, 'post_bad.mat'));

% Remove NaNs
front_good(isnan(front_good)) = [];
front_bad(isnan(front_bad)) = [];
post_good(isnan(post_good)) = [];
post_bad(isnan(post_bad)) = [];

% Take max number of samples from shortest file
minLength = min([ length(front_good)  length(front_bad) length(post_good) length(post_bad)]);
minLength = minLength / 2; % to divide in 2 halves
fprintf('training/validation sets are %g minutes long. \n', round(minLength/256/60,1));

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
outliers = isoutlier(front_bad_RMS,'gesd'); 
front_bad_RMS(outliers) = mean(front_bad_RMS);
outliers = isoutlier(front_good_peak2RMS,'gesd'); 
front_good_peak2RMS(outliers) = mean(front_good_peak2RMS);
outliers = isoutlier(front_bad_peak2RMS,'gesd'); 
front_bad_peak2RMS(outliers) = mean(front_bad_peak2RMS);
outliers = isoutlier(front_good_SKEW,'gesd'); 
front_good_SKEW(outliers) = mean(front_good_SKEW);
outliers = isoutlier(front_bad_SKEW,'gesd');
front_bad_SKEW(outliers) = mean(front_bad_SKEW);
outliers = isoutlier(front_good_HF,'gesd');
front_good_HF(outliers) = mean(front_good_HF);
outliers = isoutlier(front_bad_HF,'gesd');
front_bad_HF(outliers) = mean(front_bad_HF);
outliers = isoutlier(front_good_LF,'gesd');
front_good_LF(outliers) = mean(front_good_LF);
outliers = isoutlier(front_bad_LF,'gesd');
front_bad_LF(outliers) = mean(front_bad_LF);
outliers = isoutlier(front_good_SNR,'gesd');
front_good_SNR(outliers) = mean(front_good_SNR);
outliers = isoutlier(front_bad_SNR,'gesd');
front_bad_SNR(outliers) = mean(front_bad_SNR);
outliers = isoutlier(front_good_SAMP,'gesd');
front_good_SAMP(outliers) = mean(front_good_SAMP);
outliers = isoutlier(front_bad_SAMP,'gesd');
front_bad_SAMP(outliers) = mean(front_bad_SAMP);
outliers = isoutlier(front_good_APP,'gesd');
front_good_APP(outliers) = mean(front_good_APP);
outliers = isoutlier(front_bad_APP,'gesd');
front_bad_APP(outliers) = mean(front_bad_APP);
outliers = isoutlier(front_good_FUZ,'gesd');
front_good_FUZ(outliers) = mean(front_good_FUZ);
outliers = isoutlier(front_bad_FUZ,'gesd');
front_bad_FUZ(outliers) = mean(front_bad_FUZ);
outliers = isoutlier(post_good_RMS,'gesd');
post_good_RMS(outliers) = mean(post_good_RMS);
outliers = isoutlier(post_bad_RMS,'gesd'); 
post_bad_RMS(outliers) = mean(post_bad_RMS);
outliers = isoutlier(post_good_peak2RMS,'gesd'); 
post_good_peak2RMS(outliers) = mean(post_good_peak2RMS);
outliers = isoutlier(post_bad_peak2RMS,'gesd'); 
post_bad_peak2RMS(outliers) = mean(post_bad_peak2RMS);
outliers = isoutlier(post_good_SKEW,'gesd'); 
post_good_SKEW(outliers) = mean(post_good_SKEW);
outliers = isoutlier(post_bad_SKEW,'gesd');
post_bad_SKEW(outliers) = mean(post_bad_SKEW);
outliers = isoutlier(post_good_HF,'gesd');
post_good_HF(outliers) = mean(post_good_HF);
outliers = isoutlier(post_bad_HF,'gesd');
post_bad_HF(outliers) = mean(post_bad_HF);
outliers = isoutlier(post_good_LF,'gesd');
post_good_LF(outliers) = mean(post_good_LF);
outliers = isoutlier(post_bad_LF,'gesd');
post_bad_LF(outliers) = mean(post_bad_LF);
outliers = isoutlier(post_good_SNR,'gesd');
post_good_SNR(outliers) = mean(post_good_SNR);
outliers = isoutlier(post_bad_SNR,'gesd');
post_bad_SNR(outliers) = mean(post_bad_SNR);
outliers = isoutlier(post_good_SAMP,'gesd');
post_good_SAMP(outliers) = mean(post_good_SAMP);
outliers = isoutlier(post_bad_SAMP,'gesd');
post_bad_SAMP(outliers) = mean(post_bad_SAMP);
outliers = isoutlier(post_good_APP,'gesd');
post_good_APP(outliers) = mean(post_good_APP);
outliers = isoutlier(post_bad_APP,'gesd');
post_bad_APP(outliers) = mean(post_bad_APP);
outliers = isoutlier(post_good_FUZ,'gesd');
post_good_FUZ(outliers) = mean(post_good_FUZ);
outliers = isoutlier(post_bad_FUZ,'gesd');
post_bad_FUZ(outliers) = mean(post_bad_FUZ);

% save(fullfile(outDir, 'post_bad_FUZ'), 'post_bad_FUZ')

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
print(gcf, fullfile(outDir, 'histo_frontal.png'),'-dpng','-r300');   % 300 dpi .png

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
print(gcf, fullfile(outDir, 'histo_posterior.png'),'-dpng','-r300');   % 300 dpi .png

%% Quantiles
var_front = { 'front_good_KURT' 'front_bad_KURT' 'front_good_RMS' 'front_bad_RMS' ...
    'front_good_peak2RMS' 'front_bad_peak2RMS' 'front_good_SKEW' 'front_bad_SKEW' ...
    'front_good_HF' 'front_bad_HF' 'front_good_LF' 'front_bad_LF' 'front_good_SNR' ...
    'front_bad_SNR' 'front_good_SAMP' 'front_bad_SAMP' 'front_good_APP' ...
    'front_bad_APP' 'front_good_FUZ' 'front_bad_FUZ' };
var_post = { 'post_good_KURT' 'post_bad_KURT' 'post_good_RMS' 'post_bad_RMS'...
    'post_good_peak2RMS' 'post_bad_peak2RMS' 'post_good_SKEW' 'post_bad_SKEW'...
    'post_good_HF' 'post_bad_HF' 'post_good_LF' 'post_bad_LF' 'post_good_SNR' ...
    'post_bad_SNR' 'post_good_SAMP' 'post_bad_SAMP' 'post_good_APP' ...
    'post_bad_APP' 'post_good_FUZ' 'post_bad_FUZ' };

% Open txt file
fid = fopen(fullfile(outDir, 'quantiles.txt'),'w');

% Results for frontal features
fprintf(fid, '------------------ FRONTAL ------------------\n');
for i = 1:2:length(var_front)
    fprintf(fid, '%s: \n', extractAfter(var_front{i},'good_') );
    x = eval(var_front{i});
    y = eval(var_front{i+1});
    [t,~,CI,~,~,p] = yuend(x,y,10,0.05);
    fprintf(fid, 'p-value = %g; t = %g; 95%% CI = [%g, %g] \n', round(p,2), round(t,1), round(CI(1),1), round(CI(2),1) );
    fprintf(fid, 'Quantiles (good): %s \n', num2str(quantile(x,4)));
    fprintf(fid, 'Quantiles (bad): %s \n', num2str(quantile(y,4)));
    if p < 0.05, fprintf(fid, '---> Significant! \n'); end
    fprintf(fid, '\n');
%     plotCI(x,y,10); 
%     print(gcf, fullfile(outDir,['ci_' extractAfter(var_front{i},'good_') '.png']),'-dpng','-r300'); 
%       close(gcf);
end

% Results for frontal features
fprintf(fid, '------------------ POSTERIOR ------------------\n');
for i = 1:2:length(var_post)
    fprintf(fid, '%s: \n', extractAfter(var_post{i},'good_') );
    x = eval(var_post{i});
    y = eval(var_post{i+1});
    [t,~,CI,~,~,p] = yuend(x,y,10,0.05);
    fprintf(fid, 'p-value = %g; t = %g; 95%% CI = [%g, %g] \n', round(p,2), round(t,1), round(CI(1),1), round(CI(2),1) );
    fprintf(fid, 'Quantiles (good): %s \n', num2str(quantile(x,4)));
    fprintf(fid, 'Quantiles (bad): %s \n', num2str(quantile(y,4)));
    if p < 0.05, fprintf(fid, '---> Significant! \n'); end
    fprintf(fid, '\n');
end

fclose(fid);
disp(['results saved in: ' fullfile(outDir, 'quantiles.txt') ])

%% Use the remaining files for VALIDATION using a confusion matrix

% Frontal thresholds (based on quantiles and 95% CIs)
threshF_rms = 9.3;      % bad if above --> 9.3 is best
threshF_lf = 8.5;       % bad if above --> 8.5 is best
threshF_samp = 1.31;    % bad if below --> 1.31 is best (below increases TPR, above increases TNR)
threshF_snr = 3;        % bad if above --> 3 

% Posterior thresholds (based on quantiles and 95% CIs)
threshP_rms = 13;       % bad if above --> 13 is best
threshP_lf = 14.3;      % bad if above --> 14.3 is best (range: 14-14.6)
threshP_samp = 1.4;     % bad if below --> 1.4 (try <.8 is bad to increase TPR)
threshP_snr = 4.5;     % bad if above --> 4.5 (range: 3-5)

% Input which features to use for validation 
feat = input("Type feature names (e.g., LF, SAMP, RMS, SNR, LF_SAMP): ", 's');

true_posF = 0; true_negF = 0;
false_posF = 0; false_negF = 0;
true_posP = 0; true_negP = 0;
false_posP = 0; false_negP = 0;
for iSeg = 1:nSeg-1
    
    fprintf('segment %g \n', iSeg);

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
    
    % ------------- FRONTAL ------------- %
    % LF
    if contains(feat, 'LF')
        tmp = get_psd(front_good2(tSeg),256,'hamming',50,[],256,[0.001 3],'psd');
        if rms(tmp) < threshF_lf
            true_negF = true_negF + 1;
        else 
            false_negF = false_negF + 1;
        end
        tmp = get_psd(front_bad2(tSeg),256,'hamming',50,[],256,[0.001 3],'psd');
        if rms(tmp) > threshF_lf
            true_posF = true_posF + 1;
        else 
            false_posF = false_posF + 1;
        end
    end

    % RMS
    if contains(feat, 'RMS')
        if rms(front_good2(tSeg)) < threshF_rms
            true_negF = true_negF + 1;
        else 
            false_negF = false_negF + 1;
        end
        if rms(front_bad2(tSeg)) > threshF_rms
            true_posF = true_posF + 1;
        else 
            false_posF = false_posF + 1;
        end
    end

    % Sample entropy
    if contains(feat, 'SAMP')
        tmp = compute_se(front_good2(tSeg));
        if tmp > threshF_samp   % reversed here
            true_negF = true_negF + 1;
        else 
            false_negF = false_negF + 1;
        end
        tmp = compute_se(front_bad2(tSeg));
        if tmp < threshF_samp
            true_posF = true_posF + 1;
        else 
            false_posF = false_posF + 1;
        end
    end

    % SNR
    if contains(feat, 'SNR')
        tmp = filtfilt_fast(b,1,front_good2(tSeg)');
        if mad(front_good2(tSeg) - tmp') < threshF_snr 
            true_negF = true_negF + 1;
        else 
            false_negF = false_negF + 1;
        end
        tmp = filtfilt_fast(b,1,front_bad2(tSeg)');
        if mad(front_bad2(tSeg) - tmp') > threshF_snr 
            true_posF = true_posF + 1;
        else 
            false_posF = false_posF + 1;
        end
    end

    % ------------- POSTERIOR ------------- %
    % LF
    if contains(feat, 'LF')
        tmp = get_psd(post_good2(tSeg),256,'hamming',50,[],256,[0.001 3],'psd');
        if rms(tmp) < threshP_lf
            true_negP = true_negP + 1;
        else
            false_negP = false_negP + 1;
        end
        tmp = get_psd(post_bad2(tSeg),256,'hamming',50,[],256,[0.001 3],'psd');
        if rms(tmp) > threshP_lf
            true_posP = true_posP + 1;
        else
            false_posP = false_posP + 1;
        end
    end

    % RMS
    if contains(feat, 'RMS')
        if rms(post_good2(tSeg)) < threshP_rms
            true_negP = true_negP + 1;
        else 
            false_negP = false_negP + 1;
        end
        if rms(post_bad2(tSeg)) > threshP_rms
            true_posP = true_posP + 1;
        else 
            false_posP = false_posP + 1;
        end
    end

    % Sample entropy
    if contains(feat, 'SAMP')
        tmp = compute_se(post_good2(tSeg));
        if tmp > threshP_samp   % reversed here
            true_negP = true_negP + 1;
        else
            false_negP = false_negP + 1;
        end
        tmp = compute_se(post_bad2(tSeg));
        if tmp < threshP_samp
            true_posP = true_posP + 1;
        else
            false_posP = false_posP + 1;
        end
    end
    
    % SNR
    if contains(feat, 'SNR')
        tmp = filtfilt_fast(b,1,post_good2(tSeg)');
        if mad(post_good2(tSeg) - tmp') < threshP_snr 
            true_negP = true_negP + 1;
        else 
            false_negP = false_negP + 1;
        end
        tmp = filtfilt_fast(b,1,post_bad2(tSeg)');
        if mad(post_bad2(tSeg) - tmp') > threshP_snr 
            true_posP = true_posP + 1;
        else 
            false_posP = false_posP + 1;
        end
    end

end
fprintf('Done. \n'); gong

% Save results in txt file
fid = fopen(fullfile(outDir, sprintf('validation_%s.txt',feat)),'w');
fprintf(fid, '------------------- FRONTAL ------------------- \n');
if contains(feat, 'LF'), fprintf(fid, 'thresh_lf = %g \n', threshF_lf); end
if contains(feat, 'SAMP'), fprintf(fid, 'thresh_samp = %g \n', threshF_samp); end
if contains(feat, 'RMS'), fprintf(fid, 'thresh_rms = %g \n', threshF_rms); end
if contains(feat, 'SNR'), fprintf(fid, 'thresh_snr = %g \n', threshF_snr); end
TPR = true_posF / (true_posF + false_negF); % True positive rate (sensitivity or hit rate)
TNR = true_negF / (true_negF + false_posF); % True negative rate (specificity or selectivity)
PPV = true_posF / (true_posF + false_posF); % Positive predictive value (precision)
FNR = false_negF / (false_negF + true_posF); % False negative rate (miss rate)
FDR = false_posF / (false_posF + true_posF); % False discovery rate
ACC = (true_posF + true_negF) / (true_posF + true_negF + false_posF + false_negF); % Accuracy
fprintf(fid,['True positive rate (sensitivity or hit rate): %g \n' ...
            'True negative rate (specificity or selectivity): %g \n' ...
            'Positive predictive value (precision): %g \n' ...
            'False negative rate (miss rate): %g \n' ...
            'False discovery rate: %g \n' ...
            'Accuracy: %g \n \n'], ...
            round(TPR*100,1), round(TNR*100,1), round(PPV*100,1), ...
            round(FNR*100,1), round(FDR*100,1), round(ACC*100,1));
fprintf(fid, '------------------- POSTERIOR ------------------- \n');
if contains(feat, 'LF'), fprintf(fid, 'thresh_lf = %g \n', threshP_lf); end
if contains(feat, 'SAMP'), fprintf(fid, 'thresh_samp = %g \n', threshP_samp); end
if contains(feat, 'RMS'), fprintf(fid, 'thresh_rms = %g \n', threshP_rms); end
if contains(feat, 'SNR'), fprintf(fid, 'thresh_snr = %g \n', threshP_snr); end
TPR = true_posP / (true_posP + false_negP); % True positive rate (sensitivity or hit rate)
TNR = true_negP / (true_negP + false_posP); % True negative rate (specificity or selectivity)
PPV = true_posP / (true_posP + false_posP); % Positive predictive value (precision)
FNR = false_negP / (false_negP + true_posP); % False negative rate (miss rate)
FDR = false_posP / (false_posP + true_posP); % False discovery rate
ACC = (true_posP + true_negP) / (true_posP + true_negP + false_posP + false_negP); % Accuracy
fprintf(fid,['True positive rate (sensitivity or hit rate): %g \n' ...
            'True negative rate (specificity or selectivity): %g \n' ...
            'Positive predictive value (precision): %g \n' ...
            'False negative rate (miss rate): %g \n' ...
            'False discovery rate: %g \n' ...
            'Accuracy: %g \n'], ...
            round(TPR*100,1), round(TNR*100,1), round(PPV*100,1), ...
            round(FNR*100,1), round(FDR*100,1), round(ACC*100,1));
fclose(fid);












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

%% Select predictors for Random Forest (needs response variable to be double vector)

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

