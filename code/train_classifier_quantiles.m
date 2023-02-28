%% Find best thresholds to classify bad/good channels automatically using
% qunatiles of distributions of extracted features. Accuracy is estimated
% using a confusion matrix. 
% 
% Custom method, requires to change thresholds manually many times. 
% 
% Cedric Cannard, feb 2022

clear; close all; clc
mainDir = 'C:\Users\Tracy\Documents\MATLAB\muse_checksignal';
outDir = fullfile(mainDir, 'outputs', 'muse_2016');

load(fullfile(outDir, 'front_good_RMS'), 'front_good_RMS')
load(fullfile(outDir, 'front_bad_RMS'), 'front_bad_RMS')
load(fullfile(outDir, 'front_good_peak2RMS'), 'front_good_peak2RMS')
load(fullfile(outDir, 'front_bad_peak2RMS'), 'front_bad_peak2RMS')
load(fullfile(outDir, 'front_good_SKEW'), 'front_good_SKEW')
load(fullfile(outDir, 'front_bad_SKEW'), 'front_bad_SKEW')
load(fullfile(outDir, 'front_good_HF'), 'front_good_HF')
load(fullfile(outDir, 'front_bad_HF'), 'front_bad_HF')
load(fullfile(outDir, 'front_good_LF'), 'front_good_LF')
load(fullfile(outDir, 'front_bad_LF'), 'front_bad_LF')
load(fullfile(outDir, 'front_good_SNR'), 'front_good_SNR')
load(fullfile(outDir, 'front_bad_SNR'), 'front_bad_SNR')
load(fullfile(outDir, 'front_good_SAMP'), 'front_good_SAMP')
load(fullfile(outDir, 'front_bad_SAMP'), 'front_bad_SAMP')
load(fullfile(outDir, 'front_good_APP'), 'front_good_APP')
load(fullfile(outDir, 'front_bad_APP'), 'front_bad_APP')
load(fullfile(outDir, 'front_good_FUZ'), 'front_good_FUZ')
load(fullfile(outDir, 'front_bad_FUZ'), 'front_bad_FUZ')
load(fullfile(outDir, 'post_good_RMS'), 'post_good_RMS')
load(fullfile(outDir, 'post_bad_RMS'), 'post_bad_RMS')
load(fullfile(outDir, 'post_good_peak2RMS'), 'post_good_peak2RMS')
load(fullfile(outDir, 'post_bad_peak2RMS'), 'post_bad_peak2RMS')
load(fullfile(outDir, 'post_good_SKEW'), 'post_good_SKEW')
load(fullfile(outDir, 'post_bad_SKEW'), 'post_bad_SKEW')
load(fullfile(outDir, 'post_good_HF'), 'post_good_HF')
load(fullfile(outDir, 'post_bad_HF'), 'post_bad_HF')
load(fullfile(outDir, 'post_good_LF'), 'post_good_LF')
load(fullfile(outDir, 'post_bad_LF'), 'post_bad_LF')
load(fullfile(outDir, 'post_good_SNR'), 'post_good_SNR')
load(fullfile(outDir, 'post_bad_SNR'), 'post_bad_SNR')
load(fullfile(outDir, 'post_good_SAMP'), 'post_good_SAMP')
load(fullfile(outDir, 'post_bad_SAMP'), 'post_bad_SAMP')
load(fullfile(outDir, 'post_good_APP'), 'post_good_APP')
load(fullfile(outDir, 'post_bad_APP'), 'post_bad_APP')
load(fullfile(outDir, 'post_good_FUZ'), 'post_good_FUZ')
load(fullfile(outDir, 'post_bad_FUZ'), 'post_bad_FUZ')


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
