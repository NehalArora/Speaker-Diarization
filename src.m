final code:
clc; clear; close all;

%% 1. INPUT ACQUISITION
[audio, fs] = audioread('meeting.wav');
if fs ~= 16000
    audio = resample(audio, 16000, fs);
    fs = 16000;
end
if size(audio,2) > 1
    audio = mean(audio,2);
end
fprintf('Audio Loaded: %.2f sec, Fs = %d Hz\n', length(audio)/fs, fs);

figure('Name','Original Audio','Position',[100 100 1000 200]);
plot((1:length(audio))/fs, audio);
xlabel('Time (s)'); ylabel('Amplitude'); title('Original Audio Signal');
grid on;

%% 2. DSP PREPROCESSING
% 2.1 Noise reduction using Wiener filter
audio_denoised = wiener2(audio, [5 1]);

figure('Name','Denoised Audio','Position',[100 300 1000 200]);
plot((1:length(audio_denoised))/fs, audio_denoised);
xlabel('Time (s)'); ylabel('Amplitude'); title('After Wiener Noise Reduction');
grid on;

% 2.2 Framing (20 ms Hamming windows, 10 ms overlap)
frameLen = round(0.02 * fs);
hop = round(0.01 * fs);
frames = buffer(audio_denoised, frameLen, frameLen - hop, 'nodelay');
frames = frames .* hamming(frameLen);

%% 3. FEATURE EXTRACTION
% Compute spectrogram
win = hamming(frameLen);
[S,F,T] = spectrogram(audio_denoised, win, frameLen - hop, frameLen, fs);

% Convert to power spectrum
powerSpec = abs(S).^2;

% Create mel filterbank manually
numMelBands = 13;
numFFT = frameLen;
lowFreq = 0;
highFreq = fs / 2;
melLow = 2595 * log10(1 + lowFreq / 700);
melHigh = 2595 * log10(1 + highFreq / 700);
melPoints = linspace(melLow, melHigh, numMelBands + 2);
hzPoints = 700 * (10.^(melPoints / 2595) - 1);
bin = floor((numFFT + 1) * hzPoints / fs);

% Build filterbank matrix
filterBank = zeros(numMelBands, numFFT/2 + 1);
for m = 2:numMelBands+1
    f_m_minus = bin(m - 1);
    f_m = bin(m);
    f_m_plus = bin(m + 1);
    for k = f_m_minus:f_m
        filterBank(m-1, k+1) = (k - f_m_minus) / (f_m - f_m_minus);
    end
    for k = f_m:f_m_plus
        filterBank(m-1, k+1) = (f_m_plus - k) / (f_m_plus - f_m);
    end
end

% Apply mel filterbank
melEnergies = filterBank * powerSpec(1:numFFT/2+1, :);

% Take log and DCT to get MFCCs
logMel = log(melEnergies + eps);
coeffs = dct(logMel);
coeffs = coeffs(1:13,:)'; % Keep first 13 coefficients and transpose

figure('Name','MFCC Features','Position',[100 550 1000 250]);
imagesc(coeffs');
axis xy; colorbar;
xlabel('Frame Index'); ylabel('MFCC Coeff Index'); title('MFCC Feature Matrix');
% 3.2 Pitch Detection
numFrames = size(frames,2);
pitch = zeros(numFrames,1);
for i = 1:numFrames
    x = frames(:,i);
    r = xcorr(x);
    r = r(round(length(r)/2):end);
    [~, locs] = findpeaks(r);
    if ~isempty(locs)
        pitch(i) = fs / locs(1);
    else
        pitch(i) = 0;
    end
end

% Match dimensions before concatenation
minFrames = min(size(coeffs,1), length(pitch));
coeffs = coeffs(1:minFrames, :);
pitch = pitch(1:minFrames);

features = [coeffs, pitch];

figure('Name','Pitch Contour','Position',[100 850 1000 200]);
plot(pitch);
xlabel('Frame Index'); ylabel('Pitch (Hz)'); title('Estimated Pitch');
grid on;

%% 4. SPEAKER SEGMENTATION (VAD + BIC)
energy = sum(frames.^2);
threshold = 0.2 * max(energy);
speechIdx = find(energy > threshold);
features_vad = features(speechIdx,:);

figure('Name','Energy-based VAD','Position',[100 1100 1000 200]);
plot(energy); hold on;
yline(threshold,'r--','Threshold');
xlabel('Frame Index'); ylabel('Energy'); title('Voice Activity Detection');
grid on;

%% 5. CLUSTERING (K-means)
numSpeakers = 4; % Updated for 4 speakers
[idx, C] = kmeans(features_vad, numSpeakers, 'Replicates',5);

%% 6. VISUALIZATION
frameTime = (speechIdx * hop) / fs;
colors = lines(numSpeakers);

figure('Name','Speaker Diarization Timeline','Position',[100 1350 1000 350]);
hold on;
for s = 1:numSpeakers
    mask = (idx == s);
    ydata = double(mask) * s;
    area(frameTime, ydata, 'FaceColor', colors(s,:), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
end
xlabel('Time (seconds)');
ylabel('Speaker ID');
title('Speaker Diarization Timeline');
yticks(1:numSpeakers);
yticklabels(arrayfun(@(x) sprintf('Speaker %d', x), 1:numSpeakers, 'UniformOutput', false));
legend(arrayfun(@(x) sprintf('Speaker %d', x), 1:numSpeakers, 'UniformOutput', false), 'Location','bestoutside');
grid on;

%% 7. PRINT SEGMENTS
fprintf('\n===== Speaker Segments (Estimated) =====\n');
for s = 1:numSpeakers
    spkMask = (idx == s);
    changePoints = find(diff([0; spkMask; 0]));
    starts = frameTime(changePoints(1:2:end-1));
    stops  = frameTime(changePoints(2:2:end)-1);
    for k = 1:length(starts)
        fprintf('Speaker %d: %.2f s to %.2f s\n', s, starts(k), stops(k));
        text((starts(k)+stops(k))/2, s, sprintf('S%d', s), ...
            'Color','k','FontWeight','bold','FontSize',8, ...
            'HorizontalAlignment','center');
    end
end
hold off;
