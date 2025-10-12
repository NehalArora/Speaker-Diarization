# 🎙️ Speaker Diarization using DSP and MFCC Features

## Project Overview
This MATLAB project implements a **Speaker Diarization System**, which segments an input audio recording into regions corresponding to individual speakers. It uses **DSP techniques**, **MFCC feature extraction**, **Pitch detection**, **Voice Activity Detection (VAD)**, and **K-means clustering** to estimate speakers and their speech intervals.

---

## Workflow Summary

### 1. Input Acquisition
- Reads audio file (`meeting.wav`), resamples to **16 kHz**, converts stereo to mono.
- Plots the original waveform for visualization.

### 2. DSP Preprocessing
- **Noise Reduction:** Wiener filter (`wiener2`) to suppress background noise.
- **Framing:** 20 ms Hamming windows with 10 ms overlap.

### 3. Feature Extraction
- **MFCC Calculation:**  
  - Compute power spectrum using STFT.  
  - Apply Mel filterbank.  
  - Log compression and DCT to extract 13 MFCC coefficients.
- **Pitch Detection:** Autocorrelation-based pitch per frame.
- **Feature Vector:** Combines MFCC and pitch features.

### 4. Voice Activity Detection (VAD)
- Energy-based thresholding isolates active speech frames.

### 5. Clustering
- **K-means clustering** segments speech frames into speaker clusters.
- Number of speakers is configurable (default = 4).

### 6. Visualization
- Plots include:  
  1. Denoised audio signal  
  2. MFCC feature matrix  
  3. Speaker diarization timeline (color-coded segments per speaker)

### 7. Output
- Estimated speaker segments printed in the command window:
```
Speaker 1: 31.50 s to 43.27 s
Speaker 2: 67.42 s to 67.44 s
Speaker 3: 26.13 s to 26.14 s
Speaker 4: 22.49 s to 23.89 s
```


---

## Core DSP Techniques Used
- **Wiener Filtering** – Noise suppression
- **Framing & Windowing** – Short-term spectral analysis
- **MFCC Extraction** – Speaker-specific features
- **Pitch Estimation** – Prosodic features for separation
- **Energy-based VAD** – Isolate speech frames
- **K-means Clustering** – Group frames by acoustic similarity

---

## System Requirements
- MATLAB R2021a or later
- Signal Processing Toolbox
- Audio Toolbox (optional)

---

## Future Enhancements
- Replace K-means with **GMM** or **Spectral Clustering** for improved accuracy
- Integrate **Deep Learning embeddings** (x-vectors, d-vectors)
- Implement **real-time diarization** using sliding window inference
