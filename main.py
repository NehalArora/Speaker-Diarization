import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from scipy.stats import mode
import collections
import webrtcvad


# =========================================================
# VAD FUNCTION (SOFT FILTER)
# =========================================================
def vad_filter_audio(audio_data, sr, aggressiveness=1):
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * 0.03)

    audio_int16 = (audio_data * 32767).astype(np.int16)
    speech_mask = []

    for i in range(0, len(audio_int16) - frame_len, frame_len):
        frame = audio_int16[i:i + frame_len]
        is_speech = vad.is_speech(frame.tobytes(), sr)
        speech_mask.extend([is_speech] * frame_len)

    speech_mask = np.array(speech_mask[:len(audio_data)])
    return speech_mask


# =========================================================
# LOAD AUDIO (FULL AUDIO - FIXED)
# =========================================================
wav, sr = librosa.load("meeting.wav", sr=16000)
print("Audio Loaded")

print("Audio length:", len(wav) / sr, "seconds")

# =========================================================
# APPLY VAD
# =========================================================
speech_mask = vad_filter_audio(wav, sr)
print("VAD Applied")
print("Speech ratio:", np.mean(speech_mask))


# =========================================================
# LOAD MODEL
# =========================================================
encoder = VoiceEncoder()


# =========================================================
# WINDOW SETTINGS
# =========================================================
window_size = int(0.8 * sr)   # smaller window for better resolution
step_size = int(0.2 * sr)

# Handle very short audio safely
if len(wav) < window_size:
    window_size = len(wav)
    step_size = len(wav) // 2


# =========================================================
# EMBEDDING GENERATION
# =========================================================
embeddings = []
timestamps = []

for start in range(0, len(wav) - window_size + 1, step_size):
    end = start + window_size
    segment = wav[start:end]

    # Soft VAD filtering
    if np.mean(speech_mask[start:end]) < 0.2:
        continue

    emb = encoder.embed_utterance(segment)
    embeddings.append(emb)
    timestamps.append((start / sr, end / sr))


# =========================================================
# FALLBACK IF VAD FAILS
# =========================================================
if len(embeddings) == 0:
    print("[WARNING] VAD removed all segments → fallback to raw audio")

    for start in range(0, len(wav) - window_size + 1, step_size):
        end = start + window_size
        segment = wav[start:end]

        emb = encoder.embed_utterance(segment)
        embeddings.append(emb)
        timestamps.append((start / sr, end / sr))

embeddings = np.array(embeddings)

print("Total embeddings:", len(embeddings))


# =========================================================
# NORMALIZE EMBEDDINGS
# =========================================================
embeddings = normalize(embeddings)


# =========================================================
# CLUSTERING (FORCE MINIMUM 2 SPEAKERS FOR DEMO)
# =========================================================
num_speakers = min(4, len(embeddings))

clustering = AgglomerativeClustering(
    n_clusters=num_speakers,
    metric='cosine',
    linkage='average'
)

labels = clustering.fit_predict(embeddings)

print("Unique labels:", set(labels))


# =========================================================
# EVALUATION
# =========================================================
print("\n===== Evaluation Metrics =====")

if len(set(labels)) > 1:
    sil = silhouette_score(embeddings, labels, metric='cosine')
    print(f"Silhouette Score: {sil:.3f}")
else:
    print("Silhouette Score: N/A")


# =========================================================
# SMOOTH LABELS (LIGHT SMOOTHING)
# =========================================================
smoothed = []

for i in range(len(labels)):
    start = max(0, i - 1)
    end = min(len(labels), i + 1)
    smoothed.append(mode(labels[start:end], keepdims=True)[0][0])

labels = np.array(smoothed)


# =========================================================
# MERGE SEGMENTS
# =========================================================
segments = []

current = labels[0]
start_time = timestamps[0][0]

for i in range(1, len(labels)):
    if labels[i] != current:
        segments.append((current, start_time, timestamps[i][0]))
        current = labels[i]
        start_time = timestamps[i][0]

segments.append((current, start_time, timestamps[-1][1]))


# =========================================================
# REMOVE SMALL SEGMENTS
# =========================================================
final_segments = []

for spk, s, e in segments:
    if e - s > 0.5:
        final_segments.append((spk, s, e))


# =========================================================
# OUTPUT
# =========================================================
print("\n===== Speaker Segments =====")

if len(final_segments) == 0:
    print("No valid segments found")
else:
    for spk, s, e in final_segments:
        print(f"Speaker {spk+1}: {s:.2f}s → {e:.2f}s")
