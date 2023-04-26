import os
import glob
import torchaudio
import numpy as np
import librosa

# Set the audio parameters
sample_rate = 16000
segment_length = 3  # seconds
hop_length = 160  # 10ms

# Load the audio files
data_dir = "./data/cv-corpus-5.1-2020-06-22/en"
audio_files = glob.glob(os.path.join(data_dir, "clips", "*.mp3"))

# Resample the audio to a consistent sample rate
def resample(audio, sr):
    return torchaudio.transforms.Resample(audio.shape[1], sr)(audio)

# Split the audio into segments
def split_audio(audio, length, hop):
    num_samples = length * sample_rate
    hop_length = hop * sample_rate
    samples = []
    for i in range(0, audio.shape[1] - num_samples, hop_length):
        samples.append(audio[:, i:i+num_samples])
    return samples

# Extract features from the audio
def extract_features(audio, sr):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=80,
        f_min=0,
        f_max=8000,
    )(audio)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec

# Preprocess the audio data
preprocessed_data = []
for i, audio_file in enumerate(audio_files):
    print(f"Processing audio file {i+1}/{len(audio_files)}...")
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
    audio = resample(audio.reshape(1, -1), sample_rate)
    segments = split_audio(audio, segment_length, hop_length)
    for segment in segments:
        features = extract_features(segment, sample_rate)
        preprocessed_data.append(features.numpy())
preprocessed_data = np.array(preprocessed_data)
