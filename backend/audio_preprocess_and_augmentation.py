import torch
import torchaudio
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import simpleaudio as sa
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf

#torchaudio.set_audio_backend("soundfile")

class AudioDataset(Dataset):
    def __init__(self, file_path, transformation, target_sample_rate):
        # Load the CSV file with annotations (meta/esc50.csv)
        #self.annotations = pd.read_csv(annotations_file)
        self.file_path = file_path
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Get the file path and label from the CSV
        audio_sample_path = self.file_path
        label = 0
        signal, sr = torchaudio.load(audio_sample_path)

        # Resample if necessary
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            signal = resampler(signal)

        # Store the original signal before transformation for playback
        original_signal = signal.clone()

        # Apply transformations (e.g., MFCC)
        signal = self.transformation(signal)
        return signal, label, audio_sample_path, sr, original_signal, sr  # Include original signal for visualization

class PreProcessAudio(AudioDataset):
    def RemoveNoiseFromAudio(self):
        input_path = self.file_path
        signal, sr = librosa.load(input_path, sr=None)
        noise_sample = signal[0:int(0.5 * sr)]
        reduced_noise_signal = nr.reduce_noise(y=signal, y_noise=noise_sample, sr=sr)
        output_path = 'reduced_noise_output_audio.wav'
        sf.write(output_path, reduced_noise_signal, sr)

class AugmentAudio(AudioDataset):
    def ChangePitchOfAudio(self, n_steps = 4):
        input_path = self.file_path
        signal, sr = librosa.load(input_path, sr=None)
        shifted_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
        output_path = 'output_audio_pitch_shifted.wav'
        sf.write(output_path, shifted_signal, sr)

# Define transformations
transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

# Create Dataset and DataLoader
audio_dataset = AudioDataset(
    file_path = "Gaatri.wav",
    transformation=transformation,
    target_sample_rate=16000
)
audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=True)

preprocess_audio_dataset = PreProcessAudio(
    file_path = "Gaatri.wav",
    transformation=transformation,
    target_sample_rate=16000
)
preprocess_audio_loader = DataLoader(preprocess_audio_dataset, batch_size=1, shuffle=True)

augment_audio_dataset = AugmentAudio(
    file_path = "Gaatri.wav",
    transformation=transformation,
    target_sample_rate=16000
)
augment_audio_loader = DataLoader(augment_audio_dataset, batch_size=1, shuffle=True)

# Function to plot and display the audio widget (no playback)
def visualize_audio(signal, sr, audio_sample_path, original_signal, original_sr):
    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()  # Take the first channel

    # Save the waveform in 16-bit PCM format
    output_path = 'output_audio_int16.wav'
    torchaudio.save(output_path, original_signal, sr, encoding="PCM_S", bits_per_sample=16)

    wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    play_obj = wave_obj.play()

    # Wait for the playback to finish
    play_obj.wait_done()

    # Plot the MFCC (features on y-axis, time steps on x-axis)
    #plt.figure(figsize=(10, 4))
    #plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
    #plt.colorbar(format="%+2.0f dB")
    #plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
    #plt.xlabel("Time")
    #plt.ylabel("MFCC Coefficients")
    #plt.show()

    # Display the audio as an array (do not attempt to play it)
    #print(f"Audio data (original signal): {original_signal.numpy().squeeze()}")

# Function to plot and display the audio widget (no playback)
def visualize_preprocess_audio(signal, sr, audio_sample_path, original_signal, original_sr):
    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()  # Take the first channel

    preprocess_audio_dataset.RemoveNoiseFromAudio();

    output_path = "reduced_noise_output_audio.wav"
    wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    play_obj = wave_obj.play()

    # Wait for the playback to finish
    play_obj.wait_done()

    # Plot the MFCC (features on y-axis, time steps on x-axis)
    #plt.figure(figsize=(10, 4))
    #plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
    #plt.colorbar(format="%+2.0f dB")
    #plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
    #plt.xlabel("Time")
    #plt.ylabel("MFCC Coefficients")
    #plt.show()

    # Display the audio as an array (do not attempt to play it)
    #print(f"Audio data (original signal): {original_signal.numpy().squeeze()}")

def visualize_augment_audio(signal, sr, audio_sample_path, original_signal, original_sr):
    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()  # Take the first channel

    augment_audio_dataset.ChangePitchOfAudio();

    output_path = "output_audio_pitch_shifted.wav"
    wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    play_obj = wave_obj.play()

    # Wait for the playback to finish
    play_obj.wait_done()

    # Plot the MFCC (features on y-axis, time steps on x-axis)
    #plt.figure(figsize=(10, 4))
    #plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
    #plt.colorbar(format="%+2.0f dB")
    #plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
    #plt.xlabel("Time")
    #plt.ylabel("MFCC Coefficients")
    #plt.show()

    # Display the audio as an array (do not attempt to play it)
    #print(f"Audio data (original signal): {original_signal.numpy().squeeze()}")

# Iterate through the dataset
signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(audio_loader))
# Visualize the first audio sample in the batch
visualize_audio(signal[0], sr, audio_sample_path[0], original_signal[0], original_sr)


signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(preprocess_audio_loader))
# Visualize the first audio sample in the batch
visualize_preprocess_audio(signal[0], sr, audio_sample_path[0], original_signal[0], original_sr)

signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(augment_audio_loader))
# Visualize the first audio sample in the batch
visualize_augment_audio(signal[0], sr, audio_sample_path[0], original_signal[0], original_sr)