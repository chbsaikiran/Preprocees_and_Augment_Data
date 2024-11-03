from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Depends
import os
import traceback
import mimetypes

import text_preprocess_and_augmentation_main
from text_preprocess_and_augmentation_main import show_original_data,show_lowercase_data,\
show_after_remove_stop_words_data,show_after_synonym_replacement_data,show_after_random_insertion_data

import logging

import torch
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import simpleaudio as sa
import librosa
import librosa.display
import noisereduce as nr
import soundfile as sf

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

# Add at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

class InOutFileNames:
    def __init__(self):
        self.input_path = None

    def save_in_out_file_name(self, input_path: str):
        self.input_path = input_path
        logger.info(f"Input path set to: {self.input_path}")

    def get_in_out_file_name(self):
        if self.input_path:
            return self.input_path
        else:
            raise HTTPException(status_code=400, detail="No input path found")

# Global instance of InOutFileNames
InOutFileNames_obj = InOutFileNames()

@app.get("/")
async def read_root():
    return FileResponse("../frontend/templates/index.html")

@app.post("/api/save_in_out_file_name/")
async def save_in_out_file_name(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_upload_{file.filename}"
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        # Save the temporary file path
        InOutFileNames_obj.save_in_out_file_name(temp_file_path)
        logger.info(f"Saved file to: {temp_file_path}")
        
        return {"message": "File saved successfully", "file_path": temp_file_path}
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
async def preprocess_data(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.wav'):
            return {"type": "audio", "audioPath": "/api/audio/preprocessed"}
            
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            lowercase_result = show_lowercase_data(temp_file_path)
            stopwords_result = show_after_remove_stop_words_data(temp_file_path)
            
            return {
                "lowercase_data": lowercase_result,
                "after_remove_stop_words_data": stopwords_result
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment")
async def augment_data(file: UploadFile = File(...)):
    try:
        if file.filename.endswith('.wav'):
            return {"type": "audio", "audioPath": "/api/audio/augmented"}
            
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            synonym_result = show_after_synonym_replacement_data(temp_file_path)
            insertion_result = show_after_random_insertion_data(temp_file_path)
            
            return {
                "synonym_replacement_data": synonym_result,
                "random_insertion_data": insertion_result
            }
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/original")
async def original_data(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        
        if file.filename.endswith('.wav'):
            return {"type": "audio", "audioPath": "/api/audio/original"}
        
        # Create a temporary file with a unique name
        temp_file_path = f"temp_{file.filename}"
        
        try:
            contents = await file.read()
            with open(temp_file_path, "wb") as f:
                f.write(contents)
            
            result = show_original_data(temp_file_path)
            
            return {"output": result}
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            logger.error(traceback.format_exc())  # Log the full error traceback
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_in_out_file_name/")
async def get_in_out_file_name(InOutFileNames_objstore: InOutFileNames = Depends(lambda: InOutFileNames_obj)):
    input_path = InOutFileNames_objstore.get_in_out_file_name()
    return {"input_path": input_path}

# New endpoints to serve audio files
@app.get("/api/audio/original")
async def get_original_audio():
    # Define transformations
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
    
    # Create Dataset and DataLoader
    audio_dataset = AudioDataset(
        file_path = InOutFileNames_obj.get_in_out_file_name(),
        transformation=transformation,
        target_sample_rate=16000
    )
    audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=True)

    signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(audio_loader))

    # Save the waveform in 16-bit PCM format
    output_path = 'output_audio_int16.wav'
    torchaudio.save(output_path, original_signal[0], original_sr, encoding="PCM_S", bits_per_sample=16)

    #wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    #play_obj = wave_obj.play()

    # Wait for the playback to finish
    #play_obj.wait_done()

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="original.wav"
    )

@app.get("/api/audio/preprocessed")
async def get_preprocessed_audio():
    # Define transformations
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

    preprocess_audio_dataset = PreProcessAudio(
    file_path = InOutFileNames_obj.get_in_out_file_name(),
    transformation=transformation,
    target_sample_rate=16000
    )
    preprocess_audio_loader = DataLoader(preprocess_audio_dataset, batch_size=1, shuffle=True)

    signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(preprocess_audio_loader))

    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()  # Take the first channel

    preprocess_audio_dataset.RemoveNoiseFromAudio();

    output_path = "reduced_noise_output_audio.wav"
    #wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    #play_obj = wave_obj.play()

    # Wait for the playback to finish
    #play_obj.wait_done()

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="preprocessed.wav"
    )

@app.get("/api/audio/augmented")
async def get_augmented_audio():
    # Define transformations
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

    augment_audio_dataset = AugmentAudio(
    file_path = InOutFileNames_obj.get_in_out_file_name(),
    transformation=transformation,
    target_sample_rate=16000
    )
    augment_audio_loader = DataLoader(augment_audio_dataset, batch_size=1, shuffle=True)

    signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(augment_audio_loader))

    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()  # Take the first channel

    augment_audio_dataset.ChangePitchOfAudio();

    output_path = "output_audio_pitch_shifted.wav"
    #wave_obj = sa.WaveObject.from_wave_file(output_path)

    # Play the .wav file
    #play_obj = wave_obj.play()

    # Wait for the playback to finish
    #play_obj.wait_done()

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="augmented.wav"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 