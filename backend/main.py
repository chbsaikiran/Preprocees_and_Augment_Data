from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
from PIL import Image
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create directories if they don't exist
os.makedirs("spectrograms", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("processed_images", exist_ok=True)
os.makedirs("processed_3d_images", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the mount statements (around line 36-39)
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
app.mount("/spectrograms", StaticFiles(directory="spectrograms"), name="spectrograms")
app.mount("/processed_images", StaticFiles(directory="processed_images"), name="processed_images")
app.mount("/processed_3d_images", StaticFiles(directory="processed_3d_images"), name="processed_3d_images")
app.mount("/temp_uploads", StaticFiles(directory="temp_uploads"), name="temp_uploads")  # Add this line

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
        original_signal, original_sr = torchaudio.load(audio_sample_path)

        # Resample if necessary
        if original_sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.target_sample_rate)
            signal = resampler(original_signal)
        else:
            signal = original_signal

        # Store the original signal before transformation for playback
        original_signal = signal.clone()

        # Apply transformations (e.g., MFCC)
        signal = self.transformation(signal)
        return signal,label, audio_sample_path, self.target_sample_rate, original_signal, self.target_sample_rate  # Include original signal for visualization

class SingleSampleDataset(Dataset):
    def __init__(self, sample_file):
        self.sample_file = sample_file

    def __len__(self):
        return 1  # Only one sample

    def __getitem__(self, idx):
        # Load the mesh using trimesh
        mesh = trimesh.load(self.sample_file)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        label = "cube"  # Set a dummy label for this example
        return {'vertices': vertices, 'faces': faces, 'label': label}

# Preprocess function to normalize the mesh
def preprocess_mesh(vertices):
    # Center the mesh at the origin
    centroid = vertices.mean(axis=0)
    vertices -= centroid

    # Scale the mesh to fit within a unit sphere
    max_distance = np.linalg.norm(vertices, axis=1).max()
    vertices /= max_distance

    return vertices

# Augmentation function to apply a random rotation to the mesh
def augment_mesh(vertices):
    # Create a random rotation matrix
    rotation_axis = np.random.rand(3) - 0.5  # Random vector in 3D space
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize to unit vector
    angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians

    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle, rotation_axis
    )[:3, :3]  # Extract 3x3 rotation part

    # Apply rotation to vertices
    vertices = vertices @ rotation_matrix.T

    return vertices

# Function to visualize the mesh
def visualize_mesh(vertices, faces,three_d_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection object for the faces
    poly3d = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] for face in faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    # Set plot limits
    ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
    ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
    ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])

    plt.savefig(os.path.join('processed_3d_images', three_d_path), dpi=300, bbox_inches='tight')
    plt.close()

# Add these functions after the imports and before the endpoints
def convert_to_grayscale(input_path: str, output_path: str):
    try:
        with Image.open(input_path) as img:
            # Convert to grayscale
            grayscale_img = img.convert('L')
            # Save to processed_images directory
            save_path = os.path.join('processed_images', output_path)
            grayscale_img.save(save_path)
    except Exception as e:
        logger.error(f"Error in convert_to_grayscale: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def tilt_image(input_path: str, output_path: str, angle: float):
    try:
        with Image.open(input_path) as img:
            # Rotate the image
            rotated_img = img.rotate(angle, expand=True)
            # Save to processed_images directory
            save_path = os.path.join('processed_images', output_path)
            rotated_img.save(save_path)
    except Exception as e:
        logger.error(f"Error in tilt_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add at the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update the mount statements (around line 36-39)
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")
app.mount("/spectrograms", StaticFiles(directory="spectrograms"), name="spectrograms")
app.mount("/processed_images", StaticFiles(directory="processed_images"), name="processed_images")
app.mount("/processed_3d_images", StaticFiles(directory="processed_3d_images"), name="processed_3d_images")
app.mount("/temp_uploads", StaticFiles(directory="temp_uploads"), name="temp_uploads")  # Add this line

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
        # Save the contents of the uploaded file to a temporary location
        temp_file_path = os.path.join(os.getcwd(), "temp_uploads", file.filename)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        
        contents = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        # Save the full path
        InOutFileNames_obj.save_in_out_file_name(temp_file_path)
        logger.info(f"Saved file to: {temp_file_path}")
        
        return {"message": "File saved successfully", "file_path": temp_file_path}
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add cleanup function
def cleanup_temp_files():
    temp_dir = os.path.join(os.getcwd(), "temp_uploads")
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")

@app.post("/api/preprocess")
async def preprocess_data(file: UploadFile = File(...)):
    try:
        content_type = file.content_type
        if content_type.startswith('image/'):
            input_path = InOutFileNames_obj.get_in_out_file_name()
            output_path = 'grayscale_image.png'
            convert_to_grayscale(input_path, output_path)
            
            return {
                "type": "image",
                "imagePath": f"/processed_images/{output_path}"
            }
        elif file.filename.endswith('.wav'):
            return {
                "type": "audio",
                "audioPath": "/api/audio/preprocessed",
                "spectrogramPath": "/spectrograms/preprocessed.png"
            }
        elif file.filename.endswith('.off'):  # 3D file case
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "preprocessed_3d.png"
            
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()
                vertices = preprocess_mesh(vertices)
                faces = data['faces'][0].numpy()
                visualize_mesh(vertices, faces, three_d_path)
            
            return {
                "type": "3d",
                "imagePath": f"/processed_3d_images/{three_d_path}"
            }
            
        # For text files, use the saved content
        if file.filename.endswith('.txt'):
            text_content = InOutFileNames_obj.get_in_out_file_name()
            lowercase_result = show_lowercase_data(text_content)
            stopwords_result = show_after_remove_stop_words_data(text_content)
            
            return {
                "lowercase_data": lowercase_result,
                "after_remove_stop_words_data": stopwords_result
            }
        else:
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())  # Adjust path accordingly
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "preprocessed_3d.png"
            # Iterate through the dataloader and visualize the 3D sample
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()  # Convert from tensor to numpy for plotting
                vertices = preprocess_mesh(vertices)
                faces = data['faces'][0].numpy()  # Convert from tensor to numpy for plotting
                visualize_mesh(vertices, faces)
            
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/augment")
async def augment_data(file: UploadFile = File(...), angle: str = Form(default="45")):  # Make angle optional with default value
    try:
        content_type = file.content_type
        if content_type.startswith('image/'):
            input_path = InOutFileNames_obj.get_in_out_file_name()
            output_path = 'tilted_image.png'
            
            # Debug logging
            logger.info(f"Raw angle received: {angle}")
            angle_float = float(angle)
            logger.info(f"Converted angle: {angle_float}")
            
            tilt_image(input_path, output_path, angle=angle_float)
            
            return {
                "type": "image",
                "imagePath": f"/processed_images/{output_path}"
            }
        elif file.filename.endswith('.wav'):
            return {
                "type": "audio",
                "audioPath": "/api/audio/augmented",
                "spectrogramPath": "/spectrograms/augmented.png"
            }
        elif file.filename.endswith('.off'):  # 3D file case
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "augmented_3d.png"
            
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()
                vertices = augment_mesh(vertices)
                faces = data['faces'][0].numpy()
                visualize_mesh(vertices, faces, three_d_path)
            
            return {
                "type": "3d",
                "imagePath": f"/processed_3d_images/{three_d_path}"
            }
            
        # For text files, use the saved content
        if file.filename.endswith('.txt'):
            text_content = InOutFileNames_obj.get_in_out_file_name()
            synonym_result = show_after_synonym_replacement_data(text_content)
            insertion_result = show_after_random_insertion_data(text_content)
            
            return {
                "synonym_replacement_data": synonym_result,
                "random_insertion_data": insertion_result
            }
        else:
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())  # Adjust path accordingly
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "augmented_3d.png"
            # Iterate through the dataloader and visualize the 3D sample
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()  # Convert from tensor to numpy for plotting
                vertices = augment_mesh(vertices)
                faces = data['faces'][0].numpy()  # Convert from tensor to numpy for plotting
                visualize_mesh(vertices, faces,three_d_path)
            
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/original")
async def original_data(file: UploadFile = File(...)):
    try:
        content_type = file.content_type
        if content_type.startswith('image/'):
            # Save the original image
            file_path = os.path.join('temp_uploads', file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Save it as the input file for later processing (using correct method name)
            InOutFileNames_obj.save_in_out_file_name(file_path)  # Changed from set_in_out_file_name
            
            return {
                "type": "image",
                "imagePath": f"/temp_uploads/{file.filename}"
            }
        elif file.filename.endswith('.wav'):
            return {
                "type": "audio",
                "audioPath": "/api/audio/original",
                "spectrogramPath": "/spectrograms/original.png"
            }
        elif file.filename.endswith('.off'):  # 3D file case
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "original_3d.png"
            
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()
                faces = data['faces'][0].numpy()
                visualize_mesh(vertices, faces, three_d_path)
            
            return {
                "type": "3d",
                "imagePath": f"/processed_3d_images/{three_d_path}"
            }

        # For text files, use the saved content
        if file.filename.endswith('.txt'):
            result = show_original_data(InOutFileNames_obj.get_in_out_file_name())
            return {"output": result}
        else:
            sample_dataset = SingleSampleDataset(sample_file=InOutFileNames_obj.get_in_out_file_name())  # Adjust path accordingly
            sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
            three_d_path = "original_3d.png"
            # Iterate through the dataloader and visualize the 3D sample
            for data in sample_loader:
                vertices = data['vertices'][0].numpy()  # Convert from tensor to numpy for plotting
                faces = data['faces'][0].numpy()  # Convert from tensor to numpy for plotting
                visualize_mesh(vertices, faces,three_d_path)
            
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
    try:
        # Define transformations
        transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        
        file_path = InOutFileNames_obj.get_in_out_file_name()
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Create Dataset and DataLoader
        audio_dataset = AudioDataset(
            file_path=InOutFileNames_obj.get_in_out_file_name(),
            transformation=transformation,
            target_sample_rate=16000
        )
        audio_loader = DataLoader(audio_dataset, batch_size=1, shuffle=True)

        signal,label, audio_sample_path, sr, original_signal, original_sr = next(iter(audio_loader))

        original_signal = original_signal[0].numpy()
        original_signal = original_signal.reshape(-1)

        # Save the waveform in 16-bit PCM format
        output_path = 'original.wav'
        sf.write(output_path, original_signal, original_sr.numpy()[0])

        original_signal1, original_sr1 = torchaudio.load(output_path)

        signal = audio_dataset.transformation(original_signal1)

        # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
        # Take the first channel (mono) and transpose it to plot (time on x-axis)
        signal_mfcc = signal[0].numpy()
        signal_mfcc = np.squeeze(signal_mfcc)
        if isinstance(audio_sample_path, tuple):
            audio_sample_path = audio_sample_path[0]

        plt.figure(figsize=(10, 4))
        plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
        plt.xlabel("Time")
        plt.ylabel("MFCC Coefficients")

        # Save spectrogram to the spectrograms directory
        spectrogram_path = os.path.join('spectrograms', 'original.png')
        plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
        plt.close()

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="original.wav"
        )
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/preprocessed")
async def get_preprocessed_audio():
    # Define transformations
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

    preprocess_audio_dataset = AudioDataset(
    file_path = InOutFileNames_obj.get_in_out_file_name(),
    transformation=transformation,
    target_sample_rate=16000
    )
    preprocess_audio_loader = DataLoader(preprocess_audio_dataset, batch_size=1, shuffle=True)

    signal,label, audio_sample_path, sr, original_signal, original_sr = next(iter(preprocess_audio_loader))

    original_signal = original_signal[0].numpy()
    original_signal = original_signal.reshape(-1)

    #preprocess_audio_dataset.RemoveNoiseFromAudio();
    noise_sample = original_signal[0:int(0.5 * original_sr)]
    reduced_noise_signal = nr.reduce_noise(y=original_signal, y_noise=noise_sample, sr=original_sr.numpy()[0])
    output_path = 'preprocessed.wav'
    sf.write(output_path, reduced_noise_signal, original_sr.numpy()[0])

    original_signal1, original_sr1 = torchaudio.load(output_path)

    signal = preprocess_audio_dataset.transformation(original_signal1)

    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()
    signal_mfcc = np.squeeze(signal_mfcc)
    if isinstance(audio_sample_path, tuple):
        audio_sample_path = audio_sample_path[0]

    plt.figure(figsize=(10, 4))
    plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")

    # Save spectrogram to the spectrograms directory
    spectrogram_path = os.path.join('spectrograms', 'preprocessed.png')
    plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
    plt.close()

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="preprocessed.wav"
    )

@app.get("/api/audio/augmented")
async def get_augmented_audio():
    # Define transformations
    transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)

    augment_audio_dataset = AudioDataset(
    file_path = InOutFileNames_obj.get_in_out_file_name(),
    transformation=transformation,
    target_sample_rate=16000
    )
    augment_audio_loader = DataLoader(augment_audio_dataset, batch_size=1, shuffle=True)

    signal, label, audio_sample_path, sr, original_signal, original_sr = next(iter(augment_audio_loader))

    original_signal = original_signal[0].numpy()
    original_signal = original_signal.reshape(-1)

    shifted_signal = librosa.effects.pitch_shift(original_signal, sr=original_sr.numpy()[0], n_steps=4)
    output_path = 'augmented.wav'
    sf.write(output_path, shifted_signal, original_sr.numpy()[0])

    original_signal1, original_sr1 = torchaudio.load(output_path)

    signal = augment_audio_dataset.transformation(original_signal1)

    # MFCC output will be 3D: [channels, features (MFCC coefficients), time_steps]
    # Take the first channel (mono) and transpose it to plot (time on x-axis)
    signal_mfcc = signal[0].numpy()
    signal_mfcc = np.squeeze(signal_mfcc)
    if isinstance(audio_sample_path, tuple):
        audio_sample_path = audio_sample_path[0]

    plt.figure(figsize=(10, 4))
    plt.imshow(signal_mfcc, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"MFCC for {os.path.basename(audio_sample_path)}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
     # Save spectrogram to the spectrograms directory
    spectrogram_path = os.path.join('spectrograms', 'augmented.png')
    plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
    plt.close()

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="augmented.wav"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 