import os
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import librosa
import random
import shutil

#set a custom temp directory in the current working directory
temp_dir = os.path.join(os.getcwd(), "temp_audio_processing")
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

#paths
RAW_DIR = os.path.join("data", "raw")
SOLICITED_META_DIR = os.path.join(RAW_DIR, "solicited_coughs", "metadata")
PROCESSED_DIR = os.path.join("data", "processed")
SOLICITED_META_FILE = os.path.normpath(os.path.join(SOLICITED_META_DIR, "CODA_TB_Solicited_Meta_Info.csv"))
CLINICAL_META_FILE = os.path.normpath(os.path.join(SOLICITED_META_DIR, "CODA_TB_Clinical_Meta_Info.csv"))

#load VGGish model
model_dir = 'C:/Users/Saatvik/.vscode/AcousticTB/'
vggish_model = tf.saved_model.load(model_dir)

def add_noise(audio, noise_type, noise_factor=0.005):
    """Add different types of noise to the audio"""
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise
    elif noise_type == "pink":
        pink = np.array([1/(i+1) for i in range(len(audio))])
        pink = pink * np.random.normal(0, noise_factor, len(audio))
        return audio + pink
    elif noise_type == "ambient":
        ambient = np.sin(2 * np.pi * np.linspace(0, 10, len(audio)))
        return audio + (ambient * noise_factor)
    return audio

def extract_features(audio_path, sample_rate=16000, add_noise_prob=0.3):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    if random.random() < add_noise_prob:
        noise_type = random.choice(["gaussian", "pink", "ambient"])
        audio = add_noise(audio, noise_type)
    else:
        noise_type = "none"
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=64, 
        fmax=8000
    )
    
    mel_spectrogram = mel_spectrogram.T
    
    if mel_spectrogram.shape[1] != 64:
        mel_spectrogram = mel_spectrogram[:, :64]
    
    return mel_spectrogram, noise_type

def process_metadata(metadata_file, clinical_file, audio_dir, process_all=True):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata = pd.read_csv(metadata_file)
    clinical_data = pd.read_csv(clinical_file)
    
    #merge metadata with clinical data
    merged_data = pd.merge(metadata, clinical_data, on='participant', how='left')
    
    new_metadata = []
    output_dir = os.path.join(PROCESSED_DIR, "features")
    os.makedirs(output_dir, exist_ok=True)

    total_files = len(merged_data) if process_all else min(len(merged_data), 5)
    
    for idx, row in tqdm(merged_data.iterrows(), total=total_files, desc="Processing audio files"):
        if not process_all and idx >= 5:
            break
            
        filename = row["filename"]
        audio_path = os.path.join(audio_dir, filename)

        if os.path.exists(audio_path):
            features, noise_type = extract_features(audio_path, add_noise_prob=0.3)
            
            if features is not None:
                base_name = os.path.splitext(filename)[0]
                feature_filename = f"{base_name}_{noise_type}_noise.npy" if noise_type != "none" else f"{base_name}.npy"
                feature_file = os.path.join(output_dir, feature_filename)
                
                if not os.path.exists(feature_file):
                    np.save(feature_file, features)
                
                new_row = row.copy()
                new_row['feature_path'] = feature_file
                new_row['noise_type'] = noise_type
                new_metadata.append(new_row)
        else:
            print(f"File not found: {audio_path}")

    return pd.DataFrame(new_metadata)

def main():
    solicited_audio_dir = os.path.join(RAW_DIR, "solicited_coughs", "audio")
    if not os.path.exists(solicited_audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {solicited_audio_dir}")

    process_all = True
    solicited_metadata = process_metadata(SOLICITED_META_FILE, CLINICAL_META_FILE, solicited_audio_dir, process_all)

    output_metadata_file = os.path.join(PROCESSED_DIR, "solicited_metadata_with_noise.csv")
    solicited_metadata.to_csv(output_metadata_file, index=False)
    print(f"Saved updated metadata to: {output_metadata_file}")
    print(f"Total processed files (including noisy versions): {len(solicited_metadata)}")

    #clean up temporary directory
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
