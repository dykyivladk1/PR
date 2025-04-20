import os
import numpy as np
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from tqdm import tqdm

AUGMENT = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.3),
])

def augment_and_save(in_folder, out_folder, sample_rate=16000):
    os.makedirs(out_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(in_folder) if f.endswith('.wav')]
    for fname in tqdm(wav_files, desc=f"Augmenting {os.path.basename(in_folder)}"):
        in_path = os.path.join(in_folder, fname)
        audio, sr = sf.read(in_path)
        if sr != sample_rate:
            continue
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # convert to mono
        augmented = AUGMENT(samples=audio, sample_rate=sample_rate)
        out_path = os.path.join(out_folder, f'aug_{fname}')
        sf.write(out_path, augmented, sample_rate)

base_input = '/../../DESED_task/data/dcase/dataset/audio/train'
base_output = '/../../data/dcase/dataset/audio/train_augmented'

folders = [f for f in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, f))]
for folder_name in tqdm(folders, desc="Processing folders"):
    folder_path = os.path.join(base_input, folder_name)
    output_path = os.path.join(base_output, folder_name)
    augment_and_save(in_folder=folder_path, out_folder=output_path)
