import torch
import random
import numpy as np


def filter_augment(mel_spec, db_range=(-6, 6), band_range=(3, 6), min_bw=6, mode='linear'):
    is_tensor = torch.is_tensor(mel_spec)
    if is_tensor:
        device = mel_spec.device
        spec = mel_spec.cpu().numpy()
    else:
        spec = mel_spec.copy()
    
    if spec.shape[0] > spec.shape[1]:
        spec = spec.T
        transposed = True
    else:
        transposed = False
    
    n_mels = spec.shape[0]
    n_bands = random.randint(band_range[0], band_range[1])
    
    boundaries = [0, n_mels]
    for _ in range(n_bands - 1):
        for _ in range(100):
            candidate = random.randint(1, n_mels - 1)
            if all(abs(candidate - b) >= min_bw for b in boundaries):
                boundaries.append(candidate)
                break
    boundaries = sorted(boundaries)
    
    filter_curve = np.zeros(n_mels)
    
    if mode == 'step':
        weights = [random.uniform(db_range[0], db_range[1]) for _ in range(len(boundaries) - 1)]
        for i in range(len(boundaries) - 1):
            filter_curve[boundaries[i]:boundaries[i+1]] = weights[i]
    else:
        weights = [random.uniform(db_range[0], db_range[1]) for _ in range(len(boundaries))]
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            filter_curve[start:end] = np.linspace(weights[i], weights[i+1], end-start, endpoint=False)
        filter_curve[-1] = weights[-1]
    
    spec = spec + filter_curve[:, np.newaxis]
    
    if transposed:
        spec = spec.T
    if is_tensor:
        spec = torch.from_numpy(spec).to(device)
    
    return spec
