import torch
import random
import numpy as np


def filter_augment(mel_spec, db_range=(-6, 6), band_range=(3, 6), min_bw=6, mode='linear'):
    
    is_tensor = torch.is_tensor(mel_spec)
    # checks if the input mel_spec is tensor or np.array

    # if tensor move it to cpu and convert to numpy
    if is_tensor:
        device = mel_spec.device
        spec = mel_spec.cpu().numpy()
    else:
        spec = mel_spec.copy()


    # check if the shape is [mel_freq_bins, time_frames]
    # we check by the shape because mel_freq_bins is usually lower than time_frames
    if spec.shape[0] > spec.shape[1]:
        # meaning the shape is [time_frames, mel_freq_bins]
        spec = spec.T
        # transpose it to match [mel_freq_bins, time_frames]
        transposed = True
    else:
        transposed = False

    
    n_mels = spec.shape[0]
    # number of mel bins
    n_bands = random.randint(band_range[0], band_range[1])
    # randomly choose how many frequency bands the filter will have
    
    boundaries = [0, n_mels]
    # this creates the frequency band boundaries 
    
    for _ in range(n_bands - 1):
        # iterate over n_bands exception start and end
        for _ in range(100):
            # limit the number of attempts to avoid infinite loops
            candidate = random.randint(1, n_mels - 1)
            # randomly sample a potential freq boundary
            if all(abs(candidate - b) >= min_bw for b in boundaries):
                # ensure a min bandwidth between freq bands
                boundaries.append(candidate)
                # append the valid boundary
                break
                
    boundaries = sorted(boundaries)
    
    
    filter_curve = np.zeros(n_mels)
    # init freq filter curve with one gain value per mel bin
    
    if mode == 'step':
        weights = [random.uniform(db_range[0], db_range[1]) for _ in range(len(boundaries) - 1)]
        # each frequency band receives a constant gain
        for i in range(len(boundaries) - 1):
            # randomly saple one gain value for each freq band
            filter_curve[boundaries[i]:boundaries[i+1]] = weights[i]

    # linear mode
    else:
        weights = [random.uniform(db_range[0], db_range[1]) for _ in range(len(boundaries))]
        # randomply samples a gain value for each freq boundary
        
        for i in range(len(boundaries) - 1):
            # iterate over neighboring frequency bands
            start, end = boundaries[i], boundaries[i+1]
            # start and edn mel-bin indices of the current band
            filter_curve[start:end] = np.linspace(weights[i], weights[i+1], end-start, endpoint=False)
            # smoothly change the gain between two neighboring bands
        
        filter_curve[-1] = weights[-1]
        # set the gain for the last mel bin
    
    spec = spec + filter_curve[:, np.newaxis]
    # apply the frequency filter to all time frames of the spectrogram
    
    if transposed:
        spec = spec.T
    # transpose back to the original shape if it was transposed
    
    if is_tensor:
        spec = torch.from_numpy(spec).to(device)
    # convert back to a torch tensor and move it to the original device
    
    return spec
