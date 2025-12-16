import torch 
import random
import numpy as np

class Mixup:
    def __init__(self, alpha=0.2, mixup_prob=0.5):
        self.alpha = alpha
        self.mixup_prob = mixup_prob
    
    def __call__(self, batch):
        # decide whether to apply mixup
        if random.random() > self.mixup_prob:
            return batch
        
        batch_size = len(batch)
        if batch_size < 2:
            return batch


        # sample mixing coeff
        lam = np.random.beta(self.alpha, self.alpha)

        # random permutation so each sample is mixed with another random sample
        indices = torch.randperm(batch_size)
        
        mixed_batch = []
        for i in range(batch_size):
            original = batch[i]
            mixed_with = batch[indices[i]]
            
            mixed_audio = lam * original[0] + (1 - lam) * mixed_with[0]
            # linear interpolation of 2 audio tensors
            
            mixed_labels = lam * original[1] + (1 - lam) * mixed_with[1]
            # mix labels

            # keep metadata
            mixed_item = [mixed_audio, mixed_labels] + list(original[2:])

            # also mix features if they are presented
            if len(original) > 3:
                feat_idx = 3
                mixed_feats = lam * original[feat_idx] + (1 - lam) * mixed_with[feat_idx]
                mixed_item[feat_idx] = mixed_feats
            
            mixed_batch.append(mixed_item)
        
        return mixed_batch
