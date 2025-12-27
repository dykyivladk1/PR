import random
import torch
import torchaudio

def augment_time_stretch(wav, sr, min_rate=0.9, max_rate=1.1):
    factor = random.uniform(min_rate, max_rate)
    effects = [["tempo", f"{factor}"]]

    wav_out, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav.unsqueeze(0), sr, effects
    )
    wav_out = wav_out.squeeze(0)

    if wav_out.shape[-1] > wav.shape[-1]:
        wav_out = wav_out[: wav.shape[-1]]
    else:
        wav_out = torch.nn.functional.pad(
            wav_out, (0, wav.shape[-1] - wav_out.shape[-1])
        )

    return wav_out, factor
