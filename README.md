
# PR: Practical Work in AI JKU

# Sound Event Detection with Transformers

This repository contains the practical work for the AI course at JKU, focusing on Sound Event Detection (SED) using pre-trained transformers and the DCASE 2023 baseline system.

## Overview

Sound Event Detection aims to identify and classify sound events in audio streams, providing temporal information for their occurrence. This project reproduces the DCASE 2023 baseline system and integrates pre-trained transformers to enhance performance.

---

## Project Workflow

### Clone DCASE repository:
```bash
git clone https://github.com/DCASE-REPO/DESED_task.git
```

### Install needed dependencies:
```bash
apt-get install -y sox libsox-dev libsox-fmt-all
pip install desed
pip install pytorch_lightning
pip install sed_scores_eval
pip install codecarbon
pip install psds_eval
pip install thop
pip install torchlibrosa
```

### Note on Pretrained BEATS Model:
The weights for the 'BEATS' model are corrupted in the official repository of 2023. Ensure you download the pretrained BEATS model for extracting embeddings from the following link:  
[Download BEATS Pretrained Model](https://onedrive.live.com/?authkey=%21AGOyB4YHPatKU%2D0&id=6B83B49411CA81A7%2125958&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp)

---

### Dataset Preparation
After downloading the repository, navigate to it and download the dataset:
```bash
cd DESED_task/recipes/dcase2023_task4_baseline
python generate_dcase_task4_2023.py --only_synth
python generate_dcase_task4_2023.py --only_real
```
Use these commands if you do not need the strong labeled training set.

---

### Pre-compute Embeddings:
```bash
python extract_embeddings.py --output_dir ./embeddings --pretrained_model "beats"
```
Ensure the embeddings are stored in the `embeddings` folder.

---

### Train the Model:
Run the training script:
```bash
python train_pretrained.py --test_from_checkpoint /path/to/downloaded.ckpt
```

---

## RESULTS 

## Beats Model provided by DESED_task
PSDS_scenario 1: 0.463
I will do table soon...
