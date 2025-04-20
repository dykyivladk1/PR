
# Practical Work in AI JKU
# Sound Event Detection with Transformers

This repository contains the practical work for the AI course at JKU, focusing on Sound Event Detection (SED) using pre-trained transformers and the DCASE 2023 baseline system.

---

## Overview

Sound Event Detection aims to identify and classify sound events in audio streams, providing temporal information for their occurrence. This project reproduces the DCASE 2023 baseline system and integrates pre-trained transformers to enhance performance.

---

## Project Workflow

### Clone DCASE Repository
```bash
git clone https://github.com/DCASE-REPO/DESED_task.git
```

---

### Install Needed Dependencies
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

---

### Note on Pretrained BEATS Model
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

### Pre-compute Embeddings
```bash
python extract_embeddings.py --output_dir ./embeddings --pretrained_model "beats"
```
Ensure the embeddings are stored in the `embeddings` folder.

---


## Model Evaluation Plan

This project evaluates the performance of various BEATs models by extracting embeddings and training a CRNN classifier on them. The goal is to identify which model yields the best PSDS1 score.

---

## Evaluated Models

| Model         | Dataset   | PSDS-scenario1 | Notes                                  |
|---------------|-----------|----------------|----------------------------------------|
| BEATs weak    | Dev-test  | 0.481 | Baseline result, official evaluation   |
| BEATs strong  | Dev-test  | 0.520 | Expected to outperform weak model      |
| BEATs ssl     | Dev-test  | 0.465 | Slightly underperforms compared to weak|

All models are evaluated using the same CRNN classifier to ensure fair comparison.

---


## Data Augmentation

To improve model generalization and robustness, we apply audio data augmentation techniques using the `audiomentations` library. The augmentations include:

- Gaussian noise addition  
- Time stretching  
- Pitch shifting  
- Temporal shifting

The augmentation script is located at:

```
DESED_task/recipes/dcase2023_task4_baseline/augment_dataset.py
```

### Run Data Augmentation

To generate augmented audio files:

```bash
python augment_dataset.py
```

This script processes all `.wav` files in the training set and saves augmented copies to:

```
DESED_task/data/dcase/dataset/audio/train_augmented/
```

Ensure that these augmented files are later included during embedding extraction and model training by modifying your dataloader or preprocessing pipeline accordingly.

Also, rerun extraction of embeddings, and testing the models.

---

## Results After Augmentation

| Model         | PSDS-scenario1 (Original) | PSDS-scenario1 (Augmented) | Notes                                   |
|---------------|----------------------------|------------------------------|-----------------------------------------|
| BEATs weak    | 0.481                      | 0.496                       | Slight improvement from noise robustness|
| BEATs strong  | 0.520                      | 0.537                       | Strong model benefits from augmentation |
| BEATs ssl     | 0.465                      | 0.479                       | Minor gains due to limited capacity     |



---


---

