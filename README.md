# Practical Work in AI — Sound Event Detection with CRNN and Data Augmentation

---

## Overview

This repository contains my **Practical Work in AI** at **Johannes Kepler University Linz (JKU)**.  
The project is based on the **DCASE 2023 Task 4: Sound Event Detection with Weak Labels and Synthetic Soundscapes** baseline system.

The main goal of this work is to **reproduce the official DCASE 2023 baseline using a CRNN-based Sound Event Detection model** and to **analyze the impact of different data augmentation techniques on Sound Event Detection performance**.

---

## Project Overview

- **Task:** Sound Event Detection (SED)
- **Dataset:** DCASE 2023 Task 4
- **Baseline system:** Official DCASE 2023 Task 4 baseline
- **Model:** CRNN (Convolutional Recurrent Neural Network)
- **Training paradigm:** Weak + synthetic supervision
- **Metric:** PSDS-1
- **Focus:** Effect of **data augmentation** on SED performance

---

## Experimental Setup


### Model Architecture

- The system follows the **official DCASE 2023 baseline CRNN architecture**.
- The model consists of:
  - Convolutional layers for time–frequency feature extraction
  - Recurrent layers for temporal modeling
  - A frame-level classifier for sound event detection
- **No pretrained transformer embeddings are used**
- The **CRNN model is trained end-to-end**


---

## Dataset

This work uses the datasets provided in **DCASE 2023 Task 4**, which combine **weakly labeled real-world audio** with **synthetic soundscapes containing strong labels**.

### AudioSet (Weakly Labeled Data)
- Source: **AudioSet**
- Labels: **Weak (clip-level) labels**
- Usage: Used as weak supervision during training
- No temporal (onset/offset) annotations are provided

### DCASE Synthetic Dataset
- Source: **DCASE 2021 Synthetic Dataset**
- Content: Artificially generated soundscapes
- Labels: **Strong labels** with precise onset and offset timestamps
- Purpose: Supervised training of temporal sound event detection


### Dataset Configuration (Baseline Setup)
- Synthetic training set: 10,000 clips  
- Weakly labeled training set: 1,578 clips  
- Unlabeled in-domain training set: 14,412 clips  
- Validation set: 1,168 clips  
- Strongly labeled real data is **not used**

Dataset preparation follows the official baseline instructions and is handled using the provided scripts in the DCASE repository.


---


## Data Augmentation

The main experimental variable in this project is **data augmentation**.

Evaluated augmentation techniques:

- **Time Stretch**
- **Spectral Augmentation**
- **Filter Augmentation**
- **Mixup (optional)**

---


### Installation

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
pip install h5py
pip install torchaudio
```



## Dataset Preparation and Training Pipeline

### 1. Download the dataset

First, navigate to the **DESED** directory and run:

``` bash
python download_data.py
```

Before running the script, **make sure to update the dataset path**
inside `download_data.py`:

``` python
DESED_DIR = "/home/vlad/DESED_task"
```

Change this path to match your local setup.\
This script will download **all required datasets**.

------------------------------------------------------------------------

### 2. Prepare and clean the dataset

After the download is complete, run:

``` bash
python prepare_dataset.py
```

This step is **mandatory** because many YouTube audio files are no
longer available.\
The script: - Detects missing audio files - Cleans the corresponding
entries in the `.tsv` metadata files - Ensures dataset consistency
before training

------------------------------------------------------------------------

### 3. Run training

Navigate to the baseline training script:

```bash
cd recipes/dcase2023_task4_baseline
```

#### Baseline
```bash
python train_sed.py
```

#### FilterAugment only
```bash
python train_sed.py \
  --use_filter_aug \
  --filter_aug_prob 0.5 \
  --no_time_stretch
```

#### FilterAugment + SpecAugment
```bash
python train_sed.py \
  --use_filter_aug \
  --filter_aug_prob 0.5 \
  --use_spec_aug \
  --spec_aug_prob 0.5 \
  --no_time_stretch
```

#### Time Stretch only
```bash
python train_sed.py \
  --no_filter_aug \
  --use_time_stretch \
  --time_stretch_prob 0.5
```

#### Disable all augmentations
```bash
python train_sed.py \
  --no_filter_aug \
  --no_time_stretch
```


This script performs: - Training - Validation - Evaluation\
and outputs all metrics reported in the experiments.

------------------------------------------------------------------------

## Experimental Results

### Baseline (No Augmentation)

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3498         | 0.3595                     | 0.5449         | 0.5624                     | 64.29%               | 40.73%          |

---

### Results with Time Stretch Augmentation

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.1842         | 0.1902                     | 0.4116         | 0.4221                     | 48.98%               | 22.96%          |



### Results with Time-Stretch Augmentation (Corrected)

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3572         | 0.3668                     | 0.5536         | 0.5715                     | 65.14%               | 41.62%           |



---

### Results with Spectral Augmentation

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.2962         | 0.3021                     | 0.4821         | 0.4821                     | 57.86%               | 34.80%          |




### Results with Spectral Augmentation (Corrected)

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3601         | 0.3702                     | 0.5589         | 0.5763                     | 65.87%               | 42.41%           |

---


### Results with Filter Augmentation

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3684         | 0.3789                     | 0.5678         | 0.5856                     | 66.92%               | 43.58%           |


---

### Results with Mixup Augmentation (Non-Relevant)

**Mixup** applies data augmentation by linearly mixing two random samples in a batch (audio + labels, and features if present) with some probability, using a random weight λ drawn from a Beta distribution. This improves generalization and reduces overfitting.

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3562         | 0.3630                     | 0.5487         | 0.5564                     | 61.29%               | 39.27%          |

---

## Conclusion


The experiments clearly show that **data augmentation has a strong and method-dependent impact** on Sound Event Detection performance.

- **Baseline (no augmentation)** provides a solid and reliable reference, with balanced PSDS and F1 scores across both scenarios.
- **Time Stretch augmentation**, when applied naively, can severely degrade performance, highlighting the sensitivity of SED models to temporal distortions. After correction, its performance recovers and slightly surpasses the baseline, but the gains remain modest.
- **Spectral augmentation** demonstrates better robustness than time stretch. Once corrected, it consistently improves over the baseline in both PSDS scenarios and F1 metrics, indicating improved frequency-domain generalization.
- **Filter augmentation** achieves the **best overall performance** across all reported metrics. It delivers the highest PSDS-scenario1, PSDS-scenario2, and F1 scores, confirming that modeling acoustic environment variations is particularly beneficial for this task.
- **Mixup augmentation**, while generally effective in many audio tasks, does not provide competitive improvements here and remains below the strongest single-augmentation methods.

Overall, **FilterAugment emerges as the most effective augmentation strategy** for this setup, followed by corrected spectral augmentation. Temporal augmentations should be applied cautiously, as inappropriate configurations can harm event localization perform
