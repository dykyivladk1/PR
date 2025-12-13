wh# Practical Work in AI — Sound Event Detection with BEATs and Data Augmentation


---

## Overview

This repository contains my **Practical Work in AI** at **Johannes Kepler University Linz (JKU)**.  
The project is based on the **DCASE 2023 Task 4: Sound Event Detection with Weak Labels and Synthetic Soundscapes** baseline system.

The main goal of this work is to **reproduce the official DCASE 2023 baseline using a pretrained BEATs transformer** and to **analyze the impact of data augmentation techniques on Sound Event Detection performance**.

---

## Project Overview

- **Task:** Sound Event Detection (SED)
- **Dataset:** DCASE 2023 Task 4
- **Baseline system:** DCASE 2023 Task 4 baseline
- **Embedding extractor:** Pretrained **BEATs** transformer (frozen)
- **Classifier:** CRNN (baseline architecture)
- **Metric:** PSDS-1
- **Focus:** Effect of **data augmentation** on SED performance

---

## Experimental Setup

### Model
- The system follows the **official DCASE 2023 baseline architecture**.
- A **pretrained BEATs transformer** is used as a **frozen embedding extractor**.
- No fine-tuning of the transformer is performed.
- Only the CRNN classifier is trained.


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

Two training configurations are compared:

### 1. Baseline (No Augmentation)
- Original DCASE 2023 baseline setup
- No augmentation applied

### 2. Baseline + Data Augmentation
- Data augmentation applied **only during training**
- Validation and test data remain unchanged
- The BEATs transformer remains frozen

At the current stage, **spectrogram-based augmentation** is used.  
Additional augmentation techniques (e.g. waveform-level or alternative spectrogram augmentations) may be explored in later experiments.


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

``` bash
cd recipes/dcase2023_task4_baseline
python train_sed.py
```

This script performs: - Training - Validation - Evaluation\
and outputs all metrics reported in the experiments.

------------------------------------------------------------------------

### 4. Disable Spectral Augmentation (optional)

In the current setup, **spectral augmentation is enabled by default**
inside the dataset `__getitem__` method.

To train **BEATs exactly as in the original baseline (without
augmentation)**, comment out the following lines **in every dataset
class where `__getitem__` is implemented**:

``` python
if not self.test:
    feats = self.spec_aug(feats)
```

After disabling these lines, the model will be trained **without
spectral augmentation**, matching the original baseline configuration.



## 5. Disable Mixup Augmentation (optional)

Mixup augmentation is applied via a custom `collate_fn` during training.

To train **without Mixup**, open:
```
recipes/dcase2023_task4_baseline/train_sed.py
```

Locate the `SEDTask4` initialization:
```python
desed_training = SEDTask4(
    config,
    encoder=encoder,
    sed_student=sed_student,
    opt=opt,
    train_data=train_dataset,
    valid_data=valid_dataset,
    test_data=test_dataset,
    train_sampler=batch_sampler,
    scheduler=exp_scheduler,
    fast_dev_run=fast_dev_run,
    evaluation=evaluation,
    train_collate_fn=train_collate_fn,  # Pass collate function with Mixup
)
```

To disable Mixup, simply comment out the `train_collate_fn` line:
```python
# train_collate_fn=train_collate_fn
```

After this change, the model will be trained without Mixup augmentation, using the standard batch collation.


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

---

### Results with Spectral Augmentation

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.2962         | 0.3021                     | 0.4821         | 0.4821                     | 57.86%               | 34.80%          |

---

### Results with Mixup Augmentation

**Mixup** applies data augmentation by linearly mixing two random samples in a batch (audio + labels, and features if present) with some probability, using a random weight λ drawn from a Beta distribution. This improves generalization and reduces overfitting.

| PSDS-scenario1 | PSDS-scenario1 (sed score) | PSDS-scenario2 | PSDS-scenario2 (sed score) | Intersection-based F1 | Collar-based F1 |
|----------------|----------------------------|----------------|----------------------------|----------------------|-----------------|
| 0.3562         | 0.3630                     | 0.5487         | 0.5564                     | 61.29%               | 39.27%          |

---

## Conclusion

The experiments demonstrate that **data augmentation has a significant and method-dependent impact** on Sound Event Detection performance:

- **Baseline (no augmentation)** provides a strong and stable reference, achieving the best overall balance across PSDS and F1 metrics.

- **Time Stretch augmentation** clearly degrades performance across all metrics, indicating that temporal distortion harms event alignment and temporal precision for this task.

- **Spectral augmentation** improves over Time Stretch and recovers part of the baseline performance, suggesting better robustness to frequency-domain variations, but it still does not surpass the baseline.

- **Mixup augmentation** delivers the best overall improvement, slightly outperforming the baseline in PSDS-scenario1 and maintaining competitive PSDS-scenario2 and F1 scores. This indicates improved generalization without severe loss of temporal accuracy.



---
