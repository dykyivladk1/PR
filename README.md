# Practical Work in AI — Sound Event Detection with BEATs and Data Augmentation


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

---
