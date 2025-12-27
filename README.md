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

This work uses the datasets provided in **DCASE 2023 Task 4**, combining **weakly labeled real-world audio** with **synthetic soundscapes containing strong labels**.

### AudioSet (Weakly Labeled Data)

- Source: **AudioSet**
- Labels: **Weak (clip-level)**
- Usage: Weak supervision during training
- No onset/offset annotations

### DCASE Synthetic Dataset

- Source: **DCASE Synthetic Soundscapes**
- Labels: **Strong (onset/offset)**
- Purpose: Supervised temporal learning

### Dataset Configuration (Baseline Setup)

- Synthetic training set: 10,000 clips  
- Weakly labeled training set: 1,578 clips  
- Unlabeled in-domain training set: 14,412 clips  
- Validation set: 1,168 clips  
- Strongly labeled real data is **not used**

---

## Data Augmentation

The main experimental variable in this project is **data augmentation**.

Evaluated augmentation techniques:

- **Time Stretch**
- **Spectral Augmentation**
- **Filter Augmentation**
- **Mixup (optional)**

---

## Experimental Results

(see tables in README)

---

## Conclusion

Augmentation strategies that preserve temporal structure yield the best performance for CRNN-based Sound Event Detection.
