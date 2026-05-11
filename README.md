<div align="center">
# 🖋️ Handwritten Character Recognition — OCR
 
### Teaching machines to read human handwriting, one stroke at a time.
 
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()
 
</div>
---
 
##  Overview
 
This project implements a full end-to-end **Optical Character Recognition (OCR)** system trained specifically to recognize **handwritten characters**. From raw pen strokes to structured, machine-readable text — this pipeline handles everything: image preprocessing, feature extraction, deep learning–based classification, and a production-ready REST API for real-time inference.
 
Whether you're digitizing handwritten notes, building assistive tools for accessibility, automating form processing, or just fascinated by the intersection of computer vision and machine learning — this repo has everything you need to get up and running.
 
---
 
##  Project Structure
 
```

---
 
## Features
 
- Full OCR pipeline — from raw image to predicted character
- CNN-based model trained on handwritten character datasets
- REST API for real-time inference (single image → prediction + confidence)
- Ready-to-use test images so you can try it immediately without your own data
- Dockerized deployment
- Reproducible Jupyter notebooks with step-by-step training
- Evaluation visuals — confusion matrix, training curves, sample predictions
---

## Notebooks Walkthrough
 
The `notebooks/` folder is a complete, self-contained training environment. Run them in order:
 
| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_Data_Manger.ipynb` | Loads the dataset, visualizes class distribution, inspects sample images |
| 02 | `02_EDA&Visualization.ipynb` | Applies grayscale conversion, noise removal, binarization, resizing, and augmentation |
| 03 | `03_Augmentation.ipynb` | Defines CNN architecture, trains the model, saves weights to `models/` |
| 04 | `04_CNN_Model.ipynb` | CNN built from scratch and evaluated, plots confusion matrix and accuracy curves |
| 04 | `05_TransferLearning.ipynb` | Transfer Learning via Mobilenetv2 and evaluation |
 

---

# Transfer Learning (MobileNetV2) Overview

This project builds a high-performance handwritten character classifier (A–Z) using transfer learning with MobileNetV2.
The pipeline focuses on reducing class confusion (e.g., **O/U**, **I/L**) and overconfidence through careful training strategy, augmentation, and fine-tuning.

---

# Dataset

* **Classes:** 26 (A–Z)

## Splits

* **Training:** Augmented dataset
* **Validation:** Held-out validation set
* **Test:** Final evaluation set

---

# Preprocessing

* Convert grayscale images → RGB (3 channels)
* Resize images to **128 × 128**
* Normalize using **ImageNet mean and standard deviation**

---

# Transformations

## Training Transformations

* Resize (bilinear interpolation)
* Random rotation (**±5°**)
* Random translation (**5%**)
* Random perspective transformation (**p = 0.3**)
* Normalization (ImageNet mean/std)

### Purpose

* Improve generalization
* Reduce overfitting
* Simulate real handwriting variations

---

## Validation/Test Transformations

* Resize only
* Normalization only

This ensures a fair and consistent evaluation process.

---

# Base Model

* **Architecture:** MobileNetV2 pretrained on ImageNet

## Why MobileNetV2?

* Lightweight and efficient
* Strong feature extractor for:

  * edges
  * curves
  * handwriting strokes

---

# Model Architecture (Custom Head)

```text
Backbone (Frozen initially)
        ↓
Linear(1280 → 1024)
BatchNorm
LeakyReLU
Dropout(0.4)
        ↓
Linear(1024 → 512)
BatchNorm
LeakyReLU
Dropout(0.2)
        ↓
Linear(512 → 26)
```

---

# Key Design Choices

* **LeakyReLU**

  * Prevents the dying ReLU problem

* **Dropout**

  * Reduces overconfidence
  * Prevents memorization and overfitting

* **Deeper Classification Head**

  * Improves adaptation of pretrained features to handwriting data

---

# Training & Fine-Tuning Strategy

# Phase 1 — Feature Extraction (Head Training)

* Backbone frozen
* Train only the custom classifier head

## Settings

* **Epochs:** 15
* **Optimizer:** Adam
* **Learning Rate:** (1 \times 10^{-3})
* **Scheduler:** ReduceLROnPlateau (`patience = 2`)

### Purpose

* Fast convergence
* Stable classifier learning
* Prevents destroying pretrained features early

---

# Phase 2 — Fine-Tuning

* Unfreeze only the **top 10 layers** of the backbone
* Adapt high-level pretrained features to handwriting patterns

## Settings

* **Epochs:** 25
* **Optimizer:** Adam + Weight Decay
* **Learning Rate:** (1 \times 10^{-5})
* **Scheduler:** Cosine Annealing

### Purpose

* Fine-grained feature learning
* Better adaptation to confusing handwritten characters
* Improved generalization

---

# Loss Function

Using CrossEntropyLoss with Label Smoothing:

```python
nn.CrossEntropyLoss(label_smoothing=0.05)
```

---

# Why Label Smoothing is Important

Label smoothing was critical because it:

* Prevented overconfidence
* Improved generalization
* Reduced sharp probability spikes
* Improved performance on confusing classes such as:

  * O/U
  * I/L

---

# Final Results

* **Test Accuracy:** ~98.8% – 99.0%
* Strong performance across nearly all classes

---

# Confusion Matrix Analysis

Fine-tuning and label smoothing significantly reduced confusion between visually similar characters.

## Main Confusing Pairs

| Pair  | Errors            |
| ----- | ----------------- |
| O ↔ U | U→O: 8            |
| I ↔ L | Nearly eliminated |
| C ↔ G | C→G: 4, G→C: 1    |
| V ↔ Y | V→Y: 2            |
| P ↔ F | Very few errors   |

---

# Observations

* Most remaining errors occur between visually similar letters
* Fine-tuning improved feature sensitivity to subtle stroke differences
* Translation augmentation improved robustness to shifted handwriting

---

# Experimental Findings

After multiple experiments, several important observations were made.

## Issues Observed

* High learning rates during fine-tuning destroyed pretrained features
* Few epochs caused underfitting
* Simple classifier heads produced weak feature adaptation
* Removing translation augmentation reduced generalization

---

# Improvements Applied

* Lower fine-tuning learning rate
* Increase epochs from **10–15 → 25**
* Add translation augmentation
* Use a deeper classification head
* Apply progressive unfreezing

---

# Overall Outcome

The final pipeline achieved:

* High accuracy
* Strong generalization
* Reduced overconfidence
* Better handling of visually similar handwritten letters

The combination of:

* transfer learning,
* augmentation,
* label smoothing,
* deeper classification head,
* and gradual fine-tuning

was essential to achieving near state-of-the-art performance on handwritten alphabet classification.

---
 
## 🧠 Model Architecture in API
 
The recognition pipeline consists of five stages:
 
```
Raw Image
    │
    ▼
┌─────────────────────┐
│   1. Preprocessing  │  Grayscale → Denoise → Binarize → Resize → Normalize
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  2. Segmentation    │  Isolate individual characters from the input
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 3. Feature Extract  │  CNN layers — convolutional + pooling → spatial features
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 4. Classification   │  Fully connected layers → softmax over character classes
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 5. Postprocessing   │  Top-1 prediction + confidence score returned via API
└─────────────────────┘
```
 
---
 
## 📊 Results
 
| Metric | Score |
|--------|-------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~98% |
| Test Accuracy | ~99% |
 
Detailed plots and per-class breakdowns are in the `visuals/` folder:
 
- **`training_curves.png`** — accuracy and loss across epochs
- **`confusion_matrix.png`** — which characters get confused with which
- **`sample_predictions.png`** — input images alongside model predictions
---
 
## 📡 API Reference
 
### `POST /predict`
 
Accepts a handwritten character image and returns the predicted character with a confidence score.
 
**Request**
 
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | `file` | ✅ | PNG or JPEG image of a handwritten character |
 
**Response**
 
```json
{
  "prediction": "A",
  "confidence": 0.97
}
```
 
**Error Response**
 
```json
{
  "error": "No image provided"
}
```
 
**Example using Python `requests`:**
 
```python
import requests
 
url = "http://localhost:5000/predict"
with open("test_pictures/sample_A.png", "rb") as f:
    response = requests.post(url, files={"image": f})
 
print(response.json())
# → {"prediction": "A", "confidence": 0.97}
```
 
---
 
## 🛠️ Tech Stack
 
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pytorch | Model training & inference |
| OpenCV | Image preprocessing |
| FastAPI | REST API server |
| Docker | Containerized deployment |
| Matplotlib / Seaborn | Training visualizations |
| Jupyter Notebook | Interactive experimentation |
 
---
 
## 👩‍💻 Author
 
**Mariam Hamza**
- GitHub: [@mariamhamza3019-star](https://github.com/mariamhamza3019-star)
---
 
— feel free to use, modify, and distribute.
 
---
 
<div align="center">
*If this project helped you, drop a ⭐ — it means a lot!*
 
</div>
