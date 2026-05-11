# Handwritten Character Recognition 
- Dataset : EMNIST dataset.
- CNN, Transfer Learning (MobileNetV2)


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




