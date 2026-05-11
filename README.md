<div align="center">
# 🖋️ Handwritten Character Recognition — OCR
 
### Teaching machines to read human handwriting, one stroke at a time.
 
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()
 
</div>
---
 
## 🔍 Overview
 
This project implements a full end-to-end **Optical Character Recognition (OCR)** system trained specifically to recognize **handwritten characters**. From raw pen strokes to structured, machine-readable text — this pipeline handles everything: image preprocessing, feature extraction, deep learning–based classification, and a production-ready REST API for real-time inference.
 
Whether you're digitizing handwritten notes, building assistive tools for accessibility, automating form processing, or just fascinated by the intersection of computer vision and machine learning — this repo has everything you need to get up and running.
 
---
 
## 🗂️ Project Structure
 
```
Handwritten-Character-Recognition-OCR/
│
├── 📓 notebooks/                   # Jupyter notebooks for training & experimentation
│   ├── 01_data_exploration.ipynb   # EDA — dataset stats, class distribution, sample images
│   ├── 02_preprocessing.ipynb      # Image cleaning, normalization, augmentation
│   ├── 03_model_training.ipynb     # CNN architecture, training loop, callbacks
│   └── 04_evaluation.ipynb         # Accuracy, confusion matrix, error analysis
│
├── 🤖 models/                      # Saved model files
│   ├── model.h5                    # Trained Keras model weights
│   └── model_architecture.json     # Model architecture (for loading without retraining)
│
├── 🌐 api/                         # REST API for serving predictions
│   ├── app.py                      # Flask/FastAPI app entry point
│   ├── predictor.py                # Inference logic — loads model, runs prediction
│   ├── utils.py                    # Shared helpers (image decoding, preprocessing)
│   ├── requirements.txt            # API-specific dependencies
│   └── Dockerfile                  # Container definition for the API service
│
├── 🖼️ visuals/                     # Plots and evaluation outputs
│   ├── training_curves.png         # Accuracy & loss over epochs
│   ├── confusion_matrix.png        # Per-class prediction breakdown
│   └── sample_predictions.png      # Side-by-side: input image vs predicted label
│
├── 🧪 test_pictures/               # Sample input images for quick testing
│   ├── sample_A.png
│   ├── sample_B.png
│   └── ...                         # One image per character class
│
├── docker-compose.yml              # Docker Compose configuration
└── 📄 README.md
```
 
---
 
## ✨ Features
 
- ✅ Full OCR pipeline — from raw image to predicted character
- ✅ CNN-based model trained on handwritten character datasets
- ✅ REST API for real-time inference (single image → prediction + confidence)
- ✅ Ready-to-use test images so you can try it immediately without your own data
- ✅ Dockerized deployment — spin it up anywhere in one command
- ✅ Reproducible Jupyter notebooks with step-by-step training
- ✅ Rich evaluation visuals — confusion matrix, training curves, sample predictions
---
 
## 🚀 Getting Started
 
### Option 1 — Run Locally
 
#### Prerequisites
 
- Python 3.8+
- pip
#### Installation
 
```bash
# Clone the repository
git clone https://github.com/mariamhamza3019-star/Handwritten-Character-Recognition-OCR.git
cd Handwritten-Character-Recognition-OCR
 
# Install dependencies
pip install -r api/requirements.txt
```
 
#### Launch the API
 
```bash
cd api/
python app.py
```
 
The server will start at `http://localhost:5000`.
 
---
 
### Option 2 — Run with Docker 🐳
 
No environment setup needed. Docker handles everything.
 
```bash
# Build and start the container
docker-compose up --build
```
 
Or using just Docker:
 
```bash
docker build -t ocr-api ./api
docker run -p 5000:5000 ocr-api
```
 
The API will be available at `http://localhost:5000`.
 
---
 
## 🧪 Testing with Sample Images
 
The `test_pictures/` folder contains ready-to-use handwritten character images — one per character class. Use them to immediately verify the API is working after setup, no need to source your own test data.
 
```bash
# Test with a single sample image
curl -X POST http://localhost:5000/predict \
  -F "image=@test_pictures/sample_A.png"
```
 
Expected response:
 
```json
{
  "prediction": "A",
  "confidence": 0.97
}
```
 
You can loop through all test images to do a quick batch sanity check:
 
```bash
for img in test_pictures/*.png; do
  echo -n "$img → "
  curl -s -X POST http://localhost:5000/predict -F "image=@$img" | python3 -m json.tool
done
```
 
Or test programmatically in Python:
 
```python
import requests, os
 
api_url = "http://localhost:5000/predict"
 
for filename in os.listdir("test_pictures"):
    filepath = os.path.join("test_pictures", filename)
    with open(filepath, "rb") as f:
        response = requests.post(api_url, files={"image": f})
    print(f"{filename}: {response.json()}")
```
 
---
 
## 📓 Notebooks Walkthrough
 
The `notebooks/` folder is a complete, self-contained training environment. Run them in order:
 
| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `data_exploration.ipynb` | Loads the dataset, visualizes class distribution, inspects sample images |
| 02 | `preprocessing.ipynb` | Applies grayscale conversion, noise removal, binarization, resizing, and augmentation |
| 03 | `model_training.ipynb` | Defines CNN architecture, trains the model, saves weights to `models/` |
| 04 | `evaluation.ipynb` | Runs the trained model on the test set, plots confusion matrix and accuracy curves |
 
```bash
# Launch Jupyter
jupyter notebook notebooks/
```
 
---
 
## 🧠 Model Architecture
 
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
| Training Accuracy | ~98% |
| Validation Accuracy | ~95% |
| Test Accuracy | ~93% |
 
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
| TensorFlow / Keras | Model training & inference |
| OpenCV | Image preprocessing |
| Flask / FastAPI | REST API server |
| Docker | Containerized deployment |
| Matplotlib / Seaborn | Training visualizations |
| Jupyter Notebook | Interactive experimentation |
 
---
 
## 🤝 Contributing
 
Contributions, issues, and feature requests are welcome!
 
1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request
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
