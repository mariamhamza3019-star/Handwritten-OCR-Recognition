<div align="center">
# 🖋️ Handwritten Character Recognition — OCR
 
### Teaching machines to read human handwriting, one stroke at a time.
 
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()
 
</div>
---
 
## 🔍 Overview
 
This project implements an **Optical Character Recognition (OCR)** system specifically trained to recognize **handwritten characters**. From raw pen strokes to structured text — this pipeline handles preprocessing, feature extraction, model training, and inference via a clean API.
 
Whether you're digitizing handwritten notes, building assistive tools, or just fascinated by the intersection of human writing and machine learning — this repo has you covered.
 
---
 
## 🗂️ Project Structure
 
```
Handwritten-Character-Recognition-OCR/
│
├── 📓 notebooks/         # Exploratory data analysis & model training notebooks
├── 🤖 models/            # Trained model weights & architecture files
├── 🌐 api/               # REST API for inference (serve predictions over HTTP)
├── 📊 visuals/         # Plots, confusion matrices, training curves
├── 🗂️ docker/
├── ✨ test pic/
└── 📄 README.md
```
 
---
 
## ✨ Features
 
- ✅ End-to-end handwritten character recognition pipeline
- ✅ Trained on real handwriting datasets
- ✅ Modular architecture — swap models with ease
- ✅ REST API for serving predictions
- ✅ Training visualizations and performance metrics included
- ✅ Jupyter notebooks for full reproducibility
---
 
## 🚀 Getting Started
 
### Prerequisites
 
```bash
Python 3.8+
pip
```
 
### Installation
 
```bash
# Clone the repository
git clone https://github.com/mariamhamza3019-star/Handwritten-Character-Recognition-OCR.git
cd Handwritten-Character-Recognition-OCR
 
# Install dependencies
pip install -r requirements.txt
```
 
### Run the Notebooks
 
```bash
jupyter notebook notebooks/
```
 
### Start the API
 
```bash
cd api/
python app.py
```
 
Then send a POST request with your image:
 
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@your_handwritten_sample.png"
```
 
---
 
## 🧠 Model Architecture
 
The recognition pipeline consists of:
 
1. **Preprocessing** — grayscale conversion, noise removal, binarization & normalization
2. **Segmentation** — isolating individual characters from input images
3. **Feature Extraction** — CNN-based spatial feature learning
4. **Classification** — deep neural network outputting character probabilities
5. **Postprocessing** — sequence assembly and confidence scoring
---
 
## 📊 Results
 
| Metric | Score |
|--------|-------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~95% |
| Test Accuracy | ~93% |
 
> Detailed plots and confusion matrices are available in the `visuals/` folder.
 
---
 
## 📡 API Reference
 
**`POST /predict`**
 
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `file` | Image file of handwritten character(s) |
 
**Response:**
 
```json
{
  "prediction": "A",
  "confidence": 0.97
}
```
 
---
 
## 🛠️ Tech Stack
 
| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | Model training & inference |
| OpenCV | Image preprocessing |
| Flask | REST API |
| Matplotlib / Seaborn | Visualizations |
| Jupyter Notebook | Experimentation |
 
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
 
## 📄 License
 
This project is licensed under the **MIT License** — feel free to use, modify, and distribute.
 
---
 
<div align="center">
*If this project helped you, consider giving it a ⭐ — it means a lot!*
 
</div>
 
