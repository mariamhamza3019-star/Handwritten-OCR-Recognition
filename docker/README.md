# OCR Handwritten Character Recognition API

## What this project does
Recognizes handwritten letters (A-Z) from images using deep learning.

## How to run with Docker

### 1. Build the image
```bash
docker build -t ocr-api .
```

### 2. Run the container
```bash
docker run -p 8000:8000 ocr-api
```

### 3. Test it
Open your browser at: http://localhost:8000/docs

Or run the test script:
```bash
python test_api.py
```

## API Endpoints
- `POST /predict` — Send an image, get back a character and confidence score
- `POST /predict-word` — Send an image of a word, get back text

## Example Response
```json
{"character": "A", "confidence": 0.95}
```