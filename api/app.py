# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch, io
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import cv2
import numpy as np
import io

MODEL_PATH          = "best_model_v2.pth"          
# TRANSFER_MODEL_PATH = "best_model_transfer.pth"  
NUM_CLASSES = 26                     
MODEL_TYPE = "cnn"                   #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu instead of cpu
CLASS_NAMES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class SimpleCNN(nn.Module):
#Exact replica of CNNModel 
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # → (32, 14, 14)

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # → (64, 7, 7)

            # Block 3 — extra depth, no pooling to keep spatial info
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# def build_mobilenet(num_classes=26) -> nn.Module:
#     model = models.mobilenet_v2(weights=None)
#     # MobileNetV2 expects 3-channel input — keep original first conv
#     # We'll replicate grayscale to 3 channels in preprocessing instead
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3),
#         nn.Linear(model.last_channel, num_classes)
#     )
#     return model

# def load_model(path: str) -> nn.Module:
#     m = build_mobilenet(NUM_CLASSES)
#     try:
#         checkpoint = torch.load(path, map_location=DEVICE)
#         state = checkpoint.get('model_state_dict', checkpoint)
#         m.load_state_dict(state)
#         print(f"Loaded: {path}")
#     except FileNotFoundError:
#         print(f"Not found: {path}")
#     except Exception as e:
#         print(f"Could not load {path}: {e}")
#     m.to(DEVICE)
#     m.eval()
#     return m

# transform_mobilenet = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((224, 224)),
#     transforms.Lambda(lambda img: img.convert("RGB")),  # grayscale → RGB
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


def load_model_from(path: str) -> nn.Module: #btakhod file path w bt7wlo l pytorch model 
    m = SimpleCNN(NUM_CLASSES) #bt3ml empty SimpleCNN model with random weights (docker container thing ig)
    try:
        state = torch.load(path, map_location=DEVICE)
        m.load_state_dict(state)
        print(f" Loaded: {path}")
    except FileNotFoundError:
        print(f" Not found: {path} — using random weights (testing only)")
    except Exception as e:
        print(f" Could not load {path}: {e}")
    m.to(DEVICE)
    m.eval()
    return m

IMG_SIZE = 224 if MODEL_TYPE == "mobilenet" else 28
 
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])


def smart_preprocess(pil_image: Image.Image) -> Image.Image:
    #Handles BOTH dataset-style (white letter on black bg)
    #and real-world-style (dark letter on white bg) images.
    """
    Strategy:
    1. Convert to grayscale
    2. Detect background color (if mostly light → it's a real-world photo → invert it)
    3. Threshold to get clean binary image
    4. Crop tightly around the letter (remove whitespace)
    5. Add small padding so letter isn't touching the edge
    """
    img = pil_image.convert("L")  # grayscale
    img_np = np.array(img)
 
    # Step 1: Detect if background is light (real-world) or dark (dataset-style)
    # Check the average brightness of the border pixels (corners + edges)
    
    h, w = img_np.shape
    border = np.concatenate([
        img_np[0, :], img_np[-1, :],      # top and bottom rows
        img_np[:, 0], img_np[:, -1]       # left and right columns
    ])
    avg_border = border.mean()
    # If border is mostly bright → real-world photo (dark ink on white paper) → invert
    if avg_border > 128:
        img_np = 255 - img_np
    # Step 2: Threshold — keep only the letter (white on black after inversion)
    _, binary = cv2.threshold(img_np, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Step 3: Find bounding box of the letter and crop tightly
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, cw, ch = cv2.boundingRect(coords)
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        binary = binary[y1:y2, x1:x2]
    # Step 4: Make square (pad shorter side) so resize doesn't distort aspect ratio
    ch, cw = binary.shape
    side = max(ch, cw)
    squared = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    squared[y_off:y_off+ch, x_off:x_off+cw] = binary
 
    return Image.fromarray(squared)

def predict_single(pil_image: Image.Image) -> dict:
    preprocessed = smart_preprocess(pil_image)
    tensor = transform(preprocessed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = probs.max(dim=1)

    label = CLASS_NAMES[predicted_idx.item()]
    conf = round(confidence.item(), 4)
    return {"character": label, "confidence": conf}

def segment_characters(pil_image: Image.Image) -> list[Image.Image]:
    """
    Splits a word image into individual character images.
    Uses connected component analysis on a preprocessed binary image.
    Returns list of character PIL images sorted left-to-right.
    """
    img = pil_image.convert("L")
    img_np = np.array(img)
 
    # Invert if needed (same logic as smart_preprocess)
    border = np.concatenate([img_np[0, :], img_np[-1, :], img_np[:, 0], img_np[:, -1]])
    if border.mean() > 128:
        img_np = 255 - img_np
 
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # Dilate slightly to connect broken strokes within a single character
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
 
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
 
    char_images = []
    min_area = 10  # ignore tiny noise blobs
 
    for i in range(1, num_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        char_crop = binary[y:y+h, x:x+w]
        char_images.append((x, Image.fromarray(char_crop)))  # keep x for sorting
 
    # Sort left → right
    char_images.sort(key=lambda t: t[0])
    return [img for _, img in char_images]

app = FastAPI(title="Handwritten Character Recognition API")
model = load_model_from(MODEL_PATH)  # Load trained model
# transfer_model = load_model_from(TRANSFER_MODEL_PATH)  # fine-tuned CNN

@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }

@app.post("/predict", summary="Predict a single handwritten character")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        pil_image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    return predict_single(pil_image)


@app.post("/predict-word", summary="Predict a handwritten word")
async def predict_word(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        pil_image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    char_images = segment_characters(pil_image)

    if not char_images:
        raise HTTPException(status_code=422, detail="No characters detected.")

    characters = []
    for i, char_img in enumerate(char_images):
        result = predict_single(char_img)
        result["position"] = i + 1
        characters.append(result)

    word = "".join(c["character"] for c in characters)
    return {"word": word, "num_characters": len(characters), "characters": characters}


# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
 