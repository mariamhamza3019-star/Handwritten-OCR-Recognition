from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import io
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH          = "best_model_v2.pth"      # SimpleCNN weights
TRANSFER_MODEL_PATH = "Transfer_Final.pth"     # MobileNetV2 fine-tuned weights (Phase 3 best)
NUM_CLASSES         = 26
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES         = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

IMG_SIZE_CNN      = 28
IMG_SIZE_TRANSFER = 128   # must match notebook IMG_SIZE = 128

# ── SimpleCNN (unchanged from original) ───────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # → (32, 14, 14)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # → (64, 7, 7)

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


# ── MobileNetV2 — must exactly mirror build_model() in the notebook ────────────
# Notebook trains with IMAGENET1K_V1 weights then saves full state_dict.
# At inference we use weights=None and load everything from the .pth file.
def build_mobilenet(num_classes: int = 26) -> nn.Module:
    model = models.mobilenet_v2(weights=None)   # architecture only; weights come from .pth
    in_features = model.classifier[1].in_features  # 1280

    # Head must match the notebook's build_model() classifier exactly
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


# ── Transforms ────────────────────────────────────────────────────────────────
# CNN: grayscale 28×28, normalised to [-1, 1]
transform_cnn = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# MobileNet: convert to RGB *first* (smart_preprocess returns grayscale PIL),
# then resize and apply ImageNet normalisation.
# Order matters: Grayscale→RGB→Resize matches the notebook's CharDataset
# which stacks the single channel 3× before applying transforms.
transform_transfer = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),   # L → RGB  (no info added, just channel copy)
    transforms.Resize(
        (IMG_SIZE_TRANSFER, IMG_SIZE_TRANSFER),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model loading helpers ──────────────────────────────────────────────────────
def _load_weights(model: nn.Module, path: str, label: str) -> nn.Module:
    try:
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"[OK]  Loaded {label}: {path}")
    except FileNotFoundError:
        print(f"[WARN] Not found: {path} — running with random weights (test mode only)")
    except RuntimeError as e:
        print(f"[ERR]  Architecture mismatch loading {path}: {e}")
    except Exception as e:
        print(f"[ERR]  Could not load {path}: {e}")
    model.to(DEVICE)
    model.eval()
    return model


def load_cnn_model(path: str) -> nn.Module:
    return _load_weights(SimpleCNN(NUM_CLASSES), path, "SimpleCNN")


def load_transfer_model(path: str) -> nn.Module:
    return _load_weights(build_mobilenet(NUM_CLASSES), path, "MobileNetV2")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def smart_preprocess(pil_image: Image.Image) -> Image.Image:
    """
    Normalises an input image so it looks like dataset images:
    white letter on black background, tightly cropped, square.

    Works for both:
      • Real-world photos  (dark ink on white paper) → inverts
      • Dataset-style      (white letter on black)   → passes through
    """
    img_np = np.array(pil_image.convert("L"))
    h, w   = img_np.shape

    # Detect background colour from border pixels
    border = np.concatenate([
        img_np[0, :], img_np[-1, :],
        img_np[:, 0], img_np[:, -1],
    ])
    if border.mean() > 128:          # light background → real-world photo
        img_np = 255 - img_np

    # Otsu threshold → clean binary image
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tight crop around the letter
    coords = cv2.findNonZero(binary)
    if coords is not None:
        pad = 4
        x, y, cw, ch = cv2.boundingRect(coords)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad);  y2 = min(h, y + ch + pad)
        binary = binary[y1:y2, x1:x2]

    # Pad to square so resize doesn't distort aspect ratio
    ch, cw = binary.shape
    side    = max(ch, cw)
    squared = np.zeros((side, side), dtype=np.uint8)
    squared[(side - ch) // 2:(side - ch) // 2 + ch,
            (side - cw) // 2:(side - cw) // 2 + cw] = binary

    # Return as grayscale PIL image; each transform pipeline handles RGB conversion
    return Image.fromarray(squared)


# ── Segmentation ──────────────────────────────────────────────────────────────
def segment_characters(pil_image: Image.Image) -> list[Image.Image]:
    """
    Splits a word image into individual character PIL images (left → right).
    Uses connected-component analysis on a binarised version of the input.
    """
    img_np = np.array(pil_image.convert("L"))

    border = np.concatenate([img_np[0, :], img_np[-1, :], img_np[:, 0], img_np[:, -1]])
    if border.mean() > 128:
        img_np = 255 - img_np

    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Slight dilation to reconnect broken strokes within one character
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)

    char_images = []
    MIN_AREA    = 10   # ignore tiny noise blobs

    for i in range(1, num_labels):   # 0 = background
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        char_images.append((x, Image.fromarray(binary[y:y + ch, x:x + cw])))

    char_images.sort(key=lambda t: t[0])
    return [img for _, img in char_images]


# ── Prediction helpers ─────────────────────────────────────────────────────────
def predict_top3_cnn(pil_image: Image.Image) -> list[dict]:
    preprocessed = smart_preprocess(pil_image)
    tensor = transform_cnn(preprocessed).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(tensor), dim=1)[0]
    top3 = probs.topk(3)
    return [
        {"character": CLASS_NAMES[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top3.values, top3.indices)
    ]


def predict_top3_transfer(pil_image: Image.Image) -> list[dict]:
    preprocessed = smart_preprocess(pil_image)
    tensor = transform_transfer(preprocessed).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(transfer_model(tensor), dim=1)[0]
    top3 = probs.topk(3)
    return [
        {"character": CLASS_NAMES[i], "confidence": round(p.item(), 4)}
        for p, i in zip(top3.values, top3.indices)
    ]


# ── Load models at startup ─────────────────────────────────────────────────────
cnn_model      = load_cnn_model(MODEL_PATH)
transfer_model = load_transfer_model(TRANSFER_MODEL_PATH)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Handwritten Character Recognition API",
    description="SimpleCNN vs MobileNetV2 — A-Z recognition",
    version="2.0.0",
)


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "models": {
            "cnn": "SimpleCNN (best_model_v2.pth)",
            "mobilenet": "MobileNetV2 fine-tuned (Transfer_Final.pth)",
        },
        "img_sizes": {
            "cnn": IMG_SIZE_CNN,
            "mobilenet": IMG_SIZE_TRANSFER,
        },
    }


@app.post("/predict-compare", summary="Top-3 predictions from both models — single character")
async def predict_compare(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        pil_image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    return {
        "cnn":      predict_top3_cnn(pil_image),
        "mobilenet": predict_top3_transfer(pil_image),
    }


@app.post("/predict-word-compare", summary="Compare word prediction from both models")
async def predict_word_compare(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        pil_image = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read image file.")

    char_images = segment_characters(pil_image)
    if not char_images:
        raise HTTPException(422, "No characters detected in the image.")

    def build_result(predictor):
        characters = []
        for i, char_img in enumerate(char_images):
            top3 = predictor(char_img)
            characters.append({
                "position":   i + 1,
                "character":  top3[0]["character"],
                "confidence": top3[0]["confidence"],
                "top3":       top3,
            })
        word = "".join(c["character"] for c in characters)
        return {"word": word, "num_characters": len(characters), "characters": characters}

    return {
        "cnn":      build_result(predict_top3_cnn),
        "mobilenet": build_result(predict_top3_transfer),
    }


# Run: uvicorn app:app --reload --port 8000
# Docs: http://localhost:8000/docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
