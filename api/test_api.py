import sys
import io
import time
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://localhost:8000"
PASSED = 0
FAILED = 0


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def make_white_on_black(letter: str, size=64) -> bytes:
    """Creates a dataset-style image: white letter on black background."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((size // 4, size // 8), letter, fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_black_on_white(letter: str, size=64) -> bytes:
    """Creates a real-world-style image: dark letter on white background."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((size // 4, size // 8), letter, fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_word_image(word: str, size=64) -> bytes:
    """Creates a simple word image with spaced-out characters."""
    w = size * len(word)
    img = Image.new("L", (w, size), color=255)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(word):
        draw.text((i * size + size // 4, size // 8), ch, fill=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test(name: str, passed: bool, detail: str = ""):
    global PASSED, FAILED
    status = " PASS" if passed else " FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if passed:
        PASSED += 1
    else:
        FAILED += 1


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────

def test_health():
    print("\n[1] Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        test("Status code 200", resp.status_code == 200, f"got {resp.status_code}")
        data = resp.json()
        test("Has 'status' field", "status" in data)
        test("Status is 'ok'", data.get("status") == "ok")
        test("Has 'model_type' field", "model_type" in data)
        test("Has 'classes' list", isinstance(data.get("classes"), list))
        test("26 or 36 classes", data.get("num_classes") in [26, 36], f"got {data.get('num_classes')}")
    except requests.exceptions.ConnectionError:
        print("   FATAL: Cannot connect to API. Is app.py running?")
        sys.exit(1)


def test_predict_single():
    print("\n[2] /predict — Single Character")

    # Test 1: White on black (dataset-style)
    img_bytes = make_white_on_black("A")
    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test_A_black.png", img_bytes, "image/png")},
        timeout=10,
    )
    test("White-on-black image returns 200", resp.status_code == 200, f"got {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        test("Response has 'character' field", "character" in data)
        test("Response has 'confidence' field", "confidence" in data)
        test("Character is a single uppercase letter", len(data.get("character", "")) == 1)
        conf = data.get("confidence", 0)
        test("Confidence is between 0 and 1", 0 <= conf <= 1, f"got {conf}")

    # Test 2: Black on white (real-world style)
    img_bytes = make_black_on_white("B")
    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test_B_white.png", img_bytes, "image/png")},
        timeout=10,
    )
    test("Black-on-white image returns 200", resp.status_code == 200, f"got {resp.status_code}")

    # Test 3: Invalid file (not an image)
    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    test("Non-image file returns 400", resp.status_code == 400, f"got {resp.status_code}")

    # Test 4: Corrupted image bytes
    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("corrupt.png", b"\x89PNG corrupted data!!!!", "image/png")},
        timeout=10,
    )
    test("Corrupted image returns 400", resp.status_code == 400, f"got {resp.status_code}")


def test_predict_word():
    print("\n[3] /predict-word — Word Segmentation")

    img_bytes = make_word_image("CAT")
    resp = requests.post(
        f"{BASE_URL}/predict-word",
        files={"file": ("word_CAT.png", img_bytes, "image/png")},
        timeout=15,
    )
    test("Word image returns 200", resp.status_code == 200, f"got {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        test("Response has 'word' field", "word" in data)
        test("Response has 'characters' list", isinstance(data.get("characters"), list))
        test("Response has 'num_characters' field", "num_characters" in data)
        chars = data.get("characters", [])
        if chars:
            test("Each character has 'character' field", all("character" in c for c in chars))
            test("Each character has 'confidence' field", all("confidence" in c for c in chars))
            test("Each character has 'position' field", all("position" in c for c in chars))
            test("Positions are in order", [c["position"] for c in chars] == list(range(1, len(chars)+1)))
        predicted_word = data.get("word", "")
        test("Predicted word is a non-empty string", len(predicted_word) > 0, f"got '{predicted_word}'")

    # Test invalid file
    resp = requests.post(
        f"{BASE_URL}/predict-word",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    test("Non-image file returns 400", resp.status_code == 400, f"got {resp.status_code}")


def test_response_time():
    print("\n[4] Response Time")
    img_bytes = make_white_on_black("X")

    times = []
    for _ in range(5):
        start = time.time()
        requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
            timeout=10,
        )
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    test(f"Average response time < 5s", avg < 5.0, f"avg={avg:.3f}s")
    test(f"Max response time < 10s", max(times) < 10.0, f"max={max(times):.3f}s")
    print(f"    Avg: {avg:.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")


def test_swagger_docs():
    print("\n[5] Swagger Docs")
    resp = requests.get(f"{BASE_URL}/docs", timeout=5)
    test("Swagger UI accessible at /docs", resp.status_code == 200)
    resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    test("OpenAPI schema accessible at /openapi.json", resp.status_code == 200)
    if resp.status_code == 200:
        schema = resp.json()
        test("Schema has /predict endpoint", "/predict" in schema.get("paths", {}))
        test("Schema has /predict-word endpoint", "/predict-word" in schema.get("paths", {}))
        test("Schema has /health endpoint", "/health" in schema.get("paths", {}))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Handwritten OCR API — Automated Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 55)

    test_health()
    test_predict_single()
    test_predict_word()
    test_response_time()
    test_swagger_docs()

    print("\n" + "=" * 55)
    total = PASSED + FAILED
    print(f"  Results: {PASSED}/{total} passed  |  {FAILED} failed")
    print("=" * 55)

    sys.exit(0 if FAILED == 0 else 1)
