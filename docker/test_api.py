import sys
import io
import time
import requests
from PIL import Image, ImageDraw

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
    status = "✅ PASS" if passed else "❌ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if passed:
        PASSED += 1
    else:
        FAILED += 1


# ─────────────────────────────────────────────
# TEST 1: Health Check
# ─────────────────────────────────────────────

def test_health():
    print("\n[1] Health Check  →  GET /health")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        test("Status code 200", resp.status_code == 200, f"got {resp.status_code}")
        data = resp.json()
        test("Has 'status' field",      "status"      in data)
        test("Status is 'ok'",          data.get("status") == "ok")
        test("Has 'device' field",      "device"      in data)
        test("Has 'classes' list",      isinstance(data.get("classes"), list))
        test("Has 'models' dict",       isinstance(data.get("models"), dict))
        test("26 classes",              data.get("num_classes") == 26,
             f"got {data.get('num_classes')}")
        models = data.get("models", {})
        test("CNN model listed",        "cnn"       in models)
        test("MobileNet model listed",  "mobilenet" in models)
    except requests.exceptions.ConnectionError:
        print("   FATAL: Cannot connect to API. Is the Docker container running?")
        print("   Run: docker run -d -p 8000:8000 --name ocr-container ocr-api")
        sys.exit(1)


# ─────────────────────────────────────────────
# TEST 2: /predict-compare  (single character)
# ─────────────────────────────────────────────

def test_predict_compare():
    print("\n[2] /predict-compare  →  POST (single character, both models)")

    # ── Test A: white letter on black background (dataset-style) ──
    img_bytes = make_white_on_black("A")
    resp = requests.post(
        f"{BASE_URL}/predict-compare",
        files={"file": ("test_A_black.png", img_bytes, "image/png")},
        timeout=15,
    )
    test("White-on-black image returns 200", resp.status_code == 200,
         f"got {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        # Response must have both model keys
        test("Response has 'cnn' key",      "cnn"      in data)
        test("Response has 'mobilenet' key","mobilenet" in data)

        # Validate CNN top-3 list
        cnn_results = data.get("cnn", [])
        test("CNN returns a list",          isinstance(cnn_results, list))
        test("CNN returns 3 predictions",   len(cnn_results) == 3,
             f"got {len(cnn_results)}")
        if cnn_results:
            first = cnn_results[0]
            test("CNN top-1 has 'character' field",  "character"  in first)
            test("CNN top-1 has 'confidence' field", "confidence" in first)
            test("CNN character is single letter",
                 len(first.get("character", "")) == 1)
            conf = first.get("confidence", -1)
            test("CNN confidence between 0 and 1",   0 <= conf <= 1,
                 f"got {conf}")

        # Validate MobileNet top-3 list
        mob_results = data.get("mobilenet", [])
        test("MobileNet returns a list",         isinstance(mob_results, list))
        test("MobileNet returns 3 predictions",  len(mob_results) == 3,
             f"got {len(mob_results)}")
        if mob_results:
            first = mob_results[0]
            test("MobileNet top-1 has 'character' field",  "character"  in first)
            test("MobileNet top-1 has 'confidence' field", "confidence" in first)
            conf = first.get("confidence", -1)
            test("MobileNet confidence between 0 and 1",   0 <= conf <= 1,
                 f"got {conf}")

    # ── Test B: dark letter on white background (real-world photo style) ──
    img_bytes = make_black_on_white("B")
    resp = requests.post(
        f"{BASE_URL}/predict-compare",
        files={"file": ("test_B_white.png", img_bytes, "image/png")},
        timeout=15,
    )
    test("Black-on-white image returns 200", resp.status_code == 200,
         f"got {resp.status_code}")

    # ── Test C: non-image file must be rejected ──
    resp = requests.post(
        f"{BASE_URL}/predict-compare",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    test("Non-image file returns 400", resp.status_code == 400,
         f"got {resp.status_code}")

    # ── Test D: corrupted image bytes must be rejected ──
    resp = requests.post(
        f"{BASE_URL}/predict-compare",
        files={"file": ("corrupt.png", b"\x89PNG corrupted!!!!", "image/png")},
        timeout=10,
    )
    test("Corrupted image returns 400", resp.status_code == 400,
         f"got {resp.status_code}")


# ─────────────────────────────────────────────
# TEST 3: /predict-word-compare  (full word)
# ─────────────────────────────────────────────

def test_predict_word_compare():
    print("\n[3] /predict-word-compare  →  POST (word segmentation, both models)")

    img_bytes = make_word_image("CAT")
    resp = requests.post(
        f"{BASE_URL}/predict-word-compare",
        files={"file": ("word_CAT.png", img_bytes, "image/png")},
        timeout=20,
    )
    test("Word image returns 200", resp.status_code == 200,
         f"got {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        test("Response has 'cnn' key",       "cnn"      in data)
        test("Response has 'mobilenet' key", "mobilenet" in data)

        for model_name in ["cnn", "mobilenet"]:
            result = data.get(model_name, {})
            test(f"{model_name}: has 'word' field",
                 "word" in result)
            test(f"{model_name}: has 'characters' list",
                 isinstance(result.get("characters"), list))
            test(f"{model_name}: has 'num_characters' field",
                 "num_characters" in result)

            chars = result.get("characters", [])
            if chars:
                test(f"{model_name}: each char has 'character' field",
                     all("character"  in c for c in chars))
                test(f"{model_name}: each char has 'confidence' field",
                     all("confidence" in c for c in chars))
                test(f"{model_name}: each char has 'position' field",
                     all("position"   in c for c in chars))
                test(f"{model_name}: positions in order",
                     [c["position"] for c in chars] == list(range(1, len(chars) + 1)))
                test(f"{model_name}: each char has 'top3' list",
                     all(isinstance(c.get("top3"), list) for c in chars))

            word = result.get("word", "")
            test(f"{model_name}: predicted word is non-empty string",
                 len(word) > 0, f"got '{word}'")

    # ── Invalid file ──
    resp = requests.post(
        f"{BASE_URL}/predict-word-compare",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    test("Non-image file returns 400", resp.status_code == 400,
         f"got {resp.status_code}")


# ─────────────────────────────────────────────
# TEST 4: Response Time
# ─────────────────────────────────────────────

def test_response_time():
    print("\n[4] Response Time  →  5 requests to /predict-compare")
    img_bytes = make_white_on_black("X")

    times = []
    for _ in range(5):
        start = time.time()
        requests.post(
            f"{BASE_URL}/predict-compare",
            files={"file": ("test.png", img_bytes, "image/png")},
            timeout=30,
        )
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    test(f"Average response < 5s",  avg      < 5.0,  f"avg={avg:.3f}s")
    test(f"Max response    < 10s",  max(times) < 10.0, f"max={max(times):.3f}s")
    print(f"    Avg: {avg:.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")


# ─────────────────────────────────────────────
# TEST 5: Swagger Docs
# ─────────────────────────────────────────────

def test_swagger_docs():
    print("\n[5] Swagger Docs")
    resp = requests.get(f"{BASE_URL}/docs", timeout=5)
    test("Swagger UI accessible at /docs", resp.status_code == 200)

    resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    test("OpenAPI schema at /openapi.json", resp.status_code == 200)

    if resp.status_code == 200:
        schema = resp.json()
        paths = schema.get("paths", {})
        test("Schema has /predict-compare endpoint",
             "/predict-compare"      in paths)
        test("Schema has /predict-word-compare endpoint",
             "/predict-word-compare" in paths)
        test("Schema has /health endpoint",
             "/health"               in paths)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Handwritten OCR API — Automated Test Suite")
    print(f"  Target: {BASE_URL}")
    print("=" * 55)

    test_health()
    test_predict_compare()
    test_predict_word_compare()
    test_response_time()
    test_swagger_docs()

    print("\n" + "=" * 55)
    total = PASSED + FAILED
    print(f"  Results: {PASSED}/{total} passed  |  {FAILED} failed")
    print("=" * 55)

    sys.exit(0 if FAILED == 0 else 1)