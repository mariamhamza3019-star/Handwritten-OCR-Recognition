import streamlit as st
import requests
from PIL import Image
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="centered",
)

# HEADER
st.title("✍️ Handwritten Character Recognition")
st.markdown(
    "Upload a handwritten letter or word image and let the model recognize it.\n\n"
    "> Works with **dark ink on white paper** (real-world) "
    "and **white letter on black background** (dataset-style)."
)

# CHECK API HEALTH
with st.sidebar:
    st.header("⚙️ API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API is running ✅")
        st.write(f"**Device:** `{health.get('device', 'unknown')}`")
        st.write(f"**Classes:** {health.get('num_classes', '?')} (A-Z)")
        st.markdown("---")
        st.markdown("📖 [View Swagger Docs](http://localhost:8000/docs)")
    except Exception:
        st.error("❌ API not reachable.\nMake sure `app.py` is running.")
        st.code("python app.py", language="bash")


# TABS: Single Character vs Word

tab1, tab2= st.tabs(["🔤 Single Character", "📝 Word"])

# ───── TAB 1: Single Character ─────
with tab1:
    st.subheader("Predict a single handwritten letter")
    uploaded = st.file_uploader(
        "Upload an image of one handwritten character",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_uploader",
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

        with col2:
            if st.button("🔍 Predict Character", use_container_width=True):
                with st.spinner("Running prediction..."):
                    try:
                        uploaded.seek(0)
                        files = {"file": (uploaded.name, uploaded, uploaded.type)}
                        resp = requests.post(f"{API_URL}/predict", files=files, timeout=10)

                        if resp.status_code == 200:
                            data = resp.json()
                            char = data["character"]
                            conf = data["confidence"]

                            st.markdown(f"## Predicted: **`{char}`**")
                            st.metric("Confidence", f"{conf * 100:.1f}%")

                            # Confidence bar
                            color = "green" if conf > 0.7 else ("orange" if conf > 0.4 else "red")
                            st.progress(conf)

                            if conf < 0.5:
                                st.warning(
                                    "⚠️ Low confidence. The model isn't very sure. "
                                    "Try a cleaner image with higher contrast."
                                )
                        else:
                            st.error(f"API Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API. Is `app.py` running?")

# ───── TAB 2: Word ─────
with tab2:
    st.subheader("Predict a handwritten word")
    st.info(
        "💡 **Tips for word prediction:**\n"
        "- Characters should have visible gaps between them\n"
        "- Cursive / heavily connected writing may not segment correctly\n"
        "- Printed/block handwriting works best"
    )

    uploaded_word = st.file_uploader(
        "Upload an image of a handwritten word",
        type=["png", "jpg", "jpeg", "bmp"],
        key="word_uploader",
    )

    if uploaded_word:
        st.image(uploaded_word, caption="Uploaded word image", use_container_width=True)

        if st.button("🔍 Predict Word", use_container_width=True):
            with st.spinner("Segmenting characters and predicting..."):
                try:
                    uploaded_word.seek(0)
                    files = {"file": (uploaded_word.name, uploaded_word, uploaded_word.type)}
                    resp = requests.post(f"{API_URL}/predict-word", files=files, timeout=15)

                    if resp.status_code == 200:
                        data = resp.json()
                        word = data["word"]
                        chars = data["characters"]
                        n = data["num_characters"]

                        st.markdown(f"## Predicted Word: **`{word}`**")
                        st.write(f"Detected **{n}** character(s)")

                        # Per-character breakdown table
                        st.markdown("### Per-character breakdown")
                        col_headers = st.columns([1, 2, 3])
                        col_headers[0].markdown("**#**")
                        col_headers[1].markdown("**Letter**")
                        col_headers[2].markdown("**Confidence**")

                        for c in chars:
                            cols = st.columns([1, 2, 3])
                            cols[0].write(c["position"])
                            cols[1].markdown(f"### `{c['character']}`")
                            conf = c["confidence"]
                            conf_color = "🟢" if conf > 0.7 else ("🟡" if conf > 0.4 else "🔴")
                            cols[2].write(f"{conf_color} {conf * 100:.1f}%")

                    elif resp.status_code == 422:
                        st.error(f"Could not detect characters: {resp.json().get('detail')}")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to API. Is `app.py` running?")


st.markdown("---")
st.caption("Advanced ML Team Project · Member 6: API Developer · Powered by FastAPI + PyTorch")
