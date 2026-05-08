import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Handwritten OCR",
    page_icon="✍️",
    layout="centered",
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("✍️ Handwritten Character Recognition")
st.markdown(
    "Upload a handwritten letter or word image and compare **SimpleCNN** vs **MobileNetV2** predictions.\n\n"
    "> Works with **dark ink on white paper** (real-world) "
    "and **white letter on black background** (dataset-style)."
)

# ── SIDEBAR: API STATUS ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API is running ✅")
        st.write(f"**Device:** `{health.get('device', 'unknown')}`")
        st.write(f"**Classes:** {health.get('num_classes', '?')} (A–Z)")

        models_info = health.get("models", {})
        img_sizes   = health.get("img_sizes", {})
        if models_info:
            st.markdown("**Models loaded:**")
            st.caption(f"🧠 {models_info.get('cnn', 'SimpleCNN')} — {img_sizes.get('cnn', 28)}px")
            st.caption(f"🚀 {models_info.get('mobilenet', 'MobileNetV2')} — {img_sizes.get('mobilenet', 128)}px")

        st.markdown("---")
        st.markdown("📖 [Swagger Docs](http://localhost:8000/docs)")
    except Exception:
        st.error("❌ API not reachable.\nMake sure `app.py` is running.")
        st.code("python app.py", language="bash")

# ── HELPER ────────────────────────────────────────────────────────────────────
def render_top3(predictions: list, model_label: str):
    """Render top-3 predictions as medal rows with progress bars."""
    st.markdown(f"### {model_label}")
    medals = ["🥇", "🥈", "🥉"]
    for rank, pred in enumerate(predictions):
        conf  = pred["confidence"]
        emoji = medals[rank] if rank < 3 else f"#{rank+1}"
        st.markdown(f"{emoji} **`{pred['character']}`** — {conf*100:.1f}%")
        st.progress(conf)


def render_word_result(data: dict, model_label: str):
    """Render word-level results with per-character breakdown."""
    st.markdown(f"### {model_label}")
    st.markdown(f"## `{data['word']}`")
    st.write(f"Detected **{data['num_characters']}** character(s)")

    if data["num_characters"] > 0:
        st.markdown("#### Per-character breakdown")
        for c in data["characters"]:
            st.markdown(f"**Position {c['position']}**")
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            for rank, pred in enumerate(c["top3"]):
                cols[rank].metric(
                    f"{medals[rank]} #{rank+1}",
                    pred["character"],
                    f"{pred['confidence']*100:.1f}%",
                )
            st.divider()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔤 Single Character", "📝 Word"])


# ───── TAB 1: Single Character ────────────────────────────────────────────────
with tab1:
    st.subheader("Compare CNN vs MobileNet — Single Character")

    uploaded = st.file_uploader(
        "Upload an image of one handwritten character",
        type=["png", "jpg", "jpeg", "bmp"],
        key="single_uploader",
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded image", width=200)

        if st.button("⚖️ Compare Models", use_container_width=True, key="single_compare_btn"):
            with st.spinner("Running both models…"):
                try:
                    uploaded.seek(0)
                    resp = requests.post(
                        f"{API_URL}/predict-compare",
                        files={"file": (uploaded.name, uploaded, uploaded.type)},
                        timeout=10,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        col1, col2 = st.columns(2)
                        with col1:
                            render_top3(data["cnn"], "🧠 SimpleCNN")
                        with col2:
                            render_top3(data["mobilenet"], "🚀 MobileNet")

                        # Agreement indicator
                        cnn_top  = data["cnn"][0]["character"]
                        mob_top  = data["mobilenet"][0]["character"]
                        if cnn_top == mob_top:
                            st.success(f"✅ Both models agree: **{cnn_top}**")
                        else:
                            st.warning(
                                f"⚠️ Models disagree — CNN: **{cnn_top}**, MobileNet: **{mob_top}**"
                            )
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to API. Is `app.py` running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ───── TAB 2: Word ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Compare CNN vs MobileNet — Word")

    uploaded_word = st.file_uploader(
        "Upload an image of a handwritten word",
        type=["png", "jpg", "jpeg", "bmp"],
        key="word_uploader",
    )

    if uploaded_word:
        st.image(uploaded_word, caption="Uploaded image", use_container_width=True)

        if st.button("⚖️ Compare Models", use_container_width=True, key="word_compare_btn"):
            with st.spinner("Segmenting characters and running both models…"):
                try:
                    uploaded_word.seek(0)
                    resp = requests.post(
                        f"{API_URL}/predict-word-compare",
                        files={"file": (uploaded_word.name, uploaded_word, uploaded_word.type)},
                        timeout=15,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        col1, col2 = st.columns(2)
                        with col1:
                            render_word_result(data["cnn"], "🧠 SimpleCNN")
                        with col2:
                            render_word_result(data["mobilenet"], "🚀 MobileNet")

                        # Agreement indicator
                        cnn_word = data["cnn"]["word"]
                        mob_word = data["mobilenet"]["word"]
                        if cnn_word == mob_word:
                            st.success(f"✅ Both models agree: **{cnn_word}**")
                        else:
                            st.warning(
                                f"⚠️ Models disagree — CNN: **{cnn_word}**, MobileNet: **{mob_word}**"
                            )

                    elif resp.status_code == 422:
                        st.error(f"No characters detected: {resp.json().get('detail')}")
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to API. Is `app.py` running?")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Advanced ML Team Project · Member 6: API Developer · Powered by FastAPI + PyTorch")
