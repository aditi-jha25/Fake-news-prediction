import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Inter from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: #0f0f14;
        color: #e8e8f0;
    }

    /* Header */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #7c7c96;
        margin-bottom: 2.5rem;
    }
    .accent { color: #7c6af7; }

    /* Card */
    .card {
        background: #18181f;
        border: 1px solid #2a2a38;
        border-radius: 14px;
        padding: 1.75rem 1.75rem 1.25rem;
        margin-bottom: 1.25rem;
    }

    /* Result badges */
    .badge-fake {
        display: inline-block;
        background: #3b1a1a;
        color: #f87171;
        border: 1px solid #7f1d1d;
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-real {
        display: inline-block;
        background: #0f2e1e;
        color: #4ade80;
        border: 1px solid #166534;
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    /* Confidence bar track */
    .conf-label {
        font-size: 0.78rem;
        color: #7c7c96;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    /* Streamlit overrides */
    .stTextArea textarea {
        background: #0f0f14 !important;
        border: 1px solid #2a2a38 !important;
        border-radius: 10px !important;
        color: #e8e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.94rem !important;
    }
    .stButton > button {
        background: #7c6af7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.6rem !important;
        transition: opacity 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #13131a !important;
        border-right: 1px solid #1e1e2e !important;
    }
    [data-testid="stSidebar"] * { color: #a0a0b8 !important; }

    /* Divider */
    hr { border-color: #2a2a38 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK setup
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    return stopwords.words("english")

STOP_WORDS = download_nltk()
port_stem = PorterStemmer()


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing (mirrors the notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def stemming(title: str) -> str:
    stemmed = re.sub("[^a-zA-Z]", " ", title)
    stemmed = stemmed.lower().split()
    stemmed = [port_stem.stem(w) for w in stemmed if w not in STOP_WORDS]
    return " ".join(stemmed)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — tries pickle files first, then re-trains on train.csv
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    """
    Priority:
      1. model.pkl + vectorizer.pkl  (pre-serialised files alongside app.py)
      2. train.csv in the working directory (re-trains on the fly)
    """
    model_path = "model.pkl"
    vec_path   = "vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, "loaded from saved files"

    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv").fillna("")
        df["title"] = df["title"].apply(stemming)
        X = df["title"].values
        Y = df["label"].values

        vectorizer = TfidfVectorizer()
        X_vec = vectorizer.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_vec, Y)

        # Persist for next run
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(vec_path, "wb") as f:
            pickle.dump(vectorizer, f)

        return model, vectorizer, "trained from train.csv"

    return None, None, "missing"


model, vectorizer, model_status = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model info")
    st.markdown(f"**Algorithm:** Logistic Regression")
    st.markdown(f"**Features:** TF-IDF on title")
    st.markdown(f"**Status:** {model_status}")
    st.markdown("---")
    st.markdown("### 📋 How it works")
    st.markdown(
        "1. Enter a news headline or article text.\n"
        "2. The model stems & vectorises it.\n"
        "3. Logistic Regression classifies it as **Real** or **Fake**.\n"
        "4. A confidence score is shown."
    )
    st.markdown("---")
    st.caption("Labels: 0 = Real · 1 = Fake")


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="hero-title">Fake News <span class="accent">Detector</span></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">Paste a news headline or article title and let the model decide.</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Guard: model not available
# ─────────────────────────────────────────────────────────────────────────────
if model is None:
    st.error(
        "**Model not found.**\n\n"
        "Place either:\n"
        "- `model.pkl` + `vectorizer.pkl` (pre-trained), **or**\n"
        "- `train.csv` (the original dataset)\n\n"
        "in the same directory as `app.py`, then restart the app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Input card
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
user_input = st.text_area(
    "News headline or article text",
    placeholder="e.g. Scientists discover new method to combat climate change …",
    height=140,
    label_visibility="visible",
)
predict_btn = st.button("Analyse →", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter some text before analysing.")
    else:
        processed = stemming(user_input)
        if not processed:
            st.warning("The input didn't contain enough meaningful words after preprocessing. Try a longer headline.")
        else:
            vec_input = vectorizer.transform([processed])
            prediction = model.predict(vec_input)[0]
            proba      = model.predict_proba(vec_input)[0]   # [P(real), P(fake)]

            label_idx  = int(prediction)                     # 0=real, 1=fake
            confidence = float(proba[label_idx]) * 100

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Result**")

            if label_idx == 1:
                st.markdown('<span class="badge-fake">⚠ Fake News</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-real">✔ Real News</span>', unsafe_allow_html=True)

            st.markdown(f'<p class="conf-label">Model confidence: {confidence:.1f}%</p>', unsafe_allow_html=True)
            st.progress(int(confidence))

            with st.expander("See preprocessed input"):
                st.code(processed, language=None)

            st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Try examples
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Try an example headline:**")

examples = {
    "🟢 Likely Real": "NASA's Perseverance rover collects first Mars rock sample",
    "🔴 Likely Fake": "Government secretly putting mind-control chips in COVID vaccines confirmed",
}

col1, col2 = st.columns(2)
for col, (label, text) in zip([col1, col2], examples.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["example_text"] = text
            st.rerun()

if "example_text" in st.session_state:
    ex = st.session_state.pop("example_text")
    st.info(f"**Loaded:** {ex}\n\nPaste it into the text box above and click **Analyse →**.")
