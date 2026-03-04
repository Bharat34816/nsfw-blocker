import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import re
import unicodedata
import emoji
import html
from ftfy import fix_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Custom layer
from .custom_layers import SelfAttention

TF_ENABLE_ONEDNN_OPTS=0
# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pickle")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")


# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"SelfAttention": SelfAttention}
    )

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, tokenizer, metadata["max_len"], metadata["class_labels"]


model, tokenizer, MAX_LEN, CLASS_LABELS = load_artifacts()


# -----------------------
# Preprocessing
# -----------------------
def preprocess_text(text):
    text = str(text)
    text = fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)

    text = emoji.demojize(text)
    text = re.sub(r':[a-z0-9_]+:', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -----------------------
# Prediction
# -----------------------
def predict_text(text):
    clean = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    preds = model.predict(padded, verbose=0)[0]

    label_scores = {
        CLASS_LABELS[str(i)]: float(preds[i])
        for i in range(len(preds))
    }

    nsfw_labels = {"hate", "offensive"}

    nsfw_score = max((v for k, v in label_scores.items() if k in nsfw_labels),
    default=0.0
)
    sfw_score = sum(v for k, v in label_scores.items() if k not in nsfw_labels)

    final_label = "NSFW" if nsfw_score >= sfw_score else "SFW"

    return clean, final_label, nsfw_score, sfw_score, label_scores


if __name__ == "__main__":
    # -----------------------
    # UI
    # -----------------------
    st.set_page_config(page_title="NSFW vs SFW", layout="centered")

    st.markdown(
        """
        <style>
        .main {
            background-color: #e3f2fd;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2196f3, #1565c0);
            color: white;
            border-radius: 10px;
            height: 3em;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔵 NSFW vs SFW Text Classifier")
    st.write("Analyze text content for safety and moderation.")

    text_input = st.text_area("Enter text here:", height=150)

    if st.button("Analyze"):

        if text_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            clean, label, nsfw_score, sfw_score, original_scores = predict_text(text_input)

            st.subheader("📊 Prediction Result")

            st.write("**Input:**", text_input)
            st.write("**Cleaned:**", clean)

            if label == "NSFW":
                st.error("⚠️ NSFW Content Detected")
            else:
                st.success("✅ Safe Content (SFW)")

            st.write("### Binary Scores")
            st.progress(float(nsfw_score))
            st.write(f"NSFW: {nsfw_score:.4f}")

            st.progress(float(sfw_score))
            st.write(f"SFW: {sfw_score:.4f}")

            st.write("### Original Class Scores")
            st.json({k: round(v, 4) for k, v in original_scores.items()})