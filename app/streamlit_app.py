"""
NSFW Content Filter — Streamlit Dashboard

Interactive frontend for testing the NSFW filter:
    - Image upload & analysis
    - Video upload & analysis
    - Text analysis
    - Color-coded results with confidence gauges
    - Manual Review warnings
    - Result history in session state
"""

import streamlit as st

# Must be the first streamlit command
st.set_page_config(
    page_title="NSFW Content Filter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from NsfwContentM_main.app import predict_text
from PIL import Image, ImageFilter
from inference.predictor import NSFWPredictor, ThresholdConfig, PredictionResult

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Global & Typography */

    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #000000;
    }

    /* Ultimate Contrast forcing for all labels and text */
    p, span, label, .stMarkdown, .stCaption, .stSlider label, .stToggle label, [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Main Header Enhancement */
    .main-header {
        background: #F8FAFC;
        padding: 3rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 2px solid #E2E8F0;
    }
    .main-header h1 {
        font-size: 3.2rem;
        margin: 0;
        color: #000000 !important;
    }
    .main-header p {
        color: #1E293B !important;
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 600 !important;
    }

    /* Result cards - Ultra Solid */
    .result-card {
        padding: 1.8rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
    }
    .result-safe {
        background-color: #F0FFF4;
        border: 3px solid #059669;
    }
    .result-nsfw {
        background-color: #FFF5F5;
        border: 3px solid #DC2626;
    }
    .result-review {
        background-color: #FFFAF0;
        border: 3px solid #D97706;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 1.2rem;
        color: #000000 !important;
        background: rgba(0,0,0,0.04);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        display: inline-block;
    }

    /* Confidence bar - Thicker & Clearer */
    .confidence-bar-container {
        background: #E2E8F0;
        border-radius: 16px;
        height: 20px;
        margin: 1.2rem 0;
        overflow: hidden;
        border: 2px solid #CBD5E1;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 16px;
    }
    .bar-safe { background-color: #059669; }
    .bar-nsfw { background-color: #DC2626; }
    .bar-review { background-color: #D97706; }

    /* Stat cards - Bold & Prominent */
    .stat-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stat-value {
        font-size: 3rem;
        font-weight: 900;
        color: #000000 !important;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }

    /* Native Streamlit Component Overrides */
    
    /* Buttons - Deep & Impactful */
    div.stButton > button {
        background-color: #4F46E5 !important;
        color: white !important;
        font-weight: 800 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        width: 100%;
        transition: transform 0.2s ease, background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #4338CA !important;
        transform: scale(1.02);
    }

    /* Tabs - Clear Selection */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 12px 12px 0 0;
        padding: 8px 20px;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
    }

    /* Sidebar - Maximum Depth */
    section[data-testid="stSidebar"] {
        background-color: #E2E8F0 !important;
        border-right: 2px solid #CBD5E1;
    }
    .sidebar-section {
        background: #FFFFFF;
        border: 2px solid #CBD5E1;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    /* Force sidebar text to be black - targeted to avoid breaking internals */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* File Uploader - Ultimate Visibility */
    .stFileUploader {
        border: 3px dashed #4F46E5 !important;
        background: #F1F5F9 !important;
        padding: 2.5rem !important;
        border-radius: 20px !important;
    }
    .stFileUploader section {
        background: transparent !important;
    }
    .stFileUploader [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
    }

    /* Inputs & Text Area */
    .stTextArea textarea {
        border: 2px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Metrics Visibility */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        border: 2px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------------------------

def get_predictor():
    """Get the predictor instance, re-initializing if needed."""
    return NSFWPredictor(
        threshold_config=ThresholdConfig(nsfw_threshold=0.85, safe_threshold=0.15)
    )

if "predictor" not in st.session_state:
    with st.spinner("🚀 Initializing NSFW Predictor..."):
        st.session_state.predictor = NSFWPredictor(
            threshold_config=ThresholdConfig(nsfw_threshold=0.85, safe_threshold=0.15)
        )
        st.session_state.history = []
        st.session_state.total_scans = 0
        st.toast("Predictor Initialized!", icon="🛡️")

if "history" not in st.session_state:
    st.session_state.history = []
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def render_result(result, elapsed_ms: float):
    """Render a prediction result with color-coded card."""
    pred = result.prediction
    score = result.nsfw_score
    conf = result.confidence

    if pred == "SAFE":
        card_class, bar_class, emoji = "result-safe", "bar-safe", "✅"
        color = "#2f855a" # Darker green for text readability
    elif pred == "NSFW":
        card_class, bar_class, emoji = "result-nsfw", "bar-nsfw", "🚫"
        color = "#c53030" # Darker red for text readability
    else:
        card_class, bar_class, emoji = "result-review", "bar-review", "⚠️"
        color = "#b7791f" # Darker orange for text readability

    # Get model info for badge
    model_name = result.details.get("model", "unknown") if result.details else "unknown"
    note = result.details.get("note", "") if result.details else ""

    st.markdown(f"""
<div class="result-card {card_class}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="result-label" style="color: {color}; margin: 0;">
            {emoji} {pred}
        </div>
        <span style="font-size: 0.7rem; background: rgba(0,0,0,0.05); padding: 2px 8px; border-radius: 10px; color: #718096; font-weight: 600;">
            MODEL: {model_name.upper()}
        </span>
    </div>
    <div class="result-confidence" style="margin-top: 0.5rem;">
        NSFW Score: <strong>{score:.1%}</strong> &nbsp;|&nbsp; Confidence: <strong>{conf:.1%}</strong> &nbsp;|&nbsp; ⏱ {elapsed_ms:.0f}ms
    </div>
    {f'<div style="font-size: 0.8rem; color: #f39c12; margin-top: 0.2rem;">{note}</div>' if note else ''}
    <div class="confidence-bar-container">
        <div class="confidence-bar {bar_class}" style="width: {score * 100}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    if result.needs_review:
        st.markdown("""
<div class="review-warning">
    <strong>⚠️ Manual Review Recommended</strong><br>
    This content falls in the uncertain zone. The model is not confident
    enough to auto-classify. Please have a human moderator review this content.
</div>
""", unsafe_allow_html=True)

    if result.details:
        with st.expander("📊 Detailed Results"):
            st.json(result.details)


def add_to_history(modality: str, result, elapsed_ms: float):
    """Add a result to session history."""
    st.session_state.history.insert(0, {
        "modality": modality,
        "prediction": result.prediction,
        "nsfw_score": float(result.nsfw_score),
        "confidence": float(result.confidence),
        "needs_review": result.needs_review,
        "time_ms": float(elapsed_ms),
    })
    st.session_state.total_scans += 1
    # Keep last 50 entries
    st.session_state.history = st.session_state.history[:50]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption("v1.2.0-Aggressive-Text-Override")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 👁️ Privacy Settings")
    reveal_content = st.toggle(
        "Reveal NSFW content",
        value=False,
        help="If disabled, NSFW and Review-flagged images will be blurred.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    nsfw_thresh = st.slider(
        "NSFW Images/Videos Threshold",
        min_value=0.50, max_value=0.99, value=0.85, step=0.01,
        help="Confidence above this → auto-block as NSFW",
    )
    safe_thresh = st.slider(
        "Safe Images/Videos Threshold",
        min_value=0.01, max_value=0.50, value=0.15, step=0.01,
        help="Confidence below this → auto-approve as Safe",
    )

    # Update thresholds
    st.session_state.predictor.threshold = ThresholdConfig(
        nsfw_threshold=nsfw_thresh,
        safe_threshold=safe_thresh,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Scans", st.session_state.total_scans)
    with col2:
        flagged = sum(
            1 for h in st.session_state.history
            if h["prediction"] in ("NSFW", "REVIEW")
        )
        st.metric("Flagged", flagged)

    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_scans = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**NSFW Content Filter** v1.0  \n"
        "Custom EfficientNet-B0 + CNN-BiLSTM  \n"
        "Built with ❤️ using PyTorch & Streamlit"
    )


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🛡️ NSFW Content Filter</h1>
    <p>AI-powered multi-modal content moderation — Images • Videos • Text</p>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab_image, tab_video, tab_text, tab_history = st.tabs([
    "🖼️ Image Upload",
    "🎥 Video Upload",
    "📝 Text Analysis",
    "📋 History",
])


# === IMAGE TAB ===
with tab_image:
    st.markdown("#### Upload an image to check for NSFW content")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="image_upload",
        help="Supported formats: JPG, PNG, WebP, BMP",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        
        # We need to predict FIRST to decide whether to blur
        with st.spinner("🔍 Analyzing image..."):
            start = time.time()
            result = st.session_state.predictor.predict_image(image)
            elapsed = (time.time() - start) * 1000

        col1, col2 = st.columns([1, 1])

        with col1:
            display_image = image
            is_blurred = False
            
            # Apply blur if NSFW or REVIEW and not revealed
            if (result.prediction in ("NSFW", "REVIEW")) and not reveal_content:
                display_image = image.filter(ImageFilter.GaussianBlur(radius=30))
                is_blurred = True
            
            st.image(display_image, caption="Uploaded Image" + (" (Blurred for privacy)" if is_blurred else ""), use_container_width=True)
            
            if is_blurred:
                st.warning("⚠️ This image has been blurred. Use the 'Reveal NSFW content' toggle in the sidebar to view.")

        with col2:
            render_result(result, elapsed)
            add_to_history("🖼️ Image", result, elapsed)


# === VIDEO TAB ===
with tab_video:
    st.markdown("#### Upload a video to check for NSFW content")
    st.info(
        "📌 Videos are analyzed by extracting keyframes via scene-change detection. "
        "The video is flagged NSFW if **any** keyframe exceeds the threshold."
    )

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_upload",
        help="Supported formats: MP4, AVI, MOV, MKV, WebM",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🔍 Analyze Video", key="analyze_video", use_container_width=True):
            # Save to temp file for OpenCV
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_video.name).suffix
            ) as tmp:
                tmp.write(uploaded_video.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("🎬 Extracting keyframes and analyzing..."):
                    start = time.time()
                    result = st.session_state.predictor.predict_video(tmp_path)
                    elapsed = (time.time() - start) * 1000

                render_result(result, elapsed)
                add_to_history("🎥 Video", result, elapsed)

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# === TEXT TAB ===
with tab_text:
    st.markdown("#### Enter or paste text to check for NSFW content")

    text_input = st.text_area(
        "Text to analyze",
        height=200,
        placeholder="Paste or type text here...",
        key="text_input",
    )

    if st.button("🔍 Analyze Text", key="analyze_text", use_container_width=True):
        if text_input and text_input.strip():
            with st.spinner("📝 Analyzing text..."):
                start = time.time()
                # Use the new predict_text function
                clean_text, label, nsfw_score, sfw_score, label_scores = predict_text(text_input.strip())
                elapsed = (time.time() - start) * 1000

                # Convert to PredictionResult format for UI rendering
                # Use the PredictionResult format for UI rendering
                result = PredictionResult(
                    prediction=label, # "NSFW" or "SFW" (dashboard handles color coding)
                    confidence=max(nsfw_score, sfw_score),
                    nsfw_score=nsfw_score,
                    needs_review=False,
                    modality="text",
                    details={
                        "model": "NSFW-Content-M",
                        "cleaned_text": clean_text,
                        "label_scores": label_scores
                    }
                )

            render_result(result, elapsed)
            add_to_history("📝 Text", result, elapsed)
        else:
            st.warning("Please enter some text to analyze.")


# === HISTORY TAB ===
with tab_history:
    st.markdown("#### Scan History")

    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            pred = entry["prediction"]
            if pred == "SAFE":
                icon, color = "✅", "#2ecc71"
            elif pred == "NSFW":
                icon, color = "🚫", "#e74c3c"
            else:
                icon, color = "⚠️", "#f39c12"

            with st.container():
                cols = st.columns([1, 2, 2, 2, 1])
                cols[0].markdown(f"**{entry['modality']}**")
                cols[1].markdown(
                    f"<span style='color:{color};font-weight:600;'>"
                    f"{icon} {pred}</span>",
                    unsafe_allow_html=True,
                )
                cols[2].markdown(f"NSFW: {entry['nsfw_score']:.1%}")
                cols[3].markdown(f"Conf: {entry['confidence']:.1%}")
                cols[4].markdown(f"{entry['time_ms']:.0f}ms")
            st.divider()
    else:
        st.info("No scans yet. Upload an image, video, or enter text to get started!")
