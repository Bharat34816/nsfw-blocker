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
from inference.predictor import NSFWPredictor, ThresholdConfig

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8a8d0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
    }
    .result-safe {
        background: linear-gradient(135deg, #0d3b0d, #1a5c1a);
        border: 1px solid #2ecc71;
    }
    .result-nsfw {
        background: linear-gradient(135deg, #3b0d0d, #5c1a1a);
        border: 1px solid #e74c3c;
    }
    .result-review {
        background: linear-gradient(135deg, #3b2f0d, #5c4a1a);
        border: 1px solid #f39c12;
    }

    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #cccccc;
    }

    /* Confidence bar */
    .confidence-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        height: 24px;
        margin: 0.75rem 0;
        overflow: hidden;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 12px;
        transition: width 0.8s ease;
    }
    .bar-safe { background: linear-gradient(90deg, #27ae60, #2ecc71); }
    .bar-nsfw { background: linear-gradient(90deg, #c0392b, #e74c3c); }
    .bar-review { background: linear-gradient(90deg, #e67e22, #f39c12); }

    /* Stat cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Review warning */
    .review-warning {
        background: rgba(243, 156, 18, 0.15);
        border: 1px solid rgba(243, 156, 18, 0.4);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .review-warning strong {
        color: #f39c12;
    }

    /* History table */
    .history-item {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
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

# Force re-initialization if the class changed or user requested it
if "predictor" not in st.session_state or st.sidebar.button("🔄 Reset Models & Cache", use_container_width=True):
    with st.spinner("🚀 Initializing NSFW Predictor..."):
        # Force reload of core modules to pick up code changes
        import importlib
        import inference.predictor
        importlib.reload(inference.predictor)
        
        # Clear transformers cache if relevant
        if "transformers" in sys.modules:
             import transformers
             importlib.reload(transformers)
             
        # Use the freshly reloaded class
        from inference.predictor import NSFWPredictor
        st.session_state.predictor = NSFWPredictor(
            threshold_config=ThresholdConfig(nsfw_threshold=0.85, safe_threshold=0.15)
        )
        st.session_state.history = []
        st.toast("Predictor Hard-Reloaded!", icon="🛡️")

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
        color = "#2ecc71"
    elif pred == "NSFW":
        card_class, bar_class, emoji = "result-nsfw", "bar-nsfw", "🚫"
        color = "#e74c3c"
    else:
        card_class, bar_class, emoji = "result-review", "bar-review", "⚠️"
        color = "#f39c12"

    # Get model info for badge
    model_name = result.details.get("model", "unknown") if result.details else "unknown"
    note = result.details.get("note", "") if result.details else ""

    st.markdown(f"""
<div class="result-card {card_class}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="result-label" style="color: {color}; margin: 0;">
            {emoji} {pred}
        </div>
        <span style="font-size: 0.7rem; background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 10px; color: #888;">
            MODEL: {model_name.upper()}
        </span>
    </div>
    <div class="result-confidence" style="margin-top: 0.5rem;">
        NSFW Score: {score:.1%} &nbsp;|&nbsp; Confidence: {conf:.1%} &nbsp;|&nbsp; ⏱ {elapsed_ms:.0f}ms
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
        "nsfw_score": round(result.nsfw_score, 4),
        "confidence": round(result.confidence, 4),
        "needs_review": result.needs_review,
        "time_ms": round(elapsed_ms, 1),
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
            
            st.image(display_image, caption="Uploaded Image" + (" (Blurred for privacy)" if is_blurred else ""), use_column_width=True)
            
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
                from inference.predictor import PredictionResult
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
