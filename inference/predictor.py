"""
NSFW Content Filter — Unified Multi-Modal Predictor

Handles image, video, and text inference with:
    - Pre-trained HuggingFace model for images (Falconsai/nsfw_image_detection)
    - Custom model fallback when trained checkpoints exist
    - Keyword + pattern-based text detection
    - High-confidence thresholding (zero false positives goal)
    - "Manual Review" flag for borderline cases
    - Lazy model loading with caching
"""

from __future__ import annotations

# Imports (Heavy libraries moved inside methods for lazy loading)
import logging
import re
import pickle
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

def round_score(val: Any) -> float:
    """Safe rounding for lint and types."""
    try:
        return float(round(float(val), 4))
    except (ValueError, TypeError):
        return 0.0

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Model imports moved inside methods to avoid loading torch/cv2 at startup
# from models.efficientnet_model import EfficientNetB0
# from models.text_model import TextCNN_BiLSTM, Vocabulary
# from training.video_sampler import VideoFrameSampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# DEVICE will be determined lazily
_DEVICE = None

def get_device():
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


# ===========================================================================
# Thresholding Configuration
# ===========================================================================

class ThresholdConfig:
    """
    High-confidence thresholding for zero false-positive operation.

    Decision Zones:
        NSFW confidence >= nsfw_threshold  -> NSFW (block)
        NSFW confidence <= safe_threshold  -> SAFE (allow)
        In between                         -> MANUAL REVIEW (flag)
    """

    def __init__(
        self,
        nsfw_threshold: float = 0.85,
        safe_threshold: float = 0.15,
    ):
        self.nsfw_threshold = nsfw_threshold
        self.safe_threshold = safe_threshold


# ===========================================================================
# Result Objects
# ===========================================================================

class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class PredictionResult:
    """Structured prediction result."""
    prediction: str           # "SAFE", "NSFW", or "REVIEW"
    confidence: float         # Probability of the predicted class
    nsfw_score: float         # Raw NSFW probability (0-1)
    needs_review: bool        # True if in the manual-review zone
    modality: str             # "image", "video", or "text"
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Keyword-Based Text Detector
# ===========================================================================

class KeywordTextDetector:
    """
    Rule-based NSFW text detector using keyword matching,
    pattern detection, and context analysis.

    Works without any training data or pre-trained models.
    Uses weighted scoring: explicit terms score higher,
    contextual/suggestive terms score lower.
    """

    # NOTE: These are sanitized/abbreviated patterns for the codebase.
    # The detector uses pattern matching rather than exact word lists.
    EXPLICIT_PATTERNS = [
        r'\bp[o0]rn\w*\b', r'\bnud[ei]\w*\b', r'\bnak[ei]d\b',
        r'\bxx+\b', r'\bs[e3]x\w*\b', r'\bh[e3]ntai\b',
        r'\b(f+u+c+k+|f+k+)\w*\b', r'\bsh[i1!]t+\b',
        r'\bass\b', r'\bassh[o0]le\b', r'\bd[i1]ck\b', r'\bbastard\b',
        r'\bbitch\b', r'\bprick\b', r'\bpussy\b', r'\bdickhead\b',
        r'\bcum\b', r'\bslut\b', r'\bwh[o0]re\b', r'\bmotherfucker\b',
        r'\bb[o0][o0]b\w*\b', r'\bt[i1]t+[s5]\b',
        r'\borgas[m]?\w*\b', r'\bmast[ue]rbat\w*\b',
        r'\beroti[ck]\w*\b', r'\bfeti[s]h\w*\b',
        r'\bgenitals?\b', r'\bp[e3]n[i1][s5]\b',
        r'\bvag[i1]na\w*\b', r'\banal\b',
        r'\bstrip\s*club\b', r'\bescort\w*\b',
        r'\bprostitut\w*\b', r'\bbordello\b',
        r'\badult\s+(content|film|video|site|website|material)\b',
        r'\bnsfw\b', r'\bexplicit\b',
        r'\bobscen\w*\b', r'\bvulgar\w*\b',
        r'\bindecen\w*\b', r'\blewd\w*\b',
        r'\bperverted?\b', r'\bkinky\b',
        r'\blingerie\b', r'\bsedu[ck]ti\w*\b',
    ]

    SUGGESTIVE_PATTERNS = [
        r'\bhot\s+(girl|guy|babe|chick|woman|man)\b',
        r'\bsexy\b', r'\bsensual\b',
        r'\bintimate\b', r'\bdesire\b',
        r'\bpassion\w*\b', r'\blust\w*\b',
        r'\btantali[sz]\w*\b', r'\btempt\w*\b',
        r'\brisqu[eé]\b', r'\bracy\b',
        r'\bprovocative\b', r'\bsuggestive\b',
        r'\bscantily\b', r'\bundress\w*\b',
        r'\bbikini\b', r'\bthong\b',
        r'\bcleavage\b', r'\btopless\b',
    ]

    # Safe context words that reduce the score (medical, educational, etc.)
    SAFE_CONTEXT = [
        r'\bmedical\b', r'\bhealth\b', r'\beducat\w*\b',
        r'\bscien\w*\b', r'\bresearch\b', r'\bacademic\b',
        r'\bart\s*(history|museum|gallery|class)\b',
        r'\banatomy\b', r'\bbiology\b', r'\bpediatri\w*\b',
        r'\bbreastfeed\w*\b', r'\bnursing\b',
        r'\bsculpture\b', r'\brenaissance\b',
    ]

    def __init__(self):
        # re.compile is lightweight enough for init
        self._explicit_re = [re.compile(p, re.IGNORECASE) for p in self.EXPLICIT_PATTERNS]
        self._suggestive_re = [re.compile(p, re.IGNORECASE) for p in self.SUGGESTIVE_PATTERNS]
        self._safe_re = [re.compile(p, re.IGNORECASE) for p in self.SAFE_CONTEXT]

    def score(self, text: str) -> float:
        """
        Score text from 0.0 (safe) to 1.0 (nsfw).

        Scoring:
            - Each explicit match: +0.15
            - Each suggestive match: +0.06
            - Safe context: reduces total by 40%
            - Multiple matches compound (more matches = higher confidence)
        """
        if not text or not text.strip():
            return 0.0

        explicit_hits = sum(1 for r in self._explicit_re if r.search(text))
        suggestive_hits = sum(1 for r in self._suggestive_re if r.search(text))
        safe_hits = sum(1 for r in self._safe_re if r.search(text))

        # Scoring: 1 explicit = 0.85 (NSFW threshold), 2 explicit = 1.0 (NSFW)
        # 1 suggestive = 0.20 (REVIEW zone), 5 suggestive = 1.0 (NSFW)
        raw_score = (explicit_hits * 0.85) + (suggestive_hits * 0.20)

        # Cap at 1.0
        raw_score = min(raw_score, 1.0)

        # Reduce score if safe context is present
        if safe_hits > 0 and raw_score > 0:
            reduction = min(0.4, safe_hits * 0.15)
            raw_score = max(0.0, raw_score - reduction)

        return round_score(raw_score)


# ===========================================================================
# Pre-trained Image Classifier (HuggingFace)
# ===========================================================================

class PretrainedImageClassifier:
    """
    Uses HuggingFace transformers pipeline for NSFW image classification.
    Downloads model on first use and caches it locally.
    """

    MODEL_NAME = "Falconsai/nsfw_image_detection"

    def __init__(self):
        self._pipeline = None

    def _load(self):
        """Lazy-load the HuggingFace pipeline."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
            logger.info("Loading pre-trained NSFW image classifier: %s", self.MODEL_NAME)
            self._pipeline = pipeline(
                "image-classification",
                model=self.MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Pre-trained image classifier loaded successfully")
        except Exception as e:
            logger.error("Failed to load pre-trained classifier: %s", e)
            self._pipeline = None

    def predict(self, image: "Image.Image") -> float:
        """
        Predict NSFW score for an image.

        Returns:
            Float 0.0 (safe) to 1.0 (nsfw).
        """
        self._load()
        if self._pipeline is None:
            return 0.5  # Fallback if model failed to load

        try:
            results = self._pipeline(image)
            # Results: [{"label": "nsfw", "score": 0.95}, {"label": "normal", "score": 0.05}]
            nsfw_score = 0.0
            for r in results:
                label = r.get("label", "").lower()
                if label in ("nsfw", "porn", "sexy", "hentai", "explicit"):
                    nsfw_score = r.get("score", 0.0)
                    break
            return nsfw_score
        except Exception as e:
            logger.error("Image prediction failed: %s", e)
            return 0.5


# ===========================================================================
# Unified Predictor
# ===========================================================================

class NSFWPredictor:
    """
    Unified multi-modal NSFW predictor.

    Strategy:
        - Images: Uses pre-trained HuggingFace model (Falconsai/nsfw_image_detection)
                  OR custom EfficientNet-B0 if trained checkpoint exists
        - Videos: Extracts keyframes, classifies each via image model
        - Text:   Keyword-based pattern matching + optional custom model
    """

    CLASS_NAMES = {0: "SAFE", 1: "NSFW"}

    def __init__(
        self,
        image_model_path: Optional[str] = None,
        text_model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        use_pretrained: bool = True,
    ):
        project_root = Path(__file__).resolve().parent.parent
        self.image_model_path = image_model_path or str(project_root / "models/checkpoints/image_model.pth")
        self.text_model_path = text_model_path or str(project_root / "models/checkpoints/text_model.pth")
        self.vocab_path = vocab_path or str(project_root / "models/checkpoints/vocabulary.json")
        self.threshold = threshold_config or ThresholdConfig()
        self.use_pretrained = use_pretrained

        # Models (lazy-loaded)
        self._custom_image_model = None
        self._custom_text_model = None
        self._vocab = None
        self._pretrained_classifier: Optional[PretrainedImageClassifier] = None
        self._text_detector = KeywordTextDetector()
        self._video_sampler = None  # Lazy loaded

        # Check for custom checkpoints
        self._has_custom_image = Path(self.image_model_path).exists()
        self._has_custom_text = Path(self.text_model_path).exists()

        self.external_keras_path = str(project_root / "models/external/text_model.keras")
        self.external_tokenizer_path = str(project_root / "models/external/tokenizer.pickle")
        if not Path(self.external_tokenizer_path).exists():
            self.external_tokenizer_path = str(project_root / "models/external/tokenizer.pkl")
        
        self._has_external_text = (
            Path(self.external_keras_path).exists() and 
            Path(self.external_tokenizer_path).exists()
        )
        self._external_model = None
        self._external_tokenizer = None

        # Image transform (Lazy imported when needed)
        self._image_transform = None

        logger.info(
            "Predictor initialized — image(custom:%s, pretrained:%s), text(custom:%s, ext:%s)",
            self._has_custom_image, self.use_pretrained, self._has_custom_text, self._has_external_text,
        )

    def _get_transform(self):
        if self._image_transform is None:
            from torchvision import transforms
            self._image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        return self._image_transform

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _get_pretrained_classifier(self) -> PretrainedImageClassifier:
        """Get or create the pre-trained classifier."""
        if self._pretrained_classifier is None:
            self._pretrained_classifier = PretrainedImageClassifier()
        return self._pretrained_classifier

    def _load_custom_image_model(self) -> Optional[EfficientNetB0]:
        """Load custom image model if checkpoint exists."""
        if self._custom_image_model is not None:
            return self._custom_image_model

        if not self._has_custom_image:
            return None

        import torch
        from models.efficientnet_model import EfficientNetB0
        model = EfficientNetB0(num_classes=2)
        checkpoint = torch.load(self.image_model_path, map_location=get_device(), weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume direct state dict
            model.load_state_dict(checkpoint)
        model = model.to(get_device())
        model.eval()
        self._custom_image_model = model
        logger.info("Custom image model loaded from %s", self.image_model_path)
        return model

    def _load_custom_text_model(self) -> Optional[TextCNN_BiLSTM]:
        """Load custom text model if checkpoint exists."""
        if self._custom_text_model is not None:
            return self._custom_text_model

        if not self._has_custom_text:
            return None

        vocab_path = Path(self.vocab_path)
        if vocab_path.exists():
            from models.text_model import Vocabulary
            self._vocab = Vocabulary.load(str(vocab_path))
        else:
            return None

        import torch
        from models.text_model import TextCNN_BiLSTM
        model = TextCNN_BiLSTM(vocab_size=len(self._vocab), num_classes=2)
        checkpoint = torch.load(self.text_model_path, map_location=get_device(), weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume direct state dict
            model.load_state_dict(checkpoint)
        model = model.to(get_device())
        model.eval()
        self._custom_text_model = model
        logger.info("Custom text model loaded from %s", self.text_model_path)
        return model

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _apply_threshold(
        self, nsfw_score: float, modality: Modality, model_info: str = "custom"
    ) -> PredictionResult:
        """Apply three-zone thresholding."""
        # Special case: 0.5 is the fallback score when no model is available
        # or it's a perfect tie. Should ALWAYS be REVIEW.
        if abs(nsfw_score - 0.5) < 1e-4:
            return PredictionResult(
                prediction="REVIEW",
                confidence=0.5,
                nsfw_score=0.5,
                needs_review=True,
                modality=modality.value,
                details={"model": model_info, "note": "Prediction fallback (no model or tie)"}
            )

        if nsfw_score >= self.threshold.nsfw_threshold:
            return PredictionResult(
                prediction="NSFW",
                confidence=nsfw_score,
                nsfw_score=nsfw_score,
                needs_review=False,
                modality=modality.value,
                details={"model": model_info}
            )
        elif nsfw_score <= self.threshold.safe_threshold:
            return PredictionResult(
                prediction="SAFE",
                confidence=1.0 - nsfw_score,
                nsfw_score=nsfw_score,
                needs_review=False,
                modality=modality.value,
                details={"model": model_info}
            )
        else:
            return PredictionResult(
                prediction="REVIEW",
                confidence=max(nsfw_score, 1.0 - nsfw_score),
                nsfw_score=nsfw_score,
                needs_review=True,
                modality=modality.value,
                details={"model": model_info}
            )

    # ------------------------------------------------------------------
    # Image Prediction
    # ------------------------------------------------------------------

    def predict_image(self, image: Union[str, Path, "Image.Image"]) -> PredictionResult:
        """
        Predict NSFW status for a single image.

        Uses pre-trained HuggingFace model by default, or custom
        EfficientNet-B0 if a trained checkpoint exists.
        """
        from PIL import Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Strategy: custom model if available, else pre-trained
        # FALLBACK: If custom model results in a tie (0.5), use pretrained.
        nsfw_score = 0.5
        model_info = "untrained"

        if self._has_custom_image:
            nsfw_score = self._predict_image_custom(image)
            model_info = "custom-torch"
            
            # If custom model returns a perfect tie (likely untrained/reset), fallback to pretrained
            if abs(nsfw_score - 0.5) < 1e-4 and self.use_pretrained:
                logger.info("Custom model returned tie (0.5), falling back to pre-trained.")
                nsfw_score = self._predict_image_pretrained(image)
                model_info = "pretrained-hf (fallback)"
        
        elif self.use_pretrained:
            nsfw_score = self._predict_image_pretrained(image)
            model_info = "pretrained-hf"

        return self._apply_threshold(nsfw_score, Modality.IMAGE, model_info)

    def _predict_image_pretrained(self, image: Image.Image) -> float:
        """Classify using pre-trained HuggingFace model."""
        classifier = self._get_pretrained_classifier()
        return classifier.predict(image)

    def _predict_image_custom(self, image) -> float:
        """Classify using custom EfficientNet-B0."""
        import torch
        import torch.nn.functional as F
        model = self._load_custom_image_model()
        if model is None:
            return 0.5

        with torch.no_grad():
            tensor = self._get_transform()(image).unsqueeze(0).to(get_device())
            proba = F.softmax(model(tensor), dim=1)
            return proba[0, 1].item()

    # ------------------------------------------------------------------
    # Video Prediction
    # ------------------------------------------------------------------

    def predict_video(self, video_path: str) -> PredictionResult:
        """
        Predict NSFW status for a video.
        Flags as NSFW if ANY keyframe exceeds threshold.
        """
        if self._video_sampler is None:
            from training.video_sampler import VideoFrameSampler
            self._video_sampler = VideoFrameSampler()
        keyframes = self._video_sampler.extract_keyframes(video_path)

        if not keyframes:
            return PredictionResult(
                prediction="REVIEW",
                confidence=0.0,
                nsfw_score=0.0,
                needs_review=True,
                modality=Modality.VIDEO.value,
                details={"error": "Could not extract keyframes"},
            )

        frame_results = []
        max_nsfw_score = 0.0
        model_info = "untrained"
        if self._has_custom_image:
            model_info = "custom-torch"
        elif self.use_pretrained:
            model_info = "pretrained-hf"

        for i, frame_img in enumerate(keyframes):
            result = self.predict_image(frame_img)
            frame_results.append({
                "frame_index": i,
                "nsfw_score": result.nsfw_score,
                "prediction": result.prediction,
            })
            max_nsfw_score = max(max_nsfw_score, result.nsfw_score)

        video_result = self._apply_threshold(max_nsfw_score, Modality.VIDEO, model_info)
        video_result.details = {
            "model": model_info,
            "total_keyframes": len(keyframes),
            "max_nsfw_score": max_nsfw_score,
            "frame_results": frame_results,
        }
        return video_result

    # ------------------------------------------------------------------
    # Text Prediction
    # ------------------------------------------------------------------

    def predict_text(self, text: str) -> PredictionResult:
        """
        Predict NSFW status for text.

        Priority order:
        1. External Keras model (if files exist in models/external/)
        2. Custom PyTorch model (if trained checkpoint exists)
        3. Keyword-based fallback
        """
        # Keyword-based score (useful for immediate results or combining)
        keyword_score = self._text_detector.score(text)

        # Strategy: external > custom-torch > keywords
        model_info = "keywords"
        nsfw_score = keyword_score
        model_score = None

        if self._has_external_text:
            model_score = self._predict_text_external(text)
            
            # --- Aggressive Conflict Resolution ---
            # If Keywords find clear profanity (>=0.85), we always take the Max.
            # This prevents a 0.0 model score from dragging down a clear NSFW match.
            if keyword_score >= 0.85:
                nsfw_score = max(model_score, keyword_score)
                model_info = "external-keras (plus-keywords)" if model_score > 0.1 else "external-keras (keyword-override)"
            else:
                # Normal case: Combine with keywords: 70% model, 30% keywords
                nsfw_score = 0.7 * model_score + 0.3 * keyword_score
                model_info = "external-keras"
        elif self._has_custom_text:
            model_score = self._predict_text_custom(text)
            # Weighted combination: 60% model, 40% keywords
            nsfw_score = 0.6 * model_score + 0.4 * keyword_score
            model_info = "custom-torch"

        result = self._apply_threshold(nsfw_score, Modality.TEXT, model_info)
        result.details = {
            "model": model_info,
            "keyword_score": keyword_score,
            "model_score": model_score,
            "combined_score": nsfw_score,
        }
        return result

    def _predict_text_custom(self, text: str) -> float:
        """Classify using custom CNN-BiLSTM."""
        import torch
        import torch.nn.functional as F
        model = self._load_custom_text_model()
        if model is None:
            return 0.0

        with torch.no_grad():
            encoded = self._vocab.encode(text, max_length=256)
            tensor = torch.tensor([encoded], dtype=torch.long).to(get_device())
            proba = F.softmax(model(tensor), dim=1)
            return proba[0, 1].item()

    def _load_external_text_model(self):
        """Lazy load the external Keras model and tokenizer."""
        if self._external_model is not None:
            return self._external_model, self._external_tokenizer

        if not self._has_external_text:
            return None, None

        import tensorflow as tf
        try:
            self._external_model = tf.keras.models.load_model(self.external_keras_path)
            with open(self.external_tokenizer_path, 'rb') as f:
                self._external_tokenizer = pickle.load(f)
            logger.info("External Keras model loaded from %s", self.external_keras_path)
            return self._external_model, self._external_tokenizer
        except Exception as e:
            logger.error("Failed to load external text model: %s", e)
            return None, None

    def _predict_text_external(self, text: str) -> float:
        """Classify using external Keras model."""
        model, tokenizer = self._load_external_text_model()
        if model is None or tokenizer is None:
            return 0.0

        try:
            import numpy as np
            # Common Keras preprocessing (Tokens + Padding)
            # Optimized preprocessing for Bi-LSTM models
            # Maxlen 50 matches the suspected model architecture observed in metadata
            sequences = tokenizer.texts_to_sequences([text.lower()])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded = pad_sequences(sequences, maxlen=50, padding='pre', truncating='pre')
            
            prediction = model.predict(padded, verbose=0)
            
            # Handle both [0.8] and [[0.1, 0.9]] outputs
            if prediction.shape[-1] == 1:
                nsfw_prob = float(prediction[0][0])
            else:
                nsfw_prob = float(prediction[0][1])  # Assume index 1 is NSFW
            
            return nsfw_prob
        except Exception as e:
            logger.error("External prediction failed: %s", e)
            return 0.5  # Neutral fallback

    # ------------------------------------------------------------------
    # Batch Prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        images: Optional[List[Union[str, Image.Image]]] = None,
        texts: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        """Predict NSFW status for a batch of mixed inputs."""
        results = []
        if images:
            for img in images:
                results.append(self.predict_image(img))
        if texts:
            for text in texts:
                results.append(self.predict_text(text))
        return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = NSFWPredictor()

    dummy_image = Image.new("RGB", (224, 224), color=(100, 150, 200))
    result = predictor.predict_image(dummy_image)
    print(f"\nImage prediction: {result.to_dict()}")

    text_result = predictor.predict_text("This is a perfectly normal sentence about coding.")
    print(f"\nSafe text: {text_result.to_dict()}")

    nsfw_text = predictor.predict_text("explicit adult content nsfw porn")
    print(f"\nNSFW text: {nsfw_text.to_dict()}")

    print("\nPredictor initialized successfully!")
