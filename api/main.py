"""
NSFW Content Filter — FastAPI Backend

REST API for multi-modal NSFW detection:
    POST /predict/image  — Image file upload
    POST /predict/video  — Video file upload
    POST /predict/text   — JSON text body
    POST /predict/batch  — Multiple files + text
    GET  /health         — Health check
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import NSFWPredictor, PredictionResult, ThresholdConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Allowed file types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/avi", "video/mpeg", "video/quicktime",
    "video/x-msvideo", "video/webm", "video/x-matroska",
}
MAX_FILE_SIZE_MB = 100


# ===========================================================================
# Pydantic Models
# ===========================================================================

class TextRequest(BaseModel):
    """Request body for text prediction."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class PredictionResponse(BaseModel):
    """Standard prediction response."""
    prediction: str
    confidence: float
    nsfw_score: float
    needs_review: bool
    modality: str
    details: Optional[dict] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str


class BatchResponse(BaseModel):
    """Batch prediction response."""
    results: List[PredictionResponse]
    total_items: int
    total_nsfw: int
    total_safe: int
    total_review: int


# ===========================================================================
# Application
# ===========================================================================

app = FastAPI(
    title="NSFW Content Filter API",
    description=(
        "Real-time multi-modal NSFW detection API. "
        "Supports image, video, and text classification with "
        "high-confidence thresholding and manual review flagging."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global predictor (initialized on startup) ---
predictor: Optional[NSFWPredictor] = None


# ===========================================================================
# Startup / Shutdown
# ===========================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    predictor = NSFWPredictor(
        threshold_config=ThresholdConfig(
            nsfw_threshold=0.85,
            safe_threshold=0.15,
        )
    )
    logger.info("NSFW Predictor initialized")


# ===========================================================================
# Helper Functions
# ===========================================================================

async def save_upload_to_temp(upload: UploadFile) -> str:
    """Save uploaded file to a temporary location. Returns temp path."""
    suffix = Path(upload.filename or "file").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB",
            )
        tmp.write(content)
        return tmp.name


def result_to_response(result: PredictionResult, elapsed_ms: float) -> PredictionResponse:
    """Convert PredictionResult to PredictionResponse."""
    return PredictionResponse(
        prediction=result.prediction,
        confidence=round(result.confidence, 4),
        nsfw_score=round(result.nsfw_score, 4),
        needs_review=result.needs_review,
        modality=result.modality,
        details=result.details,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    import torch
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )


@app.post("/predict/image", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image as SAFE, NSFW, or REVIEW.

    Accepts: JPEG, PNG, WebP, BMP, GIF
    Returns: Prediction with confidence score and review flag.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {ALLOWED_IMAGE_TYPES}",
        )

    tmp_path = None
    try:
        tmp_path = await save_upload_to_temp(file)
        start = time.time()
        result = predictor.predict_image(tmp_path)
        elapsed = (time.time() - start) * 1000
        return result_to_response(result, elapsed)

    except Exception as e:
        logger.exception("Image prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/predict/video", response_model=PredictionResponse, tags=["Prediction"])
async def predict_video(file: UploadFile = File(...)):
    """
    Classify an uploaded video as SAFE, NSFW, or REVIEW.

    Extracts keyframes via scene-change detection and classifies each.
    Video is NSFW if ANY keyframe exceeds the NSFW threshold.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {ALLOWED_VIDEO_TYPES}",
        )

    tmp_path = None
    try:
        tmp_path = await save_upload_to_temp(file)
        start = time.time()
        result = predictor.predict_video(tmp_path)
        elapsed = (time.time() - start) * 1000
        return result_to_response(result, elapsed)

    except Exception as e:
        logger.exception("Video prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/predict/text", response_model=PredictionResponse, tags=["Prediction"])
async def predict_text(request: TextRequest):
    """
    Classify text as SAFE, NSFW, or REVIEW.

    Maximum text length: 10,000 characters.
    """
    try:
        start = time.time()
        result = predictor.predict_text(request.text)
        elapsed = (time.time() - start) * 1000
        return result_to_response(result, elapsed)

    except Exception as e:
        logger.exception("Text prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[str] = Form(None),
):
    """
    Batch prediction for multiple files and/or text inputs.

    - Files: Multiple image/video uploads
    - Texts: JSON array of text strings (as form field)
    """
    import json

    results: List[PredictionResponse] = []
    temp_paths: List[str] = []

    try:
        # Process uploaded files
        if files:
            for file in files:
                if file.content_type in ALLOWED_IMAGE_TYPES:
                    tmp = await save_upload_to_temp(file)
                    temp_paths.append(tmp)
                    start = time.time()
                    result = predictor.predict_image(tmp)
                    elapsed = (time.time() - start) * 1000
                    results.append(result_to_response(result, elapsed))

                elif file.content_type in ALLOWED_VIDEO_TYPES:
                    tmp = await save_upload_to_temp(file)
                    temp_paths.append(tmp)
                    start = time.time()
                    result = predictor.predict_video(tmp)
                    elapsed = (time.time() - start) * 1000
                    results.append(result_to_response(result, elapsed))

        # Process text inputs
        if texts:
            try:
                text_list = json.loads(texts)
                if isinstance(text_list, str):
                    text_list = [text_list]
            except json.JSONDecodeError:
                text_list = [texts]

            for text in text_list:
                start = time.time()
                result = predictor.predict_text(str(text))
                elapsed = (time.time() - start) * 1000
                results.append(result_to_response(result, elapsed))

        # Aggregate
        total_nsfw = sum(1 for r in results if r.prediction == "NSFW")
        total_safe = sum(1 for r in results if r.prediction == "SAFE")
        total_review = sum(1 for r in results if r.prediction == "REVIEW")

        return BatchResponse(
            results=results,
            total_items=len(results),
            total_nsfw=total_nsfw,
            total_safe=total_safe,
            total_review=total_review,
        )

    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for tmp in temp_paths:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
