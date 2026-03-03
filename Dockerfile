# ============================================================
# NSFW Content Filter — Dockerfile
# Multi-stage build for FastAPI API + Streamlit Frontend
# ============================================================

# ── Base Stage ──────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── API Stage ───────────────────────────────────────────────
FROM base AS api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Streamlit Stage ─────────────────────────────────────────
FROM base AS streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.address=0.0.0.0"]
