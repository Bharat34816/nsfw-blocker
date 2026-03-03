# 🛡️ NSFW Content Filter

**AI-powered multi-modal content moderation** — Images • Videos • Text

A complete end-to-end NSFW detection system built from scratch with custom deep learning models (no pre-trained weights), real-time inference, and a production-ready deployment stack.

---

## ✨ Features

- **🖼️ Image Classification** — Custom EfficientNet-B0 (~5.3M params) trained from scratch
- **🎥 Video Analysis** — Scene-change detection + keyframe-level classification
- **📝 Text Moderation** — Hybrid 1D-CNN + Bi-LSTM with attention mechanism
- **🎯 Zero False Positives** — Three-zone confidence thresholding (SAFE / REVIEW / NSFW)
- **⚠️ Manual Review Flag** — Borderline cases flagged for human review
- **🚀 FastAPI Backend** — REST API with batch prediction support
- **🎨 Streamlit Dashboard** — Modern dark-theme UI with real-time feedback
- **🐳 Docker Ready** — Multi-stage builds for easy deployment

---

## 📁 Project Structure

```
Blocker/
├── data_acquisition/
│   ├── scraper.py            # Web scraping strategy (Reddit, Flickr APIs)
│   └── data_cleaner.py       # Validation, dedup, balancing, train/val/test splits
│
├── models/
│   ├── efficientnet_model.py # Custom EfficientNet-B0 from scratch (PyTorch)
│   └── text_model.py         # 1D-CNN + Bi-LSTM text classifier
│
├── training/
│   ├── train_image.py        # Image model training (mixed-precision, cosine LR)
│   ├── train_text.py         # Text model training (vocab building, Adam)
│   └── video_sampler.py      # Scene-change detection + uniform sampling
│
├── inference/
│   └── predictor.py          # Unified multi-modal predictor + thresholding
│
├── api/
│   └── main.py               # FastAPI REST API (image/video/text/batch)
│
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
│
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
│
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # API + Frontend orchestration
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### 3. Run the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧠 Model Architectures

### EfficientNet-B0 (Image/Video)
| Component | Details |
|-----------|---------|
| Architecture | MBConv blocks + Squeeze-and-Excitation |
| Parameters | ~5.3M (all trainable) |
| Initialization | Kaiming (He) normal — no pre-trained weights |
| Input | 224×224×3 RGB images |
| Output | 2-class softmax (Safe/NSFW) |
| Training | AdamW, cosine LR, mixed-precision, early stopping |

### Text CNN + Bi-LSTM (Text)
| Component | Details |
|-----------|---------|
| Architecture | Multi-scale Conv1D → Bi-LSTM → Attention → FC |
| Embedding | Learned from scratch (128-dim) |
| Vocabulary | Built from training data (max 30K tokens) |
| Input | Tokenized text (max 256 tokens) |
| Output | 2-class softmax (Safe/NSFW) |

---

## 🎯 Confidence Thresholding

To achieve **zero false positives**, the system uses three-zone classification:

| Zone | NSFW Score | Decision | Action |
|------|-----------|----------|--------|
| 🟢 Safe | ≤ 0.15 | **SAFE** | Auto-allow |
| 🟡 Review | 0.15 — 0.85 | **REVIEW** | Flag for human moderation |
| 🔴 NSFW | ≥ 0.85 | **NSFW** | Auto-block |

Thresholds are adjustable in the Streamlit sidebar.

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict/image` | POST | Classify uploaded image |
| `/predict/video` | POST | Classify uploaded video |
| `/predict/text` | POST | Classify text (JSON body) |
| `/predict/batch` | POST | Batch classification |

Full interactive docs at: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

```bash
docker-compose up --build -d
```
- API: http://localhost:8000
- Frontend: http://localhost:8501

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment instructions including Streamlit Cloud.

---

## 🏋️ Training

```bash
# 1. Acquire data
python -m data_acquisition.scraper

# 2. Clean & split
python -m data_acquisition.data_cleaner

# 3. Train image model
python training/train_image.py --epochs 50 --batch-size 32

# 4. Train text model
python training/train_text.py --epochs 30 --batch-size 64
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
