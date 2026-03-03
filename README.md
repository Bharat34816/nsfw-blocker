# 🛡️ NSFW Content Filter  

AI-powered multi-modal content moderation — **Images • Videos • Text**

A complete end-to-end NSFW detection system built from scratch using custom deep learning architectures (no pre-trained weights), real-time inference, and a production-ready deployment stack.

---

## 🌐 Live Demo

🚀 **[Live Application — Click to Run](https://nsfw-blocker-ezf7k27mppedyfhqvyckhr.streamlit.app/)**

---

## ✨ Features

- 🖼️ **Image Classification** — Custom EfficientNet-B0 (~5.3M params) trained from scratch  
- 🎥 **Video Analysis** — Scene-change detection + keyframe-level classification  
- 📝 **Text Moderation** — Bi-LSTM with Attention mechanism  
- 🎯 **Zero False Positives Strategy** — Three-zone confidence thresholding  
- ⚠️ **Manual Review Flag** — Borderline cases flagged for moderation  
- 🎨 **Streamlit Dashboard** — Modern dark-theme UI with real-time feedback  

---

# 📁 Project Structure

```
.
├── .streamlit/
│
├── NsfwContentM_main/
│   ├── custom_layers/
│   ├── models/
│   ├── .gitignore
│   ├── NSFWvsSFW.ipynb
│   ├── __init__.py
│   ├── app.py
│   └── requirements.txt
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── data_acquisition/
├── inference/
├── models/
├── scripts/
├── training/
│
├── .gitignore
├── README.md
├── packages.txt
└── requirements.txt
```

---

# 🚀 Quick Start

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

Application runs at:

```
http://localhost:8501
```

---

# 🧠 Model Architectures

---

## 🖼️ EfficientNet-B0 (Image / Video)

Custom implementation trained **from scratch**.

| Component | Details |
|------------|----------|
| Architecture | MBConv + Squeeze-and-Excitation |
| Parameters | ~5.3M |
| Initialization | Kaiming (He) Normal |
| Input Size | 224 × 224 × 3 |
| Output | 2-class Softmax (Safe / NSFW) |
| Optimizer | AdamW |
| LR Strategy | Cosine Annealing |
| Precision | Mixed Precision |

No transfer learning used.

---

## 📝 Text Model — Bi-LSTM + Attention

CNN removed. Pure sequential modeling.

### Architecture Flow

```
Embedding (128-dim)
      ↓
Bidirectional LSTM
      ↓
Attention Layer
      ↓
Fully Connected Layer
      ↓
Softmax (Safe / NSFW)
```

### Model Details

| Component | Details |
|------------|----------|
| Embedding | Learned from scratch (128-dim) |
| Vocabulary | Max 30,000 tokens |
| Max Length | 256 tokens |
| Encoder | Bi-LSTM (hidden size = 128) |
| Attention | Trainable context attention |
| Output | 3-class Softmax |
| Optimizer | Adam |

---

# 🎯 Confidence Thresholding System (Images/Videos)

To minimize false positives, predictions are divided into three zones:

| Zone | NSFW Score | Decision | Action |
|------|------------|----------|--------|
| 🟢 Safe | ≤ 0.15 | SAFE | Auto-allow |
| 🟡 Review | 0.15 – 0.85 | REVIEW | Flag for human moderation |
| 🔴 NSFW | ≥ 0.85 | NSFW | Auto-block |

Thresholds are adjustable from the Streamlit sidebar.

---

# 🎥 Video Processing Pipeline

1. Scene-change detection  
2. Keyframe extraction  
3. Frame-level classification  
4. Aggregated NSFW confidence score  

Optimized for efficient real-time inference.

---

# 🏋️ Training

## 1️⃣ Train Image Model

```bash
python train_image.py --epochs 50 --batch-size 32
```

## 2️⃣ Train Text Model

```bash
python train_text.py --epochs 30 --batch-size 64
```

---

# 📊 Design Philosophy

- No pre-trained weights  
- Fully custom deep learning implementation  
- Modular yet structured architecture  
- Production-ready inference pipeline  
- Conservative thresholding for safer moderation  

---

# 👨‍💻 Author

Built as a complete end-to-end AI moderation system combining deep learning, backend engineering, and deployment automation.

---

⭐ If you found this useful, consider giving the repository a star!
