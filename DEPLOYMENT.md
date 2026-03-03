# Deployment Guide — NSFW Content Filter

## 🚀 Quick Start (Local)

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/nsfw-content-filter.git
cd nsfw-content-filter
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Run FastAPI Backend
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

### 3. Run Streamlit Frontend
```bash
streamlit run app/streamlit_app.py
```
Dashboard at: `http://localhost:8501`

---

## 🐳 Docker Deployment

### Build & Run Both Services
```bash
docker-compose up --build -d
```
- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

### Build Individual Services
```bash
# API only
docker build --target api -t nsfw-filter-api .
docker run -p 8000:8000 nsfw-filter-api

# Streamlit only
docker build --target streamlit -t nsfw-filter-ui .
docker run -p 8501:8501 nsfw-filter-ui
```

---

## ☁️ Deploying to Streamlit Cloud

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: NSFW Content Filter"
git remote add origin https://github.com/YOUR_USERNAME/nsfw-content-filter.git
git push -u origin main
```

### Step 2: Connect to Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository (`nsfw-content-filter`)
5. Set the main file path to: `app/streamlit_app.py`
6. Click **"Deploy!"**

### Step 3: Configuration
Streamlit Cloud will:
- Install dependencies from `requirements.txt` automatically
- Use `.streamlit/config.toml` for theme/server settings
- The app runs the Streamlit frontend with the built-in predictor (no separate API needed)

### Step 4: Add Model Checkpoints
For production, upload trained model checkpoints:
1. Train models locally using the training scripts
2. Add checkpoint files to your repo (or use Git LFS for large files):
   ```bash
   git lfs install
   git lfs track "checkpoints/*.pth"
   git add checkpoints/
   git commit -m "Add trained model checkpoints"
   git push
   ```

### Step 5: Environment & Packages (if needed)
Create `packages.txt` for system-level dependencies:
```
libgl1-mesa-glx
ffmpeg
```
Streamlit Cloud will install these before your Python dependencies.

---

## 📁 Required Files for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme & server config |
| `app/streamlit_app.py` | Main Streamlit entry point |
| `packages.txt` | System dependencies (Streamlit Cloud) |
| `Dockerfile` | Container build (Docker deployment) |
| `docker-compose.yml` | Multi-service orchestration |

---

## ⚙️ Training Before Deployment

Before deploying to production, train the models:

```bash
# 1. Prepare data
python -m data_acquisition.scraper
python -m data_acquisition.data_cleaner

# 2. Train image model
python training/train_image.py --data-dir data_processed --epochs 50

# 3. Train text model
python training/train_text.py --data-dir data_text --epochs 30
```

Checkpoints will be saved to `checkpoints/`.
