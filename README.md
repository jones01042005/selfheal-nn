# SelfHealingNN 🧠

An interactive demonstration of **self-healing neural network techniques** applied to a CIFAR-10 CNN. Hardware faults (weight zeroing) are injected at configurable levels, then repaired using 7 different healing strategies — from gradient-based repair to autoencoder reconstruction and neuron regrowth.

> **College Final Year Project** — built with PyTorch, FastAPI, and Vanilla JS.

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Frontend (static/)                     │  → Vercel (free static hosting)
│  HTML + CSS + JavaScript + Chart.js     │
└───────────────────┬─────────────────────┘
                    │ HTTP (BACKEND_URL env var)
┌───────────────────▼─────────────────────┐
│  Backend (api.py — FastAPI)             │  → Render / Railway / Docker
│  • /api/health   • /api/methods         │
│  • /api/baseline • /api/fault           │
│  • /api/heal                            │
└───────────────────┬─────────────────────┘
                    │ imports
┌───────────────────▼─────────────────────┐
│  Core ML Modules                        │
│  train.py  fault.py  heal.py            │
│  ae_model.py  train_ae.py  main.py      │
└───────────────────┬─────────────────────┘
                    │ loads
┌───────────────────▼─────────────────────┐
│  Checkpoints (models/)                  │
│  baseline_cnn.pth  (~10 MB)             │
│  autoencoders/ae_fc1.pth  (~36 MB)      │
│  autoencoders/ae_fc2.pth  (~0.6 MB)     │
└─────────────────────────────────────────┘
```

---

## CNN Architecture

| Layer    | Output Shape | Details |
|----------|-------------|---------|
| Input    | 3 × 32 × 32 | CIFAR-10 RGB |
| conv1    | 64 × 16 × 16 | Conv(3→64) × 2, BN, ReLU, MaxPool |
| conv2    | 128 × 8 × 8 | Conv(64→128) × 2, BN, ReLU, MaxPool |
| conv3    | 256 × 4 × 4 | Conv(128→256), BN, ReLU, MaxPool |
| fc1      | 512 | Linear(4096→512), Dropout(0.4) |
| fc2      | 10 | Linear(512→10) |

Baseline accuracy: **~85%** on CIFAR-10 test set.

The API loads only the validation dataset for baseline/fault evaluation and
loads the training dataset only for healing. CIFAR-10 is prepared in the
container image, not downloaded during a request. Baseline, faulty, and healed
accuracy use the same first 10 validation batches (1,280 samples) to limit
CPU-bound request time on small instances. Inference is chunked into groups of
16 and uses `torch.inference_mode()` to reduce inference overhead and peak
memory. Accuracy is calculated over all evaluated samples; this smaller,
deterministic validation subset is an estimate rather than full-dataset
accuracy. Per-batch timing and failures are recorded in the API logs.
Autoencoder checkpoints are loaded only when an autoencoder-based healing
strategy is selected.

---

## Healing Strategies

| Method | ID | AE Required | Best Fault Range |
|---|---|---|---|
| Adaptive Controller | `adaptive` | No | All (auto-selects) |
| Gradient-Based Repair | `gradient` | No | >35% |
| Neuron Regrowth | `regrowth` | No | 20% |
| Regrowth + Gradient | `regrowth_gradient` | No | <15% |
| Dynamic Rerouting | `rerouting` | No | 30% |
| Dual AE Reconstruction | `ae_dual` | Yes | All |
| Hybrid (AE + Gradient) | `hybrid` | Yes | All |

---

## Local Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone and set up virtual environment
```bash
git clone <repo-url>
cd SelfhealingNN
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install dependencies (development)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn[standard] python-multipart numpy pandas matplotlib Pillow scikit-learn
```

### 3. Train the model (if checkpoint missing)
```bash
# Train baseline CNN (~30 epochs, ~10 min on CPU)
python train.py

# Train autoencoders (required for ae_dual / hybrid methods)
python train_ae.py
```

### 4. Run the backend API
```bash
python api.py
# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/api/docs
```

### 5. Open the frontend
Visit `http://localhost:8000` (the API also serves the frontend and its assets).

---

## Docker Build & Run

### Build the image
```bash
docker build -t selfhealingnn .
```

The build downloads and verifies the CIFAR-10 training and test sets into the
image. The API uses those files with `download=False`, so an inference request
does not need network access or write dataset files to the Render filesystem.

### Run the container
```bash
docker run -p 10000:10000 selfhealingnn
```

### Test health endpoint
```bash
curl http://localhost:10000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "device": "cpu",
  "baseline_checkpoint": true,
  "autoencoders_loaded": true
}
```

---

## Vercel Deployment (Frontend)

The frontend is pure static HTML/CSS/JS — deploy it to Vercel directly.

### Option A — Vercel Dashboard
1. Push this repo to GitHub.
2. Go to [vercel.com/new](https://vercel.com/new) → Import Git Repository.
3. Set **Framework Preset** → `Other`.
4. Add the `BACKEND_URL` environment variable for the Production environment, for example `https://your-api.onrender.com`.
5. Deploy. `vercel.json` builds the frontend into `dist/` and generates its backend configuration.

### Option B — Vercel CLI
```bash
npm i -g vercel
vercel login
vercel --prod
```

---

## Render Deployment (Backend)

Vercel supports Python functions, but this API loads PyTorch, model checkpoints, and CIFAR-10 data at startup and runs CPU-intensive experiments. This project keeps the backend on a container host to avoid serverless cold-start, memory, and execution-limit risks, and deploys only the static frontend to Vercel.

### Steps
1. Push to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service.
3. Connect your GitHub repo.
4. **Environment**: Docker.
5. **Dockerfile Path**: `./Dockerfile`.
6. **Plan**: `render.yaml` currently selects Free (512 MB RAM); verify that this is sufficient before deployment.
7. Add environment variable:
   - `FRONTEND_URL` = `https://your-app.vercel.app`
8. Deploy.

The service's `PORT` is supplied by Render and the container listens on it. Local API startup used approximately 486 MB RSS on this machine; this is not a container measurement, but it signals little headroom on a 512 MB plan, especially during experiments. Confirm capacity before deployment; no hosted deployment has been performed or verified from this workspace.

> **Note**: Free services may spin down after inactivity. Cold starts and memory availability depend on the current plan.

### Alternative Platforms
- **Railway** (`railway.app`) — similar Docker support, $5/mo hobby plan
- **Fly.io** — 3 free VMs, 256 MB RAM (tight for PyTorch)
- **Google Cloud Run** — generous free tier, ~300 MB limit per container

---

## API Reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/api/health` | GET | — | Service health check |
| `/api/methods` | GET | — | List available healing strategies |
| `/api/baseline` | POST | `image: File` | Evaluate clean model, predict image |
| `/api/fault` | POST | `image: File, fault_ratio: int` | Inject fault, evaluate, predict |
| `/api/heal` | POST | `image: File, fault_ratio: int, method: str` | Full heal pipeline + metrics |
| `/api/docs` | GET | — | Swagger interactive docs |

Full interactive docs: `http://localhost:10000/api/docs`

---

## Required Environment Variables

| Variable | Where | Value |
|---|---|---|
| `FRONTEND_URL` | Backend (Render) | `https://your-app.vercel.app` |
| `BACKEND_URL` | Frontend (Vercel build environment) | `https://your-api.onrender.com` |
| `PORT` | Backend | Supplied by the container host; defaults to `10000` |

---

## Testing

```bash
# 1. Syntax check
python -c "import api; print('✓ api.py imports OK')"

# 2. Health endpoint
curl http://localhost:8000/api/health

# 3. Methods endpoint
curl http://localhost:8000/api/methods

# 4. Baseline (replace test.jpg with any image)
curl -X POST http://localhost:8000/api/baseline \
     -F "image=@test_images/test.jpg"

# 5. Fault injection
curl -X POST http://localhost:8000/api/fault \
     -F "image=@test_images/test.jpg" \
     -F "fault_ratio=20"

# 6. Healing (adaptive, 20% fault)
curl -X POST http://localhost:8000/api/heal \
     -F "image=@test_images/test.jpg" \
     -F "fault_ratio=20" \
     -F "method=adaptive"

# 7. Docker build
docker build -t selfhealingnn .

# 8. Docker run + health check
docker run -d -p 8000:8000 --name shn selfhealingnn
sleep 10
curl http://localhost:8000/api/health
docker stop shn && docker rm shn
```

---

## File Structure

```
SelfhealingNN/
├── api.py                  ← FastAPI backend (NEW — replaces app.py)
├── app.py                  ← Original Flask backend (preserved, not used)
├── main.py                 ← Adaptive healing controller + experiment runner
├── train.py                ← CNN definition + CIFAR-10 data loaders
├── fault.py                ← Fault injection + gradient detection
├── heal.py                 ← 7 healing strategy implementations
├── ae_model.py             ← WeightAutoencoder class
├── train_ae.py             ← Autoencoder training script
├── healing_strategy.py     ← Standalone healing comparison script
├── models/
│   ├── baseline_cnn.pth        ← Trained CNN checkpoint (~10 MB)
│   └── autoencoders/
│       ├── ae_fc1.pth           ← FC1 autoencoder (~36 MB)
│       └── ae_fc2.pth           ← FC2 autoencoder (~0.6 MB)
├── static/
│   ├── index.html           ← Frontend (NEW — redesigned)
│   ├── style.css            ← Styles (NEW — design system)
│   ├── app.js               ← Frontend logic (NEW — method selection)
│   └── chart.min.js         ← Chart.js (local copy)
├── datasets/                ← CIFAR-10 (auto-downloaded, not in git)
├── experiments/             ← Generated plots and CSVs
├── Dockerfile               ← Container build (NEW)
├── .dockerignore            ← Docker exclusions (NEW)
├── .gitignore               ← Git exclusions (NEW)
├── requirements.txt         ← Development dependencies
├── requirements-prod.txt    ← Production dependencies (NEW)
├── vercel.json              ← Vercel static frontend config (NEW)
├── render.yaml              ← Render backend Docker config (NEW)
└── README.md                ← This file (NEW)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Checkpoint not found` | `baseline_cnn.pth` missing | Run `python train.py` |
| `AE checkpoints missing` | ae_fc1/2.pth missing | Run `python train_ae.py` |
| Heal request times out | Slow CPU, large fault | Reduce `max_batches` in `api.py` |
| CORS error in browser | Backend URL not whitelisted | Add `FRONTEND_URL` env var |
| 429 Too Many Requests | Another experiment running | Wait for previous request to finish |
| `ae_dual` / `hybrid` disabled | AE not loaded | Check `autoencoders_loaded` in `/api/health` |
