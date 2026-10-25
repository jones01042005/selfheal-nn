"""
SelfHealingNN — FastAPI Backend
--------------------------------
Replaces the original Flask app.py with a production-ready FastAPI server.

All healing logic is unchanged: heal.py, fault.py, train.py, ae_model.py
are imported directly. This file only handles HTTP concerns, validation,
concurrency safety, and structured JSON responses.
"""

import os
import copy
import time
import uuid
import threading
import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# ─── Project Imports ────────────────────────────────────────────────────────
from train import CNN, get_val_loader, get_train_loader
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import (
    targeted_heal_topk_gradient,
    micro_finetune_fc,
    ae_heal_fc1_fc2,
    hybrid_heal,
    neuron_regrowth,
    regrowth_plus_gradient,
    dynamic_rerouting,
)
from main import adaptive_healing
from ae_model import WeightAutoencoder

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("selfhealingnn")

# ─── Constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pth")
AE_FC1_PATH  = os.path.join(PROJECT_ROOT, "models", "autoencoders", "ae_fc1.pth")
AE_FC2_PATH  = os.path.join(PROJECT_ROOT, "models", "autoencoders", "ae_fc2.pth")

VALID_FAULT_RATIOS = {10, 20, 30, 40}
EVALUATION_MAX_BATCHES = 20
VALID_METHODS = {
    "gradient",
    "ae_dual",
    "hybrid",
    "regrowth",
    "regrowth_gradient",
    "rerouting",
    "adaptive",
}

CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ─── Device ─────────────────────────────────────────────────────────────────
device = torch.device("cpu")          # Force CPU for portable deployment
log.info("Using device: %s", device)

# ─── Model Startup Loading ───────────────────────────────────────────────────
log.info("Loading baseline CNN checkpoint …")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Checkpoint not found: {MODEL_PATH}\n"
        "Run `python train.py` to train the baseline model first."
    )

_baseline_cnn = CNN().to(device)
try:
    _baseline_cnn.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
except TypeError:
    # weights_only kwarg added in PyTorch 1.13; fall back for older versions
    _baseline_cnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
_baseline_cnn.eval()
log.info("Baseline CNN loaded ✓")

# ─── Autoencoder Loading (optional — only needed for ae_dual/hybrid) ─────────
_ae_fc1: Optional[WeightAutoencoder] = None
_ae_fc2: Optional[WeightAutoencoder] = None

def _load_autoencoders() -> bool:
    """Load pre-trained AE checkpoints. Returns True if both loaded."""
    global _ae_fc1, _ae_fc2
    if not os.path.exists(AE_FC1_PATH) or not os.path.exists(AE_FC2_PATH):
        log.warning("AE checkpoints missing — ae_dual and hybrid methods disabled.")
        return False
    try:
        fc1_in = _baseline_cnn.fc1.weight.shape[1]  # 4096
        fc2_in = _baseline_cnn.fc2.weight.shape[1]  # 512
        _ae_fc1 = WeightAutoencoder(input_dim=fc1_in).to(device)
        _ae_fc2 = WeightAutoencoder(input_dim=fc2_in).to(device)
        try:
            _ae_fc1.load_state_dict(torch.load(AE_FC1_PATH, map_location=device, weights_only=True))
            _ae_fc2.load_state_dict(torch.load(AE_FC2_PATH, map_location=device, weights_only=True))
        except TypeError:
            _ae_fc1.load_state_dict(torch.load(AE_FC1_PATH, map_location=device))
            _ae_fc2.load_state_dict(torch.load(AE_FC2_PATH, map_location=device))
        _ae_fc1.eval()
        _ae_fc2.eval()
        log.info("Autoencoders loaded ✓  (ae_fc1=%s, ae_fc2=%s)", AE_FC1_PATH, AE_FC2_PATH)
        return True
    except Exception as exc:
        log.error("Failed to load autoencoders: %s", exc)
        return False

_ae_available = _load_autoencoders()

# ─── Data Loaders (loaded once; download=True in get_val_loader) ─────────────
log.info("Initialising CIFAR-10 data loaders …")
_val_loader   = get_val_loader()
_train_loader = get_train_loader()
log.info("Data loaders ready ✓")

# ─── Cached Baseline Accuracy ────────────────────────────────────────────────
_cached_baseline_acc: Optional[float] = None
_baseline_lock = threading.Lock()

def _get_baseline_acc() -> float:
    global _cached_baseline_acc
    with _baseline_lock:
        if _cached_baseline_acc is None:
            _cached_baseline_acc = _fast_evaluate(
                _baseline_cnn, _val_loader, max_batches=EVALUATION_MAX_BATCHES
            )
            log.info("Cached baseline accuracy: %.2f%%", _cached_baseline_acc)
    return _cached_baseline_acc

# ─── Concurrency Lock ────────────────────────────────────────────────────────
# One experiment at a time; prevents shared-model corruption.
_experiment_lock = threading.Lock()

# ─── Helpers ────────────────────────────────────────────────────────────────
def _fast_evaluate(model: nn.Module, loader, max_batches: int = 20) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def _predict(model: nn.Module, image_tensor: torch.Tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs  = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASSES[pred.item()], round(conf.item() * 100, 2)


def _decode_image(file_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return TRANSFORM(image).unsqueeze(0).to(device)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _apply_healing(
    model: nn.Module,
    method: str,
    fault_ratio: float,
    grad_faults: dict,
    grad_vals: dict,
) -> nn.Module:
    """Dispatch to the correct healing function based on `method`."""
    if method == "gradient":
        model = targeted_heal_topk_gradient(
            model, grad_faults, grad_vals, topk_ratio=0.1, heal_conv=False
        )
    elif method == "ae_dual":
        if not _ae_available:
            raise HTTPException(
                status_code=503,
                detail="Autoencoder checkpoints not available on this server.",
            )
        model = ae_heal_fc1_fc2(model, _ae_fc1, _ae_fc2, grad_faults, device)
    elif method == "hybrid":
        if not _ae_available:
            raise HTTPException(
                status_code=503,
                detail="Autoencoder checkpoints not available on this server.",
            )
        model = hybrid_heal(model, _ae_fc1, _ae_fc2, grad_faults, grad_vals, device, topk_ratio=0.05)
    elif method == "regrowth":
        model = neuron_regrowth(model, grad_faults, grad_vals, device)
    elif method == "regrowth_gradient":
        model = regrowth_plus_gradient(model, grad_faults, grad_vals, device)
    elif method == "rerouting":
        model = dynamic_rerouting(model, grad_faults, device, suppression_strength=0.5)
    elif method == "adaptive":
        model = adaptive_healing(model, fault_ratio, grad_faults, grad_vals)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
    return model


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="SelfHealingNN API",
    description="Self-Healing Neural Network demonstration — CIFAR-10 CNN",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS: allow the Vercel frontend + local development
_allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "http://localhost:5000",
]
_vercel_url = os.getenv("FRONTEND_URL", "")
if _vercel_url:
    _allowed_origins.append(_vercel_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ─── Static Files (serve frontend when running locally) ──────────────────────
_static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index():
    _index = os.path.join(_static_dir, "index.html")
    if os.path.exists(_index):
        return FileResponse(_index)
    return {"message": "SelfHealingNN API — visit /api/docs for endpoint reference."}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "baseline_checkpoint": os.path.exists(MODEL_PATH),
        "autoencoders_loaded": _ae_available,
    }


@app.get("/api/methods")
async def methods():
    """Return available healing strategies and their descriptions."""
    all_methods = [
        {"id": "gradient",          "name": "Gradient-Based Repair",      "ae_required": False,
         "description": "Reinitialises low-gradient faulty weights with small random values."},
        {"id": "ae_dual",           "name": "Dual AE Reconstruction",     "ae_required": True,
         "description": "Uses pre-trained autoencoders to reconstruct FC1 and FC2 weight matrices."},
        {"id": "hybrid",            "name": "Hybrid (AE + Gradient)",     "ae_required": True,
         "description": "AE reconstruction followed by gradient fine-repair."},
        {"id": "regrowth",          "name": "Neuron Regrowth",            "ae_required": False,
         "description": "Reinitialises neurons where >90% of incoming weights are dead."},
        {"id": "regrowth_gradient", "name": "Regrowth + Gradient",       "ae_required": False,
         "description": "Neuron regrowth followed by gradient fine-repair."},
        {"id": "rerouting",         "name": "Dynamic Rerouting",         "ae_required": False,
         "description": "Scales down outputs of damaged neurons to reroute information flow."},
        {"id": "adaptive",          "name": "Adaptive Controller",        "ae_required": False,
         "description": "Automatically selects the best strategy based on fault severity."},
    ]
    return {
        "methods": all_methods,
        "autoencoders_available": _ae_available,
    }


@app.post("/api/baseline")
async def run_baseline(
    image: UploadFile = File(...),
):
    """
    Evaluate the clean baseline CNN and predict the uploaded image.
    Returns: accuracy (estimated from val set), prediction, confidence.
    """
    file_bytes = await image.read()
    image_tensor = _decode_image(file_bytes)

    baseline_acc = _get_baseline_acc()
    label, conf  = _predict(_baseline_cnn, image_tensor)

    return {
        "accuracy":    round(baseline_acc, 2),
        "prediction":  label,
        "confidence":  conf,
    }


@app.post("/api/fault")
async def run_fault(
    image: UploadFile = File(...),
    fault_ratio: int  = Form(20),
):
    """
    Inject faults into a fresh copy of the model and evaluate.
    Returns: accuracy after fault, prediction, confidence.
    """
    if fault_ratio not in VALID_FAULT_RATIOS:
        raise HTTPException(
            status_code=422,
            detail=f"fault_ratio must be one of {sorted(VALID_FAULT_RATIOS)}.",
        )

    file_bytes   = await image.read()
    image_tensor = _decode_image(file_bytes)

    ratio = fault_ratio / 100.0
    faulty_model = copy.deepcopy(_baseline_cnn)
    faulty_model = inject_weight_fault(faulty_model, fault_ratio=ratio)

    label, conf = _predict(faulty_model, image_tensor)
    faulty_acc  = _fast_evaluate(
        faulty_model, _val_loader, max_batches=EVALUATION_MAX_BATCHES
    )

    return {
        "accuracy":    round(faulty_acc, 2),
        "prediction":  label,
        "confidence":  conf,
        "fault_ratio": fault_ratio,
    }


@app.post("/api/heal")
async def run_heal(
    image: UploadFile       = File(...),
    fault_ratio: int        = Form(20),
    method: str             = Form("adaptive"),
):
    """
    Inject faults, run the selected healing strategy, micro-fine-tune,
    then evaluate. Returns all three accuracy figures + recovery efficiency.

    This endpoint holds a threading lock to prevent concurrent experiments
    from corrupting shared model state.
    """
    if fault_ratio not in VALID_FAULT_RATIOS:
        raise HTTPException(
            status_code=422,
            detail=f"fault_ratio must be one of {sorted(VALID_FAULT_RATIOS)}.",
        )
    if method not in VALID_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"method must be one of {sorted(VALID_METHODS)}.",
        )

    file_bytes   = await image.read()
    image_tensor = _decode_image(file_bytes)

    ratio = fault_ratio / 100.0

    acquired = _experiment_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Another experiment is running. Please wait and retry.",
        )

    try:
        baseline_acc = _get_baseline_acc()

        # ── 1. Fresh faulty copy ─────────────────────────────────────────────
        faulty_model = copy.deepcopy(_baseline_cnn)
        faulty_model = inject_weight_fault(faulty_model, fault_ratio=ratio)
        faulty_acc   = _fast_evaluate(
            faulty_model, _val_loader, max_batches=EVALUATION_MAX_BATCHES
        )

        # ── 2. Gradient fault detection ───────────────────────────────────────
        grad_faults, grad_vals = detect_low_gradient_weights(
            faulty_model, _train_loader, nn.CrossEntropyLoss(), device, max_batches=2
        )

        # ── 3. Healing ────────────────────────────────────────────────────────
        t0 = time.time()
        healed_model = copy.deepcopy(faulty_model)
        healed_model = _apply_healing(healed_model, method, ratio, grad_faults, grad_vals)

        # ── 4. Micro fine-tune FC layers ──────────────────────────────────────
        healed_model = micro_finetune_fc(
            healed_model, _train_loader, device, epochs=3, max_batches=10
        )
        heal_time = round(time.time() - t0, 2)

        # ── 5. Evaluate ───────────────────────────────────────────────────────
        healed_acc = _fast_evaluate(
            healed_model, _val_loader, max_batches=EVALUATION_MAX_BATCHES
        )

        # ── 6. Recovery efficiency ────────────────────────────────────────────
        denom = baseline_acc - faulty_acc
        if abs(denom) > 0.01:
            recovery = round(((healed_acc - faulty_acc) / denom) * 100, 2)
        else:
            recovery = 100.0

        # ── 7. Prediction on the uploaded image ───────────────────────────────
        label, conf = _predict(healed_model, image_tensor)

        return {
            "baseline_acc": round(baseline_acc, 2),
            "faulty_acc":   round(faulty_acc, 2),
            "healed_acc":   round(healed_acc, 2),
            "recovery":     recovery,
            "prediction":   label,
            "confidence":   conf,
            "method":       method,
            "fault_ratio":  fault_ratio,
            "heal_time_s":  heal_time,
        }

    finally:
        _experiment_lock.release()


# ─── Global Error Handler ────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Check server logs."},
    )


# ─── Entrypoint (local development) ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
