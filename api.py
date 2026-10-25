
"""
SelfHealingNN — FastAPI Backend
Production-oriented API for CIFAR-10 fault injection and healing.

The CNN and autoencoders are loaded at startup.
CIFAR-10 data loaders are initialized lazily on the first request that
requires dataset evaluation or training-based healing.
"""

import os
import copy
import time
import threading
import logging
import io
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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


# ---------------------------------------------------------------------------
# Logging and configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("selfhealingnn")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "baseline_cnn.pth"
)
AE_FC1_PATH = os.path.join(
    PROJECT_ROOT, "models", "autoencoders", "ae_fc1.pth"
)
AE_FC2_PATH = os.path.join(
    PROJECT_ROOT, "models", "autoencoders", "ae_fc2.pth"
)
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

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
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
])

device = torch.device("cpu")
log.info("Using device: %s", device)


# ---------------------------------------------------------------------------
# Load model checkpoints at startup
# ---------------------------------------------------------------------------

def _load_state_dict(model, path):
    """Load a model checkpoint using the safest supported PyTorch option."""
    try:
        state = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        state = torch.load(path, map_location=device)

    model.load_state_dict(state)
    return model


log.info("Loading baseline CNN checkpoint ...")

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(
        f"Baseline checkpoint not found: {MODEL_PATH}"
    )

_baseline_cnn = CNN().to(device)
_load_state_dict(_baseline_cnn, MODEL_PATH)
_baseline_cnn.eval()

log.info("Baseline CNN loaded successfully")


# ---------------------------------------------------------------------------
# Optional autoencoder checkpoints
# ---------------------------------------------------------------------------

_ae_fc1: Optional[WeightAutoencoder] = None
_ae_fc2: Optional[WeightAutoencoder] = None
_ae_available = False


def _load_autoencoders() -> bool:
    global _ae_fc1, _ae_fc2

    if not (
        os.path.isfile(AE_FC1_PATH)
        and os.path.isfile(AE_FC2_PATH)
    ):
        log.warning(
            "Autoencoder checkpoints missing. "
            "AE-dependent methods will be unavailable."
        )
        return False

    try:
        fc1_in = _baseline_cnn.fc1.weight.shape[1]
        fc2_in = _baseline_cnn.fc2.weight.shape[1]

        model1 = WeightAutoencoder(input_dim=fc1_in).to(device)
        model2 = WeightAutoencoder(input_dim=fc2_in).to(device)

        _load_state_dict(model1, AE_FC1_PATH)
        _load_state_dict(model2, AE_FC2_PATH)

        model1.eval()
        model2.eval()

        _ae_fc1 = model1
        _ae_fc2 = model2

        log.info("Both autoencoder checkpoints loaded successfully")
        return True

    except Exception:
        log.exception("Could not load autoencoder checkpoints")
        _ae_fc1 = None
        _ae_fc2 = None
        return False


_ae_available = _load_autoencoders()


# ---------------------------------------------------------------------------
# Lazy CIFAR-10 data loaders
# ---------------------------------------------------------------------------

# These are deliberately NOT initialized at module import time.
# The health endpoint can therefore start without downloading CIFAR-10.

_train_loader = None
_val_loader = None
_data_loader_lock = threading.Lock()


def _get_data_loaders():
    """
    Initialize CIFAR-10 data loaders on demand.

    The lock prevents simultaneous first requests from initializing the
    data loaders more than once. Failed initialization can be retried.
    """
    global _train_loader, _val_loader

    if _train_loader is not None and _val_loader is not None:
        return _train_loader, _val_loader

    with _data_loader_lock:
        # Check again after acquiring the lock.
        if _train_loader is not None and _val_loader is not None:
            return _train_loader, _val_loader

        log.info("Initializing CIFAR-10 data loaders on demand ...")

        try:
            train_loader = get_train_loader()
            val_loader = get_val_loader()

            # Cache only after both loaders initialize successfully.
            _train_loader = train_loader
            _val_loader = val_loader

            log.info("CIFAR-10 data loaders initialized successfully")

        except Exception as exc:
            log.exception("CIFAR-10 data loader initialization failed")
            raise HTTPException(
                status_code=503,
                detail=(
                    "CIFAR-10 data is currently unavailable. "
                    "Check dataset download access and server logs, "
                    "then retry."
                ),
            ) from exc

    return _train_loader, _val_loader


# ---------------------------------------------------------------------------
# Accuracy and prediction helpers
# ---------------------------------------------------------------------------

_cached_baseline_acc: Optional[float] = None
_baseline_lock = threading.Lock()


def _fast_evaluate(
    model: nn.Module,
    loader,
    max_batches: int = EVALUATION_MAX_BATCHES,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            if index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total if total else 0.0


def _get_baseline_acc() -> float:
    global _cached_baseline_acc

    if _cached_baseline_acc is not None:
        return _cached_baseline_acc

    _, val_loader = _get_data_loaders()

    with _baseline_lock:
        if _cached_baseline_acc is None:
            _cached_baseline_acc = _fast_evaluate(
                _baseline_cnn,
                val_loader,
                max_batches=EVALUATION_MAX_BATCHES,
            )

            log.info(
                "Cached baseline accuracy: %.2f%%",
                _cached_baseline_acc,
            )

    return _cached_baseline_acc


def _predict(model: nn.Module, image_tensor: torch.Tensor):
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return (
        CLASSES[prediction.item()],
        round(confidence.item() * 100, 2),
    )


def _decode_image(file_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return TRANSFORM(image).unsqueeze(0).to(device)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Upload a valid image file.",
        ) from exc


# ---------------------------------------------------------------------------
# Healing strategy dispatcher
# ---------------------------------------------------------------------------

def _apply_healing(
    model: nn.Module,
    method: str,
    fault_ratio: float,
    grad_faults: dict,
    grad_vals: dict,
) -> nn.Module:

    if method == "gradient":
        model = targeted_heal_topk_gradient(
            model,
            grad_faults,
            grad_vals,
            topk_ratio=0.1,
            heal_conv=False,
        )

    elif method == "ae_dual":
        if not _ae_available:
            raise HTTPException(
                status_code=503,
                detail="Autoencoder checkpoints are unavailable.",
            )

        model = ae_heal_fc1_fc2(
            model, _ae_fc1, _ae_fc2, grad_faults, device
        )

    elif method == "hybrid":
        if not _ae_available:
            raise HTTPException(
                status_code=503,
                detail="Autoencoder checkpoints are unavailable.",
            )

        model = hybrid_heal(
            model,
            _ae_fc1,
            _ae_fc2,
            grad_faults,
            grad_vals,
            device,
            topk_ratio=0.05,
        )

    elif method == "regrowth":
        model = neuron_regrowth(
            model, grad_faults, grad_vals, device
        )

    elif method == "regrowth_gradient":
        model = regrowth_plus_gradient(
            model, grad_faults, grad_vals, device
        )

    elif method == "rerouting":
        model = dynamic_rerouting(
            model,
            grad_faults,
            device,
            suppression_strength=0.5,
        )

    elif method == "adaptive":
        model = adaptive_healing(
            model, fault_ratio, grad_faults, grad_vals
        )

    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported healing method: {method}",
        )

    return model


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SelfHealingNN API",
    description="Self-Healing Neural Network demonstration using CIFAR-10",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

_allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "http://localhost:5000",
]

_frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")

if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

if os.path.isdir(STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )


# One experiment at a time.
_experiment_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")

    if os.path.isfile(index_path):
        return FileResponse(index_path)

    return {
        "project": "SelfHealingNN",
        "message": "API is running. Visit /api/docs for documentation.",
    }


@app.get("/health", include_in_schema=False)
@app.get("/api/health")
async def health():
    """
    Lightweight health check.

    Does not download CIFAR-10 or calculate model accuracy.
    """
    return {
        "status": "ok",
        "device": str(device),
        "baseline_checkpoint": os.path.isfile(MODEL_PATH),
        "autoencoders_loaded": _ae_available,
        "data_loaders_initialized": (
            _train_loader is not None and _val_loader is not None
        ),
    }


@app.get("/api/methods")
async def methods():
    all_methods = [
        {
            "id": "gradient",
            "name": "Gradient-Based Repair",
            "ae_required": False,
            "description": "Reinitializes selected low-gradient faulty weights.",
        },
        {
            "id": "ae_dual",
            "name": "Dual AE Reconstruction",
            "ae_required": True,
            "description": "Reconstructs FC1 and FC2 weights using autoencoders.",
        },
        {
            "id": "hybrid",
            "name": "Hybrid (AE + Gradient)",
            "ae_required": True,
            "description": "Combines autoencoder reconstruction with gradient repair.",
        },
        {
            "id": "regrowth",
            "name": "Neuron Regrowth",
            "ae_required": False,
            "description": "Reinitializes damaged neurons.",
        },
        {
            "id": "regrowth_gradient",
            "name": "Regrowth + Gradient",
            "ae_required": False,
            "description": "Combines neuron regrowth and gradient repair.",
        },
        {
            "id": "rerouting",
            "name": "Dynamic Rerouting",
            "ae_required": False,
            "description": "Reduces the contribution of damaged neurons.",
        },
        {
            "id": "adaptive",
            "name": "Adaptive Controller",
            "ae_required": False,
            "description": "Selects a healing strategy using fault severity.",
        },
    ]

    return {
        "methods": all_methods,
        "autoencoders_available": _ae_available,
    }


@app.post("/api/baseline")
async def run_baseline(image: UploadFile = File(...)):
    file_bytes = await image.read()
    image_tensor = _decode_image(file_bytes)

    # Dataset initialization occurs only when accuracy is requested.
    baseline_acc = _get_baseline_acc()
    label, confidence = _predict(_baseline_cnn, image_tensor)

    return {
        "accuracy": round(baseline_acc, 2),
        "prediction": label,
        "confidence": confidence,
    }


@app.post("/api/fault")
async def run_fault(
    image: UploadFile = File(...),
    fault_ratio: int = Form(20),
):
    if fault_ratio not in VALID_FAULT_RATIOS:
        raise HTTPException(
            status_code=422,
            detail=f"fault_ratio must be one of {sorted(VALID_FAULT_RATIOS)}.",
        )

    file_bytes = await image.read()
    image_tensor = _decode_image(file_bytes)

    # The loader is initialized only when this endpoint is used.
    _, val_loader = _get_data_loaders()

    faulty_model = copy.deepcopy(_baseline_cnn)
    faulty_model = inject_weight_fault(
        faulty_model,
        fault_ratio=fault_ratio / 100.0,
    )

    label, confidence = _predict(faulty_model, image_tensor)
    faulty_acc = _fast_evaluate(
        faulty_model,
        val_loader,
        max_batches=EVALUATION_MAX_BATCHES,
    )

    return {
        "accuracy": round(faulty_acc, 2),
        "prediction": label,
        "confidence": confidence,
        "fault_ratio": fault_ratio,
    }


@app.post("/api/heal")
async def run_heal(
    image: UploadFile = File(...),
    fault_ratio: int = Form(20),
    method: str = Form("adaptive"),
):
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

    if method in {"ae_dual", "hybrid"} and not _ae_available:
        raise HTTPException(
            status_code=503,
            detail="The selected method requires available autoencoder checkpoints.",
        )

    file_bytes = await image.read()
    image_tensor = _decode_image(file_bytes)

    acquired = _experiment_lock.acquire(blocking=False)

    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Another experiment is running. Please wait and retry.",
        )

    try:
        train_loader, val_loader = _get_data_loaders()

        baseline_acc = _get_baseline_acc()
        ratio = fault_ratio / 100.0

        # 1. Create a fresh faulty model.
        faulty_model = copy.deepcopy(_baseline_cnn)
        faulty_model = inject_weight_fault(
            faulty_model,
            fault_ratio=ratio,
        )

        faulty_acc = _fast_evaluate(
            faulty_model,
            val_loader,
            max_batches=EVALUATION_MAX_BATCHES,
        )

        # 2. Detect faulty/low-gradient weights.
        grad_faults, grad_vals = detect_low_gradient_weights(
            faulty_model,
            train_loader,
            nn.CrossEntropyLoss(),
            device,
            max_batches=2,
        )

        # 3. Heal a separate copy of the faulty model.
        healed_model = copy.deepcopy(faulty_model)

        start_time = time.time()

        healed_model = _apply_healing(
            healed_model,
            method,
            ratio,
            grad_faults,
            grad_vals,
        )

        # 4. Lightweight fine-tuning of FC layers.
        healed_model = micro_finetune_fc(
            healed_model,
            train_loader,
            device,
            epochs=3,
            max_batches=10,
        )

        heal_time = round(time.time() - start_time, 2)

        # 5. Evaluate healed model.
        healed_acc = _fast_evaluate(
            healed_model,
            val_loader,
            max_batches=EVALUATION_MAX_BATCHES,
        )

        # 6. Calculate recovery efficiency.
        denominator = baseline_acc - faulty_acc

        if abs(denominator) > 0.01:
            recovery = (
                (healed_acc - faulty_acc) / denominator
            ) * 100
        else:
            recovery = 100.0 if abs(healed_acc - faulty_acc) <= 0.01 else 0.0

        # 7. Predict the uploaded image using the healed model.
        label, confidence = _predict(healed_model, image_tensor)

        return {
            "baseline_acc": round(baseline_acc, 2),
            "faulty_acc": round(faulty_acc, 2),
            "healed_acc": round(healed_acc, 2),
            "recovery": round(recovery, 2),
            "prediction": label,
            "confidence": confidence,
            "method": method,
            "fault_ratio": fault_ratio,
            "heal_time_s": heal_time,
        }

    finally:
        _experiment_lock.release()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error(
        "Unhandled error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error. Check server logs."
        },
    )


# ---------------------------------------------------------------------------
# Local development entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
    )
