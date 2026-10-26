
"""
SelfHealingNN — FastAPI Backend
Production-oriented API for CIFAR-10 fault injection and healing.

The CNN is loaded at startup; baseline dataset accuracy is evaluated in a
background startup task, and autoencoders are loaded on demand.
CIFAR-10 data loaders are initialized lazily when dataset evaluation or
training-based healing requires them.
"""

import asyncio
import os
import copy
import time
import threading
import logging
import io
from contextlib import asynccontextmanager
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

from train import CNN, DATA_DIR, get_val_loader, get_train_loader
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
EVALUATION_MAX_BATCHES = 10
EVALUATION_INFERENCE_BATCH_SIZE = 16

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
_autoencoder_lock = threading.Lock()
_ae_available = (
    os.path.isfile(AE_FC1_PATH)
    and os.path.isfile(AE_FC2_PATH)
)
_ae_loaded = False


def _load_autoencoders() -> bool:
    global _ae_fc1, _ae_fc2, _ae_loaded

    if _ae_loaded:
        return True

    if not _ae_available:
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
        _ae_loaded = True

        log.info("Both autoencoder checkpoints loaded successfully")
        return True

    except Exception:
        log.exception("Could not load autoencoder checkpoints")
        _ae_fc1 = None
        _ae_fc2 = None
        _ae_loaded = False
        return False


def _ensure_autoencoders_loaded() -> bool:
    if _ae_loaded:
        return True

    with _autoencoder_lock:
        return _load_autoencoders()


# ---------------------------------------------------------------------------
# Lazy CIFAR-10 data loaders
# ---------------------------------------------------------------------------

# These are deliberately NOT initialized at module import time.
# The health endpoint can therefore start without downloading CIFAR-10.

_train_loader = None
_val_loader = None
_data_loader_lock = threading.Lock()


def _get_data_loaders(include_train=True):
    """
    Initialize the required CIFAR-10 data loaders on demand.

    Startup baseline-accuracy and fault evaluation need only validation data.
    Healing also needs training data. Each loader is cached only after it
    initializes.
    """
    global _train_loader, _val_loader

    if _val_loader is not None and (
        not include_train or _train_loader is not None
    ):
        return _train_loader, _val_loader

    with _data_loader_lock:
        if _val_loader is not None and (
            not include_train or _train_loader is not None
        ):
            return _train_loader, _val_loader

        try:
            if _val_loader is None:
                started = time.perf_counter()
                log.info("Validation data-loader initialization started")
                val_loader = get_val_loader(download=False)
                _val_loader = val_loader
                log.info(
                    "Validation data loader initialized in %.2fs",
                    time.perf_counter() - started,
                )

            if include_train and _train_loader is None:
                started = time.perf_counter()
                log.info("Training data-loader initialization started")
                train_loader = get_train_loader(download=False)
                _train_loader = train_loader
                log.info(
                    "Training data loader initialized in %.2fs",
                    time.perf_counter() - started,
                )

        except Exception as exc:
            log.exception(
                "CIFAR-10 data-loader initialization failed for %s. "
                "API requests do not download missing data; include valid "
                "train and test batches in the container image.",
                DATA_DIR,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "CIFAR-10 train or validation data is missing or "
                    f"incomplete at {DATA_DIR}. Redeploy an image with valid "
                    "CIFAR-10 data; the API does not download it per request."
                ),
            ) from exc

    return _train_loader, _val_loader


# ---------------------------------------------------------------------------
# Accuracy and prediction helpers
# ---------------------------------------------------------------------------

_cached_baseline_acc: Optional[float] = None
_baseline_lock = threading.Lock()
_baseline_status_lock = threading.Lock()
_baseline_start_lock = threading.Lock()
_baseline_evaluation_status = "pending"
_baseline_evaluation_error: Optional[str] = None
_baseline_startup_started = False
_baseline_startup_thread: Optional[threading.Thread] = None


def _set_baseline_evaluation_status(
    status: str,
    error: Optional[str] = None,
) -> None:
    global _baseline_evaluation_status, _baseline_evaluation_error

    with _baseline_status_lock:
        _baseline_evaluation_status = status
        _baseline_evaluation_error = error


def _fast_evaluate(
    model: nn.Module,
    loader,
    max_batches: int = EVALUATION_MAX_BATCHES,
) -> float:
    started = time.perf_counter()
    target_batches = min(max_batches, len(loader))
    log.info(
        "Accuracy evaluation started (max_batches=%d, available_batches=%d)",
        max_batches,
        len(loader),
    )

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            if index >= max_batches:
                break

            batch_started = time.perf_counter()
            labels = labels.to(device)
            batch_correct = 0
            for start in range(
                0, images.size(0), EVALUATION_INFERENCE_BATCH_SIZE
            ):
                end = start + EVALUATION_INFERENCE_BATCH_SIZE
                image_chunk = images[start:end].to(device)
                label_chunk = labels[start:end]
                predictions = model(image_chunk).argmax(dim=1)
                batch_correct += (predictions == label_chunk).sum().item()

            correct += batch_correct
            total += labels.size(0)

            completed_batches = index + 1
            log.info(
                "Accuracy evaluation progress: %d/%d batches, "
                "%d samples (batch %.2fs, total %.2fs)",
                completed_batches,
                target_batches,
                total,
                time.perf_counter() - batch_started,
                time.perf_counter() - started,
            )

    if total == 0:
        raise RuntimeError("Accuracy evaluation received no validation samples.")

    accuracy = 100.0 * correct / total
    log.info(
        "Accuracy evaluation completed: %.2f%% over %d samples in %.2fs",
        accuracy,
        total,
        time.perf_counter() - started,
    )
    return accuracy


def _get_baseline_acc() -> float:
    global _cached_baseline_acc

    if _cached_baseline_acc is not None:
        log.info("Using cached baseline accuracy: %.2f%%", _cached_baseline_acc)
        _set_baseline_evaluation_status("completed")
        return _cached_baseline_acc

    _set_baseline_evaluation_status("running")

    try:
        _, val_loader = _get_data_loaders(include_train=False)

        with _baseline_lock:
            if _cached_baseline_acc is None:
                evaluation_started = time.perf_counter()
                log.info(
                    "Baseline accuracy evaluation started "
                    "(max_batches=%d, inference_batch_size=%d)",
                    EVALUATION_MAX_BATCHES,
                    EVALUATION_INFERENCE_BATCH_SIZE,
                )
                try:
                    accuracy = _fast_evaluate(
                        _baseline_cnn,
                        val_loader,
                        max_batches=EVALUATION_MAX_BATCHES,
                    )
                except MemoryError as exc:
                    log.exception(
                        "Baseline accuracy evaluation ran out of memory"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Baseline accuracy evaluation ran out of memory. "
                            "Retry later or use an instance with more memory."
                        ),
                    ) from exc
                except Exception as exc:
                    log.exception("Baseline accuracy evaluation failed")
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Baseline accuracy evaluation failed. "
                            "Check server logs for the traceback."
                        ),
                    ) from exc

                _cached_baseline_acc = accuracy
                log.info(
                    "Baseline accuracy evaluation completed and cached: "
                    "%.2f%% in %.2fs",
                    _cached_baseline_acc,
                    time.perf_counter() - evaluation_started,
                )
    except HTTPException as exc:
        _set_baseline_evaluation_status("failed", str(exc.detail))
        raise
    except Exception as exc:
        _set_baseline_evaluation_status("failed", str(exc))
        raise

    _set_baseline_evaluation_status("completed")
    return _cached_baseline_acc


def _run_startup_baseline_evaluation() -> None:
    try:
        _get_baseline_acc()
    except Exception:
        log.exception("Startup baseline accuracy evaluation failed")


def _start_baseline_evaluation() -> None:
    global _baseline_startup_started, _baseline_startup_thread

    with _baseline_start_lock:
        if _baseline_startup_started:
            return

        _baseline_startup_started = True
        _set_baseline_evaluation_status(
            "completed" if _cached_baseline_acc is not None else "running"
        )
        _baseline_startup_thread = threading.Thread(
            target=_run_startup_baseline_evaluation,
            name="baseline-accuracy-evaluation",
            daemon=True,
        )
        try:
            _baseline_startup_thread.start()
        except Exception as exc:
            _set_baseline_evaluation_status("failed", str(exc))
            log.exception("Could not start baseline accuracy evaluation thread")


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
        if not _ensure_autoencoders_loaded():
            raise HTTPException(
                status_code=503,
                detail="Autoencoder checkpoints are unavailable.",
            )

        model = ae_heal_fc1_fc2(
            model, _ae_fc1, _ae_fc2, grad_faults, device
        )

    elif method == "hybrid":
        if not _ensure_autoencoders_loaded():
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

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_baseline_evaluation()
    yield


app = FastAPI(
    title="SelfHealingNN API",
    description="Self-Healing Neural Network demonstration using CIFAR-10",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
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

    Does not download CIFAR-10. Baseline accuracy is evaluated in the
    background after startup.
    """
    with _baseline_status_lock:
        baseline_status = _baseline_evaluation_status
        baseline_error = _baseline_evaluation_error

    return {
        "status": "ok",
        "device": str(device),
        "baseline_checkpoint": os.path.isfile(MODEL_PATH),
        "baseline_accuracy_status": baseline_status,
        "baseline_accuracy": (
            round(_cached_baseline_acc, 2)
            if _cached_baseline_acc is not None
            else None
        ),
        "baseline_accuracy_error": baseline_error,
        "autoencoders_available": _ae_available,
        "autoencoders_loaded": _ae_loaded,
        "data_loaders_initialized": (
            _train_loader is not None and _val_loader is not None
        ),
        "training_loader_initialized": _train_loader is not None,
        "validation_loader_initialized": _val_loader is not None,
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
        "autoencoders_loaded": _ae_loaded,
    }



@app.post("/api/baseline")
async def run_baseline(image: UploadFile = File(...)):
    request_started = time.perf_counter()
    log.info("Baseline request received")

    # 1. Read the uploaded image
    read_started = time.perf_counter()
    file_bytes = await image.read()

    log.info(
        "Baseline image received (%d bytes) in %.2fs",
        len(file_bytes),
        time.perf_counter() - read_started,
    )

    # 2. Decode the image
    decode_started = time.perf_counter()
    image_tensor = _decode_image(file_bytes)

    log.info(
        "Baseline image decoding completed in %.2fs",
        time.perf_counter() - decode_started,
    )

    # 3. Predict using the pretrained baseline CNN
    prediction_started = time.perf_counter()
    label, confidence = _predict(_baseline_cnn, image_tensor)

    log.info(
        "Baseline image prediction completed in %.2fs",
        time.perf_counter() - prediction_started,
    )

    # 4. Return cached accuracy only.
    # Do not evaluate the dataset during image prediction.
    baseline_accuracy = _cached_baseline_acc

    # 5. Log total request time
    log.info(
        "Baseline request completed in %.2fs",
        time.perf_counter() - request_started,
    )

    # 6. Return the prediction and available accuracy
    return {
        "accuracy": (
            round(baseline_accuracy, 2)
            if baseline_accuracy is not None
            else None
        ),
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
    _, val_loader = _get_data_loaders(include_train=False)

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

        baseline_acc = await asyncio.to_thread(_get_baseline_acc)
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
    message = str(exc).lower()
    memory_exhausted = isinstance(exc, MemoryError) or (
        isinstance(exc, RuntimeError)
        and any(
            marker in message
            for marker in (
                "out of memory",
                "cannot allocate memory",
                "can't allocate memory",
                "failed to allocate",
            )
        )
    )

    if memory_exhausted:
        log.critical(
            "Memory exhaustion while handling %s %s; "
            "the deployment may need more memory.",
            request.method,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "The request exceeded available backend memory. "
                    "Retry on an instance with more memory."
                )
            },
        )

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
