import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -------------------------------------------------
# Project root
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

from train import CNN, get_train_loader, get_val_loader
from fault import detect_low_gradient_weights
from heal import targeted_heal_topk_gradient, micro_finetune_fc

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Accuracy evaluation
# -------------------------------------------------
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# -------------------------------------------------
# Load data
# -------------------------------------------------
train_loader = get_train_loader()
val_loader = get_val_loader()
loss_fn = nn.CrossEntropyLoss()

# -------------------------------------------------
# Load baseline and faulty models
# -------------------------------------------------
baseline = CNN().to(device)
baseline.load_state_dict(torch.load(
    os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pth"),
    map_location=device
))

faulty = CNN().to(device)
faulty.load_state_dict(torch.load(
    os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
    map_location=device
))

baseline_acc = evaluate_accuracy(baseline, val_loader)
faulty_acc = evaluate_accuracy(faulty, val_loader)

print(f"✅ Baseline Accuracy : {baseline_acc:.2f}%")
print(f"❌ Faulty Accuracy   : {faulty_acc:.2f}%")

# =================================================
# MODE SELECTION
# =================================================
# Choose one:
# "single"   -> FC-only healing evaluation
# "ablation" -> No healing vs FC vs FC+Conv
# "efficiency" -> Top-K efficiency study

MODE = "efficiency"   # 🔴 CHANGE THIS

# =================================================
# SINGLE HEALING PIPELINE
# =================================================
if MODE == "single":
    print("\n▶ Running SINGLE healing pipeline (FC only)")

    model = CNN().to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
        map_location=device
    ))

    grad_faults, grad_vals = detect_low_gradient_weights(
        model, train_loader, loss_fn, device
    )

    model = targeted_heal_topk_gradient(
        model,
        grad_faults,
        grad_vals,
        topk_ratio=0.1,
        heal_conv=False
    )

    model = micro_finetune_fc(
        model,
        train_loader,
        device,
        epochs=2,
        lr=1e-4
    )

    healed_acc = evaluate_accuracy(model, val_loader)
    print(f"🩹 Healed + Micro-FT Accuracy: {healed_acc:.2f}%")

# =================================================
# ABLATION STUDY
# =================================================
elif MODE == "ablation":
    print("\n▶ Running ABLATION study")

    results = {}

    # ---- No Healing ----
    model = CNN().to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
        map_location=device
    ))
    results["No Healing"] = evaluate_accuracy(model, val_loader)

    # ---- FC Only ----
    model = CNN().to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
        map_location=device
    ))

    grad_faults, grad_vals = detect_low_gradient_weights(
        model, train_loader, loss_fn, device
    )

    model = targeted_heal_topk_gradient(
        model, grad_faults, grad_vals, topk_ratio=0.1, heal_conv=False
    )

    model = micro_finetune_fc(model, train_loader, device)
    results["FC Only"] = evaluate_accuracy(model, val_loader)

    # ---- FC + Conv ----
    model = CNN().to(device)
    model.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
        map_location=device
    ))

    grad_faults, grad_vals = detect_low_gradient_weights(
        model, train_loader, loss_fn, device
    )

    model = targeted_heal_topk_gradient(
        model, grad_faults, grad_vals, topk_ratio=0.1, heal_conv=True
    )

    model = micro_finetune_fc(model, train_loader, device)
    results["FC + Conv"] = evaluate_accuracy(model, val_loader)

    print("\n📊 Ablation Results")
    for k, v in results.items():
        print(f"{k:<15}: {v:.2f}%")

# =================================================
# EFFICIENCY STUDY
# =================================================
elif MODE == "efficiency":
    print("\n▶ Running EFFICIENCY study")

    ratios = [0.01, 0.05, 0.10, 0.20, 1.0]
    accuracies = []

    for r in ratios:
        model = CNN().to(device)
        model.load_state_dict(torch.load(
            os.path.join(PROJECT_ROOT, "models", "faulty_cnn.pth"),
            map_location=device
        ))

        grad_faults, grad_vals = detect_low_gradient_weights(
            model, train_loader, loss_fn, device
        )

        model = targeted_heal_topk_gradient(
            model, grad_faults, grad_vals, topk_ratio=r, heal_conv=False
        )

        model = micro_finetune_fc(
            model, train_loader, device, epochs=2
        )

        acc = evaluate_accuracy(model, val_loader)
        accuracies.append(acc)

        print(f"Top-{int(r*100)}% healed → Accuracy = {acc:.2f}%")

    plt.figure()
    plt.plot([r * 100 for r in ratios], accuracies, marker="o")
    plt.xlabel("% Weights Healed")
    plt.ylabel("Accuracy (%)")
    plt.title("Efficiency Study: Healing vs Accuracy")
    plt.grid(True)
    save_path = os.path.join(EXPERIMENTS_DIR, "efficiency_study.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

else:
    raise ValueError("Invalid MODE selected")
