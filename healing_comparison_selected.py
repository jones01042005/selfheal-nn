import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from train import CNN, get_train_loader, get_val_loader
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import (
    targeted_heal_topk_gradient,
    neuron_regrowth,
    dynamic_rerouting,
    micro_finetune_fc
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pth")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


# -----------------------------
# Main Experiment
# -----------------------------
def run_selected_healing_experiment():

    fault_ratios = [0.1, 0.2, 0.3, 0.4]

    results = {
        "gradient": [],
        "regrowth": [],
        "rerouting": []
    }

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    loss_fn = nn.CrossEntropyLoss()

    for fault_ratio in fault_ratios:

        print(f"\n--- Fault: {int(fault_ratio*100)}% ---")

        # Load fresh model
        base_model = CNN().to(device)
        base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Inject fault
        faulty_model = inject_weight_fault(base_model, fault_ratio)

        # Detect faults
        grad_faults, grad_vals = detect_low_gradient_weights(
            faulty_model,
            train_loader,
            loss_fn,
            device
        )

        # -----------------------------
        # 1. Gradient Healing
        # -----------------------------
        model = CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = inject_weight_fault(model, fault_ratio)

        gf, gv = detect_low_gradient_weights(model, train_loader, loss_fn, device)

        model = targeted_heal_topk_gradient(
            model, gf, gv,
            topk_ratio=0.1,
            heal_conv=False
        )

        model = micro_finetune_fc(model, train_loader, device)
        acc = evaluate(model, val_loader)

        results["gradient"].append(acc)
        print(f"Gradient: {acc:.2f}%")

        # -----------------------------
        # 2. Neuron Regrowth
        # -----------------------------
        model = CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = inject_weight_fault(model, fault_ratio)

        gf, gv = detect_low_gradient_weights(model, train_loader, loss_fn, device)

        model = neuron_regrowth(
            model, gf, gv, device
        )

        model = micro_finetune_fc(model, train_loader, device)
        acc = evaluate(model, val_loader)

        results["regrowth"].append(acc)
        print(f"Regrowth: {acc:.2f}%")

        # -----------------------------
        # 3. Dynamic Rerouting
        # -----------------------------
        model = CNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model = inject_weight_fault(model, fault_ratio)

        gf, gv = detect_low_gradient_weights(model, train_loader, loss_fn, device)

        model = dynamic_rerouting(
            model, gf, device,
            suppression_strength=0.5
        )

        model = micro_finetune_fc(model, train_loader, device)
        acc = evaluate(model, val_loader)

        results["rerouting"].append(acc)
        print(f"Rerouting: {acc:.2f}%")

    # -----------------------------
    # Plot Graph
    # -----------------------------
    fault_percent = [10, 20, 30, 40]

    plt.figure()

    plt.plot(fault_percent, results["gradient"], marker="o", label="Gradient")
    plt.plot(fault_percent, results["regrowth"], marker="o", label="Neuron Regrowth")
    plt.plot(fault_percent, results["rerouting"], marker="o", label="Dynamic Rerouting")

    plt.xlabel("Fault Percentage (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison of Healing Strategies across Fault Levels")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(EXPERIMENTS_DIR, "healing_comparison_selected.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n📊 Graph saved at: {save_path}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_selected_healing_experiment()