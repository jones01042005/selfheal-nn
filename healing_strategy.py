import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

from train import CNN
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import (
    targeted_heal_topk_gradient,
    neuron_regrowth,
    dynamic_rerouting,
    micro_finetune_fc
)

# -------------------------------------------------
# Setup
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("experiments", exist_ok=True)

# -------------------------------------------------
# Dataset
# -------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822,0.4465),
        (0.2023,0.1994,0.2010)
    )
])

testset = torchvision.datasets.CIFAR10(
    root="./datasets",
    train=False,
    download=False,
    transform=transform
)

trainset = torchvision.datasets.CIFAR10(
    root="./datasets",
    train=True,
    download=False,
    transform=transform
)

testloader = DataLoader(testset, batch_size=128, shuffle=False)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, pred = torch.max(outputs, 1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


# -------------------------------------------------
# Load Baseline Model
# -------------------------------------------------

base_model = CNN().to(device)

base_model.load_state_dict(
    torch.load("models/baseline_cnn.pth", map_location=device)
)

base_model.eval()

baseline_acc = evaluate(base_model, testloader)

print(f"Baseline Accuracy: {baseline_acc:.2f}%")

# -------------------------------------------------
# Fault Levels
# -------------------------------------------------

fault_ratios = [0.1, 0.2, 0.3, 0.4]

gradient_results = []
regrowth_results = []
rerouting_results = []

# -------------------------------------------------
# Run Experiments
# -------------------------------------------------

for ratio in fault_ratios:

    print(f"\n===== Fault Ratio {ratio*100:.0f}% =====")

    # -----------------------
    # Inject Fault
    # -----------------------

    faulty_model = copy.deepcopy(base_model)

    inject_weight_fault(faulty_model, ratio)

    faulty_acc = evaluate(faulty_model, testloader)

    print(f"Faulty Accuracy: {faulty_acc:.2f}%")

    # Detect faults
    grad_faults, grad_vals = detect_low_gradient_weights(
        faulty_model,
        trainloader,
        nn.CrossEntropyLoss(),
        device,
        max_batches=5
    )

    # -------------------------------------------------
    # Gradient Healing
    # -------------------------------------------------

    model_grad = copy.deepcopy(faulty_model)

    model_grad = targeted_heal_topk_gradient(
        model_grad,
        grad_faults,
        grad_vals,
        topk_ratio=0.1
    )

    model_grad = micro_finetune_fc(
        model_grad,
        trainloader,
        device,
        epochs=2
    )

    acc_grad = evaluate(model_grad, testloader)

    gradient_results.append(acc_grad)

    print(f"Gradient Healing Accuracy: {acc_grad:.2f}%")

    # -------------------------------------------------
    # Neuron Regrowth
    # -------------------------------------------------

    model_reg = copy.deepcopy(faulty_model)

    model_reg = neuron_regrowth(
        model_reg,
        grad_faults,
        grad_vals,
        device
    )

    model_reg = micro_finetune_fc(
        model_reg,
        trainloader,
        device,
        epochs=2
    )

    acc_reg = evaluate(model_reg, testloader)

    regrowth_results.append(acc_reg)

    print(f"Regrowth Accuracy: {acc_reg:.2f}%")

    # -------------------------------------------------
    # Dynamic Rerouting
    # -------------------------------------------------

    model_reroute = copy.deepcopy(faulty_model)

    model_reroute = dynamic_rerouting(
        model_reroute,
        grad_faults,
        device
    )

    model_reroute = micro_finetune_fc(
        model_reroute,
        trainloader,
        device,
        epochs=2
    )

    acc_reroute = evaluate(model_reroute, testloader)

    rerouting_results.append(acc_reroute)

    print(f"Rerouting Accuracy: {acc_reroute:.2f}%")

# -------------------------------------------------
# Plot Results
# -------------------------------------------------

plt.figure(figsize=(8,5))

fault_percent = [r*100 for r in fault_ratios]

plt.plot(fault_percent, gradient_results, marker='o', label="Gradient Healing")
plt.plot(fault_percent, regrowth_results, marker='o', label="Neuron Regrowth")
plt.plot(fault_percent, rerouting_results, marker='o', label="Dynamic Rerouting")

plt.axhline(
    y=baseline_acc,
    color="red",
    linestyle="--",
    label="Baseline Accuracy"
)

plt.xlabel("Fault Percentage (%)")
plt.ylabel("Accuracy (%)")
plt.title("Healing Strategy Comparison")

plt.legend()
plt.grid(True)

plt.savefig("experiments/healing_strategy_comparison.png", dpi=300)

plt.show()

print("\nPlot saved to experiments/healing_strategy_comparison.png")