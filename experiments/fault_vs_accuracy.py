import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import copy

from train import CNN

# -------------------------------------------------
# Create experiments folder
# -------------------------------------------------

os.makedirs("experiments", exist_ok=True)

# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")

# -------------------------------------------------
# Dataset (same normalization as training)
# -------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

testset = torchvision.datasets.CIFAR10(
    root="./datasets",
    train=False,
    download=False,
    transform=transform
)

testloader = DataLoader(testset, batch_size=128, shuffle=False)

# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# -------------------------------------------------
# Fault Injection
# -------------------------------------------------

def inject_weight_fault(model, fault_ratio):

    with torch.no_grad():

        for param in model.parameters():

            num_weights = param.numel()
            num_faults = int(fault_ratio * num_weights)

            if num_faults > 0:

                idx = torch.randperm(num_weights)[:num_faults]

                param.view(-1)[idx] = 0


# -------------------------------------------------
# Load Baseline Model
# -------------------------------------------------

base_model = CNN().to(device)

base_model.load_state_dict(
    torch.load("models/baseline_cnn.pth", map_location=device)
)

base_model.eval()

# -------------------------------------------------
# Baseline Accuracy
# -------------------------------------------------

baseline_acc = evaluate(base_model, testloader)

print(f"\nBaseline Accuracy (0% fault): {baseline_acc:.2f}%")

# -------------------------------------------------
# Fault Ratios
# -------------------------------------------------

fault_ratios = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

fault_accuracies = []

# -------------------------------------------------
# Run Experiment
# -------------------------------------------------

for ratio in fault_ratios:

    faulty_model = copy.deepcopy(base_model)

    inject_weight_fault(faulty_model, ratio)

    acc = evaluate(faulty_model, testloader)

    fault_accuracies.append(acc)

    print(f"Fault Ratio: {int(ratio*100)}% → Accuracy: {acc:.2f}%")

# -------------------------------------------------
# Save Results Table
# -------------------------------------------------

# results = pd.DataFrame({
#     "Fault (%)": [r*100 for r in fault_ratios],
#     "Accuracy (%)": fault_accuracies
# })

# results.to_csv("experiments/fault_vs_accuracy.csv", index=False)

# print("\nResults saved to experiments/fault_vs_accuracy.csv")

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    [r*100 for r in fault_ratios],
    fault_accuracies,
    marker="o",
    linewidth=2,
    label="Faulty Model Accuracy"
)

plt.axhline(
    y=baseline_acc,
    color="red",
    linestyle="--",
    label="Baseline Accuracy"
)

plt.xlabel("Fault Ratio (%)")
plt.ylabel("Accuracy (%)")
plt.title("Impact of Internal Faults on CNN Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "experiments/fault_vs_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

print("Plot saved to experiments/fault_vs_accuracy.png")

plt.show()