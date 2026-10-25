import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from train import CNN, get_train_loader, get_val_loader
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import (
    targeted_heal_topk_gradient,
    micro_finetune_fc,
    ae_heal_fc1_fc2,
    hybrid_heal,
    neuron_regrowth,
    regrowth_plus_gradient,
    dynamic_rerouting
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pth")

def apply_static_quantization(model, loader):

    model.eval()
    model.cpu()

    model.conv1 = torch.quantization.fuse_modules(model.conv1, [['0','1','2'], ['3','4','5']])
    model.conv2 = torch.quantization.fuse_modules(model.conv2, [['0','1','2'], ['3','4','5']])
    model.conv3 = torch.quantization.fuse_modules(model.conv3, [['0','1','2']])

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    torch.quantization.prepare(model, inplace=True)

    # calibration
    with torch.no_grad():
        for i,(x,_) in enumerate(loader):
            model(x)
            if i > 50:
                break

    torch.quantization.convert(model, inplace=True)

    return model

# -------------------------------------------------
# Edge Inference Metric (NEW)
# -------------------------------------------------
def measure_edge_inference_time(model, loader, num_samples=100):
    """Simulates inference latency on an edge CPU environment."""
    model.eval()
    cpu_model = model.to('cpu')
    x, _ = next(iter(loader))
    x = x[:1].to('cpu') # Single image inference
    
    # Warmup
    for _ in range(10):
        _ = cpu_model(x)
        
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = cpu_model(x)
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / num_samples) * 1000
    model.to(device) # Move back to original device
    return avg_latency

# Evaluation
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


# -------------------------------------------------
# Dead Neuron Detection (ADDED)
# -------------------------------------------------
def detect_dead_neurons(gradient_faults):

    print("\nDead Neuron Analysis")

    for name, mask in gradient_faults.items():

        if "weight" not in name:
            continue

        if mask.dim() < 2:
            continue

        neuron_ratio = mask.float().mean(dim=1)

        dead = (neuron_ratio > 0.9).sum().item()

        print(f"{name} dead neurons: {dead}")

# -------------------------------------------------
# Layer-wise Dead Weight and Neuron Report
# -------------------------------------------------
def report_dead_weights_and_neurons(model, gradient_faults):

    print("\n===== Layer Health Report =====")

    for name, param in model.named_parameters():

        if name not in gradient_faults:
            continue

        mask = gradient_faults[name]

        total_weights = mask.numel()
        dead_weights = mask.sum().item()

        print(f"\nLayer: {name}")
        print(f"Total Weights : {total_weights}")
        print(f"Dead Weights  : {dead_weights}")

        # ----- Fully Connected Layers -----
        if mask.dim() == 2:

            neuron_ratio = mask.float().mean(dim=1)
            total_neurons = neuron_ratio.shape[0]
            dead_neurons = (neuron_ratio > 0.9).sum().item()

            print(f"Total Neurons : {total_neurons}")
            print(f"Dead Neurons  : {dead_neurons}")

        # ----- Convolution Layers -----
        elif mask.dim() == 4:

            neuron_ratio = mask.float().mean(dim=(1,2,3))
            total_neurons = neuron_ratio.shape[0]
            dead_neurons = (neuron_ratio > 0.9).sum().item()

            print(f"Total Neurons : {total_neurons}")
            print(f"Dead Neurons  : {dead_neurons}")

# Adaptive Healing Controller (UPDATED WITH DATA-DRIVEN LOGIC)
def adaptive_healing(model, fault_ratio, grad_faults, grad_vals):

    print(f"\nUsing Adaptive Healing Controller (Fault Ratio: {int(fault_ratio*100)}%)")

    damaged_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w_name = f"{name}.weight"
            if w_name in grad_faults:
                damaged_weights += grad_faults[w_name].sum().item()
    print(f"Damaged weights: {damaged_weights}")

    # Thresholds updated based on your experimental results for maximum accuracy
    if fault_ratio <= 0.15:
        print("Selected Strategy: Regrowth + Gradient (Best for <15% Fault)")
        model = regrowth_plus_gradient(
            model, grad_faults, grad_vals, device
        )

    elif fault_ratio <= 0.25:
        print("Selected Strategy: Neuron Regrowth (Best for 20% Fault)")
        model = neuron_regrowth(
            model, grad_faults, grad_vals, device
        )

    elif fault_ratio <= 0.35:
        print("Selected Strategy: Dynamic Rerouting (Best for 30% Fault)")
        model = dynamic_rerouting(
            model, grad_faults,
            device, suppression_strength=0.5
        )

    else:
        print("Selected Strategy: Gradient Fine Repair (Best for >35% Fault)")
        model = targeted_heal_topk_gradient(
            model, grad_faults, grad_vals,
            topk_ratio=0.15, heal_conv=True
        )

    return model



def run_full_experiment():
    fault_ratios = [0.1, 0.2, 0.3, 0.4]
    methods = ["gradient", "ae_dual", "hybrid", "regrowth", "regrowth_gradient", "rerouting", "adaptive"]
    results = {}
    recovery_results = {}
    train_loader = get_train_loader()
    val_loader = get_val_loader()

    baseline_model = CNN().to(device)
    baseline_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    baseline_acc = evaluate(baseline_model, val_loader)

    print(f"\nBaseline Accuracy: {baseline_acc:.2f}%\n")
    
    base_latency = measure_edge_inference_time(baseline_model, val_loader)
    print(f"Baseline Edge Latency: {base_latency:.2f} ms")

    ae_fc1, ae_fc2 = None, None
    if any(m in methods for m in ["ae_dual", "hybrid"]):
        from train_ae import train_autoencoders
        ae_fc1, ae_fc2 = train_autoencoders(baseline_model, device, force_train=False)

    for method in methods:
        print(f"\nRunning Method: {method}")
        method_accuracies = []
        method_recoveries = []
        for fault_ratio in fault_ratios:
            model = CNN().to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model = inject_weight_fault(model, fault_ratio)
            faulty_acc = evaluate(model, val_loader)
            grad_faults, grad_vals = detect_low_gradient_weights(model, train_loader, nn.CrossEntropyLoss(), device)
            start_time = time.time()
            if method == "gradient": model = targeted_heal_topk_gradient(model, grad_faults, grad_vals, topk_ratio=0.1, heal_conv=False)
            elif method == "ae_dual": model = ae_heal_fc1_fc2(model, ae_fc1, ae_fc2, grad_faults, device)
            elif method == "hybrid": model = hybrid_heal(model, ae_fc1, ae_fc2, grad_faults, grad_vals, device, topk_ratio=0.05)
            elif method == "regrowth": model = neuron_regrowth(model, grad_faults, grad_vals, device)
            elif method == "regrowth_gradient": model = regrowth_plus_gradient(model, grad_faults, grad_vals, device)
            elif method == "rerouting": model = dynamic_rerouting(model, grad_faults, device, suppression_strength=0.5)
            elif method == "adaptive": model = adaptive_healing(model, fault_ratio, grad_faults, grad_vals)
            
            model = micro_finetune_fc(
            model,
            train_loader,
            device,
            epochs=4,
            max_batches=25
        )
            healed_acc = evaluate(model, val_loader)
            recovery = (healed_acc - faulty_acc) / (baseline_acc - faulty_acc) if baseline_acc != faulty_acc else 0
            method_accuracies.append(healed_acc)
            method_recoveries.append(recovery)
        results[method] = method_accuracies
        recovery_results[method] = method_recoveries

    df = pd.DataFrame(results, index=[10, 20, 30, 40])
    df.to_csv("experiments/healing_results.csv")
    print(df)

# Adaptive Only Pipeline
def run_adaptive_only(fault_ratio=0.1):
    print(f"\n========== ADAPTIVE HEALING | Fault = {int(fault_ratio*100)}% ==========\n")
    train_loader = get_train_loader()
    val_loader = get_val_loader()

    model = CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    baseline_acc = evaluate(model, val_loader)
    print(f"Baseline Accuracy : {baseline_acc:.2f}%")
    lat_before = measure_edge_inference_time(model, val_loader)
    print(f"Baseline Latency  : {lat_before:.2f} ms")

    model = inject_weight_fault(model, fault_ratio)
    faulty_acc = evaluate(model, val_loader)
    print(f"After Fault       : {faulty_acc:.2f}%")

    grad_faults, grad_vals = detect_low_gradient_weights(model, train_loader, nn.CrossEntropyLoss(), device)
    report_dead_weights_and_neurons(model, grad_faults)
    
    start_time = time.time()
    model = adaptive_healing(model, fault_ratio, grad_faults, grad_vals)
    print(f"\nHealing Time: {time.time() - start_time:.4f} sec")

    model = micro_finetune_fc(
    model,
    train_loader,
    device,
    epochs=4,
    max_batches=25
)
    healed_acc = evaluate(model, val_loader)
    print(f"\nAfter Healing     : {healed_acc:.2f}%")

    lat_after = measure_edge_inference_time(model, val_loader)
    print(f"Healed Latency    : {lat_after:.2f} ms")

    # Quantization (optional for deployment, not for evaluation)
    try:
        quant_model = apply_static_quantization(model, train_loader)
        print("Quantized model created for edge deployment (INT8).")
    except Exception as e:
        print("Quantization skipped due to backend limitation:", e)

    recovery = ((healed_acc - faulty_acc) / (baseline_acc - faulty_acc))*100
    print(f"Recovery Efficiency: {recovery:.2f}%")

    recovery = ((healed_acc - faulty_acc) / (baseline_acc - faulty_acc))*100
    print(f"Recovery Efficiency: {recovery:.2f}%")

if __name__ == "__main__":
    run_adaptive_only(fault_ratio=0.1)
    #run_full_experiment()