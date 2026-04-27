import os
import torch
import torch.nn as nn
from train import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------------------------
# Fault Injection
# -------------------------------------------------
def inject_weight_fault(model, fault_ratio=0.2):

    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" not in name:
                continue
            num_weights = p.numel()
            num_faults = int(fault_ratio * num_weights)
            idx = torch.randperm(num_weights, device=p.device)[:num_faults]
            p.view(-1)[idx] = 0

    return model


# -------------------------------------------------
# Gradient Detection
# -------------------------------------------------
def detect_low_gradient_weights(
    model,
    data_loader,
    loss_fn,
    device,
    grad_threshold=1e-5,
    max_batches=5
):

    grad_sum, grad_count = {}, {}

    model.train()

    for i, (x, y) in enumerate(data_loader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if name not in grad_sum:
                grad_sum[name] = torch.zeros_like(p.grad)
                grad_count[name] = 0
            grad_sum[name] += p.grad.abs()
            grad_count[name] += 1

    gradient_faults, gradient_values = {}, {}

    for name in grad_sum:
        avg_grad = grad_sum[name] / grad_count[name]
        gradient_values[name] = avg_grad
        gradient_faults[name] = avg_grad < grad_threshold

    return gradient_faults, gradient_values

def print_fault_statistics(gradient_faults):

    print("\nFault Statistics Per Layer")

    for name, mask in gradient_faults.items():

        total = mask.numel()
        faulty = mask.sum().item()
        percent = 100 * faulty / total

        print(f"{name:<20} : {faulty}/{total} ({percent:.2f}%)")

def detect_dead_neurons(gradient_faults):

    print("\nDead Neuron Analysis")

    for name, mask in gradient_faults.items():

        if "weight" not in name:
            continue

        neuron_ratio = mask.float().mean(dim=1)

        dead = (neuron_ratio > 0.9).sum().item()

        print(f"{name} dead neurons: {dead}")


# -------------------------------------------------
# Only run if executed directly
# -------------------------------------------------
if __name__ == "__main__":

    model = CNN().to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "baseline_cnn.pth"),
        map_location=device
    ))

    print("✅ Baseline model loaded")

    model = inject_weight_fault(model, fault_ratio=0.2)

    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, "faulty_cnn.pth"))

    print("⚠️ Fault injected and saved")
