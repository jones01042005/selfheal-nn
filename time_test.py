import time
import torch
import torch.nn as nn
from train import CNN, get_val_loader, get_train_loader
from fault import inject_weight_fault, detect_low_gradient_weights
from heal import micro_finetune_fc, neuron_regrowth
from main import evaluate, adaptive_healing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNN().to(device)
model.load_state_dict(torch.load("models/baseline_cnn.pth", map_location=device))

val_loader = get_val_loader()
train_loader = get_train_loader()

t0 = time.time()
base_acc = evaluate(model, val_loader)
t1 = time.time()
print(f"Base Eval time: {t1 - t0:.2f}s, Acc: {base_acc:.2f}%")

model = inject_weight_fault(model, 0.2)
t2 = time.time()
fault_acc = evaluate(model, val_loader)
t3 = time.time()
print(f"Fault Eval time: {t3 - t2:.2f}s, Acc: {fault_acc:.2f}%")

grad_faults, grad_vals = detect_low_gradient_weights(model, train_loader, nn.CrossEntropyLoss(), device, max_batches=2)
model = adaptive_healing(model, 0.2, grad_faults, grad_vals)
model = micro_finetune_fc(model, train_loader, device, epochs=2, max_batches=25)
t4 = time.time()
healed_acc = evaluate(model, val_loader)
t5 = time.time()
print(f"Heal & Eval time: {t5 - t3:.2f}s, Acc: {healed_acc:.2f}%")
