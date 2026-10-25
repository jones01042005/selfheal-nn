import torch
import torch.nn as nn

# Targeted Gradient-Based Healing
def targeted_heal_topk_gradient(
    model,
    gradient_faults,
    gradient_values,
    topk_ratio=0.1,
    std=0.01,
    heal_conv=False
):

    healed = 0

    with torch.no_grad():
        for name, module in model.named_modules():

            if heal_conv:
                valid = isinstance(module, (nn.Linear, nn.Conv2d))
            else:
                valid = isinstance(module, nn.Linear)

            if not valid:
                continue

            w_name = f"{name}.weight"
            if w_name not in gradient_faults:
                continue

            mask = gradient_faults[w_name]
            grads = gradient_values[w_name]

            faulty_grads = grads[mask]
            if faulty_grads.numel() == 0:
                continue

            k = max(1, int(topk_ratio * faulty_grads.numel()))

            threshold = torch.topk(
                faulty_grads.abs(), k, largest=False
            ).values.max()

            topk_mask = (grads.abs() <= threshold) & mask

            module.weight.data[topk_mask] = (
                torch.randn_like(module.weight.data[topk_mask]) * std
            )

            healed += topk_mask.sum().item()

    print(f"Healed weights: {healed}")
    return model


# Micro Fine-Tuning (FC only) - Optimized for fast healing
def micro_finetune_fc(model, train_loader, device, epochs=3, lr=1e-4, max_batches=20):

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Enable gradients only for FC layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for p in module.parameters():
                p.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    loss_fn = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
        batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            if i >= 10: break  # THIS IS THE SPEED TRIGGER

            x, y = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = loss_fn(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)

        print(f"Micro-FT Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    return model


# AE Healing (Phase-2 Ready)
def ae_heal_fc(model, ae_model, faulty_mask=None, device="cpu"):

    model.eval()

    with torch.no_grad():

        fc_weights = model.fc2.weight.data.to(device)

        reconstructed = ae_model(fc_weights)

        if faulty_mask is None:
            # Replace entire FC layer
            model.fc2.weight.data = reconstructed.clone()
        else:
            # Replace only faulty weights
            model.fc2.weight.data[faulty_mask] = reconstructed[faulty_mask]

    print("AE-based reconstruction applied")
    return model

def ae_heal_fc1_fc2(model, ae_fc1, ae_fc2, grad_faults, device):

    model.eval()

    with torch.no_grad():

        # FC1 
        fc1_weights = model.fc1.weight.data.to(device)
        reconstructed_fc1 = ae_fc1(fc1_weights)

        if "fc1.weight" in grad_faults:
            mask1 = grad_faults["fc1.weight"]
            fc1_weights[mask1] = reconstructed_fc1[mask1]
        else:
            fc1_weights = reconstructed_fc1

        model.fc1.weight.data = fc1_weights.clone()

        # FC2 
        fc2_weights = model.fc2.weight.data.to(device)
        reconstructed_fc2 = ae_fc2(fc2_weights)

        if "fc2.weight" in grad_faults:
            mask2 = grad_faults["fc2.weight"]
            fc2_weights[mask2] = reconstructed_fc2[mask2]
        else:
            fc2_weights = reconstructed_fc2

        model.fc2.weight.data = fc2_weights.clone()

    print("AE reconstruction applied to FC1 + FC2")

    return model


# Hybrid Healing (AE + Gradient)
def hybrid_heal(
    model,
    ae_fc1,
    ae_fc2,
    grad_faults,
    grad_vals,
    device,
    topk_ratio=0.05
):
    print("Step 1: AE Reconstruction")
    model = ae_heal_fc1_fc2(
        model,
        ae_fc1,
        ae_fc2,
        grad_faults,
        device
    )

    print("Step 2: Gradient Fine Repair")
    model = targeted_heal_topk_gradient(
        model,
        grad_faults,
        grad_vals,
        topk_ratio=topk_ratio,
        heal_conv=False
    )

    return model

# Neuron Regrowth (FC layers)
def neuron_regrowth(
    model,
    grad_faults,
    grad_vals,
    device,
    neuron_threshold_ratio=0.9,
    std=0.02
):

    regrown = 0

    with torch.no_grad():

        for layer_name in ["fc1", "fc2"]:

            layer = getattr(model, layer_name)

            weight_name = f"{layer_name}.weight"

            if weight_name not in grad_faults:
                continue

            mask = grad_faults[weight_name]   # boolean mask
            weights = layer.weight.data

            # mask shape = same as weight matrix
            # Each row corresponds to one neuron
            neuron_fault_ratio = mask.float().mean(dim=1)

            dead_neurons = neuron_fault_ratio > neuron_threshold_ratio

            for i in range(len(dead_neurons)):
                if dead_neurons[i]:

                    # Reinitialize full neuron
                    layer.weight.data[i] = torch.randn_like(
                        layer.weight.data[i]
                    ) * std

                    if layer.bias is not None:
                        layer.bias.data[i] = 0.0

                    regrown += 1

    print(f"Regrown neurons: {regrown}")

    return model

# Regrowth + Gradient Hybrid
def regrowth_plus_gradient(
    model,
    grad_faults,
    grad_vals,
    device,
    neuron_threshold_ratio=0.9,
    topk_ratio=0.05
):

    print("Step 1: Neuron Regrowth")

    model = neuron_regrowth(
        model,
        grad_faults,
        grad_vals,
        device,
        neuron_threshold_ratio=neuron_threshold_ratio
    )

    print("Step 2: Gradient Fine Repair")

    model = targeted_heal_topk_gradient(
        model,
        grad_faults,
        grad_vals,
        topk_ratio=topk_ratio,
        heal_conv=False
    )

    return model

# Dynamic Rerouting (Functional Plasticity)
def dynamic_rerouting(
    model,
    grad_faults,
    device,
    suppression_strength=0.5
):
    """
    Scale neuron outputs based on damage level.

    Neurons with high fault ratio are suppressed.
    Healthy neurons are preserved.
    """

    print("Applying Dynamic Rerouting")

    with torch.no_grad():

        for layer_name in ["fc1", "fc2"]:

            layer = getattr(model, layer_name)
            weight_name = f"{layer_name}.weight"

            if weight_name not in grad_faults:
                continue

            mask = grad_faults[weight_name]

            # Compute fault ratio per neuron
            neuron_fault_ratio = mask.float().mean(dim=1)

            # Healthy neurons -> scale ≈ 1
            # Damaged neurons -> scale down
            scaling = 1.0 - suppression_strength * neuron_fault_ratio

            # Apply scaling to outgoing weights
            layer.weight.data *= scaling.unsqueeze(1)

    return model
