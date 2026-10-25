import os
import torch
import torch.nn as nn
import torch.optim as optim
from ae_model import WeightAutoencoder

# Create a folder to store pre-trained Autoencoders
AE_DIR = "models/autoencoders"
os.makedirs(AE_DIR, exist_ok=True)

def train_autoencoders(model, device, epochs=100, force_train=False):
    """
    On an Edge Device, we only LOAD. This function simulates that.
    """
    model.eval()
    fc1_path = os.path.join(AE_DIR, "ae_fc1.pth")
    fc2_path = os.path.join(AE_DIR, "ae_fc2.pth")

    fc1_weights = model.fc1.weight.data.clone().to(device)
    fc2_weights = model.fc2.weight.data.clone().to(device)

    ae_fc1 = WeightAutoencoder(input_dim=fc1_weights.shape[1]).to(device)
    ae_fc2 = WeightAutoencoder(input_dim=fc2_weights.shape[1]).to(device)

    # Check if we can skip training to save time
    if os.path.exists(fc1_path) and os.path.exists(fc2_path) and not force_train:
        print("📦 Loading Pre-trained Autoencoders (Simulating Edge Storage)...")
        ae_fc1.load_state_dict(torch.load(fc1_path, map_location=device))
        ae_fc2.load_state_dict(torch.load(fc2_path, map_location=device))
        return ae_fc1, ae_fc2

    print("🚀 Training Autoencoders (One-time Setup)...")
    criterion = nn.MSELoss()
    
    # Efficient training loop for both
    for name, ae, weights, path, opt in [
        ("FC1", ae_fc1, fc1_weights, fc1_path, optim.Adam(ae_fc1.parameters(), lr=1e-3)),
        ("FC2", ae_fc2, fc2_weights, fc2_path, optim.Adam(ae_fc2.parameters(), lr=1e-3))
    ]:
        for epoch in range(epochs):
            opt.zero_grad()
            output = ae(weights)
            loss = criterion(output, weights)
            loss.backward()
            opt.step()
        
        torch.save(ae.state_dict(), path)
        print(f"✅ {name} AE Trained and Saved.")

    return ae_fc1, ae_fc2

# Add this at the bottom of train_ae.py
if __name__ == "__main__":
    import os
    from train import CNN
    
    # Setup paths
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "baseline_cnn.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the baseline model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found! Run train.py first.")
    else:
        print("🕒 Initializing Autoencoder Training...")
        baseline_model = CNN().to(device)
        baseline_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # 2. Call the training function
        # We set force_train=True here to ensure it creates the files now
        ae1, ae2 = train_autoencoders(baseline_model, device, epochs=150)
        
        print("\n✅ Success! Autoencoders are trained and saved in models/autoencoders/")