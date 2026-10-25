import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True) # Ensure data dir exists
os.makedirs(os.path.join(BASE_DIR, "experiments"), exist_ok=True)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Model (Improved CNN for CIFAR-10)
# -------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: Increased filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 2: 
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 3:
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers (PRESERVED NAMES)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.4) # Slightly lower dropout helps convergence
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------------------------------
# Data Loaders (Optimized for Edge Simulation)
# -------------------------------------------------
def get_train_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # Changed download to True to ensure first-time users don't crash
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True, 
        transform=transform
    )

    return DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


def get_val_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            # Move both input and labels to the same device as the model
            x = x.to(device)
            y = y.to(device) 
            
            out = model(x)
            _, pred = torch.max(out, 1)
            
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total if total > 0 else 0


# -------------------------------------------------
# Main Training Logic
# -------------------------------------------------
if __name__ == "__main__":
    print(f"Using device: {device}")

    model = CNN().to(device)
    trainloader = get_train_loader()
    testloader = get_val_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 30
    best_acc = 0

    print("🚀 Starting Training...")

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        
        val_acc = evaluate(model, testloader)
        avg_loss = running_loss / len(trainloader)
        
        loss_history.append(avg_loss)
        acc_history.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODEL_DIR, "baseline_cnn.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐️ New Best Accuracy! Model saved to {save_path}")

    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.2f}%")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, epochs+1), loss_history, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy (%)', color=color)  
    ax2.plot(range(1, epochs+1), acc_history, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Training Curves: Loss and Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(BASE_DIR, "experiments", "training_plot.png"))
    plt.close()
    
    print("📈 Saved training curves to experiments/training_plot.png")