import torch
import torch.nn as nn

class WeightAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_ratio=0.25):
        super().__init__()
        # Ensure latent_dim is at least 1
        latent_dim = max(1, int(input_dim * latent_ratio))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))