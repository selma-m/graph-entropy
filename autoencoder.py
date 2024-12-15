import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the Autoencoder Class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, latent_dim = 32):
        super(Autoencoder, self).__init__()
        # Encoder

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),         # Input -> Hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),        # Hidden -> Latent
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),        # Latent -> Hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),         # Hidden -> Output
            nn.Sigmoid()                              # Output layer (for 0-1 range)
        )

    def forward(self, x):
        z = self.encoder(x)  # Encoding step
        reconstructed = self.decoder(z)  # Decoding step
        return reconstructed

