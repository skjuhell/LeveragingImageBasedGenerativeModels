import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


torch.manual_seed(42)

class MinMaxScaler:
    def __init__(self, feature_range=(0.0, 1.0)):
        self.min = None
        self.max = None
        self.scale_min, self.scale_max = feature_range

    def fit(self, data: torch.Tensor):
        """Compute the min and max for scaling."""
        self.min = data.min(dim=0, keepdim=True).values
        self.max = data.max(dim=0, keepdim=True).values

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Scale data to the feature range."""
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        scale = (self.scale_max - self.scale_min) / (self.max - self.min + 1e-8)
        return self.scale_min + (data - self.min) * scale

    def inverse_transform(self, scaled_data: torch.Tensor) -> torch.Tensor:
        """Descale data back to original range."""
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        scale = (self.max - self.min + 1e-8) / (self.scale_max - self.scale_min)
        return self.min + (scaled_data - self.scale_min) * scale


class DenoiseMLP(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x, t):
        t = t.unsqueeze(1).float() / 1000
        x = torch.cat([x, t], dim=1)
        return self.net(x)

# ---------------------------
# Diffusion Model Class
# ---------------------------
class DiffusionModel:
    def __init__(self, feature_dim, T=1000, device=None):
        self.feature_dim = feature_dim
        self.T = T
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DenoiseMLP(feature_dim).to(self.device)
        self.betas = torch.linspace(1e-4, 0.02, T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def train(self, data, epochs=1000, lr=1e-3):
        data = data.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            t = torch.randint(0, self.T, (data.size(0),), device=self.device)
            noise = torch.randn_like(data)
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).unsqueeze(1)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).unsqueeze(1)
            noisy_data = sqrt_alpha_hat * data + sqrt_one_minus_alpha_hat * noise

            pred_noise = self.model(noisy_data, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    @torch.no_grad()
    def sample(self, n_samples):
        x = torch.randn(n_samples, self.feature_dim).to(self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            z = torch.randn_like(x) if t > 0 else 0
            alpha = self.alphas[t]
            alpha_hat_t = self.alpha_hat[t]
            beta = self.betas[t]

            pred_noise = self.model(x, t_batch)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) + torch.sqrt(beta) * z

        return x.cpu()


def diffusion(ori_sequences: torch.Tensor,args) -> torch.Tensor:
    """
    Trains a diffusion model on the input sequences and returns synthetic samples.

    Args:
        sequences (torch.Tensor): Tensor of shape (n_samples, feature_dim)
        epochs (int): Training epochs
        T (int): Diffusion steps
        lr (float): Learning rate

    Returns:
        torch.Tensor: Synthetic samples of shape (n_samples, feature_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert input to tensor
    if isinstance(ori_sequences, list):
        sequences = torch.tensor(ori_sequences, dtype=torch.float32)
    elif isinstance(ori_sequences, np.ndarray):
        sequences = torch.from_numpy(ori_sequences).float()
    elif not isinstance(ori_sequences, torch.Tensor):
        raise TypeError("Input 'sequences' must be a torch.Tensor, list, or np.ndarray.")

    if sequences.ndim != 2:
        sequences = sequences.squeeze()
        if sequences.ndim != 2:
            raise ValueError(f"Expected 2D input of shape (n_samples, feature_dim), got shape {sequences.shape}")

    sequences = sequences.to(device)

    scaler = MinMaxScaler()
    scaler.fit(sequences)
    sequences = scaler.transform(sequences)



    n_samples, feature_dim = np.array(sequences).shape

    # Schedule
    betas = torch.linspace(1e-4, 0.02, n_samples).to(device)

    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    # Model
    model = DenoiseMLP(feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(4000):
        t = torch.randint(0, n_samples, (n_samples,), device=device)
        noise = torch.randn_like(sequences)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).unsqueeze(1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t]).unsqueeze(1)
        noisy_data = sqrt_alpha_hat * sequences + sqrt_one_minus_alpha_hat * noise

        pred_noise = model(noisy_data, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    # Sampling
    @torch.no_grad()
    def sample():
        x = torch.randn(n_samples, feature_dim).to(device)
        for t in reversed(range(n_samples)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            z = torch.randn_like(x) if t > 0 else 0
            alpha = alphas[t]
            alpha_hat_t = alpha_hat[t]
            beta = betas[t]

            pred_noise = model(x, t_batch)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * pred_noise) + torch.sqrt(beta) * z
        return x.cpu()
    samples = sample()
    samples = scaler.inverse_transform(samples.squeeze()).numpy()
    sequences = scaler.inverse_transform(sequences).numpy()

    return samples, sequences