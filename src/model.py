import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, List, Callable
from src.diagnostics import prior_mismatch_score

# import matplotlib.pyplot as plt
# from tqdm import tqdm


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TVAEConfig:
    """Configuration for TVAE model."""

    input_dim: int  # Set by preprocessor output dimension
    latent_dim: int = 10
    encoder_hidden_dims: List[int] = None
    decoder_hidden_dims: List[int] = None
    output_info: List[Tuple[int, int, str]] = None

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 300

    # Loss weighting
    seed: int = 42
    beta: float = 1.0  # Weight for KL divergence term
    beta_start: float = 0.0  # KL weight at epoch 0 (warm-up start)
    beta_warmup_epochs: int = 20  # epochs to linearly ramp beta_start -> beta
    use_beta_warmup: bool = False  # if False, beta is fixed at `beta` for all epochs

    # Regularization
    dropout_rate: float = 0.2
    use_batch_norm: bool = True

    checkpoint_metric_fn: Optional[Callable] = (
        None  # defaults to prior_mismatch_score if None
    )

    # Device
    device: str = "cpu"

    def __post_init__(self):
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [self.input_dim * 2, self.input_dim]
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [self.input_dim, self.input_dim * 2]

        if self.beta_start < 0:
            raise ValueError("beta_start must be non-negative")
        if self.beta_warmup_epochs < 0:
            raise ValueError("beta_warmup_epochs must be non-negative")

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")


class Encoder(nn.Module):
    """Encoder network that maps input to latent distribution parameters."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Latent space: mu and log_var
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns mu and log_var of the latent distribution.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.hidden_layers(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network that reconstructs input from latent representation."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        # Build hidden layers
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer: reconstruct input
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructs input from latent representation.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Reconstructed data (batch_size, output_dim)
        """
        h = self.hidden_layers(z)
        x_recon = self.output_layer(h)
        return x_recon


class TVAE(nn.Module):
    """Tabular Variational Autoencoder for mixed-type data."""

    def __init__(self, config: TVAEConfig):
        super().__init__()
        self.config = config

        # Build encoder and decoder
        self.encoder = Encoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )

        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            output_dim=config.input_dim,
            hidden_dims=config.decoder_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )

        self.sigma = nn.Parameter(torch.ones(config.input_dim) * 0.1)

        # Reconstruction loss
        self.output_info = config.output_info or [(0, config.input_dim, "continuous")]

        total = sum(dim for _, dim, _ in self.output_info)
        if total != config.input_dim:
            raise ValueError(
                f"output_info dims sum to {total}, but input_dim is {config.input_dim}. "
                f"Preprocessor and model config are out of sync."
            )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.kl_losses = []
        self.recon_losses = []

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation to reconstructed input."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode.

        Args:
            x: Input data (batch_size, input_dim)

        Returns:
            x_recon: Reconstructed data
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss: reconstruction + KL divergence.

        Args:
            x: Input data
            x_recon: Reconstructed data
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            total_loss: Sum of reconstruction and KL losses
            recon_loss: Reconstruction loss (MSE)
            kl_loss: KL divergence loss
        """
        # Reconstruction loss: Cross Entropy
        batch_size = x.shape[0]
        recon_loss = 0.0

        for start, dim, kind in self.output_info:
            end = start + dim
            if kind == "continuous":
                std = self.sigma[start:end]
                recon = torch.tanh(x_recon[:, start:end])
                nll = ((x[:, start:end] - recon) ** 2) / (2 * std**2) + torch.log(std)
                recon_loss = recon_loss + nll.sum()
            else:  # discrete: one-hot mode indicator or categorical block
                target = torch.argmax(x[:, start:end], dim=1)
                recon_loss = recon_loss + nn.functional.cross_entropy(
                    x_recon[:, start:end], target, reduction="sum"
                )

        recon_loss = recon_loss / batch_size

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def to_device(self, device: str):
        """Move model to specified device."""
        self.to(device)
        self.config.device = device


class TVAETrainer:
    """Trainer for TVAE model."""

    def __init__(self, model: TVAE, config: TVAEConfig):

        self.model = model
        self.config = config
        self.current_beta = config.beta
        self.checkpoint_metric_fn = config.checkpoint_metric_fn or prior_mismatch_score

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Move model to device
        self.model.to(config.device)

        self.best_prior_mismatch = float("inf")
        self.best_state_dict = None

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch.

        Returns:
            avg_total_loss, avg_recon_loss, avg_kl_loss
        """
        self.model.train()

        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        for batch_x in train_loader:
            batch_x = batch_x[0].to(self.config.device)

            # Forward pass
            x_recon, mu, logvar = self.model(batch_x)

            # Compute loss
            loss, recon_loss, kl_loss = self.model.compute_loss(
                batch_x, x_recon, mu, logvar, self.current_beta
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.model.sigma.clamp_(0.01, 1.0)

            # Track losses
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

        return (
            total_loss / num_batches,
            recon_loss_sum / num_batches,
            kl_loss_sum / num_batches,
        )

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model.

        Returns:
            avg_total_loss, avg_recon_loss, avg_kl_loss
        """
        self.model.eval()

        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        for batch_x in val_loader:
            batch_x = batch_x[0].to(self.config.device)

            # Forward pass
            x_recon, mu, logvar = self.model(batch_x)

            # Compute loss
            loss, recon_loss, kl_loss = self.model.compute_loss(
                batch_x, x_recon, mu, logvar, self.current_beta
            )

            # Track losses
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

        return (
            total_loss / num_batches,
            recon_loss_sum / num_batches,
            kl_loss_sum / num_batches,
        )

    def fit(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> dict:
        """Fit the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "beta": [],
        }

        for epoch in range(self.config.num_epochs):
            # Linear KL warm-up: ramp beta from beta_start up to the target beta
            if self.config.beta_warmup_epochs > 0 and self.config.use_beta_warmup:
                warmup_frac = min(1.0, (epoch + 1) / self.config.beta_warmup_epochs)
                current_beta = self.config.beta_start + warmup_frac * (
                    self.config.beta - self.config.beta_start
                )
            else:
                current_beta = self.config.beta
            self.current_beta = current_beta
            history["beta"].append(current_beta)
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_recon_loss"].append(train_recon)
            history["train_kl_loss"].append(train_kl)

            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kl = self.validate(val_loader)
                # Track prior calibration over the full run, not just at sweep time.
                # Reconstruction/classification loss can keep improving even as the
                # encoded posterior drifts away from N(0,1), which is exactly what
                # breaks generation quality later; checkpoint on calibration,
                # not on val_loss alone.
                current_mismatch = self.checkpoint_metric_fn(
                    self.model, val_loader.dataset[:][0].numpy(), self.config.device
                )
                history.setdefault("prior_mismatch", []).append(current_mismatch)

                if current_mismatch < self.best_prior_mismatch:
                    self.best_prior_mismatch = current_mismatch
                    self.best_state_dict = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                history["val_loss"].append(val_loss)
                history["val_recon_loss"].append(val_recon)
                history["val_kl_loss"].append(val_kl)

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}"
                    )

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(
                f"\nRestored best checkpoint (prior_mismatch={self.best_prior_mismatch:.4f})"
            )

        return history
