import torch
import numpy as np


def apply_continuous_activation(x_recon: np.ndarray, output_info: list) -> np.ndarray:
    """required for any reconstruction or generation to live 
    in the space the model actually learned."""
    x_recon = x_recon.copy()
    for start, dim, kind in output_info:
        if kind == "continuous":
            end = start + dim
            x_recon[:, start:end] = np.tanh(x_recon[:, start:end])
    return x_recon


@torch.no_grad()
def encode_data(model, data: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    mu, _ = model.encode(torch.FloatTensor(data).to(device))
    return mu.cpu().numpy()


@torch.no_grad()
def decode_latent(model, latent: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    return model.decode(torch.FloatTensor(latent).to(device)).cpu().numpy()


@torch.no_grad()
def reconstruct(model, data: np.ndarray, device: str, output_info: list) -> np.ndarray:
    latent = encode_data(model, data, device)
    return apply_continuous_activation(
        decode_latent(model, latent, device), output_info
    )


@torch.no_grad()
def generate_synthetic_samples(
    model, n_samples: int, device: str, output_info: list
) -> np.ndarray:
    model.eval()
    z = torch.randn(n_samples, model.config.latent_dim).to(device)
    return apply_continuous_activation(model.decode(z).cpu().numpy(), output_info)
