import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, List
from src.diagnostics import prior_mismatch_score
from src.model import TVAEConfig, TVAE, TVAETrainer, set_seed
import pandas as pd
from dataclasses import replace


def train_tvae_model(config: TVAEConfig, train_loader, val_loader, verbose=False):
    """Build, train, and return a TVAE model and its final history."""
    set_seed(config.seed)
    model = TVAE(config)
    trainer = TVAETrainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    if verbose:
        print(
            f"latent_dim={config.latent_dim} -> final val_loss={history['val_loss'][-1]:.4f}"
        )
    return model, trainer, history


def run_latent_dim_sweep(
    base_config: TVAEConfig,
    candidate_dims: List[int],
    train_loader,
    val_loader,
    verbose=True,
    prior_mismatch_threshold: float = 0.3,  # matches the diagnostics "target: < 0.3"
):
    sweep_results = []
    for ld in candidate_dims:
        cfg = replace(base_config, latent_dim=ld)
        model, _, hist = train_tvae_model(cfg, train_loader, val_loader, verbose)
        mismatch = prior_mismatch_score(
            model, train_loader.dataset[:][0].numpy(), cfg.device
        )
        sweep_results.append(
            {
                "latent_dim": ld,
                "final_val_loss": hist["val_loss"][-1],
                "prior_mismatch": mismatch,
            }
        )

    sweep_df = pd.DataFrame(sweep_results)

    passing = sweep_df[sweep_df["prior_mismatch"] < prior_mismatch_threshold]
    if passing.empty:
        # nothing calibrated well enough; fall back to the old behavior
        best_latent_dim = sweep_df.loc[sweep_df["prior_mismatch"].idxmin(), "latent_dim"]
    else:
        best_latent_dim = passing.loc[passing["final_val_loss"].idxmin(), "latent_dim"]

    if verbose:
        print(sweep_df)
        print(f"\nBest latent_dim (best val_loss among prior_mismatch < {prior_mismatch_threshold}): {best_latent_dim}")
    return int(best_latent_dim), sweep_df