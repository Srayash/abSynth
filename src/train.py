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
    """Build, train, and return a TVAE model + its final history."""
    set_seed(config.seed)
    model = TVAE(config)
    trainer = TVAETrainer(model, config)
    history = trainer.fit(train_loader, val_loader)
    if verbose:
        print(
            f"latent_dim={config.latent_dim} -> final val_loss={history['val_loss'][-1]:.4f}"
        )
    return model, trainer, history


# def train_tvae_model(
#     latent_dim,
#     num_epochs,
#     beta,
#     beta_start,
#     beta_warmup_epochs,
#     input_dim,
#     output_info,
#     train_loader,
#     val_loader,
#     encoder_hidden_dims=None,
#     decoder_hidden_dims=None,
#     batch_size=64,
#     learning_rate=1e-3,
#     dropout_rate=0.1,
#     device="cpu",
#     use_beta_warmup=False,
#     verbose=False,
# ):
#     """Build, train, and return a TVAE model + its final history."""
#     cfg = TVAEConfig(
#         input_dim=input_dim,
#         latent_dim=latent_dim,
#         encoder_hidden_dims=encoder_hidden_dims,
#         decoder_hidden_dims=decoder_hidden_dims,
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#         num_epochs=num_epochs,
#         beta=beta,
#         beta_start=beta_start,
#         beta_warmup_epochs=beta_warmup_epochs,
#         use_beta_warmup=use_beta_warmup,
#         dropout_rate=dropout_rate,
#         use_batch_norm=True,
#         device=device,
#         output_info=output_info,
#     )
#     model = TVAE(cfg)
#     trainer = TVAETrainer(model, cfg)
#     history = trainer.fit(train_loader, val_loader)
#     if verbose:
#         print(
#             f"latent_dim={latent_dim} -> final val_loss={history['val_loss'][-1]:.4f}"
#         )
#     return model, trainer, history


def run_latent_dim_sweep(
    base_config: TVAEConfig,
    candidate_dims: List[int],
    train_loader,
    val_loader,
    verbose=True,
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
    best_latent_dim = sweep_df.loc[sweep_df["prior_mismatch"].idxmin(), "latent_dim"]
    if verbose:
        print(sweep_df)
        print(f"\nBest latent_dim (by prior calibration): {best_latent_dim}")
    return int(best_latent_dim), sweep_df


# def run_latent_dim_sweep(
#     candidate_dims,
#     sweep_epochs,
#     input_dim,
#     output_info,
#     train_loader,
#     val_loader,
#     device="cpu",
#     verbose=True,
# ):
#     """Train a short run per candidate latent_dim, tie-break on prior
#     calibration (not just val_loss), and return the winner + the full
#     comparison table."""
#     sweep_results = []

#     for ld in candidate_dims:
#         model, _, hist = train_tvae_model(
#             latent_dim=ld,
#             num_epochs=sweep_epochs,
#             beta=0.5,
#             beta_start=0.0,
#             beta_warmup_epochs=15,
#             input_dim=input_dim,
#             output_info=output_info,
#             train_loader=train_loader,
#             val_loader=val_loader,
#             device=device,
#             verbose=verbose,
#         )
#         mismatch = prior_mismatch_score(
#             model, train_loader.dataset[:][0].numpy(), device
#         )
#         sweep_results.append(
#             {
#                 "latent_dim": ld,
#                 "final_val_loss": hist["val_loss"][-1],
#                 "prior_mismatch": mismatch,
#             }
#         )

#     sweep_df = pd.DataFrame(sweep_results)
#     best_latent_dim = sweep_df.loc[sweep_df["prior_mismatch"].idxmin(), "latent_dim"]
#     if verbose:
#         print(sweep_df)
#         print(f"\nBest latent_dim (by prior calibration): {best_latent_dim}")
#     return int(best_latent_dim), sweep_df
