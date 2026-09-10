import torch
import numpy as np


@torch.no_grad()
def prior_mismatch_score(model, data, device):
    """Lower is better. Measures how far the aggregate encoded posterior
    (mu, logvar) is from matching the N(0,1) prior across the dataset —
    a large value means samples drawn from the prior at generation time
    will look nothing like what the decoder was actually trained on."""
    model.eval()
    mu, logvar = model.encode(torch.FloatTensor(data).to(device))
    mu = mu.cpu().numpy()
    std = torch.exp(0.5 * logvar).cpu().numpy()
    mean_shift = abs(mu.mean(axis=0)).mean()  # want ~0
    std_dev_from_one = abs(mu.std(axis=0) - 1.0).mean()  # want ~0
    return mean_shift + std_dev_from_one


def compute_discrete_class_weights(transformed_data, output_info, beta=0.999):
    """Class-balanced weights per discrete block (Cui et al. 2019).
    Rare classes get upweighted in cross-entropy so the model predict them."""
    weights = {}
    for start, dim, kind in output_info:
        if kind != "discrete":
            continue
        targets = transformed_data[:, start : start + dim].argmax(axis=1)
        counts = np.clip(
            np.bincount(targets, minlength=dim).astype(np.float64), 1, None
        )
        effective_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / effective_num
        w = w / w.mean()
        weights[start] = torch.tensor(w, dtype=torch.float32)
    return weights
