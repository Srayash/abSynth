import torch


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
