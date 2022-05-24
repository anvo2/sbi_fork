from typing import Callable

import torch
from torch import Tensor
from torch.distributions import Distribution


def sampling_importance_resampling(
    num_samples: int,
    potential_fn: Callable,
    proposal: Distribution,
    num_importance_samples: int = 32,
    num_samples_batch: int = 10000,
    **kwargs,
) -> Tensor:
    """Perform sampling importance resampling (SIR).

    Args:
        num_samples: Number of samples to draw.
        potential_fn: Potential function, this may be used to debias the proposal.
        proposal: Proposal distribution to propose samples.
        K: Number of proposed samples form which only one is selected based on its
            importance weight.
        num_samples_batch: Number of samples processed in parallel. For large K you may
            want to reduce this, depending on your memory capabilities.

    Returns:
        Tensor: Samples of shape (num_samples, event_shape)

    """
    final_samples = []
    num_samples_batch = min(num_samples, num_samples_batch)
    iters = int(num_samples / num_samples_batch)
    for _ in range(iters):
        batch_size = min(num_samples_batch, num_samples - len(final_samples))
        with torch.no_grad():
            thetas = proposal.sample(torch.Size((batch_size * num_importance_samples,)))
            logp = potential_fn(thetas)
            logq = proposal.log_prob(thetas)
            weights = (
                (logp - logq)
                .reshape(batch_size, num_importance_samples)
                .softmax(-1)
                .cumsum(-1)
            )
            u = torch.rand(batch_size, 1, device=thetas.device)
            mask = torch.cumsum(weights >= u, -1) == 1
            samples = thetas.reshape(batch_size, num_importance_samples, -1)[mask]
            final_samples.append(samples)
    return torch.vstack(final_samples)
