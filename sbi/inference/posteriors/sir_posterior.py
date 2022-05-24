# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.
from functools import partial
from typing import Any, Callable, Optional, Union
from warnings import warn

import torch
from torch import Tensor

from sbi import utils as utils
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.samplers.sir.sir import sampling_importance_resampling
from sbi.types import Shape, TorchTransform
from sbi.utils.torchutils import ensure_theta_batched


class SIRPosterior(NeuralPosterior):
    r"""Provides rerjeciton sampling to sample from the posterior.<br/><br/>
    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `SIRPosterior` allows to sample from the posterior with sampling-importance 
    resampling (SIR).
    """

    def __init__(
        self,
        potential_fn: Callable,
        proposal: Any,
        theta_transform: Optional[TorchTransform] = None,
        max_sampling_batch_size: int = 10_000,
        num_importance_samples: int = 32,
        device: Optional[str] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples.
            proposal: The proposal distribution.
            theta_transform: Transformation that is applied to parameters. Is not used
                during but only when calling `.map()`.
            max_sampling_batch_size: The batchsize of samples being drawn from
                the proposal at every iteration.
            num_importance_samples: Number of importance samples.
            device: Training device, e.g., "cpu", "cuda" or "cuda:0". If None,
                `potential_fn.device` is used.
            x_shape: Shape of a single simulator output. If passed, it is used to check
                the shape of the observed data and give a descriptive error.
        """
        super().__init__(
            potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.proposal = proposal
        self.max_sampling_batch_size = max_sampling_batch_size
        self.num_importance_samples = num_importance_samples

        self._purpose = (
            "It provides sampling-importance resampling (SIR) to .sample() from the "
            "posterior and can evaluate the _unnormalized_ posterior density with "
            ".log_prob()."
        )

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False
    ) -> Tensor:
        r"""Returns the log-probability of theta under the posterior.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        warn(
            "`.log_prob()` is deprecated for methods that can only evaluate the log-probability up to a normalizing constant. Use `.potential()` instead."
        )
        warn("The log-probability is unnormalized!")

        self.potential_fn.set_x(self._x_else_default_x(x))

        theta = ensure_theta_batched(torch.as_tensor(theta))
        return self.potential_fn(
            theta.to(self._device), track_gradients=track_gradients
        )

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        max_sampling_batch_size: Optional[int] = None,
        num_importance_samples: Optional[int] = None,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = True,
    ):
        r"""Return samples from posterior $p(\theta|x)$ via SIR.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from posterior.
        """
        num_samples = torch.Size(sample_shape).numel()
        self.potential_fn.set_x(self._x_else_default_x(x))

        potential = partial(self.potential_fn, track_gradients=True)

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )
        # Replace arguments that were not passed with their default.
        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )
        num_importance_samples = (
            self.num_importance_samples
            if num_importance_samples is None
            else num_importance_samples
        )

        samples, _ = sampling_importance_resampling(
            potential,
            proposal=self.proposal,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            device=self._device,
        )

        return samples.reshape((*sample_shape, -1))

