"""The Model protocol: the unconstrained-space interface the sampler targets."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    """A target distribution on the unconstrained parameter space ``z``.

    Implementations expose the log density and its gradient in ``z``, a map back
    to the constrained parameters for storage, and an exact reference posterior
    used for validation.
    """

    @property
    def dim(self) -> int:
        """Dimension of the unconstrained parameter vector."""
        ...

    @property
    def param_names(self) -> list[str]:
        """Names of the constrained parameters, in storage order."""
        ...

    def logp(self, z: np.ndarray) -> float:
        """Log density at ``z`` on the unconstrained space."""
        ...

    def grad_logp(self, z: np.ndarray) -> np.ndarray:
        """Gradient of ``logp`` at ``z``."""
        ...

    def to_constrained(self, z: np.ndarray) -> np.ndarray:
        """Map ``z`` to the constrained parameters for storage."""
        ...

    def initial_point(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one overdispersed starting point in unconstrained space.

        For a model with a prior this draws from it; the spread between chains is
        what gives split-Rhat something to detect against.
        """
        ...

    def analytic_posterior_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` exact i.i.d. samples from the reference posterior."""
        ...

    def analytic_posterior_moments(self) -> dict[str, np.ndarray]:
        """Closed-form posterior moments used by the acceptance criteria."""
        ...
