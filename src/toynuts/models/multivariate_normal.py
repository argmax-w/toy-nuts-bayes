"""Friendly correlated multivariate normal, an exact smoke target.

All parameters are unconstrained, so there is no transform, and the reference
posterior draws are exact Gaussian draws. Run before the regression to confirm
the sampler recovers a known mean and covariance.
"""

from __future__ import annotations

import numpy as np


class MultivariateNormal:
    """A fixed multivariate-normal target with analytic gradient and draws."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """Store the target mean and covariance and precompute the precision.

        Args:
            mean: Target mean vector, shape ``(d,)``.
            cov: Target covariance, shape ``(d, d)``, symmetric positive definite.
        """
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        if self.cov.shape != (self.mean.size, self.mean.size):
            raise ValueError("cov must be square and match the length of mean")
        # Cholesky doubles as the positive-definite check and the draw factor.
        self._chol = np.linalg.cholesky(self.cov)
        self.cov_inv = np.linalg.inv(self.cov)
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("cov must be positive definite")
        # Normalising constant, kept so logp matches a textbook Gaussian density.
        self._log_norm = -0.5 * self.mean.size * np.log(2.0 * np.pi) - 0.5 * logdet

    @property
    def dim(self) -> int:
        """Dimension of the target."""
        return int(self.mean.size)

    @property
    def param_names(self) -> list[str]:
        """Parameter names, ``x_0 ... x_{d-1}``."""
        return [f"x_{i}" for i in range(self.dim)]

    def logp(self, z: np.ndarray) -> float:
        """Gaussian log density at ``z``."""
        diff = np.asarray(z, dtype=float) - self.mean
        return float(self._log_norm - 0.5 * diff @ self.cov_inv @ diff)

    def grad_logp(self, z: np.ndarray) -> np.ndarray:
        """Gradient of the Gaussian log density, ``-cov_inv @ (z - mean)``."""
        return -self.cov_inv @ (np.asarray(z, dtype=float) - self.mean)

    def to_constrained(self, z: np.ndarray) -> np.ndarray:
        """Identity map, the target is already unconstrained."""
        return np.asarray(z, dtype=float)

    def initial_point(self, rng: np.random.Generator) -> np.ndarray:
        """An overdispersed start, three target standard deviations of spread."""
        return self.mean + 3.0 * (self._chol @ rng.standard_normal(self.dim))

    def analytic_posterior_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact i.i.d. Gaussian draws from the target."""
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def analytic_posterior_moments(self) -> dict[str, np.ndarray]:
        """The known mean and covariance."""
        return {"mean": self.mean, "cov": self.cov}
