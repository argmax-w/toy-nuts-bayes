"""Conjugate Normal-Inverse-Gamma Bayesian linear regression.

The posterior is Normal-Inverse-Gamma in closed form, so it is an exact reference
for the sampler. Sampling is done in ``z = (beta, u)`` with ``u = log sigma``; the
log density carries the change-of-variables Jacobian and an analytic gradient.
"""

from __future__ import annotations

import numpy as np


class LinearGaussian:
    """NIG linear regression with analytic gradients, draws and moments."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        m0: np.ndarray,
        V0: np.ndarray,
        a0: float,
        b0: float,
    ) -> None:
        """Store the data and prior and precompute the posterior parameters.

        The posterior ``(m_n, V_n, a_n, b_n)`` is precomputed here so the model
        can serve exact draws and moments alongside the sampled posterior.

        Args:
            X: Design matrix, shape ``(n, p)``.
            y: Response vector, shape ``(n,)``.
            m0: Prior mean for ``beta``, shape ``(p,)``.
            V0: Prior covariance factor for ``beta``, shape ``(p, p)``.
            a0: Prior shape for ``sigma**2``.
            b0: Prior scale for ``sigma**2``.
        """
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.m0 = np.asarray(m0, dtype=float)
        self.V0 = np.asarray(V0, dtype=float)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.n, self.p = self.X.shape

        self.V0_inv = np.linalg.inv(self.V0)
        # Normal-Inverse-Gamma conjugate updates (section 3.3).
        self.Vn_inv = self.V0_inv + self.X.T @ self.X
        self.V_n = np.linalg.inv(self.Vn_inv)
        self.m_n = self.V_n @ (self.V0_inv @ self.m0 + self.X.T @ self.y)
        self.a_n = self.a0 + self.n / 2.0
        self.b_n = self.b0 + 0.5 * (
            self.y @ self.y + self.m0 @ self.V0_inv @ self.m0 - self.m_n @ self.Vn_inv @ self.m_n
        )
        self._chol_Vn = np.linalg.cholesky(self.V_n)

    @property
    def dim(self) -> int:
        """Number of coefficients plus one for ``u = log sigma``."""
        return self.p + 1

    @property
    def param_names(self) -> list[str]:
        """Parameter names, ``beta_0 ... beta_{p-1}`` then ``sigma``."""
        return [f"beta_{i}" for i in range(self.p)] + ["sigma"]

    def logp(self, z: np.ndarray) -> float:
        """Unconstrained log density at ``z = (beta, u)``, including the Jacobian.

        The linear-in-u terms collapse to ``-(n + p + 2 a0) u``, where the ``+2u``
        Jacobian has already cancelled the ``-2`` from the scale prior. The rest
        rides on ``exp(-2u)``.
        """
        z = np.asarray(z, dtype=float)
        beta, u = z[:-1], z[-1]
        resid = self.y - self.X @ beta
        dbeta = beta - self.m0
        quad = 0.5 * (resid @ resid + dbeta @ self.V0_inv @ dbeta) + self.b0
        return float(-(self.n + self.p + 2.0 * self.a0) * u - np.exp(-2.0 * u) * quad)

    def grad_logp(self, z: np.ndarray) -> np.ndarray:
        """Analytic gradient of ``logp`` at ``z = (beta, u)`` (section 3.4)."""
        z = np.asarray(z, dtype=float)
        beta, u = z[:-1], z[-1]
        resid = self.y - self.X @ beta
        dbeta = beta - self.m0
        e = np.exp(-2.0 * u)
        grad_beta = e * (self.X.T @ resid - self.V0_inv @ dbeta)
        quad = 0.5 * (resid @ resid + dbeta @ self.V0_inv @ dbeta) + self.b0
        grad_u = -(self.n + self.p + 2.0 * self.a0) + 2.0 * e * quad
        return np.concatenate([grad_beta, [grad_u]])

    def to_constrained(self, z: np.ndarray) -> np.ndarray:
        """Map ``z = (beta, u)`` to ``(beta, sigma)`` for storage."""
        z = np.asarray(z, dtype=float)
        out = z.copy()
        out[-1] = np.exp(z[-1])
        return out

    def initial_point(self, rng: np.random.Generator) -> np.ndarray:
        """An overdispersed start drawn from the prior, returned in ``z`` space."""
        sigma2 = 1.0 / rng.gamma(self.a0, 1.0 / self.b0)
        scatter = np.linalg.cholesky(self.V0) @ rng.standard_normal(self.p)
        beta = self.m0 + np.sqrt(sigma2) * scatter
        return np.concatenate([beta, [0.5 * np.log(sigma2)]])

    def prior_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact i.i.d. draws ``(beta, sigma)`` from the NIG prior.

        Mirrors ``analytic_posterior_draws`` but uses the prior parameters: draw
        ``sigma**2 ~ IG(a0, b0)`` then ``beta | sigma**2 ~ N(m0, sigma**2 V0)``.
        Used for the prior and prior-predictive views and for the SBC generator.
        """
        chol_V0 = np.linalg.cholesky(self.V0)
        sigma2 = 1.0 / rng.gamma(self.a0, 1.0 / self.b0, size=n)
        std = rng.standard_normal((n, self.p))
        beta = self.m0 + np.sqrt(sigma2)[:, None] * (std @ chol_V0.T)
        return np.concatenate([beta, np.sqrt(sigma2)[:, None]], axis=1)

    def analytic_posterior_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact i.i.d. draws ``(beta, sigma)``: ``sigma**2 ~ IG`` then ``beta | sigma**2``."""
        sigma2 = 1.0 / rng.gamma(self.a_n, 1.0 / self.b_n, size=n)
        std = rng.standard_normal((n, self.p))
        beta = self.m_n + np.sqrt(sigma2)[:, None] * (std @ self._chol_Vn.T)
        return np.concatenate([beta, np.sqrt(sigma2)[:, None]], axis=1)

    def predictive_draws(
        self,
        params: np.ndarray,
        X_new: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Predictive lines or draws at new design rows, one per parameter draw.

        With ``rng`` omitted this returns the noise-free regression lines
        ``X_new @ beta`` (the ensemble of fitted lines). With ``rng`` given it adds
        observation noise ``sigma * N(0, 1)``, giving posterior-predictive draws of
        ``y`` whose spread feeds the PIT and coverage checks.

        Args:
            params: Constrained draws ``(beta, sigma)``, shape ``(S, p + 1)``. The
                same shape ``prior_draws`` and ``analytic_posterior_draws`` return.
            X_new: Design rows to predict at, shape ``(m, p)``.
            rng: Optional generator; when given, observation noise is added.

        Returns:
            Array of shape ``(S, m)``: one line or noisy draw per parameter draw.
        """
        params = np.asarray(params, dtype=float)
        X_new = np.asarray(X_new, dtype=float)
        beta, sigma = params[:, : self.p], params[:, self.p]
        mean = beta @ X_new.T
        if rng is None:
            return mean
        return mean + sigma[:, None] * rng.standard_normal(mean.shape)

    def analytic_posterior_moments(self) -> dict[str, np.ndarray]:
        """Closed-form posterior moments of ``beta`` and ``sigma**2`` (section 3.3)."""
        scale = self.b_n / (self.a_n - 1.0)
        return {
            "beta_mean": self.m_n,
            "beta_cov": scale * self.V_n,
            "sigma2_mean": np.asarray(scale),
            "sigma2_var": np.asarray(self.b_n**2 / ((self.a_n - 1.0) ** 2 * (self.a_n - 2.0))),
        }

    @staticmethod
    def synthetic_data(
        n: int,
        beta_true: np.ndarray,
        sigma_true: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate ``(X, y)`` from a known ``(beta, sigma)`` for testing.

        Args:
            n: Number of observations.
            beta_true: True coefficients, shape ``(p,)``.
            sigma_true: True observation standard deviation.
            rng: Random generator.

        Returns:
            The design matrix and response ``(X, y)``.
        """
        beta_true = np.asarray(beta_true, dtype=float)
        X = rng.standard_normal((n, beta_true.size))
        y = X @ beta_true + sigma_true * rng.standard_normal(n)
        return X, y
