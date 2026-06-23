"""Energy terms and momentum sampling for a fixed metric.

The potential is the negative log density and the kinetic energy comes from a
Gaussian momentum ``p ~ Normal(0, M)``. The metric ``M`` is fixed for the whole
run and is never adapted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from toynuts.models.base import Model


def potential(model: Model, z: np.ndarray) -> float:
    """Potential energy, the negative log density at ``z``."""
    return -model.logp(z)


def grad_potential(model: Model, z: np.ndarray) -> np.ndarray:
    """Gradient of the potential at ``z``, the negative of ``grad_logp``."""
    return -model.grad_logp(z)


def kinetic(p: np.ndarray, m_inv: np.ndarray) -> float:
    """Gaussian kinetic energy ``0.5 * p^T M_inv p``."""
    return float(0.5 * p @ m_inv @ p)


def hamiltonian(model: Model, z: np.ndarray, p: np.ndarray, m_inv: np.ndarray) -> float:
    """Total energy ``H = U(z) + K(p)``."""
    return potential(model, z) + kinetic(p, m_inv)


def sample_momentum(metric: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Refresh the momentum, ``p ~ Normal(0, M)``.

    This is the distribution whose negative log density is the kinetic energy up
    to a constant. Drawing ``L @ standard_normal`` with ``L L^T = M`` gives a
    momentum with covariance ``M``.
    """
    chol = np.linalg.cholesky(metric)
    return chol @ rng.standard_normal(metric.shape[0])
