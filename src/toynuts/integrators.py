"""Leapfrog integrator: one reversible step of Hamiltonian dynamics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def leapfrog(
    grad_logp: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    p: np.ndarray,
    eps: float,
    m_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One leapfrog step: half kick, full drift, half kick.

    The momentum kicks use ``grad_logp``, the gradient of the log density, which
    is the negative gradient of the potential. The two half kicks straddle one
    full position drift, which is what makes the step time-reversible and
    volume-preserving.

    Args:
        grad_logp: Gradient of the target log density at a position.
        q: Position.
        p: Momentum.
        eps: Step size, signed by the integration direction.
        m_inv: Inverse mass matrix.

    Returns:
        The updated ``(q, p)`` after one step.
    """
    p_half = p + 0.5 * eps * grad_logp(q)
    q_new = q + eps * (m_inv @ p_half)
    p_new = p_half + 0.5 * eps * grad_logp(q_new)
    return q_new, p_new
