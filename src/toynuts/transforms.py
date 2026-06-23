"""Bijection for the positive scale parameter and its change-of-variables term.

The sampler works on an unconstrained space, so the positive standard deviation
``sigma`` is mapped to ``u = log sigma``. The target log density then picks up the
log absolute Jacobian of the ``sigma**2`` to ``u`` map, which is where the ``+2u``
term in the model log density comes from.
"""

from __future__ import annotations

import numpy as np


def to_unconstrained(sigma: float) -> float:
    """Map the positive scale to the unconstrained line.

    Args:
        sigma: Positive standard deviation.

    Returns:
        The unconstrained value ``u = log sigma``.
    """

    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")

    return np.log(sigma)


def to_constrained(u: float) -> float:
    """Map the unconstrained value back to the positive scale.

    Args:
        u: Unconstrained value ``u = log sigma``.

    Returns:
        The standard deviation ``sigma = exp(u)``.
    """

    return np.exp(u)


def log_abs_det_jacobian(u: float) -> float:
    """Log absolute determinant of the ``sigma**2`` to ``u`` map.

    Used by the model log density so that sampling in ``u`` targets the correct
    density. Up to an additive constant this is ``2 * u``.

    Args:
        u: Unconstrained value ``u = log sigma``.

    Returns:
        The log absolute Jacobian determinant.
    """

    return 2 * u + np.log(2)
