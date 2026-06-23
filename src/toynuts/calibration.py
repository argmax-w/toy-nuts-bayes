"""From-scratch calibration diagnostics: PIT, SBC ranks and interval coverage.

These check a different thing from the convergence diagnostics in
``diagnostics.py``. They ask whether the posterior's uncertainty is honest:

- The probability integral transform (PIT) checks the posterior predictive on
  held-out data. If the predictive is calibrated the PIT values are uniform.
- Central-interval coverage is the same idea read as a calibration curve: an
  ``alpha`` credible interval should contain a fraction ``alpha`` of held-out
  points.
- Simulation-based calibration (SBC) checks the inference itself against the
  prior. Drawing parameters from the prior, simulating data and refitting, the
  rank of each true value within its posterior draws is uniform when the sampler
  targets the correct posterior.

Everything here is generic over arrays of draws, so the same code serves the
sampler output and any analytic reference.
"""

from __future__ import annotations

import numpy as np


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their ECDF heights, for plotting against a reference.

    Args:
        values: One-dimensional sample.

    Returns:
        ``(x, F)`` where ``x`` is the sorted sample and ``F`` the step heights
        ``i / n`` at each point.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    return x, np.arange(1, n + 1) / n


def pit_values(
    y_obs: np.ndarray,
    predictive_samples: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Probability integral transform of observations under a predictive sample.

    For each observation ``y_i`` with predictive draws ``s_{i,1..S}`` the PIT value
    is the predictive CDF at ``y_i``, estimated as the fraction of draws at or
    below it. Under a calibrated predictive these values are Uniform(0, 1). For a
    continuous predictive ties are negligible; passing ``rng`` switches to the
    randomised PIT, which spreads each observation uniformly across the CDF jump
    so the transform is exactly uniform even with a discrete predictive sample.

    Args:
        y_obs: Observations, shape ``(m,)``.
        predictive_samples: Predictive draws per observation, shape ``(m, S)``.
        rng: Optional generator selecting the randomised PIT.

    Returns:
        PIT values, shape ``(m,)``.
    """
    y = np.asarray(y_obs, dtype=float)
    s = np.asarray(predictive_samples, dtype=float)
    below = (s < y[:, None]).mean(axis=1)
    at_or_below = (s <= y[:, None]).mean(axis=1)
    if rng is None:
        return at_or_below
    return below + rng.uniform(size=y.shape) * (at_or_below - below)


def quantile_coverage(
    y_obs: np.ndarray,
    predictive_samples: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Empirical coverage of central (equal-tailed) predictive intervals.

    For each central probability ``alpha`` the per-observation interval runs from
    the ``(1 - alpha) / 2`` to the ``(1 + alpha) / 2`` predictive quantile; the
    coverage is the fraction of observations the interval contains. A calibrated
    predictive returns coverage close to each nominal ``alpha``.

    Args:
        y_obs: Observations, shape ``(m,)``.
        predictive_samples: Predictive draws per observation, shape ``(m, S)``.
        levels: Central probabilities to evaluate, shape ``(k,)``.

    Returns:
        Empirical coverage per level, shape ``(k,)``.
    """
    y = np.asarray(y_obs, dtype=float)
    s = np.asarray(predictive_samples, dtype=float)
    levels = np.asarray(levels, dtype=float)
    out = np.empty(levels.shape)
    for k, alpha in enumerate(levels):
        lo = np.quantile(s, (1.0 - alpha) / 2.0, axis=1)
        hi = np.quantile(s, (1.0 + alpha) / 2.0, axis=1)
        out[k] = np.mean((y >= lo) & (y <= hi))
    return out


def rank_statistic(truth: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    """SBC rank of each true value within its posterior draws.

    The rank is the number of posterior draws strictly below the truth, an integer
    in ``0 .. L`` for ``L`` draws. Under a sampler that targets the correct
    posterior these ranks are uniform on ``0 .. L`` across simulations.

    Args:
        truth: True values, shape ``(S,)`` or ``(S, D)`` for ``S`` simulations and
            ``D`` parameters.
        posterior_samples: Posterior draws per simulation, shape ``(S, L)`` to
            match a 1-D ``truth`` or ``(S, L, D)`` to match a 2-D ``truth``.

    Returns:
        Ranks, shape ``(S,)`` or ``(S, D)``.
    """
    truth = np.asarray(truth, dtype=float)
    samples = np.asarray(posterior_samples, dtype=float)
    if truth.ndim == 1:
        return (samples < truth[:, None]).sum(axis=1)
    return (samples < truth[:, None, :]).sum(axis=1)
