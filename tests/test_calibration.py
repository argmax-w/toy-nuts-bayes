"""Calibration diagnostics and their model support, against analytic references.

PIT and coverage are exercised with a predictive sample drawn from the same law
as the observations, where the answers are known (uniform PIT, nominal coverage).
The SBC rank is checked on the exchangeable case it reduces to when calibrated:
one draw labelled the truth among ``L`` siblings has a uniform rank.
"""

import numpy as np

from toynuts.calibration import (
    empirical_cdf,
    pit_values,
    quantile_coverage,
    rank_statistic,
)
from toynuts.models.linear_gaussian import LinearGaussian


def _ks_to_uniform(values: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of a sample to Uniform(0, 1)."""
    x, f = empirical_cdf(values)
    return float(np.max(np.abs(f - x)))


def test_pit_uniform_when_predictive_matches_data():
    """A predictive drawn from the data law gives a near-uniform PIT."""
    rng = np.random.default_rng(0)
    m, s = 4000, 2000
    y = rng.standard_normal(m)
    predictive = rng.standard_normal((m, s))
    pit = pit_values(y, predictive)
    assert abs(pit.mean() - 0.5) < 0.02
    assert _ks_to_uniform(pit) < 0.05


def test_pit_detects_a_too_narrow_predictive():
    """An over-confident predictive pushes the PIT into the tails, away from uniform."""
    rng = np.random.default_rng(1)
    m, s = 4000, 2000
    y = rng.standard_normal(m)
    predictive = 0.5 * rng.standard_normal((m, s))
    assert _ks_to_uniform(pit_values(y, predictive)) > 0.1


def test_quantile_coverage_matches_nominal():
    """Central-interval coverage tracks the nominal level for a matched predictive."""
    rng = np.random.default_rng(2)
    m, s = 8000, 4000
    y = rng.standard_normal(m)
    predictive = rng.standard_normal((m, s))
    levels = np.array([0.5, 0.8, 0.95])
    cover = quantile_coverage(y, predictive, levels)
    np.testing.assert_allclose(cover, levels, atol=0.02)


def test_quantile_coverage_low_when_predictive_too_narrow():
    """A too-narrow predictive under-covers at every level."""
    rng = np.random.default_rng(3)
    m, s = 8000, 4000
    y = rng.standard_normal(m)
    predictive = 0.5 * rng.standard_normal((m, s))
    levels = np.array([0.5, 0.8, 0.95])
    assert (quantile_coverage(y, predictive, levels) < levels - 0.05).all()


def test_rank_statistic_counts_draws_below():
    """The rank is the count of draws strictly below the truth, both layouts."""
    samples = np.array([[0.1, 0.2, 0.9, 1.5]])
    assert rank_statistic(np.array([0.5]), samples)[0] == 2

    truth = np.array([[0.5, 10.0]])
    paired = np.array([[[0.1, 4.0], [0.2, 11.0], [0.9, 12.0]]])  # (S=1, L=3, D=2)
    # dim 0: 0.1, 0.2 below 0.5 -> 2; dim 1: only 4.0 below 10.0 -> 1.
    np.testing.assert_array_equal(rank_statistic(truth, paired)[0], [2, 1])


def test_rank_statistic_uniform_in_exchangeable_case():
    """With truth and draws exchangeable, ranks are uniform on ``0 .. L``."""
    rng = np.random.default_rng(4)
    sims, length = 20000, 9
    block = rng.standard_normal((sims, length + 1))
    ranks = rank_statistic(block[:, 0], block[:, 1:])
    counts = np.bincount(ranks, minlength=length + 1)
    assert counts.size == length + 1
    # Each rank is expected sims / (L + 1); allow a generous multi-sigma band.
    assert np.max(np.abs(counts - sims / (length + 1))) < 200


def test_prior_draws_recover_nig_moments():
    """``prior_draws`` reproduces the NIG prior means and covariances."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 2))
    y = rng.standard_normal(40)
    m0 = np.array([0.5, -1.0])
    V0 = np.array([[4.0, 0.0], [0.0, 9.0]])
    model = LinearGaussian(X, y, m0, V0, a0=6.0, b0=5.0)

    draws = model.prior_draws(400000, rng)
    beta, sigma = draws[:, :2], draws[:, 2]
    np.testing.assert_allclose(beta.mean(axis=0), m0, atol=0.02)
    # Marginal prior Cov[beta] = E[sigma**2] V0 = (b0 / (a0 - 1)) V0.
    np.testing.assert_allclose(np.cov(beta.T), (5.0 / 5.0) * V0, atol=0.1)
    np.testing.assert_allclose((sigma**2).mean(), 5.0 / 5.0, atol=0.02)


def test_predictive_draws_mean_and_noise():
    """Noise-free lines are exact; noisy draws centre on them with scale sigma."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((20, 2))
    y = rng.standard_normal(20)
    model = LinearGaussian(X, y, np.zeros(2), np.eye(2), a0=6.0, b0=5.0)

    params = np.array([[1.0, 2.0, 0.5]])  # one draw: beta = (1, 2), sigma = 0.5
    X_new = np.array([[1.0, 0.0], [1.0, 1.0]])
    lines = model.predictive_draws(params, X_new)
    np.testing.assert_allclose(lines[0], [1.0, 3.0])

    params = np.repeat(params, 200000, axis=0)
    noisy = model.predictive_draws(params, X_new, rng)
    np.testing.assert_allclose(noisy.mean(axis=0), [1.0, 3.0], atol=0.01)
    np.testing.assert_allclose(noisy.std(axis=0), [0.5, 0.5], atol=0.01)
