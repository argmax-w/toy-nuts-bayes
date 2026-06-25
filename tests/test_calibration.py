"""Calibration diagnostics and their model support, against analytic references.

PIT and coverage are exercised with a predictive sample drawn from the same law
as the observations, where the answers are known (uniform PIT, nominal coverage).
The SBC rank is checked on the exchangeable case it reduces to when calibrated:
one draw labelled the truth among ``L`` siblings has a uniform rank.
"""

import numpy as np
from scipy import stats as sps

from toynuts.calibration import (
    _gpdfit,
    coverage_from_pit,
    crps_mixture,
    ecdf_simultaneous_band,
    empirical_cdf,
    loo_elpd,
    loo_pit,
    pit_rb,
    pit_values,
    psis,
    quantile_coverage,
    rank_statistic,
    stratified_pit,
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


# ----------------------------------------------------------------------------
# Rao-Blackwellised PIT and coverage
# ----------------------------------------------------------------------------


def _matched_draws(rng, S=2000, sigma=1.0):
    """Tight draws of ``(beta_0, beta_1, sigma)`` near ``beta = 0`` with fixed sigma."""
    return np.column_stack(
        [rng.normal(0, 1e-3, S), rng.normal(0, 1e-3, S), np.full(S, sigma)]
    )


def test_pit_rb_uniform_for_matched_predictive():
    """The averaged-CDF PIT is uniform when the predictive matches the data law."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([np.ones(n), x])
    y = rng.standard_normal(n)  # matches beta = 0, sigma = 1
    pit = pit_rb(y, X, _matched_draws(rng))
    assert abs(pit.mean() - 0.5) < 0.02
    _x, f = empirical_cdf(pit)
    assert np.max(np.abs(f - _x)) < 0.05


def test_pit_rb_agrees_with_sampled_pit():
    """The RB PIT matches the sampled PIT, with less noise, on the same draws."""
    rng = np.random.default_rng(1)
    n, S = 800, 4000
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([0.5, -0.3]) + 1.2 * rng.standard_normal(n)
    draws = np.column_stack(
        [rng.normal(0.5, 0.05, S), rng.normal(-0.3, 0.05, S), np.abs(rng.normal(1.2, 0.05, S))]
    )
    rb = pit_rb(y, X, draws)
    sampled = pit_values(y, X @ draws[:, :2].T + draws[:, 2] * rng.standard_normal((n, S)))
    # Both estimate the same predictive CDF; agreement is to Monte Carlo error.
    assert np.mean(np.abs(rb - sampled)) < 0.02


def test_coverage_from_pit_matches_nominal():
    """Central coverage read off uniform PIT values tracks the nominal level."""
    rng = np.random.default_rng(2)
    pit = rng.random(50000)
    levels = np.array([0.5, 0.8, 0.95])
    np.testing.assert_allclose(coverage_from_pit(pit, levels), levels, atol=0.01)


# ----------------------------------------------------------------------------
# PSIS and leave-one-out
# ----------------------------------------------------------------------------


def test_gpdfit_recovers_known_shape():
    """The generalised Pareto fit recovers a known tail shape parameter."""
    rng = np.random.default_rng(3)
    k, sigma = 0.5, 1.0
    u = rng.random(20000)
    x = np.sort(sigma / k * ((1 - u) ** (-k) - 1.0))  # GPD(k, sigma) quantiles
    k_hat, _ = _gpdfit(x)
    assert abs(k_hat - k) < 0.1


def test_psis_caps_and_flags_heavy_tails():
    """Smoothed weights never exceed the largest raw weight, and k_hat reads the tail."""
    rng = np.random.default_rng(4)
    light = rng.standard_normal((1, 4000))  # well-behaved log weights
    heavy = -6.0 * np.log(rng.random((1, 4000)))  # exponential right tail in log space
    lw_light, k_light = psis(light)
    _lw_heavy, k_heavy = psis(heavy)
    # The output is centred on the largest raw weight, so nothing exceeds zero.
    assert lw_light.max() <= 1e-9
    assert np.isfinite(k_light[0])
    assert k_heavy[0] > k_light[0]  # the heavy tail reads a larger k_hat


def test_psis_loo_matches_analytic_lg_loo():
    """PSIS-LOO elpd and LOO-PIT match the exact analytic leave-one-out for LG.

    The conjugate leave-one-out predictive is Student-t in closed form; it is the
    reference here only, never used by the implementation.
    """
    rng = np.random.default_rng(7)
    n = 60
    x = rng.uniform(-2, 2, n)
    X = np.column_stack([np.ones(n), x])
    y = X @ np.array([1.0, 2.0]) + rng.standard_normal(n)
    m0, V0, a0, b0 = np.zeros(2), np.diag([2.0, 2.0]), 6.0, 5.0
    model = LinearGaussian(X, y, m0, V0, a0, b0)
    draws = model.analytic_posterior_draws(8000, np.random.default_rng(8))

    got = loo_elpd(y, X, draws)
    pit, khat = loo_pit(y, X, draws)

    exact_lpd = np.empty(n)
    exact_cdf = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        Xk, yk = X[keep], y[keep]
        Vn_inv = model.V0_inv + Xk.T @ Xk
        Vn = np.linalg.inv(Vn_inv)
        mn = Vn @ (model.V0_inv @ m0 + Xk.T @ yk)
        an = a0 + Xk.shape[0] / 2.0
        bn = b0 + 0.5 * (yk @ yk + m0 @ model.V0_inv @ m0 - mn @ Vn_inv @ mn)
        loc = X[i] @ mn
        scale = np.sqrt((bn / an) * (1.0 + X[i] @ Vn @ X[i]))
        student = sps.t(df=2 * an, loc=loc, scale=scale)
        exact_lpd[i] = student.logpdf(y[i])
        exact_cdf[i] = student.cdf(y[i])

    assert abs(float(got["elpd"]) - exact_lpd.sum()) < 0.5
    assert np.max(np.abs(pit - exact_cdf)) < 0.02
    assert np.nanmax(khat) < 0.7


# ----------------------------------------------------------------------------
# Proper scoring and simultaneous bands
# ----------------------------------------------------------------------------


def test_crps_single_gaussian_matches_formula():
    """The mixture CRPS reduces to the closed-form Gaussian CRPS for one draw."""
    mu, sigma, y = 0.7, 1.3, 2.1
    got = crps_mixture(np.array([y]), np.array([[1.0]]), np.array([[mu, sigma]]), max_draws=1)[0]
    w = (y - mu) / sigma
    expected = sigma * (w * (2 * sps.norm.cdf(w) - 1) + 2 * sps.norm.pdf(w) - 1 / np.sqrt(np.pi))
    assert abs(got - expected) < 1e-9


def test_crps_mixture_matches_monte_carlo():
    """The closed-form mixture CRPS matches a Monte-Carlo energy-score estimate."""
    rng = np.random.default_rng(5)
    S = 400
    draws = np.column_stack(
        [rng.normal(1, 0.3, S), rng.normal(2, 0.2, S), np.abs(rng.normal(1.0, 0.1, S))]
    )
    Xi, yi = np.array([[1.0, 0.5]]), np.array([2.5])
    got = crps_mixture(yi, Xi, draws, max_draws=S)[0]

    mu = (Xi @ draws[:, :2].T).ravel()
    sig = draws[:, 2]
    pick = rng.integers(0, S, 60000)
    reps = mu[pick] + sig[pick] * rng.standard_normal(60000)
    term1 = np.abs(reps - yi[0]).mean()
    term2 = 0.5 * np.abs(reps[:30000] - reps[30000:]).mean()
    assert abs(got - (term1 - term2)) < 0.01


def test_ecdf_simultaneous_band_controls_familywise_error():
    """The simultaneous band covers the whole ECDF at the nominal rate, wider than pointwise."""
    n = 100
    d = ecdf_simultaneous_band(n, prob=0.95, n_sim=4000, rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    u = np.sort(rng.random((4000, n)), axis=1)
    below, above = np.arange(n) / n, np.arange(1, n + 1) / n
    sup = np.maximum(np.abs(u - below), np.abs(u - above)).max(axis=1)
    coverage = np.mean(sup <= d)
    assert abs(coverage - 0.95) < 0.02
    assert d > 1.96 * np.sqrt(0.25 / n)  # wider than the pointwise band at the centre


def test_stratified_pit_partitions_values():
    """Stratified PIT splits the values by their stratum label."""
    pit = np.array([0.1, 0.2, 0.3, 0.4])
    strata = np.array([0, 1, 0, 1])
    out = stratified_pit(pit, strata)
    np.testing.assert_array_equal(out[0], [0.1, 0.3])
    np.testing.assert_array_equal(out[1], [0.2, 0.4])


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
