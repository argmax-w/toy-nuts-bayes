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
from scipy.special import logsumexp, ndtr


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


# ----------------------------------------------------------------------------
# Rao-Blackwellised predictive checks (the standing rule)
#
# These integrate the observation noise out analytically per draw, through the
# Gaussian CDF, and Monte Carlo only over the parameter draws. Nothing about the
# predictive ``y`` is sampled. They assume the Gaussian observation model of the
# linear regression, taking constrained draws ``(beta, sigma)`` of shape
# ``(S, p + 1)`` and design rows ``X`` of shape ``(N, p)``. The conjugate
# Student-t marginal is deliberately not used; the average is over draws.
# ----------------------------------------------------------------------------


def _z_and_sigma(
    y: np.ndarray, X: np.ndarray, draws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised residual ``(y - x . beta_s) / sigma_s`` per observation and draw.

    Returns ``(z, sigma, mu)`` with ``z`` and ``mu`` shaped ``(N, S)`` and
    ``sigma`` shaped ``(S,)``. The full matrix is materialised, which is fine for
    the in-sample checks where ``N`` is small; the held-out CDF streams instead.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    draws = np.asarray(draws, dtype=float)
    p = X.shape[1]
    beta, sigma = draws[:, :p], draws[:, p]
    mu = X @ beta.T  # (N, S): the per-draw predictive mean at each observation
    z = (y[:, None] - mu) / sigma[None, :]
    return z, sigma, mu


def predictive_mixture_cdf(
    y: np.ndarray, X: np.ndarray, draws: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """Rao-Blackwellised predictive CDF at each point, averaged over draws.

    The estimate is ``mean_s Phi((y_i - x_i . beta_s) / sigma_s)``, the per-draw
    Gaussian CDF averaged over the draws. The draws are streamed in blocks of
    ``chunk`` so peak memory is ``O(N * chunk)`` rather than ``O(N * S)``.

    Args:
        y: Points to evaluate the CDF at, shape ``(N,)``.
        X: Design rows for those points, shape ``(N, p)``.
        draws: Constrained ``(beta, sigma)`` draws, shape ``(S, p + 1)``.
        chunk: Number of draws per block.

    Returns:
        The averaged CDF value at each point, shape ``(N,)``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    draws = np.asarray(draws, dtype=float)
    p = X.shape[1]
    beta, sigma = draws[:, :p], draws[:, p]
    S = draws.shape[0]
    acc = np.zeros(y.shape)
    for start in range(0, S, chunk):
        sl = slice(start, min(start + chunk, S))
        mu = X @ beta[sl].T  # (N, k)
        z = (y[:, None] - mu) / sigma[sl][None, :]
        acc += ndtr(z).sum(axis=1)
    return acc / S


def pit_rb(y: np.ndarray, X: np.ndarray, draws: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Rao-Blackwellised PIT: the averaged predictive CDF at each observation.

    The out-of-sample replacement for :func:`pit_values`. With held-out points the
    draws never saw, this is an honest, sampling-free PIT; under a calibrated
    predictive the values are Uniform(0, 1).
    """
    return predictive_mixture_cdf(y, X, draws, chunk=chunk)


def coverage_from_pit(pit: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Central-interval coverage read straight off the PIT values.

    A central ``alpha`` interval contains ``y_i`` exactly when its PIT value lies in
    ``[(1 - alpha) / 2, (1 + alpha) / 2]``, that is ``|u_i - 0.5| <= alpha / 2``.
    This inverts the averaged predictive CDF implicitly, so the coverage inherits
    the Rao-Blackwellisation of whatever PIT it is given (held-out or LOO).

    Args:
        pit: PIT values, shape ``(N,)``.
        levels: Central probabilities to evaluate, shape ``(k,)``.

    Returns:
        Empirical coverage per level, shape ``(k,)``.
    """
    pit = np.asarray(pit, dtype=float)
    levels = np.asarray(levels, dtype=float)
    return np.array([np.mean(np.abs(pit - 0.5) <= alpha / 2.0) for alpha in levels])


def pointwise_loglik(y: np.ndarray, X: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Per-observation, per-draw Gaussian log-likelihood, shape ``(N, S)``.

    The input to importance-sampling LOO: ``log N(y_i; x_i . beta_s, sigma_s**2)``.
    """
    z, sigma, _ = _z_and_sigma(y, X, draws)
    return -0.5 * z**2 - np.log(sigma)[None, :] - 0.5 * np.log(2.0 * np.pi)


# ----------------------------------------------------------------------------
# Pareto-smoothed importance sampling, for leave-one-out reuse of a single fit
# ----------------------------------------------------------------------------


def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Fit a generalised Pareto to sorted positive exceedances (Zhang-Stephens 2009).

    Returns the shape ``k`` and scale ``sigma``. This is the estimator the PSIS
    literature uses, with the empirical-Bayes profile over a grid of ``theta`` and
    a weak prior pulling ``k`` towards ``0.5`` in small samples.
    """
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    theta = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    theta /= prior_bs * x[int(n / 4 + 0.5) - 1]
    theta += 1.0 / x[-1]
    k_theta = np.log1p(-theta[:, None] * x[None, :]).mean(axis=1)
    profile = n * (np.log(-theta / k_theta) - k_theta - 1.0)
    weights = np.exp(profile - logsumexp(profile))  # stable softmax over the grid
    theta_hat = float(np.sum(theta * weights))
    k = np.log1p(-theta_hat * x).mean()
    sigma = -k / theta_hat
    k = (n * k + prior_k * 0.5) / (n + prior_k)  # small-sample shrinkage
    return float(k), float(sigma)


def _gpinv(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """Quantile function of the generalised Pareto with location 0."""
    if sigma <= 0:
        return np.full_like(p, np.nan)
    if abs(k) < 1e-30:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def _psis_row(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smooth one observation's log importance weights, returning ``k_hat``.

    The largest ``M`` weights are replaced by the expected order statistics of a
    generalised Pareto fitted to the tail, then all weights are capped at the
    largest raw weight. ``k_hat`` above about 0.7 flags unreliable weights.
    """
    lw = log_weights - log_weights.max()  # stabilise; invariant for normalised use
    S = lw.size
    M = int(min(0.2 * S, 3.0 * np.sqrt(S)))
    if S < 50 or M < 5:
        return lw, np.nan
    order = np.argsort(lw)
    tail_idx = order[S - M :]  # the M largest, ascending in lw
    log_cutoff = lw[order[S - M - 1]]
    exceed = np.exp(lw[tail_idx]) - np.exp(log_cutoff)
    if exceed[-1] <= 0:  # degenerate tail, nothing to smooth
        return lw, np.nan
    k, sigma = _gpdfit(exceed)
    quantiles = (np.arange(1, M + 1) - 0.5) / M
    smoothed = np.log(_gpinv(quantiles, k, sigma) + np.exp(log_cutoff))
    out = lw.copy()
    out[tail_idx] = np.minimum(smoothed, 0.0)  # cap at the largest raw weight
    return out, k


def psis(log_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pareto-smoothed importance sampling over an ``(N, S)`` log-weight matrix.

    Smooths each observation's weights independently.

    Args:
        log_weights: Raw log importance weights, shape ``(N, S)``.

    Returns:
        ``(smoothed_log_weights, k_hat)`` with shapes ``(N, S)`` and ``(N,)``.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    out = np.empty_like(log_weights)
    khat = np.empty(log_weights.shape[0])
    for i, row in enumerate(log_weights):
        out[i], khat[i] = _psis_row(row)
    return out, khat


def loo_pit(y: np.ndarray, X: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out PIT via PSIS, the in-sample calibration check.

    Naive in-sample PIT is over-optimistic because each ``y_i`` shaped the
    posterior. PSIS-LOO reweights the single full-data fit by ``1 / p(y_i | theta_s)``,
    Pareto-smooths those weights, then averages the per-draw CDF with them, giving
    ``u_i = sum_s w_is Phi((y_i - x_i . beta_s) / sigma_s)`` from the leave-one-out
    predictive. Uniform(0, 1) under calibration; ``k_hat`` flags where the weights
    are unreliable.

    Args:
        y: Observations, shape ``(N,)``.
        X: Design rows, shape ``(N, p)``.
        draws: Constrained ``(beta, sigma)`` draws, shape ``(S, p + 1)``.

    Returns:
        ``(pit, k_hat)`` with shapes ``(N,)`` and ``(N,)``.
    """
    z, sigma, _ = _z_and_sigma(y, X, draws)
    loglik = -0.5 * z**2 - np.log(sigma)[None, :] - 0.5 * np.log(2.0 * np.pi)
    lw, khat = psis(-loglik)  # LOO weights are 1 / likelihood
    log_norm = logsumexp(lw, axis=1, keepdims=True)
    weights = np.exp(lw - log_norm)
    pit = (weights * ndtr(z)).sum(axis=1)
    return pit, khat


def loo_elpd(y: np.ndarray, X: np.ndarray, draws: np.ndarray) -> dict[str, np.ndarray]:
    """Expected log pointwise predictive density by PSIS-LOO, a proper score.

    Reports the leave-one-out ``elpd`` with its standard error, the pointwise
    contributions, the effective number of parameters ``p_loo`` and the per-point
    Pareto ``k_hat``. Higher ``elpd`` is better; ``p_loo`` against the parameter
    count and ``k_hat`` above 0.7 are the health flags.

    Args:
        y: Observations, shape ``(N,)``.
        X: Design rows, shape ``(N, p)``.
        draws: Constrained ``(beta, sigma)`` draws, shape ``(S, p + 1)``.

    Returns:
        A dict with ``elpd`` (scalar), ``se`` (scalar), ``pointwise`` ``(N,)``,
        ``p_loo`` (scalar) and ``k_hat`` ``(N,)``.
    """
    loglik = pointwise_loglik(y, X, draws)
    S = loglik.shape[1]
    lw, khat = psis(-loglik)
    elpd_i = logsumexp(lw + loglik, axis=1) - logsumexp(lw, axis=1)
    lpd_i = logsumexp(loglik, axis=1) - np.log(S)  # full-data log predictive density
    elpd = float(elpd_i.sum())
    se = float(np.sqrt(elpd_i.size * np.var(elpd_i, ddof=1)))
    p_loo = float((lpd_i - elpd_i).sum())
    return {
        "elpd": np.asarray(elpd),
        "se": np.asarray(se),
        "pointwise": elpd_i,
        "p_loo": np.asarray(p_loo),
        "k_hat": khat,
    }


# ----------------------------------------------------------------------------
# Proper scoring and simultaneous bands
# ----------------------------------------------------------------------------


def _crps_kernel(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """``E|Z|`` for ``Z ~ N(mu, sigma**2)``, the building block of the mixture CRPS."""
    r = mu / sigma
    return mu * (2.0 * ndtr(r) - 1.0) + 2.0 * sigma * np.exp(-0.5 * r**2) / np.sqrt(2.0 * np.pi)


def crps_mixture(
    y: np.ndarray, X: np.ndarray, draws: np.ndarray, max_draws: int = 400
) -> np.ndarray:
    """Continuous ranked probability score of the Gaussian-mixture predictive.

    For the per-draw mixture ``(1/S) sum_s N(x_i . beta_s, sigma_s**2)`` the CRPS has
    the closed form ``mean_s E|Y_s - y| - (1/2) mean_{s,t} E|Y_s - Y_t|`` (Grimit
    et al. 2006), with each expectation the Gaussian ``E|.|`` kernel. The second
    term is ``O(S**2)`` per observation, so the draws are thinned evenly to
    ``max_draws`` for it. Lower CRPS is sharper-given-calibrated.

    Args:
        y: Observations, shape ``(N,)``.
        X: Design rows, shape ``(N, p)``.
        draws: Constrained ``(beta, sigma)`` draws, shape ``(S, p + 1)``.
        max_draws: Cap on the draws used in the quadratic cross term.

    Returns:
        The per-observation CRPS, shape ``(N,)``.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    draws = np.asarray(draws, dtype=float)
    p = X.shape[1]
    S = draws.shape[0]
    idx = np.unique(np.linspace(0, S - 1, min(S, max_draws)).round().astype(int))
    beta, sigma = draws[idx, :p], draws[idx, p]
    cross_sigma = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    out = np.empty(y.shape)
    for i in range(y.size):
        mu = X[i] @ beta.T
        term1 = _crps_kernel(mu - y[i], sigma).mean()
        term2 = 0.5 * _crps_kernel(mu[:, None] - mu[None, :], cross_sigma).mean()
        out[i] = term1 - term2
    return out


def ecdf_simultaneous_band(
    n: int, prob: float = 0.95, n_sim: int = 4000, rng: np.random.Generator | None = None
) -> float:
    """Half-width of a simultaneous confidence band for the ECDF of ``n`` uniforms.

    A pointwise band controls the error at each point and is crossed somewhere
    along the curve far more than its nominal rate. This calibrates a single
    constant half-width ``d`` so the whole curve stays inside with probability
    ``prob``: the supremum deviation ``sup_t |F_n(t) - t|`` is simulated under the
    null and ``d`` is its ``prob`` quantile. Overlaid on an ECDF-minus-uniform plot
    the band is ``+/- d``.

    Args:
        n: Number of points the ECDF is built from.
        prob: Simultaneous coverage, e.g. 0.95.
        n_sim: Null replicates used to calibrate the width.
        rng: Generator for the calibration draws.

    Returns:
        The band half-width ``d``.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = np.sort(rng.random((n_sim, n)), axis=1)
    below = np.arange(n) / n  # ECDF just left of each order statistic
    above = np.arange(1, n + 1) / n  # and just right
    sup = np.maximum(np.abs(u - below[None, :]), np.abs(u - above[None, :])).max(axis=1)
    return float(np.quantile(sup, prob))


def stratified_pit(pit: np.ndarray, strata: np.ndarray) -> dict[object, np.ndarray]:
    """Split PIT values by stratum, for the conditional calibration check.

    Marginal PIT can be uniform while the model is miscalibrated within regions,
    because opposing errors cancel in aggregate. Partitioning by a covariate bin or
    a predicted-value bin and checking uniformity within each stratum localises
    where the predictive fails.

    Args:
        pit: PIT values, shape ``(N,)``.
        strata: Stratum label per value, shape ``(N,)``.

    Returns:
        A dict mapping each stratum label to its PIT values.
    """
    pit = np.asarray(pit, dtype=float)
    strata = np.asarray(strata)
    return {key: pit[strata == key] for key in np.unique(strata)}
