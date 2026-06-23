"""From-scratch convergence diagnostics, pandas in and out, no ArviZ.

Implements rank-normalised split-Rhat and bulk and tail ESS after Vehtari et al.
2021, MCSE for the mean, the standard deviation and quantiles, E-BFMI, and the
tree-depth and divergence summaries. The inputs follow the run data model:
``draws`` carries ``chain``, ``draw`` and one column per parameter, and
``sample_stats`` carries the per-draw sampler statistics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# Reshaping and transforms shared by the estimators
# ----------------------------------------------------------------------------


def _param_cols(draws: pd.DataFrame) -> list[str]:
    """Parameter columns, everything but the chain and draw indices."""
    return [c for c in draws.columns if c not in ("chain", "draw")]


def _to_chains(draws: pd.DataFrame, param: str) -> np.ndarray:
    """Reshape one parameter to a ``(n_chains, n_draws)`` array."""
    return draws.pivot(index="chain", columns="draw", values=param).to_numpy()


def _split(arr: np.ndarray) -> np.ndarray:
    """Split each chain in half, doubling the chain count (split-Rhat, split-ESS)."""
    n = arr.shape[1]
    half = n // 2
    return np.concatenate([arr[:, :half], arr[:, half : 2 * half]], axis=0)


def _rank_normalise(arr: np.ndarray) -> np.ndarray:
    """Rank-normalise pooled draws to normal scores, preserving the chain shape."""
    flat = arr.ravel()
    ranks = stats.rankdata(flat)
    n = flat.size
    z = stats.norm.ppf((ranks - 3.0 / 8.0) / (n - 0.25))
    return z.reshape(arr.shape)


# ----------------------------------------------------------------------------
# Core estimators on (chains, draws) arrays
# ----------------------------------------------------------------------------


def _rhat(arr: np.ndarray) -> float:
    """Classic Gelman-Rubin Rhat on a ``(chains, draws)`` array."""
    _, n = arr.shape
    chain_means = arr.mean(axis=1)
    within = arr.var(axis=1, ddof=1).mean()
    between = n * chain_means.var(ddof=1)
    if within <= 0:
        return np.nan
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance for lags ``0 .. n-1`` via FFT."""
    n = x.size
    centred = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    freq = np.fft.rfft(centred, size)
    acov = np.fft.irfft(freq * np.conjugate(freq), size)[:n].real
    return acov / n


def _ess(arr: np.ndarray) -> float:
    """Effective sample size from the combined autocorrelations (Geyer, Vehtari)."""
    m, n = arr.shape
    if n < 4:
        return float(m * n)
    acov = np.stack([_autocov(arr[j]) for j in range(m)])
    within = (acov[:, 0] * n / (n - 1)).mean()
    if within <= 0:
        return float(m * n)
    between = n * arr.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * within + (between / n if m > 1 else 0.0)

    mean_acov = acov.mean(axis=0)
    rho = np.ones(n)
    rho[1:] = 1.0 - (within - mean_acov[1:]) / var_plus

    # Geyer's initial positive sequence: sum consecutive lag pairs until one is
    # non-positive, then enforce monotonicity on the pair sums.
    pair_sums = []
    for k in range(n // 2):
        gamma = rho[2 * k] + rho[2 * k + 1]
        if k > 0 and gamma <= 0:
            break
        pair_sums.append(gamma)
    for k in range(1, len(pair_sums)):
        pair_sums[k] = min(pair_sums[k], pair_sums[k - 1])

    tau = -1.0 + 2.0 * sum(pair_sums)
    tau = max(tau, 1.0 / np.log10(m * n))  # cap ESS at the usual N log10 N
    return float(m * n / tau)


# ----------------------------------------------------------------------------
# Public per-parameter diagnostics
# ----------------------------------------------------------------------------


def split_rhat(draws: pd.DataFrame) -> pd.Series:
    """Rank-normalised split-Rhat per parameter, max of bulk and folded.

    Args:
        draws: Draws frame with ``chain``, ``draw`` and parameter columns.

    Returns:
        Split-Rhat indexed by parameter.
    """
    out = {}
    for param in _param_cols(draws):
        split = _split(_to_chains(draws, param))
        bulk = _rhat(_rank_normalise(split))
        folded = _rhat(_rank_normalise(np.abs(split - np.median(split))))
        out[param] = max(bulk, folded)
    return pd.Series(out, name="r_hat")


def ess_bulk(draws: pd.DataFrame) -> pd.Series:
    """Rank-normalised bulk effective sample size per parameter.

    Args:
        draws: Draws frame with ``chain``, ``draw`` and parameter columns.

    Returns:
        Bulk ESS indexed by parameter.
    """
    out = {}
    for param in _param_cols(draws):
        out[param] = _ess(_rank_normalise(_split(_to_chains(draws, param))))
    return pd.Series(out, name="ess_bulk")


def ess_tail(draws: pd.DataFrame) -> pd.Series:
    """Tail effective sample size per parameter, the min over the 5% and 95% tails.

    Args:
        draws: Draws frame with ``chain``, ``draw`` and parameter columns.

    Returns:
        Tail ESS indexed by parameter.
    """
    out = {}
    for param in _param_cols(draws):
        split = _split(_to_chains(draws, param))
        q05, q95 = np.quantile(split, [0.05, 0.95])
        low = _ess((split <= q05).astype(float))
        high = _ess((split <= q95).astype(float))
        out[param] = min(low, high)
    return pd.Series(out, name="ess_tail")


def mcse_mean(draws: pd.DataFrame) -> pd.Series:
    """Monte Carlo standard error of the posterior mean per parameter."""
    out = {}
    for param in _param_cols(draws):
        arr = _to_chains(draws, param)
        ess = _ess(_rank_normalise(_split(arr)))
        out[param] = arr.std(ddof=1) / np.sqrt(ess)
    return pd.Series(out, name="mcse_mean")


def mcse_sd(draws: pd.DataFrame) -> pd.Series:
    """Monte Carlo standard error of the posterior standard deviation per parameter."""
    out = {}
    for param in _param_cols(draws):
        arr = _to_chains(draws, param)
        ess = _ess(_rank_normalise(_split(arr)))
        # Sampling variance of the sample sd for a near-normal target is
        # sd**2 / (2 n_eff), so its standard error is sd / sqrt(2 n_eff).
        out[param] = arr.std(ddof=1) / np.sqrt(2.0 * ess)
    return pd.Series(out, name="mcse_sd")


def mcse_quantile(
    draws: pd.DataFrame,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> pd.DataFrame:
    """Monte Carlo standard error of selected posterior quantiles.

    The error is read off the spread of the draws over a probability interval set
    by the effective sample size of the quantile's indicator series.

    Args:
        draws: Draws frame with ``chain``, ``draw`` and parameter columns.
        quantiles: Quantiles to report.

    Returns:
        MCSE with parameters on the index and the quantiles on the columns.
    """
    z975 = stats.norm.ppf(0.975)
    rows = {}
    for param in _param_cols(draws):
        split = _split(_to_chains(draws, param))
        ordered = np.sort(split.ravel())
        n = ordered.size
        values = {}
        for prob in quantiles:
            indicator = (split <= np.quantile(split, prob)).astype(float)
            ess = _ess(indicator)
            spread = z975 * np.sqrt(prob * (1.0 - prob) / ess)
            lo = int(np.clip(round((prob - spread) * n), 0, n - 1))
            hi = int(np.clip(round((prob + spread) * n), 0, n - 1))
            values[prob] = (ordered[hi] - ordered[lo]) / (2.0 * z975)
        rows[param] = values
    return pd.DataFrame(rows).T


def ebfmi(energy_by_chain: np.ndarray) -> pd.Series:
    """E-BFMI per chain, ``sum((E_n - E_{n-1})**2) / sum((E_n - E_bar)**2)``.

    Args:
        energy_by_chain: Energy series, shape ``(n_chains, n_draws)``.

    Returns:
        E-BFMI indexed by chain.
    """
    energy = np.asarray(energy_by_chain, dtype=float)
    if energy.ndim == 1:
        energy = energy[None, :]
    out = []
    for row in energy:
        denom = np.sum((row - row.mean()) ** 2)
        out.append(np.sum(np.diff(row) ** 2) / denom if denom > 0 else np.nan)
    return pd.Series(out, index=pd.RangeIndex(energy.shape[0], name="chain"), name="e_bfmi")


def tree_depth_summary(sample_stats: pd.DataFrame) -> pd.Series:
    """Tree-depth distribution as proportions per depth.

    Args:
        sample_stats: Sample-stats frame with ``tree_depth``.

    Returns:
        Proportion of draws at each depth, indexed by depth.
    """
    counts = sample_stats["tree_depth"].value_counts(normalize=True).sort_index()
    counts.index.name = "tree_depth"
    counts.name = "proportion"
    return counts


def divergence_summary(sample_stats: pd.DataFrame) -> pd.Series:
    """Divergence count and rate.

    Args:
        sample_stats: Sample-stats frame with ``divergent``.

    Returns:
        A summary series with the count and the rate.
    """
    divergent = sample_stats["divergent"].to_numpy().astype(bool)
    return pd.Series(
        {"divergences": int(divergent.sum()), "rate": float(divergent.mean())},
        name="divergences",
    )


def summary(draws: pd.DataFrame, sample_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the per-parameter diagnostic table.

    Args:
        draws: Draws frame with ``chain``, ``draw`` and parameter columns.
        sample_stats: Optional sample-stats frame, accepted for a uniform call
            signature; the per-parameter table does not need it.

    Returns:
        A table with the mean, standard deviation, MCSE, split-Rhat and the bulk
        and tail ESS per parameter.
    """
    rhat = split_rhat(draws)
    bulk = ess_bulk(draws)
    tail = ess_tail(draws)
    mcse = mcse_mean(draws)
    rows = {}
    for param in _param_cols(draws):
        arr = _to_chains(draws, param)
        rows[param] = {
            "mean": arr.mean(),
            "sd": arr.std(ddof=1),
            "mcse_mean": mcse[param],
            "ess_bulk": bulk[param],
            "ess_tail": tail[param],
            "r_hat": rhat[param],
        }
    columns = ["mean", "sd", "mcse_mean", "ess_bulk", "ess_tail", "r_hat"]
    return pd.DataFrame(rows).T[columns]
