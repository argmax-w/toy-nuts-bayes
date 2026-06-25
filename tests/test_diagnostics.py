"""Diagnostics against analytic and independently derived references."""

import numpy as np
import pandas as pd

from toynuts.diagnostics import (
    autocorr,
    chain_ranks,
    diagnostic_status,
    ebfmi,
    ess_bulk,
    split_rhat,
)


def _draws_df(arr, name="x"):
    """Wrap a ``(chains, draws)`` array as a draws DataFrame for one parameter."""
    m, n = arr.shape
    return pd.DataFrame(
        {
            "chain": np.repeat(np.arange(m), n),
            "draw": np.tile(np.arange(n), m),
            name: arr.ravel(),
        }
    )


def test_split_rhat_near_one_for_iid():
    """Split-Rhat sits near 1 on i.i.d. Normal draws."""
    rng = np.random.default_rng(0)
    df = _draws_df(rng.normal(size=(4, 2000)))
    assert split_rhat(df)["x"] < 1.01


def test_split_rhat_elevated_for_non_mixed_chains():
    """Split-Rhat is elevated when chains sit at different locations."""
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(4, 2000)) + np.arange(4)[:, None] * 2.0
    assert split_rhat(_draws_df(arr))["x"] > 1.1


def test_ess_matches_ar1_formula():
    """Bulk ESS on an AR(1) series matches ``N (1 - rho) / (1 + rho)``."""
    rng = np.random.default_rng(2)
    rho, m, n = 0.6, 4, 8000
    arr = np.empty((m, n))
    for j in range(m):
        x = np.empty(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * rng.normal()
        arr[j] = x
    expected = m * n * (1 - rho) / (1 + rho)
    assert np.isclose(ess_bulk(_draws_df(arr))["x"], expected, rtol=0.15)


def test_ebfmi_matches_formula():
    """E-BFMI matches the direct formula on a controlled energy series."""
    rng = np.random.default_rng(3)
    energy = rng.normal(size=(3, 1000))
    got = ebfmi(energy)
    for j in range(3):
        row = energy[j]
        expected = np.sum(np.diff(row) ** 2) / np.sum((row - row.mean()) ** 2)
        assert np.isclose(got[j], expected)


def test_chain_ranks_uniform_for_iid():
    """Pooled ranks split by chain are uniform across the pooled range for i.i.d. draws."""
    rng = np.random.default_rng(0)
    m, n = 4, 2000
    ranks = chain_ranks(_draws_df(rng.normal(size=(m, n))), "x")
    assert ranks.shape == (m, n)
    # Every chain spans the full pooled range with no systematic offset.
    assert np.all(ranks >= 1) and ranks.max() == m * n
    assert abs(ranks.mean() - (m * n + 1) / 2) < 1.0
    assert abs(ranks.mean(axis=1).max() - ranks.mean(axis=1).min()) < 0.1 * m * n


def test_autocorr_lag0_one_and_decays_for_ar1():
    """Autocorrelation starts at 1 and decays towards the AR(1) rate."""
    rng = np.random.default_rng(1)
    rho, n = 0.7, 6000
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * rng.normal()
    acf = autocorr(_draws_df(x[None, :]), max_lag=5)
    by_lag = acf.set_index("lag")["acf"]
    assert np.isclose(by_lag[0], 1.0)
    assert abs(by_lag[1] - rho) < 0.05  # lag-1 autocorrelation is rho


def test_diagnostic_status_boundaries():
    """The traffic-light thresholds map values to the expected status."""
    assert diagnostic_status("r_hat", 1.005) == "green"
    assert diagnostic_status("r_hat", 1.015) == "amber"
    assert diagnostic_status("r_hat", 1.02) == "red"
    assert diagnostic_status("ess", 500) == "green"
    assert diagnostic_status("ess", 300) == "amber"
    assert diagnostic_status("ess", 199) == "red"
    assert diagnostic_status("e_bfmi", 0.4) == "green"
    assert diagnostic_status("e_bfmi", 0.3) == "red"
    assert diagnostic_status("divergences", 0) == "green"
    assert diagnostic_status("divergences", 2) == "amber"
    assert diagnostic_status("divergences", 50) == "red"
