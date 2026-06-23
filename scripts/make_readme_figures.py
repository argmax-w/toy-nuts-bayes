"""Render the curated figures used by the README.

The figures retrace the three-notebook narrative: the priors and what they imply
for the line, the sampled posterior against the closed-form reference, the
posterior predictive, then the convergence and calibration checks. The data
setup is reconstructed deterministically (the same seeds the Results notebook
uses) and the sampled run is read back from ``outputs/notebook_run``; nothing is
refitted here, so the slow simulation-based calibration is read from Parquet.

Run ``01_results.ipynb`` (or otherwise populate the run directory) first, then:

    python scripts/make_readme_figures.py

Figures are written to ``assets`` so they can be committed alongside the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toynuts.calibration import empirical_cdf, pit_values, quantile_coverage  # noqa: E402
from toynuts.diagnostics import (  # noqa: E402
    divergence_summary,
    ebfmi,
    ess_bulk,
    ess_tail,
    split_rhat,
    tree_depth_summary,
)
from toynuts.io import read_run  # noqa: E402
from toynuts.models.linear_gaussian import LinearGaussian  # noqa: E402

RUN_DIR = PROJECT_ROOT / "outputs" / "notebook_run"
FIG_DIR = PROJECT_ROOT / "assets"

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def build_model() -> tuple[LinearGaussian, np.ndarray, np.ndarray, np.ndarray, float]:
    """Reconstruct the Results-notebook data and model from the fixed seed."""
    data_rng = np.random.default_rng(131)
    n = 80
    x = np.sort(data_rng.uniform(-2.5, 2.5, n))
    X = np.column_stack([np.ones(n), x])
    beta_true = np.array([1.0, 2.0])
    sigma_true = 1.0
    y = X @ beta_true + sigma_true * data_rng.standard_normal(n)

    m0 = np.zeros(2)
    V0 = np.diag([2.0, 2.0])
    a0, b0 = 6.0, 5.0
    model = LinearGaussian(X, y, m0, V0, a0, b0)
    return model, x, y, beta_true, sigma_true


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to the figures directory and report it."""
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(PROJECT_ROOT))


def fig_prior_predictive(model: LinearGaussian, x, y) -> None:
    """Lines the weakly informative prior considers plausible, with the data."""
    grid_x = np.linspace(-2.8, 2.8, 120)
    grid_X = np.column_stack([np.ones_like(grid_x), grid_x])
    lines = model.predictive_draws(model.prior_draws(120, np.random.default_rng(11)), grid_X)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid_x, lines.T, color="C0", alpha=0.15, lw=0.8)
    ax.scatter(x, y, s=18, color="k", zorder=3, label="data")
    ax.plot([], [], color="C0", alpha=0.6, label="prior predictive lines")
    ax.set(xlabel="x", ylabel="y", title="Prior predictive: what the priors imply")
    ax.legend()
    save(fig, "prior_predictive.png")


def fig_posterior_marginals(model, post, beta_true, sigma_true) -> None:
    """Sampled marginals (filled) against exact closed-form draws (outline)."""
    ref = model.analytic_posterior_draws(40000, np.random.default_rng(12))
    # Append u = log sigma, the scale in the space the sampler works in.
    post_u = np.column_stack([post, np.log(post[:, 2])])
    ref_u = np.column_stack([ref, np.log(ref[:, 2])])
    names = ["beta_0 (intercept)", "beta_1 (slope)", "sigma", "u = log sigma"]
    truth = [beta_true[0], beta_true[1], sigma_true, np.log(sigma_true)]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for k, ax in enumerate(axes):
        ax.hist(post_u[:, k], bins=70, density=True, alpha=0.55, color="C0", label="NUTS posterior")
        ax.hist(ref_u[:, k], bins=70, density=True, histtype="step", color="k", lw=1.1,
                label="analytic posterior")
        ax.axvline(truth[k], color="C3", ls="--", lw=1, label="true value")
        ax.axvline(post_u[:, k].mean(), color="C1", ls=":", lw=1.2, label="posterior mean")
        ax.set_title(names[k])
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Posterior of each parameter: sampled vs closed form")
    fig.tight_layout()
    save(fig, "posterior_marginals.png")


def fig_posterior_predictive(model, post, x, y) -> None:
    """The posterior mean line with credible and predictive bands, over the data."""
    grid_x = np.linspace(-2.8, 2.8, 120)
    grid_X = np.column_stack([np.ones_like(grid_x), grid_x])
    lines = model.predictive_draws(post, grid_X)
    preds = model.predictive_draws(post, grid_X, np.random.default_rng(14))
    mean_line = lines.mean(0)
    cred_lo, cred_hi = np.quantile(lines, [0.05, 0.95], axis=0)
    pred_lo, pred_hi = np.quantile(preds, [0.05, 0.95], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(grid_x, pred_lo, pred_hi, color="C0", alpha=0.15, label="90% predictive band")
    ax.fill_between(grid_x, cred_lo, cred_hi, color="C0", alpha=0.35,
                    label="90% credible band for the line")
    ax.plot(grid_x, mean_line, color="C0", lw=2, label="posterior mean line")
    ax.scatter(x, y, s=18, color="k", zorder=3, label="data")
    ax.set(xlabel="x", ylabel="y", title="Posterior predictive: best line with bounds")
    ax.legend(fontsize=8)
    save(fig, "posterior_predictive.png")


def fig_trace(draws) -> None:
    """Per-chain traces and the overlaid per-chain posteriors."""
    params = ["beta_0", "beta_1", "sigma"]
    chains = sorted(draws["chain"].unique())
    fig, axes = plt.subplots(len(params), 2, figsize=(11, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    for row, p in enumerate(params):
        for c in chains:
            d = draws[draws.chain == c]
            axes[row, 0].plot(d["draw"], d[p], lw=0.4, alpha=0.7)
            axes[row, 1].hist(d[p], bins=40, density=True, histtype="step", alpha=0.8)
        axes[row, 0].set_ylabel(p)
    axes[0, 0].set_title("traces")
    axes[0, 1].set_title("per-chain posterior")
    axes[-1, 0].set_xlabel("draw")
    fig.tight_layout()
    save(fig, "trace.png")


def fig_energy(stats) -> None:
    """Energy overlay: marginal energy against the energy transitions."""
    from scipy.stats import gaussian_kde

    chains = sorted(stats["chain"].unique())
    energy = stats.pivot(index="chain", columns="draw", values="energy").to_numpy()
    bfmi = ebfmi(energy)

    marginal, transitions = [], []
    for c in chains:
        e = stats[stats.chain == c].sort_values("draw")["energy"].to_numpy()
        marginal.append(e - e.mean())
        transitions.append(np.diff(e))
    marginal = np.concatenate(marginal)
    transitions = np.concatenate(transitions)

    grid = np.linspace(min(marginal.min(), transitions.min()),
                       max(marginal.max(), transitions.max()), 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    for vals, label, color in [(marginal, "marginal energy", "C0"),
                               (transitions, "energy transition", "C1")]:
        density = gaussian_kde(vals)(grid)
        ax.fill_between(grid, density, color=color, alpha=0.35)
        ax.plot(grid, density, color=color, lw=1.5, label=label)
    ax.set(xlabel="energy (centred)", ylabel="density", yticks=[],
           title=f"energy overlay (E-BFMI {bfmi.min():.2f} to {bfmi.max():.2f})")
    ax.legend()
    save(fig, "energy.png")


def fig_pit(y_test, pred) -> None:
    """PIT as a histogram and as the ECDF departure from uniform, with 95% bands."""
    pit = pit_values(y_test, pred)
    xe, fe = empirical_cdf(pit)
    m = pit.size

    nbins = 20
    counts, edges = np.histogram(pit, bins=nbins, range=(0, 1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    expected = m / nbins
    bin_band = 1.96 * np.sqrt(m * (1 / nbins) * (1 - 1 / nbins))
    p_grid = np.arange(1, m + 1) / m
    ecdf_band = 1.96 * np.sqrt(p_grid * (1 - p_grid) / m)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(centres, counts, width=0.95 / nbins, color="C0", alpha=0.6)
    axes[0].axhspan(expected - bin_band, expected + bin_band, color="k", alpha=0.1,
                    label="per-bin 95% band")
    axes[0].axhline(expected, color="k", ls="--", lw=1)
    axes[0].set(title="PIT histogram", xlabel="PIT value", ylabel="count")
    axes[0].legend()
    axes[1].fill_between(xe, -ecdf_band, ecdf_band, color="k", alpha=0.12,
                         label="95% pointwise band")
    axes[1].plot(xe, fe - xe, color="C0", lw=1.3)
    axes[1].axhline(0.0, color="k", ls="--", lw=1)
    axes[1].set(title="PIT ECDF minus uniform (cumulative)", xlabel="PIT value",
                ylabel="ECDF - uniform")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "pit.png")
    return pit.mean(), float(np.max(np.abs(fe - xe)))


def fig_coverage(y_test, pred) -> None:
    """Empirical central-interval coverage against the nominal level."""
    levels = np.linspace(0.1, 0.95, 18)
    cover = quantile_coverage(y_test, pred, levels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(levels, cover, "o-", color="C0", label="empirical")
    ax.set(xlabel="nominal central probability", ylabel="empirical coverage",
           title="Quantile coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, "coverage.png")


def fig_sbc(ranks, n_sims, length) -> None:
    """SBC rank histograms, uniform when the inference is unbiased."""
    names = ["beta_0", "beta_1", "sigma"]
    nbins = 20
    expected = n_sims / nbins
    band = 1.96 * np.sqrt(n_sims * (1 / nbins) * (1 - 1 / nbins))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, ax in enumerate(axes):
        ax.hist(ranks[:, k], bins=nbins, range=(-0.5, length + 0.5), color="C0", alpha=0.7)
        ax.axhspan(expected - band, expected + band, color="k", alpha=0.1)
        ax.axhline(expected, color="k", ls="--", lw=1)
        ax.set(title=names[k], xlabel="rank")
    axes[0].set_ylabel("count")
    fig.suptitle("SBC rank histograms (uniform when calibrated)")
    fig.tight_layout()
    save(fig, "sbc.png")


def main() -> None:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"no saved run at {RUN_DIR}; run 01_results.ipynb first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model, x, y, beta_true, sigma_true = build_model()
    draws, stats, _ = read_run(RUN_DIR)
    post = draws[["beta_0", "beta_1", "sigma"]].to_numpy()

    fig_prior_predictive(model, x, y)
    fig_posterior_marginals(model, post, beta_true, sigma_true)
    fig_posterior_predictive(model, post, x, y)
    fig_trace(draws)
    fig_energy(stats)

    test = pd.read_parquet(RUN_DIR / "test_set.parquet")
    y_test = test["y_test"].to_numpy()
    pred = pd.read_parquet(RUN_DIR / "predictive.parquet").to_numpy()
    ranks = pd.read_parquet(RUN_DIR / "sbc_ranks.parquet")[["beta_0", "beta_1", "sigma"]].to_numpy()
    meta = pd.read_parquet(RUN_DIR / "sbc_meta.parquet").iloc[0]

    pit_mean, ks = fig_pit(y_test, pred)
    fig_coverage(y_test, pred)
    fig_sbc(ranks, int(meta["n_sims"]), int(meta["n_thinned"]))

    # Headline numbers, quoted in the README prose.
    rhat = split_rhat(draws)
    ess = pd.DataFrame({"bulk": ess_bulk(draws), "tail": ess_tail(draws)})
    bfmi = ebfmi(stats.pivot(index="chain", columns="draw", values="energy").to_numpy())
    print("\n--- headline numbers ---")
    print("max split-Rhat:", round(float(rhat.max()), 4))
    print("min bulk ESS:", int(ess["bulk"].min()), " min tail ESS:", int(ess["tail"].min()))
    print("divergences:", int(divergence_summary(stats)["divergences"]))
    print("max tree depth reached:", int(stats["tree_depth"].max()))
    print("E-BFMI range:", round(float(bfmi.min()), 3), "to", round(float(bfmi.max()), 3))
    print("tree depth summary:\n", tree_depth_summary(stats).to_string())
    print(f"PIT mean {pit_mean:.3f} (uniform 0.5), KS {ks:.3f}")


if __name__ == "__main__":
    main()
