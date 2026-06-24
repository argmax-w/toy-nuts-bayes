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


def _predictive_panels(model, draws, line_draws, x, y, kind) -> None:
    """Three predictive views: the averaged density, sample lines and the spread.

    The density is the Rao-Blackwellised predictive, the average of the per-draw
    Gaussian likelihood ``mean_s N(y; x @ beta_s, sigma_s**2)``. The observation
    noise is integrated out analytically (the Gaussian density, never a sampled
    ``y``) while the parameter uncertainty is averaged over the draws; for the
    posterior these are the sampler's own draws. Its 95% interval inverts the same
    Gaussian-mixture CDF, so nothing about ``y`` is sampled either. The spread panel
    integrates the noise out too, plotting the population standard deviation each
    draw implies, ``sqrt(Var(X beta) + sigma**2)``, against the observed value. Only
    the centre panel shows the draws themselves.

    Args:
        model: The fitted model, used for the design matrix and the predictions.
        draws: Constrained ``(beta, sigma)`` draws to average the density and the
            spread over (prior draws, or the sampler's posterior draws).
        line_draws: A smaller set of draws shown as sample lines.
        x: Observed inputs.
        y: Observed responses.
        kind: ``"Prior"`` or ``"Posterior"``, used in the titles and filename.
    """
    from scipy.special import ndtr

    grid_x = np.linspace(-2.8, 2.8, 120)
    grid_X = np.column_stack([np.ones_like(grid_x), grid_x])

    beta, sigma = draws[:, : model.p], draws[:, model.p]
    means = beta @ grid_X.T  # (S, nx): the line value of each draw

    # Rao-Blackwellised density p(y | x) ~= mean_s N(y; x @ beta_s, sigma_s**2),
    # one x-column at a time since the full (y, x, draw) tensor is too large. The
    # 95% interval inverts the mixture CDF, mean_s Phi((y - mean_s) / sigma_s), so
    # the observation noise is integrated out rather than sampled. The plotting
    # window spans 3.5 total sds, the mixture's law-of-total-variance spread.
    mu = means.mean(0)
    total_sd = np.sqrt((means**2 + (sigma**2)[:, None]).mean(0) - mu**2)
    y_grid = np.linspace((mu - 3.5 * total_sd).min(), (mu + 3.5 * total_sd).max(), 200)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    dens = np.empty((y_grid.size, grid_x.size))
    lo, hi = np.empty(grid_x.size), np.empty(grid_x.size)
    for j in range(grid_x.size):
        z = (y_grid[:, None] - means[:, j][None, :]) / sigma[None, :]
        dens[:, j] = (np.exp(-0.5 * z**2) * norm[None, :]).mean(axis=1)
        cdf = ndtr(z).mean(axis=1)
        lo[j], hi[j] = np.interp([0.025, 0.975], cdf, y_grid)
    ylim = (y_grid.min(), y_grid.max())

    # Sample lines for the centre panel.
    lines = model.predictive_draws(line_draws, grid_X)

    # Implied population sd of a dataset with the noise integrated out, one value
    # per draw: sqrt(Var(X beta) + sigma**2), against the observed sd.
    implied_sd = np.sqrt((beta @ model.X.T).var(axis=1) + sigma**2)
    implied_sd = np.sqrt((beta @ model.X.T).var(axis=1) + sigma**2)
    obs_sd = float(y.std())
    # An adaptive window so the wide prior and the tight posterior both read well.
    sd_lo = min(float(np.quantile(implied_sd, 0.005)), obs_sd)
    sd_hi = max(float(np.quantile(implied_sd, 0.995)), obs_sd)
    pad = 0.05 * (sd_hi - sd_lo)
    sd_range = (max(0.0, sd_lo - pad), sd_hi + pad)

    fig, (ax_l, ax_c, ax_r) = plt.subplots(1, 3, figsize=(16, 4.3))

    mesh = ax_l.pcolormesh(grid_x, y_grid, dens, cmap="Blues", shading="auto")
    fig.colorbar(mesh, ax=ax_l, label="density  p(y | x)")
    ax_l.plot(grid_x, lo, color="C0", lw=1.2, ls="--", label="95% credible interval")
    ax_l.plot(grid_x, hi, color="C0", lw=1.2, ls="--")
    ax_l.scatter(x, y, s=14, color="k", edgecolor="w", linewidth=0.3, zorder=3, label="data")
    ax_l.set(xlabel="x", ylabel="y", title=f"{kind} predictive density", ylim=ylim)
    ax_l.legend(loc="upper left", fontsize=8)

    ax_c.plot(grid_x, lines.T, color="C0", alpha=0.15, lw=0.8)
    ax_c.plot([], [], color="C0", alpha=0.6, label=f"{kind.lower()} predictive lines")
    ax_c.scatter(x, y, s=14, color="k", edgecolor="w", linewidth=0.3, zorder=3, label="data")
    ax_c.set(xlabel="x", ylabel="y", title=f"{kind} predictive samples", ylim=ylim)
    ax_c.legend(loc="upper left", fontsize=8)

    ax_r.hist(implied_sd, bins=60, range=sd_range, density=True, color="C0", alpha=0.6,
              label=f"{kind.lower()} draws")
    ax_r.axvline(obs_sd, color="k", lw=1.5, label=f"observed sd = {obs_sd:.2f}")
    ax_r.set(xlabel="standard deviation implied by each draw", ylabel="density",
             title=f"{kind} predictive spread", xlim=sd_range)
    ax_r.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{kind} predictive: the density, sample draws and the implied spread")
    fig.tight_layout()
    save(fig, f"{kind.lower()}_predictive.png")


def fig_prior_predictive(model: LinearGaussian, x, y) -> None:
    """The prior predictive three ways, before any data is seen."""
    rng = np.random.default_rng(11)
    line_draws = model.prior_draws(120, rng)
    draws = model.prior_draws(8000, rng)
    _predictive_panels(model, draws, line_draws, x, y, "Prior")


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
    """The posterior predictive three ways, from the sampler's draws."""
    idx = np.random.default_rng(13).choice(post.shape[0], 150, replace=False)
    _predictive_panels(model, post, post[idx], x, y, "Posterior")


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
