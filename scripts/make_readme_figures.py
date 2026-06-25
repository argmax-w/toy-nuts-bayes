"""Render the curated figures used by the README.

The figures retrace the three-notebook narrative: the priors and what they imply
for the line, the sampled posterior against the closed-form reference, the
posterior predictive, then the convergence and calibration checks. The data
setup is reconstructed deterministically (the same seeds the Results notebook
uses) and the sampled run is read back from ``outputs/notebook_run``; nothing is
refitted here, so the slow simulation-based calibration is read from Parquet.

Every figure follows the single plotting convention in ``toynuts.plotting``: black
for the observed data, a cool ribbon for the prior and a warm one for the
posterior, parameter hues for the marginals and a reserved status palette for the
diagnostics. The Rao-Blackwellised predictive checks (held-out PIT, LOO-PIT,
coverage) and the proper scores are computed here from the saved draws.

Run ``01_results.ipynb`` (or otherwise populate the run directory) first, then:

    python scripts/make_readme_figures.py

Figures are written to ``assets`` so they can be committed alongside the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from toynuts import plotting as P  # noqa: E402
from toynuts.calibration import (  # noqa: E402
    coverage_from_pit,
    crps_mixture,
    ecdf_simultaneous_band,
    empirical_cdf,
    loo_elpd,
    loo_pit,
    pit_rb,
)
from toynuts.diagnostics import (  # noqa: E402
    diagnostic_status,
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

# A thin white casing so a coloured line stays legible over a dark ribbon peak.
CASING = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]


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
    the centre panel shows the draws themselves. The ribbon is cool for the prior
    and warm for the posterior, per the plotting convention.

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

    lines = model.predictive_draws(line_draws, grid_X)

    # Implied population sd with the noise integrated out, one value per draw:
    # sqrt(Var(X beta) + sigma**2), against the observed sd.
    implied_sd = np.sqrt((beta @ model.X.T).var(axis=1) + sigma**2)
    obs_sd = float(y.std())
    sd_lo = min(float(np.quantile(implied_sd, 0.005)), obs_sd)
    sd_hi = max(float(np.quantile(implied_sd, 0.995)), obs_sd)
    pad = 0.05 * (sd_hi - sd_lo)
    sd_range = (max(0.0, sd_lo - pad), sd_hi + pad)

    cmap = P.cmap_for(kind)
    deep = P.deep_for(kind)

    fig, (ax_l, ax_c, ax_r) = plt.subplots(1, 3, figsize=(16, 4.3))

    mesh = ax_l.pcolormesh(grid_x, y_grid, dens, cmap=cmap, shading="auto", alpha=P.RIBBON_ALPHA)
    fig.colorbar(mesh, ax=ax_l, label="density  p(y | x)")
    for edge in (lo, hi):
        ax_l.plot(grid_x, edge, color=deep, lw=1.5, ls=P.CI_LINE_STYLE, path_effects=CASING)
    ax_l.plot([], [], color=deep, lw=1.5, ls=P.CI_LINE_STYLE, label="95% credible interval")
    P.scatter_data(ax_l, x, y)
    ax_l.set(xlabel="x", ylabel="y", title=f"{kind} predictive density", ylim=ylim)
    ax_l.legend(loc="upper left", fontsize=8)

    ax_c.plot(grid_x, lines.T, color=deep, alpha=0.14, lw=0.8)
    ax_c.plot([], [], color=deep, alpha=0.7, label=f"{kind.lower()} predictive lines")
    P.scatter_data(ax_c, x, y)
    ax_c.set(xlabel="x", ylabel="y", title=f"{kind} predictive samples", ylim=ylim)
    ax_c.legend(loc="upper left", fontsize=8)

    ax_r.hist(implied_sd, bins=60, range=sd_range, density=True, color=deep, alpha=0.55,
              label=f"{kind.lower()} draws")
    ax_r.axvline(obs_sd, color=P.DATA_COLOUR, lw=1.5, label=f"observed sd = {obs_sd:.2f}")
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


def fig_posterior_predictive(model, post, x, y) -> None:
    """The posterior predictive three ways, from the sampler's draws."""
    idx = np.random.default_rng(13).choice(post.shape[0], 150, replace=False)
    _predictive_panels(model, post, post[idx], x, y, "Posterior")


def fig_marginals_overlay(model, post, beta_true, sigma_true) -> None:
    """Prior against posterior for each parameter, so the contraction reads directly.

    Same colour per parameter; the prior is dashed and faded, the posterior solid
    and full, so the collapse from the wide prior onto the tight posterior shows
    even in greyscale. The scale is shown both as ``sigma`` and as ``u = log sigma``,
    the space the sampler works in.
    """
    from scipy.stats import gaussian_kde

    prior = model.prior_draws(40000, np.random.default_rng(10))
    prior = np.column_stack([prior, np.log(prior[:, 2])])
    posterior = np.column_stack([post, np.log(post[:, 2])])
    names = ["beta_0 (intercept)", "beta_1 (slope)", "sigma", "u = log sigma"]
    truth = [beta_true[0], beta_true[1], sigma_true, np.log(sigma_true)]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for k, ax in enumerate(axes):
        colour = P.param_colour(k)
        grid = np.linspace(
            min(prior[:, k].min(), posterior[:, k].min()),
            np.quantile(prior[:, k], 0.995),
            300,
        )
        ax.plot(grid, gaussian_kde(prior[:, k])(grid), color=colour, label="prior", **P.PRIOR_STYLE)
        ax.plot(grid, gaussian_kde(posterior[:, k])(grid), color=colour, label="posterior",
                **P.POSTERIOR_STYLE)
        ax.axvline(truth[k], color="#555555", ls=":", lw=1, label="true value")
        ax.set_title(names[k])
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Prior against posterior for each parameter: the contraction")
    fig.tight_layout()
    save(fig, "marginals_overlay.png")


def fig_posterior_marginals(model, post, beta_true, sigma_true) -> None:
    """Sampled marginals (filled) against exact closed-form draws (outline)."""
    ref = model.analytic_posterior_draws(40000, np.random.default_rng(12))
    post_u = np.column_stack([post, np.log(post[:, 2])])
    ref_u = np.column_stack([ref, np.log(ref[:, 2])])
    names = ["beta_0 (intercept)", "beta_1 (slope)", "sigma", "u = log sigma"]
    truth = [beta_true[0], beta_true[1], sigma_true, np.log(sigma_true)]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for k, ax in enumerate(axes):
        colour = P.param_colour(k)
        ax.hist(post_u[:, k], bins=70, density=True, alpha=0.55, color=colour,
                label="NUTS posterior")
        ax.hist(ref_u[:, k], bins=70, density=True, histtype="step", color="k", lw=1.1,
                label="analytic posterior")
        ax.axvline(truth[k], color="#555555", ls=":", lw=1, label="true value")
        ax.set_title(names[k])
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("Posterior of each parameter: sampled vs closed form")
    fig.tight_layout()
    save(fig, "posterior_marginals.png")


def fig_trace(draws) -> None:
    """Per-chain traces and the overlaid per-chain posteriors."""
    params = ["beta_0", "beta_1", "sigma"]
    chains = sorted(draws["chain"].unique())
    fig, axes = plt.subplots(len(params), 2, figsize=(11, 6),
                             gridspec_kw={"width_ratios": [2, 1]})
    for row, p in enumerate(params):
        for c in chains:
            colour = P.CHAIN_COLOURS[c % len(P.CHAIN_COLOURS)]
            d = draws[draws.chain == c]
            axes[row, 0].plot(d["draw"], d[p], lw=0.4, alpha=0.7, color=colour)
            axes[row, 1].hist(d[p], bins=40, density=True, histtype="step", alpha=0.8, color=colour)
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
    for vals, label, colour in [(marginal, "marginal energy", P.PARAM_COLOURS[0]),
                                (transitions, "energy transition", P.PARAM_COLOURS[4])]:
        density = gaussian_kde(vals)(grid)
        ax.fill_between(grid, density, color=colour, alpha=0.35)
        ax.plot(grid, density, color=colour, lw=1.5, label=label)
    ax.set(xlabel="energy (centred)", ylabel="density", yticks=[],
           title=f"energy overlay (E-BFMI {bfmi.min():.2f} to {bfmi.max():.2f})")
    ax.legend()
    save(fig, "energy.png")


def fig_loo_pit(model, post, y) -> dict:
    """In-sample LOO-PIT, the doc-preferred calibration check, with simultaneous bands.

    LOO-PIT corrects the optimism of evaluating the predictive on points that
    helped fit it. The histogram should be flat and the ECDF-minus-uniform curve
    should stay inside the simultaneous band, which controls the error across the
    whole curve rather than point by point.
    """
    pit, khat = loo_pit(y, model.X, post)
    xe, fe = empirical_cdf(pit)
    m = pit.size
    band = ecdf_simultaneous_band(m, prob=0.95, rng=np.random.default_rng(7))
    warm = P.POSTERIOR_DEEP

    nbins = 20
    counts, edges = np.histogram(pit, bins=nbins, range=(0, 1))
    centres = 0.5 * (edges[:-1] + edges[1:])
    expected = m / nbins
    bin_band = 1.96 * np.sqrt(m * (1 / nbins) * (1 - 1 / nbins))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(centres, counts, width=0.95 / nbins, color=warm, alpha=0.55)
    P.traffic_band(axes[0], expected - bin_band, expected + bin_band,
                   ylim=(0, max(counts.max(), expected + bin_band) * 1.1),
                   label="per-bin 95% band")
    axes[0].axhline(expected, color="k", ls="--", lw=1)
    axes[0].set(title="LOO-PIT histogram", xlabel="LOO-PIT value", ylabel="count")
    axes[0].legend()
    axes[1].plot(xe, fe - xe, color=warm, lw=1.4)
    P.traffic_band(axes[1], -band, band, label="95% simultaneous band")
    axes[1].axhline(0.0, color="k", ls="--", lw=1)
    axes[1].set(title="LOO-PIT ECDF minus uniform", xlabel="LOO-PIT value", ylabel="ECDF - uniform")
    axes[1].legend()
    fig.suptitle(f"LOO-PIT (max Pareto k = {np.nanmax(khat):.2f})")
    fig.tight_layout()
    save(fig, "loo_pit.png")
    return {"pit_mean": float(pit.mean()), "max_khat": float(np.nanmax(khat))}


def fig_coverage(model, post, x_test, y_test) -> None:
    """Empirical central-interval coverage from the held-out Rao-Blackwellised PIT."""
    X_test = np.column_stack([np.ones_like(x_test), x_test])
    pit = pit_rb(y_test, X_test, post)
    levels = np.linspace(0.1, 0.95, 18)
    cover = coverage_from_pit(pit, levels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(levels, cover, "o-", color=P.POSTERIOR_DEEP, label="empirical (held-out, RB)")
    ax.set(xlabel="nominal central probability", ylabel="empirical coverage",
           title="Quantile coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save(fig, "coverage.png")


def fig_sbc(ranks, n_sims, length) -> None:
    """SBC ranks as ECDF-minus-uniform per parameter, with simultaneous bands.

    The ECDF-difference form with a simultaneous band reads the same shapes as the
    rank histogram but controls the error across the whole curve. Under a calibrated
    sampler the curve stays inside the band.
    """
    names = ["beta_0", "beta_1", "sigma"]
    band = ecdf_simultaneous_band(n_sims, prob=0.95, rng=np.random.default_rng(9))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, ax in enumerate(axes):
        # Map ranks in 0..length to (0, 1) and read their ECDF against uniform.
        u = (ranks[:, k] + 0.5) / (length + 1)
        xe, fe = empirical_cdf(u)
        ax.plot(xe, fe - xe, color=P.param_colour(k), lw=1.4)
        P.traffic_band(ax, -band, band, ylim=(-band * 2.2, band * 2.2))
        ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set(title=names[k], xlabel="normalised rank")
    axes[0].set_ylabel("ECDF - uniform")
    fig.suptitle("SBC ranks: ECDF minus uniform with a 95% simultaneous band")
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
    fig_marginals_overlay(model, post, beta_true, sigma_true)
    fig_posterior_marginals(model, post, beta_true, sigma_true)
    fig_posterior_predictive(model, post, x, y)
    fig_trace(draws)
    fig_energy(stats)

    test = pd.read_parquet(RUN_DIR / "test_set.parquet")
    x_test, y_test = test["x_test"].to_numpy(), test["y_test"].to_numpy()
    ranks = pd.read_parquet(RUN_DIR / "sbc_ranks.parquet")[["beta_0", "beta_1", "sigma"]].to_numpy()
    meta = pd.read_parquet(RUN_DIR / "sbc_meta.parquet").iloc[0]

    loo = fig_loo_pit(model, post, y)
    fig_coverage(model, post, x_test, y_test)
    fig_sbc(ranks, int(meta["n_sims"]), int(meta["n_thinned"]))

    # Proper scores: LOO elpd on the training data, CRPS on the held-out set.
    elpd = loo_elpd(y, model.X, post)
    X_test = np.column_stack([np.ones_like(x_test), x_test])
    crps = float(crps_mixture(y_test, X_test, post).mean())

    # Headline numbers, quoted in the README prose.
    rhat = split_rhat(draws)
    ess = pd.DataFrame({"bulk": ess_bulk(draws), "tail": ess_tail(draws)})
    bfmi = ebfmi(stats.pivot(index="chain", columns="draw", values="energy").to_numpy())
    print("\n--- headline numbers ---")
    print("max split-Rhat:", round(float(rhat.max()), 4),
          "(", diagnostic_status("r_hat", float(rhat.max())), ")")
    print("min bulk ESS:", int(ess["bulk"].min()), " min tail ESS:", int(ess["tail"].min()))
    print("divergences:", int(divergence_summary(stats)["divergences"]))
    print("max tree depth reached:", int(stats["tree_depth"].max()))
    print("E-BFMI range:", round(float(bfmi.min()), 3), "to", round(float(bfmi.max()), 3))
    print("tree depth summary:\n", tree_depth_summary(stats).to_string())
    print(f"LOO-PIT mean {loo['pit_mean']:.3f} (uniform 0.5), max Pareto k {loo['max_khat']:.2f}")
    print(f"elpd_loo {float(elpd['elpd']):.1f} +/- {float(elpd['se']):.1f}, "
          f"p_loo {float(elpd['p_loo']):.2f}, mean CRPS {crps:.3f}")


if __name__ == "__main__":
    main()
