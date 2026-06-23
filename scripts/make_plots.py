"""Render the diagnostic figures for a run with matplotlib only.

Reads a run directory (the latest under ``outputs/`` by default, or a path given
on the command line) and writes the figures into a ``figures/`` subdirectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from toynuts.diagnostics import tree_depth_summary
from toynuts.io import read_run


def _latest_run() -> Path:
    """The most recent run directory under ``outputs/``."""
    runs = sorted(Path("outputs").glob("run_*"))
    if not runs:
        raise SystemExit("no runs found in outputs/; run scripts/run_linear_gaussian.py first")
    return runs[-1]


def main() -> None:
    """Read a run and render its figures into the run directory."""
    plt.switch_backend("Agg")
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run()
    draws, stats, _ = read_run(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    params = [c for c in draws.columns if c not in ("chain", "draw")]
    chains = sorted(draws["chain"].unique())
    analytic_path = run_dir / "analytic_draws.parquet"
    analytic = pd.read_parquet(analytic_path) if analytic_path.exists() else None

    # Trace plots, one row per parameter.
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 2.2 * len(params)), squeeze=False)
    for ax, p in zip(axes[:, 0], params, strict=True):
        for c in chains:
            d = draws[draws.chain == c]
            ax.plot(d["draw"], d[p], lw=0.5, alpha=0.7)
        ax.set_ylabel(p)
    axes[-1, 0].set_xlabel("draw")
    fig.suptitle("traces")
    fig.tight_layout()
    fig.savefig(fig_dir / "trace.png", dpi=110)
    plt.close(fig)

    # Rank plots: pooled ranks should be uniform within each chain when mixed.
    fig, axes = plt.subplots(len(params), 1, figsize=(9, 2.2 * len(params)), squeeze=False)
    for ax, p in zip(axes[:, 0], params, strict=True):
        ranks = sps.rankdata(draws[p].to_numpy())
        for c in chains:
            ax.hist(ranks[(draws.chain == c).to_numpy()], bins=30, histtype="step")
        ax.set_ylabel(p)
    fig.suptitle("rank plots")
    fig.tight_layout()
    fig.savefig(fig_dir / "rank.png", dpi=110)
    plt.close(fig)

    # Marginal posteriors with the analytic marginal overlaid.
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3), squeeze=False)
    for ax, p in zip(axes[0], params, strict=True):
        ax.hist(draws[p], bins=60, density=True, alpha=0.6, label="sampled")
        if analytic is not None and p in analytic:
            ax.hist(
                analytic[p], bins=60, density=True, histtype="step", color="k", label="analytic"
            )
        ax.set_title(p)
    axes[0, 0].legend()
    fig.suptitle("marginal posteriors")
    fig.tight_layout()
    fig.savefig(fig_dir / "marginals.png", dpi=110)
    plt.close(fig)

    # Pair plot over the parameters.
    k = len(params)
    arr = draws[params].to_numpy()
    fig, axes = plt.subplots(k, k, figsize=(2.2 * k, 2.2 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(arr[:, i], bins=40)
            else:
                ax.scatter(arr[:, j], arr[:, i], s=2, alpha=0.08)
            if i == k - 1:
                ax.set_xlabel(params[j])
            if j == 0:
                ax.set_ylabel(params[i])
    fig.suptitle("pairs")
    fig.tight_layout()
    fig.savefig(fig_dir / "pairs.png", dpi=110)
    plt.close(fig)

    # Energy overlay: marginal energy against energy transitions.
    marginal, transitions = [], []
    for c in chains:
        e = stats[stats.chain == c].sort_values("draw")["energy"].to_numpy()
        marginal.append(e - e.mean())
        transitions.append(np.diff(e))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.concatenate(marginal), bins=60, density=True, alpha=0.6, label="marginal energy")
    ax.hist(
        np.concatenate(transitions), bins=60, density=True, histtype="step", color="k",
        label="energy transitions",
    )
    ax.legend()
    ax.set_title("energy overlay")
    fig.tight_layout()
    fig.savefig(fig_dir / "energy.png", dpi=110)
    plt.close(fig)

    # Tree-depth histogram.
    depth = tree_depth_summary(stats)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(depth.index.astype(int), depth.to_numpy())
    ax.set_xlabel("tree depth")
    ax.set_ylabel("proportion")
    ax.set_title("tree depth")
    fig.tight_layout()
    fig.savefig(fig_dir / "tree_depth.png", dpi=110)
    plt.close(fig)

    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
