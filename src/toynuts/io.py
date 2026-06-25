"""Assemble the run DataFrames and round-trip them as Parquet via pyarrow.

A run writes three frames to ``outputs/run_<timestamp>/``: ``draws.parquet``,
``sample_stats.parquet`` and ``run_config.parquet``. The draws schema is kept
general, one column per entry of ``model.param_names``, so the same writer serves
the smoke target and the regression.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from toynuts.sampler import _resolve_metric

if TYPE_CHECKING:
    from toynuts.models.base import Model
    from toynuts.sampler import RunResult, SamplerConfig


def versions() -> dict[str, str]:
    """Record the software environment so a run can be reproduced exactly.

    Captures the Python and core library versions alongside the seeds and sampler
    settings already in the run config, per the reproducibility section of the
    workflow. Libraries that are absent are simply skipped.

    Returns:
        A mapping of ``version_<name>`` to a version string.
    """
    out = {"version_python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "pyarrow", "matplotlib"):
        try:
            module = __import__(name)
            out[f"version_{name}"] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return out


def to_dataframes(
    run_result: RunResult,
    config: SamplerConfig,
    model: Model,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the draws, sample_stats and run_config frames.

    The positions are mapped back to the constrained parameters for storage, so
    for the regression the draws frame carries ``beta_i`` and ``sigma`` rather
    than the unconstrained ``u``.

    Args:
        run_result: The per-draw arrays from a run.
        config: The run configuration.
        model: The model, used for parameter names and the constraining map.

    Returns:
        The ``(draws_df, sample_stats_df, run_config_df)`` triple.
    """
    rr = run_result
    n_chains, n_draws, dim = rr.positions.shape
    names = model.param_names

    flat = rr.positions.reshape(-1, dim)
    constrained = np.array([model.to_constrained(z) for z in flat])
    chain_idx = np.repeat(np.arange(n_chains), n_draws)
    draw_idx = np.tile(np.arange(n_draws), n_chains)

    draws = {"chain": chain_idx, "draw": draw_idx}
    for j, name in enumerate(names):
        draws[name] = constrained[:, j]
    draws_df = pd.DataFrame(draws)

    sample_stats_df = pd.DataFrame(
        {
            "chain": chain_idx,
            "draw": draw_idx,
            "logp": rr.logp.ravel(),
            "accept_stat": rr.accept_stat.ravel(),
            "step_size": np.full(n_chains * n_draws, rr.step_size),
            "tree_depth": rr.tree_depth.ravel(),
            "n_leapfrog": rr.n_leapfrog.ravel(),
            "divergent": rr.divergent.ravel(),
            "energy": rr.energy.ravel(),
        }
    )

    metric_diag = np.diag(_resolve_metric(config.metric, dim))
    cfg: dict[str, object] = {
        "seed": config.seed,
        "n_chains": config.n_chains,
        "n_draws": config.n_draws,
        "n_burnin": config.n_burnin,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "delta_max": config.delta_max,
        "dim": dim,
    }
    for i, value in enumerate(metric_diag):
        cfg[f"metric_diag_{i}"] = float(value)
    cfg.update(versions())
    run_config_df = pd.DataFrame([cfg])

    return draws_df, sample_stats_df, run_config_df


def write_run(
    path: str | Path,
    draws_df: pd.DataFrame,
    sample_stats_df: pd.DataFrame,
    run_config_df: pd.DataFrame,
) -> None:
    """Write the three Parquet files under ``path``.

    Args:
        path: Run directory, created if it does not exist.
        draws_df: The draws frame.
        sample_stats_df: The sample-stats frame.
        run_config_df: The single-row run-config frame.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    draws_df.to_parquet(directory / "draws.parquet")
    sample_stats_df.to_parquet(directory / "sample_stats.parquet")
    run_config_df.to_parquet(directory / "run_config.parquet")


def read_run(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three Parquet files back from ``path``.

    Args:
        path: Run directory written by ``write_run``.

    Returns:
        The ``(draws_df, sample_stats_df, run_config_df)`` triple.
    """
    directory = Path(path)
    return (
        pd.read_parquet(directory / "draws.parquet"),
        pd.read_parquet(directory / "sample_stats.parquet"),
        pd.read_parquet(directory / "run_config.parquet"),
    )
