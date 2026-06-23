"""Run the conjugate regression end to end, write a run and print a summary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import polygamma

from toynuts.diagnostics import divergence_summary, ebfmi, summary
from toynuts.io import to_dataframes, write_run
from toynuts.models.linear_gaussian import LinearGaussian
from toynuts.sampler import SamplerConfig, sample


def main() -> None:
    """Generate data, run NUTS, write the run and report against the posterior."""
    rng = np.random.default_rng(0)
    beta_true = np.array([0.25, -0.35, 0.2])
    X, y = LinearGaussian.synthetic_data(60, beta_true, 1.0, rng)
    model = LinearGaussian(X, y, np.zeros(3), 0.15 * np.eye(3), 6.0, 5.0)

    moments = model.analytic_posterior_moments()
    # Fixed diagonal metric matched to the marginal posterior variances; the
    # scale variance is Var[log sigma] = 0.25 * trigamma(a_n).
    var_z = np.concatenate([np.diag(moments["beta_cov"]), [0.25 * polygamma(1, model.a_n)]])
    config = SamplerConfig(
        n_chains=4, n_draws=2000, step_size=0.8, metric=np.diag(1.0 / var_z), seed=2
    )

    run = sample(model, config)
    draws, stats, run_config = to_dataframes(run, config, model)

    out = Path("outputs") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    write_run(out, draws, stats, run_config)
    # Exact i.i.d. reference draws, persisted so make_plots can overlay them.
    reference = model.analytic_posterior_draws(20000, np.random.default_rng(99))
    pd.DataFrame(reference, columns=model.param_names).to_parquet(out / "analytic_draws.parquet")

    print(f"run written to {out}\n")
    print(summary(draws, stats).to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(divergence_summary(stats).to_string())
    print("E-BFMI per chain:", np.round(ebfmi(run.energy).to_numpy(), 3), "\n")

    print(f"{'parameter':10s} {'sampled':>12s} {'analytic':>12s}")
    for i in range(model.p):
        name = f"beta_{i}"
        print(f"{name:10s} {draws[name].mean():12.4f} {moments['beta_mean'][i]:12.4f}")
    sigma2_mean = (draws["sigma"] ** 2).mean()
    print(f"{'sigma^2':10s} {sigma2_mean:12.4f} {float(moments['sigma2_mean']):12.4f}")


if __name__ == "__main__":
    main()
