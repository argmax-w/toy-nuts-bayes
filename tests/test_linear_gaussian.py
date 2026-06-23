"""End-to-end recovery on the conjugate regression and the MVN smoke run.

The acceptance criteria are checked against the analytic posterior moments.
"""

import numpy as np
import pytest
from scipy.special import polygamma

from toynuts.diagnostics import ebfmi, ess_bulk, ess_tail, mcse_mean, split_rhat
from toynuts.io import to_dataframes
from toynuts.models.linear_gaussian import LinearGaussian
from toynuts.models.multivariate_normal import MultivariateNormal
from toynuts.sampler import SamplerConfig, sample


@pytest.fixture(scope="module")
def mvn_run():
    """A friendly correlated 2-D normal, sampled with a covariance-matched metric."""
    mean = np.array([1.0, -1.0])
    cov = np.array([[1.0, 0.6], [0.6, 1.5]])
    model = MultivariateNormal(mean, cov)
    config = SamplerConfig(
        n_chains=4, n_draws=2000, step_size=0.9, metric=np.linalg.inv(cov), seed=1
    )
    run = sample(model, config)
    draws, stats, _ = to_dataframes(run, config, model)
    return model, run, draws, stats


@pytest.fixture(scope="module")
def regression_run():
    """Conjugate NIG regression with a metric set from the analytic posterior."""
    rng = np.random.default_rng(0)
    X, y = LinearGaussian.synthetic_data(60, [0.25, -0.35, 0.2], 1.0, rng)
    model = LinearGaussian(X, y, np.zeros(3), 0.15 * np.eye(3), 6.0, 5.0)
    moments = model.analytic_posterior_moments()
    # A fixed diagonal metric matched to the marginal posterior variances, with
    # the scale variance from the trigamma of the IG shape (Var[log sigma]).
    var_z = np.concatenate([np.diag(moments["beta_cov"]), [0.25 * polygamma(1, model.a_n)]])
    config = SamplerConfig(
        n_chains=4, n_draws=2000, step_size=0.8, metric=np.diag(1.0 / var_z), seed=2
    )
    run = sample(model, config)
    draws, stats, _ = to_dataframes(run, config, model)
    return model, run, draws, stats


def test_mvn_smoke_recovers_mean_and_covariance(mvn_run):
    """The MVN smoke target recovers its known mean and covariance, Rhat approx 1."""
    model, _, draws, stats = mvn_run
    moments = model.analytic_posterior_moments()
    assert (split_rhat(draws) < 1.01).all()
    assert stats["divergent"].sum() == 0

    samples = draws[model.param_names].to_numpy()
    np.testing.assert_allclose(samples.mean(axis=0), moments["mean"], atol=0.05)
    np.testing.assert_allclose(np.cov(samples.T), moments["cov"], atol=0.1)


def test_recovers_analytic_posterior_moments(regression_run):
    """Beta means and the mean of sigma**2 match the analytic NIG moments."""
    model, _, draws, _ = regression_run
    moments = model.analytic_posterior_moments()
    mcse = mcse_mean(draws)

    for i in range(model.p):
        name = f"beta_{i}"
        assert abs(draws[name].mean() - moments["beta_mean"][i]) < 5 * mcse[name]

    sigma2_est = (draws["sigma"].to_numpy() ** 2).mean()
    assert abs(sigma2_est - float(moments["sigma2_mean"])) < 0.05 * float(moments["sigma2_mean"])


def test_convergence_criteria(regression_run):
    """Split-Rhat below 1.01, bulk and tail ESS above 400 and zero divergences."""
    _, _, draws, stats = regression_run
    assert (split_rhat(draws) < 1.01).all()
    assert (ess_bulk(draws) > 400).all()
    assert (ess_tail(draws) > 400).all()
    assert stats["divergent"].sum() == 0


def test_energy_overlay_and_ebfmi(regression_run):
    """E-BFMI exceeds 0.9 and the two energy histograms have comparable spread."""
    _, run, _, _ = regression_run
    assert (ebfmi(run.energy) > 0.9).all()
    # "Largely coincide": the marginal-energy and energy-transition spreads match
    # to within a factor of two on this friendly target.
    centred = run.energy - run.energy.mean(axis=1, keepdims=True)
    transitions = np.diff(run.energy, axis=1)
    ratio = transitions.std() / centred.std()
    assert 0.5 < ratio < 2.0
