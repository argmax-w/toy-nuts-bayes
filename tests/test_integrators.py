"""Leapfrog reversibility and short-horizon energy conservation."""

import numpy as np

from toynuts.hamiltonian import hamiltonian
from toynuts.integrators import leapfrog
from toynuts.models.multivariate_normal import MultivariateNormal


def test_reversibility():
    """Integrating forward then, with negated momentum, forward again returns the start."""
    rng = np.random.default_rng(0)
    model = MultivariateNormal(np.array([0.0, 0.0]), np.array([[1.0, 0.3], [0.3, 1.0]]))
    m_inv = np.eye(2)
    q0 = rng.normal(size=2)
    p0 = rng.normal(size=2)
    eps = 0.1

    q, p = q0.copy(), p0.copy()
    for _ in range(20):
        q, p = leapfrog(model.grad_logp, q, p, eps, m_inv)
    p = -p
    for _ in range(20):
        q, p = leapfrog(model.grad_logp, q, p, eps, m_inv)
    p = -p

    np.testing.assert_allclose(q, q0, atol=1e-10)
    np.testing.assert_allclose(p, p0, atol=1e-10)


def test_energy_conservation_short_horizon():
    """Energy stays close to its start over a short horizon at small ``eps``."""
    model = MultivariateNormal(np.array([0.0]), np.array([[1.0]]))
    m_inv = np.array([[1.0]])
    q = np.array([0.5])
    p = np.array([0.3])
    h0 = hamiltonian(model, q, p, m_inv)
    eps = 0.01

    worst = 0.0
    for _ in range(200):
        q, p = leapfrog(model.grad_logp, q, p, eps, m_inv)
        worst = max(worst, abs(hamiltonian(model, q, p, m_inv) - h0))

    assert worst < 1e-2
