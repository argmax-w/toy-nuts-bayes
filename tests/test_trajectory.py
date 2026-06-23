"""The U-turn criterion, the base case, weight combination and divergence."""

import numpy as np
from scipy.special import logsumexp

from toynuts.hamiltonian import hamiltonian
from toynuts.integrators import leapfrog
from toynuts.models.multivariate_normal import MultivariateNormal
from toynuts.trajectory import build_tree, is_turning


def test_is_turning_false_when_moving_apart():
    """Ends moving apart (displacement aligned with both momenta) do not turn."""
    m_inv = np.eye(1)
    assert not is_turning(np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), m_inv)


def test_is_turning_true_when_folding_back():
    """A right momentum pointing back along the span counts as turning."""
    m_inv = np.eye(1)
    assert is_turning(np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([-1.0]), m_inv)


def test_base_case_is_single_leapfrog_step():
    """A depth-zero subtree is exactly one leapfrog step with collapsed edges."""
    model = MultivariateNormal(np.zeros(2), np.eye(2))
    rng = np.random.default_rng(0)
    m_inv = np.eye(2)
    q, p, eps = np.zeros(2), np.ones(2), 0.1
    h0 = hamiltonian(model, q, p, m_inv)

    tree = build_tree(model, q, p, +1, 0, h0, eps, m_inv, 1000.0, rng)

    q1, _ = leapfrog(model.grad_logp, q, p, eps, m_inv)
    assert tree.n_steps == 1
    np.testing.assert_allclose(tree.prop, q1)
    np.testing.assert_allclose(tree.minus[0], q1)
    np.testing.assert_allclose(tree.plus[0], q1)


def test_log_sum_exp_weight_combination():
    """A depth-one subtree's log weight is the log-sum-exp of its two leaf weights."""
    model = MultivariateNormal(np.zeros(1), np.array([[1.0]]))
    rng = np.random.default_rng(0)
    m_inv = np.eye(1)
    q, p, eps = np.array([0.2]), np.array([0.5]), 0.1
    h0 = hamiltonian(model, q, p, m_inv)

    tree = build_tree(model, q, p, +1, 1, h0, eps, m_inv, 1000.0, rng)

    q1, p1 = leapfrog(model.grad_logp, q, p, eps, m_inv)
    q2, p2 = leapfrog(model.grad_logp, q1, p1, eps, m_inv)
    h1 = hamiltonian(model, q1, p1, m_inv)
    h2 = hamiltonian(model, q2, p2, m_inv)
    np.testing.assert_allclose(tree.log_w, logsumexp([-h1, -h2]))


def test_divergence_flag_triggers():
    """A wildly overshooting step blows up the Hamiltonian and flags divergence."""
    model = MultivariateNormal(np.zeros(1), np.array([[1.0]]))
    rng = np.random.default_rng(0)
    m_inv = np.eye(1)
    q, p = np.array([50.0]), np.array([0.0])
    h0 = hamiltonian(model, q, p, m_inv)

    tree = build_tree(model, q, p, +1, 0, h0, 10.0, m_inv, 1000.0, rng)
    assert tree.diverging
