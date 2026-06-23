"""Transform round-trip, the Jacobian and the analytic gradient.

The gradient check is the key correctness test for the hand-derived analytic
gradients. It needs the regression model, so it is left until
``models/linear_gaussian.py`` exists.
"""

import numpy as np

from toynuts.models.linear_gaussian import LinearGaussian
from toynuts.transforms import log_abs_det_jacobian, to_constrained, to_unconstrained


def test_round_trip():
    """``sigma -> u -> sigma`` and ``u -> sigma -> u`` recover the original."""
    for sigma in [1e-3, 0.5, 1.23, 10.0, 1e3]:
        np.testing.assert_allclose(to_constrained(to_unconstrained(sigma)), sigma, rtol=1e-12)
    for u in [-5.0, -0.3, 0.0, 1.7, 4.0]:
        np.testing.assert_allclose(to_unconstrained(to_constrained(u)), u, rtol=1e-12, atol=1e-12)


def test_log_abs_det_jacobian_matches_finite_difference():
    """log|d(sigma**2)/du| matches a central difference of sigma**2 as a function of u.

    The Jacobian is for the ``sigma**2`` to ``u`` map, so the transform being
    differentiated is ``s(u) = to_constrained(u)**2``. Matching here, constant and
    all, is what pins down the ``log(2)`` the model is free to drop.
    """
    h = 1e-5
    for u in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        s_plus = to_constrained(u + h) ** 2
        s_minus = to_constrained(u - h) ** 2
        fd = (s_plus - s_minus) / (2 * h)
        np.testing.assert_allclose(log_abs_det_jacobian(u), np.log(abs(fd)), atol=1e-6)


def test_log_abs_det_jacobian_pins_the_constant():
    """At u = 0 the term is exactly log(2), the constant kept here but dropped in the model."""
    np.testing.assert_allclose(log_abs_det_jacobian(0.0), np.log(2.0), rtol=1e-12)


def test_grad_logp_matches_finite_difference():
    """The model ``grad_logp`` matches a central difference of ``logp``.

    This is the key analytic-gradient check: it exercises the beta gradient, the
    u gradient and the Jacobian term together against the unconstrained logp.
    """
    rng = np.random.default_rng(0)
    X, y = LinearGaussian.synthetic_data(40, [1.0, -2.0, 0.5], 1.0, rng)
    model = LinearGaussian(X, y, np.zeros(3), 10.0 * np.eye(3), 2.0, 2.0)

    z = np.array([0.8, -1.5, 0.4, 0.1])
    analytic = model.grad_logp(z)
    h = 1e-6
    numeric = np.array(
        [(model.logp(z + h * e) - model.logp(z - h * e)) / (2 * h) for e in np.eye(z.size)]
    )
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)
