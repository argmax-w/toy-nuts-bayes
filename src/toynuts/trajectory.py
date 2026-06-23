"""No-U-turn criterion and the recursive multinomial tree builder.

The trajectory grows by doubling, every step of size ``eps``. The original
recursive U-turn criterion is checked for each balanced subtree, and merges use
uniform progressive selection (the outer loop in ``transition`` uses biased
selection instead). The recursion keeps only the two edge states and a single
sampled proposal per subtree, so trajectory memory is O(tree depth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from toynuts.hamiltonian import hamiltonian
from toynuts.integrators import leapfrog

if TYPE_CHECKING:
    from toynuts.models.base import Model


@dataclass
class Subtree:
    """One balanced subtree of the NUTS trajectory.

    Attributes:
        minus: Leftmost edge state ``(q, p)`` of the subtree.
        plus: Rightmost edge state ``(q, p)`` of the subtree.
        prop: The sampled proposal position from the subtree.
        log_w: Log of the summed multinomial weight ``exp(-H)`` over the subtree.
        sum_accept: Summed Metropolis acceptance over the subtree's leapfrog steps.
        n_steps: Number of leapfrog steps in the subtree.
        turning: Whether the subtree satisfies the U-turn criterion.
        diverging: Whether any state in the subtree diverged.
    """

    minus: tuple[np.ndarray, np.ndarray]
    plus: tuple[np.ndarray, np.ndarray]
    prop: np.ndarray
    log_w: float
    sum_accept: float
    n_steps: int
    turning: bool
    diverging: bool


def is_turning(
    q_minus: np.ndarray,
    p_minus: np.ndarray,
    q_plus: np.ndarray,
    p_plus: np.ndarray,
    m_inv: np.ndarray,
) -> bool:
    """Original no-U-turn criterion, generalised to the metric.

    With velocity ``v = M_inv p``, the span from minus to plus is turning when the
    displacement ``q_plus - q_minus`` points against the velocity at either end:
    once either projection goes negative the ends are folding back towards each
    other and extending further wastes work.

    Args:
        q_minus: Position at the left edge.
        p_minus: Momentum at the left edge.
        q_plus: Position at the right edge.
        p_plus: Momentum at the right edge.
        m_inv: Inverse mass matrix.

    Returns:
        True if the span is turning.
    """
    delta = q_plus - q_minus
    return bool((delta @ (m_inv @ p_minus) < 0) or (delta @ (m_inv @ p_plus) < 0))


def build_tree(
    model: Model,
    q: np.ndarray,
    p: np.ndarray,
    direction: int,
    depth: int,
    h0: float,
    eps: float,
    m_inv: np.ndarray,
    delta_max: float,
    rng: np.random.Generator,
) -> Subtree:
    """Recursively build one balanced subtree by doubling.

    The base case is a single leapfrog step. The recursive case builds two equal
    subtrees, joins them with uniform progressive selection and propagates the
    turning and divergence flags, short-circuiting as soon as the left half stops.

    Args:
        model: Target model exposing ``grad_logp`` and ``logp``.
        q: Position at the active edge to extend from.
        p: Momentum at the active edge to extend from.
        direction: ``-1`` to extend backwards, ``+1`` forwards.
        depth: Subtree depth; the subtree spans ``2**depth`` leapfrog steps.
        h0: Hamiltonian at the start of the iteration, for the divergence test.
        eps: Leapfrog step size.
        m_inv: Inverse mass matrix.
        delta_max: Divergence threshold on the Hamiltonian error.
        rng: Random generator for progressive selection.

    Returns:
        The assembled subtree.
    """
    if depth == 0:
        # Base case: a single leapfrog step in the requested direction. The step
        # size carries the sign, so a backward step is just a negative eps.
        q1, p1 = leapfrog(model.grad_logp, q, p, direction * eps, m_inv)
        h1 = hamiltonian(model, q1, p1, m_inv)
        diverging = bool((h1 - h0) > delta_max)
        # Weight is exp(-H); the acceptance is min(1, exp(H0 - H1)), written
        # through exp(min(0, .)) to avoid overflow when the step lowers energy.
        log_w = -h1
        sum_accept = float(np.exp(min(0.0, h0 - h1)))
        state = (q1, p1)
        return Subtree(state, state, q1, log_w, sum_accept, 1, False, diverging)

    left = build_tree(model, q, p, direction, depth - 1, h0, eps, m_inv, delta_max, rng)
    if left.diverging or left.turning:
        # Stop early and propagate the flag; the partial subtree carries it up.
        return left

    if direction == -1:
        edge_q, edge_p = left.minus
        right = build_tree(
            model, edge_q, edge_p, direction, depth - 1, h0, eps, m_inv, delta_max, rng
        )
        minus, plus = right.minus, left.plus
    else:
        edge_q, edge_p = left.plus
        right = build_tree(
            model, edge_q, edge_p, direction, depth - 1, h0, eps, m_inv, delta_max, rng
        )
        minus, plus = left.minus, right.plus

    log_w = logsumexp([left.log_w, right.log_w])
    # Uniform progressive selection: adopt the right proposal with probability
    # w_right / (w_left + w_right), in logs to stay stable.
    if np.log(rng.uniform()) < right.log_w - log_w:
        prop = right.prop
    else:
        prop = left.prop

    turning = bool(right.turning or is_turning(minus[0], minus[1], plus[0], plus[1], m_inv))
    return Subtree(
        minus,
        plus,
        prop,
        log_w,
        left.sum_accept + right.sum_accept,
        left.n_steps + right.n_steps,
        turning,
        right.diverging,
    )
