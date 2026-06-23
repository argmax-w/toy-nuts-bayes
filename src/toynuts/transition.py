"""A single NUTS transition: momentum refresh, tree expansion, state selection.

The outer loop adds one doubling at a time using biased progressive selection,
which favours newer states and improves mixing, and stops on a U-turn, a
divergence or ``max_tree_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import logsumexp

from toynuts.hamiltonian import hamiltonian, sample_momentum
from toynuts.trajectory import build_tree, is_turning

if TYPE_CHECKING:
    from toynuts.models.base import Model


@dataclass
class DrawResult:
    """Outcome of one NUTS step.

    Attributes:
        q: The selected position for this draw, in unconstrained space.
        energy: Start-of-iteration Hamiltonian ``H0`` after the momentum refresh.
        tree_depth: Tree depth reached.
        n_leapfrog: Number of leapfrog steps taken.
        divergent: Whether the trajectory diverged.
        accept_stat: Mean Metropolis acceptance over the trajectory.
    """

    q: np.ndarray
    energy: float
    tree_depth: int
    n_leapfrog: int
    divergent: bool
    accept_stat: float


def nuts_step(
    model: Model,
    q0: np.ndarray,
    eps: float,
    metric: np.ndarray,
    max_tree_depth: int,
    delta_max: float,
    rng: np.random.Generator,
) -> DrawResult:
    """Advance one NUTS step from ``q0``.

    Refreshes the momentum, grows the trajectory by biased progressive doubling
    and returns the selected state with its per-draw statistics. The recorded
    energy is ``H0``, the Hamiltonian at the start of the iteration with the
    freshly refreshed momentum, which feeds both the energy overlay and E-BFMI.

    Args:
        model: Target model.
        q0: Starting position in unconstrained space.
        eps: Leapfrog step size.
        metric: Mass matrix ``M``.
        max_tree_depth: Doubling cap.
        delta_max: Divergence threshold on the Hamiltonian error.
        rng: Random generator.

    Returns:
        The draw result for this step.
    """
    # The metric is fixed, so this inverse is the same every draw; for the small
    # friendly targets here recomputing it is cheaper than complicating the API.
    m_inv = np.linalg.inv(metric)
    p0 = sample_momentum(metric, rng)
    h0 = hamiltonian(model, q0, p0, m_inv)
    energy = h0

    minus = (q0, p0)
    plus = (q0, p0)
    sample = q0
    log_w = -h0  # weight of the initial point, the first candidate
    depth = 0
    n_steps = 0
    sum_accept = 0.0
    divergent = False

    while depth < max_tree_depth:
        direction = -1 if rng.uniform() < 0.5 else 1
        if direction == -1:
            tree = build_tree(model, minus[0], minus[1], -1, depth, h0, eps, m_inv, delta_max, rng)
            minus = tree.minus
        else:
            tree = build_tree(model, plus[0], plus[1], +1, depth, h0, eps, m_inv, delta_max, rng)
            plus = tree.plus

        n_steps += tree.n_steps
        sum_accept += tree.sum_accept

        if tree.diverging:
            divergent = True
            break

        # Biased selection: adopt the new subtree's proposal with probability
        # min(1, w_new / w_old), so later states are favoured. A turning subtree
        # is never sampled from.
        if not tree.turning:
            if np.log(rng.uniform()) < tree.log_w - log_w:
                sample = tree.prop

        log_w = logsumexp([log_w, tree.log_w])

        # Stop on the new subtree turning or the whole trajectory turning.
        if tree.turning or is_turning(minus[0], minus[1], plus[0], plus[1], m_inv):
            break
        depth += 1

    accept_stat = sum_accept / max(n_steps, 1)
    return DrawResult(
        q=sample,
        energy=float(energy),
        tree_depth=int(depth),
        n_leapfrog=int(n_steps),
        divergent=bool(divergent),
        accept_stat=float(accept_stat),
    )
