"""Multi-chain NUTS driver with a fixed step size and metric, no adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from toynuts.hamiltonian import hamiltonian, sample_momentum
from toynuts.integrators import leapfrog
from toynuts.transition import nuts_step

if TYPE_CHECKING:
    from toynuts.models.base import Model


@dataclass
class SamplerConfig:
    """Fixed inputs for a run. Nothing here is adapted.

    Attributes:
        n_chains: Number of chains.
        n_draws: Draws per chain.
        step_size: Leapfrog step ``eps``, constant for the run.
        metric: Mass matrix ``M``. ``None`` is taken as the identity. A 1-D array
            is read as the diagonal of ``M``.
        max_tree_depth: Doubling cap.
        delta_max: Divergence threshold on the Hamiltonian error.
        seed: Parent seed; per-chain seeds are spawned from one ``SeedSequence``.
        init: Initialisation strategy for the overdispersed chain starts. The
            default draws each start from the model's ``initial_point``.
        n_burnin: Non-adaptive burn-in draws, run with the same fixed step size
            and metric and then discarded. This only removes the initialisation
            transient from an overdispersed start; nothing is tuned, so the
            adaptive warm-up of section 1 stays out of scope.
    """

    n_chains: int
    n_draws: int
    step_size: float
    metric: np.ndarray | None = None
    max_tree_depth: int = 10
    delta_max: float = 1000.0
    seed: int = 0
    init: str = "prior"
    n_burnin: int = 500


@dataclass
class RunResult:
    """Per-draw output of a run, arrays with leading ``(chain, draw)`` axes.

    Attributes:
        positions: Sampled positions, shape ``(n_chains, n_draws, dim)`` in the
            unconstrained ``z`` space.
        logp: Log density per draw.
        accept_stat: Mean acceptance per draw.
        tree_depth: Tree depth per draw.
        n_leapfrog: Leapfrog steps per draw.
        divergent: Divergence flag per draw.
        energy: Start-of-iteration Hamiltonian per draw.
        step_size: The fixed step size used.
        config: The configuration the run was produced with.
    """

    positions: np.ndarray
    logp: np.ndarray
    accept_stat: np.ndarray
    tree_depth: np.ndarray
    n_leapfrog: np.ndarray
    divergent: np.ndarray
    energy: np.ndarray
    step_size: float
    config: SamplerConfig


def _resolve_metric(metric: np.ndarray | None, dim: int) -> np.ndarray:
    """Turn the configured metric into a dense mass matrix."""
    if metric is None:
        return np.eye(dim)
    metric = np.asarray(metric, dtype=float)
    if metric.ndim == 1:
        return np.diag(metric)
    return metric


def sample(model: Model, config: SamplerConfig) -> RunResult:
    """Run all chains from overdispersed inits and collect per-draw arrays.

    Each chain gets its own RNG spawned from one ``SeedSequence`` so the run is
    reproducible. Per-draw arrays are preallocated and filled in place; no
    adaptation happens.

    Args:
        model: Target model.
        config: The fixed run configuration.

    Returns:
        The assembled run result.
    """
    dim = model.dim
    metric = _resolve_metric(config.metric, dim)
    n_chains, n_draws = config.n_chains, config.n_draws
    seeds = np.random.SeedSequence(config.seed).spawn(n_chains)

    positions = np.empty((n_chains, n_draws, dim))
    logp = np.empty((n_chains, n_draws))
    accept_stat = np.empty((n_chains, n_draws))
    tree_depth = np.empty((n_chains, n_draws), dtype=int)
    n_leapfrog = np.empty((n_chains, n_draws), dtype=int)
    divergent = np.empty((n_chains, n_draws), dtype=bool)
    energy = np.empty((n_chains, n_draws))

    for c in range(n_chains):
        rng = np.random.default_rng(seeds[c])
        z = model.initial_point(rng)
        # Non-adaptive burn-in: take the chain off its overdispersed start before
        # recording, with the same fixed step size and metric. Nothing is tuned.
        for _ in range(config.n_burnin):
            z = nuts_step(
                model, z, config.step_size, metric, config.max_tree_depth, config.delta_max, rng
            ).q
        for d in range(n_draws):
            result = nuts_step(
                model, z, config.step_size, metric, config.max_tree_depth, config.delta_max, rng
            )
            z = result.q
            positions[c, d] = z
            logp[c, d] = model.logp(z)
            accept_stat[c, d] = result.accept_stat
            tree_depth[c, d] = result.tree_depth
            n_leapfrog[c, d] = result.n_leapfrog
            divergent[c, d] = result.divergent
            energy[c, d] = result.energy

    return RunResult(
        positions, logp, accept_stat, tree_depth, n_leapfrog, divergent, energy,
        float(config.step_size), config,
    )


def find_reasonable_epsilon(
    model: Model,
    z0: np.ndarray,
    metric: np.ndarray,
    rng: np.random.Generator,
    max_iter: int = 100,
) -> float:
    """Heuristic starting step size, run by hand and never inside the loop.

    Doubles or halves ``eps`` until the acceptance probability of a single
    leapfrog step crosses 0.5. Used to choose a value to hardcode.

    Args:
        model: Target model.
        z0: A starting position.
        metric: Mass matrix ``M``.
        rng: Random generator.
        max_iter: Safety cap on the doubling and halving loop.

    Returns:
        A reasonable step size.
    """
    m_inv = np.linalg.inv(metric)
    eps = 1.0
    p0 = sample_momentum(metric, rng)
    h0 = hamiltonian(model, z0, p0, m_inv)
    q1, p1 = leapfrog(model.grad_logp, z0, p0, eps, m_inv)
    log_accept = h0 - hamiltonian(model, q1, p1, m_inv)
    # Head towards 0.5 acceptance: grow eps if we are above it, shrink if below.
    a = 1.0 if log_accept > np.log(0.5) else -1.0
    for _ in range(max_iter):
        if a * log_accept <= a * np.log(0.5):
            break
        eps *= 2.0**a
        q1, p1 = leapfrog(model.grad_logp, z0, p0, eps, m_inv)
        log_accept = h0 - hamiltonian(model, q1, p1, m_inv)
    return eps
