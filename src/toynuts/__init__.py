"""toy-nuts: a from-scratch No-U-Turn Sampler in pure Python.

The package is laid out so that each stage of a NUTS transition lives in its own
module: ``transforms`` and ``hamiltonian`` for the geometry, ``integrators`` for
the leapfrog step, ``trajectory`` for the recursive tree, ``transition`` for one
step, ``sampler`` for the multi-chain driver, ``diagnostics`` for the from-scratch
convergence statistics and ``io`` for the Parquet data model.
"""

from __future__ import annotations

__version__ = "0.1.0"
