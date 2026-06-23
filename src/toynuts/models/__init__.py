"""Target models for the sampler, written on the unconstrained space.

``MultivariateNormal`` is the friendly smoke target run before the regression.
``LinearGaussian`` is the conjugate Normal-Inverse-Gamma regression whose
posterior is known in closed form and serves as the exact reference.
"""

from __future__ import annotations
