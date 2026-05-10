"""Helpers to assemble offline behavior-policy mixtures for `load_from_env`.

`ReplayBuffer.load_from_env` expects each policy to be a callable
``(obs) -> action``. The behavior policies in :mod:`policies.behavior` come
in two flavors:

  * Stateless / obs-driven: ``EqualWeightPolicy``, ``DirichletPolicy`` —
    expose ``get_action(state)``.
  * History-driven: ``MomentumPolicy``, ``RiskParityPolicy`` — expose
    ``get_action_from_returns(history)`` and need access to the env's
    forward-returns track. Wrapping these requires reading ``env._t`` and
    ``env._forward_returns`` (Gymnasium wrappers forward attribute access
    via ``__getattr__``).

This module hides that asymmetry so callers can build a single uniform list
of ``(callable, weight)`` tuples to hand to ``load_from_env``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from policies.behavior import (
    DirichletPolicy,
    EqualWeightPolicy,
    MomentumPolicy,
    RiskParityPolicy,
)

__all__ = [
    "make_episode_callable",
    "default_offline_mixture",
]


def make_episode_callable(policy_obj: Any, env: Any) -> Any:
    """Wrap a behavior policy as a ``(obs) -> action`` callable.

    Stateless policies are wrapped trivially. History-driven policies read
    ``env._t`` and ``env._forward_returns`` directly to build the lookback
    window — this matches the slicing logic in
    ``data.build_dataset.build_dataset_from_env`` (warmup -> equal weight,
    then policy-specific weights using the realized returns up to step t-1).
    """
    n_assets = env.action_space.shape[0]
    equal = np.ones(n_assets, dtype=np.float64) / n_assets

    if isinstance(policy_obj, (EqualWeightPolicy, DirichletPolicy)):
        def _call(obs):
            return np.asarray(policy_obj.get_action(obs), dtype=np.float64)
        return _call

    if isinstance(policy_obj, (MomentumPolicy, RiskParityPolicy)):
        def _call(_obs):
            t = int(getattr(env, "_t", 0))
            fwd = getattr(env, "_forward_returns", None)
            start = int(getattr(env, "_start", 0))
            if fwd is None or t < 2:
                return equal.copy()
            lookback = int(getattr(policy_obj, "lookback", 60))
            lo = max(0, start + t - lookback)
            hi = start + t
            history = fwd[lo:hi]
            if history.shape[0] < 2:
                return equal.copy()
            return np.asarray(
                policy_obj.get_action_from_returns(history), dtype=np.float64
            )
        return _call

    if callable(policy_obj):
        return policy_obj
    raise TypeError(f"Unsupported behavior policy type: {type(policy_obj).__name__}")


def default_offline_mixture(env: Any, seed: int = 0) -> list[tuple[Any, float]]:
    """Canonical 4-policy mixture used to populate offline datasets.

    Returns a list suitable for ``ReplayBuffer.load_from_env(policy_mixture=...)``:
    Dirichlet, EqualWeight, Momentum, RiskParity, equally weighted. The
    Dirichlet RNG seed is the only stochastic component; momentum/RP/EW are
    deterministic given the env.
    """
    n_assets = env.action_space.shape[0]
    pols = [
        DirichletPolicy(n_assets=n_assets, alpha=1.0, seed=seed),
        EqualWeightPolicy(n_assets=n_assets),
        MomentumPolicy(n_assets=n_assets, lookback=60),
        RiskParityPolicy(n_assets=n_assets, lookback=60),
    ]
    return [(make_episode_callable(p, env), 0.25) for p in pols]
