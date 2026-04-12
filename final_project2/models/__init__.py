"""Neural network architectures.

Populated in **Milestone 4**. Contains a shared ``MLP`` trunk and three
heads: ``QNetwork`` (state+action → ensemble of Q-values), ``ValueNetwork``
(state → V), and ``PolicyNetwork`` with pluggable Dirichlet / softmax_mse
heads selected via ``cfg.model.policy_head``.
"""
