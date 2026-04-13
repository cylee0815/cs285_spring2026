"""IQL training loop.

The loop is intentionally thin: it pulls batches from the replay buffer,
calls ``agent.update``, and forwards the returned losses to whatever
logger the caller hooks up.

On top of the raw losses, this module also computes a small set of
**diagnostic** metrics (``q_mean``, ``q_std``, ``advantage_mean``,
``policy_entropy``) from a second forward pass on the same minibatch.
This leaves ``algorithms.iql`` untouched — per the project's hard
constraint that the core update rules must not change — while still
giving the training script enough signal to populate WandB and the
milestone-report tearsheet.
"""

from __future__ import annotations

from typing import Callable

import torch

from algorithms.iql import IQL
from utils.replay_buffer import ReplayBuffer


__all__ = ["train_iql", "compute_diagnostics"]


@torch.no_grad()
def compute_diagnostics(
    agent: IQL,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, float]:
    """Compute read-only training diagnostics on ``(states, actions)``.

    Returns a dict with the keys:

    * ``q_mean``, ``q_std`` — statistics of Q(s, a) on this batch.
    * ``advantage_mean``     — mean of (Q(s, a) - V(s)) on this batch.
    * ``policy_entropy``     — mean entropy H(pi(.|s)) of the softmax
      portfolio-weight distribution produced by the policy network.

    No gradients are taken and no agent parameters are modified.
    """
    q = agent.q_network(states, actions)           # (B, 1)
    v = agent.value_network(states)                # (B, 1)
    advantage = q - v                               # (B, 1)
    probs = agent.policy_network(states)            # (B, N)
    # Softmax -> valid probability distribution; clamp for log safety.
    probs_c = probs.clamp(min=1e-12)
    entropy = -(probs_c * probs_c.log()).sum(dim=-1)  # (B,)
    return {
        "q_mean": float(q.mean().item()),
        "q_std": float(q.std(unbiased=False).item()),
        "advantage_mean": float(advantage.mean().item()),
        "policy_entropy": float(entropy.mean().item()),
    }


def train_iql(
    agent: IQL,
    buffer: ReplayBuffer,
    total_steps: int = 20_000,
    batch_size: int = 256,
    log_interval: int = 1000,
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
    validation_fn: Callable[[int], None] | None = None,
    validation_interval: int | None = None,
) -> list[dict[str, float]]:
    """Run the IQL training loop.

    Parameters
    ----------
    agent:
        An initialised :class:`IQL` instance.
    buffer:
        A :class:`ReplayBuffer` loaded with the offline dataset.
    total_steps:
        Number of gradient steps.
    batch_size:
        Minibatch size per step.
    log_interval:
        Print losses every ``log_interval`` steps.
    log_fn:
        Optional callback ``(step, metrics_dict) -> None`` for custom logging.
    validation_fn:
        Optional callback invoked every ``validation_interval`` steps as
        ``validation_fn(step)``. Used by the caller to run a backtest on
        the held-out validation window and update best-model tracking.
    validation_interval:
        Step period between validation calls. Ignored if
        ``validation_fn`` is ``None``.

    Returns
    -------
    history:
        List of metric dicts, one per step.
    """
    history: list[dict[str, float]] = []
    do_validate = validation_fn is not None and validation_interval and validation_interval > 0

    for step in range(1, total_steps + 1):
        s, a, r, s_next, done = buffer.sample(batch_size)
        metrics = agent.update(s, a, r, s_next, done)

        # Attach read-only diagnostics for logging. We do this every step
        # because the cost is small (two forward passes, no backward) and
        # keeping the metric stream aligned 1-to-1 with `history` makes
        # downstream aggregation simpler.
        metrics.update(compute_diagnostics(agent, s, a))

        history.append(metrics)

        if log_fn is not None:
            log_fn(step, metrics)
        elif step % log_interval == 0:
            v = metrics["v_loss"]
            q = metrics["q_loss"]
            p = metrics["policy_loss"]
            print(f"[step {step:>7d}]  v_loss={v:.3e}  q_loss={q:.3e}  policy_loss={p:.4f}")

        if do_validate and step % validation_interval == 0:
            validation_fn(step)  # type: ignore[misc]

    return history
