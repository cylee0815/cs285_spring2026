"""IQL training loop."""

from __future__ import annotations

from algorithms.iql import IQL
from utils.replay_buffer import ReplayBuffer


def train_iql(
    agent: IQL,
    buffer: ReplayBuffer,
    total_steps: int = 100_000,
    batch_size: int = 256,
    log_interval: int = 1000,
    log_fn: callable | None = None,
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

    Returns
    -------
    history:
        List of metric dicts, one per step.
    """
    history: list[dict[str, float]] = []

    for step in range(1, total_steps + 1):
        s, a, r, s_next, done = buffer.sample(batch_size)
        metrics = agent.update(s, a, r, s_next, done)
        history.append(metrics)

        if log_fn is not None:
            log_fn(step, metrics)
        elif step % log_interval == 0:
            v = metrics["v_loss"]
            q = metrics["q_loss"]
            p = metrics["policy_loss"]
            print(f"[step {step:>7d}]  v_loss={v:.4f}  q_loss={q:.4f}  policy_loss={p:.4f}")

    return history
