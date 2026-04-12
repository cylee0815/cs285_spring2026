"""IQL training loop."""

from __future__ import annotations

from typing import Callable

from algorithms.iql import IQL
from utils.replay_buffer import ReplayBuffer


def train_iql(
    agent: IQL,
    buffer: ReplayBuffer,
    total_steps: int = 100_000,
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
        history.append(metrics)

        if log_fn is not None:
            log_fn(step, metrics)
        elif step % log_interval == 0:
            v = metrics["v_loss"]
            q = metrics["q_loss"]
            p = metrics["policy_loss"]
            print(f"[step {step:>7d}]  v_loss={v:.4f}  q_loss={q:.4f}  policy_loss={p:.4f}")

        if do_validate and step % validation_interval == 0:
            validation_fn(step)  # type: ignore[misc]

    return history
