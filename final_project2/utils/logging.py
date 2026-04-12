"""Unified experiment logger.

We avoid binding the training loop tightly to WandB: research code often
runs in three modes, and the logger must make all three ergonomic.

1. **Online WandB.** Full dashboards for real experiments.
2. **Offline WandB.** CI / air-gapped machines; artifacts sync later.
3. **Disabled.** Unit tests and quick iteration; nothing hits the network.

The :class:`Logger` class exposes a tiny surface area — ``log_metrics``,
``log_hparams``, ``close`` — so swapping backends (Tensorboard, ClearML,
neptune...) is a one-file change. In all modes, metrics are simultaneously
written to a JSONL file in ``results_dir`` so runs are auditable without
depending on any external service.

Note on naming: this module is called ``logging.py`` to match the
``CLAUDE.md`` spec. Absolute imports (PEP 328) mean ``import logging``
inside this file still resolves to the Python stdlib module, not this one.
"""

from __future__ import annotations

import json
import logging as stdlib_logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from utils.config import Config

__all__ = ["Logger", "get_console_logger"]


def get_console_logger(name: str, level: str = "info") -> stdlib_logging.Logger:
    """Return a ``logging.Logger`` wired to Rich for pretty console output.

    Idempotent: repeated calls with the same name return the same logger
    instance without stacking duplicate handlers.
    """
    logger = stdlib_logging.getLogger(name)
    if getattr(logger, "_portfolio_iql_configured", False):
        return logger

    logger.setLevel(_LEVELS[level.lower()])
    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_path=False,
        markup=False,
    )
    handler.setFormatter(stdlib_logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger._portfolio_iql_configured = True  # type: ignore[attr-defined]
    return logger


_LEVELS = {
    "debug": stdlib_logging.DEBUG,
    "info": stdlib_logging.INFO,
    "warning": stdlib_logging.WARNING,
    "error": stdlib_logging.ERROR,
}


class Logger:
    """Thin experiment logger with WandB + JSONL + console backends.

    Parameters
    ----------
    cfg:
        Fully constructed :class:`Config` from ``utils.config.load_config``.
        We accept the whole object (not just ``cfg.logging``) so the logger
        can snapshot hyperparameters atomically when the run starts.
    run_dir:
        Directory into which JSONL metrics and the effective-config snapshot
        will be written. Typically ``results/<run_name>/``.

    Example
    -------
    >>> from utils.config import load_config
    >>> from utils.logging import Logger
    >>> cfg = load_config("experiments/configs/default.yaml")
    >>> logger = Logger(cfg, run_dir="results/smoke")
    >>> logger.log_metrics({"loss/v": 0.123}, step=1)
    >>> logger.close()
    """

    def __init__(self, cfg: Config, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._console = get_console_logger(
            "portfolio_iql", level=cfg.logging.console_level
        )
        self._start_time = time.time()

        # --- JSONL metrics file -------------------------------------------
        # One JSON object per line; easy to grep, parse, and replot without
        # touching wandb. Opened in append mode so resumed runs don't clobber.
        self._jsonl_path = self.run_dir / "metrics.jsonl"
        self._jsonl_fp = self._jsonl_path.open("a", encoding="utf-8")

        # --- WandB backend ------------------------------------------------
        self._wandb_run = None
        mode = cfg.logging.wandb_mode
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                f"logging.wandb_mode must be one of "
                f"'online'/'offline'/'disabled', got '{mode}'"
            )
        if mode != "disabled":
            # Lazy import so 'disabled' mode (including unit tests) never
            # needs wandb installed or configured.
            try:
                import wandb
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "wandb is required for logging.wandb_mode != 'disabled'. "
                    "Install it with `uv add wandb` or set wandb_mode=disabled."
                ) from e
            self._wandb_run = wandb.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=cfg.logging.run_name,
                mode=mode,
                config=asdict(cfg),
                dir=str(self.run_dir),
            )

        self._console.info(
            "Logger initialized: run_dir=%s wandb=%s", self.run_dir, mode
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalar metrics to every active backend.

        JSONL receives ``{"step": step, "time": ..., **metrics}`` on one line.
        WandB receives the dict as-is. The console receives a concise
        one-line summary at INFO level.
        """
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics must be a dict, got {type(metrics).__name__}")

        record: dict[str, Any] = {
            "step": step,
            "elapsed_sec": time.time() - self._start_time,
            **metrics,
        }
        self._jsonl_fp.write(json.dumps(record, default=_json_default) + "\n")
        self._jsonl_fp.flush()

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

        # Console: keep short so it doesn't drown training output.
        summary = " ".join(f"{k}={_fmt(v)}" for k, v in sorted(metrics.items()))
        step_str = f"[step {step}] " if step is not None else ""
        self._console.info("%s%s", step_str, summary)

    def log_hparams(self, hparams: dict[str, Any]) -> None:
        """Record hyperparameters (once per run). Mirrored to console + JSONL."""
        record = {"event": "hparams", **hparams}
        self._jsonl_fp.write(json.dumps(record, default=_json_default) + "\n")
        self._jsonl_fp.flush()
        if self._wandb_run is not None:
            self._wandb_run.config.update(hparams, allow_val_change=True)

    def close(self) -> None:
        """Flush and release all backends. Safe to call multiple times."""
        if not self._jsonl_fp.closed:
            self._jsonl_fp.flush()
            self._jsonl_fp.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    # Context manager niceties — ``with Logger(...) as lg:`` cleans up on exit.
    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(o: Any) -> Any:
    """Best-effort JSON encoder for numpy scalars / tensors."""
    # Avoid importing numpy / torch at module load; this path only fires when
    # a tensor actually shows up in a metrics dict.
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist") and callable(o.tolist):
        try:
            return o.tolist()
        except Exception:
            pass
    return str(o)


def _fmt(v: Any) -> str:
    """Compact float formatting for console logging."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
