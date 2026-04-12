"""Deterministic seeding across Python, NumPy, and PyTorch.

Reproducibility in offline RL is non-negotiable: two runs with the same
config and the same seed MUST produce identical dataset rollouts, network
initializations, optimizer trajectories, and evaluation results. This module
centralizes the dance required to achieve that.

Caveats worth understanding before you trust ``set_seed``:

* **cuDNN determinism has a performance cost.** When
  ``deterministic_torch=True``, we set ``torch.backends.cudnn.deterministic``
  and disable cuDNN's autotuner. On GPUs this slows training by a few
  percent but is essential for bit-exact reproducibility.
* **Data-loader workers need their own seeds.** ``set_seed`` only seeds the
  main process. If you later add a ``DataLoader`` with ``num_workers > 0``,
  pass :func:`seed_worker` as its ``worker_init_fn``.
* **Non-deterministic CUDA ops exist.** A handful of CUDA kernels (e.g.
  certain scatter-add variants) are non-deterministic even under the flags
  above. We set ``CUBLAS_WORKSPACE_CONFIG`` and call
  ``torch.use_deterministic_algorithms(True, warn_only=True)`` to surface
  them as warnings rather than silently drifting.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["resolve_device", "seed_worker", "set_seed"]


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed every stochastic component used by the training stack.

    Parameters
    ----------
    seed:
        Non-negative integer. Applied to Python's ``random`` module, NumPy's
        global RNG, and PyTorch (CPU + all visible CUDA devices).
    deterministic_torch:
        If True, force PyTorch to use deterministic algorithms where
        possible. Slight perf hit; set to False for speed-over-repro runs.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Python-level RNGs.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch RNGs. Import lazily so that non-ML tests (e.g. pure-config tests)
    # don't pay the torch import cost, and so environments without torch
    # installed can still use the rest of ``utils.seed``.
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # cuBLAS workspace config is required for deterministic matmul in
        # CUDA ≥ 10.2; must be set before any CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: any op without a deterministic implementation
        # emits a warning instead of raising, so training doesn't crash but
        # the user is aware their run isn't bit-exact.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch versions don't support warn_only; fall back silently.
            pass


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for PyTorch ``DataLoader`` workers.

    Each worker derives its seed from ``torch.initial_seed()`` (set by the
    main process via :func:`set_seed`) so ``DataLoader`` parallelism remains
    deterministic. Pass to ``DataLoader(worker_init_fn=seed_worker, ...)``.
    """
    try:
        import torch
    except ImportError:
        return
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(preference: str) -> str:
    """Pick a torch device string from a user preference.

    ``preference`` may be:

    * ``"auto"`` — prefer ``cuda`` > ``mps`` > ``cpu``
    * ``"cpu"`` / ``"cuda"`` / ``"mps"`` — forced; the caller is asserting
      the hardware is available and we trust them (no silent fallback).
    """
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unknown device preference '{preference}'. "
            f"Expected one of: auto, cpu, cuda, mps."
        )
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # MPS detection is gated on platform; check carefully to avoid crashes
    # on older torch versions that don't expose ``torch.backends.mps``.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
