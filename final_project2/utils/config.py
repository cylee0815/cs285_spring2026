"""Typed configuration system for Portfolio IQL experiments.

Design goals
------------
* **Fully typed.** Every configurable parameter lives in a ``@dataclass`` so
  downstream code uses attribute access (``cfg.iql.expectile_tau``) and gets
  IDE completion + mypy support. No stringly-typed dict dereferences.
* **YAML-first.** Experiments are described by YAML files checked into
  ``experiments/configs/``; the loader turns them into strongly typed
  dataclass instances.
* **Command-line overrides.** Supports dotted overrides à la
  ``iql.expectile_tau=0.85 train.seed=7`` so ablations don't require a new
  YAML file per run. Override values are parsed as YAML scalars so numbers,
  booleans, lists, and nulls come through with the correct type.
* **Reproducibility.** Configs can be round-tripped (``load_config`` →
  ``dump_config``) so the *effective* hyperparameters of a run are
  preserved alongside its checkpoints, not just the original YAML file.

This module is deliberately ~200 lines of plain stdlib + PyYAML rather than
depending on Hydra or OmegaConf: it is transparent, auditable, and trivial
to extend when a new config section is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "BehaviorConfig",
    "Config",
    "DataConfig",
    "EvalConfig",
    "EnvConfig",
    "FeatureConfig",
    "IQLConfig",
    "LoggingConfig",
    "ModelConfig",
    "TrainConfig",
    "apply_overrides",
    "dump_config",
    "load_config",
]


# ---------------------------------------------------------------------------
# Dataclass schema
#
# NOTE: Fields have no defaults. The default experiment YAML is the source of
# truth; if a user writes an experiment file missing a key, loading fails
# loudly — this is the desired behavior for reproducible research.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Raw-data ingestion and walk-forward split boundaries."""

    tickers: list[str]
    start: str          # ISO date; first day of data window
    end: str            # ISO date; last day of data window
    train_end: str      # ISO date; last day of training split (inclusive)
    val_end: str        # ISO date; last day of validation split (inclusive)
    cache_dir: str      # directory for cached price parquet files
    frequency: str      # "daily" for now; "hourly" / "weekly" in future


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering parameters. All features must be CAUSAL."""

    lookback_returns: int
    volatility_window: int
    ma_windows: list[int]
    include_momentum: bool
    # Momentum feature windows (in trading days). Typical values:
    # 21 ≈ 1 month, 63 ≈ 3 months, 126 ≈ 6 months.
    momentum_windows: list[int]
    normalize: bool


@dataclass(frozen=True)
class EnvConfig:
    """Portfolio environment / MDP parameters."""

    episode_length: int
    transaction_cost: float
    max_weight: float
    reward_type: str      # "log_return" | "dsr"
    dsr_eta: float
    dsr_warmup: int


@dataclass(frozen=True)
class BehaviorConfig:
    """Offline dataset generation via behavior policies."""

    policies: list[str]
    episodes_per_policy: int
    dirichlet_alpha: float
    momentum_lookback: int
    risk_parity_lookback: int


@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture and policy-head selection."""

    hidden_sizes: list[int]
    activation: str
    policy_head: str      # "dirichlet" | "softmax_mse"
    q_ensemble_size: int
    dirichlet_alpha_floor: float


@dataclass(frozen=True)
class IQLConfig:
    """IQL-specific hyperparameters."""

    expectile_tau: float
    beta: float
    discount: float
    target_polyak: float
    adv_clip: float


@dataclass(frozen=True)
class TrainConfig:
    """Optimization / training-loop knobs."""

    steps: int
    batch_size: int
    lr_actor: float
    lr_critic: float
    lr_value: float
    grad_clip: float
    eval_interval: int
    log_interval: int
    save_interval: int
    device: str
    seed: int
    deterministic_torch: bool


@dataclass(frozen=True)
class EvalConfig:
    """Backtest metrics, baselines, and rolling-window configuration."""

    metrics: list[str]
    baselines: list[str]
    rolling_window_days: int
    risk_free_rate: float


@dataclass(frozen=True)
class LoggingConfig:
    """Experiment tracking + console logging."""

    wandb_project: str
    wandb_entity: str | None
    wandb_mode: str      # "online" | "offline" | "disabled"
    run_name: str | None
    results_dir: str
    console_level: str   # "debug" | "info" | "warning" | "error"


@dataclass(frozen=True)
class Config:
    """Root configuration bundle passed to every training / eval entrypoint."""

    data: DataConfig
    features: FeatureConfig
    env: EnvConfig
    behavior: BehaviorConfig
    model: ModelConfig
    iql: IQLConfig
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig


# ---------------------------------------------------------------------------
# YAML → dataclass machinery
# ---------------------------------------------------------------------------


T = TypeVar("T")


def _is_optional(hint: Any) -> bool:
    """Return True if ``hint`` is ``T | None`` (typing.Optional)."""
    if get_origin(hint) is None:
        return False
    args = get_args(hint)
    return type(None) in args


def _unwrap_optional(hint: Any) -> Any:
    """Return the non-None arm of a ``T | None`` hint."""
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if len(args) == 1 else hint


def _from_dict(cls: type[T], data: Any) -> T:
    """Recursively convert a nested dict into a dataclass instance.

    Nested dataclass fields are built from their sub-dicts. Scalar, list, and
    optional fields are taken as-is from PyYAML — which already yields
    native Python types for YAML scalars — so we do not aggressively coerce.
    We *do* raise a readable error when required keys are missing, to help
    experiment files fail loudly on typos.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a mapping to build {cls.__name__}, got {type(data).__name__}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for f in fields(cls):
        if f.name not in data:
            missing.append(f.name)
            continue
        value = data[f.name]
        hint = hints[f.name]
        if _is_optional(hint):
            if value is None:
                kwargs[f.name] = None
                continue
            hint = _unwrap_optional(hint)
        if is_dataclass(hint):
            kwargs[f.name] = _from_dict(hint, value)
        else:
            kwargs[f.name] = value
    if missing:
        raise KeyError(
            f"Missing required field(s) for {cls.__name__}: {missing}. "
            f"Check your YAML config against the dataclass schema."
        )
    # Catch unknown keys — these are almost always typos.
    known = {f.name for f in fields(cls)}
    unknown = [k for k in data if k not in known]
    if unknown:
        raise KeyError(
            f"Unknown key(s) in {cls.__name__}: {unknown}. "
            f"Allowed: {sorted(known)}"
        )
    return cls(**kwargs)  # type: ignore[call-arg]


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted CLI overrides to a raw config dict (in place).

    Each override has the form ``section.subkey=value`` (arbitrary nesting).
    The value is parsed as a YAML scalar, so ``seed=7`` becomes ``int(7)``,
    ``deterministic=false`` becomes ``False``, and ``tickers=[A,B]`` parses
    as a list. Strings with special characters should be quoted by the shell.

    Returns the same dict object for chaining.
    """
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(
                f"Invalid override '{raw}': expected 'dotted.key=value'."
            )
        key_path, _, raw_value = raw.partition("=")
        keys = key_path.strip().split(".")
        if not all(keys):
            raise ValueError(f"Invalid override key path '{key_path}'.")
        parsed_value = yaml.safe_load(raw_value)
        # Walk into nested dicts, creating missing intermediates as needed
        # (useful when a brand-new experimental flag is introduced).
        cursor: dict[str, Any] = data
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], dict):
                cursor[k] = {}
            cursor = cursor[k]
        cursor[keys[-1]] = parsed_value
    return data


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
) -> Config:
    """Load a YAML config file and build a typed :class:`Config` instance.

    Parameters
    ----------
    path:
        Filesystem path to a YAML config under ``experiments/configs/``.
    overrides:
        Optional list of ``section.key=value`` strings applied after the YAML
        file is parsed but before dataclass construction. Use for ablations.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    KeyError
        If the YAML is missing required fields or has unknown keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = apply_overrides(raw, list(overrides))
    return _from_dict(Config, raw)


def dump_config(cfg: Config, path: str | Path) -> None:
    """Serialize a :class:`Config` back to YAML.

    Used by the training entrypoint to snapshot the *effective* config
    (including CLI overrides) next to the run's checkpoints so experiments
    are bit-for-bit reproducible from artifacts alone.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, default_flow_style=False)
