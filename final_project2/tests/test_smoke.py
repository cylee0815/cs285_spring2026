"""Milestone 0 smoke tests.

These tests verify the scaffolding is wired correctly end-to-end:

* every top-level module listed in CLAUDE.md is importable
* the default YAML config loads into the typed dataclass schema
* dotted CLI overrides round-trip with correct types (int stays int, etc.)
* unknown config keys and missing keys raise readable errors
* :func:`utils.seed.set_seed` makes Python / NumPy / torch RNGs deterministic
* :class:`utils.logging.Logger` instantiates in ``disabled`` mode with zero
  network side-effects and writes a JSONL metrics record

Real algorithmic tests (environment transitions, dataset generation, IQL
loss shapes, metric correctness) live alongside their implementations in
later milestones.
"""

from __future__ import annotations

import importlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.config import Config, apply_overrides, dump_config, load_config
from utils.logging import Logger
from utils.seed import resolve_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "data",
        "core.envs",
        "features",
        "models",
        "algorithms",
        "training",
        "evaluation",
        "utils",
        "utils.config",
        "utils.seed",
        "utils.logging",
    ],
)
def test_module_importable(module_name: str) -> None:
    """Every top-level package listed in CLAUDE.md must import cleanly."""
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# 2. Config loading
# ---------------------------------------------------------------------------


def test_default_config_exists() -> None:
    assert DEFAULT_CONFIG.exists(), f"Missing default config at {DEFAULT_CONFIG}"


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert isinstance(cfg, Config)


def test_default_config_values() -> None:
    """Sanity-check the key invariants baked into the default experiment."""
    cfg = load_config(DEFAULT_CONFIG)

    # Asset universe (orthogonal 8-ETF basket, data starts 2008 per UUP/DBC).
    assert cfg.data.tickers == ["SPY", "EEM", "TLT", "HYG", "DBC", "GLD", "UUP", "SHY"]
    assert cfg.data.start == "2008-01-01"
    assert cfg.data.train_end < cfg.data.val_end < cfg.data.end

    # IQL defaults
    assert cfg.iql.discount == pytest.approx(0.99)
    assert 0.5 <= cfg.iql.expectile_tau <= 1.0
    assert cfg.iql.beta > 0
    assert 0.0 < cfg.iql.target_polyak <= 1.0

    # Model defaults chosen in the planning discussion.
    assert cfg.model.policy_head == "dirichlet"
    assert cfg.model.q_ensemble_size == 2
    assert cfg.model.hidden_sizes == [256, 256]

    # Env defaults
    assert cfg.env.reward_type == "log_return"
    assert cfg.env.transaction_cost >= 0
    assert 0 < cfg.env.max_weight <= 1

    # Training hyperparameters are sane.
    assert cfg.train.batch_size > 0
    assert cfg.train.steps > 0
    assert cfg.train.grad_clip > 0
    assert cfg.train.deterministic_torch is True

    # Logging defaults disable wandb so tests/CI never touch the network.
    assert cfg.logging.wandb_mode == "disabled"


def test_config_missing_key_raises(tmp_path: Path) -> None:
    """Dropping a required field should fail loudly, not silently default."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    del raw["iql"]["discount"]
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError, match="discount"):
        load_config(bad_path)


def test_config_unknown_key_raises(tmp_path: Path) -> None:
    """Typos must be surfaced, not silently ignored."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["iql"]["nonexistent_hyperparam"] = 1.0
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError, match="nonexistent_hyperparam"):
        load_config(bad_path)


# ---------------------------------------------------------------------------
# 3. CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides_scalar_types() -> None:
    cfg = load_config(
        DEFAULT_CONFIG,
        overrides=[
            "iql.expectile_tau=0.85",
            "train.seed=7",
            "train.deterministic_torch=false",
            "logging.run_name=smoke_run",
        ],
    )
    assert cfg.iql.expectile_tau == pytest.approx(0.85)
    assert cfg.train.seed == 7
    assert cfg.train.deterministic_torch is False
    assert cfg.logging.run_name == "smoke_run"


def test_cli_overrides_list_value() -> None:
    """YAML parsing of override values should preserve list types."""
    cfg = load_config(
        DEFAULT_CONFIG,
        overrides=["features.ma_windows=[10, 30]"],
    )
    assert cfg.features.ma_windows == [10, 30]


def test_cli_override_invalid_form() -> None:
    with pytest.raises(ValueError, match="Invalid override"):
        apply_overrides({}, ["not_a_valid_override"])


# ---------------------------------------------------------------------------
# 4. Config round-trip
# ---------------------------------------------------------------------------


def test_config_dump_roundtrip(tmp_path: Path) -> None:
    """dump_config + load_config must be lossless for the default file."""
    cfg = load_config(DEFAULT_CONFIG)
    out = tmp_path / "effective.yaml"
    dump_config(cfg, out)
    cfg2 = load_config(out)
    assert asdict(cfg) == asdict(cfg2)


# ---------------------------------------------------------------------------
# 5. Seeding determinism
# ---------------------------------------------------------------------------


def test_set_seed_python_and_numpy_determinism() -> None:
    set_seed(42, deterministic_torch=False)
    py_a = [random.random() for _ in range(5)]
    np_a = np.random.rand(5)

    set_seed(42, deterministic_torch=False)
    py_b = [random.random() for _ in range(5)]
    np_b = np.random.rand(5)

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)


def test_set_seed_torch_determinism() -> None:
    torch = pytest.importorskip("torch")
    set_seed(123, deterministic_torch=True)
    a = torch.randn(4, 4)
    set_seed(123, deterministic_torch=True)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)


def test_set_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)


def test_resolve_device_known_values() -> None:
    assert resolve_device("cpu") == "cpu"
    # 'auto' must return something from the allowed set.
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_resolve_device_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


# ---------------------------------------------------------------------------
# 6. Logger in disabled mode
# ---------------------------------------------------------------------------


def test_logger_disabled_mode_writes_jsonl(tmp_path: Path) -> None:
    """Disabled mode must not touch wandb AND must still persist metrics."""
    cfg = load_config(DEFAULT_CONFIG)
    run_dir = tmp_path / "run"
    with Logger(cfg, run_dir=run_dir) as lg:
        lg.log_hparams({"lr": 3e-4})
        lg.log_metrics({"loss/v": 0.123, "loss/q": 0.456}, step=1)
        lg.log_metrics({"loss/v": 0.100, "loss/q": 0.400}, step=2)

    jsonl = run_dir / "metrics.jsonl"
    assert jsonl.exists()
    lines = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    # 1 hparams record + 2 metric records
    assert len(lines) == 3
    assert lines[0]["event"] == "hparams"
    assert lines[0]["lr"] == 3e-4
    assert lines[1]["step"] == 1
    assert lines[1]["loss/v"] == pytest.approx(0.123)
    assert lines[2]["step"] == 2
