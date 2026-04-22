"""Import + instantiation smoke test for every offline-RL agent in ``offline_rl``.

This script does **not** train or evaluate anything. It imports each agent
class, loads its config factory, builds a tiny ``NStepReplayBuffer``, and
instantiates the agent with ``(obs_dim, action_dim, config, device,
offline_buffer=buffer)``. Any ImportError / TypeError / missing-attribute is
reported so we can fix compatibility issues before wiring the suite into the
real training pipeline.

Usage:
    uv run python scripts/test_offline_agents_import.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Ensure the project root is on sys.path so ``offline_rl`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from offline_rl.agents.replay_buffer import NStepReplayBuffer  # noqa: E402


# Must match run_offline.py's AGENT_MAP exactly.
AGENT_MAP = {
    "bc":                      ("offline_rl.agents.bc", "BCAgent"),
    "fisher_bc":               ("offline_rl.agents.fisher_bc", "FisherBCAgent"),
    "td3_bc":                  ("offline_rl.agents.td3_bc", "TD3BCAgent"),
    "awac":                    ("offline_rl.agents.awac", "AWACAgent"),
    "cql_vanilla":             ("offline_rl.agents.cql_vanilla", "VanillaCQLAgent"),
    "iql":                     ("offline_rl.agents.iql", "IQLAgent"),
    "edac":                    ("offline_rl.agents.edac", "EDACAgent"),
    "bcq":                     ("offline_rl.agents.bcq", "BCQAgent"),
    "mbpo":                    ("offline_rl.agents.mbpo", "MBPOAgent"),
    "mopo":                    ("offline_rl.agents.mopo", "MOPOAgent"),
    "decision_transformer":    ("offline_rl.agents.decision_transformer", "DecisionTransformerAgent"),
    "trajectory_transformer":  ("offline_rl.agents.trajectory_transformer", "TrajectoryTransformerAgent"),
}


# Match the portfolio problem: 8 assets, small obs_dim, tiny buffer.
OBS_DIM = 32
ACTION_DIM = 8
BUFFER_CAP = 256
DEVICE = torch.device("cpu")


def _load_config(name: str):
    mod = importlib.import_module(f"offline_rl.configs.{name}_config")
    return mod.get_config()


def _build_buffer() -> NStepReplayBuffer:
    buf = NStepReplayBuffer(
        BUFFER_CAP, OBS_DIM, ACTION_DIM, DEVICE,
        seq_len=20, n_step=1, gamma=0.99,
    )
    return buf


def main() -> int:
    results: list[tuple[str, str, str]] = []  # (name, status, detail)
    buffer = _build_buffer()

    for name, (mod_path, cls_name) in AGENT_MAP.items():
        try:
            config = _load_config(name)
        except Exception as e:  # noqa: BLE001
            results.append((name, "CONFIG_FAIL", f"{type(e).__name__}: {e}"))
            continue

        try:
            mod = importlib.import_module(mod_path)
            AgentClass = getattr(mod, cls_name)
        except Exception as e:  # noqa: BLE001
            results.append((name, "IMPORT_FAIL", f"{type(e).__name__}: {e}"))
            continue

        try:
            agent = AgentClass(
                OBS_DIM, ACTION_DIM, config, DEVICE, offline_buffer=buffer,
            )
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=2)
            results.append((name, "INIT_FAIL", f"{type(e).__name__}: {e}\n{tb}"))
            continue

        # Spot-check a few attributes the training loop relies on.
        has_update = hasattr(agent, "update") and callable(agent.update)
        has_get_action = hasattr(agent, "get_action") and callable(agent.get_action)
        detail = f"update={has_update} get_action={has_get_action}"
        status = "OK" if (has_update and has_get_action) else "PARTIAL"
        results.append((name, status, detail))

    # Report.
    width = max(len(n) for n, *_ in results)
    print(f"\n{'algorithm'.ljust(width)}  status       detail")
    print(f"{'-' * width}  -----------  ------")
    n_ok = n_partial = n_fail = 0
    for name, status, detail in results:
        print(f"{name.ljust(width)}  {status.ljust(11)}  {detail.splitlines()[0]}")
        if status == "OK":
            n_ok += 1
        elif status == "PARTIAL":
            n_partial += 1
        else:
            n_fail += 1

    print(f"\n{n_ok}/{len(results)} agents OK  |  {n_partial} partial  |  {n_fail} failed")

    # Print full tracebacks for failures at the bottom.
    for name, status, detail in results:
        if status not in ("OK", "PARTIAL"):
            print(f"\n--- {name} ({status}) ---\n{detail}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
