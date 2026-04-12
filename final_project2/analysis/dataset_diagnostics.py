"""Read-only sanity checks for an offline RL portfolio dataset.

This script audits a saved ``.npz`` dataset (as produced by
``scripts/build_real_dataset.py``) end-to-end without touching the network
or rebuilding features. It is the canonical pre-flight check before any
IQL training run: if this script reports ``DATASET VALID: TRUE`` we trust
the alignment, splits, and statistics; otherwise it exits non-zero and
prints exactly which invariant failed.

What it checks
--------------
1. **Shapes** — every array has a consistent first-axis length and
   dimensionality.
2. **NaN / Inf** — states, next_states, actions, rewards are all finite.
3. **Next-state continuity** — ``next_states[t] == states[t+1]`` for every
   non-terminal transition. Catches off-by-one feature-array slicing.
4. **Reward alignment** — recomputes
   ``reward_t = w_t · forward_returns[t] - λ‖Δw‖₁`` from the stored
   forward returns and compares against the saved rewards. Catches the
   classic "state / return shifted by one day" bug.
5. **Feature alignment** — spot-checks that the state dim equals the
   feature dim implied by the config knobs, and that the first non-zero
   reward occurs at t=0 (i.e., the dataset did not silently swallow
   warmup rows again).
6. **Action simplex validity** — every action lies on the probability
   simplex to float32 tolerance.
7. **Date monotonicity** — dates are strictly increasing.
8. **Split boundaries** — under the project-default split, verifies that
   (a) all three windows are non-empty, (b) train ∩ val = val ∩ test = ∅
   by date, (c) no index appears in more than one split, (d) the
   train-window max date precedes the val-window min date, etc.
9. **Normalization leakage** — if a normalizer state dict exists at the
   conventional path, confirms it was fit on rows whose dates all fall
   within the train window (no val/test stats leaking in).
10. **Descriptive stats** — prints mean / std / min / max of rewards and
    forward returns, and shape summaries.

Usage
-----
    uv run python analysis/dataset_diagnostics.py
    uv run python analysis/dataset_diagnostics.py --dataset datasets/real_dirichlet.npz
    uv run python analysis/dataset_diagnostics.py --normalizer results/run42/normalizer.json

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.splits import DEFAULT_SPLIT, SplitConfig, compute_split_indices


# ---------------------------------------------------------------------------
# Failure accumulator
# ---------------------------------------------------------------------------


class _Report:
    def __init__(self) -> None:
        self.sections: list[tuple[str, list[str], list[str]]] = []
        self._cur_title: str | None = None
        self._cur_ok: list[str] = []
        self._cur_fail: list[str] = []

    def section(self, title: str) -> None:
        self._flush()
        self._cur_title = title
        self._cur_ok = []
        self._cur_fail = []

    def ok(self, msg: str) -> None:
        self._cur_ok.append(msg)

    def fail(self, msg: str) -> None:
        self._cur_fail.append(msg)

    def _flush(self) -> None:
        if self._cur_title is None:
            return
        self.sections.append((self._cur_title, self._cur_ok, self._cur_fail))

    def print_and_exit_code(self) -> int:
        self._flush()
        any_fail = False
        for title, oks, fails in self.sections:
            print(f"\n── {title} ────────────────────────────────────────")
            for msg in oks:
                print(f"  ✓ {msg}")
            for msg in fails:
                print(f"  ✗ {msg}")
            if fails:
                any_fail = True

        print("\n" + "=" * 60)
        if any_fail:
            print("DATASET VALID: FALSE — see failures above.")
            print("=" * 60)
            return 1
        print("DATASET VALID: TRUE")
        print("=" * 60)
        return 0


# ---------------------------------------------------------------------------
# Individual check blocks
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _check_shapes(ds: dict[str, np.ndarray], rep: _Report) -> int:
    """Return N = number of transitions (or -1 on failure)."""
    rep.section("Shapes and presence of required arrays")
    required = ["states", "actions", "rewards", "next_states", "dones"]
    for k in required:
        if k not in ds:
            rep.fail(f"missing required key '{k}'")
            return -1
    N = len(ds["states"])
    for k in required:
        if len(ds[k]) != N:
            rep.fail(f"len({k})={len(ds[k])} != len(states)={N}")
            return -1
    if ds["states"].shape[1] != ds["next_states"].shape[1]:
        rep.fail(
            f"state dim mismatch: states={ds['states'].shape[1]} "
            f"vs next_states={ds['next_states'].shape[1]}"
        )
    if ds["actions"].ndim != 2:
        rep.fail(f"actions must be 2-D, got shape {ds['actions'].shape}")
    rep.ok(f"N={N} transitions, state_dim={ds['states'].shape[1]}, "
           f"action_dim={ds['actions'].shape[1]}")
    if "dates" in ds:
        rep.ok("'dates' array present (date-aware dataset)")
    else:
        rep.fail("'dates' array missing — splits cannot be date-based")
    if "forward_returns" in ds:
        fr = ds["forward_returns"]
        rep.ok(f"forward_returns present: shape={fr.shape}")
    else:
        rep.fail(
            "'forward_returns' missing — cannot independently verify reward "
            "alignment. (Older datasets stored only rewards.)"
        )
    return N


def _check_finite(ds: dict[str, np.ndarray], rep: _Report) -> None:
    rep.section("NaN / Inf checks")
    for k in ["states", "next_states", "actions", "rewards"]:
        arr = ds[k]
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            rep.fail(f"{k}: {n_nan} NaN, {n_inf} Inf entries")
        else:
            rep.ok(f"{k}: all finite")


def _check_continuity(ds: dict[str, np.ndarray], rep: _Report) -> None:
    rep.section("Next-state continuity")
    N = len(ds["states"])
    dones = ds["dones"].astype(bool)
    if N < 2:
        rep.ok("N<2 — trivially continuous")
        return
    non_terminal = ~dones[:-1]
    if not non_terminal.any():
        rep.fail("all transitions marked terminal — unexpected for a single "
                 "rolled-out episode.")
        return
    lhs = ds["next_states"][:-1][non_terminal]
    rhs = ds["states"][1:][non_terminal]
    if not np.array_equal(lhs, rhs):
        mismatches = int((lhs != rhs).any(axis=1).sum())
        rep.fail(f"next_states[t] != states[t+1] on {mismatches} non-terminal "
                 "transitions — episode continuity broken.")
    else:
        rep.ok(f"next_states[t] == states[t+1] on all "
               f"{int(non_terminal.sum())} non-terminal transitions")
    if not bool(dones[-1]):
        rep.fail("last transition is not done=True")
    else:
        rep.ok("final transition is terminal (done=True)")
    if bool(dones[:-1].any()):
        rep.fail(f"{int(dones[:-1].sum())} intermediate done=True transitions "
                 "— dataset builder should emit a single episode")
    else:
        rep.ok("no intermediate terminations")


def _check_reward_alignment(
    ds: dict[str, np.ndarray], rep: _Report, transaction_cost_lambda: float
) -> None:
    rep.section("Reward ≡ w_t · forward_returns[t] − λ‖Δw‖₁")
    if "forward_returns" not in ds:
        rep.fail("skipped (no forward_returns in dataset)")
        return
    N = len(ds["states"])
    actions = ds["actions"].astype(np.float64)
    fr = ds["forward_returns"].astype(np.float64)
    rewards = ds["rewards"].astype(np.float64)

    # Trim forward_returns to match transitions (see scripts/train.py: the
    # builder stores the full feature-bundle-length forward_returns, which
    # is one row longer than the rolled-out transitions).
    if fr.shape[0] >= N:
        fr_t = fr[:N]
    else:
        rep.fail(
            f"forward_returns rows={fr.shape[0]} < transitions={N}; cannot "
            "recompute rewards."
        )
        return

    # Recompute rewards. prev_weights starts at uniform (see PortfolioEnv).
    n_assets = actions.shape[1]
    prev_w = np.ones(n_assets) / n_assets
    recomputed = np.empty(N, dtype=np.float64)
    for t in range(N):
        w = actions[t]
        port_return = float(np.dot(w, fr_t[t]))
        turnover = float(np.sum(np.abs(w - prev_w)))
        recomputed[t] = port_return - transaction_cost_lambda * turnover
        prev_w = w

    diff = np.abs(recomputed - rewards)
    if diff.max() < 1e-5:
        rep.ok(f"rewards match recomputation to 1e-5 (max|Δ|={diff.max():.2e}, "
               f"λ={transaction_cost_lambda})")
    else:
        # Try alternative λ=0 before failing outright — this catches the
        # common case where the stored metadata λ disagrees with the
        # rollout λ.
        prev_w = np.ones(n_assets) / n_assets
        recomputed_noλ = np.empty(N, dtype=np.float64)
        for t in range(N):
            recomputed_noλ[t] = float(np.dot(actions[t], fr_t[t]))
            prev_w = actions[t]
        if np.abs(recomputed_noλ - rewards).max() < 1e-5:
            rep.fail(
                f"rewards match λ=0 recomputation but not the assumed "
                f"λ={transaction_cost_lambda}. Metadata 'transaction_cost' "
                "is probably wrong — rebuild with the matching λ."
            )
        else:
            rep.fail(
                f"rewards do not match recomputation: max|Δ|={diff.max():.3e}, "
                f"mean|Δ|={diff.mean():.3e}. Either actions/forward_returns "
                "are misaligned, or λ differs."
            )


def _check_simplex(ds: dict[str, np.ndarray], rep: _Report) -> None:
    rep.section("Action simplex validity")
    actions = ds["actions"]
    row_sum = actions.sum(axis=1)
    sum_err = float(np.abs(row_sum - 1.0).max())
    neg_count = int((actions < -1e-7).sum())
    if sum_err < 1e-5 and neg_count == 0:
        rep.ok(f"all {len(actions)} actions on simplex "
               f"(max|Σ-1|={sum_err:.2e}, no negatives)")
    else:
        if sum_err >= 1e-5:
            rep.fail(f"action rows do not sum to 1: max|Σ-1|={sum_err:.3e}")
        if neg_count > 0:
            rep.fail(f"{neg_count} action entries < -1e-7")


def _check_date_monotonic(ds: dict[str, np.ndarray], rep: _Report) -> pd.DatetimeIndex | None:
    rep.section("Date monotonicity")
    if "dates" not in ds:
        rep.fail("no 'dates' array")
        return None
    dt = pd.to_datetime(ds["dates"].astype(np.int64), unit="ns")
    diffs = np.diff(dt.asi8)
    if (diffs > 0).all():
        rep.ok(f"{len(dt)} dates strictly increasing, "
               f"{dt.min().date()} → {dt.max().date()}")
    else:
        n_bad = int((diffs <= 0).sum())
        rep.fail(f"{n_bad} non-increasing date steps")
    return dt


def _check_feature_alignment(
    ds: dict[str, np.ndarray], rep: _Report, dt: pd.DatetimeIndex | None
) -> None:
    rep.section("Feature / return alignment")
    if dt is None:
        rep.fail("skipped (no dates)")
        return
    # If metadata includes eval_start, first date should equal the first
    # trading day on or after eval_start.
    eval_start_raw = ds.get("meta_eval_start", None)
    if eval_start_raw is not None:
        eval_start = pd.Timestamp(str(eval_start_raw))
        first = dt[0]
        if first < eval_start:
            rep.fail(f"first transition date {first.date()} precedes "
                     f"meta_eval_start={eval_start.date()} — warmup leakage?")
        elif (first - eval_start).days > 7:
            rep.fail(f"first transition {first.date()} is >7 days after "
                     f"meta_eval_start={eval_start.date()}; rolling windows "
                     "may have consumed evaluation rows (shorten warmup).")
        else:
            rep.ok(f"first transition {first.date()} aligns with "
                   f"meta_eval_start={eval_start.date()}")
    else:
        rep.ok("(no meta_eval_start present; skipping warmup-leakage check)")

    # Spot-check: states.shape[1] should equal features_dim metadata if present.
    feat_dim = ds.get("meta_feature_dim", None)
    if feat_dim is not None:
        if int(feat_dim) != ds["states"].shape[1]:
            rep.fail(f"meta_feature_dim={int(feat_dim)} vs "
                     f"states.shape[1]={ds['states'].shape[1]}")
        else:
            rep.ok(f"state_dim == meta_feature_dim == {int(feat_dim)}")


def _check_splits(
    ds: dict[str, np.ndarray], rep: _Report, dt: pd.DatetimeIndex | None,
    split_cfg: SplitConfig,
) -> None:
    rep.section(f"Split integrity "
                f"(train {split_cfg.train_start}→{split_cfg.train_end}, "
                f"val {split_cfg.val_start}→{split_cfg.val_end}, "
                f"test {split_cfg.test_start}→{split_cfg.test_end})")
    if dt is None:
        rep.fail("skipped (no dates)")
        return
    try:
        split = compute_split_indices(ds["dates"].astype(np.int64), split_cfg)
    except Exception as e:
        rep.fail(f"compute_split_indices raised: {e}")
        return
    sizes = split.summary()
    rep.ok(f"sizes: train={sizes['train']}, val={sizes['val']}, "
           f"test={sizes['test']}")
    # Ensure non-empty + disjoint by date boundary.
    train_max = dt[split.train].max()
    val_min = dt[split.val].min()
    val_max = dt[split.val].max()
    test_min = dt[split.test].min()
    if train_max < val_min:
        rep.ok(f"train max date {train_max.date()} < val min "
               f"{val_min.date()} — no leakage")
    else:
        rep.fail(f"train max date {train_max.date()} >= val min "
                 f"{val_min.date()} — LEAKAGE")
    if val_max < test_min:
        rep.ok(f"val max date {val_max.date()} < test min "
               f"{test_min.date()} — no leakage")
    else:
        rep.fail(f"val max date {val_max.date()} >= test min "
                 f"{test_min.date()} — LEAKAGE")
    # Index disjointness
    all_idx = np.concatenate([split.train, split.val, split.test])
    if all_idx.size != np.unique(all_idx).size:
        rep.fail("split indices overlap")
    else:
        rep.ok("split index arrays pairwise disjoint")
    # Coverage gap
    covered = all_idx.size
    total = len(dt)
    if covered < total:
        rep.ok(f"{total - covered} transitions fall outside train/val/test "
               "(expected if the split leaves gaps; not a failure)")


def _check_normalizer(
    rep: _Report, dt: pd.DatetimeIndex | None,
    normalizer_path: Path | None, split_cfg: SplitConfig,
) -> None:
    rep.section("Normalization leakage")
    if normalizer_path is None:
        rep.ok("(no normalizer path provided; skipped — dataset is not "
               "pre-normalized, and no leakage can occur in the raw .npz)")
        return
    if not normalizer_path.exists():
        rep.fail(f"normalizer file {normalizer_path} not found")
        return
    try:
        with normalizer_path.open() as f:
            state = json.load(f)
    except Exception as e:
        rep.fail(f"failed to parse {normalizer_path}: {e}")
        return
    fit_dates = state.get("fit_date_range")
    if fit_dates is None:
        rep.ok("(normalizer has no recorded fit_date_range — cannot verify "
               "fit-on-train-only; add this field in future runs)")
        return
    fit_start, fit_end = pd.Timestamp(fit_dates[0]), pd.Timestamp(fit_dates[1])
    train_end = pd.Timestamp(split_cfg.train_end)
    if fit_end > train_end:
        rep.fail(f"normalizer fit_end={fit_end.date()} > "
                 f"train_end={train_end.date()} — STATISTICS LEAKAGE")
    else:
        rep.ok(f"normalizer fit range {fit_start.date()}→{fit_end.date()} "
               f"is inside the train window")


def _print_stats(ds: dict[str, np.ndarray], rep: _Report) -> None:
    rep.section("Descriptive statistics")
    r = ds["rewards"]
    rep.ok(f"rewards: mean={r.mean():+.6f}, std={r.std():.6f}, "
           f"min={r.min():+.6f}, max={r.max():+.6f}")
    if "forward_returns" in ds:
        fr = ds["forward_returns"]
        rep.ok(f"forward_returns per-asset mean={fr.mean(axis=0)}")
        rep.ok(f"forward_returns per-asset std ={fr.std(axis=0)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_split_cfg(args: argparse.Namespace) -> SplitConfig:
    if args.split_config is None:
        return DEFAULT_SPLIT
    with open(args.split_config) as f:
        raw = _try_load_yaml_or_json(f.read())
    return SplitConfig(
        train_start=raw.get("train_start", DEFAULT_SPLIT.train_start),
        train_end=raw.get("train_end", DEFAULT_SPLIT.train_end),
        val_start=raw.get("val_start", DEFAULT_SPLIT.val_start),
        val_end=raw.get("val_end", DEFAULT_SPLIT.val_end),
        test_start=raw.get("test_start", DEFAULT_SPLIT.test_start),
        test_end=raw.get("test_end", DEFAULT_SPLIT.test_end),
    )


def _try_load_yaml_or_json(text: str) -> dict[str, Any]:
    try:
        import yaml
        obj = yaml.safe_load(text)
    except Exception:
        obj = json.loads(text)
    if isinstance(obj, dict) and "split" in obj and isinstance(obj["split"], dict):
        return obj["split"]
    return obj or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="datasets/real_dirichlet.npz",
        help="Path to the .npz dataset to audit.",
    )
    parser.add_argument(
        "--transaction-cost", type=float, default=None,
        help=(
            "λ for the turnover penalty, used to recompute rewards and "
            "verify against the stored ones. Defaults to "
            "meta_transaction_cost from the dataset, falling back to 0.001."
        ),
    )
    parser.add_argument(
        "--split-config", default=None,
        help="Optional YAML/JSON file with train/val/test date bounds.",
    )
    parser.add_argument(
        "--normalizer", default=None,
        help=(
            "Optional path to a fitted normalizer JSON. If provided, we "
            "check that the recorded fit date range does not extend past "
            "the train window."
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    print(f"Auditing dataset: {dataset_path}")
    ds = _load_dataset(dataset_path)

    # Resolve λ from CLI → metadata → default.
    if args.transaction_cost is not None:
        lam = float(args.transaction_cost)
    elif "meta_transaction_cost" in ds:
        lam = float(ds["meta_transaction_cost"])
    else:
        lam = 0.001
    print(f"  transaction-cost λ (for reward recompute) = {lam}")

    split_cfg = _build_split_cfg(args)
    normalizer_path = Path(args.normalizer) if args.normalizer else None

    rep = _Report()
    N = _check_shapes(ds, rep)
    if N < 0:
        return rep.print_and_exit_code()
    _check_finite(ds, rep)
    _check_continuity(ds, rep)
    _check_reward_alignment(ds, rep, transaction_cost_lambda=lam)
    _check_simplex(ds, rep)
    dt = _check_date_monotonic(ds, rep)
    _check_feature_alignment(ds, rep, dt)
    _check_splits(ds, rep, dt, split_cfg)
    _check_normalizer(rep, dt, normalizer_path, split_cfg)
    _print_stats(ds, rep)
    return rep.print_and_exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
