"""Build results/paper_master_table.csv — single source of truth for
the paper.

Columns: method, family, pipeline, n_seeds, sharpe_mean, sharpe_std,
sharpe_range_lo, sharpe_range_hi, cum_return_mean, max_dd_mean,
turnover_mean, beats_ew, notes.

Sources:
  - results/phase2a_causal/per_run.csv  (causal offline RL)
  - results/phase2a/per_run.csv         (leaky offline RL — appendix only)
  - results/phase2b/per_run.csv         (online-only)
  - results/phase2c/per_run.csv         (O2O)
  - results/phase2d/per_run.csv         (GRPO group-size ablation)
  - results/classical_causal.csv        (classical baselines)
  - results/aux_iql_216d/iql_216d_seed42/metrics.json (216-d IQL one-off)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EW_REF = 0.953


def _agg_per_seed(df: pd.DataFrame, group_cols: list, label_fn) -> list[dict]:
    """Group per-run rows into per-cell rows with mean/std/range."""
    rows = []
    if df.empty:
        return rows
    for keys, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sharpes = gdf["test_sharpe"].dropna().to_numpy()
        if sharpes.size == 0:
            continue
        rows.append({
            "_keys": keys,
            "n_seeds": int(sharpes.size),
            "sharpe_mean": float(sharpes.mean()),
            "sharpe_std": float(sharpes.std(ddof=1)) if sharpes.size > 1 else float("nan"),
            "sharpe_range_lo": float(sharpes.min()),
            "sharpe_range_hi": float(sharpes.max()),
            "cum_return_mean": float(gdf["test_cumulative_return"].mean()),
            "max_dd_mean": float(gdf["test_max_drawdown"].mean()),
            "turnover_mean": float(gdf["test_turnover"].mean()),
        })
    return rows


def _classify_beats_ew(sharpe_mean, sharpe_std, sharpe_lo) -> str:
    """yes / no / unstable. 'unstable' = std > |mean - EW|, regardless of sign."""
    if np.isnan(sharpe_mean):
        return "n/a"
    if not np.isnan(sharpe_std) and sharpe_std > abs(sharpe_mean - EW_REF):
        return "unstable"
    return "yes" if sharpe_mean > EW_REF else "no"


def main() -> int:
    rows = []

    # --- Classical baselines ---------------------------------------------
    cls = pd.read_csv(ROOT / "results" / "classical_causal.csv")
    for _, r in cls.iterrows():
        rows.append({
            "method": r["strategy"],
            "family": "classical",
            "pipeline": "causal",
            "n_seeds": 1,
            "sharpe_mean": float(r["sharpe_ratio"]),
            "sharpe_std": float("nan"),
            "sharpe_range_lo": float(r["sharpe_ratio"]),
            "sharpe_range_hi": float(r["sharpe_ratio"]),
            "cum_return_mean": float(r["cumulative_return"]),
            "max_dd_mean": float(r["max_drawdown"]),
            "turnover_mean": float(r["avg_daily_turnover"]),
            "beats_ew": "no" if r["strategy"] == "equal_weight" else (
                "yes" if r["sharpe_ratio"] > EW_REF else "no"),
            "notes": "single-pass deterministic, no seed",
        })

    # --- Phase 2A causal (BC, AWAC, CQL, IQL) ---------------------------
    a_causal = pd.read_csv(ROOT / "results" / "phase2a_causal" / "per_run.csv")
    a_causal_anchored = a_causal[
        a_causal["algo"].isin(["bc", "awac", "cql_vanilla", "iql"])
        & (a_causal["lambda"] == 0.001)
    ]
    for cell in _agg_per_seed(a_causal_anchored, ["algo"], lambda k: k[0]):
        method = cell.pop("_keys")[0]
        rows.append({
            "method": method,
            "family": "offline_anchored",
            "pipeline": "causal",
            "beats_ew": _classify_beats_ew(cell["sharpe_mean"],
                                            cell["sharpe_std"],
                                            cell["sharpe_range_lo"]),
            "notes": "behavior-anchored offline RL; equal-weight basin",
            **cell,
        })

    # --- Phase 2A leaky (TD3+BC, BCQ, BC, AWAC, CQL, IQL) ---------------
    a_leaky = pd.read_csv(ROOT / "results" / "phase2a" / "per_run.csv")
    a_leaky_lambda001 = a_leaky[a_leaky["lambda"] == 0.001]
    for cell in _agg_per_seed(a_leaky_lambda001, ["algo"], lambda k: k[0]):
        method = cell.pop("_keys")[0]
        is_leak_demo = method in ("td3_bc", "bcq")
        rows.append({
            "method": f"{method}_LEAKY",
            "family": "offline_leaky_appendix" if is_leak_demo
                      else "offline_anchored_leaky_appendix",
            "pipeline": "leaky",
            "beats_ew": _classify_beats_ew(cell["sharpe_mean"],
                                            cell["sharpe_std"],
                                            cell["sharpe_range_lo"]),
            "notes": "LEAK DEMONSTRATION — appendix only" if is_leak_demo
                     else "behavior-anchored on leaky pipeline; appendix only",
            **cell,
        })

    # --- Phase 2B online-only --------------------------------------------
    b = pd.read_csv(ROOT / "results" / "phase2b" / "per_run.csv")
    for cell in _agg_per_seed(b[b["lambda"] == 0.001], ["algo"], lambda k: k[0]):
        algo = cell.pop("_keys")[0]
        notes_map = {
            "sac_dirichlet": "SAC tau-blowup pathology — static across seeds",
            "ppo_lstm": "turnover ~1e-5 across seeds — degenerate",
            "grpo": "active-trading (turnover ~0.25), no warm-start",
        }
        rows.append({
            "method": f"online_only_{algo}",
            "family": "online_only",
            "pipeline": "causal",
            "beats_ew": _classify_beats_ew(cell["sharpe_mean"],
                                            cell["sharpe_std"],
                                            cell["sharpe_range_lo"]),
            "notes": notes_map.get(algo, "online-only at lambda=0.001"),
            **cell,
        })

    # --- Phase 2C O2O (naive, adaptive) ----------------------------------
    c = pd.read_csv(ROOT / "results" / "phase2c" / "per_run.csv")
    for cell in _agg_per_seed(c[c["lambda"] == 0.001], ["algo"], lambda k: k[0]):
        algo = cell.pop("_keys")[0]
        notes_map = {
            "naive_o2o": "CQL-locked static, seed-dependent allocation; pathology not bug",
            "adaptive_o2o": "HEADLINE positive: adaptive cql_weight selects basin consistently",
        }
        rows.append({
            "method": algo,
            "family": "o2o",
            "pipeline": "causal",
            "beats_ew": _classify_beats_ew(cell["sharpe_mean"],
                                            cell["sharpe_std"],
                                            cell["sharpe_range_lo"]),
            "notes": notes_map.get(algo, "O2O condition"),
            **cell,
        })

    # --- Phase 2D GRPO ablation ------------------------------------------
    d = pd.read_csv(ROOT / "results" / "phase2d" / "per_run.csv")
    for cell in _agg_per_seed(d, ["algo", "group_size"], lambda k: f"grpo_G{int(k[1])}"):
        algo, g = cell.pop("_keys")
        rows.append({
            "method": f"grpo_G{int(g)}",
            "family": "grpo_ablation",
            "pipeline": "causal",
            "beats_ew": _classify_beats_ew(cell["sharpe_mean"],
                                            cell["sharpe_std"],
                                            cell["sharpe_range_lo"]),
            "notes": f"GRPO group-size ablation, G={int(g)}",
            **cell,
        })

    # --- 216-d IQL one-off (appendix scope condition) --------------------
    aux_path = ROOT / "results" / "aux_iql_216d" / "iql_216d_seed42" / "metrics.json"
    if aux_path.exists():
        m = json.loads(aux_path.read_text())["test"]
        rows.append({
            "method": "iql_216d",
            "family": "appendix_scope_condition",
            "pipeline": "causal_216d",
            "n_seeds": 1,
            "sharpe_mean": float(m["sharpe_ratio"]),
            "sharpe_std": float("nan"),
            "sharpe_range_lo": float(m["sharpe_ratio"]),
            "sharpe_range_hi": float(m["sharpe_ratio"]),
            "cum_return_mean": float(m["cumulative_return"]),
            "max_dd_mean": float(m["max_drawdown"]),
            "turnover_mean": float(m["turnover"]),
            "beats_ew": "no",
            "notes": "216-d feature space; basin-transient, see appendix",
        })

    df = pd.DataFrame(rows)
    cols = [
        "method", "family", "pipeline", "n_seeds",
        "sharpe_mean", "sharpe_std", "sharpe_range_lo", "sharpe_range_hi",
        "cum_return_mean", "max_dd_mean", "turnover_mean",
        "beats_ew", "notes",
    ]
    df = df[cols]

    # Sort: family family-order, then sharpe_mean within family
    family_order = [
        "classical", "offline_anchored", "online_only", "o2o",
        "grpo_ablation", "appendix_scope_condition",
        "offline_leaky_appendix", "offline_anchored_leaky_appendix",
    ]
    df["_fam_idx"] = df["family"].map({f: i for i, f in enumerate(family_order)})
    df = df.sort_values(["_fam_idx", "method"]).drop(columns=["_fam_idx"])

    out = ROOT / "results" / "paper_master_table.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"saved -> {out}")
    print(f"  total rows: {len(df)}")
    print(f"  per family: {df.groupby('family').size().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
