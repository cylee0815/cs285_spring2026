# Aggregation review — `aggregate_phase2.py` and `plot_phase2.py`

Read-only audit of the two scripts staged for post-Phase-2B aggregation.
What they emit today, what's missing for the paper's main results table,
what to patch (post-2B, not now).

## What `scripts/aggregate_phase2.py` emits today

- **Per-phase per-run CSVs**: `results/phase2{a,b,c}/per_run.csv`, one row
  per run with columns
  `algo, lambda, seed, run_name, n_step, behavior_mix,
   best_val_sharpe, best_val_step,
   test_sharpe, test_annual_return, test_annual_volatility,
   test_max_drawdown, test_turnover, test_cumulative_return,
   test_sortino, test_calmar`. Phase 2D is not collected
   (`scripts/aggregate_phase2.py:91`).
- **Concatenated summary**: `results/phase2_summary.csv`, the union of
  the three phase CSVs with an added `phase` column.
- Robust to: missing metrics.json files, missing nested `test` keys (uses
  `.get()` everywhere), filename non-conformity (parser falls back to
  `algo = run_name, lambda = None, seed = None`).

## What `scripts/plot_phase2.py` emits today

`figures/phase2/{four_way_comparison,turnover_comparison,
adaptive_beta_trajectory,friction_band_comparison,grpo_group_size_curve}.png`
plus `four_way_comparison_data.csv`. Each function guards against an
empty summary or a missing `phase` column.

## What's missing for the paper's main results table

The paper needs a table of the form:

| Method | λ=0 | λ=0.001 | λ=0.005 |
|---|---|---|---|
| BC | s ± d | s ± d | s ± d |
| TD3+BC | … | … | … |
| … | … | … | … |
| Adaptive O2O | … | … | … |

with cells reporting **mean ± std (across seeds) of test Sharpe** (and
parallel tables for cum return, max drawdown, turnover). Today's
aggregator emits per-run rows only — there is no `(method, λ) → cell`
reduction. The four-way comparison plot in `plot_phase2.py` does an
in-memory groupby for the bar chart but the result is not persisted.

Concretely:

1. **No `(method, λ)` aggregation.** Need a derived CSV with one row
   per `(method, λ)` cell: `method, lambda, n_seeds, sharpe_mean,
   sharpe_std, sharpe_sem, cum_return_mean, cum_return_std,
   max_dd_mean, max_dd_std, turnover_mean, turnover_std`. ~25 lines of
   pandas in a new function `aggregate_to_cells(summary)`.
2. **Phase 2D not surfaced.** `aggregate_phase2.py:91` iterates only
   `phase2a/b/c`. The GRPO group-size ablation lives in
   `results/phase2d/`. Add `phase2d` to the loop and parse the run-name
   `_G\d+_` token into a `group_size` column.
3. **Online-only SAC reuse for Phase 2C.** The four-way comparison's
   "online-only SAC" condition is supposed to reuse Phase 2B SAC runs
   (per `_run_phase2c.sh` design comment), not duplicate them. The
   current `plot_phase2.py:four_way_comparison` does pull from
   `phase=='phase2b' & algo=='sac_dirichlet'`, so this works in
   practice — but the **derived cells CSV** should *relabel* those rows
   under method `online_only_sac` so the paper table can be sliced by
   method without an awareness of the source phase.
4. **Best-Phase-2A-checkpoint selection for "frozen offline".** The
   four-way comparison's "frozen offline" condition is meant to be the
   *best-on-validation* offline algorithm at λ=0.001. Today's
   `four_way_comparison` picks the best per (seed, λ) by `test_sharpe`
   (`scripts/plot_phase2.py:50-53`), which leaks the test split. Should
   instead pick by `best_val_sharpe`. One-line change:
   `sort_values("best_val_sharpe", ascending=False)`.
5. **Median + IQR reporting.** With three seeds, mean ± std is noisy
   under a single outlier (e.g., the SAC tau-blowup case). Adding
   `sharpe_median`, `sharpe_iqr_low`, `sharpe_iqr_high` lets the report
   show robust statistics in the appendix. ~5 lines of pandas.
6. **Significance vs. equal-weight baseline.** The interesting null
   hypothesis is `test_sharpe = EW_sharpe (=0.953)`. Add per-cell
   `sharpe_minus_ew_mean, sharpe_minus_ew_std` for direct relative
   reporting. ~3 lines.
7. **`adaptive_beta_trajectory.png`** loads `cql_weight_traj.npy` from
   `results/phase2c/<run>/`, which only the patched `run_o2o.py`
   produces. This is correct given the bug-4 fix in
   `writeup/draft_implementation_notes.md`, but the plot script silently
   skips when no traj files exist (`scripts/plot_phase2.py:108-111`),
   which would mask the absence of the centerpiece evidence. Add a
   `[warn]` print so the failure mode is loud.

## Patch plan (post-2B, ~30 minutes)

After Phase 2B notify lands and before Phase 2A starts (the orchestrator
already separates these), apply the following:

1. Extend `aggregate_phase2.py` to (a) include `phase2d` in the loop,
   (b) parse `group_size` from Phase 2D run names, (c) emit a derived
   `results/phase2_cells.csv` with per-`(method, lambda)` mean / std /
   sem / median / IQR for sharpe / cum return / max drawdown /
   turnover / sharpe-minus-EW. Add a `--write_cells` flag and have the
   orchestrator pass it.
2. In `plot_phase2.py:four_way_comparison`, change the "frozen offline"
   selection from `sort_values("test_sharpe")` to
   `sort_values("best_val_sharpe")` to remove the leakage; relabel the
   reused Phase 2B SAC rows as `online_only_sac` in the derived CSV;
   add a loud warning when `cql_weight_traj.npy` files are absent.
3. Add a small `format_table.py` that renders `phase2_cells.csv` as a
   LaTeX `\begin{tabular}` block ready for the paper. ~30 lines of
   string formatting.

None of this requires re-running any training. The patch can ship in the
same commit as the Phase 2 results paper.

## What's already correct

- Robust handling of missing fields and broken JSON.
- Clean separation between the per-run extractor and the per-phase
  loop, so adding Phase 2D is a one-line addition to the loop tuple.
- `parse_name` correctly handles single-digit and multi-digit seeds and
  decimal lambdas.
- The summary CSV preserves per-run granularity, so any post-hoc
  re-aggregation (e.g., changing the seed protocol from mean-std to
  bootstrapped CI) works without re-running anything upstream.

The code is in better shape than the bug-littered runners we patched in
Bug 1–4. The patch list above is purely additive.
