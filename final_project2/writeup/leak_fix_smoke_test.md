# Leak fix smoke test — `phase2-leak-fix`

## Patch

Branch: `phase2-leak-fix`
Commit: `65d4040 fix(data): shift forward_returns by +1 to restore causal env contract`

`core/envs/data_utils.py` — both `make_train_test_envs` (lines 267-284
post-fix) and `make_train_val_test_envs` (lines 376-394 post-fix) now
shift `forward_returns` by +1 timestep and trim the last row of
`flat_features` to match. Diff: 20 lines added / 4 removed, mechanical.

```python
# Post-fix at both callsites:
flat_features_full = _build_flat_features(features, macro_arr, sentiment_arr)
fwd_full = np.expm1(log_returns).astype(np.float32)
# Causal shift: forward_returns[t] is the return realized AFTER the
# agent acts on features[t] (matches PortfolioEnv contract,
# core/envs/portfolio_env.py:30-31). ...
flat_features = flat_features_full[:-1]
forward_returns = fwd_full[1:]
```

The online pipeline (`features/feature_engineering.py:219-220`) already
does this; this fix brings the offline pipeline into agreement.

## Smoke test

Single BC run, 200 offline updates, λ=0.001, seed 42, 4-way mixture
behavior buffer (matches Phase 2A defaults).

```bash
uv run python scripts/run_offline.py \
    --base_config bc --seed 42 --transaction_cost 0.001 \
    --behavior_mix mixture --no_wandb \
    --n_offline_updates 200 --eval_interval 100 \
    --results_dir results/smoke --run_name bc_leakfix_smoke
```

Wall-clock: ~25 s (data download + buffer fill + 200 BC updates +
3-split final eval). Exit code 0.

## Acceptance checks

| Check | Result |
|---|---|
| (a) No NaN in any metric or weight | **Pass** — `np.isnan(weights).any() = False`, `np.isnan(PV).any() = False`, all 8 metrics.json test fields finite |
| (b) Loss curve non-degenerate | **Pass** — final test sharpe = +0.890, train sharpe = +0.552, val sharpe = +1.340 (all finite, in plausible range) |
| (c) Test-rollout actions on the simplex | **Pass** — across all 1,061 test steps, per-row weight sum is exactly 1.000000 (min = max = mean = 1.000000), per-element weight in [0.0836, 0.2411] (all non-negative) |

## Sanity numbers

Test-window backtest on the causal pipeline (1,061 trading days,
2022-01-03 → 2026-03-30):

```
sharpe_ratio:       +0.8898
annual_return:      +0.0631
annual_volatility:   0.0718
max_drawdown:        0.1230
turnover:            0.0097
cumulative_return:  +0.2946
sortino_ratio:      +1.2422
calmar_ratio:       +0.5131
```

Comparison points:
- Leaky-pipeline BC at 20k updates (Phase 2A): sharpe = +0.941 (3-seed mean, std 0.006).
- Equal-weight reference: sharpe ≈ 0.95.
- Causal smoke at 200 updates: sharpe = +0.890.

Test sharpe lands inside the user-specified [0.85, 1.05] band for
behavior-anchored algorithms on causal data, consistent with the
"behavior anchor → equal-weight basin under friction" prediction. The
~0.05 drop from leaky 0.94 vs causal 0.89 is partly attributable to
training only 200 vs 20,000 steps (the BC loss has barely converged at
200) and partly to the smaller effective dataset (one fewer training
day after the trim). **Step 4** is the official sanity test at full
training length; this smoke is only checking the fix doesn't break
anything.

## Side-effects to flag

- Test window length is 1,061 days post-fix vs 1,063 days pre-fix
  (difference is 2: one trimmed at end of train, one at end of test;
  exactly equals the count of `_make_env` calls applied to the trimmed
  arrays). Phase 2B/2D windows are unaffected (they consume the
  already-causal `datasets/real_*.npz` which has its own length).
- `metadata['T_test']` / `T_train` / `T_val` reflect pre-trim split
  sizes from `prices.index`; downstream slicing on the (T-1)-row
  arrays handles the off-by-one gracefully (Python slice past array
  length returns up to the end). Smoke confirms no breakage.
- Run logs in `logs/smoke/` and trade CSVs in
  `results/trades/bc_leakfix_smoke/` are intermediate artifacts, can
  be deleted after the user signs off.

## Status

Step 1 done. Awaiting your sign-off before launching Step 2 (Phase 2D
seed 1337) and Step 3 (Phase 2C re-run).
