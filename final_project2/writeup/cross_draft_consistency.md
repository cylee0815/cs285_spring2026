# Cross-draft numerical consistency audit

Read-only audit across `writeup/draft_*.tex` and `writeup/*.md`,
verifying each headline figure is cited identically wherever it
appears. One precision discrepancy found and patched in this
session; the rest are clean.

## Summary

| Headline figure | Status | Notes |
|---|---|---|
| Adaptive O2O 1.762 ± 0.154, range [1.629, 1.931], turnover ~7×10⁻⁵ | **clean** | All citations in `draft_experiments.tex` agree to four decimals |
| Naive O2O 1.002 ± 1.208, range [−0.236, +2.179], turnover ~0 | **patched** | `draft_experiments.tex:186` had rounded `+1.00 ± 1.21`; corrected to `+1.002 ± 1.208` to match the table at line 208 and the master CSV |
| Frozen offline (causal IQL) 0.935 ± 0.009 | **clean** | Cited at `draft_appendix_leak_detection.tex:128` and `:170`; both identical |
| Online-only SAC (Phase 2B) static / τ-blowup | **clean** | Qualitative cite only. Numerical value 0.9413 (or 0.941 rounded) appears in `sac_degeneracy_postmortem.md`, `o2o_audit.md`, and `td3_bcq_anomaly_triage.md`; `draft_appendix_leak_detection.tex:66` cites the leaky-pipeline BC at 0.941 (consistent with the postmortem's numerical record) |
| GRPO online (Phase 2B) 0.297 ± 0.017 | **gap, not discrepancy** | Number not cited in any draft section. Phase 2B GRPO is not a headline row in any current writeup — only Phase 2D ablation cites GRPO. Either add a Phase 2B GRPO sentence to `draft_experiments.tex` Phase 2B subsection, or omit (currently the Phase 2B subsection is mostly TODO scaffolding) |
| GRPO ablation G=4 0.500 ± 0.108 | **clean** | `draft_experiments.tex:329` (prose), `:368` (table mean), `:369` (table std) all consistent |
| GRPO ablation G=8 0.403 ± 0.099 | **clean** | Same locations |
| GRPO ablation G=16 0.313 ± 0.051 | **clean** | Same locations |
| Equal-weight 0.953 | **clean** | Cited 5+ times across `draft_appendix_leak_detection.tex` and `draft_experiments.tex`, identical |
| Momentum 0.910 | **clean** | Single citation in `draft_experiments.tex` (turnover figure caption); matches `results/classical_causal.csv` |
| Risk-parity 1.161 | **clean** | `draft_experiments.tex:268` and `:287`, identical |
| Phase 2A causal BC 0.935 ± 0.002 | **clean** | `draft_appendix_leak_detection.tex:125` |
| Phase 2A causal AWAC 0.940 ± 0.004 | **clean** | `draft_appendix_leak_detection.tex:126` |
| Phase 2A causal CQL 0.952 ± 0.001 | **clean** | `draft_appendix_leak_detection.tex:127` |
| Phase 2A causal IQL 0.935 ± 0.009 | **clean** | `draft_appendix_leak_detection.tex:128` and `:170` |
| 216-d IQL 0.398 (single seed) | **clean** | `216d_iql_dynamics.md` uses `0.3979` (4 decimals) at lines 18 and 142; `draft_appendix_leak_detection.tex:171` rounds to `0.398` for the table. Difference is precision, not value |
| Phase 2A leaky TD3+BC 6.785 ± 0.628 | **clean** | `td3_bcq_anomaly_triage.md:75, :106` and `draft_appendix_leak_detection.tex:63` all match |
| Phase 2A leaky BCQ 3.905 ± 0.232 | **clean** | `td3_bcq_anomaly_triage.md:76` and `draft_appendix_leak_detection.tex:64` match |
| Phase 2A leaky behavior-anchored cluster ~0.94 | **clean** | The 4 leaky rows in `draft_appendix_leak_detection.tex:66-69` match `results/phase2_appendix_leaky.csv`: BC 0.941 ± 0.006, AWAC 0.942 ± 0.011, CQL 0.945 ± 0.001, IQL (λ=0.001) 0.943 ± 0.002. These are different from the causal Phase 2A numbers because they're a different pipeline — expected and correctly distinguished |

## Patches applied this session

`writeup/draft_experiments.tex:186`:

```diff
- (\textbf{range $[-0.236, +2.179]$, mean $+1.00$, std $1.21$}) because
+ (\textbf{range $[-0.236, +2.179]$, mean $+1.002$, std $1.208$}) because
```

That's the only patch. Two-decimal rounding in prose was inconsistent
with the four-decimal table and the master CSV; the four-decimal form
matches the source data and is now the canonical citation.

## Open question (not a discrepancy)

The Phase 2B online-only GRPO Sharpe (0.297 ± 0.017 across 3 seeds) is
in `results/phase2b/per_run.csv` and the master CSV but is **not
referenced in any draft section**. The Phase 2B subsection in
`draft_experiments.tex:89-117` still contains TODO(post-2B) scaffolding
and a "pre-registered finding" paragraph for SAC's τ-blowup, but no
write-up of the actual Phase 2B numbers. Two paths:

1. **Add a Phase 2B results paragraph** to the existing subsection
   citing all three online-only methods at their tested Sharpes
   (SAC 0.941 / PPO+LSTM −0.07 to +1.27 (degenerate turnover) / GRPO
   0.297 ± 0.017). This makes the four-way comparison
   (Sec.~\ref{sec:experiments:phase2c}) self-contained — the reader can
   see "online-only SAC" came from here.
2. **Leave Phase 2B as scaffolding** if the paper's headline narrative
   doesn't depend on the online-only baselines as more than reference
   rows in Fig.~\ref{fig:turnover-by-method}. The turnover figure
   already shows all three Phase 2B methods; the bar heights speak for
   themselves.

Recommend (1) — the Phase 2B subsection is currently the weakest part
of `draft_experiments.tex` and a reviewer will notice. ~10 sentences
in a fresh-context paper-drafting session would close this gap. Not
patching here because it's a writeup decision, not a consistency fix.

## Verdict

**No substantive numerical discrepancies.** One precision-rounding
patch applied. One unresolved gap (Phase 2B online-only GRPO not yet
cited in any draft) flagged for the paper-drafting session, not a
defect in any existing draft.
