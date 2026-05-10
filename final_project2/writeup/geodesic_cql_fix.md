# Geodesic-CQL fix — recommendation and exact LaTeX changes

## Recommendation: **Option 2** (keep the name "Geodesic-CQL", add a
disclaiming footnote, drop the `ma2022geodesic` citation).

### Why Option 2 over Option 1

| Risk dimension                        | Option 1 (rename + cite Fisher-BRC)                                                                                                                                                                            | Option 2 (keep name + footnote, no external cite)                                       |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Number of source-file substitutions   | High: every occurrence of "Geodesic-CQL" in abstract, intro, related-work, method, experiments, limitations must be edited (≈ 8 sites)                                                                         | Low: one footnote in related-work; one bib deletion                                     |
| Risk of a *new* misattribution        | Non-zero. Fisher-BRC (Kostrikov et al. 2021) uses a Fisher-divergence penalty against a Boltzmann policy density — *not* the same construction as a Fisher–Rao geodesic penalty on the policy. Claiming equivalence requires verifying the offline-phase code matches Fisher-BRC, which is a separate audit step. | None. We make no external attribution.                                                  |
| Risk of confusing internal references | Method/experiments tables, postmortem MDs, and CSV columns refer to "Geodesic-CQL"; Option 1 forces a breaking rename across artifacts not in scope for this writeup pass.                                     | None.                                                                                   |
| Reviewer-facing claim                 | "Our offline phase is Fisher-BRC + a CQL action penalty" — a positive technical claim a reviewer can dispute on equivalence grounds.                                                                           | "We use this name internally; we are not aware of prior published work" — a disclaimer. |
| Bibliography hygiene                  | Adds a new citation that must be verified.                                                                                                                                                                     | Removes the hallucinated citation; adds nothing.                                        |

Option 2 is strictly minimal change, removes the hallucinated citation,
and avoids the risk of asserting a Fisher-BRC equivalence that the
codebase has not been audited to support. Treat this fix as a
defensive submission edit, not a methodological reframing — the
offline-phase algorithm itself is unchanged.

## Exact LaTeX changes

### Change 1 — `writeup/draft_related_work.tex`, lines 61–62

Replace the `\citep[Fisher-Rao penalty;][]{ma2022geodesic}` attribution
with a self-disclaiming footnote and drop the citation.

**Before** (lines 61–62):
```latex
pessimism on a per-step regime-KL signal. The Geodesic-CQL component
\citep[Fisher-Rao penalty;][]{ma2022geodesic} contributes the offline
```

**After** (lines 61–62):
```latex
pessimism on a per-step regime-KL signal. The Geodesic-CQL
component\footnote{We use this name internally for our offline-phase
objective: CQL with a Fisher--Rao penalty on the policy. We are not
aware of prior published work under this name; the name should be
read as a code-level identifier rather than a reference to an external
method.} contributes the offline
```

### Change 2 — `writeup/refs.bib`, lines 171–176

Delete the entire `@article{ma2022geodesic, ...}` entry. The entry to
remove:

```bibtex
@article{ma2022geodesic,
  title={Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning},
  author={Ma, Yecheng Jason and Yan, Andrew and Hejna, Joey and Pavone, Marco and Bajcsy, Andrea and Finn, Chelsea and Pinto, Lerrel and Bastani, Osbert},
  journal={arXiv preprint arXiv:2207.07276},
  year={2022}
}
```

### Verification

After applying:

1. `grep -n "ma2022geodesic" writeup/` should return zero matches.
2. `grep -n "Geodesic-CQL" writeup/` should still match the existing
   in-paper uses (abstract paragraph 1, intro paragraph 2, method §3.4,
   experiments Phase 2C, limitations §4.1) — those are kept verbatim.
3. The footnote in `draft_related_work.tex` is the only place that
   discloses the internal nature of the name; the rest of the paper
   refers to "Geodesic-CQL" without external attribution.

### What this fix does **not** address

- The actual algorithmic content of the offline phase is unchanged.
  This is a citation-hygiene fix, not a method change.
- A future revision that wanted external grounding could replace the
  footnote with a Fisher-BRC reference (`kostrikov2021fisher`) plus a
  one-sentence comparison, but that requires verifying the offline
  code matches Fisher-BRC's Boltzmann-density construction. Out of
  scope for this pass.
