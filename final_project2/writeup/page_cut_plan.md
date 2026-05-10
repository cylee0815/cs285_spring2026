# Page Cut Plan — target 10-12 pages of main body

Current build: **21 pages total** (`paper_skeleton.pdf`, post-Phase-F).
Estimated main-body length: **~15 pages**, appendix + bib: **~6 pages**.
Target main body: **10-12 pages**, appendices grow to absorb the
content moved out of the main body.

## Section-by-section page accounting (estimated from pdflatex shipout
log + figure/table density)

| # | Section | Source file (lines) | Est. pages | Notes |
|--:|---|---|--:|---|
| 1 | Title + Extended Abstract     | `draft_abstract.tex` (107 L)         | 1.5 | One full page of body. |
| 2 | Introduction                  | `draft_introduction.tex` (105 L)     | 1.25 | |
| 3 | Related Work                  | `draft_related_work.tex` (72 L)      | 0.75 | |
| 4 | Method                        | `draft_method.tex` (218 L)           | 2.75 | 2 subsections (Continuous GRPO, Adaptive O2O), several display equations. |
| 5 | Experiments                   | `draft_experiments.tex` (526 L)      | 7.0 | **Largest section, primary cut target.** Includes 2 figures + 5 tables. |
| 6 | Discussion + Limitations      | `draft_limitations.tex` (182 L)      | 2.5 | |
| 7 | Contributions (rubric box)    | inline in `paper_skeleton.tex`       | 0.25 | |
|   | **Main-body subtotal**       |                                      | **~16** | |
| A | Appendix: Implementation     | `draft_appendix_implementation_notes.tex` (263 L) | 2.5 | 4 bug subsections + scaffolding subsection. |
| B | Appendix: Leak detection     | `draft_appendix_leak_detection.tex` (287 L) | 2.5 | 6 subsections incl. 3 tables. |
|   | Bibliography                  | `refs.bib`                           | 1.0 | |
|   | **Total**                     |                                      | **21** | |

## Net cut required

Main body **~16 → ~11 pages** ⇒ **−5 pages from main body**, but with
**+1 page returning** because we are pulling the leak-diagnostic
contribution out of the appendix into the experiments section
(Phase H rationale below).

## What to keep (per user priority)

1. **Abstract** — keep as-is (1.5 pp).
2. **Introduction** — keep but tighten by removing the
   "milestone recap" digression (`draft_introduction.tex:11-25`).
   Could compress to 1 page.
3. **Method**:
   - Keep GRPO derivation (Sec.~\ref{sec:method:grpo}) full.
   - Keep adaptive-schedule formula (Eq.~\ref{eq:cql-schedule}) full.
   - Keep Problem setup minimal.
4. **Experiments** — keep two strongest results in the main body:
   - **Phase 2C basin-selection** (Sec.~\ref{sec:experiments:phase2c},
     `draft_experiments.tex:144-413`).
   - **Leak diagnostic**, pulled out of the appendix into a new
     experiments subsection (~1 page; see Phase H).
5. **Discussion + Limitations** — keep but compress.
6. **Contributions** — required by rubric, 0.25 page.

## What to move to appendix

| Cut | Source | Est. saved | Where to move |
|---|---|--:|---|
| C1 | Phase 2D GRPO group-size ablation. Body: `draft_experiments.tex:415-488` (74 lines, 1 figure, 1 table). | **1.5 pp** | New appendix subsection, e.g., `draft_appendix_implementation_notes.tex` → add `\subsection{GRPO group-size ablation}`, OR a new appendix section. Replace in main body with 2-sentence forward pointer in the GRPO method discussion. |
| C2 | "Static-allocation reference distribution" methodology paragraph (`draft_experiments.tex:195-218`, including the "min-variance is heavily SHY-tilted" technical aside) and the figure caption text below. | **0.5 pp** | Move methodology details to appendix (new `\subsection{Simplex reference distribution}` in leak appendix's parent appendix, OR a fresh appendix). Keep the 1-sentence summary + figure in main body. |
| C3 | Phase 2C "GRPO offline warm-start (omitted condition)" paragraph (`draft_experiments.tex:180-193`). | **0.4 pp** | Appendix-only; replace with one sentence: "We attempted a GRPO warm-start condition but found no feature-compatible source available; details in App.~\ref{app:leak:scope}." |
| C4 | Phase 2C "Naive O2O: static collapse" detailed mechanistic paragraph (`draft_experiments.tex:235-259`) — keep the table, compress the prose to 4 sentences. | **0.5 pp** | Detailed prose to appendix; main body keeps result + table. |
| C5 | Phase 2C "Turnover decomposition" paragraph + figure (`draft_experiments.tex:382-404`) — useful but secondary. | **0.6 pp** | Move figure + paragraph to appendix; keep 1-sentence summary + numeric reference in main body. |
| C6 | Phase 2C "The adaptive-vs-naive comparison is meaningful only post-fix" paragraph (`draft_experiments.tex:406-413`). | **0.25 pp** | Replace with one sentence pointing to Appendix. (Note: Phase F already condensed the parallel method-section paragraph; this one in experiments is a duplicate disclosure.) |
| C7 | Setup paragraph "Compute" (`draft_experiments.tex:39-43`) and "Behavior-policy buffer" detail (`draft_experiments.tex:45-57`). | **0.3 pp** | Move compute spec and buffer-mixture details to appendix (new "Reproduction notes" subsection). Keep one-sentence Setup intro in main body. |
| C8 | Discussion subsection (`draft_limitations.tex:10-100`) — currently 90 lines / ~2 pages. The basin-selection paragraph is critical; the "Implications for O2O" paragraph and "Continuous GRPO dominated equilibrium" paragraph can each be compressed. | **0.75 pp** | Compress in place (no appendix move). |
| C9 | Limitations subsection (`draft_limitations.tex:102-182`) — 7 paragraphs. Keep "Single test window," "Behavior-mixture sensitivity," "Continuous-GRPO scope." Move "Seed reproducibility," "$\beta_t$-trajectory verification," "Post-paper follow-up: theoretical extensions" to appendix. | **0.5 pp** | Three paragraphs → "Future work" appendix subsection. |
| C10 | "Summary scoreboard" table (`draft_experiments.tex:490-525`), added by `polish_pass.diff`. Useful but largely duplicates per-row info already in tables 3-4. | **0.5 pp** | Move table to appendix, with 1-sentence main-body pointer. Decision flag: **optional** — table is highly compressed, may earn its space; keep if budget allows. |

**Total moveable: ~5.85 pages** (less if C10 is kept).
**Plus +1 page** for promoting leak diagnostic into main body (Phase H).

**Net main-body change: ~16 + 1 − 5.85 ≈ 11.15 pages.** ✓ Lands in
the 10-12 page target.

## Pulled into main body

**P1 — Promote leak diagnostic to a 1-page experiments subsection.**

Current state: contribution-3 (algorithm-class differential leak
diagnostic) is presented in the abstract (`draft_abstract.tex:92-101`)
and intro (`draft_introduction.tex:87-93`), but the supporting evidence
lives entirely in `draft_appendix_leak_detection.tex` Tables 1, 2, 3.
A reader of the main body never sees the numbers backing
contribution-3.

Proposed move: pull approximately one page of the leak appendix into
a new experiments subsection between Phase~2A and Phase~2B (or after
Phase 2D moves to the appendix, between Phase 2C and Discussion):

```
\subsection{The algorithm-class-differential leak diagnostic}
\label{sec:experiments:leak}
```

Content: Table~\ref{tab:leak-differential} (the 6-row Phase 2A leaky
result, ~25 lines) + a half-paragraph framing + the 56-d/216-d scope
condition row from Table~\ref{tab:scope-condition}. Detailed
discussion (the "two interpretations of leakage," the one-line fix,
the diagnostic-as-recommendation paragraph) stays in the appendix.

## Suggested section ordering after cuts

1. Abstract (1 page)
2. Introduction (1 page)
3. Related Work (0.5 page)
4. Method (2.5 pages)
   - Problem setup (0.5)
   - Continuous GRPO (1.5)
   - Adaptive-O2O scheduler (0.5)
5. Experiments (4.5-5 pages)
   - Setup (compressed) (0.5)
   - Phase 2A: offline-baseline landscape (0.5)
   - Phase 2B: online-only baselines (0.5)
   - **Phase 2C: four-way O2O comparison (the central experiment)** (2.0-2.5)
   - **The algorithm-class-differential leak diagnostic** (NEW, 1.0)
6. Discussion + Limitations (1.5-1.75 pages)
   - Discussion: basin-selection mechanism revision (compressed) (0.75)
   - Limitations (compressed) (0.75-1.0)
7. Contributions (0.25 page)

**Main-body total: ~10.75 - 11.5 pages** ✓

Appendices grow from ~5 to ~7-8 pages (incl. bib), which is fine —
appendix-friendly content beyond 10 pages is the explicit standard.

## Risk register

- **R1 — Phase 2D removal weakens the GRPO methodological claim.** The
  group-size ablation is the empirical evidence that the
  compute-vs-precision tradeoff is the main GRPO knob. If reviewers
  ask about ablations, the appendix pointer must be visible from the
  Continuous GRPO method paragraph.
- **R2 — Pulling leak diagnostic into main body strengthens
  contribution-3 but weakens narrative flow** (it is currently
  compartmentalized as appendix material so the main story stays
  about basin selection). Phrase the new subsection as a methodological
  sidebar: "An incidental finding from Phase~2A: an algorithm-class
  signature..." rather than a co-equal central result.
- **R3 — Compressing Discussion paragraphs risks losing the
  basin-steering null-test calculation** (limitations.tex:36-48), which
  is one of the few quantitative null-rejection arguments in the paper.
  Preserve that paragraph verbatim during compression of surrounding
  prose.
- **R4 — The "GRPO offline warm-start (omitted condition)" cut (C3)**
  removes context for why we don't have the obvious 5th condition.
  Keep at least one sentence with a forward-pointer; otherwise
  reviewers will ask for the omitted condition.

## Application order if approved

1. C1 (Phase 2D move) — largest single cut, lowest risk
2. C7 (Setup compression) — minimal narrative risk
3. C9 (Limitations partial move) — low risk
4. C5 (Turnover figure to appendix) — low risk
5. C2, C4 (Static-allocation paragraph + Naive O2O detail compression)
6. P1 (Leak-diagnostic promote, after experiments space is freed)
7. C8 (Discussion compress) — last, as the final landing-pad
8. C10 (Summary scoreboard) — optional; defer decision until rebuilt

After each step: rebuild and verify page count + that no figure or
table is orphaned by a removed reference.
