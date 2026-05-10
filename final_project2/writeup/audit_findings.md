# Citation Audit + Polish-Pass Rigor Flags

Generated: 2026-05-10. No edits were applied to the manuscript; this report
accompanies `writeup/polish_pass.diff` (which is also unapplied).

## Pre-flight

Files inspected:
- `writeup/draft_abstract.tex` (97 lines)
- `writeup/draft_introduction.tex` (101 lines)
- `writeup/draft_related_work.tex` (67 lines)
- `writeup/draft_method.tex` (233 lines)
- `writeup/draft_experiments.tex` (493 lines)
- `writeup/draft_limitations.tex` (176 lines)
- `writeup/draft_appendix_implementation_notes.tex` (263 lines)
- `writeup/draft_appendix_leak_detection.tex` (287 lines)
- `writeup/refs.bib` (198 lines)

Web search and web fetch were available; verifications below cite live URLs
fetched during the audit.

---

## Pass A — Citation Audit

### [P1] `ma2022geodesic` — STATUS: HALLUCINATED

**Bib entry as written**
- Title: "Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning"
- Authors: Ma, Yecheng Jason and Yan, Andrew and Hejna, Joey and Pavone, Marco and Bajcsy, Andrea and Finn, Chelsea and Pinto, Lerrel and Bastani, Osbert
- arXiv ID: 2207.07276
- Year: 2022

**In-paper attribution**
- `draft_related_work.tex` line 62: `\citep[Fisher-Rao penalty;][]{ma2022geodesic}` introducing the offline-phase pretrain as "Geodesic-CQL".
- Also referenced in `draft_abstract.tex` ("Geodesic-CQL pretrain"), `draft_introduction.tex` ("Geodesic-CQL pretrain on 2008–2020 transitions"), and `draft_method.tex` §3.4 ("Geodesic-CQL critic").

**Evidence**
- Fetched arXiv 2207.07276: actual title "A Flexible Schema-Guided Dialogue Management Framework: From Friendly Peer to Virtual Standardized Cancer Patient" by Kane, Giugno, Schubert, Haut, Wohn, Hoque. Topic: clinical dialogue management. URL: https://arxiv.org/abs/2207.07276 — not RL.
- The closest real Ma et al. 2022 paper is **CAP** ("Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning"), arXiv:2112.07701, AAAI 2022, by **Yecheng Jason Ma, Andrew Shen, Osbert Bastani, Dinesh Jayaraman** (4 authors, not 8). CAP is about cost-uncertainty penalties for *model-based safe* RL — it is not "Geodesic-CQL" and contains no Fisher-Rao penalty.
- Web search for `"Geodesic-CQL" Fisher-Rao penalty offline reinforcement learning` returns no method by that name. The closest real construction is **Fisher-BRC** (Kostrikov, Fergus, Tompson, Nachum 2021, arXiv:2103.08050), a Boltzmann-policy Fisher-divergence penalty — different mathematics from "Fisher-Rao geodesic".

**Conclusion**
- Both the arXiv ID → title and the title → author list mappings are wrong.
- "Geodesic-CQL" appears to be an internal codebase name with no published reference behind it.
- The bibtex author list looks generated rather than curated (the 8-author list combines several well-known robotics names that have not co-authored together).

**Recommended fixes** (any of these is sufficient; pick one)
1. Drop the external attribution. Rename the offline-phase method to an internal name (e.g., "Fisher-penalized CQL pretrain") and describe its derivation in the method section. Remove `ma2022geodesic` from `refs.bib`.
2. If the implementation is in fact Fisher-BRC + a CQL action penalty: cite `kostrikov2021fisher` (Kostrikov et al. 2021) for the Fisher-BRC penalty and `kumar2020conservative` for the CQL action penalty, and write "Fisher-BRC-style penalty combined with CQL action regularization" rather than "Geodesic-CQL".
3. Suggested replacement bib entry if option 2 is taken:
   ```bibtex
   @article{kostrikov2021fisher,
     title={Offline Reinforcement Learning with Fisher Divergence Critic Regularization},
     author={Kostrikov, Ilya and Fergus, Rob and Tompson, Jonathan and Nachum, Ofir},
     journal={International Conference on Machine Learning (ICML)},
     year={2021}
   }
   ```

**Severity:** Submission-blocker. Cannot ship under "Geodesic-CQL [Ma et al. 2022]".

---

### [P2] `zhao2022adaptive` — STATUS: HALLUCINATED / WRONG ARXIV ID

**Bib entry as written**
- Title: "Adaptive Online Replanning with Diffusion Models"
- Authors: Zhao, Sirui and Pearce, Tim
- arXiv ID: 2210.06463
- Year: 2022

**In-paper attribution**
- `draft_related_work.tex` line 58: `\citep{nair2020awac,zheng2023adaptive,zhao2022adaptive}` — cited as O2O fine-tuning literature in robotics, alongside AWAC.

**Evidence**
- Fetched arXiv 2210.06463: actual title "Holo-Dex: Teaching Dexterity with Immersive Mixed Reality" by Sridhar Pandian Arunachalam, Irmak Güzey, Soumith Chintala, Lerrel Pinto. Topic: VR teleoperation for robot dexterity training. URL: https://arxiv.org/abs/2210.06463 — not O2O RL, not by Zhao or Pearce.
- A paper titled "Adaptive Online Replanning with Diffusion Models" exists in the literature (Zhou et al. 2023, NeurIPS) but it is by different authors and concerns model-based replanning, not O2O conservatism scheduling. It is also not a fit for the related-work claim.

**Conclusion**
- Both the arXiv ID and the title→author mapping are fabricated.
- The cited claim (O2O fine-tuning in robotics) is not supported by any paper bearing this title and author pair.

**Recommended fixes**
- Remove `zhao2022adaptive` entirely from `refs.bib` and from the related-work \citep{} group.
- Replace with at least one verified O2O reference. Two strong candidates:
  - **Cal-QL** (Nakamoto et al. 2023, NeurIPS): Conservative Q-Learning calibrated for O2O fine-tuning. Direct match for the conservatism-scheduling theme.
  - **Lee et al. 2022** (CoRL): "Offline-to-Online RL via Balanced Replay and Pessimistic Q-Ensemble". Specifically about handling distribution shift during O2O.
  - Suggested bib entries:
    ```bibtex
    @inproceedings{nakamoto2023calql,
      title={Cal-{QL}: Calibrated Offline {RL} Pre-Training for Efficient Online Fine-Tuning},
      author={Nakamoto, Mitsuhiko and Zhai, Yuexiang and Singh, Anikait and Mark, Max Sobol and Ma, Yi and Finn, Chelsea and Kumar, Aviral and Levine, Sergey},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2023}
    }
    @inproceedings{lee2022offline,
      title={Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble},
      author={Lee, Seunghyun and Seo, Younggyo and Lee, Kimin and Abbeel, Pieter and Shin, Jinwoo},
      booktitle={Conference on Robot Learning (CoRL)},
      year={2022}
    }
    ```
- Alternative if the user does not want to add new references: drop `zhao2022adaptive` from the \citep{} group with no replacement; the surrounding sentence already cites `nair2020awac` and `zheng2023adaptive` (note [P5] below).

**Severity:** Submission-blocker.

---

### [P3] `dudik2014doubly` — STATUS: VERIFIED (existence) / MISATTRIBUTED (in-paper claim)

**Bib entry as written**
- Title: "Doubly Robust Policy Evaluation and Optimization"
- Authors: Dudík, Erhan, Langford, Li
- Venue field: `booktitle={Statistical Science}`, year 2014

**In-paper attribution**
- `draft_related_work.tex` lines 47–49: `\citep{dudik2014doubly,joachims2018deep,foerster2018counterfactual}` — cited as supporting "critic removal under exogenous transitions ... the bootstrapped continuation cancels in within-state action comparisons."

**Evidence**
- Real paper: Dudík, M., Erhan, D., Langford, J., & Li, L. (2014). "Doubly Robust Policy Evaluation and Optimization." *Statistical Science* 29(4): 485–511. arXiv:1503.02834. Project Euclid: https://projecteuclid.org/journals/statistical-science/volume-29/issue-4/Doubly-Robust-Policy-Evaluation-and-Optimization/10.1214/14-STS500.pdf
- Verified author list, venue, year. Existence and authorship: ✅ correct.
- Bib field type incorrect: `booktitle={Statistical Science}` should be `journal={Statistical Science}, volume={29}, number={4}, pages={485--511}`. *Statistical Science* is a peer-reviewed journal, not a proceedings.

**Misattribution issue**
- The Dudík paper studies **contextual bandits** (one-step decision problem; no bootstrap). Its core technique is the **doubly robust estimator** = inverse-propensity weighting + a learned reward model. It does *not* derive critic removal under exogenous transitions in MDPs. The contextual-bandit setting trivially satisfies an "exogeneity-like" property because there is no next state at all — but the paper makes no MDP-level claim of the kind being attributed.
- Cited alongside Joachims (also a contextual-bandit paper, see [P4]) and Foerster (multi-agent COMA — uses a counterfactual baseline, but in standard MDPs, not under exogeneity), the trio supports the *spirit* of "you can do policy gradient without a value baseline" but does not specifically support the technical claim made in the related-work paragraph.

**Recommended fixes**
1. Fix the bib type: change `booktitle` → `journal` and add volume/number/pages.
2. Either soften the related-work attribution to: *"... is conceptually related to direct policy-gradient estimators in contextual-bandit settings, where the absence of a bootstrap term makes within-state action comparisons critic-free [dudik2014doubly,joachims2018deep]; the multi-agent counterfactual-baseline construction of Foerster et al. is a closer technical analogue [foerster2018counterfactual]."*
   OR drop `dudik2014doubly` entirely and ground the exogeneity argument in the financial-policy-gradient lineage (Moody & Saffell 2001 already cited).

**Severity:** Medium. Not a fabrication — the citation is real — but the claim it is wedged into is not what the paper proves.

---

### [P4] `joachims2018deep` — STATUS: VERIFIED

**Bib entry as written**
- Title: "Deep Learning with Logged Bandit Feedback"
- Authors: Joachims, Thorsten and Swaminathan, Adith and de Rijke, Maarten
- Venue: ICLR 2018

**Evidence**
- Verified at https://www.cs.cornell.edu/~tj/publications/joachims_etal_18a.pdf and https://openreview.net/forum?id=SJaP_-xAb. Author list, venue, and year are correct. The method is BanditNet (counterfactual risk minimization with logged bandit feedback).

**Note: same soft-misattribution flag as [P3]** — this is a contextual-bandit paper, not an MDP-with-exogenous-transitions paper. The related-work claim is loose. Not a fabrication, but the intellectual-history sentence in §2 should be tightened: BanditNet's mathematical contribution (the equivariant/IPS-self-normalized estimator) is not the same as critic removal in an exogenous-transition MDP.

**Severity:** Low (citation correct) / Medium (attribution loose).

---

### [P5] Additional bibliography issues found during audit

These were not on your priority list but are flagged because the pattern of fabrication in [P1]/[P2] argues for re-verifying every entry in `refs.bib`.

1. **`zheng2023adaptive`** — author list appears wrong.
   - Bib: arXiv 2210.13846, "Adaptive Behavior Cloning Regularization for Stable Offline-to-Online Reinforcement Learning", authors Zheng, Yi and Li, Jianxiong and Yu, Dongjie and Yang, Yinan and Lin, Sergey E. and Zhao, Hang and Liu, Boyi and Zhan, Xianyuan.
   - Reality: arXiv 2210.13846 is real and the title is correct, but the actual authors are **Yi Zhao, Rinu Boney, Alexander Ilin, Juho Kannala, Joni Pajarinen** (5 authors, none of whom match the bibtex except first-name "Yi"). Verified at https://arxiv.org/abs/2210.13846.
   - The bibtex author list looks merged from a different paper (possibly Zheng et al. 2023 "Adaptive Policy Learning for Offline-to-Online RL", AAAI 2023). "Lin, Sergey E." is not a real researcher I can locate; likely a hallucinated interpolation.
   - **Fix:** Bibkey should be `zhao2022adaptive_o2o` (renaming to avoid colliding with the dropped [P2]). Authors: Zhao, Boney, Ilin, Kannala, Pajarinen.

2. **`yu2024rlhf`** — bibkey/first-author mismatch.
   - The first author is **Hanze Dong**, not "Yu". The bibkey should be renamed (e.g., `dong2024rlhf`). Bibtex authors are otherwise correct.

3. **`li2024grpoexp`** — placeholder still present.
   - The bib entry is explicitly marked "Placeholder". `grep -n li2024grpoexp writeup/draft_*.tex` confirms it is no longer cited. Remove the entry from `refs.bib` before submission.

4. **General recommendation:** Re-verify every entry in `refs.bib` against arXiv or the publisher's record. The fact that two of the three priority-audit entries are fabricated and a third has wrong authors is consistent with an LLM-generated bibliography. Treat the entire `refs.bib` as untrusted until each entry is independently verified.

---

## Pass A Summary

| Cite key | Status | Action required |
|---|---|---|
| `ma2022geodesic` | HALLUCINATED | Remove or replace; rename method internally |
| `zhao2022adaptive` | HALLUCINATED | Remove; replace with Cal-QL or Lee et al. 2022 |
| `dudik2014doubly` | VERIFIED / MISATTRIBUTED | Fix bib field type; soften related-work claim |
| `joachims2018deep` | VERIFIED | Tighten related-work attribution |
| `zheng2023adaptive` | WRONG AUTHORS | Fix author list (Zhao/Boney/Ilin/Kannala/Pajarinen) |
| `yu2024rlhf` | KEY MISMATCH | Rename bibkey to match first author |
| `li2024grpoexp` | UNUSED PLACEHOLDER | Delete from refs.bib |

---

## Pass B.5 — Final Rigor Scan (flags only; not auto-edited)

These are remaining overclaim/unsupported-mechanism issues that the B.1
substitutions did not cover. Each is a judgement call for the author.

### Abstract (`draft_abstract.tex`)

- **L30 — "reliably converges to a near-static allocation in the top 1.25%"**
  "reliably" with n=3 seeds. Suggest "consistently across our three seeds converges to" or "converges, on the three seeds we ran, to".
- **L54 — "an eight-fold variance reduction in Sharpe space"**
  std estimates from n=3 are extremely noisy (relative error ~70–100%). The "8×" point estimate may not be meaningful. Suggest reporting both ranges literally without a ratio, or qualifying as "≈ 8-fold on this small-sample comparison".
- **L70–73 — "Every RL method we evaluate either degenerates to a static allocation or trades into the friction tax; adaptive O2O is the only configuration that lands a static allocation reliably in the top of the simplex distribution."**
  Stacking "every", "the only", "reliably" on n=3 evidence. Suggest: "Of the RL methods we evaluated, all either degenerated to a static allocation or traded into the friction tax; adaptive O2O is the only configuration whose three seeds all landed in the top of the simplex distribution."
- **L83–87 — "we propose this as a STOP condition for offline-RL benchmark builders"**
  Prescriptive on n=1 dataset. Suggest "we propose this as a diagnostic worth checking when building offline-RL benchmarks".

### Introduction (`draft_introduction.tex`)

- **L37–38 — "reliably converges to a near-static allocation in the top 1.25%"** (same as abstract).
- **L60 — "Naive's pinned conservatism leaves the basin random across seeds"**
  Causal "leaves" claim from n=3. Suggest "Naive's pinned conservatism is associated with random-across-seeds basin selection in our three runs".
- **L62–63 — "an eight-fold seed-std reduction over naive"**
  Same n=3 std issue as abstract.
- **L51–52 — "surfacing the revision properly is the second-order contribution of this paper"**
  Insider framing. A workshop reviewer with no project history will find "second-order contribution" confusing. Suggest "The mechanism we initially hypothesized is not what the data show; we revise to a basin-selection account, which is part of our contribution."

### Method (`draft_method.tex`)

- **L118 — "A subtle but load-bearing detail"**
  "load-bearing" jargon still here. Suggest "A subtle but important detail".
- **L152–156 — TODO comment about importance ratio framing**
  Internal TODO comment in the source. Comments are stripped at compile but workshop reviewers sometimes diff the source bundle. Remove before submission.
- **L209–220 — "The bug fix that made the comparison meaningful"**
  Prominent main-method discussion of an internal pipeline bug. Reads as "trust issues with our own pipeline" to a workshop reviewer. Suggest moving the entire paragraph to the appendix and replacing it with a one-sentence forward-pointer.
- **L230–233 — TODO comment about cql_weight_traj.npy**
  Internal TODO. Remove before submission.

### Experiments (`draft_experiments.tex`)

- **L137–141 — "every online RL method we ran either degenerates ... or actively trades into a friction penalty"**
  "Every" claim from 3 methods × 3 seeds. Acceptable if scoped to "we ran"; current wording is fine but flag for consistency with abstract softening.
- **L142 — "GRPO is the only online method that produces a non-degenerate trading policy"**
  "the only" claim from 3 online methods. Acceptable scope-wise, but consider "the only online method in our evaluation".
- **L142–143 — "it does so at net Sharpe cost"**
  Causal "does so" framing from n=3 of a single method. Acceptable.
- **L320–321 — "the schedule does \emph{select which static} the policy converges to"**
  This was target (g) in PASS B.1. **UNMATCHED** — the source has `\emph{}` markup that broke the exact-substring match. Recommended manual fix: replace with "the seed-clustering differential is consistent with the schedule influencing which static allocation the policy converges to" (drop the \emph{}), or keep the structure but soften: "the schedule appears to \emph{select which static} the policy converges to".
- **L326–328 — "The adaptive contribution is therefore not 'adaptive trades better than naive,' but 'adaptive selects a better basin than naive --- and does so consistently.'"**
  "consistently" with n=3. Suggest "and does so on all three seeds we ran".
- **L360 — "O2O lands reliably above the closed-form min-variance baseline"**
  "reliably" with n=3. Suggest "lands above the closed-form min-variance baseline on all three seeds we ran".
- **L370–375 — multiple stacked "every", "the only", "the only" claims**
  After the B.1 softening of "the only RL method to reliably clear", the surrounding text still has "Every other RL condition we run has at least one seed below..." and "Adaptive O2O is the only configuration in our study that places \emph{all three} seeds...". These are scope-bound to "in our study" and are factually accurate; the cumulative rhetorical effect is strong though. Optional softening.

### Limitations / Discussion (`draft_limitations.tex`)

- **L31 — "consistently steers the same SAC dynamics"**
  Covered by B.1. After substitution: "appears to consistently steer the same SAC dynamics".
- **L42 — "is therefore not consistent with the null"**
  Covered by B.4 (caveat sentence inserted after).
- **L46 — "We surface this as a mechanism revision rather than a confirmation"**
  Acceptable framing. No flag.
- **L53–55 — "we report it as a hypothesis generated by the data, to be tested in future O2O work"**
  Already a soft-claim caveat. Good.
- **L56 — "load-bearing: it shapes which fixed point of the SAC fine-tune dynamics"**
  Another "load-bearing" usage. Suggest "the relevant load is shaping which fixed point ..." → just rewrite to drop "load-bearing": "the schedule's effect is to shape which fixed point ...".
- **L161 — "$\beta_t$-trajectory verification was not explicitly verified from the saved Phase~2C artifacts"**
  Already flagged in limitations as follow-up. ✅ Good as-is.

### Cross-cutting

- **n=3 / 1-window scoping.** The abstract and introduction make many universal-sounding claims that are scoped only by the methods section / limitations. The discrepancy between abstract rhetoric and limitations honesty is the largest single rhetoric problem. Softening pass should propagate the limitations-section caveats into the abstract and intro headline sentences.
- **"Mechanism" used in mixed senses.** "Mechanism" in this paper means (a) algorithmic computation, (b) explanatory causal story, (c) statistical pattern. Disambiguating to "mechanism / pathway / pattern / explanation" depending on context would help readers distinguish empirical observation from interpretation.

---

## Pass B.5 flag count

- Abstract: 4 flags
- Introduction: 4 flags
- Method: 4 flags (incl. 2 internal TODO comments)
- Experiments: 5 flags (1 of which is the UNMATCHED B.1 target (g))
- Limitations: 2 flags
- Cross-cutting: 2 flags
- **Total: 21 flags**
