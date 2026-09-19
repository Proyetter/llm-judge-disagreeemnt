# Scripts

All scripts live flat in this directory on purpose. `judge_judgebench_factorial.py`
and `judge_judgebench_repeat.py` import `load_pairs` and `call_judge` from
`judge_judgebench.py`, and most analysis scripts import from `bootstrap_utils.py`.
Python puts the running script's own directory on the import path, so these
imports resolve as long as the files stay together. Moving any of them into a
subdirectory breaks the imports.

Analysis scripts find their input by globbing `factorial_judgebench__*.jsonl`
in the current working directory, which is why they are run from `data/`.

## Collection

| Script | Role |
| --- | --- |
| `judge_judgebench.py` | Defines `PROMPT_TEMPLATE`, response parsing, `load_pairs`, and `call_judge`. Also runs the earlier one-call-per-ordering swap design. Required as a dependency even if you never run it directly. |
| `judge_judgebench_factorial.py` | **The collection script for this paper.** Runs the repeated-order design: each pair judged `--n-reps` times under AB and `--n-reps` times under BA. Writes `factorial_judgebench__{model}__{split}.jsonl`. |
| `judge_judgebench_repeat.py` | Earlier design that repeated a single ordering to estimate a noise floor. Superseded by the factorial script. Kept because the record format migration mentioned in `configs/collection.md` originated here. |
| `inspect_judgebench.py` | One-liner that prints the JudgeBench splits and one example row. Useful for confirming the dataset fields. |

## Reported in the paper

| Script | Produces |
| --- | --- |
| `audit_factorial.py` | Data-integrity audit: row counts, duplicate keys, six-call completeness, parse-error rates broken down by ordering and by repeat index, confidence range validity, confidence granularity. Source of the parse-failure figures in Methods. Run this first. |
| `analyze_decomposition.py` | Table II. `D_cross`, `D_within`, `S`, the bootstrap confidence interval on mean `S`, and NoiseShare, per judge and split. |
| `analyze_pair_classes.py` | Table III, the stable / noise-only / position-only / tangled category shares. |
| `analyze_conditional_noise.py` | Table III, `P(noisy | reversal)`, `P(noisy | no reversal)`, their difference in percentage points, and the bootstrap interval on that difference. |
| `analyze_confidence_v2.py` | Table IV. First-call (single-shot) confidence AUROC against majority reversal, within-order instability, either form, and pooled-label correctness. The `SS-` columns are the ones reported. |
| `bootstrap_utils.py` | Shared pair-level cluster bootstrap and rank-based AUROC. Imported by several scripts above. Not run directly. |

## Supplementary, not reported in the paper

These were run during the study and are included for completeness. Some read
input files from the earlier swap or repeat designs (`judgements_judgebench__*.jsonl`,
`repeat_test_judgebench__*.jsonl`) which are not included in `data/`, so they
will find no input unless you add those files.

| Script | Question it answers |
| --- | --- |
| `analyze_order_effect.py` | Signed order effect `p - q` per item: net primacy versus recency across the corpus. |
| `analyze_coverage.py` | Coverage, accuracy on non-tied items, and effective yield. |
| `analyze_cross_model_agreement_v2.py` | Do different judges reverse on the same items? Pairwise Cohen's kappa. |
| `analyze_kappa_bootstrap.py` | Corrected confidence interval for the mean pairwise kappa, resampling response pairs rather than judge pairings. |
| `analyze_item_difficulty.py` | Is order sensitivity concentrated on harder items? Difficulty computed leave-one-model-out. |
| `analyze_margin_v2.py` | Does preference margin predict instability? Includes the documented warning about the broken combined-vote margin formula. |
| `analyze_repeat_sensitivity.py` | How much would the flip-rate estimate move if only one repeat per ordering had been run, as in the standard design? |
| `analyze_selective_judging_heldout.py` | Held-out confidence threshold sweep with a random-abstention baseline. |
| `analyze_cascade.py` | Consistency-triggered escalation to a stronger judge. Retrospective best case. |
| `analyze_cascade_confidence.py` | Confidence-triggered escalation with a frozen held-out threshold. |
| `compare_splits.py` | Cross-split replication checks. Note: this file's docstring is a stale copy of `analyze_confidence_v2.py` and describes the wrong script. |

## Known issues

- `compare_splits.py` has an incorrect docstring, as noted above. The code is
  unaffected. Fix the docstring or note it before publishing.
- `analyze_cross_model_agreement_v2.py` computes the 21 pairwise kappa values
  correctly but its confidence interval on the mean resamples the kappa values
  themselves rather than response pairs. `analyze_kappa_bootstrap.py` exists to
  supersede that interval. Use the latter.
