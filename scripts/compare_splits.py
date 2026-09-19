#!/usr/bin/env python3
"""
analyze_confidence_v2.py
----------------------------
Ke Fang's question, answered properly: does a judge's confidence predict
which of its own verdicts are unstable? This version fixes a subtle but
important flaw in earlier versions of this analysis.

THE FLAW IN THE EARLIER VERSION
---------------------------------
Earlier scripts averaged confidence across BOTH orderings (or all repeats)
to build the "confidence predicts flip" signal. But that average is only
computable AFTER you've already made the extra (expensive) calls you were
trying to decide whether to make. It answers a scientific question fine,
but it cannot be turned into an actual deployable policy: "check confidence,
then decide whether to double-check" requires a confidence value you have
BEFORE the second call exists.

THIS VERSION distinguishes two signals:
  SINGLE-SHOT (primary, deployable): confidence from ONLY the very first
      call made for a pair (order=1, rep_idx=0 by convention). This is the
      only signal available before deciding whether to spend more calls.
  SIX-CALL (ceiling, NOT deployable): mean confidence across all 6 calls.
      Useful as a scientific upper bound on how good the signal COULD be
      with unlimited budget, but cannot inform a real cost-saving decision,
      since computing it requires the very calls you'd be deciding whether
      to make.

TARGETS PREDICTED
-------------------
  position : does the majority verdict flip between AB and BA (P_i)
  noise    : is there any within-order disagreement (N_i)
  any      : position OR noise
  correct  : does the six-vote majority decision match ground truth
             (only correctness uses the SAME direction as normal calibration:
             higher confidence should predict correct=1; all others use
             lower confidence predicting instability=1)

Usage:
    python analyze_confidence_v2.py
"""

import glob
import json
import re
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci, compute_auroc
from analyze_pair_classes import classify_item, six_vote_decision


def parse_filename(path):
    m = re.search(r"factorial_judgebench__(.+)__(gpt|claude|both)\.jsonl", path)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def pretty(slug):
    return slug.replace("_", "/", 1)


def load_valid(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("parse_error", False):
                rows.append(rec)
    return rows


def build_items(rows, n_reps=3):
    """Returns per-pair records with x, y, n, gt, single-shot confidence
    (order=1, rep_idx=0), and six-call mean confidence."""
    by_item = defaultdict(lambda: {1: {}, 2: {}})  # order -> {rep_idx: row}
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]][r["rep_idx"]] = r
        gt_of[r["pair_id"]] = r["gt"]

    items = []
    for pair_id, orders in by_item.items():
        o1_rows = [orders[1][k] for k in sorted(orders[1]) if k < n_reps]
        o2_rows = [orders[2][k] for k in sorted(orders[2]) if k < n_reps]
        if len(o1_rows) < n_reps or len(o2_rows) < n_reps:
            continue

        x = sum(1 for r in o1_rows if r["picked_original"] == "A")
        y = sum(1 for r in o2_rows if r["picked_original"] == "A")

        # single-shot: the designated "first" call, order=1 rep_idx=0
        if 0 not in orders[1]:
            continue  # can't define single-shot signal for this pair, skip
        single_shot_conf = orders[1][0]["confidence"]

        all_confs = [r["confidence"] for r in o1_rows + o2_rows]
        six_call_conf = float(np.mean(all_confs))

        items.append({
            "pair_id": pair_id, "x": x, "y": y, "n": n_reps, "gt": gt_of[pair_id],
            "single_shot_conf": single_shot_conf, "six_call_conf": six_call_conf,
        })
    return items


def add_targets(item):
    """Adds binary target labels to an item: position, noise, any, correct."""
    x, y, n, gt = item["x"], item["y"], item["n"], item["gt"]
    N, P = classify_item(x, y, n)
    decision, is_tie, is_correct = six_vote_decision(x, y, n, gt)
    item["noise"] = N
    item["position"] = P
    item["any"] = int(N == 1 or P == 1)
    item["is_tie"] = is_tie
    item["correct"] = is_correct  # None if tie
    return item


def auroc_for_target(items, target, signal_key):
    """AUROC for predicting `target` from `signal_key`. For instability
    targets (position/noise/any), score = -confidence (low conf -> high
    score -> predicts instability=1). For 'correct', score = +confidence."""
    usable = [it for it in items if it.get(target) is not None]
    if not usable:
        return None
    labels = [it[target] for it in usable]
    if target == "correct":
        scores = [it[signal_key] for it in usable]
    else:
        scores = [-it[signal_key] for it in usable]
    return compute_auroc(scores, labels)


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    targets = ["position", "noise", "any", "correct"]

    print("=" * 120)
    print("CONFIDENCE PREDICTS INSTABILITY: single-shot (deployable) vs six-call (ceiling)")
    print("=" * 120)
    header = f"{'Model':<30}{'Split':>7}{'n':>6}"
    for t in targets:
        header += f"{'SS-'+t:>12}{'6c-'+t:>12}"
    print(header)
    print("-" * 120)

    for path in sorted(files, key=lambda p: (parse_filename(p)[0], parse_filename(p)[1])):
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_items(rows)
        if len(items) < 10:
            continue
        items = [add_targets(it) for it in items]

        row = f"{pretty(slug):<30}{split:>7}{len(items):>6}"
        for t in targets:
            auroc_ss = auroc_for_target(items, t, "single_shot_conf")
            auroc_6c = auroc_for_target(items, t, "six_call_conf")
            ss_str = f"{auroc_ss:.3f}" if auroc_ss is not None else "n/a"
            c6_str = f"{auroc_6c:.3f}" if auroc_6c is not None else "n/a"
            row += f"{ss_str:>12}{c6_str:>12}"
        print(row)

    print("=" * 120)
    print("SS-<target>  = single-shot AUROC (confidence from the FIRST call only --")
    print("               the only signal available before deciding whether to")
    print("               spend more API calls on this pair. THIS is what a real")
    print("               selective-judging policy would actually use.)")
    print("6c-<target>  = six-call AUROC (mean confidence across all 6 calls --")
    print("               a scientific ceiling. NOT deployable, since computing it")
    print("               requires the very calls a real policy would be deciding")
    print("               whether to make.)")
    print()
    print("Targets: position = majority flips between AB/BA; noise = any within-")
    print("order disagreement; any = position OR noise; correct = six-vote")
    print("decision matches ground truth (this one uses confidence in the NORMAL")
    print("direction -- high confidence should predict correct=1 -- while position/")
    print("noise/any use the INSTABILITY direction -- low confidence should predict")
    print("instability=1. 0.5 = uninformative, 1.0 = perfect, for all columns.)")

    # ---- detailed single-shot-only report, with bootstrap CIs (the deployable number) ----
    print()
    print("=" * 120)
    print("SINGLE-SHOT AUROC WITH 95% BOOTSTRAP CI (the number that matters for deployment)")
    print("=" * 120)
    print(f"{'Model':<30}{'Split':>7}" + "".join(f"{t:>22}" for t in targets))
    print("-" * 120)
    for path in sorted(files, key=lambda p: (parse_filename(p)[0], parse_filename(p)[1])):
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_items(rows)
        if len(items) < 10:
            continue
        items = [add_targets(it) for it in items]

        row = f"{pretty(slug):<30}{split:>7}"
        for t in targets:
            usable = [it for it in items if it.get(t) is not None]
            if len(usable) < 10:
                row += f"{'n/a':>22}"
                continue

            def stat_fn(sample, t=t):
                v = auroc_for_target(sample, t, "single_shot_conf")
                return v if v is not None else 0.5

            point, lo, hi = bootstrap_ci(usable, stat_fn, n_boot=1000, seed=0)
            row += f"{point:>7.3f} [{lo:.2f},{hi:.2f}]"
        print(row)

    print("=" * 120)
    print("A CI that excludes 0.5 means single-shot confidence is a genuinely")
    print("usable, deployable signal for that target on that model. A CI that")
    print("includes 0.5 means: don't build a cost-saving policy around this")
    print("signal for this model/target -- it isn't reliably better than chance.")


if __name__ == "__main__":
    main()
