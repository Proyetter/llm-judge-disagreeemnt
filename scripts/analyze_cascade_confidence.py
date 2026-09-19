#!/usr/bin/env python3
"""
analyze_cascade_confidence.py
---------------------------------
A confidence-TRIGGERED cascade, as opposed to the consistency-triggered one
in analyze_cascade.py. Escalate to the strong judge when the cheap judge's
OWN single-shot confidence (its first call, before any other calls exist)
is low. This only makes sense for a cheap judge whose confidence is
actually validated as informative -- check analyze_confidence_v2.py's
SS-position AUROC first. In this project, that means Gemini 2.5 Flash or
Claude Sonnet 5, not GPT-4o-mini/DeepSeek Chat/Llama (their confidence
CIs mostly include 0.5, i.e. not reliably better than chance).

METHOD (held out, not in-sample)
-----------------------------------
  1. Split shared pairs into CALIBRATION (40%) and TEST (60%).
  2. On CALIBRATION only, find the confidence threshold that escalates
     roughly a target fraction of pairs (the lowest-confidence ones).
  3. FREEZE that threshold, apply it to TEST (which never touched the
     threshold choice).
  4. Report cascade accuracy on TEST, plus a RANDOM-ESCALATION baseline
     (escalate the same NUMBER of test pairs, chosen at random) so you can
     tell whether confidence-based selection beats simply escalating a
     random subset of the same size.

This is a genuinely cheaper cascade than the one in analyze_cascade.py: it
only needs the cheap judge's FIRST call before deciding whether to spend
more, rather than all 6 of its calls.

Usage:
    python analyze_cascade_confidence.py --cheap google_gemini-2.5-flash --strong anthropic_claude-sonnet-5 --split gpt
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci
from analyze_pair_classes import six_vote_decision


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
    """Per-pair: x, y, n, gt, and single-shot confidence (order=1, rep_idx=0)."""
    by_item = defaultdict(lambda: {1: {}, 2: {}})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]][r["rep_idx"]] = r
        gt_of[r["pair_id"]] = r["gt"]

    items = {}
    for pair_id, orders in by_item.items():
        o1_rows = [orders[1][k] for k in sorted(orders[1]) if k < n_reps]
        o2_rows = [orders[2][k] for k in sorted(orders[2]) if k < n_reps]
        if len(o1_rows) < n_reps or len(o2_rows) < n_reps or 0 not in orders[1]:
            continue
        x = sum(1 for r in o1_rows if r["picked_original"] == "A")
        y = sum(1 for r in o2_rows if r["picked_original"] == "A")
        items[pair_id] = {"x": x, "y": y, "n": n_reps, "gt": gt_of[pair_id],
                          "single_shot_conf": orders[1][0]["confidence"]}
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument("--split", default="gpt", choices=["gpt", "claude"])
    ap.add_argument("--calib-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1500)
    args = ap.parse_args()

    cheap_path = f"factorial_judgebench__{args.cheap}__{args.split}.jsonl"
    strong_path = f"factorial_judgebench__{args.strong}__{args.split}.jsonl"

    cheap_rows = load_valid(cheap_path)
    strong_rows = load_valid(strong_path)
    if not cheap_rows or not strong_rows:
        print(f"Could not load one or both files:\n  {cheap_path}\n  {strong_path}")
        return

    cheap_items = build_items(cheap_rows)
    strong_items = build_items(strong_rows)
    shared_ids = sorted(set(cheap_items) & set(strong_items))
    if len(shared_ids) < 30:
        print(f"Only {len(shared_ids)} shared pairs -- too few for a calib/test split.")
        return

    records = []
    for pid in shared_ids:
        c, s = cheap_items[pid], strong_items[pid]
        gt = c["gt"]
        cheap_dec, cheap_tie, cheap_correct = six_vote_decision(c["x"], c["y"], c["n"], gt)
        strong_dec, strong_tie, strong_correct = six_vote_decision(s["x"], s["y"], s["n"], gt)
        records.append({
            "pair_id": pid, "conf": c["single_shot_conf"],
            "cheap_tie": cheap_tie, "cheap_correct": cheap_correct,
            "strong_tie": strong_tie, "strong_correct": strong_correct,
        })

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))
    n_calib = int(round(len(records) * args.calib_frac))
    calib = [records[i] for i in order[:n_calib]]
    test = [records[i] for i in order[n_calib:]]

    def cascade_accuracy(sample, threshold):
        outcomes = []
        n_escalated = 0
        for r in sample:
            if r["conf"] < threshold:
                n_escalated += 1
                if not r["strong_tie"]:
                    outcomes.append(r["strong_correct"])
            else:
                if not r["cheap_tie"]:
                    outcomes.append(r["cheap_correct"])
        acc = float(np.mean(outcomes)) if outcomes else None
        return acc, n_escalated

    def cheap_only_accuracy(sample):
        outcomes = [r["cheap_correct"] for r in sample if not r["cheap_tie"]]
        return float(np.mean(outcomes)) if outcomes else None

    def strong_only_accuracy(sample):
        outcomes = [r["strong_correct"] for r in sample if not r["strong_tie"]]
        return float(np.mean(outcomes)) if outcomes else None

    print("=" * 100)
    print(f"CONFIDENCE-TRIGGERED CASCADE: {args.cheap} (cheap) -> {args.strong} (strong)  "
          f"[{args.split} split]")
    print(f"Calibration: {len(calib)} pairs   |   Held-out test: {len(test)} pairs")
    print("=" * 100)

    targets = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
    print(f"{'TargetEsc':>10}{'Thresh':>9}{'TestEsc%':>10}{'CascAcc':>10}{'95% CI':>16}"
          f"{'RandomAcc':>11}{'Gain':>8}{'TotalCalls':>12}")
    print("-" * 100)

    calib_confs = np.array([r["conf"] for r in calib])
    for target in targets:
        threshold = float(np.quantile(calib_confs, target))  # escalate bottom `target` fraction

        casc_acc, n_esc = cascade_accuracy(test, threshold)
        if casc_acc is None:
            continue
        test_esc_frac = n_esc / len(test)

        def casc_stat(sample, thr=threshold):
            a, _ = cascade_accuracy(sample, thr)
            return a if a is not None else 0.5
        _, lo, hi = bootstrap_ci(test, casc_stat, args.n_boot, seed=0)

        # random-escalation baseline: escalate the SAME NUMBER of test pairs,
        # chosen at random, repeated many times
        rng2 = np.random.default_rng(1)
        rand_accs = []
        idx = np.arange(len(test))
        for _ in range(500):
            chosen = set(rng2.choice(idx, size=n_esc, replace=False).tolist())
            outcomes = []
            for i, r in enumerate(test):
                if i in chosen:
                    if not r["strong_tie"]:
                        outcomes.append(r["strong_correct"])
                else:
                    if not r["cheap_tie"]:
                        outcomes.append(r["cheap_correct"])
            if outcomes:
                rand_accs.append(np.mean(outcomes))
        rand_acc = float(np.mean(rand_accs)) if rand_accs else None

        gain = (casc_acc - rand_acc) if rand_acc is not None else None
        total_calls = 6 * len(test) + 6 * n_esc

        rand_s = f"{rand_acc*100:.1f}%" if rand_acc is not None else "n/a"
        gain_s = f"{gain*100:+.1f}pp" if gain is not None else "n/a"
        print(f"{target*100:>9.0f}%{threshold:>9.3f}{test_esc_frac*100:>9.1f}%"
              f"{casc_acc*100:>9.1f}%[{lo*100:.1f},{hi*100:.1f}]".rjust(16) +
              f"{rand_s:>11}{gain_s:>8}{total_calls:>12}")

    cheap_acc = cheap_only_accuracy(test)
    strong_acc = strong_only_accuracy(test)
    print("-" * 100)
    print(f"Reference: Cheap-only test accuracy  = {cheap_acc*100:.1f}%  "
          f"(0% escalated, {6*len(test)} calls)")
    print(f"Reference: Strong-only test accuracy = {strong_acc*100:.1f}%  "
          f"(100% escalated, {6*len(test)*2} calls)")
    print("=" * 100)
    print("TargetEsc  = fraction of pairs we AIMED to escalate, chosen on CALIBRATION")
    print("Thresh     = the frozen confidence cutoff (chosen without seeing test data)")
    print("TestEsc%   = fraction ACTUALLY escalated on the held-out test set")
    print("CascAcc    = accuracy of the confidence-triggered cascade on test")
    print("RandomAcc  = accuracy if you escalated the SAME NUMBER of test pairs at")
    print("             RANDOM instead of by low confidence")
    print("Gain       = CascAcc - RandomAcc: the value confidence-based selection")
    print("             adds over just escalating a random subset of the same size.")
    print("             If Gain is ~0, the cheap judge's confidence isn't actually")
    print("             choosing better pairs to escalate than chance would.")


if __name__ == "__main__":
    main()
