#!/usr/bin/env python3
"""
analyze_cascade.py
---------------------
Simulates a practical policy: use a CHEAP judge by default; escalate to a
STRONG judge only when the cheap judge shows signs of instability.

POLICY
------
For each pair, look at the cheap judge's OWN classification (from its 6
calls): if it is Stable (N=0, P=0), keep the cheap judge's six-vote
decision. Otherwise (NoiseOnly, PosOnly, or Tangled), use the STRONG
judge's six-vote decision instead.

IMPORTANT HONESTY NOTE: this is a RETROSPECTIVE, best-case estimate. It
uses the cheap judge's FULL 6-call instability classification as the
escalation trigger, which is only available because we already collected
all 6 calls for this study. A live deployment would need a CHEAPER trigger
(e.g. single-shot confidence, from analyze_confidence_v2.py) to decide
whether to escalate without first paying for the cheap judge's full 6
calls. This script measures the ACCURACY side of the tradeoff exactly; the
COST side assumes the cheap judge's 6 calls are already being paid for
regardless (a fair assumption if you want the cheap judge's own repeated-
call reliability check anyway), and adds the strong judge's 6 calls only
for the escalated subset.

COMPARISON
----------
  Cheap-only   : always use the cheap judge's decision.   Cost = 6N
  Strong-only  : always use the strong judge's decision.  Cost = 6N
  Cascade      : cheap by default, strong on escalation.  Cost = 6N + 6*n_escalated

Usage:
    python analyze_cascade.py --cheap openai_gpt-4o-mini --strong anthropic_claude-sonnet-5
    python analyze_cascade.py --cheap openai_gpt-4o-mini --strong anthropic_claude-sonnet-5 --split claude
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci
from analyze_pair_classes import classify_item, six_vote_decision


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
    by_item = defaultdict(lambda: {1: [], 2: []})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)
        gt_of[r["pair_id"]] = r["gt"]

    items = {}
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            x, y = sum(o1[:n_reps]), sum(o2[:n_reps])
            items[pair_id] = {"x": x, "y": y, "n": n_reps, "gt": gt_of[pair_id]}
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheap", required=True, help="cheap model's file-safe slug, "
                    "e.g. openai_gpt-4o-mini")
    ap.add_argument("--strong", required=True, help="strong model's file-safe slug, "
                    "e.g. anthropic_claude-sonnet-5")
    ap.add_argument("--split", default="gpt", choices=["gpt", "claude"])
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
    if len(shared_ids) < 10:
        print(f"Only {len(shared_ids)} pairs shared between the two models -- too few.")
        return

    records = []
    for pid in shared_ids:
        c = cheap_items[pid]
        s = strong_items[pid]
        gt = c["gt"]  # both should agree; ground truth is a property of the pair

        N, P = classify_item(c["x"], c["y"], c["n"])
        cheap_stable = (N == 0 and P == 0)

        cheap_dec, cheap_tie, cheap_correct = six_vote_decision(c["x"], c["y"], c["n"], gt)
        strong_dec, strong_tie, strong_correct = six_vote_decision(s["x"], s["y"], s["n"], gt)

        records.append({
            "pair_id": pid, "cheap_stable": cheap_stable,
            "cheap_tie": cheap_tie, "cheap_correct": cheap_correct,
            "strong_tie": strong_tie, "strong_correct": strong_correct,
        })

    n_total = len(records)
    n_escalated = sum(1 for r in records if not r["cheap_stable"])

    def accuracy_of(policy_fn, sample):
        outcomes = [policy_fn(r) for r in sample]
        outcomes = [o for o in outcomes if o is not None]  # exclude ties
        return float(np.mean(outcomes)) if outcomes else None

    def cheap_only(r):
        return None if r["cheap_tie"] else r["cheap_correct"]

    def strong_only(r):
        return None if r["strong_tie"] else r["strong_correct"]

    def cascade(r):
        if r["cheap_stable"]:
            return None if r["cheap_tie"] else r["cheap_correct"]
        return None if r["strong_tie"] else r["strong_correct"]

    acc_cheap = accuracy_of(cheap_only, records)
    acc_strong = accuracy_of(strong_only, records)
    acc_cascade = accuracy_of(cascade, records)

    def cascade_stat(sample):
        v = accuracy_of(cascade, sample)
        return v if v is not None else 0.5

    def cheap_stat(sample):
        v = accuracy_of(cheap_only, sample)
        return v if v is not None else 0.5

    def strong_stat(sample):
        v = accuracy_of(strong_only, sample)
        return v if v is not None else 0.5

    _, casc_lo, casc_hi = bootstrap_ci(records, cascade_stat, args.n_boot, seed=0)
    _, cheap_lo, cheap_hi = bootstrap_ci(records, cheap_stat, args.n_boot, seed=1)
    _, strong_lo, strong_hi = bootstrap_ci(records, strong_stat, args.n_boot, seed=2)

    cost_cheap_only = 6 * n_total
    cost_strong_only = 6 * n_total
    cost_cascade = 6 * n_total + 6 * n_escalated

    print("=" * 90)
    print(f"CASCADE SIMULATION: {args.cheap} (cheap) -> {args.strong} (strong)  [{args.split} split]")
    print("=" * 90)
    print(f"Shared pairs: {n_total}   |   Escalated (cheap was unstable): {n_escalated} "
          f"({100*n_escalated/n_total:.1f}%)")
    print()
    print(f"{'Policy':<16}{'Accuracy':>12}{'95% CI':>18}{'API calls':>12}{'Calls vs strong-only':>22}")
    print("-" * 90)
    print(f"{'Cheap-only':<16}{acc_cheap*100:>11.1f}%{f'[{cheap_lo*100:.1f},{cheap_hi*100:.1f}]':>18}"
          f"{cost_cheap_only:>12}{100*cost_cheap_only/cost_strong_only:>21.1f}%")
    print(f"{'Strong-only':<16}{acc_strong*100:>11.1f}%{f'[{strong_lo*100:.1f},{strong_hi*100:.1f}]':>18}"
          f"{cost_strong_only:>12}{'100.0%':>22}")
    print(f"{'Cascade':<16}{acc_cascade*100:>11.1f}%{f'[{casc_lo*100:.1f},{casc_hi*100:.1f}]':>18}"
          f"{cost_cascade:>12}{100*cost_cascade/cost_strong_only:>21.1f}%")
    print("=" * 90)
    print("Cascade accuracy should sit between Cheap-only and Strong-only. The key")
    print("question: does it get CLOSE to Strong-only accuracy while costing")
    print("meaningfully LESS than always using the strong judge?")
    print()
    print("HONESTY NOTE: this is a retrospective, best-case estimate -- it assumes")
    print("the cheap judge's instability status is known from its own 6 calls,")
    print("which is how this dataset was collected. A live system would need a")
    print("cheaper trigger (e.g. single-shot confidence) to decide whether to")
    print("escalate without paying for all 6 cheap-judge calls first.")


if __name__ == "__main__":
    main()
