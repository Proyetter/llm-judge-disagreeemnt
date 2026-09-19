#!/usr/bin/env python3
"""
analyze_coverage.py
----------------------
Reports coverage alongside accuracy, as promised in the Methods section.

COVERAGE is the proportion of eligible items that receive a non-tied final
label under the pooled six-vote decision. Because ties are excluded from
the accuracy denominator, accuracy alone overstates how much usable output
a judge actually produces: a judge can be accurate on the items it decides
while failing to decide most items at all.

EFFECTIVE YIELD combines the two: of 100 input pairs, how many emerge with
a label that is both defined and correct.

    coverage        = (items with a non-tied decision) / (all valid items)
    accuracy        = (correct decisions) / (non-tied decisions)
    effective yield = coverage x accuracy

Effective yield is the number that matters if the judge is being used to
construct a labeled dataset, since undecided items and incorrectly decided
items are both unusable.

Usage:
    python analyze_coverage.py
"""

import glob
import json
import re
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci
from analyze_pair_classes import six_vote_decision


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
    by_item = defaultdict(lambda: {1: [], 2: []})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)
        gt_of[r["pair_id"]] = r["gt"]

    items = []
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            x, y = sum(o1[:n_reps]), sum(o2[:n_reps])
            _, is_tie, is_correct = six_vote_decision(x, y, n_reps, gt_of[pair_id])
            items.append({"pair_id": pair_id, "is_tie": is_tie,
                          "is_correct": None if is_tie else int(is_correct)})
    return items


def coverage_stat(items):
    if not items:
        return 0.0
    return sum(1 for it in items if not it["is_tie"]) / len(items)


def accuracy_stat(items):
    decided = [it for it in items if not it["is_tie"]]
    if not decided:
        return 0.0
    return sum(it["is_correct"] for it in decided) / len(decided)


def yield_stat(items):
    """coverage x accuracy, computed directly rather than as a product of two
    separately-bootstrapped numbers, so the CI accounts for the fact that both
    components come from the same resampled items."""
    if not items:
        return 0.0
    return sum(1 for it in items if not it["is_tie"] and it["is_correct"] == 1) / len(items)


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 112)
    print("COVERAGE, ACCURACY, AND EFFECTIVE YIELD")
    print("=" * 112)
    print(f"{'Model':<30}{'Split':>7}{'n':>6}{'Ties':>6}"
          f"{'Coverage':>11}{'95% CI':>16}{'Accuracy':>11}{'95% CI':>16}"
          f"{'EffYield':>11}{'95% CI':>16}")
    print("-" * 112)

    rows_out = []
    for path in sorted(files, key=lambda p: (parse_filename(p)[0], parse_filename(p)[1])):
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_items(rows)
        if len(items) < 20:
            continue

        n = len(items)
        n_ties = sum(1 for it in items if it["is_tie"])

        cov, cov_lo, cov_hi = bootstrap_ci(items, coverage_stat, 2000, seed=0)
        acc, acc_lo, acc_hi = bootstrap_ci(items, accuracy_stat, 2000, seed=1)
        yld, yld_lo, yld_hi = bootstrap_ci(items, yield_stat, 2000, seed=2)

        print(f"{pretty(slug):<30}{split:>7}{n:>6}{n_ties:>6}"
              f"{cov*100:>10.1f}%" + f"[{cov_lo*100:.1f},{cov_hi*100:.1f}]".rjust(16) +
              f"{acc*100:>10.1f}%" + f"[{acc_lo*100:.1f},{acc_hi*100:.1f}]".rjust(16) +
              f"{yld*100:>10.1f}%" + f"[{yld_lo*100:.1f},{yld_hi*100:.1f}]".rjust(16))

        rows_out.append((pretty(slug), split, cov, acc, yld))

    print("=" * 112)
    print("Coverage  = proportion of valid items receiving a non-tied six-vote decision")
    print("Accuracy  = proportion of DECIDED items that match ground truth (ties excluded)")
    print("EffYield  = coverage x accuracy: of 100 input pairs, how many emerge with a")
    print("            label that is both defined and correct")
    print()
    print("Accuracy alone can overstate a judge's usefulness: a judge may be accurate on")
    print("the items it decides while failing to decide most items. EffYield is the")
    print("relevant quantity when the judge is used to build a labeled dataset.")

    if rows_out:
        print()
        print("Ranked by effective yield (GPT split):")
        gpt_rows = sorted([r for r in rows_out if r[1] == "gpt"],
                          key=lambda r: -r[4])
        for name, split, cov, acc, yld in gpt_rows:
            print(f"  {name:<32}{yld*100:>6.1f}%   "
                  f"(coverage {cov*100:.1f}% x accuracy {acc*100:.1f}%)")


if __name__ == "__main__":
    main()
