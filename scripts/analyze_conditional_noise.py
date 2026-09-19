#!/usr/bin/env python3
"""
analyze_conditional_noise.py
--------------------------------
For each judge, answers two direct questions:

  1. Given that a pair shows ORDER sensitivity (the majority verdict flips
     between AB and BA), what is the chance that pair is ALSO noisy
     (shows within-order self-disagreement)?
         P(noisy | order-sensitive)

  2. Given that a pair shows NO order sensitivity, what is the chance it
     is noisy anyway?
         P(noisy | NOT order-sensitive)

This is a more directly readable version of the phi-coefficient test used
elsewhere in this project (same underlying N/P labels, same contingency
table) -- instead of one abstract correlation number, it reports the two
conditional probabilities themselves, plus the difference between them.

  If the two probabilities are close (difference's CI includes 0): a pair
  being order-sensitive tells you nothing about whether it is also noisy --
  the two failure modes are independent in this model.

  If P(noisy | order-sensitive) is clearly HIGHER: order-sensitive pairs
  are disproportionately also the noisy ones -- the two failure modes
  overlap for this model.

  If P(noisy | order-sensitive) is clearly LOWER: order-sensitive pairs are
  disproportionately the CLEAN, non-noisy ones (consistent within each
  ordering, just reversed between orderings) -- the two failure modes
  tend to occur on separate pairs.

Uses the same N (within-order noise) and P (majority flip) labels as
analyze_pair_classes.py, computed via classify_item, so results are
directly consistent with the four-way classification and phi results
already reported.

Usage:
    python analyze_conditional_noise.py
"""

import glob
import json
import re
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci
from analyze_pair_classes import classify_item


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
    """pair_id -> (N, P): within-order-noise flag, majority-flip flag."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    items = []
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            x, y = sum(o1[:n_reps]), sum(o2[:n_reps])
            N, P = classify_item(x, y, n_reps)
            items.append({"pair_id": pair_id, "N": N, "P": P})
    return items


def cond_prob_noisy_given_flip(items):
    """P(N=1 | P=1). Returns None if no flipped pairs exist."""
    flipped = [it for it in items if it["P"] == 1]
    if not flipped:
        return None
    return float(np.mean([it["N"] for it in flipped]))


def cond_prob_noisy_given_stable(items):
    """P(N=1 | P=0). Returns None if no non-flipped pairs exist."""
    stable = [it for it in items if it["P"] == 0]
    if not stable:
        return None
    return float(np.mean([it["N"] for it in stable]))


def diff_stat(items):
    """P(N=1|P=1) - P(N=1|P=0). Returns 0.0 if either side is empty, so this
    is safe to use inside a bootstrap resample (which can occasionally drop
    one side entirely by chance)."""
    a = cond_prob_noisy_given_flip(items)
    b = cond_prob_noisy_given_stable(items)
    if a is None or b is None:
        return 0.0
    return a - b


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 118)
    print("GIVEN ORDER SENSITIVITY, HOW LIKELY IS RUN-TO-RUN (NOISE) SENSITIVITY TOO?")
    print("=" * 118)
    print(f"{'Model':<30}{'Split':>7}{'n':>6}{'nFlipped':>10}{'nStable':>9}"
          f"{'P(noisy|flip)':>16}{'P(noisy|stable)':>17}{'Diff':>9}{'95% CI (diff)':>20}")
    print("-" * 118)

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

        n_flipped = sum(1 for it in items if it["P"] == 1)
        n_stable = sum(1 for it in items if it["P"] == 0)

        p_given_flip = cond_prob_noisy_given_flip(items)
        p_given_stable = cond_prob_noisy_given_stable(items)

        if p_given_flip is None or p_given_stable is None:
            print(f"{pretty(slug):<30}{split:>7}{len(items):>6}{n_flipped:>10}{n_stable:>9}"
                  f"{'n/a':>16}{'n/a':>17}{'n/a':>9}{'(no flipped or no stable pairs)':>20}")
            continue

        point, lo, hi = bootstrap_ci(items, diff_stat, n_boot=2000, seed=0)

        flip_str = f"{p_given_flip*100:.1f}%"
        stable_str = f"{p_given_stable*100:.1f}%"
        diff_str = f"{point*100:+.1f}pp"
        ci_str = f"[{lo*100:+.1f}, {hi*100:+.1f}]"

        print(f"{pretty(slug):<30}{split:>7}{len(items):>6}{n_flipped:>10}{n_stable:>9}"
              f"{flip_str:>16}{stable_str:>17}{diff_str:>9}{ci_str:>20}")

    print("=" * 118)
    print("P(noisy|flip)    = of the pairs that showed ORDER sensitivity (majority flip),")
    print("                   what fraction ALSO showed within-order noise")
    print("P(noisy|stable)  = of the pairs that showed NO order sensitivity, what fraction")
    print("                   showed within-order noise anyway")
    print("Diff             = P(noisy|flip) - P(noisy|stable), with a pair-level bootstrap CI")
    print()
    print("If the CI excludes 0: order sensitivity and noise are NOT independent for this")
    print("model. A positive Diff means order-sensitive pairs are disproportionately ALSO")
    print("noisy (the two failure modes overlap). A negative Diff means order-sensitive")
    print("pairs tend to be clean/consistent within each ordering -- i.e. the flips are")
    print("'pure' position reversals, and noise shows up mainly on the non-flipping pairs.")
    print()
    print("If the CI includes 0: whether a pair is order-sensitive tells you nothing about")
    print("whether it is also noisy for this model -- the two are independent here.")


if __name__ == "__main__":
    main()
