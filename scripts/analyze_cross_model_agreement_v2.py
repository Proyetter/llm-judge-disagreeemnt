#!/usr/bin/env python3
"""
analyze_cross_model_agreement_v2.py
--------------------------------------
Do different judge models get tripped up by the SAME pairs, or by largely
DIFFERENT pairs? This is a redo of an earlier analysis in this project, now
using the majority-flip label (P) from the properly-powered 3xAB/3xBA
design instead of the old single-swap version.

Uses P (does the majority verdict flip between AB and BA), computed the
same way as in analyze_pair_classes.py, for every model that has valid
data for a pair. Done SEPARATELY per split (gpt/claude), since the two
splits contain different sets of pairs.

For each split:
  - distribution of "what fraction of available models flip this pair"
  - universal-flip / universal-stable / mixed-outcome counts
  - pairwise Cohen's kappa between every pair of models (bootstrap CI on
    the across-model mean)

If pairs cluster near 0% or 100% (most models agree), flipping is mostly
an ITEM property. If the distribution spreads out with low kappa, flipping
is mostly a MODEL property -- each judge trips on its own set of pairs.

Usage:
    python analyze_cross_model_agreement_v2.py
"""

import glob
import json
import re
from collections import defaultdict
from itertools import combinations

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


def build_flip_labels(rows, n_reps=3):
    """pair_id -> P (majority-flip label), for pairs with full valid data."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    labels = {}
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            x, y = sum(o1[:n_reps]), sum(o2[:n_reps])
            _, P = classify_item(x, y, n_reps)
            labels[pair_id] = P
    return labels


def cohen_kappa(labels_a, labels_b):
    """Standard two-rater Cohen's kappa for binary labels on shared items."""
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa = sum(labels_a) / n
    pb = sum(labels_b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def analyze_split(files_for_split, split_name):
    per_model_labels = {}
    for path in files_for_split:
        slug, split = parse_filename(path)
        if slug is None or split != split_name:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        per_model_labels[pretty(slug)] = build_flip_labels(rows)

    if len(per_model_labels) < 2:
        print(f"  (fewer than 2 models with data for split={split_name}, skipping)")
        return

    model_names = sorted(per_model_labels)
    pair_votes = defaultdict(list)
    for name in model_names:
        for pid, p in per_model_labels[name].items():
            pair_votes[pid].append(p)

    multi = {pid: v for pid, v in pair_votes.items() if len(v) >= 2}
    if not multi:
        print(f"  (no pairs shared across 2+ models for split={split_name})")
        return

    fracs = np.array([sum(v) / len(v) for v in multi.values()])
    universal_flip = int(np.sum(fracs == 1.0))
    universal_stable = int(np.sum(fracs == 0.0))
    mixed = len(multi) - universal_flip - universal_stable

    print(f"  Models: {', '.join(model_names)}")
    print(f"  Pairs with data from 2+ models: {len(multi)}")
    print(f"  Universal-flip (ALL models flip)  : {universal_flip} ({100*universal_flip/len(multi):.1f}%)")
    print(f"  Universal-stable (NO model flips) : {universal_stable} ({100*universal_stable/len(multi):.1f}%)")
    print(f"  Mixed outcome                     : {mixed} ({100*mixed/len(multi):.1f}%)")
    print()
    print("  Distribution of 'fraction of models that flip this pair':")
    bins = [(0.0, 0.0, "exactly 0%"), (0.0001, 0.33, "1-33%"),
            (0.3301, 0.67, "34-67%"), (0.6701, 0.9999, "68-99%"),
            (1.0, 1.0, "exactly 100%")]
    for lo, hi, label in bins:
        count = int(np.sum((fracs >= lo) & (fracs <= hi)))
        print(f"    {label:<12}: {count:>4} pairs ({100*count/len(fracs):.1f}%)")

    print()
    print(f"  {'Model pair':<62}{'n shared':>10}{'kappa':>9}")
    print("  " + "-" * 83)
    kappas = []
    for a, b in combinations(model_names, 2):
        shared = sorted(set(per_model_labels[a]) & set(per_model_labels[b]))
        if len(shared) < 10:
            continue
        la = [per_model_labels[a][pid] for pid in shared]
        lb = [per_model_labels[b][pid] for pid in shared]
        k = cohen_kappa(la, lb)
        if k is not None:
            kappas.append(k)
            print(f"  {a+' vs '+b:<62}{len(shared):>10}{k:>+9.3f}")

    if kappas:
        items_for_boot = list(zip(*[[k] for k in kappas]))  # trivial per-value resample
        arr = np.array(kappas)

        def mean_stat(sample):
            return float(np.mean(sample))

        point, lo, hi = bootstrap_ci([[k] for k in kappas],
                                     lambda s: mean_stat([x[0] for x in s]),
                                     n_boot=2000, seed=0)
        print()
        print(f"  Mean pairwise kappa: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"  (n={len(kappas)} model pairs)")
        print(f"  Range: {min(kappas):+.3f} to {max(kappas):+.3f}")


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    for split_name in ["gpt", "claude"]:
        print("=" * 100)
        print(f"CROSS-MODEL FLIP AGREEMENT -- split = {split_name}")
        print("=" * 100)
        analyze_split(files, split_name)
        print()

    print("=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print("If most pairs sit near 0% or 100% (few models disagree on THIS item),")
    print("and kappa is strongly positive, flipping is largely an ITEM property --")
    print("some comparisons are just inherently order-sensitive for any judge.")
    print()
    print("If the distribution spreads out and kappa sits near 0, flipping is")
    print("largely a MODEL property -- each judge trips on its own set of pairs,")
    print("largely independent of which pairs trip up other judges.")


if __name__ == "__main__":
    main()
