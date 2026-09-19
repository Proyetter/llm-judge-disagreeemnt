#!/usr/bin/env python3
"""
analyze_kappa_bootstrap.py
-----------------------------
Corrected confidence interval for mean pairwise Cohen's kappa across judges.

WHY THIS SCRIPT EXISTS
----------------------
analyze_cross_model_agreement_v2.py computes the 21 pairwise kappa values
correctly, but its confidence interval for the MEAN is wrong. It resamples
the 21 kappa values themselves:

    bootstrap_ci([[k] for k in kappas], ...)

That measures how much the mean would move given a different set of judge
pairings drawn from the same seven judges. It does NOT measure sampling
variability of the response pairs, which is the dominant source of
uncertainty and what the Methods section claims is being measured. The 21
values are also mutually dependent -- every one is computed from the same
items -- so treating them as 21 independent draws is invalid regardless.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
The judge panel is held FIXED. Variability is estimated across resampled
response pairs, using paired (matched) resampling:

  1. Build an item x judge matrix of majority-reversal labels, with
     missing entries where a judge has no eligible data for that item.
  2. Draw row indices with replacement. The SAME indices are used for
     every judge, so matched records stay together and the missing-data
     pattern is preserved.
  3. Recompute all 21 pairwise kappas within the resample. For a given
     judge pair, use only sampled rows where BOTH judges have a label.
  4. Average the 21 values with equal weight -- the same definition used
     for the point estimate.
  5. Repeat, then take the 2.5th and 97.5th percentiles.

Two details this implementation is careful about:

  MULTIPLICITY. Sampled row indices are kept as a list, not a set. An item
  drawn three times contributes three times to every pairwise computation.
  Deduplicating would break the bootstrap.

  UNDEFINED KAPPA. Cohen's kappa is undefined when a rater shows no
  variation within the resample (expected agreement reaches 1.0 and the
  denominator vanishes). Rather than substituting zero or averaging over a
  shrinking set of judge pairs -- either of which would silently bias the
  interval -- any resample containing an undefined kappa is DISCARDED
  whole, and the discard count is reported. This keeps the set of 21 pairs
  identical across every retained resample.

The point estimate is computed on the full data, not as the bootstrap mean.

Usage:
    python analyze_kappa_bootstrap.py
    python analyze_kappa_bootstrap.py --n-boot 5000 --seed 0
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from itertools import combinations

import numpy as np


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


def reversal_labels(rows, n_reps=3):
    """pair_id -> majority-reversal label P, for items with n_reps valid
    judgments in BOTH orderings. Same eligibility rule and same P definition
    as analyze_pair_classes.py."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    labels = {}
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            x, y = sum(o1[:n_reps]), sum(o2[:n_reps])
            maj_ab = "A" if x >= n_reps / 2 else "B"
            maj_ba = "A" if y >= n_reps / 2 else "B"
            labels[pair_id] = int(maj_ab != maj_ba)
    return labels


def cohen_kappa_arrays(la, lb):
    """Cohen's kappa for two aligned binary label arrays.
    Returns None when undefined (expected agreement reaches 1, i.e. at least
    one rater shows no variation and both agree on that constant value)."""
    n = len(la)
    if n == 0:
        return None
    po = float(np.mean(la == lb))
    pa = float(np.mean(la))
    pb = float(np.mean(lb))
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if pe >= 1.0 - 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def mean_pairwise_kappa(matrix, model_names, row_idx, min_shared=10):
    """Mean of all pairwise kappas over the rows given by row_idx.

    matrix: dict model -> float array aligned to a fixed item ordering,
            np.nan where that model has no eligible label for the item.
    row_idx: array of row positions (WITH multiplicity from resampling).

    Returns (mean_kappa, n_pairs_used, undefined_flag). undefined_flag is
    True if any judge pair yielded an undefined kappa or had too few shared
    rows, in which case mean_kappa is None."""
    kappas = []
    for a, b in combinations(model_names, 2):
        va = matrix[a][row_idx]
        vb = matrix[b][row_idx]
        both = ~np.isnan(va) & ~np.isnan(vb)
        if both.sum() < min_shared:
            return None, 0, True
        k = cohen_kappa_arrays(va[both].astype(int), vb[both].astype(int))
        if k is None:
            return None, 0, True
        kappas.append(k)
    return float(np.mean(kappas)), len(kappas), False


def analyze_split(files_for_split, split_name, n_boot, seed, min_judges):
    per_model_labels = {}
    for path in files_for_split:
        slug, split = parse_filename(path)
        if slug is None or split != split_name:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        per_model_labels[pretty(slug)] = reversal_labels(rows)

    model_names = sorted(per_model_labels)
    if len(model_names) < 2:
        print(f"  (fewer than 2 judges with data for split={split_name}; skipping)")
        return

    # fixed item ordering: items with at least min_judges eligible judges
    item_counts = defaultdict(int)
    for name in model_names:
        for pid in per_model_labels[name]:
            item_counts[pid] += 1
    items = sorted(pid for pid, c in item_counts.items() if c >= min_judges)
    n_items = len(items)
    if n_items < 20:
        print(f"  (only {n_items} items with >={min_judges} eligible judges; skipping)")
        return

    # item x judge matrix, np.nan for missing
    matrix = {}
    for name in model_names:
        lab = per_model_labels[name]
        matrix[name] = np.array([lab.get(pid, np.nan) for pid in items], dtype=float)

    n_pairs_total = len(list(combinations(model_names, 2)))

    # ---- point estimate on the FULL data ----
    all_rows = np.arange(n_items)
    point, n_pairs_used, undef = mean_pairwise_kappa(matrix, model_names, all_rows)
    if undef:
        print(f"  (mean kappa undefined on the full data for split={split_name}; skipping)")
        return

    # ---- paired item bootstrap ----
    rng = np.random.default_rng(seed)
    boot_means = []
    n_discarded = 0
    for _ in range(n_boot):
        # sample WITH replacement; keep as an array so multiplicity is retained
        idx = rng.integers(0, n_items, size=n_items)
        m, _, undef_b = mean_pairwise_kappa(matrix, model_names, idx)
        if undef_b:
            n_discarded += 1
            continue
        boot_means.append(m)

    boot_means = np.array(boot_means)
    if len(boot_means) < 100:
        print(f"  (only {len(boot_means)} usable resamples; interval unreliable)")
        return

    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    # per-judge eligible counts, for the reporting note
    elig_counts = {name: int(np.sum(~np.isnan(matrix[name]))) for name in model_names}

    print(f"  Judges: {len(model_names)}   judge pairs: {n_pairs_total}")
    print(f"  Items used: {n_items} (>= {min_judges} eligible judges)")
    print(f"  Mean pairwise kappa: {point:+.4f}")
    print(f"  95% CI (paired item bootstrap): [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Resamples: {n_boot} requested, {len(boot_means)} retained, "
          f"{n_discarded} discarded for an undefined kappa "
          f"({100*n_discarded/n_boot:.2f}%)")
    print(f"  Bootstrap SD: {np.std(boot_means):.4f}   seed: {seed}")
    print()
    print(f"  Eligible items per judge:")
    for name in model_names:
        print(f"    {name:<34}{elig_counts[name]:>5} / {n_items}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="Bootstrap resamples (default 5000).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed, recorded in the output for reproducibility.")
    ap.add_argument("--min-judges", type=int, default=2,
                    help="Minimum eligible judges for an item to be included. "
                         "Set to 7 to restrict to full-panel items only.")
    args = ap.parse_args()

    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 92)
    print("MEAN PAIRWISE COHEN'S KAPPA -- corrected paired item bootstrap")
    print("=" * 92)
    print("The judge panel is held fixed; uncertainty is estimated across resampled")
    print("response pairs. The same sampled row indices are used for every judge, so")
    print("matched records stay together and the missing-data pattern is preserved.")
    print("Resample multiplicity is retained. Any resample containing an undefined")
    print("kappa is discarded whole rather than zero-filled, and the discard rate is")
    print("reported below.")
    print("=" * 92)

    for split_name in ["gpt", "claude"]:
        print(f"\n--- split = {split_name} ---")
        analyze_split(files, split_name, args.n_boot, args.seed, args.min_judges)

    print()
    print("=" * 92)
    print("Compare these intervals with those from analyze_cross_model_agreement_v2.py.")
    print("That script's per-pair kappa values are correct and unaffected; only its")
    print("interval for the MEAN was computed by resampling the 21 summary values,")
    print("which does not measure item-level sampling variability. Use the interval")
    print("reported here for any claim about the mean.")


if __name__ == "__main__":
    main()
