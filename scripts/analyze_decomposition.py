#!/usr/bin/env python3
"""
analyze_decomposition.py
---------------------------
Splits observed instability into two parts using an EXACT identity (not an
approximation), replacing the earlier invalid subtraction
(swap_flip_rate - noise_floor).

THE IDENTITY
------------
For an item with P(pick original A | AB) = p and P(pick original A | BA) = q:

  D_cross  = probability a random AB trial and a random BA trial disagree
           = p(1-q) + (1-p)q

  D_within = average probability two trials of the SAME ordering disagree
           = [2p(1-p) + 2q(1-q)] / 2
           = p(1-p) + q(1-q)

  IDENTITY (exact, verified numerically to floating-point precision):
      D_cross = D_within + (p - q)^2

  So:  S = D_cross - D_within = (p - q)^2
       "S" is the SQUARED systematic order effect for that item -- the part
       of cross-order disagreement that is NOT explained by ordinary
       same-order randomness.

WITH 3 REPEATS PER ORDER
------------------------
Let x = number of AB trials (out of 3) choosing original A, y = same for BA.

  D_cross  = [x(3-y) + (3-x)y] / 9        (9 = 3x3 possible AB-BA pairings)
  D_within = [ x(3-x)/3 + y(3-y)/3 ] / 2  (3 = C(3,2) unordered same-order pairs)
  S        = D_cross - D_within           (estimates (p-q)^2)

S can be slightly negative for an individual pair due to sampling noise with
only 3 reps -- this is expected and should NOT be clipped to zero (clipping
biases the average upward). Average S unclipped across pairs, then bootstrap
the model-level mean.

Usage:
    python analyze_decomposition.py
    python analyze_decomposition.py --min-reps 3
"""

import argparse
import glob
import json
import re
from collections import defaultdict

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


def build_items(rows, min_reps=3):
    """pair_id -> dict with x (AB count of original-A), y (BA count),
    n1, n2 (actual valid rep counts). Only pairs with >= min_reps valid
    trials in EACH ordering are kept -- these form the 'Full6' analysis set."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    items = []
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= min_reps and len(o2) >= min_reps:
            # if a model has MORE than min_reps valid trials (rare), use exactly
            # the first min_reps for a consistent n across all pairs, so the
            # combinatorial formulas (which assume a fixed n) stay exact
            o1, o2 = o1[:min_reps], o2[:min_reps]
            items.append({"pair_id": pair_id, "o1": o1, "o2": o2,
                         "x": sum(o1), "y": sum(o2), "n": min_reps})
    return items


def decompose_item(item):
    """Returns (D_cross, D_within, S) for one item, using its x, y, n."""
    x, y, n = item["x"], item["y"], item["n"]
    n_pairs_cross = n * n
    n_pairs_within = n * (n - 1) / 2  # C(n,2)

    d_cross = (x * (n - y) + (n - x) * y) / n_pairs_cross
    if n_pairs_within > 0:
        d_ab = (x * (n - x)) / n_pairs_within
        d_ba = (y * (n - y)) / n_pairs_within
        d_within = (d_ab + d_ba) / 2
    else:
        d_within = 0.0  # n=1: no within-order pairs possible (shouldn't occur, min n=2)

    s = d_cross - d_within
    return d_cross, d_within, s


def cluster_bootstrap_mean(values, n_boot=2000, seed=0):
    """values: 1D array, one per item. Resample ITEMS with replacement."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    idx = np.arange(n)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = arr[rng.choice(idx, size=n, replace=True)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(means)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reps", type=int, default=3,
                    help="Repeats required per ordering. Use 3 to match the "
                         "'Full6' analysis set from the audit.")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 108)
    print("EXACT DECOMPOSITION: D_cross = D_within + S   (S estimates (p-q)^2)")
    print("=" * 108)
    print(f"{'Model':<32}{'Split':>7}{'nItems':>8}{'D_cross':>10}{'D_within':>10}"
          f"{'S (order)':>12}{'95% CI':>20}{'NoiseShare':>11}")
    print("-" * 108)

    for path in sorted(files, key=lambda p: (parse_filename(p)[0], parse_filename(p)[1])):
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_items(rows, min_reps=args.min_reps)
        if len(items) < 10:
            print(f"{pretty(slug):<32}{split:>7}  (only {len(items)} usable items, skipping)")
            continue

        d_cross_list, d_within_list, s_list = [], [], []
        for it in items:
            dc, dw, s = decompose_item(it)
            d_cross_list.append(dc); d_within_list.append(dw); s_list.append(s)

        mean_cross = float(np.mean(d_cross_list))
        mean_within = float(np.mean(d_within_list))
        mean_s = float(np.mean(s_list))  # EXACT sample mean -- satisfies the
                                          # identity to floating-point precision
        _, lo, hi = cluster_bootstrap_mean(s_list, args.n_boot)  # CI only

        # sanity: mean_cross should equal mean_within + mean_s (exact identity,
        # small float rounding aside)
        assert abs(mean_cross - (mean_within + mean_s)) < 1e-9, \
            "Identity broken -- this should never happen; check decompose_item"

        noise_share = mean_within / mean_cross if mean_cross > 0 else float("nan")
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        flag = "" if lo > 0 else "  (CI includes 0 -- no reliable systematic effect)"
        print(f"{pretty(slug):<32}{split:>7}{len(items):>8}{mean_cross:>10.4f}"
              f"{mean_within:>10.4f}{mean_s:>+12.4f}{ci_str:>20}{noise_share*100:>10.1f}%{flag}")

    print("=" * 108)
    print("D_cross    = observed disagreement rate between a random AB trial and")
    print("             a random BA trial (this is what the OLD 'swap flip rate'")
    print("             was trying to approximate with only 1 trial per order)")
    print("D_within   = disagreement rate between two trials of the SAME order")
    print("             (this is the noise floor, but computed properly using")
    print("             BOTH orders' repeats, not just one)")
    print("S          = D_cross - D_within, an EXACT (not approximate) estimate")
    print("             of the squared systematic order effect (p-q)^2 for this")
    print("             model's items, averaged. If the 95% CI excludes 0, the")
    print("             systematic order effect is real, not just sampling noise.")
    print("NoiseShare = D_within / D_cross: what fraction of total disagreement")
    print("             is explained by plain randomness rather than order.")
    print()
    print("This identity is EXACT: D_cross - D_within = (p-q)^2 always, for any")
    print("p, q. It replaces the earlier subtraction (swap_flip_rate - noise_floor),")
    print("which carried a spurious (1-2p) factor and was NOT a valid decomposition.")


if __name__ == "__main__":
    main()
