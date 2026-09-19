#!/usr/bin/env python3
"""
analyze_order_effect.py
--------------------------
Estimates position bias CORRECTLY from the factorial experiment
(judge_judgebench_factorial.py), replacing the invalid subtraction
   swap_flip_rate - same_order_noise_rate.

THE ESTIMATOR
-------------
For each item i, using its repeated trials:

    p_i = P(judge picks original response A | resp_a shown FIRST)
    q_i = P(judge picks original response A | resp_a shown SECOND)

    delta_i = p_i - q_i

delta_i is the signed order effect for that item:
    delta_i > 0  -> the judge favoured whichever response was shown FIRST
                    (primacy)
    delta_i < 0  -> the judge favoured whichever was shown SECOND (recency)
    delta_i = 0  -> ordering made no difference for this item

We report two population quantities, and they answer different questions:

  MEAN SIGNED delta   -> is there a net directional bias across the corpus?
                         Primacy on some items can cancel recency on others,
                         so this can be near zero even when every item is
                         strongly order-sensitive.

  MEAN ABSOLUTE delta -> how order-sensitive is the judge, ignoring direction?
                         This is the magnitude. Note it is biased UPWARD by
                         sampling noise (with few reps, |p_i - q_i| is rarely
                         exactly 0 even under the null), so we also run a
                         permutation test to establish what value it would
                         take if ordering genuinely did not matter.

Uncertainty:
  - Cluster bootstrap over ITEMS (items are the independent unit, not trials).
  - Permutation test: under H0 the order label is exchangeable within an item,
    so we shuffle order labels within each item and rebuild the null.

Also prints, for comparison, what the OLD invalid subtraction would have
reported on this same data, so the size of the correction is visible.

Usage:
    python analyze_order_effect.py
    python analyze_order_effect.py --n-boot 5000
"""

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np


def model_slug(path):
    m = re.search(r"factorial_judgebench__(.+)__(gpt|claude|both)\.jsonl", path)
    return m.group(1) if m else None


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


def build_items(rows, min_reps_per_order=2):
    """item -> dict with the binary outcomes for each ordering.

    outcome = 1 if the judge picked ORIGINAL response A on that trial.
    Items lacking min_reps_per_order valid trials in EITHER ordering are
    dropped, since p or q would be unestimable."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    items = []
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= min_reps_per_order and len(o2) >= min_reps_per_order:
            items.append({
                "pair_id": pair_id,
                "o1": np.array(o1, dtype=float),
                "o2": np.array(o2, dtype=float),
                "p": float(np.mean(o1)),
                "q": float(np.mean(o2)),
            })
    return items


def deltas(items):
    return np.array([it["p"] - it["q"] for it in items])


def cluster_bootstrap(items, stat_fn, n_boot=2000, seed=0):
    """Resample ITEMS with replacement; recompute the statistic each time."""
    rng = np.random.default_rng(seed)
    n = len(items)
    vals = np.empty(n_boot)
    idx_pool = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        vals[b] = stat_fn([items[i] for i in idx])
    return vals


def permutation_null(items, stat_fn, n_perm=2000, seed=0):
    """Under H0 (ordering irrelevant) all trials for an item are exchangeable.
    Shuffle order labels within each item, preserving the per-order counts."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm)
    for b in range(n_perm):
        permuted = []
        for it in items:
            pooled = np.concatenate([it["o1"], it["o2"]])
            rng.shuffle(pooled)
            n1 = len(it["o1"])
            new_o1, new_o2 = pooled[:n1], pooled[n1:]
            permuted.append({
                "pair_id": it["pair_id"],
                "o1": new_o1, "o2": new_o2,
                "p": float(np.mean(new_o1)), "q": float(np.mean(new_o2)),
            })
        vals[b] = stat_fn(permuted)
    return vals


def naive_old_estimate(rows):
    """Reproduce what the OLD (invalid) subtraction would have said, using
    rep_idx 0 of each ordering as the 'swap' trial pair, and the order-1
    repeats as the 'noise floor'. For comparison only."""
    by_item = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_item[r["pair_id"]][r["order"]][r["rep_idx"]] = r["picked_original"]

    flips = complete = 0
    noise_disagree = noise_total = 0
    for pair_id, orders in by_item.items():
        o1, o2 = orders.get(1, {}), orders.get(2, {})
        if 0 in o1 and 0 in o2:
            complete += 1
            if o1[0] != o2[0]:
                flips += 1
        reps = list(o1.values())
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                noise_total += 1
                if reps[i] != reps[j]:
                    noise_disagree += 1

    swap_rate = flips / complete if complete else float("nan")
    noise_rate = noise_disagree / noise_total if noise_total else float("nan")
    return swap_rate, noise_rate, swap_rate - noise_rate


def direct_same_order_noise(items):
    """The LITERAL noise floor: pairwise disagreement rate among repeats of
    the SAME ordering, computed separately for order 1 and order 2 and then
    pooled. This is a direct measurement, unlike NullAbs (which is inferred
    via permutation of averaged trials).

    NOTE: this is NOT expected to numerically equal NullAbs -- they operate
    on different scales (this is a raw pairwise single-trial disagreement
    rate; NullAbs is the expected gap between two averages of n_reps trials,
    which is smaller by construction due to averaging). The valid check is
    that the two move TOGETHER across models (higher DirectNoise should
    accompany higher NullAbs), not that they match in absolute value.

    Returns (pooled_rate, order1_rate, order2_rate)."""
    from itertools import combinations

    def pairwise_disagreement(trials_list):
        total = disagree = 0
        for trials in trials_list:
            for a, b in combinations(trials, 2):
                total += 1
                if a != b:
                    disagree += 1
        return disagree, total

    o1_lists = [it["o1"] for it in items]
    o2_lists = [it["o2"] for it in items]

    d1, t1 = pairwise_disagreement(o1_lists)
    d2, t2 = pairwise_disagreement(o2_lists)

    order1_rate = d1 / t1 if t1 else None
    order2_rate = d2 / t2 if t2 else None
    pooled_rate = (d1 + d2) / (t1 + t2) if (t1 + t2) else None
    return pooled_rate, order1_rate, order2_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--min-reps", type=int, default=2,
                    help="Minimum valid trials required in EACH ordering.")
    args = ap.parse_args()

    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        print("Run judge_judgebench_factorial.py first.")
        return

    mean_signed = lambda its: float(np.mean([it["p"] - it["q"] for it in its]))
    mean_abs = lambda its: float(np.mean([abs(it["p"] - it["q"]) for it in its]))

    print("=" * 104)
    print("ORDER EFFECT (valid estimator)")
    print("delta_i = P(pick A | A first) - P(pick A | A second), per item")
    print("=" * 104)
    print(f"{'Model':<30}{'nItems':>8}{'MeanSigned':>13}{'95% CI':>20}"
          f"{'MeanAbs':>10}{'NullAbs':>10}{'DirectNoise':>12}{'p':>8}")
    print("-" * 104)

    detail = []
    for path in files:
        slug = model_slug(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_items(rows, min_reps_per_order=args.min_reps)
        if len(items) < 5:
            print(f"{pretty(slug):<30}  (only {len(items)} usable items, skipping)")
            continue

        obs_signed = mean_signed(items)
        obs_abs = mean_abs(items)

        boot_signed = cluster_bootstrap(items, mean_signed, args.n_boot)
        lo, hi = np.percentile(boot_signed, [2.5, 97.5])

        null_abs = permutation_null(items, mean_abs, args.n_perm)
        null_abs_mean = float(np.mean(null_abs))
        # one-sided: is observed order-sensitivity larger than chance?
        p_val = float(np.mean(null_abs >= obs_abs))

        direct_noise, direct_o1, direct_o2 = direct_same_order_noise(items)

        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        direct_str = f"{direct_noise:.3f}" if direct_noise is not None else "n/a"
        print(f"{pretty(slug):<30}{len(items):>8}{obs_signed:>+13.3f}{ci_str:>20}"
              f"{obs_abs:>10.3f}{null_abs_mean:>10.3f}{direct_str:>12}{p_str:>8}")

        detail.append((slug, rows, items, obs_signed, obs_abs, null_abs_mean,
                       direct_noise, direct_o1, direct_o2))

    print("=" * 104)
    print("MeanSigned  net direction: >0 favours the FIRST-shown response (primacy),")
    print("            <0 favours the SECOND (recency). Near 0 can also mean primacy")
    print("            and recency items cancel out -- read it with MeanAbs.")
    print("95% CI      cluster bootstrap over items. Excludes 0 => a real net")
    print("            directional bias.")
    print("MeanAbs     magnitude of order sensitivity ignoring direction.")
    print("NullAbs     what MeanAbs would be if ordering truly did not matter")
    print("            (permutation null, INFERRED). MeanAbs must clearly exceed")
    print("            this to mean anything -- with few reps, NullAbs is not 0.")
    print("DirectNoise the LITERAL same-order noise floor: pairwise disagreement")
    print("            among repeats of the SAME ordering (order 1's repeats vs")
    print("            each other, and order 2's repeats vs each other, pooled).")
    print("            This is MEASURED directly, not inferred like NullAbs.")
    print("            NOTE: DirectNoise and NullAbs are on DIFFERENT SCALES by")
    print("            construction -- DirectNoise is a raw single-trial-pair")
    print("            disagreement rate, while NullAbs is the expected gap")
    print("            between two 3-trial AVERAGES (which shrinks from")
    print("            averaging). They will NOT numerically match, and should")
    print("            not be expected to. The valid check is that they move")
    print("            TOGETHER across models -- a model with higher DirectNoise")
    print("            should also show higher NullAbs. Large disagreements in")
    print("            that ranking would be worth a second look.")
    print("p           permutation test, one-sided: P(null MeanAbs >= observed).")

    # ---- comparison against the old, invalid subtraction ----
    print()
    print("=" * 104)
    print("COMPARISON: what the OLD (invalid) subtraction would have reported")
    print("=" * 104)
    print(f"{'Model':<30}{'SwapFlip':>10}{'NoiseFloor':>12}{'OldSubtract':>13}"
          f"{'ValidMeanAbs':>14}{'Understated by':>16}")
    print("-" * 104)
    for slug, rows, items, obs_signed, obs_abs, null_abs_mean, direct_noise, direct_o1, direct_o2 in detail:
        swap, noise, old = naive_old_estimate(rows)
        gap = obs_abs - old
        print(f"{pretty(slug):<30}{swap*100:>9.1f}%{noise*100:>11.1f}%"
              f"{old*100:>12.1f}%{obs_abs*100:>13.1f}%{gap*100:>+15.1f}pp")
    print("=" * 104)
    print("The old subtraction carries a spurious (1-2p) factor that shrinks the")
    print("estimate toward zero as the judge gets noisier. The gap in the last")
    print("column is how much position bias that formula was hiding.")
    print()
    print("NOTE: OldSubtract and ValidMeanAbs are not the same quantity, so this")
    print("is an illustration of scale, not a strict numerical correction. Report")
    print("the valid estimator; do not publish the subtraction.")


if __name__ == "__main__":
    main()
