#!/usr/bin/env python3
"""
analyze_repeat_sensitivity.py
---------------------------------
Almost every published position-bias study runs ONE AB call and ONE BA call
per item. You ran THREE of each. This script asks: how much would the
estimated flip rate (and model rankings) have bounced around if you'd only
run 1 repeat, like everyone else does -- and does 2 repeats fix it?

METHOD
------
For n_use in {1, 2, 3}: repeatedly (many random draws) subsample n_use of
the 3 available AB calls and n_use of the 3 available BA calls PER PAIR
(without replacement), compute the resulting flip rate over the whole
model, and record it. Do this many times to see the SPREAD of estimates
you'd get from an experiment that size.

n_use=1 simulates the standard one-AB/one-BA design. n_use=3 uses
everything (matches your main results). Comparing the spread at each level
answers: "how many repeats before the flip-rate estimate stops bouncing
around?"

ALSO: does the RANKING of models (by flip rate) change depending on which
single call happened to get used? For many draws at n_use=1, compute the
Spearman correlation between that draw's model ranking and the TRUE
(full n=3) ranking, across all models with data in a given split.

Usage:
    python analyze_repeat_sensitivity.py
    python analyze_repeat_sensitivity.py --n-draws 500
"""

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np

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


def build_item_calls(rows, n_reps=3):
    """pair_id -> (ab_calls, ba_calls), each a list of 0/1 (1 = picked original A),
    for pairs with a full n_reps valid calls in BOTH orderings."""
    by_item = defaultdict(lambda: {1: [], 2: []})
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)

    items = {}
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            items[pair_id] = (o1[:n_reps], o2[:n_reps])
    return items


def subsample_flip_rate(items, n_use, rng):
    """One draw: for each item, randomly pick n_use of the 3 AB calls and
    n_use of the 3 BA calls (without replacement), compute whether the
    resulting majority flips, and return the overall flip rate."""
    flips = 0
    total = 0
    for pair_id, (ab, ba) in items.items():
        ab_idx = rng.choice(len(ab), size=n_use, replace=False)
        ba_idx = rng.choice(len(ba), size=n_use, replace=False)
        x = sum(ab[i] for i in ab_idx)
        y = sum(ba[i] for i in ba_idx)
        _, P = classify_item(x, y, n_use)
        flips += P
        total += 1
    return flips / total if total else float("nan")


def true_flip_rate(items):
    """The full n=3 flip rate -- the best estimate available, used as the
    reference point for bias and for ranking comparisons."""
    flips = 0
    for pair_id, (ab, ba) in items.items():
        x, y = sum(ab), sum(ba)
        _, P = classify_item(x, y, len(ab))
        flips += P
    return flips / len(items) if items else float("nan")


def spearman_rho(x, y):
    def rank(vals):
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(len(vals), dtype=float)
        sorted_vals = np.array(vals)[order]
        i = 0
        cursor = 1
        while i < len(sorted_vals):
            j = i
            while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
                j += 1
            avg_rank = (cursor + (cursor + (j - i) - 1)) / 2.0
            ranks[order[i:j]] = avg_rank
            cursor += (j - i)
            i = j
        return ranks
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    by_split = defaultdict(dict)  # split -> {model_name: items}
    for path in files:
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        items = build_item_calls(rows)
        if len(items) >= 10:
            by_split[split][pretty(slug)] = items

    for split, models in by_split.items():
        print("=" * 100)
        print(f"HOW MUCH DOES THE FLIP-RATE ESTIMATE BOUNCE AROUND? -- split = {split}")
        print("=" * 100)
        print(f"{'Model':<30}{'TrueRate(n=3)':>15}{'n=1 mean':>10}{'n=1 SD':>9}"
              f"{'n=1 range':>14}{'n=2 SD':>9}{'n=3 SD':>9}")
        print("-" * 100)

        rng = np.random.default_rng(args.seed)
        model_names = sorted(models)
        true_rates = {}
        draws_by_n = {1: {}, 2: {}, 3: {}}

        for name in model_names:
            items = models[name]
            true_rate = true_flip_rate(items)
            true_rates[name] = true_rate

            rates_by_n = {}
            for n_use in (1, 2, 3):
                draws = np.array([subsample_flip_rate(items, n_use, rng)
                                  for _ in range(args.n_draws)])
                rates_by_n[n_use] = draws
            draws_by_n_for_model = rates_by_n
            draws_by_n[1][name] = rates_by_n[1]
            draws_by_n[2][name] = rates_by_n[2]
            draws_by_n[3][name] = rates_by_n[3]

            d1, d2, d3 = rates_by_n[1], rates_by_n[2], rates_by_n[3]
            print(f"{name:<30}{true_rate*100:>14.1f}%{np.mean(d1)*100:>9.1f}%"
                  f"{np.std(d1)*100:>8.1f}%"
                  f"[{np.min(d1)*100:.0f},{np.max(d1)*100:.0f}]%".rjust(14) +
                  f"{np.std(d2)*100:>8.1f}%{np.std(d3)*100:>8.1f}%")

        print()
        print("TrueRate(n=3) = best estimate available, using all 3 repeats each way")
        print("n=1 mean/SD   = average and spread of the estimate across many simulated")
        print("                'standard' single-AB/single-BA experiments")
        print("n=1 range     = the min-to-max spread seen across simulated single-swap runs")
        print("n=2 SD, n=3 SD = spread with 2 or 3 repeats -- watch this shrink")
        print()

        # ---- rank stability: does the MODEL RANKING change at n=1? ----
        if len(model_names) >= 4:
            true_ranking = [true_rates[m] for m in model_names]
            rank_corrs = []
            for draw_i in range(args.n_draws):
                draw_rates = [draws_by_n[1][m][draw_i] for m in model_names]
                rho = spearman_rho(true_ranking, draw_rates)
                if rho is not None:
                    rank_corrs.append(rho)
            if rank_corrs:
                print(f"MODEL RANKING STABILITY at n=1 (single-swap): comparing each")
                print(f"simulated single-swap ranking to the TRUE (n=3) ranking, across")
                print(f"{len(rank_corrs)} draws:")
                print(f"  mean Spearman rho = {np.mean(rank_corrs):+.3f}   "
                      f"min = {np.min(rank_corrs):+.3f}   "
                      f"(1.0 = ranking always matches truth)")
        print()

    print("=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print("If n=1 SD is large and n=3 SD is much smaller, a standard single-swap")
    print("study's flip-rate estimate could easily have landed far from the true")
    print("value, purely from which single call happened to be sampled.")
    print()
    print("If mean Spearman rho for model ranking is well below 1.0, published")
    print("comparisons BETWEEN models using single-swap designs may not reliably")
    print("reflect the true ranking -- a real methods concern for the whole field.")


if __name__ == "__main__":
    main()
