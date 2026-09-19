#!/usr/bin/env python3
"""
analyze_pair_classes.py
--------------------------
Answers the question the population-level NoiseShare (from
analyze_decomposition.py) CANNOT answer: are the pairs that flip under
reordering the SAME pairs that are noisy under repetition, or are these
largely different pairs?

NoiseShare = mean(D_within) / mean(D_cross) is an aggregate RATIO. It could
report "40% noise" in two totally different worlds:
  (a) every pair is 40% attributable to noise, uniformly, or
  (b) some pairs are pure noise and others are pure position-bias, averaging
      out to 40%.
Only item-level analysis tells you which world you're in.

FOUR-WAY CLASSIFICATION (per pair, per model)
----------------------------------------------
Let x = count of AB trials choosing original A (of 3), y = same for BA.

Within-order instability (not unanimous within an order):
    N = 1 if x in {1,2} OR y in {1,2}, else 0

Majority-vote flip (does the typical answer change with order):
    M_AB = A if x>=2 else B
    M_BA = A if y>=2 else B
    P = 1 if M_AB != M_BA, else 0

Four classes:
    Stable       : N=0, P=0  (fully consistent, in every sense)
    Noise-only   : N=1, P=0  (wobbles within an order, but majority holds)
    Position-only: N=0, P=1  (each order internally unanimous, but the
                              unanimous answer FLIPS between orders --
                              the cleanest single-item evidence of a real
                              position effect)
    Tangled      : N=1, P=1  (both noisy AND the majority flips -- cannot
                              cleanly attribute)

CONTINUOUS VERSION (rho) -- READ THE CAVEAT BELOW BEFORE TRUSTING THIS
------------------------------------------------------------------------
For each pair, also compute:
    D_within_i : this item's own within-order disagreement rate
    S_i        : this item's own systematic-order estimate (from the exact
                 decomposition identity)

IMPORTANT CAVEAT: D_within_i and S_i are NOT independently measured -- by
the exact identity, D_cross_i = D_within_i + S_i for every item. They are
two pieces of one FIXED TOTAL. If D_cross_i happens to be large for some
item, that total must be split between D_within_i and S_i, which can
produce a spurious NEGATIVE correlation between them even when there is
NO real relationship between "how noisy this item is" and "how much
position bias it has." This was confirmed by simulation: generating x and
y with zero true relationship between item-level noise and item-level
order-effect still produced rho around -0.6, purely from this mechanical
coupling.

Because of this, **phi (on the binary N/P classification) is the primary,
trustworthy item-level statistic in this script.** rho is reported as a
SECONDARY, exploratory number, and any nonzero value should be interpreted
with real caution -- it may reflect the shared-total artifact rather than
a genuine relationship. Do not present rho in the paper as equivalent
evidence to phi without this caveat attached.
"""

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


def build_items(rows, n_reps=3):
    by_item = defaultdict(lambda: {1: [], 2: []})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]].append(1 if r["picked_original"] == "A" else 0)
        gt_of[r["pair_id"]] = r["gt"]  # 'A' or 'B', same for every row of a pair

    items = []
    for pair_id, orders in by_item.items():
        o1, o2 = orders[1], orders[2]
        if len(o1) >= n_reps and len(o2) >= n_reps:
            o1, o2 = o1[:n_reps], o2[:n_reps]
            items.append({"pair_id": pair_id, "x": sum(o1), "y": sum(o2), "n": n_reps,
                         "gt": gt_of[pair_id]})
    return items


def decompose_item(x, y, n):
    """Same identity as analyze_decomposition.py: D_cross = D_within + S."""
    n_pairs_cross = n * n
    n_pairs_within = n * (n - 1) / 2
    d_cross = (x * (n - y) + (n - x) * y) / n_pairs_cross
    if n_pairs_within > 0:
        d_ab = (x * (n - x)) / n_pairs_within
        d_ba = (y * (n - y)) / n_pairs_within
        d_within = (d_ab + d_ba) / 2
    else:
        d_within = 0.0
    return d_cross, d_within, d_cross - d_within


def classify_item(x, y, n):
    """Returns (N, P): within-order-instability flag, majority-flip flag."""
    not_unanimous_ab = 0 < x < n
    not_unanimous_ba = 0 < y < n
    N = int(not_unanimous_ab or not_unanimous_ba)

    majority_ab = "A" if x >= (n / 2) else "B"
    majority_ba = "A" if y >= (n / 2) else "B"
    P = int(majority_ab != majority_ba)
    return N, P


def six_vote_decision(x, y, n, gt):
    """Combine ALL 2n calls (both orderings) into one majority decision.
    v = total votes for original A, out of 2n. Returns (decision, is_tie,
    is_correct). A tie (v == n, i.e. exactly half) is reported separately
    and excluded from the accuracy denominator -- it is not a wrong answer,
    it is a genuine "no majority" case."""
    v = x + y
    half = n  # since total votes = 2n, half = n
    if v == half:
        return None, True, None
    decision = "A" if v > half else "B"
    return decision, False, (decision == gt)


def phi_coefficient(a, b, c, d):
    num = (a * d) - (b * c)
    den = np.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return None if den == 0 else num / den


def spearman_rho(x, y):
    """Spearman rank correlation, implemented directly (average-rank ties)."""
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

    rx = rank(np.asarray(x))
    ry = rank(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_stat(items_data, stat_fn, n_boot=2000, seed=0):
    """items_data: list of per-item dicts. Resample ITEMS with replacement."""
    rng = np.random.default_rng(seed)
    n = len(items_data)
    idx = np.arange(n)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        sample = [items_data[i] for i in rng.choice(idx, size=n, replace=True)]
        vals[b] = stat_fn(sample)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 112)
    print("FOUR-WAY PAIR CLASSIFICATION + ITEM-LEVEL NOISE/FLIP OVERLAP")
    print("=" * 112)
    print(f"{'Model':<32}{'Split':>7}{'n':>6}{'Stable':>9}{'NoiseOnly':>11}"
          f"{'PosOnly':>9}{'Tangled':>9}{'phi':>8}{'phi CI':>16}{'rho':>8}{'rho CI':>16}")
    print("-" * 112)

    accuracy_results = []  # collected here, printed as a second table after the loop

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

        per_item = []
        for it in items:
            x, y, n, gt = it["x"], it["y"], it["n"], it["gt"]
            N, P = classify_item(x, y, n)
            d_cross, d_within, s = decompose_item(x, y, n)
            decision, is_tie, is_correct = six_vote_decision(x, y, n, gt)
            per_item.append({"pair_id": it["pair_id"], "N": N, "P": P,
                             "d_within": d_within, "s": s,
                             "is_tie": is_tie, "is_correct": is_correct})

        n_total = len(per_item)
        stable = sum(1 for d in per_item if d["N"] == 0 and d["P"] == 0)
        noise_only = sum(1 for d in per_item if d["N"] == 1 and d["P"] == 0)
        pos_only = sum(1 for d in per_item if d["N"] == 0 and d["P"] == 1)
        tangled = sum(1 for d in per_item if d["N"] == 1 and d["P"] == 1)

        # phi coefficient on the binary (N, P) table
        a = sum(1 for d in per_item if d["N"] == 1 and d["P"] == 1)  # noisy & flip
        b = sum(1 for d in per_item if d["N"] == 0 and d["P"] == 1)  # stable-noise & flip
        c = sum(1 for d in per_item if d["N"] == 1 and d["P"] == 0)  # noisy & no flip
        d_ = sum(1 for d in per_item if d["N"] == 0 and d["P"] == 0)  # neither
        phi = phi_coefficient(a, b, c, d_)

        # continuous Spearman correlation between item-level noise and item-level S
        d_within_vals = [d["d_within"] for d in per_item]
        s_vals = [d["s"] for d in per_item]
        rho = spearman_rho(d_within_vals, s_vals)

        def phi_stat(sample):
            aa = sum(1 for d in sample if d["N"] == 1 and d["P"] == 1)
            bb = sum(1 for d in sample if d["N"] == 0 and d["P"] == 1)
            cc = sum(1 for d in sample if d["N"] == 1 and d["P"] == 0)
            dd = sum(1 for d in sample if d["N"] == 0 and d["P"] == 0)
            v = phi_coefficient(aa, bb, cc, dd)
            return v if v is not None else 0.0

        def rho_stat(sample):
            dv = [x["d_within"] for x in sample]
            sv = [x["s"] for x in sample]
            v = spearman_rho(dv, sv)
            return v if v is not None else 0.0

        phi_lo, phi_hi = bootstrap_stat(per_item, phi_stat, 1500, seed=0)
        rho_lo, rho_hi = bootstrap_stat(per_item, rho_stat, 1500, seed=1)

        phi_str = f"{phi:+.3f}" if phi is not None else "n/a"
        phi_ci = f"[{phi_lo:+.2f}, {phi_hi:+.2f}]"
        rho_str = f"{rho:+.3f}" if rho is not None else "n/a"
        rho_ci = f"[{rho_lo:+.2f}, {rho_hi:+.2f}]"

        print(f"{pretty(slug):<32}{split:>7}{n_total:>6}"
              f"{100*stable/n_total:>8.1f}%{100*noise_only/n_total:>10.1f}%"
              f"{100*pos_only/n_total:>8.1f}%{100*tangled/n_total:>8.1f}%"
              f"{phi_str:>8}{phi_ci:>16}{rho_str:>8}{rho_ci:>16}")

        # ---- accuracy per class, using the six-vote (all 2n calls) decision ----
        def class_of(d):
            if d["N"] == 0 and d["P"] == 0:
                return "Stable"
            if d["N"] == 1 and d["P"] == 0:
                return "NoiseOnly"
            if d["N"] == 0 and d["P"] == 1:
                return "PosOnly"
            return "Tangled"

        class_stats = {}
        for cls in ["Stable", "NoiseOnly", "PosOnly", "Tangled"]:
            members = [d for d in per_item if class_of(d) == cls]
            n_members = len(members)
            n_ties = sum(1 for d in members if d["is_tie"])
            decidable = [d for d in members if not d["is_tie"]]
            n_correct = sum(1 for d in decidable if d["is_correct"])
            acc = n_correct / len(decidable) if decidable else None
            class_stats[cls] = {"n": n_members, "n_ties": n_ties,
                                "n_decidable": len(decidable), "acc": acc}

        overall_decidable = [d for d in per_item if not d["is_tie"]]
        overall_acc = (sum(1 for d in overall_decidable if d["is_correct"])
                      / len(overall_decidable)) if overall_decidable else None

        accuracy_results.append({"model": pretty(slug), "split": split,
                                 "class_stats": class_stats,
                                 "overall_acc": overall_acc,
                                 "n_total_ties": sum(1 for d in per_item if d["is_tie"])})

    print("=" * 112)
    print("Stable        = unanimous within EACH order, AND majority agrees across orders")
    print("NoiseOnly     = wobbles within an order, but the majority verdict still agrees")
    print("                across orders (looks random, doesn't change the typical answer)")
    print("PosOnly       = each order is internally UNANIMOUS, but the unanimous answer")
    print("                FLIPS between orders -- cleanest evidence of real position bias")
    print("Tangled       = both noisy AND the majority flips -- can't cleanly attribute")
    print()
    print("phi   = PRIMARY statistic: correlation between 'is this pair noisy' (N)")
    print("        and 'does its majority verdict flip' (P), as BINARY labels.")
    print("        0 = the two phenomena hit largely different pairs. Positive =")
    print("        noisy pairs tend to also be flip pairs.")
    print()
    print("rho   = SECONDARY / exploratory only. Spearman correlation between each")
    print("        pair's within-order noise rate (D_within) and its systematic")
    print("        order-effect estimate (S). CAVEAT: D_within and S are two pieces")
    print("        of one fixed total (D_cross = D_within + S by exact identity),")
    print("        so they are NOT independently measured -- a large D_cross must")
    print("        be split between them, which can produce a spurious NEGATIVE")
    print("        rho even with zero true relationship (confirmed by simulation).")
    print("        Trust phi over rho when they disagree.")
    print()
    print("If phi's CI includes 0, that is direct, item-level evidence that noise")
    print("and position bias are largely SEPARATE phenomena, even when NoiseShare")
    print("(the population ratio from analyze_decomposition.py) looks substantial.")

    # ---- second table: accuracy per class ----
    print()
    print("=" * 112)
    print("ACCURACY BY INSTABILITY CLASS (six-vote decision: all 2n=6 calls combined)")
    print("=" * 112)
    print(f"{'Model':<32}{'Split':>7}{'Overall':>9}"
          f"{'Stable':>11}{'NoiseOnly':>12}{'PosOnly':>11}{'Tangled':>11}{'Ties':>7}")
    print("-" * 112)
    for r in accuracy_results:
        def fmt(cls):
            cs = r["class_stats"][cls]
            if cs["acc"] is None:
                return f"n/a(n={cs['n']})"
            return f"{cs['acc']*100:.1f}%(n={cs['n']})"

        overall_s = f"{r['overall_acc']*100:.1f}%" if r["overall_acc"] is not None else "n/a"
        print(f"{r['model']:<32}{r['split']:>7}{overall_s:>9}"
              f"{fmt('Stable'):>16}{fmt('NoiseOnly'):>17}{fmt('PosOnly'):>16}"
              f"{fmt('Tangled'):>16}{r['n_total_ties']:>7}")

    print("=" * 112)
    print("Six-vote decision: pool all 2n=6 calls (3 AB + 3 BA) per pair, take the")
    print("majority original-response choice. A TIE (exactly 3-of-6 either way) has")
    print("NO majority and is excluded from accuracy (reported separately as Ties),")
    print("not counted as wrong -- that would be a different kind of error.")
    print()
    print("UNLIKE the old 1-AB + 1-BA design, a class's accuracy here is NOT forced")
    print("by arithmetic. A Tangled or PosOnly pair CAN be, say, 4-of-6 correct --")
    print("there is no guarantee it lands at 50%. This is now a real empirical")
    print("result: does filtering unstable pairs actually raise label accuracy on")
    print("this properly-powered design, not just on the old 2-call design where")
    print("the answer was mathematically forced.")


if __name__ == "__main__":
    main()
