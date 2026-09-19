#!/usr/bin/env python3
"""
analyze_margin_v2.py
------------------------
Does a judge's overall preference STRENGTH for a pair predict whether that
pair is unstable? Intuition: a close call should be easier to tip over
(by position, or by ordinary randomness) than a lopsided one.

TWO MARGIN DEFINITIONS -- READ THIS BEFORE USING EITHER
-----------------------------------------------------------
There is an intuitive way to define "margin" that is MATHEMATICALLY BROKEN
for this purpose: combine all 2n calls into one vote share, then measure
distance from 50/50 -- i.e. margin = |(x+y) - (2n-(x+y))| / (2n). This was
tested against synthetic data with NO true relationship between closeness
and instability, and produced a SPURIOUS correlation of rho ~= -0.78,
nearly as strong as a real effect. Why: a pure position flip (x=n, y=0 --
maximum confidence each way, just reversed) collapses to a combined vote
share of exactly 50/50, which this formula misreads as "the closest
possible call." It is actually the opposite: maximum confidence in each
direction separately. Do not use this formula. (This matches a formula
ChatGPT independently proposed with a similar name; it is the same broken
metric, confirmed algebraically identical.)

Two SAFE alternatives are used here instead, both verified against the
same synthetic independence test (rho ~= 0 confirmed for both):

  MARGIN_AB   = |2*(x/n) - 1|
      Uses ONLY the AB calls. Simple, but throws away the BA data.

  MARGIN_AVG  = ( |2*(x/n)-1| + |2*(y/n)-1| ) / 2
      Averages the SEPARATELY-COMPUTED margins of each ordering (not the
      combined vote share). This is the corrected version of "order-
      averaged preference strength" -- because absolute value is taken
      BEFORE averaging, a pure flip correctly registers as high-margin in
      EACH direction, rather than canceling out. Uses both orderings, so
      this is the PRIMARY signal in this script; MARGIN_AB is reported
      alongside it for comparison only.

IMPORTANT SECOND FINDING (found during testing, before shipping)
--------------------------------------------------------------------
Even the SAFE margin definitions above are NOT safe to correlate against
S or D_within. Both margin_AB and margin_AVG are built from x(n-x) and
y(n-y) -- the exact same quantities that mechanically DEFINE D_within.
For n=3: x=0 gives margin=1 and x(n-x)=0; x=1 gives margin=1/3 and
x(n-x)=2 -- margin is high EXACTLY where D_within's ingredient is low,
for every possible x, with no exceptions. Testing "does margin predict
noise" is close to asking "is x related to x." Verified: under true
independence, rho(margin, S) came back +0.68 and rho(margin, D_within)
came back -0.94 -- nearly deterministic, not spread.

Only P (majority flip) is a genuinely separate question from margin --
P depends on WHICH SIDE the majority favors, not on how extreme/close the
vote was, so it isn't a rescaling of the same quantity. Verified safe
under independence (rho ~= 0). THIS SCRIPT THEREFORE ONLY TESTS MARGIN
AGAINST P. Do not add S, D_within, or 'any' (which includes a D_within
component) as targets without re-deriving whether they're independent of
whatever margin definition is in use -- they were not, twice, in this
project's history.

TARGET
------
  P : does the majority verdict flip between AB and BA

Usage:
    python analyze_margin_v2.py
"""

import glob
import json
import re
from collections import defaultdict

import numpy as np

from bootstrap_utils import bootstrap_ci
from analyze_pair_classes import classify_item, six_vote_decision


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


def decompose_xy(x, y, n):
    """D_cross = D_within + S identity. Same formula used throughout this
    project (analyze_decomposition.py, analyze_pair_classes.py)."""
    n_pairs_cross = n * n
    n_pairs_within = n * (n - 1) / 2
    d_cross = (x * (n - y) + (n - x) * y) / n_pairs_cross
    if n_pairs_within > 0:
        d_ab = (x * (n - x)) / n_pairs_within
        d_ba = (y * (n - y)) / n_pairs_within
        d_within = (d_ab + d_ba) / 2
    else:
        d_within = 0.0
    return d_within, d_cross - d_within  # (D_within, S)


def margin_ab(x, y, n):
    return abs(2 * (x / n) - 1)


def margin_avg(x, y, n):
    return (abs(2 * (x / n) - 1) + abs(2 * (y / n) - 1)) / 2


def build_items(rows, n_reps=3):
    by_item = defaultdict(lambda: {1: {}, 2: {}})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]][r["rep_idx"]] = r
        gt_of[r["pair_id"]] = r["gt"]

    items = []
    for pair_id, orders in by_item.items():
        o1 = [orders[1][k] for k in sorted(orders[1]) if k < n_reps]
        o2 = [orders[2][k] for k in sorted(orders[2]) if k < n_reps]
        if len(o1) < n_reps or len(o2) < n_reps or 0 not in orders[1]:
            continue
        x = sum(1 for r in o1 if r["picked_original"] == "A")
        y = sum(1 for r in o2 if r["picked_original"] == "A")

        _, P = classify_item(x, y, n_reps)
        d_within, s = decompose_xy(x, y, n_reps)
        _, is_tie, is_correct = six_vote_decision(x, y, n_reps, gt_of[pair_id])

        items.append({
            "pair_id": pair_id, "x": x, "y": y, "n": n_reps,
            "margin_ab": margin_ab(x, y, n_reps),
            "margin_avg": margin_avg(x, y, n_reps),
            "P": P, "any": int((d_within > 0) or (P == 1)),
            "d_within": d_within, "s": s,
            "correct": None if is_tie else int(is_correct),
            "conf": orders[1][0]["confidence"],
        })
    return items


def spearman_rho(x, y):
    """Returns None if either variable has zero variance (correlation undefined),
    e.g. a model whose margin or S never varies across items."""
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


def fmt_rho(rho, lo, hi):
    if rho is None:
        return "n/a"
    return f"{rho:+.3f} [{lo:+.2f},{hi:+.2f}]"


def rho_with_ci(items, x_key, y_key, n_boot=1500, seed=0):
    xs = [it[x_key] for it in items]
    ys = [it[y_key] for it in items]
    rho = spearman_rho(xs, ys)
    if rho is None:
        return None, None, None

    def stat(sample):
        sx = [it[x_key] for it in sample]
        sy = [it[y_key] for it in sample]
        v = spearman_rho(sx, sy)
        return v if v is not None else 0.0

    _, lo, hi = bootstrap_ci(items, stat, n_boot=n_boot, seed=seed)
    return rho, lo, hi


def main():
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    print("=" * 128)
    print("DOES A CLOSE-CALL MARGIN PREDICT WHETHER THE MAJORITY VERDICT FLIPS?")
    print("=" * 128)
    print(f"{'Model':<30}{'Split':>7}{'n':>6}{'rho(margin_avg, P)':>24}{'rho(margin_ab, P)':>22}")
    print("-" * 128)

    all_results = []
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

        r_avg = rho_with_ci(items, "margin_avg", "P")
        r_ab = rho_with_ci(items, "margin_ab", "P")

        print(f"{pretty(slug):<30}{split:>7}{len(items):>6}"
              f"{fmt_rho(*r_avg):>24}{fmt_rho(*r_ab):>22}")

        all_results.append((pretty(slug), split, items))

    print("=" * 128)
    print("MARGIN_AVG = average of the two orderings' OWN margins (each computed")
    print("             separately, THEN averaged) -- verified safe from the earlier")
    print("             mechanical-artifact bug. Uses both AB and BA data. PRIMARY.")
    print("MARGIN_AB  = margin computed from AB calls only. Secondary/comparison.")
    print()
    print("P = does the MAJORITY verdict flip between AB and BA.")
    print()
    print("NEGATIVE rho means: close-call pairs (low margin) are MORE likely to")
    print("flip. A CI that excludes 0 means the relationship is real, not noise.")
    print("'n/a' means this model's margin or P never varies for it (correlation")
    print("is undefined, not zero).")
    print()
    print("NOTE: this script deliberately does NOT correlate margin against S or")
    print("D_within -- both were found to be tautologically entangled with margin")
    print("(margin and D_within are built from the same underlying quantity, just")
    print("read in opposite directions). See the docstring for the numbers that")
    print("showed this. Only P is a genuinely separate, testable question.")

    # ---- quartile breakdown, target = P (flip rate) only ----
    print()
    print("=" * 128)
    print("QUARTILE BREAKDOWN (by MARGIN_AVG) -- primary view for the paper")
    print("=" * 128)
    for name, split, items in all_results:
        sorted_items = sorted(items, key=lambda it: it["margin_avg"])
        q_size = len(sorted_items) // 4
        print(f"\n{name}  [{split} split]  (n={len(items)})")
        print(f"  {'Quartile':<18}{'MarginRange':>16}{'FlipRate%':>11}{'Accuracy':>10}{'MeanConf':>10}")
        for qi in range(4):
            start = qi * q_size
            end = (qi + 1) * q_size if qi < 3 else len(sorted_items)
            q = sorted_items[start:end]
            if not q:
                continue
            m_lo, m_hi = q[0]["margin_avg"], q[-1]["margin_avg"]
            flip_rate = 100 * np.mean([it["P"] for it in q])
            acc_vals = [it["correct"] for it in q if it["correct"] is not None]
            acc = 100 * np.mean(acc_vals) if acc_vals else float("nan")
            mean_conf = np.mean([it["conf"] for it in q])
            label = ["Closest 25%", "25-50%", "50-75%", "Most lopsided 25%"][qi]
            print(f"  {label:<18}[{m_lo:.2f},{m_hi:.2f}]".rjust(20) +
                  f"{flip_rate:>10.1f}%{acc:>9.1f}%{mean_conf:>10.3f}")

    print()
    print("=" * 128)
    print("If FlipRate% generally FALLS from 'Closest 25%' to 'Most lopsided 25%',")
    print("that is clear evidence close calls really are more prone to flipping for")
    print("this model. Accuracy and MeanConf are shown for context -- neither is")
    print("tautologically tied to margin, so real patterns there are meaningful.")
    print("(MeanS and MeanDwithin are deliberately NOT shown here -- see the note")
    print("above on why they cannot be validly compared against margin.)")


if __name__ == "__main__":
    main()
