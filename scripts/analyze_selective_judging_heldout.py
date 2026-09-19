#!/usr/bin/env python3
"""
analyze_selective_judging_heldout.py
---------------------------------------
Out-of-sample version of the confidence/coverage analysis.

WHY THIS REPLACES THE EARLIER VERSION
-------------------------------------
The earlier sweep sorted all 350 pairs by confidence and plotted flip rate on
those same 350 pairs. That is IN-SAMPLE: the threshold was chosen using the
very data it was then evaluated on, so the curve flatters itself. It shows
the signal exists; it does NOT show a usable rule generalises.

Here we:
  1. Split pairs into a CALIBRATION set and a TEST set (fixed seed).
  2. Choose a confidence threshold on CALIBRATION ONLY, to hit a target
     coverage.
  3. FREEZE that threshold and apply it to the TEST set, which was never
     used to pick it.
  4. Report the coverage actually achieved on TEST and the flip rate among
     the pairs kept there.
  5. Compare against a RANDOM-ABSTENTION baseline that keeps the same NUMBER
     of test pairs at random. Without this baseline you cannot tell whether
     an improvement came from the confidence signal or merely from keeping
     fewer pairs.

Two selection signals are compared, since it is not obvious which is better:
    mean_conf : average confidence across the two orderings
    min_conf  : the lower of the two confidences (pessimistic)

Uses ONLY existing judgements_judgebench__*__gpt.jsonl files. No API calls.

Usage:
    python analyze_selective_judging_heldout.py
    python analyze_selective_judging_heldout.py --calib-frac 0.4 --seed 0
"""

import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np


def model_slug(path):
    m = re.search(r"judgements_judgebench__(.+)__gpt\.jsonl", path)
    return m.group(1) if m else None


def pretty(slug):
    return slug.replace("_", "/", 1)


def load_pairs_with_signals(path):
    """One record per COMPLETE pair: flipped flag + confidence signals."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("parse_error", False):
                rows.append(rec)

    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["order"]] = r

    out = []
    for pair_id, orders in by_pair.items():
        if 1 not in orders or 2 not in orders:
            continue
        r1, r2 = orders[1], orders[2]
        c1, c2 = r1["confidence"], r2["confidence"]
        out.append({
            "pair_id": pair_id,
            "flipped": r1["picked_original"] != r2["picked_original"],
            "mean_conf": (c1 + c2) / 2.0,
            "min_conf": min(c1, c2),
        })
    return out


def threshold_for_coverage(calib, signal, target_cov):
    """Pick the confidence cutoff on CALIBRATION that keeps ~target_cov of it.
    Returns the threshold value; pairs with signal >= threshold are kept."""
    vals = np.array([p[signal] for p in calib])
    # keep the top target_cov fraction => cut at the (1-target_cov) quantile
    return float(np.quantile(vals, 1.0 - target_cov))


def evaluate_on_test(test, signal, threshold):
    """Apply a FROZEN threshold to held-out pairs."""
    kept = [p for p in test if p[signal] >= threshold]
    if not kept:
        return None, 0.0, 0
    flip_rate = sum(1 for p in kept if p["flipped"]) / len(kept)
    coverage = len(kept) / len(test)
    return flip_rate, coverage, len(kept)


def random_baseline(test, n_keep, n_draws=1000, seed=0):
    """Keep n_keep test pairs uniformly at random; average flip rate."""
    if n_keep <= 0 or n_keep > len(test):
        return None
    rng = np.random.default_rng(seed)
    flipped = np.array([p["flipped"] for p in test], dtype=float)
    idx = np.arange(len(test))
    rates = np.empty(n_draws)
    for b in range(n_draws):
        pick = rng.choice(idx, size=n_keep, replace=False)
        rates[b] = flipped[pick].mean()
    return float(np.mean(rates))


def bootstrap_flip_rate(kept_flags, n_boot=2000, seed=0):
    """95% CI for the flip rate among kept test pairs."""
    if len(kept_flags) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.array(kept_flags, dtype=float)
    idx = np.arange(len(arr))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = arr[rng.choice(idx, size=len(arr), replace=True)].mean()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-frac", type=float, default=0.4,
                    help="Fraction of pairs used to CHOOSE thresholds. The rest "
                         "is held out for evaluation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--signal", default="both",
                    choices=["mean_conf", "min_conf", "both"])
    args = ap.parse_args()

    files = sorted(glob.glob("judgements_judgebench__*__gpt.jsonl"))
    if not files:
        print("No judgements_judgebench__*__gpt.jsonl files found.")
        return

    targets = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    signals = ["mean_conf", "min_conf"] if args.signal == "both" else [args.signal]

    for path in files:
        slug = model_slug(path)
        if slug is None:
            continue
        pairs = load_pairs_with_signals(path)
        if len(pairs) < 60:
            print(f"\n{pretty(slug)}: only {len(pairs)} complete pairs, skipping "
                  f"(need enough to split).")
            continue

        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(pairs))
        n_calib = int(round(len(pairs) * args.calib_frac))
        calib = [pairs[i] for i in order[:n_calib]]
        test = [pairs[i] for i in order[n_calib:]]

        base_flip = sum(1 for p in test if p["flipped"]) / len(test)

        print()
        print("=" * 100)
        print(f"{pretty(slug)}")
        print(f"  calibration: {len(calib)} pairs   |   held-out test: {len(test)} pairs")
        print(f"  test-set flip rate with NO abstention: {base_flip*100:.1f}%")
        print("=" * 100)

        for signal in signals:
            print(f"\n  signal = {signal}")
            print(f"  {'Target':>7}{'Thresh':>9}{'TestCov':>9}{'TestFlip':>10}"
                  f"{'95% CI':>18}{'RandomBase':>12}{'Gain':>9}")
            print("  " + "-" * 88)
            for tc in targets:
                thr = threshold_for_coverage(calib, signal, tc)
                flip, cov, n_kept = evaluate_on_test(test, signal, thr)
                if flip is None:
                    print(f"  {tc*100:>6.0f}%{thr:>9.3f}   (nothing kept on test)")
                    continue
                kept_flags = [p["flipped"] for p in test if p[signal] >= thr]
                lo, hi = bootstrap_flip_rate(kept_flags, args.n_boot, args.seed)
                rand = random_baseline(test, n_kept, seed=args.seed)
                gain = (rand - flip) if rand is not None else None
                ci = f"[{lo*100:.1f}, {hi*100:.1f}]"
                rand_s = f"{rand*100:.1f}%" if rand is not None else "n/a"
                gain_s = f"{gain*100:+.1f}pp" if gain is not None else "n/a"
                print(f"  {tc*100:>6.0f}%{thr:>9.3f}{cov*100:>8.1f}%{flip*100:>9.1f}%"
                      f"{ci:>18}{rand_s:>12}{gain_s:>9}")

    print()
    print("=" * 100)
    print("HOW TO READ THIS")
    print("=" * 100)
    print("Target      coverage we aimed for when picking the threshold on CALIBRATION")
    print("Thresh      the frozen confidence cutoff (chosen WITHOUT seeing test data)")
    print("TestCov     coverage actually achieved on the held-out test pairs. It will")
    print("            not exactly equal Target -- that mismatch is itself a result,")
    print("            showing how well the calibrated threshold transfers.")
    print("TestFlip    flip rate among kept TEST pairs. This is the honest number.")
    print("RandomBase  flip rate if you kept the same NUMBER of test pairs at random.")
    print("Gain        RandomBase - TestFlip. THIS is the value the confidence signal")
    print("            actually adds. If Gain is ~0, abstention is working only")
    print("            because it discards pairs, not because confidence is")
    print("            informative -- the whole selective-judging claim rests on")
    print("            this column, not on TestFlip alone.")
    print()
    print("Coarse verbalized confidence takes few distinct values, so several")
    print("targets can collapse onto the same threshold and produce identical rows.")
    print("That is a property of the signal, not a bug.")


if __name__ == "__main__":
    main()
