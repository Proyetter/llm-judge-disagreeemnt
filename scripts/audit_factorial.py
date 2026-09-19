#!/usr/bin/env python3
"""
audit_factorial.py
--------------------
Data-integrity audit for the 3xAB / 3xBA factorial runs. Run this BEFORE any
substantive analysis. No result matters until this passes.

Checks, per model and split:
  1. Row count vs expected (n_pairs * 2 orders * n_reps)
  2. Duplicate (pair_id, order, rep_idx) keys
  3. Completeness: how many pairs have all 6 valid calls
  4. Parse-error rate, BROKEN DOWN BY ORDER -- this one matters most.
     If AB fails more often than BA (or vice versa), then dropping invalid
     rows removes data non-randomly and can BIAS the order effect itself.
     A large AB-vs-BA gap is a red flag, not a cosmetic issue.
  5. Parse-error rate by repeat index (drift within a run)
  6. Confidence validity: anything outside [0.5, 1.0] should be absent,
     since the parser is supposed to reject rather than clamp
  7. Confidence granularity: how many distinct values the model actually
     used. Verbalized confidence is often coarse, which limits every
     downstream calibration/AUROC analysis.

Usage:
    python audit_factorial.py
    python audit_factorial.py --expected-reps 3
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict


EXPECTED_PAIRS = {"gpt": 350, "claude": 270}


def parse_filename(path):
    m = re.search(r"factorial_judgebench__(.+)__(gpt|claude|both)\.jsonl", path)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def pretty(slug):
    return slug.replace("_", "/", 1)


def load_all_rows(path):
    """Load EVERY row, including parse errors -- the audit needs the failures."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"__malformed__": True})
    return rows


def audit_one(path, expected_reps):
    slug, split = parse_filename(path)
    if slug is None:
        return None

    rows = load_all_rows(path)
    malformed = sum(1 for r in rows if r.get("__malformed__"))
    rows = [r for r in rows if not r.get("__malformed__")]

    expected_pairs = EXPECTED_PAIRS.get(split)
    expected_rows = expected_pairs * 2 * expected_reps if expected_pairs else None

    # --- duplicates ---
    keys = Counter((r.get("pair_id"), r.get("order"), r.get("rep_idx")) for r in rows)
    duplicates = {k: c for k, c in keys.items() if c > 1}

    # --- parse errors, split by order ---
    err_by_order = defaultdict(lambda: [0, 0])  # order -> [errors, total]
    err_by_rep = defaultdict(lambda: [0, 0])
    for r in rows:
        o, rep = r.get("order"), r.get("rep_idx")
        is_err = bool(r.get("parse_error", False))
        err_by_order[o][1] += 1
        err_by_rep[rep][1] += 1
        if is_err:
            err_by_order[o][0] += 1
            err_by_rep[rep][0] += 1

    total_rows = len(rows)
    total_errs = sum(v[0] for v in err_by_order.values())

    # --- completeness per pair (valid rows only) ---
    valid_counts = defaultdict(lambda: {1: 0, 2: 0})
    for r in rows:
        if not r.get("parse_error", False):
            o = r.get("order")
            if o in (1, 2):
                valid_counts[r["pair_id"]][o] += 1

    full = partial = unusable = 0
    for pid, c in valid_counts.items():
        if c[1] >= expected_reps and c[2] >= expected_reps:
            full += 1
        elif c[1] >= 2 and c[2] >= 2:
            partial += 1
        else:
            unusable += 1
    missing_pairs = (expected_pairs - len(valid_counts)) if expected_pairs else None

    # --- confidence checks (valid rows only) ---
    confs = [r["confidence"] for r in rows
             if not r.get("parse_error", False) and "confidence" in r]
    out_of_range = [c for c in confs if c < 0.5 or c > 1.0]
    distinct = sorted(set(confs))

    return {
        "model": pretty(slug), "split": split, "path": path,
        "malformed_lines": malformed,
        "total_rows": total_rows, "expected_rows": expected_rows,
        "duplicates": duplicates,
        "total_errs": total_errs,
        "err_by_order": dict(err_by_order),
        "err_by_rep": dict(err_by_rep),
        "pairs_seen": len(valid_counts), "expected_pairs": expected_pairs,
        "missing_pairs": missing_pairs,
        "full": full, "partial": partial, "unusable": unusable,
        "n_conf": len(confs), "out_of_range": out_of_range,
        "distinct_conf": distinct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-reps", type=int, default=3)
    ap.add_argument("--order-gap-threshold", type=float, default=5.0,
                    help="Flag if AB and BA parse-error rates differ by more "
                         "than this many percentage points.")
    args = ap.parse_args()

    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    results = [a for a in (audit_one(p, args.expected_reps) for p in files) if a]
    if not results:
        print("No parseable factorial files found.")
        return

    problems = []

    print("=" * 108)
    print("1. ROW COUNTS AND COMPLETENESS")
    print("=" * 108)
    print(f"{'Model':<30}{'Split':>8}{'Rows':>8}{'Exp':>7}{'Pairs':>7}"
          f"{'Full6':>8}{'Partial':>9}{'Unusable':>10}{'Missing':>9}")
    print("-" * 108)
    for r in results:
        exp = r["expected_rows"] if r["expected_rows"] else "?"
        miss = r["missing_pairs"] if r["missing_pairs"] is not None else "?"
        print(f"{r['model']:<30}{r['split']:>8}{r['total_rows']:>8}{str(exp):>7}"
              f"{r['pairs_seen']:>7}{r['full']:>8}{r['partial']:>9}"
              f"{r['unusable']:>10}{str(miss):>9}")
        if r["expected_rows"] and r["total_rows"] != r["expected_rows"]:
            problems.append(f"{r['model']} [{r['split']}]: {r['total_rows']} rows, "
                            f"expected {r['expected_rows']}")
        if r["duplicates"]:
            problems.append(f"{r['model']} [{r['split']}]: "
                            f"{len(r['duplicates'])} DUPLICATE keys")
        if r["malformed_lines"]:
            problems.append(f"{r['model']} [{r['split']}]: "
                            f"{r['malformed_lines']} malformed JSON lines")
    print()
    print("Full6    = pairs with >= expected reps valid in BOTH orders (primary analysis set)")
    print("Partial  = pairs with >= 2 valid in both orders (usable, reduced precision)")
    print("Unusable = pairs missing too much data in at least one order")

    print()
    print("=" * 108)
    print("2. PARSE ERRORS BY ORDER  <-- the bias-critical check")
    print("=" * 108)
    print(f"{'Model':<30}{'Split':>8}{'Overall':>10}{'AB err':>10}{'BA err':>10}"
          f"{'Gap(pp)':>10}{'Flag':>8}")
    print("-" * 108)
    for r in results:
        eo = r["err_by_order"]
        ab = eo.get(1, [0, 0]); ba = eo.get(2, [0, 0])
        ab_rate = 100 * ab[0] / ab[1] if ab[1] else 0.0
        ba_rate = 100 * ba[0] / ba[1] if ba[1] else 0.0
        overall = 100 * r["total_errs"] / r["total_rows"] if r["total_rows"] else 0.0
        gap = abs(ab_rate - ba_rate)
        flag = "<<<" if gap > args.order_gap_threshold else ""
        print(f"{r['model']:<30}{r['split']:>8}{overall:>9.1f}%{ab_rate:>9.1f}%"
              f"{ba_rate:>9.1f}%{gap:>9.1f}{flag:>8}")
        if gap > args.order_gap_threshold:
            problems.append(f"{r['model']} [{r['split']}]: parse-error rate differs "
                            f"by {gap:.1f}pp between AB and BA -- dropping invalid "
                            f"rows may BIAS the order effect")
    print()
    print("If AB and BA fail at different rates, the surviving data is not a random")
    print("sample of attempts, and the measured order effect can be distorted. A gap")
    print(f"above {args.order_gap_threshold:.0f}pp is flagged with '<<<'.")

    print()
    print("=" * 108)
    print("3. PARSE ERRORS BY REPEAT INDEX (drift within a run)")
    print("=" * 108)
    for r in results:
        parts = []
        for rep in sorted(k for k in r["err_by_rep"] if k is not None):
            e, t = r["err_by_rep"][rep]
            parts.append(f"rep{rep}={100*e/t:.1f}%" if t else f"rep{rep}=n/a")
        print(f"  {r['model']:<30}[{r['split']:>6}]  " + "  ".join(parts))

    print()
    print("=" * 108)
    print("4. CONFIDENCE VALIDITY AND GRANULARITY")
    print("=" * 108)
    print(f"{'Model':<30}{'Split':>8}{'nConf':>8}{'OutOfRange':>12}"
          f"{'Distinct':>10}  Values")
    print("-" * 108)
    for r in results:
        d = r["distinct_conf"]
        shown = ", ".join(f"{v:.2f}" for v in d[:8]) + (" ..." if len(d) > 8 else "")
        print(f"{r['model']:<30}{r['split']:>8}{r['n_conf']:>8}"
              f"{len(r['out_of_range']):>12}{len(d):>10}  {shown}")
        if r["out_of_range"]:
            problems.append(f"{r['model']} [{r['split']}]: {len(r['out_of_range'])} "
                            f"confidence values outside [0.5, 1.0] -- parser should "
                            f"have REJECTED these, not stored them")
        if len(d) <= 3:
            problems.append(f"{r['model']} [{r['split']}]: only {len(d)} distinct "
                            f"confidence values -- AUROC/calibration on this model "
                            f"will have very little resolution")
    print()
    print("Few distinct values = coarse signal. Every confidence-based analysis")
    print("(AUROC, calibration, threshold sweeps) is limited by this.")

    print()
    print("=" * 108)
    if problems:
        print(f"AUDIT FINDINGS: {len(problems)} item(s) need attention")
        print("=" * 108)
        for p in problems:
            print(f"  - {p}")
        print()
        print("Resolve or explicitly document each before running analysis.")
    else:
        print("AUDIT PASSED: no structural problems detected.")
        print("=" * 108)


if __name__ == "__main__":
    main()
