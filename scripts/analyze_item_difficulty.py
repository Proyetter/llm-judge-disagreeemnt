#!/usr/bin/env python3
"""
analyze_item_difficulty.py
------------------------------
Does position sensitivity happen especially on genuinely DIFFICULT
questions, or is it a separate model quirk unrelated to difficulty?

DIFFICULTY (leave-one-model-out)
-----------------------------------
For a given TARGET model, difficulty of an item is computed using every
OTHER model's six-vote correctness on that item (never the target model's
own data, to avoid circularity):
    difficulty = 1 - (fraction of OTHER models that got it right)
A tie (no majority) counts as "not right" for this purpose, since failing
to reach a decision isn't success. Requires at least 2 other models with
valid data for that item, or the item is skipped.

Then items are split into difficulty quartiles (easiest 25%, ..., hardest
25%) and, within each quartile, we report the TARGET model's own:
    systematic order effect (S), within-order noise (D_within),
    position-only rate, accuracy, mean single-shot confidence

Also reports Spearman correlation between difficulty and S (and D_within),
with a pair-level bootstrap CI.

NOTE ON INDEPENDENCE: unlike the earlier margin analysis (which was built
from the SAME model's own calls and had a confirmed mechanical artifact),
difficulty here comes entirely from OTHER models' data, so it does not
share any computation with the target model's own S/D_within values. No
shared-total coupling is expected -- verified with a synthetic null check
during development (not included in this script; see project history).

Usage:
    python analyze_item_difficulty.py
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


def build_items(rows, n_reps=3):
    """pair_id -> {x, y, n, gt, single_shot_conf}"""
    by_item = defaultdict(lambda: {1: {}, 2: {}})
    gt_of = {}
    for r in rows:
        by_item[r["pair_id"]][r["order"]][r["rep_idx"]] = r
        gt_of[r["pair_id"]] = r["gt"]

    items = {}
    for pair_id, orders in by_item.items():
        o1 = [orders[1][k] for k in sorted(orders[1]) if k < n_reps]
        o2 = [orders[2][k] for k in sorted(orders[2]) if k < n_reps]
        if len(o1) < n_reps or len(o2) < n_reps or 0 not in orders[1]:
            continue
        x = sum(1 for r in o1 if r["picked_original"] == "A")
        y = sum(1 for r in o2 if r["picked_original"] == "A")
        items[pair_id] = {"x": x, "y": y, "n": n_reps, "gt": gt_of[pair_id],
                          "single_shot_conf": orders[1][0]["confidence"]}
    return items


def decompose_xy(x, y, n):
    """(x,y,n)-argument decomposition, matching analyze_pair_classes' convention."""
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
    files = sorted(glob.glob("factorial_judgebench__*.jsonl"))
    if not files:
        print("No factorial_judgebench__*.jsonl files found.")
        return

    by_split = defaultdict(dict)
    for path in files:
        slug, split = parse_filename(path)
        if slug is None:
            continue
        rows = load_valid(path)
        if not rows:
            continue
        by_split[split][pretty(slug)] = build_items(rows)

    for split, per_model_items in by_split.items():
        model_names = sorted(per_model_items)
        if len(model_names) < 4:
            print(f"(split={split}: fewer than 4 models, skipping -- need enough "
                  f"OTHER models for leave-one-out difficulty)")
            continue

        # correctness matrix: model -> {pair_id: correct_flag (0 or 1, tie=0)}
        correctness = {}
        for name in model_names:
            items = per_model_items[name]
            flags = {}
            for pid, it in items.items():
                _, is_tie, is_correct = six_vote_decision(it["x"], it["y"], it["n"], it["gt"])
                flags[pid] = 0 if is_tie else int(is_correct)
            correctness[name] = flags

        print("=" * 108)
        print(f"DOES ITEM DIFFICULTY PREDICT INSTABILITY? -- split = {split}")
        print("=" * 108)

        for target in model_names:
            other_models = [m for m in model_names if m != target]
            target_items = per_model_items[target]

            records = []
            for pid, it in target_items.items():
                others_scores = [correctness[m][pid] for m in other_models if pid in correctness[m]]
                if len(others_scores) < 2:
                    continue
                difficulty = 1 - float(np.mean(others_scores))

                d_cross, d_within, s = decompose_xy(it["x"], it["y"], it["n"])
                _, P = classify_item(it["x"], it["y"], it["n"])
                _, is_tie, is_correct = six_vote_decision(it["x"], it["y"], it["n"], it["gt"])

                records.append({
                    "pair_id": pid, "difficulty": difficulty, "s": s,
                    "d_within": d_within, "P": P,
                    "correct": None if is_tie else int(is_correct),
                    "conf": it["single_shot_conf"],
                })

            if len(records) < 20:
                continue

            difficulties = [r["difficulty"] for r in records]
            svals = [r["s"] for r in records]
            dwvals = [r["d_within"] for r in records]

            rho_s = spearman_rho(difficulties, svals)
            rho_dw = spearman_rho(difficulties, dwvals)

            def rho_s_stat(sample):
                d = [x["difficulty"] for x in sample]
                s = [x["s"] for x in sample]
                v = spearman_rho(d, s)
                return v if v is not None else 0.0

            def rho_dw_stat(sample):
                d = [x["difficulty"] for x in sample]
                dw = [x["d_within"] for x in sample]
                v = spearman_rho(d, dw)
                return v if v is not None else 0.0

            _, lo_s, hi_s = bootstrap_ci(records, rho_s_stat, n_boot=1500, seed=0)
            _, lo_dw, hi_dw = bootstrap_ci(records, rho_dw_stat, n_boot=1500, seed=1)

            print(f"\n{target}  (n={len(records)} items with >=2 other models' data)")
            if rho_s is None:
                print(f"  rho(difficulty, S)        = n/a (no variance in S -- this "
                      f"model's order effect never varies, so correlation is undefined)")
            else:
                print(f"  rho(difficulty, S)        = {rho_s:+.3f}  95% CI [{lo_s:+.2f}, {hi_s:+.2f}]"
                      f"   (S = systematic order effect)")
            if rho_dw is None:
                print(f"  rho(difficulty, D_within) = n/a (no variance in D_within -- this "
                      f"model shows ~zero within-order noise on every item, so correlation "
                      f"is undefined, not zero)")
            else:
                print(f"  rho(difficulty, D_within) = {rho_dw:+.3f}  95% CI [{lo_dw:+.2f}, {hi_dw:+.2f}]"
                      f"   (D_within = within-order noise)")

            # quartile breakdown
            sorted_recs = sorted(records, key=lambda r: r["difficulty"])
            q_size = len(sorted_recs) // 4
            print(f"  {'Quartile':<16}{'DiffRange':>14}{'MeanS':>9}{'MeanDwithin':>13}"
                  f"{'PosOnly%':>10}{'Accuracy':>10}{'MeanConf':>10}")
            for qi in range(4):
                start = qi * q_size
                end = (qi + 1) * q_size if qi < 3 else len(sorted_recs)
                q = sorted_recs[start:end]
                if not q:
                    continue
                d_lo, d_hi = q[0]["difficulty"], q[-1]["difficulty"]
                mean_s = np.mean([r["s"] for r in q])
                mean_dw = np.mean([r["d_within"] for r in q])
                pos_only_rate = 100 * np.mean([r["P"] == 1 and r["d_within"] == 0 for r in q])
                acc_vals = [r["correct"] for r in q if r["correct"] is not None]
                acc = 100 * np.mean(acc_vals) if acc_vals else float("nan")
                mean_conf = np.mean([r["conf"] for r in q])
                label = ["Easiest 25%", "25-50%", "50-75%", "Hardest 25%"][qi]
                print(f"  {label:<16}[{d_lo:.2f},{d_hi:.2f}]".rjust(20) +
                      f"{mean_s:>9.3f}{mean_dw:>13.3f}{pos_only_rate:>9.1f}%"
                      f"{acc:>9.1f}%{mean_conf:>10.3f}")

        print()

    print("=" * 108)
    print("HOW TO READ THIS")
    print("=" * 108)
    print("rho(difficulty, S) > 0 with a CI excluding 0: harder items (per OTHER")
    print("models) genuinely show MORE systematic order sensitivity for this model --")
    print("difficulty is a real mechanism behind position bias.")
    print()
    print("rho near 0: order sensitivity happens on items regardless of how hard")
    print("they are for other models -- it looks like a separate model-specific")
    print("quirk, not something explained by item difficulty.")
    print()
    print("In the quartile table, look for MeanS and MeanDwithin generally rising")
    print("from 'Easiest 25%' to 'Hardest 25%' -- a monotonic trend is the clearest")
    print("visual evidence of a real difficulty effect.")


if __name__ == "__main__":
    main()
