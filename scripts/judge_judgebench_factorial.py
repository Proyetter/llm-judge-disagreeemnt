#!/usr/bin/env python3
"""
judge_judgebench_factorial.py
--------------------------------
Runs a FULL FACTORIAL design: every pair is judged n times in ordering AB
AND n times in ordering BA.

WHY THIS EXISTS
---------------
The earlier approach ran each ordering ONCE (judge_judgebench.py) and
separately ran one ordering MANY times (judge_judgebench_repeat.py), then
estimated position bias by subtracting:

    true_position_effect ~= swap_flip_rate - same_order_noise_rate

That subtraction is NOT a valid estimator. Writing p = P(pick original A |
A shown first) and q = P(pick original A | A shown second):

    swap flip rate      = p(1-q) + (1-p)q
    same-order noise    = 2p(1-p)
    their difference    = (q - p)(1 - 2p)

The real order effect is (p - q), but the difference carries a spurious
(1 - 2p) factor. When p is near 0.5 that factor vanishes, so the subtraction
reports ~zero order effect no matter how large the true effect is. It
understates the effect, and does so worst for the noisiest judges.

With n repeats of BOTH orderings we can estimate p and q directly per item,
and report (p - q) with a proper confidence interval. That is interpretable
and unbiased.

Output goes to its own file, separate from the swap and repeat experiments:

    factorial_judgebench__{model}__{split}.jsonl

Each record is ONE trial, keyed by (pair_id, order, rep_idx).

Usage:
    conda activate judge
    export OPENROUTER_API_KEY="sk-or-..."

    # smoke test
    python judge_judgebench_factorial.py --model openai/gpt-4o-mini --limit 3 --n-reps 5

    # real run: 150 pairs x 2 orderings x 5 reps = 1500 calls
    python judge_judgebench_factorial.py --model openai/gpt-4o-mini --limit 150 --n-reps 5

COST NOTE: calls = pairs * 2 * n_reps. At 150 pairs and n_reps=5 that is
1500 calls per model, roughly 2x what a full 350-pair swap run costs. Start
with fewer pairs rather than fewer reps -- reps are what make the estimate
valid, pairs only tighten the confidence interval.
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# Reuse the identical prompt, parsing, and call logic as the other experiments
# so results are directly comparable across all three designs.
from judge_judgebench import load_pairs, call_judge


def already_done_factorial(out_path):
    """Set of (pair_id, order, rep_idx) already recorded, for resume."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["pair_id"], rec["order"], rec["rep_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter model string, e.g. openai/gpt-4o-mini")
    ap.add_argument("--split", default="gpt", choices=["gpt", "claude", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many PAIRS. Recommended: 100-150 "
                         "for a real run, since cost scales as pairs*2*n_reps.")
    ap.add_argument("--n-reps", type=int, default=3,
                    help="Repeats of EACH ordering. Default is 3, matching the "
                         "existing order-1 repeat data collected earlier via "
                         "judge_judgebench_repeat.py -- so migrated data lines up "
                         "without needing extra reps. Raise this later only if "
                         "you decide you want a tighter estimate and are willing "
                         "to pay for more calls on top of what's already done.")
    ap.add_argument("--max-tokens", type=int, default=150,
                    help="150 for standard models; 2000-4000 for reasoning models.")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="For reasoning models only (see judge_judgebench.py).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent API calls in flight.")
    ap.add_argument("--out", default=None, help="Override auto output filename.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=...).")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = args.out or f"factorial_judgebench__{safe_model}__{args.split}.jsonl"

    pairs = load_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    done = already_done_factorial(out_path)
    total_planned = len(pairs) * 2 * args.n_reps
    print(f"Model: {args.model}  |  max_tokens: {args.max_tokens}  |  "
          f"reasoning_effort: {args.reasoning_effort}")
    print(f"Split: {args.split}  |  pairs: {len(pairs)}  |  n_reps per ordering: {args.n_reps}")
    print(f"Planned trials: {len(pairs)} pairs x 2 orderings x {args.n_reps} reps "
          f"= {total_planned}")
    print(f"Already done: {len(done)}  |  Remaining: {total_planned - len(done)}")
    print(f"Writing to: {out_path}\n")

    write_lock = threading.Lock()

    def process_task(pair, order, rep_idx):
        """One trial. order 1 => A=resp_a first; order 2 => A=resp_b first (swapped).
        rep_idx distinguishes repeated identical trials of the SAME ordering."""
        if order == 1:
            shown_a, shown_b = pair["resp_a"], pair["resp_b"]
        else:
            shown_a, shown_b = pair["resp_b"], pair["resp_a"]

        verdict, conf = call_judge(
            client, args.model, pair["question"], shown_a, shown_b,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )

        base = {
            "pair_id": pair["pair_id"], "split": pair["split"],
            "source": pair["source"], "order": order, "rep_idx": rep_idx,
            "model": args.model,
        }
        if verdict is None:
            base["parse_error"] = True
            return base

        # map the letter it output back to which ORIGINAL response it chose
        if order == 1:
            picked_original = verdict            # A->A, B->B
        else:
            picked_original = "B" if verdict == "A" else "A"   # swapped

        base.update({
            "raw_verdict": verdict,              # position letter it emitted
            "picked_original": picked_original,  # underlying response chosen
            "confidence": conf,
            "gt": pair["gt"],
            "correct": picked_original == pair["gt"],
            "parse_error": False,
        })
        return base

    tasks = []
    for pair in pairs:
        for order in (1, 2):
            for rep_idx in range(args.n_reps):
                if (pair["pair_id"], order, rep_idx) not in done:
                    tasks.append((pair, order, rep_idx))

    with open(out_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_task, p, o, r): (p, o, r)
                for p, o, r in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="trials"):
                rec = future.result()
                with write_lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
