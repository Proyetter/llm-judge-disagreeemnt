#!/usr/bin/env python3
"""
judge_judgebench_repeat.py
----------------------------
Establishes a NOISE FLOOR for flip rate. The original swap experiment
(judge_judgebench.py) shows each pair to the judge in BOTH orderings, once
each, and calls it a "flip" if the verdict changes. But that conflates two
different things:

  1. Real position bias: the ordering itself changes the verdict.
  2. Plain run-to-run randomness: the model would give a different verdict
     even if you asked it the exact same question, in the exact same order,
     more than once.

This script isolates (2). It repeats the SAME ordering (never swapped) N
times per pair, and checks how often the verdict changes anyway. That gives
a same-order "noise floor" flip rate. Once you have both numbers:

    true_position_effect ~= swap_flip_rate - same_order_flip_rate

Reuses PROMPT_TEMPLATE, parsing, and the call_judge function from
judge_judgebench.py so this experiment is directly comparable to the swap
experiment (same prompt, same parsing rules) -- only the experimental
design (repeat vs. swap) differs.

Output is written to a SEPARATE file from the swap experiment, since it is
a different experimental design and the record schema differs (repeat_idx
instead of order):

    repeat_test_judgebench__{model}__{split}.jsonl

Usage:
    conda activate judge
    export OPENROUTER_API_KEY="sk-or-..."

    # quick sanity test: 5 pairs, 3 repeats, one model
    python judge_judgebench_repeat.py --model openai/gpt-4o-mini --split gpt --limit 5 --n-repeats 3

    # full run: all 350 pairs, 3 repeats, one model
    python judge_judgebench_repeat.py --model openai/gpt-4o-mini --split gpt --n-repeats 3
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

# Reuse the exact same dataset loader and judge-calling logic as the swap
# experiment, so results are directly comparable (same prompt, same parsing).
from judge_judgebench import load_pairs, call_judge


def already_done_repeat(out_path):
    """Return the set of (pair_id, repeat_idx) already recorded, for resume."""
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
                done.add((rec["pair_id"], rec["repeat_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter model string, e.g. openai/gpt-4o-mini")
    ap.add_argument("--split", default="gpt", choices=["gpt", "claude", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many PAIRS (mainly for a quick test run).")
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="Number of times to repeat EACH pair with the SAME "
                         "ordering (no swap). 3 is enough to detect obvious "
                         "instability; use more for a cleaner noise estimate "
                         "at added cost.")
    ap.add_argument("--max-tokens", type=int, default=150,
                    help="Same guidance as judge_judgebench.py: 150 for "
                         "standard models, 2000-4000 for reasoning models.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of concurrent API calls in flight.")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="For REASONING models only (Gemini 2.5 Pro, DeepSeek R1, "
                         "Claude Sonnet 5, etc.): caps how much internal thinking "
                         "the model does before answering. Use 'low' to keep cost "
                         "and truncation risk down. Leave unset for standard "
                         "non-reasoning models.")
    ap.add_argument("--out", default=None, help="Override the auto output filename.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=...).")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = args.out or f"repeat_test_judgebench__{safe_model}__{args.split}.jsonl"

    pairs = load_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    done = already_done_repeat(out_path)
    print(f"Model: {args.model}  |  max_tokens: {args.max_tokens}  |  n_repeats: {args.n_repeats}  |  reasoning_effort: {args.reasoning_effort}")
    print(f"Split: {args.split}  |  pairs: {len(pairs)}  |  already done: {len(done)} (pair,repeat) entries")
    print(f"Writing to: {out_path}\n")

    write_lock = threading.Lock()

    def process_task(pair, repeat_idx):
        """Run ONE repeat of a pair, ALWAYS in the same ordering (A=resp_a,
        B=resp_b) -- never swapped. Any difference across repeat_idx values
        for the same pair_id is pure run-to-run noise, not position bias."""
        verdict, conf = call_judge(
            client, args.model, pair["question"], pair["resp_a"], pair["resp_b"],
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if verdict is None:
            return {
                "pair_id": pair["pair_id"], "split": pair["split"],
                "source": pair["source"], "repeat_idx": repeat_idx,
                "model": args.model, "parse_error": True,
            }
        return {
            "pair_id": pair["pair_id"], "split": pair["split"],
            "source": pair["source"], "repeat_idx": repeat_idx, "model": args.model,
            "verdict": verdict,          # 'A' or 'B', always in the SAME ordering
            "confidence": conf,
            "gt": pair["gt"],
            "correct": verdict == pair["gt"],
            "parse_error": False,
        }

    tasks = []
    for pair in pairs:
        for repeat_idx in range(args.n_repeats):
            if (pair["pair_id"], repeat_idx) not in done:
                tasks.append((pair, repeat_idx))

    with open(out_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_task, pair, ri): (pair, ri)
                for pair, ri in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="repeat-judgments"):
                rec = future.result()
                with write_lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
