#!/usr/bin/env python3
"""
judge_judgebench.py
--------------------
Judge JudgeBench pairs with any model via OpenRouter, using VERBALIZED confidence,
running each pair in BOTH orderings (swap-and-average) so we can measure both
calibration (confidence vs. accuracy) and position bias (verdict flips).

One row of output = one judgment (so each pair produces TWO rows: order 1 and order 2).

Resume-safe: skips any (pair_id, order) already recorded in the output file.

Usage:
    conda activate judge
    export OPENROUTER_API_KEY="sk-or-..."

    # quick sanity test: 5 pairs, one model
    python judge_judgebench.py --model openai/gpt-4o-mini --split gpt --limit 5

    # full run for one model
    python judge_judgebench.py --model openai/gpt-4o-mini --split gpt

Output file is named automatically from the model + split, e.g.
    judgements_judgebench__openai_gpt-4o-mini__gpt.jsonl
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = """You are an impartial judge evaluating which of two responses better answers a question.

[Question]
{question}

[Response A]
{resp_a}

[Response B]
{resp_b}

Decide which response is better and how confident you are.

IMPORTANT: Your very first two lines MUST be exactly this format, before any explanation:
Verdict: <A or B>
Confidence: <number from 50 to 100>

The confidence is how likely it is that your chosen response is genuinely the better one \
(50 = a pure guess, 100 = certain). Do not default to a habitual round number like 85 or \
90; let genuinely close calls score lower (60s-70s) and clear-cut cases score higher (90s) \
based on the specific content above.

After those two lines, you may briefly explain your reasoning if you wish."""


def parse_response(text):
    """Pull the verdict letter and confidence number out of the model's reply.
    Returns (verdict, confidence_0_to_1) or (None, None) if it can't be parsed.
    A confidence outside [50, 100] (e.g. a negative, 0, or >100) is treated as
    MALFORMED and rejected, rather than being clamped into a plausible-looking value."""
    verdict_match = re.search(r"Verdict:\s*([AB])", text, re.IGNORECASE)
    conf_match = re.search(r"Confidence:\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if not verdict_match or not conf_match:
        return None, None
    verdict = verdict_match.group(1).upper()
    conf = float(conf_match.group(1))
    if conf < 50 or conf > 100:  # malformed / off-spec -> reject, don't invent data
        return None, None
    return verdict, conf / 100.0


def ground_truth_winner(label):
    """'A>B' -> 'A', 'B>A' -> 'B'. Returns None if the label isn't a clean preference."""
    if not label or ">" not in label:
        return None
    side = label.split(">")[0].strip().upper()
    return side if side in ("A", "B") else None


def load_pairs(split):
    """Load JudgeBench and return a flat list of pairs from the requested split(s)."""
    ds = load_dataset("ScalerLab/JudgeBench")
    splits = ["gpt", "claude"] if split == "both" else [split]
    pairs = []
    for s in splits:
        for row in ds[s]:
            gt = ground_truth_winner(row["label"])
            if gt is None:
                continue  # skip anything that isn't a clean A>B / B>A pair
            pairs.append(
                {
                    "pair_id": row["pair_id"],
                    "split": s,
                    "source": row.get("source", ""),
                    "question": row["question"],
                    "resp_a": row["response_A"],
                    "resp_b": row["response_B"],
                    "gt": gt,
                }
            )
    return pairs


def already_done(out_path):
    """Return the set of (pair_id, order) already in the output file, for resume."""
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
                done.add((rec["pair_id"], rec["order"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def call_judge(client, model, question, shown_a, shown_b, retries=3, max_tokens=150,
               reasoning_effort=None):
    """Call the model once and parse. Retries on parse/API failure. Returns (verdict, conf).

    reasoning_effort: if set (e.g. 'low'), passed through to the provider as a
    request to use LESS internal "thinking" before answering. This matters for
    reasoning models (Gemini 2.5 Pro, DeepSeek R1, Claude Sonnet 5, etc.) that
    can otherwise burn most or all of max_tokens on invisible reasoning and
    never reach the visible verdict -- raising max_tokens alone just means
    paying for more of that invisible reasoning, not fixing the truncation."""
    prompt = PROMPT_TEMPLATE.format(question=question, resp_a=shown_a, resp_b=shown_b)
    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning"] = {"effort": reasoning_effort}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None,
            )
            text = resp.choices[0].message.content or ""
            verdict, conf = parse_response(text)
            if verdict is not None:
                return verdict, conf
            else:
                sys.stderr.write(f"  [DEBUG] Could not parse. Raw output was:\n---\n{text}\n---\n")
        except Exception as e:  # noqa: BLE001 - network/API errors, just retry
            import traceback
            sys.stderr.write(f"  API error (attempt {attempt+1}):\n")
            traceback.print_exc(file=sys.stderr)
        time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter model string, e.g. openai/gpt-4o-mini")
    ap.add_argument("--split", default="gpt", choices=["gpt", "claude", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many PAIRS (mainly for a quick test run).")
    ap.add_argument("--max-tokens", type=int, default=150,
                    help="Max tokens per response. Standard instruct models are fine "
                         "at the default 150 (the verdict comes first). REASONING "
                         "models (e.g. gemini-2.5-pro, deepseek-r1) spend a large "
                         "hidden thinking budget BEFORE emitting any visible text, "
                         "so they need much more -- try 2000-4000.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of concurrent API calls in flight. Since each call "
                         "is mostly waiting on the network/model, running several "
                         "at once cuts wall-clock time roughly proportionally. "
                         "8 is a safe default; raise it if you're not hitting rate "
                         "limits, lower it (e.g. to 2-3) if you start seeing 429 "
                         "rate-limit errors.")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="For REASONING models only (Gemini 2.5 Pro, DeepSeek R1, "
                         "Claude Sonnet 5, etc.): caps how much internal thinking "
                         "the model does before answering. Use 'low' to keep cost "
                         "and truncation risk down for a simple judging task. "
                         "Leave unset for standard non-reasoning models.")
    ap.add_argument("--out", default=None, help="Override the auto output filename.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=...).")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_path = args.out or f"judgements_judgebench__{safe_model}__{args.split}.jsonl"

    pairs = load_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    done = already_done(out_path)
    print(f"Model: {args.model}  |  max_tokens: {args.max_tokens}  |  reasoning_effort: {args.reasoning_effort}")
    print(f"Split: {args.split}  |  pairs: {len(pairs)}  |  already done: "
          f"{len(done)//2} pairs")
    print(f"Writing to: {out_path}\n")

    with open(out_path, "a", encoding="utf-8") as fout:
        write_lock = threading.Lock()

        def process_task(pair, order):
            """Run one (pair, order) judgment and return the record to write."""
            if order == 1:
                shown_a, shown_b = pair["resp_a"], pair["resp_b"]
            else:
                shown_a, shown_b = pair["resp_b"], pair["resp_a"]

            verdict, conf = call_judge(
                client, args.model, pair["question"], shown_a, shown_b,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )

            if verdict is None:
                return {
                    "pair_id": pair["pair_id"], "split": pair["split"],
                    "source": pair["source"], "order": order, "model": args.model,
                    "parse_error": True,
                }
            if order == 1:
                picked_original = verdict          # A->A, B->B
            else:
                picked_original = "B" if verdict == "A" else "A"  # swapped
            return {
                "pair_id": pair["pair_id"], "split": pair["split"],
                "source": pair["source"], "order": order, "model": args.model,
                "raw_verdict": verdict,             # letter it output (position)
                "picked_original": picked_original, # underlying response it chose
                "confidence": conf,                 # 0.5 - 1.0
                "gt": pair["gt"],                   # ground-truth winner
                "correct": picked_original == pair["gt"],
                "parse_error": False,
            }

        # Build the list of work still to do (skipping anything already resumed)
        tasks = []
        for pair in pairs:
            for order in (1, 2):
                if (pair["pair_id"], order) not in done:
                    tasks.append((pair, order))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_task, pair, order): (pair, order)
                for pair, order in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="judgments"):
                rec = future.result()
                with write_lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
