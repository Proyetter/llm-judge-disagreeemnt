# Collection procedure and scheduling

This is the record the paper's limitations section depends on. Accuracy matters
more than completeness here. Write "unknown" wherever a detail was not logged.

## Design

Each of the seven judges evaluated each response pair six times: three under AB
ordering (original response A displayed first, recorded as `order: 1`) and
three under BA (`order: 2`). Scheduled total: 7 judges x 620 pairs x 6 calls =
26,040 judgments.

## Splits

| Split | Response generator | Pairs |
| --- | --- | --- |
| `claude` | Claude 3.5 Sonnet | 270 |
| `gpt` | GPT-4o | 350 |

The splits were collected and analyzed separately because their response
sources and item composition differ.

## Scheduling and sessions

The paper flags this explicitly as a limitation, so state it precisely.

- **Claude split:** AB and BA calls occurred within the same collection period.
  Order-associated differences are not confounded with session.
- **GPT split:** AB and BA orderings were collected in separate sessions.
  Order-associated and session-associated differences cannot be separated.

> TODO: fill in the dates below. To recover them from your own machine, run
> `ls -lT ~/Desktop/LLM_AS_JUDGE/factorial_judgebench__*.jsonl` on macOS, or
> check your shell history for when each command was issued.

| Split | Ordering | Collection date or dates |
| --- | --- | --- |
| claude | AB and BA | TODO |
| gpt | AB | TODO |
| gpt | BA | TODO |

## Commands issued

> TODO: list the actual command line used per judge, including any per-model
> `--max-tokens` or `--reasoning-effort` flags. The general shape was:
>
>     python ../scripts/judge_judgebench_factorial.py \
>         --model MODEL_ID --split SPLIT --n-reps 3
>
> Recovering these from shell history is worth the five minutes. `history | grep
> factorial` should surface most of them.

## Eligibility and exclusions

An item is eligible only if it has three valid judgments under each ordering,
that is six valid judgments in total. Records with `parse_error: true` remain
in the data files for transparency and are excluded from analysis.

Final parse-failure rates were below 2 percent, with a maximum AB versus BA
difference of 0.6 percentage points. `audit_factorial.py` recomputes these,
broken down by ordering, which is the breakdown that matters: if one ordering
fails more often than the other, dropping invalid rows removes data
non-randomly and can bias the order effect itself.

Per-judge eligible sample sizes are the N column of Table II.

## Data migration note

> TODO: keep this section only if it applies to the final data files, otherwise
> delete it. Earlier single-ordering repeat data collected by
> `judge_judgebench_repeat.py` was converted into the factorial record format
> to avoid re-paying for calls already made. If any records in `data/`
> originated from that migration, say so here and name which judge and split.
> If the final data was collected fresh under the factorial design, delete this
> section.

## Cost

> TODO: optional. Approximate total API spend, if you want it on the record.
