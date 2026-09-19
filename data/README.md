# Saved judgment data

Put the factorial judgment files in this directory. One JSON object per line
(JSONL). One object is one API call.

    data/factorial_judgebench__{model}__{split}.jsonl

`{model}` is the OpenRouter identifier with `/` and `:` replaced by `_`.
`{split}` is `gpt` or `claude`. For example:

    data/factorial_judgebench__openai_gpt-4o-mini__gpt.jsonl
    data/factorial_judgebench__anthropic_claude-sonnet-5__claude.jsonl

Fourteen files in total: seven judges times two splits. The analysis scripts
glob this pattern in the working directory, so the filenames must be preserved
exactly. Model identity and split are parsed from the filename as well as read
from the records.

## Field dictionary

Written by `process_task` in `scripts/judge_judgebench_factorial.py`.

| Field | Type | Meaning |
| --- | --- | --- |
| `pair_id` | string | JudgeBench item identifier. Joins back to the benchmark. |
| `split` | string | `gpt` or `claude` |
| `source` | string | JudgeBench domain tag, for example `mmlu-pro-law` |
| `order` | int | `1` is AB, original response A displayed first. `2` is BA, swapped. |
| `rep_idx` | int | Repetition index within the ordering, 0 to 2 |
| `model` | string | OpenRouter identifier of the judge |
| `parse_error` | bool | `true` if no valid verdict and in-range confidence could be extracted after three attempts |
| `raw_verdict` | string | The letter the judge emitted, as displayed: `A` or `B` |
| `picked_original` | string | The verdict mapped back to original response identity. Under `order: 2` this is the opposite letter to `raw_verdict`. |
| `confidence` | float | Verbalized confidence divided by 100, so in [0.5, 1.0] |
| `gt` | string | Ground-truth better response, `A` or `B`, in original identity |
| `correct` | bool | `picked_original == gt` |

On a record with `parse_error: true`, only `pair_id`, `split`, `source`,
`order`, `rep_idx`, `model`, and `parse_error` are present. The remaining
fields are absent, not null.

The mapping from displayed letter to original identity happens at collection
time. Analysis code reads `picked_original` and never needs to know the
ordering convention.

## Vote aggregation and ties

Let `x` be the number of AB calls selecting original A (0 to 3) and `y` the
same for BA. Pooling all six votes: A wins when `x + y > 3`, B wins when
`x + y < 3`, and `x + y = 3` is a tie. Ties are excluded from the correctness
AUROC in Table IV.

## Benchmark inputs are not included

These records store no question text and no response text, only `pair_id` and
`source`. Nothing from JudgeBench is redistributed here. To re-run collection,
the benchmark is fetched at runtime:

```python
from datasets import load_dataset
ds = load_dataset("ScalerLab/JudgeBench")
```

The analysis scripts need only the files in this directory.

## Before pushing

Check the total size. GitHub warns above roughly 50 MB per file and refuses
anything over 100 MB.

    du -sh data/

These records are compact, no raw model output is stored, so fourteen files
should come in well under any limit.
