# Judge prompt

This is the exact prompt used for every judgment in the study, sent as a single
user message. No system message was used.

It is the `PROMPT_TEMPLATE` constant in `scripts/judge_judgebench.py`, which
`judge_judgebench_factorial.py` uses through the shared `call_judge` function.
The text below was extracted from that constant programmatically, with Python's
backslash line continuations resolved, so it is exactly what the models
received.

## Template

```
You are an impartial judge evaluating which of two responses better answers a question.

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

The confidence is how likely it is that your chosen response is genuinely the better one (50 = a pure guess, 100 = certain). Do not default to a habitual round number like 85 or 90; let genuinely close calls score lower (60s-70s) and clear-cut cases score higher (90s) based on the specific content above.

After those two lines, you may briefly explain your reasoning if you wish.
```

## Field substitution

| Placeholder | Value |
| --- | --- |
| `{question}` | The JudgeBench question text, verbatim. Some questions carry their own embedded instructions; these were not modified. |
| `{resp_a}` | The response displayed first. Under AB (`order: 1`) this is original response A. Under BA (`order: 2`) this is original response B. |
| `{resp_b}` | The response displayed second. Under AB this is original response B. Under BA this is original response A. |

Displayed verdicts are mapped back to original response identities at
collection time, not analysis time. Under BA, a displayed verdict of "A" is
recorded as `picked_original: "B"`. See `data/README.md`.

## Confidence elicitation

The prompt requires the verdict and confidence as the first two response lines,
before any explanation, and permits a brief explanation afterward. The
instruction not to default to a habitual round number such as 85 or 90 was
added after a pilot run in which a judge produced only two distinct confidence
values across thirty pairs. All data reported in the paper was collected with
the prompt shown above. Pilot data collected under the earlier, looser prompt
was discarded and is not in this repository.

Confidence elicitation follows K. Tian et al., "Just ask for calibration,"
EMNLP 2023.

## Parsing rules

From `parse_response` in `scripts/judge_judgebench.py`:

- The verdict is matched case-insensitively as `Verdict:` followed by `A` or `B`.
- The confidence is matched as `Confidence:` followed by a number, which may be
  negative or fractional.
- A confidence outside the range 50 to 100 is rejected as malformed rather than
  clamped to the boundary. This matters: at least one judge emitted negative
  confidence values, and clamping would have invented data.
- A reply missing either field, or carrying an out-of-range confidence, counts
  as a failed attempt and triggers a retry.
- Accepted confidences are divided by 100, so stored values lie in [0.5, 1.0].
