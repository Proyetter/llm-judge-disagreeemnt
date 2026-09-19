# Decomposing Disagreement in LLM Judges: Order Sensitivity and Stochasticity

Code, prompt, saved judgments, and analysis outputs for the paper *Decomposing
Disagreement in LLM Judges: Order Sensitivity and Stochasticity* (Marcus Guo,
Woodside Priory High School).

Someone who clones this repository can regenerate every table in the paper from
the saved judgment files, without making any paid API calls.

> TODO: add the paper link or arXiv ID here once available.

## What the study does

Seven LLM judges evaluate every response pair in both JudgeBench
response-generator splits three times under ordering AB (original response A
displayed first) and three times under ordering BA. That repeated-order design
supports an exact decomposition of cross-order disagreement:

    D_cross = D_within + S,    S = (p - q)^2

where `p` and `q` are the probabilities of selecting original response A under
AB and under BA, `D_cross` is the probability that an AB judgment and a BA
judgment disagree, `D_within` is the average probability that two judgments of
the same ordering disagree, and `S` is the systematic order component.

## Layout

    prompts/     the exact prompt template used for every judgment
    configs/     model identifiers, generation settings, retry policy, collection scheduling
    data/        saved judgment records, one JSON object per API call, plus a field dictionary
    scripts/     collection, audit, and analysis scripts as actually run
    results/     captured output of each analysis script, as reported in the paper
    run_analysis.sh   regenerates everything in results/ from data/

## Setup

    conda create -n judge python=3.12
    conda activate judge
    pip install -r requirements.txt

## Reproducing the tables

No API key and no paid calls are required for this step.

    ./run_analysis.sh

That writes one text file per analysis into `results/`. The files already
committed in `results/` are the outputs reported in the paper, so a successful
reproduction should match them. Point estimates should match exactly.
Bootstrap confidence interval endpoints will move slightly from run to run,
since the resampling is not seeded.

The analysis scripts locate their input by globbing `factorial_judgebench__*.jsonl`
in the current working directory, so each is run from inside `data/`. To run one
by hand:

    cd data
    python ../scripts/analyze_decomposition.py

| Paper table | Script | Output in `results/` |
| --- | --- | --- |
| Table II, decomposition of cross-order disagreement | `analyze_decomposition.py` | `table2_decomposition.txt` |
| Table III, instability categories | `analyze_pair_classes.py` | `table3_instability_categories.txt` |
| Table III, conditional noise rates | `analyze_conditional_noise.py` | `table3_conditional_noise.txt` |
| Table IV, first-call confidence AUROC | `analyze_confidence_v2.py` | `table4_confidence_auroc.txt` |
| Methods, eligibility and parse-failure rates | `audit_factorial.py` | `audit.txt` |

Table I is the model list and is documented in `configs/models.yaml`.

Table IV reports the single-shot columns from `analyze_confidence_v2.py`
(`SS-position`, `SS-noise`, `SS-any`, `SS-correct`), which use the confidence
value from the first AB call only. The six-call columns in the same output are
a scientific upper bound and are not reported in the paper.

## Regenerating the judgments (optional, costs money)

Collection used the OpenRouter API. This is only needed to collect new data.

    export OPENROUTER_API_KEY="sk-or-..."
    cd data
    python ../scripts/judge_judgebench_factorial.py \
        --model openai/gpt-4o-mini --split gpt --n-reps 3

Collection is resume safe. Re-running skips any `(pair_id, order, rep_idx)`
already present in the output file. See `configs/models.yaml` for the per-model
flags, including the `--reasoning-effort low` setting required by Claude
Sonnet 5.

## Benchmark data

The questions, candidate responses, and ground-truth labels come from
JudgeBench, loaded from the Hugging Face dataset `ScalerLab/JudgeBench`. This
repository does not redistribute any of that text. The saved judgment records
reference benchmark items by `pair_id` only, so the benchmark must be
downloaded separately to re-run collection. The analysis scripts do not need
it.

S. Tan et al., "JudgeBench: A benchmark for evaluating LLM-based judges,"
International Conference on Learning Representations, 2025.

## Citation

> TODO: add the BibTeX entry for the paper here.

## License

> TODO: pick a license, add a LICENSE file at the repository root, and name it
> here. MIT is the usual default for research code.
