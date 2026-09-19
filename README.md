# llm-judge-disagreeemnt

Decomposing Disagreement in LLM Judges: Order Sensitivity and Stochasticity

Marcus Guo
Woodside Priory High School

Research materials for a study of response-order sensitivity, repeated-run variability, and self-reported confidence in LLM-based evaluation.

Manuscript: TODO: Add the paper's relative file path or public link.

Release preparation: Entries marked TODO need to be completed from the final code, configuration files, and experiment records. Reproduction commands and file locations are pending verification.

Overview

An LLM judge compares two candidate responses and selects one. Its verdict may change when the responses are swapped. It may also change across repeated calls with the same presentation order.

This study examines how much cross-order disagreement is accounted for by variability already present within a fixed ordering. Each response pair is evaluated three times in each order, producing six judgments per judge and item.

The paper focuses on three questions:

How does cross-order disagreement divide into within-order stochasticity and systematic cross-order differences?

How do observed instability patterns vary across judges?

Does confidence from the first evaluation identify items that show instability across the six-call record?

Experimental setup

The study uses both response-generator splits of JudgeBench (Tan et al., 2025). Each item contains a question, two candidate responses, and a ground-truth label identifying the correct response.

Split

Response generator

Response pairs

Claude

Claude 3.5 Sonnet

270

GPT

GPT-4o

350

The splits are analyzed separately. The design contains 26,040 scheduled judgments: seven judges, 620 response pairs, and six judgments per pair. Retry attempts are additional requests associated with those scheduled judgments.

Judge models

The identifiers below are those reported in the study. Eligible counts refer to items with complete valid repeated-order records.

Developer

Model identifier

Eligible Claude items

Eligible GPT items

Anthropic

anthropic/claude-3-haiku

269

350

Anthropic

anthropic/claude-sonnet-5

260

343

DeepSeek

deepseek/deepseek-chat

270

350

Google

google/gemini-2.5-flash

269

348

Meta

meta-llama/llama-3.1-8b-instruct

269

345

Meta

meta-llama/llama-3.3-70b-instruct

268

350

OpenAI

openai/gpt-4o-mini

270

350

Presentation order and verdict mapping

In AB, original response A appears first. In BA, original response B appears first. Each ordering receives three judgments.

Displayed verdicts are mapped back to the original response identities before analysis. For example, a displayed verdict of A under BA selects original response B. All comparisons and vote counts below use original response identities.

Collection and eligibility

The judge prompt requests a verdict and a confidence value from 50 to 100 as its first two output lines. It also discourages habitual confidence values such as 85 or 90. Calls use temperature 0, and Claude Sonnet 5 uses low reasoning effort.

Each scheduled judgment permits up to three attempts in total. API exceptions and parse failures trigger retries when attempts remain, with exponential backoff beginning at one second. After three unsuccessful attempts, the record is marked parse_error: true.

An item is eligible for a judge's six-call analysis when it has three valid judgments in each ordering. If either ordering lacks the required valid judgments, that judge–item record is excluded. Missing judgments are not imputed. Eligibility is evaluated separately for each judge.

TODO — collection record: Add the exact prompt, API routing service, collection dates, per-model token limits and reasoning settings, and request schedule. The manuscript states that most models used a 150-token limit; the configuration files must identify every model's actual limit.

Analysis

Disagreement decomposition

For a fixed judge and response pair, let p be the probability of selecting original A under AB and q the probability of selecting original A under BA. The probability interpretation assumes independent calls with stable choice probabilities within each ordering.

D_cross  = p(1 - q) + (1 - p)q
D_within = p(1 - p) + q(1 - q)
S        = (p - q)^2

D_cross = D_within + S

D_cross measures disagreement between opposite orderings. D_within averages the disagreement probabilities between two calls within the same ordering. S measures the squared difference between the two order-conditioned choice probabilities.

Let x and y count selections of original A among the three AB and three BA judgments. The sample estimates are:

D_cross_hat  = [x(3 - y) + (3 - x)y] / 9
D_within_hat = [x(3 - x) + y(3 - y)] / 6
S_hat        = D_cross_hat - D_within_hat

The estimates use all nine cross-order comparisons and the three distinct pairs of calls within each ordering. Under the stated assumptions, S_hat is unbiased. Individual estimates can be negative because of sampling variation.

Model-level results average the item-level estimates over eligible items within each split. NoiseShare is mean within-order disagreement divided by mean cross-order disagreement. It is undefined when the denominator is zero.

Observed instability categories

A majority verdict requires at least two of three votes within an ordering. Within-order instability occurs when at least one ordering produces mixed judgments. Majority reversal occurs when the AB and BA majorities select different original responses.

Category

Within-order instability

Majority reversal

Stable

No

No

Noise-only

Yes

No

Position-only

No

Yes

Tangled

Yes

Yes

These categories describe the six observed judgments. Three repetitions per ordering provide limited information about rare disagreement and underlying causes.

The analysis also compares the proportion of noisy items among reversing and non-reversing items. Their difference is reported in percentage points.

Confidence and pooled decisions

Confidence comes from the first AB evaluation. The four outcomes are majority reversal, within-order instability, either type of instability, and pooled-label correctness.

Pooling the six mapped votes gives original A the final label when x + y > 3, original B when x + y < 3, and a tie when x + y = 3.

Predictive ranking is measured using the area under the receiver operating characteristic curve (AUROC). A value of 0.5 indicates chance-level ranking, and 1.0 indicates perfect ranking. AUROC is undefined when the evaluated sample contains only one outcome class.

TODO — confidence implementation: Record the exact score transformation used for each outcome. Confirm whether instability uses negative confidence or 100 - confidence, and whether correctness uses confidence directly. State how pooled ties enter correctness AUROC and report the sample size for each judge, split, and target.

The first call contributes to the six-call outcomes. The analysis therefore measures association between initially available confidence and observed six-call behavior.

Uncertainty

The manuscript reports 95% percentile bootstrap intervals using 1,500–2,000 resamples, depending on the analysis. Response pairs are sampled with replacement, retaining all six judgments for each sampled pair. Point estimates use the full eligible dataset.

For zero observed within-order instability, the approximate rule-of-three upper bound is 3 / N, where N is the eligible item count. This bound concerns the rate of items showing within-order instability under the six-call protocol.

TODO — bootstrap record: Identify the resample count, random seed, and handling of undefined statistics for each retained analysis.

Reported findings

All seven judges have positive estimated systematic cross-order components on both splits, with reported 95% confidence intervals above zero. GPT-split estimates may also include collection-session effects.

Gemini 2.5 Flash shows no observed within-order disagreement while retaining a systematic cross-order estimate of approximately 0.28–0.30. Claude Sonnet 5 has much lower overall disagreement, with within-order variation accounting for 42–49% of the disagreement that remains.

For confidence-based ranking of majority reversal, Claude Sonnet 5 and Gemini 2.5 Flash have 95% AUROC intervals entirely above 0.5 on both splits. Evidence for the remaining judges varies across splits. Full estimates are reported in the manuscript's results tables.

Reproducing the results

Reproduction should use the saved judgment records associated with the paper. Keep collection and analysis as separate steps so that regenerating tables does not launch new API requests.

Materials and environment

Resource

Location or specification



Limitations

The study uses one benchmark, one prompt, and three repetitions per ordering. Item-level estimates have limited precision, and rare disagreements may remain unobserved. Confidence values are coarse for several judges, and inference settings include a model-specific low-reasoning setting for Sonnet 5.

GPT-split AB and BA judgments were collected in separate sessions. Response-order effects and session effects cannot be separated in that split. Claude-split calls occurred within the same collection period; the exact schedule requires documentation in the collection record.

Results depend on the parsing, retry, and eligibility procedures. Confidence-based filtering or escalation policies require separate validation on held-out data.

Citation and attribution

Project manuscript: Marcus Guo. Decomposing Disagreement in LLM Judges: Order Sensitivity and Stochasticity.

Benchmark: S. Tan et al. JudgeBench: A Benchmark for Evaluating LLM-Based Judges. International Conference on Learning Representations, 2025.

TODO: Add the manuscript's public link and the repository release identifier. Document the licenses and attribution requirements for the released code and data.
