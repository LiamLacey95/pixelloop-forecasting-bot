# What this bot does differently, and why

Forked from `Metaculus/metac-bot-template`. `template_bot.py` is Metaculus's original, kept
unmodified as the baseline being measured against.

## The scoring rule, since it drives everything

Summer FutureEval and MiniBench both use **spot peer score**: `100 x (your log score - the mean
log score of every other bot)`, on your last forecast before the question closes. Unforecast
questions score **0**, not negative.

Two consequences that matter more than any prompt:

1. **The crowd term is outside your control**, so maximising expected peer score is exactly
   maximising your own expected log score. The rule is essentially proper — predict your true
   belief. It is *not* a reason to deviate from the consensus for its own sake.
2. **Prize money is split in proportion to (sum of peer scores) squared.** Because
   `E[S²] = E[S]² + Var(S)`, variance is paid for at fixed expected score, and *coverage*
   compounds: every question you miss is a zero dragging a quadratic. Never missing a question is
   worth more than any clever aggregation.

## Changes, in descending order of expected value

### 1. Cross-model ensemble

The template draws N samples from **one** model at temperature. Those samples share a bias, so
averaging them cancels decode noise and nothing else — the ensemble is one opinion repeated.

This rotates samples round-robin across three model families (`ENSEMBLE_MODELS`). Errors are then
partly independent, which is the only condition under which extremising the aggregate is a
correction rather than bias amplification. With a single model configured, extremisation is
automatically disabled.

### 2. Logit-mean aggregation with a tail cap

Template: median of the probabilities. Here: mean in logit space, optional extremisation, clipped
to `[0.02, 0.98]`.

The cap is the one calibration intervention that separated winners in the Fall 2025 survey. It
insures against a catastrophic misread without touching the mid-range.

### 3. Disagreement triggers research, it does not blend

Sample spread is real information, but the useful response is to go find the missing fact rather
than to average the ignorance. `needs_more_research()` exposes the signal; high-disagreement
questions are flagged in the logs.

## What I removed, and why — the useful part of this document

The first version of this bot shrank every binary forecast toward a prior of 0.35, by an amount
driven by sample spread. It was wrong in four separate ways, and all four are worth stating
because they are easy mistakes to make again:

1. **The prize rule taxes caution.** Prize is quadratic in summed peer score, so variance is paid
   for. A standing haircut buys insurance the payout formula actively penalises.
2. **It could never extremise.** The trust weight was capped below 1, so the layer pulled toward
   0.35 *even when all samples agreed perfectly*. That is a directional bet on the question set,
   not a calibration correction.
3. **The prior was wrong for the questions.** "Most 'will X happen' questions resolve No" is
   folklore; Metaculus's own MiniBench analysis found that the nothing-ever-happens prior baked
   into template bots actively hurt on auto-generated questions.
4. **The evidence assumed its conclusion.** The old test simulated a forecaster that was
   overconfident *by construction*, then reported that a shrinkage layer helped. That is not
   evidence about this bot; it is arithmetic about the assumption.

The honest version of the idea is **Platt scaling** — a slope and bias in logit space, *fitted*
from this bot's own resolved forecasts. Metaculus validated it on real AIB data as the best of
five calibration methods, and ships an implementation. It needs ~200 resolved forecasts to fit,
which this bot does not have yet. A fixed prior with a fixed trust weight is Platt scaling with
both parameters guessed, which is strictly worse than not doing it.

## What it costs to run, and the two expensive mistakes

Every question is one research call, two forecast samples and a parse. Measured on bracketed runs
against OpenRouter's credits endpoint — not estimated — that is about **$0.05 a question**, or
roughly 105 questions on the credit this bot has. The season has about 200, so coverage is the
binding constraint and the arithmetic below is not academic.

**The cost lever is the thinking budget, not `max_tokens`.** `gemini-3.5-flash` is $9.00/M output
against deepseek's $0.87, and it bills thinking at that output rate. One sample costs **$0.103**
with thinking unbounded and **$0.028** with `reasoning_effort="low"`, reaching the same conclusion
by the same route. Flash is still the right model — 232 peer points per pound against 150 for the
next best, `python model_value.py` — so the answer is to bound its thinking, not to move off it.

Two mistakes cost real money and both look like prudence:

1. **Capping `max_tokens` to save money.** The cap covers thinking *and* answer out of one budget,
   so a question that provoked a long deliberation had nothing left to write its answer with. The
   rationale stopped mid-sentence, the parser correctly reported no forecast, and the question
   scored zero — but the call still billed for every token it generated. A tight cap does not save
   money; it pays full price for a guaranteed zero. It is a runaway guard and nothing else.
2. **Choosing the parser on price alone.** Swapping it to `xiaomi/mimo-v2.5` for a fraction of a
   cent produced 293 parse failures in one pass, answering `<<REQUESTED TYPE WAS NOT FOUND IN
   TEXT>>` on text that plainly held a forecast. Parsing is the right place to spend nothing, but
   "cheap" is not the same as "any cheap model". `deepseek-v4-pro` is a poor forecaster (-0.3 live
   peer) and a reliable extractor, which is exactly the job.

Both were diagnosed wrongly at first — blamed on a numeric elicitation change that had landed in
the same window and was reverted for nothing. If a run breaks, diff it against the last one that
passed before reasoning about which change is guilty.

## Testing

```bash
python test_calibration.py                     # property tests, no network, no cost
python smoke_one.py {binary|numeric|multiple_choice}   # one real question, ~$0.03
```

`smoke_one.py` exists because a full pass over the sandbox tournament costs ~$0.87 and catches the
same class of bug — a truncated rationale, a parser that cannot extract a schema, a dead model
slug — as one question does for $0.03. Verify a change with it and spend the full pass only on the
last check before enabling the schedule.

Property tests, no dependencies, no network. They check the things that must hold by
construction: unanimous samples survive untouched, aggregation is symmetric under complement
(a fixed prior breaks this, which is how the old bug would have been caught), extremisation moves
away from 0.5 in both directions, and clipping bounds hold.

The accompanying comparison reports logit-mean against the template's median across calibration
regimes rather than picking the flattering one. The two are close — this layer is not where the
points are, and the file says so.

## What the bot-maker survey says, and what we can act on

Metaculus surveyed 39 developers from the Fall 2025 tournament and merged the answers with the
final leaderboard ([notebook](https://www.metaculus.com/notebooks/43337/fall-2025-futureeval-survey/)).
Its headline is that model choice is now table stakes and **scaffolding is the differentiator** —
within the GPT-5 family alone, the best and worst scaffolds are ~27 peer points per question
apart, which is larger than three generations of model progress.

Its priority list for the next round, scored against this bot:

| # | Priority | Here |
|---|---|---|
| 1 | Two or three research sources, not one | **No** — one. Strongest predictor in the dataset (r = 0.42, p = 0.006); winners averaged 1.75 sources, non-winners 1.00 |
| 2 | Cap predictions at a max/min | **Yes** — `[0.02, 0.98]`. Strongest within-winners differentiator (r = +0.48) |
| 3 | Calculate base rates explicitly | **Yes** — added to the binary prompt (r = +0.38; 40% of top-15 winners, 7% of the bottom half, 0% of non-winners) |
| 4 | Similar past questions as a prior | **Closed to us** — see below |
| 5 | 10–30 LLM calls per question | **No** — about 5. Median winner made 28, median non-winner 7 |
| 6 | Manual-review loop for outliers | Partly — `disagreement()` logs, nothing reviews |
| 7 | Aggregate across multiple models | **No** — two samples of one model, because one model is measurably the best and the budget binds |

**Item 4 is not a gap, it is a wall.** A bot-account token receives `resolution: null` on resolved
questions and an empty `aggregations.recency_weighted.latest` on every question — list endpoint and
detail endpoint, with and without `with_cp=true`. Search works and returns genuinely adjacent
questions; only the numbers are stripped. Metaculus is deliberately stopping bots anchoring to the
crowd, which is the point of a bot benchmark, so this is a restriction to respect rather than route
around. It is also why `bakeoff.py` cannot score candidate models against resolved questions.

**One number from the survey is worth keeping in view when spending:** the median prize-winner
spent about **$0.90 per question** and the top-15 winners about **$1.40**, while *every* non-winner
spent under $1.00 and half spent under $0.10. This bot is at ~$0.17. That correlation is measured
among bots that already had full coverage, though — on a fixed budget, total peer score is
`budget × (peer per pound)`, and the cheap pipeline wins that comparison by roughly 5x. Spend more
per question only once coverage of the remaining season is already paid for.

## What is deliberately not done yet

- **Numeric and multiple-choice** fall through to the template untouched. This is where bots bleed
  the most points against human forecasters, so it is the highest-value remaining work. Proven
  open-source pipelines exist and porting one is a change that should land on its own.
- **Fitted Platt scaling**, once there are enough resolved forecasts of this bot's own.
- **A second research source.** Breadth of search correlates with score more than any single
  provider does.
