"""Forecasting bot for the Metaculus AI Benchmark.

Built on Metaculus's template. Changes from it, in descending order of how much they matter:

1. **Cross-model ensemble.** The template samples N forecasts from one model at temperature. Those
   samples share a bias, so averaging them cancels decode noise and nothing else. This rotates
   samples across model families, where errors are partly independent - which is also the only
   condition under which extremising the aggregate is defensible rather than bias amplification.

2. **Logit-mean aggregation with a tail cap** instead of median-of-probabilities. See
   calibration.py; the reasoning is in BOT.md.

3. **Disagreement is logged**, not blended. Scattered samples mean a missing fact, and the
   intended response is another research pass rather than averaging the ignorance.

Numeric and multiple-choice fall through to the template.

A 13-percentile elicitation with tail anchors was tried, blamed for a run in which every numeric
question failed with `ValidationError: NumericDistribution`, and reverted. That diagnosis was
wrong. Comparing the failing logs against the last passing run showed the percentile list had
completed fine there; what broke the run was two other changes made in the same window - a 1200
token cap that cut the rationale off before its answer, and a parser model that could not extract
a schema. Both are fixed above. The percentile change is worth re-trying now that it is not
carrying the blame for something else.
"""

import argparse
import asyncio
import itertools
import logging
from datetime import datetime
from typing import Literal

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    PredictionTypes,
)

from calibration import (  # noqa: E402
    EXTREMISE_CROSS_MODEL,
    AggregationConfig,
    aggregate,
    disagreement,
    needs_more_research,
)
from forecasting_tools.util.misc import clean_indents  # noqa: E402
from template_bot import SummerTemplateBot2026  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# READ THIS BEFORE CHANGING THE MODEL LIST.
#
# THERE ARE TWO METACULUS LEADERBOARDS AND ONLY ONE OF THEM PREDICTS ANYTHING HERE.
#
# 1. The MODEL leaderboard (metaculus.com/futureeval/leaderboard/) publishes a "skill score"
#    anchored so GPT-4o = 0. It is measured on a different question set, and as of 2026-07-26 its
#    newest entry is GPT-5.5 from 24 April.
# 2. The TOURNAMENT leaderboard, with the "advanced" toggle on, shows Metaculus's own benchmark
#    bots competing in THIS season on THESE questions. Divide Total Score by Questions and you
#    have the live peer score per question - which is the number prize money is computed from.
#
# The two disagree violently, and only the second one is the payout:
#
#   benchmark bot (live, 56-57 questions)      peer/q     model-leaderboard skill
#   metac-gemini-3-5-flash+asknews             +13.48     not listed at all
#   metac-gpt-5-5-instant+asknews              +11.24     13.84
#   metac-gpt-5-5-high+asknews                  +8.30     14.06
#   metac-claude-opus-4-7-high+asknews          +7.87     14.62
#   metac-claude-opus-4-8-high+asknews          +6.74     not listed
#   metac-gemini-3-1-pro-high+asknews           +7.40     19.84  <- top of the model leaderboard
#   metac-gemini-3-1-pro+asknews                +3.45     19.03
#   metac-deepseek-v4-pro-high+asknews          -0.30      8.66
#   metac-grok-4-3-high+asknews                 -0.39     11.35
#   metac-kimi-k2-6+asknews                     -1.74     11.39
#   metac-grok-4-20-multi-agent+asknews         -2.26     14.99
#
# gemini-3.1-pro-high tops the model leaderboard at 19.84 and earns +7.40 live. grok-4.20
# multi-agent sits fifth at 14.99 and is NEGATIVE live. Ranking by skill score picks a model that
# loses points. This file briefly did exactly that, on the reasoning that gemini-3.5-flash "had no
# measured score" - it has the best one in the tournament, on the only leaderboard that pays.
#
# So: read the tournament leaderboard in advanced mode. It is also the only source that covers
# recent models at all - claude-opus-4-8, claude-sonnet-4-6, claude-fable-5-high, minimax-m3 and
# glm-5-2 all appear there and on no model leaderboard. Kimi K3, Opus 5, GPT-5.6 and Grok 4.5 are
# not benchmarked anywhere yet, so switching to one would be a guess, and the guesses on this page
# have a poor record.
# A SOTA list - gpt-5.6-sol plus kimi-k3 - is ready on the `sota-models` branch and is the right
# thing to merge the day the budget stops binding. It is not merged, and the reason is measured
# rather than argued: one sample of Sol plus one research call cost $0.225, so a two-sample
# question lands near $0.31 against this list's $0.05. On the credit left that is 16 questions
# instead of ~100, with ~200 still to come before the season closes on 6 September.
#
# Prize share goes as the SQUARE of summed peer score, so the comparison is 16 x p against
# 100 x 13.48. Sol and K3 would have to score about 84 peer points per question to break even.
# The best bot in this tournament scores 21.6. Unmeasured upside cannot cover a gap that size.
ENSEMBLE_MODELS = [
    "openrouter/google/gemini-3.5-flash",
    "openrouter/google/gemini-3.5-flash",
]
# Slugs are verified against OpenRouter's public model list by check_models.py. Run it after any
# edit here: a wrong slug does not fail loudly, it silently removes one family from the ensemble
# and every forecast is quietly worse.

# Output cap. Flash bills thinking as output at the higher rate, so this looks like the obvious
# cost lever - and squeezing it was the single most expensive mistake in this build.
#
# At 1200 the reasoning was cut mid-sentence before it reached the probability line. The parser
# then correctly reported no forecast in the text, and the question scored nothing. A truncated
# call still bills for every token it generated, so a tight cap does not save money: it pays full
# price for a guaranteed zero. Two full sandbox passes and $1.06 went on this.
#
# The cap has to cover THINKING as well as the answer, because Flash spends both out of the same
# budget and bills them at the same rate. That is why 4000 still truncated: a question that
# provoked a long deliberation had nothing left over to write the answer with, so the length of
# the visible output depended on how hard the model happened to think.
#
# So this is a runaway guard and NOT the cost lever. Squeezing it does not save money, it buys
# truncated answers at full price.
#
# RAISED WITH THE EFFORT CHANGE, AND THE TWO MUST MOVE TOGETHER. 16000 was sized against "low".
# High effort spends far more of the shared budget on thinking, and the failure it produces is the
# expensive one already paid for once: thinking eats the allowance, the answer is cut before the
# probability line, the parser correctly finds no forecast, and the question scores zero at full
# price. Doubling the guard costs nothing when it is not reached - it bills actual tokens, not the
# cap - and prevents the one failure mode this change introduces.
MAX_FORECAST_TOKENS = 32000

# "low" because thinking bills as output and GPT-5.6 Sol charges $30/M for it, so an unbounded
# thinking budget on this list would cost more per question than the entire previous ensemble.
# Measured on the model it replaced, low effort cut cost 3.7x for an answer that reached the same
# conclusion by the same route.
#
# The counter-evidence is on the model leaderboard, where the high-effort row of the same model is
# consistently and substantially better - Grok 4.20 goes 6.13 to 14.99, GPT 5.1 goes 4.64 to
# 12.14, Kimi K2 goes 0.97 to 5.64. On a bigger budget this should be "high", and the fact that it
# is not is a budget decision rather than a forecasting one.
#
# 2026-07-26: tried "high" on the reasoning that the budget had stopped binding. MEASURED, AND
# REVERTED THE NEXT MORNING. Both halves of that reasoning were wrong.
#
# Cost: high effort billed **$0.44 to $1.26 a question**, averaging about $0.78 - not the ~$0.15
# guessed from the smoke run, where the one visible cost line turned out to be the Exa research
# call rather than the forecast. Low effort is measured at $0.161.
#
# Supply: the claim that questions were scarce came from one leg that found a single question in
# five hours. The very next leg forecast FOURTEEN overnight. A five-hour window says nothing about
# a release schedule, and one leg spent $10.88 of a $41.59 balance.
#
# Together those give ~$136 to finish the season against ~$30 left, so the credit dies in about
# forty questions and every question after it scores exactly zero. An unforecast question is not a
# cheaper forecast, it is a zero, which is why cost per question is a coverage decision and not a
# quality one. High effort is the right call the day sponsored credit lands and not before.
REASONING_EFFORT = "low"

# Second research index. Perplexity rather than another `:online` call, because OpenRouter's
# plugin is Exa behind every model and two Exa calls are not two sources. See run_research for
# why a source measured at -15.48 alone is still worth adding alongside a good one.
SECOND_RESEARCH_MODEL = "openrouter/perplexity/sonar"

# Omitted entirely rather than passed as None, so the request carries the provider's own default
# instead of an explicit null the provider may or may not interpret the same way.
EFFORT_KWARG = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}

# Zero-cost models, for plumbing checks only. Not competitive, and rate-limited to 429s under
# any sustained load. The route to frontier models at no cost is Metaculus's sponsored-credit
# programme, not OpenRouter's free tier.
FREE_ENSEMBLE_MODELS = [
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/openai/gpt-oss-20b:free",
]


def build_llms(free: bool) -> dict:
    """Pin every model role explicitly.

    Not optional: the framework's default researcher is `gpt-4o-search-preview-2025-03-11`, which
    OpenAI has deprecated. Left unset, every research call fails with a 404 and the bot produces
    no forecasts at all - it does not degrade, it stops. Found by smoke-testing against the
    bot-testing-area tournament, which is exactly what that sandbox is for.
    """
    if free:
        worker = "openrouter/google/gemma-4-31b-it:free"
        return {
            "default": GeneralLlm(model=worker, temperature=0.3, timeout=120, allowed_tries=2),
            "researcher": GeneralLlm(model=worker, temperature=0.1, timeout=120, allowed_tries=2),
            "summarizer": worker,
            "parser": worker,
        }
    return {
        # Numeric and multiple-choice run on the same model as binary, deliberately. Metaculus's
        # own analysis puts the human-versus-bot gap WIDEST on non-binary questions, and they are
        # roughly 40% of the set - so that is where a weak model bleeds the most peer score.
        # Downgrading them to save money would be cutting into the deepest wound.
        "default": GeneralLlm(
            model=ENSEMBLE_MODELS[0],
            temperature=0.3,
            timeout=120,
            allowed_tries=2,
            max_tokens=MAX_FORECAST_TOKENS,
            **EFFORT_KWARG,
        ),
        # THE RESEARCH LAYER IS WORTH MORE THAN THE MODEL. Metaculus runs the same forecasting
        # model behind several different research providers in this tournament, which isolates the
        # research variable exactly. Live peer score per question, all on deepseek-r1:
        #
        #   + exa-online       +1.27
        #   + asknews          -2.88
        #   + NO RESEARCH      -6.99
        #   + exa-answer       -8.11
        #   + sonar           -15.48   <- what this bot used
        #   + sonar-pro       -16.03
        #
        # Perplexity Sonar is 8.5 points per question WORSE THAN DOING NO RESEARCH, and 16.8
        # below the best option. Nothing else on the board - not the model, not the aggregation,
        # not the sample count - moves the score by anything like that much.
        #
        # `:online` is OpenRouter's Exa-backed web search plugin, which is the same search layer
        # behind the winning row. It bills per result on top of the model's own tokens.
        # Pinned to a cheap model deliberately, NOT to ENSEMBLE_MODELS[0]. Research is retrieval
        # and summarisation; the judgement happens in the forecast call. Running it on Sol at
        # $30/M output would roughly double the bill for the part of the pipeline where the
        # provider matters more than the model.
        "researcher": GeneralLlm(
            model="openrouter/google/gemini-3.5-flash:online",
            temperature=0.1,
            timeout=180,
            allowed_tries=2,
        ),
        # Parsing is extraction from text that a schema already constrains, so this is the right
        # place to spend nothing - but "cheap" is not the same as "any cheap model". Swapping this
        # to xiaomi/mimo-v2.5 to shave a fraction of a cent produced 293 parse failures in one
        # pass: it answered `<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>` on text that plainly held a
        # forecast. deepseek-v4-pro is a poor forecaster (-0.3 live peer) and a reliable
        # extractor, which is exactly the job. Zero parse failures over a full pass.
        "summarizer": "openrouter/deepseek/deepseek-v4-pro",
        "parser": "openrouter/deepseek/deepseek-v4-pro",
    }


class CalibratedBot(SummerTemplateBot2026):
    """Template bot with a cross-model ensemble and logit-mean aggregation."""

    # The template parses each forecast twice and compares, which turns 6 forecast calls into 12
    # parser calls and makes parsing - not forecasting - the dominant cost. Measured at
    # $0.20/question against a budget that allows $0.05. One parse, on a cheap model, with the
    # structured-output schema already constraining the result.
    _structure_output_validation_samples = 1

    def __init__(self, *args, ensemble_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = ensemble_models or []
        self._model_cycle = itertools.cycle(self.ensemble_models) if self.ensemble_models else None
        # Extremising only earns its place once samples span DIFFERENT families, where errors are
        # partly independent. Count distinct models, not list length: the list repeats a model to
        # weight it, so three entries of one model is still one opinion sampled three times, and
        # extremising it would amplify a shared bias rather than correct anything.
        distinct_families = {m.rsplit("/", 1)[0] for m in self.ensemble_models}
        self.aggregation = AggregationConfig(
            extremise=EXTREMISE_CROSS_MODEL if len(distinct_families) > 1 else 1.0
        )

    def _forecasting_llm(self):
        """Round-robin across the configured families; fall back to the framework's default."""
        if self._model_cycle is None:
            return self.get_llm("default", "llm")
        return GeneralLlm(
            model=next(self._model_cycle),
            temperature=0.3,
            timeout=90,
            allowed_tries=2,
            max_tokens=MAX_FORECAST_TOKENS,
            **EFFORT_KWARG,
        )

    async def run_research(self, question):
        """Two search indexes rather than one, because breadth is the best-evidenced lever there is.

        Metaculus's bot-maker survey found number of distinct research sources to be the strongest
        predictor of score in the whole dataset (r = 0.42, p = 0.006) - winners averaged 1.75
        sources, non-winners 1.00, and the note was explicit that "the takeaway isn't which source
        to pick, it's that one source is usually not enough".

        Two `:online` calls would not be breadth: OpenRouter's plugin is Exa behind every model.
        Perplexity queries a different index, and costs $0.005 against Exa's $0.055-0.13, so the
        second source adds well under 1% to the bill.

        The obvious objection is that `metac-deepseek-r1+sonar` scores -15.48 per question live,
        8.5 points WORSE than the same model with no research at all. That is a measurement of
        sonar as the SOLE source, where a terse and confident summary is all the forecaster sees.
        Here it is labelled, secondary, and read alongside Exa - and on the probe that motivated
        this, sonar returned the one concrete fact (a date and a location) that the Exa write-up
        buried. Being wrong about this costs half a cent a question; being right is the single
        strongest correlation in the survey.

        Either source may fail without taking the forecast down with it.
        """
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

        async def search(llm, label):
            try:
                return label, await llm.invoke(prompt)
            except Exception as e:  # noqa: BLE001 - one dead index must not lose the question
                logger.warning(f"{label} research failed for {question.page_url}: {e}")
                return label, ""

        async with self._concurrency_limiter:
            results = await asyncio.gather(
                search(self.get_llm("researcher", "llm"), "Exa web search"),
                search(
                    GeneralLlm(
                        model=SECOND_RESEARCH_MODEL, temperature=0.1, timeout=120, allowed_tries=2
                    ),
                    "Perplexity search (independent index)",
                ),
            )

        sections = [f"## {label}\n\n{text.strip()}" for label, text in results if text.strip()]
        if not sections:
            logger.warning(f"Both research sources returned nothing for {question.page_url}")
            return ""
        if len(sections) > 1:
            sections.insert(
                0,
                "Two independent search indexes were queried. Where they disagree on a fact, say "
                "so and weigh which is better sourced rather than averaging them.",
            )
        research = "\n\n".join(sections)
        logger.info(f"Found Research for URL {question.page_url}:\n{research}")
        return research

    async def _run_forecast_on_binary(self, question, research):
        """The template's prompt plus an explicit base-rate step.

        Metaculus surveyed 39 bot makers and merged the answers with the final leaderboard. Two
        findings are relevant here and both are cheap to act on:

        - "Explicitly calculate base rates in a rigorous way" is one of only three features that
          separate the TOP of the winners from the rest (r = +0.38, p = 0.032). 40% of the top 15
          winners do it against 7% of the bottom half - and ZERO of the ten non-winners.
        - The template's own prompt asks for the status quo outcome but never asks for a reference
          class or a frequency. Those are different questions: the status quo is what happens if
          nothing changes, a base rate is how often this KIND of thing happens.

        This costs nothing - it is the same call with a longer instruction - which makes it the
        best-evidenced change available while credit is the binding constraint.
        """
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) A REFERENCE CLASS and its BASE RATE. Name the class of events this question
                belongs to, state how often the Yes outcome has occurred in that class, and give
                the count you are reasoning from (for example "of the last 12 comparable votes,
                3 passed - roughly 25%"). If you cannot name a count, say so explicitly rather
                than inventing one, and explain what you are anchoring on instead.
            (c) The status quo outcome if nothing changed. This is NOT the same as (b): the base
                rate is how often this kind of thing happens, the status quo is what happens if
                nothing moves between now and resolution.
            (d) A brief description of a scenario that results in a No outcome.
            (e) A brief description of a scenario that results in a Yes outcome.
            (f) How far your final answer sits from the base rate in (b), and what specific
                evidence justifies the distance. If nothing does, stay near the base rate.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(self, question, prompt):
        # Same body as the template's, with the model chosen per call rather than fixed.
        from forecasting_tools import BinaryPrediction, ReasonedPrediction, structure_output

        llm = self._forecasting_llm()
        reasoning = await llm.invoke(prompt)
        parsed: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        # Clip only at the edges the API rejects; the real cap is applied once, at aggregation.
        decimal_pred = max(0.001, min(0.999, parsed.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if not isinstance(question, BinaryQuestion):
            return await super()._aggregate_predictions(predictions, question)

        samples = [float(p) for p in predictions if isinstance(p, (int, float))]
        if len(samples) != len(predictions):
            logger.warning(
                "Non-float binary predictions on %s; using the framework aggregator.",
                question.page_url,
            )
            return await super()._aggregate_predictions(predictions, question)

        result = aggregate(samples, self.aggregation)
        spread = disagreement(samples)

        logger.info(
            "%s | samples=%s | spread=%.2f | aggregate=%.3f%s",
            question.page_url,
            [round(s, 3) for s in samples],
            spread,
            result,
            "  <- HIGH DISAGREEMENT, ensemble is likely missing a fact"
            if needs_more_research(samples, self.aggregation)
            else "",
        )
        return result  # type: ignore[return-value]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Disable the cross-model ensemble (and extremisation with it).",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Use zero-cost models. For smoke tests only - they are not competitive.",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    bot = CalibratedBot(
        research_reports_per_question=1,
        # Three. The survey found 86% of prize winners aggregate across multiple forecasts, with
        # "median or mean of 3-10 runs" typical - two was below the range that wins anything.
        #
        # The counter-argument is in this file already: the samples come from one model at
        # temperature, so averaging cancels decode noise and little else. That caps the value of a
        # third sample, it does not make it negative, and at $0.023 a sample against a $0.17
        # question it is the cheapest lever left.
        #
        # The budget is what sets the ceiling. Roughly 175 questions remain before the season
        # closes on 6 September; at three samples that is about $34 of the ~$44 available, which
        # leaves headroom for Exa research being variable ($0.055-0.13) and for the question count
        # being an estimate. Four samples would spend essentially all of it and leave none.
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        # The framework summarises the research on every question by default, and with
        # use_research_summary_to_forecast=False that summary is never fed to the forecast - it
        # only decorates the published report. It is not a cheap call either: the input is the
        # whole research text, so it was the second largest line item in the bill behind search
        # itself, spent entirely on cosmetics. Off.
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        ensemble_models=(
            []
            if args.single_model
            else (FREE_ENSEMBLE_MODELS if args.free else ENSEMBLE_MODELS)
        ),
        llms=build_llms(free=args.free),
    )
    if args.free:
        logger.warning(
            "Running on zero-cost models. These are for verifying plumbing, not for competing: "
            "the free tier scores near zero on Metaculus's own model leaderboard."
        )

    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/summer-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-summer-2025/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
        minibench = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
        forecast_reports = seasonal + minibench
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True)
        )
    else:
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
