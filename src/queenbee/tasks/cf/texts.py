"""Planner-facing texts of the Count-Frequency task.

The header and unit legend, the API card (the shared card with its scoring
paragraph and rule 5 rewritten for this task, the ``scale_rounds_log2``
round-count option of this task's seed interpreter added, and budget caps
at n = 8), the diagnosis-cards block with the CF card legend, and the task
statement shown under TASK TEMPLATES.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Sequence

from queenbee.tasks.cf.cases import N_AGENTS, STATEMENT, TITLE

#: Round cap used for the card's budget line when the budgets name none.
PYTHON_MAX_ROUNDS = 64

CF_HEADER = (
    "QUEENBEE-EVO PROGRAM DESIGN\n"
    "You improve ONE multi-agent team program for an information-siloed task: "
    "8 worker agents each hold a private shard of one integer array, and agent 0 "
    "alone submits the team's answer (a table of counts), which is scored by its "
    "error (see SCORING AND COST in the API CARD). A frozen worker model executes "
    "the program; you write ONLY its GENOME (see the API CARD). The host runs your "
    "proposal on training units and compares it with a fresh run of its parent on "
    "the same units.\n"
)

CF_UNIT_LEGEND = (
    "Unit ids are <label>@o8: every unit is one instance of the task in TASK TEMPLATES "
    "(its own random array) solved by 8 agents; the label before '@' only names the "
    "instance, it is not a different task."
)

# Anchor / replacement pairs: each ``_X_OLD`` text must occur exactly once in
# the shared API card (:mod:`queenbee.evo.api_card`), and :func:`cf_api_card`
# replaces it with the matching ``_X_NEW`` text.
_SCORING_OLD = (
    "- Every agent submits and every agent must be correct: S = the fraction of\n"
    "  agents whose submitted answer is correct.\n"
    "- Calls = sends + n submits. Cost = the tokens of all worker calls.\n"
)
_SCORING_NEW = (
    "- information_goal is \"sink\": only agent 0 (selected_primary) submits and\n"
    "  only its answer is scored; the other agents are not called at round S.\n"
    "- RMSE = sqrt(sum over every value the task allows of (submitted count -\n"
    "  true count)^2); a value missing from the answer counts 0; the sum is not\n"
    "  divided by the number of values. S = exp(-RMSE / 5): S = 1 only for the\n"
    "  exact answer (RMSE 1 -> S 0.82, 3 -> 0.55, 5 -> 0.37, 10 -> 0.14). An\n"
    "  answer that is not a JSON object of counts, or a broken run, scores S = 0.\n"
    "- Calls = sends + 1 submit. Cost = the tokens of all worker calls.\n"
)
_RULE5_OLD = "5. At round S every agent is called once (submit) and sees the same\n"
_RULE5_NEW = (
    "5. At round S every submitter is called once (submit; under this task's\n"
    "   information goal \"sink\" only agent 0 = selected_primary submits) and\n"
    "   sees the same\n"
)
_INTERP_OLD = "  phase rounds (capped at max_rounds - 1).\n"
_INTERP_NEW = (
    "  phase rounds (capped at max_rounds - 1). An entry may instead set\n"
    "  \"scale_rounds_log2\": true: it then runs (n_agents - 1).bit_length()\n"
    "  rounds (ceil(log2(n_agents)); 3 at n_agents = 8).\n"
)


def _once(text: str, old: str, new: str, label: str) -> str:
    """``text`` with its one occurrence of ``old`` replaced by ``new``; an
    anchor that is missing or repeated raises ``RuntimeError``."""
    if text.count(old) != 1:
        raise RuntimeError(f"CF API card: anchor {label!r} found {text.count(old)} times")
    return text.replace(old, new)


@lru_cache(maxsize=1)
def cf_api_card() -> str:
    """The shared API card with this task's scoring, rule 5 and phase
    round-count text (every anchor must occur exactly once)."""

    from queenbee.evo import api_card as _card

    text = _card.API_CARD
    text = _once(text, _SCORING_OLD, _SCORING_NEW, "scoring")
    text = _once(text, _RULE5_OLD, _RULE5_NEW, "rule 5")
    text = _once(text, _INTERP_OLD, _INTERP_NEW, "phase rounds")
    return text


def render_api_card(*, budgets: object | None = None, preserve_dups: bool = False) -> str:
    """The task's ``render_api_card``: :func:`cf_api_card` (plus, under
    ``preserve_dups``, the rule that forwarded raw data keeps repeated
    values) and, with ``budgets``, the budget caps of an n = 8 run at the
    budgets' round cap (else :data:`PYTHON_MAX_ROUNDS`)."""

    from queenbee.evo import api_card as _card
    from queenbee.program.budgets import PythonRunBudgets

    card = cf_api_card()
    if preserve_dups:
        card = _once(card, _card._PRESERVE_DUPS_ANCHOR,
                     _card._PRESERVE_DUPS_ANCHOR + _card.PRESERVE_DUPS_RULE, "preserve dups")
    if budgets is None:
        return card
    rounds = getattr(budgets, "max_rounds", None)
    b8 = PythonRunBudgets.for_rounds(int(rounds) if isinstance(rounds, int) else PYTHON_MAX_ROUNDS,
                                     n_agents=N_AGENTS)
    fields = [f"{name}={getattr(b8, name)}" for name in
              ("max_rounds", "max_model_calls", "max_messages", "max_completion_tokens")
              if isinstance(getattr(b8, name, None), int)]
    return card + "Budget caps of this run: " + ", ".join(fields) + ".\n"


def diag_cards_block(unit_rows: Mapping[str, list], targets: list[str]) -> str:
    """The task's ``diag_cards_block``: the built-in block's header and
    renderer with the CF card legend."""

    from queenbee.evo.credit import credit_legend, render_diag_cards_with_credit

    cards = render_diag_cards_with_credit(unit_rows, max_chars=8000, order=targets)
    return (
        "=== DIAGNOSIS CARDS (the parent's runs on training units; answer-free) ===\n"
        "One line per run: S, failure class, submit round, msgs, lost_bodies = bodies "
        "delivered to an agent that was not called that round, tokens/phase = completion "
        "tokens per PHASES entry | submit, rmse = error of agent 0's submitted table "
        "(S = exp(-rmse/5)), sum_dev = (sum of agent 0's counts) - (number of array items), "
        "wrong_values = values whose submitted count is wrong, local = error of the "
        "single-shard count tables found in message bodies (k/8 = shards whose own counts "
        "were sent alone; others are not observed), local_est = local scaled to all 8 "
        "shards, merge_excess = rmse - local_est (error added while combining), "
        "labels_bad = a/m: of the m message bodies holding a count table, a have a TOTAL "
        "line that differs from the sum of their own table; cover_bad = b/m: b tables whose "
        "counts sum differs from the true number of items of the shards the message "
        "carries (a shard dropped or counted twice); submit_added = what agent 0 submitted "
        "minus (the tables it received + its own shard): about +128 means agent 0 added "
        "one shard a second time. "
        + credit_legend() + "\n" + (cards if cards else "no diagnosed runs\n")
    )


def t_templates(T: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The task's ``t_templates``: the one task statement every unit shares
    (shown as ``[CF] Count Frequency``)."""

    from queenbee.evo.prompt import template_statement

    return [template_statement("CF", STATEMENT, case_name=TITLE)]


__all__ = [
    "CF_HEADER",
    "CF_UNIT_LEGEND",
    "cf_api_card",
    "diag_cards_block",
    "render_api_card",
    "t_templates",
]
