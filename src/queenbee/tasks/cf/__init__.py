"""The Count-Frequency task (``--task cf``).

Eight worker agents each hold a contiguous shard of 128 integers of one
global array of 1024 integers in 0..255; agent 0 alone (information goal
``sink``) submits the count table of the whole array, scored by
``S = exp(-RMSE / 5)`` (:mod:`queenbee.tasks.cf.scoring`).  Cases are
generated from their seed (:mod:`queenbee.tasks.cf.cases`): the training
templates CF-001 .. CF-004, the validation templates CF-201 .. CF-203 and
the TEST templates CF-301 .. CF-310, all on the one rung ``o8``.

:data:`CF_TASK` wires the task into QueenBee-Evo: its ids, rung and goal;
the tree-reduction seed (:mod:`queenbee.tasks.cf.seed`); scoring and row
finalization (:mod:`queenbee.tasks.cf.scoring`); diagnosis cards and trace
enrichment (:mod:`queenbee.tasks.cf.diagnosis`); the S0 screen at n = 8
(:mod:`queenbee.tasks.cf.screen`; its work-instruction lint reads the
Silo-Bench development texts); the planner texts
(:mod:`queenbee.tasks.cf.texts`); behaviour fingerprints at n = 2, 5, 8
and 10; the opt-in count-table JSON repair in the worker sandbox; racing
thresholds, the fixed split and the run defaults.  Trace directories
(``--trace-dir``) only enrich census rows here: the near-miss and
structural unit evidence the loop derives from traces assumes Silo-Bench's
per-agent answers and is off for this task (``trace_unit_evidence``).

Typical use::

    python -m queenbee.evaluate --task cf --split train --seed-program v1 \\
        --repeats 2 --out census.json                        # seed census
    python -m queenbee.evo.loop --task cf --arm full --split-seed 1 \\
        --units-from census.json --root runs/cf_full         # one evolution run
    python -m queenbee.evaluate --task cf --split test \\
        --program champion=runs/cf_full/champion.py --repeats 1 --out test.json
"""

from __future__ import annotations

from exp_graph.mas.count_table_json import REPAIR_ENV
from queenbee.tasks.base import TaskSpec
from queenbee.tasks.cf.cases import (
    DEV_IDS,
    N_AGENTS,
    RUNG,
    TEMPLATE_PATTERN,
    TEST_IDS,
    default_split,
    instance_path,
    load_instance,
)
from queenbee.tasks.cf.diagnosis import (
    CF_GLOSSARY,
    card_suffix,
    diag_fn,
    sanitize_extra,
    trace_row,
)
from queenbee.tasks.cf.scoring import finalize_row, score_fn, summary_fields
from queenbee.tasks.cf.screen import GOAL, screen_program, vocabulary
from queenbee.tasks.cf.seed import seed_source
from queenbee.tasks.cf.texts import (
    CF_HEADER,
    CF_UNIT_LEGEND,
    diag_cards_block,
    render_api_card,
    t_templates,
)

#: Team sizes of the behaviour fingerprint (exec-cache / duplicate key).
FP_NS: tuple[int, ...] = (2, 5, 8, 10)
FP_SCHEME = "screen.behavior_fp@n2,5,8,10"

#: Racing thresholds (S = exp(-RMSE/5); one quantum, 1/n, is 0.125 at n = 8).
THRESHOLDS: dict[str, object] = {
    "tau_r1": 0.0, "tau_2": 0.1, "tau_h": 0.15, "tau_hi": 0.25,
    "loss_max": 1.0, "refute_quanta": 1.0, "source": "cf",
}

#: Defaults of the command-line tools under ``--task cf``.
CLI_DEFAULTS: dict[str, object] = {
    "budget": 48.0, "final_reserve": 12.0, "K": 3,
    "vacuous_guard": True, "local_only_diag": True, "attempt_diag": True,
    "python_max_rounds": 64, "agents": N_AGENTS,
}

CF_TASK = TaskSpec(
    name="cf",
    goal=GOAL,
    rungs={RUNG: N_AGENTS},
    rung_cost={RUNG: 1},
    rung_order={RUNG: 0},
    rung_next={},
    test_ids=TEST_IDS,
    dev_ids=DEV_IDS,
    template_pattern=TEMPLATE_PATTERN,
    fp_ns=FP_NS,
    fp_scheme=FP_SCHEME,
    seed_source=seed_source,
    instance_path=instance_path,
    load_instance=load_instance,
    t_templates=t_templates,
    score_fn=score_fn,
    diag_fn=diag_fn,
    worker_env={REPAIR_ENV: "1"},
    trace_row=trace_row,
    finalize_row=finalize_row,
    sanitize_extra=sanitize_extra,
    card_suffix=card_suffix,
    glossary=CF_GLOSSARY,
    per_agent_cards=False,
    trace_unit_evidence=False,
    screen_ns=(N_AGENTS,),
    wi_vocabulary=vocabulary,
    screen_program=screen_program,
    header=CF_HEADER,
    unit_legend=CF_UNIT_LEGEND,
    render_api_card=render_api_card,
    diag_cards_block=diag_cards_block,
    thresholds=THRESHOLDS,
    cli_defaults=CLI_DEFAULTS,
    default_split=default_split,
    summary_fields=summary_fields,
)

__all__ = ["CF_TASK", "CLI_DEFAULTS", "FP_NS", "FP_SCHEME", "THRESHOLDS"]
