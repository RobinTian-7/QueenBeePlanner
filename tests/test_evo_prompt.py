"""QueenBee-Evo planner prompt and API card.

Offline.  Pins:

* component gating: every on/off combination of the three experience
  components (diagnosis cards, hypothesis ledger, program archive) renders
  exactly the full prompt's blocks minus the gated ones, byte for byte; with
  an empty ledger and archive the "full" and "mf_elite" prompts differ ONLY
  in the diagnosis blocks;
* the always-on blocks: API card, all T templates' public statements (no
  topology, no data), parent genome, parent per-unit S and cost, brief,
  output contract (HYPOTHESIS with predicted_dS + one genome block, smallest
  effective change);
* V / TEST never reach a prompt (inputs refused, finished text audited,
  forbidden values audited in the evidence sections);
* the API card states semantics only (no mechanism advice);
* ``parse_evo_hypothesis`` keeps predicted_dS and ``@`` unit ids, and an
  evo prompt drives ``mint_python_challenger(genome_only=True)`` end to end
  with a stub planner.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest

import queenbee.program.mint as pl
from exp_graph.mas.python_code import validate_python_source
from queenbee.evo import api_card as ac
from queenbee.evo import credit as cr
from queenbee.evo import prompt as ep
from queenbee.evo import seed as es
from queenbee.program.genome import genome_region, splice_genome

# An example T / V split of the 18 development templates.
EXAMPLE_T = ["I-01", "I-02", "I-06", "I-08", "II-11", "II-12", "II-13", "II-17",
             "III-21", "III-22", "III-26", "III-27"]
EXAMPLE_V = ["I-03", "I-07", "II-16", "II-18", "III-23", "III-28"]
DEV = sorted(EXAMPLE_T + EXAMPLE_V)
_CASE_TOKEN = re.compile(r"(?<![\w-])(?:III|II|I)-\d{2}(?!\d)")
#: Diagnosis on, ledger and archive off (``--no-ledger --no-archive``).
C1_ONLY = ep.EvoFlags(True, False, False)
ALL_FLAGS = [ep.EvoFlags(*c) for c in itertools.product([True, False], repeat=3)]


def _raw_task(case: str) -> str:
    """A task description in the benchmark's markdown layout (synthetic)."""

    return (
        f"**Task: Example task {case}**\n\n"
        "Combine the values held by all agents.\n\n"
        "**Your Data (Position {agent_id}):**\n{input_shard}\n\n"
        "**Output:**\nEach agent submits one integer: the combined value.\n\n"
        "**Communication Protocol:**\n"
        "Topology: Star (optimal here: send everything to agent 0, then submit).\n"
        "Each agent submits the same final answer using submit_result(value).\n"
    )


def _write_bench(root: Path, cases: list[str], n_agents: int = 5) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (root / f"{case}_n{n_agents}.json").write_text(json.dumps(
            {"case_id": case, "case_name": f"Example {case}",
             "task_description": _raw_task(case)}), encoding="utf-8")
    return root


_T_TEMPLATES = [ep.template_statement(c, _raw_task(c), case_name=f"Example {c}")
                for c in EXAMPLE_T]


def _variant(literal_tail: str) -> str:
    return es.evo_seed_source().replace(
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "broadcast_last", "rounds": 1},\n' + literal_tail, 1,
    )


def _row(S, fclass, bits, *, C=40000.0, calls=10, credit=None):
    return {
        "S": S, "C": C, "model_calls": calls, "infra": None,
        "execution_class": "completed", "success": S >= 1.0,
        "diag": {"S": S, "failure_class": fclass, "agent_correct": bits,
                 "n_distinct_answers": 1 if fclass != "divergent" else 2,
                 "majority_correct": S > 0.5, "answer_shape": "scalar",
                 "phase_tokens": [100, 200, 300], "submit_round": 5,
                 "n_messages": 8, "n_lost_messages": 0},
        "credit": credit or {"readable": [5, 5, 5, 5, 5],
                             "retention": [0.3, 0.5, 0.8, 1.0, 1.0], "basis": "sim"},
    }


def _state(**extra: Any) -> dict[str, Any]:
    rows = {
        "I-01@o5": [_row(1.0, "ok", "11111"), _row(1.0, "ok", "11111")],
        "II-11@x5": [_row(0.6, "divergent", "11100")],
        "III-22@x5": [_row(0.2, "divergent", "10000", C=52000.0)],
        "III-27@x5": [_row(0.0, "consensus-wrong", "00000", C=91000.0)],
        "III-21@o5": [_row(1.0, "ok", "11111"),
                      {"S": 0.0, "infra": "gateway", "execution_class": "infra"}],
    }
    state = {
        "t_templates": [dict(t) for t in _T_TEMPLATES],
        "programs": {
            "v1": {"source": es.evo_seed_source()},
            "p1": {"source": _variant('    {"kind": "digest", "rounds": 1},\n')},
            "p2": _variant('    {"kind": "mesh", "rounds": 1},\n'),
        },
        "rows": {"v1": rows, "p1": {"III-22@x5": [_row(0.6, "divergent", "11100")]}},
        "ledger": [
            {"program_id": "p1", "gen": 1, "brief_id": "b0",
             "target_units": ["III-22@x5"], "target_class": "divergent",
             "mech_tags_host": ["add_phase:digest"],
             "mech_claim_norm": "digest after relay",
             "hypothesis": {"mechanism": "Add a digest round so the last relay is read."},
             "predicted": {"III-22@x5": 0.4}, "observed": {"III-22@x5": 0.4},
             "verdict": "confirmed"},
            {"program_id": "p2", "gen": 1, "brief_id": "b1",
             "target_units": ["III-27@x5"], "target_class": "consensus-wrong",
             "mech_tags_host": ["add_phase:mesh"], "observed": {"III-27@x5": 0.0},
             "verdict": "refuted"},
        ],
        "tabu": [("consensus-wrong", "add_phase:mesh")],
        "forbidden_case_ids": EXAMPLE_V,
    }
    state.update(extra)
    return state


def _brief(**extra: Any) -> dict[str, Any]:
    brief = {
        "brief_id": "brief-SECRET-7", "kind": "target",
        "target_units": ["III-22@x5", "II-11@x5", "III-27@x5"],
        "target_class": "divergent", "parent_ids": ["v1"],
        "exemplars": [
            {"program_id": "p1", "role": "fixed", "failure_class": "divergent",
             "observed": {"III-22@x5": 0.4}},
            {"program_id": "p2", "role": "failed", "failure_class": "divergent",
             "observed": {"III-27@x5": 0.0}},
        ],
    }
    brief.update(extra)
    return brief


def _join(blocks) -> str:
    return "\n".join(body.rstrip("\n") + "\n" for _n, body in blocks)


# --------------------------------------------------------------------------- #
# flag gating
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "diag,ledger,archive", list(itertools.product([True, False], repeat=3))
)
def test_every_flag_combination_is_full_minus_gated_blocks(diag, ledger, archive) -> None:
    state, brief = _state(), _brief()
    full = ep.build_evo_prompt_blocks(state, brief, "full")
    names = [n for n, _ in full]
    for gated in ep.GATED_BLOCKS.values():
        assert set(gated) - {"merge_parents"} <= set(names)
    flags = ep.EvoFlags(diag=diag, ledger=ledger, archive=archive)
    off = set()
    for name, on in (("diag", diag), ("ledger", ledger), ("archive", archive)):
        if not on:
            off |= set(ep.GATED_BLOCKS[name])
    expected = [b for b in full if b[0] not in off]
    got = ep.build_evo_prompt_blocks(state, brief, flags)
    assert got == expected
    assert ep.build_evo_prompt(state, brief, flags) == _join(expected)


def test_full_vs_mf_elite_differ_only_in_c1() -> None:
    """Same parent, the seed program (v1); same 3 targets; empty ledger and
    archive.  The mf_elite brief is derived with ``arm_brief`` (same kind +
    targets, no failure class)."""

    state = _state(ledger=[], tabu=[])
    brief = _brief(exemplars=[])
    full = ep.build_evo_prompt(state, brief, "full")
    mf_brief = ep.arm_brief(brief, "mf_elite")
    assert mf_brief["kind"] == "target" and mf_brief["target_class"] is None
    assert mf_brief["target_units"] == brief["target_units"]
    mf = ep.build_evo_prompt(state, mf_brief, "mf_elite")
    assert mf == ep.build_evo_prompt(state, brief, "mf_elite")  # class is ignored anyway
    # Ledger and archive empty (as in generation 0): full == diagnosis only.
    assert full == ep.build_evo_prompt(state, brief, C1_ONLY)
    blocks = ep.build_evo_prompt_blocks(state, brief, "full")
    c1 = [b for b in blocks if b[0] in ep.GATED_BLOCKS["diag"]]
    assert [b[0] for b in c1] == ["failure_map", "diag_cards", "target_diagnosis"]
    assert mf == _join([b for b in blocks if b[0] not in ep.GATED_BLOCKS["diag"]])
    for marker in ("=== FAILURE MAP", "=== DIAGNOSIS CARDS", "=== TARGET DIAGNOSIS",
                   "readable=", "raw_kept%=", "agents_right", "assigned to this brief"):
        assert marker in full and marker not in mf
    assert "=== LEDGER" not in full and "=== CONTRASTIVE" not in full
    # The failure class never reaches the mf_elite prompt through the brief.
    brief_mf = mf.split("=== BRIEF")[1].split("\n=== ")[0]
    assert "divergent" not in brief_mf
    assert ep.ARM_FLAGS == {"full": ep.EvoFlags(True, True, True),
                            "mf_elite": ep.EvoFlags(False, False, False)}
    assert ep.coerce_flags("MF-Elite") == ep.EvoFlags(False, False, False)
    assert ep.coerce_flags({"no_ledger": True}) == ep.EvoFlags(True, False, True)
    assert ep.coerce_flags({"C1": False, "C3": False}) == ep.EvoFlags(False, True, False)
    assert ep.coerce_flags(None) == ep.ARM_FLAGS["full"]
    with pytest.raises(ValueError):
        ep.coerce_flags("nope")


# --------------------------------------------------------------------------- #
# contents
# --------------------------------------------------------------------------- #


def test_always_on_blocks() -> None:
    text = ep.build_evo_prompt(_state(), _brief(), "mf_elite")
    assert ac.API_CARD in text
    for case in EXAMPLE_T:
        assert f"[{case}] Example task {case}\n  Output: Each agent submits one integer" in text
    genome = genome_region(es.evo_seed_source()).rstrip()
    assert f"=== PARENT GENOME (the program you modify) ===\n```python\n{genome}\n```" in text
    assert "- III-22@x5: S=0.20 (1 run) calls=10 tokens=52.0k" in text
    assert "- I-01@o5: S=1.00 (2 runs)" in text
    assert "- III-21@o5: S=1.00 (1 run)" in text  # the infra row is not scored
    assert "Target units: III-22@x5 (parent S=0.20), II-11@x5 (parent S=0.60), " \
           "III-27@x5 (parent S=0.00)" in text
    assert text.rstrip().endswith(ep.OUTPUT_CONTRACT.rstrip())
    assert '"predicted_dS"' in ep.OUTPUT_CONTRACT and "SMALLEST" in ep.OUTPUT_CONTRACT
    assert "ONE ```python code block" in ep.OUTPUT_CONTRACT
    assert "brief-SECRET-7" not in text  # bookkeeping ids are never rendered
    assert "brief-SECRET-7" not in ep.build_evo_prompt(_state(), _brief(), "full")


def test_c1_c2_c3_contents() -> None:
    text = ep.build_evo_prompt(_state(), _brief(), "full")
    cards = text.split("=== DIAGNOSIS CARDS")[1].split("\n=== ")[0]
    assert cards.index("[III-22@x5]") < cards.index("[I-01@o5]")  # targets first
    assert "readable=[5 5 5 5 5]/5 raw_kept%=[30 50 80 100 100]" in cards
    solved = cards.split("[I-01@o5]")[1].split("\n[")[0]
    assert "readable=[5 5 5 5 5]/5" in solved and "raw_kept" not in solved
    assert cr.CREDIT_LEGEND in text
    fmap = text.split("=== FAILURE MAP")[1].split("\n=== ")[0]
    assert "incumbent" not in fmap and "III-22@x5" in fmap
    ledger = text.split("=== LEDGER")[1].split("\n=== ")[0]
    assert "divergent x add_phase digest: 1 confirmed; mean dS=+0.40 over 1 executed" in ledger
    assert "consensus-wrong x add_phase mesh: 1 refuted; mean dS=+0.00" in ledger
    assert "- L1 (gen 1; targets III-22@x5; class divergent): Add a digest round" in ledger
    assert "- consensus-wrong x add_phase mesh" in ledger  # tabu
    ex = text.split("=== CONTRASTIVE EXEMPLARS")[1].split("\n=== ")[0]
    assert "exemplar A (program p1; improved class divergent; observed dS: III-22@x5 +0.40)" in ex
    assert "exemplar B (program p2; did not improve class divergent" in ex
    assert '{"kind": "digest", "rounds": 1}' in ex and '{"kind": "mesh", "rounds": 1}' in ex
    target = text.split("=== TARGET DIAGNOSIS")[1].split("\n=== ")[0]
    assert "Failure class assigned to this brief: divergent" in target
    assert "- III-27@x5: dominant class over the parent's runs = consensus-wrong" in target


def test_arm_brief_follows_spec_d() -> None:
    brief = _brief(kind="merge", parent_ids=["v1", "p1"],
                   wins={"v1": ["I-01@o5"], "p1": ["III-22@x5"]})
    full = ep.arm_brief(brief, "full")
    assert full["kind"] == "merge" and full["parent_ids"] == ["v1", "p1"]
    assert full["target_class"] == "divergent" and len(full["exemplars"]) == 2
    fb = ep.arm_brief(brief, C1_ONLY)  # diagnosis only: class kept, merge -> explore
    assert fb["kind"] == "explore" and fb["target_class"] == "divergent"
    assert fb["parent_ids"] == ["v1"] and "exemplars" not in fb and "wins" not in fb
    elite = ep.arm_brief(brief, "mf_elite")
    assert elite["target_class"] is None
    assert elite["target_units"] == brief["target_units"]
    assert elite["brief_id"] == brief["brief_id"]
    # Every flag setting's prompt = the full prompt's blocks minus its gated
    # blocks, also when it gets its derived brief.
    state = _state()
    for flags in ["mf_elite", *ALL_FLAGS]:
        derived = ep.build_evo_prompt_blocks(state, ep.arm_brief(brief, flags), flags)
        assert derived == ep.build_evo_prompt_blocks(state, brief, flags)


def test_full_prompt_survives_realistic_counts() -> None:
    """Raw 4-digit per-phase tokens, message counts and costs in a parent row
    are never printed raw, so ground truths made of such numbers do not
    raise LEAKAGE_GUARD."""

    values = [[1001, 1026, 1953, 1054, 12345] + [2000 + 37 * i for i in range(36)], 53.6]
    state = _state(ledger=[], tabu=[])
    for rows in state["rows"]["v1"].values():
        for row in rows:
            if "diag" in row:
                row["diag"]["phase_tokens"] = [1001, 1026, 402, 1953]
                row["diag"]["n_messages"] = 1054
                row["C"] = 12345.0
                row["model_calls"] = 53.6
    brief = _brief(exemplars=[])
    full = ep.build_evo_prompt(state, brief, "full", forbidden_values=values)
    assert "tokens/phase=[1.0k,1.0k,402|submit 2.0k]" in full and "msgs=1.1k" in full
    assert "calls=54 " in full  # a fractional mean (53.6) is never printed
    mf = ep.build_evo_prompt(state, ep.arm_brief(brief, "mf_elite"), "mf_elite",
                             forbidden_values=values)
    for text in (full, mf):
        assert ep.audit_evo_prompt(text, forbidden_case_ids=EXAMPLE_V,
                                   forbidden_values=values) == []


def test_merge_brief_and_its_degradation_without_archive() -> None:
    brief = _brief(kind="merge", parent_ids=["v1", "p1"], exemplars=[],
                   wins={"v1": ["I-01@o5"], "p1": ["III-22@x5"]})
    full = ep.build_evo_prompt(_state(), brief, "full")
    merge = full.split("=== MERGE PARENTS")[1].split("\n=== ")[0]
    assert "Merge parent 1 = the PARENT GENOME above; units it wins: I-01@o5" in merge
    assert "Merge parent 2 (program p1); units it wins: III-22@x5" in merge
    assert "Kind: merge" in full
    mf = ep.build_evo_prompt(_state(), brief, "mf_elite")
    assert "=== MERGE PARENTS" not in mf and "Kind: explore" in mf
    for kind in ("economize", "explore"):
        text = ep.build_evo_prompt(_state(), _brief(kind=kind, target_units=[]), "full")
        assert f"Kind: {kind}" in text and "Target units: none" in text
    with pytest.raises(ValueError):
        ep.build_evo_prompt(_state(), _brief(kind="mutate"), "full")
    with pytest.raises(ValueError):
        ep.build_evo_prompt(_state(), _brief(parent_ids=[]), "full")


# --------------------------------------------------------------------------- #
# leakage guard
# --------------------------------------------------------------------------- #


def test_no_v_or_test_id_in_any_prompt() -> None:
    for flags in [*ep.ARM_FLAGS, *ALL_FLAGS]:
        text = ep.build_evo_prompt(_state(), _brief(), flags)
        ids = set(_CASE_TOKEN.findall(text))
        assert ids <= set(EXAMPLE_T), ids - set(EXAMPLE_T)
        assert ep.audit_evo_prompt(text, forbidden_case_ids=EXAMPLE_V) == []


@pytest.mark.parametrize("mutate", [
    lambda s, b: s["t_templates"].append({"unit_id": "I-04", "title": "x"}),
    lambda s, b: s["t_templates"].append({"unit_id": "II-16", "title": "x"}),
    lambda s, b: s["rows"]["v1"].setdefault("II-18@o5", [_row(1.0, "ok", "11111")]),
    lambda s, b: s["rows"]["v1"].setdefault("III-30@x5", [_row(1.0, "ok", "11111")]),
    lambda s, b: b["target_units"].append("I-07@x5"),
    lambda s, b: b["target_units"].append("II-20@o5"),
    lambda s, b: s["ledger"][0]["target_units"].append("III-23@x5"),
    lambda s, b: s["ledger"][0]["hypothesis"].update(mechanism="copy what I-09 does"),
    lambda s, b: s["ledger"][0]["hypothesis"].update(mechanism="as on II-18 too"),
    lambda s, b: b["exemplars"][0]["observed"].update({"III-28@x5": 0.1}),
])
def test_guards_raise_on_held_out_templates(mutate) -> None:
    state, brief = _state(), _brief()
    mutate(state, brief)
    with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
        ep.build_evo_prompt(state, brief, "full")


def test_template_statements_are_public_text_only(tmp_path: Path, monkeypatch) -> None:
    bench = _write_bench(tmp_path / "benchmarks", DEV)
    items = ep.t_template_statements(DEV, benchmarks_dir=bench)
    assert [i["unit_id"] for i in items] == DEV
    for item in items:
        assert item["output_sentence"] and item["title"]
        blob = " ".join(item.values()).lower()
        assert "topology" not in blob and "optimal" not in blob
        assert "your data" not in blob and "{input_shard}" not in blob
    assert items[0] == {
        "unit_id": "I-01", "title": "Example task I-01",
        "output_sentence": "Each agent submits one integer: the combined value.",
        "protocol_sentence": "Each agent submits the same final answer using "
                             "submit_result(value).",
    }
    # without benchmarks_dir the statements come from $SILO_BENCH_DIR
    monkeypatch.setenv("SILO_BENCH_DIR", str(bench))
    assert ep.t_template_statements(DEV) == items
    for bad in ("I-05", "III-29"):
        with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
            ep.t_template_statements(["I-01", bad])


def test_forbidden_values_audit() -> None:
    # ground-truth-like values: a prefix-sum array, a large scalar, a word
    # answer and a dict answer
    values = [[1003 + 37 * i for i in range(41)], 987654, "alpha-beta-gamma",
              {"answer": 24680}]
    state, brief = _state(), _brief()
    text = ep.build_evo_prompt(state, brief, "full", forbidden_values=values)
    assert ep.audit_evo_prompt(text, forbidden_case_ids=EXAMPLE_V, forbidden_values=values) == []
    hits = ep.audit_evo_prompt(text + "\n=== FAILURE MAP\nsaw 24680 and alpha-beta-gamma\n",
                               forbidden_values=values)
    assert {h["value_index"] for h in hits} == {2, 3}
    # a value outside the evidence sections (e.g. program text) is not a hit
    assert ep.audit_evo_prompt("```python\nX = 987654\n```\n", forbidden_values=values) == []
    # A value planted in an evidence section is caught before sending.
    state["ledger"][0]["hypothesis"]["mechanism"] = "answer is 987654321"
    with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
        ep.build_evo_prompt(state, brief, "full", forbidden_values=[987654321])


# --------------------------------------------------------------------------- #
# API card
# --------------------------------------------------------------------------- #


def test_api_card_states_the_spec_semantics() -> None:
    card = " ".join(ac.API_CARD.split())
    for phrase in (
        'Only "send" calls the worker',
        "body is LOST, although its source ids still count",
        "An agent's memory is only its own previous message",
        "its previous message, the inbox delivered at round S and its private "
        "submit prompt",
        "A submit call cannot carry a work instruction",
        "S = the fraction of agents whose submitted answer is correct",
        "Calls = sends + n submits",
        "Work instructions must be task-agnostic",
        "must work for any n_agents and any task type",
        "for r in range(n_agents):",
        '"send" with recipients [] is legal',
        'attaches that "wi"',
        "The digest kind",
        "PYTHON_CONTROL_JSON {mode, recipients, work_instruction}",
        "It never sees PHASES or any other agent's action",
        "pass, del, yield (generators)",
        "the walrus operator (:=)",
        "`for entry in PHASES:`) is rejected",
        "`for i in range(len(PHASES)):`",
        "may not appear anywhere in the program",
        "Every name used must be defined",
    ):
        assert phrase in card, phrase
    assert ac.API_CARD.startswith(ac.API_CARD_HEADER)
    from exp_graph.mas.python_code import MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION

    assert MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION.rstrip("\n") in ac.API_CARD
    # The deny-lists are the validator's own (no drift) and are all listed.
    from exp_graph.mas import python_code as pc

    for name in pc._FORBIDDEN_NAMES:
        if not name.startswith("__"):
            assert re.search(rf"\b{name}\b", card), name
    for attr in pc._FORBIDDEN_ATTRIBUTE_NAMES:
        assert re.search(rf"\b{attr}\b", card), attr
    assert max(len(line) for line in ac._CORE.splitlines()) <= 78


@pytest.mark.parametrize("snippet,ok", [
    ("def f_x(a):\n    pass\n", False),
    ("def f_x(a):\n    b = [1]\n    del b[0]\n    return b\n", False),
    ("def f_x(a):\n    print(a)\n    return a\n", False),
    ("def f_x(a):\n    b = [1, 2]\n    b.remove(1)\n    return b\n", False),
    ("def f_x(a):\n    t = 0\n    for p in PHASES:\n        t = t + 1\n    return t\n", False),
    ("def f_x(a):\n    return [p for p in PHASES]\n", False),
    ("def f_x(a):\n    if (b := a):\n        return b\n    return a\n", False),
    ("def f_x(a):\n    yield a\n", False),
    ("NOTE_X = 'Todo later'\n", False),
    ("def f_x(a):\n    t = 0\n    for i in range(len(PHASES)):\n        t = t + 1\n"
     "    return t\n", True),
    ("def f_x(n_agents):\n    return [r for r in range(n_agents)]\n", True),
    ("def f_x(a):\n    t = 0\n    for i, v in enumerate([1, 2]):\n        t = t + v\n"
     "    return t\n", True),
])
def test_api_card_code_law_matches_the_validator(snippet: str, ok: bool) -> None:
    """Every code-law example the card states is what the validator does."""

    anchor = "def phase_digest("
    source = es.evo_seed_source().replace(anchor, snippet + "\n\n" + anchor, 1)
    report = validate_python_source(source, worker_contract="message_only_v2")
    assert report.valid is ok, report.errors


def test_api_card_gives_no_mechanism_advice() -> None:
    core = ac._CORE  # the card minus the verbatim runtime worker strings
    advice = re.compile(
        r"\b(prefer\w*|recommend\w*|strong\w*|consider|best|better|avoid\w*|"
        r"should|improv\w*|good|hub|mesh|tree|star|gossip|majority|vote|"
        r"consensus|redundan\w*|verif\w*|typically|usually)\b",
        re.I,
    )
    assert advice.findall(core) == []
    for word in ("relay", "broadcast"):
        assert word not in core.lower()
    assert ac.render_api_card(budgets=None) == ac.API_CARD
    from queenbee.program.budgets import PythonRunBudgets

    with_caps = ac.render_api_card(budgets=PythonRunBudgets.for_rounds(64, n_agents=5))
    assert with_caps.startswith(ac.API_CARD) and "max_rounds=64" in with_caps


# --------------------------------------------------------------------------- #
# reply parsing + stub-planner mint
# --------------------------------------------------------------------------- #


def test_normalize_mechanism_ledger_keys_and_tabu_labels() -> None:
    norm = ep.normalize_mechanism
    assert norm("Add a Mesh-round!!  before submit") == "add a mesh round before submit"
    assert norm("add_phase:digest") == "add_phase digest"
    assert norm("  UPPER lower_case 123 !@# \u00e9") == "upper lower_case 123"
    assert norm(None) == "" and norm("") == "" and norm(42) == "42"
    entry = {"target_class": "Divergent!", "mech_tags_host": ["wi_add", "add_mesh", "wi_add"]}
    assert ep._ledger_key(entry) == ("divergent", "add_mesh+wi_add")
    assert ep._ledger_key({"mech_claim_norm": "Gather To A Hub"}) == ("unknown", "gather to a hub")
    assert ep._ledger_key({}) == ("unknown", "unlabeled")
    assert ep._tabu_label(("Consensus-Wrong", "Add_Phase:Mesh")) == \
        "consensus-wrong x add_phase mesh"
    assert ep._tabu_label({"failure_class": "divergent", "mechanism": "Add a mesh!"}) == \
        "divergent x add a mesh"
    assert ep._tabu_label("Free TEXT") == "free text" and ep._tabu_label("") is None


def test_evidence_sections_extend_the_mint_audit_sections() -> None:
    base = tuple(pl.EVIDENCE_SECTION_HEADERS)
    headers = ep.EVO_EVIDENCE_SECTION_HEADERS
    assert headers[: len(base)] == base and len(set(headers)) == len(headers)
    assert set(headers) - set(base) == {
        "=== DIAGNOSIS CARDS", "=== PARENT RECORD", "=== BRIEF", "=== TARGET DIAGNOSIS",
        "=== LEDGER", "=== CONTRASTIVE EXEMPLARS", "=== MERGE PARENTS"}


def test_parse_evo_hypothesis() -> None:
    reply = (
        'noise\nHYPOTHESIS: {"target_units": ["III-22@x5", "II-11@x5"], '
        '"failure_class": "Divergent", "mechanism": "add a digest round", '
        '"predicted_dS": {"III-22@x5": 0.4, "II-11@x5": "0.2", "bad": 7}}\n```python\n'
    )
    hyp = ep.parse_evo_hypothesis(reply)
    assert hyp == {"target_units": ["III-22@x5", "II-11@x5"],
                   "failure_class": "divergent", "mechanism": "add a digest round",
                   "predicted_dS": {"III-22@x5": 0.4, "II-11@x5": 0.2, "bad": 1.0}}
    literal = "HYPOTHESIS: {'target_units': 'I-01@o5 I-02@o5', 'predicted_dS': 0.25}"
    assert ep.parse_evo_hypothesis(literal) == {
        "target_units": ["I-01@o5", "I-02@o5"], "predicted_dS": 0.25}
    older = 'HYPOTHESIS: {"mechanism": "m", "predicted_effect": "agents agree"}'
    assert ep.parse_evo_hypothesis(older) == {"mechanism": "m",
                                              "predicted_effect": "agents agree"}
    assert ep.parse_evo_hypothesis("HYPOTHESIS: add a round") == {"raw": "add a round"}
    # Decorated headers: markdown emphasis, inline code, a json fence, blank lines.
    want = {"target_units": ["I-01@o5"], "predicted_dS": {"I-01@o5": 0.3}}
    body = '{"target_units": ["I-01@o5"], "predicted_dS": {"I-01@o5": "30%"}}'
    for header in (f"**HYPOTHESIS:** {body}", f"HYPOTHESIS: `{body}`",
                   f"**HYPOTHESIS**:\n{body}", f"HYPOTHESIS:\n```json\n{body}\n```",
                   f"HYPOTHESIS:\n\n  {body}"):
        assert ep.parse_evo_hypothesis(header + "\n```python\nPHASES = [{}]\n```") == want
    # A later brace that is not the hypothesis (the genome block) is not taken.
    genome_only = 'HYPOTHESIS: none\n```python\nPHASES = [\n    {"kind": "mesh"}]\n```'
    assert ep.parse_evo_hypothesis(genome_only) == {"raw": "none"}
    assert ep._clip_dS("30%") == 0.3 and ep._clip_dS("-15 %") == -0.15
    assert ep._clip_dS("0.2") == 0.2 and ep._clip_dS(7) == 1.0
    assert ep._clip_dS("x%") is None
    assert ep.parse_evo_hypothesis("no header") is None
    assert ep.parse_evo_hypothesis(None) is None


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 20})()


class _StubPlanner:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        self.prompts.append(prompt)
        return _Resp(self.replies.pop(0))


def test_evo_prompt_drives_the_genome_only_mint(tmp_path) -> None:
    seed = es.evo_seed_source()
    genome = genome_region(seed).replace(
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "digest", "rounds": 1, "wi": "Merge every delivered body '
        'into one complete body."},\n    {"kind": "broadcast_last", "rounds": 1},\n', 1,
    )
    reply = (
        'HYPOTHESIS: {"target_units": ["III-22@x5"], "failure_class": "divergent", '
        '"mechanism": "digest before broadcast", "predicted_dS": {"III-22@x5": 0.3}}\n'
        "```python\n" + genome + "```\n"
    )
    prompt = ep.build_evo_prompt(_state(), _brief(), "full")
    request = pl.build_planner_request(
        n_agents=5, goal="all_agents", worker_contract="message_only_v2"
    )
    runtime = pl.build_mint_runtime(
        llm_provider="fake", worker_model="fake", goal="all_agents",
        worker_contract="message_only_v2", max_parallel_agents=1,
        request_timeout=30.0, n_agents=5,
    )
    planner = _StubPlanner([reply])
    hyp_out: dict[str, Any] = {}
    source = pl.mint_python_challenger(
        planner_client=planner, planner_model="stub", prompt=prompt,
        request=request, runtime=runtime, workdir=tmp_path / "mint",
        genome_only=True, incumbent_source=seed, hypothesis_out=hyp_out,
        prompt_dump_dir=tmp_path / "mint",
    )
    assert planner.prompts == [prompt]
    assert source == splice_genome(seed, genome)
    assert validate_python_source(source, worker_contract="message_only_v2").valid
    assert hyp_out.get("mechanism") == "digest before broadcast"
    raw = (tmp_path / "mint" / "mint_reply_00.txt").read_text()
    assert ep.parse_evo_hypothesis(raw)["predicted_dS"] == {"III-22@x5": 0.3}
    assert ep.read_mint_hypothesis(tmp_path / "mint") == {
        "target_units": ["III-22@x5"], "failure_class": "divergent",
        "mechanism": "digest before broadcast", "predicted_dS": {"III-22@x5": 0.3},
    }
    assert ep.read_mint_hypothesis(tmp_path / "nowhere") is None
    # The dumped prompt passes the mint's leak audit (V ids + TEST ids).
    hits = pl.audit_prompt_leaks(
        tmp_path / "mint", list(EXAMPLE_V) + list(cr.TEST_CASE_IDS),
        sections=ep.EVO_EVIDENCE_SECTION_HEADERS,
    )
    assert hits == []


def test_ledger_shows_the_failure_left_after_an_attempt() -> None:
    from queenbee.evo import prompt as P

    entries = [{"gen": 1, "target_units": ["II-16@o5"], "verdict": "inconclusive",
                "observed": {"II-16@o5": 0.0},
                "hypothesis": {"mechanism": "hub computes the total and broadcasts it"},
                "after_diag": {"class": "local-only", "n_agents": 5, "runs": 1,
                               "own_shard_only": 5, "answer_seen": 5}},
               {"gen": 1, "target_units": ["III-23@o5"], "verdict": "confirmed",
                "observed": {"III-23@o5": 0.4}, "after_diag": {"class": "ok", "n_agents": 5}}]
    lines = P._after_lines(entries, set())
    assert len(lines) == 1
    assert "L1" in lines[0] and "own_shard_only_agents=5/5" in lines[0]
    assert "answer_seen_agents=5/5" in lines[0] and "class=local-only" in lines[0]
    assert P._after_lines([{k: v for k, v in entries[0].items() if k != "after_diag"}], set()) == []
