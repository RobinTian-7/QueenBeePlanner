"""Task interface (``queenbee.tasks``), offline.

A toy task with its own template ids, rung, goal (``sink``), fingerprint team
sizes, screen policy, planner texts, diagnosis hooks, scoring hooks,
instances, split, thresholds and command-line defaults: each hook is checked
where the loop, the race, the screen, the prompt builder and the evaluator
read it.
The Silo-Bench default task renders its pinned texts (sha256) whatever task
was active in between, and ``use_task`` restores the task that was active
before it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import queenbee.program.mint as mint
import queenbee.tasks as tasks
import test_evo_loop as TL
import test_evo_prompt as TP
from exp_graph.mas.python_code_runner import CodeProcessRunner
from queenbee.bench.instance import BenchmarkInstance
from queenbee.evo import api_card as ac
from queenbee.evo import common as C
from queenbee.evo import credit as cr
from queenbee.evo import diagnosis as dg
from queenbee.evo import ladder as LD
from queenbee.evo import loop as L
from queenbee.evo import prompt as ep
from queenbee.evo import race as R
from queenbee.evo import screen as S
from queenbee.evo import units as U
from queenbee.evo.seed import evo_seed_source
from queenbee.program.budgets import PythonRunBudgets
from queenbee.tasks import TaskSpec, get_task, set_task, use_task

SEED = evo_seed_source()
TOY_DEV = ("TT-01", "TT-02", "TT-03", "TT-04", "TT-05", "TT-06")
TOY_TEST = ("TT-90", "TT-91")
TOY_PATTERN = r"(?<![A-Za-z0-9-])(TT-\d{2})(?!\d)"
TOY_HEADER = "TOY TASK PROGRAM DESIGN\nFour agents hold one shard each; agent 0 alone submits.\n"
TOY_LEGEND = "Unit ids are <template>@o4: one toy instance solved by 4 agents."
TOY_SPLIT = {"name": "toy-split", "T": ["TT-01", "TT-02", "TT-03", "TT-05"],
             "V": ["TT-04", "TT-06"]}
TOY_THRESHOLDS = {"tau_r1": 0.0, "tau_2": 0.1, "tau_h": 0.15, "tau_hi": 0.25,
                  "loss_max": 1.0, "source": "toy"}
TOY_GLOSS = {"consensus-wrong": "TOY GLOSSARY consensus: the submitter's table is off",
             "divergent": "TOY GLOSSARY divergent: the shards disagree"}
TOY_VOCAB = S.WIVocabulary(unigrams=frozenset({"toyword"}), bigrams=frozenset(),
                           templates=("TT",), source_texts=())
TOY_PROMPT = ("Find the GLOBAL MAXIMUM across all agents' data. You are Agent {agent_id} "
              "and hold: {input_shard}\n**Output:** A single integer.")
TOY_CLI = {"budget": 14.0, "final_reserve": 4.0, "K": 2, "vacuous_guard": True, "agents": 4}

#: sha256 of the Silo-Bench renderings of the ``test_evo_prompt`` fixtures:
#: the default task must render exactly these texts.
SILO_PINS = {
    "full": "41ce9f2233997929ba39964f983e0c996d24a1c1b04c4bfcaca376e9309c3d22",
    "dups": "8660db9a0827107f960a63e924ed79eca4491aa016bb8fccfcbb3cfd0f99b207",
    "mf": "d3acacb61f9ebe30de0098738a9b5ff2754c1782fbdf153c847861b64905191c",
    "cards": "21fa43154eb7ed007626d8ca32c6e7cc4fa6f3eeb0683ccbb74ae839e2035fa2",
    "fmap": "78702b38a279eaca7f146a5538ff6798cd627e19c2754010ee9c02bde19192b7",
    "card_budget": "8333e6eb9bba6ae8c6e9ee7a5dedb3b7e1a225f7986f0d1e001da3550896d27c",
}

#: A lazily registered task (``register_task(name, "test_tasks:LAZY_TASK")``).
LAZY_TASK = TaskSpec(name="lazy-toy", goal="sink")


# --------------------------------------------------------------------------- #
# the toy task
# --------------------------------------------------------------------------- #


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_instances(directory: Path, templates: tuple[str, ...] = TOY_DEV + TOY_TEST,
                     n: int = 4) -> Path:
    """One toy instance file per template (its own small JSON layout)."""

    directory.mkdir(parents=True, exist_ok=True)
    for salt, template in enumerate(templates):
        shards = [[salt + 3 * a + 1, salt + a + 7] for a in range(n)]
        doc = {"case_id": template, "n": n, "shards": shards,
               "answer": max(v for shard in shards for v in shard)}
        (directory / f"{template}.json").write_text(json.dumps(doc))
    return directory


def _load_toy(path: Path) -> BenchmarkInstance:
    doc = json.loads(Path(path).read_text())
    n = int(doc["n"])
    return BenchmarkInstance(
        benchmark="toy", case_id=str(doc["case_id"]), case_name="Toy maximum", n_agents=n,
        shards=list(doc["shards"]), ground_truth=doc["answer"], task_prompt=TOY_PROMPT,
        meta={"output_type": "distributed", "is_segmented": False,
              "expected_outputs": [doc["answer"]] * n, "toy": True},
    )


def _toy_statements(templates: Any) -> list[dict[str, str]]:
    return [{"unit_id": t, "title": f"Toy maximum {t}", "output_sentence": "A single integer.",
             "protocol_sentence": ""} for t in templates]


def _toy_extra(diag: Any) -> dict[str, Any]:
    value = diag.get("toy_err")
    ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 99
    return {"toy_err": value} if ok else {}


def _toy_suffix(card: Any) -> str:
    return f" toy_err={card['toy_err']}" if card.get("toy_err") is not None else ""


def _toy_card(*, budgets: Any = None, preserve_dups: bool = False) -> str:
    return (ac.render_api_card(budgets=budgets, preserve_dups=preserve_dups)
            + "TOY CARD NOTE: only agent 0 submits.\n")


def toy_task(instances: Path | None = None, **changes: Any) -> TaskSpec:
    """The toy task (``instances``: the directory of its instance files)."""

    fields: dict[str, Any] = dict(
        name="toy", goal="sink", rungs={"o4": 4}, rung_cost={"o4": 1}, rung_order={"o4": 0},
        rung_next={}, test_ids=TOY_TEST, dev_ids=TOY_DEV, template_pattern=TOY_PATTERN,
        fp_ns=(2, 4), fp_scheme="toy.behavior_fp@n2,4", mint_n_agents=4,
        seed_source=toy_seed,
        instance_path=(lambda uid: instances / f"{C.template_of(uid)}.json") if instances else None,
        load_instance=_load_toy, t_templates=_toy_statements,
        sanitize_extra=_toy_extra, card_suffix=_toy_suffix, glossary=dict(TOY_GLOSS),
        per_agent_cards=False, screen_ns=(4,), wi_vocabulary=lambda: TOY_VOCAB,
        header=TOY_HEADER, unit_legend=TOY_LEGEND, render_api_card=_toy_card,
        thresholds=dict(TOY_THRESHOLDS), cli_defaults=dict(TOY_CLI),
        default_split=lambda seed: dict(TOY_SPLIT),
    )
    fields.update(changes)
    return TaskSpec(**fields)


class ToyExecutor:
    """The toy task's ``execute`` hook: S from the template and the program's
    structure (a phase work instruction lifts TT-01 / TT-03 by 0.25, a
    leading mesh phase lowers S by 0.25); rows carry a ``toy_err`` card
    field."""

    BASE = {"TT-01": 0.5, "TT-02": 0.5, "TT-03": 0.25, "TT-04": 0.75, "TT-05": 1.0, "TT-06": 1.0}

    def __init__(self) -> None:
        self.units: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        unit = req["job"].unit_id
        with self._lock:
            self.units.append(unit)
        template = C.template_of(unit)
        f = TL._features(req["source"])
        s = self.BASE[template] + (0.25 if f["wi"] and template in ("TT-01", "TT-03") else 0.0)
        s = min(1.0, max(0.0, s - (0.25 if f["mesh"] else 0.0)))
        calls = 5 + 4 * f["mesh"] + f["digest"]
        return {"infra": None, "execution_class": "completed", "success": s >= 1.0, "S": s,
                "C": 1000.0 * calls, "model_calls": calls, "prompt_tokens": 400 * calls,
                "completion_tokens": 600 * calls,
                "diag": {"S": s, "failure_class": "ok" if s >= 1.0 else "consensus-wrong",
                         "answer_shape": "scalar", "toy_err": 3}}


#: The hard-coded sink branch of the seed program (v1): the other agents send to
#: agent 0 in round 0, and agent 0 reads those messages only when the submit
#: barrier is round 1.  The toy seed drops it so that PHASES drive every goal.
_SINK_BRANCH = """    if information_goal == "sink":
        if agent_id == selected_primary:
            if inbox_count:
                return {"mode": "reflect", "recipients": []}
            return {"mode": "idle", "recipients": []}
        if round_idx == 0:
            return {"mode": "send", "recipients": [selected_primary]}
        return {"mode": "idle", "recipients": []}
"""
assert SEED.count(_SINK_BRANCH) == 1
TOY_SEED = SEED.replace(_SINK_BRANCH, "")


def toy_seed() -> str:
    return TOY_SEED


def _with_phases(source: str, entries: str) -> str:
    """``source`` with its PHASES list replaced by ``entries`` (one per line)."""

    body = "".join(f"    {line},\n" for line in entries.strip().splitlines())
    out, n = re.subn(r"^PHASES = \[\n.*?^\]\n", "PHASES = [\n" + body + "]\n", source,
                     count=1, flags=re.S | re.M)
    assert n == 1
    return out


GATHER = _with_phases(TOY_SEED, '{"kind": "gather_to_hub", "rounds": 1}')
RELAY_ONLY = _with_phases(TOY_SEED, '{"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True}')
CHATTY = _with_phases(TOY_SEED, '{"kind": "relay", "rounds": 3}\n{"kind": "broadcast_last", "rounds": 20}')
TOYWORD = _with_phases(TOY_SEED, '{"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True, '
                             '"wi": "Write the toyword table."}\n'
                             '{"kind": "broadcast_last", "rounds": 1}')


@pytest.fixture(autouse=True)
def _silo_is_restored():
    assert get_task() is tasks.SILO_TASK
    yield
    leaked = get_task()
    if leaked is not tasks.SILO_TASK:
        set_task("silo")
        pytest.fail(f"test left task {leaked.name!r} active")


# --------------------------------------------------------------------------- #
# registry, activation, validation
# --------------------------------------------------------------------------- #


def test_default_task_is_silo_with_no_hooks() -> None:
    silo = tasks.SILO_TASK
    assert get_task() is silo and silo.name == tasks.DEFAULT_TASK == "silo"
    assert tasks.resolve_task("silo") is silo and tasks.available_tasks()[0] == "silo"
    hooks = {f.name: getattr(silo, f.name) for f in dataclasses.fields(silo)
             if f.name not in ("name", "per_agent_cards", "trace_unit_evidence", "cli_defaults")}
    assert all(value is None for value in hooks.values()), hooks
    assert silo.per_agent_cards is True and silo.trace_unit_evidence is True
    assert silo.cli_defaults == {}
    assert (C.task_goal(), C.task_rungs(), C.task_test_ids(), C.task_dev_ids()) == \
        (C.GOAL, C.RUNGS, C.TEST_IDS, C.DEV_IDS)
    assert (C.task_rung_order(), C.task_mint_n_agents(), L.rung_next()) == \
        (C.RUNG_ORDER, 5, L.RUNG_NEXT)
    assert (C.fingerprint_ns(), C.fingerprint_scheme()) == ((2, 5, 10), C.FP_SCHEME)
    assert cr.sealed_case_ids() == cr.TEST_CASE_IDS and C.task_score_hooks() == {}


def test_registry_registers_resolves_and_activates(monkeypatch) -> None:
    monkeypatch.setattr(tasks, "_REGISTRY", dict(tasks._REGISTRY))
    toy = toy_task()
    tasks.register_task(toy)
    assert tasks.available_tasks() == ("silo", "cf", "toy")
    assert get_task() is tasks.SILO_TASK  # registering activates nothing
    tasks.register_task(toy)  # the same TaskSpec object again
    with pytest.raises(ValueError, match="already registered"):
        tasks.register_task(toy_task())
    tasks.register_task(toy_task(), replace=True)
    with pytest.raises(ValueError, match="built-in"):
        tasks.register_task(TaskSpec(name="silo"))
    tasks.register_task("lazy-toy", "test_tasks:LAZY_TASK")
    assert tasks._REGISTRY["lazy-toy"] == "test_tasks:LAZY_TASK"  # not resolved yet
    assert tasks.resolve_task("lazy-toy") is LAZY_TASK
    assert tasks._REGISTRY["lazy-toy"] is LAZY_TASK
    tasks.register_task("made", lambda: TaskSpec(name="made"))
    assert tasks.resolve_task("made").name == "made"
    tasks.register_task("wrong", lambda: TaskSpec(name="other"))
    with pytest.raises(TypeError, match="did not give"):
        tasks.resolve_task("wrong")
    with pytest.raises(ValueError, match="module:attr"):
        tasks.register_task("bare", "no_colon")
    with pytest.raises(ValueError, match="unknown task"):
        set_task("nope")
    with pytest.raises(ValueError, match="another task"):
        set_task(TaskSpec(name="silo", goal="sink"))
    with use_task("lazy-toy") as spec:
        assert spec is LAZY_TASK and get_task() is LAZY_TASK
    assert get_task() is tasks.SILO_TASK


def test_use_task_restores_the_previous_task() -> None:
    toy, other = toy_task(), TaskSpec(name="other", goal="sink")
    with use_task(toy) as spec:
        assert spec is toy and get_task() is toy
        with use_task(other):
            assert get_task() is other
        assert get_task() is toy
        with pytest.raises(RuntimeError, match="boom"):
            with use_task(other):
                raise RuntimeError("boom")
        assert get_task() is toy
    assert get_task() is tasks.SILO_TASK
    with use_task("silo"):
        set_task(toy)  # set_task inside a use_task block is undone with it
        assert get_task() is toy
    assert get_task() is tasks.SILO_TASK


@pytest.mark.parametrize("fields, match", [
    ({"name": "Bad Name"}, "task name"),
    ({"name": "x", "goal": "every"}, "goal"),
    ({"name": "x", "rungs": {"o4": 5}}, "rung"),
    ({"name": "x", "rungs": {"big": 4}}, "rung"),
    ({"name": "x", "test_ids": ("TT-90",)}, "together"),
    ({"name": "x", "test_ids": ("TT-90",), "dev_ids": ("TT-01",),
      "template_pattern": r"(XX-\d\d)"}, "does not find"),
    ({"name": "x", "test_ids": ("TT-01",), "dev_ids": ("TT-01",),
      "template_pattern": TOY_PATTERN}, "overlap"),
    ({"name": "x", "test_ids": (), "dev_ids": (), "template_pattern": r"(a)(b)"}, "group"),
    ({"name": "x", "fp_ns": (1,)}, "fp_ns"),
    ({"name": "x", "worker_env": {"BAD-NAME": "1"}}, "worker_env"),
])
def test_task_spec_rejects_bad_fields(fields: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TaskSpec(**fields)


# --------------------------------------------------------------------------- #
# the Silo-Bench default renders its pinned texts
# --------------------------------------------------------------------------- #


def _silo_renderings() -> dict[str, str]:
    budgets = PythonRunBudgets.for_rounds(64, n_agents=5)
    state = TP._state()
    rows = state["rows"]["v1"]
    return {
        "full": ep.build_evo_prompt(state, TP._brief(), "full", forbidden_case_ids=TP.EXAMPLE_V),
        "dups": ep.build_evo_prompt(TP._state(budgets=budgets, preserve_dups=True), TP._brief(),
                                    "full", forbidden_case_ids=TP.EXAMPLE_V),
        "mf": ep.build_evo_prompt(state, TP._brief(), "mf_elite",
                                  forbidden_case_ids=TP.EXAMPLE_V),
        "cards": cr.render_diag_cards_with_credit(rows, max_chars=8000),
        "fmap": dg.render_failure_map(dg.build_failure_map(rows)),
        "card_budget": ac.render_api_card(budgets=budgets, preserve_dups=True),
    }


def test_silo_default_renders_exactly_what_it_rendered_before() -> None:
    default = {k: _sha(v) for k, v in _silo_renderings().items()}
    assert default == SILO_PINS
    with use_task("silo"):
        assert {k: _sha(v) for k, v in _silo_renderings().items()} == SILO_PINS
    with use_task(toy_task()):
        moved = {k: _sha(v) for k, v in _silo_renderings().items()}
    assert moved["full"] != SILO_PINS["full"] and moved["fmap"] != SILO_PINS["fmap"]
    assert {k: _sha(v) for k, v in _silo_renderings().items()} == SILO_PINS
    blocks = dict(ep.build_evo_prompt_blocks(TP._state(), TP._brief(), "full",
                                             forbidden_case_ids=TP.EXAMPLE_V))
    assert blocks["header"] == ep._HEADER and blocks["api_card"] == ac.API_CARD
    assert ep._UNIT_LEGEND in blocks["parent_record"]
    assert blocks["diag_cards"] == ep._block_diag_cards(TP._state()["rows"]["v1"],
                                                        TP._brief()["target_units"])


# --------------------------------------------------------------------------- #
# ids, rungs, goal, fingerprint
# --------------------------------------------------------------------------- #


def test_toy_ids_rungs_goal_and_fingerprint() -> None:
    silo_fp = C.behavior_fingerprint(SEED)
    with use_task(toy_task()):
        assert (C.task_goal(), C.task_rungs(), C.task_rung_order(), L.rung_next()) == \
            ("sink", {"o4": 4}, {"o4": 0}, {})
        assert (C.task_test_ids(), C.task_dev_ids(), C.task_mint_n_agents()) == \
            (TOY_TEST, TOY_DEV, 4)
        assert cr.sealed_case_ids() == TOY_TEST
        assert C.templates_in("runs/TT-90@o4_abc.json") == ["TT-90"] == cr.templates_in("TT-90")
        assert C.templates_in("I-04@o5") == [] == cr.templates_in("III-24")
        with pytest.raises(C.LeakageGuardError, match="TT-90"):
            C.assert_not_test(["x/TT-90@o4"], where="test")
        with pytest.raises(RuntimeError, match="TT-91"):
            cr.assert_not_test(["TT-91@o4"])
        C.assert_not_test(["I-04@o5"], where="test")  # not an id of this task
        assert LD.assert_dev_template("TT-01@o4") == "TT-01"
        with pytest.raises(LD.LadderLeakError, match="TEST"):
            LD.assert_dev_template("TT-90")
        with pytest.raises(LD.LadderLeakError, match="unknown"):
            LD.assert_dev_template("I-01")
        assert LD.assert_silo_dev_template("I-01") == "I-01"  # any task: Silo-Bench dev ids pass
        with pytest.raises(LD.LadderLeakError, match="TEST"):
            LD.assert_silo_dev_template("I-04")
        assert C.make_unit("TT-02", "o4") == "TT-02@o4"
        with pytest.raises(ValueError, match="rung"):
            C.make_unit("TT-02", "o5")
        assert U.parse_unit_id("TT-03@o4") == ("TT-03", "o4")
        units = ["TT-05@o4", "TT-01@o4", "TT-03@o4"]
        assert sorted(units, key=U._unit_sort_key) == ["TT-01@o4", "TT-03@o4", "TT-05@o4"]
        assert R.exec_eq("TT-01@o4") == 1
        with use_task(toy_task(rung_cost={"o4": 3})):
            assert R.exec_eq("TT-01@o4") == 3
        assert R.RaceConfig(fake=True, worker_model="w").worker_fingerprint()["goal"] == "sink"
        assert C.fingerprint_ns() == (2, 4) and C.fingerprint_scheme() == "toy.behavior_fp@n2,4"
        fp = C.behavior_fingerprint(SEED)
        fps = {n: S.behavior_fp(SEED, n, max_rounds=C._max_rounds(n), goal="sink") for n in (2, 4)}
        assert fp == "s:" + _sha(S.fp_key(fps))[:24]
        sink_silo_ns = C.behavior_fingerprint(SEED, ns=(2, 5, 10))
    assert fp != silo_fp and sink_silo_ns != silo_fp
    assert C.behavior_fingerprint(SEED) == silo_fp
    assert U._template_key("II-11") == (1, 11) and C.templates_in("TT-90") == []


# --------------------------------------------------------------------------- #
# screen
# --------------------------------------------------------------------------- #


def test_toy_screen_policy_goal_aware_coverage_and_hooks() -> None:
    # the coverage rule follows the goal: a gather to agent 0 serves the sink only
    sink = S.screen_program(GATHER, n_list=(4,), goal="sink", vocabulary=TOY_VOCAB)
    every = S.screen_program(GATHER, n_list=(4,), goal="all_agents", vocabulary=TOY_VOCAB)
    assert sink.ok and sink.rank_penalty == 0.0
    assert not every.ok and any(r.startswith("coverage@n4: agents [1, 2, 3]") for r in every.reasons)
    assert not S.screen_program(SEED, n_list=(4,), goal="sink", vocabulary=TOY_VOCAB).ok
    seed_calls = S.simulate_readable_coverage(TOY_SEED, 4, 64, goal="sink").calls
    with use_task(toy_task()):
        policy = S.task_screen_policy()
        assert policy == {"goal": "sink", "n_list": (4,), "v1_calls": {4: seed_calls},
                          "vocabulary": TOY_VOCAB}
        seed = S.screen_for_task(TOY_SEED)
        assert seed.ok and list(seed.per_n) == [4] and seed.fp_key.startswith("n4=")
        assert seed.fp_key == S.fp_key_for_task(TOY_SEED) == \
            S.program_fp_key(TOY_SEED, n_list=(4,), goal="sink")
        assert S.screen_for_task(GATHER).ok
        relay = S.screen_for_task(RELAY_ONLY)
        sim = S.simulate_readable_coverage(RELAY_ONLY, 4, 64, goal="sink")
        assert sim.readable[0] == 1 < min(sim.readable[1:3]) and sim.lost_edges == 1
        assert relay.reasons == [f"coverage@n4: agents [0] read fewer than 4 shards ({sim.readable})"]
        # one short submitter (1 of 4 agents) + one message body lost at the
        # submit barrier (1 of 3 messages)
        assert relay.rank_penalty == pytest.approx(1 / 4 + 1 / 3, abs=1e-6)
        chatty = S.screen_for_task(CHATTY)
        assert any(r.endswith(f"x v1 ({seed_calls})") and r.startswith("calls@n4")
                   for r in chatty.reasons)
        assert "wi-lint: task-specific terms ['toyword']" in S.screen_for_task(TOYWORD).reasons
        assert S.screen_for_task(TOY_SEED, known_fps={seed.fp_key: "v1"}).dup_of == "v1"
    assert S.fp_key_for_task(TOY_SEED) == S.program_fp_key(TOY_SEED) != seed.fp_key
    seen: dict[str, Any] = {}

    def own_screen(source: str, **kw: Any) -> Any:
        seen["screen"] = (source, sorted(kw))
        return "screened"

    def own_key(source: str, *, max_rounds: int) -> str:
        seen["key"] = max_rounds
        return "key"

    with use_task(toy_task(screen_program=own_screen, program_fp_key=own_key)):
        assert S.screen_for_task(SEED, tabu=None, known_fps={}) == "screened"
        assert S.fp_key_for_task(SEED, max_rounds=9) == "key"
    assert seen == {"screen": (SEED, ["known_fps", "tabu"]), "key": 9}


# --------------------------------------------------------------------------- #
# diagnosis cards and planner texts
# --------------------------------------------------------------------------- #


def test_toy_card_fields_suffix_and_glossary() -> None:
    raw = {"S": 0.25, "failure_class": "consensus-wrong", "toy_err": 4, "toy_note": "free text"}
    rows = {"TT-01@o4": [{"S": 0.25, "diag": raw}]}
    assert "toy_err" not in dg.sanitize_diag(raw)
    with use_task(toy_task()):
        card = dg.sanitize_diag(raw)
        assert card["toy_err"] == 4 and "toy_note" not in card
        assert dg.sanitize_diag(card) == card
        assert dg._card_line(1, card).endswith(" toy_err=4")
        assert "toy_err=4" in dg.render_diag_card("TT-01@o4", rows["TT-01@o4"])
        assert "toy_err=4" in cr.render_diag_cards_with_credit(rows)
        assert dg.failure_class_glossary()["consensus-wrong"] == TOY_GLOSS["consensus-wrong"]
        assert dg.failure_glossary_lines(["consensus-wrong"]) == \
            [f"  consensus-wrong: {TOY_GLOSS['consensus-wrong']}"]
        assert TOY_GLOSS["consensus-wrong"] in dg.render_failure_map(dg.build_failure_map(rows))
    assert dg.failure_class_glossary() is dg.FAILURE_CLASS_GLOSSARY
    assert dg.FAILURE_CLASS_GLOSSARY["consensus-wrong"] != TOY_GLOSS["consensus-wrong"]
    assert "toy_err" not in dg._card_line(1, dg.sanitize_diag(raw))


def test_toy_planner_texts_replace_only_their_blocks() -> None:
    state = TP._state(budgets=PythonRunBudgets.for_rounds(64, n_agents=4))
    silo = dict(ep.build_evo_prompt_blocks(state, TP._brief(), "full",
                                           forbidden_case_ids=TP.EXAMPLE_V))
    seen: dict[str, Any] = {}

    def card(*, budgets: Any, preserve_dups: bool) -> str:
        seen["card"] = (budgets.max_model_calls, preserve_dups)
        return "=== API CARD (toy) ===\nTOY API CARD\n"

    def cards(unit_rows: Any, targets: Any) -> str:
        seen["cards"] = (sorted(unit_rows), list(targets))
        return "=== DIAGNOSIS CARDS (toy) ===\nTOY CARDS\n"

    with use_task(toy_task(render_api_card=card, diag_cards_block=cards)):
        toy = dict(ep.build_evo_prompt_blocks(state, TP._brief(), "full",
                                              forbidden_case_ids=TP.EXAMPLE_V))
        text = ep.build_evo_prompt(state, TP._brief(), "full", forbidden_case_ids=TP.EXAMPLE_V)
    assert toy["header"] == TOY_HEADER and text.startswith(TOY_HEADER)
    assert toy["api_card"] == "=== API CARD (toy) ===\nTOY API CARD\n"
    assert seen["card"] == (PythonRunBudgets.for_rounds(64, n_agents=4).max_model_calls, False)
    assert toy["diag_cards"] == "=== DIAGNOSIS CARDS (toy) ===\nTOY CARDS\n"
    assert seen["cards"] == (sorted(state["rows"]["v1"]), TP._brief()["target_units"])
    assert toy["parent_record"] == silo["parent_record"].replace(ep._UNIT_LEGEND, TOY_LEGEND)
    assert TOY_GLOSS["divergent"] in toy["target_diagnosis"]
    assert TOY_GLOSS["divergent"] in toy["failure_map"]
    assert TOY_GLOSS["divergent"] not in silo["target_diagnosis"] + silo["failure_map"]
    for name in ("templates", "parent_genome", "ledger", "exemplars", "brief", "output_contract"):
        assert toy[name] == silo[name], name


def test_after_diag_counts_follow_the_task_card_rule() -> None:
    rows = {"p": {"TT-01@o4": {"r0": {"diag": {"S": 0.0, "failure_class": "format",
                                                "error_class": "format"}}}}}
    stub = SimpleNamespace(archive=SimpleNamespace(rows=rows))
    entry = {"verdict": "refuted", "target_units": ["TT-01@o4"], "gen": 1,
             "hypothesis": {"mechanism": "send the counts twice"}}
    silo = L.EvoRun._after_diag(stub, "p", ["TT-01@o4"])
    assert silo == {"class": "format", "n_agents": None, "runs": 1,
                    "own_shard_only": 0, "answer_seen": 0}
    line = ep._after_lines([dict(entry, after_diag=silo)], set())[0]
    assert line.endswith("class=format, own_shard_only_agents=n/a, answer_seen_agents=n/a")
    with use_task(toy_task()):
        toy = L.EvoRun._after_diag(stub, "p", ["TT-01@o4"])
        line = ep._after_lines([dict(entry, after_diag=toy)], set())[0]
    assert toy["own_shard_only"] is None and toy["answer_seen"] is None
    assert line.endswith("after the change: class=format")


# --------------------------------------------------------------------------- #
# execution, traces, instances, thresholds, split
# --------------------------------------------------------------------------- #


def test_toy_scoring_hooks_and_worker_env_reach_the_sandbox_run(tmp_path: Path,
                                                                 monkeypatch) -> None:
    import queenbee.program.execute as ex

    seen: dict[str, Any] = {}
    envs: list[dict[str, str]] = []

    def score_fn(facts: dict, *, instance: Any, output: Any, payload: Any, score: Any) -> dict:
        seen["score"] = (instance.case_id, output is not None, payload["information_goal"])
        return dict(facts) | {"S": 0.123, "toy_scored": True}

    def diag_fn(output: Any, instance: Any, facts: dict, **kw: Any) -> dict:
        seen["diag"] = (facts.get("S"), kw.get("goal"))
        return {"S": facts.get("S"), "failure_class": "consensus-wrong", "toy_err": 7}

    class Recording(ex.CodeProcessRunner):
        def __init__(self, *args: Any, **kw: Any) -> None:
            envs.append(dict(kw.get("extra_env") or {}))
            super().__init__(*args, **kw)

    monkeypatch.setattr(ex, "CodeProcessRunner", Recording)
    instance = _load_toy(_write_instances(tmp_path / "inst", ("TT-01",)) / "TT-01.json")
    req = {"cfg": SimpleNamespace(llm_provider="fake", worker_model="w", request_timeout=120.0,
                                  python_max_rounds=64),
           "job": SimpleNamespace(program_id="v1", unit_id="TT-01@o4"), "source": TOY_SEED,
           "instance": instance, "seed": 1, "artifacts_dir": tmp_path / "artifacts"}
    toy = toy_task(score_fn=score_fn, diag_fn=diag_fn, worker_env={"TOY_FLAG": "1"})
    with use_task(toy):
        assert C.task_score_hooks() == {"score_fn": score_fn, "diag_fn": diag_fn}
        row = C.task_execute(req)
    assert row["S"] == 0.123 and row["toy_scored"] is True and row["diag"]["toy_err"] == 7
    assert seen == {"score": ("TT-01", True, "sink"), "diag": (0.123, "sink")}
    assert envs == [{"TOY_FLAG": "1"}]
    with use_task(toy_task(execute=lambda r: {"S": 0.5, "by": "toy"})):
        assert C.task_execute(req) == {"S": 0.5, "by": "toy"}
    payload = {"worker_llm": {"provider": "fake"}}
    assert CodeProcessRunner(extra_env={"TOY_FLAG": "1"})._child_env(payload)["TOY_FLAG"] == "1"
    assert "TOY_FLAG" not in CodeProcessRunner()._child_env(payload)
    assert ex.task_worker_env() == {}


def test_toy_trace_row_hook_runs_after_the_credit_row(tmp_path: Path) -> None:
    calls: list[tuple] = []

    def hook(row: dict, trace: Any, instance: Any, *, source: Any, goal: Any) -> dict:
        calls.append((trace["case_id"], instance.case_id, source == SEED, goal))
        return dict(row) | {"toy_trace": True}

    instance = _load_toy(_write_instances(tmp_path / "inst", ("TT-01",)) / "TT-01.json")
    trace = {"case_id": "TT-01", "output": None,
             "facts": {"S": 0.25, "C": 10.0, "diag": {"S": 0.25, "failure_class": "format"}}}
    assert "toy_trace" not in cr.trace_row(trace, instance, source=SEED)
    with use_task(toy_task(trace_row=hook)):
        row = cr.trace_row(trace, instance, source=SEED, goal=C.task_goal())
    assert row["toy_trace"] is True and row["diag"]["failure_class"] == "format"
    assert calls == [("TT-01", "TT-01", True, "sink")]


def test_toy_instances_thresholds_and_split(tmp_path: Path) -> None:
    import queenbee.evaluate as ev

    inst = _write_instances(tmp_path / "inst")
    with use_task(toy_task(inst)):
        pool = L.UnitPool({}, SimpleNamespace(T=("TT-01",), V=("TT-04",)),
                          benchmarks_dir=tmp_path / "no_bench")
        assert pool.path("TT-01@o4") == inst / "TT-01.json" and pool.n_agents("TT-01@o4") == 4
        instance, sha = pool.get("TT-01@o4")
        assert instance.meta["toy"] is True and sha == ev.instance_sha256(instance)
        assert ev.load_instance_file(inst / "TT-02.json").meta["toy"] is True
        assert ev.split_templates("test") == TOY_TEST and ev.split_templates("train") == TOY_DEV
        th = R.load_thresholds(None)
        assert (th.tau_h, th.tau_2, th.source) == (0.15, 0.1, "toy")
        split = L.default_split({}, 7)
        assert (split.name, list(split.T), list(split.V)) == \
            ("toy-split", TOY_SPLIT["T"], TOY_SPLIT["V"])
        with use_task(toy_task(inst, default_split=lambda seed: {"T": ["TT-90"], "V": ["TT-04"]})):
            with pytest.raises(L.LeakageGuardError, match="TT-90"):
                L.default_split({}, 1)
    with use_task(toy_task(inst, instance_path=lambda uid: None)):
        with pytest.raises(L.EvoStateError, match="no instance"):
            L.UnitPool({}, SimpleNamespace(T=("TT-01",), V=()),
                       benchmarks_dir=tmp_path).path("TT-01@o4")
    assert R.load_thresholds(None) == R.Thresholds.provisional()


# --------------------------------------------------------------------------- #
# command line, the run config a resume must match (config.json), end-to-end runs
# --------------------------------------------------------------------------- #


def test_toy_cli_defaults_task_choice_and_frozen_config(tmp_path: Path, monkeypatch) -> None:
    inst = _write_instances(tmp_path / "inst")
    toy = toy_task(inst)
    monkeypatch.setitem(tasks._REGISTRY, "toy", toy)
    base = ["--split-seed", "1", "--arm", "full", "--root", str(tmp_path / "r")]
    with use_task("silo"):
        args, task = tasks.parse_task_args(L.build_parser(), base)
        assert task is tasks.SILO_TASK and args.task == "silo"
        assert (args.budget, args.final_reserve, args.K, args.vacuous_guard) == \
            (L.DEFAULT_BUDGET, L.FINAL_RESERVE, L.K_DEFAULT, False)
        assert "task" not in L.config_from_args(args).frozen()
        args, task = tasks.parse_task_args(L.build_parser(), ["--task", "toy", *base])
        assert task is toy and get_task() is toy
        assert (args.budget, args.final_reserve, args.K, args.vacuous_guard) == (14.0, 4.0, 2, True)
        cfg = L.config_from_args(args)
        assert cfg.task == "toy" and cfg.frozen()["task"] == "toy" and cfg.vacuous_guard
        args, _ = tasks.parse_task_args(L.build_parser(),
                                        ["--task", "toy", *base, "--budget", "20", "-K", "3"])
        assert (args.budget, args.K) == (20.0, 3)
        with pytest.raises(SystemExit):
            tasks.parse_task_args(L.build_parser(), ["--task", "nope", *base])
        set_task("silo")
        with pytest.raises(L.EvoStateError, match="active"):
            L.EvoRun(L.EvoConfig(root=tmp_path / "x", fake=True, task="toy"))
        assert not (tmp_path / "x").exists()
        # --write-split through main(): the task's split from its own census
        with use_task(toy):
            census = tmp_path / "census.json"
            census.write_text(json.dumps(L.fake_census()))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps([{"case_id": f"{t}@o4", "path": str(inst / f"{t}.json")}
                                        for t in TOY_DEV]))
        doc = L.main(["--task", "toy", "--split-seed", "3", "--write-split",
                      str(tmp_path / "split.json"), "--units-from", str(census),
                      "--ladder-manifest", str(manifest)])
        assert (doc["T"], doc["V"]) == (TOY_SPLIT["T"], TOY_SPLIT["V"])


def test_toy_task_runs_the_loop_end_to_end(tmp_path: Path, monkeypatch) -> None:
    inst = _write_instances(tmp_path / "inst")
    executor = ToyExecutor()
    toy = toy_task(inst, execute=executor)
    monkeypatch.setitem(tasks._REGISTRY, "toy", toy)
    requests: list[dict[str, Any]] = []
    real_request = mint.build_planner_request

    def recording(**kw: Any) -> Any:
        requests.append(dict(kw))
        return real_request(**kw)

    monkeypatch.setattr(mint, "build_planner_request", recording)
    root = tmp_path / "root"
    with use_task("toy"):
        cfg = L.EvoConfig(root=root, arm="full", split_seed=1, budget=14.0, final_reserve=4.0,
                          fake=True, K=2, planner_model="p", worker_model="w", traces=False,
                          planner_backoff_s=0.0, infra_backoff_s=0.0, parallel_cases=1,
                          outage_wait_s=0.0, planner_concurrency=2)
        summary = L.run_evo(cfg, L.EvoDeps(planner_client=L.FakeEvoPlanner()))
        toy_fp = C.behavior_fingerprint(TOY_SEED)
        worker_cfg = R.RaceConfig(fake=True, worker_model="w").worker_cfg_sha()
        again = L.run_evo(L.EvoConfig(**{**cfg.__dict__, "resume": True}),
                          L.EvoDeps(planner_client=L.FakeEvoPlanner()))
    assert summary["run_status"] == "ok" and again["champion"] == summary["champion"]
    assert summary["leak_hits"] == [] and summary["spent"]["total"] <= 14.0
    config = json.loads((root / "config.json").read_text())
    assert config["task"] == "toy"
    # every execution ran through the task's executor, on its own units only
    records = TL._jsonl(root / "executions.jsonl")
    paid = [r for r in records if not r.get("external")]
    assert paid and len(executor.units) == sum(1 for r in paid if not r["cached"])
    assert {r["unit_id"] for r in records} <= {f"{t}@o4" for t in TOY_DEV}
    assert {r["worker_cfg"] for r in records} == {worker_cfg}
    assert worker_cfg != R.RaceConfig(fake=True, worker_model="w").worker_cfg_sha()
    programs = json.loads((root / "race_programs.json").read_text())
    assert programs["v1"]["fp"] == toy_fp != C.behavior_fingerprint(TOY_SEED)
    assert requests and {(r["n_agents"], r["goal"]) for r in requests} == {(4, "sink")}
    # the screen ran at the task's team size
    gen0 = json.loads((root / "state" / "generation_000.json").read_text())
    screened = [s["screen"] for s in gen0["slots"] if (s.get("screen") or {}).get("fp_key")]
    assert screened and all(sc["fp_key"].startswith("n4=") and "|" not in sc["fp_key"]
                            for sc in screened)
    assert all(sc["readable5"] == [] and sc["pred_calls5"] == 0 for sc in screened)
    # planner prompts carry the task's texts and the n=4 budget line
    caps = PythonRunBudgets.for_rounds(64, n_agents=4)
    prompts = sorted((root / "prompts").glob("*_prompt.txt"))
    assert prompts
    for path in prompts:
        text = path.read_text()
        assert text.startswith(TOY_HEADER) and TOY_LEGEND in text and "TOY CARD NOTE" in text
        assert f"max_model_calls={caps.max_model_calls}," in text
        assert ep._HEADER not in text and ep._UNIT_LEGEND not in text
        assert TOY_GLOSS["consensus-wrong"] in text
        assert "[TT-01] Toy maximum TT-01" in text
        assert not re.search(r"(?<![\w-])(TT-04|TT-06|TT-9\d)(?!\d)", text)
    hyps = [json.loads(path.read_text()).get("hypothesis") or {}
            for path in sorted((root / "mints").glob("*/result.json"))]
    targets = [u for h in hyps for u in h.get("target_units") or []]
    assert targets and set(targets) <= {f"{t}@o4" for t in TOY_SPLIT["T"]}
    champion = json.loads((root / "champion.json").read_text())
    assert champion["thresholds"]["source"] == "toy" and champion["thresholds"]["tau_h"] == 0.15
    # a resume under another task is refused, in both directions
    with pytest.raises(L.EvoStateError, match="'task'"):
        L.EvoRun(L.EvoConfig(**{**cfg.__dict__, "resume": True, "task": None}))
    other = tmp_path / "silo_named"
    shutil.copytree(root, other)
    frozen = json.loads((other / "config.json").read_text())
    frozen.pop("task")
    (other / "config.json").write_text(json.dumps(frozen))
    with use_task("toy"):
        with pytest.raises(L.EvoStateError, match="'task'"):
            L.EvoRun(L.EvoConfig(**{**cfg.__dict__, "root": other, "resume": True}))


def test_evaluate_runs_the_toy_task_and_refuses_mixing(tmp_path: Path, monkeypatch) -> None:
    import queenbee.evaluate as ev
    import test_evaluate as TE

    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "QB_TRACE_DIR", "QUEENBEE_WORKER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    inst = _write_instances(tmp_path / "inst")

    def score_fn(facts: dict, **_kw: Any) -> dict:
        return dict(facts) | {"S": 0.25, "success": False}

    monkeypatch.setitem(tasks._REGISTRY, "toy", toy_task(inst, score_fn=score_fn))
    out = tmp_path / "eval.json"
    with use_task("silo"):
        assert ev.main(["--task", "toy", "--split", "test", "--seed-program", "seed",
                        "--llm", "fake", "--repeats", "1", "--parallel-cases", "2",
                        "--out", str(out)]) == 0
        result = json.loads(out.read_text())
        fp = result["config"]["fingerprint"]
        assert (fp["task"], fp["agents"], fp["goal"]) == ("toy", 4, "sink")
        rows = result["arms"]["seed"]["repeats"]["0"]["rows"]
        assert sorted(r["case_id"] for r in rows) == list(TOY_TEST)
        assert {r["S"] for r in rows} == {0.25}
        meta = result["arms"]["seed"]["meta"]
        assert meta["seed"] == "toy" and meta["source_sha256"] == _sha(TOY_SEED)
        bench = TE.write_bench(tmp_path / "bench", n_agents=4)
        with pytest.raises(SystemExit, match="fingerprint"):
            ev.main(["--split", "test", "--agents", "4", "--seed-program", "seed", "--llm",
                     "fake", "--repeats", "1", "--benchmarks-dir", str(bench),
                     "--out", str(out)])
