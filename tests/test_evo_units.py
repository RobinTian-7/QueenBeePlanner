"""Tests for ``queenbee.evo.units``: mapping result rows to units, unit classes,
T / V split generation, trace evidence, and the TEST guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from queenbee.evo import ladder as L
from queenbee.evo import units as U
from queenbee.paths import default_benchmarks_dir


def _row(case_id: str, S: float, *, infra: str | None = None, sha: str | None = None) -> dict:
    row = {"case_id": case_id, "S": S, "infra": infra, "success": S >= 1.0}
    if sha:
        row["instance_sha256"] = sha
    return row


def _doc(agents: int, arms: dict[str, list[list[dict]]]) -> dict:
    """A result document in the ``queenbee.evaluate`` layout: arm -> repeats -> rows."""

    return {
        "config": {"agents": agents, "case_meta": {}},
        "arms": {
            arm: {"repeats": {str(i): {"rows": rows} for i, rows in enumerate(reps)}}
            for arm, reps in arms.items()
        },
    }


# --------------------------------------------------------------------------- #
# unit classes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "v1,o5,structural,want",
    [
        ([0.4, 0.6], None, False, "frontier"),
        ([1.0, 0.8], None, False, "frontier"),
        ([1.0, 1.0], None, False, "guard"),
        ([1.0, 1.0, 1.0], None, False, "guard"),
        ([1.0], None, False, "unresolved"),  # one rep never makes a guard
        ([0.0], True, False, "frontier"),  # v1 solves the template at o5
        ([0.0, 0.0], False, True, "frontier"),  # organizational failures: headroom
        # a zero unit with no other evidence is never called dead
        ([0.0], False, False, "unresolved"),
        ([0.0, 0.0], None, False, "unresolved"),
        ([], True, True, "unresolved"),
    ],
)
def test_classify_unit_from_v1_rows_only(v1, o5, structural, want):
    assert U.classify_unit(v1, {}, o5_solved=o5, structural=structural) == want
    assert U.classify_unit(v1, None, o5_solved=o5, structural=structural) == want


@pytest.mark.parametrize(
    "v1,refs,o5,want",
    [
        ([1.0, 1.0, 1.0], {"other": [0.0]}, None, "guard"),
        ([0.0, 0.0], {"other": [0.2]}, None, "frontier"),
        ([0.0, 0.0], {"other": [0.0], "third": [0.0]}, None, "dead"),
        ([0.0], {"other": [0.0]}, False, "dead"),
        ([0.0], {"other": [0.0]}, True, "frontier"),
        ([0.0], {"other": []}, False, "unresolved"),
        ([], {"other": [1.0]}, None, "unresolved"),
    ],
)
def test_classify_unit_with_rows_of_other_arms(v1, refs, o5, want):
    assert U.classify_unit(v1, refs, o5_solved=o5) == want


def test_load_units_maps_labels_rungs_and_drops_infra(tmp_path):
    (tmp_path / "II-11_n5.json").write_text("{}")  # only its existence is read
    a0 = _doc(5, {
        "v1": [[_row("II-11", 1.0), _row("III-22", 0.6)],
               [_row("II-11", 1.0), _row("III-22", 0.0, infra="gateway 502")]],
    })
    census = _doc(5, {"v1": [[_row("II-11@x5", 0.0), _row("III-22@x5", 1.0)]]})
    o10 = _doc(10, {"v1": [[_row("II-11", 0.9)]]})
    units = U.load_units([a0, census, o10], benchmarks_dir=tmp_path)
    assert set(units) == {"II-11@o5", "III-22@o5", "II-11@x5", "III-22@x5", "II-11@o10"}
    assert units["II-11@o5"].klass == "guard"
    assert units["II-11@o5"].v1_S == (1.0, 1.0)
    assert units["II-11@o5"].ref_S == ()
    assert units["III-22@o5"].v1_S == (0.6,)  # infra rep dropped
    assert units["III-22@o5"].klass == "frontier"
    assert units["II-11@x5"].klass == "frontier"  # 0 at x5, v1 solved it at o5
    assert units["II-11@x5"].o5_solved is True
    assert units["III-22@x5"].klass == "unresolved"  # single rep at 1.0
    assert units["III-22@x5"].o5_solved is False
    assert units["II-11@o10"].n_agents == 10 and units["II-11@o10"].exec_equivalents == 2.0
    assert units["II-11@o5"].instance_path == str(tmp_path / "II-11_n5.json")
    assert units["III-22@o5"].instance_path is None  # no instance file for it in tmp_path
    assert units["II-11@x5"].instance_path is None
    summary = U.units_summary(units)
    assert "II-11@x5" in summary["frontier"] and "II-11@o5" in summary["guard"]


def test_other_arms_count_as_references_except_positive_controls():
    assert U.reference_arms_of(["v1", "other", "v1_degraded", "Degraded_b"]) == ["other"]
    a0 = _doc(5, {"v1": [[_row("II-11", 0.0)]], "other": [[_row("II-11", 0.0)]],
                  "v1_degraded": [[_row("II-11", 0.8)]]})
    unit = U.load_units([a0])["II-11@o5"]
    assert dict(unit.ref_S) == {"other": (0.0,)}  # a positive control is no reference
    assert unit.klass == "dead"
    assert U.load_units([a0], reference_arms=[])["II-11@o5"].klass == "unresolved"


def test_references_count_per_unit_not_per_template():
    """A reference > 0 on III-27@o5 does not make a zero III-27@x5 frontier;
    with no x5 reference and v1 unsolved at o5, the x5 unit is unresolved,
    not dead."""

    a0 = _doc(5, {"v1": [[_row("III-27", 0.4)]], "other": [[_row("III-27", 1.0)]]})
    census = _doc(5, {"v1": [[_row("III-27@x5", 0.0)], [_row("III-27@x5", 0.0)]]})
    units = U.load_units([a0, census])
    assert units["III-27@o5"].klass == "frontier"
    assert units["III-27@x5"].o5_solved is False and units["III-27@x5"].ref_S == ()
    assert units["III-27@x5"].klass == "unresolved"
    census_ref = _doc(5, {"v1": [[_row("III-27@x5", 0.0)]], "other": [[_row("III-27@x5", 0.0)]]})
    assert U.load_units([a0, census_ref])["III-27@x5"].klass == "dead"


def test_manifest_units_exist_before_any_row(tmp_path):
    if not (default_benchmarks_dir() / "II-11_n5.json").is_file():
        pytest.skip("Silo-Bench benchmarks not found (set SILO_BENCH_DIR)")
    L.build_ladder(["II-11", "III-21"], rung="x5", out_dir=tmp_path)
    units = U.load_units([], manifests=[tmp_path / "manifest_x5.json"])
    assert set(units) == {"II-11@x5", "III-21@x5"}
    for u in units.values():
        assert u.klass == "unresolved"
        assert Path(u.instance_path).is_file() and u.instance_sha256


def test_one_unit_two_instances_is_refused():
    a = _doc(5, {"v1": [[_row("II-11@x5", 1.0, sha="a" * 64)]]})
    b = _doc(5, {"v1": [[_row("II-11@x5", 1.0, sha="b" * 64)]]})
    with pytest.raises(ValueError):
        U.load_units([a, b])


def test_x_rung_row_in_a_file_of_another_agent_count_is_refused():
    with pytest.raises(ValueError):
        U.load_units([_doc(10, {"v1": [[_row("II-11@x5", 1.0)]]})])


# --------------------------------------------------------------------------- #
# TEST guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("test_id", L.TEST_TEMPLATE_IDS)
def test_test_units_are_refused_everywhere(test_id):
    with pytest.raises(U.LadderLeakError):
        U.assert_dev(test_id)
    with pytest.raises(U.LadderLeakError):
        U.assert_dev(f"{test_id}@x5")
    with pytest.raises(U.LadderLeakError):
        U.make_unit_id(test_id, "o5")
    with pytest.raises(U.LadderLeakError):
        U.Unit(unit_id=f"{test_id}@o5", template=test_id, rung="o5", n_agents=5)
    with pytest.raises(U.LadderLeakError):  # a result file holding a TEST row
        U.load_units([_doc(5, {"v1": [[_row("II-11", 1.0), _row(test_id, 1.0)]]})])
    with pytest.raises(U.LadderLeakError):
        U.TemplateSplit("bad", T=("II-11",), V=(test_id,))


def test_assert_dev_passes_dev_units_through():
    unit = U.Unit(unit_id="II-11@x5", template="II-11", rung="x5", n_agents=5)
    assert U.assert_dev(unit) is unit
    assert U.assert_dev("III-27@o10") == "III-27@o10"
    with pytest.raises(ValueError):
        U.Unit(unit_id="II-11@x5", template="II-11", rung="x5", n_agents=10)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def _units(frontier: dict[str, str]) -> dict[str, U.Unit]:
    """Every dev template at o5: 'frontier' for the given ones, else guard."""

    out = {}
    for t in L.DEV_TEMPLATE_IDS:
        klass = frontier.get(t, "guard")
        v1 = (0.6, 0.6) if klass == "frontier" else (1.0, 1.0)
        out[f"{t}@o5"] = U.Unit(unit_id=f"{t}@o5", template=t, rung="o5", n_agents=5,
                                klass=klass, v1_S=v1)
    return out


_FRONTIER = ("I-01", "I-03", "I-07", "II-11", "II-12", "II-16", "II-18", "III-21", "III-22",
             "III-27", "III-28")


def test_make_split_meets_constraints_and_is_deterministic():
    units = _units({t: "frontier" for t in _FRONTIER})
    a, b = U.make_split(units, 1), U.make_split(units, 1)
    assert a == b and a.name == "s1"
    assert U.make_split(units, 2) != a or U.make_split(units, 3) != a
    for seed in (1, 2, 3):
        s = U.make_split(units, seed)
        assert len(s.V) == 6 and len(s.T) == 12
        for tier in L.TIERS:
            assert sum(L.tier_of(t) == tier for t in s.V) == 2
        assert len(U.frontier_units(units, templates=s.V)) >= 3
        assert len(U.frontier_units(units, templates=s.T)) >= 5
    with pytest.raises(ValueError):
        U.make_split(_units({}), 1, max_tries=50)


def test_split_draws_are_pinned_to_the_seed():
    """The split RNG key is part of the method: these draws never change."""

    units = _units({t: "frontier" for t in _FRONTIER})
    assert U.make_split(units, 1).V == ("I-02", "I-07", "II-13", "II-18", "III-21", "III-23")
    assert U.make_split(units, 2).V == ("I-01", "I-02", "II-12", "II-13", "III-26", "III-28")
    assert U.make_split(units, 3).V == ("I-01", "I-02", "II-16", "II-18", "III-23", "III-28")


def test_split_sides_and_their_guard():
    split = U.make_split(_units({t: "frontier" for t in _FRONTIER}), 1)
    assert set(split.T) | set(split.V) == set(L.DEV_TEMPLATE_IDS)
    assert split.side_of("II-11") == "T" and split.side_of("III-23@x5") == "V"
    split.assert_side("II-11@x5", "T")
    with pytest.raises(RuntimeError):
        split.assert_side("III-23@o5", "T")
    with pytest.raises(U.LadderLeakError):
        split.side_of(L.TEST_TEMPLATE_IDS[0])


def test_classify_unit_irreducible_and_structural():
    # near-miss evidence: the benchmark's unstated rounding decides -> never targeted
    assert U.classify_unit([0.0, 0.0], {"other": [0.0]}, near_miss=True) == "irreducible"
    assert U.classify_unit([0.2, 0.0], {}, near_miss=True) == "irreducible"
    assert U.classify_unit([1.0, 1.0], {}, near_miss=True) == "guard"
    # organizational failures where every other arm also scores 0 -> headroom
    assert U.classify_unit([0.0, 0.0, 0.0], {"other": [0.0]}) == "dead"
    assert U.classify_unit([0.0, 0.0, 0.0], {"other": [0.0]}, structural=True) == "frontier"
    assert U.classify_unit([0.0, 0.0, 0.0], {}, structural=True) == "frontier"


# --------------------------------------------------------------------------- #
# trace evidence (QB_TRACE_DIR dumps)
# --------------------------------------------------------------------------- #


def _trace(path: Path, case_id: str, n: int, answers: list, truth, S: float) -> None:
    import json

    subs = [{"agent_id": i, "answer": a, "submitted_round": 3} for i, a in enumerate(answers)]
    path.write_text(json.dumps({
        "case_id": case_id, "n_agents": n, "source_sha": "0" * 12,
        "facts": {"S": S, "infra": None},
        "output": {"submissions": subs, "messages": [], "rounds_executed": 4},
        "ground_truth": truth,
    }))


def test_trace_profiles_split_near_misses_from_organizational_failures(tmp_path):
    # III-22: precision near-misses (right value, other rounding) on 2 of 2 traces
    _trace(tmp_path / "a.json", "III-22", 5, [1.0004] * 5, 1.0, 0.0)
    _trace(tmp_path / "b.json", "III-22@x5", 5, ["1.0004"] * 5, 1.0, 0.0)
    # II-11@x5: plainly wrong answers -> organizational
    _trace(tmp_path / "c.json", "II-11@x5", 5, [7, 8, 9, 10, 11], 99, 0.0)
    # solved runs and TEST traces never count
    _trace(tmp_path / "d.json", "II-16", 5, [3] * 5, 3, 1.0)
    _trace(tmp_path / "e.json", L.TEST_TEMPLATE_IDS[0], 5, [1] * 5, 2, 0.0)
    profile = U.failure_profile_from_traces([tmp_path])
    assert profile == {"III-22@o5": {"failing": 1, "near_miss": 1},
                       "III-22@x5": {"failing": 1, "near_miss": 1},
                       "II-11@x5": {"failing": 1, "near_miss": 0}}
    assert U.near_miss_units_from_traces([tmp_path]) == {"III-22@o5": 1, "III-22@x5": 1}
    assert U.structural_units_from_traces([tmp_path]) == {"II-11@x5": 1}
