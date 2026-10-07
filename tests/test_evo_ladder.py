"""Tests for ``queenbee.evo.ladder`` (the development rungs): TEST guard, x5
sizing, upstream schema, independent gold check, adapter / manifest round
trip, and the pinned bytes of the 18 generated x5 instances."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import queenbee
from queenbee.evo import common as C
from queenbee.evo import credit as CR
from queenbee.evo import ladder as L
from queenbee.paths import default_benchmarks_dir

needs_bench = pytest.mark.skipif(
    not (default_benchmarks_dir() / "I-01_n5.json").is_file(),
    reason="Silo-Bench benchmarks not found (set SILO_BENCH_DIR)",
)

#: sha256 prefixes (file bytes, ``instance_sha256``) of the x5 instances
#: built with the default salt.
X5_SHA256: dict[str, tuple[str, str]] = {
    "I-01": ("7df29cb88c5efb66", "87ee909b49af511e"),
    "I-02": ("fd8332d77dcba5c7", "794bbd6982296167"),
    "I-03": ("cc8519e4ad3e34a4", "6f5a9c489f310c95"),
    "I-06": ("0c2253b041cbb11a", "6f68113157f74dbb"),
    "I-07": ("7220535b696decde", "cfa896e8075bd46f"),
    "I-08": ("80780fa148f38cc9", "8aff57c4c0b076b6"),
    "II-11": ("be670fd0eb2b6c4b", "b0d92fa7170c2b1a"),
    "II-12": ("1abe1bfda0bf27a8", "27fc8626bb3495e6"),
    "II-13": ("0acb3e9751d8ee24", "2d4e38f5db1ae8f1"),
    "II-16": ("c083a28d3456ce81", "c480a22bb47ca618"),
    "II-17": ("d3b12d87904ef2e5", "0dcc30611be9826c"),
    "II-18": ("c4001d60244f9292", "53e400b90c99be72"),
    "III-21": ("a9d1d5dba133ab4b", "ea750aa876b33944"),
    "III-22": ("4aaee584d39ad191", "a8bbb49a1ef7455c"),
    "III-23": ("8c25fe164334004d", "a6edcf837ff113e8"),
    "III-26": ("90137ed53b085c04", "6a80045ab101091c"),
    "III-27": ("6faa73ea2a36fde3", "441f6662ea685c47"),
    "III-28": ("ce2d294e073b7c92", "076eec8a0f887a6c"),
}


@pytest.fixture(scope="module")
def x5_dir(tmp_path_factory) -> Path:
    if not (default_benchmarks_dir() / "I-01_n5.json").is_file():
        pytest.skip("Silo-Bench benchmarks not found (set SILO_BENCH_DIR)")
    out = tmp_path_factory.mktemp("ladder")
    L.build_ladder(rung="x5", n_agents=5, out_dir=out)
    return out


# --------------------------------------------------------------------------- #
# TEST guard
# --------------------------------------------------------------------------- #


def test_test_ids_are_the_shared_holdout():
    # every 5-id block of the sorted 30 templates gives its last two to TEST
    stride = tuple(t for i, t in enumerate(sorted(L.ALL_TEMPLATE_IDS)) if i % 5 in (3, 4))
    assert L.TEST_TEMPLATE_IDS == tuple(C.TEST_IDS) == tuple(CR.TEST_CASE_IDS) == stride
    assert L.DEV_TEMPLATE_IDS == tuple(C.DEV_IDS)
    assert len(L.DEV_TEMPLATE_IDS) == 18
    assert not set(L.DEV_TEMPLATE_IDS) & set(L.TEST_TEMPLATE_IDS)
    assert set(L.BUILDERS) == set(L.DEV_TEMPLATE_IDS) == set(L.SPEC)


@pytest.mark.parametrize("test_id", L.TEST_TEMPLATE_IDS)
def test_every_entry_point_refuses_test_templates(test_id, tmp_path):
    with pytest.raises(L.LadderLeakError):
        L.assert_dev_template(test_id)
    with pytest.raises(L.LadderLeakError):
        L.assert_dev_template(f"{test_id}@x5")
    with pytest.raises(L.LadderLeakError):
        L.unit_id(test_id, "x5")
    with pytest.raises(L.LadderLeakError):
        L.build_raw(test_id, 100, 5, salt=None)
    with pytest.raises(L.LadderLeakError):
        L.build_ladder(["II-11", test_id], rung="x5", out_dir=tmp_path)
    with pytest.raises(L.LadderLeakError):
        L.build_ladder([test_id], rung="o5", out_dir=tmp_path)
    assert not any(tmp_path.iterdir())  # guard fires before anything is written


def test_unknown_template_and_bad_rungs_refused(tmp_path):
    with pytest.raises(L.LadderLeakError):
        L.assert_dev_template("IV-31")
    with pytest.raises(ValueError):
        L.parse_rung("m5")
    with pytest.raises(ValueError):
        L.build_ladder(["II-11"], rung="x5", n_agents=10, out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# x5 = twice the upstream per-agent data
# --------------------------------------------------------------------------- #


@needs_bench
def test_upstream_per_agent_size_is_k_at_every_n():
    """Upstream ties data size to team size: every shipped shard holds k items
    (graph templates: about k on average -- duplicate edges are dropped and
    the remainder goes to the last agent)."""

    for template in L.DEV_TEMPLATE_IDS:
        k = L.SPEC[template]["k"]
        for n in (5, 10):
            data = json.loads(L.upstream_path(template, n).read_text())
            sizes = [_shard_size(ac["input_shard"]) for ac in data["agent_configs"]]
            assert len(sizes) == n
            if template in ("III-23", "III-28"):
                assert abs(sum(sizes) / n - k) <= 0.15 * k, (template, n, sizes)
            else:
                assert sizes == [k] * n, (template, n, sizes)


def _shard_size(shard):
    if isinstance(shard, dict) and "users" in shard:
        return len(shard["users"]) + len(shard["items"])
    return len(shard)


def test_x5_doubles_the_per_agent_data(x5_dir):
    for template in L.DEV_TEMPLATE_IDS:
        k = L.SPEC[template]["k"]
        assert L.total_size(template, "x5") == 10 * k
        doc = json.loads((x5_dir / "x5" / f"{template}_x5.json").read_text())
        sizes = [_shard_size(ac["input_shard"]) for ac in doc["agent_configs"]]
        assert len(sizes) == 5
        assert sum(sizes) == 10 * k, template
        assert min(sizes) >= 2 * k - 1, (template, sizes)


# --------------------------------------------------------------------------- #
# schema, gold, loading
# --------------------------------------------------------------------------- #


def test_x5_files_have_the_upstream_schema_and_verified_gold(x5_dir):
    for template in L.DEV_TEMPLATE_IDS:
        doc = json.loads((x5_dir / "x5" / f"{template}_x5.json").read_text())
        up = json.loads(L.upstream_path(template, 5).read_text())
        assert list(doc) == list(up)
        assert list(doc["metadata"]) == list(up["metadata"])
        assert list(doc["expected_output"]) == list(up["expected_output"])
        assert [list(a) for a in doc["agent_configs"]] == [list(a) for a in up["agent_configs"]]
        assert doc["case_id"] == f"{template}@x5"
        assert doc["case_name"] == up["case_name"]
        assert doc["metadata"]["is_segmented"] == up["metadata"]["is_segmented"]
        L.verify_instance(doc)  # independent checker + Silo-Bench scorer


def test_x5_statement_is_the_upstream_statement_with_only_size_numbers_changed(x5_dir):
    """x5 differs from o5 in data size only: the statement is upstream's,
    character for character, except the declared size slots."""

    changed = {}
    for template in L.DEV_TEMPLATE_IDS:
        up = json.loads(L.upstream_path(template, 5).read_text())["task_description"]
        x5 = json.loads((x5_dir / "x5" / f"{template}_x5.json").read_text())["task_description"]
        assert re.sub(r"\d+", "#", x5) == re.sub(r"\d+", "#", up), template
        diffs = [(a, b) for a, b in zip(re.findall(r"\d+", up), re.findall(r"\d+", x5)) if a != b]
        if diffs:
            changed[template] = diffs
    assert changed == {
        "II-18": [("40", "80")],
        "III-22": [("125", "250"), ("62", "125")],
        "III-23": [("50", "100")],
        "III-28": [("40", "80"), ("40", "80")],
    }
    # spot checks: upstream details such as I-03's answer example and
    # III-27's scoring formula and output format are kept verbatim
    i03 = json.loads((x5_dir / "x5" / "I-03_x5.json").read_text())["task_description"]
    assert '(e.g., "Candidate_A")' in i03
    iii27 = json.loads((x5_dir / "x5" / "III-27_x5.json").read_text())["task_description"]
    assert "score(user, item) = dot_product(user_vector, item_vector)" in iii27
    assert "List of top 3 tuples: [(user_id, item_id, score), ...]" in iii27


@needs_bench
def test_statement_slots_agree_with_the_vendored_generator():
    for template in L.STATEMENT_SIZE_SLOTS:
        raw = L.build_raw(template, L.total_size(template, "x5"), 5, salt="dev-v1")
        up = json.loads(L.upstream_path(template, 5).read_text())["task_description"]
        L.sized_statement(template, up, raw["metadata"], raw_desc=raw["task_description"])
        bad = dict(raw["metadata"], total_length=1, num_nodes=1)
        if re.search(r"Total nodes|Total elements", raw["task_description"]):
            with pytest.raises(L.LadderError):
                L.sized_statement(template, up, bad, raw_desc=raw["task_description"])


@needs_bench
@pytest.mark.parametrize("n", [5, 10])
def test_user_prompts_render_shards_like_upstream(n):
    for template in L.DEV_TEMPLATE_IDS:
        up = json.loads(L.upstream_path(template, n).read_text())
        for ac in up["agent_configs"]:
            prompt = L.fill(up["task_description"], agent_id=ac["agent_id"], num_agents=n,
                            max_id=n - 1, input_shard=L.render_shard(template, ac["input_shard"]))
            assert prompt == ac["user_prompt"], (template, ac["agent_id"])


def test_checker_catches_a_corrupted_gold(x5_dir):
    doc = json.loads((x5_dir / "x5" / "I-01_x5.json").read_text())
    for ac in doc["agent_configs"]:
        ac["expected_output"] += 1
    doc["expected_output"]["per_agent_values"] = [ac["expected_output"] for ac in doc["agent_configs"]]
    with pytest.raises(L.LadderError):
        L.verify_instance(doc)


def test_iii22_statement_names_the_median_position(x5_dir):
    doc = json.loads((x5_dir / "x5" / "III-22_x5.json").read_text())
    assert "Total elements: 250, median at position 125." in doc["task_description"]
    flat = [v for ac in doc["agent_configs"] for v in ac["input_shard"]]
    assert doc["agent_configs"][0]["expected_output"] == sorted(flat)[125]
    assert "median at position 125" in doc["agent_configs"][3]["user_prompt"]


def test_adapter_loads_the_rung_dir_and_manifest_resolves(x5_dir):
    from queenbee.bench.silo_bench import SiloBenchAdapter
    from queenbee.evaluate import (
        instance_sha256,
        load_manifest,
        resolve_manifest_instances,
    )

    insts = {i.case_id: i for i in SiloBenchAdapter(x5_dir / "x5").iter_instances()}
    assert set(insts) == {f"{t}@x5" for t in L.DEV_TEMPLATE_IDS}
    for inst in insts.values():
        assert inst.n_agents == 5
        assert "Communication Protocol" not in inst.task_prompt
        assert len(inst.meta["expected_outputs"]) == 5
    entries = load_manifest(x5_dir / "manifest_x5.json")
    assert [e["case_id"] for e in entries] == [f"{t}@x5" for t in L.DEV_TEMPLATE_IDS]
    instances, meta = resolve_manifest_instances(entries, {}, 5)
    for e in entries:
        assert instance_sha256(instances[e["case_id"]]) == e["instance_sha256"]
        assert meta[e["case_id"]]["template"] == e["template"]
        assert e["rung"] == "x5" and e["salt"] == L.DEFAULT_SALT
    assert L.load_ladder_manifest(x5_dir / "manifest_x5.json") == entries


@needs_bench
def test_o_rungs_point_at_upstream_with_evaluator_hashes(tmp_path):
    from queenbee.evaluate import instance_sha256, load_instance_file

    entries = L.build_ladder(["I-01", "III-22"], rung="o10", n_agents=10, out_dir=tmp_path)
    assert not (tmp_path / "o10").exists()  # nothing generated for o-rungs
    for e in entries:
        path = L.upstream_path(e["template"], 10)
        assert Path(e["path"]) == path.resolve()
        assert e["instance_sha256"] == instance_sha256(load_instance_file(path))
        assert e["case_id"] == f"{e['template']}@o10"


@needs_bench
def test_o_rung_total_size_is_the_actual_item_count(tmp_path):
    entries = {e["template"]: e for e in L.build_ladder(
        ["I-01", "III-23", "III-27", "III-28"], rung="o5", out_dir=tmp_path)}
    assert {t: e["total_size"] for t, e in entries.items()} == {
        "I-01": 100, "III-23": 98, "III-27": 30, "III-28": 74}


def test_x5_manifest_sizes_are_the_file_sizes(x5_dir):
    for e in json.loads((x5_dir / "manifest_x5.json").read_text()):
        doc = json.loads((x5_dir / e["path"]).read_text())
        assert e["total_size"] == sum(L.shard_size(ac["input_shard"]) for ac in doc["agent_configs"])
        assert e["total_size"] == L.total_size(e["template"], "x5")


@needs_bench
def test_an_existing_instance_is_never_silently_replaced(tmp_path):
    L.build_ladder(["II-11"], rung="x5", out_dir=tmp_path)
    path = tmp_path / "x5" / "II-11_x5.json"
    before = path.read_bytes()
    L.build_ladder(["II-11"], rung="x5", out_dir=tmp_path)  # same content: fine
    assert path.read_bytes() == before
    with pytest.raises(L.LadderError, match="refusing to overwrite"):
        L.build_ladder(["II-11"], rung="x5", out_dir=tmp_path, salt="dev-v2")
    assert path.read_bytes() == before
    L.build_ladder(["II-11"], rung="x5", out_dir=tmp_path, salt="dev-v2", overwrite=True)
    assert path.read_bytes() != before


@needs_bench
def test_manifest_loader_guards_the_instance_it_points_at(tmp_path):
    entries = L.build_ladder(["II-11", "III-21"], rung="x5", out_dir=tmp_path)
    manifest = tmp_path / "manifest_x5.json"
    # a dev id pointing at another dev template's file
    swapped = [dict(entries[0], path=entries[1]["path"], instance_sha256=None)]
    (tmp_path / "swapped.json").write_text(json.dumps(swapped))
    with pytest.raises(L.LadderError, match="holds instance"):
        L.load_ladder_manifest(tmp_path / "swapped.json")
    # a dev id pointing at a file whose own case id is a TEST template
    doc = json.loads((tmp_path / "x5" / "II-11_x5.json").read_text())
    doc["case_id"] = f"{L.TEST_TEMPLATE_IDS[0]}@x5"
    (tmp_path / "leak.json").write_text(json.dumps(doc))
    (tmp_path / "leak_manifest.json").write_text(
        json.dumps([dict(entries[0], path="leak.json", instance_sha256=None)]))
    with pytest.raises(L.LadderLeakError):
        L.load_ladder_manifest(tmp_path / "leak_manifest.json")
    good = L.load_ladder_manifest(manifest)
    assert [e["case_id"] for e in good] == [e["case_id"] for e in entries]


@needs_bench
def test_same_salt_is_byte_stable_and_salt_changes_the_data(tmp_path):
    a = L.build_ladder(["II-11", "III-27"], rung="x5", out_dir=tmp_path / "a")
    b = L.build_ladder(["II-11", "III-27"], rung="x5", out_dir=tmp_path / "b")
    c = L.build_ladder(["II-11", "III-27"], rung="x5", out_dir=tmp_path / "c", salt="dev-v2")
    assert [e["instance_sha256"] for e in a] == [e["instance_sha256"] for e in b]
    assert (tmp_path / "a/x5/II-11_x5.json").read_bytes() == (tmp_path / "b/x5/II-11_x5.json").read_bytes()
    assert all(x["instance_sha256"] != y["instance_sha256"] for x, y in zip(a, c))


def test_generated_data_differs_from_the_shipped_o5_instance(x5_dir):
    for template in L.DEV_TEMPLATE_IDS:
        doc = json.loads((x5_dir / "x5" / f"{template}_x5.json").read_text())
        up = json.loads(L.upstream_path(template, 5).read_text())
        assert doc["agent_configs"][0]["input_shard"] != up["agent_configs"][0]["input_shard"]


# --------------------------------------------------------------------------- #
# pinned x5 bytes, manifest provenance, import order
# --------------------------------------------------------------------------- #


def test_x5_instances_are_byte_stable(x5_dir):
    """The default-salt x5 files and their instance hashes never drift."""

    entries = {e["template"]: e for e in json.loads((x5_dir / "manifest_x5.json").read_text())}
    assert set(entries) == set(X5_SHA256) == set(L.DEV_TEMPLATE_IDS)
    for template, (file_sha, inst_sha) in X5_SHA256.items():
        data = (x5_dir / "x5" / f"{template}_x5.json").read_bytes()
        assert hashlib.sha256(data).hexdigest()[:16] == file_sha, template
        assert entries[template]["instance_sha256"][:16] == inst_sha, template


def test_manifest_generator_block(x5_dir):
    for e in json.loads((x5_dir / "manifest_x5.json").read_text()):
        assert e["salt"] == L.DEFAULT_SALT == "dev-v1"
        assert set(e["generator"]) == {"version", "source_sha256"}
        assert e["generator"]["version"] == L.GENERATOR_VERSION
        assert len(e["generator"]["source_sha256"]) == 64


@pytest.mark.parametrize("first", ["common", "ladder", "screen", "units", "credit", "seed", "diagnosis"])
def test_modules_import_in_any_order(first):
    """ladder and credit read the TEST tuple from common at import time; no
    import order may hit a partially initialised module."""

    rest = [m for m in ("common", "ladder", "screen", "units", "credit", "seed", "diagnosis")
            if m != first]
    code = "; ".join(f"import queenbee.evo.{m}" for m in [first] + rest)
    src = str(Path(queenbee.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
