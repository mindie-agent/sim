"""A5 communication Pass-4 through Pass-9 review regression controls.

[Role] Verify Pass-4 through Pass-9 A5 communication calibration regressions.
[Invariants] Unconfirmed AICPU/AIV requests never promote to confirmed regimes,
  AICPU and AIV never alias, non-A5 hardware is rejected, and boundaries stay
  strictly isolated across ingestion, curve lookup, recipes, and regimes.
[Timeouts & Locks] In-memory and tmp_path tests only; no locks or unbounded waits.
[Side-Effect Policy] Pure unit/boundary tests; never initializes a device.
"""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.observations import a5_comm_ingest as ingest_module
from mindie_sim.observations.a5_comm_binding import (
    _A5CommPlan, _iter_json_events, _read_stream_value, _wire_factor,
)
from mindie_sim.observations.a5_comm_calibration import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU, A5CommCalibrationBundle,
    bind_service_curve, build_a5_comm_recipe, build_a5_followup_manifest,
    compute_a5_contention_slowdown, evaluate_a5_mc2_chains,
    fit_a5_service_curves, ingest_a5_campaign_data, resolve_expansion_regime,
    to_service_regime,
)
from mindie_sim.observations.calibration import (
    bind_service_curve as reexported_bind_service_curve,
)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.services import ResourceService, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from test_mechanism_contract import hardware_case
from test_a5_comm_calibration import (
    A5_SOURCE_CASE, COLLECTIVES, HOLDOUT_PAYLOADS, LIVE_A5, MIB, ROOT, STAGES,
    WORLD_SIZES, _a5_case, _a5_comm_op, _campaign, _comm_context, _fit_campaign,
    _mc2_chain, _measurement_op, _mode_conditions, _pipeline_campaign,
    _raw_point, _scope_for, embedded,
)

# ---------------------------------------------------------------------------
# 11. Regression controls for Pass-4 through Pass-11 review findings
# ---------------------------------------------------------------------------

def test_pass4_finding2_parallel_axis_split_changes_bound_identity(embedded):
    _, bundle = embedded
    base, changed = _a5_case(), _a5_case()
    base["runtime"].update(tp=2, dp=4, ep=8)
    changed["runtime"].update(tp=4, dp=2, ep=8)
    base_case, changed_case = parse_case(base), parse_case(changed)
    operation = _a5_comm_op(base)
    identity = service_identity(base_case, operation, "none")
    changed_identity = service_identity(changed_case, operation, "none")
    parallel = identity["transport"]["parallel"]

    assert set(parallel) == {
        "tp", "dp", "pp", "ep", "cp", "pcp", "sp"}
    assert (parallel["tp"], parallel["dp"]) == (2, 4)
    assert (changed_identity["transport"]["parallel"]["tp"],
            changed_identity["transport"]["parallel"]["dp"]) == (4, 2)
    assert identity != changed_identity
    curve = bundle.bind_service_curve(base_case, operation)
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(
            changed, (operation,),
            recipes=RecipeBook({operation.recipe: ResourceService(curve)}))


def test_pass4_finding3_explicit_placement_scope_never_falls_back(embedded):
    _, bundle = embedded

    assert bundle.lookup_curve(
        "all_gather", 8).identity["placement_scope"] == "intra_node_ub"
    assert bundle.lookup_curve(
        "all_gather", 16).identity["placement_scope"] == "cross_node_urma"
    cross = bundle.lookup_curve(
        "all_gather", 16, placement_scope="cross_node_urma")
    assert cross.identity["placement_scope"] == "cross_node_urma"
    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.lookup_curve(
            "all_gather", 16, placement_scope="intra_node_ub")
    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.lookup_curve(
            "all_gather", 8, placement_scope="cross_node_urma")
    with pytest.raises(
            ModelingError, match="a5_placement_scope_unknown:bogus"):
        bundle.lookup_curve(
            "all_gather", 8, placement_scope="bogus")
    keys, _ = bundle._candidates(
        "all_gather", 16, placement_scope="intra_node_ub")
    assert keys == [
        ("all_gather", 16, "intra_node_ub", "bf16", "unspecified")]

    raw = _a5_case()
    operation = _a5_comm_op(raw)
    invalid = replace(operation, metadata=operation.metadata + (
        ("placement_scope", "bogus"),))
    with pytest.raises(
            ModelingError, match="a5_placement_scope_unknown:bogus"):
        bundle.bind_service_curve(parse_case(raw), invalid)
    scoped = replace(operation, metadata=operation.metadata + (
        ("placement_scope", "intra_node_ub"),))
    bound = bundle.bind_service_curve(parse_case(raw), scoped)
    assert bound.identity["transport"]["placement_scope"] == "intra_node_ub"


def test_pass4_finding4_raw_points_keep_profiler_confirmed_regimes():
    document = _campaign(label="raw-regimes", conditions=[], raw_points=[
        _raw_point(0.5, ["AI_VECTOR_CORE"]),
        _raw_point(0.7, ["hcom_allReduce_AicpuKernel"])])

    ingested = ingest_a5_campaign_data(document)

    by_regime = {row["regime"]: row for row in ingested["conditions"]}
    assert set(by_regime) == {"aicpu", "aiv"}
    assert by_regime["aiv"]["per_op_ms"] == pytest.approx(0.5)
    assert by_regime["aicpu"]["per_op_ms"] == pytest.approx(0.7)
    assert all(row["profiler_confirmed"] for row in ingested["conditions"])
    assert by_regime["aiv"]["profiler_evidence"] == ["AI_VECTOR_CORE"]


def test_pass4_finding4_raw_points_keep_confirmation_in_identity():
    document = _campaign(label="raw-confirmed", conditions=[], raw_points=[
        _raw_point(0.5, requested="aicpu"),
        _raw_point(0.7, ["hcom_allReduce_AicpuKernel"], requested="aicpu"),
        _raw_point(0.9, observed="aicpu")])

    ingested = ingest_a5_campaign_data(document)

    by_state = {(row["regime"], row["unconfirmed_tag"],
                 row["profiler_confirmed"]): row
                for row in ingested["conditions"]}
    assert set(by_state) == {
        ("unspecified", "requested_aicpu_unconfirmed", False),
        ("aicpu", None, True), ("aicpu", None, False)}
    assert by_state[("aicpu", None, True)]["profiler_evidence"] == [
        "hcom_allReduce_AicpuKernel"]


def test_pass4_finding5_confirmed_curve_is_not_reported_as_fallback():
    bundle = _fit_campaign(_campaign(
        label="confirmed-aicpu",
        conditions=_mode_conditions(
            "aicpu", evidence="hcom_allReduce_AicpuKernel",
            intercept=0.05, slope=1e-9)))
    op = _measurement_op(family="allreduce")
    plan = bundle.build_a5_comm_recipe(
        "all_reduce", world_size=2, expansion_mode="aicpu").build(op, None)

    assert plan.metadata["curve_expansion_mode"] == "aicpu"
    assert plan.metadata["unspecified_fallback"] is False


@pytest.mark.parametrize("source", ["case", "document"])
def test_pass4_finding6_rank_rows_keep_profiler_only_confirmation(source):
    rank = {
        "role": "rank", "status": "completed", "label": "rank-profiler",
        "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
        "cases": [{
            "case_id": "probe", "collective": "all_reduce",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1}}]}
    evidence = ["hcom_allReduce_AicpuKernel"]
    if source == "document":
        rank["profiler_evidence"] = evidence
    else:
        rank["cases"][0]["profiler_evidence"] = evidence

    rank_rows = ingest_module._rank_rows(rank, "memory")
    assert rank_rows[0]["profiler_evidence"] == evidence
    ingested = ingest_a5_campaign_data(rank)
    row = ingested["conditions"][0]
    assert row["regime"] == "aicpu"
    assert row["profiler_confirmed"] is True
    assert row["expansion_unconfirmed"] is False
    assert row["profiler_evidence"] == evidence


def test_pass4_finding7_parent_repeat_index_is_not_discarded(tmp_path):
    def parent(repeat):
        return {
            "role": "parent", "label": "trials", "status": "completed",
            "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
            "arch_family": A5_ARCH_FAMILY, "world_size": 2,
            "dtype": "bf16", "placement": "intra-node-ub",
            "repeat_index": repeat}

    def rank(cases):
        return {
            "role": "rank", "world_size": 2, "dtype": "bf16",
            "placement": "intra-node-ub", "cases": cases}

    def write(name, repeat, cases):
        root = tmp_path / name
        root.mkdir()
        (root / "trials.parent.json").write_text(
            json.dumps(parent(repeat)), encoding="utf-8")
        (root / "trials.rank0.json").write_text(
            json.dumps(rank(cases)), encoding="utf-8")
        return root

    flat = {"case_id": "flat", "collective": "all_reduce",
            "payload_bytes": 1024, "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1}}
    expanded = {
        "case_id": "expanded", "collective": "all_reduce",
        "samples": [
            {"per_op_ms": 0.05, "payload_bytes": 1024, "single": False},
            {"per_op_ms": 0.06, "payload_bytes": 2048, "single": False}],
        "correctness": {"expected": 1, "observed": 1}}

    inherited_flat = ingest_a5_campaign_data(write("flat", 3, [flat]))
    assert [row["repeat_index"]
            for row in inherited_flat["conditions"]] == [3]
    inherited = ingest_a5_campaign_data(write("inherited", 3, [expanded]))
    assert [row["repeat_index"] for row in inherited["conditions"]] == [3, 3]
    indexed = ingest_a5_campaign_data(write("indexed", 0, [expanded]))
    assert [row["repeat_index"] for row in indexed["conditions"]] == [0, 1]


def test_pass4_finding8_mc2_chains_inherit_campaign_trial_identity():
    document = _pipeline_campaign(
        "chain-trial", [], run_id="campaign-run", repeat_index=2,
        chains=[_mc2_chain("a"),
                _mc2_chain("b", run_id="own", repeat_index=5),
                _mc2_chain("c", run_id="zero-run", repeat_index=0)])
    ingested = ingest_a5_campaign_data(document)

    by_condition = {row["condition"]: row
                    for row in ingested["mc2_chains"]}
    assert (by_condition["a"]["run_id"],
            by_condition["a"]["repeat_index"]) == ("campaign-run", 2)
    assert (by_condition["b"]["run_id"],
            by_condition["b"]["repeat_index"]) == ("own", 5)
    assert (by_condition["c"]["run_id"],
            by_condition["c"]["repeat_index"]) == ("zero-run", 0)

    tuple_chain = _pipeline_campaign(
        "tuple-trial", [], run_id="tuple-run", repeat_index=7,
        chains=[("int8", 2, 64, 512, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0)])
    row = ingest_a5_campaign_data(tuple_chain)["mc2_chains"][0]
    assert (row["run_id"], row["repeat_index"]) == ("tuple-run", 7)


def test_pass4_finding8_bf16_chains_inherit_campaign_trial_identity():
    document = _pipeline_campaign(
        "bf16-trial", [], run_id="bf16-run", repeat_index=4,
        ranks_requested=2, median_ms_by_case={
            "e64-g512-dispatch": {"dispatch": {"0": 0.5}},
            "e64-g512-combine": {"combine": {"0": 0.5}},
            "e64-g512-gmm": {"gmm_gate_up": {"0": 1.0},
                             "gmm_down": {"0": 1.0},
                             "swiglu": {"0": 0.5}},
            "e64-g512-pipeline": {"pipeline": {"0": 2.0}}})

    rows = ingest_a5_campaign_data(document)["mc2_chains"]

    assert len(rows) == 1
    assert (rows[0]["run_id"], rows[0]["repeat_index"]) == (
        "bf16-run", 4)


def test_pass5_finding1_p2p_pair_never_overrides_explicit_scope(embedded):
    _, bundle = embedded

    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.lookup_curve(
            "p2p_send_recv", 8, placement_scope="cross_node_urma",
            pair="0->1")
    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.lookup_curve(
            "p2p_send_recv", 32, placement_scope="intra_node_ub",
            pair="0->24")

    far = bundle.lookup_curve(
        "p2p_send_recv", 32, placement_scope="cross_node_urma",
        pair="0->24")
    assert far.identity["placement_scope"] == "cross_node_urma_far"
    assert far.identity["p2p_pair"] == "0->24"
    near = bundle.lookup_curve(
        "p2p_send_recv", 8, placement_scope="intra_node_ub", pair="0->1")
    assert near.identity["placement_scope"] == "intra_node_ub"
    assert near.identity["p2p_pair"] == "0->1"


def test_pass5_finding2_raw_points_never_pool_software_conditions():
    base = _raw_point(0.5)
    points = [
        {**base, "ms": 0.5, "stack": "rack06", "cann": "9.1.0",
         "protocol": {"transport": "ub", "algorithm": "ring"}},
        {**base, "ms": 0.7, "stack": "rack08", "cann": "9.1.0",
         "protocol": {"transport": "ub", "algorithm": "ring"}},
        {**base, "ms": 0.9, "stack": "rack06", "cann": "9.9",
         "protocol": {"transport": "ub", "algorithm": "ring"}},
        {**base, "ms": 1.1, "stack": "rack06", "cann": "9.1.0",
         "protocol": {"transport": "ub", "algorithm": "tree"}},
        {**base, "ms": 1.3, "stack": "rack06", "cann": "9.1.0",
         "protocol": {"transport": "urma", "algorithm": "ring"}},
    ]

    ingested = ingest_a5_campaign_data(_campaign(
        label="raw-software", conditions=[], raw_points=points))

    rows = {(row["stack"], row["cann"], row["algorithm"],
             row["protocol"]["transport"]): row["per_op_ms"]
            for row in ingested["conditions"]}
    assert rows == {
        ("rack06", "9.1.0", "ring", "ub"): 0.5,
        ("rack08", "9.1.0", "ring", "ub"): 0.7,
        ("rack06", "9.9", "ring", "ub"): 0.9,
        ("rack06", "9.1.0", "tree", "ub"): 1.1,
        ("rack06", "9.1.0", "ring", "urma"): 1.3,
    }


@pytest.mark.parametrize("field, primary, secondary", [
    ("stack", "rack06", "rack08"),
    ("cann", "9.1.0", "9.9"),
    ("algorithm", "ring", "tree"),
    ("protocol", {"kind": "ub"}, {"kind": "urma"}),
])
def test_pass5_finding3_binding_and_recipe_enforce_operator_software(
        field, primary, secondary):
    conditions = []
    for value, intercept in ((primary, 0.05), (secondary, 0.09)):
        for payload in (1024, 1048576, 4194304):
            conditions.append({
                "collective": "all_gather", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "dtype": "bf16",
                "world_size": 8, "placement_scope": "intra_node_ub",
                "run_id": f"{value}-{payload}", "split": "train",
                field: value})
    bundle = _fit_campaign(_campaign(
        label=f"bind-{field}", stack="rack06", conditions=conditions,
        world_size=8, placement="intra-node-ub"))
    expected = bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub",
        **{field: secondary})
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    operation = replace(
        operation, metadata=operation.metadata + ((field, secondary),))

    curve = bundle.bind_service_curve(parse_case(raw), operation)
    assert curve.provenance["bound_a5_curve"] == expected.model_id
    if field in {"algorithm", "protocol"}:
        assert curve.identity["transport"][field] == secondary
    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8).build(operation, None)
    assert plan.metadata["model_id"] == expected.model_id


def test_pass5_finding3_explicit_selector_override_reaches_identity():
    conditions = []
    for algorithm, intercept in (("ring", 0.05), ("tree", 0.09)):
        for payload in (1024, 1048576, 4194304):
            conditions.append({
                "collective": "all_gather", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "dtype": "bf16",
                "world_size": 8, "placement_scope": "intra_node_ub",
                "algorithm": algorithm, "run_id": f"{algorithm}-{payload}",
                "split": "train"})
    bundle = _fit_campaign(_campaign(
        label="override", stack="rack06", conditions=conditions,
        world_size=8, placement="intra-node-ub"))
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    operation = replace(
        operation, metadata=operation.metadata + (("algorithm", "ring"),))
    expected = bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub",
        algorithm="tree")

    curve = bundle.bind_service_curve(
        parse_case(raw), operation, algorithm="tree")

    assert curve.provenance["bound_a5_curve"] == expected.model_id
    assert curve.identity["transport"]["algorithm"] == "tree"


def test_pass5_finding4_rank_rows_keep_document_expansion_mode():
    rank = {
        "role": "rank", "status": "completed", "label": "rank-observed",
        "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
        "observed_expansion_mode": "aicpu",
        "requested_expansion_mode": "aiv",
        "profiler_evidence": ["hcom_allReduce_AicpuKernel"],
        "cases": [{
            "case_id": "probe", "collective": "all_reduce",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1}}]}

    row = ingest_module._rank_rows(rank, "memory")[0]

    assert (row["observed_expansion_mode"],
            row["requested_expansion_mode"], row["regime"],
            row["expansion_unconfirmed"]) == (
                "aicpu", "aiv", "aicpu", False)
    assert row["profiler_evidence"] == ["hcom_allReduce_AicpuKernel"]


@pytest.mark.parametrize("observed_key", [
    "observed_expansion_mode", "observed_execution"])
def test_pass5_finding4_parent_defaults_merge_into_rank_documents(
        tmp_path, observed_key):
    parent = {
        "role": "parent", "label": "trials", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub", observed_key: "aicpu",
        "requested_expansion_mode": "aiv",
        "profiler_evidence": ["hcom_allReduce_AicpuKernel"],
        "stack": "rack06", "cann": "9.1.0", "algorithm": "ring",
        "protocol": {"kind": "ub"}}
    rank = {
        "role": "rank", "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub",
        "cases": [{
            "case_id": "probe", "collective": "all_reduce",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1}}]}
    root = tmp_path / "rank-parent"
    root.mkdir()
    (root / "trials.parent.json").write_text(
        json.dumps(parent), encoding="utf-8")
    (root / "trials.rank0.json").write_text(
        json.dumps(rank), encoding="utf-8")

    rows = ingest_a5_campaign_data(root)["conditions"]

    assert len(rows) == 1
    row = rows[0]
    assert (row["observed_expansion_mode"],
            row["requested_expansion_mode"], row["regime"],
            row["expansion_unconfirmed"]) == (
                "aicpu", "aiv", "aicpu", False)
    assert row["profiler_evidence"] == ["hcom_allReduce_AicpuKernel"]
    assert (row["stack"], row["cann"], row["algorithm"],
            row["protocol"]) == (
                "rack06", "9.1.0", "ring", {"kind": "ub"})


def test_pass5_finding5_confirmed_curve_precedes_unspecified_fallback():
    def conditions(regime, intercept):
        rows = []
        for payload in (1024, 1048576, 4194304):
            row = {
                "collective": "all_reduce", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "dtype": "bf16",
                "world_size": 2, "placement_scope": "intra_node_ub",
                "run_id": f"{regime}-{payload}", "split": "train"}
            if regime is not None:
                row["observed_expansion_mode"] = regime
            rows.append(row)
        return rows
    bundle = _fit_campaign(_campaign(
        label="fallback-priority",
        conditions=conditions("aicpu", 0.05) + conditions(None, 0.09)))
    confirmed = bundle.curves[
        ("all_reduce", 2, "intra_node_ub", "bf16", "aicpu")]
    fallback = bundle.curves[
        ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")]

    selected = bundle.lookup_curve(
        "all_reduce", 2, expansion_mode="aicpu",
        allow_unspecified_fallback=True, boundary="device_kernel")
    assert selected.model_id == confirmed.model_id
    assert selected.model_id != fallback.model_id
    assert selected.identity["expansion_regime"] == "aicpu"

    unavailable = bundle.lookup_curve(
        "all_reduce", 2, expansion_mode="aiv",
        allow_unspecified_fallback=True, boundary="device_kernel")
    assert unavailable.model_id == fallback.model_id
    assert unavailable.identity["expansion_regime"] == "unspecified"


def test_pass6_finding1_stack_and_cann_persist_in_execution_identity(embedded):
    _, bundle = embedded
    raw = _a5_case()
    op = _a5_comm_op(raw)
    curve = bundle.bind_service_curve(raw, op, stack="rack08", cann="9.2.0")
    assert curve.identity["transport"]["stack"] == "rack08"
    assert curve.identity["transport"]["cann"] == "9.2.0"

    other_stack = replace(op, metadata=op.metadata + (("stack", "rack06"), ("cann", "9.1.0")))
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(raw, (other_stack,),
                       recipes=RecipeBook({op.recipe: ResourceService(curve)}))


def test_pass6_finding2_parent_aggregation_preserves_software_selectors(tmp_path):
    parent = {
        "role": "parent", "label": "soft-parent", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub",
        "rank_reports": [
            {"world_size": 2, "cases": [
                {"case_id": "ring", "collective": "all_reduce", "payload_bytes": 1024,
                 "algorithm": "ring", "per_op_ms_summary": {"median": 0.05},
                 "correctness": {"expected": 1, "observed": 1}},
                {"case_id": "tree", "collective": "all_reduce", "payload_bytes": 1024,
                 "algorithm": "tree", "per_op_ms_summary": {"median": 0.09},
                 "correctness": {"expected": 1, "observed": 1}},
            ]}
            for _ in range(2)
        ],
    }
    path = tmp_path / "soft.parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    rows = ingest_a5_campaign_data(path)["conditions"]
    assert {(row["algorithm"], row["per_op_ms"]) for row in rows} == {
        ("ring", 0.05), ("tree", 0.09)}


def test_pass6_finding3_rank_observed_execution_overrides_parent_mode(tmp_path):
    parent = {
        "role": "parent", "label": "override-parent", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub", "observed_expansion_mode": "aicpu",
        "rank_reports": [{
            "world_size": 2, "observed_execution": "aiv",
            "cases": [{"case_id": "c1", "collective": "all_reduce",
                       "payload_bytes": 1024,
                       "per_op_ms_summary": {"median": 0.05},
                       "correctness": {"expected": 1, "observed": 1}}],
        }],
    }
    path = tmp_path / "override.parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    row = ingest_a5_campaign_data(path)["conditions"][0]
    assert row["observed_expansion_mode"] == "aiv"
    assert row["regime"] == "aiv"


def test_pass6_finding4_and_5_regime_and_recipe_reflect_effective_overrides(embedded):
    _, bundle = embedded
    raw = _a5_case()
    op = replace(_a5_comm_op(raw),
                 metadata=_a5_comm_op(raw).metadata + (("world_size", 8), ("stack", "rack06")))
    rule = bundle.to_service_regime(raw, op, collective="all_reduce",
                                    world_size=2, stack="rack08")
    assert rule["match"]["transport"]["world_size"] == 2
    assert rule["match"]["transport"]["stack"] == "rack08"
    assert rule["match"]["metadata.world_size"] == 2
    assert rule["match"]["metadata.stack"] == "rack08"

    recipe = bundle.build_a5_comm_recipe("all_gather", world_size=8, stack="rack08")
    prepared, plan = recipe.prepare(op, None)
    assert plan.metadata["stack"] == "rack08"
    assert dict(prepared.metadata)["program_execution"]["stack"] == "rack08"


@pytest.mark.parametrize("family", ["all_reduce", "all_gather", "all_to_all", "all_to_allv"])
def test_pass7_finding1_underscore_collective_families_include_transport_identity(embedded, family):
    _, bundle = embedded
    raw = _a5_case()
    op = _measurement_op(family=family, metadata=(), hbm_bytes=0)
    curve = bundle.bind_service_curve(raw, op)
    assert "transport" in curve.identity

    changed = _a5_case()
    changed["runtime"]["tp"] = 4
    changed["runtime"]["ep"] = 4
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(changed, (op,),
                       recipes=RecipeBook({op.recipe: ResourceService(curve)}))


def test_pass7_finding2_and_3_child_software_aliases_and_sample_overrides_win(tmp_path):
    parent = {
        "role": "parent", "label": "alias-parent", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub", "cann": "9.1.0", "algorithm": "ring",
        "observed_expansion_mode": "aicpu",
        "rank_reports": [
            {"world_size": 2, "machine": {"cann": "9.0.0"},
             "protocol": {"algorithm": "tree"},
             "cases": [{"case_id": "c1", "collective": "all_reduce",
                        "payload_bytes": 1024,
                        "samples": [{"per_op_ms": 0.09, "observed_execution": "aiv"}],
                        "correctness": {"expected": 1, "observed": 1}}]},
        ],
    }
    path = tmp_path / "alias.parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    row = ingest_a5_campaign_data(path)["conditions"][0]
    assert row["cann"] == "9.0.0"
    assert row["algorithm"] == "tree"
    assert row["regime"] == "aiv"


def test_pass7_finding4_stream_preserves_scalar_ranks(tmp_path):
    from mindie_sim.observations.a5_comm_binding import _stream_json
    doc = _campaign(label="stream-ranks", ranks=8, world_size=None, conditions=[
        {"collective": "all_reduce", "ranks": 8, "payload_bytes": 1024, "median_ms": 0.05},
    ])
    doc.pop("world_size", None)
    path = tmp_path / "ranks.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    streamed = _stream_json(path)
    assert streamed["ranks"] == 8
    assert streamed["conditions"][0]["ranks"] == 8
    ingested = ingest_a5_campaign_data(streamed)
    assert ingested["conditions"][0]["world_size"] == 8


def test_pass8_finding1_parent_split_propagates_to_rank_reports(tmp_path):
    parent = {
        "role": "parent", "label": "heldout-parent", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub", "split": "heldout",
        "rank_reports": [{
            "world_size": 2,
            "cases": [
                {"case_id": "c1", "collective": "all_reduce",
                 "payload_bytes": 1024, "per_op_ms_summary": {"median": 0.05},
                 "correctness": {"expected": 1, "observed": 1}},
                {"case_id": "c2", "collective": "all_reduce",
                 "payload_bytes": 4096, "per_op_ms_summary": {"median": 0.06},
                 "correctness": {"expected": 1, "observed": 1}},
            ],
        }],
    }
    path = tmp_path / "heldout.parent.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    ingested = ingest_a5_campaign_data(path)
    assert {row["split"] for row in ingested["conditions"]} == {"heldout"}
    bundle = fit_a5_service_curves(ingested)
    assert bundle.curves == {}
    assert len(bundle.unfitted) == 1


def test_pass8_finding2_distinct_collective_families_do_not_share_identity(embedded):
    _, bundle = embedded
    raw = _a5_case()
    ag = _measurement_op(family="all_gather", metadata=(), hbm_bytes=0)
    ar = replace(ag, family="all_reduce")
    ag_curve = bundle.bind_service_curve(raw, ag)
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(raw, (ar,),
                       recipes=RecipeBook({ar.recipe: ResourceService(ag_curve)}))


def test_pass8_finding3_and_4_stream_preserves_mixed_samples_and_sample_payloads(tmp_path):
    from mindie_sim.observations.a5_comm_binding import _stream_json
    doc = _campaign(label="mixed-samples", conditions=[{
        "collective": "all_reduce", "payload_bytes": 1024,
        "samples": [
            {"per_op_ms": 0.05},
            {"per_op_ms": 0.08, "payload_bytes": 2048},
            {"per_op_ms": 0.09, "observed_execution": "aiv"},
        ],
    }])
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    ordinary = ingest_a5_campaign_data(path)
    streamed = ingest_a5_campaign_data(_stream_json(path))
    expected = [(1024, 0, "unspecified"), (2048, 1, "unspecified"), (1024, 2, "aiv")]
    assert [(r["payload_bytes"], r["repeat_index"], r["regime"])
            for r in ordinary["conditions"]] == expected
    assert [(r["payload_bytes"], r["repeat_index"], r["regime"])
            for r in streamed["conditions"]] == expected


def test_pass8_finding5_nested_row_hardware_rejects_non_a5():
    bad = _campaign(conditions=[{
        "collective": "all_reduce", "payload_bytes": 1024, "per_op_ms": 0.05,
        "machine": {"hardware_key": "Ascend910_93", "arch_family": "A3"},
    }])
    with pytest.raises(ModelingError, match="a5_calibration_rejects_non_a5_hardware"):
        ingest_a5_campaign_data(bad)


def test_pass9_findings_boundary_single_work_floor_resource_and_stream(tmp_path, embedded):
    from mindie_sim.observations.a5_comm_binding import _stream_json
    _, bundle = embedded
    raw = _a5_case()
    op = _measurement_op(family="all_reduce", link_bytes=1048576, vector_ops=0, hbm_bytes=0)
    with pytest.raises(ModelingError, match="service_curve_requires_device_or_resource_phase_boundary"):
        ingest_a5_campaign_data(_campaign(boundary="call"))
    phase_bundle = _fit_campaign(_campaign(world_size=8, boundary="resource_phase", conditions=[
        {"collective": "all_gather", "payload_bytes": 1024, "median_ms": 0.05},
        {"collective": "all_gather", "payload_bytes": 1048576, "median_ms": 0.06}]))
    assert all(c.boundary == "resource_phase" for c in phase_bundle.curves.values())
    with pytest.raises(ModelingError):
        phase_bundle.build_a5_comm_recipe("all_gather", world_size=8).build(op, None)
    with pytest.raises(ModelingError, match="service_regime_requires_device_kernel_boundary"):
        phase_bundle.to_service_regime(raw, op, boundary="resource_phase")
    single_ing = ingest_a5_campaign_data(_campaign(single=True))
    assert len(single_ing["conditions"]) == 0 and len(single_ing["single_op_anchors"]) == 2
    vecs = {bundle.build_a5_comm_recipe("all_reduce", world_size=2, expansion_mode="aiv",
            allow_unspecified_fallback=True, active_aiv_blocks=b).build(op, None).work.apply(op).vector_ops
            for b in (1, 16, 64)}
    assert vecs == {524288.0}
    fast = _fit_campaign(_campaign(conditions=[
        {"collective": "all_reduce", "payload_bytes": p, "median_ms": m, "observed_expansion_mode": "aicpu"}
        for p, m in ((1024, 0.001), (4096, 0.002))]))
    assert fast.build_a5_comm_recipe("all_reduce", world_size=2, expansion_mode="aicpu").build(
        _measurement_op(family="all_reduce", link_bytes=1024, hbm_bytes=0), None).minimum_ms == pytest.approx(0.001)
    with pytest.raises(ModelingError, match="a5_comm_curve_requires_byte_resource"):
        bundle.bind_service_curve(raw, op, collective="all_reduce", world_size=2, resource="vector")
    with pytest.raises(ModelingError, match="a5_expansion_mode_unknown"):
        bundle.lookup_curve("all_reduce", 2, expansion_mode="not_a_mode")
    parent = {"role": "parent", **_campaign(), "rank_reports": [{"world_size": 2, "cases": [
        {"case_id": "c", "collective": "all_reduce", "payload_bytes": 1024, "per_op_ms_summary": {"median": 0.05}}]}]}
    parent.pop("conditions")
    assert len(ingest_a5_campaign_data(parent)["conditions"]) == 1
    pure = tmp_path / "pure.json"
    pure.write_text(json.dumps(_campaign(conditions=[{
        "collective": "all_reduce", "payload_bytes": 1024,
        "samples": [{"per_op_ms": 0.05}, {"per_op_ms": 0.09}]}])), encoding="utf-8")
    assert ingest_a5_campaign_data(_stream_json(pure))["conditions"][0]["per_op_ms"] == pytest.approx(0.07)


def test_pass10_findings_buckets_arrays_weighted_median_phase_and_contention(tmp_path, embedded):
    from mindie_sim.observations.a5_comm_binding import _stream_json
    _, bundle = embedded
    samples = [{"payload_bytes": 1024 * (i + 1), "per_op_ms": float(i + 1),
                "run_id": f"r-{i}", "split": "train"} for i in range(64)]
    samples.append({"payload_bytes": 1024, "per_op_ms": 99.0, "run_id": "h", "split": "heldout"})
    p = tmp_path / "many.json"
    p.write_text(json.dumps(_campaign(conditions=[{"collective": "all_reduce", "samples": samples}])), encoding="utf-8")
    fit = _fit_campaign(_stream_json(p))
    assert dict(fit.curves[("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")].knots)[1024] == pytest.approx(1.0)
    assert len(fit.holdout) == 1 and fit.holdout[0]["observed_ms"] == pytest.approx(99.0)
    for seq in ([1.0] * 128 + [100.0] * 64, [100.0] * 64 + [1.0] * 128):
        med_path = tmp_path / "med.json"
        med_path.write_text(json.dumps(_campaign(conditions=[{
            "collective": "all_reduce", "payload_bytes": 1024,
            "samples": [{"per_op_ms": v} for v in seq]}])), encoding="utf-8")
        assert ingest_a5_campaign_data(_stream_json(med_path))["conditions"][0]["per_op_ms"] == pytest.approx(1.0)
    conds = [{"collective": "all_reduce", "payload_bytes": 1024 + i, "per_op_ms": 0.05} for i in range(1025)]
    conds[-1]["status"] = "failed"
    bad_path = tmp_path / "bad1025.json"
    bad_path.write_text(json.dumps(_campaign(conditions=conds)), encoding="utf-8")
    with pytest.raises(ValueError, match="not_completed"):
        ingest_a5_campaign_data(_stream_json(bad_path))
    op = _measurement_op()
    plan = bundle.build_a5_comm_recipe("all_gather", world_size=8).build(op, None)
    with pytest.raises(ModelingError, match="service_curve_boundary_mismatch"):
        plan.lower(_comm_context(op, service_boundary="resource_phase"))
    with pytest.raises(ModelingError, match="a5_expansion_mode_unknown"):
        compute_a5_contention_slowdown("not_a_mode", concurrent_vector_util=1.0)


def test_pass11_and_pass12_findings_stream_dtype_dedupe_split_raw_and_null_ms():
    def stream(doc):
        return _read_stream_value(_iter_json_events(io.BytesIO(json.dumps(doc).encode())))

    dt_samples = [{"payload_bytes": p, "dtype": d, "per_op_ms": t * (p / 1024), "run_id": "r", "repeat_index": i}
                  for d, t in (("bf16", 0.1), ("int8", 0.2)) for p in (1024, 4096) for i in range(70)]
    dt_bundle = _fit_campaign(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": dt_samples}])))
    assert dict(dt_bundle.curves[("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")].knots) == {1024: 0.1, 4096: 0.4}
    assert dict(dt_bundle.curves[("all_reduce", 2, "intra_node_ub", "int8", "unspecified")].knots) == {1024: 0.2, 4096: 0.8}
    dup = ([{"payload_bytes": 1024, "median_ms": 1.0, "run_id": "r", "repeat_index": 0}] * 200
           + [{"payload_bytes": 1024, "median_ms": 3.0, "run_id": "r", "repeat_index": i} for i in range(1, 100)])
    assert ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": dup}])))["conditions"][0]["per_op_ms"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": dup + [{"payload_bytes": 1024, "median_ms": 9.0, "run_id": "r", "repeat_index": 0}]}])))
    split_conf = ([{"payload_bytes": 1024, "run_id": "r", "repeat_index": i, "per_op_ms": 1.0, "split": "train", "status": "completed"} for i in range(260)]
                  + [{"payload_bytes": 1024, "run_id": "r", "repeat_index": 259, "per_op_ms": 99.0, "split": "heldout"}])
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": split_conf}])))
    raw_null = [{"payload_bytes": 1024, "median_ms": None, "raw_iterations_us": [100, 200], "run_id": "r", "repeat_index": i} for i in range(260)]
    assert ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": raw_null}])))["conditions"][0]["per_op_ms"] == pytest.approx(0.15)
    null_dur = [{"payload_bytes": 1024, "median_ms": None, "duration_ms": 1.25, "run_id": "r", "repeat_index": i} for i in range(260)]
    assert ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": null_dur}])))["conditions"][0]["per_op_ms"] == pytest.approx(1.25)
    held_doc = {**_campaign(conditions=[{"collective": "all_reduce", "samples": (
        [{"payload_bytes": 1024, "per_op_ms": 1.0, "run_id": "r", "repeat_index": i} for i in range(260)]
        + [{"payload_bytes": 1024, "per_op_ms": 9.0, "run_id": "r", "repeat_index": 260, "split": "train"},
           {"payload_bytes": 4096, "per_op_ms": 18.0, "run_id": "r", "repeat_index": 0, "split": "train"}])}]), "split": "heldout"}
    assert dict(_fit_campaign(stream(held_doc)).curves[("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")].knots)[1024] == pytest.approx(9.0)
    req_samples = ([{"size_bytes": 1024, "per_op_ms": 1.0, "run_id": "r", "repeat_index": i, "requested_expansion_mode": "aiv", "observed_execution": "aicpu"} for i in range(260)]
                   + [{"size_bytes": 1024, "per_op_ms": 99.0, "run_id": "r", "repeat_index": 260, "requested_expansion_mode": "aicpu", "observed_execution": "aicpu"}])
    req_ing = ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "payload_bytes": 2048, "observed_expansion_mode": "aiv", "samples": req_samples}])))
    assert len(req_ing["conditions"]) == 1 and req_ing["conditions"][0]["payload_bytes"] == 1024 and req_ing["conditions"][0]["regime"] == "aicpu" and req_ing["conditions"][0]["per_op_ms"] == pytest.approx(1.0)
    anon = _campaign(conditions=[{"collective": "all_reduce", "payload_bytes": 1024, "samples": [{"per_op_ms": 1.0 if i % 2 == 0 else 2.0} for i in range(5000)]}])
    assert ingest_a5_campaign_data(stream(anon))["conditions"][0]["per_op_ms"] == pytest.approx(1.5)
