"""Communication observation ingestion, calibration, binding, and local services.

[Role] Verify communication campaign normalization, bounded streaming, curve
  fitting, holdout isolation, engine binding, normalized stages, and native
  local-service applicability/lineage boundaries.
[Invariants] Unconfirmed requests never promote to confirmed regimes; confirmed
  regimes never alias; incompatible hardware and software stacks never pool;
  launch anchors never train; physical work is conserved; normalized stages
  charge startup once and never compress joint or non-physical phases to link.
[Timeouts & Locks] Pure tier-0 in-memory and ``tmp_path`` tests; no child
  processes, device context, external capture mounts, or locks.
[Side-Effect Policy] Writes only inside ``tmp_path``; never mutates repo files
  or environment variables.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

from mindie_sim import compose_case, graph_costs, predict, target_graphs
from mindie_sim.observations import calibration as obs_cal, communication as comm
from mindie_sim.observations.communication import (
    binding as comm_bind, binding as comm_rec, ingest as comm_ing, ingest as comm_str)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA, RESOURCE_SERVICE_SCHEMA, normalize_resource_service)
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.service_regime_domains.services import ResourceService, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0

ROOT, HOLDOUT, MIB = Path(__file__).resolve().parents[2], (65536, 16 * 1024 * 1024), 1024 * 1024
COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "all_to_all", "all_to_allv")
WORLD_SIZES, TARGET_MODEL, OTHER_HW = (2, 4, 8, 16, 32), "qwen3.5-35b-a3b", "Ascend910B3"
CHAIN_STAGES = ("quant", "all_to_allv_dispatch", "grouped_matmul", "all_to_allv_combine")


def _raises(exc, match, fn):
    with pytest.raises(exc, match=match):
        fn()


@pytest.fixture(scope="module")
def embedded():
    ing = comm.ingest_a5_campaign_data()
    return ing, comm.fit_a5_service_curves(ing)


@pytest.fixture(scope="module")
def native_wave(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-wave")
    (root / "manifest").mkdir()
    (root / "raw" / "parsed").mkdir(parents=True)
    m_bytes = b'{"wave":"synthetic-g2"}'
    (root / "manifest" / "native-manifest.json").write_bytes(m_bytes)
    m_sha, img = hashlib.sha256(m_bytes).hexdigest(), "sha256:" + "a" * 64
    (root / "raw" / "native-versions.json").write_text(json.dumps({
        "python_packages": {"torch": "2.10.0+cpu", "torch_npu": "2.10.0.post7", "cann": "9.2.0"},
        "cann_version_info": "9.2.0", "topology_interpretation": "intrahost UB mesh",
        "hccl_algorithm": "unknown", "hccl_protocol": None}), encoding="utf-8")
    (root / "raw" / "environment.txt").write_text(f"Linux rack08os2 5.10.0\nimage_id={img}\n", encoding="utf-8")
    (root / "raw" / "topology.txt").write_text("intrahost ub_mesh\n", encoding="utf-8")
    conds = []
    for coll, base_us in (("all_gather", 40.0), ("reduce_scatter", 39.0)):
        for pay, sp, d in ((32768, "train", 0.5), (MIB, "train", 2.0), (10 * MIB, "train", 19.0),
                           (786432, "heldout", -15.0), (33558528, "heldout", 21.0)):
            c_us, u_us = base_us + d, base_us + d + 25.0
            per_rank = {str(r): {"canary_before_ok": True, "canary_after_ok": True, "mapping_ok": True,
                "actual_bytes_match": True, "actual_input_bytes": pay, "calls": [
                    {"rep_index": i, "marker": f"m-{r}-{i}", "status": "ok",
                     "ai_cpu_rows": [{"kind": "ai_cpu", "name": "HcclLaunchAicpuKernel"}],
                     "ai_cpu_union_us": c_us + 0.1 * i, "communication_union_us": u_us + 0.1 * i,
                     "overlap_us": c_us + 0.1 * i, "combined_span_union_us": u_us + 0.1 * i}
                    for i in range(3)]} for r in (0, 1)}
            conds.append({"native_id": f"native.{coll}.g2.{pay}.{sp}", "collective": coll, "group_size": 2,
                "dtype": "bf16", "status": "ok", "input_bytes": pay, "output_bytes": pay * 2,
                "tokens": 64, "hidden_size": 2048, "split": sp, "profile_rank_count": 2, "per_rank": per_rank})
    path = root / "raw" / "parsed" / "native-summary.json"
    path.write_text(json.dumps({"manifest_sha256": m_sha, "scope": "g2", "conditions": conds}), encoding="utf-8")
    return path, img, m_sha


def _row(collective="all_reduce", payload=1024, ms=0.05, **kw):
    return {"collective": collective, "payload_bytes": payload, "median_ms": ms, **kw}

def _campaign(label="probe", conditions=None, **kw):
    return {"label": label, "status": "completed", "canary_ok": True, "hardware_key": comm.A5_HARDWARE_KEY,
            "arch_family": comm.A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
            "conditions": [_row(), _row(payload=MIB, ms=0.06)] if conditions is None else conditions, **kw}

def _fit(doc):
    return comm.fit_a5_service_curves(comm.ingest_a5_campaign_data(doc))

def _mode_rows(mode, *, evidence, intercept, slope):
    return [_row(payload=p, ms=intercept + p * slope, requested_expansion_mode=mode,
                 observed_expansion_mode="not_observed", profiler_evidence=[evidence] if evidence else None,
                 single=False, dtype="bf16", world_size=2, placement="intra-node-ub", run_id=f"{mode}-{p}")
            for p in (1024, MIB, 4 * MIB)]

def _chain(condition, dtype="int8", **kw):
    return {"condition": condition, "status": "ok", "dtype": dtype, "ranks": 2, "experts": 64, "global_tokens": 512,
            "chain_ms": 4.0, "paired_overlap_ms": 1.0,
            "legs_ms": {"dispatch": 1.0, "gate_up": 1.0, "swiglu_quant": 1.0,
                        "down": 1.0, "combine": 1.0}, **kw}

def _comm_case(*, hardware=comm.A5_SKU, tp=8, dp=1, ep=8):
    raw = compose_case(TARGET_MODEL, hardware=hardware, tp=tp, dp=dp, ep=ep,
                       layers=1, prompt_tokens=16, output_tokens=1, concurrency=1)
    raw["runtime"]["graph_mode"] = "eager"
    return raw

def _comm_op(raw):
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=64,
                        sequences=1, apply_body_defaults=True)[0]
    return replace(next(op for op in ops if op.op_id == "s0.l0.attention.sp_gather"), deps=())

def _measure_op(**kw):
    vals = dict(op_id="comm", family="allgather", phase="prefill", stage=0, layer=0, deps=(), requests=(1,),
                semantic_shape=(1,), local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
                link_bytes=401408, hbm_bytes=4096, dtype="bf16", recipe="comm")
    return OpSpec(**{**vals, **kw})

def _comm_ctx(op, **kw):
    res = {"hbm": "memory@r0:s0", "vector": "vector@r0:s0", "mte2": "mte2@r0:s0", "mte3": "mte3@r0:s0"}
    return RecipeContext(**{**dict(op=op, work={"link:port0": float(op.link_bytes)}, resource_names=res,
        rates={v: 1e6 for v in res.values()}, core_count=64, resource_units={k: 64 for k in res},
        service_identity={"hardware": {"key": comm.A5_HARDWARE_KEY, "architecture": comm.A5_ARCH_FAMILY}}), **kw})

def _child_desc():
    return (
        {"kind": "native_target_call", "phase": "aicpu_child", "resource": "aicpu_child",
         "work_coordinate": "per_rank_input_bytes", "duration_unit": "ms", "source_kind": "native_ai_cpu_child",
         "service_semantics": "aicpu_child_collective_phase", "complete_collective_call": False,
         "complete_directed_call": False, "link_transit_identified": False, "hop_scaling": "not_derived",
         "algorithm_identified": False, "direction_observed": False, "explicit": True},
        {"kind": "native_single_target_call", "phase": "aicpu_child", "warmup_calls": 3, "admitted_repeats": 5,
         "profiled_ranks": [0, 1], "clock_domain": "device", "concurrent_collectives": False})


def test_expansion_regime_resolution_and_conflict_isolation():
    for mode in ("aicpu", "aiv"):
        res = comm.resolve_expansion_regime(mode, "not_observed")
        assert (res["regime"], res["expansion_unconfirmed"], res["evidence_conflict"],
                res["unconfirmed_tag"]) == ("unspecified", True, False, f"requested_{mode}_unconfirmed")
    for req, obs, ev, exp in (("unspecified", "aicpu", None, "aicpu"), ("unspecified", "aiv", None, "aiv"),
                              ("aicpu", "not_observed", "hcom_allReduce_AicpuKernel", "aicpu"),
                              ("aiv", "not_observed", "AI_VECTOR_CORE", "aiv"),
                              ("unspecified", "not_observed", {"k": "hcom_allReduce_AicpuKernel"}, "aicpu")):
        res = comm.resolve_expansion_regime(req, obs, profiler_evidence=ev)
        assert (res["regime"], res["expansion_unconfirmed"], res["unconfirmed_tag"]) == (exp, False, None)
    long_ev = ["hcom_allReduce_AicpuKernel"] + ["noop"] * 120 + ["AI_VECTOR_CORE"]
    for req in ("aicpu", "aiv", "unspecified"):
        res = comm.resolve_expansion_regime(req, "not_observed", profiler_evidence=long_ev)
        assert (res["regime"], res["evidence_conflict"], res["confirmed_by"],
                res["profiler_modes"]) == ("unspecified", True, None, ["aicpu", "aiv"])


def test_campaign_ingest_validation_and_anchor_separation():
    for patch, reason in (({"status": "failed"}, "a5_campaign_not_completed"),
                          ({"state": "aborted"}, "a5_campaign_not_completed"),
                          ({"canary_ok": False}, "a5_campaign_canary_failed"),
                          ({"canary": "mismatch"}, "a5_campaign_canary_failed"),
                          ({"correctness": {"expected": 36, "observed": 35}}, "a5_campaign_canary_failed")):
        doc = _campaign()
        doc["conditions"][0].update(patch)
        for bad in (_campaign(**patch), doc):
            _raises(ValueError, reason, lambda d=bad: comm.ingest_a5_campaign_data(d))
    for bad, reason in (
        (_campaign(conditions=[_row(samples=[{"per_op_ms": 0.05, "canary_ok": False}])]), "a5_campaign_canary_failed"),
        (_campaign(conditions=[], raw_points=[{"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.5, "status": "failed"}]), "a5_campaign_not_completed"),
        (_campaign(conditions=[], chains=[_chain("bad", canary_ok=False)]), "a5_campaign_canary_failed"),
    ):
        _raises(ValueError, reason, lambda d=bad: comm.ingest_a5_campaign_data(d))
    anc = comm.ingest_a5_campaign_data(_campaign(conditions=[
        _row(), _row(payload=MIB, ms=0.06), _row(ms=0.90, single=True), _row(payload=2 * MIB, ms=0.95, single=True)]))
    assert (len(anc["conditions"]), len(anc["single_op_anchors"])) == (2, 2)
    assert dict(comm.fit_a5_service_curves(anc).curves[("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")].knots) == {
        1024: pytest.approx(0.05), MIB: pytest.approx(0.06)}
    assert len(comm.ingest_a5_campaign_data(_campaign(single=True))["conditions"]) == 0
    sp = comm.ingest_a5_campaign_data(_campaign(conditions=[_row(samples=[
        {"per_op_ms": 0.05, "single": False}, {"per_op_ms": 9.0, "single": True}])]))
    assert (len(sp["conditions"]), len(sp["single_op_anchors"])) == (1, 1)


def test_parent_rank_merge_preserves_trials_and_overrides(tmp_path):
    parent = {"role": "parent", "label": "pair-campaign", "status": "completed", "canary_ok": True,
              "hardware_key": comm.A5_HARDWARE_KEY, "arch_family": comm.A5_ARCH_FAMILY, "world_size": 2,
              "dtype": "bf16", "placement": "intra-node-ub", "cann": "9.1.0", "algorithm": "ring",
              "observed_expansion_mode": "aicpu", "repeat_index": 3}
    (tmp_path / "pair.parent.json").write_text(json.dumps(parent), encoding="utf-8")
    for rank, (a_ms, b_ms) in ((0, (0.05, 0.10)), (1, (0.07, 0.12))):
        (tmp_path / f"pair.rank{rank}.json").write_text(json.dumps({
            "role": "rank", "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub", "cases": [
                {"case_id": "t1", "collective": "all_reduce", "payload_bytes": 1024, "run_id": "trial-a", "per_op_ms_summary": {"median": a_ms}},
                {"case_id": "t2", "collective": "all_reduce", "payload_bytes": MIB, "run_id": "trial-b", "algorithm": "tree",
                 "observed_execution": "aiv", "per_op_ms_summary": {"median": b_ms}},
                {"case_id": "l", "collective": "all_reduce", "payload_bytes": 1024, "single": True, "per_op_ms_summary": {"median": 10.0}}]}), encoding="utf-8")
    ing = comm.ingest_a5_campaign_data(tmp_path)
    by_run = {r["run_id"]: r for r in ing["conditions"]}
    assert ing["campaign_labels"] == ["pair-campaign"] and len(ing["single_op_anchors"]) == 1
    assert by_run["trial-a"]["per_op_ms"] == pytest.approx(0.06) and by_run["trial-a"]["repeat_index"] == 3 and by_run["trial-a"]["regime"] == "aicpu"
    assert by_run["trial-b"]["per_op_ms"] == pytest.approx(0.11) and by_run["trial-b"]["algorithm"] == "tree" and by_run["trial-b"]["regime"] == "aiv"
    held = _fit({**parent, "split": "heldout", "rank_reports": [{"world_size": 2, "cases": [
        {"case_id": f"c{p}", "collective": "all_reduce", "payload_bytes": p, "per_op_ms_summary": {"median": 0.05}} for p in (1024, 4096)]}]})
    assert held.curves == {} and len(held.unfitted) == 1


def test_embedded_curve_fit_holdout_isolation_and_point_to_point_ordering(embedded):
    ingested, bundle = embedded
    assert bundle.unfitted == ()
    for coll in COLLECTIVES:
        for ws in WORLD_SIZES:
            scope = "intra_node_ub" if ws <= 8 else "cross_node_urma"
            curve = bundle.lookup_curve(coll, ws, placement_scope=scope, dtype="bf16")
            assert (curve.work_unit, curve.boundary) == ("bytes", "device_kernel")
            assert 0.0 <= curve.startup_ms < min(ms for _, ms in curve.knots)
            assert {w for w, _ in curve.knots}.isdisjoint(HOLDOUT)
            if coll == "all_to_allv":
                assert curve.provenance.get("derived_from") == "all_to_all"
    mutated = deepcopy(ingested)
    for row in mutated["conditions"]:
        if row["payload_bytes"] in HOLDOUT:
            row["per_op_ms"] *= 2.5
    refit = comm.fit_a5_service_curves(mutated)
    assert all((refit.curves[k].model_id, refit.curves[k].knots) == (c.model_id, c.knots)
               for k, c in bundle.curves.items())
    intra = bundle.lookup_curve("p2p_send_recv", 8, placement_scope="intra_node_ub")
    cross = bundle.lookup_curve("p2p_send_recv", 32, placement_scope="cross_node_urma")
    far = bundle.lookup_curve("p2p_send_recv", 32, placement_scope="far")
    i_ms, c_ms, f_ms = (c.estimate(128 * MIB).duration_ms for c in (intra, cross, far))
    assert c_ms < i_ms / 3 and c_ms < f_ms and (cross.identity["p2p_pair"], far.identity["p2p_pair"]) == ("0->8", "0->24")
    for kw, msg in (
        ({"collective": "p2p_send_recv", "world_size": 8, "placement_scope": "cross_node_urma", "pair": "0->1"}, "a5_curve_unavailable"),
        ({"collective": "all_gather", "world_size": 16, "placement_scope": "intra_node_ub"}, "a5_curve_unavailable"),
        ({"collective": "all_gather", "world_size": 8, "placement_scope": "bogus"}, "a5_placement_scope_unknown:bogus"),
    ):
        _raises(ModelingError, msg, lambda k=kw: bundle.lookup_curve(**k))


def test_confirmed_regimes_never_alias_or_contaminate_and_fallback_is_explicit():
    conf = (_mode_rows("aicpu", evidence="hcom_allReduce_AicpuKernel", intercept=0.05, slope=1e-9)
            + _mode_rows("aiv", evidence="AI_VECTOR_CORE", intercept=0.08, slope=2e-9))
    unconf_r = (_mode_rows("aicpu", evidence=None, intercept=5.0, slope=1e-8)
                + _mode_rows("aiv", evidence=None, intercept=5.5, slope=1e-8))
    base = ("all_reduce", 2, "intra_node_ub", "bf16")
    clean, mixed, unconf = _fit(_campaign("c", conf)), _fit(_campaign("c", conf + unconf_r)), _fit(_campaign("u", unconf_r))
    assert clean.curves[base + ("aicpu",)].model_id != clean.curves[base + ("aiv",)].model_id
    for mode in ("aicpu", "aiv"):
        cid = clean.curves[base + (mode,)].model_id
        assert mixed.curves[base + (mode,)].model_id == cid
        assert mixed.lookup_curve("all_reduce", 2, expansion_mode=mode, allow_unspecified_fallback=True).model_id == cid
        _raises(ModelingError, f"a5_expansion_regime_unconfirmed:{mode}",
                lambda m=mode: unconf.lookup_curve("all_reduce", 2, expansion_mode=m))
    assert unconf.lookup_curve("all_reduce", 2, expansion_mode="aicpu",
                               allow_unspecified_fallback=True) is unconf.curves[base + ("unspecified",)]


def test_incompatible_hardware_rejected_across_all_entry_points(embedded):
    ingested, bundle = embedded
    reason, raw = "a5_calibration_rejects_non_a5_hardware", _comm_case()
    op, other = _comm_op(raw), parse_case(_comm_case(hardware=OTHER_HW, tp=8, dp=1, ep=8))
    recipe = bundle.build_a5_comm_recipe("all_gather", world_size=8)
    for bad in ({"hardware_key": "Ascend910_93"}, {"hardware_key": OTHER_HW}, {"arch_family": "A3"}):
        _raises(ModelingError, reason, lambda b=bad: comm.ingest_a5_campaign_data(_campaign(**b)))
        _raises(ModelingError, reason, lambda b=bad: bundle.lookup_curve("all_reduce", 2, **b))
    for fn in (lambda: comm.ingest_a5_campaign_data(_campaign(conditions=[_row(machine={"hardware_key": "Ascend910_93", "arch_family": "A3"})])),
               lambda: comm.fit_a5_service_curves({**deepcopy(ingested), "hardware_key": "Ascend910_93"}),
               lambda: bundle.bind_service_curve(other, op), lambda: bundle.to_service_regime(other, op),
               lambda: bundle.build_a5_comm_recipe("all_gather", world_size=8, hardware_key=OTHER_HW),
               lambda: recipe.build(replace(op, metadata=op.metadata + (("hardware_key", OTHER_HW),)), None),
               lambda: recipe.build(op, None).program(_comm_ctx(op, service_identity={"hardware": {"key": OTHER_HW, "architecture": "A2"}}))):
        _raises(ModelingError, reason, fn)


def test_software_stack_partitioning_and_dedupe_conflict_detection(embedded):
    _, bundle = embedded
    r6 = bundle.lookup_curve("all_gather", 8, placement_scope="intra_node_ub", stack="rack06")
    r8 = bundle.lookup_curve("all_gather", 8, placement_scope="intra_node_ub", stack="rack08")
    assert r6.model_id != r8.model_id and (r6.estimate(128 * MIB).duration_ms, r8.estimate(128 * MIB).duration_ms) == (
        pytest.approx(3.059270), pytest.approx(1.931796))
    _raises(ModelingError, "a5_incompatible_software_stacks_require_selector", lambda: _fit([
        _campaign("ca", stack="custom-a", machine={"cann": "9.9"}, conditions=[
            _row(payload=1024, ms=0.05, run_id="a"), _row(payload=MIB, ms=0.06, run_id="a")]),
        _campaign("cb", stack="custom-b", machine={"cann": "9.9"}, conditions=[
            _row(payload=1024, ms=0.07, run_id="b"), _row(payload=MIB, ms=0.08, run_id="b")])]))
    rows = [_row(payload=p, ms=ms + off, run_id=rid, repeat_index=0)
            for rid, off in (("r1", 0.0), ("r2", 0.02)) for p, ms in ((MIB, 0.06), (4 * MIB, 0.10))]
    ing = comm.ingest_a5_campaign_data(_campaign("reimport", rows))
    key = ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")
    assert comm.fit_a5_service_curves(ing).curves[key].model_id == comm.fit_a5_service_curves(
        {**deepcopy(ing), "conditions": deepcopy(ing["conditions"]) * 2}).curves[key].model_id
    for conflict in (
        [_row("all_gather", run_id="r", repeat_index=0, split="train"),
         _row("all_gather", run_id="r", repeat_index=0, split="heldout")],
        [_row(ms=0.05, run_id="r", repeat_index=0), _row(ms=0.09, run_id="r", repeat_index=0)],
    ):
        _raises(ValueError, "conflicting_primitive_repeat_or_split",
                lambda c=conflict: comm.ingest_a5_campaign_data(_campaign("conflict", c)))


def test_device_kernel_and_resource_phase_boundaries_stay_separate():
    rows = [_row("all_gather", payload=p, ms=b + p * 1e-9, boundary=bnd, run_id=f"{bnd}-{p}", split="train")
            for bnd, b in (("device_kernel", 0.05), ("resource_phase", 0.10)) for p in (1024, MIB, 4 * MIB)]
    bundle, raw = _fit(_campaign("bounds", rows, world_size=8, stack="rack06")), _comm_case()
    parsed, op = parse_case(raw), _comm_op(raw)
    phase_op = replace(op, metadata=op.metadata + (("device_kernel_group", {"g": "c"}),))
    dev_c, phase_c = bundle.bind_service_curve(parsed, op), bundle.bind_service_curve(parsed, phase_op)
    assert (dev_c.boundary, phase_c.boundary, phase_c.provenance["complete_collective_call_response"]) == (
        "device_kernel", "resource_phase", False)
    assert phase_c.estimate(op.link_bytes).duration_ms > dev_c.estimate(op.link_bytes).duration_ms
    _raises(ModelingError, "service_curve_requires_device_or_resource_phase_boundary",
            lambda: comm.ingest_a5_campaign_data(_campaign(boundary="call")))
    _raises(ModelingError, "service_regime_requires_device_kernel_boundary",
            lambda: bundle.to_service_regime(raw, op, boundary="resource_phase"))


def test_wire_factor_binding_identity_regime_and_predict_integration(embedded):
    _, bundle = embedded
    derived = bundle.lookup_curve("all_to_allv", 8, placement_scope="intra_node_ub", dtype="bf16")
    measured = bundle.lookup_curve("all_to_allv", 2, placement_scope="intra_node_ub", dtype="int8")
    assert (comm_bind._wire_factor("all_gather", 8), comm_bind._wire_factor("p2p_send_recv", 8),
            comm_bind._wire_factor("all_to_allv", 2, measured), comm_bind._wire_factor("all_reduce", 8),
            comm_bind._wire_factor("all_to_allv", 8, derived)) == (
        7, 1.0, 1.0, pytest.approx(2 * 7 / 8), pytest.approx(7 / 8))
    raw = _comm_case()
    op, parsed = _comm_op(raw), parse_case(raw)
    bound = bundle.bind_service_curve(parsed, op)
    assert bound.identity == service_identity(parsed, op, "none")
    assert (bound.provenance["wire_factor"], bound.provenance["link_transit_identified"],
            bound.provenance["hop_scaling"]) == (pytest.approx(7), False, "not_derived")
    for fn, msg in ((lambda: bundle.bind_service_curve(parsed, op, resource="vector"), "a5_comm_curve_requires_byte_resource"),
                    (lambda: bundle.bind_service_curve(parsed, op, production=True), "measured_a5_call_binding_is_diagnostic_only"),
                    (lambda: bundle.to_service_regime(raw, op, production=True), "to_service_regime_is_diagnostic_only"),
                    (lambda: bundle.build_a5_comm_recipe("all_gather", world_size=8, production=True), "measured_call_recipe_is_diagnostic_only")):
        _raises(ModelingError, msg, fn)
    other_stack = bundle.bind_service_curve(parsed, op, stack="rack08", cann="9.2.0")
    assert (other_stack.identity["transport"]["stack"], other_stack.identity["transport"]["cann"]) == ("rack08", "9.2.0")
    for cv, crv in ((raw, other_stack), (_comm_case(tp=4, dp=2, ep=8), bound)):
        _raises(ModelingError, "service_identity_mismatch", lambda c=cv, r=crv: operator_costs(c, (op,), recipes=RecipeBook({op.recipe: ResourceService(r)})))
    rule = bundle.to_service_regime(raw, op)
    assert (rule["coordinates"]["bytes"]["scale"], rule["provenance"]["diagnostic_only"], rule["provenance"]["production_eligible"]) == (pytest.approx(1 / 7), True, False)
    cal = deepcopy(raw)
    cal.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [rule]
    treated = graph_costs(cal, (op,), input_ready_ms=7)
    assert treated["duration_ms"] > graph_costs(raw, (op,), input_ready_ms=7)["duration_ms"]
    assert treated["parameters"]["service_regimes"][0]["queries"].get("observable_response") == 1
    assert predict(cal)["metrics"]["e2e_ms"]["p50"] != pytest.approx(predict(raw)["metrics"]["e2e_ms"]["p50"])
    link = dict(parsed.machine.links[0].to_dict())
    def lat_case(f):
        c = deepcopy(raw)
        c["machine"]["links"] = [{**link, "latency_us": float(link["latency_us"]) * f}]
        return c
    times = [operator_costs(c := lat_case(f), (op,), recipes=RecipeBook({op.recipe: ResourceService(bundle.bind_service_curve(c, op))}))["operators"][0]["duration_ms"]
             for f in (0.01, 1.0, 2.0)]
    assert times[0] < times[1] < times[2]


@pytest.mark.parametrize("mode", ["aiv", "aicpu"])
def test_diagnostic_recipe_preserves_work_extrapolates_and_schedules_vector(embedded, mode):
    _, bundle = embedded
    raw, op = _comm_case(), _measure_op(mte2_bytes=2048, mte3_bytes=1024, stream="other")
    vec = OpSpec("vec", "vector", "prefill", 0, 0, (), (1,), (1,), (1,), (1,), vector_ops=100_000_000, dtype="bf16", recipe="vector")
    _raises(ModelingError, f"a5_expansion_regime_unconfirmed:{mode}",
            lambda: bundle.build_a5_comm_recipe("all_gather", world_size=8, expansion_mode=mode).build(op, None))
    recipe = bundle.build_a5_comm_recipe("all_gather", world_size=8, expansion_mode=mode, allow_unspecified_fallback=True)
    plan = recipe.build(op, None)
    applied = plan.work.apply(op)
    assert isinstance(plan, comm_rec._A5CommPlan) and (applied.link_bytes, applied.mte2_bytes, applied.mte3_bytes) == (401408, 2048, 1024)
    _raises(ModelingError, "service_curve_boundary_mismatch", lambda: plan.lower(_comm_ctx(op, service_boundary="resource_phase")))
    b_busy = graph_costs(raw, (vec,))["execution"]["resource_busy_ms"]["vector@r0:s0"]
    c_busy = graph_costs(raw, (op, vec), recipes=RecipeBook({"comm": recipe}))["execution"]["resource_busy_ms"]["vector@r0:s0"]
    if mode == "aiv":
        ar = _measure_op(family="all_reduce", link_bytes=MIB, hbm_bytes=0)
        assert c_busy > b_busy and {bundle.build_a5_comm_recipe("all_reduce", world_size=2, expansion_mode="aiv",
            allow_unspecified_fallback=True, active_aiv_blocks=b).build(ar, None).work.apply(ar).vector_ops for b in (1, 16, 64)} == {524288.0}
    else:
        assert c_busy == pytest.approx(b_busy) and applied.vector_ops == 0
    small = _fit(_campaign("dom", [_row(payload=p, ms=0.05 + p * 1e-9, run_id=f"p{p}", split="train") for p in (1024, 4096)]))
    curve = small.curves[("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")]
    (x0, t0), (x1, t1) = curve.knots[0], curve.knots[-1]
    slope, rec = max((t1 - t0) / (x1 - x0), (t1 - curve.startup_ms) / x1), small.build_a5_comm_recipe("all_reduce", world_size=2)
    assert rec.build(_measure_op(family="allreduce", link_bytes=1_000_000, hbm_bytes=0), None).minimum_ms == pytest.approx(t1 + slope * (1_000_000 - x1))
    assert rec.build(_measure_op(family="allreduce", link_bytes=10, hbm_bytes=0), None).minimum_ms == pytest.approx(curve.startup_ms + (t0 - curve.startup_ms) * 10 / x0)


def test_contention_slowdown_chained_moe_and_followup_manifest(embedded):
    ingested, _ = embedded
    by_b = [comm.compute_a5_contention_slowdown("aiv", concurrent_vector_util=0.8, active_aiv_blocks=b)["vector_slowdown"]
            for b in (8, 24, 64)]
    assert by_b == sorted(by_b) and len(set(by_b)) == 3
    aicpu = comm.compute_a5_contention_slowdown("aicpu", concurrent_vector_util=0.8, concurrent_mte_util=0.6)
    assert aicpu["vector_slowdown"] == 1.0 and aicpu["mte_slowdown"] > 1.0
    _raises(ModelingError, "a5_expansion_mode_unknown", lambda: comm.compute_a5_contention_slowdown("invalid"))
    summary = comm.evaluate_a5_mc2_chains(ingested)
    assert summary["dtypes"] == ["bf16", "fp8-e4m3", "int8"] and summary["overlap_ratio"]["median"] == pytest.approx(0.78, abs=0.01)
    assert all(r["stages"] == CHAIN_STAGES and sum(r["stage_legs_ms"].values()) == pytest.approx(r["sum_leg_p50_ms"])
               and r["overlap_ratio"] == pytest.approx(r["chain_p50_ms"] / r["sum_leg_p50_ms"]) for r in summary["rows"])
    reg = {r["condition"]: r for r in comm.evaluate_a5_mc2_chains(comm.ingest_a5_campaign_data(_campaign("mc", [], chains=[
        _chain("c-aicpu", observed_expansion_mode="aicpu"), _chain("c-aiv", observed_expansion_mode="aiv"),
        _chain("c-u", requested_expansion_mode="aiv", observed_expansion_mode="not_observed")])))["rows"]}
    assert reg["c-aicpu"]["expansion_regime"] == "aicpu" and reg["c-aicpu"]["vector_slowdown"] == 1.0 and reg["c-u"]["expansion_regime"] == "unspecified"
    assert reg["c-aiv"]["expansion_regime"] == "aiv" and reg["c-aiv"]["vector_slowdown"] > 1.0
    first = comm.build_a5_followup_manifest(ingested)
    assert first == comm.build_a5_followup_manifest(ingested) and set(first["pending_cell_ids"]) == {
        "expansion_regime:aicpu/aiv", "collective:W64-8node",
        "p2p_send_recv:0->4:intra_node_ub", "contention:live_sweeps", "all_to_allv:W16-W32"}
    cleared = comm.build_a5_followup_manifest(comm.ingest_a5_campaign_data(_campaign("full", [
        _row(world_size=64, placement_scope="cross_node_urma", observed_expansion_mode="aicpu", run_id="1"),
        _row(world_size=2, placement_scope="intra_node_ub", observed_expansion_mode="aiv", run_id="2"),
        _row("all_to_allv", world_size=16, placement_scope="cross_node_urma", run_id="3"),
        _row("all_to_allv", world_size=32, placement_scope="cross_node_urma", ms=0.2, run_id="4"),
        _row("p2p_send_recv", world_size=8, placement_scope="intra_node_ub", p2p_pair="0->4", run_id="5")])))
    assert cleared["pending_cell_ids"] == ["contention:live_sweeps"]


def test_directory_ingest_chunked_streaming_and_window_isolation(tmp_path, monkeypatch):
    bad_list = tmp_path / "bad-list.json"
    bad_list.write_text(json.dumps([_chain("c1")]), encoding="utf-8")
    with pytest.raises(ValueError, match="a5_campaign_source_list_not_a_campaign"):
        comm.ingest_a5_campaign_data(bad_list)
    bad_list.unlink()
    (tmp_path / "chain-validation.json").write_text(json.dumps(_campaign("chain-validation", [], chains=[_chain("c1")])), encoding="utf-8")
    (tmp_path / "comm-lane.json").write_text(json.dumps({"raw_points": [
        {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.5},
        {"dtype": "int8", "ep": 2, "link_bytes": 2048, "ms": 0.7}]}), encoding="utf-8")
    (tmp_path / "collective-calibration.json").write_text(json.dumps(_campaign("cal")), encoding="utf-8")
    (tmp_path / "dropped.parent.json").write_text(json.dumps({"role": "parent", "label": "dropped", "status": "completed"}), encoding="utf-8")
    dir_ing = comm.ingest_a5_campaign_data(tmp_path)
    assert "dropped" not in dir_ing["campaign_labels"] and "comm-lane" in dir_ing["campaign_labels"]
    raw_b = b'{"a":[12345678901234567890,1.25,-9.5e12],"b":{"c":true,"d":null,"e":"x\\"y"}}'
    for chunk in (1, 2, 3, 7, 64):
        assert comm_str._read_stream_value(comm_str._iter_json_events(io.BytesIO(raw_b), chunk_size=chunk)) == {
            "a": [12345678901234567890.0, 1.25, -9.5e12], "b": {"c": True, "d": None, "e": 'x"y'}}
    p = tmp_path / "big.json"
    p.write_text(json.dumps({"role": "rank", "status": "completed", "label": "streamed", "world_size": 8,
        "dtype": "bf16", "placement": "intra-node-ub", "cases": [{
            "case_id": "c", "collective": "all_gather", "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05}, "raw_iterations_us": [50.0] * 400,
            "samples": [{"payload_bytes": 1024, "per_op_ms": 0.05, "split": "train", "run_id": "s1", "repeat_index": 0},
                        {"payload_bytes": MIB, "per_op_ms": 0.06, "split": "train", "run_id": "s1", "repeat_index": 1},
                        {"payload_bytes": 4 * MIB, "per_op_ms": 9.0, "split": "heldout", "run_id": "s2", "repeat_index": 0}]}]}), encoding="utf-8")
    assert comm_ing._LARGE_JSON_BYTES == 50 * MIB
    monkeypatch.setattr(comm_ing, "_LARGE_JSON_BYTES", 64)
    monkeypatch.setitem(sys.modules, "ijson", None)
    monkeypatch.setattr(comm_ing.json, "load", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    assert "raw_iterations_us" not in comm_ing._load_json(p)["cases"][0] and {r["work"] for r in _fit(p).holdout} == {4 * MIB}
    stream = lambda doc: comm_str._read_stream_value(comm_str._iter_json_events(io.BytesIO(json.dumps(doc).encode())))
    dup = ([{"payload_bytes": 1024, "median_ms": 1.0, "run_id": "r", "repeat_index": 0}] * 200
           + [{"payload_bytes": 1024, "median_ms": 3.0, "run_id": "r", "repeat_index": i} for i in range(1, 100)])
    assert comm.ingest_a5_campaign_data(stream(_campaign(conditions=[{"collective": "all_reduce", "samples": dup}])))[
        "conditions"][0]["per_op_ms"] == pytest.approx(3.0)
    _raises(ValueError, "conflicting_primitive_repeat_or_split", lambda: comm.ingest_a5_campaign_data(stream(_campaign(
        conditions=[{"collective": "all_reduce", "samples": dup + [{"payload_bytes": 1024, "median_ms": 9.0, "run_id": "r", "repeat_index": 0}]}]))))
    meas, win = _child_desc()
    child_doc = _campaign("child", world_size=2, observed_expansion_mode="aicpu", conditions=[
        {"collective": "all_gather", "payload_bytes": w, "samples": [
            {"per_op_ms": ms, "repeat_index": i, "measurement": meas, "transport_window": win} for i in range(280)]}
        for w, ms in ((32768, 0.041), (10 * MIB, 0.059))])
    c_path = tmp_path / "child.json"
    c_path.write_text(json.dumps(child_doc), encoding="utf-8")
    curve = _fit(comm_str._stream_json(c_path)).curves[("all_gather", 2, "intra_node_ub", "bf16", "aicpu")]
    assert (curve.resource, curve.provenance["transport_window"]["kind"]) == ("aicpu_child", "native_single_target_call")
    bad = deepcopy(child_doc)
    bad["conditions"][0]["samples"] = [{"per_op_ms": 0.041, "measurement": meas,
                                        "transport_window": {"kind": "campaign_steady_state", "phase": "per_op"}}]
    _raises(ValueError, "a5_measurement_transport_window_mismatch", lambda: comm.ingest_a5_campaign_data(bad))
    _raises(ValueError, "a5_measurement_boundary_mixed_sources", lambda: _fit(_campaign("mixed", world_size=2, conditions=[
        {"collective": "all_gather", "payload_bytes": 32768, "samples": [{"per_op_ms": 0.041, "measurement": meas, "transport_window": win}]},
        {"collective": "all_gather", "payload_bytes": MIB, "samples": [{"per_op_ms": 0.05}]}])))


def test_native_local_service_fit_applicability_lineage_and_supply_change(native_wave):
    path, image_id, manifest_sha = native_wave
    base_kw = dict(collective="all_gather", stack="rack08", cann="9.2.0", deployment_image_id=image_id, group_size=2)
    rec = comm.fit_a5_native_local_service(str(path), **base_kw)
    svc = rec["service"]
    assert rec["schema"] == comm.A5_COMM_LOCAL_SERVICE_SCHEMA and rec["normalized_schema"] == RESOURCE_SERVICE_SCHEMA
    assert rec["diagnostic_only"] is True and rec["production_eligible"] is False
    assert rec["overlap_with_communication"] is True and rec["child_is_complete_collective"] is False
    assert svc["scope"] == "independent_resource" and svc["boundary"] == "resource_phase" and svc["rate_scope"] == "per_owner"
    assert svc["active_units"] == 2 and rec["lineage_key"]["source.manifest_sha256"] == manifest_sha
    src, dumped = rec["source"], json.dumps(rec)
    assert src["capture_uri"].startswith("capture://") and src["source_ref"]["source_sha256"] == src["summary_sha256"]
    assert set(src["source_ref"]) == {"repository", "revision", "relative_path", "source_sha256"}
    assert not (set(src) & {"path", "manifest_path", "versions_path", "environment_path", "topology_path", "host", "environment_image_id"})
    assert all(tok not in dumped for tok in (str(path), str(path.parent), "rack08os2", image_id))
    pred = comm.a5_local_service_prediction(rec, 200_000_000, work_unit="bytes")
    assert pred["extrapolated"] is True and pred["duration_us"] == pytest.approx(svc["fixed_us"] + 200_000_000 * 1e6 / svc["effective_rate"])
    _raises(ValueError, "a5_local_service_work_unit_mismatch", lambda: comm.a5_local_service_prediction(rec, 1024, work_unit="rows"))
    lin, bad_t, miss = deepcopy(rec["domain"]), deepcopy(rec["domain"]), deepcopy(rec["domain"])
    lin["source"]["manifest_sha256"], bad_t["software"]["torch_npu"] = "0" * 64, "2.11.0"
    del miss["software"]["hccl_protocol"]
    assert comm.a5_local_service_prediction(rec, MIB, domain=lin, work_unit="bytes")["basis"] == "formula"
    assert obs_cal.service_conditions_match(svc, lin)["applies"] is True
    faster = comm.a5_local_service_supply_change(rec, 20_000_000, effective_rate=svc["effective_rate"] * 1.2, domain=rec["domain"])
    assert faster["domain_checked"] is True and faster["duration_us"] < comm.a5_local_service_prediction(rec, 20_000_000)["duration_us"]
    for exc, msg, fn in (
        (ModelingError, "lineage", lambda: comm.a5_local_service_supply_change(rec, MIB, effective_rate=svc["effective_rate"], domain=lin)),
        (ModelingError, "software.torch_npu", lambda: comm.a5_local_service_supply_change(rec, 1024, effective_rate=1.0, domain=bad_t)),
        (ModelingError, "software.hccl_protocol", lambda: comm.require_a5_local_service_domain(rec, miss)),
        (ValueError, "a5_local_service_hardware_identity_conflict", lambda: comm.a5_local_service_supply_change(
            rec, 1024, effective_rate=1.0, domain={**deepcopy(rec["domain"]), "hardware": "Ascend950PR_9579"})),
    ):
        _raises(exc, msg, fn)
    prof = {"schema": RESOURCE_SERVICE_PROFILE_SCHEMA, "hardware": comm.A5_HARDWARE_KEY, "services": [svc]}
    assert obs_cal.load_resource_profile(prof, hardware=comm.A5_HARDWARE_KEY)["service_ids"] == [svc["id"]]
    for over, err in (({"stack": "rack06"}, "native_service_environment_stack_mismatch"),
                      ({"cann": "9.1.0"}, "native_service_cann_mismatch"),
                      ({"deployment_image_id": "sha256:" + "0" * 64}, "native_service_environment_image_mismatch"),
                      ({"execution_core": "aiv"}, "native_service_requires_aicpu_active_child"),
                      ({"placement": "cross_node_urma"}, "native_service_requires_intra_node_ub"),
                      ({"collective": "all_reduce"}, "native_local_service_collective_unsupported"),
                      ({"dtype": "fp8-e4m3"}, "native_local_service_requires_bf16")):
        _raises(ValueError, err, lambda o=over: comm.fit_a5_native_local_service(str(path), **{**base_kw, **o}))


def test_native_child_binding_stage_evidence_and_envelope_provenance(native_wave, embedded):
    path, image_id, _ = native_wave
    bundle = comm.collect_native_service(str(path), stack="rack08", cann="9.2.0", execution_core="aicpu",
                                         placement="intra_node_ub", deployment_image_id=image_id, groups=(2,))
    raw = _comm_case(tp=2, dp=1, ep=2)
    op, sel = _comm_op(raw), dict(expansion_mode="aicpu", stack="rack08", cann="9.2.0", placement_scope="intra_node_ub")
    bound = bundle.bind_service_curve(raw, op, **sel)
    assert (bound.provenance["consumer_role"], bound.provenance["complete_collective_call_response"]) == ("aicpu_child_completion_floor", False)
    for fn in (lambda: bundle.bind_service_curve(raw, op, require_complete_call=True, **sel),
               lambda: bundle.to_service_regime(raw, op, require_complete_call=True, **sel),
               lambda: bundle.build_a5_comm_recipe("all_gather", world_size=2, require_complete_call=True, **sel).build(op, None)):
        _raises(ModelingError, "cannot_serve_complete_collective_call", fn)
    ev = comm.native_child_stage_evidence(str(path), collective="all_gather", stack="rack08", cann="9.2.0",
        deployment_image_id=image_id, group_size=2, split="heldout", used_for_tuning=("native.all_gather.g2.786432.heldout",))
    assert (ev["schema"], ev["summary"]["used_for_tuning_count"], ev["summary"]["all_child_within_communication"],
            ev["summary"]["fixed_term_exceeds_some_child_calls"]) == (comm.A5_NATIVE_CHILD_EVIDENCE_SCHEMA, 6, True, True)
    ev_dump = json.dumps(ev)
    assert ev["source"]["capture_uri"].startswith("capture://") and all(tok not in ev_dump for tok in (str(path), "rack08os2", image_id))
    assert all((c.provenance["link_transit_identified"], c.provenance["hop_scaling"],
                c.provenance["measurement_boundary"]["complete_collective_call"]) == (
        False, "not_derived", k[0] != "p2p_send_recv") for k, c in embedded[1].curves.items())


def test_normalized_service_stage_owner_equivalence_and_engine_sharing(native_wave):
    ident, ctx = {"hardware": {"key": "TEST"}}, {"hardware": "TEST"}
    svc = normalize_resource_service({"id": "svc-link", "scope": "independent_resource", "resources": ["link"],
        "phase": "transfer", "work_unit": "bytes", "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "active_units": 4, "rate_scope": "per_owner", "supply": {"kind": "absolute_rate", "rate": 24_000_000},
        "conditions": ctx, "provenance": {"legacy_target_ms": 9e6}})
    assert (comm.normalized_service_duration_ms(svc, 1536, work_scope="aggregate"),
            comm.normalized_service_duration_ms(svc, 384, work_scope="per_owner")) == (pytest.approx(0.018), pytest.approx(0.018))
    st = lambda w, sc: {"work": w, "work_scope": sc, "active_units": 4, "phase": "transfer", "work_unit": "bytes", "boundary": "resource_phase", "resource": "link"}
    agg = comm.build_normalized_service_stage(svc, identity=ident, conditions_context=ctx, require_conditions=True, stage=st(1536, "aggregate"))
    own = comm.build_normalized_service_stage(svc, identity=ident, conditions_context=ctx, require_conditions=True, stage=st(384, "per_owner"))
    assert (agg.duration_us, own.duration_us, agg.stage_cap_work_per_ms, own.stage_cap_work_per_ms) == (
        pytest.approx(18.0), pytest.approx(18.0), pytest.approx(96_000), pytest.approx(24_000))
    assert comm.build_normalized_service_recipe(svc, identity=ident, resource="link", conditions_context=ctx).model.estimate(3072).duration_ms == pytest.approx(0.034)
    joint = normalize_resource_service({"id": "svc-joint", "scope": "joint_phase", "resources": ["hbm", "mte2"],
        "phase": "joint", "work_unit": "bytes", "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 96_000_000}, "conditions": ctx})
    j_stage = comm.build_normalized_service_stage(joint, identity=ident, conditions_context=ctx, require_conditions=True,
        stage={"work": 1536, "work_scope": "aggregate", "active_units": 1, "phase": "joint",
               "work_unit": "bytes", "boundary": "resource_phase", "resources": ["hbm", "mte2"]})
    assert j_stage.kind == "joint_phase" and j_stage.resource_service is None
    _raises(ModelingError, "joint_phase_requires_joint_stage",
            lambda: comm.service_curve_from_normalized(joint, identity=ident, resource="link", conditions_context=ctx))
    path, image_id, _ = native_wave
    child_rec = comm.fit_a5_native_local_service(str(path), collective="all_gather", stack="rack08", cann="9.2.0", deployment_image_id=image_id, group_size=2)
    child_svc = normalize_resource_service(child_rec["service"])
    _raises(ModelingError, "normalized_service_resource_mismatch", lambda: comm.service_curve_from_normalized(
        child_svc, identity=ident, resource="link", conditions_context=child_rec["domain"]))
    _raises(ModelingError, "normalized_service_resource_not_physical", lambda: comm.build_normalized_service_stage(
        child_svc, identity=ident, conditions_context=child_rec["domain"], require_conditions=True,
        stage={"work": 786432, "work_scope": "per_owner", "active_units": 2, "phase": child_svc.phase,
               "work_unit": "bytes", "boundary": child_svc.boundary, "resource": "aicpu_child"}))
    for resources, pairs, expected in (([Resource("link", 96_000, "B", "fair")], (("a", "link"), ("b", "link")), 0.034),
                                       ([Resource("tx", 96_000, "B", "fair"), Resource("rx", 96_000, "B", "fair")], (("x", "tx"), ("y", "rx")), 0.018)):
        eng = Engine(resources)
        for name, res in pairs:
            s = eng.submit(name + ":s", latency=agg.fixed_us / 1000.0, release=0)
            done = eng.submit(name, {res: agg.work}, after=(s,), caps={res: agg.stage_cap_work_per_ms}, release=0)
        eng.advance(until=(done,))
        assert done.time == pytest.approx(expected)


def test_lazy_package_imports_and_reexport_identity():
    obs_init = ast.parse((ROOT / "mindie_sim/observations/__init__.py").read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(obs_init) if isinstance(n, (ast.Import, ast.ImportFrom))]
    comm_init = ast.parse((ROOT / "mindie_sim/observations/communication/__init__.py").read_text(encoding="utf-8"))
    assert {(n.module if isinstance(n, ast.ImportFrom) else a.name)
            for n in comm_init.body if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in n.names} <= {"__future__", "importlib"}
    cal_tree = ast.parse((ROOT / "mindie_sim/observations/calibration.py").read_text(encoding="utf-8"))
    assert all("communication" not in (n.module or "") and "a5_comm" not in (n.module or "")
               for n in cal_tree.body if isinstance(n, ast.ImportFrom))
    assert all(n in comm.__all__ and getattr(obs_cal, n) is getattr(comm, n) for n in obs_cal._A5_REEXPORTS)
