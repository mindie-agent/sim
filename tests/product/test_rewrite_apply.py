"""Exercise the rewrite wire algebra and transactional graph edits.

[Role] Decode closed JSON plans and apply each opcode to lowered step graphs.
[Invariants] Edits preserve declared work, graph cuts, and physical ownership.
[Timeouts & Locks] Small single-step fixtures and deterministic plans.
[Side-Effect Policy] All plans and mutated fixtures stay in memory.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import gcd

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.compose import compose_case
from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.fusion.ports import declared_control_order, full_operand_ports
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileCandidate, TileShape
from mindie_sim.prediction.streams import bind_stream_owners
from mindie_sim.rewrite import (
    CollectiveTransform, Edit, FuseRegion, HoistOrDeduplicate, KernelBoundary, OpRef,
    PipelineTiles, PortRef, QuantizeContract, RewritePlan, SelectTemplate, StreamCoreBudget,
    ScenarioRequest, StreamEvent, StreamLane, StreamSchedule, TilePipelineSpec, WeightPrefetch, apply_plan,
    compute_overlap_pct, decode_rewrite_plan, empty_plan, encode_rewrite_plan, finalize_plan, plan_sha256,
    prepare_scenario, resolve_scenario, verify_plan,
)
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0


def _atom(op_id):
    return OpRef("atom", op_id)


def _plan(scenario, *edits):
    return replace(empty_plan(scenario.base_graph_sha256, scenario.framing_sha256), edits=edits)


def _pipeline(family):
    buffer = LiveBufferSpec("cohort", "ub", 1024, 0, 1)
    tile = TileCandidate("resident_1pass", TileShape(16, 16, 16), 2, (1, 1), 2, 2,
                         (buffer,), owner_axis="aiv")
    return TilePipelineSpec(family, (tile,))


def _fuse(*ids, preserve=()):
    refs = tuple(map(_atom, ids))
    return FuseRegion(refs, tuple(PortRef(_atom(op), "output", 0) for op in preserve),
                      (KernelBoundary("test_abi", refs),))


@pytest.fixture(scope="module")
def dense():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    return resolve_scenario(case, StepRequest("decode", tokens=4, context=64))


@pytest.fixture(scope="module")
def routed():
    case = compose_case("qwen3-30b-a3b", REPRESENTATIVE_SKUS["a5"], tp=2, ep=2, layers=1)
    return resolve_scenario(case, StepRequest("decode", tokens=4, context=64))


@pytest.fixture
def wire_plan():
    a, b = _atom("producer"), _atom("consumer")
    operations = (
        _fuse(a.id, b.id, preserve=(a.id,)),
        SelectTemplate(OpRef("result_of", "edit-0"), "K2", "gemm", (("stages", 2), ("flag", True))),
        PipelineTiles(b, _pipeline("alltoall"), "dbo_2cohort"),
        StreamSchedule((StreamLane("main"), StreamLane("aux")), ((b, "aux"),),
                       (StreamEvent("ready", "main", a.id, "aux", (b.id,)),)),
        WeightPrefetch(PortRef(b, "weight", 0), PortRef(a, "output", 0), max_bytes=512),
        CollectiveTransform(b, "ring", "fabric"),
        QuantizeContract((a.id, b.id), "bf16", "bf16", "bf16", 32, "fp32", (b.id,)),
        HoistOrDeduplicate((a, b), "load_time"),
    )
    edits = tuple(Edit(f"edit-{i}", ("edit-2", "edit-0") if i == 7 else (), op)
                  for i, op in enumerate(operations))
    return RewritePlan(base_graph_sha256="a" * 64, framing_sha256="b" * 64, edits=edits)


def test_closed_wire_roundtrip_and_canonical_digest(wire_plan):
    payload = encode_rewrite_plan(wire_plan)
    assert payload["schema"] == "mindie-rewrite-plan/1"
    assert len({edit["op"] for edit in payload["edits"]}) == 8
    assert decode_rewrite_plan(json.loads(json.dumps(payload))) == wire_plan
    reversed_plan = replace(wire_plan, edits=tuple(
        replace(edit, after=tuple(reversed(edit.after))) for edit in reversed(wire_plan.edits)
    ))
    digest = plan_sha256(wire_plan)
    assert digest == plan_sha256(reversed_plan)
    edits = sorted(payload["edits"], key=lambda edit: edit["edit_id"])
    for edit in edits:
        edit["after"].sort()
    canonical = json.dumps({**payload, "edits": edits}, sort_keys=True, separators=(",", ":"))
    assert digest == sha256(canonical.encode()).hexdigest()
    assert plan_sha256(replace(wire_plan, framing_sha256="c" * 64)) != digest


@pytest.mark.parametrize(("mutate", "message"), (
    (lambda p: p.update(extra=True), "wire_fields_invalid:plan"),
    (lambda p: p["edits"][0].update(extra=True), "wire_fields_invalid:FuseRegion"),
    (lambda p: p["edits"][0].update(op="Unexpected"), "unknown_opcode:Unexpected"),
    (lambda p: p["edits"][0].update(op="DualStreamOverlap"), "unknown_opcode:DualStreamOverlap"),
    (lambda p: p["edits"][0].update(op="MicrobatchSplit"), "unknown_opcode:MicrobatchSplit"),
    (lambda p: p["edits"].append(p["edits"][0]), "duplicate_edit_id"),
    (lambda p: p["edits"][0].update(after=["edit-7"]), "edit_dependency_cycle"),
    (lambda p: p["edits"][6].update(exclusions=["other"]), "quant_exclusions_outside_scope"),
    (lambda p: p["edits"][3]["lanes"].pop(0), "stream_lanes_invalid"),
    (lambda p: p["edits"][4].update(max_bytes=float("nan")), "wire_type_invalid:WeightPrefetch.max_bytes"),
    (lambda p: p["edits"][4].update(max_bytes=float("inf")), "wire_type_invalid:WeightPrefetch.max_bytes"),
))
def test_wire_rejects_invalid_edits_and_fields(wire_plan, mutate, message):
    payload = encode_rewrite_plan(wire_plan)
    mutate(payload)
    with pytest.raises(ModelingError, match=message):
        decode_rewrite_plan(payload)


def test_encoder_rejects_nonfinite_parameter(wire_plan):
    edit = wire_plan.edits[1]
    invalid = replace(edit, operation=replace(edit.operation, params=(("stages", float("-inf")),)))
    with pytest.raises(ModelingError, match="rewrite_wire_value_invalid"):
        encode_rewrite_plan(replace(wire_plan, edits=(invalid,)))


@pytest.mark.parametrize("field", ("base_graph_sha256", "framing_sha256"))
def test_digest_must_match_resolved_graph_and_frame(dense, field):
    with pytest.raises(ModelingError, match=f"rewrite_digest_mismatch:{field}"):
        apply_plan(replace(_plan(dense), **{field: "f" * 64}), dense)


def test_fusion_preserves_external_cut_and_chained_handles(dense):
    original = {op.op_id: op for op in dense.atomic_ops[0].ops}
    qkv, norm, rope, value = (f"s0.l0.{suffix}" for suffix in ("qkv", "qk_norm", "rope", "qkv.value"))
    fuse = Edit("contract", operation=_fuse(qkv, norm, preserve=(qkv,)))
    first = Edit("hint", operation=SelectTemplate(OpRef("result_of", "contract"), "K1", "base"))
    second = Edit("tune", operation=SelectTemplate(OpRef("result_of", "hint"), "K2", "tuned", (("stages", 2),)))
    norm1, = original[qkv].deps
    independent = Edit("before", operation=SelectTemplate(_atom(norm1), "K3", "rms"))
    plan = _plan(dense, second, first, fuse, independent)
    result = apply_plan(plan, dense)
    assert tuple(r.edit_id for r in result.edit_receipts) == ("before", "contract", "hint", "tune")
    reordered = apply_plan(replace(plan, edits=tuple(reversed(plan.edits))), dense)
    assert result == replace(reordered, plan=plan)
    graph = {op.op_id: op for op in result.executable[0].ops}
    fused = graph["rewrite:contract"]
    assert dict(result.absorbed_atoms)[fused.op_id] == (qkv, norm)
    assert all(dict(result.handles)[name] == (fused.op_id,) for name in ("contract", "hint", "tune"))
    assert {qkv, norm}.isdisjoint(graph)
    assert fused.deps == original[qkv].deps
    assert "output" in fused.returns
    assert graph[rope].deps == graph[value].deps == (fused.op_id,)
    assert fused.flops == original[qkv].flops + original[norm].flops
    assert fused.vector_ops == original[qkv].vector_ops + original[norm].vector_ops
    assert [binding.tier for binding in result.template_bindings] == ["K3", "K1", "K2"]
    for op in (fused, graph[norm1]):
        assert "occurrence" not in dict(op.metadata)
        assert all(type(key) is str and type(val) in {str, int, float, bool} for key, val in op.metadata)
    assert dict(fused.metadata)["template_param_stages"] == 2


def test_nonconvex_edit_rolls_back_earlier_edits(dense):
    before = apply_plan(_plan(dense), dense)
    early = Edit("early", operation=SelectTemplate(_atom("s0.l0.rope"), "K3", "rotary"))
    invalid = Edit("invalid", ("early",), _fuse("s0.l0.qkv", "s0.l0.rope"))
    with pytest.raises(ModelingError, match="NON_CONVEX_REGION:invalid"):
        apply_plan(_plan(dense, early, invalid), dense)
    assert apply_plan(_plan(dense), dense) == before
    assert dict(apply_plan(_plan(dense, early), dense).handles)["early"] == ("s0.l0.rope",)


def test_fusion_rejects_undeclared_live_out_cut(dense):
    region = _fuse("s0.l0.qkv", "s0.l0.qk_norm")
    with pytest.raises(ModelingError, match="rewrite_fusion_live_out_unpreserved"):
        apply_plan(_plan(dense, Edit("contract", operation=region)), dense)


def test_result_handles_cannot_create_an_implicit_edit_cycle(dense):
    first = Edit("first", operation=SelectTemplate(OpRef("result_of", "second"), "K1", "one"))
    second = Edit("second", operation=SelectTemplate(OpRef("result_of", "first"), "K2", "two"))
    with pytest.raises(ModelingError, match="rewrite_edit_dependency_cycle"):
        apply_plan(_plan(dense, first, second), dense)


def test_tile_cohort_and_collective_route_bind_to_physical_graph(dense, routed):
    dispatch = next(op for op in routed.atomic_ops[0].ops if op.family == "alltoall")
    route = routed.case.machine.links[0].name
    tiles = Edit("tile", operation=PipelineTiles(_atom(dispatch.op_id), _pipeline(dispatch.family), "dbo_2cohort"))
    communication = Edit("route", operation=CollectiveTransform(OpRef("result_of", "tile"), "ring", route))
    applied = apply_plan(_plan(routed, communication, tiles), routed)
    binding, = applied.tile_bindings
    assert binding.tile_spec == tiles.operation.tile_spec
    record, = applied.collective_rewrites
    assert (record.target_op, record.mode, record.topology_route) == (dispatch.op_id, "ring", route)
    op = next(op for op in applied.executable[0].ops if op.op_id == dispatch.op_id)
    assert dict(op.metadata)["cohort_split"] == "dbo_2cohort"
    assert dict(op.metadata)["tile_algorithm"] == "resident_1pass"
    assert dict(op.metadata)["topology_route"] == route
    assert op.link_bytes == dispatch.link_bytes > 0
    bad_route = replace(communication, operation=replace(communication.operation, topology_route="invented"))
    with pytest.raises(ModelingError, match="rewrite_collective_route_unknown:invented"):
        apply_plan(_plan(routed, tiles, bad_route), routed)
    wrong = replace(tiles, operation=replace(tiles.operation, target=_atom("s0.l0.qkv")))
    with pytest.raises(ModelingError, match="rewrite_dbo_family_unsupported"):
        apply_plan(_plan(dense, wrong), dense)


def test_tile_cohort_admits_lowered_collective_protocol():
    case = compose_case("deepseek-v4-flash", REPRESENTATIVE_SKUS["a5"], tp=2, ep=2, layers=1)
    scenario = resolve_scenario(case, StepRequest("decode", tokens=4, context=64))
    dispatch = next(op for op in scenario.atomic_ops[0].ops
                    if op.family == "mc2_protocol" and op.op_id.endswith(".dispatch"))
    assert dispatch.local_shape[0] >= 2
    tile = Edit("cohorts", operation=PipelineTiles(_atom(dispatch.op_id), _pipeline(dispatch.family), "dbo_2cohort"))
    applied = apply_plan(_plan(scenario, tile), scenario)
    assert applied.tile_bindings[0].cohort_split == "dbo_2cohort"


def test_three_stream_gcd_core_lattice_and_zero_aic_lane(dense):
    path = dense.datapath
    common = gcd(path.aic_count, path.aiv_count)
    cube_step, vector_step = path.aic_count // common, path.aiv_count // common
    lanes = (StreamLane("main", StreamCoreBudget(path.aic_count - cube_step, path.aiv_count - vector_step)),
             StreamLane("csa", StreamCoreBudget(0, vector_step)),
             StreamLane("comm", StreamCoreBudget(0, 0)))
    qkv, norm, rope, value, cache = (f"s0.l0.{name}" for name in ("qkv", "qk_norm", "rope", "qkv.value", "cache"))
    schedule = StreamSchedule(lanes, ((_atom(norm), "csa"), (_atom(rope), "csa"), (_atom(value), "comm")),
                              (StreamEvent("qkv_ready", "main", qkv, "csa", (norm,)),
                               StreamEvent("rotary_done", "csa", rope, "comm", (value,)),
                               StreamEvent("copy_done", "comm", value, "main", (cache,))))
    plan = _plan(dense, Edit("schedule", operation=schedule))
    result = apply_plan(plan, dense)
    stream, = result.streams
    owners = bind_stream_owners(stream, path)
    assert stream.lane_ids() == ("main", "csa", "comm")
    assert (owners["csa"].aic_limit, owners["csa"].aiv_limit) == (0, vector_step)
    assert not owners["csa"].pool_aic and not owners["comm"].pool_aiv
    assert owners["main"].aic_limit * vector_step == owners["main"].aiv_limit * cube_step
    ops = {op.op_id: op for op in result.executable[0].ops}
    assert ops[rope].stream == "csa" and ops[rope].flops == 0 < ops[rope].vector_ops
    assert ops[value].stream == "comm" and rope in ops[value].deps
    assert value in ops[cache].deps
    assert all(event.record_after != rope or cache not in event.wait_before for event in stream.events)
    missing = replace(schedule, events=tuple(e for e in schedule.events if e.event_id != "rotary_done"))
    with pytest.raises(ModelingError, match="STREAM_EVENT_UNCOVERED"):
        apply_plan(_plan(dense, Edit("schedule", operation=missing)), dense)


def test_prefetch_clamps_to_l1_and_conserves_bytes(dense):
    norm, qkv, *_ = dense.atomic_ops[0].ops
    capacity = dense.datapath.usable_bytes("l1")
    assert dict(qkv.metadata)["weight_bytes"] > capacity > 0
    edit = Edit("prefetch", operation=WeightPrefetch(
        PortRef(_atom(qkv.op_id), "weight", 0), PortRef(_atom(norm.op_id), "output", 0),
        max_bytes=2 * capacity,
    ))
    applied = apply_plan(_plan(dense, edit), dense)
    lease, = applied.prefetch_leases
    assert (lease.target_op, lease.window_anchor_op, lease.max_bytes) == (qkv.op_id, norm.op_id, capacity)
    target = next(op for op in applied.executable[0].ops if op.op_id == qkv.op_id)
    meta = dict(target.metadata)
    assert target.hbm_bytes == qkv.hbm_bytes == meta["weight_prefetch_foreground_bytes"] + capacity
    assert meta["weight_prefetch_bytes"] == capacity

    noop = replace(edit, operation=replace(edit.operation, max_bytes=0))
    zero = apply_plan(_plan(dense, noop), dense)
    assert not zero.prefetch_leases and not zero.edit_receipts[0].affected_ops
    assert zero.executable == apply_plan(_plan(dense), dense).executable


def test_quantization_projection_respects_exclusions_and_checkpoint(dense):
    qkv = next(op for op in dense.atomic_ops[0].ops if op.family == "qkv_projection")
    quant = dense.case.model.quant
    spec = QuantizeContract((qkv.family, "rmsnorm"), quant.weight_format, quant.activation_format,
                            quant.kv_dtype, quant.activation_group_size, quant.activation_scale_format, ("rmsnorm",))
    result = apply_plan(_plan(dense, Edit("quantize", operation=spec)), dense)
    assert result.quant_contracts == (spec,)
    receipt, = result.edit_receipts
    assert receipt.affected_ops == (qkv.op_id,)
    for op in result.executable[0].ops:
        meta = dict(op.metadata)
        if op.op_id == qkv.op_id:
            assert (meta["quant_w_dtype"], meta["quant_a_dtype"], meta["quant_kv_dtype"],
                    meta["quant_group_size"], meta["quant_scale_dtype"]) == (
                        spec.w_dtype, spec.a_dtype, spec.kv_dtype, spec.group_size, spec.scale_dtype)
        else:
            assert "quant_w_dtype" not in meta
    with pytest.raises(ModelingError, match="rewrite_quant_precision_mismatch"):
        apply_plan(_plan(dense, Edit("quantize", operation=replace(spec, w_dtype="bf16"))), dense)


def test_hoist_load_time_and_deduplicate_intra_step_preserve_users(dense):
    rank, = dense.atomic_ops
    original, consumer, *rest = rank.ops
    pure = replace(original, metadata=(*original.metadata, ("pure", True)))
    duplicate = replace(pure, op_id="same_read")
    constant = replace(pure, op_id="immutable_shape", deps=(), metadata=(("immutable", True),))
    consumer = replace(consumer, deps=(pure.op_id, duplicate.op_id, constant.op_id))
    declared = replace(dense, atomic_ops=(replace(rank, ops=(pure, duplicate, constant, consumer, *rest)),))
    edits = (Edit("dedup", operation=HoistOrDeduplicate((_atom(duplicate.op_id), _atom(pure.op_id)), "intra_step")),
             Edit("hoist", operation=HoistOrDeduplicate((_atom(constant.op_id),), "load_time")))
    applied = apply_plan(_plan(declared, *edits), declared)
    ops = {op.op_id: op for op in applied.executable[0].ops}
    assert ops[consumer.op_id].deps == (pure.op_id,)
    assert duplicate.op_id not in ops and constant.op_id not in ops
    assert applied.hoisted_load_time == (constant.op_id,)
    invalid = Edit("hoist", operation=HoistOrDeduplicate((_atom(pure.op_id),), "load_time"))
    with pytest.raises(ModelingError, match="rewrite_load_time_dynamic_input"):
        apply_plan(_plan(declared, invalid), declared)


def test_group_list_elision_replaces_host_sync_contract(routed):
    router = next(op for op in routed.atomic_ops[0].ops if op.family == "router")
    edit = Edit("elide", operation=HoistOrDeduplicate((_atom(router.op_id),), "device_group_list_elide_d2h"))
    applied = apply_plan(_plan(routed, edit), routed)
    assert applied.elided_d2h_ops == (router.op_id,)
    updated = next(op for op in applied.executable[0].ops if op.op_id == router.op_id)
    meta = dict(updated.metadata)
    assert meta["elide_d2h"] is meta["device_group_list"] is True
    assert meta["eager_d2h_sync"] is meta["requires_host_split_sizes"] is False
    assert (updated.deps, updated.flops) == (router.deps, router.flops)


def test_fusion_redirects_control_and_typed_edges_without_flattening_ports(dense):
    rank, = dense.atomic_ops
    norm, matmul, user, *_ = rank.ops
    release = replace(norm, op_id="release", family="release", deps=(), metadata=())
    producer = replace(norm, op_id="producer", deps=(), metadata=())
    operand = {"producer": producer.op_id, "port": "output", "version": 2}
    a = replace(norm, op_id="a", deps=(), metadata=(
        ("control_deps", (release.op_id,)), ("typed_inputs", (operand,)),
    ))
    b = replace(matmul, op_id="b", deps=(a.op_id,))
    typed = {"producer": b.op_id, "port": "output", "version": 1}
    user = replace(user, op_id="user", deps=(), metadata=(
        ("control_deps", (b.op_id,)), ("typed_inputs", (typed,)),
    ))
    scenario = replace(dense, atomic_ops=(replace(rank, ops=(release, producer, a, b, user)),))
    plan = _plan(scenario, Edit("region", operation=_fuse(a.op_id, b.op_id, preserve=(b.op_id,))))
    result = apply_plan(plan, scenario)
    release_after, producer_after, region, user_after = result.executable[0].ops
    assert (release_after, producer_after) == (release, producer)
    assert set(region.deps) == {release.op_id, producer.op_id}
    assert declared_control_order(region.metadata) == (release.op_id,)
    assert dict(region.metadata)["typed_inputs"] == (operand,)
    assert user_after.deps == declared_control_order(user_after.metadata) == (region.op_id,)
    assert dict(user_after.metadata)["typed_inputs"] == ({**typed, "producer": region.op_id},)


def test_fusion_keeps_distinct_live_outputs_for_each_producer_and_version(dense):
    rank, = dense.atomic_ops
    first, second, consumer, *_ = rank.ops
    keys = (("a", 0), ("b", 0), ("a", 1), ("b", 1))
    ports = {key: {"producer": key[0], "port": "output", "version": key[1]} for key in keys}

    def producer(op, name, deps):
        outputs = tuple({**port, "value": key} for key, port in ports.items() if key[0] == name)
        return replace(op, op_id=name, deps=deps, metadata=(("typed_outputs", outputs),))

    a, b = producer(first, "a", ()), producer(second, "b", ("a",))
    user = replace(consumer, op_id="user", deps=("a", "b"), metadata=(
        ("typed_inputs", tuple(ports.values())),
    ))
    scenario = replace(dense, atomic_ops=(replace(rank, ops=(a, b, user)),))
    region = replace(_fuse("a", "b"), preserve_ports=tuple(
        PortRef(_atom(producer), "output", version) for producer, version in keys
    ))
    result = apply_plan(_plan(scenario, Edit("region", operation=region)), scenario)
    fused, rewritten = result.executable[0].ops
    assert rewritten.deps == (fused.op_id,)
    outputs = dict(fused.metadata)["typed_outputs"]
    by_port = {(port["port"], port["version"]): port["value"] for port in outputs}
    assert len(by_port) == len(outputs) == len(keys)
    inputs = dict(rewritten.metadata)["typed_inputs"]
    assert all(port["producer"] == fused.op_id for port in inputs)
    assert tuple(by_port[port["port"], port["version"]] for port in inputs) == keys
    positional = full_operand_ports(rewritten)
    assert [producer for producer, _ in positional] == [fused.op_id, fused.op_id]
    names = [port.removeprefix(f"{producer}:") for producer, port in positional]
    assert tuple(by_port["output" if name == "out" else name, 0] for name in names) == keys[:2]


def test_deduplication_requires_equal_constant_values(dense):
    rank, = dense.atomic_ops
    prototype, consumer, *_ = rank.ops
    constants = tuple(replace(prototype, op_id=f"constant-{i}", deps=(),
                              metadata=(("pure", True), ("constant_value", i))) for i in (1, 2))
    ids = tuple(op.op_id for op in constants)
    scenario = replace(dense, atomic_ops=(replace(rank, ops=(
        *constants, replace(consumer, deps=ids),
    )),))
    edit = Edit("dedup", operation=HoistOrDeduplicate(tuple(map(_atom, ids)), "intra_step"))
    with pytest.raises(ModelingError, match="rewrite_cse_signature_mismatch"):
        apply_plan(_plan(scenario, edit), scenario)
    original = apply_plan(_plan(scenario), scenario).executable[0].ops
    assert original[:2] == constants and original[-1].deps == ids


def test_pipeline_rejects_aic_buffer_in_aiv_ub(dense):
    target = next(op for op in dense.atomic_ops[0].ops if op.family == "qkv_projection")
    pipeline = _pipeline(target.family)
    tile, = pipeline.candidates
    with pytest.raises(ModelingError, match=r"live_buffer_pool_owner_mismatch:cohort:ub:aic"):
        replace(tile.buffers[0], bytes_per_owner=4 * 1024**3, owner_kind="aic")
    illegal = replace(tile.buffers[0], bytes_per_owner=4 * 1024**3)
    object.__setattr__(illegal, "owner_kind", "aic")
    pipeline = replace(pipeline, candidates=(replace(tile, buffers=(illegal,)),))
    edit = Edit("oversized", operation=PipelineTiles(_atom(target.op_id), pipeline))
    with pytest.raises(ModelingError, match=r"MULTI_POOL_CAPACITY:.*:ub:owner"):
        apply_plan(_plan(dense, edit), dense)


def test_pipeline_rejects_overlapping_hbm_buffers_and_keeps_feasible_fallback(dense):
    die, = dense.case.machine.dies
    target = _atom(dense.atomic_ops[0].ops[0].op_id)
    spec = _pipeline("scratch")
    candidate, = spec.candidates
    hbm = LiveBufferSpec("first", "hbm", die.memory_bytes // 2 + 32, 0, 1, owner_kind="die")
    overlapping = replace(candidate, buffers=(hbm, replace(hbm, name="second")))
    separated = replace(overlapping, buffers=(hbm, replace(hbm, name="second", live_start=1, live_end=2)))

    def apply(*candidates):
        tile = PipelineTiles(target, replace(spec, candidates=candidates))
        return apply_plan(_plan(dense, Edit("tile", operation=tile)), dense)

    assert apply(separated).tile_bindings
    with pytest.raises(ModelingError, match="MULTI_POOL_CAPACITY:.*:hbm:bytes"):
        apply(overlapping)

    buffer, = candidate.buffers
    oversized = replace(candidate, buffers=(replace(buffer, bytes_per_owner=dict(die.buffers)["ub"] + 1),))
    fallback = replace(candidate, algorithm="online_reduce")
    assert buffer.bytes_per_owner == 1024
    with pytest.raises(ModelingError, match="MULTI_POOL_CAPACITY"):
        apply(oversized)
    planned = apply(oversized, fallback)
    op = next(op for op in planned.executable[0].ops if op.op_id == target.id)
    assert dict(op.metadata)["tile_algorithm"] == fallback.algorithm
    assert verify_plan(dense, planned).gates[1].passed


def test_pipeline_replay_rejects_unregistered_template():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    target = next(op for op in dossier.handle.scenario.atomic_ops[0].ops if op.family == "qkv_projection")
    select = Edit("choose", operation=SelectTemplate(_atom(target.op_id), "K2", "missing_kernel"))
    tile = Edit("pipeline", (select.edit_id,), PipelineTiles(
        OpRef("result_of", select.edit_id), _pipeline(target.family),
    ))
    plan = replace(dossier.unmerge_plan, edits=(select, tile))
    with pytest.raises(ModelingError, match=r"rewrite_template_recipe_unbound:.*:missing_kernel"):
        finalize_plan(dossier.handle, plan)


def test_ring_replay_uses_the_intra_package_link():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=2, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    scenario = dossier.handle.scenario
    collective = next(op for op in scenario.atomic_ops[0].ops if op.family == "allreduce")
    route = next(link.name for link in scenario.case.machine.links if "intra_package" in link.name)
    edit = Edit("ring", operation=CollectiveTransform(_atom(collective.op_id), "ring", route))
    report = finalize_plan(dossier.handle, replace(dossier.unmerge_plan, edits=(edit,)))
    assert report.verification.passed
    row = next(row for row in report.step_eval.operators if row.occurrence.op_id == collective.op_id)
    assert row.implementation == "RingCollective"
    link_work = report.step_eval.execution_evidence["serviced"]
    assert any(name.startswith(f"link:{route}") and size > 0 for name, size in link_work.items())


def test_two_cohort_replay_records_positive_intervals_and_routing_moments(monkeypatch):
    case = compose_case("deepseek-v4-flash", REPRESENTATIVE_SKUS["a5"], tp=4, ep=4, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", sequences=8, context=64)))
    grouped = next(op for op in dossier.handle.scenario.atomic_ops[0].ops if op.family == "grouped_gemm")
    assert grouped.local_shape[0] == 12
    edit = Edit("split", operation=PipelineTiles(_atom(grouped.op_id), None, "dbo_2cohort"))
    moments = {}
    lower = ExpectedGroupedMatmul.lower

    def record(recipe, context):
        kernel = context.service_identity.get("kernel", "")
        if context.op.op_id == grouped.op_id and kernel.startswith("rewrite_cohort_"):
            meta = dict(context.op.metadata)
            moments.setdefault(kernel, set()).add((context.op.local_shape[0], meta["expected_rows"],
                                                   meta["expected_active_experts"], meta["weight_bytes"]))
        return lower(recipe, context)

    monkeypatch.setattr(ExpectedGroupedMatmul, "lower", record)
    baseline = finalize_plan(dossier.handle, dossier.unmerge_plan)
    replayed = finalize_plan(dossier.handle, replace(dossier.unmerge_plan, edits=(edit,)))
    assert not baseline.step_eval.cohort_intervals
    assert replayed.verification.passed and replayed.verification.l_star_ns <= replayed.wall_ns
    intervals = replayed.step_eval.cohort_intervals[grouped.op_id]
    assert set(intervals) == {0, 1}
    case = dossier.handle.scenario.case
    layer = case.model.layer_plans[grouped.layer]
    experts = layer.num_experts // case.runtime.parallel.ep
    per_token = layer.top_k / layer.num_experts
    expected_active = experts * (1 - (1 - per_token)**(dossier.handle.scenario.step.sequences // 2))
    assert expected_active == pytest.approx(5.792, abs=0.001)
    assert len(moments) == len(intervals)
    original = dict(grouped.metadata)
    for samples in moments.values():
        (rows, expected_rows, active, weights), = samples
        assert rows == expected_rows == grouped.local_shape[0] // 2
        assert active == pytest.approx(expected_active)
        assert weights == pytest.approx(original["weight_bytes"] * active / original["expected_active_experts"])
    row = next(row for row in replayed.step_eval.operators if row.occurrence.op_id == grouped.op_id)
    for cohort in intervals.values():
        assert cohort and all(row.start_ms <= start < end <= row.end_ms for start, end in cohort)
    assert replayed.verification.overlap_pct == compute_overlap_pct(intervals[0], intervals[1]) > 5


def test_host_split_elision_replays_with_device_count_rendezvous():
    case = compose_case("qwen3-30b-a3b", REPRESENTATIVE_SKUS["a3"], tp=2, ep=2, dp=1, layers=1,
                        feature_track="merged", _stack_phase="prefill")
    request = ScenarioRequest(case, StepRequest("prefill", tokens=12, context=64),
                              replay=ReplayOptions(include_submission=True))
    dossier = prepare_scenario(request)
    router = next(op for op in dossier.handle.scenario.atomic_ops[0].ops if op.family == "router")
    edit = Edit("elide", operation=HoistOrDeduplicate((_atom(router.op_id),), "device_group_list_elide_d2h"))
    baseline = finalize_plan(dossier.handle, dossier.unmerge_plan)
    edited = finalize_plan(dossier.handle, replace(dossier.unmerge_plan, edits=(edit,)))
    barrier, = baseline.step_eval.execution_evidence["eager_split_barriers"]
    assert baseline.verification.passed and barrier["total_d2h_bytes"] > 0
    assert edited.verification.passed and edited.applied.elided_d2h_ops == (router.op_id,)
    assert not edited.step_eval.execution_evidence.get("eager_split_barriers")
    rows = {row.occurrence.op_id: row for row in edited.step_eval.operators}
    gather = next(row for key, row in rows.items() if key.endswith(".moe_histogram_gather"))
    dispatch = rows[barrier["op_id"]]
    assert gather.occurrence in dispatch.predecessors and gather.end_ms <= dispatch.start_ms
    assert dispatch.implementation == "TopologyCollectiveRecipe"


def test_asymmetric_hbm_arbiter_and_catlass_base_byte_obligation():
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.optimization.dsl_solver import _catlass_proposals
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.prediction.streams import StreamCoreBudget, StreamLane
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite.schema import StreamSchedule
    from mindie_sim.templates.catlass_catalog import CatlassProblem

    def spine_for(*ops):
        class _Spine(PredictionSpine):
            def lower(self, case, request, **kwargs):
                prepared = super().lower(case, request, **kwargs)
                rank, = prepared.graph.executable
                return replace(prepared, graph=replace(prepared.graph, executable=(replace(rank, ops=ops),), streams=()))
        return _Spine()

    base = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], dp=1, pp=1, layers=1, tp=1))
    die, = base.machine.dies
    asym = replace(die, hbm_read_bytes_per_s=512 * 1000, hbm_write_bytes_per_s=1024 * 1000, hbm_rw_alpha=0.0,
                   rates=tuple(dict(die.rates, hbm=1024 * 1000).items()))
    case_asym = replace(base, machine=replace(base.machine, dies=(asym,)))
    mk = lambda oid, md: gemm_op(op_id=oid, deps=(), requests=(), stage=0, layer=0, m=1, n=64, k=64,
                                 quant=QuantSpec("bf16"), category="dense", phase="decode", metadata=md)
    g0 = mk("g0", (("weight_allocation_id", "W0"), ("weight_initial_residency", "hbm_cold"), ("weight_ranges", ((8192, 16384),))))
    g1 = mk("g1", (("weight_allocation_id", "W1"), ("weight_initial_residency", "hbm_cold"), ("weight_bytes", 8192)))
    dossier = prepare_scenario(ScenarioRequest(case_asym, StepRequest("decode")), spine=spine_for(g0, g1))
    dp = dossier.handle.scenario.datapath
    ha, hv = dp.aic_count // 2, dp.aiv_count // 2
    lanes = (StreamLane("main", StreamCoreBudget(ha, hv, "disjoint")),
             StreamLane("aux", StreamCoreBudget(dp.aic_count - ha, dp.aiv_count - hv, "disjoint")))
    for target in (g0, g1):
        (cat_edit,), _ = next(_catlass_proposals((target,), dp, ha, hv))
        plan = replace(dossier.unmerge_plan, edits=(
            Edit("s", operation=StreamSchedule(lanes, ((_atom("g0"), "main"), (_atom("g1"), "aux")), ())),
            Edit("c", operation=cat_edit)))
        full = finalize_plan(dossier.handle, plan)
        assert full.verification.passed and full.step_eval.span_ms >= 32.0 and full.verification.l_star_ns <= full.wall_ns

    prob = CatlassProblem(1, 64, 64, "bf16", "bf16", "bf16", "fp32", b_id="W", b_base_byte=8192)
    g_off = mk("g0", (("weight_allocation_id", "W"), ("weight_initial_residency", "hbm_cold"),
                      ("weight_payload_bytes", 8192), ("catlass_problem", prob)))
    d_off = prepare_scenario(ScenarioRequest(base, StepRequest("decode")), spine=spine_for(g_off))
    dp_off = d_off.handle.scenario.datapath
    (cat_off,), _ = next(_catlass_proposals((g_off,), dp_off, dp_off.aic_count, dp_off.aiv_count))
    full_off = finalize_plan(d_off.handle, replace(d_off.unmerge_plan, edits=(Edit("c", operation=cat_off),)))
    assert full_off.verification.passed and full_off.verification.l_star_ns <= full_off.wall_ns


def test_resident_fusion_elides_untiled_multi_mib_intermediate_when_fp32_tile_fits_ub():
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.schema.operators import OpSpec

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1)
    # 201 * 7168 * 2 B = 2.75 MiB untiled tensor (>> 192 KiB UB), while 1-row FP32 tile fits UB.
    rows, hidden = 201, 7168
    shape = (rows, hidden)
    tensor_bytes = float(rows * hidden * 2)
    norm_op = OpSpec(
        op_id="s0.l0.rmsnorm", family="rmsnorm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", vector_ops=float(rows * hidden * 4),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes, hbm_bytes=2.0 * tensor_bytes,
    )
    quant_op = OpSpec(
        op_id="s0.l0.quant", family="quantize", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=("s0.l0.rmsnorm",), dtype="fp8",
        vector_ops=float(rows * hidden * 3),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes * 0.5, hbm_bytes=1.5 * tensor_bytes,
    )

    class _ChainSpine(PredictionSpine):
        def lower(self, parsed, request, **kwargs):
            prepared = super().lower(parsed, request, **kwargs)
            rank, = prepared.graph.executable
            return replace(prepared, graph=replace(prepared.graph, executable=(replace(rank, ops=(norm_op, quant_op)),), streams=()))

    scenario = resolve_scenario(case, StepRequest("decode", tokens=rows, context=64), spine=_ChainSpine())
    assert tensor_bytes > dict(scenario.datapath.buffers)["ub"]
    applied = apply_plan(_plan(scenario, Edit("fuse_nq", operation=_fuse("s0.l0.rmsnorm", "s0.l0.quant"))), scenario)
    fused, = applied.executable[0].ops
    meta = dict(fused.metadata)
    assert meta["fusion_route"] == "resident"
    assert meta["gm_round_trip_eliminated_bytes"] == pytest.approx(2.0 * tensor_bytes)
    assert fused.hbm_bytes == pytest.approx(1.5 * tensor_bytes)
    assert fused.mte2_bytes == pytest.approx(tensor_bytes)
    assert fused.mte3_bytes == pytest.approx(0.5 * tensor_bytes)


def test_resident_fusion_spills_when_fp32_upcast_row_exceeds_ub_after_simt_carveout():
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.schema.operators import OpSpec

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1)
    # Row width 32768 * 4 B (FP32 upcast) * 2 buffers > 192 KiB - 16 KiB SIMT carveout -> must spill.
    rows, hidden = 4, 32768
    shape = (rows, hidden)
    tensor_bytes = float(rows * hidden * 2)
    norm_op = OpSpec(
        op_id="s0.l0.big_norm", family="rmsnorm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", vector_ops=float(rows * hidden * 4),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes, hbm_bytes=2.0 * tensor_bytes,
    )
    quant_op = OpSpec(
        op_id="s0.l0.big_quant", family="quantize", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=("s0.l0.big_norm",), dtype="fp8",
        vector_ops=float(rows * hidden * 3),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes * 0.5, hbm_bytes=1.5 * tensor_bytes,
    )

    class _BigRowSpine(PredictionSpine):
        def lower(self, parsed, request, **kwargs):
            prepared = super().lower(parsed, request, **kwargs)
            rank, = prepared.graph.executable
            return replace(prepared, graph=replace(prepared.graph, executable=(replace(rank, ops=(norm_op, quant_op)),), streams=()))

    scenario = resolve_scenario(case, StepRequest("decode", tokens=rows, context=64), spine=_BigRowSpine())
    applied = apply_plan(_plan(scenario, Edit("fuse_big", operation=_fuse("s0.l0.big_norm", "s0.l0.big_quant"))), scenario)
    fused, = applied.executable[0].ops
    meta = dict(fused.metadata)
    assert meta["fusion_route"] == "gm_spill"
    assert meta["gm_round_trip_eliminated_bytes"] == 0.0
    assert fused.hbm_bytes == pytest.approx(3.5 * tensor_bytes)


def test_collective_transform_accepts_oneshot_mode_and_rejects_unknown_mode(routed):
    dispatch = next(op for op in routed.atomic_ops[0].ops if op.family == "alltoall")
    route = routed.case.machine.links[0].name
    edit = Edit("oneshot_route", operation=CollectiveTransform(_atom(dispatch.op_id), "oneshot", route))
    plan = _plan(routed, edit)
    assert decode_rewrite_plan(encode_rewrite_plan(plan)) == plan
    applied = apply_plan(plan, routed)
    record, = applied.collective_rewrites
    assert (record.target_op, record.mode, record.topology_route) == (dispatch.op_id, "oneshot", route)
    with pytest.raises(ModelingError, match="rewrite_collective_mode_invalid:invented"):
        CollectiveTransform(_atom(dispatch.op_id), "invented", route)  # type: ignore[arg-type]


def test_sibling_gemm_activation_dedup_exact_bytes_and_port_contract():
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.rewrite import discover_algebraic_rewrites_for_ops
    from mindie_sim.schema.operators import OpSpec

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1)
    m_dim, n_dim, k_dim = 1, 4096, 4096
    act_bytes = float(m_dim * k_dim * 2)  # 8,192 B
    wt_bytes = float(n_dim * k_dim * 2)   # 33,554,432 B
    out_bytes = float(m_dim * n_dim * 2)  # 8,192 B
    mte2 = act_bytes + wt_bytes           # 33,562,624 B
    hbm = mte2 + out_bytes                # 33,570,816 B
    flops = float(2 * m_dim * n_dim * k_dim)

    norm_op = OpSpec(
        op_id="s0.l0.norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(m_dim, k_dim), local_shape=(m_dim, k_dim),
        padded_shape=(m_dim, k_dim), launch_shape=(m_dim, k_dim),
        dtype="bf16", vector_ops=float(m_dim * k_dim * 4),
        mte2_bytes=act_bytes, mte3_bytes=act_bytes, hbm_bytes=2.0 * act_bytes,
    )
    shared_meta = (
        ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "bf16"),
        ("weight_format", "bf16"),
        ("output_format", "bf16"),
        ("compute_format", "bf16"),
        ("packed_weight_bytes", int(wt_bytes)),
    )
    g1 = OpSpec(
        op_id="s0.l0.proj_a", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=(m_dim, n_dim, k_dim), local_shape=(m_dim, n_dim, k_dim),
        padded_shape=(m_dim, n_dim, k_dim), launch_shape=(m_dim, n_dim, k_dim),
        dtype="bf16", flops=flops, mte2_bytes=mte2, mte3_bytes=out_bytes, hbm_bytes=hbm,
        metadata=shared_meta,
    )
    g2 = replace(g1, op_id="s0.l0.proj_b")

    class _SiblingSpine(PredictionSpine):
        def __init__(self, ops):
            super().__init__()
            self._ops = ops

        def lower(self, parsed, request, **kwargs):
            prepared = super().lower(parsed, request, **kwargs)
            rank, = prepared.graph.executable
            return replace(prepared, graph=replace(prepared.graph, executable=(replace(rank, ops=self._ops),), streams=()))

    scenario = resolve_scenario(case, StepRequest("decode", tokens=1, context=64), spine=_SiblingSpine((norm_op, g1, g2)))
    alg = discover_algebraic_rewrites_for_ops((norm_op, g1, g2), datapath=scenario.datapath)
    law2 = [c for c in alg.candidates if c.law == "law2_block_matmul"]
    assert len(law2) == 1
    assert law2[0].avoidable_hbm_bytes == int(act_bytes)

    applied = apply_plan(_plan(scenario, Edit("fuse_sib", operation=_fuse("s0.l0.proj_a", "s0.l0.proj_b"))), scenario)
    fused = next(op for op in applied.executable[0].ops if op.op_id == "rewrite:fuse_sib")
    meta = dict(fused.metadata)
    assert meta["gm_round_trip_eliminated_bytes"] == pytest.approx(act_bytes)
    assert fused.mte2_bytes == pytest.approx(2.0 * mte2 - act_bytes)
    assert fused.mte3_bytes == pytest.approx(2.0 * out_bytes)
    assert fused.hbm_bytes == pytest.approx(2.0 * hbm - act_bytes)
    assert fused.local_shape == (m_dim, 2 * n_dim, k_dim)
    assert meta["weight_format"] == "bf16"
    assert meta["activation_format"] == "bf16"
    assert meta["packed_weight_bytes"] == int(2 * wt_bytes)

    # Disjoint output ports on a multi-output split producer MUST credit 0 B activation dedup.
    split_op = OpSpec(
        op_id="s0.l0.split", family="split", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(m_dim, 2 * k_dim), local_shape=(m_dim, 2 * k_dim),
        padded_shape=(m_dim, 2 * k_dim), launch_shape=(m_dim, 2 * k_dim),
        dtype="bf16", returns=("q", "k"),
    )
    g_q = replace(g1, op_id="s0.l0.proj_q", deps=("s0.l0.split",),
                  metadata=(("typed_inputs", ({"producer": "s0.l0.split", "port": "q", "version": 0},)),))
    g_k = replace(g1, op_id="s0.l0.proj_k", deps=("s0.l0.split",),
                  metadata=(("typed_inputs", ({"producer": "s0.l0.split", "port": "k", "version": 0},)),))
    scen_split = resolve_scenario(case, StepRequest("decode", tokens=1, context=64), spine=_SiblingSpine((split_op, g_q, g_k)))
    alg_split = discover_algebraic_rewrites_for_ops((split_op, g_q, g_k), datapath=scen_split.datapath)
    assert not [c for c in alg_split.candidates if c.law == "law2_block_matmul"]
    applied_split = apply_plan(_plan(scen_split, Edit("fuse_split", operation=_fuse("s0.l0.proj_q", "s0.l0.proj_k"))), scen_split)
    fused_split = next(op for op in applied_split.executable[0].ops if op.op_id == "rewrite:fuse_split")
    assert dict(fused_split.metadata)["gm_round_trip_eliminated_bytes"] == 0.0
    assert fused_split.mte2_bytes == pytest.approx(2.0 * mte2)
    assert fused_split.hbm_bytes == pytest.approx(2.0 * hbm)

    # Ambiguous multi-output producer without explicit typed port MUST also credit 0 B.
    g_amb1 = replace(g1, op_id="s0.l0.amb1", deps=("s0.l0.split",), metadata=())
    g_amb2 = replace(g1, op_id="s0.l0.amb2", deps=("s0.l0.split",), metadata=())
    scen_amb = resolve_scenario(case, StepRequest("decode", tokens=1, context=64), spine=_SiblingSpine((split_op, g_amb1, g_amb2)))
    applied_amb = apply_plan(_plan(scen_amb, Edit("fuse_amb", operation=_fuse("s0.l0.amb1", "s0.l0.amb2"))), scen_amb)
    fused_amb = next(op for op in applied_amb.executable[0].ops if op.op_id == "rewrite:fuse_amb")
    assert dict(fused_amb.metadata)["gm_round_trip_eliminated_bytes"] == 0.0

    # Mismatched activation group size MUST credit 0 B.
    g_gs1 = replace(g1, op_id="s0.l0.gs1", metadata=(
        ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "fp8_e4m3"), ("activation_group_size", 32),
    ))
    g_gs2 = replace(g1, op_id="s0.l0.gs2", metadata=(
        ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "fp8_e4m3"), ("activation_group_size", 128),
    ))
    scen_gs = resolve_scenario(case, StepRequest("decode", tokens=1, context=64), spine=_SiblingSpine((norm_op, g_gs1, g_gs2)))
    applied_gs = apply_plan(_plan(scen_gs, Edit("fuse_gs", operation=_fuse("s0.l0.gs1", "s0.l0.gs2"))), scen_gs)
    fused_gs = next(op for op in applied_gs.executable[0].ops if op.op_id == "rewrite:fuse_gs")
    assert dict(fused_gs.metadata)["gm_round_trip_eliminated_bytes"] == 0.0
