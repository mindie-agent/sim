"""NR-S2-FIX5: executed Round-5 retained-identity, lease and physics witnesses.

[Role] Reproduce every Round-5 counterexample against the typed projection,
  real ``Engine`` allocations, executed phase work and exported effective DAG.
[Invariants] A retained live-out keeps its own named port and downstream
  identity; a lease release frees exactly its own grant; an L1 grant is held
  until both copy and consumer retire; coupled tiers and per-unit residuals are
  charged once; destination storage is acquired before its first write; typed
  phase work equals reported work; state versions are monotone per target;
  rounding and capture vocabularies are shared; a merge is validated even when
  unused and its traffic sits at the causal boundary; Cube discovery is indexed
  and budget-bounded.
[Timeouts & Locks] Pure in-process virtual simulation; the 2,000-operator
  discovery witness is bounded by the caller's wall budget; no I/O or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
import time

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction import engine_plan
from mindie_sim.prediction.canonical_optimizer import (
    _canonical_stream_regions, _run_canonical,
)
from mindie_sim.prediction.capture_barriers import standalone_capture_blockers
from mindie_sim.prediction.engine_plan import (
    PlanNode, apply_fusion, baseline_plan, evaluate_plan,
)
from mindie_sim.prediction.fusion_bridge import (
    add_prefetch, apply_rewrite, canonical_plan, rewrite_fusion,
)
from mindie_sim.prediction.fusion_contract import (
    ReductionMerge, _contract,
)
from mindie_sim.prediction.fusion_feasibility import (
    analyze_region, region_legality,
)
from mindie_sim.prediction.fusion_graph import (
    FeasibilityError, TypedGraph,
)
from mindie_sim.prediction.fusion_pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_cube_prologue, synthesize_vector,
)
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.graph_optimizer_engine import (
    _candidate_specs, _cube_epilogues, _index,
)
from mindie_sim.prediction.optimizer_report import Budget, _canonical_dag
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.weight_prefetch import PrefetchPhysics, plan_prefetch
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import (
    canonical_qwen, cast_chain_graph, hw, op, quant_graph, scalar_cube_graph,
    val,
)

_CASE = None


def case():
    global _CASE
    if _CASE is None:
        _CASE = compose_case("Qwen3.8-27B", hardware="a3", tp=1, layers=1)
    return _CASE


def _op(op_id: str, deps=(), *, family: str = "add", hbm: float = 1_000_000.0,
        vector: float = 2.0, shape=(1, 4), metadata=(), stream: str = "") -> OpSpec:
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, hbm_bytes=hbm,
        vector_ops=vector, dtype="bf16", deps=tuple(deps),
        metadata=tuple(metadata), stream=stream)


def _observed(monkeypatch):
    class ObservedEngine(Engine):
        last = None

        def __init__(self, *args, **kwargs):
            kwargs["trace_limit"] = 4096
            super().__init__(*args, **kwargs)
            ObservedEngine.last = self

    monkeypatch.setattr(engine_plan, "Engine", ObservedEngine)
    return ObservedEngine


def _vector_rewrite(plan):
    specs, _ = _candidate_specs("vector_fusion", plan.graph, plan.hardware,
                                Budget(per_pass=32, total=64, wall_s=30))
    candidate = next(item for item in specs
                     if not isinstance(item, RejectedFusion))
    return apply_rewrite(plan, rewrite_fusion(plan, candidate)), candidate


def _cube_budget_graph(copies: int = 1000) -> TypedGraph:
    base = scalar_cube_graph()
    values, ops, public = {}, {}, []
    for index in range(copies):
        suffix = f"_{index}"
        local = {}
        for key, value in base.values.items():
            value_id = key + suffix
            local[key] = value_id
            values[value_id] = replace(
                value, value_id=value_id,
                producer=(value.producer + suffix) if value.producer else None)
        for key, node in base.ops.items():
            op_id = key + suffix
            ops[op_id] = replace(
                node, op_id=op_id,
                inputs=tuple(local[item] for item in node.inputs),
                outputs=tuple(local[item] for item in node.outputs))
        public.extend(local[item] for item in base.public_outputs)
    return TypedGraph(base.rows, values, ops, tuple(public))


# ---------------------------------------------------------------------------
# Witness 1: distinct retained output ports survive contraction and projection
# ---------------------------------------------------------------------------
def test_w1_distinct_retained_ports_and_qwen_residual_identity():
    plan = canonical_plan(case(), (
        _op("a", ("analysis_input",)),
        _op("b", ("a",), family="rotary",
            metadata=(("rotate_style", "interleaved"),)),
        _op("c", ("a", "b"))))
    fused_plan, candidate = _vector_rewrite(plan)
    fused = next(item for item in fused_plan.ops
                 if item.op_id.startswith("fused:"))
    ports = dict(fused.metadata)["retained_port_map"]
    assert ports == (
        ("a:out", "fused:a+b:a_out"),
        ("b:out", "fused:a+b:out"))
    assert fused_plan.graph.ops["c"].inputs == ("fused:a+b:a_out",
                                                "fused:a+b:out")
    assert set(fused_plan.graph.ops["c"].inputs) == {
        fused_plan.graph.values["fused:a+b:a_out"].value_id,
        fused_plan.graph.values["fused:a+b:out"].value_id}
    assert linked_outputs(fused_plan.graph, "c") == (
        "fused:a+b:a_out", "fused:a+b:out")
    assert fused_plan.graph.ops["c"].outputs
    assert candidate.analysis.traffic.saved_bytes > 0

    raw, ops = canonical_qwen()
    report = run_optimizer(raw, ops, evaluation_budget=64,
                           max_candidates_per_pass=32)
    final = report.canonical_plan
    fused_id = next(op.op_id for op in final.ops
                    if op.family == "native_fusion")
    residual_port = f"{fused_id}:s0.l0.attention.residual_out"
    gate_port = f"{fused_id}:out"
    assert final.graph.ops["s0.l0.ffn.residual"].inputs == (
        "s0.l0.tp:out", residual_port)
    assert final.graph.ops["s0.l0.gate_up"].inputs == (
        gate_port, "s0.l0.gate_up:weight")
    assert residual_port != gate_port
    assert final.graph.values[residual_port].shape == (16, 5120)
    assert final.graph.values[gate_port].shape == (16, 5120)
    assert report.final.hbm_bytes == 209_661_952.0


def linked_outputs(graph: TypedGraph, op_id: str) -> tuple[str, ...]:
    """Inputs of ``op_id`` that are produced by one contracted operator."""
    fused_id = next(item for item in graph.ops[op_id].deps
                    if item.startswith("fused:"))
    return tuple(value_id for value_id in graph.ops[op_id].inputs
                 if graph.values[value_id].producer == fused_id)


# ---------------------------------------------------------------------------
# Witness 2: same-scope invocations release their exact allocation handle
# ---------------------------------------------------------------------------
def test_w2_same_scope_release_frees_its_own_invocation():
    engine = Engine([Resource("hbm", 1024, "bytes")], pools={"l1": 2048},
                    pool_units={"l1": 1})
    entry = engine.submit("entry")
    holder = PoolHolder("shared")
    prefetch = PrefetchPoolProgram(holder, 1024, 1, caps={"hbm": 1024},
                                   priority=10)
    first_copy = prefetch.start(engine, "first", entry)
    second_copy = prefetch.start(engine, "second", entry)
    first_consumer = engine.submit("first:consumer", {"hbm": 1024 * 10},
                                   after=(first_copy,))
    second_consumer = engine.submit("second:consumer", {"hbm": 1024},
                                    after=(second_copy,))
    second_join = engine.submit("second:join",
                                after=(second_consumer, second_copy))
    first_join = engine.submit("first:join",
                               after=(first_consumer, first_copy))
    release = ReleasePoolProgram(holder)
    second_release = release.start(engine, "second:release", second_join)
    first_release = release.start(engine, "first:release", first_join)
    engine.advance(until=(first_release, second_release), first=False)
    trace = list(engine.allocation_trace)
    second_free = next(row for row in trace
                       if row["task"] == "second:release:free")
    assert second_free["name"] == "second:pool"
    assert second_free["time"] < first_release.time
    assert first_consumer.time is not None
    first_free = next(row for row in trace if row["task"] == "first:release:free")
    assert first_free["name"] == "first:pool"
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


# ---------------------------------------------------------------------------
# Witness 3: hold the grant for both copy and consumer; export real joins
# ---------------------------------------------------------------------------
def _busy_prefetch_plan():
    pre = replace(_op("pre", ("analysis_input",), family="cast", hbm=8192,
                      vector=0.0, shape=(1, 2048)),
                  metadata=(("resource_work", {"host": 1.0}),))
    side = 2048
    matmul = OpSpec(
        op_id="mm", family="gemm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, side, side),
        local_shape=(1, side, side), padded_shape=(1, side, side),
        launch_shape=(1, side, side), flops=2.0 * side * side,
        hbm_bytes=1e6, dtype="bf16", deps=("pre",))
    busy = replace(_op("busy", ("analysis_input",), hbm=10_000_000_000.0,
                       vector=0.0), stream="aux")
    plan = canonical_plan(case(), (pre, matmul, busy))
    decision = plan_prefetch(
        plan.graph, "mm", window_ms=50.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes,
        tensor_parallel=plan.hardware.tensor_parallel)
    assert decision.allowed and decision.target_bytes == 8 * 1024 * 1024
    updated, prefetch_id = add_prefetch(plan, decision, plan.hardware)
    return updated, prefetch_id, decision


def test_w3_prefetch_grant_waits_for_copy_and_consumer_and_export_is_real():
    plan, prefetch_id, _decision = _busy_prefetch_plan()
    result = plan.evaluate(audit_allocations=True)
    rows = {row["op_id"]: row for row in result["timeline"]}
    assert rows["mm"]["complete_ms"] == pytest.approx(0.00501024)
    assert rows[prefetch_id]["complete_ms"] == pytest.approx(
        6.2552480000000035)
    release = rows["release_W_mm"]
    assert release["ready_ms"] >= rows[prefetch_id]["complete_ms"] - 1e-12
    assert release["ready_ms"] >= rows["mm"]["complete_ms"] - 1e-12
    release_op = next(item for item in plan.ops
                      if item.op_id == "release_W_mm")
    assert release_op.deps == ("mm", prefetch_id)

    trade = [row for row in result["allocation_trace"]
             if str(row["pool"]).startswith("l1")]
    # The matrix program also grants/releases per-tile L1 panels; isolate the
    # prefetch grant by its owner name.
    acquire = next(row for row in trade
                   if row["action"] == "acquire" and "prefetch_W_mm" in row["name"])
    freed = next(row for row in trade
                 if row["action"] == "release" and "prefetch_W_mm" in row["name"])
    assert acquire["size"] * len(acquire["units"]) == 8 * 1024 * 1024
    assert freed["time"] == pytest.approx(rows[prefetch_id]["complete_ms"])
    assert acquire["time"] < freed["time"]
    edges = _canonical_dag(plan, result)
    assert not any(edge["from"] == prefetch_id and edge["to"] == "mm"
                   for edge in edges)
    assert any(edge["from"] == prefetch_id and edge["to"] == "release_W_mm"
               for edge in edges)
    mm_preds = rows["mm"]["predecessors"]
    assert not any("prefetch" in str(pred) for pred in mm_preds)


# ---------------------------------------------------------------------------
# Witness 4: coupled lockstep per tier and exactly one unit debit
# ---------------------------------------------------------------------------
def test_w4_coupled_fifo_priority_and_three_tier_units():
    engine = Engine([Resource("a", 100, "work", policy="fifo"),
                     Resource("b", 100, "work", policy="fair")])
    early = engine.submit("early", {"a": 100, "b": 100}, coupled=True,
                          priority=10)
    late = engine.submit("late", {"a": 10, "b": 10}, coupled=True,
                         priority=0, release=0.1)
    engine.advance(until=(early, late), first=False)
    assert early.time == pytest.approx(1.0)
    assert late.time == pytest.approx(1.1)

    units = Engine([Resource("hbm", 1024, "bytes", units=2)])
    tier0 = units.submit("tier0", {"hbm": 10_000}, priority=0,
                         placement={"hbm": (0,)}, caps={"hbm": 256})
    tier1 = units.submit("tier1", {"hbm": 10_000}, priority=1,
                         placement={"hbm": (1,)}, caps={"hbm": 256})
    tier2 = units.submit("tier2", {"hbm": 10_000}, priority=2,
                         placement={"hbm": (1,)})
    units.advance(horizon=1.0, first=False)
    assert units.service_of(tier0)["hbm"] == pytest.approx(256.0)
    assert units.service_of(tier1)["hbm"] == pytest.approx(256.0)
    assert units.service_of(tier2)["hbm"] == pytest.approx(256.0)


# ---------------------------------------------------------------------------
# Witness 5: overlapping UB reservations are summed per (pool, unit)
# ---------------------------------------------------------------------------
def test_w5_per_unit_ub_admission_keeps_owner_dimension():
    hardware = hw(ub_bytes_per_aiv=65536, usable_ub_bytes=4000, ub_units=2,
                  prefetch_capacity_bytes=0)
    per_unit = (PlanNode("two", (("vector", 1.0),),
                         ub_allocations=(("a", "ub", 3000, (0,)),
                                         ("b", "ub", 3000, (1,)))),)
    measured = evaluate_plan(per_unit, hardware)
    assert measured.status == "completed", measured.reason
    same_unit = (PlanNode("one", (("vector", 1.0),),
                          ub_allocations=(("a", "ub", 3000, (0,)),
                                          ("b", "ub", 3000, (0,)))),)
    failed = evaluate_plan(same_unit, hardware)
    assert failed.status == "failed"
    assert "single_op_ub_capacity_exceeded:one:ub:6000>4000" in failed.reason


# ---------------------------------------------------------------------------
# Witness 6: destination L1 is acquired before the handoff writes into it
# ---------------------------------------------------------------------------
def test_w6_destination_l1_acquired_before_prologue_handoff(monkeypatch):
    values = {
        "x": val("x", "bf16", (1, 1), storage=2),
        "w": val("w", "bf16", (1, 1), role="weight", storage=2),
        "h": val("h", "bf16", (1, 1), producer="pre", storage=2),
        "p": val("p", "bf16", (1, 1), producer="mm", storage=2),
    }
    graph = TypedGraph(1, values, {
        "pre": op("pre", "add", "vector", ("x",), ("h",),
                  work=(("vector", 1.0),)),
        "mm": op("mm", "matmul", "cube", ("h", "w"), ("p",),
                 work=(("cube", 2.0),)),
    }, ("p",))
    hardware = hw(prefetch_capacity_bytes=4, onchip_bytes_per_ms=1.0,
                  vector_ops_per_ms=1.0, cube_flops_per_ms=1.0,
                  launch_ms=0.0, device_min_ms=0.0)
    candidate = synthesize_cube_prologue(
        graph, "pre", "mm", tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hardware.cube_capacities)
    assert not isinstance(candidate, RejectedFusion)
    plan = apply_fusion(baseline_plan(graph, hardware), candidate)
    fused = next(node for node in plan if node.node_id.startswith("fused:"))
    groups = dict(fused.phase_acquisitions)
    assert "prologue_handoff" in groups
    assert any(pool == "l1" for _name, pool, _size, _units in groups[
        "prologue_handoff"])
    holder = PlanNode("hold", (("host", 10.0),), stream="aux",
                      ub_allocations=(("hold", "l1", 4, (0,)),))
    observed = _observed(monkeypatch)
    measured = evaluate_plan((holder,) + plan, hardware)
    assert measured.status == "completed", measured.reason
    rows = [row for row in observed.last.trace
            if row["task"].startswith("body:fused:pre+mm")]
    first = {row["task"].rsplit(":", 1)[-1]: row for row in rows}
    assert first["prologue_handoff"]["start_ms"] >= 10.0 - 1e-12
    assert first["prologue"]["end_ms"] <= first["prologue_handoff"]["start_ms"] + 1e-12
    assert first["prologue_handoff"]["end_ms"] <= first["cube"]["start_ms"] + 1e-12


# ---------------------------------------------------------------------------
# Witness 7: typed phase work executes every byte the report claims
# ---------------------------------------------------------------------------
def test_w7_typed_phase_work_matches_engine_service(monkeypatch):
    values = {
        "x": val("x", "bf16", (1, 4), storage=8),
        "h": val("h", "bf16", (1, 4), producer="a", storage=8),
        "y": val("y", "bf16", (1, 4), producer="b", storage=8),
    }
    graph = TypedGraph(1, values, {
        "a": op("a", "add", "vector", ("x",), ("h",),
                work=(("vector", 2.0), ("hbm", 101_000_000.0))),
        "b": op("b", "add", "vector", ("h",), ("y",),
                work=(("vector", 2.0), ("hbm", 1_000_000.0))),
    }, ("y",))
    specs, _ = _candidate_specs("vector_fusion", graph, hw(),
                                Budget(per_pass=8, total=16, wall_s=10))
    candidate = next(item for item in specs
                     if not isinstance(item, RejectedFusion))
    plan = apply_fusion(baseline_plan(graph, hw()), candidate)
    fused = next(node for node in plan if node.node_id.startswith("fused:"))
    work = fused.work_dict()
    assert work["hbm"] == 101_999_984.0
    phase_total = {}
    for _phase, demand in fused.phase_work:
        for resource, amount in demand.items():
            phase_total[resource] = phase_total.get(resource, 0.0) + amount
    assert phase_total["hbm"] == pytest.approx(work["hbm"])
    assert phase_total["vector"] == pytest.approx(work["vector"])
    observed = _observed(monkeypatch)
    measured = evaluate_plan(plan, hw())
    assert measured.status == "completed", measured.reason
    serviced = observed.last.summary()["serviced"]
    assert serviced["hbm"] == pytest.approx(work["hbm"])
    assert serviced["vector"] == pytest.approx(work["vector"])
    assert measured.hbm_bytes == pytest.approx(101_999_984.0)


# ---------------------------------------------------------------------------
# Witness 8: mixed read/write successor still checks the written target
# ---------------------------------------------------------------------------
def _state_op(op_id, deps, effects):
    return _op(op_id, deps, family="elementwise", hbm=8.0,
               metadata=(("gdn_state_effects", effects),))


def test_w8_state_monotonicity_is_per_target_across_mixed_access():
    writer = ({"target": "a", "action": "write", "version": 7},
              {"target": "b", "action": "write", "version": 1})
    successor = ({"target": "a", "action": "read", "version": 7},
                 {"target": "b", "action": "write", "version": 0})
    plan = canonical_plan(case(), (
        _state_op("w", ("analysis_input",), writer),
        _state_op("rw", ("w",), successor)))
    with pytest.raises(FeasibilityError,
                       match="state_version_not_monotone:b:w:rw"):
        plan.graph.validate()
    valid = canonical_plan(case(), (
        _state_op("w", ("analysis_input",), writer),
        _state_op("rw", ("w",), ({"target": "a", "action": "read",
                                  "version": 7},
                                 {"target": "b", "action": "write",
                                  "version": 2}))))
    valid.graph.validate()


# ---------------------------------------------------------------------------
# Witness 9: shared rounding vocabulary and nondeterminism policy
# ---------------------------------------------------------------------------
def test_w9_kv_cast_and_quant_share_rounding_vocabulary():
    from mindie_sim.prediction.fusion_graph import OpNode
    values, _ops = cast_chain_graph("t")
    values = {key: replace(item, dtype="fp16" if key != "tx" else "bf16")
              for key, item in values.items()}

    def kv(op_id, source, target, rounding):
        return OpNode(op_id, "kv_cast", "vector", (source,), (target,),
                      params=(("target_dtype", "fp16" if op_id == "ta"
                               else "bf16"), ("rounding", rounding)))

    bogus = TypedGraph(1, values, {
        "ta": kv("ta", "tx", "ty", "BOGUS"),
        "tb": kv("tb", "ty", "tz", "BOGUS")}, ("tz",))
    check = _contract(bogus, bogus.ops["ta"])
    assert "kv_cast_rounding_unknown" in check.reasons
    assert check.exact is False
    assert "kv_cast_rounding_unknown" in region_legality(
        bogus, ("ta", "tb")).reasons

    graph = quant_graph()
    stochastic = replace(
        graph.ops["quant"],
        params=tuple(sorted({**dict(graph.ops["quant"].params),
                             "rounding": "stochastic"}.items())))
    quantized = TypedGraph(16, graph.values,
                           {"quant": stochastic, "cast": graph.ops["cast"]},
                           ("y",))
    check = _contract(quantized, quantized.ops["quant"])
    assert check.reasons == ()
    assert check.numerical_policy == "relaxed"
    rejected = synthesize_vector(quantized, ("quant", "cast"), tile_rows=1,
                                 capacity_per_owner=100000)
    assert isinstance(rejected, RejectedFusion)
    assert "relaxed_numerics_not_allowed" in rejected.reasons


# ---------------------------------------------------------------------------
# Witness 10: the standalone capture audit uses the explicit blocker alias
# ---------------------------------------------------------------------------
def test_w10_explicit_capture_blockers_block_standalone_operator():
    plan = canonical_plan(case(), (
        _op("only", ("analysis_input",),
            metadata=(("explicit_capture_blockers",
                       ("runtime_pointer_escape",)),)),))
    capabilities = tuple(sorted({
        **dict(plan.hardware.capabilities), "aclgraph_capture": True}.items()))
    plan = replace(plan, hardware=replace(plan.hardware,
                                          capabilities=capabilities))
    assert standalone_capture_blockers(plan.ops[0], plan.graph.ops["only"]) == (
        "capture_blocker:runtime_pointer_escape",)
    report = _run_canonical(plan, evaluation_budget=8, per_pass=4,
                            max_streams=None, wall_budget_s=10.0)
    assert report.capture["status"] == "blocked"
    assert report.capture["evaluated"] is False
    assert "capture_blocker:runtime_pointer_escape" in report.capture["blockers"]


# ---------------------------------------------------------------------------
# Witness 11: merge validation is unconditional and merge traffic is causal
# ---------------------------------------------------------------------------
def test_w11_merge_validated_at_full_width_and_causal_hbm_boundaries():
    graph = quant_graph()
    invalid = analyze_region(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100000,
        channel_width=64,
        reduction_merge=ReductionMerge(128, float("nan"), 64))
    assert invalid.legality.ok is False
    assert "reduction_merge_work_missing" in invalid.legality.reasons
    assert invalid.traffic.candidate_bytes == 4096
    assert "merge" not in dict(invalid.phase_work)

    unused = analyze_region(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100000,
        channel_width=64,
        reduction_merge=ReductionMerge(128, 1_000_000, 64))
    assert unused.legality.ok is True
    assert unused.traffic.candidate_bytes == 4096
    assert "merge" not in dict(unused.phase_work)

    merged = analyze_region(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100000,
        channel_width=1,
        reduction_merge=ReductionMerge(128, 1_000_000, 64))
    assert merged.legality.ok is True
    phases = dict(merged.phase_work)
    assert phases["operand_read"]["hbm"] == pytest.approx(2048.0)
    assert phases["reduce"]["hbm"] == pytest.approx(128.0)
    assert phases["merge"]["hbm"] == pytest.approx(128.0)
    assert phases["merge"]["mte1"] == pytest.approx(256.0)
    assert phases["output_write"]["hbm"] == pytest.approx(2048.0)
    assert "hbm" not in phases["finalize"]


# ---------------------------------------------------------------------------
# Witness 12: Cube discovery is indexed and every branch is budget-bounded
# ---------------------------------------------------------------------------
def test_w12_two_thousand_operator_cube_discovery_is_indexed_and_bounded():
    graph = _cube_budget_graph(1000)
    assert len(graph.ops) == 2000

    class NoScanGraph(TypedGraph):
        def consumers(self, value_id):  # pragma: no cover - must never run
            raise AssertionError("quadratic consumers scan")

        def successors(self, op_id):  # pragma: no cover - must never run
            raise AssertionError("quadratic successors scan")

    no_scan = NoScanGraph(graph.rows, graph.values, graph.ops,
                          graph.public_outputs)
    budget = Budget(per_pass=32, total=64, wall_s=0.5)
    started = time.perf_counter()
    probes, aborted = _cube_epilogues(no_scan, budget=budget)
    elapsed = time.perf_counter() - started
    assert len(probes) == 2
    assert aborted is False
    assert elapsed < 0.5

    budget = Budget(per_pass=32, total=64, wall_s=0.5)
    adjacency, consumers = _index(graph, budget)
    found, aborted = _cube_epilogues(graph, adjacency=adjacency,
                                     consumers=consumers, budget=budget)
    assert found == probes
    assert aborted is False

    capabilities = tuple(sorted({
        **dict(hw().capabilities), "fixpipe_epilogue": True,
        "cube_prologue": False}.items()))
    budget = Budget(per_pass=32, total=64, wall_s=0.5)
    started = time.perf_counter()
    specs, unevaluated = _candidate_specs(
        "cube_epilogue", graph, hw(capabilities=capabilities), budget)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert all(not isinstance(item, RejectedFusion) for item in specs)
    if budget.exhausted:
        assert unevaluated
        assert unevaluated[0]["reason"] == "discovery_budget_exhausted"
    else:
        assert specs

    plan_ops = tuple(_op(f"n{index}", ("analysis_input",), hbm=1.0)
                     for index in range(2000))
    canonical = canonical_plan(case(), plan_ops)
    stream_budget = Budget(per_pass=32, total=64, wall_s=0.5)
    started = time.perf_counter()
    regions = _canonical_stream_regions(canonical, 2, stream_budget)
    assert time.perf_counter() - started < 0.5
    assert isinstance(regions, tuple)
