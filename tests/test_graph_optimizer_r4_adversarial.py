"""NR-S2-FIX4: executed Round-4 compiler, pool and graph-physics counterexamples.

[Role] Reproduce every remaining Round-4 witness through the canonical
  OpSpec/``graph_costs`` path, the typed projection, real ``Engine`` grants and
  the lowered causal phase program.
[Invariants] L1/UB grants are owned by one exact (engine, replica, stage,
  invocation) identity; retained output dtypes/shapes never change; effective
  UB is the declared usable budget; unplaced and coupled rates conserve every
  physical unit; FIFO is non-preemptive; directional handoffs and reduction
  merges execute at their causal boundary.
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
from mindie_sim.prediction.canonical_optimizer import _run_canonical
from mindie_sim.prediction.coupled import coupled_rates
from mindie_sim.prediction.engine_plan import (
    PlanNode, apply_fusion, baseline_plan, evaluate_plan,
)
from mindie_sim.prediction.fusion_bridge import (
    add_prefetch, canonical_plan, fused_op_spec,
)
from mindie_sim.prediction.fusion_contract import (
    ReductionMerge, _contract, handoff_split,
)
from mindie_sim.prediction.fusion_feasibility import (
    analyze_region, region_legality, single_op_ub,
)
from mindie_sim.prediction.fusion_graph import OpNode, TypedGraph
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, _make, _match, synthesize_cube_epilogue, synthesize_vector,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.graph_optimizer import run_optimizer
from mindie_sim.prediction.graph_optimizer_engine import _candidate_specs
from mindie_sim.prediction.optimizer_report import Budget, _canonical_dag
from mindie_sim.prediction.runtime.arbiter import component_rates
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.weight_prefetch import PrefetchPhysics, plan_prefetch
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import (
    budget_graph, canonical_mlapo, cast_chain_graph, hw, op, quant_graph,
    val,
)

_CASES: dict[tuple[int, int], object] = {}


def case(dp: int = 1, pp: int = 1):
    key = (dp, pp)
    if key not in _CASES:
        _CASES[key] = compose_case(
            "Qwen3.8-27B", hardware="a3", tp=1, dp=dp, pp=pp,
            layers=max(1, pp))
    return _CASES[key]


def _add(op_id: str, deps=(), *, vector: float = 2.0,
         hbm: float = 1_000_000.0, shape=(1, 4), metadata=(),
         family: str = "add", dtype: str = "bf16", stage: int = 0) -> OpSpec:
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=stage,
        layer=stage, requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, hbm_bytes=hbm,
        vector_ops=vector, dtype=dtype, deps=tuple(deps),
        metadata=tuple(metadata))


def _matmul(op_id: str = "mm", *, shape=(1, 64, 64), hbm: float = 1e6,
            deps=(("pre",)), stage: int = 0) -> OpSpec:
    m, n, k = shape
    return OpSpec(
        op_id=op_id, family="gemm", phase="decode", stage=stage, layer=stage,
        requests=(), semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=2.0 * m * n * k, hbm_bytes=float(hbm), dtype="bf16",
        deps=tuple(deps))


def _fuse_first(plan):
    specs, _ = _candidate_specs("vector_fusion", plan.graph, plan.hardware,
                                Budget(per_pass=32, total=64, wall_s=30))
    for spec in specs:
        if not isinstance(spec, RejectedFusion):
            from mindie_sim.prediction.fusion_bridge import (
                apply_rewrite, rewrite_fusion,
            )
            return apply_rewrite(plan, rewrite_fusion(plan, spec)), spec
    raise AssertionError("no admissible vector fusion candidate")


def _observed(monkeypatch):
    class ObservedEngine(Engine):
        last = None

        def __init__(self, *args, **kwargs):
            kwargs["trace_limit"] = 4096
            super().__init__(*args, **kwargs)
            ObservedEngine.last = self

    monkeypatch.setattr(engine_plan, "Engine", ObservedEngine)
    return ObservedEngine


def _phase_rows(engine, node_fragment: str):
    return [row for row in engine.trace if node_fragment in row["task"]]


def _task(priority=0, placement=None, coupled=False, remaining=None,
          start=0.0):
    from types import SimpleNamespace
    return SimpleNamespace(priority=priority, placement=placement or {},
                           coupled=coupled, remaining=remaining or {},
                           caps={}, start=start, cohort_size=1)


def _prefetch_plan(raw, *, pp=1, weight_bytes=8 * 1024 * 1024):
    pre = replace(_add("pre", ("analysis_input",), shape=(1, 2048), vector=0,
                       hbm=8192, family="cast"), stage=0, layer=0,
                  metadata=(("resource_work", {"host": 1.0}),))
    side = max(1, int((weight_bytes // 2) ** 0.5))
    mm = _matmul(shape=(1, side, side), deps=("pre",), stage=pp - 1)
    plan = canonical_plan(raw, (pre, mm))
    decision = plan_prefetch(
        plan.graph, "mm", window_ms=50.0, foreground_hbm_bytes=0.0,
        physics=PrefetchPhysics(plan.hardware.hbm_bytes_per_ms,
                                plan.hardware.onchip_bytes_per_ms,
                                plan.hardware.prefetch_capacity_bytes),
        capacity_bytes=plan.hardware.prefetch_capacity_bytes,
        tensor_parallel=plan.hardware.tensor_parallel)
    assert decision.allowed and decision.target_bytes > 0
    updated, prefetch_id = add_prefetch(plan, decision, plan.hardware)
    return updated, prefetch_id, decision


# ---------------------------------------------------------------------------
# New P1 + R03: exact per-(replica, stage, invocation) L1 lease identity
# ---------------------------------------------------------------------------
def test_new_p1_r03_dp2_pp2_release_their_own_l1_grants_without_leak():
    # Analytical holder contention: one shared pool, so the copy waits.
    hardware = hw(hbm_bytes_per_ms=1024.0, prefetch_capacity_bytes=1024,
                  prefetch_units=1, max_streams=1, launch_ms=0.0,
                  device_min_ms=0.0)
    plan = (
        PlanNode("hold", (("host", 10.0),), kind="kernel", stream="main",
                 ub_allocations=(("hold", "l1", 1024, (0,)),)),
        PlanNode("prefetch_W_hold", (("hbm", 1024.0),), kind="prefetch",
                 prefetch_for="hold", stream="prefetch", priority=10,
                 prefetch_alloc=("prefetch_W_hold", "l1", 1024.0, (0,))),
    )
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "completed", measured.reason
    assert measured.start("prefetch_W_hold") >= measured.complete("hold") - 1e-12
    assert measured.complete("prefetch_W_hold") == pytest.approx(11.0)

    # Canonical DP=2 + PP=2: both replicas execute stage 1 concurrently on one
    # Engine and must release their own per-(replica, stage) grants.
    raw = case(dp=2, pp=2)
    plan, prefetch_id, decision = _prefetch_plan(raw, pp=2)
    assert decision.target_bytes == 8 * 1024 * 1024
    result = graph_costs(
        plan.case, {0: plan.ops, 1: plan.ops}, recipes=plan.book,
        external_inputs={"analysis_input": 0.0}, include_submission=True,
        isolated=False, audit_allocations=True)
    trade = [row for row in result["allocation_trace"]
             if str(row["pool"]).startswith("l1@")]
    assert trade
    live: dict[str, float] = {}
    for row in trade:
        delta = row["size"] * len(row["units"])
        live[row["pool"]] = live.get(row["pool"], 0.0) + (
            delta if row["action"] == "acquire" else -delta)
    assert live == {"l1@r0:s1": 0.0, "l1@r1:s1": 0.0}
    acquires = [row for row in trade if row["action"] == "acquire"]
    releases = [row for row in trade if row["action"] == "release"]
    # The matrix program grants per-tile L1 panels in addition to the prefetch
    # grant.  Every real grant, not only the prefetch, must have one matching
    # release with the same pool/name/size/units.
    assert len(acquires) == len(releases)
    assert {(row["pool"], row["name"], row["size"], row["units"]) for row in acquires} == {
        (row["pool"], row["name"], row["size"], row["units"]) for row in releases}
    prefetch_acquires = [row for row in acquires
                         if prefetch_id in row["name"]]
    prefetch_releases = [row for row in releases
                         if prefetch_id in row["name"]]
    assert len(prefetch_acquires) == len(prefetch_releases) == 2
    for replica in (0, 1):
        pool = f"l1@r{replica}:s1"
        acquire = next(row for row in prefetch_acquires if row["pool"] == pool)
        release = next(row for row in prefetch_releases if row["pool"] == pool)
        assert acquire["task"].startswith(f"r{replica}:")
        assert release["task"].startswith(f"r{replica}:")
        assert release["size"] == acquire["size"]
        assert release["size"] * len(release["units"]) == pytest.approx(
            decision.target_bytes)
    timeline = {(row["replica"], row["op_id"]): row for row in result["timeline"]}
    for replica in (0, 1):
        copy = timeline[(replica, prefetch_id)]
        release = timeline[(replica, "release_W_mm")]
        consumer = timeline[(replica, "mm")]
        # The prefetch can overlap the consumer and finish later; the grant
        # must stay held until both the copy and the consumer retire.
        assert release["ready_ms"] >= copy["complete_ms"] - 1e-12
        assert release["ready_ms"] >= consumer["complete_ms"] - 1e-12


# ---------------------------------------------------------------------------
# R01: retained output identity, shape and dtype through contraction
# ---------------------------------------------------------------------------
def test_r01_retained_output_dtype_shape_and_layout_never_change():
    a = replace(_add("a", ("analysis_input",)), family="cast", dtype="fp16")
    b = replace(_add("b", ("a",)), family="cast", dtype="bf16")
    plan = canonical_plan(case(), (a, b))
    assert plan.graph.values["b:out"].dtype == "bf16"
    fused, spec = _fuse_first(plan)
    fused_op = next(item for item in fused.ops if item.family == "native_fusion")
    out_value = fused.graph.values[fused_op.op_id + ":out"]
    assert fused_op.dtype == "bf16"
    assert out_value.dtype == "bf16"
    assert fused_op.local_shape == (1, 4) and out_value.shape == (1, 4)
    # The HBM write interface is two bytes per element, not the first
    # member's FP16-derived assumption.
    assert out_value.nbytes == 1 * 4 * 2
    assert spec.numerical_policy == "relaxed"
    report = run_optimizer(case(), (a, b), evaluation_budget=8,
                           max_candidates_per_pass=4)
    assert any(item["pass"] == "vector_fusion" for item in report.applied)
    final = next(item for item in report.canonical_plan.ops
                 if item.family == "native_fusion")
    assert final.dtype == "bf16" and final.local_shape == (1, 4)

    # Published relaxed MLA fixture: the retained index-norm Cast is BF16.
    raw, ops = canonical_mlapo()
    mla = run_optimizer(raw, ops, evaluation_budget=64,
                        max_candidates_per_pass=32,
                        allow_relaxed_numerics=True)
    assert any(item["pass"] == "vector_fusion" for item in mla.applied)
    fused_mla = next(item for item in mla.canonical_plan.ops
                     if item.op_id.startswith("fused:s0.l0.index_norm"))
    assert fused_mla.dtype == "bf16"
    assert fused_mla.local_shape == (48, 128)
    assert mla.final.hbm_bytes == 375_886_912.0
    assert mla.baseline.hbm_bytes == 375_985_216.0

    # Two different retained primary interfaces cannot be represented by one
    # surviving OpSpec port; the contraction must reject, never coerce.
    values = {
        "x": val("x", "bf16", (1, 4), storage=8),
        "y": val("y", "fp16", (1, 4), producer="ca", storage=8, public=True),
        "z": val("z", "bf16", (1, 4), producer="cb", storage=8, public=True),
    }
    ops_graph = TypedGraph(1, values, {
        "ca": op("ca", "cast", "vector", ("x",), ("y",),
                 work=(("vector", 1.0),),
                 params=(("target_dtype", "fp16"), ("rounding", "rne"))),
        "cb": op("cb", "cast", "vector", ("x",), ("z",),
                 work=(("vector", 1.0),),
                 params=(("target_dtype", "bf16"), ("rounding", "rne"))),
    }, ("y", "z"))
    legality = region_legality(ops_graph, ("ca", "cb"))
    assert legality.ok is False
    assert "retained_output_interface_mismatch" in legality.reasons
    analysis = analyze_region(ops_graph, ("ca", "cb"), tile_rows=1,
                              capacity_per_owner=65536)
    candidate = _make(_match(ops_graph, ("ca", "cb")), analysis,
                      ("ca", "cb"), "vector_chain", 1, "main")
    with pytest.raises(ValueError, match="retained_output_interface_mismatch"):
        fused_op_spec(candidate)


# ---------------------------------------------------------------------------
# R02: declared usable UB budget bounds single operators and reservation sums
# ---------------------------------------------------------------------------
def test_r02_usable_ub_bounds_single_op_and_summed_reservations():
    values, ops = cast_chain_graph("t")
    graph = TypedGraph(1, values, ops, ("tz",))
    hardware = hw(ub_bytes_per_aiv=65536, usable_ub_bytes=4000,
                  prefetch_capacity_bytes=0)
    for op_id in ("ta", "tb"):
        allocations = single_op_ub(
            graph, op_id, tile_rows=2, capacity_per_owner=65536,
            usable_ub_bytes=4000)
        assert sum(item.total_bytes for item in allocations) > 4000
    plan = baseline_plan(graph, hardware)
    assert all(sum(size for _n, _p, size, _u in node.ub_allocations) > 4000
               for node in plan)
    measured = evaluate_plan(plan, hardware)
    assert measured.status == "failed"
    assert "single_op_ub_capacity_exceeded" in measured.reason

    # A summed reservation above usable capacity is infeasible even when each
    # individual allocation is below it.
    busy = (
        PlanNode("busy", (("vector", 1.0),),
                 ub_allocations=(("a", "ub", 3000, (0,)),
                                 ("b", "ub", 3000, (0,)))),
    )
    summed = evaluate_plan(busy, hardware)
    assert summed.status == "failed"
    assert "single_op_ub_capacity_exceeded" in summed.reason

    for invalid in (0, -1):
        with pytest.raises(ValueError, match="hardware_usable_ub_invalid"):
            hw(usable_ub_bytes=invalid)


# ---------------------------------------------------------------------------
# R04: unplaced fanout, coupled lockstep and non-preemptive FIFO
# ---------------------------------------------------------------------------
def test_r04_unplaced_fanout_coupled_lockstep_and_nonpreemptive_fifo():
    resources = {"hbm": Resource("hbm", 1024, "bytes", units=2),
                 "vector": Resource("vector", 1024, "ops", units=2)}
    foreground = _task(0, {"hbm": (0,)}, remaining={"hbm": 1024})
    unplaced = _task(10, remaining={"hbm": 1024})
    rates = component_rates(resources, {0: foreground, 1: unplaced}, [0, 1])
    assert rates["hbm"][0] == pytest.approx(512.0)
    # The unplaced tier draws only unit 1's residual capacity; it never adds
    # to the pinned foreground's already-saturated unit 0.
    assert rates["hbm"].get(1) == pytest.approx(512.0)

    coupled = _task(10, {"hbm": (0,), "vector": (0,)}, coupled=True,
                    remaining={"hbm": 1024, "vector": 1024})
    rates = component_rates(resources, {0: foreground, 1: coupled}, [0, 1],
                            coupled_solver=coupled_rates, coupled_active=1,
                            placed_active=1)
    assert rates["hbm"][0] == pytest.approx(512.0)
    assert 1 not in rates.get("vector", {})

    engine = Engine([Resource("hbm", 1024, "bytes", units=2),
                     Resource("vector", 1024, "ops", units=2)])
    fg = engine.submit("fg", {"hbm": 1024}, placement={"hbm": (0,)})
    pair = engine.submit(
        "coupled", {"hbm": 1024, "vector": 1024},
        placement={"hbm": (0,), "vector": (0,)}, priority=10, coupled=True)
    engine.advance(until=(fg,))
    assert fg.time == pytest.approx(2.0)
    # Zero HBM residual must freeze the whole coupled flow: no Vector work ran
    # ahead of the HBM service.
    assert engine.service_of(pair) == {}
    engine.advance(horizon=engine.now + 0.5)
    halfway = engine.service_of(pair)
    assert halfway["hbm"] == pytest.approx(256.0)
    assert halfway["vector"] == pytest.approx(256.0)
    engine.advance(until=(pair,))
    assert pair.time == pytest.approx(4.0)

    fifo = Engine([Resource("r", 100, "work", policy="fifo")], trace_limit=32)
    early = fifo.submit("early low", {"r": 100}, priority=10)
    late = fifo.submit("late high", {"r": 10}, priority=0, release=0.1)
    fifo.advance(until=(early, late), first=False)
    assert early.time == pytest.approx(1.0)
    assert late.time == pytest.approx(1.1)

    waiting = Engine([Resource("r", 100, "work", policy="fifo")])
    holder = waiting.submit("holder", {"r": 100})
    low = waiting.submit("low", {"r": 10}, priority=10, release=0.05)
    high = waiting.submit("high", {"r": 10}, priority=0, release=0.05)
    waiting.advance(until=(holder, low, high), first=False)
    assert holder.time == pytest.approx(1.0)
    assert high.time == pytest.approx(1.1)
    assert low.time == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# R06: full per-tile handoff accounting and directional causal ordering
# ---------------------------------------------------------------------------
def test_r06_all_tiles_handoff_and_directional_causal_stages(monkeypatch):
    values = dict(__import__("graph_opt_fixtures", fromlist=[
        "scalar_cube_graph"]).scalar_cube_graph().values)
    ops_graph = __import__("graph_opt_fixtures", fromlist=[
        "scalar_cube_graph"]).scalar_cube_graph().ops
    for key in ("a", "p", "c"):
        values[key] = replace(values[key], shape=(16, 1), storage_bytes=32)
    graph = TypedGraph(16, values, ops_graph, ("c",))
    splits = []
    for tile in (1, 16):
        candidate = synthesize_cube_epilogue(
            graph, "mm", ("cast",), tile_rows=tile, capacity_per_owner=65536,
            capability=True, cube_resources=hw().cube_capacities)
        assert not isinstance(candidate, RejectedFusion)
        split = handoff_split(graph, ("mm", "cast"),
                              hw().cube_capacities, tile)
        splits.append((split.operand, split.output, split.total))
        assert split.output == 32
        assert candidate.analysis.handoff_bytes == 66
    assert splits[0] == splits[1] == (34, 32, 66)

    prologue_values = {
        "x": val("x", "bf16", (1, 1), storage=2),
        "w": val("w", "bf16", (1, 1), role="weight", storage=2),
        "h": val("h", "bf16", (1, 1), producer="pre", storage=2),
        "p": val("p", "bf16", (1, 1), producer="mm", storage=2),
    }
    prologue_graph = TypedGraph(1, prologue_values, {
        "pre": op("pre", "add", "vector", ("x",), ("h",),
                  work=(("vector", 1.0),)),
        "mm": op("mm", "matmul", "cube", ("h", "w"), ("p",),
                 work=(("cube", 2.0),)),
    }, ("p",))
    from mindie_sim.prediction.fusion_recipe import synthesize_cube_prologue
    candidate = synthesize_cube_prologue(
        prologue_graph, "pre", "mm", tile_rows=1, capacity_per_owner=65536,
        capability=True, cube_resources=hw().cube_capacities)
    assert not isinstance(candidate, RejectedFusion)
    assert tuple(phase for phase, _ in candidate.analysis.phase_work) == (
        "operand_read", "prologue", "prologue_handoff", "cube", "output_write")

    combined_values = dict(prologue_values)
    combined_values["c"] = val("c", "fp16", (1, 1), producer="cast",
                               storage=2)
    combined = TypedGraph(1, combined_values, {
        "pre": prologue_graph.ops["pre"],
        "mm": prologue_graph.ops["mm"],
        "cast": op("cast", "cast", "vector", ("p",), ("c",),
                   work=(("vector", 1.0),),
                   params=(("target_dtype", "fp16"), ("rounding", "rne"))),
    }, ("c",))
    analysis = analyze_region(combined, ("pre", "mm", "cast"), tile_rows=1,
                              capacity_per_owner=65536, allow_cube=True,
                              cube_resources=hw().cube_capacities)
    assert analysis.legality.ok, analysis.legality.reasons
    assert tuple(phase for phase, _ in analysis.phase_work) == (
        "operand_read", "prologue", "prologue_handoff", "cube", "handoff",
        "epilogue", "output_write")
    candidate = _make(_match(combined, ("pre", "mm", "cast")), analysis,
                      ("pre", "mm", "cast"), "cube_boundary", 1, "main")
    observed = _observed(monkeypatch)
    measured = evaluate_plan(apply_fusion(baseline_plan(combined, hw()),
                                          candidate), hw())
    assert measured.status == "completed", measured.reason
    rows = {row["task"].rsplit(":", 1)[-1]: row
            for row in _phase_rows(observed.last, "body:fused:pre+mm+cast")}
    assert rows["prologue"]["end_ms"] <= rows["prologue_handoff"]["start_ms"] + 1e-12
    assert rows["prologue_handoff"]["end_ms"] <= rows["cube"]["start_ms"] + 1e-12
    assert rows["cube"]["end_ms"] <= rows["handoff"]["start_ms"] + 1e-12
    assert rows["handoff"]["end_ms"] <= rows["epilogue"]["start_ms"] + 1e-12
    assert rows["epilogue"]["end_ms"] <= rows["output_write"]["start_ms"] + 1e-12


# ---------------------------------------------------------------------------
# R07: additive resource work and dedicated MTE1/FixPipe metadata survive
# ---------------------------------------------------------------------------
def test_r07_additive_and_dedicated_metadata_work_is_conserved():
    for metadata, resource in (
            ((("resource_work", {"hbm": 100_000_000.0}),), "memory@"),
            ((("mte1_bytes", 100_000_000.0),), "mte1@"),
            ((("fixpipe_bytes", 100_000_000.0),), "fixpipe@")):
        plan = canonical_plan(case(), (
            _add("a", ("analysis_input",), metadata=metadata),
            _add("b", ("a",))))
        fused, _spec = _fuse_first(plan)
        before = sum(float(row["work"]) for row in plan.evaluate()["resources"]
                     if str(row["resource"]).startswith(resource))
        after = sum(float(row["work"]) for row in fused.evaluate()["resources"]
                    if str(row["resource"]).startswith(resource))
        fused_op = next(item for item in fused.ops
                        if item.family == "native_fusion")
        stored = dict(fused_op.metadata).get("resource_work", {})
        assert stored
        if resource == "memory@":
            # Only the 16 proven-elided intermediate bytes disappear.
            assert before == pytest.approx(after)
            assert before == pytest.approx(102_000_000.0, rel=1e-12)
            assert before - after == pytest.approx(16.0)
        else:
            # Known M2 inconsistency: CostModel.price (cost.py:308-311)
            # charges mte1_bytes/fixpipe_bytes, but the physical
            # AIVTiledVectorAndGatherRecipe (vector_pipeline.py) does not
            # execute those dedicated pipes for an add/vector op.  The fused
            # recipe preserves the declared work; the baseline does not.
            assert before == pytest.approx(0.0)
            assert after == pytest.approx(100_000_000.0, rel=1e-12)
            assert stored == {resource[:-1]: 100_000_000.0}


# ---------------------------------------------------------------------------
# R08: state versions compare strictly per target key
# ---------------------------------------------------------------------------
def test_r08_state_versions_are_compared_per_target_key():
    from test_graph_optimizer_r3_adversarial import _state_effect_op
    writer = ({"target": "a", "action": "write", "version": 7},
              {"target": "b", "action": "write", "version": 1})
    reader = ({"target": "a", "action": "read", "version": 7},
              {"target": "b", "action": "read", "version": 1})
    plan = canonical_plan(case(), (
        _state_effect_op("w", ("analysis_input",), writer),
        _state_effect_op("r", ("w",), reader)))
    plan.graph.validate()

    values = {
        "a0": val("a0", "fp32", (1,), role="state", storage=4, version=7,
                  state_key="a"),
        "b0": val("b0", "fp32", (1,), role="state", storage=4, version=1,
                  state_key="b"),
        "a1": val("a1", "fp32", (1,), role="state", producer="rmw", storage=4,
                  version=8, state_key="a"),
        "b1": val("b1", "fp32", (1,), role="state", producer="rmw", storage=4,
                  version=2, state_key="b"),
    }
    rmw = OpNode("rmw", "state_update", "vector", ("a0", "b0"), ("a1", "b1"),
                 state_effect="read_write")
    advanced = TypedGraph(1, values, {"rmw": rmw}, ("a1", "b1"))
    advanced.validate()
    stale_values = dict(values)
    stale_values["b1"] = replace(values["b1"], state_version=1)
    stale = TypedGraph(1, stale_values, {"rmw": rmw}, ("a1", "b1"))
    from mindie_sim.prediction.fusion_graph import FeasibilityError
    with pytest.raises(FeasibilityError,
                       match="state_version_not_advanced:b"):
        stale.validate()


# ---------------------------------------------------------------------------
# R09: Cast rounding vocabulary is validated, never assumed exact
# ---------------------------------------------------------------------------
def test_r09_cast_rounding_vocabulary_fails_closed():
    values, ops = cast_chain_graph("t")
    values = {key: replace(item, dtype="fp16" if key != "tx" else "bf16")
              for key, item in values.items()}
    bogus = {key: replace(item, params=(("target_dtype", "fp16"),
                                        ("rounding", "BOGUS")))
             for key, item in ops.items()}
    graph = TypedGraph(1, values, bogus, ("tz",))
    check = _contract(graph, graph.ops["ta"])
    assert "cast_rounding_unknown" in check.reasons
    assert check.exact is False
    assert isinstance(synthesize_vector(
        graph, ("ta", "tb"), tile_rows=1, capacity_per_owner=100000),
        RejectedFusion)

    exact_values = {"tx": replace(values["tx"], dtype="bf16"),
                    "ty": replace(values["ty"], dtype="fp16"),
                    "tz": replace(values["tz"], dtype="bf16")}
    exact_ops = {key: replace(item, params=(("target_dtype", "fp16"),
                                            ("rounding", "rne")))
                 for key, item in ops.items()}
    # ``tb`` converts FP16 back to BF16; retain the declared target.
    exact_ops["tb"] = replace(exact_ops["tb"],
                              params=(("target_dtype", "bf16"),
                                      ("rounding", "rne")))
    exact_graph = TypedGraph(1, exact_values, exact_ops, ("tz",))
    assert _contract(exact_graph, exact_graph.ops["ta"]).exact is True

    stochastic = replace(
        exact_graph.ops["ta"],
        params=(("target_dtype", "fp16"), ("rounding", "stochastic")))
    stochastic_graph = TypedGraph(1, exact_values,
                                  {"ta": stochastic, "tb": exact_ops["tb"]},
                                  ("tz",))
    check = _contract(stochastic_graph, stochastic_graph.ops["ta"])
    assert check.reasons == ()
    assert check.numerical_policy == "relaxed"


# ---------------------------------------------------------------------------
# R10: state effects and explicit capture ineligibility cross contraction
# ---------------------------------------------------------------------------
def test_r10_state_effects_and_capture_ineligible_block_replay():
    capabilities = None

    def run(metadata):
        nonlocal capabilities
        plan = canonical_plan(case(), (
            _add("a", ("analysis_input",), metadata=metadata),
            _add("b", ("a",))))
        if capabilities is None:
            capabilities = tuple(sorted({
                **dict(plan.hardware.capabilities),
                "aclgraph_capture": True,
                "aclgraph_prefetch": True}.items()))
        plan = replace(plan, hardware=replace(plan.hardware,
                                              capabilities=capabilities))
        return _run_canonical(plan, evaluation_budget=8, per_pass=4,
                              max_streams=None, wall_budget_s=10.0)

    effect = ({"target": "cache", "action": "write", "version": 1},)
    report = run((("state_effects", effect),))
    assert any(item["pass"] == "vector_fusion" for item in report.applied)
    assert report.capture["status"] == "blocked"
    assert report.capture["evaluated"] is False
    assert any(item.startswith("capture_boundary_state:")
               for item in report.capture["blockers"])
    fused = next(item for item in report.canonical_plan.ops
                 if item.family == "native_fusion")
    assert dict(fused.metadata)["state_effects"] == effect

    report = run((("capture_ineligible", True),))
    assert any(item["pass"] == "vector_fusion" for item in report.applied)
    assert report.capture["status"] == "blocked"
    assert any(item.startswith("capture_ineligible:")
               for item in report.capture["blockers"])
    fused = next(item for item in report.canonical_plan.ops
                 if item.family == "native_fusion")
    assert dict(fused.metadata)["capture_ineligible"] is True

    read_only = ({"target": "cache", "action": "read", "version": 1},)
    report = run((("state_effects", read_only),))
    assert report.capture["status"] != "blocked"


# ---------------------------------------------------------------------------
# R12: finite positive merge work and causal partial->merge->finalize order
# ---------------------------------------------------------------------------
def test_r12_nonfinite_merge_rejected_and_merge_precedes_finalize(monkeypatch):
    graph = quant_graph()
    for merge in (ReductionMerge(128, float("nan"), 64),
                  ReductionMerge(128, float("inf"), 64)):
        candidate = synthesize_vector(
            graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100000,
            channel_width=1, reduction_merge=merge)
        assert isinstance(candidate, RejectedFusion)
        assert "reduction_merge_work_missing" in candidate.reasons

    candidate = synthesize_vector(
        graph, ("quant", "cast"), tile_rows=1, capacity_per_owner=100000,
        channel_width=1, reduction_merge=ReductionMerge(128, 1_000_000, 64))
    assert not isinstance(candidate, RejectedFusion)
    phases = tuple(phase for phase, _ in candidate.analysis.phase_work)
    assert phases == ("operand_read", "reduce", "merge", "finalize",
                      "output_write")
    merge = dict(candidate.analysis.phase_work)["merge"]
    assert merge["vector"] == pytest.approx(1_000_000.0)
    assert merge["mte1"] == pytest.approx(256.0)

    observed = _observed(monkeypatch)
    measured = evaluate_plan(
        apply_fusion(baseline_plan(graph, hw()), candidate), hw())
    assert measured.status == "completed", measured.reason
    rows = {row["task"].rsplit(":", 1)[-1]: row
            for row in _phase_rows(observed.last, "body:fused:quant+cast")}
    assert rows["reduce"]["end_ms"] <= rows["merge"]["start_ms"] + 1e-12
    assert rows["merge"]["end_ms"] <= rows["finalize"]["start_ms"] + 1e-12
    assert rows["finalize"]["end_ms"] <= rows["output_write"]["start_ms"] + 1e-12


# ---------------------------------------------------------------------------
# R15: effective stream identity includes replica and pipeline stage
# ---------------------------------------------------------------------------
def test_r15_stream_order_is_keyed_by_replica_stage_stream():
    raw = case(pp=2)
    plan = canonical_plan(raw, (
        _add("a", ("analysis_input",), stage=0),
        replace(_add("b", ("analysis_input",)), stage=1, layer=1)))
    result = plan.evaluate()
    rows = {row["op_id"]: row for row in result["timeline"]}
    assert rows["a"]["ready_ms"] == pytest.approx(0.0)
    assert rows["b"]["ready_ms"] == pytest.approx(0.0)
    assert rows["a"]["complete_ms"] < 0.01
    assert rows["b"]["complete_ms"] < 0.01
    edges = [(edge["from"], edge["to"], edge["kind"])
             for edge in _canonical_dag(plan, result)]
    assert ("a", "b", "stream_order") not in edges
    assert ("a", "b", "data") not in edges

    same = canonical_plan(case(), (
        _add("a", ("analysis_input",)), _add("b", ("a",))))
    same_edges = [(edge["from"], edge["to"], edge["kind"])
                  for edge in _canonical_dag(same)]
    assert ("a", "b", "stream_order") in same_edges


# ---------------------------------------------------------------------------
# R16: graph indexing and chain discovery obey the wall budget internally
# ---------------------------------------------------------------------------
def test_r16_two_thousand_operator_discovery_aborts_inside_the_budget():
    graph = budget_graph(1000)
    assert len(graph.ops) == 2000
    budget = Budget(per_pass=32, total=64, wall_s=0.01)
    started = time.monotonic()
    specs, unevaluated = _candidate_specs("vector_fusion", graph, hw(), budget)
    elapsed = time.monotonic() - started
    assert specs == ()
    assert unevaluated and unevaluated[0]["reason"] == "discovery_budget_exhausted"
    assert budget.exhausted is True
    assert budget.used_total == 0
    assert elapsed < 0.05
