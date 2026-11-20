"""Device STARS descriptor queue, Prefill Eager D2H barrier, and datapath specs.

[Role]
  Verify ``DatapathCapabilitySpec`` Vol-2 extensions and ``route()`` input
  validation, device-boundary ``StarsDescriptorQueue`` FIFO + credit depth +
  whole-kernel retirement + piecewise segment / per-die scope physics,
  Prefill Eager ``TokenDispatcherWithAll2AllV`` ``D2H`` barrier (unit +
  production wiring), ServingIO root-complex contention, and spine invariants.

[Invariants]
  * ``DatapathCapabilitySpec.route()`` rejects unknown representations,
    out-of-range ``owner_group``, and invalid ``tile`` geometry.
  * ``StarsDescriptorQueue`` enforces credit depth even at ``issue_interval_us=0``,
    preserves same-stream execution order under ``descriptor_ready``, binds
    credits to whole-kernel completion, retires handles cleanly, charges
    ``first_issue_us`` once per piecewise segment, and respects ``queue_scope``.
  * ``EagerSplitBarrier`` runs ``repeat_interleave`` after ``allgather_counts``,
    keeps pinned vs pageable order-independent on one Engine, wires into
    production Eager MoE dispatch, and fences strictly when local experts <= 1.
  * ``ServingIO`` couples H2D/D2H to shared ``(node, root_complex_id, dir)``.

[Timeouts & Locks]
  Tier-0 pure in-memory Engine and spine checks (<3s); no locks or I/O.

[Side-Effect Policy]
  Constructs local in-memory Engine, Case, and Execution instances only.
"""
from __future__ import annotations

from dataclasses import replace
import importlib
import math
from types import SimpleNamespace

import pytest

from mindie_sim import api as sdk_api
from mindie_sim.backends import ascend
from mindie_sim.prediction.api import predict
from mindie_sim.prediction.collectives import RootComplexKey
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.evaluation import ExecutionBindings
from mindie_sim.prediction.execution import (
    EagerSplitBarrier,
    Execution,
    SplitBarrierSpec,
    StarsDescriptorQueue,
    compute_uncovered_stars_ns,
    is_zero_kernel_view,
)
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.device_input import StarsSpec
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import IterationPlan, IterationWork, OpSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

LEGACY_ARCH_SKU = "Ascend910_9362"
DUAL_DIE_LARGE_SKU = "Ascend910_9382"
NEXTGEN_ARCH_SKU = "Ascend950DT_9582"
TINY_MODEL_ID = "qwen3-32b"
MOE_MODEL_ID = "deepseek-v3"


def _make_machine(
    sku: str,
    *,
    num_devices: int = 2,
    ranks_per_node: int | None = None,
) -> Hardware:
    raw: dict = {"sku": sku, "num_devices": num_devices}
    if ranks_per_node is not None:
        raw["ranks_per_node"] = ranks_per_node
    return Hardware.from_mapping(ascend(raw))


def _tiny_case(*, stars_spec: StarsSpec | None = None, tp: int = 2):
    case = sdk_api.compose_case(
        TINY_MODEL_ID,
        hardware=LEGACY_ARCH_SKU,
        options={"dp": 1, "tp": tp, "pp": 1, "layers": 1},
    )
    if stars_spec is not None:
        new_dies = tuple(
            replace(die, dispatch_sync=replace(die.dispatch_sync, stars=stars_spec))
            for die in case.machine.dies
        )
        case = replace(case, machine=replace(case.machine, dies=new_dies))
    return case


def test_datapath_capabilities_and_route_argument_validation():
    """All 16 profiles populate Vol-2 fields and route() rejects invalid args."""
    all_profiles = importlib.import_module("mindie_sim.backends.ascend").profiles()
    assert len(all_profiles) == 16
    for sku_name in all_profiles:
        dp = datapath_for_die(_make_machine(sku_name, num_devices=2).dies[0])
        assert dp.subtile_wave and dp.stars_spec is not None
        assert dp.stars_spec.queue_depth > 0 and dp.stars_spec.lanes >= 1
        assert dp.memory_access is not None and dp.memory_access.sector_bytes == 32
        assert dp.hbm_read_bytes_per_s > 0.0 and dp.hbm_write_bytes_per_s > 0.0
        assert 0.0 <= dp.hbm_rw_alpha <= 1.0

    dp_legacy = datapath_for_die(_make_machine(LEGACY_ARCH_SKU).dies[0])
    dp_next = datapath_for_die(_make_machine(NEXTGEN_ARCH_SKU).dies[0])
    assert dp_legacy.simt_ub_carveout_bytes == 0
    assert dp_next.simt_ub_carveout_bytes == 16384
    assert dp_next.usable_bytes("ub") - dp_next.usable_bytes("ub", kernel="simt") == 16384

    route_ok = dp_next.route(
        "l0c", "ub", representation="fp32", owner_group=(0, 1), tile=(16, 16)
    )
    assert route_ok.path.unit == "fixpipe"
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:l0c->ub"):
        dp_legacy.route("l0c", "ub")
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_representation_unknown"):
        dp_next.route("l0c", "ub", representation="no_such_representation")
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_owner_unreachable"):
        dp_next.route("l0c", "ub", owner_group=(0, 47))
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_tile"):
        dp_next.route("l0c", "ub", tile=(1, 1))
    # ub->l1 normalizes named owners once & validates BOTH src and dst buffer capacities (#7)
    assert dp_next.route("ub", "l1", owner_group={"aic": 1, "aiv": 2}).path.source == "ub"
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_owner_unreachable"):
        dp_next.route("ub", "l1", owner_group={"aic": 2, "aiv": 1})
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_tile_exceeds_buffer"):
        dp_next.route("ub", "l1", representation="fp16", tile=(256, 512))


def test_stars_queue_covered_exposed_zero_issue_depth_and_stream_order():
    """Verify STARS overlap, zero-issue credit serialization, same-stream FIFO, retirement."""
    # 1) Covered behind compute: only first 5 us uncovered
    spec_cov = StarsSpec(
        queue_scope="die", queue_depth=8, lanes=1, issue_interval_us=5.0,
        first_issue_us=0.0, release_policy="descriptor_ready", evidence_id="ev_cov",
    )
    q_cov = StarsDescriptorQueue(spec_cov, die_id=0)
    eng = Engine([Resource("aic", 1.0, "ms", "fifo")])
    ready = eng.submit("r")
    dones = [
        q_cov.submit_kernel(eng, replay=0, stream="main", kernel_occurrence=i,
                            descriptor_ready=ready, work={"aic": 0.050})[0]
        for i in range(6)
    ]
    eng.advance()
    s_cov = q_cov.summary()
    assert s_cov["descriptor_count"] == 6 and s_cov["uncovered_stars_ns"] == 5_000
    assert s_cov["active_records"] == 0
    assert math.isclose(dones[-1].time, 0.005 + 0.300, rel_tol=1e-9)

    # 2) Zero issue_interval_us with queue_depth=1 still serializes independent 1 ms kernels
    spec_zero = StarsSpec(
        queue_scope="die", queue_depth=1, lanes=1, issue_interval_us=0.0,
        first_issue_us=0.0, release_policy="descriptor_ready", evidence_id="ev_zero",
    )
    q_zero = StarsDescriptorQueue(spec_zero, die_id=0)
    eng_z = Engine([Resource("ua", 1.0, "ms"), Resource("ub", 1.0, "ms")])
    rz = eng_z.submit("rz")
    d0, r0 = q_zero.submit_kernel(eng_z, replay=0, stream="s0", kernel_occurrence=0,
                                  descriptor_ready=rz, work={"ua": 1.0})
    d1, r1 = q_zero.submit_kernel(eng_z, replay=0, stream="s1", kernel_occurrence=1,
                                  descriptor_ready=rz, work={"ub": 1.0})
    eng_z.advance()
    assert math.isclose(d0.time, 1.0, rel_tol=1e-9)
    assert math.isclose(d1.time, 2.0, rel_tol=1e-9)
    # Event handles retired after completion (#19)
    assert r0 is not None and r0.issue_done_event is None and r0.kernel_done_event is None
    assert r1 is not None and q_zero.summary()["active_records"] == 0

    # 3) Same-stream execution order preserved under descriptor_ready while descriptor prefetches
    spec_ord = StarsSpec(
        queue_scope="die", queue_depth=4, lanes=1, issue_interval_us=1.0,
        first_issue_us=0.0, release_policy="descriptor_ready", evidence_id="ev_ord",
    )
    q_ord = StarsDescriptorQueue(spec_ord, die_id=0)
    eng_o = Engine([Resource("u0", 1.0, "ms"), Resource("u1", 1.0, "ms")])
    ro = eng_o.submit("ro")
    k0_done, rec0 = q_ord.submit_kernel(
        eng_o, replay=0, stream="s0", kernel_occurrence=0,
        descriptor_ready=ro, work={"u0": 0.100},
    )
    k1_done, rec1 = q_ord.submit_kernel(
        eng_o, replay=0, stream="s0", kernel_occurrence=1,
        descriptor_ready=ro, work={"u1": 0.001},
    )
    eng_o.advance()
    # k1 prefetched at 1..2 us, but executed strictly after k0 finished at 101 us
    assert rec0 is not None and rec1 is not None
    assert math.isclose(k0_done.time, 0.101, rel_tol=1e-9)
    assert math.isclose(k1_done.time, 0.102, rel_tol=1e-9)

    # 4) Whole-kernel group completion holds credit until full kernel retires (not entry phase)
    q_grp = StarsDescriptorQueue(spec_zero, die_id=0)
    eng_g = Engine([Resource("u0", 1.0, "ms"), Resource("u1", 1.0, "ms")])
    rg = eng_g.submit("rg")
    rec_g0, alloc_g0 = q_grp.issue_descriptor(
        eng_g, replay=0, stream="s0", kernel_occurrence=0, descriptor_ready=rg
    )
    entry_done = eng_g.submit("g0:entry", {"u0": 0.010}, after=(rec_g0.issue_done_event,))
    tail_done = eng_g.submit("g0:tail", {"u0": 0.090}, after=(entry_done,))
    q_grp.bind_kernel_execution(
        eng_g, rec_g0, alloc_g0,
        kernel_start=eng_g.start_boundary(entry_done, "g0:start"),
        kernel_done=tail_done,
    )
    g1_done, _ = q_grp.submit_kernel(
        eng_g, replay=0, stream="s1", kernel_occurrence=1,
        descriptor_ready=rg, work={"u1": 0.020},
    )
    eng_g.advance()
    assert math.isclose(g1_done.time, 0.120, rel_tol=1e-9)

    # 5) 400 completed depth-1 replays fold intervals/keys into O(1) bounded totals (#5)
    q_bnd = StarsDescriptorQueue(replace(spec_cov, queue_depth=1, first_issue_us=2.0), die_id=0)
    eng_b = Engine([Resource("aic", 1.0, "ms", "fifo")])
    prev_ev = eng_b.submit("r0")
    for rep in range(400):
        prev_ev, _ = q_bnd.submit_kernel(
            eng_b, replay=rep, stream="main", kernel_occurrence=rep,
            descriptor_ready=prev_ev, work={"aic": 0.010},
        )
    eng_b.advance()
    s_bnd = q_bnd.summary()
    assert s_bnd["descriptor_count"] == 400 and s_bnd["descriptor_union_ns"] == 400 * 7_000
    assert len(q_bnd._retired_desc_ns) == 0 and len(q_bnd._retired_comp_ns) == 0
    assert len(q_bnd._replay_first_charged) <= 1


def test_stars_release_policies_multistream_and_zero_kernel_views():
    """Compare release policies, credit depth, 3-stream union, and zero-kernel views."""
    def _run_policy(policy: str, *, queue_depth: int, count: int) -> list[tuple[float, float]]:
        spec = StarsSpec(
            queue_scope="die", queue_depth=queue_depth, lanes=1,
            issue_interval_us=10.0, first_issue_us=0.0,
            release_policy=policy, evidence_id=f"test_{policy}",
        )
        queue = StarsDescriptorQueue(spec, die_id=0)
        engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
        ready = engine.submit("ready")
        intervals: list[tuple[float, float]] = []
        recs = []
        for idx in range(count):
            _, rec = queue.submit_kernel(
                engine, replay=0, stream="s0", kernel_occurrence=idx,
                descriptor_ready=ready, work={"aic": 0.050},
            )
            assert rec is not None
            recs.append(rec)
        engine.advance()
        for rec in recs:
            assert rec.issue_interval_ms is not None
            intervals.append(rec.issue_interval_ms)
        return intervals

    ready_iv = _run_policy("descriptor_ready", queue_depth=4, count=3)
    pred_iv = _run_policy("stream_predecessor_done", queue_depth=4, count=3)
    bp_iv = _run_policy("descriptor_ready", queue_depth=2, count=3)
    assert math.isclose(ready_iv[2][0], 0.020, abs_tol=1e-9)
    assert math.isclose(pred_iv[1][0], 0.060, abs_tol=1e-9)
    assert math.isclose(bp_iv[2][0], 0.060, abs_tol=1e-9)

    spec_ms = StarsSpec(
        queue_scope="die", queue_depth=8, lanes=1, issue_interval_us=5.0,
        first_issue_us=0.0, release_policy="descriptor_ready", evidence_id="ev_ms",
    )
    q_ms = StarsDescriptorQueue(spec_ms, die_id=0)
    eng_ms = Engine([Resource(f"u{i}", 1.0, "ms") for i in range(3)])
    rms = eng_ms.submit("ready")
    assert is_zero_kernel_view(SimpleNamespace(family="reshape", metadata=())) is True
    assert is_zero_kernel_view(SimpleNamespace(family="matmul", metadata=(("zero_kernel_view", True),))) is True
    assert q_ms.submit_kernel(
        eng_ms, replay=0, stream="s0", kernel_occurrence=0, descriptor_ready=rms,
        work={"u0": 0.0}, op=SimpleNamespace(family="reshape", metadata=()),
    )[1] is None
    for idx in range(3):
        q_ms.submit_kernel(
            eng_ms, replay=0, stream=f"s{idx}", kernel_occurrence=idx + 1,
            descriptor_ready=rms, work={f"u{idx}": 0.050},
        )
    eng_ms.advance()
    s_ms = q_ms.summary()
    assert s_ms["descriptor_count"] == 3 and s_ms["uncovered_stars_ns"] == 5_000
    assert compute_uncovered_stars_ns([(0.0, 0.015)], [(0.005, 0.055), (0.010, 0.065)]) == 5_000


def test_execution_stars_piecewise_segments_queue_scope_and_host_disabled():
    """Verify include_submission=False, piecewise first_issue_us/segment, die vs package scope."""
    shape = (8, 16, 64)
    plan = IterationPlan(
        iteration=0,
        requests=(RequestView("req0", q_len=1, context=16, samples=True, phase="decode"),),
    )

    # 1) Piecewise two segments charge first_issue_us ONCE PER SEGMENT ((25+5)*2 = 60 us)
    pw_spec = StarsSpec(
        queue_scope="die", queue_depth=32, lanes=1,
        issue_interval_us=5.0, first_issue_us=25.0,
        release_policy="descriptor_ready", evidence_id="ev_pw",
    )
    pw_ops = (
        OpSpec(
            op_id="op_view", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            requests=("req0",), metadata=(("zero_kernel_view", True),),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
        OpSpec(
            op_id="op_seg0", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            requests=("req0",),
            metadata=(("graph_captured", True), ("graph_segment", 0)),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
        OpSpec(
            op_id="op_seg1", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            deps=("op_seg0",), requests=("req0",),
            metadata=(("graph_captured", True), ("graph_segment", 1)),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
    )
    ex_die = Execution(_tiny_case(stars_spec=pw_spec, tp=2), CostModel(_tiny_case(stars_spec=pw_spec, tp=2)), include_submission=False)
    ex_die.add(
        IterationWork(
            plan=plan, ops=pw_ops, stage_count=1, mode="decode", valid_tokens=1,
            padded_tokens=1, graph_bucket=1, recipe="default", replay_scope="piecewise",
        ),
        replica=0, release_ms=0.0,
    )
    ex_die.engine.advance()
    sum_die = ex_die.summary()
    assert sum_die["submission"]["included"] is False and sum_die["submission"]["host_service_ms"] == 0
    stage_die = sum_die["stars"]["r0:s0"]
    # TP=2 issues across BOTH participating dies: 2 segments x 2 dies = 4 x 30 us (#5)
    assert stage_die["descriptor_count"] == 4 and stage_die["die_count"] == 2
    assert stage_die["total_descriptor_ns"] == 120_000
    assert stage_die["dies"]["d0"]["descriptor_union_ns"] == 60_000
    assert stage_die["dies"]["d1"]["descriptor_union_ns"] == 60_000

    # 2) dp=2, tp=1 on dual-die package 0: queue_scope="die" isolates vs "package" contends
    def _run_two_replicas(q_scope: str) -> dict:
        sp = replace(pw_spec, queue_scope=q_scope, first_issue_us=0.0, issue_interval_us=8.0)
        base = sdk_api.compose_case(
            TINY_MODEL_ID, hardware=LEGACY_ARCH_SKU,
            options={"dp": 2, "tp": 1, "pp": 1, "ranks_per_node": 2, "layers": 1},
        )
        dies = tuple(replace(d, dispatch_sync=replace(d.dispatch_sync, stars=sp)) for d in base.machine.dies)
        c2 = replace(base, machine=replace(base.machine, dies=dies))
        ex2 = Execution(c2, CostModel(c2), include_submission=False)
        wk = IterationWork(
            plan=plan, ops=pw_ops, stage_count=1, mode="decode", valid_tokens=1,
            padded_tokens=1, graph_bucket=1, recipe="default", replay_scope="piecewise",
        )
        ex2.add(wk, replica=0, release_ms=0.0)
        ex2.add(wk, replica=1, release_ms=0.0)
        ex2.engine.advance()
        return ex2.summary()["stars"]

    stars_die = _run_two_replicas("die")
    stars_pkg = _run_two_replicas("package")
    # Independent die controllers: 2 descriptors (16 us) per replica
    assert stars_die["r0:s0"]["descriptor_count"] == 2 and stars_die["r0:s0"]["descriptor_union_ns"] == 16_000
    assert stars_die["r1:s0"]["descriptor_count"] == 2 and stars_die["r1:s0"]["descriptor_union_ns"] == 16_000
    # Shared package controller: 2 replicas x 2 ops = 4 descriptors (32 us) serialized on pkg 0
    assert stars_pkg["r0:s0"]["descriptor_count"] == 4 and stars_pkg["r0:s0"]["descriptor_union_ns"] == 32_000


def test_eager_split_barrier_dependency_order_independence_and_prod_wiring():
    """Verify repeat_interleave after AllGather, order-independent pinned/pageable, prod wiring."""
    machine = _make_machine(DUAL_DIE_LARGE_SKU, num_devices=4, ranks_per_node=4)
    die0 = machine.dies[0]
    rc_d2h = RootComplexKey(0, die0.root_complex_id, "d2h").resource_name()

    # 1) repeat_interleave depends on allgather_counts: 20 + 40 + 30 = 90 us (NOT 50 us!)
    spec_multi = SplitBarrierSpec(
        ep_size=16, num_local_experts=4, split_dtype_bytes=8,
        pinned=True, non_blocking=True, host_consume_us=5.0,
        histogram_ms=0.020, repeat_interleave_ms=0.030, allgather_ms=0.040,
    )
    eng_m = Engine([Resource("other_stream", 1.0, "ms")])
    rm = eng_m.submit("ready")
    slow_m = eng_m.submit("indep", {"other_stream": 0.500}, after=(rm,))
    res_m = EagerSplitBarrier.submit(
        eng_m, machine, die0, spec_multi, rm, independent_device_events=(slow_m,),
    )
    eng_m.advance()
    assert res_m.explicit_device_sync is None and rc_d2h in res_m.d2h_resources
    assert math.isclose(res_m.allgather_counts.time, 0.060, rel_tol=1e-9)
    assert math.isclose(res_m.device_branch_done.time, 0.090, rel_tol=1e-9)
    assert res_m.dispatch_ready.time >= 0.090 and res_m.dispatch_ready.time < 0.200

    # 2) num_local_experts <= 1 fences independent device work via torch.npu.synchronize()
    spec_single = SplitBarrierSpec(
        ep_size=16, num_local_experts=1, split_dtype_bytes=8,
        pinned=True, non_blocking=True, host_consume_us=5.0,
        histogram_ms=0.020, allgather_ms=0.040, sync_overhead_us=10.0,
    )
    eng_s = Engine([Resource("other_stream", 1.0, "ms")])
    rs = eng_s.submit("ready")
    slow_s = eng_s.submit("indep", {"other_stream": 0.500}, after=(rs,))
    res_s = EagerSplitBarrier.submit(
        eng_s, machine, die0, spec_single, rs, independent_device_events=(slow_s,),
    )
    eng_s.advance()
    assert res_s.explicit_device_sync is not None
    assert math.isclose(res_s.explicit_device_sync.time, 0.510, rel_tol=1e-9)
    assert math.isclose(res_s.dispatch_ready.time, 0.515, rel_tol=1e-9)

    # 3) Pinned vs pageable on the SAME Engine is 100% order-independent (#6)
    big_pin = SplitBarrierSpec(ep_size=1_000_000, num_local_experts=2, split_dtype_bytes=8, pinned=True)
    big_pag = SplitBarrierSpec(ep_size=1_000_000, num_local_experts=2, split_dtype_bytes=8, pinned=False)

    def _run_pair(first_pinned: bool) -> tuple[float, float]:
        eng = Engine([])
        t0 = eng.submit("t0")
        first = EagerSplitBarrier.submit(eng, machine, die0, big_pin if first_pinned else big_pag, t0, name="first")
        t1 = eng.submit("t1", after=(first.dispatch_ready,))
        second = EagerSplitBarrier.submit(eng, machine, die0, big_pag if first_pinned else big_pin, t1, name="second")
        eng.advance()
        d_first = first.input_splits_d2h.time
        d_second = second.input_splits_d2h.time - t1.time
        return (d_first, d_second) if first_pinned else (d_second, d_first)

    pin_a, pag_a = _run_pair(True)
    pin_b, pag_b = _run_pair(False)
    assert math.isclose(pin_a, pin_b, rel_tol=1e-9)
    assert math.isclose(pag_a, pag_b, rel_tol=1e-9)
    assert pag_a > pin_a * 2.0

    # 4) Production host+device Prefill Eager MoE dispatch wires EagerSplitBarrier (#10)
    def _run_eager_moe_dispatch(num_experts: int) -> dict:
        case = sdk_api.compose_case(
            MOE_MODEL_ID, hardware=LEGACY_ARCH_SKU,
            options={"dp": 1, "tp": 4, "ep": 4, "pp": 1, "layers": 4},
        )
        ex = Execution(case, CostModel(case), include_submission=True)
        shape = (4, 1, 512)
        op = OpSpec(
            op_id="l3.moe_dispatch", family="alltoall", recipe="expert_dispatch",
            phase="prefill", stage=0, layer=3, stream="main", dtype="bf16",
            link_bytes=4096, link_scope="ep", requests=("req0",),
            metadata=(("moe_ep_size", 4), ("moe_num_experts", num_experts)),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        )
        plan = IterationPlan(
            iteration=0,
            requests=(RequestView("req0", q_len=4, context=16, samples=True, phase="prefill"),),
        )
        work = IterationWork(
            plan=plan, ops=(op,), stage_count=1, mode="prefill", valid_tokens=4,
            padded_tokens=4, graph_bucket=None, recipe="default", replay_scope="none",
        )
        ex.add(work, replica=0, release_ms=0.0)
        ex.engine.advance()
        barriers = ex.summary()["eager_split_barriers"]
        assert len(barriers) == 1
        return barriers[0]

    b_single = _run_eager_moe_dispatch(num_experts=4)
    b_multi = _run_eager_moe_dispatch(num_experts=16)
    assert b_single["input_splits_bytes"] == 32 and b_single["output_splits_bytes"] == 32
    assert b_single["total_d2h_bytes"] == 256 and b_single["participating_ranks"] == [0, 1, 2, 3]
    assert b_single["explicit_device_sync"] is True and b_multi["explicit_device_sync"] is False
    # Real 4-layer raw_eager prefill at TP=EP=4 submits per-rank splits (256 B) + counts AllGather (#6)
    from mindie_sim import compose_case as top_compose, graph_costs as top_graph_costs, parse_case as top_parse, target_graphs as top_target_graphs
    ds_eager = top_parse(top_compose(MOE_MODEL_ID, LEGACY_ARCH_SKU, dp=1, tp=4, ep=4, pp=1, layers=4, raw_eager=True))
    gc_eager = top_graph_costs(
        ds_eager, top_target_graphs(ds_eager, phase="prefill", tokens=16, scope="body"),
        external_inputs={"analysis_input": 0}, include_submission=True, isolated=False,
    )
    ex_eager = gc_eager["execution"]
    eb = ex_eager["eager_split_barriers"][0]
    assert eb["total_d2h_bytes"] == 256 and eb["participating_ranks"] == [0, 1, 2, 3]
    assert eb["counts_allgather_mode"] == "lowered_collective"
    assert math.isclose(ex_eager["serviced"]["host_rc:n0:rc0:d2h"], 256.0, rel_tol=1e-9)


def test_serving_io_couples_concurrent_replicas_to_shared_root_complex():
    """Solo H2D caps at 59.5 GB/s per die while 8 concurrent replicas cap at 238.8 GB/s RC (#4)."""
    payload_bytes = 64_000_000
    base = sdk_api.compose_case(
        TINY_MODEL_ID, hardware=DUAL_DIE_LARGE_SKU,
        options={"dp": 8, "tp": 1, "pp": 1, "ranks_per_node": 8, "layers": 1},
    )
    io_cfg = {"h2d": {"fixed_bytes": payload_bytes, "bytes_per_s": 500.0e9, "latency_us": 0.0}}
    case = replace(base, runtime=replace(base.runtime, serving_io=io_cfg))
    solo_gbps = float(case.machine.host_solo_bw_gbps("h2d") or 0.0)
    rc_gbps = float(case.machine.host_root_complex_ceiling_gbps("h2d") or 0.0)
    assert math.isclose(solo_gbps, 59.5, rel_tol=1e-6) and math.isclose(rc_gbps, 238.8, rel_tol=1e-6)
    # 1) Single replica transfer must NOT exceed solo 59.5 GB/s even with 500 GB/s policy
    ex_solo = Execution(case, CostModel(case), include_submission=False, include_serving_io=True)
    done_solo = ex_solo.serving_io.transfer(ex_solo, "h2d", "r0", 0, ex_solo.engine.submit("s0"), tokens=1, requests=1)
    ex_solo.engine.advance()
    assert math.isclose(done_solo.time, payload_bytes / (solo_gbps * 1_000_000.0), rel_tol=1e-4)
    # 2) Eight concurrent replicas saturate shared 238.8 GB/s root complex
    ex = Execution(case, CostModel(case), include_submission=False, include_serving_io=True)
    ready = ex.engine.submit("all_ready")
    dones = [ex.serving_io.transfer(ex, "h2d", f"r{rep}", rep, ready, tokens=1, requests=1) for rep in range(8)]
    ex.engine.advance()
    assert math.isclose(max(ev.time for ev in dones), (8.0 * payload_bytes) / (rc_gbps * 1_000_000.0), rel_tol=1e-4)


def test_spine_single_evaluation_constructs_cost_model_and_bindings_once(monkeypatch):
    """Post-split Execution preserves CostModel.__init__==1 and ExecutionBindings.create==1."""
    case = _tiny_case()
    counts = {"cost_init": 0, "bindings_create": 0}
    orig_cost_init = CostModel.__init__
    orig_bindings_create = ExecutionBindings.create.__func__

    def _counted_cost_init(self, *args, **kwargs):
        counts["cost_init"] += 1
        return orig_cost_init(self, *args, **kwargs)

    def _counted_bindings_create(cls, *args, **kwargs):
        counts["bindings_create"] += 1
        return orig_bindings_create(cls, *args, **kwargs)

    monkeypatch.setattr(CostModel, "__init__", _counted_cost_init)
    monkeypatch.setattr(ExecutionBindings, "create", classmethod(_counted_bindings_create))

    report = predict(case, scope="body", phase="decode", tokens=1, context=16)
    dur = report.get("duration_ms") or report.get("body_ms") or report.get("step_ms")
    assert dur is not None and float(dur) > 0.0
    assert counts["cost_init"] == 1 and counts["bindings_create"] == 1
