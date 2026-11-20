"""Device STARS descriptor queue, Prefill Eager D2H barrier, and datapath specs.

[Role]
  Verify ``DatapathCapabilitySpec`` Vol-2 extensions (subtile waves, STARS
  specs, SIMT UB carveout, HBM R/W asymmetry, memory access granularity),
  device-boundary ``StarsDescriptorQueue`` FIFO + credit pool + multi-stream
  ``uncovered_stars_ns`` accounting (zero step-end ``N * gap`` floors),
  Prefill Eager ``TokenDispatcherWithAll2AllV`` ``D2H`` split-size barrier
  semantics, and post-split ``Execution`` spine invariants.

[Invariants]
  * ``StarsDescriptorQueue`` overlaps descriptor issue with concurrent compute
    (``uncovered_stars_ns`` excludes overlapped intervals), enforces exact FIFO
    and credit ``queue_depth`` backpressure, distinguishes ``descriptor_ready``
    vs ``stream_predecessor_done`` release policies, skips zero-kernel views,
    and executes inside the device boundary even when ``include_submission=False``.
  * ``EagerSplitBarrier`` transfers exact ``ep_size * split_dtype_bytes`` for
    both ``input_splits`` and ``output_splits`` over coupled per-die + shared
    ``(node, root_complex_id, "d2h")`` resources, overlaps ``non_blocking=True``
    copies with AllGather/``repeat_interleave``, and enforces an explicit
    device fence exclusively when ``num_local_experts <= 1``.
  * Single-call spine evaluation constructs ``CostModel.__init__`` once and
    ``ExecutionBindings.create`` once.

[Timeouts & Locks]
  Tier-0 pure in-memory Engine and spine checks (<3s); no locks or filesystem
  side effects.

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
NEXTGEN_ARCH_SKU = "Ascend950DT_9582"
TINY_MODEL_ID = "qwen3-32b"


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


def _tiny_case(*, stars_spec: StarsSpec | None = None):
    case = sdk_api.compose_case(
        TINY_MODEL_ID,
        hardware=LEGACY_ARCH_SKU,
        options={"dp": 1, "tp": 2, "pp": 1, "layers": 1},
    )
    if stars_spec is not None:
        new_dies = tuple(
            replace(
                die,
                dispatch_sync=replace(die.dispatch_sync, stars=stars_spec),
            )
            for die in case.machine.dies
        )
        case = replace(case, machine=replace(case.machine, dies=new_dies))
    return case


def test_datapath_capabilities_cover_vol2_fields_across_all_profiles():
    """All 16 profiles populate subtile waves, stars, HBM rates, and carveout."""
    ascend_mod = importlib.import_module("mindie_sim.backends.ascend")
    all_profiles = ascend_mod.profiles()
    assert len(all_profiles) == 16

    for sku_name in all_profiles:
        die = _make_machine(sku_name, num_devices=2).dies[0]
        dp = datapath_for_die(die)
        assert dp.subtile_wave, f"{sku_name} missing subtile_wave"
        assert dp.stars_spec is not None
        assert dp.stars_spec.queue_depth > 0
        assert dp.stars_spec.lanes >= 1
        assert dp.memory_access is not None
        assert dp.memory_access.sector_bytes == 32
        assert dp.hbm_read_bytes_per_s > 0.0
        assert dp.hbm_write_bytes_per_s > 0.0
        assert 0.0 <= dp.hbm_rw_alpha <= 1.0

    dp_legacy = datapath_for_die(_make_machine(LEGACY_ARCH_SKU).dies[0])
    dp_next = datapath_for_die(_make_machine(NEXTGEN_ARCH_SKU).dies[0])

    assert dp_legacy.simt_ub_carveout_bytes == 0
    assert dp_legacy.usable_bytes("ub", kernel="simt") == dp_legacy.usable_bytes("ub")
    assert dp_next.simt_ub_carveout_bytes == 16384
    assert dp_next.usable_bytes("ub") - dp_next.usable_bytes("ub", kernel="simt") == 16384

    route_next = dp_next.route("l0c", "ub")
    assert route_next.path.unit == "fixpipe"
    with pytest.raises(UnmodeledCapabilityError, match="datapath_route_unsupported:l0c->ub"):
        dp_legacy.route("l0c", "ub")


def test_stars_queue_covered_by_compute_has_zero_uncovered_and_no_step_floor():
    """When kernel compute exceeds issue_interval_us, STARS is naturally covered."""
    spec = StarsSpec(
        queue_scope="die",
        queue_depth=8,
        lanes=1,
        issue_interval_us=5.0,
        first_issue_us=0.0,
        release_policy="descriptor_ready",
        evidence_id="test_stars_covered",
    )
    queue = StarsDescriptorQueue(spec, die_id=0)
    engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
    ready = engine.submit("replay_ready")

    kernel_ms = 0.050
    done_events = []
    for idx in range(6):
        done, rec = queue.submit_kernel(
            engine,
            replay=0,
            stream="main",
            kernel_occurrence=idx,
            descriptor_ready=ready,
            work={"aic": kernel_ms},
        )
        assert rec is not None
        done_events.append(done)
    engine.advance()

    summary = queue.summary()
    assert summary["descriptor_count"] == 6
    # Only the first 5 us descriptor before kernel 0 starts is uncovered;
    # descriptors 1..5 issue overlapped behind compute!
    assert summary["uncovered_stars_ns"] == 5_000
    assert math.isclose(done_events[-1].time, 0.005 + 6 * kernel_ms, rel_tol=1e-9)


def test_stars_queue_exposed_on_tiny_kernels_and_charges_first_dispatch_once():
    """Tiny kernels serialize on STARS FIFO lane; first_issue_us charges once/replay."""
    spec = StarsSpec(
        queue_scope="die",
        queue_depth=16,
        lanes=1,
        issue_interval_us=10.0,
        first_issue_us=15.0,
        release_policy="descriptor_ready",
        evidence_id="test_stars_tiny",
    )
    queue = StarsDescriptorQueue(spec, die_id=0)
    engine = Engine([Resource("aic", 1.0, "ms")])
    ready = engine.submit("replay_ready")

    done_events = []
    for idx in range(4):
        done, _ = queue.submit_kernel(
            engine,
            replay=0,
            stream="main",
            kernel_occurrence=idx,
            descriptor_ready=ready,
            work={"aic": 0.001},
        )
        done_events.append(done)
    engine.advance()

    assert math.isclose(done_events[0].time, 0.026, rel_tol=1e-9)
    assert math.isclose(done_events[-1].time, 0.056, rel_tol=1e-9)
    summary = queue.summary()
    assert summary["descriptor_count"] == 4
    assert summary["descriptor_union_ns"] == 55_000
    assert summary["uncovered_stars_ns"] == 52_000


def test_stars_release_policies_and_credit_depth_backpressure():
    """Compare descriptor_ready vs stream_predecessor_done and credit depth limit."""
    def _run_policy(policy: str, *, queue_depth: int, count: int) -> list[tuple[float, float]]:
        spec = StarsSpec(
            queue_scope="die",
            queue_depth=queue_depth,
            lanes=1,
            issue_interval_us=10.0,
            first_issue_us=0.0,
            release_policy=policy,
            evidence_id=f"test_stars_{policy}",
        )
        queue = StarsDescriptorQueue(spec, die_id=0)
        engine = Engine([Resource("aic", 1.0, "ms", "fifo")])
        ready = engine.submit("ready")
        records = []
        for idx in range(count):
            _, rec = queue.submit_kernel(
                engine,
                replay=0,
                stream="s0",
                kernel_occurrence=idx,
                descriptor_ready=ready,
                work={"aic": 0.050},
            )
            assert rec is not None
            records.append(rec)
        engine.advance()
        return [rec.issue_interval_ms for rec in records]

    ready_intervals = _run_policy("descriptor_ready", queue_depth=4, count=3)
    assert math.isclose(ready_intervals[0][0], 0.000, abs_tol=1e-9)
    assert math.isclose(ready_intervals[1][0], 0.010, abs_tol=1e-9)
    assert math.isclose(ready_intervals[2][0], 0.020, abs_tol=1e-9)

    pred_intervals = _run_policy("stream_predecessor_done", queue_depth=4, count=3)
    assert math.isclose(pred_intervals[0][0], 0.000, abs_tol=1e-9)
    assert math.isclose(pred_intervals[1][0], 0.060, abs_tol=1e-9)
    assert math.isclose(pred_intervals[2][0], 0.120, abs_tol=1e-9)

    # Credit depth=2 backpressure holds descriptor 2 until kernel 0 completes at 60 us
    bp_intervals = _run_policy("descriptor_ready", queue_depth=2, count=3)
    assert math.isclose(bp_intervals[0][0], 0.000, abs_tol=1e-9)
    assert math.isclose(bp_intervals[1][0], 0.010, abs_tol=1e-9)
    assert math.isclose(bp_intervals[2][0], 0.060, abs_tol=1e-9)


def test_stars_multistream_interval_union_and_zero_kernel_views():
    """Three concurrent streams compute exact wall union; views consume 0 descriptors."""
    spec = StarsSpec(
        queue_scope="die",
        queue_depth=8,
        lanes=1,
        issue_interval_us=5.0,
        first_issue_us=0.0,
        release_policy="descriptor_ready",
        evidence_id="test_stars_multistream",
    )
    queue = StarsDescriptorQueue(spec, die_id=0)
    engine = Engine([
        Resource("s0_unit", 1.0, "ms"),
        Resource("s1_unit", 1.0, "ms"),
        Resource("s2_unit", 1.0, "ms"),
    ])
    ready = engine.submit("ready")

    view_op = SimpleNamespace(family="reshape", metadata=())
    alias_op = SimpleNamespace(family="matmul", metadata=(("zero_kernel_view", True),))
    real_op = SimpleNamespace(family="matmul", metadata=())
    assert is_zero_kernel_view(view_op) is True
    assert is_zero_kernel_view(alias_op) is True
    assert is_zero_kernel_view(real_op) is False

    _, rec_view = queue.submit_kernel(
        engine, replay=0, stream="s0", kernel_occurrence=0,
        descriptor_ready=ready, work={"s0_unit": 0.0}, op=view_op,
    )
    assert rec_view is None

    for idx, (stream, unit) in enumerate((("s0", "s0_unit"), ("s1", "s1_unit"), ("s2", "s2_unit"))):
        _, rec = queue.submit_kernel(
            engine, replay=0, stream=stream, kernel_occurrence=idx + 1,
            descriptor_ready=ready, work={unit: 0.050}, op=real_op,
        )
        assert rec is not None
    engine.advance()

    summary = queue.summary()
    assert summary["descriptor_count"] == 3
    assert summary["descriptor_union_ns"] == 15_000
    assert summary["compute_union_ns"] == 60_000
    assert summary["uncovered_stars_ns"] == 5_000
    assert compute_uncovered_stars_ns([(0.0, 0.015)], [(0.005, 0.055), (0.010, 0.065)]) == 5_000


def test_execution_runs_device_stars_even_when_host_submission_disabled():
    """include_submission=False removes host wrapper while retaining device STARS."""
    stars_spec = StarsSpec(
        queue_scope="die",
        queue_depth=32,
        lanes=1,
        issue_interval_us=8.0,
        first_issue_us=0.0,
        release_policy="descriptor_ready",
        evidence_id="test_stars_exec",
    )
    case = _tiny_case(stars_spec=stars_spec)
    cost = CostModel(case)
    ex = Execution(case, cost, include_submission=False)

    shape = (8, 16, 64)
    ops = (
        OpSpec(
            op_id="op_view", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            requests=("req0",), metadata=(("zero_kernel_view", True),),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
        OpSpec(
            op_id="op_k0", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            requests=("req0",),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
        OpSpec(
            op_id="op_k1", family="rotary", recipe="rotary_position_embedding",
            phase="decode", stage=0, layer=0, stream="main", dtype="bf16",
            deps=("op_k0",), requests=("req0",),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        ),
    )
    plan = IterationPlan(
        iteration=0,
        requests=(RequestView("req0", q_len=1, context=16, samples=True, phase="decode"),),
    )
    work = IterationWork(
        plan=plan, ops=ops, stage_count=1, mode="decode", valid_tokens=1, padded_tokens=1,
        graph_bucket=1, recipe="default", replay_scope="full",
    )
    ex.add(work, replica=0, release_ms=0.0)
    ex.engine.advance()

    summary = ex.summary()
    assert summary["submission"]["included"] is False
    assert summary["submission"]["host_service_ms"] == 0
    assert "stars" in summary
    stage_stars = summary["stars"]["r0:s0"]
    assert stage_stars["descriptor_count"] == 2
    assert stage_stars["descriptor_union_ns"] == 16_000


def test_eager_split_barrier_d2h_bytes_overlap_and_single_expert_device_fence():
    """Prefill Eager All2AllV D2H copies overlap when >1 local expert and fence when <=1."""
    machine = _make_machine(LEGACY_ARCH_SKU, num_devices=4, ranks_per_node=4)
    die0 = machine.dies[0]
    rc_d2h = RootComplexKey(0, die0.root_complex_id, "d2h").resource_name()

    spec_multi = SplitBarrierSpec(
        ep_size=16, num_local_experts=4, split_dtype_bytes=8,
        pinned=True, non_blocking=True, host_consume_us=5.0,
        histogram_ms=0.020, repeat_interleave_ms=0.030, allgather_ms=0.040,
    )
    assert spec_multi.input_splits_bytes == 128
    assert spec_multi.output_splits_bytes == 128
    assert spec_multi.requires_explicit_device_sync is False

    eng_multi = Engine([Resource("other_stream", 1.0, "ms")])
    ready_m = eng_multi.submit("ready")
    slow_independent_m = eng_multi.submit(
        "independent_work", {"other_stream": 0.500}, after=(ready_m,)
    )
    res_m = EagerSplitBarrier.submit(
        eng_multi, machine, die0, spec_multi, ready_m,
        independent_device_events=(slow_independent_m,),
    )
    eng_multi.advance()

    assert res_m.explicit_device_sync is None
    assert rc_d2h in res_m.d2h_resources
    assert res_m.dispatch_ready.time < 0.200
    assert res_m.dispatch_ready.time >= res_m.host_consume.time
    assert res_m.dispatch_ready.time >= res_m.device_branch_done.time
    assert math.isclose(res_m.device_branch_done.time, 0.050, rel_tol=1e-9)

    spec_single = SplitBarrierSpec(
        ep_size=16, num_local_experts=1, split_dtype_bytes=8,
        pinned=True, non_blocking=True, host_consume_us=5.0,
        histogram_ms=0.020, allgather_ms=0.040, sync_overhead_us=10.0,
    )
    assert spec_single.requires_explicit_device_sync is True

    eng_single = Engine([Resource("other_stream", 1.0, "ms")])
    ready_s = eng_single.submit("ready")
    slow_independent_s = eng_single.submit(
        "independent_work", {"other_stream": 0.500}, after=(ready_s,)
    )
    res_s = EagerSplitBarrier.submit(
        eng_single, machine, die0, spec_single, ready_s,
        independent_device_events=(slow_independent_s,),
    )
    eng_single.advance()

    assert res_s.explicit_device_sync is not None
    assert math.isclose(res_s.explicit_device_sync.time, 0.510, rel_tol=1e-9)
    assert math.isclose(res_s.dispatch_ready.time, 0.515, rel_tol=1e-9)


def test_eager_split_barrier_pinned_vs_pageable_and_root_complex_contention():
    """Pageable D2H is slower than pinned; concurrent dies contend on shared RC."""
    raw = ascend({"sku": LEGACY_ARCH_SKU, "num_devices": 8, "ranks_per_node": 8})
    raw["transfer_channels"]["d2h"]["pageable_uni_bw_gbps"] = 8.0
    machine = Hardware.from_mapping(raw)

    spec_pinned = SplitBarrierSpec(
        ep_size=131072, num_local_experts=2, split_dtype_bytes=8, pinned=True
    )
    spec_pageable = SplitBarrierSpec(
        ep_size=131072, num_local_experts=2, split_dtype_bytes=8, pinned=False
    )

    eng_pin = Engine([])
    r_pin = eng_pin.submit("r")
    res_pin = EagerSplitBarrier.submit(eng_pin, machine, machine.dies[0], spec_pinned, r_pin)
    eng_pin.advance()

    eng_pag = Engine([])
    r_pag = eng_pag.submit("r")
    res_pag = EagerSplitBarrier.submit(eng_pag, machine, machine.dies[0], spec_pageable, r_pag)
    eng_pag.advance()

    assert res_pag.input_splits_d2h.time > res_pin.input_splits_d2h.time * 2.0

    eng_rc = Engine([])
    r_rc = eng_rc.submit("r")
    results = [
        EagerSplitBarrier.submit(
            eng_rc, machine, die, spec_pinned, r_rc, name=f"d{die.rank}"
        )
        for die in machine.dies
    ]
    eng_rc.advance()
    assert max(r.input_splits_d2h.time for r in results) > res_pin.input_splits_d2h.time


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
    assert counts["cost_init"] == 1
    assert counts["bindings_create"] == 1
