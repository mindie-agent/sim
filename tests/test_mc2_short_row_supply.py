"""A5 typed PIPE_S/PIPE_V/UB supplies for MC2 short rows."""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.instructions import (
    cal_token_send_expert_work, get_sff_value_u64_i64_work,
)
from mindie_sim.prediction.mc2_phase import build, contract
from mindie_sim.prediction.mechanism import Geometry, Work
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec

RAW = dict(
    kind="dispatch", width=5120, dtype="bf16", payload_dtype="fp8_e4m3",
    dispatch_quant="mx", source_tokens=1, top_k=6, source_assignments=6,
    received_assignments=6, experts=384, ep=8,
)


def _device(sku="Ascend950DT_9582"):
    return Hardware.from_mapping({
        "backend": "ascend", "sku": sku, "num_devices": 1, "device": {},
    }).dies[0]


def _engine(d, owners=64):
    rates = {name: float(rate) / 1000.0
             for name, rate in d.rates}
    units = {name: int(d.units(name)) for name in rates}
    return Engine([Resource(name, rate, units=units[name])
                   for name, rate in rates.items()]), rates, units


def _run_single(work, *, owner=0, owners=64, sku="Ascend950DT_9582"):
    d = _device(sku)
    engine, rates, units = _engine(d, owners)
    context = RecipeContext(
        None, {}, {name: name for name in rates}, rates, owners,
        resource_units=units)
    ready = engine.submit("ready", release=0)
    demand = context.bind(work)
    placement = context.owned(owner, owners, work.resources)
    event = Stage("typed", demand, placement=placement)
    done = KernelProgram((event,)).start(engine, "op", ready)
    engine.advance(until=(done,))
    return done.time - ready.time, engine


def test_a5_shared_profile_and_typed_get_sff_resource():
    for sku in ("Ascend950DT_9581", "Ascend950DT_9582"):
        d = _device(sku)
        rates = dict(d.rates)
        assert rates["pipe_v_lanes"] == 6758400000000.0
        assert rates["pipe_v_issue"] == 211200000000.0
        assert rates["ub_read_bytes"] == rates["ub_write_bytes"] == 27033600000000.0
        assert rates["pipe_s_issue"] == 105600000000.0
        assert d.units("pipe_s_issue") == d.units("pipe_v_lanes") == 64
        mode = dict(d.service_modes)["get_sff_value_u64_i64"]
        assert mode == ("pipe_s_issue", 105600000000.0)
        # No per-op independent capacity and no generic scalar_store reuse.
        for forbidden in ("vec_duplicate_ops", "vec_sub_ops", "fp32_abs_ops",
                          "vec_mins_ops", "fp32_reduce_sum_ops", "pipe_barrier",
                          "scalar_alu_issue", "scalar_ub_access",
                          "reduce_sum_issue"):
            assert forbidden not in rates
        assert rates["scalar_store_issue"] == 5657458563.535912


@pytest.mark.parametrize("calls,expected_ns", ((17, 10.303030303), (30, 18.181818182)))
def test_get_sff_typed_pipe_s_clock(calls, expected_ns):
    d = _device()
    elapsed, engine = _run_single(
        get_sff_value_u64_i64_work(calls), owner=0)
    assert float(elapsed) * 1e6 == pytest.approx(expected_ns, rel=1e-9)
    assert engine.serviced["pipe_s_issue"] == calls


def test_get_sff_pipe_s_is_shared_not_copied_per_owner():
    # One active owner's 17 calls = 10.303 ns.  A second owner pinned to the
    # same physical PIPE_S slot shares the one 1.65 GHz issue pipe, while an
    # independent rank engine does not.
    d = _device()
    engine, rates, units = _engine(d)
    context = RecipeContext(
        None, {}, {name: name for name in rates}, rates, 64,
        resource_units=units)
    ready = engine.submit("ready", release=0)
    work = get_sff_value_u64_i64_work(17)
    one = context.bind(work)
    two = context.bind(work)
    same_owner = context.owned(0, 64, work.resources)
    e1 = Stage("one", one, placement=same_owner)
    e2 = Stage("two", two, placement=same_owner)
    done = KernelProgram((e1, e2)).start(engine, "two", ready)
    engine.advance(until=(done,))
    shared = float(done.time - ready.time) * 1e6
    assert shared == pytest.approx(20.606060606, rel=1e-9)
    left, _ = _run_single(get_sff_value_u64_i64_work(17), owner=0)
    right, _ = _run_single(get_sff_value_u64_i64_work(17), owner=0)
    assert left == right


def test_cal_token_shared_pipe_v_and_ub_work_no_per_op_or_reduce_claim():
    for count, lanes, read in ((6, 36, 24), (16, 96, 64)):
        work = cal_token_send_expert_work(count)
        assert work.resources == {
            "pipe_v_lanes": lanes,
            "pipe_v_issue": 4,
            "ub_read_bytes": read,
            "ub_write_bytes": 4,
        }
        assert "reduce_sum_issue" not in work.resources
        assert "scalar_ub_access" not in work.resources
    # The same shared PIPE_V lane/issue ports, not per-op resources.
    d = _device()
    one, _ = _run_single(cal_token_send_expert_work(6), owner=0)
    two, _ = _run_single(cal_token_send_expert_work(16), owner=0)
    # Both short rows emit the same four PIPE_V instructions; the shared
    # instruction port, not the 6/16 active-lane difference, bounds them.
    assert 0 < one == two


def test_full_source_counts_and_shared_conversion():
    op = OpSpec("p", "mc2_protocol", "decode", 0, 0, (), (), (), (), (),
                metadata=(("mc2_phase", RAW),))
    geometry = Geometry(64, {name: 64 for name in (
        "vector", "mte2_aiv", "mte3", "hbm", "pipe_barrier",
        "pipe_v_lanes", "pipe_v_issue", "ub_read_bytes", "ub_write_bytes")})
    total = defaultdict(float)
    for phase in contract(RAW, 64).phases:
        plan = build(SimpleNamespace(metadata={
            "mc2_phase": {**RAW, "phase": phase.name}}), geometry)
        for owner in plan.metadata["primitive_services"]["owners"].values():
            for item in owner.values():
                for role, amount in item.get("counts", {}).items():
                    total[role] += amount
    assert total["scalar_ub_access"] == 1694
    assert total["scalar_alu_issue"] == 384
    assert total["pipe_barrier"] == 1559
    # The current MOE15 procedure maps the five A5 REG op counts to the one
    # shared PIPE_V lane pool.  Sub<int32> costs two 64-lane slots/element.
    assert total["pipe_v_lanes"] == 13824
    assert total["pipe_v_issue"] == 1536
    assert total["ub_read_bytes"] == 9216
    assert total["ub_write_bytes"] == 1536
    # Source-only typed terms: no per-op independent resources, and the
    # software ReduceSum issue count has no authoritative lowering.
    for forbidden in ("vec_duplicate_ops", "vec_sub_ops", "fp32_abs_ops",
                      "vec_mins_ops", "fp32_reduce_sum_ops"):
        assert total[forbidden] == 0
    assert total["get_sff_value_u64_i64"] == 0  # dispatch row, not combine flags
    assert total["reduce_sum_issue"] == 1920   # present source count, no SAT rate
