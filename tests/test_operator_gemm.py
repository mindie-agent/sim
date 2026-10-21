"""Arithmetic fixtures for a declared program; none are hardware measurements."""
from __future__ import annotations

from dataclasses import asdict, replace
import math

import pytest

from mindie_sim.a3_ops import _gemm_op
from mindie_sim.a3_types import DieResources, MachineSpec
from mindie_sim.a3_resource_engine import ResourceEngine, ResourceSpec
from mindie_sim.operator_gemm import GemmProfile, GemmProvider
from mindie_sim.operator_plan import (
    EstimateContext, KernelPlan, evaluate_kernel_plan, register_kernel_plan,
)


def _profile(**changes):
    base = GemmProfile(
        core_count=1, tile_m=2, tile_n=3, tile_k=2, l1_k=4,
        l1_bytes=4096, l0a_bytes=4096, l0b_bytes=4096, l0c_bytes=4096,
        l2_bytes_per_s_per_core=1e6, mte2_bytes_per_s_per_core=1e6,
        mte1_a_bytes_per_s_per_core=1e6, mte1_b_bytes_per_s_per_core=1e6,
        mte3_bytes_per_s_per_core=1e6, fix_bytes_per_s_per_core=1e6,
        source="synthetic_fixture_not_npu_measurement",
    )
    return replace(base, **changes)


def _machine(profile, **changes):
    # Compute is exactly 1000 FLOPs/ms per core; shared transfers are generous
    # so this fixture isolates the declared per-core service recurrences.
    die = DieResources(
        rank=0, device_id=0, package_id=0, die_in_package=0,
        hbm_bytes=2**30, cube_flops_per_s=profile.core_count * 1e6,
        vector_ops_per_s=1e9, hbm_bw_bytes_per_s=1e12,
        mte_bw_bytes_per_s=1e12, mte1_bw_bytes_per_s=1e12,
        launch_us=0.0, aic_count=profile.core_count,
        l1_bytes=4096, l0a_bytes=4096, l0b_bytes=4096, l0c_bytes=4096,
        source="synthetic_fixture",
    )
    return MachineSpec(
        name="synthetic", arch_family="fixture", sku="",
        dies_per_package=1, ranks_per_node=1,
        dies=(replace(die, **changes),), links=(),
    )


def _op(m=2, n=3, k=8, **changes):
    result = _gemm_op(
        op_id="gemm", deps=(), stage=0, layer=0, requests=("r",),
        m=m, n=n, k=k, dtype="bf16", family="gemm", phase="prefill",
        recipe="synthetic_fixture",
    )
    return replace(result, **changes)


def _estimate(shape=(2, 3, 8), profile=None, **op_changes):
    p = profile or _profile()
    machine = _machine(p)
    value = GemmProvider(p).estimate(_op(*shape, **op_changes), machine, EstimateContext())
    assert value.plan is not None, value.unsupported_reason
    return value.plan, machine


def _recurrence_oracle(m, n, k, p):
    """Numerical command recurrence without constructing/reading the plan DAG.

    All rates in this fixture are 1000 bytes/ms, and per-core compute is 1000
    FLOPs/ms.  State tracks engine completion and independently reused slots.
    """
    completion = []
    rows, cols = math.ceil(m / p.tile_m), math.ceil(n / p.tile_n)
    for core in range(min(p.core_count, rows * cols)):
        dma = unpack = cube = store = 0.0
        l1_free = [0.0] * p.l1_buffers
        l0_free = [0.0] * p.l0_buffers
        c_free = [0.0] * p.c_buffers
        macro_id = micro_id = local_tile = 0
        for tile in range(core, rows * cols, p.core_count):
            row, col = divmod(tile, cols)
            mm, nn = min(p.tile_m, m-row*p.tile_m), min(p.tile_n, n-col*p.tile_n)
            c_slot = local_tile % p.c_buffers
            accumulator_ready = c_free[c_slot]
            for offset in range(0, k, p.l1_k):
                width = min(p.l1_k, k-offset)
                l1_slot = macro_id % p.l1_buffers
                dma = max(dma, l1_free[l1_slot]) + (mm+nn)*width*2/1000
                for sub in range(0, width, p.tile_k):
                    kk = min(p.tile_k, width-sub)
                    slot = micro_id % p.l0_buffers
                    unpack = max(unpack, dma, l0_free[slot]) + (mm+nn)*kk*2/1000
                    cube = max(cube, unpack, accumulator_ready) + 2*p.tile_m*p.tile_n*p.tile_k/1000
                    l0_free[slot] = cube
                    micro_id += 1
                l1_free[l1_slot] = unpack
                macro_id += 1
            store = max(store, cube) + mm*nn*2/1000
            c_free[c_slot] = store
            local_tile += 1
        completion.append(store)
    return max(completion)


@pytest.mark.parametrize("shape,cores", [
    ((2, 3, 1), 1),  # short K still issues one complete MMAD microtile
    ((2, 3, 2), 1),
    ((2, 3, 8), 1),
    ((3, 5, 5), 1),  # M, N, K tails
    ((6, 3, 9), 2),  # uneven cyclic output-tile ownership
    ((2, 3, 7), 4),  # fewer active cores than the machine owns
])
def test_engine_elapsed_matches_independent_finite_slot_recurrence(shape, cores):
    p = _profile(core_count=cores)
    plan, machine = _estimate(shape, p)
    actual = evaluate_kernel_plan(plan, machine, EstimateContext())
    assert actual["elapsed_ms"] == pytest.approx(_recurrence_oracle(*shape, p), abs=1e-10)
    assert plan.includes_dispatch is False
    assert plan.provenance["source_class"] == "uncalibrated_analytic"


def test_short_k_hand_calculation_has_additive_dependent_stages():
    plan, machine = _estimate((2, 3, 1))
    # 10 B load + 10 B unpack + 24 FLOPs full microtile + 12 B store.
    assert evaluate_kernel_plan(plan, machine, EstimateContext())["elapsed_ms"] == pytest.approx(.056)


def test_issued_compute_and_hierarchy_traffic_are_conserved_with_tails():
    plan, machine = _estimate((3, 5, 5), _profile(core_count=2))
    totals = plan.resource_totals()
    cert = plan.buffer_certificate
    # A is reloaded once per output-column tile; B once per output-row tile.
    input_bytes = 3*5*2*math.ceil(5/3) + 5*5*2*math.ceil(3/2)
    output_bytes = 3*5*2
    issued = math.ceil(3/2)*math.ceil(5/3)*math.ceil(5/2)*(2*2*3*2)
    assert totals == {
        "cube": issued, "hbm": input_bytes+output_bytes,
        "mte2": input_bytes, "mte1": input_bytes, "mte3": output_bytes,
    }
    assert cert["logical_flops"] == 2*3*5*5
    assert cert["issued_flops"] == issued
    assert cert["traffic_bytes"] == {
        "hbm_read": input_bytes, "hbm_write": output_bytes,
        "l2_to_l1": input_bytes, "l1_to_l0": input_bytes,
        "l0c_to_output": output_bytes,
    }
    result = evaluate_kernel_plan(plan, machine, EstimateContext())
    for alias in ("cube", "hbm", "mte1", "mte2", "mte3"):
        assert result["resource_serviced"][f"{alias}@s0"] == pytest.approx(totals[alias])


def test_declared_double_buffers_enable_real_overlap_and_slots_block_reuse():
    results = {}
    for key, p in {
        "double": _profile(),
        "single_l1": _profile(l1_buffers=1),
        "single_l0": _profile(l0_buffers=1),
    }.items():
        plan, machine = _estimate(profile=p)
        results[key] = evaluate_kernel_plan(plan, machine, EstimateContext())["elapsed_ms"]
        assert results[key] == pytest.approx(_recurrence_oracle(2, 3, 8, p))
    assert results["double"] == pytest.approx(.168)
    assert results["single_l1"] > results["double"]
    assert results["single_l0"] > results["double"]


def test_core_assignment_and_tail_core_are_explicit_not_a_fractional_wave():
    p = _profile(core_count=2)
    plan, machine = _estimate((6, 3, 5), p)
    assert plan.buffer_certificate["tile_owners"] == [0, 1, 0]
    assert plan.buffer_certificate["tiles_per_core"] == [2, 1]
    assert evaluate_kernel_plan(plan, machine, EstimateContext())["elapsed_ms"] == pytest.approx(
        _recurrence_oracle(6, 3, 5, p),
    )
    # Increasing the declared machine peak cannot make one tile use all cores.
    one, machine_one = _estimate((2, 3, 8), _profile(core_count=1))
    few, machine_few = _estimate((2, 3, 8), _profile(core_count=4))
    assert evaluate_kernel_plan(one, machine_one, EstimateContext())["elapsed_ms"] == pytest.approx(
        evaluate_kernel_plan(few, machine_few, EstimateContext())["elapsed_ms"],
    )


@pytest.mark.parametrize("field", ["l1_bytes", "l0a_bytes", "l0b_bytes", "l0c_bytes"])
def test_insufficient_profile_capacity_returns_explicit_fallback(field):
    p = _profile(**{field: 1})
    value = GemmProvider(p).estimate(_op(), _machine(p), EstimateContext())
    assert value.plan is None
    assert value.unsupported_reason == f"gemm_buffer_capacity_exceeded:{field}"


@pytest.mark.parametrize("changes,reason", [
    ({"dtype": "int4"}, "gemm_requires_bf16_dense"),
    ({"layout": "fractal_nz"}, "gemm_layout_unsupported"),
    ({"metadata": (("groups", 4),)}, "gemm_grouped_or_batched_unsupported"),
    ({"metadata": (("ragged", True),)}, "gemm_features_unsupported"),
    ({"metadata": (("weight_dtype", "int4"),)}, "gemm_requires_bf16_dense"),
    ({"metadata": (("transpose_b", True),)}, "gemm_features_unsupported"),
    ({"metadata": (("quant_type", "W8A8"),)}, "gemm_features_unsupported"),
    ({"family": "moe_gate_up"}, "gemm_family_unsupported"),
    ({"launch_shape": (2, 3, 4, 5)}, "gemm_shape_requires_positive_m_n_k"),
])
def test_unsupported_semantics_do_not_silently_price_a_different_kernel(changes, reason):
    p = _profile()
    value = GemmProvider(p).estimate(_op(**changes), _machine(p), EstimateContext())
    assert value.plan is None
    assert value.unsupported_reason == reason


def test_missing_profile_machine_mismatch_and_bounded_plan_fallback():
    p = _profile()
    assert GemmProvider().estimate(_op(), _machine(p), EstimateContext()).unsupported_reason == "gemm_profile_missing"
    assert GemmProvider(p).estimate(_op(), _machine(p, aic_count=2), EstimateContext()).unsupported_reason == "gemm_profile_core_count_mismatch"
    small = replace(p, max_stages=2)
    assert GemmProvider(small).estimate(_op(), _machine(p), EstimateContext()).unsupported_reason == "gemm_plan_stage_limit_exceeded"
    assert GemmProvider(p).estimate(_op(), _machine(p, l0c_bytes=64), EstimateContext()).unsupported_reason == "gemm_profile_exceeds_machine:l0c_bytes"


def test_context_quantization_cannot_silently_use_bf16_dense_plan():
    p = _profile()
    provider = GemmProvider(p)
    quantized = provider.estimate(_op(), _machine(p), EstimateContext(quantization="w4a8"))
    assert quantized.plan is None
    assert quantized.unsupported_reason == "gemm_quantization_unsupported"
    for mode in ("", "none", "unquantized"):
        assert provider.estimate(_op(), _machine(p), EstimateContext(quantization=mode)).plan is not None


@pytest.mark.parametrize("architecture", ["A2", "A3", "A5"])
def test_ascend_bf16_requires_whole_declared_matrix_fragments(architecture):
    p = _profile()
    machine = replace(_machine(p), arch_family=architecture)
    result = GemmProvider(p).estimate(_op(), machine, EstimateContext())
    assert result.plan is None
    assert result.unsupported_reason == "gemm_bf16_fragment_alignment_required"
    valid = _profile(tile_m=16, tile_n=16, tile_k=16, l1_k=32)
    machine = replace(_machine(valid), arch_family=architecture)
    accepted = GemmProvider(valid).estimate(_op(13, 17, 23), machine, EstimateContext())
    assert accepted.plan is not None  # valid fragment tiles, masked M/N/K tails
    assert accepted.plan.buffer_certificate["fragment_shape_guard"] == [16, 16, 16]
    assert "bf16_16_lane_fragment_shape_guard_not_kernel_certification" in accepted.plan.provenance["assumptions"]


def test_shared_machine_bandwidth_not_only_profile_controls_service():
    p = _profile()
    plan, fast_machine = _estimate()
    fast = evaluate_kernel_plan(plan, fast_machine, EstimateContext())["elapsed_ms"]
    slow_machine = _machine(p, hbm_bw_bytes_per_s=1e3)
    slow = evaluate_kernel_plan(plan, slow_machine, EstimateContext())["elapsed_ms"]
    assert slow > fast
    assert slow >= plan.resource_totals()["hbm"] / 1.0  # bytes / (bytes/ms)


def test_cube_stage_rate_limit_holds_physical_service_during_other_work():
    p = _profile(core_count=10, tile_m=5, tile_n=5)
    machine = _machine(p, cube_flops_per_s=100_000)
    estimate = GemmProvider(p).estimate(_op(5, 5, 2), machine, EstimateContext())
    stage = next(s for s in estimate.plan.stages if s.name.endswith(".mmad"))
    assert dict(stage.demands) == {"cube": 100}
    assert dict(stage.rate_limits) == {"cube": 10}  # FLOPs/ms per one core
    assert stage.minimum_duration_ms == 10
    isolated_compute = KernelPlan(
        "fixture", "1", (replace(stage, deps=()),), {"source": "synthetic_fixture"},
    )
    engine = ResourceEngine({"cube": ResourceSpec("cube", 100)})
    register_kernel_plan(engine, isolated_compute, {"cube": "cube"}, task_id="a")
    engine.add_task("b", {"cube": 900}, release_ms=1)
    result = engine.run()
    assert result["completed_at_ms"]["a"] == pytest.approx(10)
    # A consumes 10/ms throughout 0..10.  B gets 90/ms during 1..10, then
    # the remaining 90 FLOPs at 100/ms; an independent floor alone gave 10.
    assert result["completed_at_ms"]["b"] == pytest.approx(10.9)
    assert result["serviced"]["cube"] == pytest.approx(1000)


def test_copy_limits_bound_local_pipes_without_reserving_the_shared_hbm_bus():
    p = _profile(mte1_a_bytes_per_s_per_core=2e6)
    plan, _ = _estimate((2, 3, 1), p)
    for stage in plan.stages:
        assert "hbm" not in dict(stage.rate_limits)
        if stage.name.endswith(".mte2"):
            assert dict(stage.rate_limits) == {"mte2": 1000}
        elif stage.name.endswith(".mte1"):
            # 4 B A at 2000 B/ms plus 6 B B at 1000 B/ms.
            assert dict(stage.rate_limits) == {"mte1": pytest.approx(1250)}
        elif stage.name.endswith(".fix_store"):
            assert dict(stage.rate_limits) == {"mte3": 1000}


def test_profile_mapping_is_strict_and_source_is_required():
    p = _profile()
    assert GemmProfile.from_mapping(asdict(p)) == p
    for changes in ({"source": ""}, {"core_count": 1.5}, {"tile_k": 0},
                    {"l1_k": 3}, {"l0_buffers": 3}, {"fix_bytes_per_s_per_core": float("nan")}):
        with pytest.raises(ValueError, match="gemm_profile"):
            replace(p, **changes)
    with pytest.raises(ValueError, match="required_fields"):
        GemmProfile.from_mapping({"source": "fixture"})
    with pytest.raises(ValueError, match="unknown_field"):
        GemmProfile.from_mapping({**asdict(p), "invented": 1})
