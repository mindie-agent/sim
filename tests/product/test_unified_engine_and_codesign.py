"""Unified Engine single-spine optimization, 4-stage receipts, negative controls & co-design.

[Role] Lock end-to-end invariants of the unified single-Engine execution and
  optimization spine:
  - Retirement of ``target_schedule.py``, ``dsl_solver.py``, ``causal_tail``
    shadow schedulers, and the legacy ``kernel_synthesis`` IR body;
  - Strict separation between ``comparison_policy="simulation_only"`` and
    ``stack_track="target"`` 4-stage ``EngineStepReceipt`` execution on ONE
    ``Engine`` + ONE ``PrimitiveTable`` (including ``{"activation": "silu"}``
    handling without ``float('silu')`` crashes);
  - Negative controls blocking cross-precision ``FLOAT + MXFP4`` merges and
    hoisted prologue physical-key invariance;
  - Typed ``CoDesignOpportunity`` emission (``MegaFusionBound``,
    ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``), symbolic equations for
    ``aclnnMegaMoe`` and ``aclnnMoeGatingTopKSoftmaxV2``, OSS Ascend backends
    (**``TileLang / AscendC / Catlass``**), and ``ExposedObserver``
    $O((N+G)\\log(N+G))$ sweep-line parity & 93-layer speed.
[Invariants] Pure in-memory white-box Engine and synthesis checks; zero local
  host writes; zero silent fallbacks; zero heuristic multipliers.
[Timeouts & Locks] Tier 1: executes compact multi-layer models on ``Engine``
  and a 93-layer synthetic timeline sweep; no locks or network access.
[Side-Effect Policy] Read-only over repository assets; no persistent writes.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import time

import pytest

from _support.axes import GEN5_DT_SKU
from mindie_sim import structure_implementations as si_mod
from mindie_sim.compose import compose_case
from mindie_sim.export import bundle as bundle_mod
from mindie_sim.export.models import STRUCTURE_BACKEND_MAP
from mindie_sim.optimization import causal_tail as causal_tail_mod
from mindie_sim.optimization import kernel_synthesis as ks_mod
from mindie_sim.optimization.codesign import (
    collect_codesign_opportunities,
    evaluate_macro_wall_trigger,
)
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.program.physical_key import HOISTED_PROLOGUE_PREFIX, physical_key
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    ObservedServiceGrant,
    attribute_stream0_intervals,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.rewrite.algebra import AlgebraicTrait, _resolve_symbolic_kind
from mindie_sim.rewrite.apply import has_illegal_cross_precision_merge
from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI
from mindie_sim.sweep import pd as pd_mod
from mindie_sim.templates.contracts import (
    KernelSynthesisSpec,
    _format_megamoe_equation,
    _format_moe_gating_equation,
)

pytestmark = pytest.mark.tier1

TEST_COMPACT_LAYERS = 2
SYNTHETIC_DEPTH_LAYERS = 93
SYNTHETIC_OPS_PER_LAYER = 26
SWEEP_LINE_MAX_SECONDS = 0.5


def _reference_bruteforce_stream0(
    intervals: tuple[ObservedInterval, ...],
    grants: tuple[ObservedServiceGrant, ...],
) -> tuple[dict[str, int], int]:
    """Reference O(C * (N + G)) active-set scan verifying O((N+G) log(N+G)) sweep-line parity."""
    from collections import defaultdict
    from mindie_sim.prediction.runtime.exposed_observer import (
        _attribute_live_slice,
        _trace_stream0_witness_set,
    )

    coords = sorted({
        t
        for item in (*intervals, *grants)
        for t in (item.start_ns, item.end_ns)
        if item.end_ns > item.start_ns
    })
    witness_set = _trace_stream0_witness_set(intervals, None)
    exposed: dict[str, int] = {iv.occurrence_id: 0 for iv in intervals}
    mask_overlap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    idle = 0
    for left, right in zip(coords, coords[1:]):
        dt = right - left
        if dt <= 0:
            continue
        live = [iv for iv in intervals if iv.start_ns <= left and iv.end_ns >= right]
        if not live:
            idle += dt
            continue
        live_grants = [gr for gr in grants if gr.start_ns <= left and gr.end_ns >= right]
        _attribute_live_slice(dt, live, live_grants, witness_set, exposed, mask_overlap)
    return exposed, idle


def test_retired_legacy_schedule_and_dsl_solver_modules_removed() -> None:
    """Legacy target_schedule, dsl_solver, shadow schedulers, and kernel_synthesis body stay purged."""
    assert importlib.util.find_spec("mindie_sim.optimization.target_schedule") is None
    assert importlib.util.find_spec("mindie_sim.optimization.dsl_solver") is None
    assert importlib.util.find_spec("mindie_sim.optimization.reload_bounds") is None
    assert not hasattr(ks_mod, "_apply_pillar23_envelopes")
    assert not hasattr(ks_mod, "evaluate_formula_kernel")
    assert not hasattr(bundle_mod, "_build_archetype_representative_case")
    assert not hasattr(pd_mod, "build_archetype_representative_case")
    assert not hasattr(causal_tail_mod, "restore_folded_tail_commits")
    assert not hasattr(causal_tail_mod, "schedule_target_graph_on_engine")
    assert not hasattr(causal_tail_mod, "enforce_moe_mutex_on_nodes")
    with pytest.raises(NotImplementedError, match="LEGACY_PURGED"):
        ks_mod.solve_target_graph()


def test_target_step_receipt_simulation_vs_target_track_and_silu_activation_params() -> None:
    """Run solve_target_step_receipt end-to-end on target track without float('silu') crash."""
    raw = compose_case(
        "qwen3-32b",
        GEN5_DT_SKU,
        layers=TEST_COMPACT_LAYERS,
        dp=1,
        tp=2,
        ep=1,
        prompt_tokens=64,
        output_tokens=1,
        concurrency=4,
    )
    parsed = parse_case(raw)
    eval_ctx = EvaluationContext.create(parsed)
    pt = eval_ctx.bindings.primitive_table
    assert pt is not None

    req = StepRequest(phase="decode", tokens=4, context=64, sequences=4, scope="body")
    sim_receipt = solve_target_step_receipt(
        parsed, req, primitive_table=pt, stack_track="shipping",
        comparison_policy="simulation_only", max_trials=2,
    )
    tgt_receipt = solve_target_step_receipt(
        parsed, req, primitive_table=pt, stack_track="target",
        comparison_policy="monotonic_optimization", max_trials=4,
    )
    assert len(sim_receipt.stages) == 4 and len(tgt_receipt.stages) == 4
    assert sim_receipt.stages[3].status == "skipped"
    assert tgt_receipt.stages[3].status == "committed"
    assert tgt_receipt.stages[3].stack_track == "target"
    assert sim_receipt.stages[0].span_ns == tgt_receipt.stages[0].span_ns

    spans = [st.span_ns for st in tgt_receipt.stages]
    assert all(cur <= prev for prev, cur in zip(spans, spans[1:]))
    assert all(0 < st.l_star_ns <= st.span_ns for st in tgt_receipt.stages if st.status == "committed")
    assert tgt_receipt.stages[-1].span_ns <= sim_receipt.stages[-1].span_ns


def test_negative_controls_cross_precision_merge_and_hoisted_prologue_key() -> None:
    """Block FLOAT+MXFP4 merges via canonical guard and keep hoisted prologue physical keys layer-invariant."""
    float_abi = TensorPrecisionABI(
        linear_method="dense",
        input_dtype="bf16", weight_dtype="bf16", accum_dtype="fp32", output_dtype="bf16",
    )
    quant_abi = TensorPrecisionABI(
        linear_method="mx_fp4",
        input_dtype="mxfp8", weight_dtype="mxfp4", accum_dtype="fp32", output_dtype="bf16",
    )
    op_float_a = OpSpec(
        op_id="layer0.q_proj", family="gemm", recipe="matmul_bf16", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 1024, 1024), local_shape=(1, 1024, 1024),
        padded_shape=(1, 1024, 1024), launch_shape=(1, 1024, 1024),
        dtype="bf16", flops=1e9, precision_abi=float_abi,
    )
    op_float_b = replace(op_float_a, op_id="layer0.k_proj")
    op_quant = replace(op_float_a, op_id="layer0.gate_mxfp4", recipe="matmul_mxfp4", dtype="mxfp4", precision_abi=quant_abi)
    assert has_illegal_cross_precision_merge((op_float_a, op_quant)) is True
    assert has_illegal_cross_precision_merge((op_float_a, op_float_b)) is False

    hoisted_ref = f"s0.{HOISTED_PROLOGUE_PREFIX}gbsa_metadata"
    pkeys = {
        physical_key(
            OpSpec(
                op_id=f"s0.l{lyr}.attn", family="attention", stage=0, layer=lyr,
                phase="decode", requests=("r0",),
                semantic_shape=(1, 128), local_shape=(1, 128),
                padded_shape=(1, 128), launch_shape=(1, 128),
                metadata=(("payload_producer", hoisted_ref),),
            )
        )
        for lyr in (0, 1, 42, 92)
    }
    assert len(pkeys) == 1


def test_typed_codesign_opportunities_mega_fusion_space_time_and_cross_hw() -> None:
    """Populate MegaFusionBound, SpaceTimeTradeoffSpec (RecoverSSM & MLA) and CrossHwOffloadSpec."""
    # Hybrid GDN + PLE + speculative RecoverSSM (default engram_cpu_offload=True => Engram already in host DDR)
    raw_gdn = compose_case(
        "qwen3.8-flash-next", GEN5_DT_SKU, layers=TEST_COMPACT_LAYERS,
        dp=1, tp=1, ep=1, mtp_proposal_tokens=3,
        prompt_tokens=32, output_tokens=1, concurrency=1,
    )
    parsed_gdn = parse_case(raw_gdn)
    req_spec = StepRequest(phase="decode", tokens=4, context=32, sequences=1, scope="model")
    rec_gdn = solve_target_step_receipt(parsed_gdn, req_spec, stack_track="target", max_trials=2)
    spec_exp = KernelSynthesisSpec(
        spec_id="mf_test", kernel_name="tilelang_elem_chain", abi_tier="unbound_formula",
        suggested_backend="TileLang", status_tag="TO_IMPLEMENT", opt_pillar="kernel_fusion",
        constituent_op_ids=("l0.a", "l0.b"), constituent_traits=("elementwise", "elementwise"),
        symbolic_equations=("y = (a + b) * c",), input_tensors=("a", "b", "c"), output_tensors=("y",),
        elided_intermediates=("tmp",), elided_hbm_bytes=4096, tile_algorithm="row_tile",
        tile_shape="(16, 128)", ub_precision_mode="fp32_accum", ub_peak_bytes=16384,
        ub_budget_bytes=262144, l1_peak_bytes=0, l1_budget_bytes=524288, l0_peak_bytes=0,
        l0_budget_bytes=65536, sram_verified=True, l_star_us=30.0, before_us=120.0,
        target_us=40.0, target_attainment_pct=75.0, exposed_before_us=55.0,
    )
    opps_gdn = collect_codesign_opportunities(
        parsed_gdn, specs=(spec_exp,), step_request=req_spec, step_receipt=rec_gdn,
    )
    st_by_mech = {
        o.space_time.mechanism: o.space_time
        for o in opps_gdn
        if o.kind == "space_time_tradeoff" and o.space_time is not None
    }
    assert "recover_ssm_snapshot_ring" in st_by_mech
    assert st_by_mech["recover_ssm_snapshot_ring"].free_hbm_headroom_bytes >= 0

    xhw_gdn = [o.cross_hw for o in opps_gdn if o.kind == "cross_hw_offload_multistream" and o.cross_hw is not None]
    assert xhw_gdn
    ple_offload = xhw_gdn[0]
    assert ple_offload.source_tier == "host_ddr" and ple_offload.transfer_engine == "h2d"
    assert ple_offload.freed_hbm_bytes == 0  # Already host-resident under engram_cpu_offload=True
    assert ple_offload.net_exposed_ms == pytest.approx(
        max(0.0, ple_offload.transfer_ms - ple_offload.compute_hide_window_ms), abs=1e-9,
    )
    # When Engram is HBM-resident (engram_cpu_offload=False), offloading frees its HBM footprint.
    raw_gdn_hbm = compose_case(
        "qwen3.8-flash-next", GEN5_DT_SKU, layers=TEST_COMPACT_LAYERS,
        dp=1, tp=1, ep=1, mtp_proposal_tokens=3, engram_cpu_offload=False,
        prompt_tokens=32, output_tokens=1, concurrency=1,
    )
    opps_hbm = collect_codesign_opportunities(
        parse_case(raw_gdn_hbm), step_request=req_spec, step_receipt=rec_gdn,
    )
    xhw_hbm = [o.cross_hw for o in opps_hbm if o.kind == "cross_hw_offload_multistream" and o.cross_hw is not None]
    assert xhw_hbm and xhw_hbm[0].freed_hbm_bytes > 0

    mf_opps = [o.mega_fusion for o in opps_gdn if o.kind == "mega_fusion" and o.mega_fusion is not None]
    assert mf_opps
    assert mf_opps[0].exposed_before_ms == pytest.approx(0.055)
    assert mf_opps[0].gain_expected_ms <= mf_opps[0].gain_ceiling_ms + 1e-9
    assert mf_opps[0].tiled_engine_ms >= mf_opps[0].boundary_lower_bound_ms - 1e-9
    assert mf_opps[0].ub_live_bytes <= mf_opps[0].ub_capacity_bytes

    # Sparse MLA + Engram
    raw_mla = compose_case(
        "deepseek-v4.1-flash", GEN5_DT_SKU, layers=TEST_COMPACT_LAYERS,
        dp=1, tp=1, ep=1, mtp_proposal_tokens=0, prompt_tokens=32, output_tokens=1, concurrency=1,
    )
    parsed_mla = parse_case(raw_mla)
    req_mla = StepRequest(phase="decode", tokens=1, context=32, sequences=1, scope="model")
    rec_mla = solve_target_step_receipt(parsed_mla, req_mla, stack_track="target", max_trials=2)
    opps_mla = collect_codesign_opportunities(parsed_mla, step_request=req_mla, step_receipt=rec_mla)
    mla_mechs = {
        o.space_time.mechanism
        for o in opps_mla
        if o.kind == "space_time_tradeoff" and o.space_time is not None
    }
    assert "absorbed_kv_projection" in mla_mechs
    wall = evaluate_macro_wall_trigger(tpot_ms=65.0, tpot_slo_ms=50.0, hbm_used_bytes=20e9, hbm_usable_bytes=64e9, hbm_headroom_bytes=44e9)
    assert wall.wall_kind == "LATENCY_WALL"


def test_megamoe_and_moe_gating_symbolic_equations_and_clean_oss_backends() -> None:
    """Verify conditional MegaMoe EP/mode-3 equations, logits-only MoE gating, and OSS backends."""
    eq_ep1 = _format_megamoe_equation("x", "y", "[T, H]", "", "swiglu", {"ep": 1, "shared_expert_intermediate": 0})
    assert "AllToAllDispatch" not in eq_ep1 and "AllToAllCombine" not in eq_ep1 and "SharedFFN" not in eq_ep1

    eq_ep8_m2 = _format_megamoe_equation("x", "y", "[T, H]", "", "swiglu", {"ep": 8, "compute_dtype": "mxfp8"})
    assert "AllToAllDispatch(" in eq_ep8_m2 and "ep=8" in eq_ep8_m2 and "AllToAllCombine(" in eq_ep8_m2
    assert "SharedFFN" not in eq_ep8_m2

    eq_ep8_m3 = _format_megamoe_equation(
        "x", "y", "[T, H]", "", "swiglu",
        {"ep": 8, "compute_dtype": "mxfp8", "has_shared_expert": True, "shared_expert_intermediate": 2048},
    )
    assert "AllToAllDispatch(" in eq_ep8_m3 and "ep=8" in eq_ep8_m3 and "SharedFFN(" in eq_ep8_m3

    assert _resolve_symbolic_kind("router", "aclnnMatmul", {}, AlgebraicTrait.BILINEAR_MATMUL, "") == "matmul"
    assert _resolve_symbolic_kind(
        "moe_router", "aclnnMoeGatingTopKSoftmaxV2", {}, AlgebraicTrait.ROW_REDUCTION, "",
    ) == "moe_router_topk"
    eq_gate = _format_moe_gating_equation(
        "logits", "(weights, ids)", "[T, E]",
        {
            "router_mode": "sigmoid", "noaux_tc": True, "groups": 8,
            "group_topk": 4, "top_k": 8, "norm_topk": True, "routed_scaling_factor": 2.5,
        },
    )
    assert "sigmoid(logits)" in eq_gate and "bias" in eq_gate and "group_limited_topk" in eq_gate
    assert "k=8" in eq_gate and "groups=8" in eq_gate and "group_topk=4" in eq_gate
    assert "norm=True" in eq_gate and "scale=2.5" in eq_gate and "(weights, ids)" in eq_gate
    assert "@ W" not in eq_gate and "softplus" not in eq_gate and "A_log" not in eq_gate

    legacy_needle = "cann" + "bot"
    assert all(legacy_needle not in str(v).lower() for v in STRUCTURE_BACKEND_MAP.values())
    assert f"{legacy_needle}_dsl" not in si_mod.VALID_BACKEND_STACKS
    assert "tilelang" in si_mod.VALID_BACKEND_STACKS
    for spec in (*si_mod._RECIPE_REGISTRY.values(), *si_mod._FAMILY_REGISTRY.values()):
        assert legacy_needle not in spec.backend_stack.lower()
        assert legacy_needle not in spec.source_ref.lower()
        assert all(legacy_needle not in k.lower() for k in spec.native_kinds)


def test_exposed_observer_sweep_line_parity_and_subsecond_93_layer_speed() -> None:
    """Sweep-line ExposedObserver matches reference parity exactly and handles 93L trace < 0.5s."""
    small_intervals: list[ObservedInterval] = []
    small_grants: list[ObservedServiceGrant] = []
    t = 0
    for idx in range(40):
        stream_idx = 0 if (idx % 3 != 2) else 1
        stream_name = "main" if stream_idx == 0 else "aux"
        start = t
        dur = 120 if stream_idx == 0 else 210
        key = f"op_{idx}"
        small_intervals.append(
            ObservedInterval(
                occurrence_id=key, stream=stream_name, stream_index=stream_idx,
                start_ns=start, end_ns=start + dur, ordinal=idx,
            )
        )
        if idx % 2 == 0:
            small_grants.append(
                ObservedServiceGrant(
                    occurrence_id=key, resource="cube",
                    start_ns=start, end_ns=start + dur, delivered=float(dur * (idx + 1)),
                )
            )
        t += 90 if (idx % 5 != 4) else 260

    small_iv_tuple = tuple(small_intervals)
    small_gr_tuple = tuple(small_grants)
    small_end = max(iv.end_ns for iv in small_iv_tuple)
    obs = attribute_stream0_intervals(small_iv_tuple, start_ns=0, terminal_ns=small_end, grants=small_gr_tuple)
    ref_exp, ref_idle = _reference_bruteforce_stream0(small_iv_tuple, small_gr_tuple)
    assert obs.conservation_delta_ns == 0
    assert obs.exposed_ns_by_op == ref_exp
    assert obs.idle_or_wait_ns == ref_idle
    assert sum(obs.exposed_ns_by_op.values()) + obs.idle_or_wait_ns == obs.terminal_ns - obs.start_ns

    large_intervals: list[ObservedInterval] = []
    large_grants: list[ObservedServiceGrant] = []
    cursor = 0
    ordinal = 0
    for lyr in range(SYNTHETIC_DEPTH_LAYERS):
        for step in range(SYNTHETIC_OPS_PER_LAYER):
            stream_idx = 0 if (step % 5 != 4) else 1
            stream_name = "main" if stream_idx == 0 else "aux"
            dur = 450 if stream_idx == 0 else 720
            key = f"l{lyr}_op{step}"
            large_intervals.append(
                ObservedInterval(
                    occurrence_id=key, stream=stream_name, stream_index=stream_idx,
                    start_ns=cursor, end_ns=cursor + dur, ordinal=ordinal,
                )
            )
            if step % 3 == 0:
                large_grants.append(
                    ObservedServiceGrant(
                        occurrence_id=key, resource="cube",
                        start_ns=cursor, end_ns=cursor + dur, delivered=float(dur),
                    )
                )
            cursor += 380
            ordinal += 1

    large_end = max(iv.end_ns for iv in large_intervals)
    t0 = time.perf_counter()
    big_obs = attribute_stream0_intervals(large_intervals, start_ns=0, terminal_ns=large_end, grants=large_grants)
    elapsed_s = time.perf_counter() - t0
    assert elapsed_s < SWEEP_LINE_MAX_SECONDS
    assert big_obs.conservation_delta_ns == 0
    assert sum(big_obs.exposed_ns_by_op.values()) + big_obs.idle_or_wait_ns == big_obs.terminal_ns - big_obs.start_ns
    assert len(big_obs.exposed_ns_by_op) == SYNTHETIC_DEPTH_LAYERS * SYNTHETIC_OPS_PER_LAYER
