"""Unified Engine single-spine optimization, 4-stage receipts, negative controls & co-design.

[Role] Lock end-to-end invariants of the unified single-Engine execution and
  optimization spine:
  - Retirement of ``target_schedule.py``, ``dsl_solver.py``, ``causal_tail``
    shadow schedulers, and legacy ``kernel_synthesis`` helpers in favor of
    canonical ``optimization.codesign``, ``optimization.ladder``, and
    ``rewrite.schema`` APIs;
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
from mindie_sim.optimization import codesign as codesign_mod
from mindie_sim.optimization import ladder as ladder_mod
from mindie_sim.optimization.codesign import (
    collect_codesign_opportunities,
    evaluate_macro_wall_trigger,
)
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.rewrite.schema import load_rewrite_plan
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
    """Legacy target_schedule, dsl_solver, and shadow schedulers stay purged; canonical APIs own synthesis."""
    assert importlib.util.find_spec("mindie_sim.optimization.target_schedule") is None
    assert importlib.util.find_spec("mindie_sim.optimization.dsl_solver") is None
    assert importlib.util.find_spec("mindie_sim.optimization.reload_bounds") is None
    assert not hasattr(codesign_mod, "_apply_pillar23_envelopes")
    assert not hasattr(codesign_mod, "evaluate_formula_kernel")
    assert not hasattr(ladder_mod, "solve_target_graph")
    assert callable(solve_target_step_receipt)
    assert callable(collect_codesign_opportunities)
    assert callable(evaluate_macro_wall_trigger)
    assert callable(load_rewrite_plan)
    assert not hasattr(bundle_mod, "_build_archetype_representative_case")
    assert not hasattr(pd_mod, "build_archetype_representative_case")
    assert not hasattr(causal_tail_mod, "restore_folded_tail_commits")
    assert not hasattr(causal_tail_mod, "schedule_target_graph_on_engine")
    assert not hasattr(causal_tail_mod, "enforce_moe_mutex_on_nodes")


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
    parsed_gdn_hbm = parse_case(raw_gdn_hbm)
    rec_gdn_hbm = solve_target_step_receipt(parsed_gdn_hbm, req_spec, stack_track="target", max_trials=2)
    opps_hbm = collect_codesign_opportunities(
        parsed_gdn_hbm, step_request=req_spec, step_receipt=rec_gdn_hbm,
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


def test_dual_cert_port_links_abi_bindings_and_algebra_port_honesty() -> None:
    """Verify seed digest binding, per-port/switch dual_cert rows, ABI bindings, and algebra ports."""
    from types import SimpleNamespace
    from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
    from mindie_sim.optimization.causal_tail import _resolve_moe_weight_contracts
    from mindie_sim.optimization.dual_bound import build_dual_inputs_from_scenario
    from mindie_sim.prediction.results import RankOps
    from mindie_sim.rewrite.algebra import (
        _build_mla_prolog_ports,
        _law1_quant_output_ports,
        _resolve_declared_weight_layout,
        _verify_abi_query,
        classify_op_traits,
    )
    from mindie_sim.rewrite.apply import apply_plan
    from mindie_sim.rewrite.oracle import ScenarioRequest, prepare_scenario
    from mindie_sim.rewrite.schema import empty_plan
    from mindie_sim.rewrite.verify import _verify_step_eval_binding
    from mindie_sim.schema.abi import AbiShapeConstraints, OperandPortSig
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.hardware import Link
    from mindie_sim.schema.stack import ManifestBinding, SourceRef, StackTrackSpec

    case = parse_case(compose_case("qwen3-32b", "Ascend910_9391", tp=4, dp=1, layers=1))
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", tokens=1, context=64)))
    handle = dossier.handle
    ep = empty_plan(handle.scenario.base_graph_sha256, handle.scenario.framing_sha256)
    applied_u = apply_plan(ep, handle.scenario)
    assert handle.u_eval.scenario_sha256 == handle.scenario.scenario_sha256
    assert handle.u_eval.plan_sha256 == applied_u.plan_sha256
    assert handle.m_eval.plan_sha256 == apply_plan(dossier.mainline_plan, handle._mainline.scenario).plan_sha256
    _verify_step_eval_binding(handle.scenario, applied_u, handle.u_eval)
    with pytest.raises(ModelingError, match="step_eval_plan_mismatch"):
        _verify_step_eval_binding(handle.scenario, applied_u, replace(handle.u_eval, plan_sha256="0" * 64))

    def _op(op_id: str, family: str, shape: tuple[int, ...] = (4, 128), **kw) -> OpSpec:
        return OpSpec(
            op_id=op_id, family=family, stage=0, layer=0, phase="decode", requests=("r0",),
            semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape, **kw,
        )

    case_port = replace(case, machine=replace(case.machine, links=(
        Link(name="hccs_port", capacity_bytes_per_ms=200_000_000.0, members=(0, 1, 2, 3), semantics="per_port"),
    )))
    comm_ops = tuple(
        RankOps(replica=r, ops=(_op(
            f"ar_{r}", "allreduce",
            metadata=(
                ("source_die", r), ("destination_die", (r + 1) % 4),
                ("mandatory_comm_wire_bytes_by_link", (("hccs_port", 1_000_000),)),
            ),
        ),))
        for r in range(4)
    )
    _, _, names_port = build_dual_inputs_from_scenario(
        SimpleNamespace(case=case_port, machine=case_port.machine, atomic_ops=comm_ops, summaries=(), aliases={}),
        rate_mode="engine_ceiling",
    )
    tx_rows = [n for n in names_port if n.startswith("link:hccs_port@")]
    rx_rows = [n for n in names_port if n.startswith("link_rx:hccs_port@")]
    assert len(tx_rows) == 4 and len(rx_rows) == 4 and all(".d-1" not in n for n in (*tx_rows, *rx_rows))

    case_sw = replace(case, machine=replace(case.machine, links=(
        Link(
            name="sw_fabric", capacity_bytes_per_ms=100_000_000.0, members=(0, 1, 2, 3),
            semantics="per_port", switch_capacity_bytes_per_ms=150_000_000.0,
        ),
    )))
    _, _, names_sw = build_dual_inputs_from_scenario(
        SimpleNamespace(
            case=case_sw, machine=case_sw.machine,
            atomic_ops=(RankOps(replica=0, ops=(_op("c0", "allreduce", link_bytes=2_000_000),)),),
            summaries=(), aliases={},
        ),
        rate_mode="engine_ceiling",
    )
    assert any(n.startswith("link_switch:sw_fabric@") for n in names_sw)

    snap = default_abi_registry().get_snapshot("A3")
    base_unmerged = next(e for e in snap.entries if e.vllm_ascend_track == "unmerged_pr")
    ref = SourceRef(repository="vllm-ascend", revision="pr-999", relative_path="ops/custom.py", source_sha256="a" * 64)
    syn = replace(
        base_unmerged, entry_id="syn_unmerged_test", abi_id="aclnnSyntheticProposed",
        symbol_or_kernel="aclnnSyntheticProposed", available_on_device=True,
        dtype_tuple=(("bf16",),), allowed_output_tuples=(("bf16",),),
        allowed_rank_placements=frozenset({"local"}), allowed_world_sizes=frozenset({1}),
        shape_constraints=AbiShapeConstraints(), vllm_ascend_track="unmerged_pr",
        required_mechanism_id="mech_syn_pr", required_stack_pass_id="pass_syn_pr",
        source_ref=ref, device_kernel_expansion=("k_syn",),
    )
    snap2_raw = replace(snap, entries=snap.entries + (syn,), payload_sha256="")
    reg2 = ProbedAbiRegistry({"A3": replace(snap2_raw, payload_sha256=snap2_raw._compute_payload_sha256())})
    mb = ManifestBinding(
        mechanism_id="mech_syn_pr", state="unmerged_pr",
        ref="https://gitcode.com/vllm-ascend/pull/999", lowering_pass_id="pass_syn_pr", provenance=ref,
    )
    track = StackTrackSpec(
        stack_id="vllm_ascend_proposed", track="proposed", optimization_mode="simulation",
        active_bindings=(mb,), allowed_readiness=frozenset({"shipping", "proposed_pr"}),
    )
    verdict, _ = _verify_abi_query(
        abi_id="aclnnSyntheticProposed", soc_arch="A3", sku_id="a3", graph_motif="custom",
        operands=(OperandPortSig("x", "bf16", "ND", False, (4, 128), "dense", (), "none", ()),),
        outputs=(OperandPortSig("y", "bf16", "ND", False, (4, 128), "dense", (), "none", ()),),
        model_scope=next(iter(syn.allowed_model_scopes)), rank_placement="local", world_size=1,
        pr_track="proposed", track_spec=track, manifest_bindings=(), enabled_stack_passes=frozenset(), registry=reg2,
    )
    assert verdict.status == "admitted_proposed_pr" and verdict.proposed_pr_ref == mb.ref

    q_op = _op("l0.quant", "quant", dtype="fp8_e4m3", metadata=(
        ("quant_mode", "mxfp8"), ("scale_dtype", "ue8m0"), ("group_size", 32), ("output_dtype", "fp8_e4m3"),
    ))
    q_ports = _law1_quant_output_ports(q_op, classify_op_traits(q_op), 4, 128, y_name="y", scale_name="scale")
    assert any(p.dtype == "fp8_e4m3fn" and p.group_shape == (32,) for p in q_ports)
    assert any(p.role == "scale" and p.dtype == "ue8m0" and p.shape == (4, 2, 2) for p in q_ports)
    assert _resolve_declared_weight_layout(_op("l0.mm", "gemm", (4, 128, 256), metadata=(("weight_layout", "ND"),)), None) == "ND"
    qa_op = _op("l0.q_a", "gemm", (4, 1536, 7168), dtype="bf16")
    assert _build_mla_prolog_ports([qa_op], {qa_op.op_id: "q_a"}, {qa_op.op_id: classify_op_traits(qa_op)}, soc_arch="A3", rope_dim=64) is None
    up1 = _op("l0.moe_gate_proj", "grouped_gemm", (16, 2048, 4096), dtype="fp8_e4m3", metadata=(("weight_dtype", "fp8_e4m3"),))
    up2 = _op("l0.moe_up_proj", "grouped_gemm", (16, 2048, 4096), dtype="fp8_e4m3", metadata=(("weight_dtype", "fp8_e4m3"),))
    down = _op("l0.moe_down_proj", "grouped_gemm", (16, 4096, 2048), dtype="bf16", deps=("l0.moe_act",), metadata=(("weight_dtype", "int8"),))
    assert _resolve_moe_weight_contracts(case, [up1, up2, down]) == _resolve_moe_weight_contracts(case, [down, up2, up1])


def test_catlass_mx_span_resident_requests_and_codesign_honesty() -> None:
    """Verify MX scale group spans, cache key sensitivity, RequestView memory, and codesign bounds."""
    from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS
    from mindie_sim.optimization.codesign import (
        _ReceiptOpView,
        _compute_engram_hide_window_ms,
        _dual_tier_from_receipt,
        _engram_lookup_h2d_bytes,
    )
    from mindie_sim.optimization.ladder import _compute_resident_memory_receipt
    from mindie_sim.rewrite.apply import resolve_scenario
    from mindie_sim.schema.workload import RequestView
    from mindie_sim.templates.catlass_lower import _datapath_cache_key, _mx_scale_group_span
    from mindie_sim.templates.contracts import DualTierLoweringReceipt, TierLoweringOutcome

    assert _mx_scale_group_span(0, 32) == (0, 1)
    assert _mx_scale_group_span(16, 32) == (0, 2)
    assert _mx_scale_group_span(31, 2) == (0, 2)

    case = parse_case(compose_case("qwen3-32b", "Ascend910_9391", tp=1, dp=1, layers=1))
    dp = resolve_scenario(case, StepRequest("decode", tokens=1, context=64)).datapath
    if dp.usable_paths:
        assert _datapath_cache_key(dp) != _datapath_cache_key(replace(dp, usable_paths=dp.usable_paths[:-1]))

    req_pad = StepRequest(
        phase="decode", tokens=1, context=16, sequences=1, scope="model",
        requests=(
            RequestView("r0", 1, 64, True, "decode", padding=False),
            RequestView("pad", 1, 8192, False, "decode", padding=True),
        ),
    )
    req_real = StepRequest(
        phase="decode", tokens=2, context=16, sequences=1, scope="model",
        requests=(
            RequestView("r0", 1, 4096, True, "decode", padding=False),
            RequestView("r1", 1, 4096, True, "decode", padding=False),
        ),
    )
    assert (
        _compute_resident_memory_receipt(case, req_real).dies[0].kv_cache_allocated_bytes
        > _compute_resident_memory_receipt(case, req_pad).dies[0].kv_cache_allocated_bytes * 10
    )

    rh2d = _ReceiptOpView(
        "s0.l2.engram_rh2d", "engram_rh2d", "engram_rh2d", "bf16", 10.0,
        layer=2, replica=0, ready_ms=0.0, start_ms=0.4, end_ms=0.8, h2d_bytes=16384.0,
    )
    cons = _ReceiptOpView(
        "s0.l2.engram_gate", "engram", "engram", "bf16", 5.0,
        layer=2, replica=0, ready_ms=0.8, start_ms=1.0, end_ms=1.1,
    )
    early = _ReceiptOpView("s0.l0.qkv", "gemm", "gemm", "bf16", 500.0, layer=0, replica=0, start_ms=0.0, end_ms=0.5)
    mid = _ReceiptOpView("s0.l1.qkv", "gemm", "gemm", "bf16", 500.0, layer=1, replica=0, start_ms=0.3, end_ms=0.9)
    assert _engram_lookup_h2d_bytes([rh2d, cons]) == pytest.approx(16384.0)
    assert _compute_engram_hide_window_ms([early, mid, rh2d, cons], [rh2d, cons], first_k=2) == pytest.approx(0.9)

    r_dual = DualTierLoweringReceipt(
        region_id="r0", predecessor_step_ns=1_000_000,
        tier_outcomes=(
            TierLoweringOutcome("tilelang_simt", "tilelang", True, step_ns=600_000, ub_receipt=None),
            TierLoweringOutcome("tilelang_simd", "tilelang", True, step_ns=700_000, ub_receipt=None),
        ),
    )
    simt_g, simd_g, _ = _dual_tier_from_receipt(r_dual)
    assert simt_g == pytest.approx(0.4) and simd_g == pytest.approx(0.3)
    assert len(LINEAR_ATTENTION_KINDS) >= 6
