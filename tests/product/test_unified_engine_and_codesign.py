"""Unified Engine single-spine optimization, 4-stage waterfall, negative controls & co-design.

[Role] Lock end-to-end invariants of the unified single-Engine execution and
  optimization spine:
  - Retirement of ``target_schedule.py``, ``_apply_pillar23_envelopes``,
    3-layer archetype compressors, and synthetic tail cloning;
  - Strict separation between ``--mode simulation`` (shipping/proposed only,
    zero ``unbound_formula_projection``) and ``--mode modeling`` (4-stage
    causal waterfall ``before -> config_only -> framework_rewrite -> after``
    with exact conservation on ONE ``Engine`` + ONE ``PrimitiveTable``);
  - Negative controls blocking cross-precision ``FLOAT + FP8/INT8/MXFP4``
    target merges, preventing bound single ops (``AttnResFwd``, ``HC combine``,
    ``Conv1d``, ``GDN``, ``KDA``) from collapsing to the ``1.686 us`` floor,
    and forbidding ``L* > T_Engine`` lower-bound inversions;
  - Typed ``CoDesignOpportunity`` emission (``MegaFusionBound``,
    ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``), symbolic equations for
    ``aclnnMegaMoe`` and ``aclnnMoeGatingTopKSoftmaxV2``, ``TileLang`` /
    ``AscendC`` / ``Catlass`` backend recommendations (zero ``cannbot-dsl``),
    and ``ExposedObserver`` $O((N+G)\\log(N+G))$ sweep-line parity & speed.
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
from typing import Any

import pytest

from _support.axes import GEN5_DT_SKU
from mindie_sim.compose import compose_case
from mindie_sim.export import bundle as bundle_mod
from mindie_sim.export.bundle import build_export_bundle
from mindie_sim.export.models import STRUCTURE_BACKEND_MAP
from mindie_sim.optimization import causal_tail as causal_tail_mod
from mindie_sim.optimization import kernel_synthesis as ks_mod
from mindie_sim.optimization.kernel_synthesis import (
    BACKEND_ASCENDC_LABEL,
    BACKEND_CATLASS_LABEL,
    BACKEND_TILELANG_LABEL,
    TargetOpNode,
    _format_megamoe_equation,
    _format_moe_gating_equation,
    _has_illegal_cross_precision_merge,
    _validate_final_graph_invariants,
    solve_target_graph,
)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.results import OpOccurrence, StepRequest
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    ObservedServiceGrant,
    attribute_stream0_intervals,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.rewrite.algebra import AlgebraicTrait, OpTraitInfo, _resolve_symbolic_kind
from mindie_sim.rewrite.schema import Edit, FuseRegion, KernelBoundary, OpRef, RewritePlan
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI
from mindie_sim.schema.scenario import canonical_comparison_stages
from mindie_sim.sweep import pd as pd_mod

pytestmark = pytest.mark.tier1

TEST_COMPACT_LAYERS = 4
BOUND_OP_FLOOR_GUARD_US = 53.75
BOUND_OP_MIN_SURVIVAL_US = 50.0
SYNTHETIC_DEPTH_LAYERS = 93
SYNTHETIC_OPS_PER_LAYER = 26
SWEEP_LINE_MAX_SECONDS = 0.5


def _make_target_node(
    *,
    ordinal: int,
    op_id: str,
    family: str,
    recipe: str,
    dur_us: float,
    l_star_us: float,
    dtype: str = "bf16",
    flops: float = 0.0,
    vector_ops: float = 0.0,
    hbm_bytes: float = 65536.0,
    preds: tuple[OpOccurrence, ...] = (),
    precision_abi: TensorPrecisionABI | None = None,
) -> TargetOpNode:
    occ = OpOccurrence(replica=0, stage=0, rank=0, op_id=op_id)
    spec = OpSpec(
        op_id=op_id,
        family=family,
        recipe=recipe,
        phase="decode",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(1, 1024),
        local_shape=(1, 1024),
        padded_shape=(1, 1024),
        launch_shape=(1, 1024),
        flops=flops,
        vector_ops=vector_ops,
        hbm_bytes=hbm_bytes,
        dtype=dtype,
        deps=tuple(p.op_id for p in preds),
        precision_abi=precision_abi,
    )
    start_us = float(ordinal) * dur_us
    return TargetOpNode(
        ordinal=ordinal,
        occurrence=occ,
        op_id=op_id,
        archetype_id="arch_0",
        repeat_count=1,
        family=family,
        recipe=recipe,
        stream="compute",
        stream_index=0,
        task_type="AI_CORE" if flops > 0.0 else "AI_VECTOR_CORE",
        shape=(1, 1024),
        dtype=dtype,
        predecessors=preds,
        start_us=start_us,
        end_us=start_us + dur_us,
        dur_us=dur_us,
        exposed_dur_us=dur_us,
        is_overlapped=False,
        masked_by="",
        flops=flops,
        vector_ops=vector_ops,
        hbm_read_bytes=hbm_bytes * 0.5,
        hbm_write_bytes=hbm_bytes * 0.5,
        hbm_total_bytes=hbm_bytes,
        mte2_bytes=hbm_bytes * 0.5,
        mte3_bytes=hbm_bytes * 0.5,
        fixpipe_bytes=0.0,
        link_bytes=0.0,
        dominant_bound="memory_bound",
        bound_display="HBM",
        l_star_lean_us=l_star_us,
        gap_us=max(0.0, dur_us - l_star_us),
        gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
        roofline_efficiency=min(1.0, l_star_us / dur_us) if dur_us > 0.0 else None,
        absorbed_source_ops=(),
        semantic_atoms=(op_id,),
        logical_role=family,
        ascend_kernel=recipe,
        kernel_backend="CANN",
        kernel_status=ks_mod.STATUS_AT_FLOOR,
        needs_new_kernel=False,
        new_kernel_spec="",
        canonical_op_key=recipe,
        spec=spec,
        binding_state="bound_implementation",
        is_shipping_abi=True,
    )


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


def test_retired_legacy_schedule_and_fold_modules_removed() -> None:
    """Legacy target_schedule, pillar23 envelopes, 3-layer fold, and tail cloning stay deleted."""
    assert importlib.util.find_spec("mindie_sim.optimization.target_schedule") is None
    assert not hasattr(ks_mod, "_apply_pillar23_envelopes")
    assert not hasattr(bundle_mod, "_build_archetype_representative_case")
    assert not hasattr(pd_mod, "build_archetype_representative_case")
    assert not hasattr(causal_tail_mod, "restore_folded_tail_commits")


def test_simulation_vs_modeling_waterfall_on_single_engine_and_primitive_table() -> None:
    """Simulation stays shipping-only; modeling runs all 4 causal stages with exact conservation."""
    raw = compose_case(
        "qwen3-32b",
        GEN5_DT_SKU,
        layers=TEST_COMPACT_LAYERS,
        dp=1,
        tp=1,
        ep=1,
        prompt_tokens=32,
        output_tokens=1,
        concurrency=1,
    )
    parsed = parse_case(raw)
    eval_ctx = EvaluationContext.create(parsed)
    pt = eval_ctx.bindings.primitive_table
    assert pt is not None

    req = StepRequest(phase="decode", tokens=1, context=32, sequences=1, scope="model")
    sim_bundle = build_export_bundle(
        parsed,
        request=req,
        apply_target_envelope=True,
        allow_unbound_formula=False,
        primitive_table=pt,
    )
    mod_bundle = build_export_bundle(
        parsed,
        request=req,
        apply_target_envelope=True,
        allow_unbound_formula=True,
        primitive_table=pt,
    )

    assert sim_bundle.analysis_mode == "simulation"
    assert sim_bundle.has_unbound_formula_ops is False
    assert all(r.binding_state != "unbound_formula_projection" for r in sim_bundle.operators)
    assert mod_bundle.analysis_mode == "modeling"

    # Native per-layer timings cover all layers 0..L-1 without layers=3 compression.
    assert len(sim_bundle.per_layer_timings) >= TEST_COMPACT_LAYERS
    for lyr in range(TEST_COMPACT_LAYERS):
        assert ("target", lyr) in sim_bundle.per_layer_timings

    sim_wf = sim_bundle.stage_waterfall
    mod_wf = mod_bundle.stage_waterfall
    assert tuple(r["stage"] for r in sim_wf) == tuple(s.stage for s in canonical_comparison_stages("simulation"))
    assert tuple(r["stage"] for r in mod_wf) == tuple(s.stage for s in canonical_comparison_stages("modeling"))
    assert sim_wf[0]["step_span_us"] == pytest.approx(mod_wf[0]["step_span_us"], rel=1e-9)

    for wf in (sim_wf, mod_wf):
        assert wf[0]["stage"] == "before" and wf[0]["delta_from_prev_us"] == 0.0
        before_us = float(wf[0]["step_span_us"])
        after_us = float(wf[-1]["step_span_us"])
        sum_deltas = sum(-float(row["delta_from_prev_us"]) for row in wf)
        assert before_us - after_us == pytest.approx(sum_deltas, abs=1e-6)
        assert float(wf[-1]["cumulative_saved_us"]) == pytest.approx(before_us - after_us, abs=1e-6)
        for prev_row, cur_row in zip(wf, wf[1:]):
            assert float(cur_row["step_span_us"]) <= float(prev_row["step_span_us"]) + 1e-6

    assert mod_bundle.step_span_us <= sim_bundle.step_span_us + 1e-6
    for spec in (*sim_bundle.kernel_synthesis_specs, *mod_bundle.kernel_synthesis_specs):
        assert "cannbot" not in spec.suggested_backend.lower()
        assert spec.suggested_backend in {
            BACKEND_TILELANG_LABEL,
            BACKEND_ASCENDC_LABEL,
            BACKEND_CATLASS_LABEL,
            "CANN",
        }


def test_negative_controls_block_cross_precision_merge_floor_collapse_and_l_star_inversion() -> None:
    """Block FLOAT+MXFP4 merges, prevent bound single ops dropping to launch floor, block L* inversion."""
    raw = compose_case("qwen3-32b", GEN5_DT_SKU, layers=1, dp=1, tp=1, ep=1, prompt_tokens=32, output_tokens=1)
    parsed = parse_case(raw)
    pt = EvaluationContext.create(parsed).bindings.primitive_table
    assert pt is not None
    die0 = parsed.machine.dies[0]
    rates = dict(die0.rates)
    peaks = {
        0: {
            "die": die0,
            "default_dtype": "bf16",
            "vector_peak_ops_per_s": float(rates["vector"]),
            "hbm_peak_bytes_per_s": float(rates["hbm"]),
            "link_peak_bytes_per_s": 1e11,
        }
    }

    # (a) Cross-precision FLOAT + MXFP4 merge is rejected both by guard and explicit FuseRegion.
    float_abi = TensorPrecisionABI(
        linear_method="dense",
        input_dtype="bf16", weight_dtype="bf16", accum_dtype="fp32", output_dtype="bf16",
    )
    quant_abi = TensorPrecisionABI(
        linear_method="mx_fp4",
        input_dtype="mxfp8", weight_dtype="mxfp4", accum_dtype="fp32", output_dtype="bf16",
    )
    n_float = _make_target_node(
        ordinal=0, op_id="layer0.q_proj", family="linear", recipe="matmul_bf16",
        dur_us=30.0, l_star_us=10.0, dtype="bf16", flops=1e9, precision_abi=float_abi,
    )
    n_float_b = _make_target_node(
        ordinal=1, op_id="layer0.k_proj", family="linear", recipe="matmul_bf16",
        dur_us=30.0, l_star_us=10.0, dtype="bf16", flops=1e9, preds=(n_float.occurrence,),
        precision_abi=float_abi,
    )
    n_quant = _make_target_node(
        ordinal=1, op_id="layer0.gate_mxfp4", family="linear", recipe="matmul_mxfp4",
        dur_us=25.0, l_star_us=8.0, dtype="mxfp4", flops=1e9, preds=(n_float.occurrence,),
        precision_abi=quant_abi,
    )
    assert _has_illegal_cross_precision_merge([n_float, n_quant]) is True
    assert _has_illegal_cross_precision_merge([n_float, n_float_b]) is False

    bad_refs = (OpRef("atom", n_float.op_id), OpRef("atom", n_quant.op_id))
    bad_plan = RewritePlan(
        edits=(
            Edit(
                "bad_cross_prec",
                operation=FuseRegion(
                    members=bad_refs,
                    preserve_ports=(),
                    kernel_boundaries=(KernelBoundary("IllegalMixedGemm", bad_refs),),
                    fusion_tier="K2_epilogue",
                ),
            ),
        ),
    )
    with pytest.raises(ModelingError, match="cross_precision_abi_merge_forbidden"):
        solve_target_graph(
            [n_float, n_quant],
            [(10_000, 0, 5_000, 0, 0), (8_000, 0, 4_000, 0, 0)],
            parsed,
            peaks,
            rewrite_plan=bad_plan,
            primitive_table=pt,
            allow_unbound_formula=True,
            optimization_mode="modeling",
        )

    # (b) Bound 53.75 us single ops (AttnResFwd, HC combine, Conv1d, GDN, KDA) NEVER drop to 1.686 us.
    protected_defs = (
        ("layer0.attnres_fwd", "attnres", "attnres_fwd"),
        ("layer0.hc_combine", "mhc", "mhc_combine"),
        ("layer0.causal_conv1d", "causal_conv", "causal_conv1d_decode"),
        ("layer0.gdn_scan", "linear_attention", "gdn_recurrent"),
        ("layer0.kda_scan", "kda", "kda_recurrent"),
    )
    bound_nodes: list[TargetOpNode] = []
    bound_dems: list[tuple[int, int, int, int, int]] = []
    prev_occ: tuple[OpOccurrence, ...] = ()
    for idx, (oid, fam, rec) in enumerate(protected_defs):
        nd = _make_target_node(
            ordinal=idx, op_id=oid, family=fam, recipe=rec,
            dur_us=BOUND_OP_FLOOR_GUARD_US, l_star_us=12.0,
            vector_ops=2e7, hbm_bytes=1_048_576.0, preds=prev_occ,
        )
        dem = (0, 12_000, 10_000, 0, 0)
        # Standalone bound op MUST survive untouched (never collapsed to 1.686 us).
        solo_solved = solve_target_graph(
            [replace(nd, ordinal=0, predecessors=())], [dem], parsed, peaks,
            primitive_table=pt, allow_unbound_formula=True, optimization_mode="modeling",
        )
        solo_by_id = {n.op_id: n for n in solo_solved.nodes}
        assert oid in solo_by_id
        assert solo_by_id[oid].dur_us >= BOUND_OP_MIN_SURVIVAL_US
        bound_nodes.append(nd)
        bound_dems.append(dem)
        prev_occ = (nd.occurrence,)

    # Even in a chained sequence, no surviving or fused node may collapse to the 1.686 us floor.
    solved = solve_target_graph(
        bound_nodes, bound_dems, parsed, peaks,
        primitive_table=pt, allow_unbound_formula=True, optimization_mode="modeling",
    )
    assert solved.nodes
    assert all(n.dur_us >= 10.0 for n in solved.nodes)

    # (c) L* > T_Engine lower-bound inversion raises loudly.
    inverted = _make_target_node(
        ordinal=0, op_id="layer0.forged_inversion", family="vector", recipe="rmsnorm",
        dur_us=5.0, l_star_us=12.0, vector_ops=1e6,
    )
    with pytest.raises(ModelingError, match="lower_bound_inversion"):
        _validate_final_graph_invariants([inverted], [(0, 12_000, 0, 0, 0)])

    # (d) Strict Engine monotonicity: solve_target_graph never makes baseline slower without explicit plan.
    from mindie_sim.optimization.causal_tail import schedule_target_graph_on_engine, weighted_step_span_us
    from mindie_sim.prediction.program.physical_key import HOISTED_PROLOGUE_PREFIX, physical_key

    orig_sched = schedule_target_graph_on_engine(bound_nodes, parsed, primitive_table=pt)
    assert weighted_step_span_us(solved.nodes) <= weighted_step_span_us(orig_sched.nodes) + 1e-6

    # (e) Hoisted stage-level prologue ops (`s0.pre_forward.<name>`, no `.l0.`) stay layer-invariant.
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
    # Hybrid GDN + PLE + speculative RecoverSSM
    raw_gdn = compose_case(
        "qwen3.8-flash-next",
        GEN5_DT_SKU,
        layers=TEST_COMPACT_LAYERS,
        dp=1,
        tp=1,
        ep=1,
        mtp_proposal_tokens=3,
        prompt_tokens=32,
        output_tokens=1,
        concurrency=1,
    )
    parsed_gdn = parse_case(raw_gdn)
    req_spec = StepRequest(phase="decode", tokens=4, context=32, sequences=1, scope="model")
    bundle_gdn = build_export_bundle(
        parsed_gdn, request=req_spec, apply_target_envelope=True, allow_unbound_formula=True,
    )

    opps_gdn = bundle_gdn.co_design_opportunities
    assert opps_gdn
    st_by_mech = {
        o.space_time.mechanism: o.space_time
        for o in opps_gdn
        if o.kind == "space_time_tradeoff" and o.space_time is not None
    }
    assert "recover_ssm_snapshot_ring" in st_by_mech
    ring_spec = st_by_mech["recover_ssm_snapshot_ring"]
    assert ring_spec.free_hbm_headroom_bytes >= 0
    assert isinstance(ring_spec.feasible_under_hbm_and_sla, bool)

    xhw_gdn = [o.cross_hw for o in opps_gdn if o.kind == "cross_hw_offload_multistream" and o.cross_hw is not None]
    assert xhw_gdn
    ple_offload = xhw_gdn[0]
    assert ple_offload.source_tier == "host_ddr" and ple_offload.transfer_engine == "h2d"
    assert ple_offload.freed_hbm_bytes > 0
    assert ple_offload.net_exposed_ms == pytest.approx(
        max(0.0, ple_offload.transfer_ms - ple_offload.compute_hide_window_ms), abs=1e-9,
    )

    for opp in opps_gdn:
        if opp.kind == "mega_fusion" and opp.mega_fusion is not None:
            mf = opp.mega_fusion
            assert mf.gain_expected_ms <= mf.gain_ceiling_ms + 1e-9
            assert mf.tiled_engine_ms >= mf.boundary_lower_bound_ms - 1e-9
            assert mf.ub_live_bytes <= mf.ub_capacity_bytes

    # Sparse MLA + Engram
    raw_mla = compose_case(
        "deepseek-v4.1-flash",
        GEN5_DT_SKU,
        layers=TEST_COMPACT_LAYERS,
        dp=1,
        tp=1,
        ep=1,
        mtp_proposal_tokens=0,
        prompt_tokens=32,
        output_tokens=1,
        concurrency=1,
    )
    parsed_mla = parse_case(raw_mla)
    req_mla = StepRequest(phase="decode", tokens=1, context=32, sequences=1, scope="model")
    bundle_mla = build_export_bundle(
        parsed_mla, request=req_mla, apply_target_envelope=True, allow_unbound_formula=True,
    )
    mla_mechs = {
        o.space_time.mechanism
        for o in bundle_mla.co_design_opportunities
        if o.kind == "space_time_tradeoff" and o.space_time is not None
    }
    assert "absorbed_kv_projection" in mla_mechs


def test_megamoe_and_moe_gating_symbolic_equations_and_backend_priority() -> None:
    """Verify conditional MegaMoe EP/mode-3 equations, logits-only MoE gating, and TileLang/AscendC."""
    trait_local = OpTraitInfo(
        op_id="layer0.moe", trait=AlgebraicTrait.ASSOCIATIVE_COLLECTIVE,
        collective_axis="", activation_kind="swiglu", input_dtype="bf16",
    )
    eq_ep1 = _format_megamoe_equation(
        "x", "y", "[T, H]", "W1, W2", trait_local,
        {"megamoe_geometry": {"ep": 1, "enable_fused_mc2": 1, "shared_intermediate": 0}},
    )
    assert "AllToAllDispatch" not in eq_ep1 and "AllToAllCombine" not in eq_ep1 and "SharedFFN" not in eq_ep1

    trait_ep = OpTraitInfo(
        op_id="layer0.moe", trait=AlgebraicTrait.ASSOCIATIVE_COLLECTIVE,
        collective_axis="ep", activation_kind="swiglu", input_dtype="mxfp8",
    )
    eq_ep8_m2 = _format_megamoe_equation(
        "x", "y", "[T, H]", "W1, W2", trait_ep,
        {"megamoe_geometry": {"ep": 8, "enable_fused_mc2": 2, "shared_intermediate": 0}},
    )
    assert "AllToAllDispatch(" in eq_ep8_m2 and "ep=8" in eq_ep8_m2 and "AllToAllCombine(" in eq_ep8_m2
    assert "SharedFFN" not in eq_ep8_m2

    eq_ep8_m3 = _format_megamoe_equation(
        "x", "y", "[T, H]", "W1, W2", trait_ep,
        {"megamoe_geometry": {"ep": 8, "enable_fused_mc2": 3, "shared_intermediate": 2048}},
    )
    assert "AllToAllDispatch(" in eq_ep8_m3 and "ep=8" in eq_ep8_m3 and "SharedFFN(" in eq_ep8_m3

    # Router GEMM stays matmul; MoE gating renders strictly from logits -> (weights, ids).
    assert _resolve_symbolic_kind("router", "aclnnMatmul", {}, AlgebraicTrait.BILINEAR_MATMUL, "") == "matmul"
    assert _resolve_symbolic_kind(
        "moe_router", "aclnnMoeGatingTopKSoftmaxV2", {}, AlgebraicTrait.ROW_REDUCTION, "",
    ) == "moe_router_topk"
    eq_gate = _format_moe_gating_equation(
        "logits", "(weights, ids)", "[T, E]",
        {
            "router_mode": "sigmoid",
            "noaux_tc": True,
            "groups": 8,
            "group_topk": 4,
            "top_k": 8,
            "norm_topk": True,
            "routed_scaling_factor": 2.5,
        },
    )
    assert "sigmoid(logits)" in eq_gate and "bias" in eq_gate and "group_limited_topk" in eq_gate
    assert "k=8" in eq_gate and "groups=8" in eq_gate and "group_topk=4" in eq_gate
    assert "norm=True" in eq_gate and "scale=2.5" in eq_gate and "(weights, ids)" in eq_gate
    assert "@ W" not in eq_gate and "softplus" not in eq_gate and "A_log" not in eq_gate

    assert BACKEND_TILELANG_LABEL == "TileLang"
    assert BACKEND_ASCENDC_LABEL == "AscendC"
    assert BACKEND_CATLASS_LABEL == "Catlass"
    assert all("cannbot" not in str(v).lower() for v in STRUCTURE_BACKEND_MAP.values())


def test_exposed_observer_sweep_line_parity_and_subsecond_93_layer_speed() -> None:
    """Sweep-line ExposedObserver matches reference parity exactly and handles 93L trace < 0.5s."""
    # (a) Exact parity against brute-force reference on contended multi-stream trace.
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

    # (b) 93-layer Kimi-scale timeline runs well under 0.5s with zero conservation drift.
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


def test_synthesis_dp_tuples_causal_tail_no_byte_inflation_opportunity_bounds_and_clean_backends() -> None:
    """Verify DP tuple safety, causal-tail conservation, opportunity scoping, and zero cannbot backends."""
    from mindie_sim import structure_implementations as si_mod

    # 1. DP>=2 tuple StepRequest.sequences never raises TypeError in _peak_sequences_per_dp.
    raw = compose_case("qwen3-32b", GEN5_DT_SKU, tp=1, dp=2, layers=2, prompt_tokens=32, output_tokens=2)
    dp2_case = parse_case(raw)
    dp_req = StepRequest(phase="decode", tokens=(8, 8), context=256, sequences=(8, 8), scope="body")
    assert ks_mod._peak_sequences_per_dp(dp_req, dp2_case) == 8

    # 2. _node_physical_demand_and_placement never fabricates memory@die bytes from contended dur_us.
    dp_cap, rates, peaks, svc_modes, link_bps = causal_tail_mod._effective_die_supply(
        dp2_case.machine.dies[0], dp2_case.machine.links, dp2_case.machine.transfer_channels,
    )
    contended_node = replace(
        _make_target_node(
            ordinal=0, op_id="s0.l0.v", family="elementwise", recipe="add",
            dur_us=500.0, l_star_us=12.0, vector_ops=1024.0, hbm_bytes=2048.0,
        ),
        exposed_dur_us=50.0, is_overlapped=True,
        hbm_read_bytes=1024.0, hbm_write_bytes=1024.0, hbm_total_bytes=2048.0,
    )
    demand, _place, lat_ms, _act = causal_tail_mod._node_physical_demand_and_placement(
        contended_node, die_tag="d0", lane_id="main", datapath=dp_cap, rates=rates, peaks=peaks,
        service_modes=svc_modes, link_bps=link_bps, owners=None, has_topology_plan=False, launch_floor_us=0.0,
    )
    hbm_peak = float(rates.get("hbm") or max(dp_cap.hbm_read_bytes_per_s, dp_cap.hbm_write_bytes_per_s))
    eff_hbm = causal_tail_mod._effective_hbm_rate_bps(1024.0, 1024.0, hbm_peak, svc_modes)
    assert demand.get("memory@d0", 0.0) == pytest.approx(2048.0 * (hbm_peak / eff_hbm))
    assert lat_ms == pytest.approx(0.012)

    # 3. Unaggregated hybrid nodes (repeat_count==1) preserve native ordinal order with zero arch_barrier.
    n0 = replace(
        _make_target_node(ordinal=0, op_id="l0.gdn", family="elementwise", recipe="add", dur_us=5.0, l_star_us=1.0, vector_ops=1024.0, hbm_bytes=4096.0),
        archetype_id="gdn", start_us=0.0, end_us=5.0,
    )
    n1 = replace(
        _make_target_node(ordinal=1, op_id="l1.attn", family="elementwise", recipe="add", dur_us=5.0, l_star_us=1.0, vector_ops=1024.0, hbm_bytes=4096.0, preds=(n0.occurrence,)),
        archetype_id="attn", start_us=5.0, end_us=10.0,
    )
    n2 = replace(
        _make_target_node(ordinal=2, op_id="l2.gdn", family="elementwise", recipe="add", dur_us=5.0, l_star_us=1.0, vector_ops=1024.0, hbm_bytes=4096.0, preds=(n1.occurrence,)),
        archetype_id="gdn", start_us=10.0, end_us=15.0,
    )
    sched = causal_tail_mod.schedule_target_graph_on_engine((n0, n1, n2), dp2_case)
    assert [n.op_id for n in sched.nodes] == ["l0.gdn", "l1.attn", "l2.gdn"]
    assert sched.nodes[0].start_us < sched.nodes[1].start_us < sched.nodes[2].start_us

    # 4. RecoverSSM commit-only match, PP-stage MLA split, Engram lookup-only H2D, and MegaFusion exposed_before_ms.
    verify_node = _make_target_node(
        ordinal=0, op_id="l0.scan", family="gdn_state", recipe="gdn_arch35_recover_ssm_verify", dur_us=10.0, l_star_us=1.0,
    )
    verify_node = replace(
        verify_node, spec=replace(verify_node.spec, metadata=(("recover_ssm_stage", "verify_compact_record"),)),
    )
    commit_node = _make_target_node(
        ordinal=1, op_id="l0.commit", family="recover_ssm", recipe="aclnnRecoverSsmState", dur_us=10.0, l_star_us=1.0,
    )
    commit_node = replace(
        commit_node, spec=replace(commit_node.spec, metadata=(("recover_ssm_stage", "commit"),)),
    )
    assert not ks_mod._is_recover_ssm_commit_node(verify_node)
    assert ks_mod._is_recover_ssm_commit_node(commit_node)

    pp2_case = replace(dp2_case, runtime=replace(dp2_case.runtime, parallel=replace(dp2_case.runtime.parallel, pp=2)))
    stage_max = ks_mod._max_pp_stage_bytes(pp2_case, pp2_case.model.layer_plans, lambda _lp: 1000)
    assert stage_max == 1000

    gate_node = replace(
        _make_target_node(ordinal=0, op_id="l1.engram_gate_proj", family="matmul", recipe="matmul", dur_us=10.0, l_star_us=1.0, hbm_bytes=999_999.0),
        hbm_write_bytes=500_000.0,
    )
    rh2d_node = _make_target_node(
        ordinal=1, op_id="l1.rh2d", family="engram_rh2d", recipe="engram_cpu_decode_pinned_staging_h2d",
        dur_us=5.0, l_star_us=1.0, hbm_bytes=8192.0,
    )
    rh2d_node = replace(
        rh2d_node, spec=replace(rh2d_node.spec, metadata=(("resource_work", {"h2d": 4096.0}),)),
    )
    assert ks_mod._engram_lookup_h2d_bytes((gate_node, rh2d_node)) == pytest.approx(4096.0)

    # 4b. MegaFusionBound.exposed_before_ms uses conserved exposed_before_us, and stage waterfall has zero scalar clamping.
    spec_exp = ks_mod.KernelSynthesisSpec(
        spec_id="mf_test",
        kernel_name="tilelang_elem_chain",
        abi_tier="unbound_formula",
        suggested_backend=BACKEND_TILELANG_LABEL,
        status_tag="TO_IMPLEMENT",
        opt_pillar="kernel_fusion",
        constituent_op_ids=("l0.a", "l0.b"),
        constituent_traits=("elementwise", "elementwise"),
        symbolic_equations=("y = (a + b) * c",),
        input_tensors=("a", "b", "c"),
        output_tensors=("y",),
        elided_intermediates=("tmp",),
        elided_hbm_bytes=4096,
        tile_algorithm="row_tile",
        tile_shape="(16, 128)",
        ub_precision_mode="fp32_accum",
        ub_peak_bytes=16384,
        ub_budget_bytes=262144,
        l1_peak_bytes=0,
        l1_budget_bytes=524288,
        l0_peak_bytes=0,
        l0_budget_bytes=65536,
        sram_verified=True,
        l_star_us=30.0,
        before_us=120.0,
        target_us=40.0,
        target_attainment_pct=75.0,
        exposed_before_us=55.0,
    )
    opps = ks_mod._collect_codesign_opportunities(
        dp2_case, (), (spec_exp,), (), primitive_table=None, step_request=dp_req,
    )
    mf_opps = [o for o in opps if o.mega_fusion is not None]
    assert mf_opps and mf_opps[0].mega_fusion is not None
    assert mf_opps[0].mega_fusion.exposed_before_ms == pytest.approx(0.055)

    def _sched_of(span_us: float) -> causal_tail_mod.DieScheduleResult:
        node = replace(n0, start_us=0.0, end_us=span_us, dur_us=span_us, exposed_dur_us=span_us)
        return causal_tail_mod.DieScheduleResult((node,), {"gdn": span_us}, 0.0, span_us, span_us, {}, "", 0.0)

    wf = ks_mod._build_stage_waterfall(
        "modeling",
        {"before": _sched_of(100.0), "config_only": _sched_of(90.0), "framework_rewrite": _sched_of(80.0), "after": _sched_of(70.0)},
    )
    assert [row["step_span_us"] for row in wf] == pytest.approx([100.0, 90.0, 80.0, 70.0])

    # 5. structure_implementations exposes zero cannbot_dsl backends or paths.
    assert "cannbot_dsl" not in si_mod.VALID_BACKEND_STACKS
    assert "tilelang" in si_mod.VALID_BACKEND_STACKS
    for spec in (*si_mod._RECIPE_REGISTRY.values(), *si_mod._FAMILY_REGISTRY.values()):
        assert "cannbot" not in spec.backend_stack.lower()
        assert "cannbot" not in spec.source_ref.lower()
        assert all("cannbot" not in k.lower() for k in spec.native_kinds)
