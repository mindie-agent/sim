"""Tier-0 contracts for allowlist-only public views, 0.050 ms materiality, OSS kernel catalog & LaunchCodec."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.tier0

from mindie_sim.export.action_plan import (
    NEW_KERNEL_MIN_SAVED_MS,
    build_comparison_plan,
)
from mindie_sim.export.agent_brief import (
    AGENT_BRIEF_MAX_BYTES,
    export_agent_brief,
    format_executive_digest_markdown,
)
from mindie_sim.export.kernel_catalog import resolve_ascend_kernel_identity
from mindie_sim.export.models import (
    BACKEND_ASCENDC,
    BACKEND_CATLASS,
    BACKEND_TILELANG,
    BINDING_STATE_BOUND,
    BINDING_STATE_UNBOUND_FORMULA,
    STRUCTURE_BACKEND_MAP,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    TRACK_KERNEL,
    AlignedComparisonRow,
    ExportBundle,
    MemoryCalculatorLedger,
    OperatorExportRow,
)
from mindie_sim.export.public_view import (
    THREE_TRACK_LABELS,
    diff_public_scenario_configs,
    format_track_label,
    project_agent_brief_mapping,
    project_public_effective_config,
)
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import KernelBinding, OpSpec
from mindie_sim.schema.scenario import (
    CoDesignOpportunity,
    CrossHwOffloadSpec,
    MegaFusionBound,
    SpaceTimeTradeoffSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DSL_NEEDLE = "cann" + "bot"
RAW_INTERNAL_ENUM_RE = re.compile(r"\b(expected_ideal|unmerged_pr|raw_eager|merged)\b")


def _make_memory_ledger(
    *, dp: int = 1, tp: int = 2, ep: int = 2, conc: int = 4,
    weight_quant: str = "mxfp8_mxfp4", cache_dtype: str = "fp8",
) -> MemoryCalculatorLedger:
    return MemoryCalculatorLedger(
        model_name="synthetic-moe",
        hardware_sku="Ascend950DT_9582",
        phase="decode",
        weight_quant=weight_quant,
        cache_dtype=cache_dtype,
        engram_host_offload=False,
        dp=dp, tp=tp, ep=ep, pp=1, dcp=1, pcp=1, local_tp=tp, shared_tp=tp,
        total_dies=dp * tp,
        seq_len=4096,
        concurrency=conc,
        block_size=128,
        spec_tokens=0,
        local_cache_tokens=4096,
        die_hbm_capacity_bytes=96.0 * (1024 ** 3),
        gpu_memory_utilization=0.9,
        usable_hbm_bytes=86.4 * (1024 ** 3),
        weight_rows=(),
        cache_rows=(),
        stage_budgets=(),
        max_stage_hbm_bytes=32.0 * (1024 ** 3),
        max_stage_host_offload_bytes=0.0,
        fits_hbm=True,
        sim_memory_weight_bytes_by_stage=(20.0 * (1024 ** 3),),
        sim_memory_cache_bytes_by_stage=(8.0 * (1024 ** 3),),
        weight_quant_display=weight_quant.upper(),
        weight_footprint_per_die_bytes=20.0 * (1024 ** 3),
        max_hbm_concurrency_per_dp=16,
        native_bf16_weight_footprint_per_die_bytes=80.0 * (1024 ** 3),
        weight_compression_ratio=4.0,
    )


def _make_op_row(
    op_id: str,
    *,
    ascend_kernel: str = "FusedRMSNorm",
    exposed_dur_us: float = 100.0,
    l_star_lean_us: float = 80.0,
    repeat_count: int = 1,
    needs_new_kernel: bool = False,
    is_shipping_abi: bool = True,
    kernel_backend: str = BACKEND_TILELANG,
    source_ref: str = "",
    formula_eval: object = None,
    absorbed_source_ops: tuple[str, ...] = (),
) -> OperatorExportRow:
    return OperatorExportRow(
        ordinal=0,
        occurrence=OpOccurrence(0, 0, 0, op_id),
        op_id=op_id,
        layer=0,
        layer_label="L0",
        submodule="Attention",
        family="elementwise",
        recipe="norm",
        implementation="norm",
        stream="main",
        stream_index=0,
        task_type="AI_VECTOR_CORE",
        shape=(4, 4096),
        dtype="bf16",
        predecessors=(),
        start_us=0.0,
        end_us=exposed_dur_us,
        dur_us=exposed_dur_us,
        flops=0.0,
        vector_ops=1e6,
        hbm_read_bytes=1e6,
        hbm_write_bytes=1e6,
        hbm_total_bytes=2e6,
        mte2_bytes=1e6,
        mte3_bytes=1e6,
        fixpipe_bytes=0.0,
        link_bytes=0.0,
        l2_hit_pct=0.0,
        dominant_bound="memory_bound",
        l_star_lean_us=l_star_lean_us,
        gap_us=max(0.0, exposed_dur_us - l_star_lean_us),
        gap_ratio=1.25,
        gap_radar_rank=1,
        roofline_efficiency=0.8,
        absorbed_source_ops=absorbed_source_ops,
        exposed_dur_us=exposed_dur_us,
        repeat_count=repeat_count,
        archetype_id="arch0",
        archetype_title="Decoder Layer",
        logical_role="NormAct",
        ascend_kernel=ascend_kernel,
        kernel_backend=kernel_backend,
        kernel_status="new_custom" if needs_new_kernel else "shipping",
        source_ref=source_ref,
        needs_new_kernel=needs_new_kernel,
        is_shipping_abi=is_shipping_abi,
        binding_state=BINDING_STATE_UNBOUND_FORMULA if needs_new_kernel else BINDING_STATE_BOUND,
        formula_eval=formula_eval,
    )


def _make_bundle(
    ops: tuple[OperatorExportRow, ...],
    *,
    step_span_us: float = 1000.0,
    feature_track: str = "merged",
    effective_config: dict | None = None,
    applied_edit_receipts: tuple[dict, ...] = (),
    recommended_checkpoint: dict | None = None,
    va_serving_rules_summary: dict | None = None,
    pd_pool_summary: dict | None = None,
    co_design_opportunities: tuple[CoDesignOpportunity, ...] = (),
    memory_ledger: MemoryCalculatorLedger | None = None,
) -> ExportBundle:
    mem = memory_ledger or _make_memory_ledger()
    n_tok = float(mem.concurrency) / float(max(1, mem.tp))
    return ExportBundle(
        schema="mindie-sim.export-bundle/1",
        model_name="synthetic-moe",
        hardware_sku="Ascend950DT_9582",
        phase="decode",
        scope="model",
        graph_id="g0",
        tokens_per_seq=1,
        context_tokens=4096,
        sequences=mem.concurrency,
        step_span_us=step_span_us,
        operator_sum_us=step_span_us,
        step_l_star_lean_us=step_span_us * 0.8,
        step_certificate_valid=True,
        step_tight_resources=("memory@0",),
        throughput_tokens_per_s=200.0,
        throughput_per_die_tokens_per_s=100.0,
        network_cube_mfu=25.0,
        network_vec_mfu=15.0,
        network_mbu=65.0,
        network_link_util=10.0,
        network_roofline_efficiency=0.8,
        exposed_breakdown_us={"Attention": step_span_us},
        bottleneck_mix={"Attention": {"memory_bound": step_span_us}},
        hardware_peaks={},
        operators=ops,
        pipelines=(),
        gap_radar=(),
        memory_ledger=mem,
        feature_track=feature_track,
        hardware_die_summary="Ascend950DT_9582",
        deployment_mode="PD Decode",
        prompt_tokens=4096,
        output_tokens=256,
        tpot_slo_ms=50.0,
        ttft_slo_ms=None,
        n_tok_per_die=n_tok,
        us_per_tok_die=step_span_us / max(n_tok, 1e-6),
        recommended_checkpoint=recommended_checkpoint,
        va_serving_rules_summary=va_serving_rules_summary,
        pd_pool_summary=pd_pool_summary,
        applied_edit_receipts=applied_edit_receipts,
        effective_config=dict(effective_config or {}),
        co_design_opportunities=co_design_opportunities,
    )


def _assert_zero_canary_leaks(serialized: str) -> None:
    low = serialized.lower()
    for forbidden in (
        "canary",
        "'audit'",
        '"audit"',
        "feature_track",
        "moe_route",
        "solver_feasible",
        "hw_isa_guidance",
        "secret_debug_scalar",
        "secret_nested",
        "secret_top_key",
        LEGACY_DSL_NEEDLE,
    ):
        assert forbidden not in low, f"Found forbidden token {forbidden!r} in:\n{serialized}"
    assert RAW_INTERNAL_ENUM_RE.search(serialized) is None, (
        f"Found raw internal track enum in:\n{serialized}"
    )


def test_public_allowlist_drops_canary_keys_dicts_and_internal_tracks(tmp_path: Path) -> None:
    assert format_track_label("merged") == "当前栈基线"
    assert format_track_label("unmerged_pr") == "提案版本"
    assert format_track_label("expected_ideal") == "目标架构预演"
    assert format_track_label("raw_eager") == "Eager 基线"

    dsl_tag = f"{LEGACY_DSL_NEEDLE}-dsl"
    before_op1 = _make_op_row("l0.norm", exposed_dur_us=60.0, l_star_lean_us=40.0)
    before_op2 = _make_op_row("l0.act", exposed_dur_us=60.0, l_star_lean_us=40.0)
    after_op = _make_op_row(
        "l0.fused_norm_act",
        ascend_kernel="FusedRMSNormAct",
        exposed_dur_us=50.0,
        l_star_lean_us=40.0,
        needs_new_kernel=True,
        is_shipping_abi=False,
        kernel_backend=dsl_tag,
        source_ref=f"cann/{dsl_tag}/samples/norm_act",
        formula_eval={
            "composed_equations": ["y = RMSNorm(x)", {"audit": "canary"}, "merged expected_ideal"],
            "dominant_pipe": {"audit": "canary"},
            "ub_peak_bytes": 16384,
            "ub_budget_bytes": 262144,
            "sram_verified": True,
            "projected_us": 50.0,
        },
        absorbed_source_ops=("l0.norm", "l0.act"),
    )
    co_opps = (
        CoDesignOpportunity(
            opportunity_id="opp_mega",
            kind="mega_fusion",
            track="kernel_new",
            readiness="synthesized_target",
            title="Fuse norm and act",
            evidence_refs=(f"cann/{dsl_tag}/samples/norm",),
            mega_fusion=MegaFusionBound(
                region_op_ids=("l0.norm", "l0.act"),
                exposed_before_ms=0.12,
                boundary_lower_bound_ms=0.04,
                tiled_engine_ms=0.05,
                eliminated_gm_bytes=1048576,
                ub_live_bytes=16384,
                ub_capacity_bytes=262144,
                gain_ceiling_ms=0.08,
                gain_expected_ms=0.07,
            ),
        ),
        CoDesignOpportunity(
            opportunity_id="opp_st",
            kind="space_time_tradeoff",
            track="config",
            readiness="shipping",
            title="Absorbed KV projection trade-off",
            space_time=SpaceTimeTradeoffSpec(
                direction="time_for_space",
                mechanism="absorbed_kv_projection",
                delta_hbm_bytes=-2_000_000_000,
                free_hbm_headroom_bytes=8_000_000_000,
                delta_concurrency_per_dp=8,
                delta_step_ms=0.5,
                delta_tps_per_die=12.5,
                feasible_under_hbm_and_sla=True,
            ),
        ),
        CoDesignOpportunity(
            opportunity_id="opp_xhw",
            kind="cross_hw_offload_multistream",
            track="framework",
            readiness="proposed_pr",
            title="Engram host prefetch overlap",
            cross_hw=CrossHwOffloadSpec(
                target_component="engram_tables",
                source_tier="host_ddr",
                transfer_engine="h2d",
                compute_hide_window_ms=1.2,
                transfer_ms=0.9,
                net_exposed_ms=0.0,
                freed_hbm_bytes=4_000_000_000,
                delta_concurrency_per_dp=4,
                delta_tps_per_die=9.0,
            ),
        ),
    )
    dirty_eff = {
        "solver_feasible": True,
        "feature_track": "expected_ideal",
        "secret_debug_scalar": "canary_scalar",
        "secret_nested": {"audit": "canary"},
        "moe_route": {
            "enable_fused_mc2": 2,
            "multistream_overlap_shared_expert": False,
            "nested_leak": {"audit": "canary"},
        },
    }
    before_b = _make_bundle(
        (before_op1, before_op2),
        step_span_us=120.0,
        feature_track="merged",
        effective_config={**dirty_eff, "feature_track": "merged", "enable_fused_mc2": 0},
    )
    after_b = _make_bundle(
        (after_op,),
        step_span_us=50.0,
        feature_track="expected_ideal",
        effective_config=dirty_eff,
        applied_edit_receipts=(
            {
                "status": "applied",
                "opcode": "FuseRegion",
                "edit_id": "fuse_norm_act",
                "input_occurrences": ["l0.norm", "l0.act"],
                "output_occurrences": ["l0.fused_norm_act"],
            },
        ),
        recommended_checkpoint={
            "repo_id": "Eco-Tech/Model-QuaRot",
            "quant_preset": "mxfp8_mxfp4",
            "uses_quarot": True,
            "canary_extra": {"audit": "canary"},
        },
        va_serving_rules_summary={
            "effective_kv_cache_dtype": "fp8_e4m3",
            "speculative_enabled": True,
            "capture_sizes": [1, 2, 4, "canary_bad"],
            "hw_isa_guidance": "secret_canary_guidance",
            "unknown_rule": {"audit": "canary"},
        },
        pd_pool_summary={
            "pd_ratio": "1P:2D",
            "pd_transfer_mode": "layerwise_overlap_d2d",
            "used_dies": 16,
            "decode_parallel": {"dp": 2, "tp": 4, "canary_axis": 99},
            "canary_pool": {"audit": "canary"},
        },
        co_design_opportunities=co_opps,
    )

    pub_eff = project_public_effective_config(after_b)
    assert pub_eff == {"enable_fused_mc2": 2, "multistream_overlap_shared_expert": False}

    diffs = diff_public_scenario_configs(before_b, after_b)
    assert any("MC2" in lbl for lbl, _, _ in diffs)
    _assert_zero_canary_leaks(json.dumps(diffs, ensure_ascii=False))

    aligned = (
        AlignedComparisonRow(
            "arch0", "Decoder Layer", 1, "Attention",
            (before_op1, before_op2), (after_op,), "2->1", "fuse_norm_act",
        ),
    )
    dirty_audit = {
        "source_script": "serve.sh",
        "issues_found": [
            {
                "issue_id": "bad_flag",
                "title": f"replace {dsl_tag} merged",
                "current": "{'audit': 'canary'}",
                "recommended": "use TileLang",
                "impact": "feature_track expected_ideal",
                "extra_secret": {"audit": "canary"},
            },
        ],
        "two_step_gain": {
            "step1_immediate_zero_code": {
                "before_step_ms": 1.2,
                "tuned_step_ms": 0.9,
                "saved_ms": 0.3,
                "canary_metric": 999,
            },
            "secret_step": {"audit": "canary"},
        },
        "secret_audit_key": {"audit": "canary"},
    }
    plan = build_comparison_plan(before_b, after_b, aligned, script_audit=dirty_audit)
    assert len(plan.pending_items) == 1
    assert plan.pending_items[0].suggested_backend == BACKEND_TILELANG
    assert all("canary" not in row.action_detail.lower() for row in plan.roadmap)
    assert any(row.track_label == THREE_TRACK_LABELS[TRACK_CONFIG_ENV] for row in plan.roadmap)
    assert any(row.track_label == THREE_TRACK_LABELS[TRACK_FRAMEWORK] for row in plan.roadmap)

    out_bundle_path = tmp_path / "secret_dir_name" / "brief_bundle.json"
    dirty_files = {
        "agent_brief_json": str(out_bundle_path),
        "comparison_xlsx": str(tmp_path / "secret_dir_name" / "cmp.xlsx"),
        "canary_file": "/tmp/canary.txt",
    }
    brief_bundle = export_agent_brief(
        before_b, after_b, out_bundle_path,
        generated_files=dirty_files, script_audit=dirty_audit, comparison_plan=plan,
    )
    assert isinstance(brief_bundle, dict)
    bundle_text = out_bundle_path.read_text(encoding="utf-8")
    assert len(bundle_text.encode("utf-8")) < AGENT_BRIEF_MAX_BYTES
    _assert_zero_canary_leaks(bundle_text)
    assert brief_bundle["action_plan_3_tracks"]["track_labels"] == dict(THREE_TRACK_LABELS)
    assert len(brief_bundle["codesign_opportunities"]) == 3
    assert set(brief_bundle["generated_files"]) == {"agent_brief_json", "comparison_xlsx"}

    # Mapping overload must pass through the exact same allowlist projector.
    dirty_mapping = {
        **brief_bundle,
        "secret_top_key": {"audit": "canary"},
        "feature_track": "expected_ideal",
        "moe_route": {"enable_fused_mc2": 1},
        "solver_feasible": False,
        "generated_files": {
            "agent_brief_json": str(out_bundle_path),
            "secret_file": "/tmp/canary_leak.bin",
        },
        "va_serving_rules": {
            "effective_kv_cache_dtype": "fp8_e4m3",
            "hw_isa_guidance": "canary_isa",
        },
    }
    out_map_path = tmp_path / "brief_map.json"
    export_agent_brief(dirty_mapping, out_map_path)
    map_text = out_map_path.read_text(encoding="utf-8")
    _assert_zero_canary_leaks(map_text)
    assert project_agent_brief_mapping(dirty_mapping) == json.loads(map_text)

    digest_from_dirty = format_executive_digest_markdown(dirty_mapping)
    _assert_zero_canary_leaks(digest_from_dirty)
    assert THREE_TRACK_LABELS[TRACK_CONFIG_ENV] in digest_from_dirty
    assert THREE_TRACK_LABELS[TRACK_FRAMEWORK] in digest_from_dirty
    assert THREE_TRACK_LABELS[TRACK_KERNEL] in digest_from_dirty
    assert "secret_dir_name" not in digest_from_dirty
    assert "brief_bundle.json" in digest_from_dirty


def test_materiality_threshold_routes_sub_50us_fusions_to_residual_ops() -> None:
    assert NEW_KERNEL_MIN_SAVED_MS == pytest.approx(0.050)

    def _run_case(before_us: float, after_us: float, *, pre_edit_us: float | None = None):
        b1 = _make_op_row("l0.op1", exposed_dur_us=before_us * 0.5, l_star_lean_us=after_us * 0.4)
        b2 = _make_op_row("l0.op2", exposed_dur_us=before_us * 0.5, l_star_lean_us=after_us * 0.4)
        fe = ({"after_pre_edit_exposed_us": pre_edit_us} if pre_edit_us is not None else None)
        aft = _make_op_row(
            "l0.fused_micro",
            ascend_kernel="FusedMicroOp",
            exposed_dur_us=after_us,
            l_star_lean_us=after_us * 0.8,
            needs_new_kernel=True,
            is_shipping_abi=False,
            formula_eval=fe,
            absorbed_source_ops=("l0.op1", "l0.op2"),
        )
        b_mem = _make_memory_ledger(conc=4)
        a_mem = _make_memory_ledger(conc=8 if pre_edit_us is not None else 4)
        b_bun = _make_bundle((b1, b2), step_span_us=before_us, memory_ledger=b_mem)
        a_bun = _make_bundle(
            (aft,),
            step_span_us=after_us,
            memory_ledger=a_mem,
            applied_edit_receipts=(
                {
                    "status": "applied",
                    "opcode": "FuseRegion",
                    "edit_id": "fuse_micro",
                    "input_occurrences": ["l0.op1", "l0.op2"],
                    "output_occurrences": ["l0.fused_micro"],
                },
            ),
        )
        aligned = (
            AlignedComparisonRow(
                "arch0", "Decoder Layer", 1, "Attention", (b1, b2), (aft,), "2->1", "fuse_micro",
            ),
        )
        return build_comparison_plan(b_bun, a_bun, aligned)

    for saved_us in (1.0, 14.0, 49.9):
        plan_sub = _run_case(100.0, 100.0 - saved_us)
        assert plan_sub.pending_items == ()
        assert not any(r.track == TRACK_FRAMEWORK for r in plan_sub.roadmap)
        assert plan_sub.group_accounting[0].bucket == "residual"
        assert plan_sub.waterfall.new_kernel_saved_ms == pytest.approx(0.0, abs=1e-9)
        assert plan_sub.waterfall.residual_saved_ms == pytest.approx(saved_us / 1000.0, abs=1e-9)
        assert abs(plan_sub.waterfall.reconciliation_error_ms) <= 1e-9

    # Inclusive boundary: exactly 0.050 ms qualifies for Sheet 4 card and Track [2] roadmap.
    plan_eq = _run_case(100.0, 50.0)
    assert len(plan_eq.pending_items) == 1
    assert any(r.track == TRACK_FRAMEWORK for r in plan_eq.roadmap)
    assert plan_eq.group_accounting[0].bucket == "new"
    assert plan_eq.waterfall.new_kernel_saved_ms == pytest.approx(0.050, abs=1e-9)

    # Split attribution: net gain 0.200 ms (mostly config), kernel share only 0.014 ms -> Residual Ops.
    plan_split = _run_case(300.0, 100.0, pre_edit_us=114.0)
    assert plan_split.pending_items == ()
    assert not any(r.track == TRACK_FRAMEWORK for r in plan_split.roadmap)
    assert plan_split.group_accounting[0].bucket == "residual"


def _make_op_spec(
    op_id: str,
    family: str,
    *,
    shape: tuple[int, ...] = (4, 128),
    metadata: tuple[tuple[str, object], ...] = (),
    kernel_binding: KernelBinding | None = None,
) -> OpSpec:
    return OpSpec(
        op_id=op_id,
        family=family,
        phase="decode",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=shape,
        local_shape=shape,
        padded_shape=shape,
        launch_shape=shape,
        dtype="bf16",
        metadata=metadata,
        kernel_binding=kernel_binding,
    )


def test_kernel_catalog_birth_binding_recover_ssm_and_oss_backends() -> None:
    from mindie_sim.compose import compose_case
    import mindie_sim.export as export_pkg
    import mindie_sim.export.models as export_models
    from mindie_sim.prediction.structure import parse_case

    assert not hasattr(export_pkg, "BACKEND_CANNBOT_DSL")
    assert not hasattr(export_models, "BACKEND_CANNBOT_DSL")
    assert all(LEGACY_DSL_NEEDLE not in str(v).lower() for v in STRUCTURE_BACKEND_MAP.values())

    a5_case = parse_case(compose_case(
        "deepseek-v2-lite", hardware="Ascend950DT_9582", dp=1, tp=2, ep=2, layers=2, parallel="fixed",
    ))

    # 1. Birth KernelBinding on OpSpec is authoritative first.
    birth_spec = _make_op_spec(
        "l0.custom_op",
        "elementwise",
        kernel_binding=KernelBinding(
            kernel_id="AuthoritativeBirthKernel",
            backend="Custom-Ops",
            arch_family="dav_3510",
            readiness="proposed_pr",
            source_ref=SourceRef(
                repository="Ascend/tilelang-ascend",
                revision="main",
                relative_path="kernels/auth.py",
                source_sha256="0" * 64,
            ),
        ),
    )
    desc_birth = resolve_ascend_kernel_identity(
        "l0.custom_op", birth_spec, "elementwise", "add", (4, 128), "bf16", a5_case, None, (),
        native_binding={"kernels": [{"name": "Add"}]},
    )
    assert desc_birth.ascend_kernel == "AuthoritativeBirthKernel"
    assert desc_birth.kernel_backend == BACKEND_TILELANG
    assert "tilelang-ascend" in desc_birth.source_ref and "kernels/auth.py" in desc_birth.source_ref

    # 2. RecoverSSMStateCommit resolves cleanly via metadata or op_id (never 'Add').
    ssm_spec = _make_op_spec(
        "s0.l0.tail_op",
        "elementwise",
        metadata=(("recover_ssm_stage", "commit"),),
    )
    desc_ssm_meta = resolve_ascend_kernel_identity(
        "s0.l0.tail_op", ssm_spec, "elementwise", "add", (4, 128), "bf16", a5_case, None, (),
        native_binding={"kernels": [{"name": "Add"}]},
    )
    assert desc_ssm_meta.ascend_kernel == "RecoverSSMStateCommit"
    assert desc_ssm_meta.ascend_kernel != "Add"
    assert desc_ssm_meta.kernel_backend in {BACKEND_TILELANG, BACKEND_ASCENDC}

    # 3. A3-only kernels never leak onto A5; GEMM/grouped-GEMM defaults to Catlass.
    a5_moe_spec = _make_op_spec(
        "s0.l0.moe_ep",
        "grouped_moe",
        shape=(4, 2048),
        metadata=(("weight_format", "mxfp4"),),
    )
    desc_a5_moe = resolve_ascend_kernel_identity(
        "s0.l0.moe_ep", a5_moe_spec, "grouped_moe", "dispatch_ffn_combine", (4, 2048), "mxfp4", a5_case, None, (),
        native_binding={"kernels": [{"name": "dispatch_ffn_combine"}]},
    )
    assert desc_a5_moe.ascend_kernel != "dispatch_ffn_combine"
    assert "MegaMoe" in desc_a5_moe.ascend_kernel
    assert desc_a5_moe.kernel_backend == BACKEND_CATLASS

    desc_a5_mla = resolve_ascend_kernel_identity(
        "s0.l0.latent_attention", None, "attention", "paged_mla", (4, 128, 512), "bf16", a5_case, None, (),
        native_binding={"kernels": [{"name": "FusedInferAttentionScore"}]},
    )
    assert "FusedInferAttentionScore" not in desc_a5_mla.ascend_kernel

    synth_gemm_spec = _make_op_spec(
        "s0.l0.fused_gemm",
        "gemm",
        shape=(4, 2048),
        metadata=(("target_rewrite_fused", True), ("fused_kernel_name", "FusedCompactGroupedGemm")),
    )
    desc_a5_synth_gemm = resolve_ascend_kernel_identity(
        "s0.l0.fused_gemm", synth_gemm_spec, "gemm", "custom_gemm", (4, 2048), "mxfp4", a5_case, None, (),
    )
    assert desc_a5_synth_gemm.kernel_backend == BACKEND_CATLASS

    synth_vec_spec = _make_op_spec(
        "s0.l0.fused_vec",
        "elementwise",
        shape=(4, 2048),
        metadata=(("target_rewrite_fused", True), ("fused_kernel_name", "FusedVecNormAct")),
    )
    desc_a5_synth_vec = resolve_ascend_kernel_identity(
        "s0.l0.fused_vec", synth_vec_spec, "elementwise", "custom_vec", (4, 2048), "bf16", a5_case, None, (),
    )
    assert desc_a5_synth_vec.kernel_backend in {BACKEND_TILELANG, BACKEND_ASCENDC}


def test_zero_legacy_dsl_tokens_and_retired_script_audit_module() -> None:
    import mindie_sim.export as export_pkg
    from mindie_sim.templates.dsl_catalog import DSL_REPOSITORY

    assert importlib.util.find_spec("mindie_sim.export.script_audit") is None
    for retired in (
        "parse_launch_script", "audit_launch_script", "render_optimized_serve_sh",
        "ParsedLaunchScript", "ScriptAuditIssue", "ScriptAuditReport",
    ):
        assert not hasattr(export_pkg, retired), f"Retired shim {retired!r} must not exist"

    assert LEGACY_DSL_NEEDLE not in DSL_REPOSITORY.lower()
    assert "tilelang" in DSL_REPOSITORY.lower()

    hits: list[str] = []
    for rel_dir in ("mindie_sim/templates", "mindie_sim/export", "scripts"):
        base = REPO_ROOT / rel_dir
        for py_file in sorted(base.rglob("*.py")):
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if LEGACY_DSL_NEEDLE in text.lower():
                hits.append(str(py_file.relative_to(REPO_ROOT)))
    assert hits == [], f"Unexpected legacy DSL references in {hits}"


def test_vllm_ascend_launch_codec_roundtrip_and_cli_infeasible_honesty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mindie_sim.cli import _main as cli_main
    from mindie_sim.compose import compose_case
    from mindie_sim.stacks.vllm_ascend.launch_codec import VllmAscendLaunchCodec
    from mindie_sim.sweep import pd as pd_mod

    codec = VllmAscendLaunchCodec()
    bad_script = tmp_path / "bad_a5_serve.sh"
    bad_script.write_text(
        'export HCCL_OP_EXPANSION_MODE="AI_CPU"\n'
        'vllm serve /weights/qwen3-32b --tensor-parallel-size 2 --data-parallel-size 1 '
        '--enforce-eager --additional-config \'{"enable_fused_mc2": 1, "multistream_overlap_shared_expert": true}\'\n',
        encoding="utf-8",
    )
    ingress = codec.parse_launch_config(bad_script)
    assert ingress.deployment_mode_hint is None
    tuned_a5 = {
        "model_name": "qwen3-32b", "model_ref": "/weights/qwen3-32b",
        "hardware_sku": "Ascend950DT_9582", "hw_family": "a5",
        "role": "pd_decode", "phase": "decode",
        "dp": 1, "tp": 2, "ep": 1, "pp": 1, "dcp": 1, "pcp": 1,
        "max_num_seqs": 8, "max_model_len": 4096,
        "weight_format": "MXFP8", "kv_cache_dtype": "fp8",
        "hccl_op_expansion_mode": "CCU",
        "fits_hbm": True, "step_span_us": 12000.0, "enable_fused_mc2": 1,
    }
    audit = codec.audit_launch_config(ingress, {**tuned_a5, "step_span_us": 20000.0}, tuned_a5)
    issue_ids = {i["issue_id"] for i in audit["issues_found"]}
    assert {"mc2_mso_mutex_conflict", "missing_aclgraph_full_decode", "hccl_expansion_mode_suboptimal"} <= issue_ids

    rendered_a5 = codec.render_launch_config(tuned_a5)
    assert 'HCCL_OP_EXPANSION_MODE="CCU"' in rendered_a5
    assert "FULL_DECODE_ONLY" in rendered_a5
    reparsed_a5 = codec.parse_launch_config(rendered_a5)
    assert reparsed_a5.deployment_mode_hint == "pd_decode"
    assert reparsed_a5.parallel.tp == 2

    rendered_a3 = codec.render_launch_config({**tuned_a5, "hardware_sku": "Ascend910_9392", "hw_family": "a3", "weight_format": "W8A8", "kv_cache_dtype": "c8"})
    assert 'HCCL_OP_EXPANSION_MODE="AIV"' in rendered_a3
    assert "CCU" not in rendered_a3

    # CLI report honest infeasible outcome: zero fabricated B=1 and zero 5000.0 TTFT leak.
    def _infeasible_solve(*_a, **_kw):
        return {
            "feasible": False,
            "best_configuration": None,
            "outcome": {
                "status": "infeasible",
                "limiting_constraint": "hbm_capacity_oom",
                "physical_b_vram_max": 0,
            },
            "candidates": [{"b_vram_max": 0, "infeasible_reason": "hbm_capacity_oom"}],
        }

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "solve_optimal_deployment", _infeasible_solve)
        inf_dir = tmp_path / "inf_out"
        inf_rec = tmp_path / "inf_rec.json"
        rc_inf = cli_main([
            "report", "--model", "qwen3-32b", "--hardware", "a3",
            "--deployment-mode", "pd_decode", "--num-devices", "4",
            "--concurrency", "32", "--prompt-tokens", "256", "--output-tokens", "32",
            "--no-trace", "--no-xlsx", "--out-dir", str(inf_dir), "--json", "--receipt", str(inf_rec),
        ])
        assert rc_inf == 0 and inf_rec.is_file()
        rec_inf = json.loads(inf_rec.read_text(encoding="utf-8"))
        assert rec_inf["feasible"] is False
        assert rec_inf["status"] == "infeasible"
        assert rec_inf["limiting_constraint"] == "hbm_capacity_oom"
        assert rec_inf["feasible_b_per_dp"] == 0
        assert rec_inf["physical_b_vram_max"] == 0
        assert rec_inf["requested_concurrency"] == 32
        assert rec_inf["ttft_slo_ms"] is None
        brief_inf_text = json.dumps(rec_inf["agent_brief"], ensure_ascii=False)
        _assert_zero_canary_leaks(brief_inf_text)
        assert "5000.0" not in brief_inf_text
        assert rec_inf["agent_brief"]["scenario_and_parallelism"]["concurrency_per_dp"] == 0
        digest_inf = format_executive_digest_markdown(rec_inf["agent_brief"])
        assert "hbm_capacity_oom" in digest_inf and "Feasible B/DP=0" in digest_inf

    # CLI report solved path never overwrites checked b_feasible with higher --concurrency.
    sample_case = compose_case("qwen3-32b", "a3", tp=2, layers=1, prompt_tokens=128, output_tokens=32, quant="w8a8_dynamic")

    def _capped_solve(*_a, **_kw):
        return {
            "feasible": True,
            "winner_mode": "colocated",
            "best_configuration": {
                "deployment_mode": "colocated",
                "parallel": {"dp": 2, "tp": 2, "ep": 1, "pp": 1, "pcp": 1, "cp": 1},
                "b_feasible": 4,
                "b_vram_max": 8,
                "mtp_proposal_tokens": 0,
                "used_dies": 4,
            },
            "recommended_case": sample_case,
        }

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "solve_optimal_deployment", _capped_solve)
        cap_dir = tmp_path / "cap_out"
        cap_rec = tmp_path / "cap_rec.json"
        rc_cap = cli_main([
            "report", "--model", "qwen3-32b", "--hardware", "a3",
            "--deployment-mode", "colocated", "--num-devices", "4",
            "--concurrency", "64", "--prompt-tokens", "128", "--output-tokens", "32",
            "--mode", "simulation", "--no-trace", "--no-xlsx",
            "--out-dir", str(cap_dir), "--json", "--receipt", str(cap_rec),
        ])
        assert rc_cap == 0 and cap_rec.is_file()
        rec_cap = json.loads(cap_rec.read_text(encoding="utf-8"))
        assert rec_cap["feasible"] is True
        assert rec_cap["requested_concurrency"] == 64
        assert rec_cap["agent_brief"]["scenario_and_parallelism"]["concurrency_per_dp"] == 4
