"""Tri-generation unseen HuggingFace architecture structural generalization gate.

[Role] Prove end-to-end structural generalization of ``compose_case`` ->
  ``PredictionSpine`` -> ``Engine`` (``c_native``) + ``ResidentMemoryReceipt``
  across Ascend A2, A3, and A5 using brand-new synthetic HuggingFace
  ``config.json`` architectures absent from ``model_configs/`` and
  ``catalog/configs/``.
[Invariants] Every unseen ``(synthetic_model, generation, phase)`` tuple
  executes with zero ``Uncalibrated`` states, zero model-name lookup errors,
  exact integer-byte ``ResidentMemoryReceipt`` conservation (``0 B`` drift),
  exact step-span nanosecond conservation (``0 ns`` drift), and bit-for-bit
  prediction invariance when the exact same ``config.json`` is renamed to an
  unrelated identifier.
[Timeouts & Locks] Local Tier-0 analytical replay bounded by small 2–4 layer
  synthetic configs; no network, devices, or locks.
[Side-Effect Policy] Writes synthetic JSON configs exclusively inside pytest
  ``tmp_path``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import pytest

from _support.axes import (
    DUAL_DIE_SKU,
    GEN2_SKU,
    GEN5_DT_SKU,
    GEN5_DT_SMALL_SKU,
    PHASES,
    REPRESENTATIVE_SKUS,
)
from mindie_sim import compose_case
from mindie_sim.export import (
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_PENDING_KERNELS,
    SHEET_SUMMARY,
    align_before_after_operators,
    build_export_bundle,
    export_agent_brief,
    export_comparison_xlsx,
    export_msprof_trace,
    format_executive_digest_markdown,
)
from mindie_sim.export.action_plan import ComparisonPlan, build_comparison_plan
from mindie_sim.export.memory_ledger import _resolve_ledger_workspace_tokens
from mindie_sim.export.ooxml import NS_MAIN
from mindie_sim.model_catalog import CATALOG
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.results import LoweringRequest, ReplayOptions, StepRequest
from mindie_sim.prediction.runtime import accel_build
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.structure import parse_case

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIRS = (
    REPO_ROOT / "mindie_sim" / "model_configs",
    REPO_ROOT / "mindie_sim" / "catalog" / "configs",
)

TRI_GEN_SKUS = (
    pytest.param(GEN2_SKU, id="a2"),
    pytest.param(DUAL_DIE_SKU, id="a3"),
    pytest.param(GEN5_DT_SMALL_SKU, id="a5"),
)
MULTI_DIE_TRI_GEN_SKUS = (
    pytest.param(GEN2_SKU, id="a2"),
    pytest.param(DUAL_DIE_SKU, id="a3"),
    pytest.param(GEN5_DT_SKU, id="a5"),
)
EXPORT_TRI_GEN_SKUS = tuple(
    pytest.param(REPRESENTATIVE_SKUS[gen], id=gen) for gen in ("a2", "a3", "a5")
)

PROMPT_TOKENS = 128
OUTPUT_TOKENS = 16
CONTEXT_TOKENS = 128
PREFILL_TOKENS = 128
DECODE_TOKENS = 1
NS_PER_MS = 1_000_000.0
RENAMED_SLUG = "totally-unknown-xyz-99b"
RENAMED_MODEL_TYPE = "totally_unknown_xyz_99b"
RENAMED_ARCH_CLASS = "TotallyUnknownXyz99BForCausalLM"
ZETA_HYBRID_SLUG = "zeta-unseen-hybrid-99b"
ZETA_HYBRID_MODEL_TYPE = "zeta_unseen_hybrid_99b"
ZETA_HYBRID_ARCH_CLASS = "ZetaUnseenHybrid99BForCausalLM"
OMEGA_RENAMED_SLUG = "omega-renamed-hybrid-99b"
OMEGA_RENAMED_MODEL_TYPE = "omega_renamed_hybrid_99b"
OMEGA_RENAMED_ARCH_CLASS = "OmegaRenamedHybrid99BForCausalLM"
NS_PREFIX = f"{{{NS_MAIN}}}"
EXPECTED_VISIBLE_SHEETS = (
    SHEET_SUMMARY,
    SHEET_BREAKDOWN,
    SHEET_MEMORY,
    SHEET_PENDING_KERNELS,
)
EXPECTED_SHEET2_COL_COUNT = 27
EXPECTED_SHEET2_MAX_COL = "AA"
EXPECTED_STAGE_IDS = (
    "before_merged",
    "stage1_ops_config",
    "stage2_unmerged_pr",
    "stage3_target_after",
)
AGENT_BRIEF_MAX_BYTES = 8192
DIST_SHARD_DEGREE = 4
DIST_UNALIGNED_SEQS = 3
EXPORT_SHARD_DEGREE = 2
EXPORT_CONCURRENCY = 2

ZETA_HYBRID_DIST_CONFIG: dict[str, Any] = {
    "model_type": ZETA_HYBRID_MODEL_TYPE,
    "architectures": [ZETA_HYBRID_ARCH_CLASS],
    "num_hidden_layers": 2,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 4096,
    "moe_intermediate_size": 1024,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "vocab_size": 32768,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
    "rms_norm_eps": 1e-6,
    "layer_types": ["gdn", "full_attention"],
    "linear_attention_config": {
        "num_k_heads": 8,
        "num_v_heads": 16,
        "head_k_dim": 128,
        "head_v_dim": 128,
        "conv_kernel_size": 4,
    },
}

ZETA_HYBRID_EXPORT_CONFIG: dict[str, Any] = {
    "model_type": ZETA_HYBRID_MODEL_TYPE,
    "architectures": [ZETA_HYBRID_ARCH_CLASS],
    "num_hidden_layers": 2,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "intermediate_size": 2048,
    "moe_intermediate_size": 512,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "vocab_size": 8192,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
    "rms_norm_eps": 1e-6,
    "layer_types": ["gdn", "full_attention"],
    "linear_attention_config": {
        "num_k_heads": 8,
        "num_v_heads": 16,
        "head_k_dim": 64,
        "head_v_dim": 64,
        "conv_kernel_size": 4,
    },
}

SYNTHETIC_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "aurora-dense-gqa": {
        "model_type": "aurora_dense_gqa",
        "architectures": ["AuroraDenseGqaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 65536,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
    },
    "zephyr-moe-mla": {
        "model_type": "zephyr_moe_mla",
        "architectures": ["ZephyrMoeMlaForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 8192,
        "moe_intermediate_size": 1536,
        "n_routed_experts": 32,
        "num_experts_per_tok": 4,
        "n_shared_experts": 1,
        "vocab_size": 65536,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
    },
    "nova-hybrid-gdn-gqa": {
        "model_type": "nova_hybrid_gdn_gqa",
        "architectures": ["NovaHybridGdnGqaForCausalLM"],
        "num_hidden_layers": 4,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 65536,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "layer_types": ["gdn", "gdn", "gdn", "full_attention"],
        "linear_attention_config": {
            "num_k_heads": 16,
            "num_v_heads": 32,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "conv_kernel_size": 4,
        },
    },
}


def _evaluate_synthetic_config(
    config_path: Path, sku: str, phase: str, spine: PredictionSpine,
) -> dict[str, Any]:
    tokens = PREFILL_TOKENS if phase == "prefill" else DECODE_TOKENS
    raw_case = compose_case(
        config_path,
        hardware=sku,
        tp=1,
        dp=1,
        ep=1,
        num_devices=1,
        concurrency=1,
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )
    case = parse_case(raw_case)
    request = LoweringRequest(
        step=StepRequest(
            phase=phase,
            tokens=tokens,
            context=CONTEXT_TOKENS,
            sequences=1,
            scope="body",
        )
    )
    prepared = spine.lower(case, request)
    step_eval = spine.run_step(prepared, ReplayOptions(isolated=True))
    views = prepared.graph.request_views[0]
    receipt = Memory(case).resident_memory_receipt(
        phase=phase,
        request_views=views,
    )
    executable_ops = tuple(op for entry in prepared.graph.executable for op in entry.ops)
    total_flops = sum(float(op.flops) for op in executable_ops)
    total_hbm_bytes = sum(float(op.hbm_bytes) for op in executable_ops)
    exec_graph_signature = tuple(
        (op.op_id, op.deps, op.family, op.flops, op.hbm_bytes)
        for op in executable_ops
    )
    op_signature = tuple(
        (
            row.occurrence.op_id,
            row.ordinal,
            row.family,
            row.implementation,
            tuple(pred.op_id for pred in row.predecessors),
            row.stream,
            row.shape,
            row.dtype,
            row.ready_ms,
            row.start_ms,
            row.end_ms,
            row.isolated_ms,
            row.work,
        )
        for row in step_eval.operators
    )
    return {
        "case": case,
        "prepared": prepared,
        "views": views,
        "step_eval": step_eval,
        "receipt": receipt,
        "executable_ops": executable_ops,
        "exec_graph_signature": exec_graph_signature,
        "total_flops": total_flops,
        "total_hbm_bytes": total_hbm_bytes,
        "op_signature": op_signature,
    }


def _assert_calibrated_and_conserved(result: dict[str, Any], arch_key: str) -> None:
    case = result["case"]
    views = result["views"]
    step_eval = result["step_eval"]
    receipt = result["receipt"]
    die = receipt.dies[0]

    assert step_eval.accel_backend == "c_native"
    assert step_eval.native_steps > 0
    assert math.isfinite(step_eval.span_ms) and step_eval.span_ms > 0.0
    # PredictionSpine binds the sealed silicon-calibrated PrimitiveTable per SKU
    # (pre-spine compose_case MachineSpec remains declared_nominal until bound).
    assert step_eval.primitive_table is not None
    assert step_eval.primitive_table.provenance.calibrated is True
    assert bool(step_eval.primitive_table.provenance.campaign_id)
    assert step_eval.primitive_table.warnings == ()
    assert step_eval.primitive_table.provenance.silicon_fingerprint.aic_count == (
        case.machine.dies[0].compute_units
    )
    assert not any(
        "uncalibrated" in str(val).lower()
        for val in dict(case.runtime.metadata).values()
    )

    executable_ops = result["executable_ops"]
    assert len(step_eval.operators) == len(executable_ops) > 0
    assert {op.op_id for op in executable_ops} == {
        row.occurrence.op_id for row in step_eval.operators
    }
    assert result["total_flops"] > 0.0
    assert result["total_hbm_bytes"] > 0.0

    row_by_id = {row.occurrence.op_id: row for row in step_eval.operators}
    min_start_ms = min(row.start_ms for row in step_eval.operators)
    assert round(min_start_ms * NS_PER_MS) == 0
    cp_end_ms: dict[str, float] = {}
    internal_edges = 0
    for row in step_eval.operators:
        assert row.ready_ms is not None and row.start_ms is not None
        assert row.end_ms is not None and row.isolated_ms is not None
        assert row.ready_ms >= 0.0
        assert row.start_ms >= row.ready_ms - 1e-12
        assert row.end_ms >= row.start_ms
        assert row.isolated_ms >= 0.0
        assert all(item.amount >= 0.0 for item in row.work)
        pred_cp = 0.0
        for pred in row.predecessors:
            pred_row = row_by_id.get(pred.op_id)
            if pred_row is None:
                assert pred.op_id.endswith("_input"), f"unexpected dangling predecessor: {pred.op_id}"
                continue
            internal_edges += 1
            assert row.ready_ms >= pred_row.end_ms - 1e-12
            if pred.op_id in cp_end_ms:
                pred_cp = max(pred_cp, cp_end_ms[pred.op_id])
        cp_end_ms[row.occurrence.op_id] = pred_cp + float(row.isolated_ms)
    assert internal_edges > 0
    max_end_ms = max(row.end_ms for row in step_eval.operators)
    assert round(max_end_ms * NS_PER_MS) == round(step_eval.span_ms * NS_PER_MS)
    # 1e-6 ms (1 ns) IEEE-754 sum tolerance; rejects any real schedule drift.
    assert step_eval.span_ms >= max(cp_end_ms.values()) - 1e-6

    # Independent byte-exact KV cache conservation against lowered StepRequest views.
    kv_tokens_total = sum(int(view.context) + int(view.q_len) for view in views)
    kv_elems_per_token = 0
    for plan in case.model.layer_plans:
        if not plan.owns_long_kv:
            continue
        if int(plan.kv_cache_elements) > 0:
            kv_elems_per_token += int(plan.kv_cache_elements)
        elif plan.attention == "mla":
            kv_elems_per_token += int(plan.kv_lora_rank) + int(plan.qk_rope_head_dim)
        elif plan.attention == "gqa":
            kv_elems_per_token += 2 * int(plan.num_kv_heads) * int(plan.head_dim)
    bf16_bytes = 2
    assert die.kv_cache_valid_bytes == kv_tokens_total * kv_elems_per_token * bf16_bytes

    fixed_bytes = (
        die.checkpoint_weight_bytes
        + die.prepared_weight_bytes
        + die.vision_weight_bytes
        + die.workspace_bytes
        + die.reserve_bytes
    )
    dynamic_bytes = (
        die.kv_cache_allocated_bytes
        + die.index_cache_allocated_bytes
        + die.recurrent_state_bytes
        + die.draft_cache_bytes
        + die.staging_bytes
        + die.dynamic_grant_bytes
    )
    assert die.checkpoint_weight_bytes > 0
    assert die.kv_cache_allocated_bytes >= die.kv_cache_valid_bytes > 0
    assert die.simultaneous_live_hbm_bytes == fixed_bytes + dynamic_bytes
    assert die.simultaneous_peak_hbm_bytes >= die.simultaneous_live_hbm_bytes > 0
    assert die.free_hbm_headroom_bytes == die.usable_hbm_bytes - die.simultaneous_peak_hbm_bytes

    if arch_key == "aurora-dense-gqa":
        assert case.model.attention == "gqa" and case.model.ffn == "dense"
        assert die.recurrent_state_bytes == 0
    elif arch_key == "zephyr-moe-mla":
        assert case.model.attention == "mla" and case.model.ffn == "moe"
        assert case.model.num_experts == 32
    elif arch_key == "nova-hybrid-gdn-gqa":
        assert die.recurrent_state_bytes > 0
        assert {plan.attention for plan in case.model.layer_plans} == {"gdn", "gqa"}


def test_synthetic_architectures_are_genuinely_absent_from_packaged_catalog() -> None:
    packaged_stems = {
        path.stem for cfg_dir in MODEL_CONFIG_DIRS if cfg_dir.is_dir()
        for path in cfg_dir.glob("*.json")
    }
    for name in (*SYNTHETIC_ARCHITECTURES, RENAMED_SLUG, ZETA_HYBRID_SLUG, OMEGA_RENAMED_SLUG):
        assert name not in CATALOG
        assert name not in packaged_stems
    assert accel_build.build_accel()["backend"] == "c_native"


@pytest.mark.parametrize("arch_key", tuple(SYNTHETIC_ARCHITECTURES))
@pytest.mark.parametrize("sku", TRI_GEN_SKUS)
@pytest.mark.parametrize("phase", PHASES)
def test_unseen_synthetic_architecture_generalizes_and_is_name_invariant(
    tmp_path: Path, arch_key: str, sku: str, phase: str,
) -> None:
    base_cfg = SYNTHETIC_ARCHITECTURES[arch_key]
    renamed_cfg = {
        **base_cfg,
        "model_type": RENAMED_MODEL_TYPE,
        "architectures": [RENAMED_ARCH_CLASS],
    }
    base_path = tmp_path / f"{arch_key}.json"
    renamed_path = tmp_path / f"{RENAMED_SLUG}.json"
    base_path.write_text(json.dumps(base_cfg), encoding="utf-8")
    renamed_path.write_text(json.dumps(renamed_cfg), encoding="utf-8")

    spine = PredictionSpine()
    base_res = _evaluate_synthetic_config(base_path, sku, phase, spine)
    renamed_res = _evaluate_synthetic_config(renamed_path, sku, phase, spine)

    _assert_calibrated_and_conserved(base_res, arch_key)
    _assert_calibrated_and_conserved(renamed_res, arch_key)

    assert base_res["step_eval"].span_ms == renamed_res["step_eval"].span_ms
    assert base_res["step_eval"].native_steps == renamed_res["step_eval"].native_steps
    assert base_res["step_eval"].streams == renamed_res["step_eval"].streams
    assert base_res["exec_graph_signature"] == renamed_res["exec_graph_signature"]
    assert base_res["op_signature"] == renamed_res["op_signature"]
    assert base_res["total_flops"] == renamed_res["total_flops"]
    assert base_res["total_hbm_bytes"] == renamed_res["total_hbm_bytes"]

    base_die = base_res["receipt"].dies[0]
    renamed_die = renamed_res["receipt"].dies[0]
    assert base_die.checkpoint_weight_bytes == renamed_die.checkpoint_weight_bytes
    assert base_die.prepared_weight_bytes == renamed_die.prepared_weight_bytes
    assert base_die.kv_cache_allocated_bytes == renamed_die.kv_cache_allocated_bytes
    assert base_die.kv_cache_valid_bytes == renamed_die.kv_cache_valid_bytes
    assert base_die.recurrent_state_bytes == renamed_die.recurrent_state_bytes
    assert base_die.simultaneous_live_hbm_bytes == renamed_die.simultaneous_live_hbm_bytes
    assert base_die.simultaneous_peak_hbm_bytes == renamed_die.simultaneous_peak_hbm_bytes
    assert base_die.free_hbm_headroom_bytes == renamed_die.free_hbm_headroom_bytes


def test_renamed_packaged_configs_preserve_structural_geometry_and_weights(
    tmp_path: Path,
) -> None:
    """Renaming dual-sublayer and sandwich-sliding HF configs preserves exact plans and bytes."""
    lc_orig = REPO_ROOT / "mindie_sim" / "model_configs" / "longcat-flash-lite.json"
    lc_raw = json.loads(lc_orig.read_text(encoding="utf-8"))
    lc_ren = {
        **lc_raw,
        "model_type": "aurora_dual_block",
        "name": "aurora-dual-block",
        "model_key": "aurora-dual-block",
        "family": "aurora",
    }
    lc_ren_path = tmp_path / "aurora-dual-block.json"
    lc_ren_path.write_text(json.dumps(lc_ren), encoding="utf-8")

    lc_case_a = parse_case(compose_case(lc_orig, hardware=GEN5_DT_SMALL_SKU, tp=1, dp=1, ep=1, num_devices=1))
    lc_case_b = parse_case(compose_case(lc_ren_path, hardware=GEN5_DT_SMALL_SKU, tp=1, dp=1, ep=1, num_devices=1))
    assert len(lc_case_a.model.layer_plans) == len(lc_case_b.model.layer_plans) == 28
    assert lc_case_a.model.layer_plans == lc_case_b.model.layer_plans
    lc_mem_a = Memory(lc_case_a).resident_memory_receipt(phase="decode", sequence_tokens=128, batch_per_dp=1)
    lc_mem_b = Memory(lc_case_b).resident_memory_receipt(phase="decode", sequence_tokens=128, batch_per_dp=1)
    assert lc_mem_a.dies[0].checkpoint_weight_bytes == lc_mem_b.dies[0].checkpoint_weight_bytes

    g4_orig = REPO_ROOT / "mindie_sim" / "catalog" / "configs" / "gemma-4-31b.json"
    g4_raw = json.loads(g4_orig.read_text(encoding="utf-8"))
    g4_ren = {
        **g4_raw,
        "model_type": "aurora_sandwich",
        "architectures": ["AuroraSandwichForConditionalGeneration"],
        "text_config": {**g4_raw["text_config"], "model_type": "aurora_sandwich_text"},
    }
    g4_ren_path = tmp_path / "aurora-sandwich.json"
    g4_ren_path.write_text(json.dumps(g4_ren), encoding="utf-8")

    g4_case_a = parse_case(compose_case(g4_orig, hardware=GEN5_DT_SMALL_SKU, tp=1, dp=1, ep=1, num_devices=1))
    g4_case_b = parse_case(compose_case(g4_ren_path, hardware=GEN5_DT_SMALL_SKU, tp=1, dp=1, ep=1, num_devices=1))
    assert g4_case_a.model.layer_plans == g4_case_b.model.layer_plans
    assert max(p.head_dim for p in g4_case_b.model.layer_plans) == 512
    g4_mem_a = Memory(g4_case_a).resident_memory_receipt(phase="decode", sequence_tokens=128, batch_per_dp=1)
    g4_mem_b = Memory(g4_case_b).resident_memory_receipt(phase="decode", sequence_tokens=128, batch_per_dp=1)
    assert g4_mem_a.dies[0].checkpoint_weight_bytes == g4_mem_b.dies[0].checkpoint_weight_bytes


def _normalize_renamed_text(text: str) -> str:
    return (
        text.replace(OMEGA_RENAMED_SLUG, ZETA_HYBRID_SLUG)
        .replace(OMEGA_RENAMED_MODEL_TYPE, ZETA_HYBRID_MODEL_TYPE)
        .replace(OMEGA_RENAMED_ARCH_CLASS, ZETA_HYBRID_ARCH_CLASS)
    )


def _assert_zero_uncalibrated_and_no_envelope_fallback(
    case: Any,
    step_eval: Any,
    executable_ops: tuple[Any, ...],
) -> None:
    assert step_eval.primitive_table is not None
    assert step_eval.primitive_table.provenance.calibrated is True
    assert bool(step_eval.primitive_table.provenance.campaign_id)
    assert step_eval.primitive_table.warnings == ()
    assert not any(
        "uncalibrated" in str(val).lower()
        for val in dict(case.runtime.metadata).values()
    )
    for row in step_eval.operators:
        impl_text = str(row.implementation or "").lower()
        assert "uncalibrated" not in impl_text
        assert "resourceenvelope" not in impl_text and "resource_envelope" not in impl_text
    for op in executable_ops:
        recipe_text = str(op.recipe or "").lower()
        assert "uncalibrated" not in recipe_text
        assert "resourceenvelope" not in recipe_text and "resource_envelope" not in recipe_text


def _evaluate_multi_die_config(
    config_path: Path, sku: str, spine: PredictionSpine,
) -> dict[str, Any]:
    raw_case = compose_case(
        config_path,
        hardware=sku,
        tp=DIST_SHARD_DEGREE,
        dp=1,
        ep=DIST_SHARD_DEGREE,
        num_devices=DIST_SHARD_DEGREE,
        sp=True,
        concurrency=DIST_UNALIGNED_SEQS,
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )
    case = parse_case(raw_case)
    request = LoweringRequest(
        step=StepRequest(
            phase="decode",
            tokens=DECODE_TOKENS,
            context=CONTEXT_TOKENS,
            sequences=DIST_UNALIGNED_SEQS,
            scope="body",
        )
    )
    prepared = spine.lower(case, request)
    step_eval = spine.run_step(prepared, ReplayOptions(isolated=True))
    views = prepared.graph.request_views[0]
    receipt = Memory(case).resident_memory_receipt(phase="decode", request_views=views)
    executable_ops = tuple(op for entry in prepared.graph.executable for op in entry.ops)
    exec_graph_signature = tuple(
        (op.op_id, op.deps, op.family, op.recipe, op.flops, op.hbm_bytes)
        for op in executable_ops
    )
    op_signature = tuple(
        (
            row.occurrence.op_id,
            row.ordinal,
            row.family,
            row.implementation,
            tuple(pred.op_id for pred in row.predecessors),
            row.stream,
            row.shape,
            row.dtype,
            row.ready_ms,
            row.start_ms,
            row.end_ms,
            row.isolated_ms,
            row.work,
        )
        for row in step_eval.operators
    )
    return {
        "case": case,
        "step_eval": step_eval,
        "receipt": receipt,
        "executable_ops": executable_ops,
        "exec_graph_signature": exec_graph_signature,
        "op_signature": op_signature,
        "total_flops": sum(float(op.flops) for op in executable_ops),
        "total_hbm_bytes": sum(float(op.hbm_bytes) for op in executable_ops),
    }


@pytest.mark.parametrize("sku", MULTI_DIE_TRI_GEN_SKUS)
def test_unseen_hybrid_multi_die_sharding_generalizes_and_is_name_invariant(
    tmp_path: Path, sku: str,
) -> None:
    """Tri-generation tp=4, ep=4 distributed sharding (SP PadV3, LocalTP, MoE) on unseen hybrid."""
    renamed_cfg = {
        **ZETA_HYBRID_DIST_CONFIG,
        "model_type": OMEGA_RENAMED_MODEL_TYPE,
        "architectures": [OMEGA_RENAMED_ARCH_CLASS],
    }
    base_path = tmp_path / f"{ZETA_HYBRID_SLUG}.json"
    renamed_path = tmp_path / f"{OMEGA_RENAMED_SLUG}.json"
    base_path.write_text(json.dumps(ZETA_HYBRID_DIST_CONFIG), encoding="utf-8")
    renamed_path.write_text(json.dumps(renamed_cfg), encoding="utf-8")

    spine = PredictionSpine()
    base_res = _evaluate_multi_die_config(base_path, sku, spine)
    renamed_res = _evaluate_multi_die_config(renamed_path, sku, spine)

    for res in (base_res, renamed_res):
        case = res["case"]
        step_eval = res["step_eval"]
        executable_ops = res["executable_ops"]
        receipt = res["receipt"]

        assert len(case.machine.dies) == DIST_SHARD_DEGREE
        assert (case.runtime.parallel.tp, case.runtime.parallel.ep) == (
            DIST_SHARD_DEGREE,
            DIST_SHARD_DEGREE,
        )
        if sku == GEN2_SKU:
            assert case.machine.dies_per_package == 1
        elif sku == DUAL_DIE_SKU:
            assert case.machine.dies_per_package == 2
            assert {die.package for die in case.machine.dies} == {0, 1}
            assert any("intra_package" in link.name for link in case.machine.links)
        else:
            assert case.machine.dies[0].compute_arch.preset == "npu_arch_3510"

        assert step_eval.accel_backend == "c_native"
        assert step_eval.native_steps > 0
        assert math.isfinite(step_eval.span_ms) and step_eval.span_ms > 0.0
        _assert_zero_uncalibrated_and_no_envelope_fallback(case, step_eval, executable_ops)

        op_families = {op.family for op in executable_ops}
        assert "pad" in op_families or any("pad" in op.op_id for op in executable_ops)
        assert {"allgather", "reduce_scatter"} <= op_families
        assert {"grouped_gemm", "router"} <= op_families
        assert any(
            op.family in {"alltoall", "alltoallv", "mc2_protocol", "mc2_transport"}
            or "dispatch" in op.recipe
            or "combine" in op.recipe
            for op in executable_ops
        )
        assert {"gdn_gating", "short_convolution", "attention", "rotary"} <= op_families

        max_end_ms = max(row.end_ms for row in step_eval.operators)
        assert round(max_end_ms * NS_PER_MS) == round(step_eval.span_ms * NS_PER_MS)
        assert len(receipt.dies) == DIST_SHARD_DEGREE
        for die in receipt.dies:
            fixed_bytes = (
                die.checkpoint_weight_bytes
                + die.prepared_weight_bytes
                + die.vision_weight_bytes
                + die.workspace_bytes
                + die.reserve_bytes
            )
            dynamic_bytes = (
                die.kv_cache_allocated_bytes
                + die.index_cache_allocated_bytes
                + die.recurrent_state_bytes
                + die.draft_cache_bytes
                + die.staging_bytes
                + die.dynamic_grant_bytes
            )
            assert die.checkpoint_weight_bytes > 0
            assert die.recurrent_state_bytes > 0
            assert die.simultaneous_live_hbm_bytes == fixed_bytes + dynamic_bytes
            assert die.free_hbm_headroom_bytes == (
                die.usable_hbm_bytes - die.simultaneous_peak_hbm_bytes
            )

    assert round(base_res["step_eval"].span_ms * NS_PER_MS) == round(
        renamed_res["step_eval"].span_ms * NS_PER_MS
    )
    assert base_res["step_eval"].span_ms == renamed_res["step_eval"].span_ms
    assert base_res["step_eval"].streams == renamed_res["step_eval"].streams
    assert base_res["exec_graph_signature"] == renamed_res["exec_graph_signature"]
    assert base_res["op_signature"] == renamed_res["op_signature"]
    assert base_res["total_flops"] == renamed_res["total_flops"]
    assert base_res["total_hbm_bytes"] == renamed_res["total_hbm_bytes"]
    for base_die, ren_die in zip(base_res["receipt"].dies, renamed_res["receipt"].dies):
        assert base_die.checkpoint_weight_bytes == ren_die.checkpoint_weight_bytes
        assert base_die.kv_cache_allocated_bytes == ren_die.kv_cache_allocated_bytes
        assert base_die.recurrent_state_bytes == ren_die.recurrent_state_bytes
        assert base_die.simultaneous_live_hbm_bytes == ren_die.simultaneous_live_hbm_bytes
        assert base_die.simultaneous_peak_hbm_bytes == ren_die.simultaneous_peak_hbm_bytes


def _export_unseen_model_deliverables(
    config_path: Path, sku: str, out_dir: Path, *, check_direct_receipt: bool = True,
) -> dict[str, Any]:
    raw_bef = compose_case(
        config_path,
        hardware=sku,
        tp=EXPORT_SHARD_DEGREE,
        dp=1,
        ep=EXPORT_SHARD_DEGREE,
        num_devices=EXPORT_SHARD_DEGREE,
        concurrency=EXPORT_CONCURRENCY,
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        feature_track="shipping",
    )
    raw_aft = compose_case(
        config_path,
        hardware=sku,
        tp=EXPORT_SHARD_DEGREE,
        dp=1,
        ep=EXPORT_SHARD_DEGREE,
        num_devices=EXPORT_SHARD_DEGREE,
        concurrency=EXPORT_CONCURRENCY,
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        feature_track="target",
    )
    req = StepRequest(
        phase="decode",
        tokens=DECODE_TOKENS,
        context=CONTEXT_TOKENS,
        sequences=EXPORT_CONCURRENCY,
        scope="model",
    )
    direct_receipt = (
        solve_target_step_receipt(raw_aft, req, max_trials=1)
        if check_direct_receipt else None
    )
    b_bef = build_export_bundle(raw_bef, request=req, apply_target_envelope=False)
    b_aft = build_export_bundle(
        raw_aft, request=req, apply_target_envelope=True, target_max_trials=1,
    )
    aligned = align_before_after_operators(b_bef, b_aft)
    plan = build_comparison_plan(b_bef, b_aft, aligned)

    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "comparison_report.xlsx"
    brief_path = out_dir / "agent_brief.json"
    trace_path = out_dir / "trace_view.json"
    digest_path = out_dir / "digest.md"

    export_comparison_xlsx(b_bef, xlsx_path, after_bundle=b_aft, comparison_plan=plan)
    brief = export_agent_brief(b_bef, b_aft, brief_path, comparison_plan=plan)
    export_msprof_trace(b_aft, trace_path)
    digest_path.write_text(format_executive_digest_markdown(brief), encoding="utf-8")

    return {
        "case_bef": parse_case(raw_bef),
        "direct_receipt": direct_receipt,
        "b_bef": b_bef,
        "b_aft": b_aft,
        "plan": plan,
        "xlsx_path": xlsx_path,
        "brief_path": brief_path,
        "trace_path": trace_path,
        "digest_path": digest_path,
    }


def _assert_export_invariants(bundle_out: dict[str, Any]) -> None:
    case_bef = bundle_out["case_bef"]
    direct_receipt = bundle_out["direct_receipt"]
    b_bef = bundle_out["b_bef"]
    b_aft = bundle_out["b_aft"]
    plan = bundle_out["plan"]
    xlsx_path: Path = bundle_out["xlsx_path"]
    brief_path: Path = bundle_out["brief_path"]
    trace_path: Path = bundle_out["trace_path"]
    digest_path: Path = bundle_out["digest_path"]

    assert isinstance(plan, ComparisonPlan)
    assert b_aft.engine_step_receipt is not None
    op_ids = {row.op_id for row in b_bef.operators}
    assert {"embedding", "final_norm", "logits"} <= op_ids
    for bun in (b_bef, b_aft):
        assert sum(row.exposed_ns for row in bun.operators) + bun.idle_or_wait_ns == bun.step_span_ns
        for row in bun.operators:
            assert "uncalibrated" not in str(row.ascend_kernel).lower()
            assert "resourceenvelope" not in str(row.implementation).lower()

    # Exact 0 ns 4-stage waterfall telescoping (Before -> Stage 1 -> Stage 2 -> Stage 3 == After).
    receipts = (
        (direct_receipt, b_aft.engine_step_receipt)
        if direct_receipt is not None else (b_aft.engine_step_receipt,)
    )
    for receipt in receipts:
        assert tuple(stage.stage_id for stage in receipt.stages) == EXPECTED_STAGE_IDS
        assert len(receipt.telescoping_deltas_ns) == len(EXPECTED_STAGE_IDS) - 1
        for idx in range(len(EXPECTED_STAGE_IDS) - 1):
            assert (
                receipt.stages[idx].span_ns - receipt.telescoping_deltas_ns[idx]
                == receipt.stages[idx + 1].span_ns
            )
        assert (
            receipt.stages[0].span_ns - sum(receipt.telescoping_deltas_ns)
            == receipt.stages[-1].span_ns
        )
    if direct_receipt is not None:
        assert [s.span_ns for s in direct_receipt.stages] == [
            s.span_ns for s in b_aft.engine_step_receipt.stages
        ]
    assert plan.waterfall.reconciliation_error_ns == 0
    assert plan.waterfall.before_step_ns == b_bef.step_span_ns
    assert plan.waterfall.after_step_ns == b_aft.step_span_ns

    # Exact 0 B Sheet 3 memory bijection against Memory.resident_memory_receipt.
    led = b_bef.memory_ledger
    assert led is not None
    ws_tok = _resolve_ledger_workspace_tokens(
        case_bef, phase="decode", resolved_seq=led.seq_len, resolved_conc=led.concurrency,
    )
    die0 = Memory(case_bef, workspace_tokens=ws_tok).resident_memory_receipt(
        phase="decode", sequence_tokens=led.seq_len, batch_per_dp=led.concurrency,
    ).die_receipt(0, 0, 0)
    sb = led.stage_budgets[0]
    assert int(sb.weight_payload_bytes + sb.weight_scale_aux_bytes + sb.engram_ple_hbm_bytes) == (
        die0.checkpoint_weight_bytes + die0.prepared_weight_bytes + die0.vision_weight_bytes
    )
    assert int(sb.cache_bytes) == (
        die0.kv_cache_allocated_bytes
        + die0.index_cache_allocated_bytes
        + die0.recurrent_state_bytes
        + die0.draft_cache_bytes
    )
    assert (
        int(sb.workspace_peak_bytes),
        int(sb.reserve_bytes),
        int(sb.total_hbm_bytes),
        int(sb.usable_hbm_budget_bytes),
    ) == (
        die0.workspace_bytes,
        die0.reserve_bytes,
        die0.simultaneous_live_hbm_bytes,
        die0.usable_hbm_bytes,
    )

    # Exact 4 visible .xlsx sheets (0 hidden) and Sheet 2 A:AA (27 columns).
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet_els = wb_root.findall(f".//{NS_PREFIX}sheet")
        assert tuple(el.attrib["name"] for el in sheet_els) == EXPECTED_VISIBLE_SHEETS
        assert all(el.attrib.get("state") is None for el in sheet_els)
        s2_root = ET.fromstring(zf.read("xl/worksheets/sheet2.xml"))
        s2_cols = {
            "".join(ch for ch in cell.attrib["r"] if ch.isalpha())
            for cell in s2_root.iter(f"{NS_PREFIX}c")
        }
        assert len(s2_cols) == EXPECTED_SHEET2_COL_COUNT
        assert max(s2_cols, key=lambda col: (len(col), col)) == EXPECTED_SHEET2_MAX_COL

    # All 4 exported deliverables exist and are structurally valid.
    brief_raw = brief_path.read_bytes()
    assert 0 < len(brief_raw) < AGENT_BRIEF_MAX_BYTES
    brief_obj = json.loads(brief_raw.decode("utf-8"))
    assert brief_obj["kpis_before_vs_after"]["waterfall"]["reconciliation_error_ms"] == 0.0
    trace_obj = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_events = trace_obj.get("traceEvents") if isinstance(trace_obj, dict) else trace_obj
    assert isinstance(trace_events, list) and len(trace_events) > 0
    assert len(digest_path.read_text(encoding="utf-8").strip()) > 0


@pytest.mark.parametrize("sku", EXPORT_TRI_GEN_SKUS)
def test_unseen_hybrid_model_scope_stage_ladder_and_four_file_export_invariance(
    tmp_path: Path, sku: str,
) -> None:
    """Tri-generation scope='model' + Stage 0..3 + 4-file export with bit-for-bit rename invariance."""
    renamed_cfg = {
        **ZETA_HYBRID_EXPORT_CONFIG,
        "model_type": OMEGA_RENAMED_MODEL_TYPE,
        "architectures": [OMEGA_RENAMED_ARCH_CLASS],
    }
    base_path = tmp_path / f"{ZETA_HYBRID_SLUG}.json"
    renamed_path = tmp_path / f"{OMEGA_RENAMED_SLUG}.json"
    base_path.write_text(json.dumps(ZETA_HYBRID_EXPORT_CONFIG), encoding="utf-8")
    renamed_path.write_text(json.dumps(renamed_cfg), encoding="utf-8")

    base_out = _export_unseen_model_deliverables(
        base_path, sku, tmp_path / "out_zeta", check_direct_receipt=True,
    )
    ren_out = _export_unseen_model_deliverables(
        renamed_path, sku, tmp_path / "out_omega", check_direct_receipt=False,
    )

    _assert_export_invariants(base_out)
    _assert_export_invariants(ren_out)

    base_receipt = base_out["b_aft"].engine_step_receipt
    ren_receipt = ren_out["b_aft"].engine_step_receipt
    assert [s.span_ns for s in base_receipt.stages] == [s.span_ns for s in ren_receipt.stages]
    assert base_receipt.telescoping_deltas_ns == ren_receipt.telescoping_deltas_ns
    assert base_out["plan"].waterfall == ren_out["plan"].waterfall
    assert _normalize_renamed_text(base_out["brief_path"].read_text(encoding="utf-8")) == (
        _normalize_renamed_text(ren_out["brief_path"].read_text(encoding="utf-8"))
    )
    assert _normalize_renamed_text(base_out["trace_path"].read_text(encoding="utf-8")) == (
        _normalize_renamed_text(ren_out["trace_path"].read_text(encoding="utf-8"))
    )
    assert _normalize_renamed_text(base_out["digest_path"].read_text(encoding="utf-8")) == (
        _normalize_renamed_text(ren_out["digest_path"].read_text(encoding="utf-8"))
    )
    with zipfile.ZipFile(base_out["xlsx_path"], "r") as za, zipfile.ZipFile(
        ren_out["xlsx_path"], "r"
    ) as zb:
        assert za.namelist() == zb.namelist()
        for member in za.namelist():
            assert _normalize_renamed_text(za.read(member).decode("utf-8")) == (
                _normalize_renamed_text(zb.read(member).decode("utf-8"))
            )
