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

import pytest

from _support.axes import DUAL_DIE_SKU, GEN2_SKU, GEN5_DT_SMALL_SKU, PHASES
from mindie_sim import compose_case
from mindie_sim.model_catalog import CATALOG
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

PROMPT_TOKENS = 128
OUTPUT_TOKENS = 16
CONTEXT_TOKENS = 128
PREFILL_TOKENS = 128
DECODE_TOKENS = 1
NS_PER_MS = 1_000_000.0
RENAMED_SLUG = "totally-unknown-xyz-99b"
RENAMED_MODEL_TYPE = "totally_unknown_xyz_99b"
RENAMED_ARCH_CLASS = "TotallyUnknownXyz99BForCausalLM"

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
    for name in (*SYNTHETIC_ARCHITECTURES, RENAMED_SLUG):
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
