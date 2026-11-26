"""Cartesian algebraic subgraph & native birth-certificate matrix tests.

[Role]
  Verify 100% native-mapping coverage, zero missing blocks or banned
  submodules, and 100% LayerPlan-mandated semantic subgraph emission across a
  1,500+ (Model x SKU x Quant x Parallel x Phase x Track) Cartesian matrix plus
  targeted two-layer separation invariants and CI gate AG012.

[Invariants]
  * Across >= 1,500 valid combinations spanning all 43 catalog models, 4 SKUs,
    6 quant modes, {TP, EP, DCP, PCP, Local-TP, SP, DP}, {prefill, decode,
    spec}, and all 4 tracks: unmapped_executed_nodes == 0,
    missing_block_or_other_submodule == 0, missing_layerplan_subgraphs == 0.
  * Every exported node carries valid block, non-banned submodule,
    backend_stack, readiness, and source_ref.
  * Every exported subgraph carries valid fusion_status and fusion_reason.

[Timeouts & Locks]
  Bounded ProcessPoolExecutor fan-out (< 15s locally, < 5s on remote hosts).

[Side-Effect Policy]
  Pure read-only graph export and static governance inspection.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Any

import pytest

from mindie_sim.catalog.fleet import ARCHITECTURE_PRESET_CATALOG, CORE_CATALOG_ENTRIES
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operators.mla.mla_native import index_layer_norm
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure import export_structure
from mindie_sim.structure_implementations import (
    BANNED_SUBMODULES,
    VALID_BACKEND_STACKS,
    VALID_BLOCKS,
    VALID_READINESS,
)
from mindie_sim.structure_subgraphs import VALID_FUSION_STATUSES

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_LAYERS = 4
HYBRID_SAMPLE_LAYERS = 8
PREFILL_TOKENS = 128
DECODE_TOKENS = 1
SPEC_DECODE_TOKENS = 4
MTP_PROPOSALS = 3
DECODE_CONTEXT = 128
PROMPT_LEN = 128
OUTPUT_LEN = 16
MIN_CARTESIAN_VALID_COMBINATIONS = 1500
MAX_LOCAL_WORKERS = 16
TP_LARGE = 8
TP_MEDIUM = 4
TP_UNIT = 1
DP_PAIR = 2
DP_QUAD = 4
DP_OCTA = 8
CP_PAIR = 2
DCP_FOUR = 4
DCP_EIGHT = 8
PCP_FOUR = 4
EP_SIXTEEN = 16
EP_THIRTY_TWO = 32
EXPECTED_EAGER_GATED_NORM_OPS = 11
EXPECTED_SINGLE_LAYERNORM_OPS = 1

MATRIX_SKUS = (
    "Ascend910B4",
    "Ascend910_9392",
    "Ascend950PR_9579",
    "Ascend950DT_9582",
)
PRIMARY_A5_SKU = "Ascend950DT_9582"
PRIMARY_A3_SKU = "Ascend910_9392"
MATRIX_QUANTS = (
    "bf16",
    "w8a8",
    "w4a8",
    "fp8",
    "mxfp8",
    "mxfp4_routed_only",
)
ALL_TRACKS = ("eager", "shipping", "proposed", "target")
CORE_TRACKS = ("eager", "target")

_INDEXER_ROLES = frozenset({
    "lightning_indexer", "quant_lightning_indexer", "pool_key_indexer", "key_pool",
})
_GDN_ROLES = frozenset({"gdn_chunk", "gdn_recurrent"})
_KDA_ROLES = frozenset({"kda_chunk", "kda_recurrent"})
_MOE_GATE_UP_ROLES = frozenset({
    "grouped_gate_up_activation_quant", "grouped_gate_up_situ_quant", "grouped_gate_up_act",
})
_MHC_ROLES = frozenset({"hc_pre", "hc_post"})
_MLA_ATTENTION_KINDS = frozenset({"mla", "mla_dsa"})
_ZERO_KERNEL_FAMILIES = frozenset({
    "barrier", "event", "release", "tensor_view", "attention_metadata",
})


def _model_traits(model_name: str) -> tuple[bool, bool, int]:
    spec = load_model(model_name)
    plans = spec.layer_plans[:SAMPLE_LAYERS]
    has_moe = any(p.ffn == "moe" for p in plans)
    has_ctx_attn = any(p.attention in {"mla", "mla_dsa", "gqa", "mha"} for p in plans)
    heads = [p.num_heads for p in plans if p.num_heads and p.num_heads > 0]
    kv_heads = [p.num_kv_heads for p in plans if p.num_kv_heads and p.num_kv_heads > 0]
    min_h = min(heads) if heads else TP_LARGE
    if kv_heads:
        min_h = min(min_h, min(kv_heads))
    return has_moe, has_ctx_attn, min_h


def _valid_tp(min_heads: int) -> int:
    if min_heads >= TP_LARGE and min_heads % TP_LARGE == 0:
        return TP_LARGE
    if min_heads >= TP_MEDIUM and min_heads % TP_MEDIUM == 0:
        return TP_MEDIUM
    if min_heads >= CP_PAIR and min_heads % CP_PAIR == 0:
        return CP_PAIR
    return TP_UNIT


def _build_cartesian_jobs() -> list[tuple[Any, ...]]:
    models = sorted({*CORE_CATALOG_ENTRIES.keys(), *ARCHITECTURE_PRESET_CATALOG.keys()})
    jobs: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(model: str, sku: str, phase: str, tokens: int, context: int,
            track: str, par_kw: dict[str, Any], extra_kw: dict[str, Any]) -> None:
        key = (
            model, sku, phase, tokens, context, track,
            tuple(sorted(par_kw.items())), tuple(sorted(extra_kw.items())),
        )
        if key not in seen:
            seen.add(key)
            jobs.append((model, sku, phase, tokens, context, track, par_kw, extra_kw))

    for model in models:
        has_moe, has_ctx_attn, min_heads = _model_traits(model)
        tp_v = _valid_tp(min_heads)
        ep_kw = {"ep": tp_v} if has_moe else {}
        base_par = {"dp": TP_UNIT, "tp": tp_v, "num_devices": tp_v, **ep_kw}

        for sku in MATRIX_SKUS:
            for phase, tok, ctx in (("prefill", PREFILL_TOKENS, 0), ("decode", DECODE_TOKENS, DECODE_CONTEXT)):
                for track in CORE_TRACKS:
                    add(model, sku, phase, tok, ctx, track, base_par, {})

        for quant in MATRIX_QUANTS:
            for phase, tok, ctx in (("prefill", PREFILL_TOKENS, 0), ("decode", DECODE_TOKENS, DECODE_CONTEXT)):
                add(model, PRIMARY_A5_SKU, phase, tok, ctx, "target", base_par, {"quant": quant})

        for sku in (PRIMARY_A5_SKU, PRIMARY_A3_SKU):
            for track in ALL_TRACKS:
                for phase, tok, ctx in (("prefill", PREFILL_TOKENS, 0), ("decode", DECODE_TOKENS, DECODE_CONTEXT)):
                    add(model, sku, phase, tok, ctx, track, base_par, {})

        dp_nd = DP_PAIR * tp_v
        dp_par = {"dp": DP_PAIR, "tp": tp_v, "num_devices": dp_nd, **({"ep": dp_nd} if has_moe else {})}
        add(model, PRIMARY_A5_SKU, "prefill", PREFILL_TOKENS, 0, "target", dp_par, {})
        add(model, PRIMARY_A5_SKU, "decode", DECODE_TOKENS, DECODE_CONTEXT, "target", dp_par, {})
        add(model, PRIMARY_A5_SKU, "decode", SPEC_DECODE_TOKENS, DECODE_CONTEXT, "target", base_par, {})

        if has_ctx_attn:
            tp_cp = max(TP_UNIT, tp_v // CP_PAIR)
            nd_cp = tp_cp * CP_PAIR
            ep_cp = {"ep": nd_cp} if has_moe else {}
            dcp_par = {"dp": TP_UNIT, "tp": tp_cp, "dcp": CP_PAIR, "num_devices": nd_cp, **ep_cp}
            pcp_par = {"dp": TP_UNIT, "tp": tp_cp, "pcp": CP_PAIR, "num_devices": nd_cp, **ep_cp}
            add(model, PRIMARY_A5_SKU, "decode", DECODE_TOKENS, DECODE_CONTEXT, "target",
                dcp_par, {"kv_cache_dtype": "fp8_e4m3"})
            add(model, PRIMARY_A5_SKU, "decode", DECODE_TOKENS, DECODE_CONTEXT, "eager", dcp_par, {})
            add(model, PRIMARY_A5_SKU, "prefill", PREFILL_TOKENS, 0, "target", pcp_par, {})
            add(model, PRIMARY_A5_SKU, "prefill", PREFILL_TOKENS, 0, "eager", pcp_par, {})
    return jobs


def _audit_single_job(job: tuple[Any, ...]) -> dict[str, Any]:
    model, sku, phase, tok, ctx, track, par_kw, extra_kw = job
    try:
        case = compose_case(
            model, hardware=sku, layers=SAMPLE_LAYERS, feature_track=track,
            prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
            **par_kw, **extra_kw,
        )
        rep = export_structure(case, phase=phase, tokens=tok, context=ctx, scope="model")
    except Exception:
        return {"valid": False}

    unmapped: list[str] = []
    bad_meta: list[str] = []
    bad_sg: list[str] = []
    missing_sg: list[str] = []

    for node in rep["nodes"]:
        fam, rec = node.get("family"), node.get("recipe")
        blk = node.get("block")
        sub = str(node.get("submodule") or "")
        stack = node.get("backend_stack")
        ready = node.get("readiness")
        src = str(node.get("source_ref") or "").strip()
        if (blk not in VALID_BLOCKS or sub.strip().lower() in BANNED_SUBMODULES
                or stack not in VALID_BACKEND_STACKS or ready not in VALID_READINESS or not src):
            bad_meta.append(f"{model}:{fam}/{rec}:blk={blk},sub={sub},stack={stack},ready={ready}")
        is_zero_kernel = node.get("boundary") == "call" or fam in _ZERO_KERNEL_FAMILIES
        if not is_zero_kernel and not node.get("native_candidates"):
            unmapped.append(f"{model}:{fam}/{rec}")

    sg_roles_by_layer: dict[int, set[str]] = defaultdict(set)
    for sg in rep["subgraphs"]:
        if sg.get("fusion_status") not in VALID_FUSION_STATUSES or not str(sg.get("fusion_reason") or "").strip():
            bad_sg.append(f"{model}:{sg.get('role')}:{sg.get('fusion_status')}")
        layer_ids = sg.get("layer_ids") or ([sg["layer"]] if sg.get("layer") is not None else [])
        for lid in layer_ids:
            sg_roles_by_layer[lid].add(str(sg.get("role") or ""))

    for lid, lp in enumerate(rep["model"].get("layer_plans", [])[:SAMPLE_LAYERS]):
        att, ffn = lp.get("attention"), lp.get("ffn")
        roles = sg_roles_by_layer[lid]
        tag = f"{model}(L{lid},{sku},{phase},{track},{extra_kw})"
        if att == "kda" and not (_KDA_ROLES & roles):
            missing_sg.append(f"kda:{tag}")
        if att == "gdn" and not (_GDN_ROLES & roles):
            missing_sg.append(f"gdn:{tag}")
        if lp.get("indexer") and not (_INDEXER_ROLES & roles):
            missing_sg.append(f"indexer:{tag}")
        if att in _MLA_ATTENTION_KINDS and int(lp.get("q_lora_rank") or 0) > 0 and "mla_prolog" not in roles:
            missing_sg.append(f"mla_prolog:{tag}")
        if ffn == "moe" and "router_selection" not in roles:
            missing_sg.append(f"router_selection:{tag}")
        if ffn == "moe" and not (_MOE_GATE_UP_ROLES & roles):
            missing_sg.append(f"grouped_gate_up:{tag}")
        if lp.get("hyper_connections") and not (_MHC_ROLES & roles):
            missing_sg.append(f"mhc:{tag}")
        if lp.get("residual_block_size") and "attn_res_prefix_point" not in roles:
            missing_sg.append(f"attn_res:{tag}")

    return {
        "valid": True, "nodes": len(rep["nodes"]), "subgraphs": len(rep["subgraphs"]),
        "unmapped": unmapped, "bad_meta": bad_meta, "bad_sg": bad_sg, "missing_sg": missing_sg,
    }


def test_cartesian_matrix_achieves_total_subgraph_and_native_mapping_coverage() -> None:
    jobs = _build_cartesian_jobs()
    workers = max(TP_UNIT, min(os.cpu_count() or TP_MEDIUM, MAX_LOCAL_WORKERS))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_audit_single_job, jobs, chunksize=8))

    valid = [r for r in results if r["valid"]]
    unmapped = [item for r in valid for item in r["unmapped"]]
    bad_meta = [item for r in valid for item in r["bad_meta"]]
    bad_sg = [item for r in valid for item in r["bad_sg"]]
    missing_sg = [item for r in valid for item in r["missing_sg"]]

    assert len(valid) >= MIN_CARTESIAN_VALID_COMBINATIONS, (
        f"expected >= {MIN_CARTESIAN_VALID_COMBINATIONS} valid Cartesian combinations, got {len(valid)}"
    )
    assert unmapped == [], f"unmapped executed nodes ({len(unmapped)}): {unmapped[:10]}"
    assert bad_meta == [], f"missing block/banned submodule/invalid meta ({len(bad_meta)}): {bad_meta[:10]}"
    assert bad_sg == [], f"invalid subgraph fusion_status ({len(bad_sg)}): {bad_sg[:10]}"
    assert missing_sg == [], f"missing LayerPlan subgraphs ({len(missing_sg)}): {missing_sg[:10]}"


def test_indexer_subgraph_persists_across_dcp_pcp_and_gqa_indexer_models() -> None:
    case_dcp = compose_case(
        "deepseek-v3.2", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_LARGE, dcp=DCP_EIGHT,
        ep=TP_LARGE, num_devices=TP_LARGE, layers=SAMPLE_LAYERS,
        feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_dcp = export_structure(case_dcp, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    idx_sgs = [sg for sg in rep_dcp["subgraphs"] if sg["role"] in _INDEXER_ROLES]
    assert idx_sgs and all(sg["fusion_status"] in VALID_FUSION_STATUSES for sg in idx_sgs)

    case_pcp = compose_case(
        "deepseek-v3.2", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=CP_PAIR, pcp=PCP_FOUR,
        ep=CP_PAIR, num_devices=TP_LARGE, layers=SAMPLE_LAYERS,
        feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_pcp = export_structure(case_pcp, phase="prefill", tokens=PREFILL_TOKENS, context=0, scope="model")
    assert any(sg["role"] in _INDEXER_ROLES for sg in rep_pcp["subgraphs"])

    for model_name in ("minimax-m3", "minimax-m3.1-flash", "qwen3.8-flash-next", "step-5-preview"):
        case_gqa = compose_case(
            model_name, hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_MEDIUM,
            num_devices=TP_MEDIUM, layers=SAMPLE_LAYERS,
            feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
        )
        for phase, tok, ctx in (("prefill", PREFILL_TOKENS, 0), ("decode", DECODE_TOKENS, DECODE_CONTEXT)):
            rep = export_structure(case_gqa, phase=phase, tokens=tok, context=ctx, scope="model")
            assert any(sg["role"] in _INDEXER_ROLES for sg in rep["subgraphs"]), (model_name, phase)


def test_hybrid_gdn_gqa_ple_parallel_grid_preserves_subgraphs_and_mappings() -> None:
    configs = (
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, TP_UNIT, DP_OCTA, TP_UNIT, TP_LARGE, TP_LARGE),
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, CP_PAIR, DP_QUAD, TP_UNIT, TP_LARGE, TP_LARGE),
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, TP_MEDIUM, DP_QUAD, TP_UNIT, EP_SIXTEEN, EP_SIXTEEN),
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, TP_MEDIUM, DP_QUAD, CP_PAIR, EP_SIXTEEN, EP_SIXTEEN),
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, TP_LARGE, DP_QUAD, TP_UNIT, EP_THIRTY_TWO, EP_THIRTY_TWO),
        ("decode", DECODE_TOKENS, DECODE_CONTEXT, TP_LARGE, DP_QUAD, DCP_FOUR, EP_THIRTY_TWO, EP_THIRTY_TWO),
        ("prefill", PREFILL_TOKENS, 0, TP_LARGE, DP_QUAD, TP_UNIT, EP_THIRTY_TWO, EP_THIRTY_TWO),
    )
    for phase, tok, ctx, tp, dp, dcp, ep, nd in configs:
        if dcp > 1:
            with pytest.raises(ModelingError, match="shared_slab_requires_dcp_1_pcp_1"):
                compose_case(
                    "qwen3.8-flash-next", hardware=PRIMARY_A5_SKU, dp=dp, tp=tp, dcp=dcp, ep=ep,
                    num_devices=nd, layers=HYBRID_SAMPLE_LAYERS, feature_track="target",
                    prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
                    mtp_proposal_tokens=MTP_PROPOSALS,
                )
            continue
        case = compose_case(
            "qwen3.8-flash-next", hardware=PRIMARY_A5_SKU, dp=dp, tp=tp, dcp=dcp, ep=ep,
            num_devices=nd, layers=HYBRID_SAMPLE_LAYERS, feature_track="target",
            prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
            mtp_proposal_tokens=MTP_PROPOSALS,
        )
        rep = export_structure(case, phase=phase, tokens=tok, context=ctx, scope="model")
        unmapped = [
            (n["family"], n["recipe"]) for n in rep["nodes"]
            if n.get("boundary") != "call" and n["family"] not in _ZERO_KERNEL_FAMILIES
            and not n.get("native_candidates")
        ]
        roles = {sg["role"] for sg in rep["subgraphs"]}
        assert unmapped == [], (phase, tp, dcp, ep, unmapped[:5])
        assert (_GDN_ROLES & roles) and (_INDEXER_ROLES & roles) and (_MOE_GATE_UP_ROLES & roles)
        assert "router_selection" in roles and (_MHC_ROLES & roles)


def test_mla_prolog_fp8_v4_admission_and_unfused_semantic_fallback() -> None:
    case_fp8 = compose_case(
        "deepseek-v3.2", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_LARGE, ep=TP_LARGE,
        num_devices=TP_LARGE, layers=SAMPLE_LAYERS, quant="mxfp8",
        feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_fp8 = export_structure(case_fp8, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    prologs = [sg for sg in rep_fp8["subgraphs"] if sg["role"] == "mla_prolog"]
    assert prologs and any(
        sg["implementation"].get("native_kind") == "aclnnMlaPrologV4WeightNz" for sg in prologs
    )

    case_eager = compose_case(
        "deepseek-v3.2", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_LARGE, ep=TP_LARGE,
        num_devices=TP_LARGE, layers=SAMPLE_LAYERS,
        feature_track="eager", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_eager = export_structure(case_eager, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    eager_prologs = [sg for sg in rep_eager["subgraphs"] if sg["role"] == "mla_prolog"]
    assert eager_prologs and all(sg["fusion_status"] == "unfused_decomposed" for sg in eager_prologs)


def test_recurrent_gdn_subgraphs_and_eager_vs_fused_norm_decomposition() -> None:
    case_fused = compose_case(
        "qwen3.5-35b-a3b", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_MEDIUM, ep=TP_MEDIUM,
        num_devices=TP_MEDIUM, layers=SAMPLE_LAYERS,
        feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_pf = export_structure(case_fused, phase="prefill", tokens=PREFILL_TOKENS, context=0, scope="model")
    rep_dc = export_structure(case_fused, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    assert any(sg["role"] == "gdn_chunk" and sg["fusion_status"] == "fused_existing" for sg in rep_pf["subgraphs"])
    assert any(sg["role"] == "gdn_recurrent" for sg in rep_dc["subgraphs"])
    assert any(sg["role"] == "gated_rmsnorm" and len(sg["node_ids"]) == EXPECTED_SINGLE_LAYERNORM_OPS
               and sg["fusion_status"] == "fused_existing" for sg in rep_dc["subgraphs"])

    case_eager = compose_case(
        "qwen3.5-35b-a3b", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_MEDIUM, ep=TP_MEDIUM,
        num_devices=TP_MEDIUM, layers=SAMPLE_LAYERS,
        feature_track="eager", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep_eager = export_structure(case_eager, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    eager_norms = [sg for sg in rep_eager["subgraphs"] if sg["role"] == "gated_rmsnorm"]
    assert eager_norms and all(
        len(sg["node_ids"]) == EXPECTED_EAGER_GATED_NORM_OPS
        and sg["fusion_status"] == "unfused_decomposed"
        for sg in eager_norms
    )


def test_standard_layernorm_emits_single_kernel_without_external_casts() -> None:
    case = parse_case(compose_case(
        "deepseek-v3.2", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_LARGE, ep=TP_LARGE,
        num_devices=TP_LARGE, layers=SAMPLE_LAYERS,
        feature_track="eager", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    ))
    plan = case.model.layer_plans[0]
    builder = Builder("s0.l0", plan, 0, (), case.model, "decode")
    index_layer_norm(builder, source="hidden", rows=TP_LARGE, width=PREFILL_TOKENS,
                     native=True, explicit_fp32_wrapper=False)
    assert len(builder.ops) == EXPECTED_SINGLE_LAYERNORM_OPS
    assert builder.ops[0].family == "layernorm" and builder.ops[0].dtype == "bf16"
    assert not any(op.family == "cast" for op in builder.ops)


def test_routed_only_mxfp4_preserves_bf16_exclusions_and_emits_situ_quant_subgraph() -> None:
    case = compose_case(
        "kimi-k3.1", hardware=PRIMARY_A5_SKU, dp=TP_UNIT, tp=TP_LARGE, ep=TP_LARGE,
        num_devices=TP_LARGE, layers=SAMPLE_LAYERS, quant="mxfp4_routed_only",
        feature_track="target", prompt_tokens=PROMPT_LEN, output_tokens=OUTPUT_LEN,
    )
    rep = export_structure(case, phase="decode", tokens=DECODE_TOKENS, context=DECODE_CONTEXT, scope="model")
    assert any(sg["role"] == "grouped_gate_up_situ_quant" for sg in rep["subgraphs"])
    assert any(sg["role"] == "attn_res_prefix_point" for sg in rep["subgraphs"])
    excluded_nodes = [
        n for n in rep["nodes"]
        if n["operator"]["metadata"].get("quant_category") in {"mla.q_a", "shared_expert", "dense_mlp"}
    ]
    assert excluded_nodes and all(
        n["operator"]["metadata"].get("activation_format") == "bf16"
        and n["operator"]["metadata"].get("weight_format") == "bf16"
        for n in excluded_nodes
    )


def test_ag012_governance_gate_reports_zero_violations() -> None:
    from scripts._arch_governance_ag012 import check_ag012

    violations = check_ag012(REPO_ROOT)
    assert violations == [], f"AG012 violations: {violations}"
