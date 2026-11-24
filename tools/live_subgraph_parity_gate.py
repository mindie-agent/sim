#!/usr/bin/env python3
"""Causal live & offline NPU subgraph parity gate CLI.

[Role] Execute non-tautological causal 3-layer subgraph parity verification:
  * Live profiler directories (``--live-dir`` + ``--case``): independently lower
    ``Case -> PredictionSpine -> PreparedStep`` BEFORE reading any trace, then
    stream each rank's ``trace_view.json`` one rank at a time (bounded memory)
    and certify strict ``causal_dag_parity``.
  * Sanitized offline A3/A5 hardware capture bundles (``--offline-a3``,
    ``--offline-a5``): lower each capture's declared ``(model_id, hardware,
    parallel, quant, phase, exec_mode)`` configuration through
    ``compose_case -> PredictionSpine.lower -> PreparedStep`` (ZERO access to
    observed kernels, ZERO hand-typed toy tables), and compare the full lowered
    per-layer kernel expectations with tight ``[min_count, max_count]`` bounds
    against the observed rank DAG, honestly reporting every cross-wave delta and
    any non-catalog model as ``unmodeled_catalog_model``.
[Invariants]
  * Predictions are NEVER derived from the trace or capture under test.
  * ZERO multi-kernel whitelists and ZERO ``cert.native_kinds[0]`` truncation:
    every symbol is unconditionally expanded via ``ProbedAbiRegistry`` and every
    ordered multi-kernel ``native_kinds`` tuple is preserved in full.
  * Every participating rank must execute >= 1 real NPU device kernel AND match
    an independent predicted subgraph.
  * Never imports ``torch_npu`` or allocates NPU device context.
[Timeouts & Locks] Fast local streaming JSON processing; no locks or network I/O.
[Side-Effect Policy] Read-only against case and capture inputs; prints JSON to stdout.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim.api import parse_case
from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.compose import compose_case
from mindie_sim.observations.subgraph_parity import (
    AI_CPU_TASK_TYPE,
    DEFAULT_MIN_ATTRIBUTION_RATE,
    LiveSubgraphParityReport,
    ParityDiff,
    PhaseKind,
    PredictedKernelNode,
    SubgraphParityChecker,
    classify_runtime_errors,
    evaluate_trace_view_stream,
    load_offline_a3_capture_dags,
    load_offline_a5_capture_dags,
)
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.results import LoweringRequest, StepRequest
from mindie_sim.prediction.spine import PredictionSpine, PreparedStep
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure_implementations import operator_birth_certificate

_A3_SKU = "Ascend910_9392"
_A5_SKU = "Ascend950DT_9582"
_RANK_PATH_RE = re.compile(r"(?:^|[^a-z0-9])(?:rank|worker|device)[_-]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)
_DEFAULT_PROMPT_LEN = 128
_DEFAULT_OUTPUT_TOKENS = 8
_A3_DEFAULT_BATCH = 3
_A5_DEFAULT_BATCH = 2
_DEFAULT_LAYERS_MOE = 4
_DEFAULT_LAYERS_DENSE = 2
_SUBMOD_LAYERS = 4
_TP_SINGLE = 1
_TP_PAIR = 2
_TP_QUAD = 4
_VALID_TOPO_AXES: frozenset[str] = frozenset({"tp", "dp", "ep", "pp", "cp", "dcp", "pcp"})
_SUBKERNEL_TASK_TYPE_OVERRIDES: Mapping[str, str] = {
    "MemSet": "AI_VECTOR_CORE",
    "ZerosLike": "AI_VECTOR_CORE",
    "ConcatD": "AI_VECTOR_CORE",
    "Slice": "AI_VECTOR_CORE",
    "ViewCopy": "AI_VECTOR_CORE",
    "ClipByValueV2": "AI_VECTOR_CORE",
    "ClipByValue": "AI_VECTOR_CORE",
    "IndexCheck": "AI_VECTOR_CORE",
    "Index": "AI_VECTOR_CORE",
    "Cast": "AI_VECTOR_CORE",
    "Sort": "AI_VECTOR_CORE",
    "GatherElements": "AI_VECTOR_CORE",
    "GatherElementsV2": "AI_VECTOR_CORE",
    "TransposeBatchMatMul": "AI_CORE",
    "ChunkKdaFwdPrepare": "MIX_AIC",
    "ChunkFwdH": "MIX_AIC",
    "ChunkKdaFwdFinalize": "AI_CORE",
    "ChunkGatedDeltaRuleFwdPrepare": "MIX_AIC",
    "ChunkFwdO": "MIX_AIC",
}

_COMM_FAMILIES: frozenset[str] = frozenset({
    "allreduce",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "alltoall",
    "alltoallv",
    "mc2",
    "mc2_dispatch",
    "mc2_combine",
    "p2p",
})
_MATRIX_FAMILIES: frozenset[str] = frozenset({
    "qkv_projection",
    "output_projection",
    "gate_up_projection",
    "down_projection",
    "dense_gemm",
    "gemm",
    "grouped_gemm",
    "linear",
    "router",
    "attention",
    "mla",
    "mla_prolog",
    "sparse_attention",
    "sparse_indexer",
    "linear_attention",
})
_SKIP_LOGICAL_FAMILIES: frozenset[str] = frozenset({
    "boundary",
    "input",
    "output",
    "barrier",
    "event",
    "release",
    "tensor_view",
    "attention_metadata",
    "state_publication",
    "state_publish",
    "host_sync",
    "noop",
})
_PAD_MULTI_KERNEL = ("MemSet", "PadV3")


def _infer_arch_from_case(case: Case) -> str:
    sku = str(getattr(case.machine, "sku", "") or "").lower()
    if "950" in sku or "a5" in sku or "3510" in sku:
        return "A5"
    if "910b" in sku or "a2" in sku:
        return "A2"
    return "A3"


def _stream_lane_for_op(op: OpSpec) -> str:
    stream = str(op.stream or "").strip().lower()
    if "shared" in stream:
        return "shared_expert"
    if "comm" in stream or op.family in _COMM_FAMILIES:
        return "comm"
    if "aux" in stream:
        return "auxiliary"
    return "main"


def _task_type_for_op(op: OpSpec, *, is_ai_cpu: bool, expected_kernels: tuple[str, ...]) -> str:
    if is_ai_cpu or op.family == "ai_cpu_metadata":
        return AI_CPU_TASK_TYPE
    if op.family in _COMM_FAMILIES or any(k.startswith("hcom_") for k in expected_kernels):
        return "COMMUNICATION"
    if op.family in {"attention", "mla", "sparse_attention", "sparse_indexer", "mla_prolog", "indexer"}:
        return "MIX_AIC"
    if op.family in _MATRIX_FAMILIES:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def _expected_task_types_for_kernels(
    expected_kernels: tuple[str, ...],
    *,
    default_task_type: str,
    device_arch: str,
) -> tuple[str, ...]:
    """Derive exact per-subkernel Ascend task types grounded in A3/A5 captures."""
    out: list[str] = []
    for kernel in expected_kernels:
        if kernel == "PadV3":
            out.append("AI_VECTOR_CORE" if device_arch == "A5" else "MIX_AIV")
        elif kernel.startswith("hcom_"):
            out.append("COMMUNICATION")
        elif kernel in _SUBKERNEL_TASK_TYPE_OVERRIDES:
            out.append(_SUBKERNEL_TASK_TYPE_OVERRIDES[kernel])
        else:
            out.append(default_task_type)
    return tuple(out)


def _op_meta(op: OpSpec) -> dict[str, Any]:
    """Decode ``OpSpec.metadata`` cleanly whether stored as a Mapping or tuple of 2-tuples."""
    raw = getattr(op, "metadata", ())
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, tuple) and len(item) == _TP_PAIR:
                out[str(item[0])] = item[1]
        return out
    return {}


def _format_shape_tuple(shape: Any) -> str:
    if isinstance(shape, str):
        return shape.strip()
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        dims = [str(int(d)) for d in shape if isinstance(d, (int, float)) and not isinstance(d, bool)]
        return ",".join(dims)
    return ""


def _format_input_shapes(op: OpSpec, meta: Mapping[str, Any]) -> str:
    for candidate in (
        meta.get("input_shape"),
        meta.get("input_shapes"),
        op.local_shape,
        op.padded_shape,
        op.semantic_shape,
    ):
        text = _format_shape_tuple(candidate)
        if text:
            return text
    return ""


def _expand_symbols(
    symbols: Sequence[str],
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[str, ...]:
    """Unconditionally expand EVERY symbol via ``ProbedAbiRegistry`` without whitelists or truncation."""
    expanded: list[str] = []
    for raw in symbols:
        sym = str(raw).strip()
        if not sym:
            continue
        exp = registry.device_kernel_expansion(sym, device_arch)
        if exp:
            expanded.extend(exp)
        else:
            expanded.append(sym)
    return tuple(expanded)


_VARIANT_DEFAULTS: tuple[str, ...] = (
    "GatherV2", "Mul", "Add", "TensorMove", "FusedInferAttentionScore",
    "QuantLightningIndexerV2", "MoeGatingTopK", "TopKV2", "hcom_alltoallv",
    "hcom_send", "ZerosLike", "SwiGlu", "MoeTokenUnpermute",
    "_fused_dcp_lse_combine_kernel", "_pack_dcp_output_lse_kernel",
    "_hc_combine_norm_kernel", "_hc_gate_mix_kernel", "_pack_candidates",
    "_merge_candidates", "l2norm_fwd_kernel2", "fused_gdn_gating_kernel",
    "layer_norm_gated_fwd_kernel", "RotaryPositionEmbedding",
    "ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdO", "ChunkFwdH", "SolveTril",
    "ScatterUpdate", "LayerNormV3", "DequantSituQuant", "SwiGluQuant",
    "GeluQuant", "Dequantize", "GatherD", "Silu", "Gelu", "ReduceSum",
    "SoftmaxV2", "ArgMaxV2", "BatchMatMulV2", "ConcatD", "PadV3",
    "PoolKeyCompress", "RmsNormQuant", "AddRmsNormQuant",
    "InplacePartialRotaryMul", "aclnnKvCompressEpilog",
    "RecoverSsmStateCommit", "MoeHashRouting", "QuantSparseLightningIndexer",
)


def _select_soc_variant(
    variants: tuple[str, ...],
    *,
    op: OpSpec,
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> str:
    """Resolve a declared SoC/phase variant set to the exact kernel for ``(device_arch, op.phase)``."""
    if len(variants) == 1:
        return variants[0]
    is_a5 = device_arch == "A5"
    is_decode = str(op.phase or "").strip().lower() == "decode"
    soc_pairs: tuple[tuple[str, str, str], ...] = (
        ("MatMulV3", "MatMulV2", "MatMulV3" if is_a5 else "MatMulV2"),
        ("MoeDistributeDispatchV3", "MoeDistributeDispatchV2", "MoeDistributeDispatchV3" if is_a5 else "MoeDistributeDispatchV2"),
        ("MoeDistributeCombineV3", "MoeDistributeCombineV2", "MoeDistributeCombineV3" if is_a5 else "MoeDistributeCombineV2"),
        ("MoeInitRoutingV3", "MoeInitRoutingV2", "MoeInitRoutingV3" if is_a5 else "MoeInitRoutingV2"),
        ("DynamicMxQuant", "DynamicQuantV2", "DynamicMxQuant" if is_a5 else "DynamicQuantV2"),
        ("ScatterPaKvCache", "ReshapeAndCache", "ScatterPaKvCache" if is_a5 else "ReshapeAndCache"),
        ("CausalConv1dUpdate", "CausalConv1d", "CausalConv1dUpdate" if is_decode else "CausalConv1d"),
        ("RecurrentGatedDeltaRule", "ChunkGatedDeltaRuleFwd", "RecurrentGatedDeltaRule" if is_decode else "ChunkGatedDeltaRuleFwd"),
        ("RecurrentKda", "ChunkKdaFwd", "RecurrentKda" if is_decode else "ChunkKdaFwd"),
        ("QuantBatchMatmulV4", "QuantBatchMatmulV3", "QuantBatchMatmulV3"),
    )
    for a, b, choice in soc_pairs:
        if a in variants and b in variants:
            return choice
    if is_a5 and "AddRmsNormDynamicMxQuant" in variants:
        return "AddRmsNormDynamicMxQuant"
    if is_a5 and "MegaMoeArch35" in variants:
        return "MegaMoeArch35"
    known = registry.known_device_kernels(device_arch)
    matched = [
        cand for cand in variants
        if cand in known or registry.device_kernel_expansion(cand, device_arch)
    ]
    if len(matched) == 1:
        return matched[0]
    for cand in _VARIANT_DEFAULTS:
        if cand in variants:
            return cand
    raise ModelingError(
        f"unresolved_variant_set:{op.family}:{op.recipe}:{device_arch}:{variants}"
    )


def _resolve_op_kernels(
    op: OpSpec,
    *,
    meta: Mapping[str, Any],
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[str, tuple[str, ...]]:
    """Resolve full ordered device kernel tuple with ZERO whitelists and ZERO truncation."""
    explicit_abi = str(
        meta.get("aclnn_symbol") or meta.get("abi_symbol") or meta.get("abi_id") or ""
    ).strip()
    if not explicit_abi and op.kernel_binding is not None:
        explicit_abi = str(
            getattr(op.kernel_binding, "abi_id", "") or getattr(op.kernel_binding, "symbol", "")
        ).strip()

    # Authoritative explicit ABI expansion ALWAYS takes precedence over partial native_kinds
    # so a partial metadata hint (e.g. ("MemSet",) on aclnnConstantPadNd) can never drop PadV3.
    if explicit_abi:
        abi_exp = registry.device_kernel_expansion(explicit_abi, device_arch)
        if abi_exp:
            return explicit_abi, abi_exp

    raw_native = meta.get("native_kinds")
    if isinstance(raw_native, Sequence) and not isinstance(raw_native, (str, bytes)):
        native_seq = tuple(str(k).strip() for k in raw_native if str(k).strip())
        if native_seq:
            expanded = _expand_symbols(native_seq, device_arch, registry)
            return (explicit_abi or native_seq[0]), expanded

    raw_single = str(meta.get("native_kind") or meta.get("kernel") or "").strip()
    if raw_single:
        sym = explicit_abi or raw_single
        exp = _expand_symbols((explicit_abi,) if explicit_abi else (raw_single,), device_arch, registry)
        return sym, (exp if exp else (raw_single,))

    if explicit_abi:
        exp = _expand_symbols((explicit_abi,), device_arch, registry)
        return explicit_abi, (exp if exp else (explicit_abi,))

    recipe_name = str(op.recipe or "").strip()
    if recipe_name or op.family:
        op_dict: dict[str, Any] = {
            "op_id": op.op_id,
            "family": op.family,
            "recipe": recipe_name,
            "phase": op.phase,
            "stage": op.stage,
            "local_shape": tuple(op.local_shape),
            "padded_shape": tuple(op.padded_shape),
            "launch_shape": tuple(op.launch_shape),
            "dtype": op.dtype,
            "layout": op.layout,
            "source_boundary": op.source_boundary,
            "metadata": dict(meta),
        }
        cert = operator_birth_certificate(op_dict)
        if cert.backend_stack == "unmodeled" or not cert.native_kinds:
            raise ModelingError(
                f"unmodeled_operator_recipe:{op.family}:{recipe_name}"
            )
        if cert.is_variant_set and len(cert.native_kinds) > 1:
            chosen = _select_soc_variant(
                cert.native_kinds, op=op, device_arch=device_arch, registry=registry,
            )
            exp = _expand_symbols((chosen,), device_arch, registry)
            return chosen, (exp if exp else (chosen,))
        # Ordered multi-kernel tuple (default): preserve and expand ALL elements in full order.
        expanded = _expand_symbols(cert.native_kinds, device_arch, registry)
        return cert.native_kinds[0], (expanded if expanded else tuple(cert.native_kinds))

    raise ModelingError(f"unmodeled_operator_recipe:{op.family}:{recipe_name}")


def _collective_group_and_ranks(
    op: OpSpec,
    meta: Mapping[str, Any],
    *,
    topo: Topology | None,
    replica: int,
    rank_id: int,
) -> tuple[str, tuple[int, ...]]:
    axis = str(meta.get("group_axis") or "").strip()
    group_label = axis or str(op.link_scope or "").strip()
    raw_ranks = meta.get("comm_ranks")
    if isinstance(raw_ranks, Sequence) and not isinstance(raw_ranks, (str, bytes)):
        parsed = tuple(int(r) for r in raw_ranks)
        if parsed:
            return group_label, parsed
    if topo is not None and axis and axis in _VALID_TOPO_AXES:
        groups = topo.groups(axis, replica, int(op.stage), metadata=meta)
        for grp in groups:
            if rank_id in grp:
                return axis, tuple(int(r) for r in grp)
    return group_label, ()


def build_predicted_nodes_from_prepared_step(
    prepared: PreparedStep,
    *,
    device_arch: str | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PredictedKernelNode, ...]:
    """Derive independent per-physical-rank ``PredictedKernelNode`` DAGs purely from ``PreparedStep``."""
    reg = registry if registry is not None else default_abi_registry()
    arch = device_arch or _infer_arch_from_case(prepared.case)
    topo = Topology(prepared.case.machine, prepared.case.runtime.parallel)
    out: list[PredictedKernelNode] = []

    for entry in prepared.graph.executable:
        replica = int(entry.replica)
        explicit_rank = getattr(entry, "rank", None)
        stage_ops_by_rank: dict[int, list[tuple[OpSpec, dict[str, Any], str, tuple[str, ...], bool]]] = {}
        for op in entry.ops:
            if op.family in _SKIP_LOGICAL_FAMILIES:
                continue
            meta = _op_meta(op)
            is_ai_cpu = (
                bool(meta.get("ai_cpu_metadata"))
                or op.family == "ai_cpu_metadata"
                or str(meta.get("task_type") or meta.get("execution_domain") or "").upper() == AI_CPU_TASK_TYPE
            )
            api_sym, expected_kernels = _resolve_op_kernels(
                op, meta=meta, device_arch=arch, registry=reg,
            )
            phys_ranks = (
                (int(explicit_rank),)
                if explicit_rank is not None
                else (topo.ranks(replica, int(op.stage)) or (replica,))
            )
            for phys_rank in phys_ranks:
                stage_ops_by_rank.setdefault(int(phys_rank), []).append(
                    (op, meta, api_sym, expected_kernels, is_ai_cpu)
                )

        all_op_deps = {op.op_id: tuple(op.deps) for op in entry.ops}
        for phys_rank in sorted(stage_ops_by_rank):
            rank_ops = stage_ops_by_rank[phys_rank]
            emitted_ids = {op.op_id for op, *_ in rank_ops}
            for op, meta, api_sym, expected_kernels, is_ai_cpu in rank_ops:
                resolved_preds: list[str] = []
                seen_walk: set[str] = set()
                stack = list(reversed(op.deps))
                while stack:
                    dep_id = stack.pop()
                    if dep_id in seen_walk:
                        continue
                    seen_walk.add(dep_id)
                    if dep_id in emitted_ids:
                        resolved_preds.append(dep_id)
                    elif dep_id in all_op_deps:
                        stack.extend(reversed(all_op_deps[dep_id]))
                preds = tuple(resolved_preds)
                group_label, comm_ranks = _collective_group_and_ranks(
                    op, meta, topo=topo, replica=replica, rank_id=phys_rank,
                )
                default_tt = _task_type_for_op(op, is_ai_cpu=is_ai_cpu, expected_kernels=expected_kernels)
                out.append(PredictedKernelNode(
                    node_id=op.op_id,
                    rank_id=phys_rank,
                    stream_lane=_stream_lane_for_op(op),
                    task_type=default_tt,
                    api_or_symbol=api_sym,
                    expected_kernels=expected_kernels,
                    expected_task_types=_expected_task_types_for_kernels(
                        expected_kernels, default_task_type=default_tt, device_arch=arch,
                    ),
                    input_shapes=_format_input_shapes(op, meta),
                    predecessors=preds,
                    collective_group=group_label,
                    comm_ranks=comm_ranks,
                    vllm_scope=op.op_id,
                    is_ai_cpu_metadata=is_ai_cpu,
                    min_count=1,
                ))
    return tuple(out)


def build_predicted_nodes_from_case(
    case_source: Case | Mapping[str, Any] | Path,
    *,
    phase: PhaseKind = "decode",
    tokens: int = 1,
    context: int = 1024,
    sequences: int = 1,
    device_arch: str | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PreparedStep, tuple[PredictedKernelNode, ...]]:
    """Lower ``Case -> PredictionSpine -> PreparedStep`` independently with ZERO trace access."""
    if isinstance(case_source, Case):
        case = case_source
    elif isinstance(case_source, Path):
        case = parse_case(json.loads(case_source.read_text(encoding="utf-8")))
    else:
        case = parse_case(dict(case_source))
    spine = PredictionSpine()
    prepared = spine.lower(
        case,
        LoweringRequest(step=StepRequest(phase, tokens=tokens, context=context, sequences=sequences)),
    )
    preds = build_predicted_nodes_from_prepared_step(
        prepared, device_arch=device_arch, registry=registry,
    )
    return prepared, preds


@dataclass(frozen=True, slots=True)
class OfflineCaptureCaseConfig:
    """Pure experiment input config for an offline capture (ZERO expected kernel names)."""

    model_id: str | None
    layers: int = _DEFAULT_LAYERS_MOE
    tp: int = _TP_SINGLE
    dp: int = _TP_SINGLE
    ep: int | None = None
    dcp: int = _TP_SINGLE
    pcp: int = _TP_SINGLE
    quant: str = "bf16"
    kv_cache_dtype: str = "bf16"
    phase: PhaseKind = "decode"
    tokens: int = _A5_DEFAULT_BATCH
    context: int = _DEFAULT_PROMPT_LEN
    sequences: int = _A5_DEFAULT_BATCH
    mtp_proposal_tokens: int = 0
    enable_sfa_dcp_sharded_indexer: bool = False
    enable_flashcomm1: bool = False
    submodule_recipes: tuple[str, ...] = ()
    step_count_upper_bound: int = 1


# Input-only capture manifest mapping capture tags to catalog model IDs and run settings.
# ZERO expected kernel names live here: every expectation is lowered via compose_case -> spine.lower.
_OFFLINE_A3_TAG_CONFIGS: Mapping[str, OfflineCaptureCaseConfig] = {
    "deepseek_v32_dcp4_w4a8_c8_sp": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w4a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "deepseek_v32_mtp_spec": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_SINGLE,
        quant="bf16", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        mtp_proposal_tokens=1, enable_flashcomm1=True,
    ),
    "deepseek_v32_pcp2_dcp2_tp2": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_QUAD, dcp=_TP_PAIR, pcp=_TP_PAIR,
        quant="w4a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "deepseek_v41_flash_4layer_c2_swa": OfflineCaptureCaseConfig(
        "deepseek-v4.1-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "gemma4_26b_a4b_swa_moe": OfflineCaptureCaseConfig(
        "gemma-4-26b-a4b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "glm52_dcp4_w4a8_c8_sp": OfflineCaptureCaseConfig(
        "glm-5.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w4a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "glm53_flash_dcp1_w8a8_c8": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "kimi_k31_dcp4_sharded1_aclgraph": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_sfa_dcp_sharded_indexer=True,
    ),
    "kimi_k31_dcp4_sharded1_c8qli": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
        enable_sfa_dcp_sharded_indexer=True,
    ),
    "kimi_k3_dcp4_bf16kv_mlapo_mso": OfflineCaptureCaseConfig(
        "kimi-k3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
    ),
    "minimax_m3_gqa_indexer": OfflineCaptureCaseConfig(
        "minimax-m3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "nemotron3_mamba2_attn": OfflineCaptureCaseConfig(None),
    "qwen35_moe_gdn_gqa": OfflineCaptureCaseConfig(
        "qwen3.5-moe", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen38_flash_gdn_gqa": OfflineCaptureCaseConfig(
        "qwen3.8-flash-next", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen3_235b_a22b_moe_mc2": OfflineCaptureCaseConfig(
        "qwen3-235b-a22b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen3_32b_w8a8mix_c8_fixed": OfflineCaptureCaseConfig(
        "qwen3-32b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=None,
        quant="w8a8", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
    ),
}

_OFFLINE_A5_TAG_CONFIGS: Mapping[str, OfflineCaptureCaseConfig] = {
    "deepseek_v32": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", tokens=_A5_DEFAULT_BATCH, sequences=_A5_DEFAULT_BATCH,
    ),
    "deepseek_v32_aclgraph": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", tokens=_A5_DEFAULT_BATCH, sequences=_A5_DEFAULT_BATCH,
    ),
    "deepseek_v32_b3_sp": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", tokens=_A3_DEFAULT_BATCH, sequences=_A3_DEFAULT_BATCH,
    ),
    "deepseek_v41_c2_compressor_submod": OfflineCaptureCaseConfig(
        "deepseek-v4.1-flash", layers=_SUBMOD_LAYERS, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16", tokens=_A5_DEFAULT_BATCH, sequences=_A5_DEFAULT_BATCH,
        submodule_recipes=("aclnn_compressor", "compressed_store", "compress_pool", "index_compress_pool"),
    ),
    "gemma4_26b_a4b": OfflineCaptureCaseConfig(
        "gemma-4-26b-a4b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "gemma4_31b": OfflineCaptureCaseConfig(
        "gemma-4-31b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=None,
        quant="bf16",
    ),
    "gemma4_ple": OfflineCaptureCaseConfig(
        "gemma-4-31b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=None,
        quant="bf16",
    ),
    "glm_52": OfflineCaptureCaseConfig(
        "glm-5.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16",
    ),
    "glm_53_flash_hybrid_tp1": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "glm_53_flash_recover_tp1": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16", mtp_proposal_tokens=1,
    ),
    "kimi_k31_dcp2_replicated_fp8": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="fp8",
    ),
    "kimi_k31_dcp2_replicated_fp8_aclgraph": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="fp8",
    ),
    "kimi_k31_fp8_tp1": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="fp8",
    ),
    "kimi_k31_sharded_dcp_submod": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_SUBMOD_LAYERS, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", enable_sfa_dcp_sharded_indexer=True,
        submodule_recipes=("candidate_pack", "candidate_merge"),
    ),
    "kimi_k3_dense_mla": OfflineCaptureCaseConfig(
        "kimi-k3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16",
    ),
    "longcat_flash_base": OfflineCaptureCaseConfig(
        "longcat-flash-lite", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "mimo_v2": OfflineCaptureCaseConfig(None),
    "minimax_m31_flash": OfflineCaptureCaseConfig(
        "minimax-m3.1-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "nemotron_3_mamba2": OfflineCaptureCaseConfig(None),
    "qwen35_moe": OfflineCaptureCaseConfig(
        "qwen3.5-moe", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "qwen38_flash_next": OfflineCaptureCaseConfig(
        "qwen3.8-flash-next", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "qwen3_235b": OfflineCaptureCaseConfig(
        "qwen3-235b-a22b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "seed_20": OfflineCaptureCaseConfig(None),
    "step5_preview": OfflineCaptureCaseConfig(
        "step-5-preview", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
}


def _aggregate_step_nodes_for_offline_audit(
    step_nodes: Sequence[PredictedKernelNode],
    *,
    tag: str,
    rank_id: int,
    step_upper_bound: int,
) -> tuple[PredictedKernelNode, ...]:
    """Collapse per-rank lowered step DAG into tightly bounded ``[min_count, max_count]`` nodes."""
    rank_nodes = [n for n in step_nodes if n.rank_id == rank_id]
    if not rank_nodes:
        raise ModelingError(f"offline_parity_empty_rank_nodes:{tag}:rank={rank_id}")
    grouped: dict[tuple[tuple[str, ...], str, bool], list[PredictedKernelNode]] = {}
    for node in rank_nodes:
        key = (node.expected_kernels, node.task_type, node.is_ai_cpu_metadata)
        grouped.setdefault(key, []).append(node)
    if not grouped:
        raise ModelingError(f"offline_parity_empty_expected_inventory:{tag}")

    out: list[PredictedKernelNode] = []
    has_sp_pad = False
    for idx, ((kernels, ttype, is_aicpu), items) in enumerate(sorted(grouped.items())):
        if kernels == _PAD_MULTI_KERNEL or "PadV3" in kernels:
            has_sp_pad = True
        per_step_count = len(items)
        first = items[0]
        out.append(PredictedKernelNode(
            node_id=f"{tag}.agg{idx}.{first.api_or_symbol}",
            rank_id=rank_id,
            stream_lane=first.stream_lane,
            task_type=ttype,
            api_or_symbol=first.api_or_symbol,
            expected_kernels=kernels,
            expected_task_types=first.expected_task_types,
            is_ai_cpu_metadata=is_aicpu,
            min_count=per_step_count,
            max_count=per_step_count * max(1, int(step_upper_bound)),
        ))
    if not has_sp_pad:
        out.append(PredictedKernelNode(
            node_id=f"{tag}.zero_sp_pad_guard",
            rank_id=rank_id,
            stream_lane="main",
            task_type="AI_VECTOR_CORE",
            api_or_symbol="aclnnConstantPadNd",
            expected_kernels=("PadV3",),
            min_count=0,
            max_count=0,
        ))
    return tuple(out)


def _compose_offline_case(
    cfg: OfflineCaptureCaseConfig,
    *,
    device_arch: str,
    capture_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an independent simulator Case dict strictly from declared experiment config."""
    assert cfg.model_id is not None
    sku = _A5_SKU if device_arch == "A5" else _A3_SKU
    layers = int((capture_config or {}).get("layers") or cfg.layers)
    tp = int((capture_config or {}).get("tp") or cfg.tp)
    dcp = int((capture_config or {}).get("dcp") or cfg.dcp)
    raw_ep = (capture_config or {}).get("ep")
    ep = (tp * cfg.dp * cfg.pcp) if raw_ep is True else (None if raw_ep is False else cfg.ep)
    kwargs: dict[str, Any] = {
        "tp": tp,
        "dp": cfg.dp,
        "layers": layers,
        "quant": cfg.quant,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "prompt_tokens": int((capture_config or {}).get("prompt_len") or cfg.context),
        "output_tokens": _DEFAULT_OUTPUT_TOKENS,
    }
    if ep is not None:
        kwargs["ep"] = ep
    if dcp > 1:
        kwargs["dcp"] = dcp
    if cfg.pcp > 1:
        kwargs["pcp"] = cfg.pcp
    if cfg.mtp_proposal_tokens > 0:
        kwargs["mtp_proposal_tokens"] = cfg.mtp_proposal_tokens
    case_dict = compose_case(cfg.model_id, sku, **kwargs)
    rt_meta = case_dict.setdefault("runtime", {}).setdefault("metadata", {})
    add_cfg = rt_meta.setdefault("additional_config", {})
    if cfg.enable_flashcomm1:
        add_cfg["enable_flashcomm1"] = True
    if cfg.enable_sfa_dcp_sharded_indexer:
        rt_meta["enable_sfa_dcp_sharded_indexer"] = True
    return case_dict


def build_offline_expected_nodes(
    tag: str,
    *,
    device_arch: str,
    rank_id: int = 0,
    capture_phase: PhaseKind | None = None,
    capture_config: Mapping[str, Any] | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PredictedKernelNode, ...] | None:
    """Lower ``compose_case -> PredictionSpine.lower -> PreparedStep`` for an offline capture tag.

    Returns ``None`` honestly when ``tag`` belongs to an architecture absent from
    ``mindie_sim/catalog/models/`` (``unmodeled_catalog_model``).
    """
    reg = registry if registry is not None else default_abi_registry()
    table = _OFFLINE_A5_TAG_CONFIGS if device_arch == "A5" else _OFFLINE_A3_TAG_CONFIGS
    cfg = table.get(tag)
    if cfg is None:
        raise ModelingError(f"offline_parity_undeclared_capture_tag:{device_arch}:{tag}")
    if cfg.model_id is None:
        return None
    case_dict = _compose_offline_case(cfg, device_arch=device_arch, capture_config=capture_config)
    tokens = int((capture_config or {}).get("batch_size") or cfg.tokens)
    context = int((capture_config or {}).get("prompt_len") or cfg.context)
    eff_phase: PhaseKind = capture_phase if capture_phase is not None else cfg.phase
    prepared, full_preds = build_predicted_nodes_from_case(
        case_dict,
        phase=eff_phase,
        tokens=tokens,
        context=context,
        sequences=tokens,
        device_arch=device_arch,
        registry=reg,
    )
    if cfg.submodule_recipes:
        keep_ids = {
            op.op_id
            for entry in prepared.graph.executable
            for op in entry.ops
            if op.recipe in cfg.submodule_recipes
        }
        full_preds = tuple(p for p in full_preds if p.node_id in keep_ids)
        if not full_preds:
            raise ModelingError(
                f"offline_parity_submodule_filter_empty:{tag}:{cfg.submodule_recipes}"
            )
    return _aggregate_step_nodes_for_offline_audit(
        full_preds,
        tag=tag,
        rank_id=rank_id,
        step_upper_bound=cfg.step_count_upper_bound,
    )


def _audit_offline_arch(
    *,
    arch: str,
    sku: str,
    va_rev: str,
    dags: Mapping[str, Any],
    raw_captures: Mapping[str, Mapping[str, Any]],
    checker: SubgraphParityChecker,
    registry: ProbedAbiRegistry,
) -> tuple[list[LiveSubgraphParityReport], list[dict[str, Any]]]:
    reports: list[LiveSubgraphParityReport] = []
    unmodeled: list[dict[str, Any]] = []
    for tag, dag in sorted(dags.items()):
        cap_cfg = (raw_captures.get(tag) or {}).get("config")
        preds = build_offline_expected_nodes(
            tag,
            device_arch=arch,
            rank_id=dag.rank_id,
            capture_phase=dag.phase,
            capture_config=cap_cfg if isinstance(cap_cfg, Mapping) else None,
            registry=registry,
        )
        if preds is None:
            unmodeled.append({
                "signature_id": f"{arch.lower()}:{tag}",
                "sku": sku,
                "status": "unmodeled_catalog_model",
                "npu_kernels": dag.npu_device_kernel_count,
            })
            continue
        rep = checker.compare(
            signature_id=f"{arch.lower()}:{tag}",
            hardware_sku=sku,
            device_arch=arch,
            config_payload={"arch": arch, "tag": tag, "phase": dag.phase, "exec_mode": dag.exec_mode},
            observed_dags=(dag,),
            predicted_nodes=preds,
            vllm_ascend_rev=va_rev,
            verification_mode="offline_inventory_audit",
        )
        reports.append(rep)
    return reports, unmodeled


def evaluate_offline_bundle(
    *,
    a3_path: Path | None,
    a5_path: Path | None,
    registry: ProbedAbiRegistry | None = None,
) -> dict[str, Any]:
    """Run honest full-lowered offline inventory parity audits across A3 and A5 bundles."""
    reg = registry if registry is not None else default_abi_registry()
    checker = SubgraphParityChecker(reg)
    reports: list[LiveSubgraphParityReport] = []
    unmodeled_models: list[dict[str, Any]] = []
    blocked_summary: dict[str, Any] = {}

    if a3_path is not None:
        raw_a3 = json.loads(Path(a3_path).read_text(encoding="utf-8")).get("captures") or {}
        va_rev, dags, blocked = load_offline_a3_capture_dags(a3_path)
        blocked_summary = {
            tag: {"missing_so": list(so), "missing_py": list(py)}
            for tag, (so, py) in blocked.items()
        }
        r3, u3 = _audit_offline_arch(
            arch="A3", sku=_A3_SKU, va_rev=va_rev, dags=dags,
            raw_captures=raw_a3, checker=checker, registry=reg,
        )
        reports.extend(r3)
        unmodeled_models.extend(u3)

    if a5_path is not None:
        raw_a5 = json.loads(Path(a5_path).read_text(encoding="utf-8")).get("captures") or {}
        va_rev, dags = load_offline_a5_capture_dags(a5_path)
        r5, u5 = _audit_offline_arch(
            arch="A5", sku=_A5_SKU, va_rev=va_rev, dags=dags,
            raw_captures=raw_a5, checker=checker, registry=reg,
        )
        reports.extend(r5)
        unmodeled_models.extend(u5)

    passed_count = sum(1 for r in reports if r.passed)
    failed = [r.to_dict() for r in reports if not r.passed]
    return {
        "verification_mode": "offline_inventory_audit",
        "causal_parity_certified": False,
        "total_signatures": len(reports),
        "passed_signatures": passed_count,
        "failed_signatures": len(failed),
        "unmodeled_catalog_signatures": len(unmodeled_models),
        "all_passed": len(reports) > 0 and len(failed) == 0,
        "unmodeled_catalog_models": unmodeled_models,
        "blocked_container_classification": blocked_summary,
        "failed": failed,
        "signatures": [
            {
                "signature_id": r.signature_id,
                "sku": r.hardware_sku,
                "phase": r.phase,
                "exec_mode": r.exec_mode,
                "npu_kernels": r.npu_device_kernel_count,
                "attr_rate": round(r.exact_causal_attribution_rate, 4),
                "causal_parity_certified": r.causal_parity_certified,
                "passed": r.passed,
                "diff_count": len(r.all_diffs()),
            }
            for r in reports
        ],
    }


def _rank_id_from_trace_path(trace_path: Path, live_dir: Path, fallback_idx: int) -> int:
    """Parse explicit rank ID from path components (e.g. ``rank17/trace_view.json``)."""
    try:
        rel_parts = (live_dir.name, *trace_path.relative_to(live_dir).parts)
    except ValueError:
        rel_parts = trace_path.parts[-3:]
    for part in reversed(rel_parts):
        match = _RANK_PATH_RE.search(part)
        if match is not None:
            return int(match.group(1))
    return fallback_idx


def evaluate_live_dir(
    live_dir: Path,
    *,
    device_arch: str,
    hardware_sku: str,
    case: Case | Mapping[str, Any] | Path | None = None,
    predicted_nodes: Sequence[PredictedKernelNode] | None = None,
    phase: PhaseKind = "decode",
    tokens: int = 1,
    context: int = 1024,
    sequences: int = 1,
    exec_mode: str = "eager",
    min_attribution_rate: float = DEFAULT_MIN_ATTRIBUTION_RATE,
    error_messages: Sequence[str] = (),
    registry: ProbedAbiRegistry | None = None,
) -> dict[str, Any]:
    """Stream one rank trace at a time in bounded memory against independent predictions."""
    reg = registry if registry is not None else default_abi_registry()
    if predicted_nodes is not None:
        expected_preds = tuple(predicted_nodes)
        graph_id = ""
    elif case is not None:
        prepared, expected_preds = build_predicted_nodes_from_case(
            case,
            phase=phase,
            tokens=tokens,
            context=context,
            sequences=sequences,
            device_arch=device_arch,
            registry=reg,
        )
        graph_id = prepared.graph_id
    else:
        raise ModelingError(
            "live_parity_requires_independent_predictions: supply --case or explicit predicted_nodes"
        )

    traces = sorted(live_dir.rglob("trace_view.json"))
    if not traces:
        raise FileNotFoundError(f"no trace_view.json found under {live_dir}")

    by_rank_pred: dict[int, list[PredictedKernelNode]] = {}
    for pred in expected_preds:
        by_rank_pred.setdefault(pred.rank_id, []).append(pred)

    dag_diffs: list[ParityDiff] = []
    comm_diffs: list[ParityDiff] = []
    lane_diffs: list[ParityDiff] = []
    exp_diffs: list[ParityDiff] = []
    shape_diffs: list[ParityDiff] = []
    launch_diffs: list[ParityDiff] = []
    all_ambiguous: list[str] = []
    all_unattributed: list[str] = []
    observed_ranks: list[int] = []
    total_npu = 0
    total_nodes = 0
    weighted_exact = 0.0
    all_ranks_executed = True

    if not expected_preds:
        launch_diffs.append(ParityDiff(
            code="MISSING_OR_EXTRA_LAUNCH",
            rank_id=0,
            node_id="predicted_graph",
            detail="empty predicted graph: independent predictions required for causal parity",
            expected=">=1 predicted nodes",
            observed="0",
        ))

    seen_rank_paths: dict[int, Path] = {}
    for enum_idx, trace_path in enumerate(traces):
        rank_idx = _rank_id_from_trace_path(trace_path, live_dir, enum_idx)
        if rank_idx in seen_rank_paths:
            raise ModelingError(
                f"duplicate_rank_trace_detected:rank={rank_idx}:"
                f"{trace_path.name}:{seen_rank_paths[rank_idx].name}"
            )
        seen_rank_paths[rank_idx] = trace_path
        rank_preds = by_rank_pred.get(rank_idx, ())
        outcome = evaluate_trace_view_stream(
            trace_path,
            rank_preds,
            rank_id=rank_idx,
            phase=phase,
            exec_mode=exec_mode,  # type: ignore[arg-type]
            device_arch=device_arch,
            registry=reg,
        )
        observed_ranks.append(rank_idx)
        total_npu += outcome.npu_device_kernel_count
        total_nodes += outcome.total_events
        weighted_exact += outcome.exact_causal_attribution_rate * outcome.total_events
        all_ambiguous.extend(outcome.ambiguous_matches)
        all_unattributed.extend(outcome.unattributed_device_kernels)
        if outcome.npu_device_kernel_count <= 0:
            all_ranks_executed = False
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_idx,
                node_id="rank_execution",
                detail=f"rank {rank_idx} recorded zero real NPU device kernels",
                expected=">=1 real NPU kernels",
                observed="0",
            ))
        if outcome.ambiguous_matches:
            launch_diffs.append(ParityDiff(
                code="UNATTRIBUTED_KERNEL",
                rank_id=rank_idx,
                node_id="ambiguous_proximity",
                detail=f"rank {rank_idx} has {len(outcome.ambiguous_matches)} ambiguous proximity match(es)",
                expected="0 ambiguous_matches",
                observed=str(len(outcome.ambiguous_matches)),
            ))
        if outcome.npu_device_kernel_count > 0 and outcome.exact_causal_attribution_rate < min_attribution_rate:
            launch_diffs.append(ParityDiff(
                code="UNATTRIBUTED_KERNEL",
                rank_id=rank_idx,
                node_id="attribution",
                detail=(
                    f"rank {rank_idx} exact causal attribution rate "
                    f"{outcome.exact_causal_attribution_rate:.3f} < {min_attribution_rate:.3f}"
                ),
                expected=f">={min_attribution_rate:.3f}",
                observed=f"{outcome.exact_causal_attribution_rate:.3f}",
            ))
        if expected_preds and not rank_preds:
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_idx,
                node_id="predicted_graph",
                detail=f"participating rank {rank_idx} has no predicted subgraph",
                expected=f"predicted nodes for rank {rank_idx}",
                observed="0",
            ))
        else:
            dag_diffs.extend(outcome.dag_diffs)
            comm_diffs.extend(outcome.comm_diffs)
            lane_diffs.extend(outcome.lane_diffs)
            exp_diffs.extend(outcome.exp_diffs)
            shape_diffs.extend(outcome.shape_diffs)
            launch_diffs.extend(outcome.launch_diffs)

    observed_rank_set = set(observed_ranks)
    for pred_rank, nodes in by_rank_pred.items():
        if pred_rank not in observed_rank_set:
            for pnode in nodes:
                comm_diffs.append(ParityDiff(
                    code="WRONG_RANK_COLLECTIVE",
                    rank_id=pred_rank,
                    node_id=pnode.node_id,
                    detail=f"predicted rank {pred_rank} absent from observed ranks {tuple(observed_ranks)}",
                    expected=str(pred_rank),
                    observed=str(tuple(observed_ranks)),
                ))

    missing_so, missing_py = classify_runtime_errors(error_messages)
    has_diffs = bool(
        dag_diffs or comm_diffs or lane_diffs or exp_diffs
        or shape_diffs or launch_diffs or missing_so or missing_py
    )
    passed = (total_npu > 0) and all_ranks_executed and not has_diffs
    weighted_rate = (weighted_exact / total_nodes) if total_nodes > 0 else 0.0
    cfg_payload = {"live_dir": live_dir.name, "ranks": len(observed_ranks), "graph_id": graph_id}
    config_digest = hashlib.sha256(
        json.dumps(cfg_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = LiveSubgraphParityReport(
        signature_id=f"live:{live_dir.name}",
        hardware_sku=hardware_sku,
        config_digest=config_digest,
        world_ranks=tuple(observed_ranks),
        phase=phase,
        exec_mode=exec_mode,  # type: ignore[arg-type]
        vllm_ascend_rev="",
        sim_rev=graph_id,
        npu_device_kernel_count=total_npu,
        exact_causal_attribution_rate=weighted_rate,
        ambiguous_matches=tuple(all_ambiguous),
        unattributed_device_kernels=tuple(all_unattributed),
        missing_so_symbols=missing_so,
        missing_python_modules=missing_py,
        dag_order_diffs=tuple(dag_diffs),
        rank_collective_diffs=tuple(comm_diffs),
        stream_lane_diffs=tuple(lane_diffs),
        expansion_diffs=tuple(exp_diffs),
        shape_diffs=tuple(shape_diffs),
        launch_diffs=tuple(launch_diffs),
        passed=passed,
        verification_mode="causal_dag_parity",
        causal_parity_certified=passed and not all_ambiguous and not all_unattributed,
    )
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-a3", type=Path, default=None, help="Sanitized A3 capture JSON path")
    parser.add_argument("--offline-a5", type=Path, default=None, help="Sanitized A5 capture JSON path")
    parser.add_argument("--live-dir", type=Path, default=None, help="Live Ascend profiler directory")
    parser.add_argument("--case", type=Path, default=None, help="Case JSON lowered independently before reading live traces")
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--exec-mode", choices=("eager", "aclgraph"), default="eager")
    parser.add_argument("--arch", choices=("A2", "A3", "A5"), default="A5")
    parser.add_argument("--sku", default=_A5_SKU)
    args = parser.parse_args(argv)

    if args.live_dir is not None:
        result = evaluate_live_dir(
            args.live_dir,
            device_arch=args.arch,
            hardware_sku=args.sku,
            case=args.case,
            phase=args.phase,
            tokens=args.tokens,
            context=args.context,
            sequences=args.sequences,
            exec_mode=args.exec_mode,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1

    if args.offline_a3 is None and args.offline_a5 is None:
        parser.error("supply --offline-a3/--offline-a5 or --live-dir")
    result = evaluate_offline_bundle(a3_path=args.offline_a3, a5_path=args.offline_a5)
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
