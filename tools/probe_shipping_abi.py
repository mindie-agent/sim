#!/usr/bin/env python3
"""Offline source-proof and live-NPU hardware ABI probe & SHA-256 catalog sealer.

[Role] Probe and seal hardware-usable Ascend operator ABIs across A2, A3, and A5
  into ``data/abi/cann93_shipping_abi_{a2,a3,a5}.json`` against the L0 contract
  in ``mindie_sim.schema.abi``. Covers BOTH CANN-package backends
  (``cann_aclnn``, ``hccl_mc2``) AND non-CANN backends (``vllm_ascend_csrc``,
  ``triton``, ``torch_npu``, ``catlass``, ``third_party_wheel``).
[Invariants] Zero host-home path leakage, zero fake SHAs, zero empty-set
  wildcards. Every ``available_on_device=True`` entry carries a verified 4-field
  ``SourceRef`` hashed directly from the mounted upstream file and a canonical
  ``probe_signature_sha256``. Lazy-imports ``torch_npu`` ONLY in ``live-npu``
  mode so routine execution never touches a device context (DEV-ZERO-CTX).
[Timeouts & Locks] Bounded local git/file reads (30 s git timeout); no locks.
[Side-Effect Policy] Writes sealed JSON catalogs only when invoked with
  ``--write``; ``--mode verify`` is strictly read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim.schema.abi import (
    ABI_CATALOG_SCHEMA,
    CANN_PACKAGE_ABI_BACKENDS,
    HARDWARE_USABLE_ABI_BACKENDS,
    NON_CANN_ABI_BACKENDS,
    OPEN_SYNTHESIS_ABI_BACKENDS,
    AbiCatalogSnapshot,
    AbiShapeConstraints,
    ProbedAbiEntry,
)
from mindie_sim.schema.datapath import SourceRef

GIT_TIMEOUT_S = 30
CANN_STACK_VERSION = "cann-9.3.0+vllm-ascend@0c99d3fc"
SOC_SKU_IDS = {
    "A2": "ascend910b4",
    "A3": "ascend910_9362",
    "A5": "ascend950dt_9582",
}

# Algebraic C++ CheckParams / Tiling bounds (zero heuristic performance thresholds).
K_MIN_MC2 = 256
K_MAX_MC2_EXCL = 65535
K_MAX_GMM_A8W4_EXCL = 20000
K_MAX_GMM_A8W8_EXCL = 65536
N_MAX_GMM_ARCH22 = 10240
N_MAX_SWIGLU_QUANT = 8192
HIDDEN_DIV_INDEXER_QUANT = 32
HIDDEN_MAX_INDEXER_QUANT = 8192
HIDDEN_MIN_MEGAMOE_A23 = 1024
HIDDEN_MIN_MEGAMOE_A5 = 512
HIDDEN_MAX_MEGAMOE = 8192
HIDDEN_DIV_MEGAMOE_A23 = 512
HIDDEN_DIV_MEGAMOE_A5 = 256
INTER_MIN_MEGAMOE_A23 = 512
INTER_MIN_MEGAMOE_A5 = 128
INTER_DIV_MEGAMOE_A23 = 256
INTER_DIV_MEGAMOE_A5 = 128
TOPK_MAX_MEGAMOE_A23 = 16
TOPK_MAX_MEGAMOE_A5 = 32
MAX_SHARED_MEGAMOE_A5 = 4
MAX_EXPERTS_MEGAMOE_A23 = 1024
MAX_EXPERTS_GROUPED_MATMUL = 1024
MAX_EXPERTS_MOE_GATING = 2048
MQSMLA_HEAD_DIMS = frozenset({512})
MQSMLA_ROPE_DIMS = frozenset({64})
MQSMLA_KV_HEADS = frozenset({1})
QUANT_46_LAST_AXIS_DIV = 32
QUANT_46_RANK_RANGE = (1, 7)
MLA_PROLOG_HIDDEN_SET = frozenset({1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192})
WS_LOCAL = frozenset({1})
WS_A23_MC2 = frozenset({2, 4, 8})
WS_A5_MC2 = frozenset({2, 4, 8, 16, 32, 64})
WS_HCCL_ALL = frozenset({2, 4, 8, 16, 32, 64, 128, 256})
WS_EP_A23 = frozenset({2, 4, 8, 16, 32, 64})
WS_EP_MEGAMOE_A23 = frozenset({2, 4, 8, 16, 32, 48, 64, 96, 128})
WS_EP_A5 = frozenset({2, 4, 8, 16, 32, 64, 128, 256})

PLACEMENTS_LOCAL = frozenset({"local"})
PLACEMENTS_A2 = frozenset({"hccs_single_node", "a2_switched_multinode"})
PLACEMENTS_A3 = frozenset({"hccs_single_node", "hccs_supernode_a3"})
PLACEMENTS_A5_MC2 = frozenset({"ub_ccu_intramachine", "ub_aicpu_supernode"})
PLACEMENTS_A5_MEGAMOE = frozenset({
    "ub_ccu_intramachine", "urma_cross_supernode", "layered_ub_roce",
})
PLACEMENTS_A23_HCCL = frozenset({
    "hccs_single_node", "hccs_supernode_a3", "a2_switched_multinode", "roce_flat_cross_node",
})
PLACEMENTS_A5_HCCL = frozenset({
    "ub_ccu_intramachine", "ub_aicpu_supernode", "urma_cross_supernode",
    "layered_ub_roce", "roce_flat_cross_node",
})

REPO_URLS = {
    "vllm-ascend": "vllm-project/vllm-ascend",
    "op-plugin": "Ascend/op-plugin",
    "ops-transformer": "cann/ops-transformer",
    "ops-nn": "cann/ops-nn",
    "ops-math": "cann/ops-math",
    "ops-tensor": "cann/ops-tensor",
    "catlass": "Ascend/catlass",
    "mindie-sd": "Ascend/MindIE-SD",
    "cann-recipes-infer": "cann/cann-recipes-infer",
}


@dataclass(frozen=True)
class RepoHandle:
    name: str
    repository: str
    root: Path
    revision: str


def _git_head(path: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False, timeout=GIT_TIMEOUT_S,
    )
    if done.returncode != 0 or not done.stdout.strip():
        raise RuntimeError(f"failed to resolve git HEAD for {path}: {done.stderr.strip()}")
    return done.stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _probe_sig(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_ref(repos: dict[str, RepoHandle], repo_key: str, rel_path: str) -> SourceRef:
    handle = repos[repo_key]
    full = handle.root / rel_path
    if not full.is_file():
        raise FileNotFoundError(f"missing upstream source file: {repo_key}:{rel_path}")
    return SourceRef(
        repository=handle.repository,
        revision=handle.revision,
        relative_path=rel_path,
        source_sha256=_sha256_file(full),
    )


def _load_capture_kernels(capture_path: Path | None) -> set[str]:
    """Extract observed real NPU device kernel types/names from a sanitized capture bundle."""
    if capture_path is None or not capture_path.is_file():
        return set()
    raw = json.loads(capture_path.read_text(encoding="utf-8"))
    observed: set[str] = set()

    def _add(val: Any) -> None:
        tok = str(val or "").strip()
        if tok:
            observed.add(tok)

    # Top-level A3 trace_3layer_torch_to_npu summary if present.
    t3 = raw.get("trace_3layer_torch_to_npu")
    if isinstance(t3, dict):
        for v in t3.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for fld in ("kernel", "kernel_name", "type", "name", "cann_api"):
                            _add(item.get(fld))

    captures = raw.get("captures") or {}
    items = (
        list(captures.values()) if isinstance(captures, dict)
        else list(captures) if isinstance(captures, list)
        else []
    )
    for cap in items:
        if not isinstance(cap, dict):
            continue
        # A3 schema: top_op_signatures ("Kernel|CannApi|TaskType") + representative steps.
        for sig_item in cap.get("top_op_signatures") or ():
            if isinstance(sig_item, dict):
                sig_str = str(sig_item.get("signature") or "")
                parts = sig_str.split("|")
                if parts:
                    _add(parts[0])
                if len(parts) >= 2:
                    _add(parts[1])
        for step_key in ("representative_prefill_step", "representative_decode_step"):
            step_obj = cap.get(step_key)
            if isinstance(step_obj, dict):
                for sub_key in ("pad_ops", "hccl_or_comm"):
                    sub_val = step_obj.get(sub_key)
                    if isinstance(sub_val, dict):
                        for k_name in sub_val:
                            _add(k_name)
                    elif isinstance(sub_val, list):
                        for entry in sub_val:
                            if isinstance(entry, dict):
                                for fld in ("kernel", "name", "type", "cann_api"):
                                    _add(entry.get(fld))
                            elif isinstance(entry, str):
                                _add(entry)
        # A5 schema: decode_op_sequence_summary / prefill_op_sequence_summary / multi_kernel_cann_expansions.
        for seq_key in ("decode_op_sequence_summary", "prefill_op_sequence_summary"):
            for row in cap.get(seq_key) or ():
                if isinstance(row, dict):
                    for fld in ("kernel", "cann_api", "cpu_op"):
                        _add(row.get(fld))
        mk_exp = cap.get("multi_kernel_cann_expansions")
        if isinstance(mk_exp, dict):
            for k_exp, v_exp in mk_exp.items():
                _add(str(k_exp).split(" ")[0])
                if isinstance(v_exp, list):
                    for item in v_exp:
                        if isinstance(item, str):
                            _add(item)
                        elif isinstance(item, dict):
                            _add(item.get("kernel") or item.get("name"))
        # Generic fallback keys.
        for key in ("prefill_kernels", "decode_kernels", "kernels"):
            for k in cap.get(key) or ():
                if isinstance(k, dict):
                    for field in ("type", "name", "kernel_type", "kernel_name"):
                        _add(k.get(field))
        for key in ("top_kernels", "decode_top_kernels", "prefill_top_kernels", "kernel_counts"):
            mapping = cap.get(key)
            if isinstance(mapping, dict):
                for k_name in mapping:
                    _add(k_name)
    return observed


def _mk(
    repos: dict[str, RepoHandle],
    *,
    entry_id: str,
    abi_id: str,
    available: bool,
    backend: str,
    symbol: str,
    arch: str,
    role: str,
    dtype_tuple: tuple[tuple[str, ...], ...],
    repo_key: str,
    rel_path: str,
    shape: AbiShapeConstraints | None = None,
    graph_motif: str = "",
    track: str = "merged",
    pass_id: str = "",
    mech_id: str = "",
    va_ref: str = "",
    op_ref: str = "",
    cann_ref: str = "",
    kernels: tuple[str, ...] = (),
    outputs: tuple[tuple[str, ...], ...] = (),
    placements: frozenset[str] = PLACEMENTS_LOCAL,
    world_sizes: frozenset[int] = WS_LOCAL,
    activations: frozenset[str] = frozenset({"none"}),
    scopes: frozenset[str] = frozenset({"target", "draft"}),
    allow_bias: bool = True,
    require_no_bias: bool = False,
    require_bias: bool = False,
    note: str = "",
    observed_kernels: set[str] | None = None,
) -> ProbedAbiEntry:
    src = _source_ref(repos, repo_key, rel_path)
    live_hits = sorted(k for k in kernels if observed_kernels and k in observed_kernels)
    full_note = note
    if live_hits:
        tag = f"live_silicon_observed={','.join(live_hits)}"
        full_note = f"{note}; {tag}" if note else tag
    sig = _probe_sig({
        "entry_id": entry_id,
        "abi_id": abi_id,
        "arch": arch,
        "backend": backend,
        "symbol": symbol,
        "available": available,
        "source_sha256": src.source_sha256,
        "dtype_tuple": [list(t) for t in dtype_tuple],
        "kernels": list(kernels),
        "require_bias": require_bias,
    }) if available else ""
    return ProbedAbiEntry(
        entry_id=entry_id,
        abi_id=abi_id,
        available_on_device=available,
        backend_kind=backend,  # type: ignore[arg-type]
        symbol_or_kernel=symbol,
        device_arch=arch,  # type: ignore[arg-type]
        stack_or_cann_version=CANN_STACK_VERSION,
        role=role,
        dtype_tuple=dtype_tuple,
        shape_constraints=shape or AbiShapeConstraints(),
        source_ref=src,
        graph_motif=graph_motif,
        vllm_ascend_track=track,  # type: ignore[arg-type]
        required_stack_pass_id=pass_id,
        required_mechanism_id=mech_id,
        vllm_ascend_ref=va_ref,
        op_plugin_ref=op_ref,
        cann_check_params_ref=cann_ref,
        device_kernel_expansion=kernels,
        allowed_output_tuples=outputs,
        allowed_rank_placements=placements,  # type: ignore[arg-type]
        allowed_world_sizes=world_sizes,
        allowed_activations=activations,
        allowed_model_scopes=scopes,  # type: ignore[arg-type]
        allow_bias=allow_bias,
        require_no_bias=require_no_bias,
        require_bias=require_bias,
        probe_signature_sha256=sig,
        probe_domain_note=full_note,
    )


def _mc2_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    mc2_place = PLACEMENTS_A5_MC2 if is_a5 else (PLACEMENTS_A3 if arch == "A3" else PLACEMENTS_A2)
    mc2_ws = WS_A5_MC2 if is_a5 else WS_A23_MC2
    ep_ws = WS_EP_A5 if is_a5 else WS_EP_A23
    ag_shape = AbiShapeConstraints(
        k_min=K_MIN_MC2, k_max_exclusive=K_MAX_MC2_EXCL,
        max_comm_bytes_formula="ag_63x256MiB" if is_a5 else "none",
        require_even_k_for_fp4=is_a5, require_even_n_for_fp4_transb=is_a5,
    )
    rs_shape = AbiShapeConstraints(
        k_min=K_MIN_MC2, k_max_exclusive=K_MAX_MC2_EXCL,
        require_m_divisible_by_ws=True,
        max_comm_bytes_formula="mmrs_ccu_out_cap" if is_a5 else "none",
    )
    entries: list[ProbedAbiEntry] = [
        _mk(
            repos, entry_id=f"{p_low}_hccl_agmm_v2_dense", abi_id="aclnnAllGatherMatmulV2",
            available=True, backend="hccl_mc2", symbol="aclnnAllGatherMatmulV2", arch=arch,
            role="collective_gemm", graph_motif="allgather_matmul",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none"),
                ("x1:bf16:ND:dense::none", "x2:bf16:FRACTAL_NZ:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:FRACTAL_NZ:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",), ("y:bf16:ND",), ("y:fp16:ND",)),
            shape=ag_shape, placements=mc2_place, world_sizes=mc2_ws,
            allow_bias=is_a5, require_no_bias=not is_a5,
            repo_key="ops-transformer",
            rel_path="mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp",
            va_ref="vllm_ascend/device/device_op.py:390",
            op_ref="op_plugin/ops/opapi/AllGatherQuantMatmulKernelNpuOpApi.cpp",
            cann_ref="ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp:149",
            kernels=("AllGatherMatmul",),
            note="Direct native AGMM callsite; rejects roce_flat_cross_node",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_agmm_v2_quant", abi_id="aclnnAllGatherMatmulV2",
            available=True, backend="hccl_mc2", symbol="aclnnAllGatherMatmulV2", arch=arch,
            role="collective_gemm", graph_motif="allgather_matmul", track="unmerged_pr",
            mech_id="tp_mc2_overlap",
            dtype_tuple=(
                (
                    ("x1:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "x2:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),
                    ("x1:fp8_e4m3fn:ND:per_block_128:128:fp32", "x2:fp8_e4m3fn:FRACTAL_NZ:per_block_128:128:fp32"),
                    ("x1:hif8:ND:hif8_mx:32:fp8_e8m0", "x2:hif8:FRACTAL_NZ:hif8_mx:32:fp8_e8m0"),
                    ("x1:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "x2:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0"),
                ) if is_a5 else (
                    ("x1:int8:ND:per_token_int8::fp32", "x2:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                    ("x1:int8:ND:per_token_int8::fp32", "x2:int8:ND:per_channel_int8::fp32"),
                    ("x1:int4:ND:w4a8_int::fp32", "x2:int4:ND:w4a8_int::fp32"),
                )
            ),
            outputs=(
                (("y:bf16:ND",),) * 4 if is_a5 else (("y:bf16:ND",),) * 3
            ),
            shape=ag_shape, placements=mc2_place, world_sizes=mc2_ws,
            allow_bias=False, require_no_bias=True,
            repo_key="ops-transformer",
            rel_path="mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp",
            va_ref="vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py:91",
            op_ref="op_plugin/ops/opapi/AllGatherQuantMatmulKernelNpuOpApi.cpp",
            cann_ref="ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp:149",
            kernels=("AllGatherMatmul",),
            note="Quantized AGMM compiler pass requires proposed/target PR track",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_mmrs_v2_merged_dense", abi_id="aclnnMatmulReduceScatterV2",
            available=True, backend="hccl_mc2", symbol="aclnnMatmulReduceScatterV2", arch=arch,
            role="collective_gemm", graph_motif="matmul_reducescatter",
            pass_id="mm_reduce_scatter_fusion",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none"),
                ("x1:bf16:ND:dense::none", "x2:bf16:FRACTAL_NZ:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:FRACTAL_NZ:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",), ("y:bf16:ND",), ("y:fp16:ND",)),
            shape=rs_shape, placements=mc2_place, world_sizes=WS_A23_MC2,
            allow_bias=False, require_no_bias=True,
            repo_key="ops-transformer",
            rel_path="mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp",
            va_ref="vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py:93",
            op_ref="op_plugin/ops/opapi/MatmulReduceScatterBaseKernelNpuOpApi.cpp:80",
            cann_ref="ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp:144",
            kernels=("MatmulReduceScatter",),
            note="Merged MMRS graph-fusion pass: dense BF16/FP16 bias-free ws in {2,4,8}",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_mmrs_v2_quant", abi_id="aclnnMatmulReduceScatterV2",
            available=True, backend="hccl_mc2", symbol="aclnnMatmulReduceScatterV2", arch=arch,
            role="collective_gemm", graph_motif="matmul_reducescatter", track="unmerged_pr",
            mech_id="tp_mc2_overlap",
            dtype_tuple=(
                (
                    ("x1:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "x2:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),
                    ("x1:hif8:ND:hif8_mx:32:fp8_e8m0", "x2:hif8:FRACTAL_NZ:hif8_mx:32:fp8_e8m0"),
                    ("x1:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "x2:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0"),
                ) if is_a5 else (
                    ("x1:int8:ND:per_token_int8::fp32", "x2:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                    ("x1:int8:ND:per_token_int8::fp32", "x2:int8:ND:per_channel_int8::fp32"),
                )
            ),
            outputs=(
                (("y:bf16:ND",),) * 3 if is_a5 else (("y:bf16:ND",),) * 2
            ),
            shape=rs_shape, placements=mc2_place, world_sizes=mc2_ws,
            allow_bias=False, require_no_bias=True,
            repo_key="ops-transformer",
            rel_path="mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp",
            va_ref="vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py:93",
            op_ref="op_plugin/ops/opapi/MatmulReduceScatterBaseKernelNpuOpApi.cpp:80",
            cann_ref="ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp:153",
            kernels=("MatmulReduceScatter",),
            note="Quantized MMRS rejects int4 on A2/A3; requires proposed/target track",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_moe_distribute_dispatch",
            abi_id="aclnnMoeDistributeDispatchV2",
            available=True, backend="hccl_mc2",
            symbol="aclnnMoeDistributeDispatchV3" if is_a5 else "aclnnMoeDistributeDispatchV2",
            arch=arch, role="moe_dispatch", graph_motif="moe_decode_dispatch",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                ("x:fp16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                ("x:bf16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                ("x:fp16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                *(
                    (
                        ("x:bf16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                        ("x:fp16:ND:dense::none", "expert_ids:int32:ND:none::none"),
                        ("x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "expert_ids:int32:ND:none::none", "scales:fp8_e8m0:ND:none::none"),
                    )
                    if is_a5 else ()
                ),
            ),
            outputs=(
                (
                    "expand_x:bf16:ND", "dynamic_scales:fp32:ND", "assist_info_for_combine:int32:ND",
                    "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                    "expand_scales:fp32:ND",
                ),
                (
                    "expand_x:fp16:ND", "dynamic_scales:fp32:ND", "assist_info_for_combine:int32:ND",
                    "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                    "expand_scales:fp32:ND",
                ),
                (
                    "expand_x:int8:ND", "dynamic_scales:fp32:ND", "assist_info_for_combine:int32:ND",
                    "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                    "expand_scales:fp32:ND",
                ),
                (
                    "expand_x:int8:ND", "dynamic_scales:fp32:ND", "assist_info_for_combine:int32:ND",
                    "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                    "expand_scales:fp32:ND",
                ),
                *(
                    (
                        (
                            "expand_x:fp8_e4m3fn:ND", "dynamic_scales:fp8_e8m0:ND", "assist_info_for_combine:int32:ND",
                            "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                            "expand_scales:fp32:ND",
                        ),
                        (
                            "expand_x:fp8_e4m3fn:ND", "dynamic_scales:fp8_e8m0:ND", "assist_info_for_combine:int32:ND",
                            "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                            "expand_scales:fp32:ND",
                        ),
                        (
                            "expand_x:fp8_e4m3fn:ND", "dynamic_scales:fp8_e8m0:ND", "assist_info_for_combine:int32:ND",
                            "expert_token_nums:int64:ND", "ep_recv_counts:int32:ND", "tp_recv_counts:int32:ND",
                            "expand_scales:fp32:ND",
                        ),
                    )
                    if is_a5 else ()
                ),
            ),
            placements=mc2_place, world_sizes=ep_ws,
            repo_key="ops-transformer",
            rel_path=(
                "mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v3.cpp"
                if is_a5 else
                "mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2.cpp"
            ),
            va_ref="vllm_ascend/ops/fused_moe/token_dispatcher.py:251",
            op_ref="op_plugin/ops/opapi/MoeDistributeDispatchV2KernelOpApi.cpp:279",
            cann_ref=(
                "ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v3.cpp:27"
                if is_a5 else
                "ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2.cpp:27"
            ),
            kernels=("MoeDistributeDispatchV3",) if is_a5 else ("MoeDistributeDispatchV2",),
            note="Standard decode MoE MC2 dispatch (V3 on A5 via op-plugin probe, V2 on A2/A3; compatible with MSO=True)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_moe_distribute_combine",
            abi_id="aclnnMoeDistributeCombineV2",
            available=True, backend="hccl_mc2",
            symbol="aclnnMoeDistributeCombineV3" if is_a5 else "aclnnMoeDistributeCombineV2",
            arch=arch, role="moe_combine", graph_motif="moe_decode_combine",
            dtype_tuple=(
                (
                    "expand_x:bf16:ND:dense::none", "expert_ids:int32:ND:none::none",
                    "assist_info_for_combine:int32:ND:none::none", "ep_send_counts:int32:ND:none::none",
                    "expert_scales:fp32:ND:dense::none",
                ),
                (
                    "expand_x:fp16:ND:dense::none", "expert_ids:int32:ND:none::none",
                    "assist_info_for_combine:int32:ND:none::none", "ep_send_counts:int32:ND:none::none",
                    "expert_scales:fp32:ND:dense::none",
                ),
            ),
            outputs=(("x_out:bf16:ND",), ("x_out:fp16:ND",)),
            placements=mc2_place, world_sizes=ep_ws,
            repo_key="ops-transformer",
            rel_path=(
                "mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v3.cpp"
                if is_a5 else
                "mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v2.cpp"
            ),
            va_ref="vllm_ascend/ops/fused_moe/token_dispatcher.py:352",
            op_ref="op_plugin/ops/opapi/MoeDistributeCombineKernelV2OpApi.cpp:95",
            cann_ref=(
                "ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v3.cpp:28"
                if is_a5 else
                "ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v2.cpp:28"
            ),
            kernels=("MoeDistributeCombineV3",) if is_a5 else ("MoeDistributeCombineV2",),
            note="Standard decode MoE MC2 combine (V3 on A5 via op-plugin V6/V4/V3 probe, V2 on A2/A3; 5 required inputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_hccl_megamoe_routed", abi_id="aclnnMegaMoe",
            available=True, backend="hccl_mc2", symbol="cann_ops_transformer.ops.mega_moe",
            arch=arch, role="moe_fused_frame", graph_motif="moe_mc2_frame",
            dtype_tuple=(
                (
                    (
                        "context:int32:ND:none::none", "x:bf16:ND:dense::none",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
                        "weight2:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
                    ),
                    (
                        "context:int32:ND:none::none", "x:bf16:ND:dense::none",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0",
                        "weight2:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0",
                    ),
                    (
                        "context:int32:ND:none::none", "x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
                        "weight2:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0",
                    ),
                ) if is_a5 else (
                    (
                        "context:int32:ND:none::none", "x:bf16:ND:dense::none",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:bf16:ND:dense::none", "weight2:bf16:ND:dense::none",
                    ),
                    (
                        "context:int32:ND:none::none", "x:bf16:ND:dense::none",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:int8:FRACTAL_NZ:per_channel_int8::fp32",
                        "weight2:int8:FRACTAL_NZ:per_channel_int8::fp32",
                    ),
                    (
                        "context:int32:ND:none::none", "x:bf16:ND:dense::none",
                        "topk_ids:int32:ND:none::none", "topk_weights:fp32:ND:dense::none",
                        "weight1:int4:FRACTAL_NZ:w4a8_int::fp32", "weight2:int4:FRACTAL_NZ:w4a8_int::fp32",
                    ),
                )
            ),
            outputs=(
                ("y:bf16:ND", "expert_token_nums:int32:ND"),
                ("y:bf16:ND", "expert_token_nums:int32:ND"),
                ("y:bf16:ND", "expert_token_nums:int32:ND"),
            ),
            shape=AbiShapeConstraints(
                hidden_range=(HIDDEN_MIN_MEGAMOE_A5 if is_a5 else HIDDEN_MIN_MEGAMOE_A23, HIDDEN_MAX_MEGAMOE),
                hidden_divisibility=HIDDEN_DIV_MEGAMOE_A5 if is_a5 else HIDDEN_DIV_MEGAMOE_A23,
                inter_min=INTER_MIN_MEGAMOE_A5 if is_a5 else INTER_MIN_MEGAMOE_A23,
                inter_divisibility=INTER_DIV_MEGAMOE_A5 if is_a5 else INTER_DIV_MEGAMOE_A23,
                topk_range=(1, TOPK_MAX_MEGAMOE_A5 if is_a5 else TOPK_MAX_MEGAMOE_A23),
                max_shared_experts_per_rank=MAX_SHARED_MEGAMOE_A5 if is_a5 else 0,
                max_experts=0 if is_a5 else MAX_EXPERTS_MEGAMOE_A23,
            ),
            placements=PLACEMENTS_A5_MEGAMOE if is_a5 else mc2_place,
            world_sizes=ep_ws if is_a5 else WS_EP_MEGAMOE_A23,
            activations=(
                frozenset({"swiglu", "swiglustep", "swigluoai", "situglu", "gelu"})
                if is_a5 else frozenset({"swiglu", "situglu", "swiglustep", "swigluoai"})
            ),
            repo_key="ops-transformer",
            rel_path=(
                "mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp"
                if is_a5 else
                "mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp"
            ),
            va_ref="vllm_ascend/ops/fused_moe/moe_utils.py:144",
            op_ref="cann_ops_transformer.ops.mega_moe",
            cann_ref=(
                "ops-transformer/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp:38"
                if is_a5 else
                "ops-transformer/mc2/mega_moe/op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp:105"
            ),
            kernels=("MegaMoe",),
            note="A5 MXFP8/MXFP4 (mode 4 BF16 in or mode 0 topoType=0); A2/A3 BF16/INT8/INT4 NZ (no FP8)",
            observed_kernels=observed,
        ),
    ]
    # HCCL standard collectives available across all architectures.
    hccl_place = PLACEMENTS_A5_HCCL if is_a5 else PLACEMENTS_A23_HCCL
    for cid, sym, kname in (
        ("hcom_allReduce", "HcclAllreduce", "hcom_allReduce_"),
        ("hcom_allGather", "HcclAllGather", "hcom_allGather_"),
        ("hcom_reduceScatter", "HcclReduceScatter", "hcom_reduceScatter_"),
        ("hcom_alltoall", "HcclAlltoAll", "hcom_alltoall_"),
        ("hcom_alltoallv", "HcclAlltoAllV", "hcom_alltoallv_"),
    ):
        entries.append(_mk(
            repos, entry_id=f"{p_low}_hccl_{cid.lower()}", abi_id=cid,
            available=True, backend="hccl_mc2", symbol=sym, arch=arch,
            role="collective", graph_motif=cid,
            dtype_tuple=(
                ("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",),
                ("x:fp32:ND:dense::none",), ("x:int8:ND:dense::none",),
                ("x:int32:ND:dense::none",),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",), ("y:fp32:ND",), ("y:int8:ND",), ("y:int32:ND",)),
            placements=hccl_place, world_sizes=WS_HCCL_ALL,
            repo_key="vllm-ascend",
            rel_path="vllm_ascend/distributed/device_communicators/npu_communicator.py",
            va_ref="vllm_ascend/distributed/device_communicators/npu_communicator.py:100",
            cann_ref="hccl/ProcessGroupHCCL",
            kernels=(kname,),
            note="HCCL collective; supports intra-node HCCS/UB and inter-node RoCE",
            observed_kernels=observed,
        ))
    return entries


def _cann_moe_and_norm_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    entries: list[ProbedAbiEntry] = [
        _mk(
            repos, entry_id=f"{p_low}_cann_grouped_matmul", abi_id="aclnnGroupedMatmul",
            available=True, backend="cann_aclnn", symbol="aclnnGroupedMatmulV4", arch=arch,
            role="moe_grouped_matmul", graph_motif="grouped_gemm",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "w:bf16:FRACTAL_NZ:dense::none"),
                ("x:fp16:ND:dense::none", "w:fp16:FRACTAL_NZ:dense::none"),
                ("x:int8:ND:per_token_int8::fp32", "w:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                ("x:int8:ND:per_token_int8::fp32", "w:int4:FRACTAL_NZ:w4a8_int::fp32"),
                *(
                    (
                        ("x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "w:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),
                        ("x:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "w:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0"),
                    ) if is_a5 else ()
                ),
            ),
            outputs=(
                ("y:bf16:ND",),
                ("y:fp16:ND",),
                ("y:bf16:ND",),
                ("y:bf16:ND",),
                *((("y:bf16:ND",), ("y:bf16:ND",)) if is_a5 else ()),
            ),
            repo_key="ops-transformer",
            rel_path="gmm/grouped_matmul/op_kernel/grouped_matmul_autotiling_a8w4.h",
            va_ref="vllm_ascend/ops/fused_moe/moe_mlp.py:122",
            op_ref="op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp",
            cann_ref="ops-transformer/gmm/grouped_matmul/op_kernel/grouped_matmul_autotiling_a8w4.h:72",
            kernels=("GroupedMatmul",),
            note="Cube FRACTAL_M=16 per-expert alignment",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_gmm_swiglu_quant_v1",
            abi_id="aclnnGroupedMatmulSwigluQuant",
            available=not is_a5, backend="cann_aclnn",
            symbol="aclnnGroupedMatmulSwigluQuant", arch=arch,
            role="moe_act_quant", graph_motif="gmm_swiglu_quant",
            dtype_tuple=((
                "x:int8:ND:per_token_int8::fp32", "weight:int8:FRACTAL_NZ:per_channel_int8::fp32",
                "weight_scale:fp32:ND:dense::none", "x_scale:fp32:ND:dense::none",
                "group_list:int64:ND:none::none",
            ),),
            outputs=(("y:int8:ND", "y_scale:fp32:ND"),),
            shape=AbiShapeConstraints(
                k_max_exclusive=K_MAX_GMM_A8W8_EXCL, n_max_inclusive=N_MAX_GMM_ARCH22,
                require_even_n=True,
            ),
            activations=frozenset({"swiglu"}),
            repo_key="ops-transformer",
            rel_path="gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp",
            va_ref="vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:282",
            op_ref="op_plugin/ops/opapi/GroupedMatmulSwigluQuantNpuOpapi.cpp",
            cann_ref="ops-transformer/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp:24",
            kernels=("GroupedMatmulSwigluQuant",) if not is_a5 else (),
            note="A2/A3 shipping W8A8 GMM+SwiGLU+Quant V1 (K<65536, even N<=10240); A5 uses V2/WeightNzV2",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_gmm_swiglu_quant_v1_a8w4",
            abi_id="aclnnGroupedMatmulSwigluQuant",
            available=not is_a5, backend="cann_aclnn",
            symbol="aclnnGroupedMatmulSwigluQuant", arch=arch,
            role="moe_act_quant", graph_motif="gmm_swiglu_quant",
            dtype_tuple=(
                (
                    "x:int8:ND:per_token_int8::fp32", "weight:int4:FRACTAL_NZ:w4a8_int::fp32",
                    "bias:fp32:ND:dense::none",
                    "weight_scale:uint64:ND:none::none", "x_scale:fp32:ND:dense::none",
                    "group_list:int64:ND:none::none",
                ),
                (
                    "x:int8:ND:per_token_int8::fp32", "weight:int4:FRACTAL_NZ:w4a8_int::fp32",
                    "weight_scale:uint64:ND:none::none", "x_scale:fp32:ND:dense::none",
                    "group_list:int64:ND:none::none",
                ),
            ),
            outputs=(
                ("y:int8:ND", "y_scale:fp32:ND"),
                ("y:int8:ND", "y_scale:fp32:ND"),
            ),
            shape=AbiShapeConstraints(
                k_max_exclusive=K_MAX_GMM_A8W4_EXCL, n_max_inclusive=N_MAX_GMM_ARCH22,
                require_even_n=True,
            ),
            activations=frozenset({"swiglu"}),
            allow_bias=True,
            require_bias=True,
            repo_key="ops-transformer",
            rel_path="gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp",
            va_ref="vllm_ascend/quantization/methods/w4a8/w4a8.py:488",
            op_ref="op_plugin/ops/opapi/GroupedMatmulSwigluQuantNpuOpapi.cpp",
            cann_ref="ops-transformer/gmm/grouped_matmul_swiglu_quant/op_api/aclnn_grouped_matmul_swiglu_quant.cpp:178",
            kernels=("GroupedMatmulSwigluQuant",) if not is_a5 else (),
            note="A2/A3 shipping W4A8 GMM+SwiGLU+Quant V1 (A8W4 requires K<20000, even N<=10240, non-null bias)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_gmm_swiglu_quant_v2",
            abi_id="aclnnGroupedMatmulSwigluQuantV2",
            available=True, backend="cann_aclnn",
            symbol="aclnnGroupedMatmulSwigluQuantWeightNzV2" if is_a5 else "aclnnGroupedMatmulSwigluQuantV2",
            arch=arch, role="moe_act_quant", graph_motif="gmm_swiglu_quant",
            dtype_tuple=(
                (
                    ("x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "w:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),
                    ("x:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "w:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0"),
                    ("x:int8:ND:per_token_int8::fp32", "w:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                ) if is_a5 else (
                    ("x:int8:ND:per_token_int8::fp32", "w:int4:FRACTAL_NZ:w4a8_int::fp32"),
                    ("x:int8:ND:per_token_int8::fp32", "w:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                )
            ),
            outputs=(
                (
                    ("y:fp8_e4m3fn:ND", "y_scale:fp8_e8m0:ND"),
                    ("y:fp8_e4m3fn:ND", "y_scale:fp8_e8m0:ND"),
                    ("y:int8:ND", "y_scale:fp32:ND"),
                )
                if is_a5 else (
                    ("y:int8:ND", "y_scale:fp32:ND"),
                    ("y:int8:ND", "y_scale:fp32:ND"),
                )
            ),
            shape=AbiShapeConstraints(
                k_max_exclusive=K_MAX_GMM_A8W8_EXCL if is_a5 else K_MAX_GMM_A8W4_EXCL,
                n_max_inclusive=0 if is_a5 else N_MAX_GMM_ARCH22,
                require_even_n=True,
            ),
            activations=frozenset({"swiglu"}),
            repo_key="ops-transformer",
            rel_path=(
                "gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.cpp"
                if is_a5 else
                "gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp"
            ),
            va_ref="vllm_ascend/quantization/methods/w4a8/w4a8.py:488",
            op_ref="op_plugin/ops/opapi/GroupedMatmulSwigluQuantV2NpuOpapi.cpp:364",
            cann_ref="ops-transformer/gmm/grouped_matmul_swiglu_quant_v2/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp:215",
            kernels=("GroupedMatmulSwigluQuantV2",),
            note="Rejects BF16/FP16; A2/A3 A8W4 enforces K<20000 and even N<=10240",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_dequant_swiglu_quant",
            abi_id="aclnnDequantSwigluQuant", available=True, backend="cann_aclnn",
            symbol="aclnnDequantSwigluQuant", arch=arch,
            role="moe_act_quant", graph_motif="dequant_swiglu_quant",
            dtype_tuple=(
                ("x:int32:ND:per_token_int8::fp32",),
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
            ),
            outputs=(
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
            ),
            shape=AbiShapeConstraints(require_even_n=True),
            activations=frozenset({"swiglu"}),
            repo_key="ops-nn",
            rel_path="quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_def.cpp",
            va_ref="vllm_ascend/ops/fused_moe/shared_experts.py:931",
            op_ref="op_plugin/ops/opapi/DequantSwigluQuantOpApi.cpp:95",
            cann_ref="ops-nn/quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_def.cpp:109",
            kernels=("DequantSwigluQuant",),
            note="Shipping CANN DequantSwigluQuant across A2/A3/A5 (AddConfig ascend910b/93/DAV_3510)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_moe_gating_top_k",
            abi_id="aclnnMoeGatingTopK", available=True, backend="cann_aclnn",
            symbol="aclnnMoeGatingTopK", arch=arch,
            role="moe_routing", graph_motif="moe_gating_top_k",
            dtype_tuple=(
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
                ("x:fp32:ND:dense::none",),
                ("x:bf16:ND:dense::none", "bias:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "bias:fp16:ND:dense::none"),
                ("x:fp32:ND:dense::none", "bias:fp32:ND:dense::none"),
            ),
            outputs=(
                ("y:bf16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp32:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:bf16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp32:ND", "expert_idx:int32:ND", "out:fp32:ND"),
            ),
            shape=AbiShapeConstraints(max_experts=MAX_EXPERTS_MOE_GATING),
            repo_key="ops-transformer",
            rel_path="moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp",
            va_ref="vllm_ascend/ops/fused_moe/router/fused_topk_router.py:227",
            op_ref="op_plugin/ops/opapi/MoeGatingTopKKernelNpuOpApi.cpp:105",
            cann_ref="ops-transformer/moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp:133",
            kernels=("MoeGatingTopK",),
            note="Shipping CANN aclnnMoeGatingTopK across A2/A3/A5 (AddConfig ascend910b/93/950; optional 1D bias)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_moe_gating_top_k_softmax_v2",
            abi_id="aclnnMoeGatingTopKSoftmaxV2", available=True, backend="cann_aclnn",
            symbol="aclnnMoeGatingTopKSoftmaxV2", arch=arch,
            role="moe_routing", graph_motif="moe_gating_top_k_softmax",
            dtype_tuple=(
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
                ("x:fp32:ND:dense::none",),
            ),
            outputs=(
                ("y:bf16:ND", "expert_idx:int32:ND", "softmax_result:fp32:ND"),
                ("y:fp16:ND", "expert_idx:int32:ND", "softmax_result:fp32:ND"),
                ("y:fp32:ND", "expert_idx:int32:ND", "softmax_result:fp32:ND"),
            ),
            repo_key="ops-transformer",
            rel_path="moe/moe_gating_top_k_softmax_v2/op_host/moe_gating_top_k_softmax_v2_def.cpp",
            va_ref="vllm_ascend/_310p/fused_moe/grouped_topk_router.py:57",
            op_ref="op_plugin/ops/opapi/MoeGatingTopKSoftmaxV2KernelNpuOpApi.cpp:110",
            cann_ref="ops-transformer/moe/moe_gating_top_k_softmax_v2/op_host/moe_gating_top_k_softmax_v2_def.cpp:58",
            kernels=("MoeGatingTopKSoftmaxV2",),
            note="Shipping CANN aclnnMoeGatingTopKSoftmaxV2 across A2/A3/A5 (AddConfig ascend910b/93/950)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_dequant_situ_quant",
            abi_id="aclnnDequantSituQuant", available=not is_a5, backend="cann_aclnn",
            symbol="aclnnDequantSituQuant", arch=arch,
            role="moe_act_quant", graph_motif="dequant_situ_quant",
            dtype_tuple=(
                ("x:int8:ND:per_token_int8::fp32",),
                ("x:int32:ND:per_token_int8::fp32",),
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
            ),
            outputs=(
                ("y:int8:ND", "y_scale:fp32:ND"),
                ("y:int8:ND", "y_scale:fp32:ND"),
                ("y:int8:ND", "y_scale:fp32:ND"),
                ("y:int8:ND", "y_scale:fp32:ND"),
            ),
            shape=AbiShapeConstraints(require_even_n=True),
            activations=frozenset({"situ", "situglu"}),
            repo_key="ops-nn",
            rel_path="quant/dequant_situ_quant/op_host/dequant_situ_quant_def.cpp",
            va_ref="vllm_ascend/ops/fused_moe/shared_experts.py:945",
            op_ref="vllm_ascend/csrc/moe/dequant_situ_quant/dequant_situ_quant_torch_adpt.h",
            cann_ref="ops-nn/quant/dequant_situ_quant/op_host/dequant_situ_quant_def.cpp:77",
            kernels=("DequantSituQuant",) if not is_a5 else (),
            note="Shipping CANN on A2/A3 (AddConfig ascend910b/93 ONLY); A5 BF16 SiTU unfused (12-14 kernels)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_swiglu_quant",
            abi_id="aclnnSwiGluQuant", available=True, backend="cann_aclnn",
            symbol="aclnnSwiGluQuant", arch=arch,
            role="moe_act_quant", graph_motif="swiglu_quant",
            dtype_tuple=(
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
                ("x:fp32:ND:dense::none",),
                ("x:bf16:ND:dense::none", "group_index:int32:ND:none::none"),
                ("x:fp16:ND:dense::none", "group_index:int32:ND:none::none"),
            ),
            outputs=(
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
            ),
            shape=AbiShapeConstraints(n_max_inclusive=N_MAX_SWIGLU_QUANT, require_even_n=True),
            activations=frozenset({"swiglu"}),
            repo_key="ops-nn",
            rel_path="quant/swi_glu_quant/op_host/swi_glu_quant_def.cpp",
            va_ref="vllm_ascend/ops/fused_moe/moe_mlp.py:82",
            op_ref="op_plugin/ops/opapi/SwigluquantKernelNpuOpApi.cpp:101",
            cann_ref="ops-nn/quant/swi_glu_quant/op_host/swi_glu_quant_def.cpp:65",
            kernels=("SwiGluQuant",),
            note="Post-GMM SwiGLU+Quant ONLY (NEVER absorbs GMM); N even and <= 8192",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_clipped_swiglu",
            abi_id="aclnnClippedSwiglu", available=not is_a5, backend="cann_aclnn",
            symbol="aclnnClippedSwiglu", arch=arch,
            role="moe_act_quant", graph_motif="clipped_swiglu",
            dtype_tuple=(("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",)),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            shape=AbiShapeConstraints(require_even_n=True),
            activations=frozenset({"clipped_swiglu", "swiglu"}),
            repo_key="ops-nn",
            rel_path="activation/clipped_swiglu/op_host/clipped_swiglu_def.cpp",
            va_ref="vllm_ascend/device/device_op.py:1525" if is_a5 else "vllm_ascend/device/device_op.py:535",
            op_ref="op_plugin/config/op_plugin_functions.yaml:7045",
            cann_ref="ops-nn/activation/clipped_swiglu/op_host/clipped_swiglu_def.cpp:45",
            kernels=("ClippedSwiglu",) if not is_a5 else (),
            note=(
                "A5 DeviceAdaptor.clipped_swiglu (device_op.py:1525) runs unfused clamp+sigmoid+mul on BF16"
                if is_a5 else
                "Shipping CANN ClippedSwiglu on A2/A3 (device_op.py:535)"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_add_rmsnorm_dynamic_quant",
            abi_id="aclnnAddRmsNormDynamicQuant", available=True, backend="cann_aclnn",
            symbol="aclnnAddRmsNormDynamicQuantV2", arch=arch,
            role="norm_quant", graph_motif="add_rmsnorm_quant",
            pass_id="norm_quant_fusion",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
            ),
            outputs=(
                ("y1:int8:ND", "y2:int8:ND", "x:bf16:ND", "scale1:fp32:ND", "scale2:fp32:ND"),
                ("y1:int8:ND", "y2:int8:ND", "x:fp16:ND", "scale1:fp32:ND", "scale2:fp32:ND"),
            ),
            repo_key="ops-nn",
            rel_path="norm/add_rms_norm_dynamic_quant/op_host/op_api/aclnn_add_rms_norm_dynamic_quant_v2.cpp",
            va_ref="vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:348",
            op_ref="op_plugin/ops/opapi/AddRmsNormDynamicQuantKernelNpuOpApi.cpp",
            cann_ref="ops-nn/norm/add_rms_norm_dynamic_quant/op_host/op_api/aclnn_add_rms_norm_dynamic_quant_v2.cpp:47",
            kernels=("AddRmsNormDynamicQuant",),
            note="Merged norm_quant_fusion pass (INT8 dynamic quant; skips W4A4)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_add_rmsnorm_dynamic_mx_quant",
            abi_id="aclnnAddRmsNormDynamicMxQuant", available=is_a5, backend="cann_aclnn",
            symbol="aclnnAddRmsNormDynamicMxQuant", arch=arch,
            role="norm_quant", graph_motif="add_rmsnorm_mx_quant",
            pass_id="norm_quant_fusion",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
            ),
            outputs=(
                ("y:fp8_e4m3fn:ND", "x_out:bf16:ND", "scale:fp8_e8m0:ND"),
                ("y:fp8_e4m3fn:ND", "x_out:fp16:ND", "scale:fp8_e8m0:ND"),
            ),
            shape=AbiShapeConstraints(hidden_divisibility=HIDDEN_DIV_INDEXER_QUANT),
            repo_key="ops-nn",
            rel_path="norm/rms_norm_dynamic_mx_quant/op_host/arch35/rms_norm_dynamic_mx_quant_base_tiling.cpp",
            va_ref="vllm_ascend/compilation/passes/norm_quant_fusion_pass.py:393",
            op_ref="op_plugin/ops/opapi/AddRmsNormDynamicMxQuantKernelNpuOpApi.cpp",
            cann_ref="ops-nn/norm/rms_norm_dynamic_mx_quant/op_host/arch35/rms_norm_dynamic_mx_quant_base_tiling.cpp:49",
            kernels=("AddRmsNormDynamicMxQuant",) if is_a5 else (),
            note="A5 ONLY (DAV_3510 MXFP8 group 32 E8M0); rejected on A2/A3",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_quant_matmul_v5",
            abi_id="aclnnQuantMatmulV4", available=True, backend="cann_aclnn",
            symbol="aclnnQuantMatmulV5", arch=arch,
            role="dense_matmul", graph_motif="quant_matmul",
            dtype_tuple=(
                ("x1:int8:ND:per_token_int8::fp32", "x2:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                ("x1:int8:ND:per_token_int8::fp32", "x2:int4:FRACTAL_NZ:w4a8_int::fp32"),
                *(
                    (("x1:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "x2:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),)
                    if is_a5 else ()
                ),
            ),
            outputs=(
                (("y:bf16:ND",),) * 3 if is_a5 else (("y:bf16:ND",),) * 2
            ),
            repo_key="ops-nn",
            rel_path="matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp",
            va_ref="vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:124",
            op_ref="op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp:348",
            cann_ref="ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp:1238",
            kernels=("QuantBatchMatmulV3",),
            note="Quantized linear layers execute QuantBatchMatmulV3 on NPU",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_matmul", abi_id="aclnnMatmul",
            available=True, backend="cann_aclnn", symbol="aclnnMatmul", arch=arch,
            role="dense_matmul", graph_motif="dense_matmul",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none"),
                ("x1:bf16:ND:dense::none", "x2:bf16:FRACTAL_NZ:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",), ("y:bf16:ND",)),
            repo_key="ops-nn",
            rel_path="matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h",
            op_ref="op_plugin/ops/opapi/MatmulKernelNpuOpApi.cpp",
            cann_ref="ops-nn/matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h:25",
            kernels=("MatMulV3",) if is_a5 else ("MatMulV2",),
            note="Dense BF16/FP16 matrix multiplication",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_constant_pad_nd", abi_id="aclnnConstantPadNd",
            available=True, backend="cann_aclnn", symbol="aclnnConstantPadNd", arch=arch,
            role="sp_pad", graph_motif="sp_pad_constant",
            dtype_tuple=(("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",)),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            repo_key="ops-math",
            rel_path="conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp",
            va_ref="vllm_ascend/ops/linear_op.py:218",
            op_ref="op_plugin/ops/opapi/ConstantPadNdKernelNpuOpApi.cpp:91",
            cann_ref="ops-math/conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp:357",
            kernels=("MemSet", "PadV3"),
            note="BF16/FP16 sp_pad > 0 expands to MemSet + PadV3; sp_pad == 0 is 0 PadV3",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_constant_pad_nd_bool", abi_id="aclnnConstantPadNd",
            available=True, backend="cann_aclnn", symbol="aclnnConstantPadNd", arch=arch,
            role="sp_pad", graph_motif="sp_pad_constant_bool",
            dtype_tuple=(("x:bool:ND:none::none",),),
            outputs=(("y:bool:ND",),),
            repo_key="ops-math",
            rel_path="conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp",
            va_ref="vllm_ascend/ops/linear_op.py:218",
            op_ref="op_plugin/ops/opapi/ConstantPadNdKernelNpuOpApi.cpp:91",
            cann_ref="ops-math/conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp:339",
            kernels=("MemSet", "PadV3") if is_a5 else ("Cast", "Cast", "MemSet", "PadV3", "Cast"),
            note=(
                "A5 IsRegBase() bool ConstantPadNd executes MemSet + PadV3 directly"
                if is_a5 else
                "A2/A3 non-RegBase bool ConstantPadNd wraps PadV3 with 3x Cast (self->int8, value->int8, out->bool)"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_inplace_zero", abi_id="aclnnInplaceZero",
            available=True, backend="cann_aclnn", symbol="aclnnInplaceZero", arch=arch,
            role="sp_pad", graph_motif="inplace_zero",
            dtype_tuple=(("self:bf16:ND:dense::none",), ("self:fp16:ND:dense::none",), ("self:fp32:ND:dense::none",)),
            outputs=(("self:bf16:ND",), ("self:fp16:ND",), ("self:fp32:ND",)),
            repo_key="ops-math",
            rel_path="math/zero_op/docs/aclnnInplaceZero_en.md",
            va_ref="vllm_ascend/models/common/ops/sequence_parallel.py:70",
            cann_ref="ops-math/math/zero_op/docs/aclnnInplaceZero_en.md:1",
            kernels=("ZerosLike",),
            note="In-place zero fill launching ZerosLike (live A3 & A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_cat", abi_id="aclnnCat",
            available=True, backend="cann_aclnn", symbol="aclnnCat", arch=arch,
            role="sp_pad", graph_motif="sp_pad_cat_zeros",
            dtype_tuple=(("x:bf16:ND:dense::none", "z:bf16:ND:dense::none"), ("x:fp16:ND:dense::none", "z:fp16:ND:dense::none")),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            repo_key="ops-math",
            rel_path="conversion/concat_d/op_api/aclnn_cat.cpp",
            va_ref="vllm_ascend/models/common/ops/sequence_parallel.py:70",
            cann_ref="ops-math/conversion/concat_d/op_api/aclnn_cat.cpp:546",
            kernels=("ConcatD",),
            note="Tensor concatenation launching ConcatD (paired separately with aclnnInplaceZero -> ZerosLike)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_index", abi_id="aclnnIndex",
            available=True, backend="cann_aclnn", symbol="aclnnIndex", arch=arch,
            role="indexing", graph_motif="index_gather",
            dtype_tuple=(("x:bf16:ND:dense::none", "idx:int32:ND:none::none"), ("x:int32:ND:none::none", "idx:int32:ND:none::none")),
            outputs=(("y:bf16:ND",), ("y:int32:ND",)),
            repo_key="ops-nn",
            rel_path="index/index/op_api/aclnn_index.cpp",
            va_ref="vllm_ascend/attention/context_parallel/sfa_dcp_utils.py:167",
            op_ref="op_plugin/ops/opapi/IndexKernelNpuOpApi.cpp:40",
            cann_ref="ops-nn/index/index/op_api/aclnn_index.cpp:585",
            kernels=("IndexCheck", "Index"),
            note="Multi-kernel aten::index expansion: IndexCheck + Index",
            observed_kernels=observed,
        ),
    ]
    return entries


def _mla_prolog_in_row(
    tok_spec: str, w_spec: str, *, include_rope: bool = True,
) -> tuple[str, ...]:
    """Build exact 11-input (or 9-input doRope=false) MLA Prolog V3/V4 signature."""
    base = (
        f"token_x:{tok_spec}",
        f"weight_dq:{w_spec}",
        f"weight_uq_qr:{w_spec}",
        "weight_uk:bf16:ND:dense::none",
        f"weight_dkv_kr:{w_spec}",
        "rmsnorm_gamma_cq:bf16:ND:dense::none",
        "rmsnorm_gamma_ckv:bf16:ND:dense::none",
    )
    rope = ("rope_sin:bf16:ND:dense::none", "rope_cos:bf16:ND:dense::none") if include_rope else ()
    caches = ("kv_cache_ref:bf16:ND:dense::none", "kr_cache_ref:bf16:ND:dense::none")
    return base + rope + caches


def _cann_attention_and_linear_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    bf16_tok = "bf16:ND:dense::none"
    bf16_w = "bf16:FRACTAL_NZ:dense::none"
    mxfp8_w = "fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"
    int8_tok = "int8:ND:per_token_int8::fp32"
    int8_w = "int8:FRACTAL_NZ:per_channel_int8::fp32"
    entries: list[ProbedAbiEntry] = [
        _mk(
            repos, entry_id=f"{p_low}_cann_mla_prolog_v3", abi_id="aclnnMlaPrologV3WeightNz",
            available=True, backend="cann_aclnn", symbol="aclnnMlaPrologV3WeightNz", arch=arch,
            role="mla_prolog", graph_motif="mla_prolog",
            dtype_tuple=(
                (
                    # Upstream aclnn_mla_prolog_v3_weight_nz.cpp:110,195 & mla_v1.py:1919:
                    # 11 required inputs; A5 supports weightQuantMode {0..5} (BF16/INT8/MXFP8),
                    # A2/A3 support {0,1,2} (BF16/INT8); FP16 is rejected.
                    _mla_prolog_in_row(bf16_tok, bf16_w),
                    _mla_prolog_in_row(bf16_tok, mxfp8_w),
                    _mla_prolog_in_row(int8_tok, int8_w),
                ) if is_a5 else (
                    _mla_prolog_in_row(bf16_tok, bf16_w),
                    _mla_prolog_in_row(int8_tok, int8_w),
                )
            ),
            outputs=(
                (
                    ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:bf16:ND", "dequant_scale_q_norm_out:fp32:ND"),
                    ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:fp8_e4m3fn:ND", "dequant_scale_q_norm_out:fp8_e8m0:ND"),
                    ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:int8:ND", "dequant_scale_q_norm_out:fp32:ND"),
                ) if is_a5 else (
                    ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:bf16:ND", "dequant_scale_q_norm_out:fp32:ND"),
                    ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:int8:ND", "dequant_scale_q_norm_out:fp32:ND"),
                )
            ),
            shape=AbiShapeConstraints(hidden_allowed_set=MLA_PROLOG_HIDDEN_SET),
            repo_key="ops-transformer",
            rel_path="attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp",
            va_ref="vllm_ascend/attention/sfa_v1.py:1056",
            op_ref="op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp:258",
            cann_ref="ops-transformer/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp:194",
            kernels=("MlaPrologV3",),
            note="Exact 11-input/4-output signature (aclnn_mla_prolog_v3_weight_nz.cpp:195); A5 BF16/MXFP8/INT8 (rejects FP16); A2/A3 INT8 NZ",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_mla_prolog_v4", abi_id="aclnnMlaPrologV4WeightNz",
            available=is_a5, backend="cann_aclnn", symbol="aclnnMlaPrologV4WeightNz", arch=arch,
            role="mla_prolog", graph_motif="mla_prolog",
            dtype_tuple=(
                _mla_prolog_in_row(bf16_tok, mxfp8_w, include_rope=True),
                _mla_prolog_in_row(bf16_tok, bf16_w, include_rope=True),
                _mla_prolog_in_row(bf16_tok, mxfp8_w, include_rope=False),
                _mla_prolog_in_row(bf16_tok, bf16_w, include_rope=False),
            ),
            outputs=(
                ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:fp8_e4m3fn:ND", "dequant_scale_q_norm_out:fp8_e8m0:ND"),
                ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:bf16:ND", "dequant_scale_q_norm_out:fp32:ND"),
                ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:fp8_e4m3fn:ND", "dequant_scale_q_norm_out:fp8_e8m0:ND"),
                ("query_out:bf16:ND", "query_rope_out:bf16:ND", "query_norm_out:bf16:ND", "dequant_scale_q_norm_out:fp32:ND"),
            ),
            shape=AbiShapeConstraints(hidden_allowed_set=MLA_PROLOG_HIDDEN_SET),
            repo_key="ops-transformer",
            rel_path="attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.cpp",
            va_ref="vllm_ascend/attention/mla_v1.py:1344",
            op_ref="op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp:258",
            cann_ref="ops-transformer/attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.cpp:196",
            kernels=("MlaPrologV3",) if is_a5 else (),
            note="A5 MLA Prolog V4 NZ: exact 11-input (doRope=true) and 9-input (doRope=false) signatures -> 4 outputs",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_fia", abi_id="aclnnFusedInferAttentionScore",
            available=True, backend="cann_aclnn", symbol="aclnnFusedInferAttentionScore", arch=arch,
            role="attention", graph_motif="paged_attention",
            dtype_tuple=(
                ("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none"),
                ("q:fp16:ND:dense::none", "k:fp16:ND:dense::none", "v:fp16:ND:dense::none"),
                ("q:bf16:ND:dense::none", "k:int8:ND:per_channel_int8::fp32", "v:int8:ND:per_channel_int8::fp32"),
            ),
            outputs=(("out:bf16:ND",), ("out:fp16:ND",), ("out:bf16:ND",)),
            repo_key="ops-transformer",
            rel_path="attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score.cpp",
            va_ref="vllm_ascend/attention/attention_v1.py",
            op_ref="op_plugin/ops/opapi/FusedInferAttentionScoreKernelNpuOpApi.cpp",
            cann_ref="ops-transformer/attention/fused_infer_attention_score/op_api/aclnn_fused_infer_attention_score.cpp:1",
            kernels=("FusedInferAttentionScore",),
            note="Standard CANN FIA attention across A2/A3/A5",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_flash_mla", abi_id="aclnnFlashMla",
            available=is_a5, backend="cann_aclnn", symbol="aclnnFlashMla", arch=arch,
            role="attention", graph_motif="paged_mla",
            dtype_tuple=(
                ("q:bf16:ND:dense::none", "k_cache:bf16:ND:dense::none", "block_table:int32:ND:none::none"),
                ("q:fp16:ND:dense::none", "k_cache:fp16:ND:dense::none", "block_table:int32:ND:none::none"),
                ("q:bf16:ND:dense::none", "k_cache:bf16:ND:dense::none", "block_table:int32:ND:none::none", "cache_seqlens:int32:ND:none::none"),
                ("q:fp16:ND:dense::none", "k_cache:fp16:ND:dense::none", "block_table:int32:ND:none::none", "cache_seqlens:int32:ND:none::none"),
            ),
            outputs=(("attn_out:bf16:ND",), ("attn_out:fp16:ND",), ("attn_out:bf16:ND",), ("attn_out:fp16:ND",)),
            repo_key="ops-transformer",
            rel_path="attention/flash_mla/op_api/aclnn_flash_mla.cpp",
            va_ref="vllm_ascend/attention/flashmla.py:160",
            op_ref="ops-transformer/attention/flash_mla/torch_extension/csrc/flash_mla.cpp:1",
            cann_ref="ops-transformer/attention/flash_mla/op_host/flash_mla_tiling_info_parser.cpp:357",
            kernels=("FlashMla",) if is_a5 else (),
            note="A5 ONLY (AddConfig ascend950; BF16/FP16 only + required paged block_table per flash_mla_tiling_info_parser.cpp:357); unavailable on A2/A3",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_sparse_flash_attention",
            abi_id="aclnnSparseFlashAttention", available=True, backend="cann_aclnn",
            symbol="aclnnSparseFlashAttention", arch=arch,
            role="attention", graph_motif="sparse_mla_attention",
            dtype_tuple=(
                ("query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none", "sparse_indices:int32:ND:none::none"),
                ("query:fp16:ND:dense::none", "key:fp16:ND:dense::none", "value:fp16:ND:dense::none", "sparse_indices:int32:ND:none::none"),
            ),
            outputs=(
                ("attention_out:bf16:ND", "softmax_max:fp32:ND", "softmax_sum:fp32:ND"),
                ("attention_out:fp16:ND", "softmax_max:fp32:ND", "softmax_sum:fp32:ND"),
            ),
            repo_key="ops-transformer",
            rel_path="attention/sparse_flash_attention/op_host/op_api/aclnn_sparse_flash_attention.cpp",
            va_ref="vllm_ascend/attention/sfa_v1.py",
            cann_ref="ops-transformer/attention/sparse_flash_attention/op_host/op_api/aclnn_sparse_flash_attention.cpp:1",
            kernels=("SparseFlashAttention",),
            note="BF16/FP16 sparse flash attention (4 required inputs -> 3 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_gbsa", abi_id="aclnnGenericBlockSparseAttention",
            available=True, backend="cann_aclnn", symbol="aclnnBlockSparseAttention", arch=arch,
            role="attention", graph_motif="block_sparse_attention",
            dtype_tuple=(
                ("query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none"),
                ("query:fp16:ND:dense::none", "key:fp16:ND:dense::none", "value:fp16:ND:dense::none"),
            ),
            outputs=(("out:bf16:ND",), ("out:fp16:ND",)),
            repo_key="ops-transformer",
            rel_path="attention/block_sparse_attention/op_host/op_api/aclnn_block_sparse_attention.cpp",
            va_ref="vllm_ascend/models/qwen4_exp/gbsa.py:86",
            cann_ref="ops-transformer/attention/block_sparse_attention/op_host/block_sparse_attention_tiling.cpp:1",
            kernels=("GenericBlockSparseAttention",),
            note="Paired with GenericBlockSparseAttentionMetadata AI_CPU builder",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_lightning_indexer", abi_id="aclnnLightningIndexer",
            available=True, backend="cann_aclnn", symbol="aclnnLightningIndexer", arch=arch,
            role="sparse_indexer", graph_motif="lightning_indexer",
            dtype_tuple=(
                ("query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "weights:bf16:ND:dense::none"),
                ("query:fp16:ND:dense::none", "key:fp16:ND:dense::none", "weights:fp16:ND:dense::none"),
            ),
            outputs=(
                ("sparse_indices:int32:ND", "sparse_values:fp32:ND"),
                ("sparse_indices:int32:ND", "sparse_values:fp32:ND"),
            ),
            repo_key="ops-transformer",
            rel_path="attention/lightning_indexer/op_host/lightning_indexer_def.cpp",
            va_ref="vllm_ascend/attention/indexer.py",
            cann_ref="ops-transformer/attention/lightning_indexer/op_host/lightning_indexer_def.cpp:1",
            kernels=("LightningIndexer",),
            note="BF16/FP16 dense-weight LightningIndexer across A2/A3/A5 (3 required inputs -> 2 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_inner_quant_lightning_indexer",
            abi_id="aclnnInnerQuantLightningIndexer", available=True, backend="cann_aclnn",
            symbol="aclnnInnerQuantLightningIndexer", arch=arch,
            role="sparse_indexer", graph_motif="quant_lightning_indexer",
            dtype_tuple=(
                (
                    # DAV_3510 exact joint tuples from quant_lightning_indexer_tiling.cpp:438-475 + 5 required inputs
                    ("query:fp8_e4m3fn:ND:per_token_fp8::fp32", "key:fp8_e4m3fn:ND:per_token_fp8::fp32", "weights:bf16:ND:dense::none", "query_dequant_scale:fp32:ND:dense::none", "key_dequant_scale:fp32:ND:dense::none"),
                    ("query:fp8_e4m3fn:ND:per_token_fp8::fp16", "key:fp8_e4m3fn:ND:per_token_fp8::fp16", "weights:fp16:ND:dense::none", "query_dequant_scale:fp16:ND:dense::none", "key_dequant_scale:fp16:ND:dense::none"),
                    ("query:hif8:ND:per_token_fp8::fp32", "key:hif8:ND:per_token_fp8::fp32", "weights:bf16:ND:dense::none", "query_dequant_scale:fp32:ND:dense::none", "key_dequant_scale:fp32:ND:dense::none"),
                    ("query:int8:ND:per_token_int8::fp16", "key:int8:ND:per_token_int8::fp16", "weights:fp16:ND:dense::none", "query_dequant_scale:fp16:ND:dense::none", "key_dequant_scale:fp16:ND:dense::none"),
                ) if is_a5 else (
                    # DAV_2201 exact single tuple from quant_lightning_indexer_tiling.cpp:480-499 (FP16 weights & scales only)
                    ("query:int8:ND:per_token_int8::fp16", "key:int8:ND:per_token_int8::fp16", "weights:fp16:ND:dense::none", "query_dequant_scale:fp16:ND:dense::none", "key_dequant_scale:fp16:ND:dense::none"),
                )
            ),
            outputs=(
                (("sparse_indices:int32:ND",),) * 4 if is_a5 else (("sparse_indices:int32:ND",),)
            ),
            repo_key="ops-transformer",
            rel_path="attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp",
            va_ref="vllm_ascend/attention/indexer.py:934",
            cann_ref="ops-transformer/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp:438",
            kernels=("QuantLightningIndexer",),
            note="Usable on live A5 (FP8_E4M3/HiFloat8/INT8 joint tuples per DAV_3510) and A2/A3 INT8+FP16 weights/scales (5 required inputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_quant_lightning_indexer_v2",
            abi_id="aclnnQuantLightningIndexerV2", available=True, backend="cann_aclnn",
            symbol="aclnnQuantLightningIndexerV2", arch=arch,
            role="sparse_indexer", graph_motif="quant_lightning_indexer_v2",
            dtype_tuple=(
                (
                    # Exact 5 DAV_3510 tuples from quant_lightning_indexer_v2_def.cpp:116-140 & tiling.cpp:504-608
                    ("q:fp8_e4m3fn:ND:per_token_fp8::fp32", "k:fp8_e4m3fn:ND:per_token_fp8::fp32", "w:fp32:ND:dense::none", "q_descale:fp32:ND:dense::none", "k_descale:fp32:ND:dense::none"),
                    ("q:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "k:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "w:fp32:ND:dense::none", "q_descale:fp8_e8m0:ND:none::none", "k_descale:fp8_e8m0:ND:none::none"),
                    ("q:hif8:ND:per_token_fp8::fp32", "k:hif8:ND:per_token_fp8::fp32", "w:fp32:ND:dense::none", "q_descale:fp32:ND:dense::none", "k_descale:fp32:ND:dense::none"),
                    ("q:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "k:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "w:fp32:ND:dense::none", "q_descale:fp8_e8m0:ND:none::none", "k_descale:fp8_e8m0:ND:none::none"),
                    ("q:int8:ND:per_token_int8::fp16", "k:int8:ND:per_token_int8::fp16", "w:fp16:ND:dense::none", "q_descale:fp16:ND:dense::none", "k_descale:fp16:ND:dense::none"),
                ) if is_a5 else (
                    ("q:int8:ND:per_token_int8::fp16", "k:int8:ND:per_token_int8::fp16", "w:fp16:ND:dense::none", "q_descale:fp16:ND:dense::none", "k_descale:fp16:ND:dense::none"),
                )
            ),
            outputs=(
                (("sparse_indices:int32:ND", "sparse_values:bf16:ND"),) * 5
                if is_a5 else
                (("sparse_indices:int32:ND", "sparse_values:bf16:ND"),)
            ),
            repo_key="ops-transformer",
            rel_path="attention/quant_lightning_indexer_v2/op_api/aclnn_quant_lightning_indexer_v2.cpp",
            va_ref="vllm_ascend/device/device_op.py:772",
            cann_ref="ops-transformer/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp:85",
            kernels=("QuantLightningIndexerV2",),
            note="Standard CANN QLI V2 (5 required inputs -> sparse_indices:int32, sparse_values:bf16; A5 FP32/FP16 weights, A2/A3 INT8+FP16)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_indexer_quant_cache",
            abi_id="aclnnIndexerQuantCache", available=is_a5, backend="cann_aclnn",
            symbol="aclnnIndexerQuantCache", arch=arch,
            role="indexer_cache", graph_motif="indexer_quant_cache",
            dtype_tuple=(
                ("cache:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "cache_scale:fp8_e8m0:ND:none::none", "x:bf16:ND:mx_fp8:32:fp8_e8m0", "slot_mapping:int32:ND:none::none"),
                ("cache:fp8_e4m3fn:ND:normal_block::fp32", "cache_scale:fp32:ND:none::none", "x:bf16:ND:normal_block::fp32", "slot_mapping:int32:ND:none::none"),
                ("cache:hif8:ND:hif8_mx:32:fp8_e8m0", "cache_scale:fp8_e8m0:ND:none::none", "x:bf16:ND:hif8_mx:32:fp8_e8m0", "slot_mapping:int32:ND:none::none"),
                ("cache:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "cache_scale:fp8_e8m0:ND:none::none", "x:bf16:ND:mx_fp4:32:fp8_e8m0", "slot_mapping:int32:ND:none::none"),
                ("cache:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "cache_scale:fp8_e8m0:ND:none::none", "x:fp16:ND:mx_fp8:32:fp8_e8m0", "slot_mapping:int32:ND:none::none"),
            ),
            outputs=(
                ("cache:fp8_e4m3fn:ND", "cache_scale:fp8_e8m0:ND"),
                ("cache:fp8_e4m3fn:ND", "cache_scale:fp32:ND"),
                ("cache:hif8:ND", "cache_scale:fp8_e8m0:ND"),
                ("cache:fp4_e2m1:ND", "cache_scale:fp8_e8m0:ND"),
                ("cache:fp8_e4m3fn:ND", "cache_scale:fp8_e8m0:ND"),
            ),
            shape=AbiShapeConstraints(
                hidden_range=(HIDDEN_DIV_INDEXER_QUANT, HIDDEN_MAX_INDEXER_QUANT),
                hidden_divisibility=HIDDEN_DIV_INDEXER_QUANT,
            ),
            repo_key="ops-transformer",
            rel_path="attention/indexer_quant_cache/op_host/indexer_quant_cache_def.cpp",
            va_ref="vllm_ascend/attention/indexer.py",
            cann_ref="ops-transformer/attention/indexer_quant_cache/op_host/indexer_quant_cache_def.cpp:94",
            kernels=("IndexerQuantCache",) if is_a5 else (),
            note="A5 ONLY (AddConfig ascend950; 4 required inputs -> cache, cache_scale); A3 uses DynamicQuant + 2x ScatterNdUpdate",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_ring_attention_update_v2",
            abi_id="aclnnRingAttentionUpdateV2", available=True, backend="cann_aclnn",
            symbol="aclnnRingAttentionUpdate", arch=arch,
            role="dcp_exchange", graph_motif="ring_attention_update",
            dtype_tuple=(
                (
                    "prev_attn_out:bf16:ND:dense::none", "prev_softmax_max:fp32:ND:dense::none",
                    "prev_softmax_sum:fp32:ND:dense::none", "cur_attn_out:bf16:ND:dense::none",
                    "cur_softmax_max:fp32:ND:dense::none", "cur_softmax_sum:fp32:ND:dense::none",
                ),
                (
                    "prev_attn_out:fp16:ND:dense::none", "prev_softmax_max:fp32:ND:dense::none",
                    "prev_softmax_sum:fp32:ND:dense::none", "cur_attn_out:fp16:ND:dense::none",
                    "cur_softmax_max:fp32:ND:dense::none", "cur_softmax_sum:fp32:ND:dense::none",
                ),
            ),
            outputs=(
                ("attn_out:bf16:ND", "softmax_max:fp32:ND", "softmax_sum:fp32:ND"),
                ("attn_out:fp16:ND", "softmax_max:fp32:ND", "softmax_sum:fp32:ND"),
            ),
            repo_key="ops-transformer",
            rel_path="attention/ring_attention_update/op_host/op_api/aclnn_ring_attention_update.cpp",
            cann_ref="ops-transformer/attention/ring_attention_update/op_host/ring_attention_update_tiling.cpp:1",
            kernels=("RingAttentionUpdate",),
            note="CANN RingAttentionUpdate across A2/A3/A5 (6 required inputs -> 3 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_chunk_gdn", abi_id="aclnnChunkGatedDeltaRule",
            available=True, backend="cann_aclnn", symbol="aclnnChunkGatedDeltaRule", arch=arch,
            role="linear_attention", graph_motif="chunk_gdn",
            dtype_tuple=(
                (
                    "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                    "beta:bf16:ND:dense::none", "initial_state:bf16:ND:dense::none",
                    "actual_seq_lengths:int32:ND:none::none",
                ),
                *(
                    ((
                        "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                        "beta:bf16:ND:dense::none", "initial_state:fp32:ND:dense::none",
                        "actual_seq_lengths:int32:ND:none::none",
                    ),)
                    if is_a5 else ()
                ),
            ),
            outputs=(
                ("out:bf16:ND", "final_state:bf16:ND"),
                *((("out:bf16:ND", "final_state:fp32:ND"),) if is_a5 else ()),
            ),
            repo_key="ops-transformer",
            rel_path="attention/chunk_gated_delta_rule/op_host/op_api/aclnn_chunk_gated_delta_rule.h",
            va_ref="vllm_ascend/ops/gdn.py",
            cann_ref="ops-transformer/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_tiling.cpp:369",
            kernels=("ChunkGatedDeltaRule",),
            note="Canonical CANN aclnnChunkGatedDeltaRule (BF16-only Q/K/V/beta/out, int32 actual_seq_lengths; FP32 state A5-only per tiling.cpp:369)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_recurrent_gdn", abi_id="aclnnRecurrentGatedDeltaRule",
            available=True, backend="cann_aclnn", symbol="aclnnRecurrentGatedDeltaRule", arch=arch,
            role="linear_attention", graph_motif="recurrent_gdn",
            dtype_tuple=(
                (
                    "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                    "beta:bf16:ND:dense::none", "state:fp32:ND:dense::none",
                    "actual_seq_lengths:int32:ND:none::none", "ssm_state_indices:int32:ND:none::none",
                ),
                (
                    "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                    "beta:bf16:ND:dense::none", "state:bf16:ND:dense::none",
                    "actual_seq_lengths:int32:ND:none::none", "ssm_state_indices:int32:ND:none::none",
                ),
            ),
            outputs=(("out:bf16:ND", "state:fp32:ND"), ("out:bf16:ND", "state:bf16:ND")),
            repo_key="ops-transformer",
            rel_path="attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_def.cpp",
            va_ref="vllm_ascend/ops/gdn.py",
            cann_ref="ops-transformer/attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_def.cpp:22",
            kernels=("RecurrentGatedDeltaRule",),
            note="Decode GDN CANN kernel (BF16-only Q/K/V/beta/out, FP32/BF16 state; 7 required inputs -> 2 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_text_gated_residual", abi_id="aclnnTextGatedResidual",
            available=is_a5, backend="cann_aclnn", symbol="aclnnTextGatedResidual", arch=arch,
            role="residual_mixer", graph_motif="text_gated_residual",
            # Exact required inputs from ops-transformer/mhc/text_gated_residual/op_host/op_api/aclnn_text_gated_residual.h:42
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "w_down:bf16:ND:dense::none", "w_up:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "w_down:fp16:ND:dense::none", "w_up:fp16:ND:dense::none"),
            ),
            outputs=(("mixed:bf16:ND",), ("mixed:fp16:ND",)),
            repo_key="ops-transformer",
            rel_path="mhc/text_gated_residual/op_host/op_api/aclnn_text_gated_residual.h",
            cann_ref="ops-transformer/mhc/text_gated_residual/op_host/op_api/aclnn_text_gated_residual.h:42",
            kernels=("TextGatedResidual",) if is_a5 else (),
            note="A5 arch35 text-gated residual (3 required inputs x, w_down, w_up -> mixed)",
            observed_kernels=observed,
        ),
    ]
    return entries


def _vllm_ascend_csrc_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    entries: list[ProbedAbiEntry] = [
        _mk(
            repos, entry_id=f"{p_low}_csrc_attn_res_fwd", abi_id="AttnResFwd",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnAttnResFwd", arch=arch,
            role="residual_mixer", graph_motif="attn_res_prefix",
            # Exact AscendC OpDef order from csrc/attention/attn_res_fwd/op_host/attn_res_fwd_def.cpp:7-14
            dtype_tuple=((
                "prefix_sum:bf16:ND:dense::none",
                "block_residual:bf16:ND:dense::none",
                "proj_weight:bf16:ND:dense::none",
                "norm_weight:bf16:ND:dense::none",
                "addend:bf16:ND:dense::none",
                "output_norm:bf16:ND:dense::none",
            ),),
            outputs=((
                "hidden_states:bf16:ND",
                "prefix_out:bf16:ND",
                "materialized:bf16:ND",
            ),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/attn_res_fwd/op_host/attn_res_fwd_def.cpp",
            va_ref="vllm_ascend/models/kimi_k3/kimi_k3.py:848",
            kernels=("AttnResFwd",),
            note="AscendC AttnResFwd OpDef order across A2/A3/A5 (6 BF16 inputs -> 3 BF16 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_attn_res_fwd_torch", abi_id="attn_res_fwd",
            available=True, backend="vllm_ascend_csrc", symbol="_C_ascend.attn_res_fwd", arch=arch,
            role="residual_mixer", graph_motif="attn_res_prefix",
            # Exact Torch callable order from csrc/torch_binding.cpp:2884 (addend & output_norm_weight optional)
            dtype_tuple=(
                (
                    "prefix_sum:bf16:ND:dense::none",
                    "addend:bf16:ND:dense::none",
                    "block_residual:bf16:ND:dense::none",
                    "proj_weight:bf16:ND:dense::none",
                    "norm_weight:bf16:ND:dense::none",
                    "output_norm_weight:bf16:ND:dense::none",
                ),
                (
                    "prefix_sum:bf16:ND:dense::none",
                    "addend:bf16:ND:dense::none",
                    "block_residual:bf16:ND:dense::none",
                    "proj_weight:bf16:ND:dense::none",
                    "norm_weight:bf16:ND:dense::none",
                ),
                (
                    "prefix_sum:bf16:ND:dense::none",
                    "block_residual:bf16:ND:dense::none",
                    "proj_weight:bf16:ND:dense::none",
                    "norm_weight:bf16:ND:dense::none",
                    "output_norm_weight:bf16:ND:dense::none",
                ),
                (
                    "prefix_sum:bf16:ND:dense::none",
                    "block_residual:bf16:ND:dense::none",
                    "proj_weight:bf16:ND:dense::none",
                    "norm_weight:bf16:ND:dense::none",
                ),
            ),
            outputs=(
                ("hidden_states:bf16:ND", "prefix_out:bf16:ND", "materialized:bf16:ND"),
                ("hidden_states:bf16:ND", "prefix_out:bf16:ND", "materialized:bf16:ND"),
                ("hidden_states:bf16:ND", "prefix_out:bf16:ND", "materialized:bf16:ND"),
                ("hidden_states:bf16:ND", "prefix_out:bf16:ND", "materialized:bf16:ND"),
            ),
            repo_key="vllm-ascend",
            rel_path="csrc/torch_binding.cpp",
            va_ref="csrc/torch_binding.cpp:2884",
            kernels=("AttnResFwd",),
            note="Torch binding _C_ascend.attn_res_fwd callable order (optional addend & output_norm_weight)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_hc_pre", abi_id="hc_pre_sinkhorn",
            available=True, backend="vllm_ascend_csrc", symbol="_C_ascend.hc_pre", arch=arch,
            role="residual_mixer", graph_motif="mhc_pre",
            # Exact required signature from csrc/moe/hc_pre/op_host/hc_pre_def.cpp:24-68
            dtype_tuple=((
                "x:bf16:ND:dense::none",
                "hc_fn:fp32:ND:dense::none",
                "hc_scale:fp32:ND:dense::none",
                "hc_base:fp32:ND:dense::none",
            ),),
            outputs=(("y:bf16:ND", "post:fp32:ND", "comb_frag:fp32:ND"),),
            shape=AbiShapeConstraints(hidden_divisibility=HIDDEN_DIV_INDEXER_QUANT),
            repo_key="vllm-ascend",
            rel_path="csrc/moe/hc_pre/op_host/hc_pre_def.cpp",
            va_ref="vllm_ascend/models/glm5next/model.py:535",
            kernels=("HcPre",),
            note="Fused mHC pre-mix (MIX_AIC) verified live on A3 and A5; 4 required inputs -> 3 required outputs",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_hc_post", abi_id="HcPost",
            available=True, backend="vllm_ascend_csrc", symbol="_C_ascend.hc_post", arch=arch,
            role="residual_mixer", graph_motif="mhc_post",
            # Exact required signature from csrc/moe/hc_post/op_host/hc_post_def.cpp:24-48
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "residual:bf16:ND:dense::none", "post:fp32:ND:dense::none", "comb:fp32:ND:dense::none"),
                ("x:bf16:ND:dense::none", "residual:bf16:ND:dense::none", "post:bf16:ND:dense::none", "comb:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "residual:fp16:ND:dense::none", "post:fp32:ND:dense::none", "comb:fp32:ND:dense::none"),
                ("x:fp16:ND:dense::none", "residual:fp16:ND:dense::none", "post:fp16:ND:dense::none", "comb:fp16:ND:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:bf16:ND",), ("y:fp16:ND",), ("y:fp16:ND",)),
            repo_key="vllm-ascend",
            rel_path="csrc/moe/hc_post/op_host/hc_post_def.cpp",
            va_ref="vllm_ascend/models/glm5next/model.py:561",
            kernels=("HcPost",),
            note="Fused mHC post-mix (AI_VECTOR_CORE); 4 required inputs (x, residual, post, comb) -> y",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_add_rms_norm_bias", abi_id="AddRmsNormBias",
            available=True, backend="vllm_ascend_csrc", symbol="_C_ascend.add_rms_norm_bias", arch=arch,
            role="norm_quant", graph_motif="add_rmsnorm_bias",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
            ),
            outputs=(
                ("y:bf16:ND", "rstd:fp32:ND", "x_out:bf16:ND"),
                ("y:fp16:ND", "rstd:fp32:ND", "x_out:fp16:ND"),
            ),
            repo_key="vllm-ascend",
            rel_path="csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h",
            va_ref="csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h:49",
            kernels=("AddRmsNormBias",),
            note="Shipping vllm-ascend csrc AddRmsNormBias across A2/A3/A5 (3 required inputs -> 3 outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_kv_quant_sfa_vllm",
            abi_id="aclnnKvQuantSparseFlashAttention", available=True, backend="vllm_ascend_csrc",
            symbol="aclnnKvQuantSparseFlashAttentionVllm", arch=arch,
            role="attention", graph_motif="kv_quant_sfa",
            dtype_tuple=(
                ("q:bf16:ND:dense::none", "kv:fp8_e4m3fn:ND:per_block_128:128:fp32")
                if is_a5 else
                ("q:bf16:ND:dense::none", "kv:int8:ND:per_block_128:128:fp32"),
            ),
            outputs=(("out:bf16:ND", "lse:fp32:ND"),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/kv_quant_sparse_flash_attention_vllm/kv_quant_sparse_flash_attention_vllm_torch_adpt.h",
            va_ref="vllm_ascend/device/device_op.py:72",
            kernels=("KvQuantSparseFlashAttentionVllm",),
            note="Shipping vllm-ascend KvQuantSparseFlashAttentionVllm decode/prefill kernel",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_sparse_flash_mla", abi_id="SparseFlashMla",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnSparseFlashMla", arch=arch,
            role="attention", graph_motif="sparse_flash_mla",
            # Exact OpDef registration from sparse_flash_mla_def.cpp:24-121: DT_BF16 ONLY (no FP16, no INT8).
            dtype_tuple=(
                ("q:bf16:ND:dense::none",),
                ("q:bf16:ND:dense::none", "ori_kv:bf16:ND:dense::none"),
                ("q:bf16:ND:dense::none", "ori_kv:bf16:ND:dense::none", "cmp_kv:bf16:ND:dense::none"),
                ("q:bf16:ND:dense::none", "ori_kv:bf16:ND:dense::none"),
                ("q:bf16:ND:dense::none", "ori_kv:bf16:ND:dense::none", "cmp_kv:bf16:ND:dense::none"),
            ),
            outputs=(
                ("attn_out:bf16:ND",),
                ("attn_out:bf16:ND",),
                ("attn_out:bf16:ND",),
                ("attn_out:bf16:ND", "softmax_lse:fp32:ND"),
                ("attn_out:bf16:ND", "softmax_lse:fp32:ND"),
            ),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp",
            va_ref="vllm_ascend/attention/dsa_v41.py:744",
            kernels=("SparseFlashMla",),
            note="Exact OpDef registration (sparse_flash_mla_def.cpp:24): DT_BF16 ONLY for q/ori_kv/cmp_kv/attn_out across A2/A3/A5",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_vllm_qli_v2",
            abi_id="VllmAscendQuantLightningIndexerV2", available=not is_a5,
            backend="vllm_ascend_csrc", symbol="aclnnVllmAscendQuantLightningIndexerV2", arch=arch,
            role="sparse_indexer", graph_motif="sharded_qli_v2",
            dtype_tuple=((
                "q:int8:ND:per_token_int8::fp16", "k:int8:ND:per_token_int8::fp16",
                "w:fp16:ND:dense::none", "q_descale:fp16:ND:dense::none", "k_descale:fp16:ND:dense::none",
            ),),
            outputs=(("sparse_indices:int32:ND", "sparse_values:fp32:ND"),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/quant_lightning_indexer_v2/op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp",
            va_ref="vllm_ascend/device/device_op.py:1785",
            kernels=("VllmAscendQuantLightningIndexerV2",) if not is_a5 else (),
            note=(
                "Absent in current A5 container libopapi.so (missing .so symbol; aclnnInnerQuantLightningIndexer IS available)"
                if is_a5 else
                "Shipping A3 vllm-ascend csrc sharded QLI V2 (5 required inputs -> sparse_indices, sparse_values)"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_dispatch_ffn_combine", abi_id="dispatch_ffn_combine",
            available=(arch == "A3"), backend="vllm_ascend_csrc",
            symbol="_C_ascend.dispatch_ffn_combine", arch=arch,
            role="moe_fused_frame", graph_motif="moe_mc2_frame",
            dtype_tuple=(
                (
                    "x:bf16:ND:dense::none", "weight1:int8:FRACTAL_NZ:per_channel_int8::fp32",
                    "weight2:int8:FRACTAL_NZ:per_channel_int8::fp32", "expert_idx:int32:ND:none::none",
                    "scale1:fp32:ND:dense::none", "scale2:fp32:ND:dense::none", "probs:fp32:ND:dense::none",
                ),
                (
                    "x:bf16:ND:dense::none", "weight1:int4:FRACTAL_NZ:w4a8_int::fp32",
                    "weight2:int4:FRACTAL_NZ:w4a8_int::fp32", "expert_idx:int32:ND:none::none",
                    "scale1:fp32:ND:dense::none", "scale2:fp32:ND:dense::none",
                    "bias1:fp32:ND:dense::none", "bias2:fp32:ND:dense::none", "probs:fp32:ND:dense::none",
                ),
            ),
            outputs=(
                ("out:bf16:ND", "expert_token_nums:int64:ND"),
                ("out:bf16:ND", "expert_token_nums:int64:ND"),
            ),
            placements=PLACEMENTS_A3, world_sizes=WS_EP_A23,
            activations=frozenset({"swiglu"}),
            repo_key="vllm-ascend",
            rel_path="csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h",
            va_ref="vllm_ascend/ops/fused_moe/moe_comm_method.py:654",
            kernels=("DispatchFfnCombine",) if arch == "A3" else (),
            note="A3 enable_fused_mc2=1 with active_mask (7 INT8 / 9 INT4 required inputs -> out, expert_token_nums); superseded by MegaMoE on A5",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_key_pool", abi_id="KeyPool",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnKeyPool", arch=arch,
            role="sparse_indexer", graph_motif="kpool_indexer",
            dtype_tuple=(("k:bf16:ND:dense::none",), ("k:int8:ND:per_block_128:128:fp32",)),
            outputs=(("pooled_k:bf16:ND",), ("pooled_k:int8:ND",)),
            repo_key="vllm-ascend",
            rel_path="vllm_ascend/attention/indexer_kpool.py",
            va_ref="vllm_ascend/attention/indexer_kpool.py:789",
            kernels=("KeyPool",),
            note="M10 HARD LEGALITY: KPool indexer requires DCP == 1 (indexer_kpool.py:789-792)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_msa_index_score", abi_id="MsaIndexScore",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnMsaIndexScore", arch=arch,
            role="sparse_indexer", graph_motif="msa_index_score",
            dtype_tuple=(("q:bf16:ND:dense::none", "k:bf16:ND:dense::none"),),
            outputs=(("scores:fp32:ND",),),
            repo_key="vllm-ascend",
            rel_path="vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py",
            va_ref="vllm_ascend/models/minimax_m3/ops/msa_m3_npu.py:97",
            kernels=("MsaIndexScore",),
            note="Block-max MSA indexer score verified live on A3 and A5",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_indexer_region_score", abi_id="IndexerRegionScore",
            available=is_a5, backend="vllm_ascend_csrc", symbol="aclnnIndexerRegionScore", arch=arch,
            role="sparse_indexer", graph_motif="region_index_score",
            # Exact 9 combinations from indexer_region_score_def.cpp:9-27 (AddConfig ascend950).
            dtype_tuple=tuple(
                (
                    f"query:{qdt}:ND:{qm}::none" if qm == "dense" else f"query:{qdt}:ND:{qm}::fp32",
                    f"weights:{wdt}:ND:dense::none",
                    f"key_summary:{qdt}:ND:{qm}::none" if qm == "dense" else f"key_summary:{qdt}:ND:{qm}::fp32",
                    "actual_kv_lengths:int32:ND:none::none",
                    "block_table:int32:ND:none::none",
                    "cu_seq_lens:int32:ND:none::none",
                )
                for qdt, qm in (("fp8_e4m3fn", "per_token_fp8"), ("bf16", "dense"), ("fp16", "dense"))
                for wdt in ("fp16", "bf16", "fp32")
            ),
            outputs=(("output:fp32:ND", "selected_region_ids:int32:ND"),) * 9,
            repo_key="vllm-ascend",
            rel_path="csrc/attention/indexer_region_score/op_host/indexer_region_score_def.cpp",
            va_ref="csrc/attention/indexer_region_score/op_host/indexer_region_score_def.cpp:9",
            kernels=("IndexerRegionScore",) if is_a5 else (),
            note="Region sparse indexer score (ascend950 OpDef; 9 exact dtype triples x 6 required inputs -> output, selected_region_ids)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_recurrent_kda", abi_id="RecurrentKda",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnRecurrentKda", arch=arch,
            role="linear_attention", graph_motif="recurrent_kda",
            dtype_tuple=(("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none"),),
            outputs=(("out:bf16:ND",),),
            repo_key="vllm-ascend",
            rel_path="vllm_ascend/ops/kda.py",
            va_ref="vllm_ascend/ops/kda.py:60",
            kernels=("RecurrentKda",),
            note="KDA decode recurrent kernel verified live on A3 and A5",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_csrc_chunk_kda_fwd", abi_id="ChunkKdaFwd",
            available=True, backend="vllm_ascend_csrc", symbol="aclnnChunkKdaFwdV2", arch=arch,
            role="linear_attention", graph_motif="chunk_kda",
            dtype_tuple=((
                "q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none",
                "g:fp32:ND:dense::none", "beta:bf16:ND:dense::none",
            ),),
            outputs=(("attn_out:bf16:ND",),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/chunk_kda_fwd/docs/api.md",
            va_ref="vllm_ascend/ops/kda.py:60",
            kernels=("ChunkKdaFwdPrepare", "ChunkFwdH", "ChunkKdaFwdFinalize"),
            note="Prefill KDA: 5 required inputs (q, k, v, g, beta) -> attn_out (ordered 3-kernel aclnnChunkKdaFwdV2 expansion)",
            observed_kernels=observed,
        ),
        # AI_CPU sparse metadata kernels (M11 / A5-M7).
        _mk(
            repos, entry_id=f"{p_low}_aicpu_sparse_flash_mla_metadata",
            abi_id="SparseFlashMlaMetadata", available=True, backend="vllm_ascend_csrc",
            symbol="aclnnSparseFlashMlaMetadata", arch=arch,
            role="ai_cpu_metadata", graph_motif="smla_metadata",
            dtype_tuple=(("seq_lens:int32:ND:none::none",),),
            outputs=(("metadata:int32:ND",),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/sparse_flash_mla_metadata/op_host/op_api/aclnn_sparse_flash_mla_metadata.cpp",
            va_ref="vllm_ascend/attention/sfa_v1.py:88",
            kernels=("SparseFlashMlaMetadata",),
            note=(
                "AI_CPU metadata (~217 us on A5): returns 2048 int32 on A5 (upstream sfa_v1.py:88 hardcodes 1024)"
                if is_a5 else
                "AI_CPU metadata (~496 us on A3): returns 1024 int32 on A2/A3"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_aicpu_gbsa_metadata",
            abi_id="GenericBlockSparseAttentionMetadata", available=True,
            backend="vllm_ascend_csrc", symbol="aclnnGenericBlockSparseAttentionMetadata", arch=arch,
            role="ai_cpu_metadata", graph_motif="gbsa_metadata",
            dtype_tuple=(("sparse_block_idx:int32:ND:none::none", "sparse_block_count:int32:ND:none::none"),),
            outputs=(("metadata:int32:ND",),),
            repo_key="vllm-ascend",
            rel_path="csrc/aclnn_torch_adapter/gbsa_torch_adpt.h",
            va_ref="vllm_ascend/models/qwen4_exp/gbsa.py:86",
            kernels=("GenericBlockSparseAttentionMetadata",),
            note="AI_CPU metadata (~139-140 us A3; ~63 us dec / 710 us pre A5)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_aicpu_vllm_qli_v2_metadata",
            abi_id="VllmAscendQuantLightningIndexerV2Metadata", available=not is_a5,
            backend="vllm_ascend_csrc", symbol="aclnnVllmAscendQuantLightningIndexerV2Metadata",
            arch=arch, role="ai_cpu_metadata", graph_motif="qli_v2_metadata",
            dtype_tuple=(("seq_lens:int32:ND:none::none",),),
            outputs=(("metadata:int32:ND",),),
            repo_key="vllm-ascend",
            rel_path="csrc/attention/quant_lightning_indexer_v2_metadata/op_host/quant_lightning_indexer_v2_metadata_infershape.cpp",
            va_ref="vllm_ascend/device/device_op.py:1785",
            kernels=("VllmAscendQuantLightningIndexerV2Metadata",) if not is_a5 else (),
            note=(
                "Absent in current A5 container libopapi.so"
                if is_a5 else
                "AI_CPU metadata (~105-162 us on A3)"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_aicpu_lightning_indexer_v2_metadata",
            abi_id="LightningIndexerV2Metadata", available=True,
            backend="vllm_ascend_csrc", symbol="aclnnLightningIndexerV2Metadata", arch=arch,
            role="ai_cpu_metadata", graph_motif="li_v2_metadata",
            dtype_tuple=(("seq_lens:int32:ND:none::none",),),
            outputs=(("metadata:int32:ND",),),
            repo_key="vllm-ascend",
            rel_path="vllm_ascend/models/qwen4_exp/lightning_indexer.py",
            va_ref="vllm_ascend/models/qwen4_exp/lightning_indexer.py:1",
            kernels=("LightningIndexerV2Metadata",),
            note="AI_CPU metadata (~117 us on A3; live verified on A3 & A5)",
            observed_kernels=observed,
        ),
    ]
    return entries


def _triton_and_other_non_cann_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    # Exact per-kernel tensor signatures grounded in vllm_ascend/ops/triton/** (zero shared unary stubs).
    triton_specs: list[tuple[
        str, str, str, str, tuple[str, ...], bool, str,
        tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...],
    ]] = [
        ("kda_chunk_triton", "vllm_ascend.ops.triton.kda.kda.chunk_kda", "linear_attention",
         "vllm_ascend/ops/triton/kda/kda.py",
         (
             "chunk_local_cumsum_vector_kernel",
             "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
             "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
             "merge_16x16_to_64x64_inverse_kernel_kda",
             "recompute_w_u_fwd_kernel",
             "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda",
             "chunk_gla_fwd_kernel_o",
         ),
         True, "Triton KDA chunk 7-kernel ordered launch chain",
         (("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none", "g:fp32:ND:dense::none", "beta:bf16:ND:dense::none"),),
         (("out:bf16:ND", "final_state:fp32:ND"),)),
        ("kda_recoverssm_verify", "_kda_recoverssm_verify_kernel", "recoverssm",
         "vllm_ascend/ops/triton/kda/recoverssm.py", ("_kda_recoverssm_verify_kernel",), True,
         "KDA RecoverSSM verify/commit kernel (live A5 verified)",
         (("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none", "raw_g:fp32:ND:dense::none", "raw_beta:bf16:ND:dense::none", "state:fp32:ND:dense::none"),),
         (("out:bf16:ND",),)),
        ("prepare_commit_plan", "_prepare_commit_plan_kernel", "recoverssm",
         "vllm_ascend/ops/triton/mamba/prepare_commit_plan.py", ("_prepare_commit_plan_kernel",), True,
         "RecoverSSM commit plan preparation kernel",
         (("num_accepted:int32:ND:none::none", "request_indices:int32:ND:none::none", "state_indices:int32:ND:none::none", "query_start_loc:int32:ND:none::none"),),
         (("commit_lens:int32:ND", "final_state_indices:int32:ND"),)),
        ("gdn_recoverssm_commit", "_commit_gdn_state_kernel", "recoverssm",
         "vllm_ascend/ops/triton/gdn/gdn_recoverssm_commit.py",
         ("_commit_gdn_state_kernel",), True,
         "GDN RecoverSSM state commit kernel",
         (("state:fp32:ND:dense::none", "replay:fp32:ND:dense::none", "state_indices:int32:ND:none::none"),),
         (("state_out:fp32:ND",),)),
        ("gdn_recoverssm_verify", "_gdn_recoverssm_verify_kernel", "recoverssm",
         "vllm_ascend/ops/triton/gdn/gdn_recoverssm_verify.py", ("_gdn_recoverssm_verify_kernel",), True,
         "GDN RecoverSSM verify kernel",
         (("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none", "state:fp32:ND:dense::none", "query_start_loc:int32:ND:none::none"),),
         (("out:bf16:ND", "replay:fp32:ND"),)),
        ("minimax_qknorm_rope_cache", "_minimax_qknorm_rope_cache_kernel", "attention",
         "vllm_ascend/ops/triton/msa/minimax_qknorm_rope_cache.py", ("_minimax_qknorm_rope_cache_kernel",), True,
         "Fused QK-norm + RoPE + cache store Triton kernel (live A3 & A5)",
         (("packed:bf16:ND:dense::none", "cos_sin:bf16:ND:dense::none", "positions:int64:ND:none::none", "q_weight:bf16:ND:dense::none", "k_weight:bf16:ND:dense::none"),),
         (("query:bf16:ND", "key_cache:bf16:ND", "value_cache:bf16:ND"),)),
        ("swiglustep", "_swiglustep_kernel", "moe_act_quant",
         "vllm_ascend/ops/triton/activation/swiglustep.py", ("_swiglustep_kernel",), is_a5,
         "Fused SwiGLUStep Triton kernel (passes live A5; A3 container MLIR blocked)",
         (("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",)),
         (("out:bf16:ND",), ("out:fp16:ND",))),
        ("split_qkv_norm_rope_192", "_split_qkv_norm_rope_192_kernel", "attention",
         "vllm_ascend/ops/triton/step5/fused_qknorm_rope.py", ("_split_qkv_norm_rope_192_kernel",), is_a5,
         "Fused QKV split + norm + RoPE Triton kernel (live A5 verified)",
         (("qkv:bf16:ND:dense::none", "q_weight:bf16:ND:dense::none", "k_weight:bf16:ND:dense::none", "cos:bf16:ND:dense::none", "sin:bf16:ND:dense::none", "positions:int64:ND:none::none"),),
         (("q_out:bf16:ND", "k_out:bf16:ND", "v_out:bf16:ND"),)),
        ("hc_combine_norm", "_hc_combine_norm_kernel", "residual_mixer",
         "vllm_ascend/ops/triton/qwen4_exp/hc.py", ("_hc_combine_norm_kernel",), True,
         "Shipping Triton low-rank HC combine+norm kernel (live A3 & A5)",
         (("residual:bf16:ND:dense::none", "block_output:bf16:ND:dense::none", "injection:bf16:ND:dense::none", "weight:bf16:ND:dense::none"),),
         (("combined:bf16:ND", "normalized:bf16:ND"),)),
        ("hc_gate_mix", "_hc_gate_mix_kernel", "residual_mixer",
         "vllm_ascend/ops/triton/qwen4_exp/hc.py", ("_hc_gate_mix_kernel",), True,
         "Shipping Triton low-rank HC gate-mix kernel (live A3 & A5)",
         (("x:bf16:ND:dense::none", "gate:bf16:ND:dense::none"), ("x:fp16:ND:dense::none", "gate:fp16:ND:dense::none")),
         (("output:bf16:ND",), ("output:fp16:ND",))),
        ("prepare_local_sort_inputs", "_prepare_local_sort_inputs", "sharded_sort",
         "vllm_ascend/ops/triton/sharded_indexer.py",
         ("_prepare_local_sort_inputs",),
         True, "Sharded DCP sort input prep Triton kernel (sharded_indexer.py:439 allocates FP32 values)",
         (("indices:int32:ND:none::none", "scores:fp32:ND:dense::none"),),
         (("keys:fp32:ND", "values:fp32:ND"),)),
        ("normalize_dcp_candidates", "_normalize_dcp_candidates", "sharded_sort",
         "vllm_ascend/ops/triton/sharded_indexer.py",
         ("_normalize_dcp_candidates",), True,
         "Shipping Triton sharded indexer candidate normalizer kernel (live A3 & A5)",
         (("indices:int32:ND:none::none", "scores:fp32:ND:dense::none", "qsl:int32:ND:none::none"),),
         (("indices_out:int32:ND", "scores_out:fp32:ND"),)),
        ("pack_candidates", "_pack_candidates", "sharded_sort",
         "vllm_ascend/ops/triton/sharded_indexer.py",
         ("_pack_candidates",), True,
         "Shipping Triton sharded indexer pre-AllGather pack kernel (live A3 & A5)",
         (("indices:int32:ND:none::none", "scores:fp32:ND:dense::none"),),
         (("packed:int64:ND",),)),
        ("remap_sort", "_remap_sort", "sharded_sort",
         "vllm_ascend/ops/triton/sparse_index_remap.py", ("_remap_sort",), True,
         "Replicated DCP index remap kernel (live A3 & A5)",
         (("indices:int32:ND:none::none",),),
         (("output:int32:ND",),)),
        ("q_gather_prep", "_q_gather_prep_head_major_kernel", "dcp_exchange",
         "vllm_ascend/ops/triton/query_gather_prep.py", ("_q_gather_prep_head_major_kernel",), True,
         "DCP Q gather head-major pack kernel (live A3 & A5)",
         (("q_nope:bf16:ND:dense::none", "q_rope:bf16:ND:dense::none"), ("q_nope:fp16:ND:dense::none", "q_rope:fp16:ND:dense::none")),
         (("packed_q:bf16:ND",), ("packed_q:fp16:ND",))),
        ("pack_dcp_output_lse", "_pack_dcp_output_lse_kernel", "dcp_exchange",
         "vllm_ascend/ops/triton/dcp/dcp_a2a.py", ("_pack_dcp_output_lse_kernel",), True,
         "Generic DCP output+LSE pack kernel (live A3 & A5)",
         (("output:bf16:ND:dense::none", "lse:fp32:ND:dense::none"), ("output:fp16:ND:dense::none", "lse:fp32:ND:dense::none")),
         (("send:bf16:ND",), ("send:fp16:ND",))),
        ("fused_dcp_lse_combine", "_fused_dcp_lse_combine_kernel", "dcp_exchange",
         "vllm_ascend/ops/triton/dcp/dcp_a2a.py", ("_fused_dcp_lse_combine_kernel",), True,
         "Generic DCP LSE combine kernel (live A3 & A5)",
         (("recv:bf16:ND:dense::none", "local_output:bf16:ND:dense::none", "local_lse:fp32:ND:dense::none"),
          ("recv:fp16:ND:dense::none", "local_output:fp16:ND:dense::none", "local_lse:fp32:ND:dense::none")),
         (("output:bf16:ND",), ("output:fp16:ND",))),
        ("sfa_dcp_fast_exchange", "_pack_token_peer", "dcp_exchange",
         "vllm_ascend/ops/triton/dcp/sfa_dcp_exchange.py", ("_pack_token_peer",), True,
         "Fast pow2-head SFA DCP pack kernel (int32 word-viewed output_words & send per sfa_dcp_exchange.py:197)",
         (("output_words:int32:ND:none::none", "lse:fp32:ND:dense::none"),),
         (("send:int32:ND",),)),
        ("flash_attention_gate", "_flash_attention_gate_kernel", "attention",
         "vllm_ascend/ops/triton/flash_attention_output.py", ("_flash_attention_gate_kernel",), True,
         "MLA output gate Triton kernel (live A3 & A5)",
         (("projected:bf16:ND:dense::none", "gate:bf16:ND:dense::none"), ("projected:fp16:ND:dense::none", "gate:fp16:ND:dense::none")),
         (("out:bf16:ND",), ("out:fp16:ND",))),
        ("c2_compressor_pool", "_pool_kernel", "compressor",
         "vllm_ascend/ops/triton/compressor/compressor_triton.py", ("_pool_kernel",), True,
         "C2 Triton KV compressor pooling kernel (live A3 & A5 verified; compressor_triton.py:218-255)",
         (("kv:fp32:ND:dense::none", "score:fp32:ND:dense::none", "cache:fp32:ND:dense::none",
           "norm_w:bf16:ND:dense::none", "meta:int32:ND:none::none", "block_table:int32:ND:none::none"),
          ("kv:bf16:ND:dense::none", "score:bf16:ND:dense::none", "cache:fp32:ND:dense::none",
           "norm_w:bf16:ND:dense::none", "meta:int32:ND:none::none", "block_table:int32:ND:none::none")),
         (("out:bf16:ND",), ("out:bf16:ND",))),
        ("c2_compressor_cache_update", "_cache_update_kernel", "compressor",
         "vllm_ascend/ops/triton/compressor/compressor_triton.py", ("_cache_update_kernel",), True,
         "C2 Triton KV compressor ring-cache tail update kernel (live A3 & A5 verified)",
         (("kv:fp32:ND:dense::none", "score:fp32:ND:dense::none", "meta:int32:ND:none::none", "block_table:int32:ND:none::none"),),
         (("cache:fp32:ND",),)),
        ("fused_gdn_gating", "fused_gdn_gating_kernel", "linear_attention",
         "vllm_ascend/ops/triton/fused_gdn_gating.py", ("fused_gdn_gating_kernel",), True,
         "GDN delta gating Triton kernel (live A3 & A5)",
         (("a_log:fp32:ND:dense::none", "a:bf16:ND:dense::none", "b:bf16:ND:dense::none", "dt_bias:fp32:ND:dense::none"),),
         (("g:fp32:ND", "beta_output:bf16:ND"),)),
        ("l2norm_fwd", "l2norm_fwd_kernel2_loop", "linear_attention",
         "vllm_ascend/ops/triton/fla/l2norm.py", ("l2norm_fwd_kernel2",), True,
         "Recurrent Q/K L2Norm Triton kernel (live A3 & A5)",
         (("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",), ("x:fp32:ND:dense::none",)),
         (("y:bf16:ND",), ("y:fp16:ND",), ("y:fp32:ND",))),
        ("layer_norm_gated_fwd", "_layer_norm_fwd_1pass_kernel_npu", "linear_attention",
         "vllm_ascend/ops/triton/layernorm_gated.py", ("layer_norm_gated_fwd_kernel",), True,
         "Gated RMSNorm Triton kernel for KDA/GDN (live A3 & A5)",
         (("x:bf16:ND:dense::none", "w:bf16:ND:dense::none", "z:bf16:ND:dense::none"),
          ("x:fp16:ND:dense::none", "w:fp16:ND:dense::none", "z:fp16:ND:dense::none")),
         (("y:bf16:ND",), ("y:fp16:ND",))),
        ("triton_rope", "_triton_rope_siso", "attention",
         "vllm_ascend/ops/triton/rope.py", ("_triton_rope_siso",), True,
         "Shipping Triton RoPE kernels (live A3 & A5)",
         (("qk:bf16:ND:dense::none", "cos_sin:bf16:ND:dense::none", "pos:int64:ND:none::none"),
          ("qk:fp16:ND:dense::none", "cos_sin:fp16:ND:dense::none", "pos:int64:ND:none::none")),
         (("qk_out:bf16:ND",), ("qk_out:fp16:ND",))),
        ("mamba2_triton_ssu", "_gather_initial_states_kernel", "linear_attention",
         "vllm_ascend/ops/triton/mamba/state_ops.py", ("_gather_initial_states_kernel",), True,
         "Mamba2 Triton initial-state gather kernel (live A3 & A5)",
         (("state:bf16:ND:dense::none", "indices:int32:ND:none::none", "has_initial_state:bool:ND:none::none"),
          ("state:fp32:ND:dense::none", "indices:int32:ND:none::none", "has_initial_state:bool:ND:none::none")),
         (("output:bf16:ND",), ("output:fp32:ND",))),
        ("qsa_rmsnorm_mrope", "_qsa_rmsnorm_mrope_kernel", "attention",
         "vllm_ascend/ops/triton/linearnorm/qsa_rmsnorm_mrope.py", ("_qsa_rmsnorm_mrope_kernel",), True,
         "QSA fused RMSNorm + MRoPE Triton kernel (live A3 & A5)",
         (("input:bf16:ND:dense::none", "weight:bf16:ND:dense::none", "cos_sin:bf16:ND:dense::none"),
          ("input:fp16:ND:dense::none", "weight:fp16:ND:dense::none", "cos_sin:fp16:ND:dense::none")),
         (("output:bf16:ND",), ("output:fp16:ND",))),
        ("qsa_compress_groups", "_compress_qsa_groups_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
         ("_compress_qsa_groups_kernel",),
         True, "QSA group compression Triton kernel (live A3 & A5)",
         (("raw_keys:bf16:ND:dense::none", "raw_positions:int64:ND:none::none", "token_to_req:int32:ND:none::none", "query_start_loc:int32:ND:none::none"),),
         (("pooled:bf16:ND",),)),
        ("qsa_store_rows", "_store_qsa_rows_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
         ("_store_qsa_rows_kernel",),
         True, "QSA compressed row scatter Triton kernel (live A3 & A5)",
         (("pooled:bf16:ND:dense::none", "slots:int32:ND:none::none"),),
         (("cache:bf16:ND",),)),
        ("qsa_pack_group_indices", "_pack_qsa_group_indices_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/qwen4_exp/qsa.py",
         ("_pack_qsa_group_indices_kernel",),
         True, "QSA group index packing Triton kernel (live A3 & A5)",
         (("raw_positions:int64:ND:none::none", "query_start_loc:int32:ND:none::none"),),
         (("group_indices:int32:ND",),)),
        ("keypool_metadata_triton", "_build_keypool_metadata_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/glm5_next_cann_keypool.py",
         ("_build_keypool_metadata_kernel",),
         True, "KPool metadata prep Triton kernel (live A3 & A5; DCP==1)",
         (("raw_query:int32:ND:none::none", "raw_lengths:int32:ND:none::none", "raw_positions:int32:ND:none::none", "raw_slots:int32:ND:none::none"),),
         (("pages:int32:ND", "positions:int32:ND", "slots:int32:ND"),)),
        ("keypool_write_caches", "_write_keypool_caches_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/glm5_next_cann_keypool.py",
         ("_write_keypool_caches_kernel",),
         True, "KPool cache write Triton kernel (live A3 & A5; DCP==1)",
         (("keys:bf16:ND:dense::none", "slots:int32:ND:none::none"),),
         (("key_cache:bf16:ND",),)),
        ("keypool_prepare_nonempty", "_prepare_nonempty_keypool_inputs_kernel", "sparse_indexer",
         "vllm_ascend/ops/triton/prepare_nonempty_keypool_inputs.py",
         ("_prepare_nonempty_keypool_inputs_kernel",),
         True, "KPool nonempty-input compaction Triton kernel (live A3 & A5; DCP==1)",
         (("raw_lengths:int32:ND:none::none", "raw_positions:int32:ND:none::none"),),
         (("nonempty_indices:int32:ND",),)),
        ("ple_conv_triton", "_ple_conv_kernel", "residual_mixer",
         "vllm_ascend/ops/triton_ple.py", ("_ple_conv_kernel",),
         True, "Per-Layer Embedding causal conv Triton kernel (live A3 & A5)",
         (("x:bf16:ND:dense::none", "state:bf16:ND:dense::none", "w:bf16:ND:dense::none", "qsl:int32:ND:none::none"),),
         (("residual:bf16:ND",),)),
        ("ple_conv_writeback", "_ple_conv_writeback_kernel", "residual_mixer",
         "vllm_ascend/ops/triton_ple.py", ("_ple_conv_writeback_kernel",),
         True, "Per-Layer Embedding conv state writeback Triton kernel (live A3 & A5)",
         (("x:bf16:ND:dense::none", "state:bf16:ND:dense::none", "state_indices:int32:ND:none::none", "qsl:int32:ND:none::none"),),
         (("state_out:bf16:ND",),)),
        ("split_qkv_rmsnorm_mrope", "split_qkv_rmsnorm_mrope_kernel", "attention",
         "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py", ("split_qkv_rmsnorm_mrope_kernel",), True,
         "Split QKV + RMSNorm + MRoPE Triton kernel (live A3)",
         (("in_qkv:bf16:ND:dense::none", "q_weight:bf16:ND:dense::none", "k_weight:bf16:ND:dense::none", "cos_sin:bf16:ND:dense::none"),),
         (("out_q:bf16:ND", "out_k:bf16:ND", "out_v:bf16:ND"),)),
        ("dcp_a2a_batched", "_pack_dcp_output_lse_batched_kernel", "dcp_exchange",
         "vllm_ascend/ops/triton/dcp/dcp_a2a_batched.py",
         ("_pack_dcp_output_lse_batched_kernel",), True,
         "Batched DCP output+LSE pack Triton kernel (live A3)",
         (("output:bf16:ND:dense::none", "lse:fp32:ND:dense::none"), ("output:fp16:ND:dense::none", "lse:fp32:ND:dense::none")),
         (("send:bf16:ND",), ("send:fp16:ND",))),
        ("engram_int8_lookup", "_engram_int8_gather_dequant_kernel", "residual_mixer",
         "vllm_ascend/ops/triton/engram_lookup.py", ("_engram_int8_gather_dequant_kernel",), True,
         "Engram INT8 gather+dequant Triton kernel (live A3)",
         (("weight:int8:ND:per_channel_int8::fp32", "scale:fp32:ND:dense::none", "ids:int64:ND:none::none"),),
         (("output:bf16:ND",),)),
        ("pcp_kv_cache_copy", "_copy_pcp_kv_cache_kernel", "dcp_exchange",
         "vllm_ascend/ops/triton/pcp_kv_cache.py", ("_copy_pcp_kv_cache_kernel",), True,
         "PCP KV cache copy Triton kernel (live A5)",
         (("key_cache:bf16:ND:dense::none", "rope_cache:bf16:ND:dense::none", "slots:int32:ND:none::none"),),
         (("packed:bf16:ND",),)),
    ]
    entries: list[ProbedAbiEntry] = []
    for suffix, sym, role, rel, kernels, avail, note, in_tuples, out_tuples in triton_specs:
        entries.append(_mk(
            repos, entry_id=f"{p_low}_triton_{suffix}", abi_id=sym,
            available=avail, backend="triton", symbol=sym, arch=arch,
            role=role, graph_motif=suffix,
            dtype_tuple=in_tuples,
            outputs=out_tuples,
            repo_key="vllm-ascend", rel_path=rel, va_ref=f"{rel}:1",
            kernels=kernels if avail else (), note=note, observed_kernels=observed,
        ))
    return entries


def _cann_live_extra_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    return [
        _mk(
            repos, entry_id=f"{p_low}_cann_swiglu", abi_id="aclnnSwiGlu",
            available=True, backend="cann_aclnn", symbol="aclnnSwiGlu", arch=arch,
            role="moe_act_quant", graph_motif="swiglu",
            dtype_tuple=(("x:bf16:ND:dense::none",), ("x:fp16:ND:dense::none",)),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            activations=frozenset({"swiglu"}),
            repo_key="ops-nn", rel_path="docs/en/op_api_list.md",
            cann_ref="ops-nn/docs/en/op_api_list.md:aclnnSwiGlu",
            kernels=("SwiGlu",),
            note="Shipping CANN SwiGlu across A2/A3/A5 (live A3 & A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_dynamic_block_quant_v2", abi_id="aclnnDynamicBlockQuantV2",
            available=True, backend="cann_aclnn", symbol="aclnnDynamicBlockQuantV2", arch=arch,
            role="norm_quant", graph_motif="dynamic_block_quant",
            dtype_tuple=(
                (
                    ("x:bf16:ND:dense::none",),
                    ("x:fp16:ND:dense::none",),
                    ("x:bf16:ND:dense::none",),
                    ("x:fp16:ND:dense::none",),
                ) if is_a5 else (
                    ("x:bf16:ND:dense::none",),
                    ("x:fp16:ND:dense::none",),
                )
            ),
            outputs=(
                (
                    ("y:int8:ND", "scale:fp32:ND"),
                    ("y:int8:ND", "scale:fp32:ND"),
                    ("y:fp8_e4m3fn:ND", "scale:fp32:ND"),
                    ("y:fp8_e4m3fn:ND", "scale:fp32:ND"),
                ) if is_a5 else (
                    ("y:int8:ND", "scale:fp32:ND"),
                    ("y:int8:ND", "scale:fp32:ND"),
                )
            ),
            repo_key="ops-nn", rel_path="quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.cpp",
            op_ref="op_plugin/ops/opapi/DynamicBlockQuantNpuOpApi.cpp:1",
            cann_ref="ops-nn/quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.cpp:1",
            kernels=("DynamicBlockQuant",),
            note="Shipping CANN block-wise dynamic quant (live A3 & A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_gemma_rms_norm", abi_id="aclnnGemmaRmsNorm",
            available=True, backend="cann_aclnn", symbol="aclnnGemmaRmsNorm", arch=arch,
            role="norm_quant", graph_motif="gemma_rms_norm",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
                ("x:fp32:ND:dense::none", "gamma:fp32:ND:dense::none"),
            ),
            outputs=(
                ("y:bf16:ND", "rstd:fp32:ND"),
                ("y:fp16:ND", "rstd:fp32:ND"),
                ("y:fp32:ND", "rstd:fp32:ND"),
            ),
            repo_key="ops-nn", rel_path="norm/gemma_rms_norm/op_host/gemma_rms_norm_def.cpp",
            cann_ref="ops-nn/norm/gemma_rms_norm/op_host/gemma_rms_norm_def.cpp:1",
            kernels=("GemmaRmsNorm",),
            note="Shipping CANN unit-offset GemmaRmsNorm (2 required inputs -> y, rstd; live A3 & A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_kv_rmsnorm_rope_cache_v2", abi_id="aclnnKvRmsNormRopeCacheV2",
            available=True, backend="cann_aclnn", symbol="aclnnKvRmsNormRopeCacheV2", arch=arch,
            role="attention", graph_motif="kv_rmsnorm_rope_cache",
            dtype_tuple=(
                (
                    "kv:bf16:ND:dense::none", "gamma:bf16:ND:dense::none",
                    "cos:bf16:ND:dense::none", "sin:bf16:ND:dense::none",
                    "index:int64:ND:none::none", "k_cache:bf16:ND:dense::none", "ckv_cache:bf16:ND:dense::none",
                ),
                (
                    "kv:fp16:ND:dense::none", "gamma:fp16:ND:dense::none",
                    "cos:fp16:ND:dense::none", "sin:fp16:ND:dense::none",
                    "index:int64:ND:none::none", "k_cache:fp16:ND:dense::none", "ckv_cache:fp16:ND:dense::none",
                ),
            ),
            outputs=(
                ("k_cache:bf16:ND", "ckv_cache:bf16:ND", "k_rope:bf16:ND", "c_kv:bf16:ND"),
                ("k_cache:fp16:ND", "ckv_cache:fp16:ND", "k_rope:fp16:ND", "c_kv:fp16:ND"),
            ),
            repo_key="ops-transformer", rel_path="posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_def.cpp",
            op_ref="op_plugin/ops/opapi/KvRmsNormRopeCacheNpuOpApi.cpp:1",
            cann_ref="ops-transformer/posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_def.cpp:1",
            kernels=("KvRmsNormRopeCache",),
            note="Fused MLA KV RMSNorm + RoPE + paged cache write (7 required inputs -> 4 required outputs)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_inner_lightning_indexer_v2", abi_id="aclnnInnerLightningIndexerV2",
            available=True, backend="cann_aclnn", symbol="aclnnInnerLightningIndexerV2", arch=arch,
            role="sparse_indexer", graph_motif="lightning_indexer_v2",
            dtype_tuple=(
                ("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "w:bf16:ND:dense::none"),
                ("q:fp16:ND:dense::none", "k:fp16:ND:dense::none", "w:fp16:ND:dense::none"),
            ),
            outputs=(
                ("sparse_indices:int32:ND", "sparse_values:fp32:ND"),
                ("sparse_indices:int32:ND", "sparse_values:fp32:ND"),
            ),
            repo_key="ops-transformer", rel_path="experimental/attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp",
            cann_ref="ops-transformer/experimental/attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp:1",
            kernels=("LightningIndexerV2",),
            note="Dense BF16/FP16 LightningIndexerV2 (3 required inputs -> 2 outputs; live A3 & A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_causal_conv1d", abi_id="aclnnCausalConv1d",
            available=True, backend="cann_aclnn", symbol="aclnnCausalConv1d", arch=arch,
            role="linear_attention", graph_motif="causal_conv1d",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "weight:bf16:ND:dense::none", "conv_states:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "weight:fp16:ND:dense::none", "conv_states:fp16:ND:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            repo_key="op-plugin", rel_path="op_plugin/ops/opapi/FusedCausalConv1dNpuOpApi.cpp",
            op_ref="op_plugin/ops/opapi/FusedCausalConv1dNpuOpApi.cpp:1",
            cann_ref="op_plugin/ops/opapi/FusedCausalConv1dNpuOpApi.cpp:aclnnCausalConv1d",
            kernels=("CausalConv1d",),
            note="Shipping CANN CausalConv1d for GDN/KDA/Mamba (3 required inputs -> y; live A5 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_moe_token_permute", abi_id="aclnnMoeTokenPermute",
            available=True, backend="cann_aclnn", symbol="aclnnMoeTokenPermute", arch=arch,
            role="moe_routing", graph_motif="moe_token_permute",
            dtype_tuple=(("x:bf16:ND:dense::none", "idx:int32:ND:none::none"), ("x:fp16:ND:dense::none", "idx:int32:ND:none::none")),
            outputs=(
                ("permute_tokens:bf16:ND", "sorted_indices:int32:ND"),
                ("permute_tokens:fp16:ND", "sorted_indices:int32:ND"),
            ),
            repo_key="op-plugin", rel_path="op_plugin/ops/opapi/MoeTokenPermuteKernelNpuOpApi.cpp",
            op_ref="op_plugin/ops/opapi/MoeTokenPermuteKernelNpuOpApi.cpp:1",
            cann_ref="ops-transformer/moe/moe_token_permute/op_host/moe_token_permute_def.cpp:37",
            kernels=("MoeTokenPermute",),
            note="MoE token permute (2 required outputs: permute_tokens, sorted_indices; live A3 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_moe_token_unpermute", abi_id="aclnnMoeTokenUnpermute",
            available=True, backend="cann_aclnn", symbol="aclnnMoeTokenUnpermute", arch=arch,
            role="moe_routing", graph_motif="moe_token_unpermute",
            dtype_tuple=(("x:bf16:ND:dense::none", "idx:int32:ND:none::none"), ("x:fp16:ND:dense::none", "idx:int32:ND:none::none")),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            repo_key="op-plugin", rel_path="op_plugin/ops/opapi/MoeTokenUnpermuteKernelNpuOpApi.cpp",
            op_ref="op_plugin/ops/opapi/MoeTokenUnpermuteKernelNpuOpApi.cpp:1",
            cann_ref="op_plugin/ops/opapi/MoeTokenUnpermuteKernelNpuOpApi.cpp:1",
            kernels=("MoeTokenUnpermute",),
            note="MoE token unpermute (live A3 verified)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_chunk_gdn_arch_strategy",
            abi_id="aclnnChunkGatedDeltaRuleArch35" if is_a5 else "aclnnChunkGatedDeltaRuleArch22",
            available=True, backend="cann_aclnn",
            symbol="aclnnChunkGatedDeltaRuleArch35" if is_a5 else "aclnnChunkGatedDeltaRuleArch22",
            arch=arch, role="linear_attention", graph_motif="chunk_gdn",
            dtype_tuple=((
                "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                "beta:bf16:ND:dense::none", "initial_state:fp32:ND:dense::none",
                "actual_seq_lengths:int32:ND:none::none",
            ),),
            outputs=(("out:bf16:ND", "final_state:fp32:ND"),),
            repo_key="ops-transformer",
            rel_path="experimental/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp",
            op_ref="op_plugin/ops/opapi/ChunkGatedDeltaRuleKernelNpuOpApi.cpp:1",
            cann_ref="ops-transformer/experimental/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp:22",
            kernels=("ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO"),
            note="Arch-specialized chunk GDN strategy ABI binding (BF16-only, int32 actual_seq_lengths; 6 required inputs -> out, final_state)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_chunk_gdn_fwd",
            abi_id="aclnnChunkGatedDeltaRuleFwd",
            available=True, backend="cann_aclnn",
            symbol="aclnnChunkGatedDeltaRuleFwd",
            arch=arch, role="linear_attention", graph_motif="chunk_gdn_fwd",
            dtype_tuple=((
                "query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none",
                "beta:bf16:ND:dense::none", "initial_state:fp32:ND:dense::none",
                "actual_seq_lengths:int32:ND:none::none",
            ),),
            outputs=(("out:bf16:ND", "final_state:fp32:ND"),),
            repo_key="ops-transformer",
            rel_path="experimental/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp",
            op_ref="op_plugin/ops/opapi/ChunkGatedDeltaRuleKernelNpuOpApi.cpp:1",
            cann_ref="ops-transformer/experimental/attention/chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp:22",
            kernels=("ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO"),
            note="Ordered 3-kernel experimental chunk GDN forward expansion (BF16-only, int32 actual_seq_lengths; live A3 & A5)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_clamp",
            abi_id="aclnnClamp",
            available=True, backend="cann_aclnn",
            symbol="aclnnClamp",
            arch=arch, role="moe_act_quant", graph_motif="clamp",
            dtype_tuple=(("self:bf16:ND:dense::none",), ("self:fp16:ND:dense::none",), ("self:fp32:ND:dense::none",)),
            outputs=(("out:bf16:ND",), ("out:fp16:ND",), ("out:fp32:ND",)),
            repo_key="ops-math",
            rel_path="experimental/conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp",
            op_ref="op_plugin/ops/aclops/ClampKernelNpu.cpp:31",
            cann_ref="ops-math/experimental/conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp:18",
            kernels=("Slice", "ClipByValueV2", "ViewCopy"),
            note="Ordered 3-kernel non-contiguous slice clamp expansion (live A5 clipped_swiglu)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_transpose_batchmatmul",
            abi_id="aclnnTransposeBatchMatMul",
            available=True, backend="cann_aclnn",
            symbol="aclnnTransposeBatchMatMul",
            arch=arch, role="dense_matmul", graph_motif="transpose_batchmatmul",
            dtype_tuple=(
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            allow_bias=is_a5,
            require_no_bias=not is_a5,
            repo_key="op-plugin",
            rel_path="op_plugin/ops/opapi/TransposeBatchMatMulKernelNpuOpApi.cpp",
            op_ref="op_plugin/ops/opapi/TransposeBatchMatMulKernelNpuOpApi.cpp:85",
            cann_ref="ops-nn/matmul/transpose_batch_mat_mul/op_api/aclnn_transpose_batch_mat_mul.cpp:196",
            kernels=("Slice", "TransposeBatchMatMul"),
            note="Ordered 2-kernel TransposeBatchMatMul expansion (bias & perm_y=[0,1,2] A5-only per aclnn_transpose_batch_mat_mul.cpp:215,364)",
            observed_kernels=observed,
        ),
    ]


def _torch_npu_catlass_and_wheel_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    entries: list[ProbedAbiEntry] = []

    # torch_npu native PyTorch NPU primitives (op-plugin) with exact per-API upstream signatures.
    torch_npu_specs: list[tuple[str, str, str, str, tuple[str, ...], tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]] = [
        (
            "npu_rms_norm", "torch_npu.npu_rms_norm", "norm_quant",
            "op_plugin/ops/opapi/RmsNormKernelOpApi.cpp", ("RmsNorm",),
            (
                ("self:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("self:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
            ),
            (
                ("y:bf16:ND", "rstd:fp32:ND"),
                ("y:fp16:ND", "rstd:fp32:ND"),
            ),
        ),
        (
            "npu_add_rms_norm", "torch_npu.npu_add_rms_norm", "norm_quant",
            "op_plugin/ops/opapi/AddRmsNormV2KernelNpuOpApi.cpp", ("AddRmsNorm",),
            (
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none", "gamma:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none", "gamma:fp16:ND:dense::none"),
            ),
            (
                ("y:bf16:ND", "rstd:fp32:ND", "x_out:bf16:ND"),
                ("y:fp16:ND", "rstd:fp32:ND", "x_out:fp16:ND"),
            ),
        ),
        (
            "npu_rotary_mul", "torch_npu.npu_rotary_mul", "attention",
            "op_plugin/ops/opapi/RotaryMulKernelNpuOpApi.cpp", ("RotaryPositionEmbedding",),
            (
                ("x:bf16:ND:dense::none", "r1:bf16:ND:dense::none", "r2:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "r1:fp16:ND:dense::none", "r2:fp16:ND:dense::none"),
            ),
            (
                ("y:bf16:ND",),
                ("y:fp16:ND",),
            ),
        ),
        (
            "npu_dynamic_quant", "torch_npu.npu_dynamic_quant", "norm_quant",
            "op_plugin/ops/opapi/DynamicQuantKernelNpuOpApi.cpp", ("DynamicQuantV2",),
            (
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
            ),
            (
                ("y:int8:ND", "scale:fp32:ND"),
                ("y:int8:ND", "scale:fp32:ND"),
            ),
        ),
        (
            "npu_transpose_batchmatmul", "torch_npu.npu_transpose_batchmatmul", "dense_matmul",
            "op_plugin/ops/opapi/TransposeBatchMatMulKernelNpuOpApi.cpp", ("TransposeBatchMatMul",),
            (
                ("x1:bf16:ND:dense::none", "x2:bf16:ND:dense::none"),
                ("x1:fp16:ND:dense::none", "x2:fp16:ND:dense::none"),
            ),
            (
                ("y:bf16:ND",),
                ("y:fp16:ND",),
            ),
        ),
        (
            "npu_moe_gating_top_k", "torch_npu.npu_moe_gating_top_k", "moe_routing",
            "op_plugin/ops/opapi/MoeGatingTopKKernelNpuOpApi.cpp", ("MoeGatingTopK",),
            # MoeGatingTopKKernelNpuOpApi.cpp:42-55: x is 2D (fp16/fp32/bf16); bias is OPTIONAL 1D (expert_num,).
            (
                ("x:bf16:ND:dense::none",),
                ("x:fp16:ND:dense::none",),
                ("x:fp32:ND:dense::none",),
                ("x:bf16:ND:dense::none", "bias:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "bias:fp16:ND:dense::none"),
                ("x:fp32:ND:dense::none", "bias:fp32:ND:dense::none"),
            ),
            (
                ("y:bf16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp32:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:bf16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp16:ND", "expert_idx:int32:ND", "out:fp32:ND"),
                ("y:fp32:ND", "expert_idx:int32:ND", "out:fp32:ND"),
            ),
        ),
        (
            "npu_moe_init_routing_v2", "torch_npu.npu_moe_init_routing_v2", "moe_routing",
            "op_plugin/ops/opapi/MoeInitRoutingV2KernelNpuOpApi.cpp", ("MoeInitRoutingV3",),
            (
                ("x:bf16:ND:dense::none", "expert_idx:int32:ND:none::none"),
                ("x:fp16:ND:dense::none", "expert_idx:int32:ND:none::none"),
            ),
            (
                ("expanded_x:bf16:ND", "expanded_row_idx:int32:ND", "expert_tokens_count:int64:ND", "expanded_scale:fp32:ND"),
                ("expanded_x:fp16:ND", "expanded_row_idx:int32:ND", "expert_tokens_count:int64:ND", "expanded_scale:fp32:ND"),
            ),
        ),
        (
            "npu_moe_finalize_routing", "torch_npu.npu_moe_finalize_routing", "moe_routing",
            "op_plugin/ops/opapi/MoeFinalizeRoutingV2KernelNpuOpApi.cpp", ("MoeFinalizeRoutingV2",),
            (
                (
                    "expanded_permuted_rows:bf16:ND:dense::none",
                    "expanded_src_to_dst_row:int32:ND:none::none",
                ),
                (
                    "expanded_permuted_rows:fp16:ND:dense::none",
                    "expanded_src_to_dst_row:int32:ND:none::none",
                ),
                (
                    "expanded_permuted_rows:bf16:ND:dense::none",
                    "skip1:bf16:ND:dense::none",
                    "scales:fp32:ND:dense::none",
                    "expanded_src_to_dst_row:int32:ND:none::none",
                    "expert_for_source_row:int32:ND:none::none",
                ),
                (
                    "expanded_permuted_rows:fp16:ND:dense::none",
                    "skip1:fp16:ND:dense::none",
                    "scales:fp32:ND:dense::none",
                    "expanded_src_to_dst_row:int32:ND:none::none",
                    "expert_for_source_row:int32:ND:none::none",
                ),
            ),
            (
                ("y:bf16:ND",),
                ("y:fp16:ND",),
                ("y:bf16:ND",),
                ("y:fp16:ND",),
            ),
        ),
        (
            "npu_grouped_matmul", "torch_npu.npu_grouped_matmul", "moe_grouped_matmul",
            "op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp", ("GroupedMatmul",),
            (
                ("x:bf16:ND:dense::none", "w:bf16:FRACTAL_NZ:dense::none"),
                ("x:fp16:ND:dense::none", "w:fp16:FRACTAL_NZ:dense::none"),
                ("x:bf16:ND:dense::none", "w:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "w:fp16:ND:dense::none"),
                ("x:int8:ND:per_token_int8::fp32", "w:int8:FRACTAL_NZ:per_channel_int8::fp32"),
                ("x:int8:ND:per_token_int8::fp32", "w:int4:FRACTAL_NZ:w4a8_int::fp32"),
                *(
                    (
                        ("x:fp8_e4m3fn:ND:mx_fp8:32:fp8_e8m0", "w:fp8_e4m3fn:FRACTAL_NZ:mx_fp8:32:fp8_e8m0"),
                        ("x:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "w:fp4_e2m1:FRACTAL_NZ:mx_fp4:32:fp8_e8m0"),
                    ) if is_a5 else ()
                ),
            ),
            (
                ("y:bf16:ND",), ("y:fp16:ND",), ("y:bf16:ND",), ("y:fp16:ND",),
                ("y:bf16:ND",), ("y:bf16:ND",),
                *((("y:bf16:ND",), ("y:bf16:ND",)) if is_a5 else ()),
            ),
        ),
    ]
    for suffix, sym, role, rel, kernels, in_tuples, out_tuples in torch_npu_specs:
        spec_shape = (
            AbiShapeConstraints(max_experts=MAX_EXPERTS_MOE_GATING)
            if suffix == "npu_moe_gating_top_k"
            else (AbiShapeConstraints(max_experts=MAX_EXPERTS_GROUPED_MATMUL) if suffix == "npu_grouped_matmul" else None)
        )
        entries.append(_mk(
            repos, entry_id=f"{p_low}_torch_npu_{suffix}", abi_id=sym,
            available=True, backend="torch_npu", symbol=sym, arch=arch,
            role=role, graph_motif=suffix,
            dtype_tuple=in_tuples,
            outputs=out_tuples,
            shape=spec_shape,
            allow_bias=is_a5 if suffix == "npu_transpose_batchmatmul" else True,
            require_no_bias=(not is_a5) if suffix == "npu_transpose_batchmatmul" else False,
            repo_key="op-plugin", rel_path=rel, op_ref=f"{rel}:1",
            kernels=kernels, note="PyTorch torch_npu op-plugin dispatch (exact upstream signature)",
            observed_kernels=observed,
        ))

    # CATLASS hardware template library entries.
    entries.append(_mk(
        repos, entry_id=f"{p_low}_catlass_grouped_gemm_bf16",
        abi_id="catlass_grouped_gemm_bf16", available=True, backend="catlass",
        symbol="catlass::gemm::BlockMmad", arch=arch,
        role="catlass_template", graph_motif="catlass_gemm",
        dtype_tuple=(("x1:bf16:ND:dense::none", "x2:bf16:FRACTAL_NZ:dense::none"),),
        outputs=(("y:bf16:ND",),),
        repo_key="catlass", rel_path="include/catlass/arch/arch.hpp",
        kernels=("CatlassGroupedGemm",),
        note="CATLASS Ascend template (A2/A3 UB=192KiB, A5 UB=248KiB)",
        observed_kernels=observed,
    ))
    entries.append(_mk(
        repos, entry_id=f"{p_low}_catlass_dual_level_mx_epilogue",
        abi_id="catlass_epilogue_dual_level_quant_mx", available=is_a5, backend="catlass",
        symbol="Catlass::Epilogue::Block::BlockQuantDualLevelMx", arch=arch,
        role="catlass_epilogue", graph_motif="catlass_mx_epilogue",
        dtype_tuple=(
            ("x:bf16:ND:dense::none",),
            ("x:fp16:ND:dense::none",),
        ),
        outputs=(
            ("y:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "scale1:fp32:ND", "scale2:fp8_e8m0:ND"),
            ("y:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "scale1:fp32:ND", "scale2:fp8_e8m0:ND"),
        ),
        repo_key="catlass",
        rel_path="include/catlass/epilogue/block/block_epilogue_dual_level_quant_mx.hpp",
        kernels=("CatlassDualLevelQuantMx",) if is_a5 else (),
        note="A5 BlockQuantDualLevelMx dual-level MXFP4 epilogue (1 BF16/FP16 tile -> fp4_e2m1 + fp32 scale1 + fp8_e8m0 scale2)",
        observed_kernels=observed,
    ))

    # Third-party wheel operators (fla_npu_stable & optional packaged A5 DSL wheel).
    entries.append(_mk(
        repos, entry_id=f"{p_low}_wheel_fla_npu_chunk_kda",
        abi_id="fla_npu_stable.npu_chunk_kda_fwd", available=True,
        backend="third_party_wheel", symbol="fla_npu_stable::npu_chunk_kda_fwd", arch=arch,
        role="third_party_kernel", graph_motif="chunk_kda",
        dtype_tuple=(("q:bf16:ND:dense::none", "k:bf16:ND:dense::none", "v:bf16:ND:dense::none"),),
        outputs=(("out:bf16:ND",),),
        repo_key="vllm-ascend", rel_path="vllm_ascend/ops/kda.py",
        va_ref="vllm_ascend/ops/kda.py:60",
        kernels=("ChunkKdaFwdPrepare", "ChunkFwdH", "ChunkKdaFwdFinalize"),
        note="Third-party fla_npu_stable wheel verified live on A3 and A5",
        observed_kernels=observed,
    ))
    entries.append(_mk(
        repos, entry_id=f"{p_low}_wheel_packaged_qli_dsl",
        abi_id="ops.quant_lightning_indexer_dsl", available=False,
        backend="third_party_wheel", symbol="ops.quant_lightning_indexer_dsl", arch=arch,
        role="third_party_kernel", graph_motif="packaged_qli_dsl",
        dtype_tuple=(("q:fp8_e4m3fn:ND:per_token_fp8::fp32", "k:fp8_e4m3fn:ND:per_token_fp8::fp32"),),
        outputs=(("topk_indices:int32:ND",),),
        repo_key="vllm-ascend", rel_path="vllm_ascend/ops/packaged_attention.py",
        va_ref="vllm_ascend/ops/packaged_attention.py:26",
        kernels=(),
        note="External Python wheel not installed in baseline container image (missing Python package)",
        observed_kernels=observed,
    ))
    return entries


def _mqsmla_io_rows(
    ori_kv_spec: str,
    cmp_kv_spec: str,
    *,
    include_nd_contiguous: bool = False,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Build exact positional C++ active-tensor rows for ``aclnnMixedQuantSparseFlashMla`` / DSL."""
    q = "q:bf16:ND:dense::none"
    ori_idx = "ori_sparse_indices:int32:ND:none::none"
    cmp_idx = "cmp_sparse_indices:int32:ND:none::none"
    ori_bt = "ori_block_table:int32:ND:none::none"
    cmp_bt = "cmp_block_table:int32:ND:none::none"
    cu_q = "cu_seqlens_q:int32:ND:none::none"
    cu_ori = "cu_seqlens_ori_kv:int32:ND:none::none"
    cu_cmp = "cu_seqlens_cmp_kv:int32:ND:none::none"
    seq_ori = "seqused_ori_kv:int32:ND:none::none"
    seq_cmp = "seqused_cmp_kv:int32:ND:none::none"
    sinks = "sinks:fp32:ND:dense::none"
    meta = "metadata:int32:ND:none::none"
    base_in: list[tuple[str, ...]] = [
        # CSA (window + sparse compressed) PA_BBND: BSND & TND
        (q, ori_kv_spec, cmp_kv_spec, cmp_idx, ori_bt, cmp_bt, seq_ori, seq_cmp, sinks, meta),
        (q, ori_kv_spec, cmp_kv_spec, cmp_idx, ori_bt, cmp_bt, cu_q, seq_ori, seq_cmp, sinks, meta),
        # SWA (window-only, cmp_kv omitted) PA_BBND: BSND & TND
        (q, ori_kv_spec, ori_bt, seq_ori, sinks, meta),
        (q, ori_kv_spec, ori_bt, cu_q, seq_ori, sinks, meta),
        # HCA (window + dense compressed) PA_BBND: BSND & TND
        (q, ori_kv_spec, cmp_kv_spec, ori_bt, cmp_bt, seq_ori, seq_cmp, sinks, meta),
        (q, ori_kv_spec, cmp_kv_spec, ori_bt, cmp_bt, cu_q, seq_ori, seq_cmp, sinks, meta),
        # ORI_CMP_SPARSE (both sparse index tensors) PA_BBND: BSND & TND
        (q, ori_kv_spec, cmp_kv_spec, ori_idx, cmp_idx, ori_bt, cmp_bt, seq_ori, seq_cmp, sinks, meta),
        (q, ori_kv_spec, cmp_kv_spec, ori_idx, cmp_idx, ori_bt, cmp_bt, cu_q, seq_ori, seq_cmp, sinks, meta),
    ]
    if include_nd_contiguous:
        ori_nd = ori_kv_spec.replace(":PA_BBND:", ":ND:")
        cmp_nd = cmp_kv_spec.replace(":PA_BBND:", ":ND:")
        base_in.extend([
            (q, ori_nd, cmp_nd, cmp_idx, cu_q, cu_ori, cu_cmp, sinks, meta),
            (q, ori_nd, cu_q, cu_ori, sinks, meta),
        ])
    in_rows: list[tuple[str, ...]] = []
    out_rows: list[tuple[str, ...]] = []
    for row in base_in:
        in_rows.append(row)
        out_rows.append(("attn_out:bf16:ND",))
        in_rows.append(row)
        out_rows.append(("attn_out:bf16:ND", "softmax_lse:fp32:ND"))
    return tuple(in_rows), tuple(out_rows)


def _mqsmla_a23_io_rows() -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Build exact arch22 INT8 CSA (mode 1) and TurboQuant (mode 3) positional C++ rows."""
    cmp_idx = "cmp_sparse_indices:int32:ND:none::none"
    ori_bt = "ori_block_table:int32:ND:none::none"
    cmp_bt = "cmp_block_table:int32:ND:none::none"
    cu_q = "cu_seqlens_q:int32:ND:none::none"
    cu_ori = "cu_seqlens_ori_kv:int32:ND:none::none"
    cu_cmp = "cu_seqlens_cmp_kv:int32:ND:none::none"
    seq_ori = "seqused_ori_kv:int32:ND:none::none"
    seq_cmp = "seqused_cmp_kv:int32:ND:none::none"
    sinks = "sinks:fp32:ND:dense::none"
    meta = "metadata:int32:ND:none::none"
    in_rows: list[tuple[str, ...]] = []
    out_rows: list[tuple[str, ...]] = []
    for q_dt in ("bf16", "fp16"):
        q = f"q:{q_dt}:ND:dense::none"
        ori_i8_pa = "ori_kv:int8:PA_BBND:int8_csa_608:64:bf16"
        cmp_i8_pa = "cmp_kv:int8:PA_BBND:int8_csa_608:64:bf16"
        ori_i8_nd = "ori_kv:int8:ND:int8_csa_608:64:bf16"
        cmp_i8_nd = "cmp_kv:int8:ND:int8_csa_608:64:bf16"
        ori_tq = f"ori_kv:{q_dt}:PA_BBND:tq4_ori::none"
        cmp_tq = "cmp_kv:uint8:PA_BBND:tq4_cmp::none"
        candidates = (
            # INT8 CSA (quant_mode=1): requires ori_kv, cmp_kv, cmp_sparse_indices (no ori_sparse_indices)
            (q, ori_i8_pa, cmp_i8_pa, cmp_idx, ori_bt, cmp_bt, seq_ori, seq_cmp, sinks, meta),
            (q, ori_i8_pa, cmp_i8_pa, cmp_idx, ori_bt, cmp_bt, cu_q, seq_ori, seq_cmp, sinks, meta),
            (q, ori_i8_nd, cmp_i8_nd, cmp_idx, cu_q, cu_ori, cu_cmp, sinks, meta),
            # TurboQuant (quant_mode=3): requires TND/PA_BBND
            (q, ori_tq, cmp_tq, cmp_idx, ori_bt, cmp_bt, cu_q, seq_ori, seq_cmp, sinks, meta),
        )
        for row in candidates:
            in_rows.append(row)
            out_rows.append((f"attn_out:{q_dt}:ND",))
            in_rows.append(row)
            out_rows.append((f"attn_out:{q_dt}:ND", "softmax_lse:fp32:ND"))
    return tuple(in_rows), tuple(out_rows)


def _mqsmla_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    if is_a5:
        m1_in, m1_out = _mqsmla_io_rows(
            "ori_kv:fp8_e4m3fn:PA_BBND:packed_kv_608:64:bf16",
            "cmp_kv:fp8_e4m3fn:PA_BBND:packed_kv_608:64:bf16",
            include_nd_contiguous=True,
        )
        m2_in, m2_out = _mqsmla_io_rows(
            "ori_kv:fp8_e4m3fn:PA_BBND:packed_kv_584:64:fp8_e8m0",
            "cmp_kv:fp8_e4m3fn:PA_BBND:packed_kv_584:64:fp8_e8m0",
            include_nd_contiguous=False,
        )
        cann_in, cann_out = m1_in + m2_in, m1_out + m2_out
    else:
        cann_in, cann_out = _mqsmla_a23_io_rows()
    dsl_in, dsl_out = _mqsmla_io_rows(
        "ori_kv:fp8_e4m3fn:PA_BBND:per_group_fp8:32:bf16",
        "cmp_kv:fp4_e2m1:PA_BBND:per_group_fp4:16:bf16",
        include_nd_contiguous=False,
    )
    mqs_shape = AbiShapeConstraints(
        head_dim_allowed_set=MQSMLA_HEAD_DIMS,
        rope_dim_allowed_set=MQSMLA_ROPE_DIMS,
        kv_heads_allowed_set=MQSMLA_KV_HEADS,
    )
    return [
        _mk(
            repos, entry_id=f"{p_low}_cann_mixed_quant_sparse_flash_mla",
            abi_id="aclnnMixedQuantSparseFlashMla", available=True,
            backend="cann_aclnn", symbol="aclnnMixedQuantSparseFlashMla", arch=arch,
            role="attention", graph_motif="mixed_quant_sparse_flash_mla",
            dtype_tuple=cann_in,
            outputs=cann_out,
            shape=mqs_shape,
            repo_key="ops-transformer",
            rel_path="attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp",
            cann_ref="ops-transformer/attention/mixed_quant_sparse_flash_mla/op_host/checkers/mixed_quant_variant_checker.cpp:37",
            kernels=("MixedQuantSparseFlashMla",),
            note=(
                "A5 DAV_3510 FP8 MQSMLA (Mode-1 608B PA_BBND/ND & Mode-2 584B PA_BBND-only; "
                "full C++ positional active ports incl. block_table, seqused, sinks, metadata; D=512, RoPE=64, kvN=1)"
                if is_a5 else
                "A2/A3 DAV_2201 MQSMLA (INT8 CSA mode 1 [608B] & TurboQuant mode 3 [512/258B]; FP8 Mode 1/2 is A5-only)"
            ),
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_dsl_mixed_quant_sparse_flash_mla",
            abi_id="tilelang_dsl.mixed_quant_sparse_flash_mla", available=is_a5,
            backend="third_party_wheel", symbol="tilelang_dsl.mixed_quant_sparse_flash_mla", arch=arch,
            role="third_party_kernel", graph_motif="mixed_quant_sparse_flash_mla_dsl",
            track="unmerged_pr", mech_id="mqsmla_dsl_544_320",
            dtype_tuple=dsl_in,
            outputs=dsl_out,
            shape=mqs_shape,
            repo_key="ops-transformer",
            rel_path="attention/mixed_quant_sparse_flash_mla/op_dsl/mixed_quant_sparse_flash_mla.py",
            cann_ref="ops-transformer/attention/mixed_quant_sparse_flash_mla/op_dsl/mixed_quant_sparse_flash_mla.py:4094",
            kernels=("mixed_quant_sparse_flash_mla_dsl",) if is_a5 else (),
            note="A5 TileLang DSL MQSMLA (544B FP8 window + 320B FP4 compressed KV; unmerged_pr track requiring ManifestBinding)",
            observed_kernels=observed,
        ),
    ]


def _mindie_sd_diffusion_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    is_a5 = arch == "A5"
    p_low = arch.lower()
    bf16_qkv = ("query:bf16:ND:dense::none", "key:bf16:ND:dense::none", "value:bf16:ND:dense::none")
    fp16_qkv = ("query:fp16:ND:dense::none", "key:fp16:ND:dense::none", "value:fp16:ND:dense::none")
    eqbsa_in = tuple(
        (
            "query:int8:ND:per_token_int8::fp32",
            "key:int8:ND:per_token_int8::fp32",
            v_spec,
            f"blockSparseMask:{m_dt}:ND:none::none",
        )
        for m_dt in ("int8", "int32", "bool")
        for v_spec in ("value:int8:ND:per_token_int8::fp32", "value:fp8_e4m3fn:ND:per_token_fp8::fp32")
        for _ in ("bf16", "fp16")
    )
    eqbsa_out = tuple(
        (f"attentionOut:{o_dt}:ND",)
        for _m in ("int8", "int32", "bool")
        for _v in ("int8", "fp8_e4m3fn")
        for o_dt in ("bf16", "fp16")
    )
    return [
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_rainfusion_bsa",
            abi_id="RainFusionBlockSparseAttention", available=False,
            backend="third_party_wheel", symbol="aclnnRainFusionAttention", arch=arch,
            role="diffusion_attention", graph_motif="rainfusion_bsa",
            dtype_tuple=(
                (*bf16_qkv, "select_idx:int64:ND:none::none", "select_num_idx:int64:ND:none::none", "block_shape:int64:ND:none::none"),
                (*fp16_qkv, "select_idx:int64:ND:none::none", "select_num_idx:int64:ND:none::none", "block_shape:int64:ND:none::none"),
            ),
            outputs=(("attention_out:bf16:ND",), ("attention_out:fp16:ND",)),
            repo_key="mindie-sd", rel_path="csrc/plugin/rainfusionattention.cpp",
            kernels=(),
            note="MindIE-SD plugin wrapper (aclnnRainFusionAttention, int64 indices per rain_fusion_attention_def.cpp:31); unprobed in baseline container",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_ada_bsa",
            abi_id="AdaBlockSparseAttention", available=not is_a5,
            backend="third_party_wheel", symbol="aclnnAdaBlockSparseAttention", arch=arch,
            role="diffusion_attention", graph_motif="ada_bsa",
            dtype_tuple=(
                bf16_qkv, fp16_qkv,
                ("query:int8:ND:per_token_int8::fp32", "key:int8:ND:per_token_int8::fp32", "value:int8:ND:per_token_int8::fp32"),
                ("query:int8:ND:per_token_int8::fp32", "key:int8:ND:per_token_int8::fp32", "value:int8:ND:per_token_int8::fp32"),
            ),
            outputs=(("attention_out:bf16:ND",), ("attention_out:fp16:ND",), ("attention_out:fp16:ND",), ("attention_out:int8:ND",)),
            repo_key="mindie-sd",
            rel_path="csrc/ops/ada_block_sparse_attention/op_host/ada_block_sparse_attention_def.cpp",
            kernels=("AdaBlockSparseAttention",) if not is_a5 else (),
            note="MindIE-SD AdaBlockSparseAttention (AddConfig ascend910b/93 ONLY; all-INT8 QKV outputs fp16/int8 per def.cpp:121)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_eagle_bsa",
            abi_id="EagleBlockSparseAttention", available=not is_a5,
            backend="third_party_wheel", symbol="aclnnEagleBlockSparseAttention", arch=arch,
            role="diffusion_attention", graph_motif="eagle_bsa",
            dtype_tuple=(
                (*bf16_qkv, "blockSparseMask:int8:ND:none::none"),
                (*fp16_qkv, "blockSparseMask:int8:ND:none::none"),
            ),
            outputs=(("attentionOut:bf16:ND",), ("attentionOut:fp16:ND",)),
            repo_key="mindie-sd",
            rel_path="csrc/ops/eagle_block_sparse_attention/op_host/eagle_block_sparse_attention_def.cpp",
            kernels=("EagleBlockSparseAttention",) if not is_a5 else (),
            note="MindIE-SD EagleBlockSparseAttention (AddConfig ascend910b/93 ONLY; requires blockSparseMask:int8 per tiling.cpp:710)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_eagle_quant_bsa",
            abi_id="EagleQuantBlockSparseAttention", available=is_a5,
            backend="third_party_wheel", symbol="aclnnEagleQuantBlockSparseAttention", arch=arch,
            role="diffusion_attention", graph_motif="eagle_quant_bsa",
            dtype_tuple=eqbsa_in,
            outputs=eqbsa_out,
            repo_key="mindie-sd",
            rel_path="csrc/ops/eagle_quant_block_sparse_attention/op_host/eagle_quant_block_sparse_attention_def.cpp",
            kernels=("EagleQuantBlockSparseAttention",) if is_a5 else (),
            note="MindIE-SD EagleQuantBlockSparseAttention (AddConfig ascend950 ONLY; requires INT8 Q/K + blockSparseMask per tiling.cpp:202,443)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_adalayernorm_fused",
            abi_id="AdaLayerNormFused", available=False,
            backend="third_party_wheel", symbol="aclnnAdaLayerNormV2", arch=arch,
            role="diffusion_norm", graph_motif="adalayernorm_fused",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "scale:bf16:ND:dense::none", "shift:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "scale:fp16:ND:dense::none", "shift:fp16:ND:dense::none"),
            ),
            outputs=(
                ("y:bf16:ND", "mean:bf16:ND", "rstd:bf16:ND"),
                ("y:fp16:ND", "mean:fp16:ND", "rstd:fp16:ND"),
            ),
            repo_key="mindie-sd", rel_path="csrc/plugin/adalayernorm.cpp",
            kernels=(),
            note="MindIE-SD plugin wrapper (aclnnAdaLayerNorm/V2; mean/rstd match x dtype per adalayernorm.cpp:77); unprobed in baseline container",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_rms_norm_rope_concat",
            abi_id="RmsNormRopeConcatFused", available=True,
            backend="third_party_wheel", symbol="aclnnNormRopeConcat", arch=arch,
            role="diffusion_norm", graph_motif="norm_rope_concat",
            dtype_tuple=(
                bf16_qkv, fp16_qkv,
                ("query:fp32:ND:dense::none", "key:fp32:ND:dense::none", "value:fp32:ND:dense::none"),
            ),
            outputs=(
                ("query_output:bf16:ND", "key_output:bf16:ND", "value_output:bf16:ND"),
                ("query_output:fp16:ND", "key_output:fp16:ND", "value_output:fp16:ND"),
                ("query_output:fp32:ND", "key_output:fp32:ND", "value_output:fp32:ND"),
            ),
            repo_key="mindie-sd",
            rel_path="csrc/ops/norm_rope_concat/op_host/norm_rope_concat_def.cpp",
            kernels=("NormRopeConcat",),
            note="MindIE-SD NormRopeConcat (AddConfig ascend910b/93/950; requires 4D [B,S,N,D] with num_q_heads == num_kv_heads)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_laurel_residual",
            abi_id="LaurelLowRankResidual", available=False,
            backend="third_party_wheel", symbol="LaurelLowRankResidual", arch=arch,
            role="residual_mixer", graph_motif="laurel_low_rank_residual",
            dtype_tuple=(("x:bf16:ND:dense::none", "w_down:bf16:ND:dense::none", "w_up:bf16:ND:dense::none"),),
            outputs=(("y:bf16:ND",),),
            repo_key="mindie-sd", rel_path="csrc/plugin/register_ops.cpp",
            kernels=(),
            note="Composite residual; no native MindIE-SD wheel operator (lowered via matmul+add)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_quant_four_over_six_a5",
            abi_id="QuantFourOverSixA5", available=is_a5,
            backend="third_party_wheel", symbol="aclnnQuantFourOverSixA5V2", arch=arch,
            role="norm_quant", graph_motif="quant_four_over_six_a5",
            dtype_tuple=(
                ("x:bf16:ND:dense::none",),
                ("x:bf16:ND:dense::none",),
            ),
            outputs=(
                ("y:fp4_e2m1:ND:mx_fp4:32:fp8_e8m0", "mxscale:fp8_e8m0:ND"),
                ("y:uint8:ND:mx_fp4:32:fp8_e8m0", "mxscale:fp8_e8m0:ND"),
            ),
            shape=AbiShapeConstraints(
                rank_range=QUANT_46_RANK_RANGE,
                last_axis_divisibility=QUANT_46_LAST_AXIS_DIV,
            ),
            repo_key="mindie-sd",
            rel_path="csrc/ops/quant_four_over_six_a5/op_host/quant_four_over_six_a5_def.cpp",
            kernels=("QuantFourOverSixA5",) if is_a5 else (),
            note="MindIE-SD A5-only 4/6 adaptive MXFP4 quant (AddConfig ascend950; BF16 rank 1..7, last dim % 32 == 0 -> fp4_e2m1/uint8 + fp8_e8m0)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_mindie_sd_eagle_ffn",
            abi_id="CatlassMatmulAddFfn", available=is_a5,
            backend="third_party_wheel", symbol="aclnnEagleFfnV2", arch=arch,
            role="fused_ffn", graph_motif="eagle_ffn",
            dtype_tuple=(
                ("x:bf16:ND:dense::none", "weight1:bf16:ND:dense::none", "weight2:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "weight1:fp16:ND:dense::none", "weight2:fp16:ND:dense::none"),
            ),
            outputs=(("y:bf16:ND",), ("y:fp16:ND",)),
            activations=frozenset({"gelu", "silu", "swiglu"}),
            repo_key="mindie-sd",
            rel_path="csrc/ops/eagle_ffn/op_host/eagle_ffn_def.cpp",
            kernels=("EagleFfn",) if is_a5 else (),
            note="MindIE-SD A5-only fused EagleFfn (AddConfig ascend950; decomposed linear->act->linear on A2/A3)",
            observed_kernels=observed,
        ),
    ]


def _z_wire_extra_entries(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> list[ProbedAbiEntry]:
    p_low = arch.lower()
    return [
        _mk(
            repos, entry_id=f"{p_low}_cann_inplace_partial_rotary_mul",
            abi_id="InplacePartialRotaryMul", available=True,
            backend="cann_aclnn", symbol="aclnnInplacePartialRotaryMul", arch=arch,
            role="attention", graph_motif="v41_partial_rotary",
            dtype_tuple=(
                ("x:fp16:ND:dense::none", "cos:fp16:ND:dense::none", "sin:fp16:ND:dense::none"),
                ("x:fp32:ND:dense::none", "cos:fp32:ND:dense::none", "sin:fp32:ND:dense::none"),
                ("x:bf16:ND:dense::none", "cos:bf16:ND:dense::none", "sin:bf16:ND:dense::none"),
                ("x:fp16:ND:dense::none", "cos:fp32:ND:dense::none", "sin:fp32:ND:dense::none"),
                ("x:bf16:ND:dense::none", "cos:fp32:ND:dense::none", "sin:fp32:ND:dense::none"),
            ),
            outputs=(
                ("x:fp16:ND",), ("x:fp32:ND",), ("x:bf16:ND",), ("x:fp16:ND",), ("x:bf16:ND",),
            ),
            repo_key="cann-recipes-infer",
            rel_path="ops/ascendc/src/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp",
            cann_ref="cann-recipes-infer/ops/ascendc/src/inplace_partial_rotary_mul/op_host/inplace_partial_rotary_mul_def.cpp:22",
            kernels=("InplacePartialRotaryMul",),
            note="Shipping CANN InplacePartialRotaryMul across A2/A3/A5 (AddConfig ascend910b/93/950; backs v41_partial_rotary)",
            observed_kernels=observed,
        ),
        _mk(
            repos, entry_id=f"{p_low}_cann_kv_compress_epilog",
            abi_id="aclnnKvCompressEpilog", available=True,
            backend="cann_aclnn", symbol="aclnnKvCompressEpilog", arch=arch,
            role="compressor", graph_motif="kv_compress_epilog",
            dtype_tuple=(
                ("cache:uint8:ND:none::none", "x:bf16:ND:dense::none", "slot_mapping:int32:ND:none::none"),
                ("cache:uint8:ND:none::none", "x:bf16:ND:dense::none", "slot_mapping:int64:ND:none::none"),
            ),
            outputs=(("cache:uint8:ND",), ("cache:uint8:ND",)),
            repo_key="ops-transformer",
            rel_path="attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp",
            cann_ref="ops-transformer/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp:24",
            kernels=("KvCompressEpilog",),
            note="Shipping CANN KvCompressEpilog across A2/A3/A5 (AddConfig ascend910b/93/950; uint8 cache + bf16 x + int32/int64 slot_mapping)",
            observed_kernels=observed,
        ),
    ]


def build_snapshot(
    arch: str, repos: dict[str, RepoHandle], observed: set[str],
) -> AbiCatalogSnapshot:
    entries = (
        _mc2_entries(arch, repos, observed)
        + _cann_moe_and_norm_entries(arch, repos, observed)
        + _cann_attention_and_linear_entries(arch, repos, observed)
        + _cann_live_extra_entries(arch, repos, observed)
        + _vllm_ascend_csrc_entries(arch, repos, observed)
        + _triton_and_other_non_cann_entries(arch, repos, observed)
        + _torch_npu_catlass_and_wheel_entries(arch, repos, observed)
        + _mqsmla_entries(arch, repos, observed)
        + _mindie_sd_diffusion_entries(arch, repos, observed)
        + _z_wire_extra_entries(arch, repos, observed)
    )
    return AbiCatalogSnapshot(
        device_arch=arch,  # type: ignore[arg-type]
        sku_id=SOC_SKU_IDS[arch],
        stack_or_cann_version=CANN_STACK_VERSION,
        entries=tuple(entries),
    )


def _run_live_npu_probe(arch: str) -> dict[str, Any]:
    """Optional live-NPU runtime symbol probe (lazy torch_npu import, DEV-ZERO-CTX)."""
    import importlib
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("live-npu mode requires an available Ascend NPU device")
    probed: dict[str, bool] = {}
    for attr in (
        "npu_rms_norm", "npu_add_rms_norm", "npu_dynamic_quant",
        "npu_grouped_matmul", "npu_moe_distribute_dispatch_v2",
        "npu_moe_distribute_combine_v2", "npu_fused_infer_attention_score",
        "npu_clipped_swiglu", "npu_dequant_swiglu_quant",
    ):
        probed[f"torch_npu.{attr}"] = hasattr(torch_npu, attr)
    return {"arch": arch, "device_name": str(torch.npu.get_device_name(0)), "probed": probed}


def _parse_repos(items: list[str]) -> dict[str, RepoHandle]:
    handles: dict[str, RepoHandle] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--repo expects name=path, got {raw!r}")
        name, raw_path = raw.split("=", 1)
        name = name.strip()
        if name not in REPO_URLS:
            raise ValueError(f"unknown repo key {name!r}; expected one of {sorted(REPO_URLS)}")
        root = Path(raw_path).resolve()
        handles[name] = RepoHandle(
            name=name, repository=REPO_URLS[name], root=root, revision=_git_head(root),
        )
    missing = set(REPO_URLS) - {"ops-tensor"} - set(handles)
    if missing:
        raise ValueError(f"missing required --repo mounts: {sorted(missing)}")
    return handles


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a sibling temp file and atomically swap into ``path``."""
    import os
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _build_manifest_payload(
    seals: dict[str, str],
    *,
    arch_files: dict[str, str] | None = None,
    rollback_arch_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": ABI_CATALOG_SCHEMA,
        "description": "Schema index for sealed Ascend hardware-usable operator ABI catalogs (CANN and non-CANN backends).",
        "soc_archs": ["A2", "A3", "A5"],
        "hardware_usable_backends": sorted(HARDWARE_USABLE_ABI_BACKENDS),
        "cann_package_backends": sorted(CANN_PACKAGE_ABI_BACKENDS),
        "non_cann_backends": sorted(NON_CANN_ABI_BACKENDS),
        "open_synthesis_backends": sorted(OPEN_SYNTHESIS_ABI_BACKENDS),
        "arch_manifest_pattern": "cann93_shipping_abi_{arch}.json",
        "sealed_sha256": dict(seals),
    }
    if arch_files is not None:
        manifest["arch_files"] = dict(arch_files)
    if rollback_arch_files:
        manifest["rollback_arch_files"] = dict(rollback_arch_files)
    return manifest


_PUBLISH_LOCK_GUARD = threading.Lock()
_PUBLISH_DIR_LOCKS: dict[str, threading.RLock] = {}


@contextmanager
def _publish_dir_lock(abi_dir: Path) -> Iterator[None]:
    """Serialize concurrent thread and process publications targeting ``abi_dir``."""
    import tempfile

    key = str(abi_dir.resolve())
    with _PUBLISH_LOCK_GUARD:
        tlock = _PUBLISH_DIR_LOCKS.setdefault(key, threading.RLock())
    lock_name = f"mindie_sim_abi_pub_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.lock"
    lock_path = Path(tempfile.gettempdir()) / lock_name
    with tlock:
        try:
            import fcntl
        except ImportError:
            yield
            return
        with lock_path.open("a+b") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _read_existing_manifest_files(manifest_path: Path) -> dict[str, str]:
    """Return the active ``arch -> filename`` mapping from an existing manifest (if readable)."""
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    pattern = str(raw.get("arch_manifest_pattern", "cann93_shipping_abi_{arch}.json"))
    raw_arch = raw.get("arch_files") if isinstance(raw.get("arch_files"), dict) else {}
    return {
        arch: str(raw_arch.get(arch) or pattern.format(arch=arch.lower()))
        for arch in ("A2", "A3", "A5")
    }


def _read_referenced_manifest_files(manifest_path: Path) -> set[str]:
    """Return every active AND rollback catalog filename referenced by the on-disk manifest."""
    if not manifest_path.is_file():
        return set()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(raw, dict):
        return set()
    referenced = set(_read_existing_manifest_files(manifest_path).values())
    raw_rb = raw.get("rollback_arch_files")
    if isinstance(raw_rb, dict):
        referenced.update(str(v) for v in raw_rb.values() if isinstance(v, str) and v)
    return referenced


def publish_sealed_dir(
    abi_dir: Path,
    snapshots: dict[str, AbiCatalogSnapshot],
    *,
    canonical_layout: bool = False,
) -> dict[str, dict[str, Any]]:
    """Stage, verify, and publish a complete A2/A3/A5 generation via ONE atomic manifest switch.

    In default live publication mode (``canonical_layout=False``):
    1. Serialize publishers per directory (thread + cross-process lock).
    2. Stage and verify the full generation in a sibling temporary directory.
    3. Write immutable generation-stamped catalog files into ``abi_dir`` without
       touching the currently active generation or ``abi_schema_manifest.json``.
    4. Perform ONE atomic ``os.replace`` of ``abi_schema_manifest.json`` pointing to the
       new generation.
    5. Re-read the live on-disk manifest before pruning so concurrent or interleaved
       publications NEVER delete active or rollback files referenced by another publisher.
    """
    import re
    import shutil
    import tempfile

    abi_dir.mkdir(parents=True, exist_ok=True)
    seals = {arch: snapshots[arch].payload_sha256 for arch in ("A2", "A3", "A5")}
    gen_tag = hashlib.sha256(
        "|".join(f"{a}:{seals[a]}" for a in ("A2", "A3", "A5")).encode("utf-8")
    ).hexdigest()[:12]
    payloads = {
        arch: json.dumps(snapshots[arch].to_dict(), indent=2, sort_keys=False) + "\n"
        for arch in ("A2", "A3", "A5")
    }
    target_files = {
        arch: (
            f"cann93_shipping_abi_{arch.lower()}.json"
            if canonical_layout else
            f"cann93_shipping_abi_{arch.lower()}.{gen_tag}.json"
        )
        for arch in ("A2", "A3", "A5")
    }
    manifest_path = abi_dir / "abi_schema_manifest.json"

    with _publish_dir_lock(abi_dir):
        prev_files = _read_existing_manifest_files(manifest_path)
        stage_dir = Path(tempfile.mkdtemp(prefix=f".abi_stage_{gen_tag}_", dir=abi_dir.parent))
        written_new: list[Path] = []
        switched = False
        try:
            for arch in ("A2", "A3", "A5"):
                (stage_dir / target_files[arch]).write_text(payloads[arch], encoding="utf-8")
            stage_manifest = _build_manifest_payload(
                seals,
                arch_files=None if canonical_layout else target_files,
                rollback_arch_files=None if canonical_layout else (prev_files or None),
            )
            (stage_dir / "abi_schema_manifest.json").write_text(
                json.dumps(stage_manifest, indent=2) + "\n", encoding="utf-8",
            )
            # Fail before touching abi_dir if the staged generation fails strict verification.
            verify_sealed_dir(stage_dir)

            prev_file_set = set(prev_files.values())
            for arch in ("A2", "A3", "A5"):
                dest = abi_dir / target_files[arch]
                _atomic_write_text(dest, payloads[arch])
                if target_files[arch] not in prev_file_set:
                    written_new.append(dest)

            # Single atomic manifest switch: readers see either complete old gen or complete new gen.
            _atomic_write_text(
                manifest_path,
                json.dumps(stage_manifest, indent=2) + "\n",
            )
            switched = True

            # Re-read the active manifest right before pruning so any concurrent/interleaved
            # publisher's active and rollback files remain strictly protected.
            if not canonical_layout:
                live_referenced = _read_referenced_manifest_files(manifest_path)
                keep = live_referenced | set(target_files.values()) | prev_file_set
                gen_pattern = re.compile(r"^cann93_shipping_abi_(?:a2|a3|a5)\.[0-9a-f]{12}\.json$")
                for child in abi_dir.iterdir():
                    if child.is_file() and gen_pattern.match(child.name) and child.name not in keep:
                        child.unlink(missing_ok=True)

            return verify_sealed_dir(abi_dir)
        except BaseException:
            if not switched:
                protected = _read_referenced_manifest_files(manifest_path)
                for partial in written_new:
                    if partial.name not in protected:
                        partial.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _self_test_catalogs(snapshots: dict[str, AbiCatalogSnapshot]) -> None:
    """Verify every catalog loads strictly and passes legal + neighboring-illegal checks."""
    from mindie_sim.backends.abi_registry import ProbedAbiRegistry
    from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig

    reg = ProbedAbiRegistry(snapshots)
    p = lambda role, dt, shape=(16, 128), layout="ND", qm="dense", sd="none", tr=False: OperandPortSig(
        role=role, dtype=dt, layout=layout, transposed=tr, shape=shape,
        quant_mode=qm, group_shape=(1,), scale_dtype=sd, scale_shape=(),
    )

    def _q(arch: str, abi_id: str, motif: str, ops: tuple[OperandPortSig, ...], outs: tuple[OperandPortSig, ...], *, ws: int = 1, placement: str = "local", act: str = "none") -> AbiQueryKey:
        return AbiQueryKey(
            soc_arch=arch, sku_id=SOC_SKU_IDS[arch].lower(),  # type: ignore[arg-type]
            abi_id=abi_id, graph_motif=motif, model_scope="target", draft_quant_mode="none",
            pr_track="shipping", operands=ops, outputs=outs,
            rank_placement=placement, world_size=ws, comm_group_id="local" if ws == 1 else "tp",  # type: ignore[arg-type]
            comm_domain_width=ws, comm_dtype="none" if ws == 1 else "bf16",
            reduction_kind="none", activation_kind=act, activation_params=(),
            has_bias=False, has_active_mask=False, dispatch_quant_mode=-1,
            shared_experts_per_rank=0, top_k=0, moe_expert_num=0, qk_rope_head_dim=64,
            mla_variant="none", sfa_dcp_active=False, pcp_size=1, kv_cache_dtype="bf16",
            has_kpool_indexer=False, dcp_size=1, is_act_and_mul=False, enabled_stack_passes=frozenset(),
        )

    # 1. Invented symbol MUST be rejected even with a real motif.
    assert reg.verify(_q("A5", "NoSuchShippingSymbol", "npu_rms_norm", (p("self", "bf16"), p("gamma", "bf16")), (p("y", "bf16"), p("rstd", "fp32")))).status == "rejected_abi"
    # 2. torch_npu.npu_rms_norm: legal (self, gamma) -> (y, rstd) admitted; unary -> int8 rejected.
    assert reg.verify(_q("A5", "torch_npu.npu_rms_norm", "npu_rms_norm", (p("self", "bf16"), p("gamma", "bf16")), (p("y", "bf16"), p("rstd", "fp32")))).status == "admitted_shipping"
    assert reg.verify(_q("A5", "torch_npu.npu_rms_norm", "npu_rms_norm", (p("x", "bf16"),), (p("y", "int8"),))).status == "rejected_abi"
    # 3. AttnResFwd (OpDef order) & attn_res_fwd (Torch order) verified strictly.
    attnres_in = tuple(p(r, "bf16") for r in ("prefix_sum", "block_residual", "proj_weight", "norm_weight", "addend", "output_norm"))
    attnres_out = tuple(p(r, "bf16") for r in ("hidden_states", "prefix_out", "materialized"))
    assert reg.verify(_q("A3", "AttnResFwd", "attn_res_prefix", attnres_in, attnres_out)).status == "admitted_shipping"
    assert reg.verify(_q("A3", "AttnResFwd", "attn_res_prefix", (p("x", "fp16"),), (p("y", "fp16"),))).status == "rejected_abi"
    attnres_torch_in = tuple(p(r, "bf16") for r in ("prefix_sum", "addend", "block_residual", "proj_weight", "norm_weight", "output_norm_weight"))
    assert reg.verify(_q("A5", "attn_res_fwd", "attn_res_prefix", attnres_torch_in, attnres_out)).status == "admitted_shipping"
    assert reg.verify(_q("A5", "attn_res_fwd", "attn_res_prefix", attnres_in, attnres_out)).status == "rejected_abi"
    # 4. aclnnAllGatherMatmulV2: requires 2D x1 (untransposed), 2D x2, out=(m*ws, n); rejects 1D, 3D, transA, wrong M/K/dtype.
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "admitted_shipping"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (16, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (987, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (512,)), p("x2", "bf16", (13, 64))), (p("y", "bf16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (2, 16, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512), tr=True), p("x2", "bf16", (512, 64))), (p("y", "bf16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 64))), (p("y", "fp16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (13, 64))), (p("y", "bf16", (32, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    # 4b. Scalar operands, A5 32 GiB CCU overflow (> 15.75 GiB), and A5 vs A3 N=0 empty-tensor contract.
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", ()), p("x2", "bf16", ())), (p("y", "bf16", ()),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16_777_216, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (33_554_432, 64)),), ws=2, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16_777_216, 512)), p("x2", "bf16", (512, 64))), (p("y", "bf16", (33_554_432, 64)),), ws=2, placement="ub_aicpu_supernode")).status == "admitted_shipping"
    assert reg.verify(_q("A5", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 0))), (p("y", "bf16", (32, 0)),), ws=2, placement="ub_ccu_intramachine")).status == "admitted_shipping"
    assert reg.verify(_q("A3", "aclnnAllGatherMatmulV2", "allgather_matmul", (p("x1", "bf16", (16, 512)), p("x2", "bf16", (512, 0))), (p("y", "bf16", (32, 0)),), ws=2, placement="hccs_supernode_a3")).status == "rejected_abi"
    # 5. aclnnDequantSituQuant: requires (y:int8, y_scale:fp32) output; rejects BF16 output & A5.
    assert reg.verify(_q("A3", "aclnnDequantSituQuant", "dequant_situ_quant", (p("x", "bf16"),), (p("y", "int8"), p("y_scale", "fp32")), act="situ")).status == "admitted_shipping"
    assert reg.verify(_q("A3", "aclnnDequantSituQuant", "dequant_situ_quant", (p("x", "bf16"),), (p("y", "bf16"),), act="situ")).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnDequantSituQuant", "dequant_situ_quant", (p("x", "bf16"),), (p("y", "int8"), p("y_scale", "fp32")), act="situ")).status == "rejected_abi"
    # 6. aclnnTextGatedResidual: requires 3 inputs (x, w_down, w_up) -> mixed; rejects unary.
    assert reg.verify(_q("A5", "aclnnTextGatedResidual", "text_gated_residual", (p("x", "bf16", (2, 8, 64)), p("w_down", "bf16", (16, 64)), p("w_up", "bf16", (64, 16))), (p("mixed", "bf16", (2, 8, 16)),))).status == "admitted_shipping"
    assert reg.verify(_q("A5", "aclnnTextGatedResidual", "text_gated_residual", (p("x", "bf16"),), (p("y", "bf16"),))).status == "rejected_abi"
    # 7. A5 MlaPrologV3 requires all 11 inputs, BF16 ND 3D weight_uk, 3D/4D kv_cache_ref, and (T,N,Hckv) query_out.
    prolog_bf16_in = (
        p("token_x", "bf16", (16, 7168)), p("weight_dq", "bf16", (7168, 1536), layout="FRACTAL_NZ"),
        p("weight_uq_qr", "bf16", (1536, 1536), layout="FRACTAL_NZ"), p("weight_uk", "bf16", (128, 128, 512), layout="ND"),
        p("weight_dkv_kr", "bf16", (7168, 576), layout="FRACTAL_NZ"), p("rmsnorm_gamma_cq", "bf16", (1536,)),
        p("rmsnorm_gamma_ckv", "bf16", (512,)), p("rope_sin", "bf16", (16, 64)), p("rope_cos", "bf16", (16, 64)),
        p("kv_cache_ref", "bf16", (64, 64, 1, 512)), p("kr_cache_ref", "bf16", (64, 64, 1, 64)),
    )
    prolog_bf16_out = (p("query_out", "bf16", (16, 128, 512)), p("query_rope_out", "bf16", (16, 128, 64)), p("query_norm_out", "bf16", (16, 1536)), p("dequant_scale_q_norm_out", "fp32", (16,)))
    assert reg.verify(_q("A5", "aclnnMlaPrologV3WeightNz", "mla_prolog", prolog_bf16_in, prolog_bf16_out)).status == "admitted_shipping"
    assert reg.verify(_q("A5", "aclnnMlaPrologV3WeightNz", "mla_prolog", (p("x", "bf16", (16, 7168)), p("w_dq", "bf16", (7168, 1536), layout="FRACTAL_NZ")), (p("q_nope", "bf16"), p("q_rope", "bf16")))).status == "rejected_abi"
    prolog_nz_uk = (*prolog_bf16_in[:3], p("weight_uk", "bf16", (128, 128, 512), layout="FRACTAL_NZ"), *prolog_bf16_in[4:])
    assert reg.verify(_q("A5", "aclnnMlaPrologV3WeightNz", "mla_prolog", prolog_nz_uk, prolog_bf16_out)).status == "rejected_abi"
    prolog_2d_kv = (*prolog_bf16_in[:9], p("kv_cache_ref", "bf16", (64, 512)), prolog_bf16_in[10])
    assert reg.verify(_q("A5", "aclnnMlaPrologV3WeightNz", "mla_prolog", prolog_2d_kv, prolog_bf16_out)).status == "rejected_abi"
    # 8. torch_npu.npu_moe_gating_top_k & aclnnMoeGatingTopK: bias optional (1D), max_experts=2048.
    gating_out = (p("y", "bf16"), p("expert_idx", "int32"), p("out", "fp32"))
    assert reg.verify(_q("A5", "torch_npu.npu_moe_gating_top_k", "moe_routing", (p("x", "bf16", (4, 128)),), gating_out)).status == "admitted_shipping"
    assert reg.verify(_q("A5", "torch_npu.npu_moe_gating_top_k", "moe_routing", (p("x", "bf16", (4, 128)), p("bias", "bf16", (128,))), gating_out)).status == "admitted_shipping"
    assert reg.verify(_q("A5", "torch_npu.npu_moe_gating_top_k", "moe_routing", (p("x", "bf16", (4, 128)), p("bias", "bf16", (4, 128))), gating_out)).status == "rejected_abi"
    assert reg.verify(_q("A5", "aclnnMoeGatingTopK", "moe_gating_top_k", (p("x", "bf16", (8, 4096)),), gating_out)).status == "rejected_abi"
    # 9. SparseFlashMla: BF16 ONLY (rejects FP16).
    assert reg.verify(_q("A5", "SparseFlashMla", "sparse_flash_mla", (p("q", "bf16"), p("ori_kv", "bf16")), (p("attn_out", "bf16"),))).status == "admitted_shipping"
    assert reg.verify(_q("A5", "SparseFlashMla", "sparse_flash_mla", (p("q", "fp16"), p("ori_kv", "fp16")), (p("attn_out", "fp16"),))).status == "rejected_abi"
    # 10. Triton _prepare_local_sort_inputs: (indices:int32, scores:fp32)->(keys:fp32, values:fp32); rejects values:int32.
    assert reg.verify(_q("A5", "_prepare_local_sort_inputs", "sharded_sort", (p("indices", "int32", qm="none"), p("scores", "fp32")), (p("keys", "fp32"), p("values", "fp32")))).status == "admitted_shipping"
    assert reg.verify(_q("A5", "_prepare_local_sort_inputs", "sharded_sort", (p("indices", "int32", qm="none"), p("scores", "fp32")), (p("keys", "fp32"), p("values", "int32")))).status == "rejected_abi"
    # 11. vllm_ascend_csrc IndexerRegionScore: 6 required inputs -> 2 outputs admitted on A5; truncated rejected.
    irs_in = (p("query", "bf16"), p("weights", "bf16"), p("key_summary", "bf16"), p("actual_kv_lengths", "int32", qm="none"), p("block_table", "int32", qm="none"), p("cu_seq_lens", "int32", qm="none"))
    assert reg.verify(_q("A5", "IndexerRegionScore", "region_index_score", irs_in, (p("output", "fp32"), p("selected_region_ids", "int32")))).status == "admitted_shipping"
    assert reg.verify(_q("A5", "IndexerRegionScore", "region_index_score", (p("query", "bf16"), p("key_summary", "bf16")), (p("output", "fp32"),))).status == "rejected_abi"
    # 12. TransposeBatchMatMul: both CANN and torch_npu reject 2D and admit rank-3 perm_y=(1,0,2).
    for tbmm_id in ("aclnnTransposeBatchMatMul", "torch_npu.npu_transpose_batchmatmul"):
        assert reg.verify(_q("A5", tbmm_id, "transpose_batchmatmul", (p("x1", "bf16", (32, 128)), p("x2", "bf16", (128, 64))), (p("y", "bf16", (32, 64)),))).status == "rejected_abi"
        assert reg.verify(_q("A5", tbmm_id, "transpose_batchmatmul", (p("x1", "bf16", (8, 32, 128)), p("x2", "bf16", (8, 128, 64))), (p("y", "bf16", (32, 8, 64)),))).status == "admitted_shipping"
    # 13. Standalone HCCL collective output-shape conservation (AllGather / ReduceScatter).
    assert reg.verify(_q("A5", "hcom_allGather", "hcom_allGather", (p("x", "bf16", (16, 4096)),), (p("y", "bf16", (16, 4096)),), ws=8, placement="ub_ccu_intramachine")).status == "rejected_abi"
    assert reg.verify(_q("A5", "hcom_allGather", "hcom_allGather", (p("x", "bf16", (16, 4096)),), (p("y", "bf16", (128, 4096)),), ws=8, placement="ub_ccu_intramachine")).status == "admitted_shipping"
    # 14. A5 aclnnMixedQuantSparseFlashMla full positional PA_BBND CSA admitted; 2-arg draft rejected; A3 rejected.
    mqs_p = lambda role, dt, shape, layout="ND", qm="none", gs=(1,), sd="none": OperandPortSig(
        role=role, dtype=dt, layout=layout, transposed=False, shape=shape,
        quant_mode=qm, group_shape=gs, scale_dtype=sd, scale_shape=(),
    )
    mqs_csa_in = (
        mqs_p("q", "bf16", (2, 1, 64, 512), qm="dense"),
        mqs_p("ori_kv", "fp8_e4m3fn", (32, 64, 1, 608), layout="PA_BBND", qm="packed_kv_608", gs=(64,), sd="bf16"),
        mqs_p("cmp_kv", "fp8_e4m3fn", (16, 64, 1, 608), layout="PA_BBND", qm="packed_kv_608", gs=(64,), sd="bf16"),
        mqs_p("cmp_sparse_indices", "int32", (2, 1, 1, 512)),
        mqs_p("ori_block_table", "int32", (2, 16)),
        mqs_p("cmp_block_table", "int32", (2, 8)),
        mqs_p("seqused_ori_kv", "int32", (2,)),
        mqs_p("seqused_cmp_kv", "int32", (2,)),
        mqs_p("sinks", "fp32", (64,), qm="dense"),
        mqs_p("metadata", "int32", (2048,)),
    )
    mqs_out = (mqs_p("attn_out", "bf16", (2, 1, 64, 512), qm="dense"),)
    assert reg.verify(_q("A5", "aclnnMixedQuantSparseFlashMla", "mixed_quant_sparse_flash_mla", mqs_csa_in, mqs_out)).status == "admitted_shipping"
    assert reg.verify(_q("A5", "aclnnMixedQuantSparseFlashMla", "mixed_quant_sparse_flash_mla", mqs_csa_in[:2], mqs_out)).status == "rejected_abi"
    assert reg.verify(_q("A3", "aclnnMixedQuantSparseFlashMla", "mixed_quant_sparse_flash_mla", mqs_csa_in, mqs_out)).status == "rejected_abi"


def verify_sealed_dir(abi_dir: Path) -> dict[str, dict[str, Any]]:
    """Strictly verify all 3 sealed catalogs via ``ProbedAbiRegistry`` (requires all seals)."""
    from mindie_sim.backends.abi_registry import ProbedAbiRegistry

    reg = ProbedAbiRegistry.from_directory(abi_dir)
    summary: dict[str, dict[str, Any]] = {}
    for arch in ("A2", "A3", "A5"):
        snap = reg.get_snapshot(arch)
        summary[arch] = {
            "total": len(snap.entries),
            "available": len(snap.available_entries()),
            "cann_package": sum(1 for item in snap.entries if item.is_cann_package),
            "non_cann_package": sum(1 for item in snap.entries if item.is_non_cann_package),
            "payload_sha256": snap.payload_sha256,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("source-seal", "verify", "live-npu"), default="verify")
    parser.add_argument("--abi-dir", type=Path, default=REPO_ROOT / "data" / "abi")
    parser.add_argument("--repo", action="append", default=[], help="Mounted upstream repo: name=/path")
    parser.add_argument("--capture-a3", type=Path, default=None)
    parser.add_argument("--capture-a5", type=Path, default=None)
    parser.add_argument("--arch", choices=("A2", "A3", "A5"), default="A3")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--canonical-layout", action="store_true",
        help="Write canonical cann93_shipping_abi_{a2,a3,a5}.json names (repo checkout sealing only).",
    )
    args = parser.parse_args(argv)

    if args.mode == "live-npu":
        print(json.dumps(_run_live_npu_probe(args.arch), indent=2))
        return 0
    if args.mode == "verify":
        print(json.dumps(verify_sealed_dir(args.abi_dir), indent=2))
        return 0

    repos = _parse_repos(args.repo)
    obs_by_arch = {
        "A2": set(),
        "A3": _load_capture_kernels(args.capture_a3),
        "A5": _load_capture_kernels(args.capture_a5),
    }
    snapshots: dict[str, AbiCatalogSnapshot] = {}
    seals: dict[str, str] = {}
    stats: dict[str, dict[str, Any]] = {}
    for arch in ("A2", "A3", "A5"):
        snap = build_snapshot(arch, repos, obs_by_arch[arch])
        snapshots[arch] = snap
        seals[arch] = snap.payload_sha256
        stats[arch] = {
            "total": len(snap.entries),
            "available": len(snap.available_entries()),
            "cann_package": sum(1 for e in snap.entries if e.is_cann_package),
            "non_cann_package": sum(1 for e in snap.entries if e.is_non_cann_package),
            "live_observed_entries": sum(1 for e in snap.entries if "live_silicon_observed=" in e.probe_domain_note),
            "payload_sha256": snap.payload_sha256,
        }
    _self_test_catalogs(snapshots)
    if args.write:
        publish_sealed_dir(args.abi_dir, snapshots, canonical_layout=args.canonical_layout)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
