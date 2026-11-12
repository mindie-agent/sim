"""Native invocation binding for the normal bound-operator piecewise path.

[Role] Attach the existing BIND1 native-call contract to the path
  ``instantiate_model -> BoundOperator -> OperatorPiecewiseCompiler ->
  FamilyTemplate -> OperatorCost`` and expose one typed occurrence per
  evaluated operator.  This is the ordinary subgraph prediction entry; it
  never routes through ``OpSpec`` and it does not price or select work.
[Invariants]  A call identity is the immutable selected implementation
  signature.  Layer/repeat/index/rank live in the occurrence.  A fused call is
  charged once at its owning cost; member rows are marked ``charge=member``.
  Header files are not physical launches.  Unknown entry/source/ABI stays in
  ``unresolved``; no target CSV, model label or case id selects a symbol.
[Timeouts & Locks]  Pure bounded metadata arithmetic; no I/O, device, weights,
  runtime imports, sleep or locks.
[Side-Effect Policy]  Returns immutable records; imports source *declarations*
  from existing family modules only, never a device runtime or profiler.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from mindie_sim.prediction.mx_source_provenance import (
    image_digest, source_revision_from_oci)
from mindie_sim.prediction.native_binding import (
    NativeCallBinding, bind_native_call, declare_native_call,
    validate_invocation, coverage_from_records,
)
from mindie_sim.schema.common import ModelingError
from .native_gdn import dynamic_gdn, gdn_dims
from .native_qbmm import (
    QBMM_SOURCES, R35_IMAGE_SHA, dynamic_qbmm)
from mindie_sim.prediction.indexer_source import (
    declaration_from_theta, select_indexer_source)
from .native_moe_token_permute import (
    binding_spec as _op_perm2_spec,
    dynamic_call as _op_perm2_dynamic,
)
from .native_r64_histogram import (
    binding_spec as _op_hist_spec,
    dynamic_call as _op_hist_dynamic,
)
from .native_moe_init_routing_v3 import (
    ENTRY as _MOE_INIT3_ENTRY,
    binding_spec as _moe_init3_spec,
    dynamic_call as _moe_init3_dynamic,
)
from .native_add_rms_norm import (
    binding_spec as _add_rms_spec,
    dynamic_call as _add_rms_dynamic,
)
from .native_gated_layer_norm_npu import (
    ENTRY as _GATED_NORM_ENTRY,
    binding_spec as _gated_norm_spec,
    dynamic_call as _gated_norm_dynamic,
)
from .native_rms_norm import (
    binding_spec as _rms_plain_spec,
    dynamic_call as _rms_plain_dynamic,
)
from .native_r64_muls_add import (
    ENTRY as _MULS_ADD_ENTRY,
    binding_spec as _muls_add_spec,
    dynamic_call as _muls_add_dynamic,
)

from .native_r64_add_rms_norm_bias import (
    ENTRY as _ADD_RMS_BIAS_ENTRY,
    binding_spec as _add_rms_bias_spec,
    dynamic_call as _add_rms_bias_dynamic,
)
from .native_r64_triton_rope_siso import (
    binding_spec as _op_rope_siso_spec,
    dynamic_call as _op_rope_siso_dynamic,
)
from .native_r64_layer_norm_with_impl_mode import (
    ENTRY as _LNIM_ENTRY,
    binding_spec as _lnim_spec,
    dynamic_call as _lnim_dynamic,
)
from .native_gather_ssm_states import (
    binding_spec as _ssm_gather_spec,
    dynamic_call as _ssm_gather_dynamic,
)
from .native_scatter_ssm_states import (
    binding_spec as _ssm_scatter_spec,
    dynamic_call as _ssm_scatter_dynamic,
)

from .native_split_qkv_rmsnorm_mrope import (
    ENTRY as _QKVMR_ENTRY,
    binding_spec as _qkvmr_spec,
    dynamic_call as _qkvmr_dynamic,
)
from .native_swiglu import (
    binding_spec as _swiglu_spec,
    dynamic_call as _swiglu_dynamic,
)
from .native_r31_dynamic_quant_v2 import (
    binding_spec as _opdquant_spec,
    dynamic_call as _opdquant_dynamic,
)
from .native_r31_sigmoid import (
    binding_spec as _sig_spec,
    dynamic_call as _sig_dynamic,
)
from .native_r64_scatter_pa_cache import (
    binding_spec as _scatpa_spec,
    dynamic_call as _scatpa_dynamic,
)
from .native_r31_inplace_zero import (
    binding_spec as _inzero_spec,
    dynamic_call as _inzero_dynamic,
)
from .native_grouped_matmul_v4 import (
    ENTRY as _GMM2_ENTRY,
    binding_spec as _gmm2_spec,
    dynamic_call as _gmm2_dynamic,
)

from .native_l2norm_fwd_kernel2 import (
    binding_spec as _l2n_spec,
    dynamic_call as _l2n_dynamic,
)
from .native_r64_transpose_batch_matmul import (
    binding_spec as _tbg_spec,
    dynamic_call as _tbg_dynamic,
)
from .native_r31_inplace_copy import (
    binding_spec as _incopy_spec,
    dynamic_call as _incopy_dynamic,
)
from .native_grouped_matmul_swiglu_quant_v2 import (
    ENTRY as _GMM1_ENTRY,
    binding_spec as _gmm1_spec,
    dynamic_call as _gmm1_dynamic,
)
from .native_r31_quantmatmul_weight_nz import (
    binding_spec as _qbmm31_spec,
    dynamic_call as _qbmm31_dynamic,
)
from .native_r31_matmul_bf16 import (
    binding_spec as _mat_spec,
    dynamic_call as _mat_dynamic,
)
from .native_r64_aclnn_cat import (
    binding_spec as _cat_spec,
    dynamic_call as _cat_dynamic,
)
from .native_r31_aclnn_index import (
    binding_spec as _index_spec,
    dynamic_call as _index_dynamic,
)


_SFA_ENTRY = "torch_npu.npu_kv_quant_sparse_flash_attention"
_SFA_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d",
    "symbol": "A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm",
    "revision": "r64/8d422528",
    "state": "task_artifact",
}
_SFA_SOURCE_R35 = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "f4392f346816c85f138d7e033a924c6b628ae684cd4fcd7a6d8818518351b5c9",
    "symbol": "A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm",
    "revision": "r35/5461a1a8",
    "state": "task_artifact",
}
_SFA_SOURCES = {
    "r64": {"source": _SFA_SOURCE, "version": "r64/8d422528",
            "lines": [1122, 1181], "qmm_lines": [1158, 1177],
            "caller_sha256": "225931480b6ce5d275415d32681aabf4517617769a8a80077ca4454bb9e91a9a"},
    "r35": {"source": _SFA_SOURCE_R35, "version": "r35/5461a1a8",
            "lines": [1112, 1172], "qmm_lines": [1147, 1165],
            "caller_sha256": None},
}
_SFA_SOURCE_ENTRY_PREFIX = "3054531d"


def _sfa_descriptor(shape: list, dtype: str, layout: str,
                    role: str) -> dict:
    return {"shape": [None if item is None else int(item) for item in shape],
            "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}

SCHEMA = "mindie-sim/piecewise-native-call/1"
OCCURRENCE_SCHEMA = "mindie-sim/piecewise-native-occurrence/1"

_DTYPE_ALIAS = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
_LOW_PRECISION = frozenset({"int8", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8",
                            "mx_fp8", "mxfp8_e4m3", "int4", "mxfp4", "w8a8",
                            "w4a8", "int4_per_channel"})
_CALL_ATTRS = frozenset({
    "transpose_x1", "transpose_x2", "group_list_type", "group_sizes",
    "page_size", "cache_layout", "input_layout", "sparse_mode",
    "attention_mode", "quant_mode", "scale_repo_mode", "query_norm_flag",
    "chunk", "chunk_size", "use_exp2", "use_qk_l2norm_in_kernel",
    "disable_recompute", "state_v_first", "scale", "ops_per_element",
    "transfer", "row_source", "offset_elements", "head_dim", "kept_heads",
    "source_heads", "group_axis", "members", "stages", "collective",
    "topology", "axis", "wire_bytes", "row_binding", "sp_rows", "kernel",
    "source_kernel", "native_kind", "quant_op", "out_format",
    "output_format", "input_format", "activation_scale_format",
    "activation_scale_shape", "tiling_key", "source_branch",
    "source_revision", "mx_algorithm", "weight_quant_mode",
    "kv_cache_quant_mode", "packed_row_bytes", "store_row_bytes",
})
_HCCL_ENTRIES = {
    "allreduce": "torch.distributed.all_reduce",
    "all_gather": "torch.distributed.all_gather_into_tensor",
    "gather": "torch.distributed.all_gather_into_tensor",
    "reduce_scatter": "torch.distributed.reduce_scatter_tensor",
}
_HCCL_SOURCE = "torch/distributed/distributed_c10d.py"


@dataclass(frozen=True)
class _Surface:
    """BIND1's existing operator view, projected from a BoundOperator."""

    op_id: str
    family: str
    recipe: str
    phase: str
    source_boundary: str
    dtype: str | None
    layout: str | None
    local_shape: tuple = ()
    padded_shape: tuple = ()
    launch_shape: tuple = ()
    returns: tuple = ()
    matrix_flops: tuple = ()
    metadata: tuple = ()


@dataclass(frozen=True)
class _CallSpec:
    api: dict
    source: dict | None
    typed: dict
    kernels: tuple
    members: tuple
    owner: str
    execution_core: str | None
    unresolved: tuple


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _dtype(theta: Any, annotations: Mapping[str, Any]) -> str | None:
    for key in ("dtype", "payload_dtype", "act_format", "activation_format",
                "input_dtype", "output_format", "out_format"):
        value = theta.find(key, None) or annotations.get(key)
        if isinstance(value, str) and value:
            return _DTYPE_ALIAS.get(value, value)
    for key in ("act_bytes", "result_bytes", "bytes_per_element"):
        value = theta.find(key, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {1.0: "fp8_e4m3", 2.0: "bf16", 4.0: "fp32"}.get(float(value))
    return None


def _layout(theta: Any, family: str) -> str:
    for key in ("input_layout", "layout", "cache_layout"):
        value = theta.find(key, None)
        if isinstance(value, str) and value:
            return value
    if family in {"collective", "transport"}:
        return "contiguous"
    return "unknown"


def _annotations(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {str(key): item for key, item in (value or ())}


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or (
        isinstance(value, (tuple, list)) and all(_scalar(item) for item in value))


def _typed_view(theta: Any, annotations: Mapping[str, Any], *,
                family: str, name: str) -> dict[str, Any]:
    """Only declared call fields; token dimensions stay in the occurrence."""
    annotations = _annotations(annotations)
    dtype = _dtype(theta, annotations)
    layout = _layout(theta, family)
    matrix, heads = dict(theta.matrix), dict(theta.heads)
    quant, extras = dict(theta.quant), dict(theta.extras)
    shapes: dict[str, list] = {}
    attrs: dict[str, Any] = {}
    if "K" in matrix and "N" in matrix:
        k, n = int(matrix["K"]), int(matrix["N"])
        shapes["weight"] = [k, n]
        attrs.update(matrix_k=k, matrix_n=n)
    if "elements" in matrix:
        attrs["elements"] = int(matrix["elements"])
    if "rows" in matrix:
        attrs["rows"] = int(matrix["rows"])
        if "row_bytes" in matrix:
            attrs["row_bytes"] = float(matrix["row_bytes"])
    if heads:
        attrs["heads"] = {key: int(value) for key, value in sorted(heads.items())}
        if family in {"attention", "mla", "fused"}:
            width = (int(heads.get("dn", 0)) + int(heads.get("dr", 0))
                     or int(heads.get("dk", 0)))
            shapes["q"] = [int(heads.get("q", 1)), width]
            shapes["output"] = [int(heads.get("q", 1)),
                                int(heads.get("dv", width) or width)]
    for key in ("weight_format", "act_format", "scale_group", "quantized",
                "weight_block", "activation_scale_format"):
        if key in quant and _scalar(quant[key]):
            attrs[key] = quant[key]
    for key in ("causal", "window", "index_topk", "kpool", "absorb", "direct_q",
                "kv_bytes", "row_bytes", "read_bytes", "write_bytes"):
        if key in extras and _scalar(extras[key]):
            attrs[key] = extras[key]
    for key in _CALL_ATTRS:
        if key in extras and _scalar(extras[key]):
            attrs[key] = extras[key]
    if family in {"collective", "transport"}:
        collective = str(annotations.get("collective") or extras.get("collective") or "")
        ranks = int(theta.find("members", extras.get("members", 0)) or 0)
        wire = theta.find("wire_bytes", theta.find("bytes_per_token", None))
        attrs["hccl_collective"] = collective
        attrs["hccl_ranks"] = ranks
        attrs["hccl_topology"] = str(annotations.get("topology") or "declared")
        attrs["hccl_dtype_bytes"] = _dtype_bytes(dtype)
        if wire is not None:
            attrs["hccl_payload_bytes"] = float(wire)
    return {
        "dtype": dtype,
        "layout": layout,
        "shapes": {key: value for key, value in sorted(shapes.items()) if value},
        "attrs": {key: value for key, value in sorted(attrs.items())},
        "rank_width": max((len(value) for value in shapes.values()
                           if isinstance(value, list)), default=0),
    }


def _dtype_bytes(dtype: str | None) -> int:
    if dtype in {"fp32", "int32"}:
        return 4
    if dtype in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2", "uint8"}:
        return 1
    return 2


def _source(repository: str, path: str, *, sha256: str | None = None,
            symbol: str | None = None, revision: str | None = None,
            state: str = "source") -> dict:
    return {"repository": repository, "path": path, "sha256": sha256,
            "symbol": symbol, "revision": revision, "state": state}


def _api(namespace: str, entry: str | None, *,
         aclnn_symbol: str | None = None, version: str | None = None) -> dict:
    return {"namespace": namespace, "entry": entry,
            "aclnn_symbol": aclnn_symbol, "version": version}


def _kernel(name: str, role: str, *, path: str | None = None,
            sha256: str | None = None, evidence: str = "source_declared",
            **extra: Any) -> dict:
    row = {"name": name, "role": role, "source_path": path,
           "source_sha256": sha256, "evidence": evidence}
    row.update({key: value for key, value in extra.items() if value is not None})
    return row


def _call_spec(api: dict, source: dict | None, typed: dict, kernels: Sequence[dict],
               members: Sequence[str], owner: str, *,
               execution_core: str | None = None,
               unresolved: Sequence[str] = ()) -> _CallSpec:
    return _CallSpec(api, source, typed, tuple(kernels), tuple(members), owner,
                     execution_core, tuple(dict.fromkeys(unresolved)))


def _declared_kind(theta: Any) -> str | None:
    for key in ("native_kind", "source_kernel", "kernel", "quant_op"):
        value = theta.find(key, None)
        if isinstance(value, str) and value:
            return value
    return None


def _variant(compiled: Any) -> str:
    variant = str(getattr(compiled, "variant", "") or "")
    if variant:
        return variant
    names = {operator.name for operator in getattr(compiled, "siblings", ())}
    return "recurrent_decode" if names & {"q_l2norm", "k_l2norm"} else "chunked_prefill"


def _source_or_none(theta: Any) -> dict | None:
    """The source record already declared by the selected family module."""
    from mindie_sim.prediction import mx_dynamic_quant
    kind = _declared_kind(theta)
    extras = dict(theta.extras)
    if kind == "DynamicMxQuant" or extras.get("quant_op") == "DynamicMxQuant":
        revision = str(extras.get("source_revision", "unpinned"))
        record = mx_dynamic_quant.SOURCES.get(revision, mx_dynamic_quant.SOURCE)
        path = record.get("path")
        if path:
            return _source("cann", path, sha256=record.get("sha256"),
                           symbol=record.get("kernel"),
                           revision=record.get("image", record.get("revision")),
                           state="task_artifact")
    return None


def _dynamic_mx_spec(theta: Any, annotations: Mapping[str, Any], *,
                     name: str) -> _CallSpec | None:
    extras = dict(theta.extras)
    if (extras.get("native_kind") != "DynamicMxQuant"
            and extras.get("quant_op") != "DynamicMxQuant"):
        return None
    from mindie_sim.prediction import mx_dynamic_quant
    revision = str(extras.get("source_revision", "unpinned"))
    record = mx_dynamic_quant.SOURCES.get(revision, mx_dynamic_quant.SOURCE)
    typed = _typed_view(theta, annotations, family="quantize", name="DynamicMxQuant")
    typed["attrs"].update({
        "out_format": "fp8_e4m3", "scale_format": "fp8_e8m0", "group_size": 32,
        "tiling_key": extras.get("tiling_key"),
        "branch": extras.get("source_branch") or "tail_axis_block32_fp8",
    })
    source = _source_or_none(theta)
    kernels = (_kernel(str(record.get("kernel") or "DynamicMxQuantTailAxisFP8"),
                       "vector", path=record.get("path"),
                       sha256=record.get("kernel_sha256")
                       or record.get("sha256")),)
    unresolved = ["aclnn_symbol_unresolved"]
    if source is None:
        unresolved.append("source_reference_unresolved")
    return _call_spec(
        _api("torch", "torch_npu.npu_dynamic_mx_quant"), source, typed,
        kernels, (), name, unresolved=unresolved)


def _qbm_spec(theta: Any, annotations: Mapping[str, Any], *,
              name: str, family: str,
              feature_metadata: Iterable[tuple[str, Any]] = ()
              ) -> _CallSpec | None:
    """Admit the R64 standalone MXFP8 wrapper only from the normal image id."""
    if family != "gemm":
        return None
    matrix, quant = dict(theta.matrix), dict(theta.quant)
    if "K" not in matrix or "N" not in matrix:
        return None
    k, n = int(matrix["K"]), int(matrix["N"])
    weight = str(quant.get("weight_format", ""))
    act = str(quant.get("act_format", ""))
    method = str(quant.get("linear_method", ""))
    group = int(quant.get("scale_group", 0) or 0)
    if (method != "mx_fp8" or weight != "fp8_e4m3" or act != "fp8_e4m3"
            or group != 32):
        return None
    pairs = ((k + group - 1) // group + 1) // 2
    bias = bool(theta.find("bias", False))
    meta = dict(feature_metadata or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    digest = image_digest(image_id)
    revision = source_revision_from_oci(
        str(meta.get("image") or meta.get("deployment_image")
            or meta.get("oci_image") or ""), image_id)
    # The shared selector deliberately leaves R35 unpinned for DynamicMxQuant;
    # the standalone QBMM R35 wrapper is separately source-extracted, so only
    # its exact digest selects that branch here.
    r35 = digest.startswith(R35_IMAGE_SHA)
    selected = ("r64" if revision == "r64"
                else "r35" if r35 else None)
    typed = {
        "dtype": "fp8_e4m3", "layout": "ND",
        "shapes": {
            "x": [k], "weight": [k, n],
            "weight_scale": [pairs, n, 2],
            "pertoken_scale": [pairs, 2],
            "output": [n], "group_sizes": [3],
        },
        "attrs": {
            "matrix_k": k, "matrix_n": n, "scale_group": group,
            "group_size": group, "group_sizes": [1, 1, group],
            "scale_dtype": "fp8_e8m0",
            "pertoken_scale_dtype": "fp8_e8m0",
            "output_dtype": str(quant.get("output_format", "bf16")),
            "prequant_format": str(quant.get("prequant_format", "bf16")),
            "weight_format": "fp8_e4m3",
            "act_format": "fp8_e4m3", "linear_method": "mx_fp8",
            "implementation": "W8A8_MXFP8",
            "weight_transform": "transpose_0_1+maybe_trans_nz",
            "weight_layout": "FRACTAL_NZ", "activation_layout": "ND",
            "weight_storage_dtype": "fp8_e4m3",
            "weight_allocation_shape": [n, k],
            "weight_logical_shape": [k, n],
            "weight_device_shape": ([n // 32, k // 16, 16, 32]
                                    if n % 32 == 0 and k % 16 == 0 else None),
            "weight_scale_storage": "uint8",
            "weight_scale_dtype": "fp8_e8m0",
            "weight_scale_allocation_shape": [n, (k + group - 1) // group],
            "weight_scale_device_shape": [pairs, n, 2],
            "deployment_image_id": digest or "unresolved",
            "deployment_source_revision": selected or "unpinned",
            "source_version": (QBMM_SOURCES[selected]["version"]
                               if selected else None),
            "selected_host_variant": (QBMM_SOURCES[selected]
                                      ["selected_host_variant"]
                                      if selected else None),
            "source_lines": (QBMM_SOURCES[selected]["source_lines"]
                             if selected else None),
            "bias": bias, "offset": None,
        },
        "rank_width": 2,
    }
    if selected is None:
        return _call_spec(
            _api("torch", None), None, typed, (), (), name,
            unresolved=("deployment_source_unpinned",
                        "standalone_mxfp8_wrapper_unpinned",
                        "source_revision_has_no_authentic_quant_wrapper"))
    selected_source = QBMM_SOURCES[selected]
    wrapper = selected_source["wrapper"]
    kernel = selected_source["kernel"]
    unresolved = ["binary_selection_trace_only"]
    if selected == "r35":
        unresolved.append("kernel_source_runtime_binary_only")
    source = _source(
        str(wrapper["repository"]), str(wrapper["path"]),
        sha256=str(wrapper["sha256"]), symbol=str(wrapper["symbol"]),
        revision=str(wrapper["revision"]), state="task_artifact")
    return _call_spec(
        _api("torch", "torch_npu.npu_quant_matmul",
             version=selected_source["version"]),
        source, typed,
        (_kernel("QuantBatchMatmulV3", "cube",
                 path=kernel.get("path"), sha256=kernel.get("sha256"),
                 evidence=str(kernel.get("evidence") or "source_declared")),),
        (), name, execution_core="aic", unresolved=unresolved)


_I1_ENTRY = "torch_npu.npu_quant_lightning_indexer"
_I1_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d",
    "symbol": "A5DeviceAdaptor.indexer_select_post_process",
    "revision": "r64/8d422528",
    "state": "task_artifact",
}


_MOE_ENTRY = "torch_npu.npu_moe_gating_top_k"
_UNPERMUTE_ENTRY = "torch_npu.npu_moe_token_unpermute"
_MOE_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d",
    "revision": "r64/8d422528",
    "state": "task_artifact",
}


def _r64_source_meta(feature_metadata: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    meta = dict(feature_metadata or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    if source_revision_from_oci(
            str(meta.get("image") or meta.get("deployment_image") or ""),
            image_id) != "r64":
        return {}
    return meta


def _moe_source(lines: Sequence[int], symbol: str) -> dict:
    return _source("vllm-ascend", _MOE_SOURCE["path"],
                   sha256=_MOE_SOURCE["sha256"], symbol=symbol,
                   revision=_MOE_SOURCE["revision"], state="task_artifact")


def _f1_spec(theta: Any, annotations: Mapping[str, Any], *, kind: str,
              name: str, feature_metadata: Iterable[tuple[str, Any]] = ()
              ) -> _CallSpec | None:
    if kind != "F1" or name != "router_topk":
        return None
    meta = _r64_source_meta(feature_metadata)
    if not meta:
        return None
    ann = _annotations(annotations)
    experts = int(theta.find("E", ann.get("E", 0)) or 0)
    top_k = int(theta.find("k", ann.get("k", 0)) or 0)
    groups = int(theta.find("groups", ann.get("groups", 1)) or 1)
    group_topk = int(theta.find("group_topk", ann.get("group_topk", 0)) or 0)
    if experts < 1 or top_k < 1:
        return None
    typed = {
        "dtype": "float32", "layout": "ND",
        "shapes": {"x": [experts], "bias": [experts],
                   "topk_weights": [top_k], "topk_ids": [top_k],
                   "out": [experts]},
        "attrs": {
            "input_contract": ["x", "bias"], "output_contract":
                ["topk_weights", "topk_ids", "out"],
            "E": experts, "k": top_k, "groups": groups,
            "group_topk": group_topk, "k_group": max(1, group_topk),
            "group_count": groups, "group_select_mode": None,
            "renorm": 0, "norm_type": None, "routed_scaling_factor": None,
            "eps": None, "source_lines": [1222, 1251],
            "deployment_image_id": str(meta.get("image_id") or "unresolved"),
        },
        "rank_width": 2,
    }
    source = _moe_source([1222, 1251], "A5DeviceAdaptor.moe_gating_top_k")
    kernel = _kernel("MoeGatingTopK", "mix", path=_MOE_SOURCE["path"],
                     sha256=_MOE_SOURCE["sha256"], evidence="source_declared")
    return _call_spec(
        _api("torch", _MOE_ENTRY, version="r64/8d422528"), source, typed,
        (kernel,), (), name,
        execution_core="mix",
        unresolved=["aclnn_symbol_unresolved",
                    "routed_scaling_factor_from_caller",
                    "group_select_mode_from_caller"])


def _f2_finalize_spec(theta: Any, annotations: Mapping[str, Any], *, kind: str,
                      name: str, phase: str,
                      feature_metadata: Iterable[tuple[str, Any]] = ()
                      ) -> _CallSpec | None:
    if kind != "F2" or name != "router_finalize":
        return None
    meta = _r64_source_meta(feature_metadata)
    if not meta:
        return None
    ann = _annotations(annotations)
    hidden = int(theta.find("router_hidden", ann.get("router_hidden", 0)) or 0)
    top_k = int(theta.find("k", ann.get("k", 0)) or 0)
    if hidden < 1 or top_k < 1:
        return None
    typed = {
        "dtype": "bf16", "layout": "ND",
        "shapes": {"permuted_tokens": [hidden],
                   "probs": [top_k], "output": [hidden]},
        "attrs": {
            "input_contract": ["permuted_tokens", "sorted_indices", "probs"],
            "output_contract": ["output"], "weighted": True,
            "router_hidden": hidden, "k": top_k,
            "router_upstream": "combine",
            "router_source_operation": "MoeTokenUnpermute/MoeFinalizeRoutingV2",
            "router_selected": "post_combine_weighted_return",
            "source_lines": [1993, 1996],
            "deployment_image_id": str(meta.get("image_id") or "unresolved"),
        },
        "rank_width": 2,
    }
    source = _moe_source([1993, 1996],
                         "A5DeviceAdaptor.npu_moe_token_unpermute")
    kernel = _kernel("MoeFinalizeRoutingV2", "mix", path=_MOE_SOURCE["path"],
                     sha256=_MOE_SOURCE["sha256"], evidence="source_declared")
    return _call_spec(
        _api("torch", _UNPERMUTE_ENTRY, version="r64/8d422528"), source, typed,
        (kernel,), (), name, execution_core="mix",
        unresolved=["aclnn_symbol_unresolved",
                    "weighted_return_source_gate"])


def _f3_unweighted_spec(theta: Any, annotations: Mapping[str, Any], *, kind: str,
                        name: str, phase: str = "",
                        feature_metadata: Iterable[tuple[str, Any]] = ()
                        ) -> _CallSpec | None:
    if kind != "F3" or name != "router_unweighted_reorder":
        return None
    meta = _r64_source_meta(feature_metadata)
    if not meta:
        return None
    ann = _annotations(annotations)
    raw_received = theta.find("router_received_rows",
                              ann.get("router_received_rows", None))
    received = (None if raw_received is None else int(raw_received))
    hidden = int(theta.find("router_hidden", ann.get("router_hidden", 0)) or 0)
    if hidden < 1 or (received is not None and received < 1):
        return None
    typed = {
        "dtype": "bf16", "layout": "ND",
        "shapes": {"permuted_tokens": [hidden],
                   "output": [hidden]},
        "attrs": {
            "input_contract": ["permuted_tokens", "sorted_indices"],
            "output_contract": ["output"], "weighted": False,
            "router_received_rows": received, "router_hidden": hidden,
            "router_symbolic_input": received is None,
            "router_source_operation": "MoeTokenUnpermute/MoeFinalizeRoutingV2",
            "router_selected": "post_grouped_unweighted_return",
            "source_lines": [1993, 1996],
            "deployment_image_id": str(meta.get("image_id") or "unresolved"),
        },
        "rank_width": 2,
    }
    source = _moe_source([1993, 1996],
                         "A5DeviceAdaptor.npu_moe_token_unpermute")
    kernel = _kernel("MoeFinalizeRoutingV2", "mix", path=_MOE_SOURCE["path"],
                     sha256=_MOE_SOURCE["sha256"], evidence="source_declared")
    return _call_spec(
        _api("torch", _UNPERMUTE_ENTRY, version="r64/8d422528"), source, typed,
        (kernel,), (), name, execution_core="mix",
        unresolved=["aclnn_symbol_unresolved",
                    "unweighted_return_source_gate"])


def _indexer_spec(theta: Any, annotations: Mapping[str, Any], *,
                  kind: str, name: str,
                  feature_metadata: Iterable[tuple[str, Any]] = ()
                  ) -> _CallSpec | None:
    """R64 C8/qscale-present QuantLightningIndexer shared score+topk call."""
    if kind != "I1" or name not in {"index_score", "topk"}:
        return None
    meta = dict(feature_metadata or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    if source_revision_from_oci(
            str(meta.get("image") or meta.get("deployment_image") or ""),
            image_id) != "r64":
        return None
    ann = _annotations(annotations)
    declaration = declaration_from_theta(theta, meta)
    selection = select_indexer_source(declaration)
    if selection.get("status") != "c8_r64":
        return None
    candidate = int(theta.find("candidate_blocks", ann.get("candidate_blocks", 0)) or 0)
    if candidate != 2048:
        return None
    max_model_len = int(meta.get("max_model_len") or 0)
    capacity = ((max_model_len + 127) // 128) if max_model_len > 0 else None
    typed = {
        "dtype": "fp8_e4m3", "layout": "TND",
        "shapes": {
            "query": [32, 128], "key": [128, 1, 128],
            "weights": [32], "query_dequant_scale": [32],
            "key_dequant_scale": [128, 1], "output": [1, 2048],
        },
        "attrs": {
            "input_layout": "TND", "layout_query": "TND",
            "layout_key": "PA_BSND", "query_dtype": selection["q_dtype"],
            "key_dtype": selection["k_dtype"], "weights_dtype": "bf16",
            "query_dequant_scale_dtype": selection["query_scale_dtype"],
            "key_dequant_scale_dtype": "fp32",
            "output_dtype": "int32", "query_quant_mode": 0,
            "key_quant_mode": 0, "sparse_count": 2048,
            "sparse_mode": 3, "q_heads": 32, "head_dim": 128,
            "block_size": 128, "index_q_dtype": selection["q_dtype"],
            "index_k_dtype": selection["k_dtype"],
            "qscale_present": selection["query_scale_present"],
            "qscale_dtype": selection["query_scale_dtype"], "input_contract": [
                "query", "key", "weights", "query_dequant_scale",
                "key_dequant_scale", "actual_seq_lengths_query",
                "actual_seq_lengths_key", "block_table"],
            "block_table_capacity": capacity, "kv_pages": None,
            "source_lines": [1783, 1806],
            "deployment_image_id": image_digest(image_id) or "unresolved",
            "deployment_source_revision": "r64",
        },
        "rank_width": 3,
    }
    if capacity:
        typed["shapes"]["block_table"] = [capacity]
    source = _source("vllm-ascend", _I1_SOURCE["path"],
                     sha256=_I1_SOURCE["sha256"], symbol=_I1_SOURCE["symbol"],
                     revision=_I1_SOURCE["revision"], state="task_artifact")
    kernel = _kernel("QuantLightningIndexer", "mix",
                     path=_I1_SOURCE["path"], sha256=_I1_SOURCE["sha256"],
                     evidence="source_declared")
    return _call_spec(
        _api("torch", _I1_ENTRY, version="r64/8d422528"), source, typed,
        (kernel,), (), "I1:indexer:index_score", execution_core="mix",
        unresolved=["i1_kv_pages_not_declared"])


def _sfa_attention_spec(theta: Any, annotations: Mapping[str, Any], *,
                        name: str,
                        feature_metadata: Iterable[tuple[str, Any]] = ()
                        ) -> _CallSpec | None:
    """The selected R64 A5 G64 packed-FP8 sparse attention branch."""
    if name != "mla_attention":
        return None
    meta = dict(feature_metadata or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    revision = source_revision_from_oci(
        str(meta.get("image") or meta.get("deployment_image")
            or meta.get("oci_image") or ""), image_id)
    digest = image_digest(image_id)
    selected = ("r64" if revision == "r64"
                else "r35" if digest.startswith(R35_IMAGE_SHA) else None)
    if selected is None:
        return None
    branch = _SFA_SOURCES[selected]
    ann = _annotations(annotations)
    heads = dict(theta.heads)
    q_heads = int(heads.get("q") or ann.get("q") or 0)
    latent = int(ann.get("latent_width") or 0)
    rkv = int(ann.get("rkv") or 0)
    dr = int(heads.get("dr") or ann.get("dr") or 0)
    topk = int(theta.find("index_topk", ann.get("index_topk", 0)) or 0)
    if (q_heads, latent, rkv, dr, topk) != (64, 576, 512, 64, 2048):
        return None
    max_model_len = int(meta.get("max_model_len") or 0)
    block_table_capacity = ((max_model_len + 127) // 128) if max_model_len > 0 else None
    scale = theta.find("scale", ann.get("scale", None))
    try:
        scale = float(scale) if scale is not None else None
    except (TypeError, ValueError):
        scale = None
    typed = {
        "dtype": "bf16", "layout": "TND",
        "shapes": {
            "query": [64, 576],
            "key": [128, 1, 656],
            "value": [128, 1, 656],
            "sparse_indices": [1, 2048],
            "output": [64, 512],
        },
        "attrs": {
            "input_layout": "TND", "layout_query": "TND",
            "layout_kv": "PA_BSND", "query_dtype": "bf16",
            "key_dtype": "fp8_e4m3", "value_dtype": "fp8_e4m3",
            "output_dtype": "bf16", "quant_mode": 2,
            "key_quant_mode": 2, "value_quant_mode": 2,
            "sparse_mode": 3, "attention_mode": 2,
            "quant_scale_repo_mode": 1, "tile_size": 128,
            "rope_head_dim": 64, "block_size": 128,
            "sparse_block_size": 1, "index_topk": 2048,
            "q_heads": 64, "scale_value": scale,
            "optional_null_inputs": ["key_quant_scale",
                                     "value_quant_scale"],
            "input_aliases": [["key", "value"]],
            "source_lines": branch["lines"],
            "source_entry_prefix": ( _SFA_SOURCE_ENTRY_PREFIX
                                     if selected == "r64" else None),
            "input_contract": ["query", "key", "value", "sparse_indices",
                               "key_quant_scale", "value_quant_scale",
                               "block_table", "actual_seq_lengths_query",
                               "actual_seq_lengths_kv"],
            "block_table_capacity": block_table_capacity,
            "kv_pages": None,
            "capacity_status": "not_declared",
            "deployment_image_id": digest or "unresolved",
            "deployment_source_revision": selected,
        },
        "rank_width": 4,
    }
    if block_table_capacity:
        typed["shapes"]["block_table"] = [block_table_capacity]
    caller = branch["caller_sha256"]
    unresolved = ["binary_selection_trace_only"]
    if selected == "r64":
        unresolved.append("sfa_kernel_source_entry_prefix_only")
    elif caller is None:
        unresolved.append("sfa_kernel_caller_not_retained")
    if scale is None:
        unresolved.append("sfa_scale_value_not_declared")
    source_record = branch["source"]
    source = _source(
        str(source_record["repository"]), str(source_record["path"]),
        sha256=str(source_record["sha256"]),
        symbol=str(source_record["symbol"]),
        revision=str(source_record["revision"]), state="task_artifact")
    kernel = _kernel(
        "KvQuantSparseFlashAttention", "mix",
        evidence="runtime_caller" if caller else "source_declared",
        caller_sha256=caller,
        source_entry_prefix=(_SFA_SOURCE_ENTRY_PREFIX
                             if selected == "r64" else None))
    return _call_spec(
        _api("torch", _SFA_ENTRY, version=branch["version"]), source, typed,
        (kernel,), (), name, execution_core="mix", unresolved=unresolved)


def _attention_spec(theta: Any, annotations: Mapping[str, Any], *,
                    kind: str, name: str, phase: str = "",
                    feature_metadata: Iterable[tuple[str, Any]] = ()
                    ) -> _CallSpec | None:
    if name not in {"mla_attention", "attention"}:
        return None
    from mindie_sim.prediction import attention_prefill, paged_mla
    meta = dict(feature_metadata or ())
    digest = image_digest(str(meta.get("image_id") or meta.get("oci_image_digest")
                             or meta.get("deployment_image_id") or ""))
    r35 = digest.startswith(R35_IMAGE_SHA)
    if name == "attention":
        if not r35:
            return None
        source = _source("vllm-ascend", "vllm_ascend/attention/attention_v1.py",
                         sha256="bea4e410f797557e29cedbb495024147606e74b0e056d725194760ca69296eba",
                         symbol="torch_npu.npu_fused_infer_attention_score",
                         revision="a5-cann-r35-retained-source", state="task_artifact")
        typed = _typed_view(theta, annotations, family="attention", name=name)
        typed["attrs"].update({"input_layout": "TND", "sparse_mode": 3,
                               "source_tiling": attention_prefill.SOURCE_REFERENCES[1][2]})
        return _call_spec(
            _api("torch", "torch_npu.npu_fused_infer_attention_score"), source, typed,
            (_kernel("FusedInferAttentionScore", "mix", path=source["path"]),),
            (), name, unresolved=["aclnn_symbol_unresolved",
                                  "binary_identity_not_established"])
    sfa = _sfa_attention_spec(theta, annotations, name=name,
                              feature_metadata=feature_metadata)
    if sfa is not None:
        return sfa
    if not r35:
        return None
    typed = _typed_view(theta, annotations, family="attention", name=name)
    typed["attrs"].update({"input_layout": "TND", "sparse_mode": 3,
                           "causal": int(theta.find("causal", 1))})
    if phase == "prefill":
        source = _source(
            "vllm-ascend", "vllm_ascend/attention/attention_v1.py",
            sha256="bea4e410f797557e29cedbb495024147606e74b0e056d725194760ca69296eba",
            symbol="torch_npu.npu_fused_infer_attention_score",
            revision="a5-cann-r35-retained-source", state="task_artifact")
        typed["attrs"]["source_tiling"] = attention_prefill.SOURCE_REFERENCES[1][2]
    else:
        source = _source("cann/ops-transformer", paged_mla.SOURCE["kernel"],
                         sha256=paged_mla.SOURCE["sha256"]["kernel"],
                         symbol="FusedInferAttentionScore",
                         revision=attention_prefill.SOURCE_REFERENCES[1][1],
                         state="source")
    entry = "torch_npu.npu_fused_infer_attention_score"
    kernel = _kernel("FusedInferAttentionScore", "mix",
                     path=source["path"], sha256=source["sha256"])
    return _call_spec(_api("torch", entry), source, typed, (kernel,), (),
                      name, unresolved=["aclnn_symbol_unresolved",
                                        "binary_identity_not_established"])


def _collective_spec(theta: Any, annotations: Mapping[str, Any], *,
                     family: str, name: str) -> _CallSpec | None:
    if family not in {"collective", "transport"}:
        return None
    collective = str(annotations.get("collective")
                     or theta.find("collective", "") or "")
    entry = _HCCL_ENTRIES.get(collective)
    typed = _typed_view(theta, annotations, family=family, name=name)
    if entry is None:
        typed["attrs"]["hccl_collective"] = collective or "unresolved"
        return _call_spec(
            _api("hccl", None), None, typed, (), (), name,
            unresolved=["exact_collective_entry_unresolved",
                        "collective_algorithm_unresolved"])
    typed["attrs"]["hccl_collective"] = collective
    source = _source("hccl", _HCCL_SOURCE, symbol=entry.split(".")[-1],
                     revision="torch.distributed")
    return _call_spec(_api("hccl", entry), source, typed, (), (), name,
                      execution_core="hccl",
                      unresolved=["hccl_native_c_symbol_not_claimed",
                                  "source_sha256_unresolved"])


def _recurrent_spec(theta: Any, annotations: Mapping[str, Any], *,
                    kind: str, name: str) -> _CallSpec | None:
    if name == "gdn_gating":
        from mindie_sim.prediction import call_gdn
        source = _source("vllm-ascend", call_gdn.GATING_SOURCE["path"],
                         sha256=call_gdn.GATING_SOURCE["sha256"],
                         symbol="fused_gdn_gating_kernel", revision="r31",
                         state="task_artifact")
        typed = _typed_view(theta, annotations, family="recurrent", name=name)
        return _call_spec(
            _api("triton", "fused_gdn_gating_kernel"), source, typed,
            (_kernel("fused_gdn_gating_kernel", "vector",
                     path=source["path"], sha256=source["sha256"]),),
            (), name, unresolved=["aclnn_symbol_unresolved"])
    if name in {"q_l2norm", "k_l2norm"}:
        typed = _typed_view(theta, annotations, family="elementwise", name=name)
        return _call_spec(
            _api("triton", "l2norm_fwd_kernel2"), None, typed,
            (_kernel("l2norm_fwd_kernel2", "vector"),), (), name,
            unresolved=["aclnn_symbol_unresolved", "source_reference_unresolved"])
    if name == "conv" and kind == "A3":
        from mindie_sim.prediction import call_gdn
        from mindie_sim.prediction.operators import native_causal_conv_spec
        source = _source("vllm-ascend", call_gdn.WRAPPER_SOURCE["archive"],
                         sha256=call_gdn.WRAPPER_SOURCE["sha256"],
                         symbol="fla_npu.ops.ascendc.causal_conv1d_fn",
                         revision=native_causal_conv_spec.SOURCE_REVISION,
                         state="task_artifact")
        typed = _typed_view(theta, annotations, family="recurrent", name=name)
        typed["attrs"].update({"input_layout": "TND", "activation": "silu",
                               "state_slots": int(theta.find("history_rows", 0) or 0)})
        return _call_spec(
            _api("torch", "fla_npu.ops.ascendc.causal_conv1d_fn"), source, typed,
            (_kernel("CausalConv1d", "vector", path=source["path"],
                     sha256=source["sha256"], evidence="source_declared"),),
            (), name, unresolved=["aclnn_symbol_unresolved"])
    if name == "gated_delta":
        from mindie_sim.prediction import gdn_recurrent
        source = _source("Ascend/ops-transformer", gdn_recurrent.SOURCE["path"],
                         sha256=gdn_recurrent.SOURCE["sha256"],
                         symbol="RecurrentGatedDeltaRule", revision="arch35",
                         state="source")
        typed = _typed_view(theta, annotations, family="recurrent", name=name)
        typed["attrs"].update({"input_dtype": "bf16", "state_dtype": "fp32",
                               "layout": "BSND"})
        return _call_spec(
            _api("cann_kernel", "RecurrentGatedDeltaRule"), source, typed,
            (_kernel("RecurrentGatedDeltaRule", "mix", path=source["path"],
                     sha256=source["sha256"]),),
            (), name, unresolved=["aclnn_symbol_not_established",
                                  "exact_acl_nn_entry_not_selected"])
    return None


def _gdn_chunk_spec(compiled: Any, phase: str = "") -> _CallSpec | None:
    if getattr(compiled, "kind", "") != "A3":
        return None
    if phase != "prefill" and _variant(compiled) != "chunked_prefill":
        return None
    try:
        from mindie_sim.prediction import call_gdn
    except ImportError:  # pragma: no cover - normal environment
        return None
    key_heads, value_heads, dk, dv, _chunk = gdn_dims(compiled)
    source = _source("vllm-ascend", call_gdn.PREFILL_SOURCE["archive"],
                     sha256=call_gdn.PREFILL_SOURCE["sha256"],
                     symbol="fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
                     revision="c3d00a196905", state="task_artifact")
    typed = {
        "dtype": "bf16", "layout": "BSND", "phase": "prefill",
        "shapes": {"q": [key_heads, dk], "k": [key_heads, dk],
                   "v": [value_heads, dv], "g": [value_heads],
                   "beta": [value_heads], "state": [value_heads, dv, dk]},
        "attrs": {"chunk_size": 64, "use_exp2": True,
                  "use_qk_l2norm_in_kernel": True, "disable_recompute": True,
                  "state_v_first": True, "output_final_state": True,
                  "scale": dk ** -0.5},
        "rank_width": 3,
    }
    kernels = (
        _kernel("Cast", "copy", evidence="measured"),
        _kernel("Transpose", "copy", evidence="measured", count=3),
        _kernel("Prepare32", "vector", evidence="measured"),
        _kernel("H24", "cube", evidence="measured"),
        _kernel("O32", "cube", evidence="measured"),
    )
    return _call_spec(_api("aclnn", "aclnnChunkGatedDeltaRuleFwd",
                           aclnn_symbol="aclnnChunkGatedDeltaRuleFwd"),
                      source, typed, kernels, ("prepare", "gated_delta"),
                      f"{getattr(compiled, 'kind', '')}:recurrent:gated_delta")


def _spec_for(compiled: Any, *, phase: str = "") -> _CallSpec | None:
    operator = compiled.operator
    theta, family, name = operator.theta, operator.family, operator.name
    annotations = _annotations(compiled.program.annotations)
    if getattr(compiled, "kind", "") == "A3" and name in {"prepare", "gated_delta"}:
        spec = _gdn_chunk_spec(compiled, phase)
        if spec is not None:
            return spec
    if getattr(compiled, "kind", "") == "A3":
        for ssm in (_ssm_gather_spec(compiled, phase=phase),
                    _ssm_scatter_spec(compiled, phase=phase)):
            if ssm is not None:
                return ssm
    for candidate in (
        lambda: _dynamic_mx_spec(theta, annotations, name=name),
        lambda: _op_perm2_spec(compiled, phase=phase),
        lambda: _op_rope_siso_spec(compiled, phase=phase),
        lambda: _lnim_spec(compiled, phase=phase),
        lambda: _gated_norm_spec(compiled, phase=phase),
        lambda: _rms_plain_spec(compiled, phase=phase),
        lambda: _muls_add_spec(compiled, phase=phase),
        lambda: _add_rms_spec(compiled, phase=phase),
        lambda: _add_rms_bias_spec(compiled, phase=phase),
        lambda: _qkvmr_spec(compiled, phase=phase),
        lambda: _swiglu_spec(compiled, phase=phase),
        lambda: _opdquant_spec(compiled, phase=phase),
        lambda: _sig_spec(compiled, phase=phase),
        lambda: _scatpa_spec(compiled, phase=phase),
        lambda: _inzero_spec(compiled, phase=phase),
        lambda: _gmm2_spec(compiled, phase=phase),
        lambda: _op_hist_spec(compiled, phase=phase),
        lambda: _moe_init3_spec(compiled, phase=phase),
        lambda: _gmm1_spec(compiled, phase=phase),
        lambda: _qbmm31_spec(compiled, phase=phase),
        lambda: _l2n_spec(compiled, phase=phase),
        lambda: _tbg_spec(compiled, phase=phase),
        lambda: _incopy_spec(compiled, phase=phase),
        lambda: _mat_spec(compiled, phase=phase),
        lambda: _cat_spec(compiled, phase=phase),
        lambda: _index_spec(compiled, phase=phase),
        lambda: _f1_spec(
            theta, annotations, kind=compiled.kind, name=name,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _f2_finalize_spec(
            theta, annotations, kind=compiled.kind, name=name, phase=phase,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _f3_unweighted_spec(
            theta, annotations, kind=compiled.kind, name=name, phase=phase,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _qbm_spec(
            theta, annotations, name=name, family=family,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _indexer_spec(
            theta, annotations, kind=compiled.kind, name=name,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _attention_spec(
            theta, annotations, kind=compiled.kind, name=name, phase=phase,
            feature_metadata=getattr(compiled, "feature_metadata", ())),
        lambda: _collective_spec(theta, annotations, family=family, name=name),
        lambda: _recurrent_spec(theta, annotations, kind=compiled.kind, name=name),
    ):
        spec = candidate()
        if spec is not None:
            return spec
    return None


def _surface(compiled: Any, spec: _CallSpec, *, source_boundary: str) -> _Surface:
    operator = compiled.operator
    typed = spec.typed
    kind = str(getattr(compiled, "kind", "") or "")
    return _Surface(
        op_id=f"{kind}:{operator.family}:{operator.name}",
        family=operator.family, recipe=operator.template.role,
        phase=str(typed.get("phase") or ""), source_boundary=source_boundary,
        dtype=typed.get("dtype"), layout=typed.get("layout"),
        local_shape=tuple(typed.get("shapes", {}).get("local", ())),
        padded_shape=tuple(typed.get("shapes", {}).get("padded", ())),
        launch_shape=tuple(typed.get("shapes", {}).get("launch", ())),
        metadata=tuple(sorted(typed.get("attrs", {}).items())),
    )


def bind_compiled_operator(compiled: Any, *, phase: str = "") -> NativeCallBinding | None:
    """Bind one compiled BoundOperator to the shared native-call identity."""
    spec = _spec_for(compiled, phase=phase)
    if spec is None:
        return None
    boundary = ("call" if spec.api.get("namespace") == "hccl"
                or len(spec.members) > 1 else "kernel")
    surface = _surface(compiled, spec, source_boundary=boundary)
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None,
    )
    binding = bind_native_call(declared)
    if binding is not None and spec.owner != compiled.operator.name:
        binding = replace(binding, timing_owner={"op_id": spec.owner, "scope": "call"})
    return binding


def _dynamic_sfa(binding: NativeCallBinding, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R64 packed SFA call."""
    if (binding.api or {}).get("entry") != _SFA_ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    selected = str(attrs.get("deployment_source_revision") or "")
    branch = _SFA_SOURCES.get(selected)
    if branch is None:
        return None
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1:
        return None
    q_heads = int(attrs.get("q_heads") or 64)
    q_width = 576
    out_width = 512
    key_width = 656
    pages = None
    block = 128
    topk = 2048
    block_table_cols = int(attrs.get("block_table_capacity") or 0) or None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    batch = max(1, len(requests))
    block_table_shape = ([batch, block_table_cols] if block_table_cols
                         else [batch, None])
    actual_shapes = {
        "query": [m, q_heads, q_width],
        "key": [None, block, 1, key_width],
        "value": [None, block, 1, key_width],
        "sparse_indices": [m, 1, topk],
        "key_quant_scale": None,
        "value_quant_scale": None,
        "block_table": block_table_shape,
        "actual_seq_lengths_query": [batch],
        "actual_seq_lengths_kv": [batch],
        "output": [m, q_heads, out_width],
    }
    scale = attrs.get("scale_value")
    actual_attrs = {
        **attrs, "m": m, "pages": pages, "batch": batch,
        "sparse_indices_shape": [m, 1, topk],
        "block_table_shape": block_table_shape,
    }
    actual = {"dtype": "bf16", "layout": "TND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    null = {"shape": None, "dtype": None, "layout": None,
            "role": "input_optional", "present": False, "pointer": None}
    descriptors = {
        "query": _sfa_descriptor([m, q_heads, q_width], "bf16", "TND", "input"),
        "key": _sfa_descriptor([None, block, 1, key_width], "fp8_e4m3",
                               "PA_BSND", "input"),
        "value": _sfa_descriptor([None, block, 1, key_width], "fp8_e4m3",
                                 "PA_BSND", "input"),
        "sparse_indices": _sfa_descriptor([m, 1, topk], "int32", "ND", "input"),
        "key_quant_scale": dict(null),
        "value_quant_scale": dict(null),
        "block_table": _sfa_descriptor(block_table_shape, "int32", "ND", "input"),
        "actual_seq_lengths_query": _sfa_descriptor([batch], "int32", "ND", "input"),
        "actual_seq_lengths_kv": _sfa_descriptor([batch], "int32", "ND", "input"),
        "output": _sfa_descriptor([m, q_heads, out_width], "bf16", "TND", "output"),
    }
    descriptors["key"]["alias_group"] = "kv"
    descriptors["value"]["alias_group"] = "kv"
    arguments = {
        "query": "query", "key": "key", "value": "value",
        "sparse_indices": "sparse_indices", "scale_value": scale,
        "sparse_block_size": 1, "block_table": "block_table",
        "actual_seq_lengths_query": "actual_seq_lengths_query",
        "actual_seq_lengths_kv": "actual_seq_lengths_kv",
        "layout_query": "TND", "layout_kv": "PA_BSND",
        "sparse_mode": 3, "attention_mode": 2,
        "quant_scale_repo_mode": 1, "tile_size": 128,
        "key_quant_mode": 2, "value_quant_mode": 2,
        "rope_head_dim": 64,
    }
    prepare = {
        "schema": "mindie-sim/sfa-native-prepare/1",
        "call": _SFA_ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": dict(binding.source or {}),
        "source_lines": list(attrs.get("source_lines") or branch["lines"]),
        "kernel": (dict(binding.kernels[0]) if binding.kernels else {
            "name": "KvQuantSparseFlashAttention", "role": "mix",
            "evidence": "source_declared",
        }),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": [
            "query", "key", "value", "sparse_indices", "scale_value",
            "sparse_block_size", "block_table", "actual_seq_lengths_query",
            "actual_seq_lengths_kv", "layout_query", "layout_kv",
            "sparse_mode", "attention_mode", "quant_scale_repo_mode",
            "tile_size", "key_quant_mode", "value_quant_mode", "rope_head_dim",
        ],
        "optional_slots": {
            "key_quant_scale": "not_present",
            "value_quant_scale": "not_present",
            "sparse_indices": "present",
            "block_table": "present",
            "return_lse": "false",
        },
        "aliases": {"key": "value", "basis": "source key=value=kv",
                    "storage_shared": True},
        "complete": False,
        "partial": True,
        "partial_reasons": (["kv_pages_not_declared"]
                            if block_table_cols else
                            ["kv_pages_not_declared",
                             "block_table_capacity_not_declared"]),
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": selected,
            "selection": "source_revision_from_oci + exact SFA op/layout/types",
            "source_sha256": (binding.source or {}).get("sha256"),
            "caller_sha256": ((binding.kernels[0].get("caller_sha256")
                               if binding.kernels else None)),
        },
        "match": {
            "native_type": "KvQuantSparseFlashAttention",
            "entry": _SFA_ENTRY,
            "call_id": binding.call_id,
            "deployment_image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "source_revision": selected,
            "source_version": branch["version"],
            "source_sha256": (binding.source or {}).get("sha256"),
            "source_lines": list(attrs.get("source_lines") or branch["lines"]),
            "caller_sha256": ((binding.kernels[0].get("caller_sha256")
                               if binding.kernels else None)),
            "source_entry_prefix": ((binding.kernels[0].get("source_entry_prefix")
                                     if binding.kernels else None)),
            "query_dtype": "bf16", "key_dtype": "fp8_e4m3",
            "value_dtype": "fp8_e4m3", "output_dtype": "bf16",
            "layout_query": "TND", "layout_kv": "PA_BSND",
            "quant_mode": 2, "sparse_mode": 3, "attention_mode": 2,
            "quant_scale_repo_mode": 1, "tile_size": 128,
            "rope_head_dim": 64, "block_size": 128,
            "q_heads": q_heads, "m": m, "pages": pages,
            "batch": batch, "block_table_capacity": block_table_cols,
            "complete": False, "key_value_alias": True,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_nn_symbol_not_exported_by_python_wrapper",
            "owner_kernel": "KvQuantSparseFlashAttention",
            "source_entry_prefix": _SFA_SOURCE_ENTRY_PREFIX,
        },
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {"dtype": validated.get("dtype"),
                             "layout": validated.get("layout"),
                             "shapes": validated.get("shapes"),
                             "attrs": validated.get("attrs")}}


def _dynamic_indexer(binding: NativeCallBinding, geometry: Any, *,
                     validate: Any) -> dict | None:
    if (binding.api or {}).get("entry") != _I1_ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1:
        return None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    batch = max(1, len(requests))
    capacity = int(attrs.get("block_table_capacity") or 0) or None
    table_shape = [batch, capacity] if capacity else [batch, None]
    shapes = {
        "query": [m, 32, 128], "key": [None, 128, 1, 128],
        "weights": [m, 32], "query_dequant_scale": [m, 32],
        "key_dequant_scale": [None, 128, 1],
        "block_table": table_shape,
        "actual_seq_lengths_query": [batch],
        "actual_seq_lengths_key": [batch],
        "output": [m, 1, 2048],
    }
    actual_attrs = {**attrs, "m": m, "batch": batch,
                    "block_table_shape": table_shape}
    actual = {"dtype": "fp8_e4m3", "layout": "TND",
              "shapes": shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "query": _sfa_descriptor([m, 32, 128], "fp8_e4m3", "TND", "input"),
        "key": _sfa_descriptor([None, 128, 1, 128], "fp8_e4m3",
                               "PA_BSND", "input"),
        "weights": _sfa_descriptor([m, 32], "bf16", "ND", "input"),
        "query_dequant_scale": _sfa_descriptor([m, 32], "fp32", "ND", "input"),
        "key_dequant_scale": _sfa_descriptor([None, 128, 1], "fp32",
                                             "ND", "input"),
        "actual_seq_lengths_query": _sfa_descriptor([batch], "int32", "ND", "input"),
        "actual_seq_lengths_key": _sfa_descriptor([batch], "int32", "ND", "input"),
        "block_table": _sfa_descriptor(table_shape, "int32", "ND", "input"),
        "output": _sfa_descriptor([m, 1, 2048], "int32", "ND", "output"),
    }
    arguments = {
        "query": "query", "key": "key", "weights": "weights",
        "query_dequant_scale": "query_dequant_scale",
        "key_dequant_scale": "key_dequant_scale",
        "actual_seq_lengths_query": "actual_seq_lengths_query",
        "actual_seq_lengths_key": "actual_seq_lengths_key",
        "block_table": "block_table",
        "query_quant_mode": 0, "key_quant_mode": 0,
        "layout_query": "TND", "layout_key": "PA_BSND",
        "sparse_count": 2048, "sparse_mode": 3,
        "pre_tokens": 9223372036854775807,
        "next_tokens": 9223372036854775807,
    }
    prepare = {
        "schema": "mindie-sim/indexer-native-prepare/1",
        "call": _I1_ENTRY, "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": dict(binding.source or {}), "source_lines": [1783, 1806],
        "kernel": (dict(binding.kernels[0]) if binding.kernels else None),
        "stage": "interface_prepared", "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": list(arguments),
        "optional_slots": {"block_table": "present", "kv_pages": "not_declared"},
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors, "arrays": descriptors,
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": "r64/8d422528",
            "source_sha256": (binding.source or {}).get("sha256"),
        },
        "match": {
            "native_type": "QuantLightningIndexer", "entry": _I1_ENTRY,
            "call_id": binding.call_id,
            "deployment_image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "source_revision": "r64/8d422528",
            "source_sha256": (binding.source or {}).get("sha256"),
            "source_lines": [1783, 1806],
            "query_dtype": "fp8_e4m3", "key_dtype": "fp8_e4m3",
            "weights_dtype": "bf16", "query_scale_dtype": "fp32",
            "key_scale_dtype": "fp32", "output_dtype": "int32",
            "layout_query": "TND", "layout_key": "PA_BSND",
            "query_quant_mode": 0, "key_quant_mode": 0,
            "sparse_count": 2048, "sparse_mode": 3,
            "m": m, "batch": batch, "block_table_capacity": capacity,
            "kv_pages": None, "complete": False,
        },
        "complete": False, "partial": True,
        "partial_reasons": ["kv_pages_not_declared"],
        "binary_selection": {"status": "uncertain",
                             "reason": "acl_nn_symbol_not_exported_by_python_wrapper",
                             "owner_kernel": "QuantLightningIndexer"},
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {"dtype": validated.get("dtype"),
                             "layout": validated.get("layout"),
                             "shapes": validated.get("shapes"),
                             "attrs": validated.get("attrs")}}


def _dynamic_moe(binding: NativeCallBinding, geometry: Any, *,
                 validate: Any, env: Mapping[str, Any] | None = None) -> dict | None:
    entry = (binding.api or {}).get("entry")
    attrs = dict(binding.typed.get("attrs") or {})
    if entry == _MOE_ENTRY:
        m = int(getattr(geometry, "effective_tokens", 0) or 0)
        e = int(attrs.get("E") or 0)
        k = int(attrs.get("k") or 0)
        if m < 1 or e < 1 or k < 1:
            return None
        shapes = {"x": [m, e], "bias": [e],
                  "topk_weights": [m, k], "topk_ids": [m, k],
                  "out": [m, e]}
        actual = {"dtype": "float32", "layout": "ND", "shapes": shapes,
                  "attrs": {**attrs, "m": m}}
        validated = validate(binding, actual)
        descriptors = {
            "x": _sfa_descriptor([m, e], "float32", "ND", "input"),
            "bias": _sfa_descriptor([e], "float32", "ND", "input"),
            "topk_weights": _sfa_descriptor([m, k], "float32", "ND", "output"),
            "topk_ids": _sfa_descriptor([m, k], "int32", "ND", "output"),
            "out": _sfa_descriptor([m, e], "float32", "ND", "output"),
        }
        arguments = {
            "x": "x", "k": k, "bias": "bias",
            "k_group": int(attrs.get("k_group") or 1),
            "group_count": int(attrs.get("group_count") or 1),
            "group_select_mode": attrs.get("group_select_mode") or "caller_required",
            "renorm": 0,
            "norm_type": attrs.get("norm_type") or "caller_required",
            "routed_scaling_factor": (attrs.get("routed_scaling_factor")
                                      or "caller_required"),
            "eps": attrs.get("eps") or "caller_required",
        }
        prepare = {
            "schema": "mindie-sim/moe-native-prepare/1",
            "call": _MOE_ENTRY, "call_id": binding.call_id,
            "signature_hash": binding.typed.get("signature_hash"),
            "source": dict(binding.source or {}), "source_lines": [1222, 1251],
            "kernel": (dict(binding.kernels[0]) if binding.kernels else None),
            "stage": "interface_prepared", "runtime_executed": False,
            "allocation": "not_performed", "arguments": arguments,
            "argument_order": list(arguments),
            "shapes": dict(validated.get("shapes") or {}),
            "descriptors": descriptors, "arrays": descriptors,
            "match": {"native_type": "MoeGatingTopK", "entry": _MOE_ENTRY,
                      "call_id": binding.call_id, "m": m,
                      "source_sha256": (binding.source or {}).get("sha256"),
                      "source_lines": [1222, 1251]},
            "complete": False, "partial": True,
            "partial_reasons": ["caller_norm_type", "caller_scale_eps",
                                "caller_group_select_mode"],
            "unresolved": ["routed_scaling_factor_from_caller",
                           "group_select_mode_from_caller",
                           "norm_type_from_caller", "eps_from_caller"],
            "non_claims": ["no_runtime_allocation", "no_device_launch"],
        }
        return {"actual": validated, "prepare": prepare,
                "prepare_args": {"dtype": "float32", "layout": "ND",
                                 "shapes": shapes, "attrs": actual["attrs"]}}
    if entry != _UNPERMUTE_ENTRY:
        return None
    if env is not None and int(env.get("MOE_COMM_SELECTED_ALLTOALL", 0) or 0) != 1:
        return None
    hidden = int(attrs.get("router_hidden") or 0)
    if hidden < 1:
        return None
    if attrs.get("weighted") is True:
        m = int(getattr(geometry, "effective_tokens", 0) or 0)
        k = int(attrs.get("k") or 0)
        if m < 1 or k < 1:
            return None
        shapes = {"permuted_tokens": [m * k, hidden],
                  "sorted_indices": [m * k], "probs": [m, k],
                  "output": [m, hidden]}
        arguments = {"permuted_tokens": "permuted_tokens",
                     "sorted_indices": "sorted_indices", "probs": "probs"}
        native_type = "MoeFinalizeRoutingV2Weighted"
        weighted = True
    else:
        raw_r = attrs.get("router_received_rows")
        r = None if raw_r is None else int(raw_r)
        if r is not None and r < 1:
            return None
        symbolic = r is None
        shapes = {"permuted_tokens": [None if symbolic else r, hidden],
                  "sorted_indices": [None if symbolic else r],
                  "output": [None if symbolic else r, hidden]}
        arguments = {"permuted_tokens": "permuted_tokens",
                     "sorted_indices": "sorted_indices", "probs": None}
        native_type = "MoeFinalizeRoutingV2Unweighted"
        weighted = False
    actual = {"dtype": "bf16", "layout": "ND", "shapes": shapes,
              "attrs": {**attrs, "weighted": weighted}}
    validated = validate(binding, actual)
    descriptors = {
        "permuted_tokens": _sfa_descriptor(shapes["permuted_tokens"],
                                           "bf16", "ND", "input"),
        "sorted_indices": _sfa_descriptor(shapes["sorted_indices"],
                                          "int32", "ND", "input"),
        "output": _sfa_descriptor(shapes["output"], "bf16", "ND", "output"),
    }
    if weighted:
        descriptors["probs"] = _sfa_descriptor(shapes["probs"], "bf16", "ND", "input")
    prepare = {
        "schema": "mindie-sim/moe-native-prepare/1",
        "call": _UNPERMUTE_ENTRY, "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": dict(binding.source or {}), "source_lines": [1993, 1996],
        "kernel": (dict(binding.kernels[0]) if binding.kernels else None),
        "stage": "interface_prepared", "runtime_executed": False,
        "allocation": "not_performed", "arguments": arguments,
        "argument_order": list(arguments),
        "shapes": dict(shapes),
        "descriptors": descriptors, "arrays": descriptors,
        "match": {"native_type": native_type, "entry": _UNPERMUTE_ENTRY,
                  "call_id": binding.call_id, "weighted": weighted,
                  "router_received_rows": None if weighted else r,
                  "symbolic_input": bool(not weighted and symbolic),
                  "source_sha256": (binding.source or {}).get("sha256"),
                  "source_lines": [1993, 1996]},
        "optional_slots": {"probs": "present" if weighted else "present_none"},
        "complete": bool(weighted or not symbolic),
        "partial": bool(not weighted and symbolic),
        "partial_reasons": ([] if weighted or not symbolic
                            else ["router_received_rows_unresolved"]),
        "unresolved": (["aclnn_symbol_unresolved"]
                       if weighted or not symbolic else
                       ["aclnn_symbol_unresolved", "router_received_rows_unresolved"]),
        "non_claims": ["no_runtime_allocation", "no_device_launch"],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {"dtype": "bf16", "layout": "ND",
                             "shapes": shapes, "attrs": actual["attrs"]}}


def _o0_preselection_binding(binding: Any, geometry: Any) -> Any:
    """Attach the real O0 pre-selection activation rows to the final norm.

    O0 LM-head selection is a later copy/gather. The final RMSNorm consumes
    the full local normal rows; dense-SP static decode uses the source role's
    shard lane. No request, selected LM-head batch or graph bucket is rewritten.
    """
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != "torch.ops._C_ascend.npu_add_rms_norm_bias":
        return binding
    typed = dict(getattr(binding, "typed", {}) or {})
    attrs = dict(typed.get("attrs") or {})
    if str(attrs.get("native_norm_scope") or "") != "o0_final":
        return binding
    try:
        rows_view = geometry.operator_layout(
            "O0", role="norm", name="final_norm", family="elementwise")
        rows = int(getattr(rows_view, "effective_tokens", 0) or 0)
        lanes = dict(geometry.token_lanes())
    except Exception:
        return binding
    if rows < 1:
        return binding
    attrs.update({
        "final_rows": rows,
        "final_rows_source": "workload_preselection_local_activation_rows",
        "selected_lm_head_rows": int(lanes.get("head_tokens") or 0),
        "rank_local_shard_rows": int(lanes.get("shard_tokens") or 0),
    })
    typed["attrs"] = attrs
    return replace(binding, typed=typed)


def _actual_typed(binding: NativeCallBinding, compiled: Any, geometry: Any,
                  env: Mapping[str, Any]) -> dict[str, Any]:
    binding = _o0_preselection_binding(binding, geometry)
    dynamic = (dynamic_gdn(binding, compiled, geometry,
                           validate=validate_native_occurrence)
               or dynamic_qbmm(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _dynamic_sfa(binding, geometry,
                               validate=validate_native_occurrence)
               or _dynamic_indexer(binding, geometry,
                                   validate=validate_native_occurrence)
               or _op_perm2_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _op_hist_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _moe_init3_dynamic(binding, compiled, geometry,
                                     validate=validate_native_occurrence)
               or _gated_norm_dynamic(binding, compiled, geometry,
                                      validate=validate_native_occurrence)
               or _rms_plain_dynamic(binding, compiled, geometry,
                                     validate=validate_native_occurrence)
               or _muls_add_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _add_rms_dynamic(binding, compiled, geometry,
                                   validate=validate_native_occurrence)
               or _add_rms_bias_dynamic(binding, compiled, geometry,
                                        validate=validate_native_occurrence)
               or _op_rope_siso_dynamic(binding, compiled, geometry,
                                        validate=validate_native_occurrence)
               or _lnim_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _ssm_gather_dynamic(binding, compiled, geometry,
                                      validate=validate_native_occurrence)
               or _ssm_scatter_dynamic(binding, compiled, geometry,
                                       validate=validate_native_occurrence)
               or _qkvmr_dynamic(binding, compiled, geometry,
                                 validate=validate_native_occurrence)
               or _swiglu_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _opdquant_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _sig_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _scatpa_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _inzero_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _gmm2_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _gmm1_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _qbmm31_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _l2n_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _tbg_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _incopy_dynamic(binding, compiled, geometry,
                                   validate=validate_native_occurrence)
               or _mat_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _cat_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _index_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _dynamic_moe(binding, geometry, env=env,
                               validate=validate_native_occurrence))
    if dynamic is not None:
        return dynamic["actual"]
    if (binding.api or {}).get("entry") in {_MOE_ENTRY, _UNPERMUTE_ENTRY}:
        return None
    typed = {key: value for key, value in binding.typed.items()}
    shapes = {key: list(value) for key, value in (typed.get("shapes") or {}).items()}
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    attrs = dict(typed.get("attrs") or {})
    attrs["tokens"] = tokens
    if "weight" in shapes:
        k, n = shapes["weight"]
        shapes["input"] = [tokens, k]
        shapes["output"] = [tokens, n]
    if "q" in shapes and isinstance(attrs.get("heads"), dict):
        heads = attrs["heads"]
        q_heads = int(heads.get("q", shapes["q"][0]))
        dk = (int(heads.get("dn", 0)) + int(heads.get("dr", 0))
              or int(heads.get("dk", 0)))
        dv = int(heads.get("dv", dk) or dk)
        shapes["q"] = [tokens, q_heads, dk]
        shapes["output"] = [tokens, q_heads, dv]
        if "kv_bytes" in attrs:
            shapes["kv_cache_row"] = [int(float(attrs["kv_bytes"]))]
    if "payload" in shapes:
        payload = env.get("collective_bytes") or attrs.get("hccl_payload_bytes",
                                                           shapes["payload"][0])
        shapes["payload"] = [int(float(payload))]
    actual = {"dtype": typed.get("dtype"), "layout": typed.get("layout"),
              "shapes": shapes, "attrs": attrs}
    return validate_native_occurrence(binding, actual)


def validate_native_occurrence(binding: NativeCallBinding,
                               arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Consumer negative gate: wrong dtype/layout/shape/attr is another call."""
    typed = binding.typed
    canonical: dict[str, list] = {}
    for name, shape in (arguments.get("shapes") or {}).items():
        declared = (typed.get("shapes") or {}).get(name)
        if declared is None:
            canonical[name] = None if shape is None else list(shape)
            continue
        if shape is None:
            raise ModelingError(f"native_call_signature_shape_mismatch:{name}")
        declared, actual = tuple(declared), tuple(shape)
        # The selected static signature has no batch lane; the occurrence
        # prepends the evaluated token/batch lanes.  A mismatched declared
        # suffix is still a different invocation.
        if declared != actual and (len(actual) < len(declared)
                                   or actual[-len(declared):] != declared):
            raise ModelingError(f"native_call_signature_shape_mismatch:{name}")
        canonical[name] = list(declared)
    validate_invocation(binding, {**arguments, "shapes": canonical})
    return {**arguments, "shapes": {
        key: None if value is None else list(value)
        for key, value in (arguments.get("shapes") or {}).items()}}


def _priced(cost: Any) -> dict[str, Any]:
    return {"duration_ms": str(getattr(cost, "duration_ms", "")),
            "dominant_pipe": getattr(cost, "dominant_pipe", ""),
            "work": {key: str(value) for key, value in
                     (getattr(cost, "work", ()) or ())},
            "charged": "once_at_timing_owner"}


def _binding_bound(binding: NativeCallBinding) -> bool:
    api = binding.api or {}
    source = binding.source or {}
    if not (api.get("entry") or api.get("aclnn_symbol")):
        return False
    if not source.get("path") or not (binding.typed or {}).get("signature_hash"):
        return False
    if not source.get("sha256") and api.get("namespace") != "hccl":
        return False
    return not set(binding.unresolved) & {"exact_native_entry_unresolved",
                                           "exact_native_entry_not_selected",
                                           "attributable_source_reference_unresolved"}


def _occurrence(binding: NativeCallBinding, compiled: Any, cost: Any,
                geometry: Any, env: Mapping[str, Any], typed: dict) -> dict:
    binding = _o0_preselection_binding(binding, geometry)
    kind = str(getattr(compiled, "kind", "") or "")
    phase = str(getattr(geometry, "phase_name", "") or "")
    owner = (binding.timing_owner or {}).get("op_id")
    unresolved = [reason for reason in binding.unresolved
                  if not (reason == "aclnn_symbol_unresolved"
                          and (binding.api or {}).get("aclnn_symbol"))]
    unresolved.append("execution_rank_not_selected")
    dynamic = (dynamic_gdn(binding, compiled, geometry,
                           validate=validate_native_occurrence)
               or dynamic_qbmm(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _dynamic_sfa(binding, geometry,
                               validate=validate_native_occurrence)
               or _dynamic_indexer(binding, geometry,
                                   validate=validate_native_occurrence)
               or _op_perm2_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _op_hist_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _moe_init3_dynamic(binding, compiled, geometry,
                                     validate=validate_native_occurrence)
               or _gated_norm_dynamic(binding, compiled, geometry,
                                      validate=validate_native_occurrence)
               or _rms_plain_dynamic(binding, compiled, geometry,
                                     validate=validate_native_occurrence)
               or _muls_add_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _add_rms_dynamic(binding, compiled, geometry,
                                   validate=validate_native_occurrence)
               or _add_rms_bias_dynamic(binding, compiled, geometry,
                                        validate=validate_native_occurrence)
               or _op_rope_siso_dynamic(binding, compiled, geometry,
                                        validate=validate_native_occurrence)
               or _lnim_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _ssm_gather_dynamic(binding, compiled, geometry,
                                      validate=validate_native_occurrence)
               or _ssm_scatter_dynamic(binding, compiled, geometry,
                                       validate=validate_native_occurrence)
               or _qkvmr_dynamic(binding, compiled, geometry,
                                 validate=validate_native_occurrence)
               or _swiglu_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _opdquant_dynamic(binding, compiled, geometry,
                                    validate=validate_native_occurrence)
               or _sig_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _scatpa_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _inzero_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _gmm2_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _gmm1_dynamic(binding, compiled, geometry,
                                validate=validate_native_occurrence)
               or _qbmm31_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _l2n_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _tbg_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _incopy_dynamic(binding, compiled, geometry,
                                   validate=validate_native_occurrence)
               or _mat_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _cat_dynamic(binding, compiled, geometry,
                               validate=validate_native_occurrence)
               or _index_dynamic(binding, compiled, geometry,
                                  validate=validate_native_occurrence)
               or _dynamic_moe(binding, geometry, env=env,
                               validate=validate_native_occurrence))
    gdn_chunk = (binding.api or {}).get("aclnn_symbol") == "aclnnChunkGatedDeltaRuleFwd"
    qbmm = (binding.api or {}).get("entry") == "torch_npu.npu_quant_matmul" or (
        (binding.typed.get("attrs") or {}).get("implementation") == "W8A8_MXFP8")
    sfa = (binding.api or {}).get("entry") == _SFA_ENTRY
    indexer = (binding.api or {}).get("entry") == _I1_ENTRY
    moe = (binding.api or {}).get("entry") in {_MOE_ENTRY, _UNPERMUTE_ENTRY}
    perm2 = (binding.api or {}).get("entry") == "torch_npu.npu_moe_token_permute"
    hist = (binding.api or {}).get("entry") == "torch.histc"
    moe_init3 = (binding.api or {}).get("entry") == _MOE_INIT3_ENTRY
    gmm1 = (binding.api or {}).get("entry") == _GMM1_ENTRY
    gmm2 = (binding.api or {}).get("entry") == _GMM2_ENTRY
    lnim = (binding.api or {}).get("entry") == _LNIM_ENTRY
    rope_siso = (binding.api or {}).get("entry") == "rope_forward_triton_siso"
    ssm_gather = (binding.api or {}).get("entry") == "_gather_ssm_states_kernel"
    ssm_scatter = (binding.api or {}).get("entry") == "_scatter_ssm_states_kernel"
    gated_norm = (binding.api or {}).get("entry") == _GATED_NORM_ENTRY
    rms_plain = (binding.api or {}).get("entry") == "torch_npu.npu_rms_norm"
    muls_add = (binding.api or {}).get("entry") == _MULS_ADD_ENTRY
    if gdn_chunk and dynamic is None:
        unresolved.append("gdn_dynamic_prepare_unavailable")
    if qbmm and dynamic is None:
        unresolved.append("qbmm_dynamic_prepare_unavailable")
    if sfa and dynamic is None:
        unresolved.append("sfa_dynamic_prepare_unavailable")
    if indexer and dynamic is None:
        unresolved.append("indexer_dynamic_prepare_unavailable")
    if moe and dynamic is None:
        unresolved.append("moe_dynamic_prepare_unavailable")
    if perm2 and dynamic is None:
        unresolved.append("moe_token_permute_prepare_unavailable")
    if hist and dynamic is None:
        unresolved.append("histc_prepare_unavailable")
    if moe_init3 and dynamic is None:
        unresolved.append("moe_init_routing_v3_prepare_unavailable")
    if gmm1 and dynamic is None:
        unresolved.append("grouped_matmul_swiglu_quant_v2_prepare_unavailable")
    if gmm2 and dynamic is None:
        unresolved.append("grouped_matmul_v4_prepare_unavailable")
    if lnim and dynamic is None:
        unresolved.append("layer_norm_dynamic_prepare_unavailable")
    if rope_siso and dynamic is None:
        unresolved.append("rope_siso_dynamic_prepare_unavailable")
    if ssm_gather and dynamic is None:
        unresolved.append("ssm_gather_dynamic_prepare_unavailable")
    if ssm_scatter and dynamic is None:
        unresolved.append("ssm_scatter_dynamic_prepare_unavailable")
    if gated_norm and dynamic is None:
        unresolved.append("gated_norm_dynamic_prepare_unavailable")
    if rms_plain and dynamic is None:
        unresolved.append("rms_norm_dynamic_prepare_unavailable")
    if muls_add and dynamic is None:
        unresolved.append("muls_add_prepare_unavailable")
    status = "bound" if _binding_bound(binding) else "unresolved"
    if (gdn_chunk or qbmm or sfa or indexer or moe or perm2 or hist or gmm1
            or gmm2 or moe_init3 or lnim or rope_siso or ssm_gather
            or ssm_scatter or gated_norm or rms_plain or muls_add) and dynamic is None:
        status = "unresolved"
    if sfa and dynamic is not None and not dynamic["prepare"].get("complete", True):
        unresolved.append("sfa_prepare_partial")
    occurrence = {
        "schema": OCCURRENCE_SCHEMA,
        "status": status,
        "call_id": binding.call_id, "op_id": binding.op_id,
        "family": binding.family, "recipe": binding.recipe,
        "source_boundary": binding.source_boundary,
        "phase": phase, "kind": kind, "layer": int(getattr(compiled, "layer", 0)),
        "repeat": 1, "index": int(getattr(cost, "segment_index", -1)),
        "rank": None, "charge": "call" if owner == binding.op_id else "member",
        "call_owner": owner, "typed": typed, "priced": _priced(cost),
        "signature_matches": True, "unresolved": unresolved,
    }
    if dynamic is not None:
        occurrence["prepare"] = dynamic["prepare"]
        occurrence["prepare_args"] = dynamic["prepare_args"]
        if perm2 and not dynamic["prepare"].get("complete", True):
            unresolved.append("moe_token_permute_prepare_partial")
        if hist and not dynamic["prepare"].get("complete", True):
            unresolved.append("histc_prepare_partial")
        if moe_init3 and not dynamic["prepare"].get("complete", True):
            unresolved.append("moe_init_routing_v3_prepare_partial")
        if gmm1 and not dynamic["prepare"].get("complete", True):
            unresolved.append("gmm1_prepare_partial")
        if gmm2 and not dynamic["prepare"].get("complete", True):
            unresolved.append("grouped_matmul_v4_prepare_partial")
        if lnim and not dynamic["prepare"].get("complete", True):
            unresolved.append("layer_norm_prepare_partial")
        if rope_siso and not dynamic["prepare"].get("complete", True):
            unresolved.append("rope_siso_prepare_partial")
        if ssm_gather and not dynamic["prepare"].get("complete", True):
            unresolved.append("ssm_gather_prepare_partial")
        if ssm_scatter and not dynamic["prepare"].get("complete", True):
            unresolved.append("ssm_scatter_prepare_partial")
        if gated_norm and not dynamic["prepare"].get("complete", True):
            unresolved.append("gated_norm_prepare_partial")
        if rms_plain and not dynamic["prepare"].get("complete", True):
            unresolved.append("rms_norm_prepare_partial")
        if muls_add and not dynamic["prepare"].get("complete", True):
            unresolved.append("muls_add_prepare_partial")
    if indexer and str(getattr(getattr(compiled, "operator", None),
                                  "name", "")) == "topk":
        occurrence["charge"] = "none"
        occurrence["membership"] = "consumed_by_selected_native_call"
        occurrence["timing_owner"] = {"op_id": "I1:indexer:index_score",
                                      "scope": "call"}
    return occurrence


def _unbound_occurrence(compiled: Any, cost: Any, geometry: Any) -> dict:
    operator = compiled.operator
    typed = _typed_view(operator.theta, _annotations(compiled.program.annotations),
                        family=operator.family, name=operator.name)
    declared = _declared_kind(operator.theta)
    reasons = ["exact_native_entry_unresolved",
               "attributable_source_reference_unresolved"]
    if declared:
        reasons.append(f"declared_native_kind:{declared}")
    return {
        "schema": OCCURRENCE_SCHEMA, "status": "unbound", "call_id": None,
        "op_id": f"{getattr(compiled, 'kind', '')}:{operator.family}:{operator.name}",
        "family": operator.family, "recipe": operator.template.role,
        "source_boundary": "analytical", "phase": str(getattr(geometry, "phase_name", "")),
        "kind": str(getattr(compiled, "kind", "") or ""),
        "layer": int(getattr(compiled, "layer", 0)), "repeat": 1,
        "index": int(getattr(cost, "segment_index", -1)), "rank": None,
        "charge": "none", "call_owner": None, "typed": typed,
        "priced": _priced(cost), "signature_matches": None,
        "unresolved": reasons,
    }


def attach_native_occurrence(compiled: Any, cost: Any, geometry: Any,
                             env: Mapping[str, Any]) -> Any:
    """Evaluate-phase attachment: typed occurrence before the row is exposed."""
    phase = str(getattr(geometry, "phase_name", "") or "")
    binding = compiled.native_call_for(phase)
    if (str(getattr(compiled, "name", "") or "") == "router_histogram"
            and binding is not None):
        from .native_r64_histogram import selected_protocol
        protocol = selected_protocol(binding, geometry)
        if protocol is None or not protocol.get("separate_histc"):
            occurrence = _unbound_occurrence(compiled, cost, geometry)
            occurrence.update({
                "status": "member", "charge": "none",
                "membership": "consumed_by_selected_mc2_count_call",
                "protocol": protocol,
                "unresolved": [],
            })
            return replace(cost, duration_ms=0, work=(), times=(),
                           stage_times=(), materialization=(), internal=(),
                           derived=(), stage_work=(), stage_stats=(),
                           dominant_pipe="", tied_pipes=(),
                           native_occurrence=occurrence)
    if binding is None:
        return replace(cost, native_occurrence=_unbound_occurrence(
            compiled, cost, geometry))
    typed = _actual_typed(binding, compiled, geometry, env)
    if typed is None:
        return replace(cost, native_occurrence=_unbound_occurrence(
            compiled, cost, geometry))
    occurrence = _occurrence(binding, compiled, cost, geometry, env, typed)
    return replace(cost, native_call=compiled.native_payload_for(phase),
                   native_occurrence=occurrence)


def _fusion_theta(instance: Any) -> Any:
    for operator in instance.operators:
        if operator.name == "mla_attention":
            return operator.theta
    return instance.operators[0].theta


def bind_fusion_call(instance: Any, contract: Any,
                     members: Sequence[Any]) -> NativeCallBinding | None:
    """One fused source call; members are semantic, not extra launches."""
    contract_map = dict(getattr(instance.template, "contract", ()) or ())
    if contract.name != "mla_prolog" or "sfa_prolog_v3" not in contract_map:
        return None
    from mindie_sim.prediction import family_mla_prolog
    from mindie_sim.prediction.piecewise import sfa_prolog
    field = sfa_prolog.parse_port_spec(str(contract_map["sfa_prolog_v3"]))
    heads = dict(_fusion_theta(instance).heads)
    q_heads = int(heads.get("q", 1) or 1)
    rkv = int(heads.get("rkv", 1) or 1)
    rope = int(heads.get("dr", 0) or 0)
    q_lora = int(field["q_c"])
    scale_bytes = int(field["q_c_scale"])
    typed = {
        "dtype": "bf16", "layout": "PA_BSND",
        "shapes": {"ql_nope": [q_heads, rkv], "q_pe": [q_heads, rope],
                   "q_c": [q_lora], "q_c_scale": [scale_bytes],
                   "cache_index": [1]},
        "attrs": {"cache_layout": "PA_BSND", "weight_quant_mode": 3,
                  "kv_cache_quant_mode": 3,
                  "query_norm_flag": int(int(_fusion_theta(instance).find(
                      "index_topk", 0) or 0) > 0),
                  "packed_row_bytes": int(field["packed_row"]),
                  "store_row_bytes": int(field["store_row"]),
                  "source_lines": sfa_prolog.SOURCE.get("caller_lines")},
        "rank_width": 2,
    }
    source = _source("vllm-ascend", family_mla_prolog.SOURCE["path"],
                     sha256=family_mla_prolog.SOURCE["sha256"],
                     symbol=(family_mla_prolog.SOURCE.get("symbols") or [None])[0],
                     revision=family_mla_prolog.SOURCE["revision"],
                     state="source")
    spec = _call_spec(
        _api("torch", "torch_npu.npu_mla_prolog_v3"), source, typed,
        (_kernel("MlaPrologV3", "mix", path=source["path"],
                 sha256=source["sha256"]),),
        tuple(contract.members), contract.name,
        unresolved=["aclnn_symbol_unresolved", "binary_identity_not_established",
                    "source_revision_not_pinned_to_i_r64"],
    )
    surface = _Surface(
        op_id=f"{getattr(instance, 'kind', '')}:fused:{contract.name}",
        family="fused", recipe=contract.name, phase="", source_boundary="call",
        dtype=typed["dtype"], layout=typed["layout"],
        metadata=tuple(sorted(typed["attrs"].items())),
    )
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, unresolved=spec.unresolved,
        members=spec.members,
    )
    return bind_native_call(declared)


def attach_fusion_native(cost: Any, instance: Any, contract: Any,
                         members: Sequence[Any]) -> Any:
    """Attach one call identity to the already single-charged fused cost."""
    if contract.name == "quant_lightning_indexer":
        calls = [getattr(member, "native_call", None) for member in members]
        calls = [call for call in calls if isinstance(call, dict)]
        if calls and len({str(call.get("call_id")) for call in calls}) == 1:
            call = dict(calls[0])
            call["op_id"] = f"{getattr(instance, 'kind', '')}:fused:{contract.name}"
            call["timing_owner"] = {"op_id": call["op_id"], "scope": "call"}
            call["members"] = {"charge": "call",
                               "charge_note": "priced once at fused owner",
                               "op_ids": list(contract.members)}
            member_occurrence = next(
                (getattr(member, "native_occurrence", None) or {}
                 for member in members if isinstance(getattr(member, "native_call", None), dict)),
                {})
            occurrence = {
                "schema": OCCURRENCE_SCHEMA, "status": "bound",
                "call_id": call["call_id"], "op_id": call["op_id"],
                "family": "fused", "recipe": contract.name,
                "source_boundary": "call",
                "phase": str(member_occurrence.get("phase") or ""),
                "kind": getattr(instance, "kind", ""), "layer": int(getattr(instance, "layer", 0)),
                "repeat": 1, "index": -1, "rank": None, "charge": "call",
                "call_owner": contract.name,
                "typed": member_occurrence.get("typed") or call.get("typed"),
                "priced": _priced(cost), "signature_matches": True,
                "unresolved": list(call.get("unresolved") or ())
                + ["execution_rank_not_selected"],
            }
            if member_occurrence.get("prepare") is not None:
                occurrence["prepare"] = member_occurrence["prepare"]
            if member_occurrence.get("prepare_args") is not None:
                occurrence["prepare_args"] = member_occurrence["prepare_args"]
            return replace(cost, native_call=call, native_occurrence=occurrence)
    if contract.name in {"add_rmsnorm", "add_rmsnorm_quant"}:
        calls = [getattr(member, "native_call", None) for member in members
                 if str(getattr(member, "name", "") or "") == "rmsnorm"]
        calls = [call for call in calls if isinstance(call, dict)]
        if calls:
            call = dict(calls[0])
            call["op_id"] = f"{getattr(instance, 'kind', '')}:fused:{contract.name}"
            call["timing_owner"] = {"op_id": call["op_id"], "scope": "call"}
            call["members"] = {"charge": "call",
                               "charge_note": "priced once at fused owner",
                               "op_ids": list(contract.members)}
            member_occurrence = next(
                (getattr(member, "native_occurrence", None) or {}
                 for member in members
                 if str(getattr(member, "name", "") or "") == "rmsnorm"), {})
            occurrence = {
                "schema": OCCURRENCE_SCHEMA, "status": "bound",
                "call_id": call["call_id"], "op_id": call["op_id"],
                "family": "fused", "recipe": contract.name,
                "source_boundary": "call",
                "phase": str(member_occurrence.get("phase") or ""),
                "kind": getattr(instance, "kind", ""),
                "layer": int(getattr(instance, "layer", 0)),
                "repeat": 1, "index": -1, "rank": None, "charge": "call",
                "call_owner": contract.name,
                "typed": member_occurrence.get("typed") or call.get("typed"),
                "priced": _priced(cost), "signature_matches": True,
                "unresolved": list(call.get("unresolved") or ())
                + ["execution_rank_not_selected"],
            }
            if member_occurrence.get("prepare") is not None:
                occurrence["prepare"] = member_occurrence["prepare"]
            if member_occurrence.get("prepare_args") is not None:
                occurrence["prepare_args"] = member_occurrence["prepare_args"]
            return replace(cost, native_call=call,
                           native_occurrence=occurrence)
    if contract.name == "qkv_norm_rope":
        calls = [getattr(member, "native_call", None) for member in members
                 if str(getattr(member, "name", "") or "") in {"qk_norm", "rotary"}]
        calls = [call for call in calls if isinstance(call, dict)]
        if calls:
            call = dict(calls[0])
            call["op_id"] = f"{getattr(instance, 'kind', '')}:fused:{contract.name}"
            call["timing_owner"] = {"op_id": call["op_id"], "scope": "call"}
            call["members"] = {"charge": "call",
                               "charge_note": "priced once at fused owner",
                               "op_ids": list(contract.members)}
            member_occurrence = next(
                (getattr(member, "native_occurrence", None) or {}
                 for member in members
                 if str(getattr(member, "name", "") or "") in {"qk_norm", "rotary"}
                 and isinstance(getattr(member, "native_call", None), dict)), {})
            occurrence = {
                "schema": OCCURRENCE_SCHEMA, "status": "bound",
                "call_id": call["call_id"], "op_id": call["op_id"],
                "family": "fused", "recipe": contract.name,
                "source_boundary": "call",
                "phase": str(member_occurrence.get("phase") or ""),
                "kind": getattr(instance, "kind", ""),
                "layer": int(getattr(instance, "layer", 0)),
                "repeat": 1, "index": -1, "rank": None, "charge": "call",
                "call_owner": contract.name,
                "typed": member_occurrence.get("typed") or call.get("typed"),
                "priced": _priced(cost), "signature_matches": True,
                "unresolved": list(call.get("unresolved") or ())
                + ["execution_rank_not_selected"],
            }
            if member_occurrence.get("prepare") is not None:
                occurrence["prepare"] = member_occurrence["prepare"]
            if member_occurrence.get("prepare_args") is not None:
                occurrence["prepare_args"] = member_occurrence["prepare_args"]
            return replace(cost, native_call=call,
                           native_occurrence=occurrence)
    binding = bind_fusion_call(instance, contract, members)
    if binding is None:
        occurrence = {
            "schema": OCCURRENCE_SCHEMA, "status": "unbound", "call_id": None,
            "op_id": f"{getattr(instance, 'kind', '')}:fused:{contract.name}",
            "family": "fused", "recipe": contract.name,
            "source_boundary": "call", "phase": "", "kind": getattr(instance, "kind", ""),
            "layer": int(getattr(instance, "layer", 0)), "repeat": 1,
            "index": -1, "rank": None, "charge": "call", "call_owner": contract.name,
            "typed": None, "priced": _priced(cost), "signature_matches": None,
            "unresolved": ["fused_source_not_selected", "exact_native_entry_unresolved"],
        }
        return replace(cost, native_occurrence=occurrence)
    occurrence = {
        "schema": OCCURRENCE_SCHEMA, "status": "bound", "call_id": binding.call_id,
        "op_id": binding.op_id, "family": "fused", "recipe": contract.name,
        "source_boundary": "call", "phase": "",
        "kind": getattr(instance, "kind", ""),
        "layer": int(getattr(instance, "layer", 0)), "repeat": 1, "index": -1,
        "rank": None, "charge": "call", "call_owner": contract.name,
        "typed": binding.typed, "priced": _priced(cost), "signature_matches": True,
        "unresolved": list(binding.unresolved) + ["execution_rank_not_selected"],
    }
    return replace(cost, native_call=binding.to_dict(),
                   native_occurrence=occurrence)


def _row_bound(call: Mapping[str, Any]) -> bool:
    if str(call.get("status") or "") == "unresolved":
        return False
    api = call.get("api") or {}
    source = call.get("source") or {}
    typed = call.get("typed") or {}
    unresolved = set(call.get("unresolved") or ())
    if not (api.get("entry") or api.get("aclnn_symbol")):
        return False
    if not isinstance(source, dict) or not source.get("path") or not typed.get("signature_hash"):
        return False
    if not source.get("sha256") and api.get("namespace") != "hccl":
        return False
    return not unresolved & {"exact_native_entry_unresolved",
                             "exact_native_entry_not_selected",
                             "attributable_source_reference_unresolved"}


def forecast_native_coverage(rows: Iterable[Any]) -> dict[str, Any]:
    """Physical-occurrence coverage, with explicit fusion attribution separated."""
    rows = tuple(rows)
    physical: list[dict] = []
    attribution: list[dict] = []
    phase_attribution: list[dict] = []
    fused_owners: dict[str, dict] = {}
    unrecorded = 0
    owner_call_ids = {
        str((getattr(row, "native_call", None) or {}).get("call_id"))
        for row in rows
        if str((getattr(row, "native_occurrence", None) or {}).get("charge")) == "call"
        and isinstance(getattr(row, "native_call", None), dict)
        and (getattr(row, "native_call", None) or {}).get("call_id")
    }
    for row in rows:
        count = max(1, int(getattr(row, "count", 1) or 1))
        call = getattr(row, "native_call", None)
        occurrence = getattr(row, "native_occurrence", None) or {}
        family = str((call or {}).get("family")
                     or getattr(row, "mechanism", "")
                     or getattr(row, "family", "")
                     or "unbound")
        membership = str(occurrence.get("membership") or "")
        if membership == "selected_fused_owner":
            owner = fused_owners.setdefault(str(getattr(row, "name", "") or "-"), {
                "owner": str(getattr(row, "name", "") or "-"),
                "count": 0, "members": list(occurrence.get("members") or ()),
                "charge": occurrence.get("charge"), "physical": True,
            })
            owner["count"] += count
        consumed = (membership in {"consumed_by_selected_fusion",
                                   "consumed_by_selected_native_call"}
                    or occurrence.get("consumed_by_selected_fusion") is True)
        phase_member = (occurrence.get("charge") == "member"
                        and isinstance(call, dict)
                        and str(call.get("call_id")) in owner_call_ids
                        and str(occurrence.get("call_owner") or "") == str(
                            (call.get("timing_owner") or {}).get("op_id")
                            or occurrence.get("call_owner") or ""))
        if phase_member:
            for _ in range(count):
                phase_attribution.append({
                    "family": family,
                    "recipe": str(getattr(row, "name", "") or "-"),
                    "phase": str(occurrence.get("phase") or ""),
                    "call_id": call.get("call_id"),
                    "call_owner": occurrence.get("call_owner"),
                    "timing_owner": occurrence.get("timing_owner"),
                    "kernels": call.get("kernels", ()),
                })
            continue
        if consumed:
            for _ in range(count):
                attribution.append({
                    "family": family,
                    "recipe": str(getattr(row, "name", "") or "-"),
                    "source_boundary": "attribution",
                    "timing_owner": occurrence.get("timing_owner"),
                })
            continue
        if not isinstance(call, dict):
            unrecorded += count
            call = {"family": family,
                    "recipe": str(getattr(row, "name", "") or "-"),
                    "source_boundary": "analytical",
                    "unresolved": ["no_binding_record"]}
        call = dict(call)
        call["family"] = str(call.get("family") or family)
        call["recipe"] = str(call.get("recipe") or getattr(row, "name", "") or "-")
        call["source_boundary"] = str(call.get("source_boundary") or "analytical")
        for _ in range(count):
            physical.append(call)
    emitted = len(physical)
    grouped: dict[tuple, dict] = {}
    reasons: dict[str, int] = {}
    unique: dict[str, dict] = {}
    for record in physical:
        family = str(record.get("family") or "unbound")
        key = (family, record.get("source_boundary") or "analytical",
               record.get("recipe") or "-")
        row = grouped.setdefault(key, {"family": family, "source_boundary": key[1],
                                       "recipe": key[2], "emitted": 0, "bound": 0,
                                       "unresolved": 0, "attribution": 0})
        row["emitted"] += 1
        status = str(record.get("status") or "")
        if status == "unresolved" or not _row_bound(record):
            row["unresolved"] += 1
            for reason in (record.get("unresolved") or ("no_binding_record",)):
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        else:
            row["bound"] += 1
            unique.setdefault(record["call_id"], {
                "call_id": record["call_id"], "family": family,
                "api": record.get("api"), "source": record.get("source"),
                "typed": record.get("typed"), "kernels": record.get("kernels", ()),
                "members": (record.get("members") or {}).get("op_ids", ()),
                "occurrences": 0})
            unique[record["call_id"]]["occurrences"] += 1
    attribution_by_family: dict[str, int] = {}
    attribution_members: dict[tuple, dict] = {}
    phase_owners: dict[str, dict] = {}
    for record in phase_attribution:
        key = str(record.get("call_id"))
        owner = phase_owners.setdefault(key, {
            "call_id": key, "call_owner": record.get("call_owner"),
            "timing_owner": record.get("timing_owner"),
            "phases": {}, "kernels": list(record.get("kernels") or ()),
            "physical": 1})
        phase = str(record.get("phase") or record.get("recipe") or "-")
        owner["phases"][phase] = owner["phases"].get(phase, 0) + 1
    for record in attribution:
        family = str(record.get("family") or "unbound")
        attribution_by_family[family] = attribution_by_family.get(family, 0) + 1
        member_key = (family, str(record.get("recipe") or "-"),
                      str(record.get("timing_owner") or ""))
        member = attribution_members.setdefault(member_key, {
            "family": family, "recipe": member_key[1],
            "timing_owner": member_key[2], "attribution": 0})
        member["attribution"] += 1
        key = (family, "attribution", record.get("recipe") or "-")
        row = grouped.setdefault(key, {"family": family, "source_boundary": "attribution",
                                       "recipe": key[2], "emitted": 0, "bound": 0,
                                       "unresolved": 0, "attribution": 0})
        row["attribution"] += 1
    base = coverage_from_records(physical, emitted=emitted)
    return {
        "schema": "mindie-profiler/native-forecast-coverage/1",
        "denominator": {**base["denominator"], "emitted_rows": len(rows),
                        "recorded_bindings": emitted - unrecorded,
                        "unrecorded_emitted": unrecorded,
                        "distinct_calls": len(unique),
                        "physical_operators": emitted,
                        "attribution_operators": len(attribution),
                        "phase_attribution_operators": len(phase_attribution),
                        "total_operators": emitted + len(attribution) + len(phase_attribution)},
        "totals": {
            "emitted_occurrences": emitted,
            "bound_occurrences": sum(row["bound"] for row in grouped.values()),
            "unresolved_occurrences": sum(row["unresolved"] for row in grouped.values()),
            "recorded_occurrences": emitted - unrecorded,
            "unrecorded_occurrences": unrecorded,
            "recorded_unresolved_occurrences": (
                emitted - unrecorded
                - sum(row["bound"] for row in grouped.values())),
            "attribution_occurrences": len(attribution),
            "phase_attribution_occurrences": len(phase_attribution),
            "total_occurrences": emitted + len(attribution) + len(phase_attribution),
            "unique_calls": len(unique),
            "with_entry": sum(1 for record in physical if record
                              and (record.get("api") or {}).get("entry")),
            "with_source_sha256": sum(1 for record in physical if record
                                      and (record.get("source") or {}).get("sha256")),
            "with_kernels": sum(1 for record in physical if record
                                and record.get("kernels")),
        },
        "by_source_family": [grouped[key] for key in sorted(grouped)],
        "attribution_by_family": dict(sorted(attribution_by_family.items())),
        "attribution_members": [attribution_members[key]
                                for key in sorted(attribution_members)],
        "fused_owners": [fused_owners[key] for key in sorted(fused_owners)],
        "phased_calls": [phase_owners[key] for key in sorted(phase_owners)],
        "unresolved_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])),
        "unique_calls": [unique[key] for key in sorted(unique)],
        "non_claims": ["physical_emitted_rows_only", "fusion_members_are_attribution_only",
                       "no_catalogue_counted_as_coverage",
                       "unique_call_requires_entry_source_and_typed_abi"],
    }


def model_native_coverage(rows_by_model: Mapping[str, Iterable[Any]]) -> dict[str, Any]:
    """Per-model ordinary-emission coverage; no catalogue denominator."""
    by_model = {str(model): forecast_native_coverage(rows)
                for model, rows in sorted(rows_by_model.items())}
    return {
        "schema": "mindie-profiler/native-forecast-coverage-by-model/1",
        "models": by_model,
        "totals": {
            "models": len(by_model),
            "emitted_occurrences": sum(item["denominator"]["emitted_operators"]
                                       for item in by_model.values()),
            "bound_occurrences": sum(item["totals"]["bound_occurrences"]
                                     for item in by_model.values()),
            "unresolved_occurrences": sum(item["totals"]["unresolved_occurrences"]
                                          for item in by_model.values()),
            "unique_calls": sum(item["totals"]["unique_calls"]
                                for item in by_model.values()),
            "attribution_occurrences": sum(item["totals"]["attribution_occurrences"]
                                           for item in by_model.values()),
            "total_occurrences": sum(item["totals"]["total_occurrences"]
                                     for item in by_model.values()),
        },
        "non_claims": ["per_model_physical_emission_only"],
    }
