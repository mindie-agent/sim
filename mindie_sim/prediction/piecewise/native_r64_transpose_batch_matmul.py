"""Prepare-only R64 aclnnTransposeBatchMatMul value-expansion invocation.

[Role] Expand the R64 SFA output-projection head product
  ``torch_npu.npu_transpose_batchmatmul(x, W_UV, perm_x1=(1, 0, 2),
  perm_y=(1, 0, 2))`` from ``AscendSFAImpl._v_up_proj`` into the actual BF16
  ``[B,N,L] x [N,L,V] -> [B,N,V]`` descriptors for the evaluated token
  geometry.  This is the MLA latent-value expansion, not the sparse-attention
  call and not the downstream ``o_projection``.

[Invariants] Only the exact R64 caller blob is admitted.  The token count
  ``B`` lives in the occurrence; ``N`` (heads), ``L`` (kv_lora_rank), ``V``
  (v_head_dim), the BF16 dtypes, the permutation views and the source identity
  stay in the binding.  The Python wrapper signature and the kernel source are
  not retained, so the prepare record keeps ``complete=False`` and never
  claims a fabricated symbol, descriptor value or device launch.  A different
  image, role, dtype or permutation stays unbound.

[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.

[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal geometry; no tensor allocation, device access or measurement is
  used to price the call.
"""
from __future__ import annotations

from typing import Any, Mapping

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_SOURCE_SHA = "82eeef0272977cc2bf0c870699eba005ae71236b1eef7c4b71fe67303a621e68"
R64_HOST_BINARY_SHA = (
    "0d8feccb7246b3f53a7dda95c961c8d5abf2adc7f0062219a1628c44f2df2ab8")
REVISION = "r64/8d422528"
ENTRY = "torch_npu.npu_transpose_batchmatmul"
ACLNN_API = "aclnnTransposeBatchMatMul"
KERNEL_NAME = "TransposeBatchMatMul"
PROFILER_NAME = (
    "aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul")
SOURCE_LINES = (1336, 1350)
CALL_LINE = 1340
CALL_EXPRESSION = (
    "x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, "
    "perm_x1=(1, 0, 2), perm_y=(1, 0, 2))")
PERM_X1 = (1, 0, 2)
PERM_Y = (1, 0, 2)
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/attention/sfa_v1.py",
    "sha256": R64_SOURCE_SHA,
    "symbol": "AscendSFAImpl._v_up_proj",
    "revision": REVISION,
    "state": "task_artifact",
}
HOST_BINARY = {
    "retained_artifact": "I-r64/libophost_nn.so",
    "sha256": R64_HOST_BINARY_SHA,
    "symbols": (
        "TransposeBatchMatMul",
        "TransposeBatchMatMulTilingFunc",
        "CheckPermForTransposeBatchMatMul",
        "InferShapeForTransposeBatchMatMul",
        "TilingPrepareForTransposeBatchMatMul",
        "optiling23transpose_batch_mat_mul",
    ),
    "source_status": "runtime_binary_only",
}
OBSERVED = {
    "case": "glm51_dp16tp1ep16_pf_1k_c1",
    "declared_revision": "r64",
    "source_csv": (
        "/Volumes/data/mindie-agent-workstreams/"
        "a5-multinode-fleet-rack06-20261107/out/glm51_dp16tp1ep16/"
        "pf_1k_c1/177.25.21.1/"
        "dp2_pp0_tp0_dcp0_ep2_rank0_1467_20261107094934940_ascend_pt/"
        "kernel_details.csv.gz"),
    "source_csv_sha256": (
        "0223d793acf8adb2fac3b649eef0873c0f4493331d1fc67aafcd502eea5cb16f"),
    "count": 78,
    "prefill_count": 78,
    "prefill_input_shapes": "1024,64,512;64,512,256",
    "prefill_output_shapes": "1024,64,256",
    "decode_count": 1326,
    "decode_input_shapes": "1,64,512;64,512,256",
    "decode_output_shapes": "1,64,256",
    "input_dtypes": "DT_BF16;DT_BF16",
    "output_dtypes": "DT_BF16",
    "profiler_name": PROFILER_NAME,
}
ROLE = "value_expansion"
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pair_tuple(value: Any) -> tuple | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    from . import native_calls
    return native_calls._r64_source_meta(meta)


def _static_contract(theta: Any) -> tuple[dict, int, int, int] | None:
    """Prove one BF16 absorbed-value head product from the static view."""
    view = theta.find("native_tensor_view", None)
    if isinstance(view, Mapping):
        view = dict(view)
    elif isinstance(view, (list, tuple)):
        try:
            view = dict(view)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if str(view.get("kind") or "") != "transpose_batch_matmul":
        return None
    if str(view.get("role") or "") != ROLE:
        return None
    if _pair_tuple(view.get("input_permutation")) != PERM_X1:
        return None
    if _pair_tuple(view.get("output_permutation")) != PERM_Y:
        return None
    heads = _int(view.get("batch") or theta.find("batch", 0))
    k = _int(view.get("k"))
    n = _int(view.get("n"))
    if heads < 1 or k < 1 or n < 1:
        return None
    declared_batch = theta.find("batch", None)
    if declared_batch is not None and _int(declared_batch) != heads:
        return None
    if _int(theta.find("K", k)) != k or _int(theta.find("N", n)) != n:
        return None
    input_shape = view.get("input_shape")
    if input_shape is not None:
        shape = [int(item) for item in input_shape]
        if shape[-2:] != [heads, k]:
            return None
    weight_shape = view.get("weight_shape")
    if weight_shape is not None:
        if [int(item) for item in weight_shape] != [heads, k, n]:
            return None
    output_shape = view.get("output_shape")
    if output_shape is not None:
        if [int(item) for item in output_shape][-2:] != [heads, n]:
            return None
    formats = [str(theta.find(key, "bf16") or "bf16").lower()
               for key in ("weight_format", "act_format", "output_format")]
    if any(item not in {"bf16", "bfloat16"} for item in formats):
        return None
    if _int(theta.find("quantized", 0)) != 0:
        return None
    return dict(view), heads, k, n


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 value-expansion call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  The identity is the source-selected ``v_expand`` operator on the
    exact R64 image; a missing token axis stays an occurrence fact and never
    becomes a static dimension.
    """
    if str(getattr(compiled, "kind", "") or "") != "A1":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "gemm"
            or str(getattr(operator, "name", "") or "") != "v_expand"):
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    theta = operator.theta
    contract = _static_contract(theta)
    if contract is None:
        return None
    view, heads, k, n = contract
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "unknown", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        binary_sha256=R64_HOST_BINARY_SHA,
        kernel_source_status="kernel_source_not_retained",
        source_status="runtime_binary_only")
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {
            "input": [heads, k],
            "weight": [heads, k, n],
            "output": [heads, n],
        },
        "attrs": {
            "native_kind": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "acl_interface_status": (
                "captured_call_label_and_retained_host_tiling_binary"),
            "python_wrapper_source_status": "not_retained",
            "kernel_source_status": "not_retained",
            "binary_selection_status": "uncertain",
            "deployment_image_id": str(
                meta.get("image_id") or meta.get("deployment_image_id")
                or R64_IMAGE_SHA),
            "deployment_source_revision": "r64",
            "source_sha256": SOURCE["sha256"],
            "source_symbol": SOURCE["symbol"],
            "source_lines": list(SOURCE_LINES),
            "source_call_line": CALL_LINE,
            "source_call_expression": CALL_EXPRESSION,
            "host_binary_sha256": R64_HOST_BINARY_SHA,
            "host_tiling_symbols": list(HOST_BINARY["symbols"]),
            "matrix_batch": heads,
            "matrix_k": k,
            "matrix_n": n,
            "activation_layout": "BNL",
            "weight_layout": "NLV",
            "output_layout": "BNV",
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
            "output_dtype": "bf16",
            "perm_x1": list(PERM_X1),
            "perm_y": list(PERM_Y),
            "input_permutation": list(PERM_X1),
            "output_permutation": list(PERM_Y),
            "permutation_scope": str(
                view.get("permutation_scope")
                or "logical_view; physical_native_layout_not_selected"),
            "role": ROLE,
            "rows": "runtime_tokens",
        },
        "rank_width": 3,
    }
    owner = str(theta.find("native_charge_owner", "") or operator.name)
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=REVISION),
        source, typed, (kernel,), (), owner,
        unresolved=(
            "python_wrapper_signature_not_retained",
            "aclnn_header_not_retained",
            "kernel_source_not_retained",
            "binary_selection_uncertain",
            "acl_interface_capture_and_host_tiling_binary_only",
            "execution_core_not_declared",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected value expansion."""
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("native_kind") != KERNEL_NAME
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("deployment_source_revision") != "r64"
            or attrs.get("role") != ROLE):
        return None
    if (str(getattr(compiled, "kind", "") or "") != "A1"
            or str(getattr(getattr(compiled, "operator", None), "name", "")
                   or "") != "v_expand"):
        return None
    heads = _int(attrs.get("matrix_batch"))
    k = _int(attrs.get("matrix_k"))
    n = _int(attrs.get("matrix_n"))
    tokens = _int(getattr(geometry, "effective_tokens", 0))
    if heads < 1 or k < 1 or n < 1 or tokens < 1:
        return None
    cache_key = (binding.call_id, tokens,
                 (binding.timing_owner or {}).get("op_id"))
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "input": [tokens, heads, k],
        "weight": [heads, k, n],
        "output": [tokens, heads, n],
    }
    actual_attrs = {**attrs, "tokens": tokens}
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "input": _descriptor([tokens, heads, k], "bf16", "ND", "input"),
        "weight": _descriptor([heads, k, n], "bf16", "ND", "weight"),
        "output": _descriptor([tokens, heads, n], "bf16", "ND", "output"),
    }
    descriptors["input"].update({
        "logical_layout": "BNL", "permutation": list(PERM_X1),
        "native_view_shape": [heads, tokens, k],
        "producer": "mla_attention_output",
    })
    descriptors["weight"].update({
        "logical_layout": "NLV", "role_name": "W_UV",
        "source": (
            "kv_b_proj_weight.view(L,N,P+V).split([P,V],dim=-1)[1]"
            ".transpose(0,1).contiguous()"),
    })
    descriptors["output"].update({
        "logical_layout": "BNV", "permutation": list(PERM_Y),
        "native_view_shape": [heads, tokens, n],
        "consumer": "o_projection_or_grouped_output",
    })
    arguments = {
        "x1": "input", "x2": "weight",
        "perm_x1": list(PERM_X1), "perm_y": list(PERM_Y),
    }
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "source_status": "kernel_source_not_retained",
                   "binary_sha256": R64_HOST_BINARY_SHA})
    prepare = {
        "schema": "mindie-sim/transpose-batch-matmul-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": list(SOURCE_LINES),
        "source_call_line": CALL_LINE,
        "source_call_expression": CALL_EXPRESSION,
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": ["x1", "x2", "perm_x1", "perm_y"],
        "positional_arguments": {"x1": "input", "x2": "weight"},
        "keywords": {"perm_x1": list(PERM_X1),
                     "perm_y": list(PERM_Y)},
        "optional_slots": {
            "perm_x2": "not_passed_source_default",
            "y": "returned_tensor",
        },
        "prototype": CALL_EXPRESSION,
        "prototype_status": (
            "exact_source_call_expression; python_wrapper_signature_not_retained"),
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "native_layout": {
            "input": "BNL", "weight": "NLV", "output": "BNV",
            "perm_x1": list(PERM_X1), "perm_y": list(PERM_Y),
            "permutation_scope": (
                "logical_view; transposes are not materialized by the caller"),
        },
        "acl_interface": {
            "acl_api": ACLNN_API,
            "captured_profiler_label": PROFILER_NAME,
            "captured_native_type": KERNEL_NAME,
            "input_arity": 2,
            "output_arity": 1,
            "retained_host_tiling_binary": {
                "artifact": HOST_BINARY["retained_artifact"],
                "sha256": R64_HOST_BINARY_SHA,
                "symbols": list(HOST_BINARY["symbols"]),
            },
            "python_wrapper_source": "not_retained",
            "aclnn_header": "not_retained",
            "status": "captured_call_and_host_tiling_binary_only",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "tokens": tokens,
            "heads": heads,
            "k": k,
            "n": n,
            "input_shape": [tokens, heads, k],
            "weight_shape": [heads, k, n],
            "output_shape": [tokens, heads, n],
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
            "output_dtype": "bf16",
            "perm_x1": list(PERM_X1),
            "perm_y": list(PERM_Y),
            "source_sha256": source.get("sha256"),
            "source_symbol": source.get("symbol"),
            "source_lines": list(SOURCE_LINES),
            "source_call_line": CALL_LINE,
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "charge_owner": (binding.timing_owner or {}).get("op_id"),
            "observed_original": dict(OBSERVED),
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id")
                            or R64_IMAGE_SHA),
            "revision": REVISION,
            "selection": (
                "exact R64 image id + AscendSFAImpl._v_up_proj caller blob"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": None,
            "retained_source_artifact": (
                "I-r64/selected/vllm-workspace/vllm-ascend/"
                "vllm_ascend/attention/sfa_v1.py"),
            "host_binary_sha256": R64_HOST_BINARY_SHA,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": ("python_wrapper_and_kernel_source_not_retained; "
                       "captured ACLNN label and host tiling symbols only"),
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "host_binary_sha256": R64_HOST_BINARY_SHA,
            "host_tiling_symbols": list(HOST_BINARY["symbols"]),
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "python_wrapper_signature_not_retained",
            "aclnn_header_not_retained",
            "kernel_source_not_retained",
            "binary_selection_uncertain",
        ],
        "unresolved": [
            "python_wrapper_signature_not_retained",
            "aclnn_header_not_retained",
            "kernel_source_not_retained",
            "binary_selection_uncertain",
            "execution_core_not_declared",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_python_wrapper_signature_claim",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "bf16", "layout": "ND",
                         "shapes": actual_shapes,
                         "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
