"""Prepare-only R31 GDN ``gather_ssm_states`` Triton invocation.

[Role] Expand the selected R31 ``gather_ssm_states`` call in
  ``vllm_ascend/ops/gdn.py:790-794`` into the exact
  ``_gather_ssm_states_kernel`` arguments, row/stride metadata and output
  descriptor for the evaluated prefill request vector.
[Invariants] Only the exact R31 deployment identity and the A3
  ``fla_gdn_prefill`` prefill branch are admitted.  State inner dimensions,
  dtype, index dtype, kernel, block size and launch shape are static; batch,
  ``has_initial_state``, selected rows and the runtime state-cache row axis
  stay occurrence facts.  The state-cache row count and any padded
  ``state.stride(0)`` are never hardcoded: when the allocator does not
  declare them the prepare is explicitly partial.  No tensor is allocated,
  no pointer is fabricated and no duration is changed.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected source and the
  existing request geometry.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from mindie_sim.schema.common import dtype_bytes

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_COMMIT = "8fd0bc4885cb52e58791aaa8efacc571b9bd115d"
R31_SOURCE_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
STATE_INDEX_PATH = "vllm_ascend/ops/triton/mamba/state_index.py"
STATE_INDEX_SHA = (
    "03debd381d4101279fff9f9cf8216ccc2b92666e2246427519a0f57fea8d7d66")
GDN_PATH = "vllm_ascend/ops/gdn.py"
GDN_SHA = (
    "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed")
KERNEL_NAME = "_gather_ssm_states_kernel"
WRAPPER_SYMBOL = "gather_ssm_states"
CALL_BRANCH = "fla_gdn_prefill"
STATE_IO_BLOCK_SIZE = 1024
NUM_WARPS = 8
CALLER_LINES = [790, 794]
WRAPPER_LINES = [124, 167]
KERNEL_LINES = [14, 43]
CAPTURE_PATH = (
    "/Volumes/data/mindie-sim-data/a5/heldout_engine_20261031/"
    "a5_synth_qwen35_L8_E16_K2_tp4ep4_prefill_4k_c1/kernel_details.csv")
CAPTURE_SHA = (
    "40e58861af47f2aa05f134175f35348613c31b899791c5dd7fae6c4b0a05b9e3")
CAPTURE_IMAGE = (
    "sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4")
_DTYPE_ALIASES = {
    "float32": "fp32", "fp32": "fp32",
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16",
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Any) -> str:
    text = _text(value).lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _dtype(value: Any) -> str | None:
    return _DTYPE_ALIASES.get(_text(value).lower())


def _r31_identity_from_meta(meta: Any) -> dict[str, str] | None:
    """The exact R31 deployment/source identity, never another image."""
    meta = dict(meta or ())
    for key in ("image_id", "oci_image_digest", "deployment_image_id",
                "image", "deployment_image", "oci_image", "source_image"):
        text = _digest(meta.get(key))
        if R31_IMAGE_PREFIX in text:
            return {"image_id": text, "revision": R31_REVISION,
                    "basis": f"feature_metadata.{key}"}
    for key in ("deployment_source_revision", "source_revision",
                "mx_quant_source_revision"):
        text = _text(meta.get(key)).lower()
        if text.startswith("r31"):
            return {"image_id": R31_IMAGE_SHA, "revision": R31_REVISION,
                    "basis": f"feature_metadata.{key}"}
    return None


def _r31_identity(compiled: Any) -> dict[str, str] | None:
    return _r31_identity_from_meta(
        dict(getattr(compiled, "feature_metadata", ()) or ()))


def source_selected(metadata: Any) -> bool:
    """True only for the exact R31 deployment/source declaration.

    The shared A3 builder uses this to add the gather operator only for the
    R31 source.  Other images (including the original c3d00a Qwen36 source)
    keep their ordinary operator set unchanged.
    """
    return _r31_identity_from_meta(metadata) is not None


def _source_lowerings(compiled: Any) -> tuple[str, ...]:
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    raw = meta.get("source_lowerings") or ()
    return tuple(str(value) for value in raw)


def _heads_dims(theta: Any) -> tuple[int, int, int] | None:
    """Local value heads and the two state dimensions from declared theta."""
    heads_raw = theta.find("heads", None)
    heads = dv = dk = None
    if isinstance(heads_raw, Mapping):
        heads = (heads_raw.get("dv") or heads_raw.get("kv")
                 or heads_raw.get("q"))
        dv = heads_raw.get("dv")
        dk = heads_raw.get("dk")
    elif heads_raw is not None:
        heads = heads_raw
    if heads is None:
        heads = theta.find("value_heads", theta.find(
            "v_heads", theta.find("n_heads", None)))
    if dv is None:
        dv = theta.find("dv", theta.find(
            "d_v", theta.find("v_dim", theta.find("value_dim", None))))
    if dk is None:
        dk = theta.find("dk", theta.find(
            "d_k", theta.find("k_dim", theta.find("key_dim", None))))
    heads, dv, dk = _int_value(heads), _int_value(dv), _int_value(dk)
    if heads is None or dv is None or dk is None:
        return None
    if heads < 1 or dv < 1 or dk < 1:
        return None
    return heads, dv, dk


def _static_dims(compiled: Any, phase: str = "") -> dict[str, Any] | None:
    """The proven source signature, or ``None`` for another invocation."""
    if str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    family = str(getattr(operator, "family", "") or "")
    name = str(getattr(operator, "name", "") or "")
    if family not in {"state_gather", "copy"}:
        return None
    if not name or "gather" not in name:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "") or "")
    if phase_name and phase_name != "prefill":
        return None
    variant = str(getattr(compiled, "variant", "") or "")
    if not phase_name and variant in {"recurrent_decode", "decode", "draft"}:
        return None
    lowerings = _source_lowerings(compiled)
    identity = _r31_identity(compiled)
    if identity is None:
        return None
    theta = operator.theta
    dims = _heads_dims(theta)
    if dims is None:
        return None
    heads, dv, dk = dims
    layout = _text(theta.find("state_layout", "V,K")).upper().replace(" ", "")
    if layout not in {"V,K", "K,V"}:
        return None
    inner = [heads, dv, dk] if layout == "V,K" else [heads, dk, dv]
    state_dtype = _dtype(theta.find(
        "state_dtype", theta.find("recurrent_state_dtype", None)))
    if state_dtype is None:
        return None
    index_dtype = _text(theta.find("index_dtype", "")).lower()
    if index_dtype not in {"int32", "int64"}:
        return None
    declared_output = _dtype(theta.find("output_dtype", None))
    if declared_output is not None and declared_output != state_dtype:
        # The source's other branch (``use_fused_chunk`` lines 815-820)
        # explicitly asks for bf16; it is a different call identity.
        return None
    source_kernel = _text(theta.find("source_kernel", KERNEL_NAME))
    if source_kernel and source_kernel != KERNEL_NAME:
        return None
    branch = _text(theta.find(
        "gather_branch", theta.find("call_branch", CALL_BRANCH)))
    if branch not in {"", CALL_BRANCH}:
        return None
    row_size = heads * dv * dk
    row_bytes = theta.find("row_bytes", None)
    if row_bytes is not None:
        try:
            value = float(row_bytes)
        except (TypeError, ValueError):
            return None
        if value != row_size * dtype_bytes(state_dtype):
            return None
    raw_state_rows = theta.find("state_cache_rows", None)
    state_rows = None
    if raw_state_rows is not None:
        state_rows = _int_value(raw_state_rows)
        if state_rows is None or state_rows < 1:
            return None
    raw_stride0 = theta.find(
        "state_stride0", theta.find("state_stride0_elements", None))
    state_stride0 = None
    if raw_stride0 is not None:
        state_stride0 = _int_value(raw_stride0)
        if state_stride0 is None or state_stride0 < row_size:
            return None
    stride_indices = None
    raw_indices_stride = theta.find(
        "stride_indices", theta.find("index_stride", None))
    if raw_indices_stride is not None:
        stride_indices = _int_value(raw_indices_stride)
        if stride_indices is None or stride_indices < 1:
            return None
    stride_has_initial_state = None
    raw_has_stride = theta.find(
        "stride_has_initial_state", theta.find(
            "has_initial_state_stride", None))
    if raw_has_stride is not None:
        stride_has_initial_state = _int_value(raw_has_stride)
        if stride_has_initial_state is None or stride_has_initial_state < 1:
            return None
    return {
        "operator": operator, "name": name, "identity": identity,
        "heads": heads, "dv": dv, "dk": dk, "state_layout": layout,
        "inner": inner, "state_dtype": state_dtype,
        "index_dtype": index_dtype, "row_size": row_size,
        "state_rows": state_rows, "state_stride0": state_stride0,
        "stride_indices": stride_indices,
        "stride_has_initial_state": stride_has_initial_state,
        "source_lowerings": list(lowerings),
    }


def _typed(dims: Mapping[str, Any]) -> dict[str, Any]:
    inner = list(dims["inner"])
    row_size = int(dims["row_size"])
    attrs = {
        "state_layout": dims["state_layout"],
        "state_inner_shape": inner,
        "heads": int(dims["heads"]),
        "dv": int(dims["dv"]),
        "dk": int(dims["dk"]),
        "row_size": row_size,
        "state_dtype": dims["state_dtype"],
        "index_dtype": dims["index_dtype"],
        "output_dtype": dims["state_dtype"],
        "block_size": STATE_IO_BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "grid_blocks": math.ceil(row_size / STATE_IO_BLOCK_SIZE),
        "grid_axes": ["blocks", "batch"],
        "source_kernel": KERNEL_NAME,
        "source_module": STATE_INDEX_PATH,
        "source_sha256": STATE_INDEX_SHA,
        "source_commit": R31_SOURCE_COMMIT,
        "source_version": R31_SOURCE_VERSION,
        "wrapper_symbol": WRAPPER_SYMBOL,
        "wrapper_lines": list(WRAPPER_LINES),
        "kernel_lines": list(KERNEL_LINES),
        "caller_path": GDN_PATH,
        "caller_sha256": GDN_SHA,
        "caller_lines": list(CALLER_LINES),
        "call_branch": CALL_BRANCH,
        "deployment_image_id": dims["identity"]["image_id"],
        "deployment_source_revision": dims["identity"]["revision"],
        "deployment_basis": dims["identity"]["basis"],
        "source_lowerings": list(dims["source_lowerings"]),
        "state_cache_rows_basis": "runtime_state_allocator",
        "state_stride0_basis": (
            "state.stride(0)>=row_size; padded page stride not declared"),
        "zero_fill_when_absent": True,
        "read_mask": "has_initial_state",
        "write_mask": "row_size_offsets",
        "state_row_bytes": row_size * dtype_bytes(dims["state_dtype"]),
        "index_rank": 1,
        "has_initial_state_dtype": "bool",
        "triton_launch": "kernel[grid](..., num_warps=8)",
    }
    if dims["state_rows"] is not None:
        attrs["state_cache_rows_declared"] = int(dims["state_rows"])
    if dims["state_stride0"] is not None:
        attrs["state_stride0_declared"] = int(dims["state_stride0"])
    if dims["stride_indices"] is not None:
        attrs["stride_indices_declared"] = int(dims["stride_indices"])
    if dims["stride_has_initial_state"] is not None:
        attrs["stride_has_initial_state_declared"] = int(
            dims["stride_has_initial_state"])
    return {
        "dtype": dims["state_dtype"],
        "layout": "ND",
        "phase": "prefill",
        "shapes": {"state": inner, "output": inner},
        "attrs": attrs,
        "occurrence_attrs": {
            "batch": "positive_int",
            "has_initial_state": "declared",
            "selected_rows": "int",
        },
        "rank_width": 4,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real R31 gather ``_CallSpec``, or ``None``.

    Selection is the exact R31 deployment identity plus the A3
    ``state_gather`` operator/branch.  Decode, the fused-chunk branch,
    another image and an undeclared inner state shape return ``None`` so the
    ordinary forecast stays unchanged for those sources.
    """
    if phase not in {"", "prefill"}:
        return None
    dims = _static_dims(compiled, phase)
    if dims is None:
        return None
    from .native_calls import _api, _call_spec, _kernel, _source

    source = _source(
        "vllm-ascend", STATE_INDEX_PATH, sha256=STATE_INDEX_SHA,
        symbol=WRAPPER_SYMBOL, revision=R31_REVISION,
        state="task_artifact")
    kernel = _kernel(
        KERNEL_NAME, "vector", path=STATE_INDEX_PATH,
        sha256=STATE_INDEX_SHA, evidence="source_declared",
        lines=list(KERNEL_LINES), block_size=STATE_IO_BLOCK_SIZE,
        num_warps=NUM_WARPS, launch="triton_jit")
    return _call_spec(
        _api("triton", KERNEL_NAME, version=R31_REVISION),
        source, _typed(dims), (kernel,), (), str(dims["name"]),
        execution_core="aiv", unresolved=("aclnn_symbol_unresolved",))


def _descriptor(shape: list[Any], dtype: str, role: str,
                *, stride0: int | None = None) -> dict:
    return {"shape": list(shape), "dtype": dtype, "layout": "ND",
            "role": role, "stride0": stride0, "pointer": None,
            "allocation": "external"}


def _prefill_requests(geometry: Any) -> tuple[Any, ...]:
    requests = tuple(getattr(geometry, "requests", ()) or ())
    return tuple(request for request in requests
                 if _text(getattr(request, "phase", "")) == "prefill")


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 state-gather occurrence for the evaluated requests.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The gather is its own physical owner; the
    chunk call and the later scatter are not charged here.
    """
    del compiled
    api = binding.api or {}
    if (api.get("entry") != KERNEL_NAME
            and api.get("aclnn_symbol") != KERNEL_NAME):
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("source_kernel") != KERNEL_NAME:
        return None
    if attrs.get("call_branch") != CALL_BRANCH:
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    requests = _prefill_requests(geometry)
    if not requests:
        return None
    batch = len(requests)
    has_initial_state = []
    for request in requests:
        context = _int_value(getattr(request, "context", 0))
        if context is None:
            return None
        has_initial_state.append(bool(context > 0))
    selected_rows = sum(1 for value in has_initial_state if value)
    heads = _int_value(attrs.get("heads"))
    dv = _int_value(attrs.get("dv"))
    dk = _int_value(attrs.get("dk"))
    row_size = _int_value(attrs.get("row_size"))
    state_dtype = _text(attrs.get("state_dtype"))
    index_dtype = _text(attrs.get("index_dtype"))
    output_dtype = _text(attrs.get("output_dtype") or state_dtype)
    if (heads is None or dv is None or dk is None or row_size is None
            or heads < 1 or dv < 1 or dk < 1 or row_size < 1
            or state_dtype not in _DTYPE_ALIASES.values()
            or index_dtype not in {"int32", "int64"}
            or output_dtype != state_dtype):
        return None
    inner = ([heads, dv, dk] if attrs.get("state_layout") == "V,K"
             else [heads, dk, dv])
    if row_size != heads * dv * dk:
        return None
    state_rows = _int_value(attrs.get("state_cache_rows_declared"))
    if state_rows is not None and state_rows < 1:
        return None
    state_stride0 = _int_value(attrs.get("state_stride0_declared"))
    stride0_declared = state_stride0 is not None
    if state_stride0 is None:
        state_stride0 = row_size
    if state_stride0 < row_size:
        return None
    stride_indices_declared = _int_value(attrs.get("stride_indices_declared"))
    stride_indices = (1 if stride_indices_declared is None
                      else stride_indices_declared)
    if stride_indices < 1:
        return None
    stride_has_declared = _int_value(
        attrs.get("stride_has_initial_state_declared"))
    stride_has = 1 if stride_has_declared is None else stride_has_declared
    if stride_has < 1:
        return None
    grid_blocks = math.ceil(row_size / STATE_IO_BLOCK_SIZE)
    if grid_blocks < 1:
        return None
    cache_key = (binding.call_id, batch, tuple(has_initial_state),
                 state_rows, state_stride0)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "state": [state_rows, *inner],
        "indices": [batch],
        "has_initial_state": [batch],
        "output": [batch, *inner],
    }
    actual_attrs = {
        **attrs,
        "batch": batch,
        "has_initial_state": has_initial_state,
        "selected_rows": selected_rows,
        "state_cache_rows": state_rows,
        "state_stride0": state_stride0,
        "stride_indices": stride_indices,
        "stride_has_initial_state": stride_has,
        "grid": [grid_blocks, batch],
        "output_rows": batch,
        "call_branch": CALL_BRANCH,
    }
    actual = {"dtype": state_dtype, "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "state": _descriptor(
            [state_rows, *inner], state_dtype, "input",
            stride0=state_stride0),
        "indices": _descriptor(
            [batch], index_dtype, "input", stride0=stride_indices),
        "has_initial_state": _descriptor(
            [batch], "bool", "input", stride0=stride_has),
        "output": _descriptor(
            [batch, *inner], output_dtype, "output", stride0=row_size),
    }
    arguments = {
        "state": "state",
        "indices": "indices",
        "has_initial_state": "has_initial_state",
        "output": "output",
        "stride_state_batch": state_stride0,
        "stride_indices": stride_indices,
        "stride_has_initial_state": stride_has,
        "row_size": row_size,
        "BLOCK_SIZE": STATE_IO_BLOCK_SIZE,
        "num_warps": NUM_WARPS,
    }
    partial_reasons: list[str] = []
    if state_rows is None:
        partial_reasons.append("state_cache_rows_runtime_unresolved")
    if not stride0_declared:
        partial_reasons.append("state_stride0_padding_unobserved")
    if stride_indices_declared is None:
        partial_reasons.append("index_stride_runtime_unobserved")
    if stride_has_declared is None:
        partial_reasons.append("has_initial_state_stride_runtime_unobserved")
    complete = not partial_reasons
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "source_lines": list(KERNEL_LINES),
        "source_commit": R31_SOURCE_COMMIT,
        "launch": "triton_jit",
        "block_size": STATE_IO_BLOCK_SIZE,
        "num_warps": NUM_WARPS,
    })
    prepare = {
        "schema": "mindie-sim/gather-ssm-states-native-prepare/1",
        "call": KERNEL_NAME,
        "wrapper": ("vllm_ascend.ops.triton.mamba.state_index."
                    + WRAPPER_SYMBOL),
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R31_SOURCE_COMMIT,
        "source_version": R31_SOURCE_VERSION,
        "source_lines": {
            "kernel": list(KERNEL_LINES),
            "wrapper": list(WRAPPER_LINES),
            "call": list(CALLER_LINES),
        },
        "kernel": kernel,
        "caller": {
            "path": GDN_PATH,
            "sha256": GDN_SHA,
            "symbol": "AscendGatedDeltaNetAttention.forward",
            "lines": list(CALLER_LINES),
            "branch": CALL_BRANCH,
            "source_commit": R31_SOURCE_COMMIT,
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": [
            "state", "indices", "has_initial_state", "output",
            "stride_state_batch", "stride_indices",
            "stride_has_initial_state", "row_size", "BLOCK_SIZE",
            "num_warps",
        ],
        "positional_arguments": [
            "state", "indices", "has_initial_state", "output",
            "stride_state_batch", "stride_indices",
            "stride_has_initial_state",
        ],
        "constexpr_arguments": {
            "row_size": row_size, "BLOCK_SIZE": STATE_IO_BLOCK_SIZE,
        },
        "grid": [grid_blocks, batch],
        "block_size": STATE_IO_BLOCK_SIZE,
        "num_warps": NUM_WARPS,
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "match": {
            "native_type": KERNEL_NAME,
            "namespace": "triton",
            "entry": KERNEL_NAME,
            "call_id": binding.call_id,
            "branch": CALL_BRANCH,
            "caller_lines": list(CALLER_LINES),
            "state_layout": attrs.get("state_layout"),
            "heads": heads, "dv": dv, "dk": dk,
            "row_size": row_size,
            "index_dtype": index_dtype,
            "state_dtype": state_dtype,
            "output_dtype": output_dtype,
            "batch": batch,
            "selected_rows": selected_rows,
            "has_initial_state": has_initial_state,
            "stride_indices": stride_indices,
            "stride_has_initial_state": stride_has,
            "source_sha256": STATE_INDEX_SHA,
            "caller_sha256": GDN_SHA,
            "source_commit": R31_SOURCE_COMMIT,
            "source_lines": list(CALLER_LINES),
        },
        "deployment": {
            "image_id": attrs.get("deployment_image_id") or R31_IMAGE_SHA,
            "revision": attrs.get("deployment_source_revision")
            or R31_REVISION,
            "selection": ("exact R31 image/revision + A3 state_gather + "
                          "gdn.py:790 fla_gdn_prefill branch"),
            "source_path": STATE_INDEX_PATH,
            "source_symbol": WRAPPER_SYMBOL,
            "source_sha256": STATE_INDEX_SHA,
            "source_commit": R31_SOURCE_COMMIT,
            "caller_path": GDN_PATH,
            "caller_sha256": GDN_SHA,
        },
        "binary_selection": {
            "status": "runtime_jit",
            "reason": "triton_kernel_compiled_at_runtime",
            "owner_kernel": KERNEL_NAME,
        },
        "complete": complete,
        "partial": not complete,
        "partial_reasons": partial_reasons,
        "unresolved": [
            "aclnn_symbol_unresolved",
            *partial_reasons,
            "triton_runtime_binary_not_executed",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_state_cache_capacity_claim",
            "no_triton_binary_claim",
        ],
        "observed_capture": {
            "relation": ("retained_observed_sample_post_prediction; "
                         "not a predictive input"),
            "path": CAPTURE_PATH,
            "sha256": CAPTURE_SHA,
            "image": CAPTURE_IMAGE,
            "kernel": KERNEL_NAME,
            "task_id": "11249",
            "kernel_line": 78,
            "input_shapes": "62438,8,128,128;1;1",
            "output_shapes": "1,8,128,128",
            "execution_core": "AI_VECTOR_CORE",
            "note": ("same source kernel/caller shape semantics; the "
                     "sample's 62438 row axis and 8 local heads are not "
                     "copied into this occurrence"),
        },
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": state_dtype, "layout": "ND",
            "shapes": actual_shapes, "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
