"""Prepare-only R31 GDN ``scatter_ssm_states_`` Triton invocation.

[Role] Expand the selected R31 ``scatter_ssm_states_`` call in
  ``vllm_ascend/ops/gdn.py:833`` into the exact
  ``_scatter_ssm_states_kernel`` arguments, row size, grid and
  destination/source/index descriptors for the evaluated prefill request
  vector.
[Invariants] Only the exact R31 deployment identity and the prefill
  post-chunk state-write branch are admitted.  The helper source, kernel body,
  argument order, block size, warp count and grid derivation come from the
  retained R31 ``vllm_ascend/ops/triton/mamba/state_index.py`` blob.  State
  inner dimensions, state dtype and index dtype are static; batch,
  ``state.stride(0)``, index/source strides and the state-cache row axis stay
  occurrence facts.  A missing cache capacity or stride remains
  ``None``/unresolved and is never replaced by a captured row count, a model
  label or a default.  This is a state **write** distinct from the gather and
  from the chunk call's internal Cast/Transpose children; no complete runtime
  execution is claimed.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure host metadata derived from the selected source and
  the existing compiled operator; no tensor allocation, device launch, runtime
  import or measured duration.
"""
from __future__ import annotations

import re
from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_COMMIT = "8fd0bc4885cb52e58791aaa8efacc571b9bd115d"
QWEN36_IMAGE_SHA = (
    "c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4")
STATE_INDEX_PATH = "vllm_ascend/ops/triton/mamba/state_index.py"
STATE_INDEX_SHA = (
    "03debd381d4101279fff9f9cf8216ccc2b92666e2246427519a0f57fea8d7d66")
GDN_PATH = "vllm_ascend/ops/gdn.py"
GDN_SHA = "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed"
QWEN36_GDN_SHA = (
    "e9ed210b4ca4431ee40916f06f0272e5e781eb04b61b084bacea216799fbca98")
ENTRY = "_scatter_ssm_states_kernel"
WRAPPER_SYMBOL = "scatter_ssm_states_"
KERNEL_NAME = "_scatter_ssm_states_kernel"
ROLE = "post_chunk_prefill_state_publish"
SCHEMA = "mindie-sim/scatter-ssm-states-native-prepare/1"
STATE_IO_BLOCK_SIZE = 1024
NUM_WARPS = 8
WRAPPER_LINES = (170, 198)
KERNEL_LINES = (47, 71)
FLA_BRANCH = "fla_gdn_prefill"
FUSED_BRANCH = "use_fused_chunk"
CALLER_LINES = (806, 810)
ALTERNATE_CALLER_LINES = (833, 837)
BRANCH_LINES = (790, 810)
FUSED_BRANCH_LINES = (815, 837)
FLA_SOURCE_PREP = ("last_recurrent_state.to(ssm_state.dtype)"
                   ".contiguous()")
FUSED_SOURCE_PREP = "last_recurrent_state"
_SOURCE_BRANCHES = {
    FLA_BRANCH: {
        "source_lines": CALLER_LINES,
        "alternate_lines": ALTERNATE_CALLER_LINES,
        "branch_lines": BRANCH_LINES,
        "preparation": FLA_SOURCE_PREP,
        "contiguity": "caller_contiguous",
        "dtype_policy": "state_dtype",
    },
    FUSED_BRANCH: {
        "source_lines": ALTERNATE_CALLER_LINES,
        "alternate_lines": CALLER_LINES,
        "branch_lines": FUSED_BRANCH_LINES,
        "preparation": FUSED_SOURCE_PREP,
        "contiguity": "unretained_branch_operand",
        "dtype_policy": "declared_or_unresolved",
    },
}
SOURCE = {
    "repository": "vllm-ascend",
    "path": STATE_INDEX_PATH,
    "sha256": STATE_INDEX_SHA,
    "symbol": WRAPPER_SYMBOL,
    "revision": R31_REVISION,
    "state": "task_artifact",
}
SOURCE_LINES = CALLER_LINES
_DTYPE_ALIASES = {
    "float32": "fp32", "fp32": "fp32",
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16",
}
_SCATTER_NAMES = frozenset({"scatter_recurrent_state", "state_scatter", "scatter"})
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Any) -> str:
    text = _text(value).lower()
    match = re.search(r"sha256:([0-9a-f]{12,64})", text)
    if match:
        return match.group(1)
    return text


def _dtype(value: Any) -> str | None:
    return _DTYPE_ALIASES.get(_text(value).lower())


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _metadata(compiled: Any) -> dict[str, Any]:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _r31_identity(compiled: Any) -> dict[str, str] | None:
    """The exact R31 deployment identity; another image stays unbound."""
    meta = _metadata(compiled)
    for key in ("image_id", "oci_image_digest", "deployment_image_id"):
        digest = _digest(meta.get(key))
        if digest.startswith(R31_IMAGE_PREFIX):
            return {"image_id": digest, "basis": f"feature_metadata.{key}"}
    for key in ("image", "deployment_image", "oci_image", "source_image"):
        if R31_IMAGE_PREFIX in _text(meta.get(key)).lower():
            return {"image_id": R31_IMAGE_SHA,
                    "basis": f"feature_metadata.{key}"}
    for key in ("deployment_source_revision", "source_revision",
                "mx_quant_source_revision"):
        if _text(meta.get(key)).lower().startswith("r31"):
            return {"image_id": R31_IMAGE_SHA,
                    "basis": f"feature_metadata.{key}"}
    return None


def _lane_dims(theta: Any) -> tuple[int, int, int] | None:
    heads = dict(getattr(theta, "heads", ()) or ())
    extras = dict(getattr(theta, "extras", ()) or ())
    values = (_positive_int(extras.get("value_heads"))
              or _positive_int(heads.get("kv")))
    dk = _positive_int(extras.get("dk")) or _positive_int(heads.get("dn"))
    dv = (_positive_int(extras.get("dv"))
          or _positive_int(heads.get("dv")) or dk)
    if values is None or dk is None or dv is None:
        return None
    return values, dk, dv


def _state_dims(compiled: Any) -> dict[str, Any] | None:
    """The source-declared state lanes and runtime axes, or ``None``."""
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = getattr(operator, "theta", None)
    if theta is None:
        return None
    extras = dict(getattr(theta, "extras", ()) or ())
    dims = _lane_dims(theta)
    if dims is None:
        for sibling in getattr(compiled, "siblings", ()) or ():
            if str(getattr(sibling, "name", "") or "") not in {
                    "gated_delta", "delta"}:
                continue
            dims = _lane_dims(getattr(sibling, "theta", None))
            if dims is not None:
                break
    if dims is None:
        return None
    values, dk, dv = dims
    state_dtype = None
    for key in ("state_dtype", "recurrent_state_dtype", "destination_dtype"):
        state_dtype = _dtype(extras.get(key)) or state_dtype
    if state_dtype is None:
        return None
    index_dtype = _text(extras.get("index_dtype")).lower()
    if index_dtype not in {"int32", "int64"}:
        return None
    layout = _text(extras.get("state_layout")).upper().replace(" ", "")
    if layout not in {"V,K", "K,V"}:
        return None
    inner = [values, dv, dk] if layout == "V,K" else [values, dk, dv]
    row_size = values * dv * dk
    row_bytes = theta.find("row_bytes", None)
    if row_bytes is not None:
        try:
            declared = int(row_bytes)
        except (TypeError, ValueError):
            return None
        if declared != row_size * int(_dtype_bytes(state_dtype)):
            return None
    branch = _text(extras.get("scatter_branch"))
    if branch not in _SOURCE_BRANCHES:
        return None
    branch_record = dict(_SOURCE_BRANCHES[branch])
    declared_source_dtype = _dtype(extras.get("source_dtype"))
    if branch == FLA_BRANCH:
        if (declared_source_dtype is not None
                and declared_source_dtype != state_dtype):
            return None
        source_dtype = state_dtype
        source_dtype_status = "state_dtype_from_source_cast"
    else:
        source_dtype = declared_source_dtype
        source_dtype_status = ("explicit" if source_dtype is not None
                               else "unresolved_chunk_final_state_dtype")
    source_kernel = _text(extras.get("source_kernel", KERNEL_NAME))
    if source_kernel and source_kernel != KERNEL_NAME:
        return None
    state_rows = _positive_int(extras.get("state_cache_rows")
                               or extras.get("state_rows"))
    state_stride0 = _positive_int(extras.get("state_stride0")
                                  or extras.get("state_stride0_elements"))
    indices_stride0 = _positive_int(extras.get("indices_stride0"))
    source_stride0 = _positive_int(extras.get("source_stride0")
                                   or extras.get("source_stride0_elements"))
    if state_stride0 is not None and state_stride0 < row_size:
        return None
    if source_stride0 is not None:
        if branch == FLA_BRANCH:
            if source_stride0 != row_size:
                return None
        elif source_stride0 < row_size:
            return None
    return {
        "operator": operator,
        "name": str(getattr(operator, "name", "") or ""),
        "values": values, "dk": dk, "dv": dv,
        "state_layout": layout, "inner": inner, "row_size": row_size,
        "state_dtype": state_dtype, "index_dtype": index_dtype,
        "source_dtype": source_dtype, "source_kernel": source_kernel,
        "branch": branch,
        "source_lines": list(branch_record["source_lines"]),
        "alternate_lines": list(branch_record["alternate_lines"]),
        "branch_lines": list(branch_record["branch_lines"]),
        "source_preparation": branch_record["preparation"],
        "source_contiguity": branch_record["contiguity"],
        "source_dtype_status": source_dtype_status,
        "state_rows": state_rows, "state_stride0": state_stride0,
        "indices_stride0": indices_stride0, "source_stride0": source_stride0,
    }


def _dtype_bytes(dtype: str) -> int:
    return {"fp32": 4, "int32": 4, "bf16": 2, "fp16": 2, "int64": 8}[dtype]


def _is_state_scatter(operator: Any) -> bool:
    family = str(getattr(operator, "family", "") or "")
    name = str(getattr(operator, "name", "") or "")
    role = str(getattr(getattr(operator, "template", None), "role", "") or "")
    if family == "state_scatter":
        return (name in _SCATTER_NAMES
                or ("scatter" in name and ("state" in name or "recurrent" in name)))
    if family != "recurrent":
        return False
    # An alternate normal-entry attachment may keep the recurrent family and
    # mark the call explicitly.  A bare ``tp_reduce_scatter`` never matches.
    extras = dict(getattr(getattr(operator, "theta", None), "extras", ()) or ())
    marker = " ".join(str(extras.get(key) or "") for key in (
        "native_kind", "source_kernel", "call_kind", "role")).lower()
    return (name in _SCATTER_NAMES or role == "scatter" or "scatter" in name) \
        and "scatter" in marker and "gather" not in marker


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 post-chunk state scatter call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Decode, a non-A3 operator, a gather, an
    unfamiliar image digest and missing declared state lanes return ``None``.
    """
    if str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None or not _is_state_scatter(operator):
        return None
    if str(phase or "") != "prefill":
        return None
    identity = _r31_identity(compiled)
    if identity is None:
        return None
    dims = _state_dims(compiled)
    if dims is None:
        return None
    values, dk, dv = dims["values"], dims["dk"], dims["dv"]
    row_size = dims["row_size"]
    state_dtype = dims["state_dtype"]
    index_dtype = dims["index_dtype"]
    source_dtype = dims["source_dtype"]
    state_rows = dims["state_rows"]
    state_stride0 = dims["state_stride0"]
    indices_stride0 = dims["indices_stride0"]
    source_stride0 = dims["source_stride0"]
    branch = dims["branch"]
    source_lines = dims["source_lines"]
    alternate_lines = dims["alternate_lines"]
    branch_lines = dims["branch_lines"]
    source_preparation = dims["source_preparation"]
    source_contiguity = dims["source_contiguity"]
    source_dtype_status = dims["source_dtype_status"]
    from . import native_calls

    api = native_calls._api("triton", ENTRY, version=R31_REVISION)
    source = native_calls._source(
        "vllm-ascend", STATE_INDEX_PATH, sha256=STATE_INDEX_SHA,
        symbol=WRAPPER_SYMBOL, revision=R31_REVISION, state="task_artifact")
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", path=STATE_INDEX_PATH, sha256=STATE_INDEX_SHA,
        evidence="source_declared", lines=list(KERNEL_LINES),
        block_size=STATE_IO_BLOCK_SIZE, num_warps=NUM_WARPS,
        launch="triton_jit", wrapper_lines=list(WRAPPER_LINES),
        source_status="kernel_source_retained")
    typed = {
        "dtype": state_dtype,
        "layout": "ND",
        "phase": "prefill",
        "shapes": {
            "source": list(dims["inner"]),
            "state": list(dims["inner"]),
        },
        "attrs": {
            "call_kind": "state_scatter",
            "native_kind": "state_scatter",
            "source_entry": WRAPPER_SYMBOL,
            "source_kernel": KERNEL_NAME,
            "wrapper_symbol": WRAPPER_SYMBOL,
            "source_module": STATE_INDEX_PATH,
            "source_sha256": STATE_INDEX_SHA,
            "source_commit": R31_SOURCE_COMMIT,
            "wrapper_lines": list(WRAPPER_LINES),
            "kernel_lines": list(KERNEL_LINES),
            "caller_path": GDN_PATH,
            "caller_sha256": GDN_SHA,
            "caller_lines": list(source_lines),
            "source_lines": list(source_lines),
            "alternate_caller_lines": list(alternate_lines),
            "alternate_source_lines": list(alternate_lines),
            "branch": branch,
            "scatter_branch": branch,
            "alternate_branch": (FUSED_BRANCH if branch == FLA_BRANCH
                                 else FLA_BRANCH),
            "branch_lines": list(branch_lines),
            "source_preparation": source_preparation,
            "source_contiguity": source_contiguity,
            "source_cast_to_state_dtype": branch == FLA_BRANCH,
            "state_layout": dims["state_layout"],
            "state_inner_shape": list(dims["inner"]),
            "value_heads": values,
            "dk": dk,
            "dv": dv,
            "row_size": row_size,
            "state_row_bytes": row_size * _dtype_bytes(state_dtype),
            "state_dtype": state_dtype,
            "index_dtype": index_dtype,
            "source_dtype": source_dtype,
            "source_operand_dtype": source_dtype,
            "source_dtype_status": source_dtype_status,
            "source_dtype_policy": ("state_dtype_cast" if branch == FLA_BRANCH
                                    else "chunk_operand_declared_or_unresolved"),
            "block_size": STATE_IO_BLOCK_SIZE,
            "num_warps": NUM_WARPS,
            "grid_blocks": ((row_size + STATE_IO_BLOCK_SIZE - 1) // STATE_IO_BLOCK_SIZE),
            "grid_axes": ["blocks", "batch"],
            "state_rows": state_rows,
            "state_rows_status": ("explicit" if state_rows is not None
                                  else "unresolved"),
            "state_rows_reason": (None if state_rows is not None else
                                  "ssm_state_cache_capacity_not_declared"),
            "state_stride0": state_stride0,
            "state_stride0_status": ("explicit" if state_stride0 is not None
                                     else "unresolved"),
            "state_stride0_lower_bound": row_size,
            "indices_stride0": indices_stride0,
            "indices_stride0_status": ("explicit" if indices_stride0 is not None
                                       else "unresolved"),
            "source_stride0": source_stride0,
            "source_stride0_status": (
                "explicit" if source_stride0 is not None
                else ("caller_contiguous_unobserved" if branch == FLA_BRANCH
                      else "unretained_branch_operand")),
            "source_stride0_lower_bound": row_size,
            "strides_retained": False,
            "gather_distinct_from_gather": True,
            "gather_argument_present": False,
            "deployment_image_id": identity["image_id"],
            "deployment_source_revision": R31_REVISION,
            "deployment_basis": identity["basis"],
            "independent_caller_sha256": QWEN36_GDN_SHA,
            "independent_caller_lines": [677, 681],
        },
        "rank_width": 4,
    }
    unresolved = ["binary_selection_unresolved"]
    if state_rows is None:
        unresolved.append("state_cache_rows_unresolved")
    if state_stride0 is None:
        unresolved.append("state_stride0_unresolved")
    if indices_stride0 is None:
        unresolved.append("indices_stride0_unresolved")
    if source_stride0 is None:
        unresolved.append("source_stride0_unresolved")
    if source_dtype is None:
        unresolved.append("source_dtype_unresolved")
    if branch == FLA_BRANCH:
        unresolved = [reason for reason in unresolved
                      if reason != "source_dtype_unresolved"]
    return native_calls._call_spec(
        api, source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv", unresolved=unresolved)


def _descriptor(shape: list, dtype: str | None, layout: str, role: str,
                **extra: Any) -> dict:
    row = {
        "shape": [None if item is None else int(item) for item in shape],
        "dtype": dtype, "layout": layout, "role": role,
        "pointer": None, "allocation": "external",
    }
    row.update(extra)
    return row


def _observed_capture(branch: str = FLA_BRANCH) -> dict:
    """The true R31 original integration row; evidence, not an input."""
    return {
        "path": ("/private/tmp/mindie-7337-20261111-1457/"
                 "integration-r31-a3-input-source.json"),
        "sha256": ("a523936e982e4fd45b80071e906276d17d18fac03c5bfaca903e2"
                   "f1158f778d1"),
        "underlying_capture": (
            "/Users/maoxx241/code/mindie-agent/.workstreams/"
            "a5-multinode-fleet-rack06-20261107/out/smoke_qwen36_single/"
            "pf_4k_c1/177.25.28.1/rank0_545_20261107032704817_ascend_pt/"
            "kernel_details.csv.gz"),
        "kernel": KERNEL_NAME,
        "counts": {"_gather_ssm_states_kernel": 48,
                   "_scatter_ssm_states_kernel": 48},
        "geometry": {"phase": "prefill", "prompt_tokens": 4096, "batch": 1,
                     "value_heads": 24, "dk": 128, "dv": 128,
                     "state_rows": 2375, "state_dtype": "fp32"},
        "scatter_example": {
            "input_shapes": [[1], [1, 24, 128, 128]],
            "input_dtypes": ["INT32", "FLOAT"],
            "output_shapes": [[2375, 24, 128, 128]],
            "output_dtypes": ["FLOAT"],
        },
        "first_order": [[58, "_gather_ssm_states_kernel"],
                        [66, "_scatter_ssm_states_kernel"],
                        [98, "_gather_ssm_states_kernel"],
                        [106, "_scatter_ssm_states_kernel"]],
        "capture_branch": FLA_BRANCH,
        "selected_branch": branch,
        "capture_matches_selected_branch": branch == FLA_BRANCH,
        "separation": ("the scatter is a distinct kernel after the FLA "
                       "chunk, not a member of the chunk call"),
        "predictive_input": False,
        "note": ("true R31 original integration source (TP2 PF4096 B1); its "
                 "2375 cache rows and observed dtype are evidence only and "
                 "are not used as descriptor defaults"),
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected state scatter.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The scatter call owns its single physical
    charge; the separate gather and the signed chunk child scopes are not
    merged into this occurrence.
    """
    api = binding.api or {}
    if api.get("namespace") != "triton" or api.get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("call_kind") != "state_scatter"
            or attrs.get("source_kernel") != KERNEL_NAME):
        return None
    branch = _text(attrs.get("scatter_branch"))
    if branch not in _SOURCE_BRANCHES:
        return None
    source_preparation = _text(attrs.get("source_preparation"))
    source_contiguity = _text(attrs.get("source_contiguity"))
    source_dtype_status = _text(attrs.get("source_dtype_status"))
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    if not requests:
        return None
    lengths = tuple(int(getattr(request, "q_len", 0) or 0)
                    for request in requests)
    if not lengths or min(lengths) < 1:
        return None
    batch = len(requests)
    values = _positive_int(attrs.get("value_heads"))
    dk = _positive_int(attrs.get("dk"))
    dv = _positive_int(attrs.get("dv"))
    row_size = _positive_int(attrs.get("row_size"))
    state_dtype = _dtype(attrs.get("state_dtype"))
    index_dtype = _text(attrs.get("index_dtype")).lower()
    source_dtype = _dtype(attrs.get("source_dtype"))
    if branch == FLA_BRANCH and source_dtype != state_dtype:
        return None
    block_size = _positive_int(attrs.get("block_size"))
    num_warps = _positive_int(attrs.get("num_warps"))
    grid_blocks = _positive_int(attrs.get("grid_blocks"))
    if (values is None or dk is None or dv is None or row_size is None
            or state_dtype is None or index_dtype not in {"int32", "int64"}
            or block_size is None or num_warps is None or grid_blocks is None):
        return None
    if row_size != values * dv * dk or grid_blocks != (row_size + block_size - 1) // block_size:
        return None
    state_rows = _positive_int(attrs.get("state_rows"))
    state_stride0 = _positive_int(attrs.get("state_stride0"))
    indices_stride0 = _positive_int(attrs.get("indices_stride0"))
    source_stride0 = _positive_int(attrs.get("source_stride0"))
    if state_stride0 is not None and state_stride0 < row_size:
        return None
    if source_stride0 is not None:
        if branch == FLA_BRANCH and source_stride0 != row_size:
            return None
        if branch == FUSED_BRANCH and source_stride0 < row_size:
            return None
    source_stride0_status = (
        "explicit" if source_stride0 is not None
        else (_text(attrs.get("source_stride0_status"))
              or ("caller_contiguous_unobserved" if branch == FLA_BRANCH
                  else "unretained_branch_operand")))
    inner = list(attrs.get("state_inner_shape") or [values, dv, dk])
    if [int(item) for item in inner] != [values, dv, dk] and [
            int(item) for item in inner] != [values, dk, dv]:
        return None
    cache_key = (binding.call_id, branch, batch, lengths, values, dk, dv,
                 state_rows, state_stride0, indices_stride0, source_stride0,
                 source_dtype)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    state_shape = ([state_rows if state_rows is not None else None, *inner])
    source_shape = [batch, *inner]
    actual_shapes = {
        "indices": [batch],
        "source": source_shape,
        "state": state_shape,
    }
    actual_attrs = {
        **attrs,
        "batch": batch,
        "lengths": list(lengths),
        "tokens": sum(lengths),
        "selected_rows": batch,
        "grid": [grid_blocks, batch],
        "state_rows": state_rows,
        "state_rows_status": ("explicit" if state_rows is not None
                              else "unresolved"),
        "state_rows_reason": (None if state_rows is not None else
                              "ssm_state_cache_capacity_not_declared"),
        "state_stride0": state_stride0,
        "state_stride0_status": ("explicit" if state_stride0 is not None
                                 else "unresolved"),
        "indices_stride0": indices_stride0,
        "indices_stride0_status": ("explicit" if indices_stride0 is not None
                                   else "unresolved"),
        "source_stride0": source_stride0,
        "source_stride0_status": source_stride0_status,
        "source_preparation": source_preparation,
        "source_contiguity": source_contiguity,
        "source_dtype_status": source_dtype_status,
        "source_operand_dtype": source_dtype,
        "source_cast_to_state_dtype": branch == FLA_BRANCH,
        "scatter_branch": branch,
        "destination_layout": attrs.get("state_layout"),
    }
    actual = {"dtype": state_dtype, "layout": "ND", "shapes": actual_shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "state": _descriptor(
            state_shape, state_dtype, "ND", "inout_destination",
            state_layout=attrs.get("state_layout"), mutated=True,
            row_size=row_size, stride0=state_stride0,
            stride0_status=("explicit" if state_stride0 is not None
                            else "unresolved")),
        "indices": _descriptor(
            [batch], index_dtype, "ND", "input",
            index_layout="packed_prefill_state_indices",
            stride0=indices_stride0,
            stride0_status=("explicit" if indices_stride0 is not None
                            else "unresolved")),
        "source": _descriptor(
            source_shape, source_dtype, "ND", "input",
            state_layout=attrs.get("state_layout"),
            source_kind=("fla_contiguous_state_dtype" if branch == FLA_BRANCH
                         else "fused_chunk_final_state"),
            preparation=source_preparation, contiguity=source_contiguity,
            stride0=source_stride0, stride0_status=source_stride0_status),
    }
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "accelerator_core": "AI_VECTOR_CORE",
        "block_size": block_size, "num_warps": num_warps,
        "source_status": "kernel_source_retained",
    })
    prepare = {
        "schema": SCHEMA,
        "call": f"{STATE_INDEX_PATH[:-3].replace('/', '.')}.{WRAPPER_SYMBOL}",
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": dict(binding.source or {}),
        "caller_source": {
            "path": attrs.get("caller_path"),
            "sha256": attrs.get("caller_sha256"),
            "lines": attrs.get("source_lines"),
            "branch": branch,
            "branch_lines": attrs.get("branch_lines"),
            "alternate_lines": attrs.get("alternate_source_lines"),
        },
        "source_operand": {
            "expression": source_preparation,
            "lines": attrs.get("source_lines"),
            "dtype": source_dtype,
            "dtype_status": source_dtype_status,
            "contiguity": source_contiguity,
            "stride0": source_stride0,
            "stride0_status": source_stride0_status,
            "cast_to_state_dtype": branch == FLA_BRANCH,
        },
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "prototype": f"{WRAPPER_SYMBOL}(state, indices, source) -> None",
        "prototype_status": "source_declared",
        "source_validation": [
            "state.ndim >= 2",
            "state.shape[0] > 0",
            "indices.ndim == 1",
            "indices.dtype in {int32,int64}",
            "indices.device == state.device",
            "state.stride(0) >= row_size",
            "state[0].is_contiguous()",
            "source.shape == (indices.numel(), *state.shape[1:])",
            "source.device == state.device",
            "source[0].is_contiguous()",
            *(["source = " + FLA_SOURCE_PREP,
               "source.dtype == state.dtype"]
              if branch == FLA_BRANCH else
              ["source = " + FUSED_SOURCE_PREP,
               "source.dtype is the chunk final-state dtype"]),
        ],
        "arguments": {
            "state": "state", "indices": "indices", "source": "source",
        },
        "argument_order": ["state", "indices", "source"],
        "kernel_arguments": {
            "state_ptr": "state", "indices_ptr": "indices",
            "source_ptr": "source",
            "stride_state_batch": state_stride0,
            "stride_indices": indices_stride0,
            "stride_source_batch": source_stride0,
            "row_size": row_size, "BLOCK_SIZE": block_size,
        },
        "kernel_argument_order": [
            "state_ptr", "indices_ptr", "source_ptr",
            "stride_state_batch", "stride_indices", "stride_source_batch",
            "row_size", "BLOCK_SIZE",
        ],
        "launch_options": {"num_warps": num_warps},
        "launch": (f"{KERNEL_NAME}[({grid_blocks}, {batch})]("
                   f"state, indices, source, stride_state_batch, "
                   f"stride_indices, stride_source_batch, "
                   f"row_size={row_size}, BLOCK_SIZE={block_size}, "
                   f"num_warps={num_warps})"),
        "grid": [grid_blocks, batch],
        "block_size": block_size,
        "num_warps": num_warps,
        "shapes": dict(validated.get("shapes") or {}),
        "full_shapes": dict(actual_shapes),
        "descriptors": descriptors,
        "arrays": descriptors,
        "state_cache": {
            "shape": state_shape, "rows": state_rows,
            "row_size": row_size, "stride0": state_stride0,
            "status": ("explicit" if state_rows is not None else "unresolved"),
            "reason": (None if state_rows is not None else
                       "ssm_state_cache_capacity_not_declared"),
            "stride0_status": ("explicit" if state_stride0 is not None
                               else "unresolved"),
            "mutated_rows": batch,
        },
        "observed_capture": _observed_capture(branch),
        "match": {
            "native_type": "StateScatter",
            "entry": ENTRY,
            "kernel": KERNEL_NAME,
            "wrapper_symbol": WRAPPER_SYMBOL,
            "call_id": binding.call_id,
            "phase": "prefill",
            "batch": batch,
            "lengths": list(lengths),
            "value_heads": values, "dk": dk, "dv": dv,
            "row_size": row_size, "state_rows": state_rows,
            "state_layout": attrs.get("state_layout", "V,K"),
            "state_dtype": state_dtype, "index_dtype": index_dtype,
            "source_dtype": source_dtype,
            "source_dtype_status": source_dtype_status,
            "source_preparation": source_preparation,
            "source_contiguity": source_contiguity,
            "source_lines": attrs.get("source_lines"),
            "branch": branch,
            "deployment_image_id": attrs.get("deployment_image_id"),
            "deployment_source_revision": attrs.get(
                "deployment_source_revision"),
        },
        "deployment": {
            "image_id": attrs.get("deployment_image_id"),
            "revision": attrs.get("deployment_source_revision"),
            "selection": "exact R31 image identity",
            "source_path": STATE_INDEX_PATH,
            "source_sha256": STATE_INDEX_SHA,
            "source_commit": R31_SOURCE_COMMIT,
            "caller_path": attrs.get("caller_path"),
            "caller_sha256": attrs.get("caller_sha256"),
            "caller_lines": attrs.get("source_lines"),
            "branch": branch,
            "independent_caller_sha256": attrs.get("independent_caller_sha256"),
            "independent_caller_lines": attrs.get("independent_caller_lines"),
        },
        "binary_selection": {
            "status": "unresolved",
            "reason": "triton_jit_kernel_binary_not_retained",
            "owner_kernel": KERNEL_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            *(["state_cache_rows_unresolved"] if state_rows is None else []),
            *(["state_stride0_unresolved"] if state_stride0 is None else []),
            *(["indices_stride0_unresolved"] if indices_stride0 is None else []),
            *(["source_stride0_unresolved"] if source_stride0 is None else []),
            *(["source_dtype_unresolved"] if source_dtype is None else []),
            "runtime_execution_not_performed",
        ],
        "unresolved": list(binding.unresolved),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_complete_native_execution",
            "no_state_cache_capacity_claim", "no_runtime_stride_claim",
            "no_retained_operand_stride", "no_guessed_state_layout",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": validated.get("dtype"),
            "layout": validated.get("layout"),
            "shapes": validated.get("shapes"),
            "attrs": validated.get("attrs"),
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
