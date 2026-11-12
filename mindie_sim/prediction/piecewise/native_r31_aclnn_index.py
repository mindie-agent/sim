"""Prepare-only R31 ``aclnnIndex`` O0 selected-hidden invocation.

[Role] Bind the normal all-scope O0 ``hidden_select`` producer to the deployed
  vLLM-Ascend ``NPUModelRunner.execute_model`` postprocess and its observed
  ``aclnnIndex`` / ``IndexAiCore`` child.  The runner computes
  ``logits_indices`` (non-spec prefill ``query_start_loc[1:num_reqs+1]-1``;
  the spec ``spec_decode_metadata.logits_indices`` branch is an explicit
  exclusion) and executes
  ``sample_hidden_states = hidden_states[logits_indices]`` before ``lm_head``.
  Its L1634-1643 non-spec scope uses ``np.ones(num_reqs)`` and
  ``query_start_loc[1:num_reqs+1]-1``: one physical terminal row from every
  scheduled request, with invalid emissions discarded later (L1480-85,
  L3038+).  The caller supplies that physical selected geometry separately
  from the emitted/committed token count.  The prepared occurrence carries the
  producing BF16 hidden rows, the INT64 selected-logit shape and the BF16
  selected-row output.
[Invariants] Selection is the declared source-runner policy on the
  ``hidden_select`` row (class, invocation, output selection, index class,
  revision), not a model label or trace kind.  ``H`` and the producing row
  count come from the producing+physical selected ``WorkloadGeometry``; the
  selected row count is ``producing_lm_head_rows``.  Emitted/committed count
  never sizes it.  Index values are not captured in
  any case; when the runner indices are not attached the typed occurrence
  additionally records ``source_runner_indices_not_attached`` and stays
  partial.  The geometry's legitimate selected rows are not a fabricated
  source tensor.  The observed ACLNN kernel (``Index`` /
  ``IndexAiCore``) is kept distinct from the selected Python source
  (``hidden_states[logits_indices]``).  The C header/parameter order, Python
  wrapper, kernel source and the two 1-D sampler siblings are not retained.
  No tensor is allocated, no pointer or symbol is fabricated, no device launch
  is claimed and no measured duration prices or selects work.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the declared runner policy and
  the already-priced O0 copy row; a different runner/row returns ``None``.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_VERSION = "cann-9.2.0-r31"
R31_WRAPPER_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
ACLNN_API = "aclnnIndex"
KERNEL_NAME = "Index"
KERNEL_SYMBOL = "l0op::IndexAiCore"
PROFILER_NAME = "aclnnIndex_IndexAiCore_Index"
CALLER_ORIGINAL_PATH = (
    "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
CALLER_ARTIFACT = (
    "I-r31/selected/vllm-workspace/vllm-ascend/vllm_ascend/worker/"
    "model_runner_v1.py")
CALLER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/worker/model_runner_v1.py",
    "sha256": (
        "7fdd01b9e77c69a93282a7ca8e63bd80f30b3781e85a62369264efd5c312f75e"),
    "symbol": "NPUModelRunner.execute_model",
    "revision": R31_REVISION,
    "state": "task_artifact",
}
CALLER_LINE = 2633
INDICES_LINE = 1643
RUNNER_CLASS = "NPUModelRunner"
RUNNER_INVOCATION = "execute_model"
RUNNER_OUTPUT = "hidden_states[logits_indices]"
RUNNER_INDEX_CLASS = "advanced_tensor_index"
RUNNER_INDICES = "prefill: query_start_loc[1:num_reqs+1]-1"
LIBOPAPI_PATH = "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so"
LIBOPAPI_SHA256 = (
    "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456")
CSV_PATH = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
    "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz")
CSV_SHA256 = (
    "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a")
OBSERVED = {
    "cell_id": "smoke_qwen36_single_pf_4k_c1",
    "row": 2479,
    "task_id": 48981,
    "name": PROFILER_NAME,
    "type": "Index",
    "accelerator_core": "AI_VECTOR_CORE",
    "block_num": 64,
    "input_shapes": [[4096, 5120], [1], [2], [1]],
    "input_dtypes": ["DT_BF16", "INT64", "INT64", "INT64"],
    "output_shapes": [[1, 5120]],
    "output_dtypes": ["DT_BF16"],
    "grouped_retained_count": 3,
    "sibling_1d_shapes": [[4096], [1]],
    "sibling_1d_dtypes": [["DT_INT64", "DT_INT64", "DT_INT64", "DT_INT64"],
                          ["DT_INT32", "DT_INT64", "DT_INT64", "DT_INT64"]],
    "sequence_before": (
        "final_norm -> tp_all_gather -> aclnnIndex_IndexAiCore_Index "
        "[4096,5120]->[1,5120]"),
    "sequence_after": "aclnnMatmul_MatMulV3 [1,5120]x[124160,5120]->[1,124160]",
}
_UNRESOLVED = (
    "python_wrapper_not_retained",
    "index_values_not_captured",
    "acl_api_header_not_retained",
    "acl_parameter_order_unresolved",
    "index_kernel_source_not_retained",
    "kernel_binary_identity_trace_and_libopapi_only",
    "one_dimensional_sibling_indices_unresolved",
    "execution_rank_not_selected",
)
_ROW_NAMES = frozenset({"hidden_select"})
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _policy(compiled: Any) -> dict[str, Any] | None:
    """The declared source-runner policy on the O0 hidden_select row."""
    if str(getattr(compiled, "kind", "") or "") != "O0":
        return None
    if str(getattr(compiled, "variant", "") or "") != "all":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "family", "") or "") != "copy":
        return None
    if str(getattr(operator, "name", "") or "") not in _ROW_NAMES:
        return None
    if str(getattr(operator, "role", "") or "") not in {"select", "gather"}:
        return None
    try:
        theta = operator.theta
        hidden = _int(theta.find("elements", 0))
        act_bytes = float(theta.find("act_bytes", 0.0) or 0.0)
        read_bytes = float(theta.find("read_bytes", 0.0) or 0.0)
        write_bytes = float(theta.find("write_bytes", 0.0) or 0.0)
        fields = {
            "row_binding": str(theta.find("row_binding", "") or ""),
            "runner_class": str(theta.find("source_runner_class", "") or ""),
            "runner_invocation": str(
                theta.find("source_runner_invocation", "") or ""),
            "runner_indices": str(
                theta.find("source_runner_indices", "") or ""),
            "runner_output": str(
                theta.find("source_runner_output_selection", "") or ""),
            "runner_index_class": str(
                theta.find("source_runner_index_class", "") or ""),
            "runner_revision": str(
                theta.find("source_runner_revision", "") or ""),
            "indices_attached": bool(_int(
                theta.find("source_runner_indices_attached", 0))),
            "policy_source": str(
                theta.find("source_runner_policy_source", "") or ""),
        }
    except (AttributeError, TypeError, ValueError):
        return None
    if hidden < 1 or act_bytes != 2.0:
        return None
    if read_bytes and read_bytes != hidden * act_bytes:
        return None
    if write_bytes and write_bytes != hidden * act_bytes:
        return None
    if fields["row_binding"] != "selected":
        return None
    if fields["runner_class"] != RUNNER_CLASS:
        return None
    if fields["runner_invocation"] != RUNNER_INVOCATION:
        return None
    if fields["runner_output"] != RUNNER_OUTPUT:
        return None
    if fields["runner_index_class"] != RUNNER_INDEX_CLASS:
        return None
    if fields["runner_revision"] != R31_REVISION:
        return None
    if not fields["runner_indices"]:
        return None
    return {
        "hidden": hidden,
        "row_bytes": int(hidden * act_bytes),
        **fields,
    }


def _typed(policy: dict[str, Any], phase: str) -> dict[str, Any]:
    hidden = policy["hidden"]
    return {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "all",
        "shapes": {"hidden_states": [hidden]},
        "attrs": {
            "index_role": "logits_index_select",
            "row_binding": "selected",
            "python_entry": "Tensor.__getitem__",
            "python_lowering": (
                "aten::index with one 1-D INT64 index on dim 0"),
            "input_contract": ["hidden_states", "logits_indices"],
            "output_contract": ["sample_hidden_states"],
            "index_dim": 0,
            "index_dtype": "int64",
            "hidden": hidden,
            "hidden_row_bytes": policy["row_bytes"],
            "source_runner_class": policy["runner_class"],
            "source_runner_invocation": policy["runner_invocation"],
            "source_runner_indices": policy["runner_indices"],
            "source_runner_output_selection": policy["runner_output"],
            "source_runner_index_class": policy["runner_index_class"],
            "source_runner_revision": policy["runner_revision"],
            "source_runner_indices_attached": policy["indices_attached"],
            "source_runner_policy_source": policy["policy_source"],
            "source_runner_native_acl_api": ACLNN_API,
            "source_runner_observed_kernel": KERNEL_NAME,
            "source_runner_observed_profiler": PROFILER_NAME,
            "source_runner_observed_core": "aiv",
            "observed_kernel_vs_source": (
                "observed ACLNN Index/IndexAiCore child; selected Python "
                "source is hidden_states[logits_indices]"),
            "acl_api": ACLNN_API,
            "op_type": KERNEL_NAME,
            "kernel_symbol": KERNEL_SYMBOL,
            "profiler_name": PROFILER_NAME,
            "execution_core": "aiv",
            "source_lines": [INDICES_LINE, CALLER_LINE],
            "source_symbol": CALLER["symbol"],
            "source_sha256": CALLER["sha256"],
            "source_original_path": CALLER_ORIGINAL_PATH,
            "deployment_image_id": R31_IMAGE_SHA,
            "deployment_source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "acl_api_header_status": "not_retained",
            "acl_parameter_order_status": "unresolved",
            "python_wrapper_status": "not_retained",
            "kernel_source_status": "runtime_binary_only",
            "acl_binary_path": LIBOPAPI_PATH,
            "acl_binary_sha256": LIBOPAPI_SHA256,
            "acl_exported_symbols": ["aclnnIndex",
                                     "aclnnIndexGetWorkspaceSize"],
        },
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real declared source-runner ``aclnnIndex`` spec, or ``None``.

    The row's ``source_runner_class``/``invocation``/``output_selection``/
    ``index_class``/``revision`` extras are the declaration.  Model labels,
    trace kinds and image-name checks are not used.  A row with a different
    runner or revision returns ``None``; an unattached-indices declaration is
    still returned with a partial typed occurrence.
    """
    policy = _policy(compiled)
    if policy is None:
        return None
    from . import native_calls

    operator = compiled.operator
    source = native_calls._source(
        CALLER["repository"], CALLER["path"], sha256=CALLER["sha256"],
        symbol=CALLER["symbol"], revision=CALLER["revision"],
        state=CALLER["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "gather", path=LIBOPAPI_PATH, sha256=LIBOPAPI_SHA256,
        evidence="measured_capture_and_runtime_binary",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        observed_core="AI_VECTOR_CORE", observed_task_type="Index",
        observed_block_num=OBSERVED["block_num"],
        symbol=KERNEL_SYMBOL, kernel_source_status="runtime_binary_only",
        source_status="runtime_binary_only")
    return native_calls._call_spec(
        native_calls._api("aclnn", ACLNN_API, aclnn_symbol=ACLNN_API,
                          version=R31_SOURCE_VERSION),
        source, _typed(policy, phase), (kernel,), (),
        str(operator.name), execution_core="aiv",
        unresolved=_UNRESOLVED)


def _descriptor(shape: list, dtype: str, layout: str, role: str) -> dict:
    return {"shape": None if shape is None else
            [None if item is None else int(item) for item in shape],
            "dtype": dtype, "layout": layout, "role": role,
            "pointer": None, "allocation": "external"}


def _rows(geometry: Any) -> tuple[int, int] | None:
    """Producing rows and legitimate selected logit rows, or None."""
    producing = _int(getattr(geometry, "tokens", 0))
    selected = _int(getattr(geometry, "producing_lm_head_rows", lambda: 0)())
    if selected < 1:
        selected = _int(getattr(geometry, "selected_tokens", 0))
    if producing < 1 or selected < 1 or selected > producing:
        return None
    return producing, selected


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one source-runner selected-hidden occurrence.

    The producing+selected geometry carries both the producing hidden rows and
    the legitimate selected logit rows.  When the runner indices are not
    attached, the index values stay explicitly unobserved and the prepare is
    partial; no native runtime is claimed.
    """
    api = dict(getattr(binding, "api", {}) or {})
    if (api.get("entry") != ACLNN_API
            and api.get("aclnn_symbol") != ACLNN_API):
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("index_role") != "logits_index_select":
        return None
    if attrs.get("row_binding") != "selected":
        return None
    if attrs.get("acl_api") != ACLNN_API:
        return None
    if attrs.get("source_runner_revision") != R31_REVISION:
        return None
    policy = _policy(compiled)
    if policy is None:
        return None
    rows = _rows(geometry)
    if rows is None:
        return None
    producing, selected = rows
    hidden = policy["hidden"]
    indexes_attached = bool(policy["indices_attached"])
    cache_key = (binding.call_id, producing, selected, hidden,
                 indexes_attached)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    hidden_shape = [producing, hidden]
    actual_shapes = {
        "hidden_states": hidden_shape,
        "logits_indices": [selected],
        "sample_hidden_states": [selected, hidden],
    }
    actual_attrs = {
        **attrs,
        "producing_rows": producing,
        "selected_rows": selected,
        "source_rows": producing,
        "selected_rows_source": "producing_lm_head_rows",
        "index_values_status": "not_captured",
    }
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "hidden_states": _descriptor(hidden_shape, "bf16", "ND", "input"),
        "logits_indices": _descriptor([selected], "int64", "ND", "input"),
        "sample_hidden_states": _descriptor(
            [selected, hidden], "bf16", "ND", "output"),
    }
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {
        "name": KERNEL_NAME, "role": "gather",
        "evidence": "measured_capture_and_runtime_binary"}
    kernel.update({
        "observed_name": PROFILER_NAME,
        "observed_task_type": "Index",
        "observed_core": "AI_VECTOR_CORE",
        "observed_block_num": OBSERVED["block_num"],
        "symbol": KERNEL_SYMBOL,
        "source_status": "runtime_binary_only",
        "kernel_source_status": "runtime_binary_only",
    })
    captured_operands = [
        {"shape": list(shape), "dtype": dtype, "role": role}
        for shape, dtype, role in (
            (OBSERVED["input_shapes"][0], "DT_BF16", "self"),
            (OBSERVED["input_shapes"][1], "INT64", "acl_metadata_0"),
            (OBSERVED["input_shapes"][2], "INT64", "acl_metadata_1"),
            (OBSERVED["input_shapes"][3], "INT64", "index"),
        )
    ]
    partial_reasons = list(_UNRESOLVED)
    if not indexes_attached:
        partial_reasons.append("source_runner_indices_not_attached")
    prepare = {
        "schema": "mindie-sim/aclnn-index-native-prepare/1",
        "call": ACLNN_API,
        "native_entry": ACLNN_API,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": [INDICES_LINE, CALLER_LINE],
        "source_original_path": CALLER_ORIGINAL_PATH,
        "source_artifact": CALLER_ARTIFACT,
        "source_version": R31_SOURCE_VERSION,
        "wrapper_version": R31_WRAPPER_VERSION,
        "source_runner": {
            "class": policy["runner_class"],
            "invocation": policy["runner_invocation"],
            "indices": policy["runner_indices"],
            "output_selection": policy["runner_output"],
            "index_class": policy["runner_index_class"],
            "index_dim": 0,
            "index_dtype": "int64",
            "revision": policy["runner_revision"],
            "indices_attached": indexes_attached,
            "index_values_status": "not_captured",
            "policy_source": policy["policy_source"],
            "native_acl_api": ACLNN_API,
            "observed_kernel": KERNEL_NAME,
            "observed_profiler": PROFILER_NAME,
            "observed_core": "aiv",
        },
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {
            "self": "hidden_states",
            "index": "logits_indices",
            "out": "sample_hidden_states",
        },
        "argument_order": ["self", "index", "out"],
        "argument_order_status": (
            "caller_level_operands; exact aclnnIndex C parameter order "
            "not retained"),
        "captured_operands": captured_operands,
        "captured_operand_status": (
            "exact trace tensors; the two middle INT64 metadata slots have "
            "no retained ACLNN parameter role"),
        "lowering": {
            "python": "sample_hidden_states = hidden_states[logits_indices]",
            "source_file": CALLER["path"],
            "source_symbol": CALLER["symbol"],
            "source_line": CALLER_LINE,
            "index_dim": 0,
            "index_dtype": "int64",
            "index_value_expr": policy["runner_indices"],
            "index_value_source_line": INDICES_LINE,
            "index_values_status": "not_captured",
        },
        "shapes": dict(actual_shapes),
        "descriptors": descriptors,
        "arrays": descriptors,
        "optional_slots": {
            "self": "present",
            "index": "present",
            "out": "present",
        },
        "observed": {
            "cell_id": OBSERVED["cell_id"],
            "csv_path": CSV_PATH,
            "csv_sha256": CSV_SHA256,
            "row": OBSERVED["row"],
            "task_id": OBSERVED["task_id"],
            "profiler_name": PROFILER_NAME,
            "task_type": OBSERVED["type"],
            "accelerator_core": OBSERVED["accelerator_core"],
            "block_num": OBSERVED["block_num"],
            "input_shapes": OBSERVED["input_shapes"],
            "input_dtypes": OBSERVED["input_dtypes"],
            "output_shapes": OBSERVED["output_shapes"],
            "output_dtypes": OBSERVED["output_dtypes"],
            "grouped_retained_count": OBSERVED["grouped_retained_count"],
            "sibling_1d_shapes": OBSERVED["sibling_1d_shapes"],
            "sibling_1d_dtypes": OBSERVED["sibling_1d_dtypes"],
            "sequence_before": OBSERVED["sequence_before"],
            "sequence_after": OBSERVED["sequence_after"],
            "current_hidden_states_shape": list(hidden_shape),
            "current_logits_indices_shape": [selected],
            "current_output_shape": [selected, hidden],
            "current_producing_rows": producing,
            "current_selected_rows": selected,
            "sample_matches_current_geometry": (
                producing == 4096 and selected == 1 and hidden == 5120),
            "selected_rows_source": "producing_lm_head_rows",
            "index_values_status": "not_captured",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "entry": ACLNN_API,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "deployment_image_id": R31_IMAGE_SHA,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "source_sha256": source.get("sha256"),
            "source_symbol": CALLER["symbol"],
            "source_lines": [INDICES_LINE, CALLER_LINE],
            "hidden": hidden,
            "producing_rows": producing,
            "selected_rows": selected,
            "indices_attached": indexes_attached,
            "index_values_status": "not_captured",
            "index_dtype": "int64",
            "dtype": "bf16",
            "layout": "ND",
            "profiler_name": PROFILER_NAME,
            "complete": False,
            "partial": True,
        },
        "deployment": {
            "image_id": R31_IMAGE_SHA,
            "revision": R31_REVISION,
            "selection": (
                "declared source runner NPUModelRunner.execute_model + "
                "hidden_states[logits_indices] + observed aclnnIndex child"),
            "source_path": CALLER["path"],
            "source_original_path": CALLER_ORIGINAL_PATH,
            "source_symbol": CALLER["symbol"],
            "source_sha256": CALLER["sha256"],
            "source_lines": [INDICES_LINE, CALLER_LINE],
            "source_version": R31_SOURCE_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "acl_binary_path": LIBOPAPI_PATH,
            "acl_binary_sha256": LIBOPAPI_SHA256,
            "acl_api_header_status": "not_retained",
            "acl_parameter_order_status": "unresolved",
            "python_wrapper_status": "not_retained",
            "kernel_source_status": "runtime_binary_only",
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "python_aten_dispatch_not_retained",
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "acl_symbols": ["aclnnIndex", "aclnnIndexGetWorkspaceSize"],
            "internal_symbol": KERNEL_SYMBOL,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": partial_reasons,
        "unresolved": partial_reasons,
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_acl_api_header_claim",
            "no_argument_order_claim",
            "no_index_value_claim",
            "no_physical_completion",
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


def observed_rows() -> dict[str, dict]:
    """The original R31 PF4kC1 O0 index sample, retained for status/probes."""
    return {
        "O0:copy:hidden_select": {
            "source": [4096, 5120],
            "index": [1],
            "output": [1, 5120],
            "source_dtype": "bf16",
            "index_dtype": "int64",
            "output_dtype": "bf16",
            "profiler_name": PROFILER_NAME,
            "acl_api": ACLNN_API,
            "kernel": KERNEL_NAME,
            "caller": CALLER["symbol"],
            "caller_line": CALLER_LINE,
            "indices_line": INDICES_LINE,
            "complete": False,
            "unresolved": list(_UNRESOLVED),
        }
    }
