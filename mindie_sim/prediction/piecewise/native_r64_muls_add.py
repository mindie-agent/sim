"""Prepare-only R64 ``muls_add`` fusion from the source-selected F4 join.

[Role] Bind the R64 Triton ``muls_add_kernel`` to the normal F4
  ``elementwise.join`` recipe only when that recipe is the source-selected
  fused ``x * routed_scaling_factor + y`` combine.  The recipe's own
  ``combine_variant``/``scale_owner``/``routed_scaling_factor``,
  ``ops_per_element``, ``elements`` and read/write bytes decide fused
  ``muls_add`` vs plain add vs scale-owned-by-F1; the supplied FeatureConfig
  ``routed_scaling_factor`` is the actual value.  H, dtype, the token domain
  and the selected variant are not inferred from a model name, phase, shape
  or scale presence alone.
[Invariants] Only the R64 source revision and the source-selected F4 join
  recipe are admitted.  ``muls_add`` requires ``combine_variant=muls_add``,
  ``scale_owner=join``, ``ops_per_element=2`` and a declared scale > 1.
  ``plain_add``, ``scale_owner=router``, an identity scale and an unresolved
  variant return ``None`` (truthful absent; one fee stays on the normal F4
  join).  A shared ``token_domain`` admits both prefill and decode; a
  phase-specific source route gates only that phase.  No captured block
  number, observed clock, sample shape, external config or file read is a
  selector or a predictive supply.
[Timeouts & Locks] Pure bounded metadata arithmetic; a 32-entry geometry
  memo; no I/O, waits or locks.
[Side-Effect Policy] Read-only metadata; no tensor allocation, device import,
  file read/write or duration change to the existing join cost.
"""
from __future__ import annotations

from typing import Any

R64_REVISION = "r64"
R64_VERSION = "r64/8d422528"
ENTRY = "torch.ops.vllm.muls_add"
WRAPPER_ENTRY = "vllm_ascend.ops.triton.muls_add.muls_add_triton"
KERNEL_NAME = "muls_add_kernel"
OPERATION = "muls_add"
_FLOAT_BYTES = {
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
    "fp32": 4, "float32": 4, "fp8_e4m3": 1, "fp8": 1,
}
_BYTES_DTYPE = {1: "fp8_e4m3", 2: "bf16", 4: "fp32"}
WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/triton/muls_add.py",
    "sha256": "cd1845442d11fecd7f9de1ee1020e457a40483348bb511150039e51bb7791c20",
    "symbol": "muls_add_triton/muls_add_kernel",
    "revision": "task-local/1933f86c",
    "state": "task_artifact",
    "source_status": "not_retained_in_r64_selected_subset",
    "abi": (
        "muls_add_kernel[(num_programs,)](x, y, output, scale, n_elements, "
        "n_blocks, BLOCK_SIZE=BLOCK_SIZE); output = x * scale + y"),
}
IMAGE_CONFIG_REFERENCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ascend_config.py",
    "sha256": "6af63d7f66000d2a195732ef25dba3d4d47fd638e96a82f553a937fa0b40b047",
    "lines": [127, 459],
    "symbol": "AscendCompilationConfig.fuse_muls_add",
    "status": "retained_same_image_supplement",
    "use": "source_evidence_only_not_read_by_product",
}
COMBINE_SOURCE_REFERENCE = {
    "glm4_moe": {
        "path": "vllm/model_executor/models/glm4_moe.py",
        "lines": [195, 218],
        "combine": (
            "final_hidden_states * routed_scaling_factor + shared_output"),
        "f1_factor": 1.0,
        "f1_comment": "we do scaling outside, set factor to 1.0 to avoid double mul",
        "variant": "muls_add",
        "status": (
            "available local vLLM source; exact R64 revision not retained"),
    },
    "kimi_linear": {
        "path": "vllm/model_executor/models/kimi_linear.py",
        "lines": [167, 172],
        "combine": "(experts * routed_scaling_factor) + shared_output",
        "variant": (
            "muls_add only for a nontrivial join scale; identity scale is "
            "plain_add"),
        "k3_scale": (
            "kimi-k3.json text_config.routed_scaling_factor = 1 -> identity "
            "-> plain_add / truthful absent"),
        "status": (
            "available local Kimi Linear source; exact KimiK3 model file "
            "not retained"),
    },
}
CAPTURE_REFERENCE = {
    "cell": "glm51_dp16tp1ep16_pf_1k_c1",
    "raw_sha256": (
        "cc05e517246544f8c4c4461e6986fac07c8892ad4547f3d221339c514123630f"),
    "rows": 1350,
    "shaped_rows": 75,
    "name": KERNEL_NAME,
    "accelerator_core": "AI_VECTOR_CORE",
    "input_dtypes": "DT_BF16;DT_BF16",
    "output_dtypes": "DT_BF16",
    "use": "reference_only_not_a_selector_or_predictive_supply",
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}
_MAX_CACHE = 32


def _feature(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _theta(compiled: Any) -> Any:
    return getattr(getattr(compiled, "operator", None), "theta", None)


def _find(compiled: Any, key: str, default: Any = None) -> Any:
    theta = _theta(compiled)
    if theta is None or not hasattr(theta, "find"):
        return default
    return theta.find(key, default)


def _r64_meta(compiled: Any, meta: dict) -> dict:
    """The existing R64 image-selection helper, without a second loader."""
    from .native_calls import _r64_source_meta
    return _r64_source_meta(tuple(meta.items()))


def _act_bytes(read_bytes: float, write_bytes: float,
               hidden: int) -> int | None:
    if hidden < 1 or read_bytes <= 0 or write_bytes <= 0:
        return None
    read_per_element = read_bytes / (2.0 * hidden)
    write_per_element = write_bytes / hidden
    if read_per_element <= 0 or abs(read_per_element - write_per_element) > 1e-9:
        return None
    value = int(round(read_per_element))
    if abs(read_per_element - value) > 1e-9 or value not in {1, 2, 4}:
        return None
    return value


def _declared_dtype(compiled: Any, meta: dict,
                    act_bytes: int) -> tuple[str | None, str]:
    operator = getattr(compiled, "operator", None)
    candidates = (
        ("operator_dtype", getattr(operator, "dtype", None)),
        ("model_metadata", meta.get("activation_dtype")),
        ("model_metadata", meta.get("act_format")),
        ("model_metadata", meta.get("dtype")),
    )
    for source, value in candidates:
        text = str(value or "").lower()
        text = text.rsplit(".", 1)[-1]
        if text in _FLOAT_BYTES:
            if _FLOAT_BYTES[text] != act_bytes:
                return None, "inconsistent_declared_dtype"
            return ("bf16" if text == "bfloat16" else text), source
    return _BYTES_DTYPE.get(act_bytes), "join_recipe_bytes"


def _layout(compiled: Any) -> tuple[str, str]:
    operator = getattr(compiled, "operator", None)
    text = str(getattr(operator, "layout", "") or "").upper()
    if text:
        return text, "operator_layout"
    return "ND", "default_nd"


def _scale(compiled: Any, meta: dict) -> tuple[float | None, str]:
    """Only an actual source-selected routed_scaling_factor; never a default."""
    theta = _theta(compiled)
    value = (theta.find("routed_scaling_factor", None)
             if theta is not None and hasattr(theta, "find") else None)
    source = "join_recipe.routed_scaling_factor"
    if value is None:
        value = meta.get("routed_scaling_factor")
        source = "model_metadata.routed_scaling_factor"
    if value is None:
        model_source = meta.get("glm51_source")
        if isinstance(model_source, dict):
            value = model_source.get("routed_scaling_factor")
            source = "glm51_source_metadata.routed_scaling_factor"
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value > 0):
        return float(value), source
    return None, "unresolved_source_selected_recipe"


def _vector_cores(compiled: Any) -> tuple[int | None, str]:
    """AIV cores from the source SKU/Core config, not a captured launch grid."""
    supply = getattr(compiled, "supply", None)
    try:
        units = dict(getattr(supply, "resource_units", ()) or ())
    except (TypeError, ValueError):
        units = {}
    value = units.get("vector")
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and int(value) > 0):
        return int(value), "supply.resource_units.vector"
    return None, "unresolved_source_sku_core_config"


def _source_recipe(compiled: Any, meta: dict) -> dict | None:
    """Actual producer/role/parent/combine variant, all source-selected."""
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    hidden = int(_find(compiled, "elements", 0) or 0)
    ops = int(_find(compiled, "ops_per_element", 0) or 0)
    read_bytes = float(_find(compiled, "read_bytes", 0.0) or 0.0)
    write_bytes = float(_find(compiled, "write_bytes", 0.0) or 0.0)
    act_bytes = _act_bytes(read_bytes, write_bytes, hidden)
    if act_bytes is None:
        return None
    dtype, dtype_source = _declared_dtype(compiled, meta, act_bytes)
    if dtype is None:
        return None
    layout, layout_source = _layout(compiled)
    scale, scale_source = _scale(compiled, meta)
    vector_cores, vector_source = _vector_cores(compiled)
    combine_variant = str(
        _find(compiled, "combine_variant", "unresolved") or "unresolved")
    scale_owner = str(
        _find(compiled, "scale_owner", "unresolved") or "unresolved")
    token_domain = str(
        _find(compiled, "token_domain", "shared_routed_combine")
        or "shared_routed_combine")
    producer = f"{compiled.kind}:{operator.family}:{operator.name}"
    template = getattr(operator, "template", None)
    role = str(getattr(template, "role", "") or compiled.variant or "join")
    parent = f"{compiled.kind}:{compiled.variant}:{token_domain}"
    if token_domain not in {
            "shared_routed_combine", "both", "prefill", "decode"}:
        selected = "unresolved"
    elif (combine_variant == "muls_add" and scale_owner == "join"
          and ops == 2 and scale is not None and scale > 1.0):
        selected = "muls_add"
    elif (combine_variant == "plain_add"
          or scale_owner in {"router", "none"}
          or (scale is not None and scale == 1.0)):
        selected = "plain_add"
    else:
        selected = "unresolved"
    return {
        "producer": producer,
        "role": role,
        "parent": parent,
        "semantic_role": "post_routing_residual",
        "combine_variant": combine_variant,
        "scale_owner": scale_owner,
        "token_domain": token_domain,
        "selected_variant": selected,
        "double_scaling": False,
        "hidden": hidden,
        "ops_per_element": ops,
        "read_bytes_per_token": read_bytes,
        "write_bytes_per_token": write_bytes,
        "act_bytes": act_bytes,
        "dtype": dtype,
        "dtype_source": dtype_source,
        "layout": layout,
        "layout_source": layout_source,
        "scale": scale,
        "scale_source": scale_source,
        "vector_cores": vector_cores,
        "vector_cores_source": vector_source,
    }


def _unresolved(scale: float | None, vector_cores: int | None) -> list[str]:
    reasons = [
        "muls_add_wrapper_source_not_retained",
        "muls_add_dispatch_site_not_retained",
        "muls_add_triton_launch_not_executed",
    ]
    if scale is None:
        reasons.append("routed_scaling_factor_not_declared_in_source_selected_recipe")
    if vector_cores is None:
        reasons.append("vector_core_count_not_declared_in_source_sku_core_config")
    return reasons


def _partial_reasons(scale: float | None,
                     vector_cores: int | None) -> list[str]:
    reasons = [
        "wrapper_blob_not_retained_in_r64_selected_subset",
        "dispatch_site_not_retained_in_r64_selected_subset",
        "triton_launch_not_executed",
    ]
    if scale is None:
        reasons.append("routed_scaling_factor_not_declared_in_source_selected_recipe")
    if vector_cores is None:
        reasons.append("vector_core_count_not_declared_in_source_sku_core_config")
    return reasons


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real partial R64 ``muls_add`` call identity, or ``None``."""
    if str(getattr(compiled, "kind", "") or "") != "F4":
        return None
    if str(getattr(compiled, "variant", "") or "") != "join":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "elementwise"
            or str(getattr(operator, "name", "") or "") != "join"):
        return None
    meta = _feature(compiled)
    if not _r64_meta(compiled, meta):
        return None
    recipe = _source_recipe(compiled, meta)
    if recipe is None or recipe["hidden"] < 1:
        return None
    if recipe["selected_variant"] != "muls_add":
        return None
    if (recipe["combine_variant"] != "muls_add"
            or recipe["scale_owner"] != "join"
            or recipe["ops_per_element"] != 2
            or recipe["scale"] is None or recipe["scale"] <= 1.0):
        return None
    route = recipe["token_domain"]
    phase_name = str(phase or "")
    if route in {"prefill", "decode"} and phase_name and phase_name != route:
        return None
    from .native_calls import _api, _call_spec, _kernel, _source

    source = _source(
        WRAPPER["repository"], WRAPPER["path"],
        sha256=WRAPPER["sha256"], symbol=WRAPPER["symbol"],
        revision=WRAPPER["revision"], state=WRAPPER["state"])
    source["source_status"] = WRAPPER["source_status"]
    source["model_source"] = {
        "model_identifier": str(meta.get("model_identifier") or ""),
        "arch_family": str(meta.get("arch_family") or ""),
        "device": str(meta.get("device") or meta.get("hardware_key") or ""),
        "image_id": str(meta.get("image_id") or ""),
        "source_revision": R64_REVISION,
    }
    kernel = _kernel(
        KERNEL_NAME, "vector", path=WRAPPER["path"],
        sha256=WRAPPER["sha256"], evidence="measured",
        launch="triton_jit", source_status=WRAPPER["source_status"],
        profiler_name=KERNEL_NAME,
        vector_cores_source=recipe["vector_cores_source"])
    scale = recipe["scale"]
    vector_cores = recipe["vector_cores"]
    typed = {
        "dtype": recipe["dtype"],
        "layout": recipe["layout"],
        "shapes": {
            "x": [recipe["hidden"]],
            "y": [recipe["hidden"]],
            "output": [recipe["hidden"]],
        },
        "attrs": {
            "hidden": recipe["hidden"],
            "ops_per_element": recipe["ops_per_element"],
            "act_bytes": recipe["act_bytes"],
            "read_bytes_per_token": recipe["read_bytes_per_token"],
            "write_bytes_per_token": recipe["write_bytes_per_token"],
            "scale": scale,
            "scale_source": recipe["scale_source"],
            "vector_cores": vector_cores,
            "vector_cores_source": recipe["vector_cores_source"],
            "combine_variant": recipe["combine_variant"],
            "scale_owner": recipe["scale_owner"],
            "token_domain": recipe["token_domain"],
            "selected_variant": recipe["selected_variant"],
            "double_scaling": False,
            "producer": recipe["producer"],
            "role": recipe["role"],
            "parent": recipe["parent"],
            "semantic_role": recipe["semantic_role"],
            "model_identifier": str(meta.get("model_identifier") or ""),
            "arch_family": str(meta.get("arch_family") or ""),
            "device": str(meta.get("device") or meta.get("hardware_key") or ""),
            "deployment_image_id": str(meta.get("image_id") or ""),
            "deployment_source_revision": R64_REVISION,
            "source_operation": OPERATION,
            "kernel_name": KERNEL_NAME,
            "wrapper_entry": WRAPPER_ENTRY,
            "wrapper_source_status": WRAPPER["source_status"],
            "dtype_source": recipe["dtype_source"],
            "layout_source": recipe["layout_source"],
        },
        "rank_width": 2,
    }
    return _call_spec(
        _api("torch", ENTRY, version=R64_VERSION),
        source, typed, (kernel,), (), "join", execution_core="aiv",
        unresolved=_unresolved(scale, vector_cores))


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _scalar(value: Any, dtype: str, role: str, *,
            constexpr: bool = False, status: str | None = None) -> dict:
    row = {"shape": [], "dtype": dtype, "layout": "SCALAR",
           "role": role, "value": value, "pointer": None,
           "allocation": "host", "constexpr": bool(constexpr)}
    if status is not None:
        row["status"] = status
    return row


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the source-selected R64 join."""
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != ENTRY:
        return None
    typed = dict(getattr(binding, "typed", {}) or {})
    attrs = dict(typed.get("attrs") or {})
    if (attrs.get("source_operation") != OPERATION
            or attrs.get("deployment_source_revision") != R64_REVISION
            or attrs.get("selected_variant") != "muls_add"
            or attrs.get("combine_variant") != "muls_add"
            or attrs.get("scale_owner") != "join"):
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    route = str(attrs.get("token_domain") or "shared_routed_combine")
    if route in {"prefill", "decode"} and phase and phase != route:
        return None
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    hidden = int(attrs.get("hidden", 0) or 0)
    act_bytes = int(attrs.get("act_bytes", 0) or 0)
    if (tokens < 1 or hidden < 1 or act_bytes not in {1, 2, 4}
            or int(attrs.get("ops_per_element", 0) or 0) != 2):
        return None
    scale = attrs.get("scale")
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or scale <= 1.0):
        return None
    scale = float(scale)
    vector_cores = attrs.get("vector_cores")
    if vector_cores is not None:
        if (not isinstance(vector_cores, int) or isinstance(vector_cores, bool)
                or vector_cores < 1):
            return None
    block_size = max(hidden // 2, 1024)
    n_elements = tokens * hidden
    n_blocks = (n_elements + block_size - 1) // block_size
    if n_blocks < 1:
        return None
    num_programs = min(n_blocks, vector_cores) if vector_cores else None
    grid = [num_programs] if num_programs is not None else None
    cache_key = (str(getattr(binding, "call_id", "")), tokens, hidden,
                 act_bytes, scale, vector_cores, phase, route)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dtype = str(typed.get("dtype") or "bf16")
    layout = str(typed.get("layout") or "ND")
    actual_shapes = {
        "x": [tokens, hidden],
        "y": [tokens, hidden],
        "output": [tokens, hidden],
    }
    actual_attrs = {
        **attrs,
        "m": tokens,
        "phase": phase,
        "token_domain": route,
        "selected_variant": "muls_add",
        "n_elements": n_elements,
        "n_blocks": n_blocks,
        "block_size": block_size,
        "num_programs": num_programs,
        "grid": grid,
        "read_bytes": tokens * float(attrs.get("read_bytes_per_token") or 0.0),
        "write_bytes": tokens * float(attrs.get("write_bytes_per_token") or 0.0),
        "input_count": 2,
        "output_count": 1,
    }
    actual = {"dtype": dtype, "layout": layout,
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    scale_status = "source_selected_routed_scaling_factor"
    descriptors = {
        "x": _descriptor([tokens, hidden], dtype, layout, "input"),
        "y": _descriptor([tokens, hidden], dtype, layout, "input"),
        "output": {
            **_descriptor([tokens, hidden], dtype, layout, "output"),
            "allocation": "callee_allocated_empty_like_x",
        },
        "scale": _scalar(scale, "float32", "host_scalar",
                         status=scale_status),
        "n_elements": _scalar(n_elements, "int64", "host_scalar"),
        "n_blocks": _scalar(n_blocks, "int64", "host_scalar"),
        "BLOCK_SIZE": _scalar(block_size, "int64", "constexpr",
                              constexpr=True),
    }
    source = dict(getattr(binding, "source", {}) or {})
    kernel = (dict(getattr(binding, "kernels", ())[0])
              if getattr(binding, "kernels", ()) else {
                  "name": KERNEL_NAME, "role": "vector",
                  "evidence": "measured", "launch": "triton_jit"})
    producer = str(attrs.get("producer") or "")
    role = str(attrs.get("role") or "")
    parent = str(attrs.get("parent") or "")
    image_id = str(attrs.get("deployment_image_id") or "unresolved")
    model_identifier = str(attrs.get("model_identifier") or "")
    prepare = {
        "schema": "mindie-sim/muls-add-native-prepare/1",
        "call": ENTRY,
        "call_id": getattr(binding, "call_id", None),
        "signature_hash": typed.get("signature_hash"),
        "source": source,
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "complete": False,
        "partial": True,
        "partial_reasons": _partial_reasons(scale, vector_cores),
        "source_recipe": {
            "producer": producer,
            "role": role,
            "parent": parent,
            "semantic_role": str(attrs.get("semantic_role") or ""),
            "combine_variant": str(attrs.get("combine_variant") or ""),
            "scale_owner": str(attrs.get("scale_owner") or ""),
            "token_domain": route,
            "selected_variant": "muls_add",
            "double_scaling": False,
            "hidden": hidden,
            "act_bytes": act_bytes,
            "dtype": dtype,
            "dtype_source": str(attrs.get("dtype_source") or ""),
            "layout": layout,
            "layout_source": str(attrs.get("layout_source") or ""),
            "ops_per_element": int(attrs.get("ops_per_element") or 0),
            "read_bytes_per_token": float(
                attrs.get("read_bytes_per_token") or 0.0),
            "write_bytes_per_token": float(
                attrs.get("write_bytes_per_token") or 0.0),
            "scale": scale,
            "scale_source": str(attrs.get("scale_source") or ""),
            "vector_cores": vector_cores,
            "vector_cores_source": str(
                attrs.get("vector_cores_source") or ""),
        },
        "producer": producer,
        "role": role,
        "parent": parent,
        "arguments": {"x": "x", "y": "y", "scale": scale},
        "argument_order": ["x", "y", "scale"],
        "scale_status": scale_status,
        "kernel_arguments": {
            "x": "x", "y": "y", "output": "output",
            "scale": scale, "n_elements": n_elements,
            "n_blocks": n_blocks, "BLOCK_SIZE": block_size,
        },
        "kernel_argument_order": [
            "x", "y", "output", "scale", "n_elements", "n_blocks",
            "BLOCK_SIZE",
        ],
        "grid": grid,
        "grid_basis": (
            "min(ceil(M*H/max(H//2,1024)), supply.resource_units.vector)"
            if vector_cores is not None
            else "unresolved_source_sku_core_config"),
        "shapes": actual_shapes,
        "validated_shapes": validated.get("shapes"),
        "descriptors": descriptors,
        "arrays": descriptors,
        "attribution": {
            "physical_owner": producer,
            "role": role,
            "parent": parent,
            "semantic_role": str(attrs.get("semantic_role") or ""),
            "operation": "x * routed_scaling_factor + y",
            "scale": scale,
            "scale_source": str(attrs.get("scale_source") or ""),
            "scale_owner": str(attrs.get("scale_owner") or ""),
            "double_scaling": False,
            "x": "routed-expert combine output (F2/F3 recipe)",
            "y": "shared-expert output (F4 recipe)",
            "charge": "one_physical_native_call_at_join",
            "not_charged_here": [
                "F2 router_finalize", "F3 expert_down", "F4 shared_down"],
            "member_rows": 0,
            "caller_binding": "not_retained_in_r64_selected_subset",
        },
        "source_evidence": {
            "wrapper": dict(WRAPPER),
            "image_config": dict(IMAGE_CONFIG_REFERENCE),
            "combine_source": dict(COMBINE_SOURCE_REFERENCE),
            "capture": dict(CAPTURE_REFERENCE),
            "model_source": {
                "model_identifier": model_identifier,
                "arch_family": str(attrs.get("arch_family") or ""),
                "device": str(attrs.get("device") or ""),
                "image_id": image_id,
                "source_revision": R64_REVISION,
            },
        },
        "capture": dict(CAPTURE_REFERENCE),
        "deployment": {
            "image_id": image_id,
            "model_identifier": model_identifier,
            "arch_family": str(attrs.get("arch_family") or ""),
            "device": str(attrs.get("device") or ""),
            "source_revision": R64_REVISION,
            "source_path": WRAPPER["path"],
            "source_sha256": WRAPPER["sha256"],
            "source_status": WRAPPER["source_status"],
            "source_revision_detail": WRAPPER["revision"],
            "combine_variant": str(attrs.get("combine_variant") or ""),
            "scale_owner": str(attrs.get("scale_owner") or ""),
            "token_domain": route,
            "scale_source": str(attrs.get("scale_source") or ""),
            "vector_cores_source": str(
                attrs.get("vector_cores_source") or ""),
        },
        "binary_selection": {
            "status": "not_applicable",
            "reason": "triton_jit_kernel_no_retained_aot_binary",
            "owner_kernel": KERNEL_NAME,
            "launch": "triton_jit",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "call_id": getattr(binding, "call_id", None),
            "image_id": image_id,
            "source_revision": R64_REVISION,
            "source_sha256": WRAPPER["sha256"],
            "source_status": WRAPPER["source_status"],
            "wrapper_source_revision": WRAPPER["revision"],
            "model_identifier": model_identifier,
            "producer": producer,
            "role": role,
            "parent": parent,
            "combine_variant": str(attrs.get("combine_variant") or ""),
            "scale_owner": str(attrs.get("scale_owner") or ""),
            "token_domain": route,
            "phase": phase,
            "selected_variant": "muls_add",
            "double_scaling": False,
            "hidden": hidden,
            "m": tokens,
            "scale": scale,
            "scale_status": scale_status,
            "n_elements": n_elements,
            "n_blocks": n_blocks,
            "block_size": block_size,
            "vector_cores": vector_cores,
            "num_programs": num_programs,
            "dtype": dtype,
            "act_bytes": act_bytes,
            "input_count": 2,
            "output_count": 1,
            "status": "partial_prepare_not_executed",
        },
        "unresolved": _unresolved(scale, vector_cores),
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_complete_native_execution",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": dtype,
            "layout": layout,
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= _MAX_CACHE:
        _DYNAMIC_CACHE.pop(next(iter(_DYNAMIC_CACHE)))
    _DYNAMIC_CACHE[cache_key] = result
    return result
