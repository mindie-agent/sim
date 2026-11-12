"""Prepare-only GDN invocation reconstructed from the existing call ledger.

[Role] Expand one selected ``aclnnChunkGatedDeltaRuleFwd`` occurrence from the
  actual request geometry into the wrapper's named arguments and tensor
  descriptors, using :mod:`mindie_sim.prediction.call_gdn` as the only source
  of shapes, dtypes and varlen arrays.
[Invariants] The static binding identity stays untouched; token/batch lanes,
  ``cu_seqlens`` and ``chunk_indices`` exist only in the dynamic occurrence.
  No tensor is allocated, no runtime is imported and no duration is changed.
  Unsupported ragged geometry returns ``None`` so the caller can mark the
  occurrence unresolved instead of inventing an ABI.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure descriptors derived from the existing ledger.
"""
from __future__ import annotations

from typing import Any

from mindie_sim.schema.common import ModelingError


def _delta_theta(compiled: Any) -> Any:
    delta = next((operator for operator in getattr(compiled, "siblings", ())
                  if operator.name == "gated_delta"), None)
    return delta.theta if delta is not None else compiled.operator.theta


def gdn_dims(compiled: Any) -> tuple[int, int, int, int, int]:
    theta = _delta_theta(compiled)
    heads, extras = dict(theta.heads), dict(theta.extras)
    keys = int(heads.get("q", extras.get("key_heads", 1)) or 1)
    values = int(heads.get("kv", extras.get("value_heads", keys)) or keys)
    dk = int(heads.get("dn", extras.get("dk", 1)) or 1)
    dv = int(heads.get("dv", extras.get("dv", dk)) or dk)
    chunk = int(extras.get("chunk", 64) or 64)
    return keys, values, dk, dv, chunk


_DYNAMIC_CACHE: dict[tuple, dict] = {}


def dynamic_gdn(binding: Any, compiled: Any, geometry: Any, *,
                validate: Any) -> dict | None:
    """Full prepare-only invocation from the existing GDNCall device program."""
    if (binding.api or {}).get("aclnn_symbol") != "aclnnChunkGatedDeltaRuleFwd":
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    if not requests:
        return None
    lengths = tuple(int(request.q_len) for request in requests)
    if not lengths or min(lengths) < 1 or len(set(lengths)) != 1:
        return None
    batch, tokens = len(lengths), lengths[0]
    keys, values, dk, dv, chunk = gdn_dims(compiled)
    cache_key = (binding.call_id, batch, lengths)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected = dict(binding.typed.get("attrs") or {})
    try:
        from mindie_sim.prediction import call_gdn
        call = call_gdn.make_call("scan_prefill", {
            "batch": batch, "tokens": tokens, "v_heads": values,
            "k_heads": keys, "k_dim": dk, "v_dim": dv,
            "chunk": int(selected.get("chunk_size", chunk)),
            "chunk_size": int(selected.get("chunk_size", chunk)),
            "phase": "prefill",
            "scale": float(selected.get("scale", dk ** -0.5)),
            "a_log": None, "dt_bias": None, "input_dtype": "bf16",
            "gate_dtype": "fp32", "beta_dtype": "bf16", "state_dtype": "fp32",
            "layout": str(binding.typed.get("layout") or "BSND"),
            "use_exp2": bool(selected.get("use_exp2", True)),
            "use_qk_l2norm_in_kernel": bool(
                selected.get("use_qk_l2norm_in_kernel", True)),
            "allow_neg_eigval": bool(selected.get("allow_neg_eigval", False)),
            "disable_recompute": bool(selected.get("disable_recompute", True)),
            "state_v_first": bool(selected.get("state_v_first", True)),
            "output_final_state": bool(selected.get("output_final_state", True)),
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": bool(
                selected.get("use_beta_sigmoid_in_kernel", False)),
            "return_intermediate_states": False, "initial_state": True,
        })
    except (ImportError, ModelingError, TypeError, ValueError):
        return None
    program = call.device_program
    g_cumsum_materialization = next(
        (item for item in (program.get("materializations") or ())
         if item.get("source_output") == "g_cumsum"
         or item.get("recipe") == "g_cumsum_export"), None)
    source_layout = dict((g_cumsum_materialization or {}).get("source_layout") or {})
    target_layout = dict((g_cumsum_materialization or {}).get("target_layout") or {})
    third_transpose = None
    if source_layout and target_layout:
        third_transpose = {
            "component": "Transpose", "tensor": "g_cumsum",
            "source_shape": list(source_layout.get("shape") or ()),
            "source_strides": list(source_layout.get("strides") or ()),
            "source_storage_shape": list(source_layout.get("storage_shape") or ()),
            "target_shape": list(target_layout.get("shape") or ()),
            "target_strides": list(target_layout.get("strides") or ()),
            "target_storage_shape": list(target_layout.get("storage_shape") or ()),
            "evidence": "GDNCall.device_program.materializations:g_cumsum_export",
        }
    arguments = dict(program.get("arguments") or {})
    chunk_meta = dict(arguments.get("chunk_indices") or {})
    chunk_values = list(chunk_meta.get("values") or ())
    if not chunk_values:
        return None
    arguments["chunk_indices"] = chunk_values
    arguments["chunk_indices_shape"] = [len(chunk_values)]
    inputs = dict(program.get("inputs") or {})
    outputs = dict(program.get("outputs") or {})
    exports = {name: value for name, value in (program.get("exports") or {}).items()
               if isinstance(value, dict)}
    shapes, descriptors = {}, {}
    for name, descriptor in {**inputs, **outputs, **exports}.items():
        shape = list(descriptor.get("shape") or ())
        if not shape:
            return None
        shapes[name] = shape
        descriptors[name] = {**descriptor, "role": (
            "input" if name in inputs else "output")}
    if "initial_state" in inputs:
        shapes["state"] = list(inputs["initial_state"]["shape"])
    shapes["cu_seqlens"] = [len(arguments.get("cu_seqlens") or ())]
    shapes["chunk_indices"] = [len(chunk_values)]
    attrs = dict(binding.typed.get("attrs") or {})
    attrs.update({
        "batch": batch, "tokens": tokens, "lengths": list(lengths),
        "cu_seqlens": list(arguments.get("cu_seqlens") or ()),
        "chunk_indices": chunk_values, "chunk_size": int(arguments["chunk_size"]),
        "layout": str(arguments["layout"]), "scale": float(arguments["scale"]),
        "chunks": int(arguments.get("chunks") or 0),
    })
    actual = {"dtype": binding.typed.get("dtype"),
              "layout": binding.typed.get("layout"),
              "shapes": shapes, "attrs": attrs}
    validated = validate(binding, actual)
    named = {
        "q": "q", "k": "k", "v": "v", "g": "g", "beta": "beta",
        "initial_state": "initial_state", "a_log": None, "dt_bias": None,
        "cu_seqlens": list(arguments.get("cu_seqlens") or ()),
        "chunk_indices": chunk_values,
        "layout": str(arguments["layout"]), "scale": float(arguments["scale"]),
        "chunk_size": int(arguments["chunk_size"]),
        "use_exp2": bool(attrs.get("use_exp2")),
        "use_qk_l2norm_in_kernel": bool(attrs.get("use_qk_l2norm_in_kernel")),
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": bool(attrs.get("use_beta_sigmoid_in_kernel")),
        "allow_neg_eigval": bool(attrs.get("allow_neg_eigval")),
        "disable_recompute": bool(attrs.get("disable_recompute")),
        "output_final_state": bool(attrs.get("output_final_state")),
        "return_intermediate_states": False,
        "state_v_first": bool(attrs.get("state_v_first")),
    }
    output_slots = {
        "o": {"acl_slot": "oOut", "status": "required"},
        "final_state": {"acl_slot": "finalStateOutOptional",
                        "status": "enabled" if attrs.get("output_final_state")
                        else "disabled"},
        "g_cumsum": {"acl_slot": "gCumsumOutOptional",
                     "status": "enabled" if attrs.get("disable_recompute")
                     else "disabled"},
        "A": {"acl_slot": "aOutOptional",
              "status": "enabled" if attrs.get("disable_recompute") else "disabled"},
        "beta_eff": {"acl_slot": "betaEffOutOptional",
                     "status": "enabled"
                     if attrs.get("use_beta_sigmoid_in_kernel") else "disabled",
                     "reason": "use_beta_sigmoid_in_kernel=false"},
        "h": {"acl_slot": "hOutOptional", "status": "disabled",
              "reason": "return_intermediate_states=false"},
        "q_hat": {"acl_slot": "qHatOutOptional",
                  "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                  else "disabled"},
        "k_hat": {"acl_slot": "kHatOutOptional",
                  "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                  else "disabled"},
        "q_rstd": {"acl_slot": "qRstdOutOptional",
                   "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                   else "disabled"},
        "k_rstd": {"acl_slot": "kRstdOutOptional",
                   "status": "enabled" if attrs.get("use_qk_l2norm_in_kernel")
                   else "disabled"},
    }
    enabled_outputs = {name: descriptors[name] for name, slot in output_slots.items()
                       if slot["status"] != "disabled" and name in descriptors}
    disabled_outputs = {name: dict(slot) for name, slot in output_slots.items()
                        if slot["status"] == "disabled"}
    prepare = {
        "schema": "mindie-sim/gdn-native-prepare/1",
        "call": program.get("call"), "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "wrapper": "fla_npu.ops.ascendc._stable.npu_chunk_gated_delta_rule_fwd",
        "wrapper_lines": [1472, 1516],
        "wrapper_source": program.get("wrapper_source"),
        "acl_signature": "aclnnChunkGatedDeltaRuleFwd",
        "acl_signature_source": {
            "archive": (program.get("source") or {}).get("archive"),
            "member": "ops/ascendc/_aclnn_ctypes.py",
            "sha256": "2aae5a0d7131f78f26a285f2b2cce2cbae3e2001fb20cf7e8fd373daf9e4cd8a",
            "lines": [150, 180],
        },
        "static_shapes": dict(binding.typed.get("shapes") or {}),
        "shapes": dict(validated.get("shapes") or {}),
        "arguments": arguments, "named_arguments": named,
        "chunk_indices_meta": chunk_meta,
        "arrays": descriptors, "descriptors": descriptors,
        "outputs": enabled_outputs, "output_slots": output_slots,
        "disabled_outputs": disabled_outputs,
        "physical_presentations_per_call": 7,
        "g_cumsum_materialization": g_cumsum_materialization,
        "physical_presentations": [
            {"component": "Cast", "role": "copy", "count": 1},
            {"component": "Transpose", "role": "copy", "count": 3,
             "third_transpose": third_transpose},
            {"component": "Prepare32", "role": "vector", "count": 1},
            {"component": "H24", "role": "cube", "count": 1},
            {"component": "O32", "role": "cube", "count": 1},
        ],
        "host_arrays": {
            "cu_seqlens": {"dtype": "int64",
                           "shape": [len(arguments.get("cu_seqlens") or ())]},
            "chunk_indices": {"dtype": "int64", "shape": [len(chunk_values)]},
        },
        "source": binding.source,
        "device_kernels": list(program.get("device_kernels") or ()),
        "allocation": "not_performed",
        "boundary": "prepare_only_host_descriptors",
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    result = {"actual": validated, "prepare": prepare,
              "prepare_args": {"dtype": validated.get("dtype"),
                               "layout": validated.get("layout"),
                               "shapes": validated.get("shapes"),
                               "attrs": validated.get("attrs")}}
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


