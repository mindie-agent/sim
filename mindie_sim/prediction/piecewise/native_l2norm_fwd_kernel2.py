"""Prepare-only R31 GDN q/k L2-normalization invocation.

[Role] Expand the two real external non-spec decode calls into the source
  wrapper ``vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd``
  and its default Triton launch ``l2norm_fwd_kernel2``.  The retained
  ``vllm_ascend/ops/gdn.py`` has two non-spec branches: mixed
  decode+prefill at lines 739/740 and pure decode at lines 864/865.  The
  pure-prefill branch keeps q/k normalization inside the signed
  ``aclnnChunkGatedDeltaRuleFwd`` parent, so this adapter deliberately emits
  nothing there.  Speculative decode (lines 706/707) is a different source
  call site and is not claimed here.
[Invariants] The source identity is the exact R31 image digest
  ``9e869e814846...`` plus the retained ``gdn.py`` import/call lines and the
  retained verified vllm ``l2norm.py`` source.  Head count and head width stay
  in the static binding; the decode token lane, flattened kernel rows ``M``
  and grid live only in the occurrence.  No tensor is allocated, no pointer
  or symbol is fabricated, no device launch is claimed and no measurement is
  used to price or select work.  A non-R31 image, a non-decode phase, a
  multi-token speculative row or an unknown source stays unbound/unresolved.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal workload geometry.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_PREFIX = "9e869e81484"
REVISION = "r31/9e869e814846"
ENTRY = ("vllm.third_party.flash_linear_attention.ops."
         "l2norm.l2norm_fwd")
KERNEL_NAME = "l2norm_fwd_kernel2"
ROLE = "external_non_spec_decode_qk_l2norm"
EPS = 1e-6
MBLOCK = 32
MAX_FUSED_BYTES = 65536

GDN_SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/gdn.py",
    "sha256": "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed",
    "symbol": "AscendGatedDeltaNetAttention._forward_core",
    "revision": REVISION,
    "state": "task_artifact",
}
GDN_LINES = {
    "import": 34,
    "mixed_q_call": 739,
    "mixed_k_call": 740,
    "pure_q_call": 864,
    "pure_k_call": 865,
    "spec_q_call": 706,
    "spec_k_call": 707,
}
L2NORM_SOURCE = {
    "repository": "vllm",
    "path": "vllm/third_party/flash_linear_attention/ops/l2norm.py",
    "sha256": "520f4513ef4fc97e6b88dc07877cca8e29da0d34e44673fe989e3af002d86381",
    "symbol": "l2norm_fwd",
    "revision": None,
    "state": "source",
}
L2NORM_SOURCE_LINES = {
    "wrapper": [95, 150],
    "kernel2": [79, 92],
    "default_launch": [114, 125],
}
RETAINED_VLLM_REVISION = "58d3918e3ea0a544ffedadad2ba84559e9c51d8f"
SOURCE_REFERENCES = {
    "caller": GDN_SOURCE,
    "caller_lines": GDN_LINES,
    "wrapper": L2NORM_SOURCE,
    "wrapper_lines": L2NORM_SOURCE_LINES,
    "retained_vllm_revision": RETAINED_VLLM_REVISION,
    "deployment_image_id": R31_IMAGE_SHA,
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str, allocation: str = "external") -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": allocation}


def _r31_metadata(compiled: Any) -> dict[str, Any]:
    """The exact witnessed R31 deployment only; all other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    from mindie_sim.prediction.mx_source_provenance import image_digest
    digest = image_digest(image_id)
    if not digest.startswith(R31_IMAGE_PREFIX):
        return {}
    lowerings = meta.get("source_lowerings")
    if isinstance(lowerings, str):
        lowerings = (lowerings,)
    if lowerings is not None and "gdn_decode" not in tuple(lowerings):
        return {}
    return meta


def _qk_dims(compiled: Any) -> tuple[int, int] | None:
    """Local key heads and D from the real gated_delta sibling, never a guess."""
    siblings = tuple(getattr(compiled, "siblings", ()) or ())
    names = {operator.name for operator in siblings}
    if "gated_delta" not in names:
        return None
    try:
        from .native_gdn import gdn_dims
        key_heads, _value_heads, dk, _dv, _chunk = gdn_dims(compiled)
        key_heads, dk = int(key_heads), int(dk)
    except (ImportError, KeyError, TypeError, ValueError):
        return None
    if key_heads < 1 or dk < 1:
        return None
    elements = _int(getattr(compiled.operator, "theta").find("elements", 0), 0)
    if elements and elements != key_heads * dk:
        return None
    return key_heads, dk


def _phase_name(compiled: Any, phase: str) -> str:
    return str(phase or "").strip().lower()


def _prefill_parent_spec(compiled: Any) -> Any:
    """Member view of the genuine PF parent whose source protocol owns q/k norm.

    The R31 authoritative ``gdn.py`` (sha 5e3b0fb0) selects the A5
    ``DeviceOperator.fla_gdn_prefill`` branch, and the A5 device code passes
    ``use_qk_l2norm_in_kernel=True`` to the fused FLA chunk.  The source-owned
    PF3 prepare likewise counts q/k normalization inside Prepare.  Only that
    activated protocol makes the two standalone q/k operators members of the
    parent call.  A disjoint no-normalize parent (``use_qk_l2norm_in_kernel``
    false, or a different entry such as the torch_npu fused chunk branch that
    normalizes externally at gdn.py:317/318) returns ``None`` here, so the
    leaf keeps its real external q/k calls.
    """
    meta = _r31_metadata(compiled)
    if not meta:
        return None
    arch = str(meta.get("arch_family") or meta.get("hardware_name")
               or meta.get("hardware_sku") or "").upper()
    if arch and arch != "A5" and "950" not in arch:
        return None
    # The PF source selector stamps the activated branch.  "external" is the
    # disjoint no-normalize torch_npu.npu_chunk_gated_delta_rule branch; the
    # leaf must not shadow its real external q/k calls.
    norm_owner = str(meta.get("gdn_prefill_qk_l2norm_owner") or "internal")
    if norm_owner != "internal":
        return None
    siblings = tuple(getattr(compiled, "siblings", ()) or ())
    parent_bound = next(
        (item for item in siblings if item.name == "gated_delta"), None)
    if parent_bound is None:
        return None
    try:
        from dataclasses import replace
        from mindie_sim.schema.common import ModelingError
        from . import native_calls
        from .compiler import CompiledOperator
        parent = CompiledOperator(
            parent_bound, compiled.supply, compiled.policy,
            kind=compiled.kind, layer=compiled.layer, domain=compiled.domain,
            siblings=compiled.siblings, variant=compiled.variant,
            feature_metadata=compiled.feature_metadata)
        parent_spec = native_calls._gdn_chunk_spec(parent, phase="prefill")
    except (ModelingError, ImportError, AttributeError, KeyError,
            TypeError, ValueError):
        return None
    if parent_spec is None:
        return None
    attrs = dict(parent_spec.typed.get("attrs") or {})
    if attrs.get("use_qk_l2norm_in_kernel") is not True:
        # Disjoint real no-normalize branch: q/k remain separate external
        # calls; do not shadow them into the parent.
        return None
    if (parent_spec.api or {}).get("aclnn_symbol") != (
            "aclnnChunkGatedDeltaRuleFwd"):
        return None
    source = dict(parent_spec.source or {})
    legacy = (source.get("sha256") != GDN_SOURCE["sha256"]
              or source.get("revision") != REVISION)
    unresolved = tuple(dict.fromkeys(tuple(parent_spec.unresolved) + (
        ("parent_source_revision_legacy_not_r31_5e3b0fb0",)
        if legacy else ())))
    members = tuple(dict.fromkeys(
        ("q_l2norm", "k_l2norm") + tuple(parent_spec.members)))
    return replace(
        parent_spec,
        owner=f"{str(getattr(compiled, 'kind', '') or '')}:"
              "recurrent:gated_delta",
        members=members, unresolved=unresolved)


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 external q/k L2-normalization call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  Non-R31 images, the chunked-prefill/no-external variant, pure
    prefill and every other operator return ``None`` so ordinary prediction
    stays unchanged.
    """
    if compiled is None or str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    name = str(getattr(operator, "name", "") or "")
    if name not in {"q_l2norm", "k_l2norm"}:
        return None
    if str(getattr(operator, "family", "") or "") != "elementwise":
        return None
    variant = str(getattr(compiled, "variant", "") or "")
    if variant == "chunked_prefill":
        return None
    siblings = tuple(getattr(compiled, "siblings", ()) or ())
    sibling_names = {item.name for item in siblings}
    if not {"q_l2norm", "k_l2norm"} <= sibling_names:
        return None
    theta = operator.theta
    if str(theta.find("source_kernel", "") or "") != KERNEL_NAME:
        return None
    if str(theta.find("owner_axis", "") or "") != "sequence_key_head":
        return None
    if _int(theta.find("decode_only", 0), 0) != 1:
        return None
    phase_name = _phase_name(compiled, phase)
    if phase_name in {"prefill", "pf"}:
        return _prefill_parent_spec(compiled)
    if phase_name and phase_name not in {"decode", "dc", "mixed", "decode_only"}:
        return None
    if not _r31_metadata(compiled):
        return None
    dims = _qk_dims(compiled)
    if dims is None:
        return None
    key_heads, dk = dims
    bd = 1 << (dk - 1).bit_length()
    if bd > MAX_FUSED_BYTES // 2:
        return None
    from . import native_calls
    source = native_calls._source(
        L2NORM_SOURCE["repository"], L2NORM_SOURCE["path"],
        sha256=L2NORM_SOURCE["sha256"], symbol=L2NORM_SOURCE["symbol"],
        revision=L2NORM_SOURCE["revision"], state=L2NORM_SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", path=L2NORM_SOURCE["path"],
        sha256=L2NORM_SOURCE["sha256"], evidence="source_declared")
    typed = {
        "dtype": "bf16",
        "layout": "contiguous",
        "phase": "decode",
        "shapes": {"x": [key_heads, dk], "y": [key_heads, dk]},
        "attrs": {
            "native_type": KERNEL_NAME,
            "source_kernel": KERNEL_NAME,
            "entry": ENTRY,
            "role": ROLE,
            "owner_axis": "sequence_key_head",
            "decode_only": 1,
            "key_heads": key_heads,
            "head_dim": dk,
            "elements": key_heads * dk,
            "eps": EPS,
            "output_dtype": None,
            "use_default_fla_norm": False,
            "MBLOCK": MBLOCK,
            "BD": bd,
            "grid_expr": "ceil(M/MBLOCK)",
            "kernel_arg_order": ["X", "Y", "eps", "M", "N", "BD", "MBLOCK"],
            "caller_repository": GDN_SOURCE["repository"],
            "caller_path": GDN_SOURCE["path"],
            "caller_sha256": GDN_SOURCE["sha256"],
            "caller_lines": dict(GDN_LINES),
            "non_spec_call_lines": {
                "mixed": [GDN_LINES["mixed_q_call"],
                          GDN_LINES["mixed_k_call"]],
                "pure": [GDN_LINES["pure_q_call"],
                         GDN_LINES["pure_k_call"]],
                "spec_not_claimed": [GDN_LINES["spec_q_call"],
                                     GDN_LINES["spec_k_call"]],
            },
            "non_spec_scope": "pure_or_mixed_decode",
            "wrapper_lines": dict(L2NORM_SOURCE_LINES),
            "retained_vllm_revision": RETAINED_VLLM_REVISION,
            "deployed_vllm_revision": None,
            "deployment_image_id": R31_IMAGE_SHA,
            "deployment_source_revision": REVISION,
            "kernel_source_status": "retained_verified_source_not_r31_extracted",
        },
        "rank_width": 2,
    }
    unresolved = (
        "aclnn_symbol_unresolved",
        "deployed_vllm_commit_not_extracted",
        "l2norm_kernel_binary_identity_not_established",
        "l2norm_runtime_env_not_observed",
        "execution_core_not_declared",
    )
    return native_calls._call_spec(
        native_calls._api("triton", ENTRY), source, typed, (kernel,), (),
        name, unresolved=unresolved)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for one real external q/k L2-norm call.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The q and k launches share this interface
    but each operator occurrence owns its own call charge.  Pure-prefill
    geometry and speculative multi-token rows return ``None``: their
    normalization is either inlined in the signed GDN parent or executes at a
    different source call site that this adapter does not claim.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("native_type") != KERNEL_NAME:
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    name = str(getattr(operator, "name", "") or "")
    if name not in {"q_l2norm", "k_l2norm"}:
        return None
    if str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    phase_name = str(getattr(geometry, "phase_name", "") or "").lower()
    if phase_name in {"prefill", "pf"}:
        return None
    if phase_name and phase_name not in {"decode", "dc", "mixed", "decode_only"}:
        return None
    requests = tuple(getattr(geometry, "requests", ()) or ())
    if not requests:
        return None
    decode_tokens = 0
    prefill_requests = 0
    for request in requests:
        request_phase = str(getattr(request, "phase", "") or "")
        q_len = _int(getattr(request, "q_len", 0), 0)
        draft_len = _int(getattr(request, "draft_len", 1), 1)
        if request_phase == "decode":
            if q_len == 0:
                continue
            if q_len != 1 or draft_len != 1:
                # Speculative/multi-token decode uses the separate source
                # call sites at gdn.py:706/707; do not attach a non-spec
                # identity to it.
                return None
            decode_tokens += 1
        elif request_phase == "prefill":
            prefill_requests += 1
        else:
            return None
    if decode_tokens < 1:
        return None
    qk_axis = "q" if name.startswith("q") else "k"
    if prefill_requests:
        branch = "mixed_non_spec_decode"
        call_line = GDN_LINES[f"mixed_{qk_axis}_call"]
    else:
        branch = "pure_non_spec_decode"
        call_line = GDN_LINES[f"pure_{qk_axis}_call"]
    key_heads = _int(attrs.get("key_heads"), 0)
    dk = _int(attrs.get("head_dim"), 0)
    if key_heads < 1 or dk < 1:
        return None
    rows = decode_tokens * key_heads
    bd = 1 << (dk - 1).bit_length()
    if bd > MAX_FUSED_BYTES // 2:
        return None
    grid = (rows + MBLOCK - 1) // MBLOCK
    cache_key = (binding.call_id, name, branch, decode_tokens, key_heads, dk)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    input_shape = [1, decode_tokens, key_heads, dk]
    actual_shapes = {"x": input_shape, "y": list(input_shape)}
    actual_attrs = {
        **attrs,
        "tokens": decode_tokens,
        "m": rows,
        "n": dk,
        "flattened_shape": [rows, dk],
        "grid": [grid],
        "blocks": grid,
        "BD": bd,
        "MBLOCK": MBLOCK,
        "operator_name": name,
        "qk_axis": qk_axis,
        "branch": branch,
        "call_site": branch,
        "call_line": call_line,
        "assigned_mixed_call_line": GDN_LINES[f"mixed_{qk_axis}_call"],
        "pure_decode_call_line": GDN_LINES[f"pure_{qk_axis}_call"],
        "spec_decode_call_line": GDN_LINES[f"spec_{qk_axis}_call"],
        "element_size": 2,
        "max_fused_size": MAX_FUSED_BYTES // 2,
    }
    actual = {"dtype": "bf16", "layout": "contiguous",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor(input_shape, "bf16", "contiguous", "input"),
        "y": _descriptor(input_shape, "bf16", "contiguous", "output",
                         allocation="callee_allocated_by_wrapper"),
    }
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "grid": [grid], "blocks": grid, "MBLOCK": MBLOCK, "BD": bd,
        "N": dk, "M": rows, "flattened_shape": [rows, dk],
        "source_lines": list(L2NORM_SOURCE_LINES["kernel2"]),
        "launch_lines": list(L2NORM_SOURCE_LINES["default_launch"]),
        "num_warps": "triton_default",
    })
    source = dict(binding.source or {})
    prepare = {
        "schema": "mindie-sim/l2norm-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": dict(GDN_LINES),
        "wrapper_lines": dict(L2NORM_SOURCE_LINES),
        "caller": dict(GDN_SOURCE),
        "source_branch": branch,
        "call_line": call_line,
        "assigned_mixed_call_line": GDN_LINES[f"mixed_{qk_axis}_call"],
        "pure_decode_call_line": GDN_LINES[f"pure_{qk_axis}_call"],
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {"x": "x", "eps": EPS, "output_dtype": None},
        "argument_order": ["x", "eps", "output_dtype"],
        "positional_arguments": {"x": "x"},
        "keywords": {"eps": EPS, "output_dtype": None},
        "prototype": ("l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, "
                      "output_dtype: torch.dtype | None = None) -> torch.Tensor"),
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "kernel_arguments": {
            "X": "x", "Y": "y", "eps": EPS, "M": rows, "N": dk,
            "BD": bd, "MBLOCK": MBLOCK,
        },
        "kernel_argument_order": ["X", "Y", "eps", "M", "N", "BD", "MBLOCK"],
        "launch": {
            "grid": [grid], "grid_axis": "program_id(0)",
            "grid_expr": "ceil(M/MBLOCK)", "MBLOCK": MBLOCK,
            "num_warps": "triton_default", "flattened_shape": [rows, dk],
        },
        "optional_slots": {
            "eps": "source_default_1e-6",
            "output_dtype": "not_set_default_none",
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R31_IMAGE_SHA),
            "revision": REVISION,
            "selection": ("exact R31 image digest + retained gdn.py "
                          "import and non-spec pure/mixed decode call lines "
                          "+ retained verified vllm wrapper"),
            "caller_path": GDN_SOURCE["path"],
            "caller_sha256": GDN_SOURCE["sha256"],
            "wrapper_path": L2NORM_SOURCE["path"],
            "wrapper_sha256": L2NORM_SOURCE["sha256"],
            "retained_vllm_revision": RETAINED_VLLM_REVISION,
            "deployed_vllm_revision": None,
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "call_id": binding.call_id,
            "operator": name,
            "qk_axis": qk_axis,
            "key_heads": key_heads,
            "head_dim": dk,
            "tokens": decode_tokens,
            "m": rows,
            "n": dk,
            "bd": bd,
            "mblock": MBLOCK,
            "grid": [grid],
            "source_sha256": L2NORM_SOURCE["sha256"],
            "caller_sha256": GDN_SOURCE["sha256"],
            "source_branch": branch,
            "call_line": call_line,
            "deployment_image_id": str(attrs.get("deployment_image_id")
                                       or R31_IMAGE_SHA),
            "deployment_source_revision": REVISION,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "no_kernel_binary_identity_from_python_launch",
            "owner_kernel": KERNEL_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "deployed_vllm_commit_not_extracted",
            "l2norm_kernel_binary_identity_not_established",
            "l2norm_runtime_env_not_observed",
        ],
        "unresolved": list(dict.fromkeys(
            list(binding.unresolved) + ["execution_rank_not_selected"])),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_exact_deployed_vllm_commit",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "bf16", "layout": "contiguous",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
