"""Prepare-only R31 QKV RMSNorm + MRoPE branch.

[Role] Expand the actual R31-retained ``BaseDeviceAdaptor.split_qkv_rmsnorm_rope``
  call (``torch.ops.vllm.qkv_rmsnorm_rope`` / CANN
  ``aclnnQkvRmsNormRopeCache``) into the observed projected QKV row, the
  per-head RMS weights, the three-plane cos/sin cache and the observed Q
  output.  The retained source supplies the call signature and the MRoPE
  positions argument; the original capture supplies the real operand slots.
[Invariants] The R31 image digest is required.  A true MRoPE declaration is
  required (``mrope_section`` plus ``mrope_interleaved``); a ``qwen3_5``
  family string alone is not admission.  Q/K/V/gate geometry is derived from
  the compiled A0 projected local geometry, never from a model name.  The
  ACLNN symbol is established from the retained ``libopapi_transformer.so``;
  the custom-op -> ACLNN dispatch, kernel binary/revision, K/V/gate output
  slots, weight transform and positions semantics are not retained and stay
  explicit.  No unknown value is fabricated and the prepare is never labelled
  physically complete.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata; no tensor allocation, pointer, symbol
  fabrication, device launch or measured-duration pricing.
"""
from __future__ import annotations

from typing import Any, Mapping

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
ENTRY = "split_qkv_rmsnorm_mrope_kernel"  # observed physical call identity
A5_ENTRY = "torch.ops.vllm.qkv_rmsnorm_rope_simt"
BASE_ENTRY = "torch.ops.vllm.qkv_rmsnorm_rope"
ACLNN_SYMBOL = "aclnnQkvRmsNormRopeCache"
KERNEL_NAME = "split_qkv_rmsnorm_mrope_kernel"
DEVICE_OP_PATH = "vllm_ascend/device/device_op.py"
DEVICE_OP_SHA = "f4392f346816c85f138d7e033a924c6b628ae684cd4fcd7a6d8818518351b5c9"
HARDWARE_PROFILE_SHA = (
    "024ea110a1da68edd91658596b6e95e6e81230b3189f7abe6cabe460d34bd3d6")
SIMT_KERNEL_NAME = "split_qkv_rmsnorm_rope_simt_kernel"
SOURCE = {
    "repository": "vllm-ascend",
    "path": DEVICE_OP_PATH,
    "sha256": DEVICE_OP_SHA,
    "symbol": "A5DeviceAdaptor.split_qkv_rmsnorm_rope",
    "revision": "r31/9e869e81",
    "state": "task_artifact",
    "lines": (1950, 1963),
    "entry": A5_ENTRY,
    "dispatch_status": "exact_retained_a5_python_method",
}
BASE_CANDIDATE = {
    "path": DEVICE_OP_PATH,
    "sha256": DEVICE_OP_SHA,
    "symbol": "BaseDeviceAdaptor.split_qkv_rmsnorm_rope",
    "lines": (945, 971),
    "entry": BASE_ENTRY,
    "aclnn_symbol": ACLNN_SYMBOL,
    "status": "candidate_not_proven",
    "why_not_selected": ("the retained A5 chain binds DeviceOperator="
                         "A5DeviceAdaptor; Base is not on that chain and the "
                         "observed kernel name is not proof of Base"),
}
ACLNN_SOURCE = {
    "repository": "cann",
    "path": ("usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/"
             "libopapi_transformer.so"),
    "sha256": "34baeae3ec68bf34c00506a757cc605949163b78be371203d7cd1a640cf4301f",
    "symbol": ACLNN_SYMBOL,
    "revision": "cann-9.2.0-r31",
    "state": "runtime_binary",
    "status": "candidate_not_proven",
    "why_not_selected": ("the ACLNN symbol is retained, but no retained call "
                         "trace maps torch.ops.vllm.qkv_rmsnorm_rope or the "
                         "observed kernel to it"),
}
HARDWARE_PROFILE_SOURCE = {
    "path": "vllm_ascend/device/hardware_profile.py",
    "sha256": HARDWARE_PROFILE_SHA,
    "symbol": "AscendDeviceType.A5",
    "lines": (333, 338),
    "fact": "device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED",
}
DEVICE_OPERATOR_BINDING = {
    "path": DEVICE_OP_PATH,
    "sha256": DEVICE_OP_SHA,
    "symbol": "get_device_adaptor/DeviceOperator",
    "lines": (2067, 2076),
    "fact": ("DeviceAdaptorFamily.FP8_OPTIMIZED -> A5DeviceAdaptor; "
             "DeviceOperator=A5DeviceAdaptor"),
}
SIMT_KERNEL_EVIDENCE = {
    "capture_id": "a5_qwen30ba3b_tp4_ep4_decode_4k_c1",
    "source": ("/Volumes/data/mindie-sim-data/a5/heldout_engine_20261031/"
               "a5_qwen30ba3b_tp4_ep4_decode_4k_c1/kernel_details.csv"),
    "source_sha256": (
        "ade6fb921dcd26b831798b2727c01cc55f00999197376d69f7b91f28eddb7955"),
    "summary": "obs-r3/c4-qwen139-source-summary.json",
    "summary_sha256": (
        "a13ff09b0cf10f8408bddf61b37269904ea99b1bc6725b9eb049f740bd018e5c"),
    "kernel": SIMT_KERNEL_NAME,
    "note": ("different retained Qwen3.0 capture; the SIMT-looking name is "
             "illustrative only, not the R31 Qwen3.6 image row"),
}
MROPE_SOURCE = {
    "path": "obs-r4/bind18-original-mrope-source.json",
    "sha256": "5fee0bd595394c38d1111ac3e254372d9e25d7871fa79389d6b69adb33cb5adf",
    "source_config": (
        "/Volumes/data/mindie-sim-data/a5/weights-metadata-20261108/"
        "Qwen3.6-27B-w8a8-mxfp8/config.json"),
    "source_config_sha256": (
        "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c"),
    "model_type": "qwen3_5_text",
    "mrope_section": (11, 11, 10),
    "mrope_interleaved": True,
    "partial_rotary_factor": 0.25,
    "norm_eps": 1e-6,
    "status": "genuine_original_config_and_source_metadata",
}
CAPTURE = {
    "capture_id": "smoke_qwen36_single_pf_4k_c1",
    "csv_path": (
        "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
        "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
        "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz"),
    "csv_sha256": (
        "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a"),
    "record": 179,
    "input_shapes": ["4096,7168", "256", "256", "3,4096,64"],
    "input_dtypes": ["DT_BF16", "DT_BF16", "DT_BF16", "DT_BF16"],
    "input_layouts": ["ND", "ND", "ND", "ND"],
    "output_shapes": ["4096,3072"],
    "output_dtypes": ["DT_BF16"],
    "output_layouts": ["ND"],
    "accelerator_core": "AI_VECTOR_CORE",
    "block_num": 64,
    "scope": ("post-selection identity of the retained original row; no "
              "duration or other measurement is part of the native call"),
}
_ROLE = "selected_composed_qkv_postprocess_r31_observed"
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _meta(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _image_digest(meta: Mapping[str, Any]) -> str:
    from mindie_sim.prediction.mx_source_provenance import image_digest
    for key in ("image_id", "oci_image_digest", "deployment_image_id"):
        value = str(meta.get(key) or "")
        digest = image_digest(value)
        if len(digest) == 64 and all(char in "0123456789abcdef"
                                     for char in digest):
            return digest
    image = str(meta.get("image") or meta.get("deployment_image") or "").lower()
    if R31_IMAGE_SHA[:12] in image:
        return R31_IMAGE_SHA
    return ""


def _rope_parameters(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = meta.get("rope_parameters")
    if isinstance(raw, Mapping):
        return raw
    raw = meta.get("unconsumed_config_fields")
    if isinstance(raw, Mapping) and isinstance(raw.get("rope_parameters"),
                                                Mapping):
        return raw["rope_parameters"]
    return {}


def _mrope_facts(meta: Mapping[str, Any], head_size: int) -> dict | None:
    """Only a declared MROPE section/interleave pair is admitted."""
    rope = _rope_parameters(meta)
    raw_section = meta.get("mrope_section", rope.get("mrope_section"))
    if not raw_section:
        return None
    try:
        section = tuple(_int(value) for value in raw_section)
    except TypeError:
        return None
    if len(section) != 3 or any(value < 1 for value in section):
        return None
    interleaved = (meta.get("mrope_interleaved")
                   if "mrope_interleaved" in meta
                   else rope.get("mrope_interleaved"))
    if not isinstance(interleaved, bool):
        return None
    factor = _float(meta.get("partial_rotary_factor",
                             rope.get("partial_rotary_factor")))
    rope_dim = _int(meta.get("rope_dim"), 0)
    if rope_dim <= 0 and factor is not None:
        scaled = float(head_size) * factor
        rope_dim = _int(round(scaled))
        if abs(scaled - rope_dim) > 1e-9:
            rope_dim = 0
    if rope_dim <= 0 or rope_dim > head_size or rope_dim % 2:
        rope_dim = None
    if rope_dim is not None and sum(section) * 2 != rope_dim:
        return None
    eps = _float(meta.get("norm_eps", meta.get("model_norm_eps")))
    missing = []
    if rope_dim is None:
        missing.append("rope_dim_from_caller")
    if eps is None:
        missing.append("eps_from_caller")
    return {
        "mrope_section": section,
        "mrope_interleaved": interleaved,
        "rope_dim": rope_dim,
        "eps": eps,
        "partial_rotary_factor": factor,
        "missing": missing,
    }


def _sibling(compiled: Any, name: str) -> Any | None:
    return next(
        (sibling for sibling in getattr(compiled, "siblings", ())
         if str(getattr(sibling, "name", "") or "") == name), None)


def _heads(compiled: Any) -> tuple[int, int, int] | None:
    sibling = _sibling(compiled, "attention")
    if sibling is None:
        return None
    heads = dict(getattr(sibling, "theta").heads)
    q_heads = _int(heads.get("q"), 0)
    kv_heads = _int(heads.get("kv"), 0)
    head_dim = _int(heads.get("dk"), 0)
    if q_heads > 0 and kv_heads > 0 and head_dim > 0:
        return q_heads, kv_heads, head_dim
    return None


def _geometry(compiled: Any, q_heads: int, kv_heads: int,
              head_dim: int) -> dict | None:
    projection = _sibling(compiled, "qkv_projection")
    attention = _sibling(compiled, "attention")
    if projection is None or attention is None:
        return None
    total = _int(dict(getattr(projection, "theta").matrix).get("N"), 0)
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    if total < 1:
        return None
    gate_size = total - q_size - 2 * kv_size
    if gate_size not in (0, q_size):
        return None
    q_hidden_size = q_size + gate_size
    if total != q_hidden_size + 2 * kv_size:
        return None
    quant = dict(getattr(projection, "theta").quant)
    output_format = str(quant.get("output_format") or "")
    if output_format:
        if output_format != "bf16":
            return None
        act_bytes = 2
    else:
        act_bytes = _int(dict(getattr(attention, "theta").quant).get(
            "act_bytes"), 0)
    if act_bytes != 2:
        return None
    instance = getattr(compiled, "instance", None)
    raw = (dict(getattr(getattr(instance, "template", None), "contract", ())
                or ()).get("qkv_slices") if instance is not None else None)
    if raw:
        try:
            declared = tuple(_int(value) for value in str(raw).split(","))
        except (TypeError, ValueError):
            return None
        if declared != (q_size, gate_size, kv_size, kv_size, total, act_bytes):
            return None
    return {
        "q_size": q_size, "kv_size": kv_size, "gate_size": gate_size,
        "q_hidden_size": q_hidden_size, "kv_hidden_size": kv_size,
        "head_dim": head_dim, "total": total, "act_bytes": act_bytes,
    }


def _fusion_contract(compiled: Any) -> Any | None:
    instance = getattr(compiled, "instance", None)
    template = getattr(instance, "template", None)
    for contract in getattr(template, "fusions", ()) or ():
        if str(getattr(contract, "name", "") or "") == "qkv_norm_rope":
            if tuple(getattr(contract, "members", ()) or ()) == (
                    "qk_norm", "rotary"):
                return contract
    return None


def _anchor(compiled: Any) -> bool:
    if str(getattr(compiled, "kind", "") or "") != "A0":
        return False
    operator = getattr(compiled, "operator", None)
    if operator is None or str(getattr(operator, "name", "") or "") != "rotary":
        return False
    names = {str(getattr(sibling, "name", "") or "")
             for sibling in getattr(compiled, "siblings", ())}
    if not {"qk_norm", "rotary", "qkv_projection", "attention"} <= names:
        return False
    if getattr(compiled, "instance", None) is not None:
        return _fusion_contract(compiled) is not None
    return True


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The actual R31 QKV RMSNorm+MRoPE producer identity, or ``None``.

    Admission requires the exact R31 image, a genuine MRoPE declaration and
    the A0 ``qkv_norm_rope`` sibling geometry.  The same Qwen3.5 family without
    ``mrope_section`` + ``mrope_interleaved`` returns ``None``; a family-string
    match alone never emits a native owner.
    """
    if not _anchor(compiled):
        return None
    meta = _meta(compiled)
    if _image_digest(meta) != R31_IMAGE_SHA:
        return None
    heads = _heads(compiled)
    if heads is None:
        return None
    facts = _mrope_facts(meta, heads[2])
    if facts is None:
        return None
    geometry = _geometry(compiled, *heads)
    if geometry is None:
        return None
    missing = list(facts["missing"])
    attrs = {
        "input_contract": ["input", "q_weight", "k_weight",
                           "cos_sin_cache", "positions"],
        "output_contract": ["q_model_projected"],
        "observed_output_reference": ["q"],
        "unknown_outputs": ["k", "v", "gate"],
        "q_output_basis": (
            "model_projected_local_geometry; observed reference is "
            "[4096,3072] on original record 179"),
        "observed_q_reference_shape": [4096, 3072],
        "observed_q_reference_record": 179,
        "observed_q_reference_capture_sha256": CAPTURE["csv_sha256"],
        "q_hidden_size": geometry["q_hidden_size"],
        "kv_hidden_size": geometry["kv_hidden_size"],
        "head_dim": geometry["head_dim"],
        "q_size": geometry["q_size"],
        "kv_size": geometry["kv_size"],
        "gate_size": geometry["gate_size"],
        "has_gate": bool(geometry["gate_size"]),
        "has_bias": False,
        "q_weight_transform": "caller_required_unverified",
        "k_weight_transform": "caller_required_unverified",
        "cos_sin_cache_planes": 3,
        "cos_sin_cache_layout": "ND",
        "cos_sin_cache_dtype": "bf16",
        "positions_role": "host_int_array_candidate",
        "positions_status": "caller_required_unverified",
        "mrope_section": list(facts["mrope_section"]),
        "mrope_interleaved": facts["mrope_interleaved"],
        "role": _ROLE,
        "dispatch_owner": "A5DeviceAdaptor",
        "dispatch_status": "partial_python_acl_mapping_unresolved",
        "python_api": A5_ENTRY,
        "python_api_status": "exact_retained_a5_python_method",
        "python_api_kernel_match": False,
        "base_candidate": BASE_ENTRY,
        "base_candidate_status": "candidate_not_proven",
        "aclnn_candidate": ACLNN_SYMBOL,
        "aclnn_status": "candidate_not_proven",
        "observed_kernel": KERNEL_NAME,
        "simt_kernel_name_observed_elsewhere": SIMT_KERNEL_NAME,
        "source_lines": list(SOURCE["lines"]),
        "source_symbol": SOURCE["symbol"],
        "source_path": SOURCE["path"],
        "source_sha256": SOURCE["sha256"],
        "kernel_name": KERNEL_NAME,
        "kernel_source_status": "binary_not_retained",
        "kernel_revision_status": "not_retained",
        "implementation": "observed_r31_mrope_kernel",
        "source_exact_for_r31": False,
        "a5_python_method_exact": True,
        "base_python_method_exact": False,
        "old_interface_source_exact": False,
        "mrope_source": MROPE_SOURCE["path"],
        "mrope_source_sha256": MROPE_SOURCE["sha256"],
        "source_config_sha256": MROPE_SOURCE["source_config_sha256"],
        "deployment_image_id": R31_IMAGE_SHA,
        "deployment_image": str(meta.get("image") or ""),
    }
    if facts["rope_dim"] is not None:
        attrs["rope_dim"] = facts["rope_dim"]
    if facts["eps"] is not None:
        attrs["eps"] = facts["eps"]
    if facts["partial_rotary_factor"] is not None:
        attrs["partial_rotary_factor"] = facts["partial_rotary_factor"]
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {
            "input": [geometry["total"]],
            "q_weight": [geometry["head_dim"]],
            "k_weight": [geometry["head_dim"]],
            "q": [geometry["q_size"]],
        },
        "attrs": attrs,
        "rank_width": 2,
    }
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    source.update({
        "lines": list(SOURCE["lines"]),
        "entry": A5_ENTRY,
        "dispatch_status": "exact_retained_a5_python_method",
        "observed_kernel": KERNEL_NAME,
        "observed_kernel_matches_python_api": False,
    })
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="runtime_observed",
        python_api=A5_ENTRY,
        python_api_status="exact_retained_a5_method",
        python_api_kernel_match=False,
        simt_kernel_name_observed_elsewhere=SIMT_KERNEL_NAME,
        base_candidate_entry=BASE_ENTRY,
        base_candidate_status="candidate_not_proven",
        aclnn_candidate=ACLNN_SYMBOL,
        aclnn_status="candidate_not_proven",
        kernel_source_status="binary_not_retained",
        kernel_revision_status="not_retained",
        capture_record=CAPTURE["record"],
        capture_core=CAPTURE["accelerator_core"],
        capture_block_num=CAPTURE["block_num"],
        capture_sha256=CAPTURE["csv_sha256"])
    api = native_calls._api("cann_kernel", ENTRY, version="r31-original-capture")
    api.update({
        "dispatch_status": "partial_python_acl_mapping_unresolved",
        "dispatch_owner": "A5DeviceAdaptor",
        "observed_kernel": KERNEL_NAME,
        "observed_kernel_matches_a5_python_api": False,
        "python_api_candidates": [
            {
                "name": SOURCE["symbol"],
                "entry": A5_ENTRY,
                "status": "exact_retained_a5_python_method",
                "kernel_name_match": False,
                "simt_kernel_name_observed_elsewhere": SIMT_KERNEL_NAME,
            },
            dict(BASE_CANDIDATE),
        ],
        "aclnn_candidate": dict(ACLNN_SOURCE),
        "exact_retained_chain": {
            "hardware_profile": {
                **HARDWARE_PROFILE_SOURCE,
                "lines": list(HARDWARE_PROFILE_SOURCE["lines"]),
            },
            "device_operator": {
                **DEVICE_OPERATOR_BINDING,
                "lines": list(DEVICE_OPERATOR_BINDING["lines"]),
            },
            "python_method": {
                "path": SOURCE["path"],
                "sha256": SOURCE["sha256"],
                "symbol": SOURCE["symbol"],
                "lines": list(SOURCE["lines"]),
                "entry": A5_ENTRY,
            },
        },
    })
    unresolved = [
        "python_dispatch_mapping_partial",
        "a5_python_api_kernel_name_mismatch",
        "base_python_candidate_not_proven",
        "aclnn_mapping_not_proven",
        "kernel_binary_not_retained",
        "kernel_source_not_retained",
        "k_v_gate_outputs_unverified",
        "q_weight_transform_unverified",
        "positions_semantics_unverified",
        *missing,
    ]
    return native_calls._call_spec(
        api, source, typed, (kernel,), ("qk_norm", "rotary"),
        "fused:qkv_norm_rope", execution_core="aiv",
        unresolved=unresolved)


def _descriptor(shape: list | None, dtype: str | None, layout: str | None,
                role: str, *, present: str = "present") -> dict:
    return {
        "shape": (None if shape is None else
                  [int(item) if item is not None else None
                   for item in shape]),
        "dtype": dtype, "layout": layout, "role": role,
        "present": present, "pointer": None, "allocation": "external",
    }


def _null_descriptor(role: str) -> dict:
    return {"shape": None, "dtype": None, "layout": None, "role": role,
            "present": False, "pointer": None}


def _prepare(binding: Any, attrs: Mapping[str, Any],
             validated: Mapping[str, Any], facts: Mapping[str, Any],
             missing: list[str], launch: Mapping[str, Any]) -> dict:
    shapes = dict(validated.get("shapes") or {})
    m = _int(attrs.get("m"))
    rope_dim = facts.get("rope_dim")
    q_size = _int(attrs.get("q_size"))
    kv_size = _int(attrs.get("kv_size"))
    gate_size = _int(attrs.get("gate_size"))
    descriptors = {
        "input": _descriptor(shapes["input"], "bf16", "ND", "input"),
        "q_weight": _descriptor(shapes["q_weight"], "bf16", "ND", "input"),
        "k_weight": _descriptor(shapes["k_weight"], "bf16", "ND", "input"),
        "cos_sin_cache": _descriptor(shapes["cos_sin_cache"], "bf16", "ND",
                                     "input"),
        "q": _descriptor(shapes["q"], "bf16", "ND",
                         "output_model_projected"),
        "k_candidate": _descriptor([m, kv_size], "bf16", "ND", "output",
                                   present="unverified_candidate"),
        "v_candidate": _descriptor([m, kv_size], "bf16", "ND", "output",
                                   present="unverified_candidate"),
        "gate_candidate": _descriptor([m, gate_size], "bf16", "ND", "output",
                                      present="unverified_candidate"),
        "positions": _descriptor([3, m], "int64", "ND", "host_input",
                                 present="candidate_unverified"),
    }
    descriptors["q_bias"] = _null_descriptor("input_optional")
    descriptors["k_bias"] = _null_descriptor("input_optional")
    descriptors["q"]["observed_reference_shape"] = [4096, 3072]
    descriptors["q"]["observed_reference_record"] = 179
    descriptors["q"]["basis"] = (
        "model_projected_local_geometry; not an observation of this event")
    descriptors["q_weight"]["transform"] = "caller_required_unverified"
    descriptors["k_weight"]["transform"] = "caller_required_unverified"
    if rope_dim is None:
        descriptors["cos_sin_cache"]["unknown"] = ["rope_dim"]
    arguments = {
        "input": "input",
        "q_weight": "q_weight",
        "k_weight": "k_weight",
        "q_hidden_size": _int(attrs.get("q_hidden_size")),
        "kv_hidden_size": _int(attrs.get("kv_hidden_size")),
        "head_dim": _int(attrs.get("head_dim")),
        "eps": facts.get("eps"),
        "q_bias": None,
        "k_bias": None,
        "cos_sin_cache": "cos_sin_cache",
        "positions": "positions",
    }
    optional = {
        "q_bias": "not_observed_not_passed",
        "k_bias": "not_observed_not_passed",
        "positions": "host_int_array_candidate_unverified",
        "k_v_gate_outputs": "unverified_candidate",
        "q_weight_transform": "caller_required_unverified",
        "k_weight_transform": "caller_required_unverified",
        "dispatch": "partial_python_acl_mapping_unresolved",
    }
    for key in ("rope_dim", "eps"):
        optional[key] = ("declared" if facts.get(key) is not None
                         else "caller_required")
    kernel = (dict(binding.kernels[0]) if binding.kernels else
              {"name": KERNEL_NAME, "role": "vector",
               "evidence": "runtime_observed"})
    source = dict(binding.source or {})
    partial_reasons = list(dict.fromkeys([
        "python_dispatch_mapping_partial",
        "a5_python_api_kernel_name_mismatch",
        "base_python_candidate_not_proven",
        "aclnn_mapping_not_proven",
        "kernel_binary_not_retained",
        "kernel_source_not_retained",
        "k_v_gate_outputs_unverified",
        "q_weight_transform_unverified",
        "positions_semantics_unverified",
        *missing,
    ]))
    dispatch = {
        "status": "partial",
        "owner": "A5DeviceAdaptor",
        "observed_kernel": KERNEL_NAME,
        "observed_kernel_matches_a5_python_api": False,
        "simt_kernel_name_observed_elsewhere": SIMT_KERNEL_NAME,
        "simt_kernel_evidence": dict(SIMT_KERNEL_EVIDENCE),
        "exact_retained_chain": {
            "hardware_profile": {
                **HARDWARE_PROFILE_SOURCE,
                "lines": list(HARDWARE_PROFILE_SOURCE["lines"]),
            },
            "device_operator": {
                **DEVICE_OPERATOR_BINDING,
                "lines": list(DEVICE_OPERATOR_BINDING["lines"]),
            },
            "python_method": {
                "path": SOURCE["path"],
                "sha256": SOURCE["sha256"],
                "symbol": SOURCE["symbol"],
                "lines": list(SOURCE["lines"]),
                "entry": A5_ENTRY,
            },
        },
        "base_candidate": dict(BASE_CANDIDATE),
        "aclnn_candidate": dict(ACLNN_SOURCE),
        "phase": ("static device-adaptor selection at import; A5 profile is "
                  "not phase-branched"),
        "proof": [
            "hardware_profile.py:333-338 A5 -> DeviceAdaptorFamily.FP8_OPTIMIZED",
            "device_op.py:2067-2076 FP8_OPTIMIZED -> A5DeviceAdaptor; "
            "DeviceOperator=A5DeviceAdaptor",
            "device_op.py:1950-1963 A5DeviceAdaptor.split_qkv_rmsnorm_rope -> "
            "torch.ops.vllm.qkv_rmsnorm_rope_simt",
            "device_op.py:945-971 BaseDeviceAdaptor is the only "
            "qkv_rmsnorm_rope caller; it is not selected by the retained A5 "
            "chain",
            "observed kernel split_qkv_rmsnorm_mrope_kernel does not match "
            "the SIMT-looking name; kernel/alias/symbol do not prove Base",
        ],
    }
    prepare = {
        "schema": "mindie-sim/qkv-rmsnorm-rope-observed-prepare/3",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": list(SOURCE["lines"]),
        "source_exact_for_r31": False,
        "a5_python_method_exact": True,
        "base_python_method_exact": False,
        "python_kernel_mapping_exact": False,
        "aclnn_mapping_exact": False,
        "kernel_source_exact": False,
        "old_interface_source_exact": False,
        "old_interface": {
            "schema": "mindie-sim/opqkv-mrope-interface/2",
            "source_exact_for_r31": False,
            "superseded_by": "mindie-sim/opqkv-mrope-interface/3",
            "reasons": [
                "v1 used the local vllm-ascend Triton file ba3c7e74 and caller "
                "6e3299ab, neither retained from the R31 image",
                "v2 asserted BaseDeviceAdaptor -> torch.ops.vllm.qkv_rmsnorm_rope "
                "-> aclnnQkvRmsNormRopeCache as the real leaf; the retained R31 "
                "A5 profile binds DeviceOperator=A5DeviceAdaptor, whose method "
                "calls qkv_rmsnorm_rope_simt",
                "v2 K/V/gate source-output and complete=true claims are "
                "withdrawn; only the observed Q output is retained and the "
                "observed kernel name does not match the A5 Python API name",
            ],
        },
        "kernel": kernel,
        "dispatch": dispatch,
        "aclnn": {
            "symbol": ACLNN_SYMBOL,
            "library": dict(ACLNN_SOURCE),
            "status": "candidate_not_proven",
        },
        "mrope_source": dict(MROPE_SOURCE),
        "role": _ROLE,
        "stage": "interface_prepared",
        "execution_status": "prepare_only_no_device_launch",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": [
            "input", "q_weight", "k_weight", "q_hidden_size",
            "kv_hidden_size", "head_dim", "eps", "q_bias", "k_bias",
            "cos_sin_cache", "positions",
        ],
        "shapes": shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "optional_slots": optional,
        "outputs": {
            "observed_reference_outputs": ["q"],
            "observed_reference_shapes": {"q": [4096, 3072]},
            "observed_reference_basis": (
                "original R31 PF4kC1 capture record 179; not the current "
                "event"),
            "model_projected_shapes": {"q": [m, q_size]},
            "candidate_unverified": ["k", "v", "gate"],
            "candidate_shapes": {
                "k": [m, kv_size], "v": [m, kv_size],
                "gate": [m, gate_size],
            },
            "current_event_q_basis": "model_projected_local_geometry",
            "unknown_output_slots": True,
        },
        "launch": dict(launch),
        "observed": {
            **{key: CAPTURE[key] for key in
               ("capture_id", "csv_path", "csv_sha256", "record",
                "input_shapes", "input_dtypes", "input_layouts",
                "output_shapes", "output_dtypes", "output_layouts",
                "accelerator_core", "block_num", "scope")},
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "aclnn_candidate": ACLNN_SYMBOL,
            "call_id": binding.call_id,
            "m": m,
            "q_hidden_size": _int(attrs.get("q_hidden_size")),
            "kv_hidden_size": _int(attrs.get("kv_hidden_size")),
            "head_dim": _int(attrs.get("head_dim")),
            "q_size": q_size, "kv_size": kv_size, "gate_size": gate_size,
            "dtype": "bf16", "layout": "ND",
            "source_sha256": source.get("sha256"),
            "source_revision": source.get("revision"),
            "source_lines": list(SOURCE["lines"]),
            "deployment_image_id": R31_IMAGE_SHA,
        },
        "deployment": {
            "image_id": R31_IMAGE_SHA,
            "image": str(attrs.get("deployment_image") or ""),
            "revision": "r31/9e869e81",
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_lines": list(SOURCE["lines"]),
            "a5_python_method": dict(SOURCE),
            "base_candidate": dict(BASE_CANDIDATE),
            "aclnn_candidate": dict(ACLNN_SOURCE),
            "observed_kernel": KERNEL_NAME,
            "kernel_source_status": "binary_not_retained",
            "kernel_revision_status": "not_retained",
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "kernel_binary_and_revision_not_retained",
            "owner_kernel": KERNEL_NAME,
            "kernel_role": "vector",
            "execution_core": "aiv",
            "python_api": A5_ENTRY,
            "python_api_kernel_match": False,
            "base_candidate_status": "candidate_not_proven",
            "aclnn_candidate": ACLNN_SYMBOL,
            "aclnn_status": "candidate_not_proven",
        },
        "source_exact": {
            "a5_python_method": True,
            "base_python_method": False,
            "python_kernel_mapping": False,
            "aclnn_mapping": False,
            "kernel_source": False,
            "kernel_binary": False,
            "k_v_gate_outputs": False,
            "q_weight_transform": False,
            "positions_semantics": False,
            "old_interface": False,
        },
        "observed_abi_complete": not missing,
        "complete": False,
        "partial": True,
        "partial_reasons": partial_reasons,
        "unresolved": list(dict.fromkeys([
            *binding.unresolved, *missing, "execution_rank_not_selected",
        ])),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_kernel_binary_identity",
            "no_kernel_source_identity", "no_python_kernel_mapping_claim",
            "no_base_python_method_claim", "no_aclnn_mapping_claim",
            "no_k_v_gate_output_claim", "no_q_weight_transform_claim",
            "no_positions_semantics_claim",
            "no_physical_execution_completeness",
        ],
    }
    return prepare


def _launch() -> dict:
    return {
        "status": "unverified_no_retained_tiler_or_active_launch_evidence",
        "launched_blocks": None,
        "observed_reference": {
            "tokens": 4096,
            "block_num": CAPTURE["block_num"],
            "record": CAPTURE["record"],
            "capture_sha256": CAPTURE["csv_sha256"],
            "scope": ("retained R31 PF4kC1 structural reference only; not an "
                      "active launch fact for the evaluated event"),
        },
        "basis": ("the original 4096-token PF row had Block Num 64, but no "
                  "retained tiler/loader constant or active-launch record maps "
                  "an evaluated event to that block count"),
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the R31 retained ACLNN producer.

    The input, weights, cos/sin cache and observed Q output are carried with
    their real shapes.  K/V/gate are explicit unverified candidates from the
    projected local geometry, never observed/confirmed outputs.  The prepare
    is always partial: kernel binary/revision, custom-op dispatch, weight
    transform and positions semantics are not retained.
    """
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("role") != _ROLE:
        return None
    if attrs.get("implementation") != "observed_r31_mrope_kernel":
        return None
    q_hidden_size = _int(attrs.get("q_hidden_size"))
    kv_hidden_size = _int(attrs.get("kv_hidden_size"))
    head_dim = _int(attrs.get("head_dim"))
    q_size = _int(attrs.get("q_size"))
    kv_size = _int(attrs.get("kv_size"))
    gate_size = _int(attrs.get("gate_size"))
    total = q_hidden_size + 2 * kv_hidden_size
    declared_total = _int((binding.typed.get("shapes") or {}).get(
        "input", [0])[0])
    if (q_hidden_size < 1 or kv_hidden_size < 1 or head_dim < 1
            or q_size != q_hidden_size - gate_size
            or kv_size != kv_hidden_size
            or gate_size not in (0, q_size)
            or total != declared_total
            or total != q_size + gate_size + 2 * kv_size):
        return None
    m = _int(getattr(geometry, "effective_tokens", 0))
    if m < 1:
        return None
    facts = {
        "mrope_section": (tuple(attrs["mrope_section"])
                          if attrs.get("mrope_section") else None),
        "mrope_interleaved": attrs.get("mrope_interleaved"),
        "rope_dim": _int(attrs.get("rope_dim"), 0) or None,
        "eps": _float(attrs.get("eps")),
    }
    missing = []
    if facts["rope_dim"] is None:
        missing.append("rope_dim_from_caller")
    if facts["eps"] is None:
        missing.append("eps_from_caller")
    rope_dim = facts["rope_dim"]
    cache_key = (binding.call_id, m, rope_dim, facts["eps"],
                 facts["mrope_section"], facts["mrope_interleaved"])
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cos_sin_shape = [3, m, rope_dim] if rope_dim is not None else [3, m, None]
    actual_shapes = {
        "input": [m, total],
        "q_weight": [head_dim],
        "k_weight": [head_dim],
        "cos_sin_cache": cos_sin_shape,
        "q": [m, q_size],
        "k_candidate": [m, kv_size],
        "v_candidate": [m, kv_size],
        "gate_candidate": [m, gate_size],
        "positions_candidate": [3, m],
    }
    actual_attrs = {
        **attrs,
        "m": m,
        "num_tokens": m,
        "mrope_section": (list(facts["mrope_section"])
                          if facts["mrope_section"] else None),
        "is_interleaved": facts["mrope_interleaved"],
        "rope_dim": rope_dim,
        "eps": facts["eps"],
        "has_bias": False,
        "q_bias": None,
        "k_bias": None,
        "positions_shape_candidate": [3, m],
        "positions_observed": False,
        "outputs_observed": ["q"],
        "outputs_candidate_unverified": ["k", "v", "gate"],
        "q_observed_shape": [m, q_size],
        "observed_q_reference_shape": [4096, 3072],
        "observed_q_reference_record": 179,
        "k_candidate_shape": [m, kv_size],
        "v_candidate_shape": [m, kv_size],
        "gate_candidate_shape": [m, gate_size],
    }
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    launch = _launch()
    prepare = _prepare(binding, actual_attrs, validated, facts, missing, launch)
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": "bf16",
            "layout": "ND",
            "shapes": {key: list(value) if value is not None else None
                       for key, value in actual_shapes.items()},
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
