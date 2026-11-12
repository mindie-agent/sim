"""Prepare-only R64 I0 indexer-k aclnnLayerNormWithImplMode normalizer.

[Role] Expand the R64 SFA indexer key normalization selected by the real
  ``nn.LayerNorm`` module into the actual ``[M, D]`` input, the INT32
  one-element ``normalized_shape`` array, and the ``[D]`` gamma/beta operands,
  producing ``[M, D]`` output plus ``[M, 1]`` mean/rstd.  The row is the I0
  producer's ``index_k_norm``: the K projection's normalized key feeds the K
  rope/cache store, then the I1 indexer search.  ``D``, epsilon and the
  resolved norm dtype are the selected source slot's declared values.
[Invariants] Selection is the selected same-image ``nn.LayerNorm`` source
  slot declared by the applied I0 producer and the R64 deployment image:
  the GLM-5.1 ``deepseek_v2.py`` L685 source with ``indexer.py`` L335 call
  site, or the GLM-5.3 ``glm5next/attention.py`` L96 source with
  ``indexer_kpool.py`` L763-770 call site.  The source slot's declared
  path/symbol/lines/sha256, call site and shared ``layernorm.py`` L305-320
  implementation are consumed as declared; an unknown or mismatched slot
  returns ``None`` and stays unbound.  The row must be I0 with
  ``source_stage="I0"``; the old after-search blanket I1 linkage is not
  accepted.  Native ``aclnnLayerNormWithImplMode``/``LayerNormV4`` evidence is
  prepare-only; the ACLNN wrapper/dispatch, ``implMode`` value, C header and
  kernel binary/source are unresolved.  No tensor is allocated, no pointer or
  symbol is fabricated, no device launch is claimed and no measured clock
  selects a recipe.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the declared source slot and
  normal geometry.  No sample name, held-in size, measured duration or trace
  dimension is a selector.  Unknown residual stays absent or unresolved.
"""
from __future__ import annotations

import math
from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch.nn.functional.layer_norm"
ACLNN_API = "aclnnLayerNormWithImplMode"
KERNEL_NAME = "LayerNormV4"
PROFILER_NAME = "aclnnLayerNormWithImplMode_LayerNormV4_LayerNormV4"
SOURCE_KIND = "nn.LayerNorm"
STAGE = "I0"
LAYER_NORM_SOURCE = {
    "repository": "vllm",
    "path": "vllm/model_executor/layers/layernorm.py",
    "sha256": "696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c",
    "symbol": "LayerNorm",
    "lines": [305, 320],
    "dtype_source": "weight/bias float32; forward x.float()",
}
# The two actual source variants declared by the applied I0 producer.  The
# model identifier selects one; the leaf consumes the declaration, never a
# model name or the original enrolled case.
SOURCE_VARIANTS = {
    "glm_moe_dsa": {
        "model": {
            "repository": "vllm",
            "path": "vllm/model_executor/models/deepseek_v2.py",
            "sha256": "02e72b106da8b106b9fb0adf5d22cf1cf9713d7b0f4e002c02eeee6e35ffc305",
            "symbol": "Indexer.__init__",
            "lines": [685, 685],
        },
        "call_site": {
            "repository": "vllm-ascend",
            "path": "vllm_ascend/attention/indexer.py",
            "sha256": "99b7e507c2ba6f8f47d04b051e647f51e6b18b23ecadfc839cc351d81b785942",
            "symbol": "AscendSFAIndexerBackend.forward_k",
            "lines": [335, 335],
        },
        "call_site_variants": [{
            "path": "vllm_ascend/attention/indexer.py",
            "sha256": "99b7e507c2ba6f8f47d04b051e647f51e6b18b23ecadfc839cc351d81b785942",
            "symbol": "AscendSFAIndexerBackend.forward_k",
            "lines": [335, 335],
        }],
    },
    "glm5_next": {
        "model": {
            "repository": "vllm-ascend",
            "path": "vllm_ascend/models/glm5next/attention.py",
            "sha256": "f0e1dfa46eea4c6385e10b86b73fa024121cc18e73ae9148dd78d769cfdf8e01",
            "symbol": "Indexer.__init__",
            "lines": [96, 96],
        },
        "call_site": {
            "repository": "vllm-ascend",
            "path": "vllm_ascend/attention/indexer_kpool.py",
            "sha256": "786ccb6864ef9299b8b44b61afd446b35b281dc2ca4c4f3d74ae8b7edb3cb57c",
            "symbol": "Glm5NextKPoolIndexerBackend.forward",
            "lines": [763, 770],
        },
        "call_site_variants": [
            {"path": "vllm_ascend/attention/indexer_kpool.py",
             "sha256": "786ccb6864ef9299b8b44b61afd446b35b281dc2ca4c4f3d74ae8b7edb3cb57c",
             "symbol": "Glm5NextKPoolIndexerBackend.forward",
             "lines": [763, 770]},
            {"path": "vllm_ascend/attention/indexer_kpool.py",
             "sha256": "d921f4a30a9be7458bb63597f093981699fd7ba21ac33764e2dbd5140f500c2c",
             "symbol": "Glm5NextKPoolIndexerBackend.forward",
             "lines": [765, 768]},
            {"path": "vllm_ascend/attention/indexer_kpool.py",
             "sha256": "2b6593ff61e3461975e8fe69db7444c9c6b3aac5a2cc95adfee6d776cbe1978b",
             "symbol": "Glm5NextKPoolIndexerBackend.forward",
             "lines": [811, 814]},
            {"path": "vllm_ascend/attention/indexer_kpool.py",
             "sha256": "69bf5b86aea8e970da33bd74cf661687b671f4ffde7e69da097c58c98de30bb8",
             "symbol": "Glm5NextKPoolIndexerBackend.forward",
             "lines": [787, 790]},
            {"path": "vllm_ascend/attention/indexer_kpool.py",
             "sha256": "ebe105f6c00626a200d8176d3768fda00ddfd9c2cca03fff3aed54f14eb7277a",
             "symbol": "Glm5NextKPoolIndexerBackend.forward",
             "lines": [1079, 1082]},
        ],
    },
}
SOURCE_ORDER = {
    "stage": "I0",
    "upstream": ["index_k_projection"],
    "row": "index_k_norm",
    "downstream": ["index_k_rope", "index_cache_store"],
    "then": "I1 index_score/topk",
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}

UNRESOLVED = (
    "acl_wrapper_and_dispatch_unresolved",
    "acl_dispatch_from_python_wrapper_not_retained",
    "caller_shape_lowering_unresolved",
    "layer_norm_impl_mode_value_unresolved",
    "layer_norm_c_header_not_retained",
    "layer_norm_kernel_source_not_retained",
    "layer_norm_binary_selection_unresolved",
    "torch_layer_norm_to_aclnnLayerNormWithImplMode_measured_not_source_declared",
)
_SUPPORTED_DTYPES = frozenset({"fp32", "bf16", "fp16"})
_SOURCE_KEYS = (
    "source_kind", "source_path", "source_symbol", "source_lines",
    "source_sha256", "call_site_path", "call_site_symbol", "call_site_lines",
    "call_site_sha256", "layer_norm_source_path", "layer_norm_source_symbol",
    "layer_norm_source_lines", "layer_norm_source_sha256",
    "layer_norm_dtype_source", "source_stage", "attribution_owner",
)


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


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype == "fp32" else 2


def _dtype(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
    text = aliases.get(text, text)
    return text if text in _SUPPORTED_DTYPES else None


def _normalized_shape(value: Any) -> list[int] | None:
    """A legal declared one-dimensional normalized shape, or ``None``."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    shape = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            extent = int(item)
        except (TypeError, ValueError):
            return None
        if extent < 1:
            return None
        shape.append(extent)
    return shape


def _epsilon(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        eps = float(value)
    except (TypeError, ValueError):
        return None
    return eps if math.isfinite(eps) and eps > 0.0 else None


def _lines(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    first, second = _int(value[0], -1), _int(value[1], -1)
    return [first, second] if first >= 0 and second >= 0 else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha(value: Any) -> str:
    return _text(value).lower()


def _record_matches(declared: dict[str, Any], prefix: str,
                    record: dict[str, Any]) -> bool:
    path = f"{prefix}_path"
    symbol = f"{prefix}_symbol"
    lines = f"{prefix}_lines"
    sha = f"{prefix}_sha256"
    return (
        _text(declared.get(path)) == str(record["path"])
        and _text(declared.get(symbol)) == str(record["symbol"])
        and _lines(declared.get(lines)) == list(record["lines"])
        and _sha(declared.get(sha)) == str(record["sha256"]).lower()
    )


def _selected_variant(declared: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The declared I0 source slot, or ``None`` for unknown/mismatched input."""
    if _text(declared.get("source_kind")) != SOURCE_KIND:
        return None
    if _text(declared.get("source_stage")) != STAGE:
        return None
    for key, variant in SOURCE_VARIANTS.items():
        if not _record_matches(declared, "source", variant["model"]):
            continue
        if not _record_matches(declared, "call_site", variant["call_site"]):
            continue
        if not _record_matches(declared, "layer_norm_source", LAYER_NORM_SOURCE):
            continue
        attribution = _text(declared.get("attribution_owner"))
        if attribution and attribution != f"{STAGE}:indexer:index_k_norm":
            return None
        return key, variant
    return None


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if image_id or image:
        # An explicitly different deployment image is another source.  The
        # operator theta fallback below is only for a case whose metadata
        # carries no image at all, never for overriding a known non-R64 one.
        return meta if source_revision_from_oci(image, image_id) == "r64" else {}
    operator = getattr(compiled, "operator", None)
    theta = getattr(operator, "theta", None)
    if theta is not None:
        image_id = str(theta.find("deployment_image_id", "") or "")
        revision = str(theta.find("deployment_source_revision", "") or "")
        if (source_revision_from_oci("", image_id) == "r64"
                or revision == "r64"):
            return {**meta, "image_id": image_id,
                    "deployment_source_revision": revision}
    return {}


def _declared(compiled: Any) -> tuple[Any, dict[str, Any]] | None:
    """The explicit selected I0 source-slot declaration, or ``None``."""
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = operator.theta
    extr = dict(getattr(theta, "extras", ()) or ())
    if (str(getattr(operator, "name", "") or "") != "index_k_norm"
            or str(getattr(operator, "family", "") or "") != "elementwise"
            or str(getattr(operator, "role", "") or "") != "norm"):
        return None
    declared = {key: theta.find(key, extr.get(key, None))
                for key in _SOURCE_KEYS}
    declared.update({
        "index_dim": _int(theta.find("index_dim",
                                     extr.get("index_dim", 0))),
        "elements": _int(theta.find("elements",
                                    extr.get("elements", 0))),
        "normalized_shape": _normalized_shape(theta.find(
            "normalized_shape", extr.get("normalized_shape", None))),
        "eps": _epsilon(theta.find("norm_eps", extr.get("norm_eps", None))),
        "dtype": _dtype(theta.find("norm_dtype",
                                   extr.get("norm_dtype", None))),
    })
    return operator, declared


def _declared_contract(declared: dict[str, Any]) -> tuple[int, float] | None:
    """One declared legal 1-D width and positive epsilon, or ``None``."""
    shape = declared.get("normalized_shape")
    width = declared.get("index_dim")
    elements = declared.get("elements")
    eps = declared.get("eps")
    if not isinstance(shape, list) or len(shape) != 1:
        return None
    if width != shape[0] or elements != shape[0]:
        return None
    if eps is None:
        return None
    return shape[0], float(eps)


def _source_reference(selected_key: str,
                      variant: dict[str, Any]) -> dict[str, Any]:
    other_key = ("glm5_next" if selected_key == "glm_moe_dsa"
                 else "glm_moe_dsa")
    other = SOURCE_VARIANTS[other_key]
    selected_call = dict(variant["call_site"])
    other_call = dict(other["call_site"])
    return {
        "selected_variant": selected_key,
        "source": dict(variant["model"]),
        "call_site": selected_call,
        "layer_norm": dict(LAYER_NORM_SOURCE),
        "source_stage": STAGE,
        "source_order": dict(SOURCE_ORDER),
        "source_variants": [dict(item) for item in variant["call_site_variants"]],
        "candidate_callers": [
            {**selected_call, "variant": selected_key,
             "status": "selected_declared_source_callsite",
             "dispatch_proven": False},
            {**other_call, "variant": other_key,
             "status": "not_selected_by_declaration",
             "dispatch_proven": False},
        ],
        "known_source_variants": {
            key: {"source": dict(item["model"]),
                  "call_site": dict(item["call_site"])}
            for key, item in SOURCE_VARIANTS.items()
        },
        "device_op_sha256": R64_DEVICE_OP_SHA,
        "source_commit": R64_SOURCE_COMMIT,
        "source_revision": REVISION,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The selected R64 I0 indexer-k ``nn.LayerNorm`` call identity, or ``None``.

    The row must be the applied I0 ``index_k_norm`` producer row with
    ``source_stage="I0"`` and one of the two declared source slots
    (``deepseek_v2.py`` L685 / ``indexer.py`` L335 for GLM-5.1, or
    ``glm5next/attention.py`` L96 / ``indexer_kpool.py`` L763-770 for GLM-5.3).
    The old after-search blanket I1 linkage is not accepted.  Unknown or
    mismatched source slots, missing shape/epsilon/dtype, or an unsupported
    phase return ``None``.
    """
    if str(getattr(compiled, "kind", "") or "") != STAGE:
        return None
    declared_pair = _declared(compiled)
    if declared_pair is None:
        return None
    operator, declared = declared_pair
    selected = _selected_variant(declared)
    if selected is None:
        return None
    variant_key, variant = selected
    contract = _declared_contract(declared)
    if contract is None:
        return None
    width, eps = contract
    dtype = declared["dtype"]
    if dtype is None:
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "")
                     or meta.get("phase") or "")
    if phase_name not in {"prefill", "decode"}:
        return None
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    from mindie_sim.prediction.mx_source_provenance import image_digest
    digest = image_digest(image_id)
    from . import native_calls
    model = variant["model"]
    source = native_calls._source(
        model["repository"], model["path"], sha256=model["sha256"],
        symbol=model["symbol"], revision=REVISION, state="task_artifact")
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        accelerator_core="AI_VECTOR_CORE",
        kernel_source_status="kernel_binary_not_retained")
    reference = _source_reference(variant_key, variant)
    attribution = f"{STAGE}:indexer:index_k_norm"
    typed = {
        "dtype": dtype,
        "layout": "ND",
        "shapes": {
            "input": [width],
            "normalized_shape": [1],
            "gamma": [width],
            "beta": [width],
            "output": [width],
            "mean": [1],
            "rstd": [1],
        },
        "attrs": {
            "attribution_owner": attribution,
            "caller_status": "declared_selected_source_callsite",
            "call_site_lines": list(declared.get("call_site_lines") or []),
            "call_site_path": str(declared.get("call_site_path") or ""),
            "call_site_sha256": str(declared.get("call_site_sha256") or ""),
            "call_site_symbol": str(declared.get("call_site_symbol") or ""),
            "candidate_callers": reference["candidate_callers"],
            "deployment_image_id": digest or "unresolved",
            "deployment_source_revision": "r64",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "dtype": dtype,
            "entry": ENTRY,
            "eps": eps,
            "execution_core": "aiv",
            "gamma_dtype": dtype,
            "impl_mode": None,
            "impl_mode_status": "unresolved_value_not_retained",
            "index_dim": width,
            "input_dtype": dtype,
            "kernel": KERNEL_NAME,
            "kernel_source_status": "kernel_binary_not_retained",
            "layer_norm_dtype_source": str(
                declared.get("layer_norm_dtype_source") or ""),
            "layer_norm_source_lines": list(
                declared.get("layer_norm_source_lines") or []),
            "layer_norm_source_path": str(
                declared.get("layer_norm_source_path") or ""),
            "layer_norm_source_sha256": str(
                declared.get("layer_norm_source_sha256") or ""),
            "layer_norm_source_symbol": str(
                declared.get("layer_norm_source_symbol") or ""),
            "native_acl_api": ACLNN_API,
            "native_kind": KERNEL_NAME,
            "normalized_shape": [width],
            "normalized_shape_rank": 1,
            "output_dtype": dtype,
            "outputs": ["y", "mean", "rstd"],
            "profiler_name": PROFILER_NAME,
            "source_kind": SOURCE_KIND,
            "source_lines": list(declared.get("source_lines") or []),
            "source_path": str(declared.get("source_path") or ""),
            "source_sha256": str(declared.get("source_sha256") or ""),
            "source_stage": STAGE,
            "source_symbol": str(declared.get("source_symbol") or ""),
            "source_variants": reference["source_variants"],
            "source_revision": REVISION,
            "x_dtype": dtype,
        },
        "occurrence_attrs": {"m": "positive_int", "phase": "string"},
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=UNRESOLVED)


def _prepare_arguments(width: int, eps: float) -> tuple[dict[str, Any],
                                                       dict[str, Any]]:
    normalized_shape = [int(width)]
    arguments = {
        "input": "x",
        "normalized_shape": normalized_shape,
        "weight": "gamma",
        "bias": "beta",
        "eps": float(eps),
    }
    aclnn_arguments = {
        "x": "x",
        "normalized_shape": normalized_shape,
        "gamma": "gamma",
        "beta": "beta",
        "epsilon": float(eps),
        "y": "y",
        "mean": "mean",
        "rstd": "rstd",
        "impl_mode": None,
    }
    return arguments, aclnn_arguments


def _work(tokens: int, width: int, dtype: str, phase: str) -> dict[str, Any]:
    """Structural work from the declared shape/dtype; no measured clock."""
    bytes_per = _dtype_bytes(dtype)
    reduction = int(math.ceil(width / 64))
    elements = int(tokens) * int(width)
    return {
        "source": "declared_I0_nn_LayerNorm_shape_dtype",
        "dtype": dtype,
        "dtype_bytes": bytes_per,
        "elements": elements,
        "vector_ops": int(tokens) * (int(width) * 5 + reduction),
        "read_bytes": int(tokens) * int(width) * bytes_per
        + int(width) * bytes_per,
        "write_bytes": int(tokens) * (int(width) * bytes_per + 8),
        "gamma_bytes": int(width) * bytes_per,
        "y_bytes": int(tokens) * int(width) * bytes_per,
        "mean_rstd_bytes": int(tokens) * 8,
        "normalized_shape": [int(width)],
        "source_stage": STAGE,
        "phase": phase,
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R64 I0 K normalization.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The declared 1-D width, epsilon and
    resolved dtype are consumed from the binding; the source phase is the
    geometry phase (prefill or decode).  The row is charged at I0; the fused
    I1 QuantLightningIndexer keeps its own separate charge.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY or api.get("aclnn_symbol") != ACLNN_API:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    shape = _normalized_shape(attrs.get("normalized_shape"))
    eps = _epsilon(attrs.get("eps"))
    dtype = _dtype(attrs.get("dtype"))
    if shape is None or len(shape) != 1 or eps is None or dtype is None:
        return None
    width = shape[0]
    declared = {key: attrs.get(key) for key in _SOURCE_KEYS}
    declared.update({"index_dim": attrs.get("index_dim"),
                     "normalized_shape": shape, "eps": eps, "dtype": dtype})
    selected = _selected_variant(declared)
    if selected is None:
        return None
    variant_key, variant = selected
    if (attrs.get("source_stage") != STAGE
            or attrs.get("index_dim") != width
            or attrs.get("impl_mode") is not None):
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    if phase not in {"prefill", "decode"}:
        return None
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if tokens < 1:
        return None
    cache_key = (binding.call_id, tokens, phase)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "input": [tokens, width],
        "normalized_shape": [1],
        "gamma": [width],
        "beta": [width],
        "output": [tokens, width],
        "mean": [tokens, 1],
        "rstd": [tokens, 1],
    }
    actual_attrs = {
        **attrs,
        "m": tokens,
        "tokens": tokens,
        "phase": phase,
        "input_shape": [tokens, width],
        "output_shape": [tokens, width],
        "mean_shape": [tokens, 1],
        "rstd_shape": [tokens, 1],
    }
    actual = {"dtype": dtype, "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "accelerator_core": "AI_VECTOR_CORE",
                   "source_status": "kernel_binary_not_retained"})
    arguments, aclnn_arguments = _prepare_arguments(width, eps)
    descriptors = {
        "input": _descriptor([tokens, width], dtype, "ND", "input"),
        "normalized_shape": {
            "shape": [1], "dtype": "int32", "layout": "ND",
            "role": "host", "pointer": None, "allocation": "host",
            "kind": "aclIntArray", "values": [width], "length": 1,
        },
        "gamma": _descriptor([width], dtype, "ND", "input"),
        "beta": _descriptor([width], dtype, "ND", "input"),
        "output": _descriptor([tokens, width], dtype, "ND", "output"),
        "mean": _descriptor([tokens, 1], "fp32", "ND", "output"),
        "rstd": _descriptor([tokens, 1], "fp32", "ND", "output"),
    }
    reference = _source_reference(variant_key, variant)
    attribution = f"{STAGE}:indexer:index_k_norm"
    prepare = {
        "schema": "mindie-sim/layer-norm-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "acl_api": ACLNN_API,
        "native_type": KERNEL_NAME,
        "source": source,
        "source_kind": SOURCE_KIND,
        "source_stage": STAGE,
        "source_lines": list(attrs.get("source_lines") or []),
        "source_reference": reference,
        "source_order": dict(SOURCE_ORDER),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "prototype": ("torch.nn.functional.layer_norm(input, normalized_shape, "
                      "weight=None, bias=None, eps=<declared>)"),
        "arguments": arguments,
        "argument_order": ["input", "normalized_shape", "weight", "bias", "eps"],
        "aclnn_arguments": aclnn_arguments,
        "aclnn_argument_order": [
            "x", "normalized_shape", "gamma", "beta", "epsilon",
            "y", "mean", "rstd", "impl_mode"],
        "optional_slots": {
            "weight": "present", "bias": "present",
            "mean": "present", "rstd": "present",
            "impl_mode": "unresolved_value_not_retained",
        },
        "scope": {
            "normalized_shape_rank": 1,
            "normalized_shape": [width],
            "eps": float(eps),
            "dtype": dtype,
            "phase": phase,
            "source_stage": STAGE,
            "phase_support": "explicit prefill or decode; empty phase unbound",
        },
        "work": _work(tokens, width, dtype, phase),
        "shapes": actual_shapes,
        "declared_shapes": dict(binding.typed.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "acl_interface": {
            "symbol": ACLNN_API,
            "evidence": "observed_native_trace_not_the_source_selector",
            "c_header": None,
            "status": "not_retained",
            "impl_mode": None,
            "impl_mode_status": "unresolved_value_not_retained",
        },
        "source_lowering": {
            "python_entry": ENTRY,
            "python_module": "torch.nn.functional",
            "source_stage": STAGE,
            "source_order": dict(SOURCE_ORDER),
            "call_site_candidates": reference["candidate_callers"],
            "selected_caller": dict(variant["call_site"]),
            "selected_caller_status": "declared_by_builder_source_slot",
            "acl_symbol": ACLNN_API,
            "kernel": KERNEL_NAME,
            "evidence": ("selected_I0_source_slot_plus_declared_"
                         "callsite_plus_observed_native_trace"),
            "status": ("source_selected_callsite_acl_wrapper_"
                       "measured_not_source_declared"),
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "call_id": binding.call_id,
            "m": tokens,
            "width": width,
            "normalized_shape": [width],
            "eps": float(eps),
            "dtype": dtype,
            "phase": phase,
            "source_stage": STAGE,
            "x_dtype": dtype,
            "gamma_dtype": dtype,
            "beta_dtype": dtype,
            "output_dtype": dtype,
            "mean_dtype": "fp32",
            "rstd_dtype": "fp32",
            "input_shapes": [actual_shapes["input"], [1], [width], [width]],
            "output_shapes": [actual_shapes["output"], actual_shapes["mean"],
                              actual_shapes["rstd"]],
            "source_kind": SOURCE_KIND,
            "source_path": source.get("path"),
            "source_sha256": source.get("sha256"),
            "source_symbol": source.get("symbol"),
            "source_lines": list(attrs.get("source_lines") or []),
            "call_site_path": str(attrs.get("call_site_path") or ""),
            "call_site_symbol": str(attrs.get("call_site_symbol") or ""),
            "call_site_lines": list(attrs.get("call_site_lines") or []),
            "layer_norm_source_path": str(
                attrs.get("layer_norm_source_path") or ""),
            "layer_norm_source_sha256": str(
                attrs.get("layer_norm_source_sha256") or ""),
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or "unresolved"),
            "deployment_source_revision": "r64",
            "caller_status": "declared_selected_source_callsite",
            "attribution_owner": attribution,
            "impl_mode": None,
            "complete": False,
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": REVISION,
            "selection": ("selected I0 nn.LayerNorm source slot + R64 "
                          "device_op.py anchor + declared shape/eps/dtype + "
                          "evaluated source phase"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_stage": STAGE,
            "source_commit": R64_SOURCE_COMMIT,
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "source_variants": reference["source_variants"],
            "caller_status": "declared_selected_source_callsite",
            "candidate_callers": reference["candidate_callers"],
        },
        "binary_selection": {
            "status": "observed_name_not_binary_identity",
            "reason": "layer_norm_kernel_binary_not_retained",
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": list(UNRESOLVED),
        "unresolved": list(UNRESOLVED) + ["execution_rank_not_selected"],
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration", "no_acl_wrapper_claim",
                       "no_kernel_source_claim"],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": dtype, "layout": "ND",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
