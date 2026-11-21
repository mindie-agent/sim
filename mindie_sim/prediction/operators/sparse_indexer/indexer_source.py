"""Shared source-typed selector for the A5 indexer families.
[Role] Decide whether the deployed typed caller selected the R64 C8
  QuantLightningIndexer (FP8 q/k + F32 qscale), the BF16 LightningIndexer V2
  caller, or a different C4/V2/absent scale path.  The selection comes from an
  explicit upstream source declaration attached by the normal builder, never
  from anonymous byte counts, attention kv_dtype, candidate counts or a
  model/case name.
[Invariants] Only exact ``A5DeviceAdaptor.indexer_select_post_process``
  source identity + R64 image + FP8 q/k + present F32 query scale + TND/PA_BSND
  q/key layout + query/key quant mode 0 is C8. The BF16 LI caller is a distinct
  source family: it requires BF16 q/k, TND query and paged key, and never a
  query scale or INT8 representation. C4/V2, absent qscale, other
  images/sources remain non-C8; no defaults are invented.
[Timeouts & Locks] Constant-time metadata inspection; no I/O or locks.
[Side-Effect Policy] Pure metadata.
"""
from __future__ import annotations

from typing import Any, Mapping

SCHEMA = "mindie-sim/indexer-source/1"
SOURCE_FAMILY = "A5DeviceAdaptor.indexer_select_post_process"
BF16_SOURCE_FAMILY = "vllm_ascend.models.qwen4_exp.lightning_indexer"
BF16_WEIGHT_DTYPES = frozenset({"fp32", "bf16", "fp16"})
_DTYPE_ALIASES = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
R64_IMAGE = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"


def _image(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _select_bf16(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the explicit BF16 LightningIndexerV2 caller."""

    def normalize(value):
        text = str(value or "").strip().lower()
        return _DTYPE_ALIASES.get(text, text)

    q_dtype = normalize(declaration.get("q_dtype"))
    k_dtype = normalize(declaration.get("k_dtype"))
    weight_dtype = normalize(declaration.get("weight_dtype")
                             or declaration.get("weights_dtype") or "fp32")
    layout_query = str(declaration.get("layout_query") or "")
    layout_key = str(declaration.get("layout_key") or "")
    if not q_dtype or not k_dtype:
        return {"schema": SCHEMA, "status": "unknown_source",
                "reason": "bf16_qk_dtype_absent"}
    if q_dtype != "bf16" or k_dtype != "bf16":
        return {"schema": SCHEMA, "status": "not_bf16",
                "reason": f"qk_dtype_not_bf16:{q_dtype}/{k_dtype}"}
    if declaration.get("query_scale_present") is True:
        return {"schema": SCHEMA, "status": "not_bf16",
                "reason": "bf16_declaration_has_query_scale"}
    if weight_dtype not in BF16_WEIGHT_DTYPES:
        return {"schema": SCHEMA, "status": "not_bf16",
                "reason": f"weight_dtype_not_supported:{weight_dtype}"}
    if layout_query != "TND" or layout_key not in {"PA_BBND", "ND"}:
        return {"schema": SCHEMA, "status": "not_bf16",
                "reason": "layout_not_tnd_paged_key"}
    return {"schema": SCHEMA, "status": "bf16_li_v2",
            "source_family": BF16_SOURCE_FAMILY, "q_dtype": q_dtype,
            "k_dtype": k_dtype, "weight_dtype": weight_dtype,
            "layout_query": layout_query, "layout_key": layout_key,
            "query_scale_present": False}


def select_indexer_source(declaration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify one explicit upstream indexer declaration."""
    if not isinstance(declaration, Mapping) or not declaration:
        return {"schema": SCHEMA, "status": "unknown_source",
                "reason": "indexer_source_declaration_absent"}
    family = str(declaration.get("source_family") or "")
    if family == BF16_SOURCE_FAMILY:
        return _select_bf16(declaration)
    if family != SOURCE_FAMILY:
        return {"schema": SCHEMA, "status": "unknown_source",
                "reason": f"source_family_not_selected:{family or 'absent'}"}
    image = _image(declaration.get("image_id"))
    if image != R64_IMAGE:
        return {"schema": SCHEMA, "status": "unknown_deployment",
                "reason": "not_r64_image"}
    q_dtype = str(declaration.get("q_dtype") or "")
    k_dtype = str(declaration.get("k_dtype") or "")
    qscale_present = declaration.get("query_scale_present")
    qscale_dtype = str(declaration.get("query_scale_dtype") or "")
    layout_query = str(declaration.get("layout_query") or "")
    layout_key = str(declaration.get("layout_key") or "")
    qmode = declaration.get("query_quant_mode")
    kmode = declaration.get("key_quant_mode")
    if (q_dtype in {"fp4", "fp4_e2m1", "f4"} or k_dtype in {"fp4", "fp4_e2m1", "f4"}
            or qscale_dtype in {"fp8_e8m0", "e8m0"}):
        return {"schema": SCHEMA, "status": "c4_v2_not_supported",
                "reason": "c4_v2_metadata_path"}
    if q_dtype != "fp8_e4m3" or k_dtype != "fp8_e4m3":
        return {"schema": SCHEMA, "status": "not_c8",
                "reason": f"qk_dtype_not_fp8:{q_dtype}/{k_dtype}"}
    if qscale_present is not True:
        return {"schema": SCHEMA, "status": "not_c8",
                "reason": "query_scale_absent"}
    if qscale_dtype != "fp32":
        return {"schema": SCHEMA, "status": "not_c8",
                "reason": f"query_scale_dtype_not_fp32:{qscale_dtype}"}
    if layout_query != "TND" or layout_key != "PA_BSND":
        return {"schema": SCHEMA, "status": "not_c8",
                "reason": "layout_not_tnd_pa_bsnd"}
    if qmode != 0 or kmode != 0:
        return {"schema": SCHEMA, "status": "not_c8",
                "reason": "query_key_quant_mode_not_0"}
    return {"schema": SCHEMA, "status": "c8_r64",
            "source_family": SOURCE_FAMILY, "image_id": image,
            "q_dtype": q_dtype, "k_dtype": k_dtype,
            "query_scale_present": True, "query_scale_dtype": qscale_dtype,
            "layout_query": layout_query, "layout_key": layout_key,
            "query_quant_mode": 0, "key_quant_mode": 0}


def declaration_from_theta(theta: Any, feature_metadata: Any = ()) -> dict[str, Any] | None:
    """Read the explicit upstream declaration attached by the I1 builder."""
    value = theta.find("indexer_source", None)
    if isinstance(value, Mapping):
        declaration = dict(value)
    elif isinstance(value, (tuple, list)) and all(
            isinstance(item, (tuple, list)) and len(item) == 2
            for item in value):
        declaration = dict(value)
    else:
        return None
    meta = dict(feature_metadata or ())
    if not declaration.get("image_id"):
        declaration["image_id"] = (meta.get("image_id")
                                   or meta.get("oci_image_digest")
                                   or meta.get("deployment_image_id"))
    return declaration
