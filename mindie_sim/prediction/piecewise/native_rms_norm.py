"""R31 plain initial Gemma RMSNorm exact host/ACLNN call.

[Role] Bind the genuine R31 Qwen3.5/6 decoder's first residual-free
  ``input_layernorm`` host path (``GemmaRMSNorm`` ->
  ``AscendGemmaRMSNorm.forward_oot`` -> ``DeviceOperator.npu_gemma_rms_norm``
  -> ``torch_npu.npu_rms_norm``) and expand its two actual outputs from the
  current compiled tensor and evaluated geometry.
[Invariants] Source exactness is granted only by affirmatively declared and
  matching fields: the R31 vllm-ascend package/commit, the vLLM source commit,
  the exact source SHAs, the ``AscendGemmaRMSNorm`` caller class, source weight
  offset 1.0, initial ``input_layernorm`` role, qwen3_5 model type and A5
  arch, plus a structurally plain residual-free N0.  Empty, partial or
  conflicting metadata returns ``None``; no default constants synthesize an
  image, source class or model role.  The same-image host summary
  ``api_statistic.csv`` sha256 ``e19308de...`` affirmatively records
  ``aclnnRmsNorm`` 1 / ``aclnnRmsNormGetWorkspaceSize`` 1 /
  ``aclnnRmsNormTiling`` 1, so the ACLNN symbol is source-declared rather
  than a candidate or an ELF-digest claim.  The preceding ``1.0 + weight``
  Adds is a separate native call and is not charged here.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata from the retained source bundle and the
  existing compiled operator; no tensor, device, weights or profiler import.
"""
from __future__ import annotations

from typing import Any

SCHEMA = "mindie-sim/rms-norm-gemma-exact/1"

R31_IMAGE_ID = (
    "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_TAG = (
    "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
    "nightly-ci-releases-v0.30.0-a5-cann9.2.0_20261022115200-20261023-r31")
R31_REVISION = "r31/8fd0bc488"
VLLM_ASCEND_REVISION = "8fd0bc4885cb52e58791aaa8efacc571b9bd115d"
VLLM_ASCEND_PACKAGE = "0.30.0rc2.dev200+g8fd0bc488"
VLLM_REVISION = "58d3918e3ea0a544ffedadad2ba84559e9c51d8f"
HOST_ENTRY = "torch_npu.npu_rms_norm"
ACLNN_SYMBOL = "aclnnRmsNorm"
KERNEL = "RmsNorm"
PROFILER_NAME = "RmsNorm"
OBSERVED_CORE = "AI_VECTOR_CORE"

VLLM_ASCEND_LAYERNORM_SHA = (
    "1770057c3cd170a7433910fc02db2a36bf0443eb218215a243db31761b95e4a6")
VLLM_ASCEND_DEVICE_OP_SHA = (
    "f4392f346816c85f138d7e033a924c6b628ae684cd4fcd7a6d8818518351b5c9")
VLLM_ASCEND_UTILS_SHA = (
    "69c0b69ef9b80c08b772d191e8e66e8857c5c1ea68e719ddc91b1a197f7ee8c3")
VLLM_ASCEND_WORKER_SHA = (
    "773d289e29301ea176822efcf65eb8df48a6f778476bdbadb1d56e5ea668cdc6")
VLLM_QWEN3_5_SHA = (
    "b880f4fd4dc403d8c220d501f00701a1ed8572890b7508442a641ed650f8652b")
VLLM_QWEN3_NEXT_SHA = (
    "e195dc4d5522a8cce5399cb35d9d96d6e5a0c07bf9460ef6912c840dcb7a6dc9")
VLLM_LAYERNORM_SHA = (
    "3460eba01758849027bb59b8592626406e33ab25b23d2def729bb0ec42afce67")
VLLM_CUSTOM_OP_SHA = (
    "967b74c14b8a8e69af8828300414fbd3a6954ec573a253ca0f67aa02b0ed473f")
LIBOPAPI_PATH = (
    "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so")
LIBOPAPI_SHA256 = (
    "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456")
ORIGINAL_RUNTIME_STATUS = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/smoke_qwen36_single/status.json")
ORIGINAL_RUNTIME_STATUS_SHA256 = (
    "6e28777be4e6fef72d2d71c69bf27ee9e85d307e173e8e126f704074c1971063")
SOURCE_RAW_PATH = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
    "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz")
SOURCE_RAW_SHA256 = (
    "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a")
API_STATISTIC_PATH = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
    "rank0_545_20261107032704817_ascend_pt/api_statistic.csv")
API_STATISTIC_SHA256 = (
    "e19308dea200ead700b417475da5983576ec8884646e3b03e833be6b06e0808e")
RAW_GAMMA_SHIFT_ROW = 45
RAW_RMS_NORM_ROW = 46
MODEL_CONFIG_SHA256 = (
    "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c")
HOST_API_COUNTS = {
    "aclnnRmsNorm": 1,
    "aclnnRmsNormGetWorkspaceSize": 1,
    "aclnnRmsNormTiling": 1,
    "aclnnAddRmsNorm": 128,
}
RSTD_DTYPE = "fp32"
RSTD_ROW_BYTES = 4

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _descriptor(shape: list[int], dtype: str, layout: str, role: str,
                *, materialized: bool = False) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external", "materialized": bool(materialized)}


def _image_matches(value: str) -> bool:
    return value == R31_IMAGE_ID or value.startswith(
        "sha256:9e869e81") or value.startswith("9e869e81")


INITIAL_ROLES = {
    "input_layernorm.initial", "input_layernorm",
    "initial_input_layernorm", "attn_norm", "attn_norm.initial",
    "initial_attn_norm", "initial_norm",
}
INITIAL_BLOCKS = {
    "initial", "input_layernorm", "input_layernorm.initial",
    "attn_norm", "initial_norm",
}
_TRUE_STRINGS = {"true", "1", "yes", "present", "residual", "add",
                 "residual_add", "fused"}


def _truthy(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in {"false", "0", "no", "absent", "none", "no_residual",
                    "residual_absent"}:
            return False
    return None


def _source_contract(compiled: Any) -> dict[str, Any] | None:
    """Normalized Gemma class semantics or exact archived R31 provenance.

    The normalized class/role/residual/weight-offset/block are the semantic
    gate.  Exact R31 image/commit/file/ELF facts are used only when the full
    archived wrapper provenance is affirmatively present and matching; a
    generic real Gemma class on another deployment still binds the shared
    operator semantics without asserting those R31 facts.  Empty metadata
    returns ``None``; shape, version or A5 alone never infers Gemma.
    """
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    source_class = _text(meta.get("rms_norm_source_class")
                         or meta.get("norm_source_class"))
    weight_offset = _as_float(meta.get("rms_norm_source_weight_offset",
                                      meta.get("source_weight_offset")))
    role = _text(meta.get("rms_norm_source_role")
                 or meta.get("source_role"))
    residual = meta.get("rms_norm_source_residual",
                        meta.get("norm_source_residual"))
    block = _text(meta.get("rms_norm_source_block")
                  or meta.get("norm_source_block"))
    beta = meta.get("rms_norm_source_beta",
                    meta.get("rms_norm_source_bias_present"))
    if source_class != "AscendGemmaRMSNorm":
        return None
    if weight_offset is None or abs(weight_offset - 1.0) > 1e-9:
        return None
    if role is None or role not in INITIAL_ROLES:
        return None
    if residual is not None:
        flag = _truthy(residual)
        if flag is None or flag:
            return None
    if block is not None and block not in INITIAL_BLOCKS:
        return None
    if beta is not None and _truthy(beta):
        return None

    asc_commit = _text(meta.get("vllm_ascend_commit")
                       or meta.get("vllm_ascend_revision")
                       or meta.get("source_vllm_ascend_commit"))
    asc_package = _text(meta.get("vllm_ascend_package")
                        or meta.get("vllm_ascend_version"))
    layernorm_sha = _text(meta.get("vllm_ascend_layernorm_sha256"))
    device_op_sha = _text(meta.get("vllm_ascend_device_op_sha256"))
    vllm_commit = _text(meta.get("vllm_commit")
                        or meta.get("vllm_revision")
                        or meta.get("source_vllm_commit"))
    qwen3_5_sha = _text(meta.get("vllm_qwen3_5_sha256"))
    model_type = _text(meta.get("source_model_type")
                       or meta.get("model_type"))
    arch = _text(meta.get("arch_family") or meta.get("hardware_arch"))
    image_id = _text(meta.get("image_id") or meta.get("oci_image_digest")
                     or meta.get("deployment_image_id"))
    image = _text(meta.get("image") or meta.get("deployment_image")
                  or meta.get("oci_image"))

    exact_wrapper = (
        asc_commit is not None and asc_commit.startswith("8fd0bc488")
        and asc_package is not None and "g8fd0bc488" in asc_package
        and layernorm_sha == VLLM_ASCEND_LAYERNORM_SHA
        and device_op_sha == VLLM_ASCEND_DEVICE_OP_SHA)
    exact_model = (
        exact_wrapper
        and vllm_commit is not None and vllm_commit.startswith("58d3918e")
        and qwen3_5_sha == VLLM_QWEN3_5_SHA
        and model_type is not None and model_type.startswith("qwen3_5")
        and arch is not None
        and (arch.startswith("A5")
             or arch in {"Ascend950DT", "Ascend950DT_9582"}))
    exact_image = (image_id is None or _image_matches(image_id))
    exact_r31 = bool(exact_model and exact_image)
    if not exact_model and image_id is not None and image is not None:
        # A declared image on a non-R31 deployment is provenance, never an
        # exact-image claim; do not reject the generic class semantics.
        pass
    wrapper = {}
    if exact_wrapper:
        wrapper = {
            "vllm_ascend_commit": asc_commit,
            "vllm_ascend_package": asc_package,
            "layernorm_sha256": layernorm_sha,
            "device_op_sha256": device_op_sha,
            "tier": "exact_archived_r31_wrapper",
        }
    exact = {}
    if exact_r31:
        exact = {
            "vllm_ascend_commit": asc_commit,
            "vllm_ascend_package": asc_package,
            "layernorm_sha256": layernorm_sha,
            "device_op_sha256": device_op_sha,
            "vllm_commit": vllm_commit,
            "qwen3_5_sha256": qwen3_5_sha,
            "model_type": model_type,
            "arch_family": arch,
        }
    return {
        "tier": "exact_r31_wrapper" if exact_r31 else "generic_normalized_gemma",
        "exact_r31": exact_r31,
        "exact_wrapper": exact_wrapper,
        "source_class": "AscendGemmaRMSNorm",
        "weight_offset": 1.0,
        "role": role,
        "residual": "absent",
        "block": block or "initial",
        "model_type": model_type,
        "arch_family": arch,
        "image_id": image_id if exact_r31 else None,
        "declared_image": image_id,
        "image": image,
        "exact": exact,
        "wrapper": wrapper,
    }


def _plain_initial(compiled: Any) -> bool:
    """Structurally plain residual-free initial N0 RMS role and tensor ABI."""
    if str(getattr(compiled, "kind", "") or "") != "N0":
        return False
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return False
    if (str(getattr(operator, "family", "") or "") != "elementwise"
            or str(getattr(operator, "name", "") or "") != "rmsnorm"
            or str(getattr(operator, "role", "") or "") != "norm"):
        return False
    if any(str(getattr(item, "name", "") or "") == "residual_add"
           for item in (getattr(compiled, "siblings", ()) or ())):
        return False
    theta = operator.theta
    if int(theta.find("layernorm", 0) or 0):
        return False
    if str(theta.find("norm_arch", "") or "") not in {"rmsnorm", "gemma"}:
        return False
    offset = _as_float(theta.find("norm_weight_offset", 0.0))
    if offset not in {0.0, 1.0}:
        return False
    if theta.find("norm_has_bias", False) is True:
        return False
    if bool(int(theta.find("quantized", 0) or 0)):
        return False
    if int(theta.find("quant_passes", 0) or 0):
        return False
    if str(theta.find("native_norm_kind", "") or "") in {
            "AddRmsNormBias", "AddRmsNormDynamicQuant",
            "AddRmsNormDynamicMxQuant"}:
        return False
    hidden = int(theta.find("elements", 0) or 0)
    gamma_format = str(theta.find("gamma_format", "") or "")
    gamma_bytes = theta.find("gamma_bytes", None)
    if hidden < 1 or gamma_format not in _DTYPE_BYTES or gamma_bytes is None:
        return False
    try:
        gamma = int(gamma_bytes)
    except (TypeError, ValueError):
        return False
    return gamma == hidden * _DTYPE_BYTES[gamma_format]


def _source_record(contract: dict[str, Any]) -> dict:
    from . import native_calls
    exact = contract["exact"]
    return native_calls._source(
        "vllm-ascend", "vllm_ascend/ops/layernorm.py",
        sha256=exact["layernorm_sha256"],
        symbol="AscendGemmaRMSNorm.forward_oot",
        revision=R31_REVISION, state="task_artifact")


def _kernel_record() -> dict:
    from . import native_calls
    return native_calls._kernel(
        KERNEL, "vector", path=LIBOPAPI_PATH, sha256=LIBOPAPI_SHA256,
        evidence="measured", acl_candidate=ACLNN_SYMBOL,
        profiler_name=PROFILER_NAME, observed_core=OBSERVED_CORE,
        kernel_source_status="runtime_binary_only")


def _api_exact(contract: dict[str, Any]) -> dict:
    from . import native_calls
    api = native_calls._api("torch", HOST_ENTRY,
                            aclnn_symbol=ACLNN_SYMBOL,
                            version=contract["exact"]["vllm_ascend_package"])
    api["candidates"] = []
    api["acl_symbol_status"] = "affirmed_by_original_host_summary"
    api["acl_symbol_evidence"] = (
        "api_statistic.csv sha256 " + API_STATISTIC_SHA256
        + ": aclnnRmsNorm 1, aclnnRmsNormGetWorkspaceSize 1, "
          "aclnnRmsNormTiling 1")
    return api


def _api_generic(contract: dict[str, Any]) -> dict:
    from . import native_calls
    version = (contract["wrapper"].get("vllm_ascend_package")
               if contract["exact_wrapper"] else None)
    api = native_calls._api("torch", HOST_ENTRY, version=version)
    api["aclnn_symbol"] = None
    api["candidates"] = []
    api["acl_symbol_status"] = "generic_deployment_host_variant_uncertain"
    api["acl_symbol_evidence"] = (
        "normalized AscendGemmaRMSNorm semantics; no exact deployment "
        "api_statistic or caller ELF claimed")
    return api


def _typed(operator: Any, contract: dict[str, Any]) -> dict:
    theta = operator.theta
    hidden = int(theta.find("elements", 0) or 0)
    dtype = str(theta.find("gamma_format", "") or "")
    eps = float(theta.find("norm_eps", 0.0) or 0.0)
    return {
        "dtype": dtype,
        "layout": "ND",
        "shapes": {"weight": [hidden]},
        "attrs": {
            "elements": hidden,
            "gamma_format": dtype,
            "gamma_bytes": float(hidden * _DTYPE_BYTES[dtype]),
            "norm_eps": eps,
            "output_dtype": dtype,
            "source_class": contract["source_class"],
            "source_weight_offset": 1.0,
            "gamma_operand": "1.0 + input_layernorm.weight",
            "source_host_entry": HOST_ENTRY,
            "source_acl_symbol": (ACLNN_SYMBOL if contract["exact_r31"]
                                  else None),
            "source_role": contract["role"],
            "source_residual": "absent",
            "source_block": contract["block"],
            "source_model_type": contract["model_type"],
            "source_arch_family": contract["arch_family"],
            "source_provenance_tier": contract["tier"],
        },
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Normalized Gemma initial RMS call, exact only when fully archived.

    The generic class path is an honest shared operator/ABI semantic scope:
    host entry known, exact deployment host variant/ACLNN/ELF not claimed.
    Empty metadata and explicit class/offset/block/residual conflicts return
    ``None``; contrary provenance never overrides the declared class.
    """
    contract = _source_contract(compiled)
    if contract is None or not _plain_initial(compiled):
        return None
    from . import native_calls
    operator = compiled.operator
    typed = _typed(operator, contract)
    if contract["exact_r31"]:
        return native_calls._call_spec(
            _api_exact(contract), _source_record(contract), typed,
            (_kernel_record(),), (), str(operator.name),
            execution_core="aiv")
    return native_calls._call_spec(
        _api_generic(contract), None, typed, (), (), str(operator.name),
        unresolved=(
            "generic_host_variant_uncertain",
            "generic_source_not_exact_for_deployment",
        ))


def _rows(compiled: Any, geometry: Any) -> int | None:
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    theta = compiled.operator.theta
    sequence = str(theta.find("sequence_rows", "") or "")
    ranks = int(theta.find("sequence_ranks", 1) or 1)
    if tokens < 1 or ranks < 1:
        return None
    if sequence == "shard" and ranks > 1:
        return _ceil_div(tokens, ranks)
    if sequence == "selected":
        producer = getattr(geometry, "producing_lm_head_rows", None)
        if not callable(producer):
            return None
        rows = int(producer())
        return rows if rows > 0 else None
    return tokens


def _source_trace_exact(contract: dict[str, Any]) -> dict[str, Any]:
    exact = contract["exact"]
    return {
        "vllm": {
            "repository": "vllm",
            "revision": exact["vllm_commit"],
            "qwen3_5": {
                "path": "vllm/model_executor/models/qwen3_5.py",
                "sha256": exact["qwen3_5_sha256"],
                "lines": {"class_alias": 39, "input_layernorm": [181, 184]},
            },
            "qwen3_next": {
                "path": "vllm/model_executor/models/qwen3_next.py",
                "sha256": VLLM_QWEN3_NEXT_SHA,
                "lines": {"initial_residual_none": [504, 508],
                          "post_attention": 545, "final_norm": 709},
            },
            "layernorm": {
                "path": "vllm/model_executor/layers/layernorm.py",
                "sha256": VLLM_LAYERNORM_SHA,
                "lines": {"gemma_class": [132, 166]},
            },
            "custom_op": {
                "path": "vllm/model_executor/custom_op.py",
                "sha256": VLLM_CUSTOM_OP_SHA,
                "lines": {"forward": [133, 136], "dispatch": [174, 205],
                          "register_oot": [339, 362]},
            },
        },
        "vllm_ascend": {
            "repository": "vllm-ascend",
            "revision": exact["vllm_ascend_commit"],
            "package": exact["vllm_ascend_package"],
            "layernorm": {
                "path": "vllm_ascend/ops/layernorm.py",
                "sha256": exact["layernorm_sha256"],
                "lines": {"class": 104, "forward_oot": [105, 123],
                          "plain": 121, "fused": [110, 119]},
            },
            "device_op": {
                "path": "vllm_ascend/device/device_op.py",
                "sha256": exact["device_op_sha256"],
                "lines": {"npu_gemma_rms_norm": 1881},
            },
            "utils": {
                "path": "vllm_ascend/utils.py",
                "sha256": VLLM_ASCEND_UTILS_SHA,
                "lines": {"mapping": 885, "register_oot": [950, 951]},
            },
            "worker": {
                "path": "vllm_ascend/worker/worker.py",
                "sha256": VLLM_ASCEND_WORKER_SHA,
                "lines": {"register_call": 152},
            },
        },
        "host_summary": {
            "path": API_STATISTIC_PATH,
            "sha256": API_STATISTIC_SHA256,
            "counts": dict(HOST_API_COUNTS),
            "kernel_owner": {"RmsNorm": 1},
            "selection": "aclnnRmsNorm source-declared",
        },
        "original_runtime_manifest": {
            "path": ORIGINAL_RUNTIME_STATUS,
            "sha256": ORIGINAL_RUNTIME_STATUS_SHA256,
            "image_tag": R31_IMAGE_TAG,
            "preflight_image": R31_IMAGE_ID,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "enable_flashcomm1": True,
            "model": "Qwen3.6-27B-W8A8",
            "model_type": exact["model_type"],
            "arch_family": exact["arch_family"],
        },
        "raw_rows": {
            "raw_sha256": SOURCE_RAW_SHA256,
            "gamma_shift": RAW_GAMMA_SHIFT_ROW,
            "rms_norm": RAW_RMS_NORM_ROW,
            "native_kernel": KERNEL,
        },
    }


def _source_trace_generic(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_source": {
            "class": contract["source_class"],
            "role": contract["role"],
            "residual": "absent",
            "weight_offset": 1.0,
            "block": contract["block"],
            "model_type": contract["model_type"],
            "arch_family": contract["arch_family"],
            "provenance_tier": "generic_normalized",
        },
        "wrapper_provenance": (dict(contract["wrapper"])
                               if contract["exact_wrapper"] else None),
        "host_summary": None,
        "original_runtime_manifest": None,
        "raw_rows": None,
        "qwen_source": None,
        "non_claims": [
            "no_exact_image",
            "no_exact_acl_symbol",
            "no_qwen_source_for_unrelated_deployment",
            "no_measured_kernel_elf",
        ],
    }


def _source_trace(contract: dict[str, Any]) -> dict[str, Any]:
    if contract["exact_r31"]:
        return _source_trace_exact(contract)
    return _source_trace_generic(contract)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare the normalized Gemma initial RMS call and its two outputs.

    Exact R31 provenance yields the archived host/ACLNN declaration.  A
    generic normalized Gemma class yields the shared host/operator/ABI
    semantics with an explicitly uncertain deployment host variant: no image,
    ACLNN, Qwen-source or measured-ELF claim.  Both preserve dynamic geometry.
    """
    contract = _source_contract(compiled)
    if contract is None or not _plain_initial(compiled):
        return None
    exact = contract["exact_r31"]
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != HOST_ENTRY:
        return None
    if exact and api.get("aclnn_symbol") != ACLNN_SYMBOL:
        return None
    if not exact and api.get("aclnn_symbol") is not None:
        return None
    theta = compiled.operator.theta
    hidden = int(theta.find("elements", 0) or 0)
    dtype = str(theta.find("gamma_format", "") or "")
    if hidden < 1 or dtype not in _DTYPE_BYTES:
        return None
    rows = _rows(compiled, geometry)
    if rows is None or rows < 1:
        return None
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    cache_key = (str(getattr(binding, "call_id", "") or ""), tokens, rows,
                 hidden, dtype, contract["tier"], contract["exact_r31"],
                 contract["declared_image"], contract["role"],
                 contract["block"], contract["model_type"],
                 contract["arch_family"])
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dtype_bytes = _DTYPE_BYTES[dtype]
    gamma_bytes = hidden * dtype_bytes
    shapes = {"x": [rows, hidden], "weight": [hidden],
              "output": [rows, hidden], "rstd": [rows, 1]}
    attrs = dict(binding.typed.get("attrs") or {})
    attrs.update({
        "tokens": tokens, "rows": rows,
        "rstd_dtype": RSTD_DTYPE, "rstd_row_bytes": RSTD_ROW_BYTES,
        "output_bytes": rows * hidden * dtype_bytes,
        "rstd_bytes": rows * RSTD_ROW_BYTES,
        "gamma_shift_elements": hidden,
        "gamma_shift_read_bytes": gamma_bytes,
        "gamma_shift_write_bytes": gamma_bytes,
    })
    actual = {"dtype": dtype, "layout": "ND",
              "shapes": shapes, "attrs": attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, hidden], dtype, "ND", "input"),
        "weight": _descriptor([hidden], dtype, "ND", "input"),
        "output": _descriptor([rows, hidden], dtype, "ND", "output",
                              materialized=True),
        "rstd": _descriptor([rows, 1], RSTD_DTYPE, "ND", "output",
                            materialized=True),
    }
    eps = float(attrs.get("norm_eps", 0.0) or 0.0)
    host_suffix = (
        "GemmaRMSNorm -> AscendGemmaRMSNorm.forward_oot(residual=None) -> "
        "DeviceOperator.npu_gemma_rms_norm -> torch_npu.npu_rms_norm("
        "x, 1.0 + weight, eps)")
    wrapper_package = (contract["wrapper"].get("vllm_ascend_package")
                       if contract["exact_wrapper"] else None)
    prepare = {
        "schema": SCHEMA,
        "host_api": HOST_ENTRY,
        "native_kernel": KERNEL,
        "selected_host_entry": HOST_ENTRY,
        "selected_acl_symbol": ACLNN_SYMBOL if exact else None,
        "acl_symbol_status": ("affirmed_by_original_host_summary" if exact
                              else "generic_deployment_host_variant_uncertain"),
        "host_api_mapped": True,
        "acl_symbol_mapped": exact,
        "acl_symbol_complete": exact,
        "mapped": exact,
        "source_exactness": exact,
        "general_semantics": True,
        "host_variant_uncertain": not exact,
        "abi_binding_reality": exact,
        "call_id": getattr(binding, "call_id", None),
        "source": dict(getattr(binding, "source", None) or {}),
        "kernel": (dict(binding.kernels[0])
                   if getattr(binding, "kernels", ()) else None),
        "source_version": (contract["exact"]["vllm_ascend_package"]
                           if exact else wrapper_package),
        "source_tier": contract["tier"],
        "source_trace": _source_trace(contract),
        "host_path": host_suffix,
        "gamma_source": {
            "operation": "1.0 + input_layernorm.weight",
            "operand": "shifted_gamma",
            "class": "AscendGemmaRMSNorm",
            "device_op": "DeviceOperator.npu_gemma_rms_norm",
            "host_entry": HOST_ENTRY,
        },
        "preceding_gamma_shift": {
            "source": "AscendGemmaRMSNorm/DeviceOperator.npu_gemma_rms_norm",
            "operation": "1.0 + input_layernorm.weight",
            "native_type": "Add",
            "profiler_name": ("aclnnAdds_AddAiCore_Add" if exact else None),
            "evidence": ("archived_r31_raw_row" if exact
                         else "class_semantics"),
            "separate_native_call": True,
            "charge": "not_this_call",
            "elements": hidden,
            "read_bytes": gamma_bytes,
            "write_bytes": gamma_bytes,
            "raw_row": RAW_GAMMA_SHIFT_ROW if exact else None,
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {"x": "x", "gamma": "weight", "epsilon": eps,
                      "y": "output", "rstd": "rstd"},
        "argument_order": ["x", "gamma", "epsilon", "y", "rstd"],
        "prototype_status": ("host_and_acl_symbol_source_declared" if exact
                             else "general_shared_semantics_host_variant_uncertain"),
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "outputs": {"y": descriptors["output"],
                    "rstd": descriptors["rstd"]},
        "physical_work": {
            "role": "plain_initial_input_layernorm",
            "x_read_bytes": rows * hidden * dtype_bytes,
            "gamma_read_bytes": gamma_bytes,
            "y_write_bytes": rows * hidden * dtype_bytes,
            "rstd_write_bytes": rows * RSTD_ROW_BYTES,
            "hbm_bytes": (rows * hidden * dtype_bytes * 2
                          + gamma_bytes + rows * RSTD_ROW_BYTES),
            "observed_core": OBSERVED_CORE if exact else None,
            "kernel_evidence": "measured" if exact else None,
            "captured_dimensions_in_test": (
                "exact R31 raw row 46 shapes/dtypes/block, row 45 gamma shift "
                "and api_statistic counts are asserted in "
                "tests/test_native_rms_norm.py; they are not selector keys "
                "or ABI defaults"
                if exact else
                "generic dynamic semantics only; no R31 raw-row or ELF claim"),
        },
        "materialization": {
            "norm_output_bytes": rows * hidden * dtype_bytes,
            "rstd_bytes": rows * RSTD_ROW_BYTES,
            "total_bytes": rows * (hidden * dtype_bytes + RSTD_ROW_BYTES),
            "rows": rows,
            "rstd_shape": [rows, 1],
            "rstd_dtype": RSTD_DTYPE,
            "basis": "dynamic geometry; captured dimensions are test evidence only",
        },
        "ordinary_cost_note": {
            "rstd_bytes": rows * RSTD_ROW_BYTES,
            "status": "pending shared builder/family rstd_extra_bytes",
            "gamma_shift": "separate Adds call, not charged here",
            "call_complete": True,
            "ordinary_cost_complete": False,
        },
        "match": {
            "native_type": KERNEL,
            "profiler_name": PROFILER_NAME if exact else None,
            "host_entry": HOST_ENTRY,
            "selected_host_entry": HOST_ENTRY,
            "acl_symbol": ACLNN_SYMBOL if exact else None,
            "selected_acl_symbol": ACLNN_SYMBOL if exact else None,
            "acl_symbol_status": ("affirmed_by_original_host_summary" if exact
                                  else "generic_deployment_host_variant_uncertain"),
            "source_class": contract["source_class"],
            "source_weight_offset": 1.0,
            "source_role": contract["role"],
            "source_residual": "absent",
            "source_block": contract["block"],
            "source_tier": contract["tier"],
            "source_model_type": contract["model_type"],
            "source_arch_family": contract["arch_family"],
            "source_image": contract["image_id"] if exact else None,
            "source_revision": R31_REVISION if exact else None,
            "source_package": (contract["exact"]["vllm_ascend_package"]
                               if exact else wrapper_package),
            "host_variant_uncertain": not exact,
            "call_id": getattr(binding, "call_id", None),
            "rows": rows,
            "tokens": tokens,
            "hidden": hidden,
            "input_dtype": dtype,
            "output_dtype": dtype,
            "rstd_dtype": RSTD_DTYPE,
            "rstd_shape": [rows, 1],
        },
        "complete": True,
        "partial": False,
        "unresolved": ([] if exact else [
            "generic_host_variant_uncertain",
            "generic_source_not_exact_for_deployment",
        ]),
        "non_claims": ([
            "no_universal_elf_digest_claim",
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "ordinary_cost_not_claimed_complete",
        ] if exact else [
            "no_exact_image_claim",
            "no_exact_acl_symbol_claim",
            "no_qwen_source_for_unrelated_deployment",
            "no_measured_kernel_elf",
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "ordinary_cost_not_claimed_complete",
        ]),
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
