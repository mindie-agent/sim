"""Prepare-only AddRmsNormBias scaffold over a live fused residual norm.

[Role] Bind the declared ``AddRmsNormBias`` CANN source to an ordinary fused
  residual+normalization operator only when the live lowering selects
  ``native_norm_kind=AddRmsNormBias``.  Width, epsilon, weight dtype/shape and
  the residual output come from that live operator; the optional beta comes
  from its ``norm_has_bias`` declaration, never from an original operand sample.
[Invariants]  No model config SHA, image tag, fixed hidden size, fixed epsilon,
  original operand file/index or fourth input selects the source.  If the live
  lowering selects AddRmsNormBias but does not declare bias presence, the
  selected scaffold remains bound with ``beta_state=unresolved``, no beta
  descriptor and ``complete=False``; no beta tensor is invented.  Other native
  norm kinds, Gemma/gamma shifts, non-residual norms and bias-free other APIs
  return ``None``.
[Timeouts & Locks]  Bounded memo of 32 prepared occurrences; no I/O or locks.
[Side-Effect Policy]  Pure metadata from the compiled lowering and the declared
  source/profile records.  No tensor allocation, device import, pointer/symbol
  fabrication, measured duration or performance selection.
"""
from __future__ import annotations

from typing import Any, Mapping

ENTRY = "torch.ops._C_ascend.npu_add_rms_norm_bias"
ACLNN_SYMBOL = "aclnnAddRmsNormBias"
KERNEL_NAME = "AddRmsNormBias"
PROFILER_NAME = "AddRmsNormBias"
SOURCE_LINES = (125, 218)
SCHEMA = "mindie-sim/add-rms-norm-bias-native-prepare/1"
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

__all__ = ["binding_spec", "dynamic_call"]

# Declared source/profile evidence.  These records do not choose width, dtype,
# epsilon, beta presence or a model; the live lowering does that.
INTERFACE = {
    "op_host_def_path": "csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_def.cpp",
    "op_host_def_sha256": (
        "ad15a83b71e465f1d36a240d4bcfc4cf92bd5ab82142bb826585c6ff12c51674"),
    "infershape_path": (
        "csrc/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_infershape.cpp"),
    "infershape_sha256": (
        "03a8dc5238c13a84476eaa9025496fe6e8bab80f540bf264b671abc65201b2db"),
    "torch_adapter_path": (
        "csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h"),
    "torch_adapter_sha256": (
        "2c7d886260aa8dc037dcdfc139c41901f53e99b3ffaac54d4da5e39e12d39c9a"),
    "torch_binding_path": "csrc/torch_binding.cpp",
    "torch_binding_sha256": (
        "e9406239357e88364fe39c3544f4896b041f61178cdbc1fe7542b4e4cc89c42b"),
    "op_kernel_entry_path": (
        "csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.cpp"),
    "op_kernel_entry_sha256": (
        "b1bd7dfac1143bfe2161237a1f26253d0d938862e369ff3aa1aa4b54ecae32ce"),
    "python_entry": ENTRY,
    "aclnn_call": (
        "EXEC_NPU_CMD(aclnnAddRmsNormBias, x1, x2, gamma, beta, epsilon, "
        "y, rstd, x)"),
    "declared_inputs": ["x1", "x2", "gamma", "beta(optional)"],
    "declared_outputs": ["y", "rstd", "x"],
    "epsilon_default": 1e-6,
    "source_state": "task_artifact",
    "revision_status": (
        "content hashes retained; local checkout is a later vllm-ascend "
        "revision; op_kernel hash equals declared family_ascend_norm.ADD_SOURCE"),
}

CALLER = {
    "path": "vllm_ascend/ops/layernorm.py",
    "sha256": (
        "50dedf714c6e485d47fea1cc4f0c8fc8058b43f1ad6e33b0df36dfaca0782991"),
    "symbol": "AscendRMSNorm.forward_oot",
    "call_lines": [73, 75],
    "call": (
        "torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, self.weight, "
        "self.bias, self.variance_epsilon)"),
    "returns": "(x, residual); the second output rstd is dropped by the caller",
    "revision_status": (
        "retained task artifact at a later vllm-ascend revision; the exact "
        "runtime caller file is not retained"),
}

PACKAGE = {
    "path": "vllm_ascend/ops/layernorm.py",
    "sha256": (
        "436570c24d1611d8959b2605deb75c6e811a9210b01b1b2a3a708ccb60a576d7"),
    "revision_status": (
        "deployed R64 wrapper blob unobserved; archived local v0.30/wave4 "
        "sources classify declared semantics with partial; exact R64 "
        "layernorm.py not retained; original count is verification only"),
    "selector": (
        "any 'norm.bias' in quant_description -> AscendRMSNorm.bias=zeros; "
        "residual is not None -> npu_add_rms_norm_bias"),
}

WEIGHTS = {
    "path": "quant_model_description.json",
    "sha256": (
        "7434850d69e93c9467b4963aedf78b19abf80b992e9b839a021f3e4be6332004"),
    "selector": (
        "any 'norm.bias' key creates the zero bias; no input/post_attention "
        "layernorm.bias exists, so beta is the constructor zero"),
}

_DYNAMIC_CACHE: dict[tuple, dict] = {}
_MAX_CACHE = 32


def _theta(compiled: Any) -> Any:
    for owner in (compiled, getattr(compiled, "operator", None),
                  getattr(compiled, "instance", None)):
        theta = getattr(owner, "theta", None)
        if theta is not None and hasattr(theta, "find"):
            return theta
    return None


def _find(compiled: Any, key: str, default: Any = None) -> Any:
    theta = _theta(compiled)
    if theta is not None:
        return theta.find(key, default)
    if isinstance(compiled, Mapping):
        return compiled.get(key, default)
    return getattr(compiled, key, default)


def _kind(compiled: Any) -> str:
    for owner in (compiled, getattr(compiled, "instance", None),
                  getattr(compiled, "fused", None)):
        value = str(getattr(owner, "kind", "") or "")
        if value:
            return value
    return ""


def _semantic_norm_scope(compiled: Any, operator: Any) -> str:
    """Classify a real selected normalization operator, not just its name.

    ``rmsnorm`` is the N0 member.  ``final_norm`` is accepted only for the
    actual O0 final residual producer: selected AddRmsNormBias class, final
    normalized role, declared residual return, weighted RMSNorm with zero
    weight offset.  Other final/O0 classes or absent-residual operators stay
    rejected; this is not a name-only bypass.
    """
    name = str(getattr(operator, "name", "") or "")
    family = str(getattr(operator, "family", "") or "")
    if name == "rmsnorm":
        return "n0"
    if name != "final_norm" or family != "elementwise":
        return ""
    if _kind(compiled) != "O0":
        return ""
    theta = getattr(operator, "theta", None)
    if theta is None or not hasattr(theta, "find"):
        return ""
    if str(theta.find("native_norm_kind", "") or "") != KERNEL_NAME:
        return ""
    if str(theta.find("gamma_role", "") or "") != "final":
        return ""
    if str(theta.find("fusion", "") or "") != "add_rmsnorm":
        return ""
    returns = theta.find("returns", ()) or ()
    if not (isinstance(returns, (list, tuple)) and "residual" in returns):
        return ""
    if int(theta.find("layernorm", 0) or 0) != 0:
        return ""
    if str(theta.find("norm_arch", "") or "") != "rmsnorm":
        return ""
    if float(theta.find("norm_weight_offset", 0) or 0) != 0.0:
        return ""
    if theta.find("norm_has_weight", None) is not True:
        return ""
    return "o0_final"


def _norm_operator(compiled: Any) -> Any:
    operator = getattr(compiled, "operator", None)
    if operator is not None and _semantic_norm_scope(compiled, operator):
        return operator
    candidates = []
    for owner in (getattr(compiled, "instance", None), compiled):
        candidates.extend(getattr(owner, "operators", ()) or ())
    candidates.extend(getattr(compiled, "members", ()) or ())
    candidates.extend(getattr(compiled, "semantic", ()) or ())
    for item in candidates:
        candidate = getattr(item, "operator", item)
        if _semantic_norm_scope(compiled, candidate):
            return candidate
    return None


def _sibling_names(compiled: Any) -> set[str]:
    names: set[str] = set()
    for item in getattr(compiled, "siblings", ()) or ():
        name = str(getattr(item, "name", "") or
                   getattr(getattr(item, "operator", None), "name", ""))
        if name:
            names.add(name)
    instance = getattr(compiled, "instance", None)
    if instance is not None:
        for item in getattr(instance, "operators", ()) or ():
            name = str(getattr(item, "name", "") or
                       getattr(getattr(item, "operator", None), "name", ""))
            if name:
                names.add(name)
    return names


def _fusion_contract(compiled: Any) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    contract = getattr(compiled, "contract", None)
    if contract is not None:
        name = str(getattr(contract, "name", "") or "")
        if name:
            return (name,
                    tuple(str(item) for item in
                          (getattr(contract, "members", ()) or ())),
                    tuple(str(item) for item in
                          (getattr(contract, "keeps", ()) or ())))
    name = str(getattr(compiled, "contract_name", "") or "")
    if name:
        return (name,
                tuple(str(item) for item in
                      (getattr(compiled, "members", ()) or ())),
                tuple(str(item) for item in
                      (getattr(compiled, "keeps", ()) or ())))
    for owner in (getattr(compiled, "instance", None), compiled):
        template = getattr(owner, "template", None)
        fusions = getattr(template, "fusions", None) if template is not None else None
        for item in fusions or ():
            members = tuple(str(name) for name in
                            (getattr(item, "members", ()) or ()))
            if "residual_add" in members and "rmsnorm" in members:
                return (str(getattr(item, "name", "") or ""), members,
                        tuple(str(name) for name in
                              (getattr(item, "keeps", ()) or ())))
    return "", (), ()


def _contract_name(compiled: Any, operator: Any, theta: Any) -> str:
    name, _members, _keeps = _fusion_contract(compiled)
    if name in {"add_rmsnorm", "add_rmsnorm_quant"}:
        return name
    family = str(getattr(operator, "family", "") or "")
    if family in {"add_rmsnorm", "add_rmsnorm_quant"}:
        return family
    quant = getattr(theta, "quant", {}) or {}
    if isinstance(quant, Mapping):
        act_bytes = quant.get("act_bytes")
    else:
        try:
            act_bytes = dict(quant).get("act_bytes")
        except (TypeError, ValueError):
            act_bytes = None
    try:
        act_bytes = float(act_bytes)
    except (TypeError, ValueError):
        act_bytes = None
    if act_bytes == 1.0:
        return "add_rmsnorm_quant"
    if act_bytes == 2.0:
        return "add_rmsnorm"
    return "add_rmsnorm"


def _live_hidden(operator: Any, theta: Any) -> int:
    shape = (getattr(operator, "local_shape", None)
             or getattr(getattr(operator, "operator", None), "local_shape", None))
    elements = theta.find("elements", None)
    if shape is not None and len(shape) == 2:
        hidden = int(shape[-1])
        if elements is not None and int(elements) != hidden:
            return 0
        return hidden
    if elements is not None:
        return int(elements)
    return 0


def _live_dtype(operator: Any, theta: Any) -> str:
    candidates = (
        theta.find("gamma_format", ""),
        theta.find("dtype", ""),
        getattr(operator, "dtype", ""),
        getattr(getattr(operator, "operator", None), "dtype", ""),
    )
    for value in candidates:
        text = str(value or "")
        if text in DTYPE_BYTES:
            return text
    return ""


def _live_residual(compiled: Any, operator: Any, theta: Any) -> str:
    if "residual_add" in _sibling_names(compiled):
        return "sibling_residual_add"
    returns = theta.find("returns", ()) or ()
    if isinstance(returns, (list, tuple)) and "residual" in returns:
        return "returns_residual"
    for owner in (operator, getattr(operator, "operator", None)):
        returns = getattr(owner, "returns", None)
        if isinstance(returns, (list, tuple)) and "residual" in returns:
            return "returns_residual"
    _name, members, keeps = _fusion_contract(compiled)
    if "residual" in keeps or "residual_add" in members:
        return "fusion_contract_residual"
    return ""


def _norm_facts(compiled: Any) -> dict | None:
    operator = _norm_operator(compiled)
    if operator is None:
        return None
    scope = _semantic_norm_scope(compiled, operator)
    if not scope:
        return None
    name = str(getattr(operator, "name", "") or "")
    family = str(getattr(operator, "family", "") or "")
    if family not in {"elementwise", "add_rmsnorm", "add_rmsnorm_quant"}:
        return None
    theta = getattr(operator, "theta", None)
    if theta is None or not hasattr(theta, "find"):
        return None
    native_kind = str(theta.find("native_norm_kind", "") or "")
    if native_kind != KERNEL_NAME:
        return None
    if int(theta.find("layernorm", 0) or 0) != 0:
        return None
    arch = str(theta.find("norm_arch", "") or "")
    if arch not in {"", "rmsnorm"}:
        return None
    if float(theta.find("norm_weight_offset", 0) or 0) != 0.0:
        return None
    if theta.find("norm_has_weight", None) is not True:
        return None
    hidden = _live_hidden(operator, theta)
    if hidden < 1:
        return None
    dtype = _live_dtype(operator, theta)
    if not dtype:
        return None
    gamma_bytes = theta.find("gamma_bytes", None)
    gamma_source = "declared"
    if gamma_bytes is None:
        gamma_source = "width_times_dtype"
        gamma_bytes = hidden * DTYPE_BYTES[dtype]
    else:
        try:
            gamma_bytes = float(gamma_bytes)
        except (TypeError, ValueError):
            return None
        if abs(gamma_bytes - hidden * DTYPE_BYTES[dtype]) > 1e-6:
            return None
    eps = theta.find("norm_eps", None)
    if eps is not None:
        if (not isinstance(eps, (int, float)) or isinstance(eps, bool)
                or eps <= 0):
            return None
        eps = float(eps)
    bias_raw = theta.find("norm_has_bias", None)
    if bias_raw is True:
        beta_state = "present"
    elif bias_raw is False:
        beta_state = "absent"
    else:
        beta_state = "unresolved"
    bias_trained_raw = theta.find("norm_bias_trained", None)
    residual = _live_residual(compiled, operator, theta)
    if not residual:
        return None
    if scope == "o0_final":
        if str(theta.find("fusion", "") or "") != "add_rmsnorm":
            return None
        contract = "add_rmsnorm"
    else:
        contract = _contract_name(compiled, operator, theta)
    # O0 rows are the normal compiled pre-selection activation rows, resolved
    # in dynamic_call from the source-builder operand or WorkloadGeometry.  The
    # static per-row rstd_bytes is not a row count and is not used here.
    preselection_rows = theta.find("preselection_rows", 0)
    if isinstance(preselection_rows, bool) or not isinstance(preselection_rows, int):
        preselection_rows = 0
    return {
        "scope": scope,
        "family": family,
        "name": name,
        "preselection_rows": max(0, preselection_rows),
        "hidden": hidden,
        "dtype": dtype,
        "eps": eps,
        "beta_state": beta_state,
        "bias_raw": bias_raw,
        "gamma_bytes": float(gamma_bytes),
        "gamma_source": gamma_source,
        "residual": residual,
        "contract": contract,
        "kind": _kind(compiled),
        "role": str(theta.find("gamma_role", "") or ""),
        "bias_value": str(theta.find("norm_bias_value", "") or ""),
        "bias_origin": str(theta.find("norm_bias_origin", "") or ""),
        "bias_trained": (bias_trained_raw
                         if isinstance(bias_trained_raw, bool) else None),
        "package_sha256": str(theta.find("norm_source_package_sha256", "") or ""),
        "weights_sha256": str(theta.find("norm_source_weights_sha256", "") or ""),
        "native_norm_source": str(theta.find("native_norm_source", "") or ""),
        "sequence_rows": str(theta.find("sequence_rows", "") or ""),
        "sequence_ranks": int(theta.find("sequence_ranks", 1) or 1),
    }


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _optional_beta_descriptor(state: str) -> dict:
    return {
        "shape": None, "dtype": None, "layout": None,
        "role": "input_optional", "pointer": None, "allocation": "external",
        "present": False if state == "absent" else None,
        "resolution": ("declared_absent" if state == "absent"
                       else "not_declared_by_live_lowering"),
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the declared AddRmsNormBias scaffold for the live selected source.

    The live lowering must explicitly select ``native_norm_kind=AddRmsNormBias``
    and supply the fused residual/weight/width fields.  If it selects that
    source but does not declare beta presence, the scaffold is returned with
    ``beta_state=unresolved`` and no beta tensor.  No model/image/original-operand
    predicate is used.
    """
    facts = _norm_facts(compiled)
    if facts is None:
        return None
    try:
        from mindie_sim.prediction.family_ascend_norm import ADD_SOURCE
    except ImportError:  # pragma: no cover - normal product import
        return None
    if not ADD_SOURCE.get("path") or not ADD_SOURCE.get("sha256"):
        return None
    from . import native_calls

    source = native_calls._source(
        str(ADD_SOURCE.get("repository") or "vllm-ascend"),
        str(ADD_SOURCE["path"]), sha256=str(ADD_SOURCE["sha256"]),
        symbol=KERNEL_NAME, revision=str(ADD_SOURCE.get("base_revision") or ""),
        state="task_artifact")
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", path=str(ADD_SOURCE["path"]),
        sha256=str(ADD_SOURCE["sha256"]), evidence="source_declared",
        acl_api=ACLNN_SYMBOL, profiler_name=PROFILER_NAME,
        execution_core="aiv",
        kernel_source_status="declared_op_kernel_source_retained",
        host_path=INTERFACE["op_host_def_path"],
        host_sha256=INTERFACE["op_host_def_sha256"],
        torch_adapter_path=INTERFACE["torch_adapter_path"],
        torch_adapter_sha256=INTERFACE["torch_adapter_sha256"],
        source_lines=list(SOURCE_LINES))
    beta_state = facts["beta_state"]
    beta_present = beta_state == "present"
    input_contract = (["x1", "x2", "gamma", "beta"] if beta_present else
                      ["x1", "x2", "gamma"] if beta_state == "absent" else
                      ["x1", "x2", "gamma", "beta?"])
    scope = str(facts["scope"])
    final = scope == "o0_final"
    contract = str(facts["contract"])
    kind = facts["kind"] or "N0"
    if final:
        owner = f"{kind}:{facts['family']}:{facts['name']}"
        members: tuple[str, ...] = ()
    else:
        owner = f"{kind}:fused:{contract}"
        members = ("residual_add", "rmsnorm")
    attrs = {
        "hidden": int(facts["hidden"]),
        "norm_dtype": facts["dtype"],
        "rstd_dtype": "fp32",
        "output_dtype": facts["dtype"],
        "residual_dtype": facts["dtype"],
        "gamma_dtype": facts["dtype"],
        "input_contract": input_contract,
        "output_contract": ["y", "rstd", "x"],
        "beta_state": beta_state,
        "residual_output": True,
        "residual_source": facts["residual"],
        "native_norm_kind": KERNEL_NAME,
        "native_norm_scope": scope,
        "binding_owner": owner,
        "source_selection": "live_lowering_native_norm_kind",
        "call_name": ENTRY,
        "kernel_name": KERNEL_NAME,
        "acl_api": ACLNN_SYMBOL,
        "profiler_name": PROFILER_NAME,
        "source_revision": str(ADD_SOURCE.get("base_revision") or ""),
        "source_scope": str(ADD_SOURCE.get("scope") or ""),
        "caller_path": CALLER["path"],
        "caller_symbol": CALLER["symbol"],
        "caller_sha256": CALLER["sha256"],
        "caller_revision_status": CALLER["revision_status"],
        "fusion_contract": contract,
        "fusion_members": [] if final else ["residual_add", "rmsnorm"],
        "physical_outputs": ["y", "rstd", "x"],
        "weight_bytes": float(facts["gamma_bytes"]),
        "weight_bytes_source": facts["gamma_source"],
        "weight_role": str(facts["role"]),
    }
    if beta_present:
        attrs["beta_dtype"] = facts["dtype"]
        attrs["beta_bytes"] = float(facts["hidden"] * DTYPE_BYTES[facts["dtype"]])
        if facts["bias_value"]:
            attrs["beta_value"] = facts["bias_value"]
        if facts["bias_trained"] is not None:
            attrs["beta_trained"] = bool(facts["bias_trained"])
        if facts["bias_origin"]:
            attrs["beta_origin"] = facts["bias_origin"]
        if facts["package_sha256"]:
            attrs["source_package_sha256"] = facts["package_sha256"]
        if facts["weights_sha256"]:
            attrs["source_weights_sha256"] = facts["weights_sha256"]
    if facts["native_norm_source"]:
        attrs["native_norm_source"] = facts["native_norm_source"]
    if facts["preselection_rows"] > 0:
        attrs["preselection_rows"] = int(facts["preselection_rows"])
    if facts["sequence_rows"]:
        attrs["sequence_rows"] = facts["sequence_rows"]
    attrs["sequence_ranks"] = int(facts["sequence_ranks"])
    if facts["eps"] is not None:
        attrs["norm_eps"] = float(facts["eps"])
        attrs["epsilon_source"] = "live_lowering_norm_eps"
    if final:
        attrs["kept_contract_outputs"] = ["residual", "rstd", "norm_output"]
    elif contract == "add_rmsnorm_quant":
        attrs["kept_contract_outputs"] = ["residual", "rstd", "scale",
                                           "norm_output"]
        attrs["scale_output_owner"] = "separate_quantizer_not_this_call"
    typed = {
        "dtype": facts["dtype"],
        "layout": "ND",
        "phase": str(phase or ""),
        "shapes": {
            "x1": [int(facts["hidden"])],
            "x2": [int(facts["hidden"])],
            "gamma": [int(facts["hidden"])],
            "y": [int(facts["hidden"])],
            "rstd": [1],
            "x": [int(facts["hidden"])],
        },
        "attrs": attrs,
        "rank_width": 2,
    }
    if beta_present:
        typed["shapes"]["beta"] = [int(facts["hidden"])]
    unresolved = [
        "source_revision_not_pinned_to_runtime_image",
        "kernel_binary_identity_not_established",
        "python_caller_same_image_revision_not_retained",
        "r64_registered_package_revision_unresolved",
    ]
    if beta_state == "unresolved":
        unresolved.append("norm_bias_declaration_unresolved")
    if beta_state == "present":
        if not facts["bias_value"]:
            unresolved.append("beta_value_unresolved")
        if facts["bias_trained"] is None:
            unresolved.append("beta_train_state_unresolved")
        if not facts["bias_origin"]:
            unresolved.append("beta_origin_unresolved")
    if (facts["bias_raw"] is not None
            and not isinstance(facts["bias_raw"], bool)):
        unresolved.append("norm_bias_declaration_not_boolean")
    if facts["eps"] is None:
        unresolved.append("epsilon_from_caller_unresolved")
    if facts["gamma_source"] != "declared":
        unresolved.append("gamma_bytes_not_declared")
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_SYMBOL,
                          version=str(ADD_SOURCE.get("base_revision") or "")),
        source, typed, (kernel,), members, owner,
        execution_core="aiv", unresolved=unresolved)


def _preselection_rows(geometry: Any, attrs: Mapping) -> tuple[int, str]:
    """Resolve the O0 final norm's real pre-selection local activation rows.

    Prefer the source-builder operand from the common normalized producer, then
    the compiled WorkloadGeometry pre-selection view.  ``selected_tokens``,
    ``selected_row_count``, request count, LM-head commit rows, static
    per-row ``rstd_bytes`` and a fixed 1024 are deliberately not used.
    """
    final_rows = attrs.get("final_rows")
    if (isinstance(final_rows, int) and not isinstance(final_rows, bool)
            and final_rows > 0):
        return final_rows, str(attrs.get("final_rows_source") or
                               "workload_preselection_local_activation_rows")
    candidates = (
        ("source_builder_preselection_rows", attrs.get("preselection_rows")),
        ("geometry_preselection_tokens",
         getattr(geometry, "preselection_tokens", None)),
        ("geometry_preselection_rows",
         getattr(geometry, "preselection_rows", None)),
        ("geometry_effective_tokens",
         getattr(geometry, "effective_tokens", None)),
    )
    for source, value in candidates:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0:
            return value, source
    return 0, "preselection_rows_unresolved"


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the live-selected AddRmsNormBias.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  Beta is a real tensor only when the live
    declaration says present; absent/unresolved beta never becomes a fabricated
    fourth input.
    """
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != ENTRY or api.get("aclnn_symbol") != ACLNN_SYMBOL:
        return None
    source = dict(getattr(binding, "source", {}) or {})
    try:
        from mindie_sim.prediction.family_ascend_norm import ADD_SOURCE
    except ImportError:  # pragma: no cover - normal product import
        return None
    if source.get("sha256") != ADD_SOURCE.get("sha256"):
        return None
    attrs = dict((getattr(binding, "typed", {}) or {}).get("attrs") or {})
    beta_state = str(attrs.get("beta_state") or "")
    if (attrs.get("native_norm_kind") != KERNEL_NAME
            or int(attrs.get("hidden", 0) or 0) < 1
            or str(attrs.get("norm_dtype") or "") not in DTYPE_BYTES
            or beta_state not in {"present", "absent", "unresolved"}
            or attrs.get("residual_output") is not True):
        return None
    scope = str(attrs.get("native_norm_scope") or "n0")
    if scope not in {"n0", "o0_final"}:
        return None
    if scope == "o0_final":
        rows, rows_source = _preselection_rows(geometry, attrs)
    else:
        rows = int(getattr(geometry, "effective_tokens", 0) or 0)
        rows_source = "subgraph_effective_tokens_local_rank"
    if rows < 1:
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    hidden = int(attrs["hidden"])
    dtype = str(attrs["norm_dtype"])
    cache_key = (str(getattr(binding, "call_id", "")), rows, hidden, dtype,
                 beta_state, phase, scope)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    eps = attrs.get("norm_eps")
    beta_value = attrs.get("beta_value")
    beta_trained = attrs.get("beta_trained")
    beta_origin = attrs.get("beta_origin")
    package_sha256 = attrs.get("source_package_sha256") or PACKAGE["sha256"]
    weights_sha256 = attrs.get("source_weights_sha256") or WEIGHTS["sha256"]
    native_norm_source = attrs.get("native_norm_source") or PACKAGE["selector"]
    sequence_rows = str(attrs.get("sequence_rows") or "")
    sequence_ranks = int(attrs.get("sequence_ranks") or 1)
    actual_shapes = {
        "x1": [rows, hidden],
        "x2": [rows, hidden],
        "gamma": [hidden],
        "y": [rows, hidden],
        "rstd": [rows, 1],
        "x": [rows, hidden],
    }
    if beta_state == "present":
        actual_shapes["beta"] = [hidden]
    actual_attrs = {
        **attrs,
        "rows": rows,
        "tokens": rows,
        "phase": phase,
        "gamma_bytes": hidden * DTYPE_BYTES[dtype],
        "rstd_bytes": rows * 4,
        "residual_bytes": rows * hidden * DTYPE_BYTES[dtype],
        "norm_output_bytes": rows * hidden * DTYPE_BYTES[dtype],
    }
    if beta_state == "present":
        actual_attrs["beta_bytes"] = hidden * DTYPE_BYTES[dtype]
    role = str(_find(compiled, "gamma_role", "") or "")
    if role:
        actual_attrs["norm_role"] = role
    actual = {"dtype": dtype, "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x1": _descriptor([rows, hidden], dtype, "ND", "input"),
        "x2": _descriptor([rows, hidden], dtype, "ND", "input"),
        "gamma": _descriptor([hidden], dtype, "ND", "weight"),
        "y": _descriptor([rows, hidden], dtype, "ND", "output"),
        "rstd": _descriptor([rows, 1], "fp32", "ND", "output"),
        "x": _descriptor([rows, hidden], dtype, "ND", "output"),
    }
    if beta_state == "present":
        descriptors["beta"] = _descriptor([hidden], dtype, "ND", "weight")
        descriptors["beta"].update({
            "value": beta_value or None,
            "trained": beta_trained,
            "origin": beta_origin or None,
            "source_package_sha256": package_sha256,
            "source_weights_sha256": weights_sha256,
        })
    else:
        descriptors["beta"] = _optional_beta_descriptor(beta_state)
    arguments = {
        "x1": "x1", "x2": "x2", "gamma": "gamma",
        "beta": "beta" if beta_state == "present" else None,
        "epsilon": eps, "y": "y", "rstd": "rstd", "x": "x",
    }
    kernel = (dict(getattr(binding, "kernels", ())[0])
              if getattr(binding, "kernels", ()) else
              {"name": KERNEL_NAME, "role": "vector",
               "evidence": "source_declared"})
    owner = str(attrs.get("binding_owner") or
                f"{_kind(compiled) or 'N0'}:fused:"
                f"{attrs.get('fusion_contract') or 'add_rmsnorm'}")
    row_basis = rows_source
    unresolved = list(getattr(binding, "unresolved", ()) or ())
    if eps is None:
        unresolved.append("epsilon_from_caller_unresolved")
    prepare = {
        "schema": SCHEMA,
        "call": ENTRY,
        "call_id": getattr(binding, "call_id", None),
        "signature_hash": (getattr(binding, "typed", {}) or {}).get(
            "signature_hash"),
        "source": source,
        "source_lines": list(SOURCE_LINES),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": list(arguments),
        "optional_slots": {
            "beta": ("present" if beta_state == "present" else
                     "declared_absent" if beta_state == "absent" else
                     "unresolved_not_declared"),
            "epsilon": ("live_lowering_norm_eps" if eps is not None
                        else "caller_required"),
        },
        "argument_order_status": (
            "torch schema order; beta presence is the live norm_has_bias "
            "declaration, and unresolved beta is not a value claim; ACLNN "
            "workspace/executor slots are hidden by EXEC_NPU_CMD"),
        "shapes": actual_shapes,
        "validated_shapes": validated.get("shapes"),
        "attrs": actual_attrs,
        "descriptors": descriptors,
        "arrays": descriptors,
        "outputs": {
            "y": descriptors["y"],
            "rstd": descriptors["rstd"],
            "x": descriptors["x"],
        },
        "output_count": 3,
        "beta": {
            "state": beta_state,
            "tensor": descriptors.get("beta"),
            "value": beta_value or None,
            "trained": beta_trained,
            "origin": beta_origin or None,
            "source_package_sha256": package_sha256,
            "source_weights_sha256": weights_sha256,
        },
        "source_profile": {
            "package": dict(PACKAGE),
            "weights": dict(WEIGHTS),
        },
        "aliases": {},
        "alias_status": (
            "separate output allocation from the retained torch adapter; "
            "no input/output alias is claimed"),
        "interface": dict(INTERFACE),
        "caller": dict(CALLER),
        "fusion": {
            "contract": attrs.get("fusion_contract"),
            "members": [] if scope == "o0_final" else ["residual_add", "rmsnorm"],
            "keeps": (["residual", "rstd", "scale", "norm_output"]
                      if attrs.get("fusion_contract") == "add_rmsnorm_quant"
                      else ["residual", "rstd", "norm_output"]),
            "physical_outputs": ["y", "rstd", "x"],
            "scale_output_owner": (
                "separate_quantizer_not_this_call"
                if attrs.get("fusion_contract") == "add_rmsnorm_quant"
                else None),
            "note": (
                "standalone O0 final residual+norm producer; no fusion parent"
                if scope == "o0_final" else
                "the fused cost may also keep a quantizer scale; this "
                "AddRmsNormBias call does not produce or own that scale"),
        },
        "pricing": {
            "repriced": False,
            "charged_owner": owner,
            "note": (
                "standalone O0 final_norm cost is unchanged; this record "
                "neither prices nor selects work"
                if scope == "o0_final" else
                "existing fused residual+norm cost is unchanged; this record "
                "neither prices nor selects work"),
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_SYMBOL,
            "call_id": getattr(binding, "call_id", None),
            "source_revision": str(ADD_SOURCE.get("base_revision") or ""),
            "source_sha256": source.get("sha256"),
            "hidden": hidden,
            "norm_dtype": dtype,
            "gamma_dtype": dtype,
            "rstd_dtype": "fp32",
            "residual_dtype": dtype,
            "epsilon": eps,
            "beta_state": beta_state,
            "beta_value": beta_value or None,
            "beta_trained": beta_trained,
            "beta_origin": beta_origin or None,
            "native_norm_source": native_norm_source,
            "source_package_sha256": package_sha256,
            "source_weights_sha256": weights_sha256,
            "native_norm_scope": scope,
            "row_basis": row_basis,
            "rows_source": rows_source,
            "final_rows_source": attrs.get("final_rows_source"),
            "selected_lm_head_rows": attrs.get("selected_lm_head_rows"),
            "rank_local_shard_rows": attrs.get("rank_local_shard_rows"),
            "fusion_members": [] if scope == "o0_final" else ["residual_add", "rmsnorm"],
            "sequence_rows": sequence_rows,
            "sequence_ranks": sequence_ranks,
            "input_contract": attrs.get("input_contract"),
            "output_contract": ["y", "rstd", "x"],
            "residual_source": attrs.get("residual_source"),
            "fusion_contract": attrs.get("fusion_contract"),
            "rows": rows,
            "phase": phase,
            "input_count": 4 if beta_state == "present" else 3,
            "output_count": 3,
            "status": "partial_prepare_not_executed",
        },
        "binary_selection": {
            "status": "partial",
            "reason": (
                "declared source/interface selected by live lowering; "
                "compiled runtime binary identity is not established"),
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_SYMBOL,
            "source_sha256": source.get("sha256"),
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "acl_prepare_only",
            "source_revision_not_pinned_to_runtime_image",
            "kernel_binary_identity_not_established",
            "python_caller_same_image_revision_not_retained",
            "r64_registered_package_revision_unresolved",
        ],
        "unresolved": list(dict.fromkeys(
            unresolved + ["execution_rank_not_selected"])),
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_binary_identity",
            "no_complete_python_caller",
            "no_sample_derived_beta_or_width",
            "no_operand_shape_inferred_beta",
        ],
    }
    if beta_state == "unresolved":
        prepare["partial_reasons"].append("norm_bias_declaration_unresolved")
    if beta_state == "present":
        if not beta_value:
            prepare["partial_reasons"].append("beta_value_unresolved")
        if beta_trained is None:
            prepare["partial_reasons"].append("beta_train_state_unresolved")
        if not beta_origin:
            prepare["partial_reasons"].append("beta_origin_unresolved")
    if eps is None:
        prepare["partial_reasons"].append("epsilon_from_caller_unresolved")
    if attrs.get("weight_bytes_source") != "declared":
        prepare["partial_reasons"].append("gamma_bytes_not_declared")
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": dtype,
            "layout": "ND",
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= _MAX_CACHE:
        _DYNAMIC_CACHE.pop(next(iter(_DYNAMIC_CACHE)))
    _DYNAMIC_CACHE[cache_key] = result
    return result
