"""Native counterparts for already-declared compound semantic primitives.

[Role] Expose known implementation boundaries without manufacturing subgraphs.
[Invariants] A counterpart is neither kernel selection nor tensor/occurrence
  evidence. Optional epilogues with unmodeled work are not silently included.
  Multiple compatible native boundaries are alternatives; only an explicit
  source/region selection may name one as the plan. A candidate never becomes
  an execution claim and no observed duration enters this record.
[Timeouts & Locks] Constant per-operator metadata work; no I/O or locks.
[Side-Effect Policy] Pure metadata; no profiler, capture or device dependency.
"""
from mindie_sim.structure_contracts import predicted_contract, typed_values
from mindie_sim.structure_native_contracts import (
    MIXED_QUANT_NATIVE_KIND, shared_kv_native_name,
)


_CONTRACT_NATIVE = {
    "native_tensor_abi", "gated_norm_active_rows", "attention_active_qkv",
    "sparse_attention_active", "kv_cache_store_active_elements",
    "dynamic_quant_active", "add_rmsnorm_dynamic_quant_active", "rmsnorm_mxquant_active",
    "layernorm_active", "transpose_batchmatmul_active",
    "attention_paged_mla_qk_rope",
}
_NATIVE_CONTRACT = {
    "ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO", "CausalConv1d",
    "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
}
_SEMANTIC = {
    "rmsnorm": ("RmsNorm",),
    "add_rmsnorm": ("AddRmsNorm",),
    "swiglu": ("SwiGlu", "Swiglu", "SwiGLU"),
    "gdn_gating": ("fused_gdn_gating_kernel",),
    "recurrent_vector": ("RecurrentGatedDeltaRule",),
    "layernorm": ("LayerNormV3",),
}


def native_implementation(op, contract):
    meta = {**op["metadata"].get("semantic_metadata", {}), **op["metadata"]}
    explicit = meta.get("native_implementation")
    if explicit:
        return explicit
    if op.get("source_boundary") == "call" or op["family"] == "barrier":
        return None
    # A declared source kernel boundary names its own native entry point. The
    # contract only projects tensors, so reading the name from the contract
    # would silently relabel the mixed-quant attention kernel as FIA.
    if op.get("source_boundary") == "kernel" and isinstance(meta.get("native_kind"), str):
        return {
            "native_kinds": (meta["native_kind"],),
            "selection": "declared_source_kernel_boundary",
            "basis": "implementation_binding",
            "conditions": {},
        }
    # A cache epilog's explicit native identity is a correspondence candidate,
    # not a selected software plan.  Without this, a fused epilog exports an
    # empty implementation and an exact observed kernel finds no candidate.
    native_kinds = meta.get("native_kinds")
    if isinstance(native_kinds, str):
        native_kinds = (native_kinds,)
    if not native_kinds and isinstance(meta.get("native_kind"), str):
        native_kinds = (meta["native_kind"],)
    if native_kinds:
        kinds = tuple(str(kind) for kind in native_kinds if str(kind).strip())
        if kinds:
            return {
                "native_kinds": kinds,
                "selection": "compatible_semantic_boundary; runtime_selection_not_implied",
                "basis": "explicit_native_identity",
                "conditions": {"native_identity_not_selected": True},
            }
    kinds = ()
    if contract:
        if contract[0] in _CONTRACT_NATIVE:
            kinds = (contract[1],)
        elif contract[0] in _NATIVE_CONTRACT:
            kinds = (contract[0],)
    if not kinds:
        kinds = _SEMANTIC.get(op["family"], ())
    conditions = {}
    if not kinds and meta.get("program_kind") == "matmul":
        if op["family"] == "grouped_gemm":
            kinds = ("GroupedMatmul",)
            conditions = {"optional_bias": bool(meta.get("bias")),
                          "epilogue_scope": "declared_matrix_and_dequantization_only"}
        elif meta.get("activation_format") in {"int8", "fp8_e4m3", "fp8_e5m2"}:
            kinds = ("QuantBatchMatmulV3", "QuantBatchMatmulV4")
            conditions = {"native_version_not_selected": True,
                          "optional_bias": bool(meta.get("bias")),
                          "operand_formats": {
                              key: meta[key] for key in
                              ("activation_format", "weight_format", "output_format")}}
    if not kinds and op["family"] == "attention":
        view = meta.get("attention_tensor_view", {})
        kind = view.get("kind")
        kinds = (("SparseFlashAttention",) if kind == "indexed_latent" else
                 (MIXED_QUANT_NATIVE_KIND,) if kind == "mixed_quant_shared_kv" else
                 (shared_kv_native_name(view),) if kind == "shared_kv" else
                 ("FusedInferAttentionScore",))
        conditions = {"native_layout_dispatch_and_auxiliary_outputs_not_selected": True}
    if not kinds and op["recipe"] == "qk_l2norm":
        kinds = ("l2norm_fwd_kernel2",)
        conditions = {"scope": "declared_qk_normalization; native_row_layout_not_selected"}
    if not kinds and op["family"] == "rotary":
        kinds = ("InplacePartialRotaryMul", "_triton_rope_siso")
        conditions = {"native_layout_and_rotary_partition_not_selected": True}
    if not kinds and op["family"] == "routing" and op["recipe"] == "route_pack":
        kinds = ("MoeInitRoutingV2", "MoeInitRoutingV3", "MoeTokenPermute")
        conditions = {"scope": "declared_permutation_and_token_pack",
                      "native_version_and_optional_quantization_not_selected": True}
    if not kinds and op["family"] == "routing" and op["recipe"] == "route_unpack":
        kinds = ("MoeTokenUnpermute", "MoeFinalizeRoutingV2")
        conditions = {"scope": "declared_inverse_permutation_and_weighted_reduction",
                      "optional_residual_not_included": True}
    if not kinds:
        return None
    return {
        "native_kinds": kinds,
        "selection": "compatible_semantic_boundary; runtime_selection_not_implied",
        "basis": "declared_semantics_and_tensor_boundary",
        "conditions": conditions,
    }


_STATUS_COMPATIBLE = "compatible_not_selected"
_STATUS_DECLARED = "declared_native_implementation"
_UNSET = object()
_NOT_SELECTED = ("not_selected", "unselected", "compatible", "unresolved",
                 "undefined", "none", "candidate", "alternative")


def _implementation_kinds(implementation):
    """Return the declared native names without treating them as selected."""
    if isinstance(implementation, str):
        return [implementation]
    if not isinstance(implementation, dict):
        return []
    kinds = implementation.get("native_kinds")
    if kinds is None and implementation.get("native_kind") is not None:
        kinds = (implementation["native_kind"],)
    if kinds is None:
        return []
    if isinstance(kinds, str):
        kinds = (kinds,)
    return [kind for kind in kinds if isinstance(kind, str) and kind]


def _first_native(implementation):
    kinds = _implementation_kinds(implementation)
    return kinds[0] if len(kinds) == 1 else None


def _selection_fact(implementation, *, source):
    """Interpret a declared selection without inferring it from a name count."""
    if not isinstance(implementation, dict):
        return None
    if implementation.get("selected") is False:
        return None
    selection = implementation.get("selection")
    if selection is None:
        return None
    if isinstance(selection, dict):
        if selection.get("selected") is False:
            return None
        basis = str(selection.get("basis") or selection.get("kind") or "")
        if (selection.get("selected") is True
                or basis in {"selected_software_plan", "declared_selection"}
                or selection.get("native")):
            native = (_first_native(implementation)
                      or (selection.get("native") if isinstance(selection.get("native"), str)
                          else None)
                      or (selection.get("implementation")
                          if isinstance(selection.get("implementation"), str) else None))
            return {"native": native, "selected_by": source,
                    "evidence": basis or "declared_selection",
                    "plan": selection}
        return None
    if isinstance(selection, str):
        text = selection.strip()
        if not text or any(token in text.lower() for token in _NOT_SELECTED):
            return None
        return {"native": _first_native(implementation) or text,
                "selected_by": source,
                "evidence": text,
                "plan": {"selection": text}}
    if selection is True or implementation.get("selected") is True:
        return {"native": _first_native(implementation),
                "selected_by": source, "evidence": "declared_selection",
                "plan": None}
    return None


def _fusion_boundary(meta):
    boundary = meta.get("fusion_boundary")
    return boundary if isinstance(boundary, dict) else None


def _fusion_selection(meta):
    boundary = _fusion_boundary(meta)
    if not boundary:
        return None
    native = _first_native(boundary.get("native"))
    plan = {key: boundary.get(key) for key in
            ("implementation", "program_profile", "submitted_kernels",
             "semantic_phase_count", "timing_basis") if boundary.get(key) is not None}
    return {"native": native,
            "selected_by": "operator.metadata.fusion_boundary",
            "evidence": "selected_software_plan", "plan": plan}


def _selected_native(operator):
    """A declared selection is evidence, never an observed execution."""
    meta = operator.get("metadata", {})
    native_call = meta.get("native_call")
    if isinstance(native_call, dict):
        api = native_call.get("api") if isinstance(native_call.get("api"), dict) else {}
        native = (api.get("entry") or api.get("aclnn_symbol")
                  or native_call.get("entry") or native_call.get("aclnn_symbol"))
        if native:
            return {"native": str(native), "selected_by": "operator.metadata.native_call",
                    "evidence": "declared_native_call_boundary", "plan": None}
    if meta.get("aclnn_symbol"):
        return {"native": str(meta["aclnn_symbol"]),
                "selected_by": "operator.metadata.aclnn_symbol",
                "evidence": "declared_native_call_boundary", "plan": None}
    binding = meta.get("implementation_binding")
    if isinstance(binding, dict) and binding.get("implementation"):
        return {"native": str(binding["implementation"]),
                "selected_by": "operator.metadata.implementation_binding",
                "evidence": "declared_selected_implementation", "plan": None}
    explicit = meta.get("native_implementation")
    if isinstance(explicit, dict):
        fact = _selection_fact(explicit, source="operator.metadata.native_implementation")
        if fact is not None:
            return fact
    if meta.get("native_kind") or meta.get("native_kinds"):
        # A name is a correspondence candidate; only an explicit selection
        # fact (above) may promote it to the selected software plan.
        pass
    return _fusion_selection(meta)


def _mark_selected(candidates, selected):
    if selected is None:
        return candidates
    for candidate in candidates:
        if selected.get("native") and candidate["native"] == selected["native"]:
            candidate["selected"] = True
            candidate["status"] = "selected_declared"
    return candidates


def _candidate(operator, kind, implementation, *, explicit, index, region_id=None,
               owner_id=None):
    op_id = operator.get("op_id") if operator else None
    owner = owner_id or region_id or op_id or "operator"
    conditions = implementation.get("conditions") if isinstance(implementation, dict) else None
    return {
        "candidate_id": f"{owner}/native:{kind or index}",
        "native": kind,
        "status": _STATUS_DECLARED if explicit else _STATUS_COMPATIBLE,
        "selected": False,
        "observed": False,
        "basis": (implementation.get("basis")
                  if isinstance(implementation, dict) else None)
                 or "declared_semantics_and_tensor_boundary",
        "conditions": conditions or {},
        "gaps": [] if kind else [{"code": "native_name_unresolved",
                                  "detail": {"owner": owner}}],
    }


def _execution(operator, candidates, selected):
    if selected is not None:
        match = next((row for row in candidates
                      if selected.get("native") and row["native"] == selected["native"]), None)
        native = selected.get("native")
        return {
            "kind": "source_selected_native" if native else "source_selected_plan",
            "selection_state": "selected_declared",
            "selected": {**selected, "candidate_id": match["candidate_id"] if match else None},
            "observed": False,
            "reason": ("declared_software_selection; not a captured occurrence"
                       if native else
                       "a software plan is selected but no unique native API is named"),
        }
    if operator.get("source_boundary") == "call":
        return {
            "kind": "expanded_source_call",
            "selection_state": "expanded",
            "selected": None,
            "observed": False,
            "reason": "children carry the native boundaries; the call is not a kernel",
        }
    if not candidates:
        return {
            "kind": "unresolved_native_boundary",
            "selection_state": "unresolved",
            "selected": None,
            "observed": False,
            "reason": "no declared native candidate; see representation gaps",
        }
    state = ("single_compatible_unselected" if len(candidates) == 1
             else "multiple_compatible_unselected")
    return {
        "kind": "candidate_set_only",
        "selection_state": state,
        "selected": None,
        "observed": False,
        "reason": "declared compatible alternatives are not a runtime selection",
    }


def native_candidates(operator, contract=_UNSET, *, owner_id=None):
    """Return ``(candidates, execution, gaps)`` for one semantic operator.

    Every compatible native name becomes its own candidate.  None is selected
    unless an explicit selection fact exists.  ``selection=None`` and a lone
    native name remain unselected correspondence facts.
    """
    if contract is _UNSET:
        contract = predicted_contract(operator)
    implementation = native_implementation(operator, contract)
    meta = operator.get("metadata", {})
    explicit = bool(meta.get("native_implementation"))
    kinds = _implementation_kinds(implementation)
    fusion = _fusion_boundary(meta)
    if not kinds and fusion is not None:
        implementation = fusion.get("native") or {}
        explicit = False
        kinds = _implementation_kinds(implementation)
    candidates = [_candidate(operator, kind, implementation, explicit=explicit,
                             index=index, owner_id=owner_id)
                  for index, kind in enumerate(kinds)]
    selected = _selected_native(operator)
    _mark_selected(candidates, selected)
    execution = _execution(operator, candidates, selected)
    gaps = []
    if not kinds:
        gaps.append({
            "code": "native_candidate_unmodeled",
            "detail": {"family": operator.get("family"),
                       "recipe": operator.get("recipe"),
                       "native_tensor_abi": "declared" if contract is not None else "unavailable"},
        })
    if (selected is not None and selected.get("native")
            and not any(candidate["selected"] for candidate in candidates)):
        gaps.append({"code": "selected_native_not_in_candidates",
                     "detail": {"selected": selected["native"],
                                "candidates": [candidate["native"] for candidate in candidates]}})
    return candidates, execution, gaps


def operator_structure(operator, contract=_UNSET, *, semantic_refs=None,
                       owner_id=None, context=None):
    """Recognize one operator's values and native candidates without costing it.

    This is the independent structure entry: no Case, hardware, calibration or
    Engine is consulted.  A caller may therefore identify structure and price a
    plan separately, and a shape absent from every fixture is still recognized.
    """
    if contract is _UNSET:
        contract = predicted_contract(operator)
    values = typed_values(operator, contract, context=context)
    candidates, execution, gaps = native_candidates(operator, contract,
                                                    owner_id=owner_id)
    if values.get("selection") and not execution.get("plan_selection"):
        execution["plan_selection"] = values["selection"]
    unique = []
    for gap in (*values["gaps"], *gaps):
        if gap not in unique:
            unique.append(gap)
    return {
        "contract": contract,
        "values": values,
        "semantic_refs": list(semantic_refs) if semantic_refs else None,
        "native_candidates": candidates,
        "native_execution": execution,
        "gaps": unique,
    }


def region_structure(implementation, *, region_id, semantic_refs=None):
    """Describe one declared fused/cross-layer region and its native boundary.

    Several semantic members may be covered by one declared native region.  A
    lone native name is still only a correspondence; selection requires an
    explicit ``selection`` fact. The member operators remain separate records.
    """
    kinds = _implementation_kinds(implementation)
    explicit = bool(kinds)
    candidates = [_candidate(None, kind, implementation or {}, explicit=explicit,
                             index=index, region_id=region_id)
                  for index, kind in enumerate(kinds)]
    selected = _selection_fact(implementation, source="subgraph.declared_implementation")
    _mark_selected(candidates, selected)
    if selected is not None:
        execution = {
            "kind": "declared_native_region" if selected.get("native")
                    else "declared_region_plan",
            "selection_state": "selected_declared",
            "selected": {**selected,
                         "candidate_id": next((candidate["candidate_id"]
                                               for candidate in candidates
                                               if candidate["selected"]), None)},
            "observed": False,
            "reason": ("declared region implementation; not a captured occurrence"
                       if selected.get("native") else
                       "a region plan is selected without a unique native API"),
        }
    elif candidates:
        execution = {
            "kind": "candidate_set_only",
            "selection_state": ("single_compatible_unselected" if len(candidates) == 1
                                else "multiple_compatible_unselected"),
            "selected": None,
            "observed": False,
            "reason": "declared native names are correspondences, not a selection",
        }
    else:
        execution = {
            "kind": "semantic_region_only",
            "selection_state": "unresolved",
            "selected": None,
            "observed": False,
            "reason": "no declared native region implementation",
        }
    return {
        "semantic_refs": list(semantic_refs) if semantic_refs else None,
        "native_candidates": candidates,
        "native_execution": execution,
    }
