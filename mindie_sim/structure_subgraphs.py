"""Export declared fusion groups without adding execution nodes.

[Role] Resolve subgraph membership after ordinary lowering and call expansion.
[Invariants] The final DAG defines inputs and externally visible outputs.
  Quantization nodes remain inside or outside according to actual dependencies.
  A group carries one candidate contract, not a second copy of member work.
[Timeouts & Locks] Linear in exported nodes plus finite group slices; no I/O.
[Side-Effect Policy] Pure dictionaries; no profiler or device imports.
"""
from collections import defaultdict

from .backends.abi_registry import default_abi_registry
from .schema.abi import AbiQueryKey, OperandPortSig, SOC_ARCHS
from .schema.stack import ManifestBinding, StackTrackSpec
from .structure.native import mxfp4_reference_coverage
from .structure.native import grouped_situ_contract

MLA_PROLOG_V3_CONTRACT = "MlaPrologV3"
MLA_PROLOG_V4_CONTRACT = "aclnnMlaPrologV4WeightNz"
MLA_PROLOG_V4_KINDS = frozenset({
    MLA_PROLOG_V4_CONTRACT, "MlaPrologV4WeightNz", "MlaPrologV4",
})
VALID_FUSION_STATUSES = frozenset({
    "fused_existing",
    "fused_framework_enable",
    "fused_new_kernel_required",
    "unfused_decomposed",
})
_NATIVE_KIND_TO_ABI_ID: dict[str, str] = {
    "MlaPrologV3": "aclnnMlaPrologV3WeightNz",
    "MlaPrologV3WeightNz": "aclnnMlaPrologV3WeightNz",
    "MlaPrologV4": "aclnnMlaPrologV4WeightNz",
    "MlaPrologV4WeightNz": "aclnnMlaPrologV4WeightNz",
    "aclnnMlaPrologV4WeightNz": "aclnnMlaPrologV4WeightNz",
    "GroupedMatmulSwigluQuant": "aclnnGroupedMatmulSwigluQuant",
    "GroupedMatmulSwigluQuantV2": "aclnnGroupedMatmulSwigluQuantV2",
    "DequantSituQuant": "aclnnDequantSituQuant",
    "HcPre": "hc_pre_sinkhorn",
    "HcPost": "HcPost",
    "LightningIndexer": "aclnnLightningIndexer",
    "VllmAscendQuantLightningIndexerV2": "VllmAscendQuantLightningIndexerV2",
    "MoeGatingTopK": "aclnnMoeGatingTopK",
    "MoeGatingTopKSoftmaxV2": "aclnnMoeGatingTopKSoftmaxV2",
    "AttnResFwd": "AttnResFwd",
    "RecurrentKda": "RecurrentKda",
    "ChunkKdaFwd": "ChunkKdaFwd",
}
_LOWRANK_HC_TRITON_ABIS: tuple[str, ...] = (
    "_hc_combine_norm_kernel",
    "_hc_gate_mix_kernel",
)
_ROLE_DEFAULT_ABIS: dict[str, tuple[str, ...]] = {
    "mla_prolog": ("aclnnMlaPrologV3WeightNz", "aclnnMlaPrologV4WeightNz"),
    "grouped_gate_up_activation_quant": (
        "aclnnGroupedMatmulSwigluQuantV2",
        "aclnnGroupedMatmulSwigluQuant",
    ),
    "grouped_gate_up_situ_quant": ("GroupedMatmulSituQuant",),
    "situ_quant": ("SituMxQuant",),
    "lightning_indexer": ("aclnnLightningIndexer",),
    "quant_lightning_indexer": (
        "VllmAscendQuantLightningIndexerV2",
        "aclnnQuantLightningIndexerV2",
    ),
    "hc_pre": ("hc_pre_sinkhorn",),
    "hc_post": ("HcPost",),
    "router_selection": ("aclnnMoeGatingTopK", "aclnnMoeGatingTopKSoftmaxV2"),
    "kda_recurrent": ("RecurrentKda", "ChunkKdaFwd"),
    "gdn_chunk": ("aclnnChunkGatedDeltaRule",),
    "gdn_recurrent": ("aclnnRecurrentGatedDeltaRule",),
    "attn_res_prefix_point": ("AttnResFwd",),
}
_MULTI_OP_FUSION_ROLES = frozenset({
    "grouped_gate_up_activation_quant",
    "grouped_gate_up_situ_quant",
    "situ_quant",
    "hc_pre",
    "hc_post",
})


def _subgraph_arch(declaration, ordered_nodes, explicit_arch: str | None) -> str | None:
    candidates = [explicit_arch, declaration.get("arch_family")]
    impl = declaration.get("implementation") or {}
    if isinstance(impl, dict):
        candidates.append(impl.get("arch_family"))
    for node in ordered_nodes:
        meta = (node.get("operator") or {}).get("metadata") or {}
        if isinstance(meta, dict):
            candidates.append(meta.get("arch_family"))
    for item in candidates:
        norm = str(item or "").strip().upper()
        if norm in SOC_ARCHS:
            return norm
    return None


def _candidate_abi_symbols(declaration, ordered_nodes) -> tuple[str, ...]:
    role = str(declaration.get("role") or "")
    impl = declaration.get("implementation") or {}
    view = declaration.get("tensor_view") or {}
    if role in {"hc_pre", "hc_post"} and "hc_lowrank" in view:
        return _LOWRANK_HC_TRITON_ABIS
    raw: list[str] = []
    if isinstance(impl, dict):
        if impl.get("native_kind"):
            raw.append(str(impl["native_kind"]))
        for item in impl.get("native_kinds") or ():
            if item:
                raw.append(str(item))
    if len(ordered_nodes) == 1:
        meta = (ordered_nodes[0].get("operator") or {}).get("metadata") or {}
        if isinstance(meta, dict):
            if meta.get("native_kind"):
                raw.append(str(meta["native_kind"]))
            native_call = meta.get("native_call") or {}
            api = native_call.get("api") if isinstance(native_call, dict) else None
            if isinstance(api, dict):
                for key in ("aclnn_symbol", "entry"):
                    if api.get(key):
                        raw.append(str(api[key]))
    if not raw and role in _ROLE_DEFAULT_ABIS:
        raw.extend(_ROLE_DEFAULT_ABIS[role])
    mapped = [_NATIVE_KIND_TO_ABI_ID.get(sym, sym) for sym in raw if sym]
    return tuple(dict.fromkeys(mapped))


_SUBGRAPH_DTYPE_ALIASES: dict[str, str] = {
    "fp8_e4m3": "fp8_e4m3fn",
    "fp8_e4m3fn": "fp8_e4m3fn",
    "float8_e4m3fn": "fp8_e4m3fn",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "fp4": "fp4_e2m1",
    "mxfp4": "fp4_e2m1",
    "fp4_e2m1": "fp4_e2m1",
    "hifp8": "hifloat8",
    "hif8": "hifloat8",
}
_SUBGRAPH_DENSE_DTYPES = frozenset({"bf16", "fp16", "fp32"})
_SUBGRAPH_INPUT_ACT_ROLES = frozenset({
    "x", "token_x", "query", "q", "k", "v", "key", "value", "beta",
    "hidden_states", "residual", "prefix_sum", "block_residual",
})
_SUBGRAPH_WEIGHT_ROLES = frozenset({
    "w", "weight", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
    "proj_weight",
})
_SUBGRAPH_PRIMARY_OUT_ROLES = frozenset({
    "y", "out", "attn_out", "hidden_states",
})
_SUBGRAPH_SCALE_OUT_ROLES = frozenset({"y_scale", "scale"})
_SUBGRAPH_STATE_ROLES = frozenset({"initial_state", "state", "final_state"})
_SUBGRAPH_ROLE_ACTIVATION: dict[str, str] = {
    "grouped_gate_up_activation_quant": "swiglu",
    "grouped_gate_up_situ_quant": "situ",
    "situ_quant": "situ",
}
_SUBGRAPH_MX_GROUP = 32
_SUBGRAPH_DEFAULT_ROWS = 16
_SUBGRAPH_DEFAULT_DIM = 128


def _norm_subgraph_dtype(raw) -> str:
    low = str(raw or "").strip().lower()
    return _SUBGRAPH_DTYPE_ALIASES.get(low, low)


def _first_present(*values):
    for val in values:
        if val is not None and val != "":
            return val
    return None


def _derive_port_quant(dtype: str, *, is_weight: bool, group_size: int,
                       scale_dtype: str, explicit_qmode: str | None) -> tuple[str, tuple[int, ...], str]:
    if explicit_qmode:
        grp = (group_size,) if group_size > 0 else ()
        return explicit_qmode, grp, scale_dtype or "none"
    if dtype in _SUBGRAPH_DENSE_DTYPES:
        return "dense", (), "none"
    if dtype == "int4":
        return "w4a8_int", (), scale_dtype or "fp32"
    if dtype == "int8":
        return ("per_channel_int8" if is_weight else "per_token_int8"), (), scale_dtype or "fp32"
    if dtype == "fp8_e4m3fn":
        if group_size == _SUBGRAPH_MX_GROUP or scale_dtype in {"fp8_e8m0", "e8m0", "uint8"}:
            return "mx_fp8", (_SUBGRAPH_MX_GROUP,), scale_dtype or "fp8_e8m0"
        return ("per_channel_fp8" if is_weight else "per_token_fp8"), (), scale_dtype or "fp32"
    if dtype == "fp4_e2m1":
        return "mx_fp4", (_SUBGRAPH_MX_GROUP,), scale_dtype or "fp8_e8m0"
    return "dense", (), scale_dtype or "none"


def _extract_subgraph_contract(declaration, ordered_nodes) -> dict:
    view = declaration.get("tensor_view") or {}
    impl = declaration.get("implementation") or {}
    role = str(declaration.get("role") or "")
    node_ops = [(node.get("operator") or {}) for node in ordered_nodes]
    node_metas = [(op.get("metadata") or {}) for op in node_ops]
    first_meta = node_metas[0] if node_metas else {}
    last_meta = node_metas[-1] if node_metas else {}

    in_raw = _first_present(
        view.get("input_dtype"), view.get("activation_dtype"), view.get("dtype"),
        declaration.get("input_dtype"), declaration.get("activation_dtype"), declaration.get("dtype"),
        impl.get("input_dtype"), impl.get("activation_dtype"), impl.get("dtype"),
        first_meta.get("activation_format"), first_meta.get("input_dtype"),
        first_meta.get("input_format"), first_meta.get("dtype"),
        node_ops[0].get("dtype") if node_ops else None,
    )
    wt_raw = _first_present(
        view.get("weight_dtype"), declaration.get("weight_dtype"), impl.get("weight_dtype"),
        *(meta.get("weight_format") for meta in node_metas),
        *(meta.get("weight_dtype") for meta in node_metas),
    )
    out_raw = _first_present(
        view.get("output_dtype"), declaration.get("output_dtype"), impl.get("output_dtype"),
        last_meta.get("output_format"), last_meta.get("output_dtype"),
        node_ops[-1].get("dtype") if node_ops else None,
        in_raw,
    )
    in_dt = _norm_subgraph_dtype(in_raw)
    wt_dt = _norm_subgraph_dtype(wt_raw if wt_raw is not None else in_raw)
    out_dt = _norm_subgraph_dtype(out_raw)
    state_dt = _norm_subgraph_dtype(_first_present(view.get("state_dtype"), first_meta.get("state_dtype")))

    group_raw = _first_present(
        view.get("group_size"), view.get("activation_group_size"),
        declaration.get("group_size"), impl.get("group_size"),
        first_meta.get("activation_group_size"), first_meta.get("group_size"),
    )
    group_size = int(group_raw) if group_raw is not None else 0
    in_scale_raw = _first_present(
        view.get("scale_dtype"), view.get("activation_scale_format"),
        declaration.get("scale_dtype"), impl.get("scale_dtype"),
        first_meta.get("activation_scale_format"), first_meta.get("scale_dtype"),
    )
    out_scale_raw = _first_present(
        view.get("output_scale_dtype"), declaration.get("output_scale_dtype"),
        last_meta.get("output_scale_dtype"), last_meta.get("activation_scale_format"),
    )
    in_scale = _norm_subgraph_dtype(in_scale_raw)
    out_scale = _norm_subgraph_dtype(out_scale_raw)
    qmode_raw = _first_present(
        view.get("quant_mode"), declaration.get("quant_mode"),
        impl.get("quant_mode"), first_meta.get("quant_mode"),
    )
    in_qmode, in_grp, in_sc = _derive_port_quant(
        in_dt, is_weight=False, group_size=group_size,
        scale_dtype=in_scale, explicit_qmode=str(qmode_raw) if qmode_raw else None,
    )
    wt_qmode, wt_grp, wt_sc = _derive_port_quant(
        wt_dt, is_weight=True, group_size=group_size,
        scale_dtype=in_scale, explicit_qmode=None,
    )
    act_raw = _first_present(
        view.get("activation"), view.get("activation_kind"),
        declaration.get("activation"), declaration.get("activation_kind"),
        impl.get("activation"), impl.get("activation_kind"),
        first_meta.get("activation"), first_meta.get("ffn_activation"),
        _SUBGRAPH_ROLE_ACTIVATION.get(role),
    )
    act_kind = str(act_raw or "none").strip().lower()
    has_bias = bool(
        view.get("has_bias") or declaration.get("has_bias") or impl.get("has_bias")
        or any(meta.get("has_bias") for meta in node_metas)
    )
    shape0 = (node_ops[0].get("local_shape") or ()) if node_ops else ()
    rows_raw = _first_present(
        view.get("tokens"), shape0[0] if len(shape0) >= 2 else None,
    )
    in_w_raw = _first_present(
        view.get("input_width"), view.get("hidden"), view.get("width"),
        shape0[-1] if len(shape0) >= 2 else None,
    )
    out_w_raw = _first_present(
        view.get("intermediate_width"), view.get("output_width"), in_w_raw,
    )
    rows = int(rows_raw) if rows_raw is not None else _SUBGRAPH_DEFAULT_ROWS
    in_w = int(in_w_raw) if in_w_raw is not None else _SUBGRAPH_DEFAULT_DIM
    out_w = int(out_w_raw) if out_w_raw is not None else in_w
    experts = max(1, int(_first_present(view.get("local_experts"), first_meta.get("local_experts"), 1)))
    world_size = max(1, int(_first_present(
        view.get("world_size"), declaration.get("world_size"),
        first_meta.get("world_size"), 1,
    )))
    is_draft = declaration.get("model_scope") == "draft" or any(_is_draft_node(n) for n in ordered_nodes)
    return {
        "role": role,
        "in_dt": in_dt, "wt_dt": wt_dt, "out_dt": out_dt, "state_dt": state_dt,
        "in_layout": _first_present(view.get("input_layout"), view.get("layout"), declaration.get("layout"), impl.get("layout")),
        "wt_layout": _first_present(view.get("weight_layout"), declaration.get("weight_layout"), impl.get("weight_layout")),
        "out_layout": _first_present(view.get("output_layout"), declaration.get("output_layout")),
        "in_qmode": in_qmode, "in_grp": in_grp, "in_sc": in_sc,
        "wt_qmode": wt_qmode, "wt_grp": wt_grp, "wt_sc": wt_sc,
        "out_scale": out_scale,
        "act_kind": act_kind, "has_bias": has_bias,
        "rows": rows, "in_w": in_w, "out_w": out_w, "experts": experts,
        "world_size": world_size,
        "rank_placement": str(_first_present(
            view.get("rank_placement"), declaration.get("rank_placement"), "local",
        )),
        "model_scope": "draft" if is_draft else "target",
        "enabled_passes": frozenset(declaration.get("enabled_stack_passes") or ()),
    }


def _parse_port_template(token: str) -> tuple[str, str, str, str, tuple[int, ...], str]:
    parts = token.split(":")
    role = parts[0] if len(parts) >= 2 else "x"
    dtype = _norm_subgraph_dtype(parts[1] if len(parts) >= 2 else parts[0])
    layout = parts[2] if len(parts) >= 3 and parts[2] else "ND"
    qmode = parts[3] if len(parts) >= 4 and parts[3] else "dense"
    grp = tuple(int(v) for v in parts[4].split("x") if v) if len(parts) >= 5 and parts[4] else ()
    scale = _norm_subgraph_dtype(parts[5]) if len(parts) >= 6 and parts[5] else "none"
    return role, dtype, layout, qmode, grp, scale


def _build_subgraph_abi_query(entry, target_arch: str, contract: dict) -> AbiQueryKey:
    in_template = entry.dtype_tuple[0] if entry.dtype_tuple else ()
    out_template = entry.allowed_output_tuples[0] if entry.allowed_output_tuples else ()
    rows, in_w, out_w = contract["rows"], contract["in_w"], contract["out_w"]
    operands: list[OperandPortSig] = []
    for token in in_template:
        role, tpl_dt, tpl_lay, tpl_qm, tpl_grp, tpl_sc = _parse_port_template(token)
        low_role = role.lower()
        is_weight = low_role in _SUBGRAPH_WEIGHT_ROLES or (
            low_role.startswith("weight") and "scale" not in low_role
        )
        if is_weight:
            dt = contract["wt_dt"] or tpl_dt
            lay = str(contract["wt_layout"] or tpl_lay or "FRACTAL_NZ")
            qm = contract["wt_qmode"] if contract["wt_dt"] else tpl_qm
            grp = contract["wt_grp"] if contract["wt_dt"] else tpl_grp
            sc = contract["wt_sc"] if contract["wt_dt"] else tpl_sc
            shape = (in_w, out_w)
            sc_shape = (out_w,) if sc != "none" else ()
        elif low_role in _SUBGRAPH_INPUT_ACT_ROLES:
            dt = contract["in_dt"] or tpl_dt
            lay = str(contract["in_layout"] or tpl_lay or "ND")
            qm = contract["in_qmode"] if contract["in_dt"] else tpl_qm
            grp = contract["in_grp"] if contract["in_dt"] else tpl_grp
            sc = contract["in_sc"] if contract["in_dt"] else tpl_sc
            shape = (rows, in_w)
            sc_shape = (rows,) if sc != "none" else ()
        elif low_role == "weight_scale" and contract["wt_dt"] == "int4":
            dt, lay, qm, grp, sc = "uint64", tpl_lay, "none", (), "none"
            shape, sc_shape = (out_w,), ()
        elif low_role in _SUBGRAPH_STATE_ROLES and contract["state_dt"]:
            dt, lay, qm, grp, sc = contract["state_dt"], tpl_lay, "dense", (), "none"
            shape, sc_shape = (rows, in_w), ()
        else:
            dt, lay, qm, grp, sc = tpl_dt, tpl_lay, tpl_qm, tpl_grp, tpl_sc
            shape = (contract["experts"],) if low_role == "group_list" else (rows,)
            sc_shape = ()
        operands.append(OperandPortSig(
            role=role, dtype=dt, layout=lay, transposed=False, shape=shape,
            quant_mode=qm, group_shape=grp, scale_dtype=sc, scale_shape=sc_shape,
        ))
    outputs: list[OperandPortSig] = []
    for token in out_template:
        role, tpl_dt, tpl_lay, _, _, _ = _parse_port_template(token)
        low_role = role.lower()
        if low_role in _SUBGRAPH_PRIMARY_OUT_ROLES:
            dt = contract["out_dt"] or tpl_dt
            lay = str(contract["out_layout"] or tpl_lay or "ND")
            shape = (rows, out_w)
        elif low_role == "query_norm_out" and contract["out_dt"]:
            dt, lay, shape = contract["out_dt"], str(contract["out_layout"] or tpl_lay or "ND"), (rows, out_w)
        elif low_role in _SUBGRAPH_SCALE_OUT_ROLES:
            dt = contract["out_scale"] or tpl_dt
            lay, shape = tpl_lay, (rows,)
        elif low_role in _SUBGRAPH_STATE_ROLES and contract["state_dt"]:
            dt, lay, shape = contract["state_dt"], tpl_lay, (rows, in_w)
        else:
            dt, lay, shape = tpl_dt, tpl_lay, (rows, out_w)
        outputs.append(OperandPortSig(
            role=role, dtype=dt, layout=lay, transposed=False, shape=shape,
            quant_mode="dense" if dt in _SUBGRAPH_DENSE_DTYPES else "none",
            group_shape=(), scale_dtype="none", scale_shape=(),
        ))
    act_kind = contract["act_kind"]
    if act_kind == "none" and entry.allowed_activations and "none" not in entry.allowed_activations:
        act_kind = entry.allowed_activations[0]
    return AbiQueryKey(
        soc_arch=target_arch,  # type: ignore[arg-type]
        sku_id="generic_sku",
        abi_id=entry.abi_id,
        graph_motif=contract["role"] or "subgraph",
        model_scope=contract["model_scope"],
        draft_quant_mode="none",
        pr_track="shipping",
        operands=tuple(operands),
        outputs=tuple(outputs),
        rank_placement=contract["rank_placement"],
        world_size=contract["world_size"],
        comm_group_id="local",
        comm_domain_width=1,
        comm_dtype="none",
        reduction_kind="none",
        activation_kind=act_kind,
        activation_params=(),
        has_bias=contract["has_bias"],
        has_active_mask=False,
        dispatch_quant_mode=0,
        shared_experts_per_rank=0,
        top_k=0,
        moe_expert_num=contract["experts"],
        qk_rope_head_dim=0,
        mla_variant="none",
        sfa_dcp_active=False,
        pcp_size=1,
        kv_cache_dtype="none",
        has_kpool_indexer=False,
        dcp_size=1,
        is_act_and_mul=True,
        enabled_stack_passes=contract["enabled_passes"],
    )


def _probe_subgraph_abi(
    symbols: tuple[str, ...],
    arch: str | None,
    declaration=None,
    ordered_nodes=(),
) -> tuple[str, str]:
    """Verify candidate symbols against sealed ``ProbedAbiRegistry.verify()``."""
    if not symbols:
        return "unshipped", ""
    registry = default_abi_registry()
    archs = (arch,) if arch in SOC_ARCHS else ("A5", "A3", "A2")
    contract = _extract_subgraph_contract(declaration or {}, ordered_nodes or ())
    proposed_match = ""
    last_rejection = ""
    for sym in symbols:
        for target_arch in archs:
            entries = registry.lookup(sym, target_arch, available_only=True)
            for entry in entries:
                query = _build_subgraph_abi_query(entry, target_arch, contract)
                verdict = registry.verify(query)
                if verdict.status == "admitted_shipping":
                    return "shipping", verdict.matched_entry_id or entry.abi_id
                if entry.vllm_ascend_track == "unmerged_pr" and not proposed_match:
                    track_spec = StackTrackSpec(
                        track="proposed", enabled_passes=query.enabled_stack_passes,
                    )
                    bindings = (
                        (ManifestBinding(
                            mechanism_id=entry.required_mechanism_id,
                            pr_url="proposed", revision="proposed",
                        ),)
                        if entry.required_mechanism_id else ()
                    )
                    prop = registry.verify(query, track_spec=track_spec, manifest_bindings=bindings)
                    if prop.status == "admitted_proposed_pr":
                        proposed_match = prop.matched_entry_id or entry.abi_id
                elif verdict.rejection_reason:
                    last_rejection = verdict.rejection_reason
    if proposed_match:
        return "proposed", proposed_match
    return "unshipped", last_rejection or symbols[0]


def resolve_subgraph_fusion_status(
    declaration, ordered_nodes, *, resource_bound=False, arch: str | None = None,
):
    """Return canonical ``(fusion_status, fusion_reason)`` via ``ProbedAbiRegistry``."""
    role = str(declaration.get("role") or "")
    impl = declaration.get("implementation") or {}
    declared_status = declaration.get("fusion_status")
    declared_reason = declaration.get("fusion_reason")
    if resource_bound and declared_status in {
        "fused_existing", "fused_framework_enable", "fused_new_kernel_required",
    }:
        return declared_status, declared_reason or "selected_native_fusion_resource_boundary"
    if not resource_bound and declared_status in VALID_FUSION_STATUSES:
        return declared_status, declared_reason or "declared_subgraph_fusion_status"

    resolved_arch = _subgraph_arch(declaration, ordered_nodes, arch)
    symbols = _candidate_abi_symbols(declaration, ordered_nodes)
    tier, matched = _probe_subgraph_abi(symbols, resolved_arch, declaration, ordered_nodes)
    single_kernel = (
        len(ordered_nodes) == 1
        and (
            ordered_nodes[0].get("boundary") == "kernel"
            or (ordered_nodes[0].get("operator") or {}).get("metadata", {}).get("native_kind")
            or impl.get("fused") is True
        )
    )
    if resource_bound or single_kernel or impl.get("fused") is True:
        if tier == "shipping":
            return "fused_existing", f"probed_abi_shipping:{matched}"
        if tier == "proposed":
            return "fused_framework_enable", f"probed_abi_proposed:{matched}"
        return "fused_new_kernel_required", f"probed_abi_unshipped:{matched or role}"

    if declaration.get("native_fusion_eligible") is False:
        return "unfused_decomposed", f"native_fusion_ineligible:{role}"
    eligible_prolog = (
        role == "mla_prolog" and impl.get("selection") == "vllm_ascend_mla_prolog_v3"
    )
    if eligible_prolog or role in _MULTI_OP_FUSION_ROLES:
        if tier in {"shipping", "proposed"}:
            return "fused_framework_enable", f"probed_abi_eligible:{matched}"
        return "fused_new_kernel_required", f"probed_abi_unshipped:{matched or role}"
    return "unfused_decomposed", f"decomposed_{len(ordered_nodes)}_member_operators"


def metadata(op):
    meta = dict(op.metadata)
    return {**meta.get("semantic_metadata", {}), **meta}


def _is_draft_scope_op(op) -> bool:
    meta = metadata(op)
    return bool(
        meta.get("layer_scope") == "draft"
        or meta.get("structure_model_scope") == "draft"
        or op.phase in {"draft", "draft_prefill"}
        or str(op.op_id).startswith(("draft", "draft_prefill."))
    )


def _is_draft_node(node) -> bool:
    meta = (node.get("operator") or {}).get("metadata") or {}
    return bool(
        meta.get("layer_scope") == "draft"
        or meta.get("structure_model_scope") == "draft"
        or node.get("phase") in {"draft", "draft_prefill"}
        or str(node.get("op_id") or "").startswith(("draft", "draft_prefill."))
    )


def _draft_plans_from_model(model, execution=None):
    aux = model.get("auxiliary")
    if not isinstance(aux, dict):
        return ()
    software = (execution or {}).get("software") if isinstance(execution, dict) else None
    mode = ""
    if isinstance(software, dict):
        meta = software.get("metadata")
        if isinstance(meta, dict):
            mode = str(meta.get("draft_mode") or "")
    candidates = ("dspark", "mtp") if mode in {"dspark", "verify", "draft_verify"} else ("mtp", "dspark")
    for name in candidates:
        spec = aux.get(name)
        if isinstance(spec, dict) and spec.get("layer_plans"):
            return spec["layer_plans"]
    return ()


def layer_structure(model, layer, *, op=None, execution=None):
    if op is not None and _is_draft_scope_op(op):
        plans = _draft_plans_from_model(model, execution)
    else:
        plans = model.get("layer_plans", ())
    if layer < 0 or layer >= len(plans):
        return {}
    plan = plans[layer]
    return {
        "attention_kind": plan["attention"],
        "attention_architecture": plan.get("architecture"),
        "ffn_kind": plan["ffn"],
        "residual_kind": (
            "mhc" if plan.get("hyper_connections") else
            "attention_residual" if plan.get("residual_block_size") else "residual"),
        "cache_kind": plan.get("cache_kind"),
        "compress_ratio": plan.get("compress_ratio", 0),
        "window": plan.get("window", 0),
        "indexer": plan.get("indexer", False),
        "owns_topk": plan.get("owns_topk", False),
        "index_query_per_layer": plan.get("index_query_per_layer", False),
        "topk_owner_layer_id": plan.get("topk_owner_layer_id"),
    }


def subgraph_contract(declaration):
    if declaration.get("coverage", {}).get("kind") == "multi_launch_stage_unknown":
        return None
    view, role = declaration["tensor_view"], declaration["role"]
    if role == "grouped_gate_up_situ_quant":
        return grouped_situ_contract(view)
    if role == "situ_quant":
        rows, width = view["rows"], view["width"]
        if (rows != int(rows) or width % 64 or view["input_dtype"] != "bf16"
                or view["output_dtype"] not in {"fp8_e4m3", "fp8_e5m2"}
                or view["scale_dtype"] != "fp8_e8m0"
                or tuple(view["scale_shape"]) != (rows, width // 64, 2)):
            return None
        return (
            "situ_mxquant_active", "SituMxQuant",
            ((int(rows), 2 * width), view["input_dtype"], "ND"),
            ((int(rows), width), view["output_dtype"], "ND"),
            ((int(rows), width // 64, 2), view["scale_dtype"], "ND"),
        )
    if role == "router_selection":
        rows, experts, topk = (view[key] for key in ("tokens", "experts", "topk"))
        dtype = view["dtype"]
        return (
            "router_selection_active", view["native_kind"], view["selection_mode"],
            ((rows, experts), dtype, "ND"), ((rows, topk), dtype, "ND"),
            ((rows, topk), "int32", "ND"),
        )
    if role == "lightning_indexer":
        rows, heads, width = (view[key] for key in ("tokens", "heads", "width"))
        if view["query_dtype"] not in {"bf16", "fp16"}:
            return None
        return (
            "lightning_indexer_active", "LightningIndexer",
            (((rows, heads, width), view["query_dtype"], "ND"),
             ((rows, heads), view["weight_dtype"], "ND")),
            ((1, width), view["cache_dtype"], "ND"),
            ((rows, 1, view["topk"]), view["selection_output_dtype"], "ND"),
        )
    if role == "mla_prolog":
        if (declaration.get("native_fusion_eligible") is False
                or view.get("absorbed_weight_dtype") is None):
            return None
        rows, hidden, heads, qr, latent, nope, rope = (
            view[key] for key in
            ("tokens", "hidden", "heads", "query_rank", "latent_rank", "nope_width", "rope_width"))
        dq, uq, dkv = view["projection_weight_dtypes"]
        norm, cache = view["norm_dtype"], view["cache_dtype"]
        query_norm = (((rows, qr), view["query_norm_dtype"], "ND")
                      if view["query_norm_output"] else ((), None, None))
        cache_contract = (((1, latent), cache, "ND"), ((1, rope), cache, "ND"))
        if "packed_cache" in view:
            cache_contract = (((1, view["packed_cache"]["row_bytes"]), "int8", "ND"),
                              ((), None, None))
        implementation = (declaration.get("implementation") or {})
        prolog_kind = (MLA_PROLOG_V4_CONTRACT
                       if implementation.get("native_kind") in MLA_PROLOG_V4_KINDS
                       else MLA_PROLOG_V3_CONTRACT)
        return (
            "mla_prolog_active", prolog_kind,
            ((rows, hidden), view["token_dtype"], "ND"),
            (((hidden, qr), dq), ((qr, heads * (nope + rope)), uq),
             ((heads, nope, latent), view["absorbed_weight_dtype"]),
             ((hidden, latent + rope), dkv), ((qr,), norm), ((latent,), norm)),
            cache_contract,
            (((rows, heads, latent), norm, "ND"),
             ((rows, heads, rope), norm, "ND"), query_norm),
        )
    if role == "quant_lightning_indexer" and view["query_dtype"] == "int8":
        rows, heads, width = (view[key] for key in ("tokens", "heads", "width"))
        return (
            "quant_lightning_indexer_active", "VllmAscendQuantLightningIndexerV2",
            (((rows, heads, width), view["query_dtype"], "ND"),
             ((rows, heads), view["weight_dtype"], "ND"),
             ((rows, heads), view["scale_dtype"], "ND")),
            (((1, width), view["cache_dtype"], "ND"),
             ((1,), view["scale_dtype"], "ND")),
            ((rows, 1, view["topk"]), view["selection_output_dtype"], "ND"),
        )
    if role in {"hc_pre", "hc_post"}:
        # Low-rank HC carries ``hc_lowrank`` and keeps its own structural boundary.
        if "hc_lowrank" in view:
            return None
        rows, hidden, hc, dtype = (view[key] for key in ("tokens", "hidden", "hc", "dtype"))
        x, out = ((rows, hc, hidden), dtype, "ND"), ((rows, hidden), dtype, "ND")
        post, comb = ((rows, hc), "fp32", "ND"), ((rows, hc, hc), "fp32", "ND")
        if role == "hc_post":
            return ("hyper_connection_active", "HcPost", (out, x, post, comb), (x,))
        mixed = view.get("mixed")
        if mixed is None:
            # The fused mHC projection slot is required; a low-rank variant
            # without it keeps its own declared boundary.
            return None
        return (
            "hyper_connection_active", "HcPre",
            (x, ((mixed, hc * hidden), "fp32", "ND"),
             ((3,), "fp32", "ND"), ((mixed,), "fp32", "ND")),
            (out, post, comb),
        )
    if role == "compressor":
        rows, hidden, width, expansion = (
            view[key] for key in ("tokens", "hidden", "width", "coff"))
        slot = lambda shape, dtype: (shape, dtype, "ND")
        # A C1 compressor (ratio 1) has no gated pooling: it owns only the
        # projection and norm, not a gate matrix or its APE table.
        weights = [slot((expansion * width, hidden), view["weight_dtype"])]
        if int(view.get("ratio", 0)) > 1:
            weights.extend([
                slot((expansion * width, hidden), view["weight_dtype"]),
                slot((view["ratio"], expansion * width), view["ape_dtype"]),
            ])
        weights.append(slot((width,), view["norm_dtype"]))
        return (
            "compressor_active", "Compressor", slot((rows, hidden), view["dtype"]),
            tuple(weights),
            slot((2 * expansion * width,), view["state_dtype"]),
            (slot((view["rope_width"],), view["rope_dtype"]),) * 2,
            slot((width,), view["dtype"]),
        )
    if role == "grouped_gate_up_activation_quant":
        weight, activation = view["weight_dtype"], view["input_dtype"]
        kind = ("GroupedMatmulSwigluQuantV2" if weight == "int4" else
                "GroupedMatmulSwigluQuant" if weight == "int8" else None)
        if kind and activation == view["output_dtype"] == "int8":
            return (
                "grouped_mlp_parameters_active", kind,
                (view["local_experts"], view["input_width"], view["intermediate_width"]),
                (activation, weight, view["output_dtype"]),
            )
    if role == "kda_recurrent" and view.get("in_kernel_gate_source"):
        rows, heads, width = (view[key] for key in ("tokens", "heads", "value_width"))
        dtype = view["dtype"]
        return (
            "recurrent_kda_parameters_active", "RecurrentKda",
            (((1, rows, heads, width), dtype, "ND"),
             ((1, rows, heads), view["beta_dtype"], "ND"),
             ((heads,), "fp32", "ND"), ((heads * width,), "fp32", "ND")),
            ((dtype, "ND"),) * 3, (view["state_dtype"], "ND"), (dtype, "ND"),
        )
    return None


def subgraph_scope(role):
    if role == "grouped_gate_up_situ_quant":
        return {
            "kind": "source_verified_MX_A8W4_expert_bank_and_channel_precision",
            "not_compared": ["routed_rows", "group_list_values", "beta", "linear_beta",
                             "native_dispatch"],
            "work_assumption": "expected_routing_work_is_declared_not_measured",
            "excluded_semantic_work": ["input_quantization", "routing",
                                       "down_projection", "combine"],
            "occurrence_identity": False,
            "prediction_coverage": mxfp4_reference_coverage(),
        }
    if role == "situ_quant":
        return ("active_input_output_MX_scale_shape_dtype_layout; "
                "beta_linear_beta_activate_left_axis_rounding_and_dispatch_not_observed; "
                "no_occurrence_proof")
    if role == "router_selection":
        return (
            "active_logits_selected_weights_and_ids_shape_dtype_layout_and_hash_table_presence; "
            "score_function_renorm_groups_bias_scaling_eps_and_table_values_not_observed; "
            "logits_gemm_excluded; Hash_renorm_external_TopK_declared_renorm_internal; "
            "no_dispatch_or_occurrence_proof")
    if role == "mla_prolog":
        return (
            "active_token_rows_logical_parameters_cache_heads_and_query_outputs; "
            "query_norm_output_uses_actual_lowered_activation_format_and_quantization_boundary; "
            "physical_parameter_storage_leading_axis_grouping_rope_inputs_scales_cache_capacity_"
            "slots_and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "lightning_indexer":
        return (
            "active_query_weights_cache_heads_and_selected_index_shape_dtype; "
            "cache_capacity_page_size_sequence_lengths_index_values_scores_scales_"
            "and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "quant_lightning_indexer":
        return ("quantized_query_weights_scales_cache_head_and_indices_shape_dtype; "
                "cache_capacity_blocksize_sequence_metadata_values_and_scalar_modes_not_compared; "
                "no_dispatch_or_occurrence_proof")
    if role in {"hc_pre", "hc_post"}:
        return ("active_hyper_connection_tensor_descriptors; leading_token_grouping_"
                "epsilon_sinkhorn_scalar_optional_pre_coefficients_not_compared; "
                "no_dispatch_or_occurrence_proof")
    if role == "compressor":
        return ("active_input_parameters_state_width_rope_width_output_width_dtype; "
                "output_reserved_rows_state_capacity_page_size_slots_rope_values_sequence_"
                "metadata_and_scalar_flags_not_compared; no_dispatch_or_occurrence_proof")
    if role == "grouped_gate_up_activation_quant":
        return ("local_expert_parameter_geometry_and_operand_precision; "
                "routed_rows_group_values_weight_packing_quant_scales_and_native_version_selection_"
                "not_compared; no_dispatch_or_occurrence_proof")
    if role == "kda_recurrent":
        return ("declared_in_kernel_gate_source_parameter_domain; "
                "gate_beta_Alog_dtbias_descriptors_and_QKV_state_output_precision; "
                "QKV_and_output_active_shapes_state_shapes_strides_indices_and_scalar_flags_"
                "not_compared; no_dispatch_or_occurrence_proof")
    return "declared_subgraph; no_observed_contract"


def subgraph_comparison_scope(role, members, *, resource=None):
    """Fusion preserves missing physical work unless its selected program supplies it."""
    scope = subgraph_scope(role)
    scopes = [scope, *(member.get("contract_scope") for member in members)]
    partial = []
    msd = (
        role == "grouped_gate_up_activation_quant" and resource is not None
        and resource["recipe"] == "ascend_w4a8_msd"
        and resource["operator"]["metadata"].get("fusion_program_profile") == "ascend_w4a8_msd"
    )
    for item in scopes:
        coverage = item.get("prediction_coverage", {}) if isinstance(item, dict) else {}
        if coverage.get("kind") != "partial_program":
            continue
        if (msd and coverage.get("reason") == "quantized_epilogue_unmodeled"
                and set(coverage.get("missing_work", ())) == {
                    "packed_int4_output_rescale", "quantization_offset_correction"}):
            # The MSD pre/matrix/post ledger replaces this incomplete matrix
            # program; unrelated missing work and MXFP4 assumptions still apply.
            continue
        if coverage not in partial:
            partial.append(coverage)
    if not partial:
        return scope
    combined = partial[0] if len(partial) == 1 else {
        "kind": "partial_program", "reason": "incomplete_member_programs",
        "missing_work": sorted({work for item in partial for work in item.get("missing_work", ())}),
        "components": partial,
    }
    return {**(scope if isinstance(scope, dict) else {"comparison_scope": scope}),
            "prediction_coverage": combined}


def export_subgraphs(nodes, declarations, memberships, entry_dependencies):
    """Take dependency closure between declared members and their actual inputs."""
    by_id = {node["node_id"]: node for node in nodes}
    position = {node["node_id"]: index for index, node in enumerate(nodes)}
    layer_structures = {
        (node["replica"], node["stage"], _is_draft_node(node), node["layer"]): node["structure"]
        for node in nodes
    }
    users = defaultdict(set)
    for node in nodes:
        for dependency in node["deps"]:
            users[dependency].add(node["node_id"])
    result = []
    for key, declaration in sorted(declarations.items()):
        replica, local_id = key
        seed = memberships[key]
        if not seed:
            raise ValueError(f"structure_subgraph_has_no_members:{local_id}")
        stop = entry_dependencies[key]
        members, pending = set(seed), list(seed)
        while pending:
            current = pending.pop()
            for dependency in by_id[current]["deps"]:
                if dependency not in stop and dependency in by_id and dependency not in members:
                    members.add(dependency)
                    pending.append(dependency)
        ordered = [by_id[node_id] for node_id in sorted(members, key=position.__getitem__)]
        first = ordered[0]
        phases = {node["phase"] for node in ordered}
        valid_phases = (
            len(phases) <= 1
            or phases <= {"decode", "draft"}
            or phases <= {"prefill", "draft", "draft_prefill"}
            or phases <= {"mixed", "prefill"}
            or phases <= {"mixed", "decode"}
        )
        if any((node["replica"], node["stage"]) != (first["replica"], first["stage"])
               for node in ordered) or not valid_phases:
            raise ValueError(f"structure_subgraph_crosses_placement:{local_id}")
        owners = set()
        owner_struct_keys = set()
        for node in ordered:
            is_draft = _is_draft_node(node)
            owners.add((replica, node["stage"], node["layer"], node["block"]))
            owner_struct_keys.add((replica, node["stage"], is_draft, node["layer"]))
            for origin in node["operator"]["metadata"].get("fused_origin_scopes", ()):
                owners.add((replica, origin["stage"], origin["layer"], origin["block"]))
                owner_struct_keys.add((replica, origin["stage"], is_draft, origin["layer"]))
        scopes = [{"replica": r, "stage": s, "layer": layer, "block": block}
                  for r, s, layer, block in sorted(
                      owners, key=lambda value: (value[0], value[1], value[2], value[3] or ""))]
        layers = sorted({scope["layer"] for scope in scopes})
        blocks = {scope["block"] for scope in scopes}
        common_structure = {
            key: value for key, value in first["structure"].items()
            if all(layer_structures.get(skey, {}).get(key) == value for skey in owner_struct_keys)
        }
        inputs = sorted({dep for node in ordered for dep in node["deps"] if dep not in members})
        outputs = [node["node_id"] for node in ordered
                   if users[node["node_id"]] - members or not users[node["node_id"]] & members]
        source_calls = sorted({node["lineage"]["source_call_id"] for node in ordered
                               if node["lineage"]["source_call_id"]})
        fusion_status, fusion_reason = resolve_subgraph_fusion_status(
            declaration, ordered, resource_bound=False)
        result.append({
            "subgraph_id": f"rank:{replica}/subgraph:{local_id}",
            "node_ids": [node["node_id"] for node in ordered],
            "role": declaration["role"], "family": declaration["family"],
            "replica": replica, "stage": first["stage"],
            "layer": layers[0] if len(layers) == 1 else None,
            "layer_ids": layers, "owner_scopes": scopes,
            "phase": first["phase"], "block": next(iter(blocks)) if len(blocks) == 1 else None,
            "scope_role": first["role"],
            "structure": common_structure, "boundary": "semantic_subgraph",
            "fusion_status": fusion_status,
            "fusion_reason": fusion_reason,
            "external_inputs": inputs, "external_outputs": outputs,
            "contract": subgraph_contract(declaration),
            "contract_scope": subgraph_comparison_scope(declaration["role"], ordered),
            "implementation": declaration.get("implementation", {}),
            "coverage": declaration.get("coverage", {
                "kind": "complete_subgraph", "limitations": [],
            }),
            "lineage": {
                "declared_op_ids": declaration["op_ids"],
                "source_call_ids": source_calls,
            },
            "source": declaration.get("source"),
        })
    return result
