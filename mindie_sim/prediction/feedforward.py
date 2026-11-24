"""Dense and routed feed-forward programs.

[Role] Describe router, packing, dispatch, expert matrices, combine and shared work.
[Invariants] Routing conserves top-k assignments; zero-token experts do not read
  matrices. Expected routing is a declared probability model, not a measured
  efficiency. Expert and tensor parallelism shard different dimensions.
[Timeouts & Locks] O(experts) arithmetic per layer, no histories or I/O.
[Side-Effect Policy] Constructs operator records only.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op, barrier_op
from mindie_sim.schema.common import ModelingError, dtype_bytes, inclusion_probability
from mindie_sim.prediction.operators.vector.gated_activation import (
    declare_situ_quant,
    gated_activation,
    situ_eager_chain,
)
from mindie_sim.prediction.operators.moe_transport.deepep import (deepep_exchange_factor, resolve_exchange_options,
                     scale_exchange_operator, split_deepep_combine_overlap,
                     split_deepep_dispatch_overlap, split_deepep_down_overlap,
                     split_deepep_up_overlap)
from mindie_sim.prediction.operators.moe_transport.moe_callers import RoutingPolicy, local_expert_reduce_recipe, tp_join_reduce
from mindie_sim.prediction.operators.moe_routing.moe_gating_fusion import (
    absorb_init_routing, build_selection, cann_native_active,
    dispatch_quantizes_internally,
)
from .sharding.ffn_rules import (
    bind_expert_slots, bind_transport_gate, declare_gate_up_fusion_group,
    routed_expert_work, router_gemm,
)
from mindie_sim.prediction.operators.moe_transport import deepep, moe_transport
from mindie_sim.prediction.operators.moe_routing import moe_token_partition

INVERSE_INDEX_DTYPE = "int32"
_PACK_ROUTING_INDEX_BYTES = 4
FUSED_MC2_DISABLED = 0
FUSED_MC2_DISPATCH_FFN_COMBINE = 1
FUSED_MC2_MEGAMOE_ROUTED = 2
FUSED_MC2_MEGAMOE_SHARED = 3
SUPPORTED_FUSED_MC2_LEVELS = frozenset({
    FUSED_MC2_DISABLED,
    FUSED_MC2_DISPATCH_FFN_COMBINE,
    FUSED_MC2_MEGAMOE_ROUTED,
    FUSED_MC2_MEGAMOE_SHARED,
})


def _normalize_arch_family(raw_arch: str | None) -> str:
    low = str(raw_arch or "").strip().lower()
    if low in {"a5", "arch35", "dav_3510"}:
        return "a5"
    if low in {"a3", "arch22"}:
        return "a3"
    if low in {"a2", "arch20"}:
        return "a2"
    return ""


def _model_is_latent_moe(model) -> bool:
    """Literal upstream ``routed_expert_hidden_size is not None`` + latent check."""
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    if facts.get("routed_expert_hidden_size") is not None:
        return True
    return any(
        int(getattr(p, "expert_latent_size", 0) or 0) > 0
        for p in getattr(model, "layer_plans", ())
        if getattr(p, "ffn", "") == "moe"
    )


def _shared_expert_fusion_block_reason(
    arch_norm: str, model, *, tp: int = 1, enable_shared_expert_dp: bool = False,
) -> str:
    """Upstream ``ascend_config.py:1118-1145`` & ``shared_experts.py:216-253`` guards."""
    del tp, enable_shared_expert_dp
    if arch_norm and arch_norm != "a5":
        return f"requires_a5_arch35:{arch_norm}"
    if _model_is_latent_moe(model):
        return "latent_moe_routed_expert_hidden_size"
    moe_plans = [
        p for p in getattr(model, "layer_plans", ())
        if getattr(p, "ffn", "") == "moe"
    ]
    if not any(int(getattr(p, "num_shared_experts", 0) or 0) > 0 for p in moe_plans):
        return "no_shared_expert"
    if any(bool(getattr(p, "shared_expert_gate", False)) for p in moe_plans):
        return "shared_expert_gate_unsupported"
    if any(
        float(getattr(p, "swiglu_shared_limit", 0.0) or 0.0)
        not in {0.0, float(getattr(p, "swiglu_limit", 0.0) or 0.0)}
        for p in moe_plans
    ):
        return "shared_activation_mismatch"
    from mindie_sim.catalog.quant import projection_quant
    from mindie_sim.prediction.projections import effective_linear_method

    model_quant = getattr(model, "quant", None)
    if model_quant is not None:
        for p in moe_plans:
            if not int(getattr(p, "num_shared_experts", 0) or 0):
                continue
            lq = projection_quant(model_quant, p)
            r_method = effective_linear_method(lq, "routed_expert")
            s_method = effective_linear_method(lq, "shared_expert")
            if r_method in {"mx_fp4", "mx_fp8"} and s_method not in {"mx_fp4", "mx_fp8"}:
                return f"shared_expert_precision_mismatch:{r_method}:{s_method}"
            if r_method in {"w8a8", "w4a8"} and s_method not in {"w8a8", "w4a8"}:
                return f"shared_expert_precision_mismatch:{r_method}:{s_method}"
    return ""


def resolve_moe_route_policy(
    *,
    arch: str | None,
    model,
    requested_fused_mc2: int | None,
    requested_mso: bool,
    tp: int = 1,
    enable_shared_expert_dp: bool = False,
    strict_level3: bool = False,
    mega_moe_requested: bool = False,
) -> dict[str, object]:
    """Resolve upstream-faithful ``enable_fused_mc2`` vs MSO mutex and level-3 legality."""
    if requested_fused_mc2 is not None:
        if isinstance(requested_fused_mc2, bool) or int(requested_fused_mc2) not in SUPPORTED_FUSED_MC2_LEVELS:
            raise ModelingError(f"invalid_enable_fused_mc2:{requested_fused_mc2!r}")
        raw_level: int | None = int(requested_fused_mc2)
    else:
        raw_level = None

    arch_norm = _normalize_arch_family(arch)
    fused_requested = (
        (raw_level is not None and raw_level > FUSED_MC2_DISABLED)
        or (raw_level is None and bool(mega_moe_requested) and arch_norm == "a3")
    )
    # Upstream ``ascend_config.py:936-941``: fused MC2 disables MSO unconditionally;
    # any later eligibility downgrade never re-enables MSO implicitly.
    effective_mso = False if fused_requested else bool(requested_mso)
    effective_level = raw_level
    shared_fusion = False
    disabled_reason = ""

    if raw_level == FUSED_MC2_DISPATCH_FFN_COMBINE:
        if arch_norm in {"a2", "a5"}:
            effective_level = FUSED_MC2_DISABLED
            disabled_reason = f"dispatch_ffn_combine_unsupported_on_{arch_norm}"
    elif raw_level == FUSED_MC2_MEGAMOE_ROUTED:
        if arch_norm == "a2":
            effective_level = FUSED_MC2_DISABLED
            disabled_reason = "megamoe_unsupported_on_a2"
    elif raw_level == FUSED_MC2_MEGAMOE_SHARED:
        if arch_norm and arch_norm != "a5":
            if strict_level3:
                raise ModelingError(f"enable_fused_mc2_level3_requires_a5:{arch_norm}")
            effective_level = FUSED_MC2_MEGAMOE_ROUTED if arch_norm == "a3" else FUSED_MC2_DISABLED
            disabled_reason = f"requires_a5_arch35:{arch_norm}"
        elif _model_is_latent_moe(model):
            if strict_level3:
                raise ModelingError("enable_fused_mc2_level3_unsupported_latent_moe")
            effective_level = FUSED_MC2_MEGAMOE_ROUTED
            disabled_reason = "latent_moe_routed_expert_hidden_size"
        else:
            block_reason = _shared_expert_fusion_block_reason(
                arch_norm, model, tp=tp, enable_shared_expert_dp=enable_shared_expert_dp,
            )
            if block_reason:
                effective_level = FUSED_MC2_MEGAMOE_ROUTED
                disabled_reason = block_reason
            else:
                effective_level = FUSED_MC2_MEGAMOE_SHARED
                shared_fusion = True

    return {
        "requested_enable_fused_mc2": raw_level,
        "effective_enable_fused_mc2": effective_level,
        "requested_multistream_overlap_shared_expert": bool(requested_mso),
        "effective_multistream_overlap_shared_expert": bool(effective_mso),
        "megamoe_shared_expert_fusion": bool(shared_fusion),
        "shared_fusion_disabled_reason": disabled_reason,
        "mutex_disabled_mso": bool(fused_requested and requested_mso),
    }


def _resolve_alltoall_moe_route(model, plan, parallel, runtime, *, overlap_shared: bool):
    """Resolve effective MoE route & mutex once before shared-expert stream assignment."""
    meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    additional = meta.get("additional_config")
    additional = dict(additional) if isinstance(additional, dict) else {}
    raw_mso = bool(
        overlap_shared or getattr(plan, "overlap_shared", False)
        or getattr(runtime, "multistream_overlap_shared_expert", False)
        or meta.get("overlap_shared_expert", False)
        or meta.get("enable_shared_expert_overlap", False)
    )
    arch = meta.get("arch_family") or (meta.get("va_profile") or {}).get("hardware_arch")
    raw_fused = additional.get("enable_fused_mc2")
    mega_req = bool(
        int(additional.get("enable_mega_moe", 0) or 0)
        or int(meta.get("moe_mega_moe_supported", 0) or 0)
    )
    route = resolve_moe_route_policy(
        arch=arch,
        model=model,
        requested_fused_mc2=raw_fused,
        requested_mso=raw_mso,
        tp=int(getattr(parallel, "tp", 1) or 1),
        enable_shared_expert_dp=bool(meta.get("enable_shared_expert_dp", False)),
        strict_level3=(not _is_raw_eager(runtime) and raw_fused == FUSED_MC2_MEGAMOE_SHARED),
        mega_moe_requested=mega_req,
    )
    eff_mso = bool(route["effective_multistream_overlap_shared_expert"])
    eff_fused = route["effective_enable_fused_mc2"]
    eff_runtime = runtime
    if runtime is not None and eff_fused != raw_fused:
        updated_additional = dict(additional)
        updated_additional["enable_fused_mc2"] = eff_fused
        updated_meta = dict(meta)
        if eff_fused == FUSED_MC2_DISABLED:
            updated_additional.pop("enable_mega_moe", None)
            updated_meta.pop("moe_mega_moe_supported", None)
        updated_meta["additional_config"] = updated_additional
        eff_runtime = replace(runtime, metadata=tuple(updated_meta.items()))
    return eff_mso, eff_runtime


def _latent_norm(prefix, source, stage, plan, requests, tokens, latent, model, mode):
    return vector_op(
        op_id=prefix + ".latent_norm", deps=(source,), stage=stage, layer=plan.index,
        requests=requests, elements=tokens * latent, ops_per_element=5,
        dtype=model.activation_dtype, family="rmsnorm", phase=mode, recipe="latent_norm",
        shape=(tokens, latent),
        read_bytes_override=(tokens + 1) * latent * dtype_bytes(model.activation_dtype),
        write_bytes_override=tokens * latent * dtype_bytes(model.activation_dtype),
        metadata=(("reduction_width", latent), ("norm_has_weight", True),
                  ("norm_eps", model.norm_eps),
                  ("latent_norm_position", plan.expert_latent_norm_position)))


def _is_raw_eager(runtime) -> bool:
    if runtime is None:
        return False
    meta = dict(getattr(runtime, "metadata", ()) or ())
    policy = meta.get("execution_policy")
    return bool(
        meta.get("raw_eager", False)
        or (isinstance(policy, dict) and policy.get("raw_eager", False))
    )


def _runtime_arch_upper(runtime) -> str:
    if runtime is None:
        return ""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    return _normalize_arch_family(meta.get("arch_family")).upper()


def validate_m10_moe_activation_legality(plan) -> None:
    """Enforce hardware-verified M10 MoE ``is_act_and_mul=False`` guard (e.g. ``relu2``)."""
    act = str(getattr(plan, "ffn_activation", "") or "").strip().lower()
    is_act_and_mul = getattr(plan, "is_act_and_mul", None)
    if (getattr(plan, "ffn", "") == "moe"
            and (act in {"relu2", "relu_squared"} or is_act_and_mul is False)):
        raise ModelingError(f"m10_moe_requires_is_act_and_mul:activation={act!r}")


_CLIPPED_CLAMP_OPS_PER_ELEM = 3
_CLIPPED_CLAMP_GATE_UP_FACTOR = 2
_FLOAT_LINEAR_FORMATS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_SITU_UNFUSED_CLIPPED_KERNELS: tuple[str, ...] = (
    "Cast", "Slice", "RealDiv", "Tanh", "Mul", "Slice", "Sigmoid", "Mul",
    "Slice", "RealDiv", "Tanh", "Mul", "Mul", "Cast",
)
_SITU_UNFUSED_PLAIN_KERNELS: tuple[str, ...] = (
    "Cast", "Slice", "RealDiv", "Tanh", "Mul", "Slice", "Sigmoid", "Mul",
    "Slice", "Mul", "Cast",
)


def _is_unquantized_linear(layer_quant, category: str) -> bool:
    """Return True iff both weight and activation formats for ``category`` are unquantized floats."""
    if layer_quant is None:
        return True
    from mindie_sim.catalog.ops import compute_dtype_for

    q = layer_quant.for_category(category)
    wf = str(q.format_for(category) or "").strip().lower().replace("-", "_")
    _, act_raw = compute_dtype_for(q, category)
    af = str(act_raw or "").strip().lower().replace("-", "_")
    return wf in _FLOAT_LINEAR_FORMATS and af in _FLOAT_LINEAR_FORMATS


def _clipped_swiglu_eager_chain(
    *, plan, rows: int, width: int, op_id: str, deps: tuple[str, ...],
    stage: int, requests: tuple[str, ...], dtype: str, phase: str, recipe: str,
) -> tuple[Any, ...]:
    """Emit upstream A5 unfused ``clipped_swiglu`` (``ClipByValue`` + SwiGLU; ``device_op.py:1525``).

    Conserves the exact total clamped expression vector ops across the two ops:
    ``clamp_gate_up`` charges ``SWIGLU_CLAMP_OPS_PER_ELEMENT = 3`` ops per output
    element (``min(gate, limit)`` + two-sided ``clamp(up, -limit, limit)``), and
    the downstream SwiGLU op charges the remaining non-clamp expression ops so
    nothing is double-counted.
    """
    elem_b = dtype_bytes(dtype)
    out_elems = max(0, int(rows)) * max(0, int(width))
    gate_up_elems = out_elems * _CLIPPED_CLAMP_GATE_UP_FACTOR
    clamp = vector_op(
        op_id=op_id + ".clamp_gate_up",
        deps=deps,
        stage=stage,
        layer=plan.index,
        requests=requests,
        elements=out_elems,
        ops_per_element=_CLIPPED_CLAMP_OPS_PER_ELEM,
        dtype=dtype,
        family="elementwise",
        phase=phase,
        recipe="clipped_swiglu_clamp",
        shape=(int(rows), int(width) * _CLIPPED_CLAMP_GATE_UP_FACTOR),
        read_bytes_override=gate_up_elems * elem_b,
        write_bytes_override=gate_up_elems * elem_b,
        metadata=(
            ("native_kinds", ("ClipByValue",)),
            ("clipped_swiglu_stage", "clamp_gate_up"),
        ),
    )
    act = gated_activation(
        plan=plan, rows=rows, width=width, op_id=op_id, deps=(clamp.op_id,),
        stage=stage, requests=requests, dtype=dtype, phase=phase, recipe=recipe,
    )
    act = replace(act, vector_ops=max(0.0, float(act.vector_ops) - float(clamp.vector_ops)))
    return (clamp, act)


def _linear_act_dtype(layer_quant, category: str) -> str:
    """Return normalized activation format for ``category``."""
    if layer_quant is None:
        return "bf16"
    from mindie_sim.catalog.ops import compute_dtype_for

    q = layer_quant.for_category(category)
    _, act_raw = compute_dtype_for(q, category)
    return str(act_raw or "").strip().lower().replace("-", "_")


_SITU_ABI_PROBE_DEFAULT_WIDTH = 256


def _verify_dequant_situ_quant_abi(
    reg, arch: str, *, width: int = _SITU_ABI_PROBE_DEFAULT_WIDTH,
) -> bool:
    """Verify full input+output ``aclnnDequantSituQuant`` tuple via ``ProbedAbiRegistry.verify``."""
    from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig

    sku_map = {"A2": "ascend910b2", "A3": "ascend910_9362", "A5": "ascend950dt_9582"}
    even_w = max(2, int(width) if int(width) % 2 == 0 else int(width) + 1)
    query = AbiQueryKey(
        soc_arch=arch,  # type: ignore[arg-type]
        sku_id=sku_map[arch],
        abi_id="aclnnDequantSituQuant",
        graph_motif="dequant_situ_quant",
        model_scope="target",
        draft_quant_mode="none",
        pr_track="shipping",
        operands=(OperandPortSig(
            role="x", dtype="bf16", layout="ND", transposed=False,
            shape=(1, even_w), quant_mode="dense", group_shape=(),
            scale_dtype="none", scale_shape=(),
        ),),
        outputs=(
            OperandPortSig(
                role="y", dtype="int8", layout="ND", transposed=False,
                shape=(1, even_w // 2), quant_mode="per_token_int8", group_shape=(),
                scale_dtype="fp32", scale_shape=(1,),
            ),
            OperandPortSig(
                role="y_scale", dtype="fp32", layout="ND", transposed=False,
                shape=(1,), quant_mode="none", group_shape=(),
                scale_dtype="none", scale_shape=(),
            ),
        ),
        rank_placement="local",
        world_size=1,
        comm_group_id="none",
        comm_domain_width=1,
        comm_dtype="none",
        reduction_kind="none",
        activation_kind="situ",
        activation_params=(),
        has_bias=False,
        has_active_mask=False,
        dispatch_quant_mode=0,
        shared_experts_per_rank=0,
        top_k=0,
        moe_expert_num=0,
        qk_rope_head_dim=0,
        mla_variant="none",
        sfa_dcp_active=False,
        pcp_size=1,
        kv_cache_dtype="bf16",
        has_kpool_indexer=False,
        dcp_size=1,
        is_act_and_mul=True,
        enabled_stack_passes=frozenset(),
    )
    return reg.verify(query).status == "admitted_shipping"


def _activation_abi_lowering(
    plan, runtime, layer_quant, category: str,
) -> tuple[bool, bool, tuple[tuple[str, object], ...]]:
    """Resolve shipping CANN vs unfused activation lowering strictly via ``ProbedAbiRegistry``.

    Returns ``(unfuse_situ, unfuse_clipped, extra_metadata)``.
    ``aclnnDequantSituQuant`` requires INT8 + FP32 quantized output ports
    (``dequant_situ_quant_def.cpp:61``) verified via ``ProbedAbiRegistry.verify()``
    and is NEVER claimed on BF16 dense or non-INT8 (MXFP4/FP8) down projections.
    """
    act = str(getattr(plan, "ffn_activation", "") or "").strip().lower()
    arch = _runtime_arch_upper(runtime)
    if not arch:
        return False, False, ()
    if arch not in {"A2", "A3", "A5"}:
        raise ModelingError(f"unsupported_activation_abi_arch:{arch!r}")
    reg = default_abi_registry()
    is_unquantized = _is_unquantized_linear(layer_quant, category)
    act_fmt = _linear_act_dtype(layer_quant, category)
    if act == "situ":
        inter_w = int(
            getattr(plan, "moe_intermediate_size", 0)
            or getattr(plan, "intermediate_size", 0)
            or _SITU_ABI_PROBE_DEFAULT_WIDTH
        )
        if (not is_unquantized
                and act_fmt == "int8"
                and _verify_dequant_situ_quant_abi(reg, arch, width=inter_w * 2)):
            return False, False, (
                ("aclnn_symbol", "aclnnDequantSituQuant"),
                ("native_kinds", ("DequantSituQuant",)),
                ("activation_output_dtypes", ("int8", "fp32")),
                ("activation_abi_shipping", True),
            )
        extra: tuple[tuple[str, object], ...] = (
            ("activation_abi_shipping", False),
            ("situ_bf16_unfused", bool(is_unquantized)),
        )
        if is_unquantized:
            has_linear_beta = getattr(plan, "activation_situ_linear_beta", None) is not None
            kernels = (
                _SITU_UNFUSED_CLIPPED_KERNELS
                if has_linear_beta
                else _SITU_UNFUSED_PLAIN_KERNELS
            )
            extra += (("native_kinds", kernels),)
        return False, False, extra
    has_clip = (
        act == "swigluoai"
        or (act == "silu" and float(getattr(plan, "swiglu_limit", 0) or 0) > 0.0)
    )
    if has_clip:
        if reg.is_shipping_available("aclnnClippedSwiglu", arch, activation="clipped_swiglu"):
            return False, False, (
                ("aclnn_symbol", "aclnnClippedSwiglu"),
                ("native_kinds", ("ClippedSwiglu",)),
                ("clipped_swiglu_unfused", False),
                ("activation_abi_shipping", True),
            )
        # When unfused into (clamp_gate_up + act), clamp_gate_up owns ClipByValue;
        # act owns only the remaining SwiGLU tail kernels (no double-counted ClipByValue).
        tail_kinds = ("Sigmoid", "Mul", "Add") if is_unquantized else ("ClipByValue", "Sigmoid", "Mul", "Add")
        return False, is_unquantized, (
            ("clipped_swiglu_unfused", bool(is_unquantized)),
            ("native_kinds", tail_kinds),
            ("activation_abi_shipping", False),
        )
    return False, False, ()


def _activation_params_meta(plan) -> tuple[tuple[str, object], ...]:
    """Export canonical ``activation_params`` mapping for rewrite/algebra consumers."""
    act = str(getattr(plan, "ffn_activation", "") or "")
    params: dict[str, float] = {}
    if act == "situ":
        params["beta"] = float(plan.activation_situ_beta)
        if plan.activation_situ_linear_beta is not None:
            params["linear_beta"] = float(plan.activation_situ_linear_beta)
    elif act == "swigluoai" or (act == "silu" and float(getattr(plan, "swiglu_limit", 0) or 0) > 0):
        params["limit"] = float(plan.swiglu_limit)
        params["alpha"] = float(plan.swiglu_alpha)
        params["beta"] = float(plan.swiglu_beta)
    return (("activation_params", params),) if params else ()


def dense(*, prefix, plan, deps, stage, requests, tokens, model, parallel, mode,
          category="dense_mlp", runtime=None, reduce_output=True, **_):
    # A declared selector is configuration even when this FFN is dense. Validate
    # it here so an unsupported declaration cannot sit silently unused.
    from mindie_sim.catalog.quant import projection_quant

    moe_transport.select(runtime)
    layer_quant = projection_quant(model.quant, plan)
    width = math.ceil(plan.intermediate_size / parallel.tp)
    gate = gemm_op(
        op_id=prefix + ".gate_up", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=2 * width, k=model.hidden_size,
        quant=layer_quant, category=category, phase=mode,
        family="gate_up_projection", recipe="dense_gate_up",
        metadata=(("ffn_role", "gate_up"), ("ffn_category", category)),
    )
    act_id = prefix + (".situ" if plan.ffn_activation == "situ" else ".swiglu")
    unfuse_situ, unfuse_clipped, abi_act_meta = _activation_abi_lowering(plan, runtime, layer_quant, category)
    act_extra_meta = (
        ("ffn_role", "activation"),
        ("ffn_category", category),
        *_activation_params_meta(plan),
        *abi_act_meta,
    )
    if plan.ffn_activation == "situ" and (_is_raw_eager(runtime) or unfuse_situ):
        act_ops = list(situ_eager_chain(
            plan=plan, rows=tokens, width=width, op_id=act_id,
            deps=(gate.op_id,), stage=stage, requests=requests,
            dtype=model.activation_dtype, phase=mode,
        ))
        # situ_eager_chain already emits every individual kernel op; strip collapsed native_kinds
        # so the final Cast op is not expanded a second time into 14 additional launches.
        eager_extra_meta = tuple(item for item in act_extra_meta if item[0] != "native_kinds")
        act_ops[-1] = replace(act_ops[-1], metadata=act_ops[-1].metadata + eager_extra_meta)
        activation = act_ops[-1]
    elif unfuse_clipped:
        act_ops = list(_clipped_swiglu_eager_chain(
            plan=plan, rows=tokens, width=width, op_id=act_id,
            deps=(gate.op_id,), stage=stage, requests=requests,
            dtype=model.activation_dtype, phase=mode, recipe="swiglu",
        ))
        act_ops[-1] = replace(act_ops[-1], metadata=act_ops[-1].metadata + act_extra_meta)
        activation = act_ops[-1]
    else:
        activation = gated_activation(
            plan=plan, rows=tokens, width=width, op_id=act_id,
            deps=(gate.op_id,), stage=stage,
            requests=requests, dtype=model.activation_dtype, phase=mode, recipe="swiglu",
        )
        activation = replace(activation, metadata=activation.metadata + act_extra_meta)
        act_ops = [activation]
    down = gemm_op(
        op_id=prefix + ".down", deps=(activation.op_id,), stage=stage, layer=plan.index,
        requests=requests, m=tokens, n=model.hidden_size, k=width,
        quant=layer_quant, category=category, phase=mode,
        family="down_projection", recipe="dense_down",
        metadata=(("ffn_role", "down"), ("ffn_category", category)),
    )
    ops, last = [gate, *act_ops, down], down.op_id
    declare_situ_quant(ops, key=prefix + ".situ_quant", activation=activation,
                       consumer=down, runtime=runtime, rows=tokens, width=width)
    if parallel.tp > 1 and reduce_output:
        reduction = replace(
            comm_op(
                op_id=prefix + ".tp", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="allreduce", phase=mode, recipe="ring_allreduce",
                link_bytes=2 * (parallel.tp - 1) / parallel.tp * tokens * model.hidden_size
                           * dtype_bytes(model.activation_dtype), shape=(tokens, model.hidden_size),
                metadata=(("group_axis", "tp"), ("steps", 2 * (parallel.tp - 1))),
            ),
            dtype=model.activation_dtype,
        )
        ops.append(reduction)
        last = reduction.op_id
    return ops, last


def _zero_expert_count(plan, model) -> int:
    """Number of zero/identity MoE slots competing in routing (e.g. LongCat)."""
    facts = dict(getattr(model, "catalog_facts", ()) or {})
    return max(0, int(
        getattr(plan, "zero_expert_num", 0)
        or getattr(model, "zero_expert_num", 0)
        or facts.get("zero_expert_num", 0)
        or 0
    ))


def _route_expert_rows(
    plan, model, parallel, runtime, routing, routing_realization, *,
    caller, token_partition, sequence_plan, group_tokens, ep_owner, price_owner,
):
    local_experts = max(1, (int(plan.num_experts) + int(parallel.ep) - 1)
                        // int(parallel.ep))
    zero_experts = _zero_expert_count(plan, model)
    routing_group_tokens = (
        sequence_plan.valid_cohort_tokens
        if (routing_realization is None
            and sequence_plan is not None
            and sequence_plan.padding == "right_to_tp"
            and 0 < sequence_plan.valid_cohort_tokens < group_tokens)
        else group_tokens
    )
    partitions = (sequence_plan.source_partitions if sequence_plan else
                  getattr(routing_realization, "source_partitions",
                          min(parallel.tp, parallel.ep)))
    if caller is not None:
        # One replicated source owns the complete token selection; the native
        # ledger is projected onto this rank's contiguous expert bank.
        projected = caller.prepare(
            routing, model, plan, parallel, routing_group_tokens,
            routing_realization, runtime=runtime)
        routing_realization = projected.routing_realization
        local_rows, active = projected.local_rows, projected.active
        routing_metadata = projected.routing_metadata
        partitions = 1
    else:
        if token_partition:
            partitions = 1
            if routing_realization is not None:
                # Keep the cohort's integer expert counts; only source ownership
                # changes from implicit replicated assignment division to TP views.
                routing_realization = replace(routing_realization, source_partitions=1)
        if routing_realization is None and type(routing) is RoutingPolicy:
            local_rows, active = routing.local(
                model, plan, parallel, routing_group_tokens,
                owner=price_owner, runtime=runtime, zero_experts=zero_experts,
            )
        elif routing_realization is None:
            local_rows, active = routing.local(model, plan, parallel, routing_group_tokens)
            if zero_experts > 0 and int(plan.num_experts) > 0:
                total_experts = int(plan.num_experts) + zero_experts
                local_rows = float(local_rows) * (float(plan.num_experts) / float(total_experts))
                p_real = (
                    min(1.0, float(plan.top_k) / float(total_experts))
                    if int(plan.top_k) > 0 else 0.0
                )
                active = float(local_experts) * inclusion_probability(float(routing_group_tokens), p_real)
        else:
            local_rows, active = routing_realization.rows, routing_realization.active
        from mindie_sim.prediction.operators.moe_transport.moe_callers import _eplb_damping_active
        routing_metadata = (
            (("routing_basis", "probability_moments"), ("expected_rows", local_rows),
             ("expected_active_experts", active))
            if routing_realization is None else
            (("routing_basis", routing_realization.basis),
             ("routing_realization", routing_realization.to_dict())))
        routing_metadata += (
            ("routing_valid_tokens", int(routing_group_tokens)),
            ("routing_policy_kind", getattr(routing, "kind", type(routing).__name__)),
            ("routing_price_owner", int(price_owner) if price_owner is not None else 0),
            ("routing_bank_ids", tuple(range(local_experts))),
            ("routing_eplb_active", bool(_eplb_damping_active(parallel, runtime))),
        )
        if zero_experts > 0:
            routing_metadata += (("zero_expert_num", zero_experts),)
    if token_partition:
        routing_metadata += token_partition.metadata()
    # The local EP owner this graph represents. Base and redundant physical
    # slots are a property of that owner, not of the logical expert ledger.
    local_ep_rank = int(
        getattr(routing_realization, "representative_rank", 0) or 0)
    if routing_realization is None and ep_owner is not None:
        # Probability routing has no explicit representative rank.  The
        # physical EP owner still decides which redundant slot bank this graph
        # represents; pricing every replica as owner zero is not a copy.
        local_ep_rank = int(ep_owner) % max(1, int(parallel.ep))
    destination_rows = (
        tuple(int(value) for value in routing_realization.rank_assignment_rows)
        if routing_realization is not None
        and getattr(routing_realization, "rank_assignment_rows", None) is not None
        else None)
    represented_ranks = (
        tuple(int(value) for value in
              (getattr(routing_realization, "represented_ep_ranks", None)
               or (local_ep_rank,)))
        if destination_rows is not None else ())
    receive_rows = (
        max((destination_rows[rank] for rank in represented_ranks
             if 0 <= rank < len(destination_rows)), default=0)
        if destination_rows is not None else int(math.ceil(local_rows)))
    runtime_routing = dict(getattr(runtime, "routing", {}) or {})
    has_slots = (
        int(getattr(parallel, "redundant_experts_per_rank", 0) or 0) > 0
        or "expert_slot_map" in runtime_routing
        or (int(plan.num_experts) > 0 and int(parallel.ep) > 0
            and int(plan.num_experts) % int(parallel.ep) != 0))
    balanced = (routing_realization is not None
                and routing_realization.basis in {"forced_balanced_integer_counts",
                                                  "source_force_eplb_masked_integer_counts"})
    return (routing_realization, local_rows, active, routing_metadata, partitions,
            local_experts, local_ep_rank, receive_rows, runtime_routing, has_slots,
            balanced, destination_rows)


def _pack_routed_inputs(
    *, prefix, plan, stage, requests, model, parallel, mode, runtime,
    deps, tokens, full_tokens, hidden, latent, transport, token_partition, sequence_plan,
    routing_realization, local_rows, partitions, caller, routing_metadata, ep_owner=None,
):
    zero_experts = _zero_expert_count(plan, model)
    router_plan = (
        replace(plan, num_experts=int(plan.num_experts) + zero_experts)
        if zero_experts > 0 else plan
    )
    router_prefix, router, weight_dtype, weight_bytes = router_gemm(
        prefix=prefix, plan=router_plan, deps=deps, stage=stage, requests=requests,
        tokens=tokens, model=model, hidden=hidden, mode=mode)
    routed_input, logits_input = deps, router.op_id
    partition_views = ()
    if token_partition:
        partition_views = token_partition.prepare(
            prefix=prefix, input_id=deps, logits_id=router.op_id, width=hidden,
            experts=router_plan.num_experts, stage=stage, layer=plan.index,
            requests=requests, phase=mode, dtype=model.activation_dtype,
            logits_dtype=token_partition.logits_dtype or weight_dtype)
        routed_input, logits_input = (op.op_id for op in partition_views[-2:])
        tokens = token_partition.local_rows
    selection = build_selection(
        prefix=prefix, plan=router_plan, logits_input=logits_input,
        weight_dtype=weight_dtype, weight_bytes=weight_bytes, stage=stage,
        requests=requests, tokens=tokens, mode=mode, runtime=runtime)
    # An idle representative owns no local hidden-state row: its router and
    # selection calls are not scheduled, while the pack still names its
    # selection producer contract.
    ops = ([*router_prefix, router, *partition_views, *selection.ops]
           if tokens else [])
    select = selection.select
    ids_producer = selection.ids_producer if tokens else select.op_id
    weights_producer = selection.weights_producer if tokens else None
    exchange_options = resolve_exchange_options(runtime, transport, layer_scope=plan.scope)
    exchange_factor = deepep_exchange_factor(
        options=exchange_options, ep=parallel.ep, top_k=plan.top_k,
        tokens=tokens, hidden=latent,
        dtype_bytes_val=(transport.element_bytes(latent) if transport is not None
                         else dtype_bytes(model.activation_dtype)))
    source_traffic, payload_tokens = source_issuance(
        routing_realization=routing_realization, token_partition=token_partition,
        sequence_plan=sequence_plan, plan=plan, tokens=tokens,
        full_tokens=full_tokens, ep_owner=ep_owner)
    transport_rows = None
    transport_histogram_gather = None
    if transport is not None:
        # Count/quantize the same valid source rows the pack and explicit peer
        # ledger use.  ``tokens`` may be a graph bucket while an exact route
        # owns fewer valid source rows; reusing the bucket multiplies both
        # histogram and FP32/INT8 scale work without any routed payload.
        transport_rows = moe_transport.source_rows(
            payload_tokens, plan.top_k, partitions)
        transport_ops, transport_histogram_gather = moe_transport.histogram(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            rows=transport_rows, ep=parallel.ep, deps=deps,
            ids_producer=ids_producer if tokens else None,
            realization=routing_realization, mode=mode, transport=transport)
        ops.extend(transport_ops)
    source = routed_input
    if latent != hidden and tokens:
        from mindie_sim.catalog.quant import projection_quant

        layer_quant = projection_quant(model.quant, plan)
        projection = gemm_op(
            op_id=prefix + ".latent_down", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=tokens, n=latent, k=hidden, quant=layer_quant,
            category="latent_moe", phase=mode, family="projection", recipe="latent_projection",
        )
        ops.append(projection)
        source = projection.op_id
        if plan.expert_latent_norm_position == "pre":
            latent_norm = _latent_norm(
                prefix, source, stage, plan, requests, tokens, latent, model, mode)
            ops.append(latent_norm)
            source = latent_norm.op_id
    act_elem_bytes = dtype_bytes(model.activation_dtype)
    pack_read = payload_tokens * (latent * act_elem_bytes + plan.top_k * _PACK_ROUTING_INDEX_BYTES)
    pack_write = payload_tokens * plan.top_k * (latent * act_elem_bytes + _PACK_ROUTING_INDEX_BYTES)
    pack_ops = pack_extra = None
    if caller is not None:
        pack_read, pack_write, pack_ops = caller.pack_work(
            rows=tokens, latent=latent, local_rows=local_rows,
            weight_bytes=weight_bytes, act_bytes=dtype_bytes(model.activation_dtype))
        pack_extra = caller.pack_metadata(local_rows)
    packing = vector_op(
        op_id=prefix + ".pack",
        deps=tuple(dict.fromkeys((source, select.op_id, ids_producer))) if tokens else (deps,),
        stage=stage, layer=plan.index, requests=requests,
        elements=payload_tokens * plan.top_k, ops_per_element=1,
        vector_ops_override=pack_ops,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_pack",
        read_bytes_override=pack_read, write_bytes_override=pack_write,
        shape=(tokens, plan.top_k),
        metadata=(
            ("topk_ids_producer", ids_producer),
            ("payload_producer", source),
            ("num_experts", plan.num_experts),
            ("top_k", plan.top_k),
            ("topk_ids_dtype", "int32"),
            ("packed_layout", "expert_grouped"),
            *((pack_extra or ())),
            *routing_metadata,
        ),
    )
    if transport_histogram_gather is not None:
        packing = replace(packing, deps=tuple(dict.fromkeys((
            *packing.deps, transport_histogram_gather))))
    ops.append(packing)
    last = packing.op_id
    return (ops, packing, last, tokens, weight_dtype, weight_bytes, weights_producer,
            exchange_options, exchange_factor, source_traffic, payload_tokens,
            transport_rows, transport_histogram_gather)


def _masked_collective(op, source_traffic, row_bytes, ep, *, reverse=False):
    if source_traffic is None:
        return op
    matrix = source_traffic.payload_matrix(row_bytes, reverse=reverse)
    meta = {k: v for k, v in op.metadata if k != "reverse_of_collective"}
    meta.update(peer_volume_basis="group_mean", peer_traffic={"kind": "matrix", "payload_bytes": matrix},
                traffic_model=source_traffic.basis, source_token_mask=source_traffic.to_dict())
    return replace(op, family="alltoallv", link_bytes=sum(map(sum, matrix)) / ep,
                   metadata=tuple(meta.items()))


def _dispatch_routed_inputs(
    ops, packing, last, *, prefix, plan, stage, requests, model, parallel, mode, runtime,
    transport, transport_rows, transport_histogram_gather, latent, tokens, payload_tokens,
    partitions, routing_realization, source_traffic, exchange, exchange_options,
    exchange_factor, local_rows, receive_rows, mc2_mask_pad_id=None,
):
    transport_scale = transport_payload = transport_quantized = None
    dispatch_overlap_id = None
    dispatch_source = None
    gate_row_cut = None
    mask_pad_consumed = False
    if transport is not None:
        transport_quantized = moe_transport.quantize(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            pack=packing, transport=transport, rows=transport_rows,
            width=latent, partitions=partitions,
            mode=mode)
        ops.append(transport_quantized)
        exchange_ops, transport_scale, transport_payload = moe_transport.dispatch(
            prefix=prefix, plan=plan, stage=stage, requests=requests,
            quantized=transport_quantized, transport=transport,
            rows=transport_rows, width=latent, ep=parallel.ep,
            partitions=partitions, mode=mode,
            hist_gather=transport_histogram_gather)
        for exchange_op in exchange_ops:
            if mc2_mask_pad_id is not None and not mask_pad_consumed:
                exchange_op = replace(
                    exchange_op,
                    deps=tuple(dict.fromkeys((*exchange_op.deps, mc2_mask_pad_id))),
                    metadata=exchange_op.metadata + (("mc2_active_mask_producer", mc2_mask_pad_id),),
                )
                mask_pad_consumed = True
            if exchange_op.recipe == "expert_quantized_dispatch":
                exchange_op = scale_exchange_operator(
                    exchange_op, exchange_factor * exchange_options["eplb"],
                    exchange_options, ep=parallel.ep, top_k=plan.top_k)
                transport_cut = deepep.overlap_row_cut(
                    exchange_op.local_shape[0], exchange_options["overlap"])
                split_ops, dispatch_overlap_id = split_deepep_dispatch_overlap(
                    exchange_op, exchange_options, row_cut=transport_cut)
                ops.extend(split_ops)
            else:
                ops.append(exchange_op)
        if mc2_mask_pad_id is not None and not mask_pad_consumed:
            transport_quantized = replace(
                transport_quantized,
                deps=tuple(dict.fromkeys((*transport_quantized.deps, mc2_mask_pad_id))),
                metadata=transport_quantized.metadata + (("mc2_active_mask_producer", mc2_mask_pad_id),),
            )
            ops[-1] = transport_quantized
            mask_pad_consumed = True
        if transport_payload is not None:
            last = transport_payload
        else:
            last = transport_quantized.op_id
    # One logical source assignment belongs to one local EP rank. Both routing
    # paths use the same token-to-EP-rank ownership, so the declared volume is
    # per physical source rank rather than a replicated TP total. ETP groups
    # still need their own copies; they are separate expert shards.
    source_partitions = partitions
    routing_basis = routing_realization.basis if routing_realization is not None else "probability_moments"
    dispatch_row_bytes, combine_row_bytes, dispatch_quant_meta, combine_quant_meta, pack_quant = (
        moe_transport.default_exchange_contract(
            model, runtime, plan, transport, latent, ep=parallel.ep, prefix=prefix,
            stage=stage, requests=requests, pack=packing, tokens=payload_tokens,
            partitions=partitions, mode=mode))
    absorbed_routing = None
    absorbed_routing_meta = None
    if pack_quant is not None:
        absorbed_pack, retired = (
            absorb_init_routing(pack=packing, quantizer=pack_quant, ep=parallel.ep,
                dispatch_owns_quant=dispatch_quantizes_internally(runtime))
            if cann_native_active(runtime) else (packing, None))
        if retired is not None:
            pack_idx = next(
                (i for i in range(len(ops) - 1, -1, -1) if ops[i].op_id == packing.op_id),
                len(ops) - 1,
            )
            packing = absorbed_pack
            ops[pack_idx] = packing
            last = packing.op_id
            absorbed_routing = packing
            absorbed_routing_meta = dict(packing.metadata)
        else:
            ops.append(pack_quant)
            last = pack_quant.op_id
    # A declared source assignment is partitioned across its physical source
    # ranks for every routing path.  The old implicit min(tp, ep) special case
    # charged each replica the full assignment, multiplying the per-source
    # egress by the partition count.  The partition is the exact participant
    # count; no coefficient or operator special case.
    exchange_partitions = source_partitions
    ep_scale = tokens * plan.top_k * (parallel.ep - 1) / parallel.ep / exchange_partitions
    dispatch_exchange_bytes, combine_exchange_bytes = ep_scale * dispatch_row_bytes, ep_scale * combine_row_bytes
    collective_meta = (("group_axis", "ep"),
                       ("comm_axis", "ep"),
                       ("comm_group_size", int(parallel.ep)),
                       ("source_assignment_partitions", source_partitions),
                       ("traffic_model", "uniform_peer_payload"), ("routing_basis", routing_basis))
    if local_rows and exchange_options["overlap"] > 0.0:
        gate_row_cut = deepep.overlap_row_cut(
            max(0, int(math.ceil(local_rows))), exchange_options["overlap"])
    if parallel.ep > 1 and transport is None and exchange:
        dispatch_deps = (
            tuple(dict.fromkeys((last, mc2_mask_pad_id)))
            if (mc2_mask_pad_id is not None and not mask_pad_consumed)
            else (last,)
        )
        mask_meta = (
            (("mc2_active_mask_producer", mc2_mask_pad_id),)
            if (mc2_mask_pad_id is not None and not mask_pad_consumed)
            else ()
        )
        if mc2_mask_pad_id is not None and not mask_pad_consumed:
            mask_pad_consumed = True
        dispatch = comm_op(
            op_id=prefix + ".dispatch", deps=dispatch_deps, stage=stage, layer=plan.index,
            requests=requests, family="alltoall", phase=mode, recipe="expert_dispatch",
            link_bytes=dispatch_exchange_bytes,
            shape=(int(tokens) * int(plan.top_k), latent),
            metadata=collective_meta + (
                ("collective_id", prefix + ".dispatch"),
                ("moe_num_experts", int(plan.num_experts)),
                ("moe_ep_size", int(parallel.ep)),
            ) + dispatch_quant_meta + mask_meta,
        )
        dispatch = _masked_collective(
            dispatch, source_traffic, dispatch_row_bytes, parallel.ep)
        dispatch = scale_exchange_operator(
            dispatch, exchange_factor * exchange_options["eplb"],
            exchange_options, ep=parallel.ep, top_k=plan.top_k)
        # The receive allocation is a concrete value of this collective: the
        # row views below alias it instead of searching an ancestry for dtype.
        received_dtype = (absorbed_routing_meta or {}).get(
            "activation_payload_format") or model.activation_dtype
        dispatch = deepep.declare_received_value(
            dispatch, dtype=str(received_dtype),
            shape=(receive_rows, latent),
            row_bytes=latent * dtype_bytes(str(received_dtype)),
            storage=f"moe.rx:{prefix}.dispatch",
            binding="static", binding_name=f"{prefix}.dispatch.rx")
        dispatch_source = deepep.received_value_contract(dispatch)
        split_ops, dispatch_overlap_id = split_deepep_dispatch_overlap(
            dispatch, exchange_options, row_cut=gate_row_cut)
        ops.extend(split_ops)
        last = split_ops[0].op_id
    return (packing, last, transport_scale, transport_payload, transport_quantized,
            dispatch_overlap_id, dispatch_source, gate_row_cut, absorbed_routing,
            dispatch_quant_meta, combine_quant_meta, combine_row_bytes,
            combine_exchange_bytes, collective_meta, mask_pad_consumed)


def _emit_routed_activation(
    *, plan, runtime, layer_quant, activation_rows, expert_width, prefix,
    act_deps, stage, requests, model, mode, gate_segments, token_partition,
    caller, local_rows,
):
    """Build routed MoE activation op chain with ABI lowering and segment metadata."""
    unfuse_situ, unfuse_clipped, abi_act_meta = _activation_abi_lowering(plan, runtime, layer_quant, "routed_expert")
    is_eager_situ = plan.ffn_activation == "situ" and (_is_raw_eager(runtime) or unfuse_situ)
    if is_eager_situ:
        act_ops = list(situ_eager_chain(
            plan=plan, rows=activation_rows, width=expert_width,
            op_id=prefix + ".expert_act", deps=act_deps, stage=stage,
            requests=requests, dtype=model.activation_dtype, phase=mode,
        ))
    elif unfuse_clipped:
        act_ops = list(_clipped_swiglu_eager_chain(
            plan=plan, rows=activation_rows, width=expert_width,
            op_id=prefix + ".expert_act", deps=act_deps, stage=stage,
            requests=requests, dtype=model.activation_dtype, phase=mode,
            recipe="grouped_swiglu",
        ))
    else:
        act_ops = [gated_activation(
            plan=plan, rows=activation_rows, width=expert_width,
            op_id=prefix + ".expert_act", deps=act_deps,
            stage=stage, requests=requests, dtype=model.activation_dtype,
            phase=mode, recipe="grouped_swiglu",
        )]
    filtered_abi_meta = (
        tuple(item for item in abi_act_meta if item[0] != "native_kinds")
        if is_eager_situ else abi_act_meta
    )
    act_params_meta = _activation_params_meta(plan) + filtered_abi_meta
    if act_params_meta:
        act_ops[-1] = replace(act_ops[-1], metadata=act_ops[-1].metadata + act_params_meta)
    if gate_segments:
        gate_consumer = act_ops[0]
        gate_consumer = replace(gate_consumer, metadata=gate_consumer.metadata + (
            ("memory_accesses", tuple({
                "producer": op.op_id,
                "output": dict(op.metadata).get("primary_output") or "output",
                "direction": "read", "position": "local",
                "storage": f"moe.act:{op.op_id}",
                "bytes": float(int(op.local_shape[0]) * int(op.local_shape[1])
                               * dtype_bytes(dict(op.metadata).get("output_format") or model.activation_dtype)),
                "span": ((0, int(op.local_shape[0])),),
                "last_use": prefix + ".expert_down",
                "purpose": "gate_output_rows",
            } for op in gate_segments)),))
        segment_meta = tuple(
            (op.op_id, dict(op.metadata).get("deepep_physical_row_start"),
             dict(op.metadata).get("deepep_physical_row_stop"),
             int(op.local_shape[0]), int(op.local_shape[1]))
            for op in gate_segments)
        gate_consumer = replace(gate_consumer, metadata=gate_consumer.metadata + (
            ("activation_input_segments", segment_meta),
            ("activation_input_bytes", sum(
                int(op.local_shape[0]) * int(op.local_shape[1])
                * dtype_bytes(dict(op.metadata).get("output_format") or model.activation_dtype)
                for op in gate_segments)),
        ))
        act_ops[0] = gate_consumer
    activation = act_ops[-1]
    if token_partition is not None and token_partition.activation_owners:
        activation = replace(activation, metadata=activation.metadata + (
            ("activation_owners", token_partition.activation_owners),
            ("activation_tensor_rows", activation_rows),
            ("activation_capacity_basis", "declared_unmasked_tensor_capacity")))
    if caller is not None:
        activation = caller.tag_activation(activation, local_rows)
    act_ops[-1] = activation
    return act_ops, activation


def _run_routed_experts(
    ops, last, *, prefix, plan, stage, requests, model, parallel, mode, runtime,
    local_rows, active, expert_width, latent, local_experts, local_ep_rank,
    routing_realization, routing_metadata, balanced, transport, transport_scale,
    transport_payload, transport_quantized, dispatch_quant_meta, absorbed_routing,
    dispatch_overlap_id, dispatch_source, gate_row_cut, exchange_options, token_partition,
    caller, reduce_after_join,
):
    if not local_rows:
        return last, None
    from mindie_sim.catalog.quant import projection_quant

    layer_quant = projection_quant(model.quant, plan)
    gate_weight = layer_quant.matrix_bytes(2 * expert_width, latent, "routed_expert") * active
    down_weight = layer_quant.matrix_bytes(latent, expert_width, "routed_expert") * active
    common = dict(stage=stage, layer=plan.index, requests=requests,
                  quant=layer_quant, category="routed_expert", phase=mode, m=math.ceil(local_rows))
    expert_input_deps = (last,)
    if transport is not None and parallel.ep > 1:
        # Both received quantities are real inputs. The payload already
        # waits for scale completion, but the gate retains the direct edge.
        expert_input_deps = tuple(dict.fromkeys(
            dependency for dependency in
            (transport_scale, transport_payload, transport_quantized.op_id)
            if dependency))
    gate = gemm_op(
        op_id=prefix + ".expert_up", deps=expert_input_deps, n=2 * expert_width, k=latent,
        family="grouped_gemm", recipe="grouped_gate_up", weight_bytes_override=gate_weight, **common,
    )
    down = gemm_op(
        op_id=prefix + ".expert_down", deps=(prefix + ".expert_act",), n=latent, k=expert_width,
        family="grouped_gemm", recipe="grouped_down", weight_bytes_override=down_weight, **common,
    )
    # Probability moments stay fractional; explicit realization counts stay
    # integer. Both paths price active experts without reading empty banks.
    routed_kwargs = dict(
        local_rows=local_rows, active_experts=active, local_experts=local_experts,
        local_ep_rank=local_ep_rank, routing_realization=routing_realization,
        routing_metadata=routing_metadata, balanced=balanced,
    )
    gate = routed_expert_work(gate, n=2 * expert_width, k=latent,
                              weights=gate_weight, weight_kind="gate_up", **routed_kwargs)
    down = routed_expert_work(down, n=latent, k=expert_width,
                              weights=down_weight, weight_kind="down", **routed_kwargs)
    if dispatch_quant_meta and transport is None:
        absorbed_meta = ((("activation_scale_producer", absorbed_routing.op_id),
                          ("activation_quantization_kernel", "aclnnMoeInitRoutingV3"))
                         if absorbed_routing is not None else ())
        gate = replace(gate, metadata=gate.metadata + (
            ("activation_producer", "explicit_transport_quantization"),
            ("activation_payload_producer", last),
            *absorbed_meta,
        ))
    if transport is not None:
        gate = bind_transport_gate(
            gate, down, transport=transport, latent=latent,
            local_rows=local_rows,
            payload=transport_payload or transport_quantized.op_id,
            scale=transport_scale or transport_quantized.op_id)
    allow_up_split = not bool((getattr(runtime, "kernels", None) or {}).get("lowerings"))
    activation_rows = (
        token_partition.activation_tensor_rows
        if token_partition is not None and token_partition.activation_tensor_rows
        else (local_rows if caller is None else caller.activation_rows()))
    gate_ops, gate_last = split_deepep_up_overlap(
        gate, exchange_options, dispatch_overlap_id=dispatch_overlap_id,
        allow_split=allow_up_split, active_rows=activation_rows,
        source_contract=dispatch_source,
        row_cut=gate_row_cut, activation_id=prefix + ".expert_act")
    gate_segments = tuple(op for op in gate_ops if op.family == "grouped_gemm")
    act_deps = tuple(op.op_id for op in gate_segments) or (gate_last,)
    act_ops, activation = _emit_routed_activation(
        plan=plan, runtime=runtime, layer_quant=layer_quant,
        activation_rows=activation_rows, expert_width=expert_width,
        prefix=prefix, act_deps=act_deps, stage=stage, requests=requests,
        model=model, mode=mode, gate_segments=gate_segments,
        token_partition=token_partition, caller=caller, local_rows=local_rows,
    )
    ops.extend((*gate_ops, *act_ops, down))
    declare_gate_up_fusion_group(
        ops=ops, runtime=runtime, prefix=prefix, activation=activation,
        gate_ops=gate_ops, gate=gate, down=ops[-1],
        local_experts=local_experts, latent=latent,
        expert_width=expert_width, routing_metadata=routing_metadata,
        model=model)
    down_ops, down_mid_id, last = split_deepep_down_overlap(
        ops.pop(), exchange_options)
    ops.extend(down_ops)
    if parallel.etp > 1 and not reduce_after_join:
        reduce = replace(
            comm_op(
                op_id=prefix + ".expert_reduce", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="allreduce", phase=mode, recipe="expert_tensor_reduce",
                link_bytes=2 * (parallel.etp - 1) / parallel.etp * local_rows * latent
                           * dtype_bytes(model.activation_dtype),
                shape=(math.ceil(local_rows), latent),
                metadata=(("group_axis", "etp"), ("comm_axis", "etp"),
                          ("comm_group_size", int(parallel.etp))),
            ),
            dtype=model.activation_dtype,
        )
        ops.append(reduce)
        last = reduce.op_id
        down_mid_id = reduce.op_id
    return last, down_mid_id


def _combine_routed_outputs(
    ops, packing, last, *, prefix, plan, stage, requests, model, parallel, mode,
    exchange, transport, transport_rows, transport_payload, latent, hidden, tokens,
    partitions, combine_exchange_bytes, collective_meta, combine_quant_meta,
    source_traffic, combine_row_bytes, exchange_factor, exchange_options,
    down_mid_id, dispatch_overlap_id, weights_producer, weight_dtype, weight_bytes,
    routing_metadata, routing_realization, local_rows, absorbed_routing, caller,
    token_partition, full_tokens,
):
    if parallel.ep > 1 and exchange:
        if transport is not None:
            combine = moe_transport.combine(
                prefix=prefix, plan=plan, stage=stage, requests=requests,
                deps=last, transport=transport, rows=transport_rows, width=latent,
                ep=parallel.ep, partitions=partitions,
                mode=mode, payload_dispatch=transport_payload)
        else:
            combine = replace(
                comm_op(
                    op_id=prefix + ".combine", deps=(last,), stage=stage, layer=plan.index,
                    requests=requests, family="alltoall", phase=mode, recipe="expert_combine",
                    link_bytes=combine_exchange_bytes, shape=(tokens * plan.top_k, latent),
                    metadata=collective_meta + (("collective_id", prefix + ".combine"),
                                               ("reverse_of_collective", prefix + ".dispatch")) + combine_quant_meta,
                ),
                dtype=model.activation_dtype,
            )
        if transport is None:
            combine = _masked_collective(
                combine, source_traffic, combine_row_bytes, parallel.ep, reverse=True)
        combine = scale_exchange_operator(
            combine, exchange_factor * exchange_options["eplb"],
            exchange_options, ep=parallel.ep, top_k=plan.top_k, is_combine=True)
        overlap_deps = tuple(dict.fromkeys(
            dep for dep in (down_mid_id, dispatch_overlap_id) if dep
        ))
        combine_ops, last = split_deepep_combine_overlap(
            combine, exchange_options, overlap_deps=overlap_deps,
            main_deps=(last,), dispatch_overlap_id=dispatch_overlap_id)
        ops.extend(combine_ops)
    unpack_inputs = tuple(dict.fromkeys(
        (last, packing.op_id) + ((weights_producer,) if weights_producer is not None else ())))
    unpack_metadata = [
        ("payload_producer", last), ("inverse_index_producer", packing.op_id),
        ("topk_weights_producer", weights_producer), ("input_layout", "expert_grouped"),
        ("topk_ids_dtype", "int32"),
        ("router_weight_dtype", weight_dtype if weights_producer is not None else None),
        ("has_expert_map", False), ("apply_router_weight_on_input", False),
        ("ep_degree", parallel.ep), *routing_metadata,
    ]
    if routing_realization is not None:
        unpack_metadata.append(("input_rows", int(local_rows)))
    act_bytes = dtype_bytes(model.activation_dtype)
    unpack_read = tokens * plan.top_k * (latent * act_bytes + weight_bytes)
    if absorbed_routing is not None:
        # The native pack wrote one INT32 inverse slot per assignment.
        inverse_index_bytes = tokens * plan.top_k * dtype_bytes(INVERSE_INDEX_DTYPE)
        unpack_read += inverse_index_bytes
        unpack_metadata.extend((("inverse_index_dtype", INVERSE_INDEX_DTYPE),
                                ("inverse_index_bytes", float(inverse_index_bytes))))
    unpack_write = tokens * latent * act_bytes
    unpack_ops = unpack_extra = None
    if caller is not None:
        unpack_read, unpack_write, unpack_ops = caller.unpack_work(
            tokens=tokens, latent=latent, local_rows=local_rows,
            weight_bytes=weight_bytes, act_bytes=act_bytes)
        unpack_extra = caller.unpack_metadata(local_rows, weight_dtype)
    unpermute = vector_op(
        op_id=prefix + ".unpack", deps=unpack_inputs, stage=stage, layer=plan.index,
        requests=requests, elements=tokens * latent, ops_per_element=2 * plan.top_k,
        vector_ops_override=unpack_ops,
        dtype=model.activation_dtype, family="routing", phase=mode, recipe="route_unpack",
        read_bytes_override=unpack_read, write_bytes_override=unpack_write,
        shape=(tokens, plan.top_k, latent),
        metadata=tuple(unpack_metadata) + (unpack_extra or ()),
    )
    ops.append(unpermute)
    last = unpermute.op_id
    if latent != hidden and tokens:
        if plan.expert_latent_norm_position == "post":
            latent_norm = _latent_norm(
                prefix, last, stage, plan, requests, tokens, latent, model, mode)
            ops.append(latent_norm)
            last = latent_norm.op_id
        from mindie_sim.catalog.quant import projection_quant

        projection = gemm_op(
            op_id=prefix + ".latent_up", deps=(last,), stage=stage, layer=plan.index,
            requests=requests, m=tokens, n=hidden, k=latent,
            quant=projection_quant(model.quant, plan),
            category="latent_moe", phase=mode, family="projection", recipe="latent_projection",
        )
        ops.append(projection)
        last = projection.op_id
    routed_tokens = tokens
    if token_partition:
        gathered = token_partition.finalize(
            prefix=prefix, source=last, width=hidden, stage=stage, layer=plan.index,
            requests=requests, phase=mode, dtype=model.activation_dtype)
        ops.extend(gathered)
        last, tokens = gathered[-1].op_id, full_tokens
    return last, tokens, routed_tokens


def _join_routed_and_shared(
    ops, last, *, prefix, plan, stage, requests, model, parallel, mode, runtime,
    deps, tokens, hidden, sequence_plan, overlap_shared, reduce_after_join, caller,
):
    if plan.num_shared_experts and tokens:
        shared_dp = bool(dict(runtime.metadata).get("enable_shared_expert_dp", False)) if runtime is not None else False
        shared_input, shared_tokens, shared_prefix = deps, tokens, []
        if sequence_plan and not shared_dp:
            from .sharding.sequence_parallel import (
                allgather, emit_reduce_scatter_pad, reduce_scatter,
            )
            gathered = allgather(
                plan=sequence_plan, op_id=prefix + ".shared.sp_gather", deps=(deps,),
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                width=hidden, dtype=model.activation_dtype)
            shared_prefix.append(gathered)
            shared_tokens = (
                sequence_plan.valid_global
                if (sequence_plan.padding == "right_to_tp"
                    and 0 < sequence_plan.valid_global < sequence_plan.tokens_global)
                else sequence_plan.tokens_global
            )
            shared_input = gathered.op_id
        shared, shared_last = dense(
            prefix=prefix + ".shared", plan=replace(
                plan,
                intermediate_size=(plan.shared_intermediate_size
                                   or plan.num_shared_experts * plan.moe_intermediate_size),
                # The shared expert declares its own clamp scalar; an absent
                # shared limit inherits the routed-expert declaration.
                swiglu_limit=(plan.swiglu_shared_limit or plan.swiglu_limit)),
            deps=shared_input, stage=stage, requests=requests, tokens=tokens if shared_dp else shared_tokens, model=model,
            parallel=replace(parallel, tp=1) if shared_dp else parallel, mode=mode, category="shared_expert",
            runtime=runtime, reduce_output=not reduce_after_join and sequence_plan is None,
        )
        shared = shared_prefix + shared
        if sequence_plan and not shared_dp:
            scatter_deps = (shared_last,)
            shared_pad = emit_reduce_scatter_pad(
                plan=sequence_plan, op_id=prefix + ".shared.sp_scatter.pad",
                deps=scatter_deps, stage=stage, layer=plan.index,
                requests=requests, phase=mode, width=hidden,
                dtype=model.activation_dtype, source_rows=shared_tokens,
                target_rows=sequence_plan.tokens_global,
            )
            if shared_pad is not None:
                shared.append(shared_pad)
                scatter_deps = (shared_pad.op_id,)
            scattered = reduce_scatter(
                plan=sequence_plan, op_id=prefix + ".shared.sp_scatter", deps=scatter_deps,
                stage=stage, layer=plan.index, requests=requests, phase=mode,
                width=hidden, dtype=model.activation_dtype)
            shared.append(scattered)
            shared_last = scattered.op_id
        if plan.shared_expert_gate:
            from mindie_sim.catalog.quant import projection_quant

            gate = gemm_op(
                op_id=prefix + ".shared_gate", deps=(deps,), stage=stage, layer=plan.index,
                requests=requests, m=tokens, n=1, k=hidden,
                quant=projection_quant(model.quant, plan),
                category="router", phase=mode, recipe="shared_gate", family="router")
            applied = vector_op(
                op_id=prefix + ".shared_gate_apply", deps=(shared_last, gate.op_id),
                stage=stage, layer=plan.index, requests=requests, elements=tokens * hidden,
                ops_per_element=2, dtype=model.activation_dtype, family="elementwise",
                phase=mode, recipe="shared_gate_apply")
            shared.extend((gate, applied))
            shared_last = applied.op_id
        if overlap_shared:
            shared = [replace(op, stream="shared_stream") for op in shared]
        ops.extend(shared)
        join = vector_op(
            op_id=prefix + ".join", deps=(last, shared_last), stage=stage, layer=plan.index,
            requests=requests, elements=tokens * hidden, ops_per_element=1,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="expert_add",
            shape=(tokens, hidden),
            read_bytes_override=2 * tokens * hidden * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * hidden * dtype_bytes(model.activation_dtype),
            metadata=(("input_tensor_count", 2),),
        )
        ops.append(join)
        last = join.op_id
    if reduce_after_join and parallel.tp > 1 and tokens:
        reduction = replace(
            comm_op(
                op_id=prefix + ".moe_output_reduce", deps=(last,), stage=stage, layer=plan.index,
                requests=requests, family="allreduce", phase=mode, recipe="moe_output_reduce",
                link_bytes=2 * (parallel.tp - 1) / parallel.tp * tokens * hidden
                           * dtype_bytes(model.activation_dtype),
                shape=(tokens, hidden),
                metadata=(("group_axis", "tp"), ("steps", 2 * (parallel.tp - 1)),
                          ("reduction_boundary", "token_rows_after_routed_shared_join")),
            ),
            dtype=model.activation_dtype,
        )
        if caller is not None:
            reduction = caller.tag_reduce(reduction)
        ops.append(reduction)
        last = reduction.op_id
    return last


def alltoall(*, prefix, plan, deps, stage, requests, tokens, model, parallel, mode,
        global_tokens=None, routing=RoutingPolicy(), routing_realization=None,
        overlap_shared=False, runtime=None, reduce_after_join=False, sequence_plan=None,
        caller=None, sp_tokens=None, features=None,
        source_rows_by_replica=None, destination_rows_by_rank=None,
        ep_owner=None, price_owner=None, **_):
    validate_m10_moe_activation_legality(plan)
    pre_sp_tokens = tokens
    if sp_tokens is not None:
        tokens = sp_tokens
    exchange = caller is None or caller.exchange
    reduce_after_join = reduce_after_join or (caller is not None and caller.reduce_after_join)
    hidden = model.hidden_size
    token_partition = moe_token_partition.select(
        runtime, parallel=parallel, plan=plan, model=model, tokens=tokens,
        sequence_plan=sequence_plan, reduce_after_join=reduce_after_join)
    full_tokens = tokens
    latent = plan.expert_latent_size or hidden
    expert_width = math.ceil(plan.moe_intermediate_size / parallel.etp)
    transport = moe_transport.select(runtime, layer_scope=plan.scope)
    if transport is not None:
        if getattr(runtime, "moe_recipe", "alltoall") not in {"alltoall", "tp_join_reduce"}:
            raise ModelingError("moe_transport_requires_alltoall_recipe")
        # Reject incompatible expert representations before building any graph.
        moe_transport.validate_contract(model, plan, transport)
    overlap_shared, runtime = _resolve_alltoall_moe_route(
        model, plan, parallel, runtime, overlap_shared=bool(overlap_shared),
    )
    group_tokens = global_tokens if global_tokens is not None else pre_sp_tokens
    (routing_realization, local_rows, active, routing_metadata, partitions,
     local_experts, local_ep_rank, receive_rows, runtime_routing, has_slots,
     balanced, destination_rows) = _route_expert_rows(
        plan, model, parallel, runtime, routing, routing_realization,
        caller=caller, token_partition=token_partition, sequence_plan=sequence_plan,
        group_tokens=group_tokens, ep_owner=ep_owner, price_owner=price_owner,
    )
    (ops, packing, last, tokens, weight_dtype, weight_bytes, weights_producer,
     exchange_options, exchange_factor, source_traffic, payload_tokens,
     transport_rows, transport_histogram_gather) = _pack_routed_inputs(
        prefix=prefix, plan=plan, stage=stage, requests=requests,
        model=model, parallel=parallel, mode=mode, runtime=runtime,
        deps=deps, tokens=tokens, full_tokens=full_tokens, hidden=hidden, latent=latent,
        transport=transport, token_partition=token_partition, sequence_plan=sequence_plan,
        routing_realization=routing_realization, local_rows=local_rows, partitions=partitions,
        caller=caller, routing_metadata=routing_metadata, ep_owner=ep_owner,
    )
    mc2_mask_pad_id = None
    # Only emit mc2_mask_pad when an EP dispatch/transport consumer will actually receive it.
    if (parallel.ep > 1 and (transport is not None or exchange)
            and sequence_plan is not None and sequence_plan.mc2_active_mask_active):
        from .sharding.sequence_parallel import emit_mc2_mask_pad
        mask_pad = emit_mc2_mask_pad(
            plan=sequence_plan, op_id=prefix + ".mc2_mask_pad", deps=(deps,),
            stage=stage, layer=plan.index, requests=requests, phase=mode,
        )
        if mask_pad is not None:
            ops.append(mask_pad)
            mc2_mask_pad_id = mask_pad.op_id
    (packing, last, transport_scale, transport_payload, transport_quantized,
     dispatch_overlap_id, dispatch_source, gate_row_cut, absorbed_routing,
     dispatch_quant_meta, combine_quant_meta, combine_row_bytes,
     combine_exchange_bytes, collective_meta, _mask_pad_consumed) = _dispatch_routed_inputs(
        ops, packing, last, prefix=prefix, plan=plan, stage=stage, requests=requests,
        model=model, parallel=parallel, mode=mode, runtime=runtime,
        transport=transport, transport_rows=transport_rows,
        transport_histogram_gather=transport_histogram_gather, latent=latent,
        tokens=tokens, payload_tokens=payload_tokens, partitions=partitions,
        routing_realization=routing_realization, source_traffic=source_traffic,
        exchange=exchange, exchange_options=exchange_options, exchange_factor=exchange_factor,
        local_rows=local_rows, receive_rows=receive_rows,
        mc2_mask_pad_id=mc2_mask_pad_id,
    )
    last, down_mid_id = _run_routed_experts(
        ops, last, prefix=prefix, plan=plan, stage=stage, requests=requests,
        model=model, parallel=parallel, mode=mode, runtime=runtime,
        local_rows=local_rows, active=active, expert_width=expert_width, latent=latent,
        local_experts=local_experts, local_ep_rank=local_ep_rank,
        routing_realization=routing_realization, routing_metadata=routing_metadata,
        balanced=balanced, transport=transport, transport_scale=transport_scale,
        transport_payload=transport_payload, transport_quantized=transport_quantized,
        dispatch_quant_meta=dispatch_quant_meta, absorbed_routing=absorbed_routing,
        dispatch_overlap_id=dispatch_overlap_id, dispatch_source=dispatch_source,
        gate_row_cut=gate_row_cut, exchange_options=exchange_options,
        token_partition=token_partition, caller=caller, reduce_after_join=reduce_after_join,
    )
    last, tokens, routed_tokens = _combine_routed_outputs(
        ops, packing, last, prefix=prefix, plan=plan, stage=stage, requests=requests,
        model=model, parallel=parallel, mode=mode, exchange=exchange, transport=transport,
        transport_rows=transport_rows, transport_payload=transport_payload, latent=latent,
        hidden=hidden, tokens=tokens, partitions=partitions,
        combine_exchange_bytes=combine_exchange_bytes, collective_meta=collective_meta,
        combine_quant_meta=combine_quant_meta, source_traffic=source_traffic,
        combine_row_bytes=combine_row_bytes, exchange_factor=exchange_factor,
        exchange_options=exchange_options, down_mid_id=down_mid_id,
        dispatch_overlap_id=dispatch_overlap_id, weights_producer=weights_producer,
        weight_dtype=weight_dtype, weight_bytes=weight_bytes, routing_metadata=routing_metadata,
        routing_realization=routing_realization, local_rows=local_rows,
        absorbed_routing=absorbed_routing, caller=caller, token_partition=token_partition,
        full_tokens=full_tokens,
    )
    last = _join_routed_and_shared(
        ops, last, prefix=prefix, plan=plan, stage=stage, requests=requests,
        model=model, parallel=parallel, mode=mode, runtime=runtime,
        deps=deps, tokens=tokens, hidden=hidden, sequence_plan=sequence_plan,
        overlap_shared=overlap_shared, reduce_after_join=reduce_after_join, caller=caller,
    )
    if (has_slots and runtime is not None and int(plan.num_experts) > 0
            and routed_tokens > 0):
        # Redundant physical copies change residency and active weight reads,
        # never logical assignments or aggregate FLOPs. Bind the exact
        # per-owner slot bytes after routing and before native graph lowering.
        ops = bind_expert_slots(
            ops, model=model, plan=plan, parallel=parallel, routing=routing,
            runtime_routing=runtime_routing, routing_realization=routing_realization,
            group_tokens=group_tokens, expert_width=expert_width, latent=latent)
    if overlap_shared:
        ops = reorder_overlapped_shared(ops)
    from mindie_sim.prediction.operators.mc2.mc2_graph import lower as lower_native_mc2
    ops = lower_native_mc2(
        ops, runtime=runtime, prefix=prefix, tokens=routed_tokens, top_k=plan.top_k,
        width=latent, experts=plan.num_experts, ep=parallel.ep,
        source_partitions=partitions, received_assignments=local_rows,
        dtype=model.activation_dtype, source_routing=source_traffic,
        valid_tokens=payload_tokens, features=features,
        source_rows_by_replica=source_rows_by_replica,
        destination_rows_by_rank=destination_rows,
        phase=mode, tp_scale=(sequence_plan.tp if sequence_plan is not None else 1))
    ops, last = _absorb_redundant_mc2_unpack(
        ops, last, prefix=prefix, runtime=runtime, parallel=parallel,
        transport=transport, caller=caller)
    return ops, last


def _combine_bound_to_mc2(combine, combine_meta: dict) -> bool:
    """True when a combine op is explicitly bound to ``MoeDistributeCombine*``."""
    group = combine_meta.get("device_kernel_group")
    group_kernel = str(group.get("kernel") or "") if isinstance(group, dict) else ""
    binding = getattr(combine, "kernel_binding", None)
    bind_kernel = (
        str(binding.get("kernel_id") or "")
        if isinstance(binding, dict)
        else str(getattr(binding, "kernel_id", "") or "")
    )
    candidates = (
        str(combine_meta.get("native_kind") or ""),
        str(combine_meta.get("native_kernel") or ""),
        group_kernel,
        bind_kernel,
    )
    return any("MoeDistributeCombine" in cand for cand in candidates)


def _absorb_redundant_mc2_unpack(ops, last, *, prefix, runtime, parallel, transport, caller):
    """Omit trailing route_unpack when MC2 combine returns weighted unpermuted output."""
    if parallel.ep <= 1 or transport is not None:
        return ops, last
    by_id = {op.op_id: op for op in ops}
    combine = by_id.get(prefix + ".combine")
    unpack = by_id.get(prefix + ".unpack")
    if combine is None or unpack is None or unpack.recipe != "route_unpack":
        return ops, last
    meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    comm_mode = str(meta.get("moe_comm_mode") or "").strip().lower()
    combine_meta = dict(combine.metadata)
    is_mc2_combine = (
        _combine_bound_to_mc2(combine, combine_meta)
        or (
            caller is None
            and (
                getattr(runtime, "moe_recipe", "") == "mc2"
                or comm_mode == "mc2"
            )
        )
    )
    if not is_mc2_combine:
        return ops, last
    pack_id = prefix + ".pack"
    extra_deps = tuple(d for d in unpack.deps if d not in {combine.op_id, pack_id})
    merged_deps = tuple(dict.fromkeys((*combine.deps, *extra_deps)))
    combine_meta["absorbs_route_unpack"] = True
    merged_combine = replace(combine, deps=merged_deps, metadata=tuple(combine_meta.items()))
    rebuilt = []
    for op in ops:
        if op.op_id == unpack.op_id:
            continue
        if op.op_id == combine.op_id:
            rebuilt.append(merged_combine)
            continue
        if unpack.op_id in op.deps:
            rebuilt.append(replace(
                op, deps=tuple(combine.op_id if d == unpack.op_id else d for d in op.deps)))
        else:
            rebuilt.append(op)
    return rebuilt, (combine.op_id if last == unpack.op_id else last)


def source_issuance(*, routing_realization, token_partition, sequence_plan,
                    plan, tokens, full_tokens, ep_owner=None):
    """Declared source-traffic view and the rows this owner really issues."""
    source_traffic = (
        routing_realization if getattr(routing_realization, "basis", "")
        in {"source_force_eplb_masked_integer_counts",
            "explicit_token_routing_integer"} else
        token_partition.expected_traffic(plan.top_k)
        if token_partition and token_partition.allocated_rows != full_tokens else None)
    if sequence_plan is not None and sum(sequence_plan.valid_per_tp) == 0:
        # This representative owns physical TP/EP participants but no valid
        # source row.  Keep their control/dummy participation; suppress only
        # the payload cell in the ordinary source ledger.
        issued = 0
    elif token_partition is not None:
        # An explicit TP caller owns this slice's actual outgoing rows.
        issued = token_partition.valid_per_rank[0]
    elif source_traffic is not None:
        # A source-mask / exact token ledger owns its valid outgoing rows.
        issued = source_traffic.valid_local_tokens
    elif (sequence_plan is not None
          and sequence_plan.padding == "right_to_tp"
          and routing_realization is None
          and sequence_plan.valid_per_tp
          and 0 < sequence_plan.valid_global < sequence_plan.tokens_global):
        local_rank = (int(ep_owner) % sequence_plan.tp) if ep_owner is not None else 0
        issued = sequence_plan.valid_per_tp[local_rank]
    else:
        # Ordinary balanced and probability plans route the physical graph
        # rows, including full-capture padding.  ``valid_local_tokens`` is a
        # separate logical-valid statistic and must not shrink that plan.
        issued = tokens
    return source_traffic, issued


def reorder_overlapped_shared(ops):
    """Place the shared-expert stream before the first dispatch collective."""
    shared_ops = [op for op in ops if op.stream == "shared_stream"]
    if not shared_ops:
        return ops
    routed = [op for op in ops if op.stream != "shared_stream"]
    dispatch_index = next(
        (index for index, op in enumerate(routed)
         if op.recipe == "expert_dispatch" or op.family == "alltoall"),
        len(routed))
    return routed[:dispatch_index] + shared_ops + routed[dispatch_index:]


DEFAULT_MC2_CHUNKS = 4


def chunked_mc2(*, prefix, tokens, global_tokens=None, chunks=DEFAULT_MC2_CHUNKS, **kwargs):
    """Finite dispatch/compute/combine chunks with separate ordered streams."""
    group = tokens if global_tokens is None else global_tokens
    count = min(chunks, max(1, group))
    if global_tokens is not None and (tokens % count or group % count):
        # Dividing the cohort total and then dividing each source are not
        # interchangeable at chunk tails. No source partition was supplied.
        raise ModelingError("mc2_cross_dp_tails_require_per_source_chunk_partitions")
    ops, completions = [], []
    for index in range(count):
        local = tokens // count + int(index < tokens % count)
        global_local = group // count + int(index < group % count)
        part, last = alltoall(prefix=f"{prefix}.chunk{index}", tokens=local,
                              global_tokens=global_local, **kwargs)
        phase = "prepare"
        for op in part:
            if op.recipe == "expert_dispatch":
                phase = "dispatch"
            elif op.family == "grouped_gemm":
                phase = "experts"
            elif op.recipe == "expert_combine":
                phase = "combine"
            elif op.recipe == "route_unpack":
                phase = "finish"
            target_stream = (op.stream if op.stream == "shared_stream"
                             else f"mc2:{phase}")
            ops.append(replace(op, stream=target_stream))
        completions.append(last)
    joined = barrier_op(
        op_id=prefix + ".chunks_complete", deps=tuple(completions),
        stage=kwargs["stage"], layer=kwargs["plan"].index,
        requests=kwargs["requests"], phase=kwargs["mode"], recipe="mc2_join",
        shape=(tokens, kwargs["model"].hidden_size))
    ops.append(joined)
    return ops, joined.op_id


MOE_RECIPES = {"alltoall": alltoall, "mc2": chunked_mc2, "tp_join_reduce": tp_join_reduce,
               "local_expert_reduce": local_expert_reduce_recipe}


def moe(*, moe_recipe="alltoall", mc2_chunks=4, **kwargs):
    return MOE_RECIPES[moe_recipe](chunks=mc2_chunks, **kwargs)


FFN_BUILDERS = {"dense": dense, "moe": moe}
