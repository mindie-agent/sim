"""Fused MoE router selection and init-routing coordination.

[Role]
  Overlay the declared router selection boundary with the single CANN 9.2.1
  gating call when an Ascend runtime owns the graph, and fold an EP dispatch
  quantizer into ``aclnnMoeInitRoutingV3``.  The raw per-operator chain stays in
  ``rewrite_ffn_rules`` and is still emitted for ``raw_eager`` or a direct
  ``LayerPlan`` declaration.  The fused op keeps the logical member work; a
  registered ``PlannedRecipe`` supplies the A5 64-AIV row program and is
  selected only from declared geometry.

[Invariants]
  * The fused call conserves every declared Vector Op of the raw chain and
    reads the ``[tokens, experts]`` logits exactly once (plus the FP32
    ``noaux_tc`` correction bias) and writes only ``[tokens, top_k]`` ids and
    weights; no score, mask or gather round-trip survives.
  * The structure contract keeps the observed torch operator kind
    (``MoeGatingTopK`` / ``MoeGatingTopKHash``); the aclnn symbol is declared as
    the operator's ``native_kind`` and on ``implementation``.
  * An absorbed init-routing quantizer is billed once on the pack, writes the
    quantized payload with its scales and leaves no separate quantize node.
  * The A5 row plan is byte-for-byte conserving and A2/A3 keeps its Class D
    vector recipe; no token-count threshold selects a physical mechanism.

[Timeouts & Locks]
  Pure bounded arithmetic over one layer; no I/O, locks or device work.

[Side-Effect Policy]
  Emits immutable operator records only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.catalog.ops import vector_op
from mindie_sim.primitives.loader import resolve_dispatch_envelope
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec
from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import (
    KernelBinding,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="moe/moe_gating_top_k/op_host/moe_gating_top_k_def.cpp",
        source_sha256="c7e347cbdf8fd98dd42fd9ea475dc37e94e585842d842e2daf7ebd0af5fc950a",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="moe/moe_gating_top_k_softmax_v2/op_host/moe_gating_top_k_softmax_v2_def.cpp",
        source_sha256="0f6f80730e8fafc54869ed3d3345722ab93481995a23a388ae335ebea033b3ef",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision="48189acf95882e23fae56ea96ceeb89358cb42da",
        relative_path="op_plugin/ops/opapi/MoeGatingTopKKernelNpuOpApi.cpp",
        source_sha256="7853371fd070751cb29fd495ed4446140de45843cf660b012f6f48e59443294d",
    ),
)

from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.sharding.ffn_rules import (
    CONTRACT_HASH, CONTRACT_SCORE, HASH_ROW_ID_BYTES, HASH_SELECTION_MODE,
    NOAUX_BIAS_BYTES, ROUTED_SCALE_RECIPE, ROUTE_NORMALIZE_RECIPE,
    SCORE_SELECTION_MODE,
    SELECTED_ID_BYTES, declare_router_selection, raw_selection,
)

HASH_NATIVE_KIND = "aclnnMoeGatingTopKHash"
TOPK_NATIVE_KIND = "aclnnMoeGatingTopK"
SOFTMAX_NATIVE_KIND = "aclnnMoeGatingTopKSoftmaxV2"
INIT_ROUTING_NATIVE_KIND = "aclnnMoeInitRoutingV3"
NORM_TYPE_SOFTMAX = 0
NORM_TYPE_SIGMOID = 1
NORM_TYPE_SQRTSOFTPLUS = 2


def _native_gating_dispatch(plan, *, hash_table: bool, noaux_tc: bool) -> tuple[str, int]:
    """Select CANN fused gating symbol and norm_type from the declared score mode."""
    raw_mode = getattr(plan, "router_mode", "")
    mode = str(raw_mode or "").strip().lower()
    if hash_table or mode == "hash":
        return HASH_NATIVE_KIND, NORM_TYPE_SOFTMAX
    if mode == "sqrtsoftplus":
        return HASH_NATIVE_KIND, NORM_TYPE_SQRTSOFTPLUS
    if mode == "sigmoid" or noaux_tc:
        return TOPK_NATIVE_KIND, NORM_TYPE_SIGMOID
    if not mode or mode == "softmax":
        return SOFTMAX_NATIVE_KIND, NORM_TYPE_SOFTMAX
    raise ModelingError(f"moe_gating_router_mode_unsupported:{mode}")
#: One FP32 selected weight per routed slot beside its INT32 id.
SELECTED_WEIGHT_BYTES = 4
#: ``quant_mode=1`` carries MXFP8 (group-32 E8M0) or dynamic per-token INT8
#: (one FP32 row scale); a coarser block-scale FP8 row keeps its own quantizer.
MX_SCALE_GROUP = 32
_MX_PAYLOADS = frozenset({"fp8_e4m3", "fp8_e5m2"})
_E8M0_SCALES = frozenset({"fp8_e8m0", "e8m0"})
_PER_ROW_SCALES = frozenset({"fp32", "float32"})
_ROUTING_RECIPES = frozenset({
    "router_topk", "router_group_topk", "route_normalize", "route_pack",
    "hash_routing", "router_sigmoid_scores", "router_group_score_top2",
    "router_group_mask", "router_expert_topk", "router_weight_gather",
    "router_routed_scale", "router_correction_bias", "moe_gating_topk",
    INIT_ROUTING_NATIVE_KIND,
})
_ROUTING_FAMILIES = frozenset({"topk", "routing", "moe_gating"})
#: The declared runtime kernel entry that installs the CANN gating overlay.  A
#: documented production profile binds it; an undeclared runtime keeps the
#: per-operator decomposition so an unverified launch reduction is never
#: fabricated.  It is a plain kernel declaration (like ``native_moe_protocol``),
#: not a lowering binding that would need a registered implementation.
GATING_SELECTOR = "moe_gating"
GATING_IMPLEMENTATION = "vllm_ascend_cann_gating"
_GATING_FIELDS = frozenset({"implementation", "enabled", "native_kind", "source"})
#: ``aclnnMoeGatingTopK`` owns one token per AIV across ``ceil(T / AIV)`` waves.
#: Stage durations are priced strictly from calibrated ``PrimitiveTable`` rates
#: (``P07_vec_alu``, ``P08_vec_sfu``, ``P09_vec_reduce_cast``, ``P10_sort_topk``,
#: ``P12_intracore_flag``, ``P15_rts_stars``) with zero hardcoded cycle floors.
GATING_AIV_UNITS = 64
GATING_MIN_GROUPS = 1
GATING_PIPELINE_SLOTS = 1
GATING_SOFTMAX_WAVE_STAGES = (
    "logits_load", "softmax_reduce", "expert_topk", "renorm_store",
)
GATING_SIGMOID_WAVE_STAGES = (
    "logits_load", "sigmoid_score", "expert_topk", "renorm_store",
)
GATING_WAVE_STAGES = (
    "logits_bias_load", "sigmoid_bias", "group_top2_scan", "group_mask",
    "expert_topk", "renorm_store",
)
MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000.0
_ARCH35_IDS = frozenset({"npu_arch_3510", "dav_3510"})
_SIGMOID_SFU_OPS_PER_EXPERT = 4
_SOFTMAX_SFU_OPS_PER_EXPERT = 5
_RENORM_OPS_PER_SLOT = 4
#: Arch35 ``moe_gating_top_k_tiling_arch35.cpp`` UB queue and host sizing constants.
_GATING_ARCH35_STATIC_EXPERT_BUFFERS = 18
_GATING_ARCH35_RESERVE_EXPERT_BUFFERS = 17
_GATING_ARCH35_ROW_EXPERT_BUFFERS = 2
_GATING_ARCH35_ROW_OVERHEAD_BYTES = 128
_GATING_ARCH35_MAX_ROW_BATCH = 128
_GATING_ABI_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def _geometry_arch_family(geometry) -> str:
    dp_arch = getattr(getattr(geometry, "datapath", None), "compute_arch", None)
    ca = dp_arch or getattr(geometry, "compute_arch", None)
    arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
    if arch_id in _ARCH35_IDS:
        return "A5"
    if "2201" in arch_id:
        return "A3"
    if "910b" in arch_id:
        return "A2"
    return ""


def _is_arch35_geometry(geometry) -> bool:
    """True strictly for declared Arch35 (``npu_arch_3510``) compute architectures."""
    return _geometry_arch_family(geometry) == "A5"


def _effective_groups(meta) -> int:
    """Declared router groups, with an ungrouped router treated as one group."""
    return max(GATING_MIN_GROUPS, int(meta.get("groups", 0) or 0))


def _group_limited(op) -> bool:
    meta = dict(op.metadata)
    groups = _effective_groups(meta)
    return (int(meta.get("num_experts", 0) or 0) > 0
            and (groups <= GATING_MIN_GROUPS
                 or int(meta.get("group_topk", 0) or 0) > 0)
            and bool(meta.get("noaux_tc")))


def _is_sigmoid_router(meta) -> bool:
    mode = str(meta.get("score_function") or meta.get("router_mode") or "").strip().lower()
    norm_type = meta.get("norm_type")
    return mode == "sigmoid" or norm_type == NORM_TYPE_SIGMOID


def _general_topk_kind(op) -> str | None:
    """Return ``'softmax'`` or ``'sigmoid'`` for non-noaux_tc row-wave gating, else ``None``."""
    meta = dict(op.metadata)
    if meta.get("selection_mode") == CONTRACT_HASH or meta.get("hash_table"):
        return None
    if (int(meta.get("num_experts", 0) or 0) <= 0
            or int(meta.get("top_k", 0) or 0) <= 0
            or bool(meta.get("noaux_tc"))):
        return None
    return "sigmoid" if _is_sigmoid_router(meta) else "softmax"


def _effective_gating_aiv_budget(geometry, *, is_group: bool) -> tuple[int, tuple[int, ...]]:
    """Return ``(wave_width, physical_aiv_pool)`` honoring ``geometry.stream_owners``."""
    die_aiv = max(1, int((getattr(geometry, "resource_units", {}) or {}).get("vector", GATING_AIV_UNITS)))
    owners = getattr(geometry, "stream_owners", None)
    if owners is not None:
        aiv_limit = int(owners.aiv_limit)
        if aiv_limit < 1:
            raise ModelingError("moe_gating_requires_positive_stream_aiv_budget")
        aiv_units = min(die_aiv, aiv_limit)
        pool = owners.aiv if owners.isolation == "disjoint" and owners.aiv else tuple(range(aiv_units))
        aiv_units = min(aiv_units, len(pool))
    else:
        aiv_units = die_aiv
        pool = tuple(range(die_aiv))
    wave_width = min(GATING_AIV_UNITS, aiv_units) if is_group else aiv_units
    return max(1, wave_width), pool


def _arch35_gating_ub_live_bytes(experts: int, ub_cap: int, geometry) -> int:
    """Compute Arch35 ``MoeGatingTopK`` per-AIV UB reservation and verify pool fit."""
    static_ub = _GATING_ARCH35_STATIC_EXPERT_BUFFERS * experts * SELECTED_WEIGHT_BYTES
    reserve_ub = _GATING_ARCH35_RESERVE_EXPERT_BUFFERS * experts * SELECTED_WEIGHT_BYTES
    row_ub = (
        _GATING_ARCH35_ROW_EXPERT_BUFFERS * experts * SELECTED_WEIGHT_BYTES
        + _GATING_ARCH35_ROW_OVERHEAD_BYTES
    )
    if ub_cap <= 0:
        return static_ub + row_ub
    available = ub_cap - reserve_ub
    if available < row_ub or static_ub + row_ub > ub_cap:
        ub_live = static_ub + row_ub
    else:
        batch = min(_GATING_ARCH35_MAX_ROW_BATCH, available // row_ub)
        while batch > 1 and batch * row_ub + static_ub > ub_cap:
            batch -= 1
        ub_live = batch * row_ub + static_ub
    LiveBufferSpec(
        "moe_gating_ub", "ub", ub_live, 0, 1, owner_kind="aiv",
    ).check_capacity(geometry)
    return ub_live


def _gating_owner_schedule(
    op, geometry, *, tokens: int, waves: int, launched: int, stage_names: tuple[str, ...],
    read: float, bias_bytes: float, write: float, logical_vector: float, aiv_units: int,
) -> OwnerScheduleSpec:
    active = max(1, min(int(launched), int(aiv_units))) if tokens > 0 else 0
    ub_raw = dict(getattr(geometry, "buffers", ())).get("ub")
    ub_cap = int(ub_raw) if ub_raw is not None else 0
    meta = dict(op.metadata)
    experts = max(1, int(meta.get("num_experts", 1) or 1))
    ub_live = _arch35_gating_ub_live_bytes(experts, ub_cap, geometry)
    logits_read = max(0.0, float(read) - float(bias_bytes))
    row_logits = logits_read / float(max(1, tokens))
    row_write = float(write) / float(max(1, tokens))
    busiest_rows = max(1, int(waves)) if tokens > 0 else 0
    per_owner_read = float(busiest_rows) * row_logits + float(bias_bytes)
    per_owner_write = float(busiest_rows) * row_write
    per_stage_vec = logical_vector / float(max(1, active * len(stage_names)))
    phases = []
    for idx, name in enumerate(stage_names):
        if idx == 0:
            phases.append(SerialPhaseSpec(name, "mte2", per_owner_read, "bytes"))
        elif idx == len(stage_names) - 1:
            phases.append(SerialPhaseSpec(name, "mte3", per_owner_write, "bytes"))
        else:
            phases.append(SerialPhaseSpec(name, "vec_reduce", per_stage_vec, "op"))
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="token_row_waves",
        parallel_axis="token_rows",
        work_tiles=max(0, int(tokens)),
        launched_owners=active,
        active_owners=active,
        total_owners=max(1, int(aiv_units)),
        max_tiles_per_owner=busiest_rows,
        residency=OwnerMemoryResidency(
            ub_live_bytes=ub_live,
            ub_capacity_bytes=ub_cap,
            owner_mte2_bytes=per_owner_read,
            owner_mte3_bytes=per_owner_write,
            shared_hbm_read_bytes=read,
            shared_hbm_write_bytes=write,
            l2_admission_mode="reuse",
        ),
        serial_phases=tuple(phases),
        dispatch_envelope_source="P15_dispatch",
    )


def _gating_kernel_binding(op, *, group_limited: bool, geometry) -> KernelBinding:
    meta = dict(op.metadata)
    arch_family = _geometry_arch_family(geometry)
    if not arch_family:
        raise ModelingError("moe_gating_compute_arch_unresolved")
    is_sig = _is_sigmoid_router(meta)
    native = str(meta.get("native_kind") or (
        TOPK_NATIVE_KIND if (group_limited or is_sig) else SOFTMAX_NATIVE_KIND
    ))
    reg = default_abi_registry()
    shipping = reg.is_shipping_available(native, arch_family)
    if native == TOPK_NATIVE_KIND:
        tokens = max(1, int(meta.get("row_tokens", op.local_shape[0])))
        experts = max(1, int(meta.get("num_experts", 1) or 1))
        top_k = max(1, int(meta.get("top_k", 1) or 1))
        raw_in = str(meta.get("input_format") or "fp32").strip().lower()
        in_dtype = _GATING_ABI_DTYPE_ALIASES.get(raw_in, raw_in)
        raw_out = str(meta.get("output_format") or "fp32").strip().lower()
        out_dtype = _GATING_ABI_DTYPE_ALIASES.get(raw_out, raw_out)
        has_bias = bool(meta.get("noaux_tc"))
        bias_dtype = "fp32" if has_bias else in_dtype
        query = AbiQueryKey(
            soc_arch=arch_family,  # type: ignore[arg-type]
            sku_id="ascend",
            abi_id=native,
            graph_motif="moe_gating_top_k",
            model_scope="target",
            draft_quant_mode="none",
            pr_track="shipping",
            operands=(
                OperandPortSig(
                    role="x", dtype=in_dtype, layout="ND", transposed=False,
                    shape=(tokens, experts), quant_mode="dense",
                    group_shape=(), scale_dtype="none", scale_shape=(),
                ),
                OperandPortSig(
                    role="bias", dtype=bias_dtype, layout="ND", transposed=False,
                    shape=(experts,), quant_mode="dense",
                    group_shape=(), scale_dtype="none", scale_shape=(),
                ),
            ),
            outputs=(
                OperandPortSig(
                    role="y", dtype=out_dtype, layout="ND", transposed=False,
                    shape=(tokens, top_k), quant_mode="dense",
                    group_shape=(), scale_dtype="none", scale_shape=(),
                ),
                OperandPortSig(
                    role="expert_idx", dtype="int32", layout="ND", transposed=False,
                    shape=(tokens, top_k), quant_mode="none",
                    group_shape=(), scale_dtype="none", scale_shape=(),
                ),
                OperandPortSig(
                    role="out", dtype="fp32", layout="ND", transposed=False,
                    shape=(tokens, top_k), quant_mode="dense",
                    group_shape=(), scale_dtype="none", scale_shape=(),
                ),
            ),
            rank_placement="local",
            world_size=1,
            comm_group_id="local",
            comm_domain_width=1,
            comm_dtype="none",
            reduction_kind="none",
            activation_kind="none",
            activation_params=(),
            has_bias=has_bias,
            has_active_mask=False,
            dispatch_quant_mode=0,
            shared_experts_per_rank=0,
            top_k=top_k,
            moe_expert_num=experts,
            qk_rope_head_dim=0,
            mla_variant="none",
            sfa_dcp_active=False,
            pcp_size=1,
            kv_cache_dtype="none",
            has_kpool_indexer=False,
            dcp_size=1,
            is_act_and_mul=False,
            enabled_stack_passes=frozenset(),
        )
        verdict = reg.verify(query)
        if verdict.status == "rejected_abi" and verdict.rejection_reason.startswith(
            "rejected_shape_or_align:"
        ):
            raise ModelingError(f"moe_gating_topk_abi_rejected:{verdict.rejection_reason}")
        shipping = verdict.status == "admitted_shipping"
    return KernelBinding(
        kernel_id=native,
        backend="CANN",
        arch_family=arch_family,
        submodule="moe_gating",
        macro_module="moe_routing",
        readiness="shipping" if shipping else "proposed_pr",
        recipe_id="moe_gating_topk",
    )


def _primitive_stage_latencies_ms(
    context, geometry, stage_names: tuple[str, ...], *, experts: int, top_k: int,
) -> tuple[float, tuple[float, ...], tuple[str, ...]]:
    """Derive prologue, per-stage wave latencies, and any missing primitive names from ``PrimitiveTable``.

    When one primitive cell is unmeasured (e.g. an unmeasured SKU), all other measured
    stages preserve their exact primitive latencies while the unmeasured stage(s) fall
    back to ``0.0`` and the missing primitive IDs are surfaced explicitly.
    """
    tbl = getattr(context, "primitive_table", None)
    if tbl is None:
        return 0.0, (0.0,) * len(stage_names), ()
    dp = getattr(context, "datapath", None) or getattr(geometry, "datapath", None)
    env = resolve_dispatch_envelope(tbl, dp) if dp is not None else None
    prologue_ms = float(env.aiv_min_block_floor_ms) if env is not None else 0.0
    flag_ns = float(tbl.cell("P12_intracore_flag").band("flag_ns").p50 or 0.0)
    mte2_v_ns = float(tbl.cell("P12_intracore_flag").band("flag_mte2_v_ns").p50 or flag_ns)
    v_mte3_ns = float(tbl.cell("P12_intracore_flag").band("flag_v_mte3_ns").p50 or flag_ns)
    sfu_ms = float(tbl.cell("P08_vec_sfu").band("vec_sfu_ops_per_s_per_aiv").p50 or 0.0) / MS_PER_SECOND
    red_ms = float(tbl.cell("P09_vec_reduce_cast").band("vec_reduce_ops_per_s_per_aiv").p50 or 0.0) / MS_PER_SECOND
    sort_ms = float(tbl.cell("P10_sort_topk").band("sort_elems_per_s_per_aiv").p50 or 0.0) / MS_PER_SECOND
    alu_ms = float(tbl.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50 or 0.0) / MS_PER_SECOND
    missing: list[str] = []
    for pid, rate in (
        ("P07_vec_alu", alu_ms),
        ("P08_vec_sfu", sfu_ms),
        ("P09_vec_reduce_cast", red_ms),
        ("P10_sort_topk", sort_ms),
    ):
        if rate <= 0.0:
            missing.append(pid)
    flag_ms = flag_ns / NS_PER_MS
    sort_work = float(experts) * math.log2(max(2, experts))
    per_stage: list[float] = []
    for name in stage_names:
        if name in {"logits_load", "logits_bias_load"}:
            per_stage.append(mte2_v_ns / NS_PER_MS)
        elif name in {"sigmoid_bias", "sigmoid_score"}:
            per_stage.append((float(experts * _SIGMOID_SFU_OPS_PER_EXPERT) / sfu_ms + flag_ms) if sfu_ms > 0.0 else 0.0)
        elif name == "softmax_reduce":
            if sfu_ms > 0.0 and red_ms > 0.0:
                per_stage.append(
                    float(experts * _SOFTMAX_SFU_OPS_PER_EXPERT) / sfu_ms
                    + float(max(1, experts - 1)) / red_ms + flag_ms
                )
            else:
                per_stage.append(0.0)
        elif name in {"group_top2_scan", "group_mask"}:
            per_stage.append((float(experts) / red_ms + flag_ms) if red_ms > 0.0 else 0.0)
        elif name == "expert_topk":
            per_stage.append((sort_work / sort_ms + flag_ms) if sort_ms > 0.0 else 0.0)
        elif name == "renorm_store":
            per_stage.append((float(top_k * _RENORM_OPS_PER_SLOT) / alu_ms + v_mte3_ns / NS_PER_MS) if alu_ms > 0.0 else 0.0)
        else:
            per_stage.append(flag_ms)
    return prologue_ms, tuple(per_stage), tuple(missing)


def build_moe_gating_topk(op, geometry):
    """Row-wave owner model for group-limited and general MoE gating top-k."""
    from mindie_sim.prediction.physical_recipe_base import build_plan
    from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
    from mindie_sim.prediction.mechanism import OperatorPlan, Work
    from mindie_sim.prediction.operators.vector.vector_pipeline import AIVTiledVectorAndGatherRecipe
    is_group = _group_limited(op)
    general_kind = _general_topk_kind(op)
    if _geometry_arch_family(geometry) and (is_group or general_kind is not None):
        _gating_kernel_binding(op, group_limited=is_group, geometry=geometry)
    if not (_is_arch35_geometry(geometry) and (is_group or general_kind is not None)):
        # A2/A3 and hash/unstructured routing keep their Class D vector recipe.
        return build_plan(AIVTiledVectorAndGatherRecipe(), op, geometry)
    meta = dict(op.metadata)
    wave_width, aiv_pool = _effective_gating_aiv_budget(geometry, is_group=is_group)
    tokens = max(1, int(meta.get("row_tokens", op.local_shape[0])))
    experts = max(1, int(meta.get("num_experts", 1) or 1))
    top_k = max(1, int(meta.get("top_k", 1) or 1))
    waves = max(1, -(-tokens // wave_width))
    launched = max(1, -(-tokens // waves))
    read = float(op.hbm_bytes) - float(op.mte3_bytes)
    write = float(op.mte3_bytes)
    bias_bytes = float(
        meta.get("correction_bias_bytes")
        or (experts * NOAUX_BIAS_BYTES if meta.get("noaux_tc") else 0)
    )
    logits_read = max(0.0, read - bias_bytes)
    total_mte2 = logits_read + float(launched) * bias_bytes
    logical_vector = float(op.vector_ops)
    stage_names = (
        GATING_WAVE_STAGES if is_group
        else (GATING_SIGMOID_WAVE_STAGES if general_kind == "sigmoid" else GATING_SOFTMAX_WAVE_STAGES)
    )
    kernel_binding = _gating_kernel_binding(op, group_limited=is_group, geometry=geometry)
    owner_schedule = _gating_owner_schedule(
        op, geometry, tokens=tokens, waves=waves, launched=launched, stage_names=stage_names,
        read=read, bias_bytes=bias_bytes, write=write,
        logical_vector=logical_vector, aiv_units=wave_width,
    )
    work = Work(resources={"hbm": read + write, "mte2": total_mte2,
                           "mte3": write, "vector": logical_vector})

    def program(context):
        prologue_ms, stage_latencies, missing_primitives = _primitive_stage_latencies_ms(
            context, geometry, stage_names, experts=experts, top_k=top_k,
        )
        read_name = context.physical_role("mte2_aiv")
        read_name = (read_name if read_name in context.resource_names
                     else context.read_pipe(aiv=0))
        stage_roles = tuple(role for role in (read_name, "vector", "mte3")
                            if role in context.resource_names)
        stage_count = len(stage_names)

        def placement(active):
            units = tuple(aiv_pool[:active])
            return {context.resource_names[role]: units
                    for role in stage_roles}

        def tile(wave):
            covered = min(launched, tokens - wave * launched)
            token_share = covered / tokens
            wave_logits = logits_read * token_share
            hbm_loads = wave_logits + (bias_bytes if wave == 0 else 0.0)
            mte2_loads = wave_logits + (float(launched) * bias_bytes if wave == 0 else 0.0)
            stores = write * token_share
            stage_vector = (logical_vector * token_share) / stage_count
            owned = placement(covered)
            stages = []
            previous = ()
            for index, name in enumerate(stage_names):
                resources = {"vector": stage_vector}
                if index == 0:
                    resources.update({"hbm": hbm_loads, "mte2": mte2_loads})
                if index == stage_count - 1:
                    resources.update({"hbm": stores, "mte3": stores})
                stages.append(Stage(
                    name, context.bind(Work(resources=resources)),
                    latency=stage_latencies[index], after=previous, placement=owned))
                previous = (name,)
            return tuple(stages)

        if prologue_ms > 0.0:
            return KernelProgram((
                Stage("kernel_prologue_drain", latency=prologue_ms),
                Stage("row_waves", after=("kernel_prologue_drain",),
                      program=PipelineProgram(waves, GATING_PIPELINE_SLOTS, tile)),
            ))
        return KernelProgram((
            Stage("row_waves",
                  program=PipelineProgram(waves, GATING_PIPELINE_SLOTS, tile)),
        ))

    return OperatorPlan(work, program, {
        "native_kind": meta.get("native_kind") or kernel_binding.kernel_id,
        "row_waves": waves,
        "owner_schedule": owner_schedule.to_dict(),
        "kernel_binding": kernel_binding.to_dict(),
        "row_program": {"aiv_units": launched,
                        "wave_stages": stage_names,
                        "effective_groups": _effective_groups(meta),
                        "pricing_basis": "primitive_table_rates"},
    })


@dataclass(frozen=True)
class ScopedGatingRecipe(PlannedRecipe):
    """Register the Arch35 row planner for both group-limited and general top-k gating."""

    def prepare(self, op, geometry):
        is_group = _group_limited(op)
        general_kind = _general_topk_kind(op)
        if _geometry_arch_family(geometry) and (is_group or general_kind is not None):
            _gating_kernel_binding(op, group_limited=is_group, geometry=geometry)
        if not (_is_arch35_geometry(geometry) and (is_group or general_kind is not None)):
            from mindie_sim.prediction.physical_recipe_registry import VECTOR
            return op, VECTOR
        bound, plan = super().prepare(op, geometry)
        meta = dict(op.metadata)
        wave_width, _ = _effective_gating_aiv_budget(geometry, is_group=is_group)
        tokens = max(1, int(meta.get("row_tokens", op.local_shape[0])))
        experts = max(1, int(meta.get("num_experts", 1) or 1))
        waves = max(1, -(-tokens // wave_width))
        launched = max(1, -(-tokens // waves))
        stage_names = (
            GATING_WAVE_STAGES if is_group
            else (GATING_SIGMOID_WAVE_STAGES if general_kind == "sigmoid" else GATING_SOFTMAX_WAVE_STAGES)
        )
        read = float(op.hbm_bytes) - float(op.mte3_bytes)
        write = float(op.mte3_bytes)
        bias_bytes = float(
            meta.get("correction_bias_bytes")
            or (experts * NOAUX_BIAS_BYTES if meta.get("noaux_tc") else 0)
        )
        kernel_binding = _gating_kernel_binding(op, group_limited=is_group, geometry=geometry)
        owner_schedule = _gating_owner_schedule(
            op, geometry, tokens=tokens, waves=waves, launched=launched, stage_names=stage_names,
            read=read, bias_bytes=bias_bytes, write=write,
            logical_vector=float(op.vector_ops), aiv_units=wave_width,
        )
        bound = replace(bound, owner_schedule=owner_schedule, kernel_binding=kernel_binding)
        return bound, plan


@dataclass(frozen=True)
class Selection:
    """Declared router selection boundary for one MoE layer."""

    ops: tuple
    select: object
    ids_producer: str
    weights_producer: str | None
    fused: bool
    contract_kind: str
    cann_kind: str


def _names(value) -> set:
    if isinstance(value, dict):
        return set(value)
    return set(value or ())


def declared_gating(kernels) -> dict | None:
    """The declared CANN gating overlay entry, or ``None`` when absent."""
    raw = (kernels or {}).get(GATING_SELECTOR)
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        raw = {"implementation": raw}
    if not isinstance(raw, dict):
        raise ModelingError("moe_gating_declaration_requires_mapping_or_name")
    if set(raw) - _GATING_FIELDS:
        raise ModelingError("moe_gating_declaration_fields_unknown")
    name = str(raw.get("implementation") or GATING_IMPLEMENTATION)
    if name != GATING_IMPLEMENTATION:
        raise ModelingError(f"moe_gating_declaration_unsupported:{name}")
    if not bool(raw.get("enabled", True)):
        return None
    return dict(raw, implementation=name)


def cann_native_active(runtime) -> bool:
    """True when the Ascend CANN 9.2.1 MoE gating fusion owns this graph.

    The overlay is a *declared* lowering: the runtime must bind
    ``lowerings["moe_gating"]`` (a documented production profile does) and must
    not be a raw-eager decomposition or an explicit generic router recipe.
    """
    if runtime is None:
        return False
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    policy = dict(metadata.get("execution_policy") or {})
    single_spine = metadata.get("single_spine")
    track = str(
        metadata.get("stack_track")
        or (single_spine.get("stack_track") if isinstance(single_spine, dict) else "")
        or ""
    ).strip().lower()
    if track == "eager" or metadata.get("raw_eager") or policy.get("raw_eager"):
        return False
    backend = str(policy.get("backend") or metadata.get("backend") or "").lower()
    if backend and backend != "ascend":
        return False
    kernels = getattr(runtime, "kernels", None) or {}
    if declared_gating(kernels) is None:
        return False
    if _names(kernels.get("recipes")) & _ROUTING_RECIPES:
        return False
    return not _names(kernels.get("families")) & _ROUTING_FAMILIES


def _fused_gating(*, prefix, plan, logits_input, members, stage, requests,
                  tokens, mode, weight_dtype, weight_bytes, hash_table, noaux_tc,
                  cann_kind=None, norm_type=None):
    """One ``moe_gating_topk`` call conserving the whole member expression."""
    experts, selected = int(plan.num_experts), int(plan.top_k)
    resolved_kind, resolved_norm_type = _native_gating_dispatch(
        plan, hash_table=hash_table, noaux_tc=noaux_tc)
    native_kind = cann_kind or resolved_kind
    effective_norm_type = resolved_norm_type if norm_type is None else int(norm_type)
    vector = sum(float(member.vector_ops) for member in members)
    read = float(tokens * experts) * float(weight_bytes)
    if noaux_tc:
        read += float(experts * NOAUX_BIAS_BYTES)
    if hash_table:
        read += float(tokens) * float(HASH_ROW_ID_BYTES + plan.top_k * SELECTED_ID_BYTES)
    write = float(tokens * selected) * float(SELECTED_ID_BYTES + SELECTED_WEIGHT_BYTES)
    return vector_op(
        op_id=prefix + ".moe_gating_topk", deps=(logits_input,), stage=stage,
        layer=plan.index, requests=requests, elements=tokens * experts,
        ops_per_element=0.0, vector_ops_override=vector, dtype="fp32",
        family="topk", phase=mode, recipe="moe_gating_topk",
        shape=(tokens, selected),
        read_bytes_override=read, write_bytes_override=write,
        returns=("weights", "ids"),
        metadata=(("native_kind", native_kind),
                  ("norm_type", effective_norm_type),
                  ("input_format", weight_dtype), ("output_format", "fp32"),
                  ("topk_ids_dtype", "int32"), ("router_weight_dtype", "fp32"),
                  ("score_function", plan.router_mode),
                  ("selection_mode", HASH_SELECTION_MODE if hash_table
                   else SCORE_SELECTION_MODE),
                  ("fused_from", tuple(member.recipe for member in members)),
                  ("fused_member_count", len(members)),
                  ("renorm", bool(plan.router_norm_topk and not hash_table)),
                  ("noaux_tc", noaux_tc),
                  ("routed_scaling_factor", float(plan.routed_scaling_factor)),
                  ("groups", int(plan.router_groups)),
                  ("group_topk", int(plan.router_group_topk)),
                  ("top_k", selected), ("num_experts", experts),
                  *((("correction_bias_elements", experts),
                     ("correction_bias_bytes", experts * NOAUX_BIAS_BYTES))
                    if noaux_tc else ()),
                  ("hash_table_active_rows", tokens if hash_table else 0),
                  # Structural row geometry for the registered physical wave
                  # planner; never a timing or work-axis value.
                  ("row_tokens", tokens), ("row_experts", experts)),
    )


def _fused_boundary(ops, *, hash_table, noaux_tc):
    """Index where the one fused gating call ends and external consumers start.

    A hash call has no in-kernel renormalization, so its declared top-k
    normalization stays an external consumer.  The ``noaux_tc`` chain owns its
    normalization *and* routed scale in-kernel; every other score gating call
    owns the renorm and keeps the declared post-selection scale external.
    """
    if hash_table:
        return next((index for index, op in enumerate(ops)
                     if op.recipe == ROUTE_NORMALIZE_RECIPE), len(ops))
    if noaux_tc:
        return len(ops)
    return len(ops) - int(str(ops[-1].recipe) == ROUTED_SCALE_RECIPE)


def build_selection(*, prefix, plan, logits_input, weight_dtype, weight_bytes,
                    stage, requests, tokens, mode, runtime):
    """Build the declared selection boundary, fused when Ascend owns the graph."""
    ops, select, ids_producer, weights_producer = raw_selection(
        prefix=prefix, plan=plan, logits_input=logits_input,
        weight_dtype=weight_dtype, weight_bytes=weight_bytes, stage=stage,
        requests=requests, tokens=tokens, mode=mode)
    if not ops:
        return Selection((), select, "", None, False, CONTRACT_SCORE, "")
    hash_table = plan.router_mode == "hash"
    noaux_tc = bool(plan.router_noaux_tc) and not hash_table
    contract = CONTRACT_HASH if hash_table else CONTRACT_SCORE
    cann_kind, norm_type = _native_gating_dispatch(
        plan, hash_table=hash_table, noaux_tc=noaux_tc)
    fused = cann_native_active(runtime)
    raw_in = str(weight_dtype or "").strip().lower()
    norm_in = _GATING_ABI_DTYPE_ALIASES.get(raw_in, raw_in)
    if fused and cann_kind == TOPK_NATIVE_KIND and noaux_tc and norm_in != "fp32":
        # CANN ``aclnnMoeGatingTopK`` requires matching x/bias dtypes; an FP32
        # ``noaux_tc`` correction bias with non-FP32 logits keeps the raw graph.
        fused = False
    if fused:
        split = _fused_boundary(ops, hash_table=hash_table, noaux_tc=noaux_tc)
        gating = _fused_gating(
            prefix=prefix, plan=plan, logits_input=logits_input,
            members=tuple(ops[:split]), stage=stage, requests=requests,
            tokens=tokens, mode=mode, weight_dtype=weight_dtype,
            weight_bytes=weight_bytes, hash_table=hash_table, noaux_tc=noaux_tc,
            cann_kind=cann_kind, norm_type=norm_type)
        replaced = {op.op_id for op in ops[:split]}
        tail = [replace(op, deps=tuple(
            gating.op_id if dependency in replaced else dependency
            for dependency in op.deps)) for op in ops[split:]]
        ops = [gating, *tail]
        select = ops[-1]
        ids_producer, weights_producer = gating.op_id, select.op_id
    # A hash call owns no renormalization: its declared top-k normalization is
    # an external consumer, so it stays outside the kernel boundary.
    boundary = len(ops)
    if hash_table:
        boundary = next((index for index, op in enumerate(ops)
                         if op.recipe == ROUTE_NORMALIZE_RECIPE), len(ops))
    declare_router_selection(
        ops=ops, key=prefix + ".router_selection", plan=plan, tokens=tokens,
        weight_dtype=weight_dtype, contract_kind=contract, cann_kind=cann_kind,
        fused=fused, members=tuple(op.op_id for op in ops[:boundary]),
        entries=(ops[0].op_id,), anchor=ops[boundary - 1].op_id,
        selection_mode=HASH_SELECTION_MODE if hash_table else SCORE_SELECTION_MODE,
        noaux_tc=noaux_tc)
    return Selection(tuple(ops), select, ids_producer, weights_producer, fused,
                     contract, cann_kind)


def dispatch_quantizes_internally(runtime) -> bool:
    """True when the selected native dispatch owns its own row quantization."""
    protocol = dict((getattr(runtime, "kernels", None) or {}).get(
        "native_moe_protocol") or {})
    if str(protocol.get("kind") or "") != "arch22_v2_caller":
        return False
    return str(protocol.get("dispatch_quant") or "none") != "none"


def _routing_quantizable(meta) -> bool:
    """True for the MXFP8 / dynamic INT8 payloads ``quant_mode=1`` publishes."""
    payload = str(meta.get("output_format") or "")
    scale = str(meta.get("activation_scale_format") or "")
    group = int(meta.get("activation_group_size") or 0)
    if payload in _MX_PAYLOADS:
        return group == MX_SCALE_GROUP and scale in _E8M0_SCALES
    if payload == "int8":
        return scale in _PER_ROW_SCALES
    return False


def absorb_init_routing(*, pack, quantizer, ep, dispatch_owns_quant=False):
    """Fold one EP dispatch quantizer into ``aclnnMoeInitRoutingV3``.

    The init-routing kernel owns the permutation and the dynamic quantization
    (``quant_mode=1``): that work and the quantized payload write are billed
    once on the pack and the separate node is retired.  A dispatch that
    quantizes received rows itself keeps its declared separate quantizer.
    """
    if ep <= 1 or quantizer is None or pack is None or dispatch_owns_quant:
        return pack, None
    meta = dict(quantizer.metadata)
    if str(meta.get("quantization_scope", "")) != "moe_dispatch_expert_input":
        return pack, None
    if not _routing_quantizable(meta):
        return pack, None
    routing = {
        "native_kind": INIT_ROUTING_NATIVE_KIND,
        "quant_mode": 1,
        "input_format": meta.get("input_format"),
        "payload_format": meta.get("output_format"),
        "scale_format": meta.get("activation_scale_format"),
        "scale_shape": tuple(meta.get("activation_scale_shape") or ()),
        "scale_bytes": float(meta.get("activation_scale_bytes") or 0.0),
        "source_rows": float(meta.get("source_rows") or 0.0),
        "quantizer_op": quantizer.op_id,
        "vector_ops": float(quantizer.vector_ops),
        "write_bytes": float(quantizer.mte3_bytes),
    }
    # ``route_unpack`` still consumes the pack's inverse token-slot map after
    # the BF16 payload is replaced by the quantized payload.  That INT32
    # auxiliary write is four bytes per routed slot and must survive.
    routed_slots = 1 if pack.local_shape else 0
    for extent in pack.local_shape:
        routed_slots *= max(0, int(extent))
    inverse_bytes = float(routed_slots) * float(SELECTED_ID_BYTES)
    routing["inverse_index_bytes"] = inverse_bytes
    read = float(pack.mte2_bytes)
    write = float(quantizer.mte3_bytes) + inverse_bytes
    routing["write_bytes"] = write
    pack_meta = dict(pack.metadata)
    pack_meta.update({
        "native_kind": INIT_ROUTING_NATIVE_KIND,
        "moe_init_routing": routing,
        "activation_payload_format": routing["payload_format"],
        "activation_scale_format": routing["scale_format"],
        "activation_scale_shape": routing["scale_shape"],
        "activation_scale_bytes": routing["scale_bytes"],
        "activation_quantization_scope": "absorbed_dispatch_expert_input",
        "inverse_index_bytes": inverse_bytes,
        "inverse_index_dtype": "int32",
    })
    return replace(
        pack, recipe="moe_init_routing_v3", family="routing",
        vector_ops=float(pack.vector_ops) + float(quantizer.vector_ops),
        mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
        returns=tuple(dict.fromkeys((
            *pack.returns, "activation_scales", "inverse_indices"))),
        metadata=tuple(pack_meta.items())), quantizer.op_id
