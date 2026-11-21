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

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError

from .mechanism import PlannedRecipe
from .rewrite_ffn_rules import (
    CONTRACT_HASH, CONTRACT_SCORE, HASH_ROW_ID_BYTES, HASH_SELECTION_MODE,
    NOAUX_BIAS_BYTES, ROUTED_SCALE_RECIPE, ROUTE_NORMALIZE_RECIPE,
    SCORE_SELECTION_MODE,
    SELECTED_ID_BYTES, declare_router_selection, raw_selection,
)

HASH_NATIVE_KIND = "aclnnMoeGatingTopKHash"
SOFTMAX_NATIVE_KIND = "aclnnMoeGatingTopKSoftmaxV2"
INIT_ROUTING_NATIVE_KIND = "aclnnMoeInitRoutingV3"
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
#: ``aclnnMoeGatingTopK`` owns one token per AIV across ``ceil(T / 64)`` waves.
#: The six serial UB stages below partition one wave's declared row program;
#: their cycles are the source's own instruction recurrences, not a fit.
GATING_AIV_UNITS = 64
GATING_MIN_GROUPS = 1
GATING_PIPELINE_SLOTS = 1
GATING_PROLOGUE_DRAIN_CYCLES = 11517
GATING_LOGITS_BIAS_CYCLES = 380
GATING_SIGMOID_CYCLES = 470
GATING_GROUP_SCAN_CYCLES = 430
GATING_GROUP_MASK_CYCLES = 300
GATING_EXPERT_TOPK_CYCLES = 355
GATING_RENORM_CYCLES = 420
GATING_GROUP_EXPERT_CYCLES = 0.5
GATING_TOPK_EXPERT_CYCLES = 0.375
GATING_NOMINAL_CLOCK_HZ = 1.65e9
MS_PER_SECOND = 1000
GATING_WAVE_STAGES = (
    "logits_bias_load", "sigmoid_bias", "group_top2_scan", "group_mask",
    "expert_topk", "renorm_store",
)


def _a5_row_wave(geometry) -> bool:
    """The 64-AIV arch35 wave domain, read from device ownership only."""
    units = getattr(geometry, "resource_units", {}) or {}
    return int(units.get("vector", 0)) >= GATING_AIV_UNITS


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


def _wave_cycles(meta, share=1.0):
    """Serial UB cycles of one 64-token wave, split over its declared stages.

    ``share`` is the covered-token fraction of a partial wave.  The fixed
    stage recurrences are paid once per issued wave; only the per-expert scan
    work follows the number of covered tokens.
    """
    experts = int(meta.get("num_experts", 0) or 0)
    share = float(share)
    return (
        GATING_LOGITS_BIAS_CYCLES,
        GATING_SIGMOID_CYCLES,
        GATING_GROUP_SCAN_CYCLES + GATING_GROUP_EXPERT_CYCLES * experts * share,
        GATING_GROUP_MASK_CYCLES,
        GATING_EXPERT_TOPK_CYCLES + GATING_TOPK_EXPERT_CYCLES * experts * share,
        GATING_RENORM_CYCLES,
    )


def build_moe_gating_topk(op, geometry):
    """A5 group-limited gating row waves; otherwise the registered vector plan."""
    from .physical_recipe_base import build_plan, work_from_op
    from .program import KernelProgram, PipelineProgram, Stage
    from .mechanism import OperatorPlan, Work
    from .vector_pipeline import AIVTiledVectorAndGatherRecipe
    if not (_a5_row_wave(geometry) and _group_limited(op)):
        # A2/A3 and non-group gating keep their existing Class D vector recipe.
        return build_plan(AIVTiledVectorAndGatherRecipe(), op, geometry)
    base = work_from_op(op)
    meta = dict(op.metadata)
    tokens = int(meta.get("row_tokens", op.local_shape[0]))
    waves = max(1, -(-tokens // GATING_AIV_UNITS))
    read = float(op.hbm_bytes) - float(op.mte3_bytes)
    write = float(op.mte3_bytes)
    logical_vector = float(op.vector_ops)
    # One serial UB pipeline is issued per covered token on its own AIV; each
    # stage's instruction recurrence is modeled as serial stage latency while
    # conserving ``op.vector_ops`` across plan, bound context and Engine trace.
    work = Work(resources={"hbm": read + write, "mte2": read,
                           "mte3": write, "vector": logical_vector})

    def program(context):
        clock_ms = GATING_NOMINAL_CLOCK_HZ / MS_PER_SECOND
        read_name = context.physical_role("mte2_aiv")
        read_name = (read_name if read_name in context.resource_names
                     else context.read_pipe(aiv=0))
        stage_roles = tuple(role for role in (read_name, "vector", "mte3")
                            if role in context.resource_names)
        stage_count = len(GATING_WAVE_STAGES)

        def placement(active):
            units = tuple(range(active))
            return {context.resource_names[role]: units
                    for role in stage_roles}

        def tile(wave):
            covered = min(GATING_AIV_UNITS, tokens - wave * GATING_AIV_UNITS)
            share = covered / GATING_AIV_UNITS
            token_share = covered / tokens
            loads = read * token_share
            stores = write * token_share
            cycles = _wave_cycles(meta, share)
            stage_vector = (logical_vector * token_share) / stage_count
            owned = placement(covered)
            stages = []
            previous = ()
            for index, name in enumerate(GATING_WAVE_STAGES):
                resources = {"vector": stage_vector}
                if index == 0:
                    resources.update({"hbm": loads, "mte2": loads})
                if index == stage_count - 1:
                    resources.update({"hbm": stores, "mte3": stores})
                stages.append(Stage(
                    name, context.bind(Work(resources=resources)),
                    latency=cycles[index] / clock_ms,
                    after=previous, placement=owned))
                previous = (name,)
            return tuple(stages)

        return KernelProgram((
            Stage("kernel_prologue_drain",
                  latency=GATING_PROLOGUE_DRAIN_CYCLES / clock_ms),
            Stage("row_waves", after=("kernel_prologue_drain",),
                  program=PipelineProgram(waves, GATING_PIPELINE_SLOTS, tile)),
        ))

    return OperatorPlan(work, program, {
        "native_kind": meta.get("native_kind"),
        "row_waves": waves,
        "row_program": {"aiv_units": min(tokens, GATING_AIV_UNITS),
                        "wave_stages": GATING_WAVE_STAGES,
                        "wave_cycles": _wave_cycles(meta),
                        "effective_groups": _effective_groups(meta),
                        "prologue_drain_cycles": GATING_PROLOGUE_DRAIN_CYCLES},
    })


@dataclass(frozen=True)
class ScopedGatingRecipe(PlannedRecipe):
    """Register the A5 row planner without rebinding other gating domains."""

    def prepare(self, op, geometry):
        if not (_a5_row_wave(geometry) and _group_limited(op)):
            from .physical_recipe_registry import VECTOR
            return op, VECTOR
        return super().prepare(op, geometry)


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
    if metadata.get("raw_eager") or policy.get("raw_eager"):
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
                  tokens, mode, weight_dtype, weight_bytes, hash_table, noaux_tc):
    """One ``moe_gating_topk`` call conserving the whole member expression."""
    experts, selected = int(plan.num_experts), int(plan.top_k)
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
        metadata=(("native_kind", HASH_NATIVE_KIND if hash_table else SOFTMAX_NATIVE_KIND),
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
    cann_kind = HASH_NATIVE_KIND if hash_table else SOFTMAX_NATIVE_KIND
    fused = cann_native_active(runtime)
    if fused:
        split = _fused_boundary(ops, hash_table=hash_table, noaux_tc=noaux_tc)
        gating = _fused_gating(
            prefix=prefix, plan=plan, logits_input=logits_input,
            members=tuple(ops[:split]), stage=stage, requests=requests,
            tokens=tokens, mode=mode, weight_dtype=weight_dtype,
            weight_bytes=weight_bytes, hash_table=hash_table, noaux_tc=noaux_tc)
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
