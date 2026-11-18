"""Operator-stack selection shared by composition, structure and serving.

[Role] Own the hardware/model-generation operator defaults that the A5 V4.1
  device adaptor, the BF16 indexer and the serving recipe inventory require,
  plus the deterministic ``operator_selection_summary`` projection.
[Invariants] Structural geometry and declared hardware capability select a
  binding; a model name, an expert weight format or a measured duration never
  does. An automatically installed default is planned by the caller's own
  program builder before installation, so a default can never reach a planner
  failure. A phase-scoped default is never installed globally.
[Timeouts & Locks] O(layers) pure construction plus one bounded planner probe.
[Side-Effect Policy] Returns metadata/lowering dictionaries; writes nothing.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from mindie_sim.prediction.implementation_selection import (
    AUTOMATIC_LOWERINGS_KEY,
    fallback_note,
    is_automatic_lowering,
    selected_note,
    with_automatic_lowering,
)
from mindie_sim.prediction.family_mqsmla import (
    FORMATS as MQSMLA_CACHE_FORMATS,
    NATIVE_KIND as MQSMLA_NATIVE,
    RECIPE as MQSMLA_RECIPE,
    UNIT_CONTEXT_PARALLEL,
    supported_heads as mqsmla_supported_heads,
)
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.schema.common import ModelingError

# --- A5 V4.1 packed cache ABI -------------------------------------------------
DECLARATION_PHASE = "declaration"
A5_ARCH = "a5"
ASCEND_BACKEND = "ascend"
BF16_ACTIVATION = "bf16"
V41_ROPE_DIM = 64
V41_HEAD_DIM = 512
V41_LATENT_MQA = "latent_mqa"
V41_CACHE_KIND = "window_compressed"
V41_COMPRESSOR = "gated_ring"
MIN_ACTIVE_DECODE_TOKENS = 1
# The A5 V4.1 index cache is a packed 4-bit row: 64 payload bytes plus a
# uint8 scale per 32 values, i.e. 68 bytes for a 128-element index row.
INDEX_PACKED_ELEMENTS = 2
INDEX_GROUP_SIZE = 32
V41_A5_CACHE_FORMATS = {
    **MQSMLA_CACHE_FORMATS,
    "index_k": {
        "dtype": "uint8",
        "packed_elements": INDEX_PACKED_ELEMENTS,
        "group_size": INDEX_GROUP_SIZE,
        "scale_dtype": "uint8",
    },
}
V41_ROTARY_SELECTOR = "v41_rotary"
V41_ROTARY_IMPLEMENTATION = "v41_partial_rotary"
V41_ROTARY_NATIVE = "InplacePartialRotaryMul"
MQSMLA_SELECTOR = "mqsmla_attention"
PACKED_CACHE_FAMILY = "v41_packed_cache_abi"
MQSMLA_FAMILY = "mqsmla_decode"
SELECTED_REASON = "a5_mixed_quant_packed_cache_contract"
ROTARY_REASON = "a5_v41_mixed_quant_partial_rope"
CACHE_OVERRIDE_REASON = "explicit_cache_representation_override"
CP_REASON = "mqsmla_requires_context_parallel_cp1"
TP_REASON = "mqsmla_local_heads_outside_source_tp_tiling"
GROUPS_REASON = "mqsmla_output_groups_not_divisible_by_tp"
ARCH_REASON = "mqsmla_requires_a5_packed_cache_capability"
ACTIVATION_REASON = "mqsmla_requires_bf16_activation"
GEOMETRY_REASON = "mqsmla_requires_d512_rope64_single_kv_head"
CACHE_REASON_PREFIX = "mqsmla_cache_representation_contract_required:"

# --- BF16 LightningIndexerV2 default and serving inventory --------------------
LIGHTNING_INDEXER_SELECTOR = "lightning_indexer"
LIGHTNING_INDEXER_FAMILY = "lightning_indexer"
LIGHTNING_INDEXER_IMPLEMENTATION = "cann_bf16_lightning_indexer_v2_complete"
BF16_INDEX_CACHE_DTYPES = frozenset({"bf16", "bfloat16"})
INDEX_PROBE_ROWS = 1
SERVING_INVENTORY_KEYS = ("recipes", "programs", "families")


def normalize_regions(raw):
    """Comparable plain mapping for a declared cache-region contract."""
    if not isinstance(raw, Mapping):
        return None
    return {str(region): dict(declaration) if isinstance(declaration, Mapping)
            else declaration for region, declaration in raw.items()}


def structural_v41_latent_mqa(layer_plans) -> bool:
    """True for the declared V4.1 ``gated_ring`` window-compressed latent MQA.

    The leading sliding-window-only layers declare no compressor at all; every
    compressing layer must be the ``gated_ring`` variant, which is what
    distinguishes V4.1 from the ``weighted_pool`` V4 family.
    """
    plans = tuple(layer_plans or ())
    if not plans:
        return False
    if not all(
        plan.attention == "mla"
        and plan.mla_variant == V41_LATENT_MQA
        and plan.cache_kind == V41_CACHE_KIND
        and plan.head_dim == V41_HEAD_DIM
        and plan.qk_rope_head_dim == V41_ROPE_DIM
        and (not plan.compress_ratio or plan.compressor_kind == V41_COMPRESSOR)
        for plan in plans
    ):
        return False
    return any(plan.compressor_kind == V41_COMPRESSOR for plan in plans)


def is_a5_v41_packed_cache_stack(*, backend, arch, layer_plans, activation_dtype) -> bool:
    """A5 capability plus the structural V4.1 contract.

    Weight quantization is deliberately absent: ``vllm_ascend/models/
    deepseek_v41/cache_config.py`` selects ``DSV41_PACKED_CACHE`` from the
    detected hardware capability, so an explicit ``quant="fp8"`` checkpoint
    uses the same packed rows as the default MXFP4-expert checkpoint.
    """
    if backend != ASCEND_BACKEND or arch != A5_ARCH or activation_dtype != BF16_ACTIVATION:
        return False
    return structural_v41_latent_mqa(layer_plans)


def local_heads_supported(layer_plans, tp) -> bool:
    degree = max(int(tp), 1)
    return all(mqsmla_supported_heads(plan.num_heads // degree)
               for plan in layer_plans)


def output_groups_supported(layer_plans, tp) -> bool:
    """Grouped output projection requires ``o_groups // tp > 0`` when declared."""
    degree = max(int(tp), 1)
    return all(not int(getattr(plan, "o_groups", 0) or 0)
               or int(plan.o_groups) % degree == 0 for plan in layer_plans)


def mqsmla_decode_policy() -> dict:
    """Decode-only policy; every other geometry keeps analytical execution."""
    return {
        "cases": [{
            "domain": {
                "phases": ["decode"],
                "valid_tokens": {"min": MIN_ACTIVE_DECODE_TOKENS},
            },
            "binding": MQSMLA_RECIPE,
        }],
        "otherwise": "analytical",
    }


def automatic_mqsmla_reason(*, tp, cp, layer_plans) -> str | None:
    """Static admissibility of an automatically installed MQSMLA decode policy."""
    if int(cp) != UNIT_CONTEXT_PARALLEL:
        return CP_REASON
    if not local_heads_supported(layer_plans, tp):
        return TP_REASON
    if not output_groups_supported(layer_plans, tp):
        return GROUPS_REASON
    return None


def packed_cache_reason(model) -> str | None:
    """First declared cache region that is not the A5 packed V4.1 ABI."""
    for region, declaration in MQSMLA_CACHE_FORMATS.items():
        if cache_format(model, region) != CacheFormat(**declaration):
            return CACHE_REASON_PREFIX + region
    return None


def static_mqsmla_reason(*, arch, cp, tp, layer_plans, model) -> str | None:
    """Full static preflight for a default-installed MQSMLA decode policy."""
    if str(arch).lower() != A5_ARCH:
        return ARCH_REASON
    if model.activation_dtype != BF16_ACTIVATION:
        return ACTIVATION_REASON
    if not all(plan.num_kv_heads == 1 and plan.head_dim == V41_HEAD_DIM
               and plan.qk_rope_head_dim == V41_ROPE_DIM for plan in layer_plans):
        return GEOMETRY_REASON
    reason = automatic_mqsmla_reason(tp=tp, cp=cp, layer_plans=layer_plans)
    return reason or packed_cache_reason(model)


def apply_compose_v41_a5_defaults(*, model_dict, lowerings, backend, arch,
                                 layer_plans, activation_dtype, tp, cp,
                                 runtime_metadata=None):
    """Attach A5 V4.1 packed cache regions and decode bindings in place.

    Returns the selection notes. A caller-declared ``cache_representations``
    mapping is authoritative: when it differs from the A5 packed ABI the native
    mixed-quant claim is withdrawn and the analytical path is recorded.
    """
    notes: list[dict] = []
    if not is_a5_v41_packed_cache_stack(
        backend=backend, arch=arch, layer_plans=layer_plans,
        activation_dtype=activation_dtype,
    ):
        return notes
    metadata = model_dict.setdefault("metadata", {})
    declared = normalize_regions(metadata.get("cache_representations"))
    override = (declared is not None
                and declared != normalize_regions(V41_A5_CACHE_FORMATS))
    if declared is None:
        metadata["cache_representations"] = deepcopy(V41_A5_CACHE_FORMATS)
    if override:
        notes.append(fallback_note(PACKED_CACHE_FAMILY, CACHE_OVERRIDE_REASON))
    lowerings.setdefault(V41_ROTARY_SELECTOR, V41_ROTARY_IMPLEMENTATION)
    notes.append(selected_note(V41_ROTARY_SELECTOR, V41_ROTARY_IMPLEMENTATION,
                               V41_ROTARY_NATIVE, ROTARY_REASON))
    if MQSMLA_SELECTOR in lowerings:
        notes.append(selected_note(MQSMLA_FAMILY, MQSMLA_RECIPE, MQSMLA_NATIVE,
                                   "explicit_declaration_preserved"))
        return notes
    if override:
        notes.append(fallback_note(MQSMLA_FAMILY, CACHE_OVERRIDE_REASON))
        return notes
    reason = automatic_mqsmla_reason(tp=tp, cp=cp, layer_plans=layer_plans)
    if reason is not None:
        notes.append(fallback_note(MQSMLA_FAMILY, reason))
        return notes
    lowerings[MQSMLA_SELECTOR] = mqsmla_decode_policy()
    if runtime_metadata is not None:
        runtime_metadata[AUTOMATIC_LOWERINGS_KEY] = tuple(
            with_automatic_lowering(runtime_metadata, MQSMLA_SELECTOR)
            [AUTOMATIC_LOWERINGS_KEY])
    notes.append(selected_note(MQSMLA_FAMILY, MQSMLA_RECIPE, MQSMLA_NATIVE,
                               SELECTED_REASON))
    return notes


def v41_cached_layers(case):
    """Layers of a window-compressed latent-MQA cache stack."""
    return [p for p in case.model.layer_plans
            if p.attention == "mla" and p.mla_variant == "latent_mqa"
            and p.cache_kind == "window_compressed"]


def mqsmla_default(case, kernels):
    """Install the A5 packed-cache mixed-quant decode policy when admissible.

    The installed policy self-selects per call, and it is recorded as an
    automatic default so a per-call geometry outside the source domain keeps
    analytical execution instead of raising from the planner.
    """
    layers = v41_cached_layers(case)
    if not layers or not any(p.compressor_kind == "gated_ring" for p in layers):
        return None
    lowerings = kernels.setdefault("lowerings", {})
    automatic = installed_automatically(case.runtime.metadata, MQSMLA_SELECTOR)
    parallel = case.runtime.parallel
    reason = static_mqsmla_reason(
        arch=case.machine.arch_family, cp=parallel.cp, tp=parallel.tp,
        layer_plans=layers, model=case.model)
    if MQSMLA_SELECTOR in lowerings:
        # An explicitly declared binding keeps its own failure semantics; only a
        # binding this module installed degrades to analytical execution.
        if automatic and reason is not None:
            return {"family": MQSMLA_FAMILY, "selection": "preserved",
                    "reason": reason, "degraded": True, "installed": True}
        return {"family": MQSMLA_FAMILY,
                "selection": "default_installed" if automatic else "preserved",
                "binding": MQSMLA_RECIPE,
                "reason": SELECTED_REASON if automatic
                          else "explicit_implementation_preserved",
                "native_kind": MQSMLA_NATIVE, "degraded": False,
                "installed": automatic}
    if reason is not None:
        return {"family": MQSMLA_FAMILY, "selection": "preserved", "reason": reason,
                "degraded": True, "installed": False}
    lowerings[MQSMLA_SELECTOR] = mqsmla_decode_policy()
    return {"family": MQSMLA_FAMILY, "selection": "default_installed",
            "binding": MQSMLA_RECIPE, "reason": SELECTED_REASON,
            "native_kind": MQSMLA_NATIVE, "degraded": False, "installed": True}


def _installed_automatically(case, selector):
    from mindie_sim.compose_operator_defaults import installed_automatically
    return installed_automatically(case.runtime.metadata, selector)


def index_probe_rows(case):
    """Reachable per-call query row counts of one replica, smallest first."""
    rows = [INDEX_PROBE_ROWS, max(INDEX_PROBE_ROWS, int(case.runtime.max_num_seqs))]
    return tuple(dict.fromkeys(rows))


def lightning_indexer_reason(case, layers):
    """Source-domain preflight for an automatic BF16 indexer binding.

    The declared geometry is planned by the caller's own program builder, so an
    installed default can never reach a resident-workspace failure. The worst
    declared layer is probed at the smallest and the largest reachable decode
    row count; both bounds are real iteration geometries.
    """
    from mindie_sim.prediction.generic_latent_attention import index_cache_format
    from mindie_sim.prediction import family_lightning_indexer as li
    from mindie_sim.prediction.mechanism import Geometry
    fmt = index_cache_format(case.model)
    if fmt.dtype not in BF16_INDEX_CACHE_DTYPES or fmt.packed_elements != 1:
        return "lightning_indexer_requires_bf16_index_cache"
    if case.model.activation_dtype not in BF16_INDEX_CACHE_DTYPES:
        return "lightning_indexer_requires_bf16_query"
    if any(plan.index_heads < 1 or plan.index_dim < 1 for plan in layers):
        return "lightning_indexer_requires_declared_index_geometry"
    worst = max(layers, key=lambda plan: (plan.index_heads, plan.index_topk,
                                          plan.index_dim))
    geometry = Geometry.from_devices(case.machine.dies)
    recipe = li.recipes()[li.RECIPE]
    page = max(1, int(case.runtime.kv_block_size))
    topk = max(1, worst.index_topk)
    compression = max(1, worst.compress_ratio)
    for rows in index_probe_rows(case):
        # The exchange receive slot exists only once a row owns more than
        # ``topk`` visible compressed keys, so the probe declares that context.
        context = max(page, topk * compression + rows)
        try:
            op = li.make_op(
                query_rows=rows, context=context,
                heads=worst.index_heads, head_dim=worst.index_dim,
                topk=topk, compression=compression, page_size=page,
                kv_heads=1, query_dtype="bf16", cache_dtype=fmt.dtype,
                weight_dtype="fp32", phase="decode")
            recipe.build(op, geometry)
        except ModelingError as exc:
            return str(exc)
    return None


# Declared fusion or source roles that own the indexer boundary themselves.
INDEXER_BOUNDARY_SELECTORS = frozenset({
    "lightning_indexer", "quant_lightning_indexer", "indexed_attention",
    "sharedkv_attention",
})
INDEXER_FUSION_ROLES = frozenset({"lightning_indexer", "quant_lightning_indexer"})


def indexer_boundary_declared(lowerings):
    """True when the caller already declares the indexer fusion/source boundary."""
    if any(selector in lowerings for selector in INDEXER_BOUNDARY_SELECTORS):
        return True
    fusion = lowerings.get("native_fusion")
    roles = dict(fusion.get("options") or {}).get("roles") if isinstance(fusion, dict) else None
    return bool(INDEXER_FUSION_ROLES & set(roles or ()))


def lightning_indexer_default(case, kernels):
    """Install the BF16 LightningIndexerV2 caller for a BF16 index cache."""
    layers = [p for p in case.model.layer_plans if p.indexer and p.index_kpool == 1]
    if not layers:
        return None
    lowerings = kernels.setdefault("lowerings", {})
    if indexer_boundary_declared(lowerings):
        return {"family": LIGHTNING_INDEXER_FAMILY, "selection": "preserved",
                "binding": LIGHTNING_INDEXER_IMPLEMENTATION,
                "reason": "explicit_implementation_preserved", "degraded": False}
    reason = lightning_indexer_reason(case, layers)
    if reason is not None:
        return {"family": LIGHTNING_INDEXER_FAMILY, "selection": "preserved",
                "reason": reason, "degraded": True}
    lowerings["lightning_indexer"] = LIGHTNING_INDEXER_IMPLEMENTATION
    return {"family": LIGHTNING_INDEXER_FAMILY, "selection": "default_installed",
            "binding": LIGHTNING_INDEXER_IMPLEMENTATION,
            "reason": "declared_bf16_index_cache_geometry",
            "native_kind": "LightningIndexer", "degraded": False}


SERVING_INVENTORY_KEYS = ("recipes", "programs", "families")


def serving_recipe_inventory(case):
    """Static recipe inventory covering every phase of one serving run.

    One ``Execution`` builds one recipe book for a whole serving run, so the
    book must already contain every recipe a later iteration may reference.
    Lowerings are not an inventory: only phase-scoped policies survive, because
    a policy is re-selected per call from the actual iteration geometry. A
    decode-only scalar binding is deliberately not installed globally over a
    prefill iteration.
    """
    from mindie_sim.prediction.body_defaults import (
        DEFAULT_SCOPE, effective_body_declaration,
    )
    from mindie_sim.prediction.implementation_selection import is_policy

    decode_case, _ = effective_body_declaration(case, "decode", scope=DEFAULT_SCOPE)
    prefill_case, _ = effective_body_declaration(case, "prefill", scope=DEFAULT_SCOPE)
    kernels = dict(case.runtime.kernels)
    for key in SERVING_INVENTORY_KEYS:
        merged = {**dict(prefill_case.runtime.kernels.get(key) or {}),
                  **dict(decode_case.runtime.kernels.get(key) or {})}
        if merged:
            kernels[key] = merged
    decode_low = dict(decode_case.runtime.kernels.get("lowerings") or {})
    prefill_low = dict(prefill_case.runtime.kernels.get("lowerings") or {})
    # Phase-scoped policies are re-selected per call, so they are safe in a
    # run-wide inventory. A scalar lowering enters only when both phases
    # declare it identically; that is the phase-independent part.
    policies = {selector: value for selector, value in decode_low.items()
                if is_policy(value)}
    shared = {selector: value for selector, value in decode_low.items()
              if selector in prefill_low and prefill_low[selector] == value}
    declared = dict(case.runtime.kernels.get("lowerings") or {})
    if policies or shared or declared:
        kernels["lowerings"] = {**policies, **shared, **declared}
    return replace(case, model=decode_case.model,
                   runtime=replace(case.runtime, kernels=kernels,
                                   metadata=decode_case.runtime.metadata))


def automatic_metadata(metadata, selector):
    """Runtime metadata declaring ``selector`` a structural default."""
    return with_automatic_lowering(metadata, selector)


def installed_automatically(metadata, selector):
    """True when a structural default, not the caller, installed ``selector``."""
    return is_automatic_lowering(metadata, selector)


__all__ = [
    "AUTOMATIC_LOWERINGS_KEY", "DECLARATION_PHASE", "LIGHTNING_INDEXER_IMPLEMENTATION",
    "LIGHTNING_INDEXER_SELECTOR", "MQSMLA_SELECTOR", "V41_A5_CACHE_FORMATS",
    "apply_compose_v41_a5_defaults", "automatic_metadata", "automatic_mqsmla_reason",
    "installed_automatically", "is_a5_v41_packed_cache_stack", "lightning_indexer_default",
    "local_heads_supported", "mqsmla_decode_policy", "mqsmla_default", "normalize_regions",
    "output_groups_supported", "packed_cache_reason", "serving_recipe_inventory",
    "static_mqsmla_reason", "structural_v41_latent_mqa",
]
