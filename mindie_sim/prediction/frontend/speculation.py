"""Draft generation across all 4 speculative archetypes (DSpark, V2 MTP, EAGLE3, NGram).

[Role] Derive proposal and context-initialization operator graphs from declared
  draft structures: Archetype 1 (DSpark/DFlash context-only prepare + single
  query block + serial replicated Markov chain), Archetype 2 (Ascend V2 MTP
  single-module / cyclic_nextn / multi_module_mtp), Archetype 3 (EAGLE / EAGLE3
  with replicated fc and d2t handling), and Archetype 4 (NGram zero-neural).
[Invariants] Proposals are verified before post-accept commitment and next-step
  drafting. DSpark context preparation executes zero routed FFN and zero
  attention scores. Replicated Markov chains emit zero vocab collectives.
[Timeouts & Locks] Bounded by scheduled proposal count and draft layers. No I/O.
[Side-Effect Policy] Constructs immutable operator records only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math

from mindie_sim.catalog.draft import optional_draft_spec, draft_quant
from mindie_sim.catalog import draft_spec as _draft_spec_mod
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.prediction.structure import stage_map
from mindie_sim.prediction.implementation_selection import resolve, LoweringRegion


PROPOSAL_ID_BYTES = 8
_DRAFT_RMSNORM_OPS_PER_ELEMENT = 5
_DRAFT_SAMPLING_OPS_PER_ELEMENT = 5
_DRAFT_ROPE_OPS_PER_ELEMENT = 4
_DRAFT_D2T_SCATTER_OPS_PER_ELEMENT = 1
_DRAFT_FP32_BYTES = 4
ORG_VOCAB_PADDING_SIZE = 64
QSA_INDEX_GROUP_TOKENS = 4
QSA_INDEX_COMPRESSED_GROUPS = 512
QSA_INDEX_RAW_TAIL_TOKENS = 3
QSA_INDEX_BUFFER_COLS = QSA_INDEX_COMPRESSED_GROUPS * QSA_INDEX_GROUP_TOKENS + QSA_INDEX_RAW_TAIL_TOKENS
KPOOL_TAIL_ALIGNMENT = 128
_NON_NEURAL_SPEC_METHODS = frozenset({"ngram", "ngram_gpu"})
_DSPARK_SPEC_METHODS = frozenset({"dspark", "dflash"})
PROLOG_ROLE = "mla_prolog"
FUSION_OPTION_ROLES = "roles"
FUSION_OPTION_PROFILES = ("program_profiles", "program_options")


def compute_speculative_accept_step(accept_rate, draft_tokens) -> float:
    """Expected accepted tokens in one verification step.

    ``(1 - a**(d+1)) / (1 - a)`` with an exact ``a == 1`` branch.
    """
    if isinstance(accept_rate, bool) or not isinstance(accept_rate, (int, float)):
        raise ModelingError("speculative_accept_rate_invalid")
    rate = float(accept_rate)
    if not math.isfinite(rate) or rate < 0 or rate > 1.0:
        raise ModelingError("speculative_accept_rate_invalid")
    if (not isinstance(draft_tokens, int) or isinstance(draft_tokens, bool)
            or draft_tokens < 0):
        raise ModelingError("speculative_draft_tokens_invalid")
    if math.isclose(rate, 1.0):
        return float(draft_tokens + 1)
    return (1.0 - rate ** (draft_tokens + 1)) / (1.0 - rate)


def _ready_deps(ready: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(ready, str):
        return (ready,) if ready else ()
    return tuple(dict.fromkeys(str(dep) for dep in ready if str(dep)))


def _last_ready(ready: str | Sequence[str]) -> str:
    if isinstance(ready, str):
        return ready
    deps = _ready_deps(ready)
    return deps[-1] if deps else ""


def _spec_method(runtime) -> str:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    return str(meta.get("spec_method") or meta.get("draft_mode") or "").strip().lower()


def _tag_draft_ops(ops, rows_by_request, *, step=0, extra_meta=()):
    """Attach the actual draft producer identity and materialized row counts."""
    rows = tuple(sorted((str(request), int(count)) for request, count in rows_by_request.items()))
    extra = dict(extra_meta)
    return tuple(replace(op, metadata=tuple({
        **dict(op.metadata),
        "layer_scope": "draft",
        "producer_scope": "draft",
        "draft_step": int(step),
        "draft_rows_by_request": rows,
        **extra,
    }.items())) for op in ops)


from mindie_sim.catalog.draft_geometry import (  # noqa: F401
    _draft_layer_plan, _reference_attention_plan, draft_layer_plans,
)


def _draft_enforce_eager(runtime) -> bool:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if bool(meta.get("draft_enforce_eager", False)):
        return True
    spec_cfg = meta.get("speculative_config")
    return isinstance(spec_cfg, dict) and bool(spec_cfg.get("enforce_eager", False))


def _target_prolog_has_custom_cache_layout(binding) -> bool:
    if not isinstance(binding, dict):
        return False
    opts = binding.get("options")
    return isinstance(opts, dict) and bool(opts.get("cache_layout"))


def _draft_mla_prolog_eligible(draft_model, runtime) -> bool:
    """True iff every draft MLA layer satisfies vllm_ascend_mla_prolog_v3 validation."""
    if draft_model is None:
        return False
    from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog
    mla_plans = [
        p for p in getattr(draft_model, "layer_plans", ())
        if p.attention == "mla"
        and str(getattr(p, "mla_variant", "latent_mla") or "latent_mla") == "latent_mla"
    ]
    if not mla_plans:
        return False
    for plan in mla_plans:
        try:
            family_mla_prolog.validate(
                plan=plan,
                parallel=runtime.parallel,
                model=draft_model,
                quant=draft_model.quant,
                lowering_passes=runtime.lowering_passes,
            )
        except ModelingError:
            return False
    return True


def _independent_draft_runtime(runtime, selections=(), *, draft_model=None):
    """Use the independent draft's declared writer for its body prolog."""
    kernels = dict(getattr(runtime, "kernels", {}) or ())
    lowerings = dict(kernels.get("lowerings", {}) or ())
    eager_draft = _draft_enforce_eager(runtime)
    if eager_draft:
        lowerings.pop(PROLOG_ROLE, None)
        lowerings.pop("native_fusion", None)
        return replace(
            runtime,
            graph_capture_mode="none",
            kernels={**kernels, "lowerings": lowerings},
        )
    if PROLOG_ROLE not in lowerings:
        return runtime
    draft_binding = any(
        choice["selector"] == PROLOG_ROLE
        and choice["implementation"] != "analytical"
        for choice in selections)
    prolog_decl = lowerings.get(PROLOG_ROLE)
    compatible_plain_prolog = (
        not draft_binding
        and not _target_prolog_has_custom_cache_layout(prolog_decl)
        and (isinstance(prolog_decl, str) or (
            isinstance(prolog_decl, dict) and "cases" not in prolog_decl
        ))
        and _draft_mla_prolog_eligible(draft_model, runtime)
    )
    if not draft_binding and not compatible_plain_prolog:
        lowerings.pop(PROLOG_ROLE)
        return replace(runtime, kernels={**kernels, "lowerings": lowerings})
    if compatible_plain_prolog:
        lowerings[PROLOG_ROLE] = "vllm_ascend_mla_prolog_v3"
    fusion = lowerings.get("native_fusion")
    if isinstance(fusion, dict):
        options = dict(fusion.get("options") or {})
        options[FUSION_OPTION_ROLES] = (PROLOG_ROLE,)
        for key in FUSION_OPTION_PROFILES:
            value = options.get(key)
            if isinstance(value, dict):
                options[key] = {name: item for name, item in value.items()
                                if name == PROLOG_ROLE}
        traffic = options.get("traffic_profile")
        if isinstance(traffic, dict):
            options["traffic_profile"] = {
                name: item for name, item in traffic.items() if name == PROLOG_ROLE}
        lowerings["native_fusion"] = {**fusion, "options": options}
    kernels["lowerings"] = lowerings
    return replace(runtime, kernels=kernels)


def _cyclic_draft_model(model, plans, step):
    """One declared NextN block for the proposal step it owns."""
    plan = plans[step % len(plans)]
    return replace(model, layers=1, layer_plans=(plan,))


def _draft_cache_model(model, draft, draft_model, runtime, machine):
    from mindie_sim.prediction.operators.mla_prolog.draft_context import draft_cache_requires_prolog_v3
    draft_kv_dtype = str(getattr(draft_model.quant, "kv_dtype", "") or "bf16")
    if draft.attention_kind == "mla":
        context_prolog_v3 = draft_cache_requires_prolog_v3(
            model, runtime=runtime, machine=machine)
        cache_reps = {"mla_kv": {"dtype": draft_kv_dtype}}
    else:
        context_prolog_v3 = False
        cache_reps = {
            "gqa_kv": {"dtype": draft_kv_dtype},
            "kv_cache": {"dtype": draft_kv_dtype},
        }
    metadata = dict(draft_model.metadata)
    metadata.pop("cache_allocator", None)
    metadata["cache_representations"] = cache_reps
    metadata["attention_execution"] = {
        **metadata.get("attention_execution", {}), "dcp_indexer_sharded": False}
    draft_model = replace(
        draft_model, kv_dtype=draft_kv_dtype, metadata=tuple(metadata.items())
    )
    return draft_model, context_prolog_v3


def resolve_ar_draft_capture_mode(
    runtime,
    draft,
    *,
    step: int,
    decode_query_len: int,
    decode_request_sizes: Sequence[int] | None = None,
    moe_comm_selector=None,
) -> tuple[str, tuple[int, ...]]:
    """Enforce VA target_dependent_ar/aclgraph.py:50-106 PCP downgrade & width-1 MoE ALLTOALL filter."""
    raw_mode = str(getattr(runtime, "graph_capture_mode", "none") or "none").strip().lower()
    if _draft_enforce_eager(runtime):
        return "none", ()
    mode = raw_mode
    pcp = int(getattr(getattr(runtime, "parallel", None), "pcp", 1) or 1)
    if pcp > 1 and int(decode_query_len) > 1 and mode in {"full", "full_decode_only", "full_and_piecewise"}:
        mode = "piecewise" if mode == "full_and_piecewise" else "none"
    sizes = tuple(int(s) for s in (decode_request_sizes or ()))
    if (
        sizes
        and int(decode_query_len) == 1
        and int(getattr(draft, "num_experts", 0) or 0) > 0
        and mode in {"full", "full_decode_only", "full_and_piecewise"}
    ):
        def _comm_of(sz: int) -> str:
            if callable(moe_comm_selector):
                return str(moe_comm_selector(sz)).strip().lower()
            meta = dict(getattr(runtime, "metadata", ()) or ())
            return str(meta.get("moe_comm_mode") or getattr(runtime, "moe_recipe", "alltoall") or "").strip().lower()
        sizes = tuple(sz for sz in sizes if _comm_of(sz) != "alltoall")
    del step
    return mode, sizes


def _is_dspark_block_draft(draft, spec_method: str) -> bool:
    return bool(
        spec_method in _DSPARK_SPEC_METHODS
        or draft.kind in {"dspark", "mtp_dspark", "dflash"}
        or draft.noncausal_query_block
        or draft.execution_mode == "block_dspark"
    )


def _validate_method_draft_compatibility(spec_method: str, draft) -> None:
    """Reject contradictory explicit spec_method vs attached DraftSpec kind."""
    method = str(spec_method or "").strip().lower()
    if not _draft_spec_mod.draft_compatible_with_spec_method(draft, method):
        raise ModelingError(f"draft_structure_incompatible_for_speculative_method:{method}:{draft.kind}")


def _resolve_draft_tp(runtime, draft) -> int:
    """Resolve single draft TP consistent with target_tp; fail fast on contradictory overrides."""
    del draft
    target_tp = max(1, int(getattr(getattr(runtime, "parallel", None), "tp", 1) or 1))
    meta = dict(getattr(runtime, "metadata", ()) or ())
    raw = meta.get("draft_tensor_parallel_size", meta.get("draft_tp"))
    if raw is not None and int(raw) > 0:
        req_tp = int(raw)
        if req_tp != target_tp:
            raise ModelingError(
                f"draft_tp_mismatch_with_target_tp:draft_tp={req_tp}:target_tp={target_tp}"
            )
        return req_tp
    return target_tp


def _slot_valid_flags(entries: Sequence, feature_rows: int) -> tuple[bool, ...]:
    return tuple(s if isinstance(s, bool) else int(s) >= 0 for s in entries[:feature_rows])


def _dspark_layer_slot_spec(
    runtime, layer_idx: int, *, feature_rows: int, global_none: bool,
) -> int | tuple[bool, ...] | None:
    """Preserve per-row slot validity masks when supplied, else return scalar valid count."""
    if global_none:
        return None
    meta = dict(getattr(runtime, "metadata", ()) or ())
    per_layer = meta.get("dspark_layer_slot_mappings")
    if isinstance(per_layer, Mapping):
        if layer_idx in per_layer or str(layer_idx) in per_layer:
            entry = per_layer.get(layer_idx, per_layer.get(str(layer_idx)))
            if entry is None:
                return None
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                return _slot_valid_flags(entry, feature_rows)
            return max(0, min(feature_rows, int(entry)))
    elif isinstance(per_layer, Sequence) and not isinstance(per_layer, (str, bytes)) and layer_idx < len(per_layer):
        entry = per_layer[layer_idx]
        if entry is None:
            return None
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            return _slot_valid_flags(entry, feature_rows)
        return max(0, min(feature_rows, int(entry)))
    per_counts = meta.get("dspark_valid_slot_rows_by_layer")
    if isinstance(per_counts, Mapping) and (layer_idx in per_counts or str(layer_idx) in per_counts):
        val = per_counts.get(layer_idx, per_counts.get(str(layer_idx)))
        return None if val is None else max(0, min(feature_rows, int(val)))
    slots = meta.get("dspark_slot_mapping")
    if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
        return _slot_valid_flags(slots, feature_rows)
    if meta.get("dspark_valid_slot_rows") is not None:
        return max(0, min(feature_rows, int(meta["dspark_valid_slot_rows"])))
    return feature_rows


def _dspark_layer_valid_rows(runtime, layer_idx: int, *, feature_rows: int, global_none: bool) -> int | None:
    """Return valid slot write count for one DSpark layer, or None if layer mapping is absent."""
    spec = _dspark_layer_slot_spec(
        runtime, layer_idx, feature_rows=feature_rows, global_none=global_none,
    )
    if spec is None or isinstance(spec, int):
        return spec
    return sum(1 for valid in spec if valid)


def _dspark_physical_kv_row_bytes(model, draft, draft_model, layer_idx: int, runtime, *, kv_dtype: str) -> int:
    """Physical per-row bytes for one DSpark cache write, honoring packed pitch formats."""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if meta.get("dspark_packed_kv_row_bytes") is not None:
        return max(1, int(meta["dspark_packed_kv_row_bytes"]))
    from mindie_sim.catalog.cache_pools import _draft_row_bytes_int, _packed_mla_row_bytes
    if str(kv_dtype).strip().lower() in {"int8", "c8", "fp8", "fp8_e4m3", "mxfp8"} and layer_idx < len(draft_model.layer_plans):
        return int(_packed_mla_row_bytes(draft_model, draft_model.layer_plans[layer_idx]))
    return int(_draft_row_bytes_int(model, draft, tp=max(1, int(getattr(runtime.parallel, "tp", 1) or 1))))


def _dspark_sample_from_anchor(draft, runtime, spec_method: str) -> bool:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if "sample_from_anchor" in meta:
        val = bool(meta["sample_from_anchor"])
        if spec_method == "dflash" and val:
            raise ModelingError("dflash_rejects_sample_from_anchor")
        return val
    if spec_method == "dflash":
        return False
    return bool(draft.kind in {"dspark", "mtp_dspark"})


def _hc_collapse_before_sp_gather(draft, runtime) -> bool:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if "hc_collapse_before_sp_gather" in meta:
        return bool(meta["hc_collapse_before_sp_gather"])
    if hasattr(draft, "hc_collapse_before_sp_gather"):
        return bool(getattr(draft, "hc_collapse_before_sp_gather"))
    return bool(
        int(draft.mhc_mult) > 1
        and int(draft.layers) > 1
        and str(draft.context_projection_mode) == "shared_compressed_kv"
    )


def _draft_sp_enabled(runtime, draft, *, is_dspark: bool) -> bool:
    if not is_dspark or int(getattr(runtime.parallel, "tp", 1) or 1) <= 1:
        return False
    if "sequence_parallel" in (getattr(runtime, "kernels", {}) or {}):
        return True
    meta = dict(getattr(runtime, "metadata", ()) or ())
    sp_decl = meta.get("sequence_parallel")
    return isinstance(sp_decl, dict) and bool(sp_decl.get("implementation"))


def _draft_capture_bucket(runtime, logical_rows: int) -> int:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    explicit = int(meta.get("draft_capture_bucket", 0) or 0)
    if explicit > 0:
        return explicit
    keys = tuple(int(k) for k in (getattr(runtime, "graph_capture_keys", ()) or ()) if int(k) > 0)
    if not keys:
        return 0
    for key in sorted(keys):
        if key >= logical_rows:
            return key
    return 0


def _replicated_feature_proj(draft) -> bool:
    return bool(
        draft.kind in {"eagle", "eagle3"}
        or (draft.kind in {"dspark", "mtp_dspark", "standalone_draft"} and draft.attention_kind == "gqa")
    )


def _draft_features(model, draft, draft_model, step_model, runtime, *, previous, prefix,
                    stage, requests, rows, feature_rows, tp, phase="draft"):
    ops = []
    base_deps = _ready_deps(previous)
    feature_deps = list(base_deps)
    targets = stage_map(model, runtime)
    for layer_id in draft.target_layer_ids:
        if layer_id < len(targets):
            source_stage = targets[layer_id]
            if source_stage != stage:
                transfer = comm_op(
                    op_id=prefix + f"features_from{layer_id}", deps=base_deps, stage=stage,
                    layer=-1, requests=requests, family="pp_p2p", phase=phase,
                    recipe="draft_feature_transfer",
                    link_bytes=feature_rows * model.hidden_size
                    * dtype_bytes(model.activation_dtype),
                    shape=(feature_rows, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                              ("destination_stage", stage)))
                ops.append(transfer)
                feature_deps.append(transfer.op_id)
    replicated_fc = _replicated_feature_proj(draft)
    out_cols = step_model.hidden_size if replicated_fc else ((step_model.hidden_size + tp - 1) // tp)
    projection = gemm_op(
        op_id=prefix + "features", deps=tuple(feature_deps), stage=stage, layer=-1, requests=requests,
        m=feature_rows, n=out_cols,
        k=draft.input_width, quant=step_model.quant,
        category=draft.input_projection_category, phase=phase, family="gemm",
        recipe="draft_features",
        metadata=(("replicated_fc", replicated_fc),),
    )
    if rows and draft.input_mode == "dual_normalized":
        projected = []
        for component in ("embedding", "target_hidden"):
            normalized = vector_op(
                op_id=prefix + component + "_norm", deps=tuple(feature_deps), stage=stage,
                layer=-1, requests=requests, elements=rows * draft.hidden_size,
                ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype, family="rmsnorm",
                phase=phase, recipe="draft_input_norm")
            projection = gemm_op(
                op_id=prefix + component + "_projection", deps=(normalized.op_id,), stage=stage,
                layer=-1, requests=requests, m=rows, n=draft.hidden_size, k=draft.hidden_size,
                quant=draft_model.quant, category="linear_param_bf16", phase=phase,
                family="gemm", recipe="draft_feature_projection")
            ops.extend((normalized, projection))
            projected.append(projection.op_id)
        projection = vector_op(
            op_id=prefix + "features", deps=tuple(projected), stage=stage, layer=-1,
            requests=requests, elements=rows * draft.hidden_size, ops_per_element=1,
            dtype=draft.activation_dtype, family="elementwise", phase=phase,
            recipe="draft_feature_add")
        ops.append(projection)
    elif rows and draft.input_mode == "normalized_concat":
        norm_ids = []
        for component in ("enorm", "hnorm"):
            norm_op = vector_op(
                op_id=prefix + component, deps=tuple(feature_deps), stage=stage,
                layer=-1, requests=requests, elements=rows * draft.hidden_size,
                ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype, family="rmsnorm",
                phase=phase, recipe="draft_input_norm",
                shape=(rows, draft.hidden_size),
            )
            ops.append(norm_op)
            norm_ids.append(norm_op.op_id)
        projection = replace(projection, deps=tuple(norm_ids))
        ops.append(projection)
        if tp > 1 and not replicated_fc:
            gathered = comm_op(
                op_id=prefix + "features_gather", deps=(projection.op_id,), stage=stage,
                layer=-1, requests=requests, family="allgather", phase=phase,
                recipe="draft_features_gather",
                link_bytes=feature_rows * draft.hidden_size
                * dtype_bytes(draft.activation_dtype) * (tp - 1) / tp,
                shape=(feature_rows, draft.hidden_size), metadata=(("group_axis", "tp"), ("steps", tp - 1)))
            ops.append(gathered)
            projection = gathered
    elif rows:
        ops.append(projection)
        if tp > 1 and not replicated_fc:
            gathered = comm_op(
                op_id=prefix + "features_gather", deps=(projection.op_id,), stage=stage,
                layer=-1, requests=requests, family="allgather", phase=phase,
                recipe="draft_features_gather",
                link_bytes=feature_rows * draft.hidden_size
                * dtype_bytes(draft.activation_dtype) * (tp - 1) / tp,
                shape=(feature_rows, draft.hidden_size), metadata=(("group_axis", "tp"), ("steps", tp - 1)))
            ops.append(gathered)
            projection = gathered
    return ops, projection


def _lower_dspark_shared_kv_layers(
    ops, model, draft, draft_model, runtime, *,
    ctx_ready: str, prefix: str, stage: int, requests: tuple[str, ...],
    feature_rows: int, phase: str, slot_mapping_none: bool,
) -> str:
    """Emit stacked WKV projection and per-layer norm/RoPE/KV scatter for shared-compressed DSpark."""
    kv_rank = int(draft.kv_lora_rank or draft.kv_width)
    wkv = gemm_op(
        op_id=prefix + "context_wkv_proj", deps=(ctx_ready,), stage=stage,
        layer=-1, requests=requests, m=feature_rows, n=int(draft.layers) * kv_rank,
        k=draft.hidden_size, quant=draft_model.quant,
        category=draft.input_projection_category, phase=phase,
        family="gemm", recipe="draft_context_wkv_proj",
        metadata=(
            ("disable_tp", True),
            ("draft_layers", int(draft.layers)),
            ("kv_lora_rank", kv_rank),
            ("context_projection_mode", "shared_compressed_kv"),
        ),
    )
    ops.append(wkv)
    ctx_ready = wkv.op_id
    if slot_mapping_none:
        return ctx_ready
    kv_dtype = str(getattr(draft_model.quant, "kv_dtype", "") or draft.activation_dtype)
    act_elem_bytes = dtype_bytes(draft.activation_dtype)
    rope_dim = int(draft.qk_rope_head_dim or 0)
    for lyr in range(int(draft.layers)):
        valid_store_rows = _dspark_layer_valid_rows(
            runtime, lyr, feature_rows=feature_rows, global_none=slot_mapping_none,
        )
        if valid_store_rows is None:
            continue
        lnorm = vector_op(
            op_id=f"{prefix}l{lyr}.context_kv_norm", deps=(ctx_ready,), stage=stage,
            layer=lyr, requests=requests, elements=feature_rows * kv_rank,
            ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
            dtype=draft.activation_dtype, family="rmsnorm", phase=phase,
            recipe="rmsnorm", shape=(feature_rows, kv_rank),
            metadata=(("draft_context_stage", "kv_norm"),),
        )
        ops.append(lnorm)
        prev_l = lnorm.op_id
        if rope_dim > 0:
            lrope = vector_op(
                op_id=f"{prefix}l{lyr}.context_rope", deps=(prev_l,), stage=stage,
                layer=lyr, requests=requests, elements=feature_rows * rope_dim,
                ops_per_element=_DRAFT_ROPE_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype, family="rotary", phase=phase,
                recipe="rotary", shape=(feature_rows, rope_dim),
                metadata=(("draft_context_stage", "context_rope"),),
            )
            ops.append(lrope)
            prev_l = lrope.op_id
        if valid_store_rows <= 0:
            ctx_ready = prev_l
            continue
        row_pitch_bytes = _dspark_physical_kv_row_bytes(
            model, draft, draft_model, lyr, runtime, kv_dtype=kv_dtype,
        )
        lstore = vector_op(
            op_id=f"{prefix}l{lyr}.context_kv_store", deps=(prev_l,), stage=stage,
            layer=lyr, requests=requests, elements=valid_store_rows * kv_rank,
            ops_per_element=0, dtype=kv_dtype, family="cache_store", phase=phase,
            recipe="kv_store", shape=(valid_store_rows, kv_rank),
            read_bytes_override=valid_store_rows * kv_rank * act_elem_bytes,
            write_bytes_override=valid_store_rows * row_pitch_bytes,
            metadata=(
                ("context_projection_mode", "shared_compressed_kv"),
                ("draft_context_stage", "kv_scatter"),
                ("valid_slot_rows", valid_store_rows),
                ("feature_rows", feature_rows),
                ("physical_row_bytes", row_pitch_bytes),
            ),
        )
        ops.append(lstore)
        ctx_ready = lstore.op_id
    return ctx_ready


def lower_dspark_context_kv(
    model,
    draft,
    draft_model,
    runtime,
    *,
    previous,
    prefix: str,
    stage: int,
    requests: tuple[str, ...],
    feature_rows: int,
    query_rows: int,
    proposal_views: Sequence,
    prefill: bool,
    external_context: bool,
    context_prolog_v3: bool,
    eager_draft: bool,
    slot_mapping_none: bool = False,
    machine=None,
) -> tuple[list, str]:
    """Context-only DSpark KV preparation used on BOTH prompt init and decode verified rows.

    Executes zero routed MoE FFN and zero attention score ops. Preserves exact
    per-variant ``slot_mapping is None`` semantics: V4/V4.1 stacked WKV runs
    ``context_wkv_proj`` before returning early, whereas K3.1 MLA returns
    before KV projection (``kimi_k3_dspark.py:219``).
    """
    if feature_rows <= 0:
        return [], _last_ready(previous)
    phase = "draft_prefill" if prefill else "draft"
    tp = int(runtime.parallel.tp)
    ops, projection = _draft_features(
        model, draft, draft_model, draft_model, runtime,
        previous=previous, prefix=prefix, stage=stage, requests=requests,
        rows=feature_rows, feature_rows=feature_rows, tp=tp, phase=phase,
    )
    main_norm = vector_op(
        op_id=prefix + "main_norm", deps=(projection.op_id,), stage=stage,
        layer=-1, requests=requests, elements=feature_rows * draft.hidden_size,
        ops_per_element=_DRAFT_RMSNORM_OPS_PER_ELEMENT,
        dtype=draft.activation_dtype, family="rmsnorm", phase=phase,
        recipe="draft_main_norm", shape=(feature_rows, draft.hidden_size),
    )
    ops.append(main_norm)
    ctx_ready = main_norm.op_id
    ctx_mode = str(draft.context_projection_mode or "none")
    if ctx_mode == "none":
        ctx_mode = str(dict(draft.source).get("context_projection_mode", "none"))
    if ctx_mode == "shared_compressed_kv" or not external_context:
        ctx_ready = _lower_dspark_shared_kv_layers(
            ops, model, draft, draft_model, runtime, ctx_ready=ctx_ready,
            prefix=prefix, stage=stage, requests=requests, feature_rows=feature_rows,
            phase=phase, slot_mapping_none=slot_mapping_none,
        )
        return ops, ctx_ready
    if slot_mapping_none:
        return ops, ctx_ready
    per_layer_valid = [
        _dspark_layer_slot_spec(runtime, lyr, feature_rows=feature_rows, global_none=slot_mapping_none)
        for lyr in range(int(draft.layers))
    ]
    if all(v is None for v in per_layer_valid):
        return ops, ctx_ready
    has_explicit_slots = any(
        v is None
        or isinstance(v, Sequence)
        or (isinstance(v, int) and v < feature_rows)
        for v in per_layer_valid
    )
    from mindie_sim.catalog.context_parallel import cp_interleave
    from mindie_sim.prediction.operators.mla_prolog.draft_context import (
        draft_cache_requires_prolog_v3_scatter,
        prepare,
    )
    rows_by_view = [
        int(v.global_q_len or v.q_len) if prefill else max(int(v.q_len), int(v.draft_proposal_tokens) + 1)
        for v in proposal_views
    ]
    requires_staging_scatter = draft_cache_requires_prolog_v3_scatter(
        model, runtime=runtime, machine=machine, draft=draft, parallel=runtime.parallel,
    )
    ctx_ready = prepare(
        ops, projection=main_norm, model=draft_model, draft=draft,
        rows=feature_rows, query_rows=query_rows, prefix=prefix,
        stage=stage, requests=requests, parallel=runtime.parallel,
        rows_by_view=rows_by_view,
        views=proposal_views,
        contexts=[view.context for view in proposal_views],
        block=cp_interleave(runtime),
        use_prolog_v3=context_prolog_v3,
        layer_slot_valid_rows=per_layer_valid if has_explicit_slots else None,
        requires_staging_scatter=requires_staging_scatter,
    )
    return ops, ctx_ready


def _lmhead_tp_pad_rows(runtime, valid_rows: int, *, max_k: int) -> int:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    enabled = bool(
        meta.get("lmhead_tp_enable", False)
        or int(getattr(runtime.parallel, "lm_head_tp", 1) or 1) > 1
    )
    if not enabled or max_k <= 0:
        return valid_rows
    max_seqs = int(getattr(runtime, "max_num_seqs", 0) or meta.get("max_num_seqs", 0) or 0)
    return max(valid_rows, max_seqs * max_k) if max_seqs > 0 else valid_rows


def _append_dspark_decode_head(
    ops, frontend, model, draft, step_model, draft_model,
    runtime, selected_runtime, proposal_views, previous, *,
    prefix, stage, requests, body_input_rows, k_steps,
    spec_method: str, hc_collapsed: bool, phase: str = "draft",
    base_rows_override: int | None = None,
    per_req_ks: Sequence[int] | None = None,
):
    batch_seqs = max(1, len(proposal_views))
    req_ks = tuple(int(k) for k in per_req_ks) if per_req_ks is not None else (int(k_steps),) * batch_seqs
    logical_q = sum(int(v.q_len) for v in proposal_views)
    wanted_rows = int(base_rows_override) if base_rows_override is not None else sum(req_ks)
    base_rows = min(logical_q, wanted_rows) if logical_q > 0 else 0
    lmhead_pad = _lmhead_tp_pad_rows(runtime, base_rows, max_k=k_steps)
    heads, base_logits_id = frontend.logits(
        step_model, selected_runtime, proposal_views, previous,
        prefix=prefix, mode=phase, stage=stage, sample=False,
        input_tokens=body_input_rows,
        selected_logit_rows=base_rows,
        vocab_padding_size=ORG_VOCAB_PADDING_SIZE,
        lmhead_pad_rows=lmhead_pad,
        hc_collapsed=hc_collapsed,
    )
    ops.extend(heads)
    if not base_rows:
        return base_logits_id
    draft_vocab = int(draft.vocab_size or draft_model.vocab_size)
    target_vocab = int(model.vocab_size)
    meta = dict(getattr(runtime, "metadata", ()) or ())
    sample_method = str(meta.get("draft_sample_method", "greedy") or "greedy").strip().lower()
    probabilistic = sample_method == "probabilistic"
    run_confidence = bool(draft.confidence_executes or (runtime.dspark and runtime.dspark.confidence))
    if draft.markov_rank > 0 and spec_method != "dflash":
        prev_sample = None
        for m_step in range(k_steps):
            step_seqs = sum(1 for rk in req_ks if rk > m_step)
            if step_seqs <= 0:
                continue
            step_deps = (base_logits_id,) if prev_sample is None else (base_logits_id, prev_sample)
            lookup = vector_op(
                op_id=f"{prefix}markov_lookup_{m_step}", deps=step_deps, stage=stage, layer=-1,
                requests=requests, elements=step_seqs * draft.markov_rank, ops_per_element=1,
                dtype=draft.activation_dtype, family="embedding", phase=phase,
                recipe="markov_embedding",
                metadata=(("markov_step", m_step), ("replicated", True)),
            )
            markov = gemm_op(
                op_id=f"{prefix}markov_gemm_{m_step}", deps=(lookup.op_id,), stage=stage, layer=-1,
                requests=requests, m=step_seqs, n=draft_vocab, k=draft.markov_rank,
                quant=draft_model.quant, category="lm_head", phase=phase,
                family="logits", recipe="draft_markov_head",
                metadata=(("markov_step", m_step), ("disable_tp", True), ("replicated", True)),
            )
            correction = vector_op(
                op_id=f"{prefix}markov_bias_{m_step}", deps=(base_logits_id, markov.op_id),
                stage=stage, layer=-1, requests=requests, elements=step_seqs * draft_vocab,
                ops_per_element=1, dtype="fp32", family="elementwise", phase=phase,
                recipe="markov_bias", metadata=(("markov_step", m_step),),
            )
            ops.extend((lookup, markov, correction))
            cur = correction.op_id
            if probabilistic and draft_vocab < target_vocab:
                scatter = vector_op(
                    op_id=f"{prefix}d2t_scatter_{m_step}", deps=(cur,), stage=stage, layer=-1,
                    requests=requests, elements=step_seqs * target_vocab,
                    ops_per_element=_DRAFT_D2T_SCATTER_OPS_PER_ELEMENT, dtype="fp32",
                    family="elementwise", phase=phase, recipe="draft_d2t_scatter",
                    shape=(step_seqs, target_vocab),
                    metadata=(("draft_vocab_size", draft_vocab), ("target_vocab_size", target_vocab)),
                )
                ops.append(scatter)
                cur = scatter.op_id
            if run_confidence:
                conf_width = int(draft.confidence_input_width or (draft.hidden_size + draft.markov_rank))
                conf = vector_op(
                    op_id=f"{prefix}confidence_{m_step}", deps=(cur,), stage=stage, layer=-1,
                    requests=requests, elements=step_seqs * conf_width,
                    ops_per_element=2, dtype=model.activation_dtype, family="confidence",
                    phase=phase, recipe="draft_confidence",
                )
                ops.append(conf)
                cur = conf.op_id
            samp_vocab = target_vocab if (probabilistic and draft_vocab < target_vocab) else draft_vocab
            sample = vector_op(
                op_id=f"{prefix}sample_{m_step}", deps=(cur,), stage=stage, layer=-1,
                requests=requests, elements=step_seqs * samp_vocab,
                ops_per_element=_DRAFT_SAMPLING_OPS_PER_ELEMENT, dtype="fp32",
                family="sampling", phase=phase, recipe="draft_sample",
                metadata=(("markov_step", m_step), ("sample_method", sample_method)),
            )
            ops.append(sample)
            prev_sample = sample.op_id
        return prev_sample or base_logits_id
    sample = vector_op(
        op_id=prefix + "sample", deps=(base_logits_id,), stage=stage, layer=-1,
        requests=requests, elements=base_rows * draft_vocab,
        ops_per_element=_DRAFT_SAMPLING_OPS_PER_ELEMENT, dtype="fp32",
        family="sampling", phase=phase, recipe="draft_sample",
    )
    ops.append(sample)
    return sample.op_id


def _append_ar_decode_head(
    ops, frontend, model, draft, step_model, draft_model,
    selected_runtime, proposal_views, previous, *,
    prefix, stage, requests, body_input_rows, phase="draft",
    normalized: bool = False,
):
    sampled_views = tuple(
        v for v in proposal_views
        if not getattr(v, "padding", False) and bool(getattr(v, "samples", True))
    )
    selected_rows = len(sampled_views)
    sampled_requests = tuple(v.request_id for v in sampled_views)
    heads, previous = frontend.logits(
        step_model, selected_runtime, proposal_views, previous,
        prefix=prefix, mode=phase, stage=stage, sample=False,
        input_tokens=body_input_rows,
        normalized=normalized,
        selected_logit_rows=selected_rows,
    )
    ops.extend(heads)
    if not selected_rows:
        return previous
    draft_vocab = int(draft.vocab_size or draft_model.vocab_size)
    target_vocab = int(model.vocab_size)
    d2t_dropped = bool(draft.kind == "eagle3" and draft_vocab == target_vocab)
    if draft.kind in {"eagle", "eagle3"} and draft_vocab < target_vocab:
        scratch = vector_op(
            op_id=prefix + "d2t_scatter", deps=(previous,), stage=stage, layer=-1,
            requests=sampled_requests, elements=selected_rows * target_vocab,
            ops_per_element=_DRAFT_D2T_SCATTER_OPS_PER_ELEMENT, dtype="fp32",
            family="elementwise", phase=phase, recipe="draft_d2t_scatter",
            shape=(selected_rows, target_vocab),
            read_bytes_override=selected_rows * draft_vocab * _DRAFT_FP32_BYTES,
            write_bytes_override=selected_rows * target_vocab * _DRAFT_FP32_BYTES,
            metadata=(
                ("draft_vocab_size", draft_vocab),
                ("target_vocab_size", target_vocab),
                ("d2t_dropped", False),
            ),
        )
        ops.append(scratch)
        previous = scratch.op_id
    sample = vector_op(
        op_id=prefix + "sample", deps=(previous,), stage=stage, layer=-1,
        requests=sampled_requests, elements=selected_rows * (target_vocab if draft_vocab < target_vocab else draft_vocab),
        ops_per_element=_DRAFT_SAMPLING_OPS_PER_ELEMENT, dtype="fp32",
        family="sampling", phase=phase, recipe="draft_sample",
        metadata=(("d2t_dropped", d2t_dropped),),
    )
    ops.append(sample)
    return sample.op_id


def _proposal_view(view, *, prefill: bool, serial: bool, step: int, draft_cap: int,
                   sample_from_anchor: bool = True, next_k: int | None = None,
                   multi_module: bool = False):
    if prefill:
        q_len = int(view.q_len)
        raw_ctx = int(view.context)
    elif serial:
        if step == 0 or multi_module:
            q_len = max(int(view.q_len), 1 + int(view.draft_proposal_tokens))
            raw_ctx = int(view.context)
        else:
            q_len = 1
            advance = int(getattr(view, "draft_advanced_tokens", 0) or 0)
            if advance <= 0:
                advance = 1 + int(view.draft_proposal_tokens)
            raw_ctx = int(view.context) + advance + (step - 1)
    else:
        eff_k = int(next_k) if (next_k is not None and int(next_k) > 0) else int(view.draft_proposal_tokens)
        q_len = eff_k + (0 if sample_from_anchor else 1)
        raw_ctx = int(view.context)
    eff_ctx = min(raw_ctx, max(0, draft_cap - q_len)) if draft_cap > 0 else raw_ctx
    return replace(
        view, q_len=q_len, samples=not prefill,
        phase="prefill" if prefill else "decode",
        context=eff_ctx, draft_verify=False, draft_proposal_tokens=0,
    )


def _draft_step_runtime(runtime, draft, proposal_views, global_proposals, rows, *,
                        replay_scope, external_context, draft_model=None,
                        keep_draft_sp: bool = False, step: int = 0, decode_query_len: int = 1):
    meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_sizes = meta.get("decode_request_sizes")
    req_sizes = (
        tuple(int(s) for s in raw_sizes)
        if isinstance(raw_sizes, (tuple, list)) and raw_sizes else None
    )
    cap_mode, eligible_sizes = resolve_ar_draft_capture_mode(
        runtime, draft, step=step, decode_query_len=decode_query_len,
        decode_request_sizes=req_sizes,
    )
    if req_sizes is not None and len(proposal_views) > 0 and len(proposal_views) not in eligible_sizes and cap_mode in {"full", "full_decode_only", "full_and_piecewise"}:
        cap_mode = "piecewise" if cap_mode == "full_and_piecewise" else "none"
    eff_scope = "none" if cap_mode == "none" else ("piecewise" if cap_mode == "piecewise" else replay_scope)
    selected_runtime, selections = resolve(
        runtime, proposal_views, padded_tokens=rows, replay_scope=eff_scope,
        global_views=global_proposals, model_scope="draft")
    from mindie_sim.prediction.frontend import mark_analytical_runtime
    selected_runtime = mark_analytical_runtime(selected_runtime, selections)
    if external_context and draft.attention_kind == "mla":
        selected_runtime = _independent_draft_runtime(
            selected_runtime, selections, draft_model=draft_model)
    elif cap_mode != str(getattr(selected_runtime, "graph_capture_mode", "none") or "none").lower():
        selected_runtime = replace(selected_runtime, graph_capture_mode=cap_mode)
    if external_context and not draft.dcp_sharded and selected_runtime.parallel.cp > 1:
        selected_runtime = replace(
            selected_runtime,
            parallel=replace(selected_runtime.parallel, cp=1),
        )
    if not keep_draft_sp and "sequence_parallel" in selected_runtime.kernels:
        selected_runtime = replace(
            selected_runtime,
            kernels={k: v for k, v in selected_runtime.kernels.items()
                     if k != "sequence_parallel"})
    return selected_runtime, selections, eff_scope


def _transfer_draft_proposals(
    regions, previous, views, runtime, *, prefill, last_active,
    widths: Mapping[str, int] | None = None,
):
    return append_draft_proposal_transfer(
        regions,
        producers=((views, bool(prefill), last_active),),
        runtime=runtime,
        previous=previous,
        widths=widths,
    )


def append_draft_proposal_transfer(
    regions: list[LoweringRegion],
    *,
    producers: Sequence[tuple[Sequence[Any], bool, tuple[int, str] | None]],
    runtime,
    previous: str,
    widths: Mapping[str, int] | None = None,
) -> str:
    """Append one aggregated PP>1 draft_proposal_transfer across active prefill/decode producers."""
    if runtime.parallel.pp <= 1:
        return previous
    width_map = dict(widths or {})

    def _generated_k(view) -> int:
        if str(view.request_id) in width_map:
            return max(0, int(width_map[str(view.request_id)] or 0))
        return max(0, int(view.draft_proposal_tokens or 0))

    deps: list[str] = []
    proposal_views_local: list[Any] = []
    target_region_index: int | None = None
    for views, is_prefill, last_active in producers:
        if last_active is None:
            continue
        local = [
            view for view in views
            if not getattr(view, "padding", False)
            and (not is_prefill or bool(getattr(view, "samples", False)))
            and _generated_k(view) > 0
        ]
        if not local:
            continue
        region_index, dependency = last_active
        deps.append(dependency)
        proposal_views_local.extend(local)
        target_region_index = region_index
    if target_region_index is None or not proposal_views_local:
        return previous
    total_proposals = sum(_generated_k(view) for view in proposal_views_local)
    xfer = comm_op(
        op_id="draft_proposal_transfer",
        deps=tuple(dict.fromkeys(deps)),
        stage=runtime.parallel.pp - 1,
        layer=-1,
        requests=tuple(view.request_id for view in proposal_views_local),
        family="pp_p2p",
        phase="draft",
        recipe="pp_p2p",
        link_bytes=total_proposals * PROPOSAL_ID_BYTES,
        shape=(total_proposals,),
        metadata=(
            ("group_axis", "pp"),
            ("source_stage", runtime.parallel.pp - 1),
            ("destination_stage", 0),
        ),
    )
    xfer = _tag_draft_ops((xfer,), {
        view.request_id: _generated_k(view)
        for view in proposal_views_local
    })[0]
    region = regions[target_region_index]
    regions[target_region_index] = replace(region, ops=region.ops + (xfer,))
    return xfer.op_id


def _sparse_draft_metadata(draft, plans) -> tuple[tuple[str, object], ...]:
    extra: list[tuple[str, object]] = []
    if draft.attention_kind == "gqa" and any(
        bool(getattr(p, "index_kpool_compress", False)) and int(getattr(p, "index_kpool", 1) or 1) == QSA_INDEX_GROUP_TOKENS
        for p in plans
    ):
        extra.append(("qsa_index_buffer_cols", QSA_INDEX_BUFFER_COLS))
    if draft.attention_kind == "mla" and any(int(getattr(p, "index_kpool", 1) or 1) > 1 for p in plans):
        extra.append(("kpool_tail_aligned", KPOOL_TAIL_ALIGNMENT))
    return tuple(extra)


def _append_dspark_prefill_propose(
    regions, previous, *, frontend, model, draft, draft_model, runtime, boot_views,
    widths, sample_from_anchor, draft_cap, replay_scope, external_context,
    stage, replica, machine, spec_method,
):
    per_req_ks = tuple(
        _draft_spec_mod.validate_fia_speculative_tokens(int(widths[str(v.request_id)]))
        for v in boot_views
    )
    boot_k = max(per_req_ks, default=0)
    b_proposals = tuple(
        replace(
            v, q_len=req_k + (0 if sample_from_anchor else 1),
            context=(
                min(
                    int(v.context) + int(v.global_q_len or v.q_len),
                    max(0, draft_cap - (req_k + (0 if sample_from_anchor else 1))),
                )
                if draft_cap > 0 else (int(v.context) + int(v.global_q_len or v.q_len))
            ),
            samples=True, phase="decode", draft_verify=False, draft_proposal_tokens=0,
        )
        for v, req_k in zip(boot_views, per_req_ks)
    )
    b_valid = sum(v.q_len for v in b_proposals)
    b_rt, b_sels, b_scope = _draft_step_runtime(
        runtime, draft, b_proposals, b_proposals, b_valid,
        replay_scope=replay_scope, external_context=external_context,
        draft_model=draft_model, keep_draft_sp=False, step=0, decode_query_len=b_valid,
    )
    b_prefix = "draft_prefill_propose."
    b_ops_raw, previous = frontend.body(
        draft_model, b_rt, b_proposals, prefix=b_prefix,
        ready=previous, tokens=b_valid, mode="draft",
        stages=(stage,) * draft_model.layers, global_tokens=b_valid,
        replica=replica, machine=machine,
    )
    b_ops = list(b_ops_raw)
    previous = _append_dspark_decode_head(
        b_ops, frontend, model, draft, draft_model, draft_model,
        runtime, b_rt, b_proposals, previous,
        prefix=b_prefix, stage=stage, requests=tuple(v.request_id for v in b_proposals),
        body_input_rows=b_valid, k_steps=boot_k,
        spec_method=spec_method, hc_collapsed=False, phase="draft",
        base_rows_override=sum(per_req_ks), per_req_ks=per_req_ks,
    )
    regions.append(LoweringRegion(
        _tag_draft_ops(b_ops, {v.request_id: v.q_len for v in b_proposals}, step=0),
        b_rt, draft_model.activation_dtype, b_valid, b_sels, replay_scope=b_scope,
    ))
    return previous


def _lower_dspark_prefill_program(
    frontend, model, draft, draft_model, runtime, active, ready, *,
    external_context: bool, context_prolog_v3: bool, eager_draft: bool,
    slot_mapping_none: bool, sample_from_anchor: bool, draft_cap: int,
    replica: int, replay_scope: str, machine, spec_method: str,
    assembly: str, widths: Mapping[str, int],
):
    """PCP=1 prompt-init context-KV prepare + optional service prefill-propose for DSpark."""
    stage = runtime.parallel.pp - 1
    proposal_views = tuple(
        replace(v, q_len=int(v.global_q_len or v.q_len), phase="prefill",
                samples=False, draft_verify=False, draft_proposal_tokens=0)
        for v in active
    )
    feature_rows = sum(v.q_len for v in proposal_views)
    requests = tuple(v.request_id for v in proposal_views)
    selected_runtime, selections, eff_scope = _draft_step_runtime(
        runtime, draft, proposal_views, proposal_views, feature_rows,
        replay_scope=replay_scope, external_context=external_context,
        draft_model=draft_model, keep_draft_sp=False, step=0, decode_query_len=feature_rows,
    )
    ops, previous = lower_dspark_context_kv(
        model, draft, draft_model, selected_runtime,
        previous=ready, prefix="draft_prefill.", stage=stage,
        requests=requests, feature_rows=feature_rows, query_rows=0,
        proposal_views=proposal_views, prefill=True,
        external_context=external_context, context_prolog_v3=context_prolog_v3,
        eager_draft=eager_draft, slot_mapping_none=slot_mapping_none,
        machine=machine,
    )
    rows_by_request = {v.request_id: v.q_len for v in proposal_views}
    regions = [LoweringRegion(
        _tag_draft_ops(ops, rows_by_request),
        selected_runtime, draft_model.activation_dtype, feature_rows, selections,
        replay_scope=eff_scope,
    )]
    boot_views = tuple(
        v for v in active
        if v.samples and int(widths.get(str(v.request_id), 0) or 0) > 0
    )
    last_active = None
    if assembly == "service" and boot_views and not slot_mapping_none:
        previous = _append_dspark_prefill_propose(
            regions, previous, frontend=frontend, model=model, draft=draft,
            draft_model=draft_model, runtime=runtime, boot_views=boot_views,
            widths=widths, sample_from_anchor=sample_from_anchor, draft_cap=draft_cap,
            replay_scope=replay_scope, external_context=external_context,
            stage=stage, replica=replica, machine=machine, spec_method=spec_method,
        )
        last_active = (len(regions) - 1, previous)
    return regions, previous, last_active


def _lower_dspark_draft_program(
    frontend, model, draft, draft_model, runtime, active, all_active, ready, *,
    prefill: bool, external_context: bool, context_prolog_v3: bool,
    replica: int, replay_scope: str, machine, spec_method: str,
    assembly: str = "service", next_proposal_widths: Mapping[str, int] | None = None,
):
    stage = runtime.parallel.pp - 1
    draft_cap = int(_draft_spec_mod.effective_draft_context_cap(draft, runtime))
    eager_draft = _draft_enforce_eager(runtime)
    meta = dict(getattr(runtime, "metadata", ()) or ())
    slot_mapping_none = bool(meta.get("dspark_context_slot_mapping_none", False))
    widths = dict(next_proposal_widths or {})
    sample_from_anchor = _dspark_sample_from_anchor(draft, runtime, spec_method)
    if prefill:
        return _lower_dspark_prefill_program(
            frontend, model, draft, draft_model, runtime, active, ready,
            external_context=external_context, context_prolog_v3=context_prolog_v3,
            eager_draft=eager_draft, slot_mapping_none=slot_mapping_none,
            sample_from_anchor=sample_from_anchor, draft_cap=draft_cap,
            replica=replica, replay_scope=replay_scope, machine=machine,
            spec_method=spec_method, assembly=assembly, widths=widths,
        )
    k_steps = max(
        (int(widths.get(str(v.request_id), 0) or v.draft_proposal_tokens) for v in all_active),
        default=0,
    )
    if k_steps <= 0:
        return [], _last_ready(ready), None
    k_steps = _draft_spec_mod.validate_fia_speculative_tokens(k_steps)
    def _req_k(v) -> int:
        return int(widths.get(str(v.request_id), 0) or k_steps)
    proposal_views = tuple(
        _proposal_view(v, prefill=False, serial=False, step=0, draft_cap=draft_cap,
                       sample_from_anchor=sample_from_anchor, next_k=_req_k(v))
        for v in active
    )
    global_proposals = tuple(
        _proposal_view(v, prefill=False, serial=False, step=0, draft_cap=draft_cap,
                       sample_from_anchor=sample_from_anchor, next_k=_req_k(v))
        for v in all_active
    )
    q_valid = sum(v.q_len for v in proposal_views)
    feature_rows = sum(max(int(v.q_len), 1 + int(v.draft_proposal_tokens)) for v in active)
    use_draft_sp = _draft_sp_enabled(runtime, draft, is_dspark=True)
    tp = _resolve_draft_tp(runtime, draft)
    eff_cap_mode, _ = resolve_ar_draft_capture_mode(
        runtime, draft, step=0, decode_query_len=max(1, q_valid),
    )
    capture_eligible = (
        not eager_draft
        and replay_scope != "none"
        and eff_cap_mode not in {"none", "eager"}
    )
    b_cap = _draft_capture_bucket(runtime, q_valid) if capture_eligible else 0
    dspark_contract = _draft_spec_mod.SpeculativeExecutionContract(
        archetype="dspark",
        spec_method=spec_method if spec_method in _DSPARK_SPEC_METHODS else "dspark",
        num_speculative_tokens=k_steps,
        requires_draft_spec=True,
        neural_draft_forwards=1,
        requires_draft_prepare=True,
        draft_tp=tp,
        sample_from_anchor=sample_from_anchor,
    )
    schedule_views = tuple(
        replace(v, next_draft_proposal_tokens=_req_k(v))
        for v in active
    )
    schedule = dspark_contract.build_step_schedule(schedule_views, capture_bucket=b_cap)
    q_phys = (
        schedule.physical_draft_sp_rows[0]
        if (use_draft_sp and schedule.physical_draft_sp_rows and schedule.body_q_len_per_step == (q_valid,))
        else (tp * math.ceil(max(q_valid, b_cap) / tp) if use_draft_sp else max(q_valid, b_cap))
    )
    hc_collapse_before = _hc_collapse_before_sp_gather(draft, runtime)
    selected_runtime, selections, eff_scope = _draft_step_runtime(
        runtime, draft, proposal_views, global_proposals, q_phys,
        replay_scope=replay_scope, external_context=external_context,
        draft_model=draft_model, keep_draft_sp=use_draft_sp, step=0, decode_query_len=q_phys,
    )
    requests = tuple(v.request_id for v in proposal_views)
    prefix = "draft0."
    ops, body_ready = lower_dspark_context_kv(
        model, draft, draft_model, selected_runtime,
        previous=ready, prefix=prefix, stage=stage,
        requests=requests, feature_rows=feature_rows, query_rows=q_valid,
        proposal_views=active, prefill=False,
        external_context=external_context, context_prolog_v3=context_prolog_v3,
        eager_draft=eager_draft, slot_mapping_none=slot_mapping_none,
        machine=machine,
    )
    global_tokens = sum(v.q_len for v in global_proposals)
    body, previous = frontend.body(
        draft_model, selected_runtime, proposal_views, prefix=prefix,
        ready=body_ready, tokens=q_phys, mode="draft",
        stages=(stage,) * draft_model.layers, global_tokens=max(global_tokens, q_phys),
        replica=replica, machine=machine,
        hc_collapse_before_sp_gather=hc_collapse_before,
    )
    ops.extend(body)
    per_req_ks = tuple(_req_k(v) for v in active)
    did_early_hc_collapse = bool(
        hc_collapse_before and any(op.op_id == prefix + "sp_output_gather" for op in body)
    )
    previous = _append_dspark_decode_head(
        ops, frontend, model, draft, draft_model, draft_model,
        runtime, selected_runtime, proposal_views, previous,
        prefix=prefix, stage=stage, requests=requests,
        body_input_rows=q_phys, k_steps=k_steps,
        spec_method=spec_method, hc_collapsed=did_early_hc_collapse,
        base_rows_override=sum(per_req_ks), per_req_ks=per_req_ks,
    )
    rows_by_request = {v.request_id: v.q_len for v in proposal_views}
    region = LoweringRegion(
        _tag_draft_ops(
            ops, rows_by_request, step=0,
            extra_meta=(
                ("dspark_feature_rows", feature_rows),
                ("dspark_logical_query_rows", q_valid),
                ("dspark_physical_rows", q_phys),
            ),
        ),
        selected_runtime, draft_model.activation_dtype, q_phys, selections,
        replay_scope=eff_scope,
    )
    last_active = (0, previous) if q_valid else None
    return [region], previous, last_active


def _restore_draft_pcp_hidden(
    frontend, ops, previous, *, step_model, selected_runtime, proposal_views,
    rows: int, prefix: str, stage: int, requests: tuple[str, ...], phase_name: str,
) -> tuple[str, int, bool]:
    """Gather and unpermute PCP-sharded draft prefill hidden states before bootstrap logits."""
    from mindie_sim.prediction.frontend.body import _resolve_hc_final_collapse
    from mindie_sim.prediction.sharding.context_parallel import (
        emit_pcp_hidden_gather_restore,
        pcp_uniform_peer_rows,
    )
    per_peer_rows, restored_rows = pcp_uniform_peer_rows(
        proposal_views, selected_runtime.parallel.pcp, local_rows=rows,
    )
    if restored_rows <= 0:
        return previous, rows, False
    pcp_normalized = selected_runtime.final_norm_scope == "all"
    if pcp_normalized:
        finals, previous = frontend.final_mixer(
            step_model, previous, prefix=prefix, mode=phase_name,
            stage=stage, tokens=rows, requests=requests,
        )
        ops.extend(finals)
    hc_mult, collapse_mode = _resolve_hc_final_collapse(step_model)
    has_lowrank = any(getattr(plan, "hc_lowrank", 0) > 0 for plan in step_model.layer_plans)
    gather_hidden = (
        hc_mult * step_model.hidden_size
        if not pcp_normalized and hc_mult > 1 and (has_lowrank or collapse_mode != "none")
        else step_model.hidden_size
    )
    previous = emit_pcp_hidden_gather_restore(
        ops, prefix=prefix.rstrip("."),
        gather_name="pcp_hidden_gather",
        restore_name="pcp_hidden_restore",
        stage=stage, layer=-1, requests=requests, mode=phase_name,
        dep=previous, local_rows=rows, global_rows=restored_rows,
        hidden=gather_hidden, pcp=int(selected_runtime.parallel.pcp),
        act_dtype=step_model.activation_dtype, runtime=selected_runtime,
        source="vllm_ascend.spec_decode.mtp_proposer",
        per_peer_rows=per_peer_rows,
    )
    return previous, restored_rows, pcp_normalized


def _append_prefill_bootstrap(
    regions, ops, previous, *, frontend, model, draft, step_model, draft_model,
    plans, runtime, selected_runtime, proposal_views, selected, all_active, rows,
    prefix, stage, requests, replica, machine, selections, rows_by_request,
    step_extra, bootstrap_k: int, eff_scope: str, replay_scope: str,
    external_context: bool, draft_cap: int, widths: Mapping[str, int],
    cyclic: bool = False, multi_module: bool = False,
    body_input_rows: int | None = None, normalized: bool = False,
):
    """Seed all K >= 1 initial step-1 proposals on a final prompt chunk (samples=True)."""
    def _wants_bootstrap(view) -> bool:
        if not view.samples:
            return False
        if str(view.request_id) in widths:
            return int(widths[str(view.request_id)] or 0) > 0
        return True

    sampled_local = tuple(v for v in selected if _wants_bootstrap(v))
    sampled_global = tuple(v for v in all_active if _wants_bootstrap(v))
    if bootstrap_k < 1 or not sampled_global:
        return previous, None
    last_active = None
    if sampled_local:
        sampled_ids = {v.request_id for v in sampled_local}
        head_views = tuple(
            replace(pv, samples=(pv.request_id in sampled_ids)) for pv in proposal_views
        )
        previous = _append_ar_decode_head(
            ops, frontend, model, draft, step_model, draft_model,
            selected_runtime, head_views,
            previous, prefix=prefix, stage=stage, requests=requests,
            body_input_rows=rows if body_input_rows is None else body_input_rows,
            phase="draft_prefill", normalized=normalized,
        )
        regions[-1] = LoweringRegion(
            _tag_draft_ops(ops, rows_by_request, step=0, extra_meta=step_extra),
            selected_runtime, draft_model.activation_dtype, rows, selections,
            replay_scope=eff_scope,
        )
        last_active = (len(regions) - 1, previous)
    for b_step in range(1, bootstrap_k):
        b_views = tuple(
            replace(
                v, q_len=1,
                context=(
                    min(v.context + v.q_len + (b_step - 1), max(0, draft_cap - 1))
                    if draft_cap > 0 else v.context + v.q_len + (b_step - 1)
                ),
                samples=True, phase="decode", draft_verify=False, draft_proposal_tokens=0,
            )
            for v in sampled_local
        )
        b_global_views = tuple(
            replace(
                v, q_len=1,
                context=(
                    min(v.context + v.q_len + (b_step - 1), max(0, draft_cap - 1))
                    if draft_cap > 0 else v.context + v.q_len + (b_step - 1)
                ),
                samples=True, phase="decode", draft_verify=False, draft_proposal_tokens=0,
            )
            for v in sampled_global
        )
        b_rows = len(b_views)
        b_global = len(b_global_views)
        b_runtime, b_selections, b_scope = _draft_step_runtime(
            runtime, draft, b_views, b_global_views, b_rows,
            replay_scope=replay_scope, external_context=external_context,
            draft_model=draft_model, keep_draft_sp=False, step=b_step, decode_query_len=1,
        )
        b_step_model = (
            _cyclic_draft_model(draft_model, plans, b_step)
            if (cyclic or multi_module) else draft_model
        )
        b_prefix = f"draft_prefill_step{b_step}."
        b_reqs = tuple(v.request_id for v in b_views)
        if draft.kind == "eagle3":
            b_ops, body_ready = [], _last_ready(previous)
        else:
            b_ops, projection = _draft_features(
                model, draft, draft_model, b_step_model, b_runtime,
                previous=previous, prefix=b_prefix, stage=stage, requests=b_reqs,
                rows=b_rows, feature_rows=b_rows, tp=int(b_runtime.parallel.tp), phase="draft",
            )
            body_ready = projection.op_id if b_rows else _last_ready(previous)
        b_ops_raw, previous = frontend.body(
            b_step_model, b_runtime, b_views, prefix=b_prefix,
            ready=body_ready, tokens=b_rows, mode="draft",
            stages=(stage,) * b_step_model.layers, global_tokens=b_global,
            replica=replica, machine=machine,
        )
        b_ops.extend(b_ops_raw)
        if b_rows:
            previous = _append_ar_decode_head(
                b_ops, frontend, model, draft, b_step_model, draft_model,
                b_runtime, b_views, previous, prefix=b_prefix,
                stage=stage, requests=b_reqs,
                body_input_rows=b_rows, phase="draft",
            )
        regions.append(LoweringRegion(
            _tag_draft_ops(b_ops, {v.request_id: 1 for v in b_views}, step=b_step, extra_meta=step_extra),
            b_runtime, draft_model.activation_dtype, b_rows, b_selections,
            replay_scope=b_scope,
        ))
        if b_rows:
            last_active = (len(regions) - 1, previous)
    return previous, last_active


def _build_draft_model(model, draft, plans, runtime, machine, *, external_context: bool):
    draft_model = replace(
        model, name="draft", layers=draft.layers, hidden_size=draft.hidden_size,
        num_heads=draft.num_heads, num_kv_heads=draft.num_kv_heads, head_dim=draft.head_dim,
        intermediate_size=draft.intermediate_size, layer_plans=plans, auxiliary=None,
        expert_routing=(), num_experts=draft.num_experts, top_k=draft.top_k,
        vocab_size=draft.vocab_size or model.vocab_size,
        quant=draft_quant(model, draft), activation_dtype=draft.activation_dtype,
        compute_dtype=draft.activation_dtype,
    )
    context_prolog_v3 = True
    if external_context:
        draft_model, context_prolog_v3 = _draft_cache_model(
            model, draft, draft_model, runtime, machine)
    return draft_model, context_prolog_v3


def _resolve_ar_module_strategy(
    draft, plans, meta: Mapping[str, object], *, steps: int, configured_k: int,
) -> tuple[bool, bool]:
    """Determine structural cyclic_nextn / multi_module_mtp mode and validate configured K."""
    cyclic = bool(draft.execution_mode == "cyclic_nextn" and len(plans) > 1)
    n_predict_meta = int(meta.get("multi_module_mtp_n_predict", 0) or 0)
    multi_module = bool(
        not cyclic
        and draft.kind == "mtp"
        and len(plans) > 1
        and (
            draft.execution_mode == "multi_module_mtp"
            or n_predict_meta > 1
            or min(int(draft.layers), max(configured_k, steps)) > 1
        )
    )
    if multi_module:
        module_count = len(plans)
        n_predict = n_predict_meta or (module_count if draft.execution_mode == "multi_module_mtp" else 0)
        check_k = configured_k if configured_k > 0 else steps
        if check_k > 0 and module_count > 1 and check_k % module_count != 0:
            raise ModelingError(
                f"multi_module_mtp_non_divisible_n_predict:k={check_k}:n_predict={module_count}"
            )
        if n_predict > 0 and check_k > 0 and check_k % n_predict != 0:
            raise ModelingError(
                f"multi_module_mtp_non_divisible_n_predict:k={check_k}:n_predict={n_predict}"
            )
    return cyclic, multi_module


def _lower_ar_draft_steps(
    frontend, model, draft, draft_model, plans, runtime, active, all_active, ready, *,
    prefill: bool, external_context: bool, replica: int, replay_scope: str,
    machine, assembly: str, widths: Mapping[str, int],
    contract: _draft_spec_mod.SpeculativeExecutionContract | None = None,
    physical_rank=None,
):
    """[Cohesive-Pipeline] Lower serial MTP/EAGLE/EAGLE3 draft steps (including prefill bootstrap and multi-module MTP)."""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if prefill:
        steps = 1
    elif contract is not None:
        sched_views = tuple(
            replace(
                v,
                next_draft_proposal_tokens=(
                    int(widths[str(v.request_id)])
                    if str(v.request_id) in widths else getattr(v, "next_draft_proposal_tokens", None)
                ),
            )
            for v in all_active
        )
        steps = contract.build_step_schedule(sched_views).num_steps
    else:
        steps = max(
            (int(widths.get(str(v.request_id), 0) or v.draft_proposal_tokens) for v in all_active),
            default=1,
        )
    configured_k = int(meta.get("draft_proposal_tokens", meta.get("num_speculative_tokens", 0)) or 0)
    if configured_k <= 0:
        configured_k = steps
    if not prefill and steps > 0:
        _draft_spec_mod.validate_fia_speculative_tokens(steps)
    cyclic, multi_module = _resolve_ar_module_strategy(
        draft, plans, meta, steps=steps, configured_k=configured_k,
    )
    regions, previous = [], ready
    stage = runtime.parallel.pp - 1
    share_selection = bool(
        (draft.index_share_for_mtp_iteration
         or any(getattr(plan, "index_share_for_mtp_iteration", False) for plan in plans))
        and not prefill and draft.attention_kind in {"mla", "gqa"}
    )
    mtp_topks: dict[int, str] = {}
    tp = runtime.parallel.tp
    draft_cap = int(_draft_spec_mod.effective_draft_context_cap(draft, runtime))
    sparse_meta = _sparse_draft_metadata(draft, plans)
    last_active = None
    for step in range(steps):
        # V2 AR runs ALL K steps across the full proposing batch (never dropping requests on p_i <= step).
        selected = active
        proposal_views = tuple(
            _proposal_view(v, prefill=prefill, serial=not prefill, step=step, draft_cap=draft_cap,
                           multi_module=multi_module)
            for v in selected
        )
        global_proposals = tuple(
            _proposal_view(v, prefill=prefill, serial=not prefill, step=step, draft_cap=draft_cap,
                           multi_module=multi_module)
            for v in all_active
        )
        rows = sum(v.q_len for v in proposal_views)
        decode_q_len = max((v.q_len for v in proposal_views), default=1)
        selected_runtime, selections, eff_scope = _draft_step_runtime(
            runtime, draft, proposal_views, global_proposals, rows,
            replay_scope=replay_scope, external_context=external_context,
            draft_model=draft_model, keep_draft_sp=False, step=step, decode_query_len=decode_q_len,
        )
        step_model = (
            _cyclic_draft_model(draft_model, plans, step)
            if (cyclic or multi_module) else draft_model
        )
        pcp_prefill = bool(prefill and selected_runtime.parallel.pcp > 1)
        if pcp_prefill:
            from mindie_sim.prediction.execution.rank_ledger import selected_lane
            proposal_views, rows, _ = frontend.partition_views(
                proposal_views, selected_runtime, rows, model=step_model,
                lane=selected_lane(selected_runtime.parallel, machine, replica, physical_rank))
        requests = tuple(v.request_id for v in proposal_views)
        prefix = "draft_prefill." if prefill else f"draft{step}."
        phase_name = "draft_prefill" if prefill else "draft"
        if draft.kind == "eagle3" and not prefill and step > 0:
            ops, body_ready = [], _last_ready(previous)
        else:
            ops, projection = _draft_features(
                model, draft, draft_model, step_model, runtime, previous=previous, prefix=prefix,
                stage=stage, requests=requests, rows=rows, feature_rows=rows, tp=tp, phase=phase_name)
            body_ready = projection.op_id if rows else _last_ready(previous)
        global_tokens = sum(v.q_len for v in global_proposals)
        if pcp_prefill:
            global_tokens = frontend.pcp_cohort_tokens(
                global_proposals, proposal_views, rows, selected_runtime,
                global_tokens=global_tokens, global_views=global_proposals)
        can_reuse_topk = bool(
            share_selection and not prefill and step > 0
            and all(p.index in mtp_topks for p in step_model.layer_plans if p.indexer)
        )
        body, previous = frontend.body(
            step_model, selected_runtime, proposal_views, prefix=prefix,
            ready=body_ready, tokens=rows, mode=phase_name,
            stages=(stage,) * step_model.layers, global_tokens=global_tokens,
            replica=replica, machine=machine,
            topk_scope=mtp_topks if (share_selection and not prefill) else None,
            reuse_index_selection=can_reuse_topk,
        )
        ops.extend(body)
        body_output_rows, pcp_normalized = rows, False
        if pcp_prefill:
            previous, body_output_rows, pcp_normalized = _restore_draft_pcp_hidden(
                frontend, ops, previous, step_model=step_model,
                selected_runtime=selected_runtime, proposal_views=proposal_views,
                rows=rows, prefix=prefix, stage=stage, requests=requests,
                phase_name=phase_name,
            )
        step_extra = sparse_meta + (
            (("spec_module_idx", step % len(plans)), ("stale_kv_reprefill", step > 0))
            if multi_module else ()
        )
        if prefill:
            rows_by_request = {v.request_id: v.q_len for v in proposal_views}
            regions.append(LoweringRegion(
                _tag_draft_ops(ops, rows_by_request, step=0, extra_meta=step_extra),
                selected_runtime, draft_model.activation_dtype, rows, selections,
                replay_scope=eff_scope))
            if assembly == "service":
                bootstrap_k = max(
                    (int(widths.get(str(v.request_id), 0) or 0) for v in all_active if v.samples),
                    default=int(meta.get("draft_proposal_tokens", 0) or 0),
                )
                if bootstrap_k > 0:
                    previous, last_active = _append_prefill_bootstrap(
                        regions, ops, previous, frontend=frontend, model=model, draft=draft,
                        step_model=step_model, draft_model=draft_model,
                        plans=plans, runtime=runtime,
                        selected_runtime=selected_runtime, proposal_views=proposal_views,
                        selected=selected, all_active=all_active, rows=rows, prefix=prefix,
                        stage=stage, requests=requests, replica=replica, machine=machine,
                        selections=selections, rows_by_request=rows_by_request,
                        step_extra=step_extra, bootstrap_k=bootstrap_k, eff_scope=eff_scope,
                        replay_scope=replay_scope, external_context=external_context,
                        draft_cap=draft_cap, widths=widths,
                        cyclic=cyclic, multi_module=multi_module,
                        body_input_rows=body_output_rows, normalized=pcp_normalized,
                    )
            continue
        previous = _append_ar_decode_head(
            ops, frontend, model, draft, step_model, draft_model,
            selected_runtime, proposal_views, previous, prefix=prefix,
            stage=stage, requests=requests, body_input_rows=rows, phase="draft",
        )
        rows_by_request = {v.request_id: v.q_len for v in proposal_views}
        regions.append(LoweringRegion(
            _tag_draft_ops(ops, rows_by_request, step=step, extra_meta=step_extra),
            selected_runtime, draft_model.activation_dtype, rows, selections,
            replay_scope=eff_scope))
        if rows:
            last_active = (len(regions) - 1, previous)
    return regions, previous, last_active


def _enforce_draft_execution_contracts(
    runtime, draft, *, spec_method: str, explicit_method: str, max_k: int = 0,
) -> _draft_spec_mod.SpeculativeExecutionContract:
    """Enforce locked simulation synthetic-draft policy and Ascend DCP precedence."""
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    spine = rt_meta.get("single_spine")
    spine_map = spine if isinstance(spine, Mapping) else {}
    scen = spine_map.get("scenario_contract") if isinstance(spine_map.get("scenario_contract"), Mapping) else {}
    spec_pol = scen.get("speculative_policy") if isinstance(scen.get("speculative_policy"), Mapping) else {}
    raw_mode = str(rt_meta.get("optimization_mode") or scen.get("optimization_mode") or "").strip().lower()
    if not raw_mode:
        stack_track = str(spine_map.get("stack_track") or "").strip().lower()
        raw_mode = "modeling" if stack_track == "target" else "simulation"
    opt_mode = "modeling" if raw_mode == "modeling" else "simulation"
    allow_synth = (
        rt_meta.get("allow_synthetic_gqa_draft") is True
        or spec_pol.get("allow_synthetic_gqa") is True
    )
    is_synth = bool(getattr(draft, "is_synthetic", False))
    draft_kind = str(getattr(draft, "kind", "mtp") or "mtp").strip().lower()
    raw_eff = explicit_method or spec_method or draft_kind
    archetype = (
        "dspark" if _is_dspark_block_draft(draft, raw_eff)
        else ("eagle3" if raw_eff in {"eagle", "eagle3"} or draft_kind in {"eagle", "eagle3"} else "mtp")
    )
    if is_synth and opt_mode == "modeling" and allow_synth:
        eff_method = "eagle3"
    elif (
        explicit_method
        and explicit_method in _draft_spec_mod.SUPPORTED_SPECULATIVE_METHODS
        and explicit_method not in {"none", "ngram", "ngram_gpu"}
    ):
        eff_method = explicit_method
    elif archetype == "dspark":
        eff_method = "dspark"
    elif draft_kind in {"eagle", "eagle3"}:
        eff_method = draft_kind
    else:
        eff_method = "mtp"
    cfg_k = max(
        int(rt_meta.get("draft_proposal_tokens", rt_meta.get("num_speculative_tokens", 0)) or 0),
        int(max_k or 0),
    )
    draft_tp = _resolve_draft_tp(runtime, draft)
    use_synth_opt_in = bool(is_synth and opt_mode == "modeling" and allow_synth)
    provenance = (
        _draft_spec_mod.SYNTHETIC_EAGLE3_GQA_PROVENANCE
        if use_synth_opt_in else _draft_spec_mod.REAL_CHECKPOINT_PROVENANCE
    )
    contract = _draft_spec_mod.SpeculativeExecutionContract(
        archetype=archetype,
        spec_method=eff_method,
        num_speculative_tokens=max(1, cfg_k),
        requires_draft_spec=True,
        draft_tp=draft_tp,
        optimization_mode=opt_mode,  # type: ignore[arg-type]
        allow_synthetic_gqa=use_synth_opt_in,
        draft_provenance=provenance,
    )
    parallel = getattr(runtime, "parallel", None)
    if parallel is not None:
        contract.validate_dcp_spec_gate(draft, parallel, draft_tp_override=draft_tp)
    else:
        contract.require_draft_for_execution(draft)
    return contract


def lower_draft_regions(
    frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
    replica=0, replay_scope="none", machine=None, assembly="service",
    next_proposal_widths: Mapping[str, int] | None = None, physical_rank=None,
):
    """Lower draft regions without emitting the PP proposal transfer yet."""
    rt_meta = dict(runtime.metadata)
    explicit_method = str(rt_meta.get("spec_method") or "").strip().lower()
    if explicit_method == "none":
        return [], _last_ready(ready), None
    if explicit_method == "medusa":
        raise ModelingError("unsupported_speculative_method_on_ascend_v2:medusa")
    if explicit_method and not _draft_spec_mod.is_supported_speculative_method(explicit_method, allow_none=True):
        raise ModelingError(f"unsupported_speculative_method:{explicit_method}")
    spec_method = _spec_method(runtime)
    if spec_method in _NON_NEURAL_SPEC_METHODS:
        return [], _last_ready(ready), None
    kind = rt_meta.get("draft_mode", spec_method or "mtp")
    draft = optional_draft_spec(model, kind)
    if draft is None:
        if explicit_method:
            raise ModelingError(f"neural_speculative_method_requires_draft_spec:{explicit_method}")
        return [], _last_ready(ready), None
    if explicit_method:
        _validate_method_draft_compatibility(explicit_method, draft)
    widths = dict(next_proposal_widths or {})

    def selected_view(view):
        if prefill:
            return view.phase == "prefill"
        return int(widths.get(str(view.request_id), view.draft_proposal_tokens) or 0) > 0

    active = tuple(view for view in views if selected_view(view))
    all_active = tuple(view for view in (global_views or views) if selected_view(view))
    if not active and not all_active:
        return [], _last_ready(ready), None
    max_req_k = max(
        (int(widths.get(str(v.request_id), v.draft_proposal_tokens) or 0) for v in all_active),
        default=0,
    )
    contract = _enforce_draft_execution_contracts(
        runtime, draft, spec_method=spec_method, explicit_method=explicit_method, max_k=max_req_k,
    )
    from mindie_sim.catalog.paired_sfa import independent_draft
    external_context = independent_draft(draft)
    plans = draft_layer_plans(model, kind)
    draft_model, context_prolog_v3 = _build_draft_model(
        model, draft, plans, runtime, machine, external_context=external_context,
    )
    if _is_dspark_block_draft(draft, spec_method):
        return _lower_dspark_draft_program(
            frontend, model, draft, draft_model, runtime, active, all_active, ready,
            prefill=prefill, external_context=external_context,
            context_prolog_v3=context_prolog_v3, replica=replica,
            replay_scope=replay_scope, machine=machine, spec_method=spec_method,
            assembly=assembly, next_proposal_widths=widths,
        )
    return _lower_ar_draft_steps(
        frontend, model, draft, draft_model, plans, runtime, active, all_active, ready,
        prefill=prefill, external_context=external_context, replica=replica,
        replay_scope=replay_scope, machine=machine, assembly=assembly, widths=widths,
        contract=contract, physical_rank=physical_rank,
    )


def draft_program(frontend, model, runtime, views, ready, *, global_views=None, prefill=False,
                  replica=0, replay_scope="none", machine=None, assembly="service",
                  next_proposal_widths: Mapping[str, int] | None = None, physical_rank=None):
    regions, previous, last_active = lower_draft_regions(
        frontend, model, runtime, views, ready, global_views=global_views,
        prefill=prefill, replica=replica, replay_scope=replay_scope, machine=machine,
        assembly=assembly, next_proposal_widths=next_proposal_widths,
        physical_rank=physical_rank,
    )
    if regions:
        previous = _transfer_draft_proposals(
            regions, previous, views, runtime, prefill=prefill, last_active=last_active,
            widths=dict(next_proposal_widths or {}),
        )
    return regions, previous
