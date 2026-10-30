"""Explicit SCFA source selection for indexed, compressed shared KV.

[Role] Connect the reviewed SCFA resource program to ordinary MLA construction.
[Invariants] Only an actual compressed-index producer selects SCFA. SWA/CFA
  retain their separate analytic paths. Source metadata, ownership and cache
  residency are declarations; neither a model name nor a duration selects them.
[Timeouts & Locks] Constant metadata per selected operator; no I/O or locks.
[Side-Effect Policy] Returns immutable operators and recipes; no device work.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.catalog.cache_format import cache_format
from .mechanism import PlannedRecipe
from .sharedkv_metadata import OWNER_SCHEDULE


IMPLEMENTATION = "vllm_ascend_sharedkv_scfa"
RECIPE = "vllm_ascend_sharedkv_scfa"


def build(op, geometry):
    from .sharedkv_attention import build as source_build

    options = dict(op.metadata)["sharedkv_source_options"]
    return source_build(op, geometry, scratch_policy=options["scratch_policy"], slots=2)


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def lower_op(op, *, selection, view, plan, views, heads, parallel, model, quant):
    if any(cache_format(model, region).group_size
           or cache_format(model, region).dtype != quant.kv_dtype
           for region in ("window", "compressed_kv")):
        raise ModelingError("sharedkv_scfa_packed_cache_representation_not_supported")
    options = selection.options
    declarations = {
        "owner_schedule": OWNER_SCHEDULE,
        "selected_rows": "valid_prefix",
        "sinks_dtype": "fp32",
        "metadata_s2_split": False,
        "return_softmax_lse": False,
    }
    for key, value in declarations.items():
        if key not in options or options[key] != value:
            raise ModelingError("sharedkv_scfa_explicit_declaration_required:" + key)
    policy = options.get("scratch_policy", "offchip")
    if policy not in {"offchip", "resident"}:
        raise ModelingError("sharedkv_scfa_scratch_policy_unknown:" + str(policy))
    if (len(views) != 1 or view.phase != "prefill"
            or parallel.cp != 1 or parallel.pcp != 1
            or view.draft_proposal_tokens
            or (view.query_ranges and view.query_ranges != ((0, view.q_len),))):
        raise ModelingError("sharedkv_scfa_requires_single_contiguous_prefill_sequence")
    if (plan.mla_variant != "latent_mqa" or plan.cache_kind != "window_compressed"
            or plan.compress_ratio <= 0 or plan.index_topk <= 0):
        raise ModelingError("sharedkv_scfa_requires_compressed_sparse_index_producer")
    shape = {
        "branch": "SCFA", "query": view.q_len, "context": view.context,
        "window": plan.window, "cmp_ratio": plan.compress_ratio,
        "topk": plan.index_topk, "heads": heads, "width": plan.head_dim,
        "cp": parallel.cp, "pcp": parallel.pcp, "query_dtype": model.activation_dtype,
        "cache_dtype": quant.kv_dtype, "sparse_block_size": 1,
        "output_dtype": model.activation_dtype,
        "query_layout": "TND", "cache_layout": "PA_ND",
        "kv_heads": 1, "sequences": 1, "views": len(views), **declarations,
    }
    metadata = {**dict(op.metadata),
                "sharedkv_attention_geometry": shape,
                "sharedkv_source_options": {"scratch_policy": policy},
                "implementation_binding": {
                    "selector": "sharedkv_attention", "implementation": selection.name,
                    "branch_scope": "prefill_SCFA_with_nonempty_compressed_index_tensor",
                    "capture_dispatch": "not_established",
                }}
    return replace(op, recipe=RECIPE, source_boundary="kernel",
                   metadata=tuple(metadata.items()))
