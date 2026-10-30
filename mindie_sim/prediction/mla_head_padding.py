"""Explicit native MLA logical heads and physical decode head lanes.

[Role] Preserve the caller's weight padding, query copies and output views.
[Invariants] Only an explicit implementation selects padding. Logical output
  heads and the shared KV cache stay unchanged. Slices are views; contiguous
  copies and F.pad each have their own real byte traffic. No timing parameters.
[Timeouts & Locks] O(layers) preparation and constant work per attention call.
[Side-Effect Policy] Creates immutable metadata and caller-owned graph nodes.
"""
from dataclasses import replace

from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.schema.common import ModelingError, dtype_bytes

IMPLEMENTATION = "vllm_ascend_mla_head_padding"
SOURCE = {
    "repository": "vllm-ascend", "path": "vllm_ascend/attention/mla_v1.py",
    "sha256": "d808b7b41203cbcc4458b55b83f35d7db7152891fd87a49b4a1376eca1f64194",
    "lines": "846-849,973-979,1011-1162,1635-1712,1799-1808",
    "scope": "unquantized native prolog weights retained for local prefill; BF16 FIA decode",
    "identity": "retained_source; capture_binary_dispatch_not_proven",
}


def recipes():
    return {}


def _heads(plan, parallel):
    if plan.num_heads <= 0 or plan.num_heads % parallel.tp:
        raise ModelingError("mla_head_padding_requires_integral_local_heads")
    logical = plan.num_heads // parallel.tp
    return logical, 1 << (logical - 1).bit_length()


def prepare_model(model, runtime):
    lowerings = runtime.kernels.get("lowerings", {})
    if any(key in lowerings for key in ("mla_decode", "mla_prolog")):
        # This caller owns its native prolog and FIA padding. A separately
        # selected Triton kernel or SFA prolog has a different interface.
        raise ModelingError("mla_head_padding_conflicts_with_explicit_mla_implementation")
    prepared = {}
    for plan in model.layer_plans:
        if plan.attention != "mla":
            continue
        quant = projection_quant(model.quant, plan)
        if (plan.mla_variant != "latent_mla" or plan.cache_kind != "dense"
                or plan.index_topk or plan.window or plan.noncausal_attention
                or not plan.q_lora_rank or not plan.qk_rope_head_dim
                or runtime.parallel.cp != 1 or runtime.parallel.pcp != 1
                or model.activation_dtype != "bf16" or quant.kv_dtype != "bf16"
                or any(compute_dtype_for(quant, category)[:2] != ("bf16", "bf16")
                       for category in ("mla.q_a", "mla.q_b", "mla.kv_a", "mla_absorb"))):
            raise ModelingError("mla_head_padding_requires_native_bf16_dense_latent_domain")
        logical, physical = _heads(plan, runtime.parallel)
        prepared[str(plan.index)] = {
            "logical_heads": logical, "physical_heads": physical,
            "tp": runtime.parallel.tp,
            "prepared_weight_policy": "retain_original_prefill_weights",
            "capacity_basis": ("upper_bound_nd_parent_aliasing_unresolved" if logical == 1
                               else "distinct_contiguous_weight_allocations"),
        }
    metadata = {**dict(model.metadata), "mla_head_padding": {
        "implementation": IMPLEMENTATION, "layers": prepared, "source": SOURCE,
        "timing_boundary": "expanded_semantic_prolog_and_attention; no_native_fused_timing_claim",
    }}
    if any(row["logical_heads"] == 1 for row in prepared.values()):
        # At one head, UK may alias its ND parent. If the checkpoint tensor
        # was NZ, format_cast creates a full ND parent retained by that view;
        # if already ND, no parent copy is needed. Keep simulation available
        # and declare the conservative storage bound until layout is explicit.
        metadata["model_coverage"] = "partial"
        metadata["unmodeled_features"] = tuple(sorted(set(metadata.get("unmodeled_features", ())) | {
            "prepared_weight_aliasing:single_head_original_kv_b_layout_unspecified"}))
        metadata["capacity_coverage"] = "upper_bound_for_single_head_mla_prepared_nd_parent"
    return replace(model, metadata=tuple(sorted(metadata.items())))


def contract(model, plan, views):
    profile = dict(model.metadata).get("mla_head_padding", {})
    declared = profile.get("layers", {}).get(str(plan.index))
    if declared is None or not any(view.phase == "decode" for view in views):
        return None
    if any(view.phase != "decode" or view.q_len != 1 for view in views):
        raise ModelingError("mla_head_padding_requires_single_token_decode_batch")
    return declared


def query_inputs(b, *, declaration, tokens, query, rotary, plan):
    """Native prolog slice -> required contiguous copy -> FIA F.pad."""
    logical, physical = declaration["logical_heads"], declaration["physical_heads"]
    if logical == physical:
        return query, rotary
    byte = dtype_bytes(b.model.activation_dtype)
    common = (("implementation", IMPLEMENTATION), ("source", SOURCE),
              ("logical_heads", logical), ("physical_heads", physical),
              ("input_view", "prolog[:, :logical_heads]; zero-copy slice"))
    # A [B,P,D] -> [:,:L] slice retains row stride P*D. It is contiguous
    # when B=1, otherwise the source's q_nope.contiguous() allocates/copies.
    if tokens > 1:
        volume = tokens * logical * plan.kv_lora_rank * byte
        query = b.work("mla_query_contiguous", (query,),
                       (tokens, logical, plan.kv_lora_rank), read=volume, write=volume,
                       family="copy", dtype=b.model.activation_dtype, metadata=common)
    outputs = []
    for name, source, width in (("nope", query, plan.kv_lora_rank),
                                ("rope", rotary, plan.qk_rope_head_dim)):
        outputs.append(b.work(
            "mla_query_pad_" + name, (source,), (tokens, physical, width),
            read=tokens * logical * width * byte,
            write=tokens * physical * width * byte, family="copy",
            dtype=b.model.activation_dtype,
            metadata=common + (("zero_fill_bytes", tokens * (physical - logical) * width * byte),)))
    return tuple(outputs)


def output_metadata(declaration):
    if declaration is None:
        return ()
    return (("mla_head_contract", dict(declaration)),
            ("input_view", "FIA_NBSD[:logical_heads]; contiguous leading-head slice; no_copy"))
