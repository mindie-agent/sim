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
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    FieldDisposition, ProgramFragment, SemanticAttr, SemanticRef, WorkItem, WorkSet,
)
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet, TensorPortSpec


FIA_ABSORBED_LATENT_DIM = 512
FIA_ABSORBED_ROPE_DIM = 64

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
                or runtime.parallel.cp != 1
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


def _padding_plan(*, plan, parallel=None, tokens, input_dtype="bf16",
                  logical_heads=None, physical_heads=None):
    logical, physical = (
        (int(logical_heads), int(physical_heads))
        if logical_heads is not None and physical_heads is not None
        else _heads(plan, parallel))
    if (type(tokens) is not int or isinstance(tokens, bool) or tokens < 0):
        raise ModelingError("mla_head_padding_requires_nonnegative_token_rows")
    if input_dtype != "bf16":
        raise ModelingError("mla_head_padding_requires_bf16_query_rows")
    if (plan.kv_lora_rank, plan.qk_rope_head_dim) != (FIA_ABSORBED_LATENT_DIM, FIA_ABSORBED_ROPE_DIM):
        raise ModelingError(
            "mla_head_padding_requires_latent512_rope64:"
            f"{plan.kv_lora_rank}:{plan.qk_rope_head_dim}")
    if not 1 <= logical <= physical or physical < 1:
        raise ModelingError("mla_head_padding_requires_logical_le_physical_heads")
    byte = dtype_bytes(input_dtype)
    latent, rope = plan.kv_lora_rank, plan.qk_rope_head_dim
    copy_volume = tokens * logical * latent * byte if tokens > 1 else 0
    nope_read = tokens * logical * latent * byte
    nope_write = tokens * physical * latent * byte
    rope_read = tokens * logical * rope * byte
    rope_write = tokens * physical * rope * byte
    return {
        "logical_heads": logical,
        "physical_heads": physical,
        "tokens": tokens,
        "input_dtype": input_dtype,
        "latent": latent,
        "rope": rope,
        "copy_volume": copy_volume,
        "nope_read": nope_read,
        "nope_write": nope_write,
        "rope_read": rope_read,
        "rope_write": rope_write,
        "zero_fill_bytes": tokens * (physical - logical) * (latent + rope) * byte,
        "mte2": copy_volume + nope_read + rope_read,
        "mte3": copy_volume + nope_write + rope_write,
    }


def padding_contract(*, plan, parallel, tokens, input_dtype="bf16",
                     logical_heads=None, physical_heads=None,
                     position_start=None, occurrence=None):
    """Typed logical->physical head materialization with real copy/pad bytes."""
    if occurrence is not None and not isinstance(occurrence, Occurrence):
        raise ModelingError("mla_head_padding_occurrence_invalid")
    spec = _padding_plan(
        plan=plan, parallel=parallel, tokens=tokens, input_dtype=input_dtype,
        logical_heads=logical_heads, physical_heads=physical_heads)
    logical, physical = spec["logical_heads"], spec["physical_heads"]
    hbm = spec["mte2"] + spec["mte3"]
    return ProgramFragment(
        "mla_head_padding",
        occurrence=occurrence,
        semantic_refs=(SemanticRef("subgraph", "mla_attention",
                                   role="head_materialization"),),
        attrs=(
            SemanticAttr("implementation", IMPLEMENTATION, source="selected_program"),
            SemanticAttr("source", SOURCE, source="source_binding"),
            SemanticAttr("logical_heads", logical, unit="head"),
            SemanticAttr("physical_heads", physical, unit="head"),
            SemanticAttr("tokens", spec["tokens"], unit="row"),
            SemanticAttr("input_dtype", spec["input_dtype"], source="typed_input"),
            SemanticAttr("latent_width", spec["latent"], unit="element"),
            SemanticAttr("rope_width", spec["rope"], unit="element"),
            SemanticAttr("copy_volume", spec["copy_volume"], unit="byte"),
            SemanticAttr("zero_fill_bytes", spec["zero_fill_bytes"], unit="byte"),
            SemanticAttr("input_view",
                         "prolog[:, :logical_heads]; zero-copy slice",
                         source="caller_view"),
            SemanticAttr("output_view",
                         "FIA_NBSD[:logical_heads]; contiguous leading-head slice; no copy",
                         source="caller_view"),
            SemanticAttr("position_start", position_start, unit="token",
                         source="request_context"),
            SemanticAttr("unresolved_fields",
                         ("capture_binary_identity", "prepared_weight_aliasing_for_single_head"),
                         source="specific_request"),
        ),
        ports=PortSet(
            tensor_in=(
                TensorPortSpec("q_logical", "in", "activation", dtype=input_dtype,
                               layout_kind="BNSD", scope="layer"),
                TensorPortSpec("q_rope_logical", "in", "activation",
                               dtype=input_dtype, layout_kind="BNSD", scope="layer"),
            ),
            tensor_out=(
                TensorPortSpec("q_physical", "out", "activation",
                               dtype=input_dtype, layout_kind="BNSD", scope="layer"),
                TensorPortSpec("q_rope_physical", "out", "activation",
                               dtype=input_dtype, layout_kind="BNSD", scope="layer"),
            ),
            event_in=(EventPortSpec("q_ready", "in", "ready"),),
            event_out=(EventPortSpec("done", "out", "complete"),),
        ),
        work=WorkSet((
            WorkItem("hbm", hbm, "bytes", "issued", "selected_copy_and_pad"),
            WorkItem("mte2", spec["mte2"], "bytes", "issued", "selected_copy_and_pad"),
            WorkItem("mte3", spec["mte3"], "bytes", "issued", "selected_copy_and_pad"),
        )),
        field_effects=(
            FieldDisposition("q.dtype", "preserve"),
            FieldDisposition("q.layout", "consume",
                             source="native BNSD logical->physical view"),
            FieldDisposition("position", "preserve",
                             source="request_context"),
            FieldDisposition("scale", "preserve",
                             source="no scale materialization in this copy/pad"),
        ),
        notes=("work is the actual contiguous-copy and F.pad traffic",
               "logical KV cache and final FIA output head view are unchanged"),
    )


def query_inputs(b, *, declaration, tokens, query, rotary, plan):
    """Native prolog slice -> required contiguous copy -> FIA F.pad."""
    logical, physical = declaration["logical_heads"], declaration["physical_heads"]
    if logical == physical:
        return query, rotary
    spec = _padding_plan(
        plan=plan, parallel=None, tokens=tokens,
        input_dtype=b.model.activation_dtype,
        logical_heads=logical, physical_heads=physical)
    byte = dtype_bytes(b.model.activation_dtype)
    common = (("implementation", IMPLEMENTATION), ("source", SOURCE),
              ("logical_heads", logical), ("physical_heads", physical),
              ("input_view", "prolog[:, :logical_heads]; zero-copy slice"))
    # A [B,P,D] -> [:,:L] slice retains row stride P*D. It is contiguous
    # when B=1, otherwise the source's q_nope.contiguous() allocates/copies.
    if tokens > 1:
        volume = spec["copy_volume"]
        query = b.work("mla_query_contiguous", (query,),
                       (tokens, logical, plan.kv_lora_rank), read=volume, write=volume,
                       family="copy", dtype=b.model.activation_dtype, metadata=common)
    outputs = []
    for name, source, width, read, write in (
            ("nope", query, plan.kv_lora_rank, spec["nope_read"], spec["nope_write"]),
            ("rope", rotary, plan.qk_rope_head_dim,
             spec["rope_read"], spec["rope_write"])):
        outputs.append(b.work(
            "mla_query_pad_" + name, (source,), (tokens, physical, width),
            read=read, write=write, family="copy",
            dtype=b.model.activation_dtype,
            metadata=common + (("zero_fill_bytes",
                                tokens * (physical - logical) * width * byte),)))
    return tuple(outputs)


def output_metadata(declaration):
    if declaration is None:
        return ()
    return (("mla_head_contract", dict(declaration)),
            ("input_view", "FIA_NBSD[:logical_heads]; contiguous leading-head slice; no_copy"))
