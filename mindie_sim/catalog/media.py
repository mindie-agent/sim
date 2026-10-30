"""Vision tower, merge/aligner and media-token expansion for catalog models.

[Role]
  Lower every declared vision block plus patch embedding, merge and projector,
  and convert image/video geometry into explicit pre-merge patch tokens and
  post-merge text positions.

[Invariants]
  * Request ``media_tokens`` denotes post-merge text positions; pre-merge patch
    tokens are derived from the declared merge/downsample area.
  * A processed grid supplied in runtime metadata is used when present; raw
    pixels without processor semantics stay visible as an explicit limitation.
  * Text-only requests skip the entire tower (no zero-duration media nodes).
  * Every vision layer contributes QKV, QK/PV, output projection, MLP and
    normalization work; merge/projector layers are actual dependent work.
  * Media positions change the text sequence/cache lengths through the
    orchestrator, not only metadata.

[Timeouts & Locks]
  Pure arithmetic bounded by vision layers; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec, VisionSpec
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec


@dataclass
class MediaExpansion:
    text_positions: int = 0
    patch_tokens: int = 0
    grid: tuple[int, int, int] = (0, 0, 0)
    ops: list[OpSpec] = field(default_factory=list)
    last: str = ""
    notes: tuple[str, ...] = ()


def media_geometry(
    runtime: Any, vision: VisionSpec, media_tokens: int | None = None,
) -> tuple[int, int, int, int]:
    """Return (patch_tokens, grid_h, grid_w, frames) from explicit metadata."""
    meta = dict(getattr(runtime, "metadata", ()) or ())
    height = int(meta.get("media_height", 0) or 0)
    width = int(meta.get("media_width", 0) or 0)
    frames = max(1, int(meta.get("media_frames", 1) or 1))
    patch = max(1, int(vision.patch_size))
    temporal = max(1, int(vision.temporal_patch))
    if height > 0 and width > 0:
        grid_h = max(1, height // patch)
        grid_w = max(1, width // patch)
        frame_groups = max(1, math.ceil(frames / temporal))
        return grid_h * grid_w * frame_groups, grid_h, grid_w, frames
    media_tokens = int(
        getattr(runtime, "media_tokens", 0) if media_tokens is None else media_tokens
    )
    patch_tokens = vision.patch_tokens_for_media(media_tokens) if media_tokens else 0
    return patch_tokens, 0, 0, frames


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def lower_media(
    *, model: ModelSpec, runtime: Any, parallel: ParallelSpec,
    quant: QuantSpec, mode: str, views: Sequence[Any], stage: int = 0,
    media_tokens_override: int | None = None,
) -> MediaExpansion:
    vision = model.vision
    media_tokens = int(
        getattr(runtime, "media_tokens", 0) if media_tokens_override is None
        else media_tokens_override
    )
    if media_tokens <= 0:
        return MediaExpansion()
    view_phases = {str(getattr(view, "semantic_phase", getattr(view, "phase", ""))) for view in views}
    if view_phases and view_phases <= {"decode"}:
        # Vision work happens when media is ingested (prefill).  Decode reuses
        # already merged embeddings/positions and must not re-run the tower.
        return MediaExpansion(text_positions=0, patch_tokens=0, ops=[], last="")
    if vision is None:
        raise ModelingError("multimodal_request_without_vision_config")
    vision.validate()
    patch_tokens, grid_h, grid_w, frames = media_geometry(
        runtime, vision, media_tokens=media_tokens,
    )
    if patch_tokens <= 0:
        raise ModelingError("media_geometry_empty")
    post_merge = vision.merged_tokens(patch_tokens)
    text_positions = int(media_tokens) if grid_h == 0 else post_merge
    requests = _requests(views)
    dtype = model.activation_dtype
    ops: list[OpSpec] = []
    patch = gemm_op(
        op_id="vision.patch_embed", deps=(), stage=stage, layer=0,
        requests=requests, m=patch_tokens, n=vision.hidden_size,
        k=vision.patch_input_width, quant=quant, category="vision",
        phase=mode, family="vision_patch_embed",
        semantic_family="vision_patch_embedding",
        metadata=(("patch_tokens", patch_tokens), ("merge_area", vision.merge_area),
                  ("grid_h", grid_h), ("grid_w", grid_w), ("frames", frames),
                  ("media_text_positions", text_positions)),
    )
    ops.append(patch)
    last = patch.op_id
    if vision.pos_embed_tokens:
        pos = vector_op(
            op_id="vision.pos_embed", deps=(last,), stage=stage, layer=0,
            requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=1.0, dtype=dtype, family="embedding", phase=mode,
            recipe="catalog_vision_pos_embed",
            semantic_family="vision_position_embedding",
            metadata=(("pos_embed_tokens", vision.pos_embed_tokens),),
        )
        ops.append(pos)
        last = pos.op_id
    head_dim = vision.head_dim
    for layer in range(vision.layers):
        prefix = f"vision.l{layer}"
        norm1 = vector_op(
            op_id=f"{prefix}.norm1", deps=(last,), stage=stage, layer=layer,
            requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=5.0, dtype=dtype,
            family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
            phase=mode, recipe="catalog_vision_norm",
            shape=(patch_tokens, vision.hidden_size),
            metadata=(("tokens", patch_tokens),
                      ("reduction_width", vision.hidden_size),
                      ("norm_type", vision.norm_type),
                      ("norm_eps", vision.norm_eps)),
        )
        qkv = gemm_op(
            op_id=f"{prefix}.qkv", deps=(norm1.op_id,), stage=stage, layer=layer,
            requests=requests, m=patch_tokens, n=vision.qkv_width,
            k=vision.hidden_size, quant=quant, category="vision", phase=mode,
            family="qkv_projection", semantic_family="vision_qkv_projection",
        )
        qkv_ready = qkv.op_id
        if vision.bias:
            qkv_bias = vector_op(
                op_id=f"{prefix}.qkv_bias", deps=(qkv.op_id,), stage=stage,
                layer=layer, requests=requests,
                elements=max(1, patch_tokens * vision.qkv_width),
                ops_per_element=1.0, dtype=dtype, family="elementwise",
                phase=mode, recipe="catalog_vision_bias",
                semantic_family="vision_qkv_bias",
            )
            qkv_ready = qkv_bias.op_id
        qk = OpSpec(
            op_id=f"{prefix}.qk", family="attention_qk", phase=mode,
            stage=stage, layer=layer, requests=requests,
            semantic_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            local_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            padded_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            launch_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            flops=2.0 * vision.num_heads * patch_tokens * patch_tokens * head_dim,
            hbm_bytes=2.0 * patch_tokens * vision.query_width * dtype_bytes(dtype)
            + patch_tokens * patch_tokens * vision.num_heads * 4.0,
            mte2_bytes=2.0 * patch_tokens * vision.query_width * dtype_bytes(dtype),
            mte3_bytes=patch_tokens * patch_tokens * vision.num_heads * 4.0,
            deps=(qkv_ready,), recipe="catalog_vision_qk", dtype=dtype,
        )
        soft = vector_op(
            op_id=f"{prefix}.softmax", deps=(qk.op_id,), stage=stage, layer=layer,
            requests=requests, elements=max(1, patch_tokens * vision.num_heads * patch_tokens),
            ops_per_element=5.0, dtype="fp32", family="attention_softmax",
            phase=mode, recipe="catalog_vision_softmax",
        )
        pv = OpSpec(
            op_id=f"{prefix}.pv", family="attention_pv", phase=mode,
            stage=stage, layer=layer, requests=requests,
            semantic_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            local_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            padded_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            launch_shape=(patch_tokens, patch_tokens, vision.num_heads, head_dim),
            flops=2.0 * vision.num_heads * patch_tokens * patch_tokens * head_dim,
            hbm_bytes=2 * patch_tokens * vision.query_width * dtype_bytes(dtype)
            + patch_tokens * patch_tokens * vision.num_heads * 4.0,
            mte2_bytes=patch_tokens * vision.query_width * dtype_bytes(dtype)
            + patch_tokens * patch_tokens * vision.num_heads * 4.0,
            mte3_bytes=patch_tokens * vision.query_width * dtype_bytes(dtype),
            deps=(soft.op_id,), recipe="catalog_vision_pv", dtype=dtype,
        )
        wo = gemm_op(
            op_id=f"{prefix}.wo", deps=(pv.op_id,), stage=stage, layer=layer,
            requests=requests, m=patch_tokens, n=vision.hidden_size,
            k=vision.query_width, quant=quant, category="vision", phase=mode,
            family="o_projection", semantic_family="vision_output_projection",
        )
        wo_ready = wo.op_id
        if vision.bias:
            wo_bias = vector_op(
                op_id=f"{prefix}.wo_bias", deps=(wo.op_id,), stage=stage,
                layer=layer, requests=requests,
                elements=max(1, patch_tokens * vision.hidden_size),
                ops_per_element=1.0, dtype=dtype, family="elementwise",
                phase=mode, recipe="catalog_vision_bias",
                semantic_family="vision_wo_bias",
            )
            wo_ready = wo_bias.op_id
        add1 = vector_op(
            op_id=f"{prefix}.residual1", deps=(wo_ready, last), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=2.0, dtype=dtype, family="elementwise", phase=mode,
            recipe="catalog_vision_residual",
        )
        norm2 = vector_op(
            op_id=f"{prefix}.norm2", deps=(add1.op_id,), stage=stage, layer=layer,
            requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=5.0, dtype=dtype,
            family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
            phase=mode, recipe="catalog_vision_norm",
            shape=(patch_tokens, vision.hidden_size),
            metadata=(("tokens", patch_tokens),
                      ("reduction_width", vision.hidden_size),
                      ("norm_type", vision.norm_type),
                      ("norm_eps", vision.norm_eps)),
        )
        fc0 = gemm_op(
            op_id=f"{prefix}.fc0", deps=(norm2.op_id,), stage=stage, layer=layer,
            requests=requests, m=patch_tokens,
            n=(2 if vision.gated_mlp else 1) * vision.intermediate_size,
            k=vision.hidden_size, quant=quant, category="vision", phase=mode,
            family="gate_up_projection", semantic_family="vision_mlp_fc0",
            metadata=(("gated_mlp", vision.gated_mlp),),
        )
        fc0_ready = fc0.op_id
        if vision.bias:
            fc0_bias = vector_op(
                op_id=f"{prefix}.fc0_bias", deps=(fc0.op_id,), stage=stage,
                layer=layer, requests=requests,
                elements=max(1, patch_tokens * (2 if vision.gated_mlp else 1)
                             * vision.intermediate_size),
                ops_per_element=1.0, dtype=dtype, family="elementwise",
                phase=mode, recipe="catalog_vision_bias",
                semantic_family="vision_fc0_bias",
            )
            fc0_ready = fc0_bias.op_id
        act_elements = max(1, patch_tokens * vision.intermediate_size)
        act_bytes = dtype_bytes(dtype)
        act = vector_op(
            op_id=f"{prefix}.act", deps=(fc0_ready,), stage=stage, layer=layer,
            requests=requests, elements=act_elements,
            ops_per_element=4.0, dtype=dtype,
            family="swiglu" if vision.gated_mlp else "gelu", phase=mode,
            recipe="catalog_vision_activation",
            shape=(patch_tokens, vision.intermediate_size),
            read_bytes_override=(
                2.0 * act_elements * act_bytes if vision.gated_mlp else None
            ),
            write_bytes_override=(
                1.0 * act_elements * act_bytes if vision.gated_mlp else None
            ),
            metadata=(("activation", vision.activation),
                      ("read_halves", 2 if vision.gated_mlp else 1),
                      ("write_outputs", 1),
                      ("packed_gate_input", bool(vision.gated_mlp))),
        )
        fc1 = gemm_op(
            op_id=f"{prefix}.fc1", deps=(act.op_id,), stage=stage, layer=layer,
            requests=requests, m=patch_tokens, n=vision.hidden_size,
            k=vision.intermediate_size, quant=quant, category="vision",
            phase=mode, family="down_projection", semantic_family="vision_mlp_fc1",
        )
        fc1_ready = fc1.op_id
        if vision.bias:
            fc1_bias = vector_op(
                op_id=f"{prefix}.fc1_bias", deps=(fc1.op_id,), stage=stage,
                layer=layer, requests=requests,
                elements=max(1, patch_tokens * vision.hidden_size),
                ops_per_element=1.0, dtype=dtype, family="elementwise",
                phase=mode, recipe="catalog_vision_bias",
                semantic_family="vision_fc1_bias",
            )
            fc1_ready = fc1_bias.op_id
        add2 = vector_op(
            op_id=f"{prefix}.out", deps=(fc1_ready, add1.op_id), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=2.0, dtype=dtype, family="elementwise", phase=mode,
            recipe="catalog_vision_residual",
        )
        ordered = [norm1, qkv]
        if vision.bias:
            ordered.append(qkv_bias)
        ordered.extend([qk, soft, pv, wo])
        if vision.bias:
            ordered.append(wo_bias)
        ordered.extend([add1, norm2, fc0])
        if vision.bias:
            ordered.append(fc0_bias)
        ordered.extend([act, fc1])
        if vision.bias:
            ordered.append(fc1_bias)
        ordered.append(add2)
        ops.extend(ordered)
        last = add2.op_id
    final_norm = vector_op(
        op_id="vision.final_norm", deps=(last,), stage=stage, layer=vision.layers,
        requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=5.0, dtype=dtype,
        family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
        phase=mode, recipe="catalog_vision_final_norm",
        shape=(patch_tokens, vision.hidden_size),
        metadata=(("tokens", patch_tokens),
                  ("reduction_width", vision.hidden_size),
                  ("norm_type", vision.norm_type),
                  ("norm_eps", vision.norm_eps)),
    )
    ops.append(final_norm)
    merge = vector_op(
        op_id="vision.merge", deps=(final_norm.op_id,), stage=stage,
        layer=vision.layers, requests=requests,
        elements=max(1, patch_tokens * vision.aligner_input_width),
        ops_per_element=1.0, dtype=dtype, family="vision_merge", phase=mode,
        recipe="catalog_vision_merge",
        semantic_family="spatial_merge_concat",
        shape=(text_positions, vision.aligner_input_width),
        metadata=(("merge_area", vision.merge_area), ("patch_tokens", patch_tokens),
                  ("post_merge_tokens", text_positions)),
    )
    ops.append(merge)
    if vision.kind == "deepseek_v41":
        proj1 = gemm_op(
            op_id="vision.projector1", deps=(merge.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.out_hidden_size, k=vision.aligner_input_width,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="aligner_w1",
        )
        proj1_gelu = vector_op(
            op_id="vision.projector1_gelu", deps=(proj1.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.out_hidden_size),
            ops_per_element=4.0, dtype=dtype, family="gelu", phase=mode,
            recipe="catalog_aligner_gelu",
        )
        proj2 = gemm_op(
            op_id="vision.projector2", deps=(proj1_gelu.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.out_hidden_size, k=vision.out_hidden_size,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="aligner_w2",
        )
        aligner_bias = vector_op(
            op_id="vision.projector2_bias", deps=(proj2.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.out_hidden_size),
            ops_per_element=1.0, dtype=dtype, family="elementwise", phase=mode,
            recipe="catalog_aligner_bias",
            semantic_family="aligner_w2_bias",
        )
        ops.extend([proj1, proj1_gelu, proj2, aligner_bias])
        last = aligner_bias.op_id
    elif vision.kind == "kimi_k3":
        proj0 = gemm_op(
            op_id="vision.projector0", deps=(merge.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.aligner_hidden_size, k=vision.aligner_input_width,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="patchmergerv2_proj0",
        )
        proj0_gelu = vector_op(
            op_id="vision.projector0_gelu", deps=(proj0.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.aligner_hidden_size),
            ops_per_element=4.0, dtype=dtype, family="gelu", phase=mode,
            recipe="catalog_kimi_projector_gelu",
        )
        proj2 = gemm_op(
            op_id="vision.projector2", deps=(proj0_gelu.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.out_hidden_size, k=vision.aligner_hidden_size,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="patchmergerv2_proj2",
        )
        post_norm = vector_op(
            op_id="vision.projector_post_norm", deps=(proj2.op_id,),
            stage=stage, layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.out_hidden_size),
            ops_per_element=5.0, dtype=dtype, family="rmsnorm", phase=mode,
            recipe="catalog_kimi_projector_post_norm",
            shape=(text_positions, vision.out_hidden_size),
            metadata=(("tokens", text_positions),
                      ("reduction_width", vision.out_hidden_size),
                      ("norm_type", "rmsnorm"),
                      ("norm_eps", vision.norm_eps)),
        )
        ops.extend([proj0, proj0_gelu, proj2, post_norm])
        last = post_norm.op_id
    else:
        merger_norm = vector_op(
            op_id="vision.merger_norm", deps=(merge.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.aligner_input_width),
            ops_per_element=5.0, dtype=dtype, family="layernorm", phase=mode,
            recipe="catalog_qwen_merger_norm",
            shape=(text_positions, vision.aligner_input_width),
            metadata=(("tokens", text_positions),
                      ("reduction_width", vision.aligner_input_width),
                      ("norm_type", "layernorm"),
                      ("norm_eps", vision.norm_eps)),
        )
        merger_w1 = gemm_op(
            op_id="vision.merger_w1", deps=(merger_norm.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.aligner_input_width, k=vision.aligner_input_width,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="qwen_merger_w1",
        )
        merger_gelu = vector_op(
            op_id="vision.merger_gelu", deps=(merger_w1.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.aligner_input_width),
            ops_per_element=4.0, dtype=dtype, family="gelu", phase=mode,
            recipe="catalog_qwen_merger_gelu",
        )
        merger = gemm_op(
            op_id="vision.merger", deps=(merger_gelu.op_id,), stage=stage,
            layer=vision.layers, requests=requests, m=text_positions,
            n=vision.out_hidden_size, k=vision.aligner_input_width,
            quant=quant, category="projector", phase=mode,
            family="vision_projector", semantic_family="qwen_merger_w2",
        )
        merger_bias = vector_op(
            op_id="vision.merger_bias", deps=(merger.op_id,), stage=stage,
            layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.out_hidden_size),
            ops_per_element=1.0, dtype=dtype, family="elementwise", phase=mode,
            recipe="catalog_qwen_merger_bias",
        )
        ops.extend([merger_norm, merger_w1, merger_gelu, merger, merger_bias])
        last = merger_bias.op_id
    return MediaExpansion(
        text_positions=text_positions, patch_tokens=patch_tokens,
        grid=(grid_h, grid_w, frames), ops=ops, last=last,
        notes=("media_text_positions_expand_text_sequence",
               "raw_pixel_processor_semantics_stay_explicit"),
    )


__all__ = ["MediaExpansion", "lower_media", "media_geometry", "media_runtime_metadata"]


def media_runtime_metadata(workload: Any) -> dict[str, str]:
    """Build the explicit runtime metadata core should merge for scheduler use.

    RequestSpec expands media positions at parse time; this helper tells graph
    lowering which requests still need tower execution and that positions are
    already budgeted.  Encoded requests are excluded from tower execution.
    """
    import json

    mapping: dict[str, int] = {}
    encoded: list[str] = []
    for request in getattr(workload, "requests", ()):
        tokens = int(getattr(request, "media_tokens", 0) or 0)
        if tokens <= 0:
            continue
        if bool(getattr(request, "media_encoded", False)):
            encoded.append(request.request_id)
        else:
            mapping[request.request_id] = tokens
    metadata = {
        "media_positions_pre_expanded": "true",
        "media_tokens_by_request": json.dumps(mapping, sort_keys=True),
    }
    if encoded:
        metadata["media_encoded_requests"] = json.dumps(sorted(encoded))
    return metadata
