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

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from mindie_sim.catalog.ops import comm_op, gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec, VisionSpec
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec

VISION_NORM_OPS_PER_ELEMENT = 5.0
VISION_SOFTMAX_OPS_PER_ELEMENT = 5.0
VISION_ACT_OPS_PER_ELEMENT = 4.0
VISION_QKV_PROJECTIONS = 3


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
    frame_groups = max(1, math.ceil(frames / temporal))
    if height > 0 and width > 0:
        grid_h = max(1, height // patch)
        grid_w = max(1, width // patch)
        return grid_h * grid_w * frame_groups, grid_h, grid_w, frames
    media_tokens = int(
        getattr(runtime, "media_tokens", 0) if media_tokens is None else media_tokens
    )
    if not media_tokens:
        return 0, 0, 0, frames
    patch_tokens = vision.patch_tokens_for_media(media_tokens)
    if _uses_spatial_temporal_tpool(vision):
        patch_tokens *= frame_groups
    return patch_tokens, 0, 0, frames


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _vision_tp(runtime: Any, parallel: ParallelSpec, vision: VisionSpec) -> int:
    """Resolve effective vision encoder TP width.

    Falls back to ``1`` (encoder data-parallel / replicated execution) when
    ``mm_encoder_tp_mode == "data"`` or when vision attention heads are not
    divisible by ``parallel.tp`` (matching native MoonViT / ViT TP policy).
    """
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if str(meta.get("mm_encoder_tp_mode", "") or "").strip().lower() == "data":
        return 1
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    if tp > 1 and int(vision.num_heads) % tp != 0:
        return 1
    return tp


def _op_output_dtype(op: OpSpec, fallback: str) -> str:
    """Actual output dtype written by ``op`` (preserving explicit quant overrides)."""
    if op.precision_abi is not None and op.precision_abi.output_dtype:
        return str(op.precision_abi.output_dtype)
    return str(dict(op.metadata).get("output_format") or fallback)


def _uses_spatial_temporal_tpool(vision: VisionSpec) -> bool:
    """Structural predicate for un-biased PatchMergerV2 / MoonViT encoders.

    Matches ``weights.py``'s PatchMergerV2 branch (``aligner_hidden_size > 0``
    and ``bias=False``): full clip spatial-temporal attention followed by
    temporal mean-pooling before spatial merging.
    """
    return int(vision.aligner_hidden_size) > 0 and not bool(vision.bias)


def _attention_seq_lens(
    patch_tokens: int, grid_h: int, grid_w: int, frames: int, vision: VisionSpec,
) -> tuple[int, ...]:
    """Per-group sequence lengths matching native encoder attention boundaries.

    Spatial-temporal encoders attend across the full clip; temporal-patch
    encoders partition attention per temporal patch group.
    """
    if _uses_spatial_temporal_tpool(vision):
        return (patch_tokens,)
    temporal = max(1, int(vision.temporal_patch))
    frame_groups = max(1, math.ceil(max(1, int(frames)) / temporal))
    if grid_h > 0 and grid_w > 0:
        return (grid_h * grid_w,) * frame_groups
    if frame_groups <= 1:
        return (patch_tokens,)
    base, rem = divmod(patch_tokens, frame_groups)
    return tuple(
        base + (1 if idx < rem else 0)
        for idx in range(frame_groups)
        if base + (1 if idx < rem else 0) > 0
    )


def _post_merge_positions(
    patch_tokens: int, grid_h: int, grid_w: int, frames: int, vision: VisionSpec,
) -> int:
    """Post-merge language token positions honoring per-axis spatial padding and tpool."""
    if grid_h <= 0 or grid_w <= 0:
        if _uses_spatial_temporal_tpool(vision):
            temporal = max(1, int(vision.temporal_patch))
            frame_groups = max(1, math.ceil(max(1, int(frames)) / temporal))
            return vision.merged_tokens(math.ceil(patch_tokens / frame_groups))
        return vision.merged_tokens(patch_tokens)
    temporal = max(1, int(vision.temporal_patch))
    merged_groups = (
        1 if _uses_spatial_temporal_tpool(vision)
        else max(1, math.ceil(max(1, int(frames)) / temporal))
    )
    area = max(1, int(vision.merge_area))
    stride = int(vision.downsample_ratio) if int(vision.downsample_ratio) > 1 else math.isqrt(area)
    if stride > 1 and stride * stride == area:
        return math.ceil(grid_h / stride) * math.ceil(grid_w / stride) * merged_groups
    per_group = vision.merged_tokens(grid_h * grid_w)
    return per_group * merged_groups


def _vision_allreduce(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], rows: int, cols: int, dtype: str,
    tp: int, mode: str, semantic_family: str,
) -> OpSpec:
    """Row-parallel TP all-reduce closing a vision attention, MLP, or merger block."""
    link_bytes = 2.0 * float(tp - 1) / float(tp) * float(rows * cols) * dtype_bytes(dtype)
    return replace(
        comm_op(
            op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
            family="allreduce", phase=mode, recipe="ring_allreduce",
            link_bytes=link_bytes, shape=(rows, cols),
            metadata=(("group_axis", "tp"), ("steps", 2 * (tp - 1)),
                      ("semantic_family", semantic_family)),
        ),
        dtype=dtype,
    )


def _lower_vision_layer_attention(
    *, prefix: str, layer: int, last: str, vision: VisionSpec, patch_tokens: int,
    seq_lens: tuple[int, ...], head_dim: int, dtype: str, quant: QuantSpec,
    mode: str, requests: tuple[str, ...], stage: int, vision_tp: int,
) -> tuple[list[OpSpec], str]:
    """Lower norm1 + QKV + per-group QK/softmax/PV + Wo (+ TP all-reduce) + residual1."""
    attn_tp = vision_tp if (vision_tp > 1 and vision.num_heads % vision_tp == 0) else 1
    local_heads = vision.num_heads // attn_tp
    local_query_width = local_heads * head_dim
    local_qkv_width = VISION_QKV_PROJECTIONS * local_query_width
    score_pairs = sum(length * length for length in seq_lens)
    attn_k_len = max(seq_lens) if seq_lens else max(1, patch_tokens)

    norm1 = vector_op(
        op_id=f"{prefix}.norm1", deps=(last,), stage=stage, layer=layer,
        requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
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
        requests=requests, m=patch_tokens, n=local_qkv_width,
        k=vision.hidden_size, quant=quant, category="vision", phase=mode,
        family="qkv_projection", semantic_family="vision_qkv_projection",
        semantic_shape=(patch_tokens, vision.qkv_width, vision.hidden_size),
    )
    ordered: list[OpSpec] = [norm1, qkv]
    qkv_ready = qkv.op_id
    qkv_out_dtype = _op_output_dtype(qkv, dtype)
    if vision.bias:
        qkv_bias = vector_op(
            op_id=f"{prefix}.qkv_bias", deps=(qkv.op_id,), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * local_qkv_width),
            ops_per_element=1.0, dtype=qkv_out_dtype, family="elementwise",
            phase=mode, recipe="catalog_vision_bias",
            semantic_family="vision_qkv_bias",
        )
        ordered.append(qkv_bias)
        qkv_ready = qkv_bias.op_id
    q_bytes = float(patch_tokens * local_query_width) * dtype_bytes(dtype)
    attn_meta = (
        ("attention_group_count", len(seq_lens)),
        ("attention_max_seq_len", attn_k_len),
        ("attention_score_pairs", score_pairs),
        ("attention_seq_lens", seq_lens),
        ("attention_score_boundary", "fused_on_chip"),
    )
    qk = OpSpec(
        op_id=f"{prefix}.qk", family="attention_qk", phase=mode,
        stage=stage, layer=layer, requests=requests,
        semantic_shape=(patch_tokens, attn_k_len, vision.num_heads, head_dim),
        local_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        padded_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        launch_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        flops=2.0 * float(local_heads * score_pairs * head_dim),
        hbm_bytes=2.0 * q_bytes,
        mte2_bytes=2.0 * q_bytes,
        mte3_bytes=0.0,
        deps=(qkv_ready,), recipe="catalog_vision_qk", dtype=dtype,
        metadata=attn_meta,
    )
    soft = vector_op(
        op_id=f"{prefix}.softmax", deps=(qk.op_id,), stage=stage, layer=layer,
        requests=requests, elements=max(1, score_pairs * local_heads),
        ops_per_element=VISION_SOFTMAX_OPS_PER_ELEMENT, dtype="fp32",
        family="attention_softmax", phase=mode, recipe="catalog_vision_softmax",
        read_bytes_override=0.0, write_bytes_override=0.0, hbm_bytes_override=0.0,
        metadata=attn_meta,
    )
    pv = OpSpec(
        op_id=f"{prefix}.pv", family="attention_pv", phase=mode,
        stage=stage, layer=layer, requests=requests,
        semantic_shape=(patch_tokens, attn_k_len, vision.num_heads, head_dim),
        local_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        padded_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        launch_shape=(patch_tokens, attn_k_len, local_heads, head_dim),
        flops=2.0 * float(local_heads * score_pairs * head_dim),
        hbm_bytes=2.0 * q_bytes,
        mte2_bytes=q_bytes,
        mte3_bytes=q_bytes,
        deps=(soft.op_id,), recipe="catalog_vision_pv", dtype=dtype,
        metadata=attn_meta,
    )
    wo = gemm_op(
        op_id=f"{prefix}.wo", deps=(pv.op_id,), stage=stage, layer=layer,
        requests=requests, m=patch_tokens, n=vision.hidden_size,
        k=local_query_width, quant=quant, category="vision", phase=mode,
        family="o_projection", semantic_family="vision_output_projection",
        semantic_shape=(patch_tokens, vision.hidden_size, vision.query_width),
    )
    ordered.extend([qk, soft, pv, wo])
    wo_ready = wo.op_id
    wo_out_dtype = _op_output_dtype(wo, dtype)
    if attn_tp > 1:
        wo_ar = _vision_allreduce(
            op_id=f"{prefix}.wo_allreduce", deps=(wo_ready,), stage=stage,
            layer=layer, requests=requests, rows=patch_tokens,
            cols=vision.hidden_size, dtype=wo_out_dtype, tp=attn_tp, mode=mode,
            semantic_family="vision_output_allreduce",
        )
        ordered.append(wo_ar)
        wo_ready = wo_ar.op_id
    if vision.bias:
        wo_bias = vector_op(
            op_id=f"{prefix}.wo_bias", deps=(wo_ready,), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=1.0, dtype=wo_out_dtype, family="elementwise",
            phase=mode, recipe="catalog_vision_bias",
            semantic_family="vision_wo_bias",
        )
        ordered.append(wo_bias)
        wo_ready = wo_bias.op_id
    add1 = vector_op(
        op_id=f"{prefix}.residual1", deps=(wo_ready, last), stage=stage,
        layer=layer, requests=requests,
        elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=2.0, dtype=dtype, family="elementwise", phase=mode,
        recipe="catalog_vision_residual",
    )
    ordered.append(add1)
    return ordered, add1.op_id


def _lower_vision_layer(
    *, layer: int, last: str, vision: VisionSpec, patch_tokens: int,
    seq_lens: tuple[int, ...], head_dim: int, dtype: str, quant: QuantSpec,
    mode: str, requests: tuple[str, ...], stage: int, vision_tp: int,
) -> tuple[list[OpSpec], str]:
    """Lower one vision encoder transformer block (attention + MLP)."""
    prefix = f"vision.l{layer}"
    ordered, add1_id = _lower_vision_layer_attention(
        prefix=prefix, layer=layer, last=last, vision=vision,
        patch_tokens=patch_tokens, seq_lens=seq_lens, head_dim=head_dim,
        dtype=dtype, quant=quant, mode=mode, requests=requests, stage=stage,
        vision_tp=vision_tp,
    )
    mlp_tp = vision_tp if (vision_tp > 1 and vision.intermediate_size % vision_tp == 0) else 1
    local_inter = vision.intermediate_size // mlp_tp
    norm2 = vector_op(
        op_id=f"{prefix}.norm2", deps=(add1_id,), stage=stage, layer=layer,
        requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
        family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
        phase=mode, recipe="catalog_vision_norm",
        shape=(patch_tokens, vision.hidden_size),
        metadata=(("tokens", patch_tokens),
                  ("reduction_width", vision.hidden_size),
                  ("norm_type", vision.norm_type),
                  ("norm_eps", vision.norm_eps)),
    )
    fc0_factor = 2 if vision.gated_mlp else 1
    fc0_width = fc0_factor * vision.intermediate_size
    local_fc0_width = fc0_factor * local_inter
    fc0 = gemm_op(
        op_id=f"{prefix}.fc0", deps=(norm2.op_id,), stage=stage, layer=layer,
        requests=requests, m=patch_tokens, n=local_fc0_width,
        k=vision.hidden_size, quant=quant, category="vision", phase=mode,
        family="gate_up_projection", semantic_family="vision_mlp_fc0",
        semantic_shape=(patch_tokens, fc0_width, vision.hidden_size),
        metadata=(("gated_mlp", vision.gated_mlp),),
    )
    ordered.extend([norm2, fc0])
    fc0_ready = fc0.op_id
    fc0_out_dtype = _op_output_dtype(fc0, dtype)
    if vision.bias:
        fc0_bias = vector_op(
            op_id=f"{prefix}.fc0_bias", deps=(fc0.op_id,), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * local_fc0_width),
            ops_per_element=1.0, dtype=fc0_out_dtype, family="elementwise",
            phase=mode, recipe="catalog_vision_bias",
            semantic_family="vision_fc0_bias",
        )
        ordered.append(fc0_bias)
        fc0_ready = fc0_bias.op_id
    act_elements = max(1, patch_tokens * local_inter)
    act_bytes = dtype_bytes(dtype)
    act = vector_op(
        op_id=f"{prefix}.act", deps=(fc0_ready,), stage=stage, layer=layer,
        requests=requests, elements=act_elements,
        ops_per_element=VISION_ACT_OPS_PER_ELEMENT, dtype=dtype,
        family="swiglu" if vision.gated_mlp else "gelu", phase=mode,
        recipe="catalog_vision_activation",
        shape=(patch_tokens, local_inter),
        read_bytes_override=2.0 * act_elements * act_bytes if vision.gated_mlp else None,
        write_bytes_override=1.0 * act_elements * act_bytes if vision.gated_mlp else None,
        metadata=(("activation", vision.activation),
                  ("read_halves", 2 if vision.gated_mlp else 1),
                  ("write_outputs", 1),
                  ("packed_gate_input", bool(vision.gated_mlp))),
    )
    fc1 = gemm_op(
        op_id=f"{prefix}.fc1", deps=(act.op_id,), stage=stage, layer=layer,
        requests=requests, m=patch_tokens, n=vision.hidden_size,
        k=local_inter, quant=quant, category="vision",
        phase=mode, family="down_projection", semantic_family="vision_mlp_fc1",
        semantic_shape=(patch_tokens, vision.hidden_size, vision.intermediate_size),
    )
    ordered.extend([act, fc1])
    fc1_ready = fc1.op_id
    fc1_out_dtype = _op_output_dtype(fc1, dtype)
    if mlp_tp > 1:
        fc1_ar = _vision_allreduce(
            op_id=f"{prefix}.fc1_allreduce", deps=(fc1_ready,), stage=stage,
            layer=layer, requests=requests, rows=patch_tokens,
            cols=vision.hidden_size, dtype=fc1_out_dtype, tp=mlp_tp, mode=mode,
            semantic_family="vision_mlp_allreduce",
        )
        ordered.append(fc1_ar)
        fc1_ready = fc1_ar.op_id
    if vision.bias:
        fc1_bias = vector_op(
            op_id=f"{prefix}.fc1_bias", deps=(fc1_ready,), stage=stage,
            layer=layer, requests=requests,
            elements=max(1, patch_tokens * vision.hidden_size),
            ops_per_element=1.0, dtype=fc1_out_dtype, family="elementwise",
            phase=mode, recipe="catalog_vision_bias",
            semantic_family="vision_fc1_bias",
        )
        ordered.append(fc1_bias)
        fc1_ready = fc1_bias.op_id
    add2 = vector_op(
        op_id=f"{prefix}.out", deps=(fc1_ready, add1_id), stage=stage,
        layer=layer, requests=requests,
        elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=2.0, dtype=dtype, family="elementwise", phase=mode,
        recipe="catalog_vision_residual",
    )
    ordered.append(add2)
    return ordered, add2.op_id


def _lower_biased_aligner_projector(
    *, merge_id: str, vision: VisionSpec, text_positions: int,
    dtype: str, quant: QuantSpec, mode: str, requests: tuple[str, ...],
    stage: int, vision_tp: int, prefix: str, op_layer: int,
    use_postshuffle_norm: bool = False,
) -> tuple[list[OpSpec], str]:
    """Lower a biased two-layer aligner projector (``aligner_hidden_size > 0`` and ``bias``)."""
    proj_tp = vision_tp if (vision_tp > 1 and vision.out_hidden_size % vision_tp == 0) else 1
    local_mid = vision.out_hidden_size // proj_tp
    proj_ops: list[OpSpec] = []
    proj1_dep = merge_id
    if use_postshuffle_norm:
        aligner_norm = vector_op(
            op_id=f"{prefix}aligner_norm", deps=(merge_id,), stage=stage,
            layer=op_layer, requests=requests,
            elements=max(1, text_positions * vision.aligner_input_width),
            ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
            family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
            phase=mode, recipe="catalog_aligner_norm",
            shape=(text_positions, vision.aligner_input_width),
            metadata=(("tokens", text_positions),
                      ("reduction_width", vision.aligner_input_width),
                      ("norm_type", vision.norm_type),
                      ("norm_eps", vision.norm_eps)),
        )
        proj_ops.append(aligner_norm)
        proj1_dep = aligner_norm.op_id
    proj1 = gemm_op(
        op_id=f"{prefix}projector1", deps=(proj1_dep,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=local_mid, k=vision.aligner_input_width,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="aligner_w1",
        semantic_shape=(text_positions, vision.out_hidden_size, vision.aligner_input_width),
    )
    proj_ops.append(proj1)
    proj1_ready = proj1.op_id
    proj1_out_dtype = _op_output_dtype(proj1, dtype)
    if vision.bias:
        proj1_bias = vector_op(
            op_id=f"{prefix}projector1_bias", deps=(proj1.op_id,), stage=stage,
            layer=op_layer, requests=requests,
            elements=max(1, text_positions * local_mid),
            ops_per_element=1.0, dtype=proj1_out_dtype, family="elementwise",
            phase=mode, recipe="catalog_aligner_bias",
            semantic_family="aligner_w1_bias",
        )
        proj_ops.append(proj1_bias)
        proj1_ready = proj1_bias.op_id
    proj1_gelu = vector_op(
        op_id=f"{prefix}projector1_gelu", deps=(proj1_ready,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * local_mid),
        ops_per_element=VISION_ACT_OPS_PER_ELEMENT, dtype=dtype,
        family="gelu", phase=mode, recipe="catalog_aligner_gelu",
    )
    proj2 = gemm_op(
        op_id=f"{prefix}projector2", deps=(proj1_gelu.op_id,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=vision.out_hidden_size, k=local_mid,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="aligner_w2",
        semantic_shape=(text_positions, vision.out_hidden_size, vision.out_hidden_size),
    )
    proj_ops.extend([proj1_gelu, proj2])
    proj2_ready = proj2.op_id
    proj2_out_dtype = _op_output_dtype(proj2, dtype)
    if proj_tp > 1:
        proj2_ar = _vision_allreduce(
            op_id=f"{prefix}projector2_allreduce", deps=(proj2_ready,),
            stage=stage, layer=op_layer, requests=requests,
            rows=text_positions, cols=vision.out_hidden_size,
            dtype=proj2_out_dtype, tp=proj_tp, mode=mode,
            semantic_family="vision_projector_allreduce",
        )
        proj_ops.append(proj2_ar)
        proj2_ready = proj2_ar.op_id
    aligner_bias = vector_op(
        op_id=f"{prefix}projector2_bias", deps=(proj2_ready,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * vision.out_hidden_size),
        ops_per_element=1.0, dtype=proj2_out_dtype, family="elementwise",
        phase=mode, recipe="catalog_aligner_bias",
        semantic_family="aligner_w2_bias",
    )
    proj_ops.append(aligner_bias)
    return proj_ops, aligner_bias.op_id


def _lower_unbiased_patch_merger_projector(
    *, merge_id: str, vision: VisionSpec, text_positions: int,
    dtype: str, quant: QuantSpec, mode: str, requests: tuple[str, ...],
    stage: int, prefix: str, op_layer: int,
) -> tuple[list[OpSpec], str]:
    """Lower an unbiased replicated PatchMergerV2 projector with post-RMSNorm."""
    proj0 = gemm_op(
        op_id=f"{prefix}projector0", deps=(merge_id,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=vision.aligner_hidden_size, k=vision.aligner_input_width,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="patchmergerv2_proj0",
    )
    proj0_gelu = vector_op(
        op_id=f"{prefix}projector0_gelu", deps=(proj0.op_id,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * vision.aligner_hidden_size),
        ops_per_element=VISION_ACT_OPS_PER_ELEMENT, dtype=dtype,
        family="gelu", phase=mode, recipe="catalog_kimi_projector_gelu",
    )
    proj2 = gemm_op(
        op_id=f"{prefix}projector2", deps=(proj0_gelu.op_id,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=vision.out_hidden_size, k=vision.aligner_hidden_size,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="patchmergerv2_proj2",
    )
    post_norm = vector_op(
        op_id=f"{prefix}projector_post_norm", deps=(proj2.op_id,),
        stage=stage, layer=op_layer, requests=requests,
        elements=max(1, text_positions * vision.out_hidden_size),
        ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
        family="rmsnorm", phase=mode,
        recipe="catalog_kimi_projector_post_norm",
        shape=(text_positions, vision.out_hidden_size),
        metadata=(("tokens", text_positions),
                  ("reduction_width", vision.out_hidden_size),
                  ("norm_type", "rmsnorm"),
                  ("norm_eps", vision.norm_eps)),
    )
    return [proj0, proj0_gelu, proj2, post_norm], post_norm.op_id


def _lower_spatial_merger_projector(
    *, merge_id: str, vision: VisionSpec, text_positions: int,
    dtype: str, quant: QuantSpec, mode: str, requests: tuple[str, ...],
    stage: int, vision_tp: int, prefix: str, op_layer: int,
    use_postshuffle_norm: bool,
) -> tuple[list[OpSpec], str]:
    """Lower a spatial-merge MLP projector (``aligner_hidden_size == 0``)."""
    proj_tp = vision_tp if (vision_tp > 1 and vision.aligner_input_width % vision_tp == 0) else 1
    local_mid = vision.aligner_input_width // proj_tp
    proj_ops: list[OpSpec] = []
    w1_dep = merge_id
    if use_postshuffle_norm:
        merger_norm = vector_op(
            op_id=f"{prefix}merger_norm", deps=(merge_id,), stage=stage,
            layer=op_layer, requests=requests,
            elements=max(1, text_positions * vision.aligner_input_width),
            ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
            family="layernorm", phase=mode, recipe="catalog_qwen_merger_norm",
            shape=(text_positions, vision.aligner_input_width),
            metadata=(("tokens", text_positions),
                      ("reduction_width", vision.aligner_input_width),
                      ("norm_type", "layernorm"),
                      ("norm_eps", vision.norm_eps)),
        )
        proj_ops.append(merger_norm)
        w1_dep = merger_norm.op_id
    merger_w1 = gemm_op(
        op_id=f"{prefix}merger_w1", deps=(w1_dep,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=local_mid, k=vision.aligner_input_width,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="qwen_merger_w1",
        semantic_shape=(text_positions, vision.aligner_input_width, vision.aligner_input_width),
    )
    merger_w1_bias = vector_op(
        op_id=f"{prefix}merger_w1_bias", deps=(merger_w1.op_id,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * local_mid),
        ops_per_element=1.0, dtype=_op_output_dtype(merger_w1, dtype),
        family="elementwise", phase=mode, recipe="catalog_qwen_merger_bias",
        semantic_family="qwen_merger_w1_bias",
    )
    merger_gelu = vector_op(
        op_id=f"{prefix}merger_gelu", deps=(merger_w1_bias.op_id,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * local_mid),
        ops_per_element=VISION_ACT_OPS_PER_ELEMENT, dtype=dtype,
        family="gelu", phase=mode, recipe="catalog_qwen_merger_gelu",
    )
    merger = gemm_op(
        op_id=f"{prefix}merger", deps=(merger_gelu.op_id,), stage=stage,
        layer=op_layer, requests=requests, m=text_positions,
        n=vision.out_hidden_size, k=local_mid,
        quant=quant, category="projector", phase=mode,
        family="vision_projector", semantic_family="qwen_merger_w2",
        semantic_shape=(text_positions, vision.out_hidden_size, vision.aligner_input_width),
    )
    proj_ops.extend([merger_w1, merger_w1_bias, merger_gelu, merger])
    merger_ready = merger.op_id
    merger_out_dtype = _op_output_dtype(merger, dtype)
    if proj_tp > 1:
        merger_ar = _vision_allreduce(
            op_id=f"{prefix}merger_allreduce", deps=(merger_ready,),
            stage=stage, layer=op_layer, requests=requests,
            rows=text_positions, cols=vision.out_hidden_size,
            dtype=merger_out_dtype, tp=proj_tp, mode=mode,
            semantic_family="vision_projector_allreduce",
        )
        proj_ops.append(merger_ar)
        merger_ready = merger_ar.op_id
    merger_bias = vector_op(
        op_id=f"{prefix}merger_bias", deps=(merger_ready,), stage=stage,
        layer=op_layer, requests=requests,
        elements=max(1, text_positions * vision.out_hidden_size),
        ops_per_element=1.0, dtype=merger_out_dtype, family="elementwise",
        phase=mode, recipe="catalog_qwen_merger_bias",
    )
    proj_ops.append(merger_bias)
    return proj_ops, merger_bias.op_id


def _lower_vision_projector(
    *, merge_id: str, vision: VisionSpec, text_positions: int,
    dtype: str, quant: QuantSpec, mode: str, requests: tuple[str, ...],
    stage: int, vision_tp: int = 1, prefix: str = "vision.",
    layer: int | None = None, use_postshuffle_norm: bool = False,
) -> tuple[list[OpSpec], str]:
    """Lower the multimodal spatial-merge projector into the language width."""
    op_layer = vision.layers if layer is None else int(layer)
    if int(vision.aligner_hidden_size) > 0 and bool(vision.bias):
        return _lower_biased_aligner_projector(
            merge_id=merge_id, vision=vision, text_positions=text_positions,
            dtype=dtype, quant=quant, mode=mode, requests=requests,
            stage=stage, vision_tp=vision_tp, prefix=prefix, op_layer=op_layer,
            use_postshuffle_norm=use_postshuffle_norm,
        )
    if int(vision.aligner_hidden_size) > 0:
        return _lower_unbiased_patch_merger_projector(
            merge_id=merge_id, vision=vision, text_positions=text_positions,
            dtype=dtype, quant=quant, mode=mode, requests=requests,
            stage=stage, prefix=prefix, op_layer=op_layer,
        )
    return _lower_spatial_merger_projector(
        merge_id=merge_id, vision=vision, text_positions=text_positions,
        dtype=dtype, quant=quant, mode=mode, requests=requests,
        stage=stage, vision_tp=vision_tp, prefix=prefix, op_layer=op_layer,
        use_postshuffle_norm=use_postshuffle_norm,
    )


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
    post_merge = _post_merge_positions(patch_tokens, grid_h, grid_w, frames, vision)
    text_positions = int(media_tokens) if grid_h == 0 else post_merge
    seq_lens = _attention_seq_lens(patch_tokens, grid_h, grid_w, frames, vision)
    vision_tp = _vision_tp(runtime, parallel, vision)
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
    deepstack_outs: list[str] = []
    for layer in range(vision.layers):
        layer_ops, last = _lower_vision_layer(
            layer=layer, last=last, vision=vision, patch_tokens=patch_tokens,
            seq_lens=seq_lens, head_dim=head_dim, dtype=dtype, quant=quant,
            mode=mode, requests=requests, stage=stage, vision_tp=vision_tp,
        )
        ops.extend(layer_ops)
        for tap_pos, tap_layer in enumerate(vision.deepstack_visual_indexes):
            if tap_layer != layer:
                continue
            tap_prefix = f"vision.deepstack{tap_pos}."
            tap_merge = vector_op(
                op_id=f"{tap_prefix}merge", deps=(last,), stage=stage,
                layer=layer, requests=requests,
                elements=max(1, text_positions * vision.aligner_input_width),
                ops_per_element=1.0, dtype=dtype, family="vision_merge",
                phase=mode, recipe="catalog_vision_merge",
                semantic_family="spatial_merge_concat",
                shape=(text_positions, vision.aligner_input_width),
                metadata=(("merge_area", vision.merge_area),
                          ("patch_tokens", patch_tokens),
                          ("post_merge_tokens", text_positions),
                          ("deepstack_layer", layer),
                          ("deepstack_index", tap_pos)),
            )
            ops.append(tap_merge)
            tap_ops, tap_last = _lower_vision_projector(
                merge_id=tap_merge.op_id, vision=vision,
                text_positions=text_positions, dtype=dtype, quant=quant,
                mode=mode, requests=requests, stage=stage,
                vision_tp=vision_tp, prefix=tap_prefix, layer=layer,
                use_postshuffle_norm=True,
            )
            ops.extend(tap_ops)
            deepstack_outs.append(tap_last)
    final_norm = vector_op(
        op_id="vision.final_norm", deps=(last,), stage=stage, layer=vision.layers,
        requests=requests, elements=max(1, patch_tokens * vision.hidden_size),
        ops_per_element=VISION_NORM_OPS_PER_ELEMENT, dtype=dtype,
        family="layernorm" if vision.norm_type == "layernorm" else "rmsnorm",
        phase=mode, recipe="catalog_vision_final_norm",
        shape=(patch_tokens, vision.hidden_size),
        metadata=(("tokens", patch_tokens),
                  ("reduction_width", vision.hidden_size),
                  ("norm_type", vision.norm_type),
                  ("norm_eps", vision.norm_eps)),
    )
    ops.append(final_norm)
    merge_in_elems = max(1, patch_tokens * vision.hidden_size)
    merge_out_elems = max(1, text_positions * vision.aligner_input_width)
    tpool_reduces = _uses_spatial_temporal_tpool(vision) and merge_in_elems > merge_out_elems
    merge_dtype_bytes = dtype_bytes(dtype)
    merge_meta: tuple[tuple[str, Any], ...] = (
        ("merge_area", vision.merge_area),
        ("patch_tokens", patch_tokens),
        ("post_merge_tokens", text_positions),
    )
    if tpool_reduces:
        temporal = max(1, int(vision.temporal_patch))
        tpool_groups = max(1, math.ceil(max(1, int(frames)) / temporal))
        merge_meta += (("temporal_pool_groups", tpool_groups),)
    merge = vector_op(
        op_id="vision.merge", deps=(final_norm.op_id,), stage=stage,
        layer=vision.layers, requests=requests,
        elements=max(merge_in_elems, merge_out_elems) if tpool_reduces else merge_out_elems,
        ops_per_element=1.0, dtype=dtype, family="vision_merge", phase=mode,
        recipe="catalog_vision_merge",
        semantic_family="temporal_pool_spatial_merge" if tpool_reduces else "spatial_merge_concat",
        shape=(text_positions, vision.aligner_input_width),
        read_bytes_override=float(merge_in_elems) * merge_dtype_bytes if tpool_reduces else None,
        write_bytes_override=float(merge_out_elems) * merge_dtype_bytes if tpool_reduces else None,
        metadata=merge_meta,
    )
    ops.append(merge)
    proj_ops, last = _lower_vision_projector(
        merge_id=merge.op_id, vision=vision, text_positions=text_positions,
        dtype=dtype, quant=quant, mode=mode, requests=requests, stage=stage,
        vision_tp=vision_tp, prefix="vision.", layer=vision.layers,
        use_postshuffle_norm=False,
    )
    ops.extend(proj_ops)
    if deepstack_outs:
        total_streams = 1 + len(deepstack_outs)
        concat = vector_op(
            op_id="vision.deepstack_concat", deps=(last, *deepstack_outs),
            stage=stage, layer=vision.layers, requests=requests,
            elements=max(1, text_positions * vision.out_hidden_size * total_streams),
            ops_per_element=1.0, dtype=dtype, family="vision_merge", phase=mode,
            recipe="catalog_vision_merge",
            semantic_family="deepstack_feature_concat",
            shape=(text_positions, vision.out_hidden_size * total_streams),
            metadata=(("deepstack_streams", total_streams),
                      ("deepstack_visual_indexes", vision.deepstack_visual_indexes)),
        )
        ops.append(concat)
        last = concat.op_id
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
