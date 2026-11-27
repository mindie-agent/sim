"""Diffusion attention and QKV preparation.

[Role] Build diffusion attention and QKV preparation contracts.
[Invariants] Sparse blocks, declared tensor layouts and materialized outputs determine work.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from math import prod
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

from .contracts import (
    _as_shape,
    _contract,
    _positive_int,
    _ratio,
    _read,
    _role,
    _shape,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/ops/norm_rope_concat/op_host/norm_rope_concat_tiling.cpp",
        source_sha256="c02f0f00530e8f5f763a349aab9baadb7717517b654805456146198af0c04efa",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/ops/eagle_quant_block_sparse_attention/op_host/eagle_quant_block_sparse_attention_def.cpp",
        source_sha256="db5020ae4ef6eb4dbcd7dbd4756354653c8a2372872e9bf8ac021b47d20cd385",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/ops/eagle_quant_block_sparse_attention/op_host/eagle_quant_block_sparse_attention_tiling.cpp",
        source_sha256="378a1ddd937d4f7ce40901c931883a46018dd31fa0fe5bf788d50e50fea4e9be",
    ),
)

_BSND_RANK = 4
_ROPE_RANK = 2
_ROPE_OPS_PER_ELEMENT = 3
_HEAD_DIM_AXIS = 3
_LAYER_NORM = 1
_AFFINE_LAYER_NORM = 2
_RMS_NORM = 3
_AFFINE_RMS_NORM = 4
_NORM_OPS_PER_ELEMENT = {
    0: 0.0,
    _LAYER_NORM: 6.0,
    _AFFINE_LAYER_NORM: 7.0,
    _RMS_NORM: 4.0,
    _AFFINE_RMS_NORM: 5.0,
}


def _check_norm_rope_bhd(
    role_name: str, shape: tuple[int, ...], b: int, h: int, d: int, error_code: str,
) -> None:
    if len(shape) != _BSND_RANK:
        raise ModelingError("mindie_sd_norm_rope_requires_BSND_shapes")
    if shape[0] != b or shape[2] != h or shape[_HEAD_DIM_AXIS] != d:
        raise ModelingError(f"{error_code}:{role_name}:{shape}:expected_bhd={(b, h, d)}")


def _append_norm_rope_rotary(
    meta, reads, compute, *,
    b: int, h: int, d: int, dtype: str,
    sq: int, sk: int, sv: int, seq: int, sek: int, sev: int,
    concat_order: int, q_src: str, k_src: str, enc_q_src: str | None, enc_k_src: str | None,
) -> tuple[str, str, str | None, str | None]:
    if d % 2 != 0:
        raise ModelingError(f"mindie_sd_norm_rope_head_dim_must_be_even:{d}")
    max_seq = max(sq + seq, sk + sek, sv + sev)
    raw_rope_shape = meta.get("rope_sin_shape")
    raw_rope_len = meta.get("rope_len")
    if raw_rope_shape is not None:
        rope_shape = _as_shape(raw_rope_shape, "rope_sin")
        if len(rope_shape) != _ROPE_RANK:
            raise ModelingError("mindie_sd_norm_rope_sin_shape_invalid")
        if rope_shape[1] != d:
            raise ModelingError(f"mindie_sd_norm_rope_sin_dim_mismatch:{rope_shape[1]}:{d}")
        rope_len = int(raw_rope_len) if raw_rope_len is not None else rope_shape[0]
        if rope_shape[0] != rope_len:
            raise ModelingError(f"mindie_sd_norm_rope_len_mismatch:{rope_len}:{rope_shape[0]}")
    else:
        rope_len = int(raw_rope_len) if raw_rope_len is not None else max_seq
        if rope_len < 0:
            raise ModelingError("mindie_sd_norm_rope_len_invalid")
        rope_shape = (rope_len, d)
    if rope_len > max_seq:
        raise ModelingError(f"mindie_sd_norm_rope_len_exceeds_max_seq:{rope_len}:{max_seq}")
    raw_cos_shape = meta.get("rope_cos_shape")
    if raw_cos_shape is not None:
        cos_shape = _as_shape(raw_cos_shape, "rope_cos")
        if cos_shape != rope_shape:
            raise ModelingError(f"mindie_sd_norm_rope_cos_shape_mismatch:{cos_shape}:{rope_shape}")
    rope_dtype = str(meta.get("rope_dtype", dtype)).strip().lower()
    if rope_dtype != str(dtype).strip().lower():
        raise ModelingError(f"mindie_sd_norm_rope_dtype_mismatch:{rope_dtype}:{dtype}")
    q_off, enc_q_off = (seq, 0) if concat_order == 1 else (0, sq)
    k_off, enc_k_off = (sek, 0) if concat_order == 1 else (0, sk)
    q_rot = max(0, min(q_off + sq, rope_len) - q_off)
    k_rot = max(0, min(k_off + sk, rope_len) - k_off)
    enc_q_rot = max(0, min(enc_q_off + seq, rope_len) - enc_q_off) if enc_q_src is not None else 0
    enc_k_rot = max(0, min(enc_k_off + sek, rope_len) - enc_k_off) if enc_k_src is not None else 0
    # NormRopeConcat runs ProcessOne independently for Q, encoder Q, K, encoder K,
    # loading one (d,) row from rope_sin and rope_cos per active rotated position.
    table_bytes = float((q_rot + k_rot + enc_q_rot + enc_k_rot) * d * dtype_bytes(rope_dtype))
    reads.append({**_read("rope_sin", rope_shape, rope_dtype), "bytes": table_bytes})
    reads.append({**_read("rope_cos", rope_shape, rope_dtype), "bytes": table_bytes})
    rot_specs = (
        ("query", q_src, q_rot),
        ("key", k_src, k_rot),
        ("encoder_query", enc_q_src, enc_q_rot),
        ("encoder_key", enc_k_src, enc_k_rot),
    )
    updated = {"query": q_src, "key": k_src, "encoder_query": enc_q_src, "encoder_key": enc_k_src}
    for role, source, rot_len in rot_specs:
        if source is None or rot_len <= 0:
            continue
        compute.append({
            "name": f"{role}_rope", "kind": "vector",
            "inputs": (source, "rope_sin", "rope_cos"),
            "outputs": (f"{role}_roped",),
            "vector_ops": float(b * h * rot_len * d * _ROPE_OPS_PER_ELEMENT),
        })
        updated[role] = f"{role}_roped"
    return updated["query"], updated["key"], updated["encoder_query"], updated["encoder_key"]


def _norm_rope_contract(op, meta, dtype, returns):
    """[Cohesive-Pipeline] Build the diffusion norm + RoPE operator contract."""
    q_shape = _as_shape(meta.get("query_shape", meta.get("q_shape", _shape(op))),
                        "query")
    k_shape = _as_shape(meta.get("key_shape", q_shape), "key")
    v_shape = _as_shape(meta.get("value_shape", q_shape), "value")
    if any(len(shape) != _BSND_RANK for shape in (q_shape, k_shape, v_shape)):
        raise ModelingError("mindie_sd_norm_rope_requires_BSND_shapes")
    b, sq, h, d = q_shape
    _check_norm_rope_bhd("key", k_shape, b, h, d, "mindie_sd_norm_rope_qkv_shape_mismatch")
    _check_norm_rope_bhd("value", v_shape, b, h, d, "mindie_sd_norm_rope_qkv_shape_mismatch")
    sk, sv = k_shape[1], v_shape[1]
    if sk != sv:
        raise ModelingError(f"mindie_sd_norm_rope_kv_seq_mismatch:{sk}:{sv}")
    enc_q = meta.get("encoder_query_shape")
    enc_k = meta.get("encoder_key_shape")
    enc_v = meta.get("encoder_value_shape")
    enc_q = _as_shape(enc_q, "encoder_query") if enc_q is not None else None
    enc_k = _as_shape(enc_k, "encoder_key") if enc_k is not None else None
    enc_v = _as_shape(enc_v, "encoder_value") if enc_v is not None else None
    for role_name, shape in (
        ("encoder_query", enc_q), ("encoder_key", enc_k), ("encoder_value", enc_v),
    ):
        if shape is not None:
            _check_norm_rope_bhd(
                role_name, shape, b, h, d, "mindie_sd_norm_rope_encoder_shape_mismatch",
            )
    seq = enc_q[1] if enc_q else 0
    sek = enc_k[1] if enc_k else 0
    sev = enc_v[1] if enc_v else 0
    if sek != sev:
        raise ModelingError(f"mindie_sd_norm_rope_encoder_kv_seq_mismatch:{sek}:{sev}")
    norm_type = int(meta.get("norm_type", 0))
    norm_added_type = int(meta.get("norm_added_type", 0))
    rope_type = int(meta.get("rope_type", 0))
    concat_order = int(meta.get("concat_order", 0))
    training = bool(meta.get("is_training", False))
    if norm_type not in _NORM_OPS_PER_ELEMENT or norm_added_type not in _NORM_OPS_PER_ELEMENT:
        raise ModelingError("mindie_sd_norm_rope_invalid_norm_type")
    if rope_type not in (0, 1, 2) or concat_order not in (0, 1):
        raise ModelingError("mindie_sd_norm_rope_invalid_rope_or_concat")
    materialized = tuple(meta.get("materialized_outputs", ("query", "key", "value")))
    if set(materialized) - {"query", "key", "value"}:
        raise ModelingError("mindie_sd_norm_rope_output_roles_unknown")
    aliases = dict(meta.get("output_aliases") or {})
    reads, writes, compute = [], [], []
    reads.extend((_read("query", q_shape, dtype), _read("key", k_shape, dtype),
                  _read("value", v_shape, dtype)))
    if enc_q:
        reads.append(_read("encoder_query", enc_q, dtype))
    if enc_k:
        reads.append(_read("encoder_key", enc_k, dtype))
    if enc_v:
        reads.append(_read("encoder_value", enc_v, dtype))
    q_src, k_src, v_src = "query", "key", "value"
    enc_q_src = "encoder_query" if enc_q else None
    enc_k_src = "encoder_key" if enc_k else None
    stats_sources = {}
    if norm_type:
        q_norm_inputs, k_norm_inputs = ["query"], ["key"]
        if norm_type in (_AFFINE_LAYER_NORM, _AFFINE_RMS_NORM):
            reads.append(_read("norm_query_weight", (d,), dtype))
            reads.append(_read("norm_key_weight", (d,), dtype))
            q_norm_inputs.append("norm_query_weight")
            k_norm_inputs.append("norm_key_weight")
            if norm_type == _AFFINE_LAYER_NORM:
                reads.append(_read("norm_query_bias", (d,), dtype))
                reads.append(_read("norm_key_bias", (d,), dtype))
                q_norm_inputs.append("norm_query_bias")
                k_norm_inputs.append("norm_key_bias")
        compute.append({"name": "query_norm", "kind": "vector",
                        "inputs": tuple(q_norm_inputs),
                        "outputs": ("query_normed", "norm_query_mean", "norm_query_rstd"),
                        "vector_ops": float(b * sq * h * d * _NORM_OPS_PER_ELEMENT[norm_type])})
        compute.append({"name": "key_norm", "kind": "vector",
                        "inputs": tuple(k_norm_inputs),
                        "outputs": ("key_normed", "norm_key_mean", "norm_key_rstd"),
                        "vector_ops": float(b * sk * h * d * _NORM_OPS_PER_ELEMENT[norm_type])})
        q_src, k_src = "query_normed", "key_normed"
        stats_sources.update({"norm_query": "query_norm", "norm_key": "key_norm"})
    if enc_q and norm_added_type:
        enc_q_inputs = ["encoder_query"]
        if norm_added_type in (_AFFINE_LAYER_NORM, _AFFINE_RMS_NORM):
            reads.append(_read("norm_added_query_weight", (d,), dtype))
            enc_q_inputs.append("norm_added_query_weight")
            if norm_added_type == _AFFINE_LAYER_NORM:
                reads.append(_read("norm_added_query_bias", (d,), dtype))
                enc_q_inputs.append("norm_added_query_bias")
        compute.append({"name": "encoder_query_norm", "kind": "vector",
                        "inputs": tuple(enc_q_inputs),
                        "outputs": ("encoder_query_normed",
                                    "norm_added_query_mean", "norm_added_query_rstd"),
                        "vector_ops": float(prod(enc_q) * _NORM_OPS_PER_ELEMENT[norm_added_type])})
        enc_q_src = "encoder_query_normed"
        stats_sources["norm_added_query"] = "encoder_query_norm"
    if enc_k and norm_added_type:
        enc_k_inputs = ["encoder_key"]
        if norm_added_type in (_AFFINE_LAYER_NORM, _AFFINE_RMS_NORM):
            reads.append(_read("norm_added_key_weight", (d,), dtype))
            enc_k_inputs.append("norm_added_key_weight")
            if norm_added_type == _AFFINE_LAYER_NORM:
                reads.append(_read("norm_added_key_bias", (d,), dtype))
                enc_k_inputs.append("norm_added_key_bias")
        compute.append({"name": "encoder_key_norm", "kind": "vector",
                        "inputs": tuple(enc_k_inputs),
                        "outputs": ("encoder_key_normed",
                                    "norm_added_key_mean", "norm_added_key_rstd"),
                        "vector_ops": float(prod(enc_k) * _NORM_OPS_PER_ELEMENT[norm_added_type])})
        enc_k_src = "encoder_key_normed"
        stats_sources["norm_added_key"] = "encoder_key_norm"
    if rope_type:
        q_src, k_src, enc_q_src, enc_k_src = _append_norm_rope_rotary(
            meta, reads, compute,
            b=b, h=h, d=d, dtype=dtype,
            sq=sq, sk=sk, sv=sv, seq=seq, sek=sek, sev=sev,
            concat_order=concat_order, q_src=q_src, k_src=k_src,
            enc_q_src=enc_q_src, enc_k_src=enc_k_src,
        )
    q_out = (b, h, sq + seq, d)
    k_out = (b, h, sk + sek, d)
    v_out = (b, h, sv + sev, d)
    output_shapes = {"query": q_out, "key": k_out, "value": v_out}
    for role_name in ("query", "key", "value"):
        shape = output_shapes[role_name]
        alias = aliases.get(role_name)
        material = role_name in materialized and alias is None
        if not material:
            writes.append(_role(role_name + "_output", shape, dtype,
                                materialized=False, alias=alias))
            continue
        if role_name == "query":
            inputs = [q_src] + ([enc_q_src] if enc_q_src else [])
        elif role_name == "key":
            inputs = [k_src] + ([enc_k_src] if enc_k_src else [])
        else:
            inputs = [v_src] + (["encoder_value"] if enc_v else [])
        compute_name = role_name + "_concat_transpose"
        compute.append({"name": compute_name, "kind": "layout",
                        "inputs": tuple(inputs),
                        "outputs": (role_name + "_output_value",),
                        "vector_ops": float(prod(shape))})
        writes.append({**_role(role_name + "_output", shape, dtype),
                       "source": compute_name})
    if training and norm_type in (_LAYER_NORM, _AFFINE_LAYER_NORM):
        for name, length in (("norm_query", sq), ("norm_key", sk)):
            for suffix in ("mean", "rstd"):
                writes.append({**_role(f"{name}_{suffix}", (b, length, h, 1), "fp32"),
                               "source": stats_sources[name]})
    if training and norm_added_type in (_LAYER_NORM, _AFFINE_LAYER_NORM):
        for name, shape in (("norm_added_query", enc_q), ("norm_added_key", enc_k)):
            if shape:
                for suffix in ("mean", "rstd"):
                    writes.append({**_role(f"{name}_{suffix}", (b, shape[1], h, 1),
                                           "fp32"),
                                   "source": stats_sources[name]})
    return _contract(reads, writes, compute, attributes={
        "norm_type": norm_type, "norm_added_type": norm_added_type,
        "rope_type": rope_type, "concat_order": concat_order,
        "is_training": training, "materialized_outputs": materialized,
        "output_aliases": aliases,
    }, basis="per_value_BSND_norm_rope_concat_dag",
        limitations=("encoder_stats_only_when_layer_norm_and_training",))


def _sparse_shape(meta, op, name, default):
    raw = meta.get(f"{name}_shape", default)
    shape = _as_shape(raw, name)
    if len(shape) == 4:
        layout = str(meta.get("layout", meta.get(f"{name}_layout", "BSND"))).upper()
        if layout == "BNSD":
            b, h, s, d = shape
            canonical = (b, s, h, d)
        else:
            b, s, h, d = shape
            canonical = (b, s, h, d)
            layout = "BSND"
        lengths = tuple(int(v) for v in meta.get(
            f"{name}_lengths", (s,) * b))
        if len(lengths) != b or any(v < 0 or v > s for v in lengths):
            raise ModelingError(f"mindie_sd_sparse_{name}_lengths_invalid")
        return canonical, lengths, layout
    if len(shape) == 3:
        total, h, d = shape
        lengths = tuple(int(v) for v in meta.get(
            f"{name}_lengths", meta.get("sequence_lengths", (total,))))
        if not lengths or any(v < 0 for v in lengths) or sum(lengths) > total:
            raise ModelingError(f"mindie_sd_sparse_{name}_TND_lengths_invalid")
        return shape, lengths, "TND"
    raise ModelingError(f"mindie_sd_sparse_{name}_requires_BNSD_or_TND")


def _is_mask_flag(value) -> bool:
    return isinstance(value, (int, bool))


def _parse_head_mask(
    head, qb: int, kb: int, max_qb: int, max_kb: int, phys_qb: int, phys_kb: int,
) -> tuple[tuple[int, ...], tuple[int, int] | None]:
    if not isinstance(head, (tuple, list)):
        raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    if all(_is_mask_flag(value) for value in head):
        if len(head) == qb * kb:
            return tuple(1 if value else 0 for value in head), None
        for grid_q, grid_k in ((max_qb, max_kb), (phys_qb, phys_kb)):
            if len(head) == grid_q * grid_k:
                return tuple(
                    1 if head[qi * grid_k + ki] else 0
                    for qi in range(qb) for ki in range(kb)
                ), (grid_q, grid_k)
        raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    if (len(head) == qb
            and all(isinstance(item, (tuple, list)) and len(item) == kb
                    and all(_is_mask_flag(value) for value in item)
                    for item in head)):
        rect = (qb, kb) if (qb == max_qb and kb == max_kb) else None
        return tuple(
            1 if head[qi][ki] else 0
            for qi in range(qb) for ki in range(kb)
        ), rect
    for grid_q, grid_k in ((max_qb, max_kb), (phys_qb, phys_kb)):
        if (len(head) == grid_q
                and all(isinstance(item, (tuple, list)) and len(item) == grid_k
                        and all(_is_mask_flag(value) for value in item)
                        for item in head)):
            return tuple(
                1 if head[qi][ki] else 0
                for qi in range(qb) for ki in range(kb)
            ), (grid_q, grid_k)
    raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")


def _sparse_mask_flags(
    mask_values, total_pairs, per_batch_pairs, b, hq, phys_qb: int = 0, phys_kb: int = 0,
):
    if mask_values is None:
        return None, None, None
    if not isinstance(mask_values, (tuple, list)):
        raise ModelingError("mindie_sd_sparse_block_mask_invalid")
    if all(_is_mask_flag(value) for value in mask_values):
        if len(mask_values) == total_pairs:
            return tuple(1 if value else 0 for value in mask_values), None, None
        if (per_batch_pairs
                and all(qb * kb == len(mask_values) for _, qb, kb in per_batch_pairs)):
            return tuple(1 if value else 0 for value in mask_values) * b, None, None
        raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    if len(mask_values) < b:
        raise ModelingError("mindie_sd_sparse_block_mask_batch_missing")
    if len(mask_values) != b:
        raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    max_qb = max((qb for _, qb, _ in per_batch_pairs), default=0)
    max_kb = max((kb for _, _, kb in per_batch_pairs), default=0)
    eff_phys_qb = max(max_qb, phys_qb)
    eff_phys_kb = max(max_kb, phys_kb)
    head_rows = []
    rect_grid: tuple[int, int] | None = None
    for batch_index, qb, kb in per_batch_pairs:
        row = mask_values[batch_index]
        if not isinstance(row, (tuple, list)):
            raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
        if all(_is_mask_flag(value) for value in row):
            if len(row) == qb * kb:
                flags = tuple(1 if value else 0 for value in row)
            elif len(row) == max_qb * max_kb:
                flags = tuple(
                    1 if row[qi * max_kb + ki] else 0
                    for qi in range(qb) for ki in range(kb)
                )
                rect_grid = (max_qb, max_kb)
            elif len(row) == eff_phys_qb * eff_phys_kb:
                flags = tuple(
                    1 if row[qi * eff_phys_kb + ki] else 0
                    for qi in range(qb) for ki in range(kb)
                )
                rect_grid = (eff_phys_qb, eff_phys_kb)
            else:
                raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
            head_rows.append(tuple(flags for _ in range(hq)))
        elif len(row) == hq:
            parsed = tuple(
                _parse_head_mask(head, qb, kb, max_qb, max_kb, eff_phys_qb, eff_phys_kb)
                for head in row
            )
            for _, grid in parsed:
                if grid is not None and (rect_grid is None or grid > rect_grid):
                    rect_grid = grid
            head_rows.append(tuple(flags for flags, _ in parsed))
        else:
            raise ModelingError("mindie_sd_sparse_block_mask_shape_mismatch")
    return None, tuple(head_rows), rect_grid


_SPARSE_MAC_FACTOR = 2
_SPARSE_BLOCK_SHAPE_RANK = 2
_EAGLE_BSA_KV_BLOCK_ALIGN = 16
_EAGLE_QBSA_KV_BLOCK_ALIGN = 128
_SPARSE_DTYPE_ALIASES = {
    "fp8_e4m3fn": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
}
_QBSA_QMODE1_VALUE_DTYPES = frozenset({"int8", "fp8_e4m3"})
_QBSA_FP8_KV_DTYPES = frozenset({"fp8_e4m3", "fp8_e5m2"})


def _qbsa_dtypes_supported(q_dtype: str, k_dtype: str, v_dtype: str) -> bool:
    """Enforce pinned 950 qmode1 host tiler contract (tiling.cpp:202) while preserving FP8 KV."""
    if q_dtype == "int8" or k_dtype == "int8":
        return q_dtype == "int8" and k_dtype == "int8" and v_dtype in _QBSA_QMODE1_VALUE_DTYPES
    return k_dtype in _QBSA_FP8_KV_DTYPES and v_dtype in _QBSA_FP8_KV_DTYPES


def _resolve_sparse_block_shape(meta, options, resolved: str) -> tuple[int, int]:
    raw_shape = options.get("block_shape")
    if raw_shape is None:
        raw_shape = meta.get("block_shape")
    if raw_shape is not None:
        shape = _as_shape(raw_shape, "block_shape")
        if len(shape) != _SPARSE_BLOCK_SHAPE_RANK or shape[0] <= 0 or shape[1] <= 0:
            raise ModelingError("mindie_sd_sparse_block_shape_invalid")
        if resolved == "eagle_qbsa" and shape[1] % _EAGLE_QBSA_KV_BLOCK_ALIGN != 0:
            raise ModelingError("mindie_sd_sparse_block_shape_invalid")
        if resolved == "eagle_bsa" and shape[1] % _EAGLE_BSA_KV_BLOCK_ALIGN != 0:
            raise ModelingError("mindie_sd_sparse_block_shape_invalid")
        return shape[0], shape[1]
    block = _positive_int(meta, options.get("block_size"), "block_size")
    return block, block


def _norm_sparse_dtype(raw: str) -> str:
    clean = str(raw).strip().lower()
    return _SPARSE_DTYPE_ALIASES.get(clean, clean)


def _sparse_legacy_contract(op, dtype, ratio, block, meta, options):
    if op.flops <= 0:
        raise ModelingError("mindie_sd_sparse_requires_declared_logical_flops")
    q_legacy = _shape(op)
    active = float(op.flops) * (1.0 - ratio)
    reads = [{"name": "declared_operand", "shape": q_legacy, "dtype": dtype,
              "bytes": float(op.mte2_bytes), "materialized": True,
              "alias": None, "kind": "legacy", "producer": None}]
    writes = [{"name": "attention_out", "shape": q_legacy, "dtype": dtype,
               "bytes": float(op.mte3_bytes), "materialized": True,
               "alias": None, "kind": "legacy", "source": "declared_sparse"}]
    compute = [{"name": "declared_sparse", "kind": "matrix",
                "matrix_flops": ((dtype, active),),
                "vector_ops": float(op.vector_ops),
                "inputs": ("declared_operand",),
                "outputs": ("attention_out",)}]
    return _contract(reads, writes, compute, attributes={
        "sparsity_ratio": ratio, "block_size": block,
        "kv_dtype": options.get("kv_dtype") or meta.get("kv_dtype"),
    }, basis="declared_single_shape_legacy",
        limitations=("qkv_roles_not_declared", "aggregate_kv_not_inferred"))


def _sparse_active_totals(
    per_batch_pairs, q_lengths, k_lengths, block_q, block_kv, ratio, flat, head_flags, hq, hkv, dq, dv,
):
    qk_flops = pv_flops = 0.0
    active_score_elements = active_pairs = 0
    kv_blocks = set()
    if head_flags is None:
        cursor = 0
        for batch_index, qb, kb in per_batch_pairs:
            ql, kl = q_lengths[batch_index], k_lengths[batch_index]
            selected = []
            if flat is None:
                selected_count = max(0, min(qb * kb, int(round((1.0 - ratio) * qb * kb))))
                for index in range(selected_count):
                    selected.append(divmod(index, kb))
            else:
                for qi in range(qb):
                    for ki in range(kb):
                        if flat[cursor]:
                            selected.append((qi, ki))
                        cursor += 1
            for qi, ki in selected:
                qrows, rows = min(block_q, ql - qi * block_q), min(block_kv, kl - ki * block_kv)
                score_elements = qrows * rows * hq
                active_pairs += 1
                active_score_elements += score_elements
                qk_flops += _SPARSE_MAC_FACTOR * score_elements * dq
                pv_flops += _SPARSE_MAC_FACTOR * score_elements * dv
                for kv_head in range(hkv):
                    kv_blocks.add((batch_index, kv_head, ki))
        return qk_flops, pv_flops, active_score_elements, active_pairs, kv_blocks
    group_size = hq // hkv
    for batch_index, qb, kb in per_batch_pairs:
        ql, kl = q_lengths[batch_index], k_lengths[batch_index]
        for head in range(hq):
            flags = head_flags[batch_index][head]
            kv_head = head // group_size
            for qi in range(qb):
                for ki in range(kb):
                    if not flags[qi * kb + ki]:
                        continue
                    qrows, rows = min(block_q, ql - qi * block_q), min(block_kv, kl - ki * block_kv)
                    score_elements = qrows * rows
                    active_pairs += 1
                    active_score_elements += score_elements
                    qk_flops += _SPARSE_MAC_FACTOR * score_elements * dq
                    pv_flops += _SPARSE_MAC_FACTOR * score_elements * dv
                    kv_blocks.add((batch_index, kv_head, ki))
    return qk_flops, pv_flops, active_score_elements, active_pairs, kv_blocks


def _sparse_contract(op, meta, dtype, returns, resolved, options):
    """[Cohesive-Pipeline] Validate and assemble one sparse diffusion attention physical contract."""
    ratio = _ratio(meta, options.get("sparsity_ratio"), "sparsity_ratio")
    if not (0.0 <= ratio < 1.0):
        raise ModelingError(f"mindie_sd_invalid_sparsity_ratio:{ratio}")
    block_q, block_kv = _resolve_sparse_block_shape(meta, options, resolved)
    q = meta.get("query_shape")
    k = meta.get("key_shape")
    v = meta.get("value_shape")
    if q is None or k is None or v is None:
        return _sparse_legacy_contract(op, dtype, ratio, block_q, meta, options)
    q_shape, q_lengths, q_layout = _sparse_shape(meta, op, "query", q)
    k_shape, k_lengths, k_layout = _sparse_shape(meta, op, "key", k)
    v_shape, v_lengths, v_layout = _sparse_shape(meta, op, "value", v)
    if not (q_layout == k_layout == v_layout):
        raise ModelingError("mindie_sd_sparse_layouts_must_match")
    if len(q_lengths) != len(k_lengths) or len(q_lengths) != len(v_lengths):
        raise ModelingError("mindie_sd_sparse_batch_lengths_mismatch")
    if q_layout == "TND" and resolved in {"eagle_bsa", "eagle_qbsa"}:
        # eagle_block_sparse_attention_tiling.cpp:276-289 requires packed TND length sums == T.
        if sum(q_lengths) != q_shape[0]:
            raise ModelingError("mindie_sd_sparse_query_TND_lengths_invalid")
        if sum(k_lengths) != k_shape[0]:
            raise ModelingError("mindie_sd_sparse_key_TND_lengths_invalid")
        if sum(v_lengths) != v_shape[0]:
            raise ModelingError("mindie_sd_sparse_value_TND_lengths_invalid")
    b = len(q_lengths)
    if len(q_shape) == 4:
        _, _, hq, dq = q_shape
        _, _, hkv, dk = k_shape
        _, _, hv, dv = v_shape
    else:
        _, hq, dq = q_shape
        _, hkv, dk = k_shape
        _, hv, dv = v_shape
    if hkv != hv or dq != dk or hq % hkv:
        raise ModelingError("mindie_sd_sparse_qkv_head_geometry_mismatch")
    opt_kv_dtype = options.get("kv_dtype")
    meta_kv_dtype = meta.get("kv_dtype")
    declared_kv = opt_kv_dtype if opt_kv_dtype is not None else meta_kv_dtype
    q_dtype = _norm_sparse_dtype(meta.get("q_dtype", meta.get("query_dtype", dtype)))
    raw_k = meta.get("k_dtype")
    raw_v = meta.get("v_dtype")
    k_resolved = raw_k if raw_k is not None else declared_kv
    v_resolved = raw_v if raw_v is not None else declared_kv
    if resolved == "eagle_qbsa" and (k_resolved is None or v_resolved is None):
        raise ModelingError("mindie_sd_eagle_qbsa_requires_declared_cache_dtype")
    k_dtype = _norm_sparse_dtype(k_resolved if k_resolved is not None else dtype)
    v_dtype = _norm_sparse_dtype(v_resolved if v_resolved is not None else dtype)
    if resolved == "eagle_qbsa" and not _qbsa_dtypes_supported(q_dtype, k_dtype, v_dtype):
        raise ModelingError(
            f"mindie_sd_eagle_qbsa_cache_dtype_unsupported:k={k_dtype}:v={v_dtype}")
    per_batch_pairs = []
    for b_index, (ql, kl) in enumerate(zip(q_lengths, k_lengths)):
        per_batch_pairs.append((b_index, (ql + block_q - 1) // block_q,
                                (kl + block_kv - 1) // block_kv))
    total_pairs = sum(qb * kb for _, qb, kb in per_batch_pairs)
    phys_qb = (q_shape[1] + block_q - 1) // block_q if q_layout != "TND" else 0
    phys_kb = (k_shape[1] + block_kv - 1) // block_kv if q_layout != "TND" else 0
    mask_values = meta.get("block_mask")
    flat, head_flags, mask_rect_grid = _sparse_mask_flags(
        mask_values, total_pairs, per_batch_pairs, b, hq, phys_qb=phys_qb, phys_kb=phys_kb)
    qk_flops, pv_flops, active_score_elements, active_pairs, kv_blocks = _sparse_active_totals(
        per_batch_pairs, q_lengths, k_lengths, block_q, block_kv, ratio, flat, head_flags, hq, hkv, dq, dv)
    k_elem_bytes = dtype_bytes(k_dtype)
    v_elem_bytes = dtype_bytes(v_dtype)
    kv_load_bytes = 0.0
    for batch_index, _kv_head, ki in kv_blocks:
        rows = min(block_kv, k_lengths[batch_index] - ki * block_kv)
        kv_load_bytes += rows * (dk * k_elem_bytes + dv * v_elem_bytes)
    if head_flags is not None:
        max_qb = max((qb for _, qb, _ in per_batch_pairs), default=0)
        max_kb = max((kb for _, _, kb in per_batch_pairs), default=0)
        mask_dtype = str(meta.get("mask_dtype", "int8"))
        uniform_grid = all(qb == max_qb and kb == max_kb for _, qb, kb in per_batch_pairs)
        if mask_rect_grid is not None:
            mask_shape = (b, hq, mask_rect_grid[0], mask_rect_grid[1])
        elif uniform_grid:
            mask_shape = (b, hq, max_qb, max_kb)
        else:
            mask_shape = (sum(qb * kb for _, qb, kb in per_batch_pairs) * hq,)
        mask_bytes = float(meta.get("mask_bytes", prod(mask_shape) * dtype_bytes(mask_dtype)))
    else:
        mask_dtype = str(meta.get("mask_dtype", "int32"))
        mask_shape = (total_pairs,)
        mask_bytes = float(meta.get("mask_bytes", total_pairs * dtype_bytes(mask_dtype)))
    reads = [_read("query", q_shape, q_dtype),
             {"name": "key", "shape": k_shape, "dtype": k_dtype,
              "bytes": 0.0, "materialized": True, "alias": None,
              "kind": "paged_kv_view", "producer": None},
             {"name": "value", "shape": v_shape, "dtype": v_dtype,
              "bytes": 0.0, "materialized": True, "alias": None,
              "kind": "paged_kv_view", "producer": None},
             {"name": "kv_load", "shape": (active_pairs,), "dtype": k_dtype,
              "bytes": kv_load_bytes, "materialized": True, "alias": None,
              "kind": "block_load", "producer": None},
             {"name": "block_mask", "shape": mask_shape, "dtype": mask_dtype,
              "bytes": mask_bytes, "materialized": True, "alias": None,
              "kind": "mask", "producer": None}]
    for name, shape_key, dt_key in (
            ("q_scale", "q_scale_shape", "q_scale_dtype"),
            ("k_scale", "k_scale_shape", "k_scale_dtype"),
            ("v_scale", "v_scale_shape", "v_scale_dtype")):
        shape = meta.get(shape_key)
        if shape is not None:
            reads.append(_read(name, _as_shape(shape, name),
                               str(meta.get(dt_key, "fp32"))))
    # Native MindIE-SD wrappers allocate outputs from physical query storage extents.
    if q_layout == "BNSD":
        phys_q_shape = (b, hq, q_shape[1], dq)
        out_shape = (b, hq, q_shape[1], dv)
    elif q_layout == "TND":
        phys_q_shape = (q_shape[0], hq, dq)
        out_shape = (q_shape[0], hq, dv)
    else:
        phys_q_shape = (b, q_shape[1], hq, dq)
        out_shape = (b, q_shape[1], hq, dv)
    default_out_dtype = (
        "bf16"
        if (resolved == "eagle_qbsa"
            and (q_dtype == "int8" or meta.get("q_scale_shape") is not None))
        else dtype
    )
    out_dtype = _norm_sparse_dtype(meta.get("output_dtype", default_out_dtype))
    active_q_tokens = sum(q_lengths)
    # Keep physical allocation shape on output descriptors, but charge MTE3/HBM
    # only for active query rows written by the kernel (kernel_regular_arch22.h:553).
    out_write = {**_role("attention_out", out_shape, out_dtype), "source": "pv"}
    out_write["bytes"] = float(active_q_tokens * hq * dv * dtype_bytes(out_dtype))
    writes = [out_write]
    explicit_lse_shape = meta.get("lse_shape")
    lse_shape = explicit_lse_shape
    default_lse_dtype = q_dtype if resolved == "rainfusion" else "fp32"
    lse_dtype = _norm_sparse_dtype(meta.get("lse_dtype", default_lse_dtype))
    # rainfusionattention.cpp:57-63 unconditionally allocates and returns softmaxLse.
    if lse_shape is None and (resolved == "rainfusion" or bool(meta.get("return_softmax_lse", False))):
        if resolved == "rainfusion":
            lse_shape = phys_q_shape[:3]
        elif q_layout == "BNSD":
            lse_shape = (b, hq, q_shape[1], 1)
        elif q_layout == "TND":
            lse_shape = (q_shape[0], hq, 1)
        else:
            lse_shape = (b, q_shape[1], hq, 1)
    if lse_shape is not None:
        lse_shape = _as_shape(lse_shape, "lse")
        lse_write = {**_role("softmax_lse", lse_shape, lse_dtype), "source": "softmax"}
        if explicit_lse_shape is None and resolved != "rainfusion":
            lse_write["bytes"] = float(active_q_tokens * hq * dtype_bytes(lse_dtype))
        writes.append(lse_write)
    scale_inputs = tuple(item["name"] for item in reads
                         if item["name"] in {"q_scale", "k_scale", "v_scale"})
    pv_dtype = (
        "fp8_e4m3"
        if (resolved == "eagle_qbsa" and v_dtype == "int8")
        else v_dtype
    )
    compute = [
        {"name": "qk", "kind": "matrix",
         "matrix_flops": ((q_dtype, qk_flops),),
         "inputs": ("query", "kv_load", "block_mask") + scale_inputs,
         "outputs": ("qk_scores",)},
        {"name": "softmax", "kind": "vector",
         "vector_ops": float(active_score_elements * 2),
         "inputs": ("qk_scores", "block_mask"),
         "outputs": ("probs", "softmax_lse")},
        {"name": "pv", "kind": "matrix",
         "matrix_flops": ((pv_dtype, pv_flops),),
         "inputs": ("probs", "kv_load") + scale_inputs,
         "outputs": ("attention_out",)},
    ]
    return _contract(reads, writes, compute, attributes={
        "sparsity_ratio": ratio,
        "block_size": block_q if block_q == block_kv else block_kv,
        "block_shape": (block_q, block_kv),
        "active_block_pairs": active_pairs, "block_pairs": total_pairs,
        "active_rows": active_score_elements, "layout": q_layout, "batch": b,
        "q_lengths": q_lengths, "k_lengths": k_lengths, "v_lengths": v_lengths,
        "kv_dtype": k_dtype, "q_dtype": q_dtype, "k_dtype": k_dtype,
        "v_dtype": v_dtype, "output_dtype": out_dtype,
        "mask_bytes": mask_bytes, "active_kv_blocks": len(kv_blocks),
        "mask_layout": ("B,H,Qblock,KVblock" if head_flags is not None
                        else "global"),
        "kv_load_bytes": kv_load_bytes,
    }, basis="BNSD_TND_batch_heads_actual_active_blocks",
        limitations=("kv_block_reuse_modelled_per_batch_block",))
