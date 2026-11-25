"""TP MC2 overlap rewrite pass fusing ``AllGather+MatMul`` and ``MatMul+ReduceScatter``.

[Role]
  Rewrite sequence-parallel ``allgather -> gemm`` and ``gemm -> reduce_scatter``
  TP communication/compute boundaries into governed ``all_gather_matmul``
  (``aclnnAllGatherMatmulV2``) and ``matmul_reduce_scatter``
  (``aclnnMatmulReduceScatterV2``) operators executed by ``operators/mc2``.

[Invariants]
  * Preserves 100% of matrix FLOPs, matrix dtype attribution, and peer wire
    payload bytes; only eliminates redundant boundary serialization and
    intermediate HBM staging through governed MC2 chunked overlap.
  * Matches structural operator families, dependency edges, and TP geometry
    (``group_axis == "tp"``, ``world_size > 1``, ``m % world_size == 0``),
    never model names or SKU strings.
  * Quantized GEMMs that require explicit pre-quantization between ``allgather``
    and ``gemm`` stay untouched unless inputs are native float rows.

[Timeouts & Locks]
  Linear O(|V| + |E|) pass over one active layer graph; no I/O or locks.

[Side-Effect Policy]
  Pure functional transform returning a new immutable ``OpSpec`` list.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.fusion_regions import fold_region_metadata
from mindie_sim.prediction.implementations import rewire_replacements
from mindie_sim.prediction.rewrite_mhc_rules import users_map
from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND = "ascend"
_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_matmul_fp16_bf16.h",
        source_sha256="fe27d0c153368f57396c14bcab681e7a12c8a8b5f732bc45be4c393eb74d82a7",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/matmul_reduce_scatter_v2/op_kernel/arch35/matmul_reduce_scatter_fp16_bf16.h",
        source_sha256="121ad9870776e27c1986e612b047817267e0de8976236df7eac748ef1643eff3",
    ),
)

_AGMM_CONSUMER_FAMILIES = frozenset({
    "gemm", "qkv_projection", "gate_up_projection", "projection",
})
_MMRS_PRODUCER_FAMILIES = frozenset({
    "gemm", "output_projection", "o_projection", "down_projection", "projection",
})
_FLOAT_DTYPES = frozenset({"bf16", "fp16"})
_FLOAT_DTYPE_ALIASES = {"bfloat16": "bf16", "float16": "fp16"}
_GEMM_SHAPE_RANK = 3
_SINGLE_CHUNK = 1
_MIN_TP_WORLD = 2
_NO_PORTS: dict = {}


def _canonical_dtype(raw) -> str:
    norm = str(raw or "").strip().lower()
    return _FLOAT_DTYPE_ALIASES.get(norm, norm)


def _native_float_gemm_dtypes(
    gemm, gemm_meta: dict, *, comm_dtype: str | None = None,
) -> tuple[str, str] | None:
    """Return ``(compute_dtype, output_dtype)`` iff all GEMM operands are matching FP16/BF16."""
    if int(gemm_meta.get("matrix_groups", 1) or 1) != 1:
        return None
    pabi = getattr(gemm, "precision_abi", None)
    comp_dt = _canonical_dtype(
        gemm_meta.get("compute_format")
        or (getattr(pabi, "compute_dtype", None) if pabi is not None else None)
        or gemm.dtype
        or "bf16"
    )
    out_dt = _canonical_dtype(
        gemm_meta.get("output_format")
        or (getattr(pabi, "output_dtype", None) if pabi is not None else None)
        or comp_dt
    )
    act_dt = _canonical_dtype(
        gemm_meta.get("activation_format")
        or (getattr(pabi, "input_dtype", None) if pabi is not None else None)
        or comp_dt
    )
    weight_dt = _canonical_dtype(
        gemm_meta.get("weight_format")
        or (getattr(pabi, "weight_dtype", None) if pabi is not None else None)
        or comp_dt
    )
    if (
        comp_dt not in _FLOAT_DTYPES
        or out_dt != comp_dt
        or act_dt != comp_dt
        or weight_dt != comp_dt
    ):
        return None
    for mf_dt, mf_val in getattr(gemm, "matrix_flops", ()) or ():
        if mf_val and _canonical_dtype(mf_dt) != comp_dt:
            return None
    if comm_dtype and _canonical_dtype(comm_dtype) != comp_dt:
        return None
    return comp_dt, out_dt


def _tp_world_size(comm_meta: dict, m_dim: int) -> int:
    tok_global = int(comm_meta.get("tokens_global") or 0)
    tok_local = int(comm_meta.get("tokens_local") or 0)
    if tok_global > 0 and tok_local > 0 and tok_global % tok_local == 0:
        return tok_global // tok_local
    steps = int(comm_meta.get("steps") or 0)
    if steps > 0 and m_dim % (steps + 1) == 0:
        return steps + 1
    return 0


def _gemm_dims(op) -> tuple[int, int, int] | None:
    shape = tuple(op.local_shape or op.semantic_shape or ())
    if len(shape) != _GEMM_SHAPE_RANK:
        return None
    m_dim, n_dim, k_dim = int(shape[0]), int(shape[1]), int(shape[2])
    if m_dim <= 0 or n_dim <= 0 or k_dim <= 0:
        return None
    return m_dim, n_dim, k_dim


def _build_mc2_contract(
    *, abi: str, world: int, m_dim: int, k_dim: int, n_dim: int,
    compute_dtype: str, output_dtype: str,
) -> dict:
    return {
        "abi": abi,
        "world_size": world,
        "m": m_dim,
        "k": k_dim,
        "n": n_dim,
        "compute_dtype": compute_dtype,
        "output_dtype": output_dtype,
        "operand_dtype": compute_dtype if abi == "all_gather_matmul" else output_dtype,
        "internal_chunks": _SINGLE_CHUNK,
        "independent_buffers": True,
        "independent_handles": True,
    }


def tp_mc2_overlap_fusion(ops, source_dtype):
    """Fuse TP ``allgather -> gemm`` and ``gemm -> reduce_scatter`` into MC2 ops."""
    del source_dtype
    by_id = {op.op_id: op for op in ops}
    users = users_map(ops)
    replacements, removed, aliases = {}, set(), {}

    # Pass 1: Fuse TP AllGather -> GEMM into all_gather_matmul.
    for ag in ops:
        if ag.plan_required or ag.family != "allgather" or ag.op_id in removed:
            continue
        ag_meta = dict(ag.metadata)
        if ag_meta.get("group_axis") != "tp":
            continue
        ag_users = users.get(ag.op_id, [])
        if len(ag_users) != 1:
            continue
        gemm = by_id.get(ag_users[0])
        if (gemm is None or gemm.plan_required or gemm.op_id in replacements
                or gemm.family not in _AGMM_CONSUMER_FAMILIES
                or gemm.stage != ag.stage or (gemm.stream or "") != (ag.stream or "")):
            continue
        dims = _gemm_dims(gemm)
        if dims is None:
            continue
        m_dim, n_dim, k_dim = dims
        world = _tp_world_size(ag_meta, m_dim)
        if world < _MIN_TP_WORLD or m_dim % world != 0:
            continue
        gemm_meta = dict(gemm.metadata)
        dtypes = _native_float_gemm_dtypes(gemm, gemm_meta, comm_dtype=ag.dtype)
        if dtypes is None:
            continue
        comp_dt, out_dt = dtypes
        new_deps = tuple(dict.fromkeys(
            dep for d in gemm.deps for dep in (ag.deps if d == ag.op_id else (d,))
        ))
        contract = _build_mc2_contract(
            abi="all_gather_matmul", world=world, m_dim=m_dim, k_dim=k_dim, n_dim=n_dim,
            compute_dtype=comp_dt, output_dtype=out_dt,
        )
        fused_meta = fold_region_metadata(
            (ag, gemm),
            primary_anchor_id=gemm.op_id,
            extra=(
                ("group_axis", "tp"),
                ("mc2_contract", contract),
                ("fused_from", (ag.op_id, gemm.op_id)),
                ("absorbed_source_ops", (ag.op_id, gemm.op_id)),
                ("original_gemm_recipe", gemm.recipe),
            ),
        )
        replacements[gemm.op_id] = replace(
            gemm,
            family="all_gather_matmul",
            recipe="mc2_all_gather_matmul",
            deps=new_deps,
            link_bytes=float(ag.link_bytes or 0.0),
            metadata=fused_meta,
        )
        removed.add(ag.op_id)
        aliases[ag.op_id] = gemm.op_id

    # Pass 2: Fuse GEMM -> TP ReduceScatter into matmul_reduce_scatter.
    for rs in ops:
        if rs.plan_required or rs.family != "reduce_scatter" or rs.op_id in removed:
            continue
        rs_meta = dict(rs.metadata)
        if rs_meta.get("group_axis") != "tp" or len(rs.deps) != 1:
            continue
        gemm = by_id.get(rs.deps[0])
        if (gemm is None or gemm.plan_required or gemm.op_id in replacements
                or gemm.op_id in removed
                or gemm.family not in _MMRS_PRODUCER_FAMILIES
                or gemm.stage != rs.stage or (gemm.stream or "") != (rs.stream or "")):
            continue
        if users.get(gemm.op_id) != [rs.op_id]:
            continue
        dims = _gemm_dims(gemm)
        if dims is None:
            continue
        m_dim, n_dim, k_dim = dims
        world = _tp_world_size(rs_meta, m_dim)
        if world < _MIN_TP_WORLD or m_dim % world != 0:
            continue
        gemm_meta = dict(gemm.metadata)
        dtypes = _native_float_gemm_dtypes(gemm, gemm_meta, comm_dtype=rs.dtype)
        if dtypes is None:
            continue
        comp_dt, out_dt = dtypes
        contract = _build_mc2_contract(
            abi="matmul_reduce_scatter", world=world, m_dim=m_dim, k_dim=k_dim, n_dim=n_dim,
            compute_dtype=comp_dt, output_dtype=out_dt,
        )
        fused_meta = fold_region_metadata(
            (gemm, rs),
            primary_anchor_id=gemm.op_id,
            extra=(
                ("group_axis", "tp"),
                ("mc2_contract", contract),
                ("fused_from", (gemm.op_id, rs.op_id)),
                ("absorbed_source_ops", (gemm.op_id, rs.op_id)),
                ("original_gemm_recipe", gemm.recipe),
            ),
        )
        replacements[gemm.op_id] = replace(
            gemm,
            family="matmul_reduce_scatter",
            recipe="mc2_matmul_reduce_scatter",
            link_bytes=float(rs.link_bytes or 0.0),
            metadata=fused_meta,
        )
        removed.add(rs.op_id)
        aliases[rs.op_id] = gemm.op_id

    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=_NO_PORTS,
    )


__all__ = ["tp_mc2_overlap_fusion"]
