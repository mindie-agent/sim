"""Default per-site compute, vector, link, and boundary obligations for dual bounds.

[Role] Derive universal per-(replica, stage, die, op) semantic work, collective
  egress, cold-weight, and pure-copy boundary obligations from a resolved
  scenario without shutting off inference globally when a subset of operators
  already carries explicit ``semantic_work``.
[Invariants] Every inferred obligation is sound under all valid framework and
  kernel rewrites: matrix MACs, non-absorbable vector ops, multi-rank collective
  link egress, cold weight reads, and un-fusable pure-copy graph boundaries.
[Timeouts & Locks] Single linear pass over scenario rank graphs; no I/O or locks.
[Side-Effect Policy] Pure read-only derivation returning immutable obligations.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction

from .contracts import FusedRegionSummary, SemanticWorkItem, TensorRangeObligation

_COLD_WEIGHT_FAMILIES = frozenset({
    "gemm", "linear", "qkv_projection", "output_projection", "gate_up_projection",
    "down_projection", "projection", "lm_head", "logits",
})
_WEIGHT_ID_KEYS = ("weight_allocation_id", "allocation_id")


def _weight_allocation_id(meta: Mapping[str, object]) -> str | None:
    return next((value for key in _WEIGHT_ID_KEYS
                 if isinstance(value := meta.get(key), str) and value.strip()), None)


def _cold_weight_read(op, meta: Mapping[str, object], die: int) -> TensorRangeObligation | None:
    mte2_bytes = getattr(op, "mte2_bytes", 0)
    if mte2_bytes <= 0 or op.family not in _COLD_WEIGHT_FAMILIES:
        return None
    identity = _weight_allocation_id(meta)
    residency = meta.get("weight_initial_residency", meta.get("initial_residency"))
    if identity is None or residency != "hbm_cold":
        return None
    decl = meta.get("catlass_problem")
    raw_base = decl.get("b_base_byte", 0) if isinstance(decl, Mapping) else getattr(decl, "b_base_byte", 0)
    raw_ver = decl.get("b_version", 0) if isinstance(decl, Mapping) else getattr(decl, "b_version", 0)
    decl_base = raw_base if type(raw_base) is int and raw_base >= 0 else 0
    decl_ver = raw_ver if type(raw_ver) is int and raw_ver >= 0 else 0
    ranges = meta.get("weight_ranges", meta.get("ranges"))
    if ranges is None:
        size = meta.get("weight_payload_bytes") or meta.get("weight_bytes") or meta.get("local_weight_bytes")
        if not isinstance(size, (int, float)):
            return None
        touched = min(size, mte2_bytes)
        if not touched >= 1:
            return None
        ranges = ((decl_base, decl_base + int(touched)),)
    meta_ver = meta.get("weight_tensor_version", meta.get("tensor_version", 0))
    version = decl_ver if decl_ver else (meta_ver if type(meta_ver) is int and meta_ver >= 0 else 0)
    return TensorRangeObligation(die, identity, version, tuple(ranges), "hbm_cold", "owner")


def _pure_copy_boundary(op, meta: Mapping[str, object], die: int, replica: int, stage: int,
                        consumed: set[tuple[int, str]]):
    if meta.get("initial_residency") == "l2_warm":
        return None, None
    hbm = int(getattr(op, "hbm_bytes", 0) or 0)
    mte2 = int(getattr(op, "mte2_bytes", 0) or 0)
    mte3 = int(getattr(op, "mte3_bytes", 0) or 0)
    if not getattr(op, "deps", ()):
        read_sz = min(mte2, hbm) if (mte2 > 0 and hbm > 0) else (mte2 or hbm)
        if read_sz > 0:
            return TensorRangeObligation(
                die, f"boundary_in:r{replica}.p{stage}:{op.op_id}", 0, ((0, read_sz),), "hbm_cold", "owner",
            ), None
    if (replica, op.op_id) not in consumed:
        write_sz = min(mte3, hbm) if (mte3 > 0 and hbm > 0) else (mte3 or hbm)
        if write_sz > 0:
            return None, TensorRangeObligation(
                die, f"boundary_out:r{replica}.p{stage}:{op.op_id}", 0, ((0, write_sz),), "onchip_resident", "device",
            )
    return None, None


def derive_default_obligations(scenario, demands, aliases, covered, rank_ops_fn):
    """Derive per-(replica, stage, die, op) default obligations and die placements."""
    atomic = getattr(scenario, "atomic_step", None)
    case = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    dies = tuple(demands.devices)
    parallel = getattr(getattr(case, "runtime", None), "parallel", None)
    tp_width = (int(parallel.tp) * int(parallel.pcp)) if parallel is not None else 1
    pp_stages = int(parallel.pp) if parallel is not None else 1
    default_dt = str(getattr(getattr(case, "model", None), "compute_dtype", "") or "bf16")

    rank_ops = tuple(rank_ops_fn(scenario))
    consumed = {(replica, dep) for replica, op in rank_ops for dep in getattr(op, "deps", ())}
    multi_rank = len(demands.devices) > 1

    matrix_work: dict[tuple[int, int, int, str], int] = defaultdict(int)
    vector_work: dict[tuple[int, int, int], int] = defaultdict(int)
    link_work: dict[tuple[int, int, int], int] = defaultdict(int)
    read_obs: list[TensorRangeObligation] = []
    write_obs: list[TensorRangeObligation] = []
    placements: dict[int, tuple[int, int]] = {}
    seen: set[tuple[int, int, str]] = set()

    for replica, op in rank_ops:
        if op.op_id in aliases:
            continue
        meta = dict(op.metadata)
        stage = int(getattr(op, "stage", 0))
        die_idx = (replica * pp_stages + stage) * tp_width
        die = dies[die_idx] if 0 <= die_idx < len(dies) else (dies[0] if dies else 0)
        placements.setdefault(die, (replica, stage))
        key = (replica, stage, op.op_id)
        has_explicit_work = bool(meta.get("semantic_work"))
        has_matrix = False
        has_vector = False
        has_link = False
        if not has_explicit_work and key not in seen and key not in covered:
            seen.add(key)
            dtype = getattr(op, "dtype", "") or default_dt
            pairs = getattr(op, "matrix_flops", ()) or ((dtype, getattr(op, "flops", 0)),)
            for dt, flops in pairs:
                if flops > 0:
                    matrix_work[replica, stage, die, str(dt)] += int(flops)
                    has_matrix = True
            vec_ops = int(getattr(op, "vector_ops", 0) or 0)
            if vec_ops > 0 and not meta.get("inline_epilogue_absorbable"):
                vector_work[replica, stage, die] += vec_ops
                has_vector = True
            link_bytes = int(getattr(op, "link_bytes", 0) or 0)
            if link_bytes > 0 and not meta.get("mandatory_comm_wire_bytes_by_link") and (
                multi_rank or bool(demands.links)
            ):
                if demands.engine_rate((replica, stage, die), "link_egress", "") > 0:
                    link_work[replica, stage, die] += link_bytes
                    has_link = True

        has_explicit_mem = "read_obligations" in meta or "write_obligations" in meta
        if "read_obligations" in meta:
            read_obs.extend(meta["read_obligations"])
        elif (read := _cold_weight_read(op, meta, die)) is not None:
            read_obs.append(read)
            has_explicit_mem = True
        write_obs.extend(meta.get("write_obligations", ()))

        if not (has_explicit_work or has_explicit_mem or has_matrix or has_vector or has_link) and key not in covered:
            b_read, b_write = _pure_copy_boundary(op, meta, die, replica, stage, consumed)
            if b_read is not None:
                read_obs.append(b_read)
            if b_write is not None:
                write_obs.append(b_write)

    for (replica, stage, die), byte_count in link_work.items():
        demands.charge((replica, stage, die), "link_egress", byte_count)

    work_items: list[SemanticWorkItem] = []
    for (replica, stage, die, dt), flops in matrix_work.items():
        path = demands.paths.get(die)
        engines = ("cube", "aiv_simt") if path and path.simt_ub_carveout_bytes else ("cube",)
        work_items.append(SemanticWorkItem(
            f"mac:r{replica}.p{stage}.d{die}:{dt}", "matrix_mac", dt, flops, engines,
            die=die, replica=replica, pp_stage=stage,
        ))
    for (replica, stage, die), ops_count in vector_work.items():
        path = demands.paths.get(die)
        engines = ("aiv_simd", "aiv_simt") if path and path.simt_ub_carveout_bytes else ("aiv_simd",)
        work_items.append(SemanticWorkItem(
            f"vec:r{replica}.p{stage}.d{die}", "vector_alu", "fp32", ops_count, engines,
            die=die, replica=replica, pp_stage=stage,
        ))

    if not (work_items or read_obs or write_obs):
        return (), placements
    return (FusedRegionSummary(
        tier="K1", region_id="scenario_compulsory_floor", op_ids=(), semantic_work=tuple(work_items),
        issued_cube_flops_by_dtype=(), vector_ops_by_pipe=(),
        read_obligations=tuple(read_obs), write_obligations=tuple(write_obs),
        logical_transfer_bytes=0, issued_copy_bytes=0, dram_sector_bytes=0, turnaround_service_bytes=Fraction(),
        realized_l2_hit_bytes=0, comm_wire_bytes_by_link=(), stars_launch_count=0,
        aicpu_metadata_ms=Fraction(), sram_peaks_bytes=(), spec=None,
    ),), placements


__all__ = (
    "_COLD_WEIGHT_FAMILIES", "_WEIGHT_ID_KEYS", "_cold_weight_read",
    "_weight_allocation_id", "derive_default_obligations",
)
