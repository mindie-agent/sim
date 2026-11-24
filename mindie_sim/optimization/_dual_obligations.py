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

from mindie_sim.prediction.operators.matrix.matrix_precision import (
    PACKED_FP4_FORMATS,
    resolve_matrix_precision,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
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
    """Return (read_hbm_ob, write_hbm_ob, mte2_copy_bytes, write_copy_bytes) for an unfused copy."""
    del op, meta, die, replica, stage, consumed
    raise NotImplementedError(
        "LEGACY_PURGED: separate read (mte2>0) and terminal-store (mte3>0) boundary obligations"
    )


def _resolve_op_precision(op, stage_devices, policy: str):
    """Align matrix compute dtype/matrix_flops with hardware-bound Engine execution."""
    if not stage_devices or not isinstance(op, OpSpec):
        return op
    return resolve_matrix_precision(op, devices=stage_devices, policy=policy, recipe=None)


def derive_default_obligations(scenario, demands, aliases, covered, rank_ops_fn):
    """Derive per-(replica, stage, die, op) default obligations and die placements."""
    atomic = getattr(scenario, "atomic_step", None)
    case = getattr(atomic, "case", None) or getattr(scenario, "case", None)
    dies = tuple(demands.devices)
    parallel = getattr(getattr(case, "runtime", None), "parallel", None)
    tp_width = (int(parallel.tp) * int(parallel.pcp)) if parallel is not None else 1
    pp_stages = int(parallel.pp) if parallel is not None else 1
    default_dt = str(getattr(getattr(case, "model", None), "compute_dtype", "") or "").strip()
    matrix_policy = str(getattr(getattr(case, "runtime", None), "matrix_compute_policy", "auto") or "auto")

    rank_ops = tuple(rank_ops_fn(scenario))
    consumed = {(replica, dep) for replica, op in rank_ops for dep in getattr(op, "deps", ())}
    multi_rank = len(demands.devices) > 1

    matrix_work: dict[tuple[int, int, int, str], int] = defaultdict(int)
    vector_work: dict[tuple[int, int, int], int] = defaultdict(int)
    link_work: dict[tuple[int, int, int], int] = defaultdict(int)
    mte2_copy_work: dict[tuple[int, int, int], int] = defaultdict(int)
    write_copy_work: dict[tuple[int, int, int], int] = defaultdict(int)
    read_obs: list[TensorRangeObligation] = []
    write_obs: list[TensorRangeObligation] = []
    placements: dict[int, tuple[int, int]] = {}
    stage_devices_cache: dict[tuple[int, int], tuple] = {}
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
            resolved_op = op
            weight_fmt = str(meta.get("weight_format") or meta.get("weight_dtype") or "").strip().lower()
            if weight_fmt in PACKED_FP4_FORMATS and isinstance(op, OpSpec):
                sd_key = (replica, stage)
                stage_devices = stage_devices_cache.get(sd_key)
                if stage_devices is None:
                    stage_ranks = (
                        dies[die_idx:die_idx + tp_width]
                        if 0 <= die_idx < len(dies)
                        else ((die,) if die in demands.devices else ())
                    )
                    stage_devices = tuple(demands.devices[r] for r in stage_ranks if r in demands.devices)
                    stage_devices_cache[sd_key] = stage_devices
                resolved_op = _resolve_op_precision(op, stage_devices, matrix_policy)
            dtype = str(getattr(resolved_op, "dtype", "") or default_dt).strip()
            pairs = getattr(resolved_op, "matrix_flops", ()) or ((dtype, getattr(resolved_op, "flops", 0)),)
            for dt, flops in pairs:
                if flops > 0:
                    if not str(dt).strip():
                        raise ModelingError(f"dual_obligations_dtype_missing:{op.op_id}")
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
            b_read, b_write, b_mte2, b_mte3 = _pure_copy_boundary(op, meta, die, replica, stage, consumed)
            if b_read is not None:
                read_obs.append(b_read)
            if b_write is not None:
                write_obs.append(b_write)
            if b_mte2 > 0:
                mte2_copy_work[replica, stage, die] += b_mte2
            if b_mte3 > 0:
                write_copy_work[replica, stage, die] += b_mte3

    for (replica, stage, die), byte_count in link_work.items():
        demands.charge((replica, stage, die), "link_egress", byte_count)
    for (replica, stage, die), byte_count in mte2_copy_work.items():
        demands.charge((replica, stage, die), "mte2_total", byte_count)
    for (replica, stage, die), byte_count in write_copy_work.items():
        demands.charge((replica, stage, die), "write_total", byte_count)

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
