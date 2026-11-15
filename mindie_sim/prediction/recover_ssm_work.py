"""Family-agnostic RecoverSSM trajectory-record and single-commit lowering.

[Role] Share compact rank-1 trajectory-record and post-verify single-commit work
  across all associative linear-attention families (KDA, GDN, Mamba, GLA,
  DeltaNet, Lightning, RetNet).
[Invariants] Every rank-1 recurrence ``S_t = decay_t * S_{t-1} + k_t @ c_t^T``
  reduces speculative Verify state writes from ``q * O(Dk * Dv)`` full state
  checkpoints to ``q * O(Dk + Dv)`` compact trajectory records plus one final
  ``O(Dk * Dv)`` committed state write. Non-speculative decode (``q == 1``) and
  disabled ``recover_ssm`` preserve standard per-query checkpoint semantics.
[Timeouts & Locks] Pure O(sequences * owner cycles) arithmetic; no I/O or locks.
[Side-Effect Policy] Constructs immutable operators and physical plans only.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from mindie_sim.schema.common import ModelingError, dtype_bytes

ZERO = 0
ONE = 1
ONE_FLOAT = 1.0
FOUR = 4

GDN_ARCH35_RECOVER_SSM_RECIPE = "vllm_ascend_recurrent_gated_delta_rule_arch35_fp32_recover_ssm"


def recover_ssm_enabled(runtime: Any) -> bool:
    if runtime is None:
        return False
    if bool(getattr(runtime, "recover_ssm", False)):
        return True
    return bool(dict(getattr(runtime, "metadata", ()) or ()).get("recover_ssm", False))


def record_head_bytes(dk: int, dv: int, kg_dtype: str = "bf16") -> int:
    """Per-token per-head compact trajectory record (FP32 correction + k/g)."""
    return int(dv) * 4 + 2 * int(dk) * int(dtype_bytes(kg_dtype))


def record_slice_bytes(dk: int, slice_width: int, part_index: int, kg_dtype: str = "bf16") -> int:
    """Value-tile slice of the trajectory record; k/g is emitted once at tile 0."""
    kg = 2 * int(dk) * int(dtype_bytes(kg_dtype)) if int(part_index) == 0 else 0
    return int(slice_width) * 4 + kg


def commit_tokens_for_view(view: Any, runtime: Any = None) -> int:
    """Integer commitment only; a configured mean is never rounded here."""
    q = max(ONE, int(getattr(view, "q_len", ONE)))
    advanced = int(getattr(view, "draft_advanced_tokens", ZERO) or ZERO)
    if advanced > ZERO:
        return min(q, advanced)
    accept = float(getattr(runtime, "mtp_acceptance_length", ONE) or ONE)
    if q > ONE and accept > ONE_FLOAT:
        raise ModelingError(
            "recover_ssm_commit_requires_resolved_integer_advancement:"
            + str(getattr(view, "request_id", "")))
    return q


def emit_commit_op(b: Any, *, name: str, deps: Sequence[str], views: Sequence[Any],
                   heads: int, dk: int, dv: int, state_dtype: str, kg_dtype: str = "bf16",
                   runtime: Any = None, attention_kind: str = "kda") -> str | None:
    spec_views = tuple(v for v in views if getattr(v, "phase", "") == "decode" and int(v.q_len) > 1)
    if not spec_views:
        return None
    state_bytes = int(heads) * int(dk) * int(dv) * int(dtype_bytes(state_dtype or "fp32"))
    rec_bytes = int(heads) * record_head_bytes(dk, dv, kg_dtype)
    committed = tuple(commit_tokens_for_view(v, runtime) for v in spec_views)
    total_committed, seq_count = sum(committed), len(spec_views)
    return b.work(
        name, tuple(deps), (total_committed, heads, dk, dv),
        vector=FOUR * total_committed * heads * dk * dv,
        read=seq_count * state_bytes + total_committed * rec_bytes,
        write=seq_count * state_bytes, family="elementwise",
        metadata=(
            ("recover_ssm_stage", "commit"), ("recover_ssm", True),
            ("recover_ssm_pending_acceptance", True),
            ("recurrent_attention_kind", str(attention_kind)),
            ("sequences", seq_count), ("committed_tokens", committed),
            ("state_checkpoints", seq_count), ("record_bytes_per_token", rec_bytes),
            ("state_bytes", state_bytes), ("heads", int(heads)),
            ("dk", int(dk)), ("dv", int(dv)), ("state_dtype", state_dtype),
        ),
    )


def rewrite_gdn_analytical_scan(scan: Any, *, active_views: Sequence[Any], heads: int,
                                dk: int, dv: int, state_dtype: str = "fp32",
                                kg_dtype: str = "bf16") -> Any:
    spec_tokens = sum(int(v.q_len) for v in active_views if int(v.q_len) > 1)
    if spec_tokens <= 0:
        return scan
    state_bytes = int(heads) * int(dk) * int(dv) * int(dtype_bytes(state_dtype))
    rec_bytes = int(heads) * record_head_bytes(dk, dv, kg_dtype)
    saved_writes = spec_tokens * (state_bytes - rec_bytes)
    meta = {
        **dict(scan.metadata), "recover_ssm": True,
        "recover_ssm_stage": "verify_compact_record",
        "record_bytes_per_token": rec_bytes,
        "state_checkpoints": sum(1 for v in active_views if int(v.q_len) == 1),
    }
    return replace(scan, mte3_bytes=max(0.0, float(scan.mte3_bytes) - saved_writes),
                   hbm_bytes=max(0.0, float(scan.hbm_bytes) - saved_writes),
                   metadata=tuple(meta.items()))


def build_gdn_arch35_recover_ssm(op: Any, geometry: Any, *, compact: bool = True) -> OperatorPlan:
    """GDN arch35_fp32 decode plan writing compact records for speculative q > 1."""
    from . import gdn_recurrent

    base = gdn_recurrent.build(op, geometry, compact=compact)
    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in meta["sequence_tokens"])
    if not any(q > 1 for q in lengths):
        return base
    heads, dk, dv = (int(meta[k]) for k in ("heads", "dk", "dv"))
    ak, av = math.ceil(dk / 16) * 16, math.ceil(dv / 16) * 16
    value_step = (255 if (8192 // ak) > 256 else (8192 // ak)) // 16 * 16
    parts = math.ceil(dv / value_step)
    units = geometry.resource_units.get("vector", geometry.compute_units)
    blocks = min(units, len(lengths) * heads)
    ownership = gdn_recurrent.head_ownership(lengths, heads, blocks)
    run_counts = tuple(sum(1 for _ in gdn_recurrent.head_runs(lengths, segs, compact=compact))
                       for segs in ownership)
    read = read_role(geometry.resource_units, aiv=0)
    total = sum(lengths) * heads
    tensor_reads = len(lengths) * heads * dk * dv * 4 + total * (2 * dk + dv) * 2
    scalar_payload = total * (4 + 2 * parts)
    rec_head = record_head_bytes(dk, dv, op.dtype)
    writes = sum(
        q * heads * ((rec_head if q > 1 else dk * dv * 4) + dv * 2)
        for q in lengths
    )
    vector = len(lengths) * heads * ak * dv + total * (8 * ak * dv + dv + 6 * ak + av)
    work = Work(resources={"hbm": tensor_reads + scalar_payload + writes,
                           read: tensor_reads, "mte3": writes, "vector": vector})

    def program(context):
        def owner_program(owner, segments):
            owned = context.owned(owner, units, ("vector", read, "mte3"))

            def head(q, repeat):
                v = lambda amt: context.bind(Work(resources={"vector": amt}).scaled(repeat))
                mem = lambda amt, role: context.bind(Work(resources={"hbm": amt, role: amt}).scaled(repeat))

                def part(part_index):
                    n = min(value_step, dv - part_index * value_step)
                    out_state = record_slice_bytes(dk, n, part_index, op.dtype) if q > 1 else dk * n * 4

                    def token(t):
                        first = part_index == 0 and t == 0
                        scalar = 2 + (4 if part_index == 0 else 0)
                        stages = [Stage("scalar_inputs", context.bind(Work(resources={"hbm": scalar}).scaled(repeat)),
                                        previous=("attention_cast",))]
                        prev = "scalar_inputs"
                        if part_index == 0:
                            stages.append(Stage("gamma", v(2 * ak), after=(prev,), placement=owned))
                            prev = "gamma"
                        stages.append(Stage("decay", v(ak * n), after=(prev,), placement=owned))
                        prev = "decay"
                        if first:
                            stages.extend((
                                Stage("qk_load", mem(q * 2 * dk * 2, read), after=(prev,), placement=owned),
                                Stage("qk_cast_scale", v(q * 3 * ak), after=("qk_load",), placement=owned),
                            ))
                            prev = "qk_cast_scale"
                        stages.append(Stage("key_dot", v((2 * ak - 1) * n), after=(prev,), placement=owned))
                        prev = "key_dot"
                        if first:
                            stages.extend((
                                Stage("v_load", mem(q * dv * 2, read), after=(prev,), placement=owned),
                                Stage("v_cast", v(q * av), after=("v_load",), placement=owned),
                            ))
                            prev = "v_cast"
                        stages.extend((
                            Stage("delta", v(2 * n), after=(prev,), placement=owned),
                            Stage("rank_one_update", v(2 * ak * n), after=("delta",), placement=owned),
                            Stage("state_copy", v(ak * n), after=("rank_one_update",), previous=("state_write",), placement=owned),
                            Stage("state_write", mem(out_state, "mte3"), after=("state_copy",), previous=("attention_write",), placement=owned),
                            Stage("query_dot", v((2 * ak - 1) * n), after=("state_copy",), placement=owned),
                            Stage("attention_cast", v(n), after=("query_dot",), previous=("attention_write",), placement=owned),
                            Stage("attention_write", mem(n * 2, "mte3"), after=("attention_cast", "state_write"), placement=owned),
                        ))
                        return tuple(stages)

                    return (Stage("state_load", mem(dk * n * 4, read), placement=owned),
                            Stage("state_resident_copy", v(ak * n), after=("state_load",), placement=owned),
                            Stage("tokens", after=("state_resident_copy",), program=PipelineProgram(q, 2, token)))

                return (Stage("gamma_zero", v(q * ak), placement=owned),
                        Stage("value_tiles", after=("gamma_zero",), program=PipelineProgram(parts, 1, part)))

            pending = tuple(gdn_recurrent.head_runs(lengths, segments, compact=compact))
            heads_prog = PipelineProgram(run_counts[owner], 1, lambda idx, pending=pending: head(*pending[idx]))
            pool = context.pool_names.get("ub")
            if pool is None:
                return heads_prog
            alloc = (Allocation("recurrent_buffers", pool, gdn_recurrent.UB_BYTES,
                                unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=alloc),
                                  Stage("heads", after=("buffers",), program=heads_prog),
                                  Stage("release", after=("heads",), free=alloc)))

        return ParallelProgram(tuple(owner_program(i, segs) for i, segs in enumerate(ownership)))

    meta_out = dict(base.metadata)
    sched = dict(meta_out.get("gdn_recurrent_schedule", {}))
    sched["state_residency"] = "one_initial_load_per_head;recover_ssm_compact_record_then_single_commit"
    meta_out["gdn_recurrent_schedule"] = sched
    meta_out["recover_ssm"] = True
    return OperatorPlan(work, program, metadata=meta_out)
