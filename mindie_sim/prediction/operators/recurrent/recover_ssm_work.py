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
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from ..contracts import OperatorPlanningContext

GDN_ARCH35_RECOVER_SSM_RECIPE = "vllm_ascend_recurrent_gated_delta_rule_arch35_fp32_recover_ssm"
COMMIT_INTENT_STAGE = "commit_intent"
FP32_RECORD_BYTES = 4


def recover_ssm_enabled(runtime: Any) -> bool:
    if runtime is None:
        return False
    if bool(getattr(runtime, "recover_ssm", False)):
        return True
    return bool(dict(getattr(runtime, "metadata", ()) or ()).get("recover_ssm", False))


def record_head_bytes(dk: int, dv: int, kg_dtype: str = "bf16") -> int:
    """Per-token per-head compact trajectory record (FP32 correction + k/g)."""
    return int(dv) * FP32_RECORD_BYTES + 2 * int(dk) * int(dtype_bytes(kg_dtype))


def gdn_record_head_bytes(dk: int, dv: int) -> int:
    """Qwen GDN compact record: ``(Dk + Dv + 1)`` FP32 elements per token/head."""
    return (int(dk) + int(dv) + 1) * FP32_RECORD_BYTES


def gdn_record_slice_bytes(dk: int, slice_width: int, part_index: int) -> int:
    """Value-tile slice of the GDN record; ``Dk + 1`` is emitted at tile 0."""
    return (int(slice_width) * FP32_RECORD_BYTES
            + ((int(dk) + 1) * FP32_RECORD_BYTES if int(part_index) == 0 else 0))


def record_slice_bytes(dk: int, slice_width: int, part_index: int, kg_dtype: str = "bf16") -> int:
    """Value-tile slice of the trajectory record; k/g is emitted once at tile 0."""
    kg = 2 * int(dk) * int(dtype_bytes(kg_dtype)) if int(part_index) == 0 else 0
    return int(slice_width) * FP32_RECORD_BYTES + kg


def commit_tokens_for_view(view: Any, runtime: Any = None) -> int:
    """Integer commitment only; never averages, silently commits all q, or clamps overlarge values."""
    del runtime
    q = max(1, int(getattr(view, "q_len", 1)))
    raw_advanced = getattr(view, "draft_advanced_tokens", 0)
    if isinstance(raw_advanced, bool) or not isinstance(raw_advanced, int):
        raise ModelingError(
            "recover_ssm_commit_requires_resolved_integer_advancement:"
            + str(getattr(view, "request_id", "")))
    if raw_advanced < 0 or raw_advanced > q:
        raise ModelingError(
            f"recover_ssm_commit_advancement_out_of_range:{getattr(view, 'request_id', '')}:{raw_advanced}:{q}")
    if raw_advanced > 0:
        return int(raw_advanced)
    if bool(getattr(view, "draft_acceptance_resolved", False)):
        return 0
    if q > 1:
        raise ModelingError(
            "recover_ssm_commit_requires_resolved_integer_advancement:"
            + str(getattr(view, "request_id", "")))
    return 1


def _runtime_wy_cube_verify(runtime: Any, b: Any = None) -> bool:
    if runtime is None:
        return False
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    if not bool(rt_meta.get("wy_cube_speculative_verify", False)):
        return False
    machine = getattr(b, "machine", None) if b is not None else None
    dies = getattr(machine, "dies", ()) if machine is not None else ()
    if dies:
        return any(int(getattr(d, "compute_units", 0) or 0) > 0 for d in dies)
    return True


def emit_commit_op(b: Any, *, name: str, deps: Sequence[str], views: Sequence[Any],
                   heads: int, dk: int, dv: int, state_dtype: str, kg_dtype: str = "bf16",
                   runtime: Any = None, attention_kind: str = "kda") -> str | None:
    """Emit an uncharged deferred commit intent for speculative decode views.

    Integer advancement is accepted only after sampling/acceptance; resolving
    it here would make a pure body assembly depend on a service event.  The
    intent therefore carries geometry and remains explicitly pending, and
    ``recover_ssm_commit`` materializes the physical work at the acceptance
    boundary.
    """
    spec_views = tuple(
        view for view in views
        if getattr(view, "phase", "") == "decode"
        and int(getattr(view, "q_len", 0) or 0) > 1
        and not bool(getattr(view, "padding", False)))
    if not spec_views:
        return None
    wy_verify = _runtime_wy_cube_verify(runtime, b)
    plan = getattr(b, "plan", None)
    gdn = str(attention_kind) == "gdn"
    record_head = (gdn_record_head_bytes(dk, dv) if gdn
                   else record_head_bytes(dk, dv, kg_dtype))
    layer_idx = int(getattr(plan, "index", 0))
    owner = f"layer{layer_idx}"
    state_versions = tuple(
        (str(view.request_id), int(getattr(view, "state_version", 0) or 0))
        for view in spec_views
    )
    state_slots = tuple(
        (str(view.request_id), getattr(view, "state_slot", None))
        for view in spec_views
    )
    state_generations = tuple(
        (str(view.request_id), int(getattr(view, "state_generation", 0) or 0))
        for view in spec_views
    )
    prev_publications = tuple(
        (str(view.request_id), str(getattr(view, "previous_state_publication", "") or ""))
        for view in spec_views
        if str(getattr(view, "previous_state_publication", "") or "")
    )
    distinct_vers = sorted({ver for _, ver in state_versions})
    base_version_num = distinct_vers[0] if len(distinct_vers) == 1 else max(distinct_vers, default=0)
    per_req_inputs = tuple((req_id, f"{owner}:v{ver}") for req_id, ver in state_versions)
    input_state_ver = (
        f"{owner}:v{distinct_vers[0]}"
        if len(distinct_vers) <= 1
        else f"{owner}:per_request[" + ",".join(f"{r}:v{v}" for r, v in state_versions) + "]"
    )
    pub_deps = tuple(dict.fromkeys(f"state_input.{pub}" for _, pub in prev_publications if pub))
    verify_dep_set = set(deps)
    if verify_dep_set and hasattr(b, "ops"):
        state_stamp = (
            ("state_owner", owner),
            ("input_state_version", input_state_ver),
            ("request_input_state_versions", per_req_inputs),
            ("request_state_versions", state_versions),
            ("request_state_slots", state_slots),
            ("request_state_generations", state_generations),
            ("previous_state_publications", prev_publications),
        )
        for idx, existing in enumerate(b.ops):
            if existing.op_id in verify_dep_set:
                merged = dict(existing.metadata)
                for k_item, v_item in state_stamp:
                    merged.setdefault(k_item, v_item)
                if wy_verify and gdn:
                    merged["wy_cube_verify"] = True
                    merged["invocation_class"] = "mixed_aic_aiv"
                    merged.setdefault("heads", int(heads))
                    merged.setdefault("key_heads", int(merged.get("hk", heads)))
                    merged.setdefault("dk", int(dk))
                    merged.setdefault("dv", int(dv))
                    merged.setdefault("state_dtype", str(state_dtype))
                    merged.setdefault(
                        "sequence_tokens",
                        tuple(int(getattr(v, "q_len", 1)) for v in views if not getattr(v, "padding", False)),
                    )
                    merged["recover_ssm"] = True
                new_deps = tuple(dict.fromkeys((*existing.deps, *pub_deps)))
                updated_op = replace(existing, deps=new_deps, metadata=tuple(merged.items()))
                if wy_verify and gdn:
                    if updated_op.recipe == "decode_recurrent_scan":
                        updated_op = replace(
                            updated_op,
                            recipe=GDN_ARCH35_RECOVER_SSM_RECIPE,
                            source_boundary="kernel",
                        )
                    if updated_op.recipe in {GDN_ARCH35_RECOVER_SSM_RECIPE, "gdn_recurrent"}:
                        updated_op = stamp_gdn_kernel_spec(updated_op, compact_records=True)
                b.ops[idx] = updated_op
    geometry = {
        "heads": int(heads), "dk": int(dk), "dv": int(dv),
        "state_dtype": str(state_dtype), "kg_dtype": str(kg_dtype),
        "record_bytes_per_token": int(heads) * record_head,
        "scope": str(getattr(plan, "scope", "target")),
        "layer": layer_idx,
        "attention_kind": str(attention_kind),
    }
    request_ids = tuple(str(view.request_id) for view in spec_views)
    op_id = b.work(
        name, tuple(dict.fromkeys((*deps, *pub_deps))), (len(spec_views), int(heads), int(dk), int(dv)),
        flops=0, vector=0, read=0, write=0, family="elementwise",
        metadata=(
            ("recover_ssm", True), ("recover_ssm_intent", True),
            ("recover_ssm_stage", COMMIT_INTENT_STAGE),
            ("recover_ssm_pending_acceptance", True),
            ("recover_ssm_geometry", geometry),
            ("state_owner", owner),
            ("input_state_version", input_state_ver),
            ("request_input_state_versions", per_req_inputs),
            ("base_version", base_version_num),
            ("request_state_versions", state_versions),
            ("request_state_slots", state_slots),
            ("request_state_generations", state_generations),
            ("previous_state_publications", prev_publications),
            ("record_bytes_per_token", int(heads) * record_head),
            ("recurrent_attention_kind", str(attention_kind)),
            ("requests", request_ids),
        ),
    )
    b.ops[-1] = replace(b.ops[-1], requests=request_ids)
    return op_id


def rewrite_gdn_analytical_scan(scan: Any, *, active_views: Sequence[Any], heads: int,
                                dk: int, dv: int, state_dtype: str = "fp32",
                                kg_dtype: str = "bf16",
                                attention_kind: str = "kda") -> Any:
    spec_tokens = sum(int(v.q_len) for v in active_views if int(v.q_len) > 1)
    if spec_tokens <= 0:
        return scan
    state_bytes = int(heads) * int(dk) * int(dv) * int(dtype_bytes(state_dtype))
    rec_head = (gdn_record_head_bytes(dk, dv) if attention_kind == "gdn"
                else record_head_bytes(dk, dv, kg_dtype))
    rec_bytes = int(heads) * rec_head
    saved_writes = spec_tokens * (state_bytes - rec_bytes)
    meta = {
        **dict(scan.metadata), "recover_ssm": True,
        "recover_ssm_stage": "verify_compact_record",
        "recurrent_attention_kind": str(attention_kind),
        "record_kg_dtype": str(kg_dtype),
        "record_bytes_per_token": rec_bytes,
        "state_checkpoints": sum(1 for v in active_views if int(v.q_len) == 1),
    }
    return replace(scan, mte3_bytes=max(0.0, float(scan.mte3_bytes) - saved_writes),
                   hbm_bytes=max(0.0, float(scan.hbm_bytes) - saved_writes),
                   metadata=tuple(meta.items()))


def _is_gdn_record(op: Any) -> bool:
    return str(dict(op.metadata).get("recurrent_attention_kind", "")) == "gdn" or (
        "gdn" in str(op.recipe))


def stamp_gdn_kernel_spec(scan: Any, *, compact_records: bool) -> Any:
    """Declare the source kernel's exact closed-form work on the lowered ``OpSpec``.

    The analytical decode scan declares ``7 * Dk * Dv`` vector work and one
    full FP32 checkpoint per token. Once the arch35 kernel (and optionally the
    RecoverSSM compact-record or AIC+AIV WY verify variant) is selected, the
    executed program issues ``kernel_work_terms`` / ``wy_kernel_work_terms``;
    the spec must carry the same work so ``L*`` never exceeds executed duration.
    """
    from mindie_sim.prediction.operators.linear_attention import gdn_recurrent

    meta = dict(scan.metadata)
    lengths = tuple(int(q) for q in meta["sequence_tokens"])
    if not any(lengths):
        return scan
    heads, dk, dv = (int(meta[k]) for k in ("heads", "dk", "dv"))
    record = None
    if compact_records:
        record = gdn_record_head_bytes(dk, dv) if _is_gdn_record(scan) else record_head_bytes(dk, dv, scan.dtype)
    wy_verify = bool(meta.get("wy_cube_verify", False) or meta.get("wy_cube_speculative_verify", False))
    if wy_verify and any(q >= gdn_recurrent.MIN_WY_SPECULATIVE_TOKENS for q in lengths):
        reads, scalar, writes, vector, cube_flops = gdn_recurrent.wy_kernel_work_terms(
            lengths, heads, dk, dv, input_bytes=int(dtype_bytes(scan.dtype)),
            state_bytes=int(dtype_bytes(meta["state_dtype"])), record_head=record,
        )
        stamped = Work(
            matrix={"bf16": cube_flops} if cube_flops else {},
            resources={
                "hbm": reads + scalar + writes,
                "mte2_aiv": reads,
                "mte3": writes,
                "vector": vector,
            },
        ).apply(scan)
        return replace(stamped, mte2_bytes=float(reads), mte3_bytes=float(writes))
    reads, scalar, writes, vector = gdn_recurrent.kernel_work_terms(
        lengths, heads, dk, dv, input_bytes=int(dtype_bytes(scan.dtype)),
        state_bytes=int(dtype_bytes(meta["state_dtype"])), record_head=record)
    return replace(scan, flops=0.0, matrix_flops=(), hbm_bytes=float(reads + scalar + writes),
                   mte2_bytes=float(reads), mte3_bytes=float(writes), vector_ops=float(vector))


def build_gdn_arch35_recover_ssm(
    op: Any, geometry: Any, *, compact: bool = True, ctx: OperatorPlanningContext | None = None,
) -> OperatorPlan:
    """GDN arch35_fp32 decode plan writing compact records for speculative q > 1."""
    from mindie_sim.prediction.operators.linear_attention import gdn_recurrent

    meta = dict(op.metadata)
    lengths = tuple(int(v) for v in gdn_recurrent.recurrent_lengths(op, ctx))
    if gdn_recurrent.wy_cube_verify_active(meta, geometry, ctx, lengths):
        merged_meta = {**meta, "recover_ssm": True}
        wy_op = replace(op, metadata=tuple(merged_meta.items()))
        wy_plan = gdn_recurrent.build(wy_op, geometry, compact=compact, ctx=ctx)
        meta_wy = dict(wy_plan.metadata)
        sched_wy = dict(meta_wy.get("gdn_recurrent_schedule", {}))
        sched_wy["state_residency"] = "one_initial_load_per_head;wy_cube_verify_compact_record"
        meta_wy["gdn_recurrent_schedule"] = sched_wy
        meta_wy["recover_ssm"] = True
        meta_wy["wy_cube_verify"] = True
        return OperatorPlan(wy_plan.work, wy_plan.program, metadata=meta_wy)
    base = gdn_recurrent.build(op, geometry, compact=compact, ctx=ctx)
    if not any(q > 1 for q in lengths):
        return base
    heads, dk, dv = (int(meta[k]) for k in ("heads", "dk", "dv"))
    alignment = gdn_recurrent.VECTOR_LANE_ALIGNMENT
    ak, av = align_up(dk, alignment), align_up(dv, alignment)
    value_step = gdn_recurrent.state_value_step(ak)
    parts = math.ceil(dv / value_step)
    units, core_ids = gdn_recurrent.recurrent_owners(geometry, ctx)
    active_heads = sum(length > 0 for length in lengths) * heads
    blocks = min(len(core_ids), active_heads)
    ownership = gdn_recurrent.head_ownership(lengths, heads, blocks)
    run_counts = tuple(sum(1 for _ in gdn_recurrent.head_runs(lengths, segs, compact=compact))
                       for segs in ownership)
    read = read_role(geometry.resource_units, aiv=0)
    input_bytes = int(dtype_bytes(op.dtype))
    state_bytes = int(dtype_bytes(meta["state_dtype"]))
    gdn = _is_gdn_record(op)
    rec_head = (gdn_record_head_bytes(dk, dv) if gdn
                else record_head_bytes(dk, dv, op.dtype))
    slice_bytes = (
        (lambda n, part: gdn_record_slice_bytes(dk, n, part)) if gdn else
        (lambda n, part: record_slice_bytes(dk, n, part, op.dtype)))
    tensor_reads, scalar_payload, writes, vector = gdn_recurrent.kernel_work_terms(
        lengths, heads, dk, dv, input_bytes=input_bytes, state_bytes=state_bytes,
        record_head=rec_head)
    work = Work(resources={"hbm": tensor_reads + scalar_payload + writes,
                           read: tensor_reads, "mte3": writes, "vector": vector})

    def program(context):
        def owner_program(owner, segments, run_count):
            owned = context.owned(owner, units, ("vector", read, "mte3"))

            def head(q, repeat):
                v = lambda amt: context.bind(Work(resources={"vector": amt}).scaled(repeat))
                mem = lambda amt, role: context.bind(Work(resources={"hbm": amt, role: amt}).scaled(repeat))

                def part(part_index):
                    n = min(value_step, dv - part_index * value_step)
                    out_state = (slice_bytes(n, part_index) if q > 1
                                 else dk * n * state_bytes)

                    def token(t):
                        first = part_index == 0 and t == 0
                        scalar = input_bytes + (state_bytes if part_index == 0 else 0)
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
                                Stage("qk_load", mem(q * 2 * dk * input_bytes, read), after=(prev,), placement=owned),
                                Stage("qk_cast_scale", v(q * gdn_recurrent.QK_CAST_SCALE_OPS_PER_LANE * ak), after=("qk_load",), placement=owned),
                            ))
                            prev = "qk_cast_scale"
                        stages.append(Stage("key_dot", v((2 * ak - 1) * n), after=(prev,), placement=owned))
                        prev = "key_dot"
                        if first:
                            stages.extend((
                                Stage("v_load", mem(q * dv * input_bytes, read), after=(prev,), placement=owned),
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
                            Stage("attention_write", mem(n * input_bytes, "mte3"), after=("attention_cast", "state_write"), placement=owned),
                        ))
                        return tuple(stages)

                    return (Stage("state_load", mem(dk * n * state_bytes, read), placement=owned),
                            Stage("state_resident_copy", v(ak * n), after=("state_load",), placement=owned),
                            Stage("tokens", after=("state_resident_copy",), program=PipelineProgram(q, 2, token)))

                return (Stage("gamma_zero", v(q * ak), placement=owned),
                        Stage("value_tiles", after=("gamma_zero",), program=PipelineProgram(parts, 1, part)))

            pending = tuple(gdn_recurrent.head_runs(lengths, segments, compact=compact))
            heads_prog = PipelineProgram(run_count, 1, lambda idx, pending=pending: head(*pending[idx]))
            pool = context.pool_names.get("ub")
            if pool is None:
                return heads_prog
            alloc = (Allocation("recurrent_buffers", pool, gdn_recurrent.UB_BYTES,
                                unit_group(owner, units, context.pool_units.get("ub", units))),)
            return KernelProgram((Stage("buffers", acquire=alloc),
                                  Stage("heads", after=("buffers",), program=heads_prog),
                                  Stage("release", after=("heads",), free=alloc)))

        return ParallelProgram(tuple(owner_program(owner, segments, count)
                                     for owner, segments, count in zip(core_ids, ownership, run_counts)))

    meta_out = dict(base.metadata)
    sched = dict(meta_out.get("gdn_recurrent_schedule", {}))
    sched["state_residency"] = "one_initial_load_per_head;recover_ssm_compact_record_then_single_commit"
    meta_out["gdn_recurrent_schedule"] = sched
    meta_out["recover_ssm"] = True
    return OperatorPlan(work, program, metadata=meta_out)
