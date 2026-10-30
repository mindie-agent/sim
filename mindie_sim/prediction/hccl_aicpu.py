"""HCCL AICPU Mesh1D local resource phases and exact directed exchanges.

[Role] Model explicit user/scratch or symmetric buffers, one channel per peer and SUM.
[Invariants] Remote peers run concurrently; local reductions retain source order.
  HBM counts both local copy endpoints and read-modify-write reductions. Network
  endpoint traffic is charged once here; the transport contributes link only.
  Control counters are not calibrated microsecond constants.
[Timeouts & Locks] O(ranks squared) metadata, O(ranks) local stages; no locks.
[Side-Effect Policy] Pure plans; no device/source access or observation lookup.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .hccl_contracts import HcclPhase, HcclPlan, HcclSelection, SOURCE
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, Stage

RECIPE = "hccl_aicpu_source_phase"
_OPTIONAL = {"hccl_copy", "hccl_reduce", "hccl_control"}


def _sum(steps):
    result = {}
    for _, work in steps:
        for role, amount in work:
            result[role] = result.get(role, 0) + amount
    return tuple(sorted(result.items()))


def _copy(size):
    return (("hbm", 2 * size), ("hccl_copy", size))


def _reduce(size):
    return (("hbm", 3 * size), ("hccl_reduce", size))


def _sync(name, after, threads):
    notifies = max(0, threads - 1)
    return HcclPhase(name, after, (("hccl_control", 2 * notifies),),
                     control_counts=(("notify", notifies), ("wait", notifies)))


def _exchange(name, after, matrix, rank):
    sent = sum(matrix[rank])
    received = sum(row[rank] for row in matrix)
    return HcclPhase(name, after, (("hbm", sent + received),), exchange=matrix,
                     control_counts=(("directed_peer_messages", sum(v > 0 for v in matrix[rank])),))


def _mesh(values, *, destination):
    return tuple(tuple(0 if src == dst else values[dst if destination else src]
                       for dst in range(len(values))) for src in range(len(values)))


def _chunks(selection):
    # SplitData assigns ceil(count/R) consecutive elements to each rank; the
    # final ranks may be empty. This is not a fractional N/R partition.
    count = selection.payload_bytes // selection.dtype_bytes
    chunk = (count + selection.ranks - 1) // selection.ranks
    return tuple(max(0, min(chunk, count - i * chunk)) * selection.dtype_bytes
                 for i in range(selection.ranks))


def contract(selection: HcclSelection, rank=0):
    """Per-rank local DAG with global exchanges for the caller's common Engine.

    Input/output do not alias; buffer_mode explicitly chooses separate scratch,
    symmetric read/reduce (RS/TwoShot), or remote output reads (AG).
    AG payload is local input; RS payload is complete local input (R shards).
    """
    s = selection
    if s.template == "parallel_mesh_nhr_route0":
        from .hccl_hierarchy import contract as hierarchy_contract
        return hierarchy_contract(s, rank)
    if s.template == "nhr":
        from .hccl_nhr import contract as nhr_contract
        return nhr_contract(s, rank)
    if (s.expansion != "AICPU_TS" or s.topology != "MESH_1D"
            or s.topology_levels != 1 or not 0 <= rank < s.ranks):
        raise ModelingError("hccl_aicpu_source_requires_single_mesh_rank")
    implemented = {("allreduce", "mesh_oneshot"), ("allreduce", "mesh_twoshot"),
                   ("allgather", "mesh"), ("reduce_scatter", "mesh")}
    if (s.collective, s.template) not in implemented:
        raise ModelingError("hccl_aicpu_source_template_not_implemented")
    supported_modes = ({"separate", "remote_output"} if s.collective == "allgather" else
                       {"separate"} if s.template == "mesh_oneshot" else {"separate", "symmetric"})
    if s.buffer_mode not in supported_modes:
        raise ModelingError("hccl_aicpu_source_buffer_mode_not_supported_by_template")
    if s.dtype_bytes == 8 and s.collective != "allgather":
        raise ModelingError("hccl_aicpu_64bit_reduce_cpu_path_not_implemented")
    r, n = s.ranks, s.payload_bytes
    if r == 1:
        phase = HcclPhase("local_copy", work=_copy(n))
        return HcclPlan(s, rank, (phase,), 1, 0)
    if not n:
        return HcclPlan(s, rank, (HcclPhase("empty"),), 1, 0)

    phases = []
    coeff = ()
    if s.collective == "allreduce" and s.template == "mesh_oneshot":
        threads, scratch = r, r * n
        phases.extend((
            _sync("pre_sync", (), threads),
            HcclPhase("input_to_scratch", ("pre_sync",), _copy(n)),
            _exchange("allwrite", ("pre_sync",), _mesh((n,) * r, destination=False), rank),
            _sync("post_sync", ("input_to_scratch", "allwrite"), threads),
        ))
        steps = tuple((f"reduce_{i}", _reduce(n)) for i in range(1, r))
        phases.append(HcclPhase("local_reduce", ("post_sync",), _sum(steps), steps))
        phases.append(HcclPhase("scratch_to_output", ("local_reduce",), _copy(n)))
        coeff = (("selector_kernel_count", 15), ("selector_task_count", 10 * (r - 1)))
    elif s.collective == "allreduce":
        threads = r
        chunks = _chunks(s)
        own = chunks[rank]
        scratch = own * r
        scatter = _exchange("scatter", ("rs_pre_sync",), _mesh(chunks, destination=True), rank)
        phases.append(_sync("rs_pre_sync", (), threads))
        if s.buffer_mode == "symmetric":
            # Remote read+reduce lands in input[mySlice], whose previous value
            # is read before overwrite. It needs no staging copy or later fold.
            scatter = _remote_reduce(scatter, (r-1)*own)
            phases.extend((scatter, _sync("rs_post_sync", ("scatter",), threads),
                HcclPhase("rs_result_to_output", ("rs_post_sync",), _copy(own)),
                _sync("ag_pre_sync", ("rs_result_to_output",), threads),
                _exchange("gather", ("ag_pre_sync",), _mesh(chunks, destination=False), rank),
                _sync("ag_post_sync", ("gather",), threads)))
            scratch = 0
        else:
            phases.extend((HcclPhase("rs_local_copy", ("rs_pre_sync",), _copy(own)), scatter,
                _sync("rs_post_sync", ("rs_local_copy", "scatter"), threads)))
            steps = tuple((f"reduce_{i}", _reduce(own)) for i in range(1, r))
            phases.extend((
                HcclPhase("local_reduce", ("rs_post_sync",), _sum(steps), steps),
                _sync("ag_pre_sync", ("local_reduce",), threads),
                HcclPhase("ag_local_copy", ("ag_pre_sync",), _copy(own)),
                _exchange("gather", ("ag_pre_sync",), _mesh(chunks, destination=False), rank),
                _sync("ag_post_sync", ("ag_local_copy", "gather"), threads)))
        coeff = (("selector_kernel_count", 20), ("selector_task_count", 16 * (r - 1) + 10))
    elif s.collective == "allgather":
        threads, scratch = max(1, r - 1), r * n
        # LocalDataCopy puts the contribution in user output and scratch before
        # peers are released to read it; the other chunks go directly to output.
        steps = (("input_to_output", _copy(n)), ("input_to_scratch", _copy(n)))
        if s.buffer_mode == "remote_output":
            steps, scratch = steps[:1], 0
        phases.extend((
            HcclPhase("local_copies", work=_sum(steps), steps=steps),
            _sync("pre_sync", ("local_copies",), threads),
            _exchange("gather", ("pre_sync",), _mesh((n,) * r, destination=False), rank),
            _sync("post_sync", ("gather",), threads),
        ))
    else:
        # Standalone RS's public count is each equal output shard. Unlike AR's
        # internal split, the collective input cannot have fractional shards.
        if n % (r * s.dtype_bytes):
            raise ModelingError("hccl_reduce_scatter_requires_equal_integer_output_shards")
        threads, own = r, n // r
        scratch = n
        scatter = _exchange("scatter", ("pre_sync",), _mesh((own,) * r, destination=True), rank)
        if s.buffer_mode == "symmetric":
            scatter, scratch = _remote_reduce(scatter, (r-1)*own), 0
        phases.extend((
            _sync("pre_sync", (), threads),
            scatter,
            _sync("post_sync", ("scatter",), threads),
            HcclPhase("input_to_output", ("post_sync",), _copy(own)),
        ))
        if s.buffer_mode == "separate":
            steps = tuple((f"reduce_{i}", _reduce(own)) for i in range(1, r))
            phases.append(HcclPhase("local_reduce", ("input_to_output",), _sum(steps), steps))

    return HcclPlan(s, rank, tuple(phases), threads, scratch, coeff, (
        "notify_wait_and_task_issue_latency_without_declared_hccl_control_rate",
        "thread_channel_progress_inside_each_exchange_is_a_whole_phase_envelope",
        "scratch_capacity_and_executor_message_segmentation_not_scheduled",
        "input_output_aliases_and_executor_segmentation_not_inferred",
    ))


def _remote_reduce(phase, received):
    values = dict(phase.work)
    values["hbm"] += received
    values["hccl_reduce"] = received
    return replace(phase, work=tuple(sorted(values.items())))


def build(op, geometry):
    """Local-only phase planner; its exchange is scheduled by the caller DAG."""
    raw = dict(op.metadata)["hccl_phase"]
    selection = HcclSelection(**raw["selection"])
    plan = contract(selection, raw.get("rank", 0))
    return phase_plan(op, geometry, plan, {"source": SOURCE,
        "source_cost_counters_not_timing": dict(plan.control_counts)}, "source_phase")


def phase_plan(op, geometry, plan, metadata, coverage):
    """Bind an existing local phase to the shared resource/Engine program."""
    raw = dict(op.metadata)["hccl_phase"]
    phase = next(p for p in plan.phases if p.name == raw["phase"])
    available = set(geometry.resource_units)
    omitted = {role for role, value in phase.work if value and role in _OPTIONAL and role not in available}

    def demand(values):
        return Work(resources={role: value for role, value in values if role not in omitted})

    work = demand(phase.work)

    def program(context):
        stages, previous = [], ()
        for name, values in phase.steps or (("service", phase.work),):
            stages.append(Stage(name, context.bind(demand(values)), after=previous))
            previous = (name,)
        return KernelProgram(tuple(stages))

    return OperatorPlan(work, program, {
        **metadata, "hccl_selection": plan.selection.to_dict(),
        "hccl_phase": phase.name, "thread_count": plan.thread_count,
        "scratch_bytes": plan.scratch_bytes, "control_counts": dict(phase.control_counts),
        "unpriced_service_work": {role: value for role, value in phase.work if role in omitted},
        "coverage": coverage + ("_hbm_lower_bound" if omitted else "_declared_services"),
        "unmodeled_services": plan.unmodeled,
    })
