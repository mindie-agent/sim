"""Physical programs for the non-interleaved GDN decode caller.

[Role] Bind native update-convolution and Triton gating to shared resources.
[Invariants] Each logical program retains its physical AIV. Speculation writes
  history from input tokens; acceptance changes addresses, never query work.
  FP32 gate stores never use a BF16 scalar-store response parameter.
[Timeouts & Locks] Per-owner lazy sequence/tile windows; no locks or device I/O.
[Side-Effect Policy] Pure plans and virtual Engine submission only.
"""
import math

from .mechanism import OperatorPlan, Work
from .engine import Allocation
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .transactions import QueuedStore

GATING = "ascend_gdn_decode_gating"
CONV = "ascend_causal_conv1d_update"
CONV_SOURCE = {
    "path": "data/observations/sources/native-causal-conv-a4a7958",
    "revision": "a4a795857868729fbdbb0118c1e06c8655a7b31b",
    "kernel": "op_kernel/causal_conv1d.h",
    "ranges": ((310, 405), (463, 548), (888, 910), (1008, 1097), (1216, 1288)),
    "dispatch": "op_host/causal_conv1d_tiling_planner.h:25-73",
}


def update_geometry(batch, dim, units):
    """Exact-divisor candidates take precedence over candidates with a tail."""
    candidates = (4096, 2048, 1024, 512, 384, 192)
    for exact in (True, False):
        choices = [(tile, math.ceil(dim / tile)) for tile in candidates
                   if not exact or dim % tile == 0]
        if choices:
            over = [item for item in choices if batch * item[1] >= units]
            tile, columns = (min(over, key=lambda item: batch * item[1] - units)
                             if over else max(choices, key=lambda item: (batch * item[1], -item[0])))
            return tile, columns, min(units, batch * columns)


def conv_task_runs(lengths, dim, tile_width, columns, blocks, owner, *, compact=True):
    """Yield (query rows, channel width, repetitions) in this owner's order.

    Repetitions retain distinct weights, input and state addresses. Only their
    equal resource cycles are compacted; no cache reuse or DMA coalescing follows
    from equal geometry. Ragged/tail runs stay bounded to one pending descriptor.
    """
    count = (len(lengths) * columns - 1 - owner) // blocks + 1
    if compact and (columns == 1 or dim % tile_width == 0) and all(q == lengths[0] for q in lengths):
        yield lengths[0], min(tile_width, dim), count
        return
    previous, repeat = None, 0
    for task in range(owner, len(lengths) * columns, blocks):
        seq, col = divmod(task, columns)
        descriptor = lengths[seq], min(tile_width, dim - col * tile_width)
        if previous is not None and (not compact or descriptor != previous):
            yield *previous, repeat
            repeat = 0
        previous = descriptor
        repeat += 1
    if previous is not None:
        yield *previous, repeat


def conv_schedule(lengths, dim, units, *, width, bias, speculative, compact=True):
    """Source task/transfer diagnostics; descriptor counts are not latencies."""
    tile_width, columns, blocks = update_geometry(len(lengths), dim, units)
    counts = tuple(sum(1 for _ in conv_task_runs(
        lengths, dim, tile_width, columns, blocks, owner, compact=compact)) for owner in range(blocks))
    return {
        "kind": "causal_conv1d_serial_runs", "compact": compact,
        "logical_tasks": len(lengths) * columns, "resource_cycles": sum(counts),
        "owner_cycles": counts, "active_aiv": blocks, "tile_channels": tile_width,
        "channel_tiles": columns,
        "read_descriptors": columns * sum(width + int(bias) + width - 1 + q
                                           + (2 + q if speculative else 0) for q in lengths),
        "write_descriptors": columns * sum(q + (2 + q if speculative else width - 1) for q in lengths),
        "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
    }


def conv_plan(op, geometry, *, compact=True):
    meta = dict(op.metadata)
    lengths, width = tuple(meta["sequence_tokens"]), meta["kernel_width"]
    dim, dtype = op.local_shape[1], op.dtype
    if dtype not in {"bf16", "fp16"} or width not in {2, 3, 4}:
        raise ValueError("gdn_decode_convolution_requires_16bit_width2to4")
    speculative, bias = meta["speculative"], meta.get("bias", False)
    if speculative and (width != 4 or meta["state_history_slots"] < 2 + max(lengths)):
        raise ValueError("gdn_decode_speculative_convolution_requires_width4_history")
    units = geometry.resource_units.get("vector", geometry.compute_units)
    schedule = conv_schedule(lengths, dim, units, width=width, bias=bias,
                             speculative=speculative, compact=compact)
    tile_width, columns, blocks = schedule["tile_channels"], schedule["channel_tiles"], schedule["active_aiv"]
    def runs(owner):
        return conv_task_runs(lengths, dim, tile_width, columns, blocks, owner, compact=compact)
    run_counts = schedule["owner_cycles"]
    read = read_role(geometry.resource_units, aiv=0)
    # Source reloads weights per (sequence, channel tile), and recasts each
    # window tap before the multiply/add. SiLU is neg/exp/add/div/multiply.
    reads = sum((width + int(bias) + width - 1 + q + (2 + q if speculative else 0))
                * dim * 2 for q in lengths)
    writes = sum((q + (2 + q if speculative else width - 1)) * dim * 2 for q in lengths)
    vector = sum((width + 1 + q * (3 * width + 5 + 2 * int(bias))) * dim for q in lengths)
    work = Work(resources={"hbm": reads + writes, read: reads, "mte3": writes, "vector": vector})

    def program(context):
        def task(q, n, repeat, owner):
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            def memory(amount, role):
                return context.bind(Work(resources={"hbm": amount, role: amount}).scaled(repeat))
            def token(index):
                return (
                    Stage("input", memory(n * 2, read), placement=owned),
                    Stage("convolution", context.bind(Work(resources={
                        "vector": n * (3 * width + 5 + 2 * int(bias))}).scaled(repeat)),
                          after=("input",), previous=("convolution",), placement=owned),
                    Stage("output", memory(n * 2, "mte3"), after=("convolution",), placement=owned),
                )
            stages = [
                Stage("weights_history", memory((width + int(bias) + width - 1) * n * 2, read),
                      placement=owned),
                Stage("weights_cast", context.bind(Work(resources={"vector": (width + 1) * n}).scaled(repeat)),
                      after=("weights_history",), placement=owned),
                Stage("tokens", after=("weights_cast",), program=PipelineProgram(q, 2, token)),
            ]
            if speculative:
                # Source rereads the retained two history positions and all
                # current input tokens for speculative state restoration.
                stages.append(Stage("state_reload", memory((2 + q) * n * 2, read),
                                    after=("tokens",), placement=owned))
            stages.append(Stage("state_write", memory((2 + q if speculative else width - 1) * n * 2, "mte3"),
                                after=(stages[-1].name,), placement=owned))
            return (Stage("sequence", program=KernelProgram(tuple(stages))),)
        def owner_program(owner):
            # Source drains MTE3 between tasks. Equal cycles on one owner are a
            # time-dilated serial run, not overlapping sequences. Work and each
            # owner's fixed buffers remain exact; external contention can be
            # rephased by compaction. compact=False is the detailed oracle.
            pending = runs(owner)
            tiles = PipelineProgram(run_counts[owner], 1,
                                    lambda index: task(*next(pending), owner))
            pool = context.pool_names.get("ub")
            if pool is None:
                return tiles
            # InitSharedBuffersAndEvents reserves the fixed ring, two outputs,
            # and eight FP32 calculation slots once per executing AIV.
            storage = (5 + 2) * 4096 * 2 + 8 * 4096 * 4
            owned_pool = unit_group(owner, units, context.pool_units.get("ub", units))
            allocation = (Allocation("conv_buffers", pool, storage, owned_pool),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("tasks", after=("buffers",), program=tiles),
                                  Stage("release", after=("tasks",), free=allocation)))
        return ParallelProgram(tuple(owner_program(owner) for owner in range(blocks)))
    return OperatorPlan(work, program)


def gating_plan(op, geometry):
    meta = dict(op.metadata)
    rows, heads = op.local_shape
    units = geometry.resource_units.get("vector", geometry.compute_units)
    programs = meta.get("logical_programs") or units
    row_iterations, columns = math.ceil(rows / (64 * programs)), math.ceil(heads / 8)
    read = read_role(geometry.resource_units, aiv=0)
    beta_store = meta.get("beta_store", {})
    queue = beta_store.get("queue")
    queued = bool(queue) or beta_store.get("from_hardware", False)
    scalar = beta_store.get("resource", "mte3") if queued else "mte3"
    reads = rows * heads * 4 + programs * row_iterations * heads * 8
    writes = rows * heads * 6
    per_tile = 13 * 64 * 8 + 2 * 8
    resources = {"hbm": reads + writes, read: reads,
                 "mte3": rows * heads * (4 if queued else 6),
                 "vector": programs * row_iterations * columns * per_tile}
    if queued:
        resources[scalar] = rows * heads

    def program(context):
        selected_queue = queue
        if beta_store.get("from_hardware"):
            declared = context.store_queues.get(scalar, {})
            if declared.get("element_bytes") != 2:
                raise ValueError("gdn_beta_requires_current_hardware_bf16_store_queue")
            selected_queue = {key: value for key, value in declared.items() if key != "element_bytes"}
        def tile(index, logical, owner):
            row, col = divmod(index, columns)
            first = (logical * row_iterations + row) * 64
            valid_rows, valid_cols = max(0, min(64, rows - first)), min(8, heads - col * 8)
            elements = valid_rows * valid_cols
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            incoming = elements * 4 + valid_cols * 8
            stages = [
                Stage("inputs", context.bind(Work(resources={"hbm": incoming, read: incoming})), placement=owned),
                Stage("gates", context.bind(Work(resources={"vector": per_tile})), after=("inputs",), placement=owned),
                Stage("g_fp32", context.bind(Work(resources={"hbm": elements * 4, "mte3": elements * 4})),
                      after=("gates",), placement=owned),
            ]
            if queued:
                # Only the declared BF16 beta path uses the address-dependent
                # store queue; g is an independent FP32 DMA output.
                name = context.resource_names[scalar]
                issue_units = context.resource_units.get(scalar, units)
                issue_owners = unit_group(owner, units, issue_units)
                beta = QueuedStore(valid_rows, valid_cols, heads * 2, 2,
                                   (first * heads + col * 8) * 2, 1, name,
                                   context.rates[name] / issue_units * len(issue_owners),
                                   context.resource_names["hbm"], issue_placement=issue_owners, **selected_queue)
                stages.append(Stage("beta_bf16", after=("g_fp32",), program=beta))
            else:
                stages.append(Stage("beta_bf16", context.bind(Work(resources={"hbm": elements * 2, "mte3": elements * 2})),
                                    after=("g_fp32",), placement=owned))
            return stages
        def physical_program(owner):
            def logical(index):
                program_id = owner + index * units
                return (Stage("logical", program=PipelineProgram(
                    row_iterations * columns, 1,
                    lambda tile_id: tile(tile_id, program_id, owner))),)
            return PipelineProgram((programs - 1 - owner) // units + 1, 1, logical)
        return ParallelProgram(tuple(physical_program(owner) for owner in range(min(programs, units))))
    return OperatorPlan(Work(resources=resources), program)
