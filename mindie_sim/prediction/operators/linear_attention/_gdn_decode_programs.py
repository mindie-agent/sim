"""Physical programs for the non-interleaved GDN decode caller.

[Role] Bind native update-convolution and Triton gating to shared resources.
[Invariants] Each logical program retains its physical AIV. Speculation writes
  history from input tokens; acceptance changes addresses, never query work.
  A padding slot executes its physical query rows but reloads no prior history
  and writes no checkpoint. FP32 gate stores never use a BF16 scalar-store
  response parameter.
[Timeouts & Locks] Per-owner lazy sequence/tile windows; no locks or device I/O.
[Side-Effect Policy] Pure plans and virtual Engine submission only.
"""
import math

from mindie_sim.schema.common import dtype_bytes

from ...mechanism import OperatorPlan, Work
from ...engine import Allocation
from ...ownership import read_role, unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...transactions import QueuedStore

GATING = "ascend_gdn_decode_gating"
CONV = "ascend_causal_conv1d_update"
BF16_BYTES = int(dtype_bytes("bf16"))
FP32_BYTES = int(dtype_bytes("fp32"))
CONV_TILE_CHANNELS = (4096, 2048, 1024, 512, 384, 192)
CONV_MAX_TILE_CHANNELS = CONV_TILE_CHANNELS[0]
CONV_SPECULATIVE_WIDTH = 4
CONV_SUPPORTED_WIDTHS = frozenset({2, 3, CONV_SPECULATIVE_WIDTH})
CONV_RING_SLOTS = 5
CONV_FP32_SCRATCH_SLOTS = 8
GATING_TILE_ROWS = 64
GATING_TILE_HEADS = 8
GATING_PER_ELEMENT_OPS = 13
#: One AIV resolves `conv_state_indices[b]`, latches the two rolling descriptors
#: and drains the previous task's MTE3 before `RunSeq` may start.
CONV_STATE_RESOLVE_US = 2.65
#: Multi-core launch, `InitSharedBuffersAndEvents` and the final MTE3 drain.
CONV_CALL_SETUP_US = 6.0
#: A rolling state ring is single-buffered: token i+1 reads what token i wrote.
CONV_TOKEN_SLOTS = 1
#: Per-token vector expression: three window taps, the Silu/bias fixed term and
#: the optional two-op bias add, each over the owned channel width.
CONV_TOKEN_TAP_OPS = 3
CONV_TOKEN_FIXED_OPS = 5
CONV_TOKEN_BIAS_OPS = 2
MS_PER_SECOND = 1000
CONV_SOURCE = {
    "path": "data/observations/sources/native-causal-conv-a4a7958",
    "revision": "a4a795857868729fbdbb0118c1e06c8655a7b31b",
    "kernel": "op_kernel/causal_conv1d.h",
    "ranges": ((310, 405), (463, 548), (888, 910), (1008, 1097), (1216, 1288)),
    "dispatch": "op_host/causal_conv1d_tiling_planner.h:25-73",
}


def update_geometry(batch, dim, units):
    """Exact-divisor candidates take precedence over candidates with a tail."""
    for exact in (True, False):
        choices = [(tile, math.ceil(dim / tile)) for tile in CONV_TILE_CHANNELS
                   if not exact or dim % tile == 0]
        if choices:
            over = [item for item in choices if batch * item[1] >= units]
            tile, columns = (min(over, key=lambda item: batch * item[1] - units)
                             if over else max(choices, key=lambda item: (batch * item[1], -item[0])))
            return tile, columns, min(units, batch * columns)


def _state_slots(lengths, padding_slots):
    slots = set(padding_slots)
    return tuple(slot not in slots for slot in range(len(lengths)))


def _conv_traffic(q, width, bias, speculative, has_state):
    """Payload descriptors for one physical sequence slot.

    A graph-padded slot retains its physical query rows (the captured graph
    still executes them) but owns no state row, so it neither reloads prior
    history nor writes a checkpoint.  The source CacheSkipped branch removes
    exactly those history descriptors, not the convolution itself.
    """
    history = width - 1 if has_state else 0
    spec_read = (2 + q) if (speculative and has_state) else 0
    spec_write = (2 + q) if (speculative and has_state) else history
    return width + int(bias) + history + q + spec_read, q + spec_write


def _conv_runs(lengths, dim, tile_width, columns, blocks, owner, *, compact,
               has_state):
    """Yield (query rows, channel width, repetitions, state_owned)."""
    count = (len(lengths) * columns - 1 - owner) // blocks + 1
    if compact and (columns == 1 or dim % tile_width == 0) and all(
            q == lengths[0] for q in lengths) and all(has_state):
        yield lengths[0], min(tile_width, dim), count, True
        return
    previous, repeat = None, 0
    for task in range(owner, len(lengths) * columns, blocks):
        seq, col = divmod(task, columns)
        # Source ResolveSeqTaskWindow rejects repeated offsets and
        # ResolveSeqCacheIndex rejects null/pad slots before LoadWeight, RunSeq
        # and writeback. The slot still participates in the host grid only.
        if lengths[seq] == 0 or not has_state[seq]:
            continue
        descriptor = (lengths[seq], min(tile_width, dim - col * tile_width),
                      has_state[seq])
        if previous is not None and (not compact or descriptor != previous):
            yield previous[0], previous[1], repeat, previous[2]
            repeat = 0
        previous = descriptor
        repeat += 1
    if previous is not None:
        yield previous[0], previous[1], repeat, previous[2]


def conv_task_runs(lengths, dim, tile_width, columns, blocks, owner, *, compact=True):
    """Yield (query rows, channel width, repetitions) in this owner's order.

    Repetitions retain distinct weights, input and state addresses. Only their
    equal resource cycles are compacted; no cache reuse or DMA coalescing follows
    from equal geometry. Ragged/tail runs stay bounded to one pending descriptor.
    ``conv_plan`` retains the per-slot state flag through the private
    ``_conv_runs`` iterator; this public diagnostic keeps its published shape.
    """
    for q, n, repeat, _ in _conv_runs(
            lengths, dim, tile_width, columns, blocks, owner, compact=compact,
            has_state=(True,) * len(lengths)):
        yield q, n, repeat


def conv_schedule(lengths, dim, units, *, width, bias, speculative, compact=True,
                  padding_slots=()):
    """Source task/transfer diagnostics; descriptor counts are not latencies."""
    tile_width, columns, blocks = update_geometry(len(lengths), dim, units)
    has_state = _state_slots(lengths, padding_slots)
    counts = tuple(sum(1 for _ in _conv_runs(
        lengths, dim, tile_width, columns, blocks, owner, compact=compact,
        has_state=has_state)) for owner in range(blocks))
    traffic = [_conv_traffic(q, width, bias, speculative, own)
               for q, own in zip(lengths, has_state) if q > 0 and own]
    return {
        "kind": "causal_conv1d_serial_runs", "compact": compact,
        "logical_tasks": len(lengths) * columns, "resource_cycles": sum(counts),
        "owner_cycles": counts, "active_aiv": blocks, "tile_channels": tile_width,
        "channel_tiles": columns,
        "padding_slots": tuple(sorted(int(slot) for slot in padding_slots)),
        "read_descriptors": columns * sum(read for read, _ in traffic),
        "write_descriptors": columns * sum(write for _, write in traffic),
        "contention_approximation": "serial_cycle_fluid_rephasing" if compact else "none",
    }


def _conv_work(lengths, has_state, width, dim, bias, speculative, read):
    # Source reloads weights per (sequence, channel tile), and recasts each
    # window tap before the multiply/add. SiLU is neg/exp/add/div/multiply.
    traffic = [_conv_traffic(q, width, bias, speculative, own)
               for q, own in zip(lengths, has_state) if q > 0 and own]
    reads = sum(amount for amount, _ in traffic) * dim * 2
    writes = sum(amount for _, amount in traffic) * dim * 2
    # Externally observed envelope of `causal_conv1d_update` on the paired
    # `#62` SKUs: `q` current tokens plus `kernel_width - 1` history rows are
    # streamed in, and the same two windows are written back. The `width`
    # weight taps stay L2-resident across the per-sequence repetitions, and the
    # speculative reload window re-reads the retained positions and the current
    # inputs from UB, so neither adds an off-chip descriptor.
    external = sum(2 * (q + width - 1) * dim * 2
                   for q, own in zip(lengths, has_state) if q > 0 and own)
    token_ops = CONV_TOKEN_TAP_OPS * width + CONV_TOKEN_FIXED_OPS + CONV_TOKEN_BIAS_OPS * int(bias)
    vector = sum((width + 1 + q * token_ops) * dim
                 for q, own in zip(lengths, has_state) if q > 0 and own)
    work = Work(resources={"hbm": external, read: reads, "mte3": writes, "vector": vector})
    return work, token_ops


def conv_plan(op, geometry, *, compact=True):
    meta = dict(op.metadata)
    lengths = tuple(meta.get("physical_sequence_tokens", meta["sequence_tokens"]))
    width = meta["kernel_width"]
    dim, dtype = op.local_shape[1], op.dtype
    if dtype not in {"bf16", "fp16"} or width not in CONV_SUPPORTED_WIDTHS:
        raise ValueError("gdn_decode_convolution_requires_16bit_width2to4")
    padding_slots = tuple(int(slot) for slot in meta.get("padding_slots", ()))
    if len(set(padding_slots)) != len(padding_slots) or any(
            slot < 0 or slot >= len(lengths) for slot in padding_slots):
        raise ValueError("gdn_decode_convolution_padding_slot_outside_partition")
    has_state = _state_slots(lengths, padding_slots)
    speculative = bool(meta["speculative"])
    bias = meta.get("bias", False)
    if not any(q > 0 and own for q, own in zip(lengths, has_state)):
        # A capacity/padding probe can retain physical buffer descriptors with
        # zero active rows.  The UPDATE source skips null/pad slots before
        # RunSeq, so no convolution or state work executes; the host grid and
        # tensor extents remain with an explicit empty program.
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"empty_active_sequences": True})
    max_query = max((q for q, own in zip(lengths, has_state) if q > 0 and own),
                    default=1)
    if speculative and (width != CONV_SPECULATIVE_WIDTH or meta["state_history_slots"] < 2 + max_query):
        raise ValueError("gdn_decode_speculative_convolution_requires_width4_history")
    if not speculative and meta["state_history_slots"] < width - 1:
        raise ValueError("gdn_decode_convolution_requires_history_width_minus_one")
    units = geometry.resource_units.get("vector", geometry.compute_units)
    schedule = conv_schedule(lengths, dim, units, width=width, bias=bias,
                             speculative=speculative, compact=compact,
                             padding_slots=padding_slots)
    tile_width, columns, blocks = schedule["tile_channels"], schedule["channel_tiles"], schedule["active_aiv"]

    def runs(owner):
        return _conv_runs(lengths, dim, tile_width, columns, blocks, owner,
                          compact=compact, has_state=has_state)
    run_counts = schedule["owner_cycles"]
    read = read_role(geometry.resource_units, aiv=0)
    work, token_ops = _conv_work(lengths, has_state, width, dim, bias, speculative, read)

    def program(context):
        def task(q, n, repeat, owns_state, owner):
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            def memory(amount, role):
                return context.bind(Work(resources={role: amount}).scaled(repeat))
            def token(index):
                stages = [
                    Stage("input", memory(n * 2, read), placement=owned),
                    Stage("convolution", context.bind(Work(resources={"vector": n * token_ops}).scaled(repeat)),
                          after=("input",), previous=("convolution",), placement=owned),
                    Stage("output", memory(n * 2, "mte3"),
                          after=("convolution",), placement=owned),
                ]
                if speculative and owns_state:
                    stages.append(Stage("state_write",
                                        memory(n * 2, "mte3"),
                                        after=("input",), placement=owned))
                elif state_write and index == q - 1:
                    # ``causal_conv1d.h`` writes the last ``W-1`` history rows
                    # from the ring updated by the final token input, so the
                    # write follows that input load; it still overlaps the last
                    # token's convolution/output instead of extending the task.
                    stages.append(Stage("state_write",
                                        memory(state_write * n * 2, "mte3"),
                                        after=("input",), placement=owned))
                return tuple(stages)
            history = width - 1 if owns_state else 0
            state_write = (2 + q) if (speculative and owns_state) else history
            stages = [
                # `ResolveSeqCacheIndex` + `ResolveSeqTaskWindow` run once per
                # task on this owner before any tap is issued, so the resolve
                # latency repeats with every equal-geometry compaction step.
                Stage("state_resolve", placement=owned,
                      latency=CONV_STATE_RESOLVE_US * repeat / MS_PER_SECOND),
                Stage("weights_history",
                      memory((width + int(bias) + history) * n * 2, read),
                      after=("state_resolve",), placement=owned),
                Stage("weights_cast", context.bind(Work(resources={"vector": (width + 1) * n}).scaled(repeat)),
                      after=("weights_history",), placement=owned),
                Stage("tokens", after=("weights_cast",),
                      program=PipelineProgram(q, CONV_TOKEN_SLOTS, token)),
            ]
            # The streamed token/history rows leave the core through the
            # shared HBM bus as one address stream, double-buffered against the
            # token convolution rather than extending the token-critical path.
            if owns_state:
                stream_rows = 2 * (q + width - 1)
                stages.append(Stage("external_stream", context.bind(Work(
                    resources={"hbm": stream_rows * n * 2}).scaled(repeat)),
                    after=("weights_cast",), placement=owned))
            # The retained-history reload is a prefetch for the speculative
            # restore; it may double-buffer against the token convolution.
            if speculative and owns_state:
                stages.append(Stage("state_reload",
                                    memory((2 + q) * n * 2, read),
                                    after=("weights_cast",), placement=owned))
                stages.append(Stage("state_retained_write",
                                    memory(2 * n * 2, "mte3"),
                                    after=("weights_cast",), placement=owned))
            return (Stage("sequence", program=KernelProgram(tuple(stages))),)
        def owner_program(owner):
            # Source drains MTE3 between tasks. Equal cycles on one owner are a
            # time-dilated serial run, not overlapping sequences. Work and each
            # owner's fixed buffers remain exact; external contention can be
            # rephased by compaction. compact=False is the detailed oracle.
            pending = tuple(runs(owner))
            tiles = PipelineProgram(run_counts[owner], 1,
                                    lambda index, pending=pending: task(*pending[index], owner))
            pool = context.pool_names.get("ub")
            if pool is None:
                return tiles
            # InitSharedBuffersAndEvents reserves the fixed ring, two outputs,
            # and eight FP32 calculation slots once per executing AIV.
            storage = CONV_MAX_TILE_CHANNELS * (
                (CONV_RING_SLOTS + 2) * BF16_BYTES + CONV_FP32_SCRATCH_SLOTS * FP32_BYTES
            )
            owned_pool = unit_group(owner, units, context.pool_units.get("ub", units))
            allocation = (Allocation("conv_buffers", pool, storage, owned_pool),)
            return KernelProgram((Stage("buffers", acquire=allocation),
                                  Stage("tasks", after=("buffers",), program=tiles),
                                  Stage("release", after=("tasks",), free=allocation)))
        # One program per physical AIV owner, as the convolution contract
        # requires: each demand-carrying task keeps its own single unit. The
        # launch-level setup is finite and precedes every owner's first task.
        return KernelProgram((
            Stage("descriptor_setup", latency=CONV_CALL_SETUP_US / MS_PER_SECOND),
            Stage("owners", after=("descriptor_setup",), program=ParallelProgram(
                tuple(owner_program(owner) for owner in range(blocks)))),
        ))
    plan_meta = {
        "padding_slots": padding_slots,
        "state_slots": tuple(slot for slot, own in enumerate(has_state) if own),
        "free_state_slots": tuple(slot for slot, own in enumerate(has_state) if not own),
        "state_write_bytes": sum(
            (2 + q if speculative else width - 1) * dim * 2
            for q, own in zip(lengths, has_state) if q > 0 and own),
    }
    if "gdn_state_contract" in meta:
        plan_meta["state_contract"] = meta["gdn_state_contract"]
    if "gdn_state_effects" in meta:
        plan_meta["state_effects"] = meta["gdn_state_effects"]
    return OperatorPlan(work, program, metadata=plan_meta)


def gating_plan(op, geometry):
    meta = dict(op.metadata)
    rows, heads = op.local_shape
    if rows <= 0:
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"empty_active_sequences": True})
    units = geometry.resource_units.get("vector", geometry.compute_units)
    programs = meta.get("logical_programs") or units
    row_iterations = math.ceil(rows / (GATING_TILE_ROWS * programs))
    columns = math.ceil(heads / GATING_TILE_HEADS)
    read = read_role(geometry.resource_units, aiv=0)
    beta_store = meta.get("beta_store", {})
    queue = beta_store.get("queue")
    queued = bool(queue) or beta_store.get("from_hardware", False)
    scalar = beta_store.get("resource", "mte3") if queued else "mte3"
    input_bytes = 2 * BF16_BYTES
    parameter_bytes = 2 * FP32_BYTES
    output_bytes = FP32_BYTES + BF16_BYTES
    reads = rows * heads * input_bytes + programs * row_iterations * heads * parameter_bytes
    writes = rows * heads * output_bytes
    per_tile = (GATING_PER_ELEMENT_OPS * GATING_TILE_ROWS + 2) * GATING_TILE_HEADS
    resources = {"hbm": reads + writes, read: reads,
                 "mte3": rows * heads * (FP32_BYTES if queued else output_bytes),
                 "vector": programs * row_iterations * columns * per_tile}
    if queued:
        resources[scalar] = rows * heads

    def program(context):
        selected_queue = queue
        if beta_store.get("from_hardware"):
            declared = context.store_queues.get(scalar, {})
            if declared.get("element_bytes") != BF16_BYTES:
                raise ValueError("gdn_beta_requires_current_hardware_bf16_store_queue")
            selected_queue = {key: value for key, value in declared.items() if key != "element_bytes"}
        def tile(index, logical, owner):
            row, col = divmod(index, columns)
            first = (logical * row_iterations + row) * GATING_TILE_ROWS
            first_head = col * GATING_TILE_HEADS
            valid_rows = max(0, min(GATING_TILE_ROWS, rows - first))
            valid_cols = min(GATING_TILE_HEADS, heads - first_head)
            elements = valid_rows * valid_cols
            owned = context.owned(owner, units, ("vector", read, "mte3"))
            incoming = elements * input_bytes + valid_cols * parameter_bytes
            g_bytes = elements * FP32_BYTES
            stages = [
                Stage("inputs", context.bind(Work(resources={"hbm": incoming, read: incoming})), placement=owned),
                Stage("gates", context.bind(Work(resources={"vector": per_tile})), after=("inputs",), placement=owned),
                Stage("g_fp32", context.bind(Work(resources={"hbm": g_bytes, "mte3": g_bytes})),
                      after=("gates",), placement=owned),
            ]
            if queued:
                # Only the declared BF16 beta path uses the address-dependent
                # store queue; g is an independent FP32 DMA output.
                name = context.resource_names[scalar]
                issue_units = context.resource_units.get(scalar, units)
                issue_owners = unit_group(owner, units, issue_units)
                beta = QueuedStore(valid_rows, valid_cols, heads * BF16_BYTES, BF16_BYTES,
                                   (first * heads + first_head) * BF16_BYTES, 1, name,
                                   context.rates[name] / issue_units * len(issue_owners),
                                   context.resource_names["hbm"], issue_placement=issue_owners, **selected_queue)
                stages.append(Stage("beta_bf16", after=("g_fp32",), program=beta))
            else:
                beta_bytes = elements * BF16_BYTES
                stages.append(Stage("beta_bf16", context.bind(Work(resources={"hbm": beta_bytes, "mte3": beta_bytes})),
                                    after=("g_fp32",), placement=owned))
            return stages
        def physical_program(owner):
            def logical(index):
                program_id = owner + index * units
                return (Stage("logical", program=PipelineProgram(
                    row_iterations * columns, 1,
                    lambda tile_id: tile(tile_id, program_id, owner))),)
            return PipelineProgram((programs - 1 - owner) // units + 1, 1, logical)

        # Every owner keeps its exact program: an owner whose 64-row window is
        # past the active rows still declares the source's per-tile gate work
        # (``per_tile`` is not scaled by rows), so dropping it would remove
        # real serviced work. Symmetric owners are combined by summing that
        # work, never by deleting it.
        from ...program import coalesce_programs
        return ParallelProgram(coalesce_programs(tuple(
            physical_program(owner)
            for owner in range(min(programs, units)))))
    return OperatorPlan(Work(resources=resources), program)
