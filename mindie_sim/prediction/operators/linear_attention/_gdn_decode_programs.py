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
from ...ownership import read_role, unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...transactions import QueuedStore
from mindie_sim.prediction.operators.causal_conv import (
    CONV as CONV,
    CONV_SOURCE as CONV_SOURCE,
    CONV_SPECULATIVE_WIDTH as CONV_SPECULATIVE_WIDTH,
    CONV_SUPPORTED_WIDTHS as CONV_SUPPORTED_WIDTHS,
    conv_plan as conv_plan,
    conv_schedule as conv_schedule,
    conv_spec_terms as conv_spec_terms,
    conv_task_runs as conv_task_runs,
    conv_token_ops as conv_token_ops,
    update_geometry as update_geometry,
)

GATING = "ascend_gdn_decode_gating"
BF16_BYTES = int(dtype_bytes("bf16"))
FP32_BYTES = int(dtype_bytes("fp32"))
GATING_TILE_ROWS = 64
GATING_TILE_HEADS = 8
GATING_PER_ELEMENT_OPS = 13


def gating_spec_terms(rows, heads):
    """Unpadded ``(reads, writes, vector)`` lower bound of the gating kernel.

    Every element is loaded, gated and stored once and the per-head
    parameters are read at least once; tile padding only adds work.
    """
    elements = rows * heads
    reads = elements * 2 * BF16_BYTES + heads * 2 * FP32_BYTES
    writes = elements * (FP32_BYTES + BF16_BYTES)
    return reads, writes, elements * GATING_PER_ELEMENT_OPS


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
