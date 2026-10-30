"""Explicit noncontiguous convolution-state staging on the shared Engine.

[Role] Lower the retained Triton gather/scatter caller around a convolution.
[Invariants] Copy the current requests' complete declared state extent, not the
  cache capacity or token count. Gather, convolution and writeback are ordered;
  one source grid is one invocation. No observed duration supplies work/rates.
[Timeouts & Locks] Lazy per-owner program windows; no locks or external I/O.
[Side-Effect Policy] Builds operators and resource programs only.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .engine import Allocation
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .source_program_service import SourceProgramService

IMPLEMENTATION = "triton_noncontiguous_conv_state_copy"
SOURCE = {
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "path": "vllm_ascend/models/glm5next/ops/causal_conv1d.py",
    "sha256": "106311478c25ef1981b95b33b9c4e7fb94fc4856cef53bca6a484ef2cd0061b8",
    "lines": "13-47,60-105", "block_elements": 256,
    "scope": "retained_source_caller; capture_binary_and_compiler_schedule_unverified",
}


def recipes():
    return {IMPLEMENTATION: PlannedRecipe(build)}


def wrap(b, *, conv, views, width, state_slots=None, cache_index_dtype="int32",
         query_start_dtype="int32", joint_program_service=None):
    """Declare the dim-first persistent view and distinct valid active slots.

    The existing convolution remains its own selected/analytic program. This
    adapter adds only the source caller's staging and output initialization.
    """
    if (not isinstance(state_slots, int) or state_slots < b.plan.conv_kernel - 1
            or any(view.phase != "decode" or view.q_len < 1 for view in views)):
        raise ModelingError("conv_state_copy_requires_decode_and_declared_history_capacity")
    if cache_index_dtype not in {"int32", "int64"} or query_start_dtype not in {"int32", "int64"}:
        raise ModelingError("conv_state_copy_requires_integer_metadata")
    requests, dtype = len(views), b.plan.conv_state_dtype
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ModelingError("conv_state_copy_requires_byte_addressable_state")
    pos = next(i for i, op in enumerate(b.ops) if op.op_id == conv)
    kernel = b.ops[pos]
    before = len(b.ops)
    common = (("implementation", IMPLEMENTATION), ("source", SOURCE),
              ("native_kind", "_copy_conv_state_kernel"), ("invocation_class", "vector"),
              ("state_slots", state_slots), ("state_dtype", dtype),
              ("persistent_layout", "cache_channels_history_transposed_to_cache_history_channels"),
              ("state_strides", (1, state_slots)),
              ("cache_indices", "declared_distinct_valid_slot_per_active_request"),
              ("cache_index_dtype", cache_index_dtype), ("query_start_dtype", query_start_dtype),
              ("active_requests", (True,) * requests),
              ("logical_grid", (requests, math.ceil(state_slots * width / 256))))
    if joint_program_service:
        common += (("joint_program_service", joint_program_service),)
    gather = b.work("convolution_state_gather", kernel.deps, (requests, state_slots, width),
                    family="conv_state_copy", dtype=dtype,
                    metadata=common + (("direction", "gather"),))
    b.ops[-1] = replace(b.ops[-1], recipe=IMPLEMENTATION, source_boundary="kernel")
    q_rows = sum(view.q_len for view in views)
    zero = b.work("convolution_output_zero", kernel.deps, (q_rows, width),
                  vector=q_rows * width, write=q_rows * width * dtype_bytes(b.model.activation_dtype),
                  family="fill", dtype=b.model.activation_dtype,
                  metadata=(("caller", SOURCE), ("reason", "torch.zeros_like(x)"),
                            ("invocation_class", "vector")))
    scatter = b.work("convolution_state_scatter", (conv,), (requests, state_slots, width),
                     family="conv_state_copy", dtype=dtype,
                     metadata=common + (("direction", "scatter"),
                                        ("passthrough_result", conv)))
    b.ops[-1] = replace(b.ops[-1], recipe=IMPLEMENTATION, source_boundary="kernel")
    additions = b.ops[before:]
    b.ops = (b.ops[:pos] + additions[:2]
             + [replace(kernel, deps=tuple(dict.fromkeys((*kernel.deps, gather, zero))))]
             + [additions[2]] + b.ops[pos + 1:before])
    # Same-stream caller order keeps downstream work after state writeback.
    return scatter


def build(op, geometry):
    meta = dict(op.metadata)
    batch, slots, dim = op.local_shape
    active = tuple(meta["active_requests"])
    scatter = meta["direction"] == "scatter"
    byte = dtype_bytes(op.dtype)
    index_byte = {"int32": 4, "int64": 8}[meta["cache_index_dtype"]]
    start_byte = {"int32": 4, "int64": 8}[meta["query_start_dtype"]]
    columns = math.ceil(slots * dim / 256)
    programs = batch * columns
    units = geometry.resource_units.get("vector", geometry.compute_units)
    owners = min(units, programs)
    read = read_role(geometry.resource_units, aiv=0)
    if "ub" not in geometry.buffers:
        raise ModelingError("conv_state_copy_requires_declared_ub")
    # Each source program loads one index and two prefix entries. A conditional
    # joint response may price the entire source schedule; it cannot identify
    # independent metadata loads, compiled descriptors or address instructions.
    metadata_reads = programs * (index_byte + 2 * start_byte)
    reads = (batch if scatter else sum(active)) * slots * dim * byte
    writes = (sum(active) if scatter else batch) * slots * dim * byte
    indices_write = 0 if scatter else batch * 4
    work = Work(resources={"hbm": reads + writes + metadata_reads + indices_write,
                           read: reads, "mte3": writes + indices_write})
    joint, joint_basis = SourceProgramService.select(
        meta.get("joint_program_service"), direction=meta["direction"],
        payload_bytes=256 * byte, metadata_bytes=index_byte + 2 * start_byte,
        all_active=all(active), full_programs=slots * dim % 256 == 0,
        state_shape=(slots, dim), dtype=op.dtype, owners=units)
    if joint:
        work += joint.work(programs, owners)

    def program(context):
        def owner_program(owner):
            owned = context.owned(owner, units, ("vector", read, "mte3"))

            def tile(i):
                # First grid dimension is request. Mapping to physical AIVs is
                # an explicit round-robin execution declaration, not BlockNum
                # evidence for the retained capture's compiler launch mapping.
                logical = owner + i * owners
                col, request = divmod(logical, batch)
                n = min(256, slots * dim - 256 * col) * byte
                read_bytes = n if scatter or active[request] else 0
                write_bytes = n if not scatter or active[request] else 0
                if not scatter and col == 0:
                    write_bytes += 4
                return (
                    Stage("metadata", context.bind(Work(resources={
                        "hbm": index_byte + 2 * start_byte}))),
                    Stage("load", context.bind(Work(resources={"hbm": read_bytes, read: read_bytes})),
                          after=("metadata",), placement=owned),
                    Stage("store", context.bind(Work(resources={"hbm": write_bytes, "mte3": write_bytes})),
                          after=("load",), placement=owned),
                )
            allocation = (Allocation("copy_payload", context.pool_names["ub"], 256 * byte,
                                     unit_group(owner, units, context.pool_units.get("ub", units))),)
            count = (programs - 1 - owner) // owners + 1
            busy = (joint.stage(context, count, owner, units),) if joint else ()
            return KernelProgram((Stage("reserve", acquire=allocation),
                                  Stage("grid", after=("reserve",), program=PipelineProgram(
                                      count, 1, tile)), *busy,
                                  Stage("release", after=("grid",) + tuple(s.name for s in busy),
                                        free=allocation)))
        return ParallelProgram(tuple(owner_program(owner) for owner in range(owners)))
    return OperatorPlan(work, program, metadata={
        "execution_class": "vector", "copy_grid": (batch, columns),
        "physical_aiv_owners": owners, "metadata_loads": 3 * programs,
        "source_programs": programs,
        "source_payload_load_store_pairs": programs,
        "source_programs_per_owner": tuple(
            (programs - 1 - owner) // owners + 1 for owner in range(owners)),
        "payload_read_bytes": reads, "payload_write_bytes": writes,
        "packed_indices_write_bytes": indices_write,
        "joint_program_service_basis": joint_basis,
        "joint_pipe_busy_ns": dict(joint.work(programs, owners).resources) if joint else {},
        "joint_response_boundary": "overlapping_owner_busy_not_summed_wall_time",
        "unpriced": ("scalar_metadata_issue", "integer_address_generation",
                     "strided_memory_transaction_amplification", "compiled_register_spills"),
        "schedule": "declared_round_robin_request_minor_grid_single_payload_slot",
    })
