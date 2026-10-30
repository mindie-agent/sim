"""Finite AIV programs for explicitly selected tensor materializations.

[Role] Reuse tensor-lowering payloads on owned MTE2 -> cast -> MTE3 stages.
[Invariants] Only selected payload is moved, once in each direction. A declared
  tile/slot/owner mapping determines finite UB use; observations do not choose
  it. Same-dtype copies have zero vector arithmetic but remain AIV kernels.
[Timeouts & Locks] Lazy per-owner tile windows, O(active owners * slots) tasks.
[Side-Effect Policy] Pure plans and virtual Engine submission; no I/O.
"""
from functools import partial
import math

from mindie_sim.schema.common import dtype_bytes
from .engine import Allocation
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .ownership import unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


def recipe(*, tile_elements, slots, active_units):
    """An explicit execution declaration; there is no implicit capture tiler."""
    if any(not isinstance(value, int) or value < 1 for value in (tile_elements, slots, active_units)):
        raise ValueError("aiv_tensor_copy_requires_positive_integer_tile_slots_units")
    return PlannedRecipe(partial(build, tile_elements=tile_elements, slots=slots, active_units=active_units))


def build(op, geometry, *, tile_elements, slots, active_units):
    meta = dict(op.metadata)
    elements = math.prod(op.local_shape)
    input_dtype, output_dtype = meta.get("input_format", op.dtype), meta.get("output_format", op.dtype)
    input_bytes, output_bytes = dtype_bytes(input_dtype), dtype_bytes(output_dtype)
    if min(input_bytes, output_bytes) < 1:
        raise ValueError("aiv_tensor_copy_requires_byte_addressable_elements")
    reads, writes = elements * input_bytes, elements * output_bytes
    if (op.family != "tensor_copy" or op.flops or op.link_bytes
            or not math.isclose(op.mte2_bytes, reads) or not math.isclose(op.mte3_bytes, writes)
            or not math.isclose(op.hbm_bytes, reads + writes)):
        raise ValueError("aiv_tensor_copy_payload_contract_mismatch")
    cast = input_dtype != output_dtype
    if bool(op.vector_ops) != bool(cast and elements):
        raise ValueError("aiv_tensor_copy_cast_work_contract_mismatch")
    required = {"vector", "mte2_aiv", "mte3"}
    if not required.issubset(geometry.resource_units) or "ub" not in geometry.buffers:
        raise ValueError("aiv_tensor_copy_requires_vector_split_read_store_and_ub")
    physical_units = geometry.resource_units["vector"]
    if active_units > physical_units:
        raise ValueError("aiv_tensor_copy_active_units_exceed_physical_aiv")
    tiles = math.ceil(elements / tile_elements)
    owners = min(active_units, tiles)
    # Same-dtype DMA reuses one UB tile. Casts have distinct input/output
    # storage; no aliasing of differently sized live tensors is assumed.
    tile_storage = tile_elements * (input_bytes + output_bytes if cast else input_bytes)
    storage = slots * tile_storage
    vector_per_element = op.vector_ops / elements if elements else 0
    work = Work(resources={"hbm": reads + writes, "mte2_aiv": reads,
                           "mte3": writes, "vector": op.vector_ops})

    def program(context):
        def owner_program(owner):
            owned = context.owned(owner, physical_units, ("vector", "mte2_aiv", "mte3"))
            def tile(index):
                n = min(tile_elements, elements - (owner + index * owners) * tile_elements)
                stages = [Stage("read", context.bind(Work(resources={
                    "hbm": n * input_bytes, "mte2_aiv": n * input_bytes})),
                    previous=("read",), placement=owned)]
                if cast:
                    stages.append(Stage("cast", context.bind(Work(resources={"vector": n * vector_per_element})),
                                        after=("read",), previous=("cast",), placement=owned))
                stages.append(Stage("write", context.bind(Work(resources={
                    "hbm": n * output_bytes, "mte3": n * output_bytes})),
                    after=(stages[-1].name,), previous=("write",), placement=owned))
                return tuple(stages)
            allocation = (Allocation("copy_buffers", context.pool_names["ub"], storage,
                                     unit_group(owner, physical_units,
                                                context.pool_units.get("ub", physical_units))),)
            return KernelProgram((
                Stage("buffers", acquire=allocation),
                Stage("tiles", after=("buffers",), program=PipelineProgram(
                    (tiles - 1 - owner) // owners + 1, slots, tile)),
                Stage("release", after=("tiles",), free=allocation),
            ))
        return ParallelProgram(tuple(owner_program(owner) for owner in range(owners)))

    return OperatorPlan(work, program, metadata={
        "execution_class": "vector",
        "tensor_copy_schedule": {
            "tile_elements": tile_elements, "slots": slots, "active_units": owners,
            "declared_active_units": active_units, "physical_aiv_units": physical_units,
            "tile_count": tiles, "ub_bytes_per_owner": storage,
            "assignment": "round_robin_tiles_on_declared_physical_aiv_subset",
            "layout_service": "selected_payload_without_inferred_stride_transaction_penalty",
            "input_format": input_dtype, "output_format": output_dtype,
            "cast": cast,
        },
    })
