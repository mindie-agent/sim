"""W4A8 source rectangles on the shared matrix operand pipeline.

[Role] Bind MSD's doubled-row INT4 matrices and FP16 workspace publication.
[Invariants] Source owner/expert rectangles remain independent loads. The
  selected matrix phase replaces the former envelope; pre/post/count/assist
  work stays outside it. No observation duration becomes a service parameter.
[Timeouts & Locks] Lazy O(experts + owners) geometry; compact regular K loops.
[Side-Effect Policy] Construct programs for the common Engine only.
"""
from dataclasses import dataclass, replace
import math

from .matrix import HierarchicalMatmul
from .matrix_steady import SteadyMatmul
from .mechanism import Work
from .ownership import read_role
from .program import KernelProgram, Stage
from .w4a8_schedule import DownSchedule, GATE_BASE_K, total_work


def owner_rectangles(schedule, owner):
    """Stream the pinned rolling output grid; no sharing across expert banks."""
    start = 0
    for expert in schedule.experts:
        for local in range((owner - start) % schedule.aic_owners,
                           expert.tile_count, schedule.aic_owners):
            mi, ni = divmod(local, expert.n_tiles)
            yield (min(schedule.base_m, expert.double_rows - mi * schedule.base_m),
                   min(schedule.base_n, schedule.n - ni * schedule.base_n))
        start += expert.tile_count


def hierarchy(schedule):
    # This is the source's output-tile and declared inner matrix mapping, not
    # a recovered MatmulImpl instruction dispatch. Buffers enforce feasibility.
    return HierarchicalMatmul(
        schedule.base_m, schedule.base_n, GATE_BASE_K,
        math.gcd(128, schedule.base_m), math.gcd(128, schedule.base_n),
        reuse_inputs=False, reuse_weights=False)


def operand_bytes(schedule):
    """L1-to-L0 operands; each inner N/M tile repeats its A/B respectively."""
    mapping = hierarchy(schedule)
    left = right = 0
    for owner in range(schedule.aic_owners):
        for m, n in owner_rectangles(schedule, owner):
            left += 0.5 * m * schedule.k * math.ceil(n / mapping.compute_n)
            right += 0.5 * n * schedule.k * math.ceil(m / mapping.compute_m)
    return left, right


def matrix_work(schedule, resource_units):
    work = total_work(schedule.matrix)
    if "mte1" in resource_units:
        left, right = operand_bytes(schedule)
        work += Work(resources={"mte1": left + right})
    return work


@dataclass(frozen=True)
class WorkspacePublication:
    """SetQuantVector's existing scale work, then GetTensorC FP16 to GM."""

    def allocations(self, context, core, buffer_shape, epilogue):
        return ()

    def lower(self, context, *, core, rows, columns, **_):
        read = read_role(context.resource_names)
        scales = {"hbm": 8 * columns}
        if read in context.resource_names:
            scales[read] = 8 * columns
        output = {"hbm": 2 * rows * columns}
        if "fixpipe" in context.resource_names:
            output["fixpipe"] = 2 * rows * columns
        return KernelProgram((
            Stage("scales", context.bind(Work(resources=scales)),
                  placement=context.owned(core, context.core_count, (read,))),
            Stage("scale_vector", context.bind(Work(resources={"vector": rows * columns})),
                  after=("scales",),
                  placement=context.owned(core, context.core_count, ("vector",))),
            Stage("publish", context.bind(Work(resources=output)), after=("scale_vector",),
                  placement=context.owned(core, context.core_count, ("fixpipe",))),
        ))


def matrix_program(context, schedule, *, compact=True):
    m, n, k = 2 * schedule.m, schedule.n, schedule.k
    # The single-bank operand descriptor is consumed once by every source
    # rectangle. It does not express a bank shared between different experts.
    weight = 0.5 * n * k
    work = Work({"int4": 2 * m * n * k}, {
        "hbm": 0.5 * m * k + weight, "mte2": 0.5 * m * k + weight,
        "vector": m * n,
    })
    bound = context.for_phase(work, kernel="w4a8_msd_matrix", dtype="int4")
    metadata = {
        "packed_weight_bytes": weight,
        "activation_format": "int4", "compute_format": "int4", "output_format": "fp16",
        "matrix_epilogue": {"kind": "row_column_scale", "row_scale_bytes": 0,
                            "column_scale_bytes": 0, "operations_per_output": 1},
    }
    bound = replace(bound, op=replace(bound.op, local_shape=(m, n, k),
                                     metadata=tuple(metadata.items())))
    if bound.record_mapping is not None:
        record = bound.record_mapping
        def record_source(mapping):
            control = {
                "expert_entries": schedule.expert_scan_count,
                "active_experts": schedule.active_experts,
                "scalar_group_scan_and_flag_latency": "unresolved_not_charged",
                "column_scale_copy_bytes": 8 * sum(
                    n for owner in range(schedule.aic_owners)
                    for _, n in owner_rectangles(schedule, owner)),
            }
            if isinstance(schedule, DownSchedule):
                launched_aivs = schedule.aic_owners * schedule.aivs_per_aic
                control.update({
                    "pre_group_count_bulk_bytes": 8 * schedule.expert_scan_count * launched_aivs,
                    "pre_group_count_vector_operations": 2 * schedule.expert_scan_count * launched_aivs,
                    "matrix_group_scalar_loads": schedule.expert_scan_count * (
                        schedule.aic_owners + launched_aivs),
                    "scalar_load_width_bytes": 8,
                    "scalar_loads_are_not_assumed_uncached_HBM_transactions": True,
                })
            record({**mapping, "source_control": control,
                    "inner_mapping_basis": "declared_shared_core_not_recovered_MatmulImpl",
                    "physical_mte1_operand_bytes": list(operand_bytes(schedule)),
                    "directional_mte1": all(key in context.service_modes
                                             for key in ("mte1_l0a", "mte1_l0b"))})
        bound = replace(bound, record_mapping=record_source)
    counts = {record.owner: record.tiles for record in schedule.matrix}
    owners = tuple((counts.get(owner, 0), owner_rectangles(schedule, owner))
                   for owner in range(schedule.aic_owners))
    options = {"owner_tiles": owners, "output": WorkspacePublication(),
               "directional_mte1": True}
    mapping = hierarchy(schedule)
    if compact:
        return SteadyMatmul().lower(bound, mapping=mapping, **options)
    return mapping.lower(bound, serial_tiles=True, **options)
