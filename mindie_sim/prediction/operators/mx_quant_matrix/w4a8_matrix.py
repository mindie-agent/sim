"""W4A8 source rectangles on the shared matrix operand pipeline.

[Role] Bind MSD's doubled-row INT4 matrices and FP16 workspace publication.
[Invariants] Source owner/expert rectangles remain independent loads. The
  selected matrix phase replaces the former envelope; pre/post/count/assist
  work stays outside it. No observation duration becomes a service parameter.
[Timeouts & Locks] Lazy O(experts + owners) geometry; compact regular K loops.
[Side-Effect Policy] Construct programs for the common Engine only.
"""
from dataclasses import asdict, dataclass, replace
import math
from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.operators.matrix.matrix import (
    HierarchicalMatmul, directional_mte1_available, operand_transfer,
)
from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
from mindie_sim.prediction.operators.matrix.refill_steady import compact_refill
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .w4a8_schedule import (
    COLUMN_SCALE_BYTES, GATE_BASE_K, GROUP_COUNT_BYTES, DownSchedule, align_up, total_work,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h",
        source_sha256="1881d03e65258e846fe9bbfb736f4cbbcd770b84a6730319dd0c9d00abdc4087",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h",
        source_sha256="b092d276255c441542a5d205dbec5718d2ba5ae0ab7f5e637a2dba5fb20ee619",
    ),
)

INNER_TILE_ALIGNMENT = 128
L0C_ELEMENT_BYTES = 4
REFILL_PIPELINE_SLOTS = 2
CONSUMER_PIPELINE_SLOTS = 2


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
        math.gcd(INNER_TILE_ALIGNMENT, schedule.base_m), math.gcd(INNER_TILE_ALIGNMENT, schedule.base_n),
        reuse_inputs=False, reuse_weights=False)


def operand_bytes(schedule):
    """L1-to-L0 operands padded to INT4 subtile; each inner N/M tile repeats its A/B."""
    mapping = hierarchy(schedule)
    count, tail = divmod(schedule.k, mapping.tile_k)
    issued_k = (
        count * align_up(mapping.tile_k, schedule.subtile_k)
        + (align_up(tail, schedule.subtile_k) if tail else 0)
    )
    left = right = 0
    for owner in range(schedule.aic_owners):
        for m, n in owner_rectangles(schedule, owner):
            issued_m = align_up(m, schedule.subtile_m)
            issued_n = align_up(n, schedule.subtile_n)
            left += 0.5 * issued_m * issued_k * math.ceil(n / mapping.compute_n)
            right += 0.5 * issued_n * issued_k * math.ceil(m / mapping.compute_m)
    return left, right


def matrix_work(schedule, resource_units, *, directional=None):
    work = total_work(schedule.matrix)
    use_dir = (
        ("mte1_l0a" in resource_units and "mte1_l0b" in resource_units)
        if directional is None else bool(directional)
    )
    if use_dir:
        left, right = operand_bytes(schedule)
        work += Work(resources={"mte1_l0a": left, "mte1_l0b": right})
    elif "mte1" in resource_units:
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
        scales = {"hbm": COLUMN_SCALE_BYTES * columns}
        if read in context.resource_names:
            scales[read] = COLUMN_SCALE_BYTES * columns
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


def _record_matrix_mapping(bound, schedule, mapping):
    control = {
        "expert_entries": schedule.expert_scan_count,
        "active_experts": schedule.active_experts,
        "scalar_group_scan_and_flag_latency": "unresolved_not_charged",
        "column_scale_copy_bytes": COLUMN_SCALE_BYTES * sum(
            n for owner in range(schedule.aic_owners)
            for _, n in owner_rectangles(schedule, owner)),
    }
    if isinstance(schedule, DownSchedule):
        launched_aivs = schedule.aic_owners * schedule.aivs_per_aic
        control.update({
            "pre_group_count_bulk_bytes": GROUP_COUNT_BYTES * schedule.expert_scan_count * launched_aivs,
            "pre_group_count_vector_operations": 2 * schedule.expert_scan_count * launched_aivs,
            "matrix_group_scalar_loads": schedule.expert_scan_count * (
                schedule.aic_owners + launched_aivs),
            "scalar_load_width_bytes": GROUP_COUNT_BYTES,
            "scalar_loads_are_not_assumed_uncached_HBM_transactions": True,
        })
    bound.record_mapping({
        "basis": "declared_source_owner_mapping",
        "selected": asdict(mapping),
        "shape": [2 * schedule.m, schedule.n, schedule.k],
        "source_control": control,
        "inner_mapping_basis": "declared_shared_core_not_recovered_MatmulImpl",
        "physical_mte1_operand_bytes": list(operand_bytes(schedule)),
        "directional_mte1": directional_mte1_available(bound),
    })


def _tile_panel_parts(bound, schedule, mapping, *, core, cores, m_tile, n_tile, kd, read,
                      tile_idx, panel_idx=None, include_bufs=False):
    def alloc(name, pool, size):
        if pool not in bound.pool_names or size <= 0:
            return ()
        return (bound.allocate(name, pool, size, (core,)),)

    a_copy = 0.5 * m_tile * kd
    b_copy = 0.5 * n_tile * kd
    load_res = {"hbm": a_copy + b_copy}
    if read in bound.resource_names:
        load_res[read] = a_copy + b_copy
    l1_alloc = alloc(f"t{tile_idx}:p{panel_idx}:l1", "l1", a_copy + b_copy) if include_bufs else ()
    load_stage = Stage("load", bound.bind(Work(resources=load_res)),
                       acquire=l1_alloc, placement=bound.owned(core, cores, (read,)))
    inner_m = math.ceil(m_tile / mapping.compute_m)
    inner_n = math.ceil(n_tile / mapping.compute_n)
    issued_kd = align_up(kd, schedule.subtile_k)
    steps = []
    step_idx = 0
    for mi in range(inner_m):
        ai = min(mapping.compute_m, m_tile - mi * mapping.compute_m)
        issued_ai = align_up(ai, schedule.subtile_m)
        for ni in range(inner_n):
            bi = min(mapping.compute_n, n_tile - ni * mapping.compute_n)
            issued_bi = align_up(bi, schedule.subtile_n)
            left = 0.5 * issued_ai * issued_kd
            right = 0.5 * issued_bi * issued_kd
            local_alloc = (
                (*alloc(f"t{tile_idx}:p{panel_idx}:s{step_idx}:l0a", "l0a", left),
                 *alloc(f"t{tile_idx}:p{panel_idx}:s{step_idx}:l0b", "l0b", right))
                if include_bufs else ()
            )
            op_stage = Stage(
                "operand", operand_transfer(bound, left, right, directional=True),
                acquire=local_alloc, placement=bound.owned(core, cores, ("mte1",)),
            )
            cb_stage = Stage(
                "cube", bound.bind(Work({"int4": 2 * issued_ai * issued_bi * issued_kd})),
                after=("operand",), free=local_alloc,
                placement=bound.owned(core, cores, ("cube",)),
            )
            steps.append((op_stage, cb_stage))
            step_idx += 1
    return load_stage, tuple(steps), l1_alloc


def _compact_tile_allocations(bound, schedule, mapping, *, core, tile_idx, m_tile, n_tile, count, tail):
    def alloc(name, pool, size):
        if pool not in bound.pool_names or size <= 0:
            return ()
        return (bound.allocate(name, pool, size, (core,)),)

    l1_peak_k = (
        REFILL_PIPELINE_SLOTS * mapping.tile_k
        if count >= REFILL_PIPELINE_SLOTS
        else count * mapping.tile_k + tail
    )
    sample_panels = [mapping.tile_k] * min(count, CONSUMER_PIPELINE_SLOTS) + ([tail] if tail else [])
    inner_m = math.ceil(m_tile / mapping.compute_m)
    inner_n = math.ceil(n_tile / mapping.compute_n)
    l0a_steps, l0b_steps = [], []
    for kd in sample_panels:
        ik = align_up(kd, schedule.subtile_k)
        for mi in range(inner_m):
            ai = align_up(min(mapping.compute_m, m_tile - mi * mapping.compute_m), schedule.subtile_m)
            for ni in range(inner_n):
                bi = align_up(min(mapping.compute_n, n_tile - ni * mapping.compute_n), schedule.subtile_n)
                l0a_steps.append(0.5 * ai * ik)
                l0b_steps.append(0.5 * bi * ik)
    win = min(len(l0a_steps), CONSUMER_PIPELINE_SLOTS)
    l0a_peak = max((sum(l0a_steps[i:i + win]) for i in range(len(l0a_steps) - win + 1)), default=0)
    l0b_peak = max((sum(l0b_steps[i:i + win]) for i in range(len(l0b_steps) - win + 1)), default=0)
    l0c_bytes = (
        align_up(m_tile, schedule.subtile_m)
        * align_up(n_tile, schedule.subtile_n)
        * L0C_ELEMENT_BYTES
    )
    return (
        *alloc(f"t{tile_idx}:l1", "l1", 0.5 * (m_tile + n_tile) * l1_peak_k),
        *alloc(f"t{tile_idx}:l0a", "l0a", l0a_peak),
        *alloc(f"t{tile_idx}:l0b", "l0b", l0b_peak),
        *alloc(f"t{tile_idx}:l0c", "l0c", l0c_bytes),
    )


def matrix_program(context, schedule, *, compact=True):
    m, n, k = 2 * schedule.m, schedule.n, schedule.k
    work = matrix_work(schedule, context.resource_units, directional=directional_mte1_available(context))
    bound = context.for_phase(work, kernel="w4a8_msd_matrix", dtype="int4")
    metadata = {
        "packed_weight_bytes": 0.5 * n * k,
        "activation_format": "int4", "compute_format": "int4", "output_format": "fp16",
        "matrix_epilogue": {"kind": "row_column_scale", "row_scale_bytes": 0,
                            "column_scale_bytes": 0, "operations_per_output": 1},
    }
    bound = replace(bound, op=replace(bound.op, local_shape=(m, n, k),
                                      metadata=tuple(metadata.items())))
    mapping = hierarchy(schedule)
    if bound.record_mapping is not None:
        _record_matrix_mapping(bound, schedule, mapping)
    cores = schedule.aic_owners
    read = read_role(bound.resource_names)
    publication = WorkspacePublication()
    count, tail = divmod(k, mapping.tile_k)
    owner_rects = [
        (core, rects)
        for core in range(cores)
        if (rects := tuple(owner_rectangles(schedule, core)))
    ]
    shared_clients = (
        {bound.resource_names["hbm"]: len(owner_rects)}
        if "hbm" in bound.resource_names and len(owner_rects) > 1 else None
    )

    def owner_prog(core, rectangles):
        def tile(tile_idx):
            m_tile, n_tile = rectangles[tile_idx]
            if compact:
                allocs = _compact_tile_allocations(
                    bound, schedule, mapping, core=core, tile_idx=tile_idx,
                    m_tile=m_tile, n_tile=n_tile, count=count, tail=tail,
                )
                full_part = _tile_panel_parts(
                    bound, schedule, mapping, core=core, cores=cores, m_tile=m_tile,
                    n_tile=n_tile, kd=mapping.tile_k, read=read, tile_idx=tile_idx,
                ) if count else None
                tail_part = _tile_panel_parts(
                    bound, schedule, mapping, core=core, cores=cores, m_tile=m_tile,
                    n_tile=n_tile, kd=tail, read=read, tile_idx=tile_idx,
                ) if tail else None
                full = (((full_part[0],), full_part[1]),) if full_part is not None else ()
                last = (((tail_part[0],), tail_part[1]),) if tail_part is not None else ()
                k_prog = compact_refill(
                    bound, full, count, last,
                    refill_slots=REFILL_PIPELINE_SLOTS,
                    consumer_slots=CONSUMER_PIPELINE_SLOTS,
                    shared_clients=shared_clients,
                )
            else:
                l0c_bytes = (
                    align_up(m_tile, schedule.subtile_m)
                    * align_up(n_tile, schedule.subtile_n)
                    * L0C_ELEMENT_BYTES
                )
                allocs = (
                    (bound.allocate(f"t{tile_idx}:l0c", "l0c", l0c_bytes, (core,)),)
                    if "l0c" in bound.pool_names else ()
                )

                def panel(panel_idx):
                    kd = tail if tail and panel_idx == count else mapping.tile_k
                    load_st, steps, l1_alloc = _tile_panel_parts(
                        bound, schedule, mapping, core=core, cores=cores, m_tile=m_tile,
                        n_tile=n_tile, kd=kd, read=read, tile_idx=tile_idx,
                        panel_idx=panel_idx, include_bufs=True,
                    )
                    return (load_st, iter((pair, i + 1 == len(steps))
                                          for i, pair in enumerate(steps)), l1_alloc)

                k_prog = RefillPipeline(
                    count + bool(tail), REFILL_PIPELINE_SLOTS, CONSUMER_PIPELINE_SLOTS, panel,
                )
            out_prog = publication.lower(bound, core=core, rows=m_tile, columns=n_tile)
            stages = [
                Stage("reserve", acquire=allocs),
                Stage("K", after=("reserve",), program=k_prog),
                Stage("output", after=("K",), program=out_prog),
                Stage("release", after=("output",), free=allocs),
            ]
            return (Stage("tile", program=KernelProgram(tuple(stages))),)

        return PipelineProgram(len(rectangles), 1, tile)

    active_owners = [owner_prog(core, rects) for core, rects in owner_rects]
    return KernelProgram((
        Stage("setup", latency=bound.setup_ms),
        Stage("cores", after=("setup",), program=ParallelProgram(tuple(active_owners))),
    ), minimum_ms=bound.minimum_ms)
