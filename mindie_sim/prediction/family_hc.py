"""Source-selected hyper-connection work on the common resource engine.

[Role] Price the inspected M/K-split HcPre and D-split HcPost implementations.
[Invariants] Gram diagonal and projection share one loaded X operand. Global
  cast/partial workspaces retain HBM service; no unproved L2 hit is assumed.
  Source choice is explicit and never inferred from a captured duration.
[Timeouts & Locks] Finite arithmetic and bounded per-owner pipelines; no I/O.
[Side-Effect Policy] Returns plans only; no observations or device contexts.
"""
from collections import defaultdict
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError, dtype_bytes

from .mechanism import OperatorPlan, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


PRE_PROFILE = "hc_pre_m_k_split_175ef4d4"
POST_PROFILE = "hc_post_d_split_175ef4d4"
SOURCE_REVISION = "175ef4d49c2072f9233619636f0309f4f88b4e2c"
SOURCE = {
    "revision": SOURCE_REVISION,
    "pre_cube": {
        "path": "csrc/moe/hc_pre/op_kernel/hc_pre_cube_compute.h",
        "sha256": "9056b512f88569025bd3cec598a66d5461b5fd4c14bf6a7eb6c963ff6537ed69",
        "lines": "234-251,267-341,418-478",
    },
    "pre_tiling": {
        "path": "csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp",
        "sha256": "25bde622e9b6cc0635f0bd5802ad77868f3a69d54c9b0d29bedfb0aa7fb7d35b",
        "lines": "48-52,146-149,290-319",
    },
    "pre_owners": {
        "path": "csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h",
        "sha256": "f14eeb0888a27f80a17708fd6f5e57ec0a217176cfe97da512b6bac7218e7fac",
        "lines": "82-99,105-108,135-140,148-164",
    },
    "post": {
        "path": "csrc/moe/hc_post/op_kernel/hc_post_d_split.h",
        "sha256": "9b7a4a6e1d0fde01632813589566f173ac6561522d0c0cf62463589004f2d58a",
        "lines": "139-158,299-396",
    },
}


def _ceil(value, divisor):
    return (value + divisor - 1) // divisor


def _align(value, multiple):
    return _ceil(value, multiple) * multiple


def _memory(*, read=0, write=0, vector=0, matrix=None, read_role="mte2",
            write_role="mte3"):
    resources = {}
    if read:
        resources[read_role] = read
    if write:
        resources[write_role] = write
    if vector:
        resources["vector"] = vector
    if read + write:
        resources["hbm"] = read + write
    return Work(dict(matrix or {}), resources)


def _source_read_role(geometry, *, vector):
    """Resolve the declared private read pipe; generic-only devices stay usable."""
    try:
        return read_role(geometry.resource_units, aiv=0 if vector else None)
    except ValueError as exc:
        raise ModelingError(str(exc)) from exc


def _sum(items):
    result = Work()
    for item in items:
        result += item
    return result


def _view(op):
    view = dict(op.metadata)["fusion_tensor_view"]
    if (view["dtype"] not in {"bf16", "fp16", "fp32"}
            or min(view["tokens"], view["hidden"], view["hc"]) < 1):
        raise ModelingError("hc_source_tensor_domain_not_supported")
    return view


@dataclass(frozen=True)
class PreGeometry:
    rows: int
    width: int
    mixed: int
    m_partitions: int
    k_partitions: int
    m_tile_count: int
    padded_rows: int
    m_tile: int
    cast_ring_bytes: int
    aiv_units: int
    row_former: int
    owner_count: int
    m_span: int
    k_span: int


def _pre_owner_groups(rows, aiv):
    """Host ceil-division row ownership, grouped by equal tail row counts."""
    row_former = _ceil(rows, aiv)
    owner_count = _ceil(rows, row_former)
    tail = rows - (owner_count - 1) * row_former
    if owner_count == 1 or tail == row_former:
        return row_former, owner_count, ((row_former, tuple(range(owner_count))),)
    return row_former, owner_count, (
        (row_former, tuple(range(owner_count - 1))),
        (tail, (owner_count - 1,)),
    )


def pre_geometry(view, geometry):
    """The source fixes N=24 and uses 256-row / 256-K-aligned partitions."""
    rows, width = view["tokens"], view["hc"] * view["hidden"]
    if view["dtype"] != "bf16" or view["hc"] != 4 or width % 32:
        raise ModelingError("hc_pre_source_requires_bf16_hc4_and_k_multiple32")
    cores = geometry.compute_units
    aiv = geometry.resource_units.get("vector", cores)
    row_former, owner_count, _ = _pre_owner_groups(rows, aiv)
    m_partitions = min(cores, _ceil(rows, 256))
    m_span = _align(_ceil(rows, m_partitions), 16)
    k_span = _align(_ceil(width, cores // m_partitions), 256)
    k_partitions = _ceil(width, k_span)
    row_spans = [min(m_span, rows - index * m_span)
                 for index in range(m_partitions) if index * m_span < rows]
    return PreGeometry(
        rows, width, 24, len(row_spans), k_partitions,
        sum(_ceil(span, 256) for span in row_spans),
        sum(_align(span, 16) for span in row_spans),
        min(256, m_span), cores * 2 * min(256, m_span) * 1024 * 4,
        aiv, row_former, owner_count, m_span, k_span,
    )


def _pre_coefficient_work(view, g, *, rows, owners, read_role):
    """One source row-former owner reads its own coefficient/base payload."""
    hc = view["hc"]
    reductions = rows * (g.k_partitions - 1) * (g.mixed + 16)
    inverse_affine = reductions + rows * (4 + 3 * g.mixed)
    pre_post_gate = rows * 2 * hc * 5
    sinkhorn = rows * (4 * hc * hc * (view["sinkhorn_iters"] + 1) - 2 * hc)
    partials = rows * g.k_partitions * (g.mixed + 16) * 4
    return _memory(
        read=owners * g.mixed * 4 + partials + rows * 3 * 4,
        vector=inverse_affine + pre_post_gate + sinkhorn,
        read_role=read_role,
    )


def _pre_mix_work(view, g, *, rows, read_role):
    hc, item_bytes = view["hc"], dtype_bytes(view["dtype"])
    mix = rows * view["hidden"] * (2 * hc - 1)
    if item_bytes < 4:
        mix += rows * view["hidden"] * (hc + 1)
    return _memory(
        read=rows * g.width * item_bytes,
        write=rows * view["hidden"] * item_bytes + rows * (hc + hc * hc) * 4,
        vector=mix,
        read_role=read_role,
    )


def pre_phase_work(view, geometry):
    """Logical transferred payloads; workspace row strides are not payload."""
    g = pre_geometry(view, geometry)
    rows, width, hc = g.rows, g.width, view["hc"]
    item_bytes = dtype_bytes(view["dtype"])
    vector_read = _source_read_role(geometry, vector=True)
    projection_read = _source_read_role(geometry, vector=False)
    projection_bytes = rows * g.k_partitions * g.mixed * 4
    gram_bytes = rows * g.k_partitions * 16 * 4
    # MmadA2 computes each 16x16 Gram block, including off-diagonal products.
    # MmadAB reuses that same L1 X. MmadAB enables HF32; MmadA2 does not.
    matrix = {
        "hf32": 2 * g.padded_rows * width * g.mixed,
        "fp32": 2 * g.padded_rows * width * 16,
    }
    # Part2 host ceil division owns rowFormer rows per used AIV; a tail owner
    # owns fewer. Parameter/base reads are per owner, not per token.
    coefficient = _pre_coefficient_work(
        view, g, rows=rows, owners=g.owner_count, read_role=vector_read)
    mix = _pre_mix_work(view, g, rows=rows, read_role=vector_read)
    return {
        "cast": _memory(
            read=rows * width * item_bytes, write=rows * width * 4,
            vector=rows * width if item_bytes < 4 else 0,
            read_role=vector_read),
        "projection_and_gram": _memory(
            read=rows * width * 4 + g.m_tile_count * width * g.mixed * 4,
            write=projection_bytes + gram_bytes, matrix=matrix,
            read_role=projection_read, write_role="fixpipe"),
        "coefficients": coefficient,
        "mix_and_outputs": mix,
    }


def _owner_placement(context, owners, roles, *, domain=None):
    """Map source row owners to their own physical units, never an aggregate."""
    if domain is None:
        domain = context.resource_units.get("vector", context.core_count)
    placement = {}
    for role in roles:
        resource = context.physical_role(role)
        name = context.resource_names.get(resource)
        if name is None:
            raise ModelingError(f"hc_source_requires_read_service:{role}")
        units = context.resource_units.get(resource, context.core_count)
        assigned = set()
        for owner in owners:
            assigned.update(unit_group(owner, domain, units))
        placement.setdefault(name, set()).update(assigned)
    return {name: tuple(sorted(units)) for name, units in placement.items()}


def _owner_groups(context, groups, roles, domain, *, pools=()):
    """Collapse equal owners only when their physical partitions are disjoint."""
    shared = any(context.resource_units.get(context.physical_role(role),
                                             context.core_count) < domain
                 for role in roles)
    shared |= any(context.pool_units.get(pool, domain) < domain for pool in pools)
    if not shared:
        return groups
    return tuple((rows, (owner,)) for rows, owners in groups for owner in owners)


def _pre_stage1_owners(view, geometry):
    """AIC M/K partitions, with paired AIVs splitting each M panel's rows."""
    g = pre_geometry(view, geometry)
    if g.aiv_units != 2 * geometry.compute_units:
        raise ModelingError("hc_pre_source_requires_two_aiv_per_aic")
    vector_read = _source_read_role(geometry, vector=True)
    cube_read = _source_read_role(geometry, vector=False)
    cast, projection = [], []
    for m in range(g.m_partitions):
        rows = min(g.m_span, g.rows - m * g.m_span)
        full, tail = divmod(rows, g.m_tile)
        row_counts = (full * _ceil(g.m_tile, 2) + _ceil(tail, 2),
                      full * (g.m_tile // 2) + tail // 2)
        for k in range(g.k_partitions):
            width = min(g.k_span, g.width - k * g.k_span)
            owner = m * g.k_partitions + k
            for subcore, count in enumerate(row_counts):
                if count:
                    elements = count * width
                    cast.append((2 * owner + subcore, _memory(
                        read=2 * elements, write=4 * elements,
                        vector=elements, read_role=vector_read)))
            projection.append((owner, _memory(
                read=rows * width * 4 + _ceil(rows, 256) * width * 24 * 4,
                write=rows * 40 * 4,
                matrix={"hf32": 2 * _align(rows, 16) * width * 24,
                        "fp32": 2 * _align(rows, 16) * width * 16},
                read_role=cube_read, write_role="fixpipe")))
    return cast, projection


def _pre_plan(op, geometry):
    metadata = dict(op.metadata)
    exposed = set(metadata["fusion_external_outputs"])
    for identifier, phase in zip(metadata["fusion_semantic_ids"], metadata["fusion_phases"]):
        if identifier in exposed and phase["recipe"] not in {"hc_sinkhorn", "hc_pre_mix"}:
            raise ModelingError("hc_pre_source_cannot_export_internal_intermediate")
    if "fixpipe" not in geometry.resource_units:
        raise ModelingError("hc_pre_source_requires_fixpipe_output")
    view = _view(op)
    phases = pre_phase_work(view, geometry)
    g = pre_geometry(view, geometry)
    vector_read = _source_read_role(geometry, vector=True)
    cube_read = _source_read_role(geometry, vector=False)
    cast_owners, projection_owners = _pre_stage1_owners(view, geometry)

    def group_stage(context, work, name, owners, roles):
        return Stage(name, context.bind(work),
                     placement=_owner_placement(context, owners, roles))

    def program(context):
        def owner_phase(items, name, roles, domain):
            return ParallelProgram(tuple(KernelProgram((Stage(
                f"{name}_owner{owner}", context.bind(work),
                placement=_owner_placement(context, (owner,), roles, domain=domain),
            ),)) for owner, work in items))

        stages = [
            Stage("cast", program=owner_phase(
                cast_owners, "cast", ("vector", vector_read, "mte3"), g.aiv_units)),
            Stage("projection_and_gram",
                  after=("cast",),
                  program=owner_phase(
                      projection_owners, "projection",
                      ("cube", cube_read, "fixpipe"), geometry.compute_units)),
        ]
        coefficient_groups = []
        mix_groups = []
        _, _, ownership = _pre_owner_groups(g.rows, g.aiv_units)
        ownership = _owner_groups(
            context, ownership, ("vector", vector_read, "mte3"), g.aiv_units)
        for index, (owned_rows, owners) in enumerate(ownership):
            rows = owned_rows * len(owners)
            coefficient = _pre_coefficient_work(
                view, g, rows=rows, owners=len(owners), read_role=vector_read)
            mix = _pre_mix_work(view, g, rows=rows, read_role=vector_read)
            coefficient_groups.append(KernelProgram((
                group_stage(context, coefficient, f"coefficients{index}",
                            owners, ("vector", vector_read)),)))
            mix_groups.append(KernelProgram((
                group_stage(context, mix, f"mix{index}", owners,
                            ("vector", vector_read, "mte3")),)))
        stages.append(Stage("coefficients", after=("projection_and_gram",),
                            program=ParallelProgram(tuple(coefficient_groups))))
        stages.append(Stage("mix_and_outputs", after=("coefficients",),
                            program=ParallelProgram(tuple(mix_groups))))
        return KernelProgram(tuple(stages))

    return OperatorPlan(_sum(phases.values()), program)


def post_chunk_work(view, cols, *, first, read_role="mte2",
                    write_role="mte3"):
    """One source D-loop, with coefficient reads once per token."""
    hc, size = view["hc"], dtype_bytes(view["dtype"])
    coefficient_size = dtype_bytes(view.get("coefficient_dtype", "fp32"))
    coefficients = (hc + hc * hc) * coefficient_size if first else 0
    return (
        ("coefficients", _memory(read=coefficients, read_role=read_role)),
        ("load_x", _memory(read=cols * size, read_role=read_role)),
        ("x_cast_and_post", _memory(vector=cols * (hc + (size < 4)))),
        ("load_residual", _memory(read=cols * hc * size, read_role=read_role)),
        ("residual_cast_mix", _memory(
            vector=cols * (2 * hc * hc + hc * (size < 4)))),
        ("cast_output", _memory(vector=cols * hc * (size < 4))),
        ("store", _memory(write=cols * hc * size, write_role=write_role)),
    )


def post_work(view, *, read_role="mte2", write_role="mte3"):
    rows, width = view["tokens"], view["hidden"]
    return _sum(work for _, work in post_chunk_work(
        view, width, first=True, read_role=read_role,
        write_role=write_role)).scaled(rows)


def post_buffer_bytes(view, *, tile=2048):
    """Explicit source UB reservations for one vector owner, fixed D tile."""
    hc, item, coefficient = (view["hc"], dtype_bytes(view["dtype"]),
                             dtype_bytes(view.get("coefficient_dtype", "fp32")))
    coefficient_align = _align(hc, 32 // coefficient)
    input_queue = 2 * hc * tile * item
    output_queue = 2 * hc * tile * item
    casts = hc * tile * 4 * 2 if item == 2 else 0
    coefficient_casts = (hc * 4 + hc * coefficient_align * 4
                         if coefficient == 2 else 0)
    return (input_queue + output_queue + casts + coefficient_casts
            + 2 * hc * coefficient + 2 * hc * coefficient_align * coefficient
            + 2 * 64 * 4 + hc * tile * 4)


def _post_plan(op, geometry):
    view = _view(op)
    if view.get("coefficient_dtype", "fp32") != "fp32":
        raise ModelingError("hc_post_source_plan_requires_fp32_coefficients")
    if view["hidden"] % 64 or view["hc"] > 8:
        raise ModelingError("hc_post_source_requires_d_multiple64_and_hc_at_most8")
    rows, width = view["tokens"], view["hidden"]
    units = geometry.resource_units.get("vector", geometry.compute_units)
    used = min(rows, units)
    rows_per_core, remainder = divmod(rows, used)
    ownership = defaultdict(list)
    for unit in range(used):
        ownership[rows_per_core + (unit < remainder)].append(unit)
    chunks = _ceil(width, 2048)
    read = _source_read_role(geometry, vector=True)
    ub = geometry.buffers.get("ub")
    buffer_size = post_buffer_bytes(view)
    if ub is None:
        raise ModelingError("hc_post_source_requires_ub_capacity")
    if buffer_size > ub:
        raise ModelingError(
            f"hc_post_source_ub_capacity_insufficient:{buffer_size}>{ub}")

    def program(context):
        programs = []
        groups = _owner_groups(context, tuple(ownership.items()),
                               ("vector", read, "mte3"), units, pools=("ub",))
        for owned_rows, owners in groups:
            placement = _owner_placement(context, owners, ("vector", read, "mte3"))
            allocations = tuple(
                context.allocate(
                    f"hc_post_source_ub:{owner}", "ub", buffer_size,
                    unit_group(owner, units,
                               context.pool_units.get("ub", units)))
                for owner in owners)

            def tile(index, owners=tuple(owners), placement=placement,
                     allocations=allocations, total=owned_rows * chunks):
                chunk = index % chunks
                cols = min(2048, width - chunk * 2048)
                stages = []
                for name, work in post_chunk_work(
                        view, cols, first=chunk == 0, read_role=read):
                    stages.append(Stage(
                        name, context.bind(work.scaled(len(owners))),
                        after=(stages[-1].name,) if stages else (),
                        previous=("cast_output",) if not stages else (),
                        placement=placement,
                        acquire=allocations if index == 0 and not stages else (),
                        free=allocations if index == total - 1 and name == "store" else ()))
                return tuple(stages)

            # Source owns two input/output queue slots. Next input may start
            # after output conversion; its store can still be in flight.
            programs.append(PipelineProgram(owned_rows * chunks, 2, tile))
        return ParallelProgram(tuple(programs))

    return OperatorPlan(post_work(view, read_role=read), program)


def selected_plan(op, geometry):
    """Return only an explicitly selected source implementation's plan."""
    metadata = dict(op.metadata)
    profile = metadata.get("fusion_boundary", {}).get("traffic_profile")
    if profile == PRE_PROFILE:
        return _pre_plan(op, geometry)
    if profile == POST_PROFILE:
        return _post_plan(op, geometry)
    return None
