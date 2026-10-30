"""Shared physical ownership for mixed matrix/vector source programs.

[Role] Bind source AIC/AIV streams and resident storage to physical units.
[Invariants] Two vector subblocks share one matrix core's L1, not its UB.
  Global memory remains shared. Declared copy-engine partitions are respected.
[Timeouts & Locks] Pure bounded metadata construction.
[Side-Effect Policy] Creates programs; no runtime, source or observation I/O.
"""
from ..program import KernelProgram, Stage
from ..ownership import read_role, unit_group
from ..mechanism import Work
from ..instructions import REGISTER_WORK_ROLES


def split_reads(work, aic_bytes, roles):
    """Partition an aggregate read ledger by the source instruction owner."""
    if read_role(roles) == "mte2":
        return work
    resources = dict(work.resources)
    total = resources.pop("mte2", 0)
    if not 0 <= aic_bytes <= total:
        raise ValueError("mixed_read_partition_exceeds_total")
    resources.update(mte2_aic=aic_bytes, mte2_aiv=total - aic_bytes)
    return Work(work.matrix, resources)


def placement(context, core, aiv=None):
    if aiv is None:
        return context.owned(core, context.core_count,
                             ("cube", "mte1", "fixpipe", read_role(context.resource_names), "mte3"))
    return context.owned(2 * core + aiv, 2 * context.core_count,
                         ("vector", "ub_copy", read_role(context.resource_names, aiv), "mte3")
                         + REGISTER_WORK_ROLES)


def resident(context, core, sizes, program):
    allocations = tuple(context.allocate(
        pool, pool, size, unit_group(
            core, context.core_count, context.pool_units.get(pool, context.core_count)))
        for pool, size in sizes.items() if pool in context.pool_names)
    return KernelProgram((
        Stage("reserve", acquire=allocations),
        Stage("body", after=("reserve",), program=program),
        Stage("release", after=("body",), free=allocations),
    ))


def phase_stages(name, work, context, core, *, aiv=None, after=(), previous=(),
                 l0_after=(), compute_after=(), publish_after=(), store_after=(),
                 store_previous=()):
    """Expose consumer completion separately from a later asynchronous store."""
    resources = dict(work.resources)
    read, write = resources.pop("mte2", 0), resources.pop("mte3", 0)
    memory = resources.pop("hbm", 0)
    l0 = resources.pop("mte1", 0)
    fix = resources.pop("fixpipe", 0)
    # mte3 may include direct UB->L1, which never consumes global memory.
    gm_read = min(read, memory)
    gm_write = memory - gm_read
    owned = placement(context, core, aiv)
    prefix = name + ":" if name else ""
    stages = [
        Stage(prefix + "load", context.bind(Work(resources={
            "hbm": gm_read, read_role(context.resource_names, aiv): read})), after=after,
              previous=previous, placement=owned),
    ]
    stages.append(Stage(prefix + "l0", context.bind(Work(resources={"mte1": l0})),
                        after=(prefix + "load",) + l0_after, placement=placement(context, core)))
    stages.append(Stage(prefix + "compute", context.bind(Work(work.matrix, resources)),
                        after=(prefix + "l0",) + compute_after, placement=owned))
    if fix:
        fix_gm = gm_write if not write else 0
        gm_write -= fix_gm
        stages.append(Stage(prefix + "publish",
                            context.bind(Work(resources={"fixpipe": fix, "hbm": fix_gm})),
                            after=(prefix + "compute",) + publish_after,
                            placement=placement(context, core)))
    stages.append(Stage(prefix + "store",
                        context.bind(Work(resources={"hbm": gm_write, "mte3": write})),
                        after=(prefix + ("publish" if fix else "compute"),) + store_after,
                        previous=store_previous, placement=owned))
    if name:
        stages.append(Stage(name, after=(prefix + "store",)))
    return tuple(stages)


def phases(name, work, context, core, *, aiv=None, after=()):
    """Separate GM input, arithmetic and output publication within one stage."""
    body = KernelProgram(phase_stages("", work, context, core, aiv=aiv))
    return Stage(name, after=after, program=body)


def require_pair(geometry):
    if geometry.resource_units.get("vector", geometry.compute_units) != 2 * geometry.compute_units:
        raise ValueError("mixed_source_requires_two_vector_subblocks_per_matrix_core")
