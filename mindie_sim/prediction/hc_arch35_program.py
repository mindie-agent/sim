"""Finite arch35 HC source geometry and physical phase DAGs.

[Role] Represent mixed AIV/Cube HC pre and vector HC post on the shared Engine.
[Invariants] Cast/RMS feeds L1 directly; no GM cast ring or Cube Gram exists.
  Fixed source buffers and their reuse dependencies bound every owner. Vector
  demand is a disclosed elementary-operation envelope, not instruction timing.
  M up to the device/UB limit follows the retained source mDim/kDim and L1
  tile formulas; no M<=256 row gate is installed.
[Timeouts & Locks] Pure bounded geometry; finite source phase DAGs.
[Side-Effect Policy] Constructs immutable stages only; no files or hardware.
"""
from dataclasses import dataclass, replace
from math import ceil

from mindie_sim.schema.common import ModelingError
from .mechanism import Work
from .program import KernelProgram, ParallelProgram, Stage
from .ownership import unit_group


def align(n, unit):
    return ceil(n / unit) * unit


def memory(*, read=0, write=0, role='mte2_aiv', vector=0):
    return Work(resources={k: v for k, v in {
        'hbm': read + write, role: read, 'mte3': write, 'vector': vector}.items() if v})


@dataclass(frozen=True)
class Phase:
    name: str
    work: Work
    after: tuple = ()
    owner: int = 0
    domain: int = 1


def phase_program(phases, context):
    return KernelProgram(tuple(Stage(
        p.name, context.bind(p.work), after=p.after,
        placement=context.owned(p.owner, p.domain,
            tuple(p.work.resources.keys() - {'hbm'}) + (('cube',) if p.work.matrix else ())),
    ) for p in phases))


def total_work(groups):
    result = Work()
    for phases in groups:
        for phase in phases:
            result += phase.work
    return result


def allocations(context, label, sizes, owners, domain):
    return tuple(context.allocate(f'{label}:{pool}:{owner}', pool, amount,
                 unit_group(owner, domain, context.pool_units.get(pool, domain)))
                 for pool, amount in sizes.items() for owner in owners)


def check_capacity(geometry, sizes):
    for pool, amount in sizes.items():
        if geometry.buffers.get(pool, 0) < amount:
            raise ModelingError(f'hc_arch35_{pool}_capacity:{amount}>{geometry.buffers.get(pool, 0)}')


@dataclass(frozen=True)
class PreGeometry:
    rows: int
    hidden: int
    cores: int
    aivs: int
    k_span: int
    k_parts: int
    m_l1: int
    k_l1: int
    m_ub: int
    k_ub: int
    row_former: int
    row_factor: int
    d_factor: int
    first_ub: int
    second_ub: int
    m_dim: int = 1
    k_dim: int = 1
    single_core_m: int = 0
    m_tiles: int = 1
    k_block_num: int = 0
    tiling_key: int = 1000
    m_cnt: int = 1
    m_rounds: int = 1
    m_main: int = 0

    @property
    def width(self):
        return 4 * self.hidden


def second_ub(kparts, rows, d):
    # hc_pre_m_k_split_core_arch35.h:276-293, not the old D-split scratch.
    return (2 * rows * 4 * align(d, 16) * 2
            + 2 * (kparts * align(rows, 8) * 4 + kparts * rows * 24 * 4)
            + 2 * rows * align(d, 16) * 2 + 2 * rows * 8 * 4
            + 2 * rows * 16 * 4 + (8 + 8 + 4 * 8) * 4 + rows * 24 * 4)


def pre_geometry(view, geometry):
    rows, hidden = int(view['tokens']), int(view['hidden'])
    cores = geometry.compute_units
    aivs = geometry.resource_units.get('vector', 0)
    if (view['dtype'] != 'bf16' or view['hc'] != 4 or view.get('mixed', 24) != 24
            or rows < 1 or hidden % 32 or not 32 <= hidden <= 32768
            or cores < 2 or aivs != 2 * cores):
        raise ModelingError('hc_pre_arch35_source_geometry_invalid')
    # hc_pre_tiling_arch35.h:295-429 (retained revision 175ef4d4...):
    # M_L1_MAX_SIZE=256 is the per-core M L1 tile, never an M row limit.
    # mDimNum splits rows across AIC; each M core covers singleCoreM rows in
    # ceil(singleCoreM/mL1Size) finite L1 tiles.  kDimNum splits K; kL1Size is
    # the per-core L1 K tile.  Keep the source's two-aiv-per-aic mapping.
    k_size = 4 * hidden
    m_dim = min(cores, max(1, ceil(rows / 256)))
    single_core_m = align(ceil(rows / m_dim), 16)
    m_l1 = min(256, single_core_m)
    m_cnt = ceil(rows / m_l1)
    m_rounds = ceil(m_cnt / m_dim)
    m_main = m_cnt % m_dim
    m_tiles = m_rounds
    k_dim = max(1, cores // m_dim)
    split_k = align(ceil(k_size / k_dim), 256)
    k_block = ceil(k_size / split_k)
    kspan = split_k
    kparts = k_block
    kl1 = min((128 * 256) // m_l1, 1024) // 128 * 128
    mub, kub = ceil(m_l1 / 2), kl1 // 2
    first = (2 * mub * align(kub, 16) * 2 + 2 * align(mub, 8) * 4
             + mub * (align(kub, 8) * 4 + 32)
             + align(mub, 8) * align(kub, 8) * 4 * 2)
    former = ceil(rows / aivs)
    capacity = geometry.buffers.get('ub', 0)
    factor, d = 1, hidden
    if second_ub(kparts, factor, d) > capacity:
        divisor = 2
        while divisor <= hidden:
            d = ceil(hidden / divisor)
            if second_ub(kparts, factor, d) <= capacity:
                break
            divisor += 1
        else:
            raise ModelingError('hc_pre_arch35_stage2_ub_insufficient')
        d = d // 32 * 32 if d > 32 else d
    else:
        while factor < former and second_ub(kparts, factor + 1, d) <= capacity:
            factor += 1
    second = second_ub(kparts, factor, d)
    check_capacity(geometry, {'ub': max(first, second), 'l1': 512 * 1024,
                             'l0a': 65536, 'l0b': 65536, 'l0c': 262144})
    return PreGeometry(rows, hidden, cores, aivs, kspan, kparts, m_l1, kl1,
                       mub, kub, former, factor, d, first, second,
                       m_dim, k_dim, single_core_m, m_tiles, k_block,
                       1001 if k_dim == 1 else 1000, m_cnt, m_rounds, m_main)


def cast_rms_ops(rows, cols, *, accumulate):
    # Cast, square, source accumulator lanes, scale/reduce, optional previous
    # partial. ND->NZ movement is a separate dependent stage below.
    lanes = 64 if cols > 64 else cols
    return rows * (2 * cols + (align(cols, 64) if cols > 64 else cols)
                   + lanes + max(0, lanes - 1) + bool(accumulate))


def pre_owner(g, owner, *, msplit=False):
    """Literal source K-panel/M-tile DAG; one owner = (M core, K part).

    ``msplit`` selects the key1001 HcPreMSplitCorePart1 local path: no
    workspace projection/rms HBM stores and no global K-partial stage.

    Owner indexes are the source AIC owners; a core with singleCoreM > mL1
    executes its finite M L1 tiles serially.  MTE1 releases L1 before Cube
    finishes L0; fixed input/UB rings and tile tails are retained.
    """
    phase_list = []

    def emit(phases, tails, name, work=Work(), after=(), *, aiv=None, fifo=()):
        deps = list(after)
        for pipe in fifo:
            if pipe in tails:
                deps.append(tails[pipe])
            tails[pipe] = name
        phases.append(Phase(name, work, tuple(dict.fromkeys(deps)),
                            owner if aiv is None else 2 * owner + aiv,
                            g.cores if aiv is None else g.aivs))
        return name

    m_core, k_part = divmod(owner, g.k_parts)
    m_cnt = ceil(g.rows / g.m_l1)
    single_round = ceil(m_cnt / g.m_dim)
    main = m_cnt % g.m_dim
    if main == 0 or m_core < main:
        m_start = m_core * single_round * g.m_l1
        m_tiles = single_round
    else:
        m_start = (main * single_round + (m_core - main)
                   * (single_round - 1)) * g.m_l1
        m_tiles = single_round - 1
    k_start = k_part * g.k_span
    width = min(g.k_span, g.width - k_start)
    if width <= 0:
        return ()
    tile_tail = ()
    for tile in range(m_tiles):
        row_start = m_start + tile * g.m_l1
        rows = min(g.m_l1, g.rows - row_start)
        if rows <= 0:
            break
        m_pad = align(rows, 16)
        prefix = f'm{tile}.' if m_tiles > 1 else ''
        phases, tails = [], {}
        releases, cubes, operands = [], [], []
        ub_stores = [[], []]
        input_frees = [[], []]
        for panel, offset in enumerate(range(0, width, g.k_l1)):
            cols = min(g.k_l1, width - offset)
            reuse = (releases[panel - 2],) if panel >= 2 else ()
            b = emit(phases, tails, f'{prefix}p{panel}.weight',
                     memory(read=24 * cols * 4, role='mte2_aic'),
                     (*tile_tail, *reuse) if panel == 0 else reuse,
                     fifo=('weight_read',))
            ready_a = []
            for sub, count in enumerate((rows // 2, rows - rows // 2)):
                for chunk, start in enumerate(range(0, cols, g.k_ub)):
                    kk = min(g.k_ub, cols - start)
                    tag = f'{prefix}p{panel}.v{sub}.c{chunk}'
                    input_reuse = ((input_frees[sub][-2],)
                                   if len(input_frees[sub]) >= 2 else ())
                    load = emit(phases, tails, tag + '.read',
                                memory(read=count * kk * 2), input_reuse,
                                aiv=sub, fifo=(f'read{sub}',))
                    # Cast scratch is single-slot: the preceding transpose must
                    # consume it. Input queue can fetch two chunks ahead.
                    cast = emit(phases, tails, tag + '.cast_rms', Work(resources={
                        'vector': cast_rms_ops(
                            count, kk, accumulate=k_start + offset + start > 0)}),
                        (load,), aiv=sub, fifo=(f'vector{sub}',))
                    input_frees[sub].append(cast)
                    index = len(ub_stores[sub])
                    dep = (ub_stores[sub][index - 2],) if index >= 2 else ()
                    trans = emit(phases, tails, tag + '.transpose',
                                 Work(resources={'vector': count * kk}),
                                 (cast, *dep), aiv=sub,
                                 fifo=(f'vector{sub}',))
                    put = emit(phases, tails, tag + '.publish_l1',
                               Work(resources={'mte3': count * kk * 4}),
                               (trans, *reuse), aiv=sub, fifo=(f'write{sub}',))
                    ub_stores[sub].append(put)
                ready_a.append(ub_stores[sub][-1])
            # Source divides before multiplying: M80 uses K96, not K102.
            step = (256 // max(g.m_l1, 32)) * 32
            for micro, start in enumerate(range(0, cols, step)):
                kk = min(step, cols - start)
                deps = [b, *ready_a]
                if len(cubes) >= 2:
                    deps.append(cubes[-2])
                operand = emit(phases, tails, f'{prefix}p{panel}.k{micro}.operands',
                               Work(resources={
                                   'mte1': (m_pad + 32) * align(kk, 8) * 4}),
                               deps, fifo=('operand',))
                operands.append(operand)
                cube = emit(phases, tails, f'{prefix}p{panel}.k{micro}.cube',
                            Work(matrix={'hf32': 2 * m_pad * 32 * kk}),
                            (operand,), fifo=('cube',))
                cubes.append(cube)
            releases.append(operands[-1])
        projection = emit(phases, tails,
                          prefix + ('projection_local' if msplit else 'projection_out'),
                          Work(resources={'fixpipe': rows * 24 * 4,
                                          **({} if msplit else {'hbm': rows * 24 * 4})}),
                          (cubes[-1],))
        rms = []
        for sub, count in enumerate((rows // 2, rows - rows // 2)):
            rms.append(emit(phases, tails,
                            f'{prefix}rms_out{sub}',
                            Work() if msplit else memory(write=count * 4),
                            (tails[f'vector{sub}'],), aiv=sub,
                            fifo=(f'write{sub}',)))
        tile_tail = (projection, *rms)
        phase_list.extend(phases)
    return tuple(phase_list)


def pre_second_owner(g, owner, iterations, *, after=(), msplit=False):
    owned = min(g.row_former, g.rows - owner * g.row_former)
    phases = []
    def emit(name, work, after=()):
        phases.append(Phase(name, work, after, owner, g.aivs))
        return name
    base = emit('base', memory(read=24 * 4), after)
    previous = base
    store_tail = None
    for batch, start in enumerate(range(0, owned, g.row_factor)):
        rows = min(g.row_factor, owned - start)
        tag = f'b{batch}'
        if msplit:
            load = previous
        else:
            load = emit(tag + '.partials',
                        memory(read=rows * g.k_parts * 25 * 4), (previous,))
        # hcScale GetValue is payload only; no undocumented scalar latency.
        norm = emit(tag + '.normalize', Work(resources={'vector': rows * (
            (g.k_parts + (3 if g.k_parts >= 4 else 0)) * 25 + 27)}), (load,))
        pre_scale = emit(tag + '.pre_scale', Work(resources={'hbm': 4}), (norm,))
        prior = emit(tag + '.pre', Work(resources={'vector': rows * 4 * 7}), (pre_scale,))
        for chunk, col in enumerate(range(0, g.hidden, g.d_factor)):
            d = min(g.d_factor, g.hidden - col)
            read = emit(f'{tag}.d{chunk}.read', memory(read=rows * 4 * d * 2), (prior,))
            calc = emit(f'{tag}.d{chunk}.mix', Work(resources={
                'vector': rows * d * (4 * 3 + 1)}), (read,))
            deps = (calc,) + ((store_tail,) if store_tail else ())
            store_tail = emit(f'{tag}.d{chunk}.store', memory(write=rows * d * 2), deps)
            prior = calc
        post_scale = emit(tag + '.post_scale', Work(resources={'hbm': 4}), (prior,))
        post = emit(tag + '.post', Work(resources={'vector': rows * 4 * 7}), (post_scale,))
        post_store = emit(tag + '.post_out', memory(write=rows * 4 * 4), (post, store_tail))
        comb_scale = emit(tag + '.comb_scale', Work(resources={'hbm': 4}), (post,))
        # RowGroup reductions use two arithmetic instructions with four active
        # lanes, not an ideal three-add reduction. Register permutations and
        # issue latency remain unpriced in this elementary-operation envelope.
        softmax = 2 * 16 + 4 * (8 + 4 + 4 + 8 + 4 + 4) + 3 * 4 + 4 + 16
        iteration = 4 * (8 + 4 + 4 + 4) + 4 + 16
        sink = emit(tag + '.sinkhorn', Work(resources={'vector': rows * (
            softmax + max(0, iterations - 1) * iteration)}), (comb_scale,))
        store_tail = emit(tag + '.comb_out', memory(write=rows * 16 * 4),
                          (sink, post_store))
        previous = store_tail
    return tuple(phases)


def _msplit_rows(g, m_core):
    m_cnt = ceil(g.rows / g.m_l1)
    single_round = ceil(m_cnt / g.m_dim)
    main = m_cnt % g.m_dim
    if main == 0 or m_core < main:
        start = m_core * single_round * g.m_l1
        tiles = single_round
    else:
        start = (main * single_round + (m_core - main)
                 * (single_round - 1)) * g.m_l1
        tiles = single_round - 1
    return start, min(tiles * g.m_l1, g.rows - start)


def pre_msplit_second(g, m_core, iterations, *, after=()):
    """key1001 local vector post stage; one AIC and two AIVs, no K partials."""
    _, rows = _msplit_rows(g, m_core)
    phases = []
    for sub in range(2):
        count = rows // 2 + (1 if sub == 0 and rows % 2 else 0)
        clone = replace(g, rows=count, k_parts=1, row_former=max(1, count),
                        row_factor=min(g.row_factor, max(1, count)))
        local = pre_second_owner(clone, 0, iterations, after=after, msplit=True)
        owner = 2 * m_core + sub
        phases.extend(replace(phase, owner=owner) for phase in local)
    return tuple(phases)


def pre_msplit_groups(g, iterations):
    groups = []
    for m_core in range(g.m_dim):
        local = pre_owner(g, m_core, msplit=True)
        tails = tuple(phase.name for phase in local
                      if phase.name.endswith('projection_local')
                      or '.rms_out' in phase.name or phase.name.startswith('rms_out'))
        groups.append(tuple(local) + pre_msplit_second(
            g, m_core, iterations, after=tails))
    return tuple(groups)


def pre_program(g, first, second, context, *, compact=False):
    if g.tiling_key == 1001:
        from .program import ParallelProgram
        groups = pre_msplit_groups(g, _iterations_from_context(g, context))
        ub = max(g.first_ub, g.second_ub)
        alloc = allocations(context, 'pre_msplit',
                            {'l1': 524288, 'l0a': 65536, 'l0b': 65536,
                             'l0c': 262144}, range(g.m_dim), g.cores)
        alloc += allocations(context, 'pre_msplit', {'ub': ub},
                             range(2 * g.m_dim), g.aivs)
        return KernelProgram((Stage('source_msplit', acquire=alloc, free=alloc,
            program=ParallelProgram(tuple(phase_program(group, context)
                                          for group in groups))),))
    return _pre_program_key1000(g, first, second, context, compact=compact)


def _pre_program_key1000(g, first, second, context, *, compact=False):
    from .hc_arch35_compact import first_owner
    first_alloc = allocations(context, 'pre1', {'l1': 524288, 'l0a': 65536,
        'l0b': 65536, 'l0c': 262144}, range(g.cores), g.cores)
    first_alloc += allocations(context, 'pre1', {'ub': g.first_ub}, range(g.aivs), g.aivs)
    second_alloc = allocations(context, 'pre2', {'ub': g.second_ub}, range(g.aivs), g.aivs)
    return KernelProgram((
        Stage('source_stage1', acquire=first_alloc, free=first_alloc,
              program=ParallelProgram(tuple(first_owner(g,p,context) if compact else
                  phase_program(p, context) for p in first))),
        Stage('source_stage2', after=('source_stage1',), acquire=second_alloc, free=second_alloc,
              program=ParallelProgram(tuple(phase_program(p, context) for p in second))),
    ))


def _iterations_from_context(g, context):
    return int(dict(getattr(context, "metadata", ()) or ()).get(
        "sinkhorn_iters", getattr(g, "sinkhorn_iters", 20)))




def post_geometry(view, geometry):
    if (view['dtype'] != 'bf16' or view['hc'] != 4
            or view.get('coefficient_dtype', 'fp32') != 'fp32'
            or view['hidden'] % 32 or not 32 <= view['hidden'] <= 32768
            or view['tokens'] < 1):
        raise ModelingError('hc_post_arch35_source_geometry_invalid')
    # Double X/residual/output queues + FP32 sum scratch. Source reserves full
    # D4096 even for a short last chunk and casts the full output scratch.
    size = 2 * 4096 * 2 + 2 * 4 * 4096 * 2 + 2 * 4 * 4 + 2 * 16 * 4
    size += 2 * 4 * 4096 * 2 + 4 * 4096 * 4
    check_capacity(geometry, {'ub': size})
    return size


def post_owner(view, owner, rows, units):
    phases = []
    last_compute, last_store = None, None
    for row in range(rows):
        name = f'r{row}'
        deps = (last_compute,) if last_compute else ()
        phases.append(Phase(name + '.coeff', memory(read=80), deps, owner, units))
        prior = name + '.coeff'
        for chunk, col in enumerate(range(0, view['hidden'], 4096)):
            d = min(4096, view['hidden'] - col)
            tag = f'{name}.d{chunk}'
            phases.append(Phase(tag + '.read', memory(read=10 * d), (prior,), owner, units))
            phases.append(Phase(tag + '.mix', Work(resources={'vector': 56 * d}),
                                (tag + '.read',), owner, units))
            phases.append(Phase(tag + '.cast', Work(resources={'vector': 4 * 4096}),
                                (tag + '.mix',), owner, units))
            last_compute = prior = tag + '.cast'
            deps = (prior,) + ((last_store,) if last_store else ())
            last_store = tag + '.write'
            phases.append(Phase(last_store, memory(write=8 * d), deps, owner, units))
    return tuple(phases)
