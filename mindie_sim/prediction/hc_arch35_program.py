"""Finite arch35 HC source geometry and physical phase DAGs.

[Role] Represent mixed AIV/Cube HC pre and vector HC post on the shared Engine.
[Invariants] Cast/RMS feeds L1 directly; no GM cast ring or Cube Gram exists.
  Fixed source buffers and their reuse dependencies bound every owner. Vector
  demand is a disclosed elementary-operation envelope, not instruction timing.
[Timeouts & Locks] Pure bounded geometry; decode source domain has M <= 256.
[Side-Effect Policy] Constructs immutable stages only; no files or hardware.
"""
from dataclasses import dataclass
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
            or not 1 <= rows <= 256 or hidden % 32 or not 32 <= hidden <= 32768
            or cores < 2 or aivs != 2 * cores):
        raise ModelingError('hc_pre_arch35_requires_bf16_hc4_M1to256_Kmultiple128_two_aiv_per_aic')
    # M <= 256 keeps host mDimNum=1 and kDimNum=cores, hence key1000.
    # actualKBlockNum can still be one for a small hidden dimension.
    kspan = align(ceil(4 * hidden / cores), 256)
    kparts = ceil(4 * hidden / kspan)
    ml1 = align(rows, 16)
    kl1 = min(32768 // ml1, 1024) // 128 * 128
    mub, kub = ceil(ml1 / 2), kl1 // 2
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
    return PreGeometry(rows, hidden, cores, aivs, kspan, kparts, ml1, kl1,
                       mub, kub, former, factor, d, first, second)


def cast_rms_ops(rows, cols, *, accumulate):
    # Cast, square, source accumulator lanes, scale/reduce, optional previous
    # partial. ND->NZ movement is a separate dependent stage below.
    lanes = 64 if cols > 64 else cols
    return rows * (2 * cols + (align(cols, 64) if cols > 64 else cols)
                   + lanes + max(0, lanes - 1) + bool(accumulate))


def pre_owner(g, owner):
    """Literal finite K panel DAG; MTE1 releases L1 before Cube finishes L0."""
    phases, tails = [], {}
    def emit(name, work=Work(), after=(), *, aiv=None, fifo=()):
        deps = list(after)
        for pipe in fifo:
            if pipe in tails:
                deps.append(tails[pipe])
            tails[pipe] = name
        phases.append(Phase(name, work, tuple(dict.fromkeys(deps)),
                            owner if aiv is None else 2 * owner + aiv,
                            g.cores if aiv is None else g.aivs))
        return name
    width = min(g.k_span, g.width - owner * g.k_span)
    releases, cubes, operands = [], [], []
    ub_stores = [[], []]
    input_frees = [[], []]
    for panel, offset in enumerate(range(0, width, g.k_l1)):
        cols = min(g.k_l1, width - offset)
        reuse = (releases[panel - 2],) if panel >= 2 else ()
        b = emit(f'p{panel}.weight', memory(read=24 * cols * 4, role='mte2_aic'),
                 reuse, fifo=('weight_read',))
        ready_a = []
        for sub, count in enumerate((g.rows // 2, g.rows - g.rows // 2)):
            for chunk, start in enumerate(range(0, cols, g.k_ub)):
                kk = min(g.k_ub, cols - start)
                prefix = f'p{panel}.v{sub}.c{chunk}'
                input_reuse = ((input_frees[sub][-2],) if len(input_frees[sub]) >= 2 else ())
                load = emit(prefix + '.read', memory(read=count * kk * 2), input_reuse,
                            aiv=sub, fifo=(f'read{sub}',))
                # Cast scratch is single-slot: the preceding transpose must
                # consume it. Input queue can fetch two chunks ahead.
                cast = emit(prefix + '.cast_rms', Work(resources={
                    'vector': cast_rms_ops(count, kk, accumulate=offset + start > 0)}),
                    (load,), aiv=sub, fifo=(f'vector{sub}',))
                input_frees[sub].append(cast)
                index = len(ub_stores[sub])
                dep = (ub_stores[sub][index - 2],) if index >= 2 else ()
                trans = emit(prefix + '.transpose', Work(resources={'vector': count * kk}), (cast, *dep),
                             aiv=sub, fifo=(f'vector{sub}',))
                put = emit(prefix + '.publish_l1', Work(resources={'mte3': count * kk * 4}),
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
            operand = emit(f'p{panel}.k{micro}.operands', Work(resources={
                'mte1': (g.m_l1 + 32) * align(kk, 8) * 4}),
                deps, fifo=('operand',))
            operands.append(operand)
            cube = emit(f'p{panel}.k{micro}.cube', Work(matrix={
                'hf32': 2 * g.m_l1 * 32 * kk}), (operand,), fifo=('cube',))
            cubes.append(cube)
        releases.append(operands[-1])
    emit('projection_out', Work(resources={'fixpipe': g.rows * 24 * 4,
         'hbm': g.rows * 24 * 4}), (cubes[-1],))
    for sub, count in enumerate((g.rows // 2, g.rows - g.rows // 2)):
        emit(f'rms_out{sub}', memory(write=count * 4),
             (tails[f'vector{sub}'],), aiv=sub, fifo=(f'write{sub}',))
    return tuple(phases)


def pre_second_owner(g, owner, iterations):
    owned = min(g.row_former, g.rows - owner * g.row_former)
    phases = []
    def emit(name, work, after=()):
        phases.append(Phase(name, work, after, owner, g.aivs))
        return name
    base = emit('base', memory(read=24 * 4))
    previous = base
    store_tail = None
    for batch, start in enumerate(range(0, owned, g.row_factor)):
        rows = min(g.row_factor, owned - start)
        tag = f'b{batch}'
        load = emit(tag + '.partials', memory(read=rows * g.k_parts * 25 * 4), (previous,))
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


def pre_program(g, first, second, context, *, compact=False):
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


def post_geometry(view, geometry):
    if (view['dtype'] != 'bf16' or view['hc'] != 4
            or view.get('coefficient_dtype', 'fp32') != 'fp32'
            or view['hidden'] % 32 or not 32 <= view['hidden'] <= 32768
            or not 1 <= view['tokens'] <= 256):
        raise ModelingError('hc_post_arch35_requires_bf16_fp32coeff_hc4_M1to256_D32to32768')
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
