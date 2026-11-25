"""Finite register-based HC source geometry and physical phase DAGs.

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
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage
from mindie_sim.prediction.ownership import unit_group


HC_STREAMS = 4
_COMB_COEFFS = HC_STREAMS ** 2
MIXED_COEFFS = 2 * HC_STREAMS + _COMB_COEFFS
PARTIAL_VALUES_PER_ROW = MIXED_COEFFS + 1  # Includes the RMS partial.
FP32_BYTES = 4
_BF16_ALIGN_ELEMS = 16
_FP32_ALIGN_ELEMS = 8
_HIDDEN_ALIGNMENT = 32
_MAX_HIDDEN = 32768
MAX_M_L1 = 256
_K_SPLIT_ALIGNMENT = 256
_K_L1_ALIGNMENT = 128
_MAX_K_L1 = 1024
_K_MICRO_ALIGNMENT = 32
_CUBE_N = 32
_STAGE1_CAST_SCRATCH_BYTES_PER_ROW = 32
_RMS_ACCUMULATOR_LANES = 64
MK_SPLIT_TILING_KEY = 1000
M_SPLIT_TILING_KEY = 1001
_ASCENDC_SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024
_PRE_CUBE_BUFFERS = {'l1': 512 * 1024, 'l0a': 65536, 'l0b': 65536, 'l0c': 262144}
_K_PARTIAL_EXTRA_REDUCTIONS = 3
_K_PARTIAL_REDUCTION_THRESHOLD = 4
_NORMALIZE_OPS_PER_ROW = 27
_GATE_OPS_PER_COEFF = 7
_MIX_OPS_PER_STREAM = 3
_DEFAULT_SINKHORN_ITERS = 20
# RowGroup reductions issue two arithmetic instructions across four lanes.
_SINKHORN_INITIAL_OPS = 2 * 16 + 4 * (8 + 4 + 4 + 8 + 4 + 4) + 3 * 4 + 4 + 16
_SINKHORN_ITERATION_OPS = 4 * (8 + 4 + 4 + 4) + 4 + 16
POST_D_TILE = 4096
_POST_COEFF_BYTES = (HC_STREAMS + _COMB_COEFFS) * FP32_BYTES
_POST_INPUT_BYTES_PER_COLUMN = (1 + HC_STREAMS) * 2
_POST_MIX_OPS_PER_COLUMN = 56
_POST_OUTPUT_BYTES_PER_COLUMN = HC_STREAMS * 2


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
    tiling_key: int = MK_SPLIT_TILING_KEY
    m_cnt: int = 1
    m_rounds: int = 1
    m_main: int = 0

    @property
    def width(self):
        return HC_STREAMS * self.hidden


def second_ub(kparts, rows, d):
    # hc_pre_m_k_split_core_arch35.h:276-293, not the old D-split scratch.
    rms_partials = kparts * align(rows, _FP32_ALIGN_ELEMS)
    projection_partials = kparts * rows * MIXED_COEFFS
    scale_scratch = (2 + HC_STREAMS) * _FP32_ALIGN_ELEMS
    return (2 * rows * HC_STREAMS * align(d, _BF16_ALIGN_ELEMS) * 2
            + 2 * (rms_partials + projection_partials) * FP32_BYTES
            + 2 * rows * align(d, _BF16_ALIGN_ELEMS) * 2
            + 2 * rows * _FP32_ALIGN_ELEMS * FP32_BYTES
            + 2 * rows * _COMB_COEFFS * FP32_BYTES
            + scale_scratch * FP32_BYTES + rows * MIXED_COEFFS * FP32_BYTES)


def pre_geometry(view, geometry):
    rows, hidden = int(view['tokens']), int(view['hidden'])
    cores = geometry.compute_units
    aivs = geometry.resource_units.get('vector', 0)
    if (view['dtype'] != 'bf16' or view['hc'] != HC_STREAMS or view.get('mixed', MIXED_COEFFS) != MIXED_COEFFS
            or rows < 1 or hidden % _HIDDEN_ALIGNMENT or not _HIDDEN_ALIGNMENT <= hidden <= _MAX_HIDDEN
            or cores < 2 or aivs != 2 * cores):
        raise ModelingError('hc_pre_arch35_source_geometry_invalid')
    # hc_pre_tiling_arch35.h:295-429 (retained revision 175ef4d4...):
    # M_L1_MAX_SIZE=256 is the per-core M L1 tile, never an M row limit.
    # mDimNum splits rows across AIC; each M core covers singleCoreM rows in
    # ceil(singleCoreM/mL1Size) finite L1 tiles.  kDimNum splits K; kL1Size is
    # the per-core L1 K tile.  Keep the source's two-aiv-per-aic mapping.
    k_size = HC_STREAMS * hidden
    m_dim = min(cores, max(1, ceil(rows / MAX_M_L1)))
    single_core_m = align(ceil(rows / m_dim), _BF16_ALIGN_ELEMS)
    m_l1 = min(MAX_M_L1, single_core_m)
    m_cnt = ceil(rows / m_l1)
    m_rounds = ceil(m_cnt / m_dim)
    m_main = m_cnt % m_dim
    m_tiles = m_rounds
    k_dim = max(1, cores // m_dim)
    split_k = align(ceil(k_size / k_dim), _K_SPLIT_ALIGNMENT)
    k_block = ceil(k_size / split_k)
    kspan = split_k
    kparts = k_block
    kl1 = min((_K_L1_ALIGNMENT * MAX_M_L1) // m_l1, _MAX_K_L1) // _K_L1_ALIGNMENT * _K_L1_ALIGNMENT
    mub, kub = ceil(m_l1 / 2), kl1 // 2
    first = (2 * mub * align(kub, _BF16_ALIGN_ELEMS) * 2 + 2 * align(mub, _FP32_ALIGN_ELEMS) * FP32_BYTES
             + mub * (align(kub, _FP32_ALIGN_ELEMS) * FP32_BYTES + _STAGE1_CAST_SCRATCH_BYTES_PER_ROW)
             + align(mub, _FP32_ALIGN_ELEMS) * align(kub, _FP32_ALIGN_ELEMS) * FP32_BYTES * 2)
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
        d = d // _HIDDEN_ALIGNMENT * _HIDDEN_ALIGNMENT if d > _HIDDEN_ALIGNMENT else d
    else:
        while factor < former and second_ub(kparts, factor + 1, d) <= capacity:
            factor += 1
    second = second_ub(kparts, factor, d)
    check_capacity(geometry, {'ub': max(first, second), **_PRE_CUBE_BUFFERS})
    return PreGeometry(rows, hidden, cores, aivs, kspan, kparts, m_l1, kl1,
                       mub, kub, former, factor, d, first, second,
                       m_dim, k_dim, single_core_m, m_tiles, k_block,
                       M_SPLIT_TILING_KEY if k_dim == 1 else MK_SPLIT_TILING_KEY, m_cnt, m_rounds, m_main)


def cast_rms_ops(rows, cols, *, accumulate):
    # Cast, square, source accumulator lanes, scale/reduce, optional previous
    # partial. ND->NZ movement is a separate dependent stage below.
    lanes = _RMS_ACCUMULATOR_LANES if cols > _RMS_ACCUMULATOR_LANES else cols
    return rows * (2 * cols + (align(cols, _RMS_ACCUMULATOR_LANES) if cols > _RMS_ACCUMULATOR_LANES else cols)
                   + lanes + max(0, lanes - 1) + bool(accumulate))


def pre_owner(g, owner, *, msplit=False, round_deps=None):
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
        m_pad = align(rows, _BF16_ALIGN_ELEMS)
        # Key1001 assigns ceil(rows/2) to even AIV (sub=0) to match local Stage2 UB;
        # Key1000 assigns floor(rows/2) to even AIV (hc_pre_m_k_split_core_arch35.h:123).
        row_split = ((rows + 1) // 2, rows // 2) if msplit else (rows // 2, rows - rows // 2)
        prefix = f'm{tile}.' if m_tiles > 1 else ''
        phases, tails = [], {}
        releases, cubes, operands = [], [], []
        ub_stores = [[], []]
        input_frees = [[], []]
        prev_round = round_deps[tile - 1] if (round_deps and tile > 0) else (None, None)
        for panel, offset in enumerate(range(0, width, g.k_l1)):
            cols = min(g.k_l1, width - offset)
            reuse = (releases[panel - 2],) if panel >= 2 else ()
            b = emit(phases, tails, f'{prefix}p{panel}.weight',
                     memory(read=MIXED_COEFFS * cols * FP32_BYTES, role='mte2_aic'),
                     (*tile_tail, *reuse) if panel == 0 else reuse,
                     fifo=('weight_read',))
            ready_a = []
            for sub, count in enumerate(row_split):
                for chunk, start in enumerate(range(0, cols, g.k_ub)):
                    kk = min(g.k_ub, cols - start)
                    tag = f'{prefix}p{panel}.v{sub}.c{chunk}'
                    input_reuse = ((input_frees[sub][-2],)
                                   if len(input_frees[sub]) >= 2 else ())
                    if panel == 0 and chunk == 0 and prev_round[sub] is not None:
                        input_reuse = (*input_reuse, prev_round[sub])
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
                               Work(resources={'mte3': count * kk * FP32_BYTES}),
                               (trans, *reuse), aiv=sub, fifo=(f'write{sub}',))
                    ub_stores[sub].append(put)
                ready_a.append(ub_stores[sub][-1])
            # Source divides before multiplying: M80 uses K96, not K102.
            step = (MAX_M_L1 // max(g.m_l1, _K_MICRO_ALIGNMENT)) * _K_MICRO_ALIGNMENT
            for micro, start in enumerate(range(0, cols, step)):
                kk = min(step, cols - start)
                deps = [b, *ready_a]
                if len(cubes) >= 2:
                    deps.append(cubes[-2])
                operand = emit(phases, tails, f'{prefix}p{panel}.k{micro}.operands',
                               Work(resources={
                                   'mte1': (m_pad + _CUBE_N) * align(kk, _FP32_ALIGN_ELEMS) * FP32_BYTES}),
                               deps, fifo=('operand',))
                operands.append(operand)
                cube = emit(phases, tails, f'{prefix}p{panel}.k{micro}.cube',
                            Work(matrix={'hf32': 2 * m_pad * _CUBE_N * kk}),
                            (operand,), fifo=('cube',))
                cubes.append(cube)
            releases.append(operands[-1])
        projection = emit(phases, tails,
                          prefix + ('projection_local' if msplit else 'projection_out'),
                          Work(resources={'fixpipe': rows * MIXED_COEFFS * FP32_BYTES,
                                          **({} if msplit else {'hbm': rows * MIXED_COEFFS * FP32_BYTES})}),
                          (cubes[-1],))
        rms = []
        for sub, count in enumerate(row_split):
            rms.append(emit(phases, tails,
                            f'{prefix}rms_out{sub}',
                            Work() if msplit else memory(write=count * FP32_BYTES),
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
    base = emit('base', memory(read=MIXED_COEFFS * FP32_BYTES), after)
    previous = base
    store_tail = None
    for batch, start in enumerate(range(0, owned, g.row_factor)):
        rows = min(g.row_factor, owned - start)
        tag = f'b{batch}'
        if msplit:
            load = previous
        else:
            load = emit(tag + '.partials',
                        memory(read=rows * g.k_parts * PARTIAL_VALUES_PER_ROW * FP32_BYTES), (previous,))
        # hcScale GetValue is payload only; no undocumented scalar latency.
        extra_reductions = (_K_PARTIAL_EXTRA_REDUCTIONS
                            if g.k_parts >= _K_PARTIAL_REDUCTION_THRESHOLD else 0)
        normalize_ops = (g.k_parts + extra_reductions) * PARTIAL_VALUES_PER_ROW + _NORMALIZE_OPS_PER_ROW
        norm = emit(tag + '.normalize', Work(resources={'vector': rows * normalize_ops}), (load,))
        pre_scale = emit(tag + '.pre_scale', Work(resources={'hbm': FP32_BYTES}), (norm,))
        prior = emit(tag + '.pre', Work(resources={'vector': rows * HC_STREAMS * _GATE_OPS_PER_COEFF}), (pre_scale,))
        for chunk, col in enumerate(range(0, g.hidden, g.d_factor)):
            d = min(g.d_factor, g.hidden - col)
            read = emit(f'{tag}.d{chunk}.read', memory(read=rows * HC_STREAMS * d * 2), (prior,))
            calc = emit(f'{tag}.d{chunk}.mix', Work(resources={
                'vector': rows * d * (HC_STREAMS * _MIX_OPS_PER_STREAM + 1)}), (read,))
            deps = (calc,) + ((store_tail,) if store_tail else ())
            store_tail = emit(f'{tag}.d{chunk}.store', memory(write=rows * d * 2), deps)
            prior = calc
        post_scale = emit(tag + '.post_scale', Work(resources={'hbm': FP32_BYTES}), (prior,))
        post = emit(tag + '.post', Work(resources={'vector': rows * HC_STREAMS * _GATE_OPS_PER_COEFF}), (post_scale,))
        post_store = emit(tag + '.post_out', memory(write=rows * HC_STREAMS * FP32_BYTES), (post, store_tail))
        comb_scale = emit(tag + '.comb_scale', Work(resources={'hbm': FP32_BYTES}), (post,))
        # RowGroup reductions use two arithmetic instructions with four active
        # lanes, not an ideal three-add reduction. Register permutations and
        # issue latency remain unpriced in this elementary-operation envelope.
        sink = emit(tag + '.sinkhorn', Work(resources={'vector': rows * (
            _SINKHORN_INITIAL_OPS + max(0, iterations - 1) * _SINKHORN_ITERATION_OPS)}), (comb_scale,))
        store_tail = emit(tag + '.comb_out', memory(write=rows * _COMB_COEFFS * FP32_BYTES),
                          (sink, post_store))
        previous = store_tail
    return tuple(phases)


def pre_workspace_breakdown(g):
    """Return ``(payload_bytes, reserved_bytes, total_bytes)`` matching upstream GetWorkspaceSize."""
    if g.tiling_key != MK_SPLIT_TILING_KEY:
        return 0, 0, 0
    payload = g.rows * g.k_parts * PARTIAL_VALUES_PER_ROW * FP32_BYTES
    reserved = _ASCENDC_SYSTEM_WORKSPACE_BYTES
    return payload, reserved, payload + reserved


def _msplit_tile_rows(g, m_core):
    m_cnt = ceil(g.rows / g.m_l1)
    single_round = ceil(m_cnt / g.m_dim)
    main = m_cnt % g.m_dim
    if main == 0 or m_core < main:
        start = m_core * single_round * g.m_l1
        tiles = single_round
    else:
        start = (main * single_round + (m_core - main) * (single_round - 1)) * g.m_l1
        tiles = single_round - 1
    result = []
    for tile in range(tiles):
        rows = min(g.m_l1, g.rows - (start + tile * g.m_l1))
        if rows <= 0:
            break
        result.append(rows)
    return tuple(result)


def _msplit_second_with_tails(g, m_core, iterations, *, after=()):
    """Build per-round key1001 Stage2 phases plus per-round terminal phase names per AIV."""
    tile_rows = _msplit_tile_rows(g, m_core)
    m_tiles = len(tile_rows)
    phases = []
    round_tails = [[None, None] for _ in range(m_tiles)]
    for sub in range(2):
        owner = 2 * m_core + sub
        first_tp = 'm0.' if m_tiles > 1 else ''
        first_deps = tuple(after) if after else (f'{first_tp}projection_local', f'{first_tp}rms_out{sub}')
        base_name = f'v{sub}.base'
        phases.append(Phase(base_name, memory(read=MIXED_COEFFS * FP32_BYTES), first_deps, owner, g.aivs))
        prev_tail = base_name
        for tile, rows in enumerate(tile_rows):
            count = (rows + 1) // 2 if sub == 0 else rows // 2
            if count <= 0:
                continue
            tp = f'm{tile}.' if m_tiles > 1 else ''
            entry_deps = (prev_tail,) if tile == 0 else (
                prev_tail, f'{tp}projection_local', f'{tp}rms_out{sub}',
            )
            clone = replace(g, rows=count, k_parts=1, row_former=count,
                            row_factor=min(g.row_factor, count))
            local = pre_second_owner(clone, 0, iterations, after=(), msplit=True)
            prefix = f'v{sub}.m{tile}.' if m_tiles > 1 else f'v{sub}.'
            for phase in local[1:]:
                mapped_after = []
                for dep in phase.after:
                    if dep == 'base':
                        mapped_after.extend(entry_deps)
                    else:
                        mapped_after.append(prefix + dep)
                phases.append(replace(
                    phase,
                    name=prefix + phase.name,
                    after=tuple(dict.fromkeys(mapped_after)),
                    owner=owner,
                ))
            prev_tail = phases[-1].name
            round_tails[tile][sub] = prev_tail
    return tuple(phases), tuple((t[0], t[1]) for t in round_tails)


def pre_msplit_second(g, m_core, iterations, *, after=()):
    """key1001 local vector post stage; one AIC and two AIVs, per M-tile round."""
    phases, _ = _msplit_second_with_tails(g, m_core, iterations, after=after)
    return phases


def pre_msplit_parts(g, iterations):
    """Key1001 per-round Stage1 and local Stage2 DAGs preserving private UB lifetimes."""
    parts = []
    for m_core in range(g.m_dim):
        second, round_tails = _msplit_second_with_tails(g, m_core, iterations)
        local = pre_owner(g, m_core, msplit=True, round_deps=round_tails)
        parts.append((tuple(local), second))
    return tuple(parts)


def _order_msplit_phases(stage1, stage2):
    """Interleave multi-M-tile key1001 rounds in topological execution order."""
    if not any(p.name.startswith('m0.') for p in stage1):
        return stage1 + stage2
    s1_by_tile = {}
    for p in stage1:
        tile = int(p.name.split('.', 1)[0][1:])
        s1_by_tile.setdefault(tile, []).append(p)
    bases = []
    s2_by_tile = {}
    for p in stage2:
        parts = p.name.split('.')
        if len(parts) == 2 and parts[1] == 'base':
            bases.append(p)
        else:
            tile = int(parts[1][1:])
            s2_by_tile.setdefault(tile, []).append(p)
    ordered = []
    for tile in sorted(s1_by_tile):
        ordered.extend(s1_by_tile[tile])
        if tile == 0:
            ordered.extend(bases)
        ordered.extend(s2_by_tile.get(tile, ()))
    return tuple(ordered)


def pre_msplit_groups(g, iterations):
    return tuple(_order_msplit_phases(first, second) for first, second in pre_msplit_parts(g, iterations))


def pre_program(g, first, second, context, *, compact=False, parts=None):
    if g.tiling_key == M_SPLIT_TILING_KEY:
        from mindie_sim.prediction.program import ParallelProgram
        from .native_fusion_hc_compact import first_owner
        iterations = _iterations_from_context(g, context)
        parts = parts or pre_msplit_parts(g, iterations)
        ub = max(g.first_ub, g.second_ub)
        alloc = allocations(context, 'pre_msplit',
                            _PRE_CUBE_BUFFERS, range(g.m_dim), g.cores)
        alloc += allocations(context, 'pre_msplit', {'ub': ub},
                             range(2 * g.m_dim), g.aivs)
        def group_program(stage1, stage2):
            if not compact or any(p.name.startswith('m0.') for p in stage1):
                return phase_program(_order_msplit_phases(stage1, stage2), context)
            owner = first_owner(g, stage1, context, terminal='projection_local')
            stages = owner.stages + phase_program(stage2, context).stages
            return KernelProgram(stages)
        return KernelProgram((Stage('source_msplit', acquire=alloc, free=alloc,
            program=ParallelProgram(tuple(group_program(stage1, stage2)
                                          for stage1, stage2 in parts))),))
    return _pre_program_key1000(g, first, second, context, compact=compact)


def _pre_program_key1000(g, first, second, context, *, compact=False):
    from mindie_sim.prediction.program import PipelineTemplateProgram
    from .native_fusion_hc_compact import (
        finite_owner_template, finite_template_available, first_owner,
    )
    first_alloc = allocations(context, 'pre1', _PRE_CUBE_BUFFERS, range(g.cores), g.cores)
    first_alloc += allocations(context, 'pre1', {'ub': g.first_ub}, range(g.aivs), g.aivs)
    second_alloc = allocations(context, 'pre2', {'ub': g.second_ub}, range(g.aivs), g.aivs)
    _, _, ws_bytes = pre_workspace_breakdown(g)
    ws_alloc = ()
    if ws_bytes > 0 and 'workspace' in context.pool_names:
        ws_alloc = (context.allocate(
            'pre:workspace', 'workspace', ws_bytes,
            unit_group(0, g.cores, context.pool_units.get('workspace', 1)),
        ),)
    finite = compact and g.m_tiles == 1 and finite_template_available()
    if finite:
        # Same source Phase classes, now repeated by the existing ordered
        # PipelineTemplate instead of materializing every logical panel.
        owners=tuple(PipelineTemplateProgram(
            finite_owner_template(g,p,context)) for p in first)
    else:
        owners=tuple(first_owner(g,p,context) if compact else
                     phase_program(p, context) for p in first)
    stage1_after = ('workspace_acquire',) if ws_alloc else ()
    stages = []
    if ws_alloc:
        stages.append(Stage('workspace_acquire', acquire=ws_alloc))
    stages.extend((
        Stage('source_stage1', after=stage1_after, acquire=first_alloc, free=first_alloc,
              program=ParallelProgram(owners)),
        Stage('source_stage2', after=('source_stage1',), acquire=second_alloc, free=second_alloc,
              program=ParallelProgram(tuple(phase_program(p, context) for p in second))),
    ))
    if ws_alloc:
        stages.append(Stage('workspace_release', after=('source_stage2',), free=ws_alloc))
    return KernelProgram(tuple(stages))


def _iterations_from_context(g, context):
    return int(dict(getattr(context, "metadata", ()) or ()).get(
        "sinkhorn_iters", getattr(g, "sinkhorn_iters", _DEFAULT_SINKHORN_ITERS)))




def post_geometry(view, geometry):
    if (view['dtype'] != 'bf16' or view['hc'] != HC_STREAMS
            or view.get('coefficient_dtype', 'fp32') != 'fp32'
            or view['hidden'] % _HIDDEN_ALIGNMENT or not _HIDDEN_ALIGNMENT <= view['hidden'] <= _MAX_HIDDEN
            or view['tokens'] < 1):
        raise ModelingError('hc_post_arch35_source_geometry_invalid')
    # Double X/residual/output queues + FP32 sum scratch. Source reserves full
    # D4096 even for a short last chunk and casts the full output scratch.
    input_bytes = POST_D_TILE * _POST_INPUT_BYTES_PER_COLUMN
    output_bytes = POST_D_TILE * _POST_OUTPUT_BYTES_PER_COLUMN
    size = 2 * (input_bytes + output_bytes + _POST_COEFF_BYTES) + HC_STREAMS * POST_D_TILE * FP32_BYTES
    check_capacity(geometry, {'ub': size})
    return size


def post_owner(view, owner, rows, units):
    phases = []
    last_compute, last_store = None, None
    for row in range(rows):
        name = f'r{row}'
        deps = (last_compute,) if last_compute else ()
        phases.append(Phase(name + '.coeff', memory(read=_POST_COEFF_BYTES), deps, owner, units))
        prior = name + '.coeff'
        for chunk, col in enumerate(range(0, view['hidden'], POST_D_TILE)):
            d = min(POST_D_TILE, view['hidden'] - col)
            tag = f'{name}.d{chunk}'
            phases.append(Phase(tag + '.read', memory(read=_POST_INPUT_BYTES_PER_COLUMN * d), (prior,), owner, units))
            phases.append(Phase(tag + '.mix', Work(resources={'vector': _POST_MIX_OPS_PER_COLUMN * d}),
                                (tag + '.read',), owner, units))
            phases.append(Phase(tag + '.cast', Work(resources={'vector': HC_STREAMS * POST_D_TILE}),
                                (tag + '.mix',), owner, units))
            last_compute = prior = tag + '.cast'
            deps = (prior,) + ((last_store,) if last_store else ())
            last_store = tag + '.write'
            phases.append(Phase(last_store, memory(write=_POST_OUTPUT_BYTES_PER_COLUMN * d), deps, owner, units))
    return tuple(phases)
