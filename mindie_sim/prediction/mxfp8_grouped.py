"""Explicit AIC-only MXFP8 NZ grouped matrix source program.

[Role] Model CGMCT's rolling bank scheduler, MX copies and direct GM Fixpipe.
[Invariants] Integer rows define work, not tensor capacity. M/N remain actual
  MMAD dimensions; LoadData uses aligned operands and paired E8M0 scales.
  Source tiling is geometry/declared input, never inferred from observed time.
[Timeouts & Locks] Lazy per-owner execution; O(output tiles) plan, no I/O/locks.
[Side-Effect Policy] Submit ordinary Engine work; no device or calibration I/O.
"""
from dataclasses import dataclass
from math import ceil
from itertools import groupby
from collections import Counter

from mindie_sim.schema.common import ModelingError
from .matrix_cycle import matrix_cycle
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill import RefillPipeline

SOURCE = {
    'repository': 'https://gitcode.com/cann/ops-transformer',
    'revision': 'a3925fd49cd256ad478d2faf983bf2961b8ae2c8',
    'branch': 'MX NZ FP8, CGMCT (!IS_BLAZE), nontransposed A/B, no bias',
    'paths': [
        'gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp',
        'gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp',
        'gmm/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_mx_kernel.h',
        'gmm/common/cgmct/kernel/kernel_qgmm_mx.h',
        'gmm/common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h',
        'gmm/common/cgmct/block/block_mmad_mx.h'],
    'capture_binary_identity': 'not_established',
}


def align(value, granule):
    return ceil(value / granule) * granule


@dataclass(frozen=True)
class Mapping:
    m: int
    n: int
    k: int
    l1_k: int
    l0c_slots: int
    scale_k: int | None = None

    @property
    def buffers(self):
        # The source uses separate data and scale sections inside each L1 half.
        scale_k = 2 * ceil((self.scale_k or self.l1_k) / 64)
        return {'l1': 2 * (self.m + self.n) * (self.l1_k + scale_k),
                'l0a': 2 * self.m * self.k,
                'l0b': 2 * self.n * self.k,
                'l0c': self.l0c_slots * self.m * self.n * 4}


def mapping(m, n, k, geometry, l1_k):
    # CalBasicBlock: nontransposed FP8 A/B. CalStepKs caps full-K refills.
    bm, bn, bk = align(min(max(m, 1), 256), 16), align(min(n, 256), 128), align(min(k, 128), 64)
    if l1_k is None:
        # This full-K domain is common to the retained two-buffer host rules.
        # Larger/asymmetric schedules need an explicit host tiling declaration.
        if k > 256:
            raise ModelingError('mxfp8_grouped_larger_k_requires_declared_l1_k')
        l1_k = align(k, bk)
    if type(l1_k) is not int or l1_k <= 0 or l1_k % bk:
        raise ModelingError('mxfp8_grouped_l1_k_requires_multiple_of_base_k')
    l1_k = min(l1_k, align(k, bk))
    db = 2 if 2 * bm * bn * 4 <= geometry.buffers.get('l0c', 0) else 1
    tile = Mapping(bm, bn, bk, l1_k, db, min(k, l1_k))
    for pool, size in tile.buffers.items():
        if size > geometry.buffers.get(pool, 0):
            raise ModelingError('mxfp8_grouped_source_buffer_capacity:' + pool)
    return tile


def owner_tiles(rows, n, tile, cores):
    """ASWT with per-bank BaseMBalance and final declared bank tail split.

    Return bank, valid M/N and logical offsets, retaining each source owner.
    The final split redistributes only the last partial physical core wave.
    """
    owners = [[] for _ in range(cores)]
    cursor = 0
    for bank, m in enumerate(rows):
        if not m:
            continue
        bm = align(ceil(m / ceil(m / tile.m)), 16)
        nm, nn = ceil(m / bm), ceil(n / tile.n)
        main = min(4, nm)
        final_window = nm // main - 1
        ordered = []
        for window in range(final_window + 1):
            start = window * main
            end = nm if window == final_window else start + main
            for j in range(nn):
                ni = nn - 1 - j if window % 2 else j
                for mi in range(start, end):
                    ordered.append((bank, min(bm, m - mi * bm), min(tile.n, n - ni * tile.n),
                                    mi * bm, ni * tile.n))
        start = cursor % cores
        end = (start + len(ordered) - 1) % cores
        tail = min(end + 1, len(ordered))
        mt = nt = 1
        if bank == len(rows) - 1 and end + 1 <= cores // 2:
            remain = (cores - end - 1) // tail + 1
            mt = min(ceil((m - (nm - 1) * bm) / 16), remain)
            nt = min(ceil((n - (nn - 1) * tile.n) / 128), remain)
            while mt * nt > remain:
                if mt >= nt:
                    mt -= 1
                else:
                    nt -= 1
        for index, item in enumerate(ordered):
            owner = (start + index) % cores
            if mt * nt == 1 or index < len(ordered) - tail:
                owners[owner].append(item)
                continue
            bank, mm, width, mo, no = item
            base = end + 1 - tail + (index - len(ordered) + tail) * mt * nt
            sm, sn = ceil(mm / mt), align(ceil(width / nt), 128)
            for split in range(mt * nt):
                target = base + split
                # GetBlockShape uses the physical block index for the split.
                si, sj = (target % (mt * nt)) % mt, (target % (mt * nt)) // mt
                if si * sm < mm and sj * sn < width:
                    owners[target].append((bank, min(sm, mm - si * sm), min(sn, width - sj * sn),
                                           mo + si * sm, no + sj * sn))
        cursor += len(ordered)
    return tuple(tuple(items) for items in owners)


def panel_work(m, n, kd, bk):
    """Physical NZ reads, MX scale-pair reads, and aligned L1-to-L0 service."""
    # ND A reads only valid elements; NZ B has physical N32/K16 padding.
    scale = 2 * ceil(kd / 64)
    reads = m * kd + align(n, 32) * align(kd, 16) + (m + n) * scale
    load = Work(resources={'hbm': reads, 'mte2_aic': reads})
    steps = []
    for kk in range(0, kd, bk):
        # MX MMAD K and operand representation are padded to 64, unlike M/N.
        mk = align(min(bk, kd - kk), 64)
        # B uses transposed 2D loads: N32 payload, while the MX side
        # scale path uses N16. Odd N16 tails issue several data-only loads.
        operand = (align(m, 16) + align(n, 32)) * mk
        operand += (align(m, 16) + align(n, 16)) * 2 * ceil(mk / 64)
        steps.append((Work(resources={'mte1': operand}), Work({'fp8_e4m3': 2 * m * n * mk})))
    return load, tuple(steps)


def build(op, geometry, *, l1_k=None, compact=True, compact_outputs=True):
    md = dict(op.metadata)
    rows = tuple(md.get('matrix_group_rows', ()))
    total, n, k = op.local_shape[-3:]
    if (not rows or any(type(v) is not int or v < 0 for v in rows)
            or sum(rows) != total or min(n, k) <= 0):
        raise ModelingError('mxfp8_grouped_requires_integer_rows')
    if any(md.get(key, op.dtype) != 'fp8_e4m3' for key in ('input_format', 'weight_format', 'compute_format')):
        raise ModelingError('mxfp8_grouped_requires_e4m3_operands')
    if (md.get('output_format') != 'bf16' or md.get('activation_group_size') != 32
            or md.get('activation_scale_format') != 'fp8_e8m0'
            or md.get('weight_representation') != 'fp8_block1x32_fp8_e8m0'
            or md.get('dequant_before_compute') or md.get('matrix_epilogue')):
        raise ModelingError('mxfp8_grouped_requires_mx32_bf16_direct_output')
    cores = geometry.compute_units
    for role in ('mte1', 'mte2_aic', 'fixpipe'):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError('mxfp8_grouped_requires_private_aic:' + role)
    tile = mapping(total, n, k, geometry, l1_k)
    for pool in tile.buffers:
        if geometry.buffer_units.get(pool) != cores:
            raise ModelingError('mxfp8_grouped_requires_aic_buffers:' + pool)
    owners = owner_tiles(rows, n, tile, cores)
    scalar = Work(resources={'hbm': len(rows) * 8})
    total_work = scalar.scaled(cores)
    shapes = Counter((m, width) for items in owners for _, m, width, _, _ in items)
    for (m, width), occurrences in shapes.items():
        work = Work(resources={'hbm': 2 * m * width, 'fixpipe': 2 * m * width})
        for start in range(0, k, tile.l1_k):
            load, steps = panel_work(m, width, min(tile.l1_k, k - start), tile.k)
            work += load
            for a, b in steps:
                work += a + b
        total_work += work.scaled(occurrences)

    def program(context):
        active_owners = sum(bool(items) for items in owners)

        def one(owner, items):
            def stage(name, work, **kwargs):
                roles = set(work.resources) - {'hbm'} | ({'cube'} if work.matrix else set())
                return Stage(name, context.bind(work), placement=context.owned(owner, cores, roles), **kwargs)

            def duration(s, share_memory=False):
                return max((v / (context.rates[key] * (len(s.placement[key]) / cores
                    if key in s.placement else 1) / (active_owners if share_memory and
                    key == context.resource_names['hbm'] else 1))
                    for key, v in s.work.items()), default=0)

            runs = tuple((key, sum(1 for _ in group))
                         for key, group in groupby(items, key=lambda value: (value[1], value[2])))

            def repeated_output(index):
                # Same-owner source output tiles are serial in this abstraction.
                # Collapse only a consecutive run with identical physical work.
                (m, width), repeat = runs[index]
                count, tail = divmod(k, tile.l1_k)
                def parts(kd):
                    load, steps = panel_work(m, width, kd, tile.k)
                    return stage('load', load), tuple((stage('operand', a),
                        stage('matrix', b, after=('operand',))) for a, b in steps)
                full, end = parts(tile.l1_k) if count else None, parts(tail) if tail else None
                load = (full or end)[0]
                write = stage('store', Work(resources={'hbm': 2*m*width, 'fixpipe': 2*m*width}))
                times = lambda piece: tuple((duration(a), duration(b)) for a, b in piece[1])
                cycle = matrix_cycle(duration(full[0], True) if full else 0,
                    times(full or end), count, 2, 2,
                    tail_load_ms=duration(end[0], True) if end else None,
                    tail_micro_steps=times(end) if end else None)
                work, placement = {}, {}
                for nr, piece in ((count, full), (int(bool(tail)), end)):
                    if piece:
                        for s in (piece[0], *(v for pair in piece[1] for v in pair)):
                            for key, value in s.work.items():
                                work[key] = work.get(key, 0) + nr * repeat * value
                            placement.update(s.placement)
                for key, value in write.work.items():
                    work[key] = work.get(key, 0) + repeat * value
                placement.update(write.placement)
                for s in (load, write):
                    for key, value in s.work.items():
                        work[key] -= value
                work = {key: value for key, value in work.items() if value > 0}
                span = repeat * (cycle.span_ms + duration(write, True))
                span -= duration(load, True) + duration(write, True)
                interior = Stage('steady', work, placement=placement)
                span = max(span, duration(interior))
                return (load,
                    Stage('steady', work, after=('load',), placement=placement, coupled=True,
                          caps={key: value/span for key, value in work.items()} if span else {}),
                    Stage('store', write.work, after=('steady',), placement=write.placement))

            def output(index):
                _, m, width, _, _ = items[index]
                count, tail = divmod(k, tile.l1_k)

                def parts(kd):
                    load, steps = panel_work(m, width, kd, tile.k)
                    return stage('load', load), tuple((stage('operand', a),
                        stage('matrix', b, after=('operand',))) for a, b in steps)

                write = stage('store', Work(resources={'hbm': 2*m*width, 'fixpipe': 2*m*width}))
                if not compact:
                    def panel(index):
                        load, steps = parts(tail if tail and index == count else tile.l1_k)
                        return load, iter((s, i + 1 == len(steps)) for i, s in enumerate(steps)), ()
                    stages = (Stage('K', program=RefillPipeline(count + bool(tail), 2, 2, panel)),
                              Stage('store', write.work, after=('K',), placement=write.placement))
                else:
                    full, end = parts(tile.l1_k) if count else None, parts(tail) if tail else None
                    first, last = (full or end)[0], (end or full)[1][-1][1]
                    cycle = matrix_cycle(duration(full[0]) if full else 0,
                        tuple((duration(a), duration(b)) for a, b in (full or end)[1]), count, 2, 2,
                        tail_load_ms=duration(end[0]) if end else None,
                        tail_micro_steps=tuple((duration(a), duration(b)) for a, b in end[1]) if end else None)
                    work, placement = {}, {}
                    for repeat, piece in ((count, full), (int(bool(tail)), end)):
                        if piece:
                            for s in (piece[0], *(v for pair in piece[1] for v in pair)):
                                for key, value in s.work.items():
                                    work[key] = work.get(key, 0) + repeat * value
                                placement.update(s.placement)
                    for s in (first, last):
                        for key, value in s.work.items():
                            work[key] -= value
                    work = {key: value for key, value in work.items() if value > 0}
                    span = max(0, cycle.span_ms - duration(first) - duration(last))
                    stages = (first,
                        Stage('steady', work, after=('load',), placement=placement, coupled=True,
                              caps={key: value / span for key, value in work.items()} if span else {}),
                        Stage('matrix', last.work, after=('steady',), placement=last.placement),
                        Stage('store', write.work, after=('matrix',), placement=write.placement))
                return (Stage('tile', program=KernelProgram(stages)),)

            allocation = tuple(context.allocate(f'owner{owner}:{pool}', pool, size, (owner,))
                               for pool, size in tile.buffers.items()) if items else ()
            return KernelProgram((stage('metadata', scalar, acquire=allocation),
                Stage('outputs', after=('metadata',), program=PipelineProgram(
                    len(runs) if compact and compact_outputs else len(items), 1,
                    repeated_output if compact and compact_outputs else output)),
                Stage('release', after=('outputs',), free=allocation)))
        return ParallelProgram(tuple(one(owner, items) for owner, items in enumerate(owners)))

    return OperatorPlan(total_work, program, {
        'execution_class': 'cube', 'source_program': SOURCE,
        'source_mapping': {'base_m': tile.m, 'base_n': tile.n, 'base_k': tile.k,
            'equal_refill_k': tile.l1_k, 'scale_refill_k': tile.scale_k, 'l1_refill_basis': 'declared' if l1_k else 'host_full_k_le_256',
            'output_tiles_by_owner': tuple(len(v) for v in owners), 'group_rows': rows,
            'buffers_per_owner': tile.buffers},
        'execution_abstraction': ('source_same_owner_tile_runs_compact' if compact_outputs else
                                  'source_K_compact') if compact else 'source_K_detailed',
        'source_boundary': 'conditional_kernel',
        'approximation': 'serial_output_tiles;compact_external_contention_fluid;all_reads_offchip',
        'unmodeled_source_features': ['descriptor_and_flag_instruction_latency',
            'ND_to_NZ_transaction_efficiency', 'K_tail_L1_init_instruction_service',
            'small_M_mmad_issue_cycles',
            'cross_output_tile_prefetch', 'capture_binary_tiling_identity'],
    })


def recipe(*, l1_k=None, compact=True, compact_outputs=True):
    return PlannedRecipe(lambda op, geometry: build(op, geometry, l1_k=l1_k, compact=compact,
                                                   compact_outputs=compact_outputs))
