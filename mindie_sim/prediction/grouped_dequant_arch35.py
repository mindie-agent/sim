"""Finite arch35 full-load INT32 dequant/SwiGLU/INT8 source program.

[Role] Keep allocated host geometry distinct from valid expert row counts.
[Invariants] Host capacity selects the group/core branch, never extra tensor
  work. Owned groups retain scale reads and finite UB queue dependencies. The
  common Engine prices typed registers and explicitly declared scalar service;
  missing scalar rates remain visible as unpriced source counts.
[Timeouts & Locks] O(experts * AIVs) schedules; two live UB tiles per owner.
[Side-Effect Policy] Pure source plans, without measurements, I/O or rates.
"""
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .instructions import InstructionClass, InstructionCount, InstructionInventory, PrimitiveInventory
from .mechanism import OperatorPlan, Work
from .ownership import unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

SOURCE = {
    'repository': 'https://gitcode.com/cann/ops-nn',
    'revision': '955f8b33985b6d72989ebff341bb9c2c564a51bc',
    'host': 'quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp:756-768,939-1022',
    'host_sha256': '0ff6b530a54192ff335c38abe10c019abc32ae8b96560be981de56b74738e278',
    'kernel': 'quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h:196-223,240-843',
    'kernel_sha256': '3bf3fedad839383cee0e644dd3e07c0313e7abe9246092eb4ba3b1c3e997add0',
    'domain': 'INT32_FP32_row_column_scales_INT8_no_bias_no_smooth_swiglu0_full_load',
    'identity': 'retained_source_conditional_not_capture_binary_dispatch_proof',
}

CLASSES = {
    'Mask': InstructionClass(1), 'Load': InstructionClass(1, 1, ub_read=True),
    'LoadBRC': InstructionClass(1, 1, ub_read=True),
    'Store': InstructionClass(2, ub_write=True),
    **{name: InstructionClass(1, 1, 'reg_alu_lanes') for name in (
        'Cast32', 'Cast64', 'Mul', 'Muls', 'Adds', 'Abs', 'Max', 'Duplicate')},
    'Exp': InstructionClass(1, 1, 'reg_special_lanes'),
    'Div': InstructionClass(1, 1, 'reg_special_lanes'),
    'ReduceMax': InstructionClass(2, 1, 'reg_reduce_lanes'),
}


def ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class Config:
    rows: tuple[int, ...]
    width: int
    capacity: int
    aivs: int
    ub: int

    def __post_init__(self):
        if (not self.rows or any(type(n) is not int or n < 0 for n in self.rows)
                or type(self.width) is not int or not 1 <= self.width <= 5120
                or type(self.capacity) is not int or self.capacity < max(1, sum(self.rows))
                or self.aivs < 1):
            raise ModelingError('arch35_dequant_requires_integral_rows_capacity_and_full_load_width')
        if self.tile_rows < 1:
            raise ModelingError('arch35_dequant_full_load_does_not_fit_ub')

    @property
    def aligned_width(self):
        return ceildiv(self.width, 8) * 8

    @property
    def row_storage(self):
        # Two x/row-scale slots, two INT8 slots, one scale and one FP32 tmp.
        w = self.aligned_width
        return 2 * (2*w*4 + 32) + 2*ceildiv(self.width, 32)*32 + 32 + 4*w

    @property
    def tile_rows(self):
        return min(self.capacity, (self.ub - 1024 - 12*self.aligned_width) // self.row_storage)

    @property
    def special(self):
        # group_index is the caller's one-dimensional INT64 counts tensor.
        return len(self.rows) >= 32 and self.capacity // len(self.rows) <= 16

    @property
    def blocks(self):
        return min(self.aivs, self.capacity if self.special else ceildiv(self.capacity, self.tile_rows))

    @property
    def storage(self):
        return 12*self.aligned_width + self.tile_rows*self.row_storage


def assignments(cfg, owner):
    """Actual source groups/rows; no expert payload is read for a zero bank."""
    result = []
    cursor = owner
    for group, rows in enumerate(cfg.rows):
        if cfg.special:
            if group != cursor:
                continue
            # Pinned Process() continues before advancing cuGroupIdx on zero.
            # Reject the resulting source domain hole rather than losing work.
            if not rows:
                if any(cfg.rows[g] for g in range(group + cfg.blocks, len(cfg.rows), cfg.blocks)):
                    raise ModelingError('arch35_dequant_zero_group_stalls_later_source_owner')
                continue
            count = rows
            cursor += cfg.blocks
        else:
            count = max(0, min(ceildiv(rows, cfg.blocks), rows - owner*ceildiv(rows, cfg.blocks)))
        if count:
            result.append((group, count))
    return tuple(result)


@lru_cache(maxsize=256)
def inventories(rows, width):
    """Three source vector scopes, with exact masks/calls and UB payloads."""
    blocks = ceildiv(width, 64)
    def c(calls=0, lanes=0, read=0, write=0):
        return InstructionCount(calls, lanes, read, write)
    dequant = InstructionInventory({
        'Mask': c(blocks),
        'Load': c(2*blocks + 2*rows*blocks, (2*blocks+2*rows*blocks)*64,
                  (2*blocks+2*rows*blocks)*256),
        'LoadBRC': c(rows*blocks, rows*blocks*64, rows*blocks*4),
        'Cast32': c(2*rows*blocks, 2*rows*width),
        'Mul': c(5*rows*blocks, 5*rows*width),
        'Muls': c(rows*blocks, rows*width), 'Adds': c(rows*blocks, rows*width),
        'Exp': c(rows*blocks, rows*width), 'Div': c(rows*blocks, rows*width),
        'Store': c(rows*blocks, rows*width, write=4*rows*width),
    })
    reduction = InstructionInventory({
        'Mask': c(3), 'Duplicate': c(rows+1, 64*(rows+1)),
        'Load': c(rows*blocks, rows*blocks*64, rows*blocks*256),
        'Abs': c(rows*blocks, rows*width), 'Max': c(rows*blocks, rows*width),
        'ReduceMax': c(rows, 64*rows), 'Div': c(rows, 64*rows),
        'Store': c(rows, rows, write=4*rows),
    })
    quant = InstructionInventory({
        'Mask': c(1+rows*blocks),
        'LoadBRC': c(rows, rows*64, rows*4),
        'Load': c(rows*blocks, rows*blocks*64, rows*blocks*256),
        'Div': c(rows*blocks, rows*width),
        'Cast32': c(rows*blocks, rows*width),
        'Cast64': c(2*rows*blocks, 2*rows*width),
        'Store': c(rows*blocks, rows*width, write=rows*width),
    })
    return dequant, reduction, quant


def capacity(inventory):
    result = inventory.capacity_work(CLASSES)
    # Retained Reg table: INT32->FP32 and FP32->INT16 are 32 lanes/cycle;
    # FP32 ALU and INT16->FP16->INT8 are 64. Keep raw lane counts above.
    casts = inventory.counts.get('Cast32', InstructionCount()).active_lanes
    return result + Work(resources={'reg_alu_lanes': casts})


def tiles(cfg, schedule):
    ends, count = [], 0
    for _, rows in schedule:
        count += ceildiv(rows, cfg.tile_rows)
        ends.append(count)
    def at(index):
        group = bisect_right(ends, index)
        offset = (index - (ends[group-1] if group else 0))*cfg.tile_rows
        scans=schedule[group][0]-(schedule[group-1][0] if group else -1)
        return min(cfg.tile_rows, schedule[group][1]-offset), offset == 0, scans
    return count, at


def build(op, geometry):
    meta = dict(op.metadata); declared = meta['grouped_dequant']
    cfg = Config(tuple(declared['expert_rows']), declared['width'], declared['input_capacity_rows'],
                 geometry.resource_units.get('vector', geometry.compute_units), geometry.buffers.get('ub', 0))
    if op.local_shape != (sum(cfg.rows), cfg.width):
        raise ModelingError('arch35_dequant_valid_shape_disagrees_with_groups')
    schedules = tuple(assignments(cfg, owner) for owner in range(cfg.blocks))
    primitives=PrimitiveInventory({'scalar_gm_access':cfg.blocks*len(cfg.rows)})
    scalar_work,_=primitives.partition(geometry.resource_units)
    price_scalar=bool(scalar_work.resources)
    work = Work(resources={'hbm': cfg.blocks*len(cfg.rows)*8})
    work += scalar_work
    inventory = InstructionInventory({})
    weight_reads = 0
    for schedule in schedules:
        weight_reads += len(schedule)*8*cfg.width
        for _, rows in schedule:
            full, tail = divmod(rows, cfg.tile_rows)
            for n, repeat in ((cfg.tile_rows, full), (tail, bool(tail))):
                if repeat:
                    for inv in inventories(n, cfg.width):
                        inventory += inv.scaled(repeat)
                        work += capacity(inv).scaled(repeat)
    reads = sum(cfg.rows)*(8*cfg.width+4) + weight_reads
    writes = sum(cfg.rows)*(cfg.width+4)
    work += Work(resources={'hbm': reads+writes, 'mte2_aiv': reads, 'mte3': writes})

    def program(context):
        roles = tuple(work.resources.keys() - {'hbm'})
        def owner_program(owner, schedule):
            placement = context.owned(owner, cfg.aivs, roles)
            allocations = (context.allocate('source_ub', 'ub', cfg.storage,
                unit_group(owner, cfg.aivs, context.pool_units.get('ub', cfg.aivs))),)
            count, at = tiles(cfg, schedule)
            def tile(index):
                rows, first, scans = at(index)
                invs = inventories(rows, cfg.width)
                def transfer(pipe, amount):
                    return context.bind(Work(resources={'hbm': amount, pipe: amount}))
                descriptor=()
                if price_scalar and first:
                    descriptor=(Stage('group_descriptor',context.bind(Work(resources={
                        'hbm':scans*8,'scalar_gm_access':scans})),
                        previous=('quantize','read'),placement=placement),)
                result=(
                    Stage('group_scale', transfer('mte2_aiv', 8*cfg.width if first else 0),
                          after=('group_descriptor',) if descriptor else (),
                          previous=('quantize', 'read') if first else (), placement=placement),
                    Stage('read', transfer('mte2_aiv', rows*(8*cfg.width+4)),
                          after=('group_scale',), placement=placement),
                    Stage('dequant_swiglu', context.bind(capacity(invs[0])), after=('read',),
                          previous=('quantize', 'scale_write'), placement=placement),
                    Stage('reduce', context.bind(capacity(invs[1])), after=('dequant_swiglu',), placement=placement),
                    Stage('scale_write', transfer('mte3', 4*rows), after=('reduce',),
                          previous=('output_write',), placement=placement),
                    Stage('quantize', context.bind(capacity(invs[2])), after=('reduce',), placement=placement),
                    Stage('output_write', transfer('mte3', rows*cfg.width),
                          after=('quantize', 'scale_write'), placement=placement),
                )
                trailing=len(cfg.rows)-schedule[-1][0]-1 if schedule else 0
                tail=()
                if price_scalar and index==count-1 and trailing:
                    tail=(Stage('trailing_descriptors',context.bind(Work(resources={
                        'hbm':trailing*8,'scalar_gm_access':trailing})),
                        after=('quantize','read'),placement=placement),)
                return (*descriptor,*result,*tail)
            descriptor_work=Work(resources={'hbm':len(cfg.rows)*8})
            if price_scalar:
                # Preserve source scan order around actual group processing.
                # An idle owner still visits every descriptor; active owners
                # scan skipped/zero groups before their next data group.
                descriptor_work=(Work(resources={'hbm':len(cfg.rows)*8,
                    'scalar_gm_access':len(cfg.rows)}) if not schedule else Work())
            return KernelProgram((
                Stage('reserve', acquire=allocations),
                # GetValue is a scalar GM access, not MTE2 DMA. Its payload is
                # retained, but absent scalar instruction rate is not invented.
                Stage('descriptors', context.bind(descriptor_work),after=('reserve',),
                    placement=placement if price_scalar else {}),
                Stage('tiles', after=('descriptors',), program=PipelineProgram(count, 2, tile)),
                Stage('release', after=('tiles',), free=allocations),
            ))
        return ParallelProgram(tuple(owner_program(owner, schedule) for owner, schedule in enumerate(schedules)))
    return OperatorPlan(work, program, metadata={
        'source_execution_domain': 'arch35_AIV_full_load',
        'grouped_dequant_schedule': {
            'kind': 'arch35_full_load', 'source': SOURCE,
            'input_capacity_rows': cfg.capacity, 'valid_rows': sum(cfg.rows),
            'host_special_group_cut': cfg.special, 'launched_aiv': cfg.blocks,
            'physical_aiv': cfg.aivs, 'ub_tile_rows': cfg.tile_rows,
            'ub_bytes_per_launched_owner': cfg.storage,
            'rows_per_owner': tuple(sum(n for _, n in schedule) for schedule in schedules),
            'groups_per_owner': tuple(len(schedule) for schedule in schedules),
            'weight_scale_read_bytes': weight_reads,
            'scalar_group_index_getvalue_calls': cfg.blocks*len(cfg.rows),
            'scalar_service': 'declared_owner_service_source_group_order' if price_scalar
                              else 'payload_only_issue_latency_not_declared',
            'primitive_services':primitives.to_dict(geometry.resource_units),
            'primitive_count_boundary':'INT64_GlobalTensor_scalar_index_source_calls_not_ISA',
            'register_inventory': inventory.to_dict(),
            'limits': ('capacity_branch_is_explicit_not_inferred_from_valid_rows',
                       'source_queue_capacity_dependencies_not_compiler_instruction_latency',
                       'undeclared_scalar_service_remains_unpriced',
                       'multi_ub_tile_scale_queue_lifetime_is_conservative_resident'),
        },
    })
