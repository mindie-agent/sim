"""Register and UB accounting for the grouped INT8 epilogue.

[Role] Compute full-width tile geometry and instruction demand for dequant,
  SwiGLU, row reduction and quantization.
[Invariants] Full-width tiles respect UB capacity. Zero expert rows issue no
  payload work.
[Timeouts & Locks] Work scales with groups and active owners. The inventory
  cache is bounded.
[Side-Effect Policy] Creates local records only. No I/O or device access.

The work model uses the existing ops-nn full-load tiler and register inventory
from revision 955f8b33985b6d72989ebff341bb9c2c564a51bc.
"""

from dataclasses import dataclass
from functools import lru_cache

from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes

from mindie_sim.prediction.instructions import InstructionClass, InstructionCount, InstructionInventory
from mindie_sim.prediction.mechanism import Work

_MAX_FULL_LOAD_WIDTH = 5120
_COLUMN_SCALE_ALIGNMENT = 8
_ROW_QUEUE_ALIGNMENT_BYTES = 32
_UB_RESERVED_BYTES = 1024
_GROUP_CUT_MIN_EXPERTS = 32
_GROUP_CUT_MAX_AVERAGE_ROWS = 16
_REGISTER_LANES = 64
_INVENTORY_CACHE_SIZE = 256
_FP32_BYTES = int(dtype_bytes("fp32"))
_INT32_BYTES = int(dtype_bytes("int32"))
_DEQUANT_MULS_PER_ELEMENT = 5
_REDUCTION_MASK_CALLS = 3

_CLASSES = {
    "Mask": InstructionClass(1), "Load": InstructionClass(1, 1, ub_read=True),
    "LoadBRC": InstructionClass(1, 1, ub_read=True),
    "Store": InstructionClass(2, ub_write=True),
    **{name: InstructionClass(1, 1, "reg_alu_lanes") for name in (
        "Cast32", "Cast64", "Mul", "Muls", "Adds", "Abs", "Max", "Duplicate")},
    "Exp": InstructionClass(1, 1, "reg_special_lanes"),
    "Div": InstructionClass(1, 1, "reg_special_lanes"),
    "ReduceMax": InstructionClass(2, 1, "reg_reduce_lanes"),
}


def _ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class _FullWidthTiles:
    rows: tuple[int, ...]
    width: int
    capacity: int
    aivs: int
    ub: int

    def __post_init__(self):
        if (not self.rows or any(type(n) is not int or n < 0 for n in self.rows)
                or type(self.width) is not int or not 1 <= self.width <= _MAX_FULL_LOAD_WIDTH
                or type(self.capacity) is not int or self.capacity < max(1, sum(self.rows))
                or self.aivs < 1):
            raise ModelingError("arch35_dequant_requires_integral_rows_capacity_and_full_load_width")
        if self.tile_rows < 1:
            raise ModelingError("arch35_dequant_full_load_does_not_fit_ub")

    @property
    def aligned_width(self):
        return align_up(self.width, _COLUMN_SCALE_ALIGNMENT)

    @property
    def row_storage(self):
        # Double-buffer input and output. Keep one row scale and one FP32 temporary.
        input_slot = 2 * self.aligned_width * _INT32_BYTES + _ROW_QUEUE_ALIGNMENT_BYTES
        output_slot = align_up(self.width, _ROW_QUEUE_ALIGNMENT_BYTES)
        temporary = self.aligned_width * _FP32_BYTES
        return 2 * (input_slot + output_slot) + _ROW_QUEUE_ALIGNMENT_BYTES + temporary

    @property
    def tile_rows(self):
        # Two FP32 column scales and one FP32 column-scale temporary.
        column_storage = (2 * _FP32_BYTES + _FP32_BYTES) * self.aligned_width
        available = self.ub - _UB_RESERVED_BYTES - column_storage
        return min(self.capacity, available // self.row_storage)

    @property
    def group_cut(self):
        return (len(self.rows) >= _GROUP_CUT_MIN_EXPERTS
                and self.capacity // len(self.rows) <= _GROUP_CUT_MAX_AVERAGE_ROWS)

    @property
    def blocks(self):
        return min(self.aivs, self.capacity if self.group_cut
                   else _ceildiv(self.capacity, self.tile_rows))


def _assignments(cfg, owner):
    """Return the row count of each nonempty group visited by this owner."""
    counts = []
    cursor = owner
    group_cut, blocks = cfg.group_cut, cfg.blocks
    for group, rows in enumerate(cfg.rows):
        if group_cut:
            if group != cursor:
                continue
            if not rows:
                if any(cfg.rows[g] for g in range(group + blocks, len(cfg.rows), blocks)):
                    raise ModelingError("arch35_dequant_zero_group_stalls_later_source_owner")
                continue
            count = rows
            cursor += blocks
        else:
            per_owner = _ceildiv(rows, blocks)
            count = max(0, min(per_owner, rows - owner * per_owner))
        if count:
            counts.append(count)
    return counts


@lru_cache(maxsize=_INVENTORY_CACHE_SIZE)
def _inventories(rows, width):
    """Count calls, active lanes and UB bytes for each vector scope."""
    blocks = _ceildiv(width, _REGISTER_LANES)
    row_blocks = rows * blocks
    elements = rows * width
    register_bytes = _REGISTER_LANES * _FP32_BYTES
    dequant_loads = 2 * blocks + 2 * row_blocks
    dequant = InstructionInventory({
        "Mask": InstructionCount(blocks),
        "Load": InstructionCount(dequant_loads, dequant_loads * _REGISTER_LANES,
                                 dequant_loads * register_bytes),
        "LoadBRC": InstructionCount(row_blocks, row_blocks * _REGISTER_LANES,
                                    row_blocks * _FP32_BYTES),
        "Cast32": InstructionCount(2 * row_blocks, 2 * elements),
        "Mul": InstructionCount(_DEQUANT_MULS_PER_ELEMENT * row_blocks,
                                _DEQUANT_MULS_PER_ELEMENT * elements),
        "Muls": InstructionCount(row_blocks, elements),
        "Adds": InstructionCount(row_blocks, elements),
        "Exp": InstructionCount(row_blocks, elements),
        "Div": InstructionCount(row_blocks, elements),
        "Store": InstructionCount(row_blocks, elements, write_bytes=_FP32_BYTES * elements),
    })
    reduction = InstructionInventory({
        "Mask": InstructionCount(_REDUCTION_MASK_CALLS),
        "Duplicate": InstructionCount(rows + 1, _REGISTER_LANES * (rows + 1)),
        "Load": InstructionCount(row_blocks, row_blocks * _REGISTER_LANES,
                                 row_blocks * register_bytes),
        "Abs": InstructionCount(row_blocks, elements),
        "Max": InstructionCount(row_blocks, elements),
        "ReduceMax": InstructionCount(rows, _REGISTER_LANES * rows),
        "Div": InstructionCount(rows, _REGISTER_LANES * rows),
        "Store": InstructionCount(rows, rows, write_bytes=_FP32_BYTES * rows),
    })
    quant = InstructionInventory({
        "Mask": InstructionCount(1 + row_blocks),
        "LoadBRC": InstructionCount(rows, rows * _REGISTER_LANES, rows * _FP32_BYTES),
        "Load": InstructionCount(row_blocks, row_blocks * _REGISTER_LANES,
                                 row_blocks * register_bytes),
        "Div": InstructionCount(row_blocks, elements),
        "Cast32": InstructionCount(row_blocks, elements),
        "Cast64": InstructionCount(2 * row_blocks, 2 * elements),
        "Store": InstructionCount(row_blocks, elements, write_bytes=elements),
    })
    return dequant, reduction, quant


def _capacity(inventory):
    # The INT32-to-FP32 and FP32-to-INT16 conversions use half the FP32 ALU lanes.
    casts = inventory.counts.get("Cast32", InstructionCount()).active_lanes
    return inventory.capacity_work(_CLASSES) + Work(resources={"reg_alu_lanes": casts})


def register_epilogue_work(group_rows, width, rows, vector_units, ub_bytes):
    """Return grouped epilogue capacity demand, owner count and tile rows."""
    try:
        config = _FullWidthTiles(tuple(group_rows), width, rows, vector_units, ub_bytes)
    except (ModelingError, ValueError) as exc:
        raise ModelingError("xllm_fused_epilogue_geometry_unsupported") from exc
    tile_rows = config.tile_rows
    work = Work()
    for owner in range(config.blocks):
        for group_count in _assignments(config, owner):
            full, tail = divmod(group_count, tile_rows)
            for count, repeat in ((tile_rows, full), (tail, bool(tail))):
                if repeat:
                    for inventory in _inventories(count, config.width):
                        work += _capacity(inventory).scaled(repeat)
    return work, config.blocks, tile_rows
