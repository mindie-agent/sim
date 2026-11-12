"""Per-token DynamicQuantV2 ahead of a dynamic int activation.

The producer is the explicit ``npu_dynamic_quant`` call in vllm-ascend
W8A8 dynamic ``apply``: the GEMM consumes that int8 row and its scale.
It is not a GEMM epilogue and not the FP8/MX quantizer.

A role qualifies only when its own resolved activation format is int8
and the recipe's activation scheme is dynamic.  A BF16 role override
(``gdn.o``, ``gdn.beta_alpha``) never qualifies, because the matrix
activation format is the stream dtype.  Static int8 and FP8/MX methods
do not take this producer.  An explicit ``preset`` name is resolved
through the catalog recipe when the case source does not repeat the
scheme fields.

Traffic is one GM load of the stream row when that row fits the host
full-load check, an int8 payload, and one FP32 scale.  The scale and
cast vector body rereads the same UB x.  A second GM load is only the
large-shape split of a row that misses that check.  Double-buffer depth
is not a second GM load.  Symmetric mode does not store an offset.

The elapsed time is the per-owner engine program: one CopyIn, then the
scale VF, then the quantization VF on that same UB x, then the y write
and the distinct FP32 scale write.  Full-db and large-shape tiles overlap
through the source queue depth.  No invocation floor and no measured
time is an input.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
from fractions import Fraction
from typing import Any

from mindie_sim.catalog.quant import QUANTIZATION_PRESETS
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import PipelineProgram, Stage
from mindie_sim.schema.common import ModelingError, dtype_bytes

_INT_ACTIVATIONS = frozenset({"int8", "int4"})
_OTHER_METHODS = frozenset({
    "mxfp8", "fp8", "fp8_e4m3", "w8a8_mxfp8", "weight_only",
})
_STATIC_SCHEMES = frozenset({"static", "none"})

# CalculateTilingData, non-per-channel.  The host always counts the offset
# buffer (offsetBuffer = 1), including symmetrical mode.  Smooth is optional.
# Single: align16(N) * (7 + 2*smooth + 4).  Double: align16(N) * (14 + 4*smooth + 8).
_HOST_ALIGN = 16
_HOST_RESERVE = 1024
_HOST_RPT = 128
_MODE_UNBOUND = 0
_MODE_FULL_SINGLE = 1
_MODE_FULL_DB = 2
_MODE_LARGE = 3
# Same 32-byte UB transaction granule the binder charges on a GM copy of x.
_GM_ALIGN = 32
# One vector repeat is eight 32-byte blocks.  Counted as issue slots.
_REPEAT_BYTES = 256
_COMPARE_INT = 255

# The two existing inventory terms, one per source VF.  Not a fitted split.
VECTOR_SPLIT = "helper_1024_per_256_tile_then_round_H_59_8"
# Host coreNum pins one AIV.  Idle units are not a share coefficient.
OCCUPANCY = "pinned_block_idx_unit_rate_over_resource_units"
# c09d3e20 model_runner_v1._pad_for_sequence_parallelism.
ROW_SOURCE = "c09d3e20_pad_for_sequence_parallelism"


def _align_up(value: int, base: int) -> int:
    return (int(value) + int(base) - 1) // int(base) * int(base)


def host_source_copies(
    row_len: int, ub_bytes: int, *, has_smooth: bool = False,
) -> dict[str, Any]:
    """GM copies of x from the arch35 host row check.

    ``copies`` is the element count of each ``CopyIn`` of x.  Full-load is
    one copy of the row.  Large-shape reads each full chunk twice (max,
    then y) and the tail chunk once.  ``ub_bytes`` is the per-core unified
    buffer.  Zero means the machine buffer is not on this supply.
    """
    row_len = int(row_len)
    one = {
        "mode": "ub_unbound", "mode_code": _MODE_UNBOUND, "use_db": False,
        "gm_source_reads": 1, "copies": (row_len,),
    }
    if row_len < 1 or int(ub_bytes) <= 0:
        return one
    aligned = _align_up(row_len, _HOST_ALIGN)
    smooth = 1 if has_smooth else 0
    single = aligned * (smooth * 2 + 7 + 4)
    double = aligned * (smooth * 4 + 14 + 8)
    usable = int(ub_bytes) - _HOST_RESERVE
    if single <= usable and double <= usable:
        return {
            "mode": "full_db", "mode_code": _MODE_FULL_DB, "use_db": True,
            "gm_source_reads": 1, "copies": (row_len,),
        }
    if single <= usable:
        return {
            "mode": "full_single", "mode_code": _MODE_FULL_SINGLE, "use_db": False,
            "gm_source_reads": 1, "copies": (row_len,),
        }
    calc_large = 2 * (smooth * 2 + 4)
    usable_large = int(ub_bytes) - 2 * _HOST_RESERVE
    inner = 0
    if usable_large > 0 and calc_large > 0:
        inner = (usable_large // calc_large // _HOST_RPT) * _HOST_RPT
    if inner <= 0:
        return {
            "mode": "large", "mode_code": _MODE_LARGE, "use_db": True,
            "gm_source_reads": 1, "copies": (row_len,),
        }
    times, tail = divmod(row_len, inner)
    copies = (inner,) * (2 * times)
    if tail:
        copies = copies + (tail,)
    return {
        "mode": "large", "mode_code": _MODE_LARGE, "use_db": True,
        "gm_source_reads": 2 if times else 1, "copies": copies,
    }


def _source_map(quant: Any, category: str) -> dict[str, str]:
    resolved = quant.for_category(category)
    return {str(key): str(value) for key, value in resolved.source}


def _scheme_and_method(quant: Any, category: str) -> tuple[str, str]:
    source = _source_map(quant, category)
    scheme = source.get("activation_scheme", "")
    method = source.get("quant_method", "")
    preset_name = source.get("preset", "")
    if scheme or method or not preset_name:
        return scheme, method
    factory = QUANTIZATION_PRESETS.get(preset_name)
    if factory is None:
        return scheme, method
    baked = dict(factory().source)
    return baked.get("activation_scheme", ""), baked.get("quant_method", "")


def per_token_dynamic_int(quant: Any, category: str, act_format: str) -> bool:
    """True when this role's activation is a per-token dynamic int quant."""
    if act_format not in _INT_ACTIVATIONS:
        return False
    if quant is None:
        return act_format == "int8"
    scheme, method = _scheme_and_method(quant, category)
    if method in _OTHER_METHODS:
        return False
    if scheme in _STATIC_SCHEMES:
        return False
    if scheme == "dynamic":
        return True
    return act_format == "int8"


def dynamic_quant_traffic(
    quant: Any, category: str, elements: int, act_format: str, stream_dtype: str,
) -> tuple[float, float, tuple[tuple[str, Any], ...]] | None:
    """Per-token read, write and signature extras, or None when unqualified.

    ``stream_dtype`` is the operand that enters ``npu_dynamic_quant`` (the
    BF16 norm or activation row), not the int8 result and not the GEMM's
    declared output format.
    """
    if not per_token_dynamic_int(quant, category, act_format):
        return None
    count = int(elements)
    if count < 1:
        return None
    source_bytes = dtype_bytes(stream_dtype or "bf16")
    payload_bytes = dtype_bytes(act_format)
    if quant is None:
        scale_format = "fp32"
    else:
        scale_format = str(
            quant.for_category(category).execution_activation_scale_format() or "fp32")
    scale_bytes = dtype_bytes(scale_format)
    # One logical GM load of x.  The binder repeats a full chunk only when
    # the machine UB misses the host full-load check.  Symmetric npu_dynamic_quant
    # returns y and the per-token scale; the offset GM store stays off.
    read_bytes = float(count * source_bytes)
    write_bytes = count * payload_bytes + scale_bytes
    extra = (
        ("quant_op", "QuantV2"),
        ("out_format", act_format),
        ("category", category),
        ("quant_passes", 2),
        ("pass_read_bytes", count * source_bytes),
        ("source_bytes", source_bytes),
        ("source_format", stream_dtype or "bf16"),
        ("row_scale_bytes", scale_bytes),
        ("scale_format", scale_format),
        ("offset_gm_bytes", 0),
        ("symmetrical", 1),
        ("has_smooth", 0),
        ("block_axis", "row"),
    )
    return read_bytes, write_bytes, extra


@dataclass(frozen=True)
class RowDemand:
    """Physical quant rows and the rule that produced them."""

    valid: int
    rows: int
    source: str


def _authored_rows(produced: Any) -> tuple[int | None, int]:
    if isinstance(produced, bool) or produced is None:
        raise ModelingError("fq5_quant_rows_invalid")
    if isinstance(produced, int):
        return None, produced
    if isinstance(produced, Mapping):
        if "rows" not in produced:
            raise ModelingError("fq5_quant_rows_missing")
        valid = produced.get("valid")
        return (None if valid is None else int(valid)), int(produced["rows"])
    rows = getattr(produced, "rows", None)
    if isinstance(rows, bool) or rows is None:
        raise ModelingError("fq5_quant_rows_invalid")
    valid = getattr(produced, "valid", None)
    return (None if valid is None else int(valid)), int(rows)


def _fq5_quant_rows(geometry: Any, kind: str, tp: int) -> tuple[str, Any] | None:
    """Use FQ5 when that module already publishes ``quant_rows``. Do not invent it."""
    for module_name in (
            "mindie_sim.prediction.fq5",
            "mindie_sim.prediction.piecewise.fq5"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, "quant_rows", None)
        if not callable(fn):
            continue
        produced = fn(geometry, kind=kind, tp=tp)
        if produced is None:
            continue
        return module_name, produced
    return None


def quant_row_demand(geometry: Any, *, kind: str = "", tp: int = 1) -> RowDemand:
    """Rows on the quant tensor.

    FQ5 wins when it is already importable. A published ``physical_tokens``
    count is that tensor and is not padded again. Otherwise the base is the
    sum of request ``q_len`` (not the graph bucket and not effective tokens).
    ``c09d3e20`` ``_pad_for_sequence_parallelism`` rounds that base up to ``tp``
    when sequence parallelism is on. A uniform decode capture pads every
    quant kernel. A mixed or eager prefill pads only F0. The kernel source
    does not branch on phase; this is the capture-versus-blanket selection.
    """
    authored = _fq5_quant_rows(geometry, str(kind or ""), int(tp))
    requests = tuple(getattr(geometry, "requests", ()) or ())
    valid = sum(int(request.q_len) for request in requests)
    if authored is not None:
        module_name, produced = authored
        authored_valid, rows = _authored_rows(produced)
        if rows < 0:
            raise ModelingError(f"fq5_quant_rows_invalid:{rows}")
        return RowDemand(
            valid if authored_valid is None else authored_valid,
            rows, f"fq5:{module_name}")
    published = int(getattr(geometry, "physical_tokens", 0) or 0)
    if published > 0:
        return RowDemand(valid, published, "physical_tokens")
    width = int(tp)
    uniform_decode = bool(requests) and all(
        request.phase == "decode" for request in requests)
    pad = width > 1 and (uniform_decode or str(kind or "") == "F0")
    rows = valid
    if pad:
        rows = valid + (-valid) % width
    return RowDemand(valid, rows, ROW_SOURCE if pad else "request_q_len")


def _owner_rows(rows: int, aiv: int) -> tuple[int, ...]:
    """Launched AIV row counts. A zero-row tail core is not launched."""
    rows = int(rows)
    aiv = max(1, int(aiv))
    if rows <= 0:
        return ()
    launched = min(aiv, rows)
    if rows % aiv == 0:
        each = rows // launched
        return (each,) * launched
    head_rows = (rows + aiv - 1) // aiv
    tail_rows = rows // aiv
    if rows < aiv:
        return (head_rows,) * launched
    head_n = rows % aiv
    return (head_rows,) * head_n + (tail_rows,) * (launched - head_n)


def _multi_row(mode_code: int, use_db: bool, row_len: int, ub_bytes: int,
               has_smooth: bool, rows_on_core: int) -> int:
    """Host multiRowNum for this core: min(ubAvail, 255, rows on the core)."""
    if rows_on_core <= 1 or mode_code == _MODE_LARGE:
        return 1
    if int(ub_bytes) <= _HOST_RESERVE:
        return 1
    aligned = _align_up(row_len, _HOST_ALIGN)
    smooth = 1 if has_smooth else 0
    if use_db:
        footprint = aligned * (smooth * 4 + 14 + 8)
    else:
        footprint = aligned * (smooth * 2 + 7 + 4)
    usable = int(ub_bytes) - _HOST_RESERVE
    if footprint <= 0 or footprint > usable:
        return 1
    ub_avail = usable // footprint
    return max(1, min(ub_avail, _COMPARE_INT, rows_on_core))


def _vf_terms(row_len: int, source_bytes: float) -> tuple[int, int]:
    """Scale VF gets the 1024-per-256 helper. The y VF gets round(H*59/8).

    Instruction occupancy inside either VF is not knowable from that
    inventory. The row-axis repeat quantum is added to y only when it
    exceeds the sum, matching Max(inventory, quantum).
    """
    width = max(1, int(row_len))
    y_ops = int(round(width * 59.0 / 8.0))
    scale_ops = 1024 * math.ceil(width / 256)
    width_bytes = float(source_bytes)
    if width_bytes <= 0:
        quantum = _REPEAT_BYTES
    else:
        quantum = max(1, int(math.ceil(_REPEAT_BYTES / width_bytes)))
    inventory = y_ops + scale_ops
    if quantum > inventory:
        y_ops += quantum - inventory
    return scale_ops, y_ops


def _prorate(total: int, lengths: tuple[int, ...]) -> tuple[int, ...]:
    """Split ``total`` across chunk lengths. The last chunk keeps the remainder."""
    if not lengths:
        return ()
    span = sum(int(length) for length in lengths)
    if span <= 0:
        return (0,) * (len(lengths) - 1) + (int(total),)
    assigned: list[int] = []
    used = 0
    last = len(lengths) - 1
    for index, length in enumerate(lengths):
        if index == last:
            assigned.append(int(total) - used)
        else:
            part = int(total) * int(length) // span
            assigned.append(part)
            used += part
    return tuple(assigned)


def _aligned_bytes(elements: int, width: float) -> int:
    raw = float(elements) * float(width)
    if not math.isfinite(raw) or raw < 0:
        raise ModelingError(f"quant_copy_bytes_invalid:{elements}:{width}")
    return _align_up(int(math.ceil(raw - 1e-9)), _GM_ALIGN)


def _exact_bytes(elements: int, width: float) -> int:
    raw = float(elements) * float(width)
    if not math.isfinite(raw) or raw < 0:
        raise ModelingError(f"quant_copy_bytes_invalid:{elements}:{width}")
    return int(math.ceil(raw - 1e-9))


def _large_inner(ub_bytes: int, has_smooth: bool) -> int:
    smooth = 1 if has_smooth else 0
    calc_large = 2 * (smooth * 2 + 4)
    usable = int(ub_bytes) - 2 * _HOST_RESERVE
    if usable <= 0 or calc_large <= 0:
        return 0
    return (usable // calc_large // _HOST_RPT) * _HOST_RPT


def _full_signature(count: int, row_len: int, source_bytes: float, y_bytes: float,
                    scale_bytes: float, scale_ops: int, y_ops: int) -> tuple[int, int, int, int, int]:
    return (
        count * _aligned_bytes(row_len, source_bytes),
        count * int(scale_ops),
        count * int(y_ops),
        count * _exact_bytes(row_len, y_bytes),
        count * _exact_bytes(1, scale_bytes),
    )


def _large_row_signatures(row_len: int, source_bytes: float, y_bytes: float,
                          scale_bytes: float, scale_ops: int, y_ops: int,
                          ub_bytes: int, has_smooth: bool) -> tuple[tuple[int, int, int, int, int], ...]:
    """One row of Process then ProcessLoop. Vector work is issued once per element."""
    inner = _large_inner(ub_bytes, has_smooth)
    if inner <= 0:
        return (_full_signature(1, row_len, source_bytes, y_bytes, scale_bytes, scale_ops, y_ops),)
    times, tail = divmod(int(row_len), inner)
    covered = (inner,) * times + ((tail,) if tail else ())
    if not covered:
        return (_full_signature(1, row_len, source_bytes, y_bytes, scale_bytes, scale_ops, y_ops),)
    scale_parts = _prorate(scale_ops, covered)
    y_parts = _prorate(y_ops, covered)
    signatures: list[tuple[int, int, int, int, int]] = []
    for index in range(times):
        signatures.append((
            _aligned_bytes(inner, source_bytes), scale_parts[index], 0, 0, 0))
    if tail:
        signatures.append((
            _aligned_bytes(tail, source_bytes),
            scale_parts[times],
            y_parts[times],
            _exact_bytes(tail, y_bytes),
            0,
        ))
    for index in range(times):
        signatures.append((
            _aligned_bytes(inner, source_bytes),
            0,
            y_parts[index],
            _exact_bytes(inner, y_bytes),
            0,
        ))
    if not signatures:
        signatures.append(_full_signature(
            1, row_len, source_bytes, y_bytes, scale_bytes, scale_ops, y_ops))
    last = signatures[-1]
    signatures[-1] = last[:4] + (_exact_bytes(1, scale_bytes),)
    return tuple(signatures)


def _core_signatures(rows_on_core: int, plan: Mapping[str, Any], row_len: int,
                     source_bytes: float, y_bytes: float, scale_bytes: float,
                     ub_bytes: int, has_smooth: bool, scale_ops: int,
                     y_ops: int) -> tuple[tuple[int, int, int, int, int], ...]:
    if rows_on_core <= 0:
        return ()
    if int(plan["mode_code"]) == _MODE_LARGE:
        row = _large_row_signatures(
            row_len, source_bytes, y_bytes, scale_bytes, scale_ops, y_ops,
            ub_bytes, has_smooth)
        return row * int(rows_on_core)
    multi = _multi_row(
        int(plan["mode_code"]), bool(plan["use_db"]), row_len, ub_bytes,
        has_smooth, rows_on_core)
    signatures = []
    offset = 0
    while offset < rows_on_core:
        count = min(multi, rows_on_core - offset)
        signatures.append(_full_signature(
            count, row_len, source_bytes, y_bytes, scale_bytes, scale_ops, y_ops))
        offset += count
    return tuple(signatures)


def _pipe_units(units: Mapping[str, int], aiv: int) -> dict[str, int]:
    """Published unit counts. A missing pipe falls back to the AIV owner count.

    HBM is not in the 9579 resource_units record, so its unit count stays 1:
    the die, shared by every launched copy.
    """
    table = {str(key): int(value) for key, value in units.items()}

    def take(*names: str) -> int:
        for name in names:
            value = int(table.get(name, 0) or 0)
            if value > 0:
                return value
        return 0

    owners = max(1, int(aiv))
    return {
        "vector": take("vector") or owners,
        "mte2": take("mte2", "mte2_aiv", "vector") or owners,
        "mte3": take("mte3", "vector") or owners,
        "hbm": take("hbm") or 1,
    }


def _require_rate(rates: Mapping[str, float], name: str) -> float:
    try:
        rate = float(rates[name])
    except KeyError as exc:
        raise ModelingError(f"quant_timeline_pipe_missing:{name}") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ModelingError(f"quant_timeline_pipe_missing:{name}")
    return rate


_STAGE_NAMES = ("copy_in", "scale_vf", "y_vf", "copy_y", "copy_scale")
_STAGE_BANK: dict[tuple, tuple[Stage, ...]] = {}


def _stage_work(signature: tuple[int, int, int, int, int]) -> tuple[dict[str, float], ...]:
    copy_in, scale, y_ops, copy_y, copy_scale = signature
    return (
        {key: float(value) for key, value in (("mte2", copy_in), ("hbm", copy_in)) if value},
        {key: float(value) for key, value in (("vector", scale),) if value},
        {key: float(value) for key, value in (("vector", y_ops),) if value},
        {key: float(value) for key, value in (("mte3", copy_y), ("hbm", copy_y)) if value},
        {key: float(value) for key, value in (("mte3", copy_scale), ("hbm", copy_scale)) if value},
    )


def _tile_stages(signature: tuple[int, int, int, int, int], unit: int) -> tuple[Stage, ...]:
    key = (signature, int(unit))
    found = _STAGE_BANK.get(key)
    if found is not None:
        return found
    works = _stage_work(signature)
    pin_read = {"mte2": (int(unit),)}
    pin_vector = {"vector": (int(unit),)}
    pin_write = {"mte3": (int(unit),)}
    stages = (
        Stage("copy_in", works[0], after=(), placement=pin_read),
        Stage("scale_vf", works[1], after=("copy_in",), previous=("y_vf",),
              placement=pin_vector),
        Stage("y_vf", works[2], after=("scale_vf",), placement=pin_vector),
        Stage("copy_y", works[3], after=("y_vf",), previous=("copy_scale",),
              placement=pin_write),
        Stage("copy_scale", works[4], after=("copy_y",), placement=pin_write),
    )
    _STAGE_BANK[key] = stages
    return stages


def _serial_ms(signatures: tuple[tuple[int, int, int, int, int], ...],
               local: Mapping[str, float]) -> float:
    total = 0.0
    for signature in signatures:
        for work in _stage_work(signature):
            times = [amount / local[pipe] for pipe, amount in work.items() if amount]
            if times:
                total += max(times)
    return total


def _stage_ms(signature: tuple[int, int, int, int, int],
              local: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    result = []
    for name, work in zip(_STAGE_NAMES, _stage_work(signature)):
        times = [amount / local[pipe] for pipe, amount in work.items() if amount]
        result.append((name, max(times) if times else 0.0))
    return tuple(result)


def _freeze_pairs(items: Mapping[str, Any], numeric: type) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(key), numeric(value)) for key, value in items.items()))


@lru_cache(maxsize=48)
def _cached_timeline(rate_key: tuple, unit_key: tuple, rows: int, row_len: int,
                     source_bytes: float, y_bytes: float, scale_bytes: float,
                     ub_bytes: int, has_smooth: int, aiv: int) -> tuple:
    rates = dict(rate_key)
    units = dict(unit_key)
    plan = host_source_copies(int(row_len), int(ub_bytes), has_smooth=bool(has_smooth))
    owners = _owner_rows(int(rows), int(aiv))
    scale_ops, y_ops = _vf_terms(int(row_len), float(source_bytes))
    per_owner = tuple(
        _core_signatures(
            count, plan, int(row_len), float(source_bytes), float(y_bytes),
            float(scale_bytes), int(ub_bytes), bool(has_smooth), scale_ops, y_ops)
        for count in owners)
    pipe_units = _pipe_units(units, int(aiv))
    launched = len(per_owner)
    for name, count in pipe_units.items():
        if name == "hbm":
            continue
        if launched and count < launched:
            raise ModelingError(
                f"quant_pipe_units_below_launched:{name}:{count}:{launched}")
    die = {name: _require_rate(rates, name) for name in ("vector", "mte2", "mte3", "hbm")}
    local = {name: die[name] / pipe_units[name] for name in die}
    mte2_bytes = 0
    mte3_bytes = 0
    vector_ops = 0
    for signatures in per_owner:
        for signature in signatures:
            copy_in, scale, y_work, copy_y, copy_scale = signature
            mte2_bytes += copy_in
            mte3_bytes += copy_y + copy_scale
            vector_ops += scale + y_work
    head_serial = _serial_ms(per_owner[0], local) if per_owner else 0.0
    tail_serial = _serial_ms(per_owner[-1], local) if per_owner else 0.0
    head_stages = (_stage_ms(per_owner[0][0], local)
                   if per_owner and per_owner[0] else ())
    makespan = head_ms = tail_ms = 0.0
    busy = {"vector": 0.0, "mte2": 0.0, "mte3": 0.0, "hbm": 0.0}
    if per_owner:
        resources = tuple(
            Resource(name, die[name], "work", "fair", pipe_units[name])
            for name in ("vector", "mte2", "mte3", "hbm"))
        engine = Engine(resources)
        ready = engine.submit("ready", release=0)
        done = []
        for index, signatures in enumerate(per_owner):
            stages = tuple(_tile_stages(signature, index) for signature in signatures)

            def tile(tile_index: int, bound: tuple = stages) -> tuple[Stage, ...]:
                return bound[tile_index]

            program = PipelineProgram(
                len(signatures), 2 if plan["use_db"] else 1, tile,
                minimum_ms=0.0, setup_ms=0.0)
            done.append(program.start(engine, f"aiv{index}", ready, owner=f"aiv{index}"))
        join = engine.submit("join", after=tuple(done))
        engine.advance(until=(join,))
        origin = ready.time or 0.0
        makespan = (join.time or 0.0) - origin
        head_ms = (done[0].time or 0.0) - origin
        tail_ms = (done[-1].time or 0.0) - origin
        busy = {name: float(engine.busy_ms.get(name, 0.0)) for name in busy}
    return (
        makespan, head_ms, tail_ms, head_serial, tail_serial,
        tuple(busy[name] for name in ("vector", "mte2", "mte3", "hbm")),
        len(per_owner[0]) if per_owner else 0,
        len(per_owner[-1]) if per_owner else 0,
        2 if plan["use_db"] else 1,
        str(plan["mode"]), int(plan["mode_code"]), launched,
        owners[0] if owners else 0, owners[-1] if owners else 0,
        mte2_bytes, mte3_bytes, mte2_bytes + mte3_bytes, vector_ops,
        scale_ops, y_ops,
        tuple(local[name] for name in ("vector", "mte2", "mte3", "hbm")),
        tuple(pipe_units[name] for name in ("vector", "mte2", "mte3", "hbm")),
        head_stages,
    )


def quant_timeline(*, rates: Mapping[str, float], units: Mapping[str, int],
                   rows: int, row_len: int, source_bytes: float, y_bytes: float,
                   scale_bytes: float, ub_bytes: int, has_smooth: bool = False,
                   aiv: int = 1) -> dict[str, Any]:
    """Per-owner CopyIn, dependent VFs, and y/scale writes.

    The cache is the engine run for one physical shape. It is not a report.
    """
    raw = _cached_timeline(
        _freeze_pairs(rates, float), _freeze_pairs(units, int),
        int(rows), int(row_len), float(source_bytes), float(y_bytes),
        float(scale_bytes), int(ub_bytes), int(bool(has_smooth)), int(aiv))
    pipes = ("vector", "mte2", "mte3", "hbm")
    return {
        "makespan_ms": raw[0], "head_ms": raw[1], "tail_ms": raw[2],
        "head_serial_ms": raw[3], "tail_serial_ms": raw[4],
        "busy_ms": dict(zip(pipes, raw[5])),
        "tiles_per_head": raw[6], "tiles_per_tail": raw[7], "slots": raw[8],
        "mode": raw[9], "mode_code": raw[10], "launched": raw[11],
        "rows_per_head": raw[12], "rows_per_tail": raw[13],
        "mte2_bytes": raw[14], "mte3_bytes": raw[15], "hbm_bytes": raw[16],
        "vector_ops": raw[17], "scale_ops_per_row": raw[18], "y_ops_per_row": raw[19],
        "local_rates": dict(zip(pipes, raw[20])),
        "pipe_units": dict(zip(pipes, raw[21])),
        "head_tile_stages_ms": raw[22],
    }


def quant_program_result(*, rates: Mapping[str, float], units: Mapping[str, int],
                         rows: int, row_len: int, source_bytes: float,
                         y_bytes: float, scale_bytes: float, ub_bytes: int,
                         has_smooth: bool = False, aiv: int = 1,
                         ) -> tuple[float, str, tuple[str, ...], tuple[tuple[str, Fraction], ...]]:
    """Four-tuple elapsed program. The fifth ledger element is intentionally absent.

    No invocation floor and no measured latency is submitted. Stage work stays
    the byte and vector ledger; this result only replaces elapsed time.
    """
    report = quant_timeline(
        rates=rates, units=units, rows=rows, row_len=row_len,
        source_bytes=source_bytes, y_bytes=y_bytes, scale_bytes=scale_bytes,
        ub_bytes=ub_bytes, has_smooth=has_smooth, aiv=aiv)
    # Sibling kernels publish a float because their elapsed rate is a float.
    # The engine makespan is published as that same type. A later serial sum
    # can miss an exact float equality by one ulp. That miss is the chain.
    # The makespan is not snapped to a neighbor and is not the collapsed max.
    duration = float(report["makespan_ms"])
    busy = tuple(
        (pipe, Fraction.from_float(float(value)))
        for pipe, value in report["busy_ms"].items() if value > 0)
    if not busy:
        return duration, "", (), ()
    peak = max(value for _pipe, value in busy)
    tied = tuple(pipe for pipe, value in busy if value == peak)
    return duration, tied[0], tied, busy
