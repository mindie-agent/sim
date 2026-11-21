"""O(1) indexed recurrent-scan template construction.

[Role]
  Build at most two finite phase classes (full chunk and a real partial tail)
  from the declared recurrent recipe work, then expose a constant-time indexed
  panel ABI to the existing bounded Engine template consumer.

[Invariants]
  * Per-chunk formulas are exactly the historical ones: matrix work scales with
    the valid token fraction, vector work keeps the declared per-chunk plus
    per-token split, and state/other work stays ``total/count``.
  * Caps, stream edges, setup, minimum provenance and one live state slot are
    preserved; keyword construction avoids the historical positional misbinding.
  * ``index_of`` is O(1) and selects only the full or tail class; no plain
    ``PipelineProgram.template()`` enumeration is used for long scans.

[Timeouts & Locks]
  Constant construction work independent of token count; no I/O or locks.

[Side-Effect Policy]
  Returns immutable program/template records only.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate, PipelineTemplateProgram, Stage


FULL_CLASS = 0
TAIL_CLASS = 1
STATE_SLOTS = 1
OUTPUT_ORDINAL = 0
PANEL_SLOT = 0
INDEXED_ABI_PANELS = 1


@dataclass(frozen=True)
class _ScanGeometry:
    count: int
    chunk: int
    tokens: int
    heads: int
    chunk_vector_ops: float


def _geometry(context) -> _ScanGeometry:
    metadata = dict(context.op.metadata)
    try:
        count = int(metadata["chunks"])
        chunk = int(metadata["chunk"])
        tokens = int(metadata["tokens"])
        heads = int(metadata["heads"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelingError("recurrent_scan_geometry_missing") from exc
    if count < 0 or chunk < 1 or tokens < 0 or heads < 1:
        raise ModelingError("recurrent_scan_geometry_invalid")
    expected = ((tokens + chunk - 1) // chunk) if tokens else 0
    if count != expected:
        raise ModelingError("recurrent_scan_chunk_count_mismatch")
    return _ScanGeometry(count, chunk, tokens, heads,
                         float(metadata.get("chunk_vector_ops", 0)))


def _tile_stages(context, geometry: _ScanGeometry, valid: int) -> tuple[Stage, ...]:
    matrix = context.resource_names["cube"]
    vector = context.resource_names["vector"]
    cap = {matrix: min(1, geometry.heads / context.core_count) * context.rates[matrix]}
    fraction = float(valid) / float(geometry.tokens) if geometry.tokens else 0
    work = {}
    for key, amount in context.work.items():
        if key == matrix:
            work[key] = amount * fraction
        elif key == vector and 0 < geometry.chunk_vector_ops <= amount:
            work[key] = (
                geometry.chunk_vector_ops * (1 / geometry.count)
                + (amount - geometry.chunk_vector_ops) * fraction)
        elif key == vector:
            work[key] = amount * fraction
        else:
            work[key] = amount * (1 / geometry.count)
    first = {matrix: work.get(matrix, 0) * 0.5}
    last = dict(work)
    last[matrix] = work.get(matrix, 0) * 0.5
    vector_work = {vector: last.pop(vector, 0)}
    return (
        Stage("c1", first, caps=cap),
        Stage("delta", vector_work, after=("c1",)),
        Stage("c2_update", last, after=("delta",), caps=cap),
    )


def recurrent_scan_classes(context) -> tuple[PipelinePhase, ...]:
    """The full and tail phase classes, with at most two entries."""
    geometry = _geometry(context)
    if geometry.count <= 0:
        return ()
    full_valid = min(geometry.chunk, geometry.tokens)
    phases = [PipelinePhase(
        "recurrent_full", _tile_stages(context, geometry, full_valid), 1)]
    tail = geometry.tokens - (geometry.count - 1) * geometry.chunk
    if 0 < tail < geometry.chunk:
        phases.append(PipelinePhase(
            "recurrent_tail", _tile_stages(context, geometry, tail), 1))
    return tuple(phases)


def lower_recurrent_scan(context) -> PipelineTemplateProgram:
    """Construct the indexed finite recurrent template in O(1) work."""
    geometry = _geometry(context)
    phases = recurrent_scan_classes(context)
    full_count = (geometry.count - 1
                  if len(phases) > 1 else geometry.count)
    total = geometry.count

    def index_of(index: int) -> tuple[int, int, int, int, int]:
        if index < 0 or index >= total:
            raise ModelingError(f"recurrent_template_index_out_of_range:{index}")
        if len(phases) > 1 and index >= full_count:
            return (TAIL_CLASS, total, OUTPUT_ORDINAL, index, PANEL_SLOT)
        return (FULL_CLASS, full_count, OUTPUT_ORDINAL, index, PANEL_SLOT)

    template = PipelineTemplate(
        owner=0, slots=STATE_SLOTS, phases=phases,
        setup_ms=context.setup_ms, minimum_ms=context.minimum_ms,
        indexed_panels=total, index_of=index_of,
        tile_factory_calls=len(phases), phase_objects=len(phases),
        static_phase_classes=len(phases),
    )
    return PipelineTemplateProgram(template)
