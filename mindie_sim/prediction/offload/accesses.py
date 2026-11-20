"""State access records and DDR/SSD tier projections.

[Role] Normalize state access identities and project tier traffic.
[Invariants] Accesses retain their physical producers, storage and ranges.
[Timeouts & Locks] Bounded by selected operators and active accesses. No waits or locks.
[Side-Effect Policy] Builds records from declared state and operator metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .policy import OffloadPolicy


_INDEX_POOL_TAIL_COMPONENT_BYTES = 4


def _normalize_span(value) -> tuple[tuple[int, int], ...]:
    """Stable hashable range identity for mapping and dataclass inputs."""
    if not value:
        return ()
    normalized = []
    for item in value:
        if isinstance(item, (str, bytes)):
            raise ValueError("state_access_span_range_invalid")
        pair = tuple(item)
        if len(pair) != 2:
            raise ValueError("state_access_span_range_requires_pair")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end < start:
            raise ValueError("state_access_span_range_invalid")
        normalized.append((start, end))
    return tuple(normalized)


@dataclass(frozen=True)
class StateAccess:
    """Instance binding of one real state read/write/alias to an occurrence.

    This is a private projection over existing OpSpec/TensorValue/StateEffect
    data, not a new public IR.  ``bytes`` is the actual issued transfer volume
    for this direction/span; ``read_bytes``/``write_bytes`` are retained for
    callers that carry one record per direction.
    """

    occurrence: str
    producer: str
    request: str
    stage: int
    layer: int
    direction: str
    position: str = "cache"
    output: str = ""
    storage: str = ""
    state_version: int | None = None
    span: tuple[tuple[int, int], ...] = ()
    rank: int | None = None
    scope: str = "target"
    bytes: float = 0.0
    source: str = ""
    reader: str = ""
    grant: str = ""
    decision: str = ""
    last_use: str = ""

    def __post_init__(self):
        object.__setattr__(self, "span", _normalize_span(self.span))
        if self.direction not in {"read", "write", "alias"}:
            raise ValueError(f"state_access_direction_invalid:{self.direction}")
        if not self.occurrence or not self.producer:
            raise ValueError("state_access_requires_occurrence_and_producer")
        if isinstance(self.bytes, bool) or not math.isfinite(float(self.bytes)) or float(self.bytes) < 0:
            raise ValueError("state_access_bytes_invalid")
        object.__setattr__(self, "bytes", float(self.bytes))

    @property
    def key(self) -> tuple:
        # One outer occurrence can carry several physical writers.  The
        # producer and stage are part of the physical access identity so two
        # different stores under the same call are not collapsed into one
        # issued write.
        return (self.occurrence, self.producer, self.stage, self.output,
                self.storage, self.state_version, self.span, self.direction,
                self.rank, self.request, self.scope)

    def to_dict(self) -> dict:
        return {
            "occurrence": self.occurrence, "producer": self.producer,
            "request": self.request, "stage": self.stage, "layer": self.layer,
            "direction": self.direction, "position": self.position,
            "output": self.output, "storage": self.storage,
            "state_version": self.state_version, "span": self.span,
            "rank": self.rank, "scope": self.scope, "bytes": self.bytes,
            "source": self.source, "reader": self.reader, "grant": self.grant,
            "decision": self.decision, "last_use": self.last_use,
        }


def _as_access(value: Any) -> StateAccess:
    if isinstance(value, StateAccess):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        if "bytes" not in data:
            data["bytes"] = data.get("write_bytes", data.get("read_bytes", 0.0))
        data.pop("read_bytes", None)
        data.pop("write_bytes", None)
        data.setdefault("position", "cache")
        data.setdefault("output", "")
        data.setdefault("storage", "")
        data.setdefault("scope", "target")
        data["span"] = _normalize_span(data.get("span", ()))
        data.setdefault("rank", None)
        data.setdefault("state_version", None)
        data.setdefault("source", "")
        data.setdefault("reader", "")
        data.setdefault("grant", "")
        data.setdefault("decision", "")
        data.setdefault("last_use", "")
        return StateAccess(**data)
    raise TypeError("state_access_mapping_or_StateAccess_required")


def _as_accesses(values) -> tuple[StateAccess, ...]:
    return tuple(_as_access(value) for value in (values or ()))


def _unique_accesses(accesses) -> tuple[StateAccess, ...]:
    seen = set()
    result = []
    for access in accesses:
        if access.key in seen:
            continue
        seen.add(access.key)
        result.append(access)
    return tuple(result)


def _access_bytes(accesses, direction: str) -> float:
    return float(sum(access.bytes for access in _unique_accesses(accesses)
                     if access.direction == direction))


def _physical_producer_output(meta: Mapping, occurrence: str,
                              default_output: str = "") -> tuple[str, str]:
    """Read the producer/output port carried by the selected op.

    Only producer-owned fields are consumed.  A generic metadata key named
    ``producer`` is not guessed to be a state port; explicit lowering owners
    supply ``state_producer``/``state_output`` or ``state_producer_output``.
    When none is present the selected occurrence itself is the writer.
    """
    raw = meta.get("state_producer_output")
    producer = ""
    output = ""
    if isinstance(raw, Mapping):
        producer = str(raw.get("producer") or "")
        output = str(raw.get("output") or "")
    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
        producer, output = str(raw[0] or ""), str(raw[1] or "")
    producer = producer or str(
        meta.get("state_producer") or meta.get("cache_producer") or "")
    output = output or str(
        meta.get("state_output") or meta.get("cache_output") or "")
    return producer or occurrence, output or default_output


def _accesses_from_ops(work: Any, physical_ops=None) -> tuple[StateAccess, ...]:
    """Compatibility binding from real selected op metadata only.

    It never walks the model layer list, a token total or a capacity delta.
    Missing state-byte/output/storage bindings stay absent; the lowering owner
    must supply them through the explicit ``state_accesses`` argument.
    """
    ops = (tuple(physical_ops) if physical_ops is not None
           else tuple(getattr(work, "ops", ()) or ()))
    if not ops:
        return ()
    plan = getattr(work, "plan", None)
    padding_ids = {str(view.request_id) for view in
                   (getattr(plan, "requests", ()) or ())
                   if getattr(view, "padding", False)}
    result = []
    for op in ops:
        meta = dict(getattr(op, "metadata", ()) or ())
        stage = int(getattr(op, "stage", 0))
        layer = int(getattr(op, "layer", -1))
        raw_requests = tuple(getattr(op, "requests", ()) or ("*",))
        requests = tuple(request for request in raw_requests
                         if str(request) not in padding_ids)
        if not requests:
            continue
        scope = "draft" if meta.get("producer_scope") == "draft" else "target"
        occurrence = str(getattr(op, "op_id", ""))
        kind = str(meta.get("cache_kind", ""))
        producer, bound_output = _physical_producer_output(
            meta, occurrence, kind)
        if kind == "index_pool_tail":
            retained = int(meta.get("retained_rows", 0) or 0)
            width = int(meta.get("index_width", 0) or 0)
            write = float(meta.get("write_bytes", 0.0) or 0.0)
            if write <= 0 and retained and width:
                write = float(retained * width * 2 * _INDEX_POOL_TAIL_COMPONENT_BYTES)
            if write <= 0:
                write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
            if write > 0:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=requests[0] if requests else "*", stage=stage,
                    layer=layer, direction="write", position="tail",
                    output=bound_output or kind, storage=kind, bytes=write,
                    span=((0, retained),) if retained else (),
                    scope=scope, source=producer,
                ))
            continue
        if bool(meta.get("shared_owner", False)):
            continue
        if kind and str(getattr(op, "family", "")) == "cache_store":
            write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
            if write <= 0:
                write = float(getattr(op, "hbm_bytes", 0.0) or 0.0) / 2.0
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="write", position="cache",
                    output=bound_output or kind, storage=kind,
                    bytes=write / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
            continue
        contract = meta.get("gdn_state_contract")
        if contract:
            # The original-slot contract is the physical access authority.
            # Aggregate state_read/write_bytes are validated against a
            # per-slot formula when the producer declared one; a conv contract
            # that only declares state_write_bytes explicitly is split across
            # its real active slots instead of being dropped.
            state_dtype = str(contract.get("state_dtype")
                              or meta.get("state_dtype") or "fp32")
            heads = int(meta.get("heads", contract.get("heads", 0)) or 0)
            dk = int(meta.get("dk", contract.get("dk", 0)) or 0)
            dv = int(meta.get("dv", contract.get("dv", 0)) or 0)
            element = float(meta.get("state_element_bytes", 0.0) or 0.0)
            if not element:
                from mindie_sim.schema.common import dtype_bytes
                element = float(dtype_bytes(state_dtype))
            state_bytes = float(heads * dk * dv) * element
            read_versions = tuple(contract.get("read_versions", ()))
            checkpoints = tuple(contract.get("checkpoints", ()))
            targets = tuple(contract.get("targets", ()))
            publish_versions = tuple(contract.get("publish_versions", ()))
            slot_rows = []
            for slot, request in enumerate(raw_requests):
                if str(request) in padding_ids:
                    continue
                slot_rows.append({
                    "slot": slot, "request": request,
                    "target": (targets[slot] if slot < len(targets)
                               else (bound_output or "state")),
                    "read_version": (read_versions[slot]
                                     if slot < len(read_versions) else None),
                    "publish_version": (publish_versions[slot]
                                        if slot < len(publish_versions)
                                        else None),
                    "checkpoint": (int(checkpoints[slot])
                                   if slot < len(checkpoints) else 0),
                })
            for row in slot_rows:
                row["read"] = (state_bytes
                               if state_bytes > 0.0
                               and row["read_version"] is not None else 0.0)
                row["write"] = (state_bytes * row["checkpoint"]
                                if state_bytes > 0.0 else 0.0)
            aggregate_read = float(meta.get("state_read_bytes", 0.0) or 0.0)
            aggregate_write = float(meta.get("state_write_bytes", 0.0) or 0.0)
            if state_bytes > 0.0:
                contract_read = sum(row["read"] for row in slot_rows)
                contract_write = sum(row["write"] for row in slot_rows)
                for direction, aggregate, total in (
                        ("read", aggregate_read, contract_read),
                        ("write", aggregate_write, contract_write)):
                    if aggregate > 0.0 and abs(aggregate - total) > 1e-9:
                        raise ValueError(
                            f"state_access_contract_total_mismatch:{direction}:"
                            f"{aggregate}:{total}")
            else:
                if slot_rows:
                    if aggregate_read > 0.0:
                        for row in slot_rows:
                            row["read"] = aggregate_read / len(slot_rows)
                    if aggregate_write > 0.0:
                        for row in slot_rows:
                            row["write"] = aggregate_write / len(slot_rows)
            for row in slot_rows:
                if row["read"] > 0.0:
                    result.append(StateAccess(
                        occurrence=occurrence, producer=producer,
                        request=row["request"], stage=stage, layer=layer,
                        direction="read", position="state",
                        output=row["target"],
                        storage=str(contract.get("state_layout", "state")),
                        state_version=row["read_version"], span=((0, 1),),
                        scope=scope, bytes=row["read"], source=producer))
                if row["write"] > 0.0:
                    result.append(StateAccess(
                        occurrence=occurrence, producer=producer,
                        request=row["request"], stage=stage, layer=layer,
                        direction="write", position="state",
                        output=row["target"],
                        storage=str(contract.get("state_layout", "state")),
                        state_version=row["publish_version"],
                        span=((0, 1),), scope=scope, bytes=row["write"],
                        source=producer, decision=producer))
            continue
            continue
        read = float(meta.get("state_read_bytes", 0.0) or 0.0)
        write = float(meta.get("state_write_bytes", 0.0) or 0.0)
        if read > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="read", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_read_versions") or (None,))[0]
                    if isinstance(meta.get("state_read_versions"), (tuple, list)) else None,
                    bytes=read / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
        if write > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="write", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_commit_versions") or (None,))[0]
                    if isinstance(meta.get("state_commit_versions"), (tuple, list)) else None,
                    bytes=write / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
    return tuple(result)


def project_state_tiers(
    accesses: Any, policy: OffloadPolicy, *, expected_bytes: float | None = None,
) -> tuple[dict, ...]:
    """Project bound accesses into per-occurrence issued tier bytes.

    The returned records preserve request/rank/occurrence/output/storage/
    version/span/direction and the source/reader/grant/decision/last-use
    references.  No model layer or whole-request total is consulted.
    """
    bound = _unique_accesses(_as_accesses(accesses))
    if expected_bytes is not None:
        if (isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, (int, float))
                or not math.isfinite(float(expected_bytes))
                or float(expected_bytes) < 0.0):
            raise ValueError("state_access_total_bytes_invalid")
        if abs(float(expected_bytes) - sum(access.bytes for access in bound)) > 1e-9:
            raise ValueError("state_access_total_bytes_mismatch")
    records = []
    for access in bound:
        read = access.direction == "read"
        scale = (1.0 - policy.mtp_kvc_hit_rate) if read else 1.0
        ddr = access.bytes * policy.kvc_ddr_ratio * scale
        ssd = access.bytes * policy.kvc_ssd_ratio * scale
        records.append({
            **access.to_dict(),
            "ddr_bytes": ddr, "ssd_bytes": ssd, "hbm_bytes": ddr + ssd,
        })
    return tuple(records)
