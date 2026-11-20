"""Sparse attention MTE1 work and measured service selection.

[Role]
  Match source and implementation conditions and price the critical owner.
[Invariants]
  Only applicable work geometries use the measured per-owner service rate.
  Query work is derived from QK and PV tile payloads.
[Timeouts & Locks]
  In-memory arithmetic. No I/O or locks.
[Side-Effect Policy]
  Reads supplied records and returns new work and service results.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


_SPARSE_OWNER_COUNT = 32
_SPARSE_GROUP_SIZE = 64
_SPARSE_QUERY_WIDTH = 576
_SPARSE_VALUE_WIDTH = 512
_SPARSE_KV_TILE_ROWS = 128
_SPARSE_KV_ROW_ALIGNMENT = 16


def sparse_mte1_bytes(q_len: int, context: int, kv_len: int, *, causal: bool = True,
                      owners: int = _SPARSE_OWNER_COUNT,
                      g: int = _SPARSE_GROUP_SIZE, dq: int = _SPARSE_QUERY_WIDTH,
                      dv: int = _SPARSE_VALUE_WIDTH,
                      s2_tile: int = _SPARSE_KV_TILE_ROWS) -> dict[str, Any]:
    """Derive per-critical-owner QK/PV L1->L0A/B bytes for one geometry."""
    q_len, context, kv_len, owners = int(q_len), int(context), int(kv_len), int(owners)
    if q_len <= 0 or owners <= 0 or kv_len <= 0 or context < 0:
        raise ValueError("sparse_mte1_geometry_invalid")
    width = -(-q_len // owners)
    per_owner = []
    for start in range(0, q_len, width):
        qk = pv = ub = tiles = 0
        for index in range(start, min(start + width, q_len)):
            selected = min(kv_len, context + index + 1) if causal else min(
                kv_len, context + q_len)
            remaining = selected
            while remaining > 0:
                rows = min(int(s2_tile), remaining)
                aligned = ((rows + _SPARSE_KV_ROW_ALIGNMENT - 1)
                           // _SPARSE_KV_ROW_ALIGNMENT) * _SPARSE_KV_ROW_ALIGNMENT
                qk += 2 * g * dq + 2 * aligned * dq
                pv += 2 * g * aligned + 2 * dv * aligned
                ub += 2 * g * s2_tile
                tiles += 1
                remaining -= int(s2_tile)
        per_owner.append({"qk_pv_bytes": qk + pv, "qk_bytes": qk, "pv_bytes": pv,
                          "ub_bytes": ub, "tiles": tiles})
    critical_index = max(range(len(per_owner)),
                         key=lambda index: per_owner[index]["qk_pv_bytes"])
    critical = per_owner[critical_index]
    return {
        "q_len": q_len, "context": context, "kv_len": kv_len, "causal": bool(causal),
        "owners": owners, "g": int(g), "dq": int(dq), "dv": int(dv),
        "s2_tile": int(s2_tile), "per_owner_width": width,
        "critical_owner_index": critical_index,
        "critical_qk_pv_bytes": critical["qk_pv_bytes"],
        "critical_qk_bytes": critical["qk_bytes"],
        "critical_pv_bytes": critical["pv_bytes"],
        "critical_ub_bytes": critical["ub_bytes"],
        "critical_tiles": critical["tiles"],
        "total_qk_pv_bytes": sum(item["qk_pv_bytes"] for item in per_owner),
        "total_ub_bytes": sum(item["ub_bytes"] for item in per_owner),
        "total_tiles": sum(item["tiles"] for item in per_owner),
    }


def _digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def identity_digest(value: Any) -> str:
    """Normalize one declared image/source digest exactly once."""
    return _digest(value)


def c7_applicability(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the original C7 raw-record applicability contract.

    The raw selector, normalized service and direct binding all consume these
    conditions.  Source identity, native implementation, run state, owner
    count and the measured-q work domain stay executable; they are not merely
    provenance.
    """
    if not isinstance(record, Mapping):
        raise ValueError("c7_applicability_record_requires_mapping")
    source = dict(record.get("source_identity") or {})
    measured_q = int(record.get("measured_q_tokens") or 0)
    conditions = {
        "hardware": str(record.get("hardware") or ""),
        "dtype": str(record.get("dtype") or "").lower(),
        "native_type": str(record.get("native_type") or ""),
        "phase": str(record.get("phase") or "").lower(),
        "op_state": str(record.get("op_state") or "").lower(),
        "owners": int(record.get("owners") or 0),
        "source_identity.source_scope": str(source.get("source_scope") or ""),
        "source_identity.image_digest": identity_digest(
            source.get("image_digest")),
        "geometry.q_len": {"min": measured_q},
    }
    optional = {}
    if record.get("mix_block_num") is not None:
        optional["mix_block_num"] = int(record.get("mix_block_num"))
    geometry = dict(record.get("geometry_binding") or {})
    return {
        "conditions": conditions,
        "optional_conditions": optional,
        "work_domain": {
            "measured_q_tokens": measured_q,
            "q_len_min": measured_q,
            "geometry_binding": geometry,
            "measured_critical_bytes": int(
                record.get("measured_critical_bytes") or 0),
            "owners": int(record.get("owners") or 0),
            "work_unit": "bytes",
            "rate_scope": "per_owner",
            "rate_bytes_per_s": float(record.get("rate_bytes_per_s") or 0.0),
        },
    }


def _c7_value(context, path):
    value = context
    for part in str(path).split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return _C7_MISSING
    return value


_C7_MISSING = object()


def _c7_condition_matches(path, actual, expected):
    if path.endswith("image_digest"):
        return identity_digest(actual) == identity_digest(expected)
    if path == "dtype":
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, Mapping) and set(expected) <= {"min", "max"}:
        try:
            number = float(actual)
            if "min" in expected and number < float(expected["min"]):
                return False
            if "max" in expected and number > float(expected["max"]):
                return False
            return True
        except (TypeError, ValueError):
            return False
    return actual == expected


def c7_applicability_matches(record: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    """Apply the same extracted conditions used by normalized selection."""
    applicability = c7_applicability(record)
    for path, expected in applicability["conditions"].items():
        actual = _c7_value(context, path)
        if actual is _C7_MISSING or not _c7_condition_matches(
                path, actual, expected):
            return False
    for path, expected in applicability["optional_conditions"].items():
        actual = _c7_value(context, path)
        if actual is _C7_MISSING or actual is None:
            continue
        if not _c7_condition_matches(path, actual, expected):
            return False
    return True


def select_resource_service(
    resource_services: Sequence[Mapping[str, Any]] | None, *,
    native_type: str = "KvQuantSparseFlashAttention",
    hardware: str, source_identity: Mapping[str, Any] | None,
    dtype: str = "bf16", q_len: int, context: int, kv_len: int,
    causal: bool = True, owners: int = _SPARSE_OWNER_COUNT, phase: str = "prefill",
    op_state: str = "dynamic", mix_block_num: int | None = None,
) -> dict[str, Any] | None:
    """Return the measured MTE1 service for one saturated query geometry."""
    q_len, owners = int(q_len), int(owners)
    applicability_context = {
        "hardware": str(hardware),
        "dtype": str(dtype),
        "native_type": str(native_type),
        "phase": str(phase),
        "op_state": str(op_state),
        "owners": owners,
        "source_identity": dict(source_identity or {}),
        "geometry": {"q_len": q_len},
        "mix_block_num": mix_block_num,
    }
    for record in resource_services or ():
        if str(record.get("resource")) != "mte1":
            continue
        if not c7_applicability_matches(record, applicability_context):
            continue
        try:
            geometry = sparse_mte1_bytes(
                q_len, int(context), int(kv_len), causal=bool(causal), owners=owners,
                g=int((record.get("geometry_binding") or {}).get("g", _SPARSE_GROUP_SIZE)),
                dq=int((record.get("geometry_binding") or {}).get("dq", _SPARSE_QUERY_WIDTH)),
                dv=int((record.get("geometry_binding") or {}).get("dv", _SPARSE_VALUE_WIDTH)),
                s2_tile=int((record.get("geometry_binding") or {}).get("s2_tile", _SPARSE_KV_TILE_ROWS)),
            )
            rate = float(record["rate_bytes_per_s"])
        except (KeyError, TypeError, ValueError):
            return None
        if rate <= 0:
            return None
        critical = int(geometry["critical_qk_pv_bytes"])
        return {
            "schema_version": str(record.get("schema_version") or ""),
            "resource": "mte1",
            "native_type": native_type,
            "source_identity": dict(source_identity),
            "hardware": str(hardware),
            "dtype": str(dtype),
            "layout": str(record.get("layout") or ""),
            "phase": str(phase),
            "op_state": str(op_state),
            "shape_class": str(record.get("shape_class") or ""),
            "owners": owners,
            "q_tokens": q_len,
            "context": int(context),
            "kv_len": int(kv_len),
            "causal": bool(causal),
            "measured_q_tokens": int(record.get("measured_q_tokens") or 0),
            "mix_block_num": record.get("mix_block_num"),
            "measured_critical_bytes": int(record.get("measured_critical_bytes") or 0),
            "query_critical_bytes": critical,
            "query_total_bytes": int(geometry["total_qk_pv_bytes"]),
            "rate_bytes_per_s": rate,
            "time_us": critical / rate * 1e6,
            "event_count": int(record.get("event_count") or 0),
            "median_mte1_us": float(record.get("median_mte1_us") or 0.0),
            "geometry_binding": dict(record.get("geometry_binding") or {}),
        }
    return None
