"""Public resolution of scoped C7 resource services (MTE1 sparse SFA).

[Role] Turn one profiler `resource_services` record into a query-geometry
  service time for the ordinary sparse attention consumer.
[Invariants] The record carries the R64 source identity, native kernel, dtype,
  hardware, owner count, measured critical bytes and measured per-critical-owner
  rate.  The query must be the same source/native/hardware/format, a saturated
  prefill class (q_len >= the measured training q), and the same owner count.
  A raw Q shape without this geometry never selects the service.
[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Pure result; no case or metadata mutation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def sparse_mte1_bytes(q_len: int, context: int, kv_len: int, *, causal: bool = True,
                      owners: int = 32, g: int = 64, dq: int = 576,
                      dv: int = 512, s2_tile: int = 128) -> dict[str, Any]:
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
                aligned = ((rows + 15) // 16) * 16
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


def select_resource_service(
    resource_services: Sequence[Mapping[str, Any]] | None, *,
    native_type: str = "KvQuantSparseFlashAttention",
    hardware: str, source_identity: Mapping[str, Any] | None,
    dtype: str = "bf16", q_len: int, context: int, kv_len: int,
    causal: bool = True, owners: int = 32, phase: str = "prefill",
    op_state: str = "dynamic", mix_block_num: int | None = None,
) -> dict[str, Any] | None:
    """Return the measured MTE1 service for one saturated query geometry."""
    if not source_identity or source_identity.get("source_scope") != "r64":
        return None
    if str(phase).lower() != "prefill" or str(op_state).lower() != "dynamic":
        return None
    q_len, owners = int(q_len), int(owners)
    for record in resource_services or ():
        if str(record.get("resource")) != "mte1":
            continue
        if str(record.get("native_type")) != native_type:
            continue
        if str(record.get("hardware")) != str(hardware):
            continue
        if str(record.get("dtype")).lower() != str(dtype).lower():
            continue
        rec_source = record.get("source_identity") or {}
        if str(rec_source.get("source_scope")) != "r64":
            continue
        if _digest(rec_source.get("image_digest")) != _digest(source_identity.get("image_digest")):
            continue
        if int(record.get("owners") or 0) != owners:
            continue
        if mix_block_num is not None and int(record.get("mix_block_num") or -1) != int(mix_block_num):
            continue
        if q_len < int(record.get("measured_q_tokens") or 0):
            return None
        try:
            geometry = sparse_mte1_bytes(
                q_len, int(context), int(kv_len), causal=bool(causal), owners=owners,
                g=int((record.get("geometry_binding") or {}).get("g", 64)),
                dq=int((record.get("geometry_binding") or {}).get("dq", 576)),
                dv=int((record.get("geometry_binding") or {}).get("dv", 512)),
                s2_tile=int((record.get("geometry_binding") or {}).get("s2_tile", 128)),
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


__all__ = ["select_resource_service", "sparse_mte1_bytes"]
