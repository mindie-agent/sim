"""Correlate predicted joint collectives with genuine native calls.

[Role]
  Accept an optional, externally prepared participant profile and select the
  native call that corresponds to one predicted joint collective.
[Invariants]
  * The profile is selection metadata only.  Observed entry/completion times,
    waits, durations and clocks never become predicted ready or service time.
  * A missing dtype/group/rank count stays underdetermined; it is not guessed.
  * Selection never changes execution; it only annotates a retained observation.
[Timeouts & Locks]
  Pure in-memory validation of a bounded caller-provided mapping.
[Side-Effect Policy]
  Reads the supplied mapping and returns a new plain dictionary.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

SCHEMA = "mindie-sim/collective-participant-profile/1"
_PRIMITIVES = {
    "allgather": "all_gather", "all_gather": "all_gather",
    "reducescatter": "reduce_scatter", "reduce_scatter": "reduce_scatter",
    "alltoall": "all_to_all", "all_to_all": "all_to_all",
    "alltoallv": "all_to_all", "allreduce": "all_reduce",
    "all_reduce": "all_reduce",
}


def _primitive(value: Any) -> str | None:
    token = str(value or "").strip().lower().replace("-", "_")
    return _PRIMITIVES.get(token)


def _dtype(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    return {"bfp16": "bf16", "bfloat16": "bf16", "float16": "fp16",
            "fp16": "fp16", "float32": "fp32", "fp32": "fp32",
            "fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3",
            "fp8_e5m2": "fp8_e5m2"}.get(token, token or None)


def normalise(value: Any) -> dict[str, Any] | None:
    """Validate the optional profile without inventing absent fields."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ModelingError("collective_participant_profile_must_be_mapping")
    doc = value.get("participant_profile", value)
    if not isinstance(doc, Mapping):
        raise ModelingError("collective_participant_profile_must_be_mapping")
    schema = doc.get("schema")
    if schema not in (None, SCHEMA):
        raise ModelingError(f"collective_participant_profile_schema_unknown:{schema}")
    calls = []
    for raw in doc.get("calls") or ():
        if not isinstance(raw, Mapping):
            raise ModelingError("collective_participant_profile_call_must_be_mapping")
        primitive = _primitive(raw.get("primitive") or raw.get("collective"))
        if primitive is None:
            raise ModelingError(
                f"collective_participant_profile_primitive_unknown:{raw.get('primitive')}")
        ordinal = raw.get("ordinal")
        if ordinal is not None:
            try:
                ordinal = int(ordinal)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_ordinal_invalid:{ordinal}") from None
        group_size = raw.get("group_size")
        if group_size is not None:
            try:
                group_size = int(group_size)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_group_invalid:{group_size}") from None
        ranks = []
        for item in raw.get("ranks") or raw.get("participants") or ():
            if not isinstance(item, Mapping):
                raise ModelingError(
                    "collective_participant_profile_rank_must_be_mapping")
            rank_id = item.get("rank_id")
            if rank_id is None:
                raise ModelingError("collective_participant_profile_rank_id_missing")
            try:
                rank_id = int(rank_id)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"collective_participant_profile_rank_id_invalid:{rank_id}") from None
            ranks.append({
                "rank_id": rank_id,
                "role": str(item.get("role") or ""),
            })
        calls.append({
            "ordinal": ordinal,
            "primitive": primitive,
            "native_name": (str(raw["native_name"])
                            if raw.get("native_name") is not None else None),
            "group_size": group_size,
            "dtype": _dtype(raw.get("dtype")),
            "ranks": tuple(ranks),
        })
    return {
        "schema": SCHEMA,
        "source": dict(doc.get("source") or {}),
        "calls": tuple(calls),
    }


def select_call(profile: Mapping[str, Any] | None, *, primitive: Any,
                occurrence: int = 0, group_size: int | None = None,
                dtype: Any = None) -> dict[str, Any]:
    """Select one genuine call by primitive and occurrence, never by time."""
    if profile is None:
        return {"resolved": False, "reason": "participant_profile_absent"}
    family = _primitive(primitive)
    if family is None:
        return {"resolved": False, "reason": f"primitive_unknown:{primitive}"}
    matching = [call for call in profile.get("calls", ())
                if call["primitive"] == family]
    if not matching:
        return {"resolved": False, "reason": f"no_source_call_for_primitive:{family}"}
    if occurrence < 0 or occurrence >= len(matching):
        return {"resolved": False,
                "reason": f"source_call_occurrence_out_of_range:{occurrence}:"
                          f"available={len(matching)}"}
    call = matching[occurrence]
    if (group_size is not None and call.get("group_size") is not None
            and int(call["group_size"]) != int(group_size)):
        return {"resolved": False,
                "reason": f"source_group_size_mismatch:{call['group_size']}!={group_size}"}
    expected = _dtype(dtype)
    if (expected is not None and call.get("dtype") is not None
            and call["dtype"] != expected):
        return {"resolved": False,
                "reason": f"source_dtype_mismatch:{call['dtype']}!={expected}"}
    return {
        "resolved": True,
        "basis": "same_primitive_occurrence",
        "primitive": family,
        "occurrence": occurrence,
        "ordinal": call.get("ordinal"),
        "native_name": call.get("native_name"),
        "group_size": call.get("group_size"),
        "dtype": call.get("dtype"),
        "dtype_basis": ("declared_source" if call.get("dtype") is not None
                        else "underdetermined_source"),
        "ranks": [dict(item) for item in call.get("ranks", ())],
    }
