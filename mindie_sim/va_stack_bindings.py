"""CANN 9.3 + ``vllm-ascend@405617bae`` Layer-2 stack bindings and provenance.

[Role]
  Load the checked-in ``mindie_sim/stack_sources/vllm_ascend_cann93.json``
  manifest and export ``VLLM_ASCEND_CANN93_BINDINGS`` (``tuple[StackMechanismBinding, ...]``),
  ``VLLM_ASCEND_CANN93_PROVENANCE``, and ``PROMOTED_MERGED_PRS`` for
  ``stack_tracks.py`` and ``scripts/audit_stack_prs.py``.

[Invariants]
  * Every ``merged`` and ``unmerged_pr`` entry carries an exact public
    ``{repository, revision, relative_path, source_sha256}`` provenance record.
  * Enforces ``_A5_PROFILE`` capability truth (omits ``GRAPH_QKV_NORM_ROPE_FUSION``
    from generic A5 FX passes, removes ``#21543`` from A5 MXFP, scopes ``#22128``
    to A3 W4A8 INT, removes ``deepseek-v4-flash-0731`` from ``#22142``, and
    removes ``gemma-4-*`` from ``#21449``).

[Timeouts & Locks]
  Single bounded local JSON read (< 1 MiB) cached in memory; no locks or network.

[Side-Effect Policy]
  Read-only local manifest loading; no filesystem writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mindie_sim import feature_mechanisms as _feature_mechanisms  # noqa: F401

MANIFEST_PATH = Path(__file__).resolve().parent / "stack_sources" / "vllm_ascend_cann93.json"
VLLM_ASCEND_BASELINE_REVISION = "405617bae227fe5b097ff163efaa60bb7f22c698"
CANN_TOOLKIT_REVISION = "9.3.0-20261112_192141454"

_RAW_MANIFEST: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

PROMOTED_MERGED_PRS: tuple[dict[str, Any], ...] = tuple(
    dict(entry) for entry in (_RAW_MANIFEST.get("promoted_merged_prs") or ())
)

VLLM_ASCEND_CANN93_BINDING_SPECS: tuple[dict[str, Any], ...] = tuple(
    dict(entry) for entry in (_RAW_MANIFEST.get("bindings") or ())
)

VLLM_ASCEND_CANN93_PROVENANCE: tuple[dict[str, Any], ...] = tuple(
    {
        "mechanism_id": entry["mechanism_id"],
        "state": entry["state"],
        "ref": entry.get("ref"),
        **dict(entry.get("provenance") or {}),
    }
    for entry in VLLM_ASCEND_CANN93_BINDING_SPECS
)

_CACHED_BINDINGS: tuple[Any, ...] | None = None


def build_vllm_ascend_cann93_bindings(binding_cls: Any = None) -> tuple[Any, ...]:
    """Instantiate ``StackMechanismBinding`` objects for all CANN 9.3 bindings."""
    if binding_cls is None:
        from mindie_sim.stack_tracks import StackMechanismBinding as binding_cls
    from mindie_sim.va_profiles import VA_RECOMMENDED_PROFILES

    documented_models = tuple(sorted(VA_RECOMMENDED_PROFILES))
    scoped_main_defaults = {"aclgraph_decode_capture", "sequence_parallel"}
    items = []
    for raw in VLLM_ASCEND_CANN93_BINDING_SPECS:
        models = tuple(str(m) for m in (raw.get("models") or ()))
        if not models and raw["state"] == "merged" and raw["mechanism_id"] in scoped_main_defaults:
            models = documented_models
        items.append(binding_cls(
            mechanism_id=str(raw["mechanism_id"]),
            state=str(raw["state"]),
            ref=raw.get("ref"),
            models=models,
            archs=tuple(str(a).lower() for a in (raw.get("archs") or ())),
            phases=tuple(str(p).lower() for p in (raw.get("phases") or ())),
            options=dict(raw.get("options") or {}),
            notes=str(raw.get("notes") or ""),
        ))
    return tuple(items)


def _try_eager_bindings() -> tuple[Any, ...] | None:
    try:
        from mindie_sim import stack_tracks
        cls = getattr(stack_tracks, "StackMechanismBinding", None)
        if cls is not None:
            return build_vllm_ascend_cann93_bindings(cls)
    except ImportError:
        return None
    return None


_CACHED_BINDINGS = _try_eager_bindings()
if _CACHED_BINDINGS is not None:
    VLLM_ASCEND_CANN93_BINDINGS = _CACHED_BINDINGS


def __getattr__(name: str) -> Any:
    global _CACHED_BINDINGS
    if name == "VLLM_ASCEND_CANN93_BINDINGS":
        if _CACHED_BINDINGS is None:
            _CACHED_BINDINGS = build_vllm_ascend_cann93_bindings()
        return _CACHED_BINDINGS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CANN_TOOLKIT_REVISION",
    "MANIFEST_PATH",
    "PROMOTED_MERGED_PRS",
    "VLLM_ASCEND_BASELINE_REVISION",
    "VLLM_ASCEND_CANN93_BINDINGS",
    "VLLM_ASCEND_CANN93_BINDING_SPECS",
    "VLLM_ASCEND_CANN93_PROVENANCE",
    "build_vllm_ascend_cann93_bindings",
]
