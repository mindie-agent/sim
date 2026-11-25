"""Declared native correspondences over existing semantic operator groups.

[Role] Retain fusion membership, scope and evidence at the composing builder,
  resolving portable ``SourceRef`` provenance via ``ProbedAbiRegistry``.
[Invariants] Declarations use model/runtime geometry, never captured shapes.
  A group adds no execution node, work or measured duration. Zero hardcoded
  model-name source tables (AG014).
[Timeouts & Locks] Finite layer metadata updates; cached read-only ABI lookup;
  no locks.
[Side-Effect Policy] Replaces records in the caller-owned operator list.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_dir, default_abi_registry

_DEFAULT_REPOSITORY = "vllm-project/vllm-ascend"
_ABI_ARCH_SEARCH_ORDER = ("A5", "A3", "A2")
_VLLM_ASCEND_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "stacks" / "vllm_ascend" / "manifest.json"
)
_MAX_MANIFEST_BYTES = 50 * 1024 * 1024


def declare_group(ops, *, key, role, family, members, entries, anchor, view,
                  implementation, source=None, block="attention", scope_role=None,
                  coverage=None, fusion_status=None, fusion_reason=None,
                  native_fusion_eligible=True):
    declaration = {
        "key": key, "role": role, "family": family,
        "op_ids": tuple(members), "entry_op_ids": tuple(entries),
        "tensor_view": view, "implementation": implementation, "source": source,
    }
    if coverage is not None:
        declaration["coverage"] = coverage
    if fusion_status is not None:
        declaration["fusion_status"] = fusion_status
    if fusion_reason is not None:
        declaration["fusion_reason"] = fusion_reason
    if not native_fusion_eligible:
        declaration["native_fusion_eligible"] = False
    selected = set(members)
    for index, op in enumerate(ops):
        if op.op_id not in selected:
            continue
        meta = dict(op.metadata)
        if block is not None:
            meta["structure_block"] = block
        if scope_role is not None:
            meta["structure_role"] = scope_role
        meta["structure_subgraph_memberships"] = (
            *meta.get("structure_subgraph_memberships", ()), key)
        if op.op_id in entries:
            meta["structure_subgraph_entries"] = (
                *meta.get("structure_subgraph_entries", ()), key)
        if op.op_id == anchor:
            meta["structure_subgraphs"] = (*meta.get("structure_subgraphs", ()), declaration)
        ops[index] = replace(op, metadata=tuple(meta.items()))


_SOURCE_REF_CACHE_MAXSIZE = 256


@lru_cache(maxsize=1)
def _load_manifest_provenance_index() -> dict[str, tuple[str, str, str]]:
    """Index ``relative_path -> (repository, revision, source_sha256)`` without L4 imports."""
    if not _VLLM_ASCEND_MANIFEST_PATH.is_file():
        return {}
    try:
        if _VLLM_ASCEND_MANIFEST_PATH.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        raw = json.loads(_VLLM_ASCEND_MANIFEST_PATH.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    default_rev = str(raw.get("manifest_revision") or "")
    index: dict[str, tuple[str, str, str]] = {}
    for item in raw.get("bindings") or ():
        if not isinstance(item, dict):
            continue
        prov = item.get("provenance")
        if not isinstance(prov, dict):
            continue
        rel = str(prov.get("relative_path") or "").strip().lstrip("./")
        sha = str(prov.get("source_sha256") or "").strip()
        if rel and sha and rel not in index:
            index[rel] = (
                str(prov.get("repository") or _DEFAULT_REPOSITORY),
                str(prov.get("revision") or default_rev),
                sha,
            )
    return index


@lru_cache(maxsize=_SOURCE_REF_CACHE_MAXSIZE)
def _lookup_sealed_source_ref(norm_path: str, norm_symbol: str) -> tuple[tuple[str, str], ...]:
    if norm_path:
        hit = _load_manifest_provenance_index().get(norm_path)
        if hit is not None:
            repo, rev, sha = hit
            return (
                ("repository", repo),
                ("revision", rev),
                ("relative_path", norm_path),
                ("symbol", norm_symbol),
                ("source_sha256", sha),
                ("provenance", "stack_manifest"),
            )
    if (default_abi_dir() / "abi_schema_manifest.json").is_file():
        registry = default_abi_registry()
        for arch in _ABI_ARCH_SEARCH_ORDER:
            snapshot = registry.get_snapshot(arch)
            for entry in snapshot.entries:
                ref = entry.source_ref
                rel = str(ref.relative_path or "").strip().lstrip("./") if ref is not None else ""
                path_match = bool(norm_path and rel == norm_path)
                sym_match = bool(
                    not norm_path
                    and norm_symbol
                    and (entry.symbol_or_kernel == norm_symbol or entry.abi_id == norm_symbol)
                )
                if path_match or sym_match:
                    return (
                        ("repository", str(ref.repository if ref is not None else _DEFAULT_REPOSITORY)),
                        ("revision", str(ref.revision if ref is not None else "")),
                        ("relative_path", rel or norm_path),
                        ("symbol", norm_symbol),
                        ("source_sha256", str(ref.source_sha256 if ref is not None else "")),
                        ("provenance", "abi_registry"),
                    )
    return (
        ("repository", _DEFAULT_REPOSITORY),
        ("revision", ""),
        ("relative_path", norm_path),
        ("symbol", norm_symbol),
        ("source_sha256", ""),
        ("provenance", "declared_path"),
    )


def source_reference(path: str, symbol: str) -> dict[str, Any]:
    """Resolve portable ``SourceRef`` metadata from ``ProbedAbiRegistry`` (AG014=0)."""
    norm_path = str(path or "").strip().lstrip("./")
    norm_symbol = str(symbol or "").strip()
    return dict(_lookup_sealed_source_ref(norm_path, norm_symbol))
