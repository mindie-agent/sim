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
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_registry

_DEFAULT_REPOSITORY = "vllm-project/vllm-ascend"
_ABI_ARCH_SEARCH_ORDER = ("A5", "A3", "A2")


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


@lru_cache(maxsize=_SOURCE_REF_CACHE_MAXSIZE)
def _lookup_sealed_source_ref(norm_path: str, norm_symbol: str) -> tuple[tuple[str, str], ...]:
    registry = default_abi_registry()
    for arch in _ABI_ARCH_SEARCH_ORDER:
        snapshot = registry.get_snapshot(arch)
        for entry in snapshot.entries:
            ref = entry.source_ref
            rel = str(ref.relative_path or "").strip().lstrip("./") if ref is not None else ""
            va_ref = str(entry.vllm_ascend_ref or "").strip().lstrip("./")
            path_match = bool(norm_path and (rel == norm_path or va_ref == norm_path))
            sym_match = bool(
                norm_symbol
                and (entry.symbol_or_kernel == norm_symbol or entry.abi_id == norm_symbol)
            )
            if path_match or sym_match:
                return (
                    ("repository", str(ref.repository if ref is not None else _DEFAULT_REPOSITORY)),
                    ("revision", str(ref.revision if ref is not None else "")),
                    ("relative_path", norm_path or rel),
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
