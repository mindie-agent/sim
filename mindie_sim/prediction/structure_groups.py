"""Declared native correspondences over existing semantic operator groups.

[Role] Retain fusion membership, scope and evidence at the composing builder.
[Invariants] Declarations use model/runtime geometry, never captured shapes.
  A group adds no execution node, work or measured duration. Zero hardcoded
  model-name source tables (AG014).
[Timeouts & Locks] Finite layer metadata updates; no I/O or locks.
[Side-Effect Policy] Replaces records in the caller-owned operator list.
"""
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

_LEGACY_PURGED_SOURCE_REF_MSG = (
    "LEGACY_PURGED: resolve portable SourceRef provenance from "
    "stacks/vllm_ascend/manifest.json without hardcoded _SOURCE_FILES model-name lookup"
)


class _PurgedSourceReference(Mapping[str, Any]):
    """Import-safe lazy socket; any runtime dereference fails loudly."""

    __slots__ = ("_path", "_symbol")

    def __init__(self, path: str, symbol: str) -> None:
        self._path = path
        self._symbol = symbol

    def _fail(self) -> Any:
        raise NotImplementedError(_LEGACY_PURGED_SOURCE_REF_MSG)

    def __getitem__(self, key: str) -> Any:
        return self._fail()

    def __iter__(self):
        return self._fail()

    def __len__(self) -> int:
        return self._fail()

    def __bool__(self) -> bool:
        return self._fail()

    def get(self, key: str, default: Any = None) -> Any:
        return self._fail()

    def copy(self) -> dict[str, Any]:
        return self._fail()


def declare_group(ops, *, key, role, family, members, entries, anchor, view,
                  implementation, source=None, block="attention", scope_role=None,
                  coverage=None, fusion_status=None, fusion_reason=None,
                  native_fusion_eligible=True):
    if isinstance(source, _PurgedSourceReference):
        raise NotImplementedError(_LEGACY_PURGED_SOURCE_REF_MSG)
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


def source_reference(path, symbol):
    """Unplugged socket: hardcoded _SOURCE_FILES model-name lookup purged per AG014.

    Returns an import-safe lazy socket so module-level bindings in external
    operators do not break ``import mindie_sim``, while any dereference or
    ``declare_group(..., source=...)`` call raises ``NotImplementedError("LEGACY_PURGED: ...")``.
    """
    return _PurgedSourceReference(path, symbol)

