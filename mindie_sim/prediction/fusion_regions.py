"""Centralized subgraph-region metadata folding and dependency closure.

[Role]
  Maintain one deterministic rewriter for ``structure_subgraph_memberships``,
  ``structure_subgraph_entries``, ``structure_subgraphs``, ``fused_origin_scopes``
  and ``absorbed_source_ops`` whenever lowering passes fold semantic operators.

[Invariants]
  * Subgraph memberships and entry markers are unioned in deterministic order
    across all folded operands rather than overwritten by ``dict.update``.
  * Subgraph declarations coalesce by ``key``; conflicting definitions raise
    ``ModelingError``, and absorbed anchors re-anchor onto ``primary_anchor_id``.
  * Cross-subgraph composite folds retain ``complete_subgraph`` coverage while
    stamping ``composite_shared_subgraphs`` so downstream family and cohort
    accountants deduplicate physical ``op_id`` occurrences cleanly.

[Timeouts & Locks]
  Linear in the folded operand list and declared subgraphs; no locks or I/O.

[Side-Effect Policy]
  Pure functional metadata synthesis returning fresh dictionaries/tuples.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from mindie_sim.schema.common import ModelingError


def _effective_meta(op: Any) -> dict[str, Any]:
    raw = dict(op.metadata)
    semantic = raw.get("semantic_metadata")
    if isinstance(semantic, Mapping):
        return {**semantic, **raw}
    return raw


def _declarations_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"anchor", "coverage"}
    left_filtered = {k: v for k, v in left.items() if k not in ignored}
    right_filtered = {k: v for k, v in right.items() if k not in ignored}
    return left_filtered == right_filtered


def fold_region_metadata_dict(
    operands: Sequence[Any],
    *,
    primary_anchor_id: str | None = None,
    output_port_aliases: Mapping[str, Any] | None = None,
    extra: Sequence[tuple[str, Any]] | Mapping[str, Any] = (),
) -> dict[str, Any]:
    """Fold operand metadata while preserving subgraph memberships and anchors."""
    ops = tuple(operands)
    if not ops:
        return dict(extra)
    anchor_id = str(primary_anchor_id or ops[-1].op_id)
    merged: dict[str, Any] = {}
    origins_by_id: dict[str, dict[str, Any]] = {}
    absorbed_ids: list[str] = []
    memberships: list[str] = []
    entries: list[str] = []
    declarations_by_key: dict[str, dict[str, Any]] = {}
    declaration_order: list[str] = []

    for op in ops:
        raw = dict(op.metadata)
        eff = _effective_meta(op)
        merged.update(raw)
        for prev in eff.get("fused_origin_scopes", ()):
            if isinstance(prev, Mapping) and prev.get("op_id"):
                origins_by_id[str(prev["op_id"])] = dict(prev)
        origins_by_id[op.op_id] = {
            "op_id": op.op_id,
            "layer": op.layer,
            "stage": op.stage,
            "block": eff.get("structure_block"),
            "role": eff.get("structure_role"),
        }
        for prior_abs in eff.get("absorbed_source_ops", ()):
            prior_str = str(prior_abs)
            if prior_str and prior_str not in absorbed_ids:
                absorbed_ids.append(prior_str)
        if op.op_id not in absorbed_ids:
            absorbed_ids.append(op.op_id)
        for group_key in eff.get("structure_subgraph_memberships", ()):
            group_str = str(group_key)
            if group_str not in memberships:
                memberships.append(group_str)
        for entry_key in eff.get("structure_subgraph_entries", ()):
            entry_str = str(entry_key)
            if entry_str not in entries:
                entries.append(entry_str)
        for decl in eff.get("structure_subgraphs", ()):
            if not isinstance(decl, Mapping):
                continue
            key = str(decl["key"])
            copied = dict(decl)
            existing = declarations_by_key.get(key)
            if existing is not None:
                if not _declarations_compatible(existing, copied):
                    raise ModelingError(
                        f"fusion_region_conflicting_subgraph_declaration:{key}"
                    )
            else:
                declarations_by_key[key] = copied
                declaration_order.append(key)

    absorbed_set = set(absorbed_ids)
    composite_shared = len(ops) > 1 and len(memberships) > 1
    coalesced_decls: list[dict[str, Any]] = []
    for key in declaration_order:
        decl = dict(declarations_by_key[key])
        if decl.get("anchor") in absorbed_set:
            decl["anchor"] = anchor_id
        if composite_shared and key in memberships:
            cov = dict(decl.get("coverage") or {"kind": "complete_subgraph", "limitations": []})
            cov["composite_shared"] = True
            cov["shared_subgraphs"] = list(memberships)
            decl["coverage"] = cov
        coalesced_decls.append(decl)

    merged["fused_origin_scopes"] = tuple(origins_by_id.values())
    if len(ops) > 1:
        merged["absorbed_source_ops"] = tuple(absorbed_ids)
    if memberships:
        merged["structure_subgraph_memberships"] = tuple(memberships)
    else:
        merged.pop("structure_subgraph_memberships", None)
    if entries:
        merged["structure_subgraph_entries"] = tuple(entries)
    else:
        merged.pop("structure_subgraph_entries", None)
    if coalesced_decls:
        merged["structure_subgraphs"] = tuple(coalesced_decls)
    else:
        merged.pop("structure_subgraphs", None)
    if composite_shared:
        merged["composite_shared_subgraphs"] = tuple(memberships)
    if output_port_aliases:
        merged["output_port_aliases"] = tuple(
            (str(k), v) for k, v in dict(output_port_aliases).items()
        )

    semantic = merged.get("semantic_metadata")
    if isinstance(semantic, Mapping):
        synced = dict(semantic)
        for field in (
            "structure_subgraph_memberships",
            "structure_subgraph_entries",
            "structure_subgraphs",
        ):
            if field in merged:
                synced[field] = merged[field]
            else:
                synced.pop(field, None)
        merged["semantic_metadata"] = synced

    merged.update(dict(extra))
    return merged


def fold_region_metadata(
    operands: Sequence[Any],
    *,
    primary_anchor_id: str | None = None,
    output_port_aliases: Mapping[str, Any] | None = None,
    extra: Sequence[tuple[str, Any]] | Mapping[str, Any] = (),
) -> tuple[tuple[str, Any], ...]:
    """Tuple-encoded companion to ``fold_region_metadata_dict`` for ``OpSpec``."""
    return tuple(
        fold_region_metadata_dict(
            operands,
            primary_anchor_id=primary_anchor_id,
            output_port_aliases=output_port_aliases,
            extra=extra,
        ).items()
    )


def resolve_subgraph_groups(ops: Sequence[Any]):
    """Yield ``(declaration, ordered_member_ops)`` by backward dependency closure."""
    declarations: dict[str, dict[str, Any]] = {}
    seeds: dict[str, set[str]] = defaultdict(set)
    inputs: dict[str, set[str]] = defaultdict(set)
    by_id = {op.op_id: op for op in ops}
    positions = {op.op_id: index for index, op in enumerate(ops)}
    for op in ops:
        meta = _effective_meta(op)
        for declaration in meta.get("structure_subgraphs", ()):
            key = declaration["key"]
            previous = declarations.get(key)
            if previous is not None and not _declarations_compatible(previous, declaration):
                raise ModelingError("native_fusion_group_declaration_conflict:" + key)
            declarations[key] = declaration
        for key in meta.get("structure_subgraph_memberships", ()):
            seeds[key].add(op.op_id)
        for key in meta.get("structure_subgraph_entries", ()):
            inputs[key].update(op.deps)
    for key, declaration in declarations.items():
        members, pending = set(seeds[key]), list(seeds[key])
        while pending:
            current = pending.pop()
            for dependency in by_id[current].deps:
                if (
                    dependency in by_id
                    and dependency not in inputs[key]
                    and dependency not in members
                ):
                    members.add(dependency)
                    pending.append(dependency)
        yield declaration, [by_id[item] for item in sorted(members, key=positions.__getitem__)]


__all__ = [
    "fold_region_metadata",
    "fold_region_metadata_dict",
    "resolve_subgraph_groups",
]
