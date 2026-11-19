"""Declared composite boundaries projected into observed match targets.

[Role]
  Project an exported ``mindie-model-structure/1`` payload into node, subgraph
  and native-call target indices without duplicating any model logic.

[Invariants]
  * A composite consumes its entire declared member set; a scope containing
    only one member cannot identify the whole fusion.
  * Common structure across candidates does not prove an individual layer,
    invocation or captured occurrence.
  * Rows remain attribute-bearing carriers; this module never turns them into
    mappings or invents missing members.
  * ``StructureTargetError`` is the only carrier-validation failure; malformed
    payloads never become silent empty target sets.

[Timeouts & Locks]
  Bounded by the supplied structure and its distinct candidate sets; no waits
  or locks.

[Side-Effect Policy]
  Pure in-memory metadata processing; no model imports, device I/O or mutation
  of the supplied carrier.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .carriers import _snapshot_structure_carrier
from .errors import StructureTargetError as ProfilerUsageError


class StructureTargets:
    def __init__(self, nodes, subgraphs, native_calls=None, *, _validated: bool = False):
        if not _validated:
            nodes = _snapshot_structure_carrier(nodes, "structure nodes")
            subgraphs = _snapshot_structure_carrier(subgraphs, "structure subgraphs")
            if native_calls is not None:
                native_calls = _snapshot_structure_carrier(native_calls, "structure native_calls")
        self.nodes = nodes
        self.subgraphs = {}
        for record in subgraphs:
            key = record.get("subgraph_id")
            members = record.get("node_ids", ())
            if not key or key in nodes or key in self.subgraphs:
                raise ProfilerUsageError(f"structure subgraph_id missing or duplicated: {key}")
            if not members or len(set(members)) != len(members) or any(
                member not in nodes for member in members
            ):
                raise ProfilerUsageError(f"subgraph has invalid semantic members: {key}")
            for field in ("replica", "stage"):
                if any(nodes[member].get(field) != record.get(field) for member in members):
                    raise ProfilerUsageError(f"subgraph crosses declared {field}: {key}")
            rec_phase = record.get("phase")
            for member in members:
                mem_phase = nodes[member].get("phase")
                if (
                    mem_phase != rec_phase
                    and not {mem_phase, rec_phase} <= {"decode", "draft"}
                    and not {mem_phase, rec_phase} <= {"prefill", "draft", "draft_prefill"}
                ):
                    raise ProfilerUsageError(f"subgraph crosses declared phase: {key}")
            if record.get("layer") is not None and any(
                nodes[member].get("layer") != record["layer"] for member in members
            ):
                raise ProfilerUsageError(f"subgraph has inconsistent declared layer: {key}")
            self.subgraphs[key] = record
        self.all = {**nodes, **self.subgraphs}
        self.native_implementations = defaultdict(list)
        self.native_calls = defaultdict(list)
        self.native_call_entries = defaultdict(list)
        self.native_call_kernels = defaultdict(list)
        self.native_call_observed = defaultdict(list)
        self.native_call_alias_names = set()
        self.native_bindings = {}
        if native_calls is not None and (
            not isinstance(native_calls, Sequence) or isinstance(native_calls, (str, bytes))
        ):
            raise ProfilerUsageError("structure native_calls must be a sequence of mappings")
        for record in native_calls or ():
            if not isinstance(record, Mapping):
                raise ProfilerUsageError("structure native_calls entries must be mappings")
            node_id = record.get("node_id")
            if not isinstance(node_id, str) or not node_id or node_id not in self.nodes:
                raise ProfilerUsageError(
                    f"structure native_calls references unknown node_id: {node_id!r}"
                )
            call_id = record.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProfilerUsageError(
                    f"structure native_calls entry for {node_id} requires non-empty call_id"
                )
            prev_binding = self.native_bindings.get(node_id)
            if prev_binding is not None and prev_binding.get("call_id") != call_id:
                raise ProfilerUsageError(
                    f"structure native_calls has conflicting bindings for {node_id}"
                )
            self.native_bindings[node_id] = record
            if node_id not in self.native_calls[call_id]:
                self.native_calls[call_id].append(node_id)
            api = record.get("api")
            if api is not None and not isinstance(api, Mapping):
                raise ProfilerUsageError(f"structure native_calls api must be a mapping: {node_id}")
            api = api or {}
            for entry in (api.get("entry"), api.get("aclnn_symbol"), *(api.get("candidates") or ())):
                if entry and node_id not in self.native_call_entries[entry]:
                    self.native_call_entries[entry].append(node_id)
            kernels = record.get("kernels", ())
            if not isinstance(kernels, Sequence) or isinstance(kernels, (str, bytes)):
                raise ProfilerUsageError(
                    f"structure native_calls kernels must be a sequence: {node_id}"
                )
            for kernel in kernels:
                if not isinstance(kernel, Mapping):
                    raise ProfilerUsageError(
                        f"structure native_calls kernel entry must be a mapping: {node_id}"
                    )
                name = kernel.get("name")
                if name and node_id not in self.native_call_kernels[name]:
                    self.native_call_kernels[name].append(node_id)
            aliases = record.get("observed_aliases", ())
            if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
                raise ProfilerUsageError(
                    f"structure native_calls observed_aliases must be a sequence: {node_id}"
                )
            for alias in aliases:
                if not isinstance(alias, Mapping):
                    raise ProfilerUsageError(
                        f"structure native_calls observed_alias must be a mapping: {node_id}"
                    )
                raw = (
                    alias.get("rank_id"),
                    alias.get("global_task_id"),
                    alias.get("stream_id"),
                    alias.get("task_id"),
                    alias.get("connection_id"),
                    alias.get("device_id"),
                    alias.get("start_ns"),
                    alias.get("end_ns"),
                )
                logical = alias.get("logical_rank", alias.get("rank_index"))
                if logical is not None and str(self.nodes[node_id].get("replica")) != str(logical):
                    continue
                if all(value is not None for value in raw):
                    raw_key = tuple(str(value) for value in raw)
                    if node_id not in self.native_call_observed[raw_key]:
                        self.native_call_observed[raw_key].append(node_id)
                for field in ("compute_name", "comm_name", "name", "op_type"):
                    if alias.get(field):
                        self.native_call_alias_names.add(str(alias[field]))
        for key, target in self.all.items():
            implementation = target.get("implementation") or {}
            kinds = set(implementation.get("native_kinds", ()))
            if implementation.get("native_kind"):
                kinds.add(implementation["native_kind"])
            for kind in sorted(kinds):
                self.native_implementations[kind].append(key)
        self._classification = {}

    def describe_implementation(self, kind, targets):
        """Describe a declared native boundary without claiming a tensor match."""
        targets = list(dict.fromkeys(targets))
        node_ids, subgraph_ids = self.split(targets)
        bindings = [self.native_bindings[key] for key in targets if key in self.native_bindings]
        call_ids = {row["call_id"] for row in bindings if row.get("call_id")}
        native_call = None
        if len(call_ids) == 1:
            row = bindings[0]
            api = row.get("api") or {}
            owner_keys = {
                (
                    str(self.all.get(key, {}).get("replica")),
                    str(self.all.get(key, {}).get("stage")),
                    (self.native_bindings[key].get("timing_owner") or {}).get("op_id")
                    or self.native_bindings[key].get("op_id")
                    or key,
                )
                for key in targets
                if key in self.native_bindings
            }
            unique_timing_owner = row.get("timing_owner") if len(owner_keys) == 1 else None
            native_call = {
                "call_id": row["call_id"],
                "entry": api.get("entry"),
                "aclnn_symbol": api.get("aclnn_symbol"),
                "kernels": [
                    kernel.get("name")
                    for kernel in row.get("kernels", ())
                    if isinstance(kernel, dict) and kernel.get("name")
                ],
                "evidence": row.get("evidence") or {},
                "timing_owner": unique_timing_owner,
                "declared_target_count": len(bindings),
                "tensor_match_implied": False,
                "occurrence_binding_implied": False,
            }
        return {
            "basis": "declared_native_implementation",
            "native_operator_type": kind,
            "declared_target_count": len(targets),
            "declared_node_ids": node_ids[:64],
            "declared_subgraph_ids": subgraph_ids[:64],
            "targets_with_contract": sum(
                self.all[key].get("contract") is not None for key in targets),
            "event_matchable_targets": sum(
                self.all[key].get("event_matchable", True) for key in targets),
            "coverage_kinds": sorted({
                self.all[key].get("coverage", {}).get(
                    "kind", "declared_node_boundary" if key in self.nodes else "complete_subgraph")
                for key in targets
            }),
            "common": self.classify(targets)["common"],
            "native_call": native_call,
            "tensor_match_implied": False,
            "occurrence_binding_implied": False,
        }

    def members(self, key):
        return self.subgraphs[key]["node_ids"] if key in self.subgraphs else [key]

    def enclosing_scopes(self, keys, fields):
        """Keep both sides of a cross-block fusion without duplicating an event."""
        scopes = set()
        for key in keys:
            for member in self.members(key):
                node = self.nodes[member]
                meta = node.get("operator", {}).get("metadata", {})
                semantic = {**meta.get("semantic_metadata", {}), **meta}
                owners = [node, *semantic.get("fused_origin_scopes", ())]
                for owner in owners:
                    scope = {**owner, "replica": node.get("replica")}
                    if (scope.get("layer") is not None and scope["layer"] >= 0
                            and scope.get(fields[-1]) is not None):
                        scopes.add(tuple(scope.get(field) for field in fields))
        return scopes

    def scope(self, row):
        """Expand a declared evidence scope, preserving full fusion containment."""
        node_ids = row.get("node_ids", ())
        subgraph_ids = row.get("subgraph_ids", ())
        if not node_ids and not subgraph_ids:
            raise ProfilerUsageError("evidence scope requires node_ids or subgraph_ids")
        if any(key not in self.nodes for key in node_ids) or any(
            key not in self.subgraphs for key in subgraph_ids
        ):
            raise ProfilerUsageError("evidence scope references an unknown node or subgraph")
        members = set(node_ids)
        for key in subgraph_ids:
            members.update(self.members(key))
        return members | {
            key for key, record in self.subgraphs.items()
            if set(record["node_ids"]) <= members
        }

    def split(self, keys):
        return (
            [key for key in keys if key in self.nodes],
            [key for key in keys if key in self.subgraphs],
        )

    def classify(self, candidates):
        """Keep only properties shared by every compatible declared target."""
        key = tuple(candidates)
        if key not in self._classification:
            targets = [self.all[item] for item in candidates]
            common = {}
            for field in ("block", "role"):
                values = {target.get(field) for target in targets}
                if len(values) == 1 and None not in values:
                    common[field] = next(iter(values))
            structure = dict(targets[0].get("structure", {})) if targets else {}
            structure = {
                field: value for field, value in structure.items()
                if value is not None and all(
                    target.get("structure", {}).get(field) == value for target in targets
                )
            }
            if structure:
                common["structure"] = structure
            self._classification[key] = {
                "basis": "common_to_all_declared_tensor_candidates",
                "common": common,
                "occurrence_binding_implied": False,
            }
        return self._classification[key]
