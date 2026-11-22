"""Contraction-boundary control preservation for canonical fusion rewrites.

[Role] Extract the control-boundary logic of a contraction: collect the external
  control predecessors declared by a region's members, classify the synthesized
  operator's boundary, and rewrite every surviving consumer's scheduling and
  control references from removed member ids to the fused id.
[Invariants] An external producer that is a genuine data operand of any member
  stays a data operand even when another member also declares it as a control;
  only pure-control predecessors (not feeding an external data port) become
  ``control_deps`` on the fused node.  Every reference to a contracted member in
  ``deps`` or in ``control_deps``/``control_inputs``/``barrier_after``/
  ``event_deps``/``barrier_deps`` is rewritten and deduplicated in
  first-occurrence order for both boundary directions.  A surviving consumer's
  complete positional data sequence is materialized before scheduling ``deps``
  is deduplicated, so repeated and interleaved tensor operands keep their exact
  multiplicity.  An explicit ``fusion_input_ports`` binding on a caller marker
  is retargeted through the retained-port map, while an unbound pure marker
  keeps its control-only stripping.  Ordering is built from every node's
  complete known relation -- ``deps``, explicit data ports and all five control
  keys, including declarations on untouched survivors -- so a multi-hop control
  edge is never dropped.  The fused node is inserted after every incoming
  predecessor and before every outgoing consumer; a deterministic stable
  topological order handles the reorderable case, and a genuine cycle still
  raises ``fusion_boundary_order_invalid``.  The shared classifier owns
  data-vs-control precedence, so no genuine tensor is deleted by a merged edge.
[Timeouts & Locks] Finite tuple/mapping arithmetic; no I/O, clocks or locks.
[Side-Effect Policy] Pure functions over immutable metadata tuples.
"""
from __future__ import annotations

from dataclasses import replace
import heapq
from typing import Mapping

from mindie_sim.prediction.fusion.ports import (
    CONTROL_METADATA_KEYS, control_dependency_names, control_value_names,
    declared_control_order, input_port_entries, is_control_node, metadata_pairs,
    replace_input_ports, rewritten_dependencies,
)


def _declared_order(spec, by_name: Mapping | None) -> tuple[str, ...]:
    """Only the control names explicitly declared in the five metadata keys."""
    controls = control_dependency_names(spec, by_name)
    return tuple(name for name in declared_control_order(getattr(spec, "metadata", ()))
                 if name in controls)


def _spec_control_order(spec, by_name: Mapping | None) -> tuple[str, ...]:
    """External control names in declared/scheduling first-occurrence order."""
    controls = control_dependency_names(spec, by_name)
    ordered: list[str] = []
    for dep in getattr(spec, "deps", ()):
        name = str(dep)
        if name in controls and name not in ordered:
            ordered.append(name)
    for name in _declared_order(spec, by_name):
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def external_control_names(member_specs: tuple, members: set[str],
                           by_name: Mapping | None) -> tuple[str, ...]:
    """Every external control predecessor carried by a region's members."""
    found: list[str] = []
    for spec in member_specs:
        for name in _spec_control_order(spec, by_name):
            if name not in members and name not in found:
                found.append(name)
    return tuple(found)


def external_declared_control_names(member_specs: tuple, members: set[str],
                                    by_name: Mapping | None) -> tuple[str, ...]:
    """Only control predecessors declared through a supported metadata key."""
    found: list[str] = []
    for spec in member_specs:
        for name in _declared_order(spec, by_name):
            if name not in members and name not in found:
                found.append(name)
    return tuple(found)


def region_boundary_ports(ops: tuple, members: set[str],
                          operand_layout: Mapping[str, tuple[tuple[str, str], ...]]
                          ) -> tuple[tuple[str, str], ...]:
    """Boundary ``(producer, port)`` occurrences consumed by a region.

    Every member's complete global positional operand sequence is resolved
    before any rewrite.  Internal edges (both endpoints in the region) vanish;
    distinct ports of one external producer survive as distinct operands, and an
    exact duplicate tensor operand is listed once because one fused kernel
    consumes that physical tensor once.
    """
    found: list[tuple[str, str]] = []
    for op in ops:
        if op.op_id not in members:
            continue
        for dep, port in operand_layout[op.op_id]:
            if dep in members:
                continue
            pair = (dep, port)
            if pair not in found:
                found.append(pair)
    return tuple(found)


def incoming_boundary_controls(member_specs: tuple, members: set[str],
                               boundary: tuple[tuple[str, str], ...],
                               by_name: Mapping | None
                               ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the fused scheduling deps and its pure-control declarations.

    The scheduling tuple is the ordered union of external data producers and
    external control predecessors.  A name that is both a boundary data producer
    and a control predecessor remains a data operand; only names absent from the
    boundary become ``control_deps`` declarations on the fused node.
    """
    external = external_control_names(member_specs, members, by_name)
    declared = external_declared_control_names(member_specs, members, by_name)
    data = {str(dep) for dep, _port in boundary}
    pure = tuple(name for name in declared if name not in data)
    # Only a real operator or an already-scheduled member dependency can become
    # a typed scheduling edge.  A control declared solely in metadata stays in
    # ``control_deps`` (classification) but must not invent an unavailable
    # graph node.
    scheduled = tuple(
        name for name in external
        if (by_name is not None and name in by_name)
        or any(name in getattr(spec, "deps", ()) for spec in member_specs))
    deps = [str(dep) for dep, _port in boundary]
    deps.extend(scheduled)
    return tuple(dict.fromkeys(deps)), pure


def apply_incoming_controls(fused, member_specs: tuple, members: set[str],
                            boundary: tuple[tuple[str, str], ...],
                            by_name: Mapping | None):
    """Carry a region's external control classification onto the fused node."""
    deps, pure = incoming_boundary_controls(
        member_specs, members, boundary, by_name)
    metadata = tuple(metadata_pairs(getattr(fused, "metadata", ())))
    if pure:
        metadata = tuple((key, value) for key, value in metadata
                         if key != "control_deps")
        metadata = (*metadata, ("control_deps", pure))
    return replace(fused, deps=deps, metadata=metadata)


def rewritten_control_metadata(metadata: tuple,
                               replacements: Mapping[str, str]) -> tuple:
    """Rewrite raw pairs; deduplicate targets across each key's declarations.

    Keep pair order and mapping payloads, choosing the first payload when
    contraction merges targets.  Positional data bindings are never deduplicated.
    """
    if not replacements:
        return metadata
    updated = []
    seen: dict[str, set[str]] = {}
    for key, value in metadata_pairs(metadata):
        if key not in CONTROL_METADATA_KEYS or not isinstance(
                value, (str, tuple, list, Mapping)):
            updated.append((key, value))
            continue
        used = seen.setdefault(key, set())
        names = {}
        items = value.items() if isinstance(value, Mapping) else (
            (name, None) for name in control_value_names(value))
        for item, payload in items:
            name = replacements.get(str(item), str(item))
            if name not in used:
                used.add(name)
                names[name] = payload
        if isinstance(value, str):
            if names:
                updated.append((key, next(iter(names))))
        else:
            updated.append((key, names if isinstance(value, Mapping) else tuple(names)))
    return tuple(updated)


def references_member(op, members: set[str]) -> bool:
    """True when ``op`` names a contracted member outside its data operands."""
    if any(str(dep) in members for dep in getattr(op, "deps", ())):
        return True
    return any(name in members
               for name in declared_control_order(getattr(op, "metadata", ())))


def ordering_prerequisites(all_ops: tuple) -> dict[str, set[str]]:
    """Complete known ordering relation over survivors plus the fused node.

    Every node contributes its scheduling ``deps``, its explicit
    ``fusion_input_ports`` producers and every control name declared under the
    five supported keys.  Untouched survivors are therefore first-class ordering
    nodes: a control path through them is never omitted merely because the
    contraction did not rewrite them.  Raw metadata pairs are walked rather than
    ``dict(metadata)`` so duplicate same-key declarations all contribute.  An
    explicit data-port binding keeps precedence over the same name as a control
    declaration, although the name stays ordered through the data edge.  The
    relation is restricted to ids in ``all_ops`` and excludes self edges: a
    removed member, an unavailable external producer or a self-loop is never
    invented.
    """
    known = {op.op_id: op for op in all_ops}
    prerequisites: dict[str, set[str]] = {}
    for op in all_ops:
        op_id = op.op_id
        metadata = getattr(op, "metadata", ())
        entries = input_port_entries(metadata)
        explicit = {dep for dep, _port in entries}
        names = {str(dep) for dep in getattr(op, "deps", ())}
        names.update(dep for dep, _port in entries)
        names.update(control_dependency_names(op, known))
        names.update(name for name in declared_control_order(metadata)
                     if name not in explicit)
        prerequisites[op_id] = {
            name for name in names if name in known and name != op_id}
    return prerequisites


def _stable_topological_placement(rewritten: list, fused,
                                  first_member: int) -> tuple[list, int]:
    """Deterministically order survivors plus the fused node.

    The complete multi-hop ordering relation (:func:`ordering_prerequisites`)
    covers every survivor's scheduling, data-port and control declarations.  The
    original survivor order is kept wherever those edges allow it; ``heapq``
    breaks ties by the original index and places the fused node just before the
    survivor that occupied the first member's slot.  A real cycle (including a
    predecessor that is also an outgoing consumer) exhausts the ready queue and
    is reported as the documented boundary-order failure.
    """
    fused_id = fused.op_id
    all_ops = (*rewritten, fused)
    by_id = {op.op_id: op for op in all_ops}
    prerequisites = ordering_prerequisites(all_ops)
    successors: dict[str, list[str]] = {op_id: [] for op_id in prerequisites}
    for op_id, deps in prerequisites.items():
        for dep in deps:
            successors[dep].append(op_id)
    keys = {op_id: (index, 0) for index, op_id in enumerate(by_id)}
    keys[fused_id] = (first_member, -1)
    ready = [(keys[op_id], op_id) for op_id, deps in prerequisites.items()
             if not deps]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        _key, current = heapq.heappop(ready)
        ordered.append(current)
        for successor in successors[current]:
            prerequisites[successor].discard(current)
            if not prerequisites[successor]:
                heapq.heappush(ready, (keys[successor], successor))
    if len(ordered) != len(prerequisites):
        raise ValueError("fusion_boundary_order_invalid")
    anchor = ordered.index(fused_id)
    return [by_id[op_id] for op_id in ordered if op_id != fused_id], anchor


def rewrite_consumers(ops: tuple, members: set[str],
                      operand_layout: Mapping[str, tuple[tuple[str, str], ...]],
                      fused, value_port: Mapping[str, str],
                      by_name: Mapping | None) -> tuple[tuple, int]:
    """Rewrite surviving consumers and return them with the fused anchor.

    A consumer is rewritten when it consumes a removed member as a data operand
    or names one in ``deps``/control metadata.  Data ports are rebuilt from the
    materialized positional sequence and the complete original data-operand
    tuple is persisted even when no data position changed, so scheduling
    deduplication can never delete a repeated or interleaved genuine operand.
    The five control keys are rewritten and deduplicated; a control declared
    only in metadata still becomes a real scheduling edge so the fused producer
    is ordered before the consumer.

    A caller event/barrier node keeps its pure scheduling-marker behaviour
    except for an operand it explicitly declares in ``fusion_input_ports``: that
    is a genuine tensor binding and is retargeted through the retained-port map
    rather than dropped with the removed producer.  The fused node is placed in
    the non-empty interval after every incoming data/control predecessor and
    before every surviving consumer; only a genuine dependency cycle makes
    placement impossible.
    """
    member_set = set(members)
    fused_id = fused.op_id
    replacements = {name: fused_id for name in member_set}
    rewritten = []
    for op in ops:
        if op.op_id in member_set:
            continue
        # A record/wait/barrier marker without explicit data bindings consumes
        # only scheduling edges: a contracted producer is rewritten as control
        # so event stripping later restores the fused id rather than a removed
        # member.  An operand the caller explicitly declared is a real tensor
        # and must be retargeted instead.
        control_node = is_control_node(op.op_id, by_name)
        explicit_ports = frozenset(input_port_entries(op.metadata))
        port_entries: list[tuple[str, str]] = []
        replaced = False
        for dep, port in operand_layout[op.op_id]:
            if dep not in member_set:
                # Unrelated operands keep their exact global index, including
                # repeated external values and interleaved producers.
                port_entries.append((dep, port))
                continue
            if control_node and (dep, port) not in explicit_ports:
                continue
            port_id = value_port.get(port)
            if port_id is None:
                raise ValueError(f"retained_port_missing:{port}:{op.op_id}")
            port_entries.append((fused_id, port_id))
            replaced = True
        if not replaced and not references_member(op, member_set):
            rewritten.append(op)
            continue
        # Materialize the positional sequence before ``rewritten_dependencies``
        # deduplicates producers: deduplication must never reduce the number of
        # genuine tensor operand occurrences.
        metadata = replace_input_ports(op.metadata, tuple(port_entries))
        metadata = rewritten_control_metadata(metadata, replacements)
        scheduled = rewritten_dependencies(
            op, operand_layout[op.op_id], replacements, by_name)
        if fused_id not in scheduled:
            scheduled = (*scheduled, fused_id)
        rewritten.append(replace(op, deps=scheduled, metadata=metadata))
    # The first member's original slot is a preference, never a hard upper
    # bound: a later member's external predecessor (or generated event wait)
    # can legitimately sit after it.  Ordering bounds use the complete known
    # relation, including control declarations on untouched survivors.
    first_member = min(i for i, op in enumerate(ops) if op.op_id in member_set)
    prerequisites = ordering_prerequisites((*rewritten, fused))
    dependencies = prerequisites[fused_id]
    last_dependency = max(
        (i for i, op in enumerate(rewritten) if op.op_id in dependencies),
        default=-1)
    consumers = [i for i, op in enumerate(rewritten)
                 if fused_id in prerequisites[op.op_id]]
    lower = last_dependency + 1
    upper = min(consumers) if consumers else len(rewritten)
    if lower <= upper:
        anchor = min(max(first_member, lower), upper)
        # A single survivor slot can still invert an untouched survivor's
        # multi-hop control edge; only accept it when every relation points
        # forward, otherwise use the deterministic topological placement.
        placed = [*rewritten[:anchor], fused, *rewritten[anchor:]]
        position = {op.op_id: index for index, op in enumerate(placed)}
        if all(position[dep] < position[op_id]
               for op_id, deps in prerequisites.items() for dep in deps):
            return tuple(rewritten), anchor
    # A control-only consumer can precede the first member, while a later
    # member's external predecessor sits between members.  No single slot in
    # the original survivor order satisfies both, but the contraction is still
    # acyclic: shift only what the complete dependency relation requires.
    ordered, anchor = _stable_topological_placement(
        rewritten, fused, first_member)
    return tuple(ordered), anchor
