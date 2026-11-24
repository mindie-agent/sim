"""Identity-preserving retained output ports for canonical fusion rewrites.

[Role] Derive one named port per retained live-out of a contracted region and
  carry each downstream operand's exact producer port across the rewrite and
  the optimizer's event-insertion passes.
[Invariants] A retained value keeps its own dtype, shape, layout and storage in
  both the fused ``OpSpec`` metadata and the typed projection.  The causal sink
  keeps the ``:out`` name; every other retained live-out gets a distinct
  ``:<member>_out`` port.  Two distinct producer references are never collapsed
  to one port.  The complete global positional operand sequence is independent
  of the deduplicated scheduling ``deps``: repeated external inputs and
  interleaved producers keep their exact operand index through every rewrite.
  A generated ``event:wait:*`` and a caller event not explicitly bound as a
  data port are control edges; projection and this module share one classifier,
  and the positional data sequence is materialized before that wait replaces a
  data producer in ``deps``.  Raw metadata pairs retain every control
  declaration, including repeated keys, in first-occurrence order.
[Timeouts & Locks] Finite tuple/mapping arithmetic; no I/O, clocks or locks.
[Side-Effect Policy] Pure functions over immutable analysis/metadata records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

INPUT_PORTS_KEY = "fusion_input_ports"
RETAINED_PORTS_KEY = "retained_ports"
CONTROL_METADATA_KEYS = ("control_deps", "control_inputs", "barrier_after",
                         "event_deps", "barrier_deps")
_CONTROL_FAMILIES = frozenset({"event", "barrier"})


@dataclass(frozen=True)
class RetainedPort:
    """One named physical output of a contracted region."""
    value_id: str
    port_id: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    storage_bytes: int | None
    public: bool = False

    def metadata_record(self) -> tuple:
        return (self.port_id, self.dtype, tuple(self.shape), self.layout,
                self.storage_bytes, bool(self.public))


def _sanitize(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return text.strip("_") or "value"


def retained_interface(analysis, member_specs: tuple):
    """Primary retained output interface, or ``None`` for a helper region."""
    retained = tuple(getattr(analysis, "retained_outputs", ()))
    primary = []
    for value_id in retained:
        value = analysis.values.get(value_id)
        if value is None or value.producer is None:
            continue
        producer = analysis.ops.get(value.producer)
        if producer is None or not producer.outputs:
            continue
        if producer.outputs[0] != value_id:
            continue
        primary.append(value)
    interfaces = {(value.dtype, tuple(value.shape), value.layout)
                  for value in primary}
    if len(interfaces) > 1:
        raise ValueError("retained_output_interface_mismatch")
    if not primary:
        return None, None
    value = primary[-1]
    spec = next((item for item in member_specs
                 if item.op_id == value.producer), None)
    return value, spec


def _member_key(value) -> str:
    producer = str(value.producer or value.value_id)
    key = _sanitize(producer)
    if value.value_id.startswith(producer + ":"):
        suffix = value.value_id[len(producer) + 1:]
        if suffix and suffix != "out":
            key = f"{key}_{_sanitize(suffix)}"
    return key


def retained_ports(analysis, member_specs: tuple, fused_id: str
                   ) -> tuple[tuple[RetainedPort, ...], dict[str, str], Any]:
    """Distinct named ports for every retained live-out, in causal order."""
    primary_value, _primary_spec = retained_interface(analysis, member_specs)
    primary_id = primary_value.value_id if primary_value is not None else ""
    records: list[RetainedPort] = []
    value_port: dict[str, str] = {}
    used: set[str] = set()
    for value_id in getattr(analysis, "retained_outputs", ()):
        value = analysis.values.get(value_id)
        if value is None or value_id in value_port:
            continue
        if value_id == primary_id:
            port_id = f"{fused_id}:out"
        else:
            stem = _member_key(value)
            port_id = f"{fused_id}:{stem}_out"
            serial = 2
            while port_id in used:
                port_id = f"{fused_id}:{stem}_out_{serial}"
                serial += 1
        used.add(port_id)
        value_port[value_id] = port_id
        records.append(RetainedPort(
            value_id=value_id, port_id=port_id, dtype=value.dtype,
            shape=tuple(int(dim) for dim in value.shape), layout=value.layout,
            storage_bytes=(int(value.storage_bytes)
                           if value.storage_bytes is not None else None),
            public=bool(getattr(value, "public", False))))
    return tuple(records), value_port, primary_value


def metadata_pairs(metadata):
    """Walk mappings or raw pair sequences without collapsing repeated keys."""
    return metadata.items() if isinstance(metadata, Mapping) else metadata


def replace_scalar_metadata(metadata, **updates) -> tuple:
    """Replace selected scalar keys, preserving every unrelated raw pair."""
    kept = tuple((key, value) for key, value in metadata_pairs(metadata)
                 if key not in updates)
    return (*kept, *updates.items())


def _metadata_records(metadata, key):
    """Records from every sequence-valued declaration of ``key``, in order."""
    for name, value in metadata_pairs(metadata):
        if name == key and isinstance(value, (tuple, list)):
            yield from value


def control_value_names(value) -> tuple[str, ...]:
    """Names in one supported control declaration; mapping values are payloads."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, Mapping)):
        return tuple(str(item) for item in value)
    return ()


def declared_control_order(metadata) -> tuple[str, ...]:
    """Union every raw control declaration in first-occurrence order."""
    return tuple(dict.fromkeys(
        name for key, value in metadata_pairs(metadata)
        if key in CONTROL_METADATA_KEYS
        for name in control_value_names(value) if name))


def input_port_entries(metadata) -> tuple[tuple[str, str], ...]:
    """All explicit operand occurrences; positional repeats remain intact."""
    entries = []
    for item in _metadata_records(metadata, INPUT_PORTS_KEY):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            dep, port = str(item[0]), str(item[1])
            if dep and port:
                entries.append((dep, port))
    return tuple(entries)


def declared_control_names(metadata) -> frozenset[str]:
    """Names the operator explicitly declares as scheduling-only controls."""
    return frozenset(declared_control_order(metadata))


def is_control_node(node_id: str, ops_by_name: Mapping | None = None) -> bool:
    """True when ``node_id`` names an event/barrier scheduling node."""
    if str(node_id).startswith(("event:", "barrier:")):
        return True
    node = (ops_by_name or {}).get(node_id)
    return getattr(node, "family", "") in _CONTROL_FAMILIES


def control_dependency_names(op, ops_by_name: Mapping | None = None
                             ) -> frozenset[str]:
    """The one canonical data-vs-control dependency classifier.

    Projection and the positional-port helpers both call this function, so a
    dependency can never be control in one place and a tensor in the other.  A
    dependency is control when the operator declares it in its control metadata,
    when it is an explicit ``control:*`` node, or when it names an event/barrier
    node that the operator did not bind as an explicit data port.  A caller may
    therefore still author ``family='event'`` nodes and explicitly bind one as
    data; it is never forbidden and graph validation is not weakened.

    An explicit ``fusion_input_ports`` data binding is authoritative over the
    same name being declared in a control key: contraction may merge a genuine
    data producer of one member with a pure control predecessor of another into
    one fused id, and the surviving data port must not be erased by that merged
    control declaration.  The name stays a scheduling ``deps`` edge either way.
    """
    metadata = getattr(op, "metadata", ())
    explicit = {dep for dep, _port in input_port_entries(metadata)}
    names = set(declared_control_names(metadata))
    for dep in getattr(op, "deps", ()):
        dep = str(dep)
        if dep.startswith("control:") or (is_control_node(dep, ops_by_name)
                                          and dep not in explicit):
            names.add(dep)
    return frozenset(names - explicit)


def rewritten_dependencies(op, positions: tuple[tuple[str, str], ...],
                           replacements: Mapping[str, str],
                           ops_by_name: Mapping | None = None) -> tuple[str, ...]:
    """Rebuild scheduling ``deps`` without promoting or deleting controls.

    ``positions`` is the authoritative positional *data* operand sequence.
    Data dependencies keep their original scheduling order, control
    dependencies are carried through verbatim, and any explicit boundary
    operand absent from ``deps`` is appended so its producer stays ordered
    before the consumer.  Only this scheduling tuple is deduplicated; the
    positional expression remains separate.
    """
    data = [str(dep) for dep, _port in positions]
    controls = control_dependency_names(op, ops_by_name)
    deps = [str(dep) for dep in getattr(op, "deps", ())]
    ordered = [dep for dep in deps if dep in controls or dep in data]
    ordered.extend(dep for dep in deps if dep not in ordered)
    ordered.extend(dep for dep in data if dep not in ordered)
    return tuple(dict.fromkeys(
        str(replacements.get(dep, dep)) for dep in ordered))


def operand_ports(deps: tuple[str, ...], metadata: Mapping
                  ) -> tuple[tuple[str, str, bool], ...]:
    """Consumed port for every dependency occurrence, in operand order.

    A rewritten operator declares ``(dependency, producer_port)`` pairs in the
    same order as its data dependencies.  When the producer occurs twice, the
    two declared ports are returned in occurrence order; an occurrence without
    a declaration falls back to the producer's primary ``:out`` port.  The
    third field tells the caller whether the port was explicit, so unrelated
    bindings can be preserved without materializing synthetic entries.
    """
    pending: dict[str, list[str]] = {}
    for dep, port in input_port_entries(metadata):
        pending.setdefault(dep, []).append(port)
    counts: dict[str, int] = {}
    for dep in deps:
        counts[dep] = counts.get(dep, 0) + 1
    found = []
    for dep in deps:
        counts[dep] -= 1
        options = pending.get(dep)
        if not options:
            found.append((dep, f"{dep}:out", False))
        elif counts[dep] == 0:
            # The last occurrence of a producer consumes every remaining
            # declared port.  A deduplicated dependency list may therefore
            # still carry two or more distinct boundary tensors.
            found.extend((dep, port, True) for port in options)
            pending[dep] = []
        else:
            found.append((dep, options.pop(0), True))
    return tuple(found)


def full_operand_ports(op, ops_by_name: Mapping | None = None
                       ) -> tuple[tuple[str, str], ...]:
    """Complete global positional *data* operands of one operator.

    ``OpSpec.deps`` is a scheduling set after a contraction and may be
    deduplicated, but the tensor expression of an operator is an ordered tuple
    in which the same external value or producer may occur several times.  The
    control classification is exactly :func:`control_dependency_names`: declared
    controls, ``control:*`` nodes, and implicit event/barrier nodes that were
    not explicitly bound as data ports are excluded *before* the positional
    sequence is built.  An existing dependency therefore never changes from
    control to data merely because an event rewrite materializes the snapshot.

    The declared ``fusion_input_ports`` list is authoritative when it covers
    every data occurrence implied by ``deps``, including an empty scheduling
    tuple (a prior fusion can expose several ports of one producer). Otherwise every
    data dependency occurrence is materialized in exact index order
    ``0 .. len(data_deps) - 1``, filling that position with its declared port or
    the producer's primary ``:out`` port.  Nothing here mutates the operator.
    """
    deps = tuple(str(dep) for dep in getattr(op, "deps", ()))
    metadata = getattr(op, "metadata", ())
    controls = control_dependency_names(op, ops_by_name)
    data_deps = tuple(dep for dep in deps if dep not in controls)
    implied = operand_ports(data_deps, metadata)
    explicit = input_port_entries(metadata)
    if explicit and len(explicit) >= len(implied) \
            and set(data_deps) <= {dep for dep, _port in explicit}:
        # Preserve the complete declared sequence verbatim: it can carry more
        # boundary tensors than deduplicated scheduling producers.
        return explicit
    return tuple((dep, port) for dep, port, _explicit in implied)


def replace_input_ports(metadata: tuple,
                        entries: tuple[tuple[str, str], ...]) -> tuple:
    """Replace every declared operand binding with ``entries``."""
    kept = tuple((key, value) for key, value in metadata_pairs(metadata)
                 if key != INPUT_PORTS_KEY)
    if not entries:
        return kept
    return (*kept, (INPUT_PORTS_KEY, tuple(entries)))


def merge_input_ports(metadata: tuple, entries: tuple[tuple[str, str], ...]
                      ) -> tuple:
    """Append new producer-port bindings without disturbing earlier ones."""
    if not entries:
        return metadata
    return replace_input_ports(metadata, (*input_port_entries(metadata), *entries))


def event_control_rewrite(op, replacements: Mapping[str, str],
                          ops_by_name: Mapping | None = None
                          ) -> tuple[tuple[str, ...], tuple]:
    """Split a cross-stream rewrite into control deps and data operands.

    A generated ``event:wait:*`` is a scheduling/control edge, never a tensor.
    The complete positional data-operand sequence is materialized from the
    operator *before* any producer in ``deps`` is replaced, using the shared
    :func:`control_dependency_names` classifier, so projection still resolves
    the original producer ports (including a repeated occurrence) and an
    existing caller control event is never promoted to a tensor operand.  Only
    the scheduling tuple is deduplicated; the ordered expression is kept
    verbatim.  An already-materialized ``fusion_input_ports`` sequence is
    preserved unchanged.
    """
    deps = tuple(dict.fromkeys(str(replacements.get(dep, dep))
                               for dep in getattr(op, "deps", ())))
    metadata = replace_input_ports(getattr(op, "metadata", ()),
                                   full_operand_ports(op, ops_by_name))
    return deps, metadata


def restore_input_ports(metadata: tuple,
                        origins: Mapping[str, tuple[str, ...]]) -> tuple:
    """Restore original producer ids when generated events are stripped."""
    entries = input_port_entries(metadata)
    if not entries or not origins:
        return metadata
    restored = []
    for dep, port in entries:
        origin = origins.get(dep)
        if origin is None:
            restored.append((dep, port))
        elif len(origin) == 1:
            restored.append((origin[0], port))
    rewritten = tuple(restored)
    if rewritten == entries:
        return metadata
    return replace_input_ports(metadata, rewritten)


def projected_ports(metadata) -> tuple[RetainedPort, ...]:
    """Parse the retained-port records embedded in a fused ``OpSpec``."""
    found = []
    for item in _metadata_records(metadata, RETAINED_PORTS_KEY):
        if not isinstance(item, (tuple, list)) or len(item) < 5:
            continue
        port_id, dtype, shape, layout, storage = item[:5]
        public = bool(item[5]) if len(item) > 5 else False
        found.append(RetainedPort(
            value_id=str(port_id), port_id=str(port_id), dtype=str(dtype),
            shape=tuple(int(dim) for dim in shape), layout=str(layout),
            storage_bytes=(int(storage) if storage is not None else None),
            public=public))
    return tuple(found)
