"""Capture partitioning and structural implementation audit records.

[Role] Describe capture, fusion and value boundaries, ports and provenance.
[Invariants] Audit records retain occurrence identity as structural annotations.
  Capture, fusion and overlap are independent dimensions.
[Timeouts & Locks] No blocking operations or locks.
[Side-Effect Policy] Read operator metadata. Appending an audit updates only the
  metadata mapping supplied by the caller.
"""
from mindie_sim.schema.operators import FieldDisposition


def capture_partition(ops, runtime, replay_scope=None):
    """Predict the capture split before transforms run.

    ``GraphPolicy.apply`` normally runs after generic fusion.  The transforms
    need the same split rule now so a split (eager) family is not folded into
    a captured neighbour and then relabelled as a fused captured node.
    """
    if replay_scope == "none":
        return None
    if replay_scope == "full":
        return None
    mode = str(getattr(runtime, "graph_mode", "auto")).lower()
    if mode in {"none", "off", "eager", "disabled"}:
        return None
    phases = {op.phase for op in ops if op.phase}
    if replay_scope is None:
        full = (str(getattr(runtime, "graph_capture_mode", "full")) == "full"
                and not bool(getattr(runtime, "invalid_full_modes", False))
                and phases <= {"decode"})
        if full:
            return None
        if not (phases <= {"decode"}
                or str(getattr(runtime, "mixed_mode", "none")) not in {"none", "off"}):
            return None
        if not tuple(getattr(runtime, "piecewise_capture_keys", ())):
            return None
    split = set(getattr(runtime, "splitting_ops", None) or {
        "attention", "attention_qk", "attention_pv", "attention_softmax",
        "attention_merge",
    })
    state, segment, last_stage, last_captured = {}, 0, None, False
    for op in ops:
        captured = bool(replay_scope == "full"
                        or (op.family not in split and op.recipe not in split))
        if op.stage != last_stage or captured != last_captured:
            segment += 1
        state[op.op_id] = (captured, segment)
        last_stage, last_captured = op.stage, captured
    return state


def capture_fusion_overlap(ops, *, kind):
    """Keep capture, fusion and overlap as separate declared dimensions."""
    records = [dict(op.metadata) for op in ops]

    def common(key):
        values = tuple(record.get(key) for record in records)
        return values[0] if values and all(value == values[0] for value in values) else "mixed"

    streams = sorted({op.stream or "main" for op in ops})
    return {
        "capture": {
            "graph_captured": common("graph_captured"),
            "graph_segment": common("graph_segment"),
            "declared": any(record.get("graph_captured") is not None
                            for record in records),
        },
        "fusion": {"kind": kind, "members": len(ops),
                   "member_occurrences": tuple(op.op_id for op in ops)},
        "overlap": {
            "streams": tuple(streams),
            "cross_stream": len(streams) > 1,
            "declared_concurrency": "not_implied_by_fusion",
        },
    }


def audit_records(op):
    """Return the identity-bearing transform/port audit records.

    These records live inside ``fused_origin_scopes``, which is a structural
    annotation rather than a physical-work axis.  Keeping exact occurrence and
    consumer ids there lets the relation be exported without making repeated
    layers look physically different.
    """
    for key, value in reversed(op.metadata):
        if key == "fused_origin_scopes":
            return tuple(value) if value else ()
    return ()


def audit_boundary(op):
    for record in audit_records(op):
        if record.get("kind") in {"transform_boundary", "native_fusion_boundary",
                                  "activation_materialization"}:
            boundary = record.get("boundary")
            if isinstance(boundary, dict):
                return boundary
    return {}


def input_ports(op):
    result = []
    for record in audit_records(op):
        if record.get("kind") != "input_ports":
            continue
        for entry in record.get("ports", ()):
            if entry not in result:
                result.append(entry)
    return tuple(result)


def _audit_outputs(op):
    for record in audit_records(op):
        if record.get("kind") not in {"transform_boundary", "activation_materialization",
                                      "native_fusion_boundary", "source_value_contract"}:
            continue
        boundary = record.get("boundary")
        if not isinstance(boundary, dict):
            continue
        if boundary.get("typed_outputs"):
            return tuple((value, False) for value in boundary["typed_outputs"])
        typed = boundary.get("typed")
        if isinstance(typed, dict):
            for key in ("outputs", "external_outputs"):
                if typed.get(key):
                    return tuple((value, True) for value in typed[key])
        if boundary.get("outputs"):
            return tuple((value, True) for value in boundary["outputs"])
    return ()


def append_audit(metadata, *, kind, op_id, boundary=None, ports=(),
                 layer=None, stage=None, block=None, role=None):
    """Append/replace one audit record without adding a physical axis."""
    origins = [record for record in metadata.get("fused_origin_scopes", ())
               if not (record.get("kind") == kind and record.get("op_id") == op_id)]
    record = {"op_id": op_id, "kind": kind, "layer": layer,
              "stage": stage, "block": block, "role": role}
    if boundary is not None:
        record["boundary"] = boundary
    if ports:
        record["ports"] = tuple(ports)
    origins.append(record)
    metadata["fused_origin_scopes"] = tuple(origins)


def field_disposition(field, action, source="", detail=""):
    """Validate and freeze one preserve/rewrite/consume decision."""
    return FieldDisposition(field, action, source, detail).to_dict()


def replaced_occurrences(*ops):
    """Flatten any nested transform replacement into the current occurrence set."""
    result = []
    for op in ops:
        boundary = audit_boundary(op) or dict(op.metadata).get(
            "transform_boundary", {})
        values = (boundary.get("replaced_occurrences", (op.op_id,))
                  if isinstance(boundary, dict) else (op.op_id,))
        for value in values:
            if value not in result:
                result.append(value)
    return tuple(result)


def nested_boundary(op):
    boundary = audit_boundary(op) or dict(op.metadata).get("transform_boundary")
    return boundary if isinstance(boundary, dict) else {}


def member_contracts(ops):
    """Preserve each replaced member's own typed fields and provenance."""
    return tuple({
        "op_id": op.op_id,
        "family": op.family,
        "recipe": op.recipe,
        "dtype": op.dtype,
        "layout": op.layout,
        "returns": list(op.returns),
        "metadata": dict(op.metadata),
    } for op in ops)


def semantic_attrs(ops, keys):
    attrs = []
    for op in ops:
        metadata = dict(op.metadata)
        for key in keys:
            if key in metadata:
                attrs.append({"name": key, "value": metadata[key],
                              "unit": "", "source": op.op_id})
    return tuple(attrs)
