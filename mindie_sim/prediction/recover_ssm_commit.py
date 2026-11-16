"""Acceptance-bound RecoverSSM commit intent and physical state publication.

[Role]
  A recurrent builder emits a compact-record intent; after sampling/acceptance
  has been assembled this module materializes exactly one commit per
  request/layer state owner.  The commit depends on the verify record, the real
  acceptance event and the base state version, so an unaccepted state version
  can never become visible.

[Invariants]
  * Integer commitment comes from ``RequestView.draft_advanced_tokens`` or a
    declared body input; a configured mean is never rounded into a commit.
  * Commit reads ``S + a*R``, performs ``4*a*h*dk*dv`` vector work and writes
    ``S`` once per active sequence; zero/padding requests issue no work.
  * Duplicate commits for one state owner/version are rejected; persistent
    publication is never a dependency-free tail.
  * Source-specific record geometry/recipes remain declared by their owner.

[Timeouts & Locks]
  Pure record construction; no I/O, locks or Engine mutation.

[Side-Effect Policy]
  Returns replacement immutable operators only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .recover_ssm_work import commit_tokens_for_view


ZERO = 0
ONE = 1
FOUR = 4
FOUR_FLOAT = 4.0
TWO_FLOAT = 2.0
FIRST = 0
VERIFY_STAGE = "verify_compact_record"
INTENT_STAGE = "commit_intent"
BODY_BOUNDARY = "declared_body_input"
ACCEPTANCE_BOUNDARY = "sampling_acceptance"


def _advanced_tokens(view: Any) -> int:
    q = max(ONE, int(getattr(view, "q_len", ONE) or ONE))
    advanced = int(getattr(view, "draft_advanced_tokens", ZERO) or ZERO)
    if advanced <= ZERO:
        if q <= ONE:
            return ONE
        raise ModelingError(
            "recover_ssm_commit_requires_resolved_integer_advancement:"
            + str(getattr(view, "request_id", "")))
    return min(q, advanced)


def _record_bytes_per_token(metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("record_bytes_per_token")
    if declared is not None:
        return float(declared)
    heads = int(metadata.get("heads", ONE))
    dk = int(metadata.get("dk", ONE))
    dv = int(metadata.get("dv", ONE))
    kg_dtype = str(metadata.get("record_kg_dtype", "bf16"))
    return float(heads) * (
        float(dv) * FOUR_FLOAT + TWO_FLOAT * float(dk) * float(dtype_bytes(kg_dtype)))


def _state_bytes(metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("state_bytes")
    if declared is not None:
        return float(declared)
    heads = int(metadata.get("heads", ONE))
    dk = int(metadata.get("dk", ONE))
    dv = int(metadata.get("dv", ONE))
    dtype = str(metadata.get("state_dtype", "fp32"))
    return float(heads * dk * dv) * float(dtype_bytes(dtype))


def _decode_speculative(view: Any) -> bool:
    """A verify view is decode-only; a prefill multi-token query is not spec."""
    return (str(getattr(view, "phase", "")) == "decode"
            and int(getattr(view, "q_len", ZERO) or ZERO) > ONE
            and not bool(getattr(view, "padding", False)))


def _owner(metadata: Mapping[str, Any], op: OpSpec) -> str:
    return str(metadata.get("state_owner") or
               metadata.get("recurrent_state_owner") or f"layer{op.layer}")


def _geometry(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("recover_ssm_geometry")
    return value if isinstance(value, Mapping) else {}


def _scope(metadata: Mapping[str, Any]) -> str:
    return str(_geometry(metadata).get("scope", metadata.get("scope", "")))


def _layer(metadata: Mapping[str, Any], op: OpSpec) -> int:
    declared = _geometry(metadata).get("layer", metadata.get("layer", op.layer))
    return int(declared if declared is not None else op.layer)


def is_recover_ssm_intent(op: OpSpec) -> bool:
    """Whether this operator is the deferred, uncharged commit intent."""
    metadata = dict(op.metadata)
    return bool(metadata.get("recover_ssm_intent")) or str(
        metadata.get("recover_ssm_stage", "")) == INTENT_STAGE


def _is_pending_commit(op: OpSpec) -> bool:
    metadata = dict(op.metadata)
    return bool(metadata.get("recover_ssm_pending_acceptance"))


def bind_recover_ssm_commits(
    ops: Sequence[OpSpec], *, views: Sequence[Any],
    acceptance_events: Mapping[str, Any] | None,
    state_bindings: Mapping[str, Any] | None, runtime: Any,
) -> tuple[OpSpec, ...]:
    """Materialize acceptance-bound commits for every declared state owner."""
    active = tuple(view for view in views if _decode_speculative(view))
    if not active:
        return tuple(ops)
    events = dict(acceptance_events or {})
    bindings = dict(state_bindings or {})
    committed_by_request = {
        str(getattr(view, "request_id", "")): commit_tokens_for_view(view, runtime)
        for view in active
    }
    result: list[OpSpec] = []
    emitted: set[tuple[str, str, int, str]] = set()
    for op in ops:
        if not (is_recover_ssm_intent(op) or _is_pending_commit(op)):
            result.append(op)
            continue
        metadata = dict(op.metadata)
        owner = _owner(metadata, op)
        version = str(metadata.get("input_state_version",
                                  metadata.get("state_version", "")))
        key = (owner, _scope(metadata), _layer(metadata, op), version)
        if key in emitted:
            raise ModelingError(
                f"recover_ssm_duplicate_commit:{owner}:{version}")
        emitted.add(key)
        request_ids = tuple(str(item) for item in op.requests)
        selected = [
            (request, committed_by_request[request]) for request in request_ids
            if request in committed_by_request
        ]
        if not selected:
            result.append(op)
            continue
        dependencies = list(op.deps)
        missing = [request for request, _count in selected if request not in events]
        if missing:
            raise ModelingError(
                "recover_ssm_commit_requires_acceptance_event:"
                + ",".join(sorted(missing)))
        dependencies.extend(str(events[request]) for request, _ in selected)
        binding = bindings.get(owner)
        if binding is not None:
            dependencies.append(str(binding))
        total = sum(count for _request, count in selected)
        geometry = _geometry(metadata)
        if geometry:
            metadata["heads"] = int(geometry.get("heads", ONE))
            metadata["dk"] = int(geometry.get("dk", ONE))
            metadata["dv"] = int(geometry.get("dv", ONE))
            metadata["state_dtype"] = str(geometry.get("state_dtype", "fp32"))
            metadata["record_kg_dtype"] = str(geometry.get("kg_dtype", "bf16"))
        record = _record_bytes_per_token(metadata)
        state = _state_bytes(metadata)
        read = float(len(selected)) * state + float(total) * record
        write = float(len(selected)) * state
        heads = int(metadata.get("heads", ONE))
        dk = int(metadata.get("dk", ONE))
        dv = int(metadata.get("dv", ONE))
        vector = float(FOUR * total * heads * dk * dv)
        sequence = tuple((request, count) for request, count in selected)
        metadata.update({
            "recover_ssm_stage": "commit",
            "recover_ssm_pending_acceptance": False,
            "recover_ssm_intent": False,
            "state_slots": ONE,
            "query_tokens": sum(
                max(ONE, int(getattr(view, "q_len", ONE)))
                for view in active
                if str(getattr(view, "request_id", "")) in dict(sequence)),
            "committed_tokens": sequence,
            "record_bytes_per_token": record,
            "state_bytes": state,
            "heads": heads,
            "dk": dk,
            "dv": dv,
            "input_state_version": version,
            "published_state_version": str(metadata.get(
                "published_state_version", version)),
            "acceptance_boundary": ACCEPTANCE_BOUNDARY,
            "state_owner": owner,
        })
        result.append(replace(
            op,
            deps=tuple(dict.fromkeys(dependencies)),
            vector_ops=vector,
            hbm_bytes=read + write,
            mte2_bytes=read,
            mte3_bytes=write,
            metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])),
            notes=tuple(op.notes) + ("recover_ssm_commit_after_acceptance",),
        ))
    return tuple(result)


def bind_declared_body_commits(
    ops: Sequence[OpSpec], *, views: Sequence[Any],
) -> tuple[OpSpec, ...]:
    """Body-only assembly: require the declared integer advancement explicitly."""
    active = tuple(view for view in views if _decode_speculative(view))
    for view in active:
        _advanced_tokens(view)
    rewritten = []
    for op in ops:
        if is_recover_ssm_intent(op) or _is_pending_commit(op):
            metadata = dict(op.metadata)
            metadata["acceptance_boundary"] = BODY_BOUNDARY
            rewritten.append(replace(
                op, metadata=tuple(sorted(metadata.items(),
                                          key=lambda pair: pair[FIRST]))))
        else:
            rewritten.append(op)
    return tuple(rewritten)
