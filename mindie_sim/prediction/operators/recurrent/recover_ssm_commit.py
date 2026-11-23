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
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)
from .recover_ssm_work import FP32_RECORD_BYTES, commit_tokens_for_view

REPLAY_OPS_PER_STATE_ELEMENT = 4
COMMIT_OWNER_CAPACITY_DEFAULT = 64
COMMIT_KERNEL_ID = "RecoverSSMStateCommit"


VERIFY_STAGE = "verify_compact_record"
INTENT_STAGE = "commit_intent"
BODY_BOUNDARY = "declared_body_input"
ACCEPTANCE_BOUNDARY = "sampling_acceptance"


def _advance_published_version(
    input_version: str, owner: str, total_tokens: int, explicit: Any = None,
) -> str:
    """Guarantee strict state-version advance whenever accepted tokens > 0."""
    base = str(input_version or "")
    if explicit is not None and str(explicit) and str(explicit) != base:
        return str(explicit)
    if total_tokens <= 0:
        return base
    if base.startswith("v") and base[1:].isdigit():
        return f"v{int(base[1:]) + int(total_tokens)}"
    if base.isdigit():
        return str(int(base) + int(total_tokens))
    return f"{base}+v{int(total_tokens)}" if base else f"{owner}:v{int(total_tokens)}"


def _commit_owner_schedule(
    active_count: int,
    heads: int,
    dk: int,
    dv: int,
    total_tokens: int,
    read_bytes: float,
    write_bytes: float,
    vector_ops: float,
    state_dtype: str,
) -> OwnerScheduleSpec:
    work_tiles = active_count * max(1, heads)
    owners = min(COMMIT_OWNER_CAPACITY_DEFAULT, work_tiles) if work_tiles > 0 else 0
    max_tiles = ((work_tiles + owners - 1) // owners) if owners > 0 else 0
    per_head_state = int(dk * dv * dtype_bytes(state_dtype))
    ub_live = max(FP32_RECORD_BYTES, per_head_state)
    denom = float(max(1, owners))
    phases: tuple[SerialPhaseSpec, ...] = ()
    if owners > 0:
        phases = (
            SerialPhaseSpec("state_record_load", "mte2", read_bytes / denom, "bytes"),
            SerialPhaseSpec("ssm_replay_step", "vec_alu", vector_ops / denom, "op",
                            dependent_chain_steps=max(1, total_tokens)),
            SerialPhaseSpec("state_publish", "mte3", write_bytes / denom, "bytes"),
        )
    return OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="active_seq_head_replay",
        parallel_axis="active_seq_x_head",
        work_tiles=work_tiles,
        launched_owners=owners,
        active_owners=owners,
        total_owners=COMMIT_OWNER_CAPACITY_DEFAULT,
        max_tiles_per_owner=max_tiles,
        residency=OwnerMemoryResidency(
            ub_live_bytes=ub_live,
            ub_capacity_bytes=0,
            owner_mte2_bytes=(read_bytes / denom) if owners > 0 else 0.0,
            owner_mte3_bytes=(write_bytes / denom) if owners > 0 else 0.0,
            shared_hbm_read_bytes=read_bytes,
            shared_hbm_write_bytes=write_bytes,
            l2_admission_mode="reuse",
        ),
        serial_phases=phases,
        dispatch_envelope_source="P15_dispatch",
    )


def _commit_kernel_binding() -> KernelBinding:
    return KernelBinding(
        kernel_id=COMMIT_KERNEL_ID,
        backend="CANN",
        arch_family="A5",
        submodule="recurrent_state",
        macro_module="attention",
        readiness="shipping",
        recipe_id="recover_ssm_commit",
    )


def _advanced_tokens(view: Any) -> int:
    return commit_tokens_for_view(view)


def _record_bytes_per_token(metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("record_bytes_per_token")
    if declared is not None:
        return float(declared)
    heads = int(metadata.get("heads", 1))
    dk = int(metadata.get("dk", 1))
    dv = int(metadata.get("dv", 1))
    kg_dtype = str(metadata.get("record_kg_dtype", "bf16"))
    return float(heads) * (
        float(dv) * FP32_RECORD_BYTES + 2.0 * float(dk) * float(dtype_bytes(kg_dtype)))


def _state_bytes(metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("state_bytes")
    if declared is not None:
        return float(declared)
    heads = int(metadata.get("heads", 1))
    dk = int(metadata.get("dk", 1))
    dv = int(metadata.get("dv", 1))
    dtype = str(metadata.get("state_dtype", "fp32"))
    return float(heads * dk * dv) * float(dtype_bytes(dtype))


def _decode_speculative(view: Any) -> bool:
    """A verify view is decode-only; a prefill multi-token query is not spec."""
    return (str(getattr(view, "phase", "")) == "decode"
            and int(getattr(view, "q_len", 0) or 0) > 1
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
        active_selected = [(request, count) for request, count in selected if count > 0]
        total = sum(count for _request, count in active_selected)
        geometry = _geometry(metadata)
        if geometry:
            metadata["heads"] = int(geometry.get("heads", 1))
            metadata["dk"] = int(geometry.get("dk", 1))
            metadata["dv"] = int(geometry.get("dv", 1))
            metadata["state_dtype"] = str(geometry.get("state_dtype", "fp32"))
            metadata["record_kg_dtype"] = str(geometry.get("kg_dtype", "bf16"))
        record = _record_bytes_per_token(metadata)
        state = _state_bytes(metadata)
        active_count = len(active_selected)
        read = float(active_count) * state + float(total) * record
        write = float(active_count) * state
        heads = int(metadata.get("heads", 1))
        dk = int(metadata.get("dk", 1))
        dv = int(metadata.get("dv", 1))
        vector = float(REPLAY_OPS_PER_STATE_ELEMENT * total * heads * dk * dv)
        sequence = tuple((request, count) for request, count in selected)
        published_ver = _advance_published_version(
            version, owner, total, explicit=metadata.get("published_state_version"),
        )
        metadata.update({
            "recover_ssm_stage": "commit",
            "recover_ssm_pending_acceptance": False,
            "recover_ssm_intent": False,
            "native_kind": COMMIT_KERNEL_ID,
            "state_slots": active_count,
            "active_sequences": active_count,
            "accepted_per_request": tuple(count for _request, count in selected),
            "query_tokens": sum(
                max(1, int(getattr(view, "q_len", 1)))
                for view in active
                if str(getattr(view, "request_id", "")) in dict(sequence)),
            "committed_tokens": sequence,
            "record_bytes_per_token": record,
            "state_bytes": state,
            "heads": heads,
            "dk": dk,
            "dv": dv,
            "input_state_version": version,
            "published_state_version": published_ver,
            "acceptance_boundary": ACCEPTANCE_BOUNDARY,
            "state_owner": owner,
        })
        owner_schedule = _commit_owner_schedule(
            active_count, heads, dk, dv, total, read, write, vector,
            str(metadata.get("state_dtype", "fp32")),
        )
        result.append(replace(
            op,
            deps=tuple(dict.fromkeys(dependencies)),
            recipe="recover_ssm_commit",
            vector_ops=vector,
            hbm_bytes=read + write,
            mte2_bytes=read,
            mte3_bytes=write,
            kernel_binding=_commit_kernel_binding(),
            owner_schedule=owner_schedule,
            metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[0])),
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
                                          key=lambda pair: pair[0]))))
        else:
            rewritten.append(op)
    return tuple(rewritten)
