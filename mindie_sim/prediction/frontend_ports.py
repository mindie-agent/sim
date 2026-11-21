"""Typed tensor, state, and event ports for model programs.

[Role] Attach boundary ports and service triggers to executable operator work.
[Invariants] Ports describe existing producers and consumers without adding work.
[Timeouts & Locks] A finite scan of the current graph. No locks or I/O.
[Side-Effect Policy] Creates immutable program and port records.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.subgraphs import PortSet
from . import frontend as _frontend


def _declared_format(op, *keys):
    """Read this operator's own representation; never inherit model dtype."""
    metadata = dict(op.metadata)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _ports_for_work(work):
    """Typed boundary ports for one executable model program.

    A barrier has readiness but no tensor payload: it gets an event port only.
    A tensor port uses that boundary's declared input/output format. If the
    operator does not declare one, the port keeps an empty dtype explicitly
    rather than substituting the model activation dtype.
    """
    consumed = {dep for op in work.ops for dep in op.deps}
    tensor_in, tensor_out, state_out = [], [], []
    event_in, event_out = [], []
    for op in work.ops:
        payload = op.family != "barrier"
        input_dtype = _frontend._declared_format(op, "input_format", "activation_format")
        output_dtype = _frontend._declared_format(op, "output_format", "output_dtype")
        if not op.deps:
            if payload:
                tensor_in.append(_frontend.TensorPortSpec(
                    name="input." + op.op_id, direction="in",
                    representation=op.recipe or op.family, dtype=input_dtype,
                    scope="cohort", readiness="external"))
            event_in.append(_frontend.EventPortSpec(
                name="ready." + op.op_id, direction="in", event="ready",
                scope="cohort", readiness="external"))
        if op.op_id not in consumed and op.family not in {
                "barrier", "sampling", "acceptance", "token_commit"}:
            tensor_out.append(_frontend.TensorPortSpec(
                name="output." + op.op_id, direction="out",
                representation=op.recipe or op.family, dtype=output_dtype,
                scope="request", readiness="producer_complete"))
            event_out.append(_frontend.EventPortSpec(
                name="complete." + op.op_id, direction="out", event="complete",
                scope="request", readiness="producer_complete"))
        if op.family == "token_commit":
            event_out.append(_frontend.EventPortSpec(
                name="commit." + op.op_id, direction="out", event="commit",
                scope="request", readiness="producer_complete"))
        elif op.family in {"sampling", "acceptance"}:
            event_out.append(_frontend.EventPortSpec(
                name=("sample." if op.family == "sampling" else "accept.") + op.op_id,
                direction="out", event="complete", scope="request",
                readiness="producer_complete"))
    if any(op.family == "token_commit" for op in work.ops):
        state_out.append(_frontend.StatePortSpec(
            name="state.token_committed", direction="out",
            representation="token_position", scope="request",
            readiness="producer_complete"))
    return _frontend.PortSet(
        tensor_in=tuple(tensor_in), tensor_out=tuple(tensor_out),
        state_out=tuple(state_out), event_in=tuple(event_in),
        event_out=tuple(event_out),
    )


def program_attached_to(program, ops):
    """Reattach one program to its final cohort-bound operator occurrences."""
    work = _frontend.replace(program.work, ops=tuple(ops))
    counts = _frontend.Counter(op.family for op in work.ops)
    triggers = tuple((family, counts[family], _frontend._TRIGGERS[family])
                     for family in sorted(_frontend.SERVICE_FAMILIES) if counts[family])
    return _frontend.replace(program, work=work, ports=_frontend._ports_for_work(work),
                   triggers=triggers)


@_frontend.dataclass(frozen=True)
class ModelProgram:
    """Executable model work plus its typed ports and occurrence triggers.

    ``work`` stays the same ``IterationWork`` consumed by ``Execution.add``;
    the ports/triggers describe assembly only and never add device work.
    """

    work: IterationWork
    ports: PortSet
    triggers: tuple[tuple[str, int, str], ...]
    assembly: str
    layer_scope: str

    def to_dict(self, *, include_ops=False):
        return {
            "assembly": self.assembly,
            "layer_scope": self.layer_scope,
            "valid_tokens": self.work.valid_tokens,
            "padded_tokens": self.work.padded_tokens,
            "mode": self.work.mode,
            "graph_bucket": self.work.graph_bucket,
            "replay_scope": self.work.replay_scope,
            "ports": self.ports.to_dict(),
            "unbound_tensor_ports": [
                port.name for port in (*self.ports.tensor_in, *self.ports.tensor_out)
                if not port.dtype],
            "triggers": [
                {"family": family, "occurrences": count, "trigger": trigger}
                for family, count, trigger in self.triggers
            ],
            **({"ops": [op.to_dict() for op in self.work.ops]}
               if include_ops else {}),
        }
