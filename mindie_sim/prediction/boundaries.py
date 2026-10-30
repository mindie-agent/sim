"""Software-call expansion into physical kernel boundaries.

[Role] Preserve a source call's kernel DAG before pricing and host submission.
[Invariants] Children use local dependency ids; caller inputs precede every
  child. The original id becomes a zero-work join of all terminal children.
  Each child is a kernel, so eager submission is charged once per child.
[Timeouts & Locks] Finite operator metadata only; no locks or external waits.
[Side-Effect Policy] Returns new immutable operators without device or file I/O.
"""
from dataclasses import replace

from mindie_sim.schema.operators import OpSpec


def expand_call(op, kernels):
    children = tuple(kernels(op))
    ids = {child.op_id: op.op_id + "/" + child.op_id for child in children}
    meta = dict(op.metadata)
    semantic = {**meta.get("semantic_metadata", {}), **meta}
    scope = {key: semantic[key] for key in ("structure_block", "structure_role")
             if key in semantic}
    inherited = {key: value for key, value in op.metadata
                 if key in {"graph_captured", "graph_segment"}}
    expanded = tuple(replace(
        child, op_id=ids[child.op_id],
        deps=tuple(dict.fromkeys((*op.deps, *(ids[dep] for dep in child.deps)))),
        stage=op.stage, layer=op.layer, requests=op.requests, phase=op.phase,
        stream=child.stream or op.stream,
        metadata=tuple({**inherited, **dict(child.metadata), **scope}.items()),
    ) for child in children)
    consumed = {dep for child in children for dep in child.deps}
    terminal = tuple(ids[child.op_id] for child in children if child.op_id not in consumed)
    join = OpSpec(
        op.op_id, "barrier", op.phase, op.stage, op.layer, op.requests,
        op.semantic_shape, op.local_shape, op.padded_shape, op.launch_shape,
        deps=terminal or op.deps, stream=op.stream,
        metadata=(("expanded_implementation", op.recipe), *scope.items()),
    )
    return (*expanded, join)
