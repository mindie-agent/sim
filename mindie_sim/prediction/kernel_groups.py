"""Shared physical invocation boundaries for a resource-phase graph.

[Role] Bind one native kernel's phase DAG to one startup and completion floor.
[Invariants] Phases keep their work and dependencies. The floor is concurrent
  with startup plus the DAG, never a delay before useful service. Native kernel
  identity is separate from each phase recipe. Declared group buffers stay live
  from invocation admission through every phase and the completion floor.
  State belongs to one add call.
[Timeouts & Locks] Finite active graph state and engine events; no I/O or locks.
[Side-Effect Policy] Submits only to the owning Execution engine.
"""
from dataclasses import dataclass, field, replace

from mindie_sim.schema.common import ModelingError


def declaration(op):
    return dict(op.metadata).get("device_kernel_group")


@dataclass
class KernelGroup:
    members: list
    startup: object = None
    completion: object = None
    ready: object = None
    entry_context: object = None
    entry_recipe: object = None
    contexts: list = field(default_factory=list)
    finishes: list = field(default_factory=list)
    streams: dict = field(default_factory=dict)
    buffers: tuple = ()


class KernelGroups:
    """Compile groups without putting iteration history on Execution."""

    def __init__(self, execution, ops, replica, prefix):
        self.execution, self.replica, self.prefix = execution, replica, prefix
        self.groups = {}
        self.owner = f"replica:{replica}"
        by_id = {op.op_id: op for op in ops}
        for op in ops:
            info = declaration(op)
            if info is not None:
                if not isinstance(info, dict) or not info.get("id") or not info.get("kernel"):
                    raise ModelingError("device_kernel_group_requires_id_and_kernel")
                self.groups.setdefault(info["id"], KernelGroup([])).members.append(op)
        for group_id, group in self.groups.items():
            first, last = group.members[0], group.members[-1]
            entries = [op for op in group.members if declaration(op).get("entry")]
            exits = [op for op in group.members if declaration(op).get("exit")]
            if entries != [first] or exits != [last]:
                raise ModelingError(f"device_kernel_group_requires_ordered_entry_and_exit:{group_id}")
            if len({(op.stage, declaration(op)["kernel"]) for op in group.members}) != 1:
                raise ModelingError(f"device_kernel_group_identity_or_stage_mismatch:{group_id}")
            group.buffers = tuple(declaration(first).get("buffers", ()))
            if any(declaration(op).get("buffers") for op in group.members[1:]):
                raise ModelingError("device_kernel_group_buffers_belong_to_entry")
            if len({buffer["id"] for buffer in group.buffers}) != len(group.buffers):
                raise ModelingError("device_kernel_group_buffer_ids_must_be_unique")
            stem = f"{prefix}:kernel:{group_id}"
            group.startup = execution.engine.future(stem + ":started")
            group.completion = execution.engine.future(stem + ":complete")
        for op in ops:
            info = declaration(op)
            for dependency in op.deps:
                parent = by_id.get(dependency)
                parent_info = declaration(parent) if parent is not None else None
                if (parent_info and (info is None or info["id"] != parent_info["id"])
                        and not parent_info.get("exit")):
                    raise ModelingError(f"device_kernel_group_external_dependency_requires_exit:{dependency}")

    def group(self, op):
        info = declaration(op)
        return self.groups[info["id"]] if info else None

    def buffer_scope(self, op):
        """Only a real grouped execution supplies this lifetime to its phases."""
        group = self.group(op)
        return tuple(buffer["id"] for buffer in group.buffers) if group else ()

    def entry_dependencies(self, op, ids):
        info = declaration(op)
        if not info or not info.get("entry"):
            return ()
        group = self.group(op)
        members = {member.op_id for member in group.members}
        dependencies = {dep for member in group.members for dep in member.deps if dep not in members}
        unavailable = dependencies - ids.keys()
        if unavailable:
            raise ModelingError("device_kernel_group_inputs_must_precede_entry:" + ",".join(sorted(unavailable)))
        # Every branch reserves its external stream until whole-kernel exit.
        # Its prior tail therefore belongs to invocation readiness, not just
        # to a later phase: startup/minimum must not elapse while a required
        # branch stream is still occupied by a preceding producer.
        streams = {(self.replica, member.stage, member.stream or "main")
                   for member in group.members}
        tails = tuple(self.execution.streams[key] for key in sorted(streams)
                      if key in self.execution.streams)
        return (*tuple(ids[key] for key in sorted(dependencies)), *tails)

    def stream_tail(self, op, key):
        group = self.group(op)
        if group is None:
            return self.execution.streams.get(key)
        if not group.streams:
            # Reserve the external stream until the whole kernel completes;
            # phase-local tails still allow its different streams to overlap.
            # Reserve all branches at entry: interleaved compilation of two
            # kernels must not reverse their order on a later branch stream.
            for member in group.members:
                stream = self.replica, member.stage, member.stream or "main"
                if stream not in group.streams:
                    group.streams[stream] = self.execution.streams.get(stream)
                    self.execution.streams[stream] = group.completion
        return group.streams[key]

    def phase_ready(self, op, ready, name):
        group = self.group(op)
        if group is None:
            return ready
        if declaration(op).get("entry"):
            group.ready = ready
        return self.execution.engine.submit(name + ":phase_ready", after=(ready, group.startup),
                                            owner=self.owner)

    def finish(self, op, context, recipe, complete, key):
        group = self.group(op)
        if group is None:
            self.execution.streams[key] = complete
            return complete
        info = declaration(op)
        group.streams[key] = complete
        group.contexts.append(context)
        group.finishes.append(complete)
        if info.get("entry"):
            group.entry_context, group.entry_recipe = context, recipe
        if not info.get("exit"):
            return complete
        # Selection sees the actual prepared resource work from every phase,
        # including zero-work entry kernels with a nonempty later body.
        native, work = {}, {}
        for phase in group.contexts:
            for source, target in ((phase.native_work, native), (phase.work, work)):
                for resource, value in source.items():
                    target[resource] = target.get(resource, 0) + value
        entry = group.entry_context
        metadata = {**dict(entry.op.metadata), "kernel": info["kernel"]}
        # Link work belongs to the transport phases and does not erase a fused
        # native kernel's real AIC/AIV execution class. This copy is used only
        # for invocation-class selection; no executed demand is removed.
        execution_classes = {key: value for key, value in native.items() if key != "link"}
        invocation = replace(
            entry, op=replace(entry.op, metadata=tuple(sorted(metadata.items()))),
            native_work=execution_classes, work=work, service_boundary="device_kernel", minimum_ms=0,
            service_identity={**entry.service_identity, "kernel": info["kernel"]})
        selected = self.execution.invocations.select(
            invocation, group.entry_recipe,
            devices=self.execution.stage_dies[self.replica, op.stage])
        startup_ms, minimum_ms = 0, 0
        if selected is not None:
            timing, record = selected
            self.execution.cost.record_invocation(record)
            startup_ms, minimum_ms = timing.startup_ms, timing.minimum_ms
        # A measured native response owns the whole call, including internal
        # source phases. It never becomes another per-phase startup or delay.
        regime_context = replace(invocation, native_work=native)
        minimum_ms = max(minimum_ms, self.execution.regimes.select(
            regime_context, self.execution.cost.record_regime))
        engine = self.execution.engine
        stem = f"{self.prefix}:kernel:{info['id']}"
        allocations = tuple(entry.allocate(
            f"{stem}:buffer:{buffer['id']}", buffer["pool"], buffer["bytes"], buffer["units"])
            for buffer in group.buffers)
        # A blocked UB reservation is not useful kernel execution. Start both
        # timing constraints only after the complete declared buffer set fits.
        admitted = (engine.submit(stem + ":buffers_acquire", after=(group.ready,),
                                  acquire=allocations, owner=self.owner)
                    if allocations else group.ready)
        engine.submit(stem + ":startup", after=(admitted,), latency=startup_ms,
                      owner=self.owner,
                      on_complete=lambda _, event=group.startup: engine.resolve(event))
        dependencies = list(group.finishes)
        if minimum_ms:
            dependencies.append(engine.submit(stem + ":minimum", after=(admitted,),
                                               latency=minimum_ms, owner=self.owner))
        engine.submit(stem + ":retire", after=tuple(dependencies), free=allocations, owner=self.owner,
                      on_complete=lambda _, event=group.completion: engine.resolve(event))
        return group.completion
