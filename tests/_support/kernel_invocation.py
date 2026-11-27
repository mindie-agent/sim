"""Shared builders for native kernel invocations, phase groups and services.

[Role] One definition of the small raw-case / OpSpec / recipe builders that
  runtime pipeline, invocation-boundary and service tests share: a one-copy
  ``OpSpec`` with its declared HBM ``ServiceCurve``, invocation-class
  declarations, the four-phase ``device_kernel_group`` fork, the shared-buffer
  phase group with its finite-buffer recipe, and the owner-local source
  program executor.
[Invariants] Builders only declare work, placements and timing classes; they
  never read predictions back as expectations.  ``source_execute`` asserts the
  Engine stays idle before the external release and drains every task,
  program and allocation afterwards.
[Timeouts & Locks] None; pure in-memory construction and Engine runs.
[Side-Effect Policy] Side-effect-free; writes nothing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.service_regime_domains.services import ServiceCurve, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec

from _support.hardware import hardware_case

#: Kernel-scoped scratch buffer id shared by every phase of ``group``.
BUFFER = "shared_scratch"


def op(name, *, flops=0, memory=0, deps=(), stream="main", stage=0):
    """A neutral prefill OpSpec carrying only FLOPs / HBM bytes / deps."""
    return OpSpec(name, "test", "prefill", stage, 0, (), (1,), (1,), (1,), (1,),
                  flops=flops, hbm_bytes=memory, deps=deps, dtype="bf16", stream=stream)


def copy_op(work=2000, **changes):
    """One contiguous HBM copy moving ``work`` bytes through recipe ``copy``."""
    return replace(OpSpec(
        "copy", "copy", "prefill", 0, 0, ("r",),
        (work // 2,), (work // 2,), (work // 2,), (work // 2,),
        hbm_bytes=work, dtype="bf16", layout="contiguous", recipe="copy",
    ), **changes)


def curve(case=None, op=None, **changes):
    """Declared HBM service curve: 1 ms startup, knots (1000 B, 2 ms), (3000 B, 4 ms)."""
    case = case or parse_case(hardware_case())
    op = op or copy_op()
    values = dict(
        resource="hbm", work_unit="bytes", identity=service_identity(case, op),
        boundary="device_kernel", active_units=4, startup_ms=1,
        knots=((1000, 2), (3000, 4)),
    )
    return ServiceCurve(**{**values, **changes})


def declare(raw, *, startup=3, minimum=0, match=None):
    """Declare one runtime invocation class (startup/minimum ms) on ``raw``."""
    raw.setdefault("runtime", {}).setdefault("kernels", {})["invocations"] = [{
        "match": match or {}, "startup_ms": startup, "minimum_ms": minimum,
        "provenance": {"kind": "independent_causal_control"},
    }]
    return raw


def lower_copy(plan, model, runtime, machine):
    """Graph lowering that emits exactly one 4000-byte copy."""
    return IterationWork(
        (copy_op(4000),), plan, 1, 1, 1, None, "prefill", "invocation_control")


def phase(name, *, group="dispatch", entry=False, exit=False, **kwargs):
    """One phase of native kernel ``NativeDispatch`` inside device kernel group ``group``."""
    operation = op(name, **kwargs)
    return replace(operation, metadata=(("device_kernel_group", {
        "id": group, "kernel": "NativeDispatch", "entry": entry, "exit": exit}),))


def fork(group="dispatch", dependency=()):
    """Entry (1000 B) -> {count 2000 vec, payload 4000 B} -> exit (1000 vec)."""
    return (
        phase(group + ".entry", group=group, entry=True, memory=1000, deps=dependency),
        replace(phase(group + ".count", group=group, deps=(group + ".entry",),
                      stream="count"), vector_ops=2000),
        phase(group + ".payload", group=group, memory=4000,
              deps=(group + ".entry",), stream="payload"),
        replace(phase(group + ".exit", group=group, exit=True,
                      deps=(group + ".count", group + ".payload")), vector_ops=1000),
    )


def raw_case(*, startup=0, minimum=0):
    """Two vector/UB units (1024 B UB each) plus invocation class ``group_test``."""
    raw = hardware_case()
    raw["machine"]["device"].update(
        buffers={"ub": 1024}, buffer_units={"ub": 2},
        resource_units={"vector": 2}, rates={"memory": 1e6, "vector": 2e6},
        kernel_invocations={"group_test": {"startup_ms": startup, "minimum_ms": minimum}})
    return raw


def buffer_recipe(operation, geometry):
    """Vector service that holds a whole UB unit unless the group already owns it."""
    owner = dict(operation.metadata).get("owner", 0)
    work = Work(resources={"vector": operation.vector_ops})

    def program(context):
        allocations = (() if BUFFER in context.group_buffers else
                       (context.allocate("local", "ub", 1024, (owner,)),))
        return KernelProgram((Stage(
            "service", context.bind(work),
            placement=context.placement("vector", (owner,)),
            acquire=allocations, free=allocations),))

    return OperatorPlan(work, program)


def group(name="group"):
    """entry(+kernel buffer on unit 0) -> first (1000 vec) -> last (2000 vec) -> exit."""
    entry = phase(name + ".entry", group=name, entry=True)
    declaration = {**dict(entry.metadata)["device_kernel_group"],
                   "buffers": [{"id": BUFFER, "pool": "ub", "bytes": 1024, "units": [0]}]}
    entry = replace(entry, metadata=(("device_kernel_group", declaration),
                                     ("invocation_class", "group_test")))
    first = replace(phase(name + ".first", group=name, deps=(name + ".entry",),
                          stream=name + ".first"), vector_ops=1000, recipe="buffered")
    last = replace(phase(name + ".last", group=name, deps=(name + ".first",),
                         stream=name + ".last"), vector_ops=2000, recipe="buffered")
    return (entry, first, last,
            phase(name + ".exit", group=name, exit=True, deps=(name + ".last",)))


def recipes():
    return RecipeBook({"buffered": PlannedRecipe(buffer_recipe)})


def competitor(owner, *, deps=("late",)):
    """An independent 1000-op vector stream pinned to UB/vector unit ``owner``."""
    return replace(op("competitor", deps=deps, stream="independent"), vector_ops=1000,
                   recipe="buffered", metadata=(("owner", owner),))


def source_execute(case, op):
    """Run one source-lowered program released at t=7 ms; return (span, engine, op)."""
    case = parse_case(case)
    execution = Execution(case, CostModel(case), recipes=RecipeBook.from_mapping(case.runtime.kernels))
    op, recipe = execution.prepare(op, 0)
    context = execution._context(op, 0, None, recipe=recipe)
    engine = execution.engine
    engine.trace = deque(maxlen=12000)
    ready = engine.submit("late", release=7)
    end = execution.lower_program(context, recipe).start(engine, "source", ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(end,))
    assert not engine.summary()["live_tasks"] and not engine.summary()["live_programs"]
    assert not any(engine.allocated.values())
    return end.time - 7, engine, op


execute = source_execute
