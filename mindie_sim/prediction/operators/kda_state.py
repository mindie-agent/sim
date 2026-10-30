"""KDA prefill FWD_H state phase on the shared mixed-core resource engine.

[Role]
  Bind the retained vLLM-Ascend KDA FWD_H source phase to a bounded per-stream
  chunk pipeline.  The semantic KDA boundary and its external gate/QK
  normalization stay with the caller; this module owns only recurrent state.

[Invariants]
  * Source domain is BF16 K/V/W/U, FP32 key-wise gates and FP32 state,
    ``dk == dv == 128``, ``chunk == 64``.  Unsupported branches are errors.
  * Each ``(sequence, value_head)`` is an independent state stream.  Chunks are
    serial: C1 ``W @ H`` -> vector ``v_new = U - V_work`` -> C2
    ``K_decay^T @ v_new`` -> vector ``H = gate_last * H + H_work``.
  * The internal state is ``[K, V]`` FP32.  The caller's public ``[V, K]``
    transpose is outside this selected phase.
  * A core owns at most two live source streams (ping-pong slots), assigned by
    the source task stride.  Other cores or AIV pairs are never added as supply.
    Vector phases use aggregate pair supply; odd V1 tails are not lane-exact.
  * Work is interval arithmetic over real chunk tails; programs are lazy and
    retain only the active stream/chunk stages.

[Timeouts & Locks]
  Pure metadata and arithmetic.  No I/O, locks, device contexts or threads.

[Side-Effect Policy]
  Returns immutable plans and submits only to the caller's virtual engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .mixed_core import placement, require_pair, split_reads

RECIPE = "vllm_ascend_kda_fwd_h"
IMPLEMENTATION = "vllm_ascend_kda_chunk_state"

# Arch35 FWD_H tile shape and KDA wrapper constants.
_K_DIM = 128
_V_DIM = 128
_CHUNK = 64
_BF16_BYTES = 2
_FP32_BYTES = 4
_STREAMS_PER_CORE = 2
_IMPLEMENTATION = "chunk_kda_fwd/arch35/FWD_H/k_gated/scalar_off/exp2/state_fp32"

_PARAM_KEYS = frozenset({
    "lengths", "heads", "key_heads", "dk", "dv", "chunk", "dtype",
    "gate_dtype", "state_dtype", "use_initial_state", "output_final_state",
    "state_v_first", "use_exp2", "save_new_value",
})


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive_int(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(reason)
    return value


def _flag(params: Mapping[str, Any], name: str, default: bool) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ModelingError(f"kda_state_parameter_must_be_boolean:{name}")


def _dtype(value: Any, allowed: Mapping[str, str], reason: str) -> str:
    key = str(value).lower()
    if key not in allowed:
        raise ModelingError(reason)
    return allowed[key]


@dataclass(frozen=True)
class Params:
    lengths: tuple[int, ...]
    heads: int
    key_heads: int
    dk: int
    dv: int
    chunk: int

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def chunks_per_sequence(self) -> tuple[int, ...]:
        return tuple(_ceil_div(length, self.chunk) for length in self.lengths)

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_per_sequence)

    @property
    def head_tasks(self) -> int:
        return len(self.lengths) * self.heads

    def metadata(self) -> dict[str, Any]:
        per_core_workspace = {
            "v_work": 2 * self.chunk * self.dv * _FP32_BYTES,
            "v_update": 2 * self.chunk * self.dv * _FP32_BYTES,
            "k_decay": 2 * self.chunk * self.dk * _FP32_BYTES,
            "h_work": 2 * self.dk * self.dv * _FP32_BYTES,
        }
        return {
            "lengths": self.lengths,
            "heads": self.heads,
            "key_heads": self.key_heads,
            "group_size": self.heads // self.key_heads,
            "dk": self.dk,
            "dv": self.dv,
            "chunk": self.chunk,
            "dtype": "bf16",
            "gate_dtype": "fp32",
            "state_dtype": "fp32",
            "use_initial_state": True,
            "output_final_state": True,
            "state_v_first": True,
            "internal_state_layout": "K,V",
            "public_state_layout": "V,K",
            "caller_state_transpose": "outside_selected_fwd_h_phase",
            "use_exp2": True,
            "save_new_value": True,
            "implementation": _IMPLEMENTATION,
            "source_phase": "FWD_H",
            "native_kind": "ChunkKdaFwd/FWD_H",
            "gate_mode": "key_wise_fp32",
            "scalar_gate": False,
            "k_gated": True,
            "tile_shapes": "GDNFwdHTileShapes128",
            "streams_per_core": _STREAMS_PER_CORE,
            "workspace_bytes_per_core": sum(per_core_workspace.values()),
            "workspace_per_core": per_core_workspace,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "chunks_per_sequence": self.chunks_per_sequence,
            "head_tasks": self.head_tasks,
            "state_checkpoints_per_stream": "one_fp32_write_per_chunk",
            "h_snapshot_writes_per_stream": "one_per_chunk_including_initial_h0",
            "tail_local_l1_zero_fill_bytes": (
                sum(length % self.chunk != 0 for length in self.lengths)
                * self.heads * 4 * 128 * 128 * _BF16_BYTES),
            "tail_local_l1_zero_fill_service": "not_timed_no_declared_fill_resource",
            "schedule_scope": "owner_phase_order_with_global_h0_barrier",
            "vector_lane_scope": "aggregate_aiv_pair_odd_v1_tail_not_lane_exact",
            "dispatch": {
                "implementation": _IMPLEMENTATION,
                "source_phase": "FWD_H",
                "tile_shapes": "GDNFwdHTileShapes128",
                "k_gated": True,
                "scalar_gated": False,
                "use_exp2": True,
                "state_dtype": "fp32",
                "workspace_dtype": "fp32",
                "use_initial_state": True,
                "output_final_state": True,
            },
        }


def _parse_params(params: Mapping[str, Any]) -> Params:
    if not isinstance(params, Mapping):
        raise ModelingError("kda_state_params_must_be_a_mapping")
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise ModelingError("kda_state_unknown_parameter:" + ",".join(sorted(unknown)))

    raw_lengths = params.get("lengths")
    if (isinstance(raw_lengths, (str, bytes))
            or not isinstance(raw_lengths, (list, tuple)) or not raw_lengths):
        raise ModelingError("kda_state_lengths_must_be_a_nonempty_integer_list")
    lengths = tuple(_positive_int(value, "kda_state_lengths_must_be_positive")
                    for value in raw_lengths)

    heads = _positive_int(params.get("heads"), "kda_state_heads_must_be_positive")
    key_heads = _positive_int(params.get("key_heads"),
                              "kda_state_key_heads_must_be_positive")
    if key_heads != heads:
        raise ModelingError("kda_state_requires_equal_key_and_value_heads")
    dk = _positive_int(params.get("dk"), "kda_state_dk_must_be_positive")
    dv = _positive_int(params.get("dv"), "kda_state_dv_must_be_positive")
    chunk = _positive_int(params.get("chunk"), "kda_state_chunk_must_be_positive")
    if dk != _K_DIM:
        raise ModelingError(f"kda_state_requires_k_head_dim:{_K_DIM}")
    if dv != _V_DIM:
        raise ModelingError(f"kda_state_requires_v_head_dim:{_V_DIM}")
    if chunk != _CHUNK:
        raise ModelingError(f"kda_state_requires_chunk_size:{_CHUNK}")

    _dtype(params.get("dtype", "bf16"),
           {"bf16": "bf16", "bfloat16": "bf16"}, "kda_state_requires_bf16_inputs")
    _dtype(params.get("gate_dtype", "fp32"),
           {"fp32": "fp32", "float32": "fp32"}, "kda_state_requires_fp32_gates")
    _dtype(params.get("state_dtype", "fp32"),
           {"fp32": "fp32", "float32": "fp32"}, "kda_state_requires_fp32_state")

    flags = {
        "use_initial_state": True,
        "output_final_state": True,
        "state_v_first": True,
        "use_exp2": True,
        "save_new_value": True,
    }
    for name, required in flags.items():
        if not _flag(params, name, required):
            raise ModelingError(f"kda_state_requires_flag:{name}")
    return Params(lengths, heads, key_heads, dk, dv, chunk)


def _params_from_metadata(metadata: Mapping[str, Any]) -> Params:
    return _parse_params({key: metadata[key] for key in _PARAM_KEYS if key in metadata})


def state_offset(sequence, head, key, value, heads):
    """Internal FP32 state address: sequence-major, then head, then [K, V]."""
    return ((sequence * heads + head) * _K_DIM + key) * _V_DIM + value


def h_snapshot_offset(head, total_chunks, chunk_prefix, chunk, key, value):
    """BF16 H snapshot address in the source varlen head/chunk workspace."""
    return (((head * total_chunks + chunk_prefix + chunk) * _K_DIM + key)
            * _V_DIM + value)


# ---------------------------------------------------------------------------
# Source phase work.  Every byte and operation below is derived from the
# kernel's C1/V1/C2/V2 tensor intervals; no rate, floor or measured span enters.
# ---------------------------------------------------------------------------

def _work(*, matrix: float = 0.0, read: float = 0.0, write: float = 0.0,
          vector: float = 0.0, mte1: float = 0.0,
          fixpipe: float = 0.0) -> Work:
    resources = {
        "hbm": read + write + fixpipe,
        "mte2": read,
        "mte3": write,
        "vector": vector,
        "mte1": mte1,
        "fixpipe": fixpipe,
    }
    return Work({"bf16": matrix} if matrix else {},
                {key: value for key, value in resources.items() if value})


@dataclass(frozen=True)
class ChunkWork:
    """Source phases for one stream chunk; ``init`` exists only on chunk 0."""

    init: Work
    c1: Work
    delta: Work
    c2: Work
    state: Work

    def total(self) -> Work:
        return self.init + self.c1 + self.delta + self.c2 + self.state


def chunk_work(params: Params, valid: int, *, first: bool, last: bool) -> ChunkWork:
    """Physical work for one ``(sequence, head, chunk)`` source task."""
    k, v, chunk = params.dk, params.dv, params.chunk
    h_snapshot = k * v * _BF16_BYTES
    state = k * v * _FP32_BYTES
    v_work = valid * v * _FP32_BYTES
    k_work = valid * k * _BF16_BYTES
    h_work = k * v * _FP32_BYTES
    gate_last = k * _FP32_BYTES

    init = _work(
        read=state, write=h_snapshot, vector=k * v,
    ) if first else _work()

    # C1: V_work = W @ H. GM payload and MMAD arguments use real valid rows.
    # Tail L1 zero-fill is a separate local operation recorded in metadata;
    # it does not create padded GM reads or proven matrix instructions.
    c1 = _work(
        matrix=2 * valid * k * v,
        read=valid * k * _BF16_BYTES + h_snapshot,
        vector=0,
        mte1=2 * (valid * k + k * v),
        fixpipe=v_work,
    )

    # V1: V_new = U - V_work; K is copied to the decay workspace, and the
    # BF16 V_update operand is published for C2.
    delta = _work(
        read=valid * v * _BF16_BYTES + v_work + k_work,
        write=2 * valid * v * _BF16_BYTES + k_work,
        vector=4 * valid * v,
    )

    # C2: H_work = K_decay^T @ V_update.
    c2 = _work(
        matrix=2 * k * valid * v,
        read=k_work + valid * v * _BF16_BYTES,
        mte1=2 * (valid * k + valid * v),
        fixpipe=h_work,
    )

    # V2: H = exp2(g_last) * H + H_work.  The state is read/written through GM
    # on every chunk; non-final chunks also publish the BF16 snapshot C1 reads.
    # The initial H0 conversion is the separate ``init`` phase above.
    state_read = h_work + gate_last + state
    state_write = state + (h_snapshot if not last else 0)
    gate_ops = 2 * k
    state_vector = gate_ops + k * v * (2 + (0 if last else 1))
    state_phase = _work(
        read=state_read, write=state_write, vector=state_vector,
    )
    return ChunkWork(init, c1, delta, c2, state_phase)


# ---------------------------------------------------------------------------
# Source stream ownership.  The scheduler starts each active core at
# ``core * slots + slot`` and advances by ``core_count * slots``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamTask:
    core: int
    slot: int
    sequence: int
    head: int
    length: int
    chunks: int
    chunk_offset: int


@dataclass(frozen=True)
class CoreTasks:
    core: int
    rounds: tuple[tuple[StreamTask, ...], ...]

    @property
    def streams(self) -> tuple[StreamTask, ...]:
        return tuple(task for round in self.rounds for task in round)


def head_ownership(lengths, heads: int, core_count: int, *,
                   chunk: int = _CHUNK) -> tuple[CoreTasks, ...]:
    """Resolve the source task queue into at most two live streams per core."""
    lengths = tuple(lengths)
    if not lengths or any(isinstance(value, bool) or not isinstance(value, int)
                          or value < 1 for value in lengths):
        raise ModelingError("kda_state_lengths_must_be_positive")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ModelingError("kda_state_heads_must_be_positive")
    if isinstance(core_count, bool) or not isinstance(core_count, int) or core_count < 1:
        raise ModelingError("kda_state_requires_positive_compute_units")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ModelingError("kda_state_chunk_must_be_positive")

    total = len(lengths) * heads
    slots = _STREAMS_PER_CORE if total > core_count else 1
    stride = core_count * slots
    prefixes = [0]
    for length in lengths:
        prefixes.append(prefixes[-1] + _ceil_div(length, chunk))

    result = []
    for core in range(core_count):
        rounds = []
        base = core * slots
        while base < total:
            tasks = []
            for slot in range(slots):
                index = base + slot
                if index >= total:
                    continue
                sequence, head = divmod(index, heads)
                tasks.append(StreamTask(
                    core=core, slot=slot, sequence=sequence, head=head,
                    length=lengths[sequence],
                    chunks=_ceil_div(lengths[sequence], chunk),
                    chunk_offset=prefixes[sequence],
                ))
            if tasks:
                rounds.append(tuple(tasks))
            base += stride
        if rounds:
            result.append(CoreTasks(core, tuple(rounds)))
    return tuple(result)


def stream_work(params: Params, task: StreamTask) -> Work:
    """Sum the real chunk tails of one independent source state stream."""
    total = Work()
    for index in range(task.chunks):
        valid = min(params.chunk, task.length - index * params.chunk)
        total += chunk_work(
            params, valid, first=index == 0, last=index == task.chunks - 1).total()
    return total


# ---------------------------------------------------------------------------
# Bounded mixed-core program.
# ---------------------------------------------------------------------------

def _phases(name, work: Work, context: RecipeContext, core: int, *,
            aiv_pair: bool = False, after=(), previous=()):
    """Expose one source phase as read / inner-copy / compute / publish / write.

    The copy/vector side is pinned to both AIVs of the owning AIC.  The matrix
    side is pinned to that AIC.  No other core or AIV is admitted as supply.
    """
    resources = dict(work.resources)
    read = resources.pop("mte2", 0)
    write = resources.pop("mte3", 0)
    memory = resources.pop("hbm", 0)
    mte1 = resources.pop("mte1", 0)
    fixpipe = resources.pop("fixpipe", 0)
    role = read_role(context.resource_names, 0 if aiv_pair else None)
    if aiv_pair:
        owned = context.owned(core, context.core_count,
                              ("vector", role, "mte3"))
    else:
        owned = placement(context, core)

    gm_read = min(read, memory)
    gm_write = memory - gm_read
    prefix = name + ":"
    load = Stage(prefix + "load",
                 context.bind(Work(resources={"hbm": gm_read, role: read})),
                 after=after, previous=previous, placement=owned)
    stages = [load]
    tail = load
    if mte1:
        tail = Stage(prefix + "l0",
                     context.bind(Work(resources={"mte1": mte1})),
                     after=(load.name,), placement=owned)
        stages.append(tail)
    compute = Stage(prefix + "compute",
                    context.bind(Work(work.matrix, resources)),
                    after=(tail.name,), placement=owned)
    stages.append(compute)
    tail = compute
    if fixpipe:
        fix_gm = gm_write if not write else 0
        gm_write -= fix_gm
        publish = Stage(prefix + "publish",
                        context.bind(Work(resources={"fixpipe": fixpipe,
                                                     "hbm": fix_gm})),
                        after=(tail.name,), placement=owned)
        stages.append(publish)
        tail = publish
    stages.append(Stage(
        prefix + "store",
        context.bind(Work(resources={"hbm": gm_write, "mte3": write})),
        after=(tail.name,), placement=owned))
    stages.append(Stage(name, after=(prefix + "store",)))
    return tuple(stages)


@dataclass(frozen=True)
class SlotQueue:
    """Finite stream descriptors; chunk indexes are resolved without expansion."""

    tasks: tuple[StreamTask, ...]
    ends: tuple[int, ...]

    def at(self, iteration: int) -> tuple[StreamTask, int] | None:
        index = bisect_right(self.ends, iteration)
        if index == len(self.tasks):
            return None
        start = self.ends[index - 1] if index else 0
        return self.tasks[index], iteration - start


def slot_queues(group: CoreTasks) -> tuple[SlotQueue, ...]:
    """Source InitTasks advances each slot after its own final chunk."""
    queues = []
    for slot in range(_STREAMS_PER_CORE):
        tasks = tuple(task for task in group.streams if task.slot == slot)
        ends, count = [], 0
        for task in tasks:
            count += task.chunks
            ends.append(count)
        if tasks:
            queues.append(SlotQueue(tasks, tuple(ends)))
    return tuple(queues)


def _core_program(group: CoreTasks, params: Params,
                  context: RecipeContext) -> PipelineProgram:
    queues = slot_queues(group)

    def tile(index):
        chunks = []
        for queue in queues:
            entry = queue.at(index)
            if entry is None:
                continue
            task, chunk = entry
            valid = min(params.chunk, task.length - chunk * params.chunk)
            work = chunk_work(params, valid, first=False,
                              last=chunk == task.chunks - 1)
            chunks.append((task.slot, work))
        stages = []
        # The source AIC issues C1 for both slots before C2 for both slots;
        # the AIV pair does the analogous V1/V2 ordering. Different processors
        # may overlap, but a slot cannot overwrite its prior state or scratch.
        for phase, aiv in (("c1", False), ("delta", True),
                           ("c2", False), ("state", True)):
            phase_tail = None
            for slot, work in chunks:
                name = f"{phase}s{slot}"
                after = (() if phase_tail is None else (phase_tail,))
                previous = ()
                if phase == "c1":
                    previous = (f"states{slot}",)
                    if phase_tail is None:
                        previous += ("cube_done",)
                elif phase == "delta":
                    after += (f"c1s{slot}",)
                    if phase_tail is None:
                        previous = ("vector_done",)
                elif phase == "c2":
                    after += (f"deltas{slot}", f"c1s{chunks[-1][0]}")
                else:
                    after += (f"c2s{slot}", f"deltas{chunks[-1][0]}")
                stages.extend(_phases(
                    name, getattr(work, phase), context, group.core,
                    aiv_pair=aiv, after=after, previous=previous))
                phase_tail = name
        stages.append(Stage("cube_done", after=(f"c2s{chunks[-1][0]}",)))
        stages.append(Stage("vector_done", after=(f"states{chunks[-1][0]}",)))
        return tuple(stages)

    return PipelineProgram(max(queue.ends[-1] for queue in queues), 2, tile)


def _program(params: Params, groups: tuple[CoreTasks, ...],
             context: RecipeContext, core_count: int):
    if context.resource_units.get("vector", core_count) != 2 * core_count:
        raise ModelingError("kda_state_requires_two_vector_subblocks_per_matrix_core")
    if int(context.core_count) != core_count:
        raise ModelingError("kda_state_geometry_changed_after_build")
    init_programs = []
    for group in groups:
        # H0 for every stream is materialized before the source SyncAll. The
        # initialization is a per-owner resource phase, not individual casts.
        work = _work(read=params.dk * params.dv * _FP32_BYTES,
                     write=params.dk * params.dv * _BF16_BYTES,
                     vector=params.dk * params.dv).scaled(len(group.streams))
        init_programs.append(KernelProgram(_phases(
            "init", work, context, group.core, aiv_pair=True)))
    return KernelProgram((
        Stage("initialize", program=ParallelProgram(tuple(init_programs))),
        Stage("scan", after=("initialize",), program=ParallelProgram(tuple(
            _core_program(group, params, context) for group in groups))),
    ))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower the selected FWD_H phase onto one shared resource engine."""
    params = _params_from_metadata(dict(op.metadata))
    require_pair(geometry)
    core_count = int(geometry.compute_units)
    if core_count < 1:
        raise ModelingError("kda_state_requires_positive_compute_units")
    groups = head_ownership(params.lengths, params.heads, core_count,
                            chunk=params.chunk)

    total = Work()
    aic_mte2 = 0
    for group in groups:
        for task in group.streams:
            for index in range(task.chunks):
                valid = min(params.chunk, task.length - index * params.chunk)
                work = chunk_work(
                    params, valid, first=index == 0,
                    last=index == task.chunks - 1)
                total += work.total()
                aic_mte2 += (work.c1.resources.get("mte2", 0)
                             + work.c2.resources.get("mte2", 0))
    total = split_reads(total, aic_mte2, geometry.resource_units)

    def program(context: RecipeContext):
        return _program(params, groups, context, core_count)

    return OperatorPlan(total, program)


def make_op(params: Mapping[str, Any], *, op_id: str = "kda_fwd_h",
            deps=(), requests=(), stage: int = 0, layer: int = 0,
            phase: str = "prefill", stream: str = "",
            notes: tuple[str, ...] = ()) -> OpSpec:
    """Create the semantic record for one actual FWD_H kernel launch."""
    parsed = _parse_params(params)
    meta = parsed.metadata()
    # Aggregate source work is independent of the core count; per-core
    # partition and residency are resolved by ``build``.
    total = Work()
    chunk_offset = 0
    for sequence, length in enumerate(parsed.lengths):
        chunks = parsed.chunks_per_sequence[sequence]
        for head in range(parsed.heads):
            task = StreamTask(
                core=0, slot=0, sequence=sequence, head=head, length=length,
                chunks=chunks, chunk_offset=chunk_offset)
            total += stream_work(parsed, task)
        chunk_offset += chunks
    chunk_tokens = sum(count * parsed.chunk
                       for count in parsed.chunks_per_sequence)
    semantic = (parsed.total_tokens, parsed.heads, parsed.dk, parsed.dv)
    return OpSpec(
        op_id=op_id,
        family="recurrent_scan",
        phase=phase,
        stage=stage,
        layer=layer,
        requests=tuple(requests),
        deps=tuple(deps),
        semantic_shape=semantic,
        local_shape=semantic,
        padded_shape=(1, parsed.heads, chunk_tokens, parsed.dv),
        launch_shape=semantic,
        dtype="bf16",
        layout="TND",
        recipe=RECIPE,
        source_boundary="kernel",
        coverage="source_selected_fwd_h_phase",
        flops=sum(total.matrix.values()),
        matrix_flops=tuple(sorted(total.matrix.items())),
        hbm_bytes=total.resources.get("hbm", 0),
        mte2_bytes=total.resources.get("mte2", 0),
        mte3_bytes=total.resources.get("mte3", 0),
        vector_ops=total.resources.get("vector", 0),
        metadata=tuple(sorted(meta.items())),
        notes=(
            "one varlen FWD_H launch for all prefill sequences",
            "C1 -> V1 -> C2 -> V2 serial chunks per (sequence, head)",
            "internal FP32 state [K,V]; caller [V,K] transpose is outside phase",
        ) + tuple(notes),
        stream=stream,
    )


__all__ = [
    "ChunkWork", "CoreTasks", "IMPLEMENTATION", "Params", "RECIPE", "StreamTask",
    "build", "chunk_work", "h_snapshot_offset", "head_ownership", "make_op",
    "slot_queues", "state_offset", "stream_work",
]
