"""KDA prefill FWD_H state phase on the mixed-core resource engine.

[Role] Bind the selected FWD_H kernel to a finite per-stream chunk pipeline.
[Invariants] The source domain uses BF16 K/V/W/U, FP32 gates and state,
  and 64-token chunks with dk == dv == 128. Each (sequence, head) advances
  serially through C1, V1, C2 and V2 using actual chunk tails. The caller
  owns the public [V, K] transpose. Each owner retains at most two source
  streams and vector phases draw service from its assigned AIV pair.
[Timeouts & Locks] Lazy stream and chunk programs. No I/O, locks or threads.
[Side-Effect Policy] Plans submit only to the caller's virtual engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, check_live_buffers
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import (
    OpSpec, ProgramFragment, SemanticAttr, StateEffect,
)
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from .kda_state_contracts import (
    _CHUNK, _K_DIM, _V_DIM, _binding_attr, _binding_ports, _effect, _fragment,
    _head_view, _per_sequence, _prefill_publication_fragments, _public_value,
    _sequence_values, _state_effect, layout_fragment, prefill_state_contract,
    state_accesses, state_fragments,
)
from .kda_state_selection import (
    bind_state_value, binding_record, prefill_records, state_ports, state_value,
)
from mindie_sim.prediction.operators.vector.mixed_core import (
    placement,
    require_pair,
    resident,
    split_reads,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_FWD_H_REPO = "vllm-ascend-v41-private"
_VLLM_ASCEND_FWD_H_REV = "ae923905948f360f4b7ab06df52d0789a0c9316e"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h",
        "e98c17e528d673a9b6c2f0bbbfce83746bcccf0afaeb413ea0ac41a2809e74f8",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp",
        "0ea787730736f4ed92a35e3bafa6e54417c84716fc6aed381a339259abfa74d0",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/attention/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp",
        "72148ca10b26ba4f34919c11e973c3bbcd696cfbc9a938c3cc3bf724e41f813c",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp",
        "aa9695e2d9eba0dc0963d248199a86ab5cbd321c0f5836a9dd8ee8901944cf13",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp",
        "3e7730a090795623af026efea6a8484238d1fd02e301e286936f1b4bcd8ba741",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp",
        "8ec011c4ff4adaa0398ba77e2b5cbc331335c3f6c44d00921e9cdbcaa953dd02",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp",
        "67dbb475b2ae630bb53fbdb9bfe44af774c0be8b9da7c16074b5d1ef6d88fedb",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp",
        "c2221dbb88f92fcf04adde0a0799264bd53c6e1f74033ed3bee634fb9daac4cc",
    ),
    SourceRef(
        _VLLM_ASCEND_FWD_H_REPO,
        _VLLM_ASCEND_FWD_H_REV,
        "vllm_ascend/ops/kimi_kda.py",
        "35365b3b353fc492b4dc5d75b88916431f6802c21bf51b61b4893d39d54f11f3",
    ),
)

TAIL_L1_ZERO_FILL_BUFFERS = 4
VALUE_DELTA_OPS_PER_ELEMENT = 4
INIT_STATE_UB_KIB = 64
INIT_SNAPSHOT_UB_KIB = 32
SCAN_UB_PONG_G_SUB_OFFSET_KIB = 171
SCAN_UB_G_SUB_TILE_ROWS = 16
_BYTES_PER_KIB = 1024
_FP32_BYTES_UB = 4
INIT_UB_BYTES_PER_AIV = 2 * (INIT_STATE_UB_KIB + INIT_SNAPSHOT_UB_KIB) * _BYTES_PER_KIB
SCAN_UB_BYTES_PER_AIV = (
    SCAN_UB_PONG_G_SUB_OFFSET_KIB * _BYTES_PER_KIB
    + SCAN_UB_G_SUB_TILE_ROWS * _FP32_BYTES_UB
)

RECIPE = "vllm_ascend_kda_fwd_h"
IMPLEMENTATION = "vllm_ascend_kda_chunk_state"

# Arch35 FWD_H storage and stream constants.
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
                * self.heads * TAIL_L1_ZERO_FILL_BUFFERS * _K_DIM * _V_DIM * _BF16_BYTES),
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
        vector=VALUE_DELTA_OPS_PER_ELEMENT * valid * v,
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
                  context: RecipeContext, fragment_map=None) -> PipelineProgram:
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
            fragment = (None if fragment_map is None
                        else fragment_map.get((task.sequence, task.head, chunk)))
            chunks.append((task.slot, fragment, work))
        stages = []
        # The source AIC issues C1 for both slots before C2 for both slots;
        # the AIV pair does the analogous V1/V2 ordering. Different processors
        # may overlap, but a slot cannot overwrite its prior state or scratch.
        for phase, aiv in (("c1", False), ("delta", True),
                           ("c2", False), ("state", True)):
            phase_tail = None
            for slot, fragment, work in chunks:
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
                phase_stages = _phases(
                    name, getattr(work, phase), context, group.core,
                    aiv_pair=aiv, after=after, previous=previous)
                if phase == "state" and fragment is not None:
                    phase_stages = (*phase_stages[:-1],
                                    replace(phase_stages[-1], fragment=fragment))
                stages.extend(phase_stages)
                phase_tail = name
        stages.append(Stage("cube_done", after=(f"c2s{chunks[-1][0]}",)))
        stages.append(Stage("vector_done", after=(f"states{chunks[-1][0]}",)))
        return tuple(stages)

    return PipelineProgram(max(queue.ends[-1] for queue in queues), 2, tile)


def _program(params: Params, groups: tuple[CoreTasks, ...],
             context: RecipeContext, core_count: int,
             fragment_map=None, read_fragment=None):
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
        init_body = KernelProgram(_phases(
            "init", work, context, group.core, aiv_pair=True))
        init_programs.append(
            resident(context, group.core, {"ub": INIT_UB_BYTES_PER_AIV}, init_body)
            if "ub" in context.pool_names else init_body
        )
    initialize = Stage("initialize", program=ParallelProgram(tuple(init_programs)))
    if read_fragment is not None:
        initialize = replace(initialize, fragment=read_fragment)
    scan_programs = tuple(
        resident(
            context, group.core, {"ub": SCAN_UB_BYTES_PER_AIV},
            _core_program(group, params, context, fragment_map),
        ) if "ub" in context.pool_names
        else _core_program(group, params, context, fragment_map)
        for group in groups
    )
    return KernelProgram((
        initialize,
        Stage("scan", after=("initialize",), program=ParallelProgram(scan_programs)),
    ))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower the selected FWD_H phase onto one shared resource engine."""
    params = _params_from_metadata(dict(op.metadata))
    require_pair(geometry)
    core_count = int(geometry.compute_units)
    if core_count < 1:
        raise ModelingError("kda_state_requires_positive_compute_units")
    if "ub" in geometry.buffers:
        check_live_buffers(
            (
                LiveBufferSpec(
                    "kda_fwd_h_init_ub", "ub", INIT_UB_BYTES_PER_AIV,
                    live_start=0, live_end=0, owner_kind="aiv",
                ),
                LiveBufferSpec(
                    "kda_fwd_h_scan_ub", "ub", SCAN_UB_BYTES_PER_AIV,
                    live_start=1, live_end=1, owner_kind="aiv",
                ),
            ),
            geometry,
            family=RECIPE,
        )
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
    contract = dict(op.metadata).get("kda_state_contract")
    layout_bytes = 0.0
    if contract is not None:
        layout_bytes = float(contract["layout_bytes"])
    total = split_reads(total, aic_mte2, geometry.resource_units)

    fragments = state_fragments(op)

    def program(context: RecipeContext):
        return _program(params, groups, context, core_count,
                        fragment_map=fragments["internal"],
                        read_fragment=fragments["initialize"])

    metadata = ({} if contract is None else {
        "state_contract": contract,
        "state_effects": dict(op.metadata).get("kda_state_effects", ()),
        "state_fragment_counts": {
            "initialize": 1 if fragments["initialize"] is not None else 0,
            "internal": len(fragments["internal"]),
            "publication": len(fragments["publication"]),
        },
        "state_accesses": state_accesses(op),
        "layout_bytes": layout_bytes,
        "publication_producer": contract["layout_producer"],
        "state_bindings": tuple(
            binding_record(binding)
            for binding in contract["publication_bindings"]),
    })
    return OperatorPlan(total, program, metadata=metadata)


def make_op(params: Mapping[str, Any], *, op_id: str = "kda_fwd_h",
            deps=(), requests=(), stage: int = 0, layer: int = 0,
            phase: str = "prefill", stream: str = "",
            notes: tuple[str, ...] = (), state_contexts=None,
            state_targets=None, state_values=None, layout_values=None,
            state_ready=(), state_grants=()) -> OpSpec:
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
    if state_targets is None:
        request_ids = tuple(requests)
        layer_target = f"kda.layer[{layer}]" if layer >= 0 else "kda.call"
        state_targets = tuple(
            f"{layer_target}.recurrent_state["
            f"{request_ids[index] if index < len(request_ids) else index}]"
            for index in range(len(parsed.lengths)))
    contract, effects = prefill_state_contract(
        lengths=parsed.lengths, requests=requests, chunk=parsed.chunk,
        contexts=state_contexts, targets=state_targets, heads=parsed.heads,
        dk=parsed.dk, dv=parsed.dv, state_dtype="fp32", op_id=op_id,
        state_values=state_values, layout_values=layout_values,
        state_ready=state_ready, state_grants=state_grants,
        owner=(f"kda.layer[{layer}]" if layer >= 0 else "kda.call"))
    meta.update(
        kda_state_contract=contract, kda_state_effects=effects,
        state_targets=tuple(state_targets),
        state_checkpoint_versions=contract["checkpoint_versions"],
        state_read_versions=contract["read_versions"],
        state_commit_versions=contract["commit_versions"])
    op = OpSpec(
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
    return replace(op, metadata=op.metadata + (
        ("state_accesses", state_accesses(op)),))


__all__ = [
    "ChunkWork", "CoreTasks", "IMPLEMENTATION", "INIT_UB_BYTES_PER_AIV",
    "Params", "RECIPE", "SCAN_UB_BYTES_PER_AIV", "SOURCES", "StreamTask",
    "TARGET_BACKEND", "build", "chunk_work", "h_snapshot_offset",
    "head_ownership", "make_op", "slot_queues", "state_accesses",
    "state_fragments", "state_offset", "stream_work",
]
