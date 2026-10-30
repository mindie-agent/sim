"""Chunked GDN H-state update on contiguous per-compute-unit head ranges.

[Role]
  Derive the physical work of the arch35 ``chunk_fwd_h`` operator for the
  FP32-resident-state, scalar-gate dispatch and lower it to lazy per-core
  chunk pipelines.

[Invariants]
  * Contiguous ownership: each compute unit owns
    ``ceil(total_head_tasks / core_count)`` consecutive flattened
    ``(sequence, value_head)`` tasks. Idle units are not borrowed.
  * The recurrent state is FP32 and resident on the compute unit across chunks.
    The resident bank is initialized once from ``initial_state`` per head; the
    separate S-1 H0 cast reads the same tensor once more. The FP32 state is
    written once to ``final_state`` per head; no per-chunk state workspace
    round-trip exists.
  * Every chunk materializes a BF16 H snapshot in GM and reads it back for
    ``P = W @ H``. ``v_new`` is written to GM; the gated ``right`` operand is
    written to and read back from the GM workspace before ``D = K^T @ right``.
  * Work is interval arithmetic over heads and chunks; the chunk programs are
    lazy and never materialize per-token or per-scalar events.

[Timeouts & Locks]
  Pure metadata and arithmetic. No I/O, locks, device contexts or threads.

[Side-Effect Policy]
  Returns immutable plans and submits only to the caller's virtual engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .mixed_core import phase_stages, require_pair, resident

# ``chunk_fwd_h_policy.h`` constants. The kernel only accepts K=V=128 and a
# 64-token chunk; one AIC owns four head slots and pairs with two AIVs.
_K_DIM = 128
_V_DIM = 128
_CHUNK = 64
_AIC_HEAD_SLOTS = 4
_AIV_PER_AIC = 2
_BF16_BYTES = 2
_FP32_BYTES = 4
# FWD_H_UB_SHARE_BASE (226 KiB) and FWD_H_L1_USED_BYTES (320 KiB), per AIC.
_UB_BYTES = 226 * 1024
_L1_BYTES = 320 * 1024
_IMPLEMENTATION = "chunk_fwd_h/arch35/state_fp32/resident/scalar_g"

_PARAM_KEYS = frozenset({
    "lengths", "heads", "key_heads", "dk", "dv", "chunk", "dtype",
    "gate_dtype", "use_initial_state", "output_final_state", "use_exp2",
    "state_v_first", "save_new_value",
})


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align16(value: int) -> int:
    # FwdHAlignCube in chunk_fwd_h_utils.h: (value + 15) & ~15.
    return (value + 15) // 16 * 16


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
    raise ModelingError(f"gdn_state_parameter_must_be_boolean:{name}")


@dataclass(frozen=True)
class _Params:
    lengths: tuple[int, ...]
    heads: int
    key_heads: int
    dk: int
    dv: int
    chunk: int
    gate_dtype: str
    use_initial_state: bool
    output_final_state: bool
    use_exp2: bool
    state_v_first: bool

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def chunks_per_sequence(self) -> tuple[int, ...]:
        return tuple(_ceil_div(length, self.chunk) for length in self.lengths)

    @property
    def head_tasks(self) -> int:
        return len(self.lengths) * self.heads

    def metadata(self) -> dict[str, Any]:
        return {
            "lengths": self.lengths,
            "heads": self.heads,
            "key_heads": self.key_heads,
            "group_size": self.heads // self.key_heads,
            "dk": self.dk,
            "dv": self.dv,
            "chunk": self.chunk,
            "dtype": "bf16",
            "gate_dtype": self.gate_dtype,
            "use_initial_state": self.use_initial_state,
            "output_final_state": self.output_final_state,
            "use_exp2": self.use_exp2,
            "state_v_first": self.state_v_first,
            "save_new_value": True,
            "implementation": _IMPLEMENTATION,
            "gate_mode": "scalar_g",
            "state_precision": "fp32",
            "state_resident": True,
            "state_workspace_bytes_per_chunk": 0,
            "h_snapshot_dtype": "bf16",
            "right_workspace": "gm",
            "aic_head_slots": _AIC_HEAD_SLOTS,
            "aiv_per_aic": _AIV_PER_AIC,
            "total_tokens": self.total_tokens,
            "chunks_per_sequence": self.chunks_per_sequence,
            "head_tasks": self.head_tasks,
            "dispatch": {
                "implementation": _IMPLEMENTATION,
                "state_v_first": self.state_v_first,
                "use_exp2": self.use_exp2,
                "gate_dtype": self.gate_dtype,
                "use_initial_state": self.use_initial_state,
                "output_final_state": self.output_final_state,
            },
            "state_layout": "V,K" if self.state_v_first else "K,V",
            "h_semantic_strides": (1, self.dk) if self.state_v_first else (self.dv, 1),
            "update_operands": ("right_transposed", "kg") if self.state_v_first
                               else ("kg_transposed", "right"),
        }


def state_offset(sequence, head, key, value, heads, state_v_first=True):
    """FP32 state address in elements; dimensions follow FwdHDecodeHeadTask."""
    base = (sequence * heads + head) * _K_DIM * _V_DIM
    return base + (value * _K_DIM + key if state_v_first else key * _V_DIM + value)


def _parse_params(params: Mapping[str, Any]) -> _Params:
    if not isinstance(params, Mapping):
        raise ModelingError("gdn_state_params_must_be_a_mapping")
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise ModelingError("gdn_state_unknown_parameter:" + ",".join(sorted(unknown)))

    raw_lengths = params.get("lengths")
    if (isinstance(raw_lengths, (str, bytes))
            or not isinstance(raw_lengths, (list, tuple)) or not raw_lengths):
        raise ModelingError("gdn_state_lengths_must_be_a_nonempty_integer_list")
    lengths = tuple(_positive_int(value, "gdn_state_lengths_must_be_positive")
                    for value in raw_lengths)

    heads = _positive_int(params.get("heads"), "gdn_state_heads_must_be_positive")
    key_heads = _positive_int(params.get("key_heads"),
                              "gdn_state_key_heads_must_be_positive")
    if key_heads > heads or heads % key_heads:
        raise ModelingError("gdn_state_requires_value_heads_multiple_of_key_heads")
    dk = _positive_int(params.get("dk"), "gdn_state_dk_must_be_positive")
    dv = _positive_int(params.get("dv"), "gdn_state_dv_must_be_positive")
    chunk = _positive_int(params.get("chunk"), "gdn_state_chunk_must_be_positive")
    if dk != _K_DIM:
        raise ModelingError(f"gdn_state_requires_k_head_dim:{_K_DIM}")
    if dv != _V_DIM:
        raise ModelingError(f"gdn_state_requires_v_head_dim:{_V_DIM}")
    if chunk != _CHUNK:
        raise ModelingError(f"gdn_state_requires_chunk_size:{_CHUNK}")

    dtype = str(params.get("dtype", "bf16")).lower()
    if dtype not in {"bf16", "bfloat16"}:
        raise ModelingError("gdn_state_requires_bf16_inputs")
    gate_dtype = str(params.get("gate_dtype", "bf16")).lower()
    if gate_dtype in {"bf16", "bfloat16"}:
        gate_dtype = "bf16"
    elif gate_dtype in {"fp32", "float32"}:
        gate_dtype = "fp32"
    else:
        raise ModelingError("gdn_state_gate_dtype_must_be_bf16_or_fp32")

    use_initial_state = _flag(params, "use_initial_state", True)
    output_final_state = _flag(params, "output_final_state", True)
    use_exp2 = _flag(params, "use_exp2", False)
    state_v_first = _flag(params, "state_v_first", False)
    save_new_value = _flag(params, "save_new_value", True)
    if not save_new_value:
        raise ModelingError("gdn_state_requires_save_new_value")

    return _Params(lengths, heads, key_heads, dk, dv, chunk, gate_dtype,
                   use_initial_state, output_final_state, use_exp2, state_v_first)


def _params_from_metadata(metadata: Mapping[str, Any]) -> _Params:
    params = {key: metadata[key] for key in _PARAM_KEYS if key in metadata}
    missing = [key for key in ("lengths", "heads", "key_heads", "dk", "dv", "chunk")
               if key not in params]
    if missing:
        raise ModelingError("gdn_state_metadata_missing:" + ",".join(missing))
    return _parse_params(params)


def make_op(params: dict) -> OpSpec:
    """Create the semantic record for one chunk_fwd_h invocation."""
    parsed = _parse_params(params)
    packed_tokens = parsed.total_tokens
    chunk_tokens = sum(count * parsed.chunk
                       for count in parsed.chunks_per_sequence)
    semantic = (1, parsed.heads, packed_tokens, parsed.dv)
    return OpSpec(
        op_id="gdn_state",
        family="gdn_state",
        phase="prefill",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=semantic,
        local_shape=semantic,
        padded_shape=(1, parsed.heads, chunk_tokens, parsed.dv),
        launch_shape=semantic,
        dtype="bf16",
        layout="BNSD",
        recipe="chunk_fwd_h", source_boundary="kernel",
        metadata=tuple(sorted(parsed.metadata().items())),
        notes=(
            "variable-length packed BNSD call; chunk-level row padding is in plan work",
            "arch35 FP32 resident state; physical state layout is in dispatch",
        ),
    )


@dataclass(frozen=True)
class HeadUnit:
    """One serial head-round work unit inside a compute unit."""

    core: int
    sequence: int
    hv_begin: int
    heads: int
    length: int
    chunks: int


@dataclass(frozen=True)
class CoreOwnership:
    """Contiguous flattened head-task interval owned by one compute unit."""

    core: int
    begin: int
    end: int
    units: tuple[HeadUnit, ...]


def head_ownership(lengths, heads: int, core_count: int, *,
                   chunk: int = _CHUNK) -> tuple[CoreOwnership, ...]:
    """Resolve ``FwdHResolveCoreHeadRange`` and the kernel's serial unit loop.

    This is the exact contiguous interval policy: the flattened task index is
    ``sequence * heads + value_head`` and unit ``c`` owns
    ``[c * ceil(total / cores), min((c + 1) * ceil(total / cores), total))``.
    The second nested loop of ``Process()`` then splits an interval at sequence
    boundaries and at the four AIC head slots.
    """
    lengths = tuple(lengths)
    if not lengths or any(isinstance(value, bool) or not isinstance(value, int)
                          or value < 1 for value in lengths):
        raise ModelingError("gdn_state_lengths_must_be_positive")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ModelingError("gdn_state_heads_must_be_positive")
    if isinstance(core_count, bool) or not isinstance(core_count, int) or core_count < 1:
        raise ModelingError("gdn_state_requires_positive_compute_units")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ModelingError("gdn_state_chunk_must_be_positive")
    total = len(lengths) * heads
    heads_per_core = _ceil_div(total, core_count)
    groups = []
    for core in range(core_count):
        begin = core * heads_per_core
        if begin >= total:
            break
        end = min(begin + heads_per_core, total)
        cursor = begin
        units = []
        while cursor < end:
            sequence = cursor // heads
            hv_begin = cursor - sequence * heads
            take = min(end - cursor, heads - hv_begin, _AIC_HEAD_SLOTS)
            units.append(HeadUnit(
                core=core, sequence=sequence, hv_begin=hv_begin, heads=take,
                length=lengths[sequence], chunks=_ceil_div(lengths[sequence], chunk),
            ))
            cursor += take
        groups.append(CoreOwnership(core, begin, end, tuple(units)))
    return tuple(groups)


def _kh_count(unit: HeadUnit, params: _Params) -> int:
    group_size = params.heads // params.key_heads
    keys = {((unit.hv_begin + offset) // group_size) for offset in range(unit.heads)}
    return len(keys)


@dataclass(frozen=True)
class UnitWork:
    matrix_flops: float
    hbm_bytes: float
    mte2_bytes: float
    mte3_bytes: float
    vector_ops: float
    mte1_bytes: float
    fixpipe_bytes: float
    aic_mte2_bytes: float


def _gate_element_ops(params: _Params, tokens: int) -> int:
    # FwdHPrepareScalarGateArch35Vf: bf16 gate load cast (when bf16), gate
    # subtract, exp, optional use_exp2 multiply, and the alpha adds/exp.
    per_element = 2 + (1 if params.gate_dtype == "bf16" else 0) \
        + (1 if params.use_exp2 else 0)
    return tokens * per_element + per_element


def _unit_work(unit: HeadUnit, params: _Params) -> UnitWork:
    heads, length, chunks = unit.heads, unit.length, unit.chunks
    k, v, chunk = params.dk, params.dv, params.chunk
    h_state = k * v * _BF16_BYTES
    state_fp32 = k * v * _FP32_BYTES

    last_tokens = length - chunk * (chunks - 1)
    sum_m = chunk * (chunks - 1) + _align16(last_tokens)

    if params.use_initial_state:
        stage0_chunks, stage0_tokens, stage0_rows = chunks, length, sum_m
    elif chunks > 1:
        stage0_chunks = chunks - 1
        stage0_tokens = length - chunk
        stage0_rows = sum_m - chunk
    else:
        stage0_chunks, stage0_tokens, stage0_rows = 0, 0, 0

    if params.output_final_state:
        stage2_chunks, stage2_tokens, stage2_rows = chunks, length, sum_m
    elif chunks > 1:
        stage2_chunks = chunks - 1
        stage2_tokens = chunk * (chunks - 1)
        stage2_rows = chunk * (chunks - 1)
    else:
        stage2_chunks, stage2_tokens, stage2_rows = 0, 0, 0

    pre_read = state_fp32 * heads if params.use_initial_state else 0
    pre_write = h_state * heads if params.use_initial_state else 0
    pre_vector = k * v * heads if params.use_initial_state else 0

    s0_read = 2 * k * stage0_tokens * heads + h_state * stage0_chunks * heads
    s0_matrix = 2 * k * v * stage0_rows * heads

    gate_bytes = _BF16_BYTES if params.gate_dtype == "bf16" else _FP32_BYTES
    s1_read = 2 * v * length * heads + gate_bytes * stage2_tokens * heads
    s1_write = 2 * v * length * heads + 2 * v * stage2_tokens * heads
    s1_vector = heads * (3 * v * length + v * stage2_tokens)
    if not params.use_initial_state:
        s1_write += h_state * heads
        s1_vector += heads * v * min(chunk, length) + k * v * heads
    if stage2_chunks:
        per_element = 2 + (1 if params.gate_dtype == "bf16" else 0) \
            + (1 if params.use_exp2 else 0)
        s1_vector += heads * (stage2_tokens * per_element
                              + stage2_chunks * per_element)

    s2_read = (2 * k * stage2_tokens * _kh_count(unit, params)
               + 2 * v * stage2_tokens * heads)
    s2_matrix = 2 * k * v * stage2_rows * heads

    s3_read = state_fp32 * heads if params.use_initial_state and stage2_chunks else 0
    s3_write = h_state * (chunks - 1) * heads
    if params.output_final_state:
        s3_write += state_fp32 * heads
    s3_vector = heads * (2 * k * v * stage2_chunks + k * v * (chunks - 1))
    if not params.use_initial_state and stage2_chunks:
        # FwdHStage3Arch35Vf ZERO_STATE duplicates each state register pair
        # instead of loading the resident bank.
        s3_vector += heads * k * v

    mte2 = pre_read + s0_read + s1_read + s2_read + s3_read
    mte3 = pre_write + s1_write + s3_write
    matrix = s0_matrix + s2_matrix
    vector = pre_vector + s1_vector + s3_vector
    mte1 = 2 * heads * (k * stage0_rows + k * v * stage0_chunks
                        + (k + v) * stage2_rows)
    fixpipe = 4 * heads * (v * stage0_tokens + k * v * stage2_chunks)
    local_zero = 0
    if last_tokens < chunk:
        local_zero += heads * chunk * k * 2 if params.use_initial_state or chunks > 1 else 0
        if params.output_final_state:
            local_zero += (_kh_count(unit, params) * k + heads * v) * chunk * 2
    return UnitWork(matrix, mte2 + mte3, mte2 + local_zero, mte3, vector,
                    mte1, fixpipe, s0_read + s2_read + local_zero)


def _work(*, matrix: float = 0.0, hbm: float = 0.0, mte2: float = 0.0,
          mte3: float = 0.0, vector: float = 0.0, mte1: float = 0.0,
          fixpipe: float = 0.0) -> Work:
    resources = {role: amount for role, amount in (
        ("hbm", hbm), ("mte2", mte2), ("mte3", mte3), ("vector", vector),
        ("mte1", mte1), ("fixpipe", fixpipe),
    ) if amount}
    return Work({"bf16": matrix} if matrix else {}, resources)


def _chunk_work(unit: HeadUnit, params: _Params, index: int) -> dict[str, Work]:
    """Physical stage work for one chunk; sums to ``_unit_work`` over the unit."""
    heads = unit.heads
    k, v, chunk = params.dk, params.dv, params.chunk
    valid = min(chunk, unit.length - index * chunk)
    rows = _align16(valid)
    first = index == 0
    last = index == unit.chunks - 1
    stage0 = params.use_initial_state or not first
    stage2 = params.output_final_state or not last
    h_state = k * v * _BF16_BYTES
    state_fp32 = k * v * _FP32_BYTES

    if first and params.use_initial_state:
        pre = _work(
            hbm=heads * (state_fp32 + h_state),
            mte2=heads * state_fp32,
            mte3=heads * h_state,
            vector=heads * k * v,
        )
    else:
        pre = _work()

    if stage0:
        read = heads * (2 * k * valid + h_state)
        s0 = _work(matrix=heads * 2 * k * v * rows, hbm=read, mte2=read,
                   mte1=heads * 2 * (rows * k + k * v), fixpipe=heads * valid * v * 4)
    else:
        s0 = _work()

    gate_bytes = _BF16_BYTES if params.gate_dtype == "bf16" else _FP32_BYTES
    s1_read = heads * (2 * v * valid + (gate_bytes * valid if stage2 else 0))
    s1_write = heads * (2 * v * valid + (2 * v * valid if stage2 else 0))
    s1_vector = heads * ((4 if stage2 else 3) * v * valid)
    if not stage0:
        s1_vector += heads * v * valid
    if first and not params.use_initial_state:
        s1_write += heads * h_state
        s1_vector += heads * k * v
    if stage2:
        s1_vector += heads * _gate_element_ops(params, valid)
    s1 = _work(hbm=s1_read + s1_write, mte2=s1_read, mte3=s1_write,
               vector=s1_vector)

    if stage2:
        read = (_kh_count(unit, params) * 2 * k * valid
                + heads * 2 * v * valid)
        s2 = _work(matrix=heads * 2 * k * v * rows, hbm=read, mte2=read,
                   mte1=heads * 2 * rows * (k + v), fixpipe=heads * k * v * 4)
    else:
        s2 = _work()

    if stage2:
        s3_read = heads * state_fp32 if first and params.use_initial_state else 0
        if not last:
            s3_write = heads * h_state
        elif params.output_final_state:
            s3_write = heads * state_fp32
        else:
            s3_write = 0
        s3_vector = heads * (2 * k * v + (k * v if not last else 0))
        if first and not params.use_initial_state:
            s3_vector += heads * k * v
        s3 = _work(hbm=s3_read + s3_write, mte2=s3_read, mte3=s3_write,
                   vector=s3_vector)
    else:
        s3 = _work()
    return {"pre": pre, "s0": s0, "s1": s1, "s2": s2, "s3": s3}


def _unit_pipeline(unit: HeadUnit, params: _Params,
                   context: RecipeContext) -> PipelineProgram:
    def tile(index: int) -> tuple[Stage, ...]:
        values = [_chunk_work(replace(unit, hv_begin=unit.hv_begin + h, heads=1),
                              params, index) for h in range(unit.heads)]
        stages, cube_tail, vector_tail = [], None, {}
        valid = min(params.chunk, unit.length - index * params.chunk)
        def emit(name, value, deps=(), aiv=None, previous=(),
                 l0_after=(), compute_after=(), store_after=()):
            stages.extend(phase_stages(
                name, value, context, unit.core, after=deps, aiv=aiv,
                previous=previous, l0_after=l0_after, compute_after=compute_after,
                store_after=store_after))
            return name
        for head, work in enumerate(values):
            prior = f"pre{head - 2}" if head >= 2 else None
            emit(f"pre{head}", work["pre"], (prior + ":load",) if prior else (),
                 head % 2, compute_after=(prior + ":compute",) if prior else (),
                 store_after=(prior + ":store",) if prior else ())
        for aiv in range(min(2, unit.heads)):
            stages.append(Stage(f"pre_done{aiv}",
                                after=tuple(f"pre{h}" for h in range(aiv, unit.heads, 2))))
        inputs = {}
        for head, work in enumerate(values):
            aiv = head % 2
            read = work["s1"].resources.get("mte2", 0)
            prior_input = (f"input{head - 2}",) if head >= 2 else (f"pre_done{aiv}",)
            # Two input banks permit N+1 U/g to arrive while N waits for P/D.
            inputs[head] = emit(f"input{head}", _work(hbm=read, mte2=read),
                                prior_input, aiv,
                                () if unit.heads == 1 else (f"vector_done{aiv}",))
        for head, work in enumerate(values):
            aiv = head % 2
            source = work["s0"].resources.get("mte2", 0)
            w_read = valid * params.dk * 2 if source else 0
            # Single-head source uses alternating W/K banks; W for chunk N+1
            # is queued after right_N, before N's state-update matmul.
            prior = ("right0",) if unit.heads == 1 else ("cube_done",)
            zero = params.chunk * params.dk * 2 if valid < params.chunk and source else 0
            emit(f"w_zero{head}", _work(mte2=zero), previous=prior if head == 0 else ())
            emit(f"w{head}", _work(hbm=w_read, mte2=w_read), (f"w_zero{head}",))
            if unit.heads == 1:
                key_read = valid * params.dk * 2 if work["s2"].matrix else 0
                zero = params.chunk * params.dk * 2 if valid < params.chunk and key_read else 0
                emit("k_zero0", _work(mte2=zero), ("w0",))
                emit("k0", _work(hbm=key_read, mte2=key_read), ("k_zero0",))
            h_read = source - w_read
            load = emit(f"h{head}", _work(hbm=h_read, mte2=h_read),
                        (f"pre_done{aiv}", "k0") if unit.heads == 1 else (f"pre_done{aiv}",),
                        previous=(f"state{head}",))
            emit(f"p{head}", Work(work["s0"].matrix,
                                  {"mte1": work["s0"].resources.get("mte1", 0)}),
                 (f"w{head}", load),
                 previous=("cube_done",) if head == 0 else (),
                 l0_after=((f"p{head - 1}:l0",) if head else ()) +
                          ((f"p{head - 2}:compute",) if head >= 2 else ()),
                 compute_after=((f"p{head - 1}:compute",) if head else ()) +
                               ((f"p_publish{head - 2}",) if head >= 2 else ()))
            last_local_head = aiv + 2 * ((unit.heads - 1 - aiv) // 2)
            cube_tail = emit(f"p_publish{head}",
                             _work(fixpipe=work["s0"].resources.get("fixpipe", 0)),
                             (f"p{head}",) + ((cube_tail,) if cube_tail else ()) +
                             ((f"delta{head - 2}:compute",) if head >= 2 else ()),
                             previous=(f"state{last_local_head}:store",) if head < 2 else ())
            prior = vector_tail.get(aiv)
            value = work["s1"]
            write = value.resources.get("mte3", 0)
            gate_ops = _gate_element_ops(params, valid) if work["s2"].matrix else 0
            gate = emit(f"gate{head}", _work(vector=gate_ops),
                        (inputs[head],) + ((prior,) if prior else ()), aiv,
                        (f"vector_done{aiv}",))
            remaining = Work(value.matrix, {**value.resources, "hbm": write, "mte2": 0,
                                           "vector": value.resources.get("vector", 0) - gate_ops})
            vector_tail[aiv] = emit(f"delta{head}", remaining,
                                   (cube_tail, gate), aiv)
        last_p = cube_tail
        right_ready = dict(vector_tail)
        keys = {}
        for head, work in enumerate(values):
            aiv = head % 2
            key = (unit.hv_begin + head) // (params.heads // params.key_heads)
            key_read = valid * params.dk * 2 if work["s2"].matrix else 0
            if key not in keys:
                if unit.heads == 1:
                    keys[key] = "k0"
                else:
                    zero = params.chunk * params.dk * 2 if valid < params.chunk and key_read else 0
                    emit(f"k_zero{head}", _work(mte2=zero), (last_p,))
                    keys[key] = emit(f"k{head}", _work(hbm=key_read, mte2=key_read), (f"k_zero{head}",))
            right_read = work["s2"].resources.get("mte2", 0) - key_read
            zero = params.chunk * params.dv * 2 if valid < params.chunk and right_read else 0
            emit(f"right_zero{head}", _work(mte2=zero), (right_ready[aiv], last_p))
            right = emit(f"right{head}", _work(hbm=right_read, mte2=right_read),
                         (f"right_zero{head}",))
            emit(f"d{head}", Work(work["s2"].matrix,
                                  {"mte1": work["s2"].resources.get("mte1", 0)}),
                 (right, keys[key]),
                 l0_after=((f"d{head - 1}:l0",) if head else (last_p,)) +
                          ((f"d{head - 2}:compute",) if head >= 2 else ()),
                 compute_after=((f"d{head - 1}:compute",) if head else (last_p,)) +
                               ((f"d_publish{head - 2}",) if head >= 2 else ()))
            release = vector_tail[aiv]
            if head >= 2:
                release = f"state{head - 2}:" + ("compute" if index == unit.chunks - 1 else "store")
            cube_tail = emit(f"d_publish{head}",
                             _work(fixpipe=work["s2"].resources.get("fixpipe", 0)),
                             (f"d{head}", cube_tail, release))
            vector_tail[aiv] = emit(f"state{head}", work["s3"],
                                   (right_ready[aiv],), aiv,
                                   compute_after=(cube_tail,) +
                                   ((f"state{head - 2}:compute",) if head >= 2 else ()),
                                   store_after=(f"state{head - 2}:store",) if head >= 2 else ())
        stages.append(Stage("cube_done", after=(cube_tail,)))
        stages.extend(Stage(f"vector_done{aiv}", after=(tail,))
                      for aiv, tail in vector_tail.items())
        return tuple(stages)

    return PipelineProgram(unit.chunks, 2, tile)


def _core_program(group: CoreOwnership, params: _Params,
                  context: RecipeContext):
    programs = tuple(_unit_pipeline(unit, params, context) for unit in group.units)
    if len(programs) == 1:
        return programs[0]
    stages = []
    previous = None
    for index, program in enumerate(programs):
        name = f"unit{index}"
        stages.append(Stage(name, after=(() if previous is None else (previous,)),
                            program=program))
        previous = name
    return KernelProgram(tuple(stages))


def _program(params: _Params, groups: tuple[CoreOwnership, ...],
             context: RecipeContext):
    if int(context.core_count) < 1:
        raise ModelingError("gdn_state_requires_positive_compute_units")
    names = context.resource_names
    for role in ("cube", "hbm", "vector"):
        if role not in names:
            raise ModelingError(f"physical_resource_rate_missing:{role}")
    return ParallelProgram(tuple(resident(
        context, group.core, {"ub": _UB_BYTES, "l1": _L1_BYTES,
                             "l0a": 32 * 1024, "l0b": 64 * 1024, "l0c": 128 * 1024},
        _core_program(group, params, context)) for group in groups))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower chunk_fwd_h physical work onto one shared resource engine."""
    from .mixed_core import split_reads
    params = _params_from_metadata(dict(op.metadata))
    require_pair(geometry)
    core_count = int(geometry.compute_units)
    if core_count < 1:
        raise ModelingError("gdn_state_requires_positive_compute_units")
    groups = head_ownership(params.lengths, params.heads, core_count,
                            chunk=params.chunk)

    def units():
        for group in groups:
            yield from group.units

    matrix = math.fsum(_unit_work(unit, params).matrix_flops for unit in units())
    mte2 = math.fsum(_unit_work(unit, params).mte2_bytes for unit in units())
    mte3 = math.fsum(_unit_work(unit, params).mte3_bytes for unit in units())
    vector = math.fsum(_unit_work(unit, params).vector_ops for unit in units())
    hbm = math.fsum(_unit_work(unit, params).hbm_bytes for unit in units())
    mte1 = math.fsum(_unit_work(unit, params).mte1_bytes for unit in units())
    fixpipe = math.fsum(_unit_work(unit, params).fixpipe_bytes for unit in units())
    total = _work(matrix=matrix, hbm=hbm, mte2=mte2, mte3=mte3,
                  vector=vector, mte1=mte1, fixpipe=fixpipe)
    total = split_reads(total, math.fsum(_unit_work(unit, params).aic_mte2_bytes
                                       for unit in units()), geometry.resource_units)

    def program(context: RecipeContext):
        if int(context.core_count) != core_count:
            raise ModelingError("gdn_state_geometry_changed_after_build")
        return _program(params, groups, context)

    return OperatorPlan(total, program)


__all__ = ["CoreOwnership", "HeadUnit", "build", "head_ownership", "make_op"]
