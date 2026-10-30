"""KDA GATE_PREPARE stage on the shared mixed-core resource engine.

[Role]
  Bind the retained vLLM-Ascend ``chunk_kda_fwd`` stage 0 (GATE_PREPARE) to one
  source launch: BF16 raw-gate cumsum on the owning AIVs, the source global
  barrier, then paired-head AIC score/solve with per-subblock diagonal work.

[Invariants]
  * Source domain is varlen KDA with BF16 Q/K/V/raw-G, FP32 beta/gates/state,
    ``dk == dv == 128``, ``chunk == 64``, even head count, and lengths that are
    whole chunks.  A tail is rejected, never padded or silently dropped.
  * ``computeGateInPrepare`` is false: raw G is consumed by the native cumsum,
    and the external caller beta/sigmoid remains a separate dependency.
  * One paired task is ``chunk * (heads / 2) + pair``; tasks advance by the
    used AIC count.  Each AIV subblock owns one head of the pair.  Score is
    two BF16 32-row blocks (key width 32 then 64), not two full 64 squares.
  * The solve is six IEEE-FP32 GEMMs per head (four 16^3 and two 32^3), with
    GM->L1->L0->Cube->GM traffic and a completion edge.  Diagonal work keeps
    the 16-row dependency chain as a serial per-row phase, not instruction
    latency reconstruction.
  * Work is derived from the source intervals.  Programs are lazy: gate rows
    and paired tasks are finite slices, never an eager task graph.

[Timeouts & Locks]
  Pure metadata and arithmetic; no I/O, locks, device contexts or threads.

[Side-Effect Policy]
  Returns immutable plans; submits only to the caller's virtual engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
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
from .mixed_core import placement, require_pair, phase_stages

RECIPE = "vllm_ascend_kda_gate_prepare"
IMPLEMENTATION = "vllm_ascend_kda_chunk_prepare_state"

_K_DIM = 128
_V_DIM = 128
_CHUNK = 64
_BF16 = 2
_FP32 = 4
_SCORE_BF16_HEAD = 2 * 2 * 32 * (32 + 64) * 128      # 1,572,864
_SOLVE_FP32_HEAD = 4 * 2 * 16 ** 3 + 2 * 2 * 32 ** 3  # 163,840
_DIAG_BLOCKS = 4
_DIAG_PER_BLOCK = 16 * (119 + 119 + 14) + 16           # 4,048
_DIAG_VECTOR_HEAD = _DIAG_BLOCKS * _DIAG_PER_BLOCK     # 16,192
_MERGE_READ_HEAD = 24_576
_MERGE_WRITE_HEAD = 12_288
_SCORE_READ_HEAD = 57_344
_SCORE_FACTOR_READ_HEAD = 108_800
_SCORE_FACTOR_WRITE_HEAD = 147_456
# PrepareScoreFactorsBulk: 64 current rows plus 32 prior K-negative rows.
# These are semantic lane operations, not compiler instruction issue counts.
_FACTOR_VECTOR_HEAD = 64 * 128 * (14 + 5 + 8 + 14) + 32 * 128 * 7 + 64 + 4096
_FACTOR_SPECIAL_HEAD = 64 * 128 * 4 + 32 * 128
_DIRECT_SOLVE_VECTOR_HEAD = 64 * 64 * (3 + 1 + 2)
_FINAL_VECTOR_HEAD = 64 * 64
_SCORE_FIXPIPE_HEAD = 24_576
_SEED_WRITE_HEAD = 24_576
_FINAL_AKK_READ_HEAD = 16_384
_FINAL_AKK_WRITE_HEAD = 8_192
_GATE_ROW_ELEMENTS = 128
_SOURCE = "chunk_kda_fwd/arch35/GATE_PREPARE/safe_gate/bf16_raw_g/varlen"

_PARAM_KEYS = frozenset({
    "lengths", "heads", "key_heads", "dk", "dv", "chunk", "dtype",
    "gate_dtype", "raw_g_dtype", "beta_dtype", "compute_gate_in_prepare",
    "fuse_post_wu", "use_dense_fwd_h", "safe_gate",
    "compact",
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
    raise ModelingError(f"kda_prepare_parameter_must_be_boolean:{name}")


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
    compact: bool = True

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def chunks_per_sequence(self) -> tuple[int, ...]:
        return tuple(length // self.chunk for length in self.lengths)

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_per_sequence)

    @property
    def headchunks(self) -> int:
        return self.total_chunks * self.heads

    @property
    def paired_tasks(self) -> int:
        return self.total_chunks * (self.heads // 2)

    @property
    def gate_tasks(self) -> int:
        return len(self.lengths) * self.heads

    def metadata(self, cores: int = 32) -> dict[str, Any]:
        return {
            "lengths": self.lengths,
            "heads": self.heads,
            "key_heads": self.key_heads,
            "dk": self.dk,
            "dv": self.dv,
            "chunk": self.chunk,
            "compact": self.compact,
            "execution_abstraction": (
                "coupled_owner_runs_with_same_Engine_nominal" if self.compact
                else "expanded_finite_source_phase_control"),
            "dtype": "bf16",
            "raw_g_dtype": "bf16",
            "gate_dtype": "fp32",
            "beta_dtype": "fp32",
            "safe_gate": True,
            "compute_gate_in_prepare": False,
            "fuse_post_wu": False,
            "use_dense_fwd_h": False,
            "gate_mode": "native_bf16_raw_g_cumsum_fp32_gk",
            "beta_mode": "external_beta_sigmoid_preserved",
            "prepare_gate_products": "inside_native_gate_prepare_score",
            "source_phase": "GATE_PREPARE",
            "native_kind": "ChunkKdaFwd/GATE_PREPARE",
            "implementation": _SOURCE,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "chunks_per_sequence": self.chunks_per_sequence,
            "headchunks": self.headchunks,
            "paired_tasks": self.paired_tasks,
            "gate_tasks": self.gate_tasks,
            "gate_active_aiv": min(self.gate_tasks, 2 * cores),
            "paired_task_stride": cores,
            "paired_tasks_per_core": tuple(
                _ceil_div(max(0, self.paired_tasks - core), cores)
                for core in range(min(cores, self.paired_tasks))),
            "score_block_bf16_flops_per_head": (
                _SCORE_BF16_HEAD // 3, 2 * _SCORE_BF16_HEAD // 3),
            "solve_bf16_flops_absent": True,
            "solve_fp32_flops_per_head": (
                4 * 2 * 16 ** 3, 2 * 2 * 32 ** 3),
            "small_fp32_gemms_per_head": 6,
            "diag_vector_ops_per_head": _DIAG_VECTOR_HEAD,
            "diag_row_barriers_per_16block": 14,
            "diag_masked_lane_ops_per_16block": (119, 119, 14),
            "diag_scalar_get_set_per_16block": 16,
            "score_aic_gm_mte1_bytes_per_head": _SCORE_READ_HEAD,
            "score_factor_aiv_read_bytes_per_head": _SCORE_FACTOR_READ_HEAD,
            "score_factor_aiv_write_bytes_per_head": _SCORE_FACTOR_WRITE_HEAD,
            "score_direct_ub_fixpipe_bytes_per_head": _SCORE_FIXPIPE_HEAD,
            "solve_seed_gm_write_bytes_per_head": _SEED_WRITE_HEAD,
            "merge_read_bytes_per_head": _MERGE_READ_HEAD,
            "merge_write_bytes_per_head": _MERGE_WRITE_HEAD,
            "final_akk_fp32_read_bytes_per_head": _FINAL_AKK_READ_HEAD,
            "final_akk_bf16_write_bytes_per_head": _FINAL_AKK_WRITE_HEAD,
            "gate_row_read_bytes": _GATE_ROW_ELEMENTS * _BF16,
            "gate_row_write_bytes": _GATE_ROW_ELEMENTS * _FP32,
            "gate_row_vector_ops": 10 * _GATE_ROW_ELEMENTS,
            "gate_row_special_lanes": 2 * _GATE_ROW_ELEMENTS,
            "gate_aux_read_bytes_per_task": 532,
            "gate_chunk_reset_vector_ops": _GATE_ROW_ELEMENTS,
            "gate_barrier": "global_SyncAll_after_cumsum_before_prepare",
            "prepare_scratch": {
                "aqk_fp32_bytes": self.total_tokens * self.heads * self.chunk * _FP32,
                "akk_fp32_bytes": self.total_tokens * self.heads * self.chunk * _FP32,
                "solve_bytes": cores * 4 * 5 * self.chunk * self.chunk * _FP32,
                "score_bytes": cores * 4 * 3 * self.chunk * self.dk * _BF16,
                "post_wu_separate_bytes": (
                    self.total_tokens * self.heads * self.dk * _FP32),
            },
            "gm_assumption": "declared_GM_traffic_charged_HBM_no_cache_model; direct_UB_excluded",
            "store_qg": True,
            "factor_vector_lanes_per_head": _FACTOR_VECTOR_HEAD,
            "factor_special_lanes_per_head": _FACTOR_SPECIAL_HEAD,
            "timing_approximation": (
                "source-derived phase work; score blocks and head-pair tasks serialized; "
                "instruction issue, flag latency and scratch cache hits unmodeled"),
            "layout": "TND_varlen",
            "tail_policy": "reject_non_chunk_lengths",
        }


def parse_params(params: Mapping[str, Any]) -> Params:
    if not isinstance(params, Mapping):
        raise ModelingError("kda_prepare_params_must_be_a_mapping")
    unknown = set(params) - _PARAM_KEYS
    if unknown:
        raise ModelingError("kda_prepare_unknown_parameter:" + ",".join(sorted(unknown)))
    raw_lengths = params.get("lengths")
    if (isinstance(raw_lengths, (str, bytes))
            or not isinstance(raw_lengths, (list, tuple)) or not raw_lengths):
        raise ModelingError("kda_prepare_lengths_must_be_a_nonempty_integer_list")
    lengths = tuple(_positive_int(value, "kda_prepare_lengths_must_be_positive")
                    for value in raw_lengths)
    heads = _positive_int(params.get("heads"), "kda_prepare_heads_must_be_positive")
    key_heads = _positive_int(params.get("key_heads"),
                              "kda_prepare_key_heads_must_be_positive")
    if key_heads != heads:
        raise ModelingError("kda_prepare_requires_equal_key_and_value_heads")
    if heads % 2:
        raise ModelingError("kda_prepare_requires_even_head_count")
    dk = _positive_int(params.get("dk"), "kda_prepare_dk_must_be_positive")
    dv = _positive_int(params.get("dv"), "kda_prepare_dv_must_be_positive")
    chunk = _positive_int(params.get("chunk"), "kda_prepare_chunk_must_be_positive")
    if dk != _K_DIM:
        raise ModelingError(f"kda_prepare_requires_k_head_dim:{_K_DIM}")
    if dv != _V_DIM:
        raise ModelingError(f"kda_prepare_requires_v_head_dim:{_V_DIM}")
    if chunk != _CHUNK:
        raise ModelingError(f"kda_prepare_requires_chunk_size:{_CHUNK}")
    _dtype(params.get("dtype", "bf16"), {"bf16": "bf16", "bfloat16": "bf16"},
           "kda_prepare_requires_bf16_qkv")
    _dtype(params.get("raw_g_dtype", "bf16"),
           {"bf16": "bf16", "bfloat16": "bf16"},
           "kda_prepare_requires_bf16_raw_g")
    _dtype(params.get("gate_dtype", "fp32"), {"fp32": "fp32", "float32": "fp32"},
           "kda_prepare_requires_fp32_gates")
    _dtype(params.get("beta_dtype", "fp32"), {"fp32": "fp32", "float32": "fp32"},
           "kda_prepare_requires_fp32_beta")
    for name, required in (("compute_gate_in_prepare", False),
                           ("fuse_post_wu", False), ("use_dense_fwd_h", False),
                           ("safe_gate", True)):
        if _flag(params, name, required) != required:
            raise ModelingError(f"kda_prepare_requires_flag:{name}={required}")
    for length in lengths:
        if length % chunk:
            raise ModelingError("kda_prepare_rejects_partial_chunk_tail")
    return Params(lengths, heads, key_heads, dk, dv, chunk,
                  compact=_flag(params, "compact", True))


def _work(*, bf16: float = 0.0, fp32: float = 0.0, read: float = 0.0,
          write: float = 0.0, vector: float = 0.0, special: float = 0.0, mte1: float = 0.0,
          fixpipe: float = 0.0, read_pipe: str = "mte2") -> Work:
    resources: dict[str, float] = {"hbm": read + write + fixpipe}
    if read:
        resources[read_pipe] = read
    if write:
        resources["mte3"] = write
    if vector:
        resources["vector"] = vector
    if special:
        resources["reg_special_lanes"] = special
    if mte1:
        resources["mte1"] = mte1
    if fixpipe:
        resources["fixpipe"] = fixpipe
    matrix = {}
    if bf16:
        matrix["bf16"] = bf16
    if fp32:
        matrix["fp32"] = fp32
    return Work(matrix, {key: value for key, value in resources.items() if value})


def gate_row_work(*, first_row: bool, split_read: bool = False) -> Work:
    """One BF16 raw-gate row: double-buffered load plus serial accumulate."""
    read = _GATE_ROW_ELEMENTS * _BF16
    resources = {
        "hbm": read,
        "mte2_aiv" if split_read else "mte2": read,
    }
    return Work({}, resources)


def gate_accumulate_work(*, first_row: bool, split_read: bool = False) -> Work:
    """Safe-gate row: bias, expA, exp, div, lower-bound, cumsum and store."""
    write = _GATE_ROW_ELEMENTS * _FP32
    vector = 10 * _GATE_ROW_ELEMENTS + (_GATE_ROW_ELEMENTS if first_row else 0)
    resources = {
        "hbm": write, "mte3": write, "vector": vector,
        "reg_special_lanes": 2 * _GATE_ROW_ELEMENTS,
    }
    return Work({}, resources)


def gate_work(params: Params) -> Work:
    """Aggregate source gate work over every (sequence, head) task."""
    total = Work()
    for length in params.lengths:
        chunks = length // params.chunk
        rows = length * params.heads
        read = rows * _GATE_ROW_ELEMENTS * _BF16
        write = rows * _GATE_ROW_ELEMENTS * _FP32
        vector = rows * 10 * _GATE_ROW_ELEMENTS + chunks * params.heads * _GATE_ROW_ELEMENTS
        total += Work({}, {"hbm": read + write, "mte2": read,
                           "mte3": write, "vector": vector,
                           "reg_special_lanes": rows * 2 * _GATE_ROW_ELEMENTS})
    aux = params.gate_tasks * (4 + 128 * 4 + 2 * 8)
    resources = dict(total.resources)
    resources.update(hbm=resources["hbm"] + aux, mte2=resources["mte2"] + aux,
                     vector=resources["vector"] + 2 * params.gate_tasks,
                     reg_special_lanes=resources["reg_special_lanes"] + params.gate_tasks)
    return Work(total.matrix, resources)


def headchunk_work() -> Work:
    """One full head-chunk: direct-UB score/factor, seed, diag, six-GEMM merge."""
    aic_read = _SCORE_READ_HEAD + _MERGE_READ_HEAD
    aiv_read = _SCORE_FACTOR_READ_HEAD + _FINAL_AKK_READ_HEAD
    hbm = (aic_read + aiv_read + _SCORE_FACTOR_WRITE_HEAD + _SEED_WRITE_HEAD
           + _FINAL_AKK_WRITE_HEAD + _MERGE_WRITE_HEAD)
    return Work(
        {"bf16": _SCORE_BF16_HEAD, "fp32": _SOLVE_FP32_HEAD},
        {"hbm": hbm, "mte2_aic": aic_read, "mte2_aiv": aiv_read,
         "mte1": _SCORE_READ_HEAD + _MERGE_READ_HEAD,
         "mte3": (_SCORE_FACTOR_WRITE_HEAD + _SEED_WRITE_HEAD
                  + _FINAL_AKK_WRITE_HEAD),
         "fixpipe": _SCORE_FIXPIPE_HEAD + _MERGE_WRITE_HEAD,
         "vector": (_FACTOR_VECTOR_HEAD + _DIAG_VECTOR_HEAD
                    + _DIRECT_SOLVE_VECTOR_HEAD + _FINAL_VECTOR_HEAD),
         "reg_special_lanes": _FACTOR_SPECIAL_HEAD},
    )


def paired_work() -> Work:
    return headchunk_work().scaled(2)


def source_work(params: Params, cores: int = 32) -> Work:
    """Conserved aggregate for the whole stage, independent of core count."""
    prepare = paired_work().scaled(params.paired_tasks)
    gate = gate_work(params)
    total = gate + prepare
    resources = dict(total.resources)
    gate_read = resources.pop("mte2", 0)
    aic = resources.pop("mte2_aic", 0)
    aiv = resources.pop("mte2_aiv", 0) + gate_read
    resources.update(mte2_aic=aic, mte2_aiv=aiv)
    return Work(total.matrix, resources)


def task_ownership(params: Params, cores: int) -> tuple[tuple[int, int], ...]:
    """Finite paired-task queue: (core, per-core count) without expansion."""
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ModelingError("kda_prepare_requires_positive_compute_units")
    return tuple(
        (core, _ceil_div(max(0, params.paired_tasks - core), cores))
        for core in range(min(cores, params.paired_tasks)))


def sequence_chunk(chunk: int, params: Params) -> int:
    prefixes = [0]
    for count in params.chunks_per_sequence:
        prefixes.append(prefixes[-1] + count)
    return bisect_right(prefixes, chunk) - 1


# ---------------------------------------------------------------------------
# Programs.  Gate rows remain a two-slot per-owner pipeline; prepare tasks are
# a serial per-core phase pipeline.  Stage names own serial pipes; distinct names
# may overlap only where the source provides a completion edge.
# ---------------------------------------------------------------------------

def _aiv(context: RecipeContext, index: int):
    return placement(context, index // 2, index % 2)


def _gate_pipeline(aiv: int, params: Params, context: RecipeContext,
                   aiv_count: int) -> PipelineProgram:
    tasks = tuple(task for task in range(aiv, params.gate_tasks, aiv_count))
    ends, rows = [], 0
    entries = []
    for task in tasks:
        sequence, head = divmod(task, params.heads)
        length = params.lengths[sequence]
        entries.append((task, sequence, head, length, rows))
        rows += length
        ends.append(rows)
    owned = _aiv(context, aiv)
    split = read_role(context.resource_names, aiv % 2) == "mte2_aiv"

    def locate(index: int):
        position = bisect_right(ends, index)
        task, sequence, head, length, base = entries[position]
        return task, index - base

    def tile(index: int):
        task, local_row = locate(index)
        first = local_row % params.chunk == 0
        aux = 532 if local_row == 0 else 0
        load = gate_row_work(first_row=first, split_read=split)
        if aux:
            load = Work(load.matrix, {**load.resources,
                                      "hbm": load.resources["hbm"] + aux,
                                      ("mte2_aiv" if split else "mte2"):
                                          load.resources.get(
                                              "mte2_aiv" if split else "mte2", 0) + aux})
        compute = gate_accumulate_work(first_row=first)
        if local_row == 0:
            compute += Work({}, {"vector": 2, "reg_special_lanes": 1})
        return (
            Stage("load", context.bind(load), placement=owned),
            Stage("acc", context.bind(compute),
                  after=("load",), placement=owned),
        )

    return PipelineProgram(rows, 2, tile)


def _diag_program(lane: int, core: int, context: RecipeContext) -> KernelProgram:
    """Four serial 16-row blocks, each with 14 dependent row phases."""
    stages: list[Stage] = []
    for block in range(_DIAG_BLOCKS):
        previous = ()
        block_stages: list[Stage] = []
        for row in range(2, 16):
            vector = 2 * row * 16 + 16 + (16 if row == 15 else 0)
            name = f"row{row}"
            block_stages.append(Stage(
                name, context.bind(Work({}, {"vector": vector})),
                after=previous or (), placement=placement(context, core, lane)))
            previous = (name,)
        stages.append(Stage(f"block{block}", program=KernelProgram(tuple(block_stages)),
                           after=((f"block{block - 1}",) if block else ())))
    return KernelProgram(tuple(stages))


def _paired_tile(core: int, params: Params, context: RecipeContext):
    def tile(index: int):
        # A source task owns one head pair.  The phase abstraction serializes
        # pairs; it does not claim the source's deferred-solve overlap.
        stages: list[Stage] = []
        for lane in (0, 1):
            stages.extend(phase_stages(
                f"factor{lane}", _work(
                    read=_SCORE_FACTOR_READ_HEAD, write=_SCORE_FACTOR_WRITE_HEAD,
                    vector=_FACTOR_VECTOR_HEAD, special=_FACTOR_SPECIAL_HEAD),
                context, core, aiv=lane))
        score = Work({"bf16": 2 * _SCORE_BF16_HEAD}, {
            "hbm": 2 * _SCORE_READ_HEAD, "mte2": 2 * _SCORE_READ_HEAD,
            "mte1": 2 * _SCORE_READ_HEAD, "fixpipe": 2 * _SCORE_FIXPIPE_HEAD})
        stages.extend(phase_stages(
            "score", score, context, core, after=("factor0", "factor1")))
        for lane in (0, 1):
            owned = placement(context, core, lane)
            stages.append(Stage(
                f"mask{lane}", context.bind(Work({}, {"vector": 4 * 64 * 64})),
                after=("score",), placement=owned))
            stages.append(Stage(f"diag{lane}", program=_diag_program(lane, core, context),
                                after=(f"mask{lane}",), placement=owned))
            # X is produced by the diagonal solve.  It must be published before
            # the FP32 merge reads it, not before its producer executes.
            stages.extend(phase_stages(
                f"seed{lane}", _work(write=_SEED_WRITE_HEAD, vector=2 * 64 * 64),
                context, core, aiv=lane, after=(f"diag{lane}",)))
        previous = ("seed0", "seed1")
        for lane in (0, 1):
            for gemm, size in enumerate((16, 16, 16, 16, 32, 32)):
                name = f"merge{lane}_{gemm}"
                work = _work(fp32=2 * size ** 3, read=2 * size * size * 4,
                             mte1=2 * size * size * 4, fixpipe=size * size * 4)
                stages.extend(phase_stages(name, work, context, core, after=previous))
                previous = (name,)
        for lane in (0, 1):
            stages.extend(phase_stages(
                f"final{lane}", _work(read=_FINAL_AKK_READ_HEAD,
                                      write=_FINAL_AKK_WRITE_HEAD,
                                      vector=_FINAL_VECTOR_HEAD),
                context, core, aiv=lane, after=previous))
        stages.append(Stage("task_done", after=("final0", "final1")))
        return tuple(stages)

    count = _ceil_div(max(0, params.paired_tasks - core), context.core_count)
    return PipelineProgram(count, 1, tile)


def program(params: Params, context: RecipeContext, cores: int) -> KernelProgram:
    if context.resource_units.get("vector", cores) != 2 * cores:
        raise ModelingError("kda_prepare_requires_two_vector_subblocks_per_matrix_core")
    if int(context.core_count) != cores:
        raise ModelingError("kda_prepare_geometry_changed_after_build")
    if params.compact:
        from .kda_prepare_compact import program as compact_program
        return compact_program(params, context, cores)
    aiv_count = 2 * cores
    gate = tuple(_gate_pipeline(aiv, params, context, aiv_count)
                 for aiv in range(aiv_count)
                 if aiv < params.gate_tasks)
    prepare = tuple(_paired_tile(core, params, context)
                    for core, count in task_ownership(params, cores) if count)
    return KernelProgram((
        Stage("gate", program=ParallelProgram(gate)),
        Stage("syncall", after=("gate",)),
        Stage("prepare", after=("syncall",), program=ParallelProgram(prepare)),
    ))


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    metadata = dict(op.metadata)
    params = parse_params({key: metadata[key] for key in _PARAM_KEYS
                           if key in metadata})
    require_pair(geometry)
    cores = int(geometry.compute_units)
    if cores < 1:
        raise ModelingError("kda_prepare_requires_positive_compute_units")
    total = source_work(params, cores)
    if read_role(geometry.resource_units) == "mte2":
        resources = dict(total.resources)
        aic = resources.pop("mte2_aic", 0)
        aiv = resources.pop("mte2_aiv", 0)
        resources["mte2"] = aic + aiv
        total = Work(total.matrix, resources)

    def lower(context: RecipeContext):
        return program(params, context, cores)

    return OperatorPlan(total, lower)


def make_op(params: Mapping[str, Any], *, op_id: str = "kda_gate_prepare",
            deps=(), requests=(), stage: int = 0, layer: int = 0,
            phase: str = "prefill", stream: str = "",
            notes: tuple[str, ...] = ()) -> OpSpec:
    parsed = parse_params(params)
    meta = parsed.metadata()
    total = source_work(parsed, 32)
    semantic = (parsed.total_tokens, parsed.heads, parsed.dk, parsed.dv)
    return OpSpec(
        op_id=op_id, family="recurrent_local", phase=phase, stage=stage,
        layer=layer, requests=tuple(requests), deps=tuple(deps),
        semantic_shape=semantic, local_shape=semantic,
        padded_shape=(1, parsed.heads, parsed.total_chunks * parsed.chunk,
                      parsed.dv),
        launch_shape=semantic, dtype="bf16", layout="TND", recipe=RECIPE,
        source_boundary="kernel", coverage="source_selected_gate_prepare_stage",
        flops=sum(total.matrix.values()),
        matrix_flops=tuple(sorted(total.matrix.items())),
        hbm_bytes=total.resources.get("hbm", 0),
        mte2_bytes=(total.resources.get("mte2", 0)
                    + total.resources.get("mte2_aic", 0)
                    + total.resources.get("mte2_aiv", 0)),
        mte3_bytes=total.resources.get("mte3", 0),
        vector_ops=total.resources.get("vector", 0),
        metadata=tuple(sorted(meta.items())),
        notes=(
            "one varlen GATE_PREPARE launch for all prefill sequences",
            "BF16 raw-gate cumsum -> global barrier -> paired-head score/solve",
            "generic caller delta_gates prefill portion absorbed exactly once",
            "external beta sigmoid and generic Post-WU remain separate",
        ) + tuple(notes), stream=stream,
    )


def bind_gate_prepare(ops, *, gate, local_ops, lengths, heads, key_heads, dk, dv,
                      chunk, dtype, raw_g_dtype="bf16", gate_dtype="fp32",
                      beta_dtype="fp32", requests=(), pcp=1,
                      speculative=False):
    """Replace generic prefill gate and local chunk math with one source call."""
    if pcp != 1:
        raise ModelingError("kda_prepare_requires_pcp1")
    if speculative:
        raise ModelingError("kda_prepare_speculative_prefill_not_supported")
    local_ids = {op.op_id for op in local_ops}
    ids = {gate.op_id, *local_ids}
    params = {
        "lengths": tuple(lengths), "heads": heads, "key_heads": key_heads,
        "dk": dk, "dv": dv, "chunk": chunk, "dtype": dtype,
        "raw_g_dtype": raw_g_dtype, "gate_dtype": gate_dtype,
        "beta_dtype": beta_dtype,
    }
    replaced = {gate.op_id, *local_ids}
    deps = tuple(dict.fromkeys(
        dep for op in (gate, *local_ops) for dep in op.deps
        if dep not in replaced))
    source = make_op(params, op_id=gate.op_id, deps=deps, requests=requests,
                     stage=gate.stage, layer=gate.layer, phase=gate.phase,
                     stream=gate.stream, notes=gate.notes)
    rewritten = []
    for op in ops:
        if op.op_id in local_ids:
            continue
        if op.op_id == gate.op_id:
            rewritten.append(source)
            continue
        rewritten.append(op if not ids.intersection(op.deps) else replace(
            op, deps=tuple(source.op_id if dep in ids else dep
                           for dep in op.deps)))
    return tuple(rewritten), source


__all__ = [
    "IMPLEMENTATION", "Params", "RECIPE", "bind_gate_prepare", "build",
    "gate_accumulate_work", "gate_row_work", "gate_work", "headchunk_work",
    "make_op", "paired_work", "parse_params", "sequence_chunk", "source_work",
    "task_ownership",
]
