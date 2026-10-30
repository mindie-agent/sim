"""Chunked GDN prepare planner derived from the arch35 fused kernel.

[Role] Translate the source pack walk, stages 0-7 and their physical work into a
  finite per-core program on the shared engine.
[Invariants] Matrix work counts four dense FP32 64-cubed block products and the
  BF16 KKT/W/U products; q/k normalization and KKT are counted once per key
  head. Exact tail rows are retained for GM traffic while padded 64-row tiles
  are retained for on-chip L1 tiles. No measurement, model name or fitted
  multiplier participates.
[Timeouts & Locks] Build is O(number of sequences); execution is lazy over
  packs. No per-token list or event is materialized here.
[Side-Effect Policy] Pure metadata construction. The returned program submits
  only to the caller's engine.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    Stage,
)
from mindie_sim.schema.operators import OpSpec
from .mixed_core import phase_stages, phases, placement, require_pair, resident, split_reads
from .gdn_registers import CLASSES, cumsum, l2norm

__all__ = ["make_op", "build", "PackedChunks"]

# Source identity of the modeled implementation.  The planner is bound through
# the caller's RecipeBook; this string is not auto-discovered.
_IMPLEMENTATION = "gdn_prepare_arch35_mix_aic_aiv_1_2"
_CHUNK = 64
_HEAD_DIM_K = 128
_VALUE_DIMS = (128, 256)
_GQA_RATIOS = (1, 2, 3, 4)

# FP32 as stored by the kernel and BF16 as the q/k/v operand format.
_FP32 = 4
_BF16 = 2

# VCS and the remaining vector functions still use a lane-work envelope.
# Cumsum and L2Norm retain typed calls and independent capacity constraints.
_VCS_ROWS = 32 * 31 // 2
_VCS_OPS_PER_ROW = 64 + 64 + 4 + 2


def _int_param(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if float(value) != float(number) or number < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _bool_param(params: Mapping, name: str, default: bool) -> bool:
    value = params.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _normalize_dtype(value) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
    text = aliases.get(text, text)
    if text not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported_dtype:{value}")
    return text


def _dtype_width(dtype: str) -> int:
    return {"fp32": 4, "fp16": 2, "bf16": 2}[dtype]


def _work(*, matrix: Mapping[str, float] | None = None,
          resources: Mapping[str, float] | None = None) -> Work:
    matrix = {key: float(value) for key, value in (matrix or {}).items() if value}
    resources = {key: float(value) for key, value in (resources or {}).items() if value}
    return Work(matrix, resources)


def _scaled(work: Work, count: int) -> Work:
    return work.scaled(count) if count else Work()


def _gate_lower_ops(use_exp2: bool) -> int:
    # Sub, Max, Min, Exp, Mul(beta), negate, Select (+ optional ln2 multiply).
    return _CHUNK * _CHUNK * (8 if use_exp2 else 7)


def _vcs_ops() -> int:
    # Blocked 32x32 unit-triangular inverse.  Row iterIdx consumes iterIdx
    # clamped rows, each with a 64-lane multiply/reduce and scatter folds.
    return _VCS_ROWS * _VCS_OPS_PER_ROW


def _vcs_identity_ops() -> int:
    # 32 rows x 64 lanes x (two compares, two selects, two index adds, store).
    return 32 * 64 * 7


def _identity64_ops() -> int:
    # 64 rows x 64 lanes x (compare, select, store, index add).
    return 64 * 64 * 4


@dataclass(frozen=True)
class PackedChunks:
    """Varlen chunk table without materializing per-token work."""

    lengths: tuple[int, ...]
    chunk: int
    starts: tuple[int, ...]
    token_starts: tuple[int, ...]
    total: int
    padded_tokens: int

    @classmethod
    def from_lengths(cls, lengths: Iterable[int], chunk: int = _CHUNK) -> "PackedChunks":
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        lengths = tuple(int(value) for value in lengths)
        starts = [0]
        token_starts = [0]
        for length in lengths:
            starts.append(starts[-1] + (length + chunk - 1) // chunk)
            token_starts.append(token_starts[-1] + length)
        total = starts[-1]
        return cls(lengths, chunk, tuple(starts), tuple(token_starts), total,
                   total * chunk)

    def range(self, chunk_index: int) -> tuple[int, int, int, int]:
        """Return (sequence, local chunk, packed token start, valid rows).

        The token start matches the varlen source path in GetChunkRange:
        ``cu_seqlens[seq] + local_chunk * BT`` (common.h:565-574).  Dense BNSD
        would use just ``local_chunk * BT``; the campaign cases are varlen.
        """
        if chunk_index < 0 or chunk_index >= self.total:
            raise IndexError("chunk_index_out_of_range")
        sequence = bisect_right(self.starts, chunk_index) - 1
        local = chunk_index - self.starts[sequence]
        token_start = self.token_starts[sequence] + local * self.chunk
        rows = min(self.chunk, self.lengths[sequence] - local * self.chunk)
        return sequence, local, token_start, rows

    def all_ranges(self) -> tuple[tuple[int, int, int, int], ...]:
        """Small cases only; used by address oracles, never by the planner."""
        return tuple(self.range(index) for index in range(self.total))


@dataclass(frozen=True)
class _Config:
    lengths: tuple[int, ...]
    heads: int
    key_heads: int
    head_dim_k: int
    head_dim_v: int
    chunk: int
    gqa: int
    tasks_per_pack: int
    dtype: str
    gate_dtype: str
    gate_bytes: int
    use_qk_l2norm: bool
    use_gate: bool
    use_beta_sigmoid: bool
    allow_neg_eigval: bool
    use_exp2: bool
    output_a: bool
    has_dt_bias: bool
    has_a_log: bool
    output_beta: bool
    fp32_mmad_mode: str


def make_op(params: dict) -> OpSpec:
    """Explicit geometry adapter for the chunked GDN prepare kernel.

    Native caller flags select normalization, gate and export paths. FP32
    operands use the declared inherited MMAD control mode; the source does
    not itself enable HF32.
    """
    if not isinstance(params, Mapping):
        raise ValueError("gdn_prepare params must be a mapping")
    if not _bool_param(params, "compiler_ftz", True):
        raise ValueError("gdn_prepare_ftz_false_software_expansion_not_implemented")
    raw_lengths = params.get("lengths")
    if raw_lengths is None or isinstance(raw_lengths, (str, bytes)):
        raise ValueError("lengths must be a non-empty sequence")
    lengths = tuple(_int_param(value, "lengths[]") for value in raw_lengths)
    if not lengths:
        raise ValueError("lengths must be a non-empty sequence")

    heads = _int_param(params.get("heads", 0), "heads")
    key_heads = _int_param(params.get("key_heads", 0), "key_heads")
    if heads <= 0 or key_heads <= 0:
        raise ValueError("heads and key_heads must be positive")
    if heads % key_heads:
        raise ValueError("Hv must be divisible by Hk and Hv/Hk must be in {1,2,3,4}.")
    gqa = heads // key_heads
    if gqa not in _GQA_RATIOS:
        raise ValueError("Hv must be divisible by Hk and Hv/Hk must be in {1,2,3,4}.")

    chunk = _int_param(params.get("chunk", _CHUNK), "chunk")
    if chunk != _CHUNK:
        raise ValueError("chunk_size currently only supports 64.")
    head_dim_k = _int_param(params.get("dk", _HEAD_DIM_K), "dk")
    if head_dim_k != _HEAD_DIM_K:
        raise ValueError("K currently only supports 128.")
    head_dim_v = _int_param(params.get("dv", 128), "dv")
    if head_dim_v not in _VALUE_DIMS:
        raise ValueError("V must be 128 or 256.")

    dtype = _normalize_dtype(params.get("dtype", "bf16"))
    if dtype != "bf16":
        raise ValueError("q/k/v currently only support bfloat16.")
    gate_dtype = _normalize_dtype(params.get("gate_dtype", dtype))
    mm_mode = params.get("fp32_mmad_mode", "fp32")
    if mm_mode not in {"fp32", "hf32"}:
        raise ValueError("gdn_prepare_mmad_mode_must_be_fp32_or_hf32")

    use_qk = _bool_param(params, "use_qk_l2norm_in_kernel", True)
    use_gate = _bool_param(params, "use_gate_in_kernel", False)
    use_beta = _bool_param(params, "use_beta_sigmoid_in_kernel", True)
    allow_neg = _bool_param(params, "allow_neg_eigval", False)
    use_exp2 = _bool_param(params, "use_exp2", False)
    output_a = _bool_param(params, "output_a", True)
    has_dt_bias = _bool_param(params, "has_dt_bias", False)
    has_a_log = _bool_param(params, "a_log_present", use_gate)

    if use_gate and not has_a_log:
        raise ValueError("a_log is required when use_gate_in_kernel=True.")
    if not use_gate and (has_a_log or has_dt_bias):
        raise ValueError("a_log and dt_bias require use_gate_in_kernel=True.")
    if has_dt_bias and not use_gate:
        raise ValueError("dt_bias requires use_gate_in_kernel=True.")
    if allow_neg and not use_beta:
        raise ValueError("allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True.")

    chunks = PackedChunks.from_lengths(lengths, chunk)
    total_tokens = sum(lengths)
    tasks_per_pack = 3 if gqa == 3 else 4
    metadata = tuple(sorted({
        "implementation": _IMPLEMENTATION,
        "lengths": lengths,
        "heads": heads,
        "key_heads": key_heads,
        "head_dim_k": head_dim_k,
        "head_dim_v": head_dim_v,
        "chunk": chunk,
        "gqa_ratio": gqa,
        "tasks_per_pack": tasks_per_pack,
        "dtype": dtype,
        "gate_dtype": gate_dtype,
        "fp32_mmad_mode": mm_mode,
        "compiler_ftz": True,
        "register_service": "typed_capacity_envelope; masked_service_and_latency_uncharacterized",
        "use_qk_l2norm_in_kernel": use_qk,
        "use_gate_in_kernel": use_gate,
        "use_beta_sigmoid_in_kernel": use_beta,
        "allow_neg_eigval": allow_neg,
        "use_exp2": use_exp2,
        "output_a": output_a,
        "has_dt_bias": has_dt_bias,
        "has_a_log": has_a_log,
        "output_beta": _bool_param(params, "output_beta", True),
        "dispatch": {"implementation": _IMPLEMENTATION, "gate_dtype": gate_dtype,
                     "use_exp2": use_exp2, "use_qk_l2norm_in_kernel": use_qk,
                     "use_beta_sigmoid_in_kernel": use_beta,
                     "use_gate_in_kernel": use_gate, "output_a": output_a,
                     "fp32_mmad_mode": mm_mode, "compiler_ftz": True},
    }.items()))
    semantic = (1, heads, total_tokens, head_dim_k)
    padded = (1, heads, chunks.padded_tokens, head_dim_k)
    return OpSpec(
        op_id=f"gdn-prepare:h{heads}:hk{key_heads}:v{head_dim_v}:c{chunks.total}",
        family="gdn_prepare",
        phase="prefill",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=semantic,
        local_shape=semantic,
        padded_shape=padded,
        launch_shape=padded,
        dtype=dtype,
        recipe=_IMPLEMENTATION, source_boundary="kernel",
        metadata=metadata,
    )


def _config(op: OpSpec) -> _Config:
    meta = dict(op.metadata)
    lengths = tuple(int(value) for value in meta["lengths"])
    heads = int(meta["heads"])
    key_heads = int(meta["key_heads"])
    head_dim_k = int(meta["head_dim_k"])
    head_dim_v = int(meta["head_dim_v"])
    chunk = int(meta["chunk"])
    gqa = int(meta["gqa_ratio"])
    if not lengths or any(length < 0 for length in lengths):
        raise ValueError("lengths must be a non-empty sequence of non-negative rows")
    if heads <= 0 or key_heads <= 0 or heads % key_heads:
        raise ValueError("invalid GQA head geometry")
    if gqa != heads // key_heads or gqa not in _GQA_RATIOS:
        raise ValueError("gqa_ratio must equal heads // key_heads and be in {1,2,3,4}")
    if chunk != _CHUNK or head_dim_k != _HEAD_DIM_K:
        raise ValueError("selected source requires chunk=64 and head_dim_k=128")
    if head_dim_v not in _VALUE_DIMS:
        raise ValueError("head_dim_v must be 128 or 256")
    expected_pack = 3 if gqa == 3 else 4
    if int(meta["tasks_per_pack"]) != expected_pack:
        raise ValueError("tasks_per_pack is inconsistent with gqa_ratio")
    dtype = str(meta["dtype"])
    if dtype != "bf16":
        raise ValueError("selected source requires bf16 q/k/v")
    gate_dtype = str(meta.get("gate_dtype", dtype))
    return _Config(
        lengths=lengths,
        heads=heads,
        key_heads=key_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        chunk=chunk,
        gqa=gqa,
        tasks_per_pack=expected_pack,
        dtype=dtype,
        gate_dtype=gate_dtype,
        gate_bytes=_dtype_width(gate_dtype),
        use_qk_l2norm=bool(meta["use_qk_l2norm_in_kernel"]),
        use_gate=bool(meta["use_gate_in_kernel"]),
        use_beta_sigmoid=bool(meta["use_beta_sigmoid_in_kernel"]),
        allow_neg_eigval=bool(meta["allow_neg_eigval"]),
        use_exp2=bool(meta["use_exp2"]),
        output_a=bool(meta["output_a"]),
        has_dt_bias=bool(meta["has_dt_bias"]),
        has_a_log=bool(meta["has_a_log"]),
        output_beta=bool(meta.get("output_beta", True)),
        fp32_mmad_mode=meta.get("fp32_mmad_mode", "fp32"),
    )


def _pack_work_id(base: int, n_this: int, task_idx: int, gqa: int) -> int:
    """Source PackWorkId: swap the two middle tasks of a full G=2 pack."""
    if gqa == 2 and n_this == 4 and 0 < task_idx < 3:
        return base + 3 - task_idx
    return base + task_idx


def _stage0_work(cfg: _Config) -> Work:
    """Once per core: VCS identity/scratch, leaf zeros and the shared L1 I_64."""
    del cfg
    vector = 2 * _vcs_identity_ops() + 2 * _CHUNK * _CHUNK + _identity64_ops()
    # Four task slots x two leaves x 16 KiB, plus the 16 KiB I_64 upload.
    mte3 = 8 * (_CHUNK * _CHUNK * _FP32) + _CHUNK * _CHUNK * _FP32
    return _work(resources={"vector": vector, "mte3": mte3})


def _task_stage_works(cfg: _Config, rows: int, owns_hk: bool) -> dict[str, Work]:
    """Physical work for one (chunk, value-head) task, split by stage."""
    k = cfg.head_dim_k
    v = cfg.head_dim_v
    bt = cfg.chunk
    tail = rows < bt
    gate_bytes = cfg.gate_bytes
    bf16 = _BF16

    # ------------------------- Stage 1: gate/beta -------------------------
    gate_vec = (14 * rows + 2) if cfg.use_gate else (
        0 if cfg.gate_dtype == "fp32" else rows
    )
    beta_vec = (8 * rows) if cfg.use_beta_sigmoid else (
        0 if cfg.gate_dtype == "fp32" else rows
    )
    common_vector = gate_vec + beta_vec
    common_read = rows * gate_bytes + rows * gate_bytes
    if cfg.use_gate:
        common_read += gate_bytes * (1 + int(cfg.has_dt_bias))
    common_write = rows * _FP32
    if cfg.output_beta:
        common_write += rows * _FP32
    s1_common = _work(
        resources={
            "vector": common_vector,
            "hbm": common_read + common_write,
            "mte2": common_read,
            "mte3": common_write,
        }
    ) + cumsum(rows).capacity_work(CLASSES)

    # ----------------- Stage 1: key-head-owned normalization --------------
    pad = (2 * bt if tail else 0)
    s1k, s1k_export, s1q = Work(), Work(), Work()
    if owns_hk:
        if tail:
            pad += bt * k * (4 if cfg.use_qk_l2norm else 1)
        size = rows * k * bf16
        norm = l2norm(rows, cfg.dtype).capacity_work(CLASSES) if cfg.use_qk_l2norm else Work()
        s1k = norm + _work(resources={"hbm": size, "mte2": size, "mte3": bt * k * bf16})
        if cfg.use_qk_l2norm:
            export = size + rows * _FP32
            s1k_export = _work(resources={"hbm": export, "mte3": export})
            s1q = norm + _work(resources={
                "hbm": size + export,
                "mte2": size, "mte3": export,
            })
    s1pad = _work(resources={"vector": pad})
    s1 = s1pad + s1k + s1k_export + s1q + s1_common

    # ---------------------- Stage 2: owner KKT -----------------------------
    s2 = _work(matrix={"bf16": 2 * bt * bt * k} if owns_hk else {},
               resources={"mte1": 2 * bt * k * bf16,
                          "fixpipe": bt * bt * _FP32 * (2 if cfg.gqa > 2 else 1)} if owns_hk else {})

    # ------------------------ Stage 3: WY leaves --------------------------
    # Both UB->UB DataCopy calls use PIPE_V in arch3510, despite the
    # Prepare source's MTE3 comment. The payload is charged once per copy.
    s3parts = {
        "s3gate": _work(resources={"vector": _gate_lower_ops(cfg.use_exp2)}),
        "s3identity": _work(resources={"ub_copy": bt * (bt // 2) * _FP32}),
        "s3mul": _work(resources={"vector": bt * bt}),
        "s3pack": _work(resources={"ub_copy": bt * (bt // 2) * _FP32}),
        "s3neg": _work(resources={"mte3": bt * bt * _FP32}),
        "s3vcs": _work(resources={"vector": _vcs_ops()}),
        "s3leaf": _work(resources={"mte3": 2 * (bt // 2) ** 2 * _FP32}),
    }
    s3 = sum(s3parts.values(), Work())

    # --------------------- Stage 4: block products ------------------------
    s4i = _work(matrix={cfg.fp32_mmad_mode: 2 * bt * bt * bt})
    y_bytes = bt * bt * _FP32
    s4acc = _work(
        matrix={cfg.fp32_mmad_mode: 2 * bt * bt * bt},
        resources={"hbm": 2 * y_bytes, "mte2": y_bytes, "fixpipe": y_bytes,
                   "mte1": 2 * bt * bt * _FP32},
    )

    # ------------------ Stage 5: second-block products --------------------
    s5 = _work(matrix={cfg.fp32_mmad_mode: 2 * (2 * bt * bt * bt)})
    a_write = 0
    a_read = 0
    a_onchip = 0
    if cfg.output_a:
        a_write = rows * bt * bf16
        if rows == bt:
            a_read = bt * bt * bf16
        else:
            a_onchip = bt * bt * bf16
    else:
        a_onchip = bt * bt * bf16
    s5 = s5 + _work(resources={
        "hbm": a_write + a_read,
        "mte2": a_read,
        "fixpipe": a_write + a_onchip,
        "mte1": 4 * bt * bt * _FP32,
    })

    # --------------- Stage 6: beta-scaled V and K_hat ---------------------
    s6_vector = 0
    if tail:
        s6_vector += bt * v + bt * k
    # Stage6 calls ScaleRowsVF/ScaleRowsBetaExp2gVF with chunkSize, not the
    # valid-row count: padded zero rows still execute the row scaling.
    s6_vector += 3 * bt * v + (6 if cfg.use_exp2 else 5) * bt * k
    s6_read = rows * v * bf16 + rows * k * bf16
    s6 = _work(resources={
        "vector": s6_vector,
        "hbm": s6_read,
        "mte2": s6_read,
        "mte3": bt * v * bf16 + bt * k * bf16,
    })

    # ---------------------- Stage 7: W and U ------------------------------
    s7_write = rows * k * bf16 + rows * v * bf16
    s7 = _work(
        matrix={"bf16": 2 * bt * k * bt + 2 * bt * v * bt},
        resources={"hbm": s7_write,
                   "mte3": rows * (k + (v if v <= 128 else 0)) * bf16,
                   "fixpipe": s7_write,
                   "mte1": (1 + v // 128) * (bt * bt + bt * 128) * bf16},
    )
    return {
        "s1": s1, "s2": s2, "s3": s3, "s4i": s4i, "s4acc": s4acc,
        "s5": s5, "s6": s6, "s7": s7,
        "s1pad": s1pad, "s1k": s1k, "s1k_export": s1k_export,
        "s1q": s1q, "s1gate": s1_common, **s3parts,
    }


def _chunk_work(cfg: _Config, rows: int) -> Work:
    """All value-head tasks of one chunk at valid rows ``rows``."""
    owners = cfg.heads // cfg.gqa
    siblings = cfg.heads - owners
    owner = _task_stage_works(cfg, rows, True)
    sibling = _task_stage_works(cfg, rows, False)
    total = _scaled(owner["s1"], owners) + _scaled(sibling["s1"], siblings)
    total = total + _scaled(owner["s2"], owners)
    for name in ("s3", "s4i", "s4acc", "s5", "s6", "s7"):
        total = total + _scaled(owner[name], cfg.heads)
    return total


def _total_work(cfg: _Config, chunks: PackedChunks, cores: int) -> Work:
    total = _stage0_work(cfg).scaled(cores)
    for length in cfg.lengths:
        full, tail = divmod(length, cfg.chunk)
        if full:
            total = total + _scaled(_chunk_work(cfg, cfg.chunk), full)
        if tail:
            total = total + _chunk_work(cfg, tail)
    tasks = chunks.total * cfg.heads
    packs = (tasks + cfg.tasks_per_pack - 1) // cfg.tasks_per_pack
    total = total + Work(resources={"mte1": packs * 2 * cfg.chunk ** 2 * _FP32})
    if cfg.gqa == 2 and tasks % cfg.tasks_per_pack:
        total = total + Work(resources={"fixpipe": cfg.chunk ** 2 * _FP32})
    return total


def _tile(cfg, chunks, core, cores, pack_size, context):
    def retained(work):
        size = work["s7"].resources["mte3"]
        return Work(resources={"hbm": size, "mte3": size})

    def tasks_at(index):
        base = (core + index * cores) * pack_size
        n = max(0, min(pack_size, chunks.total * cfg.heads - base))
        tasks = []
        for task in range(n):
            work_id = _pack_work_id(base, n, task, cfg.gqa)
            hv, chunk_index = work_id % cfg.heads, work_id // cfg.heads
            rows = chunks.range(chunk_index)[3]
            work = _task_stage_works(cfg, rows, hv % cfg.gqa == 0)
            if cfg.gqa == 2 and n < 4 and hv % 2 == 0:
                work["s2"] = work["s2"] + Work(resources={"fixpipe": cfg.chunk ** 2 * _FP32})
            tasks.append((chunk_index, hv, rows, work))
        return tasks

    def tile(index):
        tasks = tasks_at(index)
        prev = tasks_at(index - 1) if index else ()
        stages, tails, owner_kkt = [], {}, {}

        def emit(name, work, after=(), aiv=None, previous=()):
            stages.append(replace(phases(name, work, context, core, aiv=aiv,
                                        after=tuple(dict.fromkeys(after))),
                                  previous=previous))
            return name

        if not tasks:
            # Source drains the last W/U pack only after its Stage7 handshake.
            for task, (_, _, _, work) in enumerate(prev):
                lane = task % 2
                tail = tails.get(lane)
                tails[lane] = emit(f"drain_last{task}", retained(work),
                                   (tail,) if tail else (), lane,
                                   ("cube_done", f"vector_done{lane}"))
            return tuple(stages)
        tail_pack = any(rows != cfg.chunk for _, _, rows, _ in (*tasks, *prev))
        for task, (chunk, hv, _, work) in enumerate(tasks):
            lane = task % 2
            tail = tails.get(lane)
            previous = (f"vector_done{lane}",)
            if tail_pack:
                previous += ("cube_done",)
            pad = emit(f"s1pad_{task}", work["s1pad"],
                       (tail,) if tail else (), lane, previous)
            ready = f"k_ready_{task}"
            stages.extend(phase_stages(
                ready, work["s1k"], context, core, aiv=lane, after=(pad,),
                store_previous=(f"y_dump_{task}",)))
            exported = emit(f"s1k_export_{task}", work["s1k_export"], (ready,), lane)
            q = f"s1q_{task}"
            stages.extend(phase_stages(q, work["s1q"], context, core, aiv=lane,
                                      after=(ready,), store_after=(exported,)))
            gate = f"s1gate_{task}"
            stages.extend(phase_stages(gate, work["s1gate"], context, core, aiv=lane,
                                      after=(q + ":compute",), store_after=(q,)))
            tails[lane] = f"s1_{task}"
            stages.append(Stage(tails[lane], after=(gate,)))
        last_cube = None
        for task, (chunk, hv, _, work) in enumerate(tasks):
            if hv % cfg.gqa == 0:
                deps = (f'k_ready_{task}',) + ((last_cube,) if last_cube else ())
                last_cube = emit(f's2_{task}', work['s2'], deps,
                                 previous=("cube_done",) if last_cube is None else ())
                owner_kkt[chunk, hv // cfg.gqa] = last_cube
        for task, (chunk, hv, _, work) in enumerate(tasks):
            lane = task % 2
            gate = emit(f"s3gate_{task}", work["s3gate"], (tails[lane],), lane)
            tails[lane] = emit(f"s3identity_{task}", work["s3identity"], (gate,), lane)
        for task, (chunk, hv, _, work) in enumerate(tasks):
            lane = task % 2
            mul = emit(f"s3mul_{task}", work["s3mul"],
                       (tails[lane], owner_kkt[chunk, hv // cfg.gqa]), lane)
            packed = emit(f"s3pack_{task}", work["s3pack"], (mul,), lane)
            neg = emit(f"s3neg_{task}", work["s3neg"], (packed,), lane)
            vcs = emit(f"s3vcs_{task}", work["s3vcs"], (packed,), lane)
            leaf = emit(f"s3leaf_{task}", work["s3leaf"], (neg, vcs), lane)
            tails[lane] = f"s3_{task}"
            stages.append(Stage(tails[lane], after=(leaf,)))
        # Each AIV finishes all S3 heads before S6; one AIC executes all
        # identity products, then all accumulation products, then S5 and S7.
        last_cube = emit("s4i_operands", Work(resources={"mte1": 2 * cfg.chunk ** 2 * _FP32}),
                         (last_cube,))
        for task, (_, _, _, work) in enumerate(tasks):
            last_cube = emit(f's4i_{task}', work['s4i'], (last_cube,))
        roundtrips = []
        for task, (_, _, _, work) in enumerate(tasks):
            value = work['s4acc']
            last_cube = emit(f's4mm_{task}', Work(value.matrix, {"mte1": value.resources["mte1"]}),
                             (last_cube, f's3_{task}'))
            write = value.resources['fixpipe']
            dump = emit(f'y_dump_{task}', Work(resources={'hbm': write, 'fixpipe': write}),
                        (last_cube,))
            read = value.resources['mte2']
            roundtrips.append(emit(f'y_reload_{task}',
                Work(resources={'hbm': read, 'mte2': read}), (dump,)))
        for task, (_, _, _, work) in enumerate(prev):
            lane = task % 2
            tails[lane] = emit(f"drain_prev{task}", retained(work),
                               (tails[lane],) if lane in tails else (), lane,
                               ("cube_done",))
        for task, (_, _, _, work) in enumerate(tasks):
            lane = task % 2
            tails[lane] = emit(f's6_{task}', work['s6'], (tails[lane],), lane)
        for task, (_, _, _, work) in enumerate(tasks):
            value = work['s5']
            # Independent operand banks let the second MTE1 load overlap the
            # first MMAD; both products accumulate into this task's L0C.
            one = Work({cfg.fp32_mmad_mode: value.matrix[cfg.fp32_mmad_mode] / 2},
                       {"mte1": value.resources["mte1"] / 2})
            body = phase_stages("left", one, context, core) + phase_stages(
                "right", one, context, core, l0_after=("left:l0",),
                compute_after=("left:compute",))
            name = f"s5mm_{task}"
            stages.append(Stage(name, after=(last_cube, roundtrips[task]), program=KernelProgram(body)))
            last_cube = name
            read = value.resources.get('mte2', 0)
            write = value.resources.get('fixpipe', 0)
            gm_write = value.resources.get('hbm', 0) - read
            last_cube = emit(f'a_dump_{task}',
                Work(resources={'hbm': gm_write, 'fixpipe': gm_write}), (last_cube,))
            if write > gm_write:
                last_cube = emit(f"a_local_{task}",
                                 Work(resources={"fixpipe": write - gm_write}), (last_cube,))
            last_cube = emit(f'a_reload_{task}',
                Work(resources={'hbm': read, 'mte2': read}), (last_cube,))
        for task, (_, _, rows, work) in enumerate(tasks):
            body = []
            for part in range(1 + cfg.head_dim_v // 128):
                size = rows * 128 * _BF16
                direct = cfg.head_dim_v > 128 and part > 0
                value = Work({"bf16": 2 * cfg.chunk * 128 * cfg.chunk}, resources={
                    "mte1": (cfg.chunk ** 2 + cfg.chunk * 128) * _BF16,
                    "fixpipe": size, "hbm": size if direct else 0})
                body.extend(phase_stages(
                    f"part{part}", value, context, core,
                    after=("part1",) if part == 2 else (),
                    l0_after=(f"part{part - 1}:l0",) if part else (),
                    compute_after=(f"part{part - 1}:compute",) if part else (),
                    publish_after=(f"part{part - 1}:publish",) if part else ()))
            name = f"s7_{task}"
            stages.append(Stage(name, after=(last_cube, f"s6_{task}"), program=KernelProgram(tuple(body))))
            last_cube = name
        stages.append(Stage("cube_done", after=(last_cube,)))
        stages.extend(Stage(f"vector_done{lane}", after=(tail,)) for lane, tail in tails.items())
        return tuple(stages)
    return tile


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Source task walk, physical ownership and ABI output work."""
    cfg = _config(op)
    require_pair(geometry)
    chunks = PackedChunks.from_lengths(cfg.lengths, cfg.chunk)
    cores = geometry.compute_units
    total_tasks = chunks.total * cfg.heads
    if total_tasks == 0:
        return OperatorPlan(Work(), lambda context: KernelProgram(()))
    pack_size = cfg.tasks_per_pack
    n_packs = (total_tasks + pack_size - 1) // pack_size
    plan_work = _total_work(cfg, chunks, cores)
    # Cube owns Y and full-A reloads; K/Q/gates/V are loaded by AIV.
    aic_reads = chunks.total * cfg.heads * cfg.chunk ** 2 * _FP32
    if cfg.output_a:
        aic_reads += sum(length // cfg.chunk for length in cfg.lengths) * cfg.heads * cfg.chunk ** 2 * _BF16
    plan_work = split_reads(plan_work, aic_reads, geometry.resource_units)

    def program(context):
        programs = []
        for core in range(cores):
            count = 0 if core >= n_packs else 1 + (n_packs - 1 - core) // cores
            pipeline = PipelineProgram(count + bool(count), 2,
                                       _tile(cfg, chunks, core, cores, pack_size, context))
            stages = []
            for aiv in range(2):
                vector = _vcs_identity_ops() + _CHUNK * _CHUNK
                copy = 4 * _CHUNK * _CHUNK * _FP32
                if aiv == 0:
                    vector += _identity64_ops()
                    copy += _CHUNK * _CHUNK * _FP32
                stages.append(phases(f's0_{aiv}', Work(resources={'vector': vector, 'mte3': copy}),
                                     context, core, aiv=aiv))
            stages.append(Stage('packs', after=('s0_0', 's0_1'), program=pipeline))
            # Highest addressed resident locations, including alignment holes.
            sizes = {'ub': 236 * 1024, 'l1': 496 * 1024,
                     'l0a': 64 * 1024, 'l0b': 64 * 1024, 'l0c': 256 * 1024}
            programs.append(resident(context, core, sizes, KernelProgram(tuple(stages))))
        return ParallelProgram(tuple(programs))
    return OperatorPlan(plan_work, program)
