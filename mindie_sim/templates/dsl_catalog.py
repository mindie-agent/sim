"""Reference TileLang/AscendC tile pipelines and host dispatch for Tier K3.

[Role] Describe concrete open-source DSL kernel ABIs, capacity and scheduler decisions.
[Invariants] Workspace extents and source-pinned constants have explicit units.
[Timeouts & Locks] Local finite schedule arithmetic. No I/O or locks.
[Side-Effect Policy] Pure CPU modeling of launcher and kernel structures.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileCandidate, TileInfeasibleError, TileShape
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import dtype_name, unmodeled
from .contracts import (
    ATTN_RES_UB_BYTES, BF16_BYTES, DIRECT_MATRIX_ARCH_IDS, FP32_BYTES, KDA_HEAD_DIM, KDA_STATE_HALF,
    DslKernelSpec, TemplateSourceRef, attn_res_tiling, attn_res_ub_footprint, ceil_div, commit_replay_bytes, commit_rows, flash_kda_rounds,
)
from .fuse_region import solve_template_tile

_DSL_SOURCE_REFS_PURGED_MSG = (
    "LEGACY_PURGED: rebuild independently verified TileLang/AscendC SourceRef table"
)


class _PurgedDslSourceRefs:
    def __iter__(self):
        raise NotImplementedError(_DSL_SOURCE_REFS_PURGED_MSG)

    def __len__(self) -> int:
        raise NotImplementedError(_DSL_SOURCE_REFS_PURGED_MSG)

    def __getitem__(self, item):
        del item
        raise NotImplementedError(_DSL_SOURCE_REFS_PURGED_MSG)

    def __bool__(self) -> bool:
        raise NotImplementedError(_DSL_SOURCE_REFS_PURGED_MSG)


DSL_SOURCE_REFS = _PurgedDslSourceRefs()
FLASH_KDA = "dsl.flash_kda"
MEGA_KDA = "dsl.mega_recurrent_kda"
COMMIT_KDA = "dsl.commit_recurrent_kda"
MLA_PROLOG = "dsl.mla_prolog_v4"
ATTN_RES = "dsl.block_attn_res_update_rms_norm"
QSLI = "dsl.quant_sparse_lightning_indexer"
MQSMLA = "dsl.mixed_quant_sparse_flash_mla"
DSL_KERNEL_IDS = (FLASH_KDA, MEGA_KDA, COMMIT_KDA, MLA_PROLOG, ATTN_RES, QSLI, MQSMLA)
INT32_BYTES = 4
KDA_CHUNK = 64
KDA_INVERSE_BUFFERS = 4
KDA_AIC_CORE_LIMIT = 32
KDA_DEFAULT_WORKSPACE_SLOTS = 861
KDA_CHUNK_TASK_WORDS = 6
KDA_HEADER_WORDS = 11
KDA_CALIBRATED_CHUNK_RANGE = range(64, 257)
KDA_CALIBRATED_GROUP_LIMIT = 16
KDA_MAX_VALUE_HEADS = 96
KDA_MAX_CHUNK_GROUP = 512
KDA_SHORT_TRAIN_GROUP = 13
KDA_SHORT_TRAIN_MIN_LENGTH = 8192
KDA_BETA_TILE = 16
MEGA_TP_DEGREES = (4, 8, 16)
MEGA_SEQUENCE_LENGTHS = (4, 8)
KDA_REPLAY_CAPACITY = max(MEGA_SEQUENCE_LENGTHS)
MEGA_MODEL_BATCH_LIMIT = 32
MEGA_AIC_QUOTAS = (8, 16, 24, 28, 32)
COMMIT_CORE_QUOTAS = (8, 24, 28, 32)
ATTN_RES_VL = 64
ROUTE_COST_SCALE = 64
PARTS_PER_MILLION = 1_000_000
_ROUTE_PARAMS_STRIDE = 16
_ROUTE_SCHEMA = 13
_ROUTE_SCHEMAS = {6, 7}
_INT64_MAX = (1 << 63) - 1
_MAX_SCALED_QUANTITY = _INT64_MAX // ROUTE_COST_SCALE
MLA_COLUMN_TILE = 96
MLA_K_TILE = 128
MLA_TOKEN_PROFILES = ((1, 1, 1), (8, 2, 8), (16, 9, 16), (32, 17, 32), (64, 33, 64), (128, 65, 128))
MLA_SMALL_PARTIAL_MAX_ROWS = 8
MLA_GLOBAL_FRONT_MAX_ROWS = 64
MLA_DEFAULT_KVR = 576
MLA_DEFAULT_HEADS = 128
MLA_DEFAULT_NOPE = 128
MLA_DEFAULT_ROPE = 64
MLA_CACHE_PAGE_SIZE = 16
MLA_CACHE_NZ_LANE_BYTES = 32
MLA_SCRATCH_TOKEN_CAP = 1024
MLA_SCRATCH_TARGET_BYTES = 32 * 1024 * 1024
MLA_SCRATCH_ALIGNMENT = 32
MLA_QA_PARTIALS = 4
MLA_NZ_PARTIAL_Q_LORA = 1536
MLA_AMAX_LANES = 64
QSLI_TILE_M = 32
QSLI_TILE_N = 512
QSLI_PACKED_D = 64
QSLI_SCALE_ROW_BYTES = 4
QSLI_TRUNK = 16384
QSLI_HISTOGRAM_BINS = 256
QSLI_MERGE_ALIGN = 128
QSLI_STAGING_DEPTH = 4
QSLI_MAX_CUBE_WORKERS = 36
QSLI_METADATA_BYTES = 1024 * 8
MQSMLA_HEADS = 64
MQSMLA_KV_HEADS = 1
MQSMLA_NOPE_DIM = 448
MQSMLA_ROPE_DIM = 64
MQSMLA_HEAD_DIM = MQSMLA_NOPE_DIM + MQSMLA_ROPE_DIM
MQSMLA_KV_TILE = 128
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
MQSMLA_L1_KV_BYTES = 384 * 1024
MQSMLA_L1_Q_BYTES = 112 * 1024
MQSMLA_L1_P_BYTES = 16 * 1024
MQSMLA_KV_DEPTH = 3
MQSMLA_FD_SLOT_BYTES = (MQSMLA_HEADS * MQSMLA_HEAD_DIM + 2 * MQSMLA_HEADS) * FP32_BYTES
MQSMLA_FD_BYTES_PER_CORE = 2 * MQSMLA_FD_SLOT_BYTES


class QkvQuantMode(str, Enum):
    A16C16 = "A16C16"
    FP8 = "FP8"
    FP8_TENSOR = "FP8_TENSOR"
    HiF8 = "HiF8"
    PA_BSND = "PA_BSND"


@dataclass(frozen=True, slots=True)
class FlashKdaProblem:
    lengths: tuple[int, ...]
    value_heads: int = 1
    query_heads: int = 1
    requested_template: str = "auto"
    route_coefficients: tuple[int, ...] = ()
    workspace_slots: int = KDA_DEFAULT_WORKSPACE_SLOTS
    max_chunk_group: int | None = None
    layout_kind: int = 0
    has_cu: bool = False
    calibration_enabled: bool = True
    metadata_word_capacity: int | None = None
    metadata_ms: float = 0
    eager_d2h_bytes: int = 0
    eager_d2h_ms: float = 0
    save_matrices: bool = False
    save_kernel_caches: bool = False
    save_chunk_states: bool = False
    device_name: str = ""
    has_initial_state: bool = False
    return_final_state: bool = True
    enable_grad: bool = False


@dataclass(frozen=True, slots=True)
class MegaRecurrentKdaProblem:
    """Equal-length optimized schedule. Batch is the model batch before TP."""

    batch: int
    tp: int
    sequence: int
    unfused_launches: tuple[str, ...] = (
        "input_projection", "causal_conv", "recurrent_kda", "output_gate", "output_projection",
    )
    state_indices: tuple[int, ...] = ()
    replay_capacity: int | None = None
    u_stride0: int | None = None
    k_stride0: int | None = None
    decay_stride0: int | None = None
    checkpoint_stride0: int | None = None


@dataclass(frozen=True, slots=True)
class CommitRecurrentKdaProblem:
    accepted: tuple[int, ...]
    heads: int = 1
    row_partitions: int | None = None
    state_indices: tuple[int, ...] = ()
    replay_capacity: int | None = None
    u_stride0: int | None = None
    k_stride0: int | None = None
    decay_stride0: int | None = None
    checkpoint_stride0: int | None = None


@dataclass(frozen=True, slots=True)
class MlaPrologProblem:
    tokens: int
    dim: int
    q_lora: int = MLA_NZ_PARTIAL_Q_LORA
    kvr: int = MLA_DEFAULT_KVR
    heads: int = MLA_DEFAULT_HEADS
    q_nope: int = MLA_DEFAULT_NOPE
    rope: int = MLA_DEFAULT_ROPE
    mode: QkvQuantMode | int = QkvQuantMode.A16C16
    weight_format: str = "nz"
    return_qr: bool = True
    cache_layout: str = "PA_BSND"
    cache_page_size: int = MLA_CACHE_PAGE_SIZE
    cache_index: tuple[int, ...] | None = None

    @property
    def quant_mode(self) -> QkvQuantMode:
        mode = self.mode
        if type(mode) is int and 0 <= mode < len(_MLA_QUANT_MODES):
            return _MLA_QUANT_MODES[mode]
        return QkvQuantMode.A16C16 if mode == QkvQuantMode.PA_BSND else QkvQuantMode(mode)

    @property
    def cache_row_bytes(self) -> int:
        if self.quant_mode == QkvQuantMode.FP8:
            latent = self.kvr - self.rope
            return latent + self.rope * BF16_BYTES + ceil_div(latent, KDA_HEAD_DIM) * FP32_BYTES
        return self.kvr * BF16_BYTES if self.quant_mode == QkvQuantMode.A16C16 else self.kvr


_MLA_QUANT_MODES = (QkvQuantMode.A16C16, QkvQuantMode.FP8, QkvQuantMode.FP8_TENSOR, QkvQuantMode.HiF8)
MLA_QUANTIZED_QUERY = {QkvQuantMode.FP8_TENSOR, QkvQuantMode.HiF8}


@dataclass(frozen=True, slots=True)
class AttnResProblem:
    tokens: int
    hidden_size: int
    dtype: str = "bf16"


@dataclass(frozen=True, slots=True)
class QsliProblem:
    query_rows: int
    candidates: int
    topk: int
    split_parts: int = 1
    metadata_ms: float = 0


@dataclass(frozen=True, slots=True)
class MixedQuantSparseMlaProblem:
    n_ori: int
    n_cmp: int
    queries: int = 1
    heads: int = MQSMLA_HEADS
    kv_heads: int = MQSMLA_KV_HEADS
    head_dim: int = MQSMLA_HEAD_DIM
    split_parts: int = 1
    metadata_ms: float = 0
    part_lengths: tuple[int, ...] = ()
    topk_indices: tuple[int, ...] | tuple[tuple[int, ...], ...] = ()
    page_table: tuple[int, ...] = ()
    block_table: tuple[int, ...] = ()
    page_size: int = MQSMLA_KV_TILE
    has_sink: bool = True
    sink_size: int = 0
    ori_block_table: tuple[int, ...] = ()
    cmp_block_table: tuple[int, ...] = ()

    @property
    def block_tables(self):
        return self.ori_block_table or self.page_table or self.block_table, self.cmp_block_table

    def sparse_rows(self, query):
        if self.topk_indices and isinstance(self.topk_indices[0], (tuple, list)):
            return self.topk_indices[query]
        return self.topk_indices


_DSL_PROBLEMS = dict(zip(DSL_KERNEL_IDS, (
    FlashKdaProblem, MegaRecurrentKdaProblem, CommitRecurrentKdaProblem, MlaPrologProblem,
    AttnResProblem, QsliProblem, MixedQuantSparseMlaProblem,
)))


def dsl_problem(kernel_id, problem):
    name = next((key for key in DSL_KERNEL_IDS if kernel_id == key or kernel_id.startswith(key + ".")), None)
    if name is None:
        raise ModelingError(f"dsl_kernel_unknown:{kernel_id}")
    cls = _DSL_PROBLEMS[name]
    normalized = cls(**problem) if isinstance(problem, Mapping) else problem
    if not isinstance(normalized, cls):
        raise ModelingError(f"dsl_problem_type_mismatch:{kernel_id}")
    return name, normalized


def flash_kda_lane_tasks(p, spec):
    chunks = tuple(ceil_div(length, KDA_CHUNK) for length in p.lengths)
    if spec.template_variant == "persistent":
        loads = [0] * spec.aic_budget
        for batch in sorted(range(len(chunks)), key=lambda i: (-chunks[i], i)):
            for head in range(p.value_heads):
                owner = min(range(len(loads)), key=loads.__getitem__)
                loads[owner] += chunks[batch]
                yield tuple((batch, head, chunk, owner, 0) for chunk in range(chunks[batch]))
        return
    _, group = flash_kda_schedule(p, spec.aic_budget)
    consumed = [0] * len(chunks)
    for first, batch_counts in flash_kda_rounds(chunks, p.value_heads, p.workspace_slots, group):
        tasks = []
        for batch, count in enumerate(batch_counts, first):
            for head in range(p.value_heads):
                owner = (batch * p.value_heads + head) % spec.aic_budget
                for chunk in range(consumed[batch], consumed[batch] + count):
                    tasks.append((batch, head, chunk, owner, len(tasks)))
            consumed[batch] += count
        yield tuple(tasks)


def flash_kda_schedule(problem: FlashKdaProblem, core_count: int) -> tuple[str, int]:
    p = problem
    return resolve_flash_kda_route(
        p.lengths, p.value_heads, core_count, requested_template=p.requested_template,
        device_name=p.device_name, route_coefficients=p.route_coefficients, layout_kind=p.layout_kind,
        workspace_slots=p.workspace_slots, max_chunk_group=p.max_chunk_group, metadata_word_capacity=p.metadata_word_capacity,
        has_cu=p.has_cu, calibration_enabled=p.calibration_enabled, enable_grad=p.enable_grad, query_heads=p.query_heads,
    )


def resolve_flash_kda_route(
    lengths: Sequence[int], value_heads: int, core_count: int, *,
    requested_template: str = "auto", device_name: str = "", route_coefficients: Sequence[int] = (),
    layout_kind: int = 0, workspace_slots: int = KDA_DEFAULT_WORKSPACE_SLOTS, max_chunk_group: int | None = None,
    metadata_word_capacity: int | None = None, has_cu: bool = False, calibration_enabled: bool = True,
    enable_grad: bool = False, query_heads: int | None = None,
) -> tuple[str, int]:
    """Replay the pinned launcher and host route comparison in integer units."""
    del lengths, value_heads, core_count, requested_template, device_name, route_coefficients
    del layout_kind, workspace_slots, max_chunk_group, metadata_word_capacity
    del has_cu, calibration_enabled, enable_grad, query_heads
    raise NotImplementedError(
        "LEGACY_PURGED: dispatch flash_kda route via structured ComputeArch capability (zero device-name string check or silent 'staged' fallback)"
    )


def _flash_route(lengths, chunks, heads, q_heads, cores, slots, max_group, metadata_capacity, coefficients):
    del lengths, chunks, heads, q_heads, cores, slots, max_group, metadata_capacity, coefficients
    raise NotImplementedError(
        "LEGACY_PURGED: remove silent 'staged' fallbacks in _flash_route and use structured ComputeArch route parameters"
    )


def select_flash_kda_template(lengths, core_count, requested_template="auto", cost_coefficients=(), **limits):
    return flash_kda_schedule(FlashKdaProblem(tuple(lengths), requested_template=requested_template,
                                              route_coefficients=tuple(cost_coefficients), **limits), core_count)[0]


def commit_aic_quota(available):
    choices = tuple(quota for quota in COMMIT_CORE_QUOTAS if quota <= available)
    if not choices:
        raise ModelingError("commit_kda_minimum_aic_quota_unavailable")
    return choices[-1]


def _aligned(value, boundary):
    return ceil_div(value, boundary) * boundary


def mla_workspaces(p, aic):
    """Replay launcher scratch shapes, alignment and alias groups."""
    mode = p.quant_mode
    profile = next((size for size, lo, hi in MLA_TOKEN_PROFILES if lo <= p.tokens <= hi), 0)
    rows = profile or _aligned(p.tokens, MLA_K_TILE)
    qp = p.heads * (p.q_nope + p.rope)
    padded_qa, padded_kva, padded_qp = (_aligned(width, MLA_COLUMN_TILE) for width in (p.q_lora, p.kvr, qp))
    quantized = mode in MLA_QUANTIZED_QUERY
    scratch_rows = rows
    if rows > MLA_SCRATCH_TOKEN_CAP:
        per_row = BF16_BYTES * (max(padded_qa, padded_qp) + padded_kva + p.q_lora + p.dim)
        if quantized:
            per_row += FP32_BYTES * p.heads * p.kvr
        scratch_rows = MLA_SCRATCH_TOKEN_CAP
        while scratch_rows > MLA_K_TILE and scratch_rows * per_row > MLA_SCRATCH_TARGET_BYTES:
            scratch_rows //= 2
    recompute = (mode == QkvQuantMode.HiF8 and rows > scratch_rows
                 and rows * p.heads * p.kvr * FP32_BYTES > MLA_SCRATCH_TARGET_BYTES
                 and (p.return_qr or p.q_lora * BF16_BYTES < p.heads * p.kvr * FP32_BYTES))
    partials = (
        p.weight_format == "nd" and 0 < profile <= MLA_SMALL_PARTIAL_MAX_ROWS
        or p.weight_format == "nz" and mode == QkvQuantMode.A16C16
        and MLA_SMALL_PARTIAL_MAX_ROWS < p.tokens <= MLA_GLOBAL_FRONT_MAX_ROWS and p.q_lora == MLA_NZ_PARTIAL_Q_LORA
    )
    specs = {
        "ws_qa": scratch_rows * padded_qa * (MLA_QA_PARTIALS * FP32_BYTES if partials else BF16_BYTES),
        "ws_kva": scratch_rows * padded_kva * BF16_BYTES,
        "ws_q_proj": scratch_rows * padded_qp * BF16_BYTES,
        "ws_qb_aux": (0 if p.return_qr else rows if recompute else scratch_rows) * p.q_lora * BF16_BYTES,
    }
    if quantized:
        q_rows = rows if mode == QkvQuantMode.HiF8 and not recompute else scratch_rows
        specs["ws_q_fp"] = q_rows * p.heads * p.kvr * FP32_BYTES
    if mode == QkvQuantMode.HiF8:
        specs["ws_amax"] = _aligned(aic, MLA_AMAX_LANES) * FP32_BYTES
    if rows != p.tokens:
        specs["ws_x_pad"] = scratch_rows * p.dim * BF16_BYTES
    groups = [max(specs["ws_qa"], specs["ws_q_proj"]), specs["ws_qb_aux"], specs.get("ws_x_pad", 0),
              specs.get("ws_amax", 0)]
    if mode == QkvQuantMode.FP8_TENSOR or recompute:
        groups.append(max(specs["ws_kva"], specs["ws_q_fp"]))
    else:
        groups.extend((specs["ws_kva"], specs.get("ws_q_fp", 0)))
    return specs, sum(_aligned(size, MLA_SCRATCH_ALIGNMENT) for size in groups), scratch_rows, recompute


def _buf(name, pool, size, slots=1, *, start=0, end=1, scale=0):
    return LiveBufferSpec(name, pool, size, start, end, owner_kind="aiv" if pool == "ub" else "aic",
                          scale_bytes=scale, slots=slots)


def _flash(p, aic, aiv, datapath, variant):
    route, _ = flash_kda_schedule(p, aic)
    route = variant or route
    if route not in {"persistent", "staged"}:
        raise ModelingError("flash_kda_variant_invalid")
    sizes = (
        _buf("wy_inverse", "l1", KDA_CHUNK**2 * BF16_BYTES, KDA_INVERSE_BUFFERS),
        _buf("q_l0", "l0a", KDA_CHUNK * KDA_HEAD_DIM * BF16_BYTES, 2),
        _buf("k_l0", "l0b", KDA_CHUNK * KDA_HEAD_DIM * BF16_BYTES, 2),
        _buf("score_l0", "l0c", KDA_CHUNK**2 * FP32_BYTES, 2),
        _buf("state_delta", "ub", KDA_HEAD_DIM * KDA_STATE_HALF * FP32_BYTES),
    )
    workspace = p.workspace_slots * KDA_HEAD_DIM * (KDA_CHUNK * BF16_BYTES + FP32_BYTES)
    workspace += aic * 2 * 2 * KDA_CHUNK * FP32_BYTES
    policy_names = []
    if p.save_matrices:
        policy_names.extend(("Akk", "Aqk"))
    if p.save_kernel_caches:
        policy_names.extend(("qg", "g_cumsum"))
    if route == "staged" or p.save_chunk_states:
        policy_names.append("h")
    if route == "persistent":
        sizes += (_buf("state_fp32", "ub", KDA_HEAD_DIM * KDA_STATE_HALF * FP32_BYTES),
                  _buf("state_l1", "l1", KDA_HEAD_DIM**2 * BF16_BYTES, 2))
    else:
        padded = sum(ceil_div(length, KDA_CHUNK) * KDA_CHUNK for length in p.lengths) * p.value_heads
        workspace += padded * (2 * 2 * KDA_HEAD_DIM * BF16_BYTES + KDA_CHUNK * BF16_BYTES)
        if not p.save_chunk_states:
            workspace += padded // KDA_CHUNK * KDA_HEAD_DIM**2 * FP32_BYTES
    return (route, (KDA_CHUNK, KDA_CHUNK, KDA_HEAD_DIM), sizes, workspace,
            (("score", "l0c", "ub"), ("operands", "ub", "l1")), tuple((name, "disable") for name in policy_names),
            sum(ceil_div(n, KDA_CHUNK) for n in p.lengths) * p.value_heads)


def _mega(p, aic, aiv, datapath, variant):
    del p, aic, aiv, datapath, variant
    raise NotImplementedError(
        "LEGACY_PURGED: parameterize mega_recurrent_kda strictly from problem.num_heads and problem.hidden_size (zero hardcoded 96/7168)"
    )


def mega_workspace_sizes(p):
    del p
    raise NotImplementedError(
        "LEGACY_PURGED: parameterize mega_workspace_sizes strictly from problem.num_heads (zero hardcoded 96)"
    )


def _commit(p, aic, aiv, datapath, variant):
    if p.heads < 1 or (p.accepted and min(p.accepted) < 0):
        raise ModelingError("commit_kda_problem_invalid")
    _validate_kda_slots(p, len(p.accepted), max(p.accepted, default=0))
    scheduling = commit_rows(p, aiv)
    rows = max((row[2] for lane in scheduling for row in lane), default=KDA_HEAD_DIM)
    buffers = (
        _buf("state_current", "ub", KDA_HEAD_DIM**2 * FP32_BYTES),
        _buf("state_next", "ub", KDA_HEAD_DIM**2 * FP32_BYTES),
        _buf("replay_u", "ub", KDA_HEAD_DIM * FP32_BYTES, 2),
        _buf("u_fp32", "ub", KDA_HEAD_DIM * FP32_BYTES),
        _buf("replay_k", "ub", KDA_HEAD_DIM * BF16_BYTES, 2),
        _buf("replay_decay", "ub", KDA_HEAD_DIM * FP32_BYTES, 2),
    )
    return (f"quota{aic}", (rows, KDA_HEAD_DIM, 1), buffers, 0, (), (), sum(map(len, scheduling)))


def _validate_kda_slots(p, batch, required):
    slots = p.state_indices
    capacity = replay_pool_capacity(p.replay_capacity, required)
    if type(capacity) is not int or capacity < max(1, required) or (slots and (
        len(slots) != batch or len(set(slots)) != batch or any(type(i) is not int or i < 0 for i in slots)
    )):
        raise ModelingError("recurrent_kda_paged_state_geometry_invalid")
    if not isinstance(p, CommitRecurrentKdaProblem):
        raise NotImplementedError(
            "LEGACY_PURGED: derive KDA heads from explicit problem.num_heads (zero hardcoded 96)"
        )
    replay_pool_strides(p, p.heads, capacity)


def replay_pool_strides(p, heads, capacity):
    dense = heads * capacity * KDA_HEAD_DIM
    checkpoint = heads * KDA_HEAD_DIM**2
    strides = (p.u_stride0, p.k_stride0, p.decay_stride0, p.checkpoint_stride0)
    widths = (dense, dense, dense, checkpoint)
    if any(stride is not None and (type(stride) is not int or stride < width)
           for stride, width in zip(strides, widths, strict=True)):
        raise ModelingError("recurrent_kda_pool_stride_invalid")
    return tuple(width if stride is None else stride for stride, width in zip(strides, widths, strict=True))


def replay_pool_capacity(requested, required):
    return max(KDA_REPLAY_CAPACITY, required) if requested is None else requested


def _mla(p, aic, aiv, datapath, variant):
    if (min(p.tokens, p.dim, p.q_lora, p.heads, p.q_nope, p.cache_page_size) < 1 or not 0 <= p.rope < p.kvr
            or p.weight_format not in {"nd", "nz"} or p.cache_layout not in {"PA_BSND", "PA_NZ"}):
        raise ModelingError("mla_prolog_problem_invalid")
    if p.cache_index is not None and (
        not isinstance(p.cache_index, (tuple, list)) or len(p.cache_index) != p.tokens
        or any(type(i) is not int for i in p.cache_index)
    ):
        raise ModelingError("mla_prolog_cache_index_invalid")
    mode = p.quant_mode
    if mode == QkvQuantMode.FP8 and p.cache_layout != "PA_BSND":
        raise ModelingError("mla_prolog_group_fp8_requires_pa_bsnd")
    if p.cache_layout == "PA_NZ" and p.cache_row_bytes % MLA_CACHE_NZ_LANE_BYTES:
        raise ModelingError("mla_prolog_pa_nz_cache_row_unaligned")
    _, workspace, scratch_rows, _ = mla_workspaces(p, aic)
    m = min(scratch_rows, MLA_K_TILE)
    tile = (m, MLA_COLUMN_TILE, MLA_K_TILE)
    buffers = (
        _buf("projection_a", "l1", m * MLA_K_TILE * BF16_BYTES, 2),
        _buf("projection_b", "l1", MLA_COLUMN_TILE * MLA_K_TILE * BF16_BYTES, 2),
        _buf("projection_a_l0", "l0a", m * MLA_K_TILE * BF16_BYTES),
        _buf("projection_b_l0", "l0b", MLA_COLUMN_TILE * MLA_K_TILE * BF16_BYTES),
        _buf("projection_out", "l0c", m * MLA_COLUMN_TILE * FP32_BYTES),
        _buf("norm_rope_quant", "ub", min(m, KDA_STATE_HALF) * MLA_COLUMN_TILE * FP32_BYTES, 2),
    )
    return QkvQuantMode.PA_BSND.value if p.mode == QkvQuantMode.PA_BSND else mode.value, tile, buffers, workspace, (), (), p.tokens


def _attn_res(p, aic, aiv, datapath, variant):
    del p, aic, aiv, datapath, variant
    raise NotImplementedError(
        "LEGACY_PURGED: parameterize _attn_res strictly from problem.hidden_size (zero hardcoded 7168 fingerprint)"
    )


def qsli_workspace_sizes(p, aic):
    staging = QSLI_MAX_CUBE_WORKERS * QSLI_STAGING_DEPTH * QSLI_TILE_N
    return {
        "staging_key": staging * QSLI_PACKED_D, "staging_scale": staging * QSLI_SCALE_ROW_BYTES,
        "key_workspace": aic * p.candidates * BF16_BYTES,
        "selected_position_workspace": aic * p.topk * INT32_BYTES,
        "partial_indices": 2 * aic * p.topk * INT32_BYTES,
        "partial_values": 2 * aic * p.topk * BF16_BYTES,
    }


QSLI_MATRIX_DTYPE = "mxfp4"


def _qsli(p, aic, aiv, datapath, variant):
    arch = datapath.compute_arch.declared
    if arch is None or not arch.supports_native("matrix", QSLI_MATRIX_DTYPE, "fp32"):
        raise unmodeled(QSLI, "qsli_requires_native_mxfp4")
    if (p.topk < 1 or p.candidates < 0 or p.query_rows < 1
            or not 1 <= p.split_parts <= min(max(1, p.candidates), aic)
            or aic > QSLI_MAX_CUBE_WORKERS):
        raise ModelingError("qsli_problem_invalid")
    if not p.candidates:
        return "histogram_topk", (1, 1, 1), (), 0, (), (), p.query_rows
    buffers = (
        _buf("q_l1", "l1", QSLI_TILE_M * QSLI_PACKED_D, scale=QSLI_TILE_M * QSLI_SCALE_ROW_BYTES),
        _buf("k_l1", "l1", QSLI_TILE_N * QSLI_PACKED_D, 2, scale=QSLI_TILE_N * QSLI_SCALE_ROW_BYTES),
        _buf("l0a", "l0a", QSLI_TILE_M * QSLI_PACKED_D),
        _buf("l0b", "l0b", QSLI_TILE_N * QSLI_PACKED_D, 2),
        _buf("l0c", "l0c", QSLI_TILE_M * QSLI_TILE_N * FP32_BYTES, 2),
        _buf("score_half", "ub", QSLI_TILE_M * QSLI_TILE_N // 2 * FP32_BYTES),
        _buf("masked_histogram", "ub", QSLI_HISTOGRAM_BINS * BF16_BYTES, start=1, end=2),
        _buf("topk_merge", "ub", 2 * _aligned(_aligned(p.topk, QSLI_HISTOGRAM_BINS) + QSLI_TRUNK,
                                               QSLI_MERGE_ALIGN) * BF16_BYTES, start=1, end=2),
    )
    return ("histogram_topk", (QSLI_TILE_M, QSLI_TILE_N, KDA_HEAD_DIM), buffers,
            sum(qsli_workspace_sizes(p, aic).values()), (("score", "l0c", "ub"),), (), p.query_rows)


def _mqsmla(p, aic, aiv, datapath, variant):
    if (p.heads, p.kv_heads, p.head_dim) != (MQSMLA_HEADS, MQSMLA_KV_HEADS, MQSMLA_HEAD_DIM):
        raise ModelingError("mqsmla_fixed_head_geometry_required")
    if min(p.n_ori, p.n_cmp) < 0 or p.queries < 1 or not 1 <= p.split_parts <= aic:
        raise ModelingError("mqsmla_problem_invalid")
    tables = p.block_tables
    if (sum(map(bool, (p.ori_block_table, p.page_table, p.block_table))) > 1 or p.page_size < 1
            or not p.has_sink or p.sink_size not in (0, p.heads)
            or any(type(page) is not int or page < 0 for table in tables for page in table)):
        raise ModelingError("mqsmla_paged_operand_invalid")
    total = p.n_ori + p.n_cmp
    if p.topk_indices:
        batched = isinstance(p.topk_indices[0], (tuple, list))
        if batched and len(p.topk_indices) != p.queries:
            raise ModelingError("mqsmla_sparse_indices_invalid")
        for row in p.topk_indices if batched else (p.topk_indices,):
            if not isinstance(row, (tuple, list)) or len(row) != total or any(
                type(i) is not int or i < 0 or (table and i // p.page_size >= len(table))
                for table, indices in zip(tables, (row[:p.n_ori], row[p.n_ori:]), strict=True) for i in indices
            ):
                raise ModelingError("mqsmla_sparse_indices_invalid")
    elif any(table and rows > len(table) * p.page_size for table, rows in zip(tables, (p.n_ori, p.n_cmp), strict=True)):
        raise ModelingError("mqsmla_paged_operand_invalid")
    if p.split_parts > 1 and p.queries * p.split_parts > 2 * aic:
        raise ModelingError("mqsmla_fd_partial_slots_exhausted")
    if p.split_parts > max(1, p.n_ori + p.n_cmp) or (
        p.part_lengths and (min(p.part_lengths) < 1 or len(p.part_lengths) != p.split_parts
                            or sum(p.part_lengths) != p.n_ori + p.n_cmp)
    ):
        raise ModelingError("mqsmla_split_geometry_invalid")
    buffers = (
        _buf("kv_l1", "l1", MQSMLA_L1_KV_BYTES // MQSMLA_KV_DEPTH, MQSMLA_KV_DEPTH),
        _buf("q_l1", "l1", MQSMLA_L1_Q_BYTES), _buf("p_l1", "l1", MQSMLA_L1_P_BYTES),
        _buf("qk_l0c", "l0c", MQSMLA_HEADS * MQSMLA_KV_TILE * FP32_BYTES, 2),
        _buf("q_l0a", "l0a", MQSMLA_HEADS * KDA_CHUNK * BF16_BYTES, 2),
        _buf("kv_l0b", "l0b", MQSMLA_KV_TILE * KDA_CHUNK * BF16_BYTES, 2),
        _buf("kv_packed", "ub", MQSMLA_KV_TILE // 2 * KV_ROW_BYTES_ORI, 2),
        _buf("online_softmax", "ub", MQSMLA_FD_SLOT_BYTES // 2),
    )
    return ("dual_bank", (MQSMLA_HEADS, MQSMLA_KV_TILE, p.head_dim), buffers,
            aic * MQSMLA_FD_BYTES_PER_CORE, (("scores", "l0c", "ub"), ("p", "ub", "l1")), (), p.queries)


_DESCRIBE = dict(zip(DSL_KERNEL_IDS, (_flash, _mega, _commit, _mla, _attn_res, _qsli, _mqsmla)))
_VECTOR_ONLY = {ATTN_RES, COMMIT_KDA}
_ALGORITHMS = {FLASH_KDA: "chunk_recurrence", MEGA_KDA: "chunk_recurrence",
               COMMIT_KDA: "chunk_recurrence", MQSMLA: "online_reduce"}


def describe_dsl_kernel(kernel_id, problem, datapath, *, aic_budget=None, aiv_budget=None,
                        memory_view=None, die=0, variant=None):
    name, p = dsl_problem(kernel_id, problem)
    if datapath.arch_id not in DIRECT_MATRIX_ARCH_IDS:
        raise unmodeled(name, "dsl_requires_arch35")
    aic = datapath.aic_count if aic_budget is None else aic_budget
    aiv = datapath.aiv_count if aiv_budget is None else aiv_budget
    if not (1 <= aic <= datapath.aic_count and 1 <= aiv <= datapath.aiv_count):
        raise ModelingError("dsl_core_budget_invalid")
    if name == COMMIT_KDA:
        aic = commit_aic_quota(aic)
        aiv = min(aiv, 2 * aic)
    if kernel_id != name:
        supplied = kernel_id.removeprefix(name + ".")
        if variant is not None and supplied != variant:
            raise ModelingError("dsl_kernel_variant_conflict")
        variant = supplied
    selected, dims, buffers, workspace, routes, cache, items = _DESCRIBE[name](p, aic, aiv, datapath, variant)
    if variant is not None and selected != variant:
        raise ModelingError(f"dsl_kernel_variant_invalid:{name}:{variant}")
    for _, src, dst in routes:
        pairs = tuple((c, v) for c, aivs in datapath.aic_aiv_pairs if c < aic for v in aivs if v < aiv)
        active = min(aic, max(1, items * p.split_parts if name in {QSLI, MQSMLA} else items))
        if not set(range(active)) <= {c for c, _ in pairs}:
            raise unmodeled(name, "dsl_cross_core_pairing_unavailable")
        if name == QSLI and any(sum(c == core for c, _ in pairs) < 2 for core in range(active)):
            raise unmodeled(name, "qsli_requires_two_paired_subblocks")
        for cube, vector in pairs:
            datapath.route(src, dst, "fp32" if src == "l0c" else "bf16",
                           (cube, vector) if src == "l0c" else (vector, cube))
    owner = "aiv" if name in _VECTOR_ONLY else "aic"
    algo = _ALGORITHMS.get(name, "resident_1pass")
    solution = solve_template_tile(datapath, name, TileCandidate(
        algo, TileShape(*dims), max((b.slots for b in buffers), default=1), (aic, aiv), items,
        ceil_div(items, aiv if owner == "aiv" else aic), buffers, owner_axis=owner,
    ), state=memory_view, die=die)
    peaks = {peak.pool: peak.peak_bytes for peak in solution.pool_peaks}
    raise NotImplementedError(_DSL_SOURCE_REFS_PURGED_MSG)


def enumerate_dsl_specs(kernel_id, problem, datapath, *, all_variants=False, **options):
    name, p = dsl_problem(kernel_id, problem)
    if all_variants and kernel_id == name == FLASH_KDA:
        feasible = []
        for variant in ("persistent", "staged"):
            try:
                feasible.append(describe_dsl_kernel(kernel_id, p, datapath, variant=variant, **options)[0])
            except TileInfeasibleError:
                continue
        return tuple(feasible)
    return (describe_dsl_kernel(kernel_id, p, datapath, **options)[0],)
