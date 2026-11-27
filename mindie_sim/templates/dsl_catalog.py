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
    ATTN_RES_BF16_ALIGN, ATTN_RES_UB_BYTES, BF16_BYTES, DIRECT_MATRIX_ARCH_IDS, FP32_BYTES, KDA_HEAD_DIM,
    KDA_STATE_HALF, DslKernelSpec, TemplateSourceRef, attn_res_tiling, attn_res_ub_footprint, ceil_div,
    commit_replay_bytes, commit_rows, flash_kda_rounds,
)
from .fuse_region import solve_template_tile

DSL_REPOSITORY = "Ascend/ops-transformer (tilelang-0.1.15)"
DSL_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
DSL_SOURCE_REFS = (
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/flash_kda/op_dsl/arch35/flash_kda_metadata.py",
        "78daefe5a42d384557ef3f78ff4716b583edb7b6189e308fa9a78750b5e54572", "_select_template_on_host", (561, 660),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/flash_kda/op_dsl/arch35/flash_kda_metadata.py",
        "78daefe5a42d384557ef3f78ff4716b583edb7b6189e308fa9a78750b5e54572", "_DT_ROUTE_PARAMETERS", (1538, 1562),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/flash_kda/op_dsl/arch35/flash_kda_persistent.py",
        "20ca2cb501778657240dce4fbaeb2a72ee2cc7dfab56008b619a572f4eb0c313", "CHUNK_SIZE", (82, 160),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/flash_kda/op_dsl/arch35/flash_kda_staged.py",
        "6d700560384b670fa08acb71642e6c147b7252a61a7ac1d02aa875254f4fa54f", "_L2_CACHE_DISABLE", (68, 130),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "experimental/attention/mega_recurrent_kda/op_dsl/arch35/mega_recurrent_kda_backend.py",
        "ed6987f093cb6cd3ad529bd85ab7f3a583495a428d76a9e62ff15618cb8da5c4", "RecurrentKDAStage", (553, 720),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/recurrent_kda/op_host/op_api/aclnn_recurrent_kda.cpp",
        "7a1b6b43921ed6f7b4c0c760ad2a865aac739ec90a2d820be18b7830de5077c7", "aclnnRecurrentKdaGetWorkspaceSize", (503, 590),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/mla_prolog_v3/examples/arch35/test_aclnn_mla_prolog_v4_weight_nz_fqkvq.cpp",
        "eacd138b302594199e390ea59a9c7a7898233d113ddfae9d009b10e66059f71a", "aclnnMlaPrologV4WeightNz", (380, 440),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/block_attn_res_update/op_api/aclnn_block_attn_res_update.cpp",
        "1ae707c1a16982aee33ca756c2d83b888c1387ae426fa39e46daa51cbd774826", "aclnnBlockAttnResUpdateGetWorkspaceSize", (36, 120),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/quant_sparse_lightning_indexer/op_dsl/quant_sparse_lightning_indexer.py",
        "2e1500889ac1ec292d1cba0c3f3c8ca876f52d346152a127df1f5289212e60f0", "QsliCube", (1005, 1110),
    ),
    TemplateSourceRef(
        DSL_REPOSITORY, DSL_REVISION, "attention/mixed_quant_sparse_flash_mla/op_api/aclnn_mixed_quant_sparse_flash_mla.cpp",
        "25299f4b4f35b01f007962d49eb3821fd7c184fd2a2564f78c7261f88261725a", "aclnnMixedQuantSparseFlashMlaGetWorkspaceSize", (43, 85),
    ),
)
FLASH_KDA = "dsl.flash_kda"
MEGA_KDA = "dsl.mega_recurrent_kda"
COMMIT_KDA = "dsl.commit_recurrent_kda"
MLA_PROLOG = "dsl.mla_prolog_v4"
ATTN_RES = "dsl.block_attn_res_update_rms_norm"
QSLI = "dsl.quant_sparse_lightning_indexer"
MQSMLA = "dsl.mixed_quant_sparse_flash_mla"
_FLASH_SOURCE_COUNT = 4
_DSL_SOURCES = dict(zip(
    (FLASH_KDA, MEGA_KDA, COMMIT_KDA, MLA_PROLOG, ATTN_RES, QSLI, MQSMLA),
    (DSL_SOURCE_REFS[:_FLASH_SOURCE_COUNT], *(DSL_SOURCE_REFS[i:i + 1] for i in range(_FLASH_SOURCE_COUNT, len(DSL_SOURCE_REFS)))),
    strict=True,
))
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
KDA_CONV_TAPS = 4
KDA_CONV_STREAMS = 3
KDA_CONV_HISTORY_KEEP = KDA_CONV_TAPS - 2
MEGA_MODEL_HEADS = 96
MEGA_HIDDEN_SIZE = 7168
MEGA_TP_DEGREES = (4, 8, 16)
MEGA_SEQUENCE_LENGTHS = (4, 8)
KDA_REPLAY_CAPACITY = max(MEGA_SEQUENCE_LENGTHS)
MEGA_MODEL_BATCH_LIMIT = 32
MEGA_AIC_QUOTAS = (8, 16, 24, 28, 32)
COMMIT_CORE_QUOTAS = (8, 24, 28, 32)
ATTN_RES_HIDDEN_SIZE = 7168
ATTN_RES_VL = 64
ROUTE_COST_SCALE = 64
PARTS_PER_MILLION = 1_000_000
_ROUTE_PARAMS_STRIDE = 16
_ROUTE_SCHEMA = 13
_ROUTE_SCHEMAS = {6, 7}
_INT64_MAX = (1 << 63) - 1
_MAX_SCALED_QUANTITY = _INT64_MAX // ROUTE_COST_SCALE
_ARCH35_BASE_ROUTE_PARAMETERS = (
    0, 2711857704, 699286307, 16555661144, 28359460, 94263834, 94263834, 12013349865,
    1045656019, 84358156, 12842477, 2527878117, 181898838, 6, 8441, 0,
    0, 692059912, 851406583, 21510815509, 73490570, 81791118, 166178779, 12029971961,
    2552246171, 459139222, 45086212, 2303638937, 143977434, 6, 18992, 0,
)
_ARCH35_HIGH_BW_ROUTE_PARAMETERS = (
    4135271144, 193981546, 366448654, 12881511041, 29417161, 4351059785, 158980371, 10652650460,
    3206978745, 332321804, 0, 419242242, 29945874, 7, 0, 0,
    20317858430, 1160598799, 1018161115, 1015361233, 14389959, 933231823, 115749683, 11499084207,
    4373896639, 413928806, 41555011, 0, 0, 7, 0, 0,
)
_FLASH_ROUTE_PROFILES: Mapping[str, tuple[int, ...]] = {
    "arch35_high_bw": _ARCH35_HIGH_BW_ROUTE_PARAMETERS,
    "arch35_base": _ARCH35_BASE_ROUTE_PARAMETERS,
}
_HIGH_BW_HBM_THRESHOLD_BPS = 2_500_000_000_000.0
_ARCH35_BASE_SKU_TAGS = frozenset({"ascend950pr", "ascend950pr_9579", "arch35_base"})
_ARCH35_HIGH_BW_SKU_TAGS = frozenset({"ascend950dt", "ascend950dt_9582", "ascend950dt_9589", "arch35_high_bw"})
MLA_COLUMN_TILE = 96
MLA_K_TILE = 128
MLA_QB_GROUP_HEADS = 4
MLA_DIM_MIN = 1024
MLA_DIM_MAX = 8192
MLA_DIM_ALIGN = 256
MLA_Q_LORA_WIDTHS = frozenset({640, 1024, 1536, 2048})
MLA_MAX_HEADS = 128
MLA_KV_LORA_DIM = 512
MLA_NOPE_ROPE_PAIRS = frozenset({(128, 64), (192, 64), (256, 0)})
MLA_PAGE_SIZE_MIN = 16
MLA_PAGE_SIZE_MAX = 1024
MLA_PAGE_SIZE_ALIGN = 16
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
MLA_FRONT_SMALL_NZ_WIDTH = 32
MLA_FRONT_PARTIAL_1536_WIDTH = 48
MLA_FRONT_MEDIUM_NZ_WIDTH = 64
MLA_GLOBAL_FRONT_1536_WIDTH = 192
MLA_SMALL_NZ_Q_LORA_WIDTHS = frozenset({640, 1024})
MLA_QB_GROUP_Q_LORA_WIDTHS = frozenset({1536, 2048})
MLA_QB_GROUP_MIN_HEADS = 96
MLA_QB_GROUP_BM16_HEADS = 2
MLA_QB_GROUP_MAX_BM = 16
MLA_QB_WIDE_WIDTHS = (256, 192)
MLA_QB_NZ_WIDTHS = (128, 96, 64)
MLA_WKB_STREAM_MIN_HEADS = 64
MLA_WKB_N_CHUNKS = 4
QSLI_TILE_M = 32
QSLI_TILE_N = 512
QSLI_PACKED_D = 64
QSLI_SCALE_ROW_BYTES = 4
QSLI_CANDIDATE_BLOCK_SIZE = 8
QSLI_CANDIDATE_CAPACITY = 2048
QSLI_MAX_TOPK = 8192
QSLI_TRUNK = 16384
QSLI_HISTOGRAM_BINS = 256
QSLI_MERGE_ALIGN = 128
QSLI_STAGING_DEPTH = 4
QSLI_MAX_CUBE_WORKERS = 36
QSLI_METADATA_BYTES = 1024 * 8
MQSMLA_HEADS = 64
MQSMLA_SMALL_HEAD_BUCKET = 32
MQSMLA_KV_HEADS = 1
MQSMLA_NOPE_DIM = 448
MQSMLA_ROPE_DIM = 64
MQSMLA_HEAD_DIM = MQSMLA_NOPE_DIM + MQSMLA_ROPE_DIM
MQSMLA_KV_TILE = 128
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
MQSMLA_L1_KV_BYTES = 384 * 1024
MQSMLA_Q_L1_WIDTH = 896
MQSMLA_L1_Q_BYTES = MQSMLA_HEADS * MQSMLA_Q_L1_WIDTH * BF16_BYTES
MQSMLA_L1_P_BYTES = MQSMLA_HEADS * MQSMLA_KV_TILE * BF16_BYTES
MQSMLA_KV_DEPTH = 3
MQSMLA_L0_SLOTS = 4
MQSMLA_PV_ACC_SLOTS = 8
MQSMLA_RESULT_CHANNELS = 2
MQSMLA_RESULT_HALF_D = MQSMLA_HEAD_DIM // 2
MQSMLA_RESULT_ROW_FACTOR = 2
MQSMLA_RAW_GATHER_ROWS = 32
MQSMLA_V0_OUT_FRACTALS = MQSMLA_HEAD_DIM // 16
MQSMLA_V0_OUT_PAD_ROWS = 17
MQSMLA_V0_OUT_LANE = 16
MQSMLA_V0_OUT_DEPTH = 2
MQSMLA_SOFTMAX_STATE_VECTORS = 8
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
    num_heads: int
    hidden_size: int
    unfused_launches: tuple[str, ...] = (
        "input_projection", "causal_conv", "recurrent_kda", "output_gate", "output_projection",
    )
    state_indices: tuple[int, ...] = ()
    replay_capacity: int | None = None
    u_stride0: int | None = None
    k_stride0: int | None = None
    decay_stride0: int | None = None
    checkpoint_stride0: int | None = None
    conv_state_stride0: int | None = None
    conv_accepted: tuple[int, ...] = ()


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
    candidate_block_indices: tuple[int, ...] | tuple[tuple[int, ...], ...] = ()
    block_table: tuple[int, ...] | tuple[tuple[int, ...], ...] = ()
    pa_block_size: int = QSLI_CANDIDATE_BLOCK_SIZE

    def block_ids(self, query):
        if self.candidate_block_indices and isinstance(self.candidate_block_indices[0], (tuple, list)):
            return self.candidate_block_indices[query]
        return self.candidate_block_indices

    def page_table(self, query):
        if self.block_table and isinstance(self.block_table[0], (tuple, list)):
            return self.block_table[query]
        return self.block_table


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


def flash_kda_lane_tasks(p, spec, *, datapath=None):
    chunks = tuple(ceil_div(length, KDA_CHUNK) for length in p.lengths)
    if spec.template_variant == "persistent":
        loads = [0] * spec.aic_budget
        for batch in sorted(range(len(chunks)), key=lambda i: (-chunks[i], i)):
            for head in range(p.value_heads):
                owner = min(range(len(loads)), key=loads.__getitem__)
                loads[owner] += chunks[batch]
                yield tuple((batch, head, chunk, owner, 0) for chunk in range(chunks[batch]))
        return
    _, group = flash_kda_schedule(p, spec.aic_budget, datapath=datapath)
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


def flash_kda_schedule(problem: FlashKdaProblem, core_count: int, *, datapath=None) -> tuple[str, int]:
    p = problem
    return resolve_flash_kda_route(
        p.lengths, p.value_heads, core_count, requested_template=p.requested_template,
        device_name=p.device_name, route_coefficients=p.route_coefficients, layout_kind=p.layout_kind,
        workspace_slots=p.workspace_slots, max_chunk_group=p.max_chunk_group, metadata_word_capacity=p.metadata_word_capacity,
        has_cu=p.has_cu, calibration_enabled=p.calibration_enabled, enable_grad=p.enable_grad, query_heads=p.query_heads,
        datapath=datapath,
    )


def _select_flash_route_parameters(*, datapath=None, route_profile: str | None = None, device_name: str = "") -> tuple[int, ...]:
    if route_profile is not None:
        if route_profile not in _FLASH_ROUTE_PROFILES:
            raise ModelingError(f"flash_kda_route_profile_unknown:{route_profile}")
        return _FLASH_ROUTE_PROFILES[route_profile]
    if datapath is not None:
        if datapath.arch_id not in DIRECT_MATRIX_ARCH_IDS:
            raise unmodeled(FLASH_KDA, "flash_kda_requires_arch35")
        hbm_r = float(getattr(datapath, "hbm_read_bytes_per_s", 0.0) or 0.0)
        if 0.0 < hbm_r < _HIGH_BW_HBM_THRESHOLD_BPS:
            return _ARCH35_BASE_ROUTE_PARAMETERS
        return _ARCH35_HIGH_BW_ROUTE_PARAMETERS
    if device_name:
        tok = device_name.strip().lower()
        if tok in _ARCH35_BASE_SKU_TAGS:
            return _ARCH35_BASE_ROUTE_PARAMETERS
        if tok in _ARCH35_HIGH_BW_SKU_TAGS:
            return _ARCH35_HIGH_BW_ROUTE_PARAMETERS
        return _ARCH35_BASE_ROUTE_PARAMETERS
    return _ARCH35_HIGH_BW_ROUTE_PARAMETERS


def resolve_flash_kda_route(
    lengths: Sequence[int], value_heads: int, core_count: int, *,
    requested_template: str = "auto", device_name: str = "", route_coefficients: Sequence[int] = (),
    layout_kind: int = 0, workspace_slots: int = KDA_DEFAULT_WORKSPACE_SLOTS, max_chunk_group: int | None = None,
    metadata_word_capacity: int | None = None, has_cu: bool = False, calibration_enabled: bool = True,
    enable_grad: bool = False, query_heads: int | None = None,
    datapath=None, route_profile: str | None = None,
) -> tuple[str, int]:
    """Replay the pinned launcher and host route comparison in integer units."""
    if (
        not lengths or min(lengths) < 1 or not 1 <= value_heads <= KDA_MAX_VALUE_HEADS
        or not 1 <= core_count <= KDA_AIC_CORE_LIMIT or layout_kind not in {0, 1}
        or workspace_slots < value_heads
    ):
        raise ModelingError("flash_kda_workspace_or_layout_invalid")
    if query_heads is None:
        query_heads = value_heads
    if query_heads < 1 or query_heads > value_heads or value_heads % query_heads != 0:
        raise ModelingError("flash_kda_workspace_or_layout_invalid")
    if max_chunk_group is None:
        short_training = (
            enable_grad and layout_kind == 0 and not has_cu and len(lengths) == 1
            and query_heads == value_heads == core_count == KDA_AIC_CORE_LIMIT
            and lengths[0] >= KDA_SHORT_TRAIN_MIN_LENGTH and lengths[0] % KDA_CHUNK == 0
        )
        max_chunk_group = KDA_SHORT_TRAIN_GROUP if short_training else KDA_MAX_CHUNK_GROUP
    if not 1 <= max_chunk_group <= KDA_MAX_CHUNK_GROUP:
        raise ModelingError("flash_kda_workspace_or_layout_invalid")
    if requested_template in {"persistent", "staged"}:
        return requested_template, max_chunk_group
    if requested_template != "auto":
        raise ModelingError("flash_kda_requested_template_invalid")
    if not calibration_enabled:
        return "staged", max_chunk_group
    chunks = tuple(ceil_div(length, KDA_CHUNK) for length in lengths)
    if (
        layout_kind == 1 and core_count == KDA_AIC_CORE_LIMIT
        and len(chunks) == 1 and KDA_AIC_CORE_LIMIT < value_heads
        and chunks[0] in KDA_CALIBRATED_CHUNK_RANGE
    ):
        max_chunk_group = min(max_chunk_group, KDA_CALIBRATED_GROUP_LIMIT)
    if has_cu and len(chunks) > 1 and max(chunks) <= 2:
        return "persistent", max_chunk_group
    params = tuple(route_coefficients) if route_coefficients else _select_flash_route_parameters(
        datapath=datapath, route_profile=route_profile, device_name=device_name,
    )
    start = layout_kind * _ROUTE_PARAMS_STRIDE
    if len(params) == _ROUTE_PARAMS_STRIDE and layout_kind == 0:
        selected_coeffs = params
    elif len(params) >= start + _ROUTE_PARAMS_STRIDE:
        selected_coeffs = params[start:start + _ROUTE_PARAMS_STRIDE]
    else:
        raise ModelingError("flash_kda_route_coefficients_invalid")
    route = _flash_route(
        lengths, chunks, value_heads, query_heads, core_count, workspace_slots,
        max_chunk_group, metadata_word_capacity, selected_coeffs,
    )
    return route, max_chunk_group


def _flash_route(lengths, chunks, heads, q_heads, cores, slots, max_group, metadata_capacity, coefficients):
    if len(coefficients) != _ROUTE_PARAMS_STRIDE:
        raise ModelingError("flash_kda_route_coefficients_invalid")
    (s0, s_parallel, s_peak, s_round, s_query, s_compute, s_mem, p0, p_peak, p_heads,
     p_query, p_compute, p_mem, schema, margin, _) = coefficients
    if schema not in _ROUTE_SCHEMAS or not 0 <= margin < PARTS_PER_MILLION or min(coefficients[:_ROUTE_SCHEMA]) < 0:
        raise ModelingError("flash_kda_route_coefficients_invalid")
    cyc, lpt, core_rows = [0] * cores, [0] * cores, [0] * cores
    head_count = heads * len(chunks)
    min_metadata = KDA_HEADER_WORDS + KDA_AIC_CORE_LIMIT + KDA_CHUNK_TASK_WORDS * head_count + len(chunks)
    if metadata_capacity is not None and min_metadata > metadata_capacity:
        return "staged"
    query_rows, total_rows = q_heads * sum(lengths), heads * sum(lengths)
    parallel = ceil_div(heads * sum(chunks), cores)
    if max(query_rows, total_rows) > _INT64_MAX or parallel > _MAX_SCALED_QUANTITY:
        raise ModelingError("flash_kda_route_cost_overflow")
    for batch, count in enumerate(chunks):
        for head in range(heads):
            cyc[(batch * heads + head) % cores] += count
    for batch in sorted(range(len(chunks)), key=lambda index: (-chunks[index], index)):
        for _ in range(heads):
            owner = min(range(cores), key=lpt.__getitem__)
            lpt[owner] += chunks[batch]
            core_rows[owner] += lengths[batch]
    staged_peak, persistent_peak = max(cyc), max(lpt)
    rounds = sum(1 for _ in flash_kda_rounds(chunks, heads, slots, max_group))
    if max(parallel, staged_peak, persistent_peak, rounds, head_count) > _MAX_SCALED_QUANTITY:
        raise ModelingError("flash_kda_route_cost_overflow")
    staged = (
        (s0 + s_parallel * parallel + s_peak * staged_peak + s_round * rounds) * ROUTE_COST_SCALE
        + s_query * query_rows + max(s_compute * parallel * ROUTE_COST_SCALE, s_mem * total_rows)
    )
    persistent = (p0 + p_peak * persistent_peak + p_heads * head_count) * ROUTE_COST_SCALE + p_query * query_rows
    previous = 0
    for rows in sorted(set(core_rows) - {0}):
        active = sum(length > previous for length in core_rows)
        mem = p_mem * active
        if mem > _INT64_MAX:
            raise ModelingError("flash_kda_route_cost_overflow")
        persistent += max(p_compute, mem) * (rows - previous)
        previous = rows
    if max(persistent, staged) > _INT64_MAX:
        raise ModelingError("flash_kda_route_cost_overflow")
    required_gain = staged * margin // PARTS_PER_MILLION
    return "persistent" if persistent < staged - required_gain else "staged"


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


def mla_projection_widths(p: MlaPrologProblem, aic: int, rows: int | None = None) -> tuple[int, int, int, bool, bool, int]:
    """Resolve native MLA prolog projection tile widths and WKB grouping by format and token profile."""
    mode = p.quant_mode
    nz_active = p.weight_format == "nz"
    bf16_cache = mode == QkvQuantMode.A16C16
    profile = next((size for size, lo, hi in MLA_TOKEN_PROFILES if lo <= p.tokens <= hi), 0)
    bm = MLA_K_TILE if profile == 0 else profile
    head_dim = p.q_nope + p.rope
    qp = p.heads * head_dim
    global_front = (
        nz_active and bf16_cache and MLA_SMALL_PARTIAL_MAX_ROWS < p.tokens <= MLA_GLOBAL_FRONT_MAX_ROWS
        and p.q_lora == MLA_NZ_PARTIAL_Q_LORA
    )
    if global_front:
        wqa_w = MLA_GLOBAL_FRONT_1536_WIDTH
    elif nz_active:
        if 0 < profile <= MLA_SMALL_PARTIAL_MAX_ROWS:
            if p.q_lora == MLA_NZ_PARTIAL_Q_LORA:
                wqa_w = MLA_FRONT_PARTIAL_1536_WIDTH
            elif p.q_lora in MLA_SMALL_NZ_Q_LORA_WIDTHS:
                wqa_w = MLA_FRONT_SMALL_NZ_WIDTH
            else:
                wqa_w = MLA_FRONT_MEDIUM_NZ_WIDTH
        else:
            wqa_w = MLA_K_TILE if profile == 0 else MLA_FRONT_MEDIUM_NZ_WIDTH
    else:
        wqa_w = MLA_COLUMN_TILE
    if nz_active and (bm <= MLA_SMALL_PARTIAL_MAX_ROWS
                      or (MLA_SMALL_PARTIAL_MAX_ROWS < bm <= MLA_GLOBAL_FRONT_MAX_ROWS and bf16_cache)):
        wkva_w = MLA_FRONT_SMALL_NZ_WIDTH
    else:
        wkva_w = MLA_FRONT_MEDIUM_NZ_WIDTH
    qb_grouped = (
        nz_active and bm <= MLA_QB_GROUP_MAX_BM and bf16_cache
        and p.q_lora in MLA_QB_GROUP_Q_LORA_WIDTHS and p.heads >= MLA_QB_GROUP_MIN_HEADS
    )
    if not nz_active:
        wqb_w = MLA_COLUMN_TILE
    elif qb_grouped:
        wqb_w = (MLA_QB_GROUP_BM16_HEADS if bm == MLA_QB_GROUP_MAX_BM else MLA_QB_GROUP_HEADS) * head_dim
    else:
        candidates = MLA_QB_WIDE_WIDTHS + MLA_QB_NZ_WIDTHS if (bf16_cache and p.heads >= MLA_WKB_STREAM_MIN_HEADS) else MLA_QB_NZ_WIDTHS
        wqb_w = next((w for w in candidates if qp % w == 0), MLA_COLUMN_TILE)
        if bf16_cache and 0 < profile <= MLA_SMALL_PARTIAL_MAX_ROWS and wqb_w > MLA_K_TILE and qp % MLA_K_TILE == 0 and qp // wqb_w < aic:
            wqb_w = MLA_K_TILE
    allow_grouped = nz_active and mode != QkvQuantMode.FP8_TENSOR
    m_tiles = ceil_div(p.tokens if rows is None else rows, bm)
    if MLA_SMALL_PARTIAL_MAX_ROWS < bm <= MLA_GLOBAL_FRONT_MAX_ROWS and allow_grouped:
        wkb_grouped = p.heads >= MLA_WKB_STREAM_MIN_HEADS
    elif bm <= MLA_SMALL_PARTIAL_MAX_ROWS and allow_grouped:
        wkb_grouped = m_tiles * p.heads >= 2 * aic
    else:
        wkb_grouped = False
    return wqa_w, wkva_w, wqb_w, qb_grouped, wkb_grouped, bm


def _buf(name, pool, size, slots=1, *, start=0, end=1, scale=0):
    return LiveBufferSpec(name, pool, size, start, end, owner_kind="aiv" if pool == "ub" else "aic",
                          scale_bytes=scale, slots=slots)


def _flash(p, aic, aiv, datapath, variant):
    route, _ = flash_kda_schedule(p, aic, datapath=datapath)
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
    if (
        p.tp not in MEGA_TP_DEGREES or p.sequence not in MEGA_SEQUENCE_LENGTHS
        or not 1 <= p.batch <= MEGA_MODEL_BATCH_LIMIT
        or p.num_heads != MEGA_MODEL_HEADS or p.hidden_size != MEGA_HIDDEN_SIZE
        or aic not in MEGA_AIC_QUOTAS or aiv < 2 * aic
    ):
        raise ModelingError("mega_kda_schedule_unsupported")
    _validate_kda_slots(p, p.batch * p.tp, p.sequence)
    heads = p.num_heads // p.tp
    rows = p.batch * p.tp * p.sequence
    workspace = mega_workspace_sizes(p)
    buffers = (
        _buf("state_ring", "ub", KDA_HEAD_DIM * KDA_STATE_HALF * FP32_BYTES, 2),
        _buf("replay", "ub", KDA_HEAD_DIM * BF16_BYTES, 2),
        _buf("a_l1", "l1", KDA_HEAD_DIM**2 * BF16_BYTES, 2),
        _buf("b_l1", "l1", KDA_HEAD_DIM**2 * BF16_BYTES, 2),
        _buf("l0a", "l0a", KDA_HEAD_DIM**2 * BF16_BYTES),
        _buf("l0b", "l0b", KDA_HEAD_DIM**2 * BF16_BYTES),
        _buf("l0c", "l0c", KDA_HEAD_DIM**2 * FP32_BYTES),
    )
    return (f"tp{p.tp}_s{p.sequence}", (p.sequence, KDA_HEAD_DIM, KDA_HEAD_DIM), buffers,
            sum(workspace.values()), (), (), rows * heads)


def mega_workspace_sizes(p):
    if p.num_heads < 1 or p.num_heads % p.tp != 0:
        raise ModelingError("mega_kda_geometry_invalid")
    heads = p.num_heads // p.tp
    rows = p.batch * p.tp * p.sequence
    head_bytes = rows * heads * KDA_HEAD_DIM * BF16_BYTES
    return {
        **dict.fromkeys(("Q", "K", "V", "decay", "output_gate"), head_bytes),
        "beta": rows * _aligned(heads, KDA_BETA_TILE) * BF16_BYTES,
        "fa": rows * KDA_HEAD_DIM * BF16_BYTES,
        "norm": KDA_HEAD_DIM**2 * FP32_BYTES,
    }


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


def mega_conv_stride_elems(p: MegaRecurrentKdaProblem) -> int:
    """Resolve physical per-slot element stride for Mega KDA causal_conv state."""
    if p.num_heads < 1 or p.num_heads % p.tp != 0:
        raise ModelingError("mega_kda_geometry_invalid")
    heads = p.num_heads // p.tp
    batch = p.batch * p.tp
    conv_width = KDA_CONV_STREAMS * heads * KDA_HEAD_DIM
    if p.conv_accepted and (len(p.conv_accepted) != batch or any(type(a) is not int or a < 0 for a in p.conv_accepted)):
        raise ModelingError("recurrent_kda_paged_state_geometry_invalid")
    max_hist_off = max((max(a - 1, 0) for a in p.conv_accepted), default=0)
    min_rows = max(KDA_CONV_HISTORY_KEEP + p.sequence, max_hist_off + KDA_CONV_TAPS - 1)
    min_stride = min_rows * conv_width
    if p.conv_state_stride0 is not None:
        if type(p.conv_state_stride0) is not int or p.conv_state_stride0 < min_stride:
            raise ModelingError("recurrent_kda_pool_stride_invalid")
        return p.conv_state_stride0
    return min_stride


def _validate_kda_slots(p, batch, required):
    slots = p.state_indices
    capacity = replay_pool_capacity(p.replay_capacity, required)
    if type(capacity) is not int or capacity < max(1, required) or (slots and (
        len(slots) != batch or len(set(slots)) != batch or any(type(i) is not int or i < 0 for i in slots)
    )):
        raise ModelingError("recurrent_kda_paged_state_geometry_invalid")
    if isinstance(p, CommitRecurrentKdaProblem):
        heads = p.heads
    else:
        mega_conv_stride_elems(p)
        heads = p.num_heads // p.tp
    replay_pool_strides(p, heads, capacity)


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
    if (
        p.tokens < 0
        or not (MLA_DIM_MIN <= p.dim <= MLA_DIM_MAX and p.dim % MLA_DIM_ALIGN == 0)
        or p.q_lora not in MLA_Q_LORA_WIDTHS
        or not 1 <= p.heads <= MLA_MAX_HEADS
        or p.kvr - p.rope != MLA_KV_LORA_DIM
        or (p.q_nope, p.rope) not in MLA_NOPE_ROPE_PAIRS
        or not (MLA_PAGE_SIZE_MIN <= p.cache_page_size <= MLA_PAGE_SIZE_MAX
                and p.cache_page_size % MLA_PAGE_SIZE_ALIGN == 0)
        or p.weight_format not in {"nd", "nz"}
        or p.cache_layout not in {"PA_BSND", "PA_NZ"}
    ):
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
    variant_name = QkvQuantMode.PA_BSND.value if p.mode == QkvQuantMode.PA_BSND else mode.value
    if p.tokens == 0:
        return variant_name, (1, MLA_COLUMN_TILE, MLA_K_TILE), (), 0, (), (), 0
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
    return variant_name, tile, buffers, workspace, (), (), p.tokens


def _attn_res(p, aic, aiv, datapath, variant):
    if p.hidden_size != ATTN_RES_HIDDEN_SIZE or dtype_name(p.dtype) != "bf16":
        raise ModelingError("attn_res_requires_fixed_bf16_width")
    if p.tokens < 0:
        raise ModelingError("attn_res_token_count_negative")
    if p.tokens == 0:
        return "inplace", (1, p.hidden_size, 1), (), 0, (), (), 0
    _, _, rows, depth = attn_res_tiling(
        p.tokens, aiv, min(ATTN_RES_UB_BYTES, datapath.usable_bytes("ub")), p.hidden_size,
    )
    buffers = (_buf("query_gamma_tile_and_stats", "ub", attn_res_ub_footprint(p.hidden_size, rows, depth)),)
    return "inplace", (rows, p.hidden_size, 1), buffers, 0, (), (), p.tokens


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
    num_tiles = max(1, ceil_div(p.candidates, QSLI_TILE_N))
    if (not 1 <= p.topk <= QSLI_MAX_TOPK or p.candidates < 0 or p.query_rows < 1
            or not 1 <= p.split_parts <= min(num_tiles, aic)
            or aic > QSLI_MAX_CUBE_WORKERS
            or type(p.pa_block_size) is not int or p.pa_block_size < QSLI_CANDIDATE_BLOCK_SIZE
            or p.pa_block_size % QSLI_CANDIDATE_BLOCK_SIZE != 0):
        raise ModelingError("qsli_problem_invalid")
    num_blocks = ceil_div(p.candidates, QSLI_CANDIDATE_BLOCK_SIZE)
    blocks_per_page = p.pa_block_size // QSLI_CANDIDATE_BLOCK_SIZE
    if p.candidate_block_indices:
        if not num_blocks:
            raise ModelingError("qsli_candidate_indices_invalid")
        batched = isinstance(p.candidate_block_indices[0], (tuple, list))
        if batched and len(p.candidate_block_indices) != p.query_rows:
            raise ModelingError("qsli_candidate_indices_invalid")
        for row in p.candidate_block_indices if batched else (p.candidate_block_indices,):
            if (not isinstance(row, (tuple, list)) or len(row) != num_blocks
                    or any(type(i) is not int or i < 0 for i in row)):
                raise ModelingError("qsli_candidate_indices_invalid")
    if p.block_table:
        batched_bt = isinstance(p.block_table[0], (tuple, list))
        if batched_bt and len(p.block_table) != p.query_rows:
            raise ModelingError("qsli_block_table_invalid")
        for q_idx in range(p.query_rows):
            table = p.page_table(q_idx)
            if not isinstance(table, (tuple, list)) or any(type(pg) is not int or pg < 0 for pg in table):
                raise ModelingError("qsli_block_table_invalid")
            if num_blocks:
                ids = p.block_ids(q_idx)
                max_blk = max(ids) if ids else num_blocks - 1
                if max_blk // blocks_per_page >= len(table):
                    raise ModelingError("qsli_block_table_invalid")
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


def mqsmla_tile_m(heads: int) -> int:
    return MQSMLA_SMALL_HEAD_BUCKET if heads <= MQSMLA_SMALL_HEAD_BUCKET else MQSMLA_HEADS


def mqsmla_tile_partitions(p: MixedQuantSparseMlaProblem) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    """Build independent ORI/CMP S2 tiles and partition strictly on tile boundaries."""
    tiles = (
        *(("ori", pos, min(MQSMLA_KV_TILE, p.n_ori - pos)) for pos in range(0, p.n_ori, MQSMLA_KV_TILE)),
        *(("cmp", pos, min(MQSMLA_KV_TILE, p.n_cmp - pos)) for pos in range(0, p.n_cmp, MQSMLA_KV_TILE)),
    )
    if p.split_parts > max(1, len(tiles)):
        raise ModelingError("mqsmla_split_geometry_invalid")
    if p.part_lengths:
        if (min(p.part_lengths) < 1 or len(p.part_lengths) != p.split_parts
                or sum(p.part_lengths) != p.n_ori + p.n_cmp):
            raise ModelingError("mqsmla_split_geometry_invalid")
        parts = []
        cursor = 0
        for target in p.part_lengths:
            part_tiles = []
            acc = 0
            while cursor < len(tiles) and acc < target:
                part_tiles.append(tiles[cursor])
                acc += tiles[cursor][2]
                cursor += 1
            if acc != target or not part_tiles:
                raise ModelingError("mqsmla_split_geometry_invalid")
            parts.append(tuple(part_tiles))
        if cursor != len(tiles):
            raise ModelingError("mqsmla_split_geometry_invalid")
        return tuple(parts)
    if not tiles:
        return ((),)
    return tuple(
        tiles[len(tiles) * part // p.split_parts:len(tiles) * (part + 1) // p.split_parts]
        for part in range(p.split_parts)
    )


def _mqsmla(p, aic, aiv, datapath, variant):
    if (not 1 <= p.heads <= MQSMLA_HEADS
            or (p.kv_heads, p.head_dim) != (MQSMLA_KV_HEADS, MQSMLA_HEAD_DIM)):
        raise ModelingError("mqsmla_fixed_head_geometry_required")
    if min(p.n_ori, p.n_cmp) < 0 or p.queries < 1 or not 1 <= p.split_parts <= aic:
        raise ModelingError("mqsmla_problem_invalid")
    tables = p.block_tables
    valid_sink = (p.sink_size in (0, p.heads)) if p.has_sink else (p.sink_size == 0)
    if (sum(map(bool, (p.ori_block_table, p.page_table, p.block_table))) > 1 or p.page_size < 1
            or not valid_sink
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
    mqsmla_tile_partitions(p)
    tile_m = mqsmla_tile_m(p.heads)
    tile_vec_m = tile_m // 2
    buffers = (
        _buf("kv_l1", "l1", MQSMLA_L1_KV_BYTES // MQSMLA_KV_DEPTH, MQSMLA_KV_DEPTH),
        _buf("q_l1", "l1", tile_m * MQSMLA_Q_L1_WIDTH * BF16_BYTES),
        _buf("p_l1", "l1", tile_m * MQSMLA_KV_TILE * BF16_BYTES),
        _buf("qk_l0c", "l0c", tile_m * MQSMLA_KV_TILE * FP32_BYTES),
        _buf("pv_l0c", "l0c", tile_m * KDA_CHUNK * FP32_BYTES, MQSMLA_PV_ACC_SLOTS),
        _buf("q_l0a", "l0a", tile_m * MQSMLA_KV_TILE * BF16_BYTES, MQSMLA_L0_SLOTS),
        _buf("kv_l0b", "l0b", MQSMLA_KV_TILE * KDA_CHUNK * BF16_BYTES, MQSMLA_L0_SLOTS),
        _buf("result_channels", "ub", MQSMLA_RESULT_ROW_FACTOR * tile_vec_m * MQSMLA_RESULT_HALF_D * FP32_BYTES,
             MQSMLA_RESULT_CHANNELS),
        _buf("qk_ub", "ub", tile_vec_m * MQSMLA_KV_TILE * FP32_BYTES),
        _buf("p_out", "ub", (tile_vec_m + 1) * MQSMLA_KV_TILE * BF16_BYTES),
        _buf("ch_batch_bytes", "ub", MQSMLA_RAW_GATHER_ROWS * KV_ROW_BYTES_ORI, MQSMLA_KV_DEPTH),
        _buf("v0_out", "ub", MQSMLA_V0_OUT_FRACTALS * MQSMLA_V0_OUT_PAD_ROWS * MQSMLA_V0_OUT_LANE * BF16_BYTES,
             MQSMLA_V0_OUT_DEPTH),
        _buf("softmax_state", "ub", MQSMLA_SOFTMAX_STATE_VECTORS * tile_vec_m * FP32_BYTES),
    )
    return ("dual_bank", (tile_m, MQSMLA_KV_TILE, p.head_dim), buffers,
            aic * MQSMLA_FD_BYTES_PER_CORE, (("scores", "l0c", "ub"), ("p", "ub", "l1")), (), p.queries)


_DESCRIBE = dict(zip(DSL_KERNEL_IDS, (_flash, _mega, _commit, _mla, _attn_res, _qsli, _mqsmla)))
_VECTOR_ONLY = {ATTN_RES, COMMIT_KDA}
_ZERO_AIC_DSL_KERNELS = frozenset({ATTN_RES})
_ALGORITHMS = {FLASH_KDA: "chunk_recurrence", MEGA_KDA: "chunk_recurrence",
               COMMIT_KDA: "chunk_recurrence", MQSMLA: "online_reduce"}


def describe_dsl_kernel(kernel_id, problem, datapath, *, aic_budget=None, aiv_budget=None,
                        memory_view=None, die=0, variant=None):
    name, p = dsl_problem(kernel_id, problem)
    if datapath.arch_id not in DIRECT_MATRIX_ARCH_IDS:
        raise unmodeled(name, "dsl_requires_arch35")
    aic = datapath.aic_count if aic_budget is None else aic_budget
    aiv = datapath.aiv_count if aiv_budget is None else aiv_budget
    min_aic = 0 if name in _ZERO_AIC_DSL_KERNELS else 1
    if not (min_aic <= aic <= datapath.aic_count and 1 <= aiv <= datapath.aiv_count):
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
        if name in {QSLI, FLASH_KDA, MQSMLA} and any(sum(c == core for c, _ in pairs) < 2 for core in range(active)):
            raise unmodeled(
                name, "qsli_requires_two_paired_subblocks" if name == QSLI else "dsl_requires_two_paired_subblocks",
            )
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
    spec = DslKernelSpec(
        name + "." + selected, "vector_only" if owner == "aiv" else "mix_cube_vector", algo, selected,
        dims, tuple((b.name, b.pool, b.slots) for b in buffers), routes, cache, workspace,
        name in {FLASH_KDA, QSLI, MQSMLA}, name == FLASH_KDA and bool(p.eager_d2h_bytes or p.eager_d2h_ms),
        False, aic, aiv, peaks.get("ub", 0), peaks.get("l1", 0), peaks.get("l0c", 0), _DSL_SOURCES[name],
    )
    return spec, solution


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
