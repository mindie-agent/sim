"""Shared contracts and physical lowering for grouped GEMM and MoE.

[Role] Normalize routing, model Blaze scheduling and bind conserved work.
[Invariants] Useful matrix work stays distinct from issued fractals. Zero-row
  experts retain their indices. Stage metadata contains only flat builtins.
[Timeouts & Locks] Finite arithmetic with owner-local counters; no I/O or locks.
[Side-Effect Policy] Build immutable plans for the caller's execution engine.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, Program, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.ring_buffer import RingBuffer
from ..contracts import OperatorPlanningContext

VLLM_CALLER_SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/ops/fused_moe/fused_moe.py",
        source_sha256="bd9b0b5612dee143e6aa583d0bc3b44c05cae3131d197ea14bd37298ce737750",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/ops/fused_moe/prepare_finalize.py",
        source_sha256="f42c57c2835494e886bf6deef054362ac1be9faaea7b1220a257210c7e996611",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/platform.py",
        source_sha256="75e8784f57fd5125ee29dd3407dab8b0cf43bf6532104fe5a89540bf7df41ddf",
    ),
)

FRACTAL_M_DIM = 16
FRACTAL_N_DIM = 16
FP32_FRACTAL_K = 8
HALF_PRECISION_FRACTAL_K = 16
BYTE_PRECISION_FRACTAL_K = 32
PACKED_PRECISION_FRACTAL_K = 64
BLAZE_BASE_M = 128
BLAZE_BASE_N = 256
BLAZE_SWIZZLE_WINDOW = 4
MX_SCALE_GROUP_SIZE = 32
INT32_HANDOFF_BYTES_PER_ELEM = 4
ROUTING_INDEX_BYTES = 4
ROUTING_WEIGHT_BYTES = 4
PER_TOKEN_SCALE_BYTES = 4
FUSED_MC2_SHARED_MODE = 3
SPARSE_GROUP_LIST_TYPE = 2
GROUP_COUNT_BYTES = 8

MX_DTYPES = frozenset({"mxfp8", "mxfp4"})
GEMM_MODES = frozenset({"dense_gemm", "grouped_gemm"})
ROUTED_MODES = frozenset({"mega_moe", "dual_batch_moe"})
SUPPORTED_MODES = GEMM_MODES | ROUTED_MODES | {"gmm_swiglu_quant", "dequant_swiglu_quant"}
MODE_ALIASES = {"grouped_moe": "grouped_gemm", "megamoe_arch35": "mega_moe"}
DTYPE_ALIASES = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "mx_fp8": "mxfp8", "mxfp8_e4m3": "mxfp8", "mx_fp4": "mxfp4",
    "mxfp4_e2m1": "mxfp4", "w8a8": "int8",
}
ACTIVATION_OPS = {
    "swiglu": 5, "swigluoai": 8, "situglu": 7, "situ": 7,
    "silu": 4, "gelu": 6, "gelu_tanh": 8, "geglu": 7,
}
QUANT_OPS_PER_ELEMENT = 4
GATED_ACTIVATIONS = frozenset({"swiglu", "swigluoai", "situglu", "situ", "geglu"})
ABI_MODES = {
    "aclnnGroupedMatmul": GEMM_MODES,
    "aclnnMoeGroupedMatmulSwigluQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluQuantV2": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluMxQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnDequantSwigluQuant": frozenset({"dequant_swiglu_quant"}),
    "aclnnMegaMoe": ROUTED_MODES,
}
AIC_ROLES = frozenset({"cube", "mte1", "mte2_aic", "fixpipe"})
AIV_ROLES = frozenset({"mte2_aiv", "mte3", "vector"})


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ModelingError(f"grouped_moe_invalid_{name}:{value!r}")
    return int(value)


def _mapping(value, name: str) -> dict:
    try:
        return dict(value)
    except (ValueError, TypeError) as exc:
        raise ModelingError(f"grouped_moe_invalid_{name}") from exc


def _dtype(value: str) -> str:
    key = str(value).lower()
    return DTYPE_ALIASES.get(key, key)


def element_bytes(elements: int, dtype: str) -> int:
    if dtype in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8"}:
        return ceil_div(elements, 2)
    return elements * (1 if dtype in {"mxfp8", "hif8"} else int(dtype_bytes(dtype)))


def tensor_bytes(rows: int, width: int, dtype: str, *, scaled: bool = False) -> int:
    scale = ceil_div(width, MX_SCALE_GROUP_SIZE) if dtype in MX_DTYPES else (
        PER_TOKEN_SCALE_BYTES if scaled else 0
    )
    return rows * (element_bytes(width, dtype) + scale)


def normalize_group_rows(
    group_rows: Iterable[int], *, cumulative: bool = False, expected_total: int | None = None,
) -> tuple[tuple[int, int], ...]:
    if type(cumulative) is not bool:
        raise ModelingError("grouped_moe_cumulative_flag_required")
    counts, offset = [], 0
    for value in group_rows:
        value = _integer(value, "group_rows")
        end = value if cumulative else offset + value
        if end < offset:
            raise ModelingError("grouped_moe_group_ends_not_monotone")
        counts.append((offset, end - offset))
        offset = end
    if expected_total is not None and offset != _integer(expected_total, "expected_total"):
        raise ModelingError(f"grouped_moe_group_total_mismatch:{offset}:{expected_total}")
    return tuple(counts)


def fractal_k0(compute_dtype: str) -> int:
    dtype = _dtype(compute_dtype)
    if dtype == "fp32":
        return FP32_FRACTAL_K
    if dtype in {"bf16", "fp16"}:
        return HALF_PRECISION_FRACTAL_K
    if dtype in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "hif8", "w4a8"}:
        return BYTE_PRECISION_FRACTAL_K
    if dtype in {"mxfp4", "fp4", "fp4_e2m1", "int4"}:
        return PACKED_PRECISION_FRACTAL_K
    raise ModelingError(f"grouped_moe_unsupported_compute_dtype:{compute_dtype}")


def fractal_issues(m: int, n: int, k: int, compute_dtype: str) -> tuple[int, int]:
    m, n, k = (_integer(dim, "fractal_dimension") for dim in (m, n, k))
    k0 = fractal_k0(compute_dtype)
    issues = ceil_div(m, FRACTAL_M_DIM) * ceil_div(n, FRACTAL_N_DIM) * ceil_div(k, k0)
    return issues, 2 * FRACTAL_M_DIM * FRACTAL_N_DIM * k0 * issues


@dataclass(frozen=True, slots=True)
class BlazeGroupSlice:
    expert: int
    offset: int
    rows: int
    m_tiles: int
    n_tiles: int
    logical_tiles: int
    tail_tiles: int
    total_tiles: int
    start_core: int
    end_block: int


@dataclass(frozen=True, slots=True)
class BlazeGroupSchedule:
    groups: tuple[BlazeGroupSlice, ...]
    owner_tiles: tuple[int, ...]
    owner_useful_flops: tuple[int, ...]
    owner_issued_fractals: tuple[int, ...]
    owner_issued_flops: tuple[int, ...]
    owner_input_rows: tuple[int, ...]
    owner_weight_columns: tuple[int, ...]
    owner_output_elements: tuple[int, ...]

    @property
    def nonempty_owners(self) -> tuple[int, ...]:
        return tuple(owner for owner, tiles in enumerate(self.owner_tiles) if tiles)

    @property
    def max_waves(self) -> int:
        return max(self.owner_tiles)

    @property
    def wave_tail_imbalance(self) -> int:
        return self.max_waves - min(self.owner_tiles)

    @property
    def useful_flops(self) -> int:
        return sum(self.owner_useful_flops)

    @property
    def issued_fractals(self) -> int:
        return sum(self.owner_issued_fractals)

    @property
    def issued_flops(self) -> int:
        return sum(self.owner_issued_flops)


def _blaze_tile(index: int, m_tiles: int, n_tiles: int) -> tuple[int, int]:
    window = min(m_tiles, BLAZE_SWIZZLE_WINDOW)
    main_windows = m_tiles // window - 1
    row = min(index // (window * n_tiles), main_windows)
    rows = window if row < main_windows else m_tiles - row * window
    col, offset = divmod(index - row * window * n_tiles, rows)
    return row * window + offset, n_tiles - 1 - col if row % 2 else col


def blaze_tasks(group: BlazeGroupSlice, n, base_m, base_n, m_tail_cnt, n_tail_cnt, n_align=1):
    tail_base = group.logical_tiles - group.tail_tiles
    split_count = m_tail_cnt * n_tail_cnt
    for task in range(group.total_tiles):
        in_tail = task >= tail_base
        logical, split = divmod(task - tail_base, split_count) if in_tail else (task, 0)
        if in_tail:
            logical += tail_base
        m_idx, n_idx = _blaze_tile(logical, group.m_tiles, group.n_tiles)
        m = min(base_m, group.rows - m_idx * base_m)
        width = min(base_n, n - n_idx * base_n)
        if in_tail:
            m_split = ceil_div(m, m_tail_cnt)
            n_split = ceil_div(width, n_tail_cnt * n_align) * n_align
            n_sub, m_sub = divmod(split, ceil_div(m, m_split))
            m = min(m_split, m - m_sub * m_split)
            width = min(n_split, width - n_sub * n_split)
        if m > 0 and width > 0:
            yield task, m, width


def schedule_blaze_groups(
    group_rows: Iterable[int], n_dim: int, k_dim: int, *, aic_cores: int,
    base_m: int = BLAZE_BASE_M, base_n: int = BLAZE_BASE_N, compute_dtype: str = "bf16",
    tail_split: bool = False, m_tail_cnt: int = 1, n_tail_cnt: int = 1,
    start_core: int = 0, weight_col_alignment: int = 1, n_tail_alignment: int = 1,
) -> BlazeGroupSchedule:
    aic_cores = _integer(aic_cores, "aic_cores", 1)
    weight_col_alignment = _integer(weight_col_alignment, "weight_col_alignment", 1)
    n_tail_alignment = _integer(n_tail_alignment, "n_tail_alignment", 1)
    n_dim, k_dim = (_integer(dim, "dimension", 1) for dim in (n_dim, k_dim))
    base_m, base_n = (_integer(dim, "base_tile", 1) for dim in (base_m, base_n))
    m_tail_cnt, n_tail_cnt = (
        _integer(count, "tail_split_count", 1) for count in (m_tail_cnt, n_tail_cnt)
    )
    if _integer(start_core, "start_core") >= aic_cores:
        raise ModelingError("grouped_moe_start_core_out_of_range")
    k0 = fractal_k0(compute_dtype)
    flops_per_issue = 2 * FRACTAL_M_DIM * FRACTAL_N_DIM * k0
    k_fractals = ceil_div(k_dim, k0)
    groups = []
    tiles, useful, issues, a_rows, b_cols, elements = (
        [0] * aic_cores for _ in ("tiles", "useful", "issues", "a", "b", "c")
    )
    for expert, (offset, rows) in enumerate(normalize_group_rows(group_rows)):
        m_tiles = ceil_div(rows, base_m)
        n_tiles = ceil_div(n_dim, base_n) if rows else 0
        logical = m_tiles * n_tiles
        tail = logical % aic_cores
        subdivided = tail if tail_split and m_tail_cnt * n_tail_cnt > 1 else 0
        total = logical + subdivided * (m_tail_cnt * n_tail_cnt - 1)
        group = BlazeGroupSlice(
            expert, offset, rows, m_tiles, n_tiles, logical, tail, total,
            start_core, start_core + total,
        )
        groups.append(group)
        for task, m, n in blaze_tasks(
            replace(group, tail_tiles=subdivided), n_dim, base_m, base_n, m_tail_cnt, n_tail_cnt, n_tail_alignment,
        ):
            owner = (start_core + task) % aic_cores
            tiles[owner] += 1
            elements[owner] += m * n
            useful[owner] += 2 * m * n * k_dim
            issues[owner] += ceil_div(m, FRACTAL_M_DIM) * ceil_div(n, FRACTAL_N_DIM) * k_fractals
            a_rows[owner] += m
            b_cols[owner] += ceil_div(n, weight_col_alignment) * weight_col_alignment
        start_core = group.end_block % aic_cores
    return BlazeGroupSchedule(
        tuple(groups), tuple(tiles), tuple(useful), tuple(issues),
        tuple(count * flops_per_issue for count in issues),
        tuple(a_rows), tuple(b_cols), tuple(elements),
    )


@dataclass(frozen=True, slots=True)
class MoECohortSplit:
    total_tokens: int
    cohort0_row_ids: tuple[int, ...]
    cohort1_row_ids: tuple[int, ...]
    cohort0_expert_rows: tuple[int, ...]
    cohort1_expert_rows: tuple[int, ...]
    top_k: int
    ep: int = 1

    def __post_init__(self) -> None:
        _integer(self.total_tokens, "total_tokens")
        _integer(self.top_k, "top_k", 1)
        _integer(self.ep, "ep", 1)
        for attr in ("cohort0_row_ids", "cohort1_row_ids", "cohort0_expert_rows", "cohort1_expert_rows"):
            object.__setattr__(self, attr, tuple(_integer(x, attr) for x in getattr(self, attr)))
        row_ids = self.cohort0_row_ids + self.cohort1_row_ids
        if (len(row_ids) != self.total_tokens or len(set(row_ids)) != self.total_tokens
                or (row_ids and max(row_ids) >= self.total_tokens)):
            raise ModelingError("grouped_moe_dbo_row_ownership")
        if len(self.cohort0_expert_rows) != len(self.cohort1_expert_rows):
            raise ModelingError("grouped_moe_dbo_expert_count_mismatch")
        if self.ep == 1 and any(
            sum(rows) > len(ids) * self.top_k for ids, rows in (
                (self.cohort0_row_ids, self.cohort0_expert_rows),
                (self.cohort1_row_ids, self.cohort1_expert_rows),
            )
        ):
            raise ModelingError("grouped_moe_dbo_routing_capacity")

    def validate_experts(self, group_rows: tuple[int, ...]) -> None:
        combined = tuple(a + b for a, b in zip(self.cohort0_expert_rows, self.cohort1_expert_rows))
        if combined != group_rows:
            raise ModelingError("grouped_moe_dbo_expert_rows_not_conserved")


@dataclass(frozen=True, slots=True)
class GroupedMoeContract:
    mode: str
    abi: str | None
    m: int
    n: int
    k: int
    group_rows: tuple[int, ...]
    allocated_rows: int
    tokens: int
    dtype_a: str
    dtype_b: str
    compute_dtype: str
    output_dtype: str
    quant_dtype: str
    handoff_dtype: str
    layout_a: str
    layout_b: str
    transpose_weight: bool
    activation: str
    base_m: int
    base_n: int
    tail_split: bool
    m_tail_cnt: int
    n_tail_cnt: int
    top_k: int
    ep: int
    enable_dbo: bool
    dbo_split: MoECohortSplit | None
    enable_fused_mc2: int
    shared_intermediate: int
    wave_rows: int
    transport_bytes: tuple[int | None, int | None]
    cohort_transport_bytes: tuple[tuple[int | None, int | None], ...] = ()

    @property
    def intermediate(self) -> int:
        return self.n // self.activation_divisor

    @property
    def activation_divisor(self) -> int:
        return 2 if self.activation in GATED_ACTIVATIONS else 1

    @property
    def padded_weight(self) -> bool:
        return self.layout_b in {"NZ", "FRACTAL_NZ"}


def _contract_rows(fields, mode, m, expected_total):
    rows = fields.get("matrix_group_rows", fields.get("group_rows", fields.get("expert_rows")))
    if rows is None and "group_list" in fields:
        kind = fields.get("group_list_type", 0)
        if kind == SPARSE_GROUP_LIST_TYPE:
            raise ModelingError("grouped_moe_sparse_group_list_requires_dense_histogram")
        if kind not in {0, 1}:
            raise ModelingError(f"grouped_moe_unsupported_group_list_type:{kind}")
        rows = tuple(count for _, count in normalize_group_rows(
            fields["group_list"], cumulative=fields.get("cumulative", kind == 0),
            expected_total=expected_total,
        ))
    if rows is None:
        if mode in {"dense_gemm", "dequant_swiglu_quant"}:
            rows = (m,)
        else:
            raise ModelingError("grouped_moe_missing_expert_histogram")
    result = tuple(count for _, count in normalize_group_rows(rows, expected_total=expected_total))
    if not result or (mode == "dense_gemm" and len(result) != 1):
        raise ModelingError("grouped_moe_invalid_expert_count")
    return result


def validate_contract(op: OpSpec, ctx: OperatorPlanningContext) -> GroupedMoeContract:
    fields = dict(op.metadata)
    for name in ("megamoe_geometry", "grouped_moe_geometry"):
        if name in fields:
            fields.update(_mapping(fields[name], name))
    source_mode = fields.get("mode", op.recipe or op.family)
    mode = MODE_ALIASES.get(source_mode, source_mode)
    if mode not in SUPPORTED_MODES and "mode" not in fields:
        mode = MODE_ALIASES.get(op.family, op.family)
    if mode not in SUPPORTED_MODES:
        raise capability_gap(op, ctx, f"unsupported_mode:{source_mode}")
    abi = ctx.requested_abi or fields.get("requested_abi") or fields.get("abi")
    if abi and mode not in ABI_MODES.get(abi, ()):
        raise capability_gap(op, ctx, f"unsupported_abi:{abi}:{mode}")

    shape = op.local_shape
    routed = mode in ROUTED_MODES
    m = fields.get("m", fields.get("M", shape[0] if shape else 0))
    k = fields.get("k", fields.get("K", fields.get("hidden", shape[1] if routed and len(shape) >= 2 else (
        shape[-1] if len(shape) > 2 else 0
    ))))
    n = fields.get("n", fields.get("N", shape[1] if len(shape) >= 2 and not routed else 0))
    activation = str(fields.get("activation", "swiglu")).lower()
    if activation not in ACTIVATION_OPS:
        raise ModelingError(f"grouped_moe_unsupported_activation:{activation}")
    divisor = 2 if activation in GATED_ACTIVATIONS else 1
    intermediate = fields.get("f", fields.get("F", fields.get("moe_intermediate")))
    if intermediate is not None:
        width = divisor * _integer(intermediate, "intermediate", 1)
        n = fields.get("n", fields.get("N", width))
        if n != width:
            raise ModelingError("grouped_moe_intermediate_dimension_mismatch")
    n = _integer(n, "n", 1)
    k = _integer(k, "k", 0 if mode == "dequant_swiglu_quant" else 1)
    if mode not in GEMM_MODES and n % divisor:
        raise ModelingError("grouped_moe_gated_width_alignment")
    m = _integer(m, "m")
    valid = fields.get("valid_rows", fields.get("routed_rows", None if routed else m))
    rows = _contract_rows(fields, mode, m, valid)
    allocated = _integer(fields.get("allocated_rows", sum(rows) if routed else m), "allocated_rows")
    if allocated < sum(rows):
        raise ModelingError("grouped_moe_valid_rows_exceed_allocation")
    if _integer(fields.get("scale_group_size", MX_SCALE_GROUP_SIZE), "scale_group_size", 1) != MX_SCALE_GROUP_SIZE:
        raise ModelingError("grouped_moe_unsupported_scale_group")

    operand_default = fields.get("compute_dtype", fields.get("compute_format", op.dtype or "bf16"))
    dtype_a = _dtype(fields.get("dtype_a", fields.get("input_dtype", operand_default)))
    if mode == "dequant_swiglu_quant" and "dtype_a" not in fields and "input_dtype" not in fields:
        input_default = _dtype(op.dtype or "int32")
        dtype_a = _dtype(fields.get("handoff_dtype",
            input_default if input_default in {"int32", "bf16", "fp16", "fp32"} else "int32"))
    dtype_b = _dtype(fields.get("dtype_b", fields.get("weight_dtype", "int4" if dtype_a == "w4a8" else dtype_a)))
    if dtype_a == "w4a8":
        dtype_a = "int8"
    compute_dtype = _dtype(fields.get("compute_dtype", fields.get("compute_format", dtype_a)))
    if mode != "dequant_swiglu_quant":
        fractal_k0(compute_dtype)
    output_dtype = _dtype(fields.get("output_dtype", fields.get("output_format", op.dtype or "bf16")))
    quant_default = dtype_a if dtype_a in MX_DTYPES or (routed and dtype_a in {"bf16", "fp16"}) else "int8"
    quant_dtype = _dtype(fields.get("quant_dtype", fields.get("quant_output_dtype", quant_default)))
    default_handoff = "int32" if compute_dtype in {"int8", "w4a8"} else output_dtype
    handoff_dtype = _dtype(fields.get("handoff_dtype", dtype_a if mode == "dequant_swiglu_quant" else default_handoff))
    for dtype in (dtype_a, dtype_b, output_dtype, quant_dtype, handoff_dtype):
        element_bytes(1, dtype)
    layout_a = str(fields.get("layout_a", "ND")).upper()
    layout_b = str(fields.get("layout_b", fields.get("weight_layout", "ND"))).upper()
    if layout_a not in {"ND", "CONTIGUOUS"} or layout_b not in {"ND", "NZ", "FRACTAL_NZ", "CONTIGUOUS"}:
        raise ModelingError(f"grouped_moe_unsupported_layout:{layout_a}:{layout_b}")
    top_k = _integer(fields.get("top_k", 1), "top_k", 1)
    ep = _integer(fields.get("ep", 1), "ep", 1)
    tokens = _integer(fields.get("tokens", m), "tokens")
    if routed and ep == 1 and sum(rows) > tokens * top_k:
        raise ModelingError("grouped_moe_routing_capacity")
    enable_dbo = fields.get("enable_dbo", False)
    tail_split = fields.get("tail_split", False)
    transpose = fields.get("transpose_weight", fields.get("trans_b", False))
    if any(type(flag) is not bool for flag in (enable_dbo, tail_split, transpose)):
        raise ModelingError("grouped_moe_boolean_flag_required")
    split = _cohort_split(fields, op, ctx, rows, tokens, top_k, ep, enable_dbo, mode)
    mc2 = _integer(fields.get("enable_fused_mc2", 2 if routed else 0), "enable_fused_mc2")
    shared = _integer(fields.get("shared_intermediate", 0), "shared_intermediate")
    if mc2 not in {0, 2, FUSED_MC2_SHARED_MODE} or (mc2 == FUSED_MC2_SHARED_MODE) != (shared > 0):
        raise ModelingError("grouped_moe_invalid_shared_mc2_contract")
    transport_fields = ("dispatch_link_bytes", "combine_link_bytes")
    transfer = tuple(_integer(fields[key], key) if key in fields else None for key in transport_fields)
    cohort_transfer = tuple(
        tuple(_integer(value, "cohort_link_bytes") for value in pair)
        for pair in fields.get("cohort_link_bytes", ())
    )
    if cohort_transfer:
        if split is None or len(cohort_transfer) != 2 or any(len(pair) != 2 for pair in cohort_transfer):
            raise ModelingError("grouped_moe_invalid_cohort_link_bytes")
        if any(amount is not None and sum(parts) != amount for amount, parts in zip(transfer, zip(*cohort_transfer))):
            raise ModelingError("grouped_moe_dbo_link_work_not_conserved")
    if (routed and ep > 1 and mc2 and sum(rows) and not cohort_transfer
            and None in transfer and not callable(fields.get("cohort_transport_builder"))):
        raise capability_gap(op, ctx, "explicit_dispatch_and_combine_transport")
    return GroupedMoeContract(
        mode, abi, sum(rows), n, k, rows, allocated, tokens, dtype_a, dtype_b, compute_dtype,
        output_dtype, quant_dtype, handoff_dtype, layout_a, layout_b, transpose, activation,
        _integer(fields.get("base_m", BLAZE_BASE_M), "base_m", 1),
        _integer(fields.get("base_n", BLAZE_BASE_N), "base_n", 1), tail_split,
        _integer(fields.get("m_tail_cnt", 1), "m_tail_cnt", 1),
        _integer(fields.get("n_tail_cnt", 1), "n_tail_cnt", 1),
        top_k, ep, enable_dbo, split, mc2, shared,
        _integer(fields.get("wave_rows", 0), "wave_rows"), transfer, cohort_transfer,
    )


def _cohort_split(fields, op, ctx, rows, tokens, top_k, ep, enabled, mode):
    if not enabled:
        if mode == "dual_batch_moe":
            raise capability_gap(op, ctx, "dbo_requires_explicit_enablement")
        return None
    if mode not in ROUTED_MODES or "dbo_split" not in fields:
        raise ModelingError("grouped_moe_dbo_requires_routed_mode_and_split")
    split = fields["dbo_split"]
    if not isinstance(split, MoECohortSplit):
        values = _mapping(split, "dbo_split")
        try:
            split = MoECohortSplit(**{"total_tokens": tokens, "top_k": top_k, "ep": ep, **values})
        except TypeError as exc:
            raise ModelingError("grouped_moe_invalid_dbo_split") from exc
    if (split.total_tokens, split.top_k, split.ep) != (tokens, top_k, ep):
        raise ModelingError("grouped_moe_dbo_parent_mismatch")
    split.validate_experts(rows)
    return split


def capability_gap(op: OpSpec, ctx: OperatorPlanningContext, reason: str) -> UnmodeledCapabilityError:
    arch = str(ctx.compute_arch.arch_id)
    return UnmodeledCapabilityError(
        capability="grouped_moe", entry="resolve_operator_plan", code=f"grouped_moe:{reason}",
        message=f"Grouped MoE {op.op_id} on {arch} requires {reason}",
        hints=(build_unmodeled_onboarding_hint(
            observed_kernel_name=ctx.requested_abi or op.recipe or op.family,
            observed_op_type=op.family,
            observed_abi_summary={"architecture": arch, "reason": reason, "op_id": op.op_id},
        ),),
    )


@dataclass(frozen=True, slots=True)
class StageDemand:
    name: str
    after: tuple[str, ...] = ()
    work: Work = field(default_factory=Work)
    kind: str = ""
    owner: int = 0
    metadata: tuple[tuple[str, object], ...] = ()
    ub_acquire: tuple[str, int] | None = None
    ub_release: tuple[str, int] | None = None


def _slot(context: RecipeContext, owner: int, spec: tuple[str, int] | None):
    if spec is None:
        return ()
    label, size = spec
    units = context.pool_units.get("ub", context.effective_aiv_cores)
    owners = context.datapath.aiv_count if context.datapath else context.resource_units.get("vector", units)
    mapped = unit_group(owner, owners, units)
    return tuple(
        _ring_slot(context.pool_names["ub"], label, size, index) for index in mapped
    )


def _ring_slot(pool, label, size, owner):
    ring = RingBuffer(owner=owner, pool=pool, allocation=f"{label}:{owner}", slot_bytes=size, depth=1)
    ring.initial_credit(0)
    return ring.to_stages(release_slots_at_end=False)[0].acquire[0]


@dataclass(frozen=True, slots=True)
class GroupedMoeProgram:
    demands: tuple[StageDemand, ...]
    aic_count: int
    aiv_count: int

    def __call__(self, context: RecipeContext) -> Program:
        stages = []
        for demand in self.demands:
            work = context.bind(demand.work)
            placement = {}
            if demand.kind:
                roles = AIC_ROLES if demand.kind == "aic" else AIV_ROLES
                owners = self.aic_count if demand.kind == "aic" else self.aiv_count
                placement = {
                    name: units for name, units in context.owned(demand.owner, owners, roles).items()
                    if name in work
                }
            stages.append(Stage(
                demand.name, work, after=demand.after, placement=placement, coupled=bool(work),
                metadata=dict(demand.metadata),
                acquire=_slot(context, demand.owner, demand.ub_acquire),
                free=_slot(context, demand.owner, demand.ub_release),
            ))
        return KernelProgram(tuple(stages))


def make_plan(
    implementation: str, ctx: OperatorPlanningContext, stages: Iterable[StageDemand],
    sources: tuple[SourceRef, ...], *, metadata: Mapping | None = None,
) -> OperatorPlan:
    stages = tuple(stages)
    work = Work()
    phases = []
    for stage in stages:
        work += stage.work
        annotations = dict(stage.metadata)
        items = tuple(
            WorkItem(role, "op" if role == "vector" else "bytes", amount)
            for role, amount in stage.work.resources.items() if amount
        ) + tuple(
            WorkItem("cube", "flop", amount, dtype)
            for dtype, amount in stage.work.matrix.items() if amount
        )
        phases.append(PipelinePhase(stage.name, stage.after, items,
                                    cohort=annotations.get("cohort_id"), slot=annotations.get("slot")))
    return OperatorPlan(
        work, GroupedMoeProgram(stages, ctx.datapath.aic_count, ctx.datapath.aiv_count),
        dict(metadata or ()),
        pipeline_spec=FusedPipelineSpec("grouped_moe", implementation, tuple(phases),
                                       (*sources, *VLLM_CALLER_SOURCES)),
    )


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    absorbed = tuple(dict(op.metadata).get("absorbed_source_op_ids", ()))
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id, *absorbed),
        absorbed, spec.phases, spec.sources, spec.repeats,
    )


def matrix_schedule(contract, ctx, rows, n, k, *, start_core=0, n_tail_alignment=1):
    return schedule_blaze_groups(
        rows, n, k, aic_cores=ctx.effective_aic_cores, base_m=contract.base_m,
        base_n=contract.base_n, compute_dtype=contract.compute_dtype,
        tail_split=contract.tail_split, m_tail_cnt=contract.m_tail_cnt,
        n_tail_cnt=contract.n_tail_cnt, start_core=start_core, n_tail_alignment=n_tail_alignment,
        weight_col_alignment=(FRACTAL_N_DIM if contract.transpose_weight else fractal_k0(contract.dtype_b))
        if contract.padded_weight else 1,
    )


def matrix_stages(
    name, after, contract, schedule, k, output_dtype, *, input_dtype=None, input_scaled=False, metadata=(),
):
    dtype = input_dtype or contract.dtype_a
    stages, ends = [], []
    k_alignment = (fractal_k0(contract.dtype_b) if contract.transpose_weight else FRACTAL_M_DIM) if (
        contract.padded_weight
    ) else 1
    weight_k = ceil_div(k, k_alignment) * k_alignment
    for owner in schedule.nonempty_owners:
        a_rows, b_cols = schedule.owner_input_rows[owner], schedule.owner_weight_columns[owner]
        a = tensor_bytes(a_rows, k, dtype, scaled=input_scaled)
        b = tensor_bytes(b_cols, weight_k, contract.dtype_b)
        out = element_bytes(schedule.owner_output_elements[owner], output_dtype)
        read, cube, fix = (f"{name}:{owner}:{step}" for step in ("read", "cube", "fixpipe"))
        stages.extend((
            StageDemand(read, after, Work(resources={"hbm": a+b, "l2": a+b, "mte2_aic": a+b}),
                        "aic", owner, metadata),
            StageDemand(cube, (read,), Work(
                {contract.compute_dtype: schedule.owner_issued_flops[owner]},
                {"mte1": a+b},
            ), "aic", owner, metadata),
            StageDemand(fix, (cube,), Work(resources={"hbm": out, "l2": out, "fixpipe": out}), "aic", owner, metadata),
        ))
        ends.append(fix)
    stages.append(StageDemand(name, tuple(ends) or after, metadata=(*metadata,
        ("useful_flops", schedule.useful_flops), ("issued_fractals", schedule.issued_fractals),
        ("max_waves", schedule.max_waves), ("wave_tail_imbalance", schedule.wave_tail_imbalance),
    )))
    return stages


def vector_stages(name, after, ctx, rows, *, read, compute, write, metadata=()):
    cores = ctx.effective_aiv_cores
    quotient, tail = divmod(sum(rows), cores)
    stages = []
    for owner in range(min(cores, sum(rows))):
        count = quotient + (owner < tail)
        base = f"{name}:{owner}"
        stages.extend((
            StageDemand(base + ":read", after, Work(resources={
                "hbm": count * read, "l2": count * read, "mte2_aiv": count * read,
            }), "aiv", owner, metadata),
            StageDemand(base + ":vector", (base + ":read",), Work(resources={"vector": count * compute}),
                        "aiv", owner, metadata),
            StageDemand(base + ":store", (base + ":vector",), Work(resources={
                "hbm": count * write, "l2": count * write, "mte3": count * write,
            }), "aiv", owner, metadata),
        ))
    stages.append(StageDemand(name, tuple(stage.name for stage in stages if stage.name.endswith(":store")) or after, metadata=metadata))
    return stages


def activation_stages(name, after, ctx, contract, rows, metadata=()):
    width = contract.intermediate
    quantized = contract.quant_dtype not in {"bf16", "fp16"}
    scale_read = PER_TOKEN_SCALE_BYTES if contract.handoff_dtype == "int32" else 0
    return vector_stages(
        name, after, ctx, rows, read=element_bytes(contract.n, contract.handoff_dtype) + scale_read,
        compute=width * (ACTIVATION_OPS[contract.activation] + (QUANT_OPS_PER_ELEMENT if quantized else 0)),
        write=tensor_bytes(1, width, contract.quant_dtype, scaled=quantized), metadata=metadata,
    )


TransportBuilder = Callable[[str, tuple[str, ...], int, tuple[tuple[str, object], ...]], Iterable[StageDemand]]


@dataclass(frozen=True, slots=True)
class DualBatchMoEPipeline:
    split: MoECohortSplit

    def stages(self, contract, ctx, emit, *, transport_builder=None):
        self.split.validate_experts(contract.group_rows)
        stages, combines = [], []
        preceding_dispatch = ()
        for cohort, (ids, rows) in enumerate((
            (self.split.cohort0_row_ids, self.split.cohort0_expert_rows),
            (self.split.cohort1_row_ids, self.split.cohort1_expert_rows),
        )):
            if not ids or not any(rows):
                if any(rows) or (contract.cohort_transport_bytes and any(contract.cohort_transport_bytes[cohort])):
                    raise ModelingError("grouped_moe_empty_dbo_cohort_has_work")
                continue
            prefix = f"c{cohort}_"
            metadata = (("cohort_id", cohort), ("slot", cohort), ("row_ids", ids),
                        ("row_start", min(ids)), ("row_end", max(ids) + 1), ("expert_rows", rows))
            transfers = contract.cohort_transport_bytes[cohort] if contract.cohort_transport_bytes else (
                tuple(_cohort_bytes(amount, sum(rows), contract.m) for amount in contract.transport_bytes)
            )
            sub = replace(contract, m=sum(rows), group_rows=rows, tokens=len(ids), transport_bytes=transfers)
            branch = emit(sub, prefix, preceding_dispatch, metadata, transport_builder)
            label = prefix + "routing"
            capacity = ctx.datapath.usable_bytes("ub")
            slot = (label, min(capacity // 2, ROUTING_INDEX_BYTES * self.split.top_k))
            if not slot[1]:
                raise ModelingError("grouped_moe_dbo_ub_slot_empty")
            branch[0] = replace(branch[0], ub_acquire=slot)
            branch[-1] = replace(branch[-1], ub_release=slot)
            stages.extend(branch)
            preceding_dispatch = (prefix + "dispatch",)
            combines.append(prefix + "combine")
        # Combine writes to original row positions, so restore is a completion fence.
        stages.append(StageDemand("join_restore", tuple(combines)))
        return stages


def _cohort_bytes(amount, rows, total):
    if amount is None or total == 0:
        return amount
    per_row, remainder = divmod(amount, total)
    if remainder:
        raise ModelingError("grouped_moe_dbo_requires_exact_cohort_link_bytes")
    return per_row * rows


def routing_stages(name, after, contract, metadata, builder: TransportBuilder | None):
    dispatch = name.endswith("dispatch")
    payload_dtype = contract.dtype_a if dispatch else contract.output_dtype
    payload = tensor_bytes(contract.m, contract.k, payload_dtype, scaled=payload_dtype == "int8")
    index_bytes = ROUTING_INDEX_BYTES if dispatch else ROUTING_WEIGHT_BYTES
    tokens = contract.tokens * element_bytes(contract.k, contract.output_dtype)
    read, write = (tokens, payload) if dispatch else (payload + tokens * bool(contract.shared_intermediate), tokens)
    read += contract.tokens * contract.top_k * index_bytes
    read += len(contract.group_rows) * GROUP_COUNT_BYTES
    vector = contract.tokens * contract.k * (
        QUANT_OPS_PER_ELEMENT if dispatch and payload_dtype not in {"bf16", "fp16"} else
        (1 if dispatch else 2 * contract.top_k + bool(contract.shared_intermediate))
    )
    local = StageDemand(name + ":local", after, Work(resources={
        "hbm": read + write, "l2": read + write, "mte2_aiv": read, "vector": vector, "mte3": write,
    }), metadata=metadata)
    amount = contract.transport_bytes[0 if dispatch else 1] or 0
    remote = name + ":transport"
    deps = (local.name,) if dispatch else after
    link = [replace(stage, metadata=tuple({**dict(stage.metadata), **dict(metadata)}.items()))
            for stage in builder(remote, deps, amount, metadata)] if builder else (
        [StageDemand(remote, deps, Work(resources={"link": amount}), metadata=metadata)] if amount else []
    )
    if contract.transport_bytes[0 if dispatch else 1] is not None and sum(
        stage.work.resources.get("link", 0) for stage in link
    ) != amount:
        raise ModelingError("grouped_moe_transport_link_work_not_conserved")
    if not dispatch and link:
        local = replace(local, after=tuple(stage.name for stage in link))
    stages = [local, *link] if dispatch else [*link, local]
    return [*stages, StageDemand(name, (local.name, *(stage.name for stage in link)), metadata=metadata)]
