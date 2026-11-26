"""Shared contracts and physical lowering for grouped GEMM and MoE.

[Role] Normalize routing, model Blaze scheduling and bind conserved work.
[Invariants] Useful matrix work stays distinct from issued fractals. Zero-row
  experts retain their indices. Stage metadata contains only flat builtins.
[Timeouts & Locks] Finite arithmetic with owner-local counters; no I/O or locks.
[Side-Effect Policy] Build immutable plans for the caller's execution engine.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ..contracts import OperatorPlanningContext
from .lowering import (
    MX_DTYPES, MX_SCALE_GROUP_SIZE, PER_TOKEN_SCALE_BYTES, QUANT_OPS_PER_ELEMENT,
    GroupedMoeProgram, StageDemand, ceil_div, element_bytes, tensor_bytes,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"

VLLM_CALLER_SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision="7a7ed073c481d5e3a2fc2194c5b879efb2d373c0",
        relative_path="vllm_ascend/ops/fused_moe/fused_moe.py",
        source_sha256="0647bf1676132121f738cf0d7c569502b5c67a745558d6107e0a39e9bbd9d6c4",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="7a7ed073c481d5e3a2fc2194c5b879efb2d373c0",
        relative_path="vllm_ascend/ops/fused_moe/prepare_finalize.py",
        source_sha256="f42c57c2835494e886bf6deef054362ac1be9faaea7b1220a257210c7e996611",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="7a7ed073c481d5e3a2fc2194c5b879efb2d373c0",
        relative_path="vllm_ascend/platform.py",
        source_sha256="cc0c5a6138c2f76e3d2d371e4a04c0d66e1ad2f9dac316b921e24d40f733689a",
    ),
)
SOURCES: tuple[SourceRef, ...] = VLLM_CALLER_SOURCES

FRACTAL_M_DIM = 16
FRACTAL_N_DIM = 16
FP32_FRACTAL_K = 8
HALF_PRECISION_FRACTAL_K = 16
BYTE_PRECISION_FRACTAL_K = 32
PACKED_PRECISION_FRACTAL_K = 64
BLAZE_BASE_M = 128
BLAZE_BASE_N = 256
BLAZE_SWIZZLE_WINDOW = 4
INT32_HANDOFF_BYTES_PER_ELEM = 4
FUSED_MC2_SHARED_MODE = 3
SPARSE_GROUP_LIST_TYPE = 2

GEMM_MODES = frozenset({"dense_gemm", "grouped_gemm"})
ROUTED_MODES = frozenset({"mega_moe", "dual_batch_moe"})
SUPPORTED_MODES = GEMM_MODES | ROUTED_MODES | {"gmm_swiglu_quant", "dequant_swiglu_quant"}
MODE_ALIASES = {"grouped_moe": "grouped_gemm", "megamoe_arch35": "mega_moe"}
DTYPE_ALIASES = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "fp8_e4m3fn": "fp8_e4m3",
    "mx_fp8": "mxfp8", "mxfp8_e4m3": "mxfp8", "mx_fp4": "mxfp4",
    "mxfp4_e2m1": "mxfp4", "w8a8": "int8",
}
ACTIVATION_OPS = {
    "swiglu": 5, "swigluoai": 8, "situglu": 7, "situ": 7,
    "silu": 4, "gelu": 6, "gelu_tanh": 8, "geglu": 7,
}
SHIPPING_MEGAMOE_ACTIVATIONS = frozenset({"swiglu", "silu", "gelu"})
SHIPPING_SWIGLU_QUANT_ACTIVATIONS = frozenset({"swiglu"})
GATED_ACTIVATIONS = frozenset({"swiglu", "swigluoai", "situglu", "situ", "geglu"})
NZ_LAYOUTS = frozenset({"NZ", "FRACTAL_NZ", "FRACTAL_NZ_C0_16", "FRACTAL_NZ_C0_32"})
VALID_LAYOUTS_B = NZ_LAYOUTS | {"ND", "CONTIGUOUS"}
MX_SCALE_DTYPES = frozenset({"fp8_e8m0", "e8m0"})
ABI_MODES = {
    "aclnnGroupedMatmul": GEMM_MODES,
    "aclnnMoeGroupedMatmulSwigluQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluQuantV2": frozenset({"gmm_swiglu_quant"}),
    "aclnnGroupedMatmulSwigluMxQuant": frozenset({"gmm_swiglu_quant"}),
    "aclnnDequantSwigluQuant": frozenset({"dequant_swiglu_quant"}),
    "aclnnMegaMoe": ROUTED_MODES,
    "aclnnDispatchFFNCombine": ROUTED_MODES,
}


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ModelingError(f"grouped_moe_invalid_{name}:{value!r}")
    return int(value)


def _nonneg_number(value, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (Integral, float)):
        raise ModelingError(f"grouped_moe_invalid_{name}:{value!r}")
    if isinstance(value, float) and (value != value or value == float("inf") or value == float("-inf")):
        raise ModelingError(f"grouped_moe_invalid_{name}:{value!r}")
    if value < 0:
        raise ModelingError(f"grouped_moe_invalid_{name}:{value!r}")
    return int(value) if isinstance(value, Integral) else float(value)


def _mapping(value, name: str) -> dict:
    try:
        return dict(value)
    except (ValueError, TypeError) as exc:
        raise ModelingError(f"grouped_moe_invalid_{name}") from exc


def _dtype(value: str) -> str:
    key = str(value).lower()
    return DTYPE_ALIASES.get(key, key)


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
        column = n_idx * base_n
        width = min(base_n, n - column)
        if in_tail:
            m_split = ceil_div(m, m_tail_cnt)
            n_split = ceil_div(width, n_tail_cnt * n_align) * n_align
            n_sub, m_sub = divmod(split, ceil_div(m, m_split))
            m = min(m_split, m - m_sub * m_split)
            width = min(n_split, width - n_sub * n_split)
            column += n_sub * n_split
        if m > 0 and width > 0:
            yield task, m, width, column


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
        for task, m, n, _ in blaze_tasks(
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
    shared_compute_dtype: str
    shared_weight_dtype: str
    wave_rows: int
    transport_bytes: tuple[int | float | None, int | float | None]
    cohort_transport_bytes: tuple[tuple[int | float | None, int | float | None], ...] = ()
    act_mx_scaled: bool = False
    weight_mx_scaled: bool = False
    act_mx_padded: bool = True
    weight_mx_padded: bool = True
    weight_scale_dtype: str = "fp32"
    topk_weights_type: int = 0
    row_scale: float = 1.0

    @property
    def intermediate(self) -> int:
        return self.n // self.activation_divisor

    @property
    def activation_divisor(self) -> int:
        return 2 if self.activation in GATED_ACTIVATIONS else 1

    @property
    def uses_mx_scales(self) -> bool:
        return (
            self.quant_dtype in MX_DTYPES
            or (self.mode == "gmm_swiglu_quant" and (self.dtype_a in MX_DTYPES or self.act_mx_scaled))
        )

    @property
    def padded_weight(self) -> bool:
        return self.layout_b in NZ_LAYOUTS

    @property
    def matrix_weight_dtype(self) -> str:
        # Routed and A2 A8W4 gated INT32 tensors are ABI containers for packed INT4 weights.
        if self.dtype_b == "int32" and (
            self.mode in ROUTED_MODES
            or (self.mode == "gmm_swiglu_quant" and self.dtype_a == "int8")
        ):
            return "int4"
        return self.dtype_b


def _contract_rows(fields, mode, m, expected_total):
    rows = fields.get("matrix_group_rows", fields.get("group_rows", fields.get("expert_rows")))
    list_rows = None
    if "group_list" in fields:
        kind = fields.get("group_list_type", 0)
        if kind == SPARSE_GROUP_LIST_TYPE:
            if rows is None:
                raise ModelingError("grouped_moe_sparse_group_list_requires_dense_histogram")
        elif kind not in {0, 1}:
            raise ModelingError(f"grouped_moe_unsupported_group_list_type:{kind}")
        else:
            list_rows = tuple(count for _, count in normalize_group_rows(
                fields["group_list"], cumulative=fields.get("cumulative", kind == 0),
                expected_total=expected_total,
            ))
            if rows is None:
                rows = list_rows
    if rows is None:
        if mode in {"dense_gemm", "dequant_swiglu_quant"}:
            rows = (m,)
        else:
            raise ModelingError("grouped_moe_missing_expert_histogram")
    result = tuple(count for _, count in normalize_group_rows(rows, expected_total=expected_total))
    if list_rows is not None and result != list_rows:
        raise ModelingError("grouped_moe_group_list_histogram_mismatch")
    if not result or (mode == "dense_gemm" and len(result) != 1):
        raise ModelingError("grouped_moe_invalid_expert_count")
    return result


def contract_fields(op: OpSpec) -> dict:
    fields = dict(op.metadata)
    for name in ("megamoe_geometry", "grouped_moe_geometry"):
        if name in fields:
            fields.update(_mapping(fields[name], name))
    return fields


def _resolve_input_dtype(op: OpSpec, fields: Mapping, mode: str, operand_default: str) -> tuple[str, bool]:
    pabi = getattr(op, "precision_abi", None)
    explicit: dict[str, str] = {}
    for key in ("dtype_a", "input_dtype", "input_format"):
        val = fields.get(key)
        if val is not None and str(val).strip():
            explicit[key] = _dtype(str(val))
    if pabi is not None and pabi.input_dtype:
        explicit["precision_abi.input_dtype"] = _dtype(pabi.input_dtype)
    if len(set(explicit.values())) > 1:
        raise ModelingError(f"grouped_moe_conflicting_input_dtypes:{sorted(explicit.items())}")
    if explicit:
        return next(iter(explicit.values())), True
    if mode == "dequant_swiglu_quant":
        input_default = _dtype(op.dtype or "int32")
        return _dtype(fields.get(
            "handoff_dtype",
            input_default if input_default in {"int32", "bf16", "fp16", "fp32"} else "int32",
        )), False
    return _dtype(operand_default), False


def _resolve_mx_scale_flags(op: OpSpec, fields: Mapping, dtype_a: str, dtype_b: str) -> tuple[bool, bool, bool, bool]:
    pabi = getattr(op, "precision_abi", None)
    act_scale_fmt = _dtype(fields.get("act_scale_dtype") or (pabi.act_scale_dtype if pabi else "") or "")
    wt_scale_fmt = _dtype(fields.get("weight_scale_dtype") or (pabi.weight_scale_dtype if pabi else "") or "")
    act_mx_scaled = dtype_a in MX_DTYPES or act_scale_fmt in MX_SCALE_DTYPES
    weight_mx_scaled = dtype_b in MX_DTYPES or wt_scale_fmt in MX_SCALE_DTYPES
    act_pack = str(fields.get("act_scale_pack") or (pabi.act_scale_pack if pabi else "") or "").lower()
    wt_pack = str(fields.get("weight_scale_pack") or (getattr(pabi, "weight_scale_pack", "") if pabi else "") or "").lower()
    act_mx_padded = dtype_a in MX_DTYPES or act_pack == "pack2" or not act_mx_scaled
    weight_mx_padded = (
        dtype_b in MX_DTYPES
        or wt_pack == "pack2"
        or (not wt_pack and act_pack == "pack2")
        or not weight_mx_scaled
    )
    return act_mx_scaled, weight_mx_scaled, act_mx_padded, weight_mx_padded


def validate_contract(op: OpSpec, ctx: OperatorPlanningContext) -> GroupedMoeContract:
    fields = contract_fields(op)
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
    cumulative_group_list = (
        "group_list" in fields
        and fields.get("group_list_type", 0) == 0
        and fields.get("cumulative", True) is True
    )
    valid = fields.get(
        "valid_rows",
        fields.get("routed_rows", None if (routed or cumulative_group_list) else m),
    )
    rows = _contract_rows(fields, mode, m, valid)
    allocated = _integer(fields.get("allocated_rows", sum(rows) if routed else m), "allocated_rows")
    if allocated < sum(rows):
        raise ModelingError("grouped_moe_valid_rows_exceed_allocation")
    if _integer(fields.get("scale_group_size", MX_SCALE_GROUP_SIZE), "scale_group_size", 1) != MX_SCALE_GROUP_SIZE:
        raise ModelingError("grouped_moe_unsupported_scale_group")

    operand_default = fields.get("compute_dtype", fields.get("compute_format", op.dtype or "bf16"))
    dtype_a, explicit_input = _resolve_input_dtype(op, fields, mode, str(operand_default))
    dtype_b = _dtype(fields.get("dtype_b", fields.get("weight_dtype", "int4" if dtype_a == "w4a8" else dtype_a)))
    if dtype_a == "w4a8":
        dtype_a = "int8"
    a8w4_packed = (routed or mode == "gmm_swiglu_quant") and dtype_a == "int8" and dtype_b in {"int4", "int32"}
    compute_default = "w4a8" if a8w4_packed else dtype_a
    compute_dtype = _dtype(fields.get("compute_dtype", fields.get("compute_format", compute_default)))
    if mode != "dequant_swiglu_quant":
        fractal_k0(compute_dtype)
    output_dtype = _dtype(fields.get("output_dtype", fields.get("output_format", op.dtype or "bf16")))
    act_mx_scaled, weight_mx_scaled, act_mx_padded, weight_mx_padded = _resolve_mx_scale_flags(
        op, fields, dtype_a, dtype_b,
    )
    quant_default = dtype_a if dtype_a in MX_DTYPES or (routed and dtype_a in {"bf16", "fp16"}) else "int8"
    if mode == "gmm_swiglu_quant" and (dtype_a in MX_DTYPES or act_mx_scaled):
        quant_default = fields.get("output_dtype", fields.get("output_format", quant_default))
    quant_dtype = _dtype(fields.get("quant_dtype", fields.get("quant_output_dtype", quant_default)))
    default_handoff = "int32" if compute_dtype in {"int8", "w4a8"} else output_dtype
    if mode == "dequant_swiglu_quant" and explicit_input and "handoff_dtype" in fields:
        if _dtype(fields["handoff_dtype"]) != dtype_a:
            raise ModelingError("grouped_moe_conflicting_dequant_handoff_dtype")
    handoff_dtype = _dtype(fields.get("handoff_dtype", dtype_a if mode == "dequant_swiglu_quant" else default_handoff))
    for dtype in (dtype_a, dtype_b, output_dtype, quant_dtype, handoff_dtype):
        element_bytes(1, dtype)
    layout_a = str(fields.get("layout_a", "ND")).upper()
    pabi = getattr(op, "precision_abi", None)
    pabi_layout = str(getattr(pabi, "weight_layout", "") or "").upper() if pabi is not None else ""
    default_layout_b = pabi_layout if pabi_layout in VALID_LAYOUTS_B else "ND"
    layout_b = str(fields.get("layout_b", fields.get("weight_layout", default_layout_b))).upper()
    if layout_a not in {"ND", "CONTIGUOUS"} or layout_b not in VALID_LAYOUTS_B:
        raise ModelingError(f"grouped_moe_unsupported_layout:{layout_a}:{layout_b}")
    pabi_wt_scale = str(getattr(pabi, "weight_scale_dtype", "") or "").strip() if pabi is not None else ""
    raw_wt_scale = fields.get("weight_scale_dtype", pabi_wt_scale or "fp32")
    weight_scale_dtype = _dtype(str(raw_wt_scale)) if raw_wt_scale else "fp32"
    topk_weights_type = _integer(fields.get("topk_weights_type", 0), "topk_weights_type")
    if topk_weights_type not in {0, 1}:
        raise ModelingError(f"grouped_moe_invalid_topk_weights_type:{topk_weights_type}")
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
    shared_compute = _dtype(fields.get("shared_compute_dtype", compute_dtype))
    shared_weight = _dtype(fields.get("shared_weight_dtype", dtype_b))
    if shared:
        fractal_k0(shared_compute)
        element_bytes(1, shared_weight)
    transport_fields = ("dispatch_link_bytes", "combine_link_bytes")
    transfer = tuple(_nonneg_number(fields[key], key) if key in fields else None for key in transport_fields)
    cohort_transfer = tuple(
        tuple(_nonneg_number(value, "cohort_link_bytes") for value in pair)
        for pair in fields.get("cohort_link_bytes", ())
    )
    if cohort_transfer:
        if split is None or len(cohort_transfer) != 2 or any(len(pair) != 2 for pair in cohort_transfer):
            raise ModelingError("grouped_moe_invalid_cohort_link_bytes")
        if any(amount is not None and sum(parts) != amount for amount, parts in zip(transfer, zip(*cohort_transfer))):
            raise ModelingError("grouped_moe_dbo_link_work_not_conserved")
    if (routed and ep > 1 and mc2 and (tokens or sum(rows)) and not cohort_transfer
            and None in transfer and not callable(fields.get("cohort_transport_builder"))):
        raise capability_gap(op, ctx, "explicit_dispatch_and_combine_transport")
    row_scale = float(_nonneg_number(fields.get("expected_row_scale", 1.0), "expected_row_scale"))
    if not (0.0 < row_scale <= 1.0):
        raise ModelingError(f"grouped_moe_invalid_expected_row_scale:{row_scale!r}")
    return GroupedMoeContract(
        mode, abi, sum(rows), n, k, rows, allocated, tokens, dtype_a, dtype_b, compute_dtype,
        output_dtype, quant_dtype, handoff_dtype, layout_a, layout_b, transpose, activation,
        _integer(fields.get("base_m", BLAZE_BASE_M), "base_m", 1),
        _integer(fields.get("base_n", BLAZE_BASE_N), "base_n", 1), tail_split,
        _integer(fields.get("m_tail_cnt", 1), "m_tail_cnt", 1),
        _integer(fields.get("n_tail_cnt", 1), "n_tail_cnt", 1),
        top_k, ep, enable_dbo, split, mc2, shared, shared_compute, shared_weight,
        _integer(fields.get("wave_rows", 0), "wave_rows"), transfer, cohort_transfer,
        act_mx_scaled, weight_mx_scaled, act_mx_padded, weight_mx_padded,
        weight_scale_dtype, topk_weights_type, row_scale,
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


def geometry_has_l2(geometry) -> bool:
    """Return True when the target die profile declares an L2 service rate or unit."""
    rates = getattr(geometry, "rates", None) or {}
    if float(rates.get("l2", 0.0) or 0.0) > 0.0:
        return True
    units = getattr(geometry, "resource_units", None) or {}
    return bool(units.get("l2", 0))


def make_plan(
    implementation: str, ctx: OperatorPlanningContext, stages: Iterable[StageDemand],
    sources: tuple[SourceRef, ...], *, metadata: Mapping | None = None,
) -> OperatorPlan:
    stages = tuple(stages)
    if not geometry_has_l2(ctx.geometry):
        stages = tuple(
            replace(stage, work=Work(
                stage.work.matrix,
                {role: amount for role, amount in stage.work.resources.items() if role != "l2"},
            ))
            if "l2" in stage.work.resources else stage
            for stage in stages
        )
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
        weight_col_alignment=(FRACTAL_N_DIM if contract.transpose_weight else fractal_k0(contract.matrix_weight_dtype))
        if contract.padded_weight else 1,
    )


def matrix_stages(
    name, after, contract, schedule, k, output_dtype, *, input_dtype=None, input_scaled=False,
    output_row_divisor: int = 1, useful_flops_override: int | None = None, metadata=(),
):
    dtype = input_dtype or contract.dtype_a
    stages, ends = [], []
    weight_dtype = contract.matrix_weight_dtype
    k_alignment = (fractal_k0(weight_dtype) if contract.transpose_weight else FRACTAL_M_DIM) if (
        contract.padded_weight
    ) else 1
    weight_k = ceil_div(k, k_alignment) * k_alignment
    act_mx = contract.act_mx_scaled if input_dtype is None else dtype in MX_DTYPES
    act_mx_pad = contract.act_mx_padded if input_dtype is None else True
    for owner in schedule.nonempty_owners:
        a_rows, b_cols = schedule.owner_input_rows[owner], schedule.owner_weight_columns[owner]
        a = tensor_bytes(a_rows, k, dtype, scaled=input_scaled, mx_scaled=act_mx, mx_padded=act_mx_pad)
        b = tensor_bytes(
            b_cols, weight_k, weight_dtype,
            mx_scaled=contract.weight_mx_scaled, mx_padded=contract.weight_mx_padded,
        )
        out_elems = schedule.owner_output_elements[owner] // output_row_divisor
        out = element_bytes(out_elems, output_dtype)
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
    useful = schedule.useful_flops if useful_flops_override is None else useful_flops_override
    stages.append(StageDemand(name, tuple(ends) or after, metadata=(*metadata,
        ("useful_flops", useful), ("issued_fractals", schedule.issued_fractals),
        ("max_waves", schedule.max_waves), ("wave_tail_imbalance", schedule.wave_tail_imbalance),
    )))
    return stages


def _vector_owner_loads(rows, cores, group_read):
    if not group_read:
        total = sum(rows)
        quotient, tail = divmod(total, cores)
        return ((quotient + (owner < tail), 0) for owner in range(min(cores, total)))
    # Each active group is repartitioned across AIVs. A participating core
    # keeps that group's channel scales resident while processing its rows.
    counts, scale_reads = [0] * cores, [0] * cores
    for group_rows in rows:
        if not group_rows:
            continue
        step = ceil_div(group_rows, cores)
        for owner, start in enumerate(range(0, group_rows, step)):
            counts[owner] += min(step, group_rows - start)
            scale_reads[owner] += group_read
    return zip(counts, scale_reads)


def vector_stages(name, after, ctx, rows, *, read, compute, write, group_read=0, metadata=()):
    stages = []
    for owner, (count, scale_read) in enumerate(_vector_owner_loads(rows, ctx.effective_aiv_cores, group_read)):
        if not count:
            continue
        read_bytes = count * read + scale_read
        base = f"{name}:{owner}"
        stages.extend((
            StageDemand(base + ":read", after, Work(resources={
                "hbm": read_bytes, "l2": read_bytes, "mte2_aiv": read_bytes,
            }), "aiv", owner, metadata),
            StageDemand(base + ":vector", (base + ":read",), Work(resources={"vector": count * compute}),
                        "aiv", owner, metadata),
            StageDemand(base + ":store", (base + ":vector",), Work(resources={
                "hbm": count * write, "l2": count * write, "mte3": count * write,
            }), "aiv", owner, metadata),
        ))
    stages.append(StageDemand(name, tuple(stage.name for stage in stages if stage.name.endswith(":store")) or after, metadata=metadata))
    return stages


def activation_stages(
    name, after, ctx, contract, rows, metadata=(), *,
    has_activation_scale=True, weight_scale_dtype: str | None = None,
):
    width = contract.intermediate
    quantized = contract.quant_dtype not in {"bf16", "fp16"}
    int32_input = contract.handoff_dtype == "int32"
    scale_read = PER_TOKEN_SCALE_BYTES if int32_input and has_activation_scale else 0
    if weight_scale_dtype is not None and int32_input:
        group_read = element_bytes(contract.n, weight_scale_dtype)
    elif int32_input and contract.mode == "dequant_swiglu_quant":
        group_read = element_bytes(contract.n, "fp32")
    else:
        group_read = 0
    return vector_stages(
        name, after, ctx, rows, read=element_bytes(contract.n, contract.handoff_dtype) + scale_read,
        compute=width * (ACTIVATION_OPS[contract.activation] + (QUANT_OPS_PER_ELEMENT if quantized else 0)),
        write=tensor_bytes(1, width, contract.quant_dtype, scaled=quantized), metadata=metadata,
        group_read=group_read,
    )
