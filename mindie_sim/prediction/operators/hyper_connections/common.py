"""Tensor contracts and physical phases for hyper connections.

[Role] Derive arithmetic, ownership, and memory traffic from caller geometry.
[Invariants] The same phase graph prepares work, describes it, and lowers it
  to the Engine. AttnRes schedules each token's passes on its AIV owner.
[Timeouts & Locks] Bounded by physical owners and operator phases. No locks.
[Side-Effect Policy] No external I/O or device runtime. Live memory admission
  occurs only during Engine execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
    TensorPrecisionABI,
)
from ...mechanism import OperatorPlan, Work
from ...runtime.memory_physics import AddressRun, AddressRunPlan, ReuseKey
from ..contracts import OperatorPlanningContext
from .lowering import FAMILY, PhaseGraph, gap

ARCH22 = ArchId("npu_arch_2201")
ARCH35 = ArchId("npu_arch_3510")
FP32_BYTES = 4
RESIDUAL_RANK = 3
RMS_SCALAR_OPS = 3
MHC_COEFFICIENT_SCALARS = 3
ENGRAM_PRODUCT_OPS_PER_ELEMENT = 5
# Two inverse RMS values and their scaled dot, signed root with clamp, then sigmoid.
ENGRAM_GATE_SCALAR_OPS = 19
ATTNRES_SOFTMAX_OPS_PER_ROW = 5
HC_SINKHORN_STREAMS = 4
SINKHORN_DEFAULT_ITERS = 20
SINKHORN_INIT_ENTRY_OPS = 3
SILU_OPS_PER_ELEMENT = 4
SIGMOID_OPS_PER_ELEMENT = 4
MHC_COLLAPSE_MAX_ROWS = 72
TEXT_GATED_EZ1008_MIN_ROWS = 513
TEXT_GATED_LARGE_MIN_ROWS = 512
TEXT_GATED_SMALL_SPLITK_FACTOR = 14
TEXT_GATED_SMALL_MAX_STREAMS = 10
TEXT_GATED_SMALL_HIDDEN_ALIGNMENT = 256
TEXT_GATED_SMALL_MAX_HIDDEN = 4096
TEXT_GATED_SMALL_RANK_ALIGNMENT = 64
TEXT_GATED_SMALL_MAX_RANK = 896
TEXT_GATED_TASK_ALIGNMENT = 32
TEXT_GATED_FULL_TASK_ROWS = 128
SMALL_M_THRESH_A2 = 1024
TEXT_GATED_A2_C0_TILE_N = 64
TEXT_GATED_A2_C1_TILE_N = 256
TEXT_GATED_C0_TILE_K = 512
TEXT_GATED_C0_SPLITK_LIMIT = 5
TEXT_GATED_C0_BASIC_SPLITK = True
TEXT_GATED_C0_CROSS_SPLITK = False
ATTNRES_UB_OVERHEAD_BYTES = 32 * 1024
ATTNRES_UB_ALIGNMENT_BYTES = 32
ATTNRES_FP32_VECTOR_LANES = 64
ATTNRES_FP32_ALIGNED_SCRATCH_ROWS = 3
ATTNRES_FP32_SCRATCH_ROWS = 4
MAX_OWNER_SERIAL_PHASES = 8
UINT32_MAX = (1 << 32) - 1
ENGRAM_REDUCTIONS = 3
ENGRAM_CHUNK_BYTES_PER_ELEMENT = 12

VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
HC_PRE_TILING_SOURCE = SourceRef(
    "vllm-ascend", VLLM_REVISION, "csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp",
    "29bd7606b274d6a9289ce02b6e6cc03ff947c3a163850263cfea622b53b5d26c",
)
ATTNRES_CALLER_SOURCES: tuple[SourceRef, ...] = (
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_tiling.cpp",
              "b77dc41317386e5ac4d9b6b71e9ef5249c52895b5d37163c54b0cd603ff4c361"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_legacy_plan.h",
              "47c863f70950dcc4ca833aa2e5d2db2cfdb980e4c6475198ee0711d74d3149c3"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_kernel/attn_res_fwd_ub_layout.h",
              "b1898c8cc2d84fd4e66573623198fbac395fec58a802f9dd6028927aa229daff"),
    SourceRef("vllm-ascend", VLLM_REVISION, "csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_reload.h",
              "46b3a1fcec42b5bac511021d1059b4e91ddbb78426563b5a0343d1a9e73db611"),
)
ENGRAM_GATE_SOURCES: tuple[SourceRef, ...] = (
    SourceRef("vllm-ascend", VLLM_REVISION, "vllm_ascend/ops/triton/engram_gate.py",
              "aa52205e9207af272e6179ebf1b6bbd552de65ef6f1472db6de893aded29e47b"),
)
VLLM_CALLER_SOURCES: tuple[SourceRef, ...] = (
    HC_PRE_TILING_SOURCE,
    *ATTNRES_CALLER_SOURCES,
    *ENGRAM_GATE_SOURCES,
)
SHARED_SOURCES: tuple[SourceRef, ...] = (
    SourceRef("cann/cann-recipes-infer", "3c9266727c2a25bd17d503ad37d189a134ff364b",
              "ops/ascendc/src/hc_pre_sinkhorn/op_kernel/hc_pre_sinkhorn.cpp",
              "bb65d5f12d447e80ce4d6f0427c112e4008ae3d047ce441dd058403bff6d95b3"),
    SourceRef("cann/ops-transformer", "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
              "mhc/text_gated_residual/op_kernel/catlass_hc_forward_mul_sum_tiling_data.h",
              "392ea8212ca6ec2ce53d8a8e5704194ca6e7f1b177ab25f4b12e1fac18f43788"),
)


def sources_for_kind(kind: str, strategy_sources: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
    """Select verified upstream provenance strictly for the active operator kind."""
    if kind == "engram_gate":
        return ENGRAM_GATE_SOURCES
    if kind == "attn_res_fwd":
        return ATTNRES_CALLER_SOURCES
    if kind == "text_gated_residual":
        ops_tx = tuple(ref for ref in strategy_sources if "text_gated_residual" in ref.relative_path)
        shared_tx = tuple(ref for ref in SHARED_SOURCES if "text_gated_residual" in ref.relative_path)
        return ops_tx + shared_tx or strategy_sources
    hc_strategy = tuple(ref for ref in strategy_sources if "hc_pre" in ref.relative_path)
    hc_shared = tuple(ref for ref in SHARED_SOURCES if "hc_pre" in ref.relative_path)
    return (hc_strategy or strategy_sources) + hc_shared + (HC_PRE_TILING_SOURCE,)

MHC_KINDS = frozenset({"hc_pre", "hc_post", "hc_pre_sinkhorn", "mhc_post_collapse"})
LEGACY_ALIASES = {
    "npu_hc_pre_v2_arch35": "hc_pre",
    "npu_hc_pre_v3_arch35": "hc_pre",
    "npu_hc_post_arch35": "hc_post",
    "mhc_post_collapse_norm_arch35": "mhc_post_collapse",
    "text_gated_residual_arch35": "text_gated_residual",
}
ABI_NAMES = {
    "hc_pre": frozenset({"npu_hc_pre", "npu_hc_pre_v2", "npu_hc_pre_v3"}),
    "hc_post": frozenset({"npu_hc_post"}),
    "hc_pre_sinkhorn": frozenset({"aclnnHcPreSinkhorn"}),
    "mhc_post_collapse": frozenset({"mhc_post_collapse_norm"}),
    "text_gated_residual": frozenset({"aclnnTextGatedResidual"}),
    "attn_res_fwd": frozenset({"npu_attn_res_fwd"}),
    "engram_gate": frozenset({"fused_engram_gate"}),
}
OWNED_RECIPES = frozenset(ABI_NAMES) | LEGACY_ALIASES.keys()
FAMILY_DEFAULTS = {"attention_residual": "attn_res_fwd", "engram_gate": "engram_gate"}


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"hyper_connections_invalid_{name}:{value}")


def _dtype(value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ModelingError(f"hyper_connections_invalid_dtype:{value}")


def _ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor

@dataclass(frozen=True, slots=True)
class MhcContract:
    rows: int
    hc_mult: int
    hidden_size: int
    n_mix: int | None = None
    sinkhorn_iters: int = SINKHORN_DEFAULT_ITERS
    split_k: int = 1
    mhc_aux_cores: int | None = None
    dtype: str = "bf16"
    has_collapse: bool = False
    unbounded_rows: bool = False

    def __post_init__(self) -> None:
        _integer("rows", self.rows)
        _integer("hc_mult", self.hc_mult, 1)
        _integer("hidden_size", self.hidden_size, 1)
        _integer("split_k", self.split_k, 1)
        _integer("sinkhorn_iters", self.sinkhorn_iters, 1)
        _dtype(self.dtype, {"bf16", "fp16", "fp32"})
        width = (2 + self.hc_mult) * self.hc_mult
        if self.n_mix is None:
            object.__setattr__(self, "n_mix", width)
        _integer("n_mix", self.n_mix, width)
        if self.split_k > self.hc_mult * self.hidden_size:
            raise ModelingError("hc_split_k_exceeds_input_width")
        if self.mhc_aux_cores is not None:
            _integer("mhc_aux_cores", self.mhc_aux_cores)
        if self.has_collapse and not self.unbounded_rows and self.rows > MHC_COLLAPSE_MAX_ROWS:
            raise ModelingError(f"mhc_post_collapse_row_cutoff_exceeded:{self.rows}")

@dataclass(frozen=True, slots=True)
class TextGatedResidualContract:
    rows: int
    hc_mult: int
    hidden_size: int
    d_dim: int
    rank: int
    has_gate: bool = False
    has_gamma: bool = False
    dtype: str = "bf16"
    stack_binding: Literal["ez1008", "cann_current"] = "cann_current"
    strict_native: bool = False

    def __post_init__(self) -> None:
        _integer("rows", self.rows)
        for name in ("hc_mult", "hidden_size", "d_dim", "rank"):
            _integer(name, getattr(self, name), 1)
        _dtype(self.dtype, {"bf16", "fp16"})
        if self.stack_binding not in {"ez1008", "cann_current"}:
            raise ModelingError(f"text_gated_residual_invalid_stack_binding:{self.stack_binding}")

@dataclass(frozen=True, slots=True)
class TextGatedGuardDecision:
    selected_template: str
    splitk_factor: int = 1
    m_task: int = 0
    used_aic: int = 0
    excluded_fusion: str | None = None


def _text_gate_a2_small(contract: TextGatedResidualContract, aic_cores: int) -> TextGatedGuardDecision:
    m, d, r = contract.rows, contract.d_dim, contract.rank
    tile_m = TEXT_GATED_TASK_ALIGNMENT
    if m >= TEXT_GATED_FULL_TASK_ROWS:
        task = _ceil(_ceil(m, 2), tile_m) * tile_m
    elif m >= TEXT_GATED_FULL_TASK_ROWS // 2:
        task = TEXT_GATED_FULL_TASK_ROWS
    else:
        task = tile_m
    # HcGetSplitkFactor ignores M/R, L1M/L1N and aicNum. HcC0L1K is
    # 512 for both element widths in this host revision.
    split = 1
    if not TEXT_GATED_C0_BASIC_SPLITK or TEXT_GATED_C0_CROSS_SPLITK:
        split = max(1, min(TEXT_GATED_C0_SPLITK_LIMIT, _ceil(d, TEXT_GATED_C0_TILE_K)))
    width = r + (contract.hc_mult if contract.has_gate else 0)
    c0_loops = _ceil(width, TEXT_GATED_A2_C0_TILE_N) * split * _ceil(task, tile_m)
    c1_slices = _ceil(d, TEXT_GATED_A2_C1_TILE_N)
    return TextGatedGuardDecision("native_a2_small", split, task, max(1, min(aic_cores, max(c0_loops, c1_slices))))


def _text_gate_a5_small_reason(c: int, h: int, r: int, aic_cores: int) -> str | None:
    if not 2 <= c <= TEXT_GATED_SMALL_MAX_STREAMS:
        return "a5_small_hc_mult"
    if not (TEXT_GATED_SMALL_HIDDEN_ALIGNMENT <= h <= TEXT_GATED_SMALL_MAX_HIDDEN
            and h % TEXT_GATED_SMALL_HIDDEN_ALIGNMENT == 0):
        return "a5_small_hidden_size"
    if not (TEXT_GATED_SMALL_RANK_ALIGNMENT <= r <= TEXT_GATED_SMALL_MAX_RANK
            and r % TEXT_GATED_SMALL_RANK_ALIGNMENT == 0):
        return "a5_small_rank"
    if aic_cores == 0:
        return "a5_small_aic_cores"
    return None


def evaluate_text_gated_guard(
    contract: TextGatedResidualContract, arch_id: ArchId, aic_cores: int,
) -> TextGatedGuardDecision:
    """Evaluate the declared stack's row and native tiling domain."""
    m, c, h, r = contract.rows, contract.hc_mult, contract.hidden_size, contract.rank
    _integer("aic_cores", aic_cores)
    if m == 0:
        return TextGatedGuardDecision("empty_noop")
    if arch_id not in {ARCH22, ARCH35}:
        raise gap("text_gated_residual_arch_unsupported", str(arch_id))
    reason = None
    if contract.stack_binding == "ez1008":
        if m < TEXT_GATED_EZ1008_MIN_ROWS:
            reason = "ez1008_m_le_512"
    elif arch_id == ARCH22 and m < SMALL_M_THRESH_A2:
        return _text_gate_a2_small(contract, aic_cores)
    elif arch_id == ARCH35 and m < TEXT_GATED_LARGE_MIN_ROWS:
        reason = _text_gate_a5_small_reason(c, h, r, aic_cores)
        if reason is None:
            return TextGatedGuardDecision("native_a5_small", TEXT_GATED_SMALL_SPLITK_FACTOR, m, aic_cores)
    if reason is not None:
        if contract.strict_native:
            code = "ez1008_row_floor_not_met" if reason == "ez1008_m_le_512" else reason
            raise ModelingError(f"text_gated_residual_{code}")
        return TextGatedGuardDecision(
            "decomposed_fallback", 1, m, min(m, aic_cores), excluded_fusion=reason,
        )
    task = TEXT_GATED_FULL_TASK_ROWS
    if _ceil(m, task) < aic_cores:
        task //= 2
    return TextGatedGuardDecision("native_large", 1, task, min(aic_cores, _ceil(m, task)))

@dataclass(frozen=True, slots=True)
class AttnResContract:
    tokens: int
    hidden_size: int
    capacity: int
    valid_history: int
    token_stride: int | None = None
    fuse_add: bool = True
    output_norm: bool = False
    cached_rows: int = 0
    dtype: str = "bf16"
    ub_overhead_bytes: int = ATTNRES_UB_OVERHEAD_BYTES
    fused_chain: bool = False
    need_backward: bool = False

    def __post_init__(self) -> None:
        _integer("tokens", self.tokens)
        _integer("hidden_size", self.hidden_size, 1)
        _integer("capacity", self.capacity, 1)
        _integer("valid_history", self.valid_history)
        _integer("cached_rows", self.cached_rows)
        _integer("ub_overhead_bytes", self.ub_overhead_bytes)
        _dtype(self.dtype, {"bf16"})
        stride = self.capacity * self.hidden_size
        if self.token_stride is None:
            object.__setattr__(self, "token_stride", stride)
        _integer("token_stride", self.token_stride, stride)
        if not self.cached_rows <= self.valid_history <= self.capacity:
            raise ModelingError("attnres_history_out_of_capacity")


def attnres_max_resident_rows(
    hidden: int, ub_bytes: int, *, fuse_add: bool = True,
    ub_overhead_bytes: int = ATTNRES_UB_OVERHEAD_BYTES,
    physical_capacity_cap: int | None = None,
    regbase: bool = True, fused_chain: bool = False, need_backward: bool = False,
) -> int:
    """Apply the host's aligned UB capacity and RegBase admission bounds."""
    _integer("hidden_size", hidden, 1)
    _integer("ub_bytes", ub_bytes)
    _integer("ub_overhead_bytes", ub_overhead_bytes)
    b16 = int(dtype_bytes("bf16"))
    align = ATTNRES_UB_ALIGNMENT_BYTES
    meta_block = align // FP32_BYTES
    row = hidden * b16
    aligned_b16 = _ceil(row, align) * align
    aligned_fp32 = _ceil(hidden * FP32_BYTES, align) * align
    lanes = ATTNRES_FP32_VECTOR_LANES
    fused_prefix = fuse_add or fused_chain

    def resident_bytes(rows: int) -> int:
        meta = _ceil(rows, meta_block) * meta_block
        scores = _ceil(meta, lanes) * lanes if regbase and rows > 2 * lanes else meta
        scratch = meta if regbase else max(meta, lanes)
        resident_rows = rows - int(fused_prefix and rows > 0)
        buffers = (2 + resident_rows + int(fused_prefix)) * aligned_b16
        buffers += ATTNRES_FP32_ALIGNED_SCRATCH_ROWS * aligned_fp32
        scalars = scores + scratch + meta * meta_block + meta_block
        if need_backward:
            scalars += meta_block + meta
        return ub_overhead_bytes + buffers + scalars * FP32_BYTES

    low, high = 0, min(ub_bytes // aligned_b16, UINT32_MAX)
    while low < high:
        mid = (low + high + 1) // 2
        if resident_bytes(mid) <= ub_bytes:
            low = mid
        else:
            high = mid - 1
    limit = low
    if regbase and limit:
        admission = hidden * ATTNRES_FP32_SCRATCH_ROWS * FP32_BYTES + (2 + int(fuse_add)) * row
        limit = min(limit, max(0, (ub_bytes - ub_overhead_bytes - admission) // (row + FP32_BYTES)))
    if physical_capacity_cap is not None:
        _integer("physical_capacity_cap", physical_capacity_cap)
        limit = min(limit, physical_capacity_cap)
    return limit

@dataclass(frozen=True, slots=True)
class EngramGateContract:
    tokens: int
    hidden_size: int
    active_tokens: int | None = None
    chunk_dim: int | None = None
    dtype: str = "bf16"
    branches: int = 1

    def __post_init__(self) -> None:
        _integer("tokens", self.tokens)
        _integer("branches", self.branches, 1)
        _integer("hidden_size", self.hidden_size, 1)
        _dtype(self.dtype, {"bf16"})
        if self.active_tokens is None:
            object.__setattr__(self, "active_tokens", self.tokens)
        if self.chunk_dim is None:
            object.__setattr__(self, "chunk_dim", self.hidden_size)
        _integer("active_tokens", self.active_tokens)
        _integer("chunk_dim", self.chunk_dim, 1)
        if self.active_tokens > self.tokens:
            raise ModelingError("engram_gate_active_tokens_exceed_tokens")


def sinkhorn_vector_ops(tokens: int, hc_mult: int, iters: int) -> int:
    """Count row-softmax initialization and subsequent row/column half-steps."""
    for label, value, minimum in (("tokens", tokens, 0), ("hc_mult", hc_mult, 1), ("sinkhorn_iters", iters, 0)):
        _integer(label, value, minimum)
    adds = hc_mult * (hc_mult - 1)
    entries = hc_mult * hc_mult
    # The initial row softmax is the first normalization. Later denominators
    # add epsilon, while initialization also shifts logits and stabilizes entries.
    initialization = adds + SINKHORN_INIT_ENTRY_OPS * entries
    half_steps = 2 * iters
    return tokens * (initialization + half_steps * (adds + entries) + max(0, half_steps - 1) * hc_mult)

def _matrix(graph, name, m, k, n, dtype, after, owners, *, gram=0) -> str:
    elt = int(dtype_bytes(dtype))
    l1 = graph.add(f"{name}_operands", after, aic=True, owners=owners,
                   work=Work(resources={"mte1": (m + n) * k * elt}))
    work = Work(matrix={dtype: 2 * m * k * n})
    if gram:
        work += Work(matrix={"fp32": gram})
    return graph.add(name, (l1,), aic=True, owners=owners, work=work)


def _partition(graph: PhaseGraph, contract: MhcContract, pre: bool):
    m, c, h = contract.rows, contract.hc_mult, contract.hidden_size
    cores = len(graph.aiv)
    aux = contract.mhc_aux_cores
    if aux is None:
        aux = 0
        if pre and cores > 1:
            primary_work = h * (2 * c - 1)
            sink_work = sinkhorn_vector_ops(1, c, contract.sinkhorn_iters)
            aux = min(range(1, cores), key=lambda n: (
                max(_ceil(m, cores - n) * primary_work, _ceil(m, n) * sink_work), n,
            ))
    if aux and aux >= cores:
        raise ModelingError("mhc_aux_leaves_no_primary_cores")
    return graph.aiv[:cores - aux], graph.aiv[cores - aux:] if aux else ()


def model_pre_sinkhorn(graph: PhaseGraph, t: MhcContract) -> dict:
    m, c, h = t.rows, t.hc_mult, t.hidden_size
    if (c, t.sinkhorn_iters) != (HC_SINKHORN_STREAMS, SINKHORN_DEFAULT_ITERS):
        raise ModelingError("hc_pre_sinkhorn_requires_hc4_and_iters20")
    sink_ops = sinkhorn_vector_ops(m, c, t.sinkhorn_iters)
    owners = graph.aiv[:m]
    if m:
        mix = t.n_mix
        elt = int(dtype_bytes(t.dtype))
        coefficients = graph.transfer(
            "precomputed_coefficient_inputs",
            (m * (mix + 1) + mix + MHC_COEFFICIENT_SCALARS) * FP32_BYTES, owners=owners,
        )
        affine = graph.vector("scale_bias", m * MHC_COEFFICIENT_SCALARS * mix, (coefficients,), owners=owners)
        gate = graph.vector("pre_post_gates", m * 2 * c * (SIGMOID_OPS_PER_ELEMENT + 1), (affine,), owners=owners)
        sink = graph.vector("sinkhorn", sink_ops, (affine,), owners=owners)
        graph.transfer("comb_output", m * c * c * FP32_BYTES, (sink,), write=True, owners=owners)
        graph.transfer("post_gate_output", m * c * FP32_BYTES, (gate,), write=True, owners=owners)
        x = graph.transfer("residual_for_pre_mix", m * c * h * elt, owners=owners)
        mixed = graph.vector("pre_mix", m * h * (2 * c - 1), (x, gate), owners=owners)
        graph.transfer("pre_mix_output", m * h * elt, (mixed,), write=True, owners=owners)
    return {"split_k": 1, "sinkhorn_ops": sink_ops, "projection_flops": 0,
            "mhc_aux_cores": 0, "primary_aiv_cores": len(owners)}


def model_mhc(graph: PhaseGraph, kind: str, t: MhcContract) -> dict:
    m, c, h = t.rows, t.hc_mult, t.hidden_size
    pre = kind == "hc_pre"
    sink_ops = sinkhorn_vector_ops(m, c, t.sinkhorn_iters) if pre else 0
    mix = t.n_mix
    metadata = {"split_k": t.split_k, "sinkhorn_ops": sink_ops,
                "projection_flops": 2 * m * c * h * mix if pre else 0}
    if not m:
        return {**metadata, "mhc_aux_cores": 0, "primary_aiv_cores": 0}
    primary, aux = _partition(graph, t, pre)
    primary = primary[:m]
    metadata.update(mhc_aux_cores=len(aux), primary_aiv_cores=len(graph.aiv) - len(aux))
    elt = int(dtype_bytes(t.dtype))
    x_bytes = m * c * h * elt
    coefficients = m * c * (c + 1) * FP32_BYTES
    if not pre:
        read = graph.transfer("post_inputs", x_bytes + m * h * elt + coefficients, owners=primary)
        # Each output stream j contracts input stream i with comb[i, j],
        # then adds layer_output * post[j] to the same owner-local row.
        post = graph.vector("stream_contraction", m * c * h * (2 * c + 1), (read,),
                            owners=primary, metadata=(("coefficient_orientation", "input_to_output"),))
        size = x_bytes
        if t.has_collapse:
            post = graph.vector("collapse_and_norm", m * (h * (2 * c - 1 + 2 + 2) + 2),
                                (post,), owners=primary)
            size = m * h * elt
        graph.transfer("post_output", size, (post,), write=True, owners=primary)
        return metadata

    aic = graph.aic[:m * t.split_k]
    load = graph.transfer("projection_inputs", x_bytes + c * h * mix * elt, aic=True, owners=aic)
    split = t.split_k
    projection = _matrix(
        graph, "split_k_partials" if split > 1 else "mix_projection", m, c * h, mix,
        t.dtype, (load,), aic, gram=m * (2 * c * h - split),
    )
    partials = m * (mix + 1) * split * FP32_BYTES
    publish = graph.transfer("projection_workspace_write", partials, (projection,), aic=True, write=True, owners=aic)
    read = graph.transfer("projection_workspace_read", partials, (publish,), owners=primary)
    if split > 1:
        read = graph.vector("split_k_merge", m * (mix + 1) * (split - 1), (read,), owners=primary)
    coef = graph.transfer("coefficient_parameters", (mix + MHC_COEFFICIENT_SCALARS) * FP32_BYTES,
                          (read,), owners=primary)
    ready = graph.vector("norm_scale_bias", m * (RMS_SCALAR_OPS + MHC_COEFFICIENT_SCALARS * mix + 2 * c * (2 + 2)),
                         (coef,), owners=primary)
    if aux:
        handoff = graph.transfer("sinkhorn_workspace_write", m * c * c * FP32_BYTES,
                                 (ready,), write=True, owners=primary)
        sink_input = graph.transfer("sinkhorn_workspace_read", m * c * c * FP32_BYTES,
                                    (handoff,), owners=aux[:m])
    else:
        sink_input = ready
    sink_owners = aux[:m] or primary
    sink = graph.vector("sinkhorn", sink_ops, (sink_input,), owners=sink_owners,
                        metadata=(("stream", "mhc_aux" if aux else "main"),))
    graph.transfer("comb_output", m * c * c * FP32_BYTES, (sink,), write=True, owners=sink_owners)
    residual = graph.transfer("residual_for_pre_mix", x_bytes, (ready,), owners=primary)
    mixed = graph.vector("pre_mix", m * h * (2 * c - 1), (residual,), owners=primary)
    graph.transfer("pre_and_post_gate_outputs", m * h * elt + m * c * FP32_BYTES,
                   (mixed,), write=True, owners=primary)
    return metadata


def model_text_gate(graph: PhaseGraph, t: TextGatedResidualContract, decision: TextGatedGuardDecision) -> dict:
    m, c, h, d, r = t.rows, t.hc_mult, t.hidden_size, t.d_dim, t.rank
    template = decision.selected_template
    metadata = {"selected_template": template, "splitk_factor": decision.splitk_factor,
                "m_task": decision.m_task, "used_aic": decision.used_aic,
                "excluded_fusion": decision.excluded_fusion}
    if not m:
        return metadata
    aic = graph.aic[:decision.used_aic]
    aiv = graph.aiv[:m]
    elt = int(dtype_bytes(t.dtype))
    split, raw = decision.splitk_factor, r + (c if t.has_gate else 0)
    residual_bytes = m * d * elt
    gamma_bytes = d * FP32_BYTES if t.has_gamma else 0
    x = graph.transfer("text_input", residual_bytes + gamma_bytes, owners=aiv)
    rms_groups = c if d == c * h else 1
    rms = m * (2 * d + d - rms_groups + rms_groups * RMS_SCALAR_OPS + d * int(t.has_gamma))
    norm = graph.vector("input_rms", rms, (x,), owners=aiv)
    xn = graph.transfer("normalized_workspace_write", residual_bytes, (norm,), write=True, owners=aiv)
    inp = graph.transfer("down_and_gate_inputs", residual_bytes + d * raw * elt, (xn,), aic=True, owners=aic)
    down = _matrix(graph, "down_and_gate_partials" if split > 1 else "down_and_gate",
                   m, d, raw, t.dtype, (inp,), aic)
    down_bytes = m * raw * (FP32_BYTES * split if split > 1 else elt)
    write = graph.transfer("down_workspace_write", down_bytes, (down,), aic=True, write=True, owners=aic)
    read = graph.transfer("down_workspace_read", down_bytes, (write,), owners=aiv)
    if split > 1:
        read = graph.vector("text_split_k_merge", m * raw * (split - 1), (read,), owners=aiv)
        raw_out = graph.transfer("merged_down_write", m * raw * elt, (read,), write=True, owners=aiv)
        read = graph.transfer("merged_down_read", m * raw * elt, (raw_out,), owners=aiv)
    activation = graph.vector("down_activation", m * r * (1 + SILU_OPS_PER_ELEMENT), (read,), owners=aiv)
    lowrank = graph.transfer("lowrank_workspace_write", m * r * elt,
                             (activation,), write=True, owners=aiv)
    up_input = graph.transfer("up_inputs", (m * r + r * d) * elt,
                              (lowrank,), aic=True, owners=aic)
    up = _matrix(graph, "up_projection", m, r, d, t.dtype, (up_input,), aic)
    up_bytes = m * d * elt
    up_write = graph.transfer("up_workspace_write", up_bytes, (up,), aic=True, write=True, owners=aic)
    if t.has_gate:
        inject = graph.vector("inject_gate", m * c * (SIGMOID_OPS_PER_ELEMENT + 2), (read,), owners=aiv)
        graph.transfer("inject_output", m * c * elt, (inject,), write=True, owners=aiv)
    inputs = graph.transfer("gated_reduction_inputs", up_bytes + residual_bytes,
                            (up_write,), owners=aiv)
    products = m * c * h
    sigmoid = graph.vector("read_gate_sigmoid", products * SIGMOID_OPS_PER_ELEMENT, (inputs,), owners=aiv)
    gated = graph.vector("sigmoid_weighted_products", products, (sigmoid,), owners=aiv)
    if template == "decomposed_fallback":
        materialized = graph.transfer("decomposed_gate_write", products * elt, (gated,), write=True, owners=aiv)
        gated = graph.transfer("decomposed_gate_read", products * elt, (materialized,), owners=aiv)
    mixed = graph.vector("channel_mean", m * h * (c - 1 + 1), (gated,), owners=aiv)
    graph.transfer("text_gated_output", m * h * elt, (mixed,), write=True, owners=aiv)
    return {**metadata, "projection_flops": 2 * m * d * (raw + r)}


def model_attnres(graph: PhaseGraph, t: AttnResContract, ub: int, physical_cap: int | None,
                  *, regbase: bool, bank_write: bool = False) -> dict:
    m, h, j = t.tokens, t.hidden_size, t.valid_history
    limit = attnres_max_resident_rows(h, ub, fuse_add=t.fuse_add,
                                     ub_overhead_bytes=t.ub_overhead_bytes, physical_capacity_cap=physical_cap,
                                     regbase=regbase, fused_chain=t.fused_chain, need_backward=t.need_backward)
    resident = j + 1 <= limit
    cached = j if resident else t.cached_rows
    template = "resident" if resident else "cooperative_reload" if 0 < cached < j else "reload"
    elt = int(dtype_bytes(t.dtype))
    row = elt * h
    prefix_reload_bytes = row if not (resident or t.fuse_add or t.fused_chain) else 0
    reload_inputs = "history_and_prefix" if prefix_reload_bytes else "history"
    metadata = {"resident_row_limit": limit, "selected_template": template,
                "reread_bytes": m * (j - cached) * row, "history_read_bytes": m * j * row}
    if m and not graph.aiv:
        raise gap("hyper_connections_requires_aiv_owners", "attn_res_fwd")
    if not m:
        return metadata
    stride = elt * t.token_stride
    # Keep the prefix tensor's sectors disjoint from the history bank in each reuse group.
    prefix_base = _ceil(m * stride, graph.sector) * graph.sector
    per_owner = _ceil(m, len(graph.aiv))
    for index, start in enumerate(range(0, m, per_owner)):
        owners = (graph.aiv[index],)
        previous = ()
        for token in range(start, min(m, start + per_owner)):
            prefix = f"token_{token}"
            prefix_bytes = row * (1 + int(t.fuse_add) + (2 if token == start else 0)) - prefix_reload_bytes
            deps = previous
            if prefix_bytes:
                deps = (graph.transfer(f"{prefix}_prefix_input", prefix_bytes, deps, owners=owners),)
            bank_base = token * stride
            prefix_runs = (AddressRun(prefix_base + token * row, 1, row, row),) if prefix_reload_bytes else ()
            runs = ((AddressRun(bank_base, 1, j * row, stride),) if j else ()) + prefix_runs
            reuse = distance = None
            if j > cached or prefix_reload_bytes:
                reuse = ReuseKey(0, f"attnres_reload_{token}", 0)
                access_plan = AddressRunPlan(runs, sector_bytes=graph.sector)
                distance = max(access_plan.aligned_copy_bytes(), access_plan.dram_sector_and_service(())[0])
            if runs:
                inp = graph.transfer(f"{prefix}_pass1_{reload_inputs}", j * row + prefix_reload_bytes, deps,
                                     owners=owners, runs=runs, reuse=reuse, distance=distance)
                deps = (inp,)
            score_per_row = 2 * (h - 1) + (2 + 2) * h + RMS_SCALAR_OPS
            fuse = (2 + 2) * h if t.fuse_add else 0
            score = graph.vector(f"{prefix}_pass1_score", (j + 1) * score_per_row + fuse, deps, owners=owners)
            softmax = graph.vector(f"{prefix}_softmax", ATTNRES_SOFTMAX_OPS_PER_ROW * (j + 1) - 1,
                                   (score,), owners=owners)
            deps = (softmax,)
            if reuse is not None:
                size = (j - cached) * row
                runs = ((AddressRun(bank_base + cached * row, 1, size, stride),) if size else ()) + prefix_runs
                reread = graph.transfer(f"{prefix}_pass2_{reload_inputs}", size + prefix_reload_bytes, deps,
                                        owners=owners, runs=runs, reuse=reuse, distance=distance,
                                        final_reuse_read=True)
                deps = (reread,)
            if t.output_norm:
                deps += (graph.transfer(f"{prefix}_output_gamma", row, owners=owners),)
            norm = (2 + 2) * h + 2 * h + h + RMS_SCALAR_OPS if t.output_norm else 0
            weighted = graph.vector(f"{prefix}_pass2_weighted", 2 * (j + 1) * h + norm, deps, owners=owners)
            output = graph.transfer(f"{prefix}_output", row * (1 + int(t.fuse_add) + int(bank_write)),
                                    (weighted,), write=True, owners=owners)
            graph.end_token()
            previous = (output,)
    return metadata


def attnres_spec_terms(op: OpSpec) -> dict[str, float]:
    """Geometry-free compulsory work of one ``AttnResFwd`` call for its ``OpSpec``.

    Mirrors ``model_attnres`` per token (vector work is owner-independent);
    bytes count each compulsory tensor once: prefix (+addend), the ``j`` valid
    history rows, the score/norm weights (only when mixing is active) and the
    stored rows. Zero-history calls bypass scoring, Softmax, mixing, and score
    weight reads. UB-capacity re-reads and per-owner weight reloads only add
    traffic on top.
    """
    contract, options = operator_contract(op, "attn_res_fwd")
    m, h, j = contract.tokens, contract.hidden_size, contract.valid_history
    row = int(dtype_bytes(contract.dtype)) * h
    fuse = int(contract.fuse_add)
    mixing = j > 0
    weights = (2 * int(mixing) + int(contract.output_norm)) * row
    if not m:
        return {"hbm_bytes": 0.0, "mte2_bytes": 0.0, "mte3_bytes": 0.0, "vector_ops": 0.0}
    reads = m * (1 + fuse + (j if mixing else 0)) * row + weights
    writes = m * (
        1 + fuse + int(options["bank_write"]) + int(options.get("save_materialized", False))
    ) * row
    norm = (2 + 2) * h + 2 * h + h + RMS_SCALAR_OPS if contract.output_norm else 0
    if mixing:
        score_per_row = 2 * (h - 1) + (2 + 2) * h + RMS_SCALAR_OPS
        per_token = ((j + 1) * score_per_row + (2 + 2) * h * fuse
                     + ATTNRES_SOFTMAX_OPS_PER_ROW * (j + 1) - 1 + 2 * (j + 1) * h + norm)
    else:
        # Zero-history bypass: optional fused add + BF16->FP32 bypass + FP32->BF16 round + optional norm.
        per_token = (2 + 2) * h * fuse + 2 * h + norm
    return {"hbm_bytes": float(reads + writes), "mte2_bytes": float(reads),
            "mte3_bytes": float(writes), "vector_ops": float(m * per_token)}


def model_engram(graph: PhaseGraph, t: EngramGateContract, ub: int) -> dict:
    m, h, active = t.tokens, t.hidden_size, t.active_tokens
    rows = m * t.branches
    chunk = min(t.chunk_dim, h, max(1, (ub - ENGRAM_REDUCTIONS * FP32_BYTES) // ENGRAM_CHUNK_BYTES_PER_ELEMENT))
    reductions = rows * ENGRAM_REDUCTIONS * (h - 1)
    metadata = {"chunk_dim": chunk, "hidden_slices": _ceil(h, chunk), "sinkhorn_ops": 0,
                "active_tokens": active, "branches": t.branches, "evaluated_rows": rows, "reduction_adds": reductions}
    if not m:
        return metadata
    aiv = graph.aiv[:rows]
    elt = int(dtype_bytes(t.dtype))
    inp = graph.transfer("engram_x_key_weights", rows * h * elt * (2 + 2), owners=aiv)
    reduce = graph.vector("hidden_slice_partial_reductions", reductions + rows * h * ENGRAM_PRODUCT_OPS_PER_ELEMENT,
                          (inp,), owners=aiv)
    gate = graph.vector("rms_signed_root_sigmoid", rows * ENGRAM_GATE_SCALAR_OPS, (reduce,), owners=aiv)
    mask = graph.transfer("token_mask", rows, (gate,), owners=aiv)
    masked = graph.vector("apply_token_mask", 2 * rows, (mask,), owners=aiv)
    value = graph.transfer("engram_value_and_spilled_x", rows * h * elt * (1 + int(chunk < h)),
                           (masked,), owners=aiv)
    out = graph.vector("engram_gated_add", 2 * rows * h, (value,), owners=aiv)
    graph.transfer("engram_output", rows * h * elt, (out,), write=True, owners=aiv)
    return metadata


def recipe_kind(op: OpSpec) -> str:
    recipe = op.recipe or FAMILY_DEFAULTS.get(op.family) or dict(op.metadata).get("variant", op.family)
    return LEGACY_ALIASES.get(recipe, recipe)

@dataclass(frozen=True, slots=True)
class HyperConnectionsStrategy:
    """Bind the common physics to one upstream ABI and UB capacity."""

    arch_id: ArchId
    abi: str
    sources: tuple[SourceRef, ...]
    ub_bytes: int
    datapath_name: str
    exclusive_abis: frozenset[str]

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        kind = recipe_kind(op)
        if ctx.compute_arch.arch_id != self.arch_id or kind not in ABI_NAMES:
            return False
        if op.recipe in LEGACY_ALIASES and self.arch_id != ARCH35:
            return False
        abi = ctx.requested_abi
        if abi is None:
            return True
        if abi == "fusedSmall":
            return kind == "text_gated_residual" and abi in self.exclusive_abis
        return abi in ABI_NAMES[kind] or abi == kind or abi in self.exclusive_abis

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if not self.accepts(op, ctx):
            raise gap("hyper_connections_strategy_unsupported",
                      f"{ctx.compute_arch.arch_id}:{op.recipe}:{ctx.requested_abi}")
        if op.source_boundary == "call":
            raise gap("hyper_connections_requires_expanded_kernel", op.op_id)
        kind = recipe_kind(op)
        graph = PhaseGraph(ctx)
        declared_ub = dict(ctx.datapath.buffers).get("ub", ctx.geometry.buffers.get("ub", self.ub_bytes))
        ub = min(self.ub_bytes, declared_ub)
        contract, options = operator_contract(op, kind)
        if kind == "hc_pre_sinkhorn":
            metadata = model_pre_sinkhorn(graph, contract)
        elif kind in MHC_KINDS:
            metadata = model_mhc(graph, kind, contract)
        elif kind == "text_gated_residual":
            decision = evaluate_text_gated_guard(contract, self.arch_id, len(graph.aic))
            if ctx.requested_abi == "fusedSmall" and decision.selected_template not in {"native_a5_small", "empty_noop"}:
                raise gap("text_gated_residual_fused_small_abi_unsupported",
                          decision.excluded_fusion or decision.selected_template)
            metadata = model_text_gate(graph, contract, decision)
        elif kind == "attn_res_fwd":
            attn_options = {
                "physical_cap": options.get("physical_cap"),
                "bank_write": bool(options.get("bank_write", False)),
            }
            metadata = model_attnres(graph, contract, ub, regbase=self.arch_id == ARCH35, **attn_options)
        else:
            metadata = model_engram(graph, contract, ub)
        if graph.phases and ctx.datapath.usable_paths:
            for source, target in (("gm", "ub"), ("ub", "gm")):
                if not ctx.datapath.has_path(source, target):
                    raise gap("hyper_connections_datapath_unsupported", f"{source}->{target}")
        metadata.update(arch_id=str(self.arch_id), datapath=self.datapath_name,
                        ub_bytes=ub, implementation=kind, governed_family=FAMILY)
        selected_sources = sources_for_kind(kind, self.sources)
        return graph.finish(kind, contract.dtype, selected_sources, metadata)


def build_governed_bindings(
    op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan,
    kind: str, sources: tuple[SourceRef, ...],
) -> tuple[TensorPrecisionABI, KernelBinding, OwnerScheduleSpec]:
    """Construct birth TensorPrecisionABI, KernelBinding, and OwnerScheduleSpec."""
    meta = dict(plan.metadata)
    dtype = str(op.dtype or "bf16").lower()
    arch_str = "A5" if ctx.compute_arch.arch_id == ARCH35 else "A3"
    kernel_map = {
        "hc_pre": ("npu_hc_pre", "CANN", "hyper_connections"),
        "hc_post": ("npu_hc_post", "CANN", "hyper_connections"),
        "hc_pre_sinkhorn": ("aclnnHcPreSinkhorn", "CANN", "hyper_connections"),
        "mhc_post_collapse": ("mhc_post_collapse_norm", "CANN", "hyper_connections"),
        "text_gated_residual": ("aclnnTextGatedResidual", "CANN", "hyper_connections"),
        "attn_res_fwd": ("AttnResFwd", "CANN", "attention_residual"),
        "engram_gate": ("fused_engram_gate", "Triton", "engram_gate"),
    }
    kernel_id, backend, submodule = kernel_map.get(kind, (kind, "CANN", FAMILY))
    precision_abi = TensorPrecisionABI(
        linear_method="dense" if plan.work.matrix else "none",
        input_dtype=dtype,
        weight_dtype=dtype,
        output_dtype=dtype,
        compute_dtype="fp32" if kind == "attn_res_fwd" else dtype,
        accum_dtype="fp32",
    )
    kernel_binding = KernelBinding(
        kernel_id=kernel_id,
        backend=backend,
        arch_family=arch_str,
        submodule=submodule,
        macro_module="residual",
        readiness="shipping",
        source_ref=sources[0] if sources else None,
        recipe_id=kind,
        template_id=str(meta.get("selected_template") or ""),
    )
    owner_schedule = _owner_schedule_from_plan(op, ctx, plan, kind, meta)
    return precision_abi, kernel_binding, owner_schedule


def _owner_schedule_from_plan(
    op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan,
    kind: str, meta: Mapping[str, Any],
) -> OwnerScheduleSpec:
    shape = op.local_shape or op.semantic_shape
    rows = int(shape[0]) if shape else 0
    total_aiv = max(0, int(ctx.effective_aiv_cores))
    total_aic = max(0, int(ctx.effective_aic_cores))
    uses_aic = bool(plan.work.matrix)
    owner_kind: Literal["aic", "aiv", "mix"] = "mix" if uses_aic and total_aiv else ("aic" if uses_aic else "aiv")
    total_owners = total_aiv if owner_kind != "aic" else total_aic
    launched = min(rows, total_owners) if rows > 0 and total_owners > 0 else 0
    max_tiles = _ceil(rows, launched) if launched > 0 else 0
    active = _ceil(rows, max_tiles) if max_tiles > 0 else 0
    ub_cap = int(meta.get("ub_bytes") or dict(ctx.datapath.buffers).get("ub", 0) or 0)
    res_map = plan.work.resources
    mte2_total = float(res_map.get("mte2_aiv", 0.0) + res_map.get("mte2_aic", 0.0) + res_map.get("mte2", 0.0))
    mte3_total = float(res_map.get("mte3", 0.0) + res_map.get("fixpipe", 0.0))
    hbm_total = float(res_map.get("hbm", 0.0))
    divisor = float(max(1, active))
    residency = OwnerMemoryResidency(
        ub_live_bytes=min(ub_cap, int(meta.get("ub_bytes_per_owner", 0) or 0)),
        ub_capacity_bytes=ub_cap,
        spill_reload_passes=2 if str(meta.get("selected_template", "")).endswith("reload") else (1 if rows > 0 else 0),
        owner_mte2_bytes=mte2_total / divisor,
        owner_mte3_bytes=mte3_total / divisor,
        shared_hbm_read_bytes=min(hbm_total, mte2_total),
        shared_hbm_write_bytes=min(hbm_total, mte3_total),
        l2_admission_mode="reuse",
    )
    serial_phases: list[SerialPhaseSpec] = []
    if plan.pipeline_spec is not None:
        for phase in plan.pipeline_spec.phases[:MAX_OWNER_SERIAL_PHASES]:
            vec = sum(item.amount for item in phase.work if item.resource == "vector")
            cube = sum(item.amount for item in phase.work if item.resource == "cube")
            if cube > 0.0:
                serial_phases.append(SerialPhaseSpec(phase.phase_id, "cube", cube / divisor, "flop"))
            elif vec > 0.0:
                serial_phases.append(SerialPhaseSpec(phase.phase_id, "vec_alu", vec / divisor, "op"))
    return OwnerScheduleSpec(
        owner_kind=owner_kind,
        tiling_strategy=str(meta.get("selected_template") or "row_owner"),
        parallel_axis="token_rows",
        work_tiles=rows,
        launched_owners=launched,
        active_owners=active,
        total_owners=total_owners,
        max_tiles_per_owner=max_tiles,
        residency=residency,
        serial_phases=tuple(serial_phases),
        dispatch_envelope_source="P15_dispatch",
    )


def operator_contract(op: OpSpec, kind: str):
    """Decode local tensors and the established flat and nested caller metadata."""
    meta = dict(op.metadata)
    for key in ("fusion_tensor_view", "hc_geometry", "attn_res_view", "contract"):
        if isinstance(view := meta.get(key), Mapping):
            meta.update(view)
    shape = op.local_shape or op.semantic_shape
    meta.setdefault("dtype", op.dtype or "bf16")
    rank = RESIDUAL_RANK if kind in MHC_KINDS or kind == "text_gated_residual" else 2
    if kind == "engram_gate":
        if len(shape) not in {2, RESIDUAL_RANK}:
            raise ModelingError(f"engram_gate_requires_rank2_or_rank3:{op.op_id}")
    elif len(shape) != rank:
        raise ModelingError(f"hyper_connections_requires_rank{rank}:{op.op_id}")
    options = {}
    if rank == RESIDUAL_RANK:
        m, c, h = shape
        if kind == "text_gated_residual":
            schema = TextGatedResidualContract
            # Explicit channels identify the native [B,S,D] input.
            channels = meta.get("hc_mult", meta.get("channels", c))
            if "channels" in meta or channels != c:
                _integer("channels", channels, 1)
                if h % channels:
                    raise ModelingError("text_gated_residual_d_dim_not_divisible_by_channels")
                m, c, h = m * c, channels, h // channels
            meta.setdefault("rank", meta.get("hc_lowrank"))
            meta.setdefault("d_dim", c * h)
            if op.recipe in LEGACY_ALIASES:
                meta = {"stack_binding": "ez1008", "has_gate": True, **meta}
        else:
            schema = MhcContract
            meta["has_collapse"] = kind == "mhc_post_collapse" or meta.get("has_collapse", False)
            if kind == "hc_pre_sinkhorn":
                meta["split_k"] = 1
            if kind == "hc_pre":
                exposed = set(meta.get("fusion_external_outputs", ()))
                for name, phase in zip(meta.get("fusion_semantic_ids", ()), meta.get("fusion_phases", ())):
                    if name in exposed and phase["recipe"] not in {"hc_sinkhorn", "hc_pre_mix"}:
                        raise ModelingError("hc_pre_source_cannot_export_internal_intermediate")
        meta.update(rows=m, hc_mult=c, hidden_size=h)
    else:
        meta.update(tokens=shape[0], hidden_size=shape[-1])
        if kind == "attn_res_fwd":
            schema = AttnResContract
            meta.setdefault("capacity", meta.get("bank_capacity"))
            meta.setdefault("valid_history", meta.get("valid_blocks", 0) if meta.get("mix", True) else 0)
            meta.setdefault("token_stride", meta.get("block_token_stride"))
            options = {"physical_cap": meta.get("physical_capacity_cap"),
                       "bank_write": meta.get("block_write_idx", -1) >= 0,
                       "save_materialized": bool(meta.get("save_materialized", False))}
        else:
            schema = EngramGateContract
            meta["branches"] = shape[1] if len(shape) == RESIDUAL_RANK else 1
            if (mask := meta.get("active_mask")) is not None:
                if len(mask) != shape[0] or any(value not in (0, 1) for value in mask):
                    raise ModelingError("engram_gate_invalid_active_mask")
                count = sum(mask)
                if meta.setdefault("active_tokens", count) != count:
                    raise ModelingError("engram_gate_active_mask_count_mismatch")
    return schema(**{item.name: meta[item.name] for item in fields(schema) if item.name in meta}), options
