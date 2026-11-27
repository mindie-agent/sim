"""Shared linear-attention tensor contracts and conserved work.

[Role] Derive WY work, state residency, convolution, and speculative replay.
[Invariants] Physical transfer bytes and logical record widths are accounted
  independently. Per-owner state liveness is independent of sequence length.
[Timeouts & Locks] Shape arithmetic and local admission simulation only.
[Side-Effect Policy] No device, filesystem, or caller-state mutations.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
import math
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import ArchId, DatapathCapabilitySpec, SourceRef
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
)
from mindie_sim.schema.workload import RequestView
from mindie_sim.templates.dsl_catalog import KDA_AIC_CORE_LIMIT as KDA_MAX_AIC_CORES
from mindie_sim.templates.dsl_catalog import KDA_DEFAULT_WORKSPACE_SLOTS as KDA_WORKSPACE_SLOTS
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, Program, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.l2_residency import AdmissionMemoryView
from ...runtime.memory_physics import AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey, VisibilityContract
from ...runtime.tile_solver import LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape, solve
from ..contracts import OperatorPlanningContext
from .gdn_domain import GDN_CHUNK32_TOKENS, GDN_CHUNK64_GQA_RATIOS, GDN_CHUNK64_TOKENS

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/chunk_gated_delta_rule/op_host/op_api/aclnn_chunk_gated_delta_rule.h",
        source_sha256="5cf63c099df4280a3874e9c547ce6ec2ddf2bf6733df4909749d68a1345d785c",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_def.cpp",
        source_sha256="fe3963e93efcfe41eed696f4709462caa8862b413067cbc5d73d45393c63068b",
    ),
)

FP32_BYTES = 4
MASK_OPS_PER_PAIR = 4
DIAG_BLOCK_WIDTH = GDN_CHUNK32_TOKENS
DEFAULT_VALUE_TILE = 128
DEFAULT_KEY_TILE = 64
NATIVE_CHUNK_GDN_MAX_HEADS = 64
NATIVE_CHUNK_GDN_MAX_HEAD_DIM = 128
NATIVE_RECURRENT_GDN_MAX_HEADS = 256
NATIVE_RECURRENT_GDN_MAX_HEAD_DIM = 512
NATIVE_RECURRENT_GDN_MAX_SEQ_LEN = 8
_ARCH22_ID = ArchId("npu_arch_2201")
_ARCH35_ID = ArchId("npu_arch_3510")
_ARCH_ID_TO_SOC: dict[ArchId, Literal["A2", "A3", "A5"]] = {
    _ARCH22_ID: "A2",
    _ARCH35_ID: "A5",
}
_SOC_DEFAULT_SKU: dict[str, str] = {
    "A2": "Ascend910B1",
    "A3": "Ascend910_9391",
    "A5": "Ascend950DT_9581",
}
REPLAY_OPS_PER_STATE_ELEMENT = 4
SILU_OPS_PER_ELEMENT = 4
CONV_CHANNEL_TILE_WIDTH = 256
SYNC_SCRATCH_BYTES = 32
ACCEPTANCE_WORD_BYTES = 4
CHUNK32_KERNELS = (
    "gdn_prepare_arch35_chunk32", "gdn_state_arch35_chunk32", "gdn_output_arch35_chunk32",
)
CHUNK32_ABIS = frozenset({
    "gdn_chunk32", *CHUNK32_KERNELS, "gdn_chunk32_prepare", "gdn_chunk32_state", "gdn_chunk32_output",
})
CHUNK32_LEAVES = dict(zip(
    ("gdn_chunk32_prepare", "gdn_chunk32_state", "gdn_chunk32_output"), CHUNK32_KERNELS, strict=True,
))
CANN_GDN_ABI = "aclnnChunkGatedDeltaRule"
FLA_GDN_ABIS = frozenset({"fla_chunk_gdn", "chunk_state_bank"})
FLASH_KDA_TEMPLATE_BY_ABI = {"flash_kda_persistent": "persistent", "flash_kda_staged": "staged"}
ABI_OPERATIONS = {
    **dict.fromkeys(CHUNK32_ABIS, "gdn_chunk32"),
    **dict.fromkeys(FLA_GDN_ABIS | {CANN_GDN_ABI, "aclnnChunkGatedDeltaRuleArch22"}, "gdn_prefill"),
    "aclnnChunkGatedDeltaRuleArch35": "gdn_prefill", "gdn_prefill_arch35": "gdn_prefill",
    "aclnnRecurrentGatedDeltaRule": "gdn_recurrent",
    **dict.fromkeys(FLASH_KDA_TEMPLATE_BY_ABI, "kda_prefill"),
    "flash_kda": "kda_prefill", "flash_kda_arch35": "kda_prefill",
    "vllm_ascend.ops.triton.kda.kda.chunk_kda": "kda_prefill", "triton_chunk_kda": "kda_prefill",
    "fused_recurrent_kda": "kda_recurrent", "triton_recurrent_kda": "kda_recurrent",
    "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda": "kda_recurrent",
    "causal_conv1d_update": "causal_conv1d", "commit_recurrent_kda": "recover_ssm_commit",
}
OPERATIONS = frozenset({
    "gdn_prefill", "gdn_chunk32", "kda_prefill", "gdn_recurrent", "kda_recurrent",
    "causal_conv1d", "recover_ssm_verify", "recover_ssm_commit",
})
RECOVER_SSM_FAMILIES = ("kda", "gdn", "deltanet", "gla", "lightning", "retnet", "mamba")
RECTANGULAR_LAYOUTS = frozenset({"BSND", "BNSD", "BSH"})
def ceil_div(value: int, width: int) -> int:
    return (value + width - 1) // width
def _positive(*values: int) -> None:
    if any(type(value) is not int or value < 1 for value in values):
        raise ModelingError("linear_attention_requires_positive_integer_geometry")
def _lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(lengths, (list, tuple)) or not lengths:
        raise ModelingError("linear_attention_requires_sequence_lengths")
    _positive(*lengths)
    return tuple(lengths)

def request_lengths(requests: tuple[RequestView, ...]) -> tuple[int, ...]:
    for view in requests:
        if not view.padding:
            _positive(view.logical_q_len, view.kv_len)
    return tuple(view.logical_q_len for view in requests)


def executed_request_lengths(requests: tuple[RequestView, ...], declared: Sequence[int] = ()) -> tuple[int, ...]:
    """Per-request rows the lowered op executes for the execution request views.

    Execution views stay logical (``q_len`` plus ``draft_proposal_tokens``)
    while a target-verify graph is lowered over ``q_len + draft_proposal_tokens``
    rows. When the op's own lowered active rows equal that exact verify
    projection the verify rows execute; otherwise the logical rows do.
    """
    raw = request_lengths(requests)
    verify = tuple(q + view.draft_proposal_tokens if q else 0 for q, view in zip(raw, requests))
    active = tuple(int(q) for q in declared if q)
    if verify != raw and active and tuple(q for q in verify if q) == active:
        return verify
    return raw


def request_layout_metadata(ctx: OperatorPlanningContext) -> dict[str, Any]:
    """Project the GM bases in each tensor's row units from the request views."""
    if not ctx.requests:
        return {}
    rectangular = ctx.layout.upper() in RECTANGULAR_LAYOUTS
    q_sizes, kv_sizes, kv_offsets, q_ranges = [], [], [], []
    q_base = kv_base = 0
    for view in ctx.requests:
        q_len = view.logical_q_len
        q_size = view.physical_q_len if rectangular else q_len
        kv_size = view.physical_kv_len if rectangular else q_len
        if not view.padding:
            kv_offsets.append(kv_base)
            q_ranges.append((q_base, q_base + q_len))
        q_sizes.append(q_size)
        kv_sizes.append(kv_size)
        q_base += q_size
        kv_base += kv_size
    return {
        "sequence_layout": ctx.layout,
        "logical_sequence_tokens": tuple(view.logical_q_len for view in ctx.requests),
        "physical_query_rows": max(ctx.padded_tokens, q_base), "physical_kv_rows": kv_base,
        "physical_sequence_tokens": tuple(q_sizes), "physical_sequence_kv_tokens": tuple(kv_sizes),
        "sequence_kv_lengths": tuple(view.kv_len for view in ctx.requests),
        "physical_row_offsets": tuple(start for start, _ in q_ranges),
        "physical_kv_row_offsets": tuple(kv_offsets), "input_row_ranges": tuple(q_ranges),
    }

def element_bytes(dtype: str) -> int:
    width = dtype_bytes(dtype)
    if not width.is_integer():
        raise ModelingError(f"linear_attention_requires_byte_addressable_dtype:{dtype}")
    return int(width)


def _validate_flags(*flags: bool) -> None:
    if any(type(flag) is not bool for flag in flags):
        raise ModelingError("linear_attention_requires_boolean_flags")

@dataclass(frozen=True, slots=True)
class LinearAttentionContract:
    operation: str
    lengths: tuple[int, ...]
    hq: int
    hk: int
    hv: int
    dk: int
    dv: int
    bt: int = GDN_CHUNK64_TOKENS
    bv: int = DEFAULT_VALUE_TILE
    bk: int = DEFAULT_KEY_TILE
    dtype: str = "bf16"
    state_dtype: str = "fp32"
    stored_chunk_state_dtype: str = "bf16"
    gate_dtype: str = "fp32"
    beta_dtype: str = "bf16"
    use_initial_state: bool = True
    output_final_state: bool = True
    save_intermediate_state: bool = False
    save_matrices: bool = False
    abi: str = ""
    kda_template_request: str = "auto"
    l2_capacity_bytes: int | None = None
    active_l2_streams: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allow_empty: InitVar[bool] = False

    def __post_init__(self, allow_empty: bool) -> None:
        if self.lengths or not allow_empty:
            object.__setattr__(self, "lengths", _lengths(self.lengths))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        _positive(self.hq, self.hk, self.hv, self.dk, self.dv, self.bt, self.bv, self.bk,
                  self.active_l2_streams)
        pinned = self.abi in CHUNK32_ABIS or self.metadata.get("chunk32") is True
        if pinned and (self.hv % self.hk or self.hv // self.hk not in GDN_CHUNK64_GQA_RATIOS):
            raise ModelingError("gdn_chunk32_unsupported_value_to_key_head_ratio")
        if self.hv % self.hk or self.hq % self.hk:
            raise ModelingError("linear_attention_head_groups_not_divisible")
        if self.operation not in OPERATIONS - {"causal_conv1d"}:
            raise ModelingError(f"linear_attention_operation_unsupported:{self.operation}")
        if self.bt == DIAG_BLOCK_WIDTH and not pinned:
            raise ModelingError("linear_attention_chunk32_requires_pinned_abi")
        if (self.bt not in (GDN_CHUNK64_TOKENS, DIAG_BLOCK_WIDTH)
                or pinned and (self.bt != DIAG_BLOCK_WIDTH
                               or self.operation not in {"gdn_chunk32", "gdn_prefill"})
                or self.operation == "gdn_chunk32" and not pinned):
            raise ModelingError("linear_attention_chunk_size_unsupported")
        if self.kda_template_request not in {"auto", "staged", "persistent"}:
            raise ModelingError("flash_kda_template_unsupported")
        capacity = self.l2_capacity_bytes
        if capacity is not None and (type(capacity) is not int or capacity < 0):
            raise ModelingError("linear_attention_invalid_l2_capacity")
        for dtype in (self.dtype, self.state_dtype, self.stored_chunk_state_dtype,
                      self.gate_dtype, self.beta_dtype):
            element_bytes(dtype)
        _validate_flags(self.use_initial_state, self.output_final_state,
                        self.save_intermediate_state, self.save_matrices)

@dataclass(frozen=True, slots=True)
class CausalConvContract:
    channels: int
    width: int
    lengths: tuple[int, ...]
    mode: Literal["varlen", "incremental"] = "varlen"
    dtype: str = "bf16"
    state_dtype: str = "bf16"
    has_bias: bool = False
    activation: str = "none"
    write_state: bool = True
    speculative_width: int = 1
    state_len: int | None = None
    state_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengths", _lengths(self.lengths))
        _positive(self.channels, self.width, self.speculative_width)
        history = self.width - 1
        state_len = history if self.state_len is None else self.state_len
        offsets = tuple(self.state_offsets) or (0,) * len(self.lengths)
        if type(state_len) is not int or state_len < history:
            raise ModelingError("causal_conv1d_state_len_unsupported")
        if len(offsets) != len(self.lengths) or any(
            type(offset) is not int or not 0 <= offset <= state_len - history for offset in offsets
        ):
            raise ModelingError("causal_conv1d_state_offset_out_of_bounds")
        object.__setattr__(self, "state_len", state_len)
        object.__setattr__(self, "state_offsets", offsets)
        _validate_flags(self.has_bias, self.write_state)
        element_bytes(self.dtype)
        element_bytes(self.state_dtype)
        if self.mode not in {"varlen", "incremental"} or self.activation not in {"silu", "none"}:
            raise ModelingError("causal_conv1d_mode_or_activation_unsupported")

def validate_contract(op: OpSpec, ctx: OperatorPlanningContext) -> LinearAttentionContract | CausalConvContract:
    metadata = dict(op.metadata)
    abi = ctx.requested_abi or metadata.get("abi") or next(
        (name for name in (op.recipe, op.family) if name in ABI_OPERATIONS), "",
    )
    operation = metadata.get("operation") or ABI_OPERATIONS.get(op.recipe, op.recipe)
    if operation not in OPERATIONS:
        operation = ABI_OPERATIONS.get(op.family, op.family)
    if operation not in OPERATIONS:
        operation = ABI_OPERATIONS.get(abi, operation)
    if operation == "gdn_prefill" and abi in CHUNK32_ABIS:
        operation = "gdn_chunk32"
    if abi and ABI_OPERATIONS.get(abi, abi) != operation and not (
        abi in CHUNK32_ABIS and operation == "gdn_prefill"
    ):
        raise ModelingError(f"linear_attention_abi_operation_mismatch:{abi}:{operation}")

    use_requests = bool(ctx.requests) and (operation.endswith(("_prefill", "_recurrent"))
                                          or operation == "gdn_chunk32")
    if use_requests:
        declared = metadata.get("lengths", metadata.get("sequence_lengths", metadata.get("sequence_tokens", ())))
        lengths = tuple(length for length in executed_request_lengths(ctx.requests, declared) if length)
    else:
        raw_lengths = metadata.get("lengths", metadata.get("sequence_lengths", metadata.get("sequence_tokens", ())))
        if not raw_lengths:
            views = getattr(ctx, "requests", ()) or getattr(ctx, "request_views", ()) or ()
            raw_lengths = tuple(
                int(getattr(v, "logical_q_len", getattr(v, "q_len", 0)) or 0)
                for v in views
                if not getattr(v, "padding", False)
                and int(getattr(v, "logical_q_len", getattr(v, "q_len", 0)) or 0) > 0
            )
        lengths = _lengths(raw_lengths)
    dtype = metadata.get("dtype", op.dtype or "bf16")
    if operation == "causal_conv1d":
        channels, width = metadata.get("channels"), metadata.get("width", metadata.get("taps"))
        _positive(channels, width)
        weights = metadata.get("weight_shape", (channels, width))
        if not isinstance(weights, (tuple, list)) or tuple(weights) != (channels, width):
            raise ModelingError("causal_conv1d_weight_or_history_shape_mismatch")
        state_len = metadata.get("state_len", metadata.get("state_history_slots"))
        history = metadata.get("history_shape")
        indices = metadata.get("cache_index", metadata.get("cache_indices"))
        if history is not None:
            if (not isinstance(history, (tuple, list)) or len(history) not in (2, 3)
                    or history[-1] != channels
                    or (len(history) == 3 and (type(history[0]) is not int or history[0] < 1))
                    or (len(history) == 3 and indices is None and history[0] < len(lengths))
                    or (state_len is not None and history[-2] != state_len)):
                raise ModelingError("causal_conv1d_weight_or_history_shape_mismatch")
            state_len = history[-2]
        if indices is not None:
            indices = indices if isinstance(indices, (tuple, list)) else (indices,)
            cache_rows = history[0] if history is not None and len(history) == 3 else len(lengths)
            if len(indices) != len(lengths) or any(
                i is not None and (type(i) is not int or not 0 <= i < cache_rows) for i in indices
            ):
                raise ModelingError("causal_conv1d_cache_index_out_of_bounds")
        accepted = metadata.get("accepted_tokens")
        offsets = metadata.get("state_offset", accepted if accepted is not None else 0)
        offsets = tuple(offsets) if isinstance(offsets, (tuple, list)) else (offsets,) * len(lengths)
        if "state_offset" not in metadata and accepted is not None:
            if any(type(n) is not int or n < 0 for n in offsets):
                raise ModelingError("causal_conv1d_invalid_accepted_tokens")
            offsets = tuple(max(0, n - 1) for n in offsets)
        return CausalConvContract(
            channels, width, lengths,
            mode=metadata.get("mode", "incremental" if abi == "causal_conv1d_update" else "varlen"),
            dtype=dtype, state_dtype=metadata.get("state_dtype", dtype),
            has_bias=metadata.get("has_bias", False), activation=metadata.get("activation", "none"),
            write_state=metadata.get("write_state", True), state_len=state_len, state_offsets=offsets,
            speculative_width=metadata.get("speculative_width", metadata.get(
                "q", max(lengths) if accepted is not None or "state_offset" in metadata else 1
            )),
        )

    pinned = abi in CHUNK32_ABIS or metadata.get("chunk32") is True
    hv = metadata.get("hv", metadata.get("heads", 1))
    hk = next((metadata[key] for key in ("key_heads", "hk", "num_k_heads", "q_heads") if key in metadata), hv)
    hq = metadata.get("hq", metadata.get("q_heads", hk))
    dk = metadata.get("dk", metadata.get("head_dim", DEFAULT_VALUE_TILE))
    dv = metadata.get("dv", metadata.get("value_dim", dk))
    fields = {key: metadata[key] for key in (
        "state_dtype", "stored_chunk_state_dtype", "gate_dtype", "beta_dtype",
        "use_initial_state", "output_final_state", "save_intermediate_state", "save_matrices",
        "kda_template_request", "l2_capacity_bytes", "active_l2_streams",
    ) if key in metadata}
    if operation == "kda_prefill":
        for alias, field_name in (("has_initial_state", "use_initial_state"), ("return_final_state", "output_final_state"),
                                  ("save_chunk_states", "save_intermediate_state")):
            if alias in metadata:
                fields[field_name] = metadata[alias]
    if (
        "state_dtype" not in fields
        and operation == "gdn_prefill"
        and abi not in FLA_GDN_ABIS
        and ctx.compute_arch.arch_id == _ARCH22_ID
    ):
        fields["state_dtype"] = "bf16"
    bt = metadata.get("bt", metadata.get("chunk_size", DIAG_BLOCK_WIDTH if pinned else GDN_CHUNK64_TOKENS))
    return LinearAttentionContract(
        operation, lengths, hq, hk, hv, dk, dv, bt,
        metadata.get("bv", DEFAULT_VALUE_TILE), metadata.get("bk", DEFAULT_KEY_TILE),
        dtype, abi=abi, metadata=metadata, allow_empty=use_requests, **fields,
    )


@dataclass(frozen=True, slots=True)
class WyChunkWork:
    kkt_flops: int
    qkt_flops: int
    offdiag_inv_flops: int
    kcd_flops: int
    u_flops: int
    r_flops: int
    o_inter_flops: int
    delta_flops: int
    o_intra_flops: int
    gate_ops: int
    mask_ops: int
    diag_inv_ops: int
    cast_ops: int
    state_update_ops: int

    @property
    def vector_ops(self) -> int:
        return self.gate_ops + self.mask_ops + self.diag_inv_ops + self.cast_ops + self.state_update_ops

def compute_wy_chunk_work(c: int, dk: int, dv: int, bt: int = GDN_CHUNK64_TOKENS) -> WyChunkWork:
    _positive(c, dk, dv, bt)
    if c > bt or bt not in (DIAG_BLOCK_WIDTH, GDN_CHUNK64_TOKENS):
        raise ModelingError("linear_attention_invalid_wy_chunk")
    blocks = (min(c, DIAG_BLOCK_WIDTH), max(0, c - DIAG_BLOCK_WIDTH))
    diagonal = sum(2 * sum(row * (row - 1) // 2 for row in range(width)) for width in blocks)
    strict_lower = c * (c - 1) // 2
    # cumsum, expG, beta/exp and Q scale/exp products, Kg exponent then scale.
    gates = c - 1 + c + 2 * c + (2 * c * dk + c * dv + c * (2 + dk))
    masks = MASK_OPS_PER_PAIR * (2 * strict_lower + c)
    score = 2 * c * c * dk
    value = 2 * c * c * dv
    cross = 2 * c * dk * dv
    inverse = 2 * (2 * DIAG_BLOCK_WIDTH * DIAG_BLOCK_WIDTH**2) if c > DIAG_BLOCK_WIDTH else 0
    return WyChunkWork(score, score, inverse, score, value, cross, cross, cross, value,
                       gates, masks, diagonal, c * dv, 2 * dk * dv)


@dataclass(frozen=True, slots=True)
class StateResidencyReport:
    total_chunks: int
    state_tile_bytes: int
    live_copies: int
    owner_ub_bytes: int
    chunk_bank_bytes: int
    l2_hit_bytes: int
    dram_miss_bytes: int
    evicted_chunks: int
    compulsory_store_bytes: int
    reload_bytes: int
    persistent_ub_resident: bool
    l1_state_ring_bytes: int = 0

    @property
    def staged_chunk_bank_bytes(self) -> int:
        return self.chunk_bank_bytes

def analyze_state_residency(
    contract: LinearAttentionContract, datapath: DatapathCapabilitySpec | OperatorPlanningContext,
    *, persistent: bool = False, live_copies: int = 1,
    qkv_gate_correction_scratch: int | None = None, sync_scratch: int = SYNC_SCRATCH_BYTES,
    l1_state_ring_bytes: int = 0,
) -> StateResidencyReport:
    ctx = datapath if isinstance(datapath, OperatorPlanningContext) else None
    path = ctx.datapath if ctx is not None else datapath
    if live_copies not in (1, 2) or type(live_copies) is not int:
        raise ModelingError("linear_attention_invalid_state_copies")
    tile = FP32_BYTES * contract.bv * contract.bk
    scratch = qkv_gate_correction_scratch
    if scratch is None:
        gate_width = contract.bk if contract.operation.startswith("kda_") else 1
        # Ascend AIV decode recurrences upcast BF16/FP16 Q/K/V/gate/beta inputs
        # into FP32 (4B/elem) UB working vectors; chunked prefill Q/K/V stay
        # in Cube L1 so only recurrent ops budget 4B/elem UB scratch here.
        is_recurrent = contract.operation.endswith("_recurrent")
        qkv_elem = max(FP32_BYTES, element_bytes(contract.dtype)) if is_recurrent else element_bytes(contract.dtype)
        gate_elem = max(FP32_BYTES, element_bytes(contract.gate_dtype)) if is_recurrent else element_bytes(contract.gate_dtype)
        beta_elem = max(FP32_BYTES, element_bytes(contract.beta_dtype)) if is_recurrent else element_bytes(contract.beta_dtype)
        scratch_tokens = min(max(contract.lengths, default=1), contract.bt) if is_recurrent else contract.bt
        scratch = scratch_tokens * (
            (2 * contract.bk + contract.bv) * qkv_elem
            + gate_width * gate_elem + beta_elem
            + FP32_BYTES * contract.bv
        )
    chunks = sum(ceil_div(length, contract.bt) for length in contract.lengths)
    buffers = (
        LiveBufferSpec("fp32_state", "ub", tile, 0, 1, owner_kind="aiv", slots=live_copies),
        LiveBufferSpec("operands_and_correction", "ub", scratch, 0, 1, owner_kind="aiv"),
        LiveBufferSpec("sync", "ub", sync_scratch, 0, 1, owner_kind="aiv"),
    )
    capacities = {"ub": path.usable_bytes("ub")}
    if l1_state_ring_bytes:
        buffers += (LiveBufferSpec("state_ring", "l1", l1_state_ring_bytes, 0, 1, owner_kind="aic"),)
        capacities["l1"] = path.usable_bytes("l1")
    candidate = TileCandidate(
        "chunk_recurrence", TileShape(contract.bt, contract.bv, contract.bk), live_copies,
        (path.aic_count, path.aiv_count), len(contract.lengths) * contract.hv, 1,
        buffers, owner_axis="aiv",
    )
    hw = SimpleNamespace(buffers=capacities, datapath=path, stream_owners=ctx.stream_owners if ctx else None)
    solve(TilePipelineSpec("linear_attention", (candidate,)), None, hw)
    ub = live_copies * tile + scratch + sync_scratch
    if persistent or contract.operation not in {"gdn_prefill", "gdn_chunk32", "kda_prefill"}:
        return StateResidencyReport(chunks, tile, live_copies, ub, 0, 0, 0, 0, 0, 0,
                                    persistent, l1_state_ring_bytes)

    head_bytes = contract.dv * contract.dk * element_bytes(contract.stored_chunk_state_dtype)
    chunk_bytes = contract.hv * head_bytes
    size = chunks * chunk_bytes
    capacity = (path.usable_bytes("l2") if contract.l2_capacity_bytes is None else contract.l2_capacity_bytes)
    admission = ctx.admission_memory_view if ctx is not None else None
    die = (ctx.occurrence.rank or 0) if ctx is not None else 0
    units = ctx.geometry.buffer_units.get("l2", 1) if ctx else 1
    allocation = "chunk_state_bank" if ctx is None else f"{ctx.occurrence.op_id}:chunk_state_bank"
    if admission is not None:
        # Predict the bank's write/read against the current shared occupancy.
        view = deepcopy(admission)
        if (excess := view.available_l2_bytes(die) - capacity) > 0:
            view.grant_reservation(die, f"{allocation}:capacity_limit", excess)
    else:
        view = AdmissionMemoryView({die: capacity}, l2_buffer_units_by_die={die: units})
        for stream in range(contract.active_l2_streams):
            view.enter_stream(die, str(stream))
    key = ReuseKey(die, allocation, 0, "0")
    runs = AddressRunPlan((AddressRun(0, chunks * contract.hv, head_bytes, head_bytes),))
    read = MemoryAccessPattern(allocation, 0, die, 0, "read", "mte2_aic", size, runs, key)
    write = replace(read, direction="write", pipe="mte3",
                    visibility=VisibilityContract(live_out=True))
    aligned = runs.aligned_copy_bytes()
    _, stores = view.admit_access(write, aligned)
    hits, misses = view.preview_access(read, aligned)
    return StateResidencyReport(chunks, tile, live_copies, ub, size, hits, misses,
                                min(chunks, ceil_div(misses, chunk_bytes)), stores, misses, False)


@dataclass(frozen=True, slots=True)
class RankOneTransition:
    family: str
    state_axes: tuple[str, str]
    decay_kind: str
    correction_field_bytes: int
    key_field_bytes: int
    decay_field_bytes: int
    extra_field_bytes: int
    reconstruction_rule: str
    rounding_points: tuple[str, ...]
    record_dtype: str
    state_dtype: str
    kg_dtype: str
    upstream_verified: bool

    @property
    def record_bytes(self) -> int:
        return self.correction_field_bytes + self.key_field_bytes + self.decay_field_bytes + self.extra_field_bytes

def rank_one_transition(family: str, dk: int, dv: int, *, state_dtype: str = "fp32",
                        kg_dtype: str = "bf16", gated: bool = False,
                        factors: Mapping[str, tuple[int, ...]] | None = None) -> RankOneTransition:
    _positive(dk, dv)
    element_bytes(state_dtype)
    packed = element_bytes(kg_dtype)
    _validate_flags(gated)
    if family not in RECOVER_SSM_FAMILIES:
        raise ModelingError(f"recover_ssm_family_unsupported:{family}")
    if factors is not None and (
        family != "mamba" or dict(factors) != {"B": (dk,), "dt": (), "A_diag": (dk,), "C_proj": (dk,)}
    ):
        raise ModelingError("recover_ssm_mamba_requires_selective_rank_one_factors")
    decay_kind, rule = {
        "kda": ("keywise", "exp(gate) * S + correction outer raw_K"),
        "gdn": ("scalar", "decay * S + correction outer normalized_K"),
        "deltanet": ("scalar" if gated else "identity", "decay * S + correction outer K"),
        "gla": ("separable", "g_v[:, None] * S * g_k[None, :] + correction outer K"),
        "lightning": ("head", "head_decay * S + V outer K"),
        "retnet": ("head", "retention_decay * S + projected_V outer projected_K"),
        "mamba": ("diagonal_selective", "exp(dt * A_diag) * S + dt * input outer B; read with C_proj"),
    }[family]
    key, decay, extra = FP32_BYTES * dk, FP32_BYTES, 0
    if family in {"kda", "gla"}:
        key = decay = dk * packed
        if family == "gla":
            extra = dv * packed  # Value-axis decay remains live with a zero correction.
    elif family == "deltanet" and not gated:
        decay = 0
    elif family == "mamba" and gated:
        key, decay, extra = dk * packed, packed, dk * packed
    mixed = family in {"kda", "gla"} or family == "mamba" and gated
    return RankOneTransition(
        family, ("value", "key"), decay_kind, FP32_BYTES * dv, key, decay, extra, rule,
        ("record_store", "checkpoint_publish") if mixed else ("checkpoint_publish",),
        f"fp32+{kg_dtype}" if mixed else "fp32", state_dtype, kg_dtype, family in {"gdn", "kda"},
    )


@dataclass(frozen=True, slots=True)
class RecoverSsmVerifyPlan:
    transition: RankOneTransition
    q: int
    state_read_bytes: int
    record_write_bytes: int
    output_tokens: int
    single_checkpoint: bool


@dataclass(frozen=True, slots=True)
class RecoverSsmCommitPlan:
    transition: RankOneTransition
    q: int
    a: int
    state_read_bytes: int
    record_read_bytes: int
    vector_ops: int
    state_write_bytes: int
    base_version: int
    published_version: int
    skipped_zero_accept: bool
    boundary_checkpoints: tuple[int, ...] = ()

def build_recover_ssm_verify(family: str, dk: int, dv: int, *, q: int, heads: int = 1,
                             sequences: int = 1, state_dtype: str = "fp32",
                             kg_dtype: str = "bf16", gated: bool = False,
                             factors: Mapping[str, tuple[int, ...]] | None = None) -> RecoverSsmVerifyPlan:
    _positive(q, heads, sequences)
    transition = rank_one_transition(family, dk, dv, state_dtype=state_dtype,
                                     kg_dtype=kg_dtype, gated=gated, factors=factors)
    owners = heads * sequences
    return RecoverSsmVerifyPlan(transition, q, owners * dk * dv * element_bytes(state_dtype),
                               owners * q * transition.record_bytes if q > 1 else 0, q * sequences, q == 1)

def build_recover_ssm_commit(family: str, dk: int, dv: int, *, q: int, a: int, heads: int = 1,
                             sequences: int = 1, state_dtype: str = "fp32", kg_dtype: str = "bf16",
                             gated: bool = False, base_version: int = 0,
                             boundary_checkpoints: tuple[int, ...] = (),
                             factors: Mapping[str, tuple[int, ...]] | None = None) -> RecoverSsmCommitPlan:
    _positive(q, heads, sequences)
    if type(a) is not int or not 0 <= a <= q:
        raise ModelingError("recover_ssm_requires_integer_acceptance")
    if type(base_version) is not int or base_version < 0:
        raise ModelingError("recover_ssm_base_version_invalid")
    boundaries = tuple(boundary_checkpoints)
    if any(type(point) is not int or not 1 <= point <= q for point in boundaries) or (
        tuple(sorted(set(boundaries))) != boundaries
    ):
        raise ModelingError("recover_ssm_boundary_checkpoints_invalid")
    verify = build_recover_ssm_verify(family, dk, dv, q=q, heads=heads, sequences=sequences,
                                     state_dtype=state_dtype, kg_dtype=kg_dtype, gated=gated, factors=factors)
    state = verify.state_read_bytes if a else 0
    checkpoints = tuple(point for point in boundaries if point <= a)
    replay_tokens = a * heads * sequences if q > 1 else 0
    if family in {"gdn", "kda"}:
        from ..recurrent.recover_ssm_commit import REVERSE_FMA_OPS_PER_STATE_ELEMENT
        vec_ops = (
            REVERSE_FMA_OPS_PER_STATE_ELEMENT * (a + 1) * heads * sequences * dk * dv
            if (q > 1 and a > 0) else 0
        )
    else:
        vec_ops = (REPLAY_OPS_PER_STATE_ELEMENT + (family == "gla")) * replay_tokens * dk * dv
    return RecoverSsmCommitPlan(
        verify.transition, q, a, state, replay_tokens * verify.transition.record_bytes,
        vec_ops,
        state * (1 + sum(point < a for point in checkpoints)),
        base_version, base_version + int(a > 0), a == 0, checkpoints,
    )

def flash_kda_workspace_groups(
    chunks: Sequence[int], slots_per_head: int, max_group: int,
) -> Iterator[tuple[int, int, tuple[int, ...]]]:
    from mindie_sim.templates.dsl_catalog import flash_kda_rounds

    previous, consumed = -1, 0
    for start, counts in flash_kda_rounds(chunks, 1, slots_per_head, max_group):
        if start != previous:
            previous, consumed = start, 0
        yield start, consumed, counts
        consumed += max(counts)

def select_flash_kda_template(
    lengths: LinearAttentionContract | Sequence[int], *, value_heads: int | None = None,
    query_heads: int | None = None, core_num: int, template_request: str = "auto",
    device_name: str = "", route_coefficients: Sequence[int] = (), layout_kind: int = 0,
    workspace_slots: int = KDA_WORKSPACE_SLOTS, max_chunk_group: int | None = None,
    metadata_word_capacity: int | None = None, has_cu: bool = False,
    calibration_enabled: bool = True, enable_grad: bool = False,
) -> str:
    from mindie_sim.templates.dsl_catalog import resolve_flash_kda_route

    if isinstance(lengths, LinearAttentionContract):
        contract = lengths
        value_heads, query_heads = contract.hv, contract.hq
        if template_request == "auto":
            template_request = FLASH_KDA_TEMPLATE_BY_ABI.get(contract.abi, contract.kda_template_request)
        device_name = device_name or contract.metadata.get("device_name", "")
        lengths = contract.lengths
    _positive(value_heads, core_num)
    return resolve_flash_kda_route(
        lengths, value_heads, core_num, requested_template=template_request, device_name=device_name,
        route_coefficients=route_coefficients, layout_kind=layout_kind, workspace_slots=workspace_slots,
        max_chunk_group=max_chunk_group, metadata_word_capacity=metadata_word_capacity, has_cu=has_cu,
        calibration_enabled=calibration_enabled, enable_grad=enable_grad, query_heads=query_heads,
    )[0]


@dataclass(frozen=True, slots=True)
class Phase:
    name: str
    work: Work
    after: tuple[str, ...] = ()

def make_pipeline_spec(implementation: str, phases: Sequence[Phase],
                       sources: tuple[SourceRef, ...] = (), *, family: str = "linear_attention") -> FusedPipelineSpec:
    return FusedPipelineSpec(family, implementation, tuple(
        PipelinePhase(phase.name, phase.after, (
            *(WorkItem("cube", "flop", amount, dtype) for dtype, amount in phase.work.matrix.items() if amount),
            *(WorkItem(role, "op" if role == "vector" else "bytes", amount)
              for role, amount in phase.work.resources.items() if amount),
        )) for phase in phases
    ), sources)

def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(ctx.occurrence, spec.family, spec.implementation, (op.op_id,),
                                   (), spec.phases, spec.sources, spec.repeats)

def missing(code: str, message: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(capability="linear_attention", entry="linear_attention.build",
                                    code=code, message=message)


def verify_native_gdn_abi(
    c: LinearAttentionContract,
    ctx: OperatorPlanningContext,
    *,
    abi_id: str,
) -> None:
    """Enforce sealed ProbedAbiRegistry dtypes and CANN host tiling bounds for native GDN."""
    from mindie_sim.backends.abi_registry import default_abi_registry
    from mindie_sim.schema.abi import AbiQueryKey, OperandPortSig

    soc_arch = _ARCH_ID_TO_SOC.get(ctx.compute_arch.arch_id)
    if soc_arch is None:
        raise missing("linear_attention_arch_unsupported", f"Unsupported ISA {ctx.compute_arch.arch_id}")
    dtype = c.dtype.lower()
    beta_dtype = c.beta_dtype.lower()
    gate_dtype = c.gate_dtype.lower()
    state_dtype = c.state_dtype.lower()
    tokens = max(1, sum(c.lengths))
    seqs = max(1, len(c.lengths))
    is_chunk = abi_id == CANN_GDN_ABI
    if c.hq != c.hk:
        raise missing(
            "linear_attention_gdn_heads_unsupported",
            f"{abi_id} requires hq == hk (got hq={c.hq}, hk={c.hk})",
        )
    if is_chunk:
        graph_motif = "chunk_gdn"
        operands = (
            OperandPortSig("query", dtype, "ND", False, (tokens, c.hk, c.dk), "dense", (), "none", ()),
            OperandPortSig("key", dtype, "ND", False, (tokens, c.hk, c.dk), "dense", (), "none", ()),
            OperandPortSig("value", dtype, "ND", False, (tokens, c.hv, c.dv), "dense", (), "none", ()),
            OperandPortSig("beta", beta_dtype, "ND", False, (tokens, c.hv), "dense", (), "none", ()),
            OperandPortSig("initial_state", state_dtype, "ND", False, (seqs, c.hv, c.dv, c.dk), "dense", (), "none", ()),
            OperandPortSig("actual_seq_lengths", "int32", "ND", False, (seqs,), "none", (), "none", ()),
        )
        outputs = (
            OperandPortSig("out", dtype, "ND", False, (tokens, c.hv, c.dv), "dense", (), "none", ()),
            OperandPortSig("final_state", state_dtype, "ND", False, (seqs, c.hv, c.dv, c.dk), "dense", (), "none", ()),
        )
    else:
        if max(c.lengths, default=0) > NATIVE_RECURRENT_GDN_MAX_SEQ_LEN:
            raise missing(
                "linear_attention_gdn_seq_len_unsupported",
                f"{abi_id} supports per-request sequence length up to {NATIVE_RECURRENT_GDN_MAX_SEQ_LEN} (got max={max(c.lengths)})",
            )
        state_blocks = int(c.metadata.get("state_blocks", c.metadata.get("state_capacity", tokens)))
        if state_blocks < tokens:
            raise missing(
                "linear_attention_gdn_state_blocks_insufficient",
                f"{abi_id} requires state BlockNum >= T (got BlockNum={state_blocks}, T={tokens})",
            )
        graph_motif = "recurrent_gdn"
        operands = (
            OperandPortSig("query", dtype, "ND", False, (tokens, c.hk, c.dk), "dense", (), "none", ()),
            OperandPortSig("key", dtype, "ND", False, (tokens, c.hk, c.dk), "dense", (), "none", ()),
            OperandPortSig("value", dtype, "ND", False, (tokens, c.hv, c.dv), "dense", (), "none", ()),
            OperandPortSig("beta", beta_dtype, "ND", False, (tokens, c.hv), "dense", (), "none", ()),
            OperandPortSig("state", state_dtype, "ND", False, (state_blocks, c.hv, c.dv, c.dk), "dense", (), "none", ()),
            OperandPortSig("actual_seq_lengths", "int32", "ND", False, (seqs,), "none", (), "none", ()),
            OperandPortSig("ssm_state_indices", "int32", "ND", False, (tokens,), "none", (), "none", ()),
        )
        outputs = (
            OperandPortSig("out", dtype, "ND", False, (tokens, c.hv, c.dv), "dense", (), "none", ()),
            OperandPortSig("state", state_dtype, "ND", False, (state_blocks, c.hv, c.dv, c.dk), "dense", (), "none", ()),
        )
    verdict = default_abi_registry().verify(
        AbiQueryKey(
            soc_arch=soc_arch,
            sku_id=_SOC_DEFAULT_SKU[soc_arch],
            abi_id=abi_id,
            graph_motif=graph_motif,
            model_scope="target",
            draft_quant_mode="none",
            pr_track="shipping",
            operands=operands,
            outputs=outputs,
            rank_placement="local",
            world_size=1,
            comm_group_id="none",
            comm_domain_width=1,
            comm_dtype="none",
            reduction_kind="none",
            activation_kind="none",
            activation_params=(),
            has_bias=False,
            has_active_mask=False,
            dispatch_quant_mode=0,
            shared_experts_per_rank=0,
            top_k=0,
            moe_expert_num=0,
            qk_rope_head_dim=0,
            mla_variant="none",
            sfa_dcp_active=False,
            pcp_size=1,
            kv_cache_dtype="none",
            has_kpool_indexer=False,
            dcp_size=1,
            is_act_and_mul=False,
            enabled_stack_passes=frozenset(),
        )
    )
    if verdict.status != "admitted_shipping":
        raise missing(
            f"linear_attention_native_abi_unsupported:{abi_id}",
            f"{abi_id} rejected on {soc_arch} (dtype={dtype}, state_dtype={state_dtype}): {verdict.status}",
        )
    if gate_dtype != "fp32":
        raise missing(
            f"linear_attention_native_abi_unsupported:{abi_id}",
            f"{abi_id} requires fp32 gate_dtype (got {c.gate_dtype})",
        )
    max_dim = NATIVE_CHUNK_GDN_MAX_HEAD_DIM if is_chunk else NATIVE_RECURRENT_GDN_MAX_HEAD_DIM
    max_heads = NATIVE_CHUNK_GDN_MAX_HEADS if is_chunk else NATIVE_RECURRENT_GDN_MAX_HEADS
    if max(c.dk, c.dv) > max_dim:
        raise missing(
            "linear_attention_gdn_dimension_unsupported",
            f"{abi_id} supports dk and dv up to {max_dim} (got dk={c.dk}, dv={c.dv})",
        )
    if max(c.hk, c.hv) > max_heads:
        raise missing(
            "linear_attention_gdn_heads_unsupported",
            f"{abi_id} supports hk and hv up to {max_heads} (got hk={c.hk}, hv={c.hv})",
        )

def owner_units(ctx: OperatorPlanningContext) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if ctx.stream_owners is not None:
        owners = ctx.stream_owners
        return owners.aic[:owners.aic_limit], owners.aiv[:owners.aiv_limit]
    return tuple(range(ctx.datapath.aic_count)), tuple(range(ctx.datapath.aiv_count))

def bind_phase(phase: Phase, rc: RecipeContext, ctx: OperatorPlanningContext,
               aics: tuple[int, ...], aivs: tuple[int, ...]) -> Stage:
    work = rc.bind(phase.work)
    placement = {}
    for role in (*phase.work.resources, *("cube",) * bool(phase.work.matrix)):
        if role in {"hbm", "l2"}:
            continue
        owned_role = rc.read_pipe(aiv=not phase.work.matrix) if role == "mte2" else role
        physical = rc.physical_role(owned_role)
        name = rc.resource_names.get(physical)
        if name not in work:
            continue
        vector = owned_role in {"vector", "mte2_aiv", "mte3"} or (
            owned_role == "mte2" and not phase.work.matrix
        )
        owners = aivs if vector else aics
        total = ctx.datapath.aiv_count if vector else ctx.datapath.aic_count
        units = rc.resource_units.get(physical, total)
        if not owners:
            raise missing("linear_attention_no_physical_owner", f"No owner for {owned_role}")
        placement[name] = tuple(sorted({u for i in owners for u in unit_group(i, total, units)}))
    return Stage(phase.name, work=work, after=phase.after, placement=placement,
                 keep_empty=not work, metadata={"phase": phase.name})

def make_plan(ctx: OperatorPlanningContext, implementation: str, phases: Sequence[Phase], *,
              sources: tuple[SourceRef, ...] = (), family: str = "linear_attention",
              launch_count: int = 1, metadata: Mapping[str, Any] | None = None,
              owners: int | None = None, program: Callable[[RecipeContext], Program] | None = None) -> OperatorPlan:
    phases = tuple(phases)
    work = Work()
    for phase in phases:
        work += phase.work

    def lower(rc: RecipeContext) -> KernelProgram:
        aics, aivs = owner_units(ctx)
        if owners is not None:
            aics, aivs = aics[:owners], aivs[:owners]
        return KernelProgram(tuple(bind_phase(phase, rc, ctx, aics, aivs) for phase in phases))

    return OperatorPlan(work, program or lower, {
        "implementation": implementation, "arch_id": str(ctx.compute_arch.arch_id),
        "launch_count": launch_count, **(metadata or {}), **request_layout_metadata(ctx),
    }, pipeline_spec=make_pipeline_spec(implementation, phases, sources, family=family))

def transfer(ctx: OperatorPlanningContext, *, read: int = 0, write: int = 0,
             aiv: bool = True, hits: int = 0) -> Work:
    read_pipe = ctx.datapath.route("gm", "ub" if aiv else "l1").path.unit if read else "mte2"
    resources = {"hbm": read + write - hits, read_pipe or "mte2": read, "mte3": write}
    if hits and "l2" in ctx.geometry.resource_units:
        resources["l2"] = hits
    return Work(resources={name: amount for name, amount in resources.items() if amount})

def build_causal_conv1d_plan(contract: CausalConvContract, ctx: OperatorPlanningContext, *,
                             implementation: str = "causal_conv1d", sources: tuple[SourceRef, ...] = (),
                             weight_copies: int = 1) -> OperatorPlan:
    c = contract
    size, stored = element_bytes(c.dtype), element_bytes(c.state_dtype)
    tokens = sum(c.lengths)
    history_rows = c.width - 1
    row_bytes = c.channels * stored
    history = len(c.lengths) * history_rows * row_bytes
    output = tokens * c.channels * size
    raw_meta = getattr(c, "metadata", None)
    raw_tile = raw_meta.get("channel_tile") if isinstance(raw_meta, Mapping) else None
    tile_width = (
        int(raw_tile)
        if isinstance(raw_tile, int) and not isinstance(raw_tile, bool) and raw_tile > 0
        else CONV_CHANNEL_TILE_WIDTH
    )
    channel_tiles = max(1, ceil_div(c.channels, tile_width))
    work_tiles = max(1, len(c.lengths) * channel_tiles)
    workers = max(1, min(ctx.effective_aiv_cores, work_tiles))
    _positive(weight_copies)
    macs = 2 * c.channels * tokens * c.width
    state_writes = history
    if c.mode == "incremental" and c.speculative_width > 1:
        state_writes = row_bytes * sum(min(c.state_len, history_rows + min(n, c.speculative_width) - 1)
                                       for n in c.lengths)
    phases = [
        Phase("input", transfer(ctx, read=output)),
        Phase("history", transfer(ctx, read=history)),
        Phase("weights", transfer(ctx, read=weight_copies * c.channels * c.width * size)),
        Phase("convolve", Work(resources={"vector": macs}), ("input", "history", "weights")),
    ]
    tail = "convolve"
    if c.has_bias:
        phases.extend((
            Phase("bias", transfer(ctx, read=weight_copies * c.channels * size)),
            Phase("add_bias", Work(resources={"vector": tokens * c.channels}), (tail, "bias")),
        ))
        tail = "add_bias"
    if c.activation == "silu":
        phases.append(Phase("silu", Work(resources={"vector": SILU_OPS_PER_ELEMENT * tokens * c.channels}), (tail,)))
        tail = "silu"
    phases.append(Phase("output", transfer(ctx, write=output), (tail,)))
    if c.write_state:
        phases.append(Phase("state_writeback", transfer(ctx, write=state_writes), ("history", "input", "output")))
    total_read = float(output + history + weight_copies * c.channels * c.width * size
                       + (weight_copies * c.channels * size if c.has_bias else 0))
    total_write = float(output + (state_writes if c.write_state else 0))
    ub_raw = dict(getattr(ctx.geometry, "buffers", ())).get("ub")
    ub_cap = int(ub_raw) if ub_raw is not None else 0
    ub_est = max(tile_width * (history_rows + 1) * max(size, stored), SYNC_SCRATCH_BYTES)
    ub_live = min(ub_cap, ub_est) if ub_cap > 0 else ub_est
    owner_schedule = OwnerScheduleSpec(
        owner_kind="aiv",
        tiling_strategy="sequence_channel_tile",
        parallel_axis="sequence_x_channel_tile",
        work_tiles=work_tiles,
        launched_owners=workers,
        active_owners=workers,
        total_owners=max(1, int(ctx.effective_aiv_cores)),
        max_tiles_per_owner=ceil_div(work_tiles, workers),
        residency=OwnerMemoryResidency(
            ub_live_bytes=ub_live,
            ub_capacity_bytes=ub_cap,
            owner_mte2_bytes=total_read / float(workers),
            owner_mte3_bytes=total_write / float(workers),
            shared_hbm_read_bytes=total_read,
            shared_hbm_write_bytes=total_write,
            l2_admission_mode="reuse",
        ),
        serial_phases=(
            SerialPhaseSpec("history_input_load", "mte2", total_read / float(workers), "bytes"),
            SerialPhaseSpec("conv_taps_act", "vec_alu", float(macs) / float(workers), "op"),
            SerialPhaseSpec("output_state_store", "mte3", total_write / float(workers), "bytes"),
        ),
        dispatch_envelope_source="P15_dispatch",
    )
    return make_plan(ctx, implementation, phases, sources=sources, family="causal_conv1d", owners=workers,
                     metadata={"mode": c.mode, "mac_flops": macs, "history_rows": history_rows,
                               "state_len": c.state_len, "state_offsets": c.state_offsets,
                               "speculative_width": c.speculative_width,
                               "channel_tile_width": tile_width, "channel_tiles": channel_tiles,
                               "owner_schedule": owner_schedule.to_dict()})

def _resolve_recover_ssm_window(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> int:
    m = c.metadata
    if "q" in m:
        return m["q"]
    if c.operation == "recover_ssm_verify":
        return max(c.lengths)
    q = max(c.lengths)
    if q > 1:
        return q
    qt = m.get("query_tokens")
    if isinstance(qt, int) and not isinstance(qt, bool) and qt > len(c.lengths):
        return max(q, qt // len(c.lengths))
    proposals = [
        int(getattr(v, "draft_proposal_tokens", 0) or 0)
        for v in (getattr(ctx, "requests", ()) or ())
        if not getattr(v, "padding", False)
    ]
    return max(q, 1 + max(proposals)) if proposals and max(proposals) > 0 else q


def _validate_per_request_acceptance(
    items: Sequence[Any], per_req_bounds: Sequence[int], default_q: int,
) -> tuple[int, ...]:
    if not items:
        raise ModelingError(f"recover_ssm_acceptance_length_mismatch:0:{len(per_req_bounds)}")
    parsed: list[int] = []
    if per_req_bounds and len(items) != len(per_req_bounds):
        raise ModelingError(f"recover_ssm_acceptance_length_mismatch:{len(items)}:{len(per_req_bounds)}")
    use_per_req = len(items) == len(per_req_bounds)
    for idx, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelingError("recover_ssm_requires_integer_acceptance")
        bound = int(per_req_bounds[idx]) if use_per_req else int(default_q)
        if item < 0 or item > bound:
            raise ModelingError(f"recover_ssm_acceptance_out_of_range:{item}:{bound}")
        parsed.append(int(item))
    return tuple(parsed)


def _resolve_recover_ssm_accepted_per_request(
    m: Mapping[str, Any], ctx: OperatorPlanningContext, q: int, sequences: int,
    per_req_bounds: Sequence[int] = (),
) -> tuple[int, ...] | None:
    """Normalize every acceptance spelling into one validated integer tuple (never averaged or rounded)."""
    seq_count = len(per_req_bounds) if per_req_bounds else max(1, int(sequences))
    if "accepted_per_request" in m and m["accepted_per_request"] is not None:
        raw_vec = m["accepted_per_request"]
        if not isinstance(raw_vec, (tuple, list)):
            raise ModelingError("recover_ssm_requires_integer_acceptance")
        return _validate_per_request_acceptance(raw_vec, per_req_bounds, q)
    raw_a = m.get("a", m.get("accepted", m.get("accepted_tokens")))
    if raw_a is not None:
        if isinstance(raw_a, (tuple, list)):
            return _validate_per_request_acceptance(raw_a, per_req_bounds, q)
        if isinstance(raw_a, bool) or not isinstance(raw_a, int):
            raise ModelingError("recover_ssm_requires_integer_acceptance")
        return _validate_per_request_acceptance((int(raw_a),) * seq_count, per_req_bounds, q)
    if "committed_tokens" in m and m["committed_tokens"] is not None:
        committed = m["committed_tokens"]
        if not isinstance(committed, (tuple, list)):
            raise ModelingError("recover_ssm_requires_integer_acceptance")
        if not committed and type(m.get("active_sequences")) is int and m["active_sequences"] == 0:
            return (0,) * seq_count
        vals = [item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else item for item in committed]
        return _validate_per_request_acceptance(vals, per_req_bounds, q)
    if type(m.get("active_sequences")) is int and m["active_sequences"] == 0:
        return (0,) * seq_count
    views = [
        v for v in (getattr(ctx, "requests", ()) or getattr(ctx, "request_views", ()) or ())
        if not getattr(v, "padding", False)
    ]
    if views:
        adv = [int(getattr(v, "draft_advanced_tokens", 0) or 0) for v in views]
        vals = adv if any(v > 0 for v in adv) else [
            int(getattr(v, "draft_accepted_tokens", 0) or 0) for v in views
        ]
        bounds = [
            max(1, int(getattr(v, "q_len", 1)), 1 + int(getattr(v, "draft_proposal_tokens", 0) or 0))
            for v in views
        ]
        return _validate_per_request_acceptance(vals, bounds, q)
    return None


def _resolve_recover_ssm_commit_totals(
    c: LinearAttentionContract,
    *,
    family: str,
    q: int,
    arch_id: str,
    accepted_raw: tuple[int, ...] | None,
    base_ver: int,
    options: Mapping[str, Any],
    replay_bytes: Callable[[RecoverSsmCommitPlan, int], int] | None,
) -> tuple[RecoverSsmCommitPlan, tuple[int, ...], int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
    """Compute unified G4 RecoverSSM commit work, HBM byte totals, and Triton tile geometry."""
    from ..recurrent.recover_ssm_commit import compute_recover_ssm_commit_work, resolve_commit_tile_geometry

    m = c.metadata
    num_layers = max(1, int(m.get("num_layers", 1) or 1))
    boundary_ops = int(m.get("boundary_matrix_ops", 0) or 0)
    boundary_write = float(m.get("boundary_write_bytes", 0.0) or 0.0)
    boundaries = m.get("boundary_checkpoints", m.get("boundaries", ()))
    if accepted_raw is None:
        build_recover_ssm_commit(
            family, c.dk, c.dv, a=None, base_version=base_ver,
            boundary_checkpoints=boundaries, **options,
        )
        raise ModelingError("recover_ssm_commit_requires_accepted_prefix")
    per_req = accepted_raw
    use_per_req_q = (
        "q" not in m
        and len(c.lengths) == len(per_req)
        and (max(c.lengths, default=0) > 1 or q == 1)
    )
    req_qs = c.lengths if use_per_req_q else (q,) * len(per_req)
    checkpoint_seqs = sum(
        1 for rq, a_i in zip(req_qs, per_req, strict=True) if rq == 1 and a_i > 0
    )
    replay_per_req = tuple(
        a_i if rq > 1 else 0 for rq, a_i in zip(req_qs, per_req, strict=True)
    )
    replay_seqs = sum(1 for a_i in replay_per_req if a_i > 0)
    active_seqs = checkpoint_seqs + replay_seqs
    max_depth = max(per_req, default=0)
    replay_max_depth = max(replay_per_req, default=0)
    commit = build_recover_ssm_commit(
        family, c.dk, c.dv, a=max_depth, base_version=base_ver,
        boundary_checkpoints=boundaries, **{**options, "sequences": max(1, active_seqs)},
    )
    v_tiles = max(1, ceil_div(c.dv, c.bv))
    head_groups = max(1, c.hv)
    rec_override_raw = m.get("record_bytes_per_token", m.get("record_bytes"))
    rec_override = float(rec_override_raw) if rec_override_raw is not None else None
    state_override_raw = m.get("state_bytes")
    state_override = float(state_override_raw) if state_override_raw is not None else None
    per_seq_state = (
        int(round(state_override))
        if state_override is not None
        else c.hv * c.dk * c.dv * element_bytes(c.state_dtype)
    )
    checkpoint_read = checkpoint_seqs * per_seq_state * num_layers
    checkpoint_write = checkpoint_seqs * per_seq_state * num_layers
    if family in {"gdn", "kda"}:
        eff_align_tiling = bool(m.get("align_mode", False)) and (
            family == "gdn" or bool(m.get("recover_ssm_commit_kind") == "boundary")
        )
        _, geom_v_tiles, head_groups = resolve_commit_tile_geometry(
            family_kind=family, dk=c.dk, dv=c.dv, heads=c.hv,
            arch_family=arch_id, state_dtype=c.state_dtype,
            align_mode=eff_align_tiling,
        )
        explicit_vt = m.get("v_tiles")
        v_tiles = int(explicit_vt) if (explicit_vt is not None and int(explicit_vt) > 0) else geom_v_tiles
        work = compute_recover_ssm_commit_work(
            family_kind=family, accepted_per_request=replay_per_req,
            heads=c.hv, dk=c.dk, dv=c.dv, state_dtype=c.state_dtype,
            kg_dtype=str(m.get("record_kg_dtype", m.get("kg_dtype", "bf16"))),
            v_tiles=v_tiles, boundary_ops=0, boundary_write_bytes=0.0,
            record_bytes_override=rec_override, state_bytes_override=state_override,
        )
        total_replay = int(work["total_tokens"])
        replay_state_read = int(round(work["state_read_bytes"])) * num_layers
        replay_state_write = int(round(work["state_write_bytes"])) * num_layers + int(round(boundary_write))
        record_read = int(round(work["physical_tiled_record_read_bytes"])) * num_layers
        if replay_bytes and total_replay > 0:
            record_read = replay_bytes(
                replace(commit, record_read_bytes=int(round(work["unique_record_read_bytes"]))),
                total_replay,
            ) * num_layers
        vector_ops = (int(work["vector_ops"]) * num_layers + boundary_ops) if total_replay > 0 else 0
    else:
        total_replay = sum(replay_per_req)
        replay_state_read = replay_seqs * per_seq_state * num_layers
        replay_state_write = replay_seqs * per_seq_state * num_layers + int(round(boundary_write))
        per_tok_rec = (
            int(round(rec_override))
            if rec_override is not None
            else c.hv * commit.transition.record_bytes
        )
        record_read = total_replay * per_tok_rec * num_layers
        if replay_bytes and total_replay > 0:
            record_read = replay_bytes(
                replace(commit, record_read_bytes=total_replay * per_tok_rec), total_replay,
            ) * num_layers
        vector_ops = (
            (REPLAY_OPS_PER_STATE_ELEMENT + (family == "gla")) * total_replay * c.hv * c.dk * c.dv * num_layers
            + boundary_ops
            if total_replay > 0 else 0
        )
    published_ver = commit.base_version + int(active_seqs > 0)
    return (
        commit, per_req, active_seqs, checkpoint_seqs, total_replay, max_depth, replay_max_depth,
        checkpoint_read, checkpoint_write, replay_state_read, replay_state_write,
        record_read, vector_ops, published_ver, v_tiles, head_groups,
    )


def _build_recover_ssm_commit_phases(
    c: LinearAttentionContract,
    ctx: OperatorPlanningContext,
    *,
    family: str,
    q: int,
    options: Mapping[str, Any],
    replay_bytes: Callable[[RecoverSsmCommitPlan, int], int] | None,
) -> tuple[list[Phase], dict[str, Any], Any]:
    from ..recurrent.recover_ssm_commit import (
        COMMIT_KERNEL_BACKEND,
        GDN_COMMIT_KERNEL_ID,
        KDA_COMMIT_KERNEL_ID,
        _parse_version_number,
    )
    m = c.metadata
    accepted_raw = _resolve_recover_ssm_accepted_per_request(
        m, ctx, q, len(c.lengths), per_req_bounds=c.lengths,
    )
    req_transitions = tuple(m.get("request_state_transitions") or ())
    het_versions = bool(m.get("heterogeneous_state_versions", False)) or (
        len({str(t[3]) for t in req_transitions if isinstance(t, (tuple, list)) and len(t) >= 5}) > 1
    )
    base_ver_raw = m.get("base_version")
    if het_versions:
        base_ver = 0
    elif base_ver_raw is None and m.get("input_state_version"):
        parsed_v, _ = _parse_version_number(m.get("input_state_version"))
        base_ver = int(parsed_v or 0)
    else:
        base_ver = 0 if base_ver_raw is None else base_ver_raw
    arch_id = str(ctx.compute_arch.arch_id)
    (
        commit, per_req, active_seqs, checkpoint_seqs, total_replay, max_depth, replay_max_depth,
        checkpoint_read, checkpoint_write, replay_state_read, replay_state_write,
        record_read, vector_ops, published_ver, v_tiles, head_groups,
    ) = _resolve_recover_ssm_commit_totals(
        c, family=family, q=q, arch_id=arch_id, accepted_raw=accepted_raw,
        base_ver=base_ver, options=options, replay_bytes=replay_bytes,
    )

    phases: list[Phase] = []
    if c.abi == "commit_recurrent_kda" or (checkpoint_seqs > 0 and total_replay == 0):
        phases.append(Phase("read_acceptance", transfer(ctx, read=len(c.lengths) * ACCEPTANCE_WORD_BYTES)))
    accepted_after = ("read_acceptance",) if phases else ()
    if checkpoint_seqs > 0:
        ckpt_publish = "publish" if total_replay == 0 else "publish_checkpoint"
        phases.extend((
            Phase("read_checkpoint", transfer(ctx, read=checkpoint_read), accepted_after),
            Phase(ckpt_publish, transfer(ctx, write=checkpoint_write), ("read_checkpoint",)),
        ))
    if total_replay > 0:
        phases.extend((
            Phase("read_base_state", transfer(ctx, read=replay_state_read), accepted_after),
            Phase("read_records", transfer(ctx, read=record_read), accepted_after),
        ))
        after, previous = ("read_base_state", "read_records"), 0
        replay_boundaries = tuple(point for point in commit.boundary_checkpoints if point <= replay_max_depth)
        for end in dict.fromkeys((*replay_boundaries, replay_max_depth)):
            name = f"replay_to_{end}" if replay_boundaries else "replay"
            share = (end - previous) / float(max(1, replay_max_depth))
            replay_ops = int(round(vector_ops * share)) if replay_boundaries else vector_ops
            phases.append(Phase(name, Work(resources={"vector": replay_ops}), after))
            after, previous = (name,), end
            if end < replay_max_depth:
                checkpoint = f"publish_boundary_{end}"
                phases.append(Phase(checkpoint, transfer(ctx, write=replay_state_read), after))
                after = (checkpoint,)
        publish_bytes = replay_state_write if not replay_boundaries else replay_state_read
        phases.append(Phase("publish", transfer(ctx, write=publish_bytes), after))
    is_triton_family = family in {"gdn", "kda"}
    kb_id = (
        (GDN_COMMIT_KERNEL_ID if family == "gdn" else KDA_COMMIT_KERNEL_ID)
        if is_triton_family else "RecoverSSMStateCommit"
    )
    kb_backend = COMMIT_KERNEL_BACKEND if is_triton_family else "CANN"
    kb_readiness = str(
        m.get("recover_ssm_readiness")
        or ("proposed_pr" if family == "gdn" else "shipping")
    )
    num_layers = max(1, int(m.get("num_layers", 1) or 1))
    result: dict[str, Any] = {
        "q": q,
        "a": max_depth,
        "accepted_per_request": per_req,
        "active_sequences": active_seqs,
        "total_replay_tokens": total_replay,
        "max_replay_depth": replay_max_depth,
        "published_version": None if het_versions else published_ver,
        "base_version": None if het_versions else commit.base_version,
        "heterogeneous_state_versions": het_versions,
        "request_state_transitions": req_transitions,
        "skipped_zero_accept": active_seqs == 0,
        "boundary_checkpoints": commit.boundary_checkpoints,
        "physical_record_read_bytes": record_read,
        "v_tiles": v_tiles,
        "head_groups": head_groups,
        "num_layers": num_layers,
        "kernel_binding": KernelBinding(
            kernel_id=kb_id,
            backend=kb_backend,
            arch_family=arch_id,
            submodule="recurrent_state",
            macro_module="attention",
            readiness=kb_readiness,  # type: ignore[arg-type]
            recipe_id=str(c.abi or c.operation or "recover_ssm_commit"),
        ).to_dict(),
    }
    if het_versions and m.get("published_state_version"):
        result["published_state_version"] = str(m["published_state_version"])
    return phases, result, commit.transition


def build_recover_ssm_plan(contract: LinearAttentionContract, ctx: OperatorPlanningContext, *,
                           sources: tuple[SourceRef, ...] = (),
                           replay_bytes: Callable[[RecoverSsmCommitPlan, int], int] | None = None) -> OperatorPlan:
    c = contract
    m = c.metadata
    q = _resolve_recover_ssm_window(c, ctx)
    options = dict(q=q, heads=c.hv, sequences=len(c.lengths), state_dtype=c.state_dtype,
                   kg_dtype=m.get("kg_dtype", "bf16"), gated=m.get("gated", False), factors=m.get("factors"))
    family = m.get(
        "ssm_family",
        m.get(
            "transition_family",
            m.get(
                "family",
                m.get("recurrent_attention_kind", "kda" if c.abi == "commit_recurrent_kda" else "gdn"),
            ),
        ),
    )
    owners = len(c.lengths) * c.hv * ceil_div(c.dv, c.bv)
    if c.operation == "recover_ssm_verify":
        verify = build_recover_ssm_verify(family, c.dk, c.dv, **options)
        req_lengths = (q,) * len(c.lengths) if "q" in m else c.lengths
        tokens = sum(req_lengths)
        single_count = sum(1 for length in req_lengths if length == 1)
        spec_tokens = sum(length for length in req_lengths if length > 1)
        gate_width = c.dk if family in {"kda", "gla"} else 1
        gate_width += c.dv if family == "gla" else 0
        inputs = tokens * (
            ((c.hq + c.hk) * c.dk + c.hv * c.dv) * element_bytes(c.dtype)
            + c.hv * (gate_width * element_bytes(c.gate_dtype) + element_bytes(c.beta_dtype))
        )
        output = tokens * c.hv * c.dv * element_bytes(c.dtype)
        single_checkpoint = spec_tokens == 0 and single_count > 0
        checkpoint_bytes = single_count * c.hv * c.dk * c.dv * element_bytes(c.state_dtype)
        record_write_bytes = spec_tokens * c.hv * verify.transition.record_bytes
        write_phases: list[Phase] = []
        if checkpoint_bytes > 0:
            write_phases.append(
                Phase("write_state_checkpoint", transfer(ctx, write=checkpoint_bytes), ("verify_tokens",))
            )
        if record_write_bytes > 0:
            write_phases.append(
                Phase("write_records", transfer(ctx, write=record_write_bytes), ("verify_tokens",))
            )
        phases = (
            Phase("read_base_state", transfer(ctx, read=verify.state_read_bytes)),
            Phase("verify_inputs", transfer(ctx, read=inputs)),
            Phase("verify_tokens", Work(resources={"vector": (2 * REPLAY_OPS_PER_STATE_ELEMENT + (family == "gla"))
                                                   * tokens * c.hv * c.dk * c.dv}),
                  ("read_base_state", "verify_inputs")),
            Phase("verify_outputs", transfer(ctx, write=output), ("verify_tokens",)),
            *write_phases,
        )
        result = {
            "q": q,
            "single_checkpoint": single_checkpoint,
            "output_tokens": tokens,
            "checkpoint_write_bytes": checkpoint_bytes,
            "record_write_bytes": record_write_bytes,
        }
        transition = verify.transition
    else:
        phases, result, transition = _build_recover_ssm_commit_phases(
            c, ctx, family=family, q=q, options=options, replay_bytes=replay_bytes,
        )
        active_seqs = int(result.get("active_sequences", 0))
        eff_head_groups = int(result.get("head_groups", c.hv) or c.hv)
        eff_v_tiles = int(result.get("v_tiles", ceil_div(c.dv, c.bv)) or 1)
        eff_layers = int(result.get("num_layers", 1) or 1)
        owners = max(1, active_seqs * eff_layers * eff_head_groups * eff_v_tiles) if active_seqs > 0 else 0
    launch_count = int(bool(phases) or m.get("recover_ssm_commit_kind") == "ordinary")
    return make_plan(ctx, c.abi or c.operation, phases, sources=sources, family="recover_ssm",
                     launch_count=launch_count, owners=owners if owners > 0 else None, metadata={
                         **result, "owners": owners, "ssm_family": family, "record_bytes": transition.record_bytes,
                         "upstream_verified": transition.upstream_verified,
                     })
