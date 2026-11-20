"""Shared linear-attention tensor contracts and conserved work.

[Role] Derive WY work, state residency, convolution, and speculative replay.
[Invariants] Physical transfer bytes and logical record widths are accounted
  independently. Per-owner state liveness is independent of sequence length.
[Timeouts & Locks] Shape arithmetic and local admission simulation only.
[Side-Effect Policy] No device, filesystem, or caller-state mutations.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from heapq import heapify, heapreplace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, Program, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.l2_residency import AdmissionMemoryView
from ...runtime.memory_physics import (
    AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey, VisibilityContract,
)
from ...runtime.tile_solver import LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape, solve
from ..contracts import OperatorPlanningContext

FP32_BYTES = 4
MASK_OPS_PER_PAIR = 4
GDN_CHUNK32_TOKENS = 32
GDN_CHUNK64_TOKENS = 64
DIAG_BLOCK_WIDTH = GDN_CHUNK32_TOKENS
DEFAULT_VALUE_TILE = 128
DEFAULT_KEY_TILE = 64
REPLAY_OPS_PER_STATE_ELEMENT = 4
SILU_OPS_PER_ELEMENT = 4
SYNC_SCRATCH_BYTES = 32
ROUTE_COST_SCALE = 64
PARTS_PER_MILLION = 1_000_000
INT64_MAX = 2**63 - 1
ROUTE_PARAMETER_WORDS = 16
ROUTE_SCHEMA_INDEX = 13
ROUTE_MARGIN_INDEX = 14
ROUTE_SCHEMAS = (6, 7)
KDA_WORKSPACE_SLOTS = 861
KDA_MAX_CHUNK_GROUP = 512
KDA_MAX_AIC_CORES = 32
KDA_MAX_VALUE_HEADS = 96
KDA_METADATA_HEADER_WORDS = 11
KDA_CHUNK_TASK_WORDS = 6
KDA_TRAINING_GROUP_LIMIT = 16
KDA_TRAINING_CHUNK_RANGE = range(64, 257)
# Default host parameter block. A caller can supply the exact block for its stack.
KDA_ROUTE_PARAMETERS = (
    0, 2711857704, 699286307, 16555661144, 28359460, 94263834, 94263834,
    12013349865, 1045656019, 84358156, 12842477, 2527878117, 181898838, 6, 8441, 0,
    0, 692059912, 851406583, 21510815509, 73490570, 81791118, 166178779,
    12029971961, 2552246171, 459139222, 45086212, 2303638937, 143977434, 6, 18992, 0,
)
CHUNK32_ABIS = frozenset({
    "gdn_chunk32", "gdn_prepare_arch35_chunk32", "gdn_state_arch35_chunk32",
    "gdn_output_arch35_chunk32",
})
ABI_OPERATIONS = {
    **dict.fromkeys(CHUNK32_ABIS, "gdn_chunk32"),
    "aclnnChunkGatedDeltaRuleArch22": "gdn_prefill",
    "aclnnChunkGatedDeltaRuleArch35": "gdn_prefill",
    "gdn_prefill_arch35": "gdn_prefill",
    "flash_kda": "kda_prefill", "flash_kda_staged": "kda_prefill",
    "flash_kda_persistent": "kda_prefill", "flash_kda_arch35": "kda_prefill",
    "fused_recurrent_kda": "kda_recurrent",
    "causal_conv1d_update": "causal_conv1d", "commit_recurrent_kda": "recover_ssm_commit",
}
OPERATIONS = frozenset({
    "gdn_prefill", "gdn_chunk32", "kda_prefill", "gdn_recurrent", "kda_recurrent",
    "causal_conv1d", "recover_ssm_verify", "recover_ssm_commit",
})
RECOVER_SSM_FAMILIES = ("kda", "gdn", "deltanet", "gla", "lightning", "retnet", "mamba")

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengths", _lengths(self.lengths))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        _positive(self.hq, self.hk, self.hv, self.dk, self.dv, self.bt, self.bv, self.bk,
                  self.active_l2_streams)
        if self.hv % self.hk or self.hq % self.hk:
            raise ModelingError("linear_attention_head_groups_not_divisible")
        if self.operation not in OPERATIONS - {"causal_conv1d"}:
            raise ModelingError(f"linear_attention_operation_unsupported:{self.operation}")
        pinned = self.abi in CHUNK32_ABIS or self.metadata.get("chunk32") is True
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "lengths", _lengths(self.lengths))
        _positive(self.channels, self.width, self.speculative_width)
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

    lengths = _lengths(metadata.get("lengths", metadata.get("sequence_lengths", metadata.get("sequence_tokens", ()))))
    dtype = metadata.get("dtype", op.dtype or "bf16")
    if operation == "causal_conv1d":
        contract = CausalConvContract(
            metadata.get("channels"), metadata.get("width", metadata.get("taps")), lengths,
            mode=metadata.get("mode", "incremental" if abi == "causal_conv1d_update" else "varlen"),
            dtype=dtype, state_dtype=metadata.get("state_dtype", dtype),
            has_bias=metadata.get("has_bias", False), activation=metadata.get("activation", "none"),
            write_state=metadata.get("write_state", True),
            speculative_width=metadata.get("speculative_width", 1),
        )
        weights = metadata.get("weight_shape", (contract.channels, contract.width))
        history = metadata.get("history_shape", (contract.width - 1, contract.channels))
        valid_history = {(contract.width - 1, contract.channels),
                         (len(lengths), contract.width - 1, contract.channels)}
        if (not isinstance(weights, (list, tuple)) or not isinstance(history, (list, tuple))
                or tuple(weights) != (contract.channels, contract.width) or tuple(history) not in valid_history):
            raise ModelingError("causal_conv1d_weight_or_history_shape_mismatch")
        return contract

    pinned = abi in CHUNK32_ABIS or metadata.get("chunk32") is True
    hv = metadata.get("hv", metadata.get("heads", 1))
    dk = metadata.get("dk", metadata.get("head_dim", DEFAULT_VALUE_TILE))
    dv = metadata.get("dv", metadata.get("value_dim", dk))
    fields = {key: metadata[key] for key in (
        "state_dtype", "stored_chunk_state_dtype", "gate_dtype", "beta_dtype",
        "use_initial_state", "output_final_state", "save_intermediate_state", "save_matrices",
        "kda_template_request", "l2_capacity_bytes", "active_l2_streams",
    ) if key in metadata}
    bt = metadata.get("bt", metadata.get("chunk_size", DIAG_BLOCK_WIDTH if pinned else GDN_CHUNK64_TOKENS))
    return LinearAttentionContract(
        operation, lengths, metadata.get("hq", hv), metadata.get("hk", hv), hv, dk, dv, bt,
        metadata.get("bv", DEFAULT_VALUE_TILE), metadata.get("bk", DEFAULT_KEY_TILE),
        dtype, abi=abi, metadata=metadata, **fields,
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
        scratch = contract.bt * (
            (2 * contract.bk + contract.bv) * element_bytes(contract.dtype)
            + gate_width * element_bytes(contract.gate_dtype) + element_bytes(contract.beta_dtype)
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
    capacity = contract.l2_capacity_bytes
    units = ctx.geometry.buffer_units.get("l2", 1) if ctx else 1
    view = AdmissionMemoryView({0: path.usable_bytes("l2") if capacity is None else capacity},
                               l2_buffer_units_by_die={0: units})
    for stream in range(contract.active_l2_streams):
        view.enter_stream(0, str(stream))
    key = ReuseKey(0, "chunk_state_bank", 0, "0")
    runs = AddressRunPlan((AddressRun(0, chunks * contract.hv, head_bytes, head_bytes),))
    read = MemoryAccessPattern("chunk_state_bank", 0, 0, 0, "read", "mte2_aic", size, runs, key)
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
        "gla": ("separable", "fold value gate into correction; apply key gate to state columns"),
        "lightning": ("head", "head_decay * S + V outer K"),
        "retnet": ("head", "retention_decay * S + projected_V outer projected_K"),
        "mamba": ("diagonal_selective", "exp(dt * A_diag) * S + dt * input outer B; read with C_proj"),
    }[family]
    key, decay, extra = FP32_BYTES * dk, FP32_BYTES, 0
    if family in {"kda", "gla"}:
        key = decay = dk * packed
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
                               owners * q * transition.record_bytes, q * sequences, q == 1)

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
    return RecoverSsmCommitPlan(
        verify.transition, q, a, state, heads * sequences * a * verify.transition.record_bytes,
        REPLAY_OPS_PER_STATE_ELEMENT * a * sequences * heads * dk * dv, state,
        base_version, base_version + int(a > 0), a == 0,
        tuple(point for point in boundaries if point <= a),
    )

def flash_kda_workspace_groups(
    chunks: Sequence[int], slots_per_head: int, max_group: int,
) -> Iterator[tuple[int, int, tuple[int, ...]]]:
    _positive(slots_per_head, max_group)
    for start in range(0, len(chunks), slots_per_head):
        group = chunks[start:start + slots_per_head]
        consumed = 0
        longest = max(group)
        while consumed < longest:
            low, high = 1, min(max_group, longest - consumed, slots_per_head)
            while low < high:
                mid = (low + high + 1) // 2
                if sum(min(max(0, n - consumed), mid) for n in group) <= slots_per_head:
                    low = mid
                else:
                    high = mid - 1
            yield start, consumed, tuple(min(max(0, n - consumed), low) for n in group)
            consumed += low

def flash_kda_group_limit(chunks: Sequence[int], heads: int, cores: int, layout: int,
                          requested: int | None, calibration_enabled: bool) -> int:
    group = KDA_MAX_CHUNK_GROUP if requested is None else requested
    _positive(group)
    if (calibration_enabled and layout == 1 and cores == KDA_MAX_AIC_CORES and len(chunks) == 1
            and KDA_MAX_AIC_CORES < heads <= KDA_MAX_VALUE_HEADS and chunks[0] in KDA_TRAINING_CHUNK_RANGE):
        return min(group, KDA_TRAINING_GROUP_LIMIT)
    return group

def select_flash_kda_template(
    lengths: LinearAttentionContract | Sequence[int], *, value_heads: int | None = None,
    query_heads: int | None = None, core_num: int, template_request: str = "auto",
    route_coefficients: Sequence[int] = KDA_ROUTE_PARAMETERS, layout_kind: int = 0,
    workspace_slots: int = KDA_WORKSPACE_SLOTS, max_chunk_group: int | None = None,
    metadata_word_capacity: int | None = None, has_cu: bool = False,
    calibration_enabled: bool = True,
) -> str:
    if isinstance(lengths, LinearAttentionContract):
        contract = lengths
        value_heads, query_heads = contract.hv, contract.hq
        template_request = contract.kda_template_request if template_request == "auto" else template_request
        lengths = contract.lengths
    lengths = _lengths(lengths)
    _positive(value_heads, query_heads, core_num, workspace_slots)
    if template_request in {"staged", "persistent"}:
        return template_request
    if (template_request != "auto" or workspace_slots < value_heads or layout_kind not in (0, 1)
            or core_num > KDA_MAX_AIC_CORES or value_heads > KDA_MAX_VALUE_HEADS):
        raise ModelingError("flash_kda_route_geometry_invalid")
    if not calibration_enabled:
        return "staged"
    chunks = tuple(ceil_div(n, GDN_CHUNK64_TOKENS) for n in lengths)
    peak = max(chunks)
    group = flash_kda_group_limit(chunks, value_heads, core_num, layout_kind, max_chunk_group, calibration_enabled)
    if has_cu and len(lengths) > 1 and peak <= 2:
        return "persistent"
    owners = len(lengths) * value_heads
    if metadata_word_capacity is not None and (
        KDA_METADATA_HEADER_WORDS + KDA_MAX_AIC_CORES + KDA_CHUNK_TASK_WORDS * owners + len(lengths)
        > metadata_word_capacity
    ):
        return "staged"
    base = layout_kind * ROUTE_PARAMETER_WORDS
    coeff = route_coefficients[base:base + ROUTE_PARAMETER_WORDS]
    if (len(coeff) != ROUTE_PARAMETER_WORDS or any(type(n) is not int or n < 0 for n in coeff[:-1])
            or coeff[ROUTE_SCHEMA_INDEX] not in ROUTE_SCHEMAS or coeff[ROUTE_MARGIN_INDEX] >= PARTS_PER_MILLION):
        return "staged"

    cyclic = [0] * core_num
    lpt = [(0, index, 0) for index in range(core_num)]
    heapify(lpt)
    for batch in sorted(range(len(chunks)), key=lambda i: (-chunks[i], i)):
        for head in range(value_heads):
            cyclic[(batch * value_heads + head) % core_num] += chunks[batch]
            load, core, rows = lpt[0]
            heapreplace(lpt, (load + chunks[batch], core, rows + lengths[batch]))
    parallel = ceil_div(sum(chunks) * value_heads, core_num)
    staged_peak, persistent_peak = max(cyclic), max(load for load, _, _ in lpt)
    if max(parallel, staged_peak, persistent_peak, owners) > INT64_MAX // ROUTE_COST_SCALE:
        return "staged"
    rounds = sum(1 for _ in flash_kda_workspace_groups(chunks, workspace_slots // value_heads, group))

    (s_base, s_parallel, s_peak, s_round, s_query, s_compute, s_memory,
     p_base, p_peak, p_head, p_query, p_compute, p_memory, _, margin, _) = coeff
    scale = ROUTE_COST_SCALE
    total_rows, query_rows = sum(lengths) * value_heads, sum(lengths) * query_heads
    staged = (scale * (s_base + s_parallel * parallel + s_peak * staged_peak + s_round * rounds)
              + s_query * query_rows + max(s_compute * parallel * scale, s_memory * total_rows))
    persistent = scale * (p_base + p_peak * persistent_peak + p_head * owners) + p_query * query_rows
    previous = 0
    for index, rows in enumerate(sorted(rows for _, _, rows in lpt)):
        persistent += max(p_compute, p_memory * (core_num - index)) * (rows - previous)
        previous = rows
    if max(staged, persistent) > INT64_MAX:
        return "staged"
    required_gain = staged * margin // PARTS_PER_MILLION
    return "persistent" if persistent < staged - required_gain else "staged"


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
        "launch_count": launch_count, **(metadata or {}),
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
    history = len(c.lengths) * (c.width - 1) * c.channels * stored
    output = tokens * c.channels * size
    workers = min(ctx.effective_aiv_cores, len(c.lengths) * c.channels)
    _positive(weight_copies)
    macs = 2 * c.channels * tokens * c.width
    state_writes = history
    if c.mode == "incremental" and c.width > 1 and c.speculative_width > 1:
        state_writes += sum(min(n, c.speculative_width) - 1 for n in c.lengths) * c.channels * stored
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
    return make_plan(ctx, implementation, phases, sources=sources, family="causal_conv1d", owners=workers,
                     metadata={"mode": c.mode, "mac_flops": macs, "history_rows": c.width - 1,
                               "speculative_width": c.speculative_width})

def build_recover_ssm_plan(contract: LinearAttentionContract, ctx: OperatorPlanningContext, *,
                           sources: tuple[SourceRef, ...] = (),
                           replay_bytes: Callable[[RecoverSsmCommitPlan], int] | None = None) -> OperatorPlan:
    c = contract
    m = c.metadata
    q = m.get("q", max(c.lengths) if c.operation == "recover_ssm_verify" else min(c.lengths))
    options = dict(q=q, heads=c.hv, sequences=len(c.lengths), state_dtype=c.state_dtype,
                   kg_dtype=m.get("kg_dtype", "bf16"), gated=m.get("gated", False), factors=m.get("factors"))
    family = m.get("ssm_family", m.get("transition_family", m.get("family", "kda" if c.abi == "commit_recurrent_kda" else "gdn")))
    owners = len(c.lengths) * c.hv * ceil_div(c.dv, c.bv)
    if c.operation == "recover_ssm_verify":
        verify = build_recover_ssm_verify(family, c.dk, c.dv, **options)
        tokens = verify.output_tokens if "q" in m else sum(c.lengths)
        gate_width = c.dk if family in {"kda", "gla"} else 1
        inputs = tokens * (
            ((c.hq + c.hk) * c.dk + c.hv * c.dv) * element_bytes(c.dtype)
            + c.hv * (gate_width * element_bytes(c.gate_dtype) + element_bytes(c.beta_dtype))
        )
        output = tokens * c.hv * c.dv * element_bytes(c.dtype)
        checkpoint = tokens == len(c.lengths)
        state_or_rec = (
            Phase("write_state_checkpoint", transfer(ctx, write=verify.state_read_bytes), ("verify_tokens",))
            if checkpoint
            else Phase("write_records", transfer(ctx, write=tokens * c.hv * verify.transition.record_bytes),
                       ("verify_tokens",))
        )
        phases = (
            Phase("read_base_state", transfer(ctx, read=verify.state_read_bytes)),
            Phase("verify_inputs", transfer(ctx, read=inputs)),
            Phase("verify_tokens", Work(resources={"vector": 2 * REPLAY_OPS_PER_STATE_ELEMENT
                                                   * tokens * c.hv * c.dk * c.dv}),
                  ("read_base_state", "verify_inputs")),
            Phase("verify_outputs", transfer(ctx, write=output), ("verify_tokens",)),
            state_or_rec,
        )
        result = {"q": q, "single_checkpoint": checkpoint, "output_tokens": tokens}
        transition = verify.transition
    else:
        commit = build_recover_ssm_commit(
            family, c.dk, c.dv, a=m.get("a", m.get("accepted")),
            base_version=m.get("base_version", 0), boundary_checkpoints=m.get("boundary_checkpoints", ()), **options,
        )
        record_read = replay_bytes(commit) if replay_bytes else commit.record_read_bytes
        phases = []
        if not commit.skipped_zero_accept:
            phases.extend((Phase("read_base_state", transfer(ctx, read=commit.state_read_bytes)),
                           Phase("read_records", transfer(ctx, read=record_read))))
            after, previous = ("read_base_state", "read_records"), 0
            for end in dict.fromkeys((*commit.boundary_checkpoints, commit.a)):
                name = f"replay_to_{end}" if commit.boundary_checkpoints else "replay"
                replay = Work(resources={"vector": commit.vector_ops // commit.a * (end - previous)})
                phases.append(Phase(name, replay, after))
                after, previous = (name,), end
            phases.append(Phase("publish", transfer(ctx, write=commit.state_write_bytes), after))
        result = {"q": q, "a": commit.a, "published_version": commit.published_version,
                  "base_version": commit.base_version, "skipped_zero_accept": commit.skipped_zero_accept,
                  "boundary_checkpoints": commit.boundary_checkpoints, "physical_record_read_bytes": record_read}
        transition = commit.transition
    return make_plan(ctx, c.abi or c.operation, phases, sources=sources, family="recover_ssm",
                     launch_count=int(bool(phases)), owners=owners, metadata={
                         **result, "ssm_family": family, "record_bytes": transition.record_bytes,
                         "upstream_verified": transition.upstream_verified,
                     })
