"""Shared paged MLA tensor contract and ABI validation.

[Role] Declare dense and mixed-quant paged MLA tensor inputs.
[Invariants] Preserve active extents, allocation geometry, and ABI identity.
[Timeouts & Locks] Validation is bounded by the input metadata. No locks.
[Side-Effect Policy] Construct immutable records without I/O.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from itertools import chain
from math import ceil

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from ..contracts import OperatorPlanningContext

FAMILY = "paged_mla"
CONTRACT_KEY = "paged_mla_contract"
FIA_ABI = "aclnnFusedInferAttentionScore"
FLASH_ABI = "aclnnFlashMla"
DSL_FLASH_ABI = "cannbot_dsl.flash_mla_with_kvcache"
MQS_ABI = "aclnnMixedQuantSparseFlashMla"
DSL_MQS_ABI = "cannbot_dsl.mixed_quant_sparse_flash_mla"
MQS_ABIS = frozenset({MQS_ABI, DSL_MQS_ABI})
SUPPORTED_ABIS = frozenset({FIA_ABI, FLASH_ABI, DSL_FLASH_ABI}) | MQS_ABIS
DEFAULT_LATENT_DIM = 512
DEFAULT_ROPE_DIM = 64
DEFAULT_PAGE_SIZE = 128
DEFAULT_S2_BASE_SIZE = 512
MQS_HEAD_DIM = 512
DSL_MQS_HEADS = 64
DSL_MQS_SCALE_ENCODINGS = ("bf16_g32", "bf16_g16")
MQS_ORIGINAL_ROW_BYTES = 544
MQS_COMPRESSED_ROW_BYTES = 320
FP32_BYTES = 4
INDEX_BYTES = 4
NZ_BLOCK_BYTES = 32
MASK_NONE = 0
MASK_CAUSAL = 3
MASK_WINDOW = 4
MAX_COMPRESSION_RATIO = 128
MQS_QUANT_MODES = frozenset({1, 2})
SCALE_ENCODINGS = frozenset({
    "bf16_g32", "bf16_g16", "fp16_g32", "fp16_g16", "e8m0_g32", "e8m0_g16",
})


@dataclass(frozen=True, slots=True)
class PagedMlaContract:
    sequences: tuple[tuple[int, int], ...]
    logical_heads: int
    physical_heads: int
    latent_dim: int = DEFAULT_LATENT_DIM
    rope_dim: int = DEFAULT_ROPE_DIM
    page_size: int = DEFAULT_PAGE_SIZE
    block_table_len: int | None = None
    kv_layout: str = "PA_BLK_BSND"
    abi: str = FIA_ABI
    s2_base_size: int = DEFAULT_S2_BASE_SIZE
    splits: int = 1
    return_lse: bool = False
    latent_alias: bool = True
    n_ori: int | tuple[int, ...] = 0
    n_cmp: int | tuple[int, ...] = 0
    quant_mode: int = 1
    cmp_ratio: int = 1
    ori_mask_mode: int = MASK_NONE
    cmp_mask_mode: int = MASK_NONE
    topk_value_mode: int = 1
    scale_encoding: str | tuple[str, str] = DSL_MQS_SCALE_ENCODINGS
    mqsmla_dim: int = MQS_HEAD_DIM
    block_table: tuple[tuple[int, ...], ...] = ()
    ori_block_table: tuple[tuple[int, ...], ...] = ()
    cmp_block_table: tuple[tuple[int, ...], ...] = ()
    ori_indices: tuple[tuple[int, ...], ...] = ()
    cmp_indices: tuple[tuple[int, ...], ...] = ()
    atomic_split_decode: bool = False
    sinks: tuple[str, tuple[int, ...]] | None = None

    @property
    def mixed_quant(self) -> bool:
        return self.abi in MQS_ABIS

    @property
    def dqk(self) -> int:
        return self.mqsmla_dim if self.mixed_quant else self.latent_dim + self.rope_dim

    @property
    def dv(self) -> int:
        return self.mqsmla_dim if self.mixed_quant else self.latent_dim

    @property
    def useful_matrix_flops(self) -> int:
        pairs = sum(q * kv for q, kv in self.sequences) * self.logical_heads
        return 2 * pairs * (self.dqk + self.dv)

    @property
    def issued_matrix_flops(self) -> int:
        pairs = sum(q * kv for q, kv in self.sequences) * self.physical_heads
        return 2 * pairs * (self.dqk + self.dv)

def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor

def align_bytes(value: int, alignment: int = NZ_BLOCK_BYTES) -> int:
    return ceil_div(value, alignment) * alignment

def _extent(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"paged_mla_invalid_{name}:{value!r}")

def mqsmla_payload_bytes(n_ori: int, n_cmp: int) -> int:
    """Return exact FP8 and packed E2M1 record payload before transactions."""
    _extent("n_ori", n_ori)
    _extent("n_cmp", n_cmp)
    return MQS_ORIGINAL_ROW_BYTES * n_ori + MQS_COMPRESSED_ROW_BYTES * n_cmp

def latent_record_bytes(
    latent_dim: int | PagedMlaContract = DEFAULT_LATENT_DIM,
    rope_dim: int = DEFAULT_ROPE_DIM,
    dtype: str = "bf16",
    *, latent_alias: bool = True,
) -> int:
    """Return the compact allocation per dense KV token before layout padding."""
    if isinstance(latent_dim, PagedMlaContract):
        rope_dim, latent_alias = latent_dim.rope_dim, latent_dim.latent_alias
        latent_dim = latent_dim.latent_dim
    _extent("latent_dim", latent_dim, 1)
    _extent("rope_dim", rope_dim)
    return ceil((latent_dim + rope_dim + (0 if latent_alias else latent_dim)) * dtype_bytes(dtype))

def _rows(value: int | tuple[int, ...], seq: int) -> int:
    return value if type(value) is int else value[seq]

def _validate_lists(name: str, values: tuple, sequences: int) -> None:
    if not isinstance(values, tuple) or (values and len(values) != sequences):
        raise ModelingError(f"paged_mla_invalid_{name}_batch")
    if any(not isinstance(row, tuple) for row in values):
        raise ModelingError(f"paged_mla_invalid_{name}_list")
    for value in chain.from_iterable(values):
        _extent(name, value)

def _validate_mixed(c: PagedMlaContract) -> None:
    for name, counts in (("n_ori", c.n_ori), ("n_cmp", c.n_cmp)):
        if isinstance(counts, tuple) and len(counts) != len(c.sequences):
            raise ModelingError(f"paged_mla_invalid_{name}_batch")
        for count in counts if isinstance(counts, tuple) else (counts,):
            _extent(name, count)
    modes = {1} if c.abi == DSL_MQS_ABI else MQS_QUANT_MODES
    if type(c.quant_mode) is not int or c.quant_mode not in modes:
        raise ModelingError(f"paged_mla_invalid_quant_mode:{c.abi}:{c.quant_mode}")
    if type(c.cmp_ratio) is not int or not 1 <= c.cmp_ratio <= MAX_COMPRESSION_RATIO:
        raise ModelingError(f"paged_mla_invalid_cmp_ratio:{c.cmp_ratio}")
    for name, value, modes in (
        ("ori_mask_mode", c.ori_mask_mode, {MASK_NONE, MASK_CAUSAL, MASK_WINDOW}),
        ("cmp_mask_mode", c.cmp_mask_mode, {MASK_NONE, MASK_CAUSAL}),
        ("topk_value_mode", c.topk_value_mode, {1}),
    ):
        if type(value) is not int or value not in modes:
            raise ModelingError(f"paged_mla_invalid_{name}:{value}")
    encodings = (c.scale_encoding,) if isinstance(c.scale_encoding, str) else c.scale_encoding
    if (not isinstance(encodings, tuple) or len(encodings) not in (1, 2)
            or any(not isinstance(e, str) or e not in SCALE_ENCODINGS for e in encodings)):
        raise ModelingError(f"paged_mla_invalid_scale_encoding:{c.scale_encoding!r}")
    if c.abi == DSL_MQS_ABI:
        if c.physical_heads != DSL_MQS_HEADS:
            raise ModelingError(f"paged_mla_dsl_mqsmla_requires_heads:{DSL_MQS_HEADS}")
        has_compressed = any(c.n_cmp) if isinstance(c.n_cmp, tuple) else bool(c.n_cmp)
        allowed = (DSL_MQS_SCALE_ENCODINGS,) if has_compressed else (
            DSL_MQS_SCALE_ENCODINGS, DSL_MQS_SCALE_ENCODINGS[:1],
        )
        if encodings not in allowed:
            raise ModelingError(f"paged_mla_invalid_scale_encoding:{c.abi}:{c.scale_encoding!r}")
    if type(c.mqsmla_dim) is not int or c.mqsmla_dim != MQS_HEAD_DIM or c.rope_dim >= c.mqsmla_dim:
        raise ModelingError("paged_mla_mqsmla_requires_packed_width_512")
    for seq, (_, kv) in enumerate(c.sequences):
        ori, cmp = _rows(c.n_ori, seq), _rows(c.n_cmp, seq)
        if ori + cmp != kv:
            raise ModelingError(f"paged_mla_sparse_pair_coverage:{seq}:{ori}+{cmp}!={kv}")
        for branch, count, indices, table in (
            ("ori", ori, c.ori_indices, c.ori_block_table),
            ("cmp", cmp, c.cmp_indices, c.cmp_block_table),
        ):
            selected = indices[seq] if indices else ()
            if indices and len(selected) != count:
                raise ModelingError(f"paged_mla_{branch}_index_count:{seq}")
            extent = max(selected, default=-1) + 1 if indices else count
            pages = ceil_div(extent, c.page_size)
            if table and pages > len(table[seq]):
                raise ModelingError(f"paged_mla_{branch}_block_table_coverage:{seq}")
            if c.block_table_len is not None and pages > c.block_table_len:
                raise ModelingError(f"paged_mla_{branch}_block_table_len:{seq}")

def validate_contract(c: PagedMlaContract) -> PagedMlaContract:
    """Check shape, declared address coverage, and branch-local MQS attributes."""
    for name in ("logical_heads", "physical_heads", "latent_dim", "page_size", "s2_base_size", "splits"):
        _extent(name, getattr(c, name), 1)
    _extent("rope_dim", c.rope_dim)
    if c.physical_heads < c.logical_heads:
        raise ModelingError("paged_mla_physical_heads_below_logical")
    if c.block_table_len is not None:
        _extent("block_table_len", c.block_table_len)
    if c.kv_layout not in ("PA_BLK_BSND", "PA_BLK_NZ", "NZ_LATENT"):
        raise ModelingError(f"paged_mla_invalid_kv_layout:{c.kv_layout}")
    if c.abi not in SUPPORTED_ABIS:
        raise capability_gap("paged_mla_unsupported_abi", c.abi)
    for name in ("return_lse", "latent_alias", "atomic_split_decode"):
        if type(getattr(c, name)) is not bool:
            raise ModelingError(f"paged_mla_invalid_{name}")
    if not isinstance(c.sequences, tuple) or any(
        not isinstance(pair, tuple) or len(pair) != 2 for pair in c.sequences
    ):
        raise ModelingError("paged_mla_invalid_sequences")
    for q, kv in c.sequences:
        _extent("q_len", q)
        _extent("kv_len", kv)
    for name in ("block_table", "ori_block_table", "cmp_block_table", "ori_indices", "cmp_indices"):
        _validate_lists(name, getattr(c, name), len(c.sequences))
    if c.mixed_quant:
        if c.block_table:
            raise ModelingError("paged_mla_mqsmla_requires_branch_local_block_tables")
        _validate_mixed(c)
    else:
        if c.n_ori or c.n_cmp or c.ori_indices or c.cmp_indices or c.ori_block_table or c.cmp_block_table:
            raise ModelingError("paged_mla_sparse_attributes_require_mqsmla")
        for seq, (_, kv) in enumerate(c.sequences):
            pages = ceil_div(kv, c.page_size)
            if ((c.block_table_len is not None and pages > c.block_table_len)
                    or (c.block_table and pages > len(c.block_table[seq]))):
                raise ModelingError(f"paged_mla_block_table_coverage:{seq}")
    if c.abi == DSL_MQS_ABI:
        expected_sinks = ("fp32", (c.physical_heads,))
        if c.sinks is None:
            return replace(c, sinks=expected_sinks)
        if c.sinks != expected_sinks:
            raise ModelingError(f"paged_mla_invalid_sinks:{c.sinks!r}:expected:{expected_sinks!r}")
    elif c.sinks is not None:
        raise capability_gap("paged_mla_sinks_unsupported_abi", c.abi)
    return c

def capability_gap(code: str, abi: str, arch_id: str = "") -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=FAMILY, entry="resolve_operator_plan", code=code,
        message=f"{FAMILY} ABI {abi!r} has no modeled path for {arch_id or 'the declared contract'}",
        hints=(build_unmodeled_onboarding_hint(
            observed_op_type=FAMILY, observed_abi_summary={"abi": abi, "arch_id": arch_id},
        ),),
    )

def _from_mapping(raw: Mapping) -> PagedMlaContract:
    fields = dict(raw)
    for name in ("sequences", "block_table", "ori_block_table", "cmp_block_table", "ori_indices", "cmp_indices"):
        if isinstance(fields.get(name), (list, tuple)):
            fields[name] = tuple(tuple(row) if isinstance(row, (list, tuple)) else row
                                 for row in fields[name])
    for name in ("n_ori", "n_cmp", "scale_encoding"):
        if isinstance(fields.get(name), list):
            fields[name] = tuple(fields[name])
    sinks = fields.get("sinks")
    if isinstance(sinks, (tuple, list)) and len(sinks) == 2 and isinstance(sinks[1], (tuple, list)):
        fields["sinks"] = (sinks[0], tuple(sinks[1]))
    try:
        return PagedMlaContract(**fields)
    except TypeError as exc:
        raise ModelingError(f"paged_mla_invalid_contract:{exc}") from exc

def contract_for(op: OpSpec, ctx: OperatorPlanningContext) -> PagedMlaContract:
    metadata = dict(op.metadata)
    raw = metadata.get(CONTRACT_KEY)
    if not isinstance(raw, Mapping):
        raise ModelingError("paged_mla_contract_required")
    requested = ctx.requested_abi or metadata.get("requested_abi")
    if requested and "abi" in raw and requested != raw["abi"]:
        raise capability_gap("paged_mla_abi_conflict", requested, ctx.compute_arch.arch_id)
    return validate_contract(_from_mapping({**raw, **({"abi": requested} if requested else {})}))

def make_op(
    contract: PagedMlaContract | Mapping | None = None, *,
    op_id: str = FAMILY, dtype: str = "bf16", phase: str = "decode",
    stage: int = 0, layer: int = 0, deps: tuple[str, ...] = (), **contract_fields,
) -> OpSpec:
    """Attach a serializable tensor and ABI declaration to a paged MLA occurrence."""
    if contract is not None and contract_fields:
        raise ModelingError("paged_mla_contract_declared_twice")
    c = validate_contract(contract if isinstance(contract, PagedMlaContract)
                          else _from_mapping(contract_fields if contract is None else contract))
    tokens = sum(q for q, _ in c.sequences)
    logical, padded = (tokens, c.logical_heads, c.dqk), (tokens, c.physical_heads, c.dqk)
    return OpSpec(
        op_id=op_id, family=FAMILY, phase=phase, stage=stage, layer=layer,
        requests=tuple(f"seq{i}" for i in range(len(c.sequences))),
        semantic_shape=logical, local_shape=logical, padded_shape=padded, launch_shape=padded,
        dtype=dtype, layout=c.kv_layout, recipe=FAMILY, coverage="source",
        deps=deps, flops=c.useful_matrix_flops,
        returns=("softmax_lse",) if c.return_lse else (),
        metadata=((CONTRACT_KEY, {f.name: getattr(c, f.name) for f in fields(c)}),
                  ("requested_abi", c.abi)),
    )
