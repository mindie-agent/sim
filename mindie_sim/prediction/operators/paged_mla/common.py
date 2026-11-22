"""Shared paged MLA tensor contract and ABI validation.

[Role] Declare dense and mixed-quant paged MLA tensor inputs.
[Invariants] Preserve active extents, allocation geometry, and ABI identity.
[Timeouts & Locks] Validation is bounded by the input metadata. No locks.
[Side-Effect Policy] Construct immutable records without I/O.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from itertools import chain
from math import ceil

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from ..contracts import OperatorPlanningContext
from ..attention.common import planning_sequence_spans

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
ACLNN_MQS_INLINE_ROW_BYTES = 608
ACLNN_MQS_DATA_BYTES = 576
ACLNN_MQS_SCALE_BYTES = 8
ACLNN_MQS_SEPARATE_SCALE_MODE = 2
ACLNN_MQS_SCALE_ENCODINGS = {1: "bf16_g64", ACLNN_MQS_SEPARATE_SCALE_MODE: "e8m0_g64"}
MQS_PAGED_LAYOUT = "PA_BBND"
MQS_CONTIGUOUS_LAYOUTS = ("BSND", "TND")
MQS_MAX_PAGE_SIZE = 1024
FP32_BYTES = 4
INDEX_BYTES = 4
NZ_BLOCK_BYTES = 32
MASK_NONE = 0
MASK_CAUSAL = 3
MASK_WINDOW = 4
MAX_COMPRESSION_RATIO = 128
QUERY_INDEX_NESTING = 3
BankCounts = int | tuple[int | tuple[int, ...], ...]
IndexRow = tuple[int, ...]
BankIndices = tuple[IndexRow | tuple[IndexRow, ...], ...]


@dataclass(frozen=True, slots=True)
class PagedMlaContract:
    """Active query and KV extents, with optional DSL bank rows per query.

    A nested bank row supplies one entry per query in its sequence. Flat bank
    rows are batch-indexed, or query-indexed when their length matches only the
    query count. With per-query counts, the sequence KV extent bounds each row.
    Paged cache tables omitted from the analytical contract denote identity
    mapping. For DSL MQSMLA, splits caps the cores participating in FD planning.
    """

    sequences: tuple[tuple[int, int], ...]
    logical_heads: int
    physical_heads: int
    latent_dim: int = DEFAULT_LATENT_DIM
    rope_dim: int = DEFAULT_ROPE_DIM
    page_size: int = DEFAULT_PAGE_SIZE
    block_table_len: int | None = None
    kv_layout: str | None = None
    abi: str = FIA_ABI
    s2_base_size: int = DEFAULT_S2_BASE_SIZE
    splits: int = 1
    return_lse: bool = False
    latent_alias: bool = True
    n_ori: BankCounts = 0
    n_cmp: BankCounts = 0
    quant_mode: int = 1
    cmp_ratio: int = 1
    ori_mask_mode: int = MASK_NONE
    cmp_mask_mode: int = MASK_NONE
    topk_value_mode: int = 1
    scale_encoding: str | tuple[str, str] | None = None
    mqsmla_dim: int = MQS_HEAD_DIM
    block_table: tuple[tuple[int, ...], ...] = ()
    ori_block_table: tuple[tuple[int, ...], ...] = ()
    cmp_block_table: tuple[tuple[int, ...], ...] = ()
    ori_indices: BankIndices = ()
    cmp_indices: BankIndices = ()
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

    def bank_counts(self, seq: int, query: int) -> tuple[int, int]:
        ori, cmp = self.n_ori, self.n_cmp
        return (_query_row(ori[seq], query) if isinstance(ori, tuple) else ori,
                _query_row(cmp[seq], query) if isinstance(cmp, tuple) else cmp)

    def query_runs(self, seq: int) -> Iterator[tuple[int, int]]:
        """Yield the first query and number of queries sharing its inputs."""
        queries = self.sequences[seq][0]
        if not queries:
            return
        if any(_per_query(rows[seq], indices=indices) for rows, indices in (
            (self.n_ori, False), (self.n_cmp, False), (self.ori_indices, True), (self.cmp_indices, True),
        ) if isinstance(rows, tuple) and rows):
            yield from ((query, 1) for query in range(queries))
        else:
            yield 0, queries

    @property
    def selected_pairs(self) -> int:
        if not self.mixed_quant:
            return sum(q * kv for q, kv in self.sequences)
        pairs = 0
        for name in ("n_ori", "n_cmp"):
            counts = getattr(self, name)
            if type(counts) is int:
                pairs += sum(q for q, _ in self.sequences) * counts
            else:
                rows = _batch_rows(self, name, counts)
                pairs += sum(sum(row) if isinstance(row, tuple) else q * row
                             for row, (q, _) in zip(rows, self.sequences, strict=True))
        return pairs

    @property
    def useful_matrix_flops(self) -> int:
        return 2 * self.selected_pairs * self.logical_heads * (self.dqk + self.dv)

    @property
    def issued_matrix_flops(self) -> int:
        return 2 * self.selected_pairs * self.physical_heads * (self.dqk + self.dv)

def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor

def align_bytes(value: int, alignment: int = NZ_BLOCK_BYTES) -> int:
    return ceil_div(value, alignment) * alignment

def _extent(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"paged_mla_invalid_{name}:{value!r}")

def mqsmla_payload_bytes(n_ori: int, n_cmp: int) -> int:
    """Return the DSL ABI's FP8 and packed E2M1 payload before transactions."""
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

def _per_query(row, *, indices: bool = False) -> bool:
    return isinstance(row, tuple) and (not indices or bool(row) and isinstance(row[0], tuple))

def _query_row(row, query: int, *, indices: bool = False):
    return row[query] if _per_query(row, indices=indices) else row

def _batch_rows(c: PagedMlaContract, name: str, rows: tuple) -> tuple:
    if len(rows) == len(c.sequences):
        return rows
    if c.abi == DSL_MQS_ABI and len(rows) == sum(q for q, _ in c.sequences):
        entries = iter(rows)
        return tuple(tuple(next(entries) for _ in range(q)) for q, _ in c.sequences)
    raise ModelingError(f"paged_mla_invalid_{name}_batch")

def _validate_lists(name: str, values: tuple, sequences: int) -> None:
    if not isinstance(values, tuple) or (values and len(values) != sequences):
        raise ModelingError(f"paged_mla_invalid_{name}_batch")
    if any(not isinstance(row, tuple) for row in values):
        raise ModelingError(f"paged_mla_invalid_{name}_list")
    for value in chain.from_iterable(values):
        _extent(name, value)

def _validate_bank_rows(c: PagedMlaContract, name: str, *, indices: bool = False):
    rows = getattr(c, name)
    if not indices and not isinstance(rows, tuple):
        _extent(name, rows)
        return rows
    if indices and rows == ():
        return rows
    if not isinstance(rows, tuple):
        raise ModelingError(f"paged_mla_invalid_{name}_batch")
    rows = _batch_rows(c, name, rows)
    for seq, (row, (queries, _)) in enumerate(zip(rows, c.sequences, strict=True)):
        per_query = _per_query(row, indices=indices)
        if per_query and (c.abi != DSL_MQS_ABI or len(row) != queries):
            raise ModelingError(f"paged_mla_invalid_{name}_query_count:{seq}")
        values = row if per_query else (row,)
        if indices:
            if any(not isinstance(value, tuple) for value in values):
                raise ModelingError(f"paged_mla_invalid_{name}_list")
            values = chain.from_iterable(values)
        for value in values:
            _extent(name, value)
    return rows

def _validate_mixed(c: PagedMlaContract, *, capacity_extents: bool = False) -> PagedMlaContract:
    c = replace(c, **{
        name: _validate_bank_rows(c, name, indices=name.endswith("_indices"))
        for name in ("n_ori", "n_cmp", "ori_indices", "cmp_indices")
    })
    dsl = c.abi == DSL_MQS_ABI
    modes = (1,) if dsl else ACLNN_MQS_SCALE_ENCODINGS
    if type(c.quant_mode) is not int or c.quant_mode not in modes:
        raise ModelingError(f"paged_mla_invalid_quant_mode:{c.abi}:{c.quant_mode}")
    layouts = (MQS_PAGED_LAYOUT,)
    if not dsl and c.quant_mode != ACLNN_MQS_SEPARATE_SCALE_MODE:
        layouts += MQS_CONTIGUOUS_LAYOUTS
    if c.kv_layout not in layouts:
        raise ModelingError(f"paged_mla_invalid_kv_layout:{c.abi}:{c.quant_mode}:{c.kv_layout}")
    if c.kv_layout == MQS_PAGED_LAYOUT:
        if c.page_size > MQS_MAX_PAGE_SIZE:
            raise ModelingError(f"paged_mla_invalid_page_size:{c.page_size}")
    elif c.ori_block_table or c.cmp_block_table or c.block_table_len is not None:
        raise ModelingError(f"paged_mla_block_tables_require_paged_kv:{c.kv_layout}")
    if type(c.cmp_ratio) is not int or not 1 <= c.cmp_ratio <= MAX_COMPRESSION_RATIO:
        raise ModelingError(f"paged_mla_invalid_cmp_ratio:{c.cmp_ratio}")
    for name, value, modes in (
        ("ori_mask_mode", c.ori_mask_mode, {MASK_NONE, MASK_CAUSAL, MASK_WINDOW}),
        ("cmp_mask_mode", c.cmp_mask_mode, {MASK_NONE, MASK_CAUSAL}),
        ("topk_value_mode", c.topk_value_mode, {1}),
    ):
        if type(value) is not int or value not in modes:
            raise ModelingError(f"paged_mla_invalid_{name}:{value}")
    if dsl:
        if c.physical_heads != DSL_MQS_HEADS:
            raise ModelingError(f"paged_mla_dsl_mqsmla_requires_heads:{DSL_MQS_HEADS}")
        has_compressed = (any(any(row) if isinstance(row, tuple) else row for row in c.n_cmp)
                          if isinstance(c.n_cmp, tuple) else bool(c.n_cmp))
        default = DSL_MQS_SCALE_ENCODINGS
        allowed = (default,) if has_compressed else (default, default[:1])
    else:
        if c.rope_dim != DEFAULT_ROPE_DIM:
            raise ModelingError(f"paged_mla_mqsmla_requires_rope_dim:{DEFAULT_ROPE_DIM}")
        default = ACLNN_MQS_SCALE_ENCODINGS[c.quant_mode]
        allowed = ((default,), (default, default))
    encoding = default if c.scale_encoding is None else c.scale_encoding
    encodings = (encoding,) if isinstance(encoding, str) else encoding
    if encodings not in allowed:
        raise ModelingError(f"paged_mla_invalid_scale_encoding:{c.abi}:{c.quant_mode}:{encoding!r}")
    c = replace(c, scale_encoding=encoding)
    if type(c.mqsmla_dim) is not int or c.mqsmla_dim != MQS_HEAD_DIM or c.rope_dim >= c.mqsmla_dim:
        raise ModelingError("paged_mla_mqsmla_requires_packed_width_512")
    for seq, (_, kv) in enumerate(c.sequences):
        per_query_counts = any(isinstance(rows, tuple) and isinstance(rows[seq], tuple)
                               for rows in (c.n_ori, c.n_cmp))
        for query, _ in c.query_runs(seq):
            ori, cmp = c.bank_counts(seq, query)
            if ori + cmp > kv or (not capacity_extents and not per_query_counts and ori + cmp != kv):
                raise ModelingError(f"paged_mla_sparse_pair_coverage:{seq}:{query}:{ori}+{cmp}:{kv}")
            for branch, count, indices, table in (
                ("ori", ori, c.ori_indices, c.ori_block_table),
                ("cmp", cmp, c.cmp_indices, c.cmp_block_table),
            ):
                selected = _query_row(indices[seq], query, indices=True) if indices else ()
                if indices and len(selected) != count:
                    raise ModelingError(f"paged_mla_{branch}_index_count:{seq}:{query}")
                extent = max(selected, default=-1) + 1 if indices else count
                pages = ceil_div(extent, c.page_size)
                if table and pages > len(table[seq]):
                    raise ModelingError(f"paged_mla_{branch}_block_table_coverage:{seq}:{query}")
                if c.block_table_len is not None and pages > c.block_table_len:
                    raise ModelingError(f"paged_mla_{branch}_block_table_len:{seq}:{query}")
    return c

def validate_contract(c: PagedMlaContract, *, capacity_extents: bool = False) -> PagedMlaContract:
    """Check shape, declared address coverage, and branch-local MQS attributes."""
    for name in ("logical_heads", "physical_heads", "latent_dim", "page_size", "s2_base_size", "splits"):
        _extent(name, getattr(c, name), 1)
    _extent("rope_dim", c.rope_dim)
    if c.physical_heads < c.logical_heads:
        raise ModelingError("paged_mla_physical_heads_below_logical")
    if c.block_table_len is not None:
        _extent("block_table_len", c.block_table_len)
    if c.abi not in SUPPORTED_ABIS:
        raise capability_gap("paged_mla_unsupported_abi", c.abi)
    if c.kv_layout is None:
        c = replace(c, kv_layout=MQS_PAGED_LAYOUT if c.mixed_quant else "PA_BLK_BSND")
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
    for name in ("block_table", "ori_block_table", "cmp_block_table"):
        _validate_lists(name, getattr(c, name), len(c.sequences))
    if c.mixed_quant:
        if c.block_table:
            raise ModelingError("paged_mla_mqsmla_requires_branch_local_block_tables")
        c = _validate_mixed(c, capacity_extents=capacity_extents)
    else:
        if c.kv_layout not in ("PA_BLK_BSND", "PA_BLK_NZ", "NZ_LATENT"):
            raise ModelingError(f"paged_mla_invalid_kv_layout:{c.kv_layout}")
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

def _tuple_rows(value, depth: int):
    if depth and isinstance(value, (tuple, list)):
        if depth == 1:
            return tuple(value)
        return tuple(_tuple_rows(row, depth - 1) for row in value)
    return value

def _from_mapping(raw: Mapping) -> PagedMlaContract:
    fields = dict(raw)
    for names, depth in (
        (("sequences", "block_table", "ori_block_table", "cmp_block_table", "n_ori", "n_cmp"), 2),
        (("ori_indices", "cmp_indices"), QUERY_INDEX_NESTING),
        (("scale_encoding",), 1),
    ):
        for name in names:
            if name in fields:
                fields[name] = _tuple_rows(fields[name], depth)
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
    c = _from_mapping({**raw, **({"abi": requested} if requested else {})})
    spans = planning_sequence_spans(ctx, c.kv_layout or "PA_BLK_BSND")
    if not spans:
        return validate_contract(c)
    sequences = tuple((q, kv) for q, kv, _, _, _ in spans)
    active = sum(not span[-1] for span in spans)
    padded_fields = {}
    for name in ("block_table", "ori_block_table", "cmp_block_table", "ori_indices", "cmp_indices", "n_ori", "n_cmp"):
        rows = getattr(c, name)
        if isinstance(rows, tuple) and rows and len(rows) == active and active < len(spans):
            values = iter(rows)
            fill = 0 if name in {"n_ori", "n_cmp"} else ()
            padded_fields[name] = tuple(fill if span[-1] else next(values) for span in spans)
    return validate_contract(replace(c, sequences=sequences, **padded_fields), capacity_extents=True)

ARCH22_MLA_L0A_RELOAD_PANELS = 4
ARCH22_MLA_CROSSCORE_SYNCS_SCORE = 2
ARCH22_MLA_CROSSCORE_SYNCS_PROB = 2
ARCH22_MLA_CROSSCORE_SYNCS_PARTIAL = 1
ARCH35_MLA_CROSSCORE_SYNCS_STEP = 1
NS_PER_MS_MLA = 1_000_000.0


def mla_step_crosscore_ms(ctx, step: str, *, gm_workspace: bool, abi: str) -> float:
    """Calibrated P13 CrossCore handoff latency per MLA pipeline step (0.0 when nominal)."""
    pt = getattr(ctx, "primitive_table", None)
    if pt is None:
        return 0.0
    if gm_workspace:
        mult = {
            "score_store": ARCH22_MLA_CROSSCORE_SYNCS_SCORE,
            "probability": ARCH22_MLA_CROSSCORE_SYNCS_PROB,
            "partial": ARCH22_MLA_CROSSCORE_SYNCS_PARTIAL,
        }.get(step, 0)
    elif abi == FIA_ABI:
        mult = ARCH35_MLA_CROSSCORE_SYNCS_STEP if step in {"score_store", "probability", "partial"} else 0
    else:
        return 0.0
    if not mult:
        return 0.0
    from mindie_sim.primitives.loader import _table_band_p50
    ns = _table_band_p50(pt, "P13_crosscore_sync", "mode2_paired_ns")
    return (float(mult) * float(ns) / NS_PER_MS_MLA) if (ns is not None and ns > 0.0) else 0.0


PAGED_MLA_MIN_SOFTMAX_OPS = 4


def _minimum_contract_io(c: PagedMlaContract, dtype: str) -> tuple[float, float, float, float]:
    """Conservative lower bound on (hbm, mte2, mte3, vector_ops) for L* soundness."""
    tokens = sum(q for q, _ in c.sequences)
    elem_b = float(dtype_bytes(dtype))
    q_bytes = float(tokens * c.logical_heads * c.dqk) * elem_b
    o_bytes = float(tokens * c.logical_heads * c.dv) * elem_b
    total_pairs = c.selected_pairs
    vector_ops = float(PAGED_MLA_MIN_SOFTMAX_OPS * c.logical_heads * total_pairs)
    total_kv = sum(kv for _, kv in c.sequences)
    unique_kv = min(total_kv, total_pairs) if c.mixed_quant else total_kv
    kv_row_b = float(MQS_COMPRESSED_ROW_BYTES if c.mixed_quant else latent_record_bytes(c, dtype=dtype))
    kv_bytes = float(unique_kv) * kv_row_b
    return q_bytes + kv_bytes + o_bytes, q_bytes + kv_bytes, o_bytes, vector_ops


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
    hbm_bytes, mte2_bytes, mte3_bytes, vector_ops = _minimum_contract_io(c, dtype)
    return OpSpec(
        op_id=op_id, family=FAMILY, phase=phase, stage=stage, layer=layer,
        requests=tuple(f"seq{i}" for i in range(len(c.sequences))),
        semantic_shape=logical, local_shape=logical, padded_shape=padded, launch_shape=padded,
        dtype=dtype, layout=c.kv_layout, recipe=FAMILY, coverage="source",
        deps=deps, flops=c.useful_matrix_flops, vector_ops=vector_ops,
        hbm_bytes=hbm_bytes, mte2_bytes=mte2_bytes, mte3_bytes=mte3_bytes,
        returns=("softmax_lse",) if c.return_lse else (),
        metadata=((CONTRACT_KEY, {f.name: getattr(c, f.name) for f in fields(c)}),
                  ("requested_abi", c.abi)),
    )
