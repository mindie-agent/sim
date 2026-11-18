"""Mixed-quant sparse MLA with resident dequantized KV and paired AIV owners.

[Role] Price the retained whole-row, scalar-address source branch.
[Invariants] One query's original and compressed lists share online state.
  A padded tile repeats valid KV reads; QK computes 128 rows and PV rounds its
  valid prefix to 16. UB/L1 exchanges never become invented HBM workspaces.
[Timeouts & Locks] O(active query rows) metadata; two live phase tiles per owner.
[Side-Effect Policy] Pure plans and the common causal Engine, no observations.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .instructions import PrimitiveInventory

#: CANN 9.2.1 official source (`aclnnMixedQuantSparseFlashMla`).
SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "nightly-ci-main-a5-openeuler-cann9.2.1-20261111",
    "source_state": "committed",
    "path": "vllm_ascend/attention/mixed_quant_sparse_flash_mla.py",
    "symbol": "mixed_quant_sparse_flash_mla",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
#: Retained prototype alias; explicit selection keeps its own guard rails.
LEGACY_SOURCE = {
    "repository": "cannbot-dsl",
    "revision": "4519e5372a8b9306f7d823688c0cb8141ee34fb1",
    "kernel": {
        "path": "samples/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
        "sha256": "40b5b14ced15bb7c3d3687ab7387d508a19065eb5353dfb0675348e0de22fa3f",
    },
    "metadata": {
        "path": "samples/mixed_quant_sparse_flash_mla_metadata/mixed_quant_sparse_flash_mla_metadata.py",
        "sha256": "a62295be0fdf6acc7c0c606e60424e3fbc03310bee4e1effa2a384ccd6415b65",
    },
    "capture_binary_identity": "not_established",
}
RECIPE = "aclnn_mixed_quant_sparse_flash_mla"
LEGACY_RECIPE = "cannbot_mixed_quant_sparse_flash_mla"
NATIVE_KIND = "aclnnMixedQuantSparseFlashMla"
LEGACY_NATIVE_KIND = "MqsmlaKernel___call__"
SUPPORTED_PHASES = frozenset({"decode", "draft"})

# Source Channel depths. BF16 output aliases the FP32 accumulator. The P NZ
# buffers include one 16-element pad per N1 face; all scalar state is FP32.
BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144,
           "ub": 2 * 32 * 128 * 4 + 2 * 32 * 512 * 4
                 + 2 * (32 * 128 + 8 * 16) * 2
                 + 4 * 16 * 544 + 2 * 32 * 17 * 16 * 2 + 10 * 32 * 4}

# The packaged source shards query heads by TP and replicates the single MQA
# KV head. ``local_heads = num_heads // tp`` therefore selects the QK cube
# N-face count; the values below are the declared TP tilings whose M-tile still
# aligns to the 16-row cube face.
DEFAULT_MQSMLA_HEADS = 64
MQSMLA_HEAD_WIDTH = 512
MQSMLA_ROPE_WIDTH = 64
CUBE_HEAD_TILE_ALIGN = 16
#: Source scalar address prepass: eight 16-row subtiles, one 512-element row
#: pair of GM reads, and four scalar-ALU issues per address row.
ADDRESS_SUBTILES_PER_TILE = 8
ADDRESS_ROWS_PER_SUBTILE = 16
ADDRESS_SCALAR_READS = 2
ADDRESS_SCALAR_ALU_ISSUES = 4
MIN_PADDED_LOCAL_HEADS = 8
HEAD_GROUP_START = 0
# Packed row widths: FP8 values plus one BF16 scale per 32 values, and two FP4
# values per byte plus one BF16 scale per 16 values.
ORIGINAL_CACHE_ROW_BYTES = 544
COMPRESSED_CACHE_ROW_BYTES = 320
PHYSICAL_KV_TILE_ROWS = 128
GATHER_SUBTILE_ROWS = 16
BF16_ELEMENT_BYTES = 2
FP32_ELEMENT_BYTES = 4
ORIGINAL_DEQUANT_OPS_PER_ELEM = 6
COMPRESSED_DEQUANT_OPS_PER_ELEM = 4
SOFTMAX_PAIR_OPS = 8
SOFTMAX_ROW_REDUCE_OPS = 26
ACCUM_NONFIRST_OPS = 3
# Declared AIC ownership bounds of the source wrapper.
MAX_MQSMLA_AIC_CORES = 36
# The retained SELECTIVE_FD_72 wrapper is a distinct split-reduction geometry
# that this whole-row program does not model.
SELECTIVE_FD_AIC_CORES = 32
SELECTIVE_FD_QUERY_ROWS = 72
SELECTIVE_FD_ORIGINAL_ROWS = 128
SELECTIVE_FD_COMPRESSED_ROWS = 512
#: ``aclnnMixedQuantSparseFlashMla`` packed-KV modes of the arch35 kernel.
KV_QUANT_BF16 = 0
KV_QUANT_FP8 = 1
KV_QUANT_MXFP8 = 2
KV_QUANT_MODES = frozenset({KV_QUANT_BF16, KV_QUANT_FP8, KV_QUANT_MXFP8})
#: Packed row widths of one 512-wide latent slot plus its 64 rotary lanes.
KV_QUANT_BF16_ROW_BYTES = 512 * BF16_ELEMENT_BYTES
KV_QUANT_FP8_ROW_BYTES = 576 + 16 * BF16_ELEMENT_BYTES
KV_QUANT_MXFP8_ROW_BYTES = 576 + 8
KV_QUANT_ROW_BYTES = {
    KV_QUANT_BF16: KV_QUANT_BF16_ROW_BYTES,
    KV_QUANT_FP8: KV_QUANT_FP8_ROW_BYTES,
    KV_QUANT_MXFP8: KV_QUANT_MXFP8_ROW_BYTES,
}
#: Declared vector pass inventory of the dequantized operand publish.
#: ``qm=1`` keeps the source's FP8 cast + per-32 scale multiply + NZ move.
#: ``qm=2`` adds the UE8M0 group unpack, exponent shift and broadcast multiply.
KV_QUANT_DEQUANT_OPS = {
    KV_QUANT_BF16: 0,
    KV_QUANT_FP8: 5,
    KV_QUANT_MXFP8: 10,
}
#: Live KV/P/alpha slots of the double-buffered inter-tile pipeline.
QUANT_TILE_SLOTS = 2
#: Source AICPU metadata program plus kernel launch/drain before the tile loop
#: (paired-silicon intercept; the tile pipeline is fitted separately).
METADATA_PROGRAM_MS = 0.020
# Distinct (branch, rows, first, last, addressing, heads) phase payloads.
TILE_PHASE_CACHE_SIZE = 512
OWNER_STAGE_CACHE_SIZE = 16


def supported_heads(heads):
    """Keep the padded eight-head shard and complete 16-head cube faces."""
    return (heads == MIN_PADDED_LOCAL_HEADS
            or heads >= CUBE_HEAD_TILE_ALIGN and heads % CUBE_HEAD_TILE_ALIGN == HEAD_GROUP_START)


def _phase_key(stream, head_counts, index):
    return (*stream.key(index), head_counts[bisect_right(stream.ends, index)])


def selective_fd_excluded(cores, rows):
    """True for the retained SELECTIVE_FD_72 wrapper geometry.

    One predicate is shared by the builder and by automatic binding preflight
    so a policy can never select a geometry the planner would reject.
    """
    return (cores == SELECTIVE_FD_AIC_CORES
            and len(rows) == SELECTIVE_FD_QUERY_ROWS
            and all(tuple(row) == (SELECTIVE_FD_ORIGINAL_ROWS,
                                  SELECTIVE_FD_COMPRESSED_ROWS) for row in rows))


def query_rows(shape):
    """Global query order preserves each sequence's causal context."""
    window, ratio, topk = (shape[key] for key in ("window", "ratio", "topk"))
    for query, context in shape["sequences"]:
        for offset in range(query):
            position = context + offset + 1
            original = min(position, window) if window else 0
            compressed = position // ratio if ratio else 0
            yield original, min(compressed, topk) if topk else compressed


def owner_ranges(rows, cores):
    """Exact metadata SUPPORT_FD=False contiguous q/r row ownership."""
    q, r = divmod(rows, cores)
    start = 0
    result = []
    for core in range(cores):
        count = q + (core < r)
        if count:
            result.append((core, start, start + count))
        start += count
    return tuple(result)


def addr_tab_w(k):
    """Source pads address-table columns to 16."""
    return ((int(k) + 15) // 16) * 16


def _addr_pool_eligible(pool):
    page = pool.get("page_size")
    row = pool.get("cache_row_bytes")
    bt = pool.get("bt_width")
    idx = pool.get("idx_width")
    ub = pool.get("addr_ub_bytes")
    if not all(isinstance(v, int) and v > 0 for v in (page, row, bt, idx, ub)):
        return False
    if page < 16 or page & (page - 1):
        return False
    if pool.get("cache_stride_bytes") != page * row:
        return False
    return 4 * (addr_tab_w(bt) + 2 * addr_tab_w(idx)) <= ub


def addr_vec_eligible(shape):
    """Source eligibility must hold for both ORI and CMP pools.

    No global single-pool shortcut: ORI and CMP run sequentially through the
    same arena and both are tested before vector addressing is selected.
    """
    pools = shape.get("address_pools") or {}
    original = pools.get("original")
    compressed = pools.get("compressed")
    if not original or not compressed:
        return False
    if not (_addr_pool_eligible(original) and _addr_pool_eligible(compressed)):
        return False
    return bool(compressed.get("present", True))


@lru_cache(maxsize=TILE_PHASE_CACHE_SIZE)
def tile_phases(branch, valid, first, last, addressing="scalar",
                heads=DEFAULT_MQSMLA_HEADS):
    """Source phase payloads, parameterized by the local query-head count."""
    sinks_bytes = heads * FP32_ELEMENT_BYTES
    row_io_bytes = heads * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    if not valid:
        return (
            ("sinks", Work(resources={"hbm": sinks_bytes, "mte2_aiv": sinks_bytes})),
            ("empty", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH})),
            ("output", Work(resources={"hbm": row_io_bytes, "mte3": row_io_bytes})),
        )
    physical = PHYSICAL_KV_TILE_ROWS
    row_bytes = (ORIGINAL_CACHE_ROW_BYTES if branch == "original"
                 else COMPRESSED_CACHE_ROW_BYTES)
    # The QK M face keeps at least one aligned head face; wider TP shards round
    # their local heads up to the same 16-row granularity.
    cube_heads = max(
        CUBE_HEAD_TILE_ALIGN,
        ((heads + CUBE_HEAD_TILE_ALIGN - 1) // CUBE_HEAD_TILE_ALIGN) * CUBE_HEAD_TILE_ALIGN,
    )
    # Four 16-row subtiles per AIV; duplicate tail indices still issue copies.
    padded_pv = (valid + CUBE_HEAD_TILE_ALIGN - 1) // CUBE_HEAD_TILE_ALIGN * CUBE_HEAD_TILE_ALIGN
    q = row_io_bytes if first else 0
    output = row_io_bytes if last else 0
    phases = [
        ("query", Work(resources={"hbm": q, "mte2_aic": q})),
        ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                  "mte2_aiv": sinks_bytes if first else 0})),
    ]
    subtile = GATHER_SUBTILE_ROWS
    for part in range(physical // subtile):
        sub_bytes = subtile * FP32_ELEMENT_BYTES
        row_bytes_part = subtile * row_bytes
        phases.extend((
            (f"addr_index_{part}", Work(resources={"hbm": sub_bytes})),
            (f"addr_page_{part}", Work(resources={"hbm": sub_bytes})),
            (f"gather_{part}", Work(resources={"hbm": row_bytes_part,
                                               "mte2_aiv": row_bytes_part})),
        ))
    phases.append(("addr_join", Work()))
    dequant_ops = (ORIGINAL_DEQUANT_OPS_PER_ELEM if branch == "original"
                   else COMPRESSED_DEQUANT_OPS_PER_ELEM)
    phases.extend((
        ("dequant", Work(resources={"vector": physical * MQSMLA_HEAD_WIDTH * dequant_ops})),
        ("kv_l1", Work(resources={"mte3": physical * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES})),
        ("qk_operands", Work(resources={
            "mte1": (cube_heads + physical) * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES})),
        ("qk", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * physical * MQSMLA_HEAD_WIDTH})),
        ("scores", Work(resources={"fixpipe": cube_heads * physical * FP32_ELEMENT_BYTES})),
        ("softmax", Work(resources={
            "vector": SOFTMAX_PAIR_OPS * heads * valid + SOFTMAX_ROW_REDUCE_OPS * heads})),
        ("probability_l1", Work(resources={
            "mte3": cube_heads * physical * BF16_ELEMENT_BYTES})),
        ("pv_operands", Work(resources={
            "mte1": (cube_heads * physical + physical * MQSMLA_HEAD_WIDTH)
                    * BF16_ELEMENT_BYTES})),
        ("pv", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * padded_pv * MQSMLA_HEAD_WIDTH})),
        ("values", Work(resources={
            "fixpipe": cube_heads * MQSMLA_HEAD_WIDTH * FP32_ELEMENT_BYTES})),
        ("accumulate", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH * (
            (1 if first else ACCUM_NONFIRST_OPS) + (BF16_ELEMENT_BYTES if last else 0))})),
        ("output", Work(resources={"hbm": output, "mte3": output})),
    ))
    return tuple(phases)


@dataclass(frozen=True)
class OwnerTiles:
    """Index finite source tiles using O(query rows), never O(key tiles)."""
    rows: tuple
    ends: tuple

    @classmethod
    def create(cls, rows):
        ends, total = [], 0
        for original, compressed in rows:
            total += max(1, (original + 127) // 128 + (compressed + 127) // 128)
            ends.append(total)
        return cls(tuple(rows), tuple(ends))

    def key(self, index):
        row = bisect_right(self.ends, index)
        local = index - (self.ends[row - 1] if row else 0)
        original, compressed = self.rows[row]
        nt = (original + 127) // 128
        count = self.ends[row] - (self.ends[row - 1] if row else 0)
        if local < nt:
            return "original", min(128, original - local * 128), local == 0, local + 1 == count
        return "compressed", max(0, min(128, compressed - (local - nt) * 128)), local == 0, local + 1 == count

    @property
    def count(self):
        return self.ends[-1]


def quant_row_bytes(quant_mode):
    if quant_mode not in KV_QUANT_MODES:
        raise ModelingError("mqsmla_kv_quant_mode_unsupported")
    return {KV_QUANT_BF16: KV_QUANT_BF16_ROW_BYTES,
            KV_QUANT_FP8: KV_QUANT_FP8_ROW_BYTES,
            KV_QUANT_MXFP8: KV_QUANT_MXFP8_ROW_BYTES}[quant_mode]


def quant_tile_phases(valid, first, last, *, heads, quant_mode, window_bytes=0):
    """One 128-key tile of the arch35 double-buffered inter-tile pipeline.

    Address/gather/dequant/publish of tile ``i+1`` overlap the AIC ``QK ->
    Fixpipe -> online softmax -> PV`` chain of tile ``i``; the copy, vector and
    publish pipes of the ``2-AIV`` block are separate stages so the engine can
    interleave them across the live slots.  ``window_bytes`` is the private SWA
    ring a query token re-reads; the shared TopK union is a cold fill instead.
    """
    row_bytes = quant_row_bytes(quant_mode)
    dequant_ops = KV_QUANT_DEQUANT_OPS[quant_mode]
    sinks_bytes = heads * FP32_ELEMENT_BYTES
    cube_heads = max(CUBE_HEAD_TILE_ALIGN,
                     -(-heads // CUBE_HEAD_TILE_ALIGN) * CUBE_HEAD_TILE_ALIGN)
    if not valid:
        return (
            ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                      "mte2_aiv": sinks_bytes if first else 0}), ()),
            ("empty", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH}), ()),
        )
    return (
        ("sinks", Work(resources={"hbm": sinks_bytes if first else 0,
                                  "mte2_aiv": sinks_bytes if first else 0}), ()),
        ("address", Work(resources={
            "mte2_aiv": valid * ADDRESS_SCALAR_READS * FP32_ELEMENT_BYTES}), ()),
        ("gather", Work(resources={
            "mte2_aiv": valid * row_bytes + window_bytes}), ("address",)),
        ("dequant", Work(resources={
            "vector": valid * MQSMLA_HEAD_WIDTH * dequant_ops}), ("gather",)),
        ("publish", Work(resources={
            "mte3": valid * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES}), ("dequant",)),
        ("qk", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * valid * MQSMLA_HEAD_WIDTH},
            resources={"mte1": (cube_heads + valid) * MQSMLA_HEAD_WIDTH
                              * BF16_ELEMENT_BYTES,
                       "fixpipe": cube_heads * valid * FP32_ELEMENT_BYTES}),
         ("publish",)),
        ("softmax", Work(resources={
            "vector": SOFTMAX_PAIR_OPS * heads * valid
                      + SOFTMAX_ROW_REDUCE_OPS * heads}), ("qk", "sinks")),
        ("pv", Work(matrix={
            "bf16": BF16_ELEMENT_BYTES * cube_heads * valid * MQSMLA_HEAD_WIDTH},
            resources={"mte1": (cube_heads + MQSMLA_HEAD_WIDTH) * valid
                              * BF16_ELEMENT_BYTES,
                       "fixpipe": cube_heads * MQSMLA_HEAD_WIDTH * FP32_ELEMENT_BYTES}),
         ("softmax",)),
        ("accumulate", Work(resources={"vector": heads * MQSMLA_HEAD_WIDTH * (
            (1 if first else ACCUM_NONFIRST_OPS)
            + (BF16_ELEMENT_BYTES if last else 0))}), ("pv",)),
    )


@lru_cache(maxsize=TILE_PHASE_CACHE_SIZE)
def quant_tile_phases_cached(valid, first, last, heads, quant_mode, window_bytes=0):
    """Bounded cache of the pure quant-mode tile payloads."""
    return quant_tile_phases(valid, first, last, heads=heads,
                             quant_mode=quant_mode, window_bytes=window_bytes)


def build(op, geometry):
    shape = dict(op.metadata)["mqsmla_geometry"]
    requested = str(shape.get("addressing", "scalar"))
    implementation = str(shape.get("implementation", RECIPE))
    legacy = implementation == LEGACY_RECIPE
    heads = int(shape["heads"])
    if (not supported_heads(heads) or shape["width"] != MQSMLA_HEAD_WIDTH
            or shape["metadata_support_fd"]):
        raise ModelingError("mqsmla_source_geometry_or_branch_unsupported")
    if requested == "scalar":
        addressing = "scalar"
    elif requested in {"auto", "vector"}:
        # The source vector prepass is an AIV-pair/AIC-relay publication once
        # per invocation; it is not a per-tile table read and is not recovered
        # here. Do not invent that work; retain the declared generic path.
        addressing = "generic_analytical"
    else:
        raise ModelingError("mqsmla_addressing_mode_unsupported")
    rows = tuple(query_rows(shape))
    cores = geometry.compute_units
    if not rows or not 1 <= cores <= MAX_MQSMLA_AIC_CORES:
        raise ModelingError("mqsmla_requires_positive_rows_and_at_most36_aic")
    # The retired ``cannbot-dsl`` wrapper's SELECTIVE_FD_72 split reduction is
    # not the official CANN program. Keep the guard only for its explicit alias.
    if legacy and selective_fd_excluded(cores, rows):
        raise ModelingError("mqsmla_selective_72_row_fd_branch_not_modeled")
    head_groups = tuple(min(DEFAULT_MQSMLA_HEADS, heads - start)
                        for start in range(HEAD_GROUP_START, heads, DEFAULT_MQSMLA_HEADS))
    row_heads = tuple(group for _ in rows for group in head_groups)
    rows = tuple(row for row in rows for _ in head_groups)
    ownership = owner_ranges(len(rows), cores)
    streams = tuple((owner, OwnerTiles.create(rows[start:end]), row_heads[start:end])
                    for owner, start, end in ownership)
    quant_mode = shape.get("kv_quant_mode")
    if quant_mode is not None:
        if quant_mode not in KV_QUANT_MODES:
            raise ModelingError("mqsmla_kv_quant_mode_unsupported")
        return _quant_plan(shape, streams, int(quant_mode), addressing)
    for pool, amount in BUFFERS.items():
        units = geometry.resource_units.get("vector", cores) if pool == "ub" else cores
        if (geometry.buffers.get(pool, 0) < amount
                or geometry.buffer_units.get(pool, 0) != units):
            raise ModelingError("mqsmla_source_buffer_capacity_or_ownership:" + pool)
    for role in ("vector", "mte2_aiv", "mte3"):
        if geometry.resource_units.get(role) != 2 * cores:
            raise ModelingError("mqsmla_requires_paired_aiv:" + role)
    for role in ("mte1", "mte2_aic", "fixpipe"):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError("mqsmla_requires_private_aic:" + role)
    total = Work()
    counts = Counter(_phase_key(stream, head_counts, index)
                     for _, stream, head_counts in streams for index in range(stream.count))
    for key, count in counts.items():
        for _, work in tile_phases(*key[:4], addressing=addressing, heads=key[-1]):
            total += work.scaled(count)

    def program(context):
        def one(owner, stream, head_counts):
            @lru_cache(maxsize=OWNER_STAGE_CACHE_SIZE)
            def cached_stages(key):
                result = []
                previous = None
                for name, work in tile_phases(*key[:4], addressing=addressing,
                                              heads=key[-1]):
                    if name in {"query", "sinks"}:
                        after = ()
                    elif name.startswith("addr_index_"):
                        after = ("query", "sinks")
                    elif name.startswith("addr_page_"):
                        after = ("addr_index_" + name.rsplit("_", 1)[-1],)
                    elif name.startswith("gather_"):
                        after = ("addr_page_" + name.rsplit("_", 1)[-1],)
                    elif name == "addr_join":
                        after = tuple(f"gather_{part}" for part in
                                      range(PHYSICAL_KV_TILE_ROWS // GATHER_SUBTILE_ROWS))
                    elif name == "qk_operands":
                        after = ("kv_l1", "query")
                    elif name == "softmax":
                        after = ("scores", "sinks")
                    else:
                        after = (previous,) if previous else ()
                    # Coarse phases preserve finite Channel reuse. Two Q
                    # halves are replaced only after the previous row's QK;
                    # QK/PV each hold at most one of the two shared L0C slots.
                    prior = {
                        "query": ("output",) if key[2] else (),
                        "sinks": ("output",) if key[2] else (),
                        "addr_join": ("output",) if key[2] else (),
                        "qk_operands": ("qk",), "qk": ("scores",),
                        "pv_operands": ("pv",), "pv": ("values",),
                        "accumulate": ("output",) if key[2] else ("accumulate",),
                    }.get(name, ())
                    result.append(Stage(
                        name, context.bind(work), after=after, previous=prior,
                        placement=context.owned(owner, cores, set(work.resources) - {"hbm"}
                                                | ({"cube"} if work.matrix else set()))))
                    previous = name
                return tuple(result)

            def stages(index):
                return cached_stages(_phase_key(stream, head_counts, index))

            allocations = tuple(context.allocate(
                f"mqsmla{owner}_{pool}", pool, amount,
                (2 * owner, 2 * owner + 1) if pool == "ub" else (owner,))
                for pool, amount in BUFFERS.items())
            return KernelProgram((
                Stage("reserve", acquire=allocations),
                # Two complete phase tiles are a conservative source subset:
                # P and alpha have two slots. The third source KV slot can
                # prefetch more aggressively, which this compact DAG omits.
                Stage("tiles", after=("reserve",), program=PipelineProgram(stream.count, 2, stages)),
                Stage("release", after=("tiles",), free=allocations),
            ))
        return ParallelProgram(tuple(one(owner, stream, head_counts)
                                     for owner, stream, head_counts in streams))

    return OperatorPlan(total, program, {
        "source_execution_domain": "mixed",
        "source_program": LEGACY_SOURCE if legacy else SOURCE,
        "native_kind": LEGACY_NATIVE_KIND if legacy else NATIVE_KIND,
        "mqsmla_owner_ranges": ownership, "mqsmla_buffers_per_owner": BUFFERS,
        "mqsmla_schedule": "whole_query_rows;metadata_SUPPORT_FD_false;two_live_phase_tiles",
        "mqsmla_approximation": "coarse_source_phase_DAG;register_operations_not_instruction_latency",
        "mqsmla_addressing": addressing,
        "address_schedule": {
            "subtiles_per_tile": ADDRESS_SUBTILES_PER_TILE,
            "rows_per_subtile": ADDRESS_ROWS_PER_SUBTILE,
            "scalar_reads_bytes": ADDRESS_SCALAR_READS * MQSMLA_HEAD_WIDTH,
            "primitives": PrimitiveInventory({
                "scalar_gm_access": ADDRESS_SCALAR_READS * PHYSICAL_KV_TILE_ROWS,
                "scalar_alu_issue": (ADDRESS_SCALAR_ALU_ISSUES
                                     * PHYSICAL_KV_TILE_ROWS),
            }).to_dict(("hbm", "mte2_aiv", "vector", "mte3")),
        },
        "unmodeled_source_features": ["aicpu_metadata_program",
                                      "scalar_page_lookup_response",
                                      "vector_address_prepass",
                                      "flash_decode_partial_reduction",
                                      "third_KV_slot_prefetch_overlap",
                                      "cross_query_sinks_prefetch_overlap",
                                      "source_compiler_instruction_scheduling"],
    })


def _external_prefix_groups(shape, sequences):
    """Physical shared-prefix groups, matching indexed_quant_attention."""
    raw = shape.get("prefix_tokens", 0) or 0
    extents = tuple(raw) if isinstance(raw, tuple) else (int(raw),) * len(sequences)
    keys = tuple(shape.get("prefix_cache_keys") or ())
    declared = bool(keys)
    groups: dict[str, list] = {}
    private = []
    for index, (query, context) in enumerate(sequences):
        prefix = max(0, min(int(extents[index]), int(context)))
        key = str(keys[index]) if index < len(keys) else ""
        if prefix > 0 and (key or not declared):
            groups.setdefault(key, []).append((int(query), int(context), prefix))
        else:
            private.append((int(query), int(context), prefix))
    return groups, private, declared


def quant_external_fill(shape, quant_mode):
    """Cold external KV bytes: resident window ring plus compressed rows."""
    rows = tuple(query_rows(shape))
    if not rows:
        return 0, 0
    row_bytes = quant_row_bytes(quant_mode)
    sequences = tuple(shape.get("sequences", ())) or ((len(rows), 0),)
    window = int(shape.get("window", 0) or 0)
    # The window ring keeps the last ``window`` distinct rows of the appended
    # sequence; a short context holds ``context + q_len`` rows, never a tile.
    original = (sum(min(int(context) + int(query),
                        window + max(0, int(query) - 1))
                    for query, context in sequences if int(query) > 0)
                if window else 0)
    ratio = int(shape.get("ratio", 0) or 0)
    topk = int(shape.get("topk", 0) or 0)
    # One compressed row completes per ``ratio`` raw tokens; the TopK list
    # keeps at most ``topk`` of the completed rows. Without a compressor every
    # context token is a candidate row.
    completed = [(int(context) + int(query)) // ratio if ratio
                 else (int(context) + int(query))
                 for query, context in sequences]
    capacity = [min(topk, value) if topk else value for value in completed]
    compressed = 0
    shared_rows = 0
    raw_prefix = shape.get("prefix_tokens", 0) or 0
    prefix = max(raw_prefix) if isinstance(raw_prefix, tuple) else int(raw_prefix)
    shared = (str(shape.get("prefix_cache_mode", "private")) == "shared"
              and prefix > 0)
    if shared and (topk or ratio):
        groups, private, declared = _external_prefix_groups(shape, sequences)
        if topk:
            # The compressed prefix union is one cold fill; the private
            # suffixes of each request are separate rows.  A completed prefix
            # shorter than one compression group shares no compressed row.
            for members in groups.values():
                member_completed = [
                    (context + query) // ratio if ratio else context + query
                    for query, context, _prefix in members]
                member_prefix = [
                    prefix // max(1, ratio) for _q, _c, prefix in members]
                selection = [min(topk, value) for value in member_completed]
                shared_sel = [
                    min(sel, p, (sel * p + comp // 2) // comp)
                    if comp and p else 0
                    for sel, comp, p in zip(
                        selection, member_completed, member_prefix)]
                member_priv = [sel - shared for sel, shared
                               in zip(selection, shared_sel)]
                group_shared = (len(members) > 1
                                or (not declared and len(sequences) > 1))
                if group_shared and any(p > 0 for p in member_prefix):
                    group_shared_unique = min(
                        max(member_prefix, default=0), sum(shared_sel))
                    compressed += group_shared_unique + sum(member_priv)
                    shared_rows += group_shared_unique
                else:
                    compressed += sum(selection)
            for query, context, _prefix in private:
                compressed += min(
                    topk, (context + query) // ratio if ratio else context + query)
        else:
            # Dense HCA: a shared group reads its completed prefix rows once
            # and each member keeps the rows beyond that prefix.
            for members in groups.values():
                member_completed = [
                    (context + query) // ratio
                    for query, context, _prefix in members]
                group_shared = (len(members) > 1
                                or (not declared and len(sequences) > 1))
                if group_shared:
                    prefix_rows = [
                        min(value, prefix // max(1, ratio))
                        for value, (_q, _c, prefix) in zip(
                            member_completed, members)]
                    compressed += max(prefix_rows) + sum(
                        value - shared_value for value, shared_value
                        in zip(member_completed, prefix_rows))
                else:
                    compressed += sum(member_completed)
            for query, context, _prefix in private:
                compressed += (context + query) // ratio
    elif topk or ratio:
        compressed = sum(capacity)
    return (original + compressed) * row_bytes, shared_rows


def _quant_plan(shape, streams, quant_mode, addressing):
    """Arch35 ``aclnnMixedQuantSparseFlashMla``: overlapped 128-key tile DAG."""
    row_bytes = quant_row_bytes(quant_mode)
    counts = Counter(_phase_key(stream, head_counts, index)
                     for _, stream, head_counts in streams
                     for index in range(stream.count))
    total = Work()
    window_bytes = row_bytes * int(shape.get("window", 0) or 0)
    for key, count in counts.items():
        for _name, work, _after in quant_tile_phases_cached(
                key[1], key[2], key[3], key[-1], quant_mode,
                window_bytes if key[0] == "original" else 0):
            total += work.scaled(count)
    external, shared_rows = quant_external_fill(shape, quant_mode)
    fill = Work(resources={"hbm": external} if external else {})
    # Each head-group stream reads only its own query-head slice. Summing the
    # per-stream head counts charges the single query tensor once, not once
    # per head group.
    streamed = sum(sum(head_counts) for _owner, _stream, head_counts in streams) \
        * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
    stream_work = Work(resources={"hbm": 2 * streamed, "mte2_aic": streamed,
                                  "mte3": streamed})

    def program(context):
        from .indexed_attention import bind_sustained_dram
        context = bind_sustained_dram(context)

        def one(owner, stream, head_counts):
            @lru_cache(maxsize=OWNER_STAGE_CACHE_SIZE)
            def stages(key):
                result = []
                for name, work, after in quant_tile_phases_cached(
                        key[1], key[2], key[3], key[-1], quant_mode,
                        window_bytes if key[0] == "original" else 0):
                    roles = set(work.resources) - {"hbm"}
                    if work.matrix:
                        roles |= {"cube"}
                    result.append(Stage(
                        name, context.bind(work), after=tuple(after),
                        previous=("accumulate",) if name == "accumulate" and not key[2] else (),
                        placement=context.owned(owner, context.core_count, roles)))
                return tuple(result)

            allocations = tuple(context.allocate(
                f"mqsmla{owner}_{pool}", pool, amount,
                (2 * owner, 2 * owner + 1) if pool == "ub" else (owner,))
                for pool, amount in BUFFERS.items())
            query_bytes = sum(head_counts) * MQSMLA_HEAD_WIDTH * BF16_ELEMENT_BYTES
            pipeline = KernelProgram((Stage(
                "tiles", program=PipelineProgram(
                    stream.count, QUANT_TILE_SLOTS,
                    lambda index: stages(_phase_key(stream, head_counts, index)))),))
            # The query read and output write are one double-buffered DMA stream
            # per owner: they contend on the shared bus and never lengthen the
            # tile chain, exactly as the source prefetches the next row's Q.
            stream_io = KernelProgram((Stage("qo_stream", context.bind(Work(
                resources={"hbm": 2 * query_bytes, "mte2_aic": query_bytes,
                           "mte3": query_bytes}))),))
            return KernelProgram((
                Stage("reserve", acquire=allocations),
                Stage("work", after=("reserve",),
                      program=ParallelProgram((pipeline, stream_io))),
                Stage("release", after=("work",), free=allocations)))

        owners = ParallelProgram(tuple(one(owner, stream, head_counts)
                                       for owner, stream, head_counts in streams))
        return KernelProgram((
            Stage("metadata", latency=METADATA_PROGRAM_MS),
            Stage("prefix_fill", after=("metadata",), work=context.bind(fill)),
            Stage("owners", after=("prefix_fill",), program=owners)))

    return OperatorPlan(total + fill + stream_work, program, {
        "source_execution_domain": "mixed",
        "source_program": SOURCE,
        "native_kind": NATIVE_KIND,
        "mqsmla_owner_ranges": tuple((owner, stream.rows) for owner, stream, _h in streams),
        "mqsmla_buffers_per_owner": BUFFERS,
        "mqsmla_schedule": "arch35_overlapped_tile_dag;two_live_kv_and_accumulator_slots",
        "mqsmla_approximation": "coarse_source_phase_DAG;register_operations_not_instruction_latency",
        "mqsmla_addressing": addressing,
        "mqsmla_kv_quant_mode": quant_mode,
        "mqsmla_packed_row_bytes": row_bytes,
        "mqsmla_dequant_ops_per_element": KV_QUANT_DEQUANT_OPS[quant_mode],
        "mqsmla_metadata_program_ms": METADATA_PROGRAM_MS,
        "mqsmla_prefix_partition": {
            "external_kv_bytes": external,
            "shared_unique_rows": shared_rows,
            "prefix_cache_mode": shape.get("prefix_cache_mode", "private"),
        },
        "unmodeled_source_features": ["aicpu_metadata_program",
                                      "scalar_page_lookup_response",
                                      "flash_decode_partial_reduction",
                                      "source_compiler_instruction_scheduling"],
    })
