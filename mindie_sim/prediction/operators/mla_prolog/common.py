"""Shared tensor contract and finite lowering for MLA prolog.

[Role] Derive logical and issued matrix work, workspace transfers, and the
  occurrence-bound pipeline description from one per-owner phase graph.
[Invariants] Work is counted once from the same phase records used at runtime.
  Logical tensor dimensions are preserved through fractal and DMA padding.
[Timeouts & Locks] Two chunk shapes per owner group suffice, with no I/O or locks.
[Side-Effect Policy] Constructs value records and programs for the common engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from operator import index

from mindie_sim.observed_contracts.vocabulary import build_unmodeled_onboarding_hint
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...results import FusedPipelineDescriptor
from ...runtime.memory_physics import AddressRun, AddressRunPlan
from ..contracts import OperatorPlanningContext
from ._lowering import _Chunk, _Demand, _Phase, _PrologProgramBuilder, _summarize, _transfer

FAMILY = "mla_prolog"
MLA_PROLOG_V4_RECIPE = "mla_prolog_v4_arch35"
ABI_V3 = "aclnnMlaPrologV3"
ABI_V3_NZ = "aclnnMlaPrologV3WeightNz"
ABI_V4_NZ = "aclnnMlaPrologV4WeightNz"
ABI_DSL = "cannbot_dsl.attn_prologue"
ACLNN_V3_ABIS = frozenset({ABI_V3, ABI_V3_NZ})
SUPPORTED_ABIS = ACLNN_V3_ABIS | {ABI_V4_NZ, ABI_DSL}
WEIGHT_QUANT_BF16 = 0
WEIGHT_QUANT_INT8 = 2
WEIGHT_QUANT_MXFP8 = 3
SUPPORTED_WEIGHT_MODES = frozenset({WEIGHT_QUANT_BF16, WEIGHT_QUANT_INT8, WEIGHT_QUANT_MXFP8})
PRECISIONS = {WEIGHT_QUANT_BF16: "bf16", WEIGHT_QUANT_INT8: "int8", WEIGHT_QUANT_MXFP8: "fp8_e4m3"}
DEFAULT_PROLOG_HIDDEN = 7168
DEFAULT_PROLOG_HEADS = 16
DEFAULT_Q_LORA_RANK = 1536
DEFAULT_KV_LORA_RANK = 512
DEFAULT_QK_NOPE_DIM = 128
DEFAULT_QK_ROPE_DIM = 64
PAGED_CACHE_MODES = frozenset({"PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ"})
BLOCK_PAGED_CACHE_MODES = frozenset({"PA_BLK_BSND", "PA_BLK_NZ"})
CACHE_MODES = PAGED_CACHE_MODES | {"BNSD", "BSND", "TND"}
SPLIT_M_MIN_TOKENS = 8192
SPLIT_M_HIDDEN_SIZES = frozenset({7168, 7680})
SPLIT_M_HEADS = 128
FRACTAL_MN = 16
FRACTAL_K_BF16 = 16
FRACTAL_K_BYTE = 32
BF16_BYTES = 2
FP32_BYTES = 4
SLOT_INDEX_BYTES = 8
MX_GROUP_SIZE = 32
C8_BLOCK_SIZE = 128
RMSNORM_OPS_PER_ELEM = 5
MX_QUANT_OPS_PER_ELEM = 4
C8_QUANT_OPS_PER_ELEM = 4
ROPE_OPS_PER_ELEM = 6
MATRIX_N_MIN = 64
KV_FIXED_N_MIN_AIC = 9
GROUP_COMPUTE_MIN_HEADS = 8
CQ_SCALE_SLOT_BYTES = 32
DEFAULT_STEP_BATCH = 128
WAVE_STEP_BATCH = 64
HIGH_THROUGHPUT_NOPE_WIDTH = 128


@dataclass(frozen=True, slots=True)
class _ResidentTiling:
    token_tile: int
    token_tiles: int
    k_splits: int
    column_tile: int
    panel: int
    a_groups: int
    b_groups: int
    gm_alignment: int


@dataclass(frozen=True, slots=True)
class PrologContract:
    tokens: int
    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    abi: str
    weight_quant_mode: int
    kv_cache_quant_mode: int
    query_quant_mode: int = 0
    cache_mode: str = "PA_BSND"
    do_rope: bool = True
    batch_seq_fused: bool = False
    actual_seq_len: tuple[int, ...] | None = None
    tile_size: int = C8_BLOCK_SIZE
    emit_query_norm_out: bool = True
    dsl_d_width: int = 0

    @property
    def is_dsl(self) -> bool:
        return self.abi == ABI_DSL

    @property
    def precision(self) -> str:
        return PRECISIONS[self.weight_quant_mode]

    @property
    def logical_flops(self) -> int:
        return sum(_logical_matrix(self).values())

    @property
    def kv_row_bytes(self) -> int:
        if self.is_dsl:
            return self.dsl_d_width + BF16_BYTES * ceil_div(self.dsl_d_width, MX_GROUP_SIZE)
        latent, rope = self.kv_lora_rank, self.qk_rope_head_dim
        if self.kv_cache_quant_mode == 0:
            return (latent + rope) * BF16_BYTES
        scale_bytes = FP32_BYTES if rope else BF16_BYTES
        return latent + rope * BF16_BYTES + ceil_div(latent, self.tile_size) * scale_bytes


def ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _align(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


def w_uk_ownership(num_heads: int, aic_cores: int) -> tuple[int, int]:
    if num_heads < 0 or aic_cores <= 0:
        raise ModelingError("mla_prolog_invalid_w_uk_geometry")
    if not num_heads:
        return 0, 0
    heads_per_owner = ceil_div(num_heads, aic_cores)
    return heads_per_owner, ceil_div(num_heads, heads_per_owner)


def fractal_issues(m: int, n: int, k: int, k0: int) -> tuple[int, int]:
    """Return the issue count and corresponding issued FLOPs."""
    if min(m, n, k) < 0 or k0 not in {FRACTAL_K_BF16, FRACTAL_K_BYTE}:
        raise ModelingError("mla_prolog_invalid_fractal_geometry")
    issues = ceil_div(m, FRACTAL_MN) * ceil_div(n, FRACTAL_MN) * ceil_div(k, k0)
    return issues, 2 * issues * FRACTAL_MN * FRACTAL_MN * k0


def uses_split_m(contract: PrologContract) -> bool:
    return (
        contract.weight_quant_mode == WEIGHT_QUANT_MXFP8
        and contract.tokens >= SPLIT_M_MIN_TOKENS
        and contract.hidden_size in SPLIT_M_HIDDEN_SIZES
        and contract.num_heads == SPLIT_M_HEADS
    )


def step_batch(contract: PrologContract) -> int:
    wave = (
        contract.weight_quant_mode == WEIGHT_QUANT_BF16
        or (contract.weight_quant_mode == WEIGHT_QUANT_INT8
            and contract.num_heads >= GROUP_COMPUTE_MIN_HEADS)
    )
    limit = (
        WAVE_STEP_BATCH if wave and contract.qk_nope_head_dim != HIGH_THROUGHPUT_NOPE_WIDTH
        else DEFAULT_STEP_BATCH
    )
    return min(contract.tokens, limit)


def _integer(value, name: str, minimum: int = 0) -> int:
    try:
        number = index(value)
    except TypeError:
        raise ModelingError(f"mla_prolog_invalid_{name}:{value!r}") from None
    if isinstance(value, bool) or number < minimum:
        raise ModelingError(f"mla_prolog_invalid_{name}:{value!r}")
    return number


def validate_contract(op: OpSpec, requested_abi: str | None = None) -> PrologContract:
    metadata = dict(op.metadata)
    shape = op.local_shape or op.semantic_shape
    raw = {**metadata, **dict(metadata.get("prolog_v4_geometry") or {}),
           **dict(metadata.get("prolog_contract") or {})}
    weight = _integer(raw.get("weight_quant_mode", WEIGHT_QUANT_MXFP8), "weight_quant_mode")
    cache = _integer(raw.get("kv_cache_quant_mode", 1), "kv_cache_quant_mode")
    query = _integer(raw.get("query_quant_mode", 0), "query_quant_mode")
    if weight not in SUPPORTED_WEIGHT_MODES or cache not in {0, 1} or query not in {0, 1}:
        raise ModelingError(f"mla_prolog_unsupported_quant_modes:{weight}:{cache}:{query}")
    declared = raw.get("abi") or raw.get("native_kind")
    requested = requested_abi or raw.get("requested_abi")
    if requested and declared and requested != declared:
        raise ModelingError(f"mla_prolog_abi_conflict:{requested}:{declared}")
    abi = requested or declared or (ABI_V4_NZ if weight == WEIGHT_QUANT_MXFP8 else ABI_V3_NZ)
    if abi not in SUPPORTED_ABIS:
        raise ModelingError(f"mla_prolog_abi_unsupported:{abi}")
    tile_size = _integer(raw.get("tile_size", C8_BLOCK_SIZE), "tile_size", 1)
    if abi != ABI_DSL and tile_size != C8_BLOCK_SIZE:
        raise ModelingError(f"mla_prolog_invalid_tile_size:{tile_size}")

    dims = {
        "tokens": shape[0] if shape else 1,
        "hidden_size": DEFAULT_PROLOG_HIDDEN,
        "num_heads": shape[1] if len(shape) > 1 else DEFAULT_PROLOG_HEADS,
        "q_lora_rank": DEFAULT_Q_LORA_RANK,
        "kv_lora_rank": DEFAULT_KV_LORA_RANK,
        "qk_nope_head_dim": DEFAULT_QK_NOPE_DIM,
        "qk_rope_head_dim": DEFAULT_QK_ROPE_DIM,
    }
    for name, default in dims.items():
        dims[name] = _integer(raw.get(name, default), name,
                              0 if name in {"tokens", "qk_rope_head_dim"} else 1)
    rope = dims["qk_rope_head_dim"]
    do_rope = raw.get("do_rope")
    do_rope = bool(rope) if do_rope is None else bool(do_rope)
    if do_rope and (not rope or rope % 2):
        raise ModelingError(f"mla_prolog_invalid_rotary_width:{rope}")
    dsl_width = raw.get("dsl_d_width")
    if dsl_width is None:
        dsl_width = dims["kv_lora_rank"] + rope if abi == ABI_DSL else 0
    dsl_width = _integer(dsl_width, "dsl_d_width", int(abi == ABI_DSL))
    if abi == ABI_DSL and rope > dsl_width:
        raise ModelingError("mla_prolog_dsl_rope_exceeds_head_width")

    mode = raw.get("cache_mode", "PA_BSND")
    if mode not in CACHE_MODES:
        raise ModelingError(f"mla_prolog_unsupported_cache_mode:{mode}")
    batch_seq_fused = bool(raw.get("batch_seq_fused", False))
    actual = raw.get("actual_seq_len")
    if actual is not None:
        try:
            actual = tuple(_integer(v, "actual_seq_len") for v in actual)
        except TypeError:
            raise ModelingError("mla_prolog_invalid_actual_seq_len") from None
    if batch_seq_fused and mode in BLOCK_PAGED_CACHE_MODES and (actual is None or (dims["tokens"] and not actual)):
        raise ModelingError("mla_prolog_actual_seq_len_required")
    return PrologContract(
        **dims, abi=abi, weight_quant_mode=weight, kv_cache_quant_mode=cache,
        query_quant_mode=query, cache_mode=mode, do_rope=do_rope,
        batch_seq_fused=batch_seq_fused, actual_seq_len=actual,
        tile_size=tile_size,
        emit_query_norm_out=bool(raw.get("emit_query_norm_out", True)), dsl_d_width=dsl_width,
    )


def _products(c: PrologContract) -> tuple[tuple[str, int, int, int, str], ...]:
    """Name, output width, reduction width, independent groups, precision."""
    rope, heads = c.qk_rope_head_dim, c.num_heads
    kv_width = c.dsl_d_width if c.is_dsl else c.kv_lora_rank + rope
    q_width = c.dsl_d_width if c.is_dsl else c.qk_nope_head_dim + rope
    mm1, mm2, mm3 = (("mm1_wqa", "mm2_wkv", "mm3_wqb") if c.is_dsl
                     else ("mm1_cq", "mm2_ckv_kr", "mm3_qc_qr"))
    products = (
        (mm1, c.q_lora_rank, c.hidden_size, 1, c.precision),
        (mm2, kv_width, c.hidden_size, 1, c.precision),
        (mm3, heads * q_width, c.q_lora_rank, 1, c.precision),
    )
    if c.is_dsl:
        return products
    return (*products, ("mm4_w_uk", c.kv_lora_rank, c.qk_nope_head_dim, heads, "bf16"))


def _logical_matrix(c: PrologContract) -> dict[str, int]:
    matrix: dict[str, int] = {}
    for _, n, k, groups, dtype in _products(c):
        matrix[dtype] = matrix.get(dtype, 0) + 2 * c.tokens * n * k * groups
    return {dtype: flops for dtype, flops in matrix.items() if flops}


def make_op(
    *, op_id: str, tokens: int, hidden_size: int, num_heads: int,
    deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode",
    q_lora_rank: int = DEFAULT_Q_LORA_RANK, kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_nope_head_dim: int = DEFAULT_QK_NOPE_DIM, qk_rope_head_dim: int = DEFAULT_QK_ROPE_DIM,
    abi: str | None = None, weight_quant_mode: int = WEIGHT_QUANT_MXFP8,
    kv_cache_quant_mode: int = 1, query_quant_mode: int = 0, cache_mode: str = "PA_BSND",
    do_rope: bool | None = None, batch_seq_fused: bool = False,
    actual_seq_len: tuple[int, ...] | None = None, tile_size: int = C8_BLOCK_SIZE,
    emit_query_norm_out: bool = True, dsl_d_width: int | None = None,
) -> OpSpec:
    raw = {
        "tokens": tokens, "hidden_size": hidden_size, "num_heads": num_heads, "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank, "qk_nope_head_dim": qk_nope_head_dim, "qk_rope_head_dim": qk_rope_head_dim,
        "abi": abi, "weight_quant_mode": weight_quant_mode, "kv_cache_quant_mode": kv_cache_quant_mode,
        "query_quant_mode": query_quant_mode, "cache_mode": cache_mode, "do_rope": do_rope,
        "batch_seq_fused": batch_seq_fused, "actual_seq_len": actual_seq_len, "tile_size": tile_size,
        "emit_query_norm_out": emit_query_norm_out, "dsl_d_width": dsl_d_width,
    }
    op = OpSpec(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
        family="native_fusion", recipe=MLA_PROLOG_V4_RECIPE, source_boundary="kernel",
        semantic_shape=(), local_shape=(), padded_shape=(), launch_shape=(), dtype="bf16",
        metadata=(("prolog_contract", raw),),
    )
    c = validate_contract(op)
    width = c.dsl_d_width if c.is_dsl else c.kv_lora_rank + c.qk_rope_head_dim
    shape = (c.tokens, c.num_heads, width)
    matrix = _logical_matrix(c)
    return replace(
        op, semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=c.logical_flops, matrix_flops=tuple(matrix.items()),
        metadata=(("native_kind", c.abi), ("prolog_v4_geometry", asdict(c))),
    )


def unsupported(op: OpSpec, ctx: OperatorPlanningContext, reason: str) -> UnmodeledCapabilityError:
    arch = str(ctx.compute_arch.arch_id)
    return UnmodeledCapabilityError(
        capability=FAMILY, entry="mla_prolog.build", code=f"mla_prolog_{reason}:{arch}",
        message=f"{op.op_id} has no MLA prolog strategy for {reason} on {arch}",
        hints=(build_unmodeled_onboarding_hint(
            observed_op_type=FAMILY, observed_abi_summary={"arch_id": arch, "abi": ctx.requested_abi},
        ),),
    )


def _plane(rows: int, width: int, *, stride: int | None = None, offset: int = 0) -> AddressRun:
    return AddressRun(offset, rows, width, width if stride is None else stride)


def _parts(size: int, count: int) -> tuple[int, ...]:
    base, large = divmod(size, count)
    return tuple(base + (owner < large) for owner in range(count))


def _dual_rows(rows: int) -> tuple[int, int]:
    first = min(rows, _align(rows, FRACTAL_MN) // 2)
    return first, rows - first


def _activation(width: int, dtype: str) -> int:
    if dtype == "bf16":
        return width * BF16_BYTES
    if dtype == "int8":
        return width + FP32_BYTES
    return width + ceil_div(width, MX_GROUP_SIZE)


def _weight(n: int, k: int, dtype: str, k0: int) -> tuple[AddressRun, ...]:
    size = BF16_BYTES if dtype == "bf16" else 1
    payload = _plane(_align(n, FRACTAL_MN), _align(k, k0) * size)
    if dtype == "bf16":
        return (payload,)
    scales = FP32_BYTES if dtype == "int8" else ceil_div(k, MX_GROUP_SIZE)
    return payload, _plane(n, scales)


@dataclass
class _ChunkGraph:
    contract: PrologContract
    rows: int
    aics: tuple[int, ...]
    aivs: tuple[int, ...]
    l2: bool
    fixpipe_alignment: int | None
    phases: list[_Phase]
    cq_scale_slot_rows: int = 0

    def add(self, name: str, after: tuple[str, ...], demands) -> str:
        self.phases.append(_Phase(name, after, tuple(demands)))
        return name

    def matrix(self, product, after=(), *, acc_bytes: int):
        name, n, k, groups, dtype = product
        k0 = FRACTAL_K_BF16 if dtype == "bf16" else FRACTAL_K_BYTE
        c, rows = self.contract, self.rows
        if groups > 1 or name == "mm4_w_uk":
            per, _ = w_uk_ownership(groups, len(self.aics))
            slices = [(n, start, min(per, groups - start)) for start in range(0, groups, per)]
        else:
            granule = k0
            if name == "mm3_qc_qr" and (dtype == "int8" and c.num_heads >= GROUP_COMPUTE_MIN_HEADS):
                granule = c.qk_nope_head_dim + c.qk_rope_head_dim
            width = _align(ceil_div(n, len(self.aics)), granule)
            if name == "mm1_cq":
                width = max(width, MATRIX_N_MIN)
            if name == "mm2_ckv_kr" and len(self.aics) >= KV_FIXED_N_MIN_AIC:
                width = max(width, MATRIX_N_MIN)
            if len(self.aics) == 1:
                width = n
            slices = [(min(width, n - start), start, 1) for start in range(0, n, width)]

        reads, operands, cubes, publish = [], [], [], []
        for owner, (cols, start, batches) in zip(self.aics, slices, strict=False):
            activation = (_plane(rows, _activation(k, dtype)),)
            weight = _weight(cols, k, dtype, k0)
            if name == "mm4_w_uk":
                head_width = k if c.weight_quant_mode != WEIGHT_QUANT_BF16 else k + c.qk_rope_head_dim
                stride = head_width * BF16_BYTES
                activation = tuple(_plane(rows, k * BF16_BYTES, stride=stride * c.num_heads,
                                          offset=head * stride)
                                   for head in range(start, start + batches))
            logical_weight = cols * k * (BF16_BYTES if dtype == "bf16" else 1)
            logical_weight += sum(run.rows * run.row_bytes for run in weight[1:])
            logical_input = sum(run.rows * run.row_bytes for run in activation)
            access = _transfer(owner, "aic", "mte2_aic", name, *activation, *(weight * batches),
                               l2=self.l2, logical_bytes=logical_input + batches * logical_weight)
            pipe_bytes = access.work.resources["mte2_aic"]
            reads.append(access)
            operands.append(_Demand(owner, "aic", Work(resources={"mte1": pipe_bytes})))
            issues, issued = fractal_issues(rows, cols, k, k0)
            cubes.append(_Demand(owner, "aic", Work(matrix={dtype: issued * batches}), (
                ("logical_flops", 2 * rows * cols * k * batches), ("fractal_issues", issues * batches),
                ("issued_flops", issued * batches), ("heads", batches),
            )))
            if c.is_dsl:
                # The physical accumulator split has two equal row halves, including tail padding.
                half = _align(rows, FRACTAL_MN) // 2
                for peer, logical_rows in zip(self.aivs, _dual_rows(rows), strict=True):
                    sink = _transfer(owner, "aic", "fixpipe", name,
                                     _plane(half, _align(cols, FRACTAL_MN) * acc_bytes),
                                     write=True, gm=False, alignment=self.fixpipe_alignment,
                                     logical_bytes=logical_rows * cols * acc_bytes)
                    publish.append(replace(sink, audit=(*sink.audit, ("aiv_destination", peer))))
            else:
                offset, stride = (start * n, n * groups) if name == "mm4_w_uk" else (start, n)
                output = _plane(rows, cols * batches * acc_bytes, stride=stride * acc_bytes,
                                offset=offset * acc_bytes)
                publish.append(_transfer(owner, "aic", "fixpipe", name, output, write=True,
                                         l2=self.l2, alignment=self.fixpipe_alignment))

        load = self.add(name + "_mte2_aic", after, reads)
        operand = self.add(name + "_mte1", (load,), operands)
        cube = self.add(name + "_cube", (operand,), cubes)
        dst = "ub" if c.is_dsl else "gm"
        return self.add(name + "_fixpipe_" + dst, (cube,), publish)

    def vector(
        self, name: str, after: tuple[str, ...], *, read_rows: int = 0, params: int = 0,
        ops: int, write_rows: int = 0, publish_scale: bool = False,
        owners_work: tuple[tuple[int, int], ...] = (),
    ) -> str:
        work = owners_work
        if not work:
            owners = self.aivs if self.contract.is_dsl else self.aivs[:self.rows]
            rows_by_owner = _dual_rows(self.rows) if self.contract.is_dsl else _parts(self.rows, len(owners))
            work = tuple((owner, rows) for owner, rows in zip(owners, rows_by_owner) if rows)
        current = after
        if read_rows or params:
            load = self.add(name + "_mte2_aiv", current, (
                _transfer(owner, "aiv", "mte2_aiv", name,
                          _plane(rows, read_rows), _plane(1, params), l2=self.l2)
                for owner, rows in work
            ))
            current = (load,)
        compute = self.add(name + "_vector", current, (
            _Demand(owner, "aiv", Work(resources={"vector": rows * ops})) for owner, rows in work
        ))
        if publish_scale:
            self.add("cq_scale_mte3_gm", (compute,), (
                _transfer(owner, "aiv", "mte3", "cq_scale", _plane(rows, CQ_SCALE_SLOT_BYTES),
                          write=True, l2=self.l2)
                for owner, rows in work
            ))
        if not write_rows:
            return compute
        return self.add(name + "_mte3_gm", (compute,), (
            _transfer(owner, "aiv", "mte3", name, _plane(rows, write_rows),
                      write=True, l2=self.l2)
            for owner, rows in work
        ))

    def resident_a(self, t: _ResidentTiling) -> tuple[str, str]:
        c = self.contract
        tile, rank, head = t.token_tile, c.q_lora_rank, c.dsl_d_width
        k = c.hidden_size // t.k_splits
        qa_cols = rank // t.column_tile
        total_cols = qa_cols + head // t.column_tile
        dtype = c.precision
        width = t.column_tile
        qa_publish, kv_publish, operands, cubes, loads = [], [], [], [], []
        twin = t.token_tiles == 2
        for job in range((1 if twin else t.token_tiles) * t.k_splits * t.a_groups):
            token_tile, split_group = divmod(job, t.k_splits * t.a_groups)
            split, group = divmod(split_group, t.a_groups)
            owner = self.aics[job % len(self.aics)]
            tiles = range(t.token_tiles) if twin else (token_tile,)
            live = sum(max(0, min(tile, c.tokens - idx * tile)) for idx in tiles)
            # Each job keeps its activation rows in L1 across QA and KV column tiles.
            a_bytes = _activation(k, dtype)
            scale_width = c.hidden_size // MX_GROUP_SIZE
            split_scales = k // MX_GROUP_SIZE
            first_row = tiles[0] * tile
            inputs = (
                _plane(tile * len(tiles), k, stride=c.hidden_size,
                       offset=first_row * c.hidden_size + split * k),
                _plane(tile * len(tiles), split_scales, stride=scale_width,
                       offset=first_row * scale_width + split * split_scales),
            )
            for column in range(group, total_cols, t.a_groups):
                is_qa = column < qa_cols
                name, output_width = ("qa_partials", rank) if is_qa else ("kv_partials", head)
                start = (column if is_qa else column - qa_cols) * width
                weight = _weight(width, k, dtype, FRACTAL_K_BYTE)
                loads.append(_transfer(owner, "aic", "mte2_aic", name, *inputs, *weight, l2=self.l2))
                inputs = ()
                operand_bytes = AddressRunPlan(weight).aligned_copy_bytes() + tile * a_bytes * len(tiles)
                operands.append(_Demand(owner, "aic", Work(resources={"mte1": operand_bytes})))
                issues, flops = fractal_issues(tile, width, k, FRACTAL_K_BYTE)
                cubes.append(_Demand(owner, "aic", Work(matrix={dtype: flops * len(tiles)}), (
                    ("split", split), ("logical_flops", 2 * live * width * k),
                    ("fractal_issues", issues * len(tiles)),
                )))
                stride = output_width * FP32_BYTES
                offset = ((split * t.token_tiles + tiles[0]) * tile * output_width + start) * FP32_BYTES
                writes = qa_publish if is_qa else kv_publish
                demand = _transfer(owner, "aic", "fixpipe", name,
                                   _plane(tile * len(tiles), width * FP32_BYTES, stride=stride, offset=offset),
                                   write=True, l2=self.l2, alignment=t.gm_alignment,
                                   logical_bytes=live * width * FP32_BYTES)
                writes.append(replace(demand, audit=(*demand.audit, ("split", split))))
        load = self.add("a1_mte2_aic", (), loads)
        operand = self.add("a1_mte1", (load,), operands)
        cube = self.add("a1_cube", (operand,), cubes)
        return (self.add("mm1_wqa_fixpipe_gm", (cube,), qa_publish),
                self.add("mm2_wkv_fixpipe_gm", (cube,), kv_publish))

    def resident_reduce(self, name, width, after, t: _ResidentTiling) -> str:
        row = width * FP32_BYTES
        stride = t.token_tiles * t.token_tile * row
        return self.add(name + "_mte2_aiv", after, (
            _transfer(owner, "aiv", "mte2_aiv", name, *(
                _plane(t.k_splits, row, stride=stride, offset=idx * row)
                for idx in range(lane, self.rows, len(self.aivs))
            ), l2=self.l2)
            for lane, owner in enumerate(self.aivs[:self.rows])
        ))

    def resident_b(self, t: _ResidentTiling, pairs, after: str) -> tuple[tuple[int, int], ...]:
        c = self.contract
        tile, panel, rank = t.token_tile, t.panel, c.q_lora_rank
        half = tile // 2
        panels = c.num_heads * c.dsl_d_width // panel
        twin = t.token_tiles == 2
        loads, operands, cubes, publish, outputs, tables = [], [], [], [], [], []
        ub_slots = {}
        for job in range((1 if twin else t.token_tiles) * t.b_groups):
            token_tile, group = divmod(job, t.b_groups)
            owner, peers = pairs[job % len(pairs)]
            tiles = range(t.token_tiles) if twin else (token_tile,)
            activations = (_plane(tile * len(tiles), rank), _plane(tile * len(tiles), rank // MX_GROUP_SIZE))
            loads.append(_transfer(owner, "aic", "mte2_aic", "qr", *activations, l2=self.l2))
            tables.extend(_transfer(peer, "aiv", "mte2_aiv", "rope_tables",
                                    _plane(half * len(tiles), 2 * c.qk_rope_head_dim * FP32_BYTES), l2=self.l2)
                          for peer in peers)
            for column in range(group, panels, t.b_groups):
                weight = _weight(panel, rank, c.precision, FRACTAL_K_BYTE)
                # The resident panel prefetch loads the whole head's scale rows.
                scales = _plane(c.dsl_d_width, rank // MX_GROUP_SIZE)
                loads.append(_transfer(owner, "aic", "mte2_aic", "wqb", weight[0], scales, l2=self.l2))
                operand_bytes = AddressRunPlan(weight).aligned_copy_bytes()
                operand_bytes += len(tiles) * tile * _activation(rank, c.precision)
                operands.append(_Demand(owner, "aic", Work(resources={"mte1": operand_bytes})))
                _, issued = fractal_issues(tile, panel, rank, FRACTAL_K_BYTE)
                cubes.append(_Demand(owner, "aic", Work(matrix={c.precision: issued * len(tiles)})))
                rope_ops = (c.qk_rope_head_dim * ROPE_OPS_PER_ELEM
                            if (column + 1) * panel % c.dsl_d_width == 0 else 0)
                for lane, peer in enumerate(peers):
                    live = sum(max(0, min(half, self.rows - idx * tile - lane * half)) for idx in tiles)
                    rows = half * len(tiles)
                    local = _transfer(owner, "aic", "fixpipe", "wqb",
                                      _plane(rows, panel * FP32_BYTES), write=True, gm=False,
                                      alignment=self.fixpipe_alignment, logical_bytes=live * panel * FP32_BYTES)
                    publish.append(replace(local, audit=(*local.audit, ("aiv_destination", peer))))
                    outputs.append((peer, rows, live, rope_ops))
                    ub_slots[peer] = 2 * half * panel * FP32_BYTES
        load = self.add("mm3_wqb_mte2_aic", (after,), loads)
        operand = self.add("mm3_wqb_mte1", (load,), operands)
        cube = self.add("mm3_wqb_cube", (operand,), cubes)
        local = self.add("mm3_wqb_fixpipe_ub", (cube,), publish)
        tables_ready = self.add("query_out_mte2_aiv", (after,), tables)
        compute = self.add("query_out_vector", (local, tables_ready), (
            _Demand(peer, "aiv", Work(resources={"vector": rows * (panel + rope)}))
            for peer, rows, _, rope in outputs
        ))
        self.add("query_out_mte3_gm", (compute,), (
            _transfer(peer, "aiv", "mte3", "query_out", _plane(rows, panel * BF16_BYTES),
                      write=True, l2=self.l2, logical_bytes=live * panel * BF16_BYTES)
            for peer, rows, live, _ in outputs
        ))
        return tuple(ub_slots.items())

    def lower_resident(self, t: _ResidentTiling, pairs) -> _Chunk:
        c = self.contract
        rank, head = c.q_lora_rank, c.dsl_d_width
        owners = tuple((owner, rows) for owner, rows in zip(self.aivs, _parts(self.rows, len(self.aivs))) if rows)
        qa, kv = self.resident_a(t)
        qa_read = self.resident_reduce("qa_partials", rank, (qa,), t)
        qr = self.vector("cq_norm", (qa_read,), params=rank * FP32_BYTES, owners_work=owners,
                         ops=rank * (t.k_splits - 1 + RMSNORM_OPS_PER_ELEM + MX_QUANT_OPS_PER_ELEM),
                         write_rows=_activation(rank, c.precision))
        if c.emit_query_norm_out:
            self.add("query_norm_out_mte3_gm", (qr,), (
                _transfer(owner, "aiv", "mte3", "query_norm_out", _plane(rows, rank * BF16_BYTES),
                          write=True, l2=self.l2) for owner, rows in owners
            ))
        kv_read = self.resident_reduce("kv_partials", head, (kv, qr), t)
        rope = c.qk_rope_head_dim
        self.vector("kv_cache", (kv_read,), params=head * FP32_BYTES, owners_work=owners,
                    read_rows=2 * rope * FP32_BYTES + (SLOT_INDEX_BYTES if c.cache_mode in PAGED_CACHE_MODES else 0),
                    ops=head * (t.k_splits - 1 + RMSNORM_OPS_PER_ELEM + C8_QUANT_OPS_PER_ELEM) + rope * ROPE_OPS_PER_ELEM,
                    write_rows=c.kv_row_bytes)
        ub = self.resident_b(t, pairs, qr)
        return _Chunk(1, tuple(self.phases), t.token_tiles * t.token_tile * t.k_splits * (rank + head) * FP32_BYTES, ub)

    def lower(self, count: int) -> _Chunk:
        c = self.contract
        rank, latent, heads = c.q_lora_rank, c.kv_lora_rank, c.num_heads
        nope, rope = c.qk_nope_head_dim, c.qk_rope_head_dim
        kv_width = c.dsl_d_width if c.is_dsl else latent + rope
        q_width = heads * (c.dsl_d_width if c.is_dsl else nope + rope)
        quantized = c.weight_quant_mode != WEIGHT_QUANT_BF16
        int8 = c.weight_quant_mode == WEIGHT_QUANT_INT8
        acc_bytes = FP32_BYTES if int8 or c.is_dsl else BF16_BYTES
        norm_param_bytes = FP32_BYTES if c.is_dsl else BF16_BYTES
        norm_row = _activation(rank, c.precision)
        rope_ops = rope * ROPE_OPS_PER_ELEM if c.do_rope else 0
        rope_tables = 2 * rope * FP32_BYTES if c.do_rope else 0
        mm1, mm2, mm3, *uk = _products(c)
        cq = self.matrix(mm1, acc_bytes=acc_bytes)
        ckv = self.matrix(mm2, acc_bytes=acc_bytes)
        self.vector(
            "kv_cache", (ckv,), read_rows=(0 if c.is_dsl else kv_width * acc_bytes)
            + rope_tables + (SLOT_INDEX_BYTES if c.cache_mode in PAGED_CACHE_MODES else 0),
            params=(kv_width if c.is_dsl else latent) * norm_param_bytes,
            ops=(kv_width if c.is_dsl else latent) * RMSNORM_OPS_PER_ELEM + rope_ops
            + (C8_QUANT_OPS_PER_ELEM * (kv_width if c.is_dsl else latent) if c.kv_cache_quant_mode else 0),
            write_rows=c.kv_row_bytes,
        )
        cq_norm = self.vector(
            "cq_norm", (cq,), read_rows=0 if c.is_dsl else rank * acc_bytes,
            params=rank * norm_param_bytes, ops=rank * (
                RMSNORM_OPS_PER_ELEM + (MX_QUANT_OPS_PER_ELEM if quantized else 0)
            ), write_rows=norm_row, publish_scale=bool(self.cq_scale_slot_rows),
        )
        cq_ready = (cq_norm,)
        if self.cq_scale_slot_rows:
            # All launched AIVs copy the host step slot, including on the tail.
            cq_ready = (self.add("cq_scale_mte2_aiv", (cq_norm, "cq_scale_mte3_gm"), (
                _transfer(owner, "aiv", "mte2_aiv", "cq_scale",
                          _plane(self.cq_scale_slot_rows, CQ_SCALE_SLOT_BYTES), l2=self.l2)
                for owner in self.aivs
            )),)
        if c.is_dsl and c.emit_query_norm_out:
            self.add("query_norm_out_mte3_gm", (cq_norm,), (
                _transfer(owner, "aiv", "mte3", "query_norm_out", _plane(rows, rank * BF16_BYTES),
                          write=True, l2=self.l2)
                for owner, rows in zip(self.aivs, _dual_rows(self.rows)) if rows
            ))
        query = self.matrix(mm3, cq_ready, acc_bytes=acc_bytes)
        if c.is_dsl:
            self.vector("query_out", (query,), read_rows=rope_tables,
                        ops=heads * (c.dsl_d_width + rope_ops), write_rows=q_width * BF16_BYTES)
            return _Chunk(count, tuple(self.phases), 0)

        qc = query
        if quantized:
            qc = self.vector("qc_dequant", (query,), read_rows=q_width * acc_bytes,
                             ops=heads * nope, write_rows=heads * nope * BF16_BYTES)
        if rope:
            self.vector("query_rope", (query,), read_rows=heads * rope * acc_bytes + rope_tables,
                        ops=heads * (rope_ops + (rope if int8 else 0)),
                        write_rows=heads * rope * BF16_BYTES)
        absorb = self.matrix(uk[0], (qc,), acc_bytes=BF16_BYTES)
        if c.query_quant_mode:
            self.vector("query_quant", (absorb,), read_rows=heads * latent * BF16_BYTES,
                        ops=heads * latent * C8_QUANT_OPS_PER_ELEM,
                        write_rows=heads * (latent + FP32_BYTES))
        cq_bytes = rank * acc_bytes if c.emit_query_norm_out else (
            rank * acc_bytes + norm_row if quantized else norm_row
        )
        workspace = cq_bytes + (kv_width + q_width) * acc_bytes
        if quantized:
            workspace += heads * nope * BF16_BYTES
        if c.query_quant_mode:
            workspace += heads * latent * BF16_BYTES
        return _Chunk(count, tuple(self.phases), self.rows * workspace + self.cq_scale_slot_rows * CQ_SCALE_SLOT_BYTES)


def build_plan(
    c: PrologContract, ctx: OperatorPlanningContext, *, sources: tuple[SourceRef, ...],
    mode: str, resident: _ResidentTiling | None = None, fixpipe_alignment: int | None = None,
    pairs: tuple[tuple[int, tuple[int, ...]], ...] = (),
    cq_scale: bool = False,
) -> OperatorPlan:
    aic, aiv = ctx.effective_aic_cores, ctx.effective_aiv_cores
    split_m = mode == "split_m"
    if min(aic, aiv) < 1 and c.tokens:
        raise ModelingError("mla_prolog_requires_aic_and_aiv_owners")
    chunk_size = resident.token_tile if resident else step_batch(c)
    if not c.is_dsl and c.tokens and ctx.datapath.usable_paths:
        fixpipe_alignment = ctx.datapath.route("l0c", "gm").path.alignment_bytes
    l2 = "l2" in ctx.geometry.rates or "l2" in ctx.geometry.resource_units
    groups = []
    if c.tokens and not resident:
        if c.is_dsl or split_m:
            groups = [(rows, (owner,), peers) for (owner, peers), rows in zip(
                pairs, _parts(c.tokens, len(pairs)), strict=True,
            ) if rows]
        else:
            groups = [(c.tokens, tuple(range(aic)), tuple(range(aiv)))]

    programs = []
    if resident:
        aics = tuple(owner for owner, _ in pairs)
        aivs = tuple(peer for _, peers in pairs for peer in peers)
        graph = _ChunkGraph(c, c.tokens, aics, aivs, l2, fixpipe_alignment, [])
        programs.append((graph.lower_resident(resident, pairs),))
    for rows, aics, aivs in groups:
        full, tail = divmod(rows, chunk_size)
        programs.append(tuple(
            _ChunkGraph(c, size, aics, aivs, l2, fixpipe_alignment, [],
                        cq_scale_slot_rows=chunk_size if cq_scale else 0).lower(count)
            for size, count in ((chunk_size, full), (tail, int(bool(tail)))) if count
        ))
    program = _PrologProgramBuilder(tuple(programs), ctx.datapath.aic_count, ctx.datapath.aiv_count,
                                    ctx.stream_owners if c.is_dsl else None)
    work, spec = _summarize(program, FAMILY, c.abi, sources)
    per, owners = (0, 0) if c.is_dsl or not c.tokens else (
        (c.num_heads, aic) if split_m else w_uk_ownership(c.num_heads, aic)
    )
    return OperatorPlan(work, program, {
        "execution_class": "mixed_aic_aiv", "governed_family": FAMILY,
        "arch_id": str(ctx.compute_arch.arch_id), "implementation": c.abi,
        "split_mode": mode, "step_batch": chunk_size, "l1_reuse": resident is not None,
        "k_splits": resident.k_splits if resident else 1,
        "heads_per_owner": per, "w_uk_owners": owners, "kv_row_bytes": c.kv_row_bytes,
        "weight_quant_mode": c.weight_quant_mode, "kv_cache_quant_mode": c.kv_cache_quant_mode,
        "do_rope": c.do_rope, "logical_flops": c.logical_flops, "source_program": sources[0].to_dict(),
        "issued_flops": sum(work.matrix.values()),
        "workspace_slot_bytes": sum(max(chunk.workspace_bytes for chunk in group) for group in programs),
    }, pipeline_spec=spec)


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )
