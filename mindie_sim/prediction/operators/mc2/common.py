"""Shared MC2 row contracts, work accounting, and finite pipeline lowering.

[Role] Validate caller and routing records and build the phases shared by MC2
  strategies. Keep one typed work ledger for both projection and execution.
[Invariants] Frame whole rows and omit masked and diagonal routes from peer
  data. CCU Dispatch exchanges counts with every peer. A buffer is reusable
  only after its consumers finish.
[Timeouts & Locks] Finite planning over declared routes and cohorts; no I/O
  or locks.
[Side-Effect Policy] Immutable plans and analytical engine declarations only.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from graphlib import TopologicalSorter
import math

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..contracts import OperatorPlanningContext
from .lowering import (
    CCU, DISPATCH, FP32_SCALE_BYTES, TRIPLE_META_BYTES, Mc2Program, PeerRoute, Phase,
    Transport, cohort_stages, peer_stages, read_work, store_work,
)

FRAME_PAYLOAD_BYTES = 480
FRAME_WIRE_BYTES = 512
CCU_DISPATCH_COUNT_BYTES = 512
CCU_DISPATCH_ROW_ALIGN_BYTES = 512
CCU_DISPATCH_H_ALIGN = 32
UB_ALIGN_BYTES = 32
MX_GROUP_SIZE = 32
MX_ALIGN_BYTES = 256
GROUP_QUANT_SIZE = 128
GEMM_DIMENSIONS = 3
FLOAT_ROWS = frozenset({"bf16", "fp16"})
FP8_ROWS = frozenset({"fp8_e4m3", "fp8_e5m2"})
FP4_ROWS = frozenset({"fp4_e2m1", "fp4"})
AGMM = "all_gather_matmul"
MMRS = "matmul_reduce_scatter"
COMBINE = "combine_v2"
COLLECTIVES = frozenset({AGMM, MMRS})
NATIVE_ABIS = {
    AGMM: "aclnnAllGatherMatmulV2",
    MMRS: "aclnnMatmulReduceScatterV2",
    DISPATCH: "aclnnMoeDistributeDispatchV2",
    COMBINE: "aclnnMoeDistributeCombineV2",
}
ABI_ALIASES = {
    **{symbol: abi for abi, symbol in NATIVE_ABIS.items()},
    **{symbol.removeprefix("aclnn"): abi for abi, symbol in NATIVE_ABIS.items()},
    "moe_dispatch_v2": DISPATCH,
    "moe_combine_v2": COMBINE,
    "all_gather_matmul_v2_arch35": AGMM,
    "matmul_reduce_scatter_v2_arch35": MMRS,
    **{f"mc2_{abi}": abi for abi in NATIVE_ABIS},
}
TRANSPORTS = frozenset({"auto", "a2_hccl_client", "a3_sio_hccs", "a5_shmem", "a5_mte", "a5_ccu"})
ALGORITHMS = frozenset({"fullmesh", "ring", "two_tier", "direct_peer"})


def _integer(value, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"mc2_invalid_{field_name}:{value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Mc2RowLayout:
    width: int
    input_dtype: str
    payload_dtype: str
    quant_mode: str
    input_bytes: int
    payload_bytes: int
    input_scale_bytes: int
    scale_bytes: int
    expert_scale_bytes: int
    scale_offset_bytes: int
    triple_offset_bytes: int
    blocks: int
    materialized_bytes: int

    @property
    def packet_bytes(self) -> int:
        return self.blocks * FRAME_WIRE_BYTES

    @property
    def data_region_bytes(self) -> int:
        return self.blocks * FRAME_PAYLOAD_BYTES

    @property
    def flag_bytes(self) -> int:
        return self.packet_bytes - self.data_region_bytes

    @property
    def protocol_padding_bytes(self) -> int:
        return self.data_region_bytes - self.materialized_bytes


def dispatch_row_layout(
    width: int, *, input_dtype: str, payload_dtype: str, quant_mode: str,
    input_scale_bytes: int = 0, has_expert_scales: bool = False,
) -> Mc2RowLayout:
    """Place scale data and the dispatch triple before framing a token."""
    _integer(width, "width", 1)
    _integer(input_scale_bytes, "input_scale_bytes")
    if input_dtype not in FLOAT_ROWS | FP8_ROWS or payload_dtype not in FLOAT_ROWS | FP8_ROWS | FP4_ROWS | {"int8"}:
        raise ModelingError("mc2_dispatch_dtype_unsupported")
    if type(has_expert_scales) is not bool:
        raise ModelingError("mc2_invalid_has_expert_scales")
    incoming = math.ceil(width * dtype_bytes(input_dtype))
    payload = math.ceil(width * dtype_bytes(payload_dtype))
    offset = align_up(payload, UB_ALIGN_BYTES)
    if quant_mode == "none" and input_dtype == payload_dtype:
        scales = input_scale_bytes
    elif input_dtype not in FLOAT_ROWS or input_scale_bytes:
        raise ModelingError("mc2_quantized_input_requires_unscaled_float_row")
    elif quant_mode == "per_token" and payload_dtype in FP8_ROWS | {"int8"}:
        scales = FP32_SCALE_BYTES
    elif quant_mode == "mx" and payload_dtype in FP8_ROWS | FP4_ROWS:
        scales = align_up(align_up(width, MX_GROUP_SIZE) // MX_GROUP_SIZE, 2)
        offset = align_up(payload, MX_ALIGN_BYTES)
    elif quant_mode == "per_group" and payload_dtype in FP8_ROWS:
        scales = align_up(width, GROUP_QUANT_SIZE) // GROUP_QUANT_SIZE * FP32_SCALE_BYTES
        offset = align_up(payload, GROUP_QUANT_SIZE)
    else:
        raise ModelingError("mc2_dispatch_quantization_unsupported")
    triple = align_up(offset + scales, UB_ALIGN_BYTES)
    blocks = align_up(triple + UB_ALIGN_BYTES, FRAME_PAYLOAD_BYTES) // FRAME_PAYLOAD_BYTES
    expert_scale = FP32_SCALE_BYTES if has_expert_scales else 0
    return Mc2RowLayout(
        width, input_dtype, payload_dtype, quant_mode, incoming, payload, input_scale_bytes,
        scales, expert_scale, offset, triple, blocks, payload + scales + TRIPLE_META_BYTES + expert_scale,
    )


def combine_row_layout(width: int, *, dtype: str = "bf16") -> Mc2RowLayout:
    """Frame returned expert data. Combine reads assist metadata separately."""
    _integer(width, "width", 1)
    if dtype not in FLOAT_ROWS:
        raise ModelingError("mc2_combine_requires_float_row")
    payload = math.ceil(width * dtype_bytes(dtype))
    blocks = align_up(payload, FRAME_PAYLOAD_BYTES) // FRAME_PAYLOAD_BYTES
    return Mc2RowLayout(width, dtype, dtype, "none", payload, payload, 0, 0, 0, 0, 0, blocks, payload)


@dataclass(frozen=True, slots=True)
class Mc2CohortSlice:
    """Caller-owned rows identified by exact IDs when supplied, or by a span."""

    cohort_id: int
    row_start: int
    row_end: int
    source_assignments: int
    received_assignments: int
    remote_assignments: int
    local_assignments: int
    masked_padding_rows: int = 0
    row_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in self.__slots__:
            if name != "row_ids":
                _integer(getattr(self, name), name)
        if not isinstance(self.row_ids, (tuple, list)):
            raise ModelingError("mc2_invalid_cohort_row_ids")
        ids = tuple(self.row_ids)
        for row_id in ids:
            _integer(row_id, "row_id")
        if len(set(ids)) != len(ids):
            raise ModelingError("mc2_duplicate_cohort_row_ids")
        object.__setattr__(self, "row_ids", ids)
        if self.row_end < self.row_start or self.remote_assignments + self.local_assignments != self.source_assignments:
            raise ModelingError("mc2_cohort_assignment_balance")
        if self.local_assignments > self.received_assignments:
            raise ModelingError("mc2_cohort_local_rows_exceed_received")

    @property
    def row_count(self) -> int:
        return len(self.row_ids) if self.row_ids else self.row_end - self.row_start


@dataclass(frozen=True, slots=True)
class Mc2Contract:
    abi: str
    transport_abi: str
    topology_algorithm: str
    ranks: tuple[int, ...]
    rank: int
    internal_chunks: int
    independent_buffers: bool
    independent_handles: bool
    dtype: str
    m: int = 0
    k: int = 0
    n: int = 0
    gathered_bytes: int | float = 0
    operand_dtype: str = ""
    output_dtype: str = ""
    row_layout: Mc2RowLayout | None = None
    top_k: int = 1
    experts: int = 0
    status_bytes: int = 0
    cohorts: tuple[Mc2CohortSlice, ...] = ()
    # Each record holds an exact cohort ID, source rank, destination rank and row count.
    assignments: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def peer_payload_bytes(self) -> int | float:
        if self.abi in COLLECTIVES:
            if self.world_size == 1:
                return 0
            return (self.world_size - 1) * self.shard_bytes(self.m)
        if self.row_layout is None:
            return 0
        framed, _, _, _ = _peer_framing(self, self.transport_abi)
        return sum(cohort.remote_assignments for cohort in self.cohorts) * framed

    def shard_bytes(self, rows: int) -> int | float:
        if not self.m:
            return 0
        if self.world_size == 1:
            return self.gathered_bytes * rows / self.m
        if self.m % self.world_size or self.gathered_bytes % 1:
            raise ModelingError("mc2_uneven_shard_not_integer")
        shard, remainder = divmod(int(self.gathered_bytes) * rows, self.m * self.world_size)
        if remainder:
            raise ModelingError("mc2_uneven_shard_not_integer")
        return shard


def _peer_framing(contract: Mc2Contract, transport_abi: str) -> tuple[int, int, int, int]:
    """Return row, count, per-row padding and per-row flag bytes for this ABI."""
    layout = contract.row_layout
    if transport_abi != CCU:
        return layout.packet_bytes, contract.status_bytes, layout.protocol_padding_bytes, layout.flag_bytes
    if contract.abi == COMBINE:
        return layout.payload_bytes, 0, 0, 0
    # CCU dispatch aligns the merged payload and scales to its direct transfer stride.
    alignment = {"mx": MX_ALIGN_BYTES, "per_group": GROUP_QUANT_SIZE}.get(
        layout.quant_mode, CCU_DISPATCH_H_ALIGN
    )
    merged = math.ceil(align_up(layout.width, alignment) * dtype_bytes(layout.payload_dtype)) + layout.scale_bytes
    stride = align_up(merged, CCU_DISPATCH_ROW_ALIGN_BYTES)
    return stride, CCU_DISPATCH_COUNT_BYTES, stride - layout.payload_bytes - layout.scale_bytes, 0


def contract_fields(op: OpSpec) -> dict:
    """Read the caller declaration, including the existing TP geometry record."""
    fields = dict(op.metadata)
    for name in ("tp_mc2_geometry", "mc2_v2_caller", "mc2", "mc2_contract"):
        if name in fields:
            fields.update(dict(fields[name]))
    return fields


def capability_gap(op: OpSpec, ctx: OperatorPlanningContext, code: str) -> UnmodeledCapabilityError:
    arch = str(ctx.compute_arch.arch_id)
    return UnmodeledCapabilityError(
        capability="prediction", entry=op.op_id, code=f"mc2_{code}",
        message=f"MC2 requires a supported caller, transport and datapath on {arch}",
        minimal_replay_input={"op_id": op.op_id, "family": op.family, "recipe": op.recipe, "arch_id": arch},
        hints=({"schema": "mindie-unmodeled-structure-hint/1", "family": "mc2", "reason": code},),
    )


def _abi(op: OpSpec, ctx: OperatorPlanningContext, fields: Mapping) -> str:
    kind = fields.get("kind")
    declarations = (
        ctx.requested_abi, fields.get("requested_abi"), fields.get("abi", fields.get("mc2_abi")),
        op.recipe, fields.get("native_kind"), fields.get("direction"),
        {"dispatch": DISPATCH, "combine": COMBINE}.get(kind, kind), op.family,
    )
    explicit = tuple(value for value in declarations[:GEMM_DIMENSIONS] if value)
    choices = {ABI_ALIASES.get(value, value) for value in declarations if value}
    if any(ABI_ALIASES.get(value, value) not in NATIVE_ABIS for value in explicit):
        raise capability_gap(op, ctx, "unsupported_operator_abi")
    known = choices.intersection(NATIVE_ABIS)
    if len(known) != 1:
        raise capability_gap(op, ctx, "ambiguous_or_missing_operator_abi")
    return known.pop()


def _group(fields: Mapping, ctx: OperatorPlanningContext) -> tuple[Mapping, tuple[int, ...], int]:
    group = dict(fields.get("group", ()))
    world = fields.get("world_size", group.get("world_size", fields.get("tp", fields.get("ep"))))
    ranks = tuple(group.get("ranks", fields.get("ranks", ())))
    rank = fields.get("rank", ctx.occurrence.rank)
    world = max(1, _integer(world, "world_size")) if world is not None else None
    ranks = ranks or (tuple(range(world)) if world and world > 1 else (0 if rank is None else rank,))
    for member in ranks:
        _integer(member, "group_rank")
    if len(set(ranks)) != len(ranks) or (world is not None and len(ranks) != world):
        raise ModelingError("mc2_group_size_mismatch")
    rank = ranks[0] if rank is None else _integer(rank, "rank")
    if rank not in ranks:
        raise ModelingError("mc2_rank_not_in_group")
    return group, ranks, rank


def _matrix_counts(raw, ranks: tuple[int, ...]) -> Counter:
    counts = Counter()
    if isinstance(raw, Mapping):
        cells = ((src, dst, value) for (src, dst), value in raw.items())
    else:
        matrix = tuple(raw)
        if len(matrix) != len(ranks) or any(len(row) != len(ranks) for row in matrix):
            raise ModelingError("mc2_assignment_matrix_group_mismatch")
        cells = ((src, dst, value) for src, row in zip(ranks, matrix) for dst, value in zip(ranks, row))
    for src, dst, value in cells:
        if type(src) is not int or type(dst) is not int or src not in ranks or dst not in ranks:
            raise ModelingError("mc2_assignment_rank_not_in_group")
        counts[src, dst] = _integer(value, "assignment_count")
    return +counts


def _route_counts(
    fields: Mapping, contract: Mc2Contract, source_extents: Mapping[int, int],
) -> tuple[Counter, Counter, Counter, set[int]]:
    """Count validated (source rank, token, top-k slot) entries, including masks."""
    counts, masked, active_tokens = Counter(), Counter(), Counter()
    preceding, local_ids = set(), set()
    ranks, top_k, experts = contract.ranks, contract.top_k, contract.experts
    owner_map, combine = fields.get("expert_ranks"), contract.abi == COMBINE
    for record in fields.get("routes", ()):
        route = dict(record)
        src, dst = route.get("source_rank"), route.get("destination_rank")
        token, slot, expert = (route.get(key) for key in ("source_token_id", "top_k_slot", "destination_expert"))
        cid = _integer(route.get("cohort_id", 0), "cohort_id")
        active = route.get("active", route.get("top_k_mask", True))
        if type(active) is not bool or type(src) is not int or src not in ranks:
            raise ModelingError("mc2_invalid_route_source_or_mask")
        _integer(slot, "top_k_slot")
        if slot >= top_k:
            raise ModelingError("mc2_route_top_k_slot_out_of_range")
        _integer(token, "source_token_id", 0 if active else -1)
        tokens = source_extents.get(src)
        if active and tokens is not None and token >= tokens:
            raise ModelingError("mc2_source_token_id_out_of_range")
        key = (src, token, slot)
        if token >= 0 and key in preceding:
            raise ModelingError("mc2_duplicate_top_k_slot")
        if token >= 0:
            preceding.add(key)
            if src == contract.rank and (active or combine):
                local_ids.add(token)
        if not active:
            masked[cid, src] += 1
            continue
        _integer(expert, "destination_expert")
        if type(dst) is not int or dst not in ranks or (experts and expert >= experts):
            raise ModelingError("mc2_invalid_route_destination")
        if owner_map is not None:
            owner = owner_map.get(expert) if isinstance(owner_map, Mapping) else (
                owner_map[expert] if expert < len(owner_map) else None
            )
            if owner != dst:
                raise ModelingError("mc2_route_expert_owner_mismatch")
        weight = route.get("routing_weight")
        if weight is not None and (type(weight) not in {int, float} or not math.isfinite(weight)):
            raise ModelingError("mc2_invalid_routing_weight")
        active_tokens[cid, src, token] += 1
        counts[(cid, dst, src) if combine else (cid, src, dst)] += 1
    return counts, masked, active_tokens, local_ids


def _routing(fields: Mapping, contract: Mc2Contract, local_shape: tuple[int, ...]) -> Mc2Contract:
    ranks, rank = contract.ranks, contract.rank
    slices = tuple(
        item if isinstance(item, Mc2CohortSlice) else Mc2CohortSlice(**dict(item))
        for item in fields.get("cohort_slices", fields.get("cohorts", ()))
    )
    aggregate = fields.get("assignment_matrix", fields.get("assignments"))
    per_cohort = fields.get("cohort_assignment_matrices")
    # A caller can declare token extents for other source ranks in the EP group.
    source_extents = dict(fields.get("source_tokens_by_rank", ()))
    for src, extent in source_extents.items():
        if type(src) is not int or src not in ranks:
            raise ModelingError("mc2_source_tokens_rank_not_in_group")
        _integer(extent, "source_tokens")
    if "source_tokens" in fields:
        source_extents[rank] = _integer(fields["source_tokens"], "source_tokens")
    elif not slices and len(local_shape) == 2:
        source_extents.setdefault(rank, _integer(local_shape[0], "source_tokens"))
    end = source_extents.get(rank)
    counts, masked, active, local_ids = _route_counts(fields, contract, source_extents)
    if per_cohort is not None:
        declared = Counter({
            (_integer(cid, "cohort_id"), src, dst): value
            for cid, matrix in dict(per_cohort).items()
            for (src, dst), value in _matrix_counts(matrix, ranks).items()
        })
        if "routes" in fields and declared != counts:
            raise ModelingError("mc2_cohort_matrix_route_mismatch")
        counts = declared
    if aggregate is not None:
        declared = _matrix_counts(aggregate, ranks)
        if per_cohort is not None or "routes" in fields:
            total = Counter()
            for (_, src, dst), value in counts.items():
                total[src, dst] += value
            if declared != total:
                raise ModelingError("mc2_assignment_matrix_route_mismatch")
        elif len(slices) <= 1:
            cid = slices[0].cohort_id if slices else 0
            counts.update({(cid, src, dst): value for (src, dst), value in declared.items()})
        else:
            raise ModelingError("mc2_multiple_cohorts_require_directed_assignments")

    expected = defaultdict(lambda: [0, 0, 0])
    for (cid, src, dst), value in counts.items():
        if rank not in {src, dst}:
            continue
        expected[cid][0] += value if src == rank else 0
        expected[cid][1] += value if dst == rank else 0
        expected[cid][2] += value if src == dst == rank else 0
    explicit_routes = "routes" in fields or aggregate is not None or per_cohort is not None
    if not slices:
        ids = sorted({*expected, *(cid for cid, src in masked if src == rank)}) or [0]
        if len(ids) > 1:
            raise ModelingError("mc2_cohort_row_ranges_required")
        cid = ids[0]
        send, receive, local = expected[cid]
        row_ids = ()
        if end is None:
            if contract.abi == DISPATCH:
                end = max(send, max(local_ids, default=-1) + 1)
            elif "routes" in fields:
                end = max(local_ids, default=-1) + 1
                if len(local_ids) != end:
                    row_ids = tuple(sorted(local_ids))
            else:
                end = receive
                if not explicit_routes:
                    end = _integer(fields.get("received_assignments", 0), "received_assignments")
                if end and contract.top_k > 1:
                    raise ModelingError("mc2_combine_source_token_extent_required")
        padding = fields.get("masked_padding_rows", masked[cid, rank])
        if not explicit_routes:
            send = _integer(fields.get("source_assignments", end * contract.top_k), "source_assignments")
            receive = _integer(fields.get("received_assignments", send), "received_assignments")
            default_local = send if len(ranks) == 1 else 0
            if "remote_assignments" in fields:
                default_local = send - _integer(fields["remote_assignments"], "remote_assignments")
            local = _integer(fields.get("local_assignments", default_local), "local_assignments")
        slices = (Mc2CohortSlice(cid, 0, end, send, receive, send - local, local, padding, row_ids),)

    ids = {part.cohort_id for part in slices}
    if len(ids) != len(slices) or set(expected) - ids:
        raise ModelingError("mc2_invalid_cohort_ids")
    exact = _cohort_row_owners(slices)
    for part in slices:
        cid, send, receive, local = part.cohort_id, part.source_assignments, part.received_assignments, part.local_assignments
        rows = send if contract.abi == DISPATCH else receive
        if rows > part.row_count * contract.top_k:
            raise ModelingError("mc2_assignment_count_exceeds_top_k")
        if explicit_routes:
            if expected[cid] != [send, receive, local] or masked[cid, rank] > part.masked_padding_rows:
                raise ModelingError("mc2_cohort_counts_do_not_match_routes")
        elif len(ranks) <= 2:
            if local:
                counts[cid, rank, rank] = local
            if len(ranks) == 2:
                peer = next(member for member in ranks if member != rank)
                if send > local:
                    counts[cid, rank, peer] = send - local
                if receive > local:
                    counts[cid, peer, rank] = receive - local
            elif send != local or receive != local:
                raise ModelingError("mc2_singleton_remote_assignments")
        elif send != local or receive != local:
            raise ModelingError("mc2_directed_assignments_required")
        elif local:
            counts[cid, rank, rank] = local
    parts = {part.cohort_id: part for part in slices}
    for cid, src, token in active:
        if src != rank:
            continue
        part = parts.get(cid)
        if part is None or (exact.get(token) != cid if part.row_ids else not part.row_start <= token < part.row_end):
            raise ModelingError("mc2_source_token_outside_cohort")
    count_fields = ("source_assignments", "received_assignments", "local_assignments",
                    "remote_assignments", "masked_padding_rows")
    for name in fields.keys() & set(count_fields):
        if _integer(fields[name], name) != sum(getattr(part, name) for part in slices):
            raise ModelingError(f"mc2_declared_{name}_mismatch")
    return replace(contract, cohorts=slices, assignments=tuple((*key, count) for key, count in sorted(counts.items())))


def _cohort_row_owners(slices: tuple[Mc2CohortSlice, ...]) -> dict[int, int]:
    spans = sorted((part.row_start, part.row_end) for part in slices if not part.row_ids and part.row_count)
    if any(end > start for (_, end), (start, _) in zip(spans, spans[1:])):
        raise ModelingError("mc2_overlapping_cohort_rows")
    starts = [start for start, _ in spans]
    exact = {}
    for part in slices:
        for row in part.row_ids:
            index = bisect_right(starts, row) - 1
            if row in exact or (index >= 0 and row < spans[index][1]):
                raise ModelingError("mc2_overlapping_cohort_rows")
            exact[row] = part.cohort_id
    return exact


def validate_contract(op: OpSpec, ctx: OperatorPlanningContext) -> Mc2Contract:
    fields = contract_fields(op)
    abi = _abi(op, ctx, fields)
    group, ranks, rank = _group(fields, ctx)
    transport = fields.get("transport_abi", "auto")
    algorithm = fields.get("topology_algorithm", group.get("topology_algorithm", ""))
    if transport not in TRANSPORTS or (algorithm and algorithm not in ALGORITHMS):
        raise capability_gap(op, ctx, "unsupported_transport_or_algorithm")
    chunks = _integer(fields.get("internal_chunks", 1), "internal_chunks", 1)
    buffers = fields.get("independent_buffers", False)
    handles = fields.get("independent_handles", False)
    if chunks not in {1, 2} or type(buffers) is not bool or type(handles) is not bool:
        raise ModelingError("mc2_invalid_chunk_binding")
    dtype = fields.get("compute_dtype", op.dtype or "bf16")
    out_dtype = fields.get("output_dtype", op.dtype or "bf16")
    contract = Mc2Contract(abi, transport, algorithm, ranks, rank, chunks, buffers, handles, dtype, output_dtype=out_dtype)
    dtype_bytes(dtype)
    dtype_bytes(out_dtype)
    if abi in COLLECTIVES:
        shape = tuple(op.local_shape or op.semantic_shape)
        dims = shape if len(shape) == GEMM_DIMENSIONS else (shape[0] if shape else None, None, shape[-1] if shape else None)
        m = _integer(fields.get("m", fields.get("global_tokens", dims[0])), "m")
        k = _integer(fields.get("k", fields.get("k_dim", dims[1])), "k")
        n = _integer(fields.get("n", fields.get("n_dim", dims[-1])), "n")
        operand = fields.get("operand_dtype", fields.get("gather_dtype", dtype if abi == AGMM else out_dtype))
        row_bytes = (k if abi == AGMM else n) * dtype_bytes(operand)
        gathered = fields.get("gathered_bytes", fields.get("operand_bytes", m * row_bytes))
        if type(gathered) not in {int, float} or not math.isfinite(gathered) or gathered < 0 or (not m and gathered):
            raise ModelingError("mc2_invalid_gathered_operand_bytes")
        contract = replace(contract, m=m, k=k, n=n, gathered_bytes=gathered, operand_dtype=operand)
        contract.shard_bytes(m)
        return contract
    if op.recipe == "arch22_v2_caller_local" and "mc2_v2_local" in fields:
        return contract
    if chunks != 1:
        raise ModelingError("mc2_moe_uses_cohorts_instead_of_internal_chunks")
    width = fields.get("width", op.local_shape[-1] if op.local_shape else None)
    fields.setdefault("source_tokens_by_rank", group.get("source_tokens_by_rank", ()))
    if abi == DISPATCH:
        incoming = fields.get("input_dtype", op.dtype or "bf16")
        quant = fields.get("quant_mode", "per_token" if fields.get("dispatch_quant") == "per_token_int8" else "none")
        payload = fields.get("payload_dtype", "int8" if quant == "per_token" else incoming)
        layout = dispatch_row_layout(width, input_dtype=incoming, payload_dtype=payload, quant_mode=quant,
                                     input_scale_bytes=fields.get("input_scale_bytes", 0),
                                     has_expert_scales=fields.get("has_expert_scales", False))
    else:
        layout = combine_row_layout(fields.get("combine_width", width), dtype=fields.get("combine_dtype", fields.get("payload_dtype", op.dtype or "bf16")))
    experts = _integer(fields.get("experts", 0), "experts")
    top_k = _integer(fields.get("top_k", 1), "top_k", 1)
    status = _integer(fields.get("status_bytes", UB_ALIGN_BYTES if abi == DISPATCH else 0), "status_bytes")
    return _routing(fields, replace(contract, row_layout=layout, top_k=top_k, experts=experts, status_bytes=status),
                    op.local_shape)


def topology_hops(fields: Mapping, group: Mapping) -> dict:
    raw = fields.get("topology_hops", group.get("topology_hops", {}))
    hops = {}
    for (src, dst), path in dict(raw).items():
        if type(src) is not int or type(dst) is not int or src == dst:
            raise ModelingError("mc2_invalid_topology_pair")
        values = (path,) if isinstance(path, str) else tuple(path)
        if not values or any(type(value) is not str or not value for value in values):
            raise ModelingError("mc2_invalid_topology_hops")
        hops[src, dst] = values
    return hops


def _phase_items(work: Work) -> tuple[WorkItem, ...]:
    return tuple(WorkItem("matrix", "flop", value, dtype) for dtype, value in work.matrix.items() if value) + tuple(
        WorkItem(role, "op" if role == "vector" else "bytes", value)
        for role, value in work.resources.items() if value
    )


def make_plan(contract: Mc2Contract, transport: Transport, phases: tuple[Phase, ...],
              sources: tuple[SourceRef, ...], *, arch_id: str) -> OperatorPlan:
    """Build both executable and inspectable programs from one phase ledger."""
    by_name = {phase.name: phase for phase in phases}
    order = TopologicalSorter({phase.name: phase.after for phase in phases}).static_order()
    phases = tuple(by_name[name] for name in order)
    work = sum((phase.work for phase in phases), Work())
    implementation = f"mc2_{contract.abi}:{transport.abi}:{transport.algorithm}"
    spec = FusedPipelineSpec("mc2", implementation, tuple(
        PipelinePhase(phase.name, phase.after, _phase_items(phase.work),
                      cohort=dict(phase.metadata).get("cohort_id"), slot=dict(phase.metadata).get("slot"))
        for phase in phases
    ), sources)
    return OperatorPlan(work, Mc2Program(phases), {
        "implementation": implementation, "governed_family": "mc2", "arch_id": arch_id,
        "abi": contract.abi, "transport_abi": transport.abi, "topology_algorithm": transport.algorithm,
        "peer_payload_bytes": replace(contract, transport_abi=transport.abi).peer_payload_bytes,
        "internal_chunks": contract.internal_chunks,
        "window_alignment_bytes": transport.window_alignment, "peer_credits": transport.credits,
    }, pipeline_spec=spec)


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(ctx.occurrence, spec.family, spec.implementation,
                                   (op.op_id,), (), spec.phases, spec.sources, spec.repeats)


def build_cohort_transport_stages(
    contract: Mc2Contract, cohort_slice: Mc2CohortSlice, transport: Transport, *,
    after: tuple[str, ...] = (), peer_after: tuple[str, ...] = (),
) -> tuple[Phase, ...]:
    """Lower one caller-owned cohort using its selected transport framing."""
    return cohort_stages(contract, cohort_slice, transport, _peer_framing(contract, transport.abi),
                         after=after, peer_after=peer_after)


def _matrix_fragment(contract: Mc2Contract, name: str, after: tuple[str, ...], rows: int,
                     *, b_bytes: int | float, slot: tuple = (), ub_output: bool = False) -> Phase:
    a_bytes = (contract.gathered_bytes * rows / contract.m if contract.abi == AGMM and contract.m
               else rows * contract.k * dtype_bytes(contract.dtype))
    load = read_work(a_bytes + b_bytes, aic=True, l2=True)
    partial = rows * contract.n * dtype_bytes(contract.output_dtype)
    write = Work(resources={"fixpipe": partial}) if ub_output else store_work(partial, fixpipe=True, l2=True)
    return Phase(name, after, load + write + Work(
        matrix={contract.dtype: 2 * rows * contract.k * contract.n},
        resources={"mte1": a_bytes + b_bytes}), "aic", slot)


def _chunked_stages(contract: Mc2Contract, transport: Transport, *,
                    ub_output: bool = False, temp_gm: bool = False) -> tuple[Phase, ...]:
    gathering = contract.abi == AGMM
    chunks = contract.internal_chunks
    ring = RingBuffer(allocation="mc2_chunks", depth=transport.slot_depth)
    fragments: dict[str, Phase] = {}
    aliases: dict[str, str] = {}
    phases: list[Phase] = []
    world = contract.world_size
    routes = transport.routes
    local_rows = contract.m // world
    head_rows = ((local_rows + chunks - 1) // chunks) * world
    for index in range(chunks):
        rows = contract.m if chunks == 1 else (head_rows if index == 0 else contract.m - head_rows)
        suffix = f"chunk{index}"
        tag = (("chunk", index), ("slot", ring.slot(index)), ("cross_core", transport.ccu))
        comm = ("ag_" if gathering else "rs_") + suffix
        gemm = f"gemm_{suffix}"
        producer, consumer = (comm, gemm) if gathering else (gemm, comm)
        events = lower_generation(ring, index, producer, (consumer,))
        # Give the ring's write and consumer nodes the public pipeline names.
        release = f"slot_release_{index}"
        aliases.update({events.write: producer, events.read_done[0]: consumer, events.last_use: release})
        fragments[events.last_use] = Phase(release, (consumer,), metadata=tag)
        comm_gate = events.credit if gathering else events.publish
        previous = (f"{'ag' if gathering else 'rs'}_chunk{index - 1}",) if index else ()
        comm_deps = (comm_gate, *previous)
        payload = contract.shard_bytes(rows)
        wire_window = len(routes) * transport.peer_bytes(payload)
        if temp_gm and wire_window:
            read = comm + "_hccl_read"
            data_bytes = len(routes) * payload
            phases.append(Phase(read, comm_deps, Work(resources={"hbm": data_bytes, "l2": data_bytes}),
                                metadata=tag))
            comm_deps = (read,)
        names = []
        for route in routes:
            name = f"{comm}_peer_{route.destination}"
            credit = (names[-transport.credits],) if len(names) >= transport.credits else ()
            phases.extend(peer_stages(name, (*comm_deps, *credit), route, transport, payload, tag))
            names.append(name)
        b_bytes = contract.k * contract.n * dtype_bytes(contract.dtype) if rows else 0
        if gathering:
            local = f"local_gemm_{suffix}"
            phases.append(_matrix_fragment(contract, local, (), rows // world, b_bytes=b_bytes, slot=tag))
            remote = rows - rows // world
            fragments[events.read_done[0]] = _matrix_fragment(
                contract, gemm, (events.publish,), remote, b_bytes=b_bytes if remote else 0, slot=tag)
            fragments[events.write] = Phase(comm, (events.credit, *names),
                                            Work(resources={"hbm": wire_window, "l2": wire_window}), metadata=tag)
        else:
            previous_gemm = (f"gemm_chunk{index - 1}",) if index else ()
            fragments[events.write] = _matrix_fragment(contract, gemm, (events.credit, *previous_gemm), rows,
                                                        b_bytes=b_bytes, slot=tag, ub_output=ub_output)
            own_bytes = rows // world * contract.n * dtype_bytes(contract.output_dtype)
            reduced = (world - 1) * (rows // world) * contract.n
            work = read_work((0 if temp_gm else wire_window) + (0 if ub_output else own_bytes), l2=True)
            work += store_work(own_bytes, l2=True) + Work(resources={"vector": reduced})
            fragments[events.read_done[0]] = Phase(
                comm, (events.publish, *names), work if world > 1 else Work(), "aiv", tag)
    for node in ring.to_stages():
        phases.append(fragments.get(node.name, Phase(aliases.get(node.name, node.name), node.after)))
    return tuple(replace(phase, after=tuple(aliases.get(dep, dep) for dep in phase.after)) for phase in phases)


def build_chunked_agmm_stages(contract: Mc2Contract, transport: Transport) -> tuple[Phase, ...]:
    return _chunked_stages(contract, transport)


def build_chunked_mmrs_stages(contract: Mc2Contract, transport: Transport, *,
                             ub_output: bool = False, temp_gm: bool = False) -> tuple[Phase, ...]:
    return _chunked_stages(contract, transport, ub_output=ub_output, temp_gm=temp_gm)


def build_transport_stages(contract: Mc2Contract, transport: Transport, *,
                           ub_output: bool = False, temp_gm: bool = False) -> tuple[Phase, ...]:
    if contract.abi == AGMM:
        return build_chunked_agmm_stages(contract, transport)
    if contract.abi == MMRS:
        return build_chunked_mmrs_stages(contract, transport, ub_output=ub_output, temp_gm=temp_gm)
    phases = []
    for index, cohort in enumerate(contract.cohorts):
        reuse = index - transport.slot_depth
        after = (f"cohort{contract.cohorts[reuse].cohort_id}_slot_release",) if reuse >= 0 else ()
        credit = (f"cohort{contract.cohorts[index - 1].cohort_id}_slot_release",) if index and not after else ()
        phases.extend(build_cohort_transport_stages(contract, cohort, transport, after=after, peer_after=credit))
    return tuple(phases)
