"""Physical row and phase contracts for explicit arch35 full-mesh MC2.

[Role] Describe source-defined framing, count channels and local materialization
  for a caller that binds existing copy/quant/transport programs.
[Invariants] Frame one whole token before multiplying assignment counts. Owner
  domains are eligible source cores, never proof that each core has payload.
  Count and payload channels are parallel; receiving materialization waits for
  both. No latency, observed residual or implicit kernel invocation is created.
[Timeouts & Locks] Pure O(AIV count) construction; no device or file access.
[Side-Effect Policy] Immutable physical contracts only; no simulation or I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from .common import (
    FRAME_PAYLOAD_BYTES, FRAME_WIRE_BYTES, GROUP_QUANT_SIZE, MX_ALIGN_BYTES, MX_GROUP_SIZE,
)
from .lowering import (
    FP32_SCALE_BYTES, FP8_QUANT_OPS, INT8_QUANT_OPS, INT32_BYTES, INT64_BYTES, TRIPLE_META_BYTES,
)
from .mc2_mx_quant import MX32_TILE_ELEMS, mx32_work, SOURCE as MX_SOURCE

TARGET_BACKEND = "ascend"
_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h",
        source_sha256="3194ed49e10aab74976053840a5e771051be946d0945664fd5a3ca34a32bdb6b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h",
        source_sha256="4e2a3f571df2e0e3718c8e61145638483be15445383ed77c3873381874ffb5a9",
    ),
)

SOURCE = {
    "repository": "cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "branch": "arch35_mte_fullmesh; shared_expert_ranks=0; no_elastic_mask",
    "dispatch_sha256": "76db25293e4d3b34784012ace8fd824a977e4055fa974a9e3370c99975467c1e",
    "combine_sha256": "133aef717fce163e06f942c82ffb162e5cc0fe8d8e9c56fcbaaccd13afd63429",
    "quant_header_sha256": "46fec97abaec206c9b41bd1b3ab690d7c0ff7fe7a09961599b8c56f304543ed5",
    "binary_match": "not_established_by_source_reference",
}


def _align(value, multiple):
    return math.ceil(value / multiple) * multiple


UB_ALIGN_BYTES = 32
EXPERT_STATUS_RECORD_BYTES = UB_ALIGN_BYTES
_FRAME_FLAG_BYTES = FRAME_WIRE_BYTES - FRAME_PAYLOAD_BYTES
_COUNT_SYNC_RECORD_BYTES = UB_ALIGN_BYTES
_COUNT_SYNC_RECORD_ELEMS = _COUNT_SYNC_RECORD_BYTES // INT32_BYTES
_EXPERTS_PER_COUNT_CORE_TARGET = 16
_MAX_COUNT_CORES = 8
_SERIAL_MODE_MAX_BS_PER_RANK = 32
_SERIAL_MODE_MAX_EP_WORLD_SIZE = 16
_COUNT_STATUS_INIT_OPS_PER_EXPERT = 9
_COUNT_PIPE_V_OPS_PER_EXPERT = 4
_COUNT_VECTOR_OPS_PER_ID = _COUNT_PIPE_V_OPS_PER_EXPERT + 1  # Includes software ReduceSum.
_ROUTING_ID_INIT_ALIGNMENT_BYTES = 256
_ROUTER_WEIGHT_BYTES = 4
EXPAND_IDX_VALUES_PER_ROW = TRIPLE_META_BYTES // INT32_BYTES
COMBINE_FLAG_ELEMS_PER_BLOCK = _FRAME_FLAG_BYTES // INT32_BYTES
COMBINE_FLAG_MASK_LANES = 64


@dataclass(frozen=True)
class RowLayout:
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
    packet_bytes: int
    materialized_bytes: int

    @property
    def data_region_bytes(self):
        return self.blocks * FRAME_PAYLOAD_BYTES

    @property
    def flag_bytes(self):
        return self.blocks * _FRAME_FLAG_BYTES


def dispatch_layout(width, *, input_dtype="bf16", payload_dtype="bf16",
                    quant_mode="none", input_scale_bytes=0, has_expert_scales=False):
    """Full token layout from QuantInit + FillTriple + 480/512 packaging."""
    if not isinstance(width, int) or width < 1:
        raise ModelingError("mc2_width_requires_positive_integer")
    if input_dtype not in {"bf16", "fp16", "fp8_e4m3", "fp8_e5m2"}:
        raise ModelingError("mc2_input_dtype_unsupported")
    if payload_dtype not in {"bf16", "fp16", "int8", "fp8_e4m3", "fp8_e5m2",
                             "fp4_e2m1", "fp4"}:
        raise ModelingError("mc2_payload_dtype_unsupported")
    if not isinstance(input_scale_bytes, int) or input_scale_bytes < 0:
        raise ModelingError("mc2_input_scales_require_nonnegative_byte_count")
    incoming = int(width * dtype_bytes(input_dtype))
    outgoing = int(width * dtype_bytes(payload_dtype))
    if quant_mode == "none":
        if input_dtype != payload_dtype:
            raise ModelingError("mc2_unquantized_requires_identical_representation")
        scale, offset = input_scale_bytes, _align(outgoing, UB_ALIGN_BYTES)
    else:
        if input_dtype not in {"bf16", "fp16"} or input_scale_bytes:
            raise ModelingError("mc2_quantized_input_requires_unscaled_float_row")
        if quant_mode == "per_token" and payload_dtype in {"int8", "fp8_e4m3", "fp8_e5m2"}:
            scale, offset = FP32_SCALE_BYTES, _align(outgoing, UB_ALIGN_BYTES)
        elif (quant_mode == "mx" and payload_dtype in {
                "fp8_e4m3", "fp8_e5m2", "fp4_e2m1", "fp4"}):
            scale, offset = _align(math.ceil(width / MX_GROUP_SIZE), 2), _align(outgoing, MX_ALIGN_BYTES)
        elif quant_mode == "per_group" and payload_dtype in {"fp8_e4m3", "fp8_e5m2"}:
            scale = math.ceil(width / GROUP_QUANT_SIZE) * FP32_SCALE_BYTES
            offset = _align(outgoing, GROUP_QUANT_SIZE)
        else:
            raise ModelingError("mc2_quantization_branch_unsupported")
    triple = _align(offset + scale, UB_ALIGN_BYTES)
    # Full-mesh replaces the basic 12B triple tail with one reserved 32B block.
    blocks = math.ceil((triple + _FRAME_FLAG_BYTES) / FRAME_PAYLOAD_BYTES)
    expert_scale = FP32_SCALE_BYTES if has_expert_scales else 0
    return RowLayout(width, input_dtype, payload_dtype, quant_mode, incoming,
                     outgoing, input_scale_bytes, scale, expert_scale, offset, triple,
                     blocks, blocks * FRAME_WIRE_BYTES, outgoing + scale + TRIPLE_META_BYTES + expert_scale)


def combine_layout(width, *, dtype="bf16"):
    """Unquantized BF16/FP16 expert return; metadata is a separate input."""
    if dtype not in {"bf16", "fp16"} or not isinstance(width, int) or width < 1:
        raise ModelingError("mc2_combine_requires_positive_float_row")
    payload = int(width * dtype_bytes(dtype))
    blocks = math.ceil(payload / FRAME_PAYLOAD_BYTES)
    return RowLayout(width, dtype, dtype, "none", payload, payload, 0, 0, 0,
                     0, 0, blocks, blocks * FRAME_WIRE_BYTES, payload)


@dataclass(frozen=True)
class ProtocolPhase:
    name: str
    after: tuple[str, ...]
    owner_domain: tuple[int, ...]
    ownership: str
    work: tuple[tuple[str, float], ...] = ()
    details: tuple[tuple[str, object], ...] = ()
    owner_basis: str = "assignment_rows"
    steps: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


def _staged(name, after, owners, ownership, steps, *, owner_basis="domain", details=()):
    """Source-ordered physical services; each byte belongs to exactly one step."""
    work = {}
    for _, values in steps:
        for role, amount in values:
            work[role] = work.get(role, 0) + amount
    return ProtocolPhase(name, after, owners, ownership, tuple(work.items()),
                         details, owner_basis, tuple(steps))


def _read(amount):
    return (("hbm", amount), ("mte2_aiv", amount))


def _write(amount):
    return (("hbm", amount), ("mte3", amount))


@dataclass(frozen=True)
class ProtocolPlan:
    kind: str
    layout: RowLayout
    phases: tuple[ProtocolPhase, ...]
    replaceable_generic_work: tuple[str, ...]
    retained_generic_work: tuple[str, ...]
    unmodeled_features: tuple[str, ...]
    count_basis: str

    def to_dict(self):
        return {**asdict(self), "source": SOURCE,
                "boundary": "physical_phase_ledger; not_a_second_execution_engine"}


def _geometry(source_assignments, received_assignments, experts, ep, aiv_count):
    if (ep < 1 or experts < 1 or experts % ep or aiv_count < 2
            or not math.isfinite(source_assignments) or source_assignments < 0
            or not math.isfinite(received_assignments) or received_assignments < 0):
        raise ModelingError("mc2_fullmesh_geometry_outside_supported_domain")
    count_cores = min(max(experts // _EXPERTS_PER_COUNT_CORE_TARGET, 1),
                      aiv_count // 2, _MAX_COUNT_CORES, experts)
    payload_cores = aiv_count - count_cores
    return tuple(range(payload_cores)), tuple(range(payload_cores, aiv_count))


def assignment_prefix_work(source_tokens, top_k, source_assignments, owners):
    """SendToMoeExpert calls CalTokenSendExpertCnt(index), independently of E counts.

    Full valid index grids give an exact per-owner arithmetic-series sum. A
    filtered/expected assignment count does not identify retained index positions.
    """
    ids = source_tokens * top_k
    if float(source_tokens) % 1 or float(ids) % 1 or source_assignments != ids:
        return None
    ids, stride = int(ids), len(owners)
    result = []
    for owner in owners:
        first = top_k + (owner - top_k) % stride
        count = max(0, (ids - 1 - first) // stride + 1)
        index_sum = count * (2 * first + (count - 1) * stride) // 2
        # Duplicate, Sub, Abs, Mins and ReduceSum on the preceding index IDs.
        result.append((owner, _COUNT_VECTOR_OPS_PER_ID * index_sum - count))
    return tuple(result)


def _dispatch_quant_details(layout, source_tokens, top_k, rows, ids, send, unknown):
    quant_ops = 0
    if layout.quant_mode == "per_token":
        ops_per_element = INT8_QUANT_OPS if layout.payload_dtype == "int8" else FP8_QUANT_OPS
        quant_ops = ops_per_element * layout.width - 1
        unknown.append("per-token scalar reciprocal and scale store instruction service")
    elif (layout.quant_mode == "mx" and layout.input_dtype == "bf16"
          and layout.payload_dtype in {"fp8_e4m3", "fp4_e2m1", "fp4"}
          and layout.width % MX32_TILE_ELEMS == 0):
        quant_ops = sum(mx32_work(layout.width, payload_dtype=layout.payload_dtype).values())
        unknown.append("MX register load/store throughput and scalar control instruction service")
    elif layout.quant_mode in {"mx", "per_group"}:
        unknown.append("fused quantization vector instruction work: keep explicit caller quant program")
    elif layout.quant_mode != "none":
        raise ModelingError("mc2_dispatch_row_layout_kind_invalid")
    prefix_work = assignment_prefix_work(source_tokens, top_k, rows, send)
    if prefix_work is None:
        unknown.append("sender prefix scan requires retained valid assignment indices; total count alone is insufficient")
    prefix_total = sum(amount for _, amount in (prefix_work or ()))
    quant_steps = (("quantize", (("vector", rows * quant_ops),)),)
    quant_details = (("quant_mode", layout.quant_mode),
        ("assignment_prefix_scan", {
            "source": "SendToMoeExpert:652-663 -> CalTokenSendExpertCnt:778-804",
            "scope": "full_integer_valid_ID_grid" if prefix_work is not None else "unresolved_retained_indices",
            "owner_vector_work": prefix_work,
            "vector_work": prefix_total,
            "owner_mapping": "expanded_assignment_index_mod_payload_AIV",
            "count_core_expert_scan": "separate_existing_E_times_full_ID_scan",
            "scalar_result_reads_unpriced": max(0, ids - top_k) if prefix_work is not None else None,
            "scalar_source_id_reads_unpriced": rows,
            "scalar_triple_stores_unpriced": EXPAND_IDX_VALUES_PER_ROW * rows,
            "cal_token_send_expert_cnt": {
                "source": "CalTokenSendExpertCnt:778-804",
                "tensor_dtype": "int32",
                "short_vector": int(top_k),
                "ops_per_expert": ("Duplicate", "Sub", "Abs", "Mins", "software ReduceSum"),
                "pipe_v_per_expert": _COUNT_PIPE_V_OPS_PER_EXPERT,
                "v_s_dependency": True,
                "int32_ub_read_bytes": rows * INT32_BYTES,
                "int32_ub_write_bytes": prefix_total * INT32_BYTES,
            },
        }))
    if layout.quant_mode == "mx" and quant_ops:
        per_row = mx32_work(layout.width, payload_dtype=layout.payload_dtype)
        quant_steps = tuple((name, (("vector", rows * amount),)) for name, amount in per_row.items())
        quant_details += (("quantization_source", MX_SOURCE),
                          ("quantization_repetitions", rows),
                          ("quantization_per_row", per_row))
    return prefix_total, quant_steps, quant_details


def _is_serial_mode(
    source_tokens, ep, *, global_bs: int | float = 0,
    is_scaling_down: bool = False, shared_expert_ranks: int = 0,
) -> bool:
    """Native A5 fullmesh serial-mode tiling guard (``moe_distribute_dispatch_v2_a5_full_mesh.h:441,453``)."""
    del shared_expert_ranks
    axis_max_bs = float(global_bs) / float(ep) if global_bs and int(ep) > 0 else float(source_tokens)
    return (
        not is_scaling_down
        and int(ep) < _SERIAL_MODE_MAX_EP_WORLD_SIZE
        and axis_max_bs <= _SERIAL_MODE_MAX_BS_PER_RANK
    )


def directed_cell_payload(
    source: int,
    destination: int,
    *,
    peer_count: int,
    packet_bytes: int = 0,
    route_matrix=None,
    route_reverse: bool = False,
    route_matrix_bytes=None,
    uniform_payload_bytes: float = 0.0,
) -> tuple[float, int]:
    """Return ``(useful_bytes, frame_count)`` for one directed ``source -> destination`` cell."""
    source, destination = int(source), int(destination)
    if source == destination:
        return 0.0, 0
    if route_matrix:
        frames = int(
            route_matrix[destination][source]
            if route_reverse
            else route_matrix[source][destination]
        )
        return float(frames) * float(packet_bytes), frames
    if route_matrix_bytes:
        return float(route_matrix_bytes[source][destination]), 0
    ep = max(1, int(peer_count))
    return (float(uniform_payload_bytes) / ep if ep > 1 else 0.0), 0


def directed_transport_cells(
    *,
    group: tuple[int, ...],
    sources: tuple[int, ...],
    experts: int = 0,
    packet_bytes: int = 0,
    route_matrix=None,
    route_reverse: bool = False,
    route_matrix_bytes=None,
    uniform_payload_bytes: float = 0.0,
    count: bool = False,
) -> tuple[tuple[int, int, int, int, float, int], ...]:
    """Canonical directed payload or count cells owned by ``sources`` over ``group``."""
    group_size = len(group)
    if group_size <= 1:
        return ()
    cells = []
    for source in sources:
        if not 0 <= source < group_size:
            continue
        if count:
            experts_per_peer = max(0, int(experts) // group_size)
            payload = float(experts_per_peer * EXPERT_STATUS_RECORD_BYTES)
            if payload > 0.0:
                for destination in range(group_size):
                    if destination == source:
                        continue
                    cells.append((
                        source, destination, group[source], group[destination], payload, 0,
                    ))
            continue
        for destination in range(group_size):
            if destination == source:
                continue
            payload, frames = directed_cell_payload(
                source,
                destination,
                peer_count=group_size,
                packet_bytes=packet_bytes,
                route_matrix=route_matrix,
                route_reverse=route_reverse,
                route_matrix_bytes=route_matrix_bytes,
                uniform_payload_bytes=uniform_payload_bytes,
            )
            if payload <= 0.0:
                continue
            cells.append((
                source, destination, group[source], group[destination], payload, frames,
            ))
    return tuple(cells)


def dispatch_plan(layout, *, source_tokens, top_k, source_assignments,
                  received_assignments, experts, ep, aiv_count,
                  global_bs=0, is_scaling_down=False, shared_expert_ranks=0):
    """Source rows include local destinations; peer routing removes diagonals.

    Fractional counts are expected work. Their eligible owner domain must not
    be interpreted as all cores being active; integer routes identify owners.
    """
    send, count = _geometry(source_assignments, received_assignments, experts, ep, aiv_count)
    all_cores = tuple(range(aiv_count))
    if source_tokens < 0 or top_k < 1 or source_assignments > source_tokens * top_k:
        raise ModelingError("mc2_assignment_count_exceeds_source_ids")
    serial = _is_serial_mode(
        source_tokens, ep, global_bs=global_bs,
        is_scaling_down=is_scaling_down, shared_expert_ranks=shared_expert_ranks,
    )
    if serial:
        send = all_cores
        count = tuple(range(min(aiv_count, experts)))
    send_after = ("count_status_exchange",) if serial else ("routing_metadata",)
    rows, recv = source_assignments, received_assignments
    ids = source_tokens * top_k
    unknown = ["scalar GetValue/SetValue, scalar prefix arithmetic and extra flag polling iteration service",
               "payload flag-predicate/reduction and active-mask instruction service",
               "control/metadata cache transaction amplification",
               "native UB-derived token window and actual nonuniform destination ownership"]
    mask_count = (max(1, int(source_assignments))
                  if source_assignments > 0 else 0)
    count_vector = len(count) * _COUNT_STATUS_INIT_OPS_PER_EXPERT * _align(experts, _COUNT_SYNC_RECORD_ELEMS)
    c, a = len(count), aiv_count
    prefix_owners = tuple(range(min(a, experts)))
    prefix_total, quant_steps, quant_details = _dispatch_quant_details(
        layout, source_tokens, top_k, rows, ids, send, unknown)
    id_init_work = (a * _align(ids * INT32_BYTES, _ROUTING_ID_INIT_ALIGNMENT_BYTES) / INT32_BYTES
                    if ids else 0)
    phases = (
        _staged("routing_metadata", (), all_cores, "each_AIV_reads_the_complete_ID_array", (
            ("initialize", (("vector", id_init_work),)),
            ("read", _read(a * ids * INT32_BYTES)),
        ), details=(("replicated_id_arrays", a),)),
        ProtocolPhase("count_compute", ("routing_metadata",), count, "contiguous_expert_intervals", (
            ("vector", count_vector),), (
                ("includes_zero_count_experts", True),
                ("scalar_count_stores", experts),
                ("serial_mode", serial),
                ("cal_expert_cnt", {
                    "source": "CalExpertCnt:808-820",
                    "tensor_dtype": "int32 indices; Abs/ReduceSum reinterpret float",
                    "mask_count": mask_count,
                    "mask_basis": ("source_assignments; activeMaskBsCnt unknown"
                                   if mask_count else "unresolved"),
                    "short_vector": mask_count,
                    "ops_per_expert": ("Duplicate", "Sub", "Abs(float)",
                                       "Mins", "software ReduceSum(float)"),
                    "owner_domain": "contiguous_expert_intervals",
                    "pipe_v_per_expert": _COUNT_PIPE_V_OPS_PER_EXPERT,
                    "v_s_dependency": True,
                    "vector_ops_per_owner": _COUNT_VECTOR_OPS_PER_ID * mask_count,
                    "int32_ub_read_bytes_per_expert": INT32_BYTES * mask_count,
                    "int32_ub_write_bytes_per_expert": INT32_BYTES,
                })), "domain"),
        ProtocolPhase("count_status_exchange", ("count_compute",), count,
                      "contiguous_expert_intervals_to_destination_ranks", (
            ("mte3", experts * EXPERT_STATUS_RECORD_BYTES), ("hbm", experts * EXPERT_STATUS_RECORD_BYTES)), (
            ("bytes_per_expert_record", EXPERT_STATUS_RECORD_BYTES), ("records_per_source", experts),
            ("bytes_per_source_including_local", experts * EXPERT_STATUS_RECORD_BYTES),
            ("remote_bytes_per_source", experts * EXPERT_STATUS_RECORD_BYTES * (ep - 1) / ep)), "domain"),
        _staged("send_read_quantize", send_after, send,
                "assignment_index_mod_payload_AIV", (
                    ("assignment_prefix_scan", (("vector", prefix_total),)),
                    ("read", _read(rows * (layout.input_bytes + layout.input_scale_bytes))),
                    *quant_steps,
                ), owner_basis="assignment_rows", details=quant_details),
        ProtocolPhase("frame_payload", ("send_read_quantize",), send,
                      "same_assignment_owner", (("vector", rows * layout.data_region_bytes / INT32_BYTES),),
                      (("per_row_packet_bytes", layout.packet_bytes),
                       ("reserved_flags_initialization", "UB buffers once per active owner; separately unresolved"))),
        ProtocolPhase("payload_exchange", ("frame_payload",), send,
                      "declared_peer_assignment_cells", (
            ("mte3", rows * layout.packet_bytes),
            ("hbm", recv * layout.packet_bytes)), (
            ("packet_bytes_per_assignment", layout.packet_bytes),
            ("source_assignments_including_local", rows),
            ("received_assignments_including_local", recv),
            ("remote_count_basis", "caller peer matrix; no implicit uniform split"))),
        _staged("receive_counts", ("count_status_exchange",), count,
                "contiguous_source_expert_status_intervals", (
                    ("read_status", _read(experts * EXPERT_STATUS_RECORD_BYTES)),
                    ("check_status", (("vector", experts - c),)),
                    ("zero_status", (("vector", experts),)),
                    ("clear_status", _write(experts * EXPERT_STATUS_RECORD_BYTES)),
                    ("sum_counts", (("vector", 2 * experts - c + (0 if serial else 2 * c * c)),)),
                    ("publish_sums", _write((c if serial else c * c) * _COUNT_SYNC_RECORD_BYTES)),
                ), details=(("polls_counted", 1), ("additional_spin_polls", "unmodeled"))),
        _staged("prefix_counts", ("receive_counts",), count,
                "count_core_prefix_exchange_and_replicated_offsets", (
                    ("read_sums", _read((c if serial else c * c) * _COUNT_SYNC_RECORD_BYTES)),
                    ("check_and_prefix", (("vector", c * (c - 1) if serial else c * (2 * c - 1) + (c - 1) ** 2),)),
                    ("zero_sums", (("vector", c * _COUNT_SYNC_RECORD_ELEMS if serial else c * _align(c, _COUNT_SYNC_RECORD_ELEMS) * _COUNT_SYNC_RECORD_ELEMS),)),
                    ("clear_sums", _write((c if serial else c * c) * _COUNT_SYNC_RECORD_BYTES)),
                    ("publish_offsets", _write((a + 1) * experts * INT32_BYTES)),
                    ("initialize_ready", (("vector", 0 if serial else c * _align(a, _COUNT_SYNC_RECORD_ELEMS) * _COUNT_SYNC_RECORD_ELEMS),)),
                    ("publish_ready", _write((0 if serial else c * a) * _COUNT_SYNC_RECORD_BYTES)),
                ), details=(("prefix_output_copies", a + 1),
                            ("scalar_prefix_additions", experts))),
        _staged("prefix_ready", ("prefix_counts",), all_cores,
                "every_AIV_checks_and_clears_its_count_core_ready_flags", (
                    ("read_ready", _read((0 if serial else a * c) * _COUNT_SYNC_RECORD_BYTES)),
                    ("check_ready", (("vector", 0 if serial else a * (c * _COUNT_SYNC_RECORD_ELEMS - 1)),)),
                    ("zero_ready", (("vector", 0 if serial else a * c * _COUNT_SYNC_RECORD_ELEMS),)),
                    ("clear_ready", _write((0 if serial else a * c) * _COUNT_SYNC_RECORD_BYTES)),
                ), details=(("polls_counted", 1),)),
        _staged("receive_metadata", ("prefix_ready",), prefix_owners,
                "each_nonempty_source_expert_interval_reads_all_prefix_offsets", (
                    ("read_prefix", _read(len(prefix_owners) * experts * INT32_BYTES)),
                )),
        _staged("expert_counts", ("receive_metadata",), (0,),
                "AIV_zero_publishes_local_INT64_expert_counts", (
                    ("write_counts", _write(experts // ep * INT64_BYTES)),
                )),
        ProtocolPhase("receive_flags", ("payload_exchange", "receive_metadata"), all_cores,
                      "same_receive_owner", (("hbm", recv * layout.flag_bytes),
                                             ("mte2_aiv", recv * layout.flag_bytes)),
                      (("polls_counted", "unknown_causal"),
                       ("poll_iteration_source", "WaitDispatch:1087-1112"),
                       ("poll_termination", "matching_peer_send_event"),
                       ("per_iteration_work", "MTE2 flag read + Compare/Sum + SFF issue"),
                       ("additional_spin_polls", "unknown; no universal count"))),
        _staged("receive_local_copy", ("receive_flags",), all_cores,
                "source_expert_intervals; active owners require route cells", (
                    ("read", _read(recv * (layout.data_region_bytes + layout.expert_scale_bytes))),
                    ("write", _write(recv * layout.materialized_bytes)),
                ), owner_basis="assignment_rows",
                details=(("source_buffer_slots", 2), ("triple_output_bytes_per_row", TRIPLE_META_BYTES),
                         ("execution", "whole_phase_read_then_write; token_window_overlap_unresolved"))),
        ProtocolPhase("clear_data_flags", ("receive_local_copy", "expert_counts"), all_cores,
                      "same_receive_owner", (("hbm", recv * layout.flag_bytes),
                                             ("mte3", recv * layout.flag_bytes)),
                      (("flag_bytes_per_block", _FRAME_FLAG_BYTES),)),
    )
    return ProtocolPlan("dispatch", layout, phases,
                        ("route_pack payload read/write", "separate expert-input quantization when physically fused"),
                        ("router IDs and weights", "non-transport upstream projections"), tuple(unknown),
                        "expected_assignments" if any(float(n) % 1 for n in (rows, recv)) else "declared_integer_counts")


def combine_plan(layout, *, source_assignments, received_assignments, experts, ep, aiv_count,
                 source_tokens=None, top_k=None):
    """Return framing/slots and optional token-owned weighted output publication."""
    _geometry(source_assignments, received_assignments, experts, ep, aiv_count)
    cores = tuple(range(aiv_count))
    rows, recv = source_assignments, received_assignments
    phases = (
        _staged("return_read_frame", (), cores, "contiguous_actual_return_rows", (
            ("read", _read(rows * (layout.input_bytes + TRIPLE_META_BYTES))),
            ("frame", (("vector", rows * (layout.packet_bytes + layout.data_region_bytes) / INT32_BYTES),)),
        ), owner_basis="assignment_rows", details=(("expand_idx_bytes_per_row", TRIPLE_META_BYTES),)),
        ProtocolPhase("payload_exchange", ("return_read_frame",), cores,
                      "transpose_of_dispatch_assignment_cells", (
            ("mte3", rows * layout.packet_bytes),
            ("hbm", recv * layout.packet_bytes)), (
            ("packet_bytes_per_assignment", layout.packet_bytes),
            ("source_assignments_including_local", rows),
            ("received_assignments_including_local", recv))),
        ProtocolPhase("receive_flags", ("payload_exchange",), cores,
                      "contiguous_output_tokens", (("hbm", recv * layout.flag_bytes),
                                                    ("mte2_aiv", recv * layout.flag_bytes)),
                      (("polls_counted", "unknown_causal"),
                       ("poll_iteration_source", "combine:958-985/1570"),
                       ("poll_termination", "matching_peer_send_event"),
                       ("per_iteration_work", "MTE2 flag read + Compare/Sum + SFF issue"),
                       ("additional_spin_polls", "unknown; no universal count")), "output_tokens"),
        ProtocolPhase("receive_slots", ("receive_flags",), cores,
                      "contiguous_output_tokens; wait only required topk slots", (
            ("hbm", recv * layout.data_region_bytes),
            ("mte2_aiv", recv * layout.data_region_bytes)),
                      (("downstream", "weighted combine arithmetic and final output store"),), "output_tokens"),
        ProtocolPhase("clear_data_flags", ("receive_slots",), cores, "same_receive_owner", (
            ("hbm", recv * layout.flag_bytes), ("mte3", recv * layout.flag_bytes)), (), "output_tokens"),
    )
    retained = ("route weights", "weighted sum arithmetic", "final output write")
    replaced = ("route_unpack input payload read",)
    if source_tokens is not None:
        if top_k is None or top_k < 1 or source_tokens < 0 or recv != source_tokens * top_k:
            raise ModelingError("mc2_combine_output_tokens_must_match_return_slots")
        # BF16/FP16 -> FP32 per slot, multiply by FP32 router weight, add into
        # an explicitly zeroed FP32 accumulator, then cast once to output.
        # ProcessExpert completes FP32 accumulation before releasing the
        # incoming flags. The final output cast/store follows that release.
        clear = phases[-1]
        # CheckPackedTokenArriveBatch reads every data-as-flag block, then
        # CompareScalar examines a 64-element-aligned FP32 flag array. Scalar
        # GetValue/SFF work is counted separately, never priced as vector FLOPs.
        flags = top_k * layout.blocks * COMBINE_FLAG_ELEMS_PER_BLOCK
        compared = _align(flags, COMBINE_FLAG_MASK_LANES)
        mask_words = compared // COMBINE_FLAG_MASK_LANES
        successful_flags = _staged("receive_flags", ("payload_exchange",), cores,
            "contiguous_output_tokens; one_successful_batch_flag_check", (
                ("initialize_flags", (("vector", source_tokens * compared),)),
                ("read_flags", _read(recv * layout.flag_bytes)),
                ("compare_flags", (("vector", source_tokens * compared),)),
            ), owner_basis="output_tokens", details=(
                ("polls_counted", "unknown_causal"),
                ("poll_iteration_source", "combine:958-985/1570"),
                ("poll_termination", "matching_peer_send_event"),
                ("per_iteration_work", "MTE2 flag read + Compare/Sum + SFF issue"),
                ("additional_spin_polls", "unknown; no universal count"),
                ("scalar_instruction_counts", {
                    "ub_uint64_mask_reads": source_tokens * mask_words,
                    "scalar_sff_calls": source_tokens * mask_words,
                    "successful_mask_branch_tests": source_tokens * mask_words,
                }),
                ("scalar_timing", "unpriced; no_scalar_instruction_service_declaration"),
                ("source_domain", "no_2D_expert_mask_or_special_experts"),
            ))
        # Payload is consumed once, inside each top-k read/reduction chain.
        # Keep the named boundary for caller graph compatibility, not a second
        # payload read. Native queue depth is unknown; this is an explicit
        # conservative serial-slot schedule, not recovered issue timing.
        ready_slots = ProtocolPhase("receive_slots", ("receive_flags",), cores,
            "payload_read_owned_by_weighted_reduce", owner_basis="output_tokens")
        steps = [("weights", _read(recv * _ROUTER_WEIGHT_BYTES)),
                 ("initialize_accumulator", (("vector", source_tokens * layout.width),))]
        for slot in range(top_k):
            steps.extend((
                (f"slot{slot}_read", _read(source_tokens * layout.data_region_bytes)),
                (f"slot{slot}_cast", (("vector", source_tokens * layout.width),)),
                (f"slot{slot}_multiply", (("vector", source_tokens * layout.width),)),
                (f"slot{slot}_accumulate", (("vector", source_tokens * layout.width),)),
            ))
        phases = phases[:2] + (successful_flags, ready_slots,
            _staged("weighted_reduce", ("receive_slots",), cores,
            "contiguous_output_tokens_with_serial_topk_reduction", tuple(steps),
            owner_basis="output_tokens", details=(
                ("token_repeat_schedule", "serial_per_token; fractional_expected_tail_retained"),
                ("slot_schedule", "conservative_serial; native_bufferNum_unidentified"),
                ("scalar_instruction_counts", {"ub_fp32_router_weight_reads": recv,
                                                "topk_loop_iterations": recv}),
                ("scalar_timing", "unpriced; no_scalar_instruction_service_declaration"),
            )),
            ProtocolPhase(clear.name, ("weighted_reduce",), clear.owner_domain,
                          clear.ownership, clear.work, clear.details, clear.owner_basis),
            _staged("publish_output", ("clear_data_flags",), cores,
                    "contiguous_output_tokens_after_FP32_accumulation_and_flag_release", (
                ("cast", (("vector", source_tokens * layout.width),)),
                ("write_output", _write(source_tokens * layout.payload_bytes)),
            ), owner_basis="output_tokens"),)
        retained, replaced = ("router weight producer",), ("route_unpack payload, weights, reduction and output",)
    return ProtocolPlan("combine", layout, phases, replaced, retained,
                        ("additional flag polling iterations and scalar instruction timing",
                         "unlisted scalar address/control instructions and optional expert metadata",
                         "native UB queue count and active owner mapping require actual routed slots"),
                        "expected_assignments" if any(float(n) % 1 for n in (rows, recv)) else "declared_integer_counts")
