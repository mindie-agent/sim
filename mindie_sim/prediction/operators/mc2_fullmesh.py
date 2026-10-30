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
from .mc2_mx_quant import mx32_work, SOURCE as MX_SOURCE


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
        return self.blocks * 480

    @property
    def flag_bytes(self):
        return self.blocks * 32


def dispatch_layout(width, *, input_dtype="bf16", payload_dtype="bf16",
                    quant_mode="none", input_scale_bytes=0, has_expert_scales=False):
    """Full token layout from QuantInit + FillTriple + 480/512 packaging."""
    if not isinstance(width, int) or width < 1:
        raise ModelingError("mc2_width_requires_positive_integer")
    if input_dtype not in {"bf16", "fp16", "fp8_e4m3", "fp8_e5m2"}:
        raise ModelingError("mc2_input_dtype_unsupported")
    if payload_dtype not in {"bf16", "fp16", "int8", "fp8_e4m3", "fp8_e5m2"}:
        raise ModelingError("mc2_payload_dtype_unsupported")
    if not isinstance(input_scale_bytes, int) or input_scale_bytes < 0:
        raise ModelingError("mc2_input_scales_require_nonnegative_byte_count")
    incoming = int(width * dtype_bytes(input_dtype))
    outgoing = int(width * dtype_bytes(payload_dtype))
    if quant_mode == "none":
        if input_dtype != payload_dtype:
            raise ModelingError("mc2_unquantized_requires_identical_representation")
        scale, offset = input_scale_bytes, _align(outgoing, 32)
    else:
        if input_dtype not in {"bf16", "fp16"} or input_scale_bytes:
            raise ModelingError("mc2_quantized_input_requires_unscaled_float_row")
        if quant_mode == "per_token" and payload_dtype in {"int8", "fp8_e4m3", "fp8_e5m2"}:
            scale, offset = 4, _align(outgoing, 32)
        elif quant_mode == "mx" and payload_dtype in {"fp8_e4m3", "fp8_e5m2"}:
            scale, offset = _align(math.ceil(width / 32), 2), _align(outgoing, 256)
        elif quant_mode == "per_group" and payload_dtype in {"fp8_e4m3", "fp8_e5m2"}:
            scale, offset = math.ceil(width / 128) * 4, _align(outgoing, 128)
        else:
            raise ModelingError("mc2_quantization_branch_unsupported")
    triple = _align(offset + scale, 32)
    # Full-mesh replaces the basic 12B triple tail with one reserved 32B block.
    blocks = math.ceil((triple + 32) / 480)
    expert_scale = 4 if has_expert_scales else 0
    return RowLayout(width, input_dtype, payload_dtype, quant_mode, incoming,
                     outgoing, input_scale_bytes, scale, expert_scale, offset, triple,
                     blocks, blocks * 512, outgoing + scale + 12 + expert_scale)


def combine_layout(width, *, dtype="bf16"):
    """Unquantized BF16/FP16 expert return; metadata is a separate input."""
    if dtype not in {"bf16", "fp16"} or not isinstance(width, int) or width < 1:
        raise ModelingError("mc2_combine_requires_positive_float_row")
    payload = int(width * dtype_bytes(dtype))
    blocks = math.ceil(payload / 480)
    return RowLayout(width, dtype, dtype, "none", payload, payload, 0, 0, 0,
                     0, 0, blocks, blocks * 512, payload)


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
    count_cores = min(max(experts // 16, 1), aiv_count // 2, 8, experts)
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
        result.append((owner, 5 * index_sum - count))
    return tuple(result)


def dispatch_plan(layout, *, source_tokens, top_k, source_assignments,
                  received_assignments, experts, ep, aiv_count):
    """Source rows include local destinations; peer routing removes diagonals.

    Fractional counts are expected work. Their eligible owner domain must not
    be interpreted as all cores being active; integer routes identify owners.
    """
    send, count = _geometry(source_assignments, received_assignments, experts, ep, aiv_count)
    all_cores = tuple(range(aiv_count))
    if source_tokens < 0 or top_k < 1 or source_assignments > source_tokens * top_k:
        raise ModelingError("mc2_assignment_count_exceeds_source_ids")
    rows, recv = source_assignments, received_assignments
    ids = source_tokens * top_k
    quant_ops = 0
    unknown = ["scalar GetValue/SetValue, scalar prefix arithmetic and extra flag polling iteration service",
               "payload flag-predicate/reduction and active-mask instruction service",
               "control/metadata cache transaction amplification",
               "native UB-derived token window and actual nonuniform destination ownership"]
    if layout.quant_mode == "per_token":
        # Cast, abs, reduction, mul, then three casts for INT8 or one for FP8.
        quant_ops = (7 if layout.payload_dtype == "int8" else 5) * layout.width - 1
        unknown.append("per-token scalar reciprocal and scale store instruction service")
    elif (layout.quant_mode == "mx" and layout.input_dtype == "bf16"
          and layout.payload_dtype == "fp8_e4m3" and layout.width % 256 == 0):
        quant_ops = sum(mx32_work(layout.width).values())
        unknown.append("MX register load/store throughput and scalar control instruction service")
    elif layout.quant_mode in {"mx", "per_group"}:
        unknown.append("fused quantization vector instruction work: keep explicit caller quant program")
    elif layout.quant_mode == "none":
        pass
    else:
        raise ModelingError("mc2_dispatch_row_layout_kind_invalid")
    # The second Duplicate sets one flag element per eight-element record.
    count_vector = experts * max(5 * ids - 1, 0) + len(count) * 9 * _align(experts, 8)
    c, a = len(count), aiv_count
    prefix_owners = tuple(range(min(a, experts)))
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
            "scalar_triple_stores_unpriced": 3 * rows,
        }))
    if layout.quant_mode == "mx" and quant_ops:
        quant_steps = tuple((name, (("vector", rows * amount),))
                            for name, amount in mx32_work(layout.width).items())
        quant_details += (("quantization_source", MX_SOURCE),
                          ("quantization_repetitions", rows),
                          ("quantization_per_row", mx32_work(layout.width)))
    phases = (
        _staged("routing_metadata", (), all_cores, "each_AIV_reads_the_complete_ID_array", (
            ("initialize", (("vector", a * _align(ids * 4, 256) / 4 if ids else 0),)),
            ("read", _read(a * ids * 4)),
        ), details=(("replicated_id_arrays", a),)),
        _staged("send_read_quantize", ("routing_metadata",), send,
                "assignment_index_mod_payload_AIV", (
                    ("assignment_prefix_scan", (("vector", prefix_total),)),
                    ("read", _read(rows * (layout.input_bytes + layout.input_scale_bytes))),
                    *quant_steps,
                ), owner_basis="assignment_rows", details=quant_details),
        ProtocolPhase("frame_payload", ("send_read_quantize",), send,
                      "same_assignment_owner", (("vector", rows * layout.data_region_bytes / 4),),
                      (("per_row_packet_bytes", layout.packet_bytes),
                       ("reserved_flags_initialization", "UB buffers once per active owner; separately unresolved"))),
        ProtocolPhase("count_compute", ("routing_metadata",), count, "contiguous_expert_intervals", (
            ("vector", count_vector),), (("includes_zero_count_experts", True),
                                        ("scalar_count_stores", experts)), "domain"),
        ProtocolPhase("count_status_exchange", ("count_compute",), count,
                      "contiguous_expert_intervals_to_destination_ranks", (
            ("mte3", experts * 32), ("hbm", experts * 32)), (
            ("bytes_per_expert_record", 32), ("records_per_source", experts),
            ("bytes_per_source_including_local", experts * 32),
            ("remote_bytes_per_source", experts * 32 * (ep - 1) / ep)), "domain"),
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
                    ("read_status", _read(experts * 32)),
                    ("check_status", (("vector", experts - c),)),
                    ("zero_status", (("vector", experts),)),
                    ("clear_status", _write(experts * 32)),
                    ("sum_counts", (("vector", 2 * experts - c + 2 * c * c),)),
                    ("publish_sums", _write(c * c * 32)),
                ), details=(("polls_counted", 1), ("additional_spin_polls", "unmodeled"))),
        _staged("prefix_counts", ("receive_counts",), count,
                "count_core_prefix_exchange_and_replicated_offsets", (
                    ("read_sums", _read(c * c * 32)),
                    ("check_and_prefix", (("vector", c * (2 * c - 1) + (c - 1) ** 2),)),
                    ("zero_sums", (("vector", c * _align(c, 8) * 8),)),
                    ("clear_sums", _write(c * c * 32)),
                    ("publish_offsets", _write((a + 1) * experts * 4)),
                    ("initialize_ready", (("vector", c * _align(a, 8) * 8),)),
                    ("publish_ready", _write(c * a * 32)),
                ), details=(("prefix_output_copies", a + 1),
                            ("scalar_prefix_additions", experts))),
        _staged("prefix_ready", ("prefix_counts",), all_cores,
                "every_AIV_checks_and_clears_its_count_core_ready_flags", (
                    ("read_ready", _read(a * c * 32)),
                    ("check_ready", (("vector", a * (c * 8 - 1)),)),
                    ("zero_ready", (("vector", a * c * 8),)),
                    ("clear_ready", _write(a * c * 32)),
                ), details=(("polls_counted", 1),)),
        _staged("receive_metadata", ("prefix_ready",), prefix_owners,
                "each_nonempty_source_expert_interval_reads_all_prefix_offsets", (
                    ("read_prefix", _read(len(prefix_owners) * experts * 4)),
                )),
        _staged("expert_counts", ("receive_metadata",), (0,),
                "AIV_zero_publishes_local_INT64_expert_counts", (
                    ("write_counts", _write(experts // ep * 8)),
                )),
        ProtocolPhase("receive_flags", ("payload_exchange", "receive_metadata"), all_cores,
                      "same_receive_owner", (("hbm", recv * layout.flag_bytes),
                                             ("mte2_aiv", recv * layout.flag_bytes)),
                      (("polls_counted", 1), ("additional_spin_polls", "unmodeled"))),
        _staged("receive_local_copy", ("receive_flags",), all_cores,
                "source_expert_intervals; active owners require route cells", (
                    ("read", _read(recv * (layout.data_region_bytes + layout.expert_scale_bytes))),
                    ("write", _write(recv * layout.materialized_bytes)),
                ), owner_basis="assignment_rows",
                details=(("source_buffer_slots", 2), ("triple_output_bytes_per_row", 12),
                         ("execution", "whole_phase_read_then_write; token_window_overlap_unresolved"))),
        ProtocolPhase("clear_data_flags", ("receive_local_copy", "expert_counts"), all_cores,
                      "same_receive_owner", (("hbm", recv * layout.flag_bytes),
                                             ("mte3", recv * layout.flag_bytes)),
                      (("flag_bytes_per_block", 32),)),
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
            ("read", _read(rows * (layout.input_bytes + 12))),
            ("frame", (("vector", rows * (layout.packet_bytes + layout.data_region_bytes) / 4),)),
        ), owner_basis="assignment_rows", details=(("expand_idx_bytes_per_row", 12),)),
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
                      (("polls_counted", 1), ("additional_spin_polls", "unmodeled")), "output_tokens"),
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
        flags = top_k * layout.blocks * 8
        compared = _align(flags, 64)
        successful_flags = _staged("receive_flags", ("payload_exchange",), cores,
            "contiguous_output_tokens; one_successful_batch_flag_check", (
                ("initialize_flags", (("vector", source_tokens * compared),)),
                ("read_flags", _read(recv * layout.flag_bytes)),
                ("compare_flags", (("vector", source_tokens * compared),)),
            ), owner_basis="output_tokens", details=(
                ("polls_counted", 1), ("additional_spin_polls", "unmodeled"),
                ("scalar_instruction_counts", {
                    "ub_uint64_mask_reads": source_tokens * (compared // 64),
                    "scalar_sff_calls": source_tokens * (compared // 64),
                    "successful_mask_branch_tests": source_tokens * (compared // 64),
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
        steps = [("weights", _read(recv * 4)),
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
