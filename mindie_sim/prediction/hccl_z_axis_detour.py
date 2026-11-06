"""Pinned AG/RS Z-axis detour source phases with explicit channel weights.

[Role] Express the AICPU Z-axis detour split on the common HcclPlan DAG:
  level-0 weight R-1, explicit level-1 byte-share weights, 4 KiB floor/tail.
[Invariants] Segment order is caller-declared and preserved. AG channel events
  are physical send matrices with native direction "read"; RS channels are
  "send". Local copies/reduce work stay explicit. Weights are byte shares, not
  a measured port inventory or hardware-bandwidth default.
[Timeouts & Locks] O(ranks * channels * segments) metadata; no locks.
[Side-Effect Policy] Pure plan construction; no I/O or device access.
"""
from __future__ import annotations

from .hccl_contracts import HcclPlan, ModelingError, exact_positive_int
from .hccl_source_shapes import (
    copy_phase, detour_split, exchange_phase, reduce_phase, sync_phase,
)

TEMPLATES = {
    "allgather": "z_axis_detour_allgather",
    "reduce_scatter": "z_axis_detour_reduce_scatter",
}


def contract(selection, rank=0):
    s = selection
    if (s.expansion != "AICPU_TS" or s.collective not in TEMPLATES
            or s.template != TEMPLATES[s.collective]
            or s.topology != "MESH_1D" or s.topology_levels != 1
            or not 0 <= rank < s.ranks):
        raise ModelingError("hccl_z_detour_requires_single_level_aicpu_ag_or_rs")
    if s.buffer_mode != "separate":
        raise ModelingError("hccl_z_detour_requires_separate_buffers")
    r, n, dtype = s.ranks, s.payload_bytes, s.dtype_bytes
    if r < 2 or n < dtype or n % dtype:
        raise ModelingError("hccl_z_detour_requires_positive_element_geometry")
    if not s.segments:
        raise ModelingError("hccl_z_detour_requires_explicit_segments")
    if not s.level1_weights:
        raise ModelingError("hccl_z_detour_requires_explicit_level1_weights")
    segments = tuple(exact_positive_int(value, "hccl_z_detour_segment")
                     for value in s.segments)
    weights = tuple(exact_positive_int(value, "hccl_z_detour_level1_weight")
                    for value in s.level1_weights)
    if any(value % dtype for value in segments):
        raise ModelingError("hccl_z_detour_segments_require_dtype_multiple")
    if s.collective == "allgather":
        peer_total = n
    else:
        if n % (r * dtype):
            raise ModelingError("hccl_z_detour_rs_requires_equal_integer_output_shards")
        peer_total = n // r
    if sum(segments) != peer_total:
        raise ModelingError("hccl_z_detour_segments_must_conserve_peer_payload")
    if s.collective == "reduce_scatter" and len(segments) > 1:
        raise ModelingError("hccl_z_detour_rs_multi_segment_not_implemented")
    channels = 1 + len(weights)
    split_by_segment = tuple(
        detour_split(segment, dtype, r - 1, weights) for segment in segments)
    if any(len(chunks) != channels for chunks in split_by_segment):
        raise ModelingError("hccl_z_detour_channel_count_mismatch")

    phases = []
    if s.collective == "allgather":
        previous = ()
        for segment_index, segment in enumerate(segments):
            out = copy_phase(f"seg{segment_index}_copy_output", previous, segment)
            phases.append(out)
            scratch = copy_phase(f"seg{segment_index}_copy_scratch",
                                 (out.name,), segment)
            phases.append(scratch)
            pre = sync_phase(f"seg{segment_index}_pre_sync",
                             (scratch.name,), (r - 1) * channels)
            phases.append(pre)
            channel_names = []
            for channel_index, chunk in enumerate(split_by_segment[segment_index]):
                matrix = tuple(
                    tuple(chunk if source != destination else 0
                          for destination in range(r))
                    for source in range(r))
                phase = exchange_phase(
                    f"seg{segment_index}_ch{channel_index}", (pre.name,),
                    matrix, rank, native_direction="read")
                phases.append(phase)
                channel_names.append(phase.name)
            post = sync_phase(f"seg{segment_index}_post_sync",
                              tuple(channel_names), (r - 1) * channels)
            phases.append(post)
            previous = (post.name,)
        scratch_bytes = r * n
        threads = (r - 1) * channels
    else:
        pre = sync_phase("pre_sync", (), (r - 1) * channels + 1)
        phases.append(pre)
        channel_names = []
        for segment_index, chunks in enumerate(split_by_segment):
            for channel_index, chunk in enumerate(chunks):
                matrix = tuple(
                    tuple(chunk if source != destination else 0
                          for destination in range(r))
                    for source in range(r))
                phase = exchange_phase(
                    f"seg{segment_index}_ch{channel_index}", (pre.name,),
                    matrix, rank, native_direction="send")
                phases.append(phase)
                channel_names.append(phase.name)
        post = sync_phase("post_sync", tuple(channel_names),
                          (r - 1) * channels + 1)
        phases.append(post)
        copy = copy_phase("result_copy", (post.name,), peer_total)
        phases.append(copy)
        phases.append(reduce_phase("local_reduce", (copy.name,), peer_total, r - 1))
        scratch_bytes = r * peer_total
        threads = (r - 1) * channels + 1

    return HcclPlan(
        s, rank, tuple(phases), threads, scratch_bytes,
        control_counts=(("directed_peer_messages", (r - 1) * channels * len(segments)),
                        ("payload_slices", (r - 1) * channels * len(segments))),
        unmodeled=(
            "level1_weights_are_explicit_native_byte_share_not_port_inventory",
            "multi_segment_sequence_is_explicit_native_capture_not_source_predicate",
            "source_selector_and_native_executor_dispatch_not_established",
            "within_phase_local_copy_and_network_work_overlap_as_phase_envelope",
            "notify_wait_and_task_issue_unpriced_without_declared_control_rate",
        ))
