"""Pinned ReduceScatter 4 KiB mesh-chunk source phases.

[Role] Express the dedicated AICPU RS mesh-chunk source: ordered per-segment
  PreCopy, R-1 write-reduce steps with frontRank mapping, PostCopy.
[Invariants] Each output shard segment is split with the source 4 KiB floor and
  exact tail. Segment order is caller-declared and preserved. Every exchange is
  a physical send matrix with native direction "send". No rate or shape table.
[Timeouts & Locks] O(ranks squared * segments) metadata; no locks.
[Side-Effect Policy] Pure plan construction; no I/O or device access.
"""
from __future__ import annotations

from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from .hccl_contracts import (
    HCCL_64BIT_ELEMENT_BYTES, HcclPlan, ModelingError, exact_positive_int,
)
from .hccl_source_shapes import (
    copy_phase, exchange_phase, rs_chunks, rs_step_matrix, sync_phase,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/reduce_scatter/algorithm/template/aicpu/ins_temp_reduce_scatter_mesh_1D_meshchunk.cc",
        source_sha256="4835863a85633a2c020e7bd5d257c128302e4879f6940dacc5c159c2d3ec5a14",
    ),
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/op_common/op_common.cc",
        source_sha256="09adbab9512711e0e7882b098b60b6f3c6a1c3be30da835868427248bc8e354a",
    ),
)

TEMPLATE = "rs_mesh_chunk"


def contract(selection, rank=0):
    s = selection
    if (s.expansion != "AICPU_TS" or s.collective != "reduce_scatter"
            or s.template != TEMPLATE or s.topology != "MESH_1D"
            or s.topology_levels != 1 or not 0 <= rank < s.ranks):
        raise ModelingError("hccl_rs_mesh_chunk_requires_single_level_aicpu_reduce_scatter")
    if s.buffer_mode != "separate":
        raise ModelingError("hccl_rs_mesh_chunk_requires_separate_buffers")
    if s.level1_weights:
        raise ModelingError("hccl_rs_mesh_chunk_does_not_accept_level1_weights")
    if s.dtype_bytes == HCCL_64BIT_ELEMENT_BYTES:
        raise ModelingError("hccl_aicpu_64bit_reduce_cpu_path_not_implemented")
    r, n, dtype = s.ranks, s.payload_bytes, s.dtype_bytes
    if r < 2 or n < r * dtype or n % (r * dtype):
        raise ModelingError("hccl_rs_mesh_chunk_requires_equal_integer_output_shards")
    shard = n // r
    if not s.segments:
        raise ModelingError("hccl_rs_mesh_chunk_requires_explicit_segments")
    segments = tuple(exact_positive_int(value, "hccl_rs_mesh_chunk_segment")
                     for value in s.segments)
    if any(value % dtype for value in segments):
        raise ModelingError("hccl_rs_mesh_chunk_segments_require_dtype_multiple")
    if sum(segments) != shard:
        raise ModelingError("hccl_rs_mesh_chunk_segments_must_conserve_output_shard")

    phases = []
    previous = ()
    for segment_index, segment in enumerate(segments):
        pre = copy_phase(f"seg{segment_index}_precopy", previous, segment)
        phases.append(pre)
        seg_pre = sync_phase(f"seg{segment_index}_pre_sync", (pre.name,), r - 1)
        phases.append(seg_pre)
        previous = (seg_pre.name,)
        chunks = rs_chunks(segment, dtype, r)
        for step in range(r - 1):
            phase = exchange_phase(
                f"seg{segment_index}_step{step}", previous,
                rs_step_matrix(chunks, step, r), rank,
                remote_reduce=True, native_direction="send")
            phases.append(phase)
            if step < r - 2:
                post = sync_phase(f"seg{segment_index}_post_sync{step}",
                                  (phase.name,), r - 1)
                nxt = sync_phase(f"seg{segment_index}_pre_sync{step + 1}",
                                 (post.name,), r - 1)
                phases.extend((post, nxt))
                previous = (nxt.name,)
            else:
                previous = (phase.name,)
        seg_post = sync_phase(f"seg{segment_index}_post_sync", previous, r - 1)
        phases.append(seg_post)
        post = copy_phase(f"seg{segment_index}_postcopy", (seg_post.name,), segment)
        phases.append(post)
        previous = (post.name,)

    return HcclPlan(
        s, rank, tuple(phases), r - 1, (r - 1) * shard,
        control_counts=(("directed_peer_messages", (r - 1) * len(segments)),
                        ("payload_slices", (r - 1) ** 2 * len(segments))),
        unmodeled=(
            "source_compatible_declaration_not_deployed_binary_identity",
            "segment_sequence_is_explicit_native_capture_not_source_predicate",
            "source_selector_and_native_executor_dispatch_not_established",
            "within_phase_local_copy_and_network_work_overlap_as_phase_envelope",
            "notify_wait_and_task_issue_unpriced_without_declared_control_rate",
        ))
