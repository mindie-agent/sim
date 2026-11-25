"""Pinned AR mesh-chunk two-shot source phases on the common HcclPlan DAG.

[Role] Express the separate-buffer AR two-shot mesh-chunk source: element-exact
  rank shards, R-1 secondary chunks per peer, ordered RS steps and final AG read.
[Invariants] Every byte is an integer. RS step exchanges are physical send
  matrices with native direction "send"; the final gather is a send matrix with
  native direction "read". Local copies/reduce work stay explicit. No rate,
  latency or shape table appears here.
[Timeouts & Locks] O(ranks squared) metadata; no locks or external work.
[Side-Effect Policy] Pure plan construction; no I/O or device access.
"""
from __future__ import annotations

from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from .hccl_contracts import HCCL_64BIT_ELEMENT_BYTES, HcclPlan, ModelingError
from .hccl_source_shapes import (
    copy_phase, exchange_phase, slice_sizes, step_matrix, subchunks, sync_phase,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot_mesh_chunk.cc",
        source_sha256="e161c54154da16a624be065580072b21131de830b2950c5055f5f17408ad2bfa",
    ),
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/op_common/op_common.cc",
        source_sha256="09adbab9512711e0e7882b098b60b6f3c6a1c3be30da835868427248bc8e354a",
    ),
)

TEMPLATE = "mesh_chunk_twoshot"


def contract(selection, rank=0):
    s = selection
    if (s.expansion != "AICPU_TS" or s.collective != "allreduce"
            or s.template != TEMPLATE or s.topology != "MESH_1D"
            or s.topology_levels != 1 or not 0 <= rank < s.ranks):
        raise ModelingError("hccl_mesh_chunk_requires_single_level_aicpu_allreduce")
    if s.buffer_mode != "separate":
        raise ModelingError("hccl_mesh_chunk_requires_separate_buffers")
    if s.segments or s.level1_weights:
        raise ModelingError("hccl_mesh_chunk_does_not_accept_segments_or_weights")
    if s.dtype_bytes == HCCL_64BIT_ELEMENT_BYTES:
        raise ModelingError("hccl_aicpu_64bit_reduce_cpu_path_not_implemented")
    r, n, dtype = s.ranks, s.payload_bytes, s.dtype_bytes
    if r < 2 or n < dtype or n % dtype:
        raise ModelingError("hccl_mesh_chunk_requires_positive_element_geometry")
    sizes = slice_sizes(n, dtype, r)
    chunks = tuple(subchunks(size, dtype, r) for size in sizes)

    phases = [copy_phase("precopy", (), sizes[rank])]
    pre = sync_phase("rs_pre_sync", ("precopy",), r)
    phases.append(pre)
    previous = (pre.name,)
    for step in range(r - 1):
        phase = exchange_phase(f"rs_step{step}", previous, step_matrix(chunks, step, r),
                               rank, remote_reduce=True, native_direction="send")
        phases.append(phase)
        if step < r - 2:
            post = sync_phase(f"rs_post_sync{step}", (phase.name,), r)
            nxt = sync_phase(f"rs_pre_sync{step + 1}", (post.name,), r)
            phases.extend((post, nxt))
            previous = (nxt.name,)
        else:
            previous = (phase.name,)
    rs_post = sync_phase("rs_post_sync", previous, r)
    phases.append(rs_post)
    ag_pre = sync_phase("ag_pre_sync", (rs_post.name,), r)
    phases.append(ag_pre)
    phases.append(copy_phase("ag_local_copy", (ag_pre.name,), sizes[rank]))
    gather = tuple(
        tuple(sizes[source] if source != destination else 0
              for destination in range(r))
        for source in range(r))
    phases.append(exchange_phase("gather", (ag_pre.name,), gather, rank,
                                 native_direction="read"))
    phases.append(sync_phase("ag_post_sync", ("ag_local_copy", "gather"), r))

    return HcclPlan(
        s, rank, tuple(phases), r, 2 * n,
        control_counts=(("directed_peer_messages", 2 * (r - 1)),
                        ("payload_slices", (r - 1) ** 2 + (r - 1))),
        unmodeled=(
            "source_compatible_declaration_not_deployed_binary_identity",
            "source_selector_and_native_executor_dispatch_not_established",
            "within_phase_local_copy_and_network_work_overlap_as_phase_envelope",
            "notify_wait_and_task_issue_unpriced_without_declared_control_rate",
            "non_power_of_two_multilevel_and_zero_tail_execution_unverified",
        ))
