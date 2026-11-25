"""Explicit source schedule for a quantized grouped callback.

[Role] Bind declared source tiling, NZ rectangles and mixed-core output programs.
[Invariants] Source identity is an opt-in scenario, never inferred from historical
  latency or API compatibility. Tensor work, private copies and publication bytes
  remain distinct. All execution uses the shared matrix and causal Engine.
[Timeouts & Locks] Lazy tile execution, O(groups + cores) retained geometry;
  physical work enumeration is streaming. No I/O, device access or locks.
[Side-Effect Policy] Constructs a call definition and scoped recipe only.
"""
from dataclasses import replace
import math
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.call_builders import CallDefinition
from .call_grouped import FP32_BYTES, INT32_ACCUMULATOR_BYTES, build_quantized_grouped_call
from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul
from mindie_sim.prediction.operators.matrix.matrix_output import PartitionedMatrixOutput
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from .quant_grouped_schedule import (
    IMPLEMENTATION_ID, QuantGroupedTiling, plan_quant_grouped,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp",
        source_sha256="320af4dac634a8c9398494cd484150801b67d3280861fda0dd945db1edef0175",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/quant_adaptive_sliding_window_templates/quant_block_sch.h",
        source_sha256="55196716e988e3d0b502e21795e0e82d6095a63a7d7d4a8a082ac442cc3f695e",
    ),
)

OUTPUT = PartitionedMatrixOutput("fixpipe", 2, 2, 8)
_ARCH35_PRESET = "npu_arch_3510"
_SOURCE_COMPUTE_TILE_LIMIT = 128
_SOURCE_GROUP_SCANS_PER_OWNER = 3


def build_quantized_grouped_source_call(arguments):
    raw = dict(arguments)
    declared_tiling = raw.pop("tiling", None)
    if declared_tiling is None:
        raise ModelingError("source_grouped_requires_explicit_tiling")
    tiling = QuantGroupedTiling(**declared_tiling)
    base = build_quantized_grouped_call(raw)
    spec, original = base.spec, base.ops[0]
    schedule = plan_quant_grouped(spec.expert_rows, spec.n, spec.k, tiling)
    hierarchy = HierarchicalMatmul(
        tiling.base_m, tiling.base_n, tiling.base_k,
        math.gcd(_SOURCE_COMPUTE_TILE_LIMIT, tiling.base_m),
        math.gcd(_SOURCE_COMPUTE_TILE_LIMIT, tiling.base_n))
    counts = dict(original.metadata)["group_list_scan_bytes"]
    base_work = Work(matrix={"int8": original.flops}, resources={
        "hbm": original.hbm_bytes - counts, "mte2": original.mte2_bytes,
        "mte3": original.mte3_bytes, "vector": original.vector_ops,
    })
    # This ledger counts declared copies, not native cache transactions or
    # scalar instructions. It streams the same finite rectangular partition.
    copies = inner = scales = publish = 0
    for owner in range(tiling.used_cores):
        for tile in schedule.iter_owner_tiles(owner):
            copies += (tile.m + tile.n) * spec.k
            inner += (tile.m * spec.k * math.ceil(tile.n / hierarchy.compute_n)
                      + tile.n * spec.k * math.ceil(tile.m / hierarchy.compute_m))
            scales += FP32_BYTES * (tile.m + tile.n * min(2, tile.m))
            publish += math.ceil(tile.m / 2) * 2 * tile.n * INT32_ACCUMULATOR_BYTES

    def plan(op, geometry):
        arch = getattr(geometry, "compute_arch", None)
        datapath = getattr(geometry, "datapath", None)
        has_l0c_to_ub = bool(datapath.has_l0c_to_ub) if datapath is not None else False
        if arch is None or arch.preset != _ARCH35_PRESET or not has_l0c_to_ub:
            preset = arch.preset if arch is not None else "unknown"
            raise ModelingError(f"source_grouped_requires_arch35_l0c_to_ub:{preset}")
        if tiling.used_cores > geometry.compute_units:
            raise ModelingError("source_grouped_tiling_exceeds_physical_owners")
        operand_role = read_role(geometry.resource_units)
        scale_role = read_role(geometry.resource_units, aiv=0)
        resources = {"hbm": original.hbm_bytes, "mte1": inner, "fixpipe": publish,
                     "vector": original.vector_ops, "mte3": original.mte3_bytes}
        resources[operand_role] = resources.get(operand_role, 0) + copies
        resources[scale_role] = resources.get(scale_role, 0) + scales

        def execute(context):
            matrix_context = context.for_phase(
                base_work, kernel=IMPLEMENTATION_ID + ":matrix")
            owner_tiles = tuple(
                (count, ((tile.m, tile.n) for tile in schedule.iter_owner_tiles(owner)))
                for owner, count in enumerate(schedule.owner_tile_counts))
            program = hierarchy.lower(
                matrix_context, owner_tiles=owner_tiles, output=OUTPUT, serial_tiles=True)
            return KernelProgram((
                Stage("group_counts", {context.resource_names["hbm"]: counts}),
                Stage("matrix", after=("group_counts",), program=program),
            ))

        return OperatorPlan(Work({"int8": original.flops}, resources), execute)

    op = replace(
        original, recipe=IMPLEMENTATION_ID, source_boundary="kernel",
        coverage="source_geometry_declared_matrix_micro_mapping",
        metadata=(*original.metadata, ("source_schedule", schedule.summary()),
                  ("historical_dispatch_identified", False)))
    ledger = {
        **base.ledger,
        "model": IMPLEMENTATION_ID,
        "coverage_basis": "pinned_source_schedule_with_declared_tiling_and_shared_matrix_core",
        "source_schedule": schedule.summary(),
        "nz_access": {
            "storage": "[E,N/32,K/16,16,32]",
            "k_panel": tiling.base_k,
            "whole_k16_n32_blocks": True,
            "partial_k_panel": "one_range_per_N32_block_with_full_K_stride",
            "storage_spans": "schedule.iter_weight_spans; not_native_DMA_transactions",
            "bandwidth_work": "payload_bytes_with_ideal_shared_HBM_reuse",
        },
        "matrix": {
            **base.ledger["matrix"], "recipe": IMPLEMENTATION_ID,
            "core_recipe": "hierarchical_matmul",
            "tiling_basis": "caller_declared_not_recovered_host_dispatch",
            "inner_mapping": {
                "compute_m": hierarchy.compute_m, "compute_n": hierarchy.compute_n,
                "basis": "shared_core_divisor_mapping_not_native_MatmulImpl_dispatch",
            },
            "owner_boundary": "next_tile_waits_for_local_output_completion",
        },
        "output_path": {
            "transfer": "fixpipe_int32_to_two_vector_local_buffers",
            "row_split": "ceil_half_with_empty_partition_skipped",
            "publication_bytes": publish,
            "private_scale_bytes": scales,
            "buffer_lifetime": "owner_invocation",
            "useful_store_bytes": original.mte3_bytes,
        },
        "group_control": {
            "original_groups": spec.experts,
            "launched_matrix_owners": tiling.used_cores,
            "source_group_visits": spec.experts * tiling.used_cores * _SOURCE_GROUP_SCANS_PER_OWNER,
            "count_hbm_policy": "one_shared_cached_argument_fetch",
            "scalar_issue_and_handshake_service": "unresolved_not_charged",
        },
        "unresolved": [
            "historical_loaded_binary_tiling_and_source_branch_identity",
            "actual_host_tiling_and_MatmulImpl_inner_load_instruction_selection",
            "scalar_group_scan_address_control_and_cross_core_handshake_service",
            "NZ_transactions_cache_residency_and_instruction_startup",
            "host_dispatch_output_allocation_and_graph_internal_submission_gaps",
        ],
        "interpretation": (
            "Explicit source-geometry scenario; no historical dispatch identification, "
            "timing calibration, event-boundary equality or accuracy claim."
        ),
    }
    recipes = RecipeBook({IMPLEMENTATION_ID: PlannedRecipe(plan)})
    return CallDefinition((op,), recipes, ledger, tuple(ledger["unresolved"]))
