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
    """[Cohesive-Pipeline] Build one source-tiled quantized grouped GEMM call with explicit tiling contract."""
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
    base_resources = {
        "hbm": original.hbm_bytes - counts, "mte2": original.mte2_bytes,
        "mte3": original.mte3_bytes, "vector": original.vector_ops,
    }
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

    def _int8_fractal_dims(geometry) -> tuple[int, int, int]:
        from .common import BYTE_PRECISION_FRACTAL_K, FRACTAL_M_DIM, FRACTAL_N_DIM
        datapath = getattr(geometry, "datapath", None)
        waves = getattr(datapath, "subtile_wave", ()) if datapath is not None else ()
        wave = next((item for item in waves if getattr(item, "native_mode", None) == "int8"), None)
        if wave is not None:
            return int(wave.tile_m), int(wave.tile_n), int(wave.tile_k)
        return FRACTAL_M_DIM, FRACTAL_N_DIM, BYTE_PRECISION_FRACTAL_K

    def _tile_subtile_flops(tile_m: int, tile_n: int, kd: int, dims: tuple[int, int, int]) -> tuple[int, ...]:
        tm, tn, tk = dims
        inner_n = math.ceil(tile_n / hierarchy.compute_n)
        count = math.ceil(tile_m / hierarchy.compute_m) * inner_n
        k_issues = math.ceil(kd / tk)
        unit = 2 * tm * tn * tk
        out = []
        for part in range(count):
            ai = min(hierarchy.compute_m, tile_m - (part // inner_n) * hierarchy.compute_m)
            bi = min(hierarchy.compute_n, tile_n - (part % inner_n) * hierarchy.compute_n)
            out.append(math.ceil(ai / tm) * math.ceil(bi / tn) * k_issues * unit)
        return tuple(out)

    def _rebind_cube_by_tile(program, context, tile_lists, dims: tuple[int, int, int]):
        from mindie_sim.prediction.operators.matrix.matrix import _StreamRefillProgram
        from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
        from mindie_sim.prediction.program.stages import ParallelProgram

        cube_name = context.resource_names.get("cube", "cube")
        nk = math.ceil(spec.k / hierarchy.tile_k)
        k_slices = tuple(
            min(hierarchy.tile_k, spec.k - kk * hierarchy.tile_k) for kk in range(nk)
        )
        nonempty_tiles = [tiles for tiles in tile_lists if tiles]

        def _wrap_refill(refill: RefillPipeline, owner_tiles_seq):
            orig_panel = refill.panel

            def _panel(index, _orig=orig_panel, _tiles=owner_tiles_seq):
                load, steps, release = _orig(index)
                tile_m, tile_n = _tiles[index // nk]
                sub_flops = _tile_subtile_flops(tile_m, tile_n, k_slices[index % nk], dims)
                new_steps = []
                for part, (pair, is_last) in enumerate(steps):
                    bound_cube = context.bind(Work(matrix={"int8": sub_flops[part]})).get(cube_name)
                    updated = tuple(
                        replace(st, work={**st.work, cube_name: bound_cube})
                        if (st.name == "matrix" and bound_cube is not None)
                        else st
                        for st in pair
                    )
                    new_steps.append((updated, is_last))
                return load, tuple(new_steps), release

            return replace(refill, panel=_panel)

        def _patch_core(core_prog, owner_tiles_seq):
            if isinstance(core_prog, _StreamRefillProgram):
                return replace(core_prog, pipeline=_wrap_refill(core_prog.pipeline, owner_tiles_seq))
            if isinstance(core_prog, KernelProgram):
                return replace(
                    core_prog,
                    stages=tuple(
                        replace(st, program=_patch_core(st.program, owner_tiles_seq))
                        if st.program is not None else st
                        for st in core_prog.stages
                    ),
                )
            return core_prog

        def _patch_outer(outer_prog):
            if isinstance(outer_prog, ParallelProgram) and len(outer_prog.programs) == len(nonempty_tiles):
                return replace(
                    outer_prog,
                    programs=tuple(
                        _patch_core(p, t) for p, t in zip(outer_prog.programs, nonempty_tiles)
                    ),
                )
            if isinstance(outer_prog, KernelProgram):
                return replace(
                    outer_prog,
                    stages=tuple(
                        replace(st, program=_patch_outer(st.program))
                        if st.program is not None else st
                        for st in outer_prog.stages
                    ),
                )
            return outer_prog

        return _patch_outer(program)

    def plan(op, geometry):
        arch = getattr(geometry, "compute_arch", None)
        datapath = getattr(geometry, "datapath", None)
        has_l0c_to_ub = bool(datapath.has_l0c_to_ub) if datapath is not None else False
        if arch is None or arch.preset != _ARCH35_PRESET or not has_l0c_to_ub:
            preset = arch.preset if arch is not None else "unknown"
            raise ModelingError(f"source_grouped_requires_arch35_l0c_to_ub:{preset}")
        if tiling.used_cores > geometry.compute_units:
            raise ModelingError("source_grouped_tiling_exceeds_physical_owners")
        dims = _int8_fractal_dims(geometry)
        nk = math.ceil(spec.k / hierarchy.tile_k)
        k_slices = tuple(min(hierarchy.tile_k, spec.k - kk * hierarchy.tile_k) for kk in range(nk))
        issued_flops = sum(
            sum(_tile_subtile_flops(tile.m, tile.n, kd, dims))
            for owner in range(tiling.used_cores)
            for tile in schedule.iter_owner_tiles(owner)
            for kd in k_slices
        )
        phase_work = Work(matrix={"int8": issued_flops}, resources=base_resources)
        operand_role = read_role(geometry.resource_units)
        scale_role = read_role(geometry.resource_units, aiv=0)
        resources = {"hbm": original.hbm_bytes, "mte1": inner, "fixpipe": publish,
                     "vector": original.vector_ops, "mte3": original.mte3_bytes}
        resources[operand_role] = resources.get(operand_role, 0) + copies
        resources[scale_role] = resources.get(scale_role, 0) + scales

        def execute(context):
            matrix_context = context.for_phase(
                phase_work, kernel=IMPLEMENTATION_ID + ":matrix")
            tile_lists = tuple(
                tuple((tile.m, tile.n) for tile in schedule.iter_owner_tiles(owner))
                for owner in range(len(schedule.owner_tile_counts))
            )
            owner_tiles = tuple(
                (count, tiles)
                for count, tiles in zip(schedule.owner_tile_counts, tile_lists)
            )
            program = hierarchy.lower(
                matrix_context, owner_tiles=owner_tiles, output=OUTPUT, serial_tiles=True)
            program = _rebind_cube_by_tile(program, matrix_context, tile_lists, dims)
            return KernelProgram((
                Stage("group_counts", {context.resource_names["hbm"]: counts}),
                Stage("matrix", after=("group_counts",), program=program),
            ))

        return OperatorPlan(Work({"int8": issued_flops}, resources), execute)

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
