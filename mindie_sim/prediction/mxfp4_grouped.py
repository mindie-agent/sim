"""CANN MXA8W4_NZNK grouped matrix, with finite AIV-to-Cube supply.

[Role] Bind the standard FP8 x FP4 NZ source branch to the common Engine.
[Invariants] AIV publish precedes Cube consumption; the two L1 weight slots
  retire only after consumption. Fixpipe writes BF16 GM; subsequent SiLU/quant
  remains a separate caller operation. Never add this plan to an old envelope.
[Timeouts & Locks] Lazy finite panels and optional equal-tile run compression.
[Side-Effect Policy] No I/O, device execution, observed timings or hidden fit.
"""
from functools import partial
from collections import Counter

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .mxfp4_geometry import Config, columns, panel_sizes, tile_runs, geometry_ledger
from .mxfp4_registers import CLASSES, TABLES
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill import RefillPipeline
from .mxfp4_panels import compiled_panels, panel_parts

KIND = "ascend_mxa8w4_nznk"
UB_BYTES = (80 + 128 + 1 + 32) * 1024
SOURCE = {
    "repository": "https://gitcode.com/cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "selector": "MXA8W4_NZNK/VECTOR_ANTIQUANT/N_FIRST_TAIL_RESPLIT/DYNAMIC_BUFFER4",
    "files": (
        "gmm/grouped_matmul/op_kernel/grouped_matmul_apt.cpp:425-432",
        "gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp:850-884,1757-1820",
        "gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h:136-251",
        "gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_basic_block.h:267-393",
        "gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_vec_compute.h:285-318,735-831",
        "gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/basic_api/weight_quant_basic_api_v1.h:111-242",
    ),
    "capture_binary_identity": "unverified",
}


def config_from_op(op):
    meta = dict(op.metadata)
    if (op.family != "grouped_gemm"
            or (meta.get("activation_format"), meta.get("weight_format"), meta.get("output_format"))
            != ("fp8_e4m3", "mxfp4", "bf16")
            or meta.get("activation_group_size") != 32
            or meta.get("activation_scale_format") != "fp8_e8m0"):
        raise ModelingError("mxa8w4_nznk_requires_fp8_mxfp4_group32_e8m0_to_bf16")
    rows = tuple(meta.get("matrix_group_rows", ()))
    m, n, k = op.local_shape[-3:]
    config = Config(k, n, rows)
    if sum(rows) != m:
        raise ModelingError("mxa8w4_nznk_integer_rows_disagree_with_operator")
    if meta.get("native_fusion_group") or meta.get("native_fusion"):
        raise ModelingError("mxa8w4_nznk_does_not_own_fused_situ_output")
    return config


def build(op, geometry, *, compact=True, compact_panels=True, mte1_mode="base", producer_mode="serial"):
    config = config_from_op(op)
    if producer_mode not in ("serial", "dynamic_buffer4"):
        raise ModelingError("mxa8w4_unknown_producer_mode")
    cores, aivs = geometry.compute_units, geometry.resource_units.get("vector", 0)
    if aivs != cores * 2:
        raise ModelingError("mxa8w4_nznk_requires_one_aic_two_aiv_source_ownership")
    if mte1_mode not in ("base", "directional"):
        raise ModelingError("mxa8w4_nznk_mte1_mode_requires_base_or_directional")
    directional = mte1_mode == "directional"
    register_roles = ("reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_shuffle_lanes")
    required = {"mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", *register_roles}
    if required - geometry.resource_units.keys():
        raise ModelingError("mxa8w4_nznk_requires_independent_physical_pipes")
    pools = {"ub": UB_BYTES, "l1": 512 * 1024, "l0a": 64 * 1024,
             "l0b": 64 * 1024, "l0c": 256 * 1024}
    if any(geometry.buffers.get(pool, 0) < size for pool, size in pools.items()):
        raise ModelingError("mxa8w4_nznk_source_static_buffers_exceed_device_capacity")
    # InitMx loads the 256-byte transpose permutation on every AIV, even when
    # that owner receives no expert tile. Scalar group-list caches are unknown.
    ledger = geometry_ledger(config, cores, compact, directional)
    total, inventory, counts, runs = ledger.work, ledger.inventory, ledger.counts, ledger.runs
    owner_signatures = [tuple(tile_runs(config, owner, cores, compact)) for owner in range(cores)]
    synchronous_owners = Counter(owner_signatures)

    def owner_program(context, owner):
        ac = context.owned(owner, cores, ("mte2_aic", "mte1", "cube", "fixpipe"))
        vec = [context.owned(owner * 2 + sub, aivs, ("mte2_aiv", "vector", "mte3", *register_roles))
               for sub in (0, 1)]
        allocations = tuple(context.allocate(f"{pool}{owner}", pool, size, (owner,))
                            for pool, size in pools.items() if pool != "ub") if counts[owner] else ()
        allocations += tuple(context.allocate(f"ub{owner}_{sub}", "ub", UB_BYTES,
                                               (2 * owner + sub,)) for sub in (0, 1))
        schedule = tuple(tile_runs(config, owner, cores, compact))

        def tile(index):
            m, n, repeat = schedule[index]
            arguments = dict(k=config.k, m=m, n=n, repeat=repeat,
                             owner=owner, cores=cores, aivs=aivs, directional=directional)
            if compact_panels:
                # Only immutable physical IR is shared. Engine events, output
                # stores and pool ownership belong to this particular call.
                body = compiled_panels(context, **arguments, producer_mode=producer_mode,
                    shared_clients={context.resource_names["hbm"]:
                                    synchronous_owners[owner_signatures[owner]]})
            elif producer_mode == "dynamic_buffer4":
                from .mxfp4_pipeline import SourcePipeline
                from .mxfp4_source_refill import source_panel
                kb, ka, _ = panel_sizes(m, n)
                body = SourcePipeline((config.k + kb - 1) // kb,
                    lambda index: source_panel(context, index=index, **arguments),
                    activation_panels=ka // kb, scale_panels=4096 // kb)
            else:
                kb = panel_sizes(m, n)[0]
                panels = (config.k + kb - 1) // kb
                def panel(index):
                    loads, steps = panel_parts(context, index=index, **arguments)
                    return (Stage("load", program=KernelProgram(loads)),
                            iter((pair, i + 1 == len(steps)) for i, pair in enumerate(steps)), ())
                body = RefillPipeline(panels, 2, 2, panel)

            return (
                Stage("panels", program=body),
                Stage("output", context.bind(Work(resources={
                    "hbm": 2 * m * n * repeat, "fixpipe": 2 * m * n * repeat})),
                      after=("panels",), placement=ac),
            )

        return KernelProgram((
            Stage("reserve", acquire=allocations),
            *(Stage(f"transpose_init{sub}", context.bind(Work(resources={
                "hbm": 256, "mte2_aiv": 256})), after=("reserve",), placement=vec[sub])
              for sub in (0, 1)),
            Stage("tiles", after=("transpose_init0", "transpose_init1"),
                  program=PipelineProgram(runs[owner], 1, tile)),
            Stage("release", after=("tiles",), free=allocations),
        ))

    def program(context):
        return ParallelProgram(tuple(owner_program(context, owner) for owner in range(cores)))

    return OperatorPlan(total, program, {
        "source": SOURCE, "implementation": KIND,
        "coverage": "source_controller_and_resource_pipeline",
        "mxa8w4": {
            "group_rows": config.rows, "valid_rows": sum(config.rows),
            "active_banks": sum(m > 0 for m in config.rows),
            "declared_banks": len(config.rows), "columns": tuple(columns(config, cores)),
            "owner_tiles": counts, "owner_runs": runs,
            "logical_flops": 2 * sum(config.rows) * config.n * config.k,
            "issued_cube_flops": total.matrix.get("fp8_e4m3", 0),
            "mte1_mode": mte1_mode,
            "register_inventory": inventory.to_dict(),
            "register_capacity_work": inventory.capacity_work(CLASSES).resources,
            "register_sources": TABLES,
            "vector_issue_assumption": "source_calls_and_retained_950_capacity_tables; no_FP32_lane_reinterpretation",
            "weight_layout": "explicit_selected_source_FRACTAL_NZ_NK",
            "output": "Fixpipe_FP32_to_BF16_GM; no_SiTU_or_Swiglu",
            "schedule": ("source_four_input_four_widen_two_scale_slots_and_two_L1_L0"
                         if producer_mode == "dynamic_buffer4" else
                         "two_L1_refills_two_L0_consumers; serial_panel_supply_and_tile_drain"),
            "producer_mode": producer_mode,
            "compression": "consecutive_equal_owner_tiles_scaled_work_fluid" if compact else "detailed_tiles",
            "k_panel_execution": "finite_nominal_cycle_coupled_interior" if compact_panels else "detailed_panels",
            "unmodeled": ["scalar_group_list_and_flag_instruction_latency", "hardware_cache_hits",
                          "cross_output_tile_overlap", *(
                              ["AIV_producer_phase_overlap_between_panels"]
                              if producer_mode == "serial" else [])],
        },
    })


def recipe(*, compact=True, compact_panels=True, mte1_mode="base", producer_mode="serial"):
    return PlannedRecipe(partial(build, compact=compact, compact_panels=compact_panels,
                                 mte1_mode=mte1_mode, producer_mode=producer_mode))
