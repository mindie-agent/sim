"""Shared-expert W4A8 single-WeightNz matrix leaf.

[Role] Bind the ordinary shared-expert ``npu_quant_matmul`` W4A8 source call
  (FP8 E4M3 activation x FP4 E2M1 FRACTAL_NZ weight, two E8M0 group-32 scales,
  BF16 output, no expert group list) to the existing finite W4A8 matrix/B8
  physical components.
[Invariants] Only shared-expert gate_up/down ordinary projections with the
  exact source ABI select this plan.  There is no routed-expert row list, no
  SiTU/quant fusion and no grouped controller.  The actual QuantBatchMatmulV4
  host binary/tiling is not claimed; the physical matrix/B8 component is
  shared with the existing A5 ABB5 W4A8 leaf.
[Timeouts & Locks] Bounded immutable panel/ledger construction; no waits.
[Side-Effect Policy] Returns immutable Work/Program metadata; no device I/O.
"""
from __future__ import annotations

from collections import Counter
from functools import partial

from mindie_sim.schema.common import ModelingError

from .mechanism import OperatorPlan, PlannedRecipe, Work
from .mxfp4_geometry import Config, columns, geometry_ledger, panel_sizes, tile_runs
from .mxfp4_grouped import UB_BYTES
from .mxfp4_panels import compiled_panels
from .mxfp4_registers import CLASSES, TABLES
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

KIND = "ascend_shared_w4a8_mxfp_single_nz"
SOURCE = {
    "caller": {
        "repository": "vllm-ascend-v027",
        "revision": "v027/68a158be",
        "path": "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
        "sha256": "68a158be11cbf33bace3032b5b89d32f5d21a0ab206b9777cebbd5520ad0f374",
        "lines": "41-105",
        "call": "torch_npu.npu_quant_matmul(x FP8, weight FP4 NZ, "
                "weight_scale E8M0, pertoken_scale E8M0, "
                "x2_dtype=float4_e2m1fn_x2, group_sizes=[0,0,32], "
                "output_dtype=BF16)",
    },
    "weight_transform": {
        "call": "npu_format_cast(...,29, customize_dtype=fp8_e4m3fn, "
                "input_dtype=float4_e2m1fn_x2); transpose(-1,-2); "
                "weight_scale.reshape(n,k//2,2).transpose(-3,-2)",
        "storage": "FRACTAL_NZ FP4 E2M1, block32 weight scale E8M0",
    },
    "physical_component": {
        "source": "existing A5 ABB5 3510 W4A8 panel/B8 matrix leaf",
        "status": "source_supported_component_not_full_V4_host_binary",
    },
    "host_binary": {
        "status": "candidate_not_identified",
        "reason": "no exact arch35 QuantBatchMatmulV4 FP8 x FP4 host "
                  "tiling/kernel source or independent per-shape profile",
    },
}

SOURCE_INTERFACE = {
    "activation_format": "fp8_e4m3",
    "input_format": "fp8_e4m3",
    "activation_group_size": 32,
    "activation_scale_format": "fp8_e8m0",
    "weight_format": "mxfp4",
    "weight_dtype": "mxfp4",
    "weight_scale_format": "fp8_e8m0",
    "weight_scale_storage_format": "fp8_e8m0",
    "weight_representation": "fp4_group32_packed",
    "output_format": "bf16",
    "compute_format": "bf16",
}
SHARED_FAMILIES = ("gate_up_projection", "down_projection")


def _matrix_shape(op):
    if len(op.local_shape) != 3:
        return None
    m, n, k = (int(value) for value in op.local_shape)
    return (m, n, k) if m > 0 and n > 0 and k > 0 else None


def _source_declaration_valid(meta, m, n, k):
    """Validate the typed declaration once, including explicit bad flags."""
    k_tail = (k + 63) // 64
    expected_xscale = (m, k_tail, 2)
    shape = meta.get("activation_scale_shape")
    if shape is None or tuple(shape) != expected_xscale:
        return False
    wshape = meta.get("weight_scale_shape")
    if wshape is not None and tuple(wshape) != (n, k_tail, 2):
        return False
    layout = meta.get("weight_layout")
    if layout not in {None, "FRACTAL_NZ", "NZ"}:
        return False
    if meta.get("bias") not in {None, False, 0}:
        return False
    if meta.get("bias_shape") not in (None, (), []):
        return False
    return True


def matches_source_interface(op):
    """Exact ordinary shared-expert W4A8 source interface.

    The shared-expert caller has no matrix_group_rows: it is one ordinary
    WeightNz matrix, not the routed GroupedMatmul branch.  W8, INT8, group-64
    weights, non-NZ explicit layouts, bias and malformed E8M0 scale shapes do
    not match.
    """
    if str(getattr(op, "family", "")) not in SHARED_FAMILIES:
        return False
    shape = _matrix_shape(op)
    if shape is None:
        return False
    m, n, k = shape
    meta = dict(op.metadata)
    if meta.get("ffn_category") != "shared_expert":
        return False
    if meta.get("quant_category") != "shared_expert":
        return False
    if meta.get("matrix_group_rows") or meta.get("matrix_groups", 1) != 1:
        return False
    if meta.get("native_fusion_group") or meta.get("native_fusion"):
        return False
    if any(meta.get(key) != value for key, value in SOURCE_INTERFACE.items()):
        return False
    if meta.get("weight_block") not in {None, 32}:
        return False
    if meta.get("weight_group_size") not in {None, 32}:
        return False
    if not _source_declaration_valid(meta, m, n, k):
        return False
    return True


def config_from_op(op):
    if not matches_source_interface(op):
        raise ModelingError("shared_w4a8_requires_fp8_mxfp4_group32_e8m0_bf16")
    m, n, k = _matrix_shape(op)
    if k % 64 or n % 64:
        raise ModelingError("shared_w4a8_requires_k64_n64")
    return Config(k, n, (m,)), (m, n, k)


def build(op, geometry):
    config, (m, n, k) = config_from_op(op)
    cores, aivs = geometry.compute_units, geometry.resource_units.get("vector", 0)
    if aivs != cores * 2:
        raise ModelingError("shared_w4a8_requires_one_aic_two_aiv_ownership")
    required = {"mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe",
                "reg_data_writes", "reg_ub_accesses", "reg_ub_stores",
                "reg_shuffle_lanes"}
    if required - geometry.resource_units.keys():
        raise ModelingError("shared_w4a8_requires_independent_physical_pipes")
    pools = {"ub": UB_BYTES, "l1": 512 * 1024, "l0a": 64 * 1024,
             "l0b": 64 * 1024, "l0c": 256 * 1024}
    if any(geometry.buffers.get(pool, 0) < size for pool, size in pools.items()):
        raise ModelingError("shared_w4a8_source_static_buffers_exceed_device_capacity")

    ledger = geometry_ledger(config, cores, True, False)
    total, inventory, counts, runs = ledger.work, ledger.inventory, ledger.counts, ledger.runs
    owner_signatures = [tuple(tile_runs(config, owner, cores, True))
                        for owner in range(cores)]
    synchronous_owners = Counter(owner_signatures)

    def owner_program(context, owner):
        ac = context.owned(owner, cores, ("mte2_aic", "mte1", "cube", "fixpipe"))
        vec = [context.owned(owner * 2 + sub, aivs,
                             ("mte2_aiv", "vector", "mte3",
                              "reg_data_writes", "reg_ub_accesses",
                              "reg_ub_stores", "reg_shuffle_lanes"))
               for sub in (0, 1)]
        allocations = tuple(context.allocate(f"{pool}{owner}", pool, size, (owner,))
                            for pool, size in pools.items() if pool != "ub") \
            if counts[owner] else ()
        allocations += tuple(context.allocate(f"ub{owner}_{sub}", "ub", UB_BYTES,
                                               (2 * owner + sub,))
                             for sub in (0, 1))
        schedule = tuple(tile_runs(config, owner, cores, True))

        def tile(index):
            tm, tn, repeat = schedule[index]
            arguments = dict(k=config.k, m=tm, n=tn, repeat=repeat,
                             owner=owner, cores=cores, aivs=aivs,
                             directional=False)
            body = compiled_panels(context, **arguments, producer_mode="serial",
                shared_clients={context.resource_names["hbm"]:
                                synchronous_owners[owner_signatures[owner]]})
            return (
                Stage("panels", program=body),
                Stage("output", context.bind(Work(resources={
                    "hbm": 2 * tm * tn * repeat,
                    "fixpipe": 2 * tm * tn * repeat})),
                    after=("panels",), placement=ac),
            )

        return KernelProgram((
            Stage("reserve", acquire=allocations),
            Stage("tiles", after=("reserve",),
                  program=PipelineProgram(runs[owner], 1, tile)),
            Stage("release", after=("tiles",), free=allocations),
        ))

    def program(context):
        return ParallelProgram(tuple(owner_program(context, owner)
                                     for owner in range(cores)))

    b8 = {
        op: {"raw_lanes": count.active_lanes,
             "bound_us": (count.active_lanes / (256 * 64 * 1.65e9) * 1e6)}
        for op, count in inventory.counts.items()
        if op in {"ShiftRightB8", "ShiftLeftB8", "SelectB8", "AndB8"}
    }
    return OperatorPlan(total, program, {
        "source": SOURCE,
        "implementation": KIND,
        "coverage": "source_abi_and_existing_W4A8_matrix_B8_leaf",
        "shared_w4a8": {
            "m": m, "n": n, "k": k,
            "valid_rows": m, "declared_banks": 1,
            "matrix_group_rows": (m,),
            "columns": tuple(columns(config, cores)),
            "owner_tiles": counts, "owner_runs": runs,
            "logical_flops": 2 * m * n * k,
            "mmad_parameter_m": (m,),
            "fixpipe_m": (m,),
            "l0a_issue_fractal_m": ((m + 15) // 16 * 16,),
            "issued_cube_flops": total.matrix.get("fp8_e4m3", 0),
            "b8_typed": {
                "raw_active_lanes": sum(
                    count.active_lanes for op, count in inventory.counts.items()
                    if op in {"ShiftRightB8", "ShiftLeftB8", "SelectB8", "AndB8"}),
                "per_opcode": b8,
                "native_i8_per_cycle_per_vf": 256,
                "vf_owners": 64,
                "nominal_clock_hz": 1.65e9,
                "prologue": {"CreateMask": 2, "DuplicateB8": 3,
                             "width_elements": 256, "once_per_invocation": True},
                "single_port_charge": False,
                "priced_on": "reg_issue_slots_and_ub_ports; no aggregate five-op charge",
            },
            "register_inventory": inventory.to_dict(),
            "register_capacity_work": inventory.capacity_work(CLASSES).resources,
            "register_sources": TABLES,
            "output": "Fixpipe BF16 GM; no SiTU/quant fusion",
            "producer_mode": "serial",
            "host_binary_status": "candidate_not_identified",
            "host_binary_reason": "arch35 QuantBatchMatmulV4 FP8xFP4 host "
                                  "tiling/kernel not available; physical "
                                  "matrix/B8 component only",
            "unmodeled": [
                "actual_QuantBatchMatmulV4_host_tiling_and_ISA",
                "actual_quant_scale_epilogue_and_per_shape_overlap",
                "hardware_cache_hits",
            ],
        },
    })


def recipe():
    return PlannedRecipe(build)
