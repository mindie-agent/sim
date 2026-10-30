"""A8W4 MSD ordinary grouped-down source branch.

[Role]
  Execute the source-verified ordinary ``npu_grouped_matmul`` INT8 x INT4
  grouped-down call as four explicit coarse phases on the common Engine:
  high/low INT4 pre-processing, INT4 matrix, FP16 dequantization, and the
  BF16 output cast/store.

[Timeouts & Locks]
  O(experts + constant phases); no I/O, captures, subprocesses or locks.

[Side-Effect Policy]
  Returns immutable Work/Program/operator records; no device contexts,
  configuration writes or observations.

[Invariants]
  * Inputs are INT8 activations with one FP32 scale per row, packed per-channel
    INT4 weights, uint64 bit-pattern FP32 column scales, FP32 scale-assist and
    per-token scales, and explicit integer expert rows.  The input
    quantization producer remains a separate ordinary operator.
  * ``bias`` in the public call is the source scale-assist matrix that
    compensates the MSD low-digit shift; it is not a generic linear bias.
  * The matrix rate is the case-declared ``int4`` peak.  There is no
    INT8/FP16 fallback; a missing peak raises
    ``compute_precision_rate_required:int4``.
  * Pre reads ``M*K`` INT8 and writes the ``M*K`` high/low INT4 scratch.  The
    matrix reads that scratch, ``0.5*N*K`` packed weights and ``8*N`` uint64
    column scales per active group, and publishes ``4*M*N`` FP16 scratch via
    Fixpipe/GetTensorC.  Post reads that scratch, the active-group assist and
    the per-token scales.  The final BF16 write is MTE3.
  * Matrix publication remains an HBM transfer.  A Fixpipe resource is added
    only when the execution geometry declares one; no rate is guessed.
  * Zero-row expert groups retain their count entries but read no weight or
    scale bytes. Active M tiles read their weight panels independently; all
    N tiles read their input panels. No shared cache hit is inferred.
  * The four phases use global coarse barriers.  Per-owner stages pin active
    AIC/AIV units and keep shared HBM unpinned.  The model makes no claim to
    native chunk overlap, exact instruction schedule, or captured binary
    identity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from mindie_sim.schema.common import ModelingError

from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, Stage
from .w4a8 import Config, _geometry_owners, _role_work
from .w4a8_schedule import down_schedule, total_work
from .w4a8_matrix import matrix_program, matrix_work


RECIPE = "ascend_w4a8_msd_grouped_down"
PROFILE = "ascend_w4a8_msd_grouped_down"
PHASES = ("pre", "matrix", "post_dequant", "output")
SUPPORTED_RECIPES = ("grouped_down", "balanced_grouped_down")
SUPPORTED_EXPERT_ROW_BASES = {
    "declared_integer_counts", "forced_balanced_integer_counts",
}
LIMITATIONS = (
    "explicit_coarse_phase_order_does_not_claim_native_workspace_pipeline_overlap",
    "source_chunk_boundaries_and_exact_tiles_not_reconstructed",
    "column_scale_charged_once_as_declared_vector_equivalent",
    "matrix_publication_HBM_retained; Fixpipe_service_requires_declared_rate",
    "CV11_or_CV12_host_selection;UB_row_chunks_rotate_with_persistent_vecCount",
    "per_owner_phases_use_global_coarse_phase_barriers_not_chunk_overlap",
    "matrix_uses_shared_steady_K_pipeline_and_declared_inner_mapping",
    "scalar_group_scan_and_cross_core_flag_latency_not_charged",
    "captured_CANN_binary_identity_not_established",
)

SOURCE = {
    "caller": {
        "repository": "vllm-project/vllm-ascend",
        "revision": "8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
        "path": "vllm_ascend/quantization/methods/w4a8/w4a8.py",
        "sha256": "efdcc610c05a44952c2548e0d8b0cd48ca7704c3e2cf8247f8f08f5e765b830b",
        "lines": "535-550",
        "symbol": "apply_gmm2",
        "call": "torch_npu.npu_grouped_matmul(x INT8, weight INT4, "
                "scale UINT64, bias FP32 assist, per_token_scale FP32) -> BF16",
    },
    "tiling": {
        "repository": "ops-transformer",
        "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        "path": "gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp",
        "sha256": "4535b3152b006ce41168c438a20ff214c6dc5892c948f6cf26c399a55c4fb1e5",
        "lines": "2153-2177",
    },
    "kernel": {
        "repository": "ops-transformer",
        "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        "path": "gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h",
        "sha256": "1851c45726a01ada89a2c2f034021185cd5a477bef963037338e6dba0e02e302",
        "lines": "442-516",
    },
    "pre": {
        "repository": "ops-transformer",
        "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        "path": "gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd_pre.h",
        "sha256": "606b515bc5df715f9c049834df81e0e77d9ea9c27a9d8713c1b4cac11d075800",
        "lines": "151-204",
        "scope": "in-place ND high/low packing; no extra NZ workspace requested",
    },
    "branch": "ordinary A8W4 MSD grouped-down (INT8 x INT4, FP16 workspace, BF16 output)",
    "capture_binary_identity": "not_established",
    "host_conditions": {
        "tuning_config": "absent", "offset": "absent",
        "bias": "nonempty FP32 MSD assist", "quant_groups": 1,
        "dispatch": "MSD_API_DEQUANT outside explicit host whitelist",
        "input_prepacking": "in_place_ND",
    },
}


def _error(reason: str):
    raise ModelingError(f"w4a8_down_{reason}")


def _require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        _error(f"{name}_must_be_integer_at_least_{minimum}")
    return value


def _sequence(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _error(f"{name}_must_be_sequence")
    return tuple(value)


def _routing(metadata):
    basis = metadata.get("routing_basis")
    if basis not in SUPPORTED_EXPERT_ROW_BASES:
        _error("rejects_probability_moment_routing")
    raw = metadata.get("routing_realization")
    if not isinstance(raw, Mapping):
        _error("requires_explicit_integer_expert_rows")
    if raw.get("basis") not in SUPPORTED_EXPERT_ROW_BASES:
        _error("rejects_probability_moment_routing")
    if raw.get("basis") != basis:
        _error("routing_basis_disagrees_with_realization")
    rows = raw.get("expert_rows", raw.get("local_expert_rows"))
    rows = tuple(_require_int(value, "expert_rows")
                 for value in _sequence(rows, "expert_rows"))
    if not rows:
        _error("requires_nonempty_expert_rows")
    if "expert_ids" in raw:
        ids = _sequence(raw["expert_ids"], "expert_ids")
        if len(ids) != len(rows) or len(set(ids)) != len(ids):
            _error("expert_ids_must_match_rows_and_be_unique")
    return rows, basis


def _validate_geometry(config):
    m, n, k = config.m, config.n, config.k
    _require_int(m, "m", 1)
    _require_int(n, "n", 1)
    _require_int(k, "k", 1)
    if sum(config.expert_rows) != m:
        _error("expert_rows_must_conserve_logical_matrix_rows")
    if n % 64:
        _error("nz_requires_n_multiple64")
    if k % 64:
        _error("k_must_be_divisible_by_64")
    if k > 18432:
        _error("k_must_not_exceed_18432")
    # The only per-channel shape in the pinned host's alternate vector-dequant
    # whitelist must not silently execute this API-dequant physical ledger.
    if (len(config.expert_rows), m, k, n) == (1, 16, 256, 512):
        _error("host_vector_dequant_whitelist_requires_separate_source_plan")


def _validate_representation(metadata, config):
    m, n, k = config.m, config.n, config.k
    if metadata.get("bias") or metadata.get("linear_bias"):
        _error("generic_linear_bias_not_supported;bias_is_scale_assist_only")
    if metadata.get("transpose_weight"):
        _error("transposed_weight_not_supported")
    if metadata.get("weight_offset") or metadata.get("offset"):
        _error("runtime_weight_offset_not_supported")
    if metadata.get("weight_format") not in {"int4", "w4a8"}:
        _error("requires_packed_int4_weights")
    if metadata.get("weight_dtype") not in {None, "int4"}:
        _error("requires_packed_int4_weights")
    if metadata.get("weight_representation") != "int4_packed_per_channel":
        _error("requires_perchannel_packed_int4_weights")
    if metadata.get("dequant_before_compute") is True:
        _error("requires_fused_source_dequantization")
    if (metadata.get("activation_format") != "int8"
            or metadata.get("input_format") not in {None, "int8"}):
        _error("requires_int8_activations")
    if metadata.get("activation_scale_format") != "fp32":
        _error("requires_fp32_activation_scale")
    if metadata.get("activation_scale_bytes") != 4 * m:
        _error("requires_one_fp32_input_scale_per_row")
    if metadata.get("activation_group_size", 0) < k:
        _error("requires_one_fp32_input_scale_per_row")
    shape = metadata.get("activation_scale_shape")
    if shape is not None and tuple(shape) != (m,):
        _error("activation_scale_shape_must_be_one_per_row")
    if metadata.get("output_format") not in {"bf16", "bfloat16"}:
        _error("requires_bf16_output")
    if metadata.get("output_dtype") not in {None, "bf16", "bfloat16"}:
        _error("requires_bf16_output")


def config_from_op(op):
    """Validate the ordinary semantic down operator and return its exact geometry."""
    if op.family != "grouped_gemm" or op.recipe not in SUPPORTED_RECIPES:
        _error(f"unsupported_recipe:{op.recipe}")
    if len(op.local_shape) != 3 or len(op.semantic_shape) != 3:
        _error("requires_matrix_local_and_semantic_shape")
    m, n, k = (_require_int(value, "matrix_axis", 1)
               for value in op.local_shape)
    if tuple(op.semantic_shape) != (m, n, k):
        _error("semantic_and_local_matrix_shape_mismatch")
    if op.flops != 2 * m * n * k:
        _error("semantic_flops_must_equal_2mnk")
    metadata = dict(op.metadata)
    rows, basis = _routing(metadata)
    config = Config(m, n, k, rows, basis)
    _validate_geometry(config)
    _validate_representation(metadata, config)
    return config


def _config_from_source(op):
    raw = dict(op.metadata).get("w4a8_msd_down")
    if not isinstance(raw, Mapping):
        _error("source_configuration_missing")
    config = Config(
        _require_int(raw.get("m"), "m", 1),
        _require_int(raw.get("n"), "n", 1),
        _require_int(raw.get("k"), "k", 1),
        tuple(_require_int(value, "expert_rows")
              for value in _sequence(raw.get("expert_rows"), "expert_rows")),
        raw.get("routing_basis"),
    )
    if config.routing_basis not in SUPPORTED_EXPERT_ROW_BASES:
        _error("rejects_probability_moment_routing")
    _validate_geometry(config)
    if tuple(op.local_shape) != (config.m, config.n, config.k):
        _error("source_configuration_disagrees_with_local_shape")
    return config


def _phase_work(config, *, fixpipe=False, resource_roles=None):
    """Global down ledger; ownership-dependent duplicates resolve at prepare."""
    schedule = down_schedule(
        config.m, config.n, config.k, config.expert_rows,
        aic_owners=1, aiv_owners=2, fixpipe=fixpipe)
    groups = {"pre": schedule.pre, "matrix": schedule.matrix,
              "post_dequant": schedule.post_dequant, "output": schedule.output}
    return {name: _role_work(total_work(group), name, resource_roles)
            for name, group in groups.items()}


def work_from_config(config, *, fixpipe=False, resource_roles=None, geometry=None):
    if geometry is None:
        phases = _phase_work(config, fixpipe=fixpipe, resource_roles=resource_roles)
    else:
        aic, aiv = _geometry_owners(geometry)
        schedule = down_schedule(
            config.m, config.n, config.k, config.expert_rows,
            aic_owners=aic, aiv_owners=aiv, fixpipe=fixpipe)
        phases = {
            name: _role_work(total_work(group), name, resource_roles)
            for name, group in (
                ("pre", schedule.pre), ("matrix", schedule.matrix),
                ("post_dequant", schedule.post_dequant), ("output", schedule.output))
        }
    total = Work()
    for work in phases.values():
        total += work
    return total


def ledger(config):
    """Reference arithmetic; actual owner work is rebound before execution."""
    m, n, k = config.m, config.n, config.k
    active = config.active
    reference = _phase_work(config)
    total = work_from_config(config)
    base_m = 64 if m // len(config.expert_rows) < 32 else 128
    m_panels = sum((2 * rows + base_m - 1) // base_m for rows in config.expert_rows)
    return {
        "reference_inventory": {"aic": 1, "aiv": 2,
                                "scope": "unbound_reference; rebind_actual_device_at_prepare"},
        "pre": {
            "read_bytes": reference["pre"].resources["mte2"], "write_bytes": m * k,
            "vector_ops": reference["pre"].resources["vector"],
            "metadata_per_launched_aiv": {"read_bytes": 8 * len(config.expert_rows),
                                          "vector_ops": 2 * len(config.expert_rows)},
            "expression": "high cast+mul+cast (3MK) + halfword-and (MK/2) "
                          "+ low cast+add+cast (3MK)",
        },
        "matrix": {
            "flops_int4": 4 * m * n * k,
            "read_bytes": reference["matrix"].resources["mte2"],
            "write_bytes": 4 * m * n,
            "publication_role": "fixpipe; additional service only when declared",
            "column_scale_ops": 2 * m * n,
            "active_weight_banks": active,
            "weight_bytes": 0.5 * n * k * m_panels,
            "encoded_scale_bytes": 8 * n * m_panels,
            "expression": "double-M INT4 product 4MNK; packed weights and "
                          "uint64 column scales per active M panel",
        },
        "post_dequant": {
            "read_bytes": reference["post_dequant"].resources["mte2"],
            "assist_bytes_per_aiv_copy": 4 * n,
            "assist_copies": "each host-selected AIV copies every AIC tile; "
                             "persistent vecCount rotates UB row chunks",
            "vector_ops": 6 * m * n,
            "expression": "2MN cast + MN high*16 + MN low add "
                          "+ MN assist + MN token scale",
        },
        "output": {
            "write_bytes": 2 * m * n, "vector_ops": m * n,
            "expression": "MN BF16 cast + BF16 store",
        },
        "totals": {
            "flops_int4": 4 * m * n * k,
            "hbm_bytes": total.resources["hbm"],
            "mte2_bytes": total.resources["mte2"],
            "mte3_bytes": total.resources["mte3"],
            "vector_ops": total.resources["vector"],
        },
        "experts": {
            "scan_count": len(config.expert_rows),
            "active_groups": active,
            "zero_bank_offsets_int4_elements": [
                index * n * k for index, rows in enumerate(config.expert_rows)
                if rows == 0],
            "expression": "all group counts are scanned; zero banks retain "
                          "their offsets and issue no tile",
        },
    }


def _config_dict(config):
    return {
        "m": config.m, "n": config.n, "k": config.k,
        "expert_rows": list(config.expert_rows),
        "active_groups": config.active,
        "expert_scan_count": len(config.expert_rows),
        "zero_expert_indices": [index for index, rows in enumerate(config.expert_rows)
                                if rows == 0],
        "zero_bank_offsets_int4_elements": [index * config.n * config.k
                                            for index, rows in enumerate(config.expert_rows)
                                            if rows == 0],
        "routing_basis": config.routing_basis,
        "weight_format": "int4", "activation_format": "int8",
        "weight_layout": "NZ", "input_prepacking": "in_place_ND",
        "output_format": "bf16", "output_dtype": "bf16",
        "workspace_format": "fp16", "swiglu": False,
        "output_quantization": False,
        "column_scale": "uint64_bitpattern_fp32_8B_per_channel",
        "assist_scale": "float32_4B_per_channel",
        "assist_role": "msd_low_digit_shift_compensation_not_linear_bias",
        "token_scale": "float32_4B_per_row",
        "source": SOURCE,
    }


def describe(config):
    return {
        "profile": PROFILE, "config": _config_dict(config),
        "ledger": ledger(config), "limitations": list(LIMITATIONS),
    }


def _base_program(config, *, fixpipe=False, resource_roles=None):
    def program(context):
        raise ModelingError("w4a8_down_execution_requires_device_geometry")
    return program


def _owner_program(schedule):
    order = (("pre", schedule.pre, schedule.aiv_owners, ()),
             ("matrix", schedule.matrix, schedule.aic_owners, "pre"),
             ("post_dequant", schedule.post_dequant, schedule.aiv_owners, "matrix"),
             ("output", schedule.output, schedule.aiv_owners, "post_dequant"))

    def program(context):
        stages, barrier = [], {}
        for name, group, owners_count, after in order:
            predecessors = barrier.get(after, ()) if after else ()
            if name == "matrix":
                stages.append(Stage("matrix", after=predecessors,
                                    program=matrix_program(context, schedule)))
                barrier[name] = ("matrix",)
                continue
            names = []
            for record in group:
                work = _role_work(record.work, name, context.resource_units)
                bound = context.for_phase(
                    work, kernel=f"grouped_matmul_antiquant_a8w4_msd_{name}",
                    dtype="int4" if name == "matrix" else None)
                # HBM stays one shared bandwidth domain.  Cube, vector,
                # read/store pipes and a declared Fixpipe follow their owner.
                roles = set(work.resources) - {"hbm"}
                if work.matrix:
                    roles.add("cube")
                stage_name = f"{name}:{record.owner}"
                stages.append(Stage(
                    stage_name, work=bound.work, after=predecessors,
                    placement=context.owned(record.owner, owners_count, roles)))
                names.append(stage_name)
            barrier[name] = tuple(names)
        return KernelProgram(tuple(stages))

    return program


def build(op, geometry=None):
    if op.recipe != RECIPE or op.source_boundary != "kernel":
        _error("build_requires_selected_kernel_source")
    if op.dtype != "int4" or dict(op.metadata).get("output_format") not in {
            "bf16", "bfloat16"}:
        _error("build_requires_bf16_output")
    config = _config_from_source(op)
    if geometry is None:
        return OperatorPlan(work_from_config(config), _base_program(config))
    fixpipe = "fixpipe" in geometry.resource_units
    aic, aiv = _geometry_owners(geometry)
    schedule = down_schedule(
        config.m, config.n, config.k, config.expert_rows,
        aic_owners=aic, aiv_owners=aiv, fixpipe=fixpipe)
    # Bind each phase with the same read-pipe role used by execution.
    total = Work()
    for name, group in (("pre", schedule.pre), ("matrix", schedule.matrix),
                        ("post_dequant", schedule.post_dequant), ("output", schedule.output)):
        work = matrix_work(schedule, geometry.resource_units) if name == "matrix" else total_work(group)
        total += _role_work(work, name, geometry.resource_units)
    return OperatorPlan(total, _owner_program(schedule))


def lower_op(op):
    """Replace one semantic down GEMM, retaining its separate input quantizer."""
    config = config_from_op(op)
    metadata = dict(op.metadata)
    if metadata.get("activation_producer") != "explicit_quantization":
        _error("requires_existing_activation_quantization_producer")
    source = replace(
        op, dtype="int4", recipe=RECIPE, source_boundary="kernel",
        metadata=op.metadata + (
            ("semantic_recipe", op.recipe), ("compute_format", "int4"),
            ("w4a8_msd_down", _config_dict(config)),
            ("w4a8_msd_down_ledger", ledger(config)),
            ("w4a8_msd_down_source", SOURCE),
            ("w4a8_msd_down_limitations", LIMITATIONS),
            ("w4a8_msd_down_phases", PHASES),
            ("source_boundary_kind", "kernel"),
        ),
        notes=op.notes + (
            "source: ordinary A8W4 MSD grouped-down (INT8 x INT4, FP16 workspace, BF16 output)",
            "input per-row INT8 quantization stays a separate producer",
            "coarse sequential phases; native chunk overlap and binary identity not claimed",
        ),
    )
    return work_from_config(config).apply(source)


def recipes():
    return {RECIPE: PlannedRecipe(build)}


__all__ = [
    "LIMITATIONS", "PHASES", "PROFILE", "RECIPE", "SOURCE",
    "SUPPORTED_RECIPES", "build", "config_from_op", "describe", "ledger",
    "lower_op", "recipes", "work_from_config",
]
