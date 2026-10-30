"""Complete-call program for the arch35 SituMxQuant BF16 -> FP8 MX kernel.

[Role] Price the inspected ``SituMxQuant`` host grid and per-owner device
  program on the common Engine, without HBM materialization of SiTU output.
[Invariants] BF16 input rounds to BF16 before exponent extraction; E8M0 scale
  work is per 32-element group; owner count and row/column partitions follow
  the source tiling; all declared GM payloads are conserved. Source identity is
  an explicit selection, never a measured-duration or binary inference.
[Timeouts & Locks] O(active owners) geometry plus lazy O(owners*slots) tiles;
  global totals are closed forms, never a per-element or per-tile enumeration.
[Side-Effect Policy] Returns immutable plans/programs; no devices, captures or I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .mechanism import OperatorPlan, Work
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


PROFILE = "situ_mx_quant_arch35"
RECIPE = PROFILE
ROLE = "situ_quant"
SOURCE_REVISION = "175ef4d49c2072f9233619636f0309f4f88b4e2c"
SOURCE_STATE = "added_in_index"
SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": SOURCE_REVISION,
    "source_state": SOURCE_STATE,
    "scope": "independent_source_semantics; not_capture_binary_identity",
    "files": (
        {"path": "csrc/moe/situ_mx_quant/op_host/arch35/situ_mx_quant_tiling_arch35.cpp",
         "sha256": "a2c44c27ef9f4230b7bb558c31976b920bff12f278cdb4f7909d97486e2c6582",
         "lines": "199-245,255-283"},
        {"path": "csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_axis_last.h",
         "sha256": "2a9bd1b0058d5c5727dc313c10f1e6920ea900390c64402d5263c8a602470284",
         "lines": "126-183,186-229,232-278"},
        {"path": "csrc/moe/situ_mx_quant/op_kernel/arch35/situ_mx_quant_common.h",
         "sha256": "bce5998513209f9a47579913321f6c924747829fbd5e42aad2f3a57c1284f858",
         "lines": "91-123,129-182,188-240,247-508"},
        {"path": "csrc/moe/situ_mx_quant/situ_mx_quant_torch_adpt.h",
         "sha256": "3564aba053a9d37ec4960effb538358c62d7580858d8d5fbf7c71b1f93ea95fb",
         "lines": "22-68"},
    ),
}
KERNEL_BLOCK = 256
SCALE_GROUP = 32
SCALE_ALIGN = 64
UB_RESERVED = 32 + 128
BASE_BLOCK_BYTES = 3120
# Source lane groups.  A vector lane operation is one scalar expression:
# compare/select/mul/add/cast count once; Exp and Div each count once.  The
# source FMA expressions are counted as 2 scalar operations (multiply + add):
# 29 single-op expressions + 3 FMA expressions, and optionally 19 + 3.
# Scalar register broadcasts and queue DataCopy are not lanes.
SITU_BASE_LANES = 29 + 3
SITU_LINEAR_BETA_LANES = 19 + 3
SCALE_LANES = 13
# Per 256 conversion lanes: 256 multiply + 512 casts + 768 packing adds.
QUANT_LANES = 6
MAX_UB_BUFFERS = (
    # name, size expression: factor * multiplier + alignment
    ("inQuex", 2 * 512 * 2, 1),
    ("outQuey", 2 * 256, 1),
    ("outQueScale", 8, 64),
    ("situ", 256 * 2, 1),
    ("maxExp", 8 * 2, 32),
    ("halfScale", 8 * 2, 32),
)


def _ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align(value: int, multiple: int) -> int:
    return _ceil(value, multiple) * multiple


def _positive_int(value, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{reason}:{value!r}")
    return value


@dataclass(frozen=True)
class Config:
    rows: int
    width: int
    output_dtype: str
    beta: float
    linear_beta: float | None
    activate_left: bool = True
    axis: int = -1
    activation_id: str = ""
    quantize_id: str = ""

    def __post_init__(self):
        _positive_int(self.rows, "situ_mx_quant_rows_must_be_positive")
        _positive_int(self.width, "situ_mx_quant_width_must_be_positive")
        if self.output_dtype not in {"fp8_e4m3", "fp8_e5m2"}:
            raise ModelingError("situ_mx_quant_requires_fp8_e4m3_or_e5m2_output")
        if not isinstance(self.beta, (int, float)) or not math.isfinite(self.beta) or self.beta <= 0:
            raise ModelingError("situ_mx_quant_beta_must_be_positive_finite")
        if self.linear_beta is not None:
            if (not isinstance(self.linear_beta, (int, float))
                    or not math.isfinite(self.linear_beta) or self.linear_beta <= 0):
                raise ModelingError("situ_mx_quant_linear_beta_must_be_positive_finite_or_absent")
        if not isinstance(self.activate_left, bool):
            raise ModelingError("situ_mx_quant_activate_left_must_be_boolean")
        if self.axis != -1:
            raise ModelingError("situ_mx_quant_only_supports_axis_minus_one")

    @property
    def scale_shape(self):
        return (self.rows, _ceil(self.width, SCALE_ALIGN), 2)

    @property
    def has_linear_beta(self):
        return self.linear_beta is not None

    def tensor_view(self):
        return {
            "rows": self.rows, "width": self.width, "input_dtype": "bf16",
            "output_dtype": self.output_dtype, "scale_dtype": "fp8_e8m0",
            "scale_shape": self.scale_shape, "beta": self.beta,
            "linear_beta": self.linear_beta, "activate_left": self.activate_left,
            "axis": self.axis,
        }

    def to_dict(self):
        return self.tensor_view()


def config_from_view(view, *, activation_id="", quantize_id=""):
    """Validate the arithmetic/tensor domain before any program is selected."""
    required = {"rows", "width", "input_dtype", "output_dtype", "scale_dtype",
                "scale_shape", "beta", "linear_beta", "activate_left", "axis"}
    if not isinstance(view, Mapping) or not required <= set(view):
        raise ModelingError("situ_mx_quant_requires_complete_declared_tensor_view")
    if view["input_dtype"] != "bf16":
        raise ModelingError("situ_mx_quant_requires_bf16_input")
    if view["scale_dtype"] != "fp8_e8m0":
        raise ModelingError("situ_mx_quant_requires_e8m0_scales")
    config = Config(
        rows=view["rows"], width=view["width"], output_dtype=view["output_dtype"],
        beta=view["beta"], linear_beta=view["linear_beta"],
        activate_left=view["activate_left"], axis=view["axis"],
        activation_id=activation_id, quantize_id=quantize_id)
    if tuple(view["scale_shape"]) != config.scale_shape:
        raise ModelingError("situ_mx_quant_scale_shape_mismatch")
    return config


def config_from_group(declaration, members, outputs):
    """Build the complete-call config from one explicitly selected group.

    The group must own the SiTU activation and its output quantizer only.  An
    externally visible activation BF16 result would force a real HBM
    materialization, so it is rejected rather than silently dropped.
    """
    role = declaration.get("role")
    if role != ROLE:
        raise ModelingError(f"situ_mx_quant_role_mismatch:{role}")
    if len(members) != 2:
        raise ModelingError("situ_mx_quant_requires_exactly_two_semantic_members")
    activations = [op for op in members if op.family == "situ"]
    quantizers = [op for op in members if op.family == "quantize"]
    if len(activations) != 1 or len(quantizers) != 1:
        raise ModelingError("situ_mx_quant_requires_situ_and_dynamic_quantize_members")
    activation, quantize = activations[0], quantizers[0]
    if members[-1] is not quantize:
        raise ModelingError("situ_mx_quant_quantizer_must_be_the_terminal_member")
    if activation.op_id not in quantize.deps:
        raise ModelingError("situ_mx_quant_quantizer_does_not_consume_activation")
    if quantize.recipe != "dynamic_quantize":
        raise ModelingError("situ_mx_quant_requires_dynamic_quantize_recipe")
    shape = (int(activation.local_shape[0]), int(activation.local_shape[-1]))
    if tuple(activation.local_shape) != shape or tuple(quantize.local_shape) != shape:
        raise ModelingError("situ_mx_quant_member_shape_mismatch")
    config = config_from_view(
        declaration["tensor_view"], activation_id=activation.op_id,
        quantize_id=quantize.op_id)
    if (config.rows, config.width) != shape:
        raise ModelingError("situ_mx_quant_tensor_view_member_shape_mismatch")
    activation_meta = dict(activation.metadata)
    if (activation.dtype != "bf16"
            or activation_meta.get("activation_linear_beta") != (None if config.linear_beta is None
                                                                 else config.linear_beta)
            or float(activation_meta.get("activation_beta", -1)) != float(config.beta)):
        raise ModelingError("situ_mx_quant_member_arithmetic_domain_mismatch")
    quant_meta = dict(quantize.metadata)
    if (quant_meta.get("output_format") != config.output_dtype
            or quant_meta.get("activation_scale_format") != "fp8_e8m0"
            or tuple(quant_meta.get("activation_scale_shape", ())) != config.scale_shape):
        raise ModelingError("situ_mx_quant_member_quantization_domain_mismatch")
    exposed = set(outputs)
    if activation.op_id in exposed:
        raise ModelingError("situ_mx_quant_external_activation_intermediate_requires_materialization")
    if exposed != {quantize.op_id}:
        raise ModelingError("situ_mx_quant_requires_only_terminal_quantizer_outputs")
    return config


@dataclass(frozen=True)
class Tiling:
    dim_n_blocks: int
    factor_dim0: int
    factor_dim1: int
    m_cores: int
    n_cores: int
    used_cores: int
    available_ub: int
    buffer_bytes: int
    buffers: tuple[tuple[str, int], ...]

    @property
    def factor_size(self):
        return self.factor_dim0 * self.factor_dim1

    def buffer(self, name):
        return dict(self.buffers)[name]


def tiling(config: Config, *, ub_bytes: int, aiv_count: int) -> Tiling:
    """Mirror the inspected host grid and fixed source UB partition."""
    ub = _positive_int(int(ub_bytes), "situ_mx_quant_ub_must_be_positive")
    aiv = _positive_int(int(aiv_count), "situ_mx_quant_aiv_must_be_positive")
    available = ub - UB_RESERVED
    if available < BASE_BLOCK_BYTES:
        raise ModelingError("situ_mx_quant_requires_ub_capacity_for_one_source_basic_block")
    dim_n_blocks = _ceil(config.width, KERNEL_BLOCK)
    basic_blocks = available // BASE_BLOCK_BYTES
    if basic_blocks >= dim_n_blocks:
        factor_dim1 = dim_n_blocks
        factor_dim0 = basic_blocks // dim_n_blocks
    else:
        factor_dim1 = max(1, basic_blocks)
        factor_dim0 = 1
    factor_size = factor_dim0 * factor_dim1
    buffers, total = [], 0
    for name, multiplier, alignment in MAX_UB_BUFFERS:
        size = factor_size * multiplier
        if alignment > 1:
            size = _align(size, alignment)
        if name == "outQueScale":
            # The source queue has two independently aligned scale slots.
            size *= 2
        buffers.append((name, size))
        total += size
    if total + UB_RESERVED > ub:
        raise ModelingError("situ_mx_quant_source_ub_partition_exceeds_device")
    m_cores = min(config.rows, aiv)
    n_cores = 1
    if m_cores < aiv:
        n_cores = min(aiv // m_cores, dim_n_blocks)
    used = m_cores * n_cores
    return Tiling(dim_n_blocks, factor_dim0, factor_dim1, m_cores, n_cores, used,
                  available, total, tuple(buffers))


@dataclass(frozen=True)
class Owner:
    index: int
    row_start: int
    rows: int
    col_start: int
    col_blocks: int
    n_loops: int
    tail_n: int


def owner_grid(config: Config, plan: Tiling):
    """The source block grid: mIdx = block/nCore, nIdx = block%nCore."""
    owners = []
    m_head = config.rows % plan.m_cores
    m_base = config.rows // plan.m_cores
    n_head = plan.dim_n_blocks % plan.n_cores
    n_base = plan.dim_n_blocks // plan.n_cores
    for index in range(plan.used_cores):
        m_idx, n_idx = divmod(index, plan.n_cores)
        row_start = (m_idx * (m_base + 1) if m_idx < m_head
                     else m_head * (m_base + 1) + (m_idx - m_head) * m_base)
        rows = m_base + 1 if m_idx < m_head else m_base
        n_front = n_base + 1
        col_start = (n_idx * n_front if n_idx < n_head
                     else n_head * n_front + (n_idx - n_head) * n_base)
        col_blocks = n_front if n_idx < n_head else n_base
        n_loops = _ceil(col_blocks, plan.factor_dim1)
        if n_idx < plan.n_cores - 1:
            tail_n = col_blocks * KERNEL_BLOCK - (n_loops - 1) * plan.factor_dim1 * KERNEL_BLOCK
        else:
            tail_n = config.width - col_start * KERNEL_BLOCK - (
                n_loops - 1) * plan.factor_dim1 * KERNEL_BLOCK
        if rows < 1 or col_blocks < 1 or tail_n < 1:
            raise ModelingError("situ_mx_quant_source_owner_partition_invalid")
        owners.append(Owner(index, row_start, rows, col_start, col_blocks, n_loops, tail_n))
    return tuple(owners)


def owner_rows(config: Config, plan: Tiling):
    """Observable owner row spans, including unequal head/tail rows."""
    return tuple(owner.rows for owner in owner_grid(config, plan))


@dataclass(frozen=True)
class OwnerWork:
    owner: Owner
    read_bytes: int
    write_bytes: int
    vector_ops: float
    quantize_lanes: int
    scale_count: int


def _loop_sum(owner: Owner, plan: Tiling, per_column):
    """Closed-form sum of a per-column quantity over this owner's N loops."""
    full_n = plan.factor_dim1 * KERNEL_BLOCK
    return ((owner.n_loops - 1) * per_column(full_n)
            + per_column(owner.tail_n))


def _quantize_lanes(rows: int, columns: int, plan: Tiling) -> int:
    """Unmasked 256-wide source conversion rounds up each owner row block."""
    align = _align(columns, SCALE_ALIGN)
    whole, tail = divmod(rows, plan.factor_dim0)
    lanes = whole * _ceil(plan.factor_dim0 * align, KERNEL_BLOCK) * KERNEL_BLOCK
    if tail:
        lanes += _ceil(tail * align, KERNEL_BLOCK) * KERNEL_BLOCK
    return lanes


def owner_work(config: Config, plan: Tiling, owner: Owner) -> OwnerWork:
    read = _loop_sum(owner, plan, lambda cols: 2 * owner.rows * cols * dtype_bytes("bf16"))
    fp8 = _loop_sum(owner, plan, lambda cols: owner.rows * cols)
    scale = _loop_sum(owner, plan, lambda cols: owner.rows * 2 * _ceil(cols, SCALE_ALIGN))
    situ = _loop_sum(owner, plan, lambda cols: owner.rows * cols) * (
        SITU_BASE_LANES + (SITU_LINEAR_BETA_LANES if config.has_linear_beta else 0))
    maxexp_lanes = _loop_sum(owner, plan, lambda cols: owner.rows * _align(cols, SCALE_ALIGN))
    scale_count = maxexp_lanes // SCALE_GROUP
    quant_lanes = _loop_sum(owner, plan, lambda cols: _quantize_lanes(owner.rows, cols, plan))
    # Each 32-element exponent group: 32 AND + 16 pair-max + 15 reduce-max.
    vector = (situ + 2 * maxexp_lanes - scale_count
              + scale_count * SCALE_LANES + quant_lanes * QUANT_LANES)
    return OwnerWork(owner, read, fp8 + scale, float(vector), quant_lanes, scale_count)


def _owner_work(config, plan):
    return tuple(owner_work(config, plan, owner) for owner in owner_grid(config, plan))


def _work(config: Config, plan: Tiling, read: str):
    total = Work()
    for item in _owner_work(config, plan):
        total += Work({}, {
            "hbm": item.read_bytes + item.write_bytes, read: item.read_bytes,
            "mte3": item.write_bytes, "vector": item.vector_ops,
        })
    return total


def _known(config, view=None):
    if not isinstance(config, Config):
        raise ModelingError("situ_mx_quant_plan_requires_bound_source_config")
    if view is not None and (config.rows, config.width) != tuple(view):
        raise ModelingError("situ_mx_quant_plan_geometry_mismatch")
    return config


def build(op, geometry):
    """Bind the complete source call to actual device UB and AIV geometry."""
    metadata = dict(op.metadata)
    raw = metadata.get("fusion_situ_config")
    if raw is None:
        raw = metadata.get("fusion_tensor_view")
    config = config_from_view(raw) if isinstance(raw, Mapping) else None
    if config is None:
        raise ModelingError("situ_mx_quant_requires_bound_source_config")
    _known(config, op.local_shape)
    if op.dtype != "bf16":
        raise ModelingError("situ_mx_quant_plan_requires_bf16_operator")
    ub = geometry.buffers.get("ub")
    if ub is None:
        raise ModelingError("situ_mx_quant_plan_requires_ub_capacity")
    aiv = geometry.resource_units.get("vector", geometry.compute_units)
    plan = tiling(config, ub_bytes=ub, aiv_count=aiv)
    read = read_role(geometry.resource_units, aiv=0)
    owners = owner_grid(config, plan)
    totals = _owner_work(config, plan)
    total_work = _work(config, plan, read)
    buffer_specs = plan.buffers

    def owner_caps(context, placement):
        caps = {}
        for role in {context.physical_role(r) for r in ("vector", read, "mte3")}:
            name = context.resource_names.get(role)
            if name is None:
                continue
            units = context.resource_units.get(role, context.core_count)
            caps[name] = context.rates[name] * len(placement[name]) / units
        return caps

    def owner_placement(context, owner_index):
        vector_units = context.resource_units.get("vector", context.core_count)
        placements = {}
        vector_name = context.resource_names.get("vector")
        if vector_name is not None:
            placements[vector_name] = (owner_index % vector_units,)
        for role in {context.physical_role(r) for r in (read, "mte3")}:
            name = context.resource_names.get(role)
            if name is None:
                continue
            units = context.resource_units.get(role, context.core_count)
            placements[name] = unit_group(owner_index, vector_units, units)
        return placements

    def owner_tile(context, owner, caps, placement):
        def tile(index):
            m_group, n_loop = divmod(index, owner.n_loops)
            rows = min(plan.factor_dim0, owner.rows - m_group * plan.factor_dim0)
            tail_loop = n_loop == owner.n_loops - 1
            cols = owner.tail_n if tail_loop else plan.factor_dim1 * KERNEL_BLOCK
            align = _align(cols, SCALE_ALIGN) if tail_loop else cols
            read_bytes = 2 * rows * cols * dtype_bytes("bf16")
            fp8_bytes = rows * cols
            scale_bytes = rows * 2 * _ceil(cols, SCALE_ALIGN)
            situ_lanes = rows * cols * (
                SITU_BASE_LANES + (SITU_LINEAR_BETA_LANES if config.has_linear_beta else 0))
            maxexp_lanes = rows * align
            scale_count = maxexp_lanes // SCALE_GROUP
            quant_lanes = _ceil(rows * align, KERNEL_BLOCK) * KERNEL_BLOCK
            stages = [
                Stage("load", context.bind(Work({}, {"hbm": read_bytes, read: read_bytes})),
                      caps=caps, placement=placement),
                Stage("situ", context.bind(Work({}, {"vector": situ_lanes})),
                      after=("load",), previous=("quantize",),
                      caps=caps, placement=placement),
                Stage("maxexp", context.bind(Work({}, {"vector": 2 * maxexp_lanes
                                                       - scale_count})),
                      after=("situ",), caps=caps, placement=placement),
                Stage("scale", context.bind(Work({}, {"vector": scale_count * SCALE_LANES})),
                      after=("maxexp",), caps=caps, placement=placement),
                Stage("quantize", context.bind(Work({}, {"vector": quant_lanes * QUANT_LANES})),
                      after=("scale",), caps=caps, placement=placement),
                Stage("store", context.bind(Work({}, {
                    "hbm": fp8_bytes + scale_bytes, "mte3": fp8_bytes + scale_bytes})),
                    after=("quantize",), caps=caps, placement=placement),
            ]
            return tuple(stages)
        return tile

    def owner_program(context, item):
        owner = item.owner
        placement = owner_placement(context, owner.index)
        caps = owner_caps(context, placement)
        vector_units = context.resource_units.get("vector", context.core_count)
        ub_units = context.pool_units.get("ub", vector_units)
        ub_owners = unit_group(owner.index, vector_units, ub_units)
        allocations = tuple(
            context.allocate(f"{name}{owner.index}", "ub", size, ub_owners)
            for name, size in buffer_specs)
        count = owner.n_loops * _ceil(owner.rows, plan.factor_dim0)
        pipeline = PipelineProgram(count, 2,
                                   owner_tile(context, owner, caps, placement))
        return KernelProgram((
            Stage("ub_acquire", acquire=allocations),
            Stage("source_tiles", after=("ub_acquire",), program=pipeline),
            Stage("ub_release", after=("source_tiles",), free=allocations),
        ))

    def program(context):
        return ParallelProgram(tuple(owner_program(context, item) for item in totals))

    return OperatorPlan(total_work, program)


def tiling_for(config: Config, geometry):
    """Public planning view used by examples/tests without an Engine."""
    ub = geometry.buffers.get("ub")
    if ub is None:
        raise ModelingError("situ_mx_quant_plan_requires_ub_capacity")
    aiv = geometry.resource_units.get("vector", geometry.compute_units)
    return tiling(config, ub_bytes=ub, aiv_count=aiv)


def make_op(config: Config, *, op_id="situ_mx_quant", deps=(), stage=0, layer=-1,
            phase="prefill", requests=()):
    """Construct one explicit isolated complete-call boundary."""
    view = config.tensor_view()
    metadata = (
        ("fusion_situ_config", view),
        ("fusion_tensor_view", view),
        ("fusion_program_profile", PROFILE),
        ("situ_source", SOURCE),
    )
    return OpSpec(
        op_id=op_id, family="situ_mx_quant", phase=phase, stage=stage, layer=layer,
        requests=tuple(requests), semantic_shape=(config.rows, 2 * config.width),
        local_shape=(config.rows, config.width), padded_shape=(config.rows, config.width),
        launch_shape=(config.rows, config.width), deps=tuple(deps), recipe=RECIPE,
        coverage="source_complete_call_bf16_to_fp8_mx", dtype="bf16",
        source_boundary="kernel", returns=("activation_scales",), metadata=metadata,
    )


def recipe_book():
    """Explicit recipe book for the isolated complete-call example."""
    from .mechanism import PlannedRecipe
    from .program import RecipeBook
    return RecipeBook({RECIPE: PlannedRecipe(build)})


def fusion_metadata(config, declaration, members, outputs, selection):
    """Source-bound boundary record retained by native fusion export."""
    return {
        "fusion_group_key": declaration["key"],
        "fusion_semantic_ids": [member.op_id for member in members],
        "fusion_semantic_recipes": [
            {"op_id": member.op_id, "family": member.family, "recipe": member.recipe,
             "local_shape": list(member.local_shape)} for member in members],
        "fusion_external_outputs": list(outputs),
        "fusion_role": declaration["role"],
        "fusion_tensor_view": declaration["tensor_view"],
        "fusion_program_profile": PROFILE,
        "fusion_situ_config": config.to_dict(),
        "fusion_boundary": {
            "implementation": selection.name,
            "native": declaration.get("implementation"),
            "source": declaration.get("source"),
            "program_profile": PROFILE,
            "traffic_profile": None,
            "evidence": {
                "semantic_boundary": declaration.get("source"),
                "source_identity": SOURCE,
                "capture_binary_identity": "not_established",
            },
            "submitted_kernels": 1, "semantic_phase_count": len(members),
            "traffic": {
                "profile": None,
                "eliminated_reads": 0.0,
                "eliminated_writes": 0.0,
                "reused_edges": [],
                "limitations": ["complete_source_program_replaces_member_traffic"],
            },
            "timing_basis": "source_host_grid_and_per_owner_device_program",
            "fidelity": {
                "kind": "complete_source_call_program",
                "host_grid": "arch35_m_n_core_partition_and_256_position_blocks",
                "ub_partition": "factorDim0_factorDim1_fixed_six_buffers_per_owner",
                "pipeline_overlap": "two_queue_slots_approximation_not_compiler_timeline",
                "native_dispatch_reconstructed": False,
                "instruction_throughput_calibrated": False,
                "limitations": [
                    "source_lane_ops_are_semantic_counts_not_ISA_latencies",
                    "scalar_issue_barrier_and_instruction_startup_not_calibrated",
                    "observed_65us_remains_unexplained_by_this_source_resource_program",
                    "source_selection_does_not_identify_capture_binary_or_executed_tiling",
                ],
            },
        },
        **{key: value for key, value in dict(members[-1].metadata).items()
           if key in {"structure_block", "structure_role", "graph_captured", "graph_segment"}},
    }
