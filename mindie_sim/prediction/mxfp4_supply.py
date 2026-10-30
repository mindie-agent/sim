"""Explicit MXFP4-to-FP8 weight supply from the retained grouped SiTU source.

[Role] Execute the AIV weight-producer segment on the common resource Engine.
[Invariants] Integer group rows select banks; tensor capacity never means work.
  Each rolling source output tile reads its own FP4 bank slice, widens in UB,
  then publishes FP8 to L1. L1 writes are MTE3 work, not HBM writes. This is a
  subprogram, not a complete grouped matrix/SiTU call or a vendor timing fit.
[Timeouts & Locks] Lazy panels, two resident L1 slots; O(experts + owners).
[Side-Effect Policy] Build immutable programs; no capture, I/O or device access.
"""
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from .mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .ownership import read_role
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage, RecipeBook

RECIPE = "mxfp4_grouped_weight_supply"
SOURCE = {
    "archive": "gmmsituquant-latest-0916/gmmsituquant-latest-overlay.zip",
    "archive_sha256": "fd5ded62862cabd7b686feceadcd8f39b692bdbb1989eb93c72eea29f4fc11da",
    "base_revision": "75f52c1d7df8df8548d854a01b64935727a70e88",
    "scope": "independent_overlay; capture_binary_identity_unknown",
    "files": (
        "op_host/grouped_matmul_situ_quant_tiling.cpp:113-146",
        "op_kernel/gmsq_vcv_controller.h:128-171,214-333",
        "op_kernel/vendor/gmsq2/gmmsq_weight_quant_vec_compute.h:245-399",
        "op_kernel/vendor/gmsq2/gmmsq_quant_mx_vf.h:340-383",
    ),
}
UB_BYTES = 64 * 1024 + 128 * 1024 + 3 * 64 * 64 * 2 + 64 * 64 + 3 * 1024 * 2
L1_WEIGHT_BYTES = 2 * 256 * 256


@dataclass(frozen=True)
class Config:
    k: int
    output_width: int
    group_rows: tuple[int, ...]
    allocated_rows: int
    group_list_type: int = 1

    def __post_init__(self):
        if (type(self.k) is not int or self.k <= 0 or self.k % 64
                or type(self.output_width) is not int or self.output_width <= 0
                or self.output_width % 64):
            raise ModelingError("mxfp4_supply_requires_source_k64_output64_geometry")
        if (not self.group_rows or any(type(x) is not int or x < 0 for x in self.group_rows)
                or type(self.allocated_rows) is not int
                or self.allocated_rows < sum(self.group_rows)):
            raise ModelingError("mxfp4_supply_requires_valid_integer_rows_within_capacity")
        # The dual-role source branch has different AIV ownership and relays.
        # This first producer segment covers its actual interleaved Decode path.
        if max(self.group_rows) >= 34:
            raise ModelingError("mxfp4_supply_dual_role_or_multiple_m_tiles_not_implemented")
        if self.group_list_type not in (0, 1):
            raise ModelingError("mxfp4_supply_group_list_type_must_be_count_or_cumulative")

    @property
    def half_tile(self):
        return 128 if self.output_width % 128 == 0 else 64

    @property
    def k_tile(self):
        return 256 if self.half_tile == 128 else 512

    @property
    def active(self):
        return sum(x > 0 for x in self.group_rows)


def tiles(config, owner, aic_count):
    """Rolling source grid across active experts; zero banks consume no tile."""
    start = 0
    ntiles = config.output_width // config.half_tile
    for expert, rows in enumerate(config.group_rows):
        if not rows:
            continue
        for tile in range((owner - start) % aic_count, ntiles, aic_count):
            yield expert, rows, tile * config.half_tile
        start += ntiles


def panels(config, owner, aic_count):
    for expert, rows, column in tiles(config, owner, aic_count):
        for offset in range(0, config.k, config.k_tile):
            yield expert, rows, column, offset


def half_work(config, k_offset, sub, read):
    k = min(config.k_tile // 2, max(0, config.k - k_offset - sub * config.k_tile // 2))
    elements = k * 2 * config.half_tile
    # Pinned widening: five 256-lane bit operations and one two-output
    # 256-lane interleave per 512 FP8 values. Register loads/stores and scalar
    # broadcasts are not invented vector arithmetic or extra HBM transfers.
    return Work(resources={"hbm": elements / 2, read: elements / 2,
                           "vector": elements * 3.5, "mte3": elements})


def ledger(config, aic_count):
    count = config.active * config.output_width // config.half_tile
    elements = config.active * 2 * config.output_width * config.k
    # Both MaxGroupL1M and Process read the group-list on every AIC/AIV.
    loads_per_scan = len(config.group_rows) if config.group_list_type == 1 else 2 * len(config.group_rows) - 1
    return {"valid_rows": sum(config.group_rows), "allocated_rows": config.allocated_rows,
            "active_banks": config.active, "declared_banks": len(config.group_rows),
            "source_output_tiles": count, "packed_weight_read_bytes": elements // 2,
            "expanded_l1_write_bytes": elements, "widen_vector_lanes": elements * 3.5,
            "scalar_group_list_loads": 2 * loads_per_scan * 3 * aic_count,
            "scalar_load_latency_and_cache_policy": "unpriced; not presumed uncached HBM",
            "owner_tile_counts": [sum(1 for _ in tiles(config, i, aic_count)) for i in range(aic_count)],
            "missing_complete_call_phases": ["A/scales GM-to-L1", "MX scale-aware FP8 Cube and MTE1",
                                             "Fixpipe FP32 relay", "SiTU/MXQuant output",
                                             "cross-basic-block epilogue overlap"],
            "supply_consumer_boundary": "L1 publish retired; Cube consumer backpressure not included"}


def build(op, geometry: Geometry):
    raw = dict(op.metadata)["mxfp4_weight_supply"]
    config = Config(**{**raw, "group_rows": tuple(raw["group_rows"])})
    aic = geometry.compute_units
    aiv = geometry.resource_units.get("vector", 0)
    if aiv != 2 * aic:
        raise ModelingError("mxfp4_supply_requires_source_one_aic_two_aiv_ownership")
    if geometry.buffers.get("ub", 0) < UB_BYTES or geometry.buffers.get("l1", 0) < L1_WEIGHT_BYTES:
        raise ModelingError("mxfp4_supply_source_buffers_exceed_device_capacity")
    read = read_role(geometry.resource_units, aiv=0)
    if read != "mte2_aiv" or "mte3" not in geometry.resource_units:
        raise ModelingError("mxfp4_supply_requires_split_aiv_read_and_write_pipes")
    details = ledger(config, aic)
    total = Work(resources={"hbm": details["packed_weight_read_bytes"],
                           read: details["packed_weight_read_bytes"],
                           "vector": details["widen_vector_lanes"],
                           "mte3": details["expanded_l1_write_bytes"]})

    def owner_program(context, owner):
        sequence = iter(panels(config, owner, aic))
        count = details["owner_tile_counts"][owner] * ((config.k + config.k_tile - 1) // config.k_tile)
        allocations = (
            context.allocate(f"weight_l1_{owner}", "l1", L1_WEIGHT_BYTES, (owner,)),
            *(context.allocate(f"weight_ub_{owner}_{s}", "ub", UB_BYTES, (2 * owner + s,)) for s in (0, 1)))

        def tile(_):
            *_, offset = next(sequence)
            stages = []
            for sub in (0, 1):
                work = half_work(config, offset, sub, read).resources
                placement = context.owned(2 * owner + sub, aiv, (read, "vector", "mte3"))
                stages.extend((
                    Stage(f"read{sub}", context.bind(Work(resources={"hbm": work["hbm"], read: work[read]})),
                          placement=placement),
                    Stage(f"widen{sub}", context.bind(Work(resources={"vector": work["vector"]})),
                          after=(f"read{sub}",), placement=placement),
                    Stage(f"publish{sub}", context.bind(Work(resources={"mte3": work["mte3"]})),
                          after=(f"widen{sub}",), placement=placement),
                ))
            return stages

        return KernelProgram((
            Stage("reserve", acquire=allocations),
            Stage("supply", after=("reserve",), program=PipelineProgram(count, 2, tile)),
            Stage("release", after=("supply",), free=allocations),
        ))

    def program(context):
        return ParallelProgram(tuple(owner_program(context, owner) for owner, count
                                     in enumerate(details["owner_tile_counts"]) if count))

    return OperatorPlan(total, program, {"coverage": "source_weight_supply_subprogram",
                                       "mxfp4_supply": details, "source": SOURCE})


def make_op(config, *, op_id="mxfp4_supply", deps=()):
    shape = (sum(config.group_rows), 2 * config.output_width, config.k)
    return OpSpec(op_id=op_id, family="mxfp4_weight_supply", recipe=RECIPE,
                  phase="decode", dtype="fp8_e4m3", source_boundary="subprogram",
                  stage=0, layer=-1, requests=(), semantic_shape=shape,
                  local_shape=shape, padded_shape=shape, launch_shape=shape,
                  deps=tuple(deps), metadata=(("mxfp4_weight_supply", dict(config.__dict__)),
                                            ("invocation_class", "source_internal_segment")),
                  coverage="source_weight_supply_only")


def config_from_group(declaration, members, *, allocated_rows=None):
    """Extract an independently declared integer workload, never capture M."""
    view = declaration.get("tensor_view", {})
    if (declaration.get("role") != "grouped_gate_up_situ_quant"
            or (view.get("input_dtype"), view.get("weight_dtype"), view.get("output_dtype"))
            != ("fp8_e4m3", "mxfp4", "fp8_e4m3")
            or view.get("weight_scale_group_size") != 32
            or view.get("input_scale_dtype") != "fp8_e8m0"
            or view.get("output_scale_dtype") != "fp8_e8m0"):
        raise ModelingError("mxfp4_supply_requires_source_fp8_mxfp4_situ_contract")
    gates = [op for op in members if op.family == "grouped_gemm"]
    if len(gates) != 1:
        raise ModelingError("mxfp4_supply_requires_one_grouped_gate")
    gate = gates[0]
    rows = dict(gate.metadata).get("matrix_group_rows")
    if not isinstance(rows, (list, tuple)):
        raise ModelingError("mxfp4_supply_requires_declared_integer_group_rows")
    m, n, k = gate.local_shape
    if sum(rows) != m:
        raise ModelingError("mxfp4_supply_group_rows_disagree_with_gate")
    return Config(k, n // 2, tuple(rows), m if allocated_rows is None else allocated_rows)


def recipe_book():
    return RecipeBook(recipes={RECIPE: PlannedRecipe(build)})
