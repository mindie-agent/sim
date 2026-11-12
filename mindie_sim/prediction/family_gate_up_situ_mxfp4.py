"""First routed gate/up source tile plan for GroupedMatmulSituQuant.

[Role] Correct the finite BASE_M128 / OUTPUT_TILE_N128 matrix stage owned by
the already-selected grouped_gate_up_situ_quant fusion.  The SiTU/quantizer
members keep their existing lowerings; this recipe replaces only the gate
matrix member's hidden generic tile work and AIC ownership.
[Invariants] Only integer active bank rows produce tiles.  Zero banks skip.
No BF16 workspace, no supply-only elapsed, no second kernel IR.
"""
from collections import Counter
from dataclasses import dataclass, replace
from math import ceil

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .mxfp4_geometry import Config as GeoConfig, panel_sizes, panel_work
from .gmmsq_quant_typed import work as gmmsq_widen_work
from .program import KernelProgram, ParallelProgram, Stage, RecipeBook
from .operators import grouped_gemm
from . import mxfp4_supply

RECIPE = PROFILE = "grouped_gate_up_situ_mxfp4_source"
SOURCE = mxfp4_supply.SOURCE

@dataclass(frozen=True)
class Config:
    rows: tuple
    width: int
    k: int
    l1_k: int = 256
    compact: bool = True


@dataclass(frozen=True)
class Tile:
    m: int
    n: int


def owner_tiles(rows, width, tile, cores):
    owners, cursor = [[] for _ in range(cores)], 0
    for bank, rows_in_bank in enumerate(rows):
        if not rows_in_bank:
            continue
        nm = ceil(rows_in_bank / tile.m)
        nn = ceil(width / tile.n)
        for mi in range(nm):
            for ni in range(nn):
                owners[cursor % cores].append(
                    (bank, min(tile.m, rows_in_bank - mi * tile.m),
                     min(tile.n, width - ni * tile.n), mi * tile.m, ni * tile.n))
                cursor += 1
    return tuple(tuple(items) for items in owners)


def matrix_work(m, n, k, tile):
    """One FP8 matrix half reads A + packed FP8 from L1 and accumulates FP32."""
    return Work(
        {"fp8_e4m3": 4 * m * n * k},
        {"mte1": m * k + 2 * n * k, "fixpipe": 8 * m * n},
    )


def build(op, geometry):
    raw = dict(op.metadata)["mxfp4_gate_config"]
    cfg = Config(tuple(int(x) for x in raw["group_rows"]),
                 int(raw["output_width"]), int(raw["k"]),
                 int(raw.get("l1_k", 256)))
    if (not cfg.rows or any(x < 0 for x in cfg.rows)
            or cfg.width < 64 or cfg.width % 64
            or cfg.k < 64 or cfg.k % 64 or cfg.l1_k < 128
            or cfg.l1_k % 128):
        raise ModelingError("mxfp4_gate_requires_integer_rows_N64_K64_L1K128")
    if not sum(cfg.rows):
        return OperatorPlan(Work(), lambda context: ParallelProgram(()), {
            "implementation": RECIPE, "source": SOURCE,
            "source_execution_domain": "mixed_cube_aiv",
            "source_output_tiles": 0, "active_aic_owners": 0,
            "matrix_group_rows": cfg.rows, "output_width": cfg.width,
            "k": cfg.k, "producer_packed_weight_bytes": 0,
            "limitations": ["zero_bank_skips_no_fake_tiles"],
        })
    cores = geometry.compute_units
    for role in ("mte1", "mte2_aic", "fixpipe"):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError("mxfp4_gate_requires_private_aic:" + role)
    # Source N output tile is 128; 24 populated tiles for 3072 and 32 AIC.
    tile = Tile(128, 128)
    owners = owner_tiles(cfg.rows, cfg.width, tile, cores)
    aiv = geometry.resource_units.get("vector", 2 * cores)
    total = Work()
    logical_flops = 0
    api_flops = 0
    physical_flops = 0
    issued_flops = 0
    fixpipe_bytes = 0
    logical_fixpipe_bytes = 0
    physical_ms = set()
    weight_scale_bytes = 0
    producer_packed_weight_bytes = 0
    raw_alu_lanes = 0
    raw_shuffle_lanes = 0
    vf_groups_total = 0
    for items in owners:
        if not items:
            continue
        for _, m, n, _, _ in items:
            geo = GeoConfig(cfg.k, 2 * n, (m,))
            kb, ka, micro = panel_sizes(m, 2 * n)
            for offset in range(0, cfg.k, kb):
                a, vectors, micros = panel_work(geo, m, 2 * n, offset)
                total += Work(resources={"hbm": a, "mte2_aic": a})
                for vector in vectors:
                    scale = vector["scale"]
                    weight_scale_bytes += scale
                    producer_packed_weight_bytes += vector["weight"]
                    groups = vector["weight"] // 256
                    typed, counts, _ = gmmsq_widen_work(groups)
                    raw_alu_lanes += counts.raw_alu_lanes
                    raw_shuffle_lanes += counts.raw_shuffle_lanes
                    vf_groups_total += counts.vf_groups
                    total += typed
                    total += Work(resources={
                        "hbm": vector["weight"] + scale,
                        "mte2_aiv": vector["weight"],
                        "mte2_aic": scale,
                        "mte3": vector["expanded"],
                    })
                for left, right, flops in micros:
                    # Shared panel_work supplies the physical Cube flops:
                    # Mmad m is CeilAlign(M,2), then the A5 16x16x32
                    # instruction pads that API M to a 16-row physical tile.
                    # Use it unscaled here.
                    issued = flops
                    issued_flops += issued
                    total += Work({"fp8_e4m3": issued}, {"mte1": left + right})
            api_m = ((m + 1) // 2) * 2
            hw_m = ((api_m + 15) // 16) * 16
            logical_flops += 2 * m * 2 * n * cfg.k
            api_flops += 2 * api_m * 2 * n * cfg.k
            physical_flops += 2 * hw_m * 2 * n * cfg.k
            logical_fixpipe = 2 * m * n * 4
            released = 2 * api_m * n * 4
            logical_fixpipe_bytes += logical_fixpipe
            fixpipe_bytes += released
            physical_ms.add(hw_m)
            total += Work(resources={"fixpipe": released})
    def lower(ctx):
        def stage(name, work, owner, units=None, **kwargs):
            roles = set(work.resources) - {"hbm"} | ({"cube"} if work.matrix else set())
            return Stage(name, ctx.bind(work),
                         placement=ctx.owned(owner, units or cores, roles), **kwargs)
        def one(owner_id, items):
            if not items:
                return KernelProgram(())
            allocation = ctx.allocate(f"gate_l1_{owner_id}", "l1",
                                      2 * mxfp4_supply.L1_WEIGHT_BYTES, (owner_id,))
            ub0 = ctx.allocate(f"gate_ub_{owner_id}_0", "ub",
                               mxfp4_supply.UB_BYTES, (2 * owner_id,))
            ub1 = ctx.allocate(f"gate_ub_{owner_id}_1", "ub",
                               mxfp4_supply.UB_BYTES, (2 * owner_id + 1,))
            stages = [Stage(f"owner{owner_id}:reserve",
                            acquire=(allocation, ub0, ub1))]
            a_total = 0
            for _, m, n, _, _ in items:
                geo = GeoConfig(cfg.k, 2 * n, (m,))
                for offset in range(0, cfg.k, panel_sizes(m, 2 * n)[0]):
                    a, _, _ = panel_work(geo, m, 2 * n, offset)
                    a_total += a
            a = stage(f"owner{owner_id}:a_read",
                      Work(resources={"hbm": a_total, "mte2_aic": a_total}),
                      owner_id)
            stages.append(a)
            previous = f"owner{owner_id}:a_read"
            for tile_index, (_, m, n, _, _) in enumerate(items):
                geo = GeoConfig(cfg.k, 2 * n, (m,))
                kb, ka, micro = panel_sizes(m, 2 * n)
                panel_prev = previous
                last_cube = previous
                for offset in range(0, cfg.k, kb):
                    a_work, vectors, micros = panel_work(geo, m, 2 * n, offset)
                    scale_total = sum(v["scale"] for v in vectors)
                    if scale_total:
                        sc = stage(f"owner{owner_id}:t{tile_index}:k{offset}:wscale",
                                   Work(resources={"hbm": scale_total,
                                                   "mte2_aic": scale_total}),
                                   owner_id, after=(panel_prev,))
                        stages.append(sc)
                        panel_prev = sc.name
                    publishes = []
                    for sub, vector in enumerate(vectors):
                        prefix = f"owner{owner_id}:t{tile_index}:k{offset}:s{sub}"
                        read = stage(prefix + ":read", Work(resources={
                            "hbm": vector["weight"],
                            "mte2_aiv": vector["weight"]}),
                            2 * owner_id + sub, aiv, after=(panel_prev,))
                        groups = vector["weight"] // 256
                        typed, _, _ = gmmsq_widen_work(groups)
                        widen_stage = stage(prefix + ":widen",
                                            typed, 2 * owner_id + sub, aiv,
                                            after=(prefix + ":read",))
                        publish = stage(prefix + ":publish",
                                        Work(resources={"mte3": vector["expanded"]}),
                                        2 * owner_id + sub, aiv,
                                        after=(prefix + ":widen",))
                        stages.extend((read, widen_stage, publish))
                        publishes.append(prefix + ":publish")
                    load = stage(f"owner{owner_id}:t{tile_index}:k{offset}:load",
                                 Work(resources={
                                     "mte1": sum(left + right for left, right, _ in micros)}),
                                 owner_id, after=tuple(publishes + [panel_prev]))
                    stages.append(load)
                    for micro_index, (left, right, flops) in enumerate(micros):
                        issued = flops
                        cube = stage(
                            f"owner{owner_id}:t{tile_index}:k{offset}:c{micro_index}",
                            Work({"fp8_e4m3": issued}), owner_id,
                            after=(load.name,))
                        stages.append(cube)
                        last_cube = cube.name
                    panel_prev = load.name
                fix = stage(f"owner{owner_id}:t{tile_index}:fixpipe",
                            Work(resources={"fixpipe": 2 * (2 * ((m + 1) // 2)) * n * 4}),
                            owner_id, after=(last_cube,))
                stages.append(fix)
                previous = fix.name
            stages.append(Stage(f"owner{owner_id}:release", after=(previous,),
                                free=(allocation, ub0, ub1)))
            return KernelProgram(tuple(stages))
        return ParallelProgram(tuple(one(i, items)
                                     for i, items in enumerate(owners)))
    return OperatorPlan(total, lower, {
        "implementation": RECIPE, "source": SOURCE,
        "source_execution_domain": "mixed_cube_aiv",
        "source_output_tiles": sum(map(len, owners)),
        "active_aic_owners": sum(bool(x) for x in owners),
        "matrix_base_tile": (128, 128, 128),
        "matrix_group_rows": cfg.rows, "output_width": cfg.width, "k": cfg.k,
        "logical_valid_flops": logical_flops,
        "source_api_flops": api_flops,
        "physical_cube_flops": issued_flops,
        "physical_cube_flops_derived": physical_flops,
        "logical_fixpipe_bytes": logical_fixpipe_bytes,
        "source_api_fixpipe_bytes": fixpipe_bytes,
        "physical_cube_m_by_bank": tuple(sorted(physical_ms)),
        "producer_packed_weight_bytes": producer_packed_weight_bytes,
        "weight_scale_bytes": weight_scale_bytes,
        "unpriced_typed_lanes": {
            "reg_alu_lanes": raw_alu_lanes,
            "reg_shuffle_lanes": raw_shuffle_lanes,
            "vf_groups": vf_groups_total,
            "boundary": "raw opcode lanes; no physical shared port",
        },
        "limitations": ["situ_quant_member_lowering_preserved",
                        "typed_widen_inventory_reg_roles",
                        "source_issued_m2_fixpipe_separate_from_logical",
                        "producer_matrix_pipeline_finite_approximation"],
    })

def recipe():
    return RecipeBook({RECIPE: PlannedRecipe(build)})
