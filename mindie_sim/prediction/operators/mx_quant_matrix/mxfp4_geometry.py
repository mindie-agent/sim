"""Source tiling and physical work for the CANN MXA8W4_NZNK branch.

[Role] Preserve integer expert banks and the host/controller tile mapping.
[Invariants] Capacity is not a row count; zero banks have no matrix tiles.
  M/N/K issue padding changes Cube work, never valid rows or expert banks.
[Timeouts & Locks] Lazy tiles, O(experts + N tiles); at most 128 cached geometry
  ledgers, each O(experts + cores + work roles). No I/O or external locks.
[Side-Effect Policy] Pure source geometry, without measurements or rates.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from types import MappingProxyType

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.mechanism import Work
from .mxfp4_registers import (
    B8_BITWISE_OPS_PER_ELEMENT, CLASSES, TRANSPOSE_INDEX_BYTES, transpose, widen,
)
from mindie_sim.prediction.instructions import InstructionInventory

NZ_N_GRANULE = 32
N_TILE_COLUMNS = 256
N_TILE_RESPLIT_COLUMNS = N_TILE_COLUMNS // 2
CUBE_MN_FRACTAL = 16
CUBE_K_FRACTAL = 32
A_REFILL_BUFFER_BYTES = 80 * 1024
B_REFILL_K = 256
WIDE_B_REFILL_K = 512
WIDE_B_REFILL_MAX_N = 128
WIDE_B_REFILL_MAX_M = (
    A_REFILL_BUFFER_BYTES // WIDE_B_REFILL_K // CUBE_MN_FRACTAL * CUBE_MN_FRACTAL
)
MICRO_K = 128
WIDE_MICRO_K = 256
WIDE_MICRO_MAX_MN = 128
AIV_N_SPLIT_ALIGNMENT = 16
SCALE_TRANSPOSE_N_TILE = 16
SCALE_TRANSPOSE_K_PAIRS = 8
MX_SCALE_GROUP = 32
MX_SCALE_GROUP_PAIR = 2 * MX_SCALE_GROUP
MX_SCALE_PUBLISH_K = 4096
SCALE_PUBLISH_COLUMNS = MX_SCALE_PUBLISH_K // MX_SCALE_GROUP


def ceildiv(value, unit):
    return (value + unit - 1) // unit


def align(value, unit):
    return ceildiv(value, unit) * unit


@dataclass(frozen=True)
class Config:
    k: int
    n: int
    rows: tuple[int, ...]

    def __post_init__(self):
        if (type(self.k) is not int or self.k <= 0 or self.k % MX_SCALE_GROUP_PAIR
                or type(self.n) is not int or self.n <= 0 or self.n % NZ_N_GRANULE):
            raise ModelingError("mxa8w4_nznk_requires_k64_n32_source_domain")
        if not self.rows or any(type(m) is not int or m < 0 for m in self.rows):
            raise ModelingError("mxa8w4_nznk_requires_integer_expert_histogram")


def columns(config, cores):
    """CalcResplitTiling uses declared groups, including inactive groups."""
    n, groups = config.n, len(config.rows)
    if n % (cores * N_TILE_COLUMNS) == 0:
        widths = [N_TILE_COLUMNS] * (n // N_TILE_COLUMNS)
    elif n > cores * N_TILE_RESPLIT_COLUMNS:
        main = max(n // (cores * N_TILE_COLUMNS) - 1, 0) * cores
        tail = n - main * N_TILE_COLUMNS
        tasks = cores * (2 if tail > cores * N_TILE_COLUMNS else 1)
        units, extra = divmod(tail // NZ_N_GRANULE, tasks)
        widths = ([N_TILE_COLUMNS] * main
                  + [units * NZ_N_GRANULE] * (tasks - extra)
                  + [(units + 1) * NZ_N_GRANULE] * extra)
    else:
        tasks = ceildiv(n, N_TILE_COLUMNS)
        if groups * tasks >= cores:
            widths = [align(ceildiv(n, tasks), NZ_N_GRANULE)] * tasks
        else:
            tasks = max(1, n // N_TILE_RESPLIT_COLUMNS)
            units, extra = divmod(n // NZ_N_GRANULE, tasks)
            widths = ([units * NZ_N_GRANULE] * (tasks - extra)
                      + [(units + 1) * NZ_N_GRANULE] * extra)
    offset = 0
    for width in widths:
        actual = min(width, n - offset)
        if actual:
            yield offset, actual
        offset += width


def owner_tiles(config, owner, cores):
    """Rolling output-tile index across experts, then M and N."""
    ntiles = tuple(columns(config, cores))
    start = 0
    for expert, rows in enumerate(config.rows):
        if not rows:
            continue
        mstep = ceildiv(rows, ceildiv(rows, M_TILE_ROWS))
        count = ceildiv(rows, mstep) * len(ntiles)
        for local in range((owner - start) % cores, count, cores):
            mi, ni = divmod(local, len(ntiles))
            column, n = ntiles[ni]
            yield expert, mi * mstep, column, min(mstep, rows - mi * mstep), n
        start += count


def tile_runs(config, owner, cores, compact):
    """Consecutive equal physical tiles; each repeat still reads its own bank."""
    tiles = owner_tiles(config, owner, cores)
    for (m, n), members in groupby(tiles, key=lambda tile: tile[-2:]):
        if compact:
            yield m, n, sum(1 for _ in members)
        else:
            for _ in members:
                yield m, n, 1


def panel_sizes(m, n):
    # The no-bias controller reserves 80 KiB for A. Its dynamic M threshold
    # is floor(80KiB / 512 / 16)*16 = 160, not a measured dispatch heuristic.
    kb = WIDE_B_REFILL_K if m <= WIDE_B_REFILL_MAX_M and n <= WIDE_B_REFILL_MAX_N else B_REFILL_K
    ma = align(m, CUBE_MN_FRACTAL)
    ka = min(ceildiv(n, 2 * ma), A_REFILL_BUFFER_BYTES // (ma * kb)) * kb
    micro = WIDE_MICRO_K if m <= WIDE_MICRO_MAX_MN and n <= WIDE_MICRO_MAX_MN else MICRO_K
    return kb, ka, micro


DEFAULT_CORES = 32
GEOMETRY_CACHE_SIZE = 128
M_TILE_ROWS = 256
#: Declared arch3510 L2 efficiency used only when a caller supplies no typed
#: physical declaration; it is the same profile default as the mechanism spec.
L2_EFFECTIVE_FRACTION = 0.75


def effective_l2_bytes(geometry):
    """Typed effective L2 capacity, or the declared profile fallback."""
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    buffers = getattr(geometry, "buffers", {}) or {}
    return L2_EFFECTIVE_FRACTION * float(buffers.get("l2", 0))


def activation_bank_bytes(config):
    """One ``[rows, K]`` activation slice across every active expert bank."""
    return float(sum(config.rows) * config.k)


def m_tile_repeats(config):
    """Second-and-later M-tile passes over one bank, and all M-tile passes."""
    passes = sum(ceildiv(rows, M_TILE_ROWS) for rows in config.rows if rows)
    repeats = sum(max(0, ceildiv(rows, M_TILE_ROWS) - 1)
                  for rows in config.rows if rows)
    return repeats, passes


def owner_panel_bytes(config, cores=DEFAULT_CORES):
    """Concurrent 32-owner 256-column B panel set, including MX scale pairs."""
    width = max((width for _, width in columns(config, cores)), default=config.n)
    payload = config.k * width // 2
    scales = config.k * width // MX_SCALE_GROUP
    return float(cores * (payload + scales))


def resplit_columns(config, cores=DEFAULT_CORES):
    """True when ``CalcResplitTiling`` narrows the declared 256-column tiling."""
    if config.n <= cores * N_TILE_RESPLIT_COLUMNS:
        return False
    return any(width < N_TILE_COLUMNS for _, width in columns(config, cores))


def weight_l2_fraction(config, cores=DEFAULT_CORES, l2_bytes=0):
    """Share of a repeated 256-column B panel served by resident L2.

    The declared working set is the concurrent 32-owner panel set plus one
    activation slice.  When that set fits in effective L2, a second M-tile pass
    re-reads its B panel on chip; the fraction is the repeated-pass share
    scaled by how much of the concurrent set the die can retain.
    """
    if l2_bytes <= 0:
        return 0.0
    repeats, passes = m_tile_repeats(config)
    if not passes:
        return 0.0
    working = owner_panel_bytes(config, cores) + activation_bank_bytes(config)
    retention = min(1.0, float(l2_bytes) / working) if working else 0.0
    return retention * repeats / passes


def activation_l2_fraction(config, cores=DEFAULT_CORES, l2_bytes=0):
    """Share of a column pass whose activation slice is served by resident L2.

    ``CalcResplitTiling`` reads one ``[m_e, K]`` slice per output column tile;
    the first pass fills L2 and the remaining passes hit it while the full
    slice fits the declared effective capacity.
    """
    if l2_bytes <= 0:
        return 0.0
    slice_bytes = activation_bank_bytes(config)
    retention = min(1.0, float(l2_bytes) / slice_bytes) if slice_bytes else 0.0
    # The producer writes the slice through L2 with declared write allocation,
    # so a resident slice serves the first consumer pass as well.
    return retention


def activation_scale_bytes(config, m, offset):
    """One panel's FP8 activation-scale payload (not a weight read)."""
    scale_k = (min(MX_SCALE_PUBLISH_K, config.k - offset)
               if offset % MX_SCALE_PUBLISH_K == 0 else 0)
    return m * scale_k // MX_SCALE_GROUP


def panel_work(config, m, n, offset):
    """Source payload and fixed scale-publish geometry for one K panel."""
    kb, ka, micro = panel_sizes(m, n)
    depth = min(kb, config.k - offset)
    scale_k = (min(MX_SCALE_PUBLISH_K, config.k - offset)
               if offset % MX_SCALE_PUBLISH_K == 0 else 0)
    a = m * min(ka, config.k - offset) if offset % ka == 0 else 0
    a_scale = activation_scale_bytes(config, m, offset)
    vectors = []
    n0 = align(n // 2, AIV_N_SPLIT_ALIGNMENT) if n > AIV_N_SPLIT_ALIGNMENT else n
    for sub in (0, 1):
        ks = min(kb // 2, max(0, depth - sub * (kb // 2)))
        ns = n0 if sub == 0 else n - n0
        b_scale = ns * scale_k // MX_SCALE_GROUP
        # Source DataCopy uses 128 scale columns even for the last K chunk.
        publish_scale = align(ns, SCALE_TRANSPOSE_N_TILE) * SCALE_PUBLISH_COLUMNS if scale_k else 0
        # Transpose gathers 16x8 uint16 blocks, including its final K tail.
        transpose = (
            align(ns, SCALE_TRANSPOSE_N_TILE)
            * align(scale_k // MX_SCALE_GROUP_PAIR, SCALE_TRANSPOSE_K_PAIRS)
            if scale_k else 0
        )
        elements = n * ks
        vectors.append({"weight": elements // 2, "expanded": elements,
                        "scale": b_scale, "scale_publish": publish_scale,
                        "bitwise_b8_lanes": B8_BITWISE_OPS_PER_ELEMENT * elements,
                        "gather_b16_lanes": transpose})
    micros = []
    for kk in range(0, depth, micro):
        k = min(micro, depth - kk)
        # Source-facing distinction: A GM->L1 ND2NZ transfers the valid m
        # rows, but LoadAL1ToL0A is a 2D fractal transfer with
        # mStep=ceil(l0M/16) unit-16 fractals even for M=1.  L0A operand and
        # scale volume therefore use align(m,16).  MMAD parameter is valid m
        # with disableGemv=true; the normal arch3510 FP8 issue granularity is
        # 16x16x32, so issued physical matrix work uses the rounded m/n/k while
        # logical FLOPs remain valid m.  This is a matrix consumption ledger,
        # never an elapsed-time multiplier.
        issued_m, issued_n = align(m, CUBE_MN_FRACTAL), align(n, CUBE_MN_FRACTAL)
        issued_k = align(k, CUBE_K_FRACTAL)
        operand_row_bytes = issued_k + ceildiv(k, MX_SCALE_GROUP_PAIR) * 2
        left, right = issued_m * operand_row_bytes, issued_n * operand_row_bytes
        micros.append((left, right, 2 * issued_m * issued_n * issued_k))
    return a + a_scale, vectors, micros


def tile_work(config, m, n, *, cores=DEFAULT_CORES, directional_mte1=False,
              activation_l2=0.0, weight_l2=0.0):
    total = Work()
    share = float(activation_l2)
    wshare = float(weight_l2)
    for offset in range(0, config.k, panel_sizes(m, n)[0]):
        a, vectors, micros = panel_work(config, m, n, offset)
        packed = sum(v["weight"] + v["scale"] for v in vectors)
        weight = sum(v["weight"] for v in vectors)
        # The separate A-scale transfer keeps its own stream; only the A
        # payload and the repeated B panel can be served from resident L2.
        payload = a - activation_scale_bytes(config, m, offset)
        total += Work({"fp8_e4m3": sum(f for _, _, f in micros)}, {
            "hbm": payload * (1.0 - share) + (a - payload) + packed
                   - weight * wshare,
            "mte2_aic": a, "mte2_aiv": packed,
            **({"l2": payload * share + weight * wshare}
               if share > 0 or wshare > 0 else {}),
            "mte3": sum(v["expanded"] + v["scale_publish"] for v in vectors),
            **({"mte1_l0a": sum(a for a, _, _ in micros), "mte1_l0b": sum(b for _, b, _ in micros)}
               if directional_mte1 else {"mte1": sum(a + b for a, b, _ in micros)}),
        })
        for vector in vectors:
            total += (widen(vector["expanded"]) + transpose(vector["gather_b16_lanes"])).capacity_work(CLASSES)
    # Source GetTensorC converts FP32 L0C directly to BF16 GM in Fixpipe.
    return total + Work(resources={"hbm": 2 * m * n, "fixpipe": 2 * m * n})


def tile_inventory(config, m, n):
    result = InstructionInventory({})
    for offset in range(0, config.k, panel_sizes(m, n)[0]):
        _, vectors, _ = panel_work(config, m, n, offset)
        for vector in vectors:
            result += widen(vector["expanded"]) + transpose(vector["gather_b16_lanes"])
    return result


@dataclass(frozen=True)
class GeometryLedger:
    """Immutable work totals and owner counts; no execution state is retained."""
    work: Work
    inventory: InstructionInventory
    counts: tuple[int, ...]
    runs: tuple[int, ...]


@lru_cache(maxsize=GEOMETRY_CACHE_SIZE)
def geometry_ledger(config: Config, cores: int, compact: bool = True,
                    directional_mte1: bool = False, l2_bytes: int = 0) -> GeometryLedger:
    """Reuse exact source work for repeated layer geometry, independent of rates.

    The source has two AIVs per AIC. InitMx reads its 256-byte permutation on
    both AIVs even when their AIC has no matrix tile. All other work is the
    existing integer tile/run ledger. No operator, device, context, observed
    value or program closure enters the key or cached value. Cached mappings
    are read-only; Work.apply/scaled and inventory conversion create fresh
    caller-owned outputs instead of exposing mutable cache state.
    """
    aivs = 2 * cores
    init_bytes = TRANSPOSE_INDEX_BYTES * aivs
    work = Work(resources={"hbm": init_bytes, "mte2_aiv": init_bytes})
    inventory = InstructionInventory({})
    counts, runs = [], []
    activation_l2 = activation_l2_fraction(config, cores, l2_bytes)
    weight_l2 = weight_l2_fraction(config, cores, l2_bytes)
    for owner in range(cores):
        count = sum(1 for _ in owner_tiles(config, owner, cores))
        run_count = 0
        for m, n, repeat in tile_runs(config, owner, cores, compact):
            work += tile_work(config, m, n, cores=cores,
                              directional_mte1=directional_mte1,
                              activation_l2=activation_l2,
                              weight_l2=weight_l2).scaled(repeat)
            inventory += tile_inventory(config, m, n).scaled(repeat)
            run_count += 1
        counts.append(count)
        runs.append(run_count)
    return GeometryLedger(
        Work(MappingProxyType(dict(work.matrix)), MappingProxyType(dict(work.resources))),
        InstructionInventory(MappingProxyType(dict(inventory.counts)), inventory.barriers),
        tuple(counts), tuple(runs),
    )
