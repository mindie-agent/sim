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
from .mechanism import Work
from .mxfp4_registers import CLASSES, widen, transpose
from .instructions import InstructionInventory


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
        if (type(self.k) is not int or self.k <= 0 or self.k % 64
                or type(self.n) is not int or self.n <= 0 or self.n % 32):
            raise ModelingError("mxa8w4_nznk_requires_k64_n32_source_domain")
        if not self.rows or any(type(m) is not int or m < 0 for m in self.rows):
            raise ModelingError("mxa8w4_nznk_requires_integer_expert_histogram")


def columns(config, cores):
    """CalcResplitTiling uses declared groups, including inactive groups."""
    n, groups = config.n, len(config.rows)
    if n % (cores * 256) == 0:
        widths = [256] * (n // 256)
    elif n > cores * 128:
        main = max(n // (cores * 256) - 1, 0) * cores
        tail = n - main * 256
        tasks = cores * (2 if tail > cores * 256 else 1)
        units, extra = divmod(tail // 32, tasks)
        widths = [256] * main + [units * 32] * (tasks - extra) + [(units + 1) * 32] * extra
    else:
        tasks = ceildiv(n, 256)
        if groups * tasks >= cores:
            widths = [align(ceildiv(n, tasks), 32)] * tasks
        else:
            tasks = max(1, n // 128)
            units, extra = divmod(n // 32, tasks)
            widths = [units * 32] * (tasks - extra) + [(units + 1) * 32] * extra
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
        mstep = ceildiv(rows, ceildiv(rows, 256))
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
    kb = 512 if m <= 160 and n <= 128 else 256
    ma = align(m, 16)
    ka = min(ceildiv(n, 2 * ma), 80 * 1024 // (ma * kb)) * kb
    micro = 256 if m <= 128 and n <= 128 else 128
    return kb, ka, micro


def panel_work(config, m, n, offset):
    """Source payload and fixed scale-publish geometry for one K panel."""
    kb, ka, micro = panel_sizes(m, n)
    depth = min(kb, config.k - offset)
    scale_k = min(4096, config.k - offset) if offset % 4096 == 0 else 0
    a = m * min(ka, config.k - offset) if offset % ka == 0 else 0
    a_scale = m * scale_k // 32
    vectors = []
    n0 = align(n // 2, 16) if n > 16 else n
    for sub in (0, 1):
        ks = min(kb // 2, max(0, depth - sub * (kb // 2)))
        ns = n0 if sub == 0 else n - n0
        b_scale = ns * scale_k // 32
        # Source DataCopy uses 128 scale columns even for the last K chunk.
        publish_scale = align(ns, 16) * 128 if scale_k else 0
        # Transpose gathers 16x8 uint16 blocks, including its final K tail.
        transpose = align(ns, 16) * align(scale_k // 64, 8) if scale_k else 0
        elements = n * ks
        vectors.append({"weight": elements // 2, "expanded": elements,
                        "scale": b_scale, "scale_publish": publish_scale,
                        "bitwise_b8_lanes": 5 * elements,
                        "gather_b16_lanes": transpose})
    micros = []
    for kk in range(0, depth, micro):
        k = min(micro, depth - kk)
        # LoadData2D uses 16-row x 32-byte data tiles and paired E8M0 scale.
        left = align(m, 16) * (align(k, 32) + ceildiv(k, 64) * 2)
        right = align(n, 16) * (align(k, 32) + ceildiv(k, 64) * 2)
        micros.append((left, right, 2 * align(m, 16) * align(n, 16) * align(k, 32)))
    return a + a_scale, vectors, micros


def tile_work(config, m, n, *, directional_mte1=False):
    total = Work()
    for offset in range(0, config.k, panel_sizes(m, n)[0]):
        a, vectors, micros = panel_work(config, m, n, offset)
        packed = sum(v["weight"] + v["scale"] for v in vectors)
        total += Work({"fp8_e4m3": sum(f for _, _, f in micros)}, {
            "hbm": a + packed, "mte2_aic": a, "mte2_aiv": packed,
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


@lru_cache(maxsize=128)
def geometry_ledger(config: Config, cores: int, compact: bool = True,
                    directional_mte1: bool = False) -> GeometryLedger:
    """Reuse exact source work for repeated layer geometry, independent of rates.

    The source has two AIVs per AIC. InitMx reads its 256-byte permutation on
    both AIVs even when their AIC has no matrix tile. All other work is the
    existing integer tile/run ledger. No operator, device, context, observed
    value or program closure enters the key or cached value. Cached mappings
    are read-only; Work.apply/scaled and inventory conversion create fresh
    caller-owned outputs instead of exposing mutable cache state.
    """
    aivs = 2 * cores
    work = Work(resources={"hbm": 256 * aivs, "mte2_aiv": 256 * aivs})
    inventory = InstructionInventory({})
    counts, runs = [], []
    for owner in range(cores):
        count = sum(1 for _ in owner_tiles(config, owner, cores))
        run_count = 0
        for m, n, repeat in tile_runs(config, owner, cores, compact):
            work += tile_work(config, m, n, directional_mte1=directional_mte1).scaled(repeat)
            inventory += tile_inventory(config, m, n).scaled(repeat)
            run_count += 1
        counts.append(count)
        runs.append(run_count)
    return GeometryLedger(
        Work(MappingProxyType(dict(work.matrix)), MappingProxyType(dict(work.resources))),
        InstructionInventory(MappingProxyType(dict(inventory.counts)), inventory.barriers),
        tuple(counts), tuple(runs),
    )
