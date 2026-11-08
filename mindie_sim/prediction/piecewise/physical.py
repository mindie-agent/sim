"""Named Ascend physical quantities used by the piecewise compiler.

[Role] Hold declared die, tile, pipe and link quantities so piecewise supply
  code does not repeat unlabeled literals.  Profiles remain the runtime source;
  these names are the formulas and the fallback when a key is absent.
[Invariants] Ascend950DT_9582 (and Ascend950DT_9581 / Ascend950PR_9579) is 32 AIC
  and 64 AIV at 1.65 GHz, not 36/72.  Ascend950DT_9582 / 9581 HBM is 4.0 TB/s
  (4.0e12 B/s); Ascend950PR_9579 HBM is 1.6 TB/s (1.6e12 B/s).  Its intra-node
  fabric is UB mesh, not HCCS.  The MMAD fractal is 16x16xk; 128x128x128 is the
  software analytical tile.  Fitted service residuals are not given physical names.
[Timeouts & Locks] Pure constants and arithmetic.  No waits or locks.
[Side-Effect Policy] No I/O, device work or global mutation.
"""
from __future__ import annotations

from typing import Mapping

# Registered arch3510 die.  A 36-AIC / 72-AIV label is not this SKU.
ARCH3510_CLOCK_HZ = 1.65e9
ARCH3510_AIC = 32
ARCH3510_AIV = 64
ARCH3510_AIV_PER_AIC = 2

# arch3510 table 2 MMAD fractal.  Distinct from the software GEMM tile.
MMAD_FRACTAL = 16
CUBE_K_BF16 = 16
CUBE_K_FP16 = 16
CUBE_K_FP8 = 32
CUBE_K_INT8 = 32
CUBE_K_FP32 = 1
CUBE_K_HF32 = 8
CUBE_K_MXFP4 = 64

# docs/architecture.md default analytical tile and two in-flight L0/L1 slots.
SOFTWARE_TILE_M = 128
SOFTWARE_TILE_N = 128
SOFTWARE_TILE_K = 128
SOFTWARE_KV_TILE = 128
SOFTWARE_PIPELINE_SLOTS = 2

# Historical A3 analytical owner count.  Not the 950PR die, and not a default
# that should silently become 32 for every vendor.
A3_ANALYTICAL_CORES = 24

# Analytical split policy, not a die parameter.
ANALYTICAL_SPLIT_TARGET = 4096
ANALYTICAL_MAX_SPLITS = 8
# max, sub, exp, sum, div.  An op count, not a timed kernel.
SOFTMAX_OPS_PER_PAIR = 5

BF16_ELEMENT_BYTES = 2

# On-chip bytes.  The MoE tile screen still defaults to the A3 128 KiB L0C;
# do not replace that default with the arch3510 256 KiB buffer.
ARCH3510_L1_BYTES = 512 * 1024
ARCH3510_L0A_BYTES = 64 * 1024
ARCH3510_L0B_BYTES = 64 * 1024
ARCH3510_L0C_BYTES = 256 * 1024
ARCH3510_UB_BYTES = 253952
A3_L0C_BYTES = 128 * 1024

# Bytes moved per cycle on one owner.  Aggregate rate is this times owners
# times clock.  On 9579, mte2_aic and mte2_aiv both land at 4.224e12 B/s.
MTE1_BYTES_PER_CYCLE_PER_AIC = 256
MTE2_AIC_BYTES_PER_CYCLE = 80
MTE2_AIV_BYTES_PER_CYCLE = 40
MTE3_BYTES_PER_CYCLE_PER_AIV = 128
FIXPIPE_BYTES_PER_CYCLE_PER_AIC = 128

# Ascend950DT_9582 / 9581 HBM is 4.0e12 B/s (4.0 TB/s); Ascend950PR_9579 is 1.6e12 B/s.
ARCH3510_9582_HBM_BYTES_PER_SECOND = 4.0e12
ARCH3510_9579_HBM_BYTES_PER_SECOND = 1.6e12

# Intra-node UB mesh and inter-node RoCEv2 400G.  HCCS is an A3 protocol,
# not the 9579 on-node fabric.  Engine link work is bytes per millisecond,
# which is decimal GB/s times 1e6.
UB_MESH_UNI_GBPS = 392.0
UB_MESH_BI_GBPS = 725.2
UB_MESH_ALPHA_US = 1.8
# Historical millisecond literal.  ``1.8e-3`` is not the same float as this.
UB_MESH_ALPHA_MS = 0.0018
ROCE_400G_UNI_GBPS = 50.0
ROCE_400G_BI_GBPS = 90.0
ROCE_400G_ALPHA_US = 5.0
REMOTE_HOST_ALPHA_US = 6.5

# Declared arch3510 per-invocation dispatch/drain minima (ms) from SKU microbench.
ARCH3510_VECTOR_MIN_MS = 0.001312
ARCH3510_CUBE_MIN_MS = 0.001578
ARCH3510_W8A8_ND_BF16_MIN_MS = 0.003037
ARCH3510_EMPTY_AIC_MIN_MS = 0.000439

# Default pipe/HBM multiple when a minimal test profile omits mte1, mte3 or fixpipe.
UNDECLARED_PIPE_OVER_HBM = 2.0
MISSING_MEMORY_BYTES_PER_MS = 1.0
US_TO_MS = 1e-3


def gbps_to_bytes_per_ms(gbps: float) -> float:
    """Decimal GB/s (1e9 B/s) to engine bytes per millisecond."""
    return float(gbps) * 1_000_000.0


def arch3510_bytes_per_second(bytes_per_cycle: int, owners: int) -> float:
    """One arch3510 pipe: bytes/cycle/owner * owners * 1.65 GHz."""
    return float(bytes_per_cycle) * float(owners) * ARCH3510_CLOCK_HZ


UB_MESH_UNI_BYTES_PER_MS = gbps_to_bytes_per_ms(UB_MESH_UNI_GBPS)
UB_MESH_BI_BYTES_PER_MS = gbps_to_bytes_per_ms(UB_MESH_BI_GBPS)
ROCE_400G_UNI_BYTES_PER_MS = gbps_to_bytes_per_ms(ROCE_400G_UNI_GBPS)
ROCE_400G_BI_BYTES_PER_MS = gbps_to_bytes_per_ms(ROCE_400G_BI_GBPS)


def aggregate_mte2(rates: Mapping[str, float]) -> float | None:
    """Combine split MTE2 supplies already expressed in engine units.

    Profiles declare ``mte2_aic`` and ``mte2_aiv`` and no combined ``mte2``.
    Equal splits use that rate.  Unequal splits use the slower path.  Absent
    splits return None so the caller can fall back; HBM is not an MTE2 rate.
    """
    aic = rates.get("mte2_aic")
    aiv = rates.get("mte2_aiv")
    present = [value for value in (aic, aiv) if value is not None and value > 0]
    if not present:
        return None
    if aic is not None and aiv is not None and aic > 0 and aiv > 0:
        return aic if aic == aiv else min(aic, aiv)
    return present[0]


def fill_undeclared_byte_pipes(rates: dict[str, float]) -> tuple[str, ...]:
    """Fill byte pipes a profile did not declare and return the defaulted pipe names."""
    filled: list[str] = []
    memory = rates.get("hbm", rates.get("memory", MISSING_MEMORY_BYTES_PER_MS))
    declared_mte2 = aggregate_mte2(rates)
    for pipe, default, is_declared in (
        ("mte1", UNDECLARED_PIPE_OVER_HBM * memory, "mte1" in rates),
        ("mte2", memory if declared_mte2 is None else declared_mte2, "mte2" in rates or declared_mte2 is not None),
        ("mte3", UNDECLARED_PIPE_OVER_HBM * memory, "mte3" in rates),
        ("fixpipe", UNDECLARED_PIPE_OVER_HBM * memory, "fixpipe" in rates),
        ("hbm", rates.get("memory", MISSING_MEMORY_BYTES_PER_MS), "hbm" in rates or "memory" in rates),
    ):
        if pipe not in rates:
            rates[pipe] = float(default)
        if not is_declared:
            filled.append(pipe)
    return tuple(filled)

def declared_aic_owners(device: object, default: int = A3_ANALYTICAL_CORES) -> int:
    """AIC owner count from a die. Missing or nonpositive values stay on the A3 default."""
    raw = getattr(device, "compute_units", None)
    cores = default if raw is None else int(raw)
    return cores if cores >= 1 else default
