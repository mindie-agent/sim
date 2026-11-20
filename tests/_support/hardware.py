"""Shared SKU case builders, independent datasheet specs, and pinned-owner recipes.

[Role] Hold the independent 16-SKU Ascend datasheet supply/L2 expectations,
  build tiny Ascend cases from a registered SKU, and lower explicit byte/FLOP
  work onto named physical owners so schema tests and legacy callers share one
  non-tautological oracle instead of reading ``profiles()`` back against itself.
[Invariants] ``EXPECTED_SKU_SPECS``, ``EXPECTED_L2_BYTES``, and ``A5_BUFFERS``
  are independent literal hardware specs (zero reads from ``profiles()``);
  recipes only bind declared work and placements; ``run`` asserts the Engine
  drains every task, program, and pool.
[Timeouts & Locks] None; pure in-memory construction.
[Side-Effect Policy] Side-effect-free; writes nothing.
"""
from __future__ import annotations

from mindie_sim.backends.ascend import profiles
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.schema.case import Case
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec

MODEL = {
    "name": "pipeline-geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
    "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16,
}
WORKLOAD = {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]}
MIB = 2 ** 20
REGBASE_ISA = "npu_arch_3510"

#: Independent physical datasheet specs: ``(clock_hz, aic, aiv, memory_bytes, hbm_bytes_per_s)``.
EXPECTED_SKU_SPECS: dict[str, tuple[float, int, int, int, float]] = {
    "Ascend910B1": (1.85e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B2C": (1.80e9, 24, 48, 68719476736, 1.8e12),
    "Ascend910B3": (1.80e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910B4": (1.50e9, 20, 40, 34359738368, 0.8e12),
    "Ascend910B4-1": (1.65e9, 20, 40, 68719476736, 1.2e12),
    "Ascend910_9391": (1.85e9, 24, 48, 68719476736, 1.6e12),
    "Ascend910_9392": (1.80e9, 24, 48, 68719476736, 1.6e12),
    "Ascend910_9381": (1.80e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910_9382": (1.65e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910_9372": (1.65e9, 20, 40, 68719476736, 1.4e12),
    "Ascend910_9362": (1.65e9, 20, 40, 68719476736, 1.2e12),
    "Ascend950DT_9581": (1.65e9, 32, 64, 154618822656, 4.0e12),
    "Ascend950DT_9582": (1.65e9, 32, 64, 103079215104, 4.0e12),
    "Ascend950PR_9579": (1.65e9, 32, 64, 68719476736, 1.6e12),
    "Ascend950PR_958b": (1.65e9, 32, 64, 120259084288, 1.4e12),
}
EXPECTED_L2_BYTES: dict[str, int] = {
    "Ascend910B1": 201326592, "Ascend910B2": 201326592,
    "Ascend910B2C": 201326592, "Ascend910B3": 201326592,
    "Ascend910B4": 100663296, "Ascend910B4-1": 100663296,
    "Ascend910_9391": 201326592, "Ascend910_9392": 201326592,
    "Ascend910_9381": 201326592, "Ascend910_9382": 201326592,
    "Ascend910_9372": 201326592, "Ascend910_9362": 176160768,
    "Ascend950DT_9581": 134217728, "Ascend950DT_9582": 134217728,
    "Ascend950PR_9579": 134217728, "Ascend950PR_958b": 134217728,
}


def declared_supply(sku: str) -> tuple[float, int, int, int, float]:
    """Independent datasheet ``(clock_hz, aic, aiv, memory_bytes, hbm_bytes_per_s)``."""
    return EXPECTED_SKU_SPECS[sku]


def isa_of(sku: str) -> str:
    return profiles()[sku]["device"]["compute_arch"]["preset"]


#: Every register-based (DAV_3510) SKU mapped to its independent datasheet supply.
A5 = {sku: EXPECTED_SKU_SPECS[sku] for sku in ("Ascend950DT_9581", "Ascend950DT_9582", "Ascend950PR_9579", "Ascend950PR_958b")}
#: Independent per-owner on-chip buffers for register-based DAV_3510 dies.
A5_BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144, "ub": 253952}


def machine(sku: str, **overrides) -> Hardware:
    return Hardware.from_mapping({"backend": "ascend", "sku": sku, "num_devices": 1, **overrides})


def sku_case(sku: str, device: dict | None = None, **machine_fields) -> Case:
    raw_machine = {"backend": "ascend", "sku": sku, **machine_fields}
    if device is not None:
        raw_machine["device"] = device
    return Case.from_mapping({"machine": raw_machine, "model": MODEL,
                              "workload": WORKLOAD, "runtime": {}})


def hardware_case() -> dict:
    """A neutral one-device raw case whose machine a test may replace."""
    return {
        "machine": {"name": "arbitrary", "num_devices": 1,
                    "device": {"memory_bytes": 1024 ** 3, "compute_units": 4,
                               "compute_peaks": {"bf16": 2e6, "fp32": 1e6},
                               "rates": {"memory": 1e6, "vector": 1e6}}},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def op(name: str, recipe: str, stream: str = "main") -> OpSpec:
    return OpSpec(name, "test", "prefill", 0, 0, (name,), (1,), (1,), (1,), (1,),
                  recipe=recipe, stream=stream, dtype="bf16")


def run(case, ops, recipes, **kwargs) -> dict:
    result = graph_costs(case, tuple(ops), recipes=RecipeBook(recipes),
                         isolated=False, trace_limit=200, **kwargs)
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
    return result


def one_owner_ms(case, role: str, amount: float, owner: int = 0) -> float:
    return run(case, (op(role, role),), {role: OneMode(role, amount, owner)})["duration_ms"]


class OneMode:
    """One instruction-direction move pinned to one physical owner."""

    def __init__(self, role, amount, owner):
        self.role, self.amount, self.owner = role, amount, owner

    def lower(self, context):
        work = context.bind(Work(resources={self.role: self.amount}))
        return KernelProgram((Stage(
            "move", work, placement=context.placement(self.role, (self.owner,))),))


class OwnerMoves:
    """Split a total amount equally over a fixed set of physical owners."""

    def __init__(self, role, amount, owners):
        self.role, self.amount, self.owners = role, amount, owners

    def _work(self, per_owner):
        return Work(resources={self.role: per_owner})

    def lower(self, context):
        per_owner = self.amount / self.owners
        return KernelProgram(tuple(
            Stage(f"move{owner}", context.bind(self._work(per_owner)),
                  placement=context.placement(self.role, (owner,)))
            for owner in range(self.owners)))


class ReadMoves(OwnerMoves):
    """Split GM transfers over owners and charge the same bytes to shared HBM."""

    def _work(self, per_owner):
        return Work(resources={"hbm": per_owner, self.role: per_owner})


class Matrix:
    """Ordinary matrix work on the profile's declared precision peak."""

    def __init__(self, dtype, flops):
        self.dtype, self.flops = dtype, flops

    def lower(self, context):
        return KernelProgram((Stage(
            "mmad", context.bind(Work(matrix={self.dtype: self.flops}))),))
