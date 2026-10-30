"""Per-SKU A3/A5 pipeline supply, precision and shared-resource controls.

Expected rates are independently recomputed from the selected profile's declared
clock and AIC/AIV counts for A3, and from the arch3510 per-core rates for A5.
The common Engine then executes explicit byte and FLOP work; no expectation is
read back from the profile field being checked.
"""
from dataclasses import replace

import pytest

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

# sku: clock_hz, AIC, AIV, memory, HBM B/s
A3 = {
    "Ascend910_9391": (1.85e9, 24, 48, 68719476736, 1.6e12),
    "Ascend910_9392": (1.80e9, 24, 48, 68719476736, 1.6e12),
    "Ascend910_9381": (1.80e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910_9382": (1.65e9, 20, 40, 68719476736, 1.6e12),
    "Ascend910_9372": (1.65e9, 20, 40, 68719476736, 1.4e12),
    "Ascend910_9362": (1.65e9, 20, 40, 68719476736, 1.2e12),
}
A5 = {
    "Ascend950DT_9581": (1.65e9, 32, 64, 154618822656, 4.0e12),
    "Ascend950DT_9582": (1.65e9, 32, 64, 103079215104, 4.0e12),
    "Ascend950PR_9579": (1.65e9, 32, 64, 68719476736, 1.6e12),
}
A3_BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 131072, "ub": 196608}
A5_BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144, "ub": 253952}


def sku_case(sku):
    return Case.from_mapping({
        "machine": {"backend": "ascend", "sku": sku},
        "model": MODEL, "workload": WORKLOAD, "runtime": {},
    })


def op(name, recipe, stream="main"):
    return OpSpec(name, "test", "prefill", 0, 0, (name,), (1,), (1,), (1,), (1,),
                  recipe=recipe, stream=stream, dtype="bf16")


def run(case, ops, recipes):
    result = graph_costs(case, tuple(ops), recipes=RecipeBook(recipes),
                         isolated=False, trace_limit=200)
    assert result["execution"]["live_tasks"] == 0
    assert result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
    return result


class OneMode:
    """One instruction-direction move pinned to one physical owner."""

    def __init__(self, role, amount, owner):
        self.role, self.amount, self.owner = role, amount, owner

    def lower(self, context):
        work = context.bind(Work(resources={self.role: self.amount}))
        return KernelProgram((Stage(
            "move", work, placement=context.placement(self.role, (self.owner,)),
        ),))


class OwnerMoves:
    """Split a total amount equally over a fixed set of physical owners."""

    def __init__(self, role, amount, owners):
        self.role, self.amount, self.owners = role, amount, owners

    def lower(self, context):
        per_owner = self.amount / self.owners
        return KernelProgram(tuple(
            Stage(f"move{owner}", context.bind(Work(resources={self.role: per_owner})),
                  placement=context.placement(self.role, (owner,)))
            for owner in range(self.owners)
        ))


class ReadMoves(OwnerMoves):
    """Split GM transfers over owners and charge the same bytes to shared HBM."""

    def lower(self, context):
        per_owner = self.amount / self.owners
        work = Work(resources={"hbm": per_owner, self.role: per_owner})
        return KernelProgram(tuple(
            Stage(f"read{owner}", context.bind(work),
                  placement=context.placement(self.role, (owner,)))
            for owner in range(self.owners)
        ))


class Matrix:
    """Ordinary matrix work on the profile's declared precision peak."""

    def __init__(self, dtype, flops):
        self.dtype, self.flops = dtype, flops

    def lower(self, context):
        return KernelProgram((Stage(
            "mmad", context.bind(Work(matrix={self.dtype: self.flops})),
        ),))


@pytest.mark.parametrize("sku", A3)
def test_a3_hardware_parses_each_sku_clock_counts_and_recomputed_rates(sku):
    clock, aic, aiv, memory, hbm = A3[sku]
    device = sku_case(sku).machine.dies[0]

    assert device.memory_bytes == memory
    assert device.launch_us == 0
    assert device.compute_units == aic
    assert dict(device.rates)["hbm"] == pytest.approx(hbm)
    assert dict(device.buffers) == A3_BUFFERS
    assert dict(device.buffer_units)["ub"] == aiv
    assert device.units("mte1") == aic
    assert device.units("mte2_aic") == aic
    assert device.units("mte2_aiv") == aiv
    assert device.units("mte3") == aiv
    assert device.units("fixpipe") == aic

    expected = {
        "mte1": 128 * aic * clock,
        "mte2_aic": 32 * aic * clock,
        "mte2_aiv": 32 * aiv * clock,
        "mte3": 64 * aiv * clock,
        "fixpipe": 128 * aic * clock,
    }
    for role, rate in expected.items():
        assert dict(device.rates)[role] == pytest.approx(rate)
    modes = dict(device.service_modes)
    assert modes["mte1_l0a"][0] == modes["mte1_l0b"][0] == "mte1"
    assert modes["mte1_l0a"][1] == pytest.approx(256 * aic * clock)
    assert modes["mte1_l0b"][1] == pytest.approx(128 * aic * clock)

    peaks = dict(device.compute_peaks)
    # Direct arch2201 table 1: int4 k64, BF16 k16, FP32 k4, HF32 k8.
    assert peaks["int4"] == pytest.approx(peaks["bf16"] * 4)
    assert peaks["fp32"] == pytest.approx(peaks["bf16"] / 4)
    assert peaks["hf32"] == pytest.approx(peaks["bf16"] / 2)
    assert peaks["hf32"] == pytest.approx(2 * peaks["fp32"])


@pytest.mark.parametrize("sku", ["Ascend910_9391", "Ascend910_9381"])
def test_a3_same_mte1_owner_serializes_ab_while_different_owners_overlap(sku):
    clock, aic, _aiv, _memory, _hbm = A3[sku]
    case = sku_case(sku)
    a_bytes = b_bytes = 1_000_000_000
    same_owner = run(case, (op("a", "a", "stream-a"), op("b", "b", "stream-b")), {
        "a": OneMode("mte1_l0a", a_bytes, 0),
        "b": OneMode("mte1_l0b", b_bytes, 0),
    })
    different_owner = run(case, (op("a", "a", "stream-a"), op("b", "b", "stream-b")), {
        "a": OneMode("mte1_l0a", a_bytes, 0),
        "b": OneMode("mte1_l0b", b_bytes, 1),
    })

    expected_same = (a_bytes / 256 + b_bytes / 128) / clock * 1000
    expected_different = max(a_bytes / 256, b_bytes / 128) / clock * 1000
    assert same_owner["duration_ms"] == pytest.approx(expected_same)
    assert different_owner["duration_ms"] == pytest.approx(expected_different)
    assert same_owner["duration_ms"] > different_owner["duration_ms"]
    # The mode's native bytes convert to base-pipe units through the owner rate.
    assert same_owner["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        a_bytes / 2 + b_bytes)
    placements = {event["task"].split(":")[-2]: event["placement"]
                  for event in same_owner["events"] if event["task"].endswith(":move")}
    assert placements["a"] == placements["b"] == {"mte1@r0:s0": (0,)}


@pytest.mark.parametrize("sku", ["Ascend910_9381", "Ascend910_9372"])
def test_a3_read_and_write_paths_contend_on_one_shared_hbm(sku):
    _clock, aic, aiv, _memory, hbm = A3[sku]
    aic_bytes = aic * 1_000_000_000
    aiv_bytes = aiv * 1_000_000_000
    result = run(sku_case(sku), (
        op("aic-read", "aic-read", "read-aic"),
        op("aiv-read", "aiv-read", "read-aiv"),
        op("write", "write", "write"),
    ), {
        "aic-read": ReadMoves("mte2_aic", aic_bytes, aic),
        "aiv-read": ReadMoves("mte2_aiv", aiv_bytes, aiv),
        "write": ReadMoves("mte3", aiv_bytes, aiv),
    })
    total = aic_bytes + 2 * aiv_bytes
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(total)
    assert "mte2@r0:s0" not in result["execution"]["serviced"]
    assert result["duration_ms"] == pytest.approx(total / hbm * 1000)


def test_a3_capacity_and_intra_package_link_are_preserved():
    hardware = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend910_9391", "num_devices": 2})
    assert [device.memory_bytes for device in hardware.dies] == [68719476736] * 2
    assert hardware.dies[0].launch_us == 0
    assert dict(hardware.dies[0].buffers) == A3_BUFFERS
    links = {link.name: link for link in hardware.links}
    link = links["d2d_intra_package:0"]
    assert link.capacity_bytes_per_ms == pytest.approx(448 * 1e6)
    assert link.latency_us == pytest.approx(0.8)
    assert link.members == (0, 1)


@pytest.mark.parametrize("sku", A5)
def test_a5_hardware_uses_arch3510_units_modes_and_retains_sku_memory(sku):
    clock, aic, aiv, memory, hbm = A5[sku]
    device = sku_case(sku).machine.dies[0]

    assert device.memory_bytes == memory
    assert device.launch_us == 0
    assert device.compute_units == aic
    assert dict(device.rates)["hbm"] == pytest.approx(hbm)
    assert dict(device.buffers) == A5_BUFFERS
    assert dict(device.buffer_units)["ub"] == aiv

    expected = {
        "mte1": 256 * aic * clock,
        "mte2_aic": 80 * aic * clock,
        "mte2_aiv": 40 * aiv * clock,
        "mte3": 128 * aiv * clock,
        "fixpipe": 128 * aic * clock,
        "reg_data_writes": 2 * aiv * clock,
        "reg_ub_accesses": 2 * aiv * clock,
        "reg_ub_stores": aiv * clock,
        "reg_alu_lanes": 128 * aiv * clock,
        "reg_special_lanes": 32 * aiv * clock,
        "reg_reduce_lanes": 64 * aiv * clock,
        "reg_shuffle_lanes": 64 * aiv * clock,
    }
    for role, rate in expected.items():
        assert dict(device.rates)[role] == pytest.approx(rate)
        assert device.units(role) == (aic if role in {"mte1", "mte2_aic", "fixpipe"} else aiv)

    modes = dict(device.service_modes)
    assert modes["ub_copy"] == ("vector", pytest.approx(256 * aiv * clock))
    assert modes["reg_issue_slots"] == ("vector", pytest.approx(2 * aiv * clock))
    peaks = dict(device.compute_peaks)
    # Direct arch3510 table 2: BF16 k16, HF32 k8, native FP32 k1.
    assert peaks["fp32"] == pytest.approx(peaks["bf16"] / 16)
    assert peaks["hf32"] == pytest.approx(peaks["bf16"] / 2)
    assert peaks["hf32"] == pytest.approx(8 * peaks["fp32"])


@pytest.mark.parametrize("sku,ratio,divisor", [
    ("Ascend910_9391", 2.0, 4),
    ("Ascend950DT_9581", 8.0, 16),
])
def test_direct_fp32_vs_hf32_matrix_service_uses_architecture_ratios(sku, ratio, divisor):
    case = sku_case(sku)
    peaks = dict(case.machine.dies[0].compute_peaks)
    flops = 1e12
    fp32 = run(case, (op("fp32", "fp32"),), {"fp32": Matrix("fp32", flops)})
    hf32 = run(case, (op("hf32", "hf32"),), {"hf32": Matrix("hf32", flops)})

    assert peaks["fp32"] == pytest.approx(peaks["bf16"] / divisor)
    assert fp32["duration_ms"] == pytest.approx(flops / peaks["fp32"] * 1000)
    assert hf32["duration_ms"] == pytest.approx(flops / peaks["hf32"] * 1000)
    assert hf32["duration_ms"] == pytest.approx(fp32["duration_ms"] / ratio)
    # The previous A3 declaration used BF16/2 for fp32, which is HF32's rate.
    # Correct FP32 is now twice as slow as that old declaration on the same work.
    if divisor == 4:
        assert fp32["duration_ms"] == pytest.approx(
            4 * flops / peaks["bf16"] * 1000)
        assert hf32["duration_ms"] == pytest.approx(
            2 * flops / peaks["bf16"] * 1000)


@pytest.mark.parametrize("sku,expected_ms", [
    ("Ascend950DT_9581", 100e9 / 4.0e12 * 1000),
    ("Ascend950PR_9579", 100e9 / 1.6e12 * 1000),
])
def test_a5_same_work_scales_with_each_sku_hbm_not_9582_capacity(sku, expected_ms):
    case = sku_case(sku)
    total = 100_000_000_000
    result = run(case, (op("read", "read"),), {
        "read": ReadMoves("mte2_aic", total, 32),
    })
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(total)
    assert result["duration_ms"] == pytest.approx(expected_ms)


def test_a5_capacity_and_ub_mesh_link_are_preserved():
    hardware = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 2})
    assert [device.memory_bytes for device in hardware.dies] == [68719476736] * 2
    assert dict(hardware.dies[0].buffers) == A5_BUFFERS
    links = {link.name: link for link in hardware.links}
    link = links["d2d_intra_node:0"]
    assert link.capacity_bytes_per_ms == pytest.approx(392 * 1e6)
    assert link.latency_us == pytest.approx(1.8)
    assert link.members == (0, 1)


@pytest.mark.parametrize("sku", tuple(profiles()))
def test_undeclared_host_has_no_launch_service_and_explicit_declaration_is_additive(sku):
    raw = {
        "machine": {"backend": "ascend", "sku": sku},
        "model": MODEL, "workload": WORKLOAD, "runtime": {},
    }
    recipes = RecipeBook({"move": OneMode("vector", 1e6, 0)})
    operation = replace(op("vector-work", "move"), vector_ops=1e6)
    baseline = graph_costs(raw, (operation,), recipes=recipes,
                           isolated=False, include_submission=True)
    raw["machine"]["device"] = {"launch_us": 7}
    explicit = graph_costs(raw, (operation,), recipes=recipes,
                           isolated=False, include_submission=True)
    assert explicit["duration_ms"] - baseline["duration_ms"] == pytest.approx(.007)
    for role, work in baseline["execution"]["serviced"].items():
        if role.startswith("host"):
            assert work == 0
        else:
            assert explicit["execution"]["serviced"][role] == pytest.approx(work)
    assert explicit["execution"]["live_tasks"] == explicit["execution"]["live_programs"] == 0
