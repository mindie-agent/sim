"""A3 Ascend910_9362 pipeline supply behavior on the common Engine.

Each test materializes the registered profile through ``Hardware.from_mapping``
and executes real ``Stage`` work. Expected durations are independently calculated
from the declared byte/cycle supply and the configured 1.65 GHz clock.
"""
import pytest

from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import OpSpec

CLOCK_HZ = 1.65e9
AIC_OWNERS = 20
AIV_OWNERS = 40
MODEL = {
    "name": "pipeline-geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
    "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16, "vocab_size": 16,
}
WORKLOAD = {"requests": [{"prompt_tokens": 1, "output_tokens": 1}]}


def pipeline_case(device=None):
    machine = {"backend": "ascend", "sku": "Ascend910_9362"}
    if device is not None:
        machine["device"] = device
    return Case.from_mapping({
        "machine": machine, "model": MODEL, "workload": WORKLOAD, "runtime": {},
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


class ReadMoves:
    """Split GM transfers over owners and charge the same bytes to shared HBM."""

    def __init__(self, role, amount, owners):
        self.role, self.amount, self.owners = role, amount, owners

    def lower(self, context):
        per_owner = self.amount / self.owners
        work = Work(resources={"hbm": per_owner, self.role: per_owner})
        return KernelProgram(tuple(
            Stage(f"read{owner}", context.bind(work),
                  placement=context.placement(self.role, (owner,)))
            for owner in range(self.owners)
        ))


class StoreMoves:
    """Local-only pipe controls; actual GM writes also require shared HBM."""

    def __init__(self, mte3_amount, mte3_owners, fixpipe_amount, fixpipe_owners):
        self.mte3_amount = mte3_amount
        self.mte3_owners = mte3_owners
        self.fixpipe_amount = fixpipe_amount
        self.fixpipe_owners = fixpipe_owners

    def lower(self, context):
        stages = []
        for role, amount, owners in (
            ("mte3", self.mte3_amount, self.mte3_owners),
            ("fixpipe", self.fixpipe_amount, self.fixpipe_owners),
        ):
            per_owner = amount / owners
            stages.extend(
                Stage(f"{role}{owner}", context.bind(Work(resources={role: per_owner})),
                      placement=context.placement(role, (owner,)))
                for owner in range(owners)
            )
        return KernelProgram(tuple(stages))


def test_direction_modes_share_one_mte1_owner_and_only_overlap_across_owners():
    case = pipeline_case()
    device = case.machine.dies[0]
    modes = dict(device.service_modes)
    assert modes["mte1_l0a"][0] == modes["mte1_l0b"][0] == "mte1"
    assert device.units("mte1") == AIC_OWNERS
    assert dict(device.rates)["mte1"] == 128 * AIC_OWNERS * CLOCK_HZ

    a_bytes = b_bytes = 1_000_000_000
    same_owner = run(case, (
        op("a", "a", "stream-a"), op("b", "b", "stream-b"),
    ), {
        "a": OneMode("mte1_l0a", a_bytes, 0),
        "b": OneMode("mte1_l0b", b_bytes, 0),
    })
    different_owner = run(case, (
        op("a", "a", "stream-a"), op("b", "b", "stream-b"),
    ), {
        "a": OneMode("mte1_l0a", a_bytes, 0),
        "b": OneMode("mte1_l0b", b_bytes, 1),
    })

    expected_same = (a_bytes / 256 + b_bytes / 128) / CLOCK_HZ * 1000
    expected_different = max(a_bytes / 256, b_bytes / 128) / CLOCK_HZ * 1000
    assert same_owner["duration_ms"] == pytest.approx(expected_same)
    assert different_owner["duration_ms"] == pytest.approx(expected_different)
    # Treating L0A/L0B as independent unlimited pipes would give the smaller
    # max() duration even on one owner.
    assert same_owner["duration_ms"] > different_owner["duration_ms"]
    assert same_owner["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        a_bytes / 2 + b_bytes)
    moves = {event["task"].split(":")[-2]: event for event in same_owner["events"]
             if event["task"].endswith(":move")}
    assert moves["a"]["placement"] == moves["b"]["placement"] == {"mte1@r0:s0": (0,)}
    different = {event["task"].split(":")[-2]: event for event in different_owner["events"]
                 if event["task"].endswith(":move")}
    assert {different["a"]["placement"]["mte1@r0:s0"],
            different["b"]["placement"]["mte1@r0:s0"]} == {(0,), (1,)}


def test_fixed_mte1_owner_cannot_borrow_idle_owners_but_full_shape_scales():
    total_bytes = 4.224e9
    one_owner = run(pipeline_case(), (op("one", "one"),), {
        "one": OwnerMoves("mte1", total_bytes, 1),
    })
    full_shape = run(pipeline_case(), (op("full", "full"),), {
        "full": OwnerMoves("mte1", total_bytes, AIC_OWNERS),
    })

    expected_one = total_bytes / (128 * CLOCK_HZ) * 1000
    expected_full = total_bytes / (AIC_OWNERS * 128 * CLOCK_HZ) * 1000
    assert one_owner["duration_ms"] == pytest.approx(expected_one)
    assert full_shape["duration_ms"] == pytest.approx(expected_full)
    assert one_owner["duration_ms"] == pytest.approx(AIC_OWNERS * expected_full)
    assert one_owner["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(total_bytes)
    assert full_shape["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(total_bytes)


def test_concurrent_aic_aiv_reads_charge_one_hbm_resource_and_owned_paths():
    bytes_per_owner = 1_000_000_000
    aic_bytes = AIC_OWNERS * bytes_per_owner
    aiv_bytes = AIV_OWNERS * bytes_per_owner
    result = run(pipeline_case(), (
        op("aic", "aic", "stream-aic"), op("aiv", "aiv", "stream-aiv"),
    ), {
        "aic": ReadMoves("mte2_aic", aic_bytes, AIC_OWNERS),
        "aiv": ReadMoves("mte2_aiv", aiv_bytes, AIV_OWNERS),
    })

    serviced = result["execution"]["serviced"]
    assert serviced["memory@r0:s0"] == pytest.approx(aic_bytes + aiv_bytes)
    assert serviced["mte2_aic@r0:s0"] == pytest.approx(aic_bytes)
    assert serviced["mte2_aiv@r0:s0"] == pytest.approx(aiv_bytes)
    assert "mte2@r0:s0" not in serviced
    assert [key for key, amount in serviced.items()
            if amount and key.startswith("memory@")] == ["memory@r0:s0"]

    expected_hbm = (aic_bytes + aiv_bytes) / 1.2e12 * 1000
    local_path_bound = aic_bytes / 1.056e12 * 1000
    assert aiv_bytes / 2.112e12 * 1000 == pytest.approx(local_path_bound)
    assert result["duration_ms"] == pytest.approx(expected_hbm)
    assert result["duration_ms"] > local_path_bound


def test_aiv_store_and_fixpipe_units_and_explicit_device_overrides_execute():
    amount = 4.224e9
    one_owner = run(pipeline_case(), (op("single", "single"),), {
        "single": StoreMoves(amount, 1, amount, 1),
    })
    full_shape = run(pipeline_case(), (op("full", "full"),), {
        "full": StoreMoves(amount, AIV_OWNERS, amount, AIC_OWNERS),
    })
    assert one_owner["duration_ms"] == pytest.approx(
        amount / (64 * CLOCK_HZ) * 1000)
    assert full_shape["duration_ms"] == pytest.approx(
        amount / (64 * AIV_OWNERS * CLOCK_HZ) * 1000)
    assert full_shape["execution"]["serviced"]["mte3@r0:s0"] == pytest.approx(amount)
    assert full_shape["execution"]["serviced"]["fixpipe@r0:s0"] == pytest.approx(amount)

    overridden_case = pipeline_case({
        "rates": {"mte3": 8.448e12},
        "resource_units": {"fixpipe": 10},
    })
    overridden = run(overridden_case, (op("overridden", "overridden"),), {
        "overridden": StoreMoves(amount, 1, amount, 1),
    })
    # MTE3 aggregate rate doubles to 8.448e12; Fixpipe stays 4.224e12 but has
    # ten owners. One owner now takes 20 ms for MTE3 and 10 ms for Fixpipe.
    assert overridden["duration_ms"] == pytest.approx(
        amount / (128 * CLOCK_HZ) * 1000)
    device = overridden_case.machine.dies[0]
    assert device.units("mte3") == AIV_OWNERS
    assert device.units("fixpipe") == 10
    assert dict(device.rates)["mte3"] == 8.448e12
    assert dict(device.rates)["fixpipe"] == 4.224e12

    assumptions = dict(pipeline_case().machine.assumptions)
    assert {"mte1", "mte2", "mte3", "fixpipe", "pipeline_provenance",
            "source_discrepancy"} <= assumptions.keys()


def test_gm_reads_and_both_output_paths_share_the_same_memory_supply():
    amount = 1_200_000_000
    result = run(pipeline_case(), (
        op("read", "read", "read"),
        op("store", "store", "store"),
        op("fix", "fix", "fix"),
    ), {
        "read": ReadMoves("mte2_aic", amount, AIC_OWNERS),
        "store": ReadMoves("mte3", amount, AIV_OWNERS),
        "fix": ReadMoves("fixpipe", amount, AIC_OWNERS),
    })
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(3 * amount)
    assert result["duration_ms"] == pytest.approx(3 * amount / 1.2e12 * 1000)


def test_ordinary_selected_prefill_uses_profile_copy_roles_and_input_readiness():
    from mindie_sim import target_operators

    raw = {
        "machine": {"backend": "ascend", "sku": "Ascend910_9362"},
        "model": MODEL,
        "workload": WORKLOAD,
        "runtime": {"kernels": {"recipes": {
            "flash_attention_prefill": {
                "kind": "staged_gqa_prefill", "query_tile": 4, "key_tile": 4,
            },
        }}},
    }
    attention = tuple(operator for operator in target_operators(
        raw, phase="prefill", tokens=5, context=3) if operator.family == "attention")
    assert len(attention) == 1
    result = graph_costs(
        raw, attention, input_ready_ms=7,
        external_inputs={dependency: 11 for operator in attention
                         for dependency in operator.deps},
        isolated=False, trace_limit=200,
    )
    serviced = result["execution"]["serviced"]
    # A single Q head has 4x7 and 1x8 causal rectangles: two K/V sweeps.
    query_bytes, kv_bytes, output_bytes = 5 * 8 * 2, 2 * (7 + 8) * 8 * 2, 5 * 8 * 2
    assert serviced["memory@r0:s0"] == pytest.approx(
        query_bytes + kv_bytes + output_bytes)
    assert serviced["mte2_aic@r0:s0"] == pytest.approx(query_bytes + kv_bytes)
    assert serviced["mte3@r0:s0"] == pytest.approx(output_bytes)
    assert serviced["mte1@r0:s0"] > 0
    assert "mte2@r0:s0" not in serviced
    assert all(event["start_ms"] >= 11 for event in result["events"] if event["demand"])
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
