"""Device rates and inherited MTE1 direction modes must serve real work."""
import pytest

from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.case import Case
from test_ascend_pipeline_profile import (
    AIC_OWNERS, AIV_OWNERS, CLOCK_HZ, MODEL, WORKLOAD, OneMode, OwnerMoves,
    ReadMoves, op, pipeline_case, run,
)

AMOUNT = 1_000_000_000


def scaled_mode_duration(amount, mode_rate, units):
    return amount * units / mode_rate * 1000


def test_mte1_rate_override_rescales_inherited_direction_modes():
    default = pipeline_case()
    half = pipeline_case({"mte1_bw_bytes_per_s": 2.112e12})
    neutral_same = pipeline_case({
        "mte1_bw_bytes_per_s": 8.448e12, "rates": {"mte1": 2.112e12}})

    for case in (default, half, neutral_same):
        device = case.machine.dies[0]
        modes = dict(device.service_modes)
        assert modes["mte1_l0a"][0] == modes["mte1_l0b"][0] == "mte1"
    assert dict(default.machine.dies[0].rates)["mte1"] == 4.224e12
    assert dict(half.machine.dies[0].rates)["mte1"] == 2.112e12
    assert dict(neutral_same.machine.dies[0].rates)["mte1"] == 2.112e12
    # Same-layer neutral rates.mte1 wins over the native alias, and only the
    # inherited mte1 direction modes move with the base supply.
    assert dict(half.machine.dies[0].service_modes) == \
        dict(neutral_same.machine.dies[0].service_modes)
    assert dict(half.machine.dies[0].service_modes)["mte1_l0a"][1] == 4.224e12
    assert dict(half.machine.dies[0].service_modes)["mte1_l0b"][1] == 2.112e12

    for mode, default_rate in (("mte1_l0a", 8.448e12),
                               ("mte1_l0b", 4.224e12)):
        before = run(default, (op("move", "move"),),
                     {"move": OneMode(mode, AMOUNT, 0)})
        after = run(half, (op("move", "move"),),
                    {"move": OneMode(mode, AMOUNT, 0)})
        assert before["duration_ms"] == pytest.approx(
            scaled_mode_duration(AMOUNT, default_rate, AIC_OWNERS))
        assert after["duration_ms"] == pytest.approx(2 * before["duration_ms"])


def test_explicit_mode_record_wins_and_same_owner_competition_uses_new_modes():
    explicit = pipeline_case({
        "mte1_bw_bytes_per_s": 2.112e12,
        "service_modes": {"mte1_l0a": {"rate": 8.448e12}},
    })
    modes = dict(explicit.machine.dies[0].service_modes)
    # A rate-only explicit record inherits its physical resource and is not
    # rescaled by the layer's mte1 override.
    assert modes["mte1_l0a"] == ("mte1", 8.448e12)
    assert modes["mte1_l0b"] == ("mte1", 2.112e12)

    explicit_rate = run(explicit, (op("a", "a"),),
                        {"a": OneMode("mte1_l0a", AMOUNT, 0)})["duration_ms"]
    scaled_rate = run(explicit, (op("b", "b"),),
                      {"b": OneMode("mte1_l0b", AMOUNT, 0)})["duration_ms"]
    assert explicit_rate == pytest.approx(
        scaled_mode_duration(AMOUNT, 8.448e12, AIC_OWNERS))
    assert scaled_rate == pytest.approx(
        scaled_mode_duration(AMOUNT, 2.112e12, AIC_OWNERS))

    default = run(pipeline_case(), (
        op("a", "a", "stream-a"), op("b", "b", "stream-b")), {
        "a": OneMode("mte1_l0a", AMOUNT, 0),
        "b": OneMode("mte1_l0b", AMOUNT, 0),
    })
    overridden = run(explicit, (
        op("a", "a", "stream-a"), op("b", "b", "stream-b")), {
        "a": OneMode("mte1_l0a", AMOUNT, 0),
        "b": OneMode("mte1_l0b", AMOUNT, 0),
    })
    assert overridden["duration_ms"] == pytest.approx(
        scaled_mode_duration(AMOUNT, 8.448e12, AIC_OWNERS)
        + scaled_mode_duration(AMOUNT, 2.112e12, AIC_OWNERS))
    assert overridden["duration_ms"] > default["duration_ms"]
    assert default["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        AMOUNT / 2 + AMOUNT)
    # The unscaled explicit L0A mode retains its higher direction rate, so
    # its normalized demand is one quarter of the byte count; L0B keeps 1:1.
    assert overridden["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        AMOUNT / 4 + AMOUNT)


def test_per_device_mte1_override_scales_only_that_device():
    raw = {
        "machine": {"backend": "ascend", "sku": "Ascend910_9362",
                    "num_devices": 2, "ranks_per_node": 2,
                    "devices": [{"mte1_bw_bytes_per_s": 2.112e12}, {}]},
        "model": MODEL, "workload": WORKLOAD, "runtime": {"dp": 2},
    }
    case = Case.from_mapping(raw)
    first, second = case.machine.dies
    assert dict(first.rates)["mte1"] == 2.112e12
    assert dict(first.service_modes)["mte1_l0a"][1] == 4.224e12
    assert dict(second.rates)["mte1"] == 4.224e12
    assert dict(second.service_modes)["mte1_l0a"][1] == 8.448e12
    # Heterogeneous mode rates are not rejected: each replica serves on its
    # own physical owners.
    for replica in (0, 1):
        report = graph_costs(
            raw, {replica: (op("move", "move"),)},
            recipes=RecipeBook({"move": OneMode("mte1_l0a", AMOUNT, 0)}),
            isolated=False, trace_limit=200)
        expected = scaled_mode_duration(
            AMOUNT, dict(case.machine.dies[replica].service_modes)["mte1_l0a"][1],
            AIC_OWNERS)
        assert report["duration_ms"] == pytest.approx(expected)


def test_mte_bw_sets_split_roles_while_explicit_roles_and_hbm_owners_win():
    device = {"mte_bw_bytes_per_s": 1e12,
              "rates": {"mte2_aic": 2e12, "mte2_aiv": 4e12, "mte3": 8e12}}
    case = pipeline_case(device)
    parsed = case.machine.dies[0]
    rates = dict(parsed.rates)
    assert rates["mte2"] == 1e12
    assert (rates["mte2_aic"], rates["mte2_aiv"], rates["mte3"]) == (
        2e12, 4e12, 8e12)
    assert rates["hbm"] == 1.2e12
    assert (parsed.units("mte2_aic"), parsed.units("mte2_aiv"),
            parsed.units("mte3")) == (AIC_OWNERS, AIV_OWNERS, AIV_OWNERS)

    convenience = pipeline_case({"mte_bw_bytes_per_s": 1e12})
    convenience_rates = dict(convenience.machine.dies[0].rates)
    assert [convenience_rates[role] for role in
            ("mte2", "mte2_aic", "mte2_aiv", "mte3")] == [1e12] * 4
    report = run(convenience, (op("convenience", "convenience"),),
                 {"convenience": OneMode("mte2_aic", AMOUNT, 0)})
    assert report["duration_ms"] == pytest.approx(
        scaled_mode_duration(AMOUNT, 1e12, AIC_OWNERS))

    for role, units, rate in (("mte2_aic", AIC_OWNERS, 2e12),
                              ("mte2_aiv", AIV_OWNERS, 4e12),
                              ("mte3", AIV_OWNERS, 8e12)):
        report = run(case, (op(role, role),),
                     {role: OneMode(role, AMOUNT, 0)})
        assert report["duration_ms"] == pytest.approx(
            scaled_mode_duration(AMOUNT, rate, units))

    # HBM keeps its shared profile supply even when mte_bw is declared.
    hbm_only = run(case, (op("hbm", "hbm"),),
                   {"hbm": OwnerMoves("hbm", AMOUNT, AIC_OWNERS)})
    assert hbm_only["duration_ms"] == pytest.approx(
        AMOUNT / 1.2e12 * 1000)
    shared = run(case, (op("aic", "aic", "a"), op("aiv", "aiv", "b")), {
        "aic": ReadMoves("mte2_aic", 20 * AMOUNT, AIC_OWNERS),
        "aiv": ReadMoves("mte2_aiv", 40 * AMOUNT, AIV_OWNERS),
    })
    assert shared["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(
        60 * AMOUNT)
    assert shared["duration_ms"] == pytest.approx(
        60 * AMOUNT / 1.2e12 * 1000)

    # Entry points stay narrow: vector and HBM declarations leave owners and
    # unrelated rates alone.
    narrow = pipeline_case({"vector_ops_per_s": 5e12,
                            "hbm_bw_bytes_per_s": 2e12})
    device = narrow.machine.dies[0]
    assert dict(device.rates)["vector"] == 5e12
    assert dict(device.rates)["hbm"] == 2e12
    assert dict(device.rates)["mte1"] == 4.224e12
    assert device.units("vector") == 40


def test_vector_modes_follow_base_override_and_explicit_mode_rate_wins():
    def a5(device=None):
        return Case.from_mapping({
            "machine": {"backend": "ascend", "sku": "Ascend950DT_9582",
                        "device": device or {}},
            "model": MODEL, "workload": WORKLOAD, "runtime": {},
        })

    base = a5()
    base_vector = dict(base.machine.dies[0].rates)["vector"]
    half = a5({"vector_ops_per_s": base_vector / 2})
    explicit = a5({
        "rates": {"vector": base_vector / 2},
        "service_modes": {"ub_copy": {"rate": 27.0336e12}},
    })
    for role, amount in (("ub_copy", 1e9), ("reg_issue_slots", 1e6)):
        base_rate = dict(base.machine.dies[0].service_modes)[role][1]
        before = run(base, (op(role, role),), {role: OneMode(role, amount, 0)})
        after = run(half, (op(role, role),), {role: OneMode(role, amount, 0)})
        assert before["duration_ms"] == pytest.approx(amount * 64 / base_rate * 1000)
        assert after["duration_ms"] == pytest.approx(2 * before["duration_ms"])
        assert dict(half.machine.dies[0].service_modes)[role][1] == base_rate / 2
        chosen = run(explicit, (op(role, role),), {role: OneMode(role, amount, 0)})
        assert chosen["duration_ms"] == pytest.approx(
            before["duration_ms"] if role == "ub_copy" else after["duration_ms"])
    # Moving vector supply cannot move the unrelated physical MTE1 service.
    original_copy = run(base, (op("mte1", "mte1"),),
                        {"mte1": OneMode("mte1", AMOUNT, 0)})
    unchanged_copy = run(half, (op("mte1", "mte1"),),
                         {"mte1": OneMode("mte1", AMOUNT, 0)})
    assert original_copy["duration_ms"] == unchanged_copy["duration_ms"]


def test_machine_device_and_rank_overrides_reach_owner_service_without_double_scaling():
    raw = {
        "machine": {
            "backend": "ascend", "sku": "Ascend910_9362", "num_devices": 2,
            "mte1_bw_bytes_per_s": 1.056e12,
            "rates": {"vector": 1e12},
            "device": {"rates": {"mte1": 2.112e12}},
            "devices": [{"mte1_bw_bytes_per_s": 1.056e12}, {}],
        },
        "model": MODEL, "workload": WORKLOAD, "runtime": {"dp": 2},
    }
    case = Case.from_mapping(raw)
    assert Case.from_mapping(case.to_dict()) == case
    for replica, rate in enumerate((1.056e12, 2.112e12)):
        device = case.machine.dies[replica]
        assert dict(device.rates)["mte1"] == rate
        assert dict(device.rates)["vector"] == 1e12
        assert dict(device.service_modes)["mte1_l0a"] == ("mte1", 2 * rate)
        report = graph_costs(
            case, {replica: (op("move", "move"),)},
            recipes=RecipeBook({"move": OneMode("mte1_l0a", AMOUNT, 0)}),
            isolated=False)
        assert report["duration_ms"] == pytest.approx(
            scaled_mode_duration(AMOUNT, 2 * rate, AIC_OWNERS))
        assert report["execution"]["live_tasks"] == 0


def test_machine_level_modes_follow_rate_changes_and_keep_explicit_rates():
    base = Case.from_mapping({
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9582"},
        "model": MODEL, "workload": WORKLOAD, "runtime": {},
    })
    vector = dict(base.machine.dies[0].rates)["vector"]
    case = Case.from_mapping({
        "machine": {
            "backend": "ascend", "sku": "Ascend950DT_9582",
            "vector_ops_per_s": vector / 2,
            "service_modes": {"ub_copy": {"rate": 27.0336e12}},
        },
        "model": MODEL, "workload": WORKLOAD, "runtime": {},
    })
    assert Case.from_mapping(case.to_dict()) == case
    for role, amount in (("ub_copy", 1e9), ("reg_issue_slots", 1e6)):
        before = run(base, (op(role, role),), {role: OneMode(role, amount, 0)})
        after = run(case, (op(role, role),), {role: OneMode(role, amount, 0)})
        ratio = 1 if role == "ub_copy" else 2
        assert after["duration_ms"] == pytest.approx(ratio * before["duration_ms"])


def test_neutral_machine_default_supply_and_specific_pipe_aliases_reach_engine():
    raw = {
        "machine": {
            "memory_bytes": 1024**3, "compute_peaks": {"bf16": 1e12},
            "rates": {"hbm": 1e12, "vector": 1e12},
            "mte_bw_bytes_per_s": 1e12,
            "mte2_aic_bw_bytes_per_s": 2e12,
            "mte2_aiv_bw_bytes_per_s": 4e12,
            "mte3_bw_bytes_per_s": 8e12,
            "device": {"rates": {"mte2_aic": 3e12}},
        },
        "model": MODEL, "workload": WORKLOAD, "runtime": {},
    }
    case = Case.from_mapping(raw)
    assert Case.from_mapping(case.to_dict()) == case
    for role, rate in (("mte2", 1e12), ("mte2_aic", 3e12),
                       ("mte2_aiv", 4e12), ("mte3", 8e12)):
        report = run(case, (op(role, role),), {role: OneMode(role, AMOUNT, 0)})
        assert report["duration_ms"] == pytest.approx(AMOUNT / rate * 1000)
