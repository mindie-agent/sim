"""Real resource checks for the retained MXFP4 AIV weight producer."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mxfp4_supply import (
    Config, UB_BYTES, config_from_group, ledger, make_op, recipe_book, tiles,
)
from mindie_sim.prediction.native_fusion import _groups

ROOT = Path(__file__).resolve().parents[1]


def case():
    return json.loads((ROOT / "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json").read_text())


def price(config, raw=None, ready=0):
    return operator_costs(raw or case(), (make_op(config),), recipes=recipe_book(),
                          input_ready_ms=ready)["operators"][0]


def test_real_balanced_frontend_supplies_integer_work_not_capacity_512():
    raw = case()
    raw["runtime"]["routing"] = {"policy": "balanced"}
    ops = target_graphs(raw, phase="decode", tokens=1, context=8, scope="body", resource_fusion=False)[0]
    declaration, members = next((d, m) for d, m in _groups(ops)
                                if d["role"] == "grouped_gate_up_situ_quant")
    config = config_from_group(declaration, members, allocated_rows=512)
    assert config.group_rows == (1, 1) + (0,) * 26
    result = price(config)
    assert result["operator"]["local_shape"] == [2, 6144, 3584]
    assert result["serviced"]["memory@r0:s0"] == 2 * 6144 * 3584 / 2
    assert result["serviced"]["mte3@r0:s0"] == 2 * 6144 * 3584
    assert result["serviced"]["vector@r0:s0"] == pytest.approx(2 * 6144 * 3584 * 3.5)
    assert "mte2_aic@r0:s0" not in result["serviced"]
    assert not result["basis"].get("invocation_models")
    assert result["live_tasks"] == 0


def test_allocated_rows_do_not_change_work_and_two_banks_do():
    one = Config(256, 128, (2,) + (0,) * 27, 2)
    padded = Config(256, 128, one.group_rows, 512)
    two = Config(256, 128, (1, 1) + (0,) * 26, 512)
    a, b, c = (price(x) for x in (one, padded, two))
    assert a["serviced"] == b["serviced"]
    assert a["duration_ms"] == b["duration_ms"]
    assert c["serviced"]["memory@r0:s0"] == 2 * a["serviced"]["memory@r0:s0"]
    assert ledger(one, 32)["scalar_group_list_loads"] == 2 * 28 * 96
    assert ledger(one, 32)["packed_weight_read_bytes"] == 128 * 256


@pytest.mark.parametrize("k,width,rows", [(64, 64, (1, 0, 2)), (320, 128, (0, 3, 0, 1)),
                                          (3584, 3072, (1, 1) + (0,) * 26)])
def test_source_tail_owner_work_and_late_readiness(k, width, rows):
    config = Config(k, width, rows, 512)
    result = price(config, ready=7)
    expected = sum(x > 0 for x in rows) * 2 * width * k
    assert result["serviced"]["memory@r0:s0"] == expected / 2
    assert result["serviced"]["mte3@r0:s0"] == expected
    assert result["serviced"]["vector@r0:s0"] == pytest.approx(expected * 3.5)
    assert not result["service_before_input"]
    assert not any(result["remaining_buffers"].values())
    assert result["live_tasks"] == 0
    for name, value in result["peak_buffers_per_unit"].items():
        if name.startswith("ub@"):
            assert value == UB_BYTES
    expected_tiles = [(e, row, n) for e, row in enumerate(rows) if row
                      for n in range(0, width, config.half_tile)]
    actual = {tile: owner for owner in range(32) for tile in tiles(config, owner, 32)}
    assert actual == {tile: index % 32 for index, tile in enumerate(expected_tiles)}


@pytest.mark.parametrize("role", ("hbm", "mte2_aiv", "mte3", "vector"))
def test_each_missing_supply_resource_is_physical_and_sensitive(role):
    raw = case()
    rates = {r: 1e18 for r in ("hbm", "mte2_aiv", "mte3", "vector")}
    values = []
    for scale in (.5, 1, 2):
        candidate = deepcopy(raw)
        candidate["machine"].setdefault("device", {})["rates"] = {**rates, role: 1e8 * scale}
        row = price(Config(256, 128, (1,), 512), candidate)
        values.append(row["duration_ms"])
        assert not row["basis"].get("invocation_models")
    assert values[0] > values[1] > values[2]
    assert values[0] / values[1] == pytest.approx(2, rel=1e-6)
    assert values[1] / values[2] == pytest.approx(2, rel=1e-6)


def test_empty_experts_do_not_fetch_a_bank_or_start_supply():
    row = price(Config(3584, 3072, (0,) * 28, 512), ready=3)
    assert not any(row["serviced"].values())
    assert row["duration_ms"] == 0
    assert not row["basis"].get("invocation_models")
