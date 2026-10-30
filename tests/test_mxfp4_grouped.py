"""Source layout, finite supply and actual Engine controls for MXA8W4 NZ."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.mxfp4_geometry import Config, columns, owner_tiles, tile_work
from mindie_sim.prediction.mxfp4_grouped import UB_BYTES, recipe
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.common import ModelingError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_case():
    raw = json.loads((ROOT / "examples/cases/v41-a5-dp4-body-current.json").read_text())
    raw["runtime"]["routing"] = {"policy": "balanced"}
    raw["runtime"].setdefault("kernels", {}).setdefault("families", {})["grouped_gemm"] = {
        "kind": "ascend_mxa8w4_nznk"}
    return raw


@pytest.fixture(scope="module")
def gate(source_case):
    return next(op for op in target_graphs(source_case, phase="decode", tokens=1,
                                          context=8, scope="body", resource_fusion=False)[0]
                if op.family == "grouped_gemm")


def physical_op(gate, rows=(1,), n=256, k=512):
    meta = {**dict(gate.metadata), "matrix_group_rows": rows,
            "matrix_weight_bank_groups": len(rows), "matrix_groups": sum(x > 0 for x in rows)}
    shape = (sum(rows), n, k)
    return replace(gate, local_shape=shape, semantic_shape=shape,
                   launch_shape=shape, metadata=tuple(meta.items()))


def price(raw, op, compact=True, ready=0, mte1_mode="base"):
    return operator_costs(raw, (op,), recipes=RecipeBook(families={"grouped_gemm": recipe(compact=compact, mte1_mode=mte1_mode)}),
                          input_ready_ms=ready)["operators"][0]


def test_ordinary_v41_frontend_selects_source_and_preserves_valid_rows(source_case):
    ops = target_graphs(source_case, phase="decode", tokens=1, context=8,
                        scope="body", resource_fusion=False)[0]
    group = [op for op in ops if op.family == "grouped_gemm"][:2]
    assert len(group) == 2
    result = operator_costs(source_case, group)["operators"]
    for op, row in zip(group, result):
        declared = dict(op.metadata)["matrix_group_rows"]
        detail = row["operator"]["metadata"]["program_execution"]["mxa8w4"]
        assert tuple(detail["group_rows"]) == declared
        assert detail["valid_rows"] == sum(declared) == 6
        assert detail["active_banks"] == 6
        assert detail["logical_flops"] == op.flops
        assert detail["issued_cube_flops"] == op.flops * 16
        assert row["operator"]["matrix_flops"] == {"fp8_e4m3": op.flops * 16}
        assert row["live_tasks"] == 0
    # The source recipe consumes only the matrix; caller activation and dynamic
    # quantization nodes stay in the ordinary graph, with their original deps.
    assert any(op.family == "swiglu" for op in ops)
    assert any(op.family == "quantize" for op in ops)


@pytest.mark.parametrize("rows,n,k", [((1, 0, 2), 320, 320), ((0, 257, 0, 3), 544, 448),
                                         ((17,), 4608, 5120), ((1, 1), 5120, 2304)])
def test_tail_work_is_conserved_on_real_resources(source_case, gate, rows, n, k):
    config = Config(k, n, rows)
    expected = {}
    flops = 0
    for owner in range(32):
        for *_, m, nt in owner_tiles(config, owner, 32):
            work = tile_work(config, m, nt)
            flops += work.matrix["fp8_e4m3"]
            for role, amount in work.resources.items():
                expected[role] = expected.get(role, 0) + amount
    expected["hbm"] += 64 * 256
    expected["mte2_aiv"] += 64 * 256
    row = price(source_case, physical_op(gate, rows, n, k), ready=7)
    for role, amount in expected.items():
        physical = "memory" if role == "hbm" else role
        if role == "reg_issue_slots":
            # Profile: vector64 FP32 lanes/cycle and two Reg issue slots/cycle.
            physical, amount = "vector", amount * 32
        assert row["serviced"][physical + "@r0:s0"] == pytest.approx(amount)
    assert row["operator"]["flops"] == flops
    assert not row["service_before_input"]
    assert row["live_tasks"] == 0
    assert not any(row["remaining_buffers"].values())
    assert row["peak_buffers_per_unit"]["ub@r0:s0:unit0"] == UB_BYTES


def test_small_cube_issue_does_not_invent_weight_banks(source_case, gate):
    one = price(source_case, physical_op(gate, (1, 0), 128, 64))
    sixteen = price(source_case, physical_op(gate, (16, 0), 128, 64))
    a = one["operator"]["metadata"]["program_execution"]["mxa8w4"]
    b = sixteen["operator"]["metadata"]["program_execution"]["mxa8w4"]
    assert a["issued_cube_flops"] == b["issued_cube_flops"]
    assert a["logical_flops"] * 16 == b["logical_flops"]
    assert one["serviced"]["mte2_aiv@r0:s0"] == sixteen["serviced"]["mte2_aiv@r0:s0"]
    # One 64-deep weight panel is 4096B, scale read256B; fixed 128 scale
    # columns publish16384B plus8192B expanded weight, without extra HBM writes.
    assert one["serviced"]["mte3@r0:s0"] == 16384 + 8192


@pytest.mark.parametrize("role", ["hbm", "mte2_aic", "mte2_aiv", "mte3", "mte1", "vector", "fixpipe"])
def test_each_pipeline_axis_changes_real_execution(source_case, gate, role):
    durations = []
    for factor in (.5, 1, 2):
        raw = deepcopy(source_case)
        raw["machine"]["device"]["rates"] = {
            **{r: 1e18 for r in ("hbm", "mte2_aic", "mte2_aiv", "mte3", "mte1", "vector", "fixpipe")},
            role: 1e8 * factor}
        row = price(raw, physical_op(gate, (1,) * 4, 256, 512))
        durations.append(row["duration_ms"])
    assert durations[0] > durations[1] > durations[2]


def test_detailed_and_compressed_owner_work_and_no_double_floor(source_case, gate):
    op = physical_op(gate, (1,) * 4, 8192, 512)
    detailed, compact = (price(source_case, op, compact=flag) for flag in (False, True))
    assert detailed["duration_ms"] == pytest.approx(compact["duration_ms"], rel=1e-11)
    assert detailed["serviced"] == pytest.approx(compact["serviced"])
    raw = deepcopy(source_case)
    raw["runtime"]["kernels"]["invocations"] = [
        {"match": {"family": "grouped_gemm"}, "startup_ms": .003, "minimum_ms": 10,
         "provenance": {"basis": "test_explicit_kernel_boundary"}}]
    a, b = (price(raw, op, compact=compact) for compact in (False, True))
    assert a["serviced"] == pytest.approx(b["serviced"])
    assert a["duration_ms"] == b["duration_ms"] == pytest.approx(10)
    assert len(a["basis"]["invocation_models"]) == len(b["basis"]["invocation_models"]) == 1


def test_l1_reuse_waits_for_cube_and_cube_waits_for_aiv_publish(gate):
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.mxfp4_grouped import build
    from mindie_sim.prediction.program import RecipeContext
    op = physical_op(gate, (1,), 128, 2048)
    register_roles = ("reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_shuffle_lanes")
    roles = {role: 1e9 for role in ("hbm", "mte1", "mte2_aic", "mte2_aiv", "mte3", "vector", "fixpipe", *register_roles)}
    roles["cube"] = 1
    units = {role: 2 if role in ("vector", "mte2_aiv", "mte3", *register_roles) else 1 for role in roles}
    pools = {"ub": 256*1024, "l1": 512*1024, "l0a": 64*1024, "l0b": 64*1024, "l0c": 256*1024}
    geometry = Geometry(1, units, pools, {p: 2 if p == "ub" else 1 for p in pools})
    plan = build(op, geometry, compact=False, compact_panels=False)
    engine = Engine([Resource(role, rate, units=units[role]) for role, rate in roles.items()],
                    pools=pools, pool_units=geometry.buffer_units, trace_limit=1000)
    context = RecipeContext(plan.work.apply(op), {}, {r: r for r in roles}, roles, 1,
                            resource_units=units, pool_names={p: p for p in pools},
                            compute_peaks={"fp8_e4m3": 1e6},
                            service_modes={"reg_issue_slots": ("vector", roles["vector"] * 1000 / 32)})
    ready = engine.submit("input", release=7)
    complete = plan.lower(context).start(engine, "source", ready)
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    events = list(engine.trace)
    def matches(panel, suffix):
        return [e for e in events if f":panel{panel}:" in e["task"] and e["task"].endswith(suffix)]
    for panel in range(4):
        published = max(matches(panel, f":publish{sub}")[0]["end_ms"] for sub in (0, 1))
        assert min(e["start_ms"] for e in matches(panel, ":operand")) >= published
        for sub in (0, 1):
            assert matches(panel, f":packed_read{sub}")[0]["end_ms"] <= matches(panel, f":widen{sub}")[0]["start_ms"]
        if panel >= 2:
            assert matches(panel, ":packed_read0")[0]["start_ms"] >= max(e["end_ms"] for e in matches(panel-2, ":cube"))
    assert not engine._tasks and not engine.allocations


def test_controller_resplit_and_rolling_owner_are_not_generic_tiles():
    c = Config(5120, 4608, (1, 0, 2) + (0,) * 45)
    assert [n for _, n in columns(c, 32)] == [128] * 16 + [160] * 16
    all_tiles = [(owner, tile) for owner in range(32) for tile in owner_tiles(c, owner, 32)]
    assert len(all_tiles) == 64
    assert all(tile[0] in (0, 2) for _, tile in all_tiles)
    c = Config(320, 544, (1, 0, 2) + (0,) * 45)
    expected = [(e, 0, col, rows, n) for e, rows in enumerate(c.rows) if rows
                for col, n in columns(c, 32)]
    actual = {tile: owner for owner in range(32) for tile in owner_tiles(c, owner, 32)}
    assert actual == {tile: i % 32 for i, tile in enumerate(expected)}


def test_source_register_counts_do_not_charge_fp32_work_again(source_case, gate):
    row = price(source_case, physical_op(gate, (1,), 128, 64))
    info = row["operator"]["metadata"]["program_execution"]["mxa8w4"]
    counts = info["register_inventory"]["instructions"]
    # 8192 widened B8 values in32 full registers, all on source sub-AIV0;
    # sub-AIV1 still runs the two masks and three constants before a zero loop.
    assert counts["ShiftRightB8"]["calls"] == 64
    assert counts["ShiftRightB8"]["active_lanes"] == 16384
    assert counts["StoreB8"]["calls"] == 32
    assert counts["StoreB8"]["write_bytes"] == 8192
    assert counts["GatherUB_B16"]["calls"] == 8
    assert info["register_capacity_work"]["reg_issue_slots"] == 302
    assert row["operator"]["vector_ops"] == 0
    assert row["serviced"]["vector@r0:s0"] == pytest.approx(302 * 32)


def test_explicit_directional_mte1_binds_both_operand_modes(source_case, gate):
    raw = deepcopy(source_case)
    device = raw["machine"]["device"]
    device["rates"]["mte1"] = 1e9
    device["service_modes"] = {
        "mte1_l0a": {"resource": "mte1", "rate": 1e9},
        "mte1_l0b": {"resource": "mte1", "rate": .5e9},
    }
    op = physical_op(gate, (1,), 128, 64)
    base = price(raw, op)
    mode = price(raw, op, mte1_mode="directional")
    assert base["serviced"]["mte1@r0:s0"] == (16 + 128) * 66
    assert mode["serviced"]["mte1@r0:s0"] == (16 + 2 * 128) * 66
    assert mode["duration_ms"] > base["duration_ms"]
    assert mode["serviced"]["memory@r0:s0"] == base["serviced"]["memory@r0:s0"]


def test_fp8_shared_and_fused_situ_are_not_silently_rebound(source_case, gate):
    for change in ({"weight_format": "fp8_e4m3"}, {"output_format": "fp8_e4m3"}):
        op = replace(gate, metadata=tuple({**dict(gate.metadata), **change}.items()))
        with pytest.raises(ModelingError):
            price(source_case, op)
