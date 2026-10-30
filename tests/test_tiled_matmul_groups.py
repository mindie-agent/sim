"""Independent equal-group geometry, traffic and residency oracles for TiledMatmul."""
from dataclasses import replace
import itertools
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import dtype_bytes


ROOT = Path(__file__).resolve().parents[1]


def op_for(total_m, n, k, *, groups=1, quant=None, vector_ops=0):
    op = gemm_op(
        op_id="grouped", deps=(), stage=0, layer=0, requests=(),
        m=total_m, n=n, k=k, weight_count=groups,
        quant=quant or QuantSpec("bf16"), category="dense_mlp", phase="prefill",
    )
    return replace(op, vector_ops=float(vector_ops)) if vector_ops else op


def setup(op, *, cores=2, pools=None, copy_rate=16, mte3_rate=4096,
          trace_limit=0):
    rates = {"cube": 512, "hbm": 4096, "mte2": copy_rate,
             "mte3": mte3_rate, "vector": 4096}
    pools = pools or {"l1": 1 << 20, "l0c": 1 << 20}
    units = {name: (1 if name == "hbm" else cores) for name in rates}
    engine = Engine(
        [Resource(name, rate, units=units[name]) for name, rate in rates.items()],
        pools=pools, pool_units={name: cores for name in pools},
        trace_limit=trace_limit,
    )
    work = {"cube": op.flops, "hbm": op.hbm_bytes,
            "mte2": op.mte2_bytes, "mte3": op.mte3_bytes}
    if op.vector_ops:
        work["vector"] = op.vector_ops
    context = RecipeContext(
        op, work, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools}, pool_capacities=pools,
        resource_units=units,
    )
    return engine, context


def run(recipe, op, *, cores=2, pools=None, copy_rate=16, mte3_rate=4096,
        release=0, trace_limit=0):
    engine, context = setup(op, cores=cores, pools=pools, copy_rate=copy_rate,
                            mte3_rate=mte3_rate, trace_limit=trace_limit)
    done = recipe.lower(context).start(
        engine, "matrix", engine.submit("ready", release=release))
    if release:
        engine.advance(horizon=release - 1)
        assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values())
    summary = engine.summary()
    assert summary["live_tasks"] == summary["live_programs"] == 0
    return engine, done.time - release


def loop_oracle(op, recipe, *, reuse_inputs, reuse_weights):
    """Enumerate scalar BF16 addresses, including repeated private copies."""
    total_m, n, k = op.local_shape[-3:]
    metadata = dict(op.metadata)
    groups = int(metadata.get("matrix_groups", 1))
    m = total_m // groups
    assert metadata["weight_format"] == metadata["activation_format"] == "bf16"
    output_width = dtype_bytes(metadata.get("output_format", "bf16"))
    expected = {"hbm": 0.0, "mte2": 0.0, "mte3": 0.0,
                "cube": 0.0, "vector": op.vector_ops}
    seen_a, seen_b, seen_output = set(), set(), set()
    copied_a = copied_b = 0
    max_panel = max_accumulator = 0.0
    for group in range(groups):
        for row in range(0, m, recipe.tile_m):
            rows = range(row, min(m, row + recipe.tile_m))
            for column in range(0, n, recipe.tile_n):
                columns = range(column, min(n, column + recipe.tile_n))
                outputs = {(group, r, c) for r in rows for c in columns}
                seen_output.update(outputs)
                max_accumulator = max(
                    max_accumulator, len(outputs) * max(4, output_width))
                for offset in range(0, k, recipe.tile_k):
                    reduction = range(offset, min(k, offset + recipe.tile_k))
                    a = {(group, r, p) for r in rows for p in reduction}
                    b = {(group, p, c) for p in reduction for c in columns}
                    seen_a.update(a)
                    seen_b.update(b)
                    copied_a += len(a)
                    copied_b += len(b)
                    expected["cube"] += 2 * len(outputs) * len(reduction)
                    max_panel = max(max_panel, 2 * (len(a) + len(b)))
    expected["mte2"] = 2 * (copied_a + copied_b)
    expected["mte3"] = len(seen_output) * output_width
    expected["hbm"] = 2 * ((len(seen_a) if reuse_inputs else copied_a)
                           + (len(seen_b) if reuse_weights else copied_b)) + expected["mte3"]
    return expected, max_panel, max_accumulator


def assert_services(engine, expected):
    assert set(engine.serviced) == set(expected)
    for role, amount in expected.items():
        assert engine.serviced[role] == pytest.approx(amount, rel=1e-12, abs=1e-9)


def test_ragged_group_row_column_k_loop_and_all_reuse_flags():
    geometry = (
        (5, 1, 4, 3), (12, 3, 7, 5), (20, 4, 5, 3), (7, 7, 5, 5),
    )
    for (total_m, groups, n, k), cores in itertools.product(geometry, (1, 3, 5)):
        op = op_for(total_m, n, k, groups=groups, vector_ops=2 * total_m * n)
        for reuse_inputs, reuse_weights in itertools.product((False, True), repeat=2):
            recipe = TiledMatmul(2, 3, 2, slots=1, reuse_inputs=reuse_inputs,
                                 reuse_weights=reuse_weights)
            expected, max_panel, max_accumulator = loop_oracle(
                op, recipe, reuse_inputs=reuse_inputs, reuse_weights=reuse_weights)
            engine, _ = run(recipe, op, cores=cores)
            assert_services(engine, expected)
            assert max(engine.peak_unit_allocated.get(("l1", unit), 0)
                       for unit in range(cores)) == pytest.approx(max_panel)
            assert max(engine.peak_unit_allocated.get(("l0c", unit), 0)
                       for unit in range(cores)) == pytest.approx(max_accumulator)


def test_group_count_does_not_multiply_live_pipeline_state():
    recipe = TiledMatmul(4, 4, 4, slots=2)
    small, _ = run(recipe, op_for(16, 4, 4, groups=4), cores=2)
    large, _ = run(recipe, op_for(64, 4, 4, groups=16), cores=2)
    assert large.peak_tasks == small.peak_tasks
    assert large.peak_allocated == small.peak_allocated
    assert large.serviced["cube"] == pytest.approx(4 * small.serviced["cube"])
    assert large.serviced["mte2"] == pytest.approx(4 * small.serviced["mte2"])


def test_missing_groups_is_the_one_group_control():
    with_metadata = op_for(8, 6, 4, groups=1)
    missing = replace(with_metadata, metadata=tuple(
        (key, value) for key, value in with_metadata.metadata
        if key != "matrix_groups"))
    recipe = TiledMatmul(3, 4, 2, slots=2)
    first, _ = run(recipe, with_metadata, cores=3)
    second, _ = run(recipe, missing, cores=3)
    assert first.serviced == pytest.approx(second.serviced)
    assert first.peak_allocated == second.peak_allocated
    expected, _, _ = loop_oracle(with_metadata, recipe, reuse_inputs=True,
                                 reuse_weights=True)
    assert_services(first, expected)
    assert first.serviced["hbm"] == pytest.approx(with_metadata.hbm_bytes)


def test_aggregate_expert_moments_keep_the_existing_buffer_approximation():
    op = op_for(16, 4, 4)
    weight = 4096
    inputs, outputs = 16 * 4 * 2, 16 * 4 * 2
    meta = {**dict(op.metadata), "packed_weight_bytes": weight,
            "matrix_mapping_domain": "aggregate_experts"}
    op = replace(op, metadata=tuple(meta.items()), mte2_bytes=inputs + weight,
                 hbm_bytes=inputs + weight + outputs)
    engine, _ = run(TiledMatmul(4, 4, 2, slots=1), op, cores=2,
                    pools={"l1": 32, "l0c": 64})
    assert engine.serviced["hbm"] == pytest.approx(inputs + weight + outputs)
    assert engine.serviced["mte2"] == pytest.approx(inputs + 4 * weight)
    assert max(engine.peak_unit_allocated[("l1", owner)] for owner in range(2)) == 32


def test_zero_m_retains_the_empty_grouped_program():
    op = op_for(0, 4, 3, groups=2)
    engine, context = setup(op, cores=2)
    done = TiledMatmul().lower(context).start(engine, "matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    assert not any(engine.serviced.values())
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_explicit_invalid_group_domains_remain_visible():
    uneven = op_for(10, 4, 2, groups=4)
    invalid = replace(uneven, metadata=tuple(
        (key, 0 if key == "matrix_groups" else value)
        for key, value in uneven.metadata))
    negative = replace(uneven, metadata=tuple(
        (key, -3 if key == "matrix_groups" else value)
        for key, value in uneven.metadata))
    for op in (uneven, invalid, negative):
        engine, context = setup(op, cores=2)
        with pytest.raises(ValueError, match="matrix_groups_require_equal_integer_rows"):
            TiledMatmul().lower(context)
        assert engine.summary()["live_tasks"] == 0


def test_grouped_input_ready_delays_service_and_releases_all_state():
    op = op_for(12, 5, 7, groups=3, vector_ops=12 * 5)
    recipe = TiledMatmul(3, 2, 3, slots=2)
    engine, _ = run(recipe, op, cores=3, release=7, trace_limit=5000)
    assert all(row["start_ms"] >= 7 for row in engine.trace if row["demand"])
    expected, _, _ = loop_oracle(op, recipe, reuse_inputs=True, reuse_weights=True)
    assert_services(engine, expected)


def test_packed_weight_and_scales_set_the_exact_panel_allocation():
    op = op_for(6, 4, 6, groups=3, quant=quant_preset("w8a8_int8"))
    recipe = TiledMatmul(2, 4, 3, slots=1)
    metadata = dict(op.metadata)
    weight = metadata["packed_weight_bytes"]
    inputs = op.mte2_bytes - weight
    # Three equal groups, one M/N tile each, two equal K panels. Row and column
    # metadata belong to the output epilogue, not the private K operand panels.
    epilogue = metadata["matrix_epilogue"]
    output_scales = epilogue["row_scale_bytes"] + epilogue["column_scale_bytes"]
    expected_panel = (inputs + weight - output_scales) / 6
    assert inputs == 6 * 6 + metadata["activation_scale_bytes"]
    assert expected_panel == 18
    engine, _ = run(recipe, op, cores=3)
    assert max(engine.peak_unit_allocated.get(("l1", unit), 0)
               for unit in range(3)) == pytest.approx(expected_panel)
    assert engine.serviced["vector"] == 4 * 6 * 4
    assert engine.serviced["hbm"] == pytest.approx(op.hbm_bytes)


def test_one_panel_capacity_holds_the_operand_until_compute_finishes():
    op = op_for(16, 4, 8, groups=4)
    tile = TiledMatmul(4, 4, 4, slots=2)
    expected, panel, _ = loop_oracle(op, tile, reuse_inputs=True, reuse_weights=True)
    roomy, roomy_ms = run(tile, op, cores=2, pools={"l1": 1 << 20, "l0c": 1 << 20})
    narrow, narrow_ms = run(tile, op, cores=2,
                            pools={"l1": panel, "l0c": 1 << 20}, trace_limit=5000)
    assert_services(narrow, expected)
    assert narrow.serviced == pytest.approx(roomy.serviced)
    assert max(narrow.peak_unit_allocated[("l1", unit)] for unit in range(2)) == pytest.approx(panel)
    assert max(roomy.peak_unit_allocated[("l1", unit)] for unit in range(2)) > panel
    assert narrow_ms > roomy_ms
    assert not any(narrow.allocated.values())
    assert narrow.summary()["live_tasks"] == narrow.summary()["live_programs"] == 0
    for program in {row["task"].split(":tile")[0] for row in narrow.trace
                    if ":tile" in row["task"]}:
        loads = {int(row["task"].split(":tile")[1].split(":")[0]): row
                 for row in narrow.trace
                 if row["task"].startswith(program + ":tile") and ":load" in row["task"]}
        computes = {int(row["task"].split(":tile")[1].split(":")[0]): row
                    for row in narrow.trace
                    if row["task"].startswith(program + ":tile") and ":compute" in row["task"]}
        for index in sorted(loads)[1:]:
            assert loads[index]["start_ms"] >= computes[index - 1]["end_ms"] - 1e-9


def test_group_accumulator_remains_live_through_the_final_store_only():
    op = op_for(8, 4, 4, groups=2)
    recipe = TiledMatmul(4, 4, 2, slots=2)
    accumulator = 4 * 4 * max(4, dtype_bytes("bf16"))
    engine, _ = run(recipe, op, cores=1, copy_rate=1 << 20, mte3_rate=2,
                    pools={"l1": 1 << 20, "l0c": accumulator},
                    trace_limit=5000)
    assert engine.peak_unit_allocated.get(("l0c", 0), 0) == accumulator
    compute = {int(row["task"].split(":tile")[1].split(":")[0]): row
               for row in engine.trace if ":compute" in row["task"]}
    stores = {int(row["task"].split(":tile")[1].split(":")[0]): row
              for row in engine.trace if ":store" in row["task"]}
    # Panel 1 is the first group's final K panel. Panel 2 is the next group and
    # cannot acquire the single accumulator until panel 1's store releases it.
    assert compute[2]["start_ms"] >= stores[1]["end_ms"] - 1e-9


def mla_case():
    raw = json.loads((ROOT / "examples/source-families/mla-decode-control.json").read_text())
    raw["model"]["num_heads"] = 16
    raw["machine"]["device"].update(
        compute_units=16, l0a_bytes=65536, l0b_bytes=65536,
        l0c_bytes=262144, l1_bytes=524288,
    )
    raw["machine"]["device"]["rates"].update(mte2=1000, mte3=1000)
    return raw


def traced_execution(raw, op, *, release=7):
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), trace_limit=5000)
    op, recipe = execution.prepare(op)
    context = execution._context(op, 0, "none", recipe=recipe)
    done = recipe.lower(context).start(
        execution.engine, "tiled", execution.engine.submit("ready", release=release))
    execution.engine.advance(horizon=release - 1)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(done,))
    return execution.engine


def service(row, prefix):
    return next(value for name, value in row["serviced"].items()
                if name.startswith(prefix))


def test_ordinary_frontend_mla_groups_and_source_bindings():
    raw = mla_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 33,
                "sequences": 1, "scope": "model"}
    ops = target_graphs(raw, **geometry)[0]
    grouped = {op.op_id.rsplit(".", 1)[-1]: op for op in ops
               if op.op_id.endswith(("q_absorb", "v_expand"))}
    assert {name: dict(op.metadata)["matrix_groups"] for name, op in grouped.items()} == {
        "q_absorb": 16, "v_expand": 16}
    assert grouped["q_absorb"].local_shape == (16, 256, 16)
    assert grouped["v_expand"].local_shape == (16, 16, 256)
    report = operator_costs(raw, ops)
    rows = {row["operator"]["op_id"].rsplit(".", 1)[-1]: row
            for row in report["operators"]}
    tile = TiledMatmul(128, 128, 128)
    for name, op in grouped.items():
        expected, _, _ = loop_oracle(op, tile, reuse_inputs=True, reuse_weights=True)
        row = rows[name]
        assert row["program"] == "TiledMatmul"
        assert service(row, "memory@") == pytest.approx(op.hbm_bytes)
        assert service(row, "mte2@") == pytest.approx(expected["mte2"])
        assert service(row, "mte3@") == pytest.approx(op.mte3_bytes)
        matrix = next(key for key in row["declared_service"] if key.startswith("matrix@"))
        assert service(row, "matrix@") == pytest.approx(row["declared_service"][matrix])
    source = [row for row in report["operators"]
              if row["operator"].get("source_boundary") == "kernel"]
    assert {row["operator"]["recipe"] for row in source} == {"triton_decode_grouped_mla"}
    assert {row["program"] for row in source} == {"OperatorPlan"}
    assert all("stage1_grouped_decode" in row["operator"]["op_id"]
               or "stage2_split_merge" in row["operator"]["op_id"] for row in source)

    engine = traced_execution(raw, grouped["v_expand"])
    computes = [row for row in engine.trace
                if row["task"].endswith(":compute") and row["demand"]]
    units, programs = set(), set()
    for row in computes:
        programs.add(row["task"].split(":tile")[0])
        for name, owners in row["placement"].items():
            if name.startswith("matrix@"):
                units.update(owners)
    assert len(programs) == len(units) == 16
    assert sum(row["task"] == "tiled:setup" for row in engine.trace) == 1
    assert all(row["start_ms"] >= 7 for row in engine.trace if row["demand"])
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
