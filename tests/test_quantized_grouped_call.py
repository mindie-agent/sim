"""Independent checks for the retained INT8/FRACTAL_NZ grouped call."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from statistics import median

import pytest

from mindie_sim.observations.call_alignment import compare_calls
from mindie_sim.observations.captures import iter_conditions
from mindie_sim.observations.grouped_binding import bind_grouped_condition
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.call_builders import build_call
from mindie_sim.prediction.call_grouped import (
    EPILOGUE_CONTRACT, NZ_TILE_K, NZ_TILE_N, RECIPE_NAME,
    build_quantized_grouped_call,
)
from mindie_sim.prediction.matrix_mapping import MappedMatmul
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from test_capture_adapters import grouped_protocol


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/observations/capture_adapters/grouped_probe.json"


def _equal_group_call(rows=(2, 0, 2, 0, 2), k=32, n=64):
    return build_quantized_grouped_call({"expert_rows": list(rows), "k": k, "n": n})


def _small_case():
    return json.loads((ROOT / "examples/calls/quantized-grouped-small.json").read_text())


def test_exact_payload_scale_output_count_and_group_domains_with_empty_experts():
    call = _equal_group_call()
    op = call.ops[0]
    metadata = dict(op.metadata)
    assert len(call.ops) == 1 and op.op_id == "grouped_matmul"
    assert metadata["expert_rows"] == (2, 0, 2, 0, 2)
    assert metadata["active_expert_ids"] == (0, 2, 4)
    assert metadata["empty_expert_ids"] == (1, 3)
    assert metadata["matrix_groups"] == 3
    assert metadata["tokens_per_active_expert"] == 2
    assert op.semantic_shape == op.local_shape == (6, 64, 32)
    assert op.flops == 2 * 6 * 64 * 32

    activation_payload = 6 * 32
    activation_scales = 4 * 6
    weight_payload = 3 * 32 * 64
    weight_scales = 4 * 3 * 64
    counts = 8 * 5
    output = 2 * 6 * 64
    assert metadata["activation_payload_bytes"] == activation_payload
    assert metadata["activation_scale_bytes"] == activation_scales
    assert metadata["weight_payload_bytes"] == weight_payload
    assert metadata["weight_scale_bytes"] == weight_scales
    assert metadata["packed_weight_bytes"] == weight_payload + weight_scales
    assert metadata["output_bytes"] == output
    assert metadata["group_list_scan_bytes"] == counts
    assert op.mte2_bytes == (activation_payload + activation_scales
                             + weight_payload + weight_scales)
    assert op.mte3_bytes == output
    assert op.hbm_bytes == op.mte2_bytes + output + counts
    assert metadata["weight_bank_residency_bytes"] == 5 * 32 * 64
    assert metadata["weight_bank_residency_bytes"] != weight_payload
    assert call.ledger["geometry"]["flops"] == op.flops
    assert metadata["weight_scale_bank_residency_bytes"] == 5 * 64 * 4
    assert call.ledger["group_list_fetch"]["included_in_matrix_operands"] is False

    epilogue = call.ledger["matrix"]["epilogue"]
    assert epilogue["vector_ops"] == 4 * 6 * 64
    assert epilogue["accumulator_dtype"] == "int32"
    assert epilogue["ordering"] == (
        "after_complete_int32_accumulation_before_output_store")
    assert epilogue["contract"] == EPILOGUE_CONTRACT
    assert metadata["matrix_epilogue"] == {
        "kind": "row_column_scale", "operations_per_output": 4,
        "row_scale_bytes": activation_scales,
        "column_scale_bytes": weight_scales,
    }
    assert op.vector_ops == epilogue["vector_ops"]
    assert call.ledger["identified_historical_compiled_kernel"] is None
    assert call.ledger["matrix"]["one_api_kernel_boundary"] is True
    assert call.ledger["quantization"]["activation_quantization_timed"] is False
    assert call.ledger["quantization"]["weight_nz_conversion_timed"] is False
    assert call.ledger["timed_region"]["activation_quantization"] == "outside"
    assert call.ledger["timed_region"]["weight_nz_conversion"] == "outside"


def test_public_builder_registration_returns_call_definition_fields():
    declaration = {
        "builder": "quantized_grouped",
        "arguments": {"expert_rows": [2, 0, 2], "k": 16, "n": 64},
    }
    call = build_call(declaration)
    assert call.ops == _equal_group_call((2, 0, 2), 16, 64).ops
    assert isinstance(call.recipes.select(call.ops[0]), MappedMatmul)
    assert call.ledger["geometry"]["logical_m"] == 4
    assert any("epilogue" in item for item in call.limitations)


def test_recipe_reuses_shared_mapped_matmul_and_nz_layout_is_whole_tile_only():
    call = _equal_group_call()
    recipe = call.recipes.select(call.ops[0])
    assert isinstance(recipe, MappedMatmul)
    assert call.recipes.recipes[RECIPE_NAME] is recipe
    assert call.ops[0].recipe == RECIPE_NAME
    metadata = dict(call.ops[0].metadata)
    assert metadata["nz_tile_kn"] == (NZ_TILE_K, NZ_TILE_N)
    assert metadata["nz_storage_shape"] == (5, 2, 2, 16, 32)
    assert math.prod(metadata["nz_storage_shape"]) == 5 * 32 * 64
    assert metadata["nz_internal_tiled_access"] == "unresolved"

    for k, n in ((24, 64), (32, 16), (16, 48)):
        with pytest.raises(ModelingError, match="fractal_nz_requires_whole_tiles"):
            build_quantized_grouped_call({"expert_rows": [1, 0, 1], "k": k, "n": n})


@pytest.mark.parametrize("arguments, reason", [
    ({"expert_rows": [2, 0, 3], "k": 16, "n": 64}, "uniform_nonempty_expert_rows"),
    ({"expert_rows": [0, 0], "k": 16, "n": 64}, "at_least_one_nonempty_expert"),
    ({"expert_rows": [1, 1], "k": 16, "n": 64, "bias": "x"}, "bias"),
    ({"expert_rows": [1, 1], "k": 16, "n": 64, "weight_layout": "ND"}, "api_argument_unsupported"),
    ({"expert_rows": [1, 1], "k": 16, "n": 64, "split_item": 0}, "api_argument_unsupported"),
    ({"expert_rows": [1, 1], "k": 16, "n": 64, "unknown": 1}, "arguments_unsupported"),
])
def test_unsupported_nonuniform_layout_and_api_branches_remain_visible(arguments, reason):
    with pytest.raises(ModelingError, match=reason):
        build_quantized_grouped_call(arguments)


def test_real_capture_binds_geometry_and_never_consumes_durations_canary_or_split():
    protocol = grouped_protocol()
    conditions = list(iter_conditions(protocol))
    assert len(conditions) == 6
    bindings = [bind_grouped_condition(condition) for condition in conditions]
    assert [binding.invocations for binding in bindings] == [1, 1, 8] * 2
    assert [binding.mode for binding in bindings] == ["eager", "graph", "graph"] * 2
    assert {binding.call.ops[0].flops for binding in bindings} == {
        2 * 32 * 256 * 2048, 2 * 32 * 2048 * 128}
    for condition, binding in zip(conditions, bindings):
        identity = condition["identity"]
        assert binding.target_identity == identity
        assert binding.ledger["timed_protocol"]["sample_division"] is False
        assert binding.ledger["timed_protocol"]["whole_replay_events"] is (
            identity["execution"]["mode"] == "graph")
        assert binding.ledger["read_scope"]["duration"] is False
        assert binding.ledger["read_scope"]["canary"] is False
        assert binding.ledger["read_scope"]["sample_split"] is False
        assert len(condition["samples"]) == 40
        assert {sample["split"] for sample in condition["samples"]} == {"unassigned"}

    source = conditions[1]
    altered = deepcopy(source)
    altered["samples"][0]["duration_ms"] *= 17
    altered["samples"][1]["split"] = "heldout"
    altered["metadata"]["canary"] = {"ok": False}
    baseline = bind_grouped_condition(source)
    changed = bind_grouped_condition(altered)
    assert changed.call.ops == baseline.call.ops
    assert changed.ledger == baseline.ledger
    assert changed.target_identity == baseline.target_identity


@pytest.mark.parametrize("mutate, reason", [
    (lambda identity: identity["implementation"].update(producer_sha256="0" * 64),
     "audited_producer"),
    (lambda identity: identity["implementation"]["contract"].update(weight="int8 [E, k, n] ND"),
     "api_contract"),
    (lambda identity: identity["dtype"].update(output="float16"), "dtype_contract"),
    (lambda identity: identity["layout"].update(weight="ND"), "layout_contract"),
    (lambda identity: identity["execution"].update(invocations=2), "execution"),
    (lambda identity: identity["shape"].update(tokens=31), "sum_to_logical_rows"),
])
def test_binder_rejects_unsupported_api_or_protocol_without_fallback(mutate, reason):
    condition = deepcopy(list(iter_conditions(grouped_protocol()))[0])
    mutate(condition["identity"])
    with pytest.raises(ModelingError, match=reason):
        bind_grouped_condition(condition)


@pytest.mark.parametrize("projection,groups,tokens", [
    ("gate_up", 8, 32), ("gate_up", 8, 128), ("gate_up", 32, 32),
    ("gate_up", 32, 128), ("gate_up", 256, 256),
    ("down", 8, 32), ("down", 8, 128), ("down", 32, 32),
    ("down", 32, 128), ("down", 256, 256),
])
def test_all_historical_producer_shapes_build_atomically(projection, groups, tokens):
    k, n = (2048, 256) if projection == "gate_up" else (128, 2048)
    rows = [tokens // groups] * groups + [0] * (256 - groups)
    call = build_quantized_grouped_call({"expert_rows": rows, "k": k, "n": n})
    assert len(call.ops) == 1
    assert call.ops[0].flops == 2 * tokens * n * k
    assert dict(call.ops[0].metadata)["matrix_groups"] == groups
    assert dict(call.ops[0].metadata)["weight_payload_bytes"] == groups * k * n


def test_public_call_executes_one_shared_boundary():
    call = _equal_group_call(rows=(1, 0, 1), k=16, n=64)
    case = _small_case()
    for mode, invocations in (("eager", 1), ("graph", 1), ("graph", 8)):
        result = call_costs(
            case, call.ops, recipes=call.recipes, invocations=invocations,
            mode=mode, max_operators=16,
        )
        assert result["single_call"]["expanded_operators"] == 1
        assert result["operator_occurrences"] == invocations
        assert result["protocol"]["invocations"] == invocations
        assert math.isfinite(result["duration_ms"])
        assert result["execution"]["live_tasks"] == 0


def _execute_small_quantized_call(rows):
    call = _equal_group_call(rows=rows)
    op = call.ops[0]
    rates = {"cube": 1, "hbm": 4096, "mte2": 1024, "mte1": 2048,
             "mte3": 1024, "vector": 256}
    pools = {"l1": 4096, "l0a": 128, "l0b": 512, "l0c": 512}
    units = {role: 1 if role == "hbm" else 2 for role in rates}
    engine = Engine(
        [Resource(name, rate, units=units[name]) for name, rate in rates.items()],
        pools=pools, pool_units={pool: 2 for pool in pools}, trace_limit=1024,
    )
    context = RecipeContext(
        op, {}, {name: name for name in rates}, rates, 2,
        pool_names={pool: pool for pool in pools}, pool_capacities=pools,
        resource_units=units, compute_peaks={"int8": 1e6},
    )
    work = Work(
        matrix={"int8": op.flops},
        resources={"vector": op.vector_ops, "hbm": op.hbm_bytes,
                   "mte2": op.mte2_bytes, "mte3": op.mte3_bytes},
    )
    context = replace(context, work=context.bind(work))
    recipe = replace(
        call.recipes.select(op), candidates=((2, 32, 16),),
        compute_m=2, compute_n=16,
    )
    complete = recipe.lower(context).start(
        engine, "quantized", engine.submit("input", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return engine


def test_quantized_scales_and_counts_keep_their_own_boundaries_across_k_panels():
    engine = _execute_small_quantized_call((2, 0, 2, 0, 2))
    # Three [2,32] @ [32,64] domains, two N macro tiles, two K panels.
    # Each A payload is read privately twice (four times by inner tiles);
    # each B payload once. Scales load only at the final K panel.
    assert engine.serviced == pytest.approx({
        "cube": 1000 * (2 * 6 * 64 * 32) / 1e6,
        "hbm": 6 * 32 + 4 * 6 + 3 * 32 * 64 + 4 * 3 * 64 + 2 * 6 * 64 + 8 * 5,
        "mte2": 2 * 6 * 32 + 2 * 4 * 6 + 3 * 32 * 64 + 4 * 3 * 64,
        "mte1": 4 * 6 * 32 + 3 * 32 * 64,
        "mte3": 2 * 6 * 64,
        "vector": 4 * 6 * 64,
    })
    epilogues = [row for row in engine.trace if row["task"].endswith(":epilogue")]
    assert len(epilogues) == 6
    for epilogue in epilogues:
        prefix, part = epilogue["task"].rsplit(":panel", 1)
        panel = int(part.split(":", 1)[0])
        assert panel % 2 == 1
        matrices = [
            row for row in engine.trace if row["task"].endswith(":matrix")
            and any(row["task"].startswith(f"{prefix}:panel{index}:")
                    for index in (panel - 1, panel))
        ]
        assert len(matrices) == 4
        assert max(row["end_ms"] for row in matrices) <= epilogue["start_ms"]

    compact = _execute_small_quantized_call((2, 2, 2))
    delta = {name: engine.serviced[name] - compact.serviced[name] for name in engine.serviced}
    assert delta == pytest.approx(
        {name: 16 if name == "hbm" else 0 for name in engine.serviced}, abs=1e-9)


def test_real_grouped_call_alignment_preserves_events_and_whole_graph_replays():
    protocol = grouped_protocol()
    observed = list(iter_conditions(protocol))
    raw = json.loads(FIXTURE.read_text())
    result = compare_calls(_small_case(), protocol)
    assert result["summary"]["samples"] == 240
    assert result["summary"]["sample_roles"] == {"unassigned": 240}
    for row, source in zip(result["conditions"], observed):
        assert row["identity"] == source["identity"]
        assert row["canary"] == source["metadata"].get("canary")
        assert row["statistics"]["sample_count"] == 40
        assert row["statistics"]["sample_median_ms"] == median(
            sample["duration_ms"] for sample in source["samples"])
    assert result["status"] == "completed"
    assert result["summary"]["unique_predictions"] == 6
    for row in result["conditions"]:
        assert row["status"] == "predicted"
        for comparison in row["comparisons"]:
            if comparison["scenario"] != "device":
                continue
            prediction = next(item for item in result["predictions"]
                              if item["prediction_id"] == comparison["prediction_id"])
            assert prediction["protocol"]["invocations"] == (
                8 if row["identity"]["execution"]["invocations"] == 8 else 1)
            assert prediction["single_call"]["expanded_operators"] == 1
    fields = ("eager_per_call", "graph_replay_1call", "graph_replay_8calls")
    for index, row in enumerate(raw["rows"]):
        for offset, field in enumerate(fields):
            source = observed[index * 3 + offset]
            assert [sample["duration_ms"] for sample in source["samples"]] == (
                row[field]["raw_samples_ms"])
            assert len(source["samples"]) == 40
            assert source["identity"]["execution"]["invocations"] == (
                8 if field == "graph_replay_8calls" else 1)
    assert observed[2]["identity"]["execution"]["invocations"] == 8
