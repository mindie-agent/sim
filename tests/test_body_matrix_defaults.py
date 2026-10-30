"""Body matrix defaults propagate to real execution without overriding choices."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.structure import export_structure
from test_prediction_system import case
from test_grouped_family_selection import grouped_case


MATRIX_FAMILIES = {"qkv_projection", "output_projection", "gate_up_projection", "down_projection"}


GEOMETRY = dict(scope="body", phase="decode", tokens=4, context=128, sequences=2)


def physical_case():
    raw = case(output=1)
    raw["model"].update(layers=1, hidden_size=256, head_dim=64, intermediate_size=512)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    return raw


def matrix_rows(report):
    return [row for row in report["timeline"] if row["family"] in MATRIX_FAMILIES]


def test_body_matrix_default_is_shared_by_predict_graph_operators_and_structure():
    raw = physical_case()
    before = deepcopy(raw)
    graphs = target_graphs(raw, **GEOMETRY)
    body = predict(raw, **GEOMETRY)
    graph = graph_costs(raw, graphs, isolated=False, external_inputs={"analysis_input": 0})
    isolated = operator_costs(raw, target_operators(raw, **GEOMETRY))
    structure = export_structure(raw, **GEOMETRY)
    assert matrix_rows(body) and all(row["program"] == "SteadyMatmul" for row in matrix_rows(body))
    assert graph["timeline"] == body["timeline"]
    assert graph["duration_ms"] == body["body_forward_ms"]
    assert graph["execution"]["serviced"] == body["execution"]["serviced"]
    assert all(row["program"] == "SteadyMatmul" for row in isolated["operators"]
               if row["operator"]["family"] in MATRIX_FAMILIES)
    assert structure["execution"]["software"]["kernels"]["programs"]["matmul"] == {
        "kind": "steady_matmul", "candidates": [[128, 128, 128], [128, 128, 64]]}
    assert structure["execution"]["default_mechanisms"] == body["default_mechanisms"]
    assert body["execution"]["live_tasks"] == body["execution"]["live_programs"] == 0
    assert raw == before


@pytest.mark.parametrize("program", ["tiled_matmul", "mapped_matmul"])
def test_explicit_matrix_program_is_preserved_in_both_body_costing_paths(program):
    raw = physical_case()
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": program}}}
    report = predict(raw, **GEOMETRY)
    expected = {"tiled_matmul": "TiledMatmul", "mapped_matmul": "MappedMatmul"}[program]
    assert all(row["program"] == expected for row in matrix_rows(report))
    isolated = operator_costs(raw, target_graphs(raw, **GEOMETRY)[0])
    assert all(row["program"] == expected for row in isolated["operators"]
               if row["operator"]["family"] in MATRIX_FAMILIES)
    selected = next(row for row in report["default_mechanisms"]["selections"]
                    if row["family"] == "matmul")
    assert selected["selection"] == "preserved"


def test_named_family_source_and_explicit_book_keep_priority():
    raw = physical_case()
    raw["runtime"]["kernels"] = {"families": {family: {"kind": "mapped_matmul"} for family in MATRIX_FAMILIES},
                                "recipes": {"dense_down": {"kind": "tiled_matmul"}}}
    body = predict(raw, **GEOMETRY)
    rows = matrix_rows(body)
    assert any(row["recipe"] == "dense_down" and row["program"] == "TiledMatmul" for row in rows)
    assert all(row["program"] == "MappedMatmul" for row in rows if row["recipe"] != "dense_down")
    source = grouped_case(source=True)
    source_body = predict(source, scope="body", phase="decode", tokens=1)
    assert all(row["program"] == "OperatorPlan" for row in source_body["timeline"]
               if row["family"] == "grouped_gemm")
    explicit = RecipeBook(programs={"matmul": TiledMatmul(16, 16)})
    report = predict(physical_case(), recipes=explicit, **GEOMETRY)
    assert all(row["program"] == "TiledMatmul" for row in matrix_rows(report))
    assert report["default_mechanisms"] == {"profile": "explicit_recipe_book_preserved"}


def test_model_scope_and_body_optout_retain_detailed_control_and_semantic_work():
    raw = physical_case()
    options = {**GEOMETRY, "scope": "model"}
    model = target_graphs(raw, **options)
    assert model == target_graphs(raw, apply_body_defaults=False, **options)
    model_costs = operator_costs(raw, model)
    assert all(row["program"] == "TiledMatmul" for row in model_costs["operators"]
               if row["operator"]["family"] in MATRIX_FAMILIES)
    generic = target_graphs(raw, apply_body_defaults=False, **GEOMETRY)
    compact = target_graphs(raw, **GEOMETRY)
    assert {r: tuple(op for op in v if op.family in MATRIX_FAMILIES)
            for r, v in compact.items()} == {
                r: tuple(op for op in v if op.family in MATRIX_FAMILIES)
                for r, v in generic.items()}
    # The choice changes physical scheduling; tensor geometry and logical work
    # remain identical, and the public opt-out continues to execute detailed.
    graph = graph_costs(raw, generic, isolated=False, external_inputs={"analysis_input": 0})
    assert all(row["program"] == "TiledMatmul" for row in matrix_rows(graph))


def test_serving_default_remains_the_explicit_tiled_control():
    raw = physical_case()
    automatic = predict(raw)
    explicit = deepcopy(raw)
    explicit["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "tiled_matmul"}}}
    control = predict(explicit)
    assert automatic["metrics"] == control["metrics"]
    assert automatic["execution"] == control["execution"]


def test_default_keeps_fp32_router_owner_grid_and_grouped_mapping_domain():
    path = Path(__file__).resolve().parents[1] / "examples/cases/glm53-a3-w4a8.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 4  # First MoE layer; preserve every physical width.
    graph = target_graphs(raw, scope="body", phase="decode", tokens=1, context=4096)[0]
    router_index = next(i for i, op in enumerate(graph) if op.layer == 3 and op.family == "router")
    router = operator_costs(raw, graph[router_index:router_index + 1])["operators"][0]
    mapping = next(row for row in router["basis"]["matrix_mappings"] if "selected" in row)
    assert router["operator"]["local_shape"] == [1, 256, 6144]
    assert router["operator"]["dtype"] == "fp32"
    assert (mapping["selected"]["tile_m"], mapping["selected"]["tile_n"],
            mapping["selected"]["tile_k"]) == (128, 128, 64)
    assert mapping["default_128_grid_tiles"] == 2
    assert router["peak_buffers"]["l0b@r0:s0"] <= 65536
    explicit = deepcopy(raw)
    explicit["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}}}
    for i, op in enumerate(graph):
        if op.layer == 3 and op.family == "grouped_gemm":
            inherited = operator_costs(raw, graph[i:i + 1])["operators"][0]
            previous = operator_costs(explicit, graph[i:i + 1])["operators"][0]
            assert inherited["serviced"] == previous["serviced"]
            assert inherited["duration_ms"] == previous["duration_ms"]
            selected = next(row["selected"] for row in inherited["basis"]["matrix_mappings"]
                            if "selected" in row)
            assert selected["tile_n"] == 256
