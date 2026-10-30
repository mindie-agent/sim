"""Public body entry points execute the same resolved physical declaration."""
from copy import copy, deepcopy

import pytest

from mindie_sim import predict
from mindie_sim.cohort_comparison import compare_cohorts
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs, target_operators
from mindie_sim.prediction.program import RecipeBook, ResourceEnvelope
from mindie_sim.structure import export_structure
from mindie_sim.study import study
from test_body_gdn_defaults import case as gdn_case
from test_cohort_comparison import _captured_report, _norm_case
from test_prediction_system import case as small_case


GEOMETRY = dict(phase="decode", tokens=1, context=128, sequences=2, scope="body")


def _gqa_case():
    raw = small_case(output=1)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    return raw


def _graph(case, graphs, **kwargs):
    return graph_costs(case, graphs, external_inputs={"analysis_input": 0},
                       isolated=False, **kwargs)


def test_gdn_body_graph_operator_and_structure_share_actual_caller():
    raw = gdn_case("Ascend950DT_9582")
    before = deepcopy(raw)
    body = predict(raw, **GEOMETRY)
    graphs = target_graphs(raw, **GEOMETRY)
    direct = _graph(raw, graphs)
    isolated = operator_costs(raw, target_operators(raw, **GEOMETRY))
    structure = export_structure(raw, **GEOMETRY)
    assert direct["timeline"] == body["timeline"]
    assert direct["duration_ms"] == body["body_forward_ms"]
    assert direct["execution"]["serviced"] == body["execution"]["serviced"]
    for recipe in ("ascend_causal_conv1d_update", "ascend_gdn_decode_gating",
                   "ascend_recurrent_gdn_fp32_state"):
        assert sum(row["recipe"] == recipe for row in structure["nodes"]) == 1
        assert sum(row["operator"]["recipe"] == recipe for row in isolated["operators"]) == 1
    selected = structure["execution"]["software"]["kernels"]["lowerings"]["gdn_decode"]
    assert selected["implementation"] == "vllm_ascend_gdn_decode"
    assert selected["options"]["gating_beta_store"]["from_hardware"]
    assert structure["execution"]["default_mechanisms"] == body["default_mechanisms"]
    assert raw == before


def test_graph_carrier_keeps_current_hardware_and_explicit_book_or_recipe_overrides():
    raw = _gqa_case()
    graphs = target_graphs(raw, **GEOMETRY)
    automatic = _graph(raw, graphs)
    attention = lambda result: [row for row in result["timeline"] if row["family"] == "attention"]
    assert all(row["program"] == "StagedGQADecode" for row in attention(automatic))
    single_rank = _graph(raw, graphs[0])
    assert single_rank["timeline"] == automatic["timeline"]
    assert single_rank["execution"]["serviced"] == automatic["execution"]["serviced"]
    assert _graph(raw, dict(graphs))["timeline"] == automatic["timeline"]
    assert _graph(raw, {0: graphs[0]})["timeline"] == automatic["timeline"]
    cloned = deepcopy(graphs)
    assert cloned[0].effective_kernels is cloned.effective_kernels
    assert _graph(raw, cloned)["timeline"] == automatic["timeline"]
    rank_costs = operator_costs(raw, graphs[0])
    assert all(row["program"] == "StagedGQADecode" for row in rank_costs["operators"]
               if row["operator"]["family"] == "attention")
    isolated = operator_costs(raw, target_operators(raw, **GEOMETRY))
    assert all(row["program"] == "StagedGQADecode" for row in isolated["operators"]
               if row["operator"]["family"] == "attention")
    subset = operator_costs(raw, target_operators(raw, **GEOMETRY)[:])
    assert subset["operators"] == isolated["operators"]
    for clone in (copy, deepcopy):
        assert operator_costs(raw, clone(graphs[0]))["operators"] == rank_costs["operators"]
    slow = deepcopy(raw)
    slow["machine"]["hbm_bw_bytes_per_s"] = 1e8
    slowed = _graph(slow, graphs)
    assert slowed["duration_ms"] > automatic["duration_ms"] * 10
    assert slowed["execution"]["serviced"] == pytest.approx(automatic["execution"]["serviced"])
    explicit = RecipeBook(families={"attention": ResourceEnvelope()})
    override = _graph(raw, graphs, recipes=explicit)
    assert all(row["program"] == "ResourceEnvelope" for row in attention(override))
    changed = deepcopy(raw)
    changed["runtime"]["kernels"] = {"recipes": {"flash_attention_decode": {"kind": "envelope"}}}
    override = _graph(changed, graphs)
    assert all(row["program"] == "ResourceEnvelope" for row in attention(override))


def test_explicit_book_does_not_install_structural_caller_and_model_scope_is_unchanged():
    raw = gdn_case("Ascend950DT_9582")
    explicit = RecipeBook()
    body = predict(raw, recipes=explicit, **GEOMETRY)
    graphs = target_graphs(raw, apply_body_defaults=False, **GEOMETRY)
    direct = _graph(raw, graphs, recipes=explicit)
    assert body["timeline"] == direct["timeline"]
    assert not any(row["recipe"] == "ascend_gdn_decode_gating" for row in body["timeline"])
    model = {**GEOMETRY, "scope": "model"}
    assert target_graphs(raw, **model) == target_graphs(raw, apply_body_defaults=False, **model)
    assert export_structure(raw, **model)["graph_id"] == export_structure(
        raw, apply_body_defaults=False, **model)["graph_id"]


def test_profiler_cohort_replays_effective_body_and_rejects_unmarked_legacy_graph(profiler_api):
    raw = _norm_case()
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    mapping, _ = _captured_report(raw, "a3_glm53_decode.csv", "RmsNorm")
    compared = compare_cohorts(raw, mapping)
    assert compared["windows"][0]["cohorts"][0]["status"] == "compared"
    effective = compared["declared_execution"]["software"]["kernels"]
    assert effective["recipes"]["flash_attention_decode"]["kind"] == "staged_gqa_decode"
    structure = export_structure(raw, phase="decode", tokens=1, scope="body",
                                 apply_body_defaults=False)
    legacy, _ = _captured_report(raw, "a3_glm53_decode.csv", "RmsNorm", structure=structure)
    # An explicitly declared generic control is still replayable.
    assert compare_cohorts(raw, legacy)["windows"][0]["cohorts"][0]["status"] == "compared"
    # Historical manifests did not declare the new opt-out; they must be rebuilt.
    legacy["declared_execution"].pop("apply_body_defaults")
    legacy["declared_execution"].pop("default_mechanisms")
    with pytest.raises(ValueError, match="cohort_case_graph_id_mismatch"):
        compare_cohorts(raw, legacy)


def test_body_study_and_cli_graph_share_selected_programs(tmp_path):
    import json
    from mindie_sim.__main__ import parser, run

    raw = _gqa_case()
    body = predict(raw, **GEOMETRY)
    result = study([{"id": "base", "case": raw}], [{"id": "body", **GEOMETRY}], level="graph")
    row = result["evaluations"][0]
    assert row["duration_ms"] == body["body_forward_ms"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(raw))
    args = ["graph", str(path), "--phase", "decode", "--tokens", "1", "--context", "128",
            "--sequences", "2", "--scope", "body", "--skip-isolated"]
    assert run(parser().parse_args(args))["duration_ms"] == body["body_forward_ms"]
    control = run(parser().parse_args([*args, "--no-body-defaults"]))
    assert all(row["program"] == "ResourceEnvelope" for row in control["timeline"]
               if row["family"] == "attention")
