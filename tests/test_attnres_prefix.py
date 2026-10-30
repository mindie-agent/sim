"""Source-selected AttnRes prefix state, traffic and engine controls.

The transition tests use the public Frontend on an independently sliced Kimi
declaration; they do not read capture rows or measured durations.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

import mindie_sim
from mindie_sim.prediction.family_attnres import (
    IMPLEMENTATION, SOURCE, attn_res_point, max_resident_rows, recipes,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import (
    operator_costs, target_graphs, target_operators,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_structure


ROOT = Path(__file__).resolve().parents[1]
OLD = {"residual_values_fp32", "attention_residual_scores",
       "attention_residual_softmax", "attention_residual_mix",
       "attention_residual_output_cast", "residual_add"}


def _binding(options=None):
    return {**parse_case(json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())).runtime.kernels,
            "lowerings": {"attention_residual": {
                "implementation": IMPLEMENTATION, "options": options or {}}}}


def _case(layers=3, block=2, selected=True, options=None):
    base = parse_case(json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text()))
    plans = tuple(replace(plan, index=index, residual_block_size=block)
                  for index, plan in enumerate(base.model.layer_plans[:layers]))
    kernels = _binding(options) if selected else base.runtime.kernels
    runtime = replace(base.runtime, pp_layer_partition=(layers,), kernels=kernels)
    return replace(base, model=replace(
        base.model, layers=layers, body_layers=layers, layer_plans=plans), runtime=runtime)


def _points(ops):
    return [op for op in ops if op.recipe == "attn_res_fwd"]


def _view(op):
    return dict(op.metadata)["attn_res_view"]


def _frontend_body(case, *, tokens=5, stages=None):
    views = (RequestView("r", tokens, 0, False, "prefill"),)
    return Frontend().body(case.model, case.runtime, views, ready="input",
                           tokens=tokens, mode="prefill", stages=stages)


def _prepared(case, op, *, aiv=None, ub=None):
    geometry = Geometry.from_devices(case.machine.dies[:1])
    if aiv is not None:
        geometry = replace(geometry, resource_units={**geometry.resource_units, "vector": aiv})
    if ub is not None:
        geometry = replace(geometry, buffers={**geometry.buffers, "ub": ub})
    return recipes()["attn_res_fwd"].prepare(op, geometry)[0]


def _read_bytes(op):
    return op.mte2_bytes + dict(op.metadata).get("resource_work", {}).get("mte2_aiv", 0)


def test_unselected_graph_is_unchanged_and_selected_path_removes_old_five_ops():
    base = _case(selected=False)
    before = target_operators(base, phase="prefill", tokens=8, scope="body")
    assert OLD <= {op.recipe for op in before}

    selected = _case(selected=True)
    ops = target_operators(selected, phase="prefill", tokens=8, scope="body")
    points = _points(ops)
    assert len(points) == 2 * 3 + 1
    assert not ({op.recipe for op in ops} & OLD)
    assert not any(op.op_id.endswith((".norm1", ".norm2")) for op in ops)
    assert {op.family for op in points} == {"attn_res_prefix"}
    for op in points:
        assert op.source_boundary == "kernel" and op.flops == 0
        assert op.returns == ("hidden_states", "prefix_out", "materialized")
        assert _view(op)["bank_capacity"] == 2
        assert _view(op)["block_token_stride"] == 2 * selected.model.hidden_size


def test_three_layer_b2_transition_absorbs_pending_mlp_once_and_keeps_bank_rows():
    case = _case(layers=3, block=2)
    ops = target_operators(case, phase="prefill", tokens=5, scope="body")
    by_id = {op.op_id: op for op in ops}
    p0, f0 = by_id["s0.l0.attention.attn_res_fwd"], by_id["s0.l0.ffn.attn_res_fwd"]
    p1, f1 = by_id["s0.l1.attention.attn_res_fwd"], by_id["s0.l1.ffn.attn_res_fwd"]
    p2, f2 = by_id["s0.l2.attention.attn_res_fwd"], by_id["s0.l2.ffn.attn_res_fwd"]
    final = by_id["final.attn_res_fwd"]

    # Layer 0 stores the embedding in slot 0 and clears the live prefix.
    assert (_view(p0)["valid_blocks"], _view(p0)["block_write_idx"],
            _view(p0)["fuse_add"]) == (0, 0, False)
    assert (_view(p0)["mix"], _view(p0)["output_norm"]) == (True, True)
    assert p0.deps == ("analysis_input",)
    assert dict(p0.metadata)["attn_res_bank_ids"] == (None, None)
    assert dict(f0.metadata)["attn_res_bank_ids"][0] == p0.op_id
    assert (_view(f0)["valid_blocks"], _view(f0)["fuse_add"]) == (1, False)
    assert p0.op_id in f0.deps and "analysis_input" not in f0.deps

    # A non-boundary prepare absorbs the previous MLP output exactly once.
    assert (_view(p1)["valid_blocks"], _view(p1)["block_write_idx"],
            _view(p1)["fuse_add"]) == (1, -1, True)
    assert (_view(f1)["valid_blocks"], _view(f1)["fuse_add"]) == (1, True)
    assert (_view(p2)["valid_blocks"], _view(p2)["block_write_idx"],
            _view(p2)["fuse_add"]) == (1, 1, True)
    assert (_view(f2)["valid_blocks"], _view(f2)["fuse_add"]) == (2, False)

    # Final absorbs the last MLP and mixes both banks, without output norm.
    assert (_view(final)["valid_blocks"], _view(final)["output_norm"],
            _view(final)["mix"], _view(final)["block_write_idx"]) == (2, False, True, -1)
    banks = dict(final.metadata)["attn_res_bank_ids"]
    assert banks == (p0.op_id, p2.op_id)

    # Each layer MLP leaves the ordinary residual chain and enters one point.
    mlp_ffn = {}
    for layer in range(3):
        ffn = [op for op in ops if op.layer == layer
               and dict(op.metadata).get("structure_block") == "ffn"
               and op.family != "attn_res_prefix"]
        mlp_ffn[layer] = ffn[-1].op_id
    for layer, mlp_id in mlp_ffn.items():
        count = sum(mlp_id in op.deps for op in _points(ops))
        assert count == 1
    assert mlp_ffn[0] in p1.deps and mlp_ffn[1] in p2.deps
    assert mlp_ffn[2] in final.deps


@pytest.mark.parametrize("layers,read_rows,write_rows", [
    (2, 12, 9), (3, 19, 13), (4, 27, 17), (5, 36, 21),
])
def test_complete_small_chains_match_independent_source_traffic_table(
        layers, read_rows, write_rows):
    case = _case(layers=layers)
    points = _points(target_operators(case, phase="prefill", tokens=5, scope="body"))
    prepared = [_prepared(case, point, ub=1024 * 1024) for point in points]
    hidden, tokens, active = case.model.hidden_size, 5, 5
    row_bytes = 2 * tokens * hidden
    # Every point except first performs score-weight initialization; only the
    # two per-layer points read output gamma. Terminal has no output norm.
    weights = 4 * hidden * active * (2 * layers)
    gamma = 2 * hidden * tokens * (2 * layers)
    assert len(points) == 2 * layers + 1
    assert sum(_read_bytes(op) for op in prepared) == row_bytes * read_rows + weights + gamma
    assert sum(op.mte3_bytes for op in prepared) == row_bytes * write_rows


def _context(case, *, tokens=5):
    return {"prefix": "p", "stage": 0, "layer": 0, "requests": (),
            "model": case.model, "mode": "prefill", "tokens": tokens}


def test_flag_traffic_is_independently_arithmetic_and_capacity_differs_from_stride():
    case = _case(layers=3, block=2)
    hidden, tokens, capacity = case.model.hidden_size, 5, 6
    stride = (capacity + 2) * hidden
    bank = tuple(f"b{index}" for index in range(capacity))
    resident = attn_res_point(
        _context(case), prefix_id="prefix", addend_id="addend", bank_ids=bank,
        valid_blocks=3, write_slot=3, output_norm=True, mix=True,
        materialize=True, tokens_local=tokens, bank_capacity=capacity,
        block_token_stride=stride, optimize_prefill=False)
    op = _prepared(case, resident.ops[0])
    pv = dict(op.metadata)["attn_res_view"]
    assert pv["resident"] is True and pv["bank_capacity"] == 6
    assert pv["block_token_stride"] == stride == 8 * hidden
    assert max_resident_rows(hidden, dict(case.machine.dies[0].buffers)["ub"]) >= 4
    expected_read = (2 * tokens * hidden * 2               # prefix + addend
                     + 2 * tokens * 3 * hidden            # resident valid rows
                     + 4 * hidden * tokens                # score weights/core
                     + 2 * hidden * tokens)               # gamma once/token
    expected_write = 2 * tokens * hidden * 4              # output, prefix, bank, materialized
    assert (_read_bytes(op), op.mte3_bytes, op.hbm_bytes) == (
        expected_read, expected_write, expected_read + expected_write)
    assert op.flops == 0 and op.vector_ops > 0
    assert resident.bank_ids[3] == op.op_id

    reload = attn_res_point(
        _context(case), prefix_id="prefix", addend_id=None, bank_ids=bank,
        valid_blocks=4, write_slot=None, output_norm=False, mix=True,
        materialize=False, tokens_local=tokens, bank_capacity=capacity,
        block_token_stride=stride, optimize_prefill=False)
    reload_op = _prepared(case, reload.ops[0])
    rv = dict(reload_op.metadata)["attn_res_view"]
    assert rv["resident"] is False
    assert reload.raw_prefix_id == "prefix" and reload.materialized_id == reload.hidden_id
    assert _read_bytes(reload_op) == 2 * tokens * hidden + 4 * tokens * 4 * hidden + 4 * hidden * tokens
    assert reload_op.mte3_bytes == 2 * tokens * hidden
    assert dict(reload.ops[0].metadata)["attn_res_prefix_out_aliases_prefix"] is True


def test_prefill_gamma_cache_tail_ownership_and_no_mix_bypass():
    case = _case(layers=3, block=2)
    hidden = case.model.hidden_size
    cached = attn_res_point(
        {**_context(case), "model": replace(case.model, hidden_size=64)},
        prefix_id="prefix", addend_id=None, bank_ids=(None,) * 4, valid_blocks=0,
        write_slot=0, output_norm=True, mix=True, materialize=False,
        tokens_local=128, bank_capacity=4, block_token_stride=4 * 64,
        optimize_prefill=True)
    cached_op = _prepared(case, cached.ops[0], aiv=2, ub=192 * 1024)
    view = dict(cached_op.metadata)["attn_res_view"]
    assert view["used_cores"] == 2
    assert view["tokens_per_core"] == 64
    assert view["core_ranges"] == ((0, 64), (64, 64))
    assert view["output_norm_cached"] is True
    assert sum(core["token_count"] for core in dict(cached_op.metadata)["attn_res_core_work"]) == 128
    assert _read_bytes(cached_op) == 2 * 128 * 64 + 2 * 64 * 2

    bypass = attn_res_point(
        _context(case), prefix_id="prefix", addend_id=None, bank_ids=("b0",) * 8,
        valid_blocks=5, write_slot=None, output_norm=False, mix=False,
        materialize=False, tokens_local=5, bank_capacity=8,
        block_token_stride=8 * hidden)
    assert dict(bypass.ops[0].metadata)["attn_res_view"]["mix"] is False
    assert _read_bytes(_prepared(case, bypass.ops[0])) == 2 * 5 * hidden
    assert bypass.ops[0].deps == ("prefix",)


def test_actual_engine_obeys_input_ready_and_services_native_bytes_without_cube():
    case = _case(layers=3, block=2)
    ops = target_operators(case, phase="prefill", tokens=5, scope="body")
    for op in _points(ops)[:2]:
        prepared = _prepared(case, op)
        row = operator_costs(case, (op,), input_ready_ms=17)["operators"][0]
        assert row["service_before_input"] == {}
        assert not any(row["remaining_buffers"].values()) and row["live_tasks"] == 0
        assert row["declared_service"] == pytest.approx(row["serviced"], rel=1e-12, abs=1e-8)
        assert row["serviced"]["memory@r0:s0"] == prepared.hbm_bytes
        role = dict(prepared.metadata)["attn_res_read_role"]
        assert row["serviced"][role + "@r0:s0"] == _read_bytes(prepared)
        assert row["serviced"]["mte3@r0:s0"] == pytest.approx(prepared.mte3_bytes, rel=1e-12, abs=1e-8)
        assert "cube" not in row["serviced"]
        assert row["duration_ms"] > 0

    graphs = target_graphs(case, phase="prefill", tokens=5, scope="model")
    result = mindie_sim.graph_costs(case, graphs, isolated=False)
    assert result["duration_ms"] > 0
    assert result["operator_occurrences"] > 0
    assert sum(1 for op in graphs[0] if op.recipe == "attn_res_fwd") == 7
    report = export_structure(case, phase="prefill", tokens=5, scope="body")
    groups = [group for group in report["subgraphs"]
              if group["role"] == "attn_res_prefix_point"]
    assert len(groups) == 7 * len(report["execution"]["replicas"])
    nodes = {node["node_id"]: node for node in report["nodes"]}
    for group in groups:
        assert group["contract"] is None
        assert group["coverage"]["kind"] == "complete_subgraph"
        assert group["physical_work"]["launches"] == 1
        member = nodes[next(iter(group["node_ids"]))]
        assert member["physical_work"]["matrix_flops"] == {}
        assert member["operator"]["metadata"]["native_kind"] == "AttnResFwd"
        assert member["operator"]["metadata"]["attn_res_source"]["image_id"] == \
            "sha256:" + SOURCE["image_id"].split("sha256:")[-1]


def test_pp_source_domain_does_not_add_a_false_attnres_boundary_kernel():
    case = _case(layers=4, block=2)
    with pytest.raises(ModelingError, match="attn_res_source_caller_requires_pp1"):
        _frontend_body(case, stages=(0, 0, 1, 1))


def test_engram_or_hyper_selection_is_an_explicit_modeling_error():
    case = _case(layers=3, block=2)
    plans = tuple(replace(plan, engram=True) for plan in case.model.layer_plans)
    selected = replace(case, model=replace(case.model, layer_plans=plans))
    with pytest.raises(ModelingError, match="attn_res_selected_semantics_unsupported"):
        target_operators(selected, phase="prefill", tokens=5, scope="body")
