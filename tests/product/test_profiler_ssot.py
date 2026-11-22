"""Declared graphs, profiler descriptor contracts, and subgraph boundary math.

[Role] Exercise graph identity, source-call lineage and shared native contracts.
[Invariants] Descriptor tuples match exactly. Sealed graphs charge each fused resource once.
[Timeouts & Locks] Bounded local graph exports and fixture reads, with no device.
[Side-Effect Policy] Writes only CLI output under pytest's temporary directory.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace

import pytest

from _support.operator_cases import declared_case, mla_case
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weights import cache_row_bytes_by_layer
from mindie_sim.compose import compose_case
from mindie_sim.observed_contracts import CONTRACT_FIELDS, FIELDS, NATIVE_CONTRACT_FAMILIES, NATIVE_FAMILIES, observed_contract
from mindie_sim.observed_contracts.attention import observed_attention_contract
from mindie_sim.observed_contracts.signatures import observed_contract_diagnostic
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.engine import Engine
import mindie_sim.structure as structure
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_op_structure, export_structure, main
from mindie_sim.structure_implementations import native_candidates, native_implementation
from mindie_sim.structure.physical import node_physical
from mindie_sim.structure.physical import select_structure_targets
from mindie_sim.structure_subgraphs import subgraph_contract
from mindie_sim.subgraph_math import (ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE,
    SEMANTIC_FAMILIES, SubgraphMathSummary, attach_subgraph_math, boundary_bytes, build_subgraph_math_rollup,
    effective_physical_work, evaluate_subgraph_efficiency, exclusive_family_members, semantic_family_of_group,)

pytestmark = pytest.mark.tier1

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/observed_contracts/captured_descriptors.json"
SIM_ROOT = Path(__file__).resolve().parents[2] / "mindie_sim"
DECLARED_MODULES = ("structure_contracts.py", "structure/native.py", "structure_subgraphs.py",)

RECOVERED_FAMILIES = frozenset({"attention_paged_active_qkv", "attention_paged_mla_qk_rope",
    "situ_mxquant_active", "dynamic_quant_v2_active", "causal_conv1d_active",})
LEGACY_OPERATOR_FAMILIES = {"gdn_gating", "recurrent_vector", "swiglu"}
OBSERVED_ONLY_FAMILIES = {"dequant_swiglu_quant_active", "dynamic_quant_v2_active", "kv_cache_store_active"}

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

DESCRIPTOR_FIELDS = tuple(f"{side}_{field}" for side in ("input", "output") for field in ("shapes", "dtypes", "formats"))

@lru_cache(maxsize=16)
def _declared_contracts(model, hardware, tp, phase, tokens, context, layers):
    case = compose_case(model, hardware, tp=tp, prompt_tokens=tokens if phase == "prefill" else 8, output_tokens=4, layers=layers)
    # The captures predate optional native fusions and declare the eager boundary.
    case["runtime"].setdefault("metadata", {})["execution_policy"] = {"automatic_optional_fusions": False}
    report = export_structure(case, phase=phase, tokens=tokens, context=context, sequences=1, scope="body")
    return {_canonical(node["contract"]) for node in report["nodes"] if node["contract"]}

@pytest.mark.requires_profiler
def test_captured_descriptor_fixtures_equal_declared_node_contracts():
    from mindie_profiler import tensor_contracts as profiler_contracts
    assert profiler_contracts.observed_contract is observed_contract
    fixtures = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixtures["schema"] == "mindie-observed-contract-fixtures/1" and len(fixtures["exact_matches"]) >= 5
    seen = set()
    for record in fixtures["exact_matches"]:
        descriptors = {key: list(map(tuple, record[key])) if key.endswith("shapes") else tuple(record[key])
                       for key in DESCRIPTOR_FIELDS}
        row = SimpleNamespace(kind=record["kind"], **descriptors)
        actual = observed_contract(row)
        assert actual is not None, record["capture_name"]
        assert profiler_contracts.observed_contract(row) == actual
        assert _canonical(actual) == _canonical(record["contract"]), record["capture_name"]
        geometry = itemgetter("model", "hardware", "tp", "phase", "tokens", "context", "layers")(record)
        assert _canonical(actual) in _declared_contracts(*geometry)
        seen.add(actual[0])
    assert {"attention_active_qkv", "kv_cache_store_active_elements", "sparse_attention_active", "layernorm_active"} <= seen

def test_captured_family_inventory_is_declared_and_vocabulary_complete():
    inventory = json.loads(FIXTURE.read_text(encoding="utf-8"))["captured_inventory"]
    assert RECOVERED_FAMILIES - {"causal_conv1d_active"} <= {row["family"] for row in inventory}
    assert RECOVERED_FAMILIES <= set(FIELDS) & NATIVE_FAMILIES
    assert CONTRACT_FIELDS is FIELDS and NATIVE_CONTRACT_FAMILIES is NATIVE_FAMILIES
    for item in inventory:
        family, kind = itemgetter("family", "kind")(item)
        assert family == kind or family in FIELDS or family in LEGACY_OPERATOR_FAMILIES, item

def test_native_family_vocabulary_has_declared_owners():
    declared = {node.value for module in DECLARED_MODULES
                for node in ast.walk(ast.parse((SIM_ROOT / module).read_text(encoding="utf-8")))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert RECOVERED_FAMILIES - OBSERVED_ONLY_FAMILIES <= declared
    assert OBSERVED_ONLY_FAMILIES <= NATIVE_FAMILIES <= declared | OBSERVED_ONLY_FAMILIES

def _nodes(report):
    return {row["node_id"]: row for row in report["nodes"]}

EAGER_MLP = {"phase": "eager_mlp", "scope": "model"}
MLP_MODEL = {"kind": "sequential"}

def _tiny_mlp_ops():
    quant = QuantSpec("fp32", weight_format="fp32", activation_format="fp32")
    placement = dict(stage=0, layer=-1, requests=(), phase=EAGER_MLP["phase"])
    def linear(module_id, dep, width_in, width_out):
        metadata = dict(module_id=module_id, module_type="Linear", input_shape=(8, width_in),
                        weight_shape=(width_out, width_in), output_shape=(8, width_out),
                        bias=True, bias_shape=(width_out,), bias_dtype="fp32")
        return gemm_op(op_id=module_id, deps=(dep,), m=8, n=width_out, k=width_in, quant=quant,
                       category="dense_mlp", family="linear", metadata=tuple(metadata.items()), **placement)
    relu = vector_op(op_id="1", deps=("0",), elements=8 * 512, ops_per_element=1,
                     dtype="fp32", family="relu", recipe="relu", shape=(8, 512),
                     metadata=(("module_id", "1"), ("module_type", "ReLU")), **placement)
    return linear("0", "input", 256, 512), relu, linear("2", "1", 512, 256)

def test_declared_mlp_preserves_bias_activation_and_dependencies():
    ops = _tiny_mlp_ops()
    report = export_op_structure(ops, model=MLP_MODEL | {"input_shape": [8, 256], "dtype": "fp32"},
        execution=EAGER_MLP | {"replica_axis": "data_parallel", "shard_scope": "unsharded"},
        source={"path": "profiler/tests/live_framework_capture_workload.py"})
    nodes = _nodes(report)
    rows = [nodes[f"rank:0/{module}"] for module in ("0", "1", "2")]
    assert [row["contract"] for row in rows] == [["matrix_bias", [8, 512, 256], "fp32", "fp32", "fp32", [512], "fp32"],
        ["relu", [8, 512], "fp32"], ["matrix_bias", [8, 256, 512], "fp32", "fp32", "fp32", [256], "fp32"],]
    assert [row["deps"] for row in rows] == [["rank:0/input"], ["rank:0/0"], ["rank:0/1"]]
    assert report["external_inputs"] == rows[0]["deps"]
    metadata = [row["operator"]["metadata"] for row in rows]
    assert [item["module_id"] for item in metadata] == ["0", "1", "2"]
    assert metadata[0]["weight_shape"] == [512, 256] and metadata[2]["output_shape"] == [8, 256]
    assert sum(op.flops for op in ops) == 2 * 8 * (256 * 512 + 512 * 256)
    assert ops[1].vector_ops == 8 * 512
    assert all(row["layer"] == -1 and row["boundary"] == "semantic" for row in nodes.values())
    assert all("flops" not in row["operator"] for row in nodes.values())

def _unexpected_execution(*args, **kwargs):
    raise AssertionError("structure export entered the timing path")

def test_declared_graph_exports_without_case_or_timing(monkeypatch):
    for owner, name in ((structure, "parse_case"), (structure, "target_graphs"), (Engine, "advance")):
        monkeypatch.setattr(owner, name, _unexpected_execution)
    ops = _tiny_mlp_ops()
    def export(nodes, revision):
        return export_op_structure({3: nodes}, model=MLP_MODEL, execution=EAGER_MLP, source={"revision": revision})
    first = export(ops, "one")
    second = export(tuple(replace(op, flops=1e12, hbm_bytes=1e12) for op in ops), "two")
    assert first["graph_id"] != second["graph_id"] and first["source"] != second["source"]
    for before, after in zip(first["nodes"], second["nodes"], strict=True):
        assert before["contract"] == after["contract"]
        assert before["physical_signature"] != after["physical_signature"]
        assert before["replica"] == 3
    assert export(ops, "two")["graph_id"] == first["graph_id"]
    assert first["external_inputs"] == ["rank:3/input"]
    no_bias = replace(ops[0], metadata=tuple((dict(ops[0].metadata) | {"bias": False}).items()))
    unbiased = export((no_bias, *ops[1:]), "one")
    assert unbiased["graph_id"] != first["graph_id"] and unbiased["nodes"][0]["contract"][0] == "matrix"

def _assert_graph_seal(report):
    excluded = {"graph_id", "source", "native_calls", "native_call_coverage"}
    payload = {key: value for key, value in report.items() if key not in excluded}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    assert (report["schema_version"], report["graph_id"]) == ("mindie-model-structure/1", "sha256:" + digest.hexdigest())

def test_export_keeps_repeated_layer_instances_and_stable_content_identity():
    raw = declared_case()
    first = export_structure(raw, phase="prefill", tokens=3, source={"revision": "one"})
    second = export_structure(deepcopy(raw), phase="prefill", tokens=3, source={"revision": "two"})
    assert first["graph_id"] == second["graph_id"] and first["source"] != second["source"]
    _assert_graph_seal(first)
    nodes = _nodes(first)
    one, two = (nodes[f"rank:0/s0.l{layer}.qkv"] for layer in (0, 1))
    assert one["contract"] == two["contract"]
    assert (one["layer"], two["layer"]) == (0, 1) and one["node_id"] != two["node_id"]
    assert one["block"] == two["block"] == "attention"
    changed = export_structure(raw, phase="prefill", tokens=4)
    assert changed["graph_id"] != first["graph_id"]
    assert _nodes(changed)[one["node_id"]]["contract"][1][0] == 4

def test_export_is_independent_of_timing_and_does_not_execute_engine(monkeypatch):
    monkeypatch.setattr(Engine, "advance", _unexpected_execution)
    raw = declared_case()
    geometry = dict(phase="decode", tokens=1, context=16)
    expanded = export_structure(raw, **geometry)
    base = export_structure(raw, **geometry, expand_calls=False)
    raw["machine"]["device"]["rates"]["memory"] *= 100
    raw["machine"]["device"]["launch_us"] = 500
    raw["runtime"]["submission"] = {"scheduler_ms": 12, "completion_ms": 3}
    assert export_structure(raw, **geometry, expand_calls=False) == base
    raw["runtime"]["kernels"] = {"recipes": {"qkv_projection": {"kind": "tiled_matmul"}}}
    changed = export_structure(raw, **geometry, expand_calls=False)
    assert changed["graph_id"] != base["graph_id"]
    assert "duration_ms" not in json.dumps(expanded)
    assert all("flops" not in row["operator"] for row in expanded["nodes"])

def test_source_call_expansion_retains_parent_and_dependency_lineage():
    raw, geometry = mla_case(), dict(phase="decode", tokens=1, context=16, sequences=2)
    parent_id = "rank:0/s0.l0.latent_attention_decode"
    parent = _nodes(export_structure(raw, **geometry, expand_calls=False))[parent_id]
    expanded = export_structure(raw, **geometry)
    call = next(row for row in expanded["source_calls"] if row["node_id"] == parent_id)
    nodes = _nodes(expanded)
    first, last = (nodes[child] for child in call["children"])
    assert parent["boundary"] == "call" and call["operator"] == parent["operator"]
    for child in (first, last):
        assert itemgetter("boundary", "layer", "phase")(child) == ("kernel", parent["layer"], "decode")
        assert child["lineage"]["source_call_id"] == parent_id
    assert set(parent["deps"]) <= set(first["deps"]) and first["node_id"] in last["deps"]
    assert nodes[parent_id]["deps"] == [last["node_id"]] and nodes["rank:0/s0.l0.v_expand"]["deps"] == [parent_id]

@pytest.mark.parametrize("native", (False, True))
def test_fusion_retains_declared_members_and_contracts(native):
    raw = declared_case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "residual_norm"]
    if native:
        raw["runtime"]["kernels"] = {"lowerings": {"add_rmsnorm": "vllm_cuda_rms_norm"}}
    report = export_structure(raw, phase="prefill", tokens=3)
    fused = [row for row in report["nodes"] if row["family"] == "add_rmsnorm"]
    assert fused
    for row in fused:
        assert row["lineage"]["fused_from"] == ["residual_add", "rmsnorm"]
        assert row["contract"] == ["add_rmsnorm", [3, 32], "bf16", True]
    if native:
        assert all(row["boundary"] == "kernel" for row in fused)
    nodes = _nodes(report)
    assert itemgetter("block", "contract")(nodes["rank:0/s0.l0.swiglu"]) == ("ffn", ["swiglu", [3, 64], "bf16"])
    assert len(report["nodes"]) == len(nodes)

def _moe_case(**parallel):
    raw = declared_case(**parallel)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    return raw

@pytest.mark.parametrize("tp,pp,ep", ((1, 1, 2), (2, 2, 4)))
def test_replica_axes_and_external_inputs_preserve_physical_ownership(tp, pp, ep):
    raw = _moe_case(dp=2, tp=tp, pp=pp, ep=ep)
    report = export_structure(raw, phase="decode", tokens=1, context=8, scope="body")
    assert raw["machine"]["num_devices"] == 2 * tp * pp
    expected = {"replicas": [0, 1], "replica_axis": "data_parallel", "shard_scope": "representative_tp_pcp_shard",
                "stage_scope": "all_pipeline_stages"}
    assert {key: report["execution"][key] for key in expected} == expected
    assert report["external_inputs"] == [f"rank:{rank}/analysis_input" for rank in (0, 1)]
    left, right = ([row for row in report["nodes"] if row["replica"] == rank] for rank in (0, 1))
    assert len(left) == len(right) and {row["node_id"] for row in left}.isdisjoint(row["node_id"] for row in right)
    for replica, nodes in enumerate((left, right)):
        assert {row["stage"] for row in nodes} == set(range(pp))
        assert {row["layer"] for row in nodes if row["layer"] >= 0} == {0, 1}
        assert all(row["node_id"].startswith(f"rank:{replica}/") for row in nodes)

def test_projection_preserves_complete_unequal_expert_traffic_and_owned_output():
    raw = _moe_case(dp=2, ep=2)
    original = deepcopy(raw)
    args = dict(phase="decode", tokens=1, context=8, scope="body", sequences=[1, 3])
    full = export_structure(raw, **args)
    projected = export_structure(raw, **args, replicas=[0])
    assert projected["nodes"] == [node for node in full["nodes"] if node["replica"] == 0]
    assert projected["subgraphs"] == [group for group in full["subgraphs"] if group["subgraph_id"].startswith("rank:0/")]
    geometry = itemgetter("cohort_replicas", "sequences", "replicas")(projected["execution"])
    assert geometry == ([0, 1], [1, 3], [0])
    assert projected["graph_id"] != full["graph_id"]
    combine = next(n for n in projected["nodes"] if n["op_id"].endswith(".combine"))
    assert combine["operator"]["metadata"]["peer_traffic"]["direction"] == "transpose"
    projected["model"]["name"] = "changed output"
    projected["execution"]["software"]["parallel"]["tp"] = 100
    assert raw == original

def test_cli_writes_source_backed_structure(tmp_path):
    input_path, output_path = tmp_path / "case.json", tmp_path / "structure.json"
    input_path.write_text(json.dumps(declared_case()), encoding="utf-8")
    assert main([str(input_path), "--phase", "prefill", "--tokens", "3", "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text())
    expected = export_structure(declared_case(), phase="prefill", tokens=3)
    assert itemgetter("graph_id", "nodes")(report) == itemgetter("graph_id", "nodes")(expected)
    assert report["source"] == {"kind": "case_file", "sha256": sha256(input_path.read_bytes()).hexdigest(),}

MLA_SPLIT_PROFILE = "mla_prolog_arch35_split_m_n_ae923905"

def _work(hbm=0.0, flops=0.0, vector=0.0, link=0.0):
    return {"hbm_bytes": hbm, "flops": flops, "vector_ops": vector, "link_bytes": link}

def _node(name, hbm=0, flops=0, *, link=0, boundary=None):
    return dict(node_id=name, physical_work=_work(hbm, flops, link=link), resource_boundary=boundary, operator={"metadata": {}})

def _report(fused):
    nodes = [_node("a", 10, 100), _node("b", 20, 200)]
    boundary = {"kind": "internal_to_selected_native_fusion", "operator_ids": ["res:0"]}
    nodes += ([_node(name, boundary=boundary) for name in ("c", "d")] if fused else [_node("c", 30, 300), _node("d", 10, 100)])
    groups = [dict(subgraph_id=f"sg:{name}", node_ids=members, role=role, family=family,
             phase="decode", replica=0, stage=0, structure={}) for name, members, role, family in (
            ("ffn", ["a", "b"], "router_selection", "moe"), ("attn", ["c", "d"], "mla_prolog", "mla"))]
    return dict(nodes=nodes, resource_operators=[_node("res:0", 40, 400, link=8)] if fused else [],
                subgraphs=groups, execution={"phase": "decode", "software": {"parallel": {"tp": 8}}},
                model={"quant": {"kv_dtype": "bf16", "weight_format": "int8", "activation_format": "int8"}})

def test_effective_physical_work_charges_fusion_once_and_detects_partial():
    fused = _report(True)
    assert effective_physical_work(_report(False), ["a", "b"]) == (_work(30, 300), ())
    assert effective_physical_work(fused, ["c", "d"]) == (_work(40, 400, link=8), ())
    assert effective_physical_work(fused, ["c"]) == (None, ("partial_selected_fusion:res:0",))
    with pytest.raises(ValueError, match="unknown_subgraph_member"):
        effective_physical_work(fused, ["missing"])

def test_effective_physical_work_reports_incomplete_ledgers():
    broken, dangling = _report(False), _report(True)
    broken["nodes"][0]["physical_work"]["hbm_bytes"] = None
    assert effective_physical_work(broken, ["a"]) == (None, ("physical_work_incomplete:a",))
    dangling["resource_operators"] = []
    with pytest.raises(ValueError, match="dangling_resource_boundary"):
        effective_physical_work(dangling, ["c", "d"])

@pytest.mark.parametrize("fused", (False, True))
def test_algebraic_work_is_fusion_invariant_even_when_unavailable(fused):
    report = _report(fused)
    attach_subgraph_math(report)
    summary, = [group["math_summary"] for group in report["subgraphs"] if group["subgraph_id"] == "sg:attn"]
    assert itemgetter("semantic_family", "realized_hbm_bytes")(summary) == ("attention", 40.0)
    assert all(summary[key] is None for key in ("algebraic_cube_flops", "algebraic_vector_ops", "algebraic_cube_flops_by_dtype",
        "minimal_boundary_hbm_bytes", "fusion_hbm_overhead_bytes"))
    assert {ALGEBRAIC_WORK_UNAVAILABLE, BOUNDARY_LEDGER_UNAVAILABLE} <= set(summary["unavailable_reasons"])

@pytest.mark.parametrize("complete,cold,expected", ((True, True, 180), (False, True, None), (True, False, None)))
def test_boundary_bytes_unions_overlaps_and_requires_complete_cold_ledger(complete, cold, expected):
    accesses = [("kv", "read", 0, 100), ("kv", "read", 50, 150), ("weight", "read", 0, 10), ("out", "write", 0, 20)]
    assert boundary_bytes(accesses, complete=complete, hbm_cold=cold) == expected

@pytest.mark.parametrize("access", (("alloc", "sideways", 0, 1), ("alloc", "read", 5, 1)))
def test_boundary_bytes_rejects_invalid_access(access):
    with pytest.raises(ValueError, match="invalid_boundary_access"):
        boundary_bytes([access], complete=True, hbm_cold=True)

def test_exclusive_family_members_unions_and_rejects_overlap():
    groups = [{"node_ids": ["a", "b"], "role": "router_selection"},
        {"node_ids": ["b", "c"], "role": "grouped_gate_up_activation_quant"},
        {"node_ids": ["d"], "role": "unknown_role", "family": "unknown"},]
    assert exclusive_family_members(groups, semantic_family_of_group)["moe_ffn"] == {"a", "b", "c"}
    assert semantic_family_of_group(groups[-1]) is None
    overlap = [{"node_ids": ["a"], "role": "router_selection"}, {"node_ids": ["a"], "role": "mla_prolog"},]
    with pytest.raises(ValueError, match="overlapping_semantic_family_ownership"):
        exclusive_family_members(overlap, semantic_family_of_group)

def _summary(**overrides):
    payload = dict(subgraph_id="sg", subgraph_kind="role", semantic_family="attention",
        phase="decode", layer_multiplicity=1, parameterization={}, algebraic_cube_flops=None, algebraic_cube_flops_by_dtype=None,
        algebraic_vector_ops=None, minimal_boundary_hbm_bytes=None, realized_hbm_bytes=None, unavailable_reasons=(),)
    return SubgraphMathSummary(**(payload | overrides))

PEAKS = {"compute_peaks": {"bf16": 1.0e15, "fp8": 2.0e15}, "vector_peak_ops_per_s": 1.0e14, "hbm_peak_bytes_per_s": 1.0e12,}

def _efficiency(summary, duration=1.0, **options):
    return evaluate_subgraph_efficiency(summary, duration_s=duration, hardware_peaks=PEAKS, **options)

def test_evaluate_subgraph_efficiency_scales_and_uses_per_dtype_peaks():
    summary = _summary(algebraic_cube_flops_by_dtype={"bf16": 1e12, "fp8": 5e11},
                       algebraic_vector_ops=2e11, minimal_boundary_hbm_bytes=1e8, realized_hbm_bytes=2e8)
    report = _efficiency(summary, 1e-3, invocations=2)
    metrics = itemgetter("algebraic_cube_mfu", "algebraic_vector_mfu", "realized_mbu",
                         "minimal_boundary_mbu", "fusion_headroom_bytes")
    assert metrics(report) == pytest.approx((2.0 + 0.5, 4.0, 0.4, 0.2, 2e8))
    assert report["byte_accounting_consistent"] is True

@pytest.mark.parametrize("by_dtype,flops,expected", (({"hif8": 1.0}, None, None),
                                                     ({}, 0.0, 0.0), ({"fp8_e4m3": 1e12}, None, 5e-4)))
def test_cube_efficiency_requires_compatible_dtype_peak(by_dtype, flops, expected):
    summary = _summary(algebraic_cube_flops=flops, algebraic_cube_flops_by_dtype=by_dtype)
    assert _efficiency(summary)["algebraic_cube_mfu"] == (None if expected is None else pytest.approx(expected))

def test_evaluate_subgraph_efficiency_preserves_observed_zero():
    summary = _summary(realized_hbm_bytes=2e8, minimal_boundary_hbm_bytes=1e8)
    report = _efficiency(summary, observed_hbm_bytes=0.0)
    assert itemgetter("realized_mbu", "fusion_headroom_bytes")(report) == (0.0, -1e8)
    assert report["byte_accounting_consistent"] is False

@pytest.mark.parametrize("work,options,error", (({}, {"duration": 0.0}, "duration_s_must_be_positive_finite"),
    ({}, {"invocations": 0}, "complete_invocation_count_required"), ({"algebraic_vector_ops": -1.0}, {}, "invalid_work"),))
def test_subgraph_efficiency_rejects_invalid_measurement_or_work(work, options, error):
    with pytest.raises(ValueError, match=error):
        _efficiency(_summary(realized_hbm_bytes=2e8, minimal_boundary_hbm_bytes=1e8, **work), **options)

def _export(model, *, tokens=128, layers=2, fuse=False):
    case = compose_case(model, "a3", tp=8, prompt_tokens=tokens, output_tokens=4, layers=layers)
    if fuse:
        case["runtime"]["kernels"].setdefault("lowerings", {})["native_fusion"] = {
            "implementation": "declared_native_fusion", "options": {"roles": ["mla_prolog"],
                        "traffic_profile": MLA_SPLIT_PROFILE},}
    return export_structure(case, phase="prefill", tokens=tokens, context=0, sequences=1, scope="body")

MATH_FIELDS = ("algebraic_cube_flops", "algebraic_vector_ops", "minimal_boundary_hbm_bytes")

def test_export_attaches_real_math_before_digest():
    _assert_graph_seal(report := _export("qwen3-30b-a3b", tokens=32, layers=1))
    rollup = report["subgraph_math_by_family"]
    assert rollup["totals_additive"] is False
    assert itemgetter("scope", "semantic_families")(rollup) == ("declared_semantic_subgraph_union", list(SEMANTIC_FAMILIES))
    assert report["subgraphs"] and report["resource_operators"] == []
    nodes = _nodes(report)
    for group in report["subgraphs"]:
        summary = group["math_summary"]
        member_sum = sum(nodes[node_id]["physical_work"]["hbm_bytes"] for node_id in group["node_ids"])
        assert summary["realized_hbm_bytes"] == pytest.approx(member_sum)
        assert all(summary[key] is not None for key in MATH_FIELDS)
        assert summary["fusion_hbm_overhead_bytes"] >= 0.0
        assert 0.0 < summary["fusion_byte_efficiency"] <= 1.0
        assert summary["unavailable_reasons"] == []
        breakdown = summary["boundary_breakdown_bytes"]
        assert breakdown["minimal_boundary_hbm_bytes"] == pytest.approx(summary["minimal_boundary_hbm_bytes"])
        assert (breakdown["unfused_member_hbm_bytes"] - breakdown["eliminated_internal_roundtrip_bytes"]
        ) == pytest.approx(summary["minimal_boundary_hbm_bytes"])
    rollup = build_subgraph_math_rollup(report)
    assert set(rollup["families"]) <= set(SEMANTIC_FAMILIES) and rollup["uncovered_node_ids"] == []
    for family in rollup["families"].values():
        assert all(family[key] is not None for key in MATH_FIELDS) and family["unavailable_reasons"] == []

def test_selected_fusion_charges_resource_not_members_twice():
    baseline, fused = (_export("deepseek-v3.2", fuse=selected) for selected in (False, True))
    assert baseline["resource_operators"] == [] and fused["resource_operators"]
    target = next(group for group in fused["subgraphs"] if group.get("resource_boundary"))
    baseline_summary = next(group["math_summary"] for group in baseline["subgraphs"]
        if group["subgraph_id"] == target["subgraph_id"])
    summary = target["math_summary"]
    resource = fused["resource_operators"][0]
    assert summary["realized_hbm_bytes"] == pytest.approx(resource["physical_work"]["hbm_bytes"])
    for key in ("algebraic_cube_flops", "algebraic_vector_ops"):
        assert summary[key] == pytest.approx(baseline_summary[key])
    field = "algebraic_cube_flops_by_dtype"
    assert dict(summary[field]) == pytest.approx(dict(baseline_summary[field]))
    assert all(summary[key] < baseline_summary[key] for key in ("realized_hbm_bytes", "fusion_hbm_overhead_bytes"))
    assert 0.0 < summary["fusion_byte_efficiency"] <= 1.0
    assert summary["unavailable_reasons"] == baseline_summary["unavailable_reasons"] == []
    baseline_families, fused_families = (report["subgraph_math_by_family"]["families"] for report in (baseline, fused))
    assert set(fused_families) == set(baseline_families)
    algebraic_work = itemgetter("algebraic_cube_flops", "algebraic_vector_ops")
    for family, row in fused_families.items():
        assert algebraic_work(row) == pytest.approx(algebraic_work(baseline_families[family]))
        assert row["unavailable_reasons"] == []
    assert target["node_ids"] and resource["node_id"] not in target["node_ids"]

def _assert_boundary_roundtrip(summary, expected):
    ledger = summary["boundary_breakdown_bytes"]
    assert ledger["eliminated_internal_roundtrip_bytes"] == pytest.approx(expected)
    minimal = ledger["unfused_member_hbm_bytes"] - expected
    assert [row["minimal_boundary_hbm_bytes"] for row in (ledger, summary)] == pytest.approx([minimal] * 2)

def test_hc_pre_internal_roundtrip_matches_tensor_work():
    report = _export("deepseek-v4.1-flash")
    group = next(group for group in report["subgraphs"] if group.get("role") == "hc_pre")
    by_id = _nodes(report)
    def mte3(suffix):
        node_id = next(key for key in group["node_ids"] if key.endswith(suffix))
        return by_id[node_id]["physical_work"]["mte3_bytes"]
    expected = 3 * mte3(".hc_values_fp32") + 2 * sum(mte3(suffix) for suffix in (".hc_project", ".hc_inv_rms", ".hc_affine"))
    summary = group["math_summary"]
    _assert_boundary_roundtrip(summary, expected)
    assert summary["unavailable_reasons"] == []

def _case(model, **overrides):
    options = {"hardware": "Ascend950DT_9582", "graph_capture_mode": "none", "prompt_tokens": 128, "output_tokens": 2,}
    return parse_case(compose_case(model, **(options | overrides)))

def _observed_row(kind, inputs, outputs):
    descriptors = (*zip(*inputs, strict=True), *zip(*outputs, strict=True))
    return SimpleNamespace(kind=kind, **dict(zip(DESCRIPTOR_FIELDS, descriptors, strict=True)))

def _mixed_quant_contracts(case, context):
    report = export_structure(case, phase="decode", tokens=1, context=context, scope="model")
    return [node["contract"] for node in report["nodes"] if node.get("recipe") == "aclnn_mixed_quant_sparse_flash_mla"]

def test_observed_mqsmla_spellings_share_the_active_contract():
    query = ((1, 64, 512), "bf16", "ND")
    empty = ((), None, None)
    inputs = [query, empty, ((16, 1, 1, 544), "uint8", "ND"), ((1, 16), "int32", "ND"), empty, empty, empty,]
    case = _case("deepseek-v4.1-flash", tp=1, dp=1, ep=1, layers=2)
    declared = _mixed_quant_contracts(case, context=2048)
    assert len(declared) == 2
    for kind, expected in (("MqsmlaKernel___call__", "MqsmlaKernel___call__"),
        ("MqsmlaKernel", "MqsmlaKernel___call__"), ("aclnnMixedQuantSparseFlashMla", "aclnnMixedQuantSparseFlashMla"),
        ("MixedQuantSparseFlashMla", "aclnnMixedQuantSparseFlashMla"),):
        contract = observed_attention_contract(_observed_row(kind, inputs, [query, empty]))
        assert contract is not None
        assert contract[:2] == ("sparse_attention_active", expected)
        assert contract[2][0][0] == query[0]
        if expected == "aclnnMixedQuantSparseFlashMla":
            assert list(contract[:2]) == list(declared[0][:2])

def test_compressor_contract_counts_inputs_by_ratio():
    view = {"tokens": 8, "hidden": 5120, "width": 512, "coff": 2, "dtype": "bf16", "weight_dtype": "bf16", "ape_dtype": "fp32",
        "norm_dtype": "bf16", "state_dtype": "fp32", "rope_width": 64, "rope_dtype": "bf16", "ratio": 1,}
    for ratio, count in ((1, 2), (4, 4)):
        row = dict(implementation={}, coverage={}, tensor_view=dict(view, ratio=ratio), role="compressor")
        assert len(subgraph_contract(row)[3]) == count

def test_mla_prolog_native_identity_matches_weight_representation():
    view = {"tokens": 2, "hidden": 7168, "heads": 16, "query_rank": 1536, "latent_rank": 512, "nope_width": 128, "rope_width": 64,
        "projection_weight_dtypes": ("fp8_e4m3", "fp8_e4m3", "fp8_e4m3"),
        "norm_dtype": "bf16", "cache_dtype": "int8", "query_norm_output": True, "query_norm_dtype": "fp8_e4m3",
        "token_dtype": "fp8_e4m3", "absorbed_weight_dtype": "bf16", "packed_cache": {"row_bytes": 656},}
    for declared_kind in ("aclnnMlaPrologV4WeightNz", "MlaPrologV4WeightNz", "MlaPrologV4"):
        declared = subgraph_contract({"implementation": {"native_kind": declared_kind},
            "coverage": {}, "tensor_view": view, "role": "mla_prolog",})
        assert declared[:2] == ("mla_prolog_active", "aclnnMlaPrologV4WeightNz")
    empty_bf16 = ((), "bf16", "ND")
    empty_none = ((), None, None)
    v4_inputs = [((2, 7168), "fp8_e4m3", "ND"), ((7168, 1536), "fp8_e4m3", "FRACTAL_NZ"),
        ((1536, 3072), "fp8_e4m3", "FRACTAL_NZ"), ((16, 128, 512), "bf16", "ND"),
        ((7168, 576), "fp8_e4m3", "FRACTAL_NZ"), ((1536,), "bf16", "ND"), ((512,), "bf16", "ND"),
        empty_none, empty_none, ((16, 128, 1, 656), "int8", "ND"), empty_bf16,] + [empty_none] * 10
    v4_outputs = [((2, 16, 512), "bf16", "ND"), ((2, 16, 64), "bf16", "ND"),
        ((16, 128, 1, 656), "int8", "ND"), empty_bf16, empty_none, ((2, 1536), "fp8_e4m3", "ND"), ((2, 1), "fp32", "ND"),]
    for observed_kind in ("aclnnMlaPrologV4WeightNz", "MlaPrologV4WeightNz"):
        row = _observed_row(observed_kind, v4_inputs, v4_outputs)
        assert observed_contract(row) == declared
        assert observed_contract_diagnostic(row)["reason"] == ("observed_tensor_contract_available")
    assert observed_contract(_observed_row("MlaPrologV3", v4_inputs, v4_outputs)) is None
    v3_inputs = [((2, 7168), "int8", "ND"), ((7168, 1536), "int8", "FRACTAL_NZ"),
        ((1536, 3072), "int8", "FRACTAL_NZ"), *v4_inputs[3:4], ((7168, 576), "int8", "FRACTAL_NZ"), *v4_inputs[5:],]
    v3_outputs = [*v4_outputs[:5], ((2, 1536), "int8", "ND"), v4_outputs[6]]
    assert (v3_contract := observed_contract(_observed_row("MlaPrologV3", v3_inputs, v3_outputs))) is not None
    assert v3_contract[:2] == ("mla_prolog_active", "MlaPrologV3")
    assert observed_contract(_observed_row("aclnnMlaPrologV4WeightNz", v3_inputs, v3_outputs)) is None

def test_explicit_cache_epilog_native_kind_is_a_candidate():
    op = {"op_id": "cache", "family": "cache_store", "recipe": "indexer_cache",
        "source_boundary": None, "dtype": "bf16", "local_shape": (1, 128), "metadata": {"native_kind": "aclnnIndexerQuantCache"},}
    candidate_rows, execution, gaps = native_candidates(op, None)
    candidate, = candidate_rows
    assert native_implementation(op, None)["native_kinds"] == (candidate["native"],) == ("aclnnIndexerQuantCache",)
    assert candidate["selected"] is False and execution["selection_state"] == "single_compatible_unselected" and not gaps

def test_selective_structure_hash_includes_onboarding_hints():
    case = _case("qwen3.5-35b-a3b", hardware="Ascend910_9392", layers=2, prompt_tokens=8, output_tokens=2)
    plans = tuple(replace(plan, attention="unmodeled", architecture="custom_attention") for plan in case.model.layer_plans)
    case = replace(case, model=replace(case.model, attention="unmodeled", layer_plans=plans))
    geometry = dict(phase="decode", tokens=1, context=8, scope="model")
    full = export_structure(case, **geometry)
    assert full["unmodeled_structure_onboarding_hints"]
    targets = [row["node_id"] for row in full["nodes"][:4]]
    assert select_structure_targets(case, targets, **geometry)["graph_id"] == full["graph_id"]

def test_draft_prefixed_occurrences_share_a_physical_signature():
    common = dict(family="grouped_gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=(), semantic_shape=(1, 2), local_shape=(1, 2), padded_shape=(1, 2), launch_shape=(1, 2), flops=1, hbm_bytes=1,
        mte2_bytes=0, mte3_bytes=0, vector_ops=0, link_bytes=0,
        link_scope="", recipe="grouped_gate_up", coverage=None, dtype="bf16", layout="ND",)
    ops = [OpSpec(op_id=name, metadata=(("producer", f"draft0.s0.l{layer}.expert_act"),), **common)
           for layer, name in enumerate(("first", "second"))]
    assert len({node_physical(op)["physical_signature"] for op in ops}) == 1

def test_moe_init_routing_projects_the_mxfp8_contract():
    inputs = [((4, 4096), "bf16", "ND"), ((4, 8), "int32", "ND")]
    outputs = [((32, 4096), "fp8_e4m3", "ND"), ((32,), "int32", "ND"), ((288,), "int64", "ND"), ((32, 128), "fp8_e8m0", "ND"),]
    row = _observed_row("MoeInitRoutingV3", inputs, outputs)
    assert observed_contract(row) == ("moe_init_routing_mxfp8_active", "MoeInitRoutingV3", tuple(inputs), tuple(outputs))
    assert observed_contract_diagnostic(row)["reason"] == ("observed_tensor_contract_available")

@pytest.mark.parametrize(("scale_dtype", "expected_row_bytes"), [("bf16", 608), ("uint8", 584),])
def test_mixed_quant_cache_rows_agree_across_capacity_and_contracts(scale_dtype, expected_row_bytes):
    raw = compose_case("deepseek-v4.1-flash", hardware="Ascend950PR_958b", tp=4, dp=1, ep=4,
        layers=4, prompt_tokens=128, output_tokens=2, graph_capture_mode="none",)
    region = dict(dtype="fp8_e4m3", group_size=32, scale_dtype=scale_dtype)
    raw["model"]["metadata"]["cache_representations"].update(window=dict(region), compressed_kv=dict(region))
    case = parse_case(raw)
    assert dict(cache_row_bytes_by_layer(case.model))[0] == expected_row_bytes
    layer0 = layer_cache_components(case.model, case.model.layer_plans[0], case.runtime.parallel, 128, runtime=case.runtime)
    assert layer0["window"] == pytest.approx(128 * expected_row_bytes)
    Memory(case).allocation_report(512)
    contracts = _mixed_quant_contracts(case, context=512)
    assert len(contracts) == 4 and contracts[2][3] == [[[1, expected_row_bytes], "uint8", "ND"]] * 2
    query = ((1, 16, 512), "bf16", "ND")
    empty = ((), None, None)
    obs_inputs = [query, empty, ((16, 1, 1, expected_row_bytes), "uint8", "ND"),
        ((1, 16), "int32", "ND"), empty, empty, empty, ((16, 1, 1, expected_row_bytes), "uint8", "ND"),
        ((1, 16), "int32", "ND"), empty,]
    def observe(inputs):
        return observed_attention_contract(_observed_row("aclnnMixedQuantSparseFlashMla", inputs, [query, empty]))
    observed = observe(obs_inputs)
    assert observed is not None and observed[3] == (((1, expected_row_bytes), "uint8", "ND"),) * 2
    obs_inputs[7] = ((16, 1, 1, 320), "uint8", "ND")
    assert observe(obs_inputs) is None

def test_subgraph_boundary_respects_producer_and_consumer_caps():
    case = _case("deepseek-v3.2", tp=8, ep=8, layers=4, quant="w8a8")
    exported = export_structure(case, phase="prefill", tokens=128, context=0, scope="model")
    prolog = next(group for group in exported["subgraphs"] if group["subgraph_id"].endswith("l0.mla_prolog"))
    nodes, members = _nodes(exported), set(prolog["node_ids"])
    dependencies = {dep for name in members for dep in nodes[name]["deps"]}
    internal_producers = (members & dependencies) - set(prolog["external_outputs"])
    produced = {node: float(nodes[node]["physical_work"]["mte3_bytes"]) for node in internal_producers}
    writes, max_reads = sum(produced.values()), 0.0
    for consumer in (nodes[node] for node in members):
        cap = float(consumer["physical_work"]["mte2_bytes"])
        max_reads += min(cap, sum(min(cap, produced[dep]) for dep in consumer["deps"] if dep in produced))
    assert max_reads == pytest.approx(1_983_488.0)
    _assert_boundary_roundtrip(prolog["math_summary"], writes + max_reads)
