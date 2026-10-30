"""Shared structure preserves instance identity and declared kernel boundaries."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.structure import export_op_structure, export_structure, main
from test_mla_family_selection import mla_case
from test_prediction_system import case


def _nodes(report):
    return {row["node_id"]: row for row in report["nodes"]}


def _tiny_mlp_ops():
    """Declared from live_framework_capture_workload.py, with no capture input."""
    quant = QuantSpec("fp32", weight_format="fp32", activation_format="fp32")

    def linear(module_id, deps, in_features, out_features):
        return gemm_op(
            op_id=module_id, deps=deps, stage=0, layer=-1,
            requests=(), m=8, n=out_features, k=in_features, quant=quant,
            category="dense_mlp", family="linear", phase="eager_mlp",
            metadata=(
                ("module_id", module_id), ("module_type", "Linear"),
                ("input_shape", (8, in_features)),
                ("weight_shape", (out_features, in_features)),
                ("output_shape", (8, out_features)),
                ("bias", True), ("bias_shape", (out_features,)), ("bias_dtype", "fp32"),
            ),
        )

    return (
        linear("0", ("input",), 256, 512),
        vector_op(
            op_id="1", deps=("0",), stage=0, layer=-1, requests=(),
            elements=8 * 512, ops_per_element=1, dtype="fp32", family="relu",
            phase="eager_mlp", recipe="relu", shape=(8, 512),
            metadata=(("module_id", "1"), ("module_type", "ReLU")),
        ),
        linear("2", ("1",), 512, 256),
    )


def test_declared_mlp_preserves_matrix_bias_activation_and_dependency_contracts():
    ops = _tiny_mlp_ops()
    report = export_op_structure(
        ops, model={"kind": "sequential", "input_shape": [8, 256], "dtype": "fp32"},
        execution={"phase": "eager_mlp", "scope": "model",
                   "replica_axis": "data_parallel", "shard_scope": "unsharded"},
        source={"path": "profiler/tests/live_framework_capture_workload.py"},
    )
    nodes = _nodes(report)
    first, activation, last = (nodes[f"rank:0/{module}"] for module in ("0", "1", "2"))
    assert [node["contract"] for node in (first, activation, last)] == [
        ["matrix_bias", [8, 512, 256], "fp32", "fp32", "fp32", [512], "fp32"],
        ["relu", [8, 512], "fp32"],
        ["matrix_bias", [8, 256, 512], "fp32", "fp32", "fp32", [256], "fp32"],
    ]
    assert first["deps"] == ["rank:0/input"]
    assert activation["deps"] == [first["node_id"]]
    assert last["deps"] == [activation["node_id"]]
    assert report["external_inputs"] == ["rank:0/input"]
    assert [node["operator"]["metadata"]["module_id"] for node in
            (first, activation, last)] == ["0", "1", "2"]
    assert all(node["layer"] == -1 for node in nodes.values())
    assert first["operator"]["metadata"]["weight_shape"] == [512, 256]
    assert last["operator"]["metadata"]["output_shape"] == [8, 256]
    assert sum(op.flops for op in ops) == 2 * 8 * (256 * 512 + 512 * 256)
    assert ops[1].vector_ops == 8 * 512
    assert all(node["boundary"] == "semantic" for node in nodes.values())
    assert all("flops" not in node["operator"] for node in nodes.values())


def test_declared_graph_does_not_need_case_hardware_or_execute_timing(monkeypatch):
    import mindie_sim.structure as structure
    from mindie_sim.prediction.engine import Engine

    def unexpected(*args, **kwargs):
        raise AssertionError("declared graph export used a Case or timing engine")

    monkeypatch.setattr(structure, "parse_case", unexpected)
    monkeypatch.setattr(structure, "target_graphs", unexpected)
    monkeypatch.setattr(Engine, "advance", unexpected)
    ops = _tiny_mlp_ops()
    declaration = {"model": {"kind": "sequential"},
                   "execution": {"phase": "eager_mlp", "scope": "model"}}
    first = export_op_structure({3: ops}, **declaration, source={"revision": "one"})
    changed_work = tuple(replace(op, flops=1e12, hbm_bytes=1e12) for op in ops)
    second = export_op_structure({3: changed_work}, **declaration,
                                 source={"revision": "two"})
    # Resource declarations now distinguish physically heterogeneous tensor
    # candidates, without running a latency model during export.
    assert first["graph_id"] != second["graph_id"]
    assert [node["contract"] for node in first["nodes"]] == [
        node["contract"] for node in second["nodes"]]
    assert all(a["physical_signature"] != b["physical_signature"]
               for a, b in zip(first["nodes"], second["nodes"]))
    assert export_op_structure({3: ops}, **declaration, source={"revision": "two"})[
        "graph_id"] == first["graph_id"]
    assert first["source"] != second["source"]
    assert first["external_inputs"] == ["rank:3/input"]
    assert all(node["replica"] == 3 for node in first["nodes"])
    no_bias = replace(ops[0], metadata=tuple(
        (key, False if key == "bias" else value) for key, value in ops[0].metadata))
    unbiased = export_op_structure({3: (no_bias, *ops[1:])}, **declaration)
    assert unbiased["graph_id"] != first["graph_id"]
    assert unbiased["nodes"][0]["contract"][0] == "matrix"


def test_export_keeps_repeated_layer_instances_and_stable_content_identity():
    raw = case()
    first = export_structure(raw, phase="prefill", tokens=3, source={"revision": "one"})
    second = export_structure(deepcopy(raw), phase="prefill", tokens=3,
                              source={"revision": "two"})
    assert first["graph_id"] == second["graph_id"]
    payload = {key: value for key, value in first.items() if key not in {"graph_id", "source"}}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert first["graph_id"] == "sha256:" + sha256(canonical.encode("utf-8")).hexdigest()
    assert first["source"] != second["source"]
    nodes = _nodes(first)
    one, two = (nodes[f"rank:0/s0.l{layer}.qkv"] for layer in (0, 1))
    assert one["contract"] == two["contract"]
    assert one["layer"] == 0 and two["layer"] == 1
    assert one["node_id"] != two["node_id"]
    assert one["block"] == two["block"] == "attention"
    changed = export_structure(raw, phase="prefill", tokens=4)
    assert changed["graph_id"] != first["graph_id"]
    assert _nodes(changed)[one["node_id"]]["contract"][1][0] == 4


def test_export_is_independent_of_timing_and_does_not_execute_engine(monkeypatch):
    from mindie_sim.prediction.engine import Engine

    def no_execution(*args, **kwargs):
        raise AssertionError("structure export executed the timing engine")

    monkeypatch.setattr(Engine, "advance", no_execution)
    raw = case()
    first = export_structure(raw, phase="decode", tokens=1, context=16)
    raw["machine"]["device"]["rates"]["memory"] *= 100
    raw["machine"]["device"]["launch_us"] = 500
    raw["runtime"]["submission"] = {"scheduler_ms": 12, "completion_ms": 3}
    second = export_structure(raw, phase="decode", tokens=1, context=16,
                              expand_calls=False)
    first_unexpanded = export_structure(case(), phase="decode", tokens=1,
                                       context=16, expand_calls=False)
    assert second == first_unexpanded
    raw["runtime"]["kernels"] = {
        "recipes": {"qkv_projection": {"kind": "envelope"}}}
    implementation = export_structure(raw, phase="decode", tokens=1, context=16,
                                      expand_calls=False)
    assert implementation["graph_id"] != first_unexpanded["graph_id"]
    assert "duration_ms" not in json.dumps(first)
    assert all("flops" not in row["operator"] for row in first["nodes"])


def test_source_call_expansion_retains_parent_and_dependency_lineage():
    raw = mla_case()
    unexpanded = export_structure(raw, phase="decode", tokens=1, context=16,
                                 sequences=2, expand_calls=False)
    expanded = export_structure(raw, phase="decode", tokens=1, context=16, sequences=2)
    parent_id = "rank:0/s0.l0.latent_attention_decode"
    parent = _nodes(unexpanded)[parent_id]
    assert parent["boundary"] == "call"
    call = next(row for row in expanded["source_calls"] if row["node_id"] == parent_id)
    nodes = _nodes(expanded)
    children = [nodes[child] for child in call["children"]]
    assert len(children) == 2
    assert all(child["boundary"] == "kernel" for child in children)
    assert all(child["lineage"]["source_call_id"] == parent_id for child in children)
    assert all(child["layer"] == parent["layer"] and child["phase"] == "decode"
               for child in children)
    assert set(parent["deps"]) <= set(children[0]["deps"])
    assert children[0]["node_id"] in children[1]["deps"]
    assert nodes[parent_id]["deps"] == [children[1]["node_id"]]
    assert nodes["rank:0/s0.l0.v_expand"]["deps"] == [parent_id]
    assert call["operator"] == parent["operator"]


def test_fusion_retains_declared_semantic_members_and_norm_ffn_contracts():
    raw = case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "residual_norm"]
    report = export_structure(raw, phase="prefill", tokens=3)
    fused = [row for row in report["nodes"] if row["family"] == "add_rmsnorm"]
    assert fused
    assert all(row["lineage"]["fused_from"] == ["residual_add", "rmsnorm"] for row in fused)
    assert all(row["contract"] == ["add_rmsnorm", [3, 32], "bf16", True] for row in fused)
    swiglu = _nodes(report)["rank:0/s0.l0.swiglu"]
    assert swiglu["block"] == "ffn"
    assert swiglu["contract"] == ["swiglu", [3, 64], "bf16"]
    assert len(report["nodes"]) == len(_nodes(report))


def test_selected_fused_kernel_preserves_semantic_fusion_lineage():
    raw = case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "residual_norm"]
    raw["runtime"]["kernels"] = {
        "lowerings": {"add_rmsnorm": "vllm_cuda_rms_norm"}}
    report = export_structure(raw, phase="prefill", tokens=3)
    fused = [row for row in report["nodes"] if row["family"] == "add_rmsnorm"]
    assert fused
    assert all(row["boundary"] == "kernel" for row in fused)
    assert all(row["lineage"]["fused_from"] == ["residual_add", "rmsnorm"] for row in fused)
    assert all(row["contract"] == ["add_rmsnorm", [3, 32], "bf16", True] for row in fused)


def test_expert_cohort_preserves_replica_ids_and_external_body_inputs():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    report = export_structure(raw, phase="decode", tokens=1, context=8, scope="body")
    assert report["execution"]["replicas"] == [0, 1]
    assert report["external_inputs"] == ["rank:0/analysis_input", "rank:1/analysis_input"]
    by_replica = {rank: [row for row in report["nodes"] if row["replica"] == rank]
                  for rank in (0, 1)}
    assert len(by_replica[0]) == len(by_replica[1])
    assert {row["node_id"] for row in by_replica[0]}.isdisjoint(
        row["node_id"] for row in by_replica[1])


def test_replica_identity_is_not_a_physical_tensor_or_pipeline_rank():
    raw = case(dp=2, tp=2, pp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    report = export_structure(raw, phase="decode", tokens=1, context=8, scope="body")
    assert raw["machine"]["num_devices"] == 8
    assert report["execution"]["replicas"] == [0, 1]
    assert report["execution"]["replica_axis"] == "data_parallel"
    assert report["execution"]["shard_scope"] == "representative_tp_pcp_shard"
    assert report["execution"]["stage_scope"] == "all_pipeline_stages"
    for replica in (0, 1):
        nodes = [row for row in report["nodes"] if row["replica"] == replica]
        assert {row["stage"] for row in nodes} == {0, 1}
        assert {row["layer"] for row in nodes if row["layer"] >= 0} == {0, 1}
        assert all(row["node_id"].startswith(f"rank:{replica}/") for row in nodes)


def test_projection_preserves_complete_unequal_expert_traffic_and_owned_output():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    original = deepcopy(raw)
    args = dict(phase="decode", tokens=1, context=8, scope="body", sequences=[1, 3])
    full = export_structure(raw, **args)
    projected = export_structure(raw, **args, replicas=[0])
    assert projected["nodes"] == [node for node in full["nodes"] if node["replica"] == 0]
    assert projected["subgraphs"] == [group for group in full["subgraphs"]
                                      if group["subgraph_id"].startswith("rank:0/")]
    assert projected["execution"]["cohort_replicas"] == [0, 1]
    assert projected["execution"]["sequences"] == [1, 3]
    assert projected["execution"]["replicas"] == [0]
    assert projected["graph_id"] != full["graph_id"]
    combine = next(n for n in projected["nodes"] if n["op_id"].endswith(".combine"))
    assert combine["operator"]["metadata"]["peer_traffic"]["direction"] == "transpose"
    # Output mutations cannot rewrite the caller's model/software declarations.
    projected["model"]["name"] = "changed output"
    projected["execution"]["software"]["parallel"]["tp"] = 100
    assert raw == original


def test_cli_writes_source_backed_structure(tmp_path):
    input_path, output_path = tmp_path / "case.json", tmp_path / "structure.json"
    input_path.write_text(json.dumps(case()), encoding="utf-8")
    assert main([str(input_path), "--phase", "prefill", "--tokens", "3",
                 "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text())
    expected = export_structure(case(), phase="prefill", tokens=3)
    assert report["graph_id"] == expected["graph_id"]
    assert report["nodes"] == expected["nodes"]
    assert report["source"]["path"] == str(input_path)
    assert len(report["source"]["sha256"]) == 64
