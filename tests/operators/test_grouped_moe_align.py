"""Grouped MoE alignment, count-list conversion and source init routing.

[Role] Exercise routing preparation through source builders and the Engine.
[Invariants] Preserve expert intervals, workspace bounds, work and ownership.
[Timeouts & Locks] Finite analytical execution with bounded active CTA streams.
[Side-Effect Policy] Read retained case inputs and simulate in memory.
"""
from collections import Counter, deque
from dataclasses import replace
from itertools import accumulate
import json
from math import ceil
from pathlib import Path

import pytest

from _support.axes import SKUS
from _support.hardware import hardware_case, machine
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators import grouped_moe
from mindie_sim.schema.operators import OpSpec

from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.family_moe_init import apply, capabilities
from mindie_sim.prediction.moe_cann_fusion import absorb_init_routing
from mindie_sim.prediction.moe_init_instructions import source_operations
from mindie_sim.prediction.moe_init_mxfp8 import RECIPE, row_runs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure_contracts import predicted_contract

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
INDEX_BYTES = 4
WARP = 32
INIT_WIDTH = 4096
INIT_EXPERTS = 288
INIT_TOP_K = 8
MX_GROUP = 32
SIMT_THREADS = 2048

@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("counts", [(1, 0, 1), (0, 17, 0, 3), (16, 16, 0), (0, 0, 0)])
def test_alignment_preserves_expert_intervals_and_publishes_only_valid_rows(hardware, counts):
    geometry = Geometry.from_devices(machine(hardware).dies[:1])
    ends = tuple(accumulate(counts))
    n = k = 16
    shape = (sum(counts), n, k)
    op = OpSpec("grouped", "grouped_gemm", "prefill", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", recipe="grouped_gemm", metadata=(("group_list", ends),))
    plan = grouped_moe.build(op, geometry)
    offsets = grouped_moe.normalize_group_rows(ends, cumulative=True, expected_total=sum(counts))
    assert offsets == tuple(zip((0, *ends[:-1]), counts))
    issues = sum(ceil(count / 16) for count in counts)
    assert plan.metadata["issued_fractals"] == issues
    assert plan.metadata["useful_flops"] == 2 * sum(counts) * n * k
    assert plan.work.resources.get("fixpipe", 0) == 2 * sum(counts) * n


@pytest.mark.parametrize("batch", [1, 4])
def test_count_list_scans_empty_experts_once_and_fused_consumer_waits(batch):
    path = ROOT / "examples/cases/longctx-decode-20261030/qwen35-35b-a3-tp4.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 2
    raw["runtime"].setdefault("kernels", {})["lowerings"] = {
        "group_list_prefix": "vllm_ascend_group_list_prefix",
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["grouped_gate_up_activation_quant"]}},
    }
    options = dict(phase="decode", tokens=2, sequences=batch, context=4096, scope="body")
    ops = target_graphs(raw, **options)[0]
    prefixes = [op for op in ops if op.recipe == "expert_group_list_cumsum"]
    experts = 256
    traffic = (2 * experts + 1) * 8
    assert len(prefixes) == raw["model"]["layers"]
    for scan in prefixes:
        assert (scan.local_shape, scan.hbm_bytes, dict(scan.metadata)["logical_integer_adds"]) == ((experts,), traffic, experts - 1)
        fused = next(op for op in ops if op.layer == scan.layer
                     and dict(op.metadata).get("fusion_role") == "grouped_gate_up_activation_quant")
        assert scan.op_id in fused.deps
        assert scan.op_id not in dict(fused.metadata)["fusion_semantic_ids"]
        down = next(op for op in ops if op.layer == scan.layer and op.recipe == "balanced_grouped_down")
        assert "group_list_producer" not in dict(down.metadata)
        report = operator_costs(raw, [scan], input_ready_ms=7)["operators"][0]
        assert report["serviced"] == {"memory@r0:s0": traffic}
        assert not any(report["service_before_input"].values()) and report["live_tasks"] == 0
    report = graph_costs(raw, {0: ops}, external_inputs={"analysis_input": 0}, isolated=False)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    raw["runtime"]["kernels"]["service_regimes"] = [dict(id="scan-control", match={"recipe": prefixes[0].recipe},
        response={"fixed_us": 40}, provenance={"kind": "synthetic-control"})]
    inclusive = operator_costs(raw, prefixes[:1])["operators"][0]
    assert (inclusive["duration_ms"], inclusive["serviced"]["memory@r0:s0"]) == pytest.approx((.04, traffic))
    del raw["runtime"]["kernels"]["lowerings"]["group_list_prefix"]
    assert all(op.recipe != "expert_group_list_cumsum" for op in target_graphs(raw, **options)[0])

def _init_case(batch, *, selected=True, fused=False):
    path = ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9"
    raw = json.loads((path / f"a5_glm53flash_tp8_c{batch}_p4k.json").read_text())
    bindings = raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})
    if selected:
        bindings["moe_init_routing"] = RECIPE
    if fused:
        role = "grouped_gate_up_activation_quant"
        bindings["native_fusion"] = {"implementation": "declared_native_fusion", "options": {
            "roles": [role], "program_profiles": {role: "ascend_mxfp8_grouped_swiglu_quant"},
            "program_options": {role: {"l1_k": 512}},
        }}
    return raw

def _init_graph(raw, batch):
    return target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=batch)[0]

@pytest.mark.parametrize("batch", [16, 32, 64])
def test_init_replaces_pack_and_up_quantizer_and_conserves_expanded_outputs(batch):
    raw = _init_case(batch)
    baseline = {op.op_id: op for op in _init_graph(_init_case(batch, selected=False), batch)}
    ops = {op.op_id: op for op in _init_graph(raw, batch)}
    selected = [op for op in ops.values() if op.recipe == RECIPE]
    assert len(selected) == sum(key.endswith(".expert_up.quantize") for key in baseline) > 0
    assert [key for key in ops if key.endswith(".expert_down.quantize")] == [
        key for key in baseline if key.endswith(".expert_down.quantize")]
    assert not any(key.endswith(".expert_up.quantize") for key in ops)
    tokens, width, assignments = batch * 4, INIT_WIDTH, batch * 4 * INIT_TOP_K
    for op in selected:
        assert dict(op.metadata)["moe_init_config"] == dict(
            tokens=tokens, topk=INIT_TOP_K, width=width, experts=INIT_EXPERTS)
        up = ops[op.op_id.replace(".pack", ".expert_up")]
        assert up.deps == (op.op_id,)
        assert dict(up.metadata)["activation_scale_producer"] == op.op_id
        assert (up.flops, up.hbm_bytes) == (baseline[up.op_id].flops, baseline[up.op_id].hbm_bytes)
        contract = predicted_contract(op.to_dict())
        assert contract[2][0] == ((tokens, width), "bf16", "ND")
        assert contract[3][3] == ((assignments, width // MX_GROUP), "fp8_e8m0", "ND")
    row = operator_costs(raw, (selected[0],), input_ready_ms=7)["operators"][0]
    physical = row["operator"]
    program = physical["metadata"]["program_execution"]
    assert (program["quantized_input_rows"], program["assignments"]) == (tokens, assignments)
    assert (program["output_data_bytes"], program["output_scale_bytes"]) == (assignments * width, assignments * width // MX_GROUP)
    assert physical["mte2_bytes"] == 0 and physical["metadata"]["resource_work"]["mte2_aiv"] > tokens * width * 2
    assert row["live_tasks"] == 0 and not (row["service_before_input"] or any(row["remaining_buffers"].values()))
    assert all(abs(value) < 1e-6 for value in row["service_delta"].values())

@pytest.mark.parametrize("tokens,topk,units", [(3, 8, 4), (7, 5, 8), (9, 6, 4)])
def test_init_reloads_only_tokens_split_across_owner_intervals(tokens, topk, units):
    total = tokens * topk
    per = ceil(total / units)
    owners = [Counter(slot // topk for slot in range(start, min(start + per, total)))
              for start in range(0, total, per)]
    for owner, expected in enumerate(owners):
        assert list(row_runs(total, topk, units, owner)) == list(expected.items())
    assert sum(sum(counts.values()) for counts in owners) == total
    assert sum(map(len, owners)) == tokens + sum(start % topk != 0 for start in range(per, total, per))

def test_init_gates_and_fused_activation_dependency():
    caps = capabilities()
    assert (caps["input_dtypes"], caps["scale_group_size"]) == (("bf16",), MX_GROUP)
    assert "no_moe_transport_quantization" in caps["requires"]
    raw = _init_case(16, selected=False)
    ops = _init_graph(raw, 16)
    pack = next(op for op in ops if op.recipe == "route_pack")
    up = next(op for op in ops if op.recipe in {"grouped_gate_up", "balanced_grouped_gate_up"})
    up, binding = replace(up, deps=(pack.op_id,)), {caps["selector"]: {"implementation": RECIPE}}
    runtime = parse_case(raw).runtime
    with pytest.raises(ModelingError, match="width_multiple_of_256"):
        apply((pack, replace(up, local_shape=(*up.local_shape[:2], 100))), binding, runtime)
    with pytest.raises(ModelingError, match="complete_BF16_to_E4M3_MX32"):
        apply((replace(pack, dtype="fp16"), up), binding, runtime)
    quantizer = next(op for op in ops if op.op_id.endswith(".expert_up.quantize"))
    meta = dict(quantizer.metadata, quantization_scope="moe_dispatch_expert_input")
    quantizer = replace(quantizer, metadata=tuple(meta.items()))
    absorbed, retired = absorb_init_routing(pack=pack, quantizer=quantizer, ep=8)
    assert retired == quantizer.op_id and absorbed.vector_ops == pack.vector_ops + quantizer.vector_ops
    incompatible = replace(quantizer, metadata=tuple(dict(meta, activation_group_size=128).items()))
    for op, ep, owns_quant in ((quantizer, 1, False), (quantizer, 8, True), (incompatible, 8, False)):
        assert absorb_init_routing(pack=pack, quantizer=op, ep=ep, dispatch_owns_quant=owns_quant) == (pack, None)
    raw = _init_case(16, fused=True)
    ops = _init_graph(raw, 16)
    init = next(op for op in ops if op.recipe == RECIPE)
    up = next(op for op in ops if op.layer == init.layer and "fusion_program_profile" in dict(op.metadata))
    assert init.op_id in up.deps and not any(op.op_id.endswith(".expert_up.quantize") for op in ops)
    external = {dep: 11 for op in (init, up) for dep in op.deps if dep != init.op_id}
    report = graph_costs(raw, [init, up], external_inputs=external)
    rows = {row["op_id"]: row for row in report["timeline"]}
    assert rows[up.op_id]["ready_ms"] >= rows[init.op_id]["complete_ms"] >= rows[init.op_id]["ready_ms"] >= 11
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

@pytest.mark.parametrize("axis", ["mte2_aiv", "mte3", "vector", "hbm"])
def test_init_duration_responds_to_each_declared_physical_rate(axis):
    raw = _init_case(64)
    op, case = next(op for op in _init_graph(raw, 64) if op.recipe == RECIPE), parse_case(raw)
    results = []
    for factor in (.5, 1, 2):
        dies = tuple(replace(die, rates=tuple((key, rate * factor if key == axis else rate)
                                             for key, rate in die.rates)) for die in case.machine.dies)
        scaled = replace(case, machine=replace(case.machine, dies=dies))
        results.append(operator_costs(scaled, (op,))["operators"][0]["duration_ms"])
    assert results[0] >= results[1] >= results[2] and results[0] > results[2]

def _source_program(op, services):
    raw = _init_case(16)
    raw["runtime"]["kernels"]["invocations"] = [{"match": {}, "startup_ms": 0, "minimum_ms": 0}]
    raw["machine"].setdefault("device", {})["compute_arch"] = {"instruction_services": {
        role: {"unit": "input_blocks32" if role.endswith("32_blocks") else "int32_thread_accesses",
               "owner_axis": "vector", "issue_ns": rate, "provenance": {"kind": "synthetic-control"}}
        for role, rate in services.items()}}
    case = parse_case(raw)
    execution = Execution(case, CostModel(case), recipes=RecipeBook.from_mapping(case.runtime.kernels))
    prepared, recipe = execution.prepare(op, 0)
    context = execution._context(prepared, 0, None, recipe=recipe)
    engine = execution.engine
    engine.trace = deque(maxlen=12000)
    end = execution.lower_program(context, recipe).start(engine, "source", engine.submit("ready", release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(end,))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    meta = dict(prepared.metadata)["program_execution"]["source_operation_services"]
    return end.time - 7, engine, meta

@pytest.mark.parametrize("factor", [.5, 1, 2])
def test_sort_macros_serially_price_owner_zero_without_erasing_known_work(factor):
    op = next(op for op in _init_graph(_init_case(16), 16) if op.recipe == RECIPE)
    before, baseline, unavailable = _source_program(op, {})
    intervals = dict(source_concat32_blocks=2000 / factor, source_sort32_blocks=3000 / factor,
                     source_extract32_blocks=5000 / factor)
    after, engine, meta = _source_program(op, intervals)
    blocks = ceil(16 * 4 * INIT_TOP_K / WARP)
    assert after - before == pytest.approx(blocks * sum(intervals.values()) / 1e6)
    assert {key: engine.serviced[key] for key in baseline.serviced} == pytest.approx(baseline.serviced)
    assert any(row["task"].endswith("sort_known_lane_work") for row in baseline.trace)
    assert not any(row["task"].endswith("_macro") for row in baseline.trace)
    assert unavailable["sort"]["sort"]["unpriced_service_work"] == {"source_sort32_blocks": blocks}
    assert meta["sort"]["sort"]["priced_service_work"] == {"source_sort32_blocks": blocks}
    rows = [row for row in engine.trace if row["task"].endswith("_macro")]
    for row, (role, rate) in zip(rows, intervals.items(), strict=True):
        resource = role + "@r0:s0"
        assert (row["serviced"][resource], row["placement"][resource]) == (blocks, (0,))
        assert row["end_ms"] - row["start_ms"] == pytest.approx(blocks * rate / 1e6)
    post = next(row for row in engine.trace if row["task"].endswith("sort_post_lane_work"))
    assert all(left["end_ms"] <= right["start_ms"] for left, right in zip(rows, (*rows[1:], post)))

@pytest.mark.parametrize("assignments,units", [(15, 64), (65, 8), (1024, 40), (4099, 2)])
def test_simt_count_and_inverse_accesses_follow_thread_iterations(assignments, units):
    per = ceil(assignments / units)
    _, counts, inverse = source_operations(assignments, per, ceil(assignments / per), INIT_EXPERTS)
    for owner, (count, scatter) in enumerate(zip(counts, inverse)):
        rows = min(per, assignments - owner * per)
        visits = [i for thread in range(SIMT_THREADS) for i in range(thread, rows, SIMT_THREADS)]
        assert count.counts == {"source_simt_gm_load32": sum(2 + (i > 0) + (i < rows - 1) for i in visits)}
        assert scatter.counts == {"source_simt_gm_load32": rows, "source_simt_gm_store32": rows}

def test_simt_service_capacity_is_per_owner_and_keeps_payload():
    op = next(op for op in _init_graph(_init_case(16), 16) if op.recipe == RECIPE)
    meta = dict(op.metadata)
    meta["moe_init_config"] = dict(meta["moe_init_config"], tokens=3, topk=5)
    op = replace(op, metadata=tuple(meta.items()))
    _, baseline, unavailable = _source_program(op, {})
    load, store = "source_simt_gm_load32", "source_simt_gm_store32"
    _, engine, priced = _source_program(op, {load: 2000, store: 3000})
    assert tuple(engine.serviced[key + "@r0:s0"] for key in (load, store)) == pytest.approx((3 * 15, 15))
    assert {key: engine.serviced[key] for key in baseline.serviced} == pytest.approx(baseline.serviced)
    assert {row["placement"][load + "@r0:s0"] for row in engine.trace if load + "@r0:s0" in row["demand"]} == {
        (i,) for i in range(15)}
    assert priced["atomic_count"]["payload_bytes_per_owner"] == INIT_EXPERTS * INDEX_BYTES
    assert sum(row["counts"][load] for row in unavailable["count_by_owner"]) == 2 * 15
    assert not any("atomic_rmw" in key or "simt_warp" in key for key in engine.serviced)

@pytest.mark.parametrize("tokens", [5, 512])
def test_explicit_alignment_keeps_payload_pack_and_expert_dependencies(tokens):
    raw = hardware_case()
    raw["model"].update(hidden_size=128, ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    ids = [(i % 2, 2) for i in range(tokens)]
    histogram = Counter(expert for row in ids for expert in row)
    counts = tuple(histogram[i] for i in range(4))
    route = dict(scope="body", layer=0, phase="prefill", tokens=tokens, topk_ids=ids, expert_rows=counts)
    raw["runtime"] = {"routing": {"realizations": [route]}}
    options = dict(phase="prefill", tokens=tokens, scope="body")
    graph = target_graphs(raw, **options)
    ops = {op.op_id.split(".", 2)[2]: op for op in graph[0] if op.op_id.startswith("s0.l0.")}
    pack = ops["pack"]
    assert ops["topk"].op_id in pack.deps
    assert pack.op_id in ops["expert_up"].deps
    assert ops["expert_down"].op_id in ops["unpack"].deps
    assert grouped_moe.normalize_group_rows(counts, expected_total=2 * tokens)[-1] == (2 * tokens, 0)
    route["expert_rows"] = (0, 0, tokens, tokens)
    with pytest.raises(ModelingError, match="must_agree_with_expert_rows"):
        target_operators(raw, **options)


@pytest.mark.parametrize("issued", [1, 8])
def test_balanced_capture_separates_logical_rows_from_issued_pack_and_scale_rows(issued):
    raw = hardware_case()
    raw["machine"].update(num_devices=2, links=[{"name": "fabric", "members": [0, 1], "capacity_bytes_per_ms": 1000}])
    raw["model"].update(hidden_size=32, num_heads=4, num_kv_heads=2, ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32, quant=quant_preset("w8a8_int8").to_dict())
    transport = dict(kind="separate_quantized_payload", payload_dtype="int8", scale_dtype="fp32",
                     scale_granularity="row", histogram_dtype="int64")
    raw["runtime"] = dict(tp=2, ep=2, routing={"policy": "balanced"}, kernels={"moe_transport": transport})
    if issued > 1:
        raw["runtime"].update(graph_mode="auto", graph_capture_keys=[issued], graph_capture_mode="full")
    ops = {op.op_id.removeprefix("s0.l0."): op for op in target_graphs(raw, phase="decode", tokens=1, scope="body")[0]}
    gate = dict(ops["expert_up"].metadata)
    route = gate["routing_realization"]
    assert (route["valid_local_tokens"], route["padded_local_tokens"], sum(route["rank_assignment_rows"])) == (1, issued, 2 * issued)
    assert dict(ops["moe_histogram"].metadata)["source_assignments"] == issued
    assert ops["moe_transport_quantize"].mte3_bytes == issued * (32 + INDEX_BYTES)
    assert gate["activation_scale_bytes"] == issued * INDEX_BYTES
    assert tuple(ops[key].link_bytes for key in ("moe_payload_dispatch", "moe_scale_dispatch")) == (issued * 16, issued * 2)
