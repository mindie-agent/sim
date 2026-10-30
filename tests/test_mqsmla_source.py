"""Real source work, ordinary caller and shared packed-cache ABI controls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.prediction.family_mqsmla import FORMATS, RECIPE
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.mqsmla import (
    BUFFERS, OwnerTiles, build, owner_ranges, query_rows, tile_phases,
)
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


ROOT = Path(__file__).resolve().parents[1]


def case():
    raw = json.loads((ROOT / "examples/cases/high-concurrency-20261030/v41-dp8/"
                     "a5_v41_fp8_tp1_dp8_ep8_tpot5ms_c1.json").read_text())
    raw["model"].setdefault("metadata", {})["cache_representations"] = deepcopy(FORMATS)
    raw["runtime"]["kernels"].setdefault("lowerings", {})["mqsmla_attention"] = RECIPE
    return parse_case(raw)


@pytest.fixture(scope="module")
def ordinary():
    model = case()
    return model, {
        batch: target_graphs(model, phase="decode", tokens=1, context=2048, sequences=batch)
        for batch in (1, 8)}


def attention(ops, layer=2):
    return next(op for op in ops if op.layer == layer and op.recipe == RECIPE)


def test_packed_cache_producer_consumer_and_capacity_share_exact_row_bytes(ordinary):
    model, graphs = ordinary
    # This exercises the ordinary HF wrapper metadata path, not a post-parse patch.
    assert dict(model.model.metadata)["cache_representations"] == FORMATS
    layer = model.model.layer_plans[2]
    components = layer_cache_components(
        model.model, layer, model.runtime.parallel, 2049, capacity_padded=False)
    assert components["window"] == 128 * 544
    assert components["compressed_kv"] == (2049 // 2) * 320
    ops = graphs[1][0]
    producer = next(op for op in ops if op.layer == 2 and op.recipe == "kv_store")
    assert producer.mte2_bytes == 512 * 2
    assert producer.mte3_bytes == 544
    assert producer.vector_ops > 0
    compressed = next(op for op in ops if op.layer == 2 and op.recipe == "compressed_store")
    # 2048 -> 2049 does not complete a ratio-two row.
    assert compressed.mte3_bytes == 0
    bound = attention(ops)
    assert producer.op_id in bound.deps and compressed.op_id in bound.deps
    assert dict(bound.metadata)["attention_tensor_view"]["compressed_cache_row_bytes"] == 320
    # PD handoff reads Memory.cache_bytes(padded=False), the same live-state
    # boundary. Check its byte delta independently across all actual owners.
    old_model = replace(model.model, metadata=tuple(
        (k, v) for k, v in model.model.metadata if k != "cache_representations"))
    old_case = replace(model, model=old_model)
    expected_saved = sum(
        min(2049, p.window) * (1024 - 544)
        + ((2049 // p.compress_ratio) * (1024 - 320) if p.owns_long_kv else 0)
        for p in model.model.layer_plans)
    assert sum(Memory(old_case).cache_bytes(2049, padded=False)) - sum(
        Memory(model).cache_bytes(2049, padded=False)) == expected_saved
    cp = replace(model.runtime.parallel, cp=2)
    sharded = layer_cache_components(model.model, layer, cp, 2049, capacity_padded=False)
    assert sharded["window"] == 128 / 2 * 544
    assert sharded["compressed_kv"] == (2049 // 2) / 2 * 320


def test_missing_packed_cache_contract_is_visible_not_bf16_relabel(ordinary):
    model, _ = ordinary
    bad = replace(model, model=replace(model.model, metadata=tuple(
        (k, v) for k, v in model.model.metadata if k != "cache_representations")))
    with pytest.raises(ModelingError, match="cache_representation_contract_required"):
        target_graphs(bad, phase="decode", tokens=1, context=2048)


def test_tail_repeats_loads_but_pv_only_computes_initialized_rows():
    phases = dict(tile_phases("original", 1, True, True))
    assert phases["gather"].resources == {"hbm": 128 * (544 + 8), "mte2_aiv": 128 * 544}
    assert phases["qk"].matrix["bf16"] == 2 * 64 * 128 * 512
    assert phases["pv"].matrix["bf16"] == 2 * 64 * 16 * 512
    # Resident exchanges consume MTE3 and FIXPIPE but never HBM scratch.
    for phase in ("kv_l1", "scores", "probability_l1", "values", "accumulate"):
        assert phases[phase].resources.get("hbm", 0) == 0
    assert phases["output"].resources["hbm"] == 64 * 512 * 2


def test_batch_rows_preserve_context_and_metadata_qr_ownership():
    shape = {"sequences": ((2, 0), (1, 2048)), "window": 128, "ratio": 2, "topk": 512}
    rows = tuple(query_rows(shape))
    assert rows == ((1, 0), (2, 1), (128, 512))
    assert owner_ranges(8, 32) == tuple((i, i, i + 1) for i in range(8))
    assert owner_ranges(5, 2) == ((0, 0, 3), (1, 3, 5))
    stream = OwnerTiles.create(rows)
    assert [stream.key(i) for i in range(stream.count)] == [
        ("original", 1, True, True),
        ("original", 2, True, False), ("compressed", 1, False, True),
        ("original", 128, True, False),
        ("compressed", 128, False, False), ("compressed", 128, False, False),
        ("compressed", 128, False, False), ("compressed", 128, False, True)]


def test_b1_b8_actual_engine_one_kernel_owners_inputs_and_work_conserve(ordinary):
    model, graphs = ordinary
    results = []
    for batch in (1, 8):
        op = attention(graphs[batch][0])
        # Exactly one batch kernel for each layer, not B sequential invocations.
        assert len([o for o in graphs[batch][0] if o.layer == 2 and o.recipe == RECIPE]) == 1
        result = operator_costs(model, (op,), input_ready_ms=7)["operators"][0]
        assert result["service_before_input"] == {}
        assert result["live_tasks"] == 0
        assert not any(result["remaining_buffers"].values())
        expected_hbm = batch * (2 * 64 * 512 * 2 + 256 + 128 * (544 + 8) + 512 * (320 + 8))
        assert result["serviced"]["memory@r0:s0"] == expected_hbm
        assert result["serviced"]["matrix@r0:s0"] == pytest.approx(
            batch * 5 * 4 * 64 * 128 * 512 / model.machine.dies[0].peak("bf16") * 1000)
        assert len(result["peak_buffers_per_unit"]) == batch * 6
        assert result["peak_buffers_per_unit"]["ub@r0:s0:unit0"] == BUFFERS["ub"]
        results.append(result)
    assert results[1]["duration_ms"] < results[0]["duration_ms"] * 2


def test_source_capacity_does_not_resize_hardware(ordinary):
    model, graphs = ordinary
    op = attention(graphs[1][0])
    geometry = Geometry.from_devices(model.machine.dies)
    with pytest.raises(ModelingError, match="source_buffer_capacity_or_ownership:ub"):
        build(op, replace(geometry, buffers={**geometry.buffers, "ub": BUFFERS["ub"] - 1}))
    shape = {**dict(op.metadata)["mqsmla_geometry"], "sequences": ((1, 2048),) * 72}
    special = replace(op, metadata=tuple({**dict(op.metadata), "mqsmla_geometry": shape}.items()))
    with pytest.raises(ModelingError, match="selective_72_row_fd_branch"):
        build(special, geometry)


def test_graph_has_one_final_write_and_waits_for_all_external_inputs(ordinary):
    model, graphs = ordinary
    op = attention(graphs[1][0])
    external = {dep: (11 if i == 0 else 3) for i, dep in enumerate(op.deps)}
    result = graph_costs(model, (op,), external_inputs=external, isolated=False)
    assert result["duration_ms"] >= 11
    assert result["execution"]["live_tasks"] == 0
    isolated = operator_costs(model, (op,))["operators"][0]
    assert result["duration_ms"] == pytest.approx(11 + isolated["duration_ms"])


def test_real_source_work_responds_to_physical_resource_axes(ordinary):
    model, graphs = ordinary
    op = attention(graphs[1][0])
    baseline = operator_costs(model, (op,))["operators"][0]
    for axis in ("hbm", "mte1", "mte2_aiv", "mte3", "vector", "cube"):
        dies = []
        for die in model.machine.dies:
            rates = tuple((key, value * (0.01 if key == axis else 1)) for key, value in die.rates)
            peaks = tuple((key, value * (0.01 if axis == "cube" else 1))
                          for key, value in die.compute_peaks)
            dies.append(replace(die, rates=rates, compute_peaks=peaks))
        slow = replace(model, machine=replace(model.machine, dies=tuple(dies)))
        result = operator_costs(slow, (op,))["operators"][0]
        assert result["duration_ms"] > baseline["duration_ms"], axis
        assert result["operator"]["flops"] == baseline["operator"]["flops"]
        assert result["operator"]["hbm_bytes"] == baseline["operator"]["hbm_bytes"]


def test_one_owner_multiple_queries_drains_previous_output_before_state_reuse(ordinary):
    model, graphs = ordinary
    op = attention(graphs[1][0])
    shape = {**dict(op.metadata)["mqsmla_geometry"], "sequences": ((1, 2048),) * 33}
    op = replace(op, metadata=tuple({**dict(op.metadata), "mqsmla_geometry": shape}.items()))
    result = graph_costs(model, (op,), external_inputs={dep: 7 for dep in op.deps},
                         isolated=False, trace_limit=5000)
    owner = [event for event in result["events"] if ":group0:" in event["task"]]
    end = next(event["end_ms"] for event in owner if event["task"].endswith(":tile4:output"))
    for phase in ("query", "sinks", "accumulate"):
        current = next(event for event in owner if event["task"].endswith(":tile5:" + phase))
        assert current["start_ms"] >= end
    assert result["execution"]["live_tasks"] == 0
    assert all(event["start_ms"] >= 7 for event in owner)


def test_prefill_preserves_analytic_path_and_packed_read_descriptors(ordinary):
    model, _ = ordinary
    ops = target_graphs(model, phase="prefill", tokens=3, context=1, sequences=1)[0]
    op = next(o for o in ops if o.layer == 0 and o.family == "attention")
    assert op.recipe != RECIPE
    assert op.mte2_bytes == 4 * 544 + 3 * 64 * 512 * 2
    assert dict(op.metadata)["attention_tensor_view"]["kind"] == "mixed_quant_shared_kv"


def test_packed_cache_and_indexer_fork_at_pre_rope_compressor_latent(ordinary):
    model, _ = ordinary
    ops = target_graphs(model, phase="decode", tokens=2, context=2048, sequences=1)[0]
    layer = [op for op in ops if op.layer == 2]
    by_id = {op.op_id: op for op in layer}
    norm = next(op for op in layer if op.recipe == "compress_norm")
    rope = next(op for op in layer if op.recipe == "compress_rotary")
    index = next(op for op in layer if op.recipe == "index_k")
    store = next(op for op in layer if op.recipe == "compressed_store")
    assert store.deps == (rope.op_id,)
    assert store.mte2_bytes == 512 * 2 and store.mte3_bytes == 320
    # Activation quantization may be an explicit producer in between; trace
    # dependencies rather than require that it disappear for this assertion.
    seen, pending = set(), list(index.deps)
    while pending:
        parent = pending.pop()
        if parent in seen:
            continue
        seen.add(parent)
        if parent in by_id:
            pending.extend(by_id[parent].deps)
    assert norm.op_id in seen
    assert rope.op_id not in seen and store.op_id not in seen


def test_bf16_scfa_source_does_not_relabel_a_packed_cache(ordinary):
    model, _ = ordinary
    kernels = deepcopy(model.runtime.kernels)
    kernels["lowerings"]["sharedkv_attention"] = {
        "implementation": "vllm_ascend_sharedkv_scfa", "options": {
            "owner_schedule": "metadata_cost_balanced", "selected_rows": "valid_prefix",
            "sinks_dtype": "fp32", "metadata_s2_split": False, "return_softmax_lse": False}}
    model = replace(model, runtime=replace(model.runtime, kernels=kernels))
    with pytest.raises(ModelingError, match="packed_cache_representation_not_supported"):
        target_graphs(model, phase="prefill", tokens=3, context=2048, sequences=1)
