"""Grouped expert routing through graph, source and execution boundaries.

[Role] Test exact routes, probability moments and source-selected callers.
[Invariants] Expert rows, work, peer traffic and gating dependencies conserve.
[Timeouts & Locks] Bounded analytical graphs and retained fixture reads.
[Side-Effect Policy] No device access or writes. Engine state is test-local.
"""
from collections import Counter
from dataclasses import replace
import json
from itertools import product
from math import ceil
from pathlib import Path

import pytest

from _support.axes import ARCH_IDS, SKUS
from _support.hardware import sku_case
from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.compose import compose_case
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.direct_peer import DirectPeerCollective
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.moe_callers import LOCAL_EXPERT_REDUCE, local_expert_reduce_recipe
from mindie_sim.prediction.moe_cann_fusion import (
    HASH_NATIVE_KIND, INIT_ROUTING_NATIVE_KIND, SOFTMAX_NATIVE_KIND, build_selection, cann_native_active)
from mindie_sim.prediction.operators import grouped_moe, mc2
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.passes_ffn_fusion import raw_selection
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.routing_realizations import select_realization
from mindie_sim.prediction.source_force_eplb import SOURCE, ledger
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_structure

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
WIDTH, BANK_SIZE, TOP_K, EP = 32, 2, 2, 4
FP32_BYTES, BF16_BYTES = 4, 2
FRACTAL = 16
SOURCE_COUNTS = (6, 10, 8, 8)
GROUPED_SOURCE = {"lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}}
FULLMESH = {"kind": "arch35_fullmesh", "max_inflight_per_rank": 4}
TRANSPORT = {"kind": "separate_quantized_payload", "payload_dtype": "int8",
             "scale_dtype": "fp32", "scale_granularity": "row", "histogram_dtype": "int64"}

def test_group_list_forms_preserve_offsets_and_empty_expert_identity():
    rows, ends = (0, 5, 0, 2, 0), (0, 5, 5, 7, 7)
    expected = ((0, 0), (0, 5), (5, 0), (5, 2), (7, 0))
    assert grouped_moe.normalize_group_rows(iter(rows), expected_total=7) == expected
    assert grouped_moe.normalize_group_rows(ends, cumulative=True, expected_total=7) == expected
    assert grouped_moe.normalize_group_rows(()) == ()
    for rows, options, reason in (
        ((0, 5, 4), {"cumulative": True}, "not_monotone"), ((1, 2), {"expected_total": 4}, "group_total_mismatch"),
        ((1, -1), {}, "group_rows"), ((1, .5), {}, "group_rows"),
    ):
        with pytest.raises(ModelingError, match=reason):
            grouped_moe.normalize_group_rows(rows, **options)

@pytest.mark.parametrize("arch_id", ARCH_IDS)
@pytest.mark.parametrize("split", [False, True])
def test_blaze_virtual_core_carries_across_empty_experts_and_conserves_tails(arch_id, split):
    cores, tile, rows = 4, 2 * FRACTAL, (0, FRACTAL, 0, 4 * FRACTAL, 0, 0)
    dtype = "fp32" if arch_id == ARCH_IDS[0].values[0] else "bf16"
    k0 = FRACTAL // 2 if dtype == "fp32" else FRACTAL
    result = grouped_moe.schedule_blaze_groups(
        rows, tile, FRACTAL + 1, aic_cores=cores, base_m=tile, base_n=tile,
        compute_dtype=dtype, start_core=3, tail_split=split, m_tail_cnt=2, n_tail_cnt=2)
    tasks = (0, 4, 0, 8, 0, 0) if split else (0, 1, 0, 2, 0, 0)
    cursor = 3
    owners = Counter()
    for expert, (group, count) in enumerate(zip(result.groups, tasks)):
        assert (group.expert, group.offset, group.rows) == (expert, sum(rows[:expert]), rows[expert])
        assert (group.start_core, group.end_block, group.total_tiles) == (cursor, cursor + count, count)
        owners.update(index % cores for index in range(cursor, cursor + count))
        cursor = (cursor + count) % cores
    assert result.owner_tiles == tuple(owners[i] for i in range(cores))
    assert (result.max_waves, result.wave_tail_imbalance) == (max(owners.values()), 0 if split else 1)
    assert (sum(result.owner_output_elements), result.useful_flops) == (sum(rows) * tile, 2 * sum(rows) * tile * (FRACTAL + 1))
    # Half-height edge tasks consume a full M fractal after splitting.
    issues = (sum(tasks) if split else 10) * ceil((FRACTAL + 1) / k0)
    assert (result.issued_fractals, result.issued_flops) == (issues, 2 * FRACTAL**2 * k0 * issues)

@pytest.mark.parametrize("hardware", SKUS)
def test_governed_entrypoint_resolves_histogram_work_and_source_descriptor(hardware):
    geometry = Geometry.from_devices(sku_case(hardware).machine.dies[:1])
    rows, n, k = (3, 0, 33, 0, 1), 32, 17
    m = sum(rows)
    op = OpSpec("grouped", "grouped_gemm", "prefill", 0, 0, (), (m, n, k),
                (m, n, k), (m, n, k), (m, n, k), dtype="bf16", recipe="grouped_gemm",
                metadata=(("group_list", (3, 3, 36, 36, 37)),))
    context = OperatorPlanningContext(OpOccurrence(0, 0, 0, op.op_id), geometry, geometry.datapath)
    selected = grouped_moe.resolve_operator_plan(op, context)
    standalone = grouped_moe.build(op, geometry)
    assert standalone.work == selected.recipe.work and standalone.pipeline_spec.phases == selected.descriptor.phases
    assert selected.descriptor.owner == context.occurrence and selected.sources == selected.descriptor.sources
    repository = "cann/ops-transformer" if hardware == SKUS[-1].values[0] else "cann/catlass"
    assert (selected.implementation, selected.sources[0].repository) == ("aclnnGroupedMatmul", repository)
    expected = sum(ceil(row / FRACTAL) for row in rows) * ceil(n / FRACTAL) * ceil(k / FRACTAL)
    assert (standalone.metadata["useful_flops"], standalone.work.matrix) == (2 * m * n * k, {"bf16": expected * 2 * FRACTAL**3})
    with pytest.raises(ModelingError, match="unsupported_compute_architecture"):
        grouped_moe.build(op, Geometry(geometry.compute_units))

def _case(*, tp=1, dp=1, ep=1, tokens=4, output=1, experts=4, topk=TOP_K, width=WIDTH):
    ranks = dp * tp
    return {
        "machine": {"name": "route-control", "num_devices": ranks, "device": {
            "memory_bytes": 1024**3, "compute_units": 4, "compute_peaks": dict.fromkeys(("bf16", "fp32", "int8"), 1e9),
            "rates": dict.fromkeys(("memory", "vector"), 1e9)}, "links": [
            {"name": "fabric", "members": list(range(ranks)), "capacity_bytes_per_ms": 1000, "semantics": "shared"}]},
        "model": dict(name="geometry", layers=1, hidden_size=width, num_heads=4, num_kv_heads=2, head_dim=8,
                      vocab_size=128, intermediate_size=64, ffn="moe", num_experts=experts, top_k=topk,
                      moe_intermediate_size=32),
        "workload": {"requests": [dict(id="r", prompt_tokens=tokens, output_tokens=output)]},
        "runtime": dict(tp=tp, dp=dp, ep=ep, max_num_batched_tokens=max(16, tokens)),
    }

def _ops(raw, tokens=4, phase="prefill", **options):
    return target_operators(raw, phase=phase, tokens=tokens, scope="body", **options)

def _by_name(ops, layer=0):
    return {op.op_id.split(".", 2)[2]: op for op in ops if op.op_id.startswith(f"s0.l{layer}.")}

def _experts(ops):
    return [op for op in ops if op.family == "grouped_gemm"]

def _closed(report):
    execution = report["execution"]
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert not any(execution.get("live_allocated", {}).values())

@pytest.mark.parametrize("tokens,tp,counts", [(32, 4, (8,) * 8), (1, 1, (1, 1) + (0,) * 6),
                                             (5, 1, (2, 2) + (1,) * 6)])
def test_balanced_banks_keep_sparse_remainders_and_tensor_shards(tokens, tp, counts):
    raw = _case(tp=tp, experts=8, tokens=tokens)
    raw["model"]["moe_intermediate_size"] = 64
    raw["runtime"]["routing"] = {"policy": "balanced"}
    ops = _experts(_ops(raw, tokens))
    total = 2 * sum(counts) * WIDTH * 3 * 64
    assert len(ops) == 2 and sum(op.flops for op in ops) * tp == total
    active = sum(count > 0 for count in counts)
    for op in ops:
        meta, (m, n, k) = dict(op.metadata), op.local_shape
        assert (m, meta["matrix_groups"]) == (sum(counts), active)
        assert meta["routing_realization"]["global_expert_rows"] == counts
        assert meta["routing_realization"]["token_identities"] == "not_declared"
        assert meta["packed_weight_bytes"] == active * n * k * BF16_BYTES
    if tp > 1:
        result = operator_costs(raw, ops)["operators"]
        assert all(row["program"] == "MappedMatmul" and not row["service_before_input"] for row in result)
        assert sum(value for row in result for key, value in row["serviced"].items()
                   if key.startswith("matrix@")) == pytest.approx(total / tp / 1e6)

def test_balanced_cross_replica_cohort_splits_source_rows_and_serving_wire_work():
    raw = _case(dp=4, tp=2, ep=4, experts=8)
    raw["runtime"].update(routing={"policy": "balanced"},
                          kernels={"families": {"alltoall": {"kind": "pairwise_collective"}}})
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="body")
    assert tuple(graphs) == (0, 1)
    assert all(op.local_shape[0] == 4 for ops in graphs.values() for op in _experts(ops))
    report = graph_costs(raw, graphs, external_inputs={"analysis_input": 0})
    wire = sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in report["isolated"]["operators"] if row["operator"]["family"] == "alltoall")
    assert wire == 2 * 4 * 3 * WIDTH * BF16_BYTES
    raw["runtime"]["dp"], raw["machine"] = 2, _case(dp=2, tp=2)["machine"]
    raw["workload"]["requests"].append({"id": "second", "prompt_tokens": 4, "output_tokens": 1})
    analytical = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    served = predict(raw)
    for result in (report, analytical, served):
        _closed(result)
    assert served["execution"]["serviced"]["link:fabric"] == analytical["execution"]["serviced"]["link:fabric"]

@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("rows", [(0, 16, 0, 16), (17, 0, 15, 0)])
def test_exact_banks_refill_only_active_weights(hardware, rows):
    n, k = 64, 32
    geometry = Geometry.from_devices(sku_case(hardware).machine.dies[:1])
    shape = (sum(rows), n, k)
    op = OpSpec("banked", "grouped_gemm", "prefill", 0, 0, (), shape,
                shape, shape, shape, dtype="bf16", recipe="grouped_gemm",
                metadata=(("group_rows", rows), ("base_m", 16), ("base_n", n)))
    plan = grouped_moe.build(op, geometry)
    refills = sum(ceil(count / 16) for count in rows)
    reads = BF16_BYTES * (sum(rows) * k + refills * k * n)
    writes = BF16_BYTES * sum(rows) * n
    assert plan.work.resources["mte2_aic"] == reads
    assert plan.work.resources["fixpipe"] == writes
    assert plan.work.resources["hbm"] == reads + writes
    assert plan.metadata["useful_flops"] == 2 * sum(rows) * n * k


def _scenario(tokens, phase="prefill", **route):
    return dict(scope="body", layer=0, phase=phase, tokens=tokens, **route)

def _histogram_case():
    raw = _case(output=3)
    raw["runtime"]["routing"] = {"realizations": [
        _scenario(4, expert_rows=[3, 0, 3, 2]), _scenario(1, "decode", expert_rows=[1, 0, 1, 0])]}
    return raw

def _rows(raw, tokens=4, phase="prefill"):
    return operator_costs(raw, _experts(_ops(raw, tokens, phase)))["operators"]

def test_integer_histograms_drive_source_tiles_and_round_trip_without_changing_semantic_work():
    raw = _histogram_case()
    analytical = _rows(raw)
    raw["runtime"]["kernels"] = GROUPED_SOURCE
    for plain, source in zip(analytical, _rows(raw)):
        meta = source["operator"]["metadata"]
        assert tuple(meta["raw_group_list_active"]) == ((0, 3), (2, 3), (3, 2))
        assert tuple(meta["group_ids"]) == (0, 1, 2, 3) and meta["raw_group_list_padding_entries"] == 1
        assert source["serviced"] == pytest.approx(plain["serviced"])
        assert source["duration_ms"] != pytest.approx(plain["duration_ms"])
    restored = parse_case(parse_case(raw).to_dict())
    assert _rows(restored, 1, "decode")[0]["operator"]["local_shape"][0] == 2
    assert predict(raw)["execution"]["work"] == predict(restored)["execution"]["work"]
    scenario = raw["runtime"]["routing"]["realizations"][0]
    scenario.update(tokens=256, expert_rows=[128, 0, 128, 256])
    aligned = _rows(raw, 256)
    scenario["expert_rows"] = [192, 0, 160, 160]
    for first, second in zip(aligned, _rows(raw, 256)):
        left, right = first["operator"], second["operator"]
        _, n, k = left["local_shape"]
        assert (right["flops"], right["hbm_bytes"]) == (left["flops"], left["hbm_bytes"] + 2 * n * k * BF16_BYTES)

@pytest.mark.parametrize("counts", [[5, 0, 3, 0], [3, 0, 2, 2], [3.5, 0, 2.5, 2]])
def test_integer_histogram_rejects_impossible_topk_mass(counts):
    raw = _histogram_case()
    raw["runtime"]["routing"]["realizations"][0]["expert_rows"] = counts
    with pytest.raises(ModelingError, match="conserve_integer_topk"):
        _rows(raw)

@pytest.mark.parametrize("runtime,reason", [
    ({"max_num_batched_tokens": 3}, "one_matching_scenario"),
    ({"graph_buckets": [4]}, "unpadded_local_tokens"),
    ({"moe_recipe": "mc2"}, "ep1_pcp1_unchunked_alltoall"),
    ({"tp": 2, "ep": 2}, "ep1_pcp1_unchunked_alltoall"),
])
def test_integer_histogram_requires_matching_unpadded_scenario(runtime, reason):
    raw = _histogram_case()
    raw["runtime"].update(runtime)
    raw["machine"] = _case(tp=runtime.get("tp", 1))["machine"]
    with pytest.raises(ModelingError, match=reason):
        predict(raw)

def test_source_histogram_requires_integer_unrestricted_unquantized_operands():
    raw = _histogram_case()
    raw["runtime"]["kernels"] = GROUPED_SOURCE
    parsed = parse_case(raw)
    layer = replace(parsed.model.layer_plans[0], router_groups=2, router_group_topk=1)
    restricted = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    with pytest.raises(ModelingError, match="unrestricted_expert_groups"):
        _rows(restricted)
    with pytest.raises(ModelingError, match="unquantized_bf16_nd"):
        _rows(replace(parsed, model=replace(parsed.model, quant=quant_preset("w8a8_int8"))))
    del raw["runtime"]["routing"]
    raw["model"]["expert_routing_histogram"] = [3, 0, 3, 2]
    with pytest.raises(ModelingError, match="explicit_integer_routing_realization"):
        _rows(raw)
    del raw["runtime"]["kernels"]
    assert 2 < dict(_experts(_ops(raw))[0].metadata)["expected_active_experts"] < 3

def _route_case(counts=SOURCE_COUNTS, *, cohort=True, dp=1, quantized=False, local=False):
    ep, tokens = len(counts), sum(counts)
    raw = _case(tp=ep if dp == 1 else 1, dp=dp, ep=ep, tokens=tokens, experts=ep * BANK_SIZE)
    patterns = tuple((i, (i + step) % (ep * BANK_SIZE))
                     for step in (1, BANK_SIZE) for i in range(ep * BANK_SIZE))
    owners = [rank for rank, count in enumerate(counts) for _ in range(count)]
    ids = ([(0, 1)] * tokens if local else [patterns[(index + rank) % len(patterns)]
                                           for rank, count in enumerate(counts) for index in range(count)])
    raw["runtime"]["routing"] = {"realizations": [_scenario(
        tokens, topk_ids=ids, source_owners=owners, cohort=cohort,
        expert_placement=[divmod(i, BANK_SIZE) for i in range(ep * BANK_SIZE)])]}
    if quantized:
        raw["model"]["quant"] = quant_preset("w8a8_int8").to_dict()
        raw["runtime"]["kernels"] = {"moe_transport": TRANSPORT.copy()}
    return raw

@pytest.mark.parametrize("quantized", [False, True])
def test_explicit_cohort_rebinds_banks_payload_and_transposed_peer_cells(quantized):
    raw = _route_case(quantized=quantized)
    route, = raw["runtime"]["routing"]["realizations"]
    peers = Counter((origin, expert // BANK_SIZE) for origin, ids in zip(route["source_owners"], route["topk_ids"])
                    for expert in ids)
    counts = Counter(expert for ids in route["topk_ids"] for expert in ids)
    matrix_rows = tuple(tuple(peers[src, dst] for dst in range(EP)) for src in range(EP))
    local = SOURCE_COUNTS[0]
    parsed = parse_case(raw)
    for rank, tokens in enumerate(SOURCE_COUNTS):
        realized = select_realization(parsed.runtime, parsed.model.layer_plans[0], phase="prefill",
                                      tokens=tokens, valid_tokens=tokens, physical_rank=rank)
        assert realized.local_expert_rows == tuple(counts[rank * BANK_SIZE + i] for i in range(BANK_SIZE))
        assert (realized.valid_local_tokens, realized.peer_assignments) == (tokens, matrix_rows)
    ops = _by_name(_ops(raw, local))
    assert ops["pack"].local_shape == (local, TOP_K)
    m = sum(counts[i] for i in range(BANK_SIZE))
    for op in _experts(ops.values()):
        _, n, k = op.local_shape
        assert (op.local_shape[0], op.flops) == (m, 2 * m * n * k)
    forward = {"moe_scale_dispatch": FP32_BYTES, "moe_payload_dispatch": WIDTH} if quantized else {"dispatch": 2 * WIDTH}
    combine = "expert_combine" if quantized else "combine"
    for name, width in (forward | {combine: WIDTH * BF16_BYTES}).items():
        expected = tuple(tuple(width * peers[(d, s) if name == combine else (s, d)] if s != d else 0
                               for d in range(EP)) for s in range(EP))
        assert dict(ops[name].metadata)["peer_traffic"]["payload_bytes"] == expected
        assert ops[name].link_bytes == sum(map(sum, expected)) / EP
    if quantized:
        for selection, issued in ((ops, local * TOP_K),
                                  (_by_name(_ops(_route_case((2, 2, 2, 26), quantized=True), local)), 2 * TOP_K)):
            quant = selection["moe_transport_quantize"]
            assert quant.deps == (selection["pack"].op_id,)
            assert (quant.mte2_bytes, quant.mte3_bytes) == (issued * WIDTH * BF16_BYTES, issued * (WIDTH + FP32_BYTES))
            assert dict(selection["moe_histogram"].metadata)["source_assignments"] == issued
        assert set(ops["expert_up"].deps) == {ops[key].op_id for key in (*forward, "moe_transport_quantize")}
    else:
        changed = _route_case(local=True)
        assert _by_name(_ops(changed, local))["expert_up"].local_shape[0] == sum(SOURCE_COUNTS) * TOP_K
        first, second = predict(raw), predict(changed)
        assert first["requests"][0]["ttft_ms"] != second["requests"][0]["ttft_ms"]
        assert first["execution"]["work"] != second["execution"]["work"]
        native_raw = raw | {"runtime": raw["runtime"] | {"kernels": {"native_moe_protocol": FULLMESH}}}
        native = _by_name(_ops(native_raw, local))
        for name, reverse in (("dispatch", False), ("unpack", True)):
            phase = dict(native[name].metadata)["mc2_phase"]
            assert (phase["route_matrix"], phase["route_reverse"]) == (matrix_rows, reverse)
    graphs = target_graphs(raw, phase="prefill", tokens=local, scope="body")
    report = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, compact=True)
    assert report["execution"]["work"]["flops"] == pytest.approx(sum(op.flops for op in graphs[0]))
    _closed(report)

def test_local_reduce_keeps_empty_physical_banks_and_rejects_noncontiguous_placement():
    raw = _route_case((4, 0), cohort=False, local=True)
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    scenario, = raw["runtime"]["routing"]["realizations"]
    parsed = parse_case(raw)
    options = dict(plan=parsed.model.layer_plans[0], model=parsed.model, parallel=parsed.runtime.parallel,
                   deps="input", stage=0, requests=("r",), tokens=4, mode="prefill", global_tokens=None, chunks=4)
    for placement in ([0, 0, 1, 1], [0, [0, 1], 1, [1, 1]], [(0, 0), (0, 1), (1, 0), (1, 1)],
                      [1, [0, 1], 0, [1, 1]]):
        scenario["expert_placement"] = placement
        runtime = parse_case(raw).runtime
        for rank in range(2):
            route = select_realization(runtime, options["plan"], phase="prefill", tokens=4,
                                       valid_tokens=4, physical_rank=rank)
            args = dict(options, prefix="rank", runtime=runtime, routing_realization=route)
            if placement[0] == 1:
                with pytest.raises(ModelingError, match="contiguous_physical_bank"):
                    local_expert_reduce_recipe(**args)
                continue
            ops, last = local_expert_reduce_recipe(**args)
            assert route.expert_placement == tuple(divmod(i, 2) for i in range(4))
            assert {"rank.pack", "rank.router", "rank.unpack", last} <= {op.op_id for op in ops}
            assert [op.local_shape[0] for op in _experts(ops)] == ([8, 8] if rank == 0 else [])

def _cursor_case(selected=True):
    folder = ROOT / "examples/cases/high-concurrency-20261030/kimi-hccl-source-conditional"
    raw = json.loads((folder / "a5_kimi_tp8_e16_bf16_c32_p4k_hccl_observed.json").read_text())
    raw["model"]["layers"] = 2
    bindings = raw["runtime"]["kernels"]
    bindings.pop("service_regimes", None)
    if selected:
        bindings["force_eplb_source"] = {"kind": "vllm_ascend_sp_mc2", "global_bs": 0}
    return raw

def test_eplb_cursor_matches_wrapped_token_oracle_with_idle_sources():
    for ep, multiple, allocated in product((2, 4, 8), (1, 2, 3), (1, 2, 5)):
        experts = multiple * ep
        for topk, valid in product({1, ep, experts - 1}, {1, allocated, allocated * ep // 2, allocated * ep - 1, allocated * ep}):
            banks, edges = Counter(), Counter()
            for index, slot in product(range(valid), range(topk)):
                source, token = divmod(index, allocated)
                bank, dest = divmod((source * (allocated * topk + 1) + token * topk + slot) % experts, ep)
                banks[dest, bank] += 1
                edges[source, dest] += 1
            route = ledger(experts, ep, topk, allocated, valid)
            assert route.rank_expert_rows == tuple(tuple(banks[r, b] for b in range(multiple)) for r in range(ep))
            assert route.peer_assignments == tuple(tuple(edges[s, d] for d in range(ep)) for s in range(ep))
            assert sum(route.expert_rows) == valid * topk

@pytest.mark.parametrize("batch", [1, 4, 8, 32])
@pytest.mark.parametrize("selected", [True, False])
def test_eplb_keeps_router_capacity_and_conserves_directed_transport(batch, selected):
    raw = _cursor_case(selected)
    options = dict(phase="decode", tokens=1, context=4104, sequences=batch, scope="body")
    ops, baseline = (_by_name(target_graphs(case, **options)[0], layer=1) for case in (raw, _cursor_case(False)))
    for name in ("router", "router_weight_gather"):
        assert (ops[name].local_shape, ops[name].flops, ops[name].vector_ops) == (
            baseline[name].local_shape, baseline[name].flops, baseline[name].vector_ops)
    ep, allocated = 8, ceil(batch / 8)
    route = ledger(16, ep, 8, allocated, batch)
    matrix = route.peer_assignments
    phases = [ops[key] for key in ("dispatch", "unpack")]
    native = [dict(op.metadata)["mc2_phase"] for op in phases]
    if selected:
        layout = dict(ops["expert_up"].metadata)["routing_realization"]
        assert (layout["source"], layout["valid_global_tokens"], layout["allocated_global_tokens"]) == (SOURCE, batch, ep * allocated)
        for op in _experts(ops.values()):
            m, n, k = op.local_shape
            expected = route.rank_expert_rows[route.representative_rank]
            assert dict(op.metadata)["matrix_group_rows"] == expected
            assert m == sum(expected) and op.flops == 2 * m * n * k
        typed = route.typed_occurrence(payload_scale_producer={"dispatch_op": phases[0].op_id})
        assert (len(typed["token_order"]), typed["source_destination_rows"]) == (batch, matrix)
        assert [(phase["route_matrix"], phase["route_reverse"]) for phase in native] == [(matrix, False), (matrix, True)]
    else:
        assert all("route_matrix" not in phase for phase in native)
    protocol = tuple(replace(op, deps=()) for op in phases)
    report = graph_costs(raw, {0: protocol}, isolated=False, compact=True, input_ready_ms=7)
    bridge = report["profile"]["native_bridge"]
    status = 16 * WIDTH // ep
    d_meta, c_meta = native[0], native[1]
    dispatch = mc2.dispatch_row_layout(d_meta["width"], input_dtype=d_meta.get("input_dtype", "bf16"), payload_dtype=d_meta.get("payload_dtype", d_meta.get("input_dtype", "bf16")), quant_mode=d_meta.get("quant_mode", "none")).packet_bytes
    combine = mc2.combine_row_layout(c_meta["width"], dtype=c_meta.get("input_dtype", "bf16")).packet_bytes
    offdiag = [(s, d) for s in range(ep) for d in range(ep) if s != d]
    serviced = {k: v for k, v in report["execution"]["serviced"].items() if k.startswith("link:")}
    if selected:
        assert serviced == pytest.approx({
            f"link:d2d_intra_node:0:pair{s}>{d}": matrix[s][d] * dispatch + matrix[d][s] * combine + status
            for s, d in offdiag})
        expected = sum(matrix[s][d] for s, d in offdiag)
        assert bridge["transport_frames"] == 2 * expected
        payload = expected * (dispatch + combine)
    else:
        assert bridge["transport_frames"] == 0
        payload = (ep - 1) * sum(phase["uniform_payload_bytes"] for phase in native)
    total = payload + len(offdiag) * status
    assert (bridge["transport_tx_bytes"], bridge["transport_rx_bytes"], sum(serviced.values())) == pytest.approx((total,) * 3)
    assert bridge["count_edges"] == len(offdiag)
    _closed(report)

def _compose(model, **options):
    return parse_case(compose_case(model=model, hardware=SKUS[-1].id, tp=1, layers=4, **options))

@pytest.mark.parametrize("model,mode,members", [
    ("deepseek-v4-flash-0731", "hash", 1), ("deepseek-v4-flash-0731", "sqrtsoftplus", 2),
    ("deepseek-v3", "noaux_tc", 9),
])
def test_gating_conserves_score_work_and_preserves_external_hash_normalization(model, mode, members):
    case = _compose(model, concurrency=1)
    plan = next(p for p in case.model.layer_plans if p.router_mode == mode or (mode == "noaux_tc" and p.router_noaux_tc))
    options = dict(prefix="s0.l0", plan=plan, logits_input="logits", weight_dtype="fp32",
                   weight_bytes=FP32_BYTES, stage=0, requests=(), tokens=8, mode="decode")
    raw, *_ = raw_selection(**options)
    selected = build_selection(**options, runtime=case.runtime)
    fused = next(op for op in selected.ops if op.recipe == "moe_gating_topk")
    meta = dict(fused.metadata)
    kind = HASH_NATIVE_KIND if mode == "hash" else SOFTMAX_NATIVE_KIND
    assert (meta["native_kind"], meta["fused_member_count"]) == (kind, members)
    if mode == "hash":
        assert not meta["renorm"] and fused.vector_ops == raw[0].vector_ops
        assert len(selected.ops) == 1 or selected.ops[-1].deps == (fused.op_id,)
    else:
        bias = plan.num_experts * FP32_BYTES if mode == "noaux_tc" else 0
        assert fused.vector_ops == pytest.approx(sum(op.vector_ops for op in raw))
        assert fused.mte2_bytes == 8 * plan.num_experts * FP32_BYTES + bias
        assert fused.mte3_bytes == 8 * plan.top_k * (FP32_BYTES + 4)
        assert meta["renorm"] and not any(op.recipe == "router_routed_scale" for op in selected.ops)

def test_gating_respects_backend_raw_eager_and_explicit_recipe_guards():
    runtime = _compose("deepseek-v3", concurrency=1).runtime
    assert cann_native_active(runtime)
    assert not cann_native_active(_compose("deepseek-v3", concurrency=1, raw_eager=True).runtime)
    assert not cann_native_active(replace(runtime, kernels=runtime.kernels | {"recipes": {"router_topk": "declared"}}))
    metadata = dict(runtime.metadata)
    metadata["execution_policy"] = dict(metadata["execution_policy"], backend="nvidia")
    assert not cann_native_active(replace(runtime, metadata=tuple(metadata.items())))

def test_router_precision_preserves_source_weight_bytes_and_declared_cast():
    case = _compose("glm-5.3-flash", dp=1, ep=1, concurrency=1)
    plan = next(p for p in case.model.layer_plans if p.ffn == "moe")
    override = replace(plan, router_dtype="fp32", router_output_dtype="", router_precast_weight_copy=True)
    logical = plan.num_experts * plan.router_input_size
    for dtype, active in (("bf16", plan), ("fp32", override)):
        layers = tuple(active if p.index == plan.index else p for p in case.model.layer_plans)
        parsed = replace(case, model=replace(case.model, layer_plans=layers))
        ops = _by_name(_ops(parsed, 4, "decode", context=128), layer=plan.index)
        meta = dict(ops["router"].metadata)
        assert (meta["weight_format"], meta["activation_format"], meta["output_format"]) == (dtype, dtype, "fp32")
        assert meta["packed_weight_bytes"] == logical * (FP32_BYTES if dtype == "fp32" else BF16_BYTES)
        terms = [t for t in layer_weight_terms(active, case.model) if t.name in {"router", "router_precast"}]
        source, resident = (sum(term_bytes(t, case.model.quant, case.runtime.parallel, source=flag) for t in terms)
                            for flag in (True, False))
        assert source == logical * BF16_BYTES and resident == source * (3 if dtype == "fp32" else 1)
        casts = [op for op in ops.values() if op.recipe == "router_input_cast"]
        if dtype == "fp32":
            cast, = casts
            assert ops["router"].deps == (cast.op_id,) and cast.mte3_bytes == 4 * plan.router_input_size * FP32_BYTES
        else:
            assert not casts

@pytest.mark.parametrize("mc2", [False, True])
def test_ep_dispatch_owns_absorbed_quantization_and_publishes_scales_once(mc2):
    options = {"moe_comm_mode": "dynamic"} if mc2 else {"exclude_optimizations": ("cann_megamoe_fusion",)}
    case = _compose("glm-5.3-flash", quant="w8a8_mxfp8", dp=8, ep=8, concurrency=16, **options)
    ops = _ops(case, 8, "decode", context=128)
    assert not any(op.op_id.endswith(".pack.quantize") for op in ops)
    producer = next(op for op in ops if "moe_init_routing" in dict(op.metadata))
    meta = dict(producer.metadata)
    config = meta["moe_init_routing"]
    assert (config["native_kind"], config["quant_mode"]) == (INIT_ROUTING_NATIVE_KIND, 1)
    assert dict(_experts(ops)[0].metadata)["activation_scale_producer"] == producer.op_id
    if mc2:
        assert "send_read_quantize" in meta["mc2_phases"] and meta["mc2_phase"]["dispatch_quant"] == "mx"
    else:
        dispatch = next(op for op in ops if op.recipe == "expert_dispatch")
        assert config["scale_format"] == "fp8_e8m0" and config["scale_bytes"] > 0
        assert config["inverse_index_bytes"] == producer.local_shape[0] * producer.local_shape[1] * 4
        assert producer.mte3_bytes == config["write_bytes"] and producer.vector_ops >= config["vector_ops"] > 0
        assert dict(_experts(ops)[0].metadata)["activation_payload_producer"] == dispatch.op_id

def test_delayed_peer_source_keeps_exact_wire_bytes_and_releases_receive_window():
    raw = _route_case(dp=4, quantized=True)
    graphs = target_graphs(raw, phase="prefill", tokens=1, sequences=SOURCE_COUNTS, scope="body")
    book = RecipeBook.from_mapping(raw["runtime"]["kernels"])
    capacity = 1024
    for name in ("expert_quantized_dispatch", "expert_scale_dispatch", "expert_combine"):
        book.recipes[name] = DirectPeerCollective(max_inflight_per_rank=1, chunk_bytes=capacity // 2,
                                                   receive_capacity_bytes_per_peer=capacity, receive_buffer_id="rx")
    baseline = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, recipes=book, compact=True)
    delayed = TargetGraphSet({rank: tuple(
        replace(op, deps=(*op.deps, "late")) if rank == 0 and op.op_id.endswith(".pack") else op
        for op in ops) for rank, ops in graphs.items()}, kernels=graphs.effective_kernels,
        selection=graphs.default_selection)
    late = graph_costs(raw, delayed, external_inputs={"analysis_input": 0, "late": 50}, recipes=book, compact=True)
    assert late["duration_ms"] > baseline["duration_ms"]
    for report in (baseline, late):
        assert report["execution"]["peak_allocated"]["transport:rx:rx"] == capacity
        _closed(report)
    assert late["execution"]["serviced"]["link:fabric"] == pytest.approx(baseline["execution"]["serviced"]["link:fabric"])

@pytest.mark.parametrize("catalog,capture,kind,tokens", [
    ("glm53-a3-w4a8", "a3_glm53_prefill", "MoeGatingTopK", 4096),
    ("qwen35-a3-tp4", "a3_qwen35", "MoeGatingTopK", 8192),
    ("v4f-a3-w8a8-tp8", "a3_v4f", "MoeGatingTopKHash", 512),
])
def test_retained_router_contracts_preserve_distinct_repeated_layer_candidates(catalog, capture, kind, tokens):
    import mindie_profiler
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.structure_mapping import map_structure
    from mindie_profiler.tensor_contracts import observed_contract
    raw = json.loads((ROOT / "examples/cases" / f"{catalog}.json").read_text())
    graph = export_structure(raw, phase="prefill", tokens=tokens, scope="body")
    fixture = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts" / f"{capture}.csv"
    rows = [row for row in kernel_rows(fixture, rank_id="rank_0", rank_index=0)
            if row.kind == kind and row.input_shapes[0][0] == tokens]
    groups = [group for group in graph["subgraphs"] if group["role"] == "router_selection" and group["contract"][1] == kind]
    contracts = [json.loads(json.dumps(observed_contract(row))) for row in rows]
    assert len(groups) > 1 and all(group["contract"] in contracts for group in groups)
    events, stat = [row.event for row in rows], fixture.stat()
    source = SourceRef("rank_0", 0, "csv", str(fixture), fixture.name, None, stat.st_size,
                       stat.st_mtime_ns, "retained_capture", "kernel_details")
    window = Window(min(event.start_ns for event in events), max(event.end_ns for event in events))
    mapped = map_structure([Observation("rank_0", 0, window, None, events, source, QueryStats("rank_0", "routing"))], graph)
    matched = sum(any(item[:3] == group["contract"][:3] for group in groups) for item in contracts)
    assert (mapped["status_counts"].get("ambiguous", 0), mapped["status_counts"].get("unmapped", 0)) == (matched, len(rows) - matched)
    assert mapped["windows"][0]["bound_busy_union_ns"] == 0
