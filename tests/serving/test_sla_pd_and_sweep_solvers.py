"""SLA sizing, efficiency rooflines, PD handoff and topology sweep solvers.

[Role] Verify the public efficiency/SLA solvers, the disaggregated prefill-decode
  physical connector and handoff oracles, serving I/O and step summaries, and
  the static pre-graph topology and P/D ratio sweeps.
[Invariants] Every assertion is an independent analytical byte/FLOP/wire/
  duration formula; zero-latency and zero-byte declarations stay exact, and
  every invalid candidate fails through its own declared error code.
[Timeouts & Locks] In-memory tier0 solvers and bounded engines; no device
  context, network, sleeps, or filesystem writes outside ``tmp_path``.
[Side-Effect Policy] Read-only over fresh per-test mappings; checkpoints live
  only under pytest ``tmp_path``.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time

import pytest

from _support.axes import DT_SKUS, PER_PORT_SKUS
from _support.serving_cases import bind_pd_pool, case, linear_work, pd_case, pd_model_case
from mindie_sim import graph_costs, predict, predict_pd, sweep as sweep_module, target_graphs
from mindie_sim.sweep import engine as sweep_engine
from mindie_sim.__main__ import parser
from mindie_sim.compose import compose_case
from mindie_sim.prediction import efficiency as efficiency_module
from mindie_sim.prediction.body_defaults import TargetGraphSet, consume_target_declaration
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.efficiency import (_vram_max_sequences_per_dp, analyze_network_efficiency, analyze_operator_row, extract_hardware_peaks, fit_case_roofline_regimes, scale_case_hardware)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.sweep import (compute_pareto_frontier, enumerate_parallel_candidates, reconstruct_sweep_summary, solve_hardware_sla_throughput, solve_sla_sizing, sweep_pd_disaggregation_ratios, sweep_topologies, sweep_topologies_iter)
from mindie_sim.sweep.engine import SCREEN_BATCH, b1_screen, build_candidate_case, validate_candidate_statically
from mindie_sim.sweep.resize import materialize_machine_for_devices

pytestmark = pytest.mark.tier0

TOL, EXACT = 1e-9, 1e-12
SWEEP = {"sla_ms": 30.0, "context": 8, "output_budget": 2, "max_sequences_per_dp": 1}
_IO = {"host_prepare_ms": 0.5, "host_result_ms": 0.5, "h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "latency_us": 250}, "d2h": {"bytes_per_s": 4000, "bytes_per_token": 4, "latency_us": 250}, "client_delivery_ms": 2}
_MOE_MODEL = {"name": "tier0-mla-moe", "layers": 4, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 1, "head_dim": 128, "intermediate_size": 2048, "vocab_size": 64000, "attention": "mla", "ffn": "moe", "activation_dtype": "bf16", "weight_dtype": "int8", "kv_dtype": "bf16", "compute_dtype": "bf16", "quantization": "w8a8", "num_experts": 16, "top_k": 2, "moe_intermediate_size": 1024, "num_shared_experts": 1, "mla": {"q_lora_rank": 1024, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128}}
_DENSE_MODEL = {"name": "tier0-dense-gqa", "layers": 2, "hidden_size": 4096, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "intermediate_size": 11008, "vocab_size": 64000, "attention": "gqa", "ffn": "dense", "activation_dtype": "bf16", "weight_dtype": "bf16", "kv_dtype": "bf16", "compute_dtype": "bf16"}
_RUNTIME = {"pp": 1, "max_model_len": 131072, "max_num_seqs": 128, "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.90, "kv_block_size": 128}
_DEVICE = {"rates": {"native_call_service": 1e9}, "resource_units": {"native_call_service": 1}, "work_units": {"native_call_service": "busy_ns"}}
_OPERATOR = {"op_id": "l0.op", "family": "custom", "recipe": "recipe", "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16", "flops": 0.0, "vector_ops": 0.0, "hbm_bytes": 0.0, "link_bytes": 0.0}
_REQUEST = {"output_tokens": 16, "arrival_ms": 0.0}
def same(actual, expected, rel=TOL):
    return math.isclose(actual, expected, rel_tol=rel)
def _declared_costs(case_spec, graphs):
    """Independent ``graph_costs`` run over the case's own declared target graphs."""
    return graph_costs(consume_target_declaration(case_spec, graphs),
                       TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels,
                                      selection=graphs.default_selection),
                       external_inputs={"analysis_input": 0}, include_submission=False, isolated=False)
def _decode_costs(case_spec, sequences, context=4096):
    return _declared_costs(case_spec, target_graphs(case_spec, phase="decode", tokens=1,
                                                    context=context, sequences=sequences, scope="body"))
def _machine(ranks, device=False):
    machine = {"backend": "ascend", "sku": "Ascend910_9362", "num_devices": ranks, "ranks_per_node": ranks}
    return {**machine, "device": deepcopy(_DEVICE)} if device else machine
def _moe_case(*, layers=4, dp=1, tp=4, ep=4, prompt=4096, device=True):
    return {"machine": _machine(dp * tp, device), "model": {**_MOE_MODEL, "layers": layers}, "runtime": {**_RUNTIME, "tp": tp, "dp": dp, "ep": ep}, "workload": {"requests": [{"request_id": "req0", "prompt_tokens": prompt, **_REQUEST}]}}
def _dense_case(*, layers=2, dp=1, tp=4, prompt=2048, **runtime):
    return {"machine": _machine(dp * tp), "model": {**_DENSE_MODEL, "layers": layers}, "runtime": {**_RUNTIME, "tp": tp, "dp": dp, "ep": 1, **runtime}, "workload": {"requests": [{"request_id": "req0", "prompt_tokens": prompt, **_REQUEST}]}}
def _op_row(physical_id, occurrences, program, duration_ms, **operator):
    return {"physical_id": physical_id, "occurrences": occurrences, "program": program,
            "duration_ms": duration_ms, "operator": {**_OPERATOR, **operator}}

# --------------------------------------------------------------------------- efficiency / SLA
def test_operator_four_dimensional_classification_pmu_and_executed_service():
    parsed = parse_case(_moe_case()); peaks = extract_hardware_peaks(parsed)
    bf16, int8 = peaks["compute_peaks"]["bf16"], peaks["compute_peaks"]["int8"]
    vec, hbm, link = peaks["vector_peak_ops_per_s"], peaks["hbm_peak_bytes_per_s"], peaks["link_peak_bytes_per_s"]
    cube = analyze_operator_row(_op_row("op_cube", 4, "SteadyMatmul", 0.20, op_id="l0.q_proj", family="gemm", recipe="matmul", dtype="int8", flops=0.10 * 1e-3 * int8 + 0.04 * 1e-3 * bf16, matrix_flops={"int8": 0.10 * 1e-3 * int8, "bf16": 0.04 * 1e-3 * bf16}, hbm_bytes=0.05 * 1e-3 * hbm), parsed, peaks=peaks)
    assert (cube["operator_kind"], cube["dominant_bound"], cube["vec_mfu"]) == ("cube", "cube_bound", None)
    assert same(cube["t_cube_ideal_ms"], 0.14) and same(cube["cube_mfu"], 0.14 / 0.20) and same(cube["mbu"], 0.05 / 0.20)
    cv_row = _op_row("op_cv", 8, "StagedGqaDecode", 0.25, op_id="l0.flash_attn", family="attention", recipe="flash_attention_decode", flops=0.08 * 1e-3 * bf16, vector_ops=0.12 * 1e-3 * vec, hbm_bytes=0.18 * 1e-3 * hbm)
    no_pmu = analyze_operator_row(cv_row, parsed, peaks=peaks)
    assert (no_pmu["operator_kind"], no_pmu["cv_overlap_ratio"]) == ("cv_fused", None)
    proven = analyze_operator_row({**cv_row, "duration_ms": 0.16}, parsed, peaks=peaks)
    assert same(proven["cv_overlap_ratio"], (0.20 - 0.16) / 0.08)
    partial = analyze_operator_row(cv_row, parsed, peaks=peaks, pmu={"aic_mac_time_us": 100.0, "aiv_vec_time_us": 80.0, "aic_wait_time_us": 20.0, "aiv_wait_time_us": 40.0})
    assert same(partial["cv_overlap_ratio"], 0.75)
    for span, ratio in ((100.0, 1.0), (180.0, 0.0)):
        measured = analyze_operator_row(cv_row, parsed, peaks=peaks, pmu={"aic_mac_time_us": 100.0, "aiv_vec_time_us": 80.0, "compute_span_us": span})
        assert same(measured["cv_overlap_ratio"], ratio)
    serviced = _op_row("op_serviced", 2, "PagedAttention", 0.20, op_id="l0.paged_attn", family="attention", recipe="paged_attention", metadata={"resource_work": {"cube": 0.05 * 1e-3 * bf16}})
    serviced.update(serviced={"memory@r0:s0": 0.10 * 1e-3 * hbm, "vector@r0:s0": 0.04 * 1e-3 * vec, "mte2_aiv@r0:s0": 0.10 * 1e-3 * hbm}, communication={"wire_by_resource": {"intra_package:0@r0": 0.02 * 1e-3 * link}})
    executed = analyze_operator_row(serviced, parsed, peaks=peaks)
    assert executed["work_basis"] == "executed_service" and link == 196e9
    assert all((same(executed["t_hbm_ideal_ms"], 0.10), same(executed["t_vec_ideal_ms"], 0.04), same(executed["t_cube_ideal_ms"], 0.05), same(executed["t_link_ideal_ms"], 0.02)))
    reference = analyze_operator_row({**serviced, "joint_native_service": {"service": "native_call_service"}, "analytical_reference_work": {"hbm": 0.08 * 1e-3 * hbm}, "serviced": {"native_call_service@r0:s0": 200000.0}}, parsed, peaks=peaks)
    assert reference["work_basis"] == "analytical_reference" and same(reference["t_hbm_ideal_ms"], 0.08)
def test_heterogeneous_dies_and_undeclared_precision_error():
    raw = _moe_case(dp=1, tp=2, ep=2); machine = deepcopy(parse_case(raw).machine.to_dict())
    for device in machine["devices"]:
        device["compute_peaks"].pop("int4", None)
    slower, faster = machine["devices"][1], machine["devices"][0]
    slower["compute_peaks"]["bf16"] = faster["compute_peaks"]["bf16"] * 0.5
    slower["rates"]["hbm"] = faster["rates"]["hbm"] * 0.6; slower["rates"]["vector"] = faster["rates"]["vector"] * 0.7
    heterogeneous = parse_case({**raw, "machine": machine}); peaks = extract_hardware_peaks(heterogeneous, stage=0, replica=0)
    assert same(peaks["compute_peaks"]["bf16"], slower["compute_peaks"]["bf16"], EXACT)
    assert same(peaks["hbm_peak_bytes_per_s"], slower["rates"]["hbm"], EXACT) and same(peaks["vector_peak_ops_per_s"], slower["rates"]["vector"], EXACT)
    with pytest.raises(ModelingError):
        analyze_operator_row(_op_row("op_int4", 1, "SteadyMatmul", 0.10, op_id="l0.int4_gemm", family="gemm", recipe="matmul", dtype="int4", flops=1e9, hbm_bytes=1e6), heterogeneous, peaks=peaks)
def test_parallel_graph_roofline_efficiency_and_fit_signature_distinctness():
    parsed = parse_case(_moe_case()); peaks = extract_hardware_peaks(parsed)
    bf16, hbm = peaks["compute_peaks"]["bf16"], peaks["hbm_peak_bytes_per_s"]
    memory_branch = _op_row("branch_mem", 1, "ResourceEnvelope", 1.0, op_id="l0.branch_a", recipe="dup_recipe", local_shape=[1, 4096], hbm_bytes=0.8 * 1e-3 * hbm)
    cube_branch = _op_row("branch_cube", 1, "SteadyMatmul", 1.0, op_id="l0.branch_b", recipe="dup_recipe", local_shape=[16, 4096], flops=0.7 * 1e-3 * bf16)
    memory_branch["total_ms"] = cube_branch["total_ms"] = 1.0
    net = analyze_network_efficiency(parsed, phase="decode", tokens=1, context=4096, sequences=1, operator_report={"operators": [memory_branch, cube_branch]}, graph_report={"duration_ms": 1.0})
    assert net["network_roofline_efficiency"] <= 1.0 and same(net["network_roofline_efficiency"], 0.80)
    assert same(net["operator_serial_roofline_efficiency"], 1.5 / 2.0) and len(fit_case_roofline_regimes(parsed, baseline_analysis=net)["by_key"]) == 2
def test_hardware_scaling_roundtrip_and_independent_graph_costs():
    raw = _moe_case(); parsed = parse_case(raw); machine = deepcopy(parsed.machine.to_dict())
    for device in machine["devices"]:
        device["service_modes"]["reg_issue_slots"] = {"resource": "vector", "rate": 2.0e12}
        arch = device.get("compute_arch") or {"instruction_services": {}}
        arch.setdefault("instruction_services", {}).update(vec_issue_clk={"unit": "ops", "owner_axis": "aiv", "clock_hz": 1.6e9, "cycles_per_unit": 1}, vec_linked={"unit": "ops", "owner_axis": "aiv", "per_owner_rate": 1.0e9, "scale_with": {"resource": "vector", "reference_per_owner_rate": 1.0e9}})
        device["compute_arch"] = arch
    for link in machine.get("links", ()):
        link["switch_capacity_bytes_per_ms"] = float(link["capacity_bytes_per_ms"]) * 0.8
    runtime = deepcopy(raw["runtime"])
    runtime["kernels"] = {"service_regimes": [{"name": "custom_hbm_rule", "match": {"recipe": "rmsnorm", "physical_rates.resources.hbm": float(machine["devices"][0]["rates"]["hbm"]), "physical_rates.service_modes.reg_issue_slots": ["vector", 2.0e12], "transport": {"links": [{"capacity_bytes_per_ms": float(machine["links"][0]["capacity_bytes_per_ms"]), "switch_capacity_bytes_per_ms": float(machine["links"][0]["switch_capacity_bytes_per_ms"])}]}}, "coordinates": {"bytes": {"path": "work.hbm"}, "tokens": {"path": "geometry.tokens"}}, "response": {"kind": "affine", "fixed_us": 1.5, "terms": [{"coordinate": "bytes", "us_per_unit": 1e-6}, {"coordinate": "tokens", "us_per_unit": 0.25}]}}]}
    rich = parse_case({**raw, "machine": machine, "runtime": runtime})
    scaled = scale_case_hardware(rich, hbm_bw_scale=2.0, cube_flops_scale=1.5, vector_ops_scale=3.0, link_bw_scale=2.5, hbm_capacity_scale=2.0)
    assert parse_case(scaled.to_dict()) == scaled
    assert same(dict(scaled.machine.dies[0].service_modes)["reg_issue_slots"][1], 6.0e12)
    assert same(scaled.machine.links[0].switch_capacity_bytes_per_ms, rich.machine.links[0].switch_capacity_bytes_per_ms * 2.5)
    terms = {term["coordinate"]: term["us_per_unit"] for term in scaled.runtime.kernels["service_regimes"][0]["response"]["terms"]}
    assert same(terms["bytes"], 1.0e-6) and same(terms["tokens"], 0.25)
    assert dict(scaled.runtime.metadata)["legacy_service_regime_diagnostics"]["scaled"] is False
    for hbm_scale, cube_scale, vec_scale in ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 2.0)):
        scaled_case = scale_case_hardware(parsed, hbm_bw_scale=hbm_scale, cube_flops_scale=cube_scale, vector_ops_scale=vec_scale)
        net = analyze_network_efficiency(scaled_case, phase="decode", tokens=1, context=4096, sequences=4)
        assert same(net["graph_duration_ms"], _decode_costs(scaled_case, 4)["duration_ms"], EXACT)
def test_sla_solver_exact_threshold_and_speculative_tokens():
    raw = _moe_case(layers=4); raw["machine"]["device"]["rates"]["hbm"] = 4e11; parsed = parse_case(raw)
    b3_ms = _decode_costs(parsed, 3)["duration_ms"]
    b4_ms = _decode_costs(parsed, 4)["duration_ms"]
    assert b3_ms < b4_ms
    sla_ms = 0.5 * (b3_ms + b4_ms)
    report = solve_sla_sizing(parsed, sla_ms=sla_ms, context=4096, tokens_per_seq=1, max_sequences_per_dp=8, scope="body", scenarios=[{"id": "2x_hbm", "label": "2x HBM", "hbm_bw_scale": 2.0}, {"id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0}, {"id": "2x_compute", "label": "2x Compute", "cube_flops_scale": 2.0, "vector_ops_scale": 2.0}])
    rows = {row["scenario_id"]: row for row in report["scenarios"]}; base, faster = rows["baseline"], rows["2x_hbm"]
    assert same(base["tps_gain_vs_baseline"], 1.0) and same(base["tps_feasible_gain_vs_baseline"], 1.0)
    assert base["b_sla"] == 3 and base["step_ms_at_sla"] <= sla_ms and base["step_ms_feasible"] <= sla_ms
    assert base["total_feasible_concurrency"] == base["b_feasible"] * parsed.runtime.parallel.dp
    assert faster["b_sla"] > base["b_sla"] and faster["tps_gain_vs_baseline"] > 1.0
    at_sla = _decode_costs(parsed, base["b_sla"])
    assert same(base["step_ms_at_sla"], at_sla["duration_ms"], EXACT) and at_sla["duration_ms"] <= sla_ms
    assert _decode_costs(parsed, base["b_sla"] + 1)["duration_ms"] > sla_ms
    speculative = _moe_case(layers=2)
    sm = speculative["model"]
    speculative["model"]["auxiliary"] = {"mtp": {
        "kind": "mtp", "layers": 1, "hidden_size": sm["hidden_size"],
        "intermediate_size": sm["intermediate_size"], "num_heads": sm["num_heads"],
        "num_kv_heads": sm["num_kv_heads"], "head_dim": sm["head_dim"],
        "q_width": sm["num_heads"] * sm["head_dim"], "kv_width": sm["num_kv_heads"] * sm["head_dim"],
    }}
    speculative["runtime"]["metadata"] = {"speculative": {"draft_len": 3, "acceptance": [0.7, 0.7, 0.7], "acceptance_mode": "conditional"}}
    resolved = solve_sla_sizing(speculative, sla_ms=30.0, context=4096, max_sequences_per_dp=2, scope="body", scenarios=[{"id": "baseline"}])
    assert resolved["tokens_per_seq"] == 4 and same(resolved["emitted_tokens_per_seq"], 1.0 + 0.7 + 0.49 + 0.343)
    single = solve_sla_sizing(speculative, sla_ms=30.0, context=4096, tokens_per_seq=1, max_sequences_per_dp=2, scope="body", scenarios=[{"id": "baseline"}])
    assert single["tokens_per_seq"] == 1 and same(single["emitted_tokens_per_seq"], 1.0)
    assert 0 < _vram_max_sequences_per_dp(parsed, context=32768, tokens_per_seq=64, output_budget=1024) <= _vram_max_sequences_per_dp(parsed, context=32768, tokens_per_seq=1, output_budget=1024)
    for bad_call in (lambda: solve_sla_sizing(parsed, sla_ms=0.0), lambda: scale_case_hardware(parsed, hbm_bw_scale=-1.0), lambda: analyze_network_efficiency(parsed, tokens=0)):
        with pytest.raises(ValueError):
            bad_call()

# --------------------------------------------------------------------------- PD connector / handoff
def _physical_case(**link):
    raw = pd_case(); raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 128, **link}}
    return raw
def _handoff(raw):
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    return result, result["execution"]["pd_handoffs"][0]["transfers"]
def _composed_pd_case(prompt, pools):
    raw = compose_case("deepseek-v3.2", hardware="a5", tp=1, dp=1, ep=1, num_devices=3, layers=1, prompt_tokens=prompt, output_tokens=2)
    raw["pools"] = {pool: {"machine": deepcopy(raw["machine"]), "runtime": deepcopy(raw["runtime"])} for pool in ("prefill", "decode")}
    for pool, ranks in pools:
        machine = raw["pools"][pool]["machine"]
        machine.update(num_devices=len(ranks), devices=[{"rank": rank} for rank in ranks]); machine["links"][0]["members"] = list(ranks)
    raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 64, "latency_us": 0, "semantics": "per_pair"}, "chunk_bytes": 64}
    return raw
@pytest.mark.parametrize("factor", (0.5, 1.0, 2.0))
def test_public_pd_physical_and_connector_bandwidth_scan(factor):
    explicit = predict_pd(_physical_case(capacity_bytes_per_ms=128 * factor), lower=linear_work)
    legacy = pd_case(); legacy["connector"]["bytes_per_s"] *= factor
    original = predict_pd(legacy, lower=linear_work); execution = explicit["execution"]
    assert explicit["requests"] == original["requests"] and explicit["requests"][0]["finish_ms"] == 5 + 3 / factor
    assert (execution["handoff_bytes"], execution["connector"]["wire_bytes"]) == (384, 384)
    assert execution["serviced"]["pd:link:handoff"] == 384
@pytest.mark.parametrize("nested", (False, True))
def test_pd_framing_and_switch_capacity_are_one_physical_transmission(nested):
    link = dict(transaction_bytes=256, alignment_bytes=256, protocol_overhead_bytes=64, switch_hops=2, switch_hop_latency_us=500, switch_capacity_bytes_per_ms=64)
    raw = _physical_case(**link) if nested else pd_case()
    if not nested:
        raw["connector"].update(link)
    result = predict_pd(raw, lower=linear_work); execution = result["execution"]
    # One 384-byte payload occupies two 320-byte transactions; the same wire payload
    # consumes link and switch capacity concurrently, not two sends.
    assert result["requests"][0]["finish_ms"] == 5 + 1 + 640 / 64
    assert (execution["handoff_bytes"], execution["connector"]["wire_bytes"], execution["connector"]["transactions"]) == (384, 640, 2)
    assert all(execution["serviced"][f"{pool}:memory@r0:s0"] == 384 for pool in ("prefill", "decode"))
    transfer = next(event for event in result["events"] if event["task"] == "pd:r:wire")
    assert (transfer["start_ms"], transfer["end_ms"]) == (4, 14)
def test_pd_full_duplex_endpoint_capacity_is_shared_across_ready_handoffs():
    raw = _physical_case(semantics="per_port")
    raw["workload"]["requests"] = [{"id": name, "prompt_tokens": 3, "output_tokens": 3} for name in ("a", "b")]
    for pool in raw["pools"].values():
        pool["runtime"]["dp"], pool["machine"]["num_devices"] = 2, 2; pool["machine"]["links"][0]["members"] = [0, 1]
    result = predict_pd(raw, lower=linear_work); service = result["execution"]["serviced"]
    assert [row["finish_ms"] for row in result["requests"]] == [11, 11]
    assert (service["pd:link:handoff:tx:port0"], service["pd:link:handoff:rx:port1"]) == (768, 768)
    assert service["pd:link:handoff:rx:port0"] == service["pd:link:handoff:tx:port1"] == 0
    assert result["execution"]["connector"]["wire_bytes"] == 768
def test_pd_destination_layout_keeps_each_representative_stage_memory_once():
    raw = _physical_case(transaction_bytes=256, protocol_overhead_bytes=64); destination = case(tp=2, pp=2)
    raw["pools"]["decode"] = {key: deepcopy(destination[key]) for key in ("machine", "runtime")}
    result = predict_pd(raw, lower=linear_work); service = result["execution"]["serviced"]
    assert result["execution"]["handoff_bytes"] == 384
    assert (service["prefill:memory@r0:s0"], service["decode:memory@r0:s0"], service["decode:memory@r0:s1"]) == (384, 96, 96)
    assert result["execution"]["connector"]["wire_bytes"] == 640
def test_pd_framed_handoff_waits_for_source_readiness_and_destination_admission():
    raw = _physical_case(transaction_bytes=256, protocol_overhead_bytes=64, switch_hops=2, switch_hop_latency_us=500)
    raw["workload"]["requests"] = [{"id": name, "prompt_tokens": 3, "output_tokens": 3, "arrival_ms": 7} for name in ("a", "b")]
    raw["pools"]["prefill"]["runtime"]["max_num_batched_tokens"] = 3; raw["pools"]["decode"]["runtime"]["max_num_seqs"] = 1
    result = predict_pd(raw, lower=linear_work)
    assert [row["first_token_abs_ms"] for row in result["requests"]] == [10, 13]
    assert [row["finish_ms"] for row in result["requests"]] == [18, 26]
    transfers = {event["task"]: event for event in result["events"] if event["task"] in {"pd:a:wire", "pd:b:wire"}}
    assert (transfers["pd:a:wire"]["start_ms"], transfers["pd:b:wire"]["start_ms"]) == (11, 19)
    assert result["execution"]["connector"]["wire_bytes"] == 1280
    assert all(value == 0 for pool in result["memory"].values() for stage in pool["live_cache_bytes_per_replica_stage"] for value in stage)
def test_pd_handoff_pairs_replicated_heads_with_every_token_shard():
    raw = pd_model_case(4, 2); bind_pd_pool(raw, "prefill", (0, 1, 2, 3), tp=4, cp=2); bind_pd_pool(raw, "decode", (4,), tp=1, cp=1)
    transfers = _handoff(raw)[1]
    assert len(transfers) == 4 and sum(row["payload_bytes"] for row in transfers) == pytest.approx(512.0)
    assert all(row["payload_bytes"] == pytest.approx(128.0) for row in transfers)
    assert {(row["source_rank"], tuple(row["source_token_span"])) for row in transfers} == {(0, (0, 2)), (1, (0, 2)), (2, (2, 4)), (3, (2, 4))}
    assert {(row["source_rank"], tuple(row["source_heads"])) for row in transfers} == {(0, (0, 1)), (1, (1, 2)), (2, (0, 1)), (3, (1, 2))}
def test_pd_handoff_weights_destination_shards_by_exact_token_rows():
    raw = pd_model_case(5, 1); bind_pd_pool(raw, "prefill", (0,), tp=1, cp=1); bind_pd_pool(raw, "decode", (1, 2), tp=2, cp=2)
    transfers = _handoff(raw)[1]
    assert {(row["destination_rank"], tuple(row["destination_token_span"]), row["payload_bytes"]) for row in transfers} == {(1, (0, 2), 128.0), (2, (2, 5), 192.0)}
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(320.0)
def test_public_baseline_round_trips_the_occurrence_dag():
    raw = case(prompt=4, output=2); raw["model"]["layers"] = 1
    public = json.loads(json.dumps(analyze_network_efficiency(raw, phase="decode", tokens=1, context=32, sequences=1, scope="model")))
    assert "occurrence_dag" in public and len(public["critical_path_nodes"]) == 15
    assert public["critical_path_roofline_ms"] == pytest.approx(0.033888) and public["network_roofline_efficiency"] <= 1.0
def test_pd_handoff_clips_the_window_tail_before_charging():
    raw = pd_model_case(10, 1)
    raw["model"]["layer_plans"] = [{"index": 0, "attention": "gqa", "ffn": "dense", "window": 4, "num_heads": 4, "num_kv_heads": 1, "head_dim": 16, "intermediate_size": 64}]
    bind_pd_pool(raw, "prefill", (2,), tp=1, cp=1); bind_pd_pool(raw, "decode", (0, 1), tp=2, cp=2)
    transfers = _handoff(raw)[1]
    assert len(transfers) == 1
    row = transfers[0]
    assert (row["component"], row["source_rank"], row["destination_rank"], row["payload_bytes"]) == ("gqa_kv", 2, 1, pytest.approx(256.0))
    assert (tuple(row["source_token_span"]), tuple(row["destination_token_span"])) == ((6, 10), (6, 10))
def test_dsa_cp_tp2_cp1_handoff_reads_the_single_cp_shard():
    raw = _composed_pd_case(4, (("prefill", (0, 1)), ("decode", (2,))))
    for pool, tp, plan in (("prefill", 2, "dsa_cp"), ("decode", 1, "none")):
        raw["pools"][pool]["runtime"]["parallel"].update(tp=tp, cp=1, dcp=1, cp_plan=plan)
    rows = {(row["component"], row["source_rank"], row["destination_rank"]): row["payload_bytes"] for row in _handoff(raw)[1]}
    assert rows == {("index_k", 0, 2): pytest.approx(520.0), ("mla_kv", 0, 2): pytest.approx(4608.0)}
def test_pd_dense_mla_window_owns_only_the_intersected_source_rows():
    raw = _composed_pd_case(10, (("prefill", (0, 1)), ("decode", (2, 3))))
    for pool in ("prefill", "decode"):
        raw["pools"][pool]["runtime"]["parallel"].update(tp=2, cp=2, dcp=2, cp_plan="dcp")
    raw["model"]["layer_plans"][0].update(num_kv_heads=0, window=8)
    rows = [(row["source_rank"], tuple(row["source_token_span"]), tuple(row["destination_token_span"]), row["payload_bytes"]) for row in _handoff(raw)[1] if row["component"] == "mla_kv" and row["destination_rank"] == 3]
    # Decode rank 3 owns [5, 10) (five 1,152 B rows); prefill rank 0 owns the
    # disjoint [2, 5) window head and must not be charged for it.
    assert rows == [(1, (5, 10), (5, 10), 5760.0)]

# --------------------------------------------------------------------------- serving I/O and step summary
def _captured_case(count):
    raw = case(requests=[{"id": str(index), "prompt_tokens": 1, "output_tokens": 2} for index in range(count)])
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full", submission={"kernel_launch_ms": 0.1, "graph_replay_ms": 0.25}, serving_io={"h2d": {"bytes_per_s": 8000, "bytes_per_token": 8, "token_basis": "padded"}})
    return raw
def test_public_transfer_phases_preserve_three_completion_boundaries():
    raw = case(prompt=4, output=3); raw["runtime"]["serving_io"] = _IO
    result = predict(raw, lower=linear_work, event_limit=200); row = result["requests"][0]
    device, host, client = (result["metric_boundaries"][key] for key in ("device_token_commit", "host_result_ready", "client_delivery"))
    assert (row["first_token_abs_ms"], row["finish_ms"], host["window_end_ms"], client["window_end_ms"]) == (8.75, 17.75, 19.5, 21.5)
    assert (row["boundaries"]["host_result_ready"]["ttft_ms"], row["boundaries"]["client_delivery"]["ttft_ms"]) == (10.5, 12.5)
    assert device["tpot_ms"]["mean"] == host["tpot_ms"]["mean"] == client["tpot_ms"]["mean"] == 4.5
    service = result["execution"]["serviced"]
    assert (service["h2d@r0"], service["d2h@r0"], service["memory@r0:s0"], service["scheduler@r0"]) == (48, 12, 60, 3)
    assert result["metrics"]["window_end_ms"] == 19.5 and result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
def test_d2h_tail_changes_host_tps_without_polluting_device_commit_window():
    raw = case(prompt=4, output=1); raw["runtime"]["serving_io"] = deepcopy(_IO)
    first = predict(raw, lower=linear_work); raw["runtime"]["serving_io"]["d2h"]["bytes_per_token"] *= 10
    second = predict(raw, lower=linear_work)
    assert first["metric_boundaries"]["device_token_commit"] == second["metric_boundaries"]["device_token_commit"]
    assert second["metric_boundaries"]["device_token_commit"]["output_tps"] == pytest.approx(1000 / 8.75)
    assert (second["metric_boundaries"]["host_result_ready"]["window_end_ms"], second["metric_boundaries"]["client_delivery"]["window_end_ms"]) == (19.5, 21.5)
    assert second["metric_boundaries"]["host_result_ready"]["output_tps"] < first["metric_boundaries"]["host_result_ready"]["output_tps"]
def test_input_copy_contends_with_other_work_on_endpoint_memory():
    raw = case(prompt=1, output=1); raw["machine"]["device"]["rates"]["memory"] = 8000
    raw["runtime"]["serving_io"] = {"h2d": {"bytes_per_s": 1e9, "bytes_per_token": 8}}; parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_serving_io=True, trace_limit=100)
    work = linear_work(IterationPlan((RequestView("r", 1, 0, True, "prefill"),)), parsed.model, parsed.runtime, parsed.machine)
    execution.engine.submit("other_memory", {"memory@r0:s0": 16}); step = execution.add(work, 0, 0); execution.engine.advance()
    assert step.completion.time == 3  # two ms shared HBM copy, then one ms compute
    assert execution.engine.serviced["memory@r0:s0"] == 24
    starts = [event["start_ms"] for event in execution.engine.trace if event["demand"].get("matrix@r0:s0")]
    assert starts and min(starts) == 2
def test_closed_loop_releases_after_explicit_client_delivery_and_think_gap():
    raw = case(requests=[{"id": "a", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1}, {"id": "b", "prompt_tokens": 1, "output_tokens": 1, "chain_id": 1, "gap_ms": 2}])
    raw["runtime"]["serving_io"] = {"host_result_ms": 1, "client_delivery_ms": 5}
    first, second = predict(raw, lower=linear_work)["requests"]
    assert (first["finish_ms"], first["boundaries"]["client_delivery"]["finish_ms"]) == (1, 7)
    assert (second["arrival_ms"], second["finish_ms"], second["boundaries"]["client_delivery"]["finish_ms"]) == (9, 10, 16)
def test_undeclared_client_latency_has_no_client_metric_and_graph_scope_omits_io():
    raw = case(prompt=2, output=2); raw["runtime"]["serving_io"] = {"host_result_ms": 3}
    result = predict(raw, lower=linear_work)
    assert set(result["metric_boundaries"]) == {"device_token_commit", "host_result_ready"} and result["requests"][0]["finish_ms"] == 6
    graphs = target_graphs(raw, phase="decode", tokens=1); report = graph_costs(raw, graphs, include_submission=True)
    control = deepcopy(raw); control["runtime"].pop("serving_io")
    assert not report["execution"]["serving_io"]["enabled"]
    assert report["duration_ms"] == graph_costs(control, graphs, include_submission=True)["duration_ms"]
def test_pd_orders_client_output_even_with_different_pool_latencies():
    raw = pd_case(); raw["pools"]["prefill"]["runtime"]["serving_io"] = {"client_delivery_ms": 20}; raw["pools"]["decode"]["runtime"]["serving_io"] = {"client_delivery_ms": 1}
    result = predict(raw, lower=linear_work); row = result["requests"][0]; delivered = row["boundaries"]["client_delivery"]
    assert (row["finish_ms"], row["boundaries"]["host_result_ready"]["finish_ms"], delivered["first_token_abs_ms"], delivered["finish_ms"]) == (8, 8, 23, 23)
    assert delivered["itl"] == {"count": 2, "mean_ms": 0, "max_ms": 0} and result["execution"]["handoff_bytes"] == 384
    assert result["metric_boundaries"]["device_token_commit"]["output_tps"] == 375
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
def test_speculative_bursts_copy_only_committed_tokens_and_keep_long_gaps():
    raw = case(output=6); raw["runtime"]["metadata"] = {"draft_mode": "mtp", "draft_proposal_tokens": 2, "draft_accept_trace": [0, 2], "draft_accept_tokens": 1}
    raw["runtime"]["serving_io"] = {"d2h": {"bytes_per_s": 4000, "bytes_per_token": 4}}
    row = predict(raw, lower=linear_work)["requests"][0]; host = row["boundaries"]["host_result_ready"]
    assert row["speculative"]["proposed_tokens"] == 4 and host["generated_tokens"] == 6 and host["itl"]["count"] == 5
    assert host["itl"]["max_ms"] > host["tpot_ms"]
def test_capture_bucket_and_oversize_eager_path_keep_input_bytes_and_output_count():
    captured, oversized = predict(_captured_case(3), trace_limit=0), predict(_captured_case(5), trace_limit=0)
    assert (captured["execution"]["submission"]["graph_replays"], oversized["execution"]["submission"]["graph_replays"]) == (1, 0)
    assert captured["execution"]["serviced"]["h2d@r0"] == (3 + 4) * 8 and oversized["execution"]["serviced"]["h2d@r0"] == pytest.approx((5 + 5) * 8)
    assert (captured["metrics"]["output_tokens"], oversized["metrics"]["output_tokens"]) == (6, 10)
    groups = captured["execution"]["step_summary"]["groups"]
    assert {(row["mode"], row["replay_scope"], row["graph_bucket"]) for row in groups} == {("prefill", "none", None), ("decode", "full", 4)}
    total = captured["execution"]["step_summary"]["total"]
    assert (total["steps"], total["valid_tokens"], total["padded_tokens"]) == (2, 6, 7)
def test_all_microbatch_statistics_are_independent_of_bounded_trace():
    raw = _captured_case(2); raw["runtime"]["pipeline_microbatches"] = 2
    without, with_trace = predict(raw, trace_limit=0, event_limit=0), predict(raw, trace_limit=1, event_limit=5)
    summary = without["execution"]["step_summary"]
    assert summary == with_trace["execution"]["step_summary"] and without["traces"] == [] and len(with_trace["traces"]) == 1
    assert (summary["total"]["steps"], without["execution"]["iterations"]) == (4, 2)
    assert (summary["total"]["valid_tokens"], summary["total"]["padded_tokens"]) == (4, 10)
    assert sum(row["duration_ms"]["sum"] for row in summary["groups"]) == pytest.approx(summary["total"]["duration_ms"]["sum"])
    eager = deepcopy(raw); eager["runtime"]["graph_mode"] = "eager"; result = predict(eager, trace_limit=0)
    assert result["execution"]["submission"]["graph_replays"] == 0 and result["execution"]["step_summary"]["total"]["padded_tokens"] == 4
    assert result["metrics"]["output_tokens"] == without["metrics"]["output_tokens"]

# --------------------------------------------------------------------------- topology / PD ratio sweeps
def _sweep_case(model="deepseek-v3.2", hardware="a2", tp=2, dp=2, ep=2, layers=4, **extra):
    defaults = dict(prompt_tokens=512, output_tokens=1024, mtp_proposal_tokens=0)
    return compose_case(model, hardware=hardware, tp=tp, dp=dp, ep=ep, layers=layers, **(defaults | extra))
@pytest.mark.parametrize("sku", DT_SKUS)
def test_dt_resize_preserves_declared_link_semantics(sku):
    raw = _sweep_case("qwen3.8-flash-next", sku, tp=4, dp=1, ep=4, layers=4, num_devices=4); parsed = parse_case(raw)
    for target in (1, 2, 8):
        machine = materialize_machine_for_devices(parsed, raw, target)
        if target == 1:
            assert len(machine.dies) == 1 and machine.links == ()
            continue
        intra = [link for link in machine.links if link.channel == "d2d_intra_node"]
        assert len(machine.dies) == target and intra and all(link.semantics == "per_pair" for link in intra)
        assert intra[0].capacity_bytes_per_ms == pytest.approx(47.5e6) and intra[0].effective_peak_capacity_bytes_per_ms == pytest.approx(56.0e6)
        assert intra[0].bw_utilization == pytest.approx(47.5 / 56.0) and intra[0].latency_us == pytest.approx(12.0)
        if target > machine.ranks_per_node:
            inter = [link for link in machine.links if link.channel == "d2d_inter_node"]
            assert inter and all(link.semantics == "per_port" for link in inter) and inter[0].capacity_bytes_per_ms == pytest.approx(50.0e6)
@pytest.mark.parametrize("sku", PER_PORT_SKUS)
def test_per_port_resize_preserves_declared_link_semantics(sku):
    raw = _sweep_case("deepseek-v3.2", sku, tp=2, dp=2, ep=2, layers=4, num_devices=4)
    machine = materialize_machine_for_devices(parse_case(raw), raw, 8)
    intra = [link for link in machine.links if link.channel == "d2d_intra_node"]
    semantics, rates = ("per_pair", (18.2e6, 24e6)) if sku == "Ascend910B1" else ("per_port", (196e6, 196e6))
    assert len(machine.dies) == 8 and intra and all(link.semantics == semantics for link in intra)
    assert all((link.capacity_bytes_per_ms, link.effective_peak_capacity_bytes_per_ms) == pytest.approx(rates) for link in intra)
def test_resize_scales_to_the_node_grid_and_keeps_channel_projection():
    raw = _sweep_case("qwen3.8-flash-next", DT_SKUS[0], tp=4, dp=1, ep=4, layers=4, num_devices=4, ranks_per_node=4); parsed = parse_case(raw)
    for target in (2, 4, 16, 64):
        assert len(materialize_machine_for_devices(parsed, raw, target).dies) == target
    legacy = deepcopy(raw); channels = legacy["machine"]["transfer_channels"]
    channels["d2d_intra_node"] = dict(channels["d2d_intra_node"], semantics="per_port", channel_semantics="unknown", effective_uni_bw_gbps=47.5)
    channels["d2d_intra_package"] = dict(channels["d2d_intra_package"], channel_semantics="none", effective_uni_bw_gbps=None)
    legacy_parsed = parse_case(legacy)
    for target in (1, 2, 8):
        machine = materialize_machine_for_devices(legacy_parsed, legacy, target)
        intra = [link for link in machine.links if link.channel == "d2d_intra_node"]
        assert len(machine.dies) == target and (not intra or intra[0].semantics == "per_port")
def test_parallel_candidate_enumeration_covers_dense_moe_pp_pcp_and_mla_replicate():
    dense = parse_case(_dense_case())
    for devices, expected in ((8, [1, 2, 4, 8]), (16, [1, 2, 4, 8, 16])):
        candidates = enumerate_parallel_candidates(dense, num_devices=devices)
        assert [row["tp"] for row in candidates] == expected
        assert all(row["dp"] * row["tp"] * row["pp"] * row["pcp"] == devices and (row["ep"], row["pp"], row["pcp"]) == (1, 1, 1) for row in candidates)
    laid_out = enumerate_parallel_candidates(parse_case(_dense_case(layers=3)), num_devices=8, allow_pp=True, allow_pcp=True, candidate_tps=[2])
    assert any(row["pp"] == 2 for row in laid_out) and any(row["pcp"] == 2 for row in laid_out)
    assert all(row["dp"] * row["tp"] * row["pp"] * row["pcp"] == 8 for row in laid_out)
    assert enumerate_parallel_candidates(dense.model, num_devices=8, allow_dp=False) == [{"dp": 1, "tp": 8, "ep": 1, "pp": 1, "pcp": 1}]
    eight = enumerate_parallel_candidates(parse_case(_moe_case(layers=2)), num_devices=8)
    assert all(row["dp"] * row["tp"] * row["pp"] * row["pcp"] == 8 for row in eight)
    assert [row["ep"] for row in eight if row["tp"] == 2] == [1, 2, 4, 8] and [row["ep"] for row in eight if row["tp"] == 8] == [1, 2, 4, 8]
    sixteen = enumerate_parallel_candidates(parse_case(_moe_case(layers=2)), num_devices=16)
    assert all(row["dp"] * row["tp"] * row["pp"] * row["pcp"] == 16 for row in sixteen)
    assert [row["ep"] for row in sixteen if row["tp"] == 4] == [1, 2, 4, 8, 16]
    assert [row["tp"] for row in enumerate_parallel_candidates({"num_attention_heads": 12, "num_key_value_heads": 4, "n_routed_experts": 0}, num_devices=8)] == [1, 2, 4]
    replicated = enumerate_parallel_candidates({"num_attention_heads": 12, "num_key_value_heads": 1, "n_routed_experts": 8, "mla_replicate": True}, num_devices=8, candidate_tps=[4, 8])
    assert {row["tp"] for row in replicated} == {4, 8}
def test_topology_sweep_pareto_frontier_and_node_topology():
    raw = _moe_case(layers=2, dp=1, tp=4, ep=4, device=False); original = deepcopy(raw)
    candidates = [{"dp": 4, "tp": 2, "ep": 4, "pp": 1, "pcp": 1}, {"dp": 2, "tp": 4, "ep": 4, "pp": 1, "pcp": 1}, {"dp": 1, "tp": 8, "ep": 8, "pp": 1, "pcp": 1}, {"dp": 1, "tp": 8, "ep": 32, "pp": 1, "pcp": 1}]
    report = sweep_topologies(raw, candidates=candidates, num_devices=8, sla_ms=15.0, context=4096, phase="decode", scope="body", output_budget=256, max_sequences_per_dp=16)
    assert raw == original and report["schema"] == "mindie-sim.topology-sweep/1" and report["num_devices"] == 8
    assert (report["search_domain"]["num_devices"], report["search_domain"]["candidate_count"], len(report["candidates"])) == (8, 4, 4)
    feasible = [row for row in report["candidates"] if row["feasible"]]; infeasible = [row for row in report["candidates"] if row["status"] == "infeasible"]
    assert len(feasible) == 3 and len(infeasible) == 1
    assert (infeasible[0]["parallel"]["ep"], infeasible[0]["pareto_optimal"], bool(infeasible[0]["reason"])) == (32, False, True)
    assert all(row["ranks_per_node"] == 4 for row in feasible)
    scaled = sweep_topologies(_dense_case(layers=2, dp=2, tp=4), candidates=[{"dp": 3, "tp": 4, "ep": 1, "pp": 1, "pcp": 1}], num_devices=12, sla_ms=15.0, context=4096, max_sequences_per_dp=4)
    wide = sweep_topologies(raw, candidates=[{"dp": 4, "tp": 4, "ep": 4, "pp": 1, "pcp": 1}], num_devices=16, sla_ms=15.0, context=4096, max_sequences_per_dp=4)
    assert (scaled["candidates"][0]["ranks_per_node"], wide["candidates"][0]["ranks_per_node"]) == (8, 4)
    slow = _dense_case(layers=1, dp=1, tp=2); slow["machine"].update(intra_node_bw_gbps=10.0, inter_node_bw_gbps=8.0)
    overrides = [{"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1}]
    from_dict = sweep_topologies(slow, candidates=overrides, num_devices=4, sla_ms=30.0, context=4096, output_budget=64, max_sequences_per_dp=4)
    from_case = sweep_topologies(parse_case(slow), candidates=overrides, num_devices=4, sla_ms=30.0, context=4096, output_budget=64, max_sequences_per_dp=4)
    assert same(from_dict["candidates"][0]["step_ms_at_b1"], from_case["candidates"][0]["step_ms_at_b1"], EXACT)
    assert same(from_dict["candidates"][0]["tps_feasible"], from_case["candidates"][0]["tps_feasible"], EXACT)
    by_tp = {row["parallel"]["tp"]: row for row in feasible}
    assert by_tp[8]["weight_bytes_per_die"] < by_tp[4]["weight_bytes_per_die"] < by_tp[2]["weight_bytes_per_die"]
    for row in feasible:
        assert row["status"] == "feasible" and row["b_feasible"] == min(row["b_sla"], row["b_vram_max"])
        assert row["b_sla"] >= 1 and row["b_vram_max"] >= 1 and row["total_feasible_concurrency"] == row["b_feasible"] * row["parallel"]["dp"]
        assert same(row["tps_per_die"], row["tps_feasible"] / 8.0)
        assert 0.0 < row["network_mbu"] <= 1.0 and 0.0 < row["network_cube_mfu"] <= 1.0 and 0.0 <= row["network_vec_mfu"] <= 1.0
    frontier = report["pareto_frontier"]; assert frontier and all(row["pareto_optimal"] is True for row in frontier)
    for first in frontier:
        for second in frontier:
            if first is second:
                continue
            assert not (second["tps_per_die"] >= first["tps_per_die"] and second["total_feasible_concurrency"] >= first["total_feasible_concurrency"] and second["step_ms_at_b1"] <= first["step_ms_at_b1"] and (second["tps_per_die"] > first["tps_per_die"] or second["total_feasible_concurrency"] > first["total_feasible_concurrency"] or second["step_ms_at_b1"] < first["step_ms_at_b1"]))
    auto = sweep_topologies(_dense_case(layers=1, dp=1, tp=4), num_devices=8, sla_ms=0.001, context=1024, max_sequences_per_dp=4)
    assert [row["parallel"]["tp"] for row in auto["candidates"]] == [1, 2, 4, 8]
    assert all(row["status"] == "sla_or_vram_exceeded" and not row["feasible"] for row in auto["candidates"]) and auto["pareto_frontier"] == []
def test_pd_ratio_sweep_conserves_kv_handoff_and_reports_each_bottleneck():
    prefill = _dense_case(layers=1, dp=2, tp=2); decode = _dense_case(layers=1, dp=1, tp=4, max_num_seqs=4)
    report = sweep_pd_disaggregation_ratios(prefill, decode, total_dies=16, prompt_tokens=128, output_tokens=256, ttft_sla_ms=2000.0, tpot_sla_ms=25.0, kv_transfer_bw_gbps=100.0)
    assert (report["schema"], report["total_dies"]) == ("mindie-sim.pd-sweep/1", 16)
    p_inst, d_inst = report["prefill_instance"], report["decode_instance"]; assert p_inst["hbm_ok"] is True
    assert same(p_inst["rps_per_instance"], (2 * 1000.0) / p_inst["step_ms"], EXACT)
    assert same(d_inst["rps_per_instance"], (d_inst["b_feasible"] * 1 * 1.0 * 1000.0) / (d_inst["step_ms_feasible"] * 255.0), EXACT)
    parsed = parse_case(decode); kv_bytes = sum(Memory(parsed).cache_bytes(128, padded=False)) * parsed.runtime.parallel.tp * parsed.runtime.parallel.pcp
    transfer_ms = kv_bytes / (100.0 * 1e6)
    assert same(report["kv_bytes_per_request"], kv_bytes, EXACT) and same(report["kv_transfer_ms_per_request"], transfer_ms, EXACT)
    assert same(report["effective_tpot_ms"], report["decode_step_ms"] + transfer_ms / 255.0, EXACT) and same(report["first_decode_itl_ms"], transfer_ms + report["decode_step_ms"], EXACT)
    ratios = {row["pd_ratio"]: row for row in report["candidates"]}
    assert {"1P:1D", "1P:2D", "1P:3D", "2P:1D", "2P:2D", "3P:1D"} <= set(ratios)
    for row in report["candidates"]:
        assert row["used_dies"] == row["n_prefill"] * 4 + row["n_decode"] * 4 <= 16 and row["feasible"] is True and row["infeasible_reason"] is None
        assert same(row["effective_tpot_ms"], report["effective_tpot_ms"], EXACT) and same(row["first_decode_itl_ms"], report["first_decode_itl_ms"], EXACT)
        assert same(row["sustainable_rps"], min(row["prefill_rps"], row["kv_transfer_rps"], row["decode_rps"]), EXACT) and same(row["sustainable_tps"], row["sustainable_rps"] * 256.0, EXACT)
    assert report["pareto_frontier"] and report["best_configuration"] in report["pareto_frontier"] and ratios["3P:1D"]["bottleneck_stage"] == "decode"
    kv_bound = sweep_pd_disaggregation_ratios(prefill, decode, total_dies=16, prompt_tokens=128, output_tokens=256, ttft_sla_ms=2000.0, tpot_sla_ms=25.0, kv_transfer_bw_gbps=0.003125)
    assert all(row["feasible"] and row["bottleneck_stage"] == "kv_transfer" for row in kv_bound["candidates"])
    exceeded = sweep_pd_disaggregation_ratios(prefill, decode, total_dies=16, prompt_tokens=128, output_tokens=256, ttft_sla_ms=2000.0, tpot_sla_ms=25.0, kv_transfer_bw_gbps=0.0000625)
    assert all(not row["feasible"] and row["infeasible_reason"] == "effective_tpot_sla_exceeded" for row in exceeded["candidates"])
    oom = deepcopy(prefill); oom["runtime"]["memory"] = {"workspace_bytes": 64 * 1024**3}
    oom_report = sweep_pd_disaggregation_ratios(oom, decode, total_dies=16, prompt_tokens=128, output_tokens=256, ttft_sla_ms=3000.0, tpot_sla_ms=30.0)
    assert oom_report["prefill_instance"]["hbm_ok"] is False and all(not row["feasible"] and row["infeasible_reason"] == "prefill_hbm_oom" for row in oom_report["candidates"])
    short = deepcopy(decode); short["workload"]["requests"][0]["prompt_tokens"] = 128; short["runtime"]["max_model_len"] = 1024
    short_report = sweep_pd_disaggregation_ratios(prefill, short, total_dies=16, prompt_tokens=128, output_tokens=2048, ttft_sla_ms=3000.0, tpot_sla_ms=30.0)
    assert all(not row["feasible"] and row["infeasible_reason"] == "decode_max_model_len_exceeded" for row in short_report["candidates"])
    oom_decode = deepcopy(decode); oom_decode["runtime"]["memory"] = {"workspace_bytes": 64 * 1024**3}
    single = sweep_pd_disaggregation_ratios(prefill, oom_decode, total_dies=16, prompt_tokens=128, output_tokens=1, ttft_sla_ms=3000.0, tpot_sla_ms=30.0)
    assert (single["kv_bytes_per_request"], single["kv_transfer_ms_per_request"], single["decode_step_ms"]) == (0.0, 0.0, 0.0)
    assert all(row["feasible"] and row["bottleneck_stage"] == "prefill" for row in single["candidates"])
    one_slot = deepcopy(prefill); one_slot["runtime"]["max_num_seqs"] = 1
    retention = sweep_pd_disaggregation_ratios(one_slot, decode, total_dies=16, prompt_tokens=128, output_tokens=256, ttft_sla_ms=3000.0, tpot_sla_ms=30.0, kv_transfer_bw_gbps=0.003125)
    assert same(retention["prefill_instance"]["rps_per_instance"], (2 * 1000.0) / (retention["prefill_instance"]["step_ms"] + retention["kv_transfer_ms_per_request"]))
def test_compute_pareto_frontier_strict_dominance():
    rows = [{"id": "a", "feasible": True, "tps_per_die": 100.0, "total_feasible_concurrency": 32, "step_ms_at_b1": 10.0}, {"id": "b", "feasible": True, "tps_per_die": 90.0, "total_feasible_concurrency": 32, "step_ms_at_b1": 11.0}, {"id": "c", "feasible": True, "tps_per_die": 80.0, "total_feasible_concurrency": 64, "step_ms_at_b1": 12.0}, {"id": "d", "feasible": False, "tps_per_die": 500.0, "total_feasible_concurrency": 128, "step_ms_at_b1": 1.0}]
    assert [row["id"] for row in compute_pareto_frontier(rows)] == ["a", "c"] and rows[1]["pareto_optimal"] is False and rows[3]["pareto_optimal"] is False

# --------------------------------------------------------------------------- streaming sweep laws
def _stream_case(layers=2):
    return compose_case("qwen3-32b", hardware="a3", tp=1, ep=1, pp=1, num_devices=1, layers=layers, prompt_tokens=8, output_tokens=2, concurrency=1, max_num_seqs=1)
def _run_sweep(raw, path, **kwargs):
    return list(sweep_topologies_iter(raw, sla_ms=30.0, context=8, output_budget=2, max_sequences_per_dp=1, two_stage=False, output_jsonl=path, **kwargs))
def test_public_sweep_symbols_and_streamed_identity():
    import mindie_sim
    for name in ("sweep_topologies", "sweep_topology_grid", "sweep_topologies_iter", "sweep_topology_grid_iter", "reconstruct_sweep_summary", "solve_sla_sizing", "solve_hardware_sla_throughput"):
        assert hasattr(mindie_sim, name)
    assert solve_hardware_sla_throughput is solve_sla_sizing
    raw = _sweep_case(); candidates = [{"dp": 2, "tp": 2, "ep": 2}, {"dp": 4, "tp": 1, "ep": 4}]
    rows = list(sweep_topologies_iter(raw, candidates=candidates, num_devices=4, sla_ms=50.0, context=512, phase="decode"))
    assert all(len(row["candidate_id"]) == 32 for row in rows) and len({row["candidate_id"] for row in rows}) == len(rows)
    assert reconstruct_sweep_summary(rows) == sweep_topologies(raw, candidates=candidates, num_devices=4, sla_ms=50.0, context=512, phase="decode")
def test_sweep_checkpoint_persists_every_device_grid_and_derivation_metadata(tmp_path):
    grid = tmp_path / "grid.jsonl"; rows = _run_sweep(_stream_case(), grid, num_devices_grid=(1, 2))
    persisted = [json.loads(line) for line in grid.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 1 and len(persisted) == len(rows) and len({row["candidate_id"] for row in persisted}) == len(rows)
    derived = tmp_path / "derived.jsonl"; derivations = [{"sku": "derived-a", "hbm_peak_bytes_per_s": 1.5e12}, {"sku": "derived-b", "hbm_peak_bytes_per_s": 1.2e12}]
    derived_rows = _run_sweep(_stream_case(), derived, num_devices=1, hardware_derivations=derivations)
    records = [json.loads(line) for line in derived.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(derived_rows) == len(records) == 2 and {row["derived_sku"] for row in records} == {"derived-a", "derived-b"}
    assert all(row["derivation"]["sku"] == row["derived_sku"] for row in records)
def test_streamed_checkpoint_repairs_a_torn_tail_and_preserves_identity(tmp_path):
    path = tmp_path / "sweep.jsonl"
    first = list(sweep_topologies_iter(_stream_case(), output_jsonl=path, num_devices_grid=(1, 2), **SWEEP))
    with path.open("ab") as stream:
        stream.write(b'{"candidate_id": "truncated')
    resumed = list(sweep_topologies_iter(_stream_case(), checkpoint_jsonl=path, resume=True, num_devices_grid=(1, 2), **SWEEP))
    assert len(resumed) == len(first) and all(row.get("resumed_from_checkpoint") is True for row in resumed)
    assert [row["candidate_id"] for row in resumed] == [row["candidate_id"] for row in first]
    actual = reconstruct_sweep_summary(resumed)["candidates"]
    for row in actual:
        row.pop("resumed_from_checkpoint", None)
    assert actual == reconstruct_sweep_summary(first)["candidates"] and path.read_bytes().endswith(b"\n")
def test_resume_upgrades_screening_rows_and_skips_committed_stages(tmp_path):
    path = tmp_path / "resume.jsonl"; rows = list(sweep_topologies_iter(_stream_case(), two_stage=True, output_jsonl=path, num_devices=1, **SWEEP))
    assert rows
    path.write_bytes(b"".join((json.dumps({**row, "evaluation_stage": "b1_screen"}, sort_keys=True) + "\n").encode() for row in rows))
    resumed = list(sweep_topologies_iter(_stream_case(), two_stage=True, output_jsonl=path, resume=True, num_devices=1, **SWEEP))
    assert resumed and not any(row.get("resumed_from_checkpoint") for row in resumed) and any(row["evaluation_stage"] == "full_sla" for row in resumed)
def test_checkpoint_alias_conflict_and_finalist_validation(tmp_path):
    with pytest.raises(ValueError, match="conflicting_checkpoint_paths"):
        list(sweep_topologies_iter(_stream_case(), output_jsonl=tmp_path / "a.jsonl", checkpoint_jsonl=tmp_path / "b.jsonl", **SWEEP))
    with pytest.raises(ValueError, match="finalist_top_k_must_be_positive"):
        list(sweep_topologies_iter(_stream_case(), two_stage=True, finalist_top_k=0, **SWEEP))
def test_static_validation_rejects_without_lowering_the_graph(monkeypatch):
    calls = []; monkeypatch.setattr(efficiency_module, "target_graphs", lambda *args, **kwargs: calls.append(1) or (_ for _ in ()).throw(AssertionError("graph lowered for a statically invalid candidate")))
    raw = _sweep_case("qwen3.5-35b-a3b", "a2", tp=2, dp=2, ep=2, layers=4)
    candidates = [{"dp": 3, "tp": 2, "ep": 2}, {"dp": 1, "tp": 3, "ep": 1}, {"dp": 1, "tp": 2, "ep": 3}, {"dp": 1, "tp": 4, "ep": 4, "cp": 8, "dcp": 8}]
    rows = list(sweep_topologies_iter(raw, candidates=candidates, num_devices=4, sla_ms=50.0, context=512, phase="decode"))
    assert calls == [] and len(rows) == len(candidates)
    assert all(row["evaluation_stage"] == "static_validation" and row["feasible"] is False and row["rejection_reason"] for row in rows)
def test_capacity_precheck_rejects_before_memory(monkeypatch):
    monkeypatch.setattr(sweep_engine, "stage_weight_bytes", lambda *args, **kwargs: [1.0e30])
    rows = list(sweep_topologies_iter(_sweep_case(), candidates=[{"dp": 2, "tp": 2, "ep": 2}], num_devices=4, sla_ms=50.0, context=512))
    assert (len(rows), rows[0]["evaluation_stage"], rows[0]["feasibility_status"]) == (1, "static_validation", "capacity_infeasible")
    assert rows[0]["rejection_reason"].startswith("capacity_exceeded:")
def test_replica_equivalence_scales_dp_exactly_and_two_stage_marks_estimates():
    report = sweep_topologies(_sweep_case(tp=2, dp=2, ep=2, layers=4), num_devices_grid=(2, 4, 8), sla_ms=50.0, context=512, phase="decode", allow_ep=True)
    assert report["cache_stats"]["model_spec_reuses"] > 0 and report["cache_stats"]["replica_equivalence_hits"] > 0
    by_dp = {row["parallel"]["dp"]: row for row in report["candidates"] if row["parallel"].get("tp") == 2 and row["parallel"].get("ep") == 2 and row["feasible"]}
    assert same(by_dp[1]["step_ms_at_b1"], by_dp[2]["step_ms_at_b1"], EXACT) and same(by_dp[1]["step_ms_at_b1"], by_dp[4]["step_ms_at_b1"], EXACT)
    assert same(by_dp[2]["tps_feasible"], 2.0 * by_dp[1]["tps_feasible"], EXACT) and same(by_dp[4]["tps_feasible"], 4.0 * by_dp[1]["tps_feasible"], EXACT)
    screened = list(sweep_topologies_iter(_sweep_case(), candidates=[{"dp": 2, "tp": 2, "ep": 2}, {"dp": 4, "tp": 1, "ep": 4}], num_devices=4, sla_ms=50.0, context=512, phase="decode", two_stage=True))
    assert all(row["is_screening_estimate"] is True for row in screened if row["evaluation_stage"] == "b1_screen")
    assert all(row["is_screening_estimate"] is False for row in screened if row["evaluation_stage"] == "full_sla")
    assert reconstruct_sweep_summary(screened)["screening_summary"]["two_stage"] is True
def test_candidate_timeout_and_outer_deadline(monkeypatch):
    raw = _sweep_case(); candidates = [{"dp": 2, "tp": 2, "ep": 2}, {"dp": 4, "tp": 1, "ep": 4}]
    real_full, calls = sweep_module.full_row, {"count": 0}
    def slow_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("candidate_timeout")
        return real_full(*args, **kwargs)
    monkeypatch.setattr(sweep_module, "full_row", slow_first)
    rows = list(sweep_topologies_iter(raw, candidates=candidates, num_devices=4, sla_ms=50.0, context=512, candidate_timeout_s=5.0))
    assert any(row["status"] == "timeout" for row in rows) and any(row.get("evaluation_stage") != "timeout" for row in rows)
    with pytest.raises(TimeoutError):
        list(sweep_topologies_iter(raw, candidates=candidates, num_devices=4, sla_ms=50.0, context=512, deadline_monotonic=time.monotonic() - 1.0))
def test_b1_screen_never_runs_the_graph(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("b1_screen executed a full graph")
    monkeypatch.setattr(efficiency_module, "_lower_and_run_graph", forbidden)
    parsed = parse_case(_stream_case()); parallel = ParallelSpec.from_mapping({"dp": 1, "tp": 1, "ep": 1, "pp": 1})
    machine = materialize_machine_for_devices(parsed, parsed, 1); candidate = build_candidate_case(parsed, machine, parallel, ())
    static = validate_candidate_statically(parsed, parallel, cand_devices=1, cand_machine=machine, part=(), phase="decode", context=8, q_tokens=1, emit_tokens=1.0, output_budget=2, max_sequences_per_dp=1)
    row = b1_screen(parsed, candidate, parallel, common={"candidate_id": "c", "parallel": {"dp": 1}, "num_devices": 1, "ranks_per_node": 1}, ctx=EvaluationContext.create(candidate), phase="decode", scope="body", context=8, q_tokens=1, emit_tokens=1.0, sla_ms=1e-9, b_vram_max=int(static["b_vram_max_joint"]), max_sequences_per_dp=1, output_budget=2, static=static)
    assert (row["evaluation_stage"], row["evaluation_boundary"], row["is_screening_estimate"], row["b1_sla_pass"]) == ("b1_screen", "b1_isolated_operator_sum", True, True)
    assert row["operator_sum_screen_ms"] == pytest.approx(row["step_ms_at_b1"]) and row["operator_sum_screen_ms"] > 0.0 and row["tps_screen_b1"] == pytest.approx(row["tps_at_b1"])
    assert row["graph_ttft_ms"] is None and row["search"]["evaluated_candidates"] == [SCREEN_BATCH]
def test_finalist_top_k_ranks_the_global_screen():
    rows = list(sweep_topologies_iter(_stream_case(), two_stage=True, finalist_top_k=1, num_devices_grid=(1, 2), **SWEEP))
    screened = [row for row in rows if row["evaluation_stage"] == "b1_screen"]; full = [row for row in rows if row["evaluation_stage"] == "full_sla"]
    best = min(screened, key=lambda row: (row["operator_sum_screen_ms"], row["candidate_id"]))
    assert len(screened) >= 2 and len(full) == 1 and full[0]["candidate_id"] == best["candidate_id"] and full[0]["graph_step_ms"] == pytest.approx(full[0]["step_ms_at_b1"])
def test_sweep_cli_exposes_the_bounded_flags(tmp_path):
    args = parser().parse_args(["sweep", "case.json", "--two-stage", "--finalist-top-k", "3", "--checkpoint-jsonl", str(tmp_path / "c.jsonl"), "--isolate-workers", "--resume"])
    assert (args.finalist_top_k, args.checkpoint_jsonl, args.isolate_workers, args.resume) == (3, str(tmp_path / "c.jsonl"), True, True) and args.output_jsonl is None
@pytest.mark.parametrize("field", ("channel_semantics", "semantics"))
def test_invalid_channel_semantics_is_rejected_with_explicit_links(field):
    raw = compose_case("qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=8, layers=1, ranks_per_node=8)["machine"]
    links = [link.to_dict() for link in Hardware.from_mapping(raw).links]; raw["links"] = links
    raw["transfer_channels"]["d2d_inter_node"][field] = "invented"; assert links
    with pytest.raises(ModelingError, match="transfer_channel_semantics_unsupported"):
        Hardware.from_mapping(raw)
def test_composed_neutral_machine_rejects_invalid_channel_semantics():
    raw = compose_case("qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=8, layers=1, ranks_per_node=8)
    raw["machine"]["transfer_channels"]["d2d_intra_node"]["semantics"] = "invented"
    with pytest.raises(ModelingError, match="transfer_channel_semantics_unsupported"):
        parse_case(raw)


def test_sla_certified_search_finds_nonmonotone_b5_and_maximizes_throughput_b4_over_b6(monkeypatch):
    """Dual-certificate SLA solver finds B=5 when B=1..4 fail and picks B=4 over lower-TPS B=6."""
    parsed = parse_case(_dense_case(layers=1))
    empty_graphs = {0: ()}

    def _fake_runner(durations: dict[int, float]):
        def _run(case_arg, *, sequences, **_kw):
            dur = float(durations[int(sequences)])
            return empty_graphs, case_arg, {"duration_ms": dur, "graph_duration_ms": dur}
        return _run

    def _fake_analysis(durations: dict[int, float]):
        def _ans(_case, *, sequences, **_kw):
            dur = float(durations[int(sequences)])
            return {
                "graph_duration_ms": dur, "operator_sum_ms": dur,
                "network_mbu": 0.5, "network_cube_mfu": 0.5, "network_vec_mfu": 0.1,
                "network_link_util": 0.0, "network_roofline_efficiency": 0.5,
                "breakdown_by_bound": {k: {"share_of_operator_sum": 0.2} for k in (
                    "memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound")},
            }
        return _ans

    # 1. Non-monotone dispatch: B=1..4 fail SLA (45 ms > 30 ms), B=5 passes (20 ms <= 30 ms).
    durs_nonmono = {1: 45.0, 2: 45.0, 3: 45.0, 4: 45.0, 5: 20.0, 6: 45.0, 7: 45.0, 8: 45.0}
    monkeypatch.setattr(efficiency_module, "target_graphs", lambda *_a, **_k: empty_graphs)
    monkeypatch.setattr(efficiency_module, "_lower_and_run_graph", _fake_runner(durs_nonmono))
    monkeypatch.setattr(efficiency_module, "analyze_network_efficiency", _fake_analysis(durs_nonmono))
    res_nm = solve_sla_sizing(parsed, sla_ms=30.0, context=128, max_sequences_per_dp=8, scope="body", scenarios=[{"id": "baseline"}])
    sc_nm = res_nm["scenarios"][0]
    assert sc_nm["b_sla"] == 5 and sc_nm["b_feasible"] == 5 and sc_nm["search"]["non_monotonic_dispatch_detected"] is True

    # 2. Throughput maximization: B=4 (4/10=0.40 seq/ms) beats B=6 (6/25=0.24 seq/ms) under SLA=30 ms.
    durs_tps = {1: 8.0, 2: 9.0, 3: 9.5, 4: 10.0, 5: 22.0, 6: 25.0, 7: 40.0, 8: 45.0}
    monkeypatch.setattr(efficiency_module, "_lower_and_run_graph", _fake_runner(durs_tps))
    monkeypatch.setattr(efficiency_module, "analyze_network_efficiency", _fake_analysis(durs_tps))
    res_tps = solve_sla_sizing(parsed, sla_ms=30.0, context=128, max_sequences_per_dp=8, scope="body", scenarios=[{"id": "baseline"}])
    sc_tps = res_tps["scenarios"][0]
    assert sc_tps["b_sla"] == 4 and sc_tps["b_feasible"] == 4 and sc_tps["step_ms_at_sla"] == pytest.approx(10.0)
