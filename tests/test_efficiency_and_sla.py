"""Tier 0 unit tests for 4D MFU/MBU/CV efficiency, target roofline fitting, and SLA scaling.

[Role] Verify operator-level and whole-network Cube_MFU, Vec_MFU, MBU, Link_Util,
  CV_Overlap, target efficiency evaluation (0.8x MBU), multi-pipe roofline
  regime fitting, and counterfactual hardware scaling (2x HBM / 2x Compute) SLA
  throughput solving.
[Invariants] Pure in-memory Tier 0 execution (< 5s total); zero subprocesses,
  zero network, zero sleep; exact conservation of FLOPs/OPs/Bytes and 0.00%
  baseline drift at 1.0x hardware scaling.
[Timeouts & Locks] In-memory synthetic and representative model geometries only.
[Side-Effect Policy] No file writes, device contexts, or global state mutation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import pytest

from mindie_sim.prediction.body_defaults import TargetGraphSet, consume_target_declaration
from mindie_sim.prediction.efficiency import (
    _vram_max_sequences_per_dp,
    analyze_network_efficiency,
    analyze_operator_row,
    evaluate_target_efficiency,
    extract_hardware_peaks,
    fit_case_roofline_regimes,
    fit_efficiency_targets,
    scale_case_hardware,
    solve_hardware_sla_throughput,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


def _make_representative_moe_case(*, layers: int = 4, dp: int = 1, tp: int = 4, ep: int = 4) -> dict:
    """Build an in-memory hybrid MLA + MoE case on Ascend910_9362."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
            "device": {
                "rates": {"native_call_service": 1e9},
                "resource_units": {"native_call_service": 1},
                "work_units": {"native_call_service": "busy_ns"},
            },
        },
        "model": {
            "name": "tier0-mla-moe",
            "layers": layers,
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 1,
            "head_dim": 128,
            "intermediate_size": 2048,
            "vocab_size": 64000,
            "attention": "mla",
            "ffn": "moe",
            "activation_dtype": "bf16",
            "weight_dtype": "int8",
            "kv_dtype": "bf16",
            "compute_dtype": "bf16",
            "quantization": "w8a8",
            "num_experts": 16,
            "top_k": 2,
            "moe_intermediate_size": 1024,
            "num_shared_experts": 1,
            "mla": {
                "q_lora_rank": 1024,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
            },
        },
        "runtime": {
            "tp": tp,
            "dp": dp,
            "ep": ep,
            "pp": 1,
            "max_model_len": 131072,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "kv_block_size": 128,
        },
        "workload": {
            "requests": [
                {"request_id": "req0", "prompt_tokens": 4096, "output_tokens": 16, "arrival_ms": 0.0}
            ]
        },
    }


def test_operator_four_dimensional_classification_pmu_and_executed_service():
    """Verify 4D classification, PMU vs unidentifiable CV overlap, and executed resource_work/serviced accounting."""
    case = parse_case(_make_representative_moe_case())
    peaks = extract_hardware_peaks(case)
    bf16_peak = peaks["compute_peaks"]["bf16"]
    int8_peak = peaks["compute_peaks"]["int8"]
    vec_peak = peaks["vector_peak_ops_per_s"]
    hbm_peak = peaks["hbm_peak_bytes_per_s"]
    link_peak = peaks["link_peak_bytes_per_s"]

    # 1. Cube-only operator (mixed int8 + bf16 matrix_flops, Cube-bound)
    cube_row = {
        "physical_id": "op_cube",
        "occurrences": 4,
        "program": "SteadyMatmul",
        "duration_ms": 0.20,
        "operator": {
            "op_id": "l0.q_proj",
            "family": "gemm",
            "recipe": "matmul",
            "phase": "decode",
            "stage": 0,
            "layer": 0,
            "dtype": "int8",
            "flops": 0.10 * 1e-3 * int8_peak + 0.04 * 1e-3 * bf16_peak,
            "matrix_flops": {"int8": 0.10 * 1e-3 * int8_peak, "bf16": 0.04 * 1e-3 * bf16_peak},
            "vector_ops": 0.0,
            "hbm_bytes": 0.05 * 1e-3 * hbm_peak,
            "link_bytes": 0.0,
        },
    }
    res_cube = analyze_operator_row(cube_row, case, peaks=peaks)
    assert res_cube["operator_kind"] == "cube"
    assert res_cube["dominant_bound"] == "cube_bound"
    assert math.isclose(res_cube["t_cube_ideal_ms"], 0.14, rel_tol=1e-9)
    assert math.isclose(res_cube["cube_mfu"], 0.14 / 0.20, rel_tol=1e-9)
    assert res_cube["vec_mfu"] is None
    assert math.isclose(res_cube["mbu"], 0.05 / 0.20, rel_tol=1e-9)

    # 2. CV-fused operator: without PMU and t_cube + t_vec <= duration, cv_overlap_ratio is None
    cv_row = {
        "physical_id": "op_cv",
        "occurrences": 8,
        "program": "StagedGqaDecode",
        "duration_ms": 0.25,
        "operator": {
            "op_id": "l0.flash_attn",
            "family": "attention",
            "recipe": "flash_attention_decode",
            "phase": "decode",
            "stage": 0,
            "layer": 0,
            "dtype": "bf16",
            "flops": 0.08 * 1e-3 * bf16_peak,
            "vector_ops": 0.12 * 1e-3 * vec_peak,
            "hbm_bytes": 0.18 * 1e-3 * hbm_peak,
            "link_bytes": 0.0,
        },
    }
    res_cv_no_pmu = analyze_operator_row(cv_row, case, peaks=peaks)
    assert res_cv_no_pmu["operator_kind"] == "cv_fused"
    assert res_cv_no_pmu["cv_overlap_ratio"] is None

    # When t_cube + t_vec (0.20ms) > duration_ms (0.16ms), lower-bound overlap is proven (> 0)
    res_cv_proven = analyze_operator_row({**cv_row, "duration_ms": 0.16}, case, peaks=peaks)
    assert math.isclose(res_cv_proven["cv_overlap_ratio"], (0.20 - 0.16) / 0.08, rel_tol=1e-9)

    # With PMU counters: partial overlap = 0.75, full = 1.0, serial = 0.0
    res_cv_partial = analyze_operator_row(
        cv_row, case, peaks=peaks,
        pmu={"aic_mac_time_us": 100.0, "aiv_vec_time_us": 80.0, "aic_wait_time_us": 20.0, "aiv_wait_time_us": 40.0},
    )
    assert math.isclose(res_cv_partial["cv_overlap_ratio"], 0.75, rel_tol=1e-9)
    res_cv_full = analyze_operator_row(
        cv_row, case, peaks=peaks,
        pmu={"aic_mac_time_us": 100.0, "aiv_vec_time_us": 80.0, "compute_span_us": 100.0},
    )
    assert math.isclose(res_cv_full["cv_overlap_ratio"], 1.0, rel_tol=1e-9)
    res_cv_serial = analyze_operator_row(
        cv_row, case, peaks=peaks,
        pmu={"aic_mac_time_us": 100.0, "aiv_vec_time_us": 80.0, "compute_span_us": 180.0},
    )
    assert math.isclose(res_cv_serial["cv_overlap_ratio"], 0.0, rel_tol=1e-9)

    # 3. Executed service & metadata.resource_work vs joint_native_service analytical_reference
    serviced_row = {
        "physical_id": "op_serviced",
        "occurrences": 2,
        "program": "PagedAttention",
        "duration_ms": 0.20,
        "serviced": {
            "memory@r0:s0": 0.10 * 1e-3 * hbm_peak,
            "vector@r0:s0": 0.04 * 1e-3 * vec_peak,
            "mte2_aiv@r0:s0": 0.10 * 1e-3 * hbm_peak,
        },
        "communication": {"wire_by_resource": {"intra_package:0@r0": 0.02 * 1e-3 * link_peak}},
        "operator": {
            "op_id": "l0.paged_attn",
            "family": "attention",
            "recipe": "paged_attention",
            "phase": "decode",
            "stage": 0,
            "layer": 0,
            "dtype": "bf16",
            "flops": 0.0,
            "vector_ops": 0.0,
            "hbm_bytes": 0.0,
            "link_bytes": 0.0,
            "metadata": {"resource_work": {"cube": 0.05 * 1e-3 * bf16_peak}},
        },
    }
    res_serv = analyze_operator_row(serviced_row, case, peaks=peaks)
    assert res_serv["work_basis"] == "executed_service"
    assert math.isclose(res_serv["t_hbm_ideal_ms"], 0.10, rel_tol=1e-9)
    assert math.isclose(res_serv["t_vec_ideal_ms"], 0.04, rel_tol=1e-9)
    assert math.isclose(res_serv["t_cube_ideal_ms"], 0.05, rel_tol=1e-9)
    assert math.isclose(res_serv["t_link_ideal_ms"], 0.02, rel_tol=1e-9)

    jn_row = {
        **serviced_row,
        "joint_native_service": {"service": "native_call_service"},
        "analytical_reference_work": {"hbm": 0.08 * 1e-3 * hbm_peak},
        "serviced": {"native_call_service@r0:s0": 200000.0},
    }
    res_jn = analyze_operator_row(jn_row, case, peaks=peaks)
    assert res_jn["work_basis"] == "analytical_reference"
    assert math.isclose(res_jn["t_hbm_ideal_ms"], 0.08, rel_tol=1e-9)


def test_heterogeneous_dies_and_undeclared_precision_error():
    """Verify conservative min peak extraction across heterogeneous dies and ModelingError on undeclared dtype."""
    raw = _make_representative_moe_case(dp=1, tp=2, ep=2)
    case = parse_case(raw)
    m_dict = deepcopy(case.machine.to_dict())
    # Make die 1 slower than die 0 on bf16, hbm, and vector, and remove int4 from compute_peaks
    for dev in m_dict["devices"]:
        dev["compute_peaks"].pop("int4", None)
    m_dict["devices"][1]["compute_peaks"]["bf16"] = m_dict["devices"][0]["compute_peaks"]["bf16"] * 0.5
    m_dict["devices"][1]["rates"]["hbm"] = m_dict["devices"][0]["rates"]["hbm"] * 0.6
    m_dict["devices"][1]["rates"]["vector"] = m_dict["devices"][0]["rates"]["vector"] * 0.7
    het_case = parse_case({**raw, "machine": m_dict})
    peaks = extract_hardware_peaks(het_case, stage=0, replica=0)
    assert math.isclose(peaks["compute_peaks"]["bf16"], m_dict["devices"][1]["compute_peaks"]["bf16"], rel_tol=1e-12)
    assert math.isclose(peaks["hbm_peak_bytes_per_s"], m_dict["devices"][1]["rates"]["hbm"], rel_tol=1e-12)
    assert math.isclose(peaks["vector_peak_ops_per_s"], m_dict["devices"][1]["rates"]["vector"], rel_tol=1e-12)

    bad_dtype_row = {
        "physical_id": "op_int4",
        "occurrences": 1,
        "program": "SteadyMatmul",
        "duration_ms": 0.10,
        "operator": {
            "op_id": "l0.int4_gemm",
            "family": "gemm",
            "recipe": "matmul",
            "phase": "decode",
            "stage": 0,
            "layer": 0,
            "dtype": "int4",
            "flops": 1e9,
            "vector_ops": 0.0,
            "hbm_bytes": 1e6,
            "link_bytes": 0.0,
        },
    }
    with pytest.raises(ModelingError):
        analyze_operator_row(bad_dtype_row, het_case, peaks=peaks)


def test_inclusive_target_efficiency_and_minimum_ms_floor():
    """Verify 0.80 target MBU solves for inclusive 0.80 MBU and respects kernel minimum_ms floor."""
    case = parse_case(_make_representative_moe_case())
    peaks = extract_hardware_peaks(case)
    hbm_peak = peaks["hbm_peak_bytes_per_s"]

    # Operator with 8ms ideal HBM, 2ms startup, 16ms actual duration -> at 0.80 target MBU, target_duration_ms == 10.0ms
    op_report = {
        "operators": [
            {
                "physical_id": "op_mem_8ms",
                "occurrences": 1,
                "program": "ResourceEnvelope",
                "duration_ms": 16.0,
                "total_ms": 16.0,
                "basis": {"invocation_models": [{"startup_ms": 2.0, "minimum_ms": 0.0}]},
                "operator": {
                    "op_id": "l0.mem_heavy",
                    "family": "attention",
                    "recipe": "paged_attention",
                    "phase": "decode",
                    "stage": 0,
                    "layer": 0,
                    "dtype": "bf16",
                    "flops": 0.0,
                    "vector_ops": 0.0,
                    "hbm_bytes": 8.0 * 1e-3 * hbm_peak,
                    "link_bytes": 0.0,
                },
            },
            {
                "physical_id": "op_min_floor",
                "occurrences": 1,
                "program": "ResourceEnvelope",
                "duration_ms": 12.0,
                "total_ms": 12.0,
                "basis": {"invocation_models": [{"startup_ms": 0.1, "minimum_ms": 10.0}]},
                "operator": {
                    "op_id": "l0.min_bounded",
                    "family": "norm",
                    "recipe": "rmsnorm",
                    "phase": "decode",
                    "stage": 0,
                    "layer": 0,
                    "dtype": "bf16",
                    "flops": 0.0,
                    "vector_ops": 0.0,
                    "hbm_bytes": 0.5 * 1e-3 * hbm_peak,
                    "link_bytes": 0.0,
                },
            },
        ]
    }
    base = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=4096, sequences=1,
        operator_report=op_report, graph_report={"duration_ms": 28.0},
    )
    ev = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=4096, sequences=1,
        target_mbu=0.80, baseline_analysis=base,
    )
    by_id = {o["op_id"]: o for o in ev["operators"]}
    assert math.isclose(by_id["l0.mem_heavy"]["target_duration_ms"], 10.0, rel_tol=1e-9)
    assert math.isclose(by_id["l0.mem_heavy"]["target_mbu"], 0.80, rel_tol=1e-9)
    assert math.isclose(by_id["l0.mem_heavy"]["service_only_target_mbu"], 1.0, rel_tol=1e-9)
    assert math.isclose(by_id["l0.min_bounded"]["target_duration_ms"], 10.0, rel_tol=1e-9)


def test_parallel_graph_roofline_efficiency_and_fit_signature_distinctness():
    """Verify stage-capacity network_roofline_efficiency <= 1.0 on parallel branches and distinct fit keys."""
    case = parse_case(_make_representative_moe_case())
    peaks = extract_hardware_peaks(case)
    hbm_peak = peaks["hbm_peak_bytes_per_s"]
    bf16_peak = peaks["compute_peaks"]["bf16"]

    # Two parallel branches: branch A uses 0.8ms HBM (1.0ms duration), branch B uses 0.7ms Cube (1.0ms duration);
    # concurrent graph span is 1.0ms while serial operator sum is 2.0ms.
    op_report = {
        "operators": [
            {
                "physical_id": "branch_mem",
                "occurrences": 1,
                "program": "ResourceEnvelope",
                "duration_ms": 1.0,
                "total_ms": 1.0,
                "operator": {
                    "op_id": "l0.branch_a",
                    "family": "custom",
                    "recipe": "dup_recipe",
                    "phase": "decode",
                    "stage": 0,
                    "layer": 0,
                    "dtype": "bf16",
                    "local_shape": [1, 4096],
                    "flops": 0.0,
                    "vector_ops": 0.0,
                    "hbm_bytes": 0.8 * 1e-3 * hbm_peak,
                    "link_bytes": 0.0,
                },
            },
            {
                "physical_id": "branch_cube",
                "occurrences": 1,
                "program": "SteadyMatmul",
                "duration_ms": 1.0,
                "total_ms": 1.0,
                "operator": {
                    "op_id": "l0.branch_b",
                    "family": "custom",
                    "recipe": "dup_recipe",
                    "phase": "decode",
                    "stage": 0,
                    "layer": 0,
                    "dtype": "bf16",
                    "local_shape": [16, 4096],
                    "flops": 0.7 * 1e-3 * bf16_peak,
                    "vector_ops": 0.0,
                    "hbm_bytes": 0.0,
                    "link_bytes": 0.0,
                },
            },
        ]
    }
    net = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=4096, sequences=1,
        operator_report=op_report, graph_report={"duration_ms": 1.0},
    )
    assert net["network_roofline_efficiency"] <= 1.0
    assert math.isclose(net["network_roofline_efficiency"], 0.80, rel_tol=1e-9)
    assert math.isclose(net["operator_serial_roofline_efficiency"], 1.5 / 2.0, rel_tol=1e-9)

    fit = fit_case_roofline_regimes(case, baseline_analysis=net)
    assert len(fit["by_key"]) == 2


def test_hardware_scaling_roundtrip_and_independent_graph_costs():
    """Verify parse_case(scaled.to_dict()) == scaled and exact agreement with independent graph_costs at 2x HBM/Compute."""
    raw = _make_representative_moe_case()
    # Add custom instruction_services (one clock-derived, one scale_with), service_modes, switch_capacity, and service_regimes
    case = parse_case(raw)
    m_dict = deepcopy(case.machine.to_dict())
    for dev in m_dict["devices"]:
        dev["service_modes"]["reg_issue_slots"] = {"resource": "vector", "rate": 2.0e12}
        arch = dev.get("compute_arch") or {"instruction_services": {}}
        ins = arch.setdefault("instruction_services", {})
        ins["vec_issue_clk"] = {"unit": "ops", "owner_axis": "aiv", "clock_hz": 1.6e9, "cycles_per_unit": 1}
        ins["vec_linked"] = {
            "unit": "ops",
            "owner_axis": "aiv",
            "per_owner_rate": 1.0e9,
            "scale_with": {"resource": "vector", "reference_per_owner_rate": 1.0e9},
        }
        dev["compute_arch"] = arch
    for lnk in m_dict.get("links", ()):
        lnk["switch_capacity_bytes_per_ms"] = float(lnk["capacity_bytes_per_ms"]) * 0.8
    rt_dict = deepcopy(raw["runtime"])
    rt_dict["kernels"] = {
        "service_regimes": [
            {
                "name": "custom_hbm_rule",
                "match": {
                    "recipe": "rmsnorm",
                    "physical_rates.resources.hbm": float(m_dict["devices"][0]["rates"]["hbm"]),
                    "physical_rates.service_modes.reg_issue_slots": ["vector", 2.0e12],
                    "transport": {"links": [{"capacity_bytes_per_ms": float(m_dict["links"][0]["capacity_bytes_per_ms"]), "switch_capacity_bytes_per_ms": float(m_dict["links"][0]["switch_capacity_bytes_per_ms"])}]},
                },
                "coordinates": {
                    "bytes": {"path": "work.hbm"},
                    "tokens": {"path": "geometry.tokens"},
                },
                "response": {
                    "kind": "affine",
                    "fixed_us": 1.5,
                    "terms": [
                        {"coordinate": "bytes", "us_per_unit": 1e-6},
                        {"coordinate": "tokens", "us_per_unit": 0.25},
                    ],
                },
            }
        ]
    }
    rich_case = parse_case({**raw, "machine": m_dict, "runtime": rt_dict})
    scaled = scale_case_hardware(
        rich_case, hbm_bw_scale=2.0, cube_flops_scale=1.5, vector_ops_scale=3.0, link_bw_scale=2.5, hbm_capacity_scale=2.0,
    )
    # Round-trip equality proves compute_arch derived supplies and rates are consistent
    assert parse_case(scaled.to_dict()) == scaled

    # Verify reg_issue_slots scaled by vector_ops_scale (3.0) and switch_capacity scaled by link_bw_scale (2.5)
    assert math.isclose(dict(scaled.machine.dies[0].service_modes)["reg_issue_slots"][1], 6.0e12, rel_tol=1e-9)
    assert math.isclose(scaled.machine.links[0].switch_capacity_bytes_per_ms, rich_case.machine.links[0].switch_capacity_bytes_per_ms * 2.5, rel_tol=1e-9)
    # Verify work.hbm term scaled by 1/2.0 while geometry.tokens term was NOT scaled
    scaled_rule = scaled.runtime.kernels["service_regimes"][0]
    terms_by_coord = {t["coordinate"]: t["us_per_unit"] for t in scaled_rule["response"]["terms"]}
    assert math.isclose(terms_by_coord["bytes"], 0.5e-6, rel_tol=1e-9)
    assert math.isclose(terms_by_coord["tokens"], 0.25, rel_tol=1e-9)

    # Independent graph_costs verification under 2x HBM and 2x Compute
    for hbm_s, cube_s, vec_s in ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 2.0)):
        sc = scale_case_hardware(case, hbm_bw_scale=hbm_s, cube_flops_scale=cube_s, vector_ops_scale=vec_s)
        net = analyze_network_efficiency(sc, phase="decode", tokens=1, context=4096, sequences=4)
        graphs = target_graphs(sc, phase="decode", tokens=1, context=4096, sequences=4, scope="body")
        eff_sc = consume_target_declaration(sc, graphs)
        sg = TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels, selection=graphs.default_selection)
        gr = graph_costs(eff_sc, sg, external_inputs={"analysis_input": 0}, include_submission=False, isolated=False)
        assert math.isclose(net["graph_duration_ms"], gr["duration_ms"], rel_tol=1e-12)


def test_sla_solver_exact_threshold_and_speculative_tokens():
    """Verify SLA solver on a multi-layer case against independent graph_costs, plus speculative token & VRAM-feasible metrics."""
    case_ml = parse_case(_make_representative_moe_case(layers=4))
    # Choose an interior SLA threshold (0.88 ms) where B=1 fits (0.71ms) and B=8 exceeds SLA (0.98ms)
    sla_report = solve_hardware_sla_throughput(
        case_ml,
        sla_ms=0.88,
        context=4096,
        tokens_per_seq=1,
        max_sequences_per_dp=8,
        scenarios=[
            {"id": "2x_hbm", "label": "2x HBM", "hbm_bw_scale": 2.0},
            {"id": "baseline", "label": "1x Baseline", "hbm_bw_scale": 1.0},
            {"id": "2x_compute", "label": "2x Compute", "cube_flops_scale": 2.0, "vector_ops_scale": 2.0},
            {"id": "target_0.8x_mbu", "label": "Target 0.8x MBU", "target_mbu": 0.80},
        ],
    )
    by_id = {row["scenario_id"]: row for row in sla_report["scenarios"]}
    base = by_id["baseline"]
    hbm2 = by_id["2x_hbm"]

    # Reordered scenario list must still identify 'baseline' with gain == 1.0
    assert math.isclose(base["tps_gain_vs_baseline"], 1.0, rel_tol=1e-9)
    assert math.isclose(base["tps_feasible_gain_vs_baseline"], 1.0, rel_tol=1e-9)
    assert 1 < base["b_sla"] < 8
    assert base["step_ms_at_sla"] <= 0.88
    assert base["step_ms_feasible"] <= 0.88
    assert base["total_feasible_concurrency"] == base["b_feasible"] * case_ml.runtime.parallel.dp
    assert hbm2["b_sla"] > base["b_sla"]
    assert hbm2["tps_gain_vs_baseline"] > 1.0

    # Independently verify on the multi-layer graph that b_sla <= 0.88ms and b_sla + 1 > 0.88ms
    g_at_sla = target_graphs(case_ml, phase="decode", tokens=1, context=4096, sequences=base["b_sla"], scope="body")
    ec_at_sla = consume_target_declaration(case_ml, g_at_sla)
    gr_at_sla = graph_costs(ec_at_sla, TargetGraphSet({0: g_at_sla[0]}, kernels=g_at_sla.effective_kernels, selection=g_at_sla.default_selection), external_inputs={"analysis_input": 0}, include_submission=False, isolated=False)
    assert math.isclose(base["step_ms_at_sla"], gr_at_sla["duration_ms"], rel_tol=1e-12)
    assert gr_at_sla["duration_ms"] <= 0.88

    g_next = target_graphs(case_ml, phase="decode", tokens=1, context=4096, sequences=base["b_sla"] + 1, scope="body")
    ec_next = consume_target_declaration(case_ml, g_next)
    gr_next = graph_costs(ec_next, TargetGraphSet({0: g_next[0]}, kernels=g_next.effective_kernels, selection=g_next.default_selection), external_inputs={"analysis_input": 0}, include_submission=False, isolated=False)
    assert gr_next["duration_ms"] > 0.88

    # Speculative token resolution: SpeculativeSpec draft_len=3, conditional 0.7 -> q_tokens=4, emit=1+0.7+0.49+0.343=2.533
    raw_spec = _make_representative_moe_case(layers=2)
    raw_spec["runtime"]["metadata"] = {
        "speculative": {"draft_len": 3, "acceptance": [0.7, 0.7, 0.7], "acceptance_mode": "conditional"}
    }
    spec_res = solve_hardware_sla_throughput(raw_spec, sla_ms=30.0, context=4096, max_sequences_per_dp=2, scenarios=[{"id": "baseline"}])
    assert spec_res["tokens_per_seq"] == 4
    assert math.isclose(spec_res["emitted_tokens_per_seq"], 1.0 + 0.7 + 0.49 + 0.343, rel_tol=1e-9)
    # Explicit tokens_per_seq=1 on a speculative case must emit 1.0 token/step
    spec_1tok = solve_hardware_sla_throughput(raw_spec, sla_ms=30.0, context=4096, tokens_per_seq=1, max_sequences_per_dp=2, scenarios=[{"id": "baseline"}])
    assert spec_1tok["tokens_per_seq"] == 1
    assert math.isclose(spec_1tok["emitted_tokens_per_seq"], 1.0, rel_tol=1e-9)

    # VRAM ceiling decreases when tokens_per_seq increases batch-dependent workspace
    b_vram_q1 = _vram_max_sequences_per_dp(case_ml, context=32768, tokens_per_seq=1, output_budget=1024)
    b_vram_q64 = _vram_max_sequences_per_dp(case_ml, context=32768, tokens_per_seq=64, output_budget=1024)
    assert 0 < b_vram_q64 <= b_vram_q1

    # Input validation checks
    with pytest.raises(ValueError):
        solve_hardware_sla_throughput(case_ml, sla_ms=0.0)
    with pytest.raises(ValueError):
        scale_case_hardware(case_ml, hbm_bw_scale=-1.0)
    with pytest.raises(ValueError):
        evaluate_target_efficiency(case_ml, target_mbu=1.5)
    with pytest.raises(ValueError):
        analyze_network_efficiency(case_ml, tokens=0)
