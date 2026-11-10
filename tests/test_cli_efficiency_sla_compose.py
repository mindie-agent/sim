"""Unit and CLI tests for deterministic Case synthesis, efficiency, and SLA commands.

[Role]
  Verify ``compose_case``, ``resolve_hardware_sku``, and the CLI subcommands
  ``compose``, ``efficiency``, and ``sla`` across catalog models and Ascend
  hardware generations (A2, A3, A5).

[Invariants]
  * ``compose_case`` outputs parse cleanly via ``parse_case`` and execute
    ``predict(..., scope="body")`` deterministically across A2, A3, and A5.
  * CLI ``efficiency`` and ``sla`` outputs satisfy ``mindie-sim.efficiency/1``
    and ``mindie-sim.hardware-sla/1`` schemas and operator duration conservation.
  * Invalid hardware SKUs or parallelism dimensions raise ``ModelingError``.

[Timeouts & Locks]
  Tier 0 pure in-memory + tmp_path JSON CLI tests (< 5s total); no device locks.

[Side-Effect Policy]
  Writes only to pytest ``tmp_path``; zero host or repository pollution.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pytest

from mindie_sim import predict
from mindie_sim.__main__ import main
from mindie_sim.compose import HARDWARE_ALIASES, compose_case, resolve_hardware_sku
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency, extract_hardware_peaks,
)
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.operator_analysis import target_graphs, target_operators
from mindie_sim.prediction.scheduler import DraftPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView


def test_resolve_hardware_sku_aliases_and_errors() -> None:
    assert resolve_hardware_sku("a2") == "Ascend910B4"
    assert resolve_hardware_sku("910b") == "Ascend910B4"
    assert resolve_hardware_sku("910b1") == "Ascend910B1"
    assert resolve_hardware_sku("910b2") == "Ascend910B2"
    assert resolve_hardware_sku("910b3") == "Ascend910B3"
    assert resolve_hardware_sku("a3") == "Ascend910_9392"
    assert resolve_hardware_sku("910c") == "Ascend910_9392"
    assert resolve_hardware_sku("910_9382") == "Ascend910_9382"
    assert resolve_hardware_sku("a5") == "Ascend910_9599"
    assert resolve_hardware_sku("950") == "Ascend910_9599"
    assert resolve_hardware_sku("Ascend950DT_9582") == "Ascend950DT_9582"
    assert set(HARDWARE_ALIASES).issuperset({"a2", "a3", "a5", "910b4", "910_9392", "910_9599"})

    with pytest.raises(ModelingError, match="unknown_hardware_sku"):
        resolve_hardware_sku("unknown_gpu_999")


@pytest.mark.parametrize(
    ("model_name", "hw_alias", "expected_sku", "tp", "dp", "ep"),
    [
        ("qwen3.5-35b-a3b", "a2", "Ascend910B4", 4, 1, 4),
        ("qwen3.5-35b-a3b", "a3", "Ascend910_9392", 4, 2, 8),
        ("qwen3.5-35b-a3b", "a5", "Ascend910_9599", 4, 2, 8),
        ("glm-5.3-flash", "a2", "Ascend910B4", 4, 1, 4),
        ("glm-5.3-flash", "a3", "Ascend910_9392", 4, 2, 8),
        ("glm-5.3-flash", "a5", "Ascend910_9599", 4, 2, 8),
        ("deepseek-v4-flash", "a2", "Ascend910B4", 8, 1, 8),
        ("deepseek-v4-flash", "a3", "Ascend910_9392", 8, 2, 16),
        ("deepseek-v4-flash", "a5", "Ascend910_9599", 8, 2, 16),
    ],
)
def test_compose_case_catalog_models_across_a2_a3_a5(
    model_name: str,
    hw_alias: str,
    expected_sku: str,
    tp: int,
    dp: int,
    ep: int,
) -> None:
    case_dict = compose_case(
        model=model_name,
        hardware=hw_alias,
        tp=tp,
        dp=dp,
        ep=ep,
        prompt_tokens=2048,
        output_tokens=64,
        concurrency=2,
    )
    assert case_dict["machine"]["sku"] == expected_sku
    assert case_dict["machine"]["num_devices"] == tp * dp
    expected_rpn = min(tp * dp, 16 if expected_sku.startswith("Ascend910_93") else 8)
    assert case_dict["machine"]["ranks_per_node"] == expected_rpn
    assert len(case_dict["workload"]["requests"]) == 2
    assert case_dict["workload"]["requests"][0]["id"] == "req-0"
    assert case_dict["workload"]["requests"][1]["id"] == "req-1"

    parsed = parse_case(case_dict)
    assert parsed.machine.sku == expected_sku

    fast_case = compose_case(
        model=model_name,
        hardware=hw_alias,
        tp=tp,
        dp=dp,
        ep=ep,
        layers=4,
        prompt_tokens=2048,
        output_tokens=64,
        concurrency=1,
    )
    pred = predict(fast_case, scope="body", phase="decode", tokens=1, context=2048, sequences=1)
    assert pred["duration_ms"] > 0.0


def test_compose_case_synthetic_layer_and_expert_overrides() -> None:
    case_dict = compose_case(
        model="qwen3.5-35b-a3b",
        hardware="a3",
        tp=2,
        dp=2,
        ep=4,
        layers=8,
        n_routed_experts=32,
        num_experts_per_tok=4,
        quant="w8a8",
    )
    model_dict = case_dict["model"]
    assert model_dict["layers"] == 8
    assert model_dict["n_routed_experts"] == 32
    assert model_dict["num_experts"] == 32
    assert model_dict["num_experts_per_tok"] == 4
    assert model_dict["top_k"] == 4
    assert len(model_dict["layer_plans"]) == 8
    moe_plans = [p for p in model_dict["layer_plans"] if p.get("ffn") == "moe"]
    assert moe_plans
    for plan in moe_plans:
        assert plan["num_experts"] == 32
        assert plan["top_k"] == 4

    parsed = parse_case(case_dict)
    assert parsed.model.layers == 8
    assert parsed.model.num_experts == 32
    assert parsed.model.top_k == 4

    pred = predict(case_dict, scope="body", phase="decode", tokens=1, context=1024, sequences=2)
    assert pred["duration_ms"] > 0.0


def test_compose_case_execution_controls_and_long_context() -> None:
    views = (RequestView("req-0", 1, 2048, False, "decode"),)
    c_full = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=2, graph_capture_mode="full")
    c_pw = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=2, graph_capture_mode="piecewise")
    assert GraphPolicy().select(views, parse_case(c_full).runtime) == (1, "full")
    assert GraphPolicy().select(views, parse_case(c_pw).runtime) == (1, "piecewise")

    c_mtp = compose_case(
        "qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=2,
        mtp_proposal_tokens=2, mtp_acceptance_length=1.0,
    )
    parsed_mtp = parse_case(c_mtp)
    draft_pol = DraftPolicy.for_request(parsed_mtp, parsed_mtp.workload.requests[0])
    assert draft_pol.mode == "mtp"
    assert draft_pol.proposal == 2
    assert draft_pol.mean == pytest.approx(1.0)

    c_dspark = compose_case(
        "kimi-k3", hardware="a3", tp=4, dp=1, ep=4, layers=2,
        mtp_proposal_tokens=2, mtp_acceptance_length=1.5,
    )
    assert c_dspark["runtime"]["metadata"]["draft_mode"] == "dspark"
    assert DraftPolicy.for_request(parse_case(c_dspark), parse_case(c_dspark).workload.requests[0]).mode == "dspark"

    c_sp = compose_case("kimi-k3", hardware="a3", tp=4, dp=1, ep=4, layers=2, sp=True)
    assert c_sp["runtime"]["kernels"]["sequence_parallel"]["padding"] == "right_to_tp"
    assert c_sp["runtime"]["metadata"]["sp"] is True
    assert predict(c_sp, scope="body", phase="decode", tokens=1, context=2048, sequences=1)["duration_ms"] > 0.0

    c_sp_qwen = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=4, dp=1, ep=4, layers=2, sp=True)
    assert c_sp_qwen["runtime"]["metadata"]["sp"] is True
    with pytest.raises(ModelingError, match="sequence_parallel"):
        compose_case("qwen3.5-35b-a3b", hardware="a3", tp=1, dp=1, ep=1, layers=2, sp=True)

    c_long = compose_case("qwen3-30b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=2, prompt_tokens=32768, output_tokens=128)
    assert c_long["runtime"]["max_model_len"] >= 32896
    assert parse_case(c_long).runtime.max_model_len >= 32896


def test_compose_case_validation_errors() -> None:
    with pytest.raises(ModelingError, match="invalid_ep_partition"):
        compose_case("qwen3.5-35b-a3b", hardware="a3", tp=4, dp=2, ep=3)
    with pytest.raises(ModelingError, match="insufficient_num_devices"):
        compose_case("qwen3.5-35b-a3b", hardware="a3", tp=4, dp=2, num_devices=4)
    with pytest.raises(ModelingError):
        compose_case("qwen3.5-35b-a3b", hardware="a3", n_routed_experts=8, num_experts_per_tok=16)
    with pytest.raises(ModelingError, match="invalid_mtp_acceptance_length"):
        compose_case("qwen3.5-35b-a3b", hardware="a3", mtp_proposal_tokens=2, mtp_acceptance_length=5.0)
    with pytest.raises(ModelingError, match="ep_requires_moe_model"):
        compose_case("glm-5.3-flash", hardware="a3", tp=4, dp=1, ep=4, layers=2)


def test_cli_compose_efficiency_and_sla_end_to_end(tmp_path: Path) -> None:
    case_path = tmp_path / "composed_case.json"
    eff_base_path = tmp_path / "eff_baseline.json"
    eff_fit_path = tmp_path / "eff_fit.json"
    sla_default_path = tmp_path / "sla_default.json"
    sla_scaled_path = tmp_path / "sla_scaled.json"

    rc_compose = main([
        "compose", "--model", "qwen3.5-35b-a3b", "--hardware", "a3",
        "--tp", "2", "--dp", "1", "--ep", "2", "--layers", "2",
        "--experts", "32", "--top-k", "4", "--prompt-tokens", "2048",
        "--output-tokens", "64", "--concurrency", "2", "-o", str(case_path),
    ])
    assert rc_compose == 0
    composed = json.loads(case_path.read_text(encoding="utf-8"))
    assert composed["machine"]["sku"] == "Ascend910_9392"
    assert composed["model"]["layers"] == 2
    assert composed["model"]["n_routed_experts"] == 32
    assert composed["model"]["num_experts_per_tok"] == 4

    rc_eff_base = main([
        "efficiency", str(case_path), "--phase", "decode", "--tokens", "1",
        "--context", "2048", "--sequences", "2", "-o", str(eff_base_path),
    ])
    assert rc_eff_base == 0
    eff_base = json.loads(eff_base_path.read_text(encoding="utf-8"))
    assert eff_base["schema"] == "mindie-sim.efficiency/1"
    assert eff_base["operator_sum_ms"] > 0.0
    bound_sum = sum(b["total_ms"] for b in eff_base["breakdown_by_bound"].values())
    fam_rows = (
        eff_base["breakdown_by_family"].values()
        if isinstance(eff_base["breakdown_by_family"], dict)
        else eff_base["breakdown_by_family"]
    )
    family_sum = sum(f["total_ms"] for f in fam_rows)
    assert bound_sum == pytest.approx(eff_base["operator_sum_ms"], rel=1e-6)
    assert family_sum == pytest.approx(eff_base["operator_sum_ms"], rel=1e-6)
    bound_share = sum(b["share_of_operator_sum"] for b in eff_base["breakdown_by_bound"].values())
    assert bound_share == pytest.approx(1.0, rel=1e-6)

    rc_eff_fit = main([
        "efficiency", str(case_path), "--phase", "decode", "--tokens", "1",
        "--context", "2048", "--sequences", "2", "--fit-targets",
        "--target-mbu", "0.80", "-o", str(eff_fit_path),
    ])
    assert rc_eff_fit == 0
    eff_fit = json.loads(eff_fit_path.read_text(encoding="utf-8"))
    assert eff_fit["schema"] == "mindie-sim.efficiency/1"
    assert eff_fit["targets"]["target_mbu"] == pytest.approx(0.80)
    assert eff_fit["target_step_duration_ms"] <= eff_fit["current_step_duration_ms"] + 1e-9
    assert "fitted_roofline_regimes" in eff_fit
    assert len(eff_fit["fitted_roofline_regimes"]) > 0

    rc_sla_default = main([
        "sla", str(case_path), "--sla-ms", "50.0", "--context", "4096",
        "--max-sequences-per-dp", "4", "-o", str(sla_default_path),
    ])
    assert rc_sla_default == 0
    sla_default = json.loads(sla_default_path.read_text(encoding="utf-8"))
    assert sla_default["schema"] == "mindie-sim.hardware-sla/1"
    assert len(sla_default["scenarios"]) >= 4
    assert sla_default["scenarios"][0].get("scenario_id", sla_default["scenarios"][0].get("id")) == "baseline"

    rc_sla_scaled = main([
        "sla", str(case_path), "--sla-ms", "50.0", "--context", "4096",
        "--max-sequences-per-dp", "4", "--hbm-bw-scale", "2.0", "-o", str(sla_scaled_path),
    ])
    assert rc_sla_scaled == 0
    sla_scaled = json.loads(sla_scaled_path.read_text(encoding="utf-8"))
    assert sla_scaled["schema"] == "mindie-sim.hardware-sla/1"
    scen_ids = [s.get("scenario_id", s.get("id")) for s in sla_scaled["scenarios"]]
    assert scen_ids == ["baseline", "scaled"]
    base_scen, scaled_scen = sla_scaled["scenarios"][0], sla_scaled["scenarios"][1]
    assert scaled_scen["scales"]["hbm_bw_scale"] == pytest.approx(2.0)
    base_b1 = base_scen.get("step_ms_at_b1", base_scen.get("b1_step_ms"))
    scaled_b1 = scaled_scen.get("step_ms_at_b1", scaled_scen.get("b1_step_ms"))
    assert scaled_b1 <= base_b1 + 1e-9
    base_b = base_scen.get("b_sla", base_scen.get("max_sequences_per_dp"))
    scaled_b = scaled_scen.get("b_sla", scaled_scen.get("max_sequences_per_dp"))
    assert scaled_b >= base_b


def test_cli_sweep_and_skew_modes(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_dict = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=2, ep=4, layers=2)
    case_path.write_text(json.dumps(case_dict), encoding="utf-8")

    topo_out = tmp_path / "sweep_topo.json"
    assert main([
        "sweep", str(case_path), "--mode", "topology",
        "--num-devices", "4", "--context", "2048", "--sla-ms", "50.0", "-o", str(topo_out),
    ]) == 0
    assert json.loads(topo_out.read_text(encoding="utf-8"))["schema"] == "mindie-sim.topology-sweep/1"

    pd_out = tmp_path / "sweep_pd.json"
    assert main([
        "sweep", str(case_path), "--mode", "pd", "--num-devices", "8",
        "--prompt-tokens", "1024", "--output-tokens", "128", "-o", str(pd_out),
    ]) == 0
    assert "pd" in json.loads(pd_out.read_text(encoding="utf-8"))["schema"]

    dp_out = tmp_path / "skew_dp.json"
    assert main([
        "skew", str(case_path), "--mode", "dp-barrier", "--dp-contexts", "1024", "4096",
        "--dp-sequences", "2", "4", "-o", str(dp_out),
    ]) == 0
    assert json.loads(dp_out.read_text(encoding="utf-8"))["schema"] == "mindie-sim.multi-dp-skew/1"

    moe_out = tmp_path / "skew_moe.json"
    assert main([
        "skew", str(case_path), "--mode", "moe-imbalance", "--imbalance-factor", "2.0",
        "--redundant-experts-per-rank", "1", "-o", str(moe_out),
    ]) == 0
    assert json.loads(moe_out.read_text(encoding="utf-8"))["schema"] == "mindie-sim.moe-imbalance/1"

    cp_out = tmp_path / "skew_cp.json"
    assert main([
        "skew", str(case_path), "--mode", "chunked-prefill", "--chunk-tokens", "512", "1024",
        "--context", "2048", "--sequences", "2", "-o", str(cp_out),
    ]) == 0
    assert json.loads(cp_out.read_text(encoding="utf-8"))["schema"] == "mindie-sim.chunked-prefill-interference/1"


def test_a5_dt_glm53_moe_dispatch_combine_modeling() -> None:
    from mindie_sim.prediction.efficiency import analyze_network_efficiency

    for hw, exp_disp_ideal_us, exp_comb_ideal_us, hop_us, slow_peers in (
        ("a5-pod-dt", 8.3649, 16.2229, 1.5, 63),
        ("a5-server-dt", 35.5509, 68.9472, 5.0, 56),
    ):
        case = compose_case(
            model="glm5.3", hardware=hw, dp=64, tp=1, ep=64,
            prompt_tokens=131072, output_tokens=1024, concurrency=1216,
            mtp_proposal_tokens=5, mtp_acceptance_length=4.5, tpot_slo_ms=20.0,
        )
        eff = analyze_network_efficiency(
            case, phase="decode", tokens=6, context=131072,
            sequences=19, scope="body", run_graph=False,
        )
        by_recipe = {
            row["recipe"]: row for row in eff["operators"]
            if row.get("recipe") in ("expert_dispatch", "expert_combine")
        }
        disp = by_recipe["expert_dispatch"]
        comb = by_recipe["expert_combine"]
        # 114 tokens/rank * top_k=8 * (63/64) = 897.75 remote rows/rank.
        # Dispatch uses FP8 payload (6144 B) + FP32 block-128 scales (48*4=192 B) = 6336 B/row.
        # Combine uses BF16 (6144 * 2 = 12288 B/row). Neither is double-counted by per_port Tx+Rx.
        assert disp["link_bytes"] == pytest.approx(897.75 * 6336.0)
        assert comb["link_bytes"] == pytest.approx(897.75 * 12288.0)
        assert disp["t_link_ideal_ms"] * 1000.0 == pytest.approx(exp_disp_ideal_us, rel=1e-3)
        assert comb["t_link_ideal_ms"] * 1000.0 == pytest.approx(exp_comb_ideal_us, rel=1e-3)
        # The generic envelope declares EP-1 propagation rounds. Removing
        # those dependencies requires an explicit direct-peer/HCCL program.
        # On eight 8-rank servers only 56 peers use the slow inter-node port;
        # its service excludes the seven local destinations.
        assert disp["duration_ms"] * 1000.0 == pytest.approx(
            exp_disp_ideal_us * slow_peers / 63 + 63 * hop_us, rel=1e-3)
        assert comb["duration_ms"] * 1000.0 == pytest.approx(
            exp_comb_ideal_us * slow_peers / 63 + 63 * hop_us, rel=1e-3)
        op_ids = [row["op_id"] for row in eff["operators"]]
        assert any(oid.endswith(".pack.quantize") for oid in op_ids)
        assert not any(oid.endswith(".expert_up.quantize") for oid in op_ids)


def test_va_recommended_compose_and_apply_helper() -> None:
    from mindie_sim import target_graphs
    from mindie_sim.compose import apply_va_recommended, va_recommended_config

    cfg_ds_a3 = va_recommended_config("deepseek-v3.2", "a3")
    assert cfg_ds_a3["quant"] == "w8a8"
    assert cfg_ds_a3["graph_capture_mode"] == "full"
    assert cfg_ds_a3["sp"] is True
    assert cfg_ds_a3["env"]["VLLM_ASCEND_ENABLE_MLAPO"] == "1"

    cfg_kimi_a5 = va_recommended_config("kimi-k3", "a5")
    assert cfg_kimi_a5["quant"] is None

    ds_case = compose_case("deepseek-v3.2", hardware="a3", tp=4, dp=2, ep=8, layers=4, va_recommended=True)
    assert ds_case["runtime"]["graph_capture_mode"] == "full"
    assert ds_case["runtime"]["metadata"]["va_recommended"] is True
    assert ds_case["runtime"]["metadata"]["va_env"]["VLLM_ASCEND_ENABLE_MLAPO"] == "1"
    ds_low = ds_case["runtime"]["kernels"]["lowerings"]
    assert ds_low.get("mla_prolog") == "vllm_ascend_mla_prolog_v3"
    assert ds_low.get("add_rmsnorm") == "vllm_ascend_native_norm"
    assert target_graphs(ds_case, phase="decode", tokens=1, context=1024, sequences=4)[0]

    kimi_case = compose_case("kimi-k3", hardware="a5", tp=4, dp=2, ep=8, layers=8, va_recommended=True)
    assert kimi_case["model"]["quant"] == compose_case(
        "kimi-k3", hardware="a5", tp=4, dp=2, ep=8, layers=8)["model"]["quant"]
    assert kimi_case["runtime"]["metadata"]["sp"] is True
    assert "sequence_parallel" in kimi_case["runtime"]["kernels"]
    kimi_low = kimi_case["runtime"]["kernels"]["lowerings"]
    assert kimi_low["kda"]["cases"][0]["binding"] == "vllm_ascend_recurrent_kda_arch35"
    assert kimi_low.get("attention_residual") == "vllm_ascend_attn_res_prefix"
    assert target_graphs(kimi_case, phase="decode", tokens=1, context=1024, sequences=4)[0]

    raw_glm = compose_case("glm-5.3", hardware="a3", tp=4, dp=2, ep=8, layers=4)
    enriched_glm = apply_va_recommended(raw_glm)
    assert enriched_glm["runtime"]["graph_capture_mode"] == "full"
    assert enriched_glm["runtime"]["metadata"]["va_recommended"] is True
    assert enriched_glm["runtime"]["kernels"]["lowerings"].get("mla_prolog") == "vllm_ascend_mla_prolog_v3"
    assert target_graphs(enriched_glm, phase="decode", tokens=1, context=1024, sequences=4)[0]


def _shared_expert_gemms(ops):
    return [op for op in ops if ".shared" in op.op_id
            and op.family in ("gate_up_projection", "down_projection")]


def _tp_collectives(ops, layer):
    return [op for op in ops if op.layer == layer
            and op.family in ("allreduce", "reduce_scatter", "allgather")
            and dict(op.metadata).get("group_axis") == "tp"]


def _causal_pairs(op):
    metadata = dict(op.metadata)
    return metadata.get("local_pairs", metadata.get("logical_pairs"))


def _assert_expert_exchange_links(ops, ep):
    exchanges = [op for op in ops
                 if op.recipe in ("expert_dispatch", "expert_combine")]
    assert exchanges
    for op in exchanges:
        metadata = dict(op.metadata)
        assert metadata["steps"] == ep - 1
        expected = (op.local_shape[0] * (ep - 1) / ep * metadata["row_bytes"]
                    / metadata["source_assignment_partitions"])
        assert op.link_bytes == pytest.approx(expected, rel=1e-12)
    return exchanges


def _native_mc2(case):
    case = deepcopy(case)
    case["runtime"].setdefault("kernels", {})["native_moe_protocol"] = {
        "kind": "arch35_fullmesh", "max_inflight_per_rank": 4,
    }
    return case


def test_feature_superposition_pp_pcp_mc2_dspark_mtp_sp() -> None:
    tokens = 128

    # Shared-expert DP: one TP rank handles 1/TP of the SP-sharded token rows
    # and drops the shared-expert TP collective; the 16-rank cohort performs
    # exactly the same shared-expert FLOPs as the TP-sharded baseline.
    shared_base = dict(
        model="deepseek-r1", hardware="a3", tp=4, dp=4, ep=16, layers=4,
        prompt_tokens=tokens, output_tokens=8, quant="w8a8",
        va_recommended=True,
    )
    shared_on = compose_case(**shared_base, enable_shared_expert_dp=True)
    shared_off = compose_case(**shared_base, enable_shared_expert_dp=False)
    graphs_on = target_graphs(shared_on, phase="prefill", tokens=tokens, scope="body")
    graphs_off = target_graphs(shared_off, phase="prefill", tokens=tokens, scope="body")
    assert len(graphs_on) == len(graphs_off) == shared_base["dp"]
    on_shared = _shared_expert_gemms(graphs_on[0])
    off_shared = _shared_expert_gemms(graphs_off[0])
    assert [op.family for op in on_shared] == ["gate_up_projection", "down_projection"]
    assert [op.family for op in off_shared] == ["gate_up_projection", "down_projection"]
    tp = shared_base["tp"]
    for on_op, off_op in zip(on_shared, off_shared):
        assert on_op.local_shape[0] * tp == off_op.local_shape[0]
        assert on_op.flops == pytest.approx(off_op.flops, rel=1e-12)
    assert on_shared[0].local_shape[1] == off_shared[0].local_shape[1] * tp
    assert on_shared[1].local_shape[2] == off_shared[1].local_shape[2] * tp
    on_cohort_flops = sum(op.flops for graph in graphs_on.values()
                          for op in _shared_expert_gemms(graph)) * tp
    off_cohort_flops = sum(op.flops for graph in graphs_off.values()
                           for op in _shared_expert_gemms(graph)) * tp
    assert on_cohort_flops == pytest.approx(off_cohort_flops, rel=1e-12)
    assert not any(op.link_bytes for graph in graphs_on.values()
                   for op in graph if ".shared" in op.op_id)
    off_shared_links = [op for graph in graphs_off.values() for op in graph
                        if ".shared" in op.op_id and op.link_bytes]
    assert len(off_shared_links) == 2 * len(graphs_off)

    eff_on = analyze_network_efficiency(
        shared_on, phase="prefill", tokens=tokens, context=0, sequences=1,
        scope="body", run_graph=False)
    eff_off = analyze_network_efficiency(
        shared_off, phase="prefill", tokens=tokens, context=0, sequences=1,
        scope="body", run_graph=False)
    assert eff_on["total_flops"] == pytest.approx(eff_off["total_flops"], rel=1e-12)
    hidden = shared_on["model"]["hidden_size"]
    shared_wire = 2 * tokens * hidden * 2 * (tp - 1) / tp
    assert eff_off["total_link_bytes"] - eff_on["total_link_bytes"] == pytest.approx(
        shared_wire, rel=1e-12)
    assert eff_on["operator_sum_ms"] < eff_off["operator_sum_ms"]

    # With SP disabled the baseline pays an explicit shared-expert TP allreduce;
    # shared-expert DP removes that op entirely.
    tp_on = compose_case(**{**shared_base, "va_recommended": False, "sp": False},
                         enable_shared_expert_dp=True)
    tp_off = compose_case(**{**shared_base, "va_recommended": False, "sp": False},
                          enable_shared_expert_dp=False)
    tp_on_ops = target_operators(tp_on, phase="prefill", tokens=tokens, scope="body")
    tp_off_ops = target_operators(tp_off, phase="prefill", tokens=tokens, scope="body")
    assert not any(op.family == "allreduce" and ".shared" in op.op_id
                   for op in tp_on_ops)
    shared_allreduce = [op for op in tp_off_ops
                        if op.family == "allreduce" and ".shared" in op.op_id]
    assert len(shared_allreduce) == 1
    assert shared_allreduce[0].link_bytes == pytest.approx(
        2 * (tp - 1) / tp * tokens * hidden * 2, rel=1e-12)

    # PCP x EP: every stage owns tokens/PCP query rows; the causal-pair split
    # sums to the single-stage triangular total.  Expert exchange wire scales
    # with the full EP domain and the exact remote-peer fraction.
    def pcp_case(model, *, pcp, ep, **kwargs):
        return compose_case(model, "a3", layers=4, prompt_tokens=tokens,
                            output_tokens=8, pcp=pcp, ep=ep, **kwargs)

    glm51 = pcp_case("glm-5.1", tp=1, dp=1, pcp=8, ep=8, va_recommended=True)
    glm51_base = pcp_case("glm-5.1", tp=1, dp=1, pcp=1, ep=1, va_recommended=True)
    glm51_ops = target_operators(glm51, phase="prefill", tokens=tokens, scope="body")
    glm51_base_ops = target_operators(glm51_base, phase="prefill", tokens=tokens,
                                      scope="body")
    attn51 = [op for op in glm51_ops if op.family == "attention" and op.layer == 0][0]
    attn51_base = [op for op in glm51_base_ops
                   if op.family == "attention" and op.layer == 0][0]
    assert attn51.local_shape[0] == tokens // 8 == 16
    tri_total = tokens * (tokens + 1) // 2
    assert _causal_pairs(attn51_base) == tri_total
    assert 8 * _causal_pairs(attn51) == tri_total
    assert 8 * attn51.flops == pytest.approx(attn51_base.flops, rel=1e-12)
    parallel51 = glm51["runtime"]["parallel"]
    assert parallel51["ep"] == parallel51["tp"] * parallel51["dp"] * parallel51["pcp"]
    _assert_expert_exchange_links(glm51_ops, parallel51["ep"])

    glm53_sp = pcp_case("glm-5.3", tp=2, dp=1, pcp=8, ep=16, sp=True)
    glm53_no = pcp_case("glm-5.3", tp=2, dp=1, pcp=8, ep=16, sp=False)
    glm53_base = pcp_case("glm-5.3", tp=2, dp=1, pcp=1, ep=2, sp=True)
    ops_sp = target_operators(glm53_sp, phase="prefill", tokens=tokens, scope="body")
    ops_no = target_operators(glm53_no, phase="prefill", tokens=tokens, scope="body")
    ops_base = target_operators(glm53_base, phase="prefill", tokens=tokens, scope="body")
    attn_sp = [op for op in ops_sp if op.family == "attention" and op.layer == 0][0]
    attn_base = [op for op in ops_base if op.family == "attention" and op.layer == 0][0]
    assert 8 * _causal_pairs(attn_sp) == _causal_pairs(attn_base) == tri_total
    assert 8 * attn_sp.flops == pytest.approx(attn_base.flops, rel=1e-12)
    parallel53 = glm53_sp["runtime"]["parallel"]
    assert parallel53["ep"] == parallel53["tp"] * parallel53["dp"] * parallel53["pcp"]
    _assert_expert_exchange_links(ops_sp, parallel53["ep"])
    _assert_expert_exchange_links(ops_no, parallel53["ep"])

    tp53 = parallel53["tp"]
    sp_norm = [op for op in ops_sp if op.layer == 0 and op.recipe == "rmsnorm"
               and "norm1" in op.op_id][0]
    no_norm = [op for op in ops_no if op.layer == 0 and op.recipe == "rmsnorm"
               and "norm1" in op.op_id][0]
    sp_residual = [op for op in ops_sp if op.layer == 0
                   and op.recipe == "residual_add"][0]
    no_residual = [op for op in ops_no if op.layer == 0
                   and op.recipe == "residual_add"][0]
    assert sp_norm.local_shape[0] * tp53 == no_norm.local_shape[0]
    assert sp_norm.vector_ops * tp53 == pytest.approx(no_norm.vector_ops, rel=1e-12)
    assert sp_residual.vector_ops * tp53 == pytest.approx(no_residual.vector_ops,
                                                          rel=1e-12)
    for layer in range(4):
        sp_links = _tp_collectives(ops_sp, layer)
        no_links = _tp_collectives(ops_no, layer)
        assert {op.family for op in sp_links} == {"allgather", "reduce_scatter"}
        assert {op.family for op in no_links} == {"allreduce"}
        assert sum(op.link_bytes for op in sp_links) == pytest.approx(
            sum(op.link_bytes for op in no_links), rel=1e-12)
    eff_sp = analyze_network_efficiency(
        glm53_sp, phase="prefill", tokens=tokens, context=0, sequences=1,
        scope="body", run_graph=False)
    eff_no = analyze_network_efficiency(
        glm53_no, phase="prefill", tokens=tokens, context=0, sequences=1,
        scope="body", run_graph=False)
    assert eff_sp["operator_sum_ms"] < eff_no["operator_sum_ms"]

    # PP x DSpark, with AttnRes state crossing stages: layer FLOPs and HBM
    # bytes are conserved across pipeline stages, while every P2P send carries
    # exactly one activation/AttnRes bank row.
    pp_base = compose_case(
        "kimi-k3", "a5", tp=4, dp=1, pp=1, ep=4, layers=4, prompt_tokens=tokens,
        output_tokens=8, mtp_proposal_tokens=2, va_recommended=True)
    pp4 = compose_case(
        "kimi-k3", "a5", tp=4, dp=1, pp=4, ep=4, layers=4, prompt_tokens=tokens,
        output_tokens=8, mtp_proposal_tokens=2, va_recommended=True)
    pp1_ops = target_operators(pp_base, phase="prefill", tokens=tokens, scope="body")
    pp4_ops = target_operators(pp4, phase="prefill", tokens=tokens, scope="body")
    assert {op.stage for op in pp4_ops} == {0, 1, 2, 3}
    layer_flops = [sum(op.flops for op in ops if op.layer >= 0)
                   for ops in (pp1_ops, pp4_ops)]
    layer_hbm = [sum(op.hbm_bytes for op in ops if op.layer >= 0)
                 for ops in (pp1_ops, pp4_ops)]
    assert layer_flops[1] == pytest.approx(layer_flops[0], rel=1e-12)
    assert layer_hbm[1] == pytest.approx(layer_hbm[0], rel=1e-12)
    stage_flops = {stage: sum(op.flops for op in pp4_ops
                              if op.stage == stage and op.layer >= 0)
                   for stage in range(4)}
    assert all(value > 0 for value in stage_flops.values())
    assert max(stage_flops.values()) < layer_flops[1] / 2
    assert sum(stage_flops.values()) == pytest.approx(layer_flops[1], rel=1e-12)
    sends = [op for op in pp4_ops if op.family == "pp_p2p"
             and op.recipe == "pipeline_send"]
    residuals = [op for op in pp4_ops if op.family == "pp_p2p"
                 and op.recipe == "pipeline_residual_send"]
    hidden_kimi = pp4["model"]["hidden_size"]
    activation_bytes = (tokens // 4) * hidden_kimi * 2
    assert len(sends) == 3 and len(residuals) == 6
    assert all(op.link_bytes == pytest.approx(activation_bytes, rel=1e-12)
               for op in (*sends, *residuals))
    assert all(op.local_shape == (tokens // 4, hidden_kimi)
               for op in (*sends, *residuals))
    assert sum(op.link_bytes for op in sends) == pytest.approx(
        3 * activation_bytes, rel=1e-12)
    assert sum(op.link_bytes for op in residuals) == pytest.approx(
        6 * activation_bytes, rel=1e-12)

    # MTP proposal rows are target verification rows: 1 + proposal_tokens.
    mtp0 = compose_case(
        "kimi-k3", "a5", tp=4, dp=1, pp=4, pcp=2, ep=8, layers=4,
        prompt_tokens=tokens, output_tokens=8, mtp_proposal_tokens=0, sp=False)
    mtp2 = compose_case(
        "kimi-k3", "a5", tp=4, dp=1, pp=4, pcp=2, ep=8, layers=4,
        prompt_tokens=tokens, output_tokens=8, mtp_proposal_tokens=2, sp=False)
    assert dict(mtp2["runtime"]["metadata"])["draft_proposal_tokens"] == 2
    decode_rows = 1 + int(dict(mtp2["runtime"]["metadata"])["draft_proposal_tokens"])
    mtp0_ops = target_operators(mtp0, phase="decode", tokens=1, context=tokens,
                                scope="body")
    mtp2_ops = target_operators(mtp2, phase="decode", tokens=decode_rows,
                                context=tokens, scope="body")
    for recipe in ("expert_up", "expert_down"):
        base = [op for op in mtp0_ops if op.op_id.endswith("." + recipe)]
        scaled = [op for op in mtp2_ops if op.op_id.endswith("." + recipe)]
        assert len(base) == len(scaled) == 3
        for row, triple in zip(base, scaled):
            assert triple.local_shape[0] == 3 * row.local_shape[0]
            assert triple.flops == pytest.approx(3 * row.flops, rel=1e-12)
            assert triple.hbm_bytes > row.hbm_bytes
    attn0 = [op for op in mtp0_ops if op.family == "attention"][0]
    attn2 = [op for op in mtp2_ops if op.family == "attention"][0]
    assert attn2.local_shape[0] == 3 * attn0.local_shape[0]

    # W4A8 x MC2: routed-expert weights halve, dispatch/combine become fused
    # source MC2 phases, and the MoE weight-read floor is strictly lower.
    quant_base = dict(
        model="deepseek-v3.2", hardware="a3", tp=2, dp=4, pcp=2, ep=16,
        layers=4, prompt_tokens=tokens, output_tokens=8, va_recommended=True,
    )
    w8 = compose_case(**quant_base, quant="w8a8")
    w4 = compose_case(**quant_base, quant="w4a8")
    parallel = w4["runtime"]["parallel"]
    assert parallel["ep"] == parallel["tp"] * parallel["dp"] * parallel["pcp"]
    eff8 = analyze_network_efficiency(w8, phase="prefill", tokens=tokens,
                                      context=0, sequences=1, scope="body",
                                      run_graph=False)
    eff4 = analyze_network_efficiency(w4, phase="prefill", tokens=tokens,
                                      context=0, sequences=1, scope="body",
                                      run_graph=False)
    assert eff4["total_flops"] == pytest.approx(eff8["total_flops"], rel=1e-12)
    assert eff4["total_hbm_bytes"] < eff8["total_hbm_bytes"]
    assert eff4["operator_sum_ms"] < eff8["operator_sum_ms"]
    mc2_ops8 = target_operators(_native_mc2(w8), phase="prefill", tokens=tokens,
                                scope="body")
    mc2_ops4 = target_operators(_native_mc2(w4), phase="prefill", tokens=tokens,
                                scope="body")
    expert8 = {recipe: [op for op in mc2_ops8 if op.op_id.endswith("." + recipe)][0]
               for recipe in ("expert_up", "expert_down")}
    expert4 = {recipe: [op for op in mc2_ops4 if op.op_id.endswith("." + recipe)][0]
               for recipe in ("expert_up", "expert_down")}
    for recipe, op8 in expert8.items():
        op4 = expert4[recipe]
        assert dict(op8.metadata)["weight_format"] == "int8"
        assert dict(op4.metadata)["weight_format"] == "int4"
        weight8, weight4 = (dict(op.metadata)["weight_bytes"] for op in (op8, op4))
        assert weight4 == pytest.approx(0.5 * weight8, rel=1e-12)
    assert not any(op.recipe in ("expert_dispatch", "expert_combine")
                   for op in mc2_ops4)
    mc2 = [op for op in mc2_ops4 if op.family == "mc2_protocol"]
    mc2_kernels = {dict(op.metadata)["device_kernel_group"]["kernel"] for op in mc2}
    assert mc2_kernels == {"MoeDistributeDispatchV3", "MoeDistributeCombineV3"}
    mc2_groups = {dict(op.metadata)["device_kernel_group"]["id"] for op in mc2}
    assert {group.rsplit(".", 1)[-1] for group in mc2_groups} == {
        "mc2_dispatch", "mc2_combine"}
    hbm_peak = extract_hardware_peaks(w4)["hbm_peak_bytes_per_s"]
    moe_hbm8 = sum(dict(op.metadata)["weight_bytes"] for op in mc2_ops8
                   if op.op_id.endswith((".expert_up", ".expert_down")))
    moe_hbm4 = sum(dict(op.metadata)["weight_bytes"] for op in mc2_ops4
                   if op.op_id.endswith((".expert_up", ".expert_down")))
    moe_time8 = moe_hbm8 / hbm_peak * 1000.0
    moe_time4 = moe_hbm4 / hbm_peak * 1000.0
    assert moe_time4 == pytest.approx(0.5 * moe_time8, rel=1e-12)
    assert moe_time4 < moe_time8

    # The same W4A8 x DP/TP/PP/PCP/EP/SP stack on two other catalogs.
    for model in ("deepseek-v4-pro", "qwen3.5-397b-a17b"):
        case = compose_case(
            model, "a3", tp=4, dp=2, pp=2, pcp=2, ep=16, layers=4,
            prompt_tokens=tokens, output_tokens=8, sp=True,
            va_recommended=True, quant="w4a8")
        parallel = case["runtime"]["parallel"]
        assert (parallel["tp"], parallel["dp"], parallel["pp"], parallel["pcp"],
                parallel["ep"]) == (4, 2, 2, 2, 16)
        assert case["runtime"]["metadata"]["sp"] is True
        ops = target_operators(case, phase="prefill", tokens=tokens, scope="body")
        assert {op.stage for op in ops} == {0, 1}
        sends = [op for op in ops if op.family == "pp_p2p"
                 and op.recipe == "pipeline_send"]
        hidden = case["model"]["hidden_size"]
        assert len(sends) == 1
        sender_layer = int(sends[0].op_id.split(".l", 1)[1].split(".", 1)[0])
        hc_mult = max(1, parse_case(case).model.layer_plans[sender_layer].hc_mult)
        assert sends[0].link_bytes == pytest.approx(
            sends[0].local_shape[0] * hidden * 2 * hc_mult, rel=1e-12)
        for recipe in ("expert_up", "expert_down"):
            expert = [op for op in ops if op.op_id.endswith("." + recipe)]
            assert expert and all(dict(op.metadata)["weight_format"] == "int4"
                                  for op in expert)
            assert all(op.hbm_bytes > 0 for op in expert)
        pred = predict(case, scope="body", phase="decode", tokens=1,
                       context=tokens, sequences=1)
        assert pred["duration_ms"] == pytest.approx(
            pred["complete_ms"] - pred["input_ready_ms"], rel=1e-12)
        assert pred["duration_ms"] == pytest.approx(pred["body_forward_ms"],
                                                    rel=1e-12)

        dec_ops = target_operators(case, phase="decode", tokens=1,
                                   context=tokens, scope="body")
        dec_hbm_floor_ms = (
            sum(op.hbm_bytes for op in dec_ops)
            / extract_hardware_peaks(case)["hbm_peak_bytes_per_s"] * 1000.0
        )
        prefill = predict(case, scope="body", phase="prefill", tokens=tokens,
                          context=0, sequences=1)
        assert pred["duration_ms"] > dec_hbm_floor_ms > 0.0
        assert pred["duration_ms"] < prefill["duration_ms"]

    # PP1 vs PP2 AttnRes cross-layer bank preservation on Kimi-K3.
    c_pp1 = compose_case("kimi-k3", "a3", tp=8, pp=1, ep=8, layers=4,
                         va_recommended=True)
    c_pp2 = compose_case("kimi-k3", "a3", tp=8, pp=2, ep=8, layers=4,
                         va_recommended=True)
    l2_pp1 = [dict(op.metadata)["attn_res_view"]["valid_blocks"]
              for op in target_operators(c_pp1, phase="prefill", tokens=128,
                                         scope="model")
              if op.layer == 2 and "attn_res_view" in dict(op.metadata)]
    l2_pp2 = [dict(op.metadata)["attn_res_view"]["valid_blocks"]
              for op in target_operators(c_pp2, phase="prefill", tokens=128,
                                         scope="model")
              if op.layer == 2 and "attn_res_view" in dict(op.metadata)]
    assert l2_pp1 == l2_pp2 == [1, 1]

    # Short prefill PCP cohort ownership (1 token per replica under DP2/TP2/EP4/PCP4).
    c_short = compose_case("deepseek-v3.2", "a3", dp=2, tp=2, ep=4, pcp=4, layers=4)
    g_short = target_graphs(c_short, phase="prefill", tokens=1, sequences=1,
                            scope="body")
    exp_up = [op for op in g_short[0] if "expert_up" in op.op_id][0]
    assert exp_up.semantic_shape[0] == 4


def test_cross_dp_sp_body_selection_preserves_body_defaults() -> None:
    # A cross-DP SP operator view delegates graph binding to ``target_graphs``.
    # That delegation must consume the declaration already resolved by the
    # caller, not re-resolve it and relabel installed defaults as explicit.
    def qwen_sp():
        return compose_case("qwen3.5-35b-a3b", "a3", tp=2, dp=2, ep=4, sp=True,
                            layers=2, prompt_tokens=64, output_tokens=8)

    ops = target_operators(qwen_sp(), phase="prefill", tokens=128, scope="body")
    grouped = [item for item in ops.default_selection["selections"]
               if item["family"] == "grouped_gemm"]
    assert [(item["selection"], item["reason"]) for item in grouped] == [(
        "default_installed", "preserve_existing_grouped_mapping_domain")]
    assert grouped[0]["program"] == {"kind": "steady_matmul"}
    matmul = [item for item in ops.default_selection["selections"]
              if item["family"] == "matmul"]
    assert [(item["selection"], item["reason"]) for item in matmul] == [(
        "default_installed", "compact_finite_k_pipeline_preserving_128_output_grid")]
    assert {"programs", "families"} <= ops.effective_kernels.keys()
    assert ops.effective_kernels["families"]["grouped_gemm"] == {"kind": "steady_matmul"}

    disabled = target_operators(qwen_sp(), phase="prefill", tokens=128,
                                scope="body", apply_body_defaults=False)
    assert disabled.default_selection["reason"] == "body_defaults_disabled"
    assert "programs" not in disabled.effective_kernels

    explicit = qwen_sp()
    explicit["runtime"].setdefault("kernels", {})["families"] = {
        "grouped_gemm": {"kind": "steady_matmul", "explicit": True}}
    preserved = [item for item in target_operators(
        explicit, phase="prefill", tokens=128, scope="body").default_selection["selections"]
        if item["family"] == "grouped_gemm"]
    assert [(item["selection"], item["reason"]) for item in preserved] == [(
        "preserved", "explicit_implementation_preserved")]
    assert preserved[0]["program"] == {"kind": "steady_matmul", "explicit": True}
