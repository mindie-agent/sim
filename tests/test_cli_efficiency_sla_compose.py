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

import json
from pathlib import Path
import pytest

from mindie_sim import predict
from mindie_sim.__main__ import main
from mindie_sim.compose import HARDWARE_ALIASES, compose_case, resolve_hardware_sku
from mindie_sim.prediction.graph import GraphPolicy
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

    with pytest.raises(ModelingError, match="sequence_parallel"):
        compose_case("qwen3.5-35b-a3b", hardware="a3", tp=4, dp=1, ep=4, layers=2, sp=True)

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
