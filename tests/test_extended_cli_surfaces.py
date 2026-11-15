"""Fast end-to-end verification of extended CLI subcommands and compose --recover-ssm."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mindie_sim.cli import EXTENDED_COMMANDS, main as cli_main


def _invoke_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    rc = cli_main(argv)
    captured = capsys.readouterr()
    assert rc == 0, f"CLI failed ({rc}) for {argv}: {captured.err}"
    payload = json.loads(captured.out)
    assert isinstance(payload, dict)
    return payload


def test_extended_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    for subcommand in ("compose", *EXTENDED_COMMANDS):
        with pytest.raises(SystemExit) as exc_info:
            cli_main([subcommand, "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert subcommand in out


def test_extended_cli_json_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    t0 = time.perf_counter()

    # 1. compose --recover-ssm on a hybrid linear-attention catalog model
    hybrid_case = tmp_path / "hybrid_case.json"
    rc = cli_main([
        "compose",
        "--model", "qwen3.5-35b-a3b",
        "--hardware", "a3",
        "--tp", "2",
        "--layers", "2",
        "--prompt-tokens", "128",
        "--output-tokens", "16",
        "--recover-ssm",
        "-o", str(hybrid_case),
    ])
    assert rc == 0
    capsys.readouterr()
    hybrid_data = json.loads(hybrid_case.read_text(encoding="utf-8"))
    assert hybrid_data["runtime"]["recover_ssm"] is True

    # 2. kv-state on the synthesized hybrid case
    kv_report = _invoke_json(capsys, ["kv-state", str(hybrid_case)])
    assert "bytes_per_token" in kv_report or "layers" in kv_report or "state_bytes" in json.dumps(kv_report)

    # 3. memory-breakdown on the synthesized hybrid case
    mem_report = _invoke_json(capsys, ["memory-breakdown", str(hybrid_case)])
    assert mem_report.get("peak_hbm_bytes", 0) > 0

    # 4. optimize-graph on a 1-layer case
    dense_case = tmp_path / "dense_1l.json"
    assert cli_main([
        "compose",
        "--model", "qwen3-32b",
        "--hardware", "a3",
        "--tp", "4",
        "--layers", "1",
        "--prompt-tokens", "64",
        "--output-tokens", "8",
        "-o", str(dense_case),
    ]) == 0
    capsys.readouterr()
    opt_report = _invoke_json(capsys, [
        "optimize-graph", str(dense_case),
        "--phase", "decode",
        "--tokens", "1",
        "--sequences", "1",
        "--context", "64",
        "--weight-prefetch",
    ])
    assert opt_report["baseline_ms"] > 0
    assert opt_report["final_ms"] <= opt_report["baseline_ms"] + 1e-9

    # 5. sweep-ep-knee on a compact 1-layer MoE case
    moe_case = tmp_path / "moe_1l.json"
    assert cli_main([
        "compose",
        "--model", "qwen3-30b-a3b",
        "--hardware", "800i-a3-16p",
        "--num-devices", "16",
        "--tp", "2",
        "--ep", "2",
        "--layers", "1",
        "--prompt-tokens", "64",
        "--output-tokens", "8",
        "-o", str(moe_case),
    ]) == 0
    capsys.readouterr()
    knee_report = _invoke_json(capsys, [
        "sweep-ep-knee", str(moe_case),
        "--ep-candidates", "2", "4",
        "--batch-sequences", "2",
        "--context-tokens", "64",
        "--hccl-buffsize-mb", "128",
    ])
    assert len(knee_report.get("rows", ())) == 2

    # 6. size-xpyd on the compact 1-layer MoE case
    xpyd_report = _invoke_json(capsys, [
        "size-xpyd", str(moe_case),
        "--total-npus", "4",
        "--isl", "64",
        "--osl", "8",
        "--ttft-slo-ms", "5000",
        "--tpot-slo-ms", "100",
        "--prefix-hit-rate", "0.25",
        "--allow-unqualified",
    ])
    assert "candidates" in xpyd_report or "recommended" in xpyd_report or "feasible" in json.dumps(xpyd_report)

    # 7. opflow bounded MCP tool dispatch
    flow_cat = _invoke_json(capsys, ["opflow", "opflow_catalog"])
    assert flow_cat["ok"] == 1
    assert flow_cat["count"] > 0
    flow_hw = _invoke_json(capsys, [
        "opflow", "opflow_hardware",
        "--args-json", '{"name": "Ascend950PR"}',
    ])
    assert flow_hw["ok"] == 1
    assert "950" in flow_hw["sku"]

    # 8. op-formula closed-form evaluation + diagnostics
    formula_report = _invoke_json(capsys, [
        "op-formula", "rmsnorm",
        "--params-json", '{"M": 16, "N": 4096, "dtype": "bf16"}',
        "--hardware", "a5-pod-dt",
    ])
    assert formula_report["operator"] == "rmsnorm"
    assert formula_report["latency"]["t_floor_us"] > 0
    assert formula_report["diagnostics"]["status"] == "unmeasured"

    assert time.perf_counter() - t0 < 5.0


def test_fast_eval_non_integer_fraction_operands_parity() -> None:
    from fractions import Fraction
    from mindie_sim.prediction.piecewise.expressions import (
        Add, CeilDiv, Const, Div, FloorDiv, Max, Min, Mul, Neg, Sub, Sum, Var,
    )

    env = {"a": Fraction(3, 2), "b": Fraction(1, 2), "c": 2, "neg": Fraction(-5, 3)}
    cases = (
        Sub(Const(Fraction(3, 2)), Const(Fraction(1, 2))),
        Sub(Var("a"), Var("b")),
        Sub(Var("a"), Var("c")),
        Sub(Var("c"), Var("a")),
        Sub(Var("neg"), Var("a")),
        Add((Var("a"), Var("b"), Var("c"))),
        Sum((Var("a"), Var("neg"))),
        Mul((Var("a"), Var("b"))),
        Mul((Var("a"), Var("c"))),
        Div(Var("a"), Var("b")),
        FloorDiv(Var("a"), Var("b")),
        FloorDiv(Var("neg"), Var("b")),
        CeilDiv(Var("a"), Var("b")),
        CeilDiv(Var("neg"), Var("b")),
        Min((Var("a"), Var("b"), Var("neg"))),
        Max((Var("a"), Var("b"), Var("neg"))),
        Neg(Var("a")),
    )
    for expr in cases:
        assert expr.evaluate(env) == expr._slow_evaluate(env)

