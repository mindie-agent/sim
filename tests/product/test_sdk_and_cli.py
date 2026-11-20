"""Unified CLI routing, ``python -m mindie_sim`` parity, and typed Python SDK.

[Role] Verify ``mindie_sim.cli:main`` and ``python -m mindie_sim`` share one
  command route, legacy vs canonical disambiguation works without filesystem
  sniffing, ``TimeoutError`` propagates cleanly, and ``mindie_sim.api``
  three-tier evaluation runs causally on ``runtime.Engine``.
[Invariants] Legacy ``predict step`` (file literally named ``step``) and
  ``calibrate record samples.jsonl`` stay unambiguous. ``TimeoutError`` is
  never masked by ``except OSError:``. ``mindie_sim.__all__`` has zero
  duplicate exports.
[Timeouts & Locks] Fast Tier-1 subprocess + in-memory checks (<5s).
[Side-Effect Policy] Writes only inside pytest ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import mindie_sim
import mindie_sim.api as sdk_api
import mindie_sim.cli as cli_mod
from mindie_sim.cli_parser import resolve_route
from mindie_sim.compose import compose_case as compose_raw_case
from mindie_sim.prediction.results import DEFAULT_TARGET_EFFICIENCY, StepRequest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_entrypoint_parity_disambiguation_and_timeout_propagation(tmp_path, monkeypatch):
    raw_case = compose_raw_case(
        model="qwen3-32b",
        hardware="Ascend910B2",
        dp=1, tp=1, pp=1, layers=1,
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(raw_case), encoding="utf-8")

    out_direct = tmp_path / "step_direct.json"
    out_module = tmp_path / "step_module.json"
    rc_direct = cli_mod.main([
        "predict", "step", str(case_path),
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--output", str(out_direct),
    ])
    assert rc_direct == 0

    proc = subprocess.run(
        [
            sys.executable, "-m", "mindie_sim",
            "predict", "step", str(case_path),
            "--phase", "decode", "--tokens", "1", "--context", "64",
            "--output", str(out_module),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    direct_payload = json.loads(out_direct.read_text(encoding="utf-8"))
    module_payload = json.loads(out_module.read_text(encoding="utf-8"))
    assert direct_payload == module_payload
    assert direct_payload["span_ms"] > 0.0
    assert direct_payload["graph_id"]
    assert len(direct_payload["operators"]) > 0

    # Legacy file literally named "step" routes to legacy `predict` (not `predict-step`).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "step").write_text("{}", encoding="utf-8")
    route_legacy, rest_legacy = resolve_route(["predict", "step"])
    assert route_legacy.command == "predict"
    assert rest_legacy == ("step",)

    route_canon, rest_canon = resolve_route(["predict", "step", str(case_path)])
    assert route_canon.command == "predict-step"
    assert rest_canon == (str(case_path),)

    # Legacy calibrate <store="record"> <samples> vs canonical calibrate record <store> <samples>.
    cal_legacy, cal_legacy_rest = resolve_route(["calibrate", "record", "samples.jsonl"])
    assert cal_legacy.command == "calibrate"
    assert cal_legacy.path == ("calibrate",)
    assert cal_legacy_rest == ("record", "samples.jsonl")

    cal_canon, cal_canon_rest = resolve_route(["calibrate", "record", "store.sqlite", "samples.jsonl"])
    assert cal_canon.command == "calibrate"
    assert cal_canon.path == ("calibrate", "record")
    assert cal_canon_rest == ("store.sqlite", "samples.jsonl")

    # Canonical three-tier CLI command.
    out_tiers = tmp_path / "tiers.json"
    rc_tiers = cli_mod.main([
        "predict", "tiers", str(case_path),
        "--phase", "decode", "--tokens", "1", "--context", "64",
        "--target-eff", "0.80", "--output", str(out_tiers),
    ])
    assert rc_tiers == 0
    tiers_payload = json.loads(out_tiers.read_text(encoding="utf-8"))
    assert tiers_payload["target_efficiency"] == pytest.approx(0.80)
    assert tiers_payload["should_achieve"]["span_ms"] > tiers_payload["mainline"]["span_ms"]
    assert tiers_payload["aligned_table"]["graph_id"] == tiers_payload["mainline"]["graph_id"]

    # TimeoutError must propagate directly and never be masked by except OSError.
    def _raise_timeout(_args):
        raise TimeoutError("simulated_cli_timeout")

    monkeypatch.setattr(cli_mod, "run", _raise_timeout)
    with pytest.raises(TimeoutError, match="simulated_cli_timeout"):
        cli_mod._main(["predict", "step", str(case_path)])


@pytest.mark.parametrize("sku", ("Ascend910B2", "Ascend910_9382", "Ascend950DT_9581"))
def test_typed_sdk_three_tier_evaluation_and_exports(sku):
    case = sdk_api.compose_case(
        "qwen3-32b", hardware=sku, options={"dp": 1, "tp": 1, "pp": 1, "layers": 1}
    )
    req = StepRequest(phase="decode", tokens=1, context=64, sequences=1, scope="body")
    tiers = sdk_api.evaluate_three_tiers(case, req, target_eff=DEFAULT_TARGET_EFFICIENCY)
    assert tiers.unmerge.span_ms > 0.0
    assert tiers.mainline.span_ms > 0.0
    assert tiers.should_achieve.span_ms > tiers.mainline.span_ms
    assert tiers.aligned_table is not None
    assert tiers.aligned_table.graph_id == tiers.mainline.graph_id
    assert len(tiers.aligned_table.tier_spans_ms) == 3
    assert len(mindie_sim.__all__) == len(set(mindie_sim.__all__))
