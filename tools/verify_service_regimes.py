"""Installed CLI control for observable native-call service responses.

[Role] Verify parameter application through the installed Body prediction path.
[Invariants] Synthetic response affects completion, never physical work; rate
  changes apply to the declared service term, and each norm is counted once.
[Timeouts & Locks] Reuses the wheel verifier's bounded subprocess runner.
[Side-Effect Policy] Writes only cases in the verifier's temporary directory.
"""
from copy import deepcopy
import json


def verify_service_regimes(run, root, work):
    base = json.loads((root / "examples/quickstart.json").read_text())
    base["model"]["layers"] = 1
    path = work / "observable-service-control.json"
    path.write_text(json.dumps(base) + "\n")
    args = ["predict", path, "--scope", "body", "--phase", "decode",
            "--tokens", 1, "--context", 16, "--sequences", 1]
    analytic = run(args, "body-observable-service-analytic")
    spans = []
    for factor in (1, 2):
        case = deepcopy(base)
        case["machine"]["device"]["rates"]["vector"] *= factor
        case["runtime"]["kernels"]["service_regimes"] = [{
            "id": "installed-normalization-response",
            "match": {"family": "rmsnorm", "local_shape": [1, 512]},
            "coordinates": {"width": {"path": "local_shape.1"}},
            "response": {"terms": [{"coordinate": "width",
                "us_per_unit": 1000 / 512,
                "scale_with": {"resource": "vector", "reference_rate": 5e12}}]},
            "provenance": {"basis": "synthetic_service_boundary_control"},
        }]
        path.write_text(json.dumps(case) + "\n")
        result = run(args, f"body-observable-service-vector-{factor}")
        spans.append(result["body_forward_ms"])
        selected = result["parameters"]["service_regimes"]
        assert len(selected) == 1
        assert selected[0]["queries"] == {"observable_response": 2}
        assert result["parameters"]["regime_selection"]["unmatched"] == []
        assert result["execution"]["work"] == analytic["execution"]["work"]
        assert result["execution"]["serviced"] == analytic["execution"]["serviced"]
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert result["execution"]["submission"]["host_service_ms"] == 0
    assert spans[0] > spans[1] > analytic["body_forward_ms"]
    # Two sequential norms each shorten from 1 ms to 0.5 ms. Other vector
    # work may also shorten; this bound does not pretend it stays constant.
    assert spans[0] - spans[1] >= .999
