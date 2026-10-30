"""A conserved program on the wrong source branch must fail campaign review."""
from types import SimpleNamespace

from tools.run_operator_cases import run
from test_mechanism_contract import operator, two_products


def test_finite_conserved_work_does_not_hide_a_missing_real_dispatch():
    module = SimpleNamespace(make_op=lambda params: operator(), build=two_products)
    result = run(module, [{
        "id": "real-call", "split": "development", "params": {},
        "expected_dispatch": {"state_v_first": True, "use_exp2": True},
    }])
    assert result["status"] == "failed"
    for row in result["cases"]:
        assert row["finite_completion"] and not row["work_mismatches"]
        assert row["dispatch_mismatches"] == {
            "state_v_first": {"expected": True, "selected": None},
            "use_exp2": {"expected": True, "selected": None},
        }
