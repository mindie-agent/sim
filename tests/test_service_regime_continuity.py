"""Static coverage and continuity contracts for service response domains.

[Role] Verify geometric coverage, continuity, knot support, and exterior policies.
[Invariants] Rule order never changes selection; gaps and overlaps are rejected.
[Timeouts & Locks] Tier 0 (<5s), no external locks.
[Side-Effect Policy] Pure read-only in-memory unit tests.
"""
from copy import deepcopy
from dataclasses import replace
import math

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.response_functions import (
    ResponseFunction, nonnegative, positive, scaled_coordinate)
from mindie_sim.prediction.service_regime_domains import (
    compile_service_regime_domains, rules_overlap, validate_service_regime_domains)
from mindie_sim.schema.common import ModelingError
from test_mechanism_contract import hardware_case
from test_resource_services import copy_op

SUPPORT = {"q": {"min": 1, "max": 200}}
WINDOW = {"q": {"min": 1, "max": 200}, "kv_len": {"min": 1, "max": 200}}
WINDOW_COORDINATES = {"q": {"path": "semantic_shape.0"},
                      "kv_len": {"path": "semantic_shape.1"}}


def affine(fixed_us=8000.0):
    return {"kind": "affine", "fixed_us": fixed_us}


def curve(outside="analytic", knots=((1, 8000.0), (200, 10000.0))):
    return {"kind": "piecewise_linear", "coordinate": "q", "outside": outside,
            "knots": [list(pair) for pair in knots]}


def rule(rid, low, high, *, group="copy-domain", support=None, response=None,
         outside="analytic", coordinates=None, extra_domain=None, execution=None,
         match=None):
    response = response or affine()
    coordinates = coordinates or {"q": {"path": "semantic_shape.0"}}
    support = support or SUPPORT
    domain = {"q": {"min": low, "max": high}}
    domain.update(extra_domain or {})
    raw = {"id": rid, "match": match or {"recipe": "copy"}, "coordinates": coordinates,
           "domain": domain, "response": response,
           "continuity": {"group": group, "support": support, "outside": outside}}
    if execution is not None:
        raw["execution"] = execution
    return raw


def rect(rid, qlo, qhi, klo, khi, **changes):
    support = changes.pop("support", WINDOW)
    return rule(rid, qlo, qhi, support=support, coordinates=WINDOW_COORDINATES,
                extra_domain={"kv_len": {"min": klo, "max": khi}}, **changes)


def compiled(*declarations):
    return compile_service_regime_domains(list(declarations))


def engine(declarations):
    raw = hardware_case()
    raw.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = list(declarations)
    return raw


def select(raw, shape):
    result = graph_costs(raw, (replace(copy_op(4000), semantic_shape=shape),), isolated=False)
    selected = [row["id"] for row in result["parameters"]["service_regimes"]
                if row["queries"].get("observable_response")]
    assert len(selected) == 1, result["parameters"]["service_regimes"]
    return selected[0], result


def test_integer_gap_between_declared_domains_is_rejected():
    support = {"q": {"min": 1, "max": 256}}
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(rule("low", 1, 128, support=support),
                 rule("high", 130, 256, support=support))


def test_fractional_gap_between_declared_domains_is_rejected():
    # (128, 129) is a real uncovered interval even though both endpoints are
    # integers and no coordinate projects outside the union of declared boxes.
    support = {"q": {"min": 1, "max": 256}}
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(rule("low", 1, 128, support=support),
                 rule("high", 129, 256, support=support))


def test_exact_shared_endpoint_is_covered_by_exactly_one_rule():
    report = compiled(rule("low", 1, 100), rule("high", 100, 200))[0]
    assert report["members"] == ["low", "high"]
    assert report["probes"] > 0
    assert report["support"]["q"]["min"] == 1
    assert report["support"]["q"]["max"] == 200


def test_nextafter_probes_land_on_the_declared_side_and_order_never_changes_selection():
    low, high = rule("low", 1, 100), rule("high", 100, 200)
    for declarations in ((low, high), (high, low)):
        raw = engine(declarations)
        assert select(raw, (50,))[0] == "low"
        assert select(raw, (math.nextafter(100.0, -math.inf),))[0] == "low"
        assert select(raw, (100.0,))[0] == "high"
        assert select(raw, (math.nextafter(100.0, math.inf),))[0] == "high"
        assert select(raw, (150.0,))[0] == "high"
    forward = [select(engine((low, high)), (q,))[1]["duration_ms"] for q in (50, 100, 150)]
    reverse = [select(engine((high, low)), (q,))[1]["duration_ms"] for q in (50, 100, 150)]
    assert forward == reverse


def test_two_dimensional_interior_hole_is_rejected_despite_covered_projections():
    # Each coordinate is projected onto a fully covered interval by the bottom
    # and top bands, yet the central rectangle has no applicable rule.
    declarations = (rect("bottom", 1, 200, 1, 50), rect("top", 1, 200, 150, 200),
                    rect("left", 1, 50, 50, 150), rect("right", 150, 200, 50, 150))
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(*declarations)


def test_uncovered_boundary_face_is_rejected():
    declarations = (rect("west", 1, 100, 1, 200), rect("east", 100, 200, 1, 100))
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(*declarations)


def test_overlapping_interiors_are_rejected():
    with pytest.raises(ModelingError, match="interior_overlap"):
        compiled(rect("wide", 1, 150, 1, 200), rect("narrow", 100, 200, 1, 200))


def test_mismatched_shared_face_function_limits_are_rejected():
    with pytest.raises(ModelingError, match="seam_limit_mismatch"):
        compiled(rule("low", 1, 100, response=affine(8000.0)),
                 rule("high", 100, 200, response=affine(9000.0)))


def test_matching_shared_face_limits_are_accepted():
    report = compiled(rule("low", 1, 100, response=affine(8000.0)),
                      rule("high", 100, 200, response=affine(8000.0)))[0]
    assert report["members"] == ["low", "high"]


def test_insufficient_knot_coverage_is_rejected():
    partial = curve(knots=((50, 4000.0), (150, 6000.0)))
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(rule("partial", 1, 200, response=partial))
    detached = curve(knots=((300, 4000.0), (400, 6000.0)))
    with pytest.raises(ModelingError, match="rule_outside_support"):
        compiled(rule("detached", 1, 200, response=detached))


def test_rate_dependent_seam_requires_a_provable_equivalent_expression():
    def scaled(fixed):
        return {"kind": "affine", "fixed_us": fixed, "terms": [
            {"coordinate": "q", "us_per_unit": 1.0,
             "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]}
    with pytest.raises(ModelingError, match="provable_equivalence"):
        compiled(rule("low", 1, 100, response=scaled(8000.0)),
                 rule("high", 100, 200, response=scaled(9000.0)))
    # Identical declarations are structurally equal; no sampled proof is used.
    compiled(rule("low", 1, 100, response=scaled(8000.0)),
             rule("high", 100, 200, response=scaled(8000.0)))


def test_non_singleton_guard_is_accepted_inside_a_continuity_group():
    selection_curve = curve()
    guarded = rule("guarded", 1, 200, response=selection_curve,
                   coordinates={"q": {"path": "semantic_shape.0"},
                                "kv_len": {"path": "semantic_shape.1"}},
                   extra_domain={"kv_len": {"min": 8192, "max": 131072}})
    report = compiled(guarded)[0]
    assert report["support"] == {"q": {"min": 1, "max": 200,
                                       "min_inclusive": True, "max_inclusive": True}}
    raw = engine((guarded,))
    chosen, result = select(raw, (50.0, 9000.0))
    assert chosen == "guarded"
    assert result["duration_ms"] == pytest.approx((8000 + 2000 * 49 / 199) / 1000)
    outside = graph_costs(raw, (replace(copy_op(4000), semantic_shape=(50.0, 1000.0)),),
                          isolated=False)
    assert outside["duration_ms"] == 4
    assert outside["parameters"]["service_regimes"][0]["queries"] == {
        "analytic_outside_support": 1}


def test_missing_coordinate_in_a_contract_is_an_explicit_error():
    broken = rule("missing", 1, 200, response=affine(),
                  coordinates={"q": {"path": "metadata.absent_q"}})
    with pytest.raises(ModelingError, match="requires_declared_coordinate:missing"):
        graph_costs(engine((broken,)), (copy_op(4000),), isolated=False)


def test_asserted_exterior_policies_are_applied_and_reported():
    analytic = curve(outside="analytic")
    raw = engine((rule("low", 1, 100, response=analytic),
                  rule("high", 100, 200, response=analytic)))
    outside = graph_costs(raw, (replace(copy_op(4000), semantic_shape=(300.0,)),),
                          isolated=False)
    assert outside["duration_ms"] == 4
    assert any(row.get("exterior_policies") == {"analytic": 1}
               and row["queries"].get("analytic_outside_support") == 1
               for row in outside["parameters"]["service_regimes"])

    linear = curve(outside="linear")
    raw = engine((rule("low", 1, 100, response=linear, outside="linear"),
                  rule("high", 100, 200, response=linear, outside="linear")))
    outside = graph_costs(raw, (replace(copy_op(4000), semantic_shape=(300.0,)),),
                          isolated=False)
    assert outside["duration_ms"] == pytest.approx((10000 + 2000 * 100 / 199) / 1000)
    assert any(row.get("exterior_policies") == {"linear": 1}
               and row["queries"].get("linear_outside_support") == 1
               for row in outside["parameters"]["service_regimes"])

    error = curve(outside="error")
    raw = engine((rule("low", 1, 100, response=error, outside="error"),
                  rule("high", 100, 200, response=error, outside="error")))
    with pytest.raises(ModelingError, match="outside_support"):
        graph_costs(raw, (replace(copy_op(4000), semantic_shape=(300.0,)),), isolated=False)


def test_exclusive_support_bounds_are_normalized_not_tolerance_erased():
    exclusive_max = {"q": {"min": 1, "max": 200, "max_inclusive": False}}
    compiled(rule("low", 1, 100, support=exclusive_max),
             rule("high", 100, 200, support=exclusive_max))
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(rule("low", 1, 100, support=exclusive_max),
                 rule("high", 100, 199, support=exclusive_max))
    exclusive_min = {"q": {"min": 1, "max": 200, "min_inclusive": False}}
    compiled(rule("low", 1, 100, support=exclusive_min),
             rule("high", 100, 200, support=exclusive_min))
    with pytest.raises(ModelingError, match="interior_gap"):
        compiled(rule("low", 2, 100, support=exclusive_min),
                 rule("high", 100, 200, support=exclusive_min))


def test_validation_work_is_bounded_and_fails_explicitly():
    coordinates = {f"x{index}": {"path": f"semantic_shape.{index}"} for index in range(4)}
    support = {name: {"min": 1, "max": 200} for name in coordinates}
    declarations = []
    for index in range(8):
        domain = {name: {"min": 1 + index * 20, "max": 20 + index * 20}
                  for name in coordinates}
        declarations.append({
            "id": f"box{index}", "match": {"recipe": "copy"},
            "coordinates": coordinates, "domain": domain,
            "response": {"kind": "affine", "fixed_us": 8000},
            "continuity": {"group": "many", "support": support, "outside": "analytic"}})
    with pytest.raises(ModelingError, match="validation_budget_exceeded"):
        compiled(*declarations)


def test_legacy_box_for_the_same_migrated_domain_is_rejected():
    covered = rule("covered", 1, 200)
    legacy = {"id": "legacy", "match": {"recipe": "copy"},
              "coordinates": {"q": {"path": "semantic_shape.0"}},
              "domain": {"q": {"min": 1, "max": 200}},
              "response": {"kind": "affine", "fixed_us": 8000}}
    with pytest.raises(ModelingError, match="requires_coverage_contract"):
        compiled(covered, legacy)


def test_group_members_must_agree_on_identity_coordinates_and_execution():
    with pytest.raises(ModelingError, match="members_differ"):
        compiled(rule("a", 1, 200, match={"recipe": "copy", "dtype": "bf16"}),
                 rule("b", 1, 200, match={"recipe": "copy", "dtype": "fp16"}))
    joint = {"kind": "joint_native", "resource": "copy_service",
             "reference_rate": 1e9, "native_kernel": "CopyKernel"}
    with pytest.raises(ModelingError, match="execution_mode"):
        compiled(rule("a", 1, 200, execution=joint), rule("b", 1, 200))
    with pytest.raises(ModelingError, match="coordinates"):
        compiled(rule("a", 1, 200),
                 rule("b", 1, 200, coordinates={"q": {"path": "semantic_shape.0",
                                                     "scale": 2}}))


def test_exported_checker_is_the_single_interpretation():
    low, high = rule("low", 1, 100), rule("high", 100, 200)
    assert validate_service_regime_domains([low, high]) == compiled(low, high)
    assert rules_overlap(low, rule("other", 50, 150))
    assert not rules_overlap(low, rule("far", 300, 400))


def test_response_declarations_reject_booleans_nonfinite_and_invalid_rates():
    coordinates = {"q": {"path": "semantic_shape.0"}}
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "affine", "fixed_us": True}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "piecewise_linear", "coordinate": "q",
            "knots": [[1, float("nan")], [2, 3]]}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "multilinear", "grid": [
            {"coordinate": "q", "knots": [1, 2]}],
            "values_us": [1, float("inf")]}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "piecewise_linear", "coordinate": "q",
            "knots": [[1, 1], [1, 2]]}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "multilinear", "grid": [
            {"coordinate": "q", "knots": [1, 2]}], "values_us": [1]}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "affine", "terms": [
            {"coordinate": "q", "us_per_unit": 1,
             "scale_with": {"resource": "hbm", "reference_rate": 0}}]}, coordinates)
    with pytest.raises(ModelingError):
        ResponseFunction.from_mapping({"kind": "roofline", "services": [
            {"coordinate": "q", "resource": "cube", "dtype": "bf16",
             "utilization": True}]}, coordinates)


def test_response_evaluation_rejects_invalid_values_rates_and_extrapolation():
    coordinates = {"q": {"path": "semantic_shape.0"}}
    function = ResponseFunction.from_mapping({"kind": "piecewise_linear",
        "coordinate": "q", "knots": [[1, 10], [2, 20]]}, coordinates)
    with pytest.raises(ModelingError):
        function.evaluate({"q": True})
    with pytest.raises(ModelingError):
        function.evaluate({})
    with pytest.raises(ModelingError):
        nonnegative(float("nan"))
    with pytest.raises(ModelingError):
        nonnegative(float("inf"))
    with pytest.raises(ModelingError):
        positive(0)
    with pytest.raises(ModelingError):
        scaled_coordinate(1e308, 10)
    decreasing = ResponseFunction.from_mapping({"kind": "piecewise_linear",
        "coordinate": "q", "outside": "linear", "knots": [[1, 100], [2, 1]]},
        coordinates)
    with pytest.raises(ModelingError):
        decreasing.evaluate({"q": 1000})
    roofline = ResponseFunction.from_mapping({"kind": "roofline", "services": [
        {"coordinate": "q", "resource": "cube", "dtype": "bf16"}]}, coordinates)
    with pytest.raises(ModelingError):
        roofline.evaluate({"q": 1}, rate_for=lambda term: 0)
    with pytest.raises(ModelingError):
        roofline.evaluate({"q": 1}, rate_for=lambda term: float("nan"))
    scaled = ResponseFunction.from_mapping({"kind": "affine", "fixed_us": 1,
        "terms": [{"coordinate": "q", "us_per_unit": 1,
                   "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]},
        coordinates)
    with pytest.raises(ModelingError):
        scaled.evaluate({"q": 1})


def test_runtime_selection_uses_compiled_knot_coverage_when_raw_domains_overlap():
    first = rule("low-knots", 1, 200, response=curve(knots=((1, 8000.0), (100, 9000.0))))
    second = rule("high-knots", 1, 200, response=curve(knots=((100, 9000.0), (200, 10000.0))))
    for declarations in ((first, second), (second, first)):
        raw = engine(declarations)
        assert select(raw, (50.0,))[0] == "low-knots"
        assert select(raw, (100.0,))[0] == "high-knots"
        chosen, result = select(raw, (150.0,))
        assert chosen == "high-knots"
        assert result["duration_ms"] == pytest.approx(9.5)


def test_ambiguous_exterior_groups_are_rejected_regardless_of_declaration_order():
    left = rule("left", 1, 100, group="left-group",
                support={"q": {"min": 1, "max": 100}},
                outside="linear",
                response=curve(outside="linear", knots=((1, 8000.0), (100, 10000.0))))
    right = rule("right", 200, 300, group="right-group",
                 support={"q": {"min": 200, "max": 300}},
                 outside="linear",
                 response=curve(outside="linear", knots=((200, 5000.0), (300, 8000.0))))
    for declarations in ((left, right), (right, left)):
        compiled(*declarations)
        with pytest.raises(ModelingError, match="ambiguous_exterior_groups"):
            graph_costs(engine(declarations), (replace(copy_op(4000), semantic_shape=(150.0,)),),
                        isolated=False)


def test_seam_continuity_is_enforced_with_fixed_singleton_support_axes():
    fixed_support = {"q": {"min": 1, "max": 200}, "kv_len": {"min": 8192, "max": 8192}}
    low = rect("low", 1, 100, 8192, 8192, support=fixed_support, response=affine(8000.0))
    high_bad = rect("high", 100, 200, 8192, 8192, support=fixed_support, response=affine(9000.0))
    with pytest.raises(ModelingError, match="seam_limit_mismatch"):
        compiled(low, high_bad)
    high_ok = rect("high", 100, 200, 8192, 8192, support=fixed_support,
                   response=curve(knots=((100, 8000.0), (200, 9000.0))))
    assert compiled(low, high_ok)[0]["seam_vertices"] >= 1


def test_disjoint_guards_and_large_exterior_distances_are_accepted_and_selected():
    coords = {"q": {"path": "semantic_shape.0"}, "kv_len": {"path": "semantic_shape.1"}}
    covered = rule("covered", 1, 200, coordinates=coords, response=curve(),
                   extra_domain={"kv_len": {"min": 1000, "max": 2000}})
    unrelated = rule("unrelated", 1, 200, coordinates=coords,
                     extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    unrelated.pop("continuity", None)
    assert len(compiled(covered, unrelated)) == 1
    second_group = rule("second", 1, 200, group="second-group", coordinates=coords,
                        response=curve(), extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    outside = graph_costs(engine((covered, second_group)),
                          (replace(copy_op(4000), semantic_shape=(50.0, 2500.0)),),
                          isolated=False)
    assert outside["duration_ms"] == 4

    hbm_coords = {"q": {"path": "work.hbm"}}
    small = rule("small", 1, 100, coordinates=hbm_coords, outside="linear",
                 response=curve(outside="linear", knots=((1, 8000.0), (100, 9000.0))))
    large = rule("large", 100, 200, coordinates=hbm_coords, outside="linear",
                 response=curve(outside="linear", knots=((100, 9000.0), (200, 11000.0))))
    extrapolated = graph_costs(engine((small, large)), (copy_op(1_000_000_000_000),),
                               isolated=False)
    assert extrapolated["parameters"]["calibration"] == "observable_kernel_regimes"
    assert "large" not in extrapolated["parameters"]["regime_selection"]["unselected"]

    low_guard = rule("low-guard", 1, 100, coordinates=coords, response=curve(),
                     extra_domain={"kv_len": {"min": 1000, "max": 2000}})
    high_guard = rule("high-guard", 100, 200, coordinates=coords, response=curve(),
                      extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    with pytest.raises(ModelingError, match="members_differ:copy-domain:guards"):
        compiled(low_guard, high_guard)
    open_low = rule("open-low", 1, 100, coordinates=coords,
                    response=curve(knots=((1, 8000.0), (100, 9000.0))),
                    extra_domain={"kv_len": {"min": 8192}})
    open_high = rule("open-high", 100, 200, coordinates=coords,
                     response=curve(knots=((100, 9000.0), (200, 10000.0))),
                     extra_domain={"kv_len": {"min": 8192}})
    assert compiled(open_low, open_high)[0]["guards"]["kv_len"] == {
        "min": 8192.0, "max": None, "min_inclusive": True, "max_inclusive": True
    }

    from test_joint_native_service import joint_case
    base_joint = joint_case()
    execution = base_joint["runtime"]["kernels"]["service_regimes"][0]["execution"]
    j_low = rule("low", 1, 100, response=curve(), execution=execution)
    j_high = rule("high", 100, 200, response=curve(), execution=execution)
    base_joint["runtime"]["kernels"]["service_regimes"] = [j_low, j_high]
    forward_ops = tuple(replace(copy_op(4000), op_id=str(i), semantic_shape=(s,))
                        for i, s in enumerate((50, 150)))
    reverse_ops = tuple(replace(copy_op(4000), op_id=str(i), semantic_shape=(s,))
                        for i, s in enumerate((150, 50)))
    forward = graph_costs(base_joint, forward_ops, isolated=False)["parameters"]["joint_native_services"]
    reverse = graph_costs(base_joint, reverse_ops, isolated=False)["parameters"]["joint_native_services"]
    assert forward == reverse
    assert forward[0]["rule_ids"] == ["high", "low"]
    assert forward[0]["rule_counts"] == {"high": 1, "low": 1}
