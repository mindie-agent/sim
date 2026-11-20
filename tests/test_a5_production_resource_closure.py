"""Ordinary production pricing and T06 profile handoff.

The production CostModel consumes physical work only.  A PrimitiveStore may be
read through the explicit diagnostic API, but its duration can never become an
ordinary completion floor.  A normalized T06 profile attached to the Case is
exposed to the executor/recipe owner without inventing a second service model.
"""
from copy import deepcopy
import json

import pytest

from pathlib import Path

from mindie_sim.observations.communication import (
    A5_NATIVE_GROUP2, fit_a5_native_local_service,
)
from mindie_sim.compose import compose_case
from mindie_sim.observations.calibration import (
    apply_profiler_supply, apply_resource_profile, bind_a5_local_service,
    bind_a5_local_service_operator, bind_operator_resource_service,
    import_profiler_supply, load_resource_profile, resource_service_binding,
    select_resource_service_candidate,
)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA, normalize_resource_service,
)
from mindie_sim.prediction import predict
from mindie_sim.prediction.cost import CostModel, primitive_key
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case, linear_work


def _op(value, plan):
    return linear_work(plan, value.model, value.runtime, value.machine).ops[0]


def test_ordinary_price_ignores_primitive_floor_and_keeps_explicit_diagnostic():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    op = _op(value, plan)
    before = predict(value, lower=linear_work)
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(value, op, "none"), duration_ms=10,
                     run_id="diagnostic-only", repeat_index=0,
                     source="independent timing oracle")
        cost = CostModel(value, store)
        priced = cost.price(op, "none")
        diagnostic = cost.lookup_diagnostic(op, "none")
    assert before["requests"][0]["ttft_ms"] == 4
    assert priced.timing is None
    assert diagnostic["timing"].duration_ms == 10
    summary = cost.summary()
    assert summary["production_price_basis"] == "physical_work_only"
    assert summary["diagnostic_primitive_lookups"] == 1
    assert summary["operator_occurrences"]["measured_primitive"] == 0
    assert summary["diagnostic_primitive_hits"] == 1
    assert "diagnostic_primitive_evidence_not_production" in summary["calibration"]


def test_ordinary_price_without_store_has_no_diagnostic_read():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    cost = CostModel(value)
    priced = cost.price(_op(value, plan), "none")
    assert priced.timing is None
    assert priced.basis == "analytic_work"
    assert cost.lookup_diagnostic(_op(value, plan), "none") is None
    assert cost.summary()["diagnostic_primitive_lookups"] == 0


def test_case_profile_exposes_normalized_services_and_selects_by_condition():
    service = normalize_resource_service({
        "id": "hbm-production", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 1, "response_us": 0,
        "active_units": 1, "rate_scope": "aggregate",
        "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "conditions": {"hardware": "declared-gpu", "rates.hbm": 1_000_000},
    })
    raw = apply_resource_profile(case(), {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "declared-gpu",
        "services": [service.to_dict()],
    })
    parsed = parse_case(raw)
    cost = CostModel(parsed)
    assert [item.service_id for item in cost.resource_services] == ["hbm-production"]
    matching = {"hardware": "declared-gpu", "rates": {"hbm": 1_000_000},
                "unrelated_new_field": {"ignored": True}}
    assert cost.select_resource_service(matching).model_id == service.model_id
    assert cost.select_resource_service(
        matching, service_id="hbm-production").model_id == service.model_id
    changed_rate = {"hardware": "declared-gpu", "rates": {"hbm": 2_000_000}}
    with pytest.raises(ValueError, match="rates.hbm") as auto_error:
        cost.select_resource_service(changed_rate)
    assert "conditions_have_no_match" in str(auto_error.value)
    with pytest.raises(ValueError, match="rates.hbm") as id_error:
        cost.select_resource_service(
            changed_rate, service_id="hbm-production")
    assert "conditions_have_no_match" in str(id_error.value)
    missing = {"hardware": "declared-gpu", "rates": {}}
    with pytest.raises(ValueError, match="<missing>"):
        cost.select_resource_service(missing, service_id="hbm-production")
    for context in ({"hardware": "other-gpu", "rates": {"hbm": 1_000_000}},
                    changed_rate):
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(
                context, service_id="hbm-production")
    with pytest.raises(ValueError, match="id_not_found"):
        cost.select_resource_service(matching, service_id="missing")
    op = OpSpec("copy", "copy", "body", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16", recipe="copy")
    bound = cost.bind_operator_resource_service(
        op, matching, resource_role="hbm")
    binding = dict(bound.metadata)["resource_service"]
    assert binding["service_id"] == "hbm-production"
    assert binding["resource_role"] == "hbm"
    assert binding["work_unit"] == "bytes"
    assert binding["phase"] == "body"
    assert binding["production_prediction"] is True
    summary = cost.summary()
    assert summary["resource_services"] == ["hbm-production"]
    assert summary["resource_service_profile_source"]["kind"] == "in_memory"


def test_costmodel_delegates_top_level_hardware_and_rejects_conflict():
    service = normalize_resource_service({
        "id": "needs-top-hardware", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "conditions": {},
    })
    profile = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "Ascend950DT_9582",
        "services": [service.to_dict()],
    }
    raw = case()
    raw["runtime"]["metadata"] = {"resource_service_profile": profile}
    cost = CostModel(parse_case(raw))
    loaded = cost.resource_services[0]
    assert loaded.conditions["hardware"] == "Ascend950DT_9582"
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        cost.select_resource_service({"hardware": "Ascend950PR_9579"})

    conflict_service = normalize_resource_service({
        "id": "pr-conflict", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "conditions": {"hardware": "Ascend950PR_9579"},
    })
    conflict = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "Ascend950DT_9582",
        "services": [conflict_service.to_dict()],
    }
    raw = case()
    raw["runtime"]["metadata"] = {"resource_service_profile": conflict}
    with pytest.raises(ValueError, match="hardware_mismatch"):
        CostModel(parse_case(raw))


def test_file_mapping_and_case_profiles_reach_the_same_selection(tmp_path):
    service = normalize_resource_service({
        "id": "hbm-choice", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "conditions": {"hardware": "declared-gpu", "rates.hbm": 1_000_000},
    })
    profile = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "declared-gpu",
        "services": [service.to_dict()],
    }
    context = {"hardware": "declared-gpu", "rates": {"hbm": 1_000_000},
               "extra": "ignored"}
    by_mapping = CostModel(parse_case(apply_resource_profile(case(), profile)))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    by_file = CostModel(parse_case(apply_resource_profile(case(), path)))
    direct = case()
    direct["runtime"]["metadata"] = {"resource_service_profile": profile}
    by_case = CostModel(parse_case(direct))
    assert by_mapping.resource_services[0].model_id == service.model_id
    assert by_file.resource_services[0].model_id == service.model_id
    assert by_case.resource_services[0].model_id == service.model_id
    assert by_mapping.select_resource_service(context).model_id == service.model_id
    assert by_file.select_resource_service(context).model_id == service.model_id
    assert by_case.select_resource_service(context).model_id == service.model_id
    assert by_mapping.resource_service_profile["source"]["kind"] == "in_memory"
    assert by_file.resource_service_profile["source"]["kind"] == "file"


def test_ambiguous_candidates_are_explicit_and_id_still_checks_conditions():
    def service(identifier):
        return normalize_resource_service({
            "id": identifier, "scope": "independent_resource",
            "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
            "boundary": "resource_phase", "rate_scope": "aggregate",
            "active_units": 1,
            "supply": {"kind": "absolute_rate", "rate": 1_000_000},
            "conditions": {"hardware": "declared-gpu"},
        })
    profile = {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "declared-gpu",
        "services": [service("hbm-a").to_dict(), service("hbm-b").to_dict()],
    }
    raw = case()
    raw["runtime"]["metadata"] = {"resource_service_profile": profile}
    cost = CostModel(parse_case(raw))
    with pytest.raises(ValueError, match="conditions_are_ambiguous:hbm-a,hbm-b"):
        cost.select_resource_service({"hardware": "declared-gpu"})
    assert cost.select_resource_service(
        {"hardware": "declared-gpu"}, service_id="hbm-a").service_id == "hbm-a"
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        cost.select_resource_service(
            {"hardware": "other-gpu"}, service_id="hbm-a")


def _h5_service(rate, service_id):
    return normalize_resource_service({
        "id": service_id, "scope": "independent_resource",
        "resources": ["hbm"], "phase": "copy", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "active_units": 1, "rate_scope": "aggregate",
        "supply": {"kind": "absolute_rate", "rate": rate},
        "conditions": {"hardware": "declared-gpu"},
    })


def _h5_service_wrapper(execution, service, work):
    op = OpSpec("copy", "copy", "copy", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=work, dtype="bf16", recipe="copy")
    bound = bind_operator_resource_service(
        op, service, resource_role="hbm", phase="copy", work_unit="bytes",
        conditions={"hardware": "declared-gpu"})
    from types import SimpleNamespace
    context = SimpleNamespace(
        op=SimpleNamespace(metadata=tuple(bound.metadata)),
        resource_names={"hbm": "memory"}, work_units={"hbm": "bytes"})
    return execution._apply_resource_service_program(
        context, KernelProgram((Stage("work", {"memory": work}),)))


def _h5_execution():
    parsed = parse_case(case())
    cost = CostModel(parsed)
    execution = Execution(parsed, cost, include_submission=False)
    execution.resource_services = {
        "h5-96": _h5_service(96_000_000, "h5-96"),
        "h5-128": _h5_service(128_000_000, "h5-128"),
    }
    return execution


def _run_single(wrapper, name, engine, ready):
    done = wrapper.start(engine, name, ready)
    engine.advance(until=(done,))
    return done.time


def test_h5_t04_formula_second_size_and_fixed_once():
    execution = _h5_execution()
    for service_id, work, expected in (
            ("h5-96", 1536, 0.018), ("h5-96", 3072, 0.034),
            ("h5-128", 1536, 0.014), ("h5-128", 3072, 0.026)):
        rate = (96_000 if service_id == "h5-96" else 128_000)
        engine = Engine([Resource("memory", rate, unit="bytes", policy="fifo")])
        ready = engine.submit("ready")
        wrapper = _h5_service_wrapper(
            execution, execution.resource_services[service_id], work)
        assert _run_single(wrapper, service_id, engine, ready) == pytest.approx(
            expected, abs=1e-12)


def test_h5_t04_shared_resource_competition_keeps_fixed_once_per_invocation():
    execution = _h5_execution()
    service = execution.resource_services["h5-96"]
    engine = Engine([Resource("memory", 96_000, unit="bytes", policy="fifo")])
    ready = engine.submit("ready")
    first = _h5_service_wrapper(execution, service, 1536)
    second = _h5_service_wrapper(execution, service, 1536)
    d1 = first.start(engine, "first", ready)
    d2 = second.start(engine, "second", ready)
    engine.advance(until=(d1, d2), first=False)
    assert sorted((d1.time, d2.time)) == pytest.approx([0.018, 0.034])


def test_costmodel_accepts_real_t04_context_object_for_select_and_bind():
    from types import SimpleNamespace
    service = normalize_resource_service({
        "id": "t04-context-service", "scope": "independent_resource",
        "resources": ["hbm"], "phase": "body", "work_unit": "bytes",
        "boundary": "resource_phase", "rate_scope": "aggregate",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": 1_000_000},
        "conditions": {"hardware": "declared-gpu"},
    })
    raw = apply_resource_profile(case(), {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": "declared-gpu",
        "services": [service.to_dict()],
    })
    cost = CostModel(parse_case(raw))
    op = OpSpec("copy", "copy", "body", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16", recipe="copy")
    context = SimpleNamespace(
        op=op,
        service_identity={"hardware": {"key": "declared-gpu"},
                          "runtime": "vllm", "revision": "test",
                          "format": "kernel", "dtype": "bf16", "layout": "nd",
                          "axes": {}},
        resource_names={"hbm": "memory"}, work_units={"hbm": "bytes"},
        work={"memory": 1000}, resource_units={})
    assert cost.select_resource_service(context).service_id == "t04-context-service"
    bound = cost.bind_operator_resource_service(
        op, context, resource_role="hbm")
    binding = dict(bound.metadata)["resource_service"]
    assert binding["op_id"] == "copy"
    assert binding["op_phase"] == "body"
    assert binding["resource_role"] == "hbm"
    assert binding["boundary"] == "resource_phase"
    assert binding["work_scope"] == "aggregate"
    assert binding["normalization"]["applied_once"] is True


def test_a5_local_service_binding_exposes_t04_t09_pw_t15_fields():
    source = Path(A5_NATIVE_GROUP2["source"])
    environment = source.parents[1] / "environment.txt"
    if not source.is_file() or not environment.is_file():
        pytest.skip(f"A5 native campaign raw mount unavailable: {source}")
    image = next(
        token.split("=", 1)[1]
        for line in environment.read_text().splitlines()
        if line.startswith("image_id=")
        for token in line.split() if token.startswith("image_id="))
    record = fit_a5_native_local_service(
        collective="all_gather", stack="rack08", cann="9.2.0",
        deployment_image_id=image, group_size=2)
    binding = bind_a5_local_service(
        record, domain=record["domain"], resource_role="aicpu_child")
    assert binding["service_id"] == record["service"]["id"]
    assert binding["resource_role"] == "aicpu_child"
    assert binding["resources"] == ["aicpu_child"]
    assert binding["work_unit"] == "bytes"
    assert binding["rate_scope"] == "per_owner"
    assert binding["work_scope"] == "per_owner"
    assert binding["active_units"] == 2
    assert binding["boundary"] == "resource_phase"
    assert binding["fixed_us"] == pytest.approx(39.78255936532216)
    assert binding["startup_us"] == binding["fixed_us"]
    assert binding["response_us"] == 0
    assert binding["normalization"]["applied_once"] is True
    assert binding["startup_once"] is True and binding["fixed_once"] is True
    assert binding["domain_key"]
    assert binding["applicability_key"]
    assert binding["lineage_key"]
    assert binding["evidence_gaps"]
    assert binding["link_transit_identified"] is False
    assert binding["hop_scaling"] == "not_derived"
    assert binding["service_semantics"] == "aicpu_child_collective_phase"
    assert binding["actual_phase"] is None

    op = OpSpec("ag", "all_gather", "collective_expand", 0, 0, ("r",),
                (1,), (1,), (1,), (1,), link_bytes=65536, dtype="bf16",
                recipe="all_gather")
    attached = bind_a5_local_service_operator(
        record, op, domain=record["domain"], resource_role="aicpu_child")
    attached_binding = dict(attached.metadata)["resource_service"]
    assert attached_binding["op_family"] == "all_gather"
    assert attached_binding["op_phase"] == "collective_expand"
    assert attached_binding["actual_phase"] == "collective_expand"

    profile = load_resource_profile({
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "hardware": binding["conditions"]["hardware"],
        "services": [record["service"]],
    })
    normalized = normalize_resource_service(profile["services"][0])
    selected = select_resource_service_candidate(
        [normalized], record["domain"])
    assert selected.service_id == record["service"]["id"]


def test_ordinary_production_ignores_amplified_or_deleted_legacy_rules():
    here = Path(__file__).parent
    supply = here / "fixtures" / "r64_qbmm_dual_load" / "supply.json"
    hardware = "Ascend950DT_9582"
    def composed():
        return compose_case(
            "glm-5.1", hardware, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
            prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
            num_devices=1, ranks_per_node=1)
    base = apply_profiler_supply(
        composed(), supply, recipe="output", hardware=hardware, owners=24)
    ordinary = predict(base, scope="body")
    rules = import_profiler_supply(
        supply, recipe="output", hardware=hardware, owners=24)["service_regimes"]
    assert rules
    amplified = deepcopy(base)
    amplified["runtime"].setdefault("kernels", {})["service_regimes"] = deepcopy(rules)
    amplified["runtime"]["kernels"]["service_regimes"][0]["response"]["fixed_us"] = 1_000_000.0
    amplified_result = predict(amplified, scope="body")
    deleted = deepcopy(base)
    deleted["runtime"].setdefault("kernels", {}).pop("service_regimes", None)
    deleted_result = predict(deleted, scope="body")
    assert (ordinary["operator_occurrences"]
            == amplified_result["operator_occurrences"]
            == deleted_result["operator_occurrences"])
    assert (ordinary["execution"]["serviced"]
            == amplified_result["execution"]["serviced"]
            == deleted_result["execution"]["serviced"])
    assert ordinary["duration_ms"] == pytest.approx(amplified_result["duration_ms"])
    assert ordinary["duration_ms"] == pytest.approx(deleted_result["duration_ms"])
