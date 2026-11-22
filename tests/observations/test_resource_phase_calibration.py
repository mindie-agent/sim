"""
[Role] Verify resource-rate identification, profile selection and stage consumption.
[Invariants] Fit each declared boundary independently and freeze held-out evidence.
[Timeouts & Locks] Small in-memory fits and deterministic CPU engine runs.
[Side-Effect Policy] Read retained evidence and write profiles only under tmp_path.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from _support.axes import REPRESENTATIVE_SKUS
from _support.captures import REPO_ROOT, fixture_path
from _support.memory_cases import SPARSE_OFFLOAD_SKU, case, linear_work

from mindie_sim import predict
from mindie_sim.compose import compose_case
from mindie_sim.observations.calibration import (
    apply_profiler_supply,
    apply_resource_profile,
    bind_a5_local_service, bind_a5_local_service_operator, bind_operator_resource_service,
    export_resource_profile,
    import_profiler_supply,
    load_resource_profile,
    resource_service_binding,
    service_conditions_match,
)
from mindie_sim.observations.communication import A5_NATIVE_GROUP2, fit_a5_native_local_service
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA,
    normalize_resource_service,
    select_resource_service,
    sparse_mte1_bytes,
)
from mindie_sim.observations.service_fitting import (
    calibration_role_partitions,
    fit_resource_service,
    fit_resource_service_from_role_records,
)
from mindie_sim.prediction.efficiency.cost import CostModel, primitive_key
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

pytestmark = pytest.mark.tier0

GATHER_RULES = REPO_ROOT / "data/observations/a5-local-service-regimes-20261030/parameters.json"
SUPPLY = fixture_path("r64_qbmm_dual_load", "supply.json")
SUPPLY_MODEL = "glm-5.1"
PROFILE_HARDWARE = SPARSE_OFFLOAD_SKU
FOREIGN_HARDWARE = REPRESENTATIVE_SKUS["a3"]
NOMINAL_GATHER_BPS = 4_224_000_000_000
GATHER_BYTES_PER_ROW = 6 * 3072 * 2
COPY_RATE_BPS = 96_000_000
FOREIGN_COPY_RATE_BPS = 128_000_000
FIXED_COPY_US = 2
OWNER_COUNT = 32
COLLECTIVE_SOURCE = Path(A5_NATIVE_GROUP2["source"])
COLLECTIVE_ENVIRONMENT = COLLECTIVE_SOURCE.parents[1] / "environment.txt"


def service(**changes):
    return {
        "id": "copy-service", "scope": "independent_resource", "resources": ["hbm"],
        "phase": "copy", "work_unit": "bytes", "boundary": "resource_phase",
        "active_units": 1, "supply": {"kind": "absolute_rate", "rate": COPY_RATE_BPS},
        **changes,
    }


def profile(*services, hardware=PROFILE_HARDWARE):
    return {"schema": RESOURCE_SERVICE_PROFILE_SCHEMA, "hardware": hardware,
            "services": list(services)}


def sample(work, duration, run="train", *, repeat=0, split="train"):
    return {"work": work, "duration_us": duration, "run_id": run, "repeat_index": repeat,
            "split": split, "source": run}


def packet(samples, **changes):
    base = service(samples=samples, **changes)
    if "supply" not in changes:
        del base["supply"]
    return base


@pytest.mark.parametrize("supply, kind, status", [
    ({"kind": "absolute_rate", "rate": 500_000_000}, "absolute_rate", "partially_identified"),
    ({"kind": "nominal_multiplier", "nominal_rate": 1e9, "multiplier": 0.5},
     "nominal_multiplier", "partially_identified"),
    ({"kind": "nominal", "nominal_rate": 500_000_000}, "nominal", "nominal_specification"),
])
def test_declared_rate_identifies_fixed_cost_with_one_work_size(supply, kind, status):
    evidence = packet([sample(1000, 5)], supply=supply, response_us=1)
    fitted = fit_resource_service(evidence)
    model = normalize_resource_service(fitted["service"])
    assert fitted["fit"]["rate_source"] == "declared"
    assert (model.startup_us, model.response_us, model.fixed_us) == pytest.approx((2, 1, 3))
    assert model.effective_rate == pytest.approx(500_000_000)
    assert model.rate_representation["input_kind"] == kind
    assert model.identification["status"] == status
    assert model.duration_us(2000) == pytest.approx(7)
    assert normalize_resource_service(model.to_dict()).duration_us(500) == pytest.approx(4)
    evidence["samples"] = [sample(1000, 1)]
    with pytest.raises(ValueError, match="contradict_declared_rate"):
        fit_resource_service(evidence)


def test_declared_fixed_cost_identifies_rate_with_one_work_size():
    evidence = packet([sample(1000, 5)], startup_us=1, response_us=1)
    fitted = fit_resource_service(evidence)
    model = normalize_resource_service(fitted["service"])
    assert fitted["fit"]["rate_source"] == "fitted_with_declared_fixed"
    assert model.fixed_us == 2
    assert not model.identification["fixed_identified"]
    assert model.effective_rate == pytest.approx(1000 * 1e6 / (5 - 2))
    assert model.duration_us(2000) == pytest.approx(8)
    evidence["samples"] = [sample(1000, 1.5)]
    with pytest.raises(ValueError, match="contradict_declared_startup"):
        fit_resource_service(evidence)


@pytest.mark.parametrize("changes, error", [
    ({}, "two_work_sizes_or_a_declared_rate"),
    ({"boundary": "step"}, "boundary_must_be_resource_phase"),
    ({"target_ms": 2}, "rejects_target_duration"),
    ({"resources": ["hbm", "mte2_aiv"]}, "mixed_window_cannot_identify"),
    ({"provenance": {"window": "joint"}}, "mixed_window_cannot_identify"),
    ({"scope": "joint_phase", "provenance": {"window": "step"}}, "rejects_step_or_body_window"),
])
def test_fit_rejects_unidentified_or_wrong_boundary_evidence(changes, error):
    with pytest.raises(ValueError, match=error):
        fit_resource_service(packet([sample(1000, 2)], **changes))


def test_fit_weights_acquisitions_equally_and_excludes_heldout_samples():
    train = [sample(1000, 1, "a"), sample(3000, 5, "a"), sample(3000, 5, "b")]
    train.extend(sample(1000, 3, "b", repeat=index) for index in range(19))
    heldout = sample(2000, 3.5, "held", split="heldout")
    fitted = fit_resource_service(packet(train + [heldout]))
    # The two run medians at 1000 bytes average to 2 us, despite 19 repeats in run b.
    assert fitted["fit"]["fixed_us"] == pytest.approx(0.5)
    assert fitted["fit"]["effective_rate"] == pytest.approx(2000 * 1e6 / (5 - 2))
    assert fitted["heldout"][0]["predicted_us"] == pytest.approx(3.5)
    changed = fit_resource_service(packet(train * 2 + [{**heldout, "duration_us": 35}]))
    assert changed["service"] == fitted["service"]
    assert changed["heldout"][0]["relative_error"] != fitted["heldout"][0]["relative_error"]


def test_joint_strided_gather_fit_predicts_unseen_rows():
    rule = next(row for row in json.loads(GATHER_RULES.read_text())["rules"]
                if row["id"] == "a5-strided-state-gather")
    fixed = rule["response"]["fixed_us"]
    us_per_row = rule["response"]["terms"][0]["us_per_unit"]
    samples = [
        sample(rows * GATHER_BYTES_PER_ROW, duration, f"batch-{rows}", split=split)
        for rows, duration, split in [
            (16, 9.384521568627452, "train"),
            (64, 26.798966793168873, "train"),
            (32, 15.043888994307407, "heldout"),
        ]
    ]
    evidence = packet(samples, scope="joint_phase", resources=["mte2_aiv"],
                      phase="prefill", boundary="device_kernel", active_units=64,
                      provenance={"window": "joint",
                                  "rate_interpretation": rule["provenance"]["rate_interpretation"]})
    nominal = {"mte2_aiv": {"rate": NOMINAL_GATHER_BPS, "work_unit": "bytes",
                           "rate_scope": "aggregate"}}
    fitted = fit_resource_service(evidence, nominal_rates=nominal)
    model = normalize_resource_service(json.loads(json.dumps(fitted, allow_nan=False))["service"])
    assert model.kind == "joint_phase" and not model.identification["resource_identified"]
    assert model.fixed_us == pytest.approx(fixed)
    assert model.effective_rate == pytest.approx(GATHER_BYTES_PER_ROW * 1e6 / us_per_row)
    assert model.nominal_multiplier == pytest.approx(model.effective_rate / NOMINAL_GATHER_BPS)
    assert abs(fitted["heldout"][0]["relative_error"]) < 0.02
    predicted = model.prediction(128 * GATHER_BYTES_PER_ROW)
    assert predicted["extrapolated"] is True
    assert predicted["duration_us"] == pytest.approx(fixed + 128 * us_per_row)
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        model.counterfactual(GATHER_BYTES_PER_ROW, nominal_rate=NOMINAL_GATHER_BPS)


def role_records(**changes):
    return [
        {"cohort_id": f"copy-{work}", "role": "in_sample", "resource": "hbm",
         "scope": "independent_resource", "resources": ["hbm"], "phase": "copy",
         "work_unit": "bytes", "boundary": "resource_phase", "rate_scope": "aggregate",
         "active_units": 1, "provenance": {"window": "resource_phase"},
         "work": work, "resource_duration_us": duration, **changes}
        for work, duration in [(1536, 18), (3072, 34)]
    ]


def fit_roles(records, **changes):
    return fit_resource_service_from_role_records(
        records, **{"resource": "hbm", "phase": "copy", "work_unit": "bytes", **changes})


def test_transfer_and_unassigned_roles_cannot_move_frozen_parameters():
    training = role_records()
    transfer = {**training[0], "cohort_id": "transfer", "resource_duration_us": 1,
                "role": "frozen_training_function_transfer_to_new_capture"}
    unassigned = {**training[1], "cohort_id": "unassigned", "resource_duration_us": 1,
                  "role": "not_declared"}
    records = training + [transfer, unassigned]
    counts = calibration_role_partitions(records)["counts"]
    assert counts == {"in_sample": 2, "frozen_transfer": 1, "not_declared": 1}
    result = fit_roles(records)
    assert result["training_ids"] == [row["cohort_id"] for row in training]
    model = normalize_resource_service(result["service"])
    assert model.fixed_us == pytest.approx(FIXED_COPY_US)
    assert model.effective_rate == pytest.approx(COPY_RATE_BPS)
    assert model.duration_us(6144) == pytest.approx(66)
    transfer["resource_duration_us"] = 1000
    unassigned["resource_duration_us"] = 2000
    assert fit_roles(records)["service"] == result["service"]


@pytest.mark.parametrize("records, error", [
    ([{"cohort_id": "row", "role": "sealed_holdout"}], "calibration_role_unknown"),
    ([{"cohort_id": "row", "role": role} for role in ("in_sample", "not_declared")],
     "duplicate_cohort_id"),
])
def test_role_partitions_reject_invalid_or_reused_identity(records, error):
    with pytest.raises(ValueError, match=error):
        calibration_role_partitions(records)


@pytest.mark.parametrize("records, options, error", [
    (role_records(work_unit="rows"), {}, "work_unit"),
    (role_records(rate_scope="per_owner", active_units=OWNER_COUNT),
     {"rate_scope": "aggregate"}, "owner_scope_conflict"),
    (role_records(rate_scope="per_owner", active_units=OWNER_COUNT),
     {"rate_scope": "per_owner", "active_units": 1}, "owner_scope_conflict"),
])
def test_role_evidence_rejects_work_unit_and_owner_scope_conflicts(records, options, error):
    with pytest.raises(ValueError, match=error):
        fit_roles(records, **options)


def test_per_owner_fit_expands_to_aggregate_work_once():
    records = role_records(rate_scope="per_owner", active_units=OWNER_COUNT)
    fitted = fit_roles(records, rate_scope="per_owner", active_units=OWNER_COUNT)
    model = normalize_resource_service(fitted["service"])
    assert model.per_owner_rate == pytest.approx(COPY_RATE_BPS)
    assert model.aggregate_rate == pytest.approx(COPY_RATE_BPS * OWNER_COUNT)
    assert model.duration_us(1536 * OWNER_COUNT, work_scope="aggregate") == pytest.approx(18)


@pytest.mark.parametrize("window, reason", [
    ({}, "missing_explicit_resource_phase_work"),
    ({"window": "mixed"}, "mixed_window_declared_as_independent_resource"),
])
def test_role_duration_alias_requires_declared_resource_phase(window, reason):
    records = role_records(provenance=window)
    for row in records:
        row["duration_us"] = row.pop("resource_duration_us")
    result = fit_roles(records)
    assert result["service"] is None and result["status"] == "underdetermined"
    assert {row["reason"] for row in result["underdetermined"]} == {reason}


def test_mixed_role_windows_support_joint_duration_only():
    records = role_records(scope="joint_phase", boundary="device_kernel",
                           provenance={"window": "mixed"})
    for row in records:
        row["duration_us"] = row.pop("resource_duration_us")
    independent = fit_roles(records)
    assert independent["service"] is None and len(independent["joint_evidence"]) == 2
    model = normalize_resource_service(fit_roles(records, scope="joint_phase")["service"])
    assert not model.identification["resource_identified"]
    assert model.duration_us(3072) == pytest.approx(34)
    with pytest.raises(ValueError, match="joint_service_does_not_identify"):
        model.counterfactual(1536, nominal_rate=COPY_RATE_BPS)


@pytest.mark.parametrize("supply, ratio, multiplier_source", [
    ({"kind": "absolute_rate", "rate": COPY_RATE_BPS, "nominal_rate": FOREIGN_COPY_RATE_BPS},
     0.75, "inferred_from_effective_rate_and_nominal"),
    ({"kind": "nominal_multiplier", "multiplier": 0.75, "nominal_rate": FOREIGN_COPY_RATE_BPS},
     0.75, "declared"),
    ({"kind": "nominal", "nominal_rate": COPY_RATE_BPS}, 1, "identity"),
])
def test_rate_intervention_preserves_fixed_cost_and_supply_representation(supply, ratio, multiplier_source):
    model = normalize_resource_service(service(supply=supply, startup_us=FIXED_COPY_US))
    restored = normalize_resource_service(model.to_dict())
    assert restored.model_id == model.model_id
    assert restored.normalization["reused"]
    changed = restored.counterfactual(1536, nominal_rate=2 * FOREIGN_COPY_RATE_BPS)
    assert changed["multiplier_source"] == multiplier_source
    assert changed["effective_rate"] == pytest.approx(2 * FOREIGN_COPY_RATE_BPS * ratio)
    assert changed["duration_us"] == pytest.approx(FIXED_COPY_US + 1536 * 1e6 / changed["effective_rate"])
    explicit = model.counterfactual_effective_rate(1536, effective_rate=FOREIGN_COPY_RATE_BPS)
    assert explicit["duration_us"] == pytest.approx(14)
    with pytest.raises(ValueError, match="counterfactual_requires_nominal_rate"):
        normalize_resource_service(service()).counterfactual(1536, nominal_rate=FOREIGN_COPY_RATE_BPS)


@pytest.mark.parametrize("changes, error", [
    ({"supply": {"kind": "absolute_rate", "rate": 1000, "multiplier": 1}},
     "supply_representation_ambiguous"),
    ({"supply": {"kind": "nominal_multiplier", "multiplier": 0.5}}, "multiplier_requires_nominal_rate"),
    ({"supply": {"kind": "absolute_rate", "rate": 1000, "nominal": 1000}}, "supply_unknown_fields"),
    ({"scope": ""}, "requires_nonempty_scope"),
    ({"supply": {"kind": "absolute_rate", "rate": 1000, "nominal_unit": "rows"}},
     "nominal_rate_unit_mismatch"),
    ({"scope": "joint_phase", "identification": {"resource_identified": True}},
     "joint_service_cannot_identify"),
    ({"resources": ["hbm", "mte3"]}, "requires_exactly_one_resource"),
    ({"identification": {"status": "identified", "resource_identified": False}}, "identified_status_requires"),
    ({"provenance": {"parameter_class": "nominal_specification"}}, "parameter_class_conflict"),
    ({"supply": {"kind": "nominal", "nominal_rate": 1000},
      "identification": {"status": "identified", "resource_identified": True}}, "nominal_rate_cannot_be_identified"),
])
def test_normalization_rejects_conflicting_supply_or_identification(changes, error):
    with pytest.raises(ValueError, match=error):
        normalize_resource_service(service(**changes))


@pytest.mark.parametrize("nominal, error", [
    (COPY_RATE_BPS, "bare_nominal_rate_requires_explicit"),
    ({"rate": COPY_RATE_BPS, "work_unit": "bytes"}, "requires_explicit_rate_scope"),
    ({"rate": COPY_RATE_BPS, "rate_scope": "aggregate"}, "requires_explicit_work_unit"),
    ({"rate": COPY_RATE_BPS, "work_unit": "rows", "rate_scope": "aggregate"}, "nominal_rate_unit_mismatch"),
    ({"rate": COPY_RATE_BPS, "work_unit": "bytes", "rate_scope": "per_owner"}, "nominal_rate_scope_mismatch"),
])
def test_profile_requires_explicit_compatible_nominal_units_and_scope(nominal, error):
    record = service(supply={"kind": "nominal_multiplier", "multiplier": 0.5})
    with pytest.raises(ValueError, match=error):
        load_resource_profile(profile(record), nominal_rates={"hbm": nominal})


def test_profile_converts_nominal_work_only_with_explicit_unit_ratio():
    record = service(work_unit="rows", supply={"kind": "nominal_multiplier", "multiplier": 0.5})
    nominal = {"rate": NOMINAL_GATHER_BPS, "work_unit": "bytes", "rate_scope": "aggregate"}
    conversion = {"target_work_unit": "rows", "source_units_per_target_unit": GATHER_BYTES_PER_ROW}
    loaded = load_resource_profile(
        profile(record), nominal_rates={"hbm": {**nominal, "conversion": conversion}})
    assert loaded["services"][0]["effective_rate"] == pytest.approx(
        0.5 * NOMINAL_GATHER_BPS / GATHER_BYTES_PER_ROW)
    with pytest.raises(ValueError, match="nominal_rate_unit_mismatch"):
        load_resource_profile(profile(record), nominal_rates={"hbm": nominal})


def test_missing_malformed_and_hardware_conflicting_profiles_raise(tmp_path):
    path = tmp_path / "profile.json"
    with pytest.raises(ValueError, match="source_missing"):
        load_resource_profile(path)
    path.write_text(json.dumps({"schema": "unknown"}))
    with pytest.raises(ValueError, match="schema_unknown"):
        load_resource_profile(path)
    conflict = profile(service(conditions={"hardware": FOREIGN_HARDWARE}))
    path.write_text(json.dumps(conflict))
    for source in (conflict, path):
        with pytest.raises(ValueError, match="hardware_mismatch"):
            apply_resource_profile(case(), source)
    with pytest.raises(ValueError, match="hardware_mismatch"):
        load_resource_profile(profile(service()), hardware=FOREIGN_HARDWARE)


def test_profile_sources_preserve_role_conditions_and_selection(tmp_path):
    conditions = {"hardware": PROFILE_HARDWARE, "dtype": "bf16", "rates.hbm": COPY_RATE_BPS}
    records = role_records(conditions=conditions)
    for record in records:
        record["duration_us"] = record.pop("resource_duration_us")
    fitted = fit_roles(records)
    profile_record = profile(fitted["service"])
    path = tmp_path / "exported.json"
    builtin = load_resource_profile(profile_record, source_kind="builtin")
    export_resource_profile(builtin, path=path)
    value = case()
    value["runtime"]["metadata"] = {"keep": 1}
    original = deepcopy(value)
    direct = deepcopy(value)
    direct["runtime"]["metadata"]["resource_service_profile"] = profile_record
    mapping = apply_resource_profile(value, profile_record)
    file = apply_resource_profile(value, path)
    assert original == value
    assert mapping["runtime"]["metadata"]["keep"] == file["runtime"]["metadata"]["keep"] == 1
    assert file["runtime"]["metadata"]["resource_service_profile_source"]["sha256"]
    context = {"hardware": PROFILE_HARDWARE, "dtype": "bf16", "rates": {"hbm": COPY_RATE_BPS},
               "unrelated": {"vector": 1}}
    for raw in (mapping, file, direct):
        cost = CostModel(parse_case(raw))
        model = cost.select_resource_service(context)
        assert model.model_id == cost.select_resource_service(context, service_id=model.service_id).model_id
        matches = resource_service_binding(model, resource_role="hbm", phase="copy",
                                           work_unit="bytes", conditions=context)
        assert matches["conditions"] == conditions
        assert service_conditions_match(model.to_dict(), context)["applies"]
        for altered in ({"rates": {}}, {"rates": {"hbm": COPY_RATE_BPS / 2}},
                        {"hardware": FOREIGN_HARDWARE}, {"dtype": "fp16"}):
            unmatched = {**context, **altered}
            assert not service_conditions_match(model, unmatched)["applies"]
            for service_id in (None, model.service_id):
                with pytest.raises(ValueError, match="conditions_have_no_match"):
                    cost.select_resource_service(unmatched, service_id=service_id)


def test_ambiguous_profiles_require_id_and_still_check_conditions():
    raw = apply_resource_profile(case(), profile(service(id="left"), service(id="right")))
    cost = CostModel(parse_case(raw))
    context = {"hardware": PROFILE_HARDWARE}
    with pytest.raises(ValueError, match="conditions_are_ambiguous"):
        cost.select_resource_service(context)
    assert cost.select_resource_service(context, service_id="left").service_id == "left"
    with pytest.raises(ValueError, match="id_not_found"):
        cost.select_resource_service(context, service_id="missing")
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        cost.select_resource_service({"hardware": FOREIGN_HARDWARE}, service_id="left")


def copy_op(work, *, role="hbm", phase="copy"):
    return OpSpec("copy", "copy", phase, 0, 0, ("r",), (1,), (1,), (1,), (1,),
                  hbm_bytes=work if role == "hbm" else 0,
                  mte2_bytes=work if role == "mte1" else 0, dtype="bf16", recipe="copy")


def bound_program(cost, bound, role, work):
    execution = Execution(cost.case, cost, include_submission=False)
    context = SimpleNamespace(op=bound, resource_names={role: role}, work_units={role: "bytes"})
    program = KernelProgram((Stage("transfer", {role: work}),))
    return execution._apply_resource_service_program(context, program)


@pytest.mark.parametrize("work, rate, copies", [
    (1536, COPY_RATE_BPS, 1), (3072, COPY_RATE_BPS, 1),
    (3072, FOREIGN_COPY_RATE_BPS, 1), (1536, COPY_RATE_BPS, 2),
])
def test_bound_rate_and_setup_apply_once_per_invocation(work, rate, copies):
    raw = apply_resource_profile(case(), profile(
        service(startup_us=1, response_us=1, supply={"kind": "absolute_rate", "rate": rate}),
        hardware="declared-gpu"))
    cost = CostModel(parse_case(raw))
    op = copy_op(work)
    selector = SimpleNamespace(
        op=op, service_identity={"hardware": {"key": "declared-gpu"}, "dtype": "bf16"},
        resource_names={"hbm": "hbm"}, work_units={"hbm": "bytes"},
        work={"hbm": work}, resource_units={})
    model = cost.select_resource_service(selector)
    bound = cost.bind_operator_resource_service(op, selector, resource_role="hbm")
    binding = dict(bound.metadata)["resource_service"]
    assert binding["model_id"] == model.model_id and binding["fixed_once"] is True
    assert bind_operator_resource_service(bound, model, resource_role="hbm",
                                          conditions={"hardware": "declared-gpu"}) == bound
    for options, error in (({"phase": "prefill"}, "phase_mismatch"), ({"work_unit": "rows"}, "work_unit_mismatch")):
        with pytest.raises(ValueError, match=error):
            resource_service_binding(model, resource_role="hbm", **options)
    engine = Engine([Resource("hbm", 2 * rate / 1000, unit="bytes", policy="fifo")])
    ready = engine.submit("ready")
    program = bound_program(cost, bound, "hbm", work)
    completed = [program.start(engine, f"copy-{index}", ready) for index in range(copies)]
    engine.advance(until=completed, first=False)
    assert sorted(event.time for event in completed) == pytest.approx(
        [(FIXED_COPY_US + index * work * 1e6 / rate) / 1000 for index in range(1, copies + 1)])
    assert engine.serviced == {"hbm": copies * work}


def composed_supply(**changes):
    raw = compose_case(SUPPLY_MODEL, PROFILE_HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
                       prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
                       num_devices=1, ranks_per_node=1)
    return apply_profiler_supply(raw, SUPPLY, recipe="output", owners=24,
                                 **{"hardware": PROFILE_HARDWARE, **changes})


def select_sparse_supply(context, query):
    record = json.loads(SUPPLY.read_text())["resource_services"][0]
    selectors = {key: value for key, value in context.items() if key != "geometry"}
    return select_resource_service([record], q_len=query, context=0, kv_len=query, **selectors)


def sparse_context(**changes):
    return {
        "hardware": PROFILE_HARDWARE, "dtype": "bf16", "native_type": "KvQuantSparseFlashAttention",
        "phase": "prefill", "op_state": "dynamic", "owners": OWNER_COUNT,
        "source_identity": {"source_scope": "r64",
                            "image_digest": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"},
        "geometry": {"q_len": 1024}, "mix_block_num": 64, **changes,
    }


@pytest.mark.parametrize("change", [
    {"source_identity": None}, {"source_identity": {"source_scope": "r64", "image_digest": "f" * 64}},
    {"native_type": "MatMul"}, {"op_state": "static"}, {"owners": 24}, {"mix_block_num": 32},
    {"hardware": FOREIGN_HARDWARE},
])
def test_sparse_supply_selectors_reject_mismatched_capture_conditions(change):
    raw = composed_supply()
    cost = CostModel(parse_case(raw))
    base = sparse_context()
    selected = cost.select_resource_service(base)
    context = sparse_context(**change)
    assert select_sparse_supply(context, query=1024) is None
    for service_id in (None, selected.service_id):
        with pytest.raises(ValueError, match="conditions_have_no_match"):
            cost.select_resource_service(context, service_id=service_id)
    with pytest.raises(ValueError, match="conditions_do_not_match|conditions_have_no"):
        resource_service_binding(selected, resource_role="mte1", phase=selected.phase,
                                 work_unit=selected.work_unit, conditions=context)


@pytest.mark.parametrize("query", [1024, 2048])
def test_sparse_supply_prices_actual_geometry_and_conserves_work(query, tmp_path):
    raw = composed_supply()
    metadata = raw["runtime"]["metadata"]
    measured = metadata["profiler_measured_supply"]
    assert "service_regimes" not in raw["runtime"].get("kernels", {})
    path = tmp_path / "normalized.json"
    export_resource_profile(metadata["resource_service_profile"], path=path)
    cost = CostModel(parse_case(apply_resource_profile(raw, path)))
    context = sparse_context(geometry={"q_len": query})
    model = cost.select_resource_service(context)
    assert model.rate_scope == "per_owner" and model.active_units == OWNER_COUNT
    assert model.aggregate_rate == pytest.approx(OWNER_COUNT * model.per_owner_rate)
    reference = select_sparse_supply(sparse_context(mix_block_num=None), query=query)
    work = sparse_mte1_bytes(query, 0, query, owners=OWNER_COUNT)["critical_qk_pv_bytes"]
    assert reference["query_critical_bytes"] == work
    assert model.per_owner_rate == pytest.approx(measured["resource_services"][0]["rate_bytes_per_s"])
    op = copy_op(work, role="mte1", phase="prefill")
    bound = cost.bind_operator_resource_service(op, context, resource_role="mte1")
    receipt = dict(bound.metadata)["resource_service"]
    assert receipt["conditions"] == model.conditions and receipt["fixed_once"] is True
    assert receipt["model_id"] == model.model_id and receipt["source"] == model.provenance["source"]
    engine = Engine([Resource("mte1", 2 * model.per_owner_rate / 1000, unit="bytes")])
    done = bound_program(cost, bound, "mte1", work).start(engine, "copy", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced == {"mte1": work}
    assert done.time * 1000 == pytest.approx(model.fixed_us + work * 1e6 / model.per_owner_rate)


def test_supply_hardware_mismatch_remains_a_selection_gap():
    raw = composed_supply(hardware=FOREIGN_HARDWARE)
    metadata = raw["runtime"]["metadata"]
    assert metadata["profiler_measured_supply"]["applicability_gaps"][0]["gap"] == "hardware_mismatch"
    assert metadata["resource_service_profile"]["hardware"] == PROFILE_HARDWARE
    with pytest.raises(ValueError, match="conditions_have_no_match"):
        CostModel(parse_case(raw)).select_resource_service(sparse_context(hardware=FOREIGN_HARDWARE))


def test_prediction_ignores_removed_or_amplified_diagnostic_rules():
    raw = composed_supply()
    baseline = predict(raw, scope="body")
    replay = composed_supply(legacy_diagnostics=True)
    assert replay["runtime"]["metadata"]["profiler_measured_supply"]["legacy_diagnostics_replay"]
    rules = replay["runtime"]["kernels"]["service_regimes"]
    assert rules == import_profiler_supply(SUPPLY, recipe="output", hardware=PROFILE_HARDWARE,
                                           owners=24)["service_regimes"]
    assert rules[0]["execution"]["kind"] == "mac_service"
    raw["runtime"].setdefault("kernels", {})["service_regimes"] = deepcopy(rules)
    raw["runtime"]["kernels"]["service_regimes"][0]["response"]["fixed_us"] = 1e6
    changed = predict(raw, scope="body")
    assert changed["operator_occurrences"] == baseline["operator_occurrences"]
    assert changed["execution"]["serviced"] == baseline["execution"]["serviced"]
    assert changed["duration_ms"] == pytest.approx(baseline["duration_ms"])


def test_cost_model_keeps_primitive_duration_in_explicit_diagnostics():
    parsed = parse_case(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    op = linear_work(plan, parsed.model, parsed.runtime, parsed.machine).ops[0]
    without_store = CostModel(parsed)
    assert without_store.lookup_diagnostic(op, "none") is None
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parsed, op, "none"), duration_ms=10, run_id="measured",
                     repeat_index=0, source="independent timing oracle")
        cost = CostModel(parsed, store)
        assert cost.price(op, "none").timing is None
        assert cost.summary()["diagnostic_primitive_hits"] == 0
        assert cost.lookup_diagnostic(op, "none")["timing"].duration_ms == 10
        assert cost.summary()["operator_occurrences"]["measured_primitive"] == 0


@pytest.mark.requires_capture_mount(str(COLLECTIVE_SOURCE))
@pytest.mark.requires_capture_mount(str(COLLECTIVE_ENVIRONMENT))
def test_native_collective_price_is_local_owner_service():
    image = next(token.split("=", 1)[1] for token in COLLECTIVE_ENVIRONMENT.read_text().split()
                 if token.startswith("image_id="))
    record = fit_a5_native_local_service(
        collective="all_gather", stack=A5_NATIVE_GROUP2["stack"], cann=A5_NATIVE_GROUP2["cann"],
        deployment_image_id=image, group_size=2)
    binding = bind_a5_local_service(record, domain=record["domain"], resource_role="aicpu_child")
    assert binding["rate_scope"] == binding["work_scope"] == "per_owner"
    assert binding["active_units"] == 2 and binding["fixed_us"] == pytest.approx(39.78255936532216)
    assert binding["fixed_once"] and not binding["link_transit_identified"]
    op = OpSpec("ag", "all_gather", "collective_expand", 0, 0, ("r",),
                (1,), (1,), (1,), (1,), link_bytes=65536, dtype="bf16", recipe="all_gather")
    attached = bind_a5_local_service_operator(record, op, domain=record["domain"],
                                              resource_role="aicpu_child")
    selected = dict(attached.metadata)["resource_service"]
    assert selected["service_id"] == binding["service_id"] == record["service"]["id"]
    assert selected["actual_phase"] == op.phase
