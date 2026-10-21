"""Framework work is explicit, scoped and served by the existing engine."""
import math

import pytest

from mindie_sim.a3_resource_engine import ResourceEngine, ResourceSpec
from mindie_sim.framework_tasks import (
    AffineCost,
    FrameworkProfile,
    FrameworkProfileError,
    add_framework_task,
    ensure_framework_resources,
)


def _engine():
    return ResourceEngine({"device": ResourceSpec("device", 1.0, "ms")})


def _add(engine, profile, kind, prefix, *, replica=0, **kwargs):
    return add_framework_task(
        engine, profile, kind, requests=1, tokens=4, kernels=2,
        prefix=prefix, replica=replica, **kwargs,
    )


def test_affine_time_uses_counts_and_microsecond_conversion():
    cost = AffineCost(10, 2, 0.5, 3)
    assert cost.duration_ms(requests=2, tokens=8, kernels=4) == pytest.approx(0.030)
    assert cost.duration_ms() == pytest.approx(0.010)
    assert AffineCost().is_zero
    assert not cost.is_zero


def test_profile_defaults_and_roundtrip_are_explicit():
    profile = FrameworkProfile.from_mapping({
        "source": "synthetic unit-test coefficients; not a CPU measurement",
        "cpu_scope": "replica",
        "schedule": {"fixed_us": 10, "per_request_us": 2},
        "submit": {"per_kernel_us": 3},
    })
    assert FrameworkProfile.from_mapping(profile.to_dict()) == profile
    assert profile.stream_scope == "process"
    assert profile.schedule.duration_ms(requests=2) == pytest.approx(0.014)
    assert profile.input.is_zero
    assert not profile.is_zero
    assert FrameworkProfile(source="zero overhead fixture").is_zero


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "2", None])
def test_device_launch_is_a_strict_separate_device_cost(value):
    with pytest.raises(FrameworkProfileError, match="device_launch_us"):
        FrameworkProfile(source="fixture", device_launch_us=value)
    with pytest.raises(FrameworkProfileError, match="device_launch_us"):
        FrameworkProfile.from_mapping({"source": "fixture", "device_launch_us": value})


def test_explicit_device_launch_activates_otherwise_zero_profile():
    profile = FrameworkProfile.from_mapping({"source": "fixture", "device_launch_us": 2.5})
    assert not profile.is_zero
    assert profile.to_dict()["device_launch_us"] == 2.5
    assert FrameworkProfile.from_mapping(profile.to_dict()) == profile


@pytest.mark.parametrize("field", [
    "fixed_us", "per_request_us", "per_token_us", "per_kernel_us",
])
@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "2", None])
def test_coefficient_values_are_strict(field, value):
    with pytest.raises(FrameworkProfileError, match="framework_value_invalid"):
        AffineCost(**{field: value})
    with pytest.raises(FrameworkProfileError, match="framework_value_invalid"):
        FrameworkProfile.from_mapping({"source": "fixture", "input": {field: value}})


@pytest.mark.parametrize("payload", [
    {}, {"source": ""}, {"source": "   "}, {"source": True},
    {"source": "fixture", "unknown": 1},
    {"source": "fixture", "cpu_scope": "host"},
    {"source": "fixture", "stream_scope": True},
    {"source": "fixture", "schedule": {"fixed_ms": 1}},
    {"source": "fixture", "schedule": None},
    {"source": "fixture", "schedule": 0},
    [], None,
])
def test_invalid_profile_mapping_is_rejected(payload):
    with pytest.raises(FrameworkProfileError):
        FrameworkProfile.from_mapping(payload)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "2"])
@pytest.mark.parametrize("field", ["requests", "tokens", "kernels"])
def test_duration_rejects_invalid_counts_even_for_zero_coefficients(field, value):
    with pytest.raises(FrameworkProfileError):
        AffineCost().duration_ms(**{field: value})


def test_duration_rejects_finite_inputs_whose_product_overflows():
    with pytest.raises(FrameworkProfileError, match="duration_not_finite"):
        AffineCost(per_token_us=1e308).duration_ms(tokens=1e308)


def test_resource_registration_reuses_process_scopes_without_resetting_service():
    engine = _engine()
    profile = FrameworkProfile(source="fixture", schedule=AffineCost(fixed_us=3000))
    first = ensure_framework_resources(engine, profile, 0)
    cpu = engine.resources["cpu"]
    task = _add(engine, profile, "schedule", "r0")
    assert engine.run()["completed_at_ms"][task] == pytest.approx(3.0)
    assert ensure_framework_resources(engine, profile, 7) == first
    assert first == {"cpu": "cpu", "stream": "dma"}
    assert engine.resources["cpu"] is cpu
    assert engine.serviced["cpu"] == pytest.approx(3.0)
    assert engine.resources["cpu"].unit == "ms"
    assert engine.resources["cpu"].scope == "process"


@pytest.mark.parametrize("resource,kind", [("cpu", "schedule"), ("dma", "stream")])
def test_existing_prepare_shared_label_is_reused_for_process_costs(resource, kind):
    engine = _engine()
    existing = ResourceSpec(resource, 1.0, "ms")
    engine.resources[resource] = existing
    engine.serviced[resource] = 0.0
    engine.add_task("existing-prepare", {resource: 2.0})
    profile = FrameworkProfile(source="fixture", **{kind: AffineCost(fixed_us=2000)})
    framework = _add(engine, profile, kind, "r0")
    assert engine.resources[resource] is existing
    result = engine.run()
    assert result["completed_at_ms"]["existing-prepare"] == pytest.approx(4.0)
    assert result["completed_at_ms"][framework] == pytest.approx(4.0)
    assert result["serviced"][resource] == pytest.approx(4.0)


@pytest.mark.parametrize("conflicting", [
    ResourceSpec("dma", 2.0, "ms", scope="process"),
    ResourceSpec("dma", 1.0, "bytes", scope="process"),
    ResourceSpec("dma", 1.0, "ms", scope="replica"),
    ResourceSpec("dma", 1.0, "ms", scope="process", policy="fifo"),
])
def test_resource_conflict_is_atomic_and_preserves_original(conflicting):
    engine = _engine()
    engine.resources["dma"] = conflicting
    before = dict(engine.resources)
    with pytest.raises(FrameworkProfileError, match="framework_resource_conflict:dma"):
        ensure_framework_resources(engine, FrameworkProfile(source="fixture"), 0)
    assert engine.resources == before
    assert "cpu" not in engine.resources
    assert engine.resources["dma"] is conflicting


@pytest.mark.parametrize("replica", [-1, True, 0.5, "0"])
def test_resource_replica_index_is_strict(replica):
    with pytest.raises(FrameworkProfileError, match="framework_replica_invalid"):
        ensure_framework_resources(_engine(), FrameworkProfile(source="fixture"), replica)


def test_process_cpu_tasks_share_capacity_across_replicas():
    engine = _engine()
    profile = FrameworkProfile(source="fixture", schedule=AffineCost(fixed_us=4000))
    a = _add(engine, profile, "schedule", "r0", replica=0)
    b = _add(engine, profile, "schedule", "r1", replica=1)
    result = engine.run()
    # A unit-capacity CPU splits service 0.5/0.5; each needs four ms of work.
    assert result["completed_at_ms"][a] == pytest.approx(8.0)
    assert result["completed_at_ms"][b] == pytest.approx(8.0)
    assert result["serviced"]["cpu"] == pytest.approx(8.0)
    assert engine.tasks[a].service_intervals == [("cpu", 0.0, 8.0, 0.5)]


def test_replica_cpu_scopes_allow_independent_service():
    engine = _engine()
    profile = FrameworkProfile(
        source="fixture", cpu_scope="replica", stream_scope="replica",
        schedule=AffineCost(fixed_us=4000),
    )
    a = _add(engine, profile, "schedule", "r0", replica=0)
    b = _add(engine, profile, "schedule", "r1", replica=1)
    result = engine.run()
    assert result["completed_at_ms"][a] == pytest.approx(4.0)
    assert result["completed_at_ms"][b] == pytest.approx(4.0)
    assert ensure_framework_resources(engine, profile, 1) == {
        "cpu": "r1:cpu", "stream": "r1:dma",
    }
    assert "cpu" not in engine.resources


def test_cpu_and_stream_resources_do_not_block_each_other():
    engine = _engine()
    profile = FrameworkProfile(
        source="fixture", schedule=AffineCost(fixed_us=3000),
        stream=AffineCost(fixed_us=4000),
    )
    cpu = _add(engine, profile, "schedule", "a")
    stream = _add(engine, profile, "stream", "b")
    result = engine.run()
    assert result["completed_at_ms"][cpu] == pytest.approx(3.0)
    assert result["completed_at_ms"][stream] == pytest.approx(4.0)
    assert result["serviced"]["cpu"] == pytest.approx(3.0)
    assert result["serviced"]["dma"] == pytest.approx(4.0)


def test_single_request_lifecycle_matches_hand_computed_dependency_chain():
    profile = FrameworkProfile(
        source="synthetic coefficients; not measured",
        schedule=AffineCost(fixed_us=10, per_request_us=2, per_token_us=1),
        input=AffineCost(fixed_us=3, per_token_us=2),
        submit=AffineCost(fixed_us=2, per_kernel_us=5),
        output=AffineCost(fixed_us=4, per_request_us=2),
        stream=AffineCost(fixed_us=5),
    )
    engine = _engine()
    schedule = _add(engine, profile, "schedule", "iteration0")
    inputs = _add(engine, profile, "input", "iteration0", dependencies=(schedule,))
    submit = _add(engine, profile, "submit", "iteration0", dependencies=(inputs,))
    engine.add_task("forward", {"device": 0.100}, dependencies=(submit,))
    output = _add(engine, profile, "output", "iteration0", dependencies=("forward",))
    stream = _add(engine, profile, "stream", "iteration0", dependencies=(output,))
    result = engine.run()
    # All times in ms: .016 + .011 + .012 + .100 + .006 + .005 = .150.
    assert result["completed_at_ms"][stream] == pytest.approx(0.150)
    assert result["serviced"]["cpu"] == pytest.approx(0.045)
    assert result["serviced"]["dma"] == pytest.approx(0.005)
    assert engine.tasks[submit].owner == "iteration0:framework"
    assert engine.tasks[submit].metadata["source"] == profile.source
    assert engine.tasks[submit].metadata["kind"] == "submit"
    assert engine.tasks[submit].metadata["framework_counts"]["kernels"] == 2


def test_zero_work_gate_preserves_dependencies_and_release_frontier():
    engine = _engine()
    engine.add_task("forward", {"device": 5.0})
    task = _add(
        engine, FrameworkProfile(source="zero fixture"), "output", "iteration0",
        dependencies=("forward",), release_ms=20.0, name="last-output",
    )
    assert task == "iteration0:framework:last-output"
    assert engine.tasks[task].demands == {}
    result = engine.run()
    assert result["completed_at_ms"][task] == pytest.approx(20.0)
    assert result["serviced"]["cpu"] == 0.0
    assert result["serviced"]["dma"] == 0.0


@pytest.mark.parametrize("field,value", [
    ("kind", "launch"), ("prefix", ""), ("prefix", True),
    ("name", ""), ("name", False),
    ("release_ms", -1), ("release_ms", math.inf), ("release_ms", True),
    ("dependencies", "forward"), ("dependencies", (None,)),
])
def test_invalid_task_configuration_fails_before_registering_resources(field, value):
    engine = _engine()
    kwargs = dict(
        kind="schedule", requests=1, tokens=4, kernels=2,
        prefix="iteration0", replica=0,
    )
    kwargs[field] = value
    with pytest.raises(FrameworkProfileError):
        add_framework_task(engine, FrameworkProfile(source="fixture"), **kwargs)
    assert set(engine.resources) == {"device"}
    assert not engine.tasks
