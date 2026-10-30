"""Independent HcPre/HcPost source ownership, service and capacity controls."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.family_hc import (
    POST_PROFILE, PRE_PROFILE, post_buffer_bytes, post_work,
    pre_geometry, pre_phase_work, selected_plan,
)
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


RATES = {
    "hbm": 1e12, "mte2_aic": 1e12, "mte2_aiv": 1e12, "mte3": 1e12,
    "fixpipe": 1e12, "vector": 1e12, "mte1": 1e12,
}
UNITS = {
    "vector": 40, "mte2_aic": 20, "mte2_aiv": 40, "mte3": 40,
    "fixpipe": 20, "mte1": 20, "hbm": 1, "cube": 20,
}
PEAKS = {"hf32": 1e14, "fp32": 1e14}


def _op(view, profile):
    return OpSpec(
        "hc", "hc_pre", "prefill", 0, 0, (), (), (), (), (), dtype="bf16",
        metadata=(
            ("fusion_tensor_view", view),
            ("fusion_external_outputs", ()),
            ("fusion_semantic_ids", ()),
            ("fusion_phases", ()),
            ("fusion_boundary", {"traffic_profile": profile}),
        ))


def _pre(tokens=1, hidden=64, **extra):
    return {"tokens": tokens, "hidden": hidden, "hc": 4, "dtype": "bf16",
            "coefficient_dtype": "fp32", "sinkhorn_iters": 3, **extra}


def _post(tokens=1, hidden=2048, **extra):
    return {"tokens": tokens, "hidden": hidden, "hc": 4, "dtype": "bf16",
            "coefficient_dtype": "fp32", **extra}


def _geometry(*, units=None, ub=196608, ub_units=None, replace_units=False):
    units = dict(units or {}) if replace_units else {**UNITS, **(units or {})}
    return Geometry(20, units, {"ub": ub},
                    {"ub": ub_units or units["vector"]})


def _run(plan, op, *, geometry, rates=None, units=None, peaks=None,
         service_modes=None, ready_ms=0, trace_limit=0,
         replace_units=False, replace_rates=False):
    rates = dict(rates or {}) if replace_rates else {**RATES, **(rates or {})}
    units = dict(units or {}) if replace_units else {**UNITS, **(units or {})}
    names = {role: {"cube": "matrix", "hbm": "memory"}.get(role, role)
             for role in units}
    engine_rates = {names[role]: rate for role, rate in rates.items()
                    if role in names and role != "cube"}
    engine_rates["matrix"] = 1
    resources = [Resource(name, engine_rates[name], units=units[role])
                 for role, name in names.items() if role != "cube"]
    resources.append(Resource("matrix", 1, units=units["cube"]))
    pools = {"ub": geometry.buffers["ub"]}
    pool_units = {"ub": geometry.buffer_units["ub"]}
    engine = Engine(resources, pools=pools, pool_units=pool_units,
                    trace_limit=trace_limit)
    context = RecipeContext(
        plan.work.apply(op), {}, names, engine_rates, geometry.compute_units,
        resource_units={**units, "cube": units["cube"]},
        compute_peaks={**PEAKS, **(peaks or {})},
        pool_names={"ub": "ub"}, pool_capacities=pools, pool_units=pool_units,
        service_modes=service_modes or {},
    )
    ready = engine.submit("ready", release=ready_ms)
    done = plan.lower(context).start(engine, "hc", ready)
    if ready_ms:
        engine.advance(horizon=ready_ms - 1)
        assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    return engine, done.time - ready_ms


def _serviced_work(engine):
    return {name: amount for name, amount in engine.serviced.items() if amount}


def test_hc_post_d2048_source_read_write_bytes_and_ub_capacity():
    view = _post(1, 2048)
    geometry = _geometry()
    plan = selected_plan(_op(view, POST_PROFILE), geometry)
    # x 4 KiB + four residual rows 16 KiB + post/comb coefficients 80 B.
    assert plan.work.resources["mte2_aiv"] == 20560
    assert plan.work.resources["mte3"] == 16384
    assert "mte2_aic" not in plan.work.resources
    assert post_buffer_bytes(view) == 164640
    engine, _ = _run(plan, _op(view, POST_PROFILE), geometry=geometry,
                     ready_ms=7, trace_limit=100)
    assert _serviced_work(engine)["mte2_aiv"] == 20560
    assert engine.summary()["peak_allocated_per_unit"] == {"ub:unit0": 164640}
    assert not any(engine.allocated.values()) and not engine.unit_allocated
    assert engine.summary()["live_tasks"] == 0


def test_hc_post_private_aiv_read_is_required_and_service_aliases_are_pinned():
    view = _post(1, 2048)
    op = _op(view, POST_PROFILE)
    geometry = _geometry()
    plan = selected_plan(op, geometry)
    slow_aiv, baseline = _run(
        plan, op, geometry=geometry,
        rates={"mte2_aiv": 20560, "vector": 1e12, "mte3": 1e12,
               "hbm": 1e12})
    faster_aiv, quick = _run(
        plan, op, geometry=geometry,
        rates={"mte2_aiv": 2 * 20560, "vector": 1e12, "mte3": 1e12,
               "hbm": 1e12})
    _, ignored_aic = _run(
        plan, op, geometry=geometry,
        rates={"mte2_aiv": 20560, "mte2_aic": 1,
               "vector": 1e12, "mte3": 1e12, "hbm": 1e12})
    assert baseline > quick
    assert ignored_aic == pytest.approx(baseline)
    assert _serviced_work(slow_aiv)["mte2_aiv"] == 20560
    assert _serviced_work(faster_aiv)["mte2_aiv"] == 20560
    assert not any("mte2_aic" in name for name in _serviced_work(slow_aiv))

    alias_geometry = _geometry()
    alias_plan = selected_plan(op, alias_geometry)
    alias, _ = _run(
        alias_plan, op, geometry=alias_geometry,
        rates={"read_shared": 1e12, "mte2_aiv": 1e12},
        units={"read_shared": 40},
        service_modes={"mte2_aiv": ("read_shared", 1e15)},
        trace_limit=100)
    assert _serviced_work(alias)["read_shared"] == 20560
    assert all(task["placement"].get("read_shared") == (0,)
               for task in alias.trace if "load_x" in task["task"])


def test_hc_post_generic_only_and_under_capacity_controls():
    view = _post(1, 2048)
    op = _op(view, POST_PROFILE)
    generic_units = {key: value for key, value in UNITS.items()
                     if key not in {"mte2_aic", "mte2_aiv"}}
    generic_units.update(mte2=40)
    generic = _geometry(units=generic_units, replace_units=True)
    plan = selected_plan(op, generic)
    assert plan.work.resources["mte2"] == 20560
    generic_rates = {key: value for key, value in RATES.items()
                     if key not in {"mte2_aic", "mte2_aiv"}}
    generic_rates["mte2"] = 1e12
    engine, _ = _run(
        plan, op, geometry=generic, ready_ms=3,
        units=generic_units, replace_units=True,
        rates=generic_rates, replace_rates=True)
    assert _serviced_work(engine)["mte2"] == 20560
    assert not any("mte2_aiv" in name for name in _serviced_work(engine))

    with pytest.raises(ModelingError, match="ub_capacity_insufficient"):
        selected_plan(op, _geometry(ub=post_buffer_bytes(view) - 1))

    # A plan without any declared private read pipe must fail, not charge HBM
    # only. Preparation is lazy, so the lowering is the actual-execution check.
    no_read_units = {key: value for key, value in generic_units.items()
                     if key != "mte2"}
    no_read = _geometry(units=no_read_units, replace_units=True)
    plan = selected_plan(op, no_read)
    no_read_rates = {key: value for key, value in generic_rates.items()
                     if key != "mte2"}
    with pytest.raises(ModelingError, match="requires_read_service"):
        _run(plan, op, geometry=no_read, units=no_read_units,
             replace_units=True, rates=no_read_rates, replace_rates=True)


def test_hc_pre_hf32_projection_and_fp32_gram_are_separate_rates():
    view = _pre(1, 64)
    op = _op(view, PRE_PROFILE)
    geometry = _geometry()
    plan = selected_plan(op, geometry)
    physical = pre_geometry(view, geometry)
    total = 2 * physical.padded_rows * physical.width * (24 + 16)
    assert sum(plan.work.matrix.values()) == total
    assert plan.work.matrix == {
        "hf32": 2 * physical.padded_rows * physical.width * 24,
        "fp32": 2 * physical.padded_rows * physical.width * 16,
    }
    _, base = _run(plan, op, geometry=geometry, peaks={"hf32": 1e9, "fp32": 1e9})
    _, slow_hf32 = _run(plan, op, geometry=geometry,
                        peaks={"hf32": 1e8, "fp32": 1e9})
    _, slow_fp32 = _run(plan, op, geometry=geometry,
                        peaks={"hf32": 1e9, "fp32": 1e8})
    assert slow_hf32 > base and slow_fp32 > base
    names = {role: {"cube": "matrix", "hbm": "memory"}.get(role, role)
             for role in UNITS}
    rates = {name: 1e12 for role, name in names.items() if role != "cube"}
    rates["matrix"] = 1
    with pytest.raises(ModelingError, match="compute_precision_rate_required:hf32"):
        plan.lower(RecipeContext(
            plan.work.apply(op), {}, names, rates, 20,
            resource_units={**UNITS, "cube": 20},
            compute_peaks={"fp32": 1e14}))


def test_hc_pre_fixpipe_writes_outputs_and_is_required():
    view = _pre(1, 64)
    op = _op(view, PRE_PROFILE)
    geometry = _geometry()
    plan = selected_plan(op, geometry)
    phases = pre_phase_work(view, geometry)
    physical = pre_geometry(view, geometry)
    expected = phases["projection_and_gram"].resources["fixpipe"]
    assert expected == (view["tokens"] * physical.k_partitions
                        * (24 + 16) * 4)
    assert "mte3" not in phases["projection_and_gram"].resources
    _, fast = _run(plan, op, geometry=geometry,
                   rates={"fixpipe": 1e12, "vector": 1e12, "mte3": 1e12})
    _, slow = _run(plan, op, geometry=geometry,
                   rates={"fixpipe": 160, "vector": 1e12, "mte3": 1e12})
    assert slow > fast
    without = _geometry(
        units={key: value for key, value in UNITS.items() if key != "fixpipe"},
        replace_units=True)
    with pytest.raises(ModelingError, match="requires_fixpipe_output"):
        selected_plan(op, without)


def test_hc_pre_part2_owner_rows_and_parameter_reads():
    geometry = _geometry()
    view = _pre(1, 64)
    g = pre_geometry(view, geometry)
    assert (g.row_former, g.owner_count) == (1, 1)
    phases = pre_phase_work(view, geometry)
    assert phases["coefficients"].resources["mte2_aiv"] == (
        g.owner_count * 24 * 4 + view["tokens"] * g.k_partitions * 40 * 4
        + view["tokens"] * 3 * 4)
    engine, _ = _run(selected_plan(_op(view, PRE_PROFILE), geometry),
                     _op(view, PRE_PROFILE), geometry=geometry,
                     trace_limit=100)
    work_tasks = [task for task in engine.trace
                  if ("coefficients" in task["task"] or ":mix" in task["task"])
                  and "vector" in task["placement"]]
    assert work_tasks and all(task["placement"]["vector"] == (0,)
                              for task in work_tasks)

    view = _pre(41, 64)
    g = pre_geometry(view, geometry)
    assert (g.row_former, g.owner_count) == (2, 21)
    phases = pre_phase_work(view, geometry)
    # 41 tokens on 40 AIVs use 21 owners: 20 owners x 2 rows and one x 1.
    assert g.owner_count * 24 * 4 == 2016
    assert phases["coefficients"].resources["mte2_aiv"] == (
        2016 + 41 * g.k_partitions * 40 * 4 + 41 * 3 * 4)
    engine, _ = _run(selected_plan(_op(view, PRE_PROFILE), geometry),
                     _op(view, PRE_PROFILE), geometry=geometry,
                     trace_limit=200)
    owners = set()
    for task in engine.trace:
        if (("coefficients" in task["task"] or ":mix" in task["task"])
                and "vector" in task["placement"]):
            owners.update(task["placement"]["vector"])
    assert owners == set(range(21))
    assert 39 not in owners


@pytest.mark.parametrize("profile,view,total_owners", [
    (PRE_PROFILE, _pre(1, 64), 1),
    (PRE_PROFILE, _pre(41, 64), 21),
    (PRE_PROFILE, _pre(1024, 7168), 40),
    (POST_PROFILE, _post(1, 2048), 1),
    (POST_PROFILE, _post(41, 2112), 40),
    (POST_PROFILE, _post(1024, 7168), 40),
])
def test_hc_source_work_equals_actual_engine_service_for_owner_tails(
        profile, view, total_owners):
    geometry = _geometry()
    op = _op(view, profile)
    plan = selected_plan(op, geometry)
    engine, _ = _run(plan, op, geometry=geometry, ready_ms=5)
    serviced = _serviced_work(engine)
    names = {role: {"cube": "matrix", "hbm": "memory"}.get(role, role)
             for role in UNITS}
    for role, amount in plan.work.resources.items():
        assert serviced.get(names.get(role, role), 0) == pytest.approx(amount), role
    if profile == PRE_PROFILE:
        matrix = sum(1000 * flops / PEAKS[dtype]
                     for dtype, flops in plan.work.matrix.items())
        assert serviced["matrix"] == pytest.approx(matrix)
        assert pre_geometry(view, geometry).owner_count == total_owners
    else:
        units = geometry.resource_units["vector"]
        used = min(view["tokens"], units)
        assert used == total_owners
    assert engine.summary()["live_tasks"] == 0
    assert not any(engine.allocated.values())


@pytest.mark.parametrize("tokens", [1, 41, 257, 1024, 5121])
def test_hc_pre_stage1_owner_service_matches_source_m_k_loop(tokens):
    geometry = _geometry()
    view = _pre(tokens, 7168)
    op = _op(view, PRE_PROFILE)
    engine, _ = _run(selected_plan(op, geometry), op, geometry=geometry,
                     ready_ms=2, trace_limit=1000)
    # Independently enumerate source host partitions and paired-vector rows.
    m_parts = min(20, (tokens + 255) // 256)
    m_span = ((tokens + m_parts - 1) // m_parts + 15) // 16 * 16
    k_span = ((28672 + 20 // m_parts - 1) // (20 // m_parts) + 255) // 256 * 256
    k_parts = (28672 + k_span - 1) // k_span
    expected_cast, expected_projection = {}, {}
    for m in range(m_parts):
        rows = min(m_span, tokens - m * m_span)
        if rows <= 0:
            continue
        for k in range(k_parts):
            width = min(k_span, 28672 - k * k_span)
            aic = m * k_parts + k
            for offset in range(0, rows, min(256, m_span)):
                panel_rows = min(256, m_span, rows - offset)
                for subcore, count in enumerate(((panel_rows + 1) // 2, panel_rows // 2)):
                    if count:
                        owner = 2 * aic + subcore
                        expected_cast[owner] = expected_cast.get(owner, 0) + count * width
            expected_projection[aic] = (
                4 * rows * width + ((rows + 255) // 256) * width * 24 * 4,
                rows * 40 * 4)
    cast = [task for task in engine.trace if ":cast_owner" in task["task"]]
    projection = [task for task in engine.trace if ":projection_owner" in task["task"]]
    assert {task["placement"]["vector"][0]: task["serviced"]["vector"]
            for task in cast} == expected_cast
    assert {task["placement"]["matrix"][0] for task in projection} == set(expected_projection)
    for task in projection:
        owner = task["placement"]["matrix"][0]
        assert (task["serviced"]["mte2_aic"], task["serviced"]["fixpipe"]) == pytest.approx(
            expected_projection[owner])
    assert all(task["placement"]["mte2_aiv"] == task["placement"]["vector"]
               == task["placement"]["mte3"] for task in cast)
    assert sum(expected_cast.values()) == tokens * 28672
    if tokens == 1:
        assert set(expected_cast) == set(range(0, 38, 2))
        assert expected_cast[36] == 1024
        assert expected_cast[0] == 1536


def test_hc_post_shared_store_pipe_preserves_each_owner_rows():
    view = _post(41, 2048)
    op = _op(view, POST_PROFILE)
    geometry = _geometry(units={"mte3": 20})
    engine, duration = _run(
        selected_plan(op, geometry), op, geometry=geometry,
        units={"mte3": 20},
        rates={**dict.fromkeys(RATES, 1e30), "mte3": 20 * 16384})
    # Owners 0/1 share pipe0: two rows + one row. The other pipes each write
    # two rows; replacing that distribution with a 20-pipe average is wrong.
    assert duration == pytest.approx(3)
    assert engine.serviced["mte3"] == 41 * 16384
    assert not any(engine.allocated.values())


def test_hc_post_source_does_not_claim_unhandled_vector_repeat_tail():
    geometry = _geometry()
    for view in (_post(1, 2051), _post(1, 2048, hc=9)):
        with pytest.raises(ModelingError, match="d_multiple64_and_hc_at_most8"):
            selected_plan(_op(view, POST_PROFILE), geometry)
    view = _post(43, 2112)
    op = _op(view, POST_PROFILE)
    engine, _ = _run(selected_plan(op, geometry), op, geometry=geometry)
    assert engine.serviced["mte3"] == 43 * 2112 * 4 * 2
