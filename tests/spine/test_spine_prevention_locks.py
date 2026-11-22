"""Live end-to-end single-spine execution-trace locks and physics regressions.

[Role] Enforce Lock C (100% governed operator dispatch, single per-die
  ``AdmissionMemoryView`` per step, and exact canonical ``RequestView`` tuple
  propagation across Dense GQA, MLA/MQSMLA, GDN/linear, and MoE+MC2 in ragged
  and padded ``TND``/``BSND`` layouts) plus clock-domain, bounded-streaming,
  mixed-HBM, cohort-split, and ``DAV_2201`` guard invariants.
[Invariants] Tier-0 in-memory execution with zero device contexts or host I/O.
  All models and SKUs sweep via canonical registries and ``_support.axes``.
[Timeouts & Locks] Bounded 1-layer replays and synthetic micro-graphs; no locks.
[Side-Effect Policy] Pure read-only simulation assertions; monkeypatches are
  automatically restored per test.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial, wraps
import hashlib
from inspect import isroutine
import json
from types import FunctionType, MethodType, SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.linear_cases import execute_one, gdn_decode_source_case
from mindie_sim import graph_costs
from mindie_sim.compose import compose_case
from mindie_sim.observations.multilevel import verify_capture_file_consistency
from mindie_sim.observations.comparison import (
    apply_capture_observed_clock,
    extract_observed_aicore_freq_mhz,
)
from mindie_sim.prediction.execution.context import _ContextMixin
from mindie_sim.prediction.execution.submission import is_zero_kernel_view
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan
from mindie_sim.prediction.operators import attention, linear_attention
from mindie_sim.prediction.operators.attention.attention_pipeline import TiledAttentionRecipe
from mindie_sim.prediction.operators.attention.common import AttentionContract, sequence_layout_spans
from mindie_sim.prediction.operators.matrix.refill_steady import peer_group_can_compact
from mindie_sim.prediction.operators.paged_mla import indexed_quant_attention
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.prediction.runtime import engine_submit, l2_residency
from mindie_sim.prediction.runtime.engine import Engine
from mindie_sim.prediction.runtime.engine_types import Resource
from mindie_sim.prediction.runtime.memory_physics import HbmBusSpec
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import ComputeArchSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.stack_registry import LoweringRequest
from scripts._arch_governance_runtime import (
    _NON_DEVICE_INFRA_RECIPES, _OWNER_MODULE_PREFIXES, _binding_module, _is_governed_module,
)

pytestmark = pytest.mark.tier0

SEQUENCE_FAMILIES = frozenset({
    "attention", "mla", "paged_mla", "mla_prolog", "indexer", "index_pool",
    "compressed_attention", "recurrent_local", "recurrent_output",
    "triangular_solve", "gdn_prepare", "gdn_state", "gdn_output", "gdn_gating",
})
RAGGED_TND_VIEWS = (
    RequestView("req_short", q_len=1, context=128, samples=True, phase="decode"),
    RequestView("req_long", q_len=1, context=512, samples=True, phase="decode"),
    RequestView("req_pad", q_len=1, context=0, samples=False, phase="decode", padding=True),
)
RAGGED_BSND_VIEWS = (
    RequestView(
        "req_a", q_len=1, context=64, samples=True, phase="decode",
        padded_q_len=4, padded_seq_len=256,
    ),
    RequestView(
        "req_b", q_len=2, context=192, samples=True, phase="decode",
        padded_q_len=4, padded_seq_len=256,
    ),
    RequestView(
        "req_pad", q_len=0, context=0, samples=False, phase="decode",
        padding=True, padded_q_len=4, padded_seq_len=256,
    ),
)
CLOCK_SCALE_RATIO = 0.75
BUS_READ_BPMS = 1024.0
BUS_WRITE_BPMS = 1024.0
BUS_REBIND_READ_BPMS = 128.0
BUS_ALPHA = 0.5
TOTAL_BYTES = 16384.0
ASYMM_BYTES = 8192.0


def _is_allowed_recipe_owner(recipe) -> bool:
    cls = recipe if isinstance(recipe, type) else type(recipe)
    if (cls.__module__, cls.__name__) in _NON_DEVICE_INFRA_RECIPES:
        return True
    return _binding_module(recipe).startswith(_OWNER_MODULE_PREFIXES)


def _program_owners(program, probe):
    pending = [program]
    seen = set()
    while pending:
        callback = pending.pop()
        if id(callback) in seen:
            continue
        seen.add(id(callback))
        if getattr(callback, "_lock_c_probe", None) is probe:
            pending.append(callback.__wrapped__)
            continue
        if isinstance(callback, partial):
            pending.append(callback.func)
            continue
        if isinstance(callback, MethodType):
            callback = callback.__func__
        if not isroutine(callback):
            callback = type(callback).__call__
        module = getattr(callback, "__globals__", {}).get(
            "__name__", getattr(callback, "__module__", type(callback).__module__))
        yield module
        wrapped = getattr(callback, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        for cell in getattr(callback, "__closure__", None) or ():
            try:
                captured = cell.cell_contents
            except ValueError:
                continue
            if isinstance(captured, partial) or isroutine(captured):
                pending.append(captured)


@pytest.fixture
def traced_plan_programs(monkeypatch):
    """Check callbacks at invocation, including plans invoked by another plan."""
    executed = set()
    get_attribute = OperatorPlan.__getattribute__

    def traced_get_attribute(self, name):
        callback = get_attribute(self, name)
        if name != "program" or getattr(callback, "_lock_c_probe", None) is executed:
            return callback

        @wraps(callback)
        def checked(context):
            outside = [module for module in _program_owners(callback, executed)
                       if not _is_governed_module(module)]
            assert not outside, f"OperatorPlan.program outside governed packages: {outside}"
            executed.add(context.op.family)
            return callback(context)

        checked._lock_c_probe = executed
        return checked

    monkeypatch.setattr(OperatorPlan, "__getattribute__", traced_get_attribute)
    return executed


@pytest.mark.parametrize(
    "model_key,sku_alias,parallel,views,layout,padded_tokens",
    [
        ("qwen3-32b", "a5", {"tp": 1}, RAGGED_TND_VIEWS, "TND", 4),
        ("qwen3-32b", "a5", {"tp": 1}, RAGGED_BSND_VIEWS, "BSND", 12),
        ("deepseek-v4-flash", "a5", {"tp": 1}, RAGGED_TND_VIEWS, "TND", 4),
        ("qwen3.5-35b-a3b", "a5", {"tp": 1}, RAGGED_TND_VIEWS, "TND", 4),
        ("qwen3-235b-a22b", "a5", {"tp": 2, "ep": 2}, RAGGED_TND_VIEWS, "TND", 4),
        ("qwen3-32b", "a3", {"tp": 1}, RAGGED_BSND_VIEWS, "BSND", 12),
    ],
)
def test_single_spine_live_execution_trace_across_architectures(
    monkeypatch, traced_plan_programs, model_key, sku_alias, parallel, views, layout, padded_tokens,
):
    """Lock C: 100% governed operator execution, single per-die L2 view, exact RequestView tuple."""
    case = Case.from_mapping(
        compose_case(model_key, hardware=REPRESENTATIVE_SKUS[sku_alias], layers=1, **parallel)
    )
    step_req = StepRequest(
        phase="decode",
        tokens=sum(v.q_len for v in views if not v.padding) or 1,
        context=max((v.context for v in views if not v.padding), default=64),
        sequences=len(views),
        scope="body",
        requests=views,
        padded_tokens=padded_tokens,
        layout=layout,
    )

    dispatched_recipes: list[tuple[str, str, object]] = []
    seen_prepare_views = []
    seen_context_views = []
    seen_seq_requests: list[tuple[str, tuple[RequestView, ...]]] = []
    serviced_operators: set[str] = set()

    orig_prepare = RecipeBook.prepare
    orig_context = _ContextMixin._context
    orig_complete = Engine._complete

    def traced_prepare(self, op, geometry):
        selected = self.select(op)
        prepared, recipe = orig_prepare(self, op, geometry)
        dispatched_recipes.append((op.family, op.recipe, selected))
        seen_prepare_views.append(geometry.admission_memory_view)
        if op.family in SEQUENCE_FAMILIES:
            seen_seq_requests.append((op.op_id, geometry.requests))
        return prepared, recipe

    def traced_context(self, op, replica, scope, *, recipe=None):
        ctx = orig_context(self, op, replica, scope, recipe=recipe)
        seen_context_views.append(ctx.admission_memory_view)
        if op.family in SEQUENCE_FAMILIES:
            seen_seq_requests.append((op.op_id, tuple(ctx.requests)))
        return ctx

    def traced_complete(self, task, moment=None, native=False):
        orig_complete(self, task, moment, native)
        if any(amount > 0.0 for amount in task.service.values()):
            # Device task names extend the replica:iteration:operator prefix.
            serviced_operators.add(":".join(task.event.name.split(":", 3)[:3]))

    spine = PredictionSpine()
    prepared = spine.lower(case, LoweringRequest(step=step_req, tier="mainline"))
    monkeypatch.setattr(RecipeBook, "prepare", traced_prepare)
    monkeypatch.setattr(_ContextMixin, "_context", traced_context)
    monkeypatch.setattr(Engine, "_complete", traced_complete)
    evaluation = spine.run_step(prepared, ReplayOptions(isolated=False))

    assert evaluation.span_ms > 0.0
    timeline = {(row.occurrence.replica, row.occurrence.op_id): row for row in evaluation.operators}
    for rank_ops in prepared.graph.executable:
        for op in rank_ops.ops:
            key = (rank_ops.replica, op.op_id)
            assert key in timeline, f"operator {key} missing from the engine timeline"
            row = timeline[key]
            assert row.start_ms is not None and row.end_ms is not None, f"operator {key} did not complete"
            if op.family == "barrier" or is_zero_kernel_view(op):
                continue
            assert row.end_ms - row.start_ms > 0.0, f"operator {key} had no positive engine duration"
            assert f"r{rank_ops.replica}:i0:{op.op_id}" in serviced_operators, (
                f"operator {key} completed without positive engine resource service"
            )
    assert dispatched_recipes, "expected live operator dispatches during step replay"
    assert traced_plan_programs, "expected executed OperatorPlan.program callbacks"
    outside = [
        (fam, rec, _binding_module(obj))
        for fam, rec, obj in dispatched_recipes
        if not _is_allowed_recipe_owner(obj)
    ]
    assert outside == [], f"executed device operators outside governed packages: {outside}"
    assert seen_context_views, "expected live execution contexts"
    shared = seen_context_views[0]
    assert isinstance(shared, l2_residency.AdmissionMemoryView)
    assert all(view is shared for view in seen_context_views)
    assert seen_prepare_views, "expected live preparation contexts"
    assert all(view is shared for view in seen_prepare_views)
    assert seen_seq_requests, "expected sequence operators during step replay"
    for op_id, received in seen_seq_requests:
        assert received == views, f"sequence op {op_id} dropped canonical RequestView tuple"
        assert all(actual is expected for actual, expected in zip(received, views, strict=True))


@pytest.mark.parametrize("family", ("short_convolution", "gdn_gating"))
def test_live_dispatch_rejects_flat_program_callbacks(monkeypatch, traced_plan_programs, family):
    from mindie_sim.prediction import call_builders as call_builders_operators
    from mindie_sim.prediction.operators.linear_attention import _gdn_decode_programs

    case, views, ops = gdn_decode_source_case((1,))
    op = next(op for op in ops if op.family == family)
    _, span = execute_one(case, views, op)
    assert span > 0 and family in traced_plan_programs
    prepare = RecipeBook.prepare

    def route_to_flat(self, op, geometry):
        prepared, plan = prepare(self, op, geometry)
        original = plan.program

        def program(_self, context):
            return original(context)

        flat = FunctionType(program.__code__, vars(call_builders_operators), closure=program.__closure__)
        delegate = partial(MethodType(flat, self))

        def governed_closure(context):
            return delegate(context)

        callback = FunctionType(governed_closure.__code__, vars(_gdn_decode_programs),
                                closure=governed_closure.__closure__)
        return prepared, replace(plan, program=callback)

    monkeypatch.setattr(RecipeBook, "prepare", route_to_flat)
    with pytest.raises(AssertionError, match=r"OperatorPlan\.program.*prediction\.call_builders_operators"):
        execute_one(case, views, op)


def test_addressed_attention_binds_the_live_admission_view(monkeypatch):
    """Replay addressed KV reads through the engine's live memory binder."""
    case = compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], layers=1)
    request = RAGGED_TND_VIEWS[0]
    contract = AttentionContract(
        request.q_len, 1, 1, 128, 128, sequences=((request.q_len, request.kv_len),),
        kv_layout="PA_BLK_BSND", kv_tensor_id="lock-c-kv", block_table=((0, 2),),
    )
    op = attention.make_attention_op(contract, op_id="addressed_attention", requests=(request.request_id,))
    seen_plans = []
    seen_bind_views = []
    orig_prepare = RecipeBook.prepare
    orig_bind = engine_submit.bind_memory_access

    def traced_prepare(self, op, geometry):
        seen_plans.append((geometry.requests, geometry.admission_memory_view))
        return orig_prepare(self, op, geometry)

    def traced_bind(pattern, state, *args, **kwargs):
        seen_bind_views.append(state)
        return orig_bind(pattern, state, *args, **kwargs)

    monkeypatch.setattr(RecipeBook, "prepare", traced_prepare)
    monkeypatch.setattr(engine_submit, "bind_memory_access", traced_bind)
    report = graph_costs(case, (op,), isolated=False, request_views=(request,))

    assert report["duration_ms"] > 0.0
    assert seen_plans, "expected live FIA preparation"
    shared = seen_plans[0][1]
    assert isinstance(shared, l2_residency.AdmissionMemoryView)
    for requests, view in seen_plans:
        assert len(requests) == 1 and requests[0] is request
        assert view is shared
    assert seen_bind_views, "expected live memory binder callbacks for addressed KV"
    assert all(view is shared for view in seen_bind_views)


@pytest.mark.parametrize("qk_dim,v_dim", ((128, 32), (32, 128)))
def test_attention_matrix_phases_preserve_asymmetric_tensor_work(qk_dim, v_dim):
    heads, keys = 4, 32
    contract = AttentionContract(1, heads, heads, qk_dim, v_dim, sequences=((1, keys),))
    op = attention.make_attention_op(contract, op_id="asymmetric_attention")
    ctx = RecipeContext(op, {"cube": op.flops}, {"cube": "cube"}, {"cube": 1.0}, 1)
    recipe = TiledAttentionRecipe()
    for program in (recipe.lower(ctx), recipe.lower_single(ctx)):
        work = {stage.name: stage.work.get("cube", 0) for stage in program.stages}
        assert work["qk"] == 2 * heads * keys * qk_dim
        assert work["pv"] == 2 * heads * keys * v_dim
        assert sum(work.values()) == op.flops


def test_chunk_timing_rejects_an_unrecognized_declared_isa():
    geometry = Geometry(1, compute_arch=ComputeArchSpec(preset="unrecognized"))
    with pytest.raises(ModelingError, match="linear_attention_arch_unsupported:unrecognized"):
        linear_attention.chunk32_timing_policy(geometry, None, 0)


def test_ragged_and_padding_request_views_preserve_zero_macs_and_layout_strides():
    """Verify padding slots carry 0 logical KV/Q work while BSND vs TND advance distinct GM strides."""
    pad_view = RequestView("pad0", q_len=4, context=1024, samples=False, phase="decode", padding=True)
    assert pad_view.logical_q_len == 0
    assert pad_view.kv_len == 0

    tnd_spans = sequence_layout_spans(RAGGED_TND_VIEWS, padded_tokens=4, layout="TND")
    assert len(tnd_spans) == len(RAGGED_TND_VIEWS)
    assert tnd_spans[0] == (1, 129, 0, 0, False)
    assert tnd_spans[1] == (1, 513, 1, 129, False)
    assert tnd_spans[2] == (0, 0, 2, 642, True)

    bsnd_spans = sequence_layout_spans(RAGGED_BSND_VIEWS, padded_tokens=12, layout="BSND")
    assert bsnd_spans[0] == (1, 65, 0, 0, False)
    assert bsnd_spans[1] == (2, 194, 4, 256, False)
    assert bsnd_spans[2] == (0, 0, 8, 512, True)


def test_observed_aicore_clock_scales_core_domains_only_and_extracts_capture_freq():
    """Observed clock scaling scales core-coupled domains and leaves HBM/links/scalar store fixed."""
    from mindie_sim.backends.ascend import profiles

    sku = REPRESENTATIVE_SKUS["a5"]
    profile = profiles()[sku]
    nominal_mhz = float(profile["source"]["declaration"]["configured_clock_mhz"])
    assert nominal_mhz > 0.0
    observed_mhz = nominal_mhz * CLOCK_SCALE_RATIO
    capture = {"metadata": {"observed_aicore_freq_mhz": observed_mhz}}
    assert extract_observed_aicore_freq_mhz(capture) == pytest.approx(observed_mhz)

    raw_case = compose_case("qwen3-32b", hardware=sku, layers=1, tp=2)
    aligned_raw = apply_capture_observed_clock(raw_case, capture)
    base_machine = Case.from_mapping(raw_case).machine
    scaled_machine = Case.from_mapping(aligned_raw).machine
    base_die, scaled_die = base_machine.dies[0], scaled_machine.dies[0]

    assert scaled_die.observed_aicore_freq_mhz == pytest.approx(observed_mhz)
    assert scaled_die.peak("bf16") == pytest.approx(base_die.peak("bf16") * CLOCK_SCALE_RATIO)
    base_rates, scaled_rates = dict(base_die.rates), dict(scaled_die.rates)
    assert scaled_rates["vector"] == pytest.approx(base_rates["vector"] * CLOCK_SCALE_RATIO)
    assert scaled_rates["hbm"] == pytest.approx(base_rates["hbm"])
    if "scalar_store_issue" in base_rates:
        assert scaled_rates["scalar_store_issue"] == pytest.approx(base_rates["scalar_store_issue"])
    base_modes = {name: mode[1] for name, mode in base_die.service_modes}
    scaled_modes = {name: mode[1] for name, mode in scaled_die.service_modes}
    if "mte2_aic_dram" in base_modes:
        assert scaled_modes["mte2_aic_dram"] == pytest.approx(base_modes["mte2_aic_dram"])
    for b_link, s_link in zip(base_machine.links, scaled_machine.links, strict=True):
        assert s_link.capacity_bytes_per_ms == pytest.approx(b_link.capacity_bytes_per_ms)


def test_mixed_hbm_accounting_asymmetric_cohort_split_and_streaming_capture_guard(tmp_path):
    """Regression locks for mixed-direction HBM rebind, asymmetric cohort split, and streaming verifier."""
    for backend in ("python", "c"):
        bus = HbmBusSpec(0, BUS_READ_BPMS, BUS_WRITE_BPMS, BUS_ALPHA, row_r="hbm:r", row_w="hbm:w")
        eng = Engine(
            [Resource("hbm", BUS_READ_BPMS), Resource("mte2_aic", BUS_READ_BPMS * 8.0), Resource("mte3", BUS_WRITE_BPMS * 8.0)],
            backend=backend,
        )
        eng.add_hbm_bus(bus)
        ev_r = eng.submit(
            "mixed_r", {"hbm": TOTAL_BYTES, "mte2_aic": TOTAL_BYTES},
            deliveries=(("hbm_r", TOTAL_BYTES),),
        )
        ev_w = eng.submit("mixed_w", {"hbm": TOTAL_BYTES, "mte3": TOTAL_BYTES})
        eng.advance(horizon=8.0)
        mid_r = eng.delivered["hbm_r"]
        assert 0.0 < mid_r < TOTAL_BYTES
        eng.bind_hbm_bus(HbmBusSpec(0, BUS_REBIND_READ_BPMS, BUS_WRITE_BPMS, BUS_ALPHA, row_r="hbm:r", row_w="hbm:w"))
        eng.advance(until=(ev_r, ev_w), first=False)
        assert ev_r.time > 8.0
        assert eng.delivered["hbm_r"] == pytest.approx(TOTAL_BYTES)
        assert eng.serviced["hbm"] == pytest.approx(TOTAL_BYTES * 2.0)

        eng2 = Engine([Resource("hbm", BUS_READ_BPMS), Resource("mte2_aic", BUS_READ_BPMS * 8.0)], backend=backend)
        bus2 = HbmBusSpec(0, BUS_READ_BPMS, BUS_WRITE_BPMS, BUS_ALPHA, row_r="hbm:r", row_w="hbm:w")
        eng2.add_hbm_bus(bus2)
        c0 = eng2.submit(
            "cohort_a", {"hbm": ASYMM_BYTES, "mte2_aic": ASYMM_BYTES},
            deliveries=(("hbm_r", ASYMM_BYTES),),
        )
        eng2.advance(horizon=3.0)
        assert eng2.delivered["hbm_r"] == pytest.approx(3072.0)
        eng2.bind_hbm_bus(HbmBusSpec(0, BUS_READ_BPMS * 0.5, BUS_WRITE_BPMS, BUS_ALPHA, row_r="hbm:r", row_w="hbm:w"))
        eng2.advance(until=(c0,))
        assert c0.time == pytest.approx(13.0)
        assert eng2.delivered["hbm_r"] == pytest.approx(ASYMM_BYTES)

    cap_path = tmp_path / "capture.jsonl"
    cap_bytes = (
        json.dumps({"event_id": "start", "clock_operator": "device_event"}) + "\n"
        + json.dumps({"event_id": "end", "clock_operator": "host_wall"}) + "\n"
    ).encode("utf-8")
    cap_path.write_bytes(cap_bytes)
    boundary = SimpleNamespace(
        raw_sha256=hashlib.sha256(cap_bytes).hexdigest(),
        start_event="start",
        end_event="end",
        event_ids=(),
        device_ranks=(0,),
        clock_domain="device",
        clock_operator="device_event",
    )
    with pytest.raises(ModelingError, match="capture_clock_operator_mismatch"):
        verify_capture_file_consistency("case0", boundary, cap_path)


def test_peer_compaction_requires_uniform_groups():
    assert peer_group_can_compact((16, 16, 16), slots=2, threshold=8) is True
    assert peer_group_can_compact((16, 4, 16), slots=2, threshold=8) is False


@pytest.mark.parametrize("sku_alias,supported", (("a3", False), ("a5", True)))
def test_packed_schedule_checks_declared_isa_without_datapath(sku_alias, supported):
    case = Case.from_mapping(
        compose_case("deepseek-v4-flash", hardware=REPRESENTATIVE_SKUS[sku_alias], layers=1, tp=1)
    )
    geometry = replace(Geometry.from_devices(case.machine.dies), datapath=None)
    op = OpSpec(
        "packed_attention", "mla", "decode", 0, 0, ("r0",),
        (1, 16, 512), (1, 16, 512), (1, 16, 512), (1, 16, 512),
        dtype="bf16", recipe="indexed_attention_packed_int8",
        metadata=((
            "indexed_attention_geometry",
            {
                "query_dtype": "bf16", "cache_dtype": "int8", "latent": 512, "rope": 64,
                "window": 0, "compression_ratio": 1, "cp": 1, "heads": 16,
                "sequences": ((1, 128),), "topk": 128,
            },
        ),),
    )
    schedule = indexed_quant_attention.SCHEDULE_ARCH35
    if supported:
        plan = indexed_quant_attention.build(op, geometry, scratch_policy="offchip", schedule=schedule)
        assert plan.metadata["kernel_schedule"] == schedule
        assert plan.work.matrix["bf16"] == 2 * 16 * 128 * (512 + 64 + 512)
    else:
        with pytest.raises(ModelingError, match=f"arch35_unsupported_on_{geometry.compute_arch.preset}"):
            indexed_quant_attention.build(op, geometry, scratch_policy="offchip", schedule=schedule)


def test_oneshot_collective_physics_and_waveplan_multigroup():
    """Verify hardware-derived OneShotCollective physics, Ring crossover, and multi-group WavePlan."""
    from mindie_sim.prediction.collectives import (
        OneShotCollective,
        OneShotPlan,
        Topology,
        api_bytes_from_link_volume,
    )
    from mindie_sim.prediction.collectives.peer_traffic import PeerTraffic
    from mindie_sim.prediction.collectives.waves import WaveGroup, WavePlan
    from mindie_sim.prediction.program import KernelProgram

    def _run_plan(topo, plan):
        eng = Engine(topo.resources())
        start = eng.submit("start")
        done = plan.start(eng, "coll", start)
        eng.advance(until=(done,))
        return done.time

    case_a5 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], layers=1, tp=8))
    case_a3 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a3"], layers=1, tp=8))
    topo_a5 = Topology(case_a5.machine, case_a5.runtime.parallel)
    topo_a3 = Topology(case_a3.machine, case_a3.runtime.parallel)

    recipe = OneShotCollective()
    assert recipe.algorithm == "oneshot"
    assert recipe.path_model == "store_forward"

    # Hardware-derived startup_ms matches declared vector minimum_ms without chip-name branching.
    vec_a5 = dict(case_a5.machine.dies[0].kernel_invocations).get("vector")
    expected_a5_ms = float(getattr(vec_a5, "minimum_ms", 0.0) or 0.0)
    assert topo_a5._oneshot_startup_ms() == pytest.approx(expected_a5_ms)

    # Small allreduce (16 KiB API tensor, stale ring steps=14 tolerated): OneShot beats Ring on both A3 & A5.
    small_api = 16 * 1024.0
    small_ring_vol = small_api * 2.0 * 7.0 / 8.0
    op_ar_small = OpSpec(
        "ar_small", "allreduce", "decode", 0, 0, ("r0",), (1, 128), (1, 128), (1, 128), (1, 128),
        link_bytes=small_ring_vol,
        metadata=(("group_axis", "tp"), ("collective", "allreduce"), ("steps", 14)),
    )
    for topo in (topo_a5, topo_a3):
        os_plan = topo.oneshot_plan(op_ar_small, 0)
        rg_plan = topo.joint_plan(op_ar_small, {0: small_ring_vol}, algorithm="ring")
        assert isinstance(os_plan, OneShotPlan)
        assert len(os_plan.groups) == 1 and len(os_plan.groups[0].messages) == 8 * 7
        expected_peer = api_bytes_from_link_volume("allreduce", 8, small_ring_vol, dict(op_ar_small.metadata))
        assert all(m.flow.useful_bytes == pytest.approx(expected_peer) for m in os_plan.groups[0].messages)
        assert _run_plan(topo, os_plan) < _run_plan(topo, rg_plan)

    # Large allreduce (64 MiB API tensor) on switched fabric: Ring beats OneShot broadcast on bandwidth.
    big_api = 64.0 * 1024.0 * 1024.0
    big_ring_vol = big_api * 2.0 * 7.0 / 8.0
    op_ar_big = replace(op_ar_small, op_id="ar_big", link_bytes=big_ring_vol)
    assert _run_plan(topo_a3, topo_a3.joint_plan(op_ar_big, {0: big_ring_vol}, algorithm="ring")) < _run_plan(
        topo_a3, topo_a3.oneshot_plan(op_ar_big, 0)
    )

    # AllGather conserves exact per-rank TX bytes (each peer gets vol / (G - 1)) and beats Ring on small msg.
    ag_vol = 28 * 1024.0
    op_ag = OpSpec(
        "ag_small", "allgather", "decode", 0, 0, ("r0",), (1, 128), (1, 128), (1, 128), (1, 128),
        link_bytes=ag_vol,
        metadata=(("group_axis", "tp"), ("collective", "allgather"), ("steps", 7)),
    )
    ag_plan = topo_a5.oneshot_plan(op_ag, 0)
    tx_by_src: dict[int, float] = {}
    for msg in ag_plan.groups[0].messages:
        tx_by_src[msg.flow.source] = tx_by_src.get(msg.flow.source, 0.0) + msg.flow.useful_bytes
    assert len(tx_by_src) == 8 and all(val == pytest.approx(ag_vol) for val in tx_by_src.values())
    assert _run_plan(topo_a5, ag_plan) < _run_plan(topo_a5, topo_a5.joint_plan(op_ag, {0: ag_vol}, algorithm="ring"))

    # alltoallv requires explicit PeerTraffic and emits exact directed payloads when declared.
    op_a2av_missing = OpSpec(
        "a2av_bad", "alltoallv", "decode", 0, 0, ("r0",), (1, 128), (1, 128), (1, 128), (1, 128),
        link_bytes=1024.0, metadata=(("group_axis", "tp"), ("collective", "alltoallv")),
    )
    with pytest.raises(ModelingError, match="oneshot_collective_alltoallv_requires_explicit_peer_matrix"):
        topo_a5.oneshot_plan(op_a2av_missing, 0)

    case_tp2 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], layers=1, tp=2))
    topo_tp2 = Topology(case_tp2.machine, case_tp2.runtime.parallel)
    pt_raw = {"kind": "matrix", "payload_bytes": ((0.0, 2048.0), (2048.0, 0.0))}
    assert PeerTraffic.from_mapping(pt_raw, size=2).row_total(0) == pytest.approx(2048.0)
    op_a2av = OpSpec(
        "a2av_ok", "alltoallv", "decode", 0, 0, ("r0",), (1, 128), (1, 128), (1, 128), (1, 128),
        link_bytes=2048.0,
        metadata=(("group_axis", "tp"), ("collective", "alltoallv"), ("peer_traffic", pt_raw)),
    )
    a2av_plan = topo_tp2.oneshot_plan(op_a2av, 0)
    assert sorted((m.flow.source, m.flow.destination, m.flow.useful_bytes) for m in a2av_plan.groups[0].messages) == [
        (0, 1, 2048.0),
        (1, 0, 2048.0),
    ]

    # OneShotCollective.lower validates transport-only work and binds Stage("oneshot").
    ctx_ok = SimpleNamespace(native_work={"link": 2048.0}, minimum_ms=0.0, collective_plan=lambda: a2av_plan)
    prog = recipe.lower(ctx_ok)
    assert isinstance(prog, KernelProgram) and len(prog.stages) == 1 and prog.stages[0].name == "oneshot"
    ctx_bad = SimpleNamespace(native_work={"link": 2048.0, "vector": 1.0}, minimum_ms=0.0, collective_plan=lambda: a2av_plan)
    with pytest.raises(ModelingError, match="oneshot_collective_requires_transport_only_work"):
        recipe.lower(ctx_bad)

    # WavePlan.start_participants executes cleanly across 2 disjoint rank groups.
    case_dp2 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], layers=1, dp=2, tp=2))
    topo_dp2 = Topology(case_dp2.machine, case_dp2.runtime.parallel)
    wave_plan = WavePlan(
        "alltoall",
        (
            WaveGroup((0, 1), ((0, 4096.0), (1, 4096.0)), 1, topo_dp2),
            WaveGroup((2, 3), ((2, 4096.0), (3, 4096.0)), 1, topo_dp2),
        ),
    )
    wave_eng = Engine(topo_dp2.resources())
    wave_ready = {r: wave_eng.submit(f"w_r{r}") for r in (0, 1, 2, 3)}
    wave_done = wave_plan.start_participants(wave_eng, "multi_wave", wave_ready)
    wave_eng.advance(until=tuple(wave_done.values()), first=False)
    assert set(wave_done) == {0, 1, 2, 3} and all(ev.time > 0.0 for ev in wave_done.values())


def test_rank_group_indexing_comm_axis_and_unfused_sp_mc2():
    """Verify etp/external_dp/DCP-TP group indexing, structured comm_axis, and unfused SP MC2."""
    from mindie_sim.prediction.collectives import Topology
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.sharding.local_tp import (
        expert_tp_groups,
        local_tp_cohorts,
        rank_zero_graph_reuse,
    )
    from mindie_sim.prediction.sharding.routing import balanced_realization
    from mindie_sim.prediction.sharding.sequence_parallel import (
        DECLARATION as SP_DECL,
        IMPLEMENTATION as SP_IMPL,
        LEGACY_IMPLEMENTATION as SP_LEGACY_IMPL,
        sequence_parallel_enabled,
    )
    from mindie_sim.schema.parallel import ParallelSpec

    # 1) Topology.groups("ep") with etp=2 uses ep_tp_w = tp // etp, orthogonal to expert_tp_groups.
    base_8 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], layers=1, tp=8))
    p_dp_etp = ParallelSpec(dp=4, tp=2, pp=1, ep=4, expert_tp=2)
    topo_dp_etp = Topology(base_8.machine, p_dp_etp)
    ep_groups_1 = topo_dp_etp.groups("ep", 0, 0)
    etp_groups_1 = expert_tp_groups(topo_dp_etp, replica=0, stage=0)
    assert ep_groups_1 == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert etp_groups_1 == ((0, 1),)
    assert all(len(set(eg) & set(tg)) <= 1 for eg in ep_groups_1 for tg in etp_groups_1)
    assert rank_zero_graph_reuse(p_dp_etp, 4, False) is False
    realized = balanced_realization(
        SimpleNamespace(parallel=p_dp_etp, moe_recipe="alltoall"),
        SimpleNamespace(num_experts=8, top_k=2),
        tokens=4, valid_tokens=4, global_tokens=16, replica=3,
    )
    assert realized.represented_ranks == (3,) and realized.representative_rank == 3

    p_pcp_etp = ParallelSpec(dp=1, tp=4, pp=1, pcp=2, ep=4, expert_tp=2)
    topo_pcp_etp = Topology(base_8.machine, p_pcp_etp)
    ep_groups_2 = topo_pcp_etp.groups("ep", 0, 0)
    etp_groups_2 = expert_tp_groups(topo_pcp_etp, replica=0, stage=0)
    assert ep_groups_2 == ((0, 1, 4, 5), (2, 3, 6, 7))
    assert etp_groups_2 == ((0, 2), (1, 3), (4, 6), (5, 7))
    assert all(len(set(eg) & set(tg)) <= 1 for eg in ep_groups_2 for tg in etp_groups_2)

    # 2) external_dp > 1 with pcp > 1: local_tp_cohorts and Topology.groups("dp") stay inside external domain.
    base_16 = Case.from_mapping(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a3"], layers=1, dp=2, tp=8))
    p_ext = ParallelSpec(dp=2, tp=2, pp=1, ep=2, pcp=2, external_dp=2)
    topo_ext = Topology(base_16.machine, p_ext)
    cohorts_ext = local_tp_cohorts(base_16.machine, p_ext)
    assert sum(len(c) for c in cohorts_ext) == 16
    dp_ext0 = topo_ext.groups("dp", 0, 0)
    dp_ext1 = topo_ext.groups("dp", 2, 0)
    assert all(max(g) < 8 for g in dp_ext0) and all(min(g) >= 8 for g in dp_ext1)

    # 3) DCP <-> TP co-domain when PCP == 1: dcp reuses tp ranks.
    p_dcp_tp = ParallelSpec(dp=2, tp=4, pp=1, ep=8, dcp=4, pcp=1)
    topo_dcp_tp = Topology(base_8.machine, p_dcp_tp)
    assert topo_dcp_tp.groups("dcp", 0, 0) == topo_dcp_tp.groups("tp", 0, 0) == ((0, 1, 2, 3),)

    # 4) Sequence-parallel backward-compat implementation name + structured comm_axis + unfused MC2.
    assert SP_IMPL == "vllm_ascend_sequence_parallel"
    assert sequence_parallel_enabled({"sequence_parallel": dict(SP_DECL)}, 4) is True
    assert sequence_parallel_enabled(
        {"sequence_parallel": {**SP_DECL, "implementation": SP_LEGACY_IMPL}}, 4
    ) is True

    raw_sp = compose_case(
        "kimi-k3.1", hardware=REPRESENTATIVE_SKUS["a5"], layers=4,
        dp=1, tp=4, dcp=4, ep=4, role="pd_decode", feature_track="merged",
        prompt_tokens=4096, output_tokens=64,
    )
    assert raw_sp["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 0
    case_sp = Case.from_mapping(raw_sp)
    case_sp = replace(
        case_sp,
        runtime=replace(
            case_sp.runtime,
            kernels={**dict(case_sp.runtime.kernels), "sequence_parallel": {**SP_DECL, "padding": "right_to_tp"}},
        ),
    )
    graph_sp = target_graphs(case_sp, phase="decode", tokens=4, context=4096, sequences=4, scope="body")[0]
    sp_gathers = [op for op in graph_sp if op.family == "allgather" and dict(op.metadata).get("comm_axis") == "tp_sp"]
    sp_scatters = [op for op in graph_sp if op.family == "reduce_scatter" and dict(op.metadata).get("comm_axis") == "tp_sp"]
    dcp_comms = [op for op in graph_sp if dict(op.metadata).get("comm_axis") == "dcp"]
    assert sp_gathers and all(dict(op.metadata)["comm_group_size"] == 4 for op in sp_gathers)
    assert sp_scatters and all(dict(op.metadata)["comm_group_size"] == 4 for op in sp_scatters)
    assert dcp_comms and all(dict(op.metadata)["comm_group_size"] == 4 for op in dcp_comms)
    assert not any(op.family in {"all_gather_matmul", "matmul_reduce_scatter"} for op in graph_sp)
