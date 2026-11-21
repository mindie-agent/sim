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
import hashlib
import json

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.backends.ascend import apply_observed_aicore_clock
from mindie_sim.compose import compose_case
from mindie_sim.observations.multilevel_stream import verify_capture_file_consistency
from mindie_sim.observations.accuracy import (
    apply_capture_observed_clock,
    extract_observed_aicore_freq_mhz,
)
from mindie_sim.prediction.execution.context import _ContextMixin
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.attention.common import sequence_layout_spans
from mindie_sim.prediction.operators.matrix.refill_steady import peer_group_can_compact
from mindie_sim.prediction.operators.paged_mla import indexed_quant_attention
from mindie_sim.prediction.program.recipes import RecipeBook
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.prediction.runtime import l2_residency
from mindie_sim.prediction.runtime.engine import Engine
from mindie_sim.prediction.runtime.engine_types import Resource
from mindie_sim.prediction.runtime.memory_physics import HbmBusSpec
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.stack_registry import LoweringRequest
from scripts._arch_governance_runtime import _NON_DEVICE_INFRA_RECIPES, _OWNER_MODULE_PREFIXES, _binding_module
from types import SimpleNamespace

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
    monkeypatch, model_key, sku_alias, parallel, views, layout, padded_tokens,
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
    seen_prepare_views: set[int] = set()
    seen_context_views: set[int] = set()
    seen_bind_views: set[int] = set()
    seen_seq_requests: list[tuple[str, tuple[RequestView, ...]]] = []

    orig_prepare = RecipeBook.prepare
    orig_context = _ContextMixin._context
    orig_bind = l2_residency.bind_memory_access

    def traced_prepare(self, op, geometry):
        selected = self.select(op)
        prepared, recipe = orig_prepare(self, op, geometry)
        dispatched_recipes.append((op.family, op.recipe, selected))
        if getattr(geometry, "admission_memory_view", None) is not None:
            seen_prepare_views.add(id(geometry.admission_memory_view))
        if op.family in SEQUENCE_FAMILIES and getattr(geometry, "requests", ()):
            seen_seq_requests.append((op.op_id, tuple(geometry.requests)))
        return prepared, recipe

    def traced_context(self, op, replica, scope, *, recipe=None):
        ctx = orig_context(self, op, replica, scope, recipe=recipe)
        if ctx.admission_memory_view is not None:
            seen_context_views.add(id(ctx.admission_memory_view))
        if op.family in SEQUENCE_FAMILIES:
            seen_seq_requests.append((op.op_id, tuple(ctx.requests)))
        return ctx

    def traced_bind(pattern, state, *args, **kwargs):
        if state is not None:
            seen_bind_views.add(id(state))
        return orig_bind(pattern, state, *args, **kwargs)

    monkeypatch.setattr(RecipeBook, "prepare", traced_prepare)
    monkeypatch.setattr(_ContextMixin, "_context", traced_context)
    monkeypatch.setattr(l2_residency, "bind_memory_access", traced_bind)

    spine = PredictionSpine()
    prepared = spine.lower(case, LoweringRequest(step=step_req, tier="mainline"))
    evaluation = spine.run_step(prepared, ReplayOptions(isolated=False))

    assert evaluation.span_ms > 0.0
    assert dispatched_recipes, "expected live operator dispatches during step replay"
    outside = [
        (fam, rec, _binding_module(obj))
        for fam, rec, obj in dispatched_recipes
        if not _is_allowed_recipe_owner(obj)
    ]
    assert outside == [], f"executed device operators outside governed packages: {outside}"
    assert len(seen_context_views) == 1, f"expected 1 shared AdmissionMemoryView, got {len(seen_context_views)}"
    if seen_prepare_views:
        assert seen_prepare_views == seen_context_views
    if seen_bind_views:
        assert seen_bind_views.issubset(seen_context_views)
    assert seen_seq_requests, "expected sequence operators during step replay"
    for op_id, received in seen_seq_requests:
        assert received == views, f"sequence op {op_id} dropped canonical RequestView tuple"


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


def test_uniform_peer_compaction_and_arch22_packed_quant_rejection():
    """Verify I-09 uniform peer compaction and A3-02 DAV_2201 rejection on arch35 packed schedule."""
    assert peer_group_can_compact((16, 16, 16), slots=2, threshold=8) is True
    assert peer_group_can_compact((16, 4, 16), slots=2, threshold=8) is False
    a3_case = Case.from_mapping(
        compose_case("deepseek-v4-flash", hardware=REPRESENTATIVE_SKUS["a3"], layers=1, tp=1)
    )
    geom_a3 = Geometry.from_devices(a3_case.machine.dies)
    op = OpSpec(
        "sfa_a3", "mla", "decode", 0, 0, ("r0",), (1, 16, 512), (1, 16, 512), (1, 16, 512), (1, 16, 512),
        dtype="bf16", recipe="indexed_attention_packed_int8",
        metadata=((
            "indexed_attention_geometry",
            {
                "query_dtype": "bf16", "cache_dtype": "int8", "latent": 512, "rope": 64,
                "window": 0, "compression_ratio": 1, "cp": 1, "heads": 16,
                "requests": ((1, 128),),
            },
        ),),
    )
    with pytest.raises(ModelingError, match="arch35_unsupported"):
        indexed_quant_attention.build(
            op, geom_a3, scratch_policy="offchip", schedule=indexed_quant_attention.SCHEDULE_ARCH35,
        )
