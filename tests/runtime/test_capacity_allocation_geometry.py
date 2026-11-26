"""Physical cache and working-set geometry used by capacity admission.

[Role] Check the byte ledgers for page layout and fixed allocation plans.
[Invariants] Valid rows, rounded allocation, access, and metadata have distinct
  totals. Each physical rank must fit its own cache and concurrent workspace.
[Timeouts & Locks] Bounded calculations over small declarations and catalog cases.
[Side-Effect Policy] Reads checked-in JSON cases and changes only local memory.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from _support.memory_cases import budget_memory, case, compressed_case
from mindie_sim.catalog.cache import cache_row_split, layer_cache_components
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.cache_pages import CachePageGroup, KVPageAllocation
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.capacity import Allocation, AllocationPlan, Memory, MemoryPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0

EXAMPLE_CASES = Path(__file__).resolve().parents[2] / "examples" / "cases"
MIB = 1024 ** 2
PAGE_GROUP = {
    "layers": [0], "components": ["gqa_kv"],
    "tokens_per_page": 3, "bytes_per_group_page": 72,
}
ROW_FORMAT = {"dtype": "int8", "group_size": 8, "scale_dtype": "fp32"}


def _with_hbm(parsed, budgets, *, weights=None):
    if weights is None:
        weights = [Memory(parsed).weights[0]] * len(budgets)
    dies = tuple(replace(die, memory_bytes=int(weight + budget))
                 for die, weight, budget in zip(parsed.machine.dies, weights, budgets))
    return replace(parsed, machine=replace(parsed.machine, dies=dies))


def _byte_views(allocation, **views):
    for view, components in views.items():
        for component, expected in components.items():
            assert allocation[view][f"{component}_bytes_per_stage_rank"] == [expected], (
                view, component
            )


@pytest.mark.parametrize(
    "element_bytes,page_scale,valid,payload,tail,scale,total,access",
    [(2, 0, 1536, 2048, 512, 0, 2176, 1664), (1, 8, 768, 1024, 256, 32, 1184, 928)],
    ids=["bf16", "int8-with-scales"],
)
def test_page_allocation_tracks_valid_payload_tail_scales_and_access(
    element_bytes, page_scale, valid, payload, tail, scale, total, access,
):
    page = KVPageAllocation(
        token_spans=(17, 31), page_tokens=16, local_kv_heads=1, head_dim=8,
        kv_count=2, element_bytes=element_bytes, scale_bytes_per_page=page_scale,
        metadata_bytes_per_page=32,
    )
    # Owners round separately, so 48 valid rows occupy 4 pages and 64 slots.
    assert (page.logical_tokens, page.pages, page.physical_token_slots) == (48, 4, 64)
    assert (page.valid_payload_bytes, page.allocated_payload_bytes) == (valid, payload)
    assert (page.page_tail_payload_bytes, page.scale_bytes, page.metadata_bytes) == (tail, scale, 128)
    assert page.total_allocation_bytes == total
    assert (page.access_payload_bytes, page.access_scale_bytes, page.access_bytes) == (valid, scale, access)
    assert page.access_payload_bytes + page.access_metadata_bytes + page.access_scale_bytes == access


@pytest.mark.parametrize("geometry,reason", [
    ({"page_tokens": 0}, "requires_positive_geometry"),
    ({"element_bytes": -1}, "invalid:element_bytes"),
])
def test_page_allocation_rejects_invalid_geometry(geometry, reason):
    with pytest.raises(ModelingError, match=f"kv_page_allocation_{reason}"):
        KVPageAllocation((1,), **geometry)


@pytest.mark.parametrize("representation,expected", [(None, (32, 0)), (ROW_FORMAT, (16, 8))])
def test_cache_row_scale_is_charged_per_group(representation, expected):
    model = parse_case(case()).model
    if representation is not None:
        model = replace(model, metadata=(("cache_representations", {"mla_kv": representation}),))
    payload, scale = cache_row_split(model, "mla_kv", 16)
    assert (payload, scale) == expected
    assert payload + scale == cache_format(model, "mla_kv").row_bytes(16)


def _joint_plan(kv, workspace, *, temporary=0, extra_live=None):
    allocations = [
        Allocation("weights", 512 * MIB, "weights"),
        Allocation("runtime_persistent", 64 * MIB, "runtime"),
        Allocation("communication_pool", 64 * MIB, "pool"),
        Allocation("communication_tx", 32 * MIB, "communication", pool="communication_pool"),
        Allocation("communication_rx", 32 * MIB, "communication", pool="communication_pool"),
        Allocation("weights_view", 4096, "workspace", alias_of="weights"),
        Allocation("kv", kv, "cache"),
        Allocation("workspace", workspace, "workspace"),
        Allocation("temporary", temporary, "temporary"),
    ]
    if extra_live is not None:
        allocations.append(Allocation("workspace_reuse", workspace, "workspace", live=extra_live))
    return AllocationPlan(tuple(allocations), capacity_bytes=1024 ** 3)


@pytest.mark.parametrize("kv,workspace,temporary,extra_live,peak,feasible", [
    (268_566_528, 128 * MIB, 0, None, 1_073_872_896, False),
    (268_566_528, 96 * MIB, 0, None, 1_040_318_464, True),
    (134_873_088, 128 * MIB, 0, None, 940_179_456, True),
    (134_873_088, 128 * MIB, 0, True, 1_074_397_184, False),
    (134_873_088, 128 * MIB, 0, False, 940_179_456, True),
    (134_873_088, 128 * MIB, 256 * MIB, None, 1_208_614_912, False),
], ids=["wide-cache", "small-workspace", "narrow-cache", "concurrent", "sequential", "dequant"])
def test_fixed_plan_peak_counts_only_concurrent_physical_allocations(
    kv, workspace, temporary, extra_live, peak, feasible,
):
    report = _joint_plan(kv, workspace, temporary=temporary, extra_live=extra_live).to_dict()
    assert (report["peak_bytes"], report["feasible"]) == (peak, feasible)
    assert report["alias_bytes_not_added"] == 4096
    assert report["pool_bytes_not_added"] == report["categories"]["pool"] == 64 * MIB
    assert report["categories"]["workspace"] == workspace * (2 if extra_live is True else 1)
    if extra_live is False:
        assert report["retired_sequential_bytes"] == workspace


def test_pool_rejects_overcommitted_slices():
    with pytest.raises(ValueError, match="allocation_pool_overcommitted"):
        AllocationPlan((
            Allocation("pool", 1024, "pool"),
            Allocation("slice", 2048, "communication", pool="pool"),
        )).resolve()


def test_page_metadata_is_charged_once_and_enters_real_admission():
    raw = case(prompt=4, output=1)
    raw["runtime"].update(kv_block_size=8, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    layer = parsed.model.layer_plans[0]
    layer_total = sum(layer_cache_components(
        parsed.model, layer, parsed.runtime.parallel, 3, capacity_padded=True,
        interleave=parsed.runtime.kv_block_size,
    ).values())
    group = {"layers": [layer.index], "tokens_per_page": 4,
             "bytes_per_layer_page": 256, "metadata_bytes_per_page": 32}
    declared = CachePageGroup.from_mapping(group)
    assert declared.pages(3, 128) == 1
    assert declared.metadata_bytes(1) == 32
    for price, payload, expected in (
        ("bytes_per_layer_page", 64, 64), ("bytes_per_group_page", 128, 32),
    ):
        two_layers = CachePageGroup.from_mapping({
            "layers": [0, 1], "tokens_per_page": 4, price: payload, "metadata_bytes_per_page": 32,
        })
        assert two_layers.metadata_bytes(1) == expected

    parsed = replace(parsed, runtime=replace(parsed.runtime, memory={
        "derive_workspace": False, "cache_page_groups": [group],
    }))
    memory = Memory(parsed)
    payload = base.cache_bytes(3)[0] - layer_total + 256
    expected = payload + 32
    assert memory.cache_bytes(3) == (expected,)
    tail = payload - memory.cache_bytes(3, padded=False)[0]
    assert tail > 0
    _byte_views(memory.cache_allocation(3), allocated={"metadata": 32}, page_tail={"total": tail})
    for capacity, admitted in [(payload, False), (expected, True)]:
        tight = Memory(_with_hbm(parsed, (capacity,)))
        assert tight.resize(0, "r", 3) is admitted
        assert tight.live == [[expected if admitted else 0.0]]


def _gqa_case(*, block=1, tp=1, window=None, explicit=True, group=None):
    raw = case(tp=tp)
    raw["runtime"].update(kv_block_size=block, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    layer = replace(parsed.model.layer_plans[0], num_kv_heads=1, head_dim=8, window=window)
    model = replace(parsed.model, layers=1, layer_plans=(layer,))
    if explicit:
        model = replace(model, metadata=(("cache_representations", {"gqa_kv": ROW_FORMAT}),))
    runtime = replace(parsed.runtime, pp_layer_partition=(1,))
    if group is not None:
        runtime = replace(runtime, memory={**runtime.memory, "cache_page_groups": [group]})
    return replace(parsed, runtime=runtime, model=model)


def test_gqa_explicit_representation_drives_total_and_admission():
    default, explicit = _gqa_case(explicit=False), _gqa_case()
    assert Memory(default).cache_bytes(1) == (32.0,)
    memory = Memory(explicit)
    assert memory.cache_bytes(1) == (24.0,)
    _byte_views(memory.cache_allocation(1), allocated={"payload": 16, "scale": 8, "total": 24},
                page_tail={"total": 0})
    tight = _with_hbm(explicit, (24,), weights=Memory(default).weights)
    assert not Memory(replace(tight, model=default.model)).resize(0, "r", 1)
    admitted = Memory(tight)
    assert admitted.resize(0, "r", 1)
    assert admitted.live == [[24.0]]


@pytest.mark.parametrize("block,span,window,tp,slots,valid,metadata", [
    (8, 3, None, 1, 3, 3, 0),
    (8, 1, None, 1, 3, 1, 0),
    (1, 3, None, 1, 3, 3, 0),
    (8, 9, 5, 1, 6, 5, 64),
    (8, 2, None, 2, 3, 1, 0),
], ids=["full-page", "one-row", "different-block", "window", "context-shard"])
def test_group_pages_separate_physical_slots_and_local_valid_rows(
    block, span, window, tp, slots, valid, metadata,
):
    group = dict(PAGE_GROUP)
    if window is not None:
        group.update(resident_tokens=window, metadata_bytes_per_page=32)
    parsed = _gqa_case(block=block, tp=tp, window=window, group=group)
    if tp == 2:
        # Isolate CP geometry with one KV head even though the public schema
        # rejects this initial head/CP placement.
        parallel = replace(parsed.runtime.parallel, cp=2, dcp=2, cp_plan="dcp")
        parsed = replace(parsed, runtime=replace(parsed.runtime, parallel=parallel))
    memory = Memory(parsed)
    # Each 16-element row has 16 payload B and two FP32 scales.
    total = slots * 24 + metadata
    tail = slots - valid
    assert memory.cache_bytes(span) == (total,)
    if tp == 2:
        assert memory.cache_bytes(span, padded=False) == (24.0,)
    _byte_views(
        memory.cache_allocation(span),
        allocated={"payload": slots * 16, "scale": slots * 8, "total": total,
                   "metadata": metadata, "reserve": 0},
        valid={"payload": valid * 16, "scale": valid * 8},
        page_tail={"payload": tail * 16, "scale": tail * 8, "total": tail * 24},
    )


def test_window_group_metadata_uses_resident_window_pages_not_full_sequence():
    parsed = compressed_case()
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "layers": [0, 1], "components": ["window"],
            "tokens_per_page": 3, "bytes_per_group_page": 96,
            "resident_tokens": 5, "metadata_bytes_per_page": 32,
        }],
    })
    memory = Memory(replace(parsed, runtime=runtime))
    group = memory.pages.groups[0]
    assert group.pages(9, parsed.runtime.max_model_len) == 2
    assert group.touched_pages(9, parsed.runtime.max_model_len) == 2
    # Six physical rows and five valid rows per layer, plus two metadata records.
    _byte_views(
        memory.cache_allocation(9),
        page_groups={"component_payload": 192, "valid_component_payload": 160},
        allocated={"payload": 352, "scale": 0, "reserve": 0, "metadata": 64},
        valid={"payload": 320, "scale": 0}, access={"metadata": 64},
        page_tail={"payload": 32, "metadata": 0},
    )


def test_grouped_page_keeps_known_scale_page_tail_and_access_metadata():
    parsed = compressed_case()
    owner, consumer = parsed.model.layer_plans
    model = replace(
        parsed.model,
        quant=replace(parsed.model.quant, index_dtype="int8",
                      index_scale_dtype="fp16"),
        layer_plans=(
            replace(owner, owns_index_k=True, index_dim=4, index_heads=2,
                    index_input="compressed"),
            replace(consumer, owns_long_kv=True, kv_owner_layer_id=1)))
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False,
        "cache_page_groups": [{
            "members": [
                {"layer": 0, "component": "compressed_kv"},
                {"layer": 0, "component": "index_k"},
                {"layer": 1, "component": "compressed_kv"},
            ],
            "tokens_per_page": 8, "bytes_per_group_page": 100,
        }],
    })
    memory = Memory(replace(parsed, model=model, runtime=runtime))
    # Two 100 B group pages include two int8 index rows (4 B payload + 2 B scale).
    _byte_views(
        memory.cache_allocation(9),
        allocated={"total": 712, "payload": 584, "scale": 4, "reserve": 124},
        valid={"total": 492, "payload": 488, "scale": 4},
        page_tail={"total": 220, "payload": 96, "scale": 0, "reserve": 124},
        access={"total": 492},
    )


@pytest.mark.parametrize("scan,accessed_pages", [(None, 1), ("all", 66)])
def test_fixed_page_reservation_reads_metadata_for_declared_scan(scan, accessed_pages):
    parsed = compressed_case()
    group = {
        "layers": [0], "components": ["window"], "tokens_per_page": 128,
        "bytes_per_group_page": 462_848, "resident_tokens": 128,
        "max_in_flight_tokens": 8192, "recycling_pages": 1,
        "allocation": "max_model_len", "metadata_bytes_per_page": 32,
    }
    if scan is not None:
        group["metadata_scan"] = scan
    runtime = replace(parsed.runtime, kv_block_size=8, memory={
        "derive_workspace": False, "cache_page_groups": [group],
    })
    memory = Memory(replace(parsed, runtime=runtime))
    group = memory.pages.groups[0]
    assert group.pages(1, parsed.runtime.max_model_len) == 66
    assert group.touched_pages(1, parsed.runtime.max_model_len) == 1
    _byte_views(memory.cache_allocation(1), allocated={"metadata": 66 * 32},
                access={"metadata": accessed_pages * 32}, page_tail={"metadata": (66 - accessed_pages) * 32})


def _context_owner_case(*, page_group):
    raw = case(tp=2)
    raw["model"].update(
        layers=1, num_kv_heads=1, head_dim=8,
        metadata={"cache_representations": {
            "gqa_kv": {"dtype": "int8", "group_size": 8,
                       "scale_dtype": "fp32"}}})
    raw["runtime"].update(
        tp=2, cp=2, dcp=2, cp_plan="dcp", kv_block_size=1, max_model_len=8,
        gpu_memory_utilization=1.0, memory={"derive_workspace": False})
    if page_group:
        raw["runtime"]["memory"]["cache_page_groups"] = [{
            "layers": [0], "components": ["gqa_kv"], "tokens_per_page": 3,
            "bytes_per_group_page": 72,
        }]
    return parse_case(raw)


def _context_owner_memory(budgets, *, page_group=False):
    parsed = _context_owner_case(page_group=page_group)
    policy = MemoryPolicy(**parsed.runtime.memory, rank_token_owners=(2, 1))
    return Memory(_with_hbm(parsed, budgets), policy)


def test_uneven_token_owners_share_page_size_and_have_distinct_valid_rows():
    grouped = _context_owner_memory((1000, 1000), page_group=True)
    assert grouped.cache_bytes(3) == (72.0,)
    for view, rows in zip(grouped.rank_cache_allocation(3), (2, 1), strict=True):
        tail = 3 - rows
        _byte_views(view, allocated={"payload": 48, "scale": 24},
                    valid={"payload": rows * 16, "scale": rows * 8},
                    page_tail={"payload": tail * 16, "scale": tail * 8})


@pytest.mark.parametrize("budgets,admitted", [
    ((48, 24), 3), ((48, 23), 2), ((47, 24), 1), ((10 ** 9, 23), 2),
])
def test_uneven_token_owners_each_require_their_own_capacity(budgets, admitted):
    memory = _context_owner_memory(budgets)
    assert memory.rank_capacity[0][0] == list(budgets)
    assert memory.capacity == [[min(budgets)]]
    assert memory.admissible_tokens(0, "r", 4) == admitted
    before = memory.budget_version
    assert memory.resize(0, "r", 3) is (admitted == 3)
    if admitted == 3:
        for view, rows in zip(memory.rank_cache_allocation(3), (2, 1), strict=True):
            _byte_views(view, allocated={"payload": rows * 16, "scale": rows * 8})
        assert memory.live_rank[0][0] == [48.0, 24.0]
        assert [memory.occupied_bytes(0, 0, rank=rank) for rank in range(2)] == [48.0, 24.0]
        plans = memory.rank_capacity_plans(0, 0, tokens=3)
        assert [plan.resolve()["categories"]["cache"] for plan in plans] == [48.0, 24.0]
    else:
        assert memory.occupied_bytes(0, 0, rank=0) == 0.0
        assert memory.live == memory.peak_combined == [[0.0]]
        assert memory.budget_version == before
        assert not memory.owners


def test_concurrent_workspace_copies_and_temporary_dequant_change_admission():
    parsed = parse_case(case())
    base = Memory(parsed, MemoryPolicy(derive_workspace=True,
                                       concurrent_workspace_copies=1))
    doubled = Memory(parsed, MemoryPolicy(derive_workspace=True,
                                         concurrent_workspace_copies=2))
    assert doubled.workspace == [2 * value for value in base.workspace]
    assert doubled.capacity == [[cap - value for cap, value in zip(
        base.capacity[0], base.workspace)]]

    temporary = Memory(parsed, MemoryPolicy(
        derive_workspace=True, concurrent_workspace_copies=1,
        temporary_dequant_bytes=12345))
    assert temporary.temporary_dequant == [12345]
    assert temporary.capacity == [[cap - 12345 for cap in base.capacity[0]]]
    report = temporary.to_dict()
    assert report["temporary_dequant_bytes_per_stage_rank"] == [12345]
    assert report["fixed_plan"] is True
    assert report["fallback"] == "none"
    with pytest.raises(ValueError, match="concurrent_workspace_copies"):
        Memory(parsed, MemoryPolicy(concurrent_workspace_copies=0))
    with pytest.raises(ValueError, match="temporary_dequant_bytes"):
        Memory(parsed, MemoryPolicy(temporary_dequant_bytes=-1))


def test_quantization_and_reserve_declarations_reach_allocation_geometry():
    parsed = parse_case(case())
    bf16 = Memory(parsed, MemoryPolicy(derive_workspace=False))
    quant = replace(bf16.case.model.quant or QuantSpec("bf16"), name="int8-kv",
                    kv_dtype="int8")
    int8 = Memory(replace(bf16.case, model=replace(bf16.case.model, quant=quant)),
                  MemoryPolicy(derive_workspace=False))
    assert int8.cache_bytes(8)[0] < bf16.cache_bytes(8)[0]
    assert int8.cache_allocation(8)["allocated"]["total_bytes_per_stage_rank"][0] < \
        bf16.cache_allocation(8)["allocated"]["total_bytes_per_stage_rank"][0]
    # Persistent capacity is unchanged, but the same span now admits under a
    # device capacity that rejected the BF16 cache representation.
    tight = _with_hbm(bf16.case, (bf16.cache_bytes(8)[0] - 1,))
    bf16_tight = Memory(tight, MemoryPolicy(derive_workspace=False))
    int8_tight = Memory(replace(tight, model=replace(tight.model, quant=quant)),
                        MemoryPolicy(derive_workspace=False))
    assert bf16_tight.resize(0, "r", 8) is False
    assert int8_tight.resize(0, "r", 8) is True

    graph = Memory(parsed, MemoryPolicy(derive_workspace=False,
                                        graph_pool_bytes=7 * MIB))
    assert graph.capacity[0][0] == bf16.capacity[0][0] - 7 * MIB
    assert graph.to_dict()["reserve_breakdown"]["graph_pool_bytes"] == 7 * MIB


def test_physical_rank_capacity_plans_preserve_heterogeneous_dies():
    memory = budget_memory((1000, 2000), tp=2)
    assert memory.capacity == [[1000.0]]
    mounted = [die.hbm_bytes for die in memory._stage_dies(0, 0)]
    plans = memory.rank_capacity_plans(0, 0, tokens=1)
    assert [plan.capacity_bytes for plan in plans] == mounted
    assert all(plan.resolve()["feasible"] for plan in plans)
    assert memory.rank_capacity_plan(0, 0, tokens=1, rank=0).capacity_bytes == mounted[0]
    assert memory.to_dict()["physical_rank_capacity_bytes_per_replica_stage"] == [[mounted]]


def test_fixed_plan_reports_oom_and_preserves_selected_representation():
    parsed = parse_case(case())
    base = Memory(parsed, MemoryPolicy(derive_workspace=False))
    tight = _with_hbm(base.case, (-1,), weights=base.weights)
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    report = memory.to_dict()
    assert (report["status"], report["fixed_plan"], report["fallback"]) == ("oom", True, "none")
    assert not memory.resize(0, "r", 1)
    assert memory.weights == base.weights
    assert memory.case.model.quant == tight.model.quant


def test_request_admission_decision_names_waitable_and_permanent_shortfall():
    memory = budget_memory(1024)
    assert memory.resize(0, "r", 4) is True
    assert memory.occupied_bytes(0, 0) == 512.0
    version = memory.budget_version
    wait = memory.admission_decision(0, "other", 8)
    assert wait["ok"] is False
    assert wait["wakeable"] is True and wait["permanent"] is False
    assert wait["reason"] == "waitable_release"
    assert wait["shortfall_bytes_per_stage_rank"] == [[512.0]]
    assert wait["releasable_bytes_per_stage_rank"] == [[512.0]]
    assert wait["budget_version"] == version
    assert memory.occupied_bytes(0, 0) == 512.0
    permanent = memory.admission_decision(0, "other", 16)
    assert permanent["ok"] is False
    assert permanent["wakeable"] is False and permanent["permanent"] is True
    assert permanent["reason"] == "permanent_budget"
    assert permanent["shortfall_bytes_per_stage_rank"] == [[1536.0]]
    assert memory.budget_version == version
    memory.release(0, "r")
    assert memory.admission_decision(0, "other", 8)["ok"] is True
    assert memory.resize(0, "other", 8) is True
    assert memory.occupied_bytes(0, 0) == 1024.0


def test_grant_decision_names_waitable_release_and_permanent_deficit():
    memory = budget_memory(4096)
    event = object()
    version = memory.budget_version
    big = memory.grant_decision(0, [{
        "name": "big", "bytes": 8192, "physical_key": "P", "storage": "S",
        "ranges": (0, 8192), "generation": 1}])
    assert big["ok"] is False
    assert big["reason"] == "permanent_budget"
    assert big["wakeable"] is False and big["permanent"] is True
    assert big["budget_version"] == version
    assert big["shortfall_bytes_per_stage_rank"] == [[4096.0]]
    assert memory.occupied_bytes(0, 0) == 0.0
    assert memory.grant_bindings == {}
    assert memory.release_events == []

    assert memory.grant_batch(0, [{
        "name": "A", "bytes": 4096, "physical_key": "P", "storage": "S",
        "ranges": (0, 4096), "generation": 1, "event": event}]) is True
    held = memory.budget_version
    wait = memory.grant_decision(0, [{
        "name": "B", "bytes": 4096, "physical_key": "Q", "storage": "S",
        "ranges": (0, 4096), "generation": 2}])
    assert wait["ok"] is False
    assert wait["reason"] == "waitable_release"
    assert wait["wakeable"] is True and wait["permanent"] is False
    assert wait["shortfall_bytes_per_stage_rank"] == [[4096.0]]
    assert wait["releasable_bytes_per_stage_rank"] == [[4096.0]]
    assert memory.budget_version == held
    assert memory.occupied_bytes(0, 0) == 4096.0
    assert any(row["kind"] == "grant" and row["name"] == "A"
               and row["event"] is event for row in wait["wake_events"])
    assert memory.release_grant(0, "A", generation=1, event=event) is True
    assert memory.grant_batch(0, [{
        "name": "B", "bytes": 4096, "physical_key": "Q", "storage": "S",
        "ranges": (0, 4096), "generation": 2}]) is True
    assert memory.occupied_bytes(0, 0) == 4096.0


@pytest.mark.parametrize("filename,tail,weight_scales", [
    ("qwen38-a5.json", 0, True), ("glm53-a5.json", 148_228_096, False),
])
def test_catalog_rank_plan_conserves_weight_and_cache_bytes(filename, tail, weight_scales):
    memory = Memory(parse_case(json.loads((EXAMPLE_CASES / filename).read_text(encoding="utf-8"))))
    span = memory.case.runtime.max_model_len
    report = memory.allocation_report(span)
    assert report["fixed_plan"] is True and report["fallback"] == "none"
    weights = report["weights"]
    assert weights["total_bytes_per_stage_rank"] == memory.weights
    assert weights["payload_bytes_per_stage_rank"] == memory.weights_payload
    assert [payload + metadata for payload, metadata in zip(
        weights["payload_bytes_per_stage_rank"], weights["quant_metadata_bytes_per_stage_rank"],
    )] == memory.weights
    assert (memory.weights_quant_metadata[0] > 0) is weight_scales
    cache = report["cache"]
    assert cache["allocated"]["total_bytes_per_stage_rank"] == list(memory.cache_bytes(span))
    assert cache["page_tail"]["total_bytes_per_stage_rank"] == [tail]
    valid = cache["valid"]
    assert valid["total_bytes_per_stage_rank"] == [payload + scale for payload, scale in zip(
        valid["payload_bytes_per_stage_rank"], valid["scale_bytes_per_stage_rank"],
    )]
    assert all(value >= 0 for row in report["headroom_bytes_per_replica_stage"] for value in row)
    if tail:
        assert cache["access"]["total_bytes_per_stage_rank"][0] < sum(memory.cache_bytes(span))
    else:
        plan = memory.rank_capacity_plan(0, 0, tokens=span).resolve()
        assert plan["feasible"]
        assert plan["categories"]["cache"] == memory.cache_bytes(span)[0]


def test_catalog_compressed_cache_reports_physical_and_access_scales():
    filename = "v41-a5-dp4.json"
    memory = Memory(parse_case(json.loads((EXAMPLE_CASES / filename).read_text(encoding="utf-8"))))
    allocation = memory.cache_allocation(memory.case.runtime.max_model_len)
    allocated, access = allocation["allocated"], allocation["access"]
    assert allocated["scale_bytes_per_stage_rank"][0] > 0
    assert access["scale_bytes_per_stage_rank"][0] > 0
    assert allocated["reserve_bytes_per_stage_rank"][0] > 0
    assert [payload + scale + reserve + metadata for payload, scale, reserve, metadata in zip(
        allocated["payload_bytes_per_stage_rank"], allocated["scale_bytes_per_stage_rank"],
        allocated["reserve_bytes_per_stage_rank"], allocated["metadata_bytes_per_stage_rank"],
    )] == allocated["total_bytes_per_stage_rank"]


def test_tile_solver_budgets_fp32_ub_upcast_and_simt_carveout():
    """AIV FP32 (4B/elem) UB upcast and 16 KiB SIMT UB carveout enforce hardware ceilings."""
    from types import SimpleNamespace
    from mindie_sim.prediction.runtime.tile_solver import (
        LiveBufferSpec, TileCandidate,
        TileInfeasibleError, TilePipelineSpec, TileShape, solve, ub_working_bytes,
    )

    assert ub_working_bytes(4096, 2, upcast_fp32=True) == 4096 * 4
    cand = TileCandidate(
        "resident_1pass", TileShape(16, 64, 64), 1, (1, 1), 1, 1,
        (LiveBufferSpec("norm_buf", "ub", 46 * 1024, 0, 1, upcast_fp32=True, elements_per_owner=46 * 1024),),
        uses_simt=True,
    )
    with pytest.raises(TileInfeasibleError):
        solve(TilePipelineSpec("norm_upcast", (cand,)), None, SimpleNamespace(buffers=(("ub", 192 * 1024),), compute_units=1, simt_ub_carveout_bytes=16 * 1024))
    sol = solve(TilePipelineSpec("norm_upcast", (cand,)), None, SimpleNamespace(buffers=(("ub", 200 * 1024),), compute_units=1, simt_ub_carveout_bytes=16 * 1024))
    assert next(p for p in sol.pool_peaks if p.pool == "ub").peak_bytes == 46 * 1024 * 4


def test_native_norm_declares_fp32_working_tiles():
    from mindie_sim.prediction.operators.vector import family_ascend_norm
    from mindie_sim.schema.operators import OpSpec

    def _make_norm(dtype: str, family: str = "add_rmsnorm", width: int = 8192):
        elem_b = 4 if dtype == "fp32" else 2
        read_b = (2 * width + width) * elem_b if family == "add_rmsnorm" else (width + width) * elem_b
        write_b = 2 * width * elem_b if family == "add_rmsnorm" else width + 256
        meta = [("norm_has_weight", True), ("norm_eps", 1e-6), ("reduction_width", width)]
        if family != "add_rmsnorm":
            meta.extend([
                ("output_format", "fp8_e4m3"), ("activation_group_size", 32),
                ("activation_scale_format", "fp8_e8m0"),
                ("activation_scale_shape", (1, 128, 2)), ("activation_scale_bytes", 256),
            ])
        raw = OpSpec(
            op_id=f"{family}_{dtype}", family=family, phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(1, width), local_shape=(1, width),
            padded_shape=(1, width), launch_shape=(1, width),
            hbm_bytes=float(read_b + write_b), mte2_bytes=float(read_b),
            mte3_bytes=float(write_b), vector_ops=float(6 * width if family == "add_rmsnorm"
                                                        else 5 * width + 4 * width + 2 * 256),
            dtype=dtype, layout="contiguous", recipe=family, metadata=tuple(meta),
        )
        return family_ascend_norm.lower_op(raw)[-1]

    bf16_op = _make_norm("bf16")
    fp16_op = _make_norm("fp16")
    fp32_op = _make_norm("fp32")
    rmsq_op = _make_norm("bf16", family="rmsnorm_quant")
    assert dict(bf16_op.metadata)["native_norm_kind"] == "AddRmsNormBias"
    assert dict(bf16_op.metadata)["native_norm_fp32_working_tiles"] == 2
    assert dict(fp16_op.metadata)["native_norm_fp32_working_tiles"] == 2
    assert dict(fp32_op.metadata)["native_norm_fp32_working_tiles"] == 1
    assert dict(rmsq_op.metadata)["native_norm_fp32_working_tiles"] == 1


def test_tile_solver_honors_zero_vs_32kib_simt_ub_carveout():
    """TileSolver honors simt_ub_carveout_bytes=0 verbatim (no hidden 16 KiB clamp) vs 32 KiB -> 216 KiB."""
    from types import SimpleNamespace
    from _support.axes import GEN5_DT_SKU
    from _support.hardware import sku_case
    from mindie_sim.prediction.runtime.tile_solver import (
        LiveBufferSpec, TileCandidate, TileInfeasibleError, TilePipelineSpec, TileShape, solve,
    )

    cand_240k = TileCandidate(
        "resident_1pass", TileShape(16, 64, 64), 1, (1, 1), 1, 1,
        (LiveBufferSpec("buf", "ub", 240 * 1024, 0, 1),), uses_simt=True,
    )
    cand_216k = TileCandidate(
        "resident_1pass", TileShape(16, 64, 64), 1, (1, 1), 1, 1,
        (LiveBufferSpec("buf", "ub", 216 * 1024, 0, 1),), uses_simt=True,
    )
    hw_zero = SimpleNamespace(buffers=(("ub", 248 * 1024),), compute_units=1, simt_ub_carveout_bytes=0)
    sol_zero = solve(TilePipelineSpec("simt_zero", (cand_240k,), simt_ub_carveout_bytes=0), None, hw_zero)
    ub_zero = next(p for p in sol_zero.pool_peaks if p.pool == "ub")
    assert ub_zero.capacity_bytes == 248 * 1024 and ub_zero.peak_bytes == 240 * 1024

    hw_32k, = sku_case(GEN5_DT_SKU).machine.dies
    assert hw_32k.simt_ub_carveout_bytes == 32 * 1024
    plain = solve(TilePipelineSpec("plain", (replace(cand_240k, uses_simt=False),)), None, hw_32k)
    assert next(p for p in plain.pool_peaks if p.pool == "ub") == ub_zero
    with pytest.raises(TileInfeasibleError):
        solve(TilePipelineSpec("simt_32k_reject", (cand_240k,)), None, hw_32k)
    sol_32k = solve(TilePipelineSpec("simt_32k_fit", (cand_216k,)), None, hw_32k)
    ub_32k = next(p for p in sol_32k.pool_peaks if p.pool == "ub")
    assert ub_32k.capacity_bytes == 216 * 1024 and ub_32k.peak_bytes == 216 * 1024


def test_rank_capacity_plans_use_per_rank_shards_global_dcp_and_stage_offload_fallback():
    """Capacity plans use per-rank expert shards, typed RankCoord DCP/EP axes, and per-stage offload fallback."""
    import copy
    from _support.memory_cases import case, sparse_mla_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.parallel import ReplicaLocalRank

    # 1. Per-rank expert weights match rank_weights instead of repeating stage max across lighter EP ranks.
    raw_asym = case(tp=2, ep=2)
    raw_asym["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=2)
    raw_asym["model"].update(ffn="moe", num_experts=5, top_k=1, moe_intermediate_size=128)
    mem_asym = Memory(parse_case(raw_asym))
    plans_asym = mem_asym.rank_capacity_plans(0, 0, include_cache=False)
    assert [int(next(a.bytes for a in p.allocations if a.name == "weights")) for p in plans_asym] == [
        int(w) for w in mem_asym.rank_weights[0][0]
    ] == [162752, 113600]

    # 2. Capacity EP die-to-group indexing matches PhysicalRankMap.coord_for under composite tp*pcp/ep layouts.
    raw_ep = case(tp=2, ep=3)
    raw_ep["runtime"].update(dp=1, tp=2, pp=1, ep=3, pcp=3)
    raw_ep["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=6)
    raw_ep["model"].update(ffn="moe", num_experts=7, top_k=2, moe_intermediate_size=128)
    parsed_ep = parse_case(raw_ep)
    mem_ep = Memory(parsed_ep)
    placement_ep = parsed_ep.machine.validate_placement(parsed_ep.runtime.parallel)
    assert [mem_ep._ep_rank_for_die(0, 0, lane, placement_ep) for lane in range(6)] == [
        placement_ep.coord_for(0, 0, ReplicaLocalRank(lane)).ep_rank for lane in range(6)
    ] == [0, 0, 1, 1, 2, 2]

    # 3. Cross-DP DCP capacity plans match live_rank for both zero-based and offset physical rank IDs.
    for devices in (None, [{"rank": rk} for rk in range(8, 16)]):
        raw_dcp = case(dp=8)
        raw_dcp["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=8)
        if devices is not None:
            raw_dcp["machine"]["devices"] = devices
        raw_dcp["runtime"].update(dcp=8, kv_block_size=1)
        mem_dcp = Memory(parse_case(raw_dcp))
        plans_dcp = [mem_dcp.rank_capacity_plan(rep, 0, tokens=1, rank=0) for rep in range(8)]
        assert mem_dcp.resize(0, "r", 1)
        assert [int(next(a.bytes for a in p.allocations if a.name == "cache_allocation")) for p in plans_dcp] == [
            int(rep[0][0]) for rep in mem_dcp.live_rank
        ] == [128, 0, 0, 0, 0, 0, 0, 0]

    # 4. Empty pages.groups sparse-offload fallback applies num_shared_layers per PP stage, not globally.
    raw_off = sparse_mla_case(
        context_tokens=8,
        tp=1,
        offloading={
            "implementation": "vllm_ascend_sparse_kv_offload",
            "options": {
                "use_fused_copy_sfa": False,
                "dram_size_per_dp_gb": 64.0,
                "layerwise_kv_offload": True,
                "num_shared_layers": 1,
            },
        },
    )
    raw_off["runtime"]["pp"] = 2
    base_plans = copy.deepcopy(raw_off["model"]["layer_plans"])
    raw_off["model"]["layer_plans"] = [
        dict(base_plans[0], index=0),
        dict(base_plans[1], index=1),
        dict(base_plans[0], index=2),
        dict(base_plans[1], index=3),
    ]
    raw_off["model"]["layers"] = 4
    raw_off["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=2)
    mem_off = Memory(parse_case(raw_off))
    rep_full = mem_off._max_model_len_report()["sparse_kv_offload"]
    mem_empty = object.__new__(Memory)
    mem_empty.__dict__.update(mem_off.__dict__)
    empty_pages = copy.copy(mem_off.pages)
    empty_pages.groups = ()
    mem_empty.pages = empty_pages
    rep_empty = mem_empty._max_model_len_report()["sparse_kv_offload"]
    assert rep_empty["npu_limit_blocks"] == rep_full["npu_limit_blocks"]


def test_draft_kv_separates_dcp_interleave_from_physical_page_block_size():
    """Inherited draft KV shards DCP by interleave size while non-inherited draft KV rounds pages by kv_block_size."""
    from mindie_sim.catalog.cache_ownership import draft_cache_components
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.structure import parse_case

    raw_inh = compose_case(
        "glm-5.3", hardware="a3", tp=8, ep=8, dcp=2, layers=4,
        block_size=128, mtp_proposal_tokens=1,
    )
    raw_inh["runtime"]["metadata"]["cp_kv_cache_interleave_size"] = 16
    parsed_inh = parse_case(raw_inh)
    per_rank = [
        int(draft_cache_components(
            parsed_inh.model, parsed_inh.runtime.parallel, draft_kind="mtp",
            sequence_tokens=32, runtime=parsed_inh.runtime, rank=rk,
            capacity_padded=False,
        )["draft.mla_kv"])
        for rk in (0, 1)
    ]
    assert per_rank == [10496, 10496]

    raw_non = compose_case(
        "qwen3.5-35b-a3b", hardware="a3", layers=4, tp=2, ep=2,
        block_size=128, mtp_proposal_tokens=1,
    )
    raw_non["runtime"].setdefault("metadata", {})["cp_kv_cache_interleave_size"] = 16
    mem_non = Memory(parse_case(raw_non))
    bytes_1 = mem_non.resident_memory_receipt(sequence_tokens=1, batch_per_dp=1).dies[0].draft_cache_bytes
    bytes_128 = mem_non.resident_memory_receipt(sequence_tokens=128, batch_per_dp=1).dies[0].draft_cache_bytes
    assert bytes_1 == bytes_128 == 131072

    # Cross-DP pp>1 rank_cache_allocation avoids double-counting shard 0 and rank=None honors skewed rank_token_owners.
    from _support.memory_cases import case as _mem_case
    from mindie_sim.catalog.draft_spec import DraftSpec
    from mindie_sim.catalog.types import AuxiliarySpec

    mla_draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=64, intermediate_size=128,
        num_heads=4, num_kv_heads=1, head_dim=16, q_width=64, kv_width=16,
        attention_kind="mla", q_lora_rank=64, kv_lora_rank=512,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64, dcp_sharded=True,
    )
    raw_pp = _mem_case(dp=2, pp=2, tp=1)
    raw_pp["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=4)
    raw_pp["runtime"].update(dcp=2, kv_block_size=1, metadata={"spec_method": "mtp", "mtp_proposal_tokens": 1, "cp_kv_cache_interleave_size": 1})
    c_pp = parse_case(raw_pp)
    mem_pp = Memory(replace(c_pp, model=replace(c_pp.model, auxiliary=AuxiliarySpec(mtp=mla_draft))))
    assert [float(a["components"][1]["valid"]["draft_kv"]["total"]) for a in mem_pp.rank_cache_allocation(1)] == [1152.0, 0.0]

    raw_sk = _mem_case(tp=2)
    raw_sk["model"]["num_kv_heads"] = 1
    raw_sk["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=2)
    raw_sk["runtime"].update(
        dcp=2, kv_block_size=128,
        metadata={"spec_method": "mtp", "mtp_proposal_tokens": 1,
                  "cp_kv_cache_interleave_size": 16, "rank_token_owners": [(0, 256), (256, 256)]},
    )
    c_sk = parse_case(raw_sk)
    mem_sk = Memory(replace(c_sk, model=replace(c_sk.model, auxiliary=AuxiliarySpec(mtp=mla_draft))))
    assert mem_sk._non_inherited_draft_cache_bytes(mem_sk.case.model, 256, padded=True, rank=None)[0] == 256 * 1152
