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
    ("qwen38-a5.json", 0, True), ("glm53-a5.json", 2_883_584, False),
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
    assert [payload + scale + metadata for payload, scale, metadata in zip(
        allocated["payload_bytes_per_stage_rank"], allocated["scale_bytes_per_stage_rank"],
        allocated["metadata_bytes_per_stage_rank"],
    )] == allocated["total_bytes_per_stage_rank"]
