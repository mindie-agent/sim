"""Physical byte accounting across the ledger protocol and serving grants.

[Role] Exercise range normalization, transactions, physical aliasing, capacity
  domains and reader lifetimes through the ledger and its serving entry points.
[Invariants] Each domain charges the union of ranges per physical backing.
  Failed transactions leave usage, peaks and epochs unchanged. Release frees
  capacity only after every owning reference has completed.
[Timeouts & Locks] Bounded in-memory cases and a seeded byte-set oracle.
[Side-Effect Policy] Only test-local runtime, engine and capacity state change.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random

import pytest

from _support.memory_cases import budget_memory, case, linear_work
from mindie_sim.prediction.cache_path import CacheGeometry, CacheState
from mindie_sim.prediction.capacity import Allocation, AllocationPlan, Memory
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.runtime import engine
from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding, AllocationKey, BackingKey, CapacityDomainState, OccurrenceKey, PoolKey,
)
from mindie_sim.prediction.runtime.ledger import Ledger, LedgerError, normalise_ranges, ranges_bytes
from mindie_sim.prediction.runtime.state import RuntimeState
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import AllocationEffect

pytestmark = pytest.mark.tier0

PAGE = 4096


@dataclass
class LedgerState:
    uid: str = "runtime-test"
    capacity_domains: dict[str, CapacityDomainState] = field(default_factory=dict)


@dataclass
class Completion:
    sequence: int
    done: bool = True
    generation: int = 0
    time: float | None = 0.0
    runtime_uid: str = "runtime-test"


def _fresh(capacity=2 * PAGE):
    state = LedgerState()
    pool = _pool(state, "memory", capacity)
    return state, pool, state.capacity_domains["memory"]


def _pool(state, domain, capacity):
    pool = PoolKey(physical_domain=domain, rank=0, tier="hbm", name="hbm")
    Ledger.ensure_domain(state, domain, capacities={pool: float(capacity)})
    return pool


def _occurrence(ordinal=0, *, launch=1):
    return OccurrenceKey(runtime_uid="runtime-test", launch=launch, ordinal=ordinal)


def _key(index=0, *, generation=0, launch=1):
    return AllocationKey(occurrence=_occurrence(index, launch=launch), scope_kind="invocation",
                         scope_ordinal=index, symbol=index, generation=generation)


def _binding(pool, index, size, *, backing=None, span=None, **attributes):
    return AllocationBinding(key=_key(index), pool=pool, amount=float(size), backing=backing,
                             byte_ranges=() if span is None else (span,), **attributes)


def _backing(name):
    return BackingKey(physical_domain="memory", rank=0, tier="hbm", storage_id=name, generation=0)


def _acquire(state, *bindings, sequence=1):
    return Ledger.try_acquire(state, bindings, holder=bindings[0].key.occurrence, at=Completion(sequence))


@pytest.mark.parametrize("raw,extent,expected", [
    ([(0, 10), (5, 15), (20, 20)], 0, ((0, 15),)),
    ([(8, 12), (0, 4), (4, 8)], 0, ((0, 12),)),
    ([(3, 4), (9, 10)], 0, ((3, 4), (9, 10))),
    (None, 7, ((0, 7),)),
])
def test_range_normalization_is_canonical(raw, extent, expected):
    result = normalise_ranges(raw, extent=extent)
    assert result == expected
    assert ranges_bytes(result) == sum(end - start for start, end in expected)


def test_overlapping_backings_charge_the_union():
    state, pool, domain = _fresh()
    backing = _backing("kv")
    assert _acquire(state, _binding(pool, 0, PAGE, backing=backing, span=(0, PAGE)),
                    _binding(pool, 1, PAGE, backing=backing, span=(2048, 6144))).granted
    assert (domain.current[pool], domain.epoch) == (6144, 1)
    release = Ledger.release(state, _key(), after=Completion(2))
    assert release.released and release.freed_by_pool == ((pool, 2048.0),)
    assert domain.current[pool] == ranges_bytes(domain.backings[backing].segments) == PAGE


@pytest.mark.parametrize("amounts,error", [
    ((PAGE, 2 * PAGE), None), ((1, 1), "allocation_duplicate"),
    ((float("nan"),), "allocation_amount_invalid"),
])
def test_rejected_batches_preserve_ledger_state(amounts, error):
    state, pool, domain = _fresh()
    if error:
        with pytest.raises(LedgerError, match=error):
            _acquire(state, *(_binding(pool, 0, amount) for amount in amounts))
    else:
        first, second = amounts
        grant = _acquire(state, _binding(pool, 0, first), _binding(pool, 1, second))
        assert not grant.granted and grant.blocked_pools == (pool,)
    assert not domain.current and not domain.grants and not domain.backings and not domain.peaks
    assert domain.epoch == 0


def test_exact_capacity_acquisition_and_release_preserve_peak():
    state, pool, domain = _fresh(PAGE)
    assert _acquire(state, _binding(pool, 0, 3072)).granted
    assert _acquire(state, _binding(pool, 1, 1024), sequence=2).granted
    assert (domain.current[pool], domain.peaks[pool], domain.epoch) == (PAGE, PAGE, 2)
    assert not _acquire(state, _binding(pool, 2, 1), sequence=3).granted
    assert domain.epoch == 2
    assert Ledger.release(state, _key(1), after=Completion(4)).released
    assert (domain.current[pool], domain.peaks[pool]) == (3072, PAGE)
    assert Ledger.release(state, _key(), after=Completion(5)).released
    assert domain.current[pool] == 0


@pytest.mark.parametrize("reader_lease,amount", [(True, PAGE), (False, 1024)], ids=["lease", "event"])
def test_release_waits_for_reader_retirement_or_event_settlement(reader_lease, amount):
    state, pool, domain = _fresh()
    assert _acquire(state, _binding(pool, 0, amount)).granted
    last_use = Completion(9, done=reader_lease, time=0.0 if reader_lease else None)
    if reader_lease:
        reader = Ledger.retain_reader(state, _key(), reader=last_use)
    boundary = Completion(10) if reader_lease else last_use
    result = Ledger.release(state, _key(), after=boundary)
    assert result.deferred and not result.released
    assert domain.current[pool] == amount
    if reader_lease:
        assert Ledger.retire_reader(state, reader).released
    else:
        assert Ledger.settle(state) == ()
        last_use.done, last_use.time = True, 9.0
        settled, = Ledger.settle(state, (last_use,))
        assert settled.released
    assert domain.current[pool] == 0


def test_stale_generation_cannot_read_or_release():
    state, pool, domain = _fresh()
    key = _key(generation=2)
    assert _acquire(state, replace(_binding(pool, 0, 1024), key=key)).granted
    with pytest.raises(LedgerError, match="generation_mismatch"):
        Ledger.retain_reader(state, key, reader=Completion(2), expected_generation=1)
    with pytest.raises(LedgerError, match="generation_mismatch"):
        Ledger.release(state, key, after=Completion(2), expected_generation=1)
    assert domain.current[pool] == 1024


@pytest.mark.parametrize("acquire,size,expected", [(False, PAGE, 0), (True, 2 * PAGE, 2 * PAGE)])
def test_only_owned_allocations_charge_capacity(acquire, size, expected):
    state, pool, domain = _fresh()
    assert _acquire(state, _binding(pool, 0, size, acquire=acquire)).granted
    assert domain.current.get(pool, 0) == expected
    assert Ledger.release(state, _key(), after=Completion(2)).released
    assert domain.current.get(pool, 0) == 0


@pytest.mark.parametrize("capacity,span,parent_first", [
    (2 * PAGE, (4096, 6144), False), (PAGE, (0, 1024), True),
])
def test_suballocation_shares_bytes_and_holds_its_parent(capacity, span, parent_first):
    state, pool, domain = _fresh(capacity)
    backing = _backing("workspace")
    assert _acquire(state, _binding(pool, 0, capacity, backing=backing, span=(0, capacity))).granted
    assert domain.current[pool] == capacity
    child = _binding(pool, 1, span[1] - span[0], backing=backing, span=span, reservation_parent=_key())
    assert _acquire(state, child, sequence=2).granted
    assert domain.current[pool] == capacity
    if parent_first:
        result = Ledger.release(state, _key(), after=Completion(3))
        assert result.deferred and not result.released and not domain.grants[_key()].released
    assert Ledger.release(state, child.key, after=Completion(4)).released
    if not parent_first:
        assert domain.current[pool] == capacity
    assert Ledger.release(state, _key(), after=Completion(5)).released
    assert domain.current[pool] == 0


@pytest.mark.parametrize("seed", [None, *range(25)])
def test_backing_union_matches_independent_byte_sets(seed):
    capacity = 16 if seed is None else 32
    state, pool, domain = _fresh(capacity)
    backings = [_backing(f"backing-{index}") for index in range(3)]
    if seed is None:
        attempts = [(backings[0], start, end) for start, end in [(0, 4), (3, 9), (8, 12), (14, 16), (1, 2)]]
    else:
        rng = Random(seed)
        attempts = []
        for _ in range(8):
            backing, start = rng.choice(backings), rng.randrange(16)
            attempts.append((backing, start, start + rng.randrange(1, 9)))

    def occupied(spans):
        return len({(backing, byte) for backing, start, end in spans for byte in range(start, end)})

    live = {}
    for index, (backing, start, end) in enumerate(attempts):
        candidate = occupied([*live.values(), (backing, start, end)])
        binding = _binding(pool, index, end - start, backing=backing, span=(start, end))
        result = _acquire(state, binding, sequence=index)
        assert result.granted is (candidate <= capacity)
        if result.granted:
            live[index] = (backing, start, end)
        assert domain.current[pool] == occupied(live.values())
    order = list(live)
    if seed is not None:
        rng.shuffle(order)
    for index in order:
        del live[index]
        assert Ledger.release(state, _key(index), after=Completion(100 + index)).released
        assert domain.current[pool] == occupied(live.values())


def test_ledger_domains_account_for_capacity_independently():
    state = LedgerState()
    for index, (domain, capacity, amount) in enumerate([("a", 1024, 1000), ("b", 2048, 2000)]):
        pool = _pool(state, domain, capacity)
        binding = replace(_binding(pool, 0, amount), key=_key(launch=index + 1))
        assert _acquire(state, binding, sequence=index + 1).granted
    usage = {domain: tuple(pools.values()) for domain, pools in Ledger.usage(state).items()}
    assert usage == {"a": (1000.0,), "b": (2000.0,)}


def test_capacity_domain_has_one_runtime_owner():
    _, _, domain = _fresh()
    with pytest.raises(LedgerError, match="memory_runtime_conflict"):
        Ledger.adopt(LedgerState(uid="runtime-other"), domain)


def test_cache_domain_has_one_runtime_owner():
    cache = CacheState(CacheGeometry(capacity_bytes=PAGE))
    state = RuntimeState(resources={})
    key = cache.bind_runtime(state)
    assert state.cache_domains[key] is cache and cache.runtime_uid == state.uid
    with pytest.raises(ModelingError, match="cache_runtime_conflict"):
        cache.bind_runtime(RuntimeState(resources={}))


def test_memory_adopts_runtime_domain_and_typed_reader_lifetime():
    memory = Memory(parse_case(case()))
    state = RuntimeState(resources={})
    domain = memory.bind_runtime(state)
    assert domain is state.capacity_domains["memory"]
    assert domain.owner_runtime_uid == state.uid and memory.runtime_state is state
    with pytest.raises(ValueError, match="memory_runtime_conflict"):
        memory.bind_runtime(RuntimeState(resources={}))
    pool = memory.ledger_pool(0, 0, 0)
    assert pool in domain.capacities
    binding = memory.typed_binding(0, "typed", stage=0, rank=0, amount=1, scope_ordinal=1,
                                   backing_key="typed-backing", ranges=((0, 1),))
    assert memory.grant_typed((binding,), holder=_occurrence(), at=Completion(1)).granted
    reader = memory.retain_reader_typed(binding.key, reader=Completion(2))
    assert memory.release_typed(binding.key, after=Completion(3)).deferred
    assert domain.current[pool] == 1
    assert Ledger.retire_reader(state, reader).released
    assert domain.current[pool] == 0


@pytest.mark.parametrize("reader", [False, True], ids=["direct-release", "last-use"])
def test_public_grants_update_the_adopted_ledger(reader):
    memory = Memory(parse_case(case()))
    domain = memory.bind_runtime(RuntimeState(resources={}))
    pool = memory.ledger_pool(0, 0, 0)
    grant = dict(name="kv", bytes=256, rank=0, stage=0, generation=1, physical_key="kv")
    assert memory.grant_batch(0, [grant])
    assert domain.current[pool] == 256
    last_use = Completion(7)
    if reader:
        assert memory.hold_for_async_consumer(0, "kv", "reader", grant="kv", generation=1, event=last_use)
    assert memory.release_grant(0, "kv")
    if reader:
        assert domain.current[pool] == 256
        assert memory.release_async_consumer(0, "kv", "reader", generation=1, event=last_use)
    assert domain.current[pool] == 0


def test_engine_admits_against_the_adopted_memory_ledger():
    memory = Memory(parse_case(case()))
    runtime_engine = engine.Engine([engine.Resource("hbm", 1)], pools={"hbm": 1024 ** 3})
    memory.bind_runtime(runtime_engine)
    assert runtime_engine.bind_memory(memory)
    task = runtime_engine.submit("work", {"hbm": 1}, acquire=(engine.Allocation("kv", "hbm", 256),),
                                 memory=memory, memory_pools=("hbm",))
    runtime_engine.advance(until=(task,))
    assert task.done
    assert memory.ledger_domain.peaks[memory.ledger_pool(0, 0, 0)] >= 256


def _grant(name, backing, generation, *, span=(0, PAGE), storage="S", **attributes):
    return dict(name=name, bytes=span[1] - span[0], physical_key=backing, storage=storage,
                ranges=span, generation=generation, **attributes)


def _selected(name, **attributes):
    return Allocation(name, PAGE, "workspace", **attributes)


def test_per_item_grant_bindings_preserve_generation_and_release_identity():
    memory = budget_memory(4 * PAGE, tp=2)
    issue = object()
    grants = [
        _grant("src", "K:page1", 3, storage="K", owner="producer:src", tier="hbm",
               rank=1, stage=0, host_bytes=512, event=issue),
        _grant("dst", "V:page1", 4, span=(8192, 12288), storage="V", owner="consumer:dst",
               tier="hbm", rank=1, stage=0, host_bytes=0, event=issue),
    ]
    version = memory.budget_version
    assert memory.grant_batch(0, []) and memory.budget_version == version
    assert memory.grant_batch(0, grants, stage=0)
    assert memory.budget_version > version
    src, dst = (memory.grant_bindings[(0, name)] for name in ("src", "dst"))
    assert (src["storage"], src["generation"], src["ranges"]) == ("K", 3, (0, PAGE))
    assert (dst["storage"], dst["generation"]) == ("V", 4)
    held = memory.budget_version
    assert not memory.release_grant(0, "src", generation=4)
    assert not memory.release_grant(0, "src", event=object())
    assert not memory.grant_batch(0, [("src", PAGE)])
    assert memory.budget_version == held and memory.live_grants(0, 0) == 2 * PAGE
    assert memory.release_grant(0, "src", generation=3)
    assert memory.live_grants(0, 0) == PAGE
    holders = memory.outstanding_holders()
    assert holders["budget_version"] > version and [row["name"] for row in holders["grants"]] == ["dst"]
    released = memory.drain_release_events()[-1]
    assert (released["kind"], released["generation"]) == ("grant", 3)
    assert memory.release_grant(0, "dst", generation=4) and memory.live_grants(0, 0) == 0
    assert memory.grant_batch(0, [("one", PAGE), ("two", PAGE)], owner="same-task")
    assert memory.live_grants(0, 0) == memory.release_grants(0, ["one", "two"]) == 2 * PAGE
    assert memory.live_grants(0, 0) == 0


def test_selected_acquisitions_are_incremental_and_aliases_hold_backing():
    memory = budget_memory(3 * PAGE)
    for name in ("a", "b"):
        assert memory.attach_allocation_plan(AllocationPlan((_selected(name),)))
    assert memory.selected_reserved == [[2 * PAGE]]
    assert not memory.grant_batch(0, [("extra", 2 * PAGE)])
    assert memory.selected_reserved == [[2 * PAGE]] and memory.granted == [[0]]
    assert memory.release_selected_allocation("a")
    assert memory.selected_reserved == [[PAGE]]
    alias = AllocationPlan((_selected("backing"), _selected("view", alias_of="backing")))
    assert memory.attach_allocation_plan(alias, replace=True) and memory.selected_reserved == [[PAGE]]
    assert memory.release_selected_allocation("backing") and memory.selected_reserved == [[PAGE]]
    assert memory.release_selected_allocation("view") and memory.selected_reserved == [[0]]
    assert memory.grant_batch(0, [("extra", 2 * PAGE)])
    assert memory.live_grants(0, 0) == memory.release_grants(0, ["extra"]) == 2 * PAGE
    assert memory.live_grants(0, 0) == 0 and memory.peak_combined[0][0] >= 2 * PAGE


def test_selected_and_onchip_effects_use_their_physical_tier_budget():
    memory = budget_memory(2 * PAGE)
    plan = AllocationPlan((_selected("workspace"), _selected("alias", alias_of="workspace")))
    assert memory.attach_allocation_plan(plan) and memory.selected_reserved == [[PAGE]]
    assert not memory.attach_allocation_plan(AllocationPlan((Allocation("extra", 2 * PAGE, "workspace"),)))
    assert memory.release_selected_plan() and memory.selected_reserved == [[0]]
    onchip = AllocationEffect.acquire("workspace", "L2", PAGE, unit="bytes")
    assert memory.attach_allocation_effects((onchip,))
    assert memory.selected_reserved == [[0]] and memory.onchip_effects
    assert memory.grant_batch(0, [("hbm", 2 * PAGE)])
    assert memory.release_grants(0, ["hbm"]) == 2 * PAGE


@pytest.mark.parametrize("owners,key,storage", [
    (("reader", "reader"), None, None),
    (("reader", "other"), "physical-page", None),
    (("reader", "reader", "other"), "P", "S"),
])
def test_shared_page_retain_charges_one_page_until_its_last_reference(owners, key, storage):
    memory = budget_memory(PAGE)
    attributes = {k: v for k, v in (("physical_key", key), ("storage", storage)) if v is not None}
    for refs, owner in enumerate(owners, start=1):
        assert memory.retain_shared_pages(0, owner, (float(PAGE),), **attributes)
        assert memory.shared_pages[(0, key or owner)]["refs"] == refs
        assert memory.occupied_bytes(0, 0) == memory.shared_reserved[0][0] == PAGE
    for refs, owner in enumerate(owners):
        assert memory.release_shared_pages(0, owner)
        expected = PAGE if refs < len(owners) - 1 else 0
        assert memory.occupied_bytes(0, 0) == memory.shared_reserved[0][0] == expected
    assert not memory.shared_pages


@pytest.mark.parametrize("selected_first,attributes", [
    (True, {}), (True, {"storage": "S"}), (False, {"storage": "S"}),
])
def test_selected_and_shared_page_aliases_transfer_charge_to_survivor(selected_first, attributes):
    memory = budget_memory(PAGE)
    plan = AllocationPlan((_selected("selected", physical_key="P", **attributes),))
    if not selected_first:
        assert memory.retain_shared_pages(0, "reader", (PAGE,), physical_key="P", **attributes)
    assert memory.attach_allocation_plan(plan)
    if selected_first:
        assert memory.retain_shared_pages(0, "reader", (PAGE,), physical_key="P", **attributes)
    assert memory.occupied_bytes(0, 0) == memory.peak_combined[0][0] == PAGE
    assert memory.selected_reserved == [[PAGE]] and memory.shared_reserved == [[0]]
    version = memory.budget_version
    replacement = AllocationPlan((_selected("replacement", physical_key="Q", **attributes),))
    assert not memory.attach_allocation_plan(replacement, replace=True)
    assert memory.budget_version == version and memory.occupied_bytes(0, 0) == PAGE
    assert list(memory.selected_items[(0, 0)]) == ["selected"]
    assert memory.release_selected_allocation("selected")
    assert memory.occupied_bytes(0, 0) == PAGE
    assert memory.selected_reserved == [[0]] and memory.shared_reserved == [[PAGE]]
    assert memory.release_shared_pages(0, "reader")
    assert memory.occupied_bytes(0, 0) == 0 and memory.peak_combined[0][0] == PAGE


def test_selected_allocation_and_grant_share_one_physical_page():
    memory = budget_memory(PAGE)
    selected = _selected("selected", physical_key="P", storage="S")
    grant = dict(name="grant", bytes=PAGE, physical_key="P", storage="S", ranges=(0, PAGE))
    assert memory.attach_allocation_plan(AllocationPlan((selected,)))
    assert memory.grant_batch(0, [grant])
    assert memory.occupied_bytes(0, 0) == memory.peak_combined[0][0] == PAGE
    assert memory.selected_reserved == [[PAGE]] and memory.granted == [[0]]
    assert memory.release_grant(0, "grant") and memory.occupied_bytes(0, 0) == PAGE
    assert memory.release_selected_allocation("selected") and memory.occupied_bytes(0, 0) == 0


def test_batch_grant_is_atomic_at_exact_capacity():
    memory = budget_memory(3 * PAGE)
    assert memory.grant_batch(0, [(name, PAGE) for name in ("a", "b", "c")])
    assert memory.live_grants(0, 0) == 3 * PAGE
    for grants in ([("a", PAGE)], [("d", PAGE)]):
        assert not memory.grant_batch(0, grants)
        assert memory.live_grants(0, 0) == 3 * PAGE
    assert memory.release_grants(0, ["b"]) == PAGE and memory.live_grants(0, 0) == 2 * PAGE
    assert memory.grant_batch(0, [("d", PAGE)]) and memory.live_grants(0, 0) == 3 * PAGE
    assert not memory.grant_batch(0, [("e", PAGE), ("f", 2 * PAGE)])
    assert memory.live_grants(0, 0) == 3 * PAGE


def test_admissible_tokens_queries_capacity_without_changing_ownership():
    memory = budget_memory(1024)
    version = memory.budget_version
    for limit, granularity, expected in [(16, 1, 8), (16, 2, 8), (7, 1, 7)]:
        assert memory.admissible_tokens(0, "r", limit, granularity=granularity) == expected
    assert memory.live == [[0]] and not memory.owners and memory.budget_version == version
    assert memory.resize(0, "r", 8)


@pytest.mark.parametrize("bound", [False, True], ids=["standalone", "bound"])
def test_scheduler_cache_and_execution_grants_share_the_same_budget(bound):
    raw = case(output=1)
    raw["runtime"]["gpu_memory_utilization"] = 1.0
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = base.cache_bytes(4)[0] + 3 * PAGE
    dies = tuple(replace(die, memory_bytes=int(base.weights[0] + base.workspace[0] + capacity))
                 for die in parsed.machine.dies)
    tight = replace(parsed, machine=replace(parsed.machine, dies=dies))
    memory = Memory(tight)
    assert memory.capacity == [[capacity]]
    plan = Scheduler(tight, list(tight.workload.requests), memory).schedule(0)
    assert plan is not None
    work = linear_work(plan, tight.model, tight.runtime, tight.machine)
    execution = Execution(tight, CostModel(tight), include_submission=False, memory=memory if bound else None)
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,))
    assert step.completion.done
    assert memory.grant_batch(0, [("attention", PAGE), ("workspace", PAGE)])
    assert memory.grant_batch(0, [("extra", PAGE)]) and memory.live_grants(0, 0) == 3 * PAGE
    assert not memory.grant_batch(0, [("fourth", PAGE)]) and memory.live_grants(0, 0) == 3 * PAGE
    assert memory.hold_for_async_consumer(0, "r", "reader", storage="workspace", ranges=(PAGE,), generation=1)
    assert memory.release_grants(0, ["workspace"]) == PAGE
    assert memory.grant_batch(0, [("fourth", PAGE)]) and memory.live_grants(0, 0) == 3 * PAGE
    assert memory.release_async_consumer(0, "r", "reader")
    memory.release(0, "r")
    assert memory.live == [[0]] and memory.live_grants(0, 0) == 3 * PAGE


def test_physical_grant_batch_commits_only_when_every_budget_fits():
    grants = [_grant("K", "P", 3, storage="K", event=object()),
              _grant("V", "Q", 4, storage="V", event=object())]
    tight = budget_memory(PAGE)
    assert tight.capacity == [[PAGE]]
    version = tight.budget_version
    assert not tight.grant_batch(0, grants)
    assert tight.live == tight.peak_combined == [[0]]
    assert not tight.grant_bindings and not tight.release_events and tight.budget_version == version
    assert tight.occupied_bytes(0, 0) == tight.host_occupied_bytes(0) == 0
    roomy = budget_memory(2 * PAGE)
    assert roomy.grant_batch(0, grants)
    assert roomy.occupied_bytes(0, 0) == roomy.release_grants(0, ["K", "V"]) == 2 * PAGE
    assert roomy.occupied_bytes(0, 0) == 0
    host = budget_memory(2 * PAGE)
    host_grants = [dict(name=name, bytes=0, physical_key=name, generation=generation, host_bytes=PAGE)
                   for name, generation in [("HK", 3), ("HV", 4)]]
    host.host_dram_capacity = [2 * PAGE - 1]
    assert not host.grant_batch(0, host_grants)
    assert host.host_occupied_bytes(0) == 0 and not host.grant_bindings
    host.host_dram_capacity = [2 * PAGE]
    assert host.grant_batch(0, host_grants) and host.host_occupied_bytes(0) == 2 * PAGE
    ranks = budget_memory(PAGE, tp=2)
    assert ranks.grant_batch(0, [dict(grant, rank=rank) for rank, grant in enumerate(grants)])
    assert [ranks.occupied_bytes(0, 0, rank=rank) for rank in range(2)] == [PAGE, PAGE]


@pytest.mark.parametrize("span,generation,expected", [
    ((0, PAGE), 3, PAGE), ((2048, 6144), 3, 6144), ((0, PAGE), 4, 2 * PAGE),
])
def test_grant_holders_charge_backing_ranges_by_generation(span, generation, expected):
    memory = budget_memory(2 * PAGE)
    first, second = _grant("a", "P", 3), _grant("b", "P", generation, span=span)
    if span[0]:
        assert memory.grant_batch(0, [first]) and memory.occupied_bytes(0, 0) == PAGE
        assert memory.grant_batch(0, [second])
    else:
        assert memory.grant_batch(0, [first, second])
    assert memory.occupied_bytes(0, 0) == memory.peak_combined[0][0] == expected
    assert memory.release_grant(0, "a", generation=3) and memory.occupied_bytes(0, 0) == PAGE
    assert memory.release_grant(0, "b", generation=generation) and memory.occupied_bytes(0, 0) == 0


def test_grant_reader_last_use_controls_reuse_of_full_budget():
    memory = budget_memory(3 * PAGE)
    events = [object() for _ in range(4)]
    grants = [_grant(f"L{i}", f"P{i}", i + 3, storage=f"L{i}", event=event)
              for i, event in enumerate(events)]
    assert memory.grant_batch(0, grants[:3]) and memory.occupied_bytes(0, 0) == 3 * PAGE
    assert memory.hold_for_async_consumer(0, "L0", "reader", grant="L0", generation=3, event=events[0])
    assert memory.release_grant(0, "L0", generation=3, event=events[0])
    assert memory.occupied_bytes(0, 0) == 3 * PAGE and memory.live_grants(0, 0) == 2 * PAGE
    version = memory.budget_version
    assert not memory.grant_batch(0, grants[3:])
    assert not memory.release_grant(0, "L0", generation=4)
    assert not memory.release_async_consumer(0, "L0", "other")
    assert not memory.release_async_consumer(0, "L0", "reader", generation=4)
    assert not memory.release_async_consumer(0, "L0", "reader", event=object())
    assert memory.budget_version == version and memory.occupied_bytes(0, 0) == 3 * PAGE
    assert memory.release_async_consumer(0, "L0", "reader", generation=3, event=events[0])
    assert memory.occupied_bytes(0, 0) == 2 * PAGE
    released, = memory.drain_release_events()
    assert (released["kind"], released["released_bytes"]) == ("grant_reader", PAGE)
    assert memory.grant_batch(0, grants[3:]) and memory.occupied_bytes(0, 0) == 3 * PAGE
    assert memory.release_grants(0, ["L1", "L2", "L3"]) == 3 * PAGE
    assert memory.occupied_bytes(0, 0) == 0
    assert {row["kind"] for row in memory.drain_release_events()} == {"grant"}


def test_reader_retry_and_reused_grant_name_preserve_each_generation():
    memory = budget_memory(2 * PAGE)
    first = {"generation": 3, "event": object(), "rank": 0}
    second = {"generation": 4, "event": object(), "rank": 0}
    assert memory.grant_batch(0, [_grant("lease", "P", **first)])
    assert memory.hold_for_async_consumer(0, "lease", "reader", grant="lease", **first)
    version = memory.budget_version
    assert memory.hold_for_async_consumer(0, "lease", "reader", grant="lease", **first)
    assert not memory.hold_for_async_consumer(0, "lease", "reader", grant="lease", **second)
    assert memory.budget_version == version and len(memory.grant_readers[(0, "lease")]) == 1
    assert memory.occupied_bytes(0, 0) == PAGE
    assert memory.release_grant(0, "lease", **first) and memory.occupied_bytes(0, 0) == PAGE
    assert not memory.release_grant(0, "lease", generation=4)
    assert memory.grant_batch(0, [_grant("lease", "P", **second)]) and memory.occupied_bytes(0, 0) == 2 * PAGE
    assert not memory.hold_for_async_consumer(0, "lease", "reader", grant="lease", **second)
    assert not memory.release_async_consumer(0, "lease", "reader", **dict(first, rank=1))
    assert memory.release_async_consumer(0, "lease", "reader", **first) and memory.occupied_bytes(0, 0) == PAGE
    assert memory.hold_for_async_consumer(0, "lease", "reader", grant="lease", **second)
    assert memory.occupied_bytes(0, 0) == PAGE
    assert memory.release_async_consumer(0, "lease", "reader", **second) and memory.occupied_bytes(0, 0) == PAGE
    assert not memory.release_grant(0, "lease", **dict(second, rank=1))
    assert memory.release_grant(0, "lease", **second) and memory.occupied_bytes(0, 0) == 0


@pytest.mark.parametrize("tier,amount,host_bytes", [("l2", PAGE, 0), ("host", 0, PAGE)])
def test_non_hbm_tiers_never_charge_hbm_capacity(tier, amount, host_bytes):
    memory = budget_memory(0)
    assert memory.capacity == [[0]]
    assert memory.grant_batch(0, [dict(name="allocation", bytes=amount, tier=tier, host_bytes=host_bytes)])
    assert memory.occupied_bytes(0, 0) == memory.live_grants(0, 0) == 0
    assert memory.host_occupied_bytes(0) == host_bytes
    assert memory.grant_bindings[(0, "allocation")]["tier"] == tier
    assert memory.release_grant(0, "allocation")
    assert memory.occupied_bytes(0, 0) == memory.host_occupied_bytes(0) == 0
    version = memory.budget_version
    assert not memory.grant_batch(0, [dict(name="negative", bytes=PAGE, ranges=(-1, PAGE))])
    assert memory.budget_version == version and not memory.grant_bindings
    assert not memory.grant_batch(0, [dict(name="hbm", bytes=PAGE)])
    assert memory.occupied_bytes(0, 0) == 0


def test_dspark_hbm_admission_reserves_verified_recurrent_state_slots():
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.scheduler.progress import DraftPolicy
    from mindie_sim.schema.dspark import DsparkSpec
    from mindie_sim.schema.model import LayerPlan
    from _support.axes import REPRESENTATIVE_SKUS

    raw = compose_case(
        "qwen3-32b", hardware=REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1,
        prompt_tokens=64, output_tokens=16,
    )
    parsed = parse_case(raw)
    gdn_layer = LayerPlan(
        index=0, attention="gdn", ffn="dense", num_heads=8, num_kv_heads=8,
        head_dim=128, linear_heads=8, linear_head_dim=128,
        linear_key_heads=8, linear_key_head_dim=128, linear_value_heads=8,
        intermediate_size=2048, recurrent_state_elements=8 * 128 * 128,
        recurrent_state_dtype="fp32", conv_state_elements=1024 * 4,
        conv_width=1024, conv_kernel=4, conv_state_dtype="bf16",
    )
    draft = DraftSpec(
        kind="dspark", layers=1, hidden_size=1024, intermediate_size=2048,
        num_heads=8, num_kv_heads=2, head_dim=128, q_width=1024, kv_width=256,
        attention_kind="gqa", block_size=3, markov_rank=16, vocab_size=1024,
    )
    model = replace(parsed.model, layers=1, layer_plans=(gdn_layer,),
                    auxiliary=AuxiliarySpec(dspark=draft), metadata=())
    rt_base = replace(parsed.runtime, dspark=None, metadata=(), recover_ssm=False)
    rt_dspark = replace(parsed.runtime, dspark=DsparkSpec(3, (0.8, 0.6, 0.4)),
                        metadata=(), recover_ssm=False)
    case_base = replace(parsed, model=model, runtime=rt_base)
    case_dspark = replace(parsed, model=model, runtime=rt_dspark)

    policy = DraftPolicy.for_request(case_dspark, case_dspark.workload.requests[0])
    assert policy.proposal == 3
    mem_base, mem_dspark = Memory(case_base), Memory(case_dspark)
    bd_base = mem_base.cache_pool_breakdown(128, padded=True)
    bd_dspark = mem_dspark.cache_pool_breakdown(128, padded=True)
    assert mem_dspark._effective_cache_query_tokens() == 1 + policy.proposal == 4
    assert bd_dspark["query_tokens"] == 4
    # Recurrent state must reserve 4 FP32 checkpoint slots instead of 1 (plus 3 conv rollback slices).
    single_state = 8 * 128 * 128 * 4
    assert bd_dspark["by_component"]["recurrent_state"] == pytest.approx(4 * single_state)
    assert bd_base["by_component"]["recurrent_state"] == pytest.approx(single_state)

    # Confidence threshold declining all proposals drops verify width back to 1.
    rt_declined = replace(
        parsed.runtime,
        dspark=DsparkSpec(3, (0.8, 0.6, 0.4), confidence=(0.1, 0.9, 0.9), confidence_threshold=0.5),
        metadata=(), recover_ssm=False,
    )
    mem_declined = Memory(replace(parsed, model=model, runtime=rt_declined))
    assert mem_declined._effective_cache_query_tokens() == 1
    assert mem_declined.cache_pool_breakdown(128, padded=True)["by_component"]["recurrent_state"] == pytest.approx(single_state)


def test_replicated_draft_kv_ignores_target_rank_token_owners():
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.capacity import MemoryPolicy
    from mindie_sim.schema.dspark import DsparkSpec
    from _support.axes import REPRESENTATIVE_SKUS

    raw = compose_case(
        "deepseek-v3.2", hardware=REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, ep=2, dcp=2,
        layers=4, prompt_tokens=8192, output_tokens=16, max_model_len=16384,
    )
    parsed = parse_case(raw)
    replicated_draft = DraftSpec(
        kind="dspark", layers=2, hidden_size=1024, intermediate_size=2048,
        num_heads=8, num_kv_heads=2, head_dim=64, q_width=512, kv_width=128,
        attention_kind="gqa", block_size=3, markov_rank=16, vocab_size=1024,
        dcp_sharded=False, max_position_embeddings=4096,
    )
    model_rep = replace(parsed.model, auxiliary=AuxiliarySpec(dspark=replicated_draft))
    rt = replace(parsed.runtime, dspark=DsparkSpec(3, (0.8, 0.6, 0.4)),
                 metadata=(("draft_mode", "dspark"), ("draft_max_model_len", 4096)))
    owners = ((0, 2048), (2048, 4096))
    mem_rep = Memory(replace(parsed, model=model_rep, runtime=rt),
                     policy=MemoryPolicy(rank_token_owners=owners))
    # TP=2, draft num_kv_heads=2 -> 1 local KV head -> 2 * 1 * 64 * 2B = 256 B/row * 2 layers = 512 B/tok.
    expected_draft_bytes = 4096 * 2 * 1 * 64 * 2 * 2
    for rank in (0, 1):
        draft_b, is_sharded = mem_rep._non_inherited_draft_cache_bytes(
            model_rep, 8192, padded=True, rank=rank,
        )
        assert is_sharded is False
        assert draft_b == pytest.approx(expected_draft_bytes)
    rank_views = mem_rep.rank_cache_allocation(8192)
    assert len(rank_views) == 2
    for view in rank_views:
        assert view["components"][0]["allocated"]["draft_kv"]["total"] == pytest.approx(expected_draft_bytes)

    # Sharded MLA draft (dcp_sharded=True) continues to respect target rank_token_owners.
    sharded_draft = DraftSpec(
        kind="dspark", layers=2, hidden_size=1024, intermediate_size=2048,
        num_heads=8, num_kv_heads=8, head_dim=64, q_width=512, kv_width=576,
        attention_kind="mla", q_lora_rank=128, kv_lora_rank=512,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64,
        kv_cache_elements=576, block_size=3, markov_rank=16, vocab_size=1024,
        dcp_sharded=True, max_position_embeddings=4096,
    )
    model_shd = replace(parsed.model, auxiliary=AuxiliarySpec(dspark=sharded_draft))
    mem_shd = Memory(replace(parsed, model=model_shd, runtime=rt),
                     policy=MemoryPolicy(rank_token_owners=owners))
    draft_b0, is_sharded0 = mem_shd._non_inherited_draft_cache_bytes(model_shd, 8192, padded=True, rank=0)
    draft_b1, is_sharded1 = mem_shd._non_inherited_draft_cache_bytes(model_shd, 8192, padded=True, rank=1)
    assert is_sharded0 is True and is_sharded1 is True
    assert draft_b0 == pytest.approx(draft_b1)
    assert draft_b0 == pytest.approx(2048 * 576 * 2 * 2)


def test_ledger_cross_engine_isolation_pending_release_and_unknown_domain_ceiling():
    """Verify runtime_uid event isolation, pending last_use preservation, and strict domain HBM ceilings."""
    state = LedgerState(uid="rt-A")
    pool = _pool(state, "memory", 2 * PAGE)
    ev_a_acq = Completion(sequence=1, done=True, time=0.0, runtime_uid="rt-A")
    ev_a_use = Completion(sequence=2, done=False, time=None, runtime_uid="rt-A")
    ev_b_seq2 = Completion(sequence=2, done=True, time=1.0, runtime_uid="rt-B")
    ev_other_done = Completion(sequence=9, done=True, time=0.5, runtime_uid="rt-A")

    def _rt_key(idx: int) -> AllocationKey:
        return AllocationKey(
            occurrence=OccurrenceKey(runtime_uid="rt-A", launch=1, ordinal=idx),
            scope_kind="invocation", scope_ordinal=idx, symbol=idx, generation=0,
        )

    alloc_key = _rt_key(1)
    binding = AllocationBinding(key=alloc_key, pool=pool, amount=float(PAGE), last_use=ev_a_use)
    res_acq = Ledger.try_acquire(state, (binding,), holder="h0", at=ev_a_acq)
    assert res_acq.granted is True and Ledger.usage(state)["memory"][pool] == pytest.approx(PAGE)

    # release(after=None) and release(after=done_other) MUST preserve pending last_use (ev_a_use).
    rel_none = Ledger.release(state, alloc_key, after=None)
    assert rel_none.released is False and rel_none.deferred is True
    rel_other = Ledger.release(state, alloc_key, after=ev_other_done)
    assert rel_other.released is False and rel_other.deferred is True
    assert Ledger.usage(state)["memory"][pool] == pytest.approx(PAGE)

    # Cross-Engine seq=2 from rt-B MUST NOT settle rt-A's pending seq=2 event.
    assert Ledger.settle(state, (ev_b_seq2,)) == ()
    assert Ledger.usage(state)["memory"][pool] == pytest.approx(PAGE)

    # Settling rt-A's own seq=2 event completes the deferred release.
    ev_a_use_done = Completion(sequence=2, done=True, time=2.0, runtime_uid="rt-A")
    settled = Ledger.settle(state, (ev_a_use_done,))
    assert len(settled) == 1 and settled[0].released is True
    assert Ledger.usage(state)["memory"][pool] == pytest.approx(0.0)

    # Unknown domain MUST fail closed even when 1 domain exists, and registered HBM ceiling MUST block overflow.
    unknown_pool = PoolKey(physical_domain="unregistered-hbm", rank=0, tier="hbm", name="hbm")
    with pytest.raises(LedgerError, match="capacity_domain_missing"):
        Ledger.try_acquire(state, (AllocationBinding(key=_rt_key(2), pool=unknown_pool, amount=float(PAGE)),), holder="h1", at=ev_a_acq)
    over = Ledger.try_acquire(state, (AllocationBinding(key=_rt_key(3), pool=pool, amount=float(3 * PAGE)),), holder="h2", at=ev_a_acq)
    assert over.granted is False and over.reason == "capacity_exceeded" and pool in over.blocked_pools
