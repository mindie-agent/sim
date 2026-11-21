"""Device HCCL expansion owners, physical service, and native call boundaries.

[Role] Exercise AICPU, AIV and CCU expansion templates through their plans and
  ordinary callers, including the observed AllGather binding.
[Invariants] Scratch is held for its declared lifetime. Every copy, reduction,
  flag, and notification contributes only its actual demand and owner scope.
[Timeouts & Locks] Local JSON fixtures and bounded in-memory execution.
[Side-Effect Policy] Read-only fixture data and isolated simulation state.
"""
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pytest
from _support.collective_cases import case, declared, isolated_calls
from _support.kernel_invocation import op as make_op

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction import hccl_aiv, hccl_aiv_gather, hccl_ccu, hccl_ccu_two_die
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_aicpu import build as build_aicpu
from mindie_sim.prediction.hccl_aicpu import contract as aicpu_contract
from mindie_sim.prediction.hccl_aiv import UB_BATCH, UB_RESIDENCY
from mindie_sim.prediction.hccl_aiv_gather import DATA_LIMIT, FLAG_BYTES
from mindie_sim.prediction.hccl_ccu_two_die import SERVICES, TEMPLATE
from mindie_sim.prediction.hccl_contracts import legacy_mesh_candidate, select
from mindie_sim.prediction.hccl_graph import lower
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.native_binding import bind_native_call, validate_invocation
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, RecipeBook, RecipeContext, Stage,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_op_structure

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "examples/source-families/hccl-comrun5a-allgather.json"
CALLER_CASES = ROOT / "examples/cases/high-concurrency-20261030/kimi-hccl-source-conditional"
CALLER_FILENAME = "a5_kimi_tp8_e16_bf16_c32_{}_hccl_observed.json"
DUAL_DIE_SKU = "Ascend910_9362"


def _assert_released(summary):
    assert summary["live_tasks"] == summary["live_programs"] == 0
    assert not any(summary["live_allocated"].values())


def _work(plan, resource):
    return sum(dict(phase.work).get(resource, 0) for phase in plan.phases)


def _phase_op(selection, name):
    return OpSpec(name, "collective", "decode", 0, 0, (), (), (), (), (), metadata=(
        ("hccl_phase", {"selection": selection.to_dict(), "phase": name, "rank": 0}),
    ))


def _run_phases(selection, contract, build, rates, units, *, only=None, ub=False, setup=0, floor=0):
    buffer_args = {"buffers": {"ub": 256 * 1024}, "buffer_units": {"ub": 64}} if ub else {}
    geometry = Geometry(32 if ub else 1, units, **buffer_args)
    phases = [p for p in contract(selection).phases if only is None or p.name == only]
    plans = {p.name: build(_phase_op(selection, p.name), geometry) for p in phases}
    names = dict(zip(rates, rates))
    ctx = RecipeContext(None, {}, names, rates, geometry.compute_units, resource_units=units,
                        pool_names={"ub": "ub"} if ub else {}, pool_units={"ub": 64} if ub else {},
                        setup_ms=setup, minimum_ms=floor)
    wire_bytes = {p.name: sum(p.exchange[0]) for p in phases if p.exchange}
    work = sum((p.work for p in plans.values()), Work()) + Work(resources={"link": sum(wire_bytes.values())})

    def program(context):
        stages = []
        for phase in phases:
            local = plans[phase.name].program(context)
            if phase.name in wire_bytes:
                wire = KernelProgram((Stage("wire", {"link": wire_bytes[phase.name]}),))
                local = ParallelProgram((local, wire))
            stages.append(Stage(phase.name, after=() if only else phase.after, program=local))
        return KernelProgram(tuple(stages))

    pool_args = {"pools": {"ub": 256 * 1024}, "pool_units": {"ub": 64}} if ub else {}
    engine = Engine([Resource(k, rate, units=units[k]) for k, rate in rates.items()],
                    trace_limit=3000, **pool_args)
    ready = engine.submit("input", release=7)
    done = OperatorPlan(work, program).lower(ctx).start(engine, "call", ready)
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.serviced == pytest.approx({k: work.resources.get(k, 0) for k in rates})
    _assert_released(engine.summary())
    return done.time - 7, engine, plans


def _aicpu_selection(collective="allreduce", template="mesh_oneshot", ranks=4, size=1024):
    return select({"expansion": "AICPU_TS", "template": template},
                  collective=collective, ranks=ranks, payload_bytes=size)


@pytest.mark.parametrize("ranks", [2, 3, 8])
def test_oneshot_overlaps_peer_and_local_copies_then_reduces(ranks):
    size = 1024
    selection = _aicpu_selection(ranks=ranks, size=size)
    for rank in range(ranks):
        plan = aicpu_contract(selection, rank)
        phases = {p.name: p for p in plan.phases}
        assert phases["input_to_scratch"].after == phases["allwrite"].after == ("pre_sync",)
        assert set(phases["post_sync"].after) == {"input_to_scratch", "allwrite"}
        assert phases["allwrite"].exchange[rank] == tuple(size if peer != rank else 0 for peer in range(ranks))
        assert _work(plan, "hbm") == (4 + 5 * (ranks - 1)) * size
        assert _work(plan, "hccl_reduce") == (ranks - 1) * size
        assert len(phases["local_reduce"].steps) == ranks - 1
        assert plan.scratch_bytes == ranks * size


@pytest.mark.parametrize("count,ranks", ((17, 4), (2, 8), (128, 3), (1024, 8)))
def test_twoshot_integer_tail_and_empty_rank_oracle(count, ranks):
    selection = _aicpu_selection(template='mesh_twoshot', ranks=ranks, size=2*count)
    chunk = (count+ranks-1)//ranks
    widths = [max(0, min(chunk, count-rank*chunk))*2 for rank in range(ranks)]
    for rank in range(ranks):
        plan = aicpu_contract(selection, rank)
        phases = {p.name: p for p in plan.phases}
        scatter, gather = phases['scatter'].exchange, phases['gather'].exchange
        assert scatter == tuple(zip(*gather))
        assert sum(scatter[rank]) == 2*count-widths[rank]
        assert sum(row[rank] for row in scatter) == (ranks-1)*widths[rank]
        assert _work(plan, 'hccl_copy') == 2*widths[rank]
        assert _work(plan, 'hccl_reduce') == (ranks-1)*widths[rank]
        assert phases['ag_pre_sync'].after == ('local_reduce',)
    assert sum(map(sum, scatter)) == (ranks - 1) * 2 * count


@pytest.mark.parametrize("collective,template,mode,saved", [
    ("allreduce", "mesh_twoshot", "symmetric", 8),
    ("reduce_scatter", "mesh", "symmetric", 6),
    ("allgather", "mesh", "remote_output", 2),
])
def test_direct_buffer_removes_copy_work_without_changing_send_matrix(collective, template, mode, saved):
    size = 1024
    selection = _aicpu_selection(collective, template, 4, 4 * size if mode == "symmetric" else size)
    separate, direct = aicpu_contract(selection), aicpu_contract(replace(selection, buffer_mode=mode))
    assert [p.exchange for p in separate.phases if p.exchange] == [p.exchange for p in direct.phases if p.exchange]
    assert direct.scratch_bytes == 0
    assert _work(separate, "hbm") - _work(direct, "hbm") == saved * size
    assert _work(separate, "hccl_reduce") == _work(direct, "hccl_reduce")


@pytest.mark.parametrize("priced", [True, False], ids=["declared-service", "memory-bound"])
def test_reduction_steps_serialize_and_conserve_memory(priced):
    rates = {"hbm": 1024}
    if priced:
        rates.update(hccl_copy=1024, hccl_reduce=256, hccl_control=10)
    elapsed, engine, plans = _run_phases(_aicpu_selection(), aicpu_contract, build_aicpu,
                                         rates, dict.fromkeys(rates, 1), only="local_reduce")
    assert elapsed == (12 if priced else 9)
    assert engine.serviced["hbm"] == 9 * 1024
    unavailable = {} if priced else {"hccl_reduce": 3 * 1024}
    assert plans["local_reduce"].metadata["unpriced_service_work"] == unavailable
    reductions = [row for row in engine.trace if ":reduce_" in row["task"] and row["demand"]]
    assert len(reductions) == 3
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(reductions, reductions[1:]))

@pytest.mark.parametrize("expansion,size,expected", [
    ("AIV", 128*1024 - 2, "mesh_oneshot"), ("AIV", 128*1024, "mesh_twoshot"),
    ("AICPU_TS", 8*1024**2, "mesh_oneshot"), ("AICPU_TS", 16*1024**2, "mesh_twoshot"),
    ("AICPU_TS", 16*1024**2 + 2, "mesh_chunk_twoshot"),
])
def test_mesh_selection_respects_explicit_template_and_size_thresholds(expansion, size, expected):
    args = dict(collective="allreduce", ranks=2, payload_bytes=size)
    assert legacy_mesh_candidate(**args, expansion=expansion) == expected
    assert select({"expansion": "AIV", "template": "mesh_oneshot", "block_limit": 3},
                  **args).template == "mesh_oneshot"



def _gather_selection(size, ranks=4, blocks=64):
    return select({'expansion': 'AIV', 'template': 'mesh', 'block_limit': blocks, 'dtype_bytes': 2},
                  collective='allgather',ranks=ranks,payload_bytes=size)


@pytest.mark.parametrize('ranks,blocks,size',[(4,64,2),(4,64,30),(4,64,4096),(8,3,42),
    (4,64,DATA_LIMIT),(3,64,DATA_LIMIT+6),(8,64,2*DATA_LIMIT),(64,64,DATA_LIMIT+2),
    (2,4,2*DATA_LIMIT+6)])
def test_independent_owner_and_flag_byte_equations(ranks,blocks,size):
    s=_gather_selection(size,ranks,blocks)
    owner_jobs, _, detail = hccl_aiv_gather.jobs(s)
    assert sum(a.size for actions in owner_jobs['precopy'].values() for a in actions if a.kind=='copy')==size
    assert sum(a.size for actions in owner_jobs['gather'].values() for a in actions if a.kind=='copy')==ranks*size
    if size>=DATA_LIMIT and blocks>=2*ranks:
        usable=blocks//ranks*ranks
        assert set(owner_jobs['precopy'])==set(range(ranks))
        assert set(owner_jobs['gather'])==set(range(ranks,usable))
        cut=(usable-ranks)//ranks
        q,tail=divmod(size//2,ranks)
        pieces=sum(min(q+(i<tail),cut) for i in range(ranks))
        records,waits=pieces,ranks*pieces
    else:
        assert set(owner_jobs['gather'])==set(range(min(blocks,ranks)))
        records=waits=2*ranks
    if not detail['pingpong']:
        records += 2 * ranks
        waits += ranks
    p=hccl_aiv_gather.contract(s)
    totals = {k: _work(p, k) for k in ('hbm', 'mte2_aiv', 'mte3')}
    assert totals=={'hbm':2*(ranks+1)*size+(records+waits)*128,
                   'mte2_aiv':(ranks+1)*size+waits*128,
                   'mte3':(ranks+1)*size+records*128}
    assert sum(dict(p.control_counts)[k] for k in ('wait','record'))==records+waits
    flags = detail["positive_input_fragments"] if detail["branch"] == "producer_consumer" else 2
    flags += int(not detail["pingpong"])
    assert sum(sum(phase.exchange[0]) for phase in p.phases if phase.exchange) == (
        (ranks - 1) * (size + flags * FLAG_BYTES))
    _run_gather(s)


def _run_gather(selection, *, setup=0, floor=0):
    units = {"hbm": 1, "mte2_aiv": 64, "mte3": 64, "vector": 64, "link": 1}
    return _run_phases(selection, hccl_aiv_gather.contract, hccl_aiv_gather.build,
                       dict.fromkeys(units, 1e9), units, ub=True, setup=setup, floor=floor)


def test_gather_copy_windows_follow_flag_order_and_one_launch_boundary():
    selection = _gather_selection(DATA_LIMIT + 6, ranks=3)
    base, engine, _ = _run_gather(selection)
    assert _run_gather(selection, setup=2)[0] == pytest.approx(base + 2)
    assert _run_gather(selection, setup=2, floor=base + 5)[0] == pytest.approx(base + 5)
    work = [row for row in engine.trace if ":precopy:" in row["task"] and ":owner0:" in row["task"]
            and row["task"].endswith((":read", ":write", ":record"))]
    assert work
    assert all(b["start_ms"] >= a["end_ms"] - 1e-12 for a, b in zip(work, work[1:]))


def _run_ccu(selection, rates=None):
    rates = rates or dict(hbm=1024, ccu_copy=1024, ccu_buffer_bytes=4096,
                          ccu_reduce_elements=512, ccu_control=1000, link=1024)
    return _run_phases(selection, hccl_ccu.contract, hccl_ccu.build, rates, dict.fromkeys(rates, 1))


def _ccu_selection(kind='reduce',ranks=4,size=1024,version=2):
    return select({'expansion':'CCU_MS' if kind=='reduce' else 'CCU_SCHED',
                   'template':f'mesh_oneshot_v{version}' if kind=='reduce' else f'mesh_mem2mem_v{version}',
                   'dtype_bytes':2},collective='allreduce' if kind=='reduce' else 'allgather',
                  ranks=ranks,payload_bytes=size)


@pytest.mark.parametrize('version',(1,2))
@pytest.mark.parametrize('ranks',(2,4,8))
def test_reduce_direct_ccu_arrivals_never_become_hbm_scratch_writes(version,ranks):
    n=384
    source=hccl_ccu.contract(_ccu_selection(ranks=ranks,size=n,version=version))
    phases={p.name:p for p in source.phases}
    expected = {'hbm': (ranks + 1) * n, 'ccu_copy': 2 * n,
                'ccu_reduce_elements': (ranks - 1) * n // 2, 'ccu_buffer_bytes': (2 * ranks + 2) * n}
    assert all(_work(source, role) == amount for role, amount in expected.items())
    assert set(phases['load_wait'].after)=={'local_input_to_ccu','remote_input_to_ccu'}
    assert not dict(phases['local_reduce'].work).get('hbm', 0)
    assert sum(map(sum,phases['remote_input_to_ccu'].exchange))==ranks*(ranks-1)*n
    _run_ccu(source.selection)


@pytest.mark.parametrize('size',(2,32768,32770,8*32768,9*32768+6))
def test_gather_equal_payload_and_copy_slots_keep_all_bytes_once(size):
    s=_ccu_selection('gather',size=size)
    source=hccl_ccu.contract(s)
    assert _work(source,'hbm')==2*s.ranks*size
    assert _work(source, 'ccu_copy') == _work(source, 'ccu_buffer_bytes') == 2 * size
    assert set(next(p.after for p in source.phases if p.name == 'write_wait')) == {'peer_write', 'local_copy'}
    _, engine, _ = _run_ccu(s)
    copies=[r for r in engine.trace if r['task'].endswith(':read') and ':local_copy:' in r['task']]
    assert len(copies)==(size+32767)//32768
    if size > 8 * 32768:
        assert min(r['start_ms'] for r in engine.trace if ':tail:' in r['task']) >= max(
            r['end_ms'] for r in engine.trace if ':full_groups:' in r['task'])
    assert hccl_ccu.buffer_geometry(s)['copy_slots']==8


def test_ccu_unpriced_work_preserves_memory_bound():
    elapsed, _, plans = _run_ccu(_ccu_selection(), rates={"hbm": 1024, "link": 1024})
    assert elapsed == pytest.approx(5)
    omitted = Counter()
    for plan in plans.values():
        omitted.update(plan.metadata["unpriced_service_work"])
    assert omitted == {"ccu_copy": 2048, "ccu_reduce_elements": 1536,
                       "ccu_buffer_bytes": 10240, "ccu_control": 18}


def test_ccu_templates_reject_incompatible_modes_and_domains():
    s=_ccu_selection()
    for incompatible in (replace(s,expansion='CCU_SCHED'),replace(s,topology='TWO_DIE_REGULAR'),
                         replace(s,payload_bytes=4098),replace(s,ranks=16)):
        with pytest.raises(ModelingError):
            hccl_ccu.contract(incompatible)
    ag=_ccu_selection('gather')
    with pytest.raises(ModelingError):
        hccl_ccu.contract(replace(ag, expansion='CCU_MS'))
    assert hccl_ccu.contract(replace(s,template='mesh_oneshot_v1')).phases==hccl_ccu.contract(s).phases


def channel_map(ranks):
    return [[None if src == dst else int(((dst - src) % ranks) >= ranks // 2)
             for dst in range(ranks)] for src in range(ranks)]


def _endpoint_declaration(ranks):
    return dict(expansion="CCU_MS", template=TEMPLATE, topology="TWO_DIE_REGULAR",
                channel_die_map=channel_map(ranks), signal_bytes=8)


def _endpoint_case(ranks=8, size=256):
    raw = case(tp=ranks)
    raw['model'].update(hidden_size=128, num_heads=16, num_kv_heads=16,
                        intermediate_size=256, vocab_size=size * ranks // 2)
    raw['machine']['device'].update(
        rates={'memory': 1e12, 'vector': 1e12,
               **{role: 1e12 for role in SERVICES}, 'hccl_control': 1e12},
        resource_units={role: 2 for role in SERVICES})
    raw['machine']['links'][0].update(capacity_bytes_per_ms=1e9)
    raw['runtime']['kernels'] = {'hccl': {'allgather': _endpoint_declaration(ranks)}}
    return raw


def _endpoint_ops(raw):
    ops = target_graphs(raw, phase="decode", tokens=1, context=16, scope="model")[0]
    selected = [op for op in ops if "hccl_source_group" in dict(op.metadata)]
    ids = {op.op_id for op in selected}
    return tuple(replace(op, deps=tuple(dep for dep in op.deps if dep in ids)) for op in selected)


def _totals(plan):
    result = Counter()
    for phase in plan.phases:
        result.update(dict(phase.work))
    return result


@pytest.mark.parametrize('ranks', [4, 8, 16])
@pytest.mark.parametrize('size', [2, 384, 4096])
def test_endpoint_loads_and_self_output_conserve_ccu_work(ranks, size):
    selection = select(_endpoint_declaration(ranks), collective='allgather', ranks=ranks, payload_bytes=size)
    plan = hccl_ccu_two_die.contract(selection)
    work = _totals(plan)
    assert work == {'hbm': (ranks + 2) * size, 'ccu_buffer_bytes': (ranks + 2) * size,
                    'ccu_copy': 3 * size, 'ccu_control': 5 * (ranks - 1) + 4, 'hccl_control': 4}
    assert sum(sum(map(sum, p.exchange)) for p in plan.phases if p.exchange) == ranks * (ranks - 1) * size
    assert sum(len(p.signals) for p in plan.phases) == 3 * ranks * (ranks - 1) and plan.scratch_bytes == 0


@pytest.mark.parametrize('ranks', [8, 16])
def test_public_model_caller_frames_payload_and_three_remote_notifications_once(ranks):
    size = 30
    raw = _endpoint_case(ranks, size)
    raw['machine']['links'][0].update(transaction_bytes=16, protocol_overhead_bytes=4)
    ops = _endpoint_ops(raw)
    result = graph_costs(raw, ops, input_ready_ms=7, trace_limit=10000, isolated=False)
    serviced = result['execution']['serviced']
    assert serviced['link:fabric'] == pytest.approx(ranks * (ranks - 1) * (ceil(size / 16) * 20 + 3 * 20))
    assert serviced['memory@r0:s0'] == pytest.approx((ranks + 2) * size)
    resource = 'ccu_copy@r0:s0'
    owned = Counter()
    for event in result['events']:
        if amount := event['demand'].get(resource):
            owned[event['placement'][resource][0]] += amount
    assert owned == {0: 2 * size, 1: size}
    rank_order = list(range(2 * ranks - 1, 0, -2))
    raw["machine"]["devices"] = [{"rank": rank} for rank in rank_order]
    raw["machine"]["links"][0]["members"] = rank_order
    remapped = graph_costs(raw, _endpoint_ops(raw), input_ready_ms=7, isolated=False)
    assert remapped["execution"]["serviced"] == pytest.approx(serviced)
    assert remapped["duration_ms"] == pytest.approx(result["duration_ms"])
    assert predict(raw)["status"] == "modeled"
    raw["machine"]["device"]["resource_units"]["ccu_copy"] = 1
    with pytest.raises(ModelingError, match="two_explicit_units"):
        graph_costs(raw, ops, isolated=False)


def test_ccu_endpoint_map_requires_homogeneous_degree():
    selection = select(_endpoint_declaration(8), collective='allgather', ranks=8, payload_bytes=256)
    mapping = channel_map(selection.ranks)
    mapping[0] = [None if value is None else 1 - value for value in mapping[0]]
    with pytest.raises(ModelingError, match="homogeneous_endpoint_degrees"):
        hccl_ccu_two_die.channels(replace(selection, channel_die_map=tuple(map(tuple, mapping))))
    for invalid in (replace(selection, expansion="CCU_SCHED"), replace(selection, payload_bytes=4098)):
        with pytest.raises(ModelingError):
            hccl_ccu_two_die.contract(invalid)


def _physical_caller(ranks, collective, template):
    raw = case(tp=ranks)
    raw["machine"] = {"backend": "ascend", "sku": DUAL_DIE_SKU, "num_devices": ranks,
                      "ranks_per_node": ranks // 2, "dies_per_package": 2,
                      "device": {"buffers": {"ub": 256 * 1024}}}
    raw["model"].update(layers=1, hidden_size=1024, num_heads=32, num_kv_heads=32,
                        head_dim=32, intermediate_size=2048)
    ops = target_graphs(raw, scope="body", phase="decode", tokens=32, context=128)[0]
    original = next(op for op in ops if op.family == "allreduce")
    payload = round(original.link_bytes / (2 * (ranks - 1) / ranks))
    factor = ranks - 1 if collective == "allgather" else 2 * (ranks - 1) / ranks
    options = {"expansion": "AIV", "template": template, "topology_levels": 2, "block_limit": 40}
    raw["runtime"]["kernels"] = {"hccl": {collective: options}}
    original = replace(original, family=collective, recipe=collective, deps=(),
                       link_bytes=payload * factor,
                       metadata=tuple({**dict(original.metadata), "collective": collective}.items()))
    parsed = parse_case(raw)
    return raw, lower((original,), parsed.runtime, parsed.model.activation_dtype), payload


@pytest.mark.parametrize('ranks', (8, 16))
@pytest.mark.parametrize('collective,template,channel', (
    ('allreduce', 'mesh_oneshot', 'd2d_intra_package'), ('allreduce', 'mesh_twoshot', 'd2d_intra_node'),
    ('allgather', 'mesh', 'd2d_inter_node')))
def test_aiv_physical_levels_keep_full_peer_algorithm_and_wire_work(ranks, collective, template, channel):
    raw, graph, payload = _physical_caller(ranks, collective, template)
    options = raw["runtime"]["kernels"]["hccl"][collective]
    selection = select(options, collective=collective, ranks=ranks, payload_bytes=payload)
    module = hccl_aiv_gather if collective == 'allgather' else hccl_aiv
    plan = module.contract(selection, 40)
    assert plan.phases == module.contract(replace(selection, topology_levels=1), 40).phases
    expected_wire = sum(sum(map(sum, phase.exchange)) for phase in plan.phases)
    report = graph_costs(raw, graph, input_ready_ms=7, trace_limit=10000)
    serviced = report['execution']['serviced']
    wire = {key: value for key, value in serviced.items() if key.startswith('link:') and value}
    for direction in (':tx:', ':rx:'):
        assert sum(value for key, value in wire.items() if direction in key) == pytest.approx(expected_wire)
    channels = ('d2d_intra_package', 'd2d_intra_node', 'd2d_inter_node')
    assert all(any(channel in key for key in wire) for channel in channels)
    _assert_released(report['execution'])
    if ranks == 8:
        raw["machine"]["transfer_channels"] = {channel: {"effective_uni_bw_gbps": 0.5}}
        slower = graph_costs(raw, graph, input_ready_ms=7)
        assert slower["duration_ms"] > report["duration_ms"]
        assert slower["execution"]["serviced"] == pytest.approx(serviced)


def test_repeated_aiv_batches_preserve_copy_and_atomic_tails():
    batches, tail, ranks, units = 1_000_000, 32, 4, 64
    payload = batches * UB_BATCH + tail
    selected = select({"expansion": "AIV", "template": "mesh_oneshot", "block_limit": units},
                      collective="allreduce", ranks=ranks, payload_bytes=payload)
    op = _phase_op(selected, "reduce_output")
    roles = {"hbm": 1, "mte2_aiv": units, "mte3": units, "vector": units}
    plan = hccl_aiv.build(op, Geometry(units, roles, buffers={"ub": 256 * 1024}))
    context = RecipeContext(op, {}, dict(zip(roles, roles)), dict.fromkeys(roles, 1000), units,
                            resource_units=roles, pool_names={"ub": "ub"})
    owner, = plan.program(context).stages
    pipeline = owner.program
    span = batches + 1
    assert pipeline.count == ranks * span and pipeline.slots == 1
    assert owner.acquire == owner.free and owner.acquire[0].units == (ranks,)
    assert owner.acquire[0].size == UB_RESIDENCY
    assert pipeline.tile(batches // 2) is pipeline.tile(0)
    assert pipeline.tile(pipeline.count - 1) is pipeline.tile(2 * span - 1)
    for index, size, factor in ((0, UB_BATCH, 1), (batches, tail, 1),
                                (span, UB_BATCH, 2), (2 * span - 1, tail, 2)):
        read, write = pipeline.tile(index)
        assert read.work == {"hbm": size, "mte2_aiv": size}
        assert write.work == {"hbm": factor * size, "mte3": size}
        assert write.after == ("read",)
        assert read.placement["mte2_aiv"] == write.placement["mte3"] == (ranks,)
    assert plan.work.resources == {"hbm": payload * (2 + 3 * (ranks - 1)),
                                   "mte2_aiv": ranks * payload, "mte3": ranks * payload}


def first_call(raw):
    ops = isolated_calls(raw)[0]
    group = dict(ops[0].metadata)['hccl_source_group']
    return tuple(op for op in ops if dict(op.metadata)['hccl_source_group'] == group)


def competing_owner(operation, geometry):
    work = Work(resources={'vector': 15625})

    def program(context):
        allocation = context.allocate('competing', 'ub', 100*1024, (2,))
        return KernelProgram((Stage('service', context.bind(work),
            placement=context.placement('vector', (2,)),
            acquire=(allocation,), free=(allocation,)),))

    return OperatorPlan(work,program)


def test_ordinary_oneshot_reducer_buffer_is_unavailable_while_producers_are_running():
    raw = declared()
    raw['model']['hidden_size'] = 1680
    operations = first_call(raw)
    books = RecipeBook.from_mapping(raw['runtime']['kernels'])
    books.recipes['competitor'] = PlannedRecipe(competing_owner)
    baseline = graph_costs(raw, operations, recipes=books, isolated=False)
    external = replace(make_op('competing', deps=('arrival',), stream='other'), recipe='competitor')
    report = graph_costs(raw,(*operations,external),recipes=books,
                         external_inputs={'arrival': .1}, isolated=False,trace_limit=1000)
    service = next(row for row in report['events']
                   if 'competing:' in row['task'] and row['task'].endswith(':service'))
    assert service['start_ms'] == pytest.approx(baseline['duration_ms'])
    assert service['end_ms'] == pytest.approx(baseline['duration_ms']+1)
    actual = report['execution']['serviced']
    assert all(actual[key] == pytest.approx(value) for key, value in baseline['execution']['serviced'].items()
               if not key.startswith('vector@'))
    _assert_released(report['execution'])


@pytest.mark.parametrize('hidden_size', (32, 1680))
def test_twoshot_whole_call_and_isolated_phase_use_distinct_lifetime_scopes(hidden_size):
    raw = declared(template='mesh_twoshot')
    raw['model']['hidden_size'] = hidden_size
    operations = first_call(raw)
    report = graph_costs(raw,operations,isolated=False)
    peaks = report['execution']['peak_allocated_per_unit']
    assert len(peaks) == 63 and set(peaks.values()) == {UB_RESIDENCY}
    phase = next(o for o in operations if o.family == 'hccl_local'
                 and dict(o.metadata)['hccl_phase']['phase'] == 'precopy')
    isolated = operator_costs(raw,(phase,))['operators'][0]
    assert isolated['peak_buffers']['ub@r0:s0'] == UB_RESIDENCY


def _observed_call(call):
    shape = (5120,)
    op = OpSpec("tp_all_gather", "allgather", "decode", 0, 0, (), shape, shape, shape, shape,
                link_bytes=10240, dtype="bf16", layout="contiguous",
                metadata=(("group_axis", "tp"), ("collective", "allgather"),
                          ("hccl_observed_call", call)))
    return lower((op,), SimpleNamespace(kernels={}), "bf16")[0]


@pytest.fixture
def observed_evidence():
    return json.loads(EVIDENCE_PATH.read_text())["call"]


def test_observed_allgather_binds_two_aliases_to_one_timing_owner(observed_evidence):
    op = _observed_call(observed_evidence)
    binding = bind_native_call(op)
    assert binding.api["entry"] == "torch.distributed.all_gather_into_tensor"
    assert (binding.execution_core, binding.typed["dtype"]) == ("aiv", "bf16")
    shapes = binding.typed["shapes"]
    assert (shapes["input"], shapes["output"]) == ([5120], [10240])
    assert binding.kernels[0]["name"] == "aiv_all_gather_bfloat16_t"
    assert {alias["rank_id"] for alias in binding.observed_aliases} == {"rank_0", "rank_1"}
    assert all(value not in json.dumps(binding.to_dict()) for value in ("56.348811", "56.462669", "duration_ms"))
    structure = export_op_structure({0: (op,), 1: (op,)}, model={},
                                    execution={"phase": "decode"}, native_evidence=True)
    records = structure["native_calls"]
    assert [record["call_id"] for record in records] == [binding.call_id] * 2
    assert {record["timing_owner"]["op_id"] for record in records} == {op.op_id}
    alternate = deepcopy(observed_evidence)
    alternate["typed"]["attrs"]["expansion"] = "AICPU_TS"
    alternate.update(kernels=[], execution_core=None)
    assert bind_native_call(_observed_call(alternate)).call_id != binding.call_id


@pytest.mark.parametrize("field,value,error", [
    ("dtype", "fp16", "dtype_mismatch"),
    ("shapes", {"input": [4096], "output": [10240]}, "shape_mismatch"),
    ("attrs", {"ranks": 1}, "attr_mismatch"),
])
def test_observed_allgather_validates_typed_invocation(observed_evidence, field, value, error):
    binding = bind_native_call(_observed_call(observed_evidence))
    args = binding.prepare_args()
    assert validate_invocation(binding, args)["call_id"] == binding.call_id
    with pytest.raises(ModelingError, match=error):
        validate_invocation(binding, {**args, field: value})


@pytest.mark.parametrize("label,context", [("p4k", 4104), ("p16k", 16392)])
def test_sequence_parallel_caller_owns_each_collective_once(label, context):
    raw = json.loads((CALLER_CASES / CALLER_FILENAME.format(label)).read_text())
    layers = raw["model"]["layers"] = 93
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=context,
                        sequences=32, apply_body_defaults=True)[0]
    phases = [meta for op in ops if "hccl_source_group" in (meta := dict(op.metadata))]
    entries, exits = [{p["hccl_source_group"]: p["hccl_selection"] for p in phases
                      if p["device_kernel_group"].get(boundary)} for boundary in ("entry", "exit")]
    for boundary in ("entry", "exit"):
        assert sum(bool(p["device_kernel_group"].get(boundary)) for p in phases) == len(entries)
    assert entries == exits and "sp_output_gather.hccl" in entries
    assert Counter(s["collective"] for s in entries.values()) == {
        "allgather": 2 * layers + 1, "reduce_scatter": 2 * layers}
    for selection in entries.values():
        tokens = 4 if selection["collective"] == "allgather" else 32
        assert selection["payload_bytes"] == tokens * 7168 * 2


def test_sequence_parallel_service_is_inclusive_and_preserves_physical_work():
    raw = json.loads((CALLER_CASES / CALLER_FILENAME.format("p4k")).read_text())
    raw["model"]["layers"] = 1
    inputs = dict(scope="body", phase="decode", tokens=1, context=4104, sequences=32)
    unpriced = deepcopy(raw)
    unpriced["runtime"]["kernels"].pop("service_regimes")
    slow = deepcopy(raw)
    slow["machine"]["device"] = {"hbm_bw_bytes_per_s": 1e8}
    before, calibrated, constrained = [predict(case, **inputs) for case in (unpriced, raw, slow)]
    assert before["body_forward_ms"] < calibrated["body_forward_ms"] < constrained["body_forward_ms"]
    assert before["execution"]["work"] == calibrated["execution"]["work"] == constrained["execution"]["work"]
    _assert_released(calibrated["execution"])
