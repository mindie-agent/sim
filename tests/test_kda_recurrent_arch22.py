"""Real Engine controls for arch22 owner-sequence beta reuse and UB updates."""
from dataclasses import replace
import math

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.operators import kda_recurrent_arch22 as arch22
from mindie_sim.prediction.operators.kda_recurrent import scheduled_runs
from mindie_sim.prediction.recurrent import linear
from mindie_sim.schema.case import Software
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from test_kda_decode_family import caller
from test_kda_recurrent_source import execute


def run(*args, **kwargs):
    return execute(*args, planner=arch22.build, constructor=arch22.make_op, **kwargs)


@pytest.mark.parametrize("lengths,heads,units", (
    ((1,), 48, 4), ((4,)*4, 12, 4), ((1, 2, 3, 4, 5, 8), 7, 4),
    ((4, 4, 1, 1, 4), 48, 40), ((6,)*4, 4, 40),
))
def test_arch22_beta_loading_follows_owner_sequence_not_value_head(lengths, heads, units):
    blocks = min(units, heads*len(lengths))
    for owner in range(blocks):
        expected = []
        for sequence, q in enumerate(lengths):
            owned = [h for h in range(heads) if (sequence*heads+h)%blocks == owner]
            expected.extend((q, 1, int(i == 0)) for i in range(len(owned)))
        assert list(scheduled_runs(lengths, heads, blocks, owner,
                                   compact=False, beta_per_head=False)) == expected
        runs = list(scheduled_runs(lengths, heads, blocks, owner,
                                   compact=True, beta_per_head=False))
        assert sum(n for _, n, _ in runs) == len(expected)
        for q in set(lengths):
            assert sum(n for length, n, _ in runs if length == q) == sum(t[0] == q for t in expected)
            assert sum(beta for length, _, beta in runs if length == q) == sum(t[2] for t in expected if t[0] == q)


@pytest.mark.parametrize("lengths,heads,units,dv,state", (
    ((1,), 48, 4, 128, "fp32"), ((4,)*4, 12, 4, 256, "bf16"),
    ((1, 2, 3, 4, 5, 8), 7, 4, 128, "fp32"), ((6,)*4, 4, 40, 128, "fp32"),
))
def test_arch22_work_delta_is_only_real_beta_reuse_and_delta_broadcast(lengths, heads, units, dv, state):
    generic = execute(lengths, heads=heads, key_heads=1, units=units, dv=dv, state_dtype=state, ub=196608)
    selected = run(lengths, heads=heads, key_heads=1, units=units, dv=dv, state_dtype=state, ub=196608)
    blocks = min(units, len(lengths)*heads)
    reused = heads-min(heads, blocks)
    saved_beta_bytes = sum(lengths)*heads*2*reused
    saved_beta_cast = sum(math.ceil(q*heads/8)*8 for q in lengths)*reused
    broadcast = sum(lengths)*heads*128*dv
    old, new = generic[1].work.resources, selected[1].work.resources
    assert old["hbm"]-new["hbm"] == saved_beta_bytes
    assert old["mte2_aiv"]-new["mte2_aiv"] == saved_beta_bytes
    assert old["mte3"] == new["mte3"]
    assert new["vector"]-old["vector"] == broadcast-saved_beta_cast
    for key, amount in new.items():
        assert selected[2].serviced[key] == pytest.approx(amount, rel=1e-10)
    # Kernel bytes are the same live UB resources, despite different service.
    assert selected[2].peak_allocated == generic[2].peak_allocated


@pytest.mark.parametrize("q", (1, 4, 6, 8))
def test_shared_queue_pipeline_retains_arch22_compact_work_and_duration(q):
    detailed = run((q,)*4, heads=3, key_heads=1, units=4, compact=False, ub=196608)
    compact = run((q,)*4, heads=3, key_heads=1, units=4, compact=True, ub=196608)
    assert detailed[1].work == compact[1].work
    assert detailed[2].serviced == pytest.approx(compact[2].serviced, rel=1e-10)
    assert detailed[3] == pytest.approx(compact[3], rel=1e-10)
    assert compact[2].completed_count < detailed[2].completed_count


def test_owner_sequence_beta_reuse_stays_detailed_when_head_cycles_differ():
    detailed = run((4,)*4, heads=12, key_heads=3, units=4, compact=False, ub=196608)
    compact = run((4,)*4, heads=12, key_heads=3, units=4, compact=True, ub=196608)
    assert compact[3] == detailed[3]
    assert compact[2].completed_count == detailed[2].completed_count
    assert compact[1].metadata["kda_recurrent_schedule"]["resource_cycles"] == 4*12


def test_partial_launch_wave_does_not_rephase_unequal_owner_head_counts():
    rates = {"hbm": 1.6e12, "vector": 1.5e12, "mte2_aiv": 1.6e12, "mte3": 1.6e12}
    detailed = run((1,)*4, heads=12, key_heads=3, units=40, ub=196608, rates=rates, compact=False)
    compact = run((1,)*4, heads=12, key_heads=3, units=40, ub=196608, rates=rates, compact=True)
    assert compact[3] == pytest.approx(detailed[3], rel=1e-10)


@pytest.mark.parametrize("resource", ("hbm", "mte2_aiv", "mte3", "vector"))
def test_arch22_bandwidth_axes_and_external_competition(resource):
    runs = [run((4,)*4, heads=12, key_heads=3, units=4, ub=196608,
                rates={resource: 1e6*factor}) for factor in (.5, 1, 2)]
    assert runs[0][3] > runs[1][3] > runs[2][3]
    for compact in (False, True):
        _, plan, engine, _ = run((4,)*4, heads=12, key_heads=3, units=4, ub=196608,
                                compact=compact, contender={resource: 1e8})
        for key, amount in plan.work.resources.items():
            assert engine.serviced[key] == pytest.approx(amount+(1e8 if key == resource else 0), rel=1e-9)


def test_explicit_arch22_selector_reaches_ordinary_graph_with_one_fused_kernel():
    case, _ = caller((6,)*4)
    runtime = Software.from_mapping({"kernels": {"lowerings": {"kda": {
        "implementation": arch22.IMPLEMENTATION, "options": {"use_beta_sigmoid": True}}}}})
    die = replace(case.machine.dies[0], buffers=(("ub", 196608),))
    case = replace(case, runtime=runtime, machine=replace(case.machine, dies=(die,)))
    views = tuple(RequestView(str(i), 6, 2048, False, "decode") for i in range(4))
    ops = linear(prefix="kda", plan=case.model.layer_plans[0], views=views,
                 stage=0, deps="input", model=case.model, parallel=ParallelSpec(),
                 quant=case.model.quant, mode="decode", padded_tokens=24,
                 policy=AttentionPolicy(), runtime=runtime).ops
    native, = [op for op in ops if op.recipe == arch22.RECIPE]
    assert dict(native.metadata)["kda_recurrent_params"]["beta_dtype"] == "bf16"
    assert not any(op.recipe in ("qk_l2norm", "delta_gates", "linear_beta_sigmoid") for op in ops)
    result = graph_costs(case, ops, input_ready_ms=7, external_inputs={"input": 7}, isolated=False)
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
