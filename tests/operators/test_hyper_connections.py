"""Hyper-connections planning, caller semantics, and physical execution.

[Role] Guard governed operator dispatch and retained source-specific callers.
[Invariants] Independent work ledgers, owner placement, and causal stage edges
  agree with the declared tensor contracts and survive Engine replay.
[Timeouts & Locks] Bounded analytical workloads and the session C11 build.
[Side-Effect Policy] Read-only case fixtures, in-memory plans and Engine state.
"""
from collections import Counter
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
import pickle

import pytest

from _support.accel import close
from _support.axes import ARCH_IDS, PHASES, REPRESENTATIVE_SKUS, SKUS
from mindie_sim import graph_costs, operator_costs, target_graphs, target_operators
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.hyper_connections.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operators import FamilyDispatcher, FamilyRegistry, OperatorPlanningContext
from mindie_sim.prediction.operators import hyper_connections as hc
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import DataPath
from mindie_sim.schema.datapath import ComputeArchIdentity, DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure import export_structure

pytestmark = pytest.mark.tier0
MEMBASE, REGBASE = (axis.values[0] for axis in ARCH_IDS)
MIN_UB = 192 * 1024
ATTNRES_RELOAD_UB = 248 * 1024
ATTNRES_RELOAD_HISTORY_ROWS = 104
ATTNRES_UNFUSED_COPY_BYTES, ATTNRES_FUSED_COPY_BYTES = 434_176, 432_128
POOLS = {"ub": 256 * 1024, "l1": 512 * 1024, "l0a": 65536, "l0b": 65536, "l0c": 262144}
PEAKS = {"bf16": 1e12, "hf32": 1e12, "fp32": 2e12}
PRE, PRE_SINK, POST, COLLAPSE = "hc_pre", "hc_pre_sinkhorn", "hc_post", "mhc_post_collapse"
TEXT, ATTN, ENGRAM = "text_gated_residual", "attn_res_fwd", "engram_gate"
ROOT = Path(__file__).resolve().parents[2] / "examples"
SPLIT_RESIDUAL = {"residual_values_fp32", "attention_residual_scores", "attention_residual_softmax",
                  "attention_residual_mix", "attention_residual_output_cast", "residual_add"}

def _raw(path):
    return json.loads((ROOT / path).read_text())

def _op(kind, shape, **meta):
    return OpSpec(kind, "hyper_connections", "decode", 0, 0, (), shape, shape, shape, shape,
                  recipe=kind, dtype="bf16", deps=("input",), source_boundary="kernel",
                  metadata=tuple(meta.items()))

def _geometry(arch=REGBASE, *, aic=4, aiv=None, ub=MIN_UB, l2=2**20):
    aiv = 2 * aic if aiv is None else aiv
    pools = {**POOLS, "ub": ub, "l2": l2}
    dp = DatapathCapabilitySpec(ComputeArchIdentity(arch, None), aic, aiv, (), (),
                                frozenset(), (), None, buffers=tuple(pools.items()))
    units = dict(cube=aic, mte1=aic, mte2_aic=aic, fixpipe=aic, mte2_aiv=aiv,
                 mte3=aiv, vector=aiv, hbm=1, l2=1)
    return Geometry(aic, units, pools, {"l2": 1, "ub": aiv}, datapath=dp)

def _context(op, geom, *, rates=None, modes=None):
    units = geom.resource_units
    return RecipeContext(op, {}, {role: role for role in units}, {**dict.fromkeys(units, 1e8), "cube": 1, **(rates or {})},
                         geom.compute_units, resource_units=units, compute_peaks=PEAKS,
                         pool_names=dict(zip(geom.buffers, geom.buffers)), pool_capacities=geom.buffers,
                         pool_units=geom.buffer_units, service_modes=modes or {})

def _engine(ctx, backend="python"):
    return Engine([Resource(role, ctx.rates[role], units=units)
                     for role, units in ctx.resource_units.items() if units],
                    pools=ctx.pool_capacities, pool_units=ctx.pool_units,
                    backend=backend, trace_limit=2000)

def _run(program, ctx, *, backend="python", blocker=None):
    engine = _engine(ctx, backend)
    ready = engine.submit("input", release=7)
    if blocker:
        role, work, owner, delay = blocker
        engine.submit("competitor", {role: work}, release=7 + delay, placement={role: (owner,)})
    done = program.start(engine, "point", ready)
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    status = engine.summary()
    assert status["live_tasks"] == status["live_programs"] == 0
    assert not any(engine.allocated.values())
    return engine, done.time - ready.time

def _serviced(engine):
    return {role: amount for role, amount in engine.serviced.items() if amount}

def _stages(plan, op, geom):
    return {stage.name: stage for stage in plan.program(_context(op, geom)).stages}

def _sinkhorn_oracle(m, c, iters):
    adds, entries = c * (c - 1), c**2
    half_steps = [adds + entries + (c if step else 0) for step in range(2 * iters)]
    return m * (adds + 3 * entries + sum(half_steps))

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("kind,c,iters,split,aux", [(PRE, 3, 2, 1, 0), (PRE, 4, 20, 3, 2), (PRE, 4, 20, 2, None)])
def test_mix_projection_sinkhorn_and_owner_partition(arch, kind, c, iters, split, aux):
    m, h = 11, 64
    k, n = c * h, c * (c + 2) + 2
    geom = _geometry(arch)
    op = _op(kind, (m, c, h), n_mix=n, sinkhorn_iters=iters, split_k=split, mhc_aux_cores=aux)
    plan = hc.build(op, geom)
    assert plan.work.matrix == {"bf16": 2 * m * k * n, "fp32": m * (2 * k - split)}
    sink = _sinkhorn_oracle(m, c, iters)
    assert hc.sinkhorn_vector_ops(m, c, iters) == plan.metadata["sinkhorn_ops"] == sink
    if aux is None:
        aux = min(range(1, 8), key=lambda a: max(ceil(m / a) * (sink // m),
                                               ceil(m / (8 - a)) * h * (2 * c - 1)))
    stages = _stages(plan, op, geom)
    assert (plan.metadata["primary_aiv_cores"], plan.metadata["mhc_aux_cores"]) == (8 - aux, aux)
    assert stages["pre_mix"].placement["vector"] == tuple(range(8 - aux))
    assert stages["sinkhorn"].placement["vector"] == tuple(range(8 - aux if aux else 0, 8))
    assert stages["sinkhorn"].work["vector"] == sink
    assert stages["projection_workspace_write"].metadata["logical_bytes"] == m * (n + 1) * split * 4
    assert stages["projection_workspace_read"].after == ("projection_workspace_write",)
    if split > 1:
        assert stages["projection_workspace_write"].after == ("split_k_partials",)
        assert stages["split_k_merge"].after == ("projection_workspace_read",)
        assert stages["split_k_merge"].work["vector"] == m * (n + 1) * (split - 1)

@pytest.mark.parametrize("arch", ARCH_IDS)
def test_standalone_sinkhorn_uses_precomputed_projection_and_inverse_rms(arch):
    m, c, h = 1, 4, 4096
    geom = _geometry(arch, aic=0, aiv=2)
    op = _op(PRE_SINK, (m, c, h), split_k=2)
    plan = hc.build(op, geom)
    assert plan.work.matrix == {} and plan.work.apply(op).flops == 0.0
    assert plan.metadata["projection_flops"] == 0
    stages = _stages(plan, op, geom)
    mix = c * (c + 2)
    assert stages["precomputed_coefficient_inputs"].metadata["logical_bytes"] == 4 * (mix + 1 + mix + 3)
    assert stages["sinkhorn"].work["vector"] == _sinkhorn_oracle(m, c, 20)
    assert stages["comb_output"].metadata["logical_bytes"] == 4 * c**2
    assert stages["post_gate_output"].metadata["logical_bytes"] == 4 * c
    assert stages["pre_mix_output"].metadata["logical_bytes"] == 2 * h
    assert all(set(stage.placement) <= {"mte2_aiv", "vector", "mte3"} for stage in stages.values())
    ctx = _context(op, geom)
    assert _serviced(_run(plan.lower(ctx), ctx)[0]) == pytest.approx(ctx.bind(plan.work))

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("kind,m", [(POST, 73), (COLLAPSE, 72), (PRE, 0), (PRE_SINK, 0),
                                    (POST, 0), (COLLAPSE, 0)])
def test_post_contraction_collapse_and_idle_work(arch, kind, m):
    c, h = 4, 128
    op, geom = _op(kind, (m, c, h)), _geometry(arch)
    plan = hc.build(op, geom)
    assert not plan.work.matrix and plan.metadata["sinkhorn_ops"] == 0
    stages = _stages(plan, op, geom)
    if not m:
        assert not stages and not plan.work.resources
        return
    assert stages["post_inputs"].metadata["logical_bytes"] == m * (2 * h * (c + 1) + 4 * c * (c + 1))
    assert stages["stream_contraction"].work["vector"] == m * c * h * (2 * c + 1)
    assert stages["post_output"].metadata["logical_bytes"] == 2 * m * h * (1 if kind == COLLAPSE else c)
    if kind == COLLAPSE:
        assert stages["collapse_and_norm"].work["vector"] == m * (h * (2 * c + 3) + 2)
        assert stages["post_output"].after == ("collapse_and_norm",)

@pytest.mark.parametrize("kind,shape,meta,error", [
    (PRE_SINK, (1, 3, 64), {}, "requires_hc4_and_iters20"),
    (PRE_SINK, (1, 4, 64), {"sinkhorn_iters": 19}, "requires_hc4_and_iters20"),
    (COLLAPSE, (73, 4, 64), {}, "row_cutoff_exceeded"),
    (PRE, (1, 4, 64), {"mhc_aux_cores": 8}, "leaves_no_primary"),
])
def test_mhc_rejects_out_of_domain_contracts(kind, shape, meta, error):
    with pytest.raises(ModelingError, match=error):
        hc.build(_op(kind, shape, **meta), _geometry())

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("binding,m,excluded", [("ez1008", 512, True), ("ez1008", 513, False),
                                                ("cann_current", 1024, False), ("cann_current", 0, False)])
def test_text_gate_stack_boundary(arch, binding, m, excluded):
    geom = _geometry(arch)
    op = _op(TEXT, (m, 4, 256), rank=64, has_gate=True, stack_binding=binding)
    plan = hc.build(op, geom)
    stages = _stages(plan, op, geom)
    if not m:
        assert plan.metadata["selected_template"] == "empty_noop"
        assert not stages and not plan.work.resources and not plan.work.matrix
        return
    assert plan.metadata["selected_template"] == ("decomposed_fallback" if excluded else "native_large")
    assert plan.metadata["excluded_fusion"] == ("ez1008_m_le_512" if excluded else None)
    assert ("decomposed_gate_write" in stages) == excluded
    assert stages["inject_gate"].work["vector"] == m * 4 * (4 + 2)
    assert stages["channel_mean"].work["vector"] == m * 256 * 4
    assert plan.work.matrix == {"bf16": 2 * m * (4 * 256) * (2 * 64 + 4)}


@pytest.mark.parametrize("m,template,task,used", [
    (1, "native_a2_small", 32, 2), (64, "native_a2_small", 128, 4),
    (640, "native_a2_small", 320, 10), (1024, "native_large", 64, 16),
])
def test_text_gate_membase_task_boundary(m, template, task, used):
    geom = _geometry(MEMBASE, aic=24)
    plan = hc.build(_op(TEXT, (m, 2, 256), rank=64, stack_binding="cann_current"), geom)
    # The host's basic, non-cross-core C0 split-K configuration gives one partition.
    assert tuple(plan.metadata[key] for key in ("selected_template", "m_task", "used_aic", "splitk_factor")) == (
        template, task, used, 1,
    )


@pytest.mark.parametrize("arch", ARCH_IDS)
def test_text_gate_read_sigmoid_and_mean_without_injection(arch):
    m, c, h, rank = 1, 2, 256, 64
    geom = _geometry(arch)
    op = _op(TEXT, (m, c, h), rank=rank, has_gate=False)
    plan = hc.build(op, geom)
    stages = _stages(plan, op, geom)
    sigmoid, multiply, mean = (stages[name] for name in ("read_gate_sigmoid", "sigmoid_weighted_products", "channel_mean"))
    assert plan.work.matrix == {"bf16": 2 * m * c * h * 2 * rank}
    assert sigmoid.work["vector"] == 4 * 512
    assert multiply.work["vector"] == 512 and multiply.after == (sigmoid.name,)
    assert mean.work["vector"] == h * ((c - 1) + 1) and mean.after == (multiply.name,)
    assert "inject_output" not in stages

@pytest.mark.parametrize("c,h,r,aic,reason", [
    (2, 256, 64, 1, None), (10, 4096, 896, 4, None),
    (1, 256, 64, 4, "hc_mult"), (11, 256, 64, 4, "hc_mult"),
    (4, 128, 64, 4, "hidden_size"), (4, 257, 64, 4, "hidden_size"),
    (4, 4352, 64, 4, "hidden_size"), (4, 256, 32, 4, "rank"),
    (4, 256, 65, 4, "rank"), (4, 256, 960, 4, "rank"),
    (4, 256, 64, 0, "aic_cores"),
])
def test_text_gate_small_fused_domain(c, h, r, aic, reason):
    m, geom = 511, _geometry(aic=aic)
    guard = hc.evaluate_text_gated_guard(hc.TextGatedResidualContract(m, c, h, c * h, r), REGBASE, aic)
    if reason:
        assert (guard.selected_template, guard.splitk_factor, guard.excluded_fusion) == (
            "decomposed_fallback", 1, f"a5_small_{reason}")
        if aic:
            plan = hc.build(_op(TEXT, (m, c, h), rank=r), geom)
            assert plan.metadata["excluded_fusion"] == guard.excluded_fusion
            assert plan.metadata["selected_template"] == guard.selected_template
            with pytest.raises(UnmodeledCapabilityError, match="fused_small_abi_unsupported"):
                hc.build(_op(TEXT, (m, c, h), rank=r, requested_abi="fusedSmall"), geom)
        return
    op = _op(TEXT, (m, c, h), rank=r, requested_abi="fusedSmall")
    plan = hc.build(op, geom)
    assert plan.metadata["selected_template"] == guard.selected_template == "native_a5_small"
    assert plan.metadata["splitk_factor"] == guard.splitk_factor == 14
    stages = _stages(plan, op, geom)
    assert stages["text_split_k_merge"].work["vector"] == m * r * (14 - 1)

def _resident_limit_oracle(h, ub, *, regbase, fuse_add, fused_chain=False, need_backward=False):
    bf16, fp32 = 32 * ceil(h / 16), 32 * ceil(h / 8)
    candidates = []
    for rows in range(1, 1 + ub // bf16):
        meta = 8 * ceil(rows / 8)
        score = 64 * ceil(meta / 64) if regbase and rows > 128 else meta
        scratch = meta if regbase else max(meta, 64)
        resident = rows - (fuse_add or fused_chain)
        buffers = (2 + resident + (fuse_add or fused_chain)) * bf16 + 3 * fp32
        scalars = score + scratch + 8 * meta + 8 + (meta + 8 if need_backward else 0)
        if buffers + 4 * scalars + 32 * 1024 <= ub:
            candidates.append(rows)
    cap = max(candidates, default=0)
    if regbase:
        cap = min(cap, max(0, (ub - 32 * 1024 - h * (20 + 2 * fuse_add)) // (2 * h + 4)))
    return cap

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("h,fuse,chain,backward,physical_cap", [
    (1024, False, False, False, None), (1024, True, False, False, 5),
    (17, True, True, True, None), (17, False, True, False, 999),
])
def test_attnres_exact_resident_cutoff(arch, h, fuse, chain, backward, physical_cap):
    ub, m = MIN_UB, 3
    cap = _resident_limit_oracle(h, ub, regbase=arch == REGBASE, fuse_add=fuse,
                                 fused_chain=chain, need_backward=backward)
    limit = cap if physical_cap is None else min(cap, physical_cap)
    for j, cached, template in ((limit - 1, 0, "resident"), (limit, 0, "reload"),
                                (limit, 2, "cooperative_reload")):
        op = _op(ATTN, (m, h), capacity=limit + 2, valid_history=j, fuse_add=fuse,
                 cached_rows=cached, physical_capacity_cap=physical_cap, fused_chain=chain, need_backward=backward)
        plan = hc.build(op, _geometry(arch, ub=ub))
        assert plan.metadata["resident_row_limit"] == limit
        assert plan.metadata["selected_template"] == template
        assert plan.metadata["reread_bytes"] == (0 if template == "resident" else 2 * h * m * (j - cached))

def test_attnres_regbase_resident_admission_boundary():
    h, ub = 1024, ATTNRES_RELOAD_UB
    assert hc.attnres_max_resident_rows(h, ub, fuse_add=True) == 96
    for rows, template in ((96, "resident"), (97, "reload")):
        op = _op(ATTN, (1, h), capacity=rows, valid_history=rows - 1, fuse_add=True)
        assert hc.build(op, _geometry(ub=ub)).metadata["selected_template"] == template

@pytest.mark.parametrize("cached", [0, 2])
@pytest.mark.parametrize("fits", [True, False])
@pytest.mark.parametrize("backend", ["python", "c"])
@pytest.mark.parametrize("m,fuse_add,fused_chain", [(3, True, False), (1, False, True), (1, False, False)])
def test_attnres_reload_service_tracks_per_token_live_l2(cached, fits, backend, m, fuse_add, fused_chain):
    h, j = 1024, ATTNRES_RELOAD_HISTORY_ROWS
    row, unfused = 2 * h, not (fuse_add or fused_chain)
    working_set = row * (j + unfused)
    geom = _geometry(aiv=1, l2=working_set if fits else working_set - row, ub=ATTNRES_RELOAD_UB)
    op = _op(ATTN, (m, h), capacity=j + 1, valid_history=j, cached_rows=cached,
             fuse_add=fuse_add, fused_chain=fused_chain)
    plan, ctx = hc.build(op, geom), _context(op, geom)
    engine, _ = _run(plan.lower(ctx), ctx, backend=backend)
    if m == 1 and not cached:
        copy_bytes = ATTNRES_UNFUSED_COPY_BYTES if unfused else ATTNRES_FUSED_COPY_BYTES
        assert engine.serviced["mte2_aiv"] == pytest.approx(copy_bytes, rel=1e-12)
    trace = {task["task"].rsplit(":", 1)[-1]: task for task in engine.trace}
    inputs = "history_and_prefix" if unfused else "history"
    names = (f"pass1_{inputs}", "pass1_score", "softmax", f"pass2_{inputs}", "pass2_weighted", "output")
    for i in range(m):
        phases = [trace[f"token_{i}_{name}"] for name in names]
        assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(phases, phases[1:]))
        assert phases[0]["placement"]["mte2_aiv"] == phases[3]["placement"]["mte2_aiv"] == (0,)
        if i + 1 < m:
            assert phases[-1]["end_ms"] <= trace[f"token_{i + 1}_prefix_input"]["start_ms"]
    reread = m * row * (j - cached + unfused)
    assert engine.serviced.get("l2", 0) == pytest.approx(reread if fits else 0)
    first_reads_and_writes = row * (2 + m * (j + 2 + 2 * fuse_add))
    assert engine.serviced["hbm"] == pytest.approx(first_reads_and_writes + (0 if fits else reread))
    assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
    close(engine)

@pytest.mark.parametrize("fuse_add", [True, False])
def test_attnres_concurrent_reload_eviction_on_shared_die(fuse_add):
    h, j = 128, 3
    working_set = 2 * h * (j + (not fuse_add))
    geom = _geometry(aiv=1, l2=working_set)
    op = _op(ATTN, (1, h), capacity=j, valid_history=j, physical_capacity_cap=j, fuse_add=fuse_add)
    plan, ctx = hc.build(op, geom), _context(op, geom)
    program = plan.lower(ctx)
    assert _run(program, ctx)[0].serviced["l2"] == pytest.approx(working_set)
    engine = _engine(ctx)
    ready = engine.submit("input", release=7)
    first = program.start(engine, "first", ready, owner="stream_a")
    second = program.start(engine, "second", ready, owner="stream_b")
    engine.advance(until=(first, second))
    assert not engine.serviced.get("l2")
    assert engine.serviced["hbm"] == pytest.approx(2 * (plan.work.resources["hbm"] + working_set))
    # Finishing both streams retires their working sets for the next token.
    third = program.start(engine, "third", second)
    engine.advance(until=(third,))
    assert engine.serviced["l2"] == pytest.approx(working_set)

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("mask,chunk,branches", [
    ((1, 1, 1), 256, 1), ((1, 0, 1), 64, 3), ((0, 0, 0), 64, 3),
])
def test_engram_reductions_gate_and_mask(arch, mask, chunk, branches):
    m, h, active = len(mask), 256, sum(mask)
    shape = (m, h) if branches == 1 else (m, branches, h)
    op, geom = _op(ENGRAM, shape, active_mask=mask, chunk_dim=chunk), _geometry(arch)
    plan = hc.build(op, geom)
    stages = _stages(plan, op, geom)
    rows = m * branches
    assert plan.metadata["active_tokens"] == active and plan.metadata["sinkhorn_ops"] == 0 and not plan.work.matrix
    assert plan.metadata["evaluated_rows"] == rows
    assert plan.metadata["reduction_adds"] == rows * (3 * (h - 1)) == rows * 765
    reduce, gate = stages["hidden_slice_partial_reductions"], stages["rms_signed_root_sigmoid"]
    assert reduce.work["vector"] == rows * (765 + h * (2 + 3))
    assert gate.work["vector"] == rows * (2 * 4 + 3 + 4 + 4)
    assert gate.after == (reduce.name,)
    assert stages["token_mask"].after == (gate.name,)
    assert stages["apply_token_mask"].after == ("token_mask",)
    value = stages["engram_value_and_spilled_x"]
    assert value.after == ("apply_token_mask",)
    assert value.metadata["logical_bytes"] == 2 * rows * h * (1 + (chunk < h))
    assert stages["engram_gated_add"].work["vector"] == 2 * rows * h
    assert stages["engram_output"].metadata["logical_bytes"] == 2 * rows * h
    count_only = hc.build(_op(ENGRAM, shape, active_tokens=active, chunk_dim=chunk), geom)
    assert count_only.work == plan.work

def test_datapath_exclusive_abi_and_missing_route():
    op = _op(TEXT, (7, 4, 256), rank=64, requested_abi="fusedSmall")
    with pytest.raises(UnmodeledCapabilityError, match="strategy_unsupported"):
        hc.build(op, _geometry(MEMBASE))
    geom = _geometry()
    dp = replace(geom.datapath, usable_paths=(DataPath("store", "ub", "gm", ("mte3",)),))
    with pytest.raises(UnmodeledCapabilityError, match="datapath_unsupported"):
        hc.build(op, replace(geom, datapath=dp))

@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("kind,shape,meta", [
    (PRE, (3, 4, 128), {"split_k": 2}), (POST, (3, 4, 128), {}),
    (PRE_SINK, (2, 4, 128), {}), (COLLAPSE, (3, 4, 128), {}),
    (TEXT, (512, 4, 256), {"rank": 64}), (ATTN, (3, 128), {"capacity": 4, "valid_history": 2}),
    (ENGRAM, (3, 128), {"active_mask": (1, 0, 1)}),
])
def test_family_dispatch_descriptor_work_and_engine_replay(hardware, kind, shape, meta):
    raw = _raw("source-families/attnres-prefix-control.json")
    raw["machine"] = {"backend": "ascend", "sku": hardware, "num_devices": 1}
    device = Geometry.from_devices(parse_case(raw).machine.dies)
    geom = _geometry(device.datapath.arch_id)
    geom = replace(geom, datapath=replace(geom.datapath, usable_paths=device.datapath.usable_paths))
    op = _op(kind, shape, **meta)
    occurrence = OpOccurrence(2, op.stage, 3, op.op_id)
    ctx = OperatorPlanningContext(occurrence, geom, geom.datapath)
    registry = FamilyRegistry((hc.register_family(),))
    assert registry.lookup(op).resolve is hc.resolve_operator_plan
    bound = FamilyDispatcher(registry).resolve_plan(op, ctx)
    plan, descriptor = pickle.loads(pickle.dumps((bound.recipe, bound.descriptor)))
    assert descriptor.owner == occurrence and descriptor.member_op_ids == (op.op_id,)
    assert descriptor.sources == bound.sources and descriptor.sources
    assert descriptor.implementation == kind and descriptor.repeats == 1
    assert bound.op.local_shape == op.local_shape and bound.op.deps == op.deps
    ctx = _context(bound.op, geom)
    program = plan.program(ctx)
    assert [(p.phase_id, p.after) for p in descriptor.phases] == [(s.name, s.after) for s in program.stages]
    ledger = Counter()
    for phase in descriptor.phases:
        for item in phase.work:
            ledger[item.resource, item.dtype if item.unit == "flop" else item.unit] += item.amount
    assert dict(ledger) == {**{("cube", dtype): flops for dtype, flops in plan.work.matrix.items()},
                            **{(role, "op" if role == "vector" else "bytes"): amount
                               for role, amount in plan.work.resources.items() if amount}}
    reference = _run(bound.recipe.lower(ctx), ctx)
    native = _run(plan.lower(ctx), ctx, backend="c")
    for engine, _ in (reference, native):
        assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
    assert native[1] == pytest.approx(reference[1], abs=1e-10)
    status = native[0].accel_status()
    assert status["native_steps"] > 0 and status["fallback_reason"] is None
    close(native[0])

@pytest.mark.parametrize("m", [1, 41, 1025])
def test_projection_precision_and_owner_tail_service(m):
    h, c, cores = 7168, 4, 20
    geom = _geometry(MEMBASE, aic=cores)
    op = _op(PRE, (m, c, h), split_k=3, mhc_aux_cores=0)
    plan = hc.build(op, geom)
    width, mix, split = c * h, c * (c + 2), 3
    assert plan.work.matrix == {"bf16": 2 * m * width * mix, "fp32": m * (2 * width - split)}
    assert plan.work.resources["mte2_aic"] == 2 * (m + mix) * width
    ctx = _context(plan.work.apply(op), geom)
    engine, duration = _run(plan.lower(ctx), ctx)
    assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
    cube = next(task for task in engine.trace if task["task"].endswith(":split_k_partials"))
    assert cube["placement"]["cube"] == tuple(range(min(cores, m * split)))
    for dtype in plan.work.matrix:
        slow = replace(ctx, compute_peaks={**PEAKS, dtype: PEAKS[dtype] / 10})
        assert _run(plan.lower(slow), slow)[1] > duration

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("m,h", [(1, 2048), (41, 2112), (43, 4128)])
def test_post_owner_tail_service_and_shared_store(arch, m, h):
    geom = _geometry(arch, aic=20, ub=POOLS["ub"])
    geom = replace(geom, resource_units={**geom.resource_units, "mte3": 20})
    op = _op(POST, (m, 4, h))
    plan = hc.build(op, geom)
    read, write = m * (10 * h + 80), m * 8 * h
    assert plan.work.resources["mte2_aiv"] == 32 * ceil(read / 32)
    assert plan.work.resources["mte3"] == write
    assert plan.work.resources["vector"] == m * 4 * h * (2 * 4 + 1)
    ctx = _context(plan.work.apply(op), geom,
                   rates={**dict.fromkeys(geom.resource_units, 1e14), "mte3": 20 * 8 * h})
    engine, duration = _run(plan.lower(ctx), ctx)
    store_units = min(20, ceil(m / 2))
    assert duration == pytest.approx(m / store_units, abs=1e-6)
    assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))

@pytest.mark.parametrize("slow", ["vector", "mte3", "mte1", "mte2_aiv", "hbm"])
def test_split_projection_backpressure_and_terminal_service(slow):
    geom = _geometry(aic=32, ub=POOLS["ub"])
    op = _op(PRE, (128, 4, 1024), split_k=3, mhc_aux_cores=2)
    plan = hc.build(op, geom)
    ctx = _context(op, geom, rates={slow: 1e6})
    basis, elapsed = _run(plan.lower(ctx), ctx)
    faster = replace(ctx, rates={**ctx.rates, slow: 1e8})
    assert _run(plan.lower(faster), faster)[1] < elapsed
    competitor = ("mte2_aiv", 16384, 0, 0.001)
    contested, _ = _run(plan.lower(ctx), ctx, blocker=competitor)
    expected = ctx.bind(plan.work)
    expected["mte2_aiv"] += competitor[1]
    assert _serviced(contested) == pytest.approx(expected)
    trace = {row["task"].rsplit(":", 1)[-1]: row for row in basis.trace}
    assert trace["projection_workspace_write"]["end_ms"] <= trace["projection_workspace_read"]["start_ms"]
    assert trace["sinkhorn_workspace_write"]["end_ms"] <= trace["sinkhorn_workspace_read"]["start_ms"]
    ready = basis.submit("ready_again")
    done = plan.lower(ctx).start(basis, "repeat", ready)
    basis.advance(until=(done,))
    assert done.time == pytest.approx(ready.time + elapsed)

def test_attnres_owner_arithmetic_gamma_and_empty_history():
    h = 64
    for aiv in (2, 64):
        geom = _geometry(aiv=aiv)
        op = _op(ATTN, (1, h), capacity=8, valid_history=1, fuse_add=False, output_norm=True)
        plan = hc.build(op, geom)
        ctx = _context(op, geom, rates={"vector": 1e6, "mte2_aiv": 1e16, "mte3": 1e16, "hbm": 1e16})
        engine, elapsed = _run(plan.lower(ctx), ctx)
        assert engine.serviced["mte2_aiv"] == 640 and not engine.serviced.get("mte2_aic")
        assert elapsed == pytest.approx(plan.work.resources["vector"] * aiv / 1e6)
        assert plan.work.resources["vector"] >= 2 * (7 * h - 2) and not plan.work.matrix
    empty = hc.build(_op(ATTN, (5, h), capacity=8, valid_history=0, fuse_add=False), _geometry())
    assert empty.metadata["history_read_bytes"] == empty.metadata["reread_bytes"] == 0


def test_prefix_block_transition_caller_scope_and_semantic_rejection():
    selected = parse_case(_raw("source-families/attnres-prefix-control.json"))
    unselected = replace(selected, runtime=replace(selected.runtime, kernels={}))
    options = dict(phase="prefill", tokens=5, scope="body")
    assert SPLIT_RESIDUAL <= {op.recipe for op in target_operators(unselected, **options)}
    ops = target_operators(selected, **options)
    assert not SPLIT_RESIDUAL.intersection(op.recipe for op in ops)
    points = [op for op in ops if op.recipe == ATTN]
    views = [dict(op.metadata)["attn_res_view"] for op in points]
    assert [(v["valid_blocks"], v["block_write_idx"], v["fuse_add"]) for v in views] == [
        (0, 0, False), (1, -1, False), (1, -1, True), (1, -1, True),
        (1, 1, True), (2, -1, False), (2, -1, True)]
    banks = (points[0].op_id, points[4].op_id)
    assert dict(points[-1].metadata)["attn_res_bank_ids"] == banks
    for layer, consumer in enumerate(points[2::2]):
        mlp = [op for op in ops if op.layer == layer and dict(op.metadata).get("structure_block") == "ffn"
               and op.recipe != ATTN][-1]
        assert [point for point in points if mlp.op_id in point.deps] == [consumer]
    with pytest.raises(ModelingError, match="requires_pp1"):
        Frontend().body(selected.model, selected.runtime, (RequestView("r", 5, 0, False, "prefill"),),
                        ready="input", tokens=5, mode="prefill", stages=(0, 0, 1), machine=selected.machine)
    for flag in ("engram", "hyper_connections"):
        plans = tuple(replace(layer, **{flag: True}) for layer in selected.model.layer_plans)
        with pytest.raises(ModelingError, match="selected_semantics_unsupported"):
            target_operators(replace(selected, model=replace(selected.model, layer_plans=plans)), **options)

@pytest.mark.parametrize("batch", (1, 32))
def test_bf16_caller_replaces_split_ops_and_retains_prefix(batch):
    raw = _raw(f"cases/high-concurrency-20261030/kimi-variants/a5_kimi_tp8_e16_bf16_c{batch}_p4k.json")
    inputs = dict(phase="decode", tokens=1, sequences=batch, context=4096, scope="body")
    ops = target_operators(raw, **inputs)
    points = [op for op in ops if op.recipe == ATTN]
    assert len(points) == 2 * 93 + 1
    assert not SPLIT_RESIDUAL.intersection(op.recipe for op in ops)
    banks = tuple(op.op_id for op in points[:-1:24])
    assert dict(points[-1].metadata)["attn_res_bank_ids"] == banks and set(banks) <= set(points[-1].deps)
    by_id = {op.op_id: op for op in ops}
    for layer, point in enumerate(points[:-1:2]):
        assert point.local_shape == (ceil(batch / 8), 7168)
        assert by_id[f"s0.l{layer}.attention.sp_gather"].deps == (point.op_id,)
    assert by_id["sp_output_gather"].deps == (points[-1].op_id,)
    for row in operator_costs(raw, (points[0], points[-1]), input_ready_ms=17)["operators"]:
        assert row["duration_ms"] > 0 and not row["service_before_input"]
        assert not row["live_tasks"] and not any(row["remaining_buffers"].values())
        assert row["declared_service"] == pytest.approx(row["serviced"])
        assert not any(name.startswith(("cube", "matrix")) for name in row["serviced"])

def test_receive_only_replicas_keep_expert_work_and_active_selection_is_checked():
    case = parse_case(_raw("source-families/v41-hc-source-control.json"))
    layers = case.model.layer_plans[:1]
    case = replace(case, model=replace(case.model, layers=1, body_layers=1, layer_plans=layers),
                   runtime=replace(case.runtime, pp_layer_partition=(1,)))
    options = dict(phase="decode", tokens=1, context=2048, sequences=[8] + [0] * 7, scope="body")
    graphs = target_graphs(case, **options)
    for replica, ops in graphs.items():
        assert sum(dict(op.metadata).get("fusion_role") in {PRE, POST} for op in ops) == (0 if replica else 4)
        assert any(op.recipe == "expert_combine" for op in ops)
    result = graph_costs(case, graphs, input_ready_ms=7, external_inputs={"analysis_input": 7}, isolated=False)
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    assert {row["replica"] for row in result["timeline"] if row["family"] == "grouped_gemm"} == set(graphs)
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    plain = replace(case, model=replace(case.model, layer_plans=(replace(layers[0], hyper_connections=False),)))
    with pytest.raises(ModelingError, match="native_fusion_selected_roles_unmatched"):
        target_graphs(plain, **options)

@pytest.mark.parametrize("phase", PHASES)
def test_manifest_prepares_source_work_and_preserves_public_consumers(phase):
    raw = _raw("cases/v4f-a3-w8a8-tp8.json")
    raw["machine"]["sku"] = REPRESENTATIVE_SKUS["a3"]
    raw.setdefault("runtime", {}).setdefault("kernels", {}).setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion", "options": {"roles": [PRE], "traffic_profile": "hc_pre_m_k_split_175ef4d4"}}
    raw["model"]["layers"] = 1
    before, case = json.dumps(raw), parse_case(raw)
    options = dict(phase=phase, tokens=8, scope="body")
    semantic = target_operators(case, resource_fusion=False, **options)
    _, members = next((d, ops) for d, ops in _groups(semantic) if d["role"] == PRE)
    before_members = tuple(op.to_dict() for op in members)
    groups = export_structure(case, **options)["subgraphs"]
    work = next(group["physical_work"] for group in groups if group["role"] == PRE)
    geom = Geometry.from_devices(case.machine.dies)
    fused = lower_native_fusions(members, case.runtime)[0]
    view = dict(fused.metadata)["fusion_tensor_view"]
    m, c, h = (view[key] for key in ("tokens", "hc", "hidden"))
    plan = hc.build(replace(fused, recipe=PRE, local_shape=(m, c, h)), geom)
    assert work["launches"] == 1 and work["matrix_flops"]
    assert plan.work.matrix["bf16"] == 2 * m * (c * h) * c * (c + 2)
    assert json.dumps(raw) == before
    consumer = replace(_op("consume", (8, 64)), deps=(members[-1].op_id,))
    fused, output = lower_native_fusions((*members, consumer), case.runtime)
    assert output.deps == (fused.op_id,)
    affine = next(op for op in members if op.recipe == "hc_normalize_scale_bias_gate")
    fused = lower_native_fusions((*members, replace(consumer, deps=(affine.op_id,))), case.runtime)[0]
    with pytest.raises(ModelingError, match="cannot_export_internal_intermediate"):
        hc.build(replace(fused, recipe=PRE, local_shape=(m, c, h)), geom)
    assert tuple(op.to_dict() for op in members) == before_members
