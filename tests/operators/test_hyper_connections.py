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
from mindie_sim.prediction import family_attnres as prefix, family_hc as source
from mindie_sim.prediction import family_hc_arch35 as panel_source
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.hc_arch35_compact import finite_owner_template, first_owner
from mindie_sim.prediction.hc_arch35_program import phase_program, pre_geometry, pre_owner
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operators import FamilyDispatcher, FamilyRegistry, OperatorPlanningContext
from mindie_sim.prediction.operators import hyper_connections as hc
from mindie_sim.prediction.program import PipelineTemplateProgram, RecipeContext
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

def _run(program, ctx, *, backend="python", blocker=None):
    engine = Engine([Resource(role, ctx.rates[role], units=units)
                     for role, units in ctx.resource_units.items()],
                    pools=ctx.pool_capacities, pool_units=ctx.pool_units,
                    backend=backend, trace_limit=2000)
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
@pytest.mark.parametrize("kind,c,iters,split,aux", [(PRE, 3, 2, 1, 0), (PRE, 4, 20, 3, 2),
                                                   (PRE_SINK, 4, 20, 2, None)])
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
@pytest.mark.parametrize("binding,m,template", [("ez1008", 512, "decomposed_fallback"),
                                                ("ez1008", 513, "native_large"),
                                                ("cann_current", 512, "native_large"),
                                                ("cann_current", 513, "native_large"),
                                                ("cann_current", 0, "empty_noop")])
def test_text_gate_stack_boundary(arch, binding, m, template):
    geom = _geometry(arch)
    op = _op(TEXT, (m, 4, 256), rank=64, has_gate=True, stack_binding=binding)
    plan = hc.build(op, geom)
    assert plan.metadata["selected_template"] == template
    stages = _stages(plan, op, geom)
    if not m:
        assert not stages and not plan.work.resources and not plan.work.matrix
        return
    decomposed = template == "decomposed_fallback"
    assert plan.metadata["excluded_fusion"] == ("ez1008_m_le_512" if decomposed else None)
    assert ("decomposed_gate_write" if decomposed else "residual_add_and_gated_reduction") in stages
    assert plan.work.matrix == {"bf16": 2 * m * (4 * 256) * (2 * 64 + 4)}

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

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("physical_cap", [None, 5, 999])
def test_attnres_exact_resident_cutoff(arch, fuse, physical_cap):
    h, m, ub = 1024, 3, MIN_UB
    row = 2 * h
    ub_cap = (ub - (16 * h + (2 + fuse) * row + 16384)) // (row + 4)
    limit = ub_cap if physical_cap is None else min(ub_cap, physical_cap)
    assert hc.attnres_max_resident_rows(h, ub, fuse_add=fuse, physical_capacity_cap=physical_cap) == limit
    for j, cached, template in ((limit - 1, 0, "resident"), (limit, 0, "reload"),
                                (limit, 2, "cooperative_reload")):
        op = _op(ATTN, (m, h), capacity=limit + 2, valid_history=j, fuse_add=fuse,
                 cached_rows=cached, physical_capacity_cap=physical_cap)
        plan = hc.build(op, _geometry(arch, ub=ub))
        assert plan.metadata["resident_row_limit"] == limit
        assert plan.metadata["selected_template"] == template
        assert plan.metadata["reread_bytes"] == (0 if j + 1 == limit else row * m * (j - cached))

@pytest.mark.parametrize("cached", [0, 1])
def test_attnres_reload_uses_shared_reuse_key_and_only_misses_reach_hbm(monkeypatch, cached):
    witnessed = []
    bind = RecipeContext.bind_access

    def record(self, pattern, *args, **kwargs):
        bound = bind(self, pattern, *args, **kwargs)
        if pattern.reuse_key is not None:
            witnessed.append((pattern, bound.ledger))
        return bound

    monkeypatch.setattr(RecipeContext, "bind_access", record)
    m, h, j = 5, 128, 3
    op = _op(ATTN, (m, h), capacity=6, valid_history=j, cached_rows=cached, physical_capacity_cap=j)
    first, second = 2 * m * h * j, 2 * m * h * (j - cached)
    totals = []
    for l2 in (0, 2**20):
        witnessed.clear()
        geom = _geometry(aiv=3, l2=l2)
        plan = hc.build(op, geom)
        ctx = _context(op, geom)
        program = plan.program(ctx)
        passes = [tuple(stage for stage in program.stages if stage.name.startswith(f"history_pass{n}"))
                  for n in (1, 2)]
        for group, size in zip(passes, (first, second), strict=True):
            assert [stage.placement["mte2_aiv"] for stage in group] == [(0,), (1,), (2,)]
            assert sum(stage.metadata["logical_bytes"] for stage in group) == size
        assert len({pattern.reuse_key for pattern, _ in witnessed}) == 1
        assert [pattern.runs.runs[0].base_byte for pattern, _ in witnessed] == [
            2 * h * (6 * row + offset) for offset in (0, cached) for row in (0, 2, 4)]
        vector = {s.name: s.work["vector"] for s in program.stages if "vector" in s.work}
        assert sum(vector[f"score_softmax_{i}"] for i in range(3)) == m * ((j + 1) * (6 * h + 6) - 1 + 4 * h)
        assert sum(vector[f"weighted_sum_{i}"] for i in range(3)) == m * 2 * (j + 1) * h
        assert sum(ledger.dram_bus_transaction_bytes for _, ledger in witnessed[:3]) == first
        assert sum(ledger.l2_hit_bytes for _, ledger in witnessed[3:]) == (second if l2 else 0)
        engine, _ = _run(program, ctx)
        assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
        totals.append(engine.serviced["hbm"])
    assert totals[0] - totals[1] == pytest.approx(second)

@pytest.mark.parametrize("arch", ARCH_IDS)
@pytest.mark.parametrize("mask,chunk", [((1, 1, 1), 256), ((1, 0, 1), 64), ((0, 0, 0), 64)])
def test_engram_reductions_gate_and_mask(arch, mask, chunk):
    m, h, active = len(mask), 256, sum(mask)
    op, geom = _op(ENGRAM, (m, h), active_mask=mask, chunk_dim=chunk), _geometry(arch)
    plan = hc.build(op, geom)
    stages = _stages(plan, op, geom)
    assert plan.metadata["active_tokens"] == active and plan.metadata["sinkhorn_ops"] == 0
    sums = dict.fromkeys(("x_square_sum", "key_square_sum", "dot_sum"), active * (h - 1))
    assert plan.metadata["reduction_adds"] == sum(sums.values())
    if active:
        reduce, gate = stages["hidden_slice_partial_reductions"], stages["rms_signed_root_sigmoid_mask"]
        assert reduce.work["vector"] == sum(sums.values()) + active * h * (2 + 3)
        # Inverse RMS, scaled dot with 1/sqrt(H), signed root, sigmoid, mask.
        assert gate.work["vector"] == active * (2 * 4 + 3 + 4 + 4 + 1)
        assert gate.after == (reduce.name,)
        assert stages["engram_value_and_spilled_x"].metadata["logical_bytes"] == 2 * active * h * (1 + (chunk < h))
        assert stages["engram_gated_add"].work["vector"] == 2 * active * h
    else:
        assert not plan.work.resources.get("vector")
    if active < m:
        assert stages["engram_inactive_input"].after == ("token_mask",)
        assert stages["engram_inactive_output"].metadata["logical_bytes"] == 2 * (m - active) * h

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

def _source_op(m, h, profile):
    pre = profile in (source.PRE_PROFILE, panel_source.PRE_PROFILE, panel_source.COMPACT_PRE_PROFILE)
    return _op(PRE if pre else POST, (m, 4, h),
               fusion_tensor_view=dict(tokens=m, hidden=h, hc=4, dtype="bf16", sinkhorn_iters=3),
               fusion_boundary={"traffic_profile": profile}, fusion_external_outputs=(),
               fusion_semantic_ids=(), fusion_phases=())

@pytest.mark.parametrize("m", [1, 41, 1025])
def test_shared_projection_gram_and_owner_tail_service(m):
    h, c, cores = 7168, 4, 20
    geom = _geometry(MEMBASE, aic=cores)
    op = _source_op(m, h, source.PRE_PROFILE)
    plan = source.selected_plan(op, geom)
    width = c * h
    m_parts = ceil(m / 256)
    m_span = 16 * ceil(m / m_parts / 16)
    k_span = 256 * ceil(width / (cores // m_parts) / 256)
    k_parts = ceil(width / k_span)
    rows = [min(m_span, m - start) for start in range(0, m, m_span)]
    padded = sum(16 * ceil(count / 16) for count in rows)
    assert plan.work.matrix == {"hf32": 2 * padded * width * 24, "fp32": 2 * padded * width * 16}
    assert plan.work.resources["mte2_aic"] == (4 * m + 24 * len(rows) * 4) * width
    assert plan.work.resources["fixpipe"] == m * k_parts * 40 * 4
    ctx = _context(plan.work.apply(op), geom)
    engine, duration = _run(plan.lower(ctx), ctx)
    assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
    owners = {owner for task in engine.trace if ":coefficients" in task["task"]
              for owner in task["placement"].get("vector", ())}
    n_owners = ceil(m / ceil(m / (2 * cores)))
    assert owners == set(range(n_owners))
    assert source.pre_phase_work(dict(op.metadata)["fusion_tensor_view"], geom)["coefficients"].resources["mte2_aiv"] == (
        n_owners * 24 * 4 + m * (k_parts * 40 + 3) * 4)
    for dtype in plan.work.matrix:
        slow = replace(ctx, compute_peaks={**PEAKS, dtype: PEAKS[dtype] / 10})
        assert _run(plan.lower(slow), slow)[1] > duration
    without = replace(geom, resource_units={k: v for k, v in geom.resource_units.items() if k != "fixpipe"})
    with pytest.raises(ModelingError, match="requires_fixpipe_output"):
        source.selected_plan(op, without)

@pytest.mark.parametrize("m,h,planner,store_span", [
    (1, 2048, source, 1), (41, 2112, source, 3), (43, 2112, source, 4), (3, 4128, panel_source, 2),
])
def test_post_tail_bytes_private_reads_ub_and_shared_store(m, h, planner, store_span):
    panel = planner is panel_source
    geom = _geometry(REGBASE if panel else MEMBASE, aic=20, ub=POOLS["ub"])
    geom = replace(geom, resource_units={**geom.resource_units, "mte3": 20})
    op = _source_op(m, h, planner.POST_PROFILE)
    plan = planner.selected_plan(op, geom)
    read, write = m * (10 * h + 80), m * 8 * h
    assert plan.work.resources["mte2_aiv"] == read and plan.work.resources["mte3"] == write
    ub = 213152 if panel else 164640
    vector_per_row = 56 * h + 4 * 4096 * ceil(h / 4096) if panel else h * (4 * 9 + 9)
    assert plan.work.resources["vector"] == m * vector_per_row
    prepared = plan.work.apply(op)
    ctx = _context(prepared, geom, rates={**dict.fromkeys(geom.resource_units, 1e14), "mte3": 20 * 8 * h})
    engine, duration = _run(plan.lower(ctx), ctx)
    assert duration == pytest.approx(store_span, abs=1e-6)
    assert _serviced(engine) == pytest.approx(ctx.bind(plan.work))
    assert set(engine.summary()["peak_allocated_per_unit"].values()) == {ub}
    with pytest.raises(ModelingError, match="capacity"):
        planner.selected_plan(op, replace(geom, buffers={**geom.buffers, "ub": ub - 1}))
    if m == 1:
        alias = _context(prepared, geom, modes={"mte2_aiv": ("mte2_aic", 1e11)})
        alias = replace(alias, resource_units={**alias.resource_units, "mte2_aic": 40})
        aliased, _ = _run(plan.lower(alias), alias)
        assert aliased.serviced["mte2_aic"] == read and not aliased.serviced["mte2_aiv"]
        assert {task["placement"]["mte2_aic"] for task in aliased.trace
                if "mte2_aic" in task["placement"]} == {(0,)}

@pytest.mark.parametrize("m", [257, 512, 513, 4097])
def test_panel_tiling_compact_equivalence_and_repeated_layer_ir(m):
    h, cores = 1024, 4
    geom = _geometry(aic=cores, ub=POOLS["ub"])
    op = _source_op(m, h, panel_source.PRE_PROFILE)
    plan = panel_source.selected_plan(op, geom)
    compact = panel_source.selected_plan(_source_op(m, h, panel_source.COMPACT_PRE_PROFILE), geom)
    mk = compact.metadata["m_k_tiling"]
    md = min(cores, ceil(m / 256))
    kd = cores // md
    tile = min(256, 16 * ceil(m / md / 16))
    assert (mk["m_dim"], mk["k_dim"], mk["m_l1"], mk["tiling_key"]) == (md, kd, tile, 1001 if kd == 1 else 1000)
    assert plan.work.matrix["hf32"] == 2 * sum(16 * ceil(min(tile, m - i) / 16)
                                                for i in range(0, m, tile)) * 32 * (4 * h)
    assert plan.work == compact.work
    assert panel_source.selected_plan(replace(op, op_id="next_layer"), geom).work is plan.work
    ctx = _context(plan.work.apply(op), geom)
    detail, packed = plan.program(ctx), compact.program(ctx)
    assert packed.stages[0] is not compact.program(ctx).stages[0]
    if kd == 1:
        assert [s.name for s in packed.stages] == [s.name for s in detail.stages] == ["source_msplit"]
        first = packed.stages[0].program.programs[0].stages
        assert first == detail.stages[0].program.programs[0].stages
        assert any("projection_local" in s.name for s in first)
        assert any(s.name.startswith("b") and s.work.get("mte3") for s in first)
    else:
        assert packed.stages[1] == detail.stages[1] and packed.stages[1].after == ("source_stage1",)
    with pytest.raises(ModelingError, match="capacity"):
        panel_source.selected_plan(op, replace(geom, buffers={**geom.buffers, "l1": POOLS["l1"] - 1}))

@pytest.mark.parametrize("slow", ["vector", "mte3", "mte1", "mte2_aiv", "hbm"])
def test_compact_panel_slots_backpressure_and_terminal_service(slow):
    geom = _geometry(aic=32, ub=POOLS["ub"])
    op = _source_op(128, 32768, panel_source.COMPACT_PRE_PROFILE)
    g = pre_geometry(dict(op.metadata)["fusion_tensor_view"], geom)
    phases = pre_owner(g, 0)
    ctx = _context(op, geom, rates={slow: 1e6})
    exact = first_owner(g, phases, ctx)
    assert exact.stages == phase_program(phases, ctx).stages
    links = (("p1.v0.c0.read", "p0.v0.c0.cast_rms"),
             ("p1.v0.c0.transpose", "p0.v0.c0.publish_l1"),
             ("p2.v0.c0.publish_l1", "p0.k3.operands"), ("p2.weight", "p0.k3.operands"),
             ("p0.k0.operands", "p0.v1.c1.publish_l1"), ("p0.k2.operands", "p0.k0.cube"))
    assert "p0.k3.cube" not in next(s.after for s in exact.stages if s.name == "p2.weight")
    template = finite_owner_template(g, phases, ctx)
    assert template.phase_objects <= 2 and template.stage_objects <= 110
    assert template.tiles == 8 and template.dynamic_instances == 16
    assert {s.name for s in template.drain} == {"projection_out", "rms_out0", "rms_out1"}
    baseline, slow_span = _run(exact, ctx)
    fast = replace(ctx, rates={**ctx.rates, slow: 1e8})
    quicker, fast_span = _run(first_owner(g, phases, fast), fast)
    assert slow_span > fast_span and quicker.serviced == pytest.approx(baseline.serviced)
    competitor = ("mte2_aiv", 16384, 0, 0.001)
    reference, duration = _run(exact, ctx, blocker=competitor)
    finite, elapsed = _run(PipelineTemplateProgram(template), ctx, blocker=competitor)
    assert finite.serviced == pytest.approx(reference.serviced)
    assert finite.serviced["mte2_aiv"] == pytest.approx(baseline.serviced["mte2_aiv"] + competitor[1])
    assert elapsed == pytest.approx(duration, rel=0.002)
    trace = {row["task"].removeprefix("point:"): row for row in baseline.trace}
    assert all(trace[child]["start_ms"] >= trace[parent]["end_ms"] for child, parent in links)
    last = template.dynamic_instances - 1
    assert trace["rms_out0"]["start_ms"] >= trace[f"p{last}.v0.c1.publish_l1"]["end_ms"]
    assert trace["rms_out0"]["end_ms"] < trace["projection_out"]["end_ms"]
    done = exact.start(baseline, "repeat", baseline.submit("ready_again"))
    baseline.advance(until=(done,))
    assert done.time == pytest.approx(7 + 2 * slow_span)

def _point(case, tokens, valid, **options):
    params = dict(prefix_id="prefix", addend_id=None, bank_ids=tuple(map(str, range(8))),
                  valid_blocks=valid, write_slot=None, output_norm=True, mix=True, materialize=False,
                  tokens_local=tokens, bank_capacity=8, block_token_stride=8 * case.model.hidden_size)
    return prefix.attn_res_point({"prefix": "point", "stage": 0, "layer": 0, "model": case.model,
                                 "tokens": tokens, "mode": "prefill"}, **{**params, **options}).ops[0]

def test_prefix_owner_arithmetic_gamma_cache_and_bypass():
    case = parse_case(_raw("source-families/attnres-prefix-control.json"))
    case = replace(case, model=replace(case.model, hidden_size=64))
    caps = prefix.capabilities()
    assert tuple(caps[key] for key in ("soc_families", "auto_phases", "dtypes")) == (("A5",), ("decode",), ("bf16",))
    op = _point(case, 1, 1)
    for aiv in (2, 64):
        geom = _geometry(aiv=aiv)
        plan = prefix.build(op, geom)
        ctx = _context(op, geom, rates={"vector": 1e6, "mte2_aiv": 1e16, "mte3": 1e16, "hbm": 1e16})
        engine, elapsed = _run(plan.lower(ctx), ctx)
        assert engine.serviced["mte2_aiv"] == 640 and not engine.serviced.get("mte2_aic")
        assert elapsed == pytest.approx(plan.work.resources["vector"] * aiv / 1e6)
        assert plan.work.resources["vector"] >= 2 * (7 * 64 - 2) and not plan.work.matrix
    cached = _point(case, 129, 0, optimize_prefill=True)
    metrics = prefix.point_metrics(dict(cached.metadata)["attn_res_view"], MIN_UB, 2)
    assert metrics["view"]["core_ranges"] == ((0, 65), (65, 64))
    assert metrics["view"]["output_norm_cached"] and metrics["resources"]["mte2"] == 2 * 64 * (129 + 2)
    bypass = _point(case, 5, 5, mix=False, output_norm=False)
    assert bypass.deps == ("prefix",)
    assert prefix.build(bypass, _geometry()).work.resources["mte2_aiv"] == 2 * 5 * 64

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
                        ready="input", tokens=5, mode="prefill", stages=(0, 0, 1))
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
        "implementation": "declared_native_fusion", "options": {"roles": [PRE], "traffic_profile": source.PRE_PROFILE}}
    raw["model"]["layers"] = 1
    before, case = json.dumps(raw), parse_case(raw)
    options = dict(phase=phase, tokens=8, scope="body")
    semantic = target_operators(case, resource_fusion=False, **options)
    _, members = next((d, ops) for d, ops in _groups(semantic) if d["role"] == PRE)
    before_members = tuple(op.to_dict() for op in members)
    groups = export_structure(case, **options)["subgraphs"]
    work = next(group["physical_work"] for group in groups if group["role"] == PRE)
    geom = Geometry.from_devices(case.machine.dies)
    plan = source.selected_plan(lower_native_fusions(members, case.runtime)[0], geom)
    assert work["launches"] == 1 and work["matrix_flops"] == plan.work.matrix
    assert (work["hbm_bytes"], work["vector_ops"]) == (plan.work.resources["hbm"], plan.work.resources["vector"])
    assert json.dumps(raw) == before
    consumer = replace(_op("consume", (8, 64)), deps=(members[-1].op_id,))
    fused, output = lower_native_fusions((*members, consumer), case.runtime)
    assert output.deps == (fused.op_id,)
    affine = next(op for op in members if op.recipe == "hc_normalize_scale_bias_gate")
    fused = lower_native_fusions((*members, replace(consumer, deps=(affine.op_id,))), case.runtime)[0]
    with pytest.raises(ModelingError, match="cannot_export_internal_intermediate"):
        source.selected_plan(fused, geom)
    assert tuple(op.to_dict() for op in members) == before_members
