"""Differential locks for the seven Lean specification modules.

[Role] Audit the formal kernel and compare its theorem domains with live simulation.
[Invariants] The source audit allows no proof escapes. Seeded oracles use exact
  arithmetic where the formal contracts do.
[Timeouts & Locks] Seeded trials and a bounded Lean check when available.
[Side-Effect Policy] Analytical replay and a temporary Lean parity fixture.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
from os import environ
from pathlib import Path
import random
import re
from shutil import which
from statistics import median
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.api import optimize_ladder
from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS as POOL_LINEAR_KINDS
from mindie_sim.catalog.expert_slots import expert_slot_plan, routed_expert_weight_terms
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.projection_placement import (
    ROLE_ATTENTION_OUTPUT, ROLE_LINEAR_INPUT, ROLE_LOGITS, placed_weight_shape,
)
from mindie_sim.catalog.speculative_state import (
    LINEAR_ATTENTION_KINDS as SPEC_LINEAR_KINDS, MODE_CHECKPOINT, MODE_REPLAY, MODE_SINGLE,
    record_head_bytes, speculative_state_plan,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _sharded_shape, term_bytes
from mindie_sim.compose import compose_case
from mindie_sim.optimization import (
    BilinearContractionSpec, DagEdge, EventDagSpec,
    FarkasCertificate, OptimalityCertificate, ResourceChoiceProgram, TensorRangeObligation,
    build_dual_inputs_from_scenario, certify_lower_bound, check_farkas_dual,
    check_optimality_certificate, eval_checked_path,
    solve_checked_critical_path, solve_farkas_dual,
)
from mindie_sim.prediction.runtime.reload_bounds import (
    build_surplus_coupling_sites, direct_mac_pebble_reload_bound, verify_epoch_trace,
)
from mindie_sim.prediction.collectives.startup import resolve_collective_startup, validate_steps
from mindie_sim.prediction.efficiency import extract_hardware_peaks
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import physical_slot_work
from mindie_sim.prediction.efficiency.exposed_breakdown import (
    ExposedClocks, advance_clocks, finish_breakdown, join_clocks as prod_join_clocks, roofline_clocks,
)
from mindie_sim.prediction.fusion.graph import OpNode, TensorValue, TypedGraph
from mindie_sim.prediction.sharding.local_tp import lower_local_projection_tp
from mindie_sim.prediction.operators.recurrent.recover_ssm_commit import bind_recover_ssm_commits
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import FP32_RECORD_BYTES, emit_commit_op, gdn_record_head_bytes
from mindie_sim.prediction.efficiency.roofline_graph import (
    PhysicalWorkGraph, PhysicalWorkNode, RooflineTargets, critical_path_roofline,
)
from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding, AllocationKey, BackingKey, EventKey, OccurrenceKey, PoolKey,
)
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.runtime.ledger import Ledger, normalise_ranges, ranges_bytes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision, PrefetchPhysics, apply_to_work, plan_prefetch
from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, ScenarioRequest, finalize_plan, prepare_scenario
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from scripts.check_lean_no_sorry import FORBIDDEN_RE, strip_lean_comments_and_strings

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_DIR = REPO_ROOT / "formal"
EXPECTED_LEAN_MODULES = (
    "FourClock.lean", "PrefetchAndLedger.lean", "ShardingAndLayout.lean",
    "RecoverSSM.lean", "AllToAllRounds.lean", "RooflineAndMesh.lean", "OptimalityCertificate.lean",
)
SEED = 20261115
TRIALS = 250
ELAN_HOME = Path("/tmp/elan-mindie-s3b")
ELAN_LEAN = ELAN_HOME / "bin" / "lean"
LEAN = which("lean") or (str(ELAN_LEAN) if ELAN_LEAN.is_file() else None)

def test_formal_kernel_structure_and_zero_sorry_audit() -> None:
    umbrella, axiom_check = (strip_lean_comments_and_strings((FORMAL_DIR / name).read_text())
                             for name in ("MindIESimSpec.lean", "AxiomCheck.lean"))
    assert "import MindIESimSpec" in axiom_check
    for name in ("physical_ideal_le_predicted", "kMesh_capacity_partition"):
        assert f"#print axioms MindIESimSpec.RooflineAndMesh.{name}\n" in axiom_check
    for name in ("farkas_dual_le_any_feasible_schedule", "epoch_bilinear_pair_cap", "capacity_reload_trace_bound",
                 "max_plus_critical_path_bound", "certified_ladder_chain"):
        assert f"#print axioms MindIESimSpec.OptimalityCertificate.{name}\n" in axiom_check

    theorem_re = re.compile(r"\btheorem\s+[A-Za-z0-9_']+")
    spec_dir = FORMAL_DIR / "MindIESimSpec"
    assert sorted(p.name for p in spec_dir.glob("*.lean")) == sorted(EXPECTED_LEAN_MODULES)

    for mod_name in EXPECTED_LEAN_MODULES:
        stem = mod_name[:-5]
        assert f"import MindIESimSpec.{stem}" in umbrella
        assert f"#print axioms MindIESimSpec.{stem}." in axiom_check
        cleaned = strip_lean_comments_and_strings((spec_dir / mod_name).read_text())
        assert not FORBIDDEN_RE.search(cleaned), f"forbidden token in {mod_name}"
        count = len(theorem_re.findall(cleaned))
        assert count >= 4, f"expected >=4 theorems in {mod_name}, found {count}"

    kernel = strip_lean_comments_and_strings((spec_dir / "OptimalityCertificate.lean").read_text())
    for name in ("checkDagAcyclic", "evalCheckedPath"):
        assert re.search(rf"\bdef {name}\b", kernel)
    for name in ("max_plus_critical_path_bound", "certified_ladder_chain"):
        assert re.search(rf"\btheorem {name}\b", kernel)
    for lean_path in sorted(FORMAL_DIR.glob("*.lean")):
        cleaned = strip_lean_comments_and_strings(lean_path.read_text(encoding="utf-8"))
        assert not FORBIDDEN_RE.search(cleaned), f"forbidden token in {lean_path}"

def test_four_clock_max_plus_conservation_and_order_invariance() -> None:
    def _rc(c: float = 0.0, v: float = 0.0, m: float = 0.0, n: float = 0.0) -> ExposedClocks:
        return roofline_clocks(cube_ms=c, vector_ms=v, memory_ms=m, communication_ms=n)

    branch_a = advance_clocks([_rc(c=2.0)], _rc(n=8.0))
    branch_b = advance_clocks([_rc(c=5.0)], _rc(m=3.0))
    assert branch_a.all_ms > branch_b.all_ms
    assert branch_b.without_communication_ms > branch_a.without_communication_ms
    joined = prod_join_clocks([branch_a, branch_b])
    rep = finish_breakdown(joined, duration_ms=joined.all_ms, boundary="device_step")
    ms = rep["exposed_breakdown_ms"]
    assert (ms["cube"], ms["vector"], ms["memory"], ms["communication"]) == (5.0, 0.0, 3.0, 2.0)

    rng = random.Random(SEED)
    for _ in range(TRIALS):
        c_hid = float(rng.randint(0, 500))
        m_hid = float(rng.randint(0, int(c_hid)))
        hidden = finish_breakdown(
            _rc(c=c_hid, m=m_hid), duration_ms=c_hid, boundary="device_step",
        )["exposed_breakdown_ms"]
        assert (hidden["cube"], hidden["vector"], hidden["memory"], hidden["communication"]) == (
            c_hid, 0.0, 0.0, 0.0,
        )
        durs = [
            roofline_clocks(
                cube_ms=float(rng.randint(0, 80)), vector_ms=float(rng.randint(0, 80)),
                memory_ms=float(rng.randint(0, 80)), communication_ms=float(rng.randint(0, 80)),
            )
            for _ in range(12)
        ]
        clocks_a: list[ExposedClocks] = []
        clocks_b: list[ExposedClocks] = []
        for node in range(12):
            preds = rng.sample(range(node), rng.randint(1, min(4, node))) if node else []
            perm = list(preds)
            rng.shuffle(perm)
            ca = advance_clocks([clocks_a[p] for p in preds], durs[node])
            cb = advance_clocks([clocks_b[p] for p in perm], durs[node])
            assert ca == cb
            assert 0.0 <= ca.cube_ms <= ca.compute_ms <= ca.without_communication_ms <= ca.all_ms
            clocks_a.append(ca)
            clocks_b.append(cb)
        terminal = prod_join_clocks(clocks_a)
        done = finish_breakdown(terminal, duration_ms=terminal.all_ms, boundary="device_step")
        buckets = done["exposed_breakdown_ms"]
        assert all(v >= 0.0 for v in buckets.values())
        assert sum(buckets.values()) == pytest.approx(terminal.all_ms, abs=1e-9)
        if terminal.all_ms > 0.0:
            assert abs(math.fsum(done["exposed_breakdown_pct"].values()) - 100.0) <= 1e-9

def test_prefetch_byte_conservation_and_half_open_interval_ledger() -> None:
    state = SimpleNamespace(uid="rt-lock", capacity_domains={})
    pool = PoolKey(physical_domain="dom0", rank=0, tier="hbm", name="workspace")
    Ledger.ensure_domain(state, "dom0", capacities={pool: 512.0})
    shared_backing = BackingKey(physical_domain="dom0", rank=0, tier="hbm", storage_id="buf0", generation=0)
    occ = OccurrenceKey(runtime_uid="rt-lock", launch=0)
    k1 = AllocationKey(occurrence=occ, scope_kind="op", scope_ordinal=0, symbol=1, generation=0)
    k2 = AllocationKey(occurrence=occ, scope_kind="op", scope_ordinal=1, symbol=2, generation=0)
    ev1 = SimpleNamespace(key=EventKey(runtime_uid="rt-lock", sequence=1), done=True, time=1.0)
    ev2 = SimpleNamespace(key=EventKey(runtime_uid="rt-lock", sequence=2), done=True, time=2.0)

    g1 = Ledger.try_acquire(
        state, (AllocationBinding(key=k1, pool=pool, amount=64.0, backing=shared_backing, byte_ranges=((0, 64),)),),
        holder="op1", at=ev1,
    )
    assert g1.granted and Ledger.usage(state)["dom0"][pool] == 64.0 and Ledger.peaks(state)["dom0"][pool] == 64.0
    rel1 = Ledger.release(state, k1, after=ev1)
    assert rel1.released and Ledger.usage(state)["dom0"][pool] == 0.0

    g2 = Ledger.try_acquire(
        state, (AllocationBinding(key=k2, pool=pool, amount=96.0, backing=shared_backing, byte_ranges=((0, 96),)),),
        holder="op2", at=ev2,
    )
    assert g2.granted and Ledger.usage(state)["dom0"][pool] == 96.0
    assert Ledger.peaks(state)["dom0"][pool] == 96.0  # Disjoint reuse: peak is 96B, never 160B.
    assert Ledger.release(state, k2, after=ev2).released and Ledger.usage(state)["dom0"][pool] == 0.0

    k_static = AllocationKey(occurrence=occ, scope_kind="static", scope_ordinal=0, symbol=3, generation=0)
    k_tr = AllocationKey(occurrence=occ, scope_kind="op", scope_ordinal=2, symbol=4, generation=0)
    b_static = BackingKey(physical_domain="dom0", rank=0, tier="hbm", storage_id="static0", generation=0)
    assert Ledger.try_acquire(
        state,
        (
            AllocationBinding(key=k_static, pool=pool, amount=128.0, backing=b_static, byte_ranges=((0, 128),)),
            AllocationBinding(key=k_tr, pool=pool, amount=64.0, backing=shared_backing, byte_ranges=((0, 64),)),
        ),
        holder="step", at=ev2,
    ).granted
    assert Ledger.usage(state)["dom0"][pool] == 192.0 and Ledger.peaks(state)["dom0"][pool] == 192.0
    assert Ledger.release(state, k_tr, after=ev2).released
    assert Ledger.usage(state)["dom0"][pool] == 128.0
    assert ranges_bytes(normalise_ranges([(0, 128), (0, 64)])) == 128

    graph = TypedGraph(
        rows=16,
        values={
            "x": TensorValue("x", "bf16", (16, 128), role="activation"),
            "w": TensorValue("w", "bf16", (128, 128), role="weight", sharded=True),
            "y": TensorValue("y", "bf16", (16, 128), role="output", producer="gemm"),
        },
        ops={
            "gemm": OpNode(
                op_id="gemm", kind="matmul", resource="cube",
                inputs=("x", "w"), outputs=("y",), work=(("hbm", 32768.0),),
            ),
        },
    )
    physics = PrefetchPhysics(hbm_bytes_per_ms=1e7, onchip_bytes_per_ms=4e7, capacity_bytes=32768, alignment=256)
    prev_prefetched = -1
    for free_cap in (0, 4096, 16384, 32768):
        dec = plan_prefetch(
            graph, "gemm", window_ms=1.0, foreground_hbm_bytes=0.0,
            physics=physics, capacity_bytes=32768, resident_bytes=32768 - free_cap,
        )
        assert dec.conserved and dec.bytes_prefetched + dec.hbm_read_bytes == dec.weight_bytes == 32768
        assert 0 <= dec.bytes_prefetched <= free_cap
        assert dec.bytes_prefetched >= prev_prefetched
        prev_prefetched = dec.bytes_prefetched
        if dec.allowed:
            updated = dict(apply_to_work((("hbm", 32768.0),), dec))
            assert updated["hbm"] + updated["mte1"] == pytest.approx(32768.0)

    rng = random.Random(SEED + 1)
    for _ in range(TRIALS):
        w = rng.randint(0, 1_000_000)
        c_free = rng.randint(-200_000, 1_200_000)
        p = 0 if c_free <= 0 else min(w, c_free)
        dec = PrefetchDecision(
            "op", ("w0",), w, p, 0, w, 0, 1.0, 1e9, 0.0, False, "init", (),
            hbm_rate_ms=100.0, onchip_rate_ms=400.0, alignment=1,
        ).settled(p)
        assert dec.conserved and dec.bytes_prefetched + dec.hbm_read_bytes == w

def test_sharding_ceil_div_fractal_nz_and_redundant_expert_slot_growth() -> None:
    def fractal_nz_bytes(d: int, n: int, c0: int, elem: int) -> int:
        return _ceil_div(d, c0) * _ceil_div(n, 16) * 16 * c0 * elem

    q_bf16 = QuantSpec("bf16")
    par_local = ParallelSpec(dp=4, tp=1, ep=4, o_proj_tp=4, linear_proj_tp=2, lm_head_tp=2)
    machine = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=4, dp=1, ep=1, layers=1)).machine
    software = Software(parallel=par_local)
    for role, category, shard, rows, cols in (
        (ROLE_ATTENTION_OUTPUT, "mla.o", "in_tp", 128, 256),
        (ROLE_LINEAR_INPUT, "kda.q", "out_tp", 256, 128),
        (ROLE_LOGITS, "lm_head", "out_tp", 256, 128),
    ):
        term = WeightTerm(category, rows=rows, cols=cols, category=category, shard=shard, quant=q_bf16, placement_role=role)
        sh_r, sh_c, _ = _sharded_shape(term, par_local, node_ranks=4)
        pl_r, pl_c, _ = placed_weight_shape(term, par_local, node_ranks=4)
        assert (sh_r, sh_c) == (pl_r, pl_c)
        ledger_bytes = term_bytes(term, q_bf16, par_local, node_ranks=4)
        assert ledger_bytes == pytest.approx(fractal_nz_bytes(sh_r, sh_c, 16, 2))
        single_tok_op = gemm_op(
            op_id=f"op_{category}", deps=("x",), stage=0, layer=0, requests=("r0",),
            m=1, n=rows, k=cols, quant=q_bf16, category=category, phase="decode",
        )
        lowered = lower_local_projection_tp(
            (single_tok_op,), model=None, runtime=software, machine=machine,
            replica=0, cohort_views={}, scope="target",
        )
        gemm_node = next(op for op in lowered if op.family in {"gemm", "output_projection", "qkv_projection", "logits", "projection"})
        _m_loc, n_loc, k_loc = gemm_node.local_shape
        frontend_cold_weight_bytes = fractal_nz_bytes(n_loc, k_loc, 16, 2)
        assert (n_loc, k_loc) == (sh_r, sh_c)
        assert ledger_bytes == pytest.approx(frontend_cold_weight_bytes)

    rng = random.Random(SEED + 2)
    for _ in range(TRIALS):
        a, b = rng.randint(0, 8192), rng.randint(1, 128)
        q = _ceil_div(a, b)
        assert a <= q * b < a + b and (a == 0 or q > 0)
        assert _ceil_div(b * (a // b), b) * b == b * (a // b)
        d = rng.randint(1, 512)
        n = rng.randint(1, 16) * 16 + rng.randint(0, 15)
        elem = rng.choice((1, 2))
        c0 = 32 // elem
        assert fractal_nz_bytes(d, n, c0, elem) >= d * n * elem
        d_al, n_al = _ceil_div(d, c0) * c0, _ceil_div(n, 16) * 16
        assert fractal_nz_bytes(d_al, n_al, c0, elem) == d_al * n_al * elem

        ep = rng.choice((2, 4, 8))
        e_routed = rng.randint(ep * 2, ep * 12)
        r_extra = rng.randint(1, max(1, min(3, e_routed - _ceil_div(e_routed, ep))))
        p_base = ParallelSpec(tp=ep, ep=ep, redundant_experts_per_rank=0)
        p_extra = ParallelSpec(tp=ep, ep=ep, redundant_experts_per_rank=r_extra)
        base_plan = expert_slot_plan(None, SimpleNamespace(num_experts=e_routed, index=0), p_base)
        extra_plan = expert_slot_plan(None, SimpleNamespace(num_experts=e_routed, index=0), p_extra)
        gate = WeightTerm("expert_gate", rows=64, cols=32, category="routed_expert",
                          shard="expert_gate_up", count=e_routed, quant=q_bf16)
        base_terms = routed_expert_weight_terms([gate], base_plan, ep_rank=0)
        extra_terms = routed_expert_weight_terms([gate], extra_plan, ep_rank=0)
        base_ledger = sum(term_bytes(t, q_bf16, p_base) for t in base_terms)
        extra_ledger = sum(term_bytes(t, q_bf16, p_extra) for t in extra_terms)
        assert extra_ledger > base_ledger
        incl = {e: 1.0 / e_routed for e in range(e_routed)}
        sat_base = physical_slot_work(slot_plan=base_plan, logical_inclusions=incl, global_tokens=10**9, packed_weight_bytes=64 * 32 * 2)
        sat_extra = physical_slot_work(slot_plan=extra_plan, logical_inclusions=incl, global_tokens=10**9, packed_weight_bytes=64 * 32 * 2)
        assert sat_extra.weight_read_bytes(0) == pytest.approx(extra_ledger)
        assert sat_base.weight_read_bytes(0) == pytest.approx(base_ledger)

def test_recover_ssm_commit_prefix_and_checkpoint_savings() -> None:
    expected_families = frozenset({"kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"})
    assert SPEC_LINEAR_KINDS == expected_families and POOL_LINEAR_KINDS == expected_families

    class _IntentBuilder:
        def __init__(self, layer_idx: int) -> None:
            self.plan = SimpleNamespace(scope="target", index=layer_idx)
            self.ops: list[OpSpec] = []

        def work(self, op_id, deps, shape, *, flops, vector, read, write, family, metadata):
            sh = tuple(shape)
            self.ops.append(OpSpec(
                op_id=op_id, family=family, phase="decode", stage=0, layer=self.plan.index,
                deps=tuple(deps), requests=(), semantic_shape=sh, local_shape=sh,
                padded_shape=sh, launch_shape=sh, flops=float(flops), vector_ops=float(vector),
                hbm_bytes=float(read + write), mte2_bytes=float(read), mte3_bytes=float(write),
                metadata=tuple(metadata),
            ))
            return op_id

    parallel = ParallelSpec(tp=1)
    heads, dk, dv = 16, 128, 128
    state_bytes = heads * dk * dv * FP32_RECORD_BYTES
    for idx, family in enumerate(sorted(expected_families)):
        plan = LayerPlan(
            index=idx, attention=family, ffn="dense", num_heads=heads, num_kv_heads=4,
            head_dim=dv, linear_heads=heads, linear_key_heads=heads, linear_value_heads=heads,
            linear_head_dim=dv, linear_key_head_dim=dk, conv_kernel=4,
            recurrent_state_elements=heads * dk * dv, conv_state_elements=heads * dv * 4,
        )
        single_off = speculative_state_plan(None, plan, parallel, query_tokens=1, recover_ssm=False, capacity_padded=True)
        single_on = speculative_state_plan(None, plan, parallel, query_tokens=1, recover_ssm=True, capacity_padded=True)
        assert single_off == single_on and single_on.mode == MODE_SINGLE and single_on.replay_bytes == 0

        ckpt = speculative_state_plan(None, plan, parallel, query_tokens=5, recover_ssm=False, capacity_padded=True)
        replay = speculative_state_plan(None, plan, parallel, query_tokens=5, recover_ssm=True, capacity_padded=True)
        expected_rec = ((dk + dv + 1) * FP32_RECORD_BYTES if family == "gdn" else dv * FP32_RECORD_BYTES + 2 * dk * 2)
        assert ckpt.mode == MODE_CHECKPOINT and ckpt.state_slots == 5
        assert replay.mode == MODE_REPLAY and replay.state_slots == 1
        assert replay.replay_bytes == pytest.approx(5 * heads * expected_rec)
        assert replay.total_bytes < ckpt.total_bytes
        record_size = gdn_record_head_bytes if family == "gdn" else record_head_bytes
        assert record_size(dk, dv) == expected_rec

        b_zero = _IntentBuilder(idx)
        assert emit_commit_op(
            b_zero, name=f"{family}_zero", deps=("v0",),
            views=(RequestView("r0", 1, 64, False, "decode"), RequestView("pad", 5, 64, False, "decode", padding=True)),
            heads=heads, dk=dk, dv=dv, state_dtype="fp32", attention_kind=family,
        ) is None
        commit_reads: dict[int, float] = {}
        for accepted in (2, 5):
            builder = _IntentBuilder(idx)
            view = RequestView("r0", 5, 64, False, "decode", draft_advanced_tokens=accepted)
            oid = emit_commit_op(
                builder, name=f"{family}_commit_{accepted}", deps=("verify0",),
                views=(view,), heads=heads, dk=dk, dv=dv, state_dtype="fp32", attention_kind=family,
            )
            assert oid is not None
            bound = bind_recover_ssm_commits(
                builder.ops, views=(view,), acceptance_events={"r0": "accept0"},
                state_bindings={f"layer{idx}": "base_state"},
                runtime=SimpleNamespace(mtp_acceptance_length=1.0, metadata=(("recover_ssm_readiness", "shipping"),)),
                machine=SimpleNamespace(arch_family="A5"),
            )
            op = bound[0]
            op_meta = dict(op.metadata)
            assert op_meta["recover_ssm_stage"] == "commit"
            assert op_meta["unique_record_read_bytes"] == pytest.approx(accepted * heads * expected_rec)
            expected_phys_rec = (
                accepted * heads * (op_meta["v_tiles"] * (dk + 1) + dv) * FP32_RECORD_BYTES
                if family == "gdn"
                else accepted * heads * (dv * FP32_RECORD_BYTES + op_meta["v_tiles"] * 2 * dk * 2)
            )
            assert op_meta["physical_tiled_record_read_bytes"] == pytest.approx(expected_phys_rec)
            assert op.mte2_bytes == pytest.approx(state_bytes + expected_phys_rec)
            assert op.mte3_bytes == pytest.approx(state_bytes)
            # Upstream-exact GDN & locked GA-09 KDA reverse accumulation: 2 * (u + 1) * E.
            assert op.vector_ops == pytest.approx(2 * (accepted + 1) * heads * dk * dv)
            commit_reads[accepted] = op.mte2_bytes
        assert commit_reads[2] < commit_reads[5]

def test_alltoall_round_scaling_and_step_bounds_against_collective_startup() -> None:
    log_128 = resolve_collective_startup(family="alltoallv", group_size=128, metadata={}, runtime_mode="log2", topology_protocols={})
    ring_128 = resolve_collective_startup(family="alltoallv", group_size=128, metadata={}, runtime_mode="ring", topology_protocols={})
    assert (log_128.startup_rounds, ring_128.startup_rounds) == (7, 127)
    assert log_128.peer_phases == ring_128.peer_phases == 127

    rng = random.Random(SEED + 4)
    for idx in range(TRIALS):
        g = [1, 2, 3, 4, 64, 128][idx] if idx < 6 else rng.randint(1, 4096)
        log_desc = resolve_collective_startup(family="alltoall", group_size=g, metadata={}, runtime_mode="log2", topology_protocols={})
        ring_desc = resolve_collective_startup(family="alltoall", group_size=g, metadata={}, runtime_mode="ring", topology_protocols={})
        lg = log_desc.startup_rounds
        assert g <= (1 << lg)
        if g <= 1:
            assert lg == 0 and ring_desc.startup_rounds == 0
            with pytest.raises(ModelingError):
                validate_steps(1, g)
        else:
            assert 1 <= lg <= g - 1 and ring_desc.startup_rounds == g - 1
            assert validate_steps(lg, g) == lg and validate_steps(1, g) == 1 and validate_steps(g - 1, g) == g - 1
            for bad in (g, 0, True):
                with pytest.raises(ModelingError):
                    validate_steps(bad, g)
        if g >= 4:
            assert lg < ring_desc.startup_rounds

def test_roofline_critical_path_monotonicity_and_direct_mesh_scaling() -> None:
    rng = random.Random(SEED + 5)
    for _ in range(TRIALS):
        n_nodes = rng.randint(2, 10)
        nodes = tuple(
            PhysicalWorkNode(
                node_id=f"n{idx}",
                cube_ms=float(rng.randint(0, 40)), vector_ms=float(rng.randint(0, 40)),
                hbm_ms=float(rng.randint(0, 40)), link_ms=float(rng.randint(0, 40)),
                predecessors=tuple(f"n{p}" for p in (rng.sample(range(idx), rng.randint(1, min(3, idx))) if idx else ())),
            )
            for idx in range(n_nodes)
        )
        graph = PhysicalWorkGraph(nodes=nodes)
        result = critical_path_roofline(graph, RooflineTargets())
        predicted = sum(max(n.cube_ms, n.vector_ms, n.hbm_ms, n.link_ms) for n in nodes)
        assert 0 <= result.physical_ideal_ms == result.target_duration_ms <= predicted

    for width in (2, 4, 8):
        case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=width, dp=1, ep=1, layers=1)
        parsed = parse_case(case)
        peers = [link for link in parsed.machine.links if link.semantics == "per_pair"]
        assert peers
        b_peer = peers[0].capacity_bytes_per_ms * 1000
        assert extract_hardware_peaks(parsed)["link_peak_bytes_per_s"] == pytest.approx((width - 1) * b_peer)

def test_optimality_certificate_differential_zero_inversions() -> None:
    rng = random.Random(SEED + 6)
    solve_ms = []
    for trial in range(TRIALS):
        dimensions = rng.randint(1, 4)
        def work():
            return tuple(rng.randint(0, 50) for _ in range(dimensions))
        sites = tuple(tuple(work() for _ in range(rng.randint(2, 4))) for _ in range(rng.randint(1, 5)))
        zero = (0,) * dimensions
        program = (ResourceChoiceProgram(dimensions, zero, tuple((zero, *modes) for modes in sites))
                   if trial % 25 == 0 else ResourceChoiceProgram(dimensions, work(), sites))
        farkas, report = solve_farkas_dual(program)
        solve_ms.append(report.solve_ms)
        assert check_farkas_dual(program, farkas) is True
        if trial % 25 == 0:
            assert farkas.lower_bound_ns == report.lp_optimum_ns == 0
        mixtures = []
        for modes in program.site_modes:
            weights = [rng.randint(1, 8) for _ in modes]
            mixtures.append(tuple(Fraction(sum(a * b for a, b in zip(weights, column)), sum(weights))
                                  for column in zip(*modes)))
        mixture_wall_ns = max(map(sum, zip(program.global_work, *mixtures)))
        assert farkas.lower_bound_ns <= mixture_wall_ns, trial

        bits_a, bits_b = rng.choice((4, 8, 16)), rng.choice((4, 8, 16))
        capacity = rng.randint(128, 256)
        a = rng.randint(0, 8 * capacity // bits_a)
        b = (8 * capacity - a * bits_a) // bits_b
        assert 4 * bits_a * bits_b * a * b <= (8 * capacity) ** 2
        spec = BilinearContractionSpec("contract", 0, rng.randint(72, 96), rng.randint(72, 96),
                                       rng.randint(4, 8), bits_a, bits_b, 32, capacity, distinct_operands=True, address_injective_across_k=True)
        reload = direct_mac_pebble_reload_bound(spec)
        assert reload.eligible and reload.epochs > 1
        assert verify_epoch_trace(reload.epoch_accounts, reload.total_pairs) is True
        assert reload.total_pairs == spec.m * spec.n
        assert 0 < reload.min_io_bits <= spec.m * spec.n * (spec.k_red * (bits_a + bits_b) + 32)
        site, = build_surplus_coupling_sites({}, (reload,), alpha_rw=Fraction(1, 2))
        size = Fraction(reload.min_io_bytes)
        assert dict(site.modes) == {
            "mode_surplus_read": (("hbm_bus_r@d0", size), ("hbm_bus_w@d0", size / 2), ("mte2_total@d0", size)),
            "mode_surplus_write": (("hbm_bus_r@d0", size / 2), ("hbm_bus_w@d0", size), ("write_total@d0", size)),
        }

        quanta = tuple(rng.randint(1, 50) for _ in range(rng.randint(3, 9)))
        edges = [DagEdge(i, j, rng.randint(0, 20)) for j in range(1, len(quanta))
                 for i in range(j) if i == j - 1 or rng.random() < 0.4]
        rng.shuffle(edges)
        dag = EventDagSpec(quanta, tuple(edges))
        expected = []
        for j, quantum in enumerate(quanta):
            expected.append(quantum + max((expected[e.src] + e.wire_ns for e in edges if e.dst == j), default=0))
        path, sink, longest = solve_checked_critical_path(dag)
        assert eval_checked_path(dag, path, sink) == longest == max(expected)
        assert eval_checked_path(dag, (*path, len(edges)), sink) is None
        assert eval_checked_path(dag, (path[-1], path[-1]), sink) is None
        cert, _ = certify_lower_bound(program, dag)
        assert check_optimality_certificate(program, dag, cert) is True
        assert cert.l_star_lean_ns == max(farkas.lower_bound_ns, longest)
    assert median(solve_ms) < 5.0

    for kind in ("gdn", "kda"):
        rank_one = direct_mac_pebble_reload_bound(replace(spec, kind=kind))
        assert (rank_one.eligible, rank_one.min_io_bytes) == (False, 0)


def test_edited_collective_schedule_preserves_certified_bound():
    sku = REPRESENTATIVE_SKUS["a5"]
    case = compose_case("qwen3-32b", sku, tp=4, dp=1, layers=1)
    step = StepRequest("prefill", tokens=64, context=64)
    ladder = optimize_ladder(case, step, max_trials=4).ladder
    dossier = prepare_scenario(ScenarioRequest(case, step))
    scenario = dossier.handle.scenario
    collective = next(op for op in scenario.atomic_ops[0].ops if op.family == "allreduce")
    route, = scenario.case.machine.links
    edit = Edit("oneshot", operation=CollectiveTransform(OpRef("atom", collective.op_id), "oneshot", route.name))
    edited = finalize_plan(dossier.handle, replace(dossier.unmerge_plan, edits=(edit,)))
    row = next(row for row in edited.step_eval.operators if row.occurrence.op_id == collective.op_id)
    assert edited.verification.passed and row.implementation == "OneShotCollective"
    assert 0 < ladder.l_star_lean_ns == edited.verification.l_star_ns <= edited.wall_ns
    assert edited.wall_ns != dossier.u_wall_ns

DAG_CASES = (
    (EventDagSpec((10,), (DagEdge(0, 0, 0),)), (0,) * 10, 0, 110, None, False),
    (EventDagSpec((10, 10), (DagEdge(0, 1, 0), DagEdge(1, 0, 0))), (0, 1), 0, 30, None, False),
    (EventDagSpec((10, 10), (DagEdge(0, 1, 0), DagEdge(1, 0, 0))), (0,), 1, 20, 20, False),
    (EventDagSpec((10, 10), (DagEdge(0, 1, 0),)), (0,), 1, 20, 20, True),
)

@pytest.mark.parametrize("dag,path,sink,bound,path_ns,acyclic", DAG_CASES)
def test_certificate_requires_acyclic_graph_and_simple_path(dag, path, sink, bound, path_ns, acyclic):
    program = ResourceChoiceProgram(0, (), ())
    cert = OptimalityCertificate(FarkasCertificate(1, (), (), 0), path, sink, bound)
    assert eval_checked_path(dag, path, sink) == path_ns
    assert check_optimality_certificate(program, dag, cert) is acyclic

@pytest.mark.tier1
def test_lean_certificate_agrees_on_cycles_and_valid_path(tmp_path, lean_binary):
    from subprocess import run
    lean = lean_binary
    declarations = ["open MindIESimSpec.OptimalityCertificate"]
    for i, (dag, path, sink, bound, path_ns, acyclic) in enumerate(DAG_CASES):
        edges = ", ".join(f"⟨{e.src}, {e.dst}, {e.wire_ns}⟩" for e in dag.edges)
        accepted = str(acyclic).lower()
        expected = "none" if path_ns is None else f"some {path_ns}"
        declarations += [
            f"def dag{i} : EventDagSpec := ⟨{list(dag.node_quantum_ns)}, [{edges}]⟩",
            f"example : checkDagAcyclic dag{i} = {accepted} := by decide",
            f"example : evalCheckedPath dag{i} {list(path)} {sink} = {expected} := by decide",
            f"example : checkOptimalityCertificate ⟨0, [], []⟩ dag{i} "
            f"⟨⟨1, [], [], 0⟩, {list(path)}, {sink}, {bound}⟩ = {accepted} := by decide",
        ]
    lean_file = tmp_path / "Parity.lean"
    kernel = FORMAL_DIR / "MindIESimSpec" / "OptimalityCertificate.lean"
    lean_file.write_text("\n".join((kernel.read_text(), *declarations)), encoding="utf-8")
    env = {**environ, "ELAN_HOME": str(ELAN_HOME)} if lean == str(ELAN_LEAN) else None
    checked = run([lean, str(lean_file)], cwd=FORMAL_DIR, env=env, capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stdout + checked.stderr

def test_aliased_contraction_bound_respects_engine_wall():
    n = 16
    footprint = len({i + k for i in range(n) for k in range(n)})
    assert footprint == 2 * n - 1
    dev = Device(rank=0, compute_units=1, memory_bytes=1024, rates=(("hbm", 10**9),),
                 compute_peaks=(("int8", 10**13),))
    read = TensorRangeObligation(0, "x", 0, ((0, footprint),), "hbm_cold", "owner")
    region = SimpleNamespace(op_ids=(), semantic_work=(), read_obligations=(read,), write_obligations=())
    spec = BilinearContractionSpec("aliased", 0, n, n, n, 8, 8, 32, 512, write_free=True)
    for unproved in (spec, replace(spec, distinct_operands=True),
                     replace(spec, address_injective_across_k=True)):
        reload = direct_mac_pebble_reload_bound(unproved)
        assert not reload.eligible and reload.min_io_bytes == 0
        assert build_surplus_coupling_sites({}, (reload,)) == ()
    disjoint = direct_mac_pebble_reload_bound(replace(
        spec, distinct_operands=True, address_injective_across_k=True,
    ))
    assert disjoint.eligible and disjoint.epochs == 1
    assert disjoint.min_io_bits**2 == 4 * n**2 * n**2 * spec.bits_a * spec.bits_b
    assert disjoint.epoch_accounts[0].epoch_pair_cap == n**2
    program, dag, _ = build_dual_inputs_from_scenario(
        SimpleNamespace(machine=SimpleNamespace(dies=(dev,), links=()), bilinear_contractions=(spec,)),
        region_summaries=(region,),
    )
    cert, _ = certify_lower_bound(program, dag)
    eng = Engine((Resource("hbm", 10**6), Resource("cube", 10**10)), backend="python")
    loaded = eng.submit("read_x", {"hbm": 32})
    done = eng.submit("contract", {"cube": 2 * n**3}, after=(loaded,))
    eng.advance(until=(done,))
    wall_ns = eng.now * 10**6
    assert wall_ns == pytest.approx(32.8192)
    assert 0 < cert.l_star_lean_ns <= wall_ns
