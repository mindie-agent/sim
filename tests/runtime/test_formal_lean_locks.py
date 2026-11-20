"""Tier-0 live correspondence locks for all 6 Lean 4 formal specification modules.

[Role]
  Audit the Lean 4 formal verification kernel (`formal/MindIESimSpec/*.lean`
  and `formal/AxiomCheck.lean`) for 6-module coverage and zero escape tokens
  (`sorry`, `sorryAx`, `admit`, custom `axiom`), and differentially lock all
  six formal theorem modules directly against live `mindie_sim` runtime,
  catalog, collective, roofline, and interconnect modules.

[Invariants]
  * Pure in-memory Tier-0 execution (`< 2.0s`); never spawns `lake`, never
    mutates host env vars, and uses zero mocks.
  * Covers all 6 `.lean` modules: `FourClock`, `PrefetchAndLedger`,
    `ShardingAndLayout`, `RecoverSSM` (all 7 linear families),
    `AllToAllRounds`, and `RooflineAndMesh`.

[Timeouts & Locks]
  Deterministic seeded PRNG; zero locks.

[Side-Effect Policy]
  Read-only inspection of `formal/` files and pure in-memory module calls;
  zero filesystem or environment mutation.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import re
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS as POOL_LINEAR_KINDS
from mindie_sim.catalog.expert_slots import expert_slot_plan, routed_expert_weight_terms
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.projection_placement import (
    ROLE_ATTENTION_OUTPUT,
    ROLE_LINEAR_INPUT,
    ROLE_LOGITS,
    placed_weight_shape,
)
from mindie_sim.catalog.speculative_state import (
    LINEAR_ATTENTION_KINDS as SPEC_LINEAR_KINDS,
    MODE_CHECKPOINT,
    MODE_REPLAY,
    MODE_SINGLE,
    record_head_bytes,
    speculative_state_plan,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _sharded_shape, term_bytes
from mindie_sim.compose import compose_case
from mindie_sim.prediction.collective_startup import resolve_collective_startup, validate_steps
from mindie_sim.prediction.efficiency import evaluate_target_efficiency, extract_hardware_peaks
from mindie_sim.prediction.expert_slot_work import physical_slot_work
from mindie_sim.prediction.exposed_breakdown import (
    ExposedClocks,
    advance_clocks,
    finish_breakdown,
    join_clocks as prod_join_clocks,
    roofline_clocks,
)
from mindie_sim.prediction.fusion_graph import OpNode, TensorValue, TypedGraph
from mindie_sim.prediction.local_projection_tp import lower_local_projection_tp
from mindie_sim.prediction.recover_ssm_commit import bind_recover_ssm_commits
from mindie_sim.prediction.recover_ssm_work import (
    FP32_RECORD_BYTES,
    emit_commit_op,
    gdn_record_head_bytes,
)
from mindie_sim.prediction.roofline_graph import (
    PhysicalWorkGraph,
    PhysicalWorkNode,
    RooflineTargets,
    critical_path_roofline,
)
from mindie_sim.prediction.runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    EventKey,
    OccurrenceKey,
    PoolKey,
)
from mindie_sim.prediction.runtime.ledger import Ledger, normalise_ranges, ranges_bytes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.weight_prefetch import (
    PrefetchDecision,
    PrefetchPhysics,
    apply_to_work,
    plan_prefetch,
)
from mindie_sim.schema.case import Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_DIR = REPO_ROOT / "formal"
EXPECTED_LEAN_MODULES = (
    "FourClock.lean",
    "PrefetchAndLedger.lean",
    "ShardingAndLayout.lean",
    "RecoverSSM.lean",
    "AllToAllRounds.lean",
    "RooflineAndMesh.lean",
)
SEED = 20261115
TRIALS = 200


def _strip_lean_comments_and_strings(src: str) -> str:
    out: list[str] = []
    idx, length, depth = 0, len(src), 0
    while idx < length:
        if depth == 0 and src[idx] == '"':
            idx += 1
            while idx < length and src[idx] != '"':
                idx += 2 if src[idx] == "\\" else 1
            idx += 1
            out.append('""')
        elif src[idx : idx + 2] == "/-":
            depth, idx = depth + 1, idx + 2
        elif depth > 0 and src[idx : idx + 2] == "-/":
            depth, idx = depth - 1, idx + 2
        elif depth == 0 and src[idx : idx + 2] == "--":
            while idx < length and src[idx] != "\n":
                idx += 1
        else:
            if depth == 0 or src[idx] == "\n":
                out.append(src[idx])
            idx += 1
    assert depth == 0, "unclosed Lean block comment"
    return "".join(out)


def test_formal_kernel_structure_and_zero_sorry_audit() -> None:
    toolchain = (FORMAL_DIR / "lean-toolchain").read_text(encoding="utf-8").strip()
    assert toolchain.startswith("leanprover/lean4:v4.")
    lakefile = (FORMAL_DIR / "lakefile.toml").read_text(encoding="utf-8")
    assert 'name = "mindie_sim_spec"' in lakefile and '"MindIESimSpec"' in lakefile

    umbrella = _strip_lean_comments_and_strings(
        (FORMAL_DIR / "MindIESimSpec.lean").read_text(encoding="utf-8")
    )
    axiom_check = _strip_lean_comments_and_strings(
        (FORMAL_DIR / "AxiomCheck.lean").read_text(encoding="utf-8")
    )
    assert "import MindIESimSpec" in axiom_check
    assert "MindIESimSpec.RooflineAndMesh.ideal_le_target_le_predicted" in axiom_check
    assert "MindIESimSpec.RooflineAndMesh.kMesh_capacity_partition" in axiom_check

    forbidden_re = re.compile(r"\b(?:sorry|sorryAx|admit|axiom)\b")
    theorem_re = re.compile(r"\btheorem\s+[A-Za-z0-9_']+")
    spec_dir = FORMAL_DIR / "MindIESimSpec"
    assert sorted(p.name for p in spec_dir.glob("*.lean")) == sorted(EXPECTED_LEAN_MODULES)

    total_theorems = 0
    for mod_name in EXPECTED_LEAN_MODULES:
        stem = mod_name[:-5]
        assert f"import MindIESimSpec.{stem}" in umbrella
        assert f"#print axioms MindIESimSpec.{stem}." in axiom_check
        raw = (spec_dir / mod_name).read_text(encoding="utf-8")
        assert raw.strip(), f"{mod_name} is empty"
        cleaned = _strip_lean_comments_and_strings(raw)
        assert not forbidden_re.search(cleaned), f"forbidden token in {mod_name}"
        count = len(theorem_re.findall(cleaned))
        assert count >= 4, f"expected >=4 theorems in {mod_name}, found {count}"
        total_theorems += count
    assert total_theorems >= 35

    for lean_path in sorted(FORMAL_DIR.rglob("*.lean")):
        cleaned = _strip_lean_comments_and_strings(lean_path.read_text(encoding="utf-8"))
        assert not forbidden_re.search(cleaned), f"forbidden token in {lean_path}"


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
    # 1. Production runtime.ledger: adjacent 64B then 96B on one BackingKey -> peak 96B (not 160B).
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

    # Static + transient peak coincidence: 128B static + 64B live transient = 192B.
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

    # 2. Production weight_prefetch.plan_prefetch + apply_to_work byte conservation & monotonicity.
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

    # 1. Production static weight ledger vs lowered frontend single-token cold reads
    #    across o_proj_tp, linear_proj_tp, and lm_head_tp.
    q_bf16 = QuantSpec("bf16")
    par_local = ParallelSpec(dp=4, tp=1, ep=4, o_proj_tp=4, linear_proj_tp=2, lm_head_tp=2)
    machine = parse_case(compose_case("qwen3-32b", "Ascend910_9362", tp=4, dp=1, ep=1, layers=1)).machine
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

    # 2. Randomized ceilDiv, FRACTAL_NZ non-underallocation, and redundant expert slot growth.
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
        base_terms = routed_expert_weight_terms(
            [WeightTerm("expert_gate", rows=64, cols=32, category="routed_expert", shard="expert_gate_up", count=e_routed, quant=q_bf16)],
            base_plan, ep_rank=0,
        )
        extra_terms = routed_expert_weight_terms(
            [WeightTerm("expert_gate", rows=64, cols=32, category="routed_expert", shard="expert_gate_up", count=e_routed, quant=q_bf16)],
            extra_plan, ep_rank=0,
        )
        base_ledger = sum(term_bytes(t, q_bf16, p_base) for t in base_terms)
        extra_ledger = sum(term_bytes(t, q_bf16, p_extra) for t in extra_terms)
        assert extra_ledger > base_ledger
        incl = {e: 1.0 / e_routed for e in range(e_routed)}
        sat_base = physical_slot_work(slot_plan=base_plan, logical_inclusions=incl, global_tokens=10**9, packed_weight_bytes=64 * 32 * 2)
        sat_extra = physical_slot_work(slot_plan=extra_plan, logical_inclusions=incl, global_tokens=10**9, packed_weight_bytes=64 * 32 * 2)
        assert sat_extra.weight_read_bytes(0) == pytest.approx(extra_ledger)
        assert sat_base.weight_read_bytes(0) == pytest.approx(base_ledger)


def test_recover_ssm_all_seven_families_rollback_equivalence_and_savings() -> None:
    expected_families = frozenset({"kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"})
    assert SPEC_LINEAR_KINDS == expected_families and POOL_LINEAR_KINDS == expected_families

    def step(st: tuple[int, int, int], tok: tuple[int, int]) -> tuple[int, int, int]:
        s0, s1, s2 = st
        dec, dlt = tok
        return (
            (s0 * dec + dlt + (s2 ^ dlt)) % 1_000_003,
            (s1 * (dec + 3) + s0 * dlt) % 1_000_003,
            (s2 + s1 ^ (dec * dlt)) % 1_000_003,
        )

    rng = random.Random(SEED + 3)
    for _ in range(TRIALS):
        k_spec = rng.randint(1, 10)
        s0 = (rng.randint(0, 9999), rng.randint(0, 9999), rng.randint(0, 9999))
        draft = [(rng.randint(1, 31), rng.randint(0, 500)) for _ in range(k_spec)]
        ckpts = [s0]
        for tok in draft:
            ckpts.append(step(ckpts[-1], tok))
        for accepted in (0, k_spec // 2, k_spec):
            replayed = s0
            for tok in draft[:accepted]:
                replayed = step(replayed, tok)
            assert ckpts[accepted] == replayed

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
        # 1. q=1 identity across recover_ssm=False and recover_ssm=True.
        single_off = speculative_state_plan(None, plan, parallel, query_tokens=1, recover_ssm=False, capacity_padded=True)
        single_on = speculative_state_plan(None, plan, parallel, query_tokens=1, recover_ssm=True, capacity_padded=True)
        assert single_off == single_on and single_on.mode == MODE_SINGLE and single_on.replay_bytes == 0

        # 2. q>1 verify residency savings against independent family record layouts.
        ckpt = speculative_state_plan(None, plan, parallel, query_tokens=5, recover_ssm=False, capacity_padded=True)
        replay = speculative_state_plan(None, plan, parallel, query_tokens=5, recover_ssm=True, capacity_padded=True)
        expected_rec = ((dk + dv + 1) * FP32_RECORD_BYTES if family == "gdn" else dv * FP32_RECORD_BYTES + 2 * dk * 2)
        assert ckpt.mode == MODE_CHECKPOINT and ckpt.state_slots == 5
        assert replay.mode == MODE_REPLAY and replay.state_slots == 1
        assert replay.replay_bytes == pytest.approx(5 * heads * expected_rec)
        assert replay.total_bytes < ckpt.total_bytes
        if family == "gdn":
            assert gdn_record_head_bytes(dk, dv) == expected_rec
        else:
            assert record_head_bytes(dk, dv) == expected_rec

        # 3. Zero (padding/q=1), partial (a=2), and full (a=5) accepted prefix commits.
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
                state_bindings={f"layer{idx}": "base_state"}, runtime=SimpleNamespace(mtp_acceptance_length=1.0),
            )
            op = bound[0]
            assert dict(op.metadata)["recover_ssm_stage"] == "commit"
            assert op.mte2_bytes == pytest.approx(state_bytes + accepted * heads * expected_rec)
            assert op.mte3_bytes == pytest.approx(state_bytes)
            assert op.vector_ops == pytest.approx(4 * accepted * heads * dk * dv)
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
    # 1. Production efficiency.evaluate_target_efficiency clamp_to_target lock:
    #    ideal=4, pred=6, raw_target=8 -> bounded target=6 (0 <= ideal <= target <= predicted).
    case = parse_case(compose_case("qwen3-32b", "Ascend910_9362", tp=4, dp=1, ep=1, layers=1))
    synthetic_base = {
        "schema": "mindie-sim.efficiency/1",
        "geometry": {"phase": "decode", "tokens": 1, "context": 64, "sequences": 1, "scope": "body"},
        "operator_sum_ms": 6.0, "graph_duration_ms": 6.0,
        "total_hbm_ideal_ms": 4.0, "total_cube_ideal_ms": 0.0,
        "total_vec_ideal_ms": 0.0, "total_roofline_ideal_ms": 4.0,
        "network_mbu": 4.0 / 6.0, "network_cube_mfu": 0.0, "network_vec_mfu": 0.0,
        "network_link_util": 0.0, "network_roofline_efficiency": 4.0 / 6.0,
        "operators": [{
            "op_id": "op0", "physical_id": "op0", "family": "gemm", "recipe": "matmul",
            "operator_kind": "compute", "occurrences": 1,
            "flops": 0.0, "vector_ops": 0.0, "hbm_bytes": 4.0e9, "link_bytes": 0.0,
            "t_cube_ideal_ms": 0.0, "t_vec_ideal_ms": 0.0, "t_hbm_ideal_ms": 4.0, "t_link_ideal_ms": 0.0,
            "roofline_ideal_ms": 4.0, "fixed_ms": 0.0, "minimum_ms": 0.0,
            "duration_ms": 6.0, "total_ms": 6.0,
            "mbu": 4.0 / 6.0, "cube_mfu": 0.0, "vec_mfu": 0.0,
        }],
    }
    clamped = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=1, scope="body",
        target_mbu=0.5, mode="clamp_to_target", baseline_analysis=synthetic_base,
    )
    assert clamped["physical_ideal_ms"] == pytest.approx(4.0)
    assert clamped["raw_target_duration_ms"] == pytest.approx(8.0)
    assert clamped["target_step_duration_ms"] == pytest.approx(6.0)
    assert clamped["operators"][0]["raw_target_duration_ms"] == pytest.approx(8.0)
    assert clamped["operators"][0]["target_duration_ms"] == pytest.approx(6.0)
    assert 0.0 <= clamped["physical_ideal_ms"] <= clamped["target_step_duration_ms"] <= clamped["current_step_duration_ms"]

    interior = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=64, sequences=1, scope="body",
        target_mbu=0.8, mode="clamp_to_target", baseline_analysis=synthetic_base,
    )
    assert interior["target_step_duration_ms"] == pytest.approx(5.0)
    assert 0.0 <= interior["physical_ideal_ms"] <= interior["target_step_duration_ms"] <= interior["current_step_duration_ms"]

    # 2. Randomized DAG critical-path roofline monotonicity.
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
        lo = RooflineTargets(cube=rng.uniform(0.35, 0.70), vector=rng.uniform(0.35, 0.70), hbm=rng.uniform(0.35, 0.70), link=rng.uniform(0.35, 0.70))
        hi = RooflineTargets(cube=min(1.0, lo.cube + 0.25), vector=min(1.0, lo.vector + 0.25), hbm=min(1.0, lo.hbm + 0.25), link=min(1.0, lo.link + 0.25))
        res_hi, res_lo = critical_path_roofline(graph, hi), critical_path_roofline(graph, lo)
        assert 0.0 <= res_hi.physical_ideal_ms <= res_hi.target_duration_ms <= res_lo.target_duration_ms + 1e-9

    # 3. Direct K_W mesh scaling on calibrated per_pair UB fabric: (W - 1) * b_peer.
    mesh_sku = "Ascend950DT_9581"
    b_peer_per_rank: dict[int, float] = {}
    for width in (2, 4, 8):
        parsed = parse_case(compose_case("qwen3-32b", mesh_sku, tp=width, dp=1, ep=1, layers=1))
        peaks = extract_hardware_peaks(parsed)
        intra_links = [lnk for lnk in parsed.machine.links if lnk.semantics == "per_pair"]
        assert intra_links
        b_peer = intra_links[0].capacity_bytes_per_ms * 1000.0
        expected_per_rank = (width - 1) * b_peer
        assert peaks["link_peak_bytes_per_s"] == pytest.approx(expected_per_rank, rel=1e-9)
        assert width * (width - 1) * b_peer == pytest.approx(width * expected_per_rank, rel=1e-9)
        b_peer_per_rank[width] = expected_per_rank / b_peer
    assert b_peer_per_rank == {2: pytest.approx(1.0), 4: pytest.approx(3.0), 8: pytest.approx(7.0)}
