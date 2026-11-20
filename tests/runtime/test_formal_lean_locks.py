"""Tier-0 live correspondence locks for all 6 Lean 4 formal specification modules.

[Role]
  Audit the Lean 4 formal verification kernel (`formal/MindIESimSpec/*.lean`)
  for structural completeness, axiom-check coverage, and zero escape tokens
  (`sorry`, `sorryAx`, `admit`, custom `axiom`), and differentially lock all
  six formal theorem modules directly against live `mindie_sim` runtime,
  catalog, collective, roofline, and interconnect modules.

[Invariants]
  * Pure in-memory Tier-0 execution (`< 1.0s`); never spawns `lake`, never
    touches host `$HOME`, and uses zero mocks.
  * Covers all 6 `.lean` modules: `FourClock`, `PrefetchAndLedger`,
    `ShardingAndLayout`, `RecoverSSM` (all 7 linear families),
    `AllToAllRounds`, and `RooflineAndMesh`.

[Timeouts & Locks]
  Deterministic seeded PRNG (`250` trials per theorem domain); zero locks.

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
from mindie_sim.catalog.projection_placement import ROLE_ATTENTION_OUTPUT, placed_weight_shape
from mindie_sim.catalog.speculative_state import (
    LINEAR_ATTENTION_KINDS as SPEC_LINEAR_KINDS,
    MODE_CHECKPOINT,
    MODE_REPLAY,
    record_head_bytes,
    speculative_state_plan,
)
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _sharded_shape
from mindie_sim.compose import compose_case
from mindie_sim.prediction.collective_startup import resolve_collective_startup, validate_steps
from mindie_sim.prediction.efficiency import extract_hardware_peaks
from mindie_sim.prediction.exposed_breakdown import (
    ExposedClocks,
    advance_clocks,
    finish_breakdown,
    join_clocks as prod_join_clocks,
    roofline_clocks,
)
from mindie_sim.prediction.recover_ssm_work import FP32_RECORD_BYTES, gdn_record_head_bytes
from mindie_sim.prediction.roofline_graph import (
    PhysicalWorkGraph,
    PhysicalWorkNode,
    RooflineTargets,
    critical_path_roofline,
)
from mindie_sim.prediction.runtime.ledger import normalise_ranges, ranges_bytes
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.weight_prefetch import PrefetchDecision
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.parallel import ParallelSpec

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
TRIALS = 250


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

    forbidden_re = re.compile(r"\b(?:sorry|sorryAx|admit|axiom)\b")
    theorem_re = re.compile(r"\btheorem\s+[A-Za-z0-9_']+")
    spec_dir = FORMAL_DIR / "MindIESimSpec"
    assert sorted(p.name for p in spec_dir.glob("*.lean")) == sorted(EXPECTED_LEAN_MODULES)

    total_theorems = 0
    for mod_name in EXPECTED_LEAN_MODULES:
        stem = mod_name[:-5]
        assert f"import MindIESimSpec.{stem}" in umbrella
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

    # Branch-winner flip oracle: branch A wins allT (10 > 8), branch B wins non-comm (8 > 2).
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
            _rc(c=c_hid, m=m_hid),
            duration_ms=c_hid, boundary="device_step",
        )["exposed_breakdown_ms"]
        assert (hidden["cube"], hidden["vector"], hidden["memory"], hidden["communication"]) == (
            c_hid, 0.0, 0.0, 0.0,
        )

        durs = [
            roofline_clocks(
                cube_ms=float(rng.randint(0, 80)),
                vector_ms=float(rng.randint(0, 80)),
                memory_ms=float(rng.randint(0, 80)),
                communication_ms=float(rng.randint(0, 80)),
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
    def active_bytes_at(allocs: list[tuple[int, int, int]], t: int) -> int:
        return sum(b for s, f, b in allocs if s <= t < f)

    rng = random.Random(SEED + 1)
    for _ in range(TRIALS):
        w = rng.randint(0, 1_000_000)
        c_free = rng.randint(-200_000, 1_200_000)
        p = 0 if c_free <= 0 else min(w, c_free)
        assert 0 <= p <= w and (c_free < 0 or p <= c_free) and (w - p) + p == w

        p_more_w = 0 if c_free <= 0 else min(w + rng.randint(0, 500), c_free)
        c_next = c_free + rng.randint(0, 500)
        p_more_c = 0 if c_next <= 0 else min(w, c_next)
        assert p_more_w >= p and p_more_c >= p

        dec = PrefetchDecision(
            "op", ("w0",), w, p, 0, w, 0, 1.0, 1e9, 0.0, False, "init", (),
            hbm_rate_ms=100.0, onchip_rate_ms=400.0, alignment=1,
        ).settled(p)
        assert dec.conserved and dec.bytes_prefetched + dec.hbm_read_bytes == w

        t_end = rng.randint(10, 200)
        t_mid = rng.randint(2, t_end - 2)
        a1 = (rng.randint(0, t_mid - 1), t_mid, rng.randint(1, 4096))
        a2 = (rng.randint(t_mid, t_end - 1), t_end, rng.randint(1, 4096))
        for t_probe in (0, a1[0], t_mid - 1, t_mid, a2[0], t_end - 1, t_end):
            assert active_bytes_at([a1, a2], t_probe) <= max(a1[2], a2[2])
        s_alloc = (0, t_end, rng.randint(512, 8192))
        assert active_bytes_at([s_alloc, a1], a1[0]) == s_alloc[2] + a1[2]
        for t_probe in (0, a1[0], t_mid, t_end):
            expected = (s_alloc[2] if t_probe < t_end else 0) + active_bytes_at([a1, a2], t_probe)
            assert active_bytes_at([s_alloc, a1, a2], t_probe) == expected
        merged = normalise_ranges([(0, s_alloc[2]), (0, min(s_alloc[2], a1[2]))])
        assert ranges_bytes(merged) == s_alloc[2]


def test_sharding_ceil_div_fractal_nz_and_redundant_expert_slot_growth() -> None:
    def fractal_nz_bytes(d: int, n: int, c0: int, elem: int) -> int:
        return _ceil_div(d, c0) * _ceil_div(n, 16) * 16 * c0 * elem

    rng = random.Random(SEED + 2)
    for _ in range(TRIALS):
        a, b = rng.randint(0, 8192), rng.randint(1, 128)
        q = _ceil_div(a, b)
        assert a <= q * b < a + b and (a == 0 or q > 0)
        assert _ceil_div(b * (a // b), b) * b == b * (a // b)

        d = rng.randint(1, 1024)
        n = rng.randint(1, 16) * 16 + rng.randint(0, 15)
        elem = rng.choice((1, 2))
        c0 = 32 // elem
        assert fractal_nz_bytes(d, n, c0, elem) >= d * n * elem
        d_al, n_al = _ceil_div(d, c0) * c0, _ceil_div(n, 16) * 16
        assert fractal_nz_bytes(d_al, n_al, c0, elem) == d_al * n_al * elem

        local_tp = rng.choice((1, 2, 4, 8))
        term = WeightTerm(
            "o_proj", rows=d, cols=n, category="attn.o", shard="in_tp",
            placement_role=ROLE_ATTENTION_OUTPUT,
        )
        local_par = ParallelSpec(tp=1, o_proj_tp=local_tp)
        sh_r, sh_c, _ = _sharded_shape(term, local_par, local_tp)
        placed_r, placed_c, _ = placed_weight_shape(term, local_par, local_tp)
        assert (sh_r, sh_c) == (placed_r, placed_c)
        assert sh_r == d and sh_c >= _ceil_div(n, local_tp)
        global_r, global_c, _ = _sharded_shape(
            WeightTerm("proj", rows=d, cols=n, category="attn.o", shard="in_tp"),
            ParallelSpec(tp=local_tp), local_tp,
        )
        assert (global_r, global_c) == (d, _ceil_div(n, local_tp))
        ledger_b = fractal_nz_bytes(sh_r, sh_c, c0, elem)
        assert ledger_b == fractal_nz_bytes(placed_r, placed_c, c0, elem)

        ep = rng.choice((2, 4, 8, 16))
        e_routed = rng.randint(ep * 2, ep * 16)
        max_extra = max(1, min(4, e_routed - _ceil_div(e_routed, ep)))
        r_extra = rng.randint(1, max_extra)
        base_plan = expert_slot_plan(
            None, SimpleNamespace(num_experts=e_routed, index=0),
            ParallelSpec(tp=ep, ep=ep, redundant_experts_per_rank=0),
        )
        extra_plan = expert_slot_plan(
            None, SimpleNamespace(num_experts=e_routed, index=0),
            ParallelSpec(tp=ep, ep=ep, redundant_experts_per_rank=r_extra),
        )
        assert base_plan.base_slots_by_rank[0] == _ceil_div(e_routed, ep)
        assert extra_plan.base_slots_by_rank[0] + extra_plan.redundant_slots_by_rank[0] == (
            _ceil_div(e_routed, ep) + r_extra
        )
        base_terms = routed_expert_weight_terms(
            [WeightTerm("expert_gate", rows=d, cols=n, category="routed_expert",
                        shard="expert_gate_up", count=e_routed)],
            base_plan, ep_rank=0,
        )
        extra_terms = routed_expert_weight_terms(
            [WeightTerm("expert_gate", rows=d, cols=n, category="routed_expert",
                        shard="expert_gate_up", count=e_routed)],
            extra_plan, ep_rank=0,
        )
        assert len(base_terms) == 1 and len(extra_terms) == 2
        assert extra_terms[1].source_owned is False and extra_terms[1].count == r_extra
        assert (extra_plan.base_slots_by_rank[0] + r_extra) * ledger_b > (
            base_plan.base_slots_by_rank[0] * ledger_b
        )


def test_recover_ssm_all_seven_families_rollback_equivalence_and_savings() -> None:
    expected_families = frozenset({"kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"})
    assert SPEC_LINEAR_KINDS == expected_families
    assert POOL_LINEAR_KINDS == expected_families

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
        k_spec = rng.randint(1, 12)
        s0 = (rng.randint(0, 9999), rng.randint(0, 9999), rng.randint(0, 9999))
        draft = [(rng.randint(1, 31), rng.randint(0, 500)) for _ in range(k_spec)]
        ckpts = [s0]
        for tok in draft:
            ckpts.append(step(ckpts[-1], tok))
        assert len(ckpts) == k_spec + 1
        for accepted in range(k_spec + 1):
            replayed = s0
            for tok in draft[:accepted]:
                replayed = step(replayed, tok)
            assert ckpts[accepted] == replayed

    # Live speculative_state_plan residency savings across all 7 linear families.
    parallel = ParallelSpec(tp=1)
    for family in sorted(expected_families):
        plan = LayerPlan(
            index=0, attention=family, ffn="dense", num_heads=16, num_kv_heads=4,
            head_dim=128, linear_heads=16, linear_key_heads=16, linear_value_heads=16,
            linear_head_dim=128, linear_key_head_dim=128, conv_kernel=4,
            recurrent_state_elements=16 * 128 * 128, conv_state_elements=16 * 128 * 4,
        )
        ckpt_plan = speculative_state_plan(
            None, plan, parallel, query_tokens=5, recover_ssm=False, capacity_padded=True,
        )
        replay_plan = speculative_state_plan(
            None, plan, parallel, query_tokens=5, recover_ssm=True, capacity_padded=True,
        )
        assert ckpt_plan.mode == MODE_CHECKPOINT
        assert replay_plan.mode == MODE_REPLAY
        assert ckpt_plan.state_slots == 5
        assert replay_plan.state_slots == 1
        assert replay_plan.total_bytes < ckpt_plan.total_bytes
        if family == "kda":
            expected_rec_per_head = record_head_bytes(128, 128)
            state_per_head = 128 * 128 * FP32_RECORD_BYTES
            assert state_per_head == 64 * expected_rec_per_head
            assert replay_plan.replay_bytes == pytest.approx(5 * 16 * expected_rec_per_head)
        elif family == "gdn":
            gdn_rec = gdn_record_head_bytes(128, 128)
            assert gdn_rec == (128 + 128 + 1) * FP32_RECORD_BYTES
            assert replay_plan.replay_bytes == pytest.approx(5 * 16 * gdn_rec)


def test_alltoall_round_scaling_and_step_bounds_against_collective_startup() -> None:
    log_128 = resolve_collective_startup(
        family="alltoallv", group_size=128, metadata={},
        runtime_mode="log2", topology_protocols={},
    )
    ring_128 = resolve_collective_startup(
        family="alltoallv", group_size=128, metadata={},
        runtime_mode="ring", topology_protocols={},
    )
    assert (log_128.startup_rounds, ring_128.startup_rounds) == (7, 127)
    assert log_128.peer_phases == ring_128.peer_phases == 127

    rng = random.Random(SEED + 4)
    for idx in range(TRIALS):
        g = [1, 2, 3, 4, 64, 128][idx] if idx < 6 else rng.randint(1, 4096)
        log_desc = resolve_collective_startup(
            family="alltoall", group_size=g, metadata={},
            runtime_mode="log2", topology_protocols={},
        )
        ring_desc = resolve_collective_startup(
            family="alltoall", group_size=g, metadata={},
            runtime_mode="ring", topology_protocols={},
        )
        lg = log_desc.startup_rounds
        assert g <= (1 << lg)
        if g <= 1:
            assert lg == 0 and ring_desc.startup_rounds == 0
            with pytest.raises(ModelingError):
                validate_steps(1, g)
        else:
            assert 1 <= lg <= g - 1
            assert ring_desc.startup_rounds == g - 1
            assert validate_steps(lg, g) == lg
            assert validate_steps(1, g) == 1
            assert validate_steps(g - 1, g) == g - 1
            with pytest.raises(ModelingError):
                validate_steps(g, g)
            with pytest.raises(ModelingError):
                validate_steps(0, g)
            with pytest.raises(ModelingError):
                validate_steps(True, g)
        if g >= 4:
            assert lg < ring_desc.startup_rounds


def test_roofline_critical_path_monotonicity_and_direct_mesh_scaling() -> None:
    rng = random.Random(SEED + 5)
    for _ in range(TRIALS):
        n_nodes = rng.randint(2, 10)
        nodes: list[PhysicalWorkNode] = []
        for idx in range(n_nodes):
            preds = tuple(f"n{p}" for p in (rng.sample(range(idx), rng.randint(1, min(3, idx))) if idx else ()))
            nodes.append(PhysicalWorkNode(
                node_id=f"n{idx}",
                cube_ms=float(rng.randint(0, 40)),
                vector_ms=float(rng.randint(0, 40)),
                hbm_ms=float(rng.randint(0, 40)),
                link_ms=float(rng.randint(0, 40)),
                predecessors=preds,
            ))
        graph = PhysicalWorkGraph(nodes=tuple(nodes))
        targets_lo = RooflineTargets(
            cube=rng.uniform(0.35, 0.70), vector=rng.uniform(0.35, 0.70),
            hbm=rng.uniform(0.35, 0.70), link=rng.uniform(0.35, 0.70),
        )
        targets_hi = RooflineTargets(
            cube=min(1.0, targets_lo.cube + 0.25), vector=min(1.0, targets_lo.vector + 0.25),
            hbm=min(1.0, targets_lo.hbm + 0.25), link=min(1.0, targets_lo.link + 0.25),
        )
        res_hi = critical_path_roofline(graph, targets_hi)
        res_lo = critical_path_roofline(graph, targets_lo)
        # Lower target efficiency => higher or equal roofline bound; ideal <= target(hi) <= target(lo).
        assert 0.0 <= res_hi.physical_ideal_ms <= res_hi.target_duration_ms + 1e-9
        assert res_hi.target_duration_ms <= res_lo.target_duration_ms + 1e-9
        assert res_hi.critical_path_ideal_ms <= res_hi.critical_path_target_ms + 1e-9

    # Direct K_W mesh scaling on calibrated per_pair UB fabric: (W - 1) * b_peer.
    mesh_sku = "Ascend950DT_9581"
    b_peer_per_rank: dict[int, float] = {}
    for width in (2, 4, 8):
        parsed = parse_case(compose_case("qwen3-32b", mesh_sku, tp=width, dp=1, ep=1, layers=1))
        peaks = extract_hardware_peaks(parsed)
        intra_links = [lnk for lnk in parsed.machine.links if lnk.semantics == "per_pair"]
        assert intra_links, "expected per_pair direct mesh link"
        b_peer = intra_links[0].capacity_bytes_per_ms * 1000.0
        expected_per_rank = (width - 1) * b_peer
        assert peaks["link_peak_bytes_per_s"] == pytest.approx(expected_per_rank, rel=1e-9)
        assert width * (width - 1) * b_peer == pytest.approx(width * expected_per_rank, rel=1e-9)
        b_peer_per_rank[width] = expected_per_rank / b_peer
    assert b_peer_per_rank == {2: pytest.approx(1.0), 4: pytest.approx(3.0), 8: pytest.approx(7.0)}
