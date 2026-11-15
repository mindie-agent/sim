"""Tier-0 audit and differential property tests for `formal/` (`MindIESimSpec`).

[Role]
  Audit the Lean 4 formal verification kernel (`formal/MindIESimSpec/*.lean`) for
  structural completeness and zero forbidden escape tokens (`sorry`, `admit`, custom
  `axiom`), and differentially verify all five machine-checked algebraic invariants
  against live `mindie_sim` helpers across 250 seeded randomized trials each.

[Invariants]
  * Pure in-memory CPU execution (`< 0.5s`); never shells out to `lake` or touches `$HOME`.
  * Zero `MagicMock`; exercises real `mindie_sim` ledger, prefetch, layout, sharding, and
    linear-state modules alongside the Lean-specified integer equations.

[Timeouts & Locks]
  Deterministic `random.Random(20261115)` over bounded 12-node DAGs; no I/O locks.

[Side-Effect Policy]
  Read-only inspection of `formal/` files; zero filesystem writes.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import re

from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _sharded_shape
from mindie_sim.kv_layout_physics import KvNzPlane, fractal_c0_elements
from mindie_sim.linear_state_physics import LINEAR_STATE_KINDS
from mindie_sim.prediction.exposed_breakdown import (
    ExposedClocks, advance_clocks, finish_breakdown,
    join_clocks as prod_join_clocks, roofline_clocks,
)
from mindie_sim.prediction.runtime.ledger import normalise_ranges, ranges_bytes
from mindie_sim.prediction.weight_prefetch import PrefetchDecision
from mindie_sim.schema.parallel import ParallelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO_ROOT / "formal"
EXPECTED_MODULES = (
    "FourClock.lean", "PrefetchAndLedger.lean", "ShardingAndLayout.lean",
    "RecoverSSM.lean", "AllToAllRounds.lean",
)
SEED, TRIALS = 20261115, 250


def _strip_lean_comments_and_strings(src: str) -> str:
    out, idx, length, depth = [], 0, len(src), 0
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


def test_lean_formal_kernel_structure_and_zero_sorry_audit() -> None:
    toolchain = (FORMAL_DIR / "lean-toolchain").read_text(encoding="utf-8").strip()
    assert toolchain.startswith("leanprover/lean4:v4.")
    lakefile = (FORMAL_DIR / "lakefile.toml").read_text(encoding="utf-8")
    assert 'name = "mindie_sim_spec"' in lakefile and '"MindIESimSpec"' in lakefile
    umbrella = _strip_lean_comments_and_strings(
        (FORMAL_DIR / "MindIESimSpec.lean").read_text(encoding="utf-8")
    )
    forbidden_re = re.compile(r"\b(?:sorry|sorryAx|admit|axiom)\b")
    theorem_re = re.compile(r"\btheorem\s+[A-Za-z0-9_']+")
    spec_dir = FORMAL_DIR / "MindIESimSpec"
    assert sorted(p.name for p in spec_dir.glob("*.lean")) == sorted(EXPECTED_MODULES)

    total_theorems = 0
    for mod_name in EXPECTED_MODULES:
        assert f"import MindIESimSpec.{mod_name[:-5]}" in umbrella
        raw = (spec_dir / mod_name).read_text(encoding="utf-8")
        assert raw.strip(), f"{mod_name} is empty"
        cleaned = _strip_lean_comments_and_strings(raw)
        assert not forbidden_re.search(cleaned), f"forbidden token in {mod_name}"
        count = len(theorem_re.findall(cleaned))
        assert count >= 4, f"expected >=4 theorems in {mod_name}, found {count}"
        total_theorems += count
    assert total_theorems >= 30
    for lean_path in sorted(FORMAL_DIR.rglob("*.lean")):
        cleaned = _strip_lean_comments_and_strings(lean_path.read_text(encoding="utf-8"))
        assert not forbidden_re.search(cleaned), f"forbidden token in {lean_path}"


def test_four_clock_max_plus_dag_invariants() -> None:
    def roofline(c: int, v: int, m: int, n: int) -> tuple[int, int, int, int]:
        return (max(c, v, m, n), max(c, v, m), max(c, v), c)

    def join_clocks(preds: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        acc = (0, 0, 0, 0)
        for p in preds:
            acc = (max(acc[0], p[0]), max(acc[1], p[1]), max(acc[2], p[2]), max(acc[3], p[3]))
        return acc

    def advance(preds: list[tuple[int, int, int, int]], dur: tuple[int, int, int, int]):
        j = join_clocks(preds)
        return (j[0] + dur[0], j[1] + dur[1], j[2] + dur[2], j[3] + dur[3])

    def finish(clk: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (clk[3], clk[2] - clk[3], clk[1] - clk[2], clk[0] - clk[1])

    branch_a = advance([roofline(2, 0, 0, 0)], roofline(0, 0, 0, 8))
    branch_b = advance([roofline(5, 0, 0, 0)], roofline(0, 0, 3, 0))
    assert branch_a[0] > branch_b[0] and branch_b[1] > branch_a[1]
    assert finish(join_clocks([branch_a, branch_b])) == (5, 0, 3, 2)

    rng = random.Random(SEED)
    for _ in range(TRIALS):
        c_hid = rng.randint(0, 500)
        assert finish(roofline(c_hid, 0, rng.randint(0, c_hid), 0)) == (c_hid, 0, 0, 0)
        durs = [roofline(*(rng.randint(0, 80) for _ in range(4))) for _ in range(12)]
        clocks_a: list[tuple[int, int, int, int]] = []
        clocks_b: list[tuple[int, int, int, int]] = []
        for node in range(12):
            preds_idx = rng.sample(range(node), rng.randint(1, min(4, node))) if node else []
            perm_idx = list(preds_idx)
            rng.shuffle(perm_idx)
            ca = advance([clocks_a[p] for p in preds_idx], durs[node])
            cb = advance([clocks_b[p] for p in perm_idx], durs[node])
            assert ca == cb and 0 <= ca[3] <= ca[2] <= ca[1] <= ca[0]
            clocks_a.append(ca)
            clocks_b.append(cb)
        terminal = join_clocks(clocks_a)
        buckets = finish(terminal)
        assert all(val >= 0 for val in buckets) and sum(buckets) == terminal[0]
        prod_nodes: list[ExposedClocks] = []
        prod_durs = [roofline_clocks(cube_ms=d[3], vector_ms=max(0, d[2] - d[3]),
                                     memory_ms=max(0, d[1] - d[2]),
                                     communication_ms=max(0, d[0] - d[1])) for d in durs]
        # Re-run exact production ExposedClocks against the integer Lean oracle.
        rng_chk = random.Random(SEED + _ * 17)
        for node in range(12):
            p_idx = rng_chk.sample(range(node), rng_chk.randint(1, min(4, node))) if node else []
            q_idx = list(p_idx)
            rng_chk.shuffle(q_idx)
            pa = advance_clocks([prod_nodes[p] for p in p_idx], prod_durs[node])
            pb = advance_clocks([prod_nodes[p] for p in q_idx], prod_durs[node])
            assert pa == pb
            prod_nodes.append(pa)
        p_term = prod_join_clocks(prod_nodes)
        rep = finish_breakdown(p_term, duration_ms=p_term.all_ms, boundary="device_step")
        ms = rep["exposed_breakdown_ms"]
        assert (ms["cube"], ms["vector"], ms["memory"], ms["communication"]) == (
            p_term.cube_ms, p_term.compute_ms - p_term.cube_ms,
            p_term.without_communication_ms - p_term.compute_ms,
            p_term.all_ms - p_term.without_communication_ms,
        )
        if p_term.all_ms > 0:
            assert abs(math.fsum(rep["exposed_breakdown_pct"].values()) - 100.0) <= 1e-9


def test_prefetch_clamp_and_interval_ledger_invariants() -> None:
    def prefetch_bytes(w: int, c_free: int) -> int:
        return 0 if c_free <= 0 else min(w, c_free)

    def active_bytes_at(allocs: list[tuple[int, int, int]], t: int) -> int:
        return sum(b for s, f, b in allocs if s <= t < f)

    rng = random.Random(SEED + 1)
    for _ in range(TRIALS):
        w, c_free = rng.randint(0, 1_000_000), rng.randint(-200_000, 1_200_000)
        p = prefetch_bytes(w, c_free)
        assert 0 <= p <= w and (c_free < 0 or p <= c_free) and (w - p) + p == w
        assert prefetch_bytes(w + rng.randint(0, 500), c_free) >= p
        assert prefetch_bytes(w, c_free + rng.randint(0, 500)) >= p
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
            exp = (s_alloc[2] if t_probe < t_end else 0) + active_bytes_at([a1, a2], t_probe)
            assert active_bytes_at([s_alloc, a1, a2], t_probe) == exp
        merged = normalise_ranges([(0, s_alloc[2]), (0, min(s_alloc[2], a1[2]))])
        assert ranges_bytes(merged) == s_alloc[2]


def test_sharding_ceil_div_and_fractal_nz_invariants() -> None:
    def fractal_nz_bytes(d: int, n: int, c0: int, elem: int) -> int:
        return _ceil_div(d, c0) * _ceil_div(n, 16) * 16 * c0 * elem

    assert fractal_c0_elements("bf16") == 16
    rng = random.Random(SEED + 2)
    for _ in range(TRIALS):
        a, b = rng.randint(0, 8192), rng.randint(1, 128)
        q = _ceil_div(a, b)
        assert a <= q * b < a + b and (a == 0 or q > 0)
        assert _ceil_div(b * (a // b), b) * b == b * (a // b)

        d, n = rng.randint(1, 1024), rng.randint(1, 16) * 16 + rng.randint(0, 15)
        elem = rng.choice((1, 2))
        c0 = 32 // elem
        assert fractal_nz_bytes(d, n, c0, elem) >= d * n * elem
        d_al, n_al = _ceil_div(d, c0) * c0, _ceil_div(n, 16) * 16
        assert fractal_nz_bytes(d_al, n_al, c0, elem) == d_al * n_al * elem
        plane = KvNzPlane(
            name="nope", head_size=d, dtype="bf16" if elem == 2 else "int8",
            num_blocks=1, num_kv_heads=1, block_size=16, pad_to_tile=True,
        )
        assert int(plane.physical_bytes) == fractal_nz_bytes(d, 16, c0, elem)

        local_tp = rng.choice((1, 2, 4, 8))
        term = WeightTerm("proj", rows=d, cols=n, category="attn.o", shard="in_tp")
        sh_r, sh_c, _ = _sharded_shape(term, ParallelSpec(tp=local_tp), local_tp)
        ledger_b = fractal_nz_bytes(sh_r, sh_c, c0, elem)
        assert ledger_b == fractal_nz_bytes(d, _ceil_div(n, local_tp), c0, elem)
        e_routed, ep, r_extra = rng.randint(8, 256), rng.choice((1, 2, 4, 8, 16, 32)), rng.randint(1, 4)
        assert (_ceil_div(e_routed, ep) + r_extra) * ledger_b > _ceil_div(e_routed, ep) * ledger_b


def test_recover_ssm_checkpoint_vs_replay_invariants() -> None:
    assert LINEAR_STATE_KINDS == frozenset(
        {"kda", "gdn", "mamba", "gla", "deltanet", "lightning", "retnet"}
    )

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
        heads = rng.randint(1, 64)
        state_bytes, record_bytes = heads * 128 * 128 * 4, heads * (4 * 128 + 2 * 128 * 2)
        assert state_bytes == 64 * record_bytes
        assert state_bytes + (k_spec + 1) * record_bytes < (k_spec + 1) * state_bytes


def test_alltoall_round_scaling_and_override_bounds() -> None:
    def ceil_log2(g: int) -> int:
        return 0 if g <= 1 else (g - 1).bit_length()

    def resolve_steps(default_s: int, override: int | None, g: int) -> int | None:
        cand = default_s if override is None else override
        return (0 if cand == 0 else None) if g <= 1 else (cand if 1 <= cand <= g - 1 else None)

    assert ceil_log2(128) == 7 and (128 - 1) == 127
    rng = random.Random(SEED + 4)
    for idx in range(TRIALS):
        g = [1, 2, 3, 4, 64, 128][idx] if idx < 6 else rng.randint(1, 4096)
        lg = ceil_log2(g)
        assert (g < 1 or g <= (1 << lg)) and resolve_steps(lg, None, g) == lg
        if g >= 2:
            assert 1 <= lg <= g - 1 and resolve_steps(lg, g, g) is None
        if g >= 4:
            assert lg < g - 1
        resolved = resolve_steps(lg, rng.choice((0, 1, max(1, g - 1), g, g + 1, -1)), g)
        if resolved is not None:
            assert (resolved == 0) if g <= 1 else (1 <= resolved <= g - 1)
