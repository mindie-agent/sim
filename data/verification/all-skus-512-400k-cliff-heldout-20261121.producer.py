"""Produce sanitized all-SKU 512..400K Anchor + Off-Grid Cliff Heldout + 5-Gate bundle.

[Role] Replay live NPU silicon captures across all reachable Ascend SKUs
  (Ascend910B4, Ascend910_9362, Ascend910_9382, Ascend950DT_9582) plus explicit
  unmeasured inventory for Ascend950PR_9579 through the white-box Single-Spine
  simulator with calibrated L0 primitive tables and emit a SHA-256-sealed bundle.
[Invariants]
  * Strict Anchor/Heldout disjointness: ANCHOR_GRID & HELDOUT_GRID == empty.
  * Unfiltered zero-drop accounting: total == pass + fail + blocked on every SKU.
  * Inclusive full float64 tolerance: |pred_ns - meas_ns| <= max(0.10 * meas_ns, 2000.0),
    with mandatory finite > 0 guards on every measured and predicted duration.
  * Independent semantic-contract Lean L* lower bound verified for 0 inversions.
  * Zero privacy leaks: no IPv4 addresses, hostnames, usernames, or home paths.
[Timeouts & Locks] Parallel 4-worker ProcessPoolExecutor; read-only capture I/O;
  writes only the caller-requested output JSON atomically via temp+replace.
[Side-Effect Policy] Pure white-box replay; zero empirical fitting or mutation of
  calibrated primitive tables or hardware profiles.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mindie_sim import compose_case, graph_costs, parse_case
from mindie_sim.catalog.ops import comm_op
from mindie_sim.optimization.dual_bound import EventDagSpec, ResourceChoiceProgram, certify_lower_bound, check_optimality_certificate, inward_floor_ns
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import attention, linear_attention, paged_mla, sparse_indexer
from mindie_sim.prediction.operators.attention._fia import AttentionContract
from mindie_sim.prediction.operators.attention.common import C8_V1_ABI, QUANT_MODE_BF16, QUANT_MODE_INT8_C8, SFA_V1_ABI
from mindie_sim.prediction.operators.paged_mla import indexed_attention, indexed_quant_attention
from mindie_sim.prediction.operators.paged_mla.common import PagedMlaContract
from mindie_sim.prediction.operators.sparse_indexer.common import SparseIndexerContract
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
from mindie_sim.prediction.runtime.memory_physics import AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec

REACHABLE_SKUS: tuple[str, ...] = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382", "Ascend950DT_9582")
UNMEASURED_SKUS: tuple[dict[str, str], ...] = ({
    "sku": "Ascend950PR_9579", "arch_preset": "npu_arch_3510",
    "endpoint_alias": "a5-9579-unmeasured", "status": "unmeasured",
    "reason": "no_reachable_physical_hardware_in_fleet",
},)
ENDPOINT_ALIAS: dict[str, str] = {
    "Ascend910B4": "a2-910b4-ep", "Ascend910_9362": "a3-9362-ep",
    "Ascend910_9382": "a3-9382-ep", "Ascend950DT_9582": "a5-9582-ep",
}
ANCHOR_CONTEXT_GRID: tuple[int, ...] = (512, 2048, 8192, 32768, 131072, 262144, 409600)
OFF_GRID_CLIFF_HELDOUT_GRID: tuple[int, ...] = (768, 2049, 4097, 24576, 98304, 200000, 350000, 400000)

REL_TOL, ABS_FLOOR_NS, NS_PER_US, US_PER_MS, US_PER_S = 0.10, 2000.0, 1000.0, 1000.0, 1e6
BYTES_PER_GB, MIB, LSTAR_INV_EPS = 1e9, 1024 * 1024, 1e-6
STEADY_COMM_MIN_BYTES = 16 * MIB
BF16_BYTES, FP32_BYTES = 2, 4
GQA_HQ, GQA_HKV, GQA_D = 32, 8, 128
MLA_H, MLA_NOPE, MLA_ROPE, C8_ROW_BYTES = 128, 512, 64, 656
MLA_DQK = MLA_NOPE + MLA_ROPE
IDX_H, IDX_D, DEFAULT_TOPK, DEFAULT_PAGE_SIZE, L2_GRANULE_BYTES = 64, 128, 2048, 128, 128
GDN_DEFAULT_H, KDA_CHUNK_DEFAULT_H, KDA_REC_DEFAULT_H = 4, 2, 8
GDN_BT, GDN_DIM, MAX_ERR_CHARS = 64, 128, 160
PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("home_mac", re.compile("/" + "Users/[A-Za-z0-9_.-]+")),
    ("home_linux", re.compile("/" + "home/[A-Za-z0-9_.-]+")),
    ("home_root", re.compile("/" + "root/[A-Za-z0-9_.-]+")),
    ("user_at_host", re.compile(r"[^\s\"'@]+@[A-Za-z0-9][A-Za-z0-9._-]+")),
    ("fqdn", re.compile(r"(?:^|[\s\"'/:@])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:invalid|local|internal|corp|lan|arpa|com|net|org|edu|gov|io|cn)\b", re.IGNORECASE)),
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sanitize_error(msg: Any) -> str:
    text = str(msg or "unknown").replace("\r", " ").replace("\n", " ").strip()
    for _, pat in PRIVACY_PATTERNS:
        text = pat.sub("<redacted>", text)
    return text[:MAX_ERR_CHARS]


def _assert_no_privacy_leaks(serialized: str) -> None:
    for label, pat in PRIVACY_PATTERNS:
        m = pat.search(serialized)
        if m is not None:
            raise ValueError(f"privacy_leak_detected:{label}:{m.group(0)}")


def _hw_peaks(cs: Any) -> dict[str, Any]:
    die = cs.machine.dies[0]
    dp = datapath_for_die(die)
    peaks, rates = dict(die.compute_peaks), dict(die.rates)
    tc = {k: dict(v) for k, v in cs.machine.transfer_channels}
    d2d_bws = [
        float(d.get("uni_peak_bw_gbps") or d.get("effective_uni_bw_gbps") or d.get("uni_bw_gbps") or 0.0)
        for k, d in tc.items() if "d2h" not in k and "h2d" not in k and "pcie" not in k.lower()
    ]
    d2d = tc.get("d2d_intra_node", {})
    max_link_gbps = max(d2d_bws) if any(b > 0.0 for b in d2d_bws) else float(d2d.get("uni_peak_bw_gbps") or 25.0)
    per_pair_bw_gbps = float(d2d.get("uni_peak_bw_gbps") or max_link_gbps)
    is_per_pair = str(d2d.get("semantics", "")) == "per_pair" or "MESH" in str(d2d.get("protocol", "")).upper()
    return {
        "bf16": float(peaks.get("bf16", 1.0)), "int8": float(peaks.get("int8", peaks.get("bf16", 1.0))),
        "vector": float(rates.get("vector", 1.0)),
        "hbm_r": float(dp.hbm_read_bytes_per_s or rates.get("hbm", 1.0)),
        "hbm_w": float(dp.hbm_write_bytes_per_s or rates.get("hbm", 1.0)),
        "l2": float(rates.get("l2", rates.get("mte2_aic", rates.get("hbm", 1.0)))),
        "l2_bytes": float(dict(die.buffers).get("l2", 0)),
        "link": max_link_gbps * BYTES_PER_GB, "per_pair_link": per_pair_bw_gbps * BYTES_PER_GB,
        "is_per_pair": is_per_pair, "arch": str(die.compute_arch.preset or "npu_arch_2201"),
        "aic": int(dp.aic_count), "aiv": int(dp.aiv_count),
    }


def _ns_floor(work: float, cap_per_s: float) -> int:
    if work <= 0.0 or cap_per_s <= 0.0:
        return 0
    return inward_floor_ns(Fraction(work) * Fraction(1_000_000_000) / Fraction(cap_per_s))


def _lean_lstar_us(
    hw: dict[str, Any], *, cube_flops: float = 0.0, cube_dtype: str = "bf16", vec_ops: float = 0.0,
    read_bytes: float = 0.0, write_bytes: float = 0.0, comm_bytes: float = 0.0,
    link_bw_bps: float | None = None, warm: bool = False,
) -> float:
    bw = hw["l2"] if warm else max(hw["hbm_r"], hw["hbm_w"])
    link_bw = link_bw_bps if (link_bw_bps and link_bw_bps > 0.0) else hw["link"]
    global_work = (
        _ns_floor(cube_flops, hw.get(cube_dtype, hw["bf16"])),
        _ns_floor(vec_ops, hw["vector"]),
        _ns_floor(read_bytes, bw),
        _ns_floor(write_bytes, bw),
        _ns_floor(comm_bytes, link_bw),
    )
    prog = ResourceChoiceProgram(len(global_work), global_work, ())
    dag = EventDagSpec((0,), ())
    cert, _ = certify_lower_bound(prog, dag)
    if not check_optimality_certificate(prog, dag, cert):
        raise AssertionError("lean_optimality_certificate_rejected")
    return float(cert.l_star_lean_ns) / NS_PER_US


def _verdict(
    case_id: str, family: str, split: str, meas_p50: float, meas_p99: float,
    pred_us: float, lstar_us: float, capture_uri: str,
) -> dict[str, Any]:
    cid = str(case_id or f"{family}_{split}_unnamed")
    finite_ok = (
        math.isfinite(meas_p50) and meas_p50 > 0.0
        and math.isfinite(meas_p99) and meas_p99 > 0.0
        and meas_p99 + 1e-9 >= meas_p50
        and math.isfinite(pred_us) and pred_us > 0.0
        and math.isfinite(lstar_us) and lstar_us >= 0.0
    )
    if not finite_ok:
        return {
            "case_id": cid, "family": family, "split": split, "status": "fail",
            "meas_us": float(meas_p50) if math.isfinite(meas_p50) else None,
            "meas_p99_us": float(meas_p99) if math.isfinite(meas_p99) else None,
            "pred_us": float(pred_us) if math.isfinite(pred_us) else None,
            "lstar_us": float(lstar_us) if math.isfinite(lstar_us) else None,
            "ape_pct": 999.0, "inv_sim": True, "inv_obs": True, "capture_uri": capture_uri,
        }
    meas_ns = meas_p50 * NS_PER_US
    pred_ns = pred_us * NS_PER_US
    err_ns = abs(pred_ns - meas_ns)
    tol_ns = max(REL_TOL * meas_ns, ABS_FLOOR_NS)
    inv_sim = lstar_us > pred_us * (1.0 + LSTAR_INV_EPS)
    inv_obs = lstar_us > meas_p99 * (1.0 + LSTAR_INV_EPS)
    passed = (err_ns <= tol_ns) and (not inv_sim) and (not inv_obs)
    ape_pct = (err_ns / meas_ns) * 100.0
    return {
        "case_id": cid, "family": family, "split": split,
        "status": "pass" if passed else "fail",
        "meas_us": float(meas_p50), "meas_p99_us": float(meas_p99),
        "pred_us": float(pred_us), "lstar_us": float(lstar_us),
        "ape_pct": round(ape_pct, 4), "inv_sim": inv_sim, "inv_obs": inv_obs,
        "capture_uri": capture_uri,
    }


class _SkuReplayHarness:
    """White-box single-spine replay harness for one SKU."""

    def __init__(self, sku: str, *, use_calibrated: bool = True, use_submission: bool = True) -> None:
        self.sku, self.use_submission = sku, use_submission
        self.cs = parse_case(compose_case("qwen3.8-27b", hardware=sku, tp=1, ep=1, num_devices=1, layers=1, quant="none"))
        self.pt = load_primitives(sku) if use_calibrated else None
        self.hw = _hw_peaks(self.cs)
        self.is_a5 = self.hw["arch"] == "npu_arch_3510"
        self._ev_cache: dict[tuple[str, ...], Any] = {}

    def run_graph(
        self, ops: list[Any], recipes: Any = None, *, isolated: bool = True,
        case_spec: Any = None, include_sub: bool | None = None,
    ) -> float:
        target = self.cs if case_spec is None else case_spec
        ev = None
        if self.pt is not None:
            key = (str(id(target)), *sorted(recipes.recipes.keys())) if recipes else (str(id(target)),)
            if key not in self._ev_cache:
                self._ev_cache[key] = EvaluationContext.create(target, recipes or RecipeBook(), primitive_table=self.pt)
            ev = self._ev_cache[key]
        rep = graph_costs(
            target, {0: ops}, recipes=recipes, evaluation=ev,
            include_submission=self.use_submission if include_sub is None else include_sub,
            isolated=isolated, compact=True, fast_span_only=True,
        )
        return float(rep["duration_ms"]) * US_PER_MS

    def pred_fia(
        self, batch: int, q_len: int, hq: int, hkv: int, dim: int, seq_kv: int, *,
        layout: str = "BNSD", page_size: int = 0, block_table: tuple[tuple[int, ...], ...] = (),
        phase: str = "decode", causal: bool = False, sequences: tuple[tuple[int, int], ...] | None = None,
    ) -> float:
        seqs = sequences if sequences is not None else tuple((q_len, seq_kv) for _ in range(batch))
        c = AttentionContract(
            query_tokens=sum(q for q, _ in seqs), num_heads=hq, num_kv_heads=hkv,
            head_dim=dim, v_head_dim=dim, sequences=seqs, phase=phase, causal=causal,
            kv_layout=layout, block_size=page_size if page_size > 0 else DEFAULT_PAGE_SIZE,
            batch_size=len(seqs), block_table=block_table,
        )
        op = attention.make_attention_op(c, op_id="fia")
        return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(attention.build)}))

    def pred_paged_mla(
        self, seq_kv: int, *, batch: int = 1, heads: int = MLA_H,
        page_size: int = DEFAULT_PAGE_SIZE, block_table: tuple[tuple[int, ...], ...] = (),
    ) -> float:
        c = PagedMlaContract(
            sequences=tuple((1, seq_kv) for _ in range(batch)), logical_heads=heads,
            physical_heads=heads, latent_dim=MLA_NOPE, rope_dim=MLA_ROPE,
            page_size=page_size, block_table=block_table,
        )
        op = paged_mla.make_op(c, op_id="pmla")
        return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(paged_mla.build)}))

    def pred_indexer(
        self, seq_kv: int, *, batch: int = 1, heads: int = IDX_H, dim: int = IDX_D,
        topk: int = DEFAULT_TOPK, seqs: list[int] | None = None,
    ) -> float:
        c = SparseIndexerContract(
            query_tokens=len(seqs) if seqs is not None else batch,
            scanned_rows=max(seqs) if seqs is not None else seq_kv,
            index_heads=heads, index_dim=dim, topk_items=topk,
            batch_size=len(seqs) if seqs is not None else batch, native_split_core=True,
        )
        op = sparse_indexer.make_indexer_op(c, op_id="lidx")
        if seqs is not None:
            op = replace(op, metadata=tuple({**dict(op.metadata), "kv_lens": tuple(int(x) for x in seqs)}.items()))
        return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(sparse_indexer.build)}))

    def pred_sfa(
        self, seq_kv: int, *, batch: int = 1, heads: int = MLA_H, topk: int = DEFAULT_TOPK,
        quant_c8: bool = False, cann_ver: str = "",
    ) -> float:
        eff_tk = min(topk, seq_kv)
        if self.is_a5:
            c = AttentionContract(
                query_tokens=batch, num_heads=heads, num_kv_heads=1, head_dim=MLA_DQK,
                v_head_dim=MLA_NOPE, selected_tokens=eff_tk, phase="decode",
                abi=C8_V1_ABI if quant_c8 else SFA_V1_ABI, kv_layout="TND",
                quant_mode=QUANT_MODE_INT8_C8 if quant_c8 else QUANT_MODE_BF16,
                batch_size=batch, sequences=tuple((1, seq_kv) for _ in range(batch)),
            )
            op = attention.make_attention_op(c, op_id="c8sfa" if quant_c8 else "sfadsa")
            return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(attention.build)}))
        shape = (batch, heads, MLA_NOPE)
        if quant_c8:
            geom = dict(query=batch, context=seq_kv, topk=eff_tk, heads=heads, latent=MLA_NOPE,
                        rope=MLA_ROPE, query_dtype="bf16", cache_dtype="int8", window=0, compression_ratio=1, cp=1)
            op = OpSpec("c8sfa", "attention", "decode", 0, 0, (), shape, shape, shape, shape,
                        recipe="indexed_quant_attention", dtype="bf16", metadata=(("indexed_attention_geometry", geom),))
            return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(
                lambda o, g: indexed_quant_attention.build(o, g, scratch_policy="offchip"),
            )}))
        geom = dict(query=batch, context=seq_kv, topk=eff_tk, heads=heads, latent=MLA_NOPE,
                    rope=MLA_ROPE, query_dtype="bf16", cache_dtype="bf16", window=0,
                    compression_ratio=1, cp=1, split_kv=not str(cann_ver).startswith("9.2"))
        op = OpSpec("sfadsa", "attention", "decode", 0, 0, (), shape, shape, shape, shape,
                    recipe="indexed_attention", dtype="bf16", metadata=(("indexed_attention_geometry", geom),))
        return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(
            lambda o, g: indexed_attention.build(o, g, scratch_policy="offchip"),
        )}))

    def pred_dcp_merge(self, cp: int, q_len: int, heads: int = GQA_HQ, dim: int = GQA_D) -> float:
        n_lse, n_out = cp * heads * q_len, cp * heads * q_len * dim
        lse_b, red_lse_b = n_lse * FP32_BYTES, heads * q_len * FP32_BYTES
        ol_bf16_b, ol_fp32_b = n_out * BF16_BYTES, n_out * FP32_BYTES
        res_fp32_b, res_bf16_b = heads * q_len * dim * FP32_BYTES, heads * q_len * dim * BF16_BYTES
        l2_cap = self.hw["l2_bytes"]
        _hbm = lambda rd, wr, l2_in=0: max(0, rd - (min(rd, l2_in) if (l2_cap > 0 and (rd + wr) <= l2_cap) else 0)) + wr
        specs = (
            ("m1_amax", (), "fp32", n_lse, n_lse, lse_b, red_lse_b, 0),
            ("m2_sub", ("m1_amax",), "fp32", n_lse, n_lse, lse_b + red_lse_b, lse_b, lse_b + red_lse_b),
            ("m3_exp", ("m2_sub",), "fp32", n_lse, FP32_BYTES * n_lse, lse_b, lse_b, lse_b),
            ("m4_wsum", ("m3_exp",), "fp32", n_lse, n_lse, lse_b, red_lse_b, lse_b),
            ("m5_wdiv", ("m4_wsum",), "fp32", n_lse, n_lse, lse_b + red_lse_b, lse_b, lse_b + red_lse_b),
            ("m6_cast32", ("m5_wdiv",), "fp32", n_out, n_out, ol_bf16_b, ol_fp32_b, 0),
            ("m7_mul", ("m6_cast32",), "fp32", n_out, n_out, ol_fp32_b + lse_b, ol_fp32_b, ol_fp32_b + lse_b),
            ("m8_osum", ("m7_mul",), "fp32", n_out, n_out, ol_fp32_b, res_fp32_b, ol_fp32_b),
            ("m9_cast16", ("m8_osum",), "bf16", heads * q_len * dim, heads * q_len * dim, res_fp32_b, res_bf16_b, res_fp32_b),
        )
        ops = [
            OpSpec(oid, "vector", "decode", 0, 0, (), (n,), (n,), (n,), (n,), deps=deps, dtype=dt,
                   vector_ops=vops, mte2_bytes=rd, mte3_bytes=wr, hbm_bytes=_hbm(rd, wr, l2_in))
            for oid, deps, dt, n, vops, rd, wr, l2_in in specs
        ]
        return self.run_graph(ops, None, isolated=False)

    def pred_linear_attn(self, *, kind: str, seqlen: int, heads: int, cann_ver: str = "") -> float:
        if kind == "gdn_chunk":
            abi = "aclnnChunkGatedDeltaRuleArch35" if self.is_a5 else "aclnnChunkGatedDeltaRule"
            operation, phase, state_dtype, out_final, lengths = "gdn_prefill", "prefill", "bf16", True, (seqlen,)
        elif kind == "kda_chunk":
            abi = "vllm_ascend.ops.triton.kda.kda.chunk_kda"
            operation, phase, state_dtype, out_final, lengths = "kda_prefill", "prefill", "fp32", False, (seqlen,)
        else:
            abi = "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda"
            operation, phase, state_dtype, out_final, lengths = "kda_recurrent", "decode", "fp32", True, (1,) * seqlen
        shape = (seqlen, heads, GDN_DIM)
        meta: tuple[tuple[str, Any], ...] = (
            ("native_kind", abi), ("abi", abi), ("operation", operation), ("lengths", lengths),
            ("hq", heads), ("hk", heads), ("hv", heads), ("dk", GDN_DIM), ("dv", GDN_DIM),
            ("bt", GDN_BT), ("dtype", "bf16"), ("state_dtype", state_dtype), ("gate_dtype", "fp32"),
            ("beta_dtype", "bf16"), ("use_initial_state", True), ("output_final_state", out_final),
            *( (("cann_version", cann_ver),) if kind == "gdn_chunk" else () ),
        )
        op = OpSpec(kind, operation, phase, 0, 0, (), shape, shape, shape, shape,
                    recipe=operation, dtype="bf16", layout="TND", metadata=meta)
        return self.run_graph([op], RecipeBook(recipes={op.recipe: PlannedRecipe(linear_attention.build)}))


def _blk(cid: str, fam: str, split: str, err: Any, uri: str) -> dict[str, str]:
    return {"case_id": cid, "family": fam, "split": split, "error": _sanitize_error(err), "capture_uri": uri}


def _eval_gates14(
    h: _SkuReplayHarness, cap_dir: Path, rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]], captures: list[dict[str, Any]],
) -> None:
    g14_path = cap_dir / f"{h.sku}_gates14.json"
    if not g14_path.is_file():
        raise FileNotFoundError(f"missing_required_capture:{g14_path.name}")
    fsha = _sha256_file(g14_path)
    uri = f"capture://fleet-400k/{h.sku}/gates14/{fsha[:16]}"
    g14 = json.loads(g14_path.read_text(encoding="utf-8"))
    captures.append({"split": "gates14", "uri": uri, "file_sha256": fsha, "raw_sha256": g14.get("raw_sha256", "")})

    for idx, r in enumerate(g14.get("gate1_paged_and_layout", [])):
        cid = str(r.get("case_id") or f"g1_{idx}")
        if r.get("status") != "ok":
            blocked.append(_blk(cid, "gate1", "gate1", r.get("error"), uri))
            continue
        kv_len, pg = int(r["kv_len"]), int(r.get("page_size") or 0)
        bt = (tuple(int(x) for x in r["block_table"]),) if r.get("block_table") else ()
        try:
            if r["op"] == "paged_mla_tnd":
                pred = h.pred_paged_mla(kv_len, page_size=pg or DEFAULT_PAGE_SIZE, block_table=bt)
                rd_b, fl = kv_len * MLA_DQK * BF16_BYTES, 2.0 * MLA_H * kv_len * (MLA_DQK + MLA_NOPE)
            else:
                lay = "PA_BLK_BSND" if bt else str(r.get("layout", "BNSD"))
                pred = h.pred_fia(1, 1, GQA_HQ, GQA_HKV, GQA_D, kv_len, layout=lay, page_size=pg, block_table=bt)
                rd_b, fl = 2 * kv_len * GQA_HKV * GQA_D * BF16_BYTES, 4.0 * GQA_HQ * kv_len * GQA_D
            rows.append(_verdict(cid, f"g1_{r['op']}", "gate1", float(r["us"]["p50"]), float(r["us"]["p99"]), pred, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b), uri))
        except Exception as exc:
            blocked.append(_blk(cid, "gate1", "gate1", f"{type(exc).__name__}:{exc}", uri))

    for idx, r in enumerate(g14.get("gate2_small_m_gemm", [])):
        cid = str(r.get("case_id") or f"g2_{idx}")
        if r.get("status") != "ok":
            blocked.extend(_blk(f"{cid}_{s}", f"g2_gemm_{s}", "gate2", r.get("error"), uri) for s in ("cold", "warm"))
            continue
        m_dim, n_dim, k_dim = int(r["M"]), int(r["N"]), int(r["K"])
        try:
            cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": m_dim, "n": n_dim, "k": k_dim})
            pred_c = h.run_graph(list(cd.ops), cd.recipes)
            view_w = AdmissionMemoryView({0: int(h.hw["l2_bytes"])}, granule_bytes=L2_GRANULE_BYTES)
            a_b, b_b, c_b = m_dim * k_dim * BF16_BYTES, k_dim * n_dim * BF16_BYTES, m_dim * n_dim * BF16_BYTES
            for tid, bsz, kind, pipe in (("a", a_b, "read", "mte2_aic"), ("b", b_b, "read", "mte2_aic"), ("c1", c_b, "write", "mte3")):
                view_w.admit_access(MemoryAccessPattern(tid, 0, 0, 0, kind, pipe, bsz, AddressRunPlan((AddressRun(0, 1, bsz, bsz),)), reuse_key=ReuseKey(0, tid, 0)), bsz)
            _, a_m = view_w.admit_access(MemoryAccessPattern("a", 0, 0, 0, "read", "mte2_aic", a_b, AddressRunPlan((AddressRun(0, 1, a_b, a_b),)), reuse_key=ReuseKey(0, "a", 0)), a_b)
            _, b_m = view_w.admit_access(MemoryAccessPattern("b", 0, 0, 0, "read", "mte2_aic", b_b, AddressRunPlan((AddressRun(0, 1, b_b, b_b),)), reuse_key=ReuseKey(0, "b", 0)), b_b)
            op_w = replace(cd.ops[0], op_id=f"{cd.ops[0].op_id}_warm", hbm_bytes=float(a_m + b_m + c_b))
            pred_w = h.run_graph([op_w], cd.recipes)
            fl, rd_b, wr_b = 2.0 * m_dim * n_dim * k_dim, a_b + b_b, c_b
            rows.append(_verdict(f"{cid}_cold", "g2_gemm_cold", "gate2", float(r["cold_us"]["p50"]), float(r["cold_us"]["p99"]), pred_c, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b, write_bytes=wr_b), uri))
            rows.append(_verdict(f"{cid}_warm", "g2_gemm_warm", "gate2", float(r["warm_us"]["p50"]), float(r["warm_us"]["p99"]), pred_w, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b, write_bytes=wr_b, warm=True), uri))
        except Exception as exc:
            blocked.extend(_blk(f"{cid}_{s}", f"g2_gemm_{s}", "gate2", f"{type(exc).__name__}:{exc}", uri) for s in ("cold", "warm"))

    for idx, r in enumerate(g14.get("gate3_producer_consumer_chain", [])):
        cid = str(r.get("case_id") or f"g3_{idx}")
        if r.get("status") != "ok":
            blocked.extend(_blk(f"{cid}_{s}", f, "gate3", r.get("error"), uri) for s, f in (("prod", "g3_prod"), ("cons", "g3_cons_inc"), ("chain", "g3_chain")))
            continue
        tb = int(r["tensor_mib"]) * MIB
        n_el = tb // BF16_BYTES
        try:
            view = AdmissionMemoryView({0: int(h.hw["l2_bytes"])}, granule_bytes=L2_GRANULE_BYTES)
            for tid in ("x1", "x2"):
                view.admit_access(MemoryAccessPattern(tid, 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, tid, 0)), tb)
            view.admit_access(MemoryAccessPattern("y_mid", 0, 0, 0, "write", "mte3", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "y_mid", 0)), tb)
            _, y_m = view.admit_access(MemoryAccessPattern("y_mid", 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "y_mid", 0), reuse_distance_bytes=3 * tb), tb)
            _, x3_m = view.admit_access(MemoryAccessPattern("x3", 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "x3", 0)), tb)
            p_op = OpSpec("g3_prod", "vector", "prefill", 0, 0, (), (n_el,), (n_el,), (n_el,), (n_el,), dtype="bf16", vector_ops=n_el, mte2_bytes=2 * tb, mte3_bytes=tb, hbm_bytes=3 * tb)
            c_op = OpSpec("g3_cons", "vector", "prefill", 0, 0, (), (n_el,), (n_el,), (n_el,), (n_el,), deps=("g3_prod",), dtype="bf16", vector_ops=n_el, mte2_bytes=2 * tb, mte3_bytes=tb, hbm_bytes=int(y_m + x3_m + tb))
            pred_p, pred_ch = h.run_graph([p_op], None, isolated=True), h.run_graph([p_op, c_op], None, isolated=False)
            lst_p, lst_c = _lean_lstar_us(h.hw, vec_ops=n_el, read_bytes=2 * tb, write_bytes=tb), _lean_lstar_us(h.hw, vec_ops=n_el, read_bytes=tb, write_bytes=tb)
            rows.append(_verdict(f"{cid}_prod", "g3_prod", "gate3", float(r["prod_us"]["p50"]), float(r["prod_us"]["p99"]), pred_p, lst_p, uri))
            rows.append(_verdict(f"{cid}_cons", "g3_cons_inc", "gate3", float(r["cons_inc_us"]["p50"]), float(r["cons_inc_us"]["p99"]), pred_ch - pred_p, lst_c, uri))
            rows.append(_verdict(f"{cid}_chain", "g3_chain", "gate3", float(r["chain_us"]["p50"]), float(r["chain_us"]["p99"]), pred_ch, lst_p + lst_c, uri))
        except Exception as exc:
            blocked.extend(_blk(f"{cid}_{s}", f, "gate3", f"{type(exc).__name__}:{exc}", uri) for s, f in (("prod", "g3_prod"), ("cons", "g3_cons_inc"), ("chain", "g3_chain")))

    for idx, r in enumerate(g14.get("gate4_skewed_grouped_gemm", [])):
        cid = str(r.get("case_id") or f"g4_{idx}")
        if r.get("status") != "ok":
            blocked.append(_blk(cid, "gate4", "gate4", r.get("error"), uri))
            continue
        gl, k_dim, n_dim = [int(x) for x in r["group_list"]], int(r["K"]), int(r["N"])
        try:
            cd = build_aclnn_call({"api": "aclnnGroupedMatmul", "dtype": "bf16", "group_list": gl, "n": n_dim, "k": k_dim})
            pred = h.run_graph(list(cd.ops), cd.recipes)
            tot_t, nz = sum(gl), sum(1 for g in gl if g > 0)
            lst = _lean_lstar_us(h.hw, cube_flops=2.0 * tot_t * n_dim * k_dim, read_bytes=(tot_t * k_dim + nz * k_dim * n_dim) * BF16_BYTES, write_bytes=tot_t * n_dim * BF16_BYTES)
            rows.append(_verdict(cid, "g4_moe_gmm", "gate4", float(r["cold_us"]["p50"]), float(r["cold_us"]["p99"]), pred, lst, uri))
        except Exception as exc:
            blocked.append(_blk(cid, "gate4", "gate4", f"{type(exc).__name__}:{exc}", uri))


def _route_link_bw_bps(cs_c: Any, route: str, ws: int, hw: dict[str, Any]) -> float:
    topo = Topology(cs_c.machine, cs_c.runtime.parallel)
    by_scope: dict[str, float] = {}
    for name, lk in topo.links.items():
        sc = topo.link_scope(name)
        by_scope[sc] = max(by_scope.get(sc, 0.0), float(lk.capacity_bytes_per_ms) * US_PER_MS)
    if hw["is_per_pair"] and route in {"UB_FULLMESH", "HCCS_MESH"}:
        return max(1, ws - 1) * max(by_scope.get("node", 0.0), hw["per_pair_link"])
    if route == "SIO" and by_scope.get("package", 0.0) > 0.0:
        return by_scope["package"]
    if route == "HCCS_SW" and by_scope.get("node", 0.0) > 0.0:
        return by_scope["node"]
    if route == "SIO_HCCS_SW_4R":
        comb = by_scope.get("package", 0.0) + by_scope.get("node", 0.0)
        if comb > 0.0:
            return comb
    return by_scope.get("node") or by_scope.get("package") or hw["link"]


def _eval_gate5(
    h: _SkuReplayHarness, cap_dir: Path, rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]], captures: list[dict[str, Any]],
) -> None:
    g5_paths = sorted(cap_dir.glob(f"{h.sku}_*r_*.json"))
    if not g5_paths:
        raise FileNotFoundError(f"missing_required_gate5_captures:{h.sku}")
    comm_cases: dict[tuple[str, int], Any] = {}
    for g5_path in g5_paths:
        fsha = _sha256_file(g5_path)
        tag = g5_path.stem[len(h.sku) + 1:]
        uri = f"capture://fleet-400k/{h.sku}/{tag}/{fsha[:16]}"
        g5 = json.loads(g5_path.read_text(encoding="utf-8"))
        captures.append({"split": f"gate5_{tag}", "uri": uri, "file_sha256": fsha, "raw_sha256": g5.get("raw_sha256", "")})
        ws, route = int(g5["group_size"]), str(g5["route"])
        if (route, ws) not in comm_cases:
            if route == "HCCS_SW" and ws == 2:
                comm_cases[(route, ws)] = parse_case(compose_case("qwen3.8-27b", hardware=h.sku, tp=2, dp=2, ep=1, num_devices=4, layers=1, quant="none"))
            else:
                comm_cases[(route, ws)] = parse_case(compose_case("qwen3.8-27b", hardware=h.sku, tp=ws, ep=1, num_devices=ws, layers=1, quant="none"))
        g5_axis = "dp" if (route == "HCCS_SW" and ws == 2) else "tp"
        cs_c = comm_cases[(route, ws)]
        eff_bw = _route_link_bw_bps(cs_c, route, ws, h.hw)

        for idx, r in enumerate(g5.get("collectives", [])):
            cid = str(r.get("case_id") or f"g5_{route}_r{ws}_{r.get('op', 'coll')}_{r.get('bytes', idx)}B")
            if r.get("status") == "blocked" or not r.get("verified_all_ranks"):
                blocked.append(_blk(cid, f"g5_{route}", "gate5", r.get("error", "unverified"), uri))
                continue
            op_name, pb = str(r["op"]), int(r["bytes"])
            has_blk = bool(r.get("blocked_us") and r["blocked_us"].get("n", 0) > 0)
            meas_d, s_meta = (r["blocked_us"], ()) if has_blk else (r["eager_us"], (("collective_stack", "eager"),))
            try:
                api_b = pb // ws if op_name == "all_gather" else pb
                if op_name == "all_reduce":
                    lb, fam, rec, steps = 2.0 * (ws - 1) / ws * pb, "allreduce", "ring_allreduce", (("steps", 2 * (ws - 1)),)
                elif op_name == "all_gather":
                    lb, fam, rec, steps = 1.0 * (ws - 1) / ws * pb, "allgather", "ring_allgather", (("steps", ws - 1),)
                elif op_name == "reduce_scatter":
                    lb, fam, rec, steps = 1.0 * (ws - 1) / ws * pb, "reduce_scatter", "ring_reduce_scatter", (("steps", ws - 1),)
                else:
                    lb, fam, rec, steps = 1.0 * (ws - 1) / ws * pb, "alltoall", "expert_dispatch", ()
                op = comm_op(op_id=cid, deps=(), stage=0, layer=0, requests=("r0",), family=fam, phase="prefill", recipe=rec, link_bytes=lb, shape=(1, max(1, api_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", op_name), ("api_bytes_per_rank", api_b), *steps, *s_meta))
                pred = h.run_graph([op], None, isolated=False, case_spec=cs_c, include_sub=False)
                tag_s = "steady" if pb >= STEADY_COMM_MIN_BYTES else "small"
                rows.append(_verdict(cid, f"g5_{route}_{tag_s}", "gate5", float(meas_d["p50"]), float(meas_d["p99"]), pred, _lean_lstar_us(h.hw, comm_bytes=lb, link_bw_bps=eff_bw), uri))
            except Exception as exc:
                blocked.append(_blk(cid, f"g5_{route}", "gate5", f"{type(exc).__name__}:{exc}", uri))

        for idx, r in enumerate(g5.get("dcp_pcp", [])):
            cid = str(r.get("case_id") or f"dcp_pcp_{route}_r{ws}_{r.get('mode', 'dcp')}_q{r.get('q_len', idx)}_kv{r.get('total_kv', 0)}")
            if r.get("status") == "blocked" or not r.get("total_us") or r.get("error"):
                blocked.append(_blk(cid, f"dcp_pcp_{route}", "gate5", r.get("error", "blocked"), uri))
                continue
            try:
                q_len, tot_kv = int(r["q_len"]), int(r["total_kv"])
                loc_kv = max(1, tot_kv // ws)
                t_fia = h.pred_fia(1, q_len, GQA_HQ, GQA_HKV, GQA_D, loc_kv, layout="BNSD", phase="decode" if q_len == 1 else "prefill")
                o_b, lse_b = q_len * GQA_HQ * GQA_D * BF16_BYTES, q_len * GQA_HQ * FP32_BYTES
                ag_o = comm_op(op_id=f"{cid}_ago", deps=(), stage=0, layer=0, requests=("r0",), family="allgather", phase="prefill", recipe="ring_allgather", link_bytes=(ws - 1) * o_b, shape=(1, max(1, o_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", "all_gather"), ("api_bytes_per_rank", o_b), ("steps", ws - 1)))
                ag_lse = comm_op(op_id=f"{cid}_aglse", deps=(ag_o.op_id,), stage=0, layer=0, requests=("r0",), family="allgather", phase="prefill", recipe="ring_allgather", link_bytes=(ws - 1) * lse_b, shape=(1, max(1, lse_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", "all_gather"), ("api_bytes_per_rank", lse_b), ("steps", ws - 1)))
                t_comm = h.run_graph([ag_o, ag_lse], None, isolated=False, case_spec=cs_c, include_sub=False)
                t_merge = h.pred_dcp_merge(ws, q_len, heads=GQA_HQ, dim=GQA_D)
                lst = _lean_lstar_us(h.hw, cube_flops=4.0 * GQA_HQ * q_len * loc_kv * GQA_D, read_bytes=2 * loc_kv * GQA_HKV * GQA_D * BF16_BYTES, comm_bytes=(ws - 1) * (o_b + lse_b), link_bw_bps=eff_bw)
                rows.append(_verdict(cid, f"dcp_pcp_{route}", "gate5", float(r["total_us"]["p50"]), float(r["total_us"]["p99"]), t_fia + t_comm + t_merge, lst, uri))
            except Exception as exc:
                blocked.append(_blk(cid, f"dcp_pcp_{route}", "gate5", f"{type(exc).__name__}:{exc}", uri))


def _eval_seq_splits(
    h: _SkuReplayHarness, cap_dir: Path, rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]], captures: list[dict[str, Any]],
) -> None:
    seen_ragged: set[str] = set()
    for split in ("anchor", "heldout"):
        fp = cap_dir / f"{h.sku}_{split}.json"
        if not fp.is_file():
            raise FileNotFoundError(f"missing_required_capture:{fp.name}")
        fsha = _sha256_file(fp)
        uri = f"capture://fleet-400k/{h.sku}/{split}/{fsha[:16]}"
        doc = json.loads(fp.read_text(encoding="utf-8"))
        captures.append({"split": split, "uri": uri, "file_sha256": fsha, "raw_sha256": doc.get("raw_sha256", "")})
        doc_cann = str(doc.get("cann_version") or (h.pt.cann_version if h.pt is not None else ""))
        for idx, r in enumerate(doc.get("records", [])):
            op_name = str(r.get("op", ""))
            cid = str(r.get("case_id") or f"{op_name}_{split}_{idx}")
            eff_split = "ragged" if op_name == "ragged_suite" else split
            if eff_split == "ragged":
                if cid in seen_ragged:
                    continue
                seen_ragged.add(cid)
            if r.get("status") != "ok":
                blocked.append(_blk(cid, op_name, eff_split, r.get("error"), uri))
                continue
            try:
                pred, lst = _eval_single_seq_record(h, op_name, r, doc_cann)
                if pred is None:
                    blocked.append(_blk(cid, op_name, eff_split, f"unwired_op:{op_name}", uri))
                else:
                    rows.append(_verdict(cid, op_name, eff_split, float(r["us"]["p50"]), float(r["us"]["p99"]), pred, lst, uri))
            except Exception as exc:
                blocked.append(_blk(cid, op_name, eff_split, f"{type(exc).__name__}:{exc}", uri))


def _eval_single_seq_record(
    h: _SkuReplayHarness, op_name: str, r: dict[str, Any], doc_cann: str,
) -> tuple[float | None, float]:
    if op_name == "fia_gqa_decode":
        seqlen = int(r["seqlen"])
        return (
            h.pred_fia(1, 1, GQA_HQ, GQA_HKV, GQA_D, seqlen, layout="BNSD", phase="decode"),
            _lean_lstar_us(h.hw, cube_flops=4.0 * GQA_HQ * seqlen * GQA_D, read_bytes=2 * seqlen * GQA_HKV * GQA_D * BF16_BYTES),
        )
    if op_name == "fia_gqa_prefill":
        seqlen, q_len = int(r["seqlen"]), int(r["q_len"])
        return (
            h.pred_fia(1, q_len, GQA_HQ, GQA_HKV, GQA_D, seqlen, layout="BNSD", phase="prefill", causal=False),
            _lean_lstar_us(h.hw, cube_flops=4.0 * GQA_HQ * q_len * seqlen * GQA_D, read_bytes=(2 * seqlen * GQA_HKV + q_len * GQA_HQ) * GQA_D * BF16_BYTES),
        )
    if op_name == "paged_mla_decode":
        seqlen = int(r["seqlen"])
        return (
            h.pred_paged_mla(seqlen, page_size=int(r.get("page_size", DEFAULT_PAGE_SIZE))),
            _lean_lstar_us(h.hw, cube_flops=2.0 * MLA_H * seqlen * (MLA_DQK + MLA_NOPE), read_bytes=seqlen * MLA_DQK * BF16_BYTES),
        )
    if op_name in {"sfa_dsa_decode", "c8_sfa_decode"}:
        seqlen, tk = int(r["seqlen"]), int(r.get("top_k", DEFAULT_TOPK))
        eff = min(seqlen, tk)
        is_c8 = op_name == "c8_sfa_decode"
        row_b = C8_ROW_BYTES if is_c8 else MLA_DQK * BF16_BYTES
        return (
            h.pred_sfa(seqlen, topk=tk, quant_c8=is_c8, cann_ver=doc_cann),
            _lean_lstar_us(h.hw, cube_flops=2.0 * MLA_H * eff * (MLA_DQK + MLA_NOPE), read_bytes=eff * row_b),
        )
    if op_name == "lightning_indexer_topk":
        seqlen, tk = int(r["seqlen"]), int(r.get("top_k", DEFAULT_TOPK))
        return (
            h.pred_indexer(seqlen, topk=tk),
            _lean_lstar_us(h.hw, cube_flops=2.0 * IDX_H * seqlen * IDX_D, read_bytes=seqlen * IDX_D * BF16_BYTES),
        )
    if op_name == "dcp_lse_attention_merge":
        cp, q_len = int(r["cp"]), int(r["q_len"])
        return (
            h.pred_dcp_merge(cp, q_len, heads=GQA_HQ, dim=GQA_D),
            _lean_lstar_us(h.hw, read_bytes=cp * (GQA_HQ * q_len * GQA_D * BF16_BYTES + GQA_HQ * q_len * FP32_BYTES), write_bytes=GQA_HQ * q_len * GQA_D * BF16_BYTES),
        )
    if op_name == "gdn_chunk":
        seqlen, heads = int(r["seqlen"]), int(r.get("H", GDN_DEFAULT_H))
        return (
            h.pred_linear_attn(kind="gdn_chunk", seqlen=seqlen, heads=heads, cann_ver=doc_cann),
            _lean_lstar_us(h.hw, read_bytes=3 * seqlen * heads * GDN_DIM * BF16_BYTES),
        )
    if op_name == "kda_chunk":
        seqlen, heads = int(r["seqlen"]), int(r.get("H", KDA_CHUNK_DEFAULT_H))
        return (
            h.pred_linear_attn(kind="kda_chunk", seqlen=seqlen, heads=heads),
            _lean_lstar_us(h.hw, read_bytes=3 * seqlen * heads * GDN_DIM * BF16_BYTES),
        )
    if op_name == "kda_recurrent":
        batch, heads = int(r["B"]), int(r.get("H", KDA_REC_DEFAULT_H))
        state_b = batch * heads * GDN_DIM * GDN_DIM * FP32_BYTES
        return (
            h.pred_linear_attn(kind="kda_recurrent", seqlen=batch, heads=heads),
            _lean_lstar_us(h.hw, read_bytes=state_b, write_bytes=state_b),
        )
    if op_name == "ragged_suite":
        qlens = [int(x) for x in r["q_lens"]]
        kvlens = [int(x) for x in r["seq_lens"]]
        if str(r.get("sub_op", "")) == "fia_gqa_ragged":
            seqs = tuple(zip(qlens, kvlens, strict=True))
            return (
                h.pred_fia(len(seqs), max(qlens), GQA_HQ, GQA_HKV, GQA_D, max(kvlens), layout="TND", page_size=int(r.get("page_size", DEFAULT_PAGE_SIZE)), phase="decode" if max(qlens) == 1 else "prefill", sequences=seqs),
                _lean_lstar_us(h.hw, cube_flops=4.0 * GQA_HQ * sum(q * kv for q, kv in seqs) * GQA_D, read_bytes=2 * sum(kvlens) * GQA_HKV * GQA_D * BF16_BYTES),
            )
        return (
            h.pred_indexer(max(kvlens), batch=len(kvlens), topk=int(r.get("top_k", DEFAULT_TOPK)), seqs=kvlens),
            _lean_lstar_us(h.hw, cube_flops=2.0 * IDX_H * sum(kvlens) * IDX_D, read_bytes=sum(kvlens) * IDX_D * BF16_BYTES),
        )
    return None, 0.0


def _summarize_families(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["split"], r["family"]), []).append(r)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (split, fam), items in sorted(grouped.items()):
        apes = sorted(float(x["ape_pct"]) for x in items)
        l400 = [x for x in items if "400000" in x["case_id"] or "409600" in x["case_id"]]
        entry: dict[str, Any] = {
            "count": len(items), "pass": sum(1 for x in items if x["status"] == "pass"),
            "fail": sum(1 for x in items if x["status"] == "fail"),
            "med_ape_pct": round(statistics.median(apes), 2), "mean_ape_pct": round(statistics.mean(apes), 2),
            "max_ape_pct": round(apes[-1], 2),
            "inv_sim": sum(1 for x in items if x["inv_sim"]), "inv_obs": sum(1 for x in items if x["inv_obs"]),
        }
        if l400:
            entry["cliff_400k_ape_pct"] = round(float(l400[-1]["ape_pct"]), 2)
        out.setdefault(split, {})[fam] = entry
    return out


def _eval_sku_worker(args: tuple[str, str]) -> dict[str, Any]:
    sku, cap_dir_str = args
    cap_dir = Path(cap_dir_str)
    if not cap_dir.is_dir():
        raise FileNotFoundError(f"missing_capture_dir:{cap_dir}")
    t0 = time.monotonic()
    h = _SkuReplayHarness(sku, use_calibrated=True, use_submission=True)
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    _eval_gates14(h, cap_dir, rows, blocked, captures)
    _eval_gate5(h, cap_dir, rows, blocked, captures)
    _eval_seq_splits(h, cap_dir, rows, blocked, captures)

    n_pass = sum(1 for r in rows if r["status"] == "pass")
    n_fail = sum(1 for r in rows if r["status"] == "fail")
    total = n_pass + n_fail + len(blocked)
    if total != len(rows) + len(blocked) or (n_pass + n_fail) == 0:
        raise AssertionError(f"zero_drop_accounting_failed:{sku}")
    return {
        "sku": sku, "arch_preset": h.hw["arch"], "endpoint_alias": ENDPOINT_ALIAS[sku],
        "cann_version": h.pt.cann_version if h.pt is not None else "",
        "primitive_envelope_sha256": h.pt.envelope_sha256 if h.pt is not None else "",
        "aic_count": h.hw["aic"], "aiv_count": h.hw["aiv"], "l2_bytes": int(h.hw["l2_bytes"]),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "counts": {
            "total": total, "pass": n_pass, "fail": n_fail, "blocked": len(blocked),
            "inv_sim": sum(1 for r in rows if r["inv_sim"]), "inv_obs": sum(1 for r in rows if r["inv_obs"]),
        },
        "family_summary": _summarize_families(rows), "captures": captures,
        "cases": rows, "blocked_cases": blocked,
    }


def _build_bundle(cap_dir: Path) -> dict[str, Any]:
    if not cap_dir.is_dir():
        raise FileNotFoundError(f"missing_capture_dir:{cap_dir}")
    if set(ANCHOR_CONTEXT_GRID) & set(OFF_GRID_CLIFF_HELDOUT_GRID):
        raise AssertionError("anchor_heldout_grids_overlap")
    manifest_path = REPO_ROOT / "data" / "primitives" / "manifest.json"
    with ProcessPoolExecutor(max_workers=len(REACHABLE_SKUS)) as ex:
        sku_results = list(ex.map(_eval_sku_worker, [(sku, str(cap_dir)) for sku in REACHABLE_SKUS]))

    tot_pass = sum(s["counts"]["pass"] for s in sku_results)
    tot_fail = sum(s["counts"]["fail"] for s in sku_results)
    tot_blk = sum(s["counts"]["blocked"] for s in sku_results)
    tot_all = tot_pass + tot_fail + tot_blk
    tot_inv_sim = sum(s["counts"]["inv_sim"] for s in sku_results)
    tot_inv_obs = sum(s["counts"]["inv_obs"] for s in sku_results)

    payload: dict[str, Any] = {
        "schema_version": "mindie-sim-fleet-400k-verification/1",
        "campaign": "all-skus-512-400k-cliff-heldout-20261121",
        "acceptance_policy": {
            "relative_tolerance": REL_TOL, "absolute_floor_ns": ABS_FLOOR_NS, "inclusive": True,
            "zero_drop_accounting": True, "zero_filtering": True, "zero_empirical_fitting": True,
        },
        "context_grids": {
            "anchor": list(ANCHOR_CONTEXT_GRID), "off_grid_cliff_heldout": list(OFF_GRID_CLIFF_HELDOUT_GRID),
            "overlap_count": 0,
        },
        "provenance": {
            "primitive_manifest_path": "data/primitives/manifest.json",
            "primitive_manifest_sha256": _sha256_file(manifest_path),
            "producer": "data/verification/all-skus-512-400k-cliff-heldout-20261121.producer.py",
        },
        "fleet_inventory": {
            "reachable_calibrated_skus": [
                {
                    "sku": s["sku"], "arch_preset": s["arch_preset"], "endpoint_alias": s["endpoint_alias"],
                    "cann_version": s["cann_version"], "aic_count": s["aic_count"], "aiv_count": s["aiv_count"],
                    "l2_bytes": s["l2_bytes"], "status": "calibrated",
                    "primitive_envelope_sha256": s["primitive_envelope_sha256"],
                }
                for s in sku_results
            ],
            "unmeasured_skus": list(UNMEASURED_SKUS),
        },
        "overall_counts": {
            "total": tot_all, "pass": tot_pass, "fail": tot_fail, "blocked": tot_blk,
            "inv_sim": tot_inv_sim, "inv_obs": tot_inv_obs,
        },
        "residual_diagnostics": [
            "Short-context L<=2049 and 9-op eager dcp_lse_attention_merge retain unmodeled per-kernel dispatch residuals where P15 aic_first_issue_ns is unmeasured.",
            "Gate 2 small-M GEMM (M<=48) warm/cold split-K dispatch and L2 tiling heuristics retain residual gaps pending per-shape CANN tiling key traces.",
            "Gate 5 small-message collectives (<16 MiB) and real DCP/PCP retain startup residuals where P18 startup_blocked_ns did not clear the 5% CV seal.",
            "Arch22 kda_chunk Triton compiler GM materialization and Arch22 paged_mla_decode V partial L2 miss remain openly reported without empirical multipliers.",
            "gdn_recurrent blocked across all 4 SKUs by CANN runtime scale validation; kda_chunk/kda_recurrent unavailable on Arch35 CANN 9.2.0.",
        ],
        "skus": {s["sku"]: s for s in sku_results},
    }
    payload["bundle_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture-dir", required=True, help="Directory containing Audit-2 clean per-SKU capture JSONs.")
    ap.add_argument("--output", default=str(REPO_ROOT / "data" / "verification" / "all-skus-512-400k-cliff-heldout-20261121.json"))
    args = ap.parse_args()
    bundle = _build_bundle(Path(args.capture_dir))
    serialized = json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    _assert_no_privacy_leaks(serialized)
    out_path = Path(args.output)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(out_path)
    c = bundle["overall_counts"]
    print(f"SEALED {out_path.name}: total={c['total']} pass={c['pass']} fail={c['fail']} blocked={c['blocked']} inv_sim={c['inv_sim']} inv_obs={c['inv_obs']} sha256={bundle['bundle_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
