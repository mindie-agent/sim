"""Produce sanitized all-SKU 512..400K Anchor + Off-Grid Cliff Heldout + 5-Gate bundle.

[Role] Replay live NPU silicon captures across all reachable Ascend SKUs
  (Ascend910B4, Ascend910_9362, Ascend910_9382, Ascend950DT_9582) plus explicit
  unmeasured inventory for Ascend950PR_9579 through the white-box Single-Spine
  simulator with calibrated L0 primitive tables and emit a SHA-256-sealed bundle.
[Invariants]
  * Every capture JSON MUST pass canonical raw_sha256 verification before replay.
  * Replay scales core-clock-coupled machine peaks/rates to capture-recorded
    observed_aicore_freq_mhz before building EvaluationContext and Lean L* bounds.
  * Strict Anchor/Heldout disjointness by actual geometry; exact multiset
    reconciliation over (acquisition_uid, case_id, boundary) with zero drops.
  * Inclusive full float64 tolerance: |pred_ns - meas_ns| <= max(0.10 * meas_ns, 2000.0),
    mandatory finite > 0 duration & Lean L* > 0, and 0 inversions against P50/P99.
  * Zero privacy leaks: no IPv4 addresses, hostnames, usernames, or home paths.
[Timeouts & Locks] Parallel 4-worker ProcessPoolExecutor; read-only capture I/O;
  writes only the caller-requested output JSON atomically via temp+replace.
[Side-Effect Policy] Pure white-box replay; zero empirical fitting or mutation of
  calibrated primitive tables or hardware profiles.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
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
from mindie_sim.optimization.dual_bound import (
    EventDagSpec, ResourceChoiceProgram, certify_lower_bound,
    check_optimality_certificate, inward_floor_ns,
)
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import attention, linear_attention, paged_mla, sparse_indexer
from mindie_sim.prediction.operators.attention._fia import AttentionContract
from mindie_sim.prediction.operators.attention.common import C8_V1_ABI, QUANT_MODE_BF16, QUANT_MODE_INT8_C8, SFA_V1_ABI
from mindie_sim.prediction.operators.context_parallel import DCP_LSE_MERGE_EAGER_RECIPE, recipes as cp_recipes
from mindie_sim.prediction.operators.paged_mla import indexed_attention, indexed_quant_attention
from mindie_sim.prediction.operators.paged_mla.common import PagedMlaContract
from mindie_sim.prediction.operators.sparse_indexer.common import SparseIndexerContract
from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
from mindie_sim.prediction.runtime.memory_physics import AddressRun, AddressRunPlan, MemoryAccessPattern, ReuseKey
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from tools._fleet_400k_reconcile import (
    ABS_FLOOR_NS, DIGEST_PREFIX_LEN, L1_CLIFF_ABS_FLOOR_NS, L1_CLIFF_REL_TOL,
    L1_MEDIAN_APE_MAX_PCT, L1_P90_APE_MAX_PCT, NS_PER_US, REL_TOL,
    apply_sku_observed_clock, dcp_reference_verified, extract_observed_clock_mhz,
    is_worker_diagnostic_record, load_verified_capture, reconcile_sku_accounting,
    score_verdict, seq_geometry_key,
)

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

US_PER_MS, BYTES_PER_GB, MIB, NS_PER_S = 1000.0, 1e9, 1024 * 1024, 1_000_000_000
STEADY_COMM_MIN_BYTES = 16 * MIB
BF16_BYTES, FP32_BYTES, QKV_OPERANDS, PAIR_FACTOR, FMA_PAIR_FACTOR = 2, 4, 3, 2, 4
GQA_HQ, GQA_HKV, GQA_D = 32, 8, 128
MLA_H, MLA_NOPE, MLA_ROPE, C8_ROW_BYTES = 128, 512, 64, 656
MLA_DQK = MLA_NOPE + MLA_ROPE
IDX_H, IDX_D, DEFAULT_TOPK, DEFAULT_PAGE_SIZE, L2_GRANULE_BYTES = 64, 128, 2048, 128, 128
GDN_DEFAULT_H, GDN_REC_DEFAULT_H, KDA_CHUNK_DEFAULT_H, KDA_REC_DEFAULT_H = 4, 16, 2, 8
GDN_BT, GDN_DIM, MAX_ERR_CHARS, ROUND_SUMMARY_DIGITS = 64, 128, 160, 2
P90_QUANTILE = 0.9
STARVED_CV_PCT = 25.0
STARVED_SMALL_MAX_BYTES = 64 * 1024
STARVED_BLOCKED_FLOOR_US = 50.0
STARVED_SIO_ROUTES: frozenset[str] = frozenset({"SIO", "SIO_HCCS_SW_4R"})
DEFAULT_FALLBACK_LINK_GBPS = 25.0
TWO_RANK_WORLD, FOUR_RANK_WORLD = 2, 4
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


def _capture_meta(split: str, uri: str, fsha: str, doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "split": split, "uri": uri, "acquisition_uid": str(doc.get("acquisition_uid") or ""),
        "file_sha256": fsha, "raw_sha256": str(doc.get("raw_sha256") or ""),
        "inner_digest_verified": True,
    }
    if doc.get("prior_raw_sha256"):
        out["prior_raw_sha256"] = str(doc["prior_raw_sha256"])
    return out


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
    max_link_gbps = max(d2d_bws) if any(b > 0.0 for b in d2d_bws) else float(d2d.get("uni_peak_bw_gbps") or DEFAULT_FALLBACK_LINK_GBPS)
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
    return inward_floor_ns(Fraction(work) * Fraction(NS_PER_S) / Fraction(cap_per_s))


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


def _admit_warm_gemm_l2_hits(l2_bytes: int, a_b: int, b_b: int, c_b: int) -> tuple[int, float]:
    """Compute independently admitted L2 read hit bytes via AdmissionMemoryView."""
    view = AdmissionMemoryView({0: l2_bytes}, granule_bytes=L2_GRANULE_BYTES)
    for tid, bsz, kind, pipe in (("a", a_b, "read", "mte2_aic"), ("b", b_b, "read", "mte2_aic"), ("c1", c_b, "write", "mte3")):
        view.admit_access(MemoryAccessPattern(tid, 0, 0, 0, kind, pipe, bsz, AddressRunPlan((AddressRun(0, 1, bsz, bsz),)), reuse_key=ReuseKey(0, tid, 0)), bsz)
    _, a_m = view.admit_access(MemoryAccessPattern("a", 0, 0, 0, "read", "mte2_aic", a_b, AddressRunPlan((AddressRun(0, 1, a_b, a_b),)), reuse_key=ReuseKey(0, "a", 0)), a_b)
    _, b_m = view.admit_access(MemoryAccessPattern("b", 0, 0, 0, "read", "mte2_aic", b_b, AddressRunPlan((AddressRun(0, 1, b_b, b_b),)), reuse_key=ReuseKey(0, "b", 0)), b_b)
    miss_total = int(a_m + b_m + c_b)
    admitted_read_hits = float(max(0, (a_b + b_b) - int(a_m + b_m)))
    return miss_total, admitted_read_hits


class _SkuReplayHarness:
    """White-box single-spine replay harness for one SKU at its observed board clock."""

    def __init__(
        self, sku: str, *, observed_clock_mhz: float | None = None,
        use_calibrated: bool = True, use_submission: bool = True,
    ) -> None:
        self.sku, self.use_submission = sku, use_submission
        self.pt = load_primitives(sku) if use_calibrated else None
        self.observed_clock_mhz = observed_clock_mhz
        raw_case = compose_case("qwen3.8-27b", hardware=sku, tp=1, ep=1, num_devices=1, layers=1, quant="none")
        self.cs = parse_case(apply_sku_observed_clock(raw_case, observed_clock_mhz))
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

    def pred_dcp_merge(
        self, cp: int, q_len: int, heads: int = GQA_HQ, dim: int = GQA_D, *, include_stack: bool = False,
    ) -> float:
        shape = (q_len, heads, dim)
        op = OpSpec(
            "dcp_lse_merge", "attention_merge", "decode", 0, 0, (),
            shape, shape, shape, shape,
            recipe=DCP_LSE_MERGE_EAGER_RECIPE, dtype="bf16",
            metadata=(
                ("partial_count", cp),
                ("l2_bytes", int(self.hw["l2_bytes"])),
                ("include_stack", include_stack),
            ),
        )
        return self.run_graph(
            [op],
            RecipeBook(recipes={DCP_LSE_MERGE_EAGER_RECIPE: cp_recipes()[DCP_LSE_MERGE_EAGER_RECIPE]}),
            isolated=False,
        )

    def pred_linear_attn(self, *, kind: str, seqlen: int, heads: int, cann_ver: str = "") -> float:
        if kind == "gdn_chunk":
            abi = "aclnnChunkGatedDeltaRuleArch35" if self.is_a5 else "aclnnChunkGatedDeltaRule"
            operation, phase, state_dtype, out_final, lengths = "gdn_prefill", "prefill", "bf16", True, (seqlen,)
        elif kind == "gdn_recurrent":
            abi = "aclnnRecurrentGatedDeltaRule"
            operation, phase, state_dtype, out_final, lengths = "gdn_recurrent", "decode", "bf16", True, (1,) * seqlen
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


def _blk(
    cid: str, fam: str, split: str, err: Any, uri: str,
    *, acquisition_uid: str = "", boundary: str = "cold", geometry_key: str = "",
) -> dict[str, str]:
    return {
        "case_id": cid, "family": fam, "split": split,
        "acquisition_uid": acquisition_uid, "boundary": boundary, "geometry_key": geometry_key,
        "error": _sanitize_error(err), "capture_uri": uri,
    }


def _eval_gates14(
    h: _SkuReplayHarness, g14: dict[str, Any], fsha: str,
    rows: list[dict[str, Any]], blocked: list[dict[str, Any]],
    captures: list[dict[str, Any]], expected_keys: Counter[tuple[str, str, str]],
) -> None:
    uri = f"capture://fleet-400k/{h.sku}/gates14/{fsha[:DIGEST_PREFIX_LEN]}"
    acq = str(g14.get("acquisition_uid") or "")
    captures.append(_capture_meta("gates14", uri, fsha, g14))

    for idx, r in enumerate(g14.get("gate1_paged_and_layout", [])):
        cid = str(r.get("case_id") or f"g1_{idx}")
        expected_keys[(acq, cid, "cold")] += 1
        if r.get("status") != "ok":
            blocked.append(_blk(cid, "gate1", "gate1", r.get("error"), uri, acquisition_uid=acq, boundary="cold"))
            continue
        kv_len, pg = int(r["kv_len"]), int(r.get("page_size") or 0)
        bt = (tuple(int(x) for x in r["block_table"]),) if r.get("block_table") else ()
        try:
            if r["op"] == "paged_mla_tnd":
                pred = h.pred_paged_mla(kv_len, page_size=pg or DEFAULT_PAGE_SIZE, block_table=bt)
                rd_b, fl = kv_len * MLA_DQK * BF16_BYTES, float(PAIR_FACTOR) * MLA_H * kv_len * (MLA_DQK + MLA_NOPE)
            else:
                lay = "PA_BLK_BSND" if bt else str(r.get("layout", "BNSD"))
                pred = h.pred_fia(1, 1, GQA_HQ, GQA_HKV, GQA_D, kv_len, layout=lay, page_size=pg, block_table=bt)
                rd_b, fl = PAIR_FACTOR * kv_len * GQA_HKV * GQA_D * BF16_BYTES, float(FMA_PAIR_FACTOR) * GQA_HQ * kv_len * GQA_D
            rows.append(score_verdict(cid, f"g1_{r['op']}", "gate1", float(r["us"]["p50"]), float(r["us"]["p99"]), pred, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b), uri, acquisition_uid=acq, boundary="cold"))
        except Exception as exc:
            blocked.append(_blk(cid, "gate1", "gate1", f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary="cold"))

    for idx, r in enumerate(g14.get("gate2_small_m_gemm", [])):
        cid = str(r.get("case_id") or f"g2_{idx}")
        expected_keys[(acq, f"{cid}_cold", "cold")] += 1
        expected_keys[(acq, f"{cid}_warm", "warm")] += 1
        if r.get("status") != "ok":
            blocked.extend(_blk(f"{cid}_{s}", f"g2_gemm_{s}", "gate2", r.get("error"), uri, acquisition_uid=acq, boundary=s) for s in ("cold", "warm"))
            continue
        m_dim, n_dim, k_dim = int(r["M"]), int(r["N"]), int(r["K"])
        try:
            cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": m_dim, "n": n_dim, "k": k_dim})
            pred_c = h.run_graph(list(cd.ops), cd.recipes)
            a_b, b_b, c_b = m_dim * k_dim * BF16_BYTES, k_dim * n_dim * BF16_BYTES, m_dim * n_dim * BF16_BYTES
            miss_b, admitted_hit = _admit_warm_gemm_l2_hits(int(h.hw["l2_bytes"]), a_b, b_b, c_b)
            w_meta = tuple({**dict(cd.ops[0].metadata), "admitted_l2_read_hit_bytes": admitted_hit}.items())
            op_w = replace(cd.ops[0], op_id=f"{cd.ops[0].op_id}_warm", hbm_bytes=float(miss_b), metadata=w_meta)
            pred_w = h.run_graph([op_w], cd.recipes)
            fl, rd_b, wr_b = float(PAIR_FACTOR) * m_dim * n_dim * k_dim, a_b + b_b, c_b
            rows.append(score_verdict(f"{cid}_cold", "g2_gemm_cold", "gate2", float(r["cold_us"]["p50"]), float(r["cold_us"]["p99"]), pred_c, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b, write_bytes=wr_b), uri, acquisition_uid=acq, boundary="cold"))
            rows.append(score_verdict(f"{cid}_warm", "g2_gemm_warm", "gate2", float(r["warm_us"]["p50"]), float(r["warm_us"]["p99"]), pred_w, _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=rd_b, write_bytes=wr_b, warm=True), uri, acquisition_uid=acq, boundary="warm"))
        except Exception as exc:
            blocked.extend(_blk(f"{cid}_{s}", f"g2_gemm_{s}", "gate2", f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary=s) for s in ("cold", "warm"))

    for idx, r in enumerate(g14.get("gate3_producer_consumer_chain", [])):
        cid = str(r.get("case_id") or f"g3_{idx}")
        for suf in ("prod", "cons", "chain"):
            expected_keys[(acq, f"{cid}_{suf}", suf)] += 1
        if r.get("status") != "ok":
            blocked.extend(_blk(f"{cid}_{s}", f, "gate3", r.get("error"), uri, acquisition_uid=acq, boundary=s) for s, f in (("prod", "g3_prod"), ("cons", "g3_cons_inc"), ("chain", "g3_chain")))
            continue
        tb = int(r["tensor_mib"]) * MIB
        n_el = tb // BF16_BYTES
        try:
            view = AdmissionMemoryView({0: int(h.hw["l2_bytes"])}, granule_bytes=L2_GRANULE_BYTES)
            for tid in ("x1", "x2"):
                view.admit_access(MemoryAccessPattern(tid, 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, tid, 0)), tb)
            view.admit_access(MemoryAccessPattern("y_mid", 0, 0, 0, "write", "mte3", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "y_mid", 0)), tb)
            _, y_m = view.admit_access(MemoryAccessPattern("y_mid", 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "y_mid", 0), reuse_distance_bytes=QKV_OPERANDS * tb), tb)
            _, x3_m = view.admit_access(MemoryAccessPattern("x3", 0, 0, 0, "read", "mte2_aiv", tb, AddressRunPlan((AddressRun(0, 1, tb, tb),)), reuse_key=ReuseKey(0, "x3", 0)), tb)
            p_op = OpSpec("g3_prod", "vector", "prefill", 0, 0, (), (n_el,), (n_el,), (n_el,), (n_el,), dtype="bf16", vector_ops=n_el, mte2_bytes=PAIR_FACTOR * tb, mte3_bytes=tb, hbm_bytes=QKV_OPERANDS * tb)
            c_op = OpSpec("g3_cons", "vector", "prefill", 0, 0, (), (n_el,), (n_el,), (n_el,), (n_el,), deps=("g3_prod",), dtype="bf16", vector_ops=n_el, mte2_bytes=PAIR_FACTOR * tb, mte3_bytes=tb, hbm_bytes=int(y_m + x3_m + tb))
            pred_p, pred_ch = h.run_graph([p_op], None, isolated=True), h.run_graph([p_op, c_op], None, isolated=False)
            lst_p, lst_c = _lean_lstar_us(h.hw, vec_ops=n_el, read_bytes=PAIR_FACTOR * tb, write_bytes=tb), _lean_lstar_us(h.hw, vec_ops=n_el, read_bytes=tb, write_bytes=tb)
            rows.append(score_verdict(f"{cid}_prod", "g3_prod", "gate3", float(r["prod_us"]["p50"]), float(r["prod_us"]["p99"]), pred_p, lst_p, uri, acquisition_uid=acq, boundary="prod"))
            rows.append(score_verdict(f"{cid}_cons", "g3_cons_inc", "gate3", float(r["cons_inc_us"]["p50"]), float(r["cons_inc_us"]["p99"]), pred_ch - pred_p, lst_c, uri, acquisition_uid=acq, boundary="cons"))
            rows.append(score_verdict(f"{cid}_chain", "g3_chain", "gate3", float(r["chain_us"]["p50"]), float(r["chain_us"]["p99"]), pred_ch, lst_p + lst_c, uri, acquisition_uid=acq, boundary="chain"))
        except Exception as exc:
            blocked.extend(_blk(f"{cid}_{s}", f, "gate3", f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary=s) for s, f in (("prod", "g3_prod"), ("cons", "g3_cons_inc"), ("chain", "g3_chain")))

    for idx, r in enumerate(g14.get("gate4_skewed_grouped_gemm", [])):
        cid = str(r.get("case_id") or f"g4_{idx}")
        expected_keys[(acq, f"{cid}_cold", "cold")] += 1
        expected_keys[(acq, f"{cid}_warm", "warm")] += 1
        if r.get("status") != "ok":
            blocked.extend(_blk(f"{cid}_{s}", f"g4_moe_gmm_{s}", "gate4", r.get("error"), uri, acquisition_uid=acq, boundary=s) for s in ("cold", "warm"))
            continue
        gl, k_dim, n_dim = [int(x) for x in r["group_list"]], int(r["K"]), int(r["N"])
        try:
            cd = build_aclnn_call({"api": "aclnnGroupedMatmul", "dtype": "bf16", "group_list": gl, "n": n_dim, "k": k_dim})
            pred_c = h.run_graph(list(cd.ops), cd.recipes)
            tot_t, nz = sum(gl), sum(1 for g in gl if g > 0)
            x_b, w_b, y_b = tot_t * k_dim * BF16_BYTES, nz * k_dim * n_dim * BF16_BYTES, tot_t * n_dim * BF16_BYTES
            miss_b, admitted_hit = _admit_warm_gemm_l2_hits(int(h.hw["l2_bytes"]), x_b, w_b, y_b)
            w_meta = tuple({**dict(cd.ops[0].metadata), "admitted_l2_read_hit_bytes": admitted_hit}.items())
            op_w = replace(cd.ops[0], op_id=f"{cd.ops[0].op_id}_warm", hbm_bytes=float(miss_b), metadata=w_meta)
            pred_w = h.run_graph([op_w], cd.recipes)
            fl = float(PAIR_FACTOR) * tot_t * n_dim * k_dim
            lst_c = _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=x_b + w_b, write_bytes=y_b)
            lst_w = _lean_lstar_us(h.hw, cube_flops=fl, read_bytes=x_b + w_b, write_bytes=y_b, warm=True)
            rows.append(score_verdict(f"{cid}_cold", "g4_moe_gmm_cold", "gate4", float(r["cold_us"]["p50"]), float(r["cold_us"]["p99"]), pred_c, lst_c, uri, acquisition_uid=acq, boundary="cold"))
            rows.append(score_verdict(f"{cid}_warm", "g4_moe_gmm_warm", "gate4", float(r["warm_us"]["p50"]), float(r["warm_us"]["p99"]), pred_w, lst_w, uri, acquisition_uid=acq, boundary="warm"))
        except Exception as exc:
            blocked.extend(_blk(f"{cid}_{s}", f"g4_moe_gmm_{s}", "gate4", f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary=s) for s in ("cold", "warm"))


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


def _eval_dcp_pcp_row(
    h: _SkuReplayHarness, r: dict[str, Any], *, cid: str, route: str, ws: int,
    g5_axis: str, cs_c: Any, eff_bw: float, uri: str, acq: str,
    rows: list[dict[str, Any]], blocked: list[dict[str, Any]],
    expected_keys: Counter[tuple[str, str, str]],
) -> None:
    ok_ref, ref_err = dcp_reference_verified(r)
    if not ok_ref:
        expected_keys[(acq, cid, "total")] += 1
        blocked.append(_blk(cid, f"dcp_pcp_{route}", "gate5", ref_err, uri, acquisition_uid=acq, boundary="total"))
        return
    for bnd, b_cid in (("fia", f"{cid}_fia"), ("comm", f"{cid}_comm"), ("merge", f"{cid}_merge"), ("total", cid)):
        expected_keys[(acq, b_cid, bnd)] += 1
    try:
        q_len, tot_kv = int(r["q_len"]), int(r["total_kv"])
        loc_kv = max(1, tot_kv // ws)
        t_fia = h.pred_fia(1, q_len, GQA_HQ, GQA_HKV, GQA_D, loc_kv, layout="BNSD", phase="decode" if q_len == 1 else "prefill")
        o_b, lse_b = q_len * GQA_HQ * GQA_D * BF16_BYTES, q_len * GQA_HQ * FP32_BYTES
        ag_o = comm_op(op_id=f"{cid}_ago", deps=(), stage=0, layer=0, requests=("r0",), family="allgather", phase="prefill", recipe="ring_allgather", link_bytes=(ws - 1) * o_b, shape=(1, max(1, o_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", "all_gather"), ("api_bytes_per_rank", o_b), ("steps", ws - 1)))
        ag_lse = comm_op(op_id=f"{cid}_aglse", deps=(ag_o.op_id,), stage=0, layer=0, requests=("r0",), family="allgather", phase="prefill", recipe="ring_allgather", link_bytes=(ws - 1) * lse_b, shape=(1, max(1, lse_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", "all_gather"), ("api_bytes_per_rank", lse_b), ("steps", ws - 1)))
        t_comm = h.run_graph([ag_o, ag_lse], None, isolated=False, case_spec=cs_c, include_sub=False)
        t_merge = h.pred_dcp_merge(ws, q_len, heads=GQA_HQ, dim=GQA_D, include_stack=True)

        fia_fl = float(FMA_PAIR_FACTOR) * GQA_HQ * q_len * loc_kv * GQA_D
        fia_rd = (PAIR_FACTOR * loc_kv * GQA_HKV + q_len * GQA_HQ) * GQA_D * BF16_BYTES
        comm_b = float(ws - 1) * (o_b + lse_b)
        mrg_rd = ws * (o_b + lse_b)
        lst_fia = _lean_lstar_us(h.hw, cube_flops=fia_fl, read_bytes=fia_rd, write_bytes=o_b + lse_b)
        lst_comm = _lean_lstar_us(h.hw, comm_bytes=comm_b, link_bw_bps=eff_bw)
        lst_mrg = _lean_lstar_us(h.hw, vec_ops=ws * q_len * GQA_HQ * GQA_D, read_bytes=mrg_rd, write_bytes=o_b)
        lst_tot = _lean_lstar_us(h.hw, cube_flops=fia_fl, read_bytes=PAIR_FACTOR * loc_kv * GQA_HKV * GQA_D * BF16_BYTES, comm_bytes=comm_b, link_bw_bps=eff_bw)

        v_fia = score_verdict(f"{cid}_fia", f"dcp_pcp_{route}_fia", "gate5", float(r["fia_us"]["p50"]), float(r["fia_us"]["p99"]), t_fia, lst_fia, uri, acquisition_uid=acq, boundary="fia")
        v_comm = score_verdict(f"{cid}_comm", f"dcp_pcp_{route}_comm", "gate5", float(r["comm_us"]["p50"]), float(r["comm_us"]["p99"]), t_comm, lst_comm, uri, acquisition_uid=acq, boundary="comm")
        v_mrg = score_verdict(f"{cid}_merge", f"dcp_pcp_{route}_merge", "gate5", float(r["merge_us"]["p50"]), float(r["merge_us"]["p99"]), t_merge, lst_mrg, uri, acquisition_uid=acq, boundary="merge")
        v_tot = score_verdict(cid, f"dcp_pcp_{route}", "gate5", float(r["total_us"]["p50"]), float(r["total_us"]["p99"]), t_fia + t_comm + t_merge, lst_tot, uri, acquisition_uid=acq, boundary="total")
        failed_inner = [n for n, v in (("fia", v_fia), ("comm", v_comm), ("merge", v_mrg)) if v["status"] != "pass"]
        if failed_inner:
            v_tot["status"] = "fail"
            v_tot["inner_component_failures"] = failed_inner
        if any(v["l1_cliff_status"] != "pass" for v in (v_fia, v_comm, v_mrg)):
            v_tot["l1_cliff_status"] = "fail"
        for item in (v_fia, v_comm, v_mrg, v_tot):
            item["capture_quality_flags"] = ["capture_unbarriered_eager_event"]
        rows.extend((v_fia, v_comm, v_mrg, v_tot))
    except Exception as exc:
        err_s = f"{type(exc).__name__}:{exc}"
        blocked.extend(
            _blk(b_cid, f"dcp_pcp_{route}", "gate5", err_s, uri, acquisition_uid=acq, boundary=bnd)
            for bnd, b_cid in (("fia", f"{cid}_fia"), ("comm", f"{cid}_comm"), ("merge", f"{cid}_merge"), ("total", cid))
        )


def _eval_gate5(
    h: _SkuReplayHarness, cap_dir: Path, rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]], captures: list[dict[str, Any]],
    expected_keys: Counter[tuple[str, str, str]],
) -> None:
    g5_paths = sorted(cap_dir.glob(f"{h.sku}_*r_*.json"))
    if not g5_paths:
        raise FileNotFoundError(f"missing_required_gate5_captures:{h.sku}")
    comm_cases: dict[tuple[str, int], Any] = {}
    for g5_path in g5_paths:
        g5, fsha = load_verified_capture(g5_path)
        tag = g5_path.stem[len(h.sku) + 1:]
        uri = f"capture://fleet-400k/{h.sku}/{tag}/{fsha[:DIGEST_PREFIX_LEN]}"
        acq = str(g5.get("acquisition_uid") or "")
        captures.append(_capture_meta(f"gate5_{tag}", uri, fsha, g5))
        ws, route = int(g5["group_size"]), str(g5["route"])
        g5_clk = extract_observed_clock_mhz(g5, fallback_mhz=h.observed_clock_mhz)
        if (route, ws) not in comm_cases:
            if route == "HCCS_SW" and ws == TWO_RANK_WORLD:
                raw_c = compose_case("qwen3.8-27b", hardware=h.sku, tp=TWO_RANK_WORLD, dp=TWO_RANK_WORLD, ep=1, num_devices=FOUR_RANK_WORLD, layers=1, quant="none")
            else:
                raw_c = compose_case("qwen3.8-27b", hardware=h.sku, tp=ws, ep=1, num_devices=ws, layers=1, quant="none")
            comm_cases[(route, ws)] = parse_case(apply_sku_observed_clock(raw_c, g5_clk))
        g5_axis = "dp" if (route == "HCCS_SW" and ws == TWO_RANK_WORLD) else "tp"
        cs_c = comm_cases[(route, ws)]
        eff_bw = _route_link_bw_bps(cs_c, route, ws, h.hw)

        for idx, r in enumerate(g5.get("collectives", [])):
            cid = str(r.get("case_id") or f"g5_{route}_r{ws}_{r.get('op', 'coll')}_{r.get('bytes', idx)}B")
            has_blk = bool(r.get("blocked_us") and r["blocked_us"].get("n", 0) > 0)
            bnd = "blocked" if has_blk else "eager"
            expected_keys[(acq, cid, bnd)] += 1
            if r.get("status") == "blocked" or not r.get("verified_all_ranks"):
                blocked.append(_blk(cid, f"g5_{route}", "gate5", r.get("error", "unverified"), uri, acquisition_uid=acq, boundary=bnd))
                continue
            op_name, pb = str(r["op"]), int(r["bytes"])
            meas_d, s_meta = (r["blocked_us"], ()) if has_blk else (r["eager_us"], (("collective_stack", "eager"),))
            try:
                api_b = pb // ws if op_name == "all_gather" else pb
                if op_name == "all_reduce":
                    lb, fam, rec, steps = float(PAIR_FACTOR) * (ws - 1) / ws * pb, "allreduce", "ring_allreduce", (("steps", PAIR_FACTOR * (ws - 1)),)
                elif op_name == "all_gather":
                    lb, fam, rec, steps = float(ws - 1) / ws * pb, "allgather", "ring_allgather", (("steps", ws - 1),)
                elif op_name == "reduce_scatter":
                    lb, fam, rec, steps = float(ws - 1) / ws * pb, "reduce_scatter", "ring_reduce_scatter", (("steps", ws - 1),)
                else:
                    lb, fam, rec, steps = float(ws - 1) / ws * pb, "alltoall", "expert_dispatch", ()
                op = comm_op(op_id=cid, deps=(), stage=0, layer=0, requests=("r0",), family=fam, phase="prefill", recipe=rec, link_bytes=lb, shape=(1, max(1, api_b // BF16_BYTES)), metadata=(("group_axis", g5_axis), ("collective", op_name), ("api_bytes_per_rank", api_b), *steps, *s_meta))
                pred = h.run_graph([op], None, isolated=False, case_spec=cs_c, include_sub=False)
                tag_s = "steady" if pb >= STEADY_COMM_MIN_BYTES else "small"
                verdict = score_verdict(cid, f"g5_{route}_{tag_s}", "gate5", float(meas_d["p50"]), float(meas_d["p99"]), pred, _lean_lstar_us(h.hw, comm_bytes=lb, link_bw_bps=eff_bw), uri, acquisition_uid=acq, boundary=bnd)
                if has_blk:
                    blk_p50 = float(meas_d["p50"])
                    blk_cv = float(meas_d.get("cv_pct") or 0.0)
                    eager_p50 = float((r.get("eager_us") or {}).get("p50") or 0.0)
                    if (
                        (eager_p50 > 0.0 and blk_p50 > eager_p50)
                        or (route in STARVED_SIO_ROUTES and pb <= STARVED_SMALL_MAX_BYTES and blk_p50 > STARVED_BLOCKED_FLOOR_US)
                        or blk_cv > STARVED_CV_PCT
                    ):
                        verdict["capture_quality_flags"] = ["capture_host_starved"]
                rows.append(verdict)
            except Exception as exc:
                blocked.append(_blk(cid, f"g5_{route}", "gate5", f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary=bnd))

        for idx, r in enumerate(g5.get("dcp_pcp", [])):
            cid = str(r.get("case_id") or f"dcp_pcp_{route}_r{ws}_{r.get('mode', 'dcp')}_q{r.get('q_len', idx)}_kv{r.get('total_kv', 0)}")
            _eval_dcp_pcp_row(h, r, cid=cid, route=route, ws=ws, g5_axis=g5_axis, cs_c=cs_c, eff_bw=eff_bw, uri=uri, acq=acq, rows=rows, blocked=blocked, expected_keys=expected_keys)


def _eval_seq_splits(
    h: _SkuReplayHarness, cap_dir: Path, rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
    captures: list[dict[str, Any]], expected_keys: Counter[tuple[str, str, str]],
) -> None:
    anchor_geoms: set[str] = set()
    for split in ("anchor", "heldout"):
        doc, fsha = load_verified_capture(cap_dir / f"{h.sku}_{split}.json")
        uri = f"capture://fleet-400k/{h.sku}/{split}/{fsha[:DIGEST_PREFIX_LEN]}"
        doc_acq = str(doc.get("acquisition_uid") or "")
        captures.append(_capture_meta(split, uri, fsha, doc))
        doc_cann = str(doc.get("cann_version") or (h.pt.cann_version if h.pt is not None else ""))
        for idx, r in enumerate(doc.get("records", [])):
            op_name = str(r.get("op", ""))
            raw_cid = str(r.get("case_id") or f"{op_name}_{split}_{idx}")
            acq = str(r.get("acquisition_uid") or doc_acq)
            if is_worker_diagnostic_record(r):
                expected_keys[(acq, raw_cid, "diagnostic")] += 1
                diagnostics.append({
                    "case_id": raw_cid, "family": op_name, "split": split,
                    "acquisition_uid": acq, "boundary": "diagnostic",
                    "reason": "worker_process_abort_diagnostic",
                    "error": _sanitize_error(r.get("error")), "capture_uri": uri,
                })
                continue
            eff_split = "ragged" if op_name == "ragged_suite" else split
            cid = f"{raw_cid}_{split}" if eff_split == "ragged" else raw_cid
            gk = seq_geometry_key(op_name, r)
            if eff_split == "anchor":
                anchor_geoms.add(gk)
            elif eff_split == "heldout" and gk in anchor_geoms:
                expected_keys[(acq, cid, "diagnostic")] += 1
                diagnostics.append({
                    "case_id": cid, "family": op_name, "split": split,
                    "acquisition_uid": acq, "boundary": "diagnostic", "geometry_key": gk,
                    "reason": "duplicate_anchor_geometry_in_heldout_capture",
                    "error": _sanitize_error(r.get("error", "duplicate_geometry")), "capture_uri": uri,
                })
                continue
            expected_keys[(acq, cid, "cold")] += 1
            if r.get("status") != "ok":
                blocked.append(_blk(cid, op_name, eff_split, r.get("error"), uri, acquisition_uid=acq, boundary="cold", geometry_key=gk))
                continue
            try:
                pred, lst = _eval_single_seq_record(h, op_name, r, doc_cann)
                if pred is None:
                    blocked.append(_blk(cid, op_name, eff_split, f"unwired_op:{op_name}", uri, acquisition_uid=acq, boundary="cold", geometry_key=gk))
                else:
                    rows.append(score_verdict(cid, op_name, eff_split, float(r["us"]["p50"]), float(r["us"]["p99"]), pred, lst, uri, acquisition_uid=acq, boundary="cold", geometry_key=gk))
            except Exception as exc:
                blocked.append(_blk(cid, op_name, eff_split, f"{type(exc).__name__}:{exc}", uri, acquisition_uid=acq, boundary="cold", geometry_key=gk))


def _eval_single_seq_record(
    h: _SkuReplayHarness, op_name: str, r: dict[str, Any], doc_cann: str,
) -> tuple[float | None, float]:
    if op_name == "fia_gqa_decode":
        seqlen = int(r["seqlen"])
        return (
            h.pred_fia(1, 1, GQA_HQ, GQA_HKV, GQA_D, seqlen, layout="BNSD", phase="decode"),
            _lean_lstar_us(h.hw, cube_flops=float(FMA_PAIR_FACTOR) * GQA_HQ * seqlen * GQA_D, read_bytes=(PAIR_FACTOR * seqlen * GQA_HKV + GQA_HQ) * GQA_D * BF16_BYTES, write_bytes=GQA_HQ * GQA_D * BF16_BYTES),
        )
    if op_name == "fia_gqa_prefill":
        seqlen, q_len = int(r["seqlen"]), int(r["q_len"])
        return (
            h.pred_fia(1, q_len, GQA_HQ, GQA_HKV, GQA_D, seqlen, layout="BNSD", phase="prefill", causal=False),
            _lean_lstar_us(h.hw, cube_flops=float(FMA_PAIR_FACTOR) * GQA_HQ * q_len * seqlen * GQA_D, read_bytes=(PAIR_FACTOR * seqlen * GQA_HKV + q_len * GQA_HQ) * GQA_D * BF16_BYTES, write_bytes=q_len * GQA_HQ * GQA_D * BF16_BYTES),
        )
    if op_name == "paged_mla_decode":
        seqlen = int(r["seqlen"])
        return (
            h.pred_paged_mla(seqlen, page_size=int(r.get("page_size", DEFAULT_PAGE_SIZE))),
            _lean_lstar_us(h.hw, cube_flops=float(PAIR_FACTOR) * MLA_H * seqlen * (MLA_DQK + MLA_NOPE), read_bytes=(seqlen + MLA_H) * MLA_DQK * BF16_BYTES, write_bytes=MLA_H * MLA_NOPE * BF16_BYTES),
        )
    if op_name in {"sfa_dsa_decode", "c8_sfa_decode"}:
        seqlen, tk = int(r["seqlen"]), int(r.get("top_k", DEFAULT_TOPK))
        eff = min(seqlen, tk)
        is_c8 = op_name == "c8_sfa_decode"
        row_b = C8_ROW_BYTES if is_c8 else MLA_DQK * BF16_BYTES
        return (
            h.pred_sfa(seqlen, topk=tk, quant_c8=is_c8, cann_ver=doc_cann),
            _lean_lstar_us(h.hw, cube_flops=float(PAIR_FACTOR) * MLA_H * eff * (MLA_DQK + MLA_NOPE), read_bytes=eff * row_b + MLA_H * MLA_DQK * BF16_BYTES, write_bytes=MLA_H * MLA_NOPE * BF16_BYTES),
        )
    if op_name == "lightning_indexer_topk":
        seqlen, tk = int(r["seqlen"]), int(r.get("top_k", DEFAULT_TOPK))
        return (
            h.pred_indexer(seqlen, topk=tk),
            _lean_lstar_us(h.hw, cube_flops=float(PAIR_FACTOR) * IDX_H * seqlen * IDX_D, read_bytes=(seqlen + IDX_H) * IDX_D * BF16_BYTES, write_bytes=min(seqlen, tk) * FP32_BYTES),
        )
    if op_name == "dcp_lse_attention_merge":
        cp, q_len = int(r["cp"]), int(r["q_len"])
        return (
            h.pred_dcp_merge(cp, q_len, heads=GQA_HQ, dim=GQA_D, include_stack=False),
            _lean_lstar_us(h.hw, read_bytes=cp * (GQA_HQ * q_len * GQA_D * BF16_BYTES + GQA_HQ * q_len * FP32_BYTES), write_bytes=GQA_HQ * q_len * GQA_D * BF16_BYTES),
        )
    if op_name == "gdn_chunk":
        seqlen, heads = int(r["seqlen"]), int(r.get("H", GDN_DEFAULT_H))
        st_b = heads * GDN_DIM * GDN_DIM * BF16_BYTES
        rd_b = QKV_OPERANDS * seqlen * heads * GDN_DIM * BF16_BYTES + seqlen * heads * (FP32_BYTES + BF16_BYTES) + st_b
        wr_b = seqlen * heads * GDN_DIM * BF16_BYTES + st_b
        return h.pred_linear_attn(kind="gdn_chunk", seqlen=seqlen, heads=heads, cann_ver=doc_cann), _lean_lstar_us(h.hw, read_bytes=rd_b, write_bytes=wr_b)
    if op_name == "gdn_recurrent":
        shape_d = r.get("actual_executed_shape") or {}
        batch = int(r.get("B") or shape_d.get("batch") or r.get("seqlen") or 1)
        heads = int(r.get("H") or shape_d.get("heads") or GDN_REC_DEFAULT_H)
        state_b = batch * heads * GDN_DIM * GDN_DIM * BF16_BYTES
        rd_b = state_b + batch * (QKV_OPERANDS * heads * GDN_DIM * BF16_BYTES + heads * (FP32_BYTES + BF16_BYTES))
        wr_b = state_b + batch * heads * GDN_DIM * BF16_BYTES
        return h.pred_linear_attn(kind="gdn_recurrent", seqlen=batch, heads=heads, cann_ver=doc_cann), _lean_lstar_us(h.hw, read_bytes=rd_b, write_bytes=wr_b)
    if op_name == "kda_chunk":
        seqlen, heads = int(r["seqlen"]), int(r.get("H", KDA_CHUNK_DEFAULT_H))
        # Captured cold inputs: q,k,v (bf16) + g (fp32, full D!) + beta (bf16) + h0 (fp32); output o (bf16).
        rd_b = (
            QKV_OPERANDS * seqlen * heads * GDN_DIM * BF16_BYTES
            + seqlen * heads * GDN_DIM * FP32_BYTES
            + seqlen * heads * BF16_BYTES
            + heads * GDN_DIM * GDN_DIM * FP32_BYTES
        )
        wr_b = seqlen * heads * GDN_DIM * BF16_BYTES
        return h.pred_linear_attn(kind="kda_chunk", seqlen=seqlen, heads=heads), _lean_lstar_us(h.hw, read_bytes=rd_b, write_bytes=wr_b)
    if op_name == "kda_recurrent":
        batch, heads = int(r["B"]), int(r.get("H", KDA_REC_DEFAULT_H))
        state_b = batch * heads * GDN_DIM * GDN_DIM * FP32_BYTES
        rd_b = state_b + batch * (QKV_OPERANDS * heads * GDN_DIM * BF16_BYTES + heads * GDN_DIM * FP32_BYTES + heads * BF16_BYTES)
        wr_b = state_b + batch * heads * GDN_DIM * BF16_BYTES
        return h.pred_linear_attn(kind="kda_recurrent", seqlen=batch, heads=heads), _lean_lstar_us(h.hw, read_bytes=rd_b, write_bytes=wr_b)
    if op_name == "ragged_suite":
        qlens = [int(x) for x in r["q_lens"]]
        kvlens = [int(x) for x in r["seq_lens"]]
        if str(r.get("sub_op", "")) == "fia_gqa_ragged":
            seqs = tuple(zip(qlens, kvlens, strict=True))
            return (
                h.pred_fia(len(seqs), max(qlens), GQA_HQ, GQA_HKV, GQA_D, max(kvlens), layout="TND", page_size=int(r.get("page_size", DEFAULT_PAGE_SIZE)), phase="decode" if max(qlens) == 1 else "prefill", sequences=seqs),
                _lean_lstar_us(h.hw, cube_flops=float(FMA_PAIR_FACTOR) * GQA_HQ * sum(q * kv for q, kv in seqs) * GQA_D, read_bytes=(PAIR_FACTOR * sum(kvlens) * GQA_HKV + sum(qlens) * GQA_HQ) * GQA_D * BF16_BYTES, write_bytes=sum(qlens) * GQA_HQ * GQA_D * BF16_BYTES),
            )
        return (
            h.pred_indexer(max(kvlens), batch=len(kvlens), topk=int(r.get("top_k", DEFAULT_TOPK)), seqs=kvlens),
            _lean_lstar_us(h.hw, cube_flops=float(PAIR_FACTOR) * IDX_H * sum(kvlens) * IDX_D, read_bytes=(sum(kvlens) + len(kvlens) * IDX_H) * IDX_D * BF16_BYTES, write_bytes=len(kvlens) * min(int(r.get("top_k", DEFAULT_TOPK)), min(kvlens)) * FP32_BYTES),
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
        p90_idx = min(len(apes) - 1, int(P90_QUANTILE * (len(apes) - 1)))
        med_ape = round(statistics.median(apes), ROUND_SUMMARY_DIGITS)
        p90_ape = round(apes[p90_idx], ROUND_SUMMARY_DIGITS)
        c_pass = sum(1 for x in items if x.get("l1_cliff_status") == "pass")
        c_fail = len(items) - c_pass
        inv_sim = sum(1 for x in items if x["inv_sim"])
        entry: dict[str, Any] = {
            "count": len(items), "pass": sum(1 for x in items if x["status"] == "pass"),
            "fail": sum(1 for x in items if x["status"] == "fail"),
            "l1_cliff_pass": c_pass, "l1_cliff_fail": c_fail,
            "med_ape_pct": med_ape, "p90_ape_pct": p90_ape,
            "mean_ape_pct": round(statistics.mean(apes), ROUND_SUMMARY_DIGITS),
            "max_ape_pct": round(apes[-1], ROUND_SUMMARY_DIGITS),
            "l1_two_tier_gate_pass": (
                med_ape <= L1_MEDIAN_APE_MAX_PCT
                and p90_ape <= L1_P90_APE_MAX_PCT
                and c_fail == 0 and inv_sim == 0
            ),
            "inv_sim": inv_sim, "inv_obs": sum(1 for x in items if x["inv_obs"]),
        }
        if l400:
            entry["cliff_400k_ape_pct"] = round(float(l400[-1]["ape_pct"]), ROUND_SUMMARY_DIGITS)
        out.setdefault(split, {})[fam] = entry
    return out


def _eval_sku_worker(args: tuple[str, str]) -> dict[str, Any]:
    sku, cap_dir_str = args
    cap_dir = Path(cap_dir_str)
    if not cap_dir.is_dir():
        raise FileNotFoundError(f"missing_capture_dir:{cap_dir}")
    t0 = time.monotonic()
    g14, g14_sha = load_verified_capture(cap_dir / f"{sku}_gates14.json")
    sku_clock_mhz = extract_observed_clock_mhz(g14)
    h = _SkuReplayHarness(sku, observed_clock_mhz=sku_clock_mhz, use_calibrated=True, use_submission=True)
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    expected_keys: Counter[tuple[str, str, str]] = Counter()
    _eval_gates14(h, g14, g14_sha, rows, blocked, captures, expected_keys)
    _eval_gate5(h, cap_dir, rows, blocked, captures, expected_keys)
    _eval_seq_splits(h, cap_dir, rows, blocked, diagnostics, captures, expected_keys)

    reconcile = reconcile_sku_accounting(expected_keys, rows, blocked, diagnostics)
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    n_fail = sum(1 for r in rows if r["status"] == "fail")
    n_cpass = sum(1 for r in rows if r.get("l1_cliff_status") == "pass")
    n_cfail = len(rows) - n_cpass
    flag_counts = Counter(flag for r in rows for flag in r.get("capture_quality_flags", ()))
    clean_rows = [r for r in rows if not r.get("capture_quality_flags")]
    total = n_pass + n_fail + len(blocked)
    if total != len(rows) + len(blocked) or (n_pass + n_fail) == 0:
        raise AssertionError(f"zero_drop_accounting_failed:{sku}")
    return {
        "sku": sku, "arch_preset": h.hw["arch"], "endpoint_alias": ENDPOINT_ALIAS[sku],
        "cann_version": h.pt.cann_version if h.pt is not None else "",
        "primitive_envelope_sha256": h.pt.envelope_sha256 if h.pt is not None else "",
        "observed_aicore_freq_mhz": sku_clock_mhz,
        "aic_count": h.hw["aic"], "aiv_count": h.hw["aiv"], "l2_bytes": int(h.hw["l2_bytes"]),
        "elapsed_s": round(time.monotonic() - t0, ROUND_SUMMARY_DIGITS),
        "counts": {
            "total": total, "pass": n_pass, "fail": n_fail,
            "l1_cliff_pass": n_cpass, "l1_cliff_fail": n_cfail,
            "blocked": len(blocked), "worker_diagnostics": len(diagnostics),
            "inv_sim": sum(1 for r in rows if r["inv_sim"]), "inv_obs": sum(1 for r in rows if r["inv_obs"]),
        },
        "capture_quality_summary": {
            "clean_scored": len(clean_rows),
            "clean_pass": sum(1 for r in clean_rows if r["status"] == "pass"),
            "clean_l1_cliff_pass": sum(1 for r in clean_rows if r.get("l1_cliff_status") == "pass"),
            "flagged_rows": len(rows) - len(clean_rows),
            "flags": dict(sorted(flag_counts.items())),
        },
        "reconciliation": reconcile, "family_summary": _summarize_families(rows),
        "captures": captures, "cases": rows, "blocked_cases": blocked,
        "worker_diagnostics": diagnostics,
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
    tot_cpass = sum(s["counts"]["l1_cliff_pass"] for s in sku_results)
    tot_cfail = sum(s["counts"]["l1_cliff_fail"] for s in sku_results)
    tot_blk = sum(s["counts"]["blocked"] for s in sku_results)
    tot_diag = sum(s["counts"]["worker_diagnostics"] for s in sku_results)
    tot_all = tot_pass + tot_fail + tot_blk
    tot_inv_sim = sum(s["counts"]["inv_sim"] for s in sku_results)
    tot_inv_obs = sum(s["counts"]["inv_obs"] for s in sku_results)
    overall_flags: Counter[str] = Counter()
    for s in sku_results:
        overall_flags.update(s["capture_quality_summary"]["flags"])

    payload: dict[str, Any] = {
        "schema_version": "mindie-sim-fleet-400k-verification/1",
        "campaign": "all-skus-512-400k-cliff-heldout-20261121",
        "acceptance_policy": {
            "relative_tolerance": REL_TOL, "absolute_floor_ns": ABS_FLOOR_NS, "inclusive": True,
            "l1_two_tier_policy": {
                "median_ape_max_pct": L1_MEDIAN_APE_MAX_PCT,
                "p90_ape_max_pct": L1_P90_APE_MAX_PCT,
                "cliff_relative_tolerance": L1_CLIFF_REL_TOL,
                "cliff_absolute_floor_ns": L1_CLIFF_ABS_FLOOR_NS,
            },
            "zero_drop_accounting": True, "zero_filtering": True, "zero_empirical_fitting": True,
            "strict_p50_lstar_non_inversion": True, "require_positive_lstar": True,
            "inner_digest_fail_closed": True, "observed_clock_replay": True,
            "dcp_pcp_component_gating": True, "geometry_disjoint_splits": True,
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
                    "l2_bytes": s["l2_bytes"], "observed_aicore_freq_mhz": s["observed_aicore_freq_mhz"],
                    "status": "calibrated", "primitive_envelope_sha256": s["primitive_envelope_sha256"],
                }
                for s in sku_results
            ],
            "unmeasured_skus": list(UNMEASURED_SKUS),
        },
        "overall_counts": {
            "total": tot_all, "pass": tot_pass, "fail": tot_fail,
            "l1_cliff_pass": tot_cpass, "l1_cliff_fail": tot_cfail,
            "blocked": tot_blk, "worker_diagnostics": tot_diag,
            "inv_sim": tot_inv_sim, "inv_obs": tot_inv_obs,
        },
        "capture_quality_summary": {
            "clean_scored": sum(s["capture_quality_summary"]["clean_scored"] for s in sku_results),
            "clean_pass": sum(s["capture_quality_summary"]["clean_pass"] for s in sku_results),
            "clean_l1_cliff_pass": sum(s["capture_quality_summary"]["clean_l1_cliff_pass"] for s in sku_results),
            "flagged_rows": sum(s["capture_quality_summary"]["flagged_rows"] for s in sku_results),
            "flags": dict(sorted(overall_flags.items())),
        },
        "residual_diagnostics": [
            "Audit-2 gdn_recurrent records failed on-device capture due to missing explicit scale=1/sqrt(K); evaluator wiring is verified on valid captures while Audit-2 failures stay accounted in blocked_cases.",
            "Audit-2 9362 2r_sio / 4r_mixed blocked small collectives and unbarriered eager dcp_pcp event breakdowns carry capture-intrinsic capture_quality_flags without dropping or filtering any scored verdict.",
            "DCP/PCP L>32768 remains blocked where rank-0 reference verification was not executed; kda_chunk/kda_recurrent remain unavailable on Arch35 CANN 9.2.0.",
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
    print(f"SEALED {out_path.name}: total={c['total']} pass={c['pass']} fail={c['fail']} blocked={c['blocked']} diag={c['worker_diagnostics']} inv_sim={c['inv_sim']} inv_obs={c['inv_obs']} sha256={bundle['bundle_sha256'][:DIGEST_PREFIX_LEN]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
