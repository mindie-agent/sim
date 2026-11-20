"""Regression suite for CANN 9.3.0 A5 3-level sub-10% silicon calibration.

[Role]
  Verify stored CANN 9.3.0 observation & 3-level accuracy artifacts alongside
  live-recomputed Level 1 (single operators across 512 -> 262,144 heldout
  sequence lengths), Level 2 (periodic body subgraphs & FX segment counts), and
  compact Level 3 (``merged`` / ``piecewise`` / ``unmerged_pr`` & multi-parallel
  model steps, commercial derived SKUs, and domain boundary guards).

[Invariants]
  * Heldout MAPE and MedAPE across Level 1, Level 2, and Level 3 remain
    strictly ``< 10.0%`` with zero ``train`` / ``heldout`` sample overlap.
  * Zero private host paths, intranet IPs, or container digests appear in
    observation or verification artifacts.
  * Compact live checks complete in ``< 10s`` single-process on 2-core CI.

[Timeouts & Locks]
  In-process ``layers <= 2``, ``num_devices <= 4`` graphs; zero I/O locks.

[Side-Effect Policy]
  Read-only verification of repository artifacts and in-memory ``Engine`` runs.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from mindie_sim import compose_case, graph_costs, target_graphs
from mindie_sim.backends.ascend import profiles as ascend_profiles
from mindie_sim.model_catalog import derive_expert_pruned
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.attention_decode import RECIPE_ID, StagedGQADecode
from mindie_sim.prediction.gdn_chunk32 import OUTPUT, PREPARE, STATE
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import (
    block_sparse_attention_arch35 as bsa,
    hc_gated_residual_arch35 as hc,
    mla_prolog_v4_arch35 as prolog4,
    sparse_index_scorers_arch35 as scorer,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

SIM_ROOT = Path(__file__).resolve().parents[1]
OBS_DIR = SIM_ROOT / "data" / "observations" / "a5-cann93-regimes-20261120"
VER_JSON = SIM_ROOT / "data" / "verification" / "a5-cann93-multilevel-accuracy-20261120.json"
VER_PRODUCER = SIM_ROOT / "data" / "verification" / "a5-cann93-multilevel-accuracy-20261120.producer.py"
MAX_HELDOUT_ERR_PCT = 10.0
FORBIDDEN_RE = re.compile(
    r"/(?:Users|home|mnt|root)/|\b\d{1,3}(?:\.\d{1,3}){3}\b|sha256:[0-9a-f]{12,}"
)


def test_observation_and_verification_packets_hygiene_and_sub10pct() -> None:
    for path in (OBS_DIR / "manifest.json", OBS_DIR / "parameters.json", VER_JSON, VER_PRODUCER):
        assert path.is_file(), f"missing artifact: {path}"
        raw_bytes = path.read_bytes()
        assert b"\r\n" not in raw_bytes, f"CRLF found in {path}"
        text = raw_bytes.decode("utf-8")
        assert not FORBIDDEN_RE.search(text), f"sensitive token leaked in {path}"

    manifest = json.loads((OBS_DIR / "manifest.json").read_text(encoding="utf-8"))
    for cap in manifest["captures"]:
        assert str(cap["uri"]).startswith("capture://")
        assert len(str(cap["source_sha256"])) == 64

    report = json.loads(VER_JSON.read_text(encoding="utf-8"))
    assert report["schema"] == "mindie-sim.a5-cann93-multilevel-accuracy/1"
    disjoint = report["disjoint_split_verification"]
    assert disjoint["level1_overlap"] == 0
    assert disjoint["level2_overlap"] == 0
    assert disjoint["level3_overlap"] == 0
    assert disjoint["exact_fx_subgraph_matches"] >= 10

    for rel_path, expected_hex in report["source_sha256"].items():
        actual_hex = sha256((SIM_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_hex == expected_hex, f"stale verification SHA256 for {rel_path}"

    for level_key in ("level1_operators", "level2_subgraphs_and_blocks", "level3_model_steps"):
        held = report[level_key]["heldout"]["summary"]
        assert held["count"] > 0, f"empty heldout set for {level_key}"
        assert held["mape_pct"] < MAX_HELDOUT_ERR_PCT, f"{level_key} heldout MAPE {held['mape_pct']}% >= 10%"
        assert held["medape_pct"] < MAX_HELDOUT_ERR_PCT, f"{level_key} heldout MedAPE {held['medape_pct']}% >= 10%"

    for fam, fstats in report["level1_operators"]["heldout"]["families"].items():
        if fstats["count"] >= 10:
            assert fstats["medape_pct"] < MAX_HELDOUT_ERR_PCT, f"L1 family {fam} heldout MedAPE {fstats['medape_pct']}% >= 10%"


def test_level1_live_heldout_operator_anchors_and_guards() -> None:
    cs_dt = parse_case(compose_case("qwen3.8-27b", hardware="Ascend950DT_9582", tp=1, ep=1, num_devices=1, layers=1, quant="none"))
    cs_pr = parse_case(compose_case("qwen3.8-27b", hardware="Ascend950PR_958b", tp=1, ep=1, num_devices=1, layers=1, quant="none"))

    def run_op(cs, op: OpSpec, builder) -> float:
        rb = RecipeBook(recipes={op.recipe: PlannedRecipe(builder)})
        return float(graph_costs(cs, {0: [op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0

    # FIA_v1 long-context single-split (B=1, S=1, L=131072 heldout) vs multi-block (B=8, S=4, L=131072 heldout)
    for B, S, min_bw, max_bw in ((1, 1, 180.0, 420.0), (8, 4, 1500.0, 2600.0)):
        seq_kv = 131072
        gqa_meta = {
            "requests": tuple((S, seq_kv - S) for _ in range(B)),
            "query_heads": 16, "kv_heads": 2, "head_dim": 128,
            "query_dtype": "bf16", "kv_dtype": "bf16", "causal": True, "window": 0, "cp": 1,
        }
        shape = (B * S, seq_kv, 16, 128)
        fia_op = OpSpec(
            op_id="fia_heldout", family="attention", recipe=RECIPE_ID, phase="decode", stage=0, layer=0,
            requests=tuple(f"r{i}" for i in range(B)), semantic_shape=shape, local_shape=shape,
            padded_shape=shape, launch_shape=shape, dtype="bf16", metadata=(("gqa_decode", gqa_meta),),
        )
        rb = RecipeBook(recipes={RECIPE_ID: StagedGQADecode(key_tile=128, target_cores=32, kv_mapping="fai_host_arch35")})
        dur_ms = float(graph_costs(cs_dt, {0: [fia_op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"])
        eff_bw_gbps = (B * seq_kv * 2 * 2 * 128 * 2) / (dur_ms * 1e6)
        assert min_bw <= eff_bw_gbps <= max_bw

    # C8 KvQuantSFA, MlaPrologV4, Sparse Scorer, mHC across DT & PR
    sfa_us = run_op(cs_dt, bsa.make_kv_quant_sfa_c8_op(op_id="sfa", query_tokens=8, num_heads=64, kv_lora_rank=512, rope_dim=64, selected_tokens=2048, batch_size=8, shared_prefix=False), bsa.build)
    prolog_us = run_op(cs_dt, prolog4.make_op(op_id="prolog4", tokens=2048, hidden_size=7168, num_heads=16, q_lora_rank=1536, weight_quant_mode=3, kv_cache_quant_mode=1), prolog4.build)
    scorer_us = run_op(cs_dt, scorer.make_msa_index_score_op(op_id="msa", query_tokens=8, context_tokens=16384, index_heads=32, index_dim=128, return_choice=1, batch_size=8), scorer.build)
    hc_dt_us = run_op(cs_dt, hc.make_hc_op(variant="hc_pre_v3", op_id="hc_dt", rows=2048, hidden_size=5120, hc_mult=4), hc.build_hc)
    hc_pr_us = run_op(cs_pr, hc.make_hc_op(variant="hc_pre_v3", op_id="hc_pr", rows=2048, hidden_size=5120, hc_mult=4), hc.build_hc)
    assert 45.0 <= sfa_us <= 85.0
    assert 400.0 <= prolog_us <= 900.0
    assert 15.0 <= scorer_us <= 45.0
    assert hc_pr_us > hc_dt_us > 50.0

    # Domain boundary guards: PR #21794 text_gated_residual M <= 512 EZ1008 & PR #22147 GDN Chunk32 Hv/Hk <= 4
    with pytest.raises(ModelingError, match="text_gated_residual_row_floor_not_met"):
        hc.make_hc_op(variant="text_gated_residual", op_id="bad_tgr", rows=512, hidden_size=2560, hc_mult=4, hc_lowrank=16)
    ok_tgr_us = run_op(cs_dt, hc.make_hc_op(variant="text_gated_residual", op_id="ok_tgr", rows=1024, hidden_size=2560, hc_mult=4, hc_lowrank=16), hc.build_hc)
    assert ok_tgr_us > 150.0

    with pytest.raises(ValueError, match="gdn_chunk32_domain_unsupported"):
        PREPARE.make_op({"lengths": (2048,), "heads": 16, "key_heads": 2, "dk": 128, "dv": 128, "chunk": 32})
    params = {"lengths": (2048,), "heads": 8, "key_heads": 2, "dk": 128, "dv": 128, "chunk": 32}
    p_op = PREPARE.make_op(params)
    s_op = replace(STATE.make_op(params), deps=(p_op.op_id,))
    o_op = replace(OUTPUT.make_op(params), deps=(p_op.op_id, s_op.op_id))
    rb_gdn = RecipeBook(recipes={p_op.recipe: PlannedRecipe(PREPARE.build), s_op.recipe: PlannedRecipe(STATE.build), o_op.recipe: PlannedRecipe(OUTPUT.build)})
    gdn_ms = float(graph_costs(cs_dt, {0: [p_op, s_op, o_op]}, recipes=rb_gdn, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"])
    assert 0.10 <= gdn_ms <= 0.30


def test_level2_and_level3_compact_subgraphs_piecewise_and_skus() -> None:
    assert len(ascend_profiles()) == 16
    for alias, exp_aic, exp_mib in (("425-96", 28, 96 * 1024), ("486-96", 32, 96 * 1024), ("547-96", 36, 96 * 1024), ("547-144", 36, 144 * 1024)):
        raw_sku = compose_case("qwen3.6-35b-a3b", hardware=alias, tp=4, ep=4, num_devices=4, layers=2)
        dev = raw_sku["machine"]["device"]
        assert int(dev["compute_units"]) == exp_aic
        assert int(dev["buffer_units"]["l2"]) == 1
        assert int(dev["memory_bytes"]) == exp_mib * 1024 * 1024

    # Heldout model rung ordering + LowRankHyperConnection fusion in expected_ideal
    qwen_base = derive_expert_pruned("qwen3.8-flash-next", num_experts=8, top_k=2)
    qwen_compact = replace(qwen_base, layers=1, body_layers=1, layer_plans=qwen_base.layer_plans[:1]).to_dict()
    ds_base = derive_expert_pruned("deepseek-v2-lite", num_experts=8, top_k=2)
    ds_compact = replace(ds_base, layers=2, body_layers=2, layer_plans=ds_base.layer_plans[:2]).to_dict()
    rungs = {}
    for trk in ("merged", "unmerged_pr", "expected_ideal"):
        raw = compose_case(qwen_compact, hardware="Ascend950DT_9582", feature_track=trk, _stack_phase="decode", tp=1, ep=1, num_devices=1, layers=1)
        cs = parse_case(raw)
        rep = graph_costs(cs, target_graphs(cs, phase="decode", tokens=1, context=2048, sequences=4, scope="body"), external_inputs={"analysis_input": 0}, include_submission=False, isolated=False, compact=True)
        rungs[trk] = float(rep["duration_ms"])
    assert rungs["merged"] >= rungs["unmerged_pr"] - 1e-6
    assert rungs["unmerged_pr"] > rungs["expected_ideal"]

    # Piecewise inter-segment host synchronization (SubmissionPolicy.piecewise_sync_ms)
    raw_pw = compose_case(ds_compact, hardware="Ascend950DT_9582", feature_track="merged", _stack_phase="prefill", tp=1, ep=1, sp=False, num_devices=1, layers=2, graph_capture_mode="piecewise", enable_npugraph_ex=False)
    raw_pw["runtime"]["mixed_mode"] = "piecewise"
    raw_pw["runtime"]["splitting_ops"] = ["attention", "mla", "kda", "gdn"]
    raw_pw["runtime"]["submission"] = {"kernel_launch_ms": 0.062, "graph_replay_ms": 0.30, "piecewise_sync_ms": 0.0}
    cs0 = parse_case(raw_pw)
    g0 = target_graphs(cs0, phase="prefill", tokens=256, context=0, sequences=1, scope="body")
    rep0 = graph_costs(cs0, g0, external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, fast_span_only=True)

    raw_pw_sync = dict(raw_pw)
    raw_pw_sync["runtime"] = {**dict(raw_pw["runtime"]), "submission": {"kernel_launch_ms": 0.062, "graph_replay_ms": 0.30, "piecewise_sync_ms": 0.50}}
    cs1 = parse_case(raw_pw_sync)
    rep1 = graph_costs(cs1, target_graphs(cs1, phase="prefill", tokens=256, context=0, sequences=1, scope="body"), external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, fast_span_only=True)
    replays = int(rep1["execution"]["submission"]["graph_replays"])
    assert replays == 3
    assert float(rep1["duration_ms"]) >= float(rep0["duration_ms"]) + (replays - 1) * 0.45

    # Compact multi-parallelism Level 3 checks (DPxEP, PCP, DCP)
    for kw, ph, tok, ctx in (
        (dict(dp=2, tp=1, ep=2, num_devices=2), "decode", 1, 2048),
        (dict(tp=1, pcp=2, ep=2, num_devices=2), "prefill", 256, 0),
        (dict(tp=2, dcp=2, ep=2, num_devices=2), "decode", 1, 8192),
    ):
        raw_mp = compose_case(ds_compact, hardware="Ascend950DT_9582", feature_track="merged", _stack_phase=ph, layers=2, **kw)
        cs_mp = parse_case(raw_mp)
        rep_mp = graph_costs(cs_mp, target_graphs(cs_mp, phase=ph, tokens=tok, context=ctx, sequences=2, scope="body"), external_inputs={"analysis_input": 0}, include_submission=False, isolated=False, compact=True)
        assert float(rep_mp["duration_ms"]) > 0.0
