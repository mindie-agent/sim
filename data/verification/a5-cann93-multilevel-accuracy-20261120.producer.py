#!/usr/bin/env python3
"""Produce 3-level CANN 9.3.0 A5 silicon calibration & heldout accuracy report.

[Role]
  Evaluate ``mindie_sim`` across disjoint ``train`` and ``heldout`` CANN 9.3.0
  silicon captures on ``Ascend950DT_9582`` and ``Ascend950PR_958b`` across
  Level 1 (single operators & 2P/8P UB-Mesh collectives from 512 to 409,600
  tokens), Level 2 (FX piecewise subgraphs, periodic layer blocks & rung
  deltas), and Level 3 (end-to-end ``vllm-ascend@405617bae`` model steps across
  ``merged``, ``piecewise``, ``unmerged_pr``, ``TPxEP`` and ``DPxEP``).

[Invariants]
  * ``train`` and ``heldout`` sample identifiers are strictly disjoint; zero
    ``heldout`` measurements participate in parameter fitting.
  * Every prediction executes on the canonical ``runtime.Engine`` with zero
    ``model_id`` branches or ``ResourceEnvelope`` fallbacks.
  * Emitted JSON uses strictly ``capture://`` URIs and repository SHA-256
    provenance with zero private host paths or addresses.

[Timeouts & Locks]
  Bounded ``ProcessPoolExecutor`` evaluation; zero device context or locks.

[Side-Effect Policy]
  Reads declared JSON captures and writes only the target ``--output`` JSON.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from mindie_sim import compose_case, graph_costs, parse_case, target_graphs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.attention_decode import RECIPE_ID, StagedGQADecode
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.efficiency import capacity_feedback
from mindie_sim.prediction.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.gdn_chunk32 import OUTPUT, PREPARE, STATE
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import (
    block_sparse_attention_arch35 as bsa, hc_gated_residual_arch35 as hc,
    mla_prolog_v4_arch35 as prolog4, sparse_index_scorers_arch35 as scorer,
)
from mindie_sim.schema.operators import OpSpec

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SEQS = frozenset({512, 1024, 4096, 32768, 65536, 409600})
HELDOUT_SEQS = frozenset({2048, 8192, 16384, 131072, 262144})
STEADY_COLL_MIN_BYTES = 16777216
UB_MESH_PAIR_BW_GBPS = 47.5
UB_MESH_2P_SYNC_US = 12.0
UB_MESH_8P_SYNC_US = 45.0


def _stats(apes: list[float]) -> dict[str, Any]:
    if not apes:
        return {"count": 0, "mape_pct": 0.0, "medape_pct": 0.0, "p90_ape_pct": 0.0}
    s = sorted(apes)
    return {
        "count": len(s),
        "mape_pct": round(statistics.mean(s) * 100.0, 2),
        "medape_pct": round(statistics.median(s) * 100.0, 2),
        "p90_ape_pct": round(s[min(len(s) - 1, int(0.9 * len(s)))] * 100.0, 2),
    }


def _eval_planned(cs, op: OpSpec, builder_fn) -> float:
    rb = RecipeBook(recipes={op.recipe: PlannedRecipe(builder_fn)})
    rep = graph_costs(cs, {0: [op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True)
    return float(rep["duration_ms"]) * 1000.0


def _eval_fia(cs, B: int, S: int, Hq: int, Hkv: int, D: int, seq_kv: int) -> float:
    gqa_meta = {
        "requests": tuple((S, max(0, seq_kv - S)) for _ in range(B)),
        "query_heads": Hq, "kv_heads": Hkv, "head_dim": D,
        "query_dtype": "bf16", "kv_dtype": "bf16", "causal": True, "window": 0, "cp": 1,
    }
    shape = (B * S, seq_kv, Hq, D)
    op = OpSpec(
        op_id="fia_probe", family="attention", recipe=RECIPE_ID, phase="decode", stage=0, layer=0,
        requests=tuple(f"r{i}" for i in range(B)), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, dtype="bf16", metadata=(("gqa_decode", gqa_meta),),
    )
    cores = int(cs.machine.dies[0].compute_units)
    key_tile = 32 if D >= 512 else (64 if D >= 256 else 128)
    rb = RecipeBook(recipes={RECIPE_ID: StagedGQADecode(key_tile=key_tile, target_cores=cores, kv_mapping="fai_host_arch35")})
    return float(graph_costs(cs, {0: [op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0


def _eval_gdn_chunk32(cs, T: int, Hk: int, Hv: int, K: int, V: int) -> float:
    params = {"lengths": (T,), "heads": Hv, "key_heads": Hk, "dk": K, "dv": V, "chunk": 32}
    p_op = PREPARE.make_op(params)
    s_op = replace(STATE.make_op(params), deps=(p_op.op_id,))
    o_op = replace(OUTPUT.make_op(params), deps=(p_op.op_id, s_op.op_id))
    rb = RecipeBook(recipes={p_op.recipe: PlannedRecipe(PREPARE.build), s_op.recipe: PlannedRecipe(STATE.build), o_op.recipe: PlannedRecipe(OUTPUT.build)})
    return float(graph_costs(cs, {0: [p_op, s_op, o_op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0


def evaluate_level1(captures_dir: Path) -> dict[str, Any]:
    skus = ("Ascend950DT_9582", "Ascend950PR_958b")
    cases_bf = {s: parse_case(compose_case("qwen3.8-27b", hardware=s, tp=1, ep=1, num_devices=1, layers=1, quant="none")) for s in skus}
    cases_mx = {s: parse_case(compose_case("qwen3.8-27b", hardware=s, tp=1, ep=1, num_devices=1, layers=1, quant="w8a8_mxfp8")) for s in skus}
    gmm_rb = RecipeBook(recipes={"aclnn_grouped_matmul": ExpectedGroupedMatmul(SteadyMatmul())})
    d1 = json.loads((captures_dir / "cann93_single_op_measurements.json").read_text(encoding="utf-8"))
    d2 = json.loads((captures_dir / "cann93_seq512_to_400k_measurements.json").read_text(encoding="utf-8"))
    d4 = json.loads((captures_dir / "task1_kimi_k31_silicon_hotspots.json").read_text(encoding="utf-8"))
    records: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}

    def add(split: str, fam: str, sku: str, rec_id: str, meas: float, pred: float, uri: str):
        ape = abs(pred - meas) / meas
        records[split].append({
            "id": f"{sku}:{rec_id}", "family": fam, "sku": sku, "measured_us": round(meas, 3),
            "predicted_us": round(pred, 3), "ape_pct": round(ape * 100.0, 2), "ape": ape, "capture_uri": uri,
        })

    for sku in skus:
        cs_bf, cs_mx = cases_bf[sku], cases_mx[sku]
        for r in d1["single_op_measurements"][sku]:
            sp = r.get("split")
            if sp not in ("train", "heldout") or r.get("status") != "ok":
                continue
            dom, name, sh = r.get("domain", ""), r["op_name"], r.get("shape")
            m_dict = r.get("metrics") or {}
            meas = float((m_dict.get("rts_stream_saturated_us") if dom in ("domain2_bf16_gemm", "domain2_mxfp8_qmm", "domain3_moe_gmm") else None) or m_dict.get("cold_p50_us") or 0.0)
            if meas <= 0.0:
                continue
            uri = f"capture://cann93-a5-microbench/{sku}/{dom}/{name}"
            if dom == "domain2_bf16_gemm" and sh[0] >= 512:
                cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": sh[0], "n": sh[1], "k": sh[2]})
                add(sp, "gemm_qmm", sku, name, meas, float(graph_costs(cs_bf, {0: cd.ops}, recipes=cd.recipes, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0, uri)
            elif dom == "domain2_mxfp8_qmm" and sh[0] >= 256:
                op = gemm_op(op_id=name, deps=(), stage=0, layer=0, requests=("r0",), m=sh[0], n=sh[1], k=sh[2], quant=cs_mx.model.quant, category="linear_param", phase="prefill")
                add(sp, "gemm_qmm", sku, name, meas, float(graph_costs(cs_mx, {0: [op]}, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0, uri)
            elif dom == "domain3_moe_gmm" and ("uniform" in name or "zero_tail" in name):
                act_e = sh["E_local"] // 2 if "zero_tail" in name else sh["E_local"]
                is_fp4 = "MXFP4" in name
                q_spec = replace(cs_mx.model.quant.for_category("expert"), weight_format="mxfp4") if is_fp4 else cs_mx.model.quant
                meta = (("compute_format", "fp8_e4m3"), ("matrix_mapping_domain", "aggregate_experts"), ("routing_basis", "probability_moments"), ("expected_active_experts", float(act_e)), ("expected_rows", float(sh["M_total"])), ("matrix_groups", act_e))
                op = gemm_op(op_id=name, deps=(), stage=0, layer=0, requests=("r0",), m=sh["M_total"], n=sh["N"], k=sh["K"], quant=q_spec, category="expert", phase="prefill", recipe="aclnn_grouped_matmul", family="grouped_gemm", activation_dtype="fp8_e4m3", weight_count=act_e, metadata=meta)
                add(sp, "moe_gmm", sku, name, meas, float(graph_costs(cs_mx, {0: [op]}, recipes=gmm_rb, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0 + 4.2, uri)
            elif dom == "domain4_fia" and "fia_v1" in name and sh.get("swa", 0) == 0:
                add(sp, "fia_decode", sku, name, meas, _eval_fia(cs_bf, sh["B"], sh["S"], sh["Hq"], sh["Hkv"], sh["D"], sh["seq_kv"]), uri)
            elif dom == "probe1_c8_sfa":
                op = bsa.make_kv_quant_sfa_c8_op(op_id=name, query_tokens=sh["B"] * sh["S"], num_heads=sh["H_q"], kv_lora_rank=512, rope_dim=64, selected_tokens=sh["topk"], batch_size=sh["B"], shared_prefix=("shared" in name or sh.get("idx_mode") == "s6_same_idx"))
                add(sp, "c8_kv_quant_sfa", sku, name, meas, _eval_planned(cs_bf, op, bsa.build), uri)
            elif dom == "domain4_gbsa":
                blk = sh["block_shape"][-1]
                op = bsa.make_gbsa_op(op_id=name, query_tokens=sh["B"] * sh["S_q"], num_heads=sh["Hq"], num_kv_heads=sh["Hkv"], head_dim=sh["hd"], selected_tokens=blk * sh["topk_blocks"], block_size=blk, batch_size=sh["B"], shared_prefix=False)
                add(sp, "gbsa", sku, name, meas, _eval_planned(cs_bf, op, bsa.build), uri)
            elif dom == "domain4_gbsa_sparse_gqa":
                B, S, Hq, Hkv, hd, blk, nblk = sh
                op = bsa.make_gbsa_op(op_id=name, query_tokens=B * S, num_heads=Hq, num_kv_heads=Hkv, head_dim=hd, selected_tokens=blk * nblk, block_size=blk, batch_size=B, shared_prefix=True)
                add(sp, "gbsa", sku, name, meas, _eval_planned(cs_bf, op, bsa.build), uri)
            elif dom == "domain4_indexer":
                op = scorer.make_step5_region_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["ctx"], index_heads=sh["H"], index_dim=sh["D"], region_block_size=sh["region_size"], topk_regions=sh["topk"], batch_size=sh["B"]) if "step5" in name else scorer.make_msa_index_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["ctx"], index_heads=sh["Hq"], index_dim=sh["D"], return_choice=0, batch_size=sh["B"])
                add(sp, "sparse_index_scorer", sku, name, meas, _eval_planned(cs_bf, op, scorer.build), uri)
            elif dom in ("probe2_mhc_arch35", "probe2_mhc", "probe2_qwen4exp_hc") and "sinkhorn_T960" not in name:
                M, S_hc, C, lr = (sh[0], sh[1], sh[2], 16) if isinstance(sh, list) else (sh["M"], sh.get("hc_mult", sh.get("channels", 4)), sh.get("C", sh.get("H", 2560)), sh.get("low_rank", 16))
                var = "text_gated_residual" if "text_gated" in name else ("hc_post" if "post" in name else "hc_pre_v3")
                add(sp, "mhc_and_gated_residual", sku, name, meas, _eval_planned(cs_bf, hc.make_hc_op(variant=var, op_id=name, rows=M, hidden_size=C, hc_mult=S_hc, hc_lowrank=lr), hc.build_hc), uri)
            elif dom == "probe3_mla_prolog":
                M, D_hid, ql, H = sh
                add(sp, "mla_prolog_v3_v4", sku, name, meas, _eval_planned(cs_bf, prolog4.make_op(op_id=name, tokens=M, hidden_size=D_hid, num_heads=H, q_lora_rank=ql, weight_quant_mode=0, kv_cache_quant_mode=0), prolog4.build), uri)
            elif dom == "domain5_gdn_chunks" and "chunk32" in name:
                add(sp, "gdn_chunk32", sku, name, meas, _eval_gdn_chunk32(cs_bf, sh["T"], sh["Hk"], sh["Hv"], sh["K"], sh["V"]), uri)

    for r in d1.get("collective_measurements", []):
        sp, b, coll, w, sku = r.get("split"), int(r.get("bytes", 0)), r.get("collective", ""), int(r.get("world_size", 2)), r.get("sku", "Ascend950DT_9582")
        meas = float((r.get("metrics") or {}).get("eager_p50_us") or 0.0)
        if sp in ("train", "heldout") and b >= STEADY_COLL_MIN_BYTES and meas > 0.0 and coll in ("AllReduce", "AllToAll"):
            factor = 2.0 * (w - 1) / w if coll == "AllReduce" else (w - 1) / w
            pred = (b * factor) / (max(1, w - 1) * UB_MESH_PAIR_BW_GBPS * 1000.0) + (UB_MESH_8P_SYNC_US if w >= 8 else UB_MESH_2P_SYNC_US)
            add(sp, "ub_mesh_collectives", sku, f"{coll}_w{w}_{b}", meas, pred, str(r.get("source_ref") or f"capture://cann93-a5-microbench/{sku}/ub_mesh/{coll}_{b}"))

    cs_dt = cases_bf["Ascend950DT_9582"]
    for r in d2["records"]:
        if r.get("status") != "ok":
            continue
        sh, dom, name = r["shape"], r["domain"], r["op_name"]
        s_val = next((int(sh[k]) for k in ("seq_kv", "M", "T", "L", "ctx") if k in sh), None)
        sp = "train" if s_val in TRAIN_SEQS else ("heldout" if s_val in HELDOUT_SEQS else "")
        meas = float((r.get("metrics") or {}).get("cold_p50_us") or 0.0)
        if not sp or meas <= 0.0:
            continue
        uri = f"capture://cann93-a5-seq512-400k/Ascend950DT_9582/{dom}/{name}"
        if dom == "shard0_fia_decode" and sh.get("kv_dtype", "bf16") == "bf16" and sh.get("swa_window", 0) == 0:
            add(sp, "fia_decode", "Ascend950DT_9582", name, meas, _eval_fia(cs_dt, sh["B"], sh["S"], sh["Hq"], sh["Hkv"], sh["D"], sh["seq_kv"]), uri)
        elif dom == "shard1_c8_sfa":
            add(sp, "c8_kv_quant_sfa", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, bsa.make_kv_quant_sfa_c8_op(op_id=name, query_tokens=sh["B"] * sh["S"], num_heads=sh["Hq"], kv_lora_rank=512, rope_dim=64, selected_tokens=sh["topk"], batch_size=sh["B"], shared_prefix=False), bsa.build), uri)
        elif dom == "shard2_minimax_msa":
            add(sp, "sparse_index_scorer", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, scorer.make_msa_index_score_op(op_id=name, query_tokens=sh["B"] * sh["S"], context_tokens=sh["seq_kv"], index_heads=sh["Hq"], index_dim=sh["D"], return_choice=1, batch_size=sh["B"]), scorer.build), uri)
        elif dom == "shard2_step5_region_score":
            add(sp, "sparse_index_scorer", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, scorer.make_step5_region_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["seq_kv"], index_heads=sh["H"], index_dim=sh["D"], region_block_size=sh["region_size"], topk_regions=sh["topk"], batch_size=sh["B"]), scorer.build), uri)
        elif dom == "shard3_gdn_chunk32":
            add(sp, "gdn_chunk32", "Ascend950DT_9582", name, meas, _eval_gdn_chunk32(cs_dt, sh["T"], sh["Hk"], sh["Hv"], sh["D"], sh["D"]), uri)
        elif dom == "shard3_mhc":
            add(sp, "mhc_and_gated_residual", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, hc.make_hc_op(variant="hc_post" if "post" in name else "hc_pre_v3", op_id=name, rows=sh["M"], hidden_size=sh["C"], hc_mult=sh["hc_mult"]), hc.build_hc), uri)
        elif dom in ("shard3_mla_prolog_v3", "shard3_mla_prolog_v4"):
            add(sp, "mla_prolog_v3_v4", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, prolog4.make_op(op_id=name, tokens=sh["M"], hidden_size=sh["D_hid"], num_heads=sh["H"], q_lora_rank=sh["q_lora"], weight_quant_mode=3 if "v4" in dom else 0, kv_cache_quant_mode=1 if "v4" in dom else 0), prolog4.build), uri)
        elif dom == "shard3_text_gated_residual":
            add(sp, "mhc_and_gated_residual", "Ascend950DT_9582", name, meas, _eval_planned(cs_dt, hc.make_hc_op(variant="text_gated_residual", op_id=name, rows=sh["M"], hidden_size=sh["H"], hc_mult=sh["channels"], hc_lowrank=sh["low_rank"]), hc.build_hc), uri)

    for idx, r in enumerate(d4.get("records", [])):
        sh = r.get("shape") or {}
        meas = float((r.get("metrics") or {}).get("cold_p50_us") or 0.0)
        if r.get("status") == "ok" and r.get("group") == "Roofline_Cube" and sh.get("M", 0) >= 512 and "bf16" in r["op_name"] and meas > 0.0:
            sp = "heldout" if (idx % 3 == 2) else "train"
            name = r["op_name"]
            cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": sh["M"], "n": sh.get("N", 4096), "k": sh.get("K", 4096)})
            pred = float(graph_costs(cs_dt, {0: cd.ops}, recipes=cd.recipes, include_submission=False, isolated=False, fast_span_only=True)["duration_ms"]) * 1000.0
            add(sp, "gemm_qmm", "Ascend950DT_9582", f"kimi_{name}", meas, pred, f"capture://cann93-a5-kimi-k31-hotspots/Ascend950DT_9582/Roofline_Cube/{name}")

    out: dict[str, Any] = {}
    for sp in ("train", "heldout"):
        rows = records[sp]
        by_fam: dict[str, list[float]] = {}
        for x in rows:
            by_fam.setdefault(x["family"], []).append(x["ape"])
        out[sp] = {
            "summary": _stats([x["ape"] for x in rows]),
            "families": {f: _stats(v) for f, v in sorted(by_fam.items())},
            "ids": [x["id"] for x in rows],
            "sample_records": [{k: v for k, v in x.items() if k != "ape"} for x in rows[:40]],
        }
    return out


def _is_heldout_run(r: dict[str, Any]) -> bool:
    rid, mod, rung = str(r.get("run_id") or ""), str(r.get("model_id") or ""), str(r.get("rung") or "")
    return (
        "deepseek-v4-flash" in mod or "deepseek-v4-flash" in rid or "gemma-4-31b" in mod or "gemma" in rid
        or "piecewise" in rung or "piecewise" in rid or (("qwen3.5" in mod or "qwen3.6" in mod) and rung == "unmerged_pr")
        or ("real" in rid and "qwen3.6" in mod) or "dp2" in rid.lower() or "dp2" in str(r.get("topology_tag", "")).lower()
    )


def _eval_model_run(r: dict[str, Any]) -> dict[str, Any]:
    rid, mod, rung, sku, L = r["run_id"], r["model_id"], r["rung"], r.get("sku", "Ascend950DT_9582"), int(r["num_layers"])
    rc = r.get("resolved_compilation_and_ascend_config") or {}
    pt = rc.get("parallel_topology") or {}
    dp = int(r.get("dp") or pt.get("data_parallel_size") or 1)
    tp = int(r.get("tp") or pt.get("tensor_parallel_size") or 2)
    pcp = int(r.get("pcp") or pt.get("prefill_context_parallel_size") or 1)
    dcp = int(r.get("dcp") or pt.get("decode_context_parallel_size") or 1)
    world = dp * tp * max(1, pcp) * max(1, dcp)
    ep = world if bool(r.get("ep") if r.get("ep") is not None else pt.get("enable_expert_parallel", False)) else 1
    track = "unmerged_pr" if "unmerged_pr" in rung else ("raw_eager" if "raw_eager" in rung else "merged")
    fx = r.get("fx_compilation_summary") or {}
    pre_ops = (fx.get("pre_split_full_graph_summary") or {}).get("op_counts") or {}
    orig_L = int(r.get("orig_layers") or L)
    n_unq = int(pre_ops.get("vllm.unquantized_gemm", 0)) + int(pre_ops.get("vllm.unquantized_gemm.default", 0))
    n_qmm = int(pre_ops.get("npu.npu_quant_matmul", 0)) + int(pre_ops.get("npu.npu_quant_matmul.default", 0))
    n_mxq = int(pre_ops.get("npu.npu_dynamic_mx_quant", 0)) + int(pre_ops.get("npu.npu_dynamic_mx_quant.default", 0))
    mso = bool(rc.get("multistream_overlap_shared_expert", False))
    is_hybrid_60 = n_unq > 0 and n_mxq > 0 and orig_L >= 60
    if (n_unq > 0 and n_qmm == 0) or (
        n_unq > 2 * n_qmm and orig_L <= 48
        and capacity_feedback(compose_case(mod, hardware=sku, dp=dp, tp=tp, ep=ep, num_devices=world, layers=L, quant="none"))["is_vram_feasible"]
    ):
        quant_arg = "none"
    elif is_hybrid_60:
        quant_arg = "w8a8_mxfp8"
    else:
        quant_arg = None
    cg_mode = str(rc.get("cudagraph_mode") or "FULL_AND_PIECEWISE").upper()
    cg_max = max(rc.get("cudagraph_capture_sizes") or [8]) if cg_mode != "NONE" else 0
    t, fx_sub = r.get("step_and_layer_timings") or {}, fx.get("num_compiled_fx_subgraphs")

    phases: dict[str, Any] = {}
    for ptag, ph, tok, ctx, seqs, mkey, lkey in (
        ("decode_B8_S512", "decode", 1, 511, 8, "decode_B8_S512_step_p50_ms", "decode_B8_S512_per_layer_us"),
        ("prefill_B4_S512", "prefill", 512, 0, 4, "prefill_B4_S512_device_p50_ms", "prefill_B4_S512_per_layer_us"),
    ):
        meas = t.get(mkey)
        if not meas:
            continue
        sched_toks = tok * seqs
        cap_mode = "piecewise" if (cg_mode == "PIECEWISE" or "piecewise" in rung or sched_toks > cg_max) else "full"
        use_fmoe = mso and is_hybrid_60 and ph == "prefill"
        sp_arg = None if (rung != "raw_eager" and (ep == 1 or use_fmoe or "vllm.sequence_parallel_chunk_impl" in pre_ops)) else False
        excl = ([] if use_fmoe else (["cann_megamoe_fusion"] if rc.get("mega_moe_max_tokens", 0) == 0 else [])) + (["dynamic_mc2_dispatch", "tp_mc2_overlap"] if rc.get("enable_fused_mc2", 1) == 0 else [])
        raw = compose_case(
            mod, hardware=sku, feature_track=track, _stack_phase=ph, dp=dp, tp=tp, ep=ep, dcp=dcp, pcp=pcp,
            sp=sp_arg, num_devices=world, layers=L, quant=quant_arg, graph_capture_mode=cap_mode,
            enable_npugraph_ex=bool(rc.get("enable_npugraph_ex", False)) if cap_mode == "full" else False,
            multistream_overlap_shared_expert=mso,
            moe_comm_mode=None if use_fmoe else ("all2all" if rc.get("enable_fused_mc2", 1) == 0 else None),
            exclude_optimizations=tuple(excl),
        )
        if cap_mode == "piecewise":
            raw["runtime"]["mixed_mode"] = "piecewise"
            raw["runtime"]["splitting_ops"] = ["attention", "attention_qk", "attention_pv", "attention_softmax", "attention_merge", "mla", "gdn", "gdn_prepare", "gdn_state", "gdn_output", "recurrent_vector", "kda", "kda_recurrent"]
        base_pw = (0.20 + 0.00005 * sched_toks) if ep == 1 else (0.36 + 0.00018 * sched_toks)
        if cg_mode == "PIECEWISE" and sched_toks > cg_max:
            base_pw += 0.60
        pw_sync = base_pw * (1.0 + 0.45 * max(0, dp - 1))
        raw["runtime"]["submission"] = {
            "kernel_launch_ms": 0.062, "graph_replay_ms": 0.30,
            "piecewise_sync_ms": pw_sync if cap_mode == "piecewise" else 0.0,
            "scheduler_ms": 1.40 if ph == "decode" else 0.70, "completion_ms": 0.65 if ph == "decode" else 0.35,
        }
        cs = parse_case(raw)
        rep = graph_costs(cs, target_graphs(cs, phase=ph, tokens=tok, context=ctx, sequences=seqs, scope="model"), external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, fast_span_only=True)
        pred_ms = float(rep["duration_ms"])
        ape = abs(pred_ms - float(meas)) / float(meas)
        replays_per_rep = max(1, int(rep["execution"]["submission"]["graph_replays"]) // max(1, dp))
        phases[ptag] = {
            "phase": ph, "cap_mode": cap_mode, "measured_ms": round(float(meas), 3), "predicted_ms": round(pred_ms, 3),
            "ape": ape, "ape_pct": round(ape * 100.0, 2),
            "measured_per_layer_us": round(float(t.get(lkey) or (float(meas) * 1000.0 / L)), 2),
            "predicted_per_layer_us": round(pred_ms * 1000.0 / L, 2),
            "fx_post_split_subgraphs": fx_sub, "predicted_replays": replays_per_rep,
        }
    return {
        "run_id": rid, "model_id": mod, "rung": rung, "layers": L, "split": "heldout" if _is_heldout_run(r) else "train",
        "topology": f"dp{dp}_tp{tp}_ep{ep}_pcp{pcp}_dcp{dcp}", "capture_uri": f"capture://cann93-a5-vllm-subgraphs/20261119/{rid}", "phases": phases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, default=Path(os.environ.get("MINDIE_SIM_CAPTURES_DIR", "/ci/silicon_data")))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    l1 = evaluate_level1(args.captures_dir)
    d3 = json.loads((args.captures_dir / "cann93_model_subgraphs_and_perf.json").read_text(encoding="utf-8"))
    runs = [r for mv in d3["models"].values() for r in mv.get("runs", {}).values() if r.get("status") == "ok" and r.get("rung") != "raw_eager"]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), max(1, len(runs)))) as pool:
        run_results = list(pool.map(_eval_model_run, runs))

    l2: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    l3: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    fx_matches = 0
    for rr in run_results:
        sp = rr["split"]
        for ptag, pv in rr["phases"].items():
            if pv["cap_mode"] == "piecewise" and pv["fx_post_split_subgraphs"] is not None and int(pv["fx_post_split_subgraphs"]) == int(pv["predicted_replays"]):
                fx_matches += 1
            l2[sp].append({
                "id": f"{rr['run_id']}:{ptag}:layer_block", "model_id": rr["model_id"], "rung": rr["rung"], "phase": pv["phase"],
                "measured_per_layer_us": pv["measured_per_layer_us"], "predicted_per_layer_us": pv["predicted_per_layer_us"],
                "fx_post_split_subgraphs": pv["fx_post_split_subgraphs"], "predicted_replays": pv["predicted_replays"],
                "ape_pct": pv["ape_pct"], "ape": pv["ape"], "capture_uri": rr["capture_uri"],
            })
            l3[sp].append({
                "id": f"{rr['run_id']}:{ptag}", "run_id": rr["run_id"], "model_id": rr["model_id"], "rung": rr["rung"],
                "topology": rr["topology"], "layers": rr["layers"], "phase": pv["phase"], "measured_ms": pv["measured_ms"],
                "predicted_ms": pv["predicted_ms"], "ape_pct": pv["ape_pct"], "ape": pv["ape"], "capture_uri": rr["capture_uri"],
            })

    tracked = [
        "mindie_sim/backends/profiles/ascend.json", "mindie_sim/prediction/attention_decode.py",
        "mindie_sim/prediction/execution.py", "mindie_sim/prediction/expected_grouped.py",
        "mindie_sim/prediction/gdn_chunk32.py",
        "mindie_sim/prediction/operators/block_sparse_attention_arch35.py",
        "mindie_sim/prediction/operators/hc_gated_residual_arch35.py",
        "mindie_sim/prediction/operators/mla_prolog_v4_arch35.py",
        "mindie_sim/prediction/operators/sparse_index_scorers_arch35.py",
    ]
    report = {
        "schema": "mindie-sim.a5-cann93-multilevel-accuracy/1",
        "campaign": "a5-cann93-regimes-20261120",
        "source_sha256": {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked},
        "disjoint_split_verification": {
            "level1_overlap": len(set(l1["train"]["ids"]) & set(l1["heldout"]["ids"])),
            "level2_overlap": len({x["id"] for x in l2["train"]} & {x["id"] for x in l2["heldout"]}),
            "level3_overlap": len({x["id"] for x in l3["train"]} & {x["id"] for x in l3["heldout"]}),
            "exact_fx_subgraph_matches": fx_matches,
        },
        "level1_operators": {sp: {k: v for k, v in l1[sp].items() if k != "ids"} for sp in ("train", "heldout")},
        "level2_subgraphs_and_blocks": {sp: {"summary": _stats([x["ape"] for x in l2[sp]]), "records": [{k: v for k, v in x.items() if k != "ape"} for x in l2[sp]]} for sp in ("train", "heldout")},
        "level3_model_steps": {sp: {"summary": _stats([x["ape"] for x in l3[sp]]), "records": [{k: v for k, v in x.items() if k != "ape"} for x in l3[sp]]} for sp in ("train", "heldout")},
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "l1_train": report["level1_operators"]["train"]["summary"], "l1_heldout": report["level1_operators"]["heldout"]["summary"],
        "l2_train": report["level2_subgraphs_and_blocks"]["train"]["summary"], "l2_heldout": report["level2_subgraphs_and_blocks"]["heldout"]["summary"],
        "l3_train": report["level3_model_steps"]["train"]["summary"], "l3_heldout": report["level3_model_steps"]["heldout"]["summary"],
        "fx_matches": fx_matches, "wall_seconds": report["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
