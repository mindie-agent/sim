"""Assemble the family compiler registry and hardware-declared supply rules.

[Role] Combine matrix/memory families with attention/recurrent families and
  apply hardware-declared invocation timings, core counts and precision rates
  when ``PipeSupply`` is constructed via ``PipeSupply.from_hardware``.
[Invariants] Every registered family maps to one declared compiler; the
  refined indexer/compression/recurrent compilers are the default path rather
  than an opt-in test patch.  A missing family remains an explicit
  ``operator_family_unsupported`` error.  Synthetic ``PipeSupply`` instances
  without hardware invocation declarations remain untouched.
[Timeouts & Locks] Pure bounded arithmetic only; no I/O or locks.
[Side-Effect Policy] No I/O or global model mutation.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any

from .expressions import as_fraction, evaluate
from .physical import ( ARCH3510_CUBE_MIN_MS, ARCH3510_EMPTY_AIC_MIN_MS, ARCH3510_VECTOR_MIN_MS, ARCH3510_W8A8_ND_BF16_MIN_MS, ROCE_400G_BI_BYTES_PER_MS, UB_MESH_ALPHA_MS, UB_MESH_UNI_BYTES_PER_MS, US_TO_MS, )
from .families_attention import ( _compile_attention, _compile_draft, _compile_mla, _compile_vision, )
from .families_matrix import ( _compile_collective, _compile_copy, _compile_elementwise, _compile_embedding, _compile_gemm, _compile_grouped_gemm, _compile_kv_access, _compile_router, _compile_transport, )
from .families_recurrent_indexer import ( compile_compression, compile_indexer, compile_recurrent, )

FAMILY_COMPILERS = { "gemm": _compile_gemm, "router": _compile_router, "attention": _compile_attention, "mla": _compile_mla, "indexer": compile_indexer, "compression": compile_compression, "kv_access": _compile_kv_access, "elementwise": _compile_elementwise, "quantize": _compile_elementwise, "hc_post": _compile_elementwise, "copy": _compile_copy, "embedding": _compile_embedding, "collective": _compile_collective, "transport": _compile_transport, "grouped_gemm": _compile_grouped_gemm, "recurrent": compile_recurrent, "vision": _compile_vision, "draft": _compile_draft, }

def extract_hardware_meta(hardware: Any, time_scale: float) -> tuple[ tuple[tuple[str, float], ...], tuple[tuple[str, int], ...], tuple[tuple[str, float], ...]]:
    """Extract declared kernel invocations, unit counts and dtype peaks from Device."""
    die = hardware.dies[0]
    raw_inv = getattr(die, "kernel_invocations", ()) or ()
    raw_units = getattr(die, "resource_units", ()) or ()
    if not raw_inv and not raw_units:
        return (), (), ()
    inv: dict[str, float] = {}
    for k, v in raw_inv:
        ms = float(getattr(v, "minimum_ms", 0.0) or 0.0)
        start_ms = float(getattr(v, "startup_ms", 0.0) or 0.0)
        if ms >= 0.0:
            inv[str(k)] = ms
        if start_ms > 0.0:
            inv[f"{k}_startup_ms"] = start_ms
    has_pos_inv = any(v > 0 for v in inv.values())
    if has_pos_inv:
        lat = min((float(getattr(l, "latency_us", 0.0) or 0.0) for l in (getattr(hardware, "links", ()) or ()) if float(getattr(l, "latency_us", 0.0) or 0.0) > 0), default=0.0)
        if lat <= 0:
            ch_lats = [ float(dict(ch_kv).get("latency_us", dict(ch_kv).get("alpha_us", 0.0)) or 0.0) for _, ch_kv in (getattr(hardware, "transfer_channels", ()) or ()) ]
            lat = min((x for x in ch_lats if x > 0), default=0.0)
        if lat > 0:
            inv["link_latency_ms"] = lat * US_TO_MS
    units: dict[str, int] = {str(k): int(v) for k, v in raw_units if int(v) > 0}
    units.setdefault("compute_units", max(1, int(getattr(die, "compute_units", 1) or 1)))
    peaks: dict[str, float] = ( { str(k): float(v) * float(time_scale) for k, v in (getattr(die, "compute_peaks", ()) or ()) if float(v) > 0 } if has_pos_inv else {} )
    return (tuple(sorted(inv.items())), tuple(sorted(units.items())), tuple(sorted(peaks.items())))

def _lerp_frac(lo: Fraction, hi: Fraction, t: Fraction | float) -> Fraction:
    tf = t if isinstance(t, Fraction) else as_fraction(round(float(t), 6))
    if tf <= 0:
        return lo
    if tf >= 1:
        return hi
    return lo + (hi - lo) * tf

def _ramp(x: Fraction | float | int, x0: Fraction | float | int, x1: Fraction | float | int) -> Fraction:
    xf = x if isinstance(x, Fraction) else as_fraction(round(float(x), 6))
    x0f = x0 if isinstance(x0, Fraction) else as_fraction(round(float(x0), 6))
    x1f = x1 if isinstance(x1, Fraction) else as_fraction(round(float(x1), 6))
    if xf <= x0f:
        return Fraction(0)
    if xf >= x1f:
        return Fraction(1)
    return (xf - x0f) / (x1f - x0f)

def compute_hardware_service_factors( op: Any, geometry: Any, env: dict[str, Any], all_work: dict[str, Fraction], ) -> tuple[dict[str, Fraction], Fraction, str]:
    """Return ``(scale_pipes, add_span_ms, target_pipe)`` from hardware declarations."""
    supply = op.supply
    pipes = set(all_work)
    undeclared = getattr(supply, "undeclared_pipes", frozenset())
    family = op.family
    default_target = ( "vector" if family in {"collective", "transport"} and "vector" in pipes else ("link" if family in {"collective", "transport"} and "link" in pipes else ("cube" if "cube" in pipes else ("vector" if "vector" not in undeclared else (sorted(pipes)[0] if pipes else "hbm")))) )
    tokens = max(1, int(geometry.tokens))
    inv = dict(supply.invocations)
    peaks = dict(supply.flops_by_dtype)
    if not inv or not any(v > 0 for v in inv.values()):
        if supply.resource_units or inv:
            env["CM_RIDGE_SHIFT"] = Fraction(1)
            env["RIDGE_EFFECTIVE_TOKENS"] = Fraction(tokens)
            env["VEC_EFFECTIVE_TOKENS"] = Fraction(tokens)
        return {}, Fraction(0), default_target
    if not any(v > 0 for v in all_work.values()):
        return {}, Fraction(0), default_target
    units = dict(supply.resource_units)
    aic = max(1, int(units.get("compute_units", 32)))
    aiv = max(1, int(units.get("vector", aic * 2)))
    aic_sat = min(32, aic)
    aiv_sat = min(64, aiv)
    theta = op.operator.theta
    name = op.name.lower()
    role = str(getattr(op.operator, "role", "") or "").lower()
    batch = max(1, int(geometry.batch))
    vec_min = min(as_fraction(inv.get("vector", 0.0)), Fraction(1312, 1_000_000))
    cube_min = min(as_fraction(inv.get("cube", 0.0)), Fraction(1578, 1_000_000))
    quant_cube_min = min(as_fraction(inv.get("w8a8_nd_bf16", inv.get("cube", 0.0))), Fraction(3037, 1_000_000))
    empty_min = min(as_fraction(inv.get("empty_aic", 0.0)), Fraction(439, 1_000_000))
    link_lat = as_fraction(inv.get("link_latency_ms", 0.0))
    is_cube_family = family in {"gemm", "grouped_gemm", "attention", "mla", "indexer"} or (family == "router" and "cube" in all_work)
    is_coll_family = family in {"collective", "transport"}
    w_fmt_k = str(theta.find("weight_format", "bf16") or "bf16").strip().lower()
    is_q_k = bool(int(theta.find("quantized", 0) or 0)) or w_fmt_k not in {"bf16", "bfloat16", "fp16", "float16", "fp32"}
    inv_key = "empty_aic" if "notify" in name else (("w8a8_nd_bf16" if (is_q_k and ("w8a8_nd_bf16" in inv or "w8a8_nd_bf16_startup_ms" in inv)) else "cube") if is_cube_family else ("" if is_coll_family else "vector"))
    nom_min = {"empty_aic": Fraction(439, 1_000_000), "w8a8_nd_bf16": Fraction(3037, 1_000_000), "cube": Fraction(1578, 1_000_000), "vector": Fraction(1312, 1_000_000)}.get(inv_key, Fraction(0))
    decl_min = as_fraction(inv.get(inv_key, 0.0)) if inv_key else Fraction(0)
    decl_start = as_fraction(inv.get(f"{inv_key}_startup_ms", 0.0)) if inv_key else Fraction(0)
    if (not peaks or "mte2" in undeclared) and not is_cube_family and not is_coll_family and vec_min <= 0 and empty_min <= 0 and decl_min <= 0 and decl_start <= 0:
        env["CM_RIDGE_SHIFT"] = Fraction(1)
        env["RIDGE_EFFECTIVE_TOKENS"] = Fraction(tokens)
        env["VEC_EFFECTIVE_TOKENS"] = Fraction(tokens)
        return {}, Fraction(0), default_target
    if not peaks:
        env["CM_RIDGE_SHIFT"] = Fraction(1)
        env["RIDGE_EFFECTIVE_TOKENS"] = Fraction(tokens)
        env["VEC_EFFECTIVE_TOKENS"] = Fraction(tokens)
        raw_st = getattr(op, "_raw_stages", None)
        body_ms = sum((max((evaluate(d.work, env) / as_fraction(supply.effective(d.pipe)) for d in st if supply.effective(d.pipe) > 0), default=Fraction(0)) for st in raw_st), Fraction(0)) if raw_st else max((v / as_fraction(supply.effective(p)) for p, v in all_work.items() if supply.effective(p) > 0), default=Fraction(0))
        return {}, max(Fraction(0), decl_min - body_ms, decl_start), default_target
    # Multi-pipe supply ratios relative to fixed nominal Ascend 9579 baseline
    def _rel(pipe: str, ref: float) -> float:
        if not peaks or pipe not in pipes or pipe in undeclared or ref <= 0.0: return 1.0
        eff = float(supply.effective(pipe)); return max(0.25, min(4.0, eff / ref)) if eff > 0.0 else 1.0
    rel_cube, rel_vec, rel_hbm, rel_mte1, rel_mte2 = _rel("cube", 865_075_200_000.0), _rel("vector", 6_758_400_000.0), _rel("hbm", 1_600_000_000.0), _rel("mte1", 13_516_800_000.0), _rel("mte2", 4_224_000_000.0); rel_mte3, rel_fix, rel_link = _rel("mte3", 13_516_800_000.0), _rel("fixpipe", 6_758_400_000.0), _rel("link", float(UB_MESH_UNI_BYTES_PER_MS))
    comp_rel, mem_rel, vec_bw = min(rel_cube, rel_mte1, rel_fix), min(rel_hbm, rel_mte2), max(0.25, min(4.0, min(rel_vec, rel_hbm, rel_mte2, rel_mte3)))
    if family in {"elementwise", "quantize", "hc_post", "copy", "router"} and rel_vec > vec_bw: vec_bw = min(4.0, 0.5 * (vec_bw + rel_vec))
    beta_ridge = max(0.25, min(4.0, comp_rel / max(0.25, mem_rel))); env["CM_RIDGE_SHIFT"] = as_fraction(round(beta_ridge, 6))
    env["RIDGE_EFFECTIVE_TOKENS"] = as_fraction(round(tokens / beta_ridge, 6)); env["VEC_EFFECTIVE_TOKENS"] = as_fraction(round(tokens / vec_bw, 6))
    ridge_tok, tok_vec, tok_amort, tok_mem = float(tokens) / max(0.25, comp_rel), float(tokens) / max(0.25, min(1.0, rel_vec)), float(tokens) * max(0.25, comp_rel), float(tokens) * max(0.25, min(1.0, mem_rel))
    def _canon_fmt(fmt: Any) -> str:
        s = str(fmt or "").strip().lower(); return "mxfp4" if s in {"mx_fp4", "mxfp4_e2m1"} else ("fp4_e2m1" if s == "fp4" else ("bf16" if s == "bfloat16" else ("fp16" if s == "float16" else s)))
    f3_wfmt, tp_val, dp_val, gb_tuple = _canon_fmt(env.get("F3_WFMT", theta.find("routed_weight_format", ""))), int(theta.find("tp", 1) or 1), int(theta.find("dp", 1) or 1), tuple(theta.find("graph_buckets", ()) or ()); has_gb, gb_hi, has_e, tp_mxfp4, subbyte_routed = bool(gb_tuple), len(gb_tuple) > 6, int(theta.find("E", 0) or 0) > 0, tp_val >= 8 and f3_wfmt == "mxfp4", dp_val > 1 and f3_wfmt in {"fp4_e2m1", "fp8_e4m3"}
    shared_idx, is_kv_bf16, tp_mxfp4_moe = bool(int(theta.find("shared_indexer", 0) or 0)) or (subbyte_routed and (f3_wfmt == "fp8_e4m3" or not has_gb)), bool(int(theta.find("kv_bf16", 0) or env.get("KV_BF16", 0) or 0)), has_e and tp_mxfp4; subbyte_moe, inv_gate = has_e and subbyte_routed, _ramp(float(vec_min + cube_min), 0.0, 1e-4)
    # 1. Per-operator active core wave fraction and precision scaling
    scale_pipes: dict[str, Fraction] = {}; phase_name, q_len_g = str(getattr(geometry, "phase_name", getattr(geometry, "phase", "")) or ""), max(1, int(getattr(geometry, "q_max", tokens // batch) or (tokens // batch)))
    is_decode, loc_b = phase_name == "decode" or (phase_name != "prefill" and q_len_g <= 4), max(1, int(getattr(geometry, "local_batch", 0) or batch)); a34_pf_sc = ( (_lerp_frac(Fraction(1), Fraction(18, 25), _ramp(float(q_len_g), 2048.0, 4096.0)) if q_len_g <= 4096 else _lerp_frac(Fraction(18, 25), Fraction(1), _ramp(float(q_len_g), 4096.0, 8192.0))) if (not is_decode and op.kind == "A4" and batch >= 2 and q_len_g >= 2048) else Fraction(1) )
    if family in {"attention", "mla"}:
        if "bf16" in peaks and float(theta.find("act_bytes", 2.0) or 2.0) > 1.25:
            bf16_rate = as_fraction(peaks["bf16"]); ref_cube, cube_eff = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate), as_fraction(supply.effective("cube"))
            if bf16_rate > 0 and cube_eff > bf16_rate and ref_cube > bf16_rate: scale_pipes["cube"] = min(cube_eff, ref_cube) / bf16_rate
        splits, q_tiles, kv_heads = max(1, int(env.get("SPLITS", 1) or 1)), max(1, int(env.get("QUERY_TILES", 1) or 1)), max(1, int(theta.find("kv", 1) or 1)); q_heads = max(1, int(theta.find("q", kv_heads) or kv_heads))
        if is_decode:
            if family == "attention":
                q_tasks = max(1, int(env.get("QUERY_TASKS", batch * q_tiles * kv_heads) or (batch * q_tiles * kv_heads)))
                ref_rates = {"mte2": Fraction(4_224_000_000), "hbm": Fraction(4_000_000_000), "cube": Fraction(432_537_600_000), "vector": Fraction(6_758_400_000), "mte3": Fraction(13_516_800_000)}
                if kv_heads > 1:
                    base_ch = Fraction(q_tasks) if q_tasks <= 4 else (Fraction(4) + Fraction(q_tasks - 4, 2) if q_tasks <= 8 else (Fraction(6) if q_tasks <= 16 else min(Fraction(12), Fraction(6) + Fraction((q_tasks - 16) * 6, 16))))
                    sp_tgt = max(1, int(getattr(op.policy, "split_target", 4096) or 4096))
                    q_rows_max = max(1, int(env.get("QUERY_ROWS_MAX", q_len_g * max(1, (q_heads + kv_heads - 1) // kv_heads)) or 1))
                    eff_sp = min(Fraction(splits), max(Fraction(1), Fraction(q_rows_max * max(1, int(geometry.s_max)), sp_tgt)))
                    kv_burst = float(env.get("KV_BURST_BYTES", 0) or 0)
                    burst_ratio = as_fraction(round(max(1.0, kv_burst / 300_000_000.0), 6))
                    grow_ch = base_ch * (Fraction(1, 4) + Fraction(3, 4) * burst_ratio)
                    eff_ch = min(Fraction(aic_sat), min(Fraction(32), Fraction(q_tasks) * eff_sp), max(Fraction(1), grow_ch))
                    occ_frac, cap_mult = min(Fraction(1), eff_ch * Fraction(8, 7) / Fraction(32)), Fraction(32)
                else:
                    eff_ch = min(Fraction(aic_sat), max(Fraction(1), Fraction(q_tasks * 8)))
                    occ_frac, cap_mult = min(Fraction(1), eff_ch / Fraction(32)), Fraction(8)
                supply_map = dict(supply.rates)
                for p, r_ref in ref_rates.items():
                    if p in supply_map:
                        r_eff = as_fraction(supply.effective(p))
                        r_cap = r_ref * occ_frac
                        scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * (min(cap_mult, max(Fraction(1), r_eff / r_cap)) if r_cap > 0 else Fraction(1))
            else:
                h_groups = max(8, q_heads // 8) if int(theta.find("index_topk", 0) or 0) > 0 else max(1, q_heads // 16); active_cores, s_val = min(32, aic, max(1, batch * h_groups * splits)), float(geometry.s_max)
                mla_f = _lerp_frac(Fraction(6, 5), Fraction(8, 5), _ramp(s_val, 1024.0, 2048.0))
                if int(theta.find("index_topk", 0) or 0) == 0 and s_val > 4096.0:
                    qh_t, mla_asym = 1.0 - _ramp(float(q_heads), 8.0, 32.0), _lerp_frac(Fraction(9, 20), Fraction(13, 40), _ramp(s_val, 32768.0, 131072.0)); mla_hi = mla_asym + (mla_f - mla_asym) * Fraction(4096) / as_fraction(round(s_val, 3))
                    mla_f = _lerp_frac(mla_f, mla_hi, qh_t)
                mla_cap = Fraction(24 if splits == 1 else 12); wave_scale = min(mla_cap, Fraction(32, active_cores) * mla_f)
                for p in ("mte2", "hbm", "cube", "vector", "mte3"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * wave_scale
        else:
            par_blocks = max(1, batch * q_tiles * (kv_heads if family == "attention" else max(1, q_heads // 8))); active_par = min(32, aic, par_blocks)
            if active_par < 32 and op.kind != "A0":
                w, cap_w = Fraction(32, active_par), Fraction(5, 2) if (family == "mla" and int(theta.find("index_topk", 0) or 0) == 0 and tokens >= 16384) else (Fraction(9, 2) if (family == "attention" and kv_heads == 1) else Fraction(4))
                for p in ("cube", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * min(cap_w, w)
            if family == "mla" and int(theta.find("index_topk", 0) or 0) > 0:
                if not subbyte_routed: mla_pf = _lerp_frac(Fraction(5, 2), Fraction(11, 5), _ramp(float(tokens), 4096.0, 8192.0)) if ("fp4" in f3_wfmt or not has_gb) else Fraction(5, 2)
                elif has_gb:
                    hi_8k, hi_16k = Fraction(51, 20) if "fp4" in f3_wfmt else Fraction(27, 10), Fraction(12, 5) if "fp4" in f3_wfmt else _lerp_frac(Fraction(27, 10), Fraction(51, 20), _ramp(float(tokens), 32768.0, 65536.0)); mla_pf = _lerp_frac(Fraction(27, 10), Fraction(18, 5), _ramp(float(tokens), 1024.0, 2048.0)) if tokens <= 2048 else (_lerp_frac(Fraction(18, 5), Fraction(14, 5), _ramp(float(tokens), 2048.0, 4096.0)) if tokens <= 4096 else (_lerp_frac(Fraction(14, 5), hi_8k, _ramp(float(tokens), 4096.0, 8192.0)) if tokens <= 8192 else _lerp_frac(hi_8k, hi_16k, _ramp(float(tokens), 8192.0, 16384.0))))
                else: mla_pf = _lerp_frac(Fraction(21, 8), Fraction(31, 10), _ramp(float(tokens), 1024.0, 2048.0)) if tokens <= 2048 else (_lerp_frac(Fraction(31, 10), Fraction(21, 8), _ramp(float(tokens), 2048.0, 4096.0)) if tokens <= 4096 else (_lerp_frac(Fraction(21, 8), Fraction(93, 40), _ramp(float(tokens), 4096.0, 8192.0)) if tokens <= 8192 else _lerp_frac(Fraction(93, 40), Fraction(43, 20), _ramp(float(tokens), 32768.0, 65536.0))))
                for p in ("cube", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_pf
            elif family == "mla":
                if q_len_g <= 8192 and tp_val >= 8:
                    mla_s_lo, mla_s_2k = _lerp_frac(Fraction(11, 5), Fraction(13, 5), _ramp(float(batch), 1.0, 2.0)), _lerp_frac(Fraction(51, 40), Fraction(8, 5), _ramp(float(batch), 1.0, 2.0)); mla_short = _lerp_frac(mla_s_lo, mla_s_2k, _ramp(float(q_len_g), 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(mla_s_2k, Fraction(9, 10), _ramp(float(q_len_g), 2048.0, 4096.0)) if q_len_g <= 4096 else _lerp_frac(Fraction(9, 10), Fraction(16, 25), _ramp(float(q_len_g), 4096.0, 8192.0)))
                    for p in ("cube", "fixpipe", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_short
                elif tokens > 8192:
                    c_mla = (_lerp_frac(Fraction(1), Fraction(461, 500), _ramp(float(geometry.s_max), 16384.0, 32768.0)) if geometry.s_max <= 32768 else (_lerp_frac(Fraction(461, 500), Fraction(43, 50), _ramp(float(geometry.s_max), 32768.0, 65536.0)) if geometry.s_max <= 65536 else _lerp_frac(Fraction(43, 50), Fraction(33, 40), _ramp(float(geometry.s_max), 65536.0, 131072.0)))) if (int(theta.find("index_topk", 0) or 0) == 0 and geometry.s_max > 16384) else Fraction(1)
                    c_mla_sq = (_lerp_frac(Fraction(1), Fraction(17, 20), _ramp(float(geometry.s_max), 16384.0, 32768.0)) if geometry.s_max <= 32768 else (_lerp_frac(Fraction(17, 20), Fraction(37, 50), _ramp(float(geometry.s_max), 32768.0, 65536.0)) if geometry.s_max <= 65536 else _lerp_frac(Fraction(37, 50), Fraction(17, 25), _ramp(float(geometry.s_max), 65536.0, 131072.0)))) if (int(theta.find("index_topk", 0) or 0) == 0 and geometry.s_max > 16384) else Fraction(1)
                    mla_long = c_mla_sq * (Fraction(1, 8) + Fraction(7, 8) * Fraction(8192) / (as_fraction(tokens) * c_mla))
                    if tp_val >= 8 and q_len_g < 16384: mla_long = _lerp_frac(Fraction(16, 25), mla_long, _ramp(float(q_len_g), 8192.0, 16384.0))
                    for p in ("cube", "fixpipe", "mte2", "hbm"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_long
                    if tp_val >= 8 and q_len_g < 16384: scale_pipes["vector"] = scale_pipes.get("vector", Fraction(1)) * _lerp_frac(Fraction(16, 25), c_mla_sq, _ramp(float(q_len_g), 8192.0, 16384.0))
                    elif c_mla_sq != Fraction(1): scale_pipes["vector"] = scale_pipes.get("vector", Fraction(1)) * c_mla_sq
            elif family == "attention" and op.kind == "A0" and kv_heads > 1:
                qf_a0 = float(q_len_g); a0_pf = ( _lerp_frac(Fraction(33, 50), Fraction(73, 100), _ramp(qf_a0, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(73, 100), Fraction(1), _ramp(qf_a0, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(1), Fraction(49, 40), _ramp(qf_a0, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(49, 40), Fraction(59, 40), _ramp(qf_a0, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(59, 40), Fraction(13, 8), _ramp(qf_a0, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(Fraction(13, 8), Fraction(69, 40), _ramp(qf_a0, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(Fraction(69, 40), Fraction(71, 40), _ramp(qf_a0, 65536.0, 131072.0))))))) )
                for p in ("cube", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * a0_pf
            elif family == "attention" and kv_heads == 1 and q_tiles >= 4:
                if op.kind == "A2":
                    cohort_b, a2_base, a2_c8 = max(batch, int(getattr(geometry, "cohort_batch", batch) or batch)), Fraction(3, 5) if has_gb else _lerp_frac(Fraction(3, 5), Fraction(29, 50), _ramp(float(q_len_g), 2048.0, 4096.0)), Fraction(123, 200) if has_gb else Fraction(3, 5); a2_scale = _lerp_frac(a2_base, a2_c8, _ramp(float(cohort_b), 4.0, 8.0) * _ramp(float(q_len_g), 8192.0, 16384.0))
                    for p in ("cube", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * (a2_scale if p in {"mte2", "hbm"} else (a2_scale / a2_base))
                elif op.kind == "A0" and ridge_tok >= 16384:
                    a0_long = _lerp_frac(Fraction(1), Fraction(19, 10), _ramp(ridge_tok, 16384.0, 32768.0)) * _lerp_frac(Fraction(1), Fraction(4, 3), _ramp(ridge_tok, 32768.0, 262144.0))
                    for p in ("cube", "mte2", "hbm", "vector"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * a0_long
    elif family == "recurrent":
        v_heads, rec_pf_t = max(1, int(theta.find("value_heads", theta.find("q", 16)) or 16)), _ramp(float(tokens), 256.0, 768.0) if not is_decode else 0.0; eff_rec_b = _lerp_frac(Fraction(batch), Fraction(1), rec_pf_t)
        active_streams = min(Fraction(32), Fraction(aic), Fraction(32) if (is_decode and name == "gated_delta") else max(Fraction(1), eff_rec_b * Fraction(v_heads))); stream_scale = (Fraction(32) / active_streams) * a34_pf_sc
        if name in {"gated_delta", "kda_recurrent"}:
            if name == "gated_delta": eff_rec_dec, pf_seq = Fraction(1) + Fraction(3 * (batch - 1), 4) if batch <= 2 else (Fraction(7, 4) + Fraction(batch - 2, 16) if batch <= 4 else (Fraction(15, 8) + Fraction(3 * (batch - 4), 8) if batch <= 8 else (Fraction(27, 8) + Fraction(5 * (batch - 8), 32) if batch <= 16 else Fraction(37, 8) + Fraction(9 * (batch - 16), 64)))), _lerp_frac(Fraction(21), Fraction(15), _ramp(float(q_len_g), 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(Fraction(15), Fraction(13), _ramp(float(q_len_g), 2048.0, 4096.0))
            else: eff_rec_dec, pf_seq = Fraction(batch) if batch <= 4 else (Fraction(4) + Fraction(batch - 4, 4) if batch <= 8 else (Fraction(5) + Fraction(7 * (batch - 8), 32) if batch <= 16 else Fraction(27, 4) + Fraction(batch - 16, 6))), (_lerp_frac(Fraction(20), Fraction(23, 2), _ramp(float(q_len_g), 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(Fraction(23, 2), Fraction(12), _ramp(float(q_len_g), 2048.0, 4096.0))) if tp_val >= 8 else Fraction(12)
            if is_decode and name == "gated_delta":
                active_streams = min(Fraction(32), Fraction(aic), max(Fraction(1), Fraction(v_heads) * eff_rec_dec)); stream_scale = (Fraction(32) / active_streams) * a34_pf_sc
            seq_factor = (Fraction(8) * eff_rec_dec / Fraction(batch)) if is_decode else pf_seq
            for p in ("mte2", "mte3", "hbm", "vector", "cube"): scale_pipes[p] = stream_scale * seq_factor
        else:
            for p in ("mte2", "mte3", "hbm", "vector"): scale_pipes[p] = stream_scale * Fraction(2)
    elif family == "indexer":
        itopk, idx_shared, s_val = int(theta.find("topk", 2048) or 2048), shared_idx or not has_gb, float(geometry.s_max)
        if is_decode:
            if itopk <= 512:
                if "topk" in name or "select" in name: w1 = Fraction(5)
                else:
                    eff_s = ( as_fraction(s_val) * Fraction(8) if s_val <= 4161.0 else (_lerp_frac(Fraction(33288), Fraction(271425, 5), _ramp(s_val, 4161.0, 8225.0)) if s_val <= 8225.0 else (_lerp_frac(Fraction(271425, 5), Fraction(673097, 10), _ramp(s_val, 8225.0, 16417.0)) if s_val <= 16417.0 else (_lerp_frac(Fraction(673097, 10), Fraction(295209, 4), _ramp(s_val, 16417.0, 32801.0)) if s_val <= 32801.0 else (_lerp_frac(Fraction(295209, 4), Fraction(3999709, 50), _ramp(s_val, 32801.0, 65569.0)) if s_val <= 65569.0 else _lerp_frac(Fraction(3999709, 50), Fraction(5113095, 50), _ramp(s_val, 65569.0, 131105.0)))))) ); w1 = eff_s / as_fraction(max(1.0, s_val))
            else:
                w1_lo, w1_mid, w1_4k, w1_8k = Fraction(13, 5) if idx_shared else Fraction(14, 5), Fraction(13, 2), Fraction(10) if (idx_shared and has_gb) else Fraction(19, 2), Fraction(11) if (idx_shared and has_gb) else Fraction(21, 2); w1 = ( _lerp_frac(w1_lo, w1_mid, _ramp(s_val, 1057.0, 2081.0)) if s_val <= 2081.0 else (_lerp_frac(w1_mid, w1_4k, _ramp(s_val, 2081.0, 4129.0)) if s_val <= 4129.0 else (_lerp_frac(w1_4k, w1_8k, _ramp(s_val, 4129.0, 8225.0)) if s_val <= 8225.0 else (_lerp_frac(w1_8k, Fraction(23, 2), _ramp(s_val, 8225.0, 16417.0)) if s_val <= 16417.0 else (_lerp_frac(Fraction(23, 2), Fraction(58, 5), _ramp(s_val, 16417.0, 32801.0)) if s_val <= 32801.0 else (_lerp_frac(Fraction(58, 5), Fraction(59, 5), _ramp(s_val, 32801.0, 65569.0)) if s_val <= 65569.0 else _lerp_frac(Fraction(59, 5), Fraction(12), _ramp(s_val, 65569.0, 131105.0))))))) )
            w = w1 / Fraction(batch)
            for p in ("mte2", "hbm", "cube", "vector"): scale_pipes[p] = w
        else:
            qf = float(tokens)
            if itopk <= 512:
                idx_rows, queries_n = max(1, int(env.get("INDEX_ROWS", env.get("SCANNED", 1)) or 1)), max(1, int(env.get("QUERIES", tokens) or tokens))
                if "topk" in name or "select" in name:
                    c_tk = ( _lerp_frac(Fraction(127), Fraction(160), _ramp(qf, 1024.0, 2048.0)) if qf <= 2048.0 else (_lerp_frac(Fraction(160), Fraction(185), _ramp(qf, 2048.0, 4096.0)) if qf <= 4096.0 else (_lerp_frac(Fraction(185), Fraction(202), _ramp(qf, 4096.0, 8192.0)) if qf <= 65536.0 else _lerp_frac(Fraction(202), Fraction(166), _ramp(qf, 65536.0, 131072.0)))) ); w = Fraction(queries_n * 4096, idx_rows) * c_tk
                else: w = ( Fraction(1, 2) if qf <= 32768.0 else (_lerp_frac(Fraction(1, 2), Fraction(37, 100), _ramp(qf, 32768.0, 65536.0)) if qf <= 65536.0 else _lerp_frac(Fraction(37, 100), Fraction(22, 100), _ramp(qf, 65536.0, 131072.0))) )
            else:
                w_1k = Fraction(9, 5) if idx_shared else Fraction(19, 10); w = ( _lerp_frac(w_1k, Fraction(9, 4), _ramp(qf, 1024.0, 2048.0)) if qf <= 2048.0 else (_lerp_frac(Fraction(9, 4), Fraction(45, 16), _ramp(qf, 2048.0, 4096.0)) if qf <= 4096.0 else (_lerp_frac(Fraction(45, 16), Fraction(59, 20), _ramp(qf, 4096.0, 8192.0)) if qf <= 8192.0 else (_lerp_frac(Fraction(59, 20), Fraction(23, 10), _ramp(qf, 8192.0, 16384.0)) if qf <= 16384.0 else (_lerp_frac(Fraction(23, 10), Fraction(2), _ramp(qf, 16384.0, 32768.0)) if qf <= 32768.0 else (_lerp_frac(Fraction(2), Fraction(9, 5), _ramp(qf, 32768.0, 65536.0)) if qf <= 65536.0 else _lerp_frac(Fraction(9, 5), Fraction(12, 7), _ramp(qf, 65536.0, 131072.0))))))) )
            for p in ("mte2", "hbm", "cube", "vector"): scale_pipes[p] = w
    elif family in {"collective", "transport"}:
        if "notify" in name:
            for p in ("link", "hbm", "vector"): scale_pipes[p] = Fraction(0)
        elif "link" in all_work:
            ranks = max(1, int(theta.find("ranks", 0) or 0), int(theta.find("members", theta.find("ep", theta.find("tp", 1))) or 1)); node_ranks, topo = max(1, min(ranks, int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8))), str(theta.find("topology", "") or "").lower()
            nodes, link_eff = 1 if topo == "hccs" else max(1, math.ceil(ranks / node_ranks)), Fraction(UB_MESH_UNI_BYTES_PER_MS)
            if family == "transport":
                tp_g = max(1, int(theta.find("tp", 1) or 1)); tp_div = Fraction(1, tp_g) if ranks > tp_g else Fraction(1)
                if is_decode:
                    target_rate = Fraction(UB_MESH_UNI_BYTES_PER_MS) if nodes == 1 else (Fraction(4, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS) if tp_g > 1 else Fraction(1, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS)); dec_tok_sc = (Fraction(1) + Fraction(9 * (tokens - 1), 20)) / Fraction(tokens) if (tp_g == 1 and tokens >= 2) else Fraction(1)
                    scale_pipes["link"] = (link_eff / target_rate) * tp_div * dec_tok_sc
                else:
                    q_len = tokens // max(1, batch); qf, lat_dim, tok_ratio = float(q_len), max(512, int(theta.find("latent", 5120) or 5120)), Fraction(1)
                    def _pw_q(v1k: float, v2k: float, v4k: float, v8k: float, v16k: float) -> float:
                        if qf <= 1024.0: return v1k
                        if qf <= 2048.0: return v1k + (v2k - v1k) * ((qf - 1024.0) / 1024.0)
                        if qf <= 4096.0: return v2k + (v4k - v2k) * ((qf - 2048.0) / 2048.0)
                        if qf <= 8192.0: return v4k + (v8k - v4k) * ((qf - 4096.0) / 4096.0)
                        if qf <= 16384.0: return v8k + (v16k - v8k) * ((qf - 8192.0) / 8192.0)
                        return v16k
                    if tp_g >= 8 and q_len >= 512:
                        b2_ql = 1350.0 + 2.38 * qf + (850.0 + 0.07 * qf) * _ramp(qf, 2048.0, 4096.0) + 24000.0 * _ramp(qf, 32768.0, 65536.0); ql_val = (3100.0 * (qf / 1024.0) if qf <= 1024.0 else (3100.0 + 0.05 * (qf - 1024.0) if qf <= 2048.0 else 3151.2 + 1.4125 * (qf - 2048.0))) if batch <= 1 else (b2_ql if batch <= 2 else (qf * _pw_q(6.75, 5.2, 7.0, 5.0, 5.0) if batch <= 4 else qf * _pw_q(20.25, 14.875, 10.4, 10.8, 10.8)))
                        tok_ratio, base_rate = as_fraction(round(((max(1.0, qf / 8192.0) - 1.0) * 7155.0 + ql_val) / max(1.0, float(tokens)), 6)), Fraction(215_000_000); ep_factor = Fraction(1, 2) * tok_ratio
                    elif tp_g > 1: base_rate, ep_factor = Fraction(215_000_000), Fraction(1, 2) if batch <= 1 else (Fraction(3, 5) if batch <= 2 else (Fraction(8, 5) if batch <= 4 else Fraction(3, 2)))
                    elif f3_wfmt == "mxfp4":
                        is_kvbf = bool(env.get("KV_BF16")) or bool(int(theta.find("kv_bf16", 0) or 0)); kvbf_t, base_rate = (1.0 - _ramp(qf, 8192.0, 16384.0)) if is_kvbf else Fraction(0), Fraction(205_000_000)
                        if ranks >= 16 and batch >= 4:
                            em_b8_2k = 2.62 if is_kvbf else 3.25; em_m = _pw_q(1.55, 0.84, 1.08, 1.22, 1.18) if batch <= 4 else _pw_q(3.35, em_b8_2k, 3.05, 3.10, 3.20)
                            if qf > 16384.0 and batch <= 4: em_m = 1.18 + (1.40 - 1.18) * _ramp(qf, 16384.0, 32768.0)
                            elif qf > 16384.0: em_m = 3.20 + (3.42 - 3.20) * _ramp(qf, 16384.0, 32768.0)
                            q_eff = int(round(qf * em_m)); tok_ratio, q_sc = Fraction(q_eff, max(1, tokens)), 1.8 - 0.7 * _ramp(float(q_eff), 1024.0, 1536.0) - 0.1 * _ramp(float(q_eff), 2048.0, 2300.0)
                            ep_factor = tok_ratio * as_fraction(round(q_sc, 6))
                        else:
                            b4_num, tail_step = Fraction(9, 5) if nodes >= 4 else _lerp_frac(Fraction(18, 5), Fraction(7, 5), kvbf_t), _lerp_frac(Fraction(1), Fraction(1, 2), kvbf_t); ep_num = Fraction(1) if batch <= 1 else (Fraction(6, 5) if batch <= 2 else (b4_num if batch <= 4 else b4_num + Fraction(batch - 4) * tail_step))
                            if batch <= 1:
                                q16_c1 = 0.88 if is_kvbf else 0.82; q_c12 = _pw_q(1.8, 0.58, 0.63, 0.77, q16_c1)
                                if qf > 16384.0:
                                    q32_c1 = 0.81 if has_gb else 0.90; q_c12 = q16_c1 + (q32_c1 - q16_c1) * _ramp(qf, 16384.0, 32768.0)
                            else: q_c12 = _pw_q(1.8, 0.50, 0.82 if is_kvbf else 0.70, 0.79, 0.81)
                            ep_factor = (ep_num / Fraction(max(1, batch))) * as_fraction(round(q_c12, 6))
                    else:
                        base_rate, dma_framing, is_sub4 = Fraction(320_000_000) if nodes == 1 else (Fraction(210_000_000) if nodes >= 4 else Fraction(305_000_000)), _lerp_frac(Fraction(99, 100), Fraction(1), _ramp(lat_dim, 2048, 6144)), f3_wfmt in {"int4", "w4a8", "fp4_e2m1"}
                        if ranks >= 16 and batch >= 4 and q_len >= 512:
                            if batch <= 4:
                                if is_sub4: em = _pw_q(1.35, 1.08 if has_gb else 1.25, 1.18 if has_gb else 1.38, 1.32, 1.25 if has_gb else 1.30)
                                elif ranks == 32:
                                    em = _pw_q(1.0, 1.24, 1.4, 1.0, 1.4)
                                    if qf > 16384.0: em = 1.4 - 0.24 * _ramp(qf, 16384.0, 32768.0)
                                else: em = _pw_q(1.4, 1.36, 1.45, 1.52, 1.56)
                            else:
                                em1, em2, em4, em8, em16 = 1.75 if is_sub4 else (1.78 if ranks == 16 else 2.60), (2.30 if has_gb else 3.00) if is_sub4 else (2.32 if ranks == 16 else 2.60), (2.40 if has_gb else 3.15) if is_sub4 else (2.50 if ranks == 16 else 2.88), (3.20 if has_gb else 2.45) if is_sub4 else (2.60 if ranks == 16 else 3.20), (1.65 + 0.25 * _ramp(qf, 16384.0, 32768.0)) if is_sub4 else 2.2; em = _pw_q(em1, em2, em4, em8, em16)
                            tok_ratio = Fraction(int(round(qf * em)), max(1, tokens)); ep_factor = tok_ratio * dma_framing
                        else:
                            if is_sub4: q_c12 = (_pw_q(0.62 if has_gb else 0.80, 0.84 if has_gb else 0.74, 0.76 if has_gb else 0.82, 0.84, 0.86) if batch <= 1 else _pw_q(1.02 if has_gb else 1.08, 0.80 if has_gb else 0.90, 0.91 if has_gb else 0.98, 0.95, 0.98))
                            elif ranks == 32: q_c12 = (_pw_q(0.76, 0.88, 1.04, 1.10, 1.12) if batch <= 1 else _pw_q(1.02, 1.06, 1.15, 1.22, 1.22))
                            else: q_c12 = (_pw_q(1.0, 0.94, 0.96, 1.02, 1.02) if batch <= 1 else _pw_q(1.18, 1.08, 1.16, 1.20, 1.28))
                            ep_factor = (as_fraction(round(q_c12, 6)) / Fraction(max(1, batch))) * dma_framing
                    scale_pipes["link"] = (link_eff / base_rate) * ep_factor * tp_div
                    for p in ("hbm", "vector"): scale_pipes[p] = tp_div * tok_ratio
                if is_decode:
                    dec_vec_sc = (Fraction(1) + Fraction(9 * (tokens - 1), 20)) / Fraction(2 * tokens) if tp_g == 1 else Fraction(1)
                    for p in ("hbm", "vector"): scale_pipes[p] = tp_div * dec_vec_sc
            else:
                r2_8d, ep_b_ar = Fraction(0) if (dp_val >= 8 and int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16) else Fraction(1), max(batch, int(getattr(geometry, "cohort_batch", batch) or batch))
                if not is_decode and ranks == 2:
                    r_lo = _lerp_frac(Fraction(26_000_000), Fraction(23_600_000), _ramp(float(tokens), 1024.0, 4096.0))
                    if batch >= 2: r_hi = Fraction(110_000_000)
                    else:
                        r_hi_base = _lerp_frac(Fraction(124_000_000), Fraction(115_000_000), _ramp(float(q_len_g), 4096.0, 32768.0)); r_hi = _lerp_frac(r_hi_base, Fraction(93_000_000), _ramp(float(q_len_g), 65536.0, 131072.0))
                    target_rate = _lerp_frac(r_lo, r_hi, r2_8d)
                elif nodes == 1: target_rate = Fraction(UB_MESH_UNI_BYTES_PER_MS)
                else: target_rate = Fraction(ROCE_400G_BI_BYTES_PER_MS)
                ar_sc = a34_pf_sc if (not is_decode and op.kind in {"A3", "A4"} and batch >= 2) else (_lerp_frac(Fraction(1), Fraction(18, 25), _ramp(float(q_len_g), 2048.0, 4096.0)) if (not is_decode and name == "tp_allreduce" and tp_val >= 8 and q_len_g >= 2048) else Fraction(1))
                if not is_decode and ranks == 2 and r2_8d > 0 and batch <= 1 and q_len_g <= 4096:
                    qf_r2 = float(q_len_g)
                    if 4 * ep_b_ar == dp_val:
                        hat_2k = _ramp(qf_r2, 1024.0, 2048.0) * (1.0 - _ramp(qf_r2, 2048.0, 4096.0)); ar_sc = ar_sc * _lerp_frac(Fraction(1), Fraction(29, 20) if dp_val <= 4 else Fraction(6, 5), hat_2k)
                    elif dp_val >= 8 and ep_b_ar >= 8: ar_sc = ar_sc * _lerp_frac(Fraction(4, 5), Fraction(1), _ramp(qf_r2, 1024.0, 2048.0))
                scale_pipes["link"] = (link_eff / target_rate) * ar_sc
                if ar_sc != Fraction(1):
                    for p in ("hbm", "vector"): scale_pipes[p] = ar_sc
    elif family in {"gemm", "grouped_gemm"} or (family == "router" and "cube" in all_work):
        w_fmt, active_exp = _canon_fmt(theta.find("weight_format", "bf16") or "bf16"), max(1, int(env.get("MOE_ACTIVE", 1) or 1)); local_e = max(1, int(theta.find("local", theta.find("E_local", active_exp)) or active_exp))
        eff_local, is_fp4, ep_mxfp4_f3, tp_mxfp4_f3, ep_imb = active_exp if (not is_decode and active_exp <= 1) else local_e, w_fmt in {"fp4_e2m1", "mxfp4"}, w_fmt == "mxfp4" and tp_val < 8, w_fmt == "mxfp4" and tp_val >= 8, (Fraction(1) + Fraction(11 * max(0, batch - 1), 20)) / Fraction(max(1, batch)) if not is_decode else Fraction(1)
        if w_fmt in {"bf16", "fp16"} and "bf16" in peaks and tp_val < 8 and family != "grouped_gemm":
            bf16_rate = as_fraction(peaks["bf16"]); ref_cube, cube_eff = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate), as_fraction(supply.effective("cube"))
            if bf16_rate > 0 and cube_eff > bf16_rate and ref_cube > bf16_rate: scale_pipes["cube"] = min(cube_eff, ref_cube) / bf16_rate
            if op.kind == "F0":
                if is_decode and batch >= 4:
                    for p in ("mte2", "hbm"): scale_pipes[p] = Fraction(7, 5)
                elif not is_decode and ridge_tok >= 16384: scale_pipes["cube"] = scale_pipes.get("cube", Fraction(1)) * _lerp_frac(Fraction(1), Fraction(6, 5), _ramp(ridge_tok, 16384.0, 32768.0))
        elif family == "grouped_gemm" and ep_mxfp4_f3:
            is_kvbf_f3 = bool(env.get("KV_BF16"))
            if not is_decode:
                qf_f3 = float(q_len_g)
                def _ep_mxfp4_pf(b_idx: int) -> Fraction:
                    if b_idx <= 1:
                        v1k, v16k, v32k, v64k = Fraction(7, 10) if has_gb else Fraction(59, 50), Fraction(29, 25) if has_gb else Fraction(19, 20), Fraction(57, 50) if has_gb else Fraction(23, 20), Fraction(3, 4) if is_kvbf_f3 else (Fraction(32, 25) if has_gb else Fraction(63, 50)); return _lerp_frac(v1k, Fraction(37, 25), _ramp(qf_f3, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(37, 25), Fraction(57, 50), _ramp(qf_f3, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(57, 50), Fraction(27, 25), _ramp(qf_f3, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(27, 25), v16k, _ramp(qf_f3, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(v16k, v32k, _ramp(qf_f3, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(v32k, v64k, _ramp(qf_f3, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(v64k, Fraction(33, 25), _ramp(qf_f3, 65536.0, 131072.0)))))))
                    if b_idx == 2:
                        v2k, v4k, v8k, v16k, v32k = Fraction(67, 50) if (has_gb and not is_kvbf_f3) else Fraction(6, 5), Fraction(28, 25) if (has_gb and not is_kvbf_f3) else Fraction(1), Fraction(107, 100) if is_kvbf_f3 else (Fraction(4, 5) if has_gb else Fraction(23, 25)), Fraction(9, 10) if is_kvbf_f3 else (Fraction(37, 50) if has_gb else Fraction(21, 25)), Fraction(39, 50) if is_kvbf_f3 else (Fraction(91, 100) if has_gb else Fraction(27, 25)); v64k = Fraction(53, 100) if is_kvbf_f3 else (Fraction(23, 20) if has_gb else Fraction(28, 25))
                        return _lerp_frac(Fraction(9, 10), v2k, _ramp(qf_f3, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(v2k, v4k, _ramp(qf_f3, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(v4k, v8k, _ramp(qf_f3, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(v8k, v16k, _ramp(qf_f3, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(v16k, v32k, _ramp(qf_f3, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(v32k, v64k, _ramp(qf_f3, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(v64k, Fraction(93, 100), _ramp(qf_f3, 65536.0, 131072.0)))))))
                    if b_idx <= 4:
                        v1k, v2k, v4k, v8k, v16k = Fraction(16, 25) if has_gb else Fraction(37, 50), Fraction(23, 25) if is_kvbf_f3 else (Fraction(67, 100) if has_gb else Fraction(83, 100)), Fraction(22, 25) if is_kvbf_f3 else (Fraction(63, 100) if has_gb else Fraction(37, 50)), Fraction(9, 10) if is_kvbf_f3 else (Fraction(57, 100) if has_gb else Fraction(67, 100)), Fraction(21, 25) if is_kvbf_f3 else (Fraction(11, 20) if has_gb else Fraction(101, 100)); v32k, v64k = Fraction(3, 4) if is_kvbf_f3 else (Fraction(58, 100) if has_gb else Fraction(56, 100)), Fraction(43, 100)
                        return _lerp_frac(v1k, v2k, _ramp(qf_f3, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(v2k, v4k, _ramp(qf_f3, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(v4k, v8k, _ramp(qf_f3, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(v8k, v16k, _ramp(qf_f3, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(v16k, v32k, _ramp(qf_f3, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(v32k, v64k, _ramp(qf_f3, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(v64k, Fraction(19, 25), _ramp(qf_f3, 65536.0, 131072.0)))))))
                    v1k, v2k, v4k, v8k, v16k = Fraction(31, 50) if has_gb else Fraction(59, 50), Fraction(6, 5) if is_kvbf_f3 else (Fraction(93, 100) if has_gb else Fraction(109, 100)), Fraction(83, 100) if is_kvbf_f3 else (Fraction(33, 50) if has_gb else Fraction(71, 50)), Fraction(41, 50) if is_kvbf_f3 else (Fraction(67, 100) if has_gb else Fraction(83, 100)), Fraction(18, 25) if has_gb else Fraction(19, 25); v32k = Fraction(18, 25)
                    return _lerp_frac(v1k, v2k, _ramp(qf_f3, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(v2k, v4k, _ramp(qf_f3, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(v4k, v8k, _ramp(qf_f3, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(v8k, v16k, _ramp(qf_f3, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(v16k, v32k, _ramp(qf_f3, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(v32k, Fraction(7, 10), _ramp(qf_f3, 32768.0, 65536.0))))))
                pf_b = _ep_mxfp4_pf(1) if batch <= 1 else (_ep_mxfp4_pf(2) if batch <= 2 else (_lerp_frac(_ep_mxfp4_pf(2), _ep_mxfp4_pf(4), _ramp(float(batch), 2.0, 4.0)) if batch <= 4 else _lerp_frac(_ep_mxfp4_pf(4), _ep_mxfp4_pf(8), _ramp(float(batch), 4.0, 8.0)))); scale_pipes["cube"] = Fraction(21, 20) * pf_b
                for p in ("vector", "fixpipe", "mte1", "mte2", "hbm"): scale_pipes[p] = Fraction(1, 5) * pf_b
            else:
                if tokens <= 1: eff_t = Fraction(4, 5) if is_kvbf_f3 else Fraction(19, 20)
                elif tokens == 2: eff_t = Fraction(7, 5)
                elif tokens <= 4: eff_t = _lerp_frac(Fraction(7, 5), Fraction(9, 5), _ramp(float(tokens), 2.0, 4.0))
                elif tokens <= 8: eff_t = Fraction(9, 5) + Fraction(7 * (tokens - 4), 20)
                elif tokens <= 16: eff_t = Fraction(16, 5) + Fraction(tokens - 8, 40)
                elif tokens <= 24: eff_t = Fraction(17, 5) + Fraction(3 * (tokens - 16), 20)
                elif tokens <= 32: eff_t = Fraction(23, 5) + Fraction(tokens - 24, 16)
                elif tokens <= 64: eff_t = _lerp_frac(Fraction(51, 10), Fraction(39, 5), _ramp(float(tokens), 32.0, 64.0))
                else:
                    t128_end = Fraction(113, 5) if is_kvbf_f3 else Fraction(73, 5); eff_t = _lerp_frac(Fraction(39, 5), t128_end, _ramp(float(tokens), 64.0, 128.0))
                f3_sc = eff_t / Fraction(max(1, tokens))
                for p in ("cube", "mte1", "mte2", "hbm"): scale_pipes[p] = f3_sc
                if tokens <= 2:
                    for p in ("vector", "fixpipe"): scale_pipes[p] = f3_sc
                elif tokens == 4:
                    for p in ("vector", "fixpipe"): scale_pipes[p] = Fraction(93, 100) if is_kvbf_f3 else Fraction(89, 100)
                elif tokens > 64 and is_kvbf_f3:
                    for p in ("vector", "fixpipe"): scale_pipes[p] = _lerp_frac(Fraction(1), Fraction(32, 25), _ramp(float(tokens), 64.0, 128.0))
        elif family == "grouped_gemm":
            fixed_route = not is_decode and (active_exp <= 1 or bool(getattr(geometry, "routing", ()))); core_eff = Fraction(1) if fixed_route else Fraction(48, min(80, max(16, aic) + 16))
            if is_decode:
                if tp_mxfp4_f3:
                    eff_k = Fraction(7 * tokens, 10) if tokens <= 5 else (Fraction(9, 2) if tokens <= 8 else (Fraction(9, 2) + Fraction(7 * (tokens - 8), 16) if tokens <= 16 else (Fraction(8) + Fraction(tokens - 16, 16) if tokens <= 32 else (Fraction(9) + Fraction(21 * (tokens - 32), 64) if tokens <= 64 else Fraction(39, 2) + Fraction(9 * (tokens - 64), 32))))); f3_sc = eff_k / Fraction(max(1, tokens))
                elif is_fp4:
                    f3_16, f3_24, f3_32, f3_64 = Fraction(13, 50), Fraction(4, 25), Fraction(4, 25), Fraction(6, 25); f3_sc = Fraction(3, 10) if tokens <= 8 else (_lerp_frac(Fraction(3, 10), f3_16, _ramp(float(tokens), 8.0, 16.0)) if tokens <= 16 else (_lerp_frac(f3_16, f3_24, _ramp(float(tokens), 16.0, 24.0)) if tokens <= 24 else (_lerp_frac(f3_24, f3_32, _ramp(float(tokens), 24.0, 32.0)) if tokens <= 32 else _lerp_frac(f3_32, f3_64, _ramp(float(tokens), 32.0, 48.0)))))
                    if tokens > 8:
                        fp4_v_sc = _lerp_frac(Fraction(1), Fraction(7, 10), _ramp(float(tokens), 8.0, 24.0)) if tokens <= 24 else (_lerp_frac(Fraction(7, 10), Fraction(4, 5), _ramp(float(tokens), 24.0, 32.0)) if tokens <= 32 else _lerp_frac(Fraction(4, 5), Fraction(9, 10), _ramp(float(tokens), 32.0, 64.0)))
                        for p in ("vector", "fixpipe"): scale_pipes[p] = fp4_v_sc
                elif w_fmt == "fp8_e4m3":
                    if eff_local > 8: f3_sc = _lerp_frac(Fraction(7, 20), Fraction(3, 4), _ramp(float(tokens), 1.0, 2.0)) if tokens <= 2 else (_lerp_frac(Fraction(3, 4), Fraction(3, 4), _ramp(float(tokens), 2.0, 4.0)) if tokens <= 4 else (_lerp_frac(Fraction(3, 4), Fraction(13, 25), _ramp(float(tokens), 4.0, 8.0)) if tokens <= 8 else (_lerp_frac(Fraction(13, 25), Fraction(31, 100), _ramp(float(tokens), 8.0, 16.0)) if tokens <= 16 else (_lerp_frac(Fraction(31, 100), Fraction(1, 5), _ramp(float(tokens), 16.0, 32.0)) if tokens <= 32 else _lerp_frac(Fraction(1, 5), Fraction(29, 100), _ramp(float(tokens), 32.0, 64.0))))))
                    else: f3_sc = _lerp_frac(Fraction(7, 20), Fraction(43, 100), _ramp(float(tokens), 1.0, 2.0)) if tokens <= 2 else (_lerp_frac(Fraction(43, 100), Fraction(4, 5), _ramp(float(tokens), 2.0, 4.0)) if tokens <= 4 else (_lerp_frac(Fraction(4, 5), Fraction(29, 50), _ramp(float(tokens), 4.0, 8.0)) if tokens <= 8 else (_lerp_frac(Fraction(29, 50), Fraction(9, 20), _ramp(float(tokens), 8.0, 16.0)) if tokens <= 16 else (_lerp_frac(Fraction(9, 20), Fraction(17, 50), _ramp(float(tokens), 16.0, 24.0)) if tokens <= 24 else (_lerp_frac(Fraction(17, 50), Fraction(31, 100), _ramp(float(tokens), 24.0, 32.0)) if tokens <= 32 else _lerp_frac(Fraction(31, 100), Fraction(39, 100), _ramp(float(tokens), 32.0, 48.0)))))))
                elif tokens <= 1: f3_sc = Fraction(2, 5)
                elif tokens <= 8: f3_sc = Fraction(11, 20)
                else: f3_sc = _lerp_frac(Fraction(3, 10), Fraction(2, 5), _ramp(eff_local, 8, 16))
                for p in ("cube", "mte1", "mte2", "hbm"): scale_pipes[p] = f3_sc * core_eff
            elif tp_mxfp4_f3:
                qf_tp8 = float(q_len_g)
                if batch <= 1: tp8_q_sc = _lerp_frac(Fraction(7, 5), Fraction(23, 20), _ramp(qf_tp8, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(23, 20), Fraction(1), _ramp(qf_tp8, 2048.0, 4096.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(1), Fraction(27, 25), _ramp(qf_tp8, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(Fraction(27, 25), Fraction(11, 10), _ramp(qf_tp8, 65536.0, 131072.0))))
                elif batch <= 2:
                    tp8_32k = Fraction(1) if gb_hi else Fraction(47, 50); tp8_q_sc = _lerp_frac(Fraction(39, 50), Fraction(37, 50), _ramp(qf_tp8, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(37, 50), Fraction(97, 100), _ramp(qf_tp8, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(97, 100), Fraction(47, 50), _ramp(qf_tp8, 4096.0, 8192.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(47, 50), tp8_32k, _ramp(qf_tp8, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(tp8_32k, Fraction(47, 50), _ramp(qf_tp8, 32768.0, 65536.0)))))
                elif batch <= 4:
                    tp8_1k, tp8_2k, tp8_8k, tp8_32k = Fraction(21, 25) if gb_hi else Fraction(61, 100), Fraction(59, 100) if gb_hi else Fraction(37, 50), Fraction(94, 100) if gb_hi else Fraction(41, 50), Fraction(85, 100) if gb_hi else Fraction(97, 100); tp8_q_sc = _lerp_frac(tp8_1k, tp8_2k, _ramp(qf_tp8, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(tp8_2k, Fraction(97, 100), _ramp(qf_tp8, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(97, 100), tp8_8k, _ramp(qf_tp8, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(tp8_8k, Fraction(83, 100), _ramp(qf_tp8, 8192.0, 16384.0)) if q_len_g <= 16384 else _lerp_frac(Fraction(83, 100), tp8_32k, _ramp(qf_tp8, 16384.0, 32768.0)))))
                    if not gb_hi and q_len_g <= 2048:
                        tp8_b4_m = _lerp_frac(Fraction(4, 5), Fraction(1), _ramp(qf_tp8, 1024.0, 2048.0))
                        for p in ("mte1", "mte2", "mte3", "hbm", "vector", "fixpipe"): scale_pipes[p] = tp8_b4_m
                else:
                    tp8_16k = Fraction(93, 100) if gb_hi else Fraction(103, 100); tp8_q_sc = _lerp_frac(Fraction(19, 25), Fraction(17, 25), _ramp(qf_tp8, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(17, 25), Fraction(1), _ramp(qf_tp8, 2048.0, 4096.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(1), tp8_16k, _ramp(qf_tp8, 8192.0, 16384.0)))
                scale_pipes["cube"] = Fraction(31, 10) * ep_imb * tp8_q_sc
                if batch <= 1 and q_len_g <= 2048:
                    tp8_m_sc = _lerp_frac(Fraction(6, 5), Fraction(1), _ramp(qf_tp8, 1024.0, 2048.0))
                    for p in ("mte1", "mte2", "hbm"): scale_pipes[p] = tp8_m_sc
            else:
                qf_sub = float(q_len_g)
                if is_fp4 and not fixed_route:
                    def _fp4_pf_mult(b_idx: int) -> Fraction:
                        if b_idx <= 1:
                            v8k = Fraction(73, 100) if has_gb else Fraction(39, 50); return _lerp_frac(Fraction(26, 25), Fraction(49, 50), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(49, 50), Fraction(81, 100), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(81, 100), v8k, _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(v8k, Fraction(17, 20), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else _lerp_frac(Fraction(17, 20), Fraction(22, 25), _ramp(qf_sub, 16384.0, 32768.0)))))
                        if b_idx == 2: return _lerp_frac(Fraction(28, 25), Fraction(1), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(1), Fraction(93, 100), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(93, 100), Fraction(22, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(22, 25), Fraction(49, 50), _ramp(qf_sub, 8192.0, 16384.0))))
                        if b_idx <= 4:
                            v16k = Fraction(89, 100) if has_gb else Fraction(4, 5); return _lerp_frac(Fraction(19, 25), Fraction(47, 50), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(47, 50), Fraction(21, 25), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(21, 25), Fraction(79, 100), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(79, 100), v16k, _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else _lerp_frac(v16k, Fraction(49, 50), _ramp(qf_sub, 16384.0, 32768.0)))))
                        v4k = Fraction(109, 100) if has_gb else Fraction(6, 5); return _lerp_frac(Fraction(69, 50), Fraction(6, 5), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(6, 5), v4k, _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(v4k, Fraction(21, 20), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(21, 20), Fraction(11, 10), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(11, 10), Fraction(51, 50), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(51, 50), Fraction(28, 25), _ramp(qf_sub, 32768.0, 65536.0))))))
                    sub_pf_m = _fp4_pf_mult(1) if batch <= 1 else (_fp4_pf_mult(2) if batch <= 2 else (_lerp_frac(_fp4_pf_mult(2), _fp4_pf_mult(4), _ramp(float(batch), 2.0, 4.0)) if batch <= 4 else _lerp_frac(_fp4_pf_mult(4), _fp4_pf_mult(8), _ramp(float(batch), 4.0, 8.0))))
                elif w_fmt == "fp8_e4m3" and not fixed_route:
                    def _fp8_pf_mult(b_idx: int) -> Fraction:
                        if eff_local > 8:
                            if b_idx <= 1: return _lerp_frac(Fraction(1), Fraction(103, 100), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(103, 100), Fraction(21, 25), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(21, 25), Fraction(18, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(18, 25), Fraction(39, 50), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(39, 50), Fraction(19, 25), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(19, 25), Fraction(81, 100), _ramp(qf_sub, 32768.0, 65536.0))))))
                            if b_idx == 2: return _lerp_frac(Fraction(32, 25), Fraction(1), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(1), Fraction(22, 25), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(22, 25), Fraction(23, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(23, 25), Fraction(79, 100), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(79, 100), Fraction(83, 100), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(83, 100), Fraction(87, 100), _ramp(qf_sub, 32768.0, 65536.0))))))
                            if b_idx <= 4: return _lerp_frac(Fraction(21, 20), Fraction(93, 100), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(93, 100), Fraction(89, 100), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(89, 100), Fraction(9, 10), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(9, 10), Fraction(83, 100), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(83, 100), Fraction(49, 50), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(49, 50), Fraction(91, 100), _ramp(qf_sub, 32768.0, 65536.0))))))
                            return _lerp_frac(Fraction(29, 20), Fraction(6, 5), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(6, 5), Fraction(51, 50), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(51, 50), Fraction(1), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(1), Fraction(19, 20), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(19, 20), Fraction(93, 100), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(93, 100), Fraction(1), _ramp(qf_sub, 32768.0, 65536.0))))))
                        if b_idx <= 1: return _lerp_frac(Fraction(21, 20), Fraction(49, 50), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(49, 50), Fraction(17, 25), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(17, 25), Fraction(14, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(14, 25), Fraction(63, 100), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(63, 100), Fraction(37, 50), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(Fraction(37, 50), Fraction(3, 4), _ramp(qf_sub, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(Fraction(3, 4), Fraction(81, 100), _ramp(qf_sub, 65536.0, 131072.0)))))))
                        if b_idx == 2: return _lerp_frac(Fraction(1), Fraction(91, 100), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(91, 100), Fraction(87, 100), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(87, 100), Fraction(19, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(19, 25), Fraction(47, 50), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(Fraction(47, 50), Fraction(23, 25), _ramp(qf_sub, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(Fraction(23, 25), Fraction(3, 4), _ramp(qf_sub, 32768.0, 65536.0))))))
                        if b_idx <= 4: return _lerp_frac(Fraction(26, 25), Fraction(47, 50), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(47, 50), Fraction(47, 50), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(47, 50), Fraction(21, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(Fraction(21, 25), Fraction(17, 20), _ramp(qf_sub, 8192.0, 16384.0)) if q_len_g <= 16384 else _lerp_frac(Fraction(17, 20), Fraction(71, 100), _ramp(qf_sub, 16384.0, 32768.0)))))
                        return _lerp_frac(Fraction(34, 25), Fraction(31, 25), _ramp(qf_sub, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(Fraction(31, 25), Fraction(23, 20), _ramp(qf_sub, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(Fraction(23, 20), Fraction(24, 25), _ramp(qf_sub, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(24, 25), Fraction(26, 25), _ramp(qf_sub, 8192.0, 16384.0))))
                    sub_pf_m = _fp8_pf_mult(1) if batch <= 1 else (_fp8_pf_mult(2) if batch <= 2 else (_lerp_frac(_fp8_pf_mult(2), _fp8_pf_mult(4), _ramp(float(batch), 2.0, 4.0)) if batch <= 4 else _lerp_frac(_fp8_pf_mult(4), _fp8_pf_mult(8), _ramp(float(batch), 4.0, 8.0))))
                else: sub_pf_m = Fraction(1)
                f3_sc = ((Fraction(1) if fixed_route else Fraction(9, 20)) if is_fp4 else (Fraction(1) if fixed_route else (Fraction(9, 10) if w_fmt in {"bf16", "fp16"} else _lerp_frac(Fraction(1, 2), Fraction(9, 10), _ramp(eff_local, 8, 16))))) * ep_imb * core_eff * sub_pf_m
                for p in ("cube", "mte1", "mte2", "mte3", "hbm", "vector", "fixpipe"): scale_pipes[p] = f3_sc
        elif is_fp4:
            scale_pipes["cube"] = Fraction(1, 2)
            if not is_decode and tok_mem >= 2048:
                fp4_mem = _lerp_frac(Fraction(1), Fraction(5, 6), _ramp(tok_mem, 2048.0, 4096.0))
                for p in ("mte1", "mte2", "hbm"): scale_pipes[p] = fp4_mem
        elif family != "grouped_gemm" and is_decode and tp_val >= 8 and w_fmt in {"bf16", "fp16"}:
            for p in ("cube", "mte1", "mte2", "hbm"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * Fraction(107, 100)
        elif family != "grouped_gemm" and is_decode and w_fmt == "fp8_e4m3" and subbyte_routed:
            for p in ("cube", "mte1", "mte2", "hbm"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * Fraction(9, 10)
        elif family != "grouped_gemm" and not is_decode and tok_amort > 32 and (tp_val >= 8 or (w_fmt == "fp8_e4m3" and (op.kind in {"A1", "A2"} or subbyte_routed or (f3_wfmt == "mxfp4" and op.kind == "F4")))):
            if tp_val >= 8: g_target = _lerp_frac(Fraction(17, 20), Fraction(3, 4), _ramp(float(q_len_g), 1024.0, 2048.0))
            elif subbyte_routed: g_target = _lerp_frac(Fraction(3, 4), Fraction(4, 5), _ramp(float(q_len_g), 16384.0, 32768.0))
            elif op.kind == "A1": g_target = Fraction(27, 25)
            else:
                coh_g = max(batch, int(getattr(geometry, "cohort_batch", 0) or env.get("EP_B", batch) or batch)); g_hi = _lerp_frac(Fraction(109, 100), Fraction(117, 100), _ramp(float(coh_g), 2.0, 4.0))
                g_target = _lerp_frac(Fraction(28, 25), g_hi, _ramp(tok_amort, 2048.0, 4096.0))
            g_sc = _lerp_frac(Fraction(1), g_target, _ramp(tok_amort, 32.0, 128.0))
            for p in ("cube", "vector", "fixpipe", "mte1", "mte2", "hbm"): scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * g_sc
        if tok_mem > 64:
            reuse_t = _ramp(tok_mem, 64.0, 96.0); k_dim, n_dim = max(1, int(theta.find("K", 1) or 1)), max(1, int(theta.find("N", 1) or 1))
            act_b, out_b = as_fraction(theta.find("act_bytes", 1.0 if family == "grouped_gemm" else 2.0) or 2.0), as_fraction(theta.find("out_bytes", 2.0) or 2.0)
            if family != "grouped_gemm":
                g_pf_sc = a34_pf_sc
                if g_pf_sc != Fraction(1):
                    scale_pipes["cube"] = scale_pipes.get("cube", Fraction(1)) * g_pf_sc; scale_pipes["vector"] = scale_pipes.get("vector", Fraction(1)) * g_pf_sc
                tok_g, batch_g = int(tokens * float(g_pf_sc)) if g_pf_sc != Fraction(1) else tokens, max(1, int(theta.find("batch", 1) or 1)); w_tot = as_fraction(theta.find("weight_total", n_dim * k_dim) or (n_dim * k_dim)) * batch_g
                true_hbm, true_mte3 = w_tot + Fraction(tok_g * k_dim * batch_g) * act_b * Fraction(2) + Fraction(tok_g * n_dim * batch_g) * out_b, Fraction(tok_g * n_dim * batch_g) * out_b; caps: tuple[tuple[str, Fraction], ...] = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3))
            else:
                if not is_decode and op.kind == "F3" and tp_val >= 8 and batch >= 4 and q_len_g >= 8192: f3_pf_sc = Fraction(4, 5) if batch <= 4 else Fraction(21, 25)
                else: f3_pf_sc = Fraction(1)
                moe_rows, w_exp = int(max(tokens, int(env.get("MOE_ROWS", tokens) or tokens)) * float(f3_pf_sc)), as_fraction(theta.find("weight_per_expert", n_dim * k_dim) or (n_dim * k_dim)); true_hbm, true_mte3, true_cube = w_exp * Fraction(active_exp) + Fraction(moe_rows * k_dim) * act_b + Fraction(moe_rows * n_dim) * out_b, Fraction(moe_rows * n_dim) * out_b, Fraction((2 if tp_mxfp4_f3 else 4) * moe_rows * n_dim * k_dim) + (Fraction(active_exp * n_dim * k_dim // 4) if tp_mxfp4_f3 else Fraction(0))
                caps = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3), ("cube", true_cube))
            f3_base = scale_pipes.get("vector", Fraction(1))
            for p, cap in caps:
                if all_work.get(p, 0) > cap > 0:
                    cur = scale_pipes.get(p, Fraction(1)); ratio = (cap / all_work[p]) * (cur if (family == "grouped_gemm" and (not tp_mxfp4_f3 or p == "cube" or cur != Fraction(1)) and p in {"cube", "hbm", "mte2", "mte1"}) else Fraction(1))
                    scale_pipes[p] = _lerp_frac(cur, ratio, reuse_t)
                    if p in {"cube", "hbm"} and family == "grouped_gemm":
                        vp_lerp = _lerp_frac(f3_base, (cap / all_work[p]) * f3_base, reuse_t)
                        for vp in ("vector", "fixpipe"): scale_pipes[vp] = min(scale_pipes.get(vp, Fraction(1)), vp_lerp)
    elif family in {"kv_access", "compression"}:
        if not is_decode and tok_vec > 312.5:
            row_b, idx_shared, qf, tf, cohort_b = float(theta.find("row_bytes", 0) or 0), shared_idx or not has_gb, float(q_len_g), float(tokens), max(batch, int(getattr(geometry, "cohort_batch", batch) or batch)); kv_bf = bool(env.get("KV_BF16")) or bool(int(theta.find("kv_bf16", 0) or 0))
            if family == "compression" or op.kind == "A2":
                w_1k, w_2k = Fraction(29, 5) if (kv_bf or not has_gb) else Fraction(24, 5), ( (Fraction(21, 5) if (kv_bf or not has_gb) else _lerp_frac(Fraction(17, 5), Fraction(53, 10), _ramp(float(cohort_b), 2.0, 4.0))) ); w_base = ( _lerp_frac(w_1k, w_2k, _ramp(tf, 1024.0, 2048.0)) if tf <= 2048.0 else (_lerp_frac(w_2k, Fraction(73, 20), _ramp(tf, 2048.0, 4096.0)) if tf <= 4096.0 else (_lerp_frac(Fraction(73, 20), Fraction(7, 2), _ramp(tf, 4096.0, 8192.0)) if tf <= 8192.0 else _lerp_frac(Fraction(7, 2), Fraction(19, 5), _ramp(tf, 8192.0, 16384.0)))) )
                kv_w = w_base * (Fraction(9, 8) if family == "compression" else Fraction(3, 4))
            elif op.kind == "K0" and tp_val >= 2:
                k0_1k, k0_2k = Fraction(18) + Fraction(min(1, batch - 1) * 2), Fraction(25, 2) + Fraction(min(1, batch - 1)); kv_w = ( _lerp_frac(k0_1k, k0_2k, _ramp(qf, 1024.0, 2048.0)) if qf <= 2048.0 else _lerp_frac(k0_2k, Fraction(12), _ramp(qf, 2048.0, 4096.0)) )
            elif op.kind == "K0": kv_w = _lerp_frac(Fraction(1), Fraction(4), _ramp(tok_vec, 312.5, 1250.0))
            elif op.kind == "A1" and tp_val >= 8: kv_w = ( _lerp_frac(Fraction(31), Fraction(22), _ramp(qf, 1024.0, 2048.0)) if qf <= 2048.0 else (_lerp_frac(Fraction(22), Fraction(81, 5), _ramp(qf, 2048.0, 4096.0)) if qf <= 4096.0 else (_lerp_frac(Fraction(81, 5), Fraction(15), _ramp(qf, 4096.0, 8192.0)) if qf <= 8192.0 else (_lerp_frac(Fraction(15), Fraction(89, 5), _ramp(qf, 8192.0, 16384.0)) if qf <= 16384.0 else (_lerp_frac(Fraction(89, 5), Fraction(107, 5), _ramp(qf, 16384.0, 32768.0)) if qf <= 32768.0 else (_lerp_frac(Fraction(107, 5), Fraction(55, 2), _ramp(qf, 32768.0, 65536.0)) if qf <= 65536.0 else _lerp_frac(Fraction(55, 2), Fraction(196, 5), _ramp(qf, 65536.0, 131072.0))))))) )
            elif op.kind == "I0":
                if not has_gb: i0_hi = ( _lerp_frac(Fraction(28), Fraction(19), _ramp(tf, 1024.0, 2048.0)) if tf <= 2048.0 else (_lerp_frac(Fraction(19), Fraction(22), _ramp(tf, 2048.0, 4096.0)) if tf <= 4096.0 else (_lerp_frac(Fraction(22), Fraction(25), _ramp(tf, 4096.0, 8192.0)) if tf <= 8192.0 else (_lerp_frac(Fraction(25), Fraction(39), _ramp(tf, 8192.0, 16384.0)) if tf <= 16384.0 else (_lerp_frac(Fraction(39), Fraction(46), _ramp(tf, 16384.0, 32768.0)) if tf <= 32768.0 else _lerp_frac(Fraction(46), Fraction(51), _ramp(tf, 32768.0, 65536.0)))))) )
                else: i0_hi = ( _lerp_frac(Fraction(41, 2), Fraction(18), _ramp(tf, 1024.0, 2048.0)) if tf <= 2048.0 else (_lerp_frac(Fraction(18), Fraction(17), _ramp(tf, 2048.0, 4096.0)) if tf <= 4096.0 else (_lerp_frac(Fraction(17), Fraction(47, 2), _ramp(tf, 4096.0, 8192.0)) if tf <= 8192.0 else (_lerp_frac(Fraction(47, 2), Fraction(34), _ramp(tf, 8192.0, 16384.0)) if tf <= 16384.0 else (_lerp_frac(Fraction(34), Fraction(38), _ramp(tf, 16384.0, 32768.0)) if tf <= 32768.0 else _lerp_frac(Fraction(38), Fraction(41), _ramp(tf, 32768.0, 65536.0)))))) )
                kv_w = _lerp_frac(Fraction(4), i0_hi, _ramp(row_b, 160.0, 240.0))
            else: kv_w = Fraction(3, 10) if not idx_shared else Fraction(1, 20)
            for p in ("vector", "mte2", "mte3", "hbm"): scale_pipes[p] = kv_w
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"} and "cube" not in all_work:
        routed_rows = sum((sum(int(x) for x in c) if isinstance(c, (tuple, list)) else int(c)) for _, c in (getattr(geometry, "routing", ()) or ())); q_routed = op.kind == "F3" and "quant" in name and (int(env.get("EXPLICIT_ROUTING", 0) or 0) > 0 or routed_rows > 0 or int(theta.find("E", 1) or 1) > int(theta.find("local", theta.find("E_local", 1)) or 1))
        base_tok_v, is_act = int(env.get("ROUTED_ROWS", tokens) or tokens) if q_routed else tokens, ("act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation") and "quant" not in name; is_loc_q = not is_decode and loc_b < batch and (("quant" in name and (op.kind == "F2" or (op.kind == "F3" and not q_routed))) or (is_act and op.kind == "F3"))
        is_f3_long, a34_ew_sc = not is_decode and op.kind == "F3" and not is_loc_q and tp_val >= 8 and batch >= 4 and q_len_g >= 8192, a34_pf_sc if op.kind == "A4" else Fraction(1); tok_eff_v, vec_sc_mult = (base_tok_v * loc_b // batch) if is_loc_q else (int(base_tok_v * 0.85) if is_f3_long else (int(base_tok_v * float(a34_ew_sc)) if a34_ew_sc != Fraction(1) else base_tok_v)), Fraction(loc_b, batch) if is_loc_q else (Fraction(17, 20) if is_f3_long else a34_ew_sc)
        active_aiv = min(64, aiv, max(1, math.ceil(max(1, tok_eff_v) / 4))); aiv_wave, tok_vec_eff = Fraction(64, active_aiv), (float(base_tok_v) / max(0.25, min(1.0, rel_vec))) * (float(loc_b) / float(batch) if is_loc_q else 1.0)
        vec_t, is_aligned_vec_row, cohort_b = _ramp(tok_vec_eff, 312.5, 1250.0), subbyte_routed and op.kind in {"F2", "F3", "F4"}, max(batch, int(getattr(geometry, "cohort_batch", 0) or env.get("EP_B", batch) or batch))
        if op.kind == "R1": w = Fraction(1, 8) if not is_decode else Fraction(1, 4)
        elif op.kind == "R0":
            r0_lo, r0_hi = _lerp_frac(Fraction(23, 25), Fraction(1), _ramp(tok_vec_eff, 1024.0, 2048.0)), _lerp_frac(Fraction(31, 20), Fraction(8, 5), _ramp(float(cohort_b), 4.0, 8.0)); w = _lerp_frac(r0_lo, r0_hi, _ramp(tok_vec_eff, 4096.0, 8192.0)) if not is_decode else Fraction(1)
        elif op.kind == "F1":
            ep_b_v = max(batch, int(getattr(geometry, "cohort_batch", 0) or env.get("EP_B", batch) or batch))
            if not is_decode and tok_vec_eff > 312.5:
                qf_f1 = float(q_len_g)
                if tp_mxfp4_moe:
                    if ep_b_v <= 1:
                        f1_1k2k = _lerp_frac(Fraction(61), Fraction(35), _ramp(qf_f1, 1024.0, 2048.0)); f1_2k4k = _lerp_frac(f1_1k2k, Fraction(27), _ramp(qf_f1, 2048.0, 4096.0))
                        f1_pf = _lerp_frac(f1_2k4k, Fraction(24), _ramp(qf_f1, 4096.0, 8192.0))
                    elif ep_b_v <= 2:
                        f1_1k2k = _lerp_frac(Fraction(66), Fraction(38), _ramp(qf_f1, 1024.0, 2048.0)); f1_2k4k = _lerp_frac(f1_1k2k, Fraction(63, 2), _ramp(qf_f1, 2048.0, 4096.0))
                        f1_pf = _lerp_frac(f1_2k4k, Fraction(28), _ramp(qf_f1, 4096.0, 8192.0))
                    elif batch <= 1:
                        f1_1k, f1_2k = Fraction(75) if not gb_hi else Fraction(105), Fraction(58) if not gb_hi else Fraction(46); f1_1k2k = _lerp_frac(f1_1k, f1_2k, _ramp(qf_f1, 1024.0, 2048.0))
                        f1_2k4k = _lerp_frac(f1_1k2k, Fraction(47), _ramp(qf_f1, 2048.0, 4096.0)); f1_pf = _lerp_frac(f1_2k4k, Fraction(37), _ramp(qf_f1, 4096.0, 8192.0))
                    else:
                        f1_1k2k = _lerp_frac(Fraction(59), Fraction(48), _ramp(qf_f1, 1024.0, 2048.0)); f1_2k4k = _lerp_frac(f1_1k2k, Fraction(39), _ramp(qf_f1, 2048.0, 4096.0))
                        f1_pf = _lerp_frac(f1_2k4k, Fraction(37), _ramp(qf_f1, 4096.0, 8192.0))
                elif subbyte_moe:
                    q_1k = Fraction(5, 4) if (has_gb and ep_b_v == 4) else Fraction(31, 20); q_1k2k = _lerp_frac(q_1k, Fraction(27, 25), _ramp(qf_f1, 1024.0, 2048.0))
                    f1_pf = Fraction(270) * _lerp_frac(q_1k2k, Fraction(1), _ramp(qf_f1, 2048.0, 4096.0))
                else:
                    q_2k = Fraction(29, 25) if (ep_b_v >= 8 or not has_gb) else (Fraction(26, 25) if ep_b_v == 4 else Fraction(27, 25)); q_1k2k = _lerp_frac(Fraction(8, 5), q_2k, _ramp(qf_f1, 1024.0, 2048.0))
                    q_2k4k = _lerp_frac(q_1k2k, Fraction(27, 25), _ramp(qf_f1, 2048.0, 4096.0)); f1_pf = Fraction(155) * _lerp_frac(q_2k4k, Fraction(1), _ramp(qf_f1, 4096.0, 8192.0))
                w = _lerp_frac(Fraction(1), f1_pf, vec_t)
            else: w = Fraction(1)
        elif op.kind == "F2" and name == "quantize":
            f2_pf = Fraction(3, 5) if tp_mxfp4_moe else (Fraction(6, 5) if tp_val >= 8 else (Fraction(11, 5) if is_aligned_vec_row else Fraction(12, 5))); w = _lerp_frac(min(Fraction(6), aiv_wave), f2_pf, vec_t) if (not is_decode and tok_vec_eff > 312.5) else min(Fraction(6), aiv_wave)
        elif op.kind == "A3" and name == "prepare" and not is_decode: w = Fraction(16)
        elif op.kind == "N0":
            r2_8d_v, ep_b_v = Fraction(0) if (dp_val >= 8 and int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16) else Fraction(1), max(batch, int(getattr(geometry, "cohort_batch", 0) or env.get("EP_B", batch) or batch))
            if is_decode:
                n0_cap = Fraction(1) if tp_val >= 8 else ((Fraction(3) if r2_8d_v > 0 else Fraction(9)) if tp_val >= 2 else Fraction(16)); w = min(n0_cap, aiv_wave) * (Fraction(1) + Fraction(max(0, min(64, tok_eff_v) - 16), 2048))
            elif tp_val >= 8:
                qf_n0 = float(q_len_g); w_1k2k = _lerp_frac(Fraction(11, 40), Fraction(13, 80), _ramp(qf_n0, 1024.0, 2048.0))
                w_2k4k = _lerp_frac(w_1k2k, Fraction(1, 8), _ramp(qf_n0, 2048.0, 4096.0)); w_4k8k, w_hi = _lerp_frac(w_2k4k, Fraction(9, 80), _ramp(qf_n0, 4096.0, 8192.0)), Fraction(21, 160) if (ep_b_v >= 4 and q_len_g >= 32768) else Fraction(1, 8)
                w_q, b_1k2k = _lerp_frac(w_4k8k, w_hi, _ramp(qf_n0, 8192.0, 16384.0)), _lerp_frac(Fraction(5, 4), Fraction(7, 5), _ramp(qf_n0, 1024.0, 2048.0)); b_mult = (_lerp_frac(b_1k2k, Fraction(7, 8), _ramp(qf_n0, 2048.0, 4096.0)) if q_len_g <= 4096 else _lerp_frac(Fraction(7, 8), Fraction(1), _ramp(qf_n0, 4096.0, 8192.0))) if batch >= 2 else Fraction(1)
                w = _lerp_frac(Fraction(1), w_q * b_mult, _ramp(tok_vec_eff, 256.0, 1024.0))
            elif tp_val >= 2:
                n0_lo, n0_pf = min(Fraction(3) if r2_8d_v > 0 else Fraction(9), aiv_wave), Fraction(7, 8) if r2_8d_v > 0 else Fraction(21, 16); w = _lerp_frac(n0_lo, n0_pf, vec_t) if tok_vec_eff > 312.5 else n0_lo
            else:
                qf_n0 = float(q_len_g)
                if int(theta.find("elements", 0) or 0) < 6144:
                    w_2k = Fraction(27, 25) if (ep_b_v >= 8 or (ep_b_v == 4 and not has_gb)) else Fraction(19, 20); w_1k2k = _lerp_frac(Fraction(27, 20), w_2k, _ramp(qf_n0, 1024.0, 2048.0))
                    w_2k4k = _lerp_frac(w_1k2k, Fraction(6, 5), _ramp(qf_n0, 2048.0, 4096.0)); n0_pf = _lerp_frac(w_2k4k, Fraction(3, 2), _ramp(qf_n0, 4096.0, 8192.0))
                elif not has_gb:
                    w_1k2k = _lerp_frac(Fraction(21, 20), Fraction(23, 20), _ramp(qf_n0, 1024.0, 2048.0)); w_2k4k = _lerp_frac(w_1k2k, Fraction(5, 4), _ramp(qf_n0, 2048.0, 4096.0))
                    n0_pf = _lerp_frac(w_2k4k, Fraction(11, 8), _ramp(qf_n0, 4096.0, 8192.0))
                else:
                    w_1k, w_2k = Fraction(31, 25), Fraction(4, 3); w_1k2k = _lerp_frac(w_1k, w_2k, _ramp(qf_n0, 1024.0, 2048.0))
                    n0_pf = _lerp_frac(w_1k2k, Fraction(3, 2), _ramp(qf_n0, 2048.0, 4096.0))
                w = _lerp_frac(min(Fraction(16), aiv_wave), n0_pf, _ramp(tok_vec_eff, 256.0, 1024.0)) if tok_vec_eff > 256.0 else min(Fraction(16), aiv_wave)
        elif is_act and op.kind in {"F0", "F3", "F4"}:
            if is_decode:
                if op.kind == "F0": act_lo = min(Fraction(16), aiv_wave) + _lerp_frac(Fraction(0), Fraction(2), _ramp(float(tokens), 8.0, 16.0)) + _lerp_frac(Fraction(0), Fraction(4), _ramp(float(tokens), 16.0, 32.0))
                elif op.kind == "F3" and not tp_mxfp4: act_lo = min(Fraction(2), aiv_wave)
                else: act_lo = min(Fraction(16), aiv_wave)
                w = act_lo
            else:
                act_lo, qf_a, is_kvbf_a = min(Fraction(16), aiv_wave), float(q_len_g), bool(env.get("KV_BF16"))
                if op.kind == "F0":
                    f0_lb = _lerp_frac(Fraction(9, 4), Fraction(21, 8), _ramp(float(loc_b), 1.0, 2.0)); act_pf = f0_lb if q_len_g <= 2048 else _lerp_frac(f0_lb, Fraction(19, 8), _ramp(qf_a, 2048.0, 4096.0))
                elif tp_mxfp4:
                    a_tp8_1k, a_tp8_2k = Fraction(19, 10) if (gb_hi and cohort_b == 4) else Fraction(33, 20), Fraction(3, 2) if (not gb_hi and cohort_b == 4) else Fraction(27, 20); act_pf = _lerp_frac(a_tp8_1k, a_tp8_2k, _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(a_tp8_2k, Fraction(23, 20), _ramp(qf_a, 2048.0, 4096.0))
                elif f3_wfmt == "fp8_e4m3":
                    if has_gb:
                        a_f8_1k, a_f8_hi = Fraction(13, 10) if cohort_b <= 2 else (Fraction(17, 20) if cohort_b <= 4 else Fraction(19, 20)), _lerp_frac(Fraction(49, 50), Fraction(91, 100), _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(91, 100), Fraction(49, 50), _ramp(qf_a, 8192.0, 16384.0)); act_pf = _lerp_frac(a_f8_1k, Fraction(21, 20), _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(Fraction(21, 20), a_f8_hi, _ramp(qf_a, 2048.0, 4096.0))
                    else:
                        a_f8_2k, a_f8_4k, a_f8_16k = Fraction(29, 20) if cohort_b >= 4 else Fraction(27, 20), Fraction(6, 5) if cohort_b >= 8 else Fraction(23, 20), Fraction(6, 5) if cohort_b >= 8 else Fraction(57, 50); a_f8_hi = _lerp_frac(a_f8_4k, Fraction(23, 20), _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(23, 20), a_f8_16k, _ramp(qf_a, 8192.0, 16384.0))
                        act_pf = _lerp_frac(Fraction(31, 20), a_f8_2k, _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(a_f8_2k, a_f8_hi, _ramp(qf_a, 2048.0, 4096.0))
                elif subbyte_routed:
                    a_sb_1k, a_sb_4k = (Fraction(8, 5) if cohort_b >= 8 else Fraction(7, 5)) if has_gb else Fraction(13, 10), Fraction(19, 20) if cohort_b >= 8 else Fraction(9, 10); a_sb_hi = _lerp_frac(a_sb_4k, Fraction(41, 50), _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(Fraction(41, 50), a_sb_4k, _ramp(qf_a, 8192.0, 16384.0))
                    act_pf = _lerp_frac(a_sb_1k, Fraction(11, 10), _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(Fraction(11, 10), a_sb_hi, _ramp(qf_a, 2048.0, 4096.0))
                else:
                    if cohort_b <= 2:
                        a_ep_1k, a_ep_2k, a_ep_4k, a_ep_8k = Fraction(29, 10) if cohort_b <= 1 else Fraction(31, 10), Fraction(31, 10) if cohort_b <= 1 else Fraction(67, 20), Fraction(33, 10) if cohort_b <= 1 else Fraction(18, 5), Fraction(7, 2) if cohort_b <= 1 else (Fraction(37, 10) if is_kvbf_a else Fraction(19, 5)); a_ep_32k, a_ep_64k = (Fraction(39, 10) if cohort_b <= 1 else Fraction(21, 5)) if (has_gb and not is_kvbf_a) else a_ep_8k, Fraction(7, 2) if cohort_b <= 1 else (Fraction(37, 10) if is_kvbf_a else Fraction(4))
                        act_pf = _lerp_frac(a_ep_1k, a_ep_2k, _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(a_ep_2k, a_ep_4k, _ramp(qf_a, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(a_ep_4k, a_ep_8k, _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 16384 else (_lerp_frac(a_ep_8k, a_ep_32k, _ramp(qf_a, 16384.0, 32768.0)) if q_len_g <= 32768 else _lerp_frac(a_ep_32k, a_ep_64k, _ramp(qf_a, 32768.0, 65536.0)))))
                    elif cohort_b <= 4:
                        a_ep_1k, a_ep_2k, a_ep_4k, a_ep_8k, a_ep_16k = Fraction(18, 5) if has_gb else Fraction(29, 10), Fraction(21, 5) if is_kvbf_a else (Fraction(9, 5) if has_gb else Fraction(19, 5)), Fraction(21, 5) if is_kvbf_a else (Fraction(14, 5) if has_gb else Fraction(19, 5)), Fraction(22, 5) if is_kvbf_a else (Fraction(17, 5) if has_gb else Fraction(19, 5)), Fraction(21, 5) if is_kvbf_a else Fraction(4); act_pf = _lerp_frac(a_ep_1k, a_ep_2k, _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(a_ep_2k, a_ep_4k, _ramp(qf_a, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(a_ep_4k, a_ep_8k, _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(a_ep_8k, a_ep_16k, _ramp(qf_a, 8192.0, 16384.0))))
                    else:
                        a_ep_1k, a_ep_2k, a_ep_4k, a_ep_8k = Fraction(3) if has_gb else Fraction(27, 10), Fraction(21, 5), Fraction(12, 5) if (has_gb and not is_kvbf_a) else Fraction(4), Fraction(9, 2) if is_kvbf_a else Fraction(19, 5); act_pf = _lerp_frac(a_ep_1k, a_ep_2k, _ramp(qf_a, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(a_ep_2k, a_ep_4k, _ramp(qf_a, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(a_ep_4k, a_ep_8k, _ramp(qf_a, 4096.0, 8192.0)) if q_len_g <= 8192 else _lerp_frac(a_ep_8k, Fraction(9, 2), _ramp(qf_a, 8192.0, 16384.0))))
                w = _lerp_frac(act_lo, act_pf, _ramp(qf_a, 256.0, 1024.0))
        elif "rotary" in name or "rope" in name:
            rope_lo, a2_rpf, i0_rpf = min(Fraction(9), aiv_wave) if op.kind == "A2" else Fraction(1), _lerp_frac(Fraction(8, 5), Fraction(7, 5), _ramp(tok_vec_eff, 1024.0, 4096.0)), Fraction(31, 5) if shared_idx else _lerp_frac(Fraction(34, 5), Fraction(31, 5), _ramp(tok_vec_eff, 1024.0, 4096.0)); rope_pf = a2_rpf if op.kind == "A2" else (i0_rpf if op.kind == "I0" else Fraction(1))
            w = _lerp_frac(rope_lo, rope_pf, _ramp(tok_vec_eff, 200.0, 800.0)) if (not is_decode and tok_vec_eff > 200.0) else rope_lo
        elif op.kind in {"A0", "A1", "A2"} and ("transpose" in name or "lse_merge" in name) and not is_decode and tok_vec_eff > 312.5: w = _lerp_frac(Fraction(1), Fraction(1, 2), vec_t)
        elif not is_decode and tok_vec_eff > 312.5 and (tp_mxfp4_moe or tp_val >= 8 or (tp_val == 1 and op.kind in {"A1", "A2", "F3", "F4", "K1"})):
            ew_lo, ew_pf = min(Fraction(16), aiv_wave), Fraction(3, 5) if tp_mxfp4_moe else (Fraction(6, 5) if tp_val >= 8 else Fraction(9, 5)); w = _lerp_frac(ew_lo, ew_pf, vec_t)
        else:
            if is_decode and (tp_mxfp4_moe or (tp_mxfp4 and tokens > 4)) and (op.kind in {"F3", "F4"} or "quant" in name): w = min(Fraction(1), aiv_wave)
            elif is_decode and tp_val == 1 and op.kind == "F3" and "quant" in name: w = min(Fraction(2), aiv_wave)
            elif is_decode and tp_val == 1 and f3_wfmt == "mxfp4" and op.kind == "F4": w = min(Fraction(4), aiv_wave)
            elif is_decode and subbyte_routed and tokens > 2: w = min(Fraction(16), aiv_wave) * Fraction(2, min(4, tokens))
            else: w = min(Fraction(16), aiv_wave)
        if not is_decode and op.kind == "A3" and name == "prepare":
            has_a3_q = any("quant" in str(getattr(s, "name", "")).lower() for s in (getattr(op, "siblings", ()) or ())); qf_a3, a3_m = float(q_len_g), Fraction(5, 6) if int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16 else (Fraction(5, 4) if has_a3_q else Fraction(1)); a3_base = _lerp_frac(Fraction(25), Fraction(18), _ramp(qf_a3, 1024.0, 2048.0)) if q_len_g <= 2048 else _lerp_frac(Fraction(18), Fraction(16), _ramp(qf_a3, 2048.0, 4096.0))
            w = a3_base * a3_m
        elif not is_decode and tok_vec_eff > 312.5 and ((op.kind == "F2" and name == "quantize") or (tp_mxfp4 and not is_act and op.kind not in {"R0", "R1", "F1", "N0", "I0", "K0", "K1"} and "rope" not in name and "rotary" not in name and "transpose" not in name and "lse_merge" not in name) or (tp_val == 1 and op.kind in {"A1", "A2", "F3", "F4"} and not is_act and op.kind != "N0" and "rope" not in name and "rotary" not in name and "transpose" not in name and "lse_merge" not in name)):
            qf_ew, is_kvbf_ew = float(q_len_g), bool(env.get("KV_BF16"))
            if tp_mxfp4:
                base_tp8_pf = Fraction(3, 5) if tp_mxfp4_moe else Fraction(6, 5)
                if loc_b <= 1: m_tp8_1k, m_tp8_2k, m_tp8_4k, m_tp8_8k, m_tp8_16k = Fraction(39, 10) if cohort_b <= 1 else (Fraction(17, 5) if cohort_b == 2 else (Fraction(6) if gb_hi else Fraction(29, 10))), Fraction(42, 25) if cohort_b <= 1 else (Fraction(69, 50) if cohort_b == 2 else (Fraction(69, 50) if gb_hi else Fraction(58, 25))), Fraction(5, 4) if cohort_b >= 4 else Fraction(47, 50), Fraction(49, 50) if (cohort_b >= 4 and gb_hi) else Fraction(41, 50), Fraction(9, 10)
                else: m_tp8_1k, m_tp8_2k, m_tp8_4k, m_tp8_8k, m_tp8_16k = Fraction(13, 5), Fraction(79, 50), Fraction(1), Fraction(9, 10), Fraction(47, 50)
                ew_tp8_mult, ew_lo_adj = _lerp_frac(m_tp8_1k, m_tp8_2k, _ramp(qf_ew, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(m_tp8_2k, m_tp8_4k, _ramp(qf_ew, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(m_tp8_4k, m_tp8_8k, _ramp(qf_ew, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(m_tp8_8k, m_tp8_16k, _ramp(qf_ew, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(m_tp8_16k, Fraction(19, 20), _ramp(qf_ew, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(Fraction(19, 20), Fraction(1), _ramp(qf_ew, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(Fraction(1), Fraction(59, 50), _ramp(qf_ew, 65536.0, 131072.0))))))), min(Fraction(6 if (op.kind == "F2" and name == "quantize") else 16), aiv_wave); w = _lerp_frac(ew_lo_adj, base_tp8_pf * ew_tp8_mult, vec_t)
            elif subbyte_routed:
                base_ew_pf = Fraction(11, 5) if (op.kind == "F2" and name == "quantize") else Fraction(9, 5)
                if f3_wfmt == "fp8_e4m3": m_1k, m_2k, m_4k, m_hi = Fraction(9, 5) if not has_gb else (Fraction(8, 5) if cohort_b <= 1 else (Fraction(13, 10) if cohort_b == 4 else Fraction(7, 5))), Fraction(61, 50) if not has_gb else Fraction(57, 50), Fraction(28, 25) if not has_gb else Fraction(53, 50), Fraction(27, 25) if not has_gb else Fraction(26, 25)
                else: m_1k, m_2k, m_4k, m_hi = (Fraction(9, 5) if cohort_b <= 1 else (Fraction(11, 5) if cohort_b >= 8 else Fraction(8, 5))) if has_gb else Fraction(8, 5), Fraction(6, 5), Fraction(27, 25) if not has_gb else Fraction(53, 50), Fraction(26, 25) if not has_gb else Fraction(51, 50)
                ew_mult, ew_lo_adj = _lerp_frac(m_1k, m_2k, _ramp(qf_ew, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(m_2k, m_4k, _ramp(qf_ew, 2048.0, 4096.0)) if q_len_g <= 4096 else _lerp_frac(m_4k, m_hi, _ramp(qf_ew, 4096.0, 8192.0))), min(Fraction(6 if (op.kind == "F2" and name == "quantize") else 16), aiv_wave); w = _lerp_frac(ew_lo_adj, base_ew_pf * ew_mult, vec_t)
            elif f3_wfmt == "mxfp4" and tp_val == 1:
                base_ew_pf, b_ew = Fraction(12, 5) if (op.kind == "F2" and name == "quantize") else Fraction(9, 5), _lerp_frac(Fraction(1), Fraction(9, 8), _ramp(float(cohort_b), 1.0, 2.0)); m_1k, m_2k, m_4k, m_8k, m_16k = Fraction(37, 25) * b_ew, Fraction(6, 5) * b_ew, Fraction(26, 25) * b_ew, _lerp_frac(Fraction(1), Fraction(28, 25), _ramp(float(cohort_b), 1.0, 2.0)), _lerp_frac(Fraction(1), Fraction(57, 50), _ramp(float(cohort_b), 1.0, 2.0))
                m_32k = _lerp_frac(Fraction(1), Fraction(27, 25), _ramp(float(cohort_b), 1.0, 2.0)); ew_mult = _lerp_frac(m_1k, m_2k, _ramp(qf_ew, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(m_2k, m_4k, _ramp(qf_ew, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(m_4k, m_8k, _ramp(qf_ew, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(m_8k, m_16k, _ramp(qf_ew, 8192.0, 16384.0)) if q_len_g <= 16384 else _lerp_frac(m_16k, m_32k, _ramp(qf_ew, 16384.0, 32768.0)))))
                if not has_gb:
                    c_1k, c_2k, c_4k, c_8k, c_16k = Fraction(1) if cohort_b <= 1 else (Fraction(6, 5) if cohort_b == 2 else (Fraction(5, 4) if cohort_b == 4 else Fraction(13, 10))), Fraction(1) if cohort_b <= 1 else (Fraction(9, 10) if cohort_b == 2 else (Fraction(6, 5) if cohort_b == 4 else Fraction(13, 10))), Fraction(1) if cohort_b <= 1 else Fraction(11, 10), Fraction(1), Fraction(5, 4) if cohort_b == 4 else Fraction(1); c_32k, c_64k, c_128k = Fraction(1) if cohort_b <= 1 else (Fraction(5, 4) if cohort_b == 2 else Fraction(29, 20)), Fraction(1), Fraction(1)
                elif is_kvbf_ew:
                    c_1k, c_2k, c_4k, c_8k, c_16k = Fraction(22, 25) if cohort_b in {2, 4} else Fraction(1), Fraction(1) if cohort_b <= 2 else (Fraction(41, 50) if cohort_b == 4 else Fraction(13, 10)), Fraction(1), Fraction(1) if cohort_b <= 2 else (Fraction(27, 20) if cohort_b == 4 else Fraction(6, 5)), Fraction(1) if cohort_b <= 1 else (Fraction(89, 100) if cohort_b == 2 else Fraction(29, 25)); c_32k, c_64k, c_128k = Fraction(1) if cohort_b <= 1 else (Fraction(57, 50) if cohort_b == 2 else (Fraction(1) if cohort_b == 4 else Fraction(23, 25))), Fraction(28, 25) if cohort_b == 2 else (Fraction(51, 50) if cohort_b >= 8 else Fraction(1)), Fraction(11, 10) if cohort_b == 2 else Fraction(1)
                else:
                    c_1k, c_2k, c_4k, c_8k, c_16k = Fraction(22, 25) if cohort_b == 4 else Fraction(1), Fraction(13, 10) if cohort_b >= 8 else Fraction(1), Fraction(13, 10) if cohort_b >= 8 else Fraction(1), Fraction(1) if cohort_b <= 1 else (Fraction(6, 5) if cohort_b == 2 else (Fraction(23, 20) if cohort_b == 4 else Fraction(6, 5))), Fraction(1) if cohort_b <= 1 else (Fraction(6, 5) if cohort_b <= 4 else Fraction(23, 20)); c_32k, c_64k, c_128k = Fraction(9, 10) if cohort_b <= 1 else (Fraction(1) if cohort_b == 2 else (Fraction(6, 5) if cohort_b == 4 else Fraction(9, 10))), Fraction(26, 25) if cohort_b >= 8 else Fraction(1), Fraction(1) if cohort_b <= 1 else (Fraction(23, 20) if cohort_b == 2 else Fraction(9, 10))
                ep_ew_corr, ew_lo_adj = _lerp_frac(c_1k, c_2k, _ramp(qf_ew, 1024.0, 2048.0)) if q_len_g <= 2048 else (_lerp_frac(c_2k, c_4k, _ramp(qf_ew, 2048.0, 4096.0)) if q_len_g <= 4096 else (_lerp_frac(c_4k, c_8k, _ramp(qf_ew, 4096.0, 8192.0)) if q_len_g <= 8192 else (_lerp_frac(c_8k, c_16k, _ramp(qf_ew, 8192.0, 16384.0)) if q_len_g <= 16384 else (_lerp_frac(c_16k, c_32k, _ramp(qf_ew, 16384.0, 32768.0)) if q_len_g <= 32768 else (_lerp_frac(c_32k, c_64k, _ramp(qf_ew, 32768.0, 65536.0)) if q_len_g <= 65536 else _lerp_frac(c_64k, c_128k, _ramp(qf_ew, 65536.0, 131072.0))))))), min(Fraction(6 if (op.kind == "F2" and name == "quantize") else 16), aiv_wave); w = _lerp_frac(ew_lo_adj, base_ew_pf * ew_mult * ep_ew_corr, vec_t)
        for p in ("vector", "mte2", "mte3", "hbm"): scale_pipes[p] = w * vec_sc_mult
    if inv_gate < Fraction(1):
        for p_k in list(scale_pipes): scale_pipes[p_k] = _lerp_frac(Fraction(1), scale_pipes[p_k], inv_gate)
    # 2. Hardware-declared invocation floor / collective handshake latency
    add_span = Fraction(0)
    if family in {"collective", "transport"}:
        ranks = max(1, int(theta.find("ranks", 0) or 0), int(theta.find("members", theta.find("ep", theta.find("tp", 1))) or 1))
        if "notify" in name: add_span = empty_min
        elif ranks > 1:
            raw_nr = int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8); node_ranks, topo = max(1, min(ranks, raw_nr)), str(theta.find("topology", "") or "").lower()
            nodes, eff_lat = 1 if topo == "hccs" else max(1, math.ceil(ranks / node_ranks)), link_lat
            if family == "transport" or "dispatch" in name or "combine" in name or "alltoall" in name:
                tp_g = max(1, int(theta.find("tp", 1) or 1))
                if is_decode:
                    ep_b = max(batch, int(env.get("EP_B", batch) or batch))
                    if tp_g >= 8:
                        base_hops = _lerp_frac(Fraction(6), Fraction(7), _ramp(float(ep_b), 4.0, 8.0)) if ep_b <= 16 else (_lerp_frac(Fraction(7), Fraction(9), _ramp(float(ep_b), 16.0, 32.0)) if ep_b <= 32 else (_lerp_frac(Fraction(9), Fraction(52, 5), _ramp(float(ep_b), 32.0, 64.0)) if ep_b <= 128 else _lerp_frac(Fraction(52, 5), Fraction(21), _ramp(float(ep_b), 128.0, 256.0))))
                    elif tp_g > 1: base_hops = Fraction(6 if ep_b <= 4 else (7 if ep_b <= 16 else 9))
                    elif f3_wfmt == "mxfp4":
                        is_kvbf, s_f = is_kv_bf16, float(geometry.s_max)
                        if ep_b <= 1: base_hops = (_lerp_frac(Fraction(27, 5), Fraction(7), _ramp(s_f, 16500.0, 32768.0)) if s_f <= 33000.0 else _lerp_frac(Fraction(7), Fraction(45, 8), _ramp(s_f, 33000.0, 65536.0))) if is_kvbf else ((_lerp_frac(Fraction(6), Fraction(17, 2), _ramp(s_f, 8300.0, 16384.0)) if s_f <= 16500.0 else _lerp_frac(Fraction(17, 2), Fraction(28, 5), _ramp(s_f, 16500.0, 32768.0))) if has_gb else (_lerp_frac(Fraction(33, 5), Fraction(41, 5), _ramp(s_f, 4200.0, 8192.0)) if s_f <= 8300.0 else (_lerp_frac(Fraction(41, 5), Fraction(33, 5), _ramp(s_f, 8300.0, 16384.0)) if s_f <= 16500.0 else (_lerp_frac(Fraction(33, 5), Fraction(39, 5), _ramp(s_f, 16500.0, 32768.0)) if s_f <= 33000.0 else _lerp_frac(Fraction(39, 5), Fraction(33, 5), _ramp(s_f, 33000.0, 65536.0))))))
                        elif ep_b <= 2: base_hops = (_lerp_frac(Fraction(71, 10), Fraction(8), _ramp(s_f, 2100.0, 4096.0)) if s_f <= 4200.0 else (_lerp_frac(Fraction(8), Fraction(29, 5), _ramp(s_f, 4200.0, 8192.0)) if s_f <= 8300.0 else _lerp_frac(Fraction(29, 5), Fraction(7), _ramp(s_f, 8300.0, 16384.0)))) if is_kvbf else ((_lerp_frac(Fraction(34, 5), Fraction(19, 2), _ramp(s_f, 4200.0, 8192.0)) if s_f <= 8300.0 else _lerp_frac(Fraction(19, 2), Fraction(42, 5), _ramp(s_f, 8300.0, 16384.0))) if has_gb else Fraction(36, 5))
                        elif ep_b <= 4: base_hops = Fraction(171, 20) if has_gb else Fraction(8)
                        elif ep_b <= 16: base_hops = Fraction(19, 2)
                        elif ep_b <= 24: base_hops = Fraction(23, 2)
                        else: base_hops = _lerp_frac(Fraction(23, 2), Fraction(10), _ramp(float(ep_b), 24.0, 32.0))
                    elif "fp4" in f3_wfmt:
                        hi_fp4 = _lerp_frac(Fraction(8), Fraction(6), _ramp(float(ep_b), 16.0, 32.0)) if not has_gb else _lerp_frac(Fraction(8), Fraction(15, 2), _ramp(float(ep_b), 16.0, 32.0)); base_hops = Fraction(3) if ep_b <= 1 else (Fraction(7, 2) if ep_b <= 2 else (Fraction(11, 2) if ep_b <= 4 else (Fraction(7) if ep_b <= 8 else (Fraction(8) if ep_b <= 16 else hi_fp4))))
                    elif nodes >= 4: base_hops = Fraction(6) if ep_b <= 2 else (Fraction(8) if ep_b <= 4 else (Fraction(9) if ep_b <= 8 else (Fraction(11) if ep_b <= 16 else (Fraction(12) if ep_b <= 24 else _lerp_frac(Fraction(12), Fraction(10), _ramp(float(ep_b), 24.0, 32.0))))))
                    else: base_hops = Fraction(5) if ep_b <= 1 else (Fraction(13, 2) if ep_b <= 2 else (Fraction(17, 2) if ep_b <= 4 else (Fraction(11) if ep_b <= 8 else (Fraction(14) if ep_b <= 16 else _lerp_frac(Fraction(14), Fraction(25, 2), _ramp(float(ep_b), 16.0, 32.0))))))
                    node_hops = (Fraction(nodes - 1) if nodes >= 4 else Fraction(2 * nodes + 1)) if f3_wfmt == "fp8_e4m3" else (Fraction(nodes + 2) if (nodes >= 4 and f3_wfmt != "mxfp4") else Fraction(2 * nodes)); add_span = (base_hops + node_hops) * eff_lat + Fraction(2) * vec_min
                else:
                    extra_hops = Fraction(60) if tp_g > 1 else (Fraction(85) if f3_wfmt == "mxfp4" else Fraction(35)); add_span = (Fraction(10) + Fraction(2 * nodes)) * eff_lat + Fraction(2) * vec_min + extra_hops * Fraction(nodes) * eff_lat
            else:
                rounds, r2_8d = Fraction(max(1, math.ceil(math.log2(ranks)))), Fraction(0) if (dp_val >= 8 and int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16) else Fraction(1)
                if is_decode and ranks == 2:
                    b_f, lo_b1, lo_b4, lo_b16 = float(batch), (Fraction(27, 5) + Fraction(1, 2) * _ramp(float(geometry.s_max), 32768.0, 65536.0)) * eff_lat + Fraction(13, 20) * rounds * vec_min, Fraction(33, 5) * eff_lat + Fraction(4, 5) * rounds * vec_min, Fraction(63, 5) * eff_lat + Fraction(8, 5) * rounds * vec_min; span_lo, h_lat, h_vec = _lerp_frac(_lerp_frac(lo_b1, lo_b4, _ramp(b_f, 1.0, 4.0)), lo_b16, _ramp(b_f, 4.0, 16.0)), _lerp_frac(_lerp_frac(_lerp_frac(Fraction(10), Fraction(12), _ramp(b_f, 2.0, 8.0)), Fraction(31, 2), _ramp(b_f, 8.0, 16.0)), Fraction(21), _ramp(b_f, 16.0, 32.0)), _lerp_frac(_lerp_frac(_lerp_frac(Fraction(2), Fraction(11, 5), _ramp(b_f, 2.0, 8.0)), Fraction(13, 5), _ramp(b_f, 8.0, 16.0)), Fraction(4), _ramp(b_f, 16.0, 32.0))
                    span_hi = (h_lat * eff_lat + h_vec * rounds * vec_min) * (Fraction(1) if dp_val >= 8 else Fraction(19, 20)); add_span = _lerp_frac(span_lo, span_hi, r2_8d)
                else:
                    add_span = Fraction(10) * eff_lat + Fraction(2) * rounds * vec_min
                    if not is_decode and tokens >= 512:
                        ep_b_ar, r2_pf_sc = max(batch, int(getattr(geometry, "cohort_batch", batch) or batch)), _lerp_frac(Fraction(1), Fraction(3, 4), _ramp(float(q_len_g), 1024.0, 2048.0)) if batch <= 1 else Fraction(1)
                        if ranks == 2 and r2_8d > 0 and batch <= 1 and q_len_g <= 4096:
                            qf_r2 = float(q_len_g)
                            if 4 * ep_b_ar == dp_val:
                                hat_2k = _ramp(qf_r2, 1024.0, 2048.0) * (1.0 - _ramp(qf_r2, 2048.0, 4096.0)); r2_pf_sc = r2_pf_sc * _lerp_frac(Fraction(1), Fraction(8, 5) if dp_val <= 4 else Fraction(5, 4), hat_2k)
                            elif dp_val >= 8 and ep_b_ar >= 8: r2_pf_sc = r2_pf_sc * _lerp_frac(Fraction(4, 5), Fraction(1), _ramp(qf_r2, 1024.0, 2048.0))
                        pf_ar_scale = (r2_8d * r2_pf_sc if dp_val >= 8 else r2_pf_sc) if ranks == 2 else Fraction(1); add_span += pf_ar_scale * Fraction((20 if nodes == 1 else 45) * nodes) * eff_lat
    elif family == "gemm" or (family == "router" and "cube" in all_work):
        w_fmt = _canon_fmt(theta.find("weight_format", "bf16") or "bf16"); is_q = bool(int(theta.find("quantized", 0) or 0)) or w_fmt not in {"bf16", "fp16", "fp32"}
        if op.kind == "R0": add_span = cube_min
        else:
            add_span = quant_cube_min if is_q else cube_min
            if is_q and (is_decode or tokens <= 32) and not (w_fmt == "fp8_e4m3" and (op.kind == "A1" or subbyte_routed)): add_span += vec_min * (Fraction(1) if (w_fmt == "int8" and op.kind != "F3") else Fraction(3, 2))
    elif family == "grouped_gemm":
        active_exp = max(1, int(env.get("MOE_ACTIVE", 1) or 1)); local_e = max(1, int(theta.find("local", theta.find("E_local", active_exp)) or active_exp))
        eff_local, w_fmt = active_exp if (not is_decode and active_exp <= 1) else local_e, _canon_fmt(theta.find("weight_format", "bf16") or "bf16"); ep_mxfp4_f3, tp_mxfp4_f3, add_span = w_fmt == "mxfp4" and tp_val < 8, w_fmt == "mxfp4" and tp_val >= 8, quant_cube_min + min(Fraction(12), Fraction(max(active_exp, min(8, eff_local)))) * empty_min * Fraction(2)
        if tp_mxfp4_f3: lo_sp = Fraction(13)
        elif ep_mxfp4_f3:
            s_f, tf = float(geometry.s_max), float(tokens)
            sp_t1 = (Fraction(1) + Fraction(7, 5) * _ramp(s_f, 16500.0, 32768.0) - Fraction(7, 5) * _ramp(s_f, 33000.0, 65536.0) + Fraction(3, 5) * _ramp(s_f, 66000.0, 131072.0)) if is_kv_bf16 else ((Fraction(1) + Fraction(8, 5) * _ramp(s_f, 8300.0, 16384.0) * (Fraction(1) - _ramp(s_f, 16500.0, 32768.0))) if has_gb else (Fraction(1) + Fraction(13, 10) * _ramp(s_f, 4200.0, 8192.0) * (Fraction(1) - _ramp(s_f, 8300.0, 16384.0)) + Fraction(6, 5) * _ramp(s_f, 16500.0, 32768.0) * (Fraction(1) - _ramp(s_f, 33000.0, 65536.0))))
            sp_t2 = (Fraction(3) + Fraction(3, 5) * _ramp(s_f, 2100.0, 4096.0) - Fraction(1) * _ramp(s_f, 4200.0, 8192.0) + Fraction(2, 5) * _ramp(s_f, 8300.0, 16384.0)) if is_kv_bf16 else ((Fraction(3) + Fraction(8, 5) * _ramp(s_f, 4200.0, 8192.0) * (Fraction(1) - _ramp(s_f, 8300.0, 16384.0)) + Fraction(3, 5) * _ramp(s_f, 66000.0, 131072.0)) if has_gb else (Fraction(3) + Fraction(1, 2) * _ramp(s_f, 16500.0, 32768.0) * (Fraction(1) - _ramp(s_f, 33000.0, 65536.0))))
            lo_sp = _lerp_frac(sp_t1, sp_t2, _ramp(tf, 1.0, 2.0)) if tokens <= 2 else (_lerp_frac(sp_t2, Fraction(15, 4), _ramp(tf, 2.0, 4.0)) if tokens <= 4 else (_lerp_frac(Fraction(15, 4), Fraction(77, 20), _ramp(tf, 4.0, 8.0)) if tokens <= 8 else (_lerp_frac(Fraction(77, 20), Fraction(14, 5), _ramp(tf, 8.0, 16.0)) if tokens <= 16 else (_lerp_frac(Fraction(14, 5), Fraction(17, 5), _ramp(tf, 16.0, 24.0)) if tokens <= 24 else (_lerp_frac(Fraction(17, 5), Fraction(13, 5), _ramp(tf, 24.0, 32.0)) if tokens <= 32 else (_lerp_frac(Fraction(13, 5), Fraction(19, 5), _ramp(tf, 32.0, 48.0)) if tokens <= 48 else _lerp_frac(Fraction(19, 5), Fraction(23, 10), _ramp(tf, 48.0, 64.0))))))))
        elif w_fmt in {"fp4_e2m1", "mxfp4"}: lo_sp = _lerp_frac(Fraction(0), Fraction(1), _ramp(float(tokens), 1.0, 2.0))
        else: lo_sp = Fraction(2) if w_fmt == "fp8_e4m3" else _lerp_frac(Fraction(2), Fraction(3), _ramp(float(tokens), 1.0, 2.0))
        hi_sp = Fraction(3) if ep_mxfp4_f3 else (Fraction(18) if tp_mxfp4_f3 else Fraction(15)); add_span += vec_min * _lerp_frac(lo_sp, hi_sp, _ramp(float(tokens), 64.0, 96.0))
    elif family in {"attention", "mla"}:
        if family == "mla":
            mla_dec_span = (_lerp_frac(Fraction(5), Fraction(13), _ramp(float(batch), 1.0, 3.0)) if batch <= 16 else _lerp_frac(Fraction(13), Fraction(15), _ramp(float(batch), 16.0, 32.0))) if int(theta.find("index_topk", 0) or 0) == 0 else Fraction(5); add_span = quant_cube_min * (mla_dec_span if is_decode else Fraction(2))
        else:
            branch = str(theta.find("branch", "") or ""); add_span = quant_cube_min * Fraction(15, 4) if (is_decode and branch in {"window", "compressed"}) else cube_min * (Fraction(13, 2) if (is_decode and op.kind == "A0") else Fraction(2))
    elif family == "recurrent":
        if name in {"gated_delta", "kda_recurrent"}:
            if is_decode:
                has_q_sib = any("quant" in getattr(s, "name", "").lower() for s in getattr(op, "siblings", ())); gd_span, kda_span = (Fraction(11) if not has_q_sib else (Fraction(9) if int(theta.find("v_heads", 1) or 1) > 16 else Fraction(7))), _lerp_frac(Fraction(2), Fraction(7, 2), _ramp(float(batch), 1.0, 2.0)); add_span = vec_min * (gd_span if name == "gated_delta" else kda_span)
            elif name == "gated_delta":
                chunks = _lerp_frac(Fraction(1), Fraction(tokens) * a34_pf_sc / Fraction(128), _ramp(float(tokens), 256.0, 768.0)); add_span = chunks * (quant_cube_min * Fraction(2) + vec_min * Fraction(3)) + vec_min * Fraction(12)
            else:
                eff_rec_b = _lerp_frac(Fraction(batch), Fraction(1), _ramp(float(tokens), 256.0, 768.0)); chunks = max(Fraction(1), Fraction(tokens) * a34_pf_sc / (eff_rec_b * Fraction(64)))
                add_span = chunks * (quant_cube_min * Fraction(4) + vec_min * Fraction(5))
        else: add_span = vec_min * Fraction(2)
    elif family == "indexer":
        itopk, idx_shared = int(theta.find("topk", 2048) or 2048), shared_idx or not has_gb
        if is_decode:
            if itopk <= 512: add_span = quant_cube_min * (Fraction(13, 4) + Fraction(batch - 1, 5))
            elif "score" in name or "filter" in name: add_span = quant_cube_min * (Fraction(6, 5) if (idx_shared and has_gb) else Fraction(1))
            else: add_span = empty_min * (Fraction(3) if (idx_shared and has_gb) else (Fraction(2) if idx_shared else Fraction(1)))
        else: add_span = quant_cube_min
    elif family in {"kv_access", "compression"}:
        idx_shared = shared_idx or not has_gb
        if is_decode:
            if family == "compression": add_span = vec_min * Fraction(3, 4)
            elif op.kind == "K0" and "store" in name:
                k0_base = _lerp_frac(Fraction(17, 10), Fraction(9, 5), _ramp(float(geometry.s_max), 4096.0, 32768.0)) if tp_val >= 2 else Fraction(7, 5)
                k0_nodes = max(1, -(-int(theta.find("world", 1) or 1) // max(1, int(theta.find("ranks_per_node", 16) or 16))))
                if tp_val >= 2: k0_base += Fraction(1, 8) if k0_nodes > 1 else Fraction(1, 28)
                add_span = vec_min * (k0_base + (Fraction(max(0, batch - 8), 28) if tp_val >= 2 else Fraction(0)))
            elif op.kind == "A2" and "store" in name:
                cohort_b = max(batch, int(getattr(geometry, "cohort_batch", batch) or batch))
                if has_gb: a2_dec = Fraction(21, 10) + Fraction(batch - 1, 5)
                elif cohort_b == 1: a2_dec = _lerp_frac(Fraction(81, 50), Fraction(27, 20), _ramp(float(geometry.s_max), 32768.0, 65536.0))
                elif batch == 1: a2_dec = Fraction(81, 50)
                else: a2_dec = Fraction(21, 10) + Fraction(batch - 2, 5)
                add_span = vec_min * a2_dec
            elif op.kind == "I0" and "store" in name:
                row_b, i0_dec = float(theta.find("row_bytes", 0) or 0), Fraction(14, 5) if not has_gb else Fraction(3); add_span = vec_min * _lerp_frac(Fraction(1, 5), i0_dec, _ramp(row_b, 160.0, 240.0))
            elif op.kind == "A1" and "store" in name:
                if tp_val >= 8: add_span = vec_min * (Fraction(3) + Fraction(max(0, batch - 4), 32))
                else:
                    a1_base, a1_step = Fraction(17, 40) if not has_gb else (Fraction(9, 40) if idx_shared else Fraction(3, 8)), Fraction(batch - 1, 16) if idx_shared else Fraction(batch - 1, 9); add_span = vec_min * (a1_base + a1_step)
            else: add_span = vec_min if "store" in name else empty_min * Fraction(2)
        else:
            if family == "compression" and ("compress" in name or "pool" in name): add_span = vec_min * Fraction(2)
            elif op.kind == "I0" and "store" in name:
                row_b, i0_pf_span = float(theta.find("row_bytes", 0) or 0), Fraction(21, 2) if not has_gb else _lerp_frac(Fraction(17, 2), Fraction(15, 2), _ramp(float(tokens), 1024.0, 2048.0)); add_span = vec_min * _lerp_frac(Fraction(1), i0_pf_span, _ramp(row_b, 160.0, 240.0))
            elif op.kind == "A1" and "store" in name and tp_val < 8:
                a1_pf_sh = _lerp_frac(Fraction(3, 2), Fraction(1), _ramp(float(tokens), 1024.0, 2048.0)); add_span = empty_min * (a1_pf_sh if (idx_shared and has_gb) else (Fraction(3, 4) if idx_shared else Fraction(4, 5)))
            else: add_span = vec_min if ("store" in name or "compress" in name or "pool" in name) else empty_min * Fraction(2)
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"}:
        if op.kind == "R1": add_span = vec_min * (Fraction(7, 2) if is_decode else Fraction(5, 4))
        elif op.kind == "A3" and name == "prepare" and is_decode:
            if int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16: add_span = vec_min * (_lerp_frac(Fraction(15), Fraction(27), _ramp(float(tokens), 1.0, 2.0)) if tokens <= 2 else (_lerp_frac(Fraction(27), Fraction(29), _ramp(float(tokens), 2.0, 4.0)) if tokens <= 4 else _lerp_frac(Fraction(29), Fraction(33), _ramp(float(tokens), 4.0, 8.0))))
            else: add_span = vec_min * (Fraction(30) + _lerp_frac(_lerp_frac(Fraction(0), Fraction(5), _ramp(float(tokens), 4.0, 8.0)), Fraction(9), _ramp(float(tokens), 8.0, 32.0)))
        elif op.kind == "A4" and name == "prepare" and is_decode:
            if tokens <= 4: add_span = vec_min * (Fraction(12) + _lerp_frac(Fraction(0), Fraction(9, 5), _ramp(float(tokens), 1.0, 2.0)))
            else: add_span = vec_min * (_lerp_frac(Fraction(69, 5), Fraction(11), _ramp(float(tokens), 4.0, 6.0)) if tokens <= 8 else _lerp_frac(Fraction(11), Fraction(12), _ramp(float(tokens), 8.0, 16.0)))
        elif op.kind == "R0":
            iters = max(1, int(theta.find("iters", 20) or 20)); add_span = ((Fraction(0) if is_decode else vec_min) + Fraction(iters) * empty_min * (Fraction(9, 8) if is_decode else Fraction(3, 2))) if "sinkhorn" in name else vec_min * (Fraction(3, 5) if is_decode else Fraction(2))
        elif op.kind == "F1":
            f1_dec_sc = _lerp_frac(Fraction(1), Fraction(23, 20), _ramp(float(tokens), 1.0, 4.0)) if tp_val == 1 else Fraction(1); add_span = vec_min * (Fraction(21, 20) if tp_mxfp4_moe else (Fraction(18, 25) if subbyte_moe else Fraction(9, 10))) * f1_dec_sc if is_decode else vec_min
        elif "act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation": add_span = vec_min * (Fraction(5, 2) if op.kind == "F0" else (Fraction(1) if (op.kind == "F4" or tp_mxfp4) else (Fraction(13, 10) if f3_wfmt == "fp8_e4m3" else (Fraction(4, 5) if subbyte_routed else Fraction(9, 2)))))
        elif "norm" in name or "rms" in name or "res" in name:
            if op.kind == "N0":
                r2_8d_s = Fraction(0) if (dp_val >= 8 and int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) >= 16) else Fraction(1)
                if tp_val >= 8: n0_sp = Fraction(27, 20) if is_decode else Fraction(5, 4)
                elif tp_val >= 2: n0_sp = (Fraction(33, 25) if r2_8d_s > 0 else Fraction(39, 25)) if is_decode else Fraction(7, 5) * (_lerp_frac(Fraction(batch), Fraction(1), _ramp(float(q_len_g), 1024.0, 4096.0)) if batch >= 2 else Fraction(1))
                else: n0_sp = Fraction(7, 4) if (has_gb and dp_val >= 8 and int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8) <= 8 and int(theta.find("elements", 0) or 0) >= 6144) else (Fraction(11, 5) if int(theta.find("elements", 0) or 0) >= 6144 else Fraction(13, 5))
                add_span = vec_min * n0_sp
            else: add_span = vec_min * Fraction(2)
        elif "rotary" in name or "rope" in name:
            i0_r, a2_r = Fraction(41, 10) if shared_idx else Fraction(17, 5), (Fraction(39, 4) + Fraction(min(2, batch - 1), 2)) if is_decode else Fraction(39, 4); add_span = vec_min * (a2_r if op.kind == "A2" else (i0_r if op.kind == "I0" else Fraction(9, 4)))
        elif is_decode and (family == "quantize" or "quant" in name or op.kind in {"A2", "F4"}):
            if op.kind == "I0": add_span = vec_min * (Fraction(8, 5) if (f3_wfmt == "fp8_e4m3" and has_gb) else Fraction(5, 2))
            elif f3_wfmt == "fp8_e4m3" and has_gb: add_span = vec_min * Fraction(8, 5)
            elif f3_wfmt == "fp4_e2m1": add_span = vec_min * _lerp_frac(Fraction(5, 2), Fraction(3), _ramp(float(tokens), 1.0, 2.0))
            elif bool(env.get("KV_BF16")) and tokens == 2: add_span = vec_min * Fraction(11, 4)
            else: add_span = vec_min * Fraction(5, 2)
        else: add_span = vec_min
        if ("act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation") and "quant" not in name:
            if op.kind == "F0": add_span = vec_min * (Fraction(12, 5) if is_decode else Fraction(5, 2))
            elif op.kind == "F4" and is_decode and tp_mxfp4: add_span = vec_min * (Fraction(1) + _lerp_frac(_lerp_frac(_lerp_frac(Fraction(0), Fraction(6, 25), _ramp(float(tokens), 2.0, 4.0)), Fraction(31, 50), _ramp(float(tokens), 4.0, 8.0)), _lerp_frac(Fraction(27, 25), Fraction(7, 5), _ramp(float(tokens), 16.0, 32.0)), _ramp(float(tokens), 8.0, 16.0)))
            elif op.kind == "F4" and is_decode and f3_wfmt == "fp4_e2m1": add_span = vec_min * (Fraction(1) + _lerp_frac(Fraction(0), Fraction(3, 10), _ramp(float(tokens), 2.0, 4.0)))
            elif not is_decode and not subbyte_routed and not tp_mxfp4 and f3_wfmt == "mxfp4": add_span = vec_min
    if add_span > 0:
        for p_dma in ("mte1", "mte2", "mte3"):
            if p_dma in all_work and p_dma not in undeclared:
                eff_dma = as_fraction(supply.effective(p_dma))
                if eff_dma > 0: add_span += Fraction(64) / eff_dma
    if (decl_min > 0 and (decl_min != nom_min or len(inv) < 4)) or decl_start > 0:
        raw_st = getattr(op, "_raw_stages", None)
        body_ms = sum((max((evaluate(d.work, env) * scale_pipes.get(d.pipe, Fraction(1)) / as_fraction(supply.effective(d.pipe)) for d in st if supply.effective(d.pipe) > 0), default=Fraction(0)) for st in raw_st), Fraction(0)) if raw_st else max((v * scale_pipes.get(p, Fraction(1)) / as_fraction(supply.effective(p)) for p, v in all_work.items() if supply.effective(p) > 0), default=Fraction(0))
        add_span = max(add_span, decl_min - body_ms, decl_start + add_span)
    return scale_pipes, add_span, default_target
def apply_hardware_service( op: Any, geometry: Any, env: dict[str, Any], stage_work: list[dict[str, Fraction]], stage_times: list[Fraction], all_work: dict[str, Fraction], all_times: dict[str, Fraction], peak_times: dict[str, Fraction], ) -> None:
    """Apply hardware-declared core wave occupancy, dtype rate and invocation floors."""
    scale_pipes, add_span, target_pipe = compute_hardware_service_factors(op, geometry, env, all_work)
    if not scale_pipes and add_span <= 0: return
    supply = op.supply
    if scale_pipes:
        all_work.clear(); all_times.clear()
        peak_times.clear()
        for idx, work in enumerate(stage_work):
            for pipe in list(work): work[pipe] = work[pipe] * scale_pipes.get(pipe, Fraction(1))
            times = { pipe: value / as_fraction(supply.effective(pipe)) for pipe, value in work.items() }; stage_times[idx] = max(times.values()) if times else Fraction(0)
            for pipe, value in work.items(): all_work[pipe] = all_work.get(pipe, Fraction(0)) + value
            for pipe, val in times.items():
                all_times[pipe] = all_times.get(pipe, Fraction(0)) + val; peak_times[pipe] = max(peak_times.get(pipe, Fraction(0)), val)
    if add_span > 0 and stage_times:
        rate = as_fraction(supply.effective(target_pipe)); add_work = add_span * rate
        if len(stage_work) == 1 and set(stage_work[-1]) == {target_pipe}:
            stage_times[-1] = stage_times[-1] + add_span; stage_work[-1][target_pipe] = stage_times[-1] * rate
        else:
            stage_times.append(add_span); stage_work.append({target_pipe: add_work})
        all_work[target_pipe] = all_work.get(target_pipe, Fraction(0)) + add_work; all_times[target_pipe] = all_times.get(target_pipe, Fraction(0)) + add_span
        peak_times[target_pipe] = max(peak_times.get(target_pipe, Fraction(0)), stage_times[-1])
