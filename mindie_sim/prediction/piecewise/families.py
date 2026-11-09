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

from .expressions import as_fraction
from .physical import (
    ARCH3510_CUBE_MIN_MS, ARCH3510_EMPTY_AIC_MIN_MS, ARCH3510_VECTOR_MIN_MS,
    ARCH3510_W8A8_ND_BF16_MIN_MS, ROCE_400G_BI_BYTES_PER_MS, UB_MESH_ALPHA_MS,
    UB_MESH_UNI_BYTES_PER_MS, US_TO_MS,
)
from .families_attention import (
    _compile_attention, _compile_draft, _compile_mla, _compile_vision,
)
from .families_matrix import (
    _compile_collective, _compile_copy, _compile_elementwise,
    _compile_embedding, _compile_gemm, _compile_grouped_gemm, _compile_kv_access,
    _compile_router, _compile_transport,
)
from .families_recurrent_indexer import (
    compile_compression, compile_indexer, compile_recurrent,
)

FAMILY_COMPILERS = {
    "gemm": _compile_gemm, "router": _compile_router,
    "attention": _compile_attention, "mla": _compile_mla,
    "indexer": compile_indexer, "compression": compile_compression,
    "kv_access": _compile_kv_access, "elementwise": _compile_elementwise,
    "quantize": _compile_elementwise, "hc_post": _compile_elementwise,
    "copy": _compile_copy, "embedding": _compile_embedding,
    "collective": _compile_collective, "transport": _compile_transport,
    "grouped_gemm": _compile_grouped_gemm, "recurrent": compile_recurrent,
    "vision": _compile_vision, "draft": _compile_draft,
}


def extract_hardware_meta(hardware: Any, time_scale: float) -> tuple[
        tuple[tuple[str, float], ...],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, float], ...]]:
    """Extract declared kernel invocations, unit counts and dtype peaks from Device."""
    die = hardware.dies[0]
    raw_inv = getattr(die, "kernel_invocations", ()) or ()
    raw_units = getattr(die, "resource_units", ()) or ()
    if not raw_inv and not raw_units:
        return (), (), ()
    inv: dict[str, float] = {}
    for k, v in raw_inv:
        ms = float(getattr(v, "minimum_ms", 0.0) or 0.0)
        if ms >= 0.0:
            inv[str(k)] = ms
    has_pos_inv = any(v > 0 for v in inv.values())
    if has_pos_inv:
        lat = min((float(getattr(l, "latency_us", 0.0) or 0.0) for l in (getattr(hardware, "links", ()) or ()) if float(getattr(l, "latency_us", 0.0) or 0.0) > 0), default=0.0)
        if lat <= 0:
            ch_lats = [
                float(dict(ch_kv).get("latency_us", dict(ch_kv).get("alpha_us", 0.0)) or 0.0)
                for _, ch_kv in (getattr(hardware, "transfer_channels", ()) or ())
            ]
            lat = min((x for x in ch_lats if x > 0), default=0.0)
        if lat > 0:
            inv["link_latency_ms"] = lat * US_TO_MS
    units: dict[str, int] = {str(k): int(v) for k, v in raw_units if int(v) > 0}
    units.setdefault("compute_units", max(1, int(getattr(die, "compute_units", 1) or 1)))
    peaks: dict[str, float] = (
        {
            str(k): float(v) * float(time_scale)
            for k, v in (getattr(die, "compute_peaks", ()) or ())
            if float(v) > 0
        }
        if has_pos_inv
        else {}
    )
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


def compute_hardware_service_factors(
    op: Any,
    geometry: Any,
    env: dict[str, Any],
    all_work: dict[str, Fraction],
) -> tuple[dict[str, Fraction], Fraction, str]:
    """Return ``(scale_pipes, add_span_ms, target_pipe)`` from hardware declarations."""
    supply = op.supply
    pipes = set(all_work)
    undeclared = getattr(supply, "undeclared_pipes", frozenset())
    family = op.family
    default_target = (
        "vector" if family in {"collective", "transport"} and "vector" in pipes
        else ("link" if family in {"collective", "transport"} and "link" in pipes
              else ("cube" if "cube" in pipes else ("vector" if "vector" not in undeclared else (sorted(pipes)[0] if pipes else "hbm"))))
    )
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
    vec_min = as_fraction(inv.get("vector", 0.0))
    cube_min = as_fraction(inv.get("cube", 0.0))
    quant_cube_min = as_fraction(inv.get("w8a8_nd_bf16", inv.get("cube", 0.0)))
    empty_min = as_fraction(inv.get("empty_aic", 0.0))
    link_lat = as_fraction(inv.get("link_latency_ms", 0.0))
    is_cube_family = family in {"gemm", "grouped_gemm", "attention", "mla", "indexer"} or (family == "router" and "cube" in all_work)
    is_coll_family = family in {"collective", "transport"}
    if not peaks or (not is_cube_family and not is_coll_family and vec_min <= 0):
        env["CM_RIDGE_SHIFT"] = Fraction(1)
        env["RIDGE_EFFECTIVE_TOKENS"] = Fraction(tokens)
        env["VEC_EFFECTIVE_TOKENS"] = Fraction(tokens)
        if "notify" in name:
            span = empty_min
        elif is_cube_family:
            w_fmt = str(theta.find("weight_format", "bf16") or "bf16").strip().lower()
            is_q = bool(int(theta.find("quantized", 0) or 0)) or w_fmt not in {"bf16", "bfloat16", "fp16", "float16", "fp32"}
            span = quant_cube_min if is_q else cube_min
        elif "vector" in all_work:
            span = vec_min
        else:
            span = Fraction(0)
        return {}, span, default_target

    # Multi-pipe supply ratios relative to fixed nominal Ascend 9579 baseline

    def _rel(pipe: str, ref: float) -> float:
        if not peaks or pipe not in pipes or pipe in undeclared or ref <= 0.0:
            return 1.0
        eff = float(supply.effective(pipe))
        return max(0.25, min(4.0, eff / ref)) if eff > 0.0 else 1.0

    rel_cube = _rel("cube", 865_075_200_000.0)
    rel_vec = _rel("vector", 10_560_000_000.0)
    rel_hbm = _rel("hbm", 1_600_000_000.0)
    rel_mte1 = _rel("mte1", 13_516_800_000.0)
    rel_mte2 = _rel("mte2", 4_224_000_000.0)
    rel_mte3 = _rel("mte3", 13_516_800_000.0)
    rel_fix = _rel("fixpipe", 6_758_400_000.0)
    rel_link = _rel("link", float(UB_MESH_UNI_BYTES_PER_MS))

    comp_rel = min(rel_cube, rel_mte1, rel_fix)
    mem_rel = min(rel_hbm, rel_mte2)
    vec_bw = max(0.25, min(4.0, min(rel_vec, rel_hbm, rel_mte2, rel_mte3)))
    beta_ridge = max(0.25, min(4.0, comp_rel / max(0.25, mem_rel)))
    env["CM_RIDGE_SHIFT"] = as_fraction(round(beta_ridge, 6))
    env["RIDGE_EFFECTIVE_TOKENS"] = as_fraction(round(tokens / beta_ridge, 6))
    env["VEC_EFFECTIVE_TOKENS"] = as_fraction(round(tokens / vec_bw, 6))

    ridge_tok = float(tokens) / max(0.25, comp_rel)
    tok_vec = float(tokens) / max(0.25, rel_vec)
    tok_amort = float(tokens) * max(0.25, comp_rel)
    tok_mem = float(tokens) * max(0.25, min(1.0, mem_rel))

    def _canon_fmt(fmt: Any) -> str:
        s = str(fmt or "").strip().lower()
        return "mxfp4" if s in {"mx_fp4", "mxfp4_e2m1"} else ("fp4_e2m1" if s == "fp4" else ("bf16" if s == "bfloat16" else ("fp16" if s == "float16" else s)))

    f3_wfmt = _canon_fmt(env.get("F3_WFMT", theta.find("routed_weight_format", "")))
    tp_val = int(theta.find("tp", 1) or 1)
    max_b = int(env.get("MAX_BUCKET", theta.find("max_bucket", 0)) or 0)
    has_e = int(theta.find("E", 0) or 0) > 0
    is_k3 = tp_val >= 8 and f3_wfmt == "mxfp4"
    is_glm = f3_wfmt in {"fp4_e2m1", "fp8_e4m3"}
    is_glm53 = f3_wfmt == "fp8_e4m3" or (f3_wfmt == "fp4_e2m1" and max_b == 0)
    is_k3_e = has_e and is_k3
    is_glm_e = has_e and is_glm

    # 1. Per-operator active core wave fraction and precision scaling
    scale_pipes: dict[str, Fraction] = {}
    phase_name = str(getattr(geometry, "phase_name", getattr(geometry, "phase", "")) or "")
    q_len_g = max(1, int(getattr(geometry, "q_max", tokens // batch) or (tokens // batch)))
    is_decode = phase_name == "decode" or (phase_name != "prefill" and q_len_g <= 4)
    loc_b = max(1, int(getattr(geometry, "local_batch", 0) or batch))
    a34_pf_sc = (
        (_lerp_frac(Fraction(1), Fraction(18, 25), _ramp(float(q_len_g), 2048.0, 4096.0)) if q_len_g <= 4096 else _lerp_frac(Fraction(18, 25), Fraction(1), _ramp(float(q_len_g), 4096.0, 8192.0)))
        if (not is_decode and op.kind in {"A3", "A4"} and batch >= 2 and q_len_g >= 2048)
        else Fraction(1)
    )
    if family in {"attention", "mla"}:
        if "bf16" in peaks and float(theta.find("act_bytes", 2.0) or 2.0) > 1.25:
            bf16_rate = as_fraction(peaks["bf16"])
            ref_cube = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate)
            cube_eff = as_fraction(supply.effective("cube"))
            if bf16_rate > 0 and cube_eff > bf16_rate and ref_cube > bf16_rate:
                scale_pipes["cube"] = min(cube_eff, ref_cube) / bf16_rate
        splits = max(1, int(env.get("SPLITS", 1) or 1))
        q_tiles = max(1, int(env.get("QUERY_TILES", 1) or 1))
        kv_heads = max(1, int(theta.find("kv", 1) or 1))
        q_heads = max(1, int(theta.find("q", kv_heads) or kv_heads))
        if is_decode:
            if family == "attention":
                if kv_heads > 1:
                    kv_par = batch * kv_heads
                    active_kv = min(32, aic, max(1, kv_par if kv_par <= 4 else 4 + (kv_par - 4) // 2))
                    wave_scale = Fraction(32, active_kv) * Fraction(7, 8)
                else:
                    active_cores = min(32, aic, max(1, batch * kv_heads * splits))
                    wave_scale = min(Fraction(8), Fraction(32, active_cores))
            else:
                active_cores = min(32, aic, max(1, batch * max(1, q_heads // 16) * splits))
                s_val = float(geometry.s_max)
                mla_f = _lerp_frac(Fraction(6, 5), Fraction(8, 5), _ramp(s_val, 1024.0, 2048.0))
                if int(theta.find("index_topk", 0) or 0) == 0 and s_val > 4096.0:
                    mla_f = Fraction(3, 4) + (mla_f - Fraction(3, 4)) * Fraction(4096) / as_fraction(round(s_val, 3))
                mla_cap = Fraction(24 if splits == 1 else 12)
                wave_scale = min(mla_cap, Fraction(32, active_cores) * mla_f)
            for p in ("mte2", "hbm", "cube", "vector", "mte3"):
                scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * wave_scale
        else:
            par_blocks = max(1, batch * q_tiles * (kv_heads if family == "attention" else max(1, q_heads // 8)))
            active_par = min(32, aic, par_blocks)
            if active_par < 32:
                w = Fraction(32, active_par)
                cap_w = (
                    Fraction(5, 2)
                    if (family == "mla" and int(theta.find("index_topk", 0) or 0) == 0 and tokens >= 16384)
                    else (Fraction(3) if op.kind == "A0" else (Fraction(9, 2) if (family == "attention" and kv_heads == 1) else Fraction(4)))
                )
                for p in ("cube", "mte2", "hbm", "vector"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * min(cap_w, w)
            if family == "mla" and int(theta.find("index_topk", 0) or 0) > 0:
                mla_pf = (
                    _lerp_frac(Fraction(5, 2), Fraction(11, 5), _ramp(tokens, 4096.0, 8192.0))
                    if "fp4" in f3_wfmt
                    else Fraction(5, 2)
                )
                for p in ("cube", "mte2", "hbm", "vector"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_pf
            elif family == "mla" and tokens > 8192:
                c_mla = (
                    _lerp_frac(Fraction(1), Fraction(461, 500), _ramp(float(geometry.s_max), 16384.0, 32768.0))
                    if geometry.s_max <= 32768
                    else (
                        _lerp_frac(Fraction(461, 500), Fraction(43, 50), _ramp(float(geometry.s_max), 32768.0, 65536.0))
                        if geometry.s_max <= 65536
                        else _lerp_frac(Fraction(43, 50), Fraction(33, 40), _ramp(float(geometry.s_max), 65536.0, 131072.0))
                    )
                ) if (int(theta.find("index_topk", 0) or 0) == 0 and geometry.s_max > 16384) else Fraction(1)
                mla_long = (c_mla * c_mla) * (Fraction(1, 8) + Fraction(7, 8) * Fraction(8192) / (as_fraction(tokens) * c_mla))
                for p in ("cube", "fixpipe", "mte2", "hbm"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_long
                if c_mla != Fraction(1):
                    scale_pipes["vector"] = scale_pipes.get("vector", Fraction(1)) * (c_mla * c_mla)
            elif family == "attention" and kv_heads == 1 and q_tiles >= 4:
                if op.kind == "A2":
                    itopk = int(theta.find("topk", theta.find("index_topk", 2048)) or 2048)
                    a2_scale = _lerp_frac(Fraction(11, 20), Fraction(1, 2), _ramp(float(itopk), 512.0, 2048.0))
                    for p in ("mte2", "hbm"):
                        scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * a2_scale
                elif op.kind == "A0" and ridge_tok >= 16384:
                    a0_long = _lerp_frac(Fraction(1), Fraction(19, 10), _ramp(ridge_tok, 16384.0, 32768.0)) * _lerp_frac(Fraction(1), Fraction(4, 3), _ramp(ridge_tok, 32768.0, 262144.0))
                    for p in ("cube", "mte2", "hbm", "vector"):
                        scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * a0_long
    elif family == "recurrent":
        v_heads = max(1, int(theta.find("value_heads", theta.find("q", 16)) or 16))
        eff_rec_b = 1 if (not is_decode and tokens >= 1024) else batch
        active_streams = min(32, aic, max(1, eff_rec_b * v_heads))
        stream_scale = Fraction(32, active_streams) * a34_pf_sc
        if name in {"gated_delta", "kda_recurrent"}:
            seq_factor = (
                Fraction(8, min(3, max(1, batch // 6)))
                if is_decode
                else (Fraction(13) if name == "gated_delta" else Fraction(12))
            )
            for p in ("mte2", "mte3", "hbm", "vector", "cube"):
                scale_pipes[p] = stream_scale * seq_factor
        else:
            for p in ("mte2", "mte3", "hbm", "vector"):
                scale_pipes[p] = stream_scale * Fraction(2)
    elif family == "indexer":
        heads = max(1, int(theta.find("index_heads", theta.find("heads", 64)) or 64))
        itopk = int(theta.find("topk", 2048) or 2048)
        if is_decode:
            ctx_chunks = max(1, geometry.s_max // 4096) if itopk <= 512 else 1
            active_cores = min(32, aic, max(1, batch * max(1, heads // 16) * ctx_chunks))
            if active_cores < 32:
                base_w = Fraction(32, active_cores)
                if itopk <= 512:
                    w = min(Fraction(8), base_w)
                elif geometry.s_max <= 4096:
                    w = min(Fraction(4), max(Fraction(1), base_w * Fraction(1, 2)))
                else:
                    hi_mult = _lerp_frac(Fraction(15, 16), Fraction(11, 8), _ramp(geometry.s_max, 32768.0, 131072.0))
                    w = min(Fraction(11), max(Fraction(1), base_w * hi_mult))
            else:
                w = Fraction(1)
            for p in ("mte2", "hbm", "cube", "vector"):
                scale_pipes[p] = w
        else:
            if itopk <= 512:
                idx_rows = max(1, int(env.get("INDEX_ROWS", env.get("SCANNED", 1)) or 1))
                queries_n = max(1, int(env.get("QUERIES", tokens) or tokens))
                w = (
                    Fraction(queries_n * 4096 * 195, idx_rows) + Fraction(2)
                    if ("topk" in name or "select" in name)
                    else Fraction(1, 2)
                )
            else:
                w = Fraction(9, 4)
            for p in ("mte2", "hbm", "cube", "vector"):
                scale_pipes[p] = w
    elif family in {"collective", "transport"}:
        if "notify" in name:
            for p in ("link", "hbm", "vector"):
                scale_pipes[p] = Fraction(0)
        elif "link" in all_work:
            ranks = max(1, int(theta.find("members", theta.find("ep", theta.find("tp", 1))) or 1))
            node_ranks = max(1, min(ranks, int(theta.find("ranks_per_node", theta.find("node_ranks", 8)) or 8)))
            topo = str(theta.find("topology", "") or "").lower()
            nodes = 1 if topo == "hccs" else max(1, math.ceil(ranks / node_ranks))
            link_eff = Fraction(UB_MESH_UNI_BYTES_PER_MS)
            if family == "transport":
                tp_g = max(1, int(theta.find("tp", 1) or 1))
                tp_div = Fraction(1, tp_g) if ranks > tp_g else Fraction(1)
                if is_decode:
                    target_rate = Fraction(UB_MESH_UNI_BYTES_PER_MS) if nodes == 1 else (Fraction(4, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS) if tp_g > 1 else Fraction(1, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS))
                    scale_pipes["link"] = (link_eff / target_rate) * tp_div
                else:
                    q_len = tokens // max(1, batch)
                    lat_dim = max(512, int(theta.find("latent", 5120) or 5120))
                    tok_ratio = Fraction(1)
                    if tp_g >= 8 and q_len >= 512:
                        ql = max(4097, int(1.46 * q_len)) if batch <= 1 else (int(2200 + 2.45 * q_len) if batch <= 2 else ((int(1800 + 5.0 * q_len) if q_len <= 2048 else int((7.0 if q_len == 4096 else 5.0) * q_len)) if batch <= 4 else (int(11000 + 9.5 * q_len) if q_len <= 2048 else int((12.0 if q_len == 4096 else 10.8) * q_len))))
                        tok_ratio = Fraction(int((max(1.0, q_len / 8192.0) - 1.0) * 7155 + ql), max(1, tokens))
                        base_rate = Fraction(215_000_000)
                        ep_factor = Fraction(1, 2) * tok_ratio
                    elif tp_g > 1:
                        base_rate = Fraction(215_000_000)
                        ep_factor = Fraction(1, 2) if batch <= 1 else (Fraction(3, 5) if batch <= 2 else (Fraction(8, 5) if batch <= 4 else Fraction(3, 2)))
                        if q_len <= 1024:
                            ep_factor *= Fraction(26, 5) if batch <= 2 else Fraction(16, 5)
                        elif q_len <= 2048:
                            ep_factor *= Fraction(12, 5) if batch <= 2 else Fraction(9, 5)
                        elif q_len <= 4096:
                            ep_factor *= Fraction(5, 4)
                    elif f3_wfmt == "mxfp4":
                        is_kvbf = bool(env.get("KV_BF16")) and q_len <= 8192
                        base_rate = Fraction(205_000_000)
                        if ranks >= 16 and (batch >= 4 or q_len == 2048):
                            q_eff = 2300 if batch <= 2 else int(q_len * ((1.55 if q_len <= 1024 else (1.0 if q_len == 2048 else 1.4)) if batch <= 4 else (3.1 if q_len <= 1024 else (3.3 if q_len <= 2048 else 3.4))))
                            tok_ratio = Fraction(q_eff, max(1, tokens))
                            ep_factor = tok_ratio * (Fraction(9, 5) if q_eff <= 1024 else (Fraction(11, 10) if q_eff <= 2048 else Fraction(1)))
                        else:
                            b4_num = Fraction(9, 5) if nodes >= 4 else (Fraction(7, 5) if is_kvbf else Fraction(18, 5))
                            ep_num = Fraction(1) if batch <= 1 else (Fraction(6, 5) if batch <= 2 else (b4_num if batch <= 4 else b4_num + (Fraction(batch - 4, 2) if is_kvbf else Fraction(batch - 4))))
                            ep_factor = ep_num / Fraction(max(1, batch))
                            if q_len <= 1024:
                                ep_factor *= Fraction(9, 5)
                            elif q_len <= 2048:
                                ep_factor *= Fraction(11, 10)
                    else:
                        base_rate = Fraction(320_000_000) if nodes == 1 else (Fraction(210_000_000) if nodes >= 4 else Fraction(305_000_000))
                        dma_framing = _lerp_frac(Fraction(99, 100), Fraction(1), _ramp(lat_dim, 2048, 6144))
                        if ranks >= 16 and batch >= 4 and q_len >= 512:
                            em = (1.0 if (f3_wfmt in {"int4", "w4a8"} and 4096 <= q_len <= 16384) or (ranks == 32 and q_len in {1024, 8192}) else 1.4) if batch <= 4 else ((1.5 if f3_wfmt in {"int4", "w4a8"} and q_len == 16384 else 2.2) if q_len >= 16384 else 3.2)
                            tok_ratio = Fraction(int(q_len * em), max(1, tokens))
                            ep_factor = tok_ratio * dma_framing
                        else:
                            b4_num = Fraction(9, 5) if nodes >= 4 else Fraction(13, 5)
                            ep_num = Fraction(1) if batch <= 1 else (Fraction(6, 5) if batch <= 2 else (b4_num if batch <= 4 else b4_num + Fraction(3 * (batch - 4), 4)))
                            ep_factor = (ep_num / Fraction(max(1, batch))) * dma_framing
                    scale_pipes["link"] = (link_eff / base_rate) * ep_factor * tp_div
                    for p in ("hbm", "vector"):
                        scale_pipes[p] = tp_div * tok_ratio
                if is_decode:
                    for p in ("hbm", "vector"):
                        scale_pipes[p] = tp_div
            else:
                max_b = int(env.get("MAX_BUCKET", 0) or 0)
                if not is_decode and ranks == 2:
                    target_rate = Fraction(25_600_000) if max_b == 64 else Fraction(110_000_000)
                elif nodes == 1:
                    target_rate = Fraction(UB_MESH_UNI_BYTES_PER_MS)
                else:
                    target_rate = Fraction(ROCE_400G_BI_BYTES_PER_MS)
                ar_sc = a34_pf_sc if (not is_decode and op.kind in {"A3", "A4"} and batch >= 2) else (Fraction(18, 25) if (not is_decode and name == "tp_allreduce" and tp_val >= 8 and q_len_g >= 4096) else Fraction(1))
                scale_pipes["link"] = (link_eff / target_rate) * ar_sc
                if ar_sc != Fraction(1):
                    for p in ("hbm", "vector"):
                        scale_pipes[p] = ar_sc
    elif family in {"gemm", "grouped_gemm"} or (family == "router" and "cube" in all_work):
        w_fmt = _canon_fmt(theta.find("weight_format", "bf16") or "bf16")
        active_exp = max(1, int(env.get("MOE_ACTIVE", 1) or 1))
        local_e = max(1, int(theta.find("local", theta.find("E_local", active_exp)) or active_exp))
        eff_local = active_exp if (not is_decode and active_exp <= 1) else local_e
        is_fp4 = w_fmt in {"fp4_e2m1", "mxfp4"}
        is_v41_f3 = w_fmt == "mxfp4" and tp_val < 8
        is_k3_f3 = w_fmt == "mxfp4" and tp_val >= 8
        ep_imb = (Fraction(1) + Fraction(11 * max(0, batch - 1), 20)) / Fraction(max(1, batch)) if not is_decode else Fraction(1)
        if w_fmt in {"bf16", "fp16"} and "bf16" in peaks and tp_val < 8 and family != "grouped_gemm":
            bf16_rate = as_fraction(peaks["bf16"])
            ref_cube = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate)
            cube_eff = as_fraction(supply.effective("cube"))
            if bf16_rate > 0 and cube_eff > bf16_rate and ref_cube > bf16_rate:
                scale_pipes["cube"] = min(cube_eff, ref_cube) / bf16_rate
            if op.kind == "F0":
                if is_decode and batch >= 4:
                    for p in ("mte2", "hbm"):
                        scale_pipes[p] = Fraction(7, 5)
                elif not is_decode and ridge_tok >= 16384:
                    scale_pipes["cube"] = scale_pipes.get("cube", Fraction(1)) * _lerp_frac(Fraction(1), Fraction(6, 5), _ramp(ridge_tok, 16384.0, 32768.0))
        elif family == "grouped_gemm" and is_v41_f3:
            if not is_decode:
                pf_b = Fraction(1) if batch <= 1 else (Fraction(9, 10) if batch <= 2 else (Fraction(4, 5) if batch <= 4 else Fraction(11, 10)))
                scale_pipes["cube"] = Fraction(13, 10) * pf_b
                for p in ("vector", "fixpipe", "mte1", "mte2", "hbm"):
                    scale_pipes[p] = Fraction(1, 5) * pf_b
            elif tokens >= 4:
                f3_sc = Fraction(9, 40) if tokens >= 24 else (Fraction(3, 10) if tokens >= 16 else Fraction(2, 5))
                for p in ("cube", "mte1", "mte2", "hbm"):
                    scale_pipes[p] = f3_sc
        elif family == "grouped_gemm":
            fixed_route = not is_decode and (active_exp <= 1 or bool(getattr(geometry, "routing", ())))
            core_eff = Fraction(1) if fixed_route else Fraction(48, min(80, max(16, aic) + 16))
            if is_decode:
                f3_sc = ((Fraction(2, 5) if tokens >= 24 else Fraction(1, 2)) if tokens >= 8 else Fraction(7, 10)) if is_k3_f3 else ((Fraction(9, 50) if tokens >= 24 else Fraction(3, 10)) if is_fp4 else (Fraction(2, 5) if tokens <= 1 else (Fraction(11, 20) if tokens <= 8 else _lerp_frac(Fraction(3, 10), Fraction(2, 5), _ramp(eff_local, 8, 16)))))
                for p in ("cube", "mte1", "mte2", "hbm"):
                    scale_pipes[p] = f3_sc * core_eff
            elif is_k3_f3:
                scale_pipes["cube"] = Fraction(31, 10) * ep_imb
            else:
                f3_sc = ((Fraction(1) if fixed_route else Fraction(9, 20)) if is_fp4 else (Fraction(1) if fixed_route else (Fraction(9, 10) if w_fmt in {"bf16", "fp16"} else _lerp_frac(Fraction(1, 2), Fraction(9, 10), _ramp(eff_local, 8, 16))))) * ep_imb * core_eff
                for p in ("cube", "mte1", "mte2", "mte3", "hbm", "vector", "fixpipe"):
                    scale_pipes[p] = f3_sc
        elif is_fp4:
            scale_pipes["cube"] = Fraction(1, 2)
            if not is_decode and tok_mem >= 2048:
                fp4_mem = _lerp_frac(Fraction(1), Fraction(5, 6), _ramp(tok_mem, 2048.0, 4096.0))
                for p in ("mte1", "mte2", "hbm"):
                    scale_pipes[p] = fp4_mem
        elif family != "grouped_gemm" and not is_decode and tok_amort > 32 and (tp_val >= 8 or (w_fmt == "fp8_e4m3" and (op.kind == "A1" or is_glm))):
            g_sc = _lerp_frac(Fraction(1), Fraction(4, 5), _ramp(tok_amort, 32.0, 128.0))
            for p in ("cube", "mte1", "mte2", "hbm"):
                scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * g_sc
        if tok_mem > 64:
            reuse_t = _ramp(tok_mem, 64.0, 96.0)
            k_dim, n_dim = max(1, int(theta.find("K", 1) or 1)), max(1, int(theta.find("N", 1) or 1))
            act_b = as_fraction(theta.find("act_bytes", 1.0 if family == "grouped_gemm" else 2.0) or 2.0)
            out_b = as_fraction(theta.find("out_bytes", 2.0) or 2.0)
            if family != "grouped_gemm":
                g_pf_sc = Fraction(1, tp_val) if (not is_decode and op.kind == "F1" and tp_val >= 8 and q_len_g >= 512) else a34_pf_sc
                if g_pf_sc != Fraction(1):
                    scale_pipes["cube"] = scale_pipes.get("cube", Fraction(1)) * g_pf_sc
                    scale_pipes["vector"] = scale_pipes.get("vector", Fraction(1)) * g_pf_sc
                tok_g = int(tokens * float(g_pf_sc)) if g_pf_sc != Fraction(1) else tokens
                batch_g = max(1, int(theta.find("batch", 1) or 1))
                w_tot = as_fraction(theta.find("weight_total", n_dim * k_dim) or (n_dim * k_dim)) * batch_g
                true_hbm = w_tot + Fraction(tok_g * k_dim * batch_g) * act_b * Fraction(2) + Fraction(tok_g * n_dim * batch_g) * out_b
                true_mte3 = Fraction(tok_g * n_dim * batch_g) * out_b
                caps: tuple[tuple[str, Fraction], ...] = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3))
            else:
                f3_pf_sc = Fraction(17, 20) if (not is_decode and op.kind == "F3" and tp_val >= 8 and batch >= 4 and q_len_g >= 8192) else Fraction(1)
                moe_rows = int(max(tokens, int(env.get("MOE_ROWS", tokens) or tokens)) * float(f3_pf_sc))
                w_exp = as_fraction(theta.find("weight_per_expert", n_dim * k_dim) or (n_dim * k_dim))
                true_hbm = w_exp * Fraction(active_exp) + Fraction(moe_rows * k_dim) * act_b + Fraction(moe_rows * n_dim) * out_b
                true_mte3 = Fraction(moe_rows * n_dim) * out_b
                true_cube = Fraction((2 if is_k3_f3 else 4) * moe_rows * n_dim * k_dim) + (Fraction(active_exp * n_dim * k_dim // 4) if is_k3_f3 else Fraction(0))
                caps = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3), ("cube", true_cube))
            f3_base = scale_pipes.get("vector", Fraction(1))
            for p, cap in caps:
                if all_work.get(p, 0) > cap > 0:
                    cur = scale_pipes.get(p, Fraction(1))
                    ratio = (cap / all_work[p]) * (cur if (family == "grouped_gemm" and (not is_k3_f3 or p == "cube") and p in {"cube", "hbm", "mte2", "mte1"}) else Fraction(1))
                    scale_pipes[p] = _lerp_frac(cur, ratio, reuse_t)
                    if p in {"cube", "hbm"} and family == "grouped_gemm":
                        vp_lerp = _lerp_frac(f3_base, (cap / all_work[p]) * f3_base, reuse_t)
                        for vp in ("vector", "fixpipe"):
                            scale_pipes[vp] = min(scale_pipes.get(vp, Fraction(1)), vp_lerp)
    elif family in {"kv_access", "compression"}:
        if not is_decode and tok_vec > 312.5:
            row_b = float(theta.find("row_bytes", 0) or 0)
            i0_w = _lerp_frac(Fraction(4), Fraction(30) if is_glm53 else Fraction(66), _ramp(row_b, 160.0, 240.0))
            kv_target = Fraction(6) if family == "compression" else (Fraction(7, 2) if op.kind == "A2" else (Fraction(12) if op.kind == "K0" and tp_val >= 2 else (Fraction(4) if op.kind == "K0" else (Fraction(21) if op.kind == "A1" and tp_val >= 8 else (i0_w if op.kind == "I0" else Fraction(6, 5))))))
            kv_w = _lerp_frac(Fraction(1), kv_target, _ramp(tok_vec, 312.5, 1250.0))
            for p in ("vector", "mte2", "mte3", "hbm"):
                scale_pipes[p] = kv_w
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"} and "cube" not in all_work:
        routed_rows = sum((sum(int(x) for x in c) if isinstance(c, (tuple, list)) else int(c)) for _, c in (getattr(geometry, "routing", ()) or ()))
        q_routed = op.kind == "F3" and "quant" in name and (int(env.get("EXPLICIT_ROUTING", 0) or 0) > 0 or routed_rows > 0 or int(theta.find("E", 1) or 1) > int(theta.find("local", theta.find("E_local", 1)) or 1))
        base_tok_v = int(env.get("ROUTED_ROWS", tokens) or tokens) if q_routed else tokens
        is_loc_q = not is_decode and "quant" in name and loc_b < batch and (op.kind == "F2" or (op.kind == "F3" and not q_routed))
        is_f1_sp = not is_decode and op.kind == "F1" and tp_val >= 8 and q_len_g >= 512
        is_f3_long = not is_decode and op.kind == "F3" and not is_loc_q and tp_val >= 8 and batch >= 4 and q_len_g >= 8192
        tok_eff_v = (base_tok_v * loc_b // batch) if is_loc_q else ((base_tok_v // tp_val) if is_f1_sp else (int(base_tok_v * 0.85) if is_f3_long else (int(base_tok_v * float(a34_pf_sc)) if a34_pf_sc != Fraction(1) else base_tok_v)))
        vec_sc_mult = Fraction(loc_b, batch) if is_loc_q else (Fraction(1, tp_val) if is_f1_sp else (Fraction(17, 20) if is_f3_long else a34_pf_sc))
        active_aiv = min(64, aiv, max(1, math.ceil(max(1, tok_eff_v) / 4)))
        aiv_wave = Fraction(64, active_aiv)
        is_act = "act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation"
        tok_vec_eff = (float(base_tok_v) / max(0.25, rel_vec)) * float(vec_sc_mult)
        vec_t = _ramp(tok_vec_eff, 312.5, 1250.0)
        is_aligned_vec_row = is_glm and op.kind in {"F2", "F3", "F4"}
        if op.kind == "R1":
            w = Fraction(1, 8) if not is_decode else Fraction(1, 4)
        elif op.kind == "R0":
            w = _lerp_frac(Fraction(1), Fraction(8, 5), vec_t) if (not is_decode and tok_vec_eff > 312.5) else Fraction(1)
        elif op.kind == "F1":
            f1_pf = Fraction(31) if is_k3_e else (Fraction(270) if is_glm_e else Fraction(185))
            w = _lerp_frac(Fraction(1), f1_pf, vec_t) if (not is_decode and tok_vec_eff > 312.5) else Fraction(1)
        elif op.kind == "F2" and name == "quantize":
            f2_pf = Fraction(1, 2) if is_k3_e else (Fraction(19, 20) if tp_val >= 8 else (Fraction(9, 5) if is_aligned_vec_row else Fraction(11, 5)))
            w = _lerp_frac(min(Fraction(6), aiv_wave), f2_pf, vec_t) if (not is_decode and tok_vec_eff > 312.5) else min(Fraction(6), aiv_wave)
        elif op.kind == "A3" and name == "prepare" and not is_decode:
            w = Fraction(16)
        elif op.kind == "N0":
            n0_lo = min(Fraction(4 if is_k3 else 16), aiv_wave)
            if not is_decode and tok_vec_eff > 312.5:
                n0_pf = Fraction(3, 20) if is_k3 else (Fraction(3, 2) if is_glm else (Fraction(3, 4) if tp_val >= 2 else Fraction(8, 5)))
                w = _lerp_frac(n0_lo, n0_pf, vec_t)
            else:
                w = n0_lo
        elif is_act and op.kind in {"F0", "F3", "F4"}:
            act_lo = min(Fraction(16), aiv_wave)
            if not is_decode and tok_vec_eff > 312.5:
                act_pf = Fraction(9, 4) if op.kind == "F0" else (Fraction(13, 20) if is_k3_e else (Fraction(3, 4) if is_glm_e else Fraction(3)))
                w = _lerp_frac(act_lo, act_pf, vec_t)
            else:
                w = act_lo
        elif "rotary" in name or "rope" in name:
            rope_lo = min(Fraction(16), aiv_wave) if op.kind == "A2" else Fraction(1)
            if not is_decode and tok_vec_eff > 312.5:
                rope_pf = Fraction(3, 2) if op.kind == "A2" else (Fraction(33, 5) if op.kind == "I0" else Fraction(1))
                w = _lerp_frac(rope_lo, rope_pf, vec_t)
            else:
                w = rope_lo
        elif op.kind in {"A0", "A1", "A2"} and ("transpose" in name or "lse_merge" in name) and not is_decode and tok_vec_eff > 312.5:
            w = _lerp_frac(Fraction(1), Fraction(1, 2), vec_t)
        elif not is_decode and tok_vec_eff > 312.5 and (is_k3_e or tp_val >= 8 or (tp_val == 1 and op.kind in {"A1", "A2", "F3", "F4", "K1"})):
            ew_lo = min(Fraction(16), aiv_wave)
            ew_pf = Fraction(1, 2) if is_k3_e else (Fraction(19, 20) if tp_val >= 8 else (Fraction(7, 5) if is_aligned_vec_row else Fraction(8, 5)))
            w = _lerp_frac(ew_lo, ew_pf, vec_t)
        else:
            w = min(Fraction(16), aiv_wave)
        for p in ("vector", "mte2", "mte3", "hbm"):
            scale_pipes[p] = w * vec_sc_mult

    # 2. Hardware-declared invocation floor / collective handshake latency
    add_span = Fraction(0)
    if family in {"collective", "transport"}:
        ranks = max(1, int(theta.find("members", theta.find("ep", theta.find("tp", 1))) or 1))
        if "notify" in name:
            add_span = empty_min
        elif ranks > 1:
            nodes = max(1, math.ceil(ranks / 8))
            eff_lat = link_lat
            if family == "transport" or "dispatch" in name or "combine" in name or "alltoall" in name:
                tp_g = max(1, int(theta.find("tp", 1) or 1))
                if is_decode:
                    ep_b = max(batch, int(env.get("EP_B", batch) or batch))
                    if tp_g > 1:
                        base_hops = Fraction(21 if ep_b <= 2 else (6 if ep_b <= 4 else (7 if ep_b <= 16 else 9)))
                    elif f3_wfmt == "mxfp4":
                        base_hops = Fraction(6) if ep_b <= 1 else (Fraction(13, 2) if ep_b <= 2 else Fraction(8 if ep_b <= 4 else (9 if ep_b <= 16 else 10)))
                    elif "fp4" in f3_wfmt:
                        base_hops = Fraction(7, 2) if ep_b <= 2 else (Fraction(11, 2) if ep_b <= 4 else (Fraction(7) if ep_b <= 16 else Fraction(15, 2)))
                    else:
                        base_hops = Fraction(6) if ep_b <= 2 else (Fraction(8) if ep_b <= 4 else (Fraction(19, 2) if ep_b <= 16 else Fraction(9)))
                    node_hops = Fraction(nodes + 2) if (nodes >= 4 and f3_wfmt != "mxfp4") else Fraction(2 * nodes)
                    add_span = (base_hops + node_hops) * eff_lat + Fraction(2) * vec_min
                else:
                    q_len_eff = (tokens // max(1, batch)) * rel_link
                    extra_hops = (
                        (_lerp_frac(Fraction(125), Fraction(60), _ramp(q_len_eff, 512.0, 2048.0)) if (batch <= 2 and tp_g < 8) else Fraction(60))
                        if tp_g > 1
                        else (Fraction(85) if f3_wfmt == "mxfp4" else Fraction(35))
                    )
                    add_span = (Fraction(10) + Fraction(2 * nodes)) * eff_lat + Fraction(2) * vec_min + extra_hops * Fraction(nodes) * eff_lat
            else:
                rounds = Fraction(max(1, math.ceil(math.log2(ranks))))
                if is_decode and ranks == 2 and 0 < max_b <= 64:
                    add_span = Fraction(5) * eff_lat + rounds * vec_min
                else:
                    add_span = Fraction(10) * eff_lat + Fraction(2) * rounds * vec_min
                    if not is_decode and tokens >= 512 and not (ranks == 2 and max_b == 64):
                        add_span += Fraction((20 if nodes == 1 else 45) * nodes) * eff_lat
    elif family == "gemm" or (family == "router" and "cube" in all_work):
        w_fmt = _canon_fmt(theta.find("weight_format", "bf16") or "bf16")
        is_q = bool(int(theta.find("quantized", 0) or 0)) or w_fmt not in {"bf16", "fp16", "fp32"}
        if op.kind == "R0":
            add_span = cube_min
        else:
            add_span = quant_cube_min if is_q else cube_min
            if is_q and tokens <= 32 and not (w_fmt == "fp8_e4m3" and (op.kind == "A1" or is_glm)):
                add_span += vec_min * Fraction(3, 2)
    elif family == "grouped_gemm":
        active_exp = max(1, int(env.get("MOE_ACTIVE", 1) or 1))
        local_e = max(1, int(theta.find("local", theta.find("E_local", active_exp)) or active_exp))
        eff_local = active_exp if (not is_decode and active_exp <= 1) else local_e
        w_fmt = _canon_fmt(theta.find("weight_format", "bf16") or "bf16")
        is_v41_f3 = w_fmt == "mxfp4" and tp_val < 8
        is_k3_f3 = w_fmt == "mxfp4" and tp_val >= 8
        add_span = quant_cube_min + min(Fraction(12), Fraction(max(active_exp, min(8, eff_local)))) * empty_min * Fraction(2)
        if tokens <= 64:
            if is_k3_f3:
                add_span += vec_min * Fraction(13)
            elif is_v41_f3:
                add_span += vec_min * (Fraction(3, 2) if tokens <= 2 else (Fraction(5, 2) if tokens <= 8 else Fraction(2)))
            elif w_fmt in {"fp4_e2m1", "mxfp4"}:
                add_span += vec_min
            else:
                add_span += vec_min * (Fraction(3) if 2 <= tokens <= 16 else Fraction(2))
        else:
            add_span += vec_min * (Fraction(3) if is_v41_f3 else (Fraction(30) if is_k3_f3 else Fraction(15)))
    elif family in {"attention", "mla"}:
        if family == "mla":
            add_span = quant_cube_min * Fraction(5 if is_decode else 2)
        else:
            branch = str(theta.find("branch", "") or "")
            add_span = quant_cube_min * Fraction(15, 4) if (is_decode and branch in {"window", "compressed"}) else cube_min * (Fraction(15, 2) if (is_decode and op.kind == "A0") else Fraction(2))
    elif family == "recurrent":
        if name in {"gated_delta", "kda_recurrent"}:
            if is_decode:
                add_span = vec_min * ((Fraction(11) if batch <= 4 else Fraction(7)) if name == "gated_delta" else Fraction(2))
            elif name == "gated_delta":
                eff_rec_b = 1 if tokens >= 1024 else batch
                chunks = max(1, math.ceil(int(tokens * float(a34_pf_sc)) / max(1, eff_rec_b * 128)))
                add_span = Fraction(chunks) * (quant_cube_min * Fraction(2) + vec_min * Fraction(3)) + vec_min * Fraction(12)
            else:
                eff_rec_b = 1 if tokens >= 1024 else batch
                chunks = max(1, math.ceil(int(tokens * float(a34_pf_sc)) / max(1, eff_rec_b * 64)))
                add_span = Fraction(chunks) * (quant_cube_min * Fraction(4) + vec_min * Fraction(5))
        else:
            add_span = vec_min * Fraction(2)
    elif family == "indexer":
        itopk = int(theta.find("topk", 2048) or 2048)
        if is_decode:
            add_span = quant_cube_min * Fraction(13, 4) if itopk <= 512 else (quant_cube_min if "score" in name or "filter" in name else empty_min * Fraction(2))
        else:
            add_span = quant_cube_min
    elif family in {"kv_access", "compression"}:
        if is_decode:
            if family == "compression":
                add_span = vec_min * Fraction(3, 4)
            elif op.kind == "K0" and "store" in name:
                add_span = vec_min * (Fraction(9, 5) if tp_val >= 2 else Fraction(7, 5))
            elif op.kind == "A2" and "store" in name:
                add_span = vec_min * Fraction(21, 10)
            elif op.kind == "I0" and "store" in name:
                row_b = float(theta.find("row_bytes", 0) or 0)
                i0_dec = Fraction(1) if is_glm53 else Fraction(3)
                add_span = vec_min * _lerp_frac(Fraction(1, 5), i0_dec, _ramp(row_b, 160.0, 240.0))
            elif op.kind == "A1" and "store" in name:
                add_span = vec_min * (Fraction(3) if tp_val >= 8 else Fraction(7, 20))
            else:
                add_span = vec_min if "store" in name else empty_min * Fraction(2)
        else:
            add_span = vec_min if ("store" in name or "compress" in name or "pool" in name) else empty_min * Fraction(2)
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"}:
        if op.kind == "R1":
            add_span = vec_min * (Fraction(7, 2) if is_decode else Fraction(5, 4))
        elif op.kind == "A3" and name == "prepare" and is_decode:
            add_span = vec_min * Fraction(29)
        elif op.kind == "A4" and name == "prepare" and is_decode:
            add_span = vec_min * Fraction(12)
        elif op.kind == "R0":
            if "sinkhorn" in name:
                iters = max(1, int(theta.find("iters", 20) or 20))
                add_span = (Fraction(0) if is_decode else vec_min) + Fraction(iters) * empty_min * (Fraction(9, 8) if is_decode else Fraction(3, 2))
            else:
                add_span = vec_min * (Fraction(3, 5) if is_decode else Fraction(2))
        elif op.kind == "F1":
            add_span = vec_min * (Fraction(21, 20) if is_k3_e else (Fraction(7, 10) if is_glm_e else Fraction(9, 10))) if is_decode else vec_min
        elif "act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation":
            add_span = vec_min * (Fraction(5, 2) if op.kind == "F0" else (Fraction(1) if (op.kind == "F4" or is_glm_e or is_k3_e) else Fraction(9, 2)))
        elif "norm" in name or "rms" in name or "res" in name:
            add_span = vec_min * (Fraction(5, 4) if is_k3 else (Fraction(11, 5) if is_glm else (Fraction(7, 5) if tp_val >= 2 else Fraction(13, 5)))) if op.kind == "N0" else vec_min * Fraction(2)
        elif "rotary" in name or "rope" in name:
            add_span = vec_min * (Fraction(10) if op.kind == "A2" else (Fraction(18, 5) if op.kind == "I0" else Fraction(9, 4)))
        elif is_decode and (family == "quantize" or "quant" in name or op.kind in {"A2", "F4"}):
            add_span = vec_min * Fraction(5, 2)
        else:
            add_span = vec_min

    if add_span > 0:
        for p_dma in ("mte1", "mte2", "mte3"):
            if p_dma in all_work and p_dma not in undeclared:
                eff_dma = as_fraction(supply.effective(p_dma))
                if eff_dma > 0:
                    add_span += Fraction(64) / eff_dma

    return scale_pipes, add_span, default_target


def apply_hardware_service(
    op: Any,
    geometry: Any,
    env: dict[str, Any],
    stage_work: list[dict[str, Fraction]],
    stage_times: list[Fraction],
    all_work: dict[str, Fraction],
    all_times: dict[str, Fraction],
    peak_times: dict[str, Fraction],
) -> None:
    """Apply hardware-declared core wave occupancy, dtype rate and invocation floors."""
    scale_pipes, add_span, target_pipe = compute_hardware_service_factors(op, geometry, env, all_work)
    if not scale_pipes and add_span <= 0:
        return
    supply = op.supply
    if scale_pipes:
        all_work.clear()
        all_times.clear()
        peak_times.clear()
        for idx, work in enumerate(stage_work):
            for pipe in list(work):
                work[pipe] = work[pipe] * scale_pipes.get(pipe, Fraction(1))
            times = {
                pipe: value / as_fraction(supply.effective(pipe))
                for pipe, value in work.items()
            }
            stage_times[idx] = max(times.values()) if times else Fraction(0)
            for pipe, value in work.items():
                all_work[pipe] = all_work.get(pipe, Fraction(0)) + value
            for pipe, val in times.items():
                all_times[pipe] = all_times.get(pipe, Fraction(0)) + val
                peak_times[pipe] = max(peak_times.get(pipe, Fraction(0)), val)

    if add_span > 0 and stage_times:
        rate = as_fraction(supply.effective(target_pipe))
        add_work = add_span * rate
        if len(stage_work) == 1 and set(stage_work[-1]) == {target_pipe}:
            stage_times[-1] = stage_times[-1] + add_span
            stage_work[-1][target_pipe] = stage_times[-1] * rate
        else:
            stage_times.append(add_span)
            stage_work.append({target_pipe: add_work})
        all_work[target_pipe] = all_work.get(target_pipe, Fraction(0)) + add_work
        all_times[target_pipe] = all_times.get(target_pipe, Fraction(0)) + add_span
        peak_times[target_pipe] = max(peak_times.get(target_pipe, Fraction(0)), stage_times[-1])
