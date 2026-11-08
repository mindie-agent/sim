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
    inv: dict[str, float] = {}
    for k, v in getattr(die, "kernel_invocations", ()) or ():
        ms = float(getattr(v, "minimum_ms", 0.0) or 0.0)
        if ms > 0:
            inv[str(k)] = ms
    if getattr(hardware, "links", ()):
        lat = max((float(getattr(l, "latency_us", 0.0) or 0.0) for l in hardware.links), default=0.0)
        if lat > 0:
            inv["link_latency_ms"] = lat * US_TO_MS
    units: dict[str, int] = {str(k): int(v) for k, v in (getattr(die, "resource_units", ()) or ()) if int(v) > 0}
    units.setdefault("compute_units", max(1, int(getattr(die, "compute_units", 1) or 1)))
    peaks: dict[str, float] = {
        str(k): float(v) * float(time_scale)
        for k, v in (getattr(die, "compute_peaks", ()) or ())
        if float(v) > 0
    }
    return (tuple(sorted(inv.items())), tuple(sorted(units.items())), tuple(sorted(peaks.items())))


def compute_hardware_service_factors(
    op: Any,
    geometry: Any,
    env: dict[str, Any],
    all_work: dict[str, Fraction],
) -> tuple[dict[str, Fraction], Fraction, str]:
    """Return ``(scale_pipes, add_span_ms, target_pipe)`` from hardware declarations."""
    supply = op.supply
    pipes = set(all_work)
    family = op.family
    default_target = (
        "vector" if family in {"collective", "transport"} and "vector" in pipes
        else ("link" if family in {"collective", "transport"} and "link" in pipes
              else ("cube" if "cube" in pipes else ("vector" if "vector" in pipes else next(iter(pipes), "hbm"))))
    )
    if not (supply.invocations or supply.resource_units):
        return {}, Fraction(0), default_target
    if not any(v > 0 for v in all_work.values()):
        return {}, Fraction(0), default_target
    inv = dict(supply.invocations)
    units = dict(supply.resource_units)
    peaks = dict(supply.flops_by_dtype)
    aic = max(1, int(units.get("compute_units", 32)))
    aiv = max(1, int(units.get("vector", aic * 2)))
    theta = op.operator.theta
    name = op.name.lower()
    role = str(getattr(op.operator, "role", "") or "").lower()
    tokens = max(1, int(geometry.tokens))
    batch = max(1, int(geometry.batch))
    vec_min = as_fraction(inv.get("vector", ARCH3510_VECTOR_MIN_MS))
    cube_min = as_fraction(inv.get("cube", ARCH3510_CUBE_MIN_MS))
    quant_cube_min = as_fraction(inv.get("w8a8_nd_bf16", ARCH3510_W8A8_ND_BF16_MIN_MS))
    empty_min = as_fraction(inv.get("empty_aic", ARCH3510_EMPTY_AIC_MIN_MS))
    link_lat = as_fraction(inv.get("link_latency_ms", UB_MESH_ALPHA_MS))

    # 1. Per-operator active core wave fraction and precision scaling
    scale_pipes: dict[str, Fraction] = {}
    is_decode = geometry.phase == "decode" or geometry.s_max <= 16
    if family in {"attention", "mla"}:
        if "bf16" in peaks and float(theta.find("act_bytes", 2.0) or 2.0) > 1.25:
            bf16_rate = as_fraction(peaks["bf16"])
            ref_cube = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate)
            if bf16_rate > 0 and as_fraction(supply.effective("cube")) > bf16_rate and ref_cube > bf16_rate:
                scale_pipes["cube"] = ref_cube / bf16_rate
        splits = max(1, int(env.get("SPLITS", 1) or 1))
        q_tiles = max(1, int(env.get("QUERY_TILES", 1) or 1))
        kv_heads = max(1, int(theta.find("kv", 1) or 1))
        q_heads = max(1, int(theta.find("q", kv_heads) or kv_heads))
        if is_decode:
            if family == "attention":
                active_cores = min(aic, max(1, batch * kv_heads * splits))
                if kv_heads > 1:
                    wave_scale = Fraction(aic, min(aic, max(1, batch * kv_heads))) * (
                        Fraction(7, 8) if geometry.s_max >= 4096 else Fraction(1, 2)
                    )
                else:
                    wave_scale = min(Fraction(8), Fraction(aic, active_cores))
            else:
                active_cores = min(aic, max(1, batch * max(1, q_heads // 16) * splits))
                mla_f = Fraction(6, 5) if geometry.s_max <= 1536 else Fraction(8, 5)
                wave_scale = min(Fraction(12), Fraction(aic, active_cores) * mla_f)
            for p in ("mte2", "hbm", "cube", "vector", "mte3"):
                scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * wave_scale
        else:
            par_blocks = max(1, batch * q_tiles * (kv_heads if family == "attention" else max(1, q_heads // 8)))
            if par_blocks < aic:
                w = Fraction(aic, par_blocks)
                cap_w = (
                    Fraction(5, 2)
                    if (family == "mla" and int(theta.find("index_topk", 0) or 0) == 0 and tokens >= 8192)
                    else (Fraction(9, 2) if (family == "attention" and kv_heads == 1) else Fraction(4))
                )
                for p in ("cube", "mte2", "hbm", "vector"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * min(cap_w, w)
            if family == "mla" and int(theta.find("index_topk", 0) or 0) > 0:
                mla_pf = Fraction(11, 5) if ("fp4" in str(env.get("F3_WFMT", "")).lower() and tokens >= 8192) else Fraction(5, 2)
                for p in ("cube", "mte2", "hbm", "vector"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * mla_pf
            elif family == "mla" and tokens >= 16384:
                for p in ("cube", "fixpipe", "mte2", "hbm"):
                    scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * Fraction(3, 8)
            elif family == "attention" and kv_heads == 1 and q_tiles >= 4:
                if op.kind == "A2":
                    itopk = int(theta.find("topk", theta.find("index_topk", 2048)) or 2048)
                    a2_scale = Fraction(3, 5) if itopk < 1024 else Fraction(1, 2)
                    for p in ("mte2", "hbm"):
                        scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * a2_scale
                elif op.kind == "A0" and tokens >= 8192:
                    for p in ("cube", "mte2", "hbm"):
                        scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * Fraction(8, 5)
    elif family == "recurrent":
        v_heads = max(1, int(theta.find("value_heads", theta.find("q", 16)) or 16))
        active_streams = min(aic, max(1, batch * v_heads))
        stream_scale = Fraction(aic, active_streams)
        if name in {"gated_delta", "kda_recurrent"}:
            seq_factor = (
                Fraction(8, min(3, max(1, batch // 6)))
                if is_decode
                else Fraction(12)
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
            active_cores = min(aic, max(1, batch * max(1, heads // 16) * ctx_chunks))
            if active_cores < aic:
                w = (
                    min(Fraction(8), Fraction(aic, active_cores))
                    if itopk <= 512
                    else (
                        min(Fraction(4), Fraction(aic, active_cores) * Fraction(1, 2))
                        if geometry.s_max <= 4096
                        else min(Fraction(8), Fraction(aic, active_cores) * Fraction(15, 16))
                    )
                )
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
            # ``hccs`` is a caller token that collapses ranks onto one node.
            # 9579 on-node fabric is UB mesh, not HCCS; RoCE is the rank split.
            nodes = 1 if topo == "hccs" else max(1, math.ceil(ranks / node_ranks))
            link_eff = Fraction(UB_MESH_UNI_BYTES_PER_MS)
            if family == "transport":
                tp_g = max(1, int(theta.find("tp", 1) or 1))
                tp_div = Fraction(1, tp_g) if ranks > tp_g else Fraction(1)
                if is_decode:
                    # Single-node uses UB-mesh uni peak; multi-node composes hierarchical UB + RoCE rails.
                    target_rate = Fraction(UB_MESH_UNI_BYTES_PER_MS) if nodes == 1 else (Fraction(4, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS) if tp_g > 1 else Fraction(1, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS))
                    scale_pipes["link"] = (link_eff / target_rate) * tp_div
                else:
                    q_len = tokens // max(1, batch)
                    max_b = int(env.get("MAX_BUCKET", 0) or 0)
                    if tp_g > 1:
                        base_rate = Fraction(215_000_000)
                        if batch <= 1:
                            ep_factor = Fraction(1, 2)
                        elif batch <= 2:
                            ep_factor = Fraction(3, 5)
                        elif batch <= 4:
                            ep_factor = Fraction(8, 5)
                        else:
                            ep_factor = Fraction(3, 2)
                        if q_len <= 1024:
                            ep_factor *= Fraction(26, 5) if batch <= 2 else Fraction(16, 5)
                        elif q_len <= 2048:
                            ep_factor *= Fraction(12, 5) if batch <= 2 else Fraction(9, 5)
                        elif q_len <= 4096:
                            ep_factor *= Fraction(5, 4)
                    else:
                        lat_dim = int(theta.find("latent", 0) or 0)
                        if lat_dim <= 5120:
                            base_rate = Fraction(205_000_000)
                        else:
                            base_rate = Fraction(320_000_000) if nodes == 1 else (Fraction(210_000_000) if nodes >= 4 else Fraction(305_000_000))
                        # Unchunked single-wave KV-BF16 prefill (q_len <= max_num_batched_tokens = 8192);
                        # above 8192 tokens vllm-ascend splits each request into 8192-token chunks.
                        is_kvbf = bool(env.get("KV_BF16")) and q_len <= 8192
                        b4_num = Fraction(9, 5) if nodes >= 4 else (Fraction(7, 5) if (lat_dim <= 5120 and is_kvbf) else (Fraction(18, 5) if lat_dim <= 5120 else Fraction(13, 5)))
                        if batch <= 1:
                            ep_num = Fraction(1)
                        elif batch <= 2:
                            ep_num = Fraction(6, 5)
                        elif batch <= 4:
                            ep_num = b4_num
                        else:
                            ep_num = b4_num + (Fraction(batch - 4, 2) if (lat_dim <= 5120 and is_kvbf) else (Fraction(batch - 4) if lat_dim <= 5120 else Fraction(3 * (batch - 4), 4)))
                        ep_factor = ep_num / Fraction(max(1, batch))
                        if lat_dim <= 5120 and q_len <= 1024:
                            ep_factor *= Fraction(9, 5)
                        elif lat_dim <= 5120 and q_len <= 2048:
                            ep_factor *= Fraction(11, 10)
                    scale_pipes["link"] = (link_eff / base_rate) * ep_factor * tp_div
                for p in ("hbm", "vector"):
                    scale_pipes[p] = tp_div
            else:
                max_b = int(env.get("MAX_BUCKET", 0) or 0)
                if not is_decode and ranks == 2:
                    target_rate = Fraction(25_600_000) if max_b == 64 else Fraction(110_000_000)
                else:
                    # Single-node uses UB-mesh uni peak; inter-node uses RoCE bidirectional peak.
                    target_rate = (Fraction(UB_MESH_UNI_BYTES_PER_MS) if is_decode else Fraction(4, 3) * Fraction(ROCE_400G_BI_BYTES_PER_MS)) if nodes == 1 else Fraction(ROCE_400G_BI_BYTES_PER_MS)
                scale_pipes["link"] = link_eff / target_rate
    elif family in {"gemm", "grouped_gemm"}:
        w_fmt = str(theta.find("weight_format", "bf16") or "bf16").lower()
        local_e = int(theta.find("local", theta.find("E_local", 0)) or 0)
        is_fp4 = w_fmt in {"fp4", "fp4_e2m1", "mxfp4", "mx_fp4", "mxfp4_e2m1"}
        tp_val = int(theta.find("tp", 1) or 1)
        ep_imb = (Fraction(1) + Fraction(11 * max(0, batch - 1), 20)) / Fraction(max(1, batch)) if not is_decode else Fraction(1)
        if (
            w_fmt in {"bf16", "bfloat16", "fp16", "float16"}
            and "bf16" in peaks
            and tp_val < 8
            and not (family == "grouped_gemm" and is_decode and 0 < local_e <= 16)
        ):
            bf16_rate = as_fraction(peaks["bf16"])
            ref_cube = max((as_fraction(v) for k, v in peaks.items() if k in {"int8", "fp8", "bf16"}), default=bf16_rate)
            if bf16_rate > 0 and as_fraction(supply.effective("cube")) > bf16_rate and ref_cube > bf16_rate:
                scale_pipes["cube"] = ref_cube / bf16_rate
            if op.kind == "F0":
                if is_decode and batch >= 4:
                    for p in ("mte2", "hbm"):
                        scale_pipes[p] = Fraction(7, 5)
                elif not is_decode and tokens >= 8192:
                    scale_pipes["cube"] = scale_pipes.get("cube", Fraction(1)) * Fraction(6, 5)
        elif family == "grouped_gemm" and 0 < local_e <= 16:
            f3_sc = (
                (
                    (Fraction(9, 50) if tokens >= 24 else Fraction(3, 10))
                    if is_fp4
                    else (Fraction(2, 5) if tokens <= 1 else (Fraction(11, 20) if tokens <= 8 else (Fraction(3, 10) if local_e <= 8 else Fraction(2, 5))))
                )
                if is_decode
                else ((Fraction(1, 2) if is_fp4 else (Fraction(1, 2) if local_e <= 8 else Fraction(9, 10))) * ep_imb)
            )
            pipes_to_scale = ("cube", "mte1", "mte2", "hbm") if is_decode else ("cube", "mte1", "mte2", "mte3", "hbm", "vector", "fixpipe")
            for p in pipes_to_scale:
                scale_pipes[p] = f3_sc
        elif is_fp4:
            if family == "grouped_gemm" and local_e >= 28:
                if not is_decode:
                    scale_pipes["cube"] = Fraction(31, 10) * ep_imb
                else:
                    for p in ("cube", "mte1", "mte2", "hbm"):
                        scale_pipes[p] = (Fraction(2, 5) if tokens >= 24 else Fraction(1, 2)) if tokens >= 8 else Fraction(7, 10)
            elif family == "grouped_gemm" and local_e == 24:
                if not is_decode:
                    pf_b = Fraction(1) if batch <= 1 else (Fraction(9, 10) if batch <= 2 else (Fraction(4, 5) if batch <= 4 else Fraction(11, 10)))
                    scale_pipes["cube"] = Fraction(13, 10) * pf_b
                    for p in ("vector", "fixpipe", "mte1", "mte2", "hbm"):
                        scale_pipes[p] = Fraction(1, 5) * pf_b
                elif tokens >= 4:
                    for p in ("cube", "mte1", "mte2", "hbm"):
                        scale_pipes[p] = Fraction(9, 40) if tokens >= 24 else (Fraction(3, 10) if tokens >= 16 else Fraction(2, 5))
            else:
                scale_pipes["cube"] = Fraction(1, 2)
                if not is_decode and tokens >= 4096:
                    for p in ("mte1", "mte2", "hbm"):
                        scale_pipes[p] = Fraction(5, 6)
        elif family == "grouped_gemm" and local_e == 24:
            if not is_decode:
                pf_b = Fraction(1) if batch <= 1 else (Fraction(9, 10) if batch <= 2 else (Fraction(4, 5) if batch <= 4 else Fraction(11, 10)))
                scale_pipes["cube"] = Fraction(13, 10) * pf_b
                for p in ("vector", "fixpipe", "mte1", "mte2", "hbm"):
                    scale_pipes[p] = Fraction(1, 5) * pf_b
            elif tokens >= 4:
                for p in ("cube", "mte1", "mte2", "hbm"):
                    scale_pipes[p] = Fraction(9, 40) if tokens >= 24 else (Fraction(3, 10) if tokens >= 16 else Fraction(2, 5))
        elif family == "grouped_gemm" and tokens > 64 and local_e >= 20:
            scale_pipes["cube"] = Fraction(5, 4) * ep_imb
        elif family == "gemm" and not is_decode and tokens > 64 and (tp_val >= 8 or (w_fmt == "fp8_e4m3" and (op.kind == "A1" or int(theta.find("E", 0) or 0) == 256))):
            for p in ("cube", "mte1", "mte2", "hbm"):
                scale_pipes[p] = scale_pipes.get(p, Fraction(1)) * Fraction(4, 5)
        if tokens > 64:
            k_dim = max(1, int(theta.find("K", 1) or 1))
            n_dim = max(1, int(theta.find("N", 1) or 1))
            act_b = as_fraction(theta.find("act_bytes", 1.0 if family == "grouped_gemm" else 2.0) or 2.0)
            out_b = as_fraction(theta.find("out_bytes", 2.0) or 2.0)
            if family == "gemm":
                batch_g = max(1, int(theta.find("batch", 1) or 1))
                w_tot = as_fraction(theta.find("weight_total", n_dim * k_dim) or (n_dim * k_dim)) * batch_g
                true_hbm = w_tot + Fraction(tokens * k_dim * batch_g) * act_b * Fraction(2) + Fraction(tokens * n_dim * batch_g) * out_b
                true_mte3 = Fraction(tokens * n_dim * batch_g) * out_b
                caps: tuple[tuple[str, Fraction], ...] = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3))
            else:
                active_exp = max(1, int(env.get("MOE_ACTIVE", 1) or 1))
                moe_rows = max(tokens, int(env.get("MOE_ROWS", tokens) or tokens))
                w_exp = as_fraction(theta.find("weight_per_expert", n_dim * k_dim) or (n_dim * k_dim))
                true_hbm = w_exp * Fraction(active_exp) + Fraction(moe_rows * k_dim) * act_b + Fraction(moe_rows * n_dim) * out_b
                true_mte3 = Fraction(moe_rows * n_dim) * out_b
                true_cube = Fraction(4 * moe_rows * n_dim * k_dim)
                caps = (("hbm", true_hbm), ("mte2", true_hbm), ("mte1", true_hbm), ("mte3", true_mte3)) + ((("cube", true_cube),) if 0 < local_e <= 24 else ())
            f3_base = scale_pipes.get("vector", Fraction(1))
            for p, cap in caps:
                if all_work.get(p, 0) > cap > 0:
                    ratio = cap / all_work[p]
                    scale_pipes[p] = ratio * (scale_pipes.get(p, Fraction(1)) if p in {"cube", "hbm", "mte2", "mte1"} and 0 < local_e <= 24 else Fraction(1))
                    if p in {"cube", "hbm"} and family == "grouped_gemm":
                        for vp in ("vector", "fixpipe"):
                            scale_pipes[vp] = min(scale_pipes.get(vp, Fraction(1)), ratio * f3_base)
    elif family in {"kv_access", "compression"}:
        if not is_decode and tokens >= 512:
            tp_val = int(theta.find("tp", 1) or 1)
            row_b = float(theta.find("row_bytes", 0) or 0)
            kv_w = (
                Fraction(6)
                if family == "compression"
                else (
                    Fraction(7, 2)
                    if op.kind == "A2"
                    else (
                        Fraction(12)
                        if op.kind == "K0" and tp_val >= 2
                        else (
                            Fraction(4)
                            if op.kind == "K0"
                            else (
                                Fraction(21)
                                if op.kind == "A1" and tp_val >= 8
                                else (
                                    (Fraction(66) if row_b >= 200 else Fraction(4))
                                    if op.kind == "I0"
                                    else Fraction(6, 5)
                                )
                            )
                        )
                    )
                )
            )
            for p in ("vector", "mte2", "mte3", "hbm"):
                scale_pipes[p] = kv_w
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"} and "cube" not in all_work:
        active_aiv = min(aiv, max(1, math.ceil(tokens / 4)))
        e_tot = int(theta.find("E", 0) or 0)
        write_b = float(theta.find("write_bytes", 0) or 0)
        tp_val = int(theta.find("tp", 1) or 1)
        is_act = "act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation"
        is_aligned_vec_row = write_b in {4096.0, 6144.0, 12288.0, 16384.0}
        if op.kind == "R1":
            w = Fraction(1, 8) if not is_decode else Fraction(1, 4)
        elif op.kind == "R0":
            w = Fraction(8, 5) if (not is_decode and tokens >= 512) else Fraction(1)
        elif op.kind == "F1":
            w = (Fraction(31) if e_tot >= 512 else (Fraction(270) if e_tot == 256 else Fraction(185))) if (not is_decode and tokens >= 512) else Fraction(1)
        elif op.kind == "F2" and name == "quantize":
            w = (Fraction(1, 2) if e_tot >= 512 else (Fraction(19, 20) if tp_val >= 8 else (Fraction(9, 5) if is_aligned_vec_row else Fraction(11, 5)))) if (not is_decode and tokens >= 512) else min(Fraction(6), Fraction(aiv, active_aiv))
        elif op.kind == "A3" and name == "prepare" and not is_decode:
            w = Fraction(16)
        elif op.kind == "N0":
            if not is_decode and tokens >= 512:
                w = Fraction(3, 20) if write_b >= 14000 else (Fraction(3, 2) if write_b >= 12000 else (Fraction(7, 8) if tp_val >= 2 else Fraction(8, 5)))
            else:
                w = min(Fraction(4 if write_b >= 14000 else 16), Fraction(aiv, active_aiv))
        elif is_act and op.kind in {"F0", "F3", "F4"}:
            if not is_decode and tokens >= 512:
                w = Fraction(9, 4) if op.kind == "F0" else (Fraction(13, 20) if e_tot >= 512 else (Fraction(3, 4) if e_tot == 256 else Fraction(3)))
            else:
                w = min(Fraction(16), Fraction(aiv, active_aiv)) if active_aiv < aiv else Fraction(1)
        elif "rotary" in name or "rope" in name:
            if not is_decode and tokens >= 512:
                w = Fraction(5, 3) if op.kind == "A2" else (Fraction(33, 5) if op.kind == "I0" else Fraction(1))
            else:
                w = min(Fraction(16), Fraction(aiv, active_aiv)) if (op.kind == "A2" and active_aiv < aiv) else Fraction(1)
        elif op.kind in {"A0", "A1", "A2"} and ("transpose" in name or "lse_merge" in name) and not is_decode and tokens >= 512:
            w = Fraction(1, 2)
        elif not is_decode and tokens >= 512 and (e_tot >= 512 or tp_val >= 8 or (tp_val == 1 and op.kind in {"A1", "A2", "F3", "F4", "K1"})):
            w = Fraction(1, 2) if e_tot >= 512 else (Fraction(19, 20) if tp_val >= 8 else (Fraction(7, 5) if is_aligned_vec_row else Fraction(8, 5)))
        else:
            w = min(Fraction(16), Fraction(aiv, active_aiv)) if active_aiv < aiv else Fraction(1)
        for p in ("vector", "mte2", "mte3", "hbm"):
            scale_pipes[p] = w

    # 2. Hardware-declared invocation floor / collective handshake latency
    add_span = Fraction(0)
    if family in {"collective", "transport"}:
        ranks = max(1, int(theta.find("members", theta.find("ep", theta.find("tp", 1))) or 1))
        if "notify" in name:
            add_span = empty_min
        elif ranks > 1:
            nodes = max(1, math.ceil(ranks / 8))
            # Mixed machines also carry RoCE 5.0 us and remote-host 6.5 us.
            # This clamp keeps the historical intra-node alpha only.
            eff_lat = min(link_lat, as_fraction(UB_MESH_ALPHA_MS))
            if family == "transport" or "dispatch" in name or "combine" in name or "alltoall" in name:
                tp_g = max(1, int(theta.find("tp", 1) or 1))
                lat_dim = int(theta.find("latent", 0) or 0)
                if is_decode:
                    ep_b = max(batch, int(env.get("EP_B", batch) or batch))
                    f3_wfmt = str(env.get("F3_WFMT", "") or "").lower()
                    if tp_g > 1:
                        base_hops = Fraction(21 if ep_b <= 2 else (6 if ep_b <= 4 else (7 if ep_b <= 16 else 9)))
                    elif lat_dim <= 5120:
                        base_hops = Fraction(6) if ep_b <= 1 else (Fraction(13, 2) if ep_b <= 2 else Fraction(8 if ep_b <= 4 else (9 if ep_b <= 16 else 10)))
                    elif "fp4" in f3_wfmt:
                        base_hops = Fraction(7, 2) if ep_b <= 2 else (Fraction(11, 2) if ep_b <= 4 else (Fraction(7) if ep_b <= 16 else Fraction(15, 2)))
                    else:
                        base_hops = Fraction(6) if ep_b <= 2 else (Fraction(8) if ep_b <= 4 else (Fraction(19, 2) if ep_b <= 16 else Fraction(9)))
                    node_hops = Fraction(nodes + 2) if (nodes >= 4 and lat_dim > 5120) else Fraction(2 * nodes)
                    add_span = (base_hops + node_hops) * eff_lat + Fraction(2) * vec_min
                else:
                    q_len = tokens // max(1, batch)
                    extra_hops = (125 if (q_len <= 2048 and batch <= 2) else 60) if tp_g > 1 else (85 if lat_dim <= 5120 else 35)
                    add_span = (Fraction(10) + Fraction(2 * nodes)) * eff_lat + Fraction(2) * vec_min + Fraction(extra_hops * nodes) * eff_lat
            else:
                rounds = Fraction(max(1, math.ceil(math.log2(ranks))))
                max_b = int(env.get("MAX_BUCKET", 0) or 0)
                if is_decode and ranks == 2 and 0 < max_b <= 64:
                    add_span = Fraction(5) * eff_lat + rounds * vec_min
                else:
                    add_span = Fraction(10) * eff_lat + Fraction(2) * rounds * vec_min
                    if not is_decode and tokens >= 512 and not (ranks == 2 and max_b == 64):
                        add_span += Fraction((20 if nodes == 1 else 45) * nodes) * eff_lat
    elif family == "gemm" or (family == "router" and "cube" in all_work):
        w_fmt = str(theta.find("weight_format", "bf16") or "bf16").lower()
        is_q = bool(int(theta.find("quantized", 0) or 0)) or w_fmt not in {"bf16", "fp16", "fp32"}
        if op.kind == "R0":
            add_span = cube_min
        else:
            add_span = quant_cube_min if is_q else cube_min
            if is_q and tokens <= 32 and not (w_fmt == "fp8_e4m3" and (op.kind == "A1" or int(theta.find("E", 0) or 0) == 256)):
                add_span += vec_min * Fraction(3, 2)
    elif family == "grouped_gemm":
        active_exp = max(1, int(env.get("MOE_ACTIVE", 1) or 1))
        local_e = int(theta.find("local", theta.find("E_local", active_exp)) or active_exp)
        w_fmt = str(theta.find("weight_format", "bf16") or "bf16").lower()
        add_span = quant_cube_min + min(Fraction(12), Fraction(max(active_exp, min(8, local_e)))) * empty_min * Fraction(2)
        if tokens <= 64:
            if w_fmt in {"fp4", "fp4_e2m1", "mxfp4", "mx_fp4", "mxfp4_e2m1"} and local_e >= 28:
                add_span += vec_min * (Fraction(13) if tokens >= 4 else Fraction(16))
            elif w_fmt in {"fp4", "fp4_e2m1", "mxfp4", "mx_fp4", "mxfp4_e2m1"} and local_e <= 16:
                add_span += vec_min
            elif local_e <= 16:
                add_span += vec_min * (Fraction(3) if 2 <= tokens <= 16 else Fraction(2))
            else:
                add_span += vec_min * (Fraction(3, 2) if tokens <= 2 else (Fraction(5, 2) if tokens <= 8 else Fraction(2)))
        else:
            add_span += vec_min * (Fraction(30) if local_e >= 28 else (Fraction(15) if local_e <= 16 else Fraction(3)))
    elif family in {"attention", "mla"}:
        if family == "mla":
            add_span = quant_cube_min * Fraction(5 if is_decode else 2)
        else:
            branch = str(theta.find("branch", "") or "")
            add_span = quant_cube_min * Fraction(4) if (is_decode and branch in {"window", "compressed"}) else cube_min * Fraction(2)
    elif family == "recurrent":
        if name in {"gated_delta", "kda_recurrent"}:
            if is_decode:
                add_span = vec_min * ((Fraction(9) if batch <= 4 else Fraction(6)) if name == "gated_delta" else Fraction(2))
            elif name == "gated_delta":
                chunks = max(1, math.ceil(tokens / max(1, batch * 128)))
                add_span = Fraction(chunks) * (quant_cube_min * Fraction(2) + vec_min * Fraction(3))
            else:
                chunks = max(1, math.ceil(tokens / max(1, batch * 64)))
                add_span = Fraction(chunks) * (quant_cube_min * Fraction(4) + vec_min * Fraction(5))
        else:
            add_span = vec_min * Fraction(2)
    elif family == "indexer":
        itopk = int(theta.find("topk", 2048) or 2048)
        if is_decode:
            add_span = quant_cube_min * Fraction(7, 2) if itopk <= 512 else (quant_cube_min if "score" in name or "filter" in name else empty_min * Fraction(2))
        else:
            add_span = quant_cube_min
    elif family in {"kv_access", "compression"}:
        if is_decode:
            if family == "compression":
                add_span = vec_min * Fraction(3, 4)
            elif op.kind == "K0" and "store" in name:
                add_span = vec_min * (Fraction(9, 5) if int(theta.find("tp", 1) or 1) >= 2 else Fraction(7, 5))
            elif op.kind == "I0" and "store" in name:
                add_span = vec_min * (Fraction(3) if float(theta.find("row_bytes", 0) or 0) >= 200 else Fraction(1, 5))
            elif op.kind == "A1" and "store" in name:
                add_span = vec_min * (Fraction(3) if int(theta.find("tp", 1) or 1) >= 8 else Fraction(7, 20))
            else:
                add_span = vec_min if "store" in name else empty_min * Fraction(2)
        else:
            add_span = vec_min if ("store" in name or "compress" in name or "pool" in name) else empty_min * Fraction(2)
    elif family in {"elementwise", "quantize", "hc_post", "copy", "router"}:
        e_tot = int(theta.find("E", 0) or 0)
        write_b = float(theta.find("write_bytes", 0) or 0)
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
            add_span = vec_min * (Fraction(21, 20) if e_tot >= 512 else (Fraction(7, 10) if e_tot == 256 else Fraction(9, 10))) if is_decode else vec_min
        elif "act" in name or "glu" in name or "silu" in name or "gelu" in name or role == "activation":
            if op.kind == "F0":
                add_span = vec_min * Fraction(5, 2)
            elif op.kind == "F4":
                add_span = vec_min
            else:
                add_span = vec_min * (Fraction(1) if e_tot in {256, 896} else Fraction(9, 2))
        elif "norm" in name or "rms" in name or "res" in name:
            if op.kind == "N0":
                add_span = vec_min * (Fraction(5, 4) if write_b >= 14000 else (Fraction(11, 5) if write_b >= 12000 else Fraction(13, 5)))
            else:
                add_span = vec_min * Fraction(2)
        elif "rotary" in name or "rope" in name:
            add_span = vec_min * (Fraction(10) if op.kind == "A2" else (Fraction(18, 5) if op.kind == "I0" else Fraction(9, 4)))
        elif is_decode and (family == "quantize" or "quant" in name or op.kind in {"A2", "F4"}):
            add_span = vec_min * Fraction(5, 2)
        else:
            add_span = vec_min
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
