"""Turnkey ``mindie-sim report`` and ``mindie-sim quickstart`` CLI commands.

[Role] Deliver zero-boilerplate, cold-start Agent-friendly evaluation via:
  (1) ``mindie-sim report``: auto-solves HBM-safe parallelism & concurrency
      when omitted, supports blind-box HuggingFace ``config.json``/directories,
      defaults to chip-matched ``auto_va`` production quantization (QuaRot-first
      guidance), parses ``--from-script <serve.sh>`` statically without execution,
      and writes ``*_comparison.xlsx``, ``*_trace_view.json``, compact
      ``< 8 KB`` ``*_agent_brief.json``, optional ``*_optimized_serve.sh``, and
      a ~25-line stdout Markdown Executive Digest;
  (2) ``mindie-sim quickstart [--json]``: prints <45-line golden commands,
      hardware/quant aliases, QuaRot-first priority, and the 3-step Agent loop.
[Invariants] Never executes user launch scripts (pure ``shlex`` static parse);
  never weakens low-level ``compose_case()`` checkpoint defaults; never
  swallows ``TimeoutError`` under ``except OSError:``.
[Timeouts & Locks] Bounded 50 MiB metadata reads and single-pass O(Archetypes)
  export delegation; no locks or device contexts.
[Side-Effect Policy] Writes artifacts only into caller-specified ``--out-dir``
  (and optional ``--receipt`` path).
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import re
import shlex
from typing import Any

from mindie_sim.compose.inputs import (
    BYTES_PER_GIB,
    DERIVED_ASCEND_SKUS,
    _QUANT_ALIASES,
    _arch_key,
    _resolve_hardware,
    _resolve_model_spec,
)
from mindie_sim.export import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_TPOT_SLO_MS,
    build_export_bundle,
    compose_export_case,
    export_comparison_xlsx,
    export_msprof_trace,
)
from mindie_sim.export.comparison import (
    align_before_after_operators,
    build_macro_module_breakdown,
    build_pending_kernel_details,
)
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.schema.common import ModelingError

REPORT_COMMANDS = frozenset({"report", "quickstart"})
REPORT_BOOLEAN_FLAGS = frozenset({"--json", "--no-trace", "--no-xlsx"})
AGENT_BRIEF_SCHEMA = "mindie-sim.agent-brief/1"
MAX_SCRIPT_BYTES = 50 * 1024 * 1024
DEFAULT_REPORT_PROMPT_TOKENS = 32768
LONG_CONTEXT_KV_QUANT_THRESHOLD = 32768
AUTO_SIZING_WEIGHT_FRACTION = 0.65
AUTO_DEVICE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
TOKEN_SHORTHAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")
TOKEN_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024 * 1024}
SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")
ASSIGN_ENV_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
SCRIPT_QUANT_TOKENS = (
    ("mxfp8_mxfp4", "mxfp8_mxfp4"),
    ("mxfp4", "mxfp4"),
    ("mxfp8", "mxfp8"),
    ("w4a8", "w4a8"),
    ("w8a8", "w8a8"),
    ("fp8", "fp8"),
    ("bf16", "bf16"),
    ("fp16", "fp16"),
)


def parse_token_shorthand(raw: str | int) -> int:
    """Parse integer or ``32k`` / ``128k`` / ``1m`` token counts."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw <= 0:
            raise argparse.ArgumentTypeError(f"token count must be > 0, got {raw}")
        return raw
    cleaned = str(raw).strip().replace("_", "").replace(",", "")
    match = TOKEN_SHORTHAND_RE.match(cleaned)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid token count {raw!r}; use integer or shorthand like 32k / 128k"
        )
    value = int(round(float(match.group(1)) * TOKEN_MULTIPLIERS[match.group(2).lower()]))
    if value <= 0:
        raise argparse.ArgumentTypeError(f"token count must be > 0, got {raw!r}")
    return value


def _is_moe_model(model_spec: Any) -> bool:
    if int(getattr(model_spec, "routed_experts", 0) or 0) > 0:
        return True
    if int(getattr(model_spec, "num_experts", 0) or 0) > 1:
        return True
    for layer in getattr(model_spec, "layer_plans", ()) or ():
        if int(getattr(layer, "routed_experts", 0) or 0) > 0 or str(getattr(layer, "ffn", "")) == "moe":
            return True
    return False


def _lookup_quarot_checkpoint_hint(model_name: str, arch: str, weight_quant: str | None) -> str:
    """Return a QuaRot-first Eco-Tech / msmodelslim recommendation when available."""
    try:
        ckpt_mod = importlib.import_module("mindie_sim.catalog.npu_checkpoints")
        finder = getattr(ckpt_mod, "recommend_npu_checkpoint", None) or getattr(
            ckpt_mod, "get_recommended_checkpoint", None
        )
        if callable(finder):
            rec = finder(model_name, arch=arch, quant=weight_quant)
            if isinstance(rec, Mapping) and rec.get("checkpoint"):
                return str(rec["checkpoint"])
            if isinstance(rec, str) and rec:
                return rec
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    if arch == "a5":
        return (
            "Prefer QuaRot-calibrated Eco-Tech / msmodelslim MXFP8+MXFP4 checkpoint "
            "(Hadamard outlier rotation before MXFP8 attn/shared + MXFP4 routed experts; "
            "keep norm/router/embed/lm_head BF16)"
        )
    return (
        "Prefer *-QuaRot Eco-Tech checkpoint when available (higher accuracy than plain W8A8/W4A8; "
        "otherwise quantize via msmodelslim QuaRot + --quantization ascend)"
    )


def resolve_auto_va_policy(
    model: str | Path | Mapping[str, Any],
    hardware: str,
    *,
    context_tokens: int = DEFAULT_REPORT_PROMPT_TOKENS,
    weight_quant: str | None = "auto_va",
    cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Resolve chip-matched Ascend ``vllm-ascend`` production quantization defaults."""
    from mindie_sim.catalog.quant import QUANTIZATION_PRESETS

    model_spec = _resolve_model_spec(model)
    sku, _backend, preset = _resolve_hardware(hardware)
    arch = _arch_key(sku)
    raw_q = (str(weight_quant).strip().lower() if weight_quant is not None else "auto_va")
    explicit_cache = _QUANT_ALIASES.get(str(cache_dtype).strip().lower(), str(cache_dtype).strip().lower()) if cache_dtype else None

    if raw_q in {"checkpoint", "native"}:
        q_val: str | None = None
        kv_val = explicit_cache
        idx_val = None
        source = "checkpoint_native"
    elif raw_q in {"", "auto", "auto_va"}:
        quant_mod = importlib.import_module("mindie_sim.catalog.quant")
        hook = getattr(quant_mod, "resolve_auto_va_quant", None)
        if callable(hook):
            q_val, hook_kv, idx_val = hook(model_spec, sku, context_tokens=int(context_tokens))
            kv_val = explicit_cache or hook_kv
        else:
            is_moe = _is_moe_model(model_spec)
            long_ctx = int(context_tokens) >= LONG_CONTEXT_KV_QUANT_THRESHOLD
            if arch == "a5":
                if is_moe:
                    q_val = "mxfp8_mxfp4" if "mxfp8_mxfp4" in QUANTIZATION_PRESETS else "mxfp4"
                    kv_val = explicit_cache or "fp8_e4m3"
                    idx_val = "fp8_e4m3"
                else:
                    q_val = "mxfp8"
                    kv_val = explicit_cache or ("fp8_e4m3" if long_ctx else "bf16")
                    idx_val = "fp8_e4m3"
            elif arch in {"a2", "a3"}:
                q_val = "w4a8" if is_moe else "w8a8"
                kv_val = explicit_cache or ("int8" if long_ctx else "bf16")
                idx_val = None
            else:
                q_val = "fp8" if is_moe else None
                kv_val = explicit_cache or ("fp8_e4m3" if long_ctx else None)
                idx_val = None
        source = "auto_va_production"
    else:
        q_val = _QUANT_ALIASES.get(raw_q, raw_q)
        kv_val = explicit_cache
        idx_val = None
        source = "explicit_flag"

    return {
        "model_spec": model_spec,
        "sku": sku,
        "arch": arch,
        "topology_preset": preset or None,
        "weight_quant": q_val,
        "kv_cache_dtype": kv_val,
        "indexer_kv_dtype": idx_val,
        "policy_source": source,
        "quarot_guidance": _lookup_quarot_checkpoint_hint(model_spec.name, arch, q_val),
    }


def _infer_model_and_quant_from_arg(model_arg: str) -> tuple[str, str | None]:
    """Infer catalog model + quant from a local dir or basename token."""
    raw = str(model_arg or "").strip()
    path = Path(raw).expanduser()
    if path.exists():
        quant_hint = None
        qdesc = (path if path.is_dir() else path.parent) / "quant_model_description.json"
        if qdesc.is_file() and qdesc.stat().st_size <= MAX_SCRIPT_BYTES:
            try:
                raw_q = str(json.loads(qdesc.read_text(encoding="utf-8")).get("model_quant_type", "")).lower()
                for tok, qname in SCRIPT_QUANT_TOKENS:
                    if tok in raw_q:
                        quant_hint = qname
                        break
            except (OSError, ValueError):
                pass
        return str(path), quant_hint

    lower = Path(raw).name.lower()
    quant_hint = next((qname for tok, qname in SCRIPT_QUANT_TOKENS if tok in lower), None)
    cleaned = re.sub(r"[-_](?:quarot|mtp|w4a8|w8a8|mxfp8|mxfp4|fp8|bf16|fp16|int8|c8|instruct|chat|base)\b", "", lower)
    from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES
    pool = sorted(set(CATALOG) | set(MODEL_ALIASES), key=len, reverse=True)
    for key in pool:
        if key in lower or key in cleaned:
            return MODEL_ALIASES.get(key, key), quant_hint
    return raw, quant_hint


def _parse_static_script_fallback(path: Path) -> dict[str, Any]:
    """Pure ``shlex`` zero-exec fallback parser for ``serve.sh`` launch scripts."""
    if not path.is_file():
        raise FileNotFoundError(f"launch_script_not_found:{path}")
    if path.stat().st_size > MAX_SCRIPT_BYTES:
        raise ModelingError("launch_script_exceeds_50_mib_limit")
    text = path.read_text(encoding="utf-8")
    logical_lines: list[str] = []
    buf = ""
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
        else:
            logical_lines.append((buf + stripped).strip())
            buf = ""
    if buf.strip():
        logical_lines.append(buf.strip())

    env: dict[str, str] = {}
    cmd_tokens: list[str] = []
    for line in logical_lines:
        match = ASSIGN_ENV_RE.match(line)
        if match and " " not in match.group(2).strip().split(";")[0]:
            env[match.group(1)] = match.group(2).strip().strip("'\"")
            continue
        if any(k in line for k in ("vllm serve", "api_server", "mindie", "--tensor-parallel", "-tp ")):
            cmd_tokens = shlex.split(line)

    flags: dict[str, Any] = {}
    model_arg = None
    idx = 0
    while idx < len(cmd_tokens):
        tok = cmd_tokens[idx]
        if "=" in tok and not tok.startswith("-"):
            k, v = tok.split("=", 1)
            if k.isidentifier():
                env[k] = v
            idx += 1
            continue
        if tok == "serve" and idx + 1 < len(cmd_tokens) and not cmd_tokens[idx + 1].startswith("-"):
            model_arg = cmd_tokens[idx + 1]
            idx += 2
            continue
        if tok in {"--enable-expert-parallel", "--enforce-eager"}:
            flags[tok] = True
            idx += 1
            continue
        if tok.startswith("-"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                flags[k] = v
                idx += 1
            elif idx + 1 < len(cmd_tokens) and not cmd_tokens[idx + 1].startswith("-"):
                flags[tok] = cmd_tokens[idx + 1]
                idx += 2
            else:
                flags[tok] = True
                idx += 1
            continue
        idx += 1

    model_raw = str(flags.get("--model") or model_arg or "")
    inferred_model, inferred_quant = _infer_model_and_quant_from_arg(model_raw) if model_raw else ("", None)
    vis = env.get("ASCEND_RT_VISIBLE_DEVICES", "")
    vis_count = len([x for x in vis.split(",") if x.strip()]) if vis else None
    tp = int(flags.get("-tp") or flags.get("--tensor-parallel-size") or 1)
    dp = int(flags.get("-dp") or flags.get("--data-parallel-size") or 1)
    pp = int(flags.get("-pp") or flags.get("--pipeline-parallel-size") or 1)
    dcp = int(flags.get("--dcp-size") or flags.get("--cp-size") or 1)
    ep = (dp * tp) if flags.get("--enable-expert-parallel") else 1
    num_devices = vis_count or (dp * tp * pp)
    issues: list[str] = []
    if flags.get("--enforce-eager"):
        issues.append("--enforce-eager disables ACLGraph capture (high host launch overhead)")
    if env.get("HCCL_OP_EXPANSION_MODE", "").upper() not in {"AIV", "CCU"}:
        issues.append("HCCL_OP_EXPANSION_MODE not set to AIV/CCU (~90us AI_CPU collective startup penalty)")
    if not flags.get("--quantization") and (inferred_quant in {None, "bf16", "fp16"}):
        issues.append("Unquantized weights detected — switch to chip-matched QuaRot auto_va quant (--quantization ascend)")
    if env.get("VLLM_ASCEND_ENABLE_MLAPO") != "1":
        issues.append("VLLM_ASCEND_ENABLE_MLAPO=1 not enabled (MLA Prolog fusion disabled)")

    return {
        "script_path": str(path),
        "env": env,
        "flags": flags,
        "model": inferred_model or None,
        "weight_quant": inferred_quant or ("bf16" if not flags.get("--quantization") else None),
        "cache_dtype": flags.get("--kv-cache-dtype"),
        "dp": dp,
        "tp": tp,
        "ep": ep,
        "pp": pp,
        "dcp": dcp,
        "num_devices": num_devices,
        "max_model_len": int(flags["--max-model-len"]) if "--max-model-len" in flags else None,
        "concurrency": int(flags["--max-num-seqs"]) if "--max-num-seqs" in flags else None,
        "enforce_eager": bool(flags.get("--enforce-eager", False)),
        "issues": issues,
    }


def parse_script_for_report(path: str | Path) -> dict[str, Any]:
    """Delegate launch-script parsing to ``mindie_sim.export.parse_launch_script`` when present."""
    target = Path(path).expanduser()
    try:
        export_mod = importlib.import_module("mindie_sim.export")
        hook = getattr(export_mod, "parse_launch_script", None)
        if callable(hook):
            parsed = hook(target)
            if isinstance(parsed, Mapping):
                return dict(parsed)
            if hasattr(parsed, "to_dict"):
                return dict(parsed.to_dict())
    except (ModuleNotFoundError, AttributeError):
        pass
    return _parse_static_script_fallback(target)


def _estimate_auto_num_devices(model_spec: Any, sku: str, weight_quant: str | None) -> int:
    """Pick the smallest power-of-2 die count that safely fits quantized weights."""
    from mindie_sim.backends.ascend import profiles as ascend_profiles
    from mindie_sim.backends.nvidia import profiles as nvidia_profiles
    from mindie_sim.catalog.weights import model_weight_bytes

    reg = {**ascend_profiles(), **DERIVED_ASCEND_SKUS, **nvidia_profiles()}
    entry = reg.get(sku, {})
    mem_b = float(entry.get("memory_bytes") or (entry.get("device") or {}).get("memory_bytes") or 64 * BYTES_PER_GIB)
    bytes_per_param = 0.7 if (weight_quant and "4" in weight_quant) else (1.1 if weight_quant in {"w8a8", "mxfp8", "fp8"} else 2.0)
    try:
        total_w = float(model_weight_bytes(model_spec)) * (bytes_per_param / 2.0)
    except Exception:
        total_w = 16.0 * BYTES_PER_GIB
    target_per_die = mem_b * AUTO_SIZING_WEIGHT_FRACTION
    min_dies = max(1, int((total_w + target_per_die - 1.0) // target_per_die))
    for count in AUTO_DEVICE_GRID:
        if count >= min_dies:
            return count
    return AUTO_DEVICE_GRID[-1]


def _auto_solve_report_layout(
    *,
    model: str,
    sku: str,
    topology_preset: str | None,
    num_devices: int,
    prompt_tokens: int,
    output_tokens: int,
    tpot_slo_ms: float,
    ttft_slo_ms: float | None,
    quant: str | None,
    kv_cache_dtype: str | None,
    indexer_kv_dtype: str | None,
    deployment_mode: str,
) -> dict[str, Any]:
    """Auto-solve HBM-safe parallelism + concurrency via fast ``solve_optimal_deployment``."""
    from mindie_sim.sweep.pd import solve_optimal_deployment

    dep = "colocated" if deployment_mode == "colocated" else (
        "pd_disaggregated" if deployment_mode.startswith("pd_") else "auto"
    )
    solved = solve_optimal_deployment(
        model,
        sku,
        num_devices=int(num_devices),
        prompt_tokens=int(prompt_tokens),
        output_tokens=int(output_tokens),
        ttft_sla_ms=ttft_slo_ms,
        tpot_sla_ms=float(tpot_slo_ms),
        deployment=dep,
        quant=quant,
        kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset,
        allow_unconstrained_ttft=(ttft_slo_ms is None),
    )
    best = solved.get("best_configuration")
    if best is None:
        cands = [c for c in solved.get("candidates", ()) if isinstance(c, Mapping) and int(c.get("b_vram", 0) or 0) >= 1]
        if cands:
            best = max(cands, key=lambda c: (int(c.get("b_vram", 0) or 0), -float(c.get("effective_tpot_ms", 1e9) or 1e9)))
    par = (best or {}).get("decode_parallel") or (best or {}).get("parallel") or {}
    return {
        "dp": int(par.get("dp", 1) or 1),
        "tp": int(par.get("tp", min(num_devices, 8)) or 1),
        "ep": int(par.get("ep", par.get("dp", 1) * par.get("tp", 1)) or 1),
        "pp": int(par.get("pp", 1) or 1),
        "dcp": int(par.get("cp", par.get("dcp", 1)) or 1),
        "pcp": int(par.get("pcp", 1) or 1),
        "concurrency": max(1, int((best or {}).get("b_feasible") or (best or {}).get("b_vram") or 1)),
        "spec_tokens": int((best or {}).get("mtp_proposal_tokens", 0) or 0),
        "deployment_mode": "pd_unpooled" if solved.get("winner_mode") == "pd_disaggregated" else "colocated",
        "solver_feasible": bool(solved.get("feasible", False)),
    }


def _build_fallback_agent_brief(
    before_bundle: Any,
    after_bundle: Any,
    *,
    policy: Mapping[str, Any],
    generated_files: Mapping[str, str],
    script_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact (<8 KB) structured brief when ``export.export_agent_brief`` is not yet merged."""
    led = after_bundle.memory_ledger
    aligned = align_before_after_operators(before_bundle, after_bundle)
    macros = build_macro_module_breakdown(before_bundle, after_bundle, aligned)
    pending = build_pending_kernel_details(before_bundle, after_bundle, aligned)
    b_step_ms = round(before_bundle.step_span_us / 1000.0, 3)
    a_step_ms = round(after_bundle.step_span_us / 1000.0, 3)
    a_lstar_ms = round(after_bundle.step_l_star_lean_us / 1000.0, 3)
    b_tpot = round(b_step_ms / max(1.0, before_bundle.avg_accepted_tokens), 3)
    a_tpot = round(a_step_ms / max(1.0, after_bundle.avg_accepted_tokens), 3)
    arch = str(policy.get("arch", ""))
    hccl_mode = "CCU" if arch == "a5" else "AIV"
    kernel_tasks = [
        {
            "card_id": f"Card #{item.priority}",
            "kernel_key": item.kernel_key,
            "target_kernel": item.target_kernel,
            "category": item.category,
            "backend": item.suggested_backend,
            "expected_gain_ms": round(float(item.expected_gain_ms), 3),
            "gap_vs_lstar_ms": round(float(item.gap_vs_lstar_ms), 3),
            "tensor_contract": item.tensor_contract[:140],
        }
        for item in pending[:6]
    ]
    override_example = (
        {
            pending[0].kernel_key: {
                "config_howto": f"export HCCL_OP_EXPANSION_MODE={hccl_mode} VLLM_ASCEND_ENABLE_MLAPO=1",
                "kernel_design": "Replace with repo-grounded AscendC/Triton tiling notes",
                "framework_work": "Wire vllm-ascend graph pass / fusion flag",
                "acceptance": "BF16/FP8 cosine >= 0.999, msprof replay within 10%",
            }
        }
        if pending else {}
    )
    return {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": {
            "model": after_bundle.model_name,
            "hardware_sku": after_bundle.hardware_sku,
            "phase": after_bundle.phase,
            "dp": led.dp,
            "tp": led.tp,
            "ep": led.ep,
            "dcp": led.dcp,
            "pp": led.pp,
            "pcp": led.pcp,
            "total_dies": led.total_dies,
            "prompt_tokens": after_bundle.prompt_tokens,
            "output_tokens": after_bundle.output_tokens,
            "concurrency": after_bundle.sequences,
            "weight_quant": led.weight_quant_display or led.weight_quant,
            "cache_dtype": led.cache_dtype,
            "quantized_weight_gib_per_die": round(led.weight_footprint_per_die_bytes / BYTES_PER_GIB, 2),
            "max_hbm_concurrency_per_dp": int(led.max_hbm_concurrency_per_dp),
            "fits_hbm": bool(led.fits_hbm),
            "quarot_guidance": str(policy.get("quarot_guidance", "")),
        },
        "kpis_before_vs_after": {
            "before_track": before_bundle.feature_track,
            "after_track": after_bundle.feature_track,
            "before_step_ms": b_step_ms,
            "after_step_ms": a_step_ms,
            "speedup_x": round(b_step_ms / a_step_ms, 3) if a_step_ms > 0 else 1.0,
            "before_effective_tpot_ms": b_tpot,
            "after_effective_tpot_ms": a_tpot,
            "tpot_slo_ms": float(after_bundle.tpot_slo_ms),
            "sla_verdict": "PASS" if a_tpot <= float(after_bundle.tpot_slo_ms) else "EXCEEDS_SLO",
            "single_die_tps_before": round(before_bundle.throughput_per_die_tokens_per_s, 2),
            "single_die_tps_after": round(after_bundle.throughput_per_die_tokens_per_s, 2),
            "cluster_tps_before": round(before_bundle.throughput_tokens_per_s, 2),
            "cluster_tps_after": round(after_bundle.throughput_tokens_per_s, 2),
            "l_star_ms": a_lstar_ms,
            "l_star_attainment_pct": round(100.0 * a_lstar_ms / a_step_ms, 1) if a_step_ms > 0 else 0.0,
            "cube_util_pct": round(100.0 * after_bundle.network_cube_mfu, 1),
            "vector_util_pct": round(100.0 * after_bundle.network_vec_mfu, 1),
            "hbm_mbu_pct": round(100.0 * after_bundle.network_mbu, 1),
            "hccl_link_util_pct": round(100.0 * after_bundle.network_link_util, 1),
        },
        "module_hierarchy": [
            {
                "module": row.module_name,
                "before_ms": round(row.before_ms, 3),
                "after_ms": round(row.after_ms, 3),
                "saved_ms": round(row.saved_ms, 3),
                "after_share_pct": round(row.after_share_pct, 1),
                "bound": row.dominant_bound,
            }
            for row in macros[:8]
        ],
        "action_plan_3_tracks": {
            "config_and_env": {
                "env": [
                    f"export HCCL_OP_EXPANSION_MODE={hccl_mode}",
                    "export VLLM_ASCEND_ENABLE_MLAPO=1",
                    "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
                ],
                "quantization_recipe": str(policy.get("quarot_guidance", "")),
                "vllm_flags": (
                    f"--tensor-parallel-size {led.tp} --data-parallel-size {led.dp} "
                    f"--pipeline-parallel-size {led.pp} --quantization ascend "
                    f"--kv-cache-dtype {led.cache_dtype.lower()}"
                ),
            },
            "framework_tasks": [
                {"module": row.module_name, "saved_ms": round(row.saved_ms, 3)}
                for row in macros if row.saved_ms > 0.05
            ][:5],
            "kernel_tasks": kernel_tasks,
            "kernel_designs_override_example": override_example,
        },
        "script_audit": dict(script_audit) if script_audit else None,
        "generated_files": dict(generated_files),
    }


def _format_fallback_digest_markdown(brief: Mapping[str, Any]) -> str:
    """Format a ~22-line executive Markdown digest for stdout."""
    sc = brief["scenario_and_parallelism"]
    kp = brief["kpis_before_vs_after"]
    files = brief["generated_files"]
    lines = [
        f"## MindIE-Sim Report: `{sc['model']}` @ `{sc['hardware_sku']}` ({sc['phase']})",
        f"- **Parallelism & Memory**: DP{sc['dp']}·TP{sc['tp']}·EP{sc['ep']}·DCP{sc['dcp']}·PP{sc['pp']} "
        f"({sc['total_dies']} dies), B={sc['concurrency']}/DP, ctx={sc['prompt_tokens']}, "
        f"quant=`{sc['weight_quant']}` / KV=`{sc['cache_dtype']}` "
        f"({sc['quantized_weight_gib_per_die']} GiB/die, max HBM B={sc['max_hbm_concurrency_per_dp']})",
        f"- **KPIs (`{kp['before_track']}` -> `{kp['after_track']}`)**: "
        f"Step `{kp['before_step_ms']} ms -> {kp['after_step_ms']} ms` (**{kp['speedup_x']}x**), "
        f"TPOT `{kp['before_effective_tpot_ms']} -> {kp['after_effective_tpot_ms']} ms` "
        f"(SLO `{kp['tpot_slo_ms']} ms`: **{kp['sla_verdict']}**), "
        f"Cluster `{kp['cluster_tps_before']} -> {kp['cluster_tps_after']} tok/s` "
        f"(`{kp['single_die_tps_after']} tok/s/die`)",
        f"- **Hardware Roofline**: L*=`{kp['l_star_ms']} ms` ({kp['l_star_attainment_pct']}% attainment) | "
        f"Cube `{kp['cube_util_pct']}%` · Vec `{kp['vector_util_pct']}%` · MBU `{kp['hbm_mbu_pct']}%` · Link `{kp['hccl_link_util_pct']}%`",
        f"- **Quantization (QuaRot-First)**: {sc['quarot_guidance']}",
        "",
        "### Top Bottleneck Modules & Action Tracks",
    ]
    for row in brief.get("module_hierarchy", ())[:5]:
        lines.append(
            f"- `{row['module']}`: `{row['before_ms']} -> {row['after_ms']} ms` "
            f"(saved `{row['saved_ms']} ms`, share `{row['after_share_pct']}%`, `{row['bound']}`)"
        )
    env_cmds = "; ".join(brief["action_plan_3_tracks"]["config_and_env"]["env"])
    lines.append(f"- **Track 1 (Config/Env)**: `{env_cmds}`")
    if brief.get("script_audit") and brief["script_audit"].get("issues"):
        lines.append("- **Script Findings**: " + "; ".join(brief["script_audit"]["issues"][:3]))
    lines.append("")
    lines.append("### Generated Artifacts")
    for label, fpath in files.items():
        lines.append(f"- **{label}**: `{fpath}`")
    return "\n".join(lines) + "\n"


def _write_optimized_serve_script(
    out_path: Path,
    *,
    before_bundle: Any,
    after_bundle: Any,
    script_audit: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> str:
    """Write ``<prefix>_optimized_serve.sh`` via Wave-8 hook or clean fallback."""
    try:
        export_mod = importlib.import_module("mindie_sim.export")
        hook = getattr(export_mod, "render_optimized_serve_sh", None)
        if callable(hook):
            rendered = hook(script_audit, after_bundle, before_bundle=before_bundle)
            out_path.write_bytes((str(rendered).rstrip("\n") + "\n").encode("utf-8"))
            return str(out_path)
    except (ModuleNotFoundError, AttributeError, TypeError):
        pass
    led = after_bundle.memory_ledger
    hccl_mode = "CCU" if str(policy.get("arch", "")) == "a5" else "AIV"
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by mindie-sim report --from-script (QuaRot-first Ascend vllm-ascend tuned launch)",
        f"# Quantization tip: {policy.get('quarot_guidance', '')}",
        "set -euo pipefail",
        f"export HCCL_OP_EXPANSION_MODE={hccl_mode}",
        "export VLLM_ASCEND_ENABLE_MLAPO=1",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
        f"vllm serve {after_bundle.model_name} \\",
        f"  --tensor-parallel-size {led.tp} \\",
        f"  --data-parallel-size {led.dp} \\",
        f"  --pipeline-parallel-size {led.pp} \\",
        *(["  --enable-expert-parallel \\"] if led.ep > 1 else []),
        *(["  --dcp-size " + str(led.dcp) + " \\"] if led.dcp > 1 else []),
        "  --quantization ascend \\",
        f"  --kv-cache-dtype {led.cache_dtype.lower()} \\",
        f"  --max-model-len {after_bundle.prompt_tokens + after_bundle.output_tokens} \\",
        f"  --max-num-seqs {after_bundle.sequences}",
    ]
    out_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return str(out_path)


def add_report_commands(subparsers) -> None:
    """Register turnkey ``report`` and ``quickstart`` commands."""
    rep = subparsers.add_parser(
        "report",
        help="turnkey one-shot evaluation: auto-solve parallelism/quant -> .xlsx + trace_view.json + agent_brief.json",
    )
    rep.add_argument("--model", default=None, help="catalog model, alias, HuggingFace config.json, or model directory")
    rep.add_argument("--from-script", default=None, help="audit & optimize an existing serve.sh launch script (zero-exec)")
    rep.add_argument("--hardware", default="a3", help="hardware SKU or alias (e.g. 950dt, a5, a3, 910c, a2, 910b)")
    rep.add_argument("--topology-preset", default=None, help="optional topology preset override (e.g. a5_pod, a3_16p)")
    rep.add_argument("--num-devices", type=int, default=None, help="total NPU dies (auto-sized when omitted)")
    rep.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    for flag in ("--dp", "--tp", "--dcp", "--ep", "--pp", "--pcp"):
        rep.add_argument(flag, type=int, default=None)
    rep.add_argument(
        "--prompt-tokens", "--context", dest="prompt_tokens", type=parse_token_shorthand, default=None,
        help="input prompt/context length (accepts shorthand like 32k, 128k; default 32k)",
    )
    rep.add_argument("--output-tokens", type=parse_token_shorthand, default=DEFAULT_SCENARIO_OUTPUT_TOKENS)
    rep.add_argument("--concurrency", type=int, default=None, help="sequences per DP replica (auto-solved when omitted)")
    rep.add_argument("--tpot-ms", type=float, default=DEFAULT_TPOT_SLO_MS)
    rep.add_argument("--ttft-ms", type=float, default=None)
    rep.add_argument("--spec-tokens", "--mtp", dest="spec_tokens", type=int, default=None)
    rep.add_argument("--accept-len", type=float, default=None)
    rep.add_argument("--compare-track", default="unmerged_pr", help="Before track (default: unmerged_pr)")
    rep.add_argument("--target-track", default="expected_ideal", help="After track (default: expected_ideal)")
    rep.add_argument(
        "--weight-quant", default="auto_va",
        help="weight quant preset (default: auto_va chip-matched QuaRot production quant; or checkpoint/bf16/w4a8/mxfp4)",
    )
    rep.add_argument("--cache-dtype", default=None, help="KV cache dtype override (bf16/c8/int8/fp8_e4m3)")
    rep.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    rep.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")
    rep.add_argument(
        "--deployment-mode", choices=("auto", "pd_unpooled", "pd_pooled", "colocated"), default="auto",
    )
    rep.add_argument("--kernel-designs-json", default=None, help="Pass-2 Agent design JSON keyed by Sheet 4 card_id/kernel_key")
    rep.add_argument("--observations", "--kernel-csv", dest="observations", default=None)
    rep.add_argument("--out-dir", default="./mindie_sim_report", help="output directory for generated artifacts")
    rep.add_argument("--prefix", default=None, help="optional filename prefix inside --out-dir")
    rep.add_argument("--no-xlsx", action="store_true")
    rep.add_argument("--no-trace", action="store_true")
    rep.add_argument("--no-pipe-lanes", dest="include_pipe_lanes", action="store_false")
    rep.add_argument("--no-counters", dest="include_counters", action="store_false")
    rep.add_argument("--no-overlap", dest="include_overlap", action="store_false")
    rep.add_argument("--include-submission", action="store_true")
    rep.add_argument("--json", dest="emit_json", action="store_true", help="emit structured JSON instead of Markdown digest")
    rep.add_argument("--receipt", dest="output", default=None, help="optional JSON path to save structured receipt")

    qs = subparsers.add_parser(
        "quickstart",
        help="print <45-line golden commands, QuaRot quant aliases, and the 3-step Agent loop",
    )
    qs.add_argument("--json", dest="emit_json", action="store_true", help="emit machine-readable quickstart JSON")
    qs.add_argument("--output", "-o", default=None)


def _emit_report_artifacts(
    *,
    before_bundle: Any,
    after_bundle: Any,
    args: argparse.Namespace,
    policy: Mapping[str, Any],
    script_audit: Mapping[str, Any] | None,
    load_designs: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = args.prefix or f"{after_bundle.model_name}_{after_bundle.hardware_sku}_{after_bundle.phase}"
    prefix = SAFE_PREFIX_RE.sub("_", raw_prefix).strip("_") or "mindie_sim_report"

    generated: dict[str, str] = {}
    summary: dict[str, Any] = {}
    if not getattr(args, "no_xlsx", False):
        xlsx_path = out_dir / f"{prefix}_comparison.xlsx"
        xlsx_sum = export_comparison_xlsx(
            before_bundle, xlsx_path, compare_source=after_bundle,
            agent_kernel_designs=load_designs(getattr(args, "kernel_designs_json", None)),
        )
        generated["comparison_xlsx"] = str(xlsx_path)
        summary["xlsx"] = xlsx_sum.to_dict()
    if not getattr(args, "no_trace", False):
        trace_path = out_dir / f"{prefix}_trace_view.json"
        trace_sum = export_msprof_trace(
            after_bundle, trace_path,
            include_pipe_lanes=bool(getattr(args, "include_pipe_lanes", True)),
            include_counters=bool(getattr(args, "include_counters", True)),
            include_overlap=bool(getattr(args, "include_overlap", True)),
        )
        generated["trace_view_json"] = str(trace_path)
        summary["trace"] = trace_sum.to_dict()
    if script_audit:
        generated["optimized_serve_sh"] = _write_optimized_serve_script(
            out_dir / f"{prefix}_optimized_serve.sh",
            before_bundle=before_bundle, after_bundle=after_bundle,
            script_audit=script_audit, policy=policy,
        )

    brief_path = out_dir / f"{prefix}_agent_brief.json"
    generated["agent_brief_json"] = str(brief_path)
    export_mod = importlib.import_module("mindie_sim.export")
    brief_hook = getattr(export_mod, "export_agent_brief", None)
    if callable(brief_hook):
        brief = brief_hook(
            before_bundle, after_bundle, brief_path,
            generated_files=generated, script_audit=script_audit,
        )
        brief_dict = dict(brief.to_dict()) if hasattr(brief, "to_dict") else dict(brief)
    else:
        brief_dict = _build_fallback_agent_brief(
            before_bundle, after_bundle, policy=policy,
            generated_files=generated, script_audit=script_audit,
        )
        brief_path.write_bytes((json.dumps(brief_dict, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    digest_hook = getattr(export_mod, "format_executive_digest_markdown", None)
    digest_md = str(digest_hook(brief_dict)) if callable(digest_hook) else _format_fallback_digest_markdown(brief_dict)
    return summary, brief_dict, digest_md


def run_report_command(args: argparse.Namespace, *, load_designs: Any) -> dict[str, Any]:
    """Execute ``mindie-sim report`` with auto-parallelism, ``auto_va``, and ``--from-script``."""
    script_audit = parse_script_for_report(args.from_script) if getattr(args, "from_script", None) else None
    model_arg = args.model or (script_audit.get("model") if script_audit else None)
    if not model_arg:
        raise ModelingError(
            "report_requires_model — pass --model <catalog|config.json|hf_dir> or --from-script <serve.sh>"
        )
    prompt_tokens = int(
        args.prompt_tokens
        or ((script_audit or {}).get("max_model_len") or DEFAULT_REPORT_PROMPT_TOKENS)
    )
    output_tokens = int(args.output_tokens or DEFAULT_SCENARIO_OUTPUT_TOKENS)
    policy = resolve_auto_va_policy(
        model_arg, args.hardware, context_tokens=prompt_tokens,
        weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
    )
    topo_preset = args.topology_preset or policy["topology_preset"]
    explicit_parallel = any(getattr(args, k, None) is not None for k in ("dp", "tp", "ep", "pp", "dcp", "pcp"))

    if explicit_parallel:
        dp = int(args.dp or 1)
        tp = int(args.tp or 1)
        ep = int(args.ep or 1)
        pp = int(args.pp or 1)
        dcp = int(args.dcp or 1)
        pcp = int(args.pcp or 1)
        conc = int(args.concurrency or 1)
        spec_toks = int(args.spec_tokens or 0)
        dep_mode = str(args.deployment_mode or "auto")
    else:
        n_dev = int(
            args.num_devices
            or (script_audit.get("num_devices") if script_audit else None)
            or _estimate_auto_num_devices(policy["model_spec"], policy["sku"], policy["weight_quant"])
        )
        solved = _auto_solve_report_layout(
            model=model_arg, sku=policy["sku"], topology_preset=topo_preset,
            num_devices=n_dev, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=args.ttft_ms,
            quant=policy["weight_quant"], kv_cache_dtype=policy["kv_cache_dtype"],
            indexer_kv_dtype=policy["indexer_kv_dtype"], deployment_mode=str(args.deployment_mode or "auto"),
        )
        dp, tp, ep, pp = solved["dp"], solved["tp"], solved["ep"], solved["pp"]
        dcp, pcp = solved["dcp"], solved["pcp"]
        conc = int(args.concurrency or solved["concurrency"])
        spec_toks = int(args.spec_tokens if args.spec_tokens is not None else solved["spec_tokens"])
        dep_mode = solved["deployment_mode"] if args.deployment_mode == "auto" else str(args.deployment_mode)

    accept_len = float(args.accept_len if args.accept_len is not None else (1.0 + 0.7 * spec_toks))
    before_quant = (
        script_audit.get("weight_quant")
        if (script_audit and script_audit.get("weight_quant"))
        else policy["weight_quant"]
    )
    before_kv = (
        script_audit.get("cache_dtype")
        if (script_audit and script_audit.get("cache_dtype"))
        else policy["kv_cache_dtype"]
    )
    after_quant = policy["weight_quant"]
    after_kv = policy["kv_cache_dtype"]
    if str(args.weight_quant or "").strip().lower() in {"bf16", "checkpoint", "native"}:
        after_policy = resolve_auto_va_policy(model_arg, args.hardware, context_tokens=prompt_tokens, weight_quant="auto_va")
        after_quant, after_kv = after_policy["weight_quant"], after_policy["kv_cache_dtype"]

    before_par = (
        dict(dp=script_audit["dp"], tp=script_audit["tp"], ep=script_audit["ep"],
             pp=script_audit["pp"], dcp=script_audit["dcp"], pcp=1)
        if script_audit else dict(dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp, pcp=pcp)
    )
    before_track = "raw_eager" if (script_audit and script_audit.get("enforce_eager")) else str(args.compare_track)
    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=topo_preset, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=float(args.ttft_ms or 5000.0),
        mtp_proposal_tokens=spec_toks, mtp_acceptance_length=accept_len,
        engram_cpu_offload=args.engram_offload, deployment_mode=dep_mode,
    )
    raw_before, req_b = compose_export_case(
        model_arg, policy["sku"], feature_track=before_track,
        concurrency=int((script_audit or {}).get("concurrency") or conc),
        quant=before_quant, kv_cache_dtype=before_kv, **before_par, **common_kw,
    )
    raw_after, req_a = compose_export_case(
        model_arg, policy["sku"], feature_track=str(args.target_track),
        concurrency=conc, quant=after_quant, kv_cache_dtype=after_kv,
        dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp, pcp=pcp, **common_kw,
    )
    step_b = StepRequest(
        phase=args.phase, tokens=req_b.tokens if args.phase == "decode" else prompt_tokens,
        context=req_b.context if args.phase == "decode" else 0, sequences=req_b.sequences, scope="model",
    )
    step_a = StepRequest(
        phase=args.phase, tokens=req_a.tokens if args.phase == "decode" else prompt_tokens,
        context=req_a.context if args.phase == "decode" else 0, sequences=req_a.sequences, scope="model",
    )
    before_bundle = build_export_bundle(
        raw_before, request=step_b, replay=replay, observations_path=args.observations,
        weight_quant=before_quant, cache_dtype=before_kv, engram_host_offload=args.engram_offload,
    )
    after_bundle = build_export_bundle(
        raw_after, request=step_a, replay=replay,
        weight_quant=after_quant, cache_dtype=after_kv, engram_host_offload=args.engram_offload,
    )
    summary, brief_dict, digest_md = _emit_report_artifacts(
        before_bundle=before_bundle, after_bundle=after_bundle,
        args=args, policy=policy, script_audit=script_audit, load_designs=load_designs,
    )
    receipt: dict[str, Any] = {
        "ok": True,
        "command": "report",
        "model_name": after_bundle.model_name,
        "hardware_sku": after_bundle.hardware_sku,
        "phase": after_bundle.phase,
        "agent_brief": brief_dict,
        **summary,
    }
    if not getattr(args, "emit_json", False):
        receipt["_stdout_markdown"] = digest_md
    return receipt


def run_quickstart_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return <45-line turnkey quickstart guide (Markdown by default, or ``--json``)."""
    payload: dict[str, Any] = {
        "schema": "mindie-sim.quickstart/1",
        "golden_commands": [
            {
                "name": "zero_config_report",
                "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out",
            },
            {
                "name": "audit_launch_script",
                "cmd": "mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out",
            },
            {
                "name": "blind_box_hf_config",
                "cmd": "mindie-sim report --model ./my_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out",
            },
            {
                "name": "pinned_with_msprof",
                "cmd": "mindie-sim report --model qwen3.8-flash-next --hardware 950dt --dp 4 --tp 2 --ep 8 --concurrency 64 --kernel-csv ./kernel_details.csv --out-dir ./out",
            },
        ],
        "hardware_aliases": {
            "a5": "Ascend950DT_9582 (aliases: 950dt, 950dt-96g-pod, 950dt-144g-pod, a5-pod)",
            "a3": "Ascend910_9392 (aliases: 910c, 800i-a3-16p, a3-pod)",
            "a2": "Ascend910B4 (aliases: 910b, 800i-a2, a2-8p)",
        },
        "quant_defaults_auto_va": {
            "quarot_priority": "Always prefer *-QuaRot Eco-Tech checkpoints (higher accuracy via Hadamard outlier rotation); for uncataloged models use msmodelslim QuaRot.",
            "a2_a3": "MoE -> w4a8 + c8(int8) KV (>=32k); Dense -> w8a8 + c8 KV (>=32k)",
            "a5": "MoE -> mxfp8_mxfp4 + fp8 KV/Indexer; Dense -> mxfp8 + fp8 KV (>=32k)",
        },
        "agent_3_step_loop": [
            "Step 1: Run `mindie-sim report` (or `--from-script serve.sh`) — never hand-author raw Case JSONs.",
            "Step 2: Read `<prefix>_agent_brief.json` (<8 KB) — never scrape binary `.xlsx` XML.",
            "Step 3: Present 3-Track conclusions (Config/Env, Framework, Kernels); optionally re-run with `--kernel-designs-json`.",
        ],
    }
    if not getattr(args, "emit_json", False):
        md_lines = [
            "# MindIE-Sim Quickstart (Turnkey Ascend NPU Performance Compass)",
            "",
            "## 1. Golden Commands",
            "```bash",
            "# (A) Zero-config catalog report (auto-solves DP/TP/EP/DCP + auto_va quant -> .xlsx + trace + brief)",
            "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out",
            "# (B) Audit & auto-tune an existing vllm-ascend serve.sh (emits *_optimized_serve.sh)",
            "mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out",
            "# (C) Blind-box uncataloged HuggingFace config.json or model directory",
            "mindie-sim report --model ./new_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out",
            "# (D) Pinned parallelism + measured msprof kernel_details.csv reconciliation",
            "mindie-sim report --model qwen3.8-flash-next --hardware 950dt --dp 4 --tp 2 --ep 8 --concurrency 64 \\",
            "  --kernel-csv ./kernel_details.csv --out-dir ./out",
            "```",
            "",
            "## 2. Hardware Aliases & Production Quantization (`auto_va`, QuaRot-First)",
            "- **Hardware**: `950dt`/`a5`/`950dt-96g-pod`/`950dt-144g-pod` | `a3`/`910c`/`a3-pod` | `a2`/`910b`/`800i-a2`",
            "- **A2 / A3 (`64G`)**: MoE -> `w4a8` + `c8` KV (`>=32k`); Dense -> `w8a8` + `c8` KV (`--weight-quant checkpoint` keeps native)",
            "- **A5 (`96G/144G`)**: MoE -> `mxfp8_mxfp4` + `fp8` KV/Indexer; Dense -> `mxfp8` + `fp8` KV",
            "- **QuaRot Priority**: Always prefer `*-QuaRot` Eco-Tech checkpoints (better accuracy via Hadamard outlier rotation); for uncataloged models use `msmodelslim` QuaRot + `--quantization ascend`.",
            "",
            "## 3. Agent 3-Step Co-Design Loop",
            "1. Run `mindie-sim report` (or `--from-script <serve.sh>`) — never hand-write raw Case JSON.",
            "2. Read `<prefix>_agent_brief.json` (<8 KB) — never write XML scrapers for `.xlsx`.",
            "3. Deliver 3-Track actions (Env/CLI, Framework, Kernels) & link `.xlsx` / `trace_view.json` (MindStudio Insight); optionally run Pass-2 `--kernel-designs-json`.",
        ]
        payload["_stdout_markdown"] = "\n".join(md_lines) + "\n"
    return payload


__all__ = [
    "AGENT_BRIEF_SCHEMA",
    "REPORT_BOOLEAN_FLAGS",
    "REPORT_COMMANDS",
    "add_report_commands",
    "parse_script_for_report",
    "parse_token_shorthand",
    "resolve_auto_va_policy",
    "run_quickstart_command",
    "run_report_command",
]
