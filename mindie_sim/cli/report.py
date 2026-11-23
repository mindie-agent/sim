"""Turnkey ``mindie-sim report`` and ``mindie-sim quickstart`` CLI commands.

[Role] Deliver zero-boilerplate, Agent-first evaluation via:
  (1) ``mindie-sim report``: auto-solves HBM-safe parallelism & concurrency
      (forwarding partial parallel axis pins instead of bypassing search),
      defaults script-free ``Before`` to ``merged`` shipping stack with
      ``catalog.quant.resolve_auto_va_quant``, parses ``--from-script <serve.sh>``
      statically via ``mindie_sim.export.parse_launch_script`` with three-point
      Original -> Tuned Shipping -> Target evaluation, accepts Pass-2
      ``--rewrite-plan <plan.json>`` & ``--mode {modeling,simulation}``, and
      writes ``*_comparison.xlsx``, ``*_trace_view.json``, ``< 8 KB``
      ``*_agent_brief.json``, optional ``*_optimized_serve.sh`` (strictly when
      ``--from-script`` is supplied), and a stdout Markdown digest;
  (2) ``mindie-sim quickstart [--json]``: prints golden commands, aliases, and
      the 2-pass Agent co-design loop.
[Invariants] Never executes user launch scripts; never duplicates
  ``catalog.quant.resolve_auto_va_quant``; script-free mode passes
  ``script_audit=None`` and ``serve_script_supplied=False`` with zero
  ``optimized_serve.sh``; always re-raises ``TimeoutError`` before ``OSError``.
[Timeouts & Locks] Bounded 50 MiB metadata reads and single-pass O(Archetypes)
  export delegation; no locks or device contexts.
[Side-Effect Policy] Writes artifacts only into caller-specified ``--out-dir``
  (and optional ``--receipt`` path).
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace as dc_replace
import importlib
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.compose.inputs import (
    BYTES_PER_GIB, DERIVED_ASCEND_SKUS, _QUANT_ALIASES,
    _arch_key, _resolve_hardware, _resolve_model_spec,
)
from mindie_sim.export import (
    AGENT_BRIEF_SCHEMA, DEFAULT_SCENARIO_OUTPUT_TOKENS, DEFAULT_TPOT_SLO_MS, DEFAULT_TTFT_SLO_MS,
    MODE_MODELING, MODE_SIMULATION, ParsedLaunchScript, audit_launch_script, build_export_bundle,
    compose_export_case, export_agent_brief, export_comparison_xlsx, export_msprof_trace,
    format_executive_digest_markdown, parse_launch_script, render_optimized_serve_sh,
)
from mindie_sim.prediction.frontend.speculation import compute_speculative_accept_step
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep.pd import DEFAULT_SPECULATIVE_ACCEPT_RATE

REPORT_COMMANDS = frozenset({"report", "quickstart"})
REPORT_BOOLEAN_FLAGS = frozenset({"--json", "--no-trace", "--no-xlsx"})
MAX_SCRIPT_BYTES = 50 * 1024 * 1024
DEFAULT_REPORT_PROMPT_TOKENS = 32768
AUTO_SIZING_WEIGHT_FRACTION = 0.65
AUTO_DEVICE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_FALLBACK_MEM_GIB = 64
DEFAULT_FALLBACK_WEIGHT_GIB = 16.0
DEFAULT_FALLBACK_TP_CAP = 8
BYTES_PER_PARAM_4BIT = 0.7
BYTES_PER_PARAM_8BIT = 1.1
BYTES_PER_PARAM_BF16 = 2.0
FALLBACK_TPOT_SENTINEL_MS = 1e9
TOKEN_SHORTHAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")
TOKEN_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024 * 1024}
SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")
_URL_PREFIXES = ("http://", "https://")
_PARALLEL_AXES = ("dp", "tp", "ep", "pp", "dcp", "pcp")


def parse_token_shorthand(raw: str | int) -> int:
    """Parse integer or ``32k`` / ``128k`` / ``1m`` token counts."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw <= 0:
            raise argparse.ArgumentTypeError(f"token count must be > 0, got {raw}")
        return raw
    cleaned = str(raw).strip().replace("_", "").replace(",", "")
    match = TOKEN_SHORTHAND_RE.match(cleaned)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid token count {raw!r}; use integer or shorthand like 32k / 128k")
    value = int(round(float(match.group(1)) * TOKEN_MULTIPLIERS[match.group(2).lower()]))
    if value <= 0:
        raise argparse.ArgumentTypeError(f"token count must be > 0, got {raw!r}")
    return value


def _is_moe_model(model_spec: Any) -> bool:
    if int(getattr(model_spec, "routed_experts", 0) or 0) > 0 or int(getattr(model_spec, "num_experts", 0) or 0) > 1:
        return True
    return any(
        int(getattr(lp, "routed_experts", 0) or 0) > 0 or str(getattr(lp, "ffn", "")) == "moe"
        for lp in (getattr(model_spec, "layer_plans", ()) or ())
    )


def _lookup_quarot_checkpoint_hint(model_name: str, arch: str, weight_quant: str | None) -> str:
    try:
        ckpt_mod = importlib.import_module("mindie_sim.catalog.npu_checkpoints")
        finder = getattr(ckpt_mod, "recommend_npu_checkpoint", None) or getattr(ckpt_mod, "get_recommended_checkpoint", None)
        if callable(finder):
            rec = finder(model_name, arch=arch, quant=weight_quant)
            if isinstance(rec, Mapping) and rec.get("checkpoint"):
                return str(rec["checkpoint"])
            if isinstance(rec, str) and rec:
                return rec
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    return (
        "Prefer QuaRot-calibrated MXFP8+MXFP4 checkpoint on A5 (keep norm/router/embed/lm_head BF16)"
        if arch == "a5" else "Prefer *-QuaRot Eco-Tech checkpoint when available (or quantize via msmodelslim QuaRot)"
    )


def _normalize_model_argument(model_arg: str) -> str:
    raw = str(model_arg or "").strip()
    if raw.lower().startswith(_URL_PREFIXES):
        from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES
        tail = raw.rstrip("/").rsplit("/", 1)[-1].lower()
        for key in sorted(set(CATALOG) | set(MODEL_ALIASES), key=len, reverse=True):
            if key in tail:
                return MODEL_ALIASES.get(key, key)
        raise ModelingError(f"report_model_url_requires_local_config_json:{raw} (download config.json locally and pass --model ./config.json)")
    return raw


def resolve_auto_va_policy(
    model: str | Path | Mapping[str, Any], hardware: str, *,
    context_tokens: int = DEFAULT_REPORT_PROMPT_TOKENS,
    weight_quant: str | None = "auto_va", cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Resolve chip-matched Ascend ``vllm-ascend`` production quantization via single SSOT."""
    from mindie_sim.catalog.quant import resolve_auto_va_quant

    model_spec = _resolve_model_spec(model)
    sku, _backend, preset = _resolve_hardware(hardware)
    arch = _arch_key(sku)
    raw_q = (str(weight_quant).strip().lower() if weight_quant is not None else "auto_va")
    explicit_cache = _QUANT_ALIASES.get(str(cache_dtype).strip().lower(), str(cache_dtype).strip().lower()) if cache_dtype else None

    if raw_q in {"checkpoint", "native"}:
        q_val, kv_val, idx_val, source = None, explicit_cache, None, "checkpoint_native"
    elif raw_q in {"", "auto", "auto_va"}:
        q_val, hook_kv, idx_val = resolve_auto_va_quant(model_spec, sku, context_tokens=int(context_tokens))
        kv_val, source = explicit_cache or hook_kv, "auto_va_production"
    else:
        q_val, kv_val, idx_val, source = _QUANT_ALIASES.get(raw_q, raw_q), explicit_cache, None, "explicit_flag"

    return {
        "model_spec": model_spec, "sku": sku, "arch": arch, "topology_preset": preset or None,
        "weight_quant": q_val, "kv_cache_dtype": kv_val, "indexer_kv_dtype": idx_val,
        "policy_source": source, "quarot_guidance": _lookup_quarot_checkpoint_hint(model_spec.name, arch, q_val),
    }


def parse_script_for_report(path: str | Path) -> dict[str, Any]:
    """Statically parse a launch script through canonical ``export.parse_launch_script``."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"launch_script_not_found:{target}")
    if target.stat().st_size > MAX_SCRIPT_BYTES:
        raise ModelingError("launch_script_exceeds_50_mib_limit")
    parsed: ParsedLaunchScript = parse_launch_script(target)
    spec_k = parsed.speculative_config.get("num_speculative_tokens")
    return {
        "script_path": str(target), "parsed": parsed,
        "model": parsed.local_config_path or parsed.inferred_catalog_model or parsed.model_arg or None,
        "weight_quant": parsed.inferred_quant or None, "cache_dtype": parsed.kv_cache_dtype or None,
        "dp": int(parsed.dp or 1), "tp": int(parsed.tp or 1),
        "ep": int(parsed.dp * parsed.tp if parsed.ep_enabled else 1),
        "pp": int(parsed.pp or 1), "dcp": int(parsed.dcp or 1),
        "num_devices": int(parsed.inferred_num_devices or max(1, parsed.dp * parsed.tp * parsed.pp)),
        "max_model_len": int(parsed.max_model_len) if parsed.max_model_len else None,
        "concurrency": int(parsed.max_num_seqs) if parsed.max_num_seqs else None,
        "spec_tokens": int(spec_k) if isinstance(spec_k, int) and spec_k >= 0 else None,
        "enforce_eager": bool(parsed.enforce_eager),
    }


def _apply_script_overrides_to_case(raw_case: dict[str, Any], parsed: ParsedLaunchScript) -> dict[str, Any]:
    rt = dict(raw_case.get("runtime") or {})
    meta = dict(rt.get("metadata") or {})
    if parsed.env_vars:
        meta["script_env"] = dict(parsed.env_vars)
    if parsed.compilation_config:
        meta["script_compilation_config"] = dict(parsed.compilation_config)
    if parsed.additional_config:
        meta["script_additional_config"] = dict(parsed.additional_config)
    if parsed.enforce_eager:
        meta["enforce_eager"] = True
        rt["graph_capture_mode"] = "none"
        rt["enable_torch_compile"] = False
    rt["metadata"] = meta
    raw_case["runtime"] = rt
    return raw_case


def _load_rewrite_plan(path: str | Path | None) -> Any:
    if not path:
        return None
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"rewrite_plan_not_found:{target}")
    if target.stat().st_size > MAX_SCRIPT_BYTES:
        raise ModelingError("rewrite_plan_exceeds_50_mib_limit")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelingError(f"rewrite_plan_invalid:{target}:{exc}") from exc


def _estimate_auto_num_devices(model_spec: Any, sku: str, weight_quant: str | None) -> int:
    from mindie_sim.backends.ascend import profiles as ascend_profiles
    from mindie_sim.backends.nvidia import profiles as nvidia_profiles
    from mindie_sim.catalog.quant import QUANTIZATION_PRESETS, quant_preset
    from mindie_sim.catalog.weights import sharded_weight_bytes
    from mindie_sim.schema.parallel import ParallelSpec

    entry = {**ascend_profiles(), **DERIVED_ASCEND_SKUS, **nvidia_profiles()}.get(sku, {})
    mem_b = float(entry.get("memory_bytes") or (entry.get("device") or {}).get("memory_bytes") or DEFAULT_FALLBACK_MEM_GIB * BYTES_PER_GIB)
    eff_spec = model_spec
    if weight_quant and str(weight_quant).strip().lower() in QUANTIZATION_PRESETS:
        eff_spec = dc_replace(model_spec, quantization=quant_preset(str(weight_quant), model=model_spec))
    total_w = float(sharded_weight_bytes(eff_spec, ParallelSpec()))
    target_per_die = max(1.0, mem_b * AUTO_SIZING_WEIGHT_FRACTION)
    min_dies = max(1, int((total_w + target_per_die - 1.0) // target_per_die))
    count = 1
    while count < min_dies:
        count *= 2
    return count


def _auto_solve_report_layout(
    *, model: str, sku: str, topology_preset: str | None, num_devices: int,
    prompt_tokens: int, output_tokens: int, tpot_slo_ms: float, ttft_slo_ms: float | None,
    quant: str | None, kv_cache_dtype: str | None, indexer_kv_dtype: str | None,
    deployment_mode: str, feature_track: str | None = None,
    mtp_proposal_tokens: int | None = None, mtp_acceptance_length: float | None = None,
    apply_target_envelope: bool = False, rewrite_plan: Any = None,
    allow_unbound_formula: bool = True, phase: str = "decode",
    pinned_parallel: Mapping[str, int] | None = None, max_sequences_per_dp: int | None = None,
) -> dict[str, Any]:
    """Auto-solve HBM-safe parallelism + concurrency via ``solve_optimal_deployment``."""
    from mindie_sim.solve import _resolve_kv_pooling_mode
    from mindie_sim.sweep.pd import solve_optimal_deployment

    dep = "colocated" if deployment_mode == "colocated" else ("pd_disaggregated" if deployment_mode.startswith("pd_") else "auto")
    if dep == "pd_disaggregated" and int(num_devices) < 2:
        raise ModelingError(f"pd_disaggregated_requires_at_least_two_devices:{int(num_devices)}")
    solve_kw: dict[str, Any] = dict(
        num_devices=int(num_devices), prompt_tokens=int(prompt_tokens), output_tokens=int(output_tokens),
        ttft_sla_ms=ttft_slo_ms, tpot_sla_ms=float(tpot_slo_ms), deployment=dep,
        pd_transfer_mode=_resolve_kv_pooling_mode({"pd_pooled": True, "pd_unpooled": False}.get(deployment_mode)),
        quant=quant, kv_cache_dtype=kv_cache_dtype, indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset, allow_unconstrained_ttft=(ttft_slo_ms is None),
        feature_track=feature_track, mtp_proposal_tokens=mtp_proposal_tokens,
        mtp_acceptance_length=mtp_acceptance_length, apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
    )
    if pinned_parallel:
        solve_kw["pinned_parallel"] = dict(pinned_parallel)
    if max_sequences_per_dp is not None:
        solve_kw["max_sequences_per_dp"] = int(max_sequences_per_dp)
    solved = solve_optimal_deployment(model, sku, **solve_kw)
    best = solved.get("best_configuration")
    if best is None:
        cands = [c for c in solved.get("candidates", ()) if isinstance(c, Mapping) and int(c.get("b_vram_max") or c.get("b_vram") or 0) >= 1]
        if cands:
            best = max(cands, key=lambda c: (int(c.get("b_vram_max") or c.get("b_vram") or 0), -float(c.get("effective_tpot_ms", FALLBACK_TPOT_SENTINEL_MS) or FALLBACK_TPOT_SENTINEL_MS)))
    best = best or {}
    par = best.get("prefill_parallel" if phase == "prefill" else "decode_parallel") or best.get("parallel") or {}
    is_pd = solved.get("winner_mode") == "pd_disaggregated"
    return {
        "dp": int(par.get("dp", 1) or 1), "tp": int(par.get("tp", min(num_devices, DEFAULT_FALLBACK_TP_CAP)) or 1),
        "ep": int(par.get("ep", par.get("dp", 1) * par.get("tp", 1)) or 1),
        "pp": int(par.get("pp", 1) or 1), "dcp": int(par.get("cp", par.get("dcp", 1)) or 1), "pcp": int(par.get("pcp", 1) or 1),
        "concurrency": max(1, int(best.get("b_feasible") or best.get("b_vram_max") or best.get("b_vram") or 1)),
        "spec_tokens": int(best.get("mtp_proposal_tokens", 0) or 0),
        "deployment_mode": "colocated" if not is_pd else ("pd_pooled" if best.get("pd_transfer_mode") == "pooled_dram_ssd" else "pd_unpooled"),
        "solver_feasible": bool(solved.get("feasible", False)), "solved": solved,
    }


def _solved_column_inputs(
    layout: Mapping[str, Any], *, model: str, sku: str, feature_track: str, phase: str,
    prompt_tokens: int, output_tokens: int, concurrency: int | None, compose_kw: Mapping[str, Any],
) -> tuple[dict[str, Any], StepRequest, dict[str, Any]]:
    from mindie_sim.solve import _bundle_inputs_for_solution

    raw, step, meta = _bundle_inputs_for_solution(
        layout["solved"], model=model, hardware=sku, runtime_stack="vllm-ascend",
        base_kw=compose_kw, feature_track=feature_track,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens, scope="model", phase=phase,
    )
    if concurrency is not None:
        step = dc_replace(step, sequences=int(concurrency))
    pool = meta.get("pd_pool_summary")
    dep_mode = "colocated" if not pool else ("pd_pooled" if pool.get("pd_transfer_mode") == "pooled_dram_ssd" else "pd_unpooled")
    rt = dict(raw.get("runtime") or {})
    rt["metadata"] = {**dict(rt.get("metadata") or {}), "deployment_mode": dep_mode}
    raw["runtime"] = rt
    return raw, step, meta


def _report_column(
    layout: Mapping[str, Any] | None, *, model: str, sku: str, phase: str, track: str,
    quant: str | None, kv: str | None, idx_kv: str | None, par: Mapping[str, int],
    conc: int, concurrency: int | None, common_kw: Mapping[str, Any],
    spec_tokens: int | None = None, accept_len: float | None = None,
) -> tuple[dict[str, Any], StepRequest, dict[str, Any]]:
    prompt_tokens, output_tokens = int(common_kw["prompt_tokens"]), int(common_kw["output_tokens"])
    if layout is not None and layout.get("solved") is not None:
        compose_kw = dict(
            quant=quant, kv_cache_dtype=kv, indexer_kv_dtype=idx_kv,
            topology_preset=common_kw.get("topology_preset"), prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        )
        raw, step, meta = _solved_column_inputs(
            layout, model=model, sku=sku, feature_track=track, phase=phase,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens, concurrency=concurrency, compose_kw=compose_kw,
        )
        return raw, step, {
            "handoff_tpot_ms": meta.get("handoff_tpot_ms"),
            "deployment_total_dies": meta.get("deployment_total_dies") or None,
            "pd_pool_summary": meta.get("pd_pool_summary"),
        }
    col_kw = dict(common_kw)
    if spec_tokens is not None:
        col_kw["mtp_proposal_tokens"] = int(spec_tokens)
    if accept_len is not None:
        col_kw["mtp_acceptance_length"] = float(accept_len)
    raw, req = compose_export_case(
        model, sku, feature_track=track, concurrency=conc, quant=quant, kv_cache_dtype=kv, **dict(par), **col_kw,
    )
    step = StepRequest(
        phase=phase, tokens=req.tokens if phase == "decode" else prompt_tokens,
        context=req.context if phase == "decode" else 0, sequences=req.sequences, scope="model",
    )
    return raw, step, {}


def _attach_bundle_context(bundle: Any, *, serve_script_supplied: bool, solver_feasible: bool | None = None) -> Any:
    eff = dict(getattr(bundle, "effective_config", {}) or {})
    if solver_feasible is not None:
        eff["solver_feasible"] = bool(solver_feasible)
    return dc_replace(bundle, serve_script_supplied=bool(serve_script_supplied), effective_config=eff)


def add_report_commands(subparsers) -> None:
    """Register turnkey ``report`` and ``quickstart`` commands."""
    rep = subparsers.add_parser("report", help="turnkey one-shot evaluation -> .xlsx + trace_view.json + agent_brief.json")
    rep.add_argument("--model", default=None, help="catalog model, alias, HuggingFace config.json, or local directory")
    rep.add_argument("--from-script", default=None, help="audit & optimize an existing serve.sh launch script (zero-exec)")
    rep.add_argument("--hardware", default="a3", help="hardware SKU or alias (950dt, a5, a3, 910c, a2, 910b)")
    rep.add_argument("--topology-preset", default=None, help="optional topology preset override (a5_pod, a3_16p)")
    rep.add_argument("--num-devices", type=int, default=None, help="total NPU dies (auto-sized when omitted)")
    rep.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    for flag in ("--dp", "--tp", "--dcp", "--ep", "--pp", "--pcp"):
        rep.add_argument(flag, type=int, default=None)
    rep.add_argument("--prompt-tokens", "--context", dest="prompt_tokens", type=parse_token_shorthand, default=None)
    rep.add_argument("--output-tokens", type=parse_token_shorthand, default=DEFAULT_SCENARIO_OUTPUT_TOKENS)
    rep.add_argument("--concurrency", type=int, default=None, help="sequences per DP replica (auto-solved when omitted)")
    rep.add_argument("--tpot-ms", type=float, default=DEFAULT_TPOT_SLO_MS)
    rep.add_argument("--ttft-ms", type=float, default=None)
    rep.add_argument("--spec-tokens", "--mtp", dest="spec_tokens", type=int, default=None)
    rep.add_argument("--accept-len", type=float, default=None)
    rep.add_argument("--mode", choices=(MODE_MODELING, MODE_SIMULATION), default=MODE_MODELING)
    rep.add_argument("--compare-track", default="merged", help="Before track (default: merged)")
    rep.add_argument("--target-track", default="expected_ideal", help="After track (default: expected_ideal)")
    rep.add_argument("--rewrite-plan", default=None, help="Pass-2 Agent semantic rewrite plan JSON")
    rep.add_argument("--weight-quant", default="auto_va", help="weight quant preset (default: auto_va)")
    rep.add_argument("--cache-dtype", default=None, help="KV cache dtype override (bf16/c8/int8/fp8_e4m3)")
    rep.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    rep.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")
    rep.add_argument("--deployment-mode", choices=("auto", "pd_unpooled", "pd_pooled", "colocated"), default="auto")
    rep.add_argument("--kernel-designs-json", default=None, help="[retired] raises ModelingError — use --rewrite-plan")
    rep.add_argument("--observations", "--kernel-csv", dest="observations", default=None)
    rep.add_argument("--out-dir", default="./mindie_sim_report")
    rep.add_argument("--prefix", default=None)
    rep.add_argument("--no-xlsx", action="store_true")
    rep.add_argument("--no-trace", action="store_true")
    rep.add_argument("--no-pipe-lanes", dest="include_pipe_lanes", action="store_false")
    rep.add_argument("--no-counters", dest="include_counters", action="store_false")
    rep.add_argument("--no-overlap", dest="include_overlap", action="store_false")
    rep.add_argument("--include-submission", action="store_true")
    rep.add_argument("--json", dest="emit_json", action="store_true")
    rep.add_argument("--receipt", dest="output", default=None)

    qs = subparsers.add_parser("quickstart", help="print golden commands, quant aliases, and 2-pass Agent loop")
    qs.add_argument("--json", dest="emit_json", action="store_true")
    qs.add_argument("--output", "-o", default=None)


def _emit_report_artifacts(
    *, before_bundle: Any, after_bundle: Any, tuned_bundle: Any | None,
    args: argparse.Namespace, script_info: Mapping[str, Any] | None, load_designs: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    designs = load_designs(getattr(args, "kernel_designs_json", None))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = args.prefix or f"{after_bundle.model_name}_{after_bundle.hardware_sku}_{after_bundle.phase}"
    prefix = SAFE_PREFIX_RE.sub("_", raw_prefix).strip("_") or "mindie_sim_report"

    generated: dict[str, str] = {}
    summary: dict[str, Any] = {}
    audit_dict: dict[str, Any] | None = None
    if script_info and isinstance(script_info.get("parsed"), ParsedLaunchScript):
        opt_sh_path = out_dir / f"{prefix}_optimized_serve.sh"
        render_optimized_serve_sh(
            script_info["parsed"], after_bundle=after_bundle, tuned_bundle=tuned_bundle, output_path=opt_sh_path,
        )
        generated["optimized_serve_sh"] = str(opt_sh_path)
        audit_dict = audit_launch_script(
            script_info["parsed"], before_bundle=before_bundle, after_bundle=after_bundle,
            tuned_bundle=tuned_bundle, optimized_script_path=str(opt_sh_path),
        ).to_dict()

    if not getattr(args, "no_xlsx", False):
        xlsx_path = out_dir / f"{prefix}_comparison.xlsx"
        summary["xlsx"] = export_comparison_xlsx(
            before_bundle, xlsx_path, compare_source=after_bundle,
            agent_kernel_designs=designs, script_audit=audit_dict,
        ).to_dict()
        generated["comparison_xlsx"] = str(xlsx_path)
    if not getattr(args, "no_trace", False):
        trace_path = out_dir / f"{prefix}_trace_view.json"
        summary["trace"] = export_msprof_trace(
            after_bundle, trace_path,
            include_pipe_lanes=bool(getattr(args, "include_pipe_lanes", True)),
            include_counters=bool(getattr(args, "include_counters", True)),
            include_overlap=bool(getattr(args, "include_overlap", True)),
        ).to_dict()
        generated["trace_view_json"] = str(trace_path)

    brief_path = out_dir / f"{prefix}_agent_brief.json"
    generated["agent_brief_json"] = str(brief_path)
    brief = export_agent_brief(before_bundle, after_bundle, brief_path, generated_files=generated, script_audit=audit_dict)
    brief_dict = dict(brief) if isinstance(brief, Mapping) else {}
    return summary, brief_dict, format_executive_digest_markdown(brief_dict)


def _report_tracks(mode: str, *, compare_track: str, target_track: str, enforce_eager: bool) -> tuple[str, str]:
    before = "raw_eager" if enforce_eager else compare_track
    if mode == MODE_SIMULATION and before.strip().lower() == "expected_ideal":
        before = "merged"
    after = target_track
    if mode == MODE_SIMULATION and after.strip().lower() == "expected_ideal":
        after = before
    return before, after


def _accept_len_for(spec_tokens: int, explicit_accept: float | None) -> float:
    if explicit_accept is not None:
        return float(explicit_accept)
    return float(compute_speculative_accept_step(DEFAULT_SPECULATIVE_ACCEPT_RATE, int(spec_tokens))) if int(spec_tokens) > 0 else 1.0


def run_report_command(args: argparse.Namespace, *, load_designs: Any) -> dict[str, Any]:
    """Execute ``mindie-sim report`` with Iso-SLA auto-solve, ``auto_va``, and ``--rewrite-plan``."""
    load_designs(getattr(args, "kernel_designs_json", None))
    script_info = parse_script_for_report(args.from_script) if getattr(args, "from_script", None) else None
    raw_model = args.model or (script_info.get("model") if script_info else None)
    if not raw_model:
        raise ModelingError("report_requires_model — pass --model <catalog|config.json|hf_dir> or --from-script <serve.sh>")
    model_arg = _normalize_model_argument(str(raw_model))
    mode = str(getattr(args, "mode", MODE_MODELING) or MODE_MODELING).strip().lower()
    rewrite_plan = _load_rewrite_plan(getattr(args, "rewrite_plan", None))
    output_tokens = int(args.output_tokens or DEFAULT_SCENARIO_OUTPUT_TOKENS)
    script_max_len = (script_info or {}).get("max_model_len")
    if args.prompt_tokens is not None:
        prompt_tokens = int(args.prompt_tokens)
    elif script_max_len:
        prompt_tokens = max(1, int(script_max_len) - output_tokens)
    else:
        prompt_tokens = DEFAULT_REPORT_PROMPT_TOKENS
    policy = resolve_auto_va_policy(
        model_arg, args.hardware, context_tokens=prompt_tokens, weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
    )
    max_pos = int(getattr(policy["model_spec"], "max_position_embeddings", 0) or 0)
    if args.prompt_tokens is None and max_pos > output_tokens and (prompt_tokens + output_tokens) > max_pos:
        prompt_tokens = max(1, max_pos - output_tokens)
    topo_preset = args.topology_preset or policy["topology_preset"]
    pinned_parallel = {axis: int(getattr(args, axis)) for axis in _PARALLEL_AXES if getattr(args, axis, None) is not None}
    full_explicit = (
        args.num_devices is None and args.dp is not None and args.tp is not None
        and (args.ep is not None or not _is_moe_model(policy["model_spec"])) and args.concurrency is not None
    )

    before_track, after_track = _report_tracks(
        mode, compare_track=str(args.compare_track), target_track=str(args.target_track),
        enforce_eager=bool(script_info and script_info.get("enforce_eager")),
    )
    has_target_after = (mode == MODE_MODELING) or (rewrite_plan is not None)
    before_quant = (script_info.get("weight_quant") if script_info else None) or policy["weight_quant"]
    before_kv = (script_info.get("cache_dtype") if script_info else None) or policy["kv_cache_dtype"]
    after_quant, after_kv, after_idx_kv = policy["weight_quant"], policy["kv_cache_dtype"], policy["indexer_kv_dtype"]
    explicit_spec = int(args.spec_tokens) if args.spec_tokens is not None else None
    explicit_accept = float(args.accept_len) if args.accept_len is not None else None
    layout_before = layout_tuned = layout_after = None

    if full_explicit:
        after_par = {axis: int(getattr(args, axis) or 1) for axis in _PARALLEL_AXES}
        conc, after_spec = int(args.concurrency or 1), int(explicit_spec or 0)
        before_spec = int(script_info["spec_tokens"] if (script_info and script_info.get("spec_tokens") is not None) else after_spec)
        dep_mode = str(args.deployment_mode or "auto")
        before_par, before_conc = dict(after_par), conc
        tuned_par, tuned_conc = dict(after_par), conc
    else:
        n_dev = int(args.num_devices or (script_info.get("num_devices") if script_info else None) or _estimate_auto_num_devices(policy["model_spec"], policy["sku"], after_quant))
        common_solve = dict(
            model=model_arg, sku=policy["sku"], topology_preset=topo_preset, num_devices=n_dev,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens, tpot_slo_ms=float(args.tpot_ms),
            ttft_slo_ms=args.ttft_ms, deployment_mode=str(args.deployment_mode or "auto"),
            mtp_proposal_tokens=explicit_spec, mtp_acceptance_length=explicit_accept, phase=args.phase,
            pinned_parallel=pinned_parallel or None, max_sequences_per_dp=int(args.concurrency) if args.concurrency is not None else None,
        )
        layout_after = _auto_solve_report_layout(
            **common_solve, quant=after_quant, kv_cache_dtype=after_kv, indexer_kv_dtype=after_idx_kv,
            feature_track=after_track, apply_target_envelope=has_target_after,
            rewrite_plan=rewrite_plan if has_target_after else None, allow_unbound_formula=(mode == MODE_MODELING),
        )
        after_par = {k: layout_after[k] for k in _PARALLEL_AXES}
        conc = int(args.concurrency or layout_after["concurrency"])
        after_spec = int(explicit_spec if explicit_spec is not None else layout_after["spec_tokens"])
        dep_mode = layout_after["deployment_mode"] if args.deployment_mode == "auto" else str(args.deployment_mode)

        if script_info:
            layout_tuned = _auto_solve_report_layout(
                **common_solve, quant=after_quant, kv_cache_dtype=after_kv, indexer_kv_dtype=after_idx_kv,
                feature_track="merged", apply_target_envelope=False, rewrite_plan=None, allow_unbound_formula=False,
            )
            tuned_par, tuned_conc = {k: layout_tuned[k] for k in _PARALLEL_AXES}, int(args.concurrency or layout_tuned["concurrency"])
            before_par = dict(dp=script_info["dp"], tp=script_info["tp"], ep=script_info["ep"], pp=script_info["pp"], dcp=script_info["dcp"], pcp=1)
            before_conc = int(script_info.get("concurrency") or tuned_conc)
            before_spec = int(script_info["spec_tokens"] if script_info.get("spec_tokens") is not None else (explicit_spec or 0))
        elif has_target_after or (before_track, before_quant, before_kv) != (after_track, after_quant, after_kv):
            layout_before = _auto_solve_report_layout(
                **common_solve, quant=before_quant, kv_cache_dtype=before_kv, indexer_kv_dtype=policy["indexer_kv_dtype"],
                feature_track=before_track, apply_target_envelope=False, rewrite_plan=None, allow_unbound_formula=False,
            )
            before_par = {k: layout_before[k] for k in _PARALLEL_AXES}
            before_conc = int(args.concurrency or layout_before["concurrency"])
            before_spec = int(explicit_spec if explicit_spec is not None else layout_before["spec_tokens"])
            tuned_par, tuned_conc = dict(before_par), before_conc
        else:
            before_par, before_conc, before_spec = dict(after_par), conc, after_spec
            tuned_par, tuned_conc, layout_before = dict(after_par), conc, layout_after

    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=topo_preset, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=float(args.ttft_ms or DEFAULT_TTFT_SLO_MS),
        mtp_proposal_tokens=after_spec, mtp_acceptance_length=_accept_len_for(after_spec, explicit_accept),
        engram_cpu_offload=args.engram_offload, deployment_mode=dep_mode,
    )
    column_kw = dict(model=model_arg, sku=policy["sku"], phase=args.phase, common_kw=common_kw, concurrency=args.concurrency)
    raw_before, step_b, meta_b = _report_column(
        layout_before, track=before_track, quant=before_quant, kv=before_kv, idx_kv=policy["indexer_kv_dtype"],
        par=before_par, conc=before_conc, spec_tokens=before_spec, accept_len=_accept_len_for(before_spec, explicit_accept), **column_kw,
    )
    if script_info and isinstance(script_info.get("parsed"), ParsedLaunchScript):
        raw_before = _apply_script_overrides_to_case(raw_before, script_info["parsed"])
    raw_after, step_a, meta_a = _report_column(
        layout_after, track=after_track, quant=after_quant, kv=after_kv, idx_kv=after_idx_kv, par=after_par, conc=conc, **column_kw,
    )
    script_supplied = script_info is not None
    before_bundle = _attach_bundle_context(
        build_export_bundle(
            raw_before, request=step_b, replay=replay, observations_path=args.observations,
            cache_dtype=before_kv, engram_host_offload=args.engram_offload, apply_target_envelope=False, **meta_b,
        ),
        serve_script_supplied=script_supplied,
        solver_feasible=(bool(layout_before.get("solver_feasible", True)) if layout_before is not None else None),
    )
    tuned_bundle = None
    if script_info is not None:
        raw_tuned, step_t, meta_t = _report_column(
            layout_tuned, track="merged", quant=after_quant, kv=after_kv, idx_kv=after_idx_kv, par=tuned_par, conc=tuned_conc, **column_kw,
        )
        tuned_bundle = _attach_bundle_context(
            build_export_bundle(
                raw_tuned, request=step_t, replay=replay, cache_dtype=after_kv,
                engram_host_offload=args.engram_offload, apply_target_envelope=False, **meta_t,
            ),
            serve_script_supplied=True,
            solver_feasible=(bool(layout_tuned.get("solver_feasible", True)) if layout_tuned is not None else None),
        )
    after_bundle = _attach_bundle_context(
        build_export_bundle(
            raw_after, request=step_a, replay=replay, cache_dtype=after_kv, engram_host_offload=args.engram_offload,
            apply_target_envelope=has_target_after, rewrite_plan=rewrite_plan if has_target_after else None,
            allow_unbound_formula=(mode == MODE_MODELING), **meta_a,
        ),
        serve_script_supplied=script_supplied,
        solver_feasible=(bool(layout_after.get("solver_feasible", True)) if layout_after is not None else None),
    )
    summary, brief_dict, digest_md = _emit_report_artifacts(
        before_bundle=before_bundle, after_bundle=after_bundle, tuned_bundle=tuned_bundle,
        args=args, script_info=script_info, load_designs=load_designs,
    )
    receipt: dict[str, Any] = {
        "ok": True, "command": "report", "mode": mode, "model_name": after_bundle.model_name,
        "hardware_sku": after_bundle.hardware_sku, "phase": after_bundle.phase, "agent_brief": brief_dict, **summary,
    }
    if not getattr(args, "emit_json", False):
        receipt["_stdout_markdown"] = digest_md
    return receipt


def run_quickstart_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return turnkey quickstart guide (Markdown by default, or ``--json``)."""
    payload: dict[str, Any] = {
        "schema": "mindie-sim.quickstart/1",
        "golden_commands": [
            {"name": "zero_config_modeling_report", "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out"},
            {"name": "pass2_agent_rewrite_plan", "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --rewrite-plan ./out/plan.json --out-dir ./out"},
            {"name": "strict_simulation_mode", "cmd": "mindie-sim report --model qwen3-32b --hardware a3 --mode simulation --compare-track merged --target-track unmerged_pr --out-dir ./out"},
            {"name": "audit_launch_script", "cmd": "mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out"},
            {"name": "blind_box_hf_config", "cmd": "mindie-sim report --model ./my_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out"},
        ],
        "hardware_aliases": {
            "a5": "Ascend950DT_9582 (950dt, 950dt-96g-pod, 950dt-144g-pod, a5-pod)",
            "a3": "Ascend910_9392 (910c, 800i-a3-16p, a3-pod)",
            "a2": "Ascend910B4 (910b, 800i-a2, a2-8p)",
        },
        "quant_defaults_auto_va": {
            "quarot_priority": "Prefer *-QuaRot Eco-Tech checkpoints; use msmodelslim QuaRot for uncataloged models.",
            "a2_a3": "MoE -> w4a8 + c8(int8) KV; Dense -> w8a8 + c8 KV",
            "a5": "MoE -> mxfp8_mxfp4 + fp8 KV/Indexer; Dense -> mxfp8 + fp8 KV",
        },
        "agent_2_pass_loop": [
            "Pass 1: Run `mindie-sim report` -> read `<prefix>_agent_brief.json` (<8 KB).",
            "Pass 2: Emit `plan.json` and re-run `mindie-sim report --rewrite-plan plan.json`.",
            "Mode Split: `--mode simulation` enforces Layer-2 bound ops only; `--mode modeling` evaluates verified target ceiling.",
        ],
    }
    if not getattr(args, "emit_json", False):
        payload["_stdout_markdown"] = "\n".join([
            "# MindIE-Sim Quickstart (Agent-First Ascend NPU Performance Compass)", "",
            "## 1. Golden Commands", "```bash",
            *(item["cmd"] for item in payload["golden_commands"]), "```", "",
            "## 2. Hardware Aliases & Production Quantization (`auto_va`)",
            "- **A5 (`950dt`/`a5`)**: MoE -> `mxfp8_mxfp4` + `fp8` KV/Indexer | Dense -> `mxfp8` + `fp8` KV",
            "- **A3 (`a3`/`910c`) / A2 (`a2`/`910b`)**: MoE -> `w4a8` + `c8` KV | Dense -> `w8a8` + `c8` KV", "",
            "## 3. Agent 2-Pass Co-Design Loop",
            *(f"{i}. {line}" for i, line in enumerate(payload["agent_2_pass_loop"], 1)), "",
        ])
    return payload


__all__ = [
    "AGENT_BRIEF_SCHEMA", "REPORT_BOOLEAN_FLAGS", "REPORT_COMMANDS",
    "add_report_commands", "parse_script_for_report", "parse_token_shorthand",
    "resolve_auto_va_policy", "run_quickstart_command", "run_report_command",
]
