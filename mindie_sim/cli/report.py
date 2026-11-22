"""Turnkey ``mindie-sim report`` and ``mindie-sim quickstart`` CLI commands.

[Role] Deliver zero-boilerplate, Agent-first evaluation via:
  (1) ``mindie-sim report``: auto-solves HBM-safe parallelism & concurrency
      when omitted, supports blind-box HuggingFace ``config.json``/directories,
      defaults to chip-matched ``auto_va`` production quantization (QuaRot-first
      guidance), parses ``--from-script <serve.sh>`` statically via
      ``mindie_sim.export.parse_launch_script``, accepts Pass-2 Agent
      ``--rewrite-plan <plan.json>`` & ``--mode {modeling,simulation}``, and
      writes ``*_comparison.xlsx``, ``*_trace_view.json``, compact ``< 8 KB``
      ``*_agent_brief.json``, optional ``*_optimized_serve.sh``, and a ~25-line
      stdout Markdown Executive Digest;
  (2) ``mindie-sim quickstart [--json]``: prints <45-line golden commands,
      hardware/quant aliases, Simulation-vs-Modeling split, and the 2-pass
      Agent co-design loop.
[Invariants] Never executes user launch scripts (pure ``shlex`` static parse);
  never duplicates ``mindie_sim.export`` parsers/renderers; never weakens
  low-level ``compose_case()`` defaults; always re-raises ``TimeoutError``
  before ``OSError``.
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
    AGENT_BRIEF_SCHEMA,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_TPOT_SLO_MS,
    MODE_MODELING,
    MODE_SIMULATION,
    ParsedLaunchScript,
    audit_launch_script,
    build_export_bundle,
    compose_export_case,
    export_agent_brief,
    export_comparison_xlsx,
    export_msprof_trace,
    format_executive_digest_markdown,
    parse_launch_script,
    render_optimized_serve_sh,
)
from mindie_sim.prediction.frontend.speculation import compute_speculative_accept_step
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep.pd import DEFAULT_SPECULATIVE_ACCEPT_RATE

REPORT_COMMANDS = frozenset({"report", "quickstart"})
REPORT_BOOLEAN_FLAGS = frozenset({"--json", "--no-trace", "--no-xlsx"})
MAX_SCRIPT_BYTES = 50 * 1024 * 1024
DEFAULT_REPORT_PROMPT_TOKENS = 32768
LONG_CONTEXT_KV_QUANT_THRESHOLD = 32768
AUTO_SIZING_WEIGHT_FRACTION = 0.65
AUTO_DEVICE_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
TOKEN_SHORTHAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")
TOKEN_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024 * 1024}
SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")
_URL_PREFIXES = ("http://", "https://")


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


def _normalize_model_argument(model_arg: str) -> str:
    """Reject uncached remote URLs cleanly with actionable local config.json guidance."""
    raw = str(model_arg or "").strip()
    if raw.lower().startswith(_URL_PREFIXES):
        from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES
        tail = raw.rstrip("/").rsplit("/", 1)[-1].lower()
        for key in sorted(set(CATALOG) | set(MODEL_ALIASES), key=len, reverse=True):
            if key in tail:
                return MODEL_ALIASES.get(key, key)
        raise ModelingError(
            "report_model_url_requires_local_config_json:"
            f"{raw} (download config.json locally and pass --model ./config.json)"
        )
    return raw


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
    explicit_cache = (
        _QUANT_ALIASES.get(str(cache_dtype).strip().lower(), str(cache_dtype).strip().lower())
        if cache_dtype else None
    )

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


def parse_script_for_report(path: str | Path) -> dict[str, Any]:
    """Statically parse a launch script through canonical ``export.parse_launch_script``."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"launch_script_not_found:{target}")
    if target.stat().st_size > MAX_SCRIPT_BYTES:
        raise ModelingError("launch_script_exceeds_50_mib_limit")
    parsed: ParsedLaunchScript = parse_launch_script(target)
    model_val = parsed.local_config_path or parsed.inferred_catalog_model or parsed.model_arg or None
    return {
        "script_path": str(target),
        "parsed": parsed,
        "model": model_val,
        "weight_quant": parsed.inferred_quant or None,
        "cache_dtype": parsed.kv_cache_dtype or None,
        "dp": int(parsed.dp or 1),
        "tp": int(parsed.tp or 1),
        "ep": int(parsed.dp * parsed.tp if parsed.ep_enabled else 1),
        "pp": int(parsed.pp or 1),
        "dcp": int(parsed.dcp or 1),
        "num_devices": int(parsed.inferred_num_devices or max(1, parsed.dp * parsed.tp * parsed.pp)),
        "max_model_len": int(parsed.max_model_len) if parsed.max_model_len else None,
        "concurrency": int(parsed.max_num_seqs) if parsed.max_num_seqs else None,
        "enforce_eager": bool(parsed.enforce_eager),
    }


def _load_rewrite_plan(path: str | Path | None) -> Any:
    """Load an Agent Pass-2 semantic rewrite plan JSON bounded to 50 MiB."""
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
    except TimeoutError:
        raise
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
    feature_track: str | None = None,
    mtp_proposal_tokens: int | None = None,
    mtp_acceptance_length: float | None = None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
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
        feature_track=feature_track,
        mtp_proposal_tokens=mtp_proposal_tokens,
        mtp_acceptance_length=mtp_acceptance_length,
        apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
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


def add_report_commands(subparsers) -> None:
    """Register turnkey ``report`` and ``quickstart`` commands."""
    rep = subparsers.add_parser(
        "report",
        help="turnkey one-shot evaluation: auto-solve parallelism/quant -> .xlsx + trace_view.json + agent_brief.json",
    )
    rep.add_argument("--model", default=None, help="catalog model, alias, HuggingFace config.json, or local model directory")
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
    rep.add_argument(
        "--mode", choices=(MODE_MODELING, MODE_SIMULATION), default=MODE_MODELING,
        help="modeling (Before sim vs After SLA target ceiling) or simulation (strict Layer-2 bound implementations only)",
    )
    rep.add_argument("--compare-track", default="unmerged_pr", help="Before track (default: unmerged_pr)")
    rep.add_argument("--target-track", default="expected_ideal", help="After track (default: expected_ideal)")
    rep.add_argument(
        "--rewrite-plan", default=None,
        help="Pass-2 Agent semantic rewrite plan JSON (applied to After modeling graph)",
    )
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
        help="print <45-line golden commands, QuaRot quant aliases, and the 2-pass Agent loop",
    )
    qs.add_argument("--json", dest="emit_json", action="store_true", help="emit machine-readable quickstart JSON")
    qs.add_argument("--output", "-o", default=None)


def _emit_report_artifacts(
    *,
    before_bundle: Any,
    after_bundle: Any,
    args: argparse.Namespace,
    script_info: Mapping[str, Any] | None,
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

    audit_dict: dict[str, Any] | None = None
    if script_info and isinstance(script_info.get("parsed"), ParsedLaunchScript):
        opt_sh_path = out_dir / f"{prefix}_optimized_serve.sh"
        render_optimized_serve_sh(
            script_info["parsed"], after_bundle=after_bundle, output_path=opt_sh_path,
        )
        generated["optimized_serve_sh"] = str(opt_sh_path)
        audit_report = audit_launch_script(
            script_info["parsed"],
            before_bundle=before_bundle,
            after_bundle=after_bundle,
            optimized_script_path=str(opt_sh_path),
        )
        audit_dict = audit_report.to_dict()

    brief_path = out_dir / f"{prefix}_agent_brief.json"
    generated["agent_brief_json"] = str(brief_path)
    brief = export_agent_brief(
        before_bundle, after_bundle, brief_path,
        generated_files=generated, script_audit=audit_dict,
    )
    brief_dict = dict(brief) if isinstance(brief, Mapping) else {}
    digest_md = format_executive_digest_markdown(brief_dict)
    return summary, brief_dict, digest_md


def run_report_command(args: argparse.Namespace, *, load_designs: Any) -> dict[str, Any]:
    """Execute ``mindie-sim report`` with Iso-SLA auto-solve, ``auto_va``, and ``--rewrite-plan``."""
    script_info = parse_script_for_report(args.from_script) if getattr(args, "from_script", None) else None
    raw_model = args.model or (script_info.get("model") if script_info else None)
    if not raw_model:
        raise ModelingError(
            "report_requires_model — pass --model <catalog|config.json|hf_dir> or --from-script <serve.sh>"
        )
    model_arg = _normalize_model_argument(str(raw_model))
    mode = str(getattr(args, "mode", MODE_MODELING) or MODE_MODELING).strip().lower()
    rewrite_plan = _load_rewrite_plan(getattr(args, "rewrite_plan", None))
    prompt_tokens = int(
        args.prompt_tokens
        or ((script_info or {}).get("max_model_len") or DEFAULT_REPORT_PROMPT_TOKENS)
    )
    output_tokens = int(args.output_tokens or DEFAULT_SCENARIO_OUTPUT_TOKENS)
    policy = resolve_auto_va_policy(
        model_arg, args.hardware, context_tokens=prompt_tokens,
        weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
    )
    topo_preset = args.topology_preset or policy["topology_preset"]
    explicit_parallel = any(getattr(args, k, None) is not None for k in ("dp", "tp", "ep", "pp", "dcp", "pcp"))

    before_track = "raw_eager" if (script_info and script_info.get("enforce_eager")) else str(args.compare_track)
    after_track = str(args.target_track) if mode == MODE_MODELING else (
        str(args.target_track) if str(args.target_track) != "expected_ideal" else str(args.compare_track)
    )
    has_target_after = (mode == MODE_MODELING) or (rewrite_plan is not None)
    before_quant = (script_info.get("weight_quant") if script_info else None) or policy["weight_quant"]
    before_kv = (script_info.get("cache_dtype") if script_info else None) or policy["kv_cache_dtype"]
    after_quant, after_kv = policy["weight_quant"], policy["kv_cache_dtype"]
    after_idx_kv = policy["indexer_kv_dtype"]
    raw_wq_flag = str(args.weight_quant or "").strip().lower()
    if mode == MODE_MODELING and raw_wq_flag in {"checkpoint", "native"}:
        after_policy = resolve_auto_va_policy(
            model_arg, args.hardware, context_tokens=prompt_tokens,
            weight_quant="auto_va", cache_dtype=args.cache_dtype,
        )
        after_quant = after_policy["weight_quant"]
        after_kv = after_policy["kv_cache_dtype"]
        after_idx_kv = after_policy["indexer_kv_dtype"]
    explicit_spec = int(args.spec_tokens) if args.spec_tokens is not None else None
    explicit_accept = float(args.accept_len) if args.accept_len is not None else None

    if explicit_parallel:
        dp, tp, ep = int(args.dp or 1), int(args.tp or 1), int(args.ep or 1)
        pp, dcp, pcp = int(args.pp or 1), int(args.dcp or 1), int(args.pcp or 1)
        conc = int(args.concurrency or 1)
        spec_toks = int(explicit_spec or 0)
        dep_mode = str(args.deployment_mode or "auto")
        before_par = dict(dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp, pcp=pcp)
        before_conc = conc
    else:
        n_dev = int(
            args.num_devices
            or (script_info.get("num_devices") if script_info else None)
            or _estimate_auto_num_devices(policy["model_spec"], policy["sku"], after_quant)
        )
        solved_after = _auto_solve_report_layout(
            model=model_arg, sku=policy["sku"], topology_preset=topo_preset,
            num_devices=n_dev, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=args.ttft_ms,
            quant=after_quant, kv_cache_dtype=after_kv,
            indexer_kv_dtype=after_idx_kv, deployment_mode=str(args.deployment_mode or "auto"),
            feature_track=after_track,
            mtp_proposal_tokens=explicit_spec,
            mtp_acceptance_length=explicit_accept,
            apply_target_envelope=has_target_after,
            rewrite_plan=rewrite_plan if has_target_after else None,
            allow_unbound_formula=(mode == MODE_MODELING),
        )
        dp, tp, ep, pp = solved_after["dp"], solved_after["tp"], solved_after["ep"], solved_after["pp"]
        dcp, pcp = solved_after["dcp"], solved_after["pcp"]
        conc = int(args.concurrency or solved_after["concurrency"])
        spec_toks = int(explicit_spec if explicit_spec is not None else solved_after["spec_tokens"])
        dep_mode = solved_after["deployment_mode"] if args.deployment_mode == "auto" else str(args.deployment_mode)
        if script_info:
            before_par = dict(
                dp=script_info["dp"], tp=script_info["tp"], ep=script_info["ep"],
                pp=script_info["pp"], dcp=script_info["dcp"], pcp=1,
            )
            before_conc = int(script_info.get("concurrency") or conc)
        elif has_target_after and (
            before_track != after_track or before_quant != after_quant or before_kv != after_kv or rewrite_plan is not None
        ):
            solved_before = _auto_solve_report_layout(
                model=model_arg, sku=policy["sku"], topology_preset=topo_preset,
                num_devices=n_dev, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
                tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=args.ttft_ms,
                quant=before_quant, kv_cache_dtype=before_kv,
                indexer_kv_dtype=policy["indexer_kv_dtype"], deployment_mode=str(args.deployment_mode or "auto"),
                feature_track=before_track,
                mtp_proposal_tokens=explicit_spec,
                mtp_acceptance_length=explicit_accept,
                apply_target_envelope=False,
                rewrite_plan=None,
                allow_unbound_formula=False,
            )
            before_par = dict(
                dp=solved_before["dp"], tp=solved_before["tp"], ep=solved_before["ep"],
                pp=solved_before["pp"], dcp=solved_before["dcp"], pcp=solved_before["pcp"],
            )
            before_conc = int(args.concurrency or solved_before["concurrency"])
        else:
            before_par = dict(dp=dp, tp=tp, ep=ep, pp=pp, dcp=dcp, pcp=pcp)
            before_conc = conc

    accept_len = float(
        explicit_accept if explicit_accept is not None
        else (
            compute_speculative_accept_step(DEFAULT_SPECULATIVE_ACCEPT_RATE, spec_toks)
            if spec_toks > 0 else 1.0
        )
    )

    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=topo_preset, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        tpot_slo_ms=float(args.tpot_ms), ttft_slo_ms=float(args.ttft_ms or 5000.0),
        mtp_proposal_tokens=spec_toks, mtp_acceptance_length=accept_len,
        engram_cpu_offload=args.engram_offload, deployment_mode=dep_mode,
    )
    raw_before, req_b = compose_export_case(
        model_arg, policy["sku"], feature_track=before_track,
        concurrency=before_conc, quant=before_quant, kv_cache_dtype=before_kv, **before_par, **common_kw,
    )
    raw_after, req_a = compose_export_case(
        model_arg, policy["sku"], feature_track=after_track,
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
        apply_target_envelope=False,
    )
    after_bundle = build_export_bundle(
        raw_after, request=step_a, replay=replay,
        weight_quant=after_quant, cache_dtype=after_kv, engram_host_offload=args.engram_offload,
        apply_target_envelope=has_target_after,
        rewrite_plan=rewrite_plan if has_target_after else None,
        allow_unbound_formula=(mode == MODE_MODELING),
    )
    summary, brief_dict, digest_md = _emit_report_artifacts(
        before_bundle=before_bundle, after_bundle=after_bundle,
        args=args, script_info=script_info, load_designs=load_designs,
    )
    receipt: dict[str, Any] = {
        "ok": True,
        "command": "report",
        "mode": mode,
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
                "name": "zero_config_modeling_report",
                "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out",
            },
            {
                "name": "pass2_agent_rewrite_plan",
                "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --rewrite-plan ./out/plan.json --out-dir ./out",
            },
            {
                "name": "strict_simulation_mode",
                "cmd": "mindie-sim report --model qwen3-32b --hardware a3 --mode simulation --compare-track merged --target-track unmerged_pr --out-dir ./out",
            },
            {
                "name": "audit_launch_script",
                "cmd": "mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out",
            },
            {
                "name": "blind_box_hf_config",
                "cmd": "mindie-sim report --model ./my_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out",
            },
        ],
        "hardware_aliases": {
            "a5": "Ascend950DT_9582 (aliases: 950dt, 950dt-96g-pod, 950dt-144g-pod, a5-pod)",
            "a3": "Ascend910_9392 (aliases: 910c, 800i-a3-16p, a3-pod)",
            "a2": "Ascend910B4 (aliases: 910b, 800i-a2, a2-8p)",
        },
        "quant_defaults_auto_va": {
            "quarot_priority": "Always prefer *-QuaRot Eco-Tech checkpoints (Hadamard outlier rotation); for uncataloged models use msmodelslim QuaRot.",
            "a2_a3": "MoE -> w4a8 + c8(int8) KV (>=32k); Dense -> w8a8 + c8 KV (>=32k)",
            "a5": "MoE -> mxfp8_mxfp4 + fp8 KV/Indexer; Dense -> mxfp8 + fp8 KV (>=32k)",
        },
        "agent_2_pass_loop": [
            "Pass 1: Run `mindie-sim report` (or `--from-script serve.sh`) -> read `<prefix>_agent_brief.json` (<8 KB), inspecting `semantic_audit_triggers` & `macro_wall_trigger`.",
            "Pass 2: Inspect cited source code, emit `plan.json` / `designs.json`, and re-run `mindie-sim report --rewrite-plan plan.json --kernel-designs-json designs.json`.",
            "Mode Split: `--mode simulation` enforces Layer-2 bound implementations only; `--mode modeling` projects Layer-1 formula-driven ceiling targets.",
        ],
    }
    if not getattr(args, "emit_json", False):
        md_lines = [
            "# MindIE-Sim Quickstart (Agent-First Ascend NPU Performance Compass)",
            "",
            "## 1. Golden Commands",
            "```bash",
            "# (A) Zero-config Iso-SLA modeling report (Before sim vs After SLA target ceiling)",
            "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --context 128k --out-dir ./out",
            "# (B) Pass-2 Agent semantic rewrite plan + custom kernel design injection",
            "mindie-sim report --model kimi-k3.1 --hardware 950dt --num-devices 32 --rewrite-plan ./out/plan.json --out-dir ./out",
            "# (C) Strict Simulation Mode (Layer-2 bound implementations only, zero unbound formula ops)",
            "mindie-sim report --model qwen3-32b --hardware a3 --mode simulation --compare-track merged --target-track unmerged_pr --out-dir ./out",
            "# (D) Audit & auto-tune an existing vllm-ascend serve.sh (static zero-exec parse)",
            "mindie-sim report --from-script ./serve.sh --hardware a3 --out-dir ./out",
            "# (E) Blind-box uncataloged HuggingFace config.json or model directory",
            "mindie-sim report --model ./new_hf_model/config.json --hardware a3 --tpot-ms 20 --out-dir ./out",
            "```",
            "",
            "## 2. Hardware Aliases & Production Quantization (`auto_va`, QuaRot-First)",
            "- **Hardware**: `950dt`/`a5`/`950dt-96g-pod`/`950dt-144g-pod` | `a3`/`910c`/`a3-pod` | `a2`/`910b`/`800i-a2`",
            "- **A2 / A3 (`64G`)**: MoE -> `w4a8` + `c8` KV (`>=32k`); Dense -> `w8a8` + `c8` KV",
            "- **A5 (`96G/144G`)**: MoE -> `mxfp8_mxfp4` + `fp8` KV/Indexer; Dense -> `mxfp8` + `fp8` KV",
            "",
            "## 3. Agent 2-Pass Co-Design Loop",
            "1. **Pass 1**: Run `mindie-sim report` -> read `<prefix>_agent_brief.json` (`< 8 KB`, zero `.xlsx` XML scraping).",
            "2. **Inspect**: Check `semantic_audit_triggers`, `macro_wall_trigger` (`HBM_CAPACITY_WALL` vs `LATENCY_COMPUTE_WALL`), and Sheet 4 `card_id`s.",
            "3. **Pass 2**: Re-run with `--rewrite-plan ./out/plan.json --kernel-designs-json ./out/designs.json` to seal exact conserved `.xlsx` & `trace_view.json`.",
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
