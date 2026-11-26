"""Turnkey ``mindie-sim report`` and ``mindie-sim quickstart`` CLI commands.

[Role] Deliver zero-boilerplate, Agent-first evaluation via:
  (1) ``mindie-sim report``: enforces explicit ``--deployment-mode``
      (``{colocated, pd_disaggregated, pd_prefill, pd_decode}`` or unambiguous
      ``--from-script`` hint; zero silent ``auto`` guessing and zero ``5000.0``
      TTFT coercion), performs domain-aware ``RuntimeStackAdapter`` routing via
      ``stacks.registry.resolve_stack_for_scenario`` (defaulting LLM/VLM to
      ``vllm-ascend`` and Diffusion/DiT to ``diffusion-native``), parses
      ``--from-script <serve.sh>`` statically through ``adapter.launch_codec``,
      resolves checkpoints once without bare-string re-expansion, accepts
      Pass-2 ``--rewrite-plan <plan.json>`` & ``--mode {modeling,simulation}``,
      and writes ``*_comparison.xlsx``, ``*_trace_view.json``, ``< 8 KB``
      ``*_agent_brief.json``, optional ``*_optimized_serve.sh`` (strictly when
      ``--from-script`` is supplied), and a stdout Markdown digest;
  (2) ``mindie-sim quickstart [--json]``: prints golden commands, aliases, and
      the 2-pass Agent co-design loop.
[Invariants] Never executes user launch scripts; never duplicates
  ``catalog.quant.resolve_auto_va_quant`` or re-expands curated checkpoints via
  bare strings; never coerces ``ttft_slo_ms=None`` to ``5000.0``; script-free
  mode passes ``script_audit=None`` and ``serve_script_supplied=False`` with
  zero ``optimized_serve.sh``; always re-raises ``TimeoutError`` before
  ``OSError``.
[Timeouts & Locks] Bounded 50 MiB metadata reads and single-pass O(Archetypes)
  export delegation; no locks or device contexts.
[Side-Effect Policy] Writes artifacts only into caller-specified ``--out-dir``
  (and optional ``--receipt`` path).
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace as dc_replace
import importlib
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.compose.inputs import (
    DERIVED_ASCEND_SKUS, _QUANT_ALIASES,
    _arch_key, _resolve_hardware, _resolve_model_spec,
    default_work_unit_for_domain, infer_workload_domain,
)
from mindie_sim.export import (
    AGENT_BRIEF_SCHEMA, DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEPLOYMENT_MODE_LABELS, MODE_MODELING, MODE_SIMULATION, US_PER_MS,
    align_before_after_operators, build_default_comparison_request,
    build_export_bundle, compose_export_case, export_agent_brief,
    export_comparison_xlsx, export_msprof_trace, format_executive_digest_markdown,
    project_two_step_gain, sanitize_public_text,
)
from mindie_sim.export.action_plan import build_comparison_plan
from mindie_sim.export.models import NS_PER_MS, bundle_step_int_ns
from mindie_sim.prediction.efficiency.sla import _prefill_chunk_schedule
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.scenario import validate_deployment_mode
from mindie_sim.schema.stack import STACK_TRACKS, LaunchConfigIngress
from mindie_sim.schema.workload import RequestView
from mindie_sim.stacks.registry import resolve_stack_for_scenario

REPORT_COMMANDS = frozenset({"report", "quickstart"})
REPORT_BOOLEAN_FLAGS = frozenset({"--json", "--no-trace", "--no-xlsx"})
MAX_SCRIPT_BYTES = 50 * 1024 * 1024
DEFAULT_REPORT_PROMPT_TOKENS = 32768
AUTO_SIZING_WEIGHT_FRACTION = 0.65
STEP_MS_ROUND_DIGITS = 3
TPS_ROUND_DIGITS = 2
SPEEDUP_ROUND_DIGITS = 2
TOKEN_SHORTHAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$")
TOKEN_MULTIPLIERS = {"": 1, "k": 1024, "m": 1024 * 1024}
SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")
_URL_PREFIXES = ("http://", "https://")
_PARALLEL_AXES = ("dp", "tp", "ep", "pp", "dcp", "pcp")
_DEPLOYMENT_MODE_CHOICES = ("colocated", "pd_disaggregated", "pd_prefill", "pd_decode")
_KV_POOLING_CHOICES = ("auto", "off", "pooled_dram_ssd", "layerwise_overlap_d2d", "direct_blocking_d2d")


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
    except ModuleNotFoundError:
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


def _unfreeze_ingress_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {k: _unfreeze_ingress_value(v) for k, v in value}
        return [_unfreeze_ingress_value(item) for item in value]
    return value


def _resolve_launch_codec(requested_stack: str | None, hardware: str) -> Any:
    sku, backend, _ = _resolve_hardware(hardware)
    hw_sel = sku if backend == "ascend" else backend
    adapter = resolve_stack_for_scenario(
        requested_stack=requested_stack,
        workload_domain="llm",
        work_unit="prefill_decode_step",
        hardware_family=hw_sel,
    )
    codec = adapter.launch_codec
    if codec is None:
        raise ModelingError(
            f"launch_script_codec_unsupported_by_stack:{adapter.capabilities().stack_id}"
        )
    return codec


def resolve_auto_va_policy(
    model: str | Path | Mapping[str, Any], hardware: str, *,
    context_tokens: int = DEFAULT_REPORT_PROMPT_TOKENS,
    weight_quant: str | None = "auto_va", cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Resolve chip-matched Ascend production quantization via single SSOT without re-expansion."""
    from mindie_sim.catalog.quant import resolve_auto_va_quant

    model_spec = _resolve_model_spec(model)
    sku, backend, preset = _resolve_hardware(hardware)
    arch = _arch_key(sku)
    raw_q = (str(weight_quant).strip().lower() if weight_quant is not None else "auto_va")
    explicit_cache = _QUANT_ALIASES.get(str(cache_dtype).strip().lower(), str(cache_dtype).strip().lower()) if cache_dtype else None

    if raw_q in {"checkpoint", "native"}:
        q_val, compose_q, kv_val, idx_val, source = None, None, explicit_cache, None, "checkpoint_native"
    elif raw_q in {"", "auto", "auto_va"}:
        q_val, hook_kv, idx_val = resolve_auto_va_quant(model_spec, sku, context_tokens=int(context_tokens))
        compose_q, kv_val, source = "auto_va", (explicit_cache or hook_kv), "auto_va_production"
    else:
        norm_q = _QUANT_ALIASES.get(raw_q, raw_q)
        q_val, compose_q, kv_val, idx_val, source = norm_q, norm_q, explicit_cache, None, "explicit_flag"

    return {
        "model_spec": model_spec, "sku": sku, "backend": backend, "arch": arch,
        "topology_preset": preset or None,
        "weight_quant": q_val, "compose_quant": compose_q,
        "kv_cache_dtype": kv_val, "indexer_kv_dtype": idx_val,
        "policy_source": source, "quarot_guidance": _lookup_quarot_checkpoint_hint(model_spec.name, arch, q_val),
    }


def parse_script_for_report(
    path: str | Path,
    *,
    codec: Any = None,
    requested_stack: str | None = None,
    hardware: str = "a3",
) -> dict[str, Any]:
    """Statically parse a launch script through ``adapter.launch_codec``."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"launch_script_not_found:{target}")
    if target.stat().st_size > MAX_SCRIPT_BYTES:
        raise ModelingError("launch_script_exceeds_50_mib_limit")
    active_codec = codec if codec is not None else _resolve_launch_codec(requested_stack, hardware)
    ingress: LaunchConfigIngress = active_codec.parse_launch_config(target)
    flags = {k: _unfreeze_ingress_value(v) for k, v in ingress.raw_stack_flags}
    limits = {k: _unfreeze_ingress_value(v) for k, v in ingress.runtime_limits}
    spec_cfg = flags.get("speculative_config") if isinstance(flags.get("speculative_config"), Mapping) else {}
    spec_k = spec_cfg.get("num_speculative_tokens")
    model_ref = str(ingress.model_ref or "")
    local_cfg = model_ref if (model_ref and Path(model_ref).expanduser().exists()) else None
    par = ingress.parallel
    inferred_devs = int(limits.get("inferred_num_devices") or max(1, par.dp * par.tp * par.pp * par.pcp))
    return {
        "script_path": str(target), "ingress": ingress, "codec": active_codec,
        "deployment_mode_hint": ingress.deployment_mode_hint,
        "model": local_cfg or flags.get("inferred_catalog_model") or model_ref or None,
        "weight_quant": flags.get("inferred_quant") or None,
        "cache_dtype": flags.get("kv_cache_dtype") or None,
        "dp": int(par.dp), "tp": int(par.tp), "ep": int(par.ep),
        "pp": int(par.pp), "dcp": int(par.cp), "pcp": int(par.pcp),
        "num_devices": inferred_devs,
        "max_model_len": int(limits["max_model_len"]) if limits.get("max_model_len") else None,
        "concurrency": int(limits["max_num_seqs"]) if limits.get("max_num_seqs") else None,
        "spec_tokens": int(spec_k) if isinstance(spec_k, int) and spec_k >= 0 else None,
        "enforce_eager": bool(flags.get("enforce_eager")),
        "env_vars": dict(flags.get("env_vars") or {}),
        "compilation_config": dict(flags.get("compilation_config") or {}),
        "additional_config": dict(flags.get("additional_config") or {}),
    }


def _apply_script_overrides_to_case(raw_case: dict[str, Any], script_info: Mapping[str, Any]) -> dict[str, Any]:
    rt = dict(raw_case.get("runtime") or {})
    meta = dict(rt.get("metadata") or {})
    if script_info.get("env_vars"):
        meta["script_env"] = dict(script_info["env_vars"])
    if script_info.get("compilation_config"):
        meta["script_compilation_config"] = dict(script_info["compilation_config"])
    if script_info.get("additional_config"):
        meta["script_additional_config"] = dict(script_info["additional_config"])
    if script_info.get("enforce_eager"):
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
    from mindie_sim.catalog.quant import QUANTIZATION_PRESETS, quant_preset
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.catalog.weights import sharded_weight_bytes
    from mindie_sim.schema.parallel import ParallelSpec

    entry = {**ascend_profiles(), **DERIVED_ASCEND_SKUS}.get(sku, {})
    raw_mem = entry.get("memory_bytes") or (entry.get("device") or {}).get("memory_bytes")
    if not raw_mem or float(raw_mem) <= 0.0:
        raise ModelingError(f"missing_device_memory_bytes:{sku}")
    mem_b = float(raw_mem)
    eff_spec = model_spec
    if isinstance(weight_quant, QuantSpec):
        eff_spec = dc_replace(model_spec, quant=weight_quant, quantization=str(weight_quant.name))
    elif isinstance(weight_quant, str):
        qkey = _QUANT_ALIASES.get(weight_quant.strip().lower(), weight_quant.strip().lower())
        if qkey in QUANTIZATION_PRESETS:
            qspec = quant_preset(qkey, model=model_spec)
            eff_spec = dc_replace(model_spec, quant=qspec, quantization=str(qspec.name))
    total_w = float(sharded_weight_bytes(eff_spec, ParallelSpec()))
    target_per_die = max(1.0, mem_b * AUTO_SIZING_WEIGHT_FRACTION)
    min_dies = max(1, int((total_w + target_per_die - 1.0) // target_per_die))
    count = 1
    while count < min_dies:
        count *= 2
    return count


def _extract_infeasible_details(
    solved: Mapping[str, Any],
    *,
    deployment_mode: str,
    requested_concurrency: int | None,
) -> dict[str, Any]:
    """Extract honest infeasible outcome & limiting constraint without fabricating B=1."""
    outcome = solved.get("outcome") if isinstance(solved.get("outcome"), Mapping) else {}
    raw_cands = [c for c in (solved.get("candidates") or ()) if isinstance(c, Mapping)]
    max_b_vram = max(
        (int(c.get("b_vram_max") or c.get("b_vram") or 0) for c in raw_cands),
        default=0,
    )
    if isinstance(outcome.get("physical_b_vram_max"), int) and not isinstance(outcome.get("physical_b_vram_max"), bool):
        max_b_vram = max(max_b_vram, int(outcome["physical_b_vram_max"]))
    reason = ""
    for candidate_val in (
        outcome.get("limiting_constraint"),
        solved.get("limiting_constraint"),
        solved.get("infeasible_reason"),
    ):
        if isinstance(candidate_val, str) and candidate_val.strip():
            reason = sanitize_public_text(candidate_val)
            break
    if not reason:
        for cand in raw_cands:
            for key in ("limiting_constraint", "infeasible_reason", "infeasible", "dcp_codomain_reason"):
                val = cand.get(key)
                if isinstance(val, str) and val.strip():
                    reason = sanitize_public_text(val)
                    break
            if reason:
                break
    if not reason:
        reason = "hbm_capacity_oom" if max_b_vram <= 0 else "sla_latency_bound"
    status = sanitize_public_text(outcome.get("status") or "infeasible") or "infeasible"
    return {
        "status": status,
        "deployment_mode": deployment_mode,
        "limiting_constraint": reason,
        "physical_b_vram_max": int(max_b_vram),
        "feasible_b_per_dp": 0,
        "requested_concurrency": (int(requested_concurrency) if requested_concurrency is not None else None),
    }


def _auto_solve_report_layout(
    *, model: str, sku: str, topology_preset: str | None, num_devices: int,
    prompt_tokens: int, output_tokens: int, tpot_slo_ms: float | None, ttft_slo_ms: float | None,
    quant: str | None, kv_cache_dtype: str | None, indexer_kv_dtype: str | None,
    deployment_mode: str, kv_pooling: str | None = None,
    runtime_stack: str | None = None, feature_track: str | None = None,
    mtp_proposal_tokens: int | None = None, mtp_acceptance_length: float | None = None,
    apply_target_envelope: bool = False, rewrite_plan: Any = None,
    allow_unbound_formula: bool = True, phase: str = "decode",
    pinned_parallel: Mapping[str, int] | None = None, max_sequences_per_dp: int | None = None,
) -> dict[str, Any]:
    """Auto-solve HBM-safe parallelism + concurrency via ``solve_optimal_deployment``."""
    from mindie_sim.solve import _resolve_kv_pooling_mode
    from mindie_sim.sweep.pd import solve_optimal_deployment

    validated_mode = validate_deployment_mode(deployment_mode)
    if validated_mode == "pd_disaggregated" and int(num_devices) < 2:
        raise ModelingError(f"pd_disaggregated_requires_at_least_two_devices:{int(num_devices)}")
    dep_for_solver = "pd_disaggregated" if validated_mode == "pd_disaggregated" else "colocated"
    compose_overrides: dict[str, Any] = {}
    if validated_mode in {"pd_prefill", "pd_decode"}:
        compose_overrides["role"] = validated_mode
    eff_ttft = None if validated_mode == "pd_decode" else ttft_slo_ms
    solve_kw: dict[str, Any] = dict(
        num_devices=int(num_devices), prompt_tokens=int(prompt_tokens), output_tokens=int(output_tokens),
        ttft_sla_ms=eff_ttft,
        tpot_sla_ms=(float(tpot_slo_ms) if tpot_slo_ms is not None else None),
        deployment=dep_for_solver,
        pd_transfer_mode=_resolve_kv_pooling_mode(kv_pooling) if validated_mode == "pd_disaggregated" else None,
        quant=quant, kv_cache_dtype=kv_cache_dtype, indexer_kv_dtype=indexer_kv_dtype,
        topology_preset=topology_preset, allow_unconstrained_ttft=(eff_ttft is None),
        runtime_stack=runtime_stack, feature_track=feature_track,
        mtp_proposal_tokens=mtp_proposal_tokens, mtp_acceptance_length=mtp_acceptance_length,
        apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
    )
    if compose_overrides:
        solve_kw["compose_overrides"] = compose_overrides
    if pinned_parallel:
        solve_kw["pinned_parallel"] = dict(pinned_parallel)
    if max_sequences_per_dp is not None:
        solve_kw["max_sequences_per_dp"] = int(max_sequences_per_dp)
    solved = solve_optimal_deployment(model, sku, **solve_kw)
    is_feasible = bool(solved.get("feasible", False)) and isinstance(solved.get("best_configuration"), Mapping)
    best = dict(solved["best_configuration"]) if is_feasible else {}
    infeasible_outcome = (
        None if is_feasible else _extract_infeasible_details(
            solved, deployment_mode=validated_mode, requested_concurrency=max_sequences_per_dp,
        )
    )
    par = best.get("prefill_parallel" if phase == "prefill" else "decode_parallel") or best.get("parallel") or {}
    if is_feasible and not par.get("tp"):
        raise ModelingError("solver_best_configuration_missing_explicit_tp")
    tp_val = int(par["tp"]) if is_feasible else 0
    dp_val = int(par.get("dp", 1) or 1) if is_feasible else 0
    return {
        "dp": dp_val, "tp": tp_val,
        "ep": int(par.get("ep", dp_val * tp_val) or 1) if is_feasible else 0,
        "pp": int(par.get("pp", 1) or 1) if is_feasible else 0,
        "dcp": int(par.get("cp", par.get("dcp", 1)) or 1) if is_feasible else 0,
        "pcp": int(par.get("pcp", 1) or 1) if is_feasible else 0,
        "concurrency": (max(1, int(best.get("b_feasible") or 1)) if is_feasible else 0),
        "spec_tokens": int(best.get("mtp_proposal_tokens", 0) or 0),
        "deployment_mode": validated_mode,
        "solver_feasible": is_feasible,
        "infeasible_outcome": infeasible_outcome,
        "solved": solved,
    }


def _report_column(
    layout: Mapping[str, Any] | None, *, model: str, sku: str, phase: str, track: str,
    quant: str | None, kv: str | None, idx_kv: str | None, par: Mapping[str, int],
    conc: int, common_kw: Mapping[str, Any],
    spec_tokens: int | None = None, accept_len: float | None = None,
) -> tuple[dict[str, Any], StepRequest, dict[str, Any]]:
    prompt_tokens, output_tokens = int(common_kw["prompt_tokens"]), int(common_kw["output_tokens"])
    if layout is not None and layout.get("solver_feasible", True) and layout.get("solved") is not None:
        from mindie_sim.solve import _bundle_inputs_for_solution

        return _bundle_inputs_for_solution(
            layout["solved"], model=model, hardware=sku, runtime_stack=common_kw.get("runtime_stack"),
            base_kw=common_kw, feature_track=track, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens, scope="model", phase=phase,
        )
    col_kw = dict(common_kw)
    col_kw.pop("phase", None)
    if spec_tokens is not None:
        col_kw["mtp_proposal_tokens"] = int(spec_tokens)
    if accept_len is not None:
        col_kw["mtp_acceptance_length"] = float(accept_len)
    raw, req = compose_export_case(
        model, sku, feature_track=track, concurrency=max(1, int(conc)),
        quant=quant, kv_cache_dtype=kv, indexer_kv_dtype=idx_kv, phase=phase, **dict(par), **col_kw,
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


def _bundle_facts_for_codec(
    bundle: Any, *, ingress: LaunchConfigIngress | None = None, role: str = "colocated",
) -> dict[str, Any]:
    mem = bundle.memory_ledger
    eff = dict(getattr(bundle, "effective_config", {}) or {})
    return {
        "model_name": bundle.model_name,
        "model_ref": (ingress.model_ref if ingress is not None and ingress.model_ref else f"/weights/{bundle.model_name}"),
        "hardware_sku": bundle.hardware_sku,
        "hw_family": eff.get("arch_family") or bundle.hardware_sku,
        "role": role,
        "deployment_mode": role,
        "phase": bundle.phase,
        "dp": mem.dp, "tp": mem.tp, "ep": mem.ep, "pp": mem.pp,
        "dcp": mem.dcp, "pcp": mem.pcp,
        "concurrency": mem.concurrency,
        "max_num_seqs": mem.concurrency,
        "seq_len": bundle.prompt_tokens + bundle.output_tokens,
        "max_model_len": bundle.prompt_tokens + bundle.output_tokens,
        "context_tokens": bundle.prompt_tokens,
        "weight_format": mem.weight_quant,
        "kv_cache_dtype": mem.cache_dtype,
        "spec_proposals": bundle.spec_proposals,
        "spec_method": bundle.spec_method,
        "has_mla": bool(eff.get("has_mla", False)),
        "fits_hbm": bool(mem.fits_hbm),
        "peak_hbm_bytes": float(mem.max_stage_hbm_bytes),
        "step_span_us": float(bundle.step_span_us),
        "multistream_overlap_shared_expert": bool(eff.get("multistream_overlap_shared_expert", False)),
        "enable_fused_mc2": int(eff.get("enable_fused_mc2", 0) or 0),
    }


def add_report_commands(subparsers) -> None:
    """Register turnkey ``report`` and ``quickstart`` commands."""
    rep = subparsers.add_parser("report", help="turnkey one-shot evaluation -> .xlsx + trace_view.json + agent_brief.json")
    rep.add_argument("--model", default=None, help="catalog model, alias, HuggingFace config.json, or local directory")
    rep.add_argument("--from-script", default=None, help="audit & optimize an existing serve.sh launch script (zero-exec)")
    rep.add_argument("--hardware", default="a3", help="hardware SKU or alias (950dt, a5, a3, 910c, a2, 910b)")
    rep.add_argument("--stack", default=None, help="runtime stack (vllm-ascend, ideal-target, raw-eager, diffusion-native, or /path/to/manifest.json)")
    rep.add_argument("--topology-preset", default=None, help="optional topology preset override (a5_pod, a3_16p)")
    rep.add_argument("--num-devices", type=int, default=None, help="total NPU dies (auto-sized when omitted)")
    rep.add_argument("--phase", choices=("prefill", "decode"), default=None)
    for flag in ("--dp", "--tp", "--dcp", "--ep", "--pp", "--pcp"):
        rep.add_argument(flag, type=int, default=None)
    rep.add_argument("--prompt-tokens", "--context", dest="prompt_tokens", type=parse_token_shorthand, default=None)
    rep.add_argument("--output-tokens", type=parse_token_shorthand, default=DEFAULT_SCENARIO_OUTPUT_TOKENS)
    rep.add_argument("--concurrency", type=int, default=None, help="sequences per DP replica (auto-solved when omitted)")
    rep.add_argument("--tpot-ms", "--tpot-slo-ms", dest="tpot_ms", type=float, default=None)
    rep.add_argument("--ttft-ms", "--ttft-slo-ms", dest="ttft_ms", type=float, default=None)
    rep.add_argument("--spec-tokens", "--mtp", "--mtp-tokens", dest="spec_tokens", type=int, default=None)
    rep.add_argument("--accept-len", "--mtp-accept", dest="accept_len", type=float, default=None)
    rep.add_argument("--mode", choices=(MODE_MODELING, MODE_SIMULATION), default=MODE_MODELING)
    rep.add_argument(
        "--compare-track",
        choices=("eager", "shipping", "proposed", "target"),
        default="shipping",
        help="Before track (default: shipping)",
    )
    rep.add_argument(
        "--target-track",
        choices=("eager", "shipping", "proposed", "target"),
        default="target",
        help="After track (default: target)",
    )
    rep.add_argument("--rewrite-plan", default=None, help="Pass-2 Agent semantic rewrite plan JSON")
    rep.add_argument("--weight-quant", default="auto_va", help="weight quant preset (default: auto_va)")
    rep.add_argument("--cache-dtype", default=None, help="KV cache dtype override (bf16/c8/int8/fp8_e4m3)")
    rep.add_argument("--engram-offload", dest="engram_offload", action="store_true", default=None)
    rep.add_argument("--no-engram-offload", dest="engram_offload", action="store_false")
    rep.add_argument(
        "--deployment-mode",
        choices=_DEPLOYMENT_MODE_CHOICES,
        default=None,
        help="explicit deployment mode: colocated | pd_disaggregated | pd_prefill | pd_decode",
    )
    rep.add_argument(
        "--kv-pooling",
        choices=_KV_POOLING_CHOICES,
        default=None,
        help="optional PD KV transfer/pooling policy for pd_disaggregated",
    )
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


def _enrich_two_step_gain(
    audit_dict: dict[str, Any],
    *,
    before_bundle: Any,
    tuned_bundle: Any | None,
    after_bundle: Any,
) -> dict[str, Any]:
    two_step = dict(audit_dict.get("two_step_gain") or {})
    orig_fits = bool(before_bundle.memory_ledger.fits_hbm)
    b_step = float(before_bundle.step_span_us) / US_PER_MS
    a_step = float(after_bundle.step_span_us) / US_PER_MS
    t_step = (float(tuned_bundle.step_span_us) / US_PER_MS) if tuned_bundle is not None else None
    a_acc = max(1.0, float(after_bundle.avg_accepted_tokens))
    a_tpot = (
        (a_step / a_acc) + float(getattr(after_bundle, "handoff_tpot_ms", 0.0) or 0.0)
        if after_bundle.phase == "decode" else a_step
    )
    two_step["step2_deep_kernel_and_framework"] = {
        "label": "Step 2: Deep Kernel & Framework Optimization",
        "original_fits_hbm": orig_fits,
        "before_step_ms": round(b_step, STEP_MS_ROUND_DIGITS),
        "after_step_ms": round(a_step, STEP_MS_ROUND_DIGITS),
        "saved_ms": (round(b_step - a_step, STEP_MS_ROUND_DIGITS) if orig_fits else None),
        "saved_from_tuned_ms": (
            round(t_step - a_step, STEP_MS_ROUND_DIGITS)
            if (t_step is not None and t_step > 0.0) else None
        ),
        "after_effective_tpot_ms": round(a_tpot, STEP_MS_ROUND_DIGITS),
        "after_single_die_tps": round(float(after_bundle.throughput_per_die_tokens_per_s), TPS_ROUND_DIGITS),
        "speedup_vs_original": (
            round(b_step / a_step, SPEEDUP_ROUND_DIGITS) if (orig_fits and a_step > 0.0) else None
        ),
    }
    audit_dict["two_step_gain"] = project_two_step_gain(two_step)
    return audit_dict


def _emit_report_artifacts(
    *, before_bundle: Any, after_bundle: Any, tuned_bundle: Any | None,
    args: argparse.Namespace, script_info: Mapping[str, Any] | None,
    dep_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = args.prefix or f"{after_bundle.model_name}_{after_bundle.hardware_sku}_{after_bundle.phase}"
    prefix = SAFE_PREFIX_RE.sub("_", raw_prefix).strip("_") or "mindie_sim_report"

    generated: dict[str, str] = {}
    summary: dict[str, Any] = {}
    audit_dict: dict[str, Any] | None = None
    if script_info and isinstance(script_info.get("ingress"), LaunchConfigIngress):
        codec = script_info["codec"]
        ingress = script_info["ingress"]
        before_facts = _bundle_facts_for_codec(before_bundle, ingress=ingress, role=dep_mode)
        tuned_facts = _bundle_facts_for_codec(tuned_bundle or after_bundle, ingress=ingress, role=dep_mode)
        opt_sh_path = out_dir / f"{prefix}_optimized_serve.sh"
        opt_sh_path.write_text(codec.render_launch_config(tuned_facts), encoding="utf-8", newline="\n")
        generated["optimized_serve_sh"] = opt_sh_path.name
        audit_dict = dict(codec.audit_launch_config(ingress, before_facts, tuned_facts))
        audit_dict["optimized_script_path"] = opt_sh_path.name
        audit_dict = _enrich_two_step_gain(
            audit_dict, before_bundle=before_bundle, tuned_bundle=tuned_bundle, after_bundle=after_bundle,
        )

    aligned = align_before_after_operators(before_bundle, after_bundle)
    shared_plan = build_comparison_plan(
        before_bundle, after_bundle, aligned, script_audit=audit_dict,
    )
    cmp_req = build_default_comparison_request(before_bundle, after_bundle)

    if not getattr(args, "no_xlsx", False):
        xlsx_path = out_dir / f"{prefix}_comparison.xlsx"
        summary["xlsx"] = export_comparison_xlsx(
            before_bundle, xlsx_path, after_bundle=after_bundle,
            comparison_request=cmp_req,
            comparison_plan=shared_plan,
            script_audit=audit_dict,
        ).to_dict()
        generated["comparison_xlsx"] = xlsx_path.name
    if not getattr(args, "no_trace", False):
        trace_path = out_dir / f"{prefix}_trace_view.json"
        summary["trace"] = export_msprof_trace(
            after_bundle, trace_path,
            include_pipe_lanes=bool(getattr(args, "include_pipe_lanes", True)),
            include_counters=bool(getattr(args, "include_counters", True)),
            include_overlap=bool(getattr(args, "include_overlap", True)),
        ).to_dict()
        generated["trace_view_json"] = trace_path.name

    brief_path = out_dir / f"{prefix}_agent_brief.json"
    generated["agent_brief_json"] = brief_path.name
    brief = export_agent_brief(
        before_bundle, after_bundle, brief_path,
        generated_files=generated, script_audit=audit_dict,
        comparison_plan=shared_plan,
    )
    brief_dict = dict(brief) if isinstance(brief, Mapping) else {}
    return summary, brief_dict, format_executive_digest_markdown(brief_dict)


def _report_tracks(mode: str, *, compare_track: str, target_track: str, enforce_eager: bool) -> tuple[str, str]:
    raw_before = "eager" if enforce_eager else str(compare_track or "").strip().lower()
    raw_after = str(target_track or "").strip().lower()
    if raw_before not in STACK_TRACKS:
        raise ModelingError(f"invalid_stack_track:{raw_before!r}")
    if raw_after not in STACK_TRACKS:
        raise ModelingError(f"invalid_stack_track:{raw_after!r}")
    before = "shipping" if (mode == MODE_SIMULATION and raw_before == "target") else raw_before
    after = before if (mode == MODE_SIMULATION and raw_after == "target") else raw_after
    return before, after


def _accept_len_for(spec_tokens: int, explicit_accept: float | None) -> float:
    if explicit_accept is not None:
        return max(1.0, float(explicit_accept))
    del spec_tokens
    return 1.0


def _resolve_report_mode_and_phase(
    args: argparse.Namespace, script_info: Mapping[str, Any] | None,
) -> tuple[str, str, float | None]:
    raw_dep = args.deployment_mode or (script_info.get("deployment_mode_hint") if script_info else None)
    dep_mode = validate_deployment_mode(raw_dep)
    if dep_mode == "pd_decode" and args.ttft_ms is not None:
        raise ModelingError(
            "pd_decode_ttft_unavailable: --ttft-ms is unavailable in pd_decode mode (must be None)"
        )
    req_phase = getattr(args, "phase", None)
    if dep_mode == "pd_decode" and req_phase == "prefill":
        raise ModelingError("pd_decode_incompatible_with_prefill_phase")
    if dep_mode == "pd_prefill":
        if req_phase == "decode":
            raise ModelingError("pd_prefill_incompatible_with_decode_phase")
        if args.tpot_ms is not None:
            raise ModelingError("pd_prefill_tpot_unavailable: --tpot-ms is unavailable in pd_prefill mode")
    eff_phase = str(req_phase) if req_phase else ("prefill" if dep_mode == "pd_prefill" else "decode")
    eff_ttft = None if (dep_mode == "pd_decode" or args.ttft_ms is None) else float(args.ttft_ms)
    return dep_mode, eff_phase, eff_ttft


def _solve_report_columns(
    *, args: argparse.Namespace, model_arg: str, policy: Mapping[str, Any],
    script_info: Mapping[str, Any] | None, dep_mode: str, eff_phase: str,
    eff_ttft: float | None, mode: str, rewrite_plan: Any, topo_preset: str | None,
    prompt_tokens: int, output_tokens: int, before_track: str, after_track: str,
    before_quant: str | None, before_kv: str | None,
    after_quant: str | None, after_kv: str | None, after_idx_kv: str | None,
) -> dict[str, Any]:
    pinned_parallel = {axis: int(getattr(args, axis)) for axis in _PARALLEL_AXES if getattr(args, axis, None) is not None}
    full_explicit = (
        args.num_devices is None and args.dp is not None and args.tp is not None
        and (args.ep is not None or not _is_moe_model(policy["model_spec"])) and args.concurrency is not None
    )
    has_target_after = (mode == MODE_MODELING) or (rewrite_plan is not None)
    explicit_spec = int(args.spec_tokens) if args.spec_tokens is not None else None
    explicit_accept = float(args.accept_len) if args.accept_len is not None else None
    layout_before = layout_tuned = layout_after = None

    if full_explicit:
        after_par = {axis: max(1, int(pinned_parallel.get(axis, 1))) for axis in _PARALLEL_AXES}
        conc = max(1, int(args.concurrency))
        after_spec = max(0, int(explicit_spec or 0))
        if script_info:
            before_par = {axis: max(1, int(script_info.get(axis, after_par[axis]) or 1)) for axis in _PARALLEL_AXES}
            before_conc = max(1, int(script_info.get("concurrency") or conc))
            before_spec = max(0, int(script_info["spec_tokens"] if script_info.get("spec_tokens") is not None else after_spec))
            tuned_par, tuned_conc = dict(after_par), conc
        else:
            before_par, before_conc, before_spec = dict(after_par), conc, after_spec
            tuned_par, tuned_conc = dict(after_par), conc
    else:
        n_dev = int(args.num_devices or (script_info.get("num_devices") if script_info else None) or _estimate_auto_num_devices(policy["model_spec"], policy["sku"], policy["weight_quant"]))
        common_solve = dict(
            model=model_arg, sku=policy["sku"], topology_preset=topo_preset, num_devices=n_dev,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            tpot_slo_ms=(float(args.tpot_ms) if args.tpot_ms is not None else None),
            ttft_slo_ms=eff_ttft, deployment_mode=dep_mode, kv_pooling=getattr(args, "kv_pooling", None),
            runtime_stack=getattr(args, "stack", None), mtp_proposal_tokens=explicit_spec,
            mtp_acceptance_length=explicit_accept, phase=eff_phase,
            pinned_parallel=pinned_parallel or None,
            max_sequences_per_dp=int(args.concurrency) if args.concurrency is not None else None,
        )
        layout_after = _auto_solve_report_layout(
            **common_solve, quant=after_quant, kv_cache_dtype=after_kv, indexer_kv_dtype=after_idx_kv,
            feature_track=after_track, apply_target_envelope=has_target_after,
            rewrite_plan=rewrite_plan if has_target_after else None, allow_unbound_formula=(mode == MODE_MODELING),
        )
        after_par = {k: layout_after[k] for k in _PARALLEL_AXES}
        conc = int(layout_after["concurrency"])
        after_spec = int(explicit_spec if explicit_spec is not None else layout_after["spec_tokens"])

        if not layout_after["solver_feasible"]:
            before_par, before_conc, before_spec = dict(after_par), 0, after_spec
            tuned_par, tuned_conc = dict(after_par), 0
        elif script_info:
            layout_tuned = _auto_solve_report_layout(
                **common_solve, quant=after_quant, kv_cache_dtype=after_kv, indexer_kv_dtype=after_idx_kv,
                feature_track="shipping", apply_target_envelope=False, rewrite_plan=None, allow_unbound_formula=False,
            )
            tuned_par, tuned_conc = {k: layout_tuned[k] for k in _PARALLEL_AXES}, int(layout_tuned["concurrency"])
            before_par = dict(dp=script_info["dp"], tp=script_info["tp"], ep=script_info["ep"], pp=script_info["pp"], dcp=script_info["dcp"], pcp=script_info["pcp"])
            before_conc = int(script_info.get("concurrency") or max(1, tuned_conc))
            before_spec = int(script_info["spec_tokens"] if script_info.get("spec_tokens") is not None else (explicit_spec or 0))
        elif has_target_after or (before_track, before_quant, before_kv) != (after_track, after_quant, after_kv):
            layout_before = _auto_solve_report_layout(
                **common_solve, quant=before_quant, kv_cache_dtype=before_kv, indexer_kv_dtype=policy["indexer_kv_dtype"],
                feature_track=before_track, apply_target_envelope=False, rewrite_plan=None, allow_unbound_formula=False,
            )
            before_par = {k: layout_before[k] for k in _PARALLEL_AXES}
            before_conc = int(layout_before["concurrency"])
            before_spec = int(explicit_spec if explicit_spec is not None else layout_before["spec_tokens"])
            tuned_par, tuned_conc = dict(before_par), before_conc
        else:
            before_par, before_conc, before_spec = dict(after_par), conc, after_spec
            tuned_par, tuned_conc, layout_before = dict(after_par), conc, layout_after
    return {
        "layout_before": layout_before, "layout_tuned": layout_tuned, "layout_after": layout_after,
        "before_par": before_par, "before_conc": before_conc, "before_spec": before_spec,
        "tuned_par": tuned_par, "tuned_conc": tuned_conc,
        "after_par": after_par, "conc": conc, "after_spec": after_spec,
        "explicit_accept": explicit_accept, "has_target_after": has_target_after,
    }


def _build_infeasible_report_result(
    *,
    args: argparse.Namespace,
    policy: Mapping[str, Any],
    cols: Mapping[str, Any],
    dep_mode: str,
    eff_phase: str,
    eff_ttft: float | None,
    mode: str,
    resolved_stack_id: str,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Emit honest infeasible ``agent_brief.json``, digest, and receipt (zero fake B=1)."""
    infeasible = dict(cols["layout_after"]["infeasible_outcome"] or {
        "status": "infeasible",
        "deployment_mode": dep_mode,
        "limiting_constraint": "no_feasible_configuration",
        "physical_b_vram_max": 0,
        "feasible_b_per_dp": 0,
        "requested_concurrency": (int(args.concurrency) if args.concurrency is not None else None),
    })
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(policy["model_spec"].name)
    sku = str(policy["sku"])
    raw_prefix = args.prefix or f"{model_name}_{sku}_{eff_phase}"
    prefix = SAFE_PREFIX_RE.sub("_", raw_prefix).strip("_") or "mindie_sim_report"
    brief_path = out_dir / f"{prefix}_agent_brief.json"
    raw_brief: dict[str, Any] = {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": {
            "model": model_name,
            "hardware_sku": sku,
            "topology_preset": str(args.topology_preset or policy.get("topology_preset") or ""),
            "phase": eff_phase,
            "deployment_mode": DEPLOYMENT_MODE_LABELS.get(dep_mode, dep_mode),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "tpot_slo_ms": (float(args.tpot_ms) if args.tpot_ms is not None else None),
            "ttft_slo_ms": eff_ttft,
            "concurrency_per_dp": 0,
            "cluster_concurrency": 0,
            "physical_b_vram_max": int(infeasible["physical_b_vram_max"]),
            "requested_concurrency": infeasible.get("requested_concurrency"),
            "limiting_constraint": infeasible["limiting_constraint"],
            "outcome_status": infeasible["status"],
            "fits_hbm": bool(infeasible["physical_b_vram_max"] > 0),
        },
        "infeasible_outcome": infeasible,
        "generated_files": {"agent_brief_json": brief_path.name},
    }
    export_agent_brief(raw_brief, brief_path)
    projected_brief = json.loads(brief_path.read_text(encoding="utf-8"))
    digest_md = format_executive_digest_markdown(projected_brief)
    receipt: dict[str, Any] = {
        "ok": True,
        "feasible": False,
        "status": infeasible["status"],
        "limiting_constraint": infeasible["limiting_constraint"],
        "physical_b_vram_max": int(infeasible["physical_b_vram_max"]),
        "feasible_b_per_dp": 0,
        "requested_concurrency": infeasible.get("requested_concurrency"),
        "command": "report",
        "mode": mode,
        "stack_id": resolved_stack_id,
        "deployment_mode": dep_mode,
        "ttft_slo_ms": eff_ttft,
        "model_name": model_name,
        "hardware_sku": sku,
        "phase": eff_phase,
        "agent_brief": projected_brief,
    }
    if not getattr(args, "emit_json", False):
        receipt["_stdout_markdown"] = digest_md
    return receipt


def _colocated_prefill_requests(
    raw_case: Mapping[str, Any], request: StepRequest,
) -> tuple[StepRequest, ...]:
    """Expand a chunked prefill request into ordered legal chunk ``StepRequest``s."""
    parsed = parse_case(raw_case)
    chunks = _prefill_chunk_schedule(
        parsed, phase="prefill", context=int(request.context), q_tokens=int(request.tokens),
    )
    if len(chunks) <= 1:
        return (request,)
    last_idx = len(chunks) - 1
    out: list[StepRequest] = []
    for idx, (chunk_ctx, chunk_tok) in enumerate(chunks):
        view = RequestView(
            request_id="prefill_ttft",
            q_len=chunk_tok,
            context=chunk_ctx,
            phase="prefill",
            samples=(idx == last_idx),
        )
        out.append(StepRequest(
            phase="prefill", tokens=chunk_tok, context=chunk_ctx,
            sequences=1, scope=request.scope, requests=(view,),
        ))
    return tuple(out)


def _evaluate_prefill_case_bundles(
    raw_case: Mapping[str, Any],
    request: StepRequest,
    build_kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Evaluate one prefill case across all legal prefill chunks."""
    reqs = _colocated_prefill_requests(raw_case, request)
    return tuple(build_export_bundle(raw_case, request=req, **build_kwargs) for req in reqs)


def _resolve_colocated_prefill_eval(
    *,
    args: argparse.Namespace,
    after_bundle: Any,
    prefill_bundle: Any = None,
    prefill_bundles: Sequence[Any] | None = None,
    prefill_case: Mapping[str, Any] | None = None,
    prefill_request: StepRequest | None = None,
    prefill_build_kwargs: Mapping[str, Any] | None = None,
    mode: str = MODE_SIMULATION,
    dep_mode: str = "colocated",
) -> tuple[float, bool]:
    """Resolve colocated prefill ``(ttft_ms, fits_hbm)`` honoring chunked prefill schedules."""
    if prefill_bundles:
        total_ns = sum(bundle_step_int_ns(b) for b in prefill_bundles)
        fits_hbm = all(bool(b.memory_ledger.fits_hbm) for b in prefill_bundles)
        return float(total_ns) / float(NS_PER_MS), fits_hbm
    if prefill_case is not None:
        req = prefill_request
        if req is None:
            p_tok = int(
                getattr(after_bundle, "prompt_tokens", 0)
                or getattr(args, "prompt_tokens", 0)
                or DEFAULT_REPORT_PROMPT_TOKENS
            )
            scope = str(getattr(after_bundle, "scope", "model") or "model")
            req = StepRequest(phase="prefill", tokens=p_tok, context=0, sequences=1, scope=scope)
        chunk_reqs = _colocated_prefill_requests(prefill_case, req)
        if len(chunk_reqs) == 1 and prefill_bundle is not None:
            return (
                float(bundle_step_int_ns(prefill_bundle)) / float(NS_PER_MS),
                bool(prefill_bundle.memory_ledger.fits_hbm),
            )
        bkw = dict(prefill_build_kwargs or {})
        bkw.setdefault("apply_target_envelope", bool(getattr(after_bundle, "target_envelope_applied", False)))
        bkw.setdefault("allow_unbound_formula", mode == MODE_MODELING)
        bundles = tuple(build_export_bundle(prefill_case, request=cr, **bkw) for cr in chunk_reqs)
        total_ns = sum(bundle_step_int_ns(b) for b in bundles)
        fits_hbm = all(bool(b.memory_ledger.fits_hbm) for b in bundles)
        return float(total_ns) / float(NS_PER_MS), fits_hbm
    if prefill_bundle is not None:
        return (
            float(bundle_step_int_ns(prefill_bundle)) / float(NS_PER_MS),
            bool(prefill_bundle.memory_ledger.fits_hbm),
        )
    eff_cfg = getattr(after_bundle, "effective_config", None)
    if isinstance(eff_cfg, Mapping) and eff_cfg.get("prefill_ttft_ms") is not None:
        return float(eff_cfg["prefill_ttft_ms"]), bool(eff_cfg.get("prefill_fits_hbm", True))
    mem = after_bundle.memory_ledger
    model_ref = getattr(args, "model", None) or after_bundle.model_name
    hw_ref = getattr(after_bundle, "hardware_sku", None) or getattr(args, "hardware", "a3")
    prompt_tokens = int(
        getattr(after_bundle, "prompt_tokens", 0) or getattr(args, "prompt_tokens", 0) or DEFAULT_REPORT_PROMPT_TOKENS
    )
    output_tokens = int(
        getattr(after_bundle, "output_tokens", 0) or getattr(args, "output_tokens", 0) or DEFAULT_SCENARIO_OUTPUT_TOKENS
    )
    conc = max(1, int(getattr(mem, "concurrency", 0) or getattr(after_bundle, "sequences", 1) or 1))
    req_quant = getattr(args, "weight_quant", None)
    quant_arg = req_quant if req_quant is not None else (getattr(mem, "weight_quant", None) or None)
    kv_arg = getattr(args, "cache_dtype", None)
    p_raw, p_req = compose_export_case(
        str(model_ref),
        str(hw_ref),
        topology_preset=(getattr(after_bundle, "topology_preset", None) or getattr(args, "topology_preset", None) or None),
        dp=max(1, int(getattr(mem, "dp", 1) or 1)),
        tp=max(1, int(getattr(mem, "tp", 1) or 1)),
        ep=max(1, int(getattr(mem, "ep", 1) or 1)),
        pp=max(1, int(getattr(mem, "pp", 1) or 1)),
        dcp=max(1, int(getattr(mem, "dcp", 1) or 1)),
        pcp=max(1, int(getattr(mem, "pcp", 1) or 1)),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=conc,
        ttft_slo_ms=after_bundle.ttft_slo_ms,
        tpot_slo_ms=after_bundle.tpot_slo_ms,
        mtp_proposal_tokens=int(getattr(after_bundle, "spec_proposals", 0) or 0),
        mtp_acceptance_length=max(1.0, float(getattr(after_bundle, "avg_accepted_tokens", 1.0) or 1.0)),
        feature_track=str(getattr(after_bundle, "feature_track", "") or "shipping"),
        quant=quant_arg,
        kv_cache_dtype=kv_arg,
        engram_cpu_offload=getattr(args, "engram_offload", None),
        deployment_mode=dep_mode,
        runtime_stack=getattr(args, "stack", None),
        phase="prefill",
    )
    bkw = dict(prefill_build_kwargs or {})
    bkw.setdefault("apply_target_envelope", bool(getattr(after_bundle, "target_envelope_applied", False)))
    bkw.setdefault("allow_unbound_formula", mode == MODE_MODELING)
    bundles = _evaluate_prefill_case_bundles(p_raw, p_req, bkw)
    total_ns = sum(bundle_step_int_ns(b) for b in bundles)
    fits_hbm = all(bool(b.memory_ledger.fits_hbm) for b in bundles)
    return float(total_ns) / float(NS_PER_MS), fits_hbm


def _evaluate_colocated_prefill_ttft_ms(
    *,
    args: argparse.Namespace,
    after_bundle: Any,
    prefill_bundle: Any = None,
    prefill_bundles: Sequence[Any] | None = None,
    prefill_case: Mapping[str, Any] | None = None,
    prefill_request: StepRequest | None = None,
    prefill_build_kwargs: Mapping[str, Any] | None = None,
    mode: str = MODE_SIMULATION,
    dep_mode: str = "colocated",
) -> float:
    """Resolve colocated prefill TTFT (ms) from supplied prefill bundle(s) or exact chunked replay."""
    ttft_ms, _ = _resolve_colocated_prefill_eval(
        args=args, after_bundle=after_bundle,
        prefill_bundle=prefill_bundle, prefill_bundles=prefill_bundles,
        prefill_case=prefill_case, prefill_request=prefill_request,
        prefill_build_kwargs=prefill_build_kwargs, mode=mode, dep_mode=dep_mode,
    )
    return ttft_ms


def _assemble_report_receipt(
    *,
    args: argparse.Namespace,
    after_bundle: Any,
    summary: Mapping[str, Any],
    brief_dict: dict[str, Any],
    digest_md: str,
    mode: str,
    resolved_stack_id: str,
    dep_mode: str,
    prefill_bundle: Any = None,
    prefill_bundles: Sequence[Any] | None = None,
    prefill_case: Mapping[str, Any] | None = None,
    prefill_request: StepRequest | None = None,
    prefill_build_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute canonical SLA/HBM feasibility and assemble the CLI report receipt."""
    is_decode = after_bundle.phase == "decode"
    after_step_ms = float(bundle_step_int_ns(after_bundle)) / float(NS_PER_MS)
    acc = float(after_bundle.avg_accepted_tokens)
    eff_tpot_ms = (
        ((after_step_ms / acc) if acc > 0.0 else after_step_ms)
        + float(getattr(after_bundle, "handoff_tpot_ms", 0.0) or 0.0)
        if is_decode else after_step_ms
    )
    tpot_slo = after_bundle.tpot_slo_ms if is_decode else None
    prefill_ttft_ms: float | None = None
    prefill_fits_hbm = True
    if not is_decode:
        ttft_slo = after_bundle.ttft_slo_ms
        if prefill_bundles or prefill_case is not None:
            prefill_ttft_ms, prefill_fits_hbm = _resolve_colocated_prefill_eval(
                args=args, after_bundle=after_bundle,
                prefill_bundle=after_bundle, prefill_bundles=prefill_bundles,
                prefill_case=prefill_case, prefill_request=prefill_request,
                prefill_build_kwargs=prefill_build_kwargs, mode=mode, dep_mode=dep_mode,
            )
        else:
            prefill_ttft_ms = after_step_ms
    elif dep_mode == "colocated" and after_bundle.ttft_slo_ms is not None:
        ttft_slo = after_bundle.ttft_slo_ms
        prefill_ttft_ms, prefill_fits_hbm = _resolve_colocated_prefill_eval(
            args=args, after_bundle=after_bundle,
            prefill_bundle=prefill_bundle, prefill_bundles=prefill_bundles,
            prefill_case=prefill_case, prefill_request=prefill_request,
            prefill_build_kwargs=prefill_build_kwargs, mode=mode, dep_mode=dep_mode,
        )
    elif dep_mode == "colocated" and (prefill_bundles or prefill_bundle is not None):
        ttft_slo = None
        prefill_fits_hbm = (
            all(bool(b.memory_ledger.fits_hbm) for b in prefill_bundles)
            if prefill_bundles else bool(prefill_bundle.memory_ledger.fits_hbm)
        )
    else:
        ttft_slo = None
    tpot_met = (tpot_slo is None) or (eff_tpot_ms <= float(tpot_slo))
    ttft_met = (ttft_slo is None) or (prefill_ttft_ms is not None and prefill_ttft_ms <= float(ttft_slo))
    all_slas_met = bool(
        tpot_met
        and ttft_met
        and str(after_bundle.sla_regime or "").upper() != "SLA_VIOLATED"
    )
    hbm_feasible = bool(after_bundle.memory_ledger.fits_hbm and prefill_fits_hbm)
    feasible = bool(all_slas_met and hbm_feasible)
    receipt: dict[str, Any] = {
        "ok": True,
        "feasible": feasible,
        "status": "feasible" if feasible else "infeasible",
        "all_slas_met": all_slas_met,
        "ttft_met": ttft_met,
        "hbm_feasible": hbm_feasible,
        "sla_regime": after_bundle.sla_regime,
        "command": "report",
        "mode": mode,
        "stack_id": resolved_stack_id,
        "deployment_mode": dep_mode,
        "ttft_slo_ms": after_bundle.ttft_slo_ms,
        "model_name": after_bundle.model_name,
        "hardware_sku": after_bundle.hardware_sku,
        "phase": after_bundle.phase,
        "agent_brief": brief_dict,
        **summary,
    }
    if prefill_ttft_ms is not None:
        receipt["prefill_ttft_ms"] = round(prefill_ttft_ms, STEP_MS_ROUND_DIGITS)
    if args.concurrency is not None:
        receipt["requested_concurrency"] = int(args.concurrency)
    if not getattr(args, "emit_json", False):
        receipt["_stdout_markdown"] = digest_md
    return receipt


def run_report_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute ``mindie-sim report`` with explicit deployment-mode, domain stack routing, and zero TTFT coercion.

    [Cohesive-Pipeline] End-to-end turnkey report orchestration from CLI args through solve, export, and receipt emission.
    """
    stack_arg = getattr(args, "stack", None)
    script_info = (
        parse_script_for_report(args.from_script, requested_stack=stack_arg, hardware=args.hardware)
        if getattr(args, "from_script", None) else None
    )
    dep_mode, eff_phase, eff_ttft = _resolve_report_mode_and_phase(args, script_info)
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
        model_arg, args.hardware, context_tokens=prompt_tokens,
        weight_quant=args.weight_quant, cache_dtype=args.cache_dtype,
    )
    workload_domain = infer_workload_domain(policy["model_spec"])
    work_unit = default_work_unit_for_domain(workload_domain)
    hw_sel = policy["sku"] if policy["backend"] == "ascend" else policy["backend"]
    adapter = resolve_stack_for_scenario(
        requested_stack=stack_arg,
        workload_domain=workload_domain,
        work_unit=work_unit,
        hardware_family=hw_sel,
    )
    resolved_stack_id = adapter.capabilities().stack_id

    max_pos = int(getattr(policy["model_spec"], "max_position_embeddings", 0) or 0)
    if args.prompt_tokens is None and max_pos > output_tokens and (prompt_tokens + output_tokens) > max_pos:
        prompt_tokens = max(1, max_pos - output_tokens)
    topo_preset = args.topology_preset or policy["topology_preset"]
    before_track, after_track = _report_tracks(
        mode, compare_track=str(args.compare_track), target_track=str(args.target_track),
        enforce_eager=bool(script_info and script_info.get("enforce_eager")),
    )
    before_quant = (script_info.get("weight_quant") if script_info else None) or policy["compose_quant"]
    before_kv = (script_info.get("cache_dtype") if script_info else None) or policy["kv_cache_dtype"]
    after_quant, after_kv, after_idx_kv = policy["compose_quant"], policy["kv_cache_dtype"], policy["indexer_kv_dtype"]

    cols = _solve_report_columns(
        args=args, model_arg=model_arg, policy=policy, script_info=script_info,
        dep_mode=dep_mode, eff_phase=eff_phase, eff_ttft=eff_ttft,
        mode=mode, rewrite_plan=rewrite_plan, topo_preset=topo_preset,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        before_track=before_track, after_track=after_track,
        before_quant=before_quant, before_kv=before_kv,
        after_quant=after_quant, after_kv=after_kv, after_idx_kv=after_idx_kv,
    )
    if cols["layout_after"] is not None and not cols["layout_after"]["solver_feasible"]:
        return _build_infeasible_report_result(
            args=args, policy=policy, cols=cols, dep_mode=dep_mode,
            eff_phase=eff_phase, eff_ttft=eff_ttft, mode=mode,
            resolved_stack_id=resolved_stack_id,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        )
    replay = ReplayOptions(include_submission=bool(getattr(args, "include_submission", False)))
    common_kw = dict(
        topology_preset=topo_preset, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        tpot_slo_ms=(float(args.tpot_ms) if args.tpot_ms is not None else None),
        ttft_slo_ms=eff_ttft,
        mtp_proposal_tokens=cols["after_spec"],
        mtp_acceptance_length=_accept_len_for(cols["after_spec"], cols["explicit_accept"]),
        engram_cpu_offload=args.engram_offload, deployment_mode=dep_mode,
        runtime_stack=stack_arg,
    )
    column_kw = dict(model=model_arg, sku=policy["sku"], phase=eff_phase, common_kw=common_kw)
    raw_before, step_b, meta_b = _report_column(
        cols["layout_before"], track=before_track, quant=before_quant, kv=before_kv,
        idx_kv=policy["indexer_kv_dtype"], par=cols["before_par"], conc=cols["before_conc"],
        spec_tokens=cols["before_spec"], accept_len=_accept_len_for(cols["before_spec"], cols["explicit_accept"]),
        **column_kw,
    )
    if script_info is not None:
        raw_before = _apply_script_overrides_to_case(raw_before, script_info)
    raw_after, step_a, meta_a = _report_column(
        cols["layout_after"], track=after_track, quant=after_quant, kv=after_kv,
        idx_kv=after_idx_kv, par=cols["after_par"], conc=cols["conc"], **column_kw,
    )
    script_supplied = script_info is not None
    before_bundle = _attach_bundle_context(
        build_export_bundle(
            raw_before, request=step_b, replay=replay, observations_path=args.observations,
            cache_dtype=before_kv, engram_host_offload=args.engram_offload, apply_target_envelope=False, **meta_b,
        ),
        serve_script_supplied=script_supplied,
        solver_feasible=(bool(cols["layout_before"].get("solver_feasible", True)) if cols["layout_before"] is not None else None),
    )
    tuned_bundle = None
    if script_info is not None:
        raw_tuned, step_t, meta_t = _report_column(
            cols["layout_tuned"], track="shipping", quant=after_quant, kv=after_kv,
            idx_kv=after_idx_kv, par=cols["tuned_par"], conc=cols["tuned_conc"], **column_kw,
        )
        tuned_bundle = _attach_bundle_context(
            build_export_bundle(
                raw_tuned, request=step_t, replay=replay, cache_dtype=after_kv,
                engram_host_offload=args.engram_offload, apply_target_envelope=False, **meta_t,
            ),
            serve_script_supplied=True,
            solver_feasible=(bool(cols["layout_tuned"].get("solver_feasible", True)) if cols["layout_tuned"] is not None else None),
        )
    after_bundle = _attach_bundle_context(
        build_export_bundle(
            raw_after, request=step_a, replay=replay, cache_dtype=after_kv, engram_host_offload=args.engram_offload,
            apply_target_envelope=cols["has_target_after"],
            rewrite_plan=rewrite_plan if cols["has_target_after"] else None,
            allow_unbound_formula=(mode == MODE_MODELING), **meta_a,
        ),
        serve_script_supplied=script_supplied,
        solver_feasible=(bool(cols["layout_after"].get("solver_feasible", True)) if cols["layout_after"] is not None else None),
    )
    if cols["has_target_after"] and after_bundle.step_span_us > before_bundle.step_span_us:
        extra_exclusions: list[str] = []
        if (
            bool(before_bundle.effective_config.get("multistream_overlap_shared_expert", False))
            and int(after_bundle.effective_config.get("enable_fused_mc2", 0) or 0) > 0
            and not bool(after_bundle.effective_config.get("multistream_overlap_shared_expert", False))
        ):
            extra_exclusions.append("cann_megamoe_fusion")
        after_rt_meta = (raw_after.get("runtime") or {}).get("metadata") or {}
        before_rt_meta = (raw_before.get("runtime") or {}).get("metadata") or {}
        if (
            bool(after_rt_meta.get("mqsmla_dsl_544_320_active"))
            or any(
                "tilelang_dsl.mixed_quant_sparse_flash_mla" in str(getattr(r, "ascend_kernel", ""))
                for r in after_bundle.operators
            )
        ) and not bool(before_rt_meta.get("mqsmla_dsl_544_320_active")):
            extra_exclusions.append("mqsmla_dsl_544_320")
        for excl_id in extra_exclusions:
            alt_common = dict(common_kw)
            prev_excl = tuple(alt_common.get("exclude_optimizations") or ())
            next_excl = tuple(dict.fromkeys((*prev_excl, excl_id)))
            alt_common["exclude_optimizations"] = next_excl
            if dep_mode == "pd_disaggregated":
                alt_common["deployment_mode"] = "pd_prefill" if eff_phase == "prefill" else "pd_decode"
                if eff_phase == "prefill":
                    alt_common["tpot_slo_ms"] = None
                else:
                    alt_common["ttft_slo_ms"] = None
            solved_after_accept = (raw_after.get("runtime") or {}).get("mtp_acceptance_length")
            alt_col_kw = dict(column_kw, common_kw=alt_common)
            raw_after_alt, step_a_alt, meta_a_alt = _report_column(
                None, track=after_track, quant=after_quant, kv=after_kv,
                idx_kv=after_idx_kv, par=cols["after_par"], conc=cols["conc"],
                accept_len=(float(solved_after_accept) if solved_after_accept is not None else None),
                **alt_col_kw,
            )
            if cols["layout_after"] is not None:
                step_a_alt = step_a
                meta_a_alt = dict(meta_a)
            alt_bundle = _attach_bundle_context(
                build_export_bundle(
                    raw_after_alt, request=step_a_alt, replay=replay, cache_dtype=after_kv,
                    engram_host_offload=args.engram_offload,
                    apply_target_envelope=cols["has_target_after"],
                    rewrite_plan=rewrite_plan if cols["has_target_after"] else None,
                    allow_unbound_formula=(mode == MODE_MODELING), **meta_a_alt,
                ),
                serve_script_supplied=script_supplied,
                solver_feasible=(bool(cols["layout_after"].get("solver_feasible", True)) if cols["layout_after"] is not None else None),
            )
            if alt_bundle.step_span_us < after_bundle.step_span_us:
                common_kw = dict(common_kw, exclude_optimizations=next_excl)
                column_kw = dict(column_kw, common_kw=common_kw)
                raw_after, step_a, meta_a, after_bundle = raw_after_alt, step_a_alt, meta_a_alt, alt_bundle
    prefill_after_bundles: tuple[Any, ...] = ()
    prefill_after_case: Mapping[str, Any] | None = None
    prefill_after_req: StepRequest | None = None
    prefill_build_kw: dict[str, Any] | None = None
    if dep_mode == "colocated" and after_bundle.phase == "decode" and after_bundle.ttft_slo_ms is not None:
        raw_prefill_after, step_prefill_a, meta_prefill_a = _report_column(
            cols["layout_after"],
            model=model_arg, sku=policy["sku"], phase="prefill",
            track=after_track, quant=after_quant, kv=after_kv, idx_kv=after_idx_kv,
            par=cols["after_par"], conc=cols["conc"], common_kw=common_kw,
        )
        prefill_after_case = raw_prefill_after
        prefill_after_req = step_prefill_a
        prefill_build_kw = dict(
            replay=replay, cache_dtype=after_kv,
            engram_host_offload=args.engram_offload,
            apply_target_envelope=cols["has_target_after"],
            rewrite_plan=rewrite_plan if cols["has_target_after"] else None,
            allow_unbound_formula=(mode == MODE_MODELING), **meta_prefill_a,
        )
        prefill_after_bundles = _evaluate_prefill_case_bundles(
            raw_prefill_after, step_prefill_a, prefill_build_kw,
        )
    elif after_bundle.phase == "prefill":
        prefill_after_case = raw_after
        prefill_after_req = step_a
        prefill_build_kw = dict(
            replay=replay, cache_dtype=after_kv,
            engram_host_offload=args.engram_offload,
            apply_target_envelope=cols["has_target_after"],
            rewrite_plan=rewrite_plan if cols["has_target_after"] else None,
            allow_unbound_formula=(mode == MODE_MODELING), **meta_a,
        )
        chunk_reqs = _colocated_prefill_requests(raw_after, step_a)
        if len(chunk_reqs) > 1:
            prefill_after_bundles = tuple(
                build_export_bundle(raw_after, request=cr, **prefill_build_kw)
                for cr in chunk_reqs
            )
    summary, brief_dict, digest_md = _emit_report_artifacts(
        before_bundle=before_bundle, after_bundle=after_bundle, tuned_bundle=tuned_bundle,
        args=args, script_info=script_info, dep_mode=dep_mode,
    )
    return _assemble_report_receipt(
        args=args, after_bundle=after_bundle, summary=summary, brief_dict=brief_dict,
        digest_md=digest_md, mode=mode, resolved_stack_id=resolved_stack_id, dep_mode=dep_mode,
        prefill_bundles=prefill_after_bundles, prefill_case=prefill_after_case,
        prefill_request=prefill_after_req, prefill_build_kwargs=prefill_build_kw,
    )


def run_quickstart_command(args: argparse.Namespace) -> dict[str, Any]:
    """Return turnkey quickstart guide (Markdown by default, or ``--json``)."""
    payload: dict[str, Any] = {
        "schema": "mindie-sim.quickstart/1",
        "golden_commands": [
            {"name": "zero_config_modeling_report", "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --deployment-mode pd_decode --num-devices 32 --context 128k --out-dir ./out"},
            {"name": "pass2_agent_rewrite_plan", "cmd": "mindie-sim report --model kimi-k3.1 --hardware 950dt --deployment-mode pd_decode --num-devices 32 --rewrite-plan ./out/plan.json --out-dir ./out"},
            {"name": "strict_simulation_mode", "cmd": "mindie-sim report --model qwen3-32b --hardware a3 --deployment-mode colocated --mode simulation --compare-track shipping --target-track proposed --out-dir ./out"},
            {"name": "audit_launch_script", "cmd": "mindie-sim report --from-script ./serve.sh --hardware a3 --deployment-mode colocated --out-dir ./out"},
            {"name": "blind_box_hf_config", "cmd": "mindie-sim report --model ./my_hf_model/config.json --hardware a3 --deployment-mode colocated --tpot-ms 20 --out-dir ./out"},
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
            "Pass 1: Run `mindie-sim report --deployment-mode <colocated|pd_disaggregated|pd_prefill|pd_decode>` -> read `<prefix>_agent_brief.json` (<8 KB).",
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
