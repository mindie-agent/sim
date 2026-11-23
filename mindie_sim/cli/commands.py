"""CLI command dispatch, bounded JSON ingress, and UTF-8/LF emission.

[Role] Execute parsed CLI commands by routing canonical subcommands through
  ``mindie_sim.api`` -> ``PredictionSpine`` and preserving all 29 legacy
  command handlers, bounded 50 MiB JSON reads, and profiler supply injection.
[Invariants] Every JSON read enforces ``MAX_CLI_JSON_BYTES`` (50 MiB). Output
  files are written as UTF-8 with Unix LF endings and ``allow_nan=False``;
  non-UTF-8 stdout streams fall back to ASCII-escaped JSON cleanly.
[Timeouts & Locks] Bounded metadata reads and streaming observation imports;
  propagates ``TimeoutError`` without masking.
[Side-Effect Policy] Writes only explicitly requested output, index or store
  paths.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

from mindie_sim import api as sdk_api
from mindie_sim import predict
from .export import EXPORT_COMMANDS, run_export_command
from .onboarding import ONBOARDING_COMMANDS, run_onboarding_command
from .sku import SKU_COMMANDS, run_sku_command
from mindie_sim.prediction.offload.policy import missing_scenario_error
from mindie_sim.prediction.program.primitives import PrimitiveKey, PrimitiveStore
from mindie_sim.prediction.results import ReplayOptions, StepRequest, json_payload
from mindie_sim.prediction.structure import parse_case

MIB_BYTES = 1024 * 1024
MAX_CLI_JSON_BYTES = 50 * MIB_BYTES
BYTES_PER_TB = 1e12
GIB_DECIMALS = 1
TBPS_DECIMALS = 3


class CliUsageError(RuntimeError):
    """Local usage rejection; never a profiler runtime dependency."""


def load_cli_json(path: str | Path) -> object:
    """Bounded CLI JSON ingress: size gate, UTF-8 and finite constants."""
    target = Path(path)
    if target.stat().st_size > MAX_CLI_JSON_BYTES:
        raise CliUsageError("JSON input exceeds 50 MiB budget")
    try:
        raw = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CliUsageError("invalid UTF-8 JSON input") from exc

    def bad_const(token: str) -> None:
        raise CliUsageError(f"non-finite JSON constant: {token}")

    try:
        return json.loads(raw, parse_constant=bad_const)
    except (ValueError, RecursionError) as exc:
        raise CliUsageError(f"invalid JSON input: {exc}") from exc


def read_json(path: str | Path, /, *, bounded: bool = False) -> object:
    """Read a JSON file bounded to 50 MiB."""
    target = Path(path)
    if target.stat().st_size > MAX_CLI_JSON_BYTES:
        raise ValueError("metadata_input_exceeds_50_MiB:use_streaming_observation_import")
    if bounded:
        return load_cli_json(target)
    with target.open(encoding="utf-8") as stream:
        return json.load(stream)


def emit(value: object, output: str | Path | None, /) -> None:
    """Serialize JSON with Unix LF endings and CP936-safe stdout fallback."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    data = text.encode("utf-8")
    if output:
        Path(output).write_bytes(data)
        return
    try:
        print(text, end="")
    except UnicodeEncodeError:
        print(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False))


def _case_with_supply(args: argparse.Namespace):
    raw = read_json(args.case)
    if getattr(args, "supply", None):
        from mindie_sim.observations.calibration import apply_profiler_supply
        raw = apply_profiler_supply(
            raw,
            args.supply,
            recipe=getattr(args, "supply_recipe", None),
            kernel=getattr(args, "supply_kernel", None),
            hardware=getattr(args, "supply_hardware", None),
            owners=getattr(args, "supply_owners", None),
            legacy_diagnostics=bool(getattr(args, "supply_legacy_diagnostics", False)),
        )
    return raw


def _export_structure_sidecar(args: argparse.Namespace, case, *, phase, tokens, context, sequences):
    from mindie_sim.structure import export_structure
    report = export_structure(
        case,
        phase=phase,
        tokens=tokens,
        context=context,
        sequences=sequences,
        scope=args.scope,
        media_tokens_per_sequence=getattr(args, "media_tokens_per_sequence", None),
        apply_body_defaults=args.apply_body_defaults,
        draft_prepare=bool(getattr(args, "draft_prepare", False)),
    )
    path = Path(args.export_structure)
    data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {
        "structure_export": {
            "path": str(path),
            "nodes": len(report["nodes"]),
            "graph_scope": report.get("graph_scope"),
        }
    }


def _inspect_case(raw_case) -> dict:
    from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
    from mindie_sim.prediction.body_defaults import effective_body_declaration
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.prediction.graph import GraphPolicy
    from mindie_sim.prediction.implementation_selection import (
        combine_summaries, records_summary, resolve,
    )
    from mindie_sim.prediction.scheduler import Scheduler

    case = parse_case(raw_case)
    memory = Memory(case)
    scheduler = Scheduler(case, list(case.workload.requests), memory)
    plan = scheduler.schedule(min(r.arrival_ms for r in case.workload.requests))
    if plan is None:
        return {
            "case": case.to_dict(),
            "memory": memory.to_dict(),
            "operators": [],
            "operator_selection_summary": {},
        }
    views = tuple(plan.requests)
    phase = next((view.phase for view in views), "decode")
    effective, selection = effective_body_declaration(
        case, phase, tokens=max(1, sum(view.q_len for view in views)), scope="body"
    )
    runtime = scheduler.next_runtime(plan)
    bucket, replay_scope = GraphPolicy().select(views, runtime)
    padded = max(sum(view.q_len for view in views), bucket or 0)
    runtime, records = resolve(
        runtime, views, padded_tokens=padded, replay_scope=replay_scope
    )
    frontend = Frontend(attention_policy=AttentionPolicy(**runtime.attention_policy))
    work = frontend(plan, effective.model, runtime, effective.machine)
    summary = combine_summaries([
        selection.get("operator_selection_summary"),
        records_summary(
            arch_family=effective.machine.arch_family, phase=phase, records=records
        ),
    ])
    return {
        "case": case.to_dict(),
        "memory": memory.to_dict(),
        "operators": [op.to_dict() for op in work.ops],
        "operator_selection_summary": summary,
    }


def _dispatch_predict_step(args: argparse.Namespace) -> object:
    raw = _case_with_supply(args)
    view = str(getattr(args, "view", "step"))
    if view == "efficiency":
        from mindie_sim.prediction.efficiency import analyze_network_efficiency
        from mindie_sim.prediction.evaluation import EvaluationContext
        from mindie_sim.prediction.graph_analysis import graph_costs
        from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs

        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            if (
                args.draft_prepare
                or args.media_tokens_per_sequence is not None
                or store is not None
                or bool(getattr(args, "include_submission", False))
            ):
                case = parse_case(raw)
                ops = target_graphs(
                    case, phase=args.phase, tokens=args.tokens, context=args.context,
                    sequences=args.sequences, scope=args.scope,
                    media_tokens_per_sequence=args.media_tokens_per_sequence,
                    draft_prepare=bool(args.draft_prepare),
                )
                ext_inputs = (
                    {"analysis_input": 0.0}
                    if any("analysis_input" in op.deps for g in ops.values() for op in g)
                    else None
                )
                ctx = EvaluationContext.create(case, None, store)
                op_report = operator_costs(case, ops, primitives=store, evaluation=ctx)
                gr_report = graph_costs(
                    case, ops, primitives=store,
                    external_inputs=ext_inputs,
                    include_submission=bool(args.include_submission),
                    isolated=False, evaluation=ctx,
                )
                return analyze_network_efficiency(
                    case, phase=args.phase, tokens=args.tokens, context=args.context,
                    sequences=args.sequences, scope=args.scope,
                    operator_report=op_report, graph_report=gr_report,
                    evaluation_context=ctx,
                )
            return analyze_network_efficiency(
                raw, phase=args.phase, tokens=args.tokens, context=args.context,
                sequences=args.sequences, scope=args.scope,
            )
    if view in {"operators", "graph"}:
        from mindie_sim.prediction.graph_analysis import graph_costs
        from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
        case = parse_case(raw)
        ops = target_graphs(
            case, phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            media_tokens_per_sequence=args.media_tokens_per_sequence,
            draft_prepare=args.draft_prepare,
        )
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            body = (
                operator_costs(case, ops, primitives=store)
                if view == "operators"
                else graph_costs(
                    case, ops, primitives=store,
                    include_submission=bool(args.include_submission),
                    isolated=bool(args.isolated),
                )
            )
            return {
                **body,
                "graph_scope": args.scope,
                "tokens_per_sequence": args.tokens,
                "sequences": args.sequences,
                "sequences_per_replica": args.sequences,
                "draft_prepare": args.draft_prepare,
                "deployment_scope": "first_independent_dp_cohort",
            }
    req = StepRequest(
        phase=args.phase, tokens=args.tokens, context=args.context,
        sequences=args.sequences, scope=args.scope,
        media_tokens_per_sequence=args.media_tokens_per_sequence,
        draft_prepare=bool(args.draft_prepare),
    )
    replay = ReplayOptions(
        include_submission=bool(args.include_submission),
        isolated=bool(args.isolated),
        event_limit=int(args.event_limit),
    )
    with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
        result = sdk_api.evaluate_step(raw, req, replay=replay, primitives=store)
    return json_payload(result)


def _dispatch_sdk(args: argparse.Namespace) -> object | None:
    cmd = args.command
    if cmd == "predict-service":
        raw = _case_with_supply(args)
        replay = ReplayOptions(event_limit=int(args.event_limit))
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            result = sdk_api.evaluate_service(raw, replay=replay, primitives=store)
        return json_payload(result)
    if cmd == "predict-step":
        return _dispatch_predict_step(args)
    if cmd == "predict-tiers":
        raw = _case_with_supply(args)
        req = StepRequest(
            phase=args.phase,
            tokens=args.tokens,
            context=args.context,
            sequences=args.sequences,
            scope=args.scope,
        )
        replay = ReplayOptions(
            isolated=bool(args.isolated),
            event_limit=int(args.event_limit),
        )
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            result = sdk_api.evaluate_three_tiers(
                raw, req,
                replay=replay, primitives=store,
            )
        return json_payload(result)
    if cmd == "optimize":
        raw = _case_with_supply(args)
        req = StepRequest(
            phase=args.phase,
            tokens=args.tokens,
            context=args.context,
            sequences=args.sequences,
            scope=args.scope,
        )
        replay = ReplayOptions(
            isolated=bool(args.isolated),
            event_limit=int(args.event_limit),
        )
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            result = sdk_api.optimize_ladder(
                raw, req,
                replay=replay, primitives=store,
                stack_id=str(args.stack),
                max_trials=int(args.max_trials),
            )
        return json_payload(result)
    return None


def _dispatch_sweep(args: argparse.Namespace) -> object:
    from .options import (
        apply_case_mapping_overrides,
        case_overrides,
        load_hardware_derivations,
    )
    from mindie_sim.sweep import (
        solve_optimal_deployment,
        sweep_pd_disaggregation_ratios,
        sweep_topologies,
    )

    case_data = apply_case_mapping_overrides(read_json(args.case), case_overrides(args))
    if args.mode == "auto":
        parsed = parse_case(case_data)
        return solve_optimal_deployment(
            parsed.model.to_dict(),
            parsed.machine.to_dict(),
            num_devices=args.num_devices,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            ttft_sla_ms=args.ttft_sla_ms,
            tpot_sla_ms=args.tpot_sla_ms,
            deployment=str(getattr(args, "deployment", "auto") or "auto"),
            pd_transfer_mode=getattr(args, "pd_transfer_mode", None),
            kv_transfer_bw_gbps=getattr(args, "kv_transfer_bw_gbps", None),
            pool_dram_ratio=getattr(args, "pool_dram_ratio", None),
            pool_ssd_ratio=float(getattr(args, "pool_ssd_ratio", 0.0) or 0.0),
            pool_prefix_hit_ratio=float(getattr(args, "pool_prefix_hit_ratio", 0.0) or 0.0),
            ddr_bw_gbps=getattr(args, "ddr_bw_gbps", None),
            ssd_bw_gbps=getattr(args, "ssd_bw_gbps", None),
        )
    if args.mode == "pd":
        if args.num_devices is None:
            raise missing_scenario_error(("num_devices",))
        decode_data = read_json(args.decode_case) if args.decode_case else case_data
        pd_mode = getattr(args, "pd_transfer_mode", None) or "direct_blocking_d2d"
        return sweep_pd_disaggregation_ratios(
            case_data,
            decode_data,
            total_dies=int(args.num_devices),
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            ttft_sla_ms=args.ttft_sla_ms,
            tpot_sla_ms=args.tpot_sla_ms,
            kv_transfer_bw_gbps=getattr(args, "kv_transfer_bw_gbps", None),
            pd_transfer_mode=pd_mode,
            pool_dram_ratio=getattr(args, "pool_dram_ratio", None),
            pool_ssd_ratio=float(getattr(args, "pool_ssd_ratio", 0.0) or 0.0),
            pool_prefix_hit_ratio=float(getattr(args, "pool_prefix_hit_ratio", 0.0) or 0.0),
            ddr_bw_gbps=getattr(args, "ddr_bw_gbps", None),
            ssd_bw_gbps=getattr(args, "ssd_bw_gbps", None),
        )
    if args.resume and not (args.output_jsonl or args.checkpoint_jsonl):
        raise CliUsageError("--resume requires --checkpoint-jsonl or --output-jsonl")
    grids = (
        [int(item) for item in str(args.num_devices_grid).split(",") if item.strip()]
        if args.num_devices_grid else None
    )
    dcp_candidates = (
        [int(item) for item in str(args.dcp_candidates).split(",") if item.strip()]
        if args.dcp_candidates else None
    )
    explicit = read_json(args.candidates) if args.candidates else None
    derivations = (
        load_hardware_derivations(args.hardware_derivations)
        if args.hardware_derivations else None
    )
    progress = None
    if args.progress:
        def progress(row):
            print(
                f"{row.get('candidate_id', '')} {row.get('status', '')} "
                f"{row.get('evaluation_stage', '')} d={row.get('num_devices', '')} "
                f"b1={row.get('step_ms_at_b1')}",
                file=sys.stderr, flush=True,
            )
    return sweep_topologies(
        case_data,
        candidates=explicit,
        num_devices=args.num_devices,
        num_devices_grid=grids,
        sla_ms=args.sla_ms,
        context=args.context,
        tokens_per_seq=args.tokens_per_seq,
        output_budget=args.output_budget,
        max_sequences_per_dp=args.max_sequences_per_dp,
        phase=args.phase,
        scope=args.scope,
        allow_pp=args.allow_pp,
        allow_pcp=args.allow_pcp,
        allow_dcp=args.allow_dcp,
        dcp_candidates=dcp_candidates,
        hardware_derivations=derivations,
        two_stage=args.two_stage,
        finalist_top_k=args.finalist_top_k,
        candidate_timeout_s=args.candidate_timeout_s,
        output_jsonl=args.output_jsonl,
        checkpoint_jsonl=args.checkpoint_jsonl,
        isolate_workers=args.isolate_workers,
        resume=args.resume,
        progress_callback=progress,
    )


def _dispatch_sla(args: argparse.Namespace) -> object:
    from mindie_sim import runtime_advisor
    from .options import apply_case_mapping_overrides, case_overrides
    from mindie_sim.prediction.efficiency import DEFAULT_SLA_SCENARIOS, solve_sla_sizing

    case_data = apply_case_mapping_overrides(read_json(args.case), case_overrides(args))
    parsed = parse_case(case_data)
    resolved_sla = args.sla_ms
    if resolved_sla is None:
        resolved_sla = parsed.slo.ttft_ms if args.phase == "prefill" else parsed.slo.tpot_ms
    resolved_output = args.output_budget
    if resolved_output is None:
        meta = dict(parsed.runtime.metadata)
        reqs = tuple(parsed.workload.requests or ())
        if (meta.get("scenario_declared", True) is not False
                and reqs and reqs[0].request_id != "__structure_probe__"):
            resolved_output = int(reqs[0].output_tokens)
    missing = [
        name for name, val in (
            ("sla_ms", resolved_sla),
            ("context", args.context),
            ("output_budget", resolved_output),
        ) if val is None
    ]
    if missing:
        raise missing_scenario_error(missing)
    has_custom_scale = any(
        v is not None for v in (
            args.hbm_bw_scale, args.cube_flops_scale, args.vector_ops_scale,
            args.link_bw_scale, args.hbm_capacity_scale,
        )
    )
    if args.scenarios:
        scenarios = read_json(args.scenarios)
    elif has_custom_scale:
        scenarios = [
            {
                "id": "baseline", "label": "1x Baseline",
                "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0,
                "vector_ops_scale": 1.0, "link_bw_scale": 1.0,
                "hbm_capacity_scale": 1.0,
            },
            {
                "id": "scaled", "label": "Custom Scaled",
                "hbm_bw_scale": 1.0 if args.hbm_bw_scale is None else args.hbm_bw_scale,
                "cube_flops_scale": 1.0 if args.cube_flops_scale is None else args.cube_flops_scale,
                "vector_ops_scale": 1.0 if args.vector_ops_scale is None else args.vector_ops_scale,
                "link_bw_scale": 1.0 if args.link_bw_scale is None else args.link_bw_scale,
                "hbm_capacity_scale": 1.0 if args.hbm_capacity_scale is None else args.hbm_capacity_scale,
            },
        ]
    else:
        scenarios = None
    jobs = runtime_advisor.resolve_jobs(
        args.jobs, task_count=len(scenarios or DEFAULT_SLA_SCENARIOS)
    )
    result = solve_sla_sizing(
        case_data,
        sla_ms=float(resolved_sla),
        context=int(args.context),
        tokens_per_seq=args.tokens_per_seq,
        emitted_tokens_per_seq=args.emitted_tokens_per_seq,
        max_sequences_per_dp=args.max_sequences_per_dp,
        phase=args.phase,
        scope=args.scope,
        output_budget=int(resolved_output),
        scenarios=scenarios,
        sensitivity=args.sensitivity,
        jobs=jobs,
    )
    result["sensitivity"] = args.sensitivity
    result["jobs"] = jobs
    return result


def _dispatch_catalog(args: argparse.Namespace) -> object:
    """Fast (< 50ms) discovery for models, hardware SKUs, topology presets, and quant modes."""
    want_hw = bool(getattr(args, "hardware", False))
    want_presets = bool(getattr(args, "presets", False))
    want_quant = bool(getattr(args, "quant", False))
    if not (want_hw or want_presets or want_quant):
        from mindie_sim.model_catalog import list_models, load_model
        return load_model(args.name).to_dict() if args.name else list_models()

    out: dict[str, Any] = {"schema": "mindie-sim.catalog/1"}
    if want_hw:
        from mindie_sim.backends.ascend import profiles as ascend_profiles
        from mindie_sim.backends.nvidia import profiles as nvidia_profiles
        from mindie_sim.compose.inputs import (
            BYTES_PER_GIB,
            DERIVED_ASCEND_SKUS,
            HARDWARE_ALIASES,
            HARDWARE_TOPOLOGY_ALIASES,
        )
        from mindie_sim.primitives import load_manifest

        calibrated = set((load_manifest().get("tables") or {}).keys())
        aliases_by_sku: dict[str, list[str]] = {}
        for alias, target in sorted(HARDWARE_ALIASES.items()):
            aliases_by_sku.setdefault(target, []).append(alias)
        for alias, (target, preset) in sorted(HARDWARE_TOPOLOGY_ALIASES.items()):
            tag = f"{alias} ({preset})"
            if tag not in aliases_by_sku.setdefault(target, []):
                aliases_by_sku[target].append(tag)
        skus: list[dict[str, Any]] = []
        reg = {**ascend_profiles(), **nvidia_profiles()}
        for sku_name, prof in sorted(reg.items()):
            dev = prof.get("device") or {}
            mem_b = float(dev.get("memory_bytes") or 0.0)
            hbm_bps = float((dev.get("rates") or {}).get("memory") or 0.0)
            res_units = dev.get("resource_units") or {}
            skus.append({
                "sku": sku_name,
                "architecture": prof.get("architecture", ""),
                "hbm_gib": round(mem_b / float(BYTES_PER_GIB), GIB_DECIMALS) if mem_b else None,
                "aic_count": dev.get("compute_units"),
                "aiv_count": res_units.get("vector"),
                "hbm_bw_tbps": round(hbm_bps / BYTES_PER_TB, TBPS_DECIMALS) if hbm_bps else None,
                "l0_calibrated": sku_name in calibrated,
                "aliases": aliases_by_sku.get(sku_name, []),
            })
        for sku_name, spec in sorted(DERIVED_ASCEND_SKUS.items()):
            skus.append({
                "sku": sku_name,
                "architecture": "derived",
                "base_sku": spec["base"],
                "hbm_gib": round(float(spec["memory_bytes"]) / float(BYTES_PER_GIB), GIB_DECIMALS),
                "aic_count": spec.get("aic_count"),
                "l0_calibrated": spec["base"] in calibrated,
                "aliases": aliases_by_sku.get(sku_name, []),
            })
        out["hardware"] = skus
    if want_presets:
        from mindie_sim.backends.ascend import ASCEND_TOPOLOGY_PRESETS
        out["presets"] = {name: dict(cfg) for name, cfg in sorted(ASCEND_TOPOLOGY_PRESETS.items())}
    if want_quant:
        from mindie_sim.catalog.quant import QUANTIZATION_PRESETS
        from mindie_sim.compose.inputs import _QUANT_ALIASES
        out["quant"] = {
            "presets": sorted(QUANTIZATION_PRESETS),
            "aliases": dict(sorted(_QUANT_ALIASES.items())),
            "cache_dtypes": ["bf16", "fp16", "fp8_e4m3", "fp8", "int8", "c8"],
        }
    return out


def _dispatch_solve(args: argparse.Namespace) -> object:
    from mindie_sim.solve import solve_deployment
    plan_path = getattr(args, "rewrite_plan", None)
    rewrite_plan = read_json(plan_path, bounded=True) if plan_path else None
    return solve_deployment(
        args.model,
        args.hardware,
        num_devices=args.num_devices,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        tpot_slo_ms=args.tpot_slo_ms,
        ttft_slo_ms=args.ttft_slo_ms,
        mtp_tokens=args.mtp_tokens,
        mtp_accept=args.mtp_accept,
        kv_pooling=args.kv_pooling,
        deployment=args.deployment,
        quant=args.quant,
        kv_cache_dtype=args.kv_cache_dtype,
        indexer_kv_dtype=args.indexer_kv_dtype,
        topology_preset=args.topology_preset,
        ranks_per_node=args.ranks_per_node,
        layers=args.layers,
        max_sequences_per_dp=args.max_sequences_per_dp,
        ablate_prs=bool(args.ablate_prs),
        pool_dram_ratio=args.pool_dram_ratio,
        pool_ssd_ratio=float(args.pool_ssd_ratio or 0.0),
        pool_prefix_hit_ratio=float(args.pool_prefix_hit_ratio or 0.0),
        ddr_bw_gbps=args.ddr_bw_gbps,
        ssd_bw_gbps=args.ssd_bw_gbps,
        kv_transfer_bw_gbps=args.kv_transfer_bw_gbps,
        raw_eager=bool(args.raw_eager),
        runtime_stack=getattr(args, "stack", None),
        xlsx_path=getattr(args, "xlsx", None),
        trace_path=getattr(args, "trace", None),
        compare_track=str(getattr(args, "compare_track", None) or "unmerged_pr"),
        target_track=str(getattr(args, "target_track", None) or "expected_ideal"),
        mode=str(getattr(args, "mode", None) or "modeling"),
        rewrite_plan=rewrite_plan,
    )


def _dispatch_core_analysis(args: argparse.Namespace) -> object | None:
    cmd = args.command
    if cmd == "call":
        from mindie_sim.prediction.call_builders import build_call, call_costs
        declaration = read_json(args.declaration)
        call = build_call(declaration)
        return {
            **call_costs(
                read_json(args.case), call.ops, recipes=call.recipes,
                include_submission=args.include_submission,
                **declaration.get("protocol", {}),
            ),
            "call": call.ledger,
            "limitations": list(call.limitations),
        }
    if cmd == "compare-calls":
        from mindie_sim.observations.calibration import compare_calls
        return compare_calls(
            read_json(args.case),
            read_json(args.protocol),
            root=args.root,
            include_submission=args.include_submission,
            submission=read_json(args.submission) if args.submission else None,
            max_predictions=args.max_predictions,
            max_conditions=args.max_conditions,
        )
    if cmd == "study":
        from mindie_sim.study import study
        packet = read_json(args.manifest)
        root = Path(args.manifest).resolve().parent
        candidates = [
            {**row, "case": read_json(root / row["case"])}
            if isinstance(row.get("case"), str) else row
            for row in packet["candidates"]
        ]
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return study(candidates, packet["probes"], primitives=store, **packet.get("options", {}))
    if cmd == "accuracy":
        from mindie_sim.observations.comparison import accuracy_report
        return accuracy_report(read_json(args.packet))
    if cmd == "graph":
        from mindie_sim.prediction.graph_analysis import graph_costs
        from mindie_sim.prediction.operator_analysis import target_graphs
        case = parse_case(_case_with_supply(args))
        graphs = target_graphs(
            case, phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            media_tokens_per_sequence=args.media_tokens_per_sequence,
            draft_prepare=args.draft_prepare,
            apply_body_defaults=args.apply_body_defaults,
        )
        inputs = {"analysis_input": args.input_ready_ms} if args.scope == "body" else None
        exported = (
            _export_structure_sidecar(
                args, case, phase=args.phase, tokens=args.tokens,
                context=args.context, sequences=args.sequences,
            )
            if args.export_structure else {}
        )
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return {
                **graph_costs(
                    case, graphs, primitives=store, external_inputs=inputs,
                    input_ready_ms=args.input_ready_ms,
                    include_submission=args.include_submission,
                    replay_scope=args.replay_scope, isolated=not args.skip_isolated,
                    trace_limit=args.event_limit,
                ),
                "phase": args.phase,
                "context": args.context,
                "graph_scope": args.scope,
                "tokens_per_sequence": args.tokens,
                "sequences_per_replica": args.sequences,
                "media_tokens_per_sequence": args.media_tokens_per_sequence,
                "draft_prepare": args.draft_prepare,
                "execution_geometry": {
                    "phase": args.phase,
                    "tokens": args.tokens,
                    "context": args.context,
                    "sequences": args.sequences,
                    "scope": args.scope,
                    "media_tokens_per_sequence": args.media_tokens_per_sequence,
                    "draft_prepare": args.draft_prepare,
                },
                "deployment_scope": "first_independent_dp_cohort",
                **exported,
            }
    if cmd == "operators":
        from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
        case = parse_case(_case_with_supply(args))
        ops = target_graphs(
            case, phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            media_tokens_per_sequence=args.media_tokens_per_sequence,
            draft_prepare=args.draft_prepare,
            apply_body_defaults=args.apply_body_defaults,
        )
        return {
            **operator_costs(case, ops),
            "graph_scope": args.scope,
            "tokens_per_sequence": args.tokens,
            "sequences": args.sequences,
            "sequences_per_replica": args.sequences,
            "draft_prepare": args.draft_prepare,
            "deployment_scope": "first_independent_dp_cohort",
        }
    if cmd == "predict":
        with PrimitiveStore(args.primitives) if args.primitives else nullcontext(None) as store:
            return predict(
                _case_with_supply(args), primitives=store,
                trace_limit=args.trace_limit, event_limit=args.event_limit,
                scope=args.scope, phase=args.phase, tokens=args.tokens,
                context=args.context, sequences=args.sequences,
            )
    if cmd == "inspect":
        return _inspect_case(read_json(args.case))
    return None


def _dispatch_observations(args: argparse.Namespace) -> object | None:
    cmd = args.command
    if cmd == "observations":
        from mindie_sim.observations.captures import ObservationIndex
        with ObservationIndex(args.index) as index:
            for directory in args.scan:
                index.scan(directory)
            if args.attach_profiler:
                index.attach_profiler_catalogue(args.attach_profiler)
            return index.find_sources(args.find) if args.find else index.inventory()
    if cmd == "profiler-rows":
        from itertools import islice
        from mindie_sim.observations.captures import ProfilerCatalogue, profiler_rows
        if Path(args.source).suffix == ".csv":
            return list(islice(profiler_rows(args.source), min(1000, args.limit)))
        with ProfilerCatalogue(args.source) as catalog:
            return list(catalog.variants(limit=args.limit, after=args.after))
    if cmd == "calibrate":
        import ijson
        with PrimitiveStore(args.store, writable=True) as store, Path(args.samples).open("rb") as stream:
            for sample in ijson.items(stream, "", multiple_values=True, use_float=True):
                key = PrimitiveKey.from_mapping(sample.pop("key"))
                store.record(key, **sample)
            return store.inventory()
    if cmd == "validate":
        from mindie_sim.observations.captures import compare, service_observation
        return compare(
            read_json(args.prediction), service_observation(args.observation),
            boundary=args.boundary,
        )
    if cmd == "bind-profiler":
        import ijson
        from mindie_sim.observations.calibration import import_profiler
        with PrimitiveStore(args.store, writable=True) as store, Path(args.bindings).open("rb") as stream:
            return import_profiler(
                args.catalogue, store,
                ijson.items(stream, "", multiple_values=True, use_float=True),
            )
    if cmd == "fit-rate":
        from mindie_sim.observations.captures import fit_resource_rate
        return fit_resource_rate(read_json(args.observations)["samples"])
    if cmd == "fit-service":
        from mindie_sim.observations.service_fitting import fit_service_curve
        return fit_service_curve(read_json(args.observations))
    return None


def _dispatch_compose_and_efficiency(args: argparse.Namespace) -> object | None:
    cmd = args.command
    if cmd == "compose":
        from .options import compose_case_kwargs
        from mindie_sim.compose import compose_case
        if not args.model:
            raise ValueError("compose_requires_model")
        return compose_case(
            model=args.model, hardware=args.hardware or "a3",
            **compose_case_kwargs(args),
        )
    if cmd == "feature-tracks":
        from .options import _linear_tp
        from mindie_sim.feature_mechanisms import compare_feature_tracks
        ft_kw: dict[str, Any] = dict(
            tp=args.tp or 1, dp=args.dp or 1, ep=args.ep or 1,
            pp=args.pp or 1, pcp=args.pcp or 1, dcp=args.dcp or 1,
            num_devices=args.num_devices, ranks_per_node=args.ranks_per_node,
            layers=args.layers, quant=args.quant,
            kv_cache_dtype=args.kv_cache_dtype,
            indexer_kv_dtype=args.indexer_kv_dtype,
            topology_preset=args.topology_preset,
            mtp_proposal_tokens=args.mtp_tokens or 0,
            linear_proj_tp=_linear_tp(args),
        )
        if args.mtp_accept is not None:
            ft_kw["mtp_acceptance_length"] = float(args.mtp_accept)
        for key in ("q_proj_tp", "o_proj_tp", "dcp_linear_tp",
                    "mlp_tp", "shared_expert_tp", "lm_head_tp"):
            val = getattr(args, key, None)
            if val is not None:
                ft_kw[key] = int(val)
        return compare_feature_tracks(
            args.model, args.hardware,
            runtime_stack=(args.stack_manifest or args.stack),
            phase=args.phase, tokens=args.tokens, context=args.context,
            sequences=args.sequences, scope=args.scope,
            ablate_prs=bool(args.ablate_prs),
            **ft_kw,
        )
    if cmd == "efficiency":
        from mindie_sim.prediction.efficiency import analyze_network_efficiency, fit_case_roofline_regimes
        case_data = read_json(args.case)
        if args.fit_targets:
            base = analyze_network_efficiency(
                case_data, phase=args.phase, tokens=args.tokens,
                context=args.context, sequences=args.sequences, scope=args.scope,
            )
            regimes = fit_case_roofline_regimes(case_data, baseline_analysis=base)
            return {
                **base,
                "baseline_analysis": base,
                "fitted_roofline_regimes": regimes["fitted_regimes"],
                "schema": "mindie-sim.efficiency/1",
            }
        return analyze_network_efficiency(
            case_data, phase=args.phase, tokens=args.tokens,
            context=args.context, sequences=args.sequences, scope=args.scope,
            observed_durations_ms=(
                read_json(args.observed_durations) if args.observed_durations else None
            ),
            pmu_by_op=read_json(args.pmu) if args.pmu else None,
            run_graph=not args.skip_graph,
        )
    return None


def dispatch(args: argparse.Namespace, /) -> object:
    """Execute one parsed command namespace and return its report payload."""
    cmd = args.command
    if cmd in EXPORT_COMMANDS:
        return run_export_command(args)
    if cmd == "catalog":
        return _dispatch_catalog(args)
    if cmd == "solve":
        return _dispatch_solve(args)
    if cmd == "sweep":
        return _dispatch_sweep(args)
    if cmd == "sla":
        return _dispatch_sla(args)
    if cmd in SKU_COMMANDS:
        return run_sku_command(args)
    if cmd in ONBOARDING_COMMANDS:
        return run_onboarding_command(args)
    if cmd == "build-accel":
        from mindie_sim.prediction.runtime.accel_build import build_accel
        return build_accel(force=bool(args.force), check_only=bool(args.check))
    for handler in (_dispatch_sdk, _dispatch_core_analysis, _dispatch_observations, _dispatch_compose_and_efficiency):
        result = handler(args)
        if result is not None:
            return result
    raise ValueError("unknown_command")


__all__ = [
    "CliUsageError",
    "MAX_CLI_JSON_BYTES",
    "dispatch",
    "emit",
    "load_cli_json",
    "read_json",
]
