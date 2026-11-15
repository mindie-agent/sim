"""CLI adapters for Day-0 graph optimization, EP/xPyD sizing, memory/KV ledgers, OpFlow, and OpFormulas.

[Role]
  Expose the public Pillar 2 / 4 / 6a / 8 / 9 / 10 simulation and operator
  engines as direct ``mindie-sim`` subcommands emitting one canonical JSON
  object to stdout or ``--output``.

[Invariants]
  * Every subcommand delegates directly to its authoritative library entry
    point without duplicating physics, defaults, or schema validation.
  * Exit code ``0`` indicates a completed report; exit code ``2`` indicates
    invalid usage, rejected schema input, or an explicit tool error envelope.
  * ``TimeoutError`` is never swallowed by CLI error formatting.

[Timeouts & Locks]
  Bounded in-process evaluation delegated to target modules; no locks or NPU contexts.

[Side-Effect Policy]
  Reads only explicit input paths/JSON strings and writes only ``--output`` when requested.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import sys
from typing import Any

from .__main__ import emit, read_json
from .schema.common import ModelingError

EXTENDED_COMMANDS = (
    "optimize-graph", "sweep-ep-knee", "size-xpyd",
    "memory-breakdown", "kv-state", "opflow", "op-formula",
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_json_object(raw: str | None, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{label}_requires_json_object")
    return dict(loaded)


def _parse_ep_candidates(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    parsed = [int(tok.strip()) for raw in raw_values for tok in str(raw).split(",") if tok.strip()]
    if not parsed:
        raise ValueError("ep_candidates_empty")
    return parsed


def build_parser(prog: str = "mindie-sim") -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog=prog)
    sub = root.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize-graph", help="run Day-0 canonical graph optimization")
    opt.add_argument("case")
    opt.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    opt.add_argument("--tokens", type=int, default=None)
    opt.add_argument("--sequences", type=int, default=None)
    opt.add_argument("--context", type=int, default=None)
    opt.add_argument("--weight-prefetch", action=argparse.BooleanOptionalAction, default=None)
    opt.add_argument("--output", "-o")

    knee = sub.add_parser("sweep-ep-knee", help="sweep Expert Parallelism knee across candidate world sizes")
    knee.add_argument("case")
    knee.add_argument("--ep-candidates", nargs="+", default=None)
    knee.add_argument("--batch-sequences", type=int, default=None)
    knee.add_argument("--context-tokens", type=int, default=None)
    knee.add_argument("--hccl-buffsize-mb", type=float, default=None)
    knee.add_argument("--output", "-o")

    xpyd = sub.add_parser("size-xpyd", help="solve disaggregated xPyD prefill/decode pool sizing")
    xpyd.add_argument("case")
    xpyd.add_argument("--total-npus", type=int, required=True)
    xpyd.add_argument("--isl", type=int, default=None)
    xpyd.add_argument("--osl", type=int, default=None)
    xpyd.add_argument("--ttft-slo-ms", type=float, default=None)
    xpyd.add_argument("--tpot-slo-ms", type=float, default=None)
    xpyd.add_argument("--prefix-hit-rate", type=float, default=0.0)
    xpyd.add_argument("--allow-unqualified", action="store_true")
    xpyd.add_argument("--output", "-o")

    for cmd, desc in (
        ("memory-breakdown", "compute per-rank HBM static/dynamic/workspace memory ledger"),
        ("kv-state", "evaluate hybrid KV and recurrent linear-attention state footprint"),
    ):
        node = sub.add_parser(cmd, help=desc)
        node.add_argument("case")
        node.add_argument("--output", "-o")

    flow = sub.add_parser("opflow", help="invoke a bounded OpFlow MCP tool")
    flow.add_argument("tool")
    flow.add_argument("--args-json", default=None)
    flow.add_argument("--output", "-o")

    formula = sub.add_parser("op-formula", help="evaluate closed-form operator bounds and diagnostics")
    formula.add_argument("operator")
    formula.add_argument("--params-json", default=None)
    formula.add_argument("--hardware", default=None)
    formula.add_argument("--output", "-o")
    return root


def _run_extended(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "optimize-graph":
        from .prediction.canonical_optimizer import _run_canonical
        from .prediction.fusion_bridge import canonical_plan
        from .prediction.graph_optimizer import run_optimizer
        from .prediction.operator_analysis import target_graphs
        from .prediction.structure import parse_case

        parsed = parse_case(read_json(args.case))
        reqs = parsed.workload.requests
        req = reqs[0] if reqs else None
        tokens = args.tokens if args.tokens is not None else (
            1 if args.phase == "decode" else (req.prompt_tokens if req else 1))
        sequences = len(reqs) or 1 if args.sequences is None else args.sequences
        context = args.context if args.context is not None else (
            (req.prompt_tokens if args.phase == "decode" else req.prefix_tokens) if req else 0)
        ops = next(iter(target_graphs(
            parsed, phase=args.phase, tokens=tokens, context=context, sequences=sequences, scope="body",
        ).values()))
        if args.weight_prefetch is False:
            defs = run_optimizer.__kwdefaults__ or {}
            plan = canonical_plan(parsed, ops, options={"hardware": {"prefetch_capacity_bytes": 0}})
            report = _run_canonical(
                plan, evaluation_budget=defs["evaluation_budget"],
                per_pass=defs["max_candidates_per_pass"], max_streams=None, wall_budget_s=defs["wall_budget_s"],
            )
        else:
            report = run_optimizer(parsed, ops)
        return report.to_dict()

    if args.command == "sweep-ep-knee":
        from .parallel_ep_knee import sweep_ep_knee

        candidates = _parse_ep_candidates(args.ep_candidates)
        kwargs = {k: v for k, v in (
            ("ep_candidates", candidates), ("batch_sequences", args.batch_sequences),
            ("context_tokens", args.context_tokens), ("hccl_buffsize_mb", args.hccl_buffsize_mb),
        ) if v is not None}
        return sweep_ep_knee(read_json(args.case), **kwargs)

    if args.command == "size-xpyd":
        from .prediction.structure import parse_case
        from .xpyd_pool_solver import solve_xpyd_pools

        raw_case = read_json(args.case)
        parsed = parse_case(raw_case)
        req = parsed.workload.requests[0]
        return solve_xpyd_pools(
            raw_case, total_npus=args.total_npus,
            isl=req.prompt_tokens if args.isl is None else args.isl,
            osl=req.output_tokens if args.osl is None else args.osl,
            prefix_hit_rate=args.prefix_hit_rate,
            ttft_slo_ms=parsed.slo.ttft_ms if args.ttft_slo_ms is None else args.ttft_slo_ms,
            tpot_slo_ms=parsed.slo.tpot_ms if args.tpot_slo_ms is None else args.tpot_slo_ms,
            allow_unqualified=bool(args.allow_unqualified),
        )

    if args.command == "memory-breakdown":
        from .memory_breakdown import analyze_memory_breakdown
        return analyze_memory_breakdown(read_json(args.case)).to_dict()

    if args.command == "kv-state":
        from .hybrid_kv_state_report import evaluate_hybrid_kv_state
        return evaluate_hybrid_kv_state(read_json(args.case))

    if args.command == "opflow":
        from .opflow.opflow_mcp import call_tool
        return call_tool(args.tool, _parse_json_object(args.args_json, "args_json"))

    if args.command == "op-formula":
        from .compose import HARDWARE_ALIASES
        from .op_formulas.engine import evaluate_with_diagnostics

        params = _parse_json_object(args.params_json, "params_json")
        sku_kw = {"sku": HARDWARE_ALIASES.get(args.hardware.strip().lower(), args.hardware.strip())} if args.hardware else {}
        evaluation, diagnostics = evaluate_with_diagnostics(args.operator, params, **sku_kw)
        return {**evaluation.as_dict(), "diagnostics": diagnostics}

    raise ValueError("unknown_extended_command")


def extended_main(argv: list[str] | None = None, *, prog: str = "mindie-sim") -> int:
    parsed_args = build_parser(prog=prog).parse_args(argv)
    try:
        result = _jsonable(_run_extended(parsed_args))
        emit(result, parsed_args.output)
        if parsed_args.command == "opflow" and not result.get("ok"):
            return 2
    except TimeoutError:
        raise
    except (ModelingError, KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


__all__ = ["EXTENDED_COMMANDS", "build_parser", "extended_main"]
