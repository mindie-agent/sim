"""Multi-SKU comparison and runtime-readiness CLI surface.

[Role] Register and dispatch the ``compare-skus``, ``doctor`` and
  ``remote-guide`` commands on the shared argument parser.
[Invariants] Comparison evaluates every SKU under one model/deployment config;
  the readiness probe never starts simulation or a device context.  Defaults are
  named here so the main CLI stays a thin command router.
[Timeouts & Locks] Argument construction is inert; dispatch delegates all
  bounded work to :mod:`mindie_sim.sku_comparison` or
  :mod:`mindie_sim.runtime_advisor`.
[Side-Effect Policy] Writes only the explicitly requested ``--output`` path via
  the caller's emitter; no other filesystem or device effects.
"""
from __future__ import annotations

#: Default latency budget of a multi-SKU comparison (ms).
DEFAULT_COMPARE_SLA_MS = 50.0

#: Commands owned by this module.
SKU_COMMANDS = frozenset({"compare-skus", "doctor", "remote-guide"})


def add_sku_commands(subparsers) -> None:
    """Register ``compare-skus`` and the two readiness-probe aliases."""
    compare = subparsers.add_parser(
        "compare-skus",
        help="compare two or more hardware SKUs under one model/deployment config")
    compare.add_argument("--model", required=True)
    compare.add_argument("--skus", nargs="+", required=True,
                         help="SKU names (a3, a5, ...) or custom hardware JSON paths")
    for flag in ("--tp", "--dp", "--ep", "--pp", "--pcp", "--num-devices",
                 "--layers"):
        compare.add_argument(flag, type=int, default=None)
    compare.add_argument("--sla-ms", type=float,
                         default=DEFAULT_COMPARE_SLA_MS)
    compare.add_argument("--max-batch", type=int, default=None)
    compare.add_argument("--sensitivity", choices=("baseline", "full"),
                         default="baseline")
    compare.add_argument("--scope", choices=("body", "model"), default="body")
    compare.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    compare.add_argument("--context", type=int, default=None)
    compare.add_argument("--query-tokens", type=int, default=None)
    compare.add_argument("--quant", default=None,
                         help="explicit quantization override for every SKU")
    compare.add_argument("--jobs", default="auto")
    compare.add_argument("--no-advisory", dest="emit_advisory",
                         action="store_false")
    compare.add_argument("--output", "-o")
    doctor = subparsers.add_parser(
        "doctor", help="probe CPU/RAM/C11 readiness and print agent offload guidance")
    doctor.add_argument("--no-compile", dest="compile_accel",
                        action="store_false")
    doctor.add_argument("--output", "-o")
    guide = subparsers.add_parser(
        "remote-guide", help="alias of `doctor` for agent offload")
    guide.add_argument("--no-compile", dest="compile_accel",
                       action="store_false")
    guide.add_argument("--output", "-o")


def run_sku_command(args):
    """Execute one command owned by this module."""
    if args.command == "compare-skus":
        from .prediction.efficiency import DEFAULT_MAX_SEQUENCES
        from .sku_comparison import compare_skus
        options = {flag: getattr(args, flag.lstrip("-").replace("-", "_"), None)
                   for flag in ("tp", "dp", "ep", "pp", "pcp", "num_devices",
                                "layers")}
        max_batch = (DEFAULT_MAX_SEQUENCES if args.max_batch is None
                     else args.max_batch)
        return compare_skus(
            args.model, args.skus, sla_ms=args.sla_ms, max_batch=max_batch,
            sensitivity=args.sensitivity, scope=args.scope, phase=args.phase,
            context=args.context, query_tokens_per_sequence=args.query_tokens,
            quant=args.quant, jobs=args.jobs,
            emit_advisory=bool(args.emit_advisory), **options)
    if args.command in {"doctor", "remote-guide"}:
        from .runtime_advisor import doctor_report
        report = doctor_report(compile_accel=bool(args.compile_accel))
        report["command"] = args.command
        return report
    raise ValueError(f"not_a_sku_command:{args.command}")
