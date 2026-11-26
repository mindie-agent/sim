"""CLI entrypoint for ``mindie-sim matrix`` multi-model operator matrix & bench-plan export.

[Role] Parse CLI arguments for ``mindie-sim matrix`` (catalog models, local HF
  ``config.json`` paths, Ascend SKUs, phases, sequence ranges with physics
  cliffs, concurrency ladders, parallel-arm presets, optional campaign
  reconciliation, ``--plan-only``, ``--compat-v1``, ``--emit-bench-plan``, and
  ``--tar-gz``) and dispatch cleanly to ``export_operator_matrix``.
[Invariants]
  * ``TimeoutError`` propagates before ``OSError``.
  * Separate internal ``--emit-bench-plan <dir>`` is enforced outside
    ``--out-dir``; outward bundles are always checked by
    ``assert_zero_internal_jargon``.
[Timeouts & Locks] Delegates bounded execution to ``export.operator_matrix`` and
  ``export.matrix_writer``; no locks or device contexts.
[Side-Effect Policy] Writes exclusively into ``--out-dir``, optional
  ``<out_dir>.tar.gz``, and optional ``--emit-bench-plan <dir>``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from mindie_sim.catalog import fleet
from mindie_sim.export.matrix_writer import export_operator_matrix
from mindie_sim.export.operator_matrix import (
    DEFAULT_LEGACY_SKUS,
    enumerate_parallel_arms,
    parse_parallel_arm_tokens,
    parse_seq_range,
)
from mindie_sim.schema.bench_matrix import (
    DEFAULT_ANCHOR_LADDER,
    DEFAULT_CLIFF_BUDGET_PER_ARM,
    MatrixSpec,
    ParallelArmSpec,
)
from mindie_sim.schema.common import ModelingError

MATRIX_BOOLEAN_FLAGS: frozenset[str] = frozenset({
    "--auto-cliffs",
    "--no-auto-cliffs",
    "--plan-only",
    "--compat-v1",
    "--tar-gz",
    "--no-fabric",
})


def _split_csv_tokens(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values:
        for part in str(raw or "").split(","):
            tok = part.strip()
            if tok and tok not in out:
                out.append(tok)
    return tuple(out)


def _expand_models(raw_models: Sequence[str]) -> tuple[str, ...]:
    tokens = _split_csv_tokens(raw_models)
    expanded: list[str] = []
    for tok in tokens:
        if tok.lower() == "all":
            for mid in sorted(fleet.CORE_CATALOG_ENTRIES):
                if mid not in expanded:
                    expanded.append(mid)
        elif tok not in expanded:
            expanded.append(tok)
    if not expanded:
        raise ModelingError("matrix_cli_models_required")
    return tuple(expanded)


def _parse_concurrencies(raw_values: Sequence[str]) -> tuple[int, ...]:
    tokens = _split_csv_tokens(raw_values)
    out: list[int] = []
    for tok in tokens:
        if tok.lower() == "auto_max":
            continue
        val = int(tok)
        if val < 1:
            raise ModelingError(f"matrix_cli_concurrency_invalid:{tok!r}")
        if val not in out:
            out.append(val)
    return tuple(out or (1,))


def _single_arm_spec(low: str, *, deployment_mode: str) -> ParallelArmSpec:
    presets: dict[str, ParallelArmSpec] = {
        "auto_sla": ParallelArmSpec(arm_id="auto_sla", auto_sla=True, deployment_mode=deployment_mode),
        "tp1": ParallelArmSpec(arm_id="tp1", dp=1, tp=1, ep=1, num_devices=1, deployment_mode=deployment_mode),
        "hisparse": ParallelArmSpec(arm_id="hisparse_tp8_ep8", dp=1, tp=8, ep=8, sparse_kv_offload=True, num_devices=8, deployment_mode=deployment_mode),
        "dp_x_dcp": ParallelArmSpec(arm_id="dp_dcp2_tp8_ep8", dp=1, tp=8, ep=8, dcp=2, num_devices=8, deployment_mode=deployment_mode),
        "dp_dcp": ParallelArmSpec(arm_id="dp_dcp2_tp8_ep8", dp=1, tp=8, ep=8, dcp=2, num_devices=8, deployment_mode=deployment_mode),
        "local_tp": ParallelArmSpec(arm_id="local_tp8_sh8", dp=1, tp=8, ep=8, q_proj_tp=8, o_proj_tp=8, dcp_linear_tp=8, shared_expert_tp=8, num_devices=8, deployment_mode=deployment_mode),
        "pcp_x_dcp": ParallelArmSpec(arm_id="pcp2_dcp2_tp4", dp=1, tp=4, ep=8, pcp=2, dcp=2, num_devices=8, deployment_mode=deployment_mode),
        "tp_x_ep": ParallelArmSpec(arm_id="tp8_ep8", dp=1, tp=8, ep=8, num_devices=8, deployment_mode=deployment_mode),
        "dp_x_ep": ParallelArmSpec(arm_id="dp8_ep8", dp=8, tp=1, ep=8, num_devices=8, deployment_mode=deployment_mode),
    }
    if low in presets:
        return presets[low]
    if low.startswith("tp") and low[2:].isdigit():
        t = max(1, int(low[2:]))
        return ParallelArmSpec(arm_id=f"tp{t}", dp=1, tp=t, ep=1, num_devices=t, deployment_mode=deployment_mode)
    raise ModelingError(f"matrix_cli_unknown_parallel_arm:{low!r}")


def _build_arm_specs(arm_tokens: Sequence[str], *, deployment_mode: str) -> tuple[ParallelArmSpec, ...]:
    parsed = parse_parallel_arm_tokens(arm_tokens)
    if not parsed or any(tok.lower() == "auto_production" for tok in parsed):
        return ()
    return tuple(_single_arm_spec(tok.lower(), deployment_mode=deployment_mode) for tok in parsed)


def build_matrix_parser(*, prog: str = "mindie-sim matrix") -> argparse.ArgumentParser:
    """Create the argument parser for ``mindie-sim matrix``."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Multi-model Ascend operator matrix, outward msprof package & deduplicated bench-plan/2 compiler",
    )
    parser.add_argument("--models", nargs="+", required=True, help="Catalog model IDs, 'all', or local HF config.json paths")
    parser.add_argument("--skus", nargs="+", default=["a5"], help="Target Ascend SKUs or aliases (e.g. a2,a3,a5)")
    parser.add_argument("--phases", nargs="+", default=["prefill,decode"], help="Execution phases: prefill,decode")
    parser.add_argument("--seq-range", default="512..400k", help="Sequence length range, e.g. 512..400k or 2048")
    parser.add_argument("--auto-cliffs", dest="auto_cliffs", action="store_true", default=True)
    parser.add_argument("--no-auto-cliffs", dest="auto_cliffs", action="store_false")
    parser.add_argument("--cliff-budget", type=int, default=DEFAULT_CLIFF_BUDGET_PER_ARM)
    parser.add_argument("--concurrencies", nargs="+", default=["1"], help="Comma-separated per-DP concurrencies")
    parser.add_argument("--parallel-arms", nargs="+", default=["auto_production"], help="Parallel arm presets (auto_production,auto_sla,tp_x_ep,dp_x_dcp,local_tp,pcp_x_dcp,hisparse,tp1)")
    parser.add_argument("--stages", nargs="+", default=["before"], help="Stages to evaluate (before,after)")
    parser.add_argument("--quant", default="auto_va", help="Quantization policy (auto_va|checkpoint|<preset>)")
    parser.add_argument("--mtp-tokens", type=int, default=0)
    parser.add_argument("--mtp-accept", type=float, default=None)
    parser.add_argument("--deployment-mode", default="colocated", choices=("colocated", "pd_disaggregated", "pd_prefill", "pd_decode"))
    parser.add_argument("--ttft-slo-ms", "--ttft-ms", dest="ttft_slo_ms", type=float, default=None)
    parser.add_argument("--tpot-slo-ms", "--tpot-ms", dest="tpot_slo_ms", type=float, default=None)
    parser.add_argument("--layers", type=int, default=None, help="Optional layer depth override for fast local probes")
    parser.add_argument("--feature-track", default="shipping", choices=("eager", "shipping", "proposed", "target"))
    parser.add_argument("--campaign", default=None, help="Optional mindie-bench-campaign/1 manifest JSON to attach silicon timings")
    parser.add_argument("--plan-only", action="store_true", help="Emit only deduplicated bench_plan_v2 JSON files")
    parser.add_argument("--compat-v1", action="store_true", help="Also emit mindie-bench-subgraph-plan/1 compat projection")
    parser.add_argument("--emit-bench-plan", dest="emit_bench_plan", default=None, help="Separate directory for internal bench_plan_v2 JSON (must be outside --out-dir)")
    parser.add_argument("--no-fabric", dest="include_fabric", action="store_false", default=True)
    parser.add_argument("--tar-gz", dest="tar_gz", action="store_true", default=False)
    parser.add_argument("--cache-partition", default="shared", choices=("shared", "train", "heldout"))
    parser.add_argument("--out-dir", "--out", dest="out_dir", required=True, help="Output directory")
    return parser


def matrix_main(argv: Sequence[str] | None = None, *, prog: str = "mindie-sim matrix") -> int:
    """CLI process entrypoint for ``mindie-sim matrix``."""
    parser = build_matrix_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        models = _expand_models(args.models)
        skus = _split_csv_tokens(args.skus)
        phases = tuple(p.lower() for p in _split_csv_tokens(args.phases))
        seq_min, seq_max = parse_seq_range(args.seq_range)
        concurrencies = _parse_concurrencies(args.concurrencies)
        stages = _split_csv_tokens(args.stages) or ("before",)
        arms = _build_arm_specs(args.parallel_arms, deployment_mode=args.deployment_mode)
        spec = MatrixSpec(
            models=models,
            skus=skus,
            phases=phases,  # type: ignore[arg-type]
            seq_min=seq_min,
            seq_max=seq_max,
            anchor_ladder=DEFAULT_ANCHOR_LADDER,
            auto_physical_cliffs=bool(args.auto_cliffs),
            cliff_budget_per_arm=int(args.cliff_budget),
            concurrencies=concurrencies,
            parallel_arms=arms,
            stages=stages,
            quant_policy=str(args.quant),
            mtp_proposal_tokens=int(args.mtp_tokens),
            mtp_acceptance_length=args.mtp_accept,
            deployment_mode=str(args.deployment_mode),
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            cache_partition=args.cache_partition,
        )
        receipt = export_operator_matrix(
            spec,
            Path(args.out_dir),
            campaign=args.campaign,
            plan_only=bool(args.plan_only),
            compat_v1=bool(args.compat_v1),
            emit_bench_plan_dir=args.emit_bench_plan,
            include_fabric=bool(args.include_fabric),
            tar_gz=bool(args.tar_gz),
            layers=args.layers,
            feature_track=str(args.feature_track),
        )
        print(json.dumps({"status": "ok", **receipt.to_dict()}, sort_keys=True))
        return 0
    except TimeoutError:
        raise
    except (ModelingError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def export_subgraph_bench_plan_cli(argv: Sequence[str] | None = None) -> int:
    """Backward-compatible CLI for ``tools/export_subgraph_bench_plan.py``."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--out-dir" in raw or "--seq-range" in raw or "--parallel-arms" in raw:
        if "--plan-only" not in raw:
            raw = ["--plan-only", "--compat-v1", *raw]
        return matrix_main(raw, prog="export_subgraph_bench_plan")
    parser = argparse.ArgumentParser(
        prog="export_subgraph_bench_plan",
        description="Export deduplicated bench_plan_v2 and legacy subgraph_plan_v1 artifacts.",
    )
    parser.add_argument("--catalog", choices=("maintained",), default="maintained")
    parser.add_argument("--skus", nargs="+", default=list(DEFAULT_LEGACY_SKUS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--geometries", nargs="+", default=["prefill", "decode"])
    parser.add_argument("--tracks", nargs="+", default=["merged", "unmerged_pr"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-heldout-geometry-probes", action="store_true")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(raw)
    models = list(args.models) if args.models else ["all"]
    phases = [g for g in args.geometries if g in {"prefill", "decode"}] or ["prefill", "decode"]
    translated = [
        "--models", *models,
        "--skus", *args.skus,
        "--phases", ",".join(phases),
        "--plan-only", "--compat-v1",
        "--out-dir", str(args.out),
    ]
    if args.no_heldout_geometry_probes:
        translated.append("--no-auto-cliffs")
    if args.layers is not None:
        translated.extend(["--layers", str(int(args.layers))])
    return matrix_main(translated, prog="export_subgraph_bench_plan")


__all__ = [
    "MATRIX_BOOLEAN_FLAGS",
    "build_matrix_parser",
    "enumerate_parallel_arms",
    "export_subgraph_bench_plan_cli",
    "matrix_main",
]
