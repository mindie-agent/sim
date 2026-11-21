"""CLI parsers and dispatchers for ``onboard-sku`` and ``onboard-operator``.

[Role] Register ``onboard-sku`` and ``onboard-operator`` subcommands on the
  shared ``mindie-sim`` argument parser and provide standalone CLI entrypoints
  reused by ``tools/onboard_sku.py`` and ``tools/onboard_operator.py``.
[Invariants] Both ``mindie-sim onboard-sku`` / ``tools/onboard_sku.py`` and
  ``mindie-sim onboard-operator`` / ``tools/onboard_operator.py`` share one
  authoritative flag grammar and return exit code ``2`` whenever any gate,
  contract check, ``calibrated_count == 0``, or ``plan_only`` fails
  (``ok is False``). ``TimeoutError`` propagates cleanly.
[Timeouts & Locks] Pure argument construction; delegates bounded execution to
  ``mindie_sim.onboarding``.
[Side-Effect Policy] Writes only caller-requested output paths via the
  onboarding pipelines or CLI emitter.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from .onboarding import (
    CONTEXT_MAX_TOKENS,
    CONTEXT_MIN_TOKENS,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_FABRIC_BUDGET_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DEFAULT_L0_BUDGET_S,
    DEFAULT_PROBE_TIMEOUT_S,
    bench_and_score_operator,
    calibrate_l0_and_fabric,
    certify_operator,
    generate_operator_cliff_plan,
    probe_soc,
    run_operator_onboarding,
    run_sku_onboarding,
    scaffold_operator,
    validate_512_400k,
    validate_operator_contract,
    verify_sku_gates,
)
from .onboarding._lean_and_receipts import use_stage_key_file
from .onboarding.contracts import read_bounded_json

ONBOARDING_COMMANDS = frozenset({"onboard-sku", "onboard-operator"})
ONBOARDING_BOOLEAN_FLAGS = frozenset({
    "--allow-scaffold",
    "--live-bench",
    "--live-profiler",
    "--skip-repo-gates",
    "--overwrite",
    "--no-scaffold-test",
    "--no-require-sim-owner",
    "--no-require-taxonomy",
    "--no-require-profiler-builder",
    "--no-require-tiling-axes",
    "--require-mounted-refs",
})
DEFAULT_TILE_DIM = 128
CLI_FAILURE_EXIT = 2


def _parse_repo_map(entries: Sequence[str] | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for raw in entries or ():
        name, sep, path_str = raw.partition("=")
        name, path_str = name.strip(), path_str.strip()
        if not sep or not name or not path_str:
            raise ValueError(f"invalid_repo_map_entry:{raw}")
        mapping[name] = Path(path_str)
    return mapping


def _configure_sku_stages(parser: argparse.ArgumentParser) -> None:
    stages = parser.add_subparsers(dest="stage", required=True)

    p1 = stages.add_parser("probe-soc", help="probe SoC truth and validate/scaffold ascend.json")
    p1.add_argument("--endpoint-alias", default="")
    p1.add_argument("--probe-json", default=None)
    p1.add_argument("--profiles-path", default=None)
    p1.add_argument("--scaffold-out", default=None)
    p1.add_argument("--allow-scaffold", action="store_true")
    p1.add_argument("--base-template-sku", default="")
    p1.add_argument("--container", default="")
    p1.add_argument("--npu-id", type=int, default=0)
    p1.add_argument("--campaign-id", default="camp-probe-soc")
    p1.add_argument("--timeout-s", type=int, default=DEFAULT_PROBE_TIMEOUT_S)
    p1.add_argument("--output", "-o", default=None)

    p2 = stages.add_parser("calibrate-l0-and-fabric", help="calibrate P01..P20 L0 + fabric/DMA via canonical sealer")
    p2.add_argument("--sku", required=True)
    p2.add_argument("--endpoint-alias", default="")
    p2.add_argument("--raw-l0-json", default=None)
    p2.add_argument("--raw-fabric-json", default=None)
    p2.add_argument("--out-primitives", default=None)
    p2.add_argument("--manifest", default=None)
    p2.add_argument("--train-manifest-out", default=None)
    p2.add_argument("--campaign-id", default="camp-onboard-l0")
    p2.add_argument("--cann-version", default="unmeasured")
    p2.add_argument("--l0-budget-s", type=int, default=DEFAULT_L0_BUDGET_S)
    p2.add_argument("--fabric-budget-s", type=int, default=DEFAULT_FABRIC_BUDGET_S)
    p2.add_argument("--cleanup-grace-s", type=int, default=DEFAULT_CLEANUP_GRACE_S)
    p2.add_argument("--outer-timeout-s", type=int, default=None)
    p2.add_argument("--live-bench", action="store_true")
    p2.add_argument("--repo-root", default=None)
    p2.add_argument("--profiler-root", default=None)
    p2.add_argument("--coordinator-bin", default="")
    p2.add_argument("--output", "-o", default=None)

    p3 = stages.add_parser("validate-512-400k", help="score disjoint 2D 512..400K Anchor + Cliff + L1/L2/L3")
    p3.add_argument("--sku", required=True)
    p3.add_argument("--observations-json", default=None)
    p3.add_argument("--capture-root", default=None)
    p3.add_argument("--train-manifest-json", default=None)
    p3.add_argument("--primitives-json", default=None)
    p3.add_argument("--plan-out", default=None)
    p3.add_argument("--report-out", default=None)
    p3.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    p3.add_argument("--campaign-id", default="camp-onboard-512-400k")
    p3.add_argument("--live-profiler", action="store_true")
    p3.add_argument("--endpoint-alias", default="")
    p3.add_argument("--profiler-root", default=None)
    p3.add_argument("--coordinator-bin", default="")
    p3.add_argument("--stage-key-file", default=None)
    p3.add_argument("--output", "-o", default=None)

    p4 = stages.add_parser("verify-gates", help="verify pointwise multilevel, witnessed Lean, governance, orthogonality")
    p4.add_argument("--sku", required=True)
    p4.add_argument("--validation-report-json", default=None)
    p4.add_argument("--multilevel-manifest-json", default=None)
    p4.add_argument("--capture-root", default=None)
    p4.add_argument("--fits-json", default=None)
    p4.add_argument("--lean-bounds-json", default=None)
    p4.add_argument("--primitives-json", default=None)
    p4.add_argument("--expected-cann-version", default="")
    p4.add_argument("--repo-root", default=None)
    p4.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    p4.add_argument("--skip-repo-gates", action="store_true")
    p4.add_argument("--orthogonality-base", default=None)
    p4.add_argument("--out-receipt", default=None)
    p4.add_argument("--diagnosis-out", default=None)
    p4.add_argument("--stage-key-file", default=None)
    p4.add_argument("--output", "-o", default=None)

    pa = stages.add_parser("run-all", help="execute all 4 SKU onboarding stages end-to-end")
    pa.add_argument("--sku", default=None)
    pa.add_argument("--endpoint-alias", default="")
    pa.add_argument("--probe-json", default=None)
    pa.add_argument("--raw-l0-json", default=None)
    pa.add_argument("--raw-fabric-json", default=None)
    pa.add_argument("--observations-json", default=None)
    pa.add_argument("--capture-root", default=None)
    pa.add_argument("--lean-bounds-json", default=None)
    pa.add_argument("--out-dir", default=None)
    pa.add_argument("--repo-root", default=None)
    pa.add_argument("--profiler-root", default=None)
    pa.add_argument("--coordinator-bin", default="")
    pa.add_argument("--allow-scaffold", action="store_true")
    pa.add_argument("--base-template-sku", default="")
    pa.add_argument("--live-bench", action="store_true")
    pa.add_argument("--live-profiler", action="store_true")
    pa.add_argument("--skip-repo-gates", action="store_true")
    pa.add_argument("--orthogonality-base", default=None)
    pa.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    pa.add_argument("--cann-version", default="unmeasured")
    pa.add_argument("--output", "-o", default=None)


def _configure_operator_stages(parser: argparse.ArgumentParser) -> None:
    stages = parser.add_subparsers(dest="stage", required=True)

    p1a = stages.add_parser("scaffold", help="scaffold governed operator package skeleton with TilingAxes")
    p1a.add_argument("--family", required=True)
    p1a.add_argument("--upstream-abi-arch22", required=True)
    p1a.add_argument("--upstream-abi-arch35", required=True)
    p1a.add_argument("--source-refs-json", required=True)
    p1a.add_argument("--target-root", default=".")
    p1a.add_argument("--target-kind", choices=("operator_family", "template"), default="operator_family")
    p1a.add_argument("--profiler-builder", default="")
    p1a.add_argument("--tile-dim", type=int, default=DEFAULT_TILE_DIM)
    p1a.add_argument("--overwrite", action="store_true")
    p1a.add_argument("--no-scaffold-test", dest="scaffold_test", action="store_false")
    p1a.add_argument("--output", "-o", default=None)

    p1b = stages.add_parser("validate-contract", help="validate ABI, SourceRef, real Sim & Profiler owners, taxonomy")
    p1b.add_argument("--target", required=True)
    p1b.add_argument("--repo-root", default=None)
    p1b.add_argument("--refs-root", default=None)
    p1b.add_argument("--repo-map", action="append", default=[])
    p1b.add_argument("--profiler-root", default=None)
    p1b.add_argument("--profiler-family", default="")
    p1b.add_argument("--profiler-recipe", default="")
    p1b.add_argument("--profiler-impl", default="")
    p1b.add_argument("--template-id", default="")
    p1b.add_argument("--no-require-sim-owner", dest="require_sim_owner", action="store_false")
    p1b.add_argument("--no-require-taxonomy", dest="require_taxonomy", action="store_false")
    p1b.add_argument("--no-require-profiler-builder", dest="require_profiler_builder", action="store_false")
    p1b.add_argument("--no-require-tiling-axes", dest="require_tiling_axes", action="store_false")
    p1b.add_argument("--require-mounted-refs", action="store_true")
    p1b.add_argument("--stage-key-file", default=None)
    p1b.add_argument("--output", "-o", default=None)

    p2 = stages.add_parser("generate-cliff-plan", help="synthesize disjoint 2D cliff + ABI-rejection plan")
    p2.add_argument("--operator-id", required=True)
    p2.add_argument("--family", required=True)
    p2.add_argument("--sku", required=True)
    p2.add_argument("--tile-dim", type=int, required=True)
    p2.add_argument("--core-count", type=int, required=True)
    p2.add_argument("--l2-bytes", type=int, required=True)
    p2.add_argument("--bytes-per-token", type=int, default=0)
    p2.add_argument("--topk-threshold", type=int, default=None)
    p2.add_argument("--context-min", type=int, default=CONTEXT_MIN_TOKENS)
    p2.add_argument("--context-max", type=int, default=CONTEXT_MAX_TOKENS)
    p2.add_argument("--campaign-id", default="onboard-op")
    p2.add_argument("--plan-out", default=None)
    p2.add_argument("--output", "-o", default=None)

    p3 = stages.add_parser("bench-and-score", help="score sealed white-box Engine predictions vs silicon p50")
    p3.add_argument("--cliff-plan-json", required=True)
    p3.add_argument("--sku", required=True)
    p3.add_argument("--captured-json", default=None)
    p3.add_argument("--live-profiler", action="store_true")
    p3.add_argument("--endpoint-alias", default="")
    p3.add_argument("--profiler-root", default=None)
    p3.add_argument("--coordinator-bin", default="")
    p3.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    p3.add_argument("--out-score", default=None)
    p3.add_argument("--capture-root", default=None)
    p3.add_argument("--stage-key-file", default=None)
    p3.add_argument("--output", "-o", default=None)

    p4 = stages.add_parser("certify", help="certify pointwise <10% cliff APE, witnessed Lean, strict gates")
    p4.add_argument("--operator-id", required=True)
    p4.add_argument("--sku", required=True)
    p4.add_argument("--contract-report-json", required=True)
    p4.add_argument("--score-report-json", required=True)
    p4.add_argument("--lean-bounds-json", required=True)
    p4.add_argument("--capture-root", default=None)
    p4.add_argument("--repo-root", default=None)
    p4.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    p4.add_argument("--skip-repo-gates", action="store_true")
    p4.add_argument("--orthogonality-base", default=None)
    p4.add_argument("--out-receipt", default=None)
    p4.add_argument("--diagnosis-out", default=None)
    p4.add_argument("--stage-key-file", default=None)
    p4.add_argument("--output", "-o", default=None)

    pa = stages.add_parser("run-all", help="execute all 4 operator onboarding stages end-to-end")
    pa.add_argument("--operator-id", required=True)
    pa.add_argument("--family", default="")
    pa.add_argument("--target", required=True)
    pa.add_argument("--sku", required=True)
    pa.add_argument("--tile-dim", type=int, required=True)
    pa.add_argument("--core-count", type=int, required=True)
    pa.add_argument("--l2-bytes", type=int, required=True)
    pa.add_argument("--bytes-per-token", type=int, default=0)
    pa.add_argument("--topk-threshold", type=int, default=None)
    pa.add_argument("--captured-json", default=None)
    pa.add_argument("--lean-bounds-json", default=None)
    pa.add_argument("--repo-root", default=None)
    pa.add_argument("--refs-root", default=None)
    pa.add_argument("--repo-map", action="append", default=[])
    pa.add_argument("--profiler-root", default=None)
    pa.add_argument("--profiler-family", default="")
    pa.add_argument("--profiler-recipe", default="")
    pa.add_argument("--profiler-impl", default="")
    pa.add_argument("--template-id", default="")
    pa.add_argument("--endpoint-alias", default="")
    pa.add_argument("--coordinator-bin", default="")
    pa.add_argument("--live-profiler", action="store_true")
    pa.add_argument("--no-require-sim-owner", dest="require_sim_owner", action="store_false")
    pa.add_argument("--no-require-taxonomy", dest="require_taxonomy", action="store_false")
    pa.add_argument("--no-require-profiler-builder", dest="require_profiler_builder", action="store_false")
    pa.add_argument("--no-require-tiling-axes", dest="require_tiling_axes", action="store_false")
    pa.add_argument("--require-mounted-refs", action="store_true")
    pa.add_argument("--out-dir", default=None)
    pa.add_argument("--capture-root", default=None)
    pa.add_argument("--skip-repo-gates", action="store_true")
    pa.add_argument("--orthogonality-base", default=None)
    pa.add_argument("--threshold-pct", type=float, default=DEFAULT_HELDOUT_APE_THRESHOLD_PCT)
    pa.add_argument("--output", "-o", default=None)


def add_onboarding_commands(subparsers) -> None:
    """Register ``onboard-sku`` and ``onboard-operator`` on the shared parser."""
    sku_parser = subparsers.add_parser(
        "onboard-sku",
        help="standardized 4-stage turnkey New-SKU onboarding pipeline",
    )
    _configure_sku_stages(sku_parser)
    op_parser = subparsers.add_parser(
        "onboard-operator",
        help="standardized 4-stage turnkey New-Operator onboarding pipeline",
    )
    _configure_operator_stages(op_parser)


def _run_sku_stage(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    if stage == "probe-soc":
        return probe_soc(
            endpoint_alias=args.endpoint_alias,
            probe_json=args.probe_json,
            profiles_path=args.profiles_path,
            scaffold_out=args.scaffold_out,
            allow_scaffold=bool(args.allow_scaffold),
            base_template_sku=args.base_template_sku,
            container=args.container,
            npu_id=args.npu_id,
            campaign_id=args.campaign_id,
            timeout_s=args.timeout_s,
        )
    if stage == "calibrate-l0-and-fabric":
        return calibrate_l0_and_fabric(
            sku=args.sku,
            endpoint_alias=args.endpoint_alias,
            raw_l0_json=args.raw_l0_json,
            raw_fabric_json=args.raw_fabric_json,
            out_primitives=args.out_primitives,
            manifest_path=args.manifest,
            train_manifest_out=args.train_manifest_out,
            campaign_id=args.campaign_id,
            cann_version=args.cann_version,
            l0_budget_s=args.l0_budget_s,
            fabric_budget_s=args.fabric_budget_s,
            cleanup_grace_s=args.cleanup_grace_s,
            outer_timeout_s=args.outer_timeout_s,
            live_bench=bool(args.live_bench),
            repo_root=args.repo_root,
            profiler_root=args.profiler_root,
            coordinator_bin=args.coordinator_bin,
        )
    if stage == "validate-512-400k":
        return validate_512_400k(
            sku=args.sku,
            observations_json=args.observations_json,
            train_manifest_json=args.train_manifest_json,
            primitives_json=args.primitives_json,
            plan_out=args.plan_out,
            report_out=args.report_out,
            threshold_pct=args.threshold_pct,
            campaign_id=args.campaign_id,
            live_profiler=bool(args.live_profiler),
            endpoint_alias=args.endpoint_alias,
            profiler_root=args.profiler_root,
            coordinator_bin=args.coordinator_bin,
            capture_root=args.capture_root,
            require_capture_files=True,
        )
    if stage == "verify-gates":
        return verify_sku_gates(
            sku=args.sku,
            validation_report_json=args.validation_report_json,
            multilevel_manifest_json=args.multilevel_manifest_json,
            fits_json=args.fits_json,
            lean_bounds_json=args.lean_bounds_json,
            primitives_json=args.primitives_json,
            expected_cann_version=args.expected_cann_version,
            capture_root=args.capture_root,
            require_capture_files=True,
            repo_root=args.repo_root,
            threshold_pct=args.threshold_pct,
            run_repo_gates=not bool(args.skip_repo_gates),
            orthogonality_base=args.orthogonality_base,
            out_receipt=args.out_receipt,
            diagnosis_out=args.diagnosis_out,
        )
    if stage == "run-all":
        return run_sku_onboarding(
            sku=args.sku,
            endpoint_alias=args.endpoint_alias,
            probe_json=args.probe_json,
            raw_l0_json=args.raw_l0_json,
            raw_fabric_json=args.raw_fabric_json,
            observations_json=args.observations_json,
            lean_bounds_json=args.lean_bounds_json,
            out_dir=args.out_dir,
            repo_root=args.repo_root,
            allow_scaffold=bool(args.allow_scaffold),
            base_template_sku=args.base_template_sku,
            run_repo_gates=not bool(args.skip_repo_gates),
            orthogonality_base=args.orthogonality_base,
            threshold_pct=args.threshold_pct,
            cann_version=args.cann_version,
            live_bench=bool(args.live_bench),
            live_profiler=bool(args.live_profiler),
            profiler_root=args.profiler_root,
            coordinator_bin=args.coordinator_bin,
            capture_root=args.capture_root,
            require_capture_files=True,
        )
    raise ValueError(f"unknown_sku_stage:{stage}")


def _run_operator_stage(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    if stage == "scaffold":
        raw_refs = read_bounded_json(args.source_refs_json)
        refs = raw_refs.get("source_refs", raw_refs) if isinstance(raw_refs, Mapping) else raw_refs
        return scaffold_operator(
            family=args.family,
            upstream_abi_arch22=args.upstream_abi_arch22,
            upstream_abi_arch35=args.upstream_abi_arch35,
            source_refs=list(refs),
            target_root=args.target_root,
            target_kind=args.target_kind,
            profiler_builder=args.profiler_builder,
            tile_dim=args.tile_dim,
            overwrite=bool(args.overwrite),
            scaffold_test=bool(args.scaffold_test),
        )
    if stage == "validate-contract":
        return validate_operator_contract(
            target_path=args.target,
            repo_root=args.repo_root,
            refs_root=args.refs_root,
            repo_map=_parse_repo_map(args.repo_map) or None,
            profiler_root=args.profiler_root,
            profiler_family=args.profiler_family,
            profiler_recipe=args.profiler_recipe,
            profiler_impl=args.profiler_impl,
            template_id=args.template_id,
            require_sim_owner=bool(args.require_sim_owner),
            require_taxonomy=bool(args.require_taxonomy),
            require_profiler_builder=bool(args.require_profiler_builder),
            require_tiling_axes=bool(args.require_tiling_axes),
            require_mounted_refs=bool(args.require_mounted_refs),
        )
    if stage == "generate-cliff-plan":
        return generate_operator_cliff_plan(
            operator_id=args.operator_id,
            family=args.family,
            sku=args.sku,
            tile_dim=args.tile_dim,
            core_count=args.core_count,
            l2_bytes=args.l2_bytes,
            bytes_per_token=args.bytes_per_token,
            topk_threshold=args.topk_threshold,
            context_min=args.context_min,
            context_max=args.context_max,
            campaign_id=args.campaign_id,
            plan_out=args.plan_out,
        )
    if stage == "bench-and-score":
        return bench_and_score_operator(
            cliff_plan=args.cliff_plan_json,
            sku=args.sku,
            captured_json=args.captured_json,
            live_profiler=bool(args.live_profiler),
            endpoint_alias=args.endpoint_alias,
            profiler_root=args.profiler_root,
            coordinator_bin=args.coordinator_bin,
            threshold_pct=args.threshold_pct,
            out_score=args.out_score,
            capture_root=args.capture_root,
            require_capture_files=True,
        )
    if stage == "certify":
        return certify_operator(
            operator_id=args.operator_id,
            sku=args.sku,
            contract_report=args.contract_report_json,
            score_report=args.score_report_json,
            lean_bounds_json=args.lean_bounds_json,
            capture_root=args.capture_root,
            require_capture_files=True,
            repo_root=args.repo_root,
            threshold_pct=args.threshold_pct,
            run_repo_gates=not bool(args.skip_repo_gates),
            orthogonality_base=args.orthogonality_base,
            out_receipt=args.out_receipt,
            diagnosis_out=args.diagnosis_out,
        )
    if stage == "run-all":
        inferred_family = args.family or Path(args.target).stem.removesuffix("_spec")
        return run_operator_onboarding(
            operator_id=args.operator_id,
            family=inferred_family,
            target_path=args.target,
            sku=args.sku,
            tile_dim=args.tile_dim,
            core_count=args.core_count,
            l2_bytes=args.l2_bytes,
            bytes_per_token=args.bytes_per_token,
            topk_threshold=args.topk_threshold,
            captured_json=args.captured_json,
            lean_bounds_json=args.lean_bounds_json,
            repo_root=args.repo_root,
            refs_root=args.refs_root,
            repo_map=_parse_repo_map(args.repo_map) or None,
            profiler_root=args.profiler_root,
            profiler_family=args.profiler_family,
            profiler_recipe=args.profiler_recipe,
            profiler_impl=args.profiler_impl,
            template_id=args.template_id,
            require_sim_owner=bool(args.require_sim_owner),
            require_taxonomy=bool(args.require_taxonomy),
            require_profiler_builder=bool(args.require_profiler_builder),
            require_tiling_axes=bool(args.require_tiling_axes),
            require_mounted_refs=bool(args.require_mounted_refs),
            out_dir=args.out_dir,
            run_repo_gates=not bool(args.skip_repo_gates),
            orthogonality_base=args.orthogonality_base,
            threshold_pct=args.threshold_pct,
            live_profiler=bool(args.live_profiler),
            endpoint_alias=args.endpoint_alias,
            coordinator_bin=args.coordinator_bin,
            capture_root=args.capture_root,
            require_capture_files=True,
        )
    raise ValueError(f"unknown_operator_stage:{stage}")


def run_onboarding_command(args: argparse.Namespace) -> dict[str, Any]:
    """Execute ``onboard-sku`` or ``onboard-operator`` from the main CLI."""
    stage_key_file = getattr(args, "stage_key_file", None)
    create_key = getattr(args, "stage", "") in {"validate-512-400k", "validate-contract", "bench-and-score"}
    with use_stage_key_file(stage_key_file, create_if_missing=create_key):
        if args.command == "onboard-sku":
            return _run_sku_stage(args)
        if args.command == "onboard-operator":
            return _run_operator_stage(args)
    raise ValueError(f"not_an_onboarding_command:{args.command}")


def _standalone_main(
    command: str, argv: Sequence[str] | None, *, prog: str
) -> int:
    parser = argparse.ArgumentParser(prog=prog)
    if command == "onboard-sku":
        _configure_sku_stages(parser)
    else:
        _configure_operator_stages(parser)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    args.command = command
    try:
        result = run_onboarding_command(args)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if getattr(args, "output", None):
            Path(args.output).write_bytes(rendered.encode("utf-8"))
        else:
            sys.stdout.write(rendered)
        return 0 if result.get("ok", True) else CLI_FAILURE_EXIT
    except TimeoutError:
        raise
    except (ValueError, TypeError, OSError) as exc:
        sys.stderr.write(json.dumps({"status": "failed", "ok": False, "error": str(exc)}) + "\n")
        return CLI_FAILURE_EXIT


def sku_main(argv: Sequence[str] | None = None) -> int:
    """Standalone entrypoint for ``tools/onboard_sku.py``."""
    return _standalone_main("onboard-sku", argv, prog="onboard_sku.py")


def operator_main(argv: Sequence[str] | None = None) -> int:
    """Standalone entrypoint for ``tools/onboard_operator.py``."""
    return _standalone_main("onboard-operator", argv, prog="onboard_operator.py")


__all__ = [
    "ONBOARDING_BOOLEAN_FLAGS",
    "ONBOARDING_COMMANDS",
    "add_onboarding_commands",
    "operator_main",
    "run_onboarding_command",
    "sku_main",
]
