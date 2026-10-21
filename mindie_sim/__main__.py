"""Command-line entry point for ``python -m mindie_sim`` and ``mindie-sim``.

[Role]
  Dispatch the four stateless subcommands ``hardware``, ``pipeline``,
  ``subgraph`` and ``net``; load explicit JSON inputs; print or write the
  structured JSON report.

[Invariants]
  * The CLI never fabricates a missing duration, capacity or provider value.
  * Known validation failures return a structured error with a stable reason
    and a non-zero exit code; they are not printed as successful empty reports.
  * Output is one JSON document; diagnostics go to the same document so a
    caller can distinguish modeled, uncalibrated and infeasible states.

[Timeouts & Locks]
  File reads are bounded by the caller's path.  Provider construction may run
  bounded git probes; no network, service, lock or sleep is started here.

[Side-Effect Policy]
  Reads the explicit input paths and writes only the explicit ``--output`` path.
  Never changes device state, installs packages, or mutates source checkouts.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .hardware import (
    A5_FLASHMLA_C8_GOLDEN_WORKLOAD,
    BUILTIN_SKU_CATALOG,
    HardwareError,
    default_link_ceilings,
    hardware_report,
    is_ambiguous_family_token,
    load_npu_smi_snapshot,
    parse_cannsim_link_ceilings,
    parse_npu_smi_topo,
    parse_pcie_link_info,
    parse_platform_ini,
)
from .net import NetError, net_report
from .pipeline import PipelineError, compare_pipeline_variants, solve_periodic_pipeline
from .subgraph import SubgraphError, analyze_subgraph_fusion
from .a3_types import A3InputError, CaseSpec
from .a3_framework import predict
from .a3_should import should_cost
from .a3_search import search as a3_search
from .a3_cost import CalibrationStore
from .a3_calibration_store import SQLiteCalibrationStore
from .a3_measurement import record_prediction_misses
from dataclasses import replace as _dataclass_replace
from .a3_measurement import CpuMatmulRunner, TorchNpuRunner, run_pending
from .a3_real_data import (
    CommunicationCalibration, CvCalibration, import_a3_measurements,
    import_communication_measurements, import_linearity_calibration_cases,
    import_linearity_split_results, machine_from_environment,
)
from .model_catalog import list_models as list_catalog_models, load_model as load_catalog_model
from .a3_resources import ResourceError, simulate_microbatch_pipeline

def _load_json(path: str | Path) -> Any:
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"input_unreadable:{source}:{type(exc).__name__}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input_json_invalid:{source}") from exc


def _write(payload: Mapping[str, Any], output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _mhz_range(text: str | None) -> list[float] | None:
    if not text:
        return None
    parts = [part for part in text.replace(",", ":").split(":") if part.strip()]
    try:
        values = sorted(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"sustained_mhz_invalid:{text}") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"sustained_mhz_invalid:{text}")
    if len(values) == 1:
        values = [values[0], values[0]]
    return values[:2]


def _find_platform_ini(soc: str) -> Path:
    direct = Path(soc).expanduser()
    if direct.is_file():
        return direct
    env_dir = os.environ.get("MINDIE_SIM_PLATFORM_CONFIG_DIR")
    candidates: list[str] = []
    if env_dir:
        candidates.append(str(Path(env_dir).expanduser() / f"{soc}.ini"))
    for pattern in (
        f"/usr/local/Ascend/*/*/data/platform_config/{soc}.ini",
        f"/usr/local/Ascend/*/data/platform_config/{soc}.ini",
        f"/usr/local/Ascend/*/*/*/data/platform_config/{soc}.ini",
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    for candidate in candidates:
        if Path(candidate).is_file():
            return Path(candidate)
    raise HardwareError(f"platform_ini_not_found_for_soc:{soc}")


def _profile_merge(dag: Mapping[str, Any], report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(dag)
    nodes = [dict(node) for node in dag.get("nodes", ())]
    by_id = {str(node.get("id")): node for node in nodes}
    applied: list[str] = []
    unmatched: list[str] = []
    mapping = report.get("node_durations_us")
    if isinstance(mapping, Mapping):
        for node_id, duration in mapping.items():
            node = by_id.get(str(node_id))
            if node is None:
                unmatched.append(str(node_id))
                continue
            if node.get("duration_us") is None and isinstance(duration, (int, float)):
                node["duration_us"] = float(duration)
                applied.append(str(node_id))
    for kernel in report.get("kernels", ()) or ():
        if not isinstance(kernel, Mapping):
            continue
        node_id = kernel.get("node", kernel.get("id", kernel.get("name")))
        node = by_id.get(str(node_id))
        if node is None:
            unmatched.append(str(node_id))
            continue
        duration = kernel.get("duration_us")
        if duration is None and kernel.get("duration_ms") is not None:
            duration = float(kernel["duration_ms"]) * 1000.0
        if node.get("duration_us") is None and isinstance(duration, (int, float)):
            node["duration_us"] = float(duration)
            applied.append(str(node_id))
    merged["nodes"] = nodes
    return merged, {"applied_nodes": applied, "unmatched_nodes": unmatched}


def _hardware(args: argparse.Namespace) -> dict[str, Any]:
    ini: Path | None = None
    if args.ini:
        ini = Path(args.ini).expanduser()
    elif args.soc and not is_ambiguous_family_token(args.soc):
        try:
            ini = _find_platform_ini(args.soc)
        except HardwareError:
            if args.soc not in BUILTIN_SKU_CATALOG and not args.sku_profile:
                raise
    if ini is None and not args.soc and not args.sku_profile:
        raise HardwareError("platform_ini_or_soc_required")
    config = parse_platform_ini(ini) if ini is not None else None
    snapshot = load_npu_smi_snapshot(args.npu_smi_file) if args.npu_smi_file else None
    topo_snapshot = parse_npu_smi_topo(args.topo_file) if args.topo_file else None
    pcie_snapshot = parse_pcie_link_info(args.pcie_link_info) if args.pcie_link_info else None
    cannsim = parse_cannsim_link_ceilings(args.cannsim_constants) if args.cannsim_constants else default_link_ceilings()
    workload = _load_json(args.workload) if args.workload else None
    buffer_plan = _load_json(args.buffers) if args.buffers else None
    transfer_plan = _load_json(args.transfers) if args.transfers else None
    simulator_ub = None if args.simulator_ub_kib is None or args.simulator_ub_kib <= 0 else float(args.simulator_ub_kib)
    thermal_overrides = {
        key: value
        for key, value in {
            "fp8_flops": args.fp8_flops,
            "bf16_flops": args.bf16_flops,
            "target_us": args.target_us,
            "wall_cycles": args.wall_cycles,
            "counted_cycles": args.counted_cycles,
            "vector_cycles": args.vector_cycles,
            "fix_cycles": args.fix_cycles,
            "cube_reference_cycles": args.cube_reference_cycles,
            "fp8_ops_per_cycle": args.fp8_ops_per_cycle,
            "bf16_ops_per_cycle": args.bf16_ops_per_cycle,
            "measurement_mhz": args.measurement_mhz,
        }.items()
        if value is not None
    }
    if args.golden_a5_flashmla:
        thermal_overrides = {**A5_FLASHMLA_C8_GOLDEN_WORKLOAD, **thermal_overrides}
    payload = hardware_report(
        config,
        soc=args.soc,
        sku_profile=args.sku_profile,
        snapshot=snapshot,
        topo_snapshot=topo_snapshot,
        pcie_snapshot=pcie_snapshot,
        verified_runtime_hbm_mib=args.verified_runtime_hbm_mib,
        simulator_ub_kib=simulator_ub,
        allocatable_ub_kib=args.allocatable_ub_kib,
        # No measured sustained band is the default: 1250-1300 MHz was the A5
        # cannsim reference ladder, never a soak of any device, and injecting it
        # as a CLI default made every report claim a band nobody measured.
        sustained_mhz_range=_mhz_range(args.sustained_mhz),
        short_burst_mhz_range=_mhz_range(args.short_burst_mhz) or (1640.0, 1650.0),
        buffer_plan=buffer_plan,
        workload=workload,
        transfer_plan=transfer_plan,
        cannsim_links=cannsim,
        **thermal_overrides,
    )
    payload["command"] = "hardware"
    return payload


def _pipeline(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_json(args.graph)
    durations = _load_json(args.durations) if args.durations else None
    if args.compare_variant:
        payload = compare_pipeline_variants(
            graph,
            _load_json(args.compare_variant),
            durations,
            target_us=args.target_us,
            total_tiles=args.total_tiles,
            reference_mhz=args.reference_mhz,
        )
    else:
        payload = solve_periodic_pipeline(
            graph,
            durations,
            target_us=args.target_us,
            total_tiles=args.total_tiles,
            reference_mhz=args.reference_mhz,
            run_self_check=True,
        )
    payload["command"] = "pipeline"
    return payload


def _subgraph(args: argparse.Namespace) -> dict[str, Any]:
    dag = _load_json(args.dag)
    merge_status = None
    if args.profile_report:
        dag, merge_status = _profile_merge(dag, _load_json(args.profile_report))
    candidate = _load_json(args.candidate_fusion) if args.candidate_fusion else None
    payload = analyze_subgraph_fusion(dag, candidate)
    payload["command"] = "subgraph"
    if merge_status is not None:
        payload["profile_merge"] = merge_status
    return payload


def _net(args: argparse.Namespace) -> dict[str, Any]:
    model_spec = _load_json(args.model_spec)
    if not isinstance(model_spec, dict):
        raise NetError("model_spec_invalid")
    if args.parallel:
        parallel = _load_json(args.parallel)
        if not isinstance(parallel, Mapping):
            raise NetError("parallel_invalid")
        model_spec = {**model_spec, "parallel": dict(parallel)}
    payload = net_report(
        model_spec,
        model_spec.get("parallel"),
        measured_device_ms=args.measured_ms,
        provider_source=args.provider_source,
        device_name=args.device,
        expected_commit=args.expected_commit,
    )
    payload["command"] = "net"
    return payload

def _a3_case(path: str) -> CaseSpec:
    payload = _load_json(path)
    return CaseSpec.from_mapping(payload)


def _calibration_from_args(args: argparse.Namespace) -> Any:
    if getattr(args, "calibration_db", None):
        return SQLiteCalibrationStore(args.calibration_db).as_calibration_store()
    if getattr(args, "calibration", None):
        return CalibrationStore.from_mapping(_load_json(args.calibration))
    return None


def _physical_calibration_from_args(args: argparse.Namespace) -> Any:
    """Load a fitted physical-calibration artifact, refusing a malformed one.

    The artifact is one machine's calibration: this only decodes it, and the
    hardware-key agreement with the case machine is enforced by
    ``a3_framework.predict``.
    """
    path = getattr(args, "physical_calibration", None)
    if not path:
        return None
    from .a3_physical_calibration import PhysicalCalibration, PhysicalCalibrationError

    try:
        return PhysicalCalibration.load(path)
    except (OSError, ValueError, PhysicalCalibrationError) as exc:
        raise A3InputError(
            f"physical_calibration_unreadable:{path}:{type(exc).__name__}:{exc}"
        ) from exc


def _timing_scope_from_args(args: argparse.Namespace) -> str | None:
    scope = str(getattr(args, "timing_scope", None) or "").strip()
    if not scope:
        return None
    from .a3_physical_calibration import ACCEPTED_TIMING_SCOPES

    if scope not in ACCEPTED_TIMING_SCOPES:
        raise A3InputError(
            f"timing_scope_unknown:{scope}:expected_one_of:{','.join(ACCEPTED_TIMING_SCOPES)}"
        )
    return scope


def _attach_curve_artifact(case: Any, path: str | None) -> Any:
    if not path:
        return case
    from .a3_operator_curve_forecast import load_curve_artifact
    return _dataclass_replace(
        case, curve_artifact=load_curve_artifact(path),
    )


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    case = _attach_curve_artifact(_a3_case(args.input), args.curve_artifact)
    calibration = _calibration_from_args(args)
    physical_calibration = _physical_calibration_from_args(args)
    timing_scope = _timing_scope_from_args(args)
    if args.calibration_db:
        source = SQLiteCalibrationStore(args.calibration_db)
        comm = CommunicationCalibration.from_store(source)
        cv = CvCalibration.from_store(source)
        # Database contents cannot declare a concurrent workload.  CV
        # contention applies only when the case explicitly supplies cv_query.
        case = _dataclass_replace(
            case, communication_calibration=comm, cv_calibration=cv,
        )
    estimator = None
    if args.operator_config:
        from .operator_estimation import create_operator_estimator
        estimator = create_operator_estimator(_load_json(args.operator_config))
    payload = predict(
        case, calibration=calibration,
        physical_calibration=physical_calibration, timing_scope=timing_scope,
        op_estimator=estimator,
        framework_profile=_load_json(args.framework_config) if args.framework_config else None,
    )
    if args.record_misses:
        if not args.calibration_db:
            raise A3InputError("record_misses_requires_calibration_db")
        store = SQLiteCalibrationStore(args.calibration_db)
        recording = record_prediction_misses(
            store, payload, hardware_key=case.machine.hardware_key,
            run_id=args.run_id or "",
        )
        payload["measurement_requests"] = recording
        if args.requests_output:
            manifest = store.manifest(limit=256)
            Path(args.requests_output).expanduser().write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    payload["command"] = "predict"
    payload["purpose"] = "predictive_operational"
    return payload


def _operator_plan(args: argparse.Namespace) -> dict[str, Any]:
    from .operator_estimation import operator_plan_report
    return operator_plan_report(_load_json(args.input))


def _should(args: argparse.Namespace) -> dict[str, Any]:
    case = _attach_curve_artifact(_a3_case(args.input), args.curve_artifact)
    constraints = dict(_load_json(args.constraints)) if args.constraints else {}
    if getattr(args, "search_budget", None) is not None:
        constraints["search_budget"] = int(args.search_budget)
    payload = should_cost(case, constraints=constraints or None)
    payload["command"] = "should"
    payload["purpose"] = "normative_should_cost"
    return payload


def _search(args: argparse.Namespace) -> dict[str, Any]:
    case = _attach_curve_artifact(_a3_case(args.input), args.curve_artifact)
    calibration = _calibration_from_args(args)
    if calibration is not None:
        case = _dataclass_replace(case, calibration=calibration)
    if args.calibration_db:
        source = SQLiteCalibrationStore(args.calibration_db)
        comm = CommunicationCalibration.from_store(source)
        cv = CvCalibration.from_store(source)
        # Database contents cannot declare a concurrent workload.  CV
        # contention applies only when the case explicitly supplies cv_query.
        case = _dataclass_replace(
            case, communication_calibration=comm, cv_calibration=cv,
        )
    raw_case = _load_json(args.input)
    spec = _load_json(args.search_spec) if args.search_spec else (
        raw_case.get("search") if isinstance(raw_case, Mapping) else None
    )
    payload = a3_search(case, spec)
    payload["command"] = "search"
    payload["purpose"] = "configuration_search_full_predictive"
    return payload


def _calibration(args: argparse.Namespace) -> dict[str, Any]:
    store = SQLiteCalibrationStore(args.db)
    action = args.calibration_command
    if action == "inventory":
        payload = store.inventory()
    elif action == "coverage":
        queries = _load_json(args.queries) if getattr(args, "queries", None) else None
        if isinstance(queries, Mapping):
            queries = queries.get("queries", [])
        payload = store.coverage(queries=queries)
    elif action == "requests":
        payload = {"requests": store.requests(status=args.status, limit=args.limit)}
    elif action == "manifest":
        manifest = store.manifest(limit=args.limit)
        if args.output:
            Path(args.output).expanduser().write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        payload = manifest
    elif action == "ingest":
        results = _load_json(args.results)
        payload = store.ingest_results(results)
    elif action == "run":
        runner = CpuMatmulRunner(iterations=args.iterations) if args.runner == "cpu" else TorchNpuRunner(iterations=args.iterations)
        payload = run_pending(store, runner, limit=args.limit, run_id=args.run_id or "")
    elif action == "import-a3":
        document = _load_json(args.input)
        cases = document.get("cases") if isinstance(document, Mapping) else None
        exported_calibration_cases = bool(
            isinstance(cases, Sequence) and cases
            and all(
                isinstance(case, Mapping)
                and "source_family" in case
                and "raw_device_ms_per_op" in case
                and "samples" not in case
                for case in cases
            )
            and "started_unix_s" not in document
        )
        linearity = bool(exported_calibration_cases) or bool(
            isinstance(cases, Sequence) and cases
            and isinstance(cases[0], Mapping)
            and "shape_mkn" in cases[0]
            and "raw_device_ms_per_op" in cases[0]
            and "source_family" in cases[0]
            and "started_unix_s" not in document
        )
        split_results = bool(
            isinstance(document, Mapping)
            and isinstance(cases, Sequence) and cases
            and all(
                isinstance(case, Mapping)
                and isinstance(case.get("request"), Mapping)
                and isinstance(case.get("production_result"), Mapping)
                for case in cases
            )
        )
        if split_results:
            payload = import_linearity_split_results(
                store, args.input, include_heldout=bool(args.include_heldout),
            )
        elif linearity:
            payload = import_linearity_calibration_cases(
                store, args.input, include_heldout=bool(args.include_heldout),
            )
        else:
            payload = import_a3_measurements(
                store, args.input, include_heldout=bool(args.include_heldout),
            )
    elif action == "cv-summary":
        calibration = CvCalibration.from_store(store)
        payload = {
            "conditions": [condition.__dict__ for condition in calibration.conditions],
            "note": "measured paired-stream lane work; call CvCalibration.predict for a query",
        }
    elif action == "import-comm":
        payload = import_communication_measurements(
            store, args.input, include_heldout=bool(args.include_heldout),
        )
    elif action == "comm-summary":
        calibration = CommunicationCalibration.from_store(store)
        payload = {
            "conditions": [condition.__dict__ for condition in calibration.conditions],
            "note": "affine isolated curves + shared-engine joint span; no SHA gate",
        }
    elif action == "comm-predict":
        calibration = CommunicationCalibration.from_store(store)
        payload = calibration.predict(
            placement=args.placement, size_bytes=args.size_bytes,
            requested_delay_ms=args.delay_ms, repetitions=args.repetitions,
            mode=args.mode, execution_backend=args.backend,
            observed_execution=args.observed_backend,
            collective=args.collective, reduction=args.reduction,
            group_size=args.group_size,
            route=getattr(args, "route", ""),
            timing_completion=getattr(args, "timing_completion", ""),
            requested_aiv_cores=getattr(args, "requested_aiv_cores", None),
            observed_aiv_cores=getattr(args, "observed_aiv_cores", None),
            algorithm_regime=getattr(args, "algorithm_regime", ""),
            topology_class=getattr(args, "topology_class", ""),
        )
    elif action == "complete":
        payload = store.complete_request(
            args.request_key, duration_ms=args.duration_ms,
            analytic_ms=args.analytic_ms, source=args.source,
            source_class=args.source_class, run_id=args.run_id or "",
            revision=args.revision or "", split=args.split.value,
        )
    elif action == "fail":
        payload = store.fail_request(args.request_key, reason=args.reason, run_id=args.run_id or "")
    elif action == "retry":
        payload = store.requeue_request(
            args.request_key, run_id=args.run_id or "", reason=args.reason,
        )
    elif action == "retry-failed":
        repaired = []
        for request in store.requests(status="failed", limit=args.limit):
            repaired.append(store.requeue_request(
                request["request_key"], run_id=args.run_id or "",
                reason=args.reason,
            ))
        payload = {"status": "requeued", "requests": repaired}
    else:
        raise A3InputError(f"calibration_command_unknown:{action}")
    payload["command"] = f"calibration.{action}"
    return payload


def _models(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source": "se-tools actual configs, provenance ae8779fdd5a602bd3a4358bca93b2161d134a47b (not an admission gate)",
        "models": list_catalog_models(),
    }


def _model_info(args: argparse.Namespace) -> dict[str, Any]:
    model = load_catalog_model(args.key)
    return model.to_dict()


def _pipeline_microbatches(args: argparse.Namespace) -> dict[str, Any]:
    stages = [float(item) for item in args.stages.replace(":", ",").split(",") if item.strip()]
    payload = simulate_microbatch_pipeline(stages, args.microbatches, transfer_ms=args.transfer_ms)
    payload["command"] = "pipeline-microbatches"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mindie-sim",
        description="Ascend NPU four-layer simulation and boundary solver.",
    )
    parser.add_argument("--version", action="version", version=f"mindie-sim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("hardware", help="Layer 1 hardware, buffers and thermal bounds")
    hardware.add_argument("--soc", default=None, help="SoC/platform name or direct .ini path")
    hardware.add_argument("--ini", default=None, help="explicit CANN platform .ini")
    hardware.add_argument("--sku-profile", default=None, help="external JSON SKU capability profile")
    hardware.add_argument("--npu-smi-file", default=None, help="npu-smi snapshot file or directory")
    hardware.add_argument("--topo-file", default=None, help="npu-smi info -t topo file")
    hardware.add_argument("--pcie-link-info", default=None, help="lspci LnkCap/LnkSta file or text")
    hardware.add_argument("--cannsim-constants", default=None, help="explicit cannsim link constants")
    hardware.add_argument("--buffers", default=None, help="buffer allocation plan JSON")
    hardware.add_argument("--workload", default=None, help="workload/thermal JSON")
    hardware.add_argument(
        "--golden-a5-flashmla",
        action="store_true",
        help="use the explicit A5 C8 FlashMLA Golden workload fixture",
    )
    hardware.add_argument("--fp8-flops", type=float, default=None, help="explicit FP8 FLOPs")
    hardware.add_argument("--bf16-flops", type=float, default=None, help="explicit BF16 FLOPs")
    hardware.add_argument("--target-us", type=float, default=None, help="explicit target latency in us")
    hardware.add_argument("--wall-cycles", type=float, default=None, help="measured wall cycles W")
    hardware.add_argument("--counted-cycles", type=float, default=None, help="measured counted cycles C")
    hardware.add_argument("--vector-cycles", type=float, default=None, help="measured vector cycles")
    hardware.add_argument("--fix-cycles", type=float, default=None, help="measured fix cycles")
    hardware.add_argument("--cube-reference-cycles", type=float, default=None, help="explicit reference Cube cycles")
    hardware.add_argument("--fp8-ops-per-cycle", type=float, default=None, help="explicit total per-die FP8 Cube ops/cycle")
    hardware.add_argument("--bf16-ops-per-cycle", type=float, default=None, help="explicit total per-die BF16 Cube ops/cycle")
    hardware.add_argument("--measurement-mhz", type=float, default=None, help="explicit clock basis for W/C PMU cycles")
    hardware.add_argument("--transfers", default=None, help="D2D/H2D/D2H/RH2D/RD2H transfer & collective plan JSON")
    hardware.add_argument(
        "--sustained-mhz", default=None,
        help="measured sustained thermal range, e.g. 1650:1650; omitted = unmeasured",
    )
    hardware.add_argument("--short-burst-mhz", default="1640:1650", help="short-burst thermal range, e.g. 1640:1650")
    hardware.add_argument("--simulator-ub-kib", type=float, default=256.0, help="simulator UB reference in KiB; 0 disables")
    hardware.add_argument("--allocatable-ub-kib", type=float, default=None, help="explicit allocatable UB capacity in KiB")
    hardware.add_argument("--verified-runtime-hbm-mib", type=float, default=None, help="explicit verified runtime HBM capacity")
    hardware.add_argument("--output", default=None)
    hardware.set_defaults(handler=_hardware)

    pipeline = subparsers.add_parser("pipeline", help="Layer 2 periodic microstage pipeline solver")
    pipeline.add_argument("--graph", required=True, help="microstage graph JSON")
    pipeline.add_argument("--durations", default=None, help="calibrated stage durations JSON")
    pipeline.add_argument("--compare-variant", default=None, help="alternative graph JSON")
    pipeline.add_argument("--self-check", action="store_true", help="run topological and relaxation self-checks")
    pipeline.add_argument("--target-us", type=float, default=None)
    pipeline.add_argument("--total-tiles", type=int, default=None)
    pipeline.add_argument("--reference-mhz", type=float, default=None)
    pipeline.add_argument("--output", default=None)
    pipeline.set_defaults(handler=_pipeline)

    subgraph = subparsers.add_parser("subgraph", help="Layer 3 subgraph fusion analyzer")
    subgraph.add_argument("--dag", required=True, help="producer-consumer DAG JSON")
    subgraph.add_argument("--candidate-fusion", default=None, help="candidate fusion JSON")
    subgraph.add_argument("--profile-report", default=None, help="optional measured node durations JSON")
    subgraph.add_argument("--output", default=None)
    subgraph.set_defaults(handler=_subgraph)

    net = subparsers.add_parser("net", help="Layer 4 parallel and whole-network decomposition")
    net.add_argument("--model-spec", required=True, help="model/step spec JSON")
    net.add_argument("--parallel", default=None, help="explicit DP/TP/EP/PP/DCP JSON")
    net.add_argument("--provider-source", default=None, help="pinned msmodeling checkout")
    net.add_argument("--device", default=None, help="explicit upstream DeviceProfile name")
    net.add_argument("--expected-commit", default=None)
    net.add_argument("--measured-ms", type=float, default=None)
    net.add_argument("--output", default=None)
    net.set_defaults(handler=_net)
    operator_parser = subparsers.add_parser("operator-plan", help="Finite Vector/Cube operator DAG with explicit profiles")
    operator_parser.add_argument("--input", required=True)
    operator_parser.add_argument("--output", default=None)
    operator_parser.set_defaults(handler=_operator_plan)
    predict_parser = subparsers.add_parser("predict", help="Predictive operational performance for one A3 case")
    predict_parser.add_argument("--operator-config", default=None, help="optional per-op Vector/Cube profiles JSON")
    predict_parser.add_argument("--framework-config", default=None, help="explicit host lifecycle costs and resource scopes JSON")
    predict_parser.add_argument("--input", required=True, help="case JSON: machine/model/workload/runtime")
    predict_parser.add_argument("--calibration", default=None, help="optional calibration observations JSON")
    predict_parser.add_argument("--calibration-db", default=None, help="optional SQLite calibration store")
    predict_parser.add_argument("--curve-artifact", default=None, help="optional public train-only piecewise curve artifact")
    predict_parser.add_argument(
        "--physical-calibration", default=None,
        help="fitted physical-calibration artifact JSON (one hardware key)",
    )
    predict_parser.add_argument(
        "--timing-scope", default=None,
        help=(
            "declared measurement protocol scope for the analytic and fitted "
            "physical paths, e.g. device_event_warm or device_event_warm_batched"
        ),
    )
    predict_parser.add_argument("--record-misses", action="store_true", help="opt in: record uncalibrated ops as pending measurement requests")
    predict_parser.add_argument("--requests-output", default=None, help="optional pending-request manifest output path")
    predict_parser.add_argument("--run-id", default=None, help="external run id for recorded requests")
    predict_parser.add_argument("--output", default=None)
    predict_parser.set_defaults(handler=_predict)

    should = subparsers.add_parser("should", help="Normative should-cost: semantic L and feasible target U")
    should.add_argument("--input", required=True, help="case JSON")
    should.add_argument("--constraints", default=None, help="optional should-cost plan-family constraints JSON")
    should.add_argument("--curve-artifact", default=None, help="optional public train-only piecewise curve artifact")
    should.add_argument("--search-budget", type=int, default=None,
                        help="explicit finite plan simulation budget")
    should.add_argument("--output", default=None)
    should.set_defaults(handler=_should)

    search = subparsers.add_parser("search", help="Rank legal configurations using the full predictive flow")
    search.add_argument("--input", required=True, help="case JSON with optional search domain")
    search.add_argument("--search-spec", default=None, help="optional search domain JSON")
    search.add_argument("--calibration", default=None, help="optional calibration observations JSON")
    search.add_argument("--calibration-db", default=None, help="optional SQLite calibration store")
    search.add_argument("--curve-artifact", default=None, help="optional public train-only piecewise curve artifact")
    search.add_argument("--output", default=None)
    search.set_defaults(handler=_search)

    cal = subparsers.add_parser("calibration", help="SQLite measurement inventory, coverage and re-measurement loop")
    cal.add_argument("--db", required=True, help="explicit SQLite calibration database path")
    cal.add_argument("--output", default=None)
    cal_sub = cal.add_subparsers(dest="calibration_command", required=True)
    inv = cal_sub.add_parser("inventory", help="sample and request counts")
    inv.set_defaults(handler=_calibration)
    cov = cal_sub.add_parser("coverage", help="observed shape coverage")
    cov.add_argument("--queries", default=None, help="optional JSON list of query shapes")
    cov.set_defaults(handler=_calibration)
    req = cal_sub.add_parser("requests", help="list measurement requests")
    req.add_argument("--status", choices=("pending", "running", "completed", "failed"), default=None)
    req.add_argument("--limit", type=int, default=100)
    req.set_defaults(handler=_calibration)
    man = cal_sub.add_parser("manifest", help="generate an executable measurement manifest")
    man.add_argument("--limit", type=int, default=32)
    man.add_argument("--output", default=None)
    man.set_defaults(handler=_calibration)
    ing = cal_sub.add_parser("ingest", help="ingest runner results and complete requests")
    ing.add_argument("--results", required=True)
    ing.set_defaults(handler=_calibration)
    imp = cal_sub.add_parser("import-a3", help="import real A3 training/measurement JSON")
    imp.add_argument("--input", required=True)
    imp.add_argument("--include-heldout", action="store_true")
    imp.set_defaults(handler=_calibration)
    cvs = cal_sub.add_parser("cv-summary", help="summarize measured paired C/V conditions")
    cvs.set_defaults(handler=_calibration)
    impc = cal_sub.add_parser("import-comm", help="import real HCCL/P2P training JSON")
    impc.add_argument("--input", required=True)
    impc.add_argument("--include-heldout", action="store_true")
    impc.set_defaults(handler=_calibration)
    comms = cal_sub.add_parser("comm-summary", help="summarize communication training curves")
    comms.set_defaults(handler=_calibration)
    commp = cal_sub.add_parser("comm-predict", help="predict allreduce+p2p shared-lane span")
    commp.add_argument("--placement", choices=("sio", "hccssw"), required=True)
    commp.add_argument("--size-bytes", type=int, required=True)
    commp.add_argument("--delay-ms", type=float, default=0.0)
    commp.add_argument("--repetitions", type=int, default=1)
    commp.add_argument("--mode", choices=("all_reduce_plus_p2p", "all_reduce_only", "p2p_only"),
                       default="all_reduce_plus_p2p")
    commp.add_argument("--backend", default="unspecified",
                       choices=("unspecified", "aiv", "aicpu", "other"))
    commp.add_argument("--observed-backend", default="unknown",
                       choices=("unknown", "aiv", "aicpu", "other"))
    commp.add_argument("--collective", default="")
    commp.add_argument("--reduction", default="")
    commp.add_argument("--group-size", type=int, default=0)
    commp.add_argument("--route", default="")
    commp.add_argument("--timing-completion", default="",
                       choices=("", "completion", "launch"))
    commp.add_argument("--requested-aiv-cores", type=int, default=None)
    commp.add_argument("--observed-aiv-cores", type=int, default=None)
    commp.add_argument("--algorithm-regime", default="")
    commp.add_argument("--topology-class", default="")
    commp.set_defaults(handler=_calibration)
    run = cal_sub.add_parser("run", help="run pending requests through a local or managed runner")
    run.add_argument("--runner", choices=("cpu", "torch-npu"), default="cpu")
    run.add_argument("--limit", type=int, default=8)
    run.add_argument("--iterations", type=int, default=5)
    run.add_argument("--run-id", default="")
    run.set_defaults(handler=_calibration)
    comp = cal_sub.add_parser("complete", help="manually ingest one measured duration")
    comp.add_argument("--request-key", required=True)
    comp.add_argument("--duration-ms", type=float, required=True)
    comp.add_argument("--analytic-ms", type=float, required=True)
    comp.add_argument("--source", required=True)
    comp.add_argument("--source-class", choices=("measured", "observed", "derived"), default="measured")
    comp.add_argument("--run-id", default="")
    comp.add_argument("--revision", default="")
    comp.add_argument("--split", choices=("train", "heldout"), default="train")
    comp.set_defaults(handler=_calibration)
    fail = cal_sub.add_parser("fail", help="mark a request failed and keep its run id")
    fail.add_argument("--request-key", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--run-id", default="")
    fail.set_defaults(handler=_calibration)
    retry = cal_sub.add_parser("retry", help="explicitly requeue a failed request")
    retry.add_argument("--request-key", required=True)
    retry.add_argument("--reason", default="explicit_retry")
    retry.add_argument("--run-id", default="")
    retry.set_defaults(handler=_calibration)
    retry_all = cal_sub.add_parser("retry-failed", help="explicitly requeue failed requests")
    retry_all.add_argument("--limit", type=int, default=32)
    retry_all.add_argument("--reason", default="explicit_retry")
    retry_all.add_argument("--run-id", default="")
    retry_all.set_defaults(handler=_calibration)
    cal.set_defaults(handler=_calibration)

    models = subparsers.add_parser("models", help="List the nine actual se-tools model configs")
    models.add_argument("--output", default=None)
    models.set_defaults(handler=_models)

    model_info = subparsers.add_parser("model-info", help="Show one catalog model's layer/facts record")
    model_info.add_argument("--key", required=True)
    model_info.add_argument("--output", default=None)
    model_info.set_defaults(handler=_model_info)

    pm = subparsers.add_parser("pipeline-microbatches", help="Finite PP microbatch timeline (not total/PP)")
    pm.add_argument("--stages", required=True, help="comma-separated stage durations in ms, e.g. 3,5")
    pm.add_argument("--microbatches", type=int, required=True)
    pm.add_argument("--transfer-ms", type=float, default=0.0)
    pm.add_argument("--output", default=None)
    pm.set_defaults(handler=_pipeline_microbatches)
    campaign = subparsers.add_parser(
        "campaign", help="Operator shape-linearity campaign plan/run/analyze adapter",
    )
    campaign.add_argument(
        "campaign_command", choices=("plan", "inspect", "run", "analyze"),
    )
    campaign.add_argument(
        "remainder", nargs=argparse.REMAINDER,
        help="arguments passed to the operator campaign module",
    )
    campaign.set_defaults(handler=_campaign)
    identity = subparsers.add_parser(
        "comm-identity", help="Validate profiler-observed communication identity",
    )
    identity.add_argument("--input", required=True)
    identity.add_argument("--output", default=None)
    identity.set_defaults(handler=_comm_identity)
    return parser


def _comm_identity(args: argparse.Namespace) -> dict[str, Any]:
    from .a3_comm_identity import identity_fixture, load_identity

    payload = _load_json(args.input)
    load_identity(payload)
    return identity_fixture(payload) if "results" in payload else payload


def _campaign_manifest_path(argv: Sequence[str]) -> str | None:
    for index, item in enumerate(argv):
        if item == "--manifest":
            if index + 1 >= len(argv):
                raise A3InputError("campaign_manifest_path_missing")
            return argv[index + 1]
        if item.startswith("--manifest="):
            return item.split("=", 1)[1]
    return None


def _campaign_analyze_argv(argv: list[str]) -> list[str]:
    """Choose the operator analysis adapter from the actual document shape.

    An explicit ``--adapter`` is authoritative.  Otherwise a manifest whose
    requests declare ``point`` plus ``origins`` is a native campaign wave and
    uses the campaign adapter; a manifest whose requests declare the worker
    condition/shape business fields uses the worker adapter.  A document that
    satisfies neither is rejected instead of silently analyzed as another kind.
    """
    if "--adapter" in argv:
        return argv
    manifest_path = _campaign_manifest_path(argv)
    if not manifest_path:
        raise A3InputError("campaign_manifest_required_for_analyze")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise A3InputError("campaign_manifest_invalid")
    requests = manifest.get("requests")
    if (not isinstance(requests, Sequence) or isinstance(requests, (str, bytes))
            or not requests):
        raise A3InputError("campaign_manifest_requests_missing")
    if not all(isinstance(request, Mapping) for request in requests):
        raise A3InputError("campaign_manifest_requests_invalid")
    if all(
        isinstance(request.get("point"), Mapping)
        and isinstance(request.get("origins"), Sequence)
        and not isinstance(request.get("origins"), (str, bytes))
        and bool(request.get("origins"))
        for request in requests
    ):
        return ["--adapter", "campaign", *argv]
    if all(
        isinstance(request.get("condition"), Mapping)
        and isinstance(request.get("shape"), Mapping)
        for request in requests
    ):
        return argv
    raise A3InputError("campaign_manifest_unrecognized_business_document")


def _campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Top-level adapter for the operator campaign lane CLIs.

    The lane owns extraction/worker/curve policy.  This wrapper only routes
    ``plan|inspect|analyze`` to the campaign/curve module and ``run`` to the
    device-injected worker.  No device is acquired here.
    """
    command = str(args.campaign_command)
    rest = list(args.remainder or ())
    buffer = io.StringIO()
    if command == "run":
        from .a3_operator_worker import main as module_main
        argv = rest
    elif command in {"plan", "inspect"}:
        from .a3_operator_campaign import main as module_main
        argv = [command, *rest]
    elif command == "analyze":
        from .a3_operator_curves import main as module_main
        argv = _campaign_analyze_argv(rest)
    else:
        raise A3InputError(f"campaign_command_unsupported:{command}")
    with contextlib.redirect_stdout(buffer):
        code = int(module_main(argv))
    text = buffer.getvalue().strip()
    if code != 0:
        raise A3InputError(f"campaign_command_failed:{command}:{code}")
    try:
        payload = json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        raise A3InputError("campaign_output_invalid") from exc
    if not isinstance(payload, dict):
        payload = {"status": "complete", "result": payload}
    return {"status": "complete", "campaign_command": command, **payload}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
    except (HardwareError, PipelineError, SubgraphError, NetError, A3InputError, ResourceError, ValueError, OSError, TypeError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        _write({
            "status": "error",
            "command": getattr(args, "command", None),
            "reason": reason,
        }, getattr(args, "output", None))
        return 1
    _write(payload, getattr(args, "output", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
