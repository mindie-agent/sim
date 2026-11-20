"""Native AI_CPU child local service fitting, evidence inspection, and counterfactuals.

[Role] Validate native communication campaign artifacts, fit T06 normalized
  AI_CPU child local services, expose per-call stage/overlap evidence, and run
  domain-checked prediction and supply interventions.
[Invariants] The fitted response is the directly observed ``HcclLaunchAicpuKernel``
  AI_CPU child phase, never the COMMUNICATION envelope or a per-hop link price.
  Environment host, image, manifest SHA-256, CANN, and topology must match
  before any fit or supply intervention.
[Timeouts & Locks] Bounded read-only JSON and text artifact reads; LRU caches
  are keyed by path, mtime, size, and deployment selectors. No locks or devices.
[Side-Effect Policy] Read-only over local artifact files; never mutates captures
  or launches device work.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Mapping

from ..resource_services import normalize_resource_service
from ..service_fitting import fit_resource_service
from .calibration import _measurement_boundary, fit_a5_service_curves
from .golden import A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_NATIVE_GROUP2
from .ingest import ingest_a5_campaign_data as _ingest_a5_campaign_data
from .local_domain import (
    _NATIVE_LOCAL_COLLECTIVES,
    _NATIVE_LOCAL_COLLECTIVE_SET,
    _NATIVE_LOCAL_DIRECTION,
    _local_service_evidence_gaps,
    _native_local_domain,
    a5_local_service_applicability_key,
    a5_local_service_domain_key,
    a5_local_service_lineage_key,
    require_a5_local_service_applicability,
    require_a5_local_service_domain,
)
from .normalization import (
    NATIVE_WINDOW_ADMITTED_REPEATS,
    NATIVE_WINDOW_WARMUP_CALLS,
    _collective,
    _dtype,
)

A5_COMM_LOCAL_SERVICE_SCHEMA = "a5-comm-local-service/2"
A5_NATIVE_CHILD_EVIDENCE_SCHEMA = "a5-native-child-stage-evidence/1"
NATIVE_A5_SERVICE_SOURCE = A5_NATIVE_GROUP2["source"]

DEFAULT_NATIVE_GROUPS = (2, 4, 8)
_COLLECT_NATIVE_CACHE_SIZE = 4
_FIT_NATIVE_CACHE_SIZE = 8
_WEAK_IDENTIFICATION_REL_ERROR = 0.20

__all__ = [
    "A5_COMM_LOCAL_SERVICE_SCHEMA",
    "A5_NATIVE_CHILD_EVIDENCE_SCHEMA",
    "DEFAULT_NATIVE_GROUPS",
    "NATIVE_A5_SERVICE_SOURCE",
    "a5_local_service_applicability_key",
    "a5_local_service_domain_key",
    "a5_local_service_lineage_key",
    "a5_local_service_prediction",
    "a5_local_service_supply_change",
    "collect_native_service",
    "fit_a5_native_local_service",
    "native_child_stage_evidence",
    "require_a5_local_service_applicability",
    "require_a5_local_service_domain",
]


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_source_identity(path, *, stack, cann, execution_core, placement,
                            deployment_image_id):
    """Validate the native wave's actual source/environment and return it.

    The identity is the environment that produced the observations: manifest
    revision, software packages, host/image and intrahost UB topology.  A
    caller may not respecify any of these and then reuse the fitted supply.
    """
    path = Path(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    source_id = str(summary.get("manifest_sha256") or "")
    manifest = path.parents[2] / "manifest" / "native-manifest.json"
    if not source_id or not manifest.is_file() or _file_sha256(manifest) != source_id:
        raise ValueError("native_service_manifest_identity_mismatch")
    versions = path.parents[1] / "native-versions.json"
    if not versions.is_file():
        raise ValueError("native_service_versions_missing")
    version = json.loads(versions.read_text(encoding="utf-8"))
    packages = version.get("python_packages") or {}
    if (str(cann) not in str(packages.get("cann") or "")
            and str(cann) not in str(version.get("cann_version_info") or "")):
        raise ValueError("native_service_cann_mismatch")
    if "intrahost" not in str(version.get("topology_interpretation") or "").lower():
        raise ValueError("native_service_topology_not_intrahost")
    environment = path.parents[1] / "environment.txt"
    if not environment.is_file():
        raise ValueError("native_service_environment_missing")
    env_text = environment.read_text(encoding="utf-8", errors="replace")
    host = next((line.split()[1] for line in env_text.splitlines()
                 if line.startswith("Linux ")), "")
    image_line = next((line for line in env_text.splitlines()
                       if line.startswith("image_id=")), "")
    env_image = ""
    for token in image_line.split():
        if token.startswith("image_id="):
            env_image = token.split("=", 1)[1]
    if not env_image or str(deployment_image_id) != env_image:
        raise ValueError("native_service_environment_image_mismatch")
    if not host or not host.lower().startswith(str(stack).lower()):
        raise ValueError("native_service_environment_stack_mismatch")
    if str(execution_core).lower() != "aicpu":
        raise ValueError("native_service_requires_aicpu_active_child")
    if str(placement) != "intra_node_ub":
        raise ValueError("native_service_requires_intra_node_ub")
    topology = path.parents[1] / "topology.txt"

    # Preserve missing vs declared-unknown vs a concrete value.  The native ABI
    # does not expose the deployed HCCL algorithm/protocol; do not turn that
    # into permission to migrate to any requested value.
    def _declared(value):
        text = str(value or "").strip()
        if not text:
            return None
        return "unknown" if text.lower().startswith("unknown") else text

    identity = {
        "path": str(path), "summary_sha256": _file_sha256(path),
        "manifest_path": str(manifest), "manifest_sha256": source_id,
        "versions_path": str(versions), "versions_sha256": _file_sha256(versions),
        "environment_path": str(environment),
        "environment_sha256": _file_sha256(environment),
        "environment_image_id": env_image, "host": host,
        "topology_path": str(topology),
        "topology_sha256": _file_sha256(topology) if topology.is_file() else None,
        "topology_interpretation": str(version.get("topology_interpretation") or ""),
        "scope": str(summary.get("scope") or ""),
        "python_packages": {key: packages.get(key)
                            for key in ("torch", "torch_npu", "cann")
                            if packages.get(key)},
        "hccl_algorithm": _declared(version.get("hccl_algorithm")),
        "hccl_protocol": _declared(version.get("hccl_protocol")),
        "hccl_communicator_name": version.get("hccl_communicator_name"),
    }
    return summary, identity


def _native_service_conditions(summary, path, groups):
    """Build one condition per native target call, never summing ranks."""
    wanted = {int(group) for group in groups}
    conditions, heldout = [], set()
    for row in summary.get("conditions") or ():
        if int(row.get("group_size") or 0) not in wanted:
            continue
        if str(row.get("collective")) not in _NATIVE_LOCAL_COLLECTIVE_SET:
            continue
        if str(row.get("dtype") or "bf16").lower() not in {"bf16", "bfloat16"}:
            continue
        if str(row.get("status") or "ok") != "ok":
            raise ValueError("native_service_condition_status_not_ok")
        active, call_samples = [], []
        payload = int(row.get("input_bytes") or 0)
        rank_keys = sorted(int(key) for key in (row.get("per_rank") or {}))
        group = int(row.get("group_size") or 0)
        if rank_keys != [0, group - 1] or int(row.get("profile_rank_count") or 0) != 2:
            raise ValueError("native_service_rank_placement_not_declared")
        native_id = str(row.get("native_id")
                        or f"{row['collective']}:g{group}:{payload}")
        for rank_key, rank in sorted((int(key), value) for key, value in
                                     (row.get("per_rank") or {}).items()):
            if (rank.get("canary_before_ok") is False
                    or rank.get("canary_after_ok") is False
                    or rank.get("mapping_ok") is False):
                raise ValueError("native_service_rank_evidence_failed")
            if (rank.get("actual_bytes_match") is False
                    or int(rank.get("actual_input_bytes") or 0) != payload):
                raise ValueError("native_service_actual_input_mismatch")
            calls = rank.get("calls") or ()
            rank_values = []
            for call_index, call in enumerate(calls):
                ai_rows = call.get("ai_cpu_rows") or ()
                if not ai_rows or any(
                    ai_row.get("kind") != "ai_cpu"
                    or "HcclLaunchAicpuKernel" not in str(ai_row.get("name") or "")
                    for ai_row in ai_rows
                ):
                    raise ValueError("native_service_active_aicpu_evidence_missing")
                value = call.get("ai_cpu_union_us")
                if value is None:
                    continue
                value = float(value)
                rep_index = int(call.get("rep_index", call_index))
                rank_values.append(value)
                call_samples.append({
                    "native_id": native_id, "rank": rank_key,
                    "repeat_index": rep_index,
                    "marker": call.get("marker"),
                    "range_marker": call.get("range_marker"),
                    "status": call.get("status"),
                    "duration_us": value,
                    "communication_union_us": call.get("communication_union_us"),
                    "overlap_us": call.get("overlap_us"),
                    "combined_span_union_us": call.get("combined_span_union_us"),
                    "range_duration_us": call.get("range_duration_us"),
                    "arrival_ns": call.get("arrival_ns"),
                    "enqueue_start_ns": call.get("enqueue_start_ns"),
                    "enqueue_end_ns": call.get("enqueue_end_ns"),
                    "complete_ns": call.get("complete_ns"),
                    "time_base_us": call.get("time_base_us"),
                    "span_start_rel_us": call.get("span_start_rel_us"),
                    "span_end_rel_us": call.get("span_end_rel_us"),
                    "ai_cpu_row_count": call.get("ai_cpu_row_count"),
                    "communication_row_count": call.get("communication_row_count"),
                })
            samples = rank.get("ai_cpu_union_us") or ()
            samples = [float(value) for value in samples if value is not None]
            if samples:
                active.append(median(samples))
            elif rank_values:
                active.append(median(rank_values))
        if not active or payload <= 0:
            continue
        split = str(row.get("split") or "train")
        if split == "heldout":
            heldout.add(payload)
        conditions.append({
            "native_id": native_id,
            "collective": row["collective"],
            "world_size": group,
            "tokens": row.get("tokens"),
            "hidden_size": row.get("hidden_size"),
            "payload_bytes": payload,
            "output_bytes": row.get("output_bytes"),
            "per_op_ms": median(active) / 1000.0,
            "dtype": "bf16", "placement_scope": "intra_node_ub",
            "split": split, "single": False,
            "run_id": native_id,
            "observed_expansion_mode": "aicpu",
            "requested_expansion_mode": "unspecified",
            "source": str(path),
            "measurement": {
                **_measurement_boundary(row["collective"], "device_kernel", True),
                "explicit": True,
            },
            "transport_window": {
                "kind": "native_single_target_call", "phase": "aicpu_child",
                "warmup_calls": NATIVE_WINDOW_WARMUP_CALLS,
                "admitted_repeats": NATIVE_WINDOW_ADMITTED_REPEATS,
                "profiled_ranks": [0, group - 1],
                "clock_domain": "device", "concurrent_collectives": False,
            },
            "direction_observed": False,
            "algorithm_identified": False,
            "call_samples": call_samples,
            "aggregation": "median_of_rank_medians",
            "sample_source": "native_ai_cpu_child",
        })
    if not conditions:
        raise ValueError("native_service_no_compatible_conditions")
    return conditions, heldout


@lru_cache(maxsize=_COLLECT_NATIVE_CACHE_SIZE)
def _collect_native_service_cached(source: str, groups: tuple, stack: str,
                                   cann: str, execution_core: str,
                                   placement: str, mtime_ns: int, size: int,
                                   deployment_image_id: str):
    """Fit the ServiceCurve bundle from a validated native wave.

    The cached producer reads the source artifacts only on a cache miss.  It
    validates actual environment host/image, manifest SHA, CANN/topology,
    group/dtype/canary/mapping evidence and the requested deployment image.
    The fitted response is the active AICPU child, never the COMMUNICATION
    envelope or a sum of parent/alias.
    """
    del mtime_ns, size
    path = Path(source)
    summary, _identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, heldout = _native_service_conditions(summary, path, groups)
    campaign = {
        "label": f"native-service:{str(summary.get('manifest_sha256'))[:12]}",
        "kind": "a5_collective_calibration", "status": "completed",
        "canary": "all_ranks_matched", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "stack": str(stack), "cann": str(cann), "dtype": "bf16",
        "placement": placement, "observed_expansion_mode": execution_core,
        "requested_expansion_mode": "unspecified", "source": str(path),
        "conditions": conditions,
    }
    ingested = _ingest_a5_campaign_data(campaign)
    return fit_a5_service_curves(
        ingested, holdout_payloads=tuple(sorted(heldout)))


def collect_native_service(source: str, *, stack: str, cann: str,
                           execution_core: str, placement: str,
                           deployment_image_id: str,
                           groups=DEFAULT_NATIVE_GROUPS):
    """Public cached collector; only source revision/deployment and axes key it."""
    path = Path(source)
    stat = path.stat()
    return _collect_native_service_cached(
        str(path), tuple(int(group) for group in groups), str(stack),
        str(cann), str(execution_core), str(placement),
        int(stat.st_mtime_ns), int(stat.st_size), str(deployment_image_id))


@lru_cache(maxsize=_FIT_NATIVE_CACHE_SIZE)
def _fit_native_local_service_cached(source: str, collective: str, group_size: int,
                                     stack: str, cann: str, execution_core: str,
                                     placement: str, deployment_image_id: str,
                                     dtype: str, mtime_ns: int, size: int):
    del mtime_ns, size
    path = Path(source)
    summary, source_identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, heldout = _native_service_conditions(
        summary, path, (int(group_size),))
    rows = [row for row in conditions if row["collective"] == collective]
    if not rows:
        raise ValueError(f"native_local_service_collective_missing:{collective}")
    direction = _NATIVE_LOCAL_DIRECTION[collective]
    domain, domain_key, applicability_key, lineage_key = _native_local_domain(
        source_identity, group_size=int(group_size),
        stack=stack, cann=cann, direction=direction, dtype=dtype)
    gaps = _local_service_evidence_gaps(collective, int(group_size))
    samples, source_records = [], []
    for row in rows:
        calls = row.get("call_samples") or ()
        if not calls:
            samples.append({
                "work": int(row["payload_bytes"]),
                "duration_us": float(row["per_op_ms"]) * 1000.0,
                "run_id": str(row["run_id"]), "repeat_index": 0,
                "split": str(row["split"]),
                "source": str(row.get("source") or path),
            })
            source_records.append({
                "native_id": row["native_id"], "rank": None,
                "repeat_index": None, "marker": None,
                "status": "aggregate_only", "used_in_fit": True,
                "work": int(row["payload_bytes"]), "split": str(row["split"]),
                "source": str(row.get("source") or path),
            })
            continue
        for call in calls:
            ok = call.get("status") in (None, "ok")
            source_records.append({
                "native_id": row["native_id"], "rank": call["rank"],
                "repeat_index": call["repeat_index"],
                "marker": call.get("marker"),
                "range_marker": call.get("range_marker"),
                "status": call.get("status"), "used_in_fit": ok,
                "work": int(row["payload_bytes"]), "split": str(row["split"]),
                "source": str(row.get("source") or path),
            })
            if not ok:
                continue
            samples.append({
                "work": int(row["payload_bytes"]),
                "duration_us": float(call["duration_us"]),
                "run_id": f"{row['native_id']}:rank{call['rank']}",
                "repeat_index": int(call["repeat_index"]),
                "split": str(row["split"]),
                "source": (f"{path}:{row['native_id']}:rank{call['rank']}:"
                           f"rep{call['repeat_index']}:{call.get('marker')}"),
            })
    if not samples:
        raise ValueError("native_local_service_no_training_or_heldout_samples")
    packet = {
        "id": (f"a5-native-aicpu-child:{collective}:W{int(group_size)}:"
               f"{stack}:{cann}:{placement}"),
        "scope": "independent_resource",
        "resources": ["aicpu_child"],
        "phase": "collective_expand",
        "work_unit": "bytes",
        "boundary": "resource_phase",
        "active_units": int(group_size),
        "rate_scope": "per_owner",
        "conditions": dict(applicability_key),
        "identification": {
            "status": "declared_effective_service",
            "resource_identified": False,
            "concurrency_scope": "isolated_invocation",
        },
        "provenance": {
            "source": str(path), "sources": [str(path)],
            "window": "single_target_call",
            "observation_kind": "native_ai_cpu_child",
            "boundary_kind": "resource_phase", "scope": "independent",
            "service_semantics": "aicpu_child_collective_phase",
            "measurement_boundary": domain["measurement_boundary"],
            "direction": direction,
            "shared_domain": domain["shared_domain"]["id"],
            "transport_window": domain["transport_window"],
            "rate_interpretation": (
                "effective_aicpu_child_phase_rate_not_an_isolated_aicpu_"
                "or_link_peak"),
            "source_identity": source_identity,
            "source_records": source_records,
            "aggregation": "median_of_run_medians_per_rank",
            "sample_source": "native_ai_cpu_child",
            "evidence_gaps": gaps,
        },
        "samples": samples,
    }
    fitted = fit_resource_service(packet)
    heldout_error = max((abs(float(row["relative_error"]))
                         for row in fitted["heldout"]), default=None)
    if heldout_error is not None and heldout_error > _WEAK_IDENTIFICATION_REL_ERROR:
        gaps = gaps + [
            f"phase_rate_weakly_identified:heldout_max_relative_error="
            f"{heldout_error:.3f}"]
    domain = {**domain, "evidence_gaps": list(gaps)}
    all_calls = [call for row in rows for call in (row.get("call_samples") or ())]
    child_overlapped = bool(all_calls) and all(
        call.get("overlap_us") is not None
        and call.get("combined_span_union_us") is not None
        and call.get("communication_union_us") is not None
        and abs(float(call["duration_us"]) - float(call["overlap_us"])) < 1e-9
        and abs(float(call["combined_span_union_us"])
                - float(call["communication_union_us"])) < 1e-9
        for call in all_calls)
    return {
        "schema": A5_COMM_LOCAL_SERVICE_SCHEMA,
        "diagnostic_only": True,
        "production_eligible": False,
        "overlap_with_communication": child_overlapped,
        "child_is_complete_collective": False,
        "service": fitted["service"],
        "normalized_schema": fitted["service"].get("schema"),
        "formula": fitted["formula"],
        "model_id": fitted["model_id"],
        "fit": fitted["fit"],
        "heldout": fitted["heldout"],
        "heldout_max_abs_relative_error": heldout_error,
        "domain": domain,
        "domain_key": domain_key,
        "applicability_key": applicability_key,
        "lineage_key": lineage_key,
        "service_semantics": "aicpu_child_collective_phase",
        "evidence_gaps": gaps,
        "source": source_identity,
        "source_records": source_records,
        "holdout_payloads": sorted(heldout),
    }


def fit_a5_native_local_service(source: str = NATIVE_A5_SERVICE_SOURCE, *,
                                collective: str, stack: str, cann: str,
                                deployment_image_id: str, group_size: int = 2,
                                execution_core: str = "aicpu",
                                placement: str = "intra_node_ub",
                                dtype: str = "bf16"):
    """Fit one T06 normalized local service from native AI_CPU evidence.

    The service is the directly observed ``HcclLaunchAicpuKernel`` AI_CPU child
    phase of one target call.  It is a phase service, not an isolated AICPU
    hardware peak and not the COMMUNICATION envelope; the link transit and
    per-hop price remain evidence gaps.  The work axis is the captured per-rank
    input payload, so no hop factor is invented here.
    """
    name = _collective(collective)
    if name not in _NATIVE_LOCAL_COLLECTIVES:
        raise ValueError(f"native_local_service_collective_unsupported:{name}")
    if _dtype(dtype) != "bf16":
        raise ValueError("native_local_service_requires_bf16")
    if int(group_size) < 2:
        raise ValueError("native_local_service_requires_group_size_at_least_two")
    path = Path(source)
    stat = path.stat()
    return _fit_native_local_service_cached(
        str(path), name, int(group_size), str(stack), str(cann),
        str(execution_core), str(placement), str(deployment_image_id),
        "bf16", int(stat.st_mtime_ns), int(stat.st_size))


def native_child_stage_evidence(source: str = NATIVE_A5_SERVICE_SOURCE, *,
                                collective: str, stack: str, cann: str,
                                deployment_image_id: str, group_size: int = 2,
                                execution_core: str = "aicpu",
                                placement: str = "intra_node_ub", split=None,
                                used_for_tuning=()):
    """Per-call native child stage/overlap evidence; no rank/rep aggregation.

    This is a diagnostic/handoff source, not a prediction lookup.  Rows used
    to explain or tune the model keep their original split/role and are marked
    ``used_for_tuning``; they are not presented as independent generalization.
    """
    name = _collective(collective)
    if name not in _NATIVE_LOCAL_COLLECTIVES:
        raise ValueError(f"native_local_service_collective_unsupported:{name}")
    path = Path(source)
    summary, source_identity = _native_source_identity(
        path, stack=stack, cann=cann, execution_core=execution_core,
        placement=placement, deployment_image_id=deployment_image_id)
    conditions, _heldout = _native_service_conditions(
        summary, path, (int(group_size),))
    service_record = fit_a5_native_local_service(
        source, collective=name, stack=stack, cann=cann,
        deployment_image_id=deployment_image_id, group_size=group_size,
        execution_core=execution_core, placement=placement)
    service = service_record["service"]
    fixed_us = float(service["fixed_us"])
    rate = float(service["effective_rate"])
    tuning = set(str(item) for item in (used_for_tuning or ()))
    records = []
    for condition in conditions:
        if condition["collective"] != name:
            continue
        if split is not None and condition["split"] != split:
            continue
        work = int(condition["payload_bytes"])
        predicted_us = fixed_us + work * 1e6 / rate
        for call in condition.get("call_samples") or ():
            if call.get("status") not in (None, "ok"):
                continue
            child = float(call["duration_us"])
            communication = call.get("communication_union_us")
            overlap = call.get("overlap_us")
            combined = call.get("combined_span_union_us")
            communication = None if communication is None else float(communication)
            overlap = None if overlap is None else float(overlap)
            combined = None if combined is None else float(combined)
            used = condition["native_id"] in tuning
            records.append({
                "native_id": condition["native_id"],
                "collective": condition["collective"],
                "tokens": condition.get("tokens"),
                "hidden_size": condition.get("hidden_size"),
                "work_bytes": work,
                "original_split": condition["split"],
                "evidence_role": (condition["split"]
                                  + ("_used_for_tuning" if used else "")),
                "used_for_tuning": used,
                "rank": call["rank"],
                "repeat_index": call["repeat_index"],
                "marker": call.get("marker"),
                "range_marker": call.get("range_marker"),
                "child_us": child,
                "communication_union_us": communication,
                "overlap_us": overlap,
                "combined_span_union_us": combined,
                "range_duration_us": call.get("range_duration_us"),
                "arrival_ns": call.get("arrival_ns"),
                "enqueue_start_ns": call.get("enqueue_start_ns"),
                "enqueue_end_ns": call.get("enqueue_end_ns"),
                "complete_ns": call.get("complete_ns"),
                "child_fraction_of_communication": (
                    None if not communication else child / communication),
                "overlap_fraction_of_child": (
                    None if not child or overlap is None else overlap / child),
                "child_within_communication": bool(
                    overlap is not None and communication is not None
                    and combined is not None
                    and abs(child - overlap) < 1e-9
                    and abs(combined - communication) < 1e-9),
                "predicted_us": predicted_us,
                "relative_error": predicted_us / child - 1,
                "source": str(path),
            })
    if not records:
        raise ValueError("native_child_evidence_no_records")
    split_counts = {}
    for record in records:
        split_counts[record["original_split"]] = split_counts.get(
            record["original_split"], 0) + 1
    return {
        "schema": A5_NATIVE_CHILD_EVIDENCE_SCHEMA,
        "source": source_identity,
        "collective": name,
        "group_size": int(group_size),
        "aggregation": "per_call_no_cross_rank_median",
        "service_model_id": service.get("model_id") or service_record.get("model_id"),
        "service_formula": service_record["formula"],
        "service_diagnostic_only": service_record.get("diagnostic_only"),
        "service_production_eligible": service_record.get("production_eligible"),
        "records": records,
        "summary": {
            "count": len(records),
            "split_counts": split_counts,
            "used_for_tuning_native_ids": sorted(tuning),
            "used_for_tuning_count": sum(
                1 for record in records if record["used_for_tuning"]),
            "all_child_within_communication": all(
                record["child_within_communication"] for record in records),
            "service_fixed_us": fixed_us,
            "service_effective_rate": rate,
            "child_min_us": min(record["child_us"] for record in records),
            "child_max_us": max(record["child_us"] for record in records),
            "fixed_term_exceeds_some_child_calls": any(
                record["child_us"] < fixed_us for record in records),
            "child_by_work": {
                str(work): {
                    "count": sum(1 for record in records
                                 if record["work_bytes"] == work),
                    "min_us": min(
                        (record["child_us"] for record in records
                         if record["work_bytes"] == work), default=None),
                    "max_us": max(
                        (record["child_us"] for record in records
                         if record["work_bytes"] == work), default=None),
                }
                for work in sorted({record["work_bytes"] for record in records})
            },
            "warning": ("used-for-tuning rows keep their original split and "
                        "are not independent generalization"),
        },
    }


def _normalized_local_service(record):
    if (isinstance(record, Mapping)
            and record.get("schema") == A5_COMM_LOCAL_SERVICE_SCHEMA):
        return normalize_resource_service(record["service"])
    return normalize_resource_service(record)


def _require_local_work_unit(service, work_unit):
    if work_unit is not None and str(work_unit) != service.work_unit:
        raise ValueError("a5_local_service_work_unit_mismatch")


def a5_local_service_prediction(record, work, *, domain=None, work_unit=None,
                                work_scope=None):
    """Evaluate the ordinary T06 formula, optionally after a domain/unit check."""
    service = _normalized_local_service(record)
    _require_local_work_unit(service, work_unit)
    if domain is not None:
        require_a5_local_service_applicability(
            record if isinstance(record, Mapping) else {}, domain)
    return service.prediction(work, work_scope=work_scope)


def a5_local_service_supply_change(record, work, *, effective_rate, domain,
                                   work_unit=None):
    """Change the declared effective phase supply only inside its domain.

    ``domain`` is required: a supply update that omits the measured environment
    would be a cross-domain reuse by construction.  Generic T06 callers that
    really want an unscoped rate intervention use
    ``NormalizedResourceService.counterfactual_effective_rate`` directly.
    """
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    require_a5_local_service_domain(record, domain)
    service = _normalized_local_service(record)
    _require_local_work_unit(service, work_unit)
    changed = service.counterfactual_effective_rate(
        work, effective_rate=effective_rate)
    return {
        **changed,
        "domain_checked": True,
        "domain_key": dict(record.get("domain_key") or {}),
        "source_model_id": service.model_id,
        "evidence_gaps": list(record.get("evidence_gaps") or ()),
    }
