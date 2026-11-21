"""Stage-2 L0 primitive & fabric calibration for the New-SKU Onboarding Pipeline.

[Role] Execute Stage 2 (``calibrate-l0-and-fabric``) of the standardized
  New-SKU onboarding pipeline: enforce bounded timeout arithmetic, run optional
  idle-die telemetry prechecks, acquire a verified ``mindie-coordinator``
  ``HostQueue`` lease bound to granted physical devices on live runs (passing
  ``profile_override`` into the L0 collector subprocess and collecting or
  explicitly reporting fabric coverage under the same lease), unwrap canonical
  ``{"tag", "payload"}`` capture envelopes, cross-check CANN provenance against
  capture metadata AND canonical ``data/primitives/manifest.json`` digests
  (marking absent version evidence ``unknown``), delegate sealing exclusively
  to installed ``mindie_sim.primitives.sealer`` preserving per-metric conflict
  reasons and ``SKU x CANN x primitive x mode x metric`` coverage, and emit
  opaque sanitized public receipts.
[Invariants] Zero duplicate primitive sealers; ``calibrated_count == 0`` ALWAYS
  returns ``ok=False`` and ``status="uncalibrated"``; validates BOTH L0 and
  fabric ``device_name`` against ``sku`` before merging; ``TimeoutError``
  propagates before ``OSError``; zero raw IPs/hostnames/usernames/lease IDs in
  public payloads.
[Timeouts & Locks] Enforces ``outer_timeout >= l0_budget + fabric_budget +
  cleanup_grace`` via ``TimeoutBudget``; ``HostQueue`` leases always unwind via
  context managers and surface release failures via ``receipt_warning``.
[Side-Effect Policy] Writes only caller-requested primitive table, raw capture
  sidecar, manifest, and train-manifest paths via atomic ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.backends import ascend, ascend_profiles
from mindie_sim.primitives.loader import default_primitives_dir
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware

from ._gate_runners import (
    acquire_coordinator_host_queue_lease,
    load_canonical_l0_sealer,
    resolve_repo_root,
    run_live_l0_bench_subprocess,
    run_profiler_bench_subprocess,
    validate_host_queue_grant,
)
from ._soc_probe import ARCH_PRESET_35, verify_idle_die_telemetry
from .contracts import (
    COLD_RING_L2_MULTIPLIER,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_FABRIC_BUDGET_S,
    DEFAULT_L0_BUDGET_S,
    MAX_READ_JSON_BYTES,
    SHORT_DIGEST_LEN,
    TRAIN_OBSERVATION_MANIFEST_SCHEMA,
    TimeoutBudget,
    assert_sanitized_payload,
    collect_receipt_warnings,
    sanitize_lease_for_receipt,
    sanitize_public_campaign_id,
    write_atomic_bytes,
    write_atomic_json,
)

_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_FABRIC_DOMAIN_ID = 5
_ABSENT_CANN_TOKENS = frozenset({"", "unmeasured", "unknown"})


def _read_bounded_json_and_sha256(path: Path) -> tuple[Any, str]:
    """Enforce the 50 MiB file limit BEFORE reading bytes, returning ``(data, sha256)``."""
    try:
        size = path.stat().st_size
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_json_stat_failed:{path.name}:{exc}") from exc
    if size > MAX_READ_JSON_BYTES:
        raise ModelingError(
            f"onboarding_json_exceeds_50mib_limit:{path.name}:size={size}:limit={MAX_READ_JSON_BYTES}"
        )
    try:
        raw_bytes = path.read_bytes()
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_json_read_failed:{path.name}:{exc}") from exc
    if len(raw_bytes) > MAX_READ_JSON_BYTES:
        raise ModelingError(
            f"onboarding_json_exceeds_50mib_limit:{path.name}:size={len(raw_bytes)}:limit={MAX_READ_JSON_BYTES}"
        )
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelingError(f"onboarding_json_parse_failed:{path.name}:{exc}") from exc
    return parsed, hashlib.sha256(raw_bytes).hexdigest()


@contextmanager
def temporary_scaffold_profile(
    sku: str,
    profile_override: Mapping[str, Any] | None,
) -> Iterator[None]:
    """Temporarily expose a scaffolded new-SKU profile to the canonical sealer."""
    if profile_override is None:
        yield
        return
    reg = ascend_profiles()
    already_present = sku in reg
    if not already_present:
        reg[sku] = deepcopy(dict(profile_override))
    try:
        yield
    finally:
        if not already_present:
            reg.pop(sku, None)


def materialize_sku_hardware(
    sku: str,
    *,
    profile_override: Mapping[str, Any] | None = None,
) -> Hardware:
    """Materialize a 1-device ``Hardware`` descriptor for an existing or scaffolded SKU."""
    if profile_override is not None:
        raw_machine = {
            "sku": sku,
            "architecture": profile_override["architecture"],
            "dies_per_package": profile_override["dies_per_package"],
            "ranks_per_node": int(profile_override.get("ranks_per_node", 1)),
            "num_devices": 1,
            "device": deepcopy(profile_override["device"]),
            "transfer_channels": deepcopy(profile_override.get("transfer_channels", {})),
        }
        return Hardware.from_mapping(raw_machine)
    return Hardware.from_mapping(ascend({"sku": sku, "num_devices": 1}))


def unwrap_capture_envelope(raw: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Unwrap canonical ``{'tag', 'payload'}`` capture files."""
    if raw is None:
        return None
    if set(raw.keys()) == {"tag", "payload"} and isinstance(raw.get("payload"), Mapping):
        return raw["payload"]
    return raw


def _validate_sample_device_name(raw: Mapping[str, Any] | None, *, sku: str, label: str) -> None:
    if not isinstance(raw, Mapping):
        return
    sample_dev = str(raw.get("device_name") or "").strip()
    if sample_dev and sample_dev != sku:
        raise ModelingError(f"bench_sample_sku_mismatch:{label}:expected={sku}:got={sample_dev}")


def _merge_l0_and_fabric_measurements(
    raw_l0: Mapping[str, Any] | None,
    raw_fabric: Mapping[str, Any] | None,
    *,
    sku: str,
) -> dict[str, Any] | None:
    _validate_sample_device_name(raw_l0, sku=sku, label="raw_l0")
    _validate_sample_device_name(raw_fabric, sku=sku, label="raw_fabric")
    if raw_l0 is None and raw_fabric is None:
        return None
    merged: dict[str, Any] = dict(raw_l0 or {})
    if raw_fabric:
        for key, value in raw_fabric.items():
            if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
                combined = dict(merged[key])
                combined.update(value)
                merged[key] = combined
            else:
                merged[key] = value
    return merged


def _extract_cann_version(raw: Mapping[str, Any] | None) -> str:
    if not isinstance(raw, Mapping):
        return ""
    val = str(raw.get("cann_version") or "").strip()
    if not val and isinstance(raw.get("observed_silicon"), Mapping):
        val = str(raw["observed_silicon"].get("cann_version") or "").strip()
    return "" if val.lower() in _ABSENT_CANN_TOKENS else val


def _manifest_digest_to_cann(
    *,
    repo_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, str]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(Path(manifest_path))
    if repo_root is not None:
        candidates.append(resolve_repo_root(repo_root) / "data" / "primitives" / "manifest.json")
    candidates.append(default_primitives_dir() / "manifest.json")
    mapping: dict[str, str] = {}
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            doc = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tables = doc.get("tables") if isinstance(doc, Mapping) else None
        if not isinstance(tables, Mapping):
            continue
        for entry in tables.values():
            if not isinstance(entry, Mapping):
                continue
            cann = str(entry.get("cann_version") or "").strip()
            if cann.lower() in _ABSENT_CANN_TOKENS:
                continue
            cap_uri = str(entry.get("capture_uri") or "").strip()
            if cap_uri and "/" in cap_uri:
                mapping[cap_uri.rsplit("/", 1)[-1].lower()] = cann
            env_sha = str(entry.get("envelope_sha256") or "").strip().lower()
            if env_sha:
                mapping[env_sha] = cann
    return mapping


def check_cann_version_consistency(
    raw_l0: Mapping[str, Any] | None,
    requested_cann: str,
    *,
    raw_fabric: Mapping[str, Any] | None = None,
    probe_cann: str = "",
    capture_digests: Sequence[str] = (),
    manifest_cann_by_digest: Mapping[str, str] | None = None,
    allow_unbacked_requested: bool = False,
) -> str:
    """Reject CANN contradictions across probe, capture, manifest, and request; emit ``unknown`` if absent."""
    evidence: list[tuple[str, str]] = []
    for label, val in (
        ("probe", str(probe_cann or "").strip()),
        ("l0_capture", _extract_cann_version(raw_l0)),
        ("fabric_capture", _extract_cann_version(raw_fabric)),
    ):
        if val.lower() not in _ABSENT_CANN_TOKENS:
            evidence.append((label, val))
    if manifest_cann_by_digest:
        for dig in capture_digests:
            rec_cann = manifest_cann_by_digest.get(str(dig).strip().lower(), "")
            if rec_cann and rec_cann.lower() not in _ABSENT_CANN_TOKENS:
                evidence.append(("manifest_capture", rec_cann))
                break

    req = str(requested_cann or "").strip()
    req_present = req.lower() not in _ABSENT_CANN_TOKENS
    sources = [*evidence, *( [("requested", req)] if req_present else [] )]
    distinct = {v for _lbl, v in sources}
    if len(distinct) > 1:
        detail = ":".join(f"{lbl}={val}" for lbl, val in sources)
        raise ModelingError(f"cann_version_provenance_conflict:{detail}")
    if evidence:
        return evidence[0][1]
    if raw_l0 is None and raw_fabric is None:
        return req if req_present else "unmeasured"
    if req_present:
        if allow_unbacked_requested:
            return req
        raise ModelingError(f"cann_version_evidence_missing:requested={req}:observed=unknown")
    return "unknown"


def _collect_live_fabric_via_profiler(
    *,
    sku: str,
    endpoint_alias: str,
    profiler_root: str | Path | None,
    granted_devices: Sequence[int],
    fence_token: int | None,
    timeout_s: int,
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any] | None:
    fab_plan = {
        "schema": "mindie-sim.cliff-bench-plan/2",
        "sku": sku,
        "family": "fabric_train",
        "plan_sha256": hashlib.sha256(f"fabric-train:{sku}".encode("utf-8")).hexdigest(),
    }
    obs_rows, _abi = run_profiler_bench_subprocess(
        fab_plan, sku=sku, profiler_root=profiler_root,
        endpoint_alias=endpoint_alias, allowed_phy_ids=granted_devices,
        fence_token=fence_token, domain=_FABRIC_DOMAIN_ID, timeout_s=timeout_s,
        ssh_runner=ssh_runner,
    )
    h2d_bps, d2h_bps = [], []
    for row in obs_rows:
        dur_ns = float(row.get("measured_ns") or 0.0)
        shape = row.get("actual_shape") if isinstance(row.get("actual_shape"), Mapping) else {}
        nbytes = float(shape.get("bytes") or row.get("work_bytes") or 0.0)
        direction = str(shape.get("direction") or "").lower()
        if dur_ns > 0.0 and nbytes > 0.0:
            rate = nbytes / (dur_ns * 1e-9)
            if direction == "h2d":
                h2d_bps.append(rate)
            elif direction == "d2h":
                d2h_bps.append(rate)
    if not h2d_bps and not d2h_bps:
        return None
    out: dict[str, Any] = {"device_name": sku, "P20_host_dma_sync": {}}
    if h2d_bps:
        out["P20_host_dma_sync"]["host_h2d_bytes_per_s"] = h2d_bps
    if d2h_bps:
        out["P20_host_dma_sync"]["host_d2h_bytes_per_s"] = d2h_bps
    return out


def _collect_l0_and_fabric_under_lease(
    *,
    sku: str,
    endpoint_alias: str,
    raw_l0_json: str | Path | None,
    raw_fabric_json: str | Path | None,
    campaign_id: str,
    cann_version: str,
    l0_budget_s: int,
    fabric_budget_s: int,
    budget: TimeoutBudget,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None,
    l0_runner: Callable[[str, int], Mapping[str, Any]] | None,
    fabric_runner: Callable[[str, int], Mapping[str, Any]] | None,
    live_bench: bool,
    use_live: bool,
    repo_root: str | Path | None,
    profiler_root: str | Path | None,
    coordinator_bin: str,
    profile_override: Mapping[str, Any] | None,
    ssh_runner: Callable[[str, str, int], Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str], Mapping[str, Any], str]:
    require_grant = bool(live_bench or lease_acquirer is not None)
    if lease_acquirer is not None:
        lease_ctx: AbstractContextManager[Mapping[str, Any]] = lease_acquirer(sku, budget.outer_timeout_s)
    elif live_bench:
        lease_ctx = acquire_coordinator_host_queue_lease(
            endpoint_alias, npu_count=1, ttl_seconds=budget.outer_timeout_s,
            min_subprocess_timeout_s=budget.total_inner_s, cleanup_grace_s=budget.cleanup_grace_s,
            coordinator_bin=coordinator_bin,
        )
    else:
        lease_ctx = nullcontext({
            "lease_mode": "offline_replay" if (raw_l0_json is not None or l0_runner is not None) else "uncalibrated_dry_run"
        })

    extra_train_digests: list[str] = []
    raw_lease_info: Mapping[str, Any] = {}
    fabric_coverage = "not_provided"
    with lease_ctx as acquired:
        raw_lease_info = acquired if isinstance(acquired, Mapping) else {}
        if require_grant:
            _, fence_val, granted_devices, _ = validate_host_queue_grant(
                raw_lease_info, required_devices=1,
                min_ttl_seconds=budget.total_inner_s + budget.cleanup_grace_s,
            )
        else:
            granted_devices = [int(d) for d in (raw_lease_info.get("devices") or ()) if not isinstance(d, bool)]
            fence_val = raw_lease_info.get("fence_token")
        if raw_l0_json is not None:
            parsed_l0, l0_sha = _read_bounded_json_and_sha256(Path(raw_l0_json))
            extra_train_digests.append(l0_sha)
            raw_l0: dict[str, Any] | None = unwrap_capture_envelope(parsed_l0)
        elif l0_runner is not None:
            raw_l0 = unwrap_capture_envelope(dict(l0_runner(sku, int(l0_budget_s))))
        elif live_bench:
            prof_entry = profile_override if profile_override is not None else ascend_profiles().get(sku)
            dies_per_pkg = max(1, int((prof_entry or {}).get("dies_per_package", 1)))
            raw_l0 = run_live_l0_bench_subprocess(
                sku=sku, endpoint_alias=endpoint_alias, campaign_id=campaign_id,
                cann_version=cann_version, repo_root=repo_root, timeout_s=int(l0_budget_s),
                visible_devices=granted_devices, dies_per_package=dies_per_pkg,
                profile_override=profile_override,
            )
        else:
            raw_l0 = None

        remote_ep = endpoint_alias.strip().lower() not in {"", "local", "localhost", "127.0.0.1", "::1"}
        if raw_fabric_json is not None:
            parsed_fab, fab_sha = _read_bounded_json_and_sha256(Path(raw_fabric_json))
            extra_train_digests.append(fab_sha)
            raw_fabric: dict[str, Any] | None = unwrap_capture_envelope(parsed_fab)
            fabric_coverage = "provided_offline"
        elif fabric_runner is not None:
            raw_fabric = unwrap_capture_envelope(dict(fabric_runner(sku, int(fabric_budget_s))))
            fabric_coverage = "provided_runner"
        elif use_live and (profiler_root is not None or ssh_runner is not None or remote_ep):
            raw_fabric = _collect_live_fabric_via_profiler(
                sku=sku, endpoint_alias=endpoint_alias, profiler_root=profiler_root,
                granted_devices=granted_devices,
                fence_token=int(fence_val) if type(fence_val) is int else None,
                timeout_s=int(fabric_budget_s), ssh_runner=ssh_runner,
            )
            fabric_coverage = "collected_live_profiler" if raw_fabric else "missing_live_fabric_samples"
        elif use_live:
            raw_fabric = None
            fabric_coverage = "missing_live_fabric_collector"
        else:
            raw_fabric = None
    return raw_l0, raw_fabric, extra_train_digests, raw_lease_info, fabric_coverage


def calibrate_l0_and_fabric(
    *,
    sku: str,
    endpoint_alias: str = "",
    raw_l0_json: str | Path | None = None,
    raw_fabric_json: str | Path | None = None,
    out_primitives: str | Path | None = None,
    manifest_path: str | Path | None = None,
    train_manifest_out: str | Path | None = None,
    campaign_id: str = "camp-onboard-l0",
    cann_version: str = "unmeasured",
    l0_budget_s: int = DEFAULT_L0_BUDGET_S,
    fabric_budget_s: int = DEFAULT_FABRIC_BUDGET_S,
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S,
    outer_timeout_s: int | None = None,
    idle_precheck: Mapping[str, Any] | None = None,
    lease_acquirer: Callable[[str, int], AbstractContextManager[Mapping[str, Any]]] | None = None,
    l0_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    fabric_runner: Callable[[str, int], Mapping[str, Any]] | None = None,
    live_bench: bool = False,
    repo_root: str | Path | None = None,
    profiler_root: str | Path | None = None,
    coordinator_bin: str = "",
    profile_override: Mapping[str, Any] | None = None,
    probe_cann_version: str = "",
    ssh_runner: Callable[[str, str, int], Any] | None = None,
) -> dict[str, Any]:
    """Stage 2: Acquire HostQueue lease, run L0+fabric calibration, seal via canonical sealer."""
    resolved_outer = (
        l0_budget_s + fabric_budget_s + cleanup_grace_s
        if outer_timeout_s is None
        else int(outer_timeout_s)
    )
    budget = TimeoutBudget(
        inner_budgets_s=(
            ("l0_primitives_p01_p20", int(l0_budget_s)),
            ("fabric_host_dma_train", int(fabric_budget_s)),
        ),
        cleanup_grace_s=int(cleanup_grace_s),
        outer_timeout_s=resolved_outer,
    )
    if idle_precheck is not None:
        verify_idle_die_telemetry(
            str(idle_precheck.get("info_text") or ""),
            str(idle_precheck.get("common_text") or ""),
            npu_id=int(idle_precheck.get("npu_id", 0)),
        )

    use_live = bool(live_bench and raw_l0_json is None and l0_runner is None)
    raw_l0, raw_fabric, extra_train_digests, raw_lease_info, fabric_coverage = (
        _collect_l0_and_fabric_under_lease(
            sku=sku, endpoint_alias=endpoint_alias, raw_l0_json=raw_l0_json,
            raw_fabric_json=raw_fabric_json, campaign_id=campaign_id, cann_version=cann_version,
            l0_budget_s=int(l0_budget_s), fabric_budget_s=int(fabric_budget_s), budget=budget,
            lease_acquirer=lease_acquirer, l0_runner=l0_runner, fabric_runner=fabric_runner,
            live_bench=live_bench, use_live=use_live, repo_root=repo_root,
            profiler_root=profiler_root, coordinator_bin=coordinator_bin,
            profile_override=profile_override, ssh_runner=ssh_runner,
        )
    )

    sealer = load_canonical_l0_sealer(repo_root)
    merged_raw = _merge_l0_and_fabric_measurements(raw_l0, raw_fabric, sku=sku)
    cap_digests = list(extra_train_digests)
    if raw_l0 is not None:
        cap_digests.append(hashlib.sha256(sealer.canonical_capture_bytes(sku, raw_l0)).hexdigest())
    if merged_raw is not None:
        cap_digests.append(hashlib.sha256(sealer.canonical_capture_bytes(sku, merged_raw)).hexdigest())
    manifest_cann_map = _manifest_digest_to_cann(repo_root=repo_root, manifest_path=manifest_path)
    effective_cann = check_cann_version_consistency(
        raw_l0, cann_version, raw_fabric=raw_fabric, probe_cann=probe_cann_version,
        capture_digests=cap_digests, manifest_cann_by_digest=manifest_cann_map,
        allow_unbacked_requested=use_live,
    )

    rec_alias = sealer.sanitize_record_alias(endpoint_alias or sku, sku)
    pub_campaign = sanitize_public_campaign_id(campaign_id)
    with temporary_scaffold_profile(sku, profile_override):
        payload, metric_conflicts, coverage_matrix = sealer.build_primitive_table_with_diagnostics(
            sku=sku, cann_version=effective_cann, endpoint_alias=rec_alias,
            campaign_id=pub_campaign, raw_measurements=merged_raw,
        )
    parse_primitive_table(payload, verify_envelope=True)
    capture_uri = str(payload["provenance"]["capture_uri"])
    capture_sha = capture_uri.rsplit("/", 1)[-1]
    clean_sku_slug = _SAFE_TOKEN_RE.sub("_", sku.lower()).strip("_")
    arch_prefix = "arch35" if payload["arch_preset"] == ARCH_PRESET_35 else "arch22"
    default_rel = f"data/primitives/{arch_prefix}_{clean_sku_slug}.json"

    if out_primitives is not None:
        assert_sanitized_payload(payload, field_name="calibrate_l0_primitives_payload")
        assert_sanitized_payload(merged_raw, field_name="calibrate_l0_capture_payload")
        out_path = Path(out_primitives)
        write_atomic_json(out_path, payload)
        cap_path = out_path.parent / "captures" / f"{out_path.stem}.capture.json"
        write_atomic_bytes(cap_path, sealer.canonical_capture_bytes(sku, merged_raw))
        if manifest_path is not None:
            sealer.update_manifest(Path(manifest_path), out_path, payload)

    calibrated_list = sorted(k for k, v in payload["primitives"].items() if v["status"] == "calibrated")
    calibrated_count = len(calibrated_list)
    train_root = f"acq-train-{payload['envelope_sha256'][:SHORT_DIGEST_LEN]}"
    train_digests = sorted({capture_sha, payload["envelope_sha256"], *extra_train_digests})
    train_manifest = {
        "schema": TRAIN_OBSERVATION_MANIFEST_SCHEMA,
        "sku": sku, "cann_version": effective_cann, "split": "train", "campaign_id": pub_campaign,
        "train_lineage_root": train_root, "train_digests": train_digests,
        "train_capture_uris": [capture_uri], "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "primitives_envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri, "calibrated_primitives": calibrated_list,
    }
    if train_manifest_out is not None:
        assert_sanitized_payload(train_manifest, field_name="calibrate_l0_train_manifest")
        write_atomic_json(train_manifest_out, train_manifest)

    public_lease = sanitize_lease_for_receipt(raw_lease_info)
    cal_ok = calibrated_count > 0
    result: dict[str, Any] = {
        "stage": "calibrate-l0-and-fabric", "sku": sku, "cann_version": effective_cann,
        "ok": cal_ok, "status": "ok" if cal_ok else "uncalibrated",
        "error": "" if cal_ok else "l0_calibration_zero_calibrated_primitives",
        "sealer_owner": "mindie_sim.primitives.sealer", "fabric_coverage": fabric_coverage,
        "default_table_path": default_rel, "timeout_budget": budget.to_dict(),
        "lease": public_lease, "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri, "calibrated_count": calibrated_count,
        "total_primitives": len(payload["primitives"]), "metric_conflicts": metric_conflicts,
        "coverage_matrix": coverage_matrix, "train_manifest": train_manifest, "primitives_payload": payload,
    }
    rec_unconf, rec_warn = collect_receipt_warnings(public_lease)
    if rec_unconf or rec_warn:
        result["receipt_unconfirmed"] = rec_unconf
        result["receipt_warning"] = rec_warn
    assert_sanitized_payload(result, field_name="calibrate_l0_result")
    return result


__all__ = [
    "calibrate_l0_and_fabric",
    "check_cann_version_consistency",
    "materialize_sku_hardware",
    "temporary_scaffold_profile",
    "unwrap_capture_envelope",
]
