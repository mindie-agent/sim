"""Stage-2 L0 primitive & fabric calibration for the New-SKU Onboarding Pipeline.

[Role] Execute Stage 2 (``calibrate-l0-and-fabric``) of the standardized
  New-SKU onboarding pipeline: enforce bounded timeout arithmetic, run optional
  idle-die telemetry prechecks, acquire a real ``mindie-coordinator``
  ``HostQueue`` lease bound to granted physical devices on live runs, unwrap
  canonical ``{"tag", "payload"}`` capture envelopes, enforce CANN version
  provenance consistency, delegate sealing exclusively to the canonical
  ``tools/bench_l0_primitives.py`` owner (supporting temporary registration for
  newly scaffolded SKUs), and emit a SHA-256-sealed ``train`` observation
  manifest carrying raw capture digests.
[Invariants] Zero duplicate primitive sealers; ``calibrated_count == 0`` ALWAYS
  returns ``ok=False`` and ``status="uncalibrated"``; live calibration fails
  closed without ``lease_id`` and ``fence_token``; ``TimeoutError`` propagates
  before ``OSError``; zero raw IPs/hostnames/usernames in emitted payloads.
[Timeouts & Locks] Enforces ``outer_timeout >= l0_budget + fabric_budget +
  cleanup_grace`` via ``TimeoutBudget``; ``HostQueue`` leases always unwind via
  context managers; temporary scaffold registrations always unwind in
  ``finally``.
[Side-Effect Policy] Writes only caller-requested primitive table, raw capture
  sidecar, manifest, and train-manifest paths via atomic ``os.replace``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Any

from mindie_sim.backends import ascend, ascend_profiles
from mindie_sim.primitives.schema import parse_primitive_table
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware

from ._gate_runners import (
    acquire_coordinator_host_queue_lease,
    load_canonical_l0_sealer,
    run_live_l0_bench_subprocess,
)
from ._soc_probe import ARCH_PRESET_35, verify_idle_die_telemetry
from .contracts import (
    COLD_RING_L2_MULTIPLIER,
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_FABRIC_BUDGET_S,
    DEFAULT_L0_BUDGET_S,
    SHORT_DIGEST_LEN,
    TRAIN_OBSERVATION_MANIFEST_SCHEMA,
    TimeoutBudget,
    assert_sanitized_payload,
    read_bounded_json,
    write_atomic_json,
)

_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9]+")


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
    """Unwrap canonical ``bench_l0_primitives`` ``{'tag', 'payload'}`` capture files."""
    if raw is None:
        return None
    if set(raw.keys()) == {"tag", "payload"} and isinstance(raw.get("payload"), Mapping):
        return raw["payload"]
    return raw


def _merge_l0_and_fabric_measurements(
    raw_l0: Mapping[str, Any] | None,
    raw_fabric: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
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


def check_cann_version_consistency(
    raw_l0: Mapping[str, Any] | None,
    requested_cann: str,
) -> str:
    """Reject requested CANN version strings that contradict captured silicon metadata."""
    obs_cann = ""
    if isinstance(raw_l0, Mapping):
        obs_cann = str(raw_l0.get("cann_version") or "").strip()
        if not obs_cann and isinstance(raw_l0.get("observed_silicon"), Mapping):
            obs_cann = str(raw_l0["observed_silicon"].get("cann_version") or "").strip()
    req = str(requested_cann or "unmeasured").strip()
    if obs_cann and req and req != "unmeasured" and obs_cann != req:
        raise ModelingError(
            f"cann_version_provenance_conflict:observed={obs_cann}:requested={req}"
        )
    return obs_cann or req or "unmeasured"


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
    coordinator_bin: str = "",
    profile_override: Mapping[str, Any] | None = None,
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
    if lease_acquirer is not None:
        lease_ctx: AbstractContextManager[Mapping[str, Any]] = lease_acquirer(
            sku, budget.outer_timeout_s
        )
    elif use_live:
        lease_ctx = acquire_coordinator_host_queue_lease(
            endpoint_alias,
            npu_count=1,
            ttl_seconds=budget.outer_timeout_s,
            coordinator_bin=coordinator_bin,
        )
    else:
        lease_ctx = nullcontext(
            {
                "lease_mode": (
                    "offline_replay"
                    if (raw_l0_json is not None or l0_runner is not None)
                    else "uncalibrated_dry_run"
                )
            }
        )

    extra_train_digests: list[str] = []
    lease_info: dict[str, Any] = {}
    with lease_ctx as acquired:
        lease_info = dict(acquired or {})
        granted_devices = [
            int(d) for d in (lease_info.get("devices") or ()) if not isinstance(d, bool)
        ]
        if use_live and (not lease_info.get("lease_id") or lease_info.get("fence_token") is None):
            raise ModelingError("live_calibration_requires_host_queue_lease_id_and_fence_token")
        if raw_l0_json is not None:
            raw_path = Path(raw_l0_json)
            if raw_path.is_file():
                extra_train_digests.append(hashlib.sha256(raw_path.read_bytes()).hexdigest())
            raw_l0 = unwrap_capture_envelope(read_bounded_json(raw_path))
        elif l0_runner is not None:
            raw_l0 = unwrap_capture_envelope(dict(l0_runner(sku, int(l0_budget_s))))
        elif live_bench:
            raw_l0 = run_live_l0_bench_subprocess(
                sku=sku,
                endpoint_alias=endpoint_alias,
                campaign_id=campaign_id,
                cann_version=cann_version,
                repo_root=repo_root,
                timeout_s=int(l0_budget_s),
                visible_devices=granted_devices,
            )
        else:
            raw_l0 = None
        if raw_fabric_json is not None:
            fab_path = Path(raw_fabric_json)
            if fab_path.is_file():
                extra_train_digests.append(hashlib.sha256(fab_path.read_bytes()).hexdigest())
            raw_fabric = unwrap_capture_envelope(read_bounded_json(fab_path))
        elif fabric_runner is not None:
            raw_fabric = unwrap_capture_envelope(dict(fabric_runner(sku, int(fabric_budget_s))))
        else:
            raw_fabric = None

    effective_cann = check_cann_version_consistency(raw_l0, cann_version)
    sealer = load_canonical_l0_sealer(repo_root)
    merged_raw = _merge_l0_and_fabric_measurements(raw_l0, raw_fabric)
    rec_alias = sealer.sanitize_record_alias(endpoint_alias or sku, sku)
    with temporary_scaffold_profile(sku, profile_override):
        payload = sealer.build_primitive_table_payload(
            sku=sku,
            cann_version=effective_cann,
            endpoint_alias=rec_alias,
            campaign_id=campaign_id,
            raw_measurements=merged_raw,
        )
    parse_primitive_table(payload, verify_envelope=True)
    capture_uri = str(payload["provenance"]["capture_uri"])
    capture_sha = capture_uri.rsplit("/", 1)[-1]
    clean_sku_slug = _SAFE_TOKEN_RE.sub("_", sku.lower()).strip("_")
    arch_prefix = "arch35" if payload["arch_preset"] == ARCH_PRESET_35 else "arch22"
    default_rel = f"data/primitives/{arch_prefix}_{clean_sku_slug}.json"

    if out_primitives is not None:
        out_path = Path(out_primitives)
        write_atomic_json(out_path, payload)
        cap_path = out_path.parent / "captures" / f"{out_path.stem}.capture.json"
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path.write_bytes(sealer.canonical_capture_bytes(sku, merged_raw))
        if manifest_path is not None:
            sealer.update_manifest(Path(manifest_path), out_path, payload)

    calibrated_list = sorted(
        k for k, v in payload["primitives"].items() if v["status"] == "calibrated"
    )
    calibrated_count = len(calibrated_list)
    train_root = f"acq-train-{payload['envelope_sha256'][:SHORT_DIGEST_LEN]}"
    train_digests = sorted({capture_sha, payload["envelope_sha256"], *extra_train_digests})
    train_manifest = {
        "schema": TRAIN_OBSERVATION_MANIFEST_SCHEMA,
        "sku": sku,
        "cann_version": effective_cann,
        "split": "train",
        "campaign_id": campaign_id,
        "train_lineage_root": train_root,
        "train_digests": train_digests,
        "train_capture_uris": [capture_uri],
        "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "primitives_envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri,
        "calibrated_primitives": calibrated_list,
    }
    if train_manifest_out is not None:
        write_atomic_json(train_manifest_out, train_manifest)

    cal_ok = calibrated_count > 0
    result = {
        "stage": "calibrate-l0-and-fabric",
        "sku": sku,
        "cann_version": effective_cann,
        "ok": cal_ok,
        "status": "ok" if cal_ok else "uncalibrated",
        "error": "" if cal_ok else "l0_calibration_zero_calibrated_primitives",
        "sealer_owner": "tools/bench_l0_primitives.py",
        "default_table_path": default_rel,
        "timeout_budget": budget.to_dict(),
        "lease": lease_info,
        "envelope_sha256": payload["envelope_sha256"],
        "capture_uri": capture_uri,
        "calibrated_count": calibrated_count,
        "total_primitives": len(payload["primitives"]),
        "train_manifest": train_manifest,
        "primitives_payload": payload,
    }
    assert_sanitized_payload(result, field_name="calibrate_l0_result")
    return result


__all__ = [
    "calibrate_l0_and_fabric",
    "check_cann_version_consistency",
    "materialize_sku_hardware",
    "temporary_scaffold_profile",
    "unwrap_capture_envelope",
]
