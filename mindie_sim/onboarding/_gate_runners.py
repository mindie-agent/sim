"""Bounded repository gate, HostQueue lease, canonical sealer, and live L0 runners.

[Role] Execute ``check_arch_governance.py``, ``check_test_orthogonality.py``,
  ``check_template_provenance.py``, ``diagnose_tail_cluster.py``, live
  ``tools/bench_l0_primitives.py`` (supporting newly scaffolded SKU profiles
  via ``--profile-json``), and ``mindie-coordinator`` ``HostQueue`` leases in
  bounded windowless subprocesses; load the single canonical installed sealer
  ``mindie_sim.primitives.sealer`` cleanly inside wheels without ``tools/``.
[Invariants]
  * Every subprocess enforces explicit timeout, propagates ``TimeoutError``
    before ``OSError``, uses windowless spawn flags on Windows, and cleans up
    temp files unconditionally.
  * ``acquire_coordinator_host_queue_lease`` verifies grant state, positive
    integer ``fence_token > 0``, exact granted device count == ``npu_count``,
    and ``ttl_seconds >= min_subprocess_timeout_s + cleanup_grace_s``; release
    failures never silently disappear—they attach ``receipt_unconfirmed=True``
    and ``receipt_warning`` onto the lease record (Principle 14).
[Timeouts & Locks] Synchronous bounded subprocesses (default 120s); no local
  locks.
[Side-Effect Policy] Read-only over repo sources; writes only explicit
  caller-requested diagnosis outputs and ephemeral temp files cleaned on exit.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from mindie_sim.primitives import sealer as _canonical_sealer
from mindie_sim.schema.common import ModelingError

from ._profiler_dispatch import normalize_profiler_records, run_profiler_bench_subprocess
from .contracts import (
    DEFAULT_CLEANUP_GRACE_S,
    DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    GateCheckOutcome,
    assert_sanitized_payload,
    sanitize_public_warning,
    subprocess_window_kwargs,
    write_atomic_json,
)

_VIOLATION_COUNT_RE = re.compile(r"(?:violations|findings):\s*(\d+)")
_DIAGNOSE_ESCALATED_EXIT = 1
_PREVIEW_LINES = 8
_COORDINATOR_LEASE_RPC_TIMEOUT_S = 45
_DEFAULT_LEASE_TTL_S = 600
_VALID_LEASE_OUTCOMES = frozenset({"", "success", "ok"})
_VALID_LEASE_STATUSES = frozenset({"", "ready", "granted", "acquired", "active", "ok"})


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the repository root containing ``mindie_sim`` and ``scripts``."""
    if repo_root is not None:
        return Path(repo_root).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mindie_sim").is_dir() and (parent / "scripts").is_dir():
            return parent
    return here.parents[2]


def _run_python_script(
    script: Path,
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-B", str(script), *args]
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"onboarding_gate_timeout:{script.name}:{timeout_s}s") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"onboarding_gate_exec_failed:{script.name}:{exc}") from exc


def _extract_count(text: str, default: int) -> int:
    match = _VIOLATION_COUNT_RE.search(text)
    return int(match.group(1)) if match else default


def run_arch_governance_gate(
    repo_root: str | Path | None = None,
    *,
    strict: bool = True,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    """Run ``scripts/check_arch_governance.py`` and return a typed outcome."""
    root = resolve_repo_root(repo_root)
    script = root / "scripts" / "check_arch_governance.py"
    if not script.is_file():
        return GateCheckOutcome("arch_governance", False, "script_missing", "scripts/check_arch_governance.py not found under repo root")
    cli_args = ["--root", str(root)]
    if strict:
        cli_args.append("--strict")
    if refs_root is not None:
        cli_args.extend(["--refs-root", str(Path(refs_root).resolve())])
    for name, path_val in sorted((repo_map or {}).items()):
        cli_args.extend(["--repo-map", f"{name}={Path(path_val).resolve()}"])
    done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
    combined = (done.stdout or "") + ("\n" + done.stderr if done.stderr else "")
    count = _extract_count(combined, 0 if done.returncode == 0 else 1)
    passed = done.returncode == 0 and count == 0
    lines = [line for line in combined.splitlines() if line.strip()]
    detail = "; ".join(lines[:_PREVIEW_LINES]) if not passed else "0 violations"
    return GateCheckOutcome(
        name="arch_governance", passed=passed, status="pass" if passed else "fail", detail=detail,
        metrics=(("violations", count), ("strict", strict), ("returncode", done.returncode)),
    )


def run_test_orthogonality_gate(
    repo_root: str | Path | None = None,
    *,
    strict: bool = True,
    base: str | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    """Run ``scripts/check_test_orthogonality.py`` in strict or differential mode."""
    root = resolve_repo_root(repo_root)
    script = root / "scripts" / "check_test_orthogonality.py"
    if not script.is_file():
        return GateCheckOutcome("test_orthogonality", False, "script_missing", "scripts/check_test_orthogonality.py not found under repo root")
    cli_args = ["--root", str(root), "--format", "json"]
    if base is not None:
        cli_args.extend(["--base", str(base)])
        mode = f"--base {base}"
    elif strict:
        cli_args.append("--strict")
        mode = "--strict"
    else:
        cli_args.extend(["--base", "HEAD"])
        mode = "--base HEAD"
    done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
    fail_count, warn_count, detail = (0 if done.returncode == 0 else 1), 0, "0 findings"
    try:
        parsed = json.loads(done.stdout)
        findings = parsed.get("findings") or []
        fails = [f for f in findings if f.get("severity", "fail") == "fail"]
        warns = [f for f in findings if f.get("severity", "fail") != "fail"]
        fail_count, warn_count = len(fails), len(warns)
        if fails:
            detail = "; ".join(f"{f.get('path')}:{f.get('code')}:{f.get('message')}" for f in fails[:_PREVIEW_LINES])
    except ValueError:
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "orthogonality_gate_error").strip()
    passed = done.returncode == 0 and fail_count == 0
    return GateCheckOutcome(
        name="test_orthogonality", passed=passed, status="pass" if passed else "fail", detail=detail,
        metrics=(("findings", fail_count), ("warnings", warn_count), ("mode", mode), ("returncode", done.returncode)),
    )


def run_template_provenance_gate(
    repo_root: str | Path | None = None,
    *,
    refs_root: str | Path | None = None,
    repo_map: Mapping[str, str | Path] | None = None,
    runtime: bool = True,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> GateCheckOutcome:
    """Run ``scripts/check_template_provenance.py`` and return a typed outcome."""
    root = resolve_repo_root(repo_root)
    script = root / "scripts" / "check_template_provenance.py"
    if not script.is_file():
        return GateCheckOutcome("template_provenance", False, "script_missing", "scripts/check_template_provenance.py not found under repo root")
    cli_args = ["--root", str(root), "--format", "json"]
    if not runtime:
        cli_args.append("--no-runtime")
    if refs_root is not None:
        cli_args.extend(["--refs-root", str(Path(refs_root).resolve())])
    for name, path_val in sorted((repo_map or {}).items()):
        cli_args.extend(["--repo-map", f"{name}={Path(path_val).resolve()}"])
    done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
    checked, violations = 0, []
    try:
        parsed = json.loads(done.stdout)
        checked = int(parsed.get("checked", 0))
        violations = list(parsed.get("violations") or [])
    except ValueError:
        if done.returncode != 0:
            return GateCheckOutcome("template_provenance", False, "fail", (done.stderr or done.stdout or "template_provenance_error").strip(), (("returncode", done.returncode),))
    passed = done.returncode == 0 and not violations
    detail = f"0 violations ({checked} refs)" if passed else "; ".join(f"{v.get('location')}:{v.get('reason')}" for v in violations[:_PREVIEW_LINES])
    return GateCheckOutcome(
        name="template_provenance", passed=passed, status="pass" if passed else "fail", detail=detail,
        metrics=(("checked", checked), ("violations", len(violations))),
    )


def run_tail_cluster_diagnosis(
    residuals: Sequence[Mapping[str, Any]],
    *,
    repo_root: str | Path | None = None,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    lineage: Sequence[Mapping[str, Any]] = (),
    engine_receipts: Sequence[Mapping[str, Any]] = (),
    output_path: str | Path | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute ``scripts/diagnose_tail_cluster.py`` over failing residuals."""
    pkg_root = resolve_repo_root(None)
    candidate_root = resolve_repo_root(repo_root)
    exec_root = candidate_root if (candidate_root / "mindie_sim" / "observations" / "multilevel_gate.py").is_file() else pkg_root
    script = exec_root / "scripts" / "diagnose_tail_cluster.py"
    if not script.is_file():
        return {
            "schema": "mindie-sim.tail-cluster-diagnosis/1", "status": "diagnostic_script_unavailable",
            "cluster_count": 0, "clusters": [], "detail": "scripts/diagnose_tail_cluster.py not found under repo root",
        }
    res_payload = {"residuals": list(residuals)}
    assert_sanitized_payload(res_payload, field_name="tail_diagnosis.residuals")
    lin_payload = {"lineage": list(lineage)} if lineage else None
    if lin_payload is not None:
        assert_sanitized_payload(lin_payload, field_name="tail_diagnosis.lineage")
    rcpt_payload = {"receipts": list(engine_receipts)} if engine_receipts else None
    if rcpt_payload is not None:
        assert_sanitized_payload(rcpt_payload, field_name="tail_diagnosis.receipts")
    with tempfile.TemporaryDirectory(prefix="msim-diag-") as tmp_dir:
        tmp = Path(tmp_dir)
        res_file = tmp / "residuals.json"
        write_atomic_json(res_file, res_payload)
        cli_args = ["--residuals", str(res_file), "--threshold-pct", str(float(threshold_pct))]
        if lin_payload is not None:
            lin_file = tmp / "lineage.json"
            write_atomic_json(lin_file, lin_payload)
            cli_args.extend(["--lineage", str(lin_file)])
        if rcpt_payload is not None:
            rcpt_file = tmp / "receipts.json"
            write_atomic_json(rcpt_file, rcpt_payload)
            cli_args.extend(["--engine-receipts", str(rcpt_file)])
        done = _run_python_script(script, cli_args, cwd=exec_root, timeout_s=timeout_s)
    if done.returncode not in (0, _DIAGNOSE_ESCALATED_EXIT) or not (done.stdout or "").strip():
        raise ModelingError(f"diagnose_tail_cluster_failed:rc={done.returncode}:{(done.stderr or '').strip()}")
    try:
        report = json.loads(done.stdout)
    except ValueError as exc:
        raise ModelingError(f"diagnose_tail_cluster_invalid_json:{exc}") from exc
    assert_sanitized_payload(report, field_name="tail_diagnosis.report")
    if output_path is not None:
        write_atomic_json(output_path, report)
    return report


def run_live_l0_bench_subprocess(
    *,
    sku: str,
    endpoint_alias: str,
    campaign_id: str,
    cann_version: str,
    repo_root: str | Path | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    container: str = "",
    npu_id: int = 0,
    chip_id: int = 0,
    device_id: int = 0,
    visible_devices: Sequence[int] = (),
    dies_per_package: int = 1,
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``tools/bench_l0_primitives.py`` as a bounded live subprocess bound to granted devices."""
    if not endpoint_alias.strip():
        raise ModelingError("live_l0_bench_requires_endpoint_alias")
    root = resolve_repo_root(repo_root)
    script = root / "tools" / "bench_l0_primitives.py"
    if not script.is_file():
        raise ModelingError(f"bench_l0_primitives_not_found:{script}")
    granted = [int(d) for d in visible_devices]
    dies_pkg = max(1, int(dies_per_package))
    eff_npu_id, eff_chip_id = divmod(granted[0], dies_pkg) if granted else (int(npu_id), int(chip_id))
    with tempfile.TemporaryDirectory(prefix="msim-l0-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        out_table, out_manifest = tmp / "l0_table.json", tmp / "manifest.json"
        cli_args = [
            "--endpoint-alias", endpoint_alias, "--sku", sku,
            "--campaign-id", campaign_id, "--cann-version", cann_version,
            "--npu-id", str(eff_npu_id), "--chip-id", str(eff_chip_id),
            "--device-id", str(int(device_id)),
            "--out", str(out_table), "--manifest", str(out_manifest),
        ]
        if profile_override is not None:
            prof_file = tmp / "scaffold_profile.json"
            prof_payload = {sku: dict(profile_override)}
            assert_sanitized_payload(prof_payload, field_name="live_l0_profile_override")
            write_atomic_json(prof_file, prof_payload)
            cli_args.extend(["--profile-json", str(prof_file)])
        if granted:
            cli_args.extend(["--visible-devices", ",".join(str(d) for d in granted)])
        if container:
            cli_args.extend(["--container", container])
        done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
        if done.returncode != 0:
            raise ModelingError(f"bench_l0_primitives_failed:rc={done.returncode}:{(done.stderr or '').strip()}")
        cap_file = tmp / "captures" / f"{out_table.stem}.capture.json"
        if not cap_file.is_file():
            raise ModelingError("bench_l0_primitives_missing_capture_payload")
        try:
            cap_doc = json.loads(cap_file.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ModelingError(f"bench_l0_primitives_invalid_capture_json:{exc}") from exc
    payload = cap_doc.get("payload") if isinstance(cap_doc, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ModelingError("bench_l0_primitives_missing_capture_dict")
    return dict(payload)


def load_canonical_l0_sealer(repo_root: str | Path | None = None) -> Any:
    """Return the single canonical installed L0 primitive sealer ``mindie_sim.primitives.sealer``."""
    del repo_root
    return _canonical_sealer


def _run_coordinator_lease_cmd(
    *, action: str, endpoint_alias: str, payload: Mapping[str, Any], timeout_s: int, coordinator_bin: str = "",
) -> dict[str, Any]:
    exe = coordinator_bin.strip() or shutil.which("mindie-coordinator") or ""
    if not exe:
        raise ModelingError("coordinator_cli_not_found:mindie-coordinator")
    cmd = [exe, "lease", "--action", action, "--alias", endpoint_alias, "--json", json.dumps(dict(payload), sort_keys=True)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False, **subprocess_window_kwargs())
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"coordinator_lease_{action}_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"coordinator_lease_{action}_os_error:{exc}") from exc
    if done.returncode != 0:
        raise ModelingError(f"coordinator_lease_{action}_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}")
    try:
        parsed = json.loads(done.stdout or "{}")
    except ValueError as exc:
        raise ModelingError(f"coordinator_lease_{action}_invalid_json:{exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ModelingError(f"coordinator_lease_{action}_non_mapping_response")
    return dict(parsed)


def validate_host_queue_grant(
    lease_info: Mapping[str, Any] | None,
    *,
    required_devices: int,
    min_ttl_seconds: int,
) -> tuple[str, int, list[int], int]:
    """Validate lease_id, positive fence_token, exact device count, and minimum TTL."""
    req_npus = int(required_devices)
    if req_npus < 1:
        raise ModelingError(f"host_queue_lease_npu_count_invalid:{req_npus}")
    info = dict(lease_info or {})
    lease_id = str(info.get("lease_id") or "").strip()
    fence_raw = info.get("fence_token")
    if not lease_id or type(fence_raw) is not int or fence_raw <= 0:
        raise ModelingError("host_queue_lease_missing_lease_id_or_positive_fence_token")
    raw_devs = info.get("devices")
    if not isinstance(raw_devs, Sequence) or isinstance(raw_devs, (str, bytes)):
        raise ModelingError("host_queue_lease_missing_granted_devices")
    granted_devices: list[int] = []
    for dev in raw_devs:
        if type(dev) is not int or dev < 0:
            raise ModelingError(f"host_queue_lease_device_invalid:{dev!r}")
        granted_devices.append(dev)
    if len(set(granted_devices)) != len(granted_devices) or len(granted_devices) != req_npus:
        raise ModelingError(
            f"host_queue_lease_device_count_mismatch:expected={req_npus}:granted={granted_devices}"
        )
    ttl_raw = info.get("ttl_seconds")
    min_ttl = max(1, int(min_ttl_seconds))
    if type(ttl_raw) is not int or ttl_raw < min_ttl:
        raise ModelingError(
            f"host_queue_lease_ttl_below_subprocess_plus_grace:ttl={ttl_raw!r}:required={min_ttl}"
        )
    return lease_id, fence_raw, granted_devices, ttl_raw


@contextmanager
def acquire_coordinator_host_queue_lease(
    endpoint_alias: str,
    *,
    npu_count: int = 1,
    devices: Sequence[int] | None = None,
    ttl_seconds: int = _DEFAULT_LEASE_TTL_S,
    min_subprocess_timeout_s: int = 0,
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S,
    rpc_timeout_s: int = _COORDINATOR_LEASE_RPC_TIMEOUT_S,
    coordinator_bin: str = "",
) -> Iterator[dict[str, Any]]:
    """Acquire and release a verified ``mindie-coordinator`` HostQueue lease."""
    alias = endpoint_alias.strip()
    if not alias:
        raise ModelingError("host_queue_lease_requires_endpoint_alias")
    req_npus = int(npu_count)
    if req_npus < 1:
        raise ModelingError(f"host_queue_lease_npu_count_invalid:{req_npus}")
    eff_ttl = int(ttl_seconds)
    min_required = int(min_subprocess_timeout_s) + int(cleanup_grace_s) if int(min_subprocess_timeout_s) > 0 else 1
    if eff_ttl < min_required:
        raise ModelingError(
            f"host_queue_lease_ttl_below_subprocess_plus_grace:ttl={eff_ttl}:required={min_required}"
        )
    req_payload: dict[str, Any] = {
        "npu_count": req_npus, "ttl_seconds": eff_ttl, "purpose": "mindie-sim-onboarding",
    }
    if devices:
        req_payload["devices"] = [int(d) for d in devices]
    acquired = _run_coordinator_lease_cmd(
        action="acquire", endpoint_alias=alias, payload=req_payload,
        timeout_s=int(rpc_timeout_s), coordinator_bin=coordinator_bin,
    )
    outcome = str(acquired.get("outcome") or "").strip().lower()
    status = str(acquired.get("status") or "").strip().lower()
    data = acquired.get("data") if isinstance(acquired.get("data"), Mapping) else acquired
    lease_id = str(data.get("lease_id") or "").strip()
    fence_raw = data.get("fence_token")
    has_fenced_lease = bool(lease_id and type(fence_raw) is int and fence_raw > 0)
    lease_record: dict[str, Any] = {
        "lease_mode": "coordinator_host_queue",
        "lease_id": lease_id,
        "fence_token": fence_raw if type(fence_raw) is int else 0,
        "devices": [],
        "ttl_seconds": eff_ttl,
        "receipt_unconfirmed": False,
        "receipt_warning": "",
    }
    warnings: list[str] = []

    def _record_warning(raw_msg: str) -> None:
        lease_record["receipt_unconfirmed"] = True
        clean = sanitize_public_warning(raw_msg)
        if clean and clean not in warnings:
            warnings.append(clean)
        lease_record["receipt_warning"] = "; ".join(warnings)

    acq_unconf = bool(acquired.get("receipt_unconfirmed") or data.get("receipt_unconfirmed"))
    acq_warn = str(acquired.get("receipt_warning") or data.get("receipt_warning") or "").strip()
    if acq_unconf or acq_warn:
        _record_warning(acq_warn or "host_queue_lease_acquire_unconfirmed")
    try:
        if outcome not in _VALID_LEASE_OUTCOMES:
            raise ModelingError(f"host_queue_lease_acquire_rejected:outcome={outcome}")
        if status not in _VALID_LEASE_STATUSES:
            raise ModelingError(f"host_queue_lease_acquire_status_failed:status={status}")
        granted_ttl = data.get("ttl_seconds", eff_ttl)
        _, _, granted_devices, validated_ttl = validate_host_queue_grant(
            {**dict(data), "lease_id": lease_id, "fence_token": fence_raw, "ttl_seconds": granted_ttl},
            required_devices=req_npus,
            min_ttl_seconds=min_required,
        )
        lease_record["devices"] = granted_devices
        lease_record["ttl_seconds"] = validated_ttl
        yield lease_record
    finally:
        if has_fenced_lease:
            try:
                rel_resp = _run_coordinator_lease_cmd(
                    action="release", endpoint_alias=alias,
                    payload={"lease_id": lease_id, "fence_token": fence_raw},
                    timeout_s=int(rpc_timeout_s), coordinator_bin=coordinator_bin,
                )
                rel_data = rel_resp.get("data") if isinstance(rel_resp.get("data"), Mapping) else {}
                rel_outcome = str(rel_resp.get("outcome") or "success").strip().lower()
                rel_status = str(rel_resp.get("status") or "ok").strip().lower()
                rel_unconf = bool(rel_resp.get("receipt_unconfirmed") or rel_data.get("receipt_unconfirmed"))
                rel_warn = str(rel_resp.get("receipt_warning") or rel_data.get("receipt_warning") or "").strip()
                if rel_unconf or rel_warn or rel_outcome not in _VALID_LEASE_OUTCOMES or rel_status not in _VALID_LEASE_STATUSES:
                    _record_warning(
                        rel_warn or f"host_queue_lease_release_unconfirmed:{rel_status or rel_outcome}"
                    )
            except TimeoutError:
                raise
            except (ModelingError, OSError) as exc:
                _record_warning(f"host_queue_lease_release_unconfirmed:{exc}")


def resolve_profiler_builder_strict(
    family: str, recipe: str, impl: str, *,
    dtypes: Sequence[str] = (), meta: Mapping[str, Any] | None = None,
    profiler_root: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve ``(builder_name, upstream_abi)`` via ``mindie_profiler.bench.operators.resolve_builder_strict``."""
    added_path, saved_modules = "", {}
    if profiler_root is not None:
        root_str = str(Path(profiler_root).resolve())
        for mod_name in list(sys.modules):
            if mod_name == "mindie_profiler" or mod_name.startswith("mindie_profiler."):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            added_path = root_str
    try:
        try:
            from mindie_profiler.bench.operators import BenchUnmodeledOperatorError, resolve_builder_strict
        except ImportError as exc:
            raise ModelingError(f"profiler_owner_unavailable:{exc}") from exc
        try:
            builder_name, resolved_abi = resolve_builder_strict(family, recipe, impl, dtypes=tuple(dtypes), meta=meta or {})
        except BenchUnmodeledOperatorError as exc:
            raise ModelingError(f"profiler_builder_unmodeled:{exc}") from exc
        return str(builder_name), str(resolved_abi)
    finally:
        if added_path and added_path in sys.path:
            sys.path.remove(added_path)
        if profiler_root is not None:
            for mod_name in list(sys.modules):
                if mod_name == "mindie_profiler" or mod_name.startswith("mindie_profiler."):
                    sys.modules.pop(mod_name, None)
            sys.modules.update(saved_modules)


__all__ = [
    "acquire_coordinator_host_queue_lease",
    "load_canonical_l0_sealer",
    "normalize_profiler_records",
    "resolve_profiler_builder_strict",
    "resolve_repo_root",
    "run_arch_governance_gate",
    "run_live_l0_bench_subprocess",
    "run_profiler_bench_subprocess",
    "run_tail_cluster_diagnosis",
    "run_template_provenance_gate",
    "run_test_orthogonality_gate",
    "validate_host_queue_grant",
]
