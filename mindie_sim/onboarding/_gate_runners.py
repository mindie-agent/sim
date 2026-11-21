"""Bounded repository gate and tail-cluster diagnosis subprocess runners.

[Role] Execute ``check_arch_governance.py``, ``check_test_orthogonality.py``,
  ``check_template_provenance.py``, and ``diagnose_tail_cluster.py`` in
  bounded, windowless subprocesses and convert their outputs into typed
  ``GateCheckOutcome`` / diagnosis payloads.
[Invariants] Every subprocess enforces an explicit timeout, propagates
  ``TimeoutError`` before ``OSError``, uses windowless spawn flags on Windows,
  and cleans up temporary residual files unconditionally. Missing gate scripts
  fail closed rather than silently passing.
[Timeouts & Locks] Synchronous bounded subprocesses (default 120s); no locks.
[Side-Effect Policy] Read-only over repository sources; writes only explicit
  caller-requested diagnosis output paths and ephemeral temp files cleaned on
  exit.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    GateCheckOutcome,
    subprocess_window_kwargs,
    write_atomic_json,
)

_VIOLATION_COUNT_RE = re.compile(r"(?:violations|findings):\s*(\d+)")
_DIAGNOSE_ESCALATED_EXIT = 1
_PREVIEW_LINES = 8
_COORDINATOR_LEASE_RPC_TIMEOUT_S = 45
_DEFAULT_LEASE_TTL_S = 600
_CANONICAL_SEALER_CACHE: dict[str, Any] = {}


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
        return GateCheckOutcome(
            name="arch_governance",
            passed=False,
            status="script_missing",
            detail="scripts/check_arch_governance.py not found under repo root",
        )
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
        name="arch_governance",
        passed=passed,
        status="pass" if passed else "fail",
        detail=detail,
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
        return GateCheckOutcome(
            name="test_orthogonality",
            passed=False,
            status="script_missing",
            detail="scripts/check_test_orthogonality.py not found under repo root",
        )
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
    fail_count = 0 if done.returncode == 0 else 1
    warn_count = 0
    detail = "0 findings"
    try:
        parsed = json.loads(done.stdout)
        findings = parsed.get("findings") or []
        fails = [f for f in findings if f.get("severity", "fail") == "fail"]
        warns = [f for f in findings if f.get("severity", "fail") != "fail"]
        fail_count = len(fails)
        warn_count = len(warns)
        if fails:
            detail = "; ".join(
                f"{f.get('path')}:{f.get('code')}:{f.get('message')}"
                for f in fails[:_PREVIEW_LINES]
            )
    except ValueError:
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "orthogonality_gate_error").strip()
    passed = done.returncode == 0 and fail_count == 0
    return GateCheckOutcome(
        name="test_orthogonality",
        passed=passed,
        status="pass" if passed else "fail",
        detail=detail,
        metrics=(
            ("findings", fail_count),
            ("warnings", warn_count),
            ("mode", mode),
            ("returncode", done.returncode),
        ),
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
        return GateCheckOutcome(
            name="template_provenance",
            passed=False,
            status="script_missing",
            detail="scripts/check_template_provenance.py not found under repo root",
        )
    cli_args = ["--root", str(root), "--format", "json"]
    if not runtime:
        cli_args.append("--no-runtime")
    if refs_root is not None:
        cli_args.extend(["--refs-root", str(Path(refs_root).resolve())])
    for name, path_val in sorted((repo_map or {}).items()):
        cli_args.extend(["--repo-map", f"{name}={Path(path_val).resolve()}"])
    done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
    checked = 0
    violations: list[dict[str, Any]] = []
    try:
        parsed = json.loads(done.stdout)
        checked = int(parsed.get("checked", 0))
        violations = list(parsed.get("violations") or [])
    except ValueError:
        if done.returncode != 0:
            return GateCheckOutcome(
                name="template_provenance",
                passed=False,
                status="fail",
                detail=(done.stderr or done.stdout or "template_provenance_error").strip(),
                metrics=(("returncode", done.returncode),),
            )
    passed = done.returncode == 0 and not violations
    detail = (
        f"0 violations ({checked} refs)"
        if passed
        else "; ".join(f"{v.get('location')}:{v.get('reason')}" for v in violations[:_PREVIEW_LINES])
    )
    return GateCheckOutcome(
        name="template_provenance",
        passed=passed,
        status="pass" if passed else "fail",
        detail=detail,
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
    has_full_pkg = (
        candidate_root / "mindie_sim" / "observations" / "multilevel_gate.py"
    ).is_file()
    exec_root = candidate_root if has_full_pkg else pkg_root
    script = exec_root / "scripts" / "diagnose_tail_cluster.py"
    if not script.is_file():
        return {
            "schema": "mindie-sim.tail-cluster-diagnosis/1",
            "status": "diagnostic_script_unavailable",
            "cluster_count": 0,
            "clusters": [],
            "detail": "scripts/diagnose_tail_cluster.py not found under repo root",
        }
    with tempfile.TemporaryDirectory(prefix="msim-diag-") as tmp_dir:
        tmp = Path(tmp_dir)
        res_file = tmp / "residuals.json"
        write_atomic_json(res_file, {"residuals": list(residuals)})
        cli_args = [
            "--residuals", str(res_file),
            "--threshold-pct", str(float(threshold_pct)),
        ]
        if lineage:
            lin_file = tmp / "lineage.json"
            write_atomic_json(lin_file, {"lineage": list(lineage)})
            cli_args.extend(["--lineage", str(lin_file)])
        if engine_receipts:
            rcpt_file = tmp / "receipts.json"
            write_atomic_json(rcpt_file, {"receipts": list(engine_receipts)})
            cli_args.extend(["--engine-receipts", str(rcpt_file)])
        done = _run_python_script(script, cli_args, cwd=exec_root, timeout_s=timeout_s)
    if done.returncode not in (0, _DIAGNOSE_ESCALATED_EXIT) or not (done.stdout or "").strip():
        raise ModelingError(
            f"diagnose_tail_cluster_failed:rc={done.returncode}:{(done.stderr or '').strip()}"
        )
    try:
        report = json.loads(done.stdout)
    except ValueError as exc:
        raise ModelingError(f"diagnose_tail_cluster_invalid_json:{exc}") from exc
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
    device_id: int = 0,
    visible_devices: Sequence[int] = (),
) -> dict[str, Any]:
    """Execute ``tools/bench_l0_primitives.py`` as a bounded live subprocess bound to granted devices."""
    if not endpoint_alias.strip():
        raise ModelingError("live_l0_bench_requires_endpoint_alias")
    root = resolve_repo_root(repo_root)
    script = root / "tools" / "bench_l0_primitives.py"
    if not script.is_file():
        raise ModelingError(f"bench_l0_primitives_not_found:{script}")
    granted = [int(d) for d in visible_devices]
    eff_npu_id = granted[0] if granted else int(npu_id)
    with tempfile.TemporaryDirectory(prefix="msim-l0-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        out_table = tmp / "l0_table.json"
        out_manifest = tmp / "manifest.json"
        cli_args = [
            "--endpoint-alias", endpoint_alias,
            "--sku", sku,
            "--campaign-id", campaign_id,
            "--cann-version", cann_version,
            "--npu-id", str(eff_npu_id),
            "--device-id", str(int(device_id)),
            "--out", str(out_table),
            "--manifest", str(out_manifest),
        ]
        if granted:
            cli_args.extend(["--visible-devices", ",".join(str(d) for d in granted)])
        if container:
            cli_args.extend(["--container", container])
        done = _run_python_script(script, cli_args, cwd=root, timeout_s=timeout_s)
        if done.returncode != 0:
            raise ModelingError(
                f"bench_l0_primitives_failed:rc={done.returncode}:{(done.stderr or '').strip()}"
            )
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
    """Load the single canonical L0 primitive sealer module ``tools/bench_l0_primitives.py``."""
    root = resolve_repo_root(repo_root)
    script = root / "tools" / "bench_l0_primitives.py"
    if not script.is_file():
        script = resolve_repo_root(None) / "tools" / "bench_l0_primitives.py"
    if not script.is_file():
        raise ModelingError(f"canonical_l0_sealer_not_found:{script}")
    cache_key = str(script.resolve())
    cached = _CANONICAL_SEALER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("_mindie_canonical_bench_l0", script)
    if spec is None or spec.loader is None:
        raise ModelingError(f"canonical_l0_sealer_unimportable:{script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CANONICAL_SEALER_CACHE[cache_key] = mod
    return mod


def _run_coordinator_lease_cmd(
    *,
    action: str,
    endpoint_alias: str,
    payload: Mapping[str, Any],
    timeout_s: int,
    coordinator_bin: str = "",
) -> dict[str, Any]:
    exe = coordinator_bin.strip() or shutil.which("mindie-coordinator") or ""
    if not exe:
        raise ModelingError("coordinator_cli_not_found:mindie-coordinator")
    cmd = [
        exe,
        "lease",
        "--action",
        action,
        "--alias",
        endpoint_alias,
        "--json",
        json.dumps(dict(payload), sort_keys=True),
    ]
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            **subprocess_window_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"coordinator_lease_{action}_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"coordinator_lease_{action}_os_error:{exc}") from exc
    if done.returncode != 0:
        raise ModelingError(
            f"coordinator_lease_{action}_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}"
        )
    try:
        parsed = json.loads(done.stdout or "{}")
    except ValueError as exc:
        raise ModelingError(f"coordinator_lease_{action}_invalid_json:{exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ModelingError(f"coordinator_lease_{action}_non_mapping_response")
    return dict(parsed)


@contextmanager
def acquire_coordinator_host_queue_lease(
    endpoint_alias: str,
    *,
    npu_count: int = 1,
    devices: Sequence[int] | None = None,
    ttl_seconds: int = _DEFAULT_LEASE_TTL_S,
    rpc_timeout_s: int = _COORDINATOR_LEASE_RPC_TIMEOUT_S,
    coordinator_bin: str = "",
) -> Iterator[dict[str, Any]]:
    """Acquire and release a real ``mindie-coordinator`` HostQueue lease (``mindie.lease`` schema)."""
    alias = endpoint_alias.strip()
    if not alias:
        raise ModelingError("host_queue_lease_requires_endpoint_alias")
    req_payload: dict[str, Any] = {
        "npu_count": int(npu_count),
        "ttl_seconds": int(ttl_seconds),
        "purpose": "mindie-sim-onboarding",
    }
    if devices:
        req_payload["devices"] = [int(d) for d in devices]
    acquired = _run_coordinator_lease_cmd(
        action="acquire",
        endpoint_alias=alias,
        payload=req_payload,
        timeout_s=int(rpc_timeout_s),
        coordinator_bin=coordinator_bin,
    )
    if str(acquired.get("outcome") or "").lower() in ("failed", "blocked", "timeout"):
        raise ModelingError(f"host_queue_lease_acquire_rejected:{acquired.get('status') or acquired.get('outcome')}")
    if str(acquired.get("status") or "").lower() in ("failed", "blocked", "timeout", "insufficient_capacity"):
        raise ModelingError(f"host_queue_lease_acquire_status_failed:{acquired.get('status')}")
    data = acquired.get("data") if isinstance(acquired.get("data"), Mapping) else acquired
    lease_id = str(data.get("lease_id") or "").strip()
    fence_raw = data.get("fence_token")
    if not lease_id or fence_raw is None or isinstance(fence_raw, bool):
        raise ModelingError("host_queue_lease_missing_lease_id_or_fence_token")
    fence_token = int(fence_raw)
    granted_devices = [int(d) for d in (data.get("devices") or ()) if not isinstance(d, bool)]
    lease_record = {
        "lease_mode": "coordinator_host_queue",
        "lease_id": lease_id,
        "fence_token": fence_token,
        "devices": granted_devices,
    }
    try:
        yield lease_record
    finally:
        try:
            _run_coordinator_lease_cmd(
                action="release",
                endpoint_alias=alias,
                payload={"lease_id": lease_id, "fence_token": fence_token},
                timeout_s=int(rpc_timeout_s),
                coordinator_bin=coordinator_bin,
            )
        except TimeoutError:
            raise
        except ModelingError:
            pass


def resolve_profiler_builder_strict(
    family: str,
    recipe: str,
    impl: str,
    *,
    dtypes: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
    profiler_root: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve ``(builder_name, upstream_abi)`` via canonical ``mindie_profiler.bench.operators.resolve_builder_strict``."""
    added_path = ""
    saved_modules: dict[str, Any] = {}
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
            from mindie_profiler.bench.operators import (
                BenchUnmodeledOperatorError,
                resolve_builder_strict,
            )
        except ImportError as exc:
            raise ModelingError(f"profiler_owner_unavailable:{exc}") from exc
        try:
            builder_name, resolved_abi = resolve_builder_strict(
                family,
                recipe,
                impl,
                dtypes=tuple(dtypes),
                meta=meta or {},
            )
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


def run_profiler_bench_subprocess(
    plan_doc: Mapping[str, Any],
    *,
    sku: str,
    profiler_root: str | Path | None = None,
    allowed_phy_ids: Sequence[int] = (),
    domain: int | None = None,
    dry_run: bool = False,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Invoke ``python -m mindie_profiler.bench.bench_sku_microbench`` with real CLI flags."""
    del plan_doc
    if profiler_root is None:
        raise ModelingError("profiler_bench_requires_profiler_root")
    prof_root = Path(profiler_root).resolve()
    mod_file = prof_root / "mindie_profiler" / "bench" / "bench_sku_microbench.py"
    if not mod_file.is_file():
        raise ModelingError(f"profiler_bench_script_not_found:{mod_file}")
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{prof_root}{os.pathsep}{existing_pp}" if existing_pp else str(prof_root)
    with tempfile.TemporaryDirectory(prefix="msim-prof-bench-") as tmp_dir:
        tmp = Path(tmp_dir)
        out_file = tmp / "bench_receipt.json"
        obs_file = tmp / "observations.json"
        cmd = [
            sys.executable, "-B", "-m", "mindie_profiler.bench.bench_sku_microbench",
            "--sku", sku,
            "--out", str(out_file),
            "--observations", str(obs_file),
        ]
        if allowed_phy_ids:
            cmd.extend(["--allowed-phy-ids", ",".join(str(int(d)) for d in allowed_phy_ids)])
        if domain is not None:
            cmd.extend(["--domain", str(int(domain))])
        if dry_run:
            cmd.append("--dry-run")
        try:
            done = subprocess.run(
                cmd,
                cwd=str(prof_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                **subprocess_window_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"profiler_bench_timeout:timeout_s={timeout_s}") from exc
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(f"profiler_bench_os_error:{exc}") from exc
        if done.returncode != 0:
            raise ModelingError(
                f"profiler_bench_failed:rc={done.returncode}:{(done.stderr or done.stdout).strip()}"
            )
        target_file = obs_file if obs_file.is_file() else out_file
        if not target_file.is_file():
            raise ModelingError("profiler_bench_missing_output_json")
        try:
            raw = json.loads(target_file.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ModelingError(f"profiler_bench_invalid_json:{exc}") from exc
    if isinstance(raw, Mapping):
        rows = raw.get("observations") if raw.get("observations") is not None else raw.get("cases", [])
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ModelingError("profiler_bench_observations_not_list")
    return [dict(r) for r in rows if isinstance(r, Mapping)]


__all__ = [
    "acquire_coordinator_host_queue_lease",
    "load_canonical_l0_sealer",
    "resolve_profiler_builder_strict",
    "resolve_repo_root",
    "run_arch_governance_gate",
    "run_live_l0_bench_subprocess",
    "run_profiler_bench_subprocess",
    "run_tail_cluster_diagnosis",
    "run_template_provenance_gate",
    "run_test_orthogonality_gate",
]
