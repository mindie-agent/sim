"""Authoritative raw-capture timing, clock parity, and live-silicon evidence verifier.

[Role] Resolve and hash-verify on-disk raw capture files for onboarding release
  gates, extract capture-authoritative ``clock_operator``, ``measured_ns`` (from
  repeat samples or clock-matched capture fields), and ``measured_p99_ns`` (from
  repeat samples or explicit capture ``p99``, never row overrides), re-verify
  signed stage reports against ``capture_root`` / ``capture_files``, and derive
  ``silicon_certified`` strictly from verified ``coordinator_host_queue`` lease
  and capture evidence (never caller strings).
[Invariants]
  * Never trusts row-supplied ``measured_ns``, ``busy_union_ns``,
    ``elapsed_span_ns``, or ``measured_p99_ns`` over hash-verified raw capture
    bytes.
  * Raw captures lacking both repeat samples and explicit capture ``p99`` return
    ``cap_p99_ns = None``, causing strict Lean non-inversion to fail closed with
    ``lean_captured_p99_missing``.
  * ``verify_live_silicon_evidence`` requires HMAC-verified stage evidence,
    ``lease_mode == "coordinator_host_queue"``, opaque ``lease_digest``,
    positive ``fence_token``, sufficient granted ``device_count``, and
    ``capture_files_verified is True``.
[Timeouts & Locks] Pure local bounded file verification; zero locks, subprocesses,
  or device contexts.
[Side-Effect Policy] Strictly read-only over local capture files.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from pathlib import Path
import re
import statistics
from typing import Any

from mindie_sim.observations.multilevel_gate import CaptureBoundary, GateCase
from mindie_sim.observations.multilevel_stream import (
    digest_file_streaming,
    resolve_capture_file,
    verify_capture_file_consistency,
)
from mindie_sim.schema.common import ModelingError

from .contracts import (
    MAX_READ_JSON_BYTES,
    NS_PER_MS,
    NS_PER_US,
    VALID_CLOCK_OPERATORS,
    read_bounded_json,
)

_P99_QUANTILE = 0.99
_SAMPLE_DECLARE_TOLERANCE_NS = 1
_MAX_CAPTURE_SCAN_FILES = 4096
_OPAQUE_LEASE_DIGEST_RE = re.compile(r"^lease-[0-9a-f]{16}$")


def _norm_sha_hex(raw_sha256: str) -> str:
    sha = str(raw_sha256 or "").strip().lower()
    return sha[7:] if sha.startswith("sha256:") else sha


def build_content_digest_index(root_path: Path) -> dict[str, Path]:
    """Build a bounded private ``{sha256: Path}`` index over retained files under ``root_path``."""
    index: dict[str, Path] = {}
    if not root_path.is_dir():
        return index
    scanned = 0
    for cand in sorted(root_path.rglob("*")):
        if not cand.is_file():
            continue
        scanned += 1
        try:
            if cand.stat().st_size <= MAX_READ_JSON_BYTES:
                digest = digest_file_streaming(cand)
                index.setdefault(digest, cand)
        except TimeoutError:
            raise
        except OSError:
            pass
        if scanned >= _MAX_CAPTURE_SCAN_FILES:
            break
    return index


def _find_capture_by_digest(
    root_path: Path,
    raw_sha256: str,
    *,
    digest_index: dict[str, Path] | None = None,
) -> Path | None:
    sha = _norm_sha_hex(raw_sha256)
    if not sha or not root_path.is_dir():
        return None
    for name in (f"{sha}.json", f"{sha}.jsonl"):
        cand = root_path / name
        if cand.is_file():
            return cand
    if digest_index is not None:
        return digest_index.get(sha)
    return build_content_digest_index(root_path).get(sha)


def _quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return int(sorted_ns[lo])
    return int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def _positive_int_ns(val: Any, *, scale: float = 1.0) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        num = float(val) * scale
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num <= 0.0:
        return None
    return int(round(num))


def _extract_samples_ns(containers: Sequence[Mapping[str, Any]]) -> list[int]:
    sample_specs = (
        ("samples_ns", 1.0),
        ("repeats_ns", 1.0),
        ("durations_ns", 1.0),
        ("samples_us", float(NS_PER_US)),
        ("repeats_us", float(NS_PER_US)),
        ("durations_us", float(NS_PER_US)),
        ("samples_ms", float(NS_PER_MS)),
        ("repeats_ms", float(NS_PER_MS)),
        ("durations_ms", float(NS_PER_MS)),
    )
    for container in containers:
        for key, scale in sample_specs:
            raw = container.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
                vals: list[int] = []
                for item in raw:
                    parsed = _positive_int_ns(item, scale=scale)
                    if parsed is not None:
                        vals.append(parsed)
                if vals:
                    return sorted(vals)
    return []


def _extract_capture_containers(cap_obj: Any, cid: str) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    if isinstance(cap_obj, Mapping):
        for cand in (cap_obj, cap_obj.get("boundary"), cap_obj.get("metadata"), cap_obj.get("payload")):
            if isinstance(cand, Mapping):
                containers.append(cand)
        for list_key in ("events", "records", "cases"):
            seq = cap_obj.get(list_key)
            if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
                for item in seq:
                    if isinstance(item, Mapping) and str(item.get("case_id") or "") == cid:
                        containers.append(item)
    elif isinstance(cap_obj, Sequence) and not isinstance(cap_obj, (str, bytes)):
        for item in cap_obj:
            if isinstance(item, Mapping) and str(item.get("case_id") or "") == cid:
                containers.append(item)
    return containers


def resolve_and_verify_capture_truth(
    boundary: CaptureBoundary,
    *,
    cid: str,
    clock_op: str,
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
    digest_index: dict[str, Path] | None = None,
    resolved_files: dict[str, str | Path] | None = None,
) -> tuple[bool, int | None, int | None]:
    """Hash-verify on-disk capture and derive authoritative ``(measured_ns, p99_ns)``."""
    root_path = Path(capture_root) if capture_root is not None else None
    resolved = resolve_capture_file(
        boundary.capture_uri,
        boundary.source_ref,
        capture_root=root_path,
        capture_files=capture_files,
        case_id=cid,
        raw_sha256=boundary.raw_sha256,
    )
    if resolved is None and root_path is not None:
        resolved = _find_capture_by_digest(
            root_path, boundary.raw_sha256, digest_index=digest_index
        )
    if resolved is None:
        if require_capture_files:
            raise ModelingError(f"capture_file_missing:{cid}")
        return False, None, None

    verify_capture_file_consistency(cid, boundary, resolved)
    if resolved_files is not None:
        resolved_files[cid] = resolved
    cap_obj = read_bounded_json(resolved)
    containers = _extract_capture_containers(cap_obj, cid)
    if not containers:
        raise ModelingError(f"capture_payload_empty:{cid}")

    cap_clocks: set[str] = set()
    for cont in containers:
        raw_clk = cont.get("clock_operator")
        if raw_clk is not None and str(raw_clk).strip():
            clk_s = str(raw_clk).strip()
            if clk_s not in VALID_CLOCK_OPERATORS:
                raise ModelingError(f"capture_clock_operator_invalid:{cid}:{clk_s}")
            cap_clocks.add(clk_s)
    if len(cap_clocks) > 1:
        raise ModelingError(f"capture_clock_operator_conflict:{cid}:{sorted(cap_clocks)}")
    if cap_clocks:
        cap_clk = next(iter(cap_clocks))
        if cap_clk != clock_op:
            raise ModelingError(
                f"capture_clock_operator_mismatch:{cid}:expected={clock_op}:capture={cap_clk}"
            )
    elif require_capture_files:
        raise ModelingError(f"capture_clock_operator_missing:{cid}")

    samples_ns = _extract_samples_ns(containers)
    clock_field = "busy_union_ns" if clock_op == "busy_union" else "elapsed_span_ns"
    explicit_clock_ns: int | None = None
    explicit_generic_ns: int | None = None
    explicit_p99_ns: int | None = None
    for cont in containers:
        if explicit_clock_ns is None:
            explicit_clock_ns = _positive_int_ns(cont.get(clock_field))
        if explicit_generic_ns is None:
            for key, scale in (
                ("measured_ns", 1.0),
                ("p50_ns", 1.0),
                ("measured_us", float(NS_PER_US)),
                ("measured_ms", float(NS_PER_MS)),
            ):
                val = _positive_int_ns(cont.get(key), scale=scale)
                if val is not None:
                    explicit_generic_ns = val
                    break
        if explicit_p99_ns is None:
            for pkey in ("measured_p99_ns", "silicon_p99_ns", "p99_ns"):
                pval = _positive_int_ns(cont.get(pkey))
                if pval is not None:
                    explicit_p99_ns = pval
                    break

    if samples_ns:
        cap_measured_ns: int | None = int(round(statistics.median(samples_ns)))
        cap_p99_ns: int | None = _quantile_ns(samples_ns, _P99_QUANTILE)
        if explicit_clock_ns is not None and abs(explicit_clock_ns - cap_measured_ns) > _SAMPLE_DECLARE_TOLERANCE_NS:
            raise ModelingError(
                f"capture_duration_internal_mismatch:{cid}:samples_p50={cap_measured_ns}:declared_{clock_field}={explicit_clock_ns}"
            )
        if explicit_p99_ns is not None and abs(explicit_p99_ns - cap_p99_ns) > _SAMPLE_DECLARE_TOLERANCE_NS:
            raise ModelingError(
                f"capture_p99_internal_mismatch:{cid}:samples_p99={cap_p99_ns}:declared_p99={explicit_p99_ns}"
            )
    else:
        cap_measured_ns = explicit_clock_ns if explicit_clock_ns is not None else explicit_generic_ns
        cap_p99_ns = explicit_p99_ns

    if cap_measured_ns is None and require_capture_files:
        raise ModelingError(f"capture_measured_duration_missing:{cid}:{clock_field}")
    return True, cap_measured_ns, cap_p99_ns


def reverify_report_captures(
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
) -> bool:
    """Independently re-verify every residual row's raw capture bytes and timing truth."""
    if not rows:
        return False
    must_verify = bool(require_capture_files or capture_root is not None or capture_files is not None)
    if not must_verify:
        return False
    digest_index = build_content_digest_index(Path(capture_root)) if capture_root is not None else None
    for row in rows:
        cid = str(row.get("case_id") or "").strip()
        raw_sha = str(row.get("raw_sha256") or "").strip()
        cap_uri = str(row.get("capture_uri") or "").strip()
        acq = str(row.get("acquisition_root") or "").strip()
        run_uid = str(row.get("run_uid") or "").strip()
        ev_ids = tuple(str(e).strip() for e in (row.get("event_ids") or ()) if str(e).strip())
        ranks = tuple(int(r) for r in (row.get("device_ranks") or ()) if not isinstance(r, bool))
        clock_op = str(row.get("clock_operator") or "").strip()
        if not (cid and raw_sha and cap_uri and acq and run_uid and ev_ids and ranks and clock_op in VALID_CLOCK_OPERATORS):
            raise ModelingError(f"report_capture_boundary_incomplete:{cid or 'unknown'}")
        boundary = CaptureBoundary(
            raw_sha256=raw_sha,
            acquisition_uid=acq,
            run_uid=run_uid,
            parent_digests=tuple(str(p) for p in (row.get("parent_digests") or ())),
            event_ids=ev_ids,
            device_ranks=ranks,
            clock_domain=str(row.get("clock_domain") or "device_monotonic"),
            start_event=str(row.get("start_event") or "k_start"),
            end_event=str(row.get("end_event") or "k_end"),
            boundary_kind=str(row.get("boundary_kind") or "device_kernel"),
            unit="ns",
            capture_uri=cap_uri,
        )
        verified, cap_ns, cap_p99 = resolve_and_verify_capture_truth(
            boundary,
            cid=cid,
            clock_op=clock_op,
            capture_root=capture_root,
            capture_files=capture_files,
            require_capture_files=True,
            digest_index=digest_index,
        )
        if not verified or cap_ns is None:
            raise ModelingError(f"capture_file_missing:{cid}")
        row_ns = int(row.get("measured_ns") or 0)
        if row_ns != cap_ns:
            raise ModelingError(
                f"capture_measured_duration_mismatch:{cid}:capture={cap_ns}:report={row_ns}"
            )
        row_p99 = row.get("measured_p99_ns")
        if row_p99 is not None and cap_p99 is None:
            raise ModelingError(f"capture_p99_missing_for_report_claim:{cid}")
        if row_p99 is not None and cap_p99 is not None and int(row_p99) != cap_p99:
            raise ModelingError(
                f"capture_p99_mismatch:{cid}:capture={cap_p99}:report={int(row_p99)}"
            )
    return True


def verify_live_silicon_evidence(
    report_doc: Mapping[str, Any] | None,
    *,
    required_ranks: int = 1,
    stage2_lease: Mapping[str, Any] | None = None,
) -> bool:
    """Return True iff verified HMAC stage evidence proves live ``coordinator_host_queue`` acquisition."""
    if not isinstance(report_doc, Mapping):
        return False
    if str(report_doc.get("acquisition_mode") or "") != "live_silicon":
        return False
    if report_doc.get("capture_files_verified") is not True:
        return False
    lease = report_doc.get("lease")
    if not isinstance(lease, Mapping):
        return False
    if str(lease.get("lease_mode") or "") != "coordinator_host_queue":
        return False
    digest = str(lease.get("lease_digest") or "").strip()
    if not _OPAQUE_LEASE_DIGEST_RE.fullmatch(digest):
        return False
    fence = lease.get("fence_token")
    dev_cnt = lease.get("device_count")
    ttl = lease.get("ttl_seconds")
    req = max(1, int(report_doc.get("required_ranks") or required_ranks or 1))
    if (
        type(fence) is not int or fence <= 0
        or type(dev_cnt) is not int or dev_cnt < req
        or type(ttl) is not int or ttl <= 0
    ):
        return False
    if stage2_lease is not None:
        if not isinstance(stage2_lease, Mapping):
            return False
        if str(stage2_lease.get("lease_mode") or "") != "coordinator_host_queue":
            return False
        s2_dig = str(stage2_lease.get("lease_digest") or "").strip()
        s2_fence = stage2_lease.get("fence_token")
        s2_devs = stage2_lease.get("device_count")
        if (
            not _OPAQUE_LEASE_DIGEST_RE.fullmatch(s2_dig)
            or type(s2_fence) is not int or s2_fence <= 0
            or type(s2_devs) is not int or s2_devs < 1
        ):
            return False
    return True


def verify_capture_and_override_case(
    gc: GateCase,
    row: Mapping[str, Any],
    *,
    clock_op: str,
    capture_root: str | Path | None,
    capture_files: Mapping[str, str | Path] | None,
    require_capture_files: bool,
    row_p99_ns: int | None,
    digest_index: dict[str, Path] | None = None,
    resolved_files: dict[str, str | Path] | None = None,
) -> tuple[GateCase, bool, int | None]:
    """Bind a ``GateCase`` strictly to hash-verified raw capture duration and ``p99``."""
    del row
    verified, cap_ns, cap_p99 = resolve_and_verify_capture_truth(
        gc.boundary,
        cid=gc.case_id,
        clock_op=clock_op,
        capture_root=capture_root,
        capture_files=capture_files,
        require_capture_files=require_capture_files,
        digest_index=digest_index,
        resolved_files=resolved_files,
    )
    if not verified:
        return gc, False, row_p99_ns
    if cap_ns is not None:
        if int(gc.measured_ns) != cap_ns:
            raise ModelingError(
                f"capture_measured_duration_mismatch:{gc.case_id}:capture={cap_ns}:row={gc.measured_ns}"
            )
        gc = replace(gc, measured_ns=cap_ns)
    if cap_p99 is not None and row_p99_ns is not None and int(row_p99_ns) != cap_p99:
        raise ModelingError(
            f"capture_p99_mismatch:{gc.case_id}:capture={cap_p99}:row={int(row_p99_ns)}"
        )
    return gc, True, cap_p99


__all__ = [
    "build_content_digest_index",
    "resolve_and_verify_capture_truth",
    "reverify_report_captures",
    "verify_capture_and_override_case",
    "verify_live_silicon_evidence",
]
