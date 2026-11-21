"""Authoritative raw-capture timing, clock parity, and live-silicon evidence verifier.

[Role] Resolve and hash-verify on-disk raw capture files for onboarding release
  gates, maintain a local private digest-to-relative-path index under
  ``capture_root`` (never emitted into public reports/receipts) so opaque
  ``capture://`` locators resolve intact nested non-digest files in O(1) even
  with >4,096 unrelated files in the tree, extract capture-authoritative
  ``clock_operator``, ``measured_ns`` (from repeat samples or clock-matched
  capture fields), and ``measured_p99_ns``, re-verify signed stage reports
  against ``capture_root`` / ``capture_files``, and derive ``silicon_certified``
  strictly from verified ``coordinator_host_queue`` lease and capture evidence.
[Invariants]
  * Never trusts row-supplied ``measured_ns``, ``busy_union_ns``,
    ``elapsed_span_ns``, or ``measured_p99_ns`` over hash-verified raw capture
    bytes.
  * Every lookup via the private digest index STILL rehashes the target file
    via ``verify_capture_file_consistency`` (failing closed if deleted/tampered)
    and rejects any path escaping ``capture_root``.
  * Fallback tree walks stream via ``os.walk`` without materializing the full
    directory list; ``TimeoutError`` always propagates before ``OSError``.
  * ``verify_live_silicon_evidence`` requires HMAC-verified stage evidence,
    ``lease_mode == "coordinator_host_queue"``, opaque ``lease_digest``,
    positive ``fence_token``, sufficient granted ``device_count``, and
    ``capture_files_verified is True``.
[Timeouts & Locks] Pure local bounded file verification; zero locks, subprocesses,
  or device contexts.
[Side-Effect Policy] Read-only over capture files; best-effort atomic write of
  the private ``.mindie_capture_index.json`` sidecar strictly inside
  ``capture_root`` when writable (never leaked to public receipts).
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
import os
from pathlib import Path, PurePosixPath
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
    canonical_json_bytes,
    read_bounded_json,
    sanitize_public_run_id,
    write_atomic_json,
)

_P99_QUANTILE = 0.99
_SAMPLE_DECLARE_TOLERANCE_NS = 1
_MAX_CAPTURE_SCAN_FILES = 4096
_PRIVATE_INDEX_MAX_ENTRIES = 8192
_PRIVATE_INDEX_FILENAME = ".mindie_capture_index.json"
_PRIVATE_INDEX_SCHEMA = "mindie-private-capture-index/1"
_OPAQUE_LEASE_DIGEST_RE = re.compile(r"^lease-[0-9a-f]{16}$")
_ALLOWED_CAPTURE_STATUSES = frozenset({"", "ok", "pass", "passed", "success"})
_PRIVATE_DIGEST_CACHE: OrderedDict[tuple[str, str], str] = OrderedDict()


def _norm_sha_hex(raw_sha256: str) -> str:
    sha = str(raw_sha256 or "").strip().lower()
    return sha[7:] if sha.startswith("sha256:") else sha


def _safe_relative_candidate(root_path: Path, rel_posix: str) -> Path | None:
    rel = PurePosixPath(rel_posix)
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        return None
    cand = root_path / Path(*rel.parts)
    try:
        resolved_root = root_path.resolve()
        resolved_cand = cand.resolve()
        if not resolved_cand.is_relative_to(resolved_root):
            return None
        return resolved_cand if resolved_cand.is_file() else None
    except TimeoutError:
        raise
    except OSError:
        return None


def _load_private_index_entries(root_path: Path) -> dict[str, str]:
    idx_path = root_path / _PRIVATE_INDEX_FILENAME
    if not idx_path.is_file():
        return {}
    try:
        doc = read_bounded_json(idx_path)
    except TimeoutError:
        raise
    except (ModelingError, OSError):
        return {}
    entries = doc.get("entries") if isinstance(doc, Mapping) else None
    if not isinstance(entries, Mapping):
        return {}
    return {
        _norm_sha_hex(str(k)): str(v)
        for k, v in entries.items()
        if str(k).strip() and isinstance(v, str) and v.strip()
    }


def _record_private_capture_location(root_path: Path, raw_sha256: str, resolved_file: Path) -> None:
    """Persist a private ``sha256 -> relative_posix_path`` entry strictly under ``root_path``."""
    sha = _norm_sha_hex(raw_sha256)
    if not sha:
        return
    try:
        r_root = root_path.resolve()
        r_file = resolved_file.resolve()
        if not r_file.is_relative_to(r_root) or r_file.name == _PRIVATE_INDEX_FILENAME:
            return
        rel_posix = r_file.relative_to(r_root).as_posix()
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return
    cache_key = (str(r_root), sha)
    _PRIVATE_DIGEST_CACHE[cache_key] = rel_posix
    _PRIVATE_DIGEST_CACHE.move_to_end(cache_key)
    while len(_PRIVATE_DIGEST_CACHE) > _PRIVATE_INDEX_MAX_ENTRIES:
        _PRIVATE_DIGEST_CACHE.popitem(last=False)
    try:
        entries = _load_private_index_entries(r_root)
        if entries.get(sha) != rel_posix:
            entries[sha] = rel_posix
            if len(entries) > _PRIVATE_INDEX_MAX_ENTRIES:
                for drop_k in list(entries.keys())[: len(entries) - _PRIVATE_INDEX_MAX_ENTRIES]:
                    entries.pop(drop_k, None)
            write_atomic_json(
                r_root / _PRIVATE_INDEX_FILENAME,
                {"schema": _PRIVATE_INDEX_SCHEMA, "entries": entries},
            )
    except TimeoutError:
        raise
    except (ModelingError, OSError):
        pass


def _lookup_private_index(
    root_path: Path, sha: str, *, private_entries: dict[str, str] | None = None
) -> Path | None:
    try:
        root_key = str(root_path.resolve())
    except TimeoutError:
        raise
    except OSError:
        root_key = str(root_path)
    rel = _PRIVATE_DIGEST_CACHE.get((root_key, sha))
    if rel:
        cand = _safe_relative_candidate(root_path, rel)
        if cand is not None:
            return cand
    entries = private_entries if private_entries is not None else _load_private_index_entries(root_path)
    rel_disk = entries.get(sha)
    if rel_disk:
        return _safe_relative_candidate(root_path, rel_disk)
    return None


def build_content_digest_index(root_path: Path) -> dict[str, Path]:
    """Stream a bounded private ``{sha256: Path}`` index over ``root_path`` without full-tree materialization."""
    index: dict[str, Path] = {}
    if not root_path.is_dir():
        return index
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        filenames.sort()
        for fname in filenames:
            if fname == _PRIVATE_INDEX_FILENAME:
                continue
            cand = Path(dirpath) / fname
            scanned += 1
            try:
                if cand.is_file() and cand.stat().st_size <= MAX_READ_JSON_BYTES:
                    digest = digest_file_streaming(cand)
                    index.setdefault(digest, cand)
            except TimeoutError:
                raise
            except OSError:
                pass
            if scanned >= _MAX_CAPTURE_SCAN_FILES:
                return index
    return index


def _find_capture_by_digest(
    root_path: Path,
    raw_sha256: str,
    *,
    digest_index: dict[str, Path] | None = None,
    private_entries: dict[str, str] | None = None,
) -> Path | None:
    sha = _norm_sha_hex(raw_sha256)
    if not sha or not root_path.is_dir():
        return None
    for name in (f"{sha}.json", f"{sha}.jsonl"):
        cand = root_path / name
        if cand.is_file():
            return cand
    indexed = _lookup_private_index(root_path, sha, private_entries=private_entries)
    if indexed is not None:
        return indexed
    if digest_index is not None:
        return digest_index.get(sha)
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        filenames.sort()
        for fname in filenames:
            if fname == _PRIVATE_INDEX_FILENAME:
                continue
            cand = Path(dirpath) / fname
            scanned += 1
            try:
                if cand.is_file() and cand.stat().st_size <= MAX_READ_JSON_BYTES:
                    if digest_file_streaming(cand) == sha:
                        return cand
            except TimeoutError:
                raise
            except OSError:
                pass
            if scanned >= _MAX_CAPTURE_SCAN_FILES:
                return None
    return None


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


_SKIP_CONTAINER_KEYS = frozenset({
    "actual_shape", "executed_shape", "requested_shape", "node_shapes",
    "measured_binding", "plan_key", "shape", "shapes",
})
_MAX_CONTAINER_DEPTH = 6


def _walk_capture_mappings(
    obj: Any, cid: str, out: list[Mapping[str, Any]], *, depth: int = 0
) -> None:
    if depth > _MAX_CONTAINER_DEPTH or not isinstance(obj, Mapping):
        return
    out.append(obj)
    for sub_key in ("boundary", "metadata", "payload", "summary"):
        sub = obj.get(sub_key)
        if isinstance(sub, Mapping):
            _walk_capture_mappings(sub, cid, out, depth=depth + 1)
    for k, v in obj.items():
        if k in _SKIP_CONTAINER_KEYS or k in {"boundary", "metadata", "payload", "summary"}:
            continue
        if isinstance(v, Mapping):
            _walk_capture_mappings(v, cid, out, depth=depth + 1)
    for list_key in ("events", "records", "cases"):
        seq = obj.get(list_key)
        if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes)):
            for item in seq:
                if isinstance(item, Mapping) and str(item.get("case_id") or "") == cid:
                    _walk_capture_mappings(item, cid, out, depth=depth + 1)


def _extract_capture_containers(cap_obj: Any, cid: str) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    if isinstance(cap_obj, Mapping):
        _walk_capture_mappings(cap_obj, cid, containers, depth=0)
    elif isinstance(cap_obj, Sequence) and not isinstance(cap_obj, (str, bytes)):
        for item in cap_obj:
            if isinstance(item, Mapping) and str(item.get("case_id") or "") == cid:
                _walk_capture_mappings(item, cid, containers, depth=0)
    return containers


def _collect_capture_evidence(containers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ev: dict[str, Any] = {}
    for cont in containers:
        for key in ("case_id", "sku", "target_family", "target_op_id"):
            if key not in ev and str(cont.get(key) or "").strip():
                ev[key] = str(cont[key]).strip()
        if "split" not in ev and str(cont.get("split") or "").strip():
            ev["split"] = str(cont["split"]).strip().lower()
        if "run_id" not in ev:
            rid = str(cont.get("run_id") or cont.get("run_uid") or "").strip()
            if rid:
                ev["run_id"] = rid
        if "status" not in ev and cont.get("status") is not None:
            ev["status"] = str(cont["status"]).strip().lower()
        if "level" not in ev:
            raw_lvl = str(cont.get("level") or "").strip().upper()
            if raw_lvl in {"L1", "L2", "L3"}:
                ev["level"] = raw_lvl
        if "timed_nodes" not in ev:
            tn = cont.get("timed_nodes")
            if isinstance(tn, Sequence) and not isinstance(tn, (str, bytes)) and tn:
                ev["timed_nodes"] = [str(x) for x in tn]
        if "target_op_ids" not in ev:
            tids = cont.get("target_op_ids")
            if isinstance(tids, Sequence) and not isinstance(tids, (str, bytes)) and tids:
                ev["target_op_ids"] = [str(x) for x in tids]
        if "measured_binding" not in ev:
            mb = cont.get("measured_binding")
            if isinstance(mb, Mapping) and mb:
                ev["measured_binding"] = dict(mb)
        if "actual_shape" not in ev:
            for skey in ("actual_shape", "executed_shape"):
                sh = cont.get(skey)
                if isinstance(sh, Mapping) and sh:
                    ev["actual_shape"] = dict(sh)
                    break
        if "plan_key" not in ev:
            pk = cont.get("plan_key")
            if isinstance(pk, Mapping) and pk:
                ev["plan_key"] = dict(pk)
    return ev


def _verify_capture_container_invariants(
    containers: Sequence[Mapping[str, Any]], *, cid: str, require_identity: bool = False
) -> dict[str, Any]:
    ev = _collect_capture_evidence(containers)
    seen_levels: set[str] = set()
    seen_cids: set[str] = set()
    seen_splits: set[str] = set()
    cap_busy: int | None = None
    cap_elapsed: int | None = None
    for cont in containers:
        raw_lvl = cont.get("level")
        if raw_lvl is not None and str(raw_lvl).strip():
            lvl_up = str(raw_lvl).strip().upper()
            if lvl_up not in {"L1", "L2", "L3"}:
                raise ModelingError(f"capture_level_invalid:{cid}:level={lvl_up}")
            seen_levels.add(lvl_up)
        raw_st = cont.get("status")
        if raw_st is not None and str(raw_st).strip():
            st_low = str(raw_st).strip().lower()
            if st_low not in _ALLOWED_CAPTURE_STATUSES:
                raise ModelingError(f"capture_status_invalid:{cid}:status={st_low}")
        if bool(cont.get("debug_cross_level")):
            raise ModelingError(f"capture_debug_cross_level_forbidden:{cid}")
        if bool(cont.get("is_counterfactual")):
            raise ModelingError(f"capture_counterfactual_forbidden:{cid}")
        if cont.get("validated") is False or cont.get("qualified") is False:
            raise ModelingError(f"capture_unqualified:{cid}")
        c_cid = str(cont.get("case_id") or "").strip()
        if c_cid:
            seen_cids.add(c_cid)
        c_sp = str(cont.get("split") or "").strip().lower()
        if c_sp:
            seen_splits.add(c_sp)
        c_busy = _positive_int_ns(cont.get("busy_union_ns"))
        c_elapsed = _positive_int_ns(cont.get("elapsed_span_ns"))
        if c_busy is not None and c_elapsed is not None and c_busy > c_elapsed:
            raise ModelingError(
                f"capture_busy_union_exceeds_elapsed_span:{cid}:busy={c_busy}:elapsed={c_elapsed}"
            )
        if cap_busy is None and c_busy is not None:
            cap_busy = c_busy
        if cap_elapsed is None and c_elapsed is not None:
            cap_elapsed = c_elapsed
    if len(seen_levels) > 1:
        raise ModelingError(f"capture_level_conflict:{cid}:{sorted(seen_levels)}")
    if require_identity and not seen_cids:
        raise ModelingError(f"capture_case_id_missing:{cid}")
    foreign_cids = sorted(seen_cids - {cid})
    if foreign_cids:
        raise ModelingError(f"capture_case_id_mismatch:{cid}:capture={foreign_cids[0]}")
    if require_identity and not seen_splits:
        raise ModelingError(f"capture_split_missing:{cid}")
    if len(seen_splits) > 1:
        raise ModelingError(f"capture_split_conflict:{cid}:{sorted(seen_splits)}")
    if cap_busy is not None and cap_elapsed is not None and cap_busy > cap_elapsed:
        raise ModelingError(
            f"capture_busy_union_exceeds_elapsed_span:{cid}:busy={cap_busy}:elapsed={cap_elapsed}"
        )
    return ev


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
    private_entries: dict[str, str] | None = None,
    capture_evidence_out: dict[str, Any] | None = None,
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
            root_path, boundary.raw_sha256, digest_index=digest_index, private_entries=private_entries,
        )
    if resolved is None:
        if require_capture_files:
            raise ModelingError(f"capture_file_missing:{cid}")
        return False, None, None

    verify_capture_file_consistency(cid, boundary, resolved)
    if root_path is not None:
        _record_private_capture_location(root_path, boundary.raw_sha256, resolved)
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
    ev = _verify_capture_container_invariants(
        containers, cid=cid, require_identity=require_capture_files
    )
    if capture_evidence_out is not None:
        capture_evidence_out.update(ev)
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
    root_path = Path(capture_root) if capture_root is not None else None
    private_entries = _load_private_index_entries(root_path) if root_path is not None else None
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
        cap_ev: dict[str, Any] = {}
        verified, cap_ns, cap_p99 = resolve_and_verify_capture_truth(
            boundary,
            cid=cid,
            clock_op=clock_op,
            capture_root=capture_root,
            capture_files=capture_files,
            require_capture_files=True,
            private_entries=private_entries,
            capture_evidence_out=cap_ev,
        )
        if not verified or cap_ns is None:
            raise ModelingError(f"capture_file_missing:{cid}")
        row_lvl = str(row.get("level") or "").strip().upper()
        if cap_ev.get("level") and row_lvl and cap_ev["level"] != row_lvl:
            raise ModelingError(f"capture_level_mismatch:{cid}:capture={cap_ev['level']}:report={row_lvl}")
        row_split = str(row.get("split") or "").strip().lower()
        if cap_ev.get("split") and row_split and cap_ev["split"] != row_split:
            raise ModelingError(f"capture_split_mismatch:{cid}:capture={cap_ev['split']}:report={row_split}")
        cap_run = str(cap_ev.get("run_id") or "").strip()
        if cap_run and cap_run != run_uid and sanitize_public_run_id(cap_run) != run_uid:
            raise ModelingError(f"capture_run_id_mismatch:{cid}:capture={cap_run}:report={run_uid}")
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
    """Bind a ``GateCase`` strictly to hash-verified raw capture duration, ``p99``, level, timed nodes, and geometry."""
    del row
    cap_ev: dict[str, Any] = {}
    verified, cap_ns, cap_p99 = resolve_and_verify_capture_truth(
        gc.boundary,
        cid=gc.case_id,
        clock_op=clock_op,
        capture_root=capture_root,
        capture_files=capture_files,
        require_capture_files=require_capture_files,
        digest_index=digest_index,
        resolved_files=resolved_files,
        capture_evidence_out=cap_ev,
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
    if cap_ev.get("level") and str(cap_ev["level"]) != str(gc.level).upper():
        raise ModelingError(
            f"capture_level_mismatch:{gc.case_id}:capture={cap_ev['level']}:row={gc.level}"
        )
    if cap_ev.get("sku") and str(cap_ev["sku"]) != str(gc.cohort.sku):
        raise ModelingError(
            f"capture_sku_mismatch:{gc.case_id}:capture={cap_ev['sku']}:row={gc.cohort.sku}"
        )
    if cap_ev.get("split") and str(cap_ev["split"]) != str(gc.split).lower():
        raise ModelingError(
            f"capture_split_mismatch:{gc.case_id}:capture={cap_ev['split']}:row={gc.split}"
        )
    cap_run = str(cap_ev.get("run_id") or "").strip()
    if cap_run and cap_run != gc.boundary.run_uid and sanitize_public_run_id(cap_run) != gc.boundary.run_uid:
        raise ModelingError(
            f"capture_run_id_mismatch:{gc.case_id}:capture={cap_run}:row={gc.boundary.run_uid}"
        )
    payload_map = dict(gc.payload)
    payload_changed = False
    for skey in ("target_family", "target_op_id"):
        cap_val = str(cap_ev.get(skey) or "").strip()
        if cap_val:
            row_val = str(payload_map.get(skey) or "").strip()
            if row_val and row_val != cap_val:
                raise ModelingError(f"capture_{skey}_mismatch:{gc.case_id}:capture={cap_val}:row={row_val}")
            payload_map[skey] = cap_val
            payload_changed = True
    for lkey in ("timed_nodes", "target_op_ids"):
        cap_list = cap_ev.get(lkey)
        if cap_list:
            row_list = payload_map.get(lkey)
            if row_list is not None and [str(x) for x in row_list] != list(cap_list):
                raise ModelingError(f"capture_{lkey}_mismatch:{gc.case_id}")
            payload_map[lkey] = list(cap_list)
            payload_changed = True
    for mkey in ("actual_shape", "measured_binding"):
        cap_map = cap_ev.get(mkey)
        if isinstance(cap_map, Mapping) and cap_map:
            row_map = payload_map.get(mkey)
            if isinstance(row_map, Mapping) and row_map:
                if canonical_json_bytes(dict(row_map)) != canonical_json_bytes(dict(cap_map)):
                    raise ModelingError(f"capture_{mkey}_mismatch:{gc.case_id}")
            payload_map[mkey] = dict(cap_map)
            payload_changed = True
    cap_pk = cap_ev.get("plan_key")
    if isinstance(cap_pk, Mapping) and cap_pk and not payload_map.get("plan_key"):
        payload_map["plan_key"] = dict(cap_pk)
        payload_changed = True
    if payload_changed:
        gc = replace(gc, payload=tuple(sorted(payload_map.items())))
    return gc, True, cap_p99


__all__ = [
    "build_content_digest_index",
    "resolve_and_verify_capture_truth",
    "reverify_report_captures",
    "verify_capture_and_override_case",
    "verify_live_silicon_evidence",
]
