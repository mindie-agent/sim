"""Readers for native validation and capture evidence.

[Role]
  Load validation metadata, locate CSV lines and read capture provenance.
[Invariants]
  Capture locators are one-based CSV lines. Capture hashes and source
  identities come from caller-selected evidence.
[Timeouts & Locks]
  Local reads with streamed CSV and hash input. No locks.
[Side-Effect Policy]
  Reads explicit capture files and their side metadata without writing.
"""


import math


def _capture_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _capture_int(value):
    number = _capture_float(value)
    return None if number is None else int(number)


def load_native_cohort_validation(path, *, expected_sha256=None):
    """Read the T03 native cohort validation CSV and its source hash.

    The file is already the normalised T03 output; this function never rescans
    a capture and never uses a filename as evidence.
    """
    import csv
    import hashlib
    from pathlib import Path
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("native_cohort_validation_source_missing:" + str(source))
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if expected_sha256 and sha != str(expected_sha256):
        raise ValueError(
            "native_cohort_validation_sha256_mismatch:"
            + sha + "!=" + str(expected_sha256))
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    by_cohort = {}
    for row in rows:
        cohort_id = str(row.get("cohort_id") or "").strip()
        if not cohort_id:
            raise ValueError("native_cohort_validation_row_requires_cohort_id")
        if cohort_id in by_cohort:
            raise ValueError(
                "native_cohort_validation_duplicate_cohort_id:" + cohort_id)
        by_cohort[cohort_id] = row
    counts = {}
    for row in rows:
        key = (str(row.get("calibration_relation") or ""),
               str(row.get("contract_match_status") or ""))
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "mindie-sim.native-cohort-validation/1",
        "path": str(source),
        "sha256": sha,
        "bytes": source.stat().st_size,
        "rows": rows,
        "by_cohort": by_cohort,
        "count": len(rows),
        "counts": {f"{role}:{status}": value
                   for (role, status), value in sorted(counts.items())},
    }


def read_capture_locators(capture_path, locators, *, with_sha256=False):
    """Stream one capture and return only the explicitly requested CSV lines."""
    import csv
    import hashlib
    from pathlib import Path
    path = Path(capture_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("capture_source_missing:" + str(path))
    requested = sorted({int(item) for item in locators})
    if not requested:
        raise ValueError("capture_locator_lines_required")
    if requested[0] < 2:
        raise ValueError("capture_locator_line_is_one_based_data_line")
    found = {}
    rows_scanned = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Task ID" not in reader.fieldnames:
            raise ValueError("capture_source_requires_kernel_details_header")
        for row in reader:
            rows_scanned += 1
            if reader.line_num in requested:
                found[reader.line_num] = dict(row)
    sha = None
    if with_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "requested_lines": requested,
        "rows": found,
        "missing_lines": [line for line in requested if line not in found],
        "rows_scanned": rows_scanned,
    }


def capture_image_identity(capture_path):
    """Read the same-capture image identity from its retained side files."""
    import hashlib
    import re
    from pathlib import Path
    source = Path(capture_path).expanduser().resolve()
    directory = source.parent
    checks = []
    readme = directory / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"image\s+`([0-9a-fA-F]{12,64})`(?:\s+\(([^)]+)\))?", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": (match.group(2) or "").strip() or None,
                "source": str(readme),
                "raw": match.group(0),
                "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(readme), "status": "image_not_found"})
    env = directory / "session.env"
    if env.is_file():
        text = env.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^image=sha256:([0-9a-fA-F]{12,64})\s*$", text)
        tag = re.search(r"(?m)^image_tags=(.+)$", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": None,
                "image_tags": (tag.group(1).strip() if tag else None),
                "source": str(env),
                "raw": match.group(0),
                "sha256": hashlib.sha256(env.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(env), "status": "image_not_found"})
    return {"image_id": None, "revision": None, "source": None,
            "checks": checks, "reason": "capture_image_identity_not_declared"}
