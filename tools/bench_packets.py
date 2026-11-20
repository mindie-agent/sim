"""Compile profiler receipts into the four maintained ``mindie_sim`` inputs.

[Role]
  Stream raw JSONL observation receipts produced by the device-side bench and
  fabric probes, reject unsafe provenance and unqualified/split-less rows, and
  emit the four CLI formats: ``fit-service`` JSON, ``fit-rate`` JSON,
  ``calibrate`` JSONL and ``accuracy`` JSON.  The tool never fits, predicts or
  invents a duration; it only projects immutable observations.

[Invariants]
  * Raw receipts are recursively scrubbed before any field is projected;
    percent-encoded and registry-uppercase leaks are rejected, and a source
    must always be an explicit ``capture://`` URI.
  * Every observation carries an explicit ``train``/``heldout`` split and an
    explicit qualified status.  Missing split/status and failed validation are
    hard errors, never silent defaults; duplicates dedupe idempotently while a
    conflicting split or duration is fatal.
  * ``fit-service``/``fit-rate`` group multi-size sweeps by a normalized curve
    identity and omit single-point operator groups; explicit ``service``/
    ``rate``/``fabric`` inputs still require two distinct training points.
  * ``calibrate`` lines contain exactly the six maintained top-level keys and a
    validated primitive key.  Heldout rows never enter a training packet, and
    ``accuracy`` observations are heldout-only (including fabric) with a full
    identity.

[Timeouts & Locks]
  Line-oriented bounded reads of local JSONL; no subprocess, device, network
  or lock is used.

[Side-Effect Policy]
  Writes only the requested packet files under ``--out``; never mutates input.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from mindie_sim.prediction.primitives import PrimitiveKey

SCHEMA = "mindie-sim.bench-packets/1"
FABRIC_SAMPLE_SCHEMA = "mindie-fabric-probe-sample/3"
FABRIC_SUMMARY_SCHEMA = "mindie-fabric-probe-samples/3"
FIT_SERVICE_NAME, FIT_RATE_NAME = "fit-service.json", "fit-rate.json"
CALIBRATE_NAME, ACCURACY_NAME, MANIFEST_NAME = "calibrate.jsonl", "accuracy.json", "manifest.json"
SERVICE_BOUNDARIES = frozenset({"resource_phase", "device_kernel"})
ACCURACY_LEVELS = frozenset({"operator", "subgraph", "model"})
IDENTITY_FIELDS = ("level", "boundary", "hardware", "implementation", "phase", "shape", "dtype", "layout", "execution")
PRIMITIVE_KEY_FIELDS = ("hardware", "implementation", "kernel", "dtype", "shape", "scope")
SENSITIVE_KEYS = frozenset({
    "host", "host_name", "hostname", "host_ip", "ip", "ip_address",
    "device_node", "device_path", "container", "container_id", "container_name",
    "image", "image_id", "image_locator", "ssh", "ssh_destination",
    "checkpoint", "checkpoint_path", "checkpoint_location",
})
HOME_MARKERS = ("$HOME", "${HOME}", "~/", "/home/", "/Users/", "/root/", "/Volumes/", "/mnt/", "\\Users\\")
DEVICE_MARKERS = ("/dev/davinci", "/dev/davinci_manager", "/dev/devmm_svm")
REGISTRY_MARKERS = ("docker" + ".io/", "swr" + ".cn-", "quay" + ".io/")
DIGEST_MARKER = "@sha" + "256:"
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_RE = re.compile(r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:.])")
CAPTURE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OK_STATUSES = frozenset({"ok", "completed", "passed", "success", "measured"})
SKIP_KINDS = frozenset({"status", "control"})
SKIP_RECORD_TYPES = frozenset({"case_complete", "control", "summary"})
REQUIRED_IDENTITY_KEYS = ("hardware", "implementation", "phase", "dtype")
FABRIC_GROUP_FIELDS = (
    "run_id", "tier", "op", "group_size", "bytes", "protocol", "regime",
    "timing_mode", "split_style", "trial_index", "repeat_index", "pair_index", "round_index",
)
CURVE_SWEEP_KEYS = frozenset({"bytes", "work_bytes", "length", "tokens", "m", "work"})
EXPLICIT_PACKET_KINDS = frozenset({"service", "rate", "primitive", "accuracy"})


class PacketError(ValueError):
    """A receipt, condition or provenance fact that cannot be exported safely."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any, prefix: str = "") -> str:
    return prefix + hashlib.sha256(_canonical(value).encode()).hexdigest()[:20]


def _scan_ip_literals(text: str, path: str) -> None:
    for match in IPV4_RE.findall(text):
        if all(0 <= int(part) <= 255 for part in match.split(".")):
            raise PacketError(f"private_provenance_ipv4:{path}:{match}")
    for match in IPV6_RE.findall(text):
        try:
            address = ipaddress.ip_address(match)
        except ValueError:
            continue
        if address.version == 6:
            raise PacketError(f"private_provenance_ipv6:{path}:{match}")


def _scan_uri(text: str, path: str) -> None:
    if "://" not in text:
        return
    parsed = urlsplit(text)
    if parsed.scheme != "capture":
        raise PacketError(f"provenance_requires_capture_uri:{path}:{parsed.scheme}")
    if (not parsed.netloc or not CAPTURE_ID_RE.match(parsed.netloc)
            or IPV4_RE.search(parsed.netloc) or ":" in parsed.netloc):
        raise PacketError(f"capture_uri_requires_opaque_id:{path}")


def _scan_string(text: str, path: str) -> None:
    cur = text
    while (nxt := unquote(cur)) != cur:
        cur = nxt
    lowered, raw = cur.lower(), text.lower()
    for marker in HOME_MARKERS:
        if marker in cur or marker in text:
            raise PacketError(f"private_provenance_home_path:{path}:{marker}")
    if cur == "/root" or cur.startswith("/root/"):
        raise PacketError(f"private_provenance_home_path:{path}:/root")
    for marker in DEVICE_MARKERS:
        if marker in cur or marker in text:
            raise PacketError(f"private_provenance_device_node:{path}:{marker}")
    if DIGEST_MARKER in lowered or DIGEST_MARKER in raw:
        raise PacketError(f"private_provenance_image_digest:{path}")
    for marker in REGISTRY_MARKERS:
        if marker in lowered or marker in raw:
            raise PacketError(f"private_provenance_registry:{path}:{marker}")
    _scan_ip_literals(cur, path)
    _scan_uri(cur, path)


def assert_public_provenance(value: Any, path: str = "$") -> Any:
    """Recursively reject private paths, hosts, devices and image locators."""
    if isinstance(value, str):
        _scan_string(value, path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _scan_string(key, f"{path}.<key>")
                if key.lower() in SENSITIVE_KEYS and item not in (None, "", {}, []):
                    raise PacketError(f"private_provenance_sensitive_key:{path}.{key}")
            assert_public_provenance(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_public_provenance(item, f"{path}[{index}]")
    return value


def is_public_provenance(value: Any) -> bool:
    try:
        assert_public_provenance(value)
    except PacketError:
        return False
    return True


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise PacketError(f"{name}_requires_positive_finite_number")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PacketError(f"{name}_requires_positive_integer")
    return value


def _repeat_index(value: Any) -> int:
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise PacketError("repeat_index_requires_nonnegative_integer")
    return value


def _sample_id(raw: Mapping[str, Any]) -> str:
    explicit = raw.get("sample_id")
    if explicit is not None:
        assert_public_provenance(str(explicit), "$.sample_id")
        return str(explicit)
    return _digest({key: raw.get(key) for key in (
        "case_id", "run_id", "repeat_index", "trial", "rank", "arm", "timing_mode", "residency")}, "sample-")


def _source(raw: Mapping[str, Any], sample_id: str, *, fabric: bool = False) -> str:
    del fabric
    value = raw.get("source")
    if value is None or not str(value).strip():
        raise PacketError(f"observation_requires_explicit_source:{sample_id}")
    source = str(value).strip()
    if not source.startswith("capture://"):
        raise PacketError(f"observation_source_requires_capture_uri:{sample_id}:{source}")
    assert_public_provenance(source, "$.source")
    return source


def _run_id(raw: Mapping[str, Any], sample_id: str) -> str:
    run_id = raw.get("run_id")
    if run_id is None or not str(run_id).strip():
        return f"run-{sample_id}"
    assert_public_provenance(str(run_id), "$.run_id")
    return str(run_id)


def _normalize_layout(value: Any) -> str:
    text = str(value or "").strip()
    return "contiguous" if text in {"", "declared"} else text


def _is_matrix_observation(identity: Mapping[str, Any], raw: Mapping[str, Any] | None = None) -> bool:
    shape = identity.get("shape")
    impl = str(identity.get("implementation") or "").lower()
    res = str((raw or {}).get("resource") or "").lower()
    return (res == "cube" or "matmul" in impl or "gemm" in impl
            or (isinstance(shape, Mapping) and {"n", "k"} <= set(shape)))


def _validate_identity_fields(identity: Mapping[str, Any], sample_id: str,
                              raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    missing = [
        key for key in REQUIRED_IDENTITY_KEYS
        if not identity.get(key) or str(identity.get(key)).strip() in {"", "declared"}
    ]
    if missing:
        raise PacketError(f"identity_missing_required_fields:{sample_id}:{','.join(missing)}")
    if str(identity.get("layout") or "").strip() == "declared" and _is_matrix_observation(identity, raw):
        raise PacketError(f"identity_declared_placeholder_forbidden:{sample_id}:layout")
    out = deepcopy(dict(identity))
    out["layout"] = _normalize_layout(out.get("layout"))
    return out


def _identity(raw: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    identity = raw.get("service_identity") or raw.get("identity")
    if isinstance(identity, Mapping):
        return _validate_identity_fields(identity, sample_id, raw)
    boundary = str(raw.get("boundary") or "device_kernel")
    candidate = {
        "level": str(raw.get("level") or "operator"),
        "boundary": boundary,
        "hardware": raw.get("hardware"),
        "implementation": raw.get("implementation"),
        "phase": raw.get("phase"),
        "shape": deepcopy(raw.get("shape") or {}),
        "dtype": raw.get("dtype"),
        "layout": raw.get("layout"),
        "execution": raw.get("execution") or {"mode": "eager"},
    }
    return _validate_identity_fields(candidate, sample_id, raw)


def _curve_identity(raw: Mapping[str, Any], sample_id: str, *, kind_key: str) -> dict[str, Any]:
    declared = raw.get(kind_key)
    identity = (_validate_identity_fields(declared, sample_id, raw)
                if isinstance(declared, Mapping) else _identity(raw, sample_id))
    shape = identity.get("shape")
    if isinstance(shape, Mapping):
        identity["shape"] = {key: value for key, value in shape.items() if key not in CURVE_SWEEP_KEYS}
    return identity


def _accuracy_identity(raw: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    identity = dict(raw.get("identity") or {})
    for field in IDENTITY_FIELDS:
        if not identity.get(field) and raw.get(field) is not None:
            identity[field] = raw[field]
    if not identity:
        raise PacketError(f"accuracy_observation_requires_full_identity:{sample_id}")
    identity["layout"] = _normalize_layout(identity.get("layout"))
    missing = [
        field for field in IDENTITY_FIELDS
        if not identity.get(field) or str(identity.get(field)).strip() == "declared"
        or (field == "execution" and isinstance(identity.get(field), Mapping)
            and str(identity[field].get("mode") or "").strip() == "declared")
    ]
    if missing:
        raise PacketError(f"accuracy_identity_missing:{sample_id}:{','.join(missing)}")
    if identity["level"] not in ACCURACY_LEVELS:
        raise PacketError(f"accuracy_level_invalid:{sample_id}:{identity['level']}")
    return deepcopy(dict(identity))


def _status(raw: Mapping[str, Any]) -> str:
    value = raw.get("status")
    if value is None:
        raise PacketError("observation_requires_explicit_status")
    status = str(value).strip().lower()
    if status not in OK_STATUSES:
        raise PacketError(f"observation_status_invalid:{status}")
    return status


def _split(raw: Mapping[str, Any]) -> str:
    value = raw.get("split")
    if value not in {"train", "heldout"}:
        raise PacketError("observation_requires_explicit_split")
    return str(value)


def _assert_qualified(raw: Mapping[str, Any], sample_id: str) -> bool:
    if raw.get("validated") is False:
        raise PacketError(f"observation_validation_failed:{sample_id}")
    if raw.get("timing_mode") == "host_fallback":
        raise PacketError(f"observation_host_fallback_rejected:{sample_id}")
    if raw.get("saturated_qualified") is False and raw.get("qualified") is not False:
        raise PacketError(f"observation_saturated_unqualified:{sample_id}")
    if raw.get("qualified") is False or raw.get("timer_source") == "perf_counter_host_fallback":
        if raw.get("kind") in EXPLICIT_PACKET_KINDS:
            raise PacketError(f"observation_unqualified:{sample_id}")
        return False
    return True


def _is_control(raw: Mapping[str, Any]) -> bool:
    return (raw.get("record_type") in SKIP_RECORD_TYPES
            or raw.get("schema") == FABRIC_SUMMARY_SCHEMA
            or raw.get("kind") in SKIP_KINDS)


def _observation(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if _is_control(raw):
        return None, None
    sample_id = _sample_id(raw)
    _status(raw)
    split = _split(raw)
    if not _assert_qualified(raw, sample_id):
        return None, sample_id
    _source(raw, sample_id)
    if raw.get("kind") == "accuracy":
        _accuracy_identity(raw, sample_id)
    elif (not isinstance(raw.get("key"), Mapping)
          or raw.get("identity") is not None
          or raw.get("service_identity") is not None
          or raw.get("rate_identity") is not None):
        if raw.get("kind") == "rate" and isinstance(raw.get("rate_identity"), Mapping):
            _validate_identity_fields(raw["rate_identity"], sample_id)
        else:
            _identity(raw, sample_id)
    return {**raw, "split": split}, None


def _fabric_hardware(raw: Mapping[str, Any]) -> str:
    nested = raw.get("identity") if isinstance(raw.get("identity"), Mapping) else {}
    return str(raw.get("hardware") or nested.get("hardware") or "").strip()


def _fabric_reduced(raws: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, str | None]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    repeat_splits: dict[tuple[Any, ...], set[str]] = {}
    rejected_count, last_unqualified = 0, None
    for raw in raws:
        if raw.get("validated") is not True:
            raise PacketError("fabric_sample_validation_failed")
        if raw.get("timing_mode") == "host_fallback":
            raise PacketError("fabric_sample_host_fallback_rejected")
        if raw.get("device_completion_us") is None:
            raise PacketError("fabric_sample_missing_device_completion_us")
        group_ranks = tuple(int(rank) for rank in (raw.get("group_ranks") or ()))
        pair = tuple(int(rank) for rank in (raw.get("pair") or ()))
        size = int(raw.get("bytes") or 0)
        raw_split = raw.get("split")
        if raw_split not in {"train", "heldout"}:
            raise PacketError(f"fabric_sample_requires_explicit_split:{raw_split}")
        row_split = str(raw_split)
        _source(raw, _sample_id(raw))
        split_key = (
            raw.get("run_id"), raw.get("tier"), raw.get("op"), raw.get("group_size"),
            size, raw.get("protocol"), raw.get("regime"), raw.get("split_style"),
            raw.get("trial_index"), raw.get("repeat_index"), raw.get("pair_index"),
            raw.get("round_index"), group_ranks, pair,
        )
        repeat_splits.setdefault(split_key, set()).add(row_split)
        if len(repeat_splits[split_key]) != 1:
            raise PacketError(f"conflicting_fabric_repeat_splits:{split_key}")
        if (raw.get("qualified") is False
                or (raw.get("timing_mode") == "stream_saturated" and raw.get("saturated_qualified") is not True)):
            rejected_count += 1
            last_unqualified = str(raw.get("run_id") or "fabric")
            continue
        key = (*tuple(raw.get(name) for name in FABRIC_GROUP_FIELDS), group_ranks, pair)
        group = groups.setdefault(key, {"rows": [], "ranks": set(), "group_ranks": set(), "pair": pair, "sample": raw})
        group["rows"].append(raw)
        if raw.get("rank") is not None:
            group["ranks"].add(int(raw["rank"]))
        group["group_ranks"].update(group_ranks)
    reduced = []
    for key, group in groups.items():
        declared_group_size = _positive_int(key[3], "fabric_group_size")
        expected_ranks = set(group["pair"]) if group["pair"] else group["group_ranks"]
        if not expected_ranks or group["ranks"] != expected_ranks:
            raise PacketError(f"incomplete_fabric_repeat_ranks:{key}")
        if len(group["ranks"]) != declared_group_size:
            raise PacketError(
                f"fabric_group_size_rank_count_mismatch:declared={declared_group_size},observed={len(group['ranks'])}"
            )
        sample = group["sample"]
        size = int(key[4])
        duration = max(float(row["device_completion_us"]) for row in group["rows"]) / 1000.0
        row_splits = {str(row["split"]) for row in group["rows"]}
        if len(row_splits) != 1:
            raise PacketError(f"conflicting_fabric_repeat_splits:{key}")
        split = next(iter(row_splits))
        hardwares = {_fabric_hardware(row) for row in group["rows"]}
        if not hardwares or len(hardwares) != 1:
            raise PacketError("fabric_sample_requires_consistent_hardware")
        hardware = next(iter(hardwares))
        if not hardware or hardware == "declared":
            raise PacketError("fabric_sample_requires_hardware")
        for row in group["rows"]:
            _source(row, _sample_id(row), fabric=True)
        nested = sample.get("identity") if isinstance(sample.get("identity"), Mapping) else {}
        boundary = str(nested.get("boundary") or "device_kernel")
        dtype = str(sample.get("dtype") or nested.get("dtype") or "fp32")
        raw_layout = sample.get("layout") if sample.get("layout") is not None else nested.get("layout")
        if str(raw_layout or "").strip() == "declared":
            raise PacketError("fabric_sample_declared_layout_rejected")
        layout = _normalize_layout(raw_layout)
        execution = (deepcopy(dict(nested["execution"]))
                     if isinstance(nested.get("execution"), Mapping) else {"mode": "eager"})
        sample_id = _digest({"fabric": [list(item) if isinstance(item, tuple) else item for item in key]}, "fabric-")
        rate_identity = {
            "level": "operator", "boundary": boundary, "hardware": hardware,
            "implementation": str(key[2]), "phase": "fabric",
            "shape": {"group_size": int(key[3] or 0)}, "dtype": dtype,
            "layout": layout, "execution": execution,
        }
        reduced.append({
            "kind": "fabric", "record_type": "observation", "sample_id": sample_id,
            "run_id": sample.get("run_id"), "repeat_index": _repeat_index(sample.get("repeat_index")),
            "case_id": f"fabric:{key[2]}:{key[3]}:{size}:{key[6]}:{key[8]}",
            "work_bytes": size, "duration_ms": duration,
            "source": _source(sample, sample_id, fabric=True), "status": "ok", "split": split,
            "route": sample.get("route"), "tier": key[1], "protocol": key[5],
            "split_style": key[8], "regime": key[6],
            "identity": {**rate_identity, "shape": {**rate_identity["shape"], "bytes": size}},
            "rate_identity": rate_identity,
        })
    return reduced, rejected_count, last_unqualified


def _observations(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    fabric, plain, count, rejected, last_unqualified = [], [], 0, 0, None
    for index, item in enumerate(records):
        raw = dict(item)
        count += 1
        assert_public_provenance(raw, f"$[{index}]")
        if raw.get("schema") == FABRIC_SAMPLE_SCHEMA:
            fabric.append(raw)
        else:
            observation, unqualified_id = _observation(raw)
            if unqualified_id is not None:
                rejected += 1
                last_unqualified = unqualified_id
            elif observation is not None:
                plain.append(observation)
    fab_obs, fab_rejected, fab_last = _fabric_reduced(fabric)
    rejected += fab_rejected
    if fab_last is not None:
        last_unqualified = fab_last
    observations = plain + fab_obs
    if rejected > 0 and not observations:
        raise PacketError(f"observation_unqualified:{last_unqualified}")
    return observations, count, rejected


def _dedupe(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique, seen = [], {}
    for raw in observations:
        sample_id = _sample_id(raw)
        signature = (raw["split"], raw.get("duration_ms"))
        if sample_id in seen:
            if seen[sample_id] != signature:
                raise PacketError(f"conflicting_sample:{sample_id}")
            continue
        seen[sample_id] = signature
        unique.append(dict(raw))
    return unique


def iter_receipts(paths: Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    """Stream non-empty JSONL object lines; reject arrays and blank objects."""
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PacketError(f"receipt_json_invalid:{path}:{line_number}") from error
                if not isinstance(record, Mapping):
                    raise PacketError(f"receipt_requires_object:{path}:{line_number}")
                yield dict(record)


def _service_packets(receipts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    for raw in receipts:
        if raw.get("work") is None or raw["split"] != "train":
            continue
        if raw.get("mixed_domains") is True and raw.get("kind") != "service":
            continue
        sample_id = _sample_id(raw)
        if raw.get("resource") is None or raw.get("work_unit") is None:
            raise PacketError(f"fit_service_requires_resource_and_work_unit:{sample_id}")
        boundary = str(raw.get("boundary") or "device_kernel")
        if boundary not in SERVICE_BOUNDARIES:
            raise PacketError(f"service_boundary_invalid:{sample_id}:{boundary}")
        identity = _curve_identity(raw, sample_id, kind_key="service_identity")
        active_units = _positive_int(raw.get("active_units", raw.get("units", 1)), "active_units")
        key = _canonical([raw["resource"], raw["work_unit"], identity, boundary, active_units])
        group = groups.setdefault(key, {
            "resource": str(raw["resource"]), "work_unit": str(raw["work_unit"]),
            "identity": identity, "boundary": boundary, "active_units": active_units,
            "samples": [], "seen": {}, "explicit": False,
        })
        group["explicit"] = group["explicit"] or raw.get("kind") == "service"
        sample = {
            "work": _positive(raw["work"], "work"),
            "duration_ms": _positive(raw["duration_ms"], "duration_ms"),
            "run_id": _run_id(raw, sample_id), "repeat_index": _repeat_index(raw.get("repeat_index")),
            "source": _source(raw, sample_id), "split": "train",
        }
        repeat_key = (sample["run_id"], sample["repeat_index"], sample["work"])
        if repeat_key in group["seen"] and group["seen"][repeat_key] != sample["duration_ms"]:
            raise PacketError(f"service_repeat_conflict:{sample_id}")
        group["seen"][repeat_key] = sample["duration_ms"]
        group["samples"].append(sample)
    packets, skipped = [], 0
    for group in groups.values():
        if len({sample["work"] for sample in group["samples"]}) < 2:
            if group["explicit"]:
                raise PacketError("fit_service_requires_two_distinct_training_work_values")
            skipped += 1
            continue
        packets.append({
            "resource": group["resource"], "work_unit": group["work_unit"],
            "identity": group["identity"], "boundary": group["boundary"],
            "active_units": group["active_units"],
            "provenance": {"source_kind": "mindie-profiler.bench", "export": "fit-service/1"},
            "samples": sorted(group["samples"], key=lambda s: (s["work"], s["run_id"], s["repeat_index"])),
        })
    return packets, skipped


def _rate_packets(receipts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    groups: dict[str, dict[str, Any]] = {}
    for raw in receipts:
        work_bytes = raw.get("work_bytes")
        if work_bytes is None or raw["split"] != "train":
            continue
        if type(work_bytes) is int and work_bytes == 0 and raw.get("kind") != "rate":
            continue
        sample_id = _sample_id(raw)
        identity = _curve_identity(raw, sample_id, kind_key="rate_identity")
        boundary = str(raw.get("boundary") or "device_kernel")
        route, tier, protocol = raw.get("route"), raw.get("tier"), raw.get("protocol")
        split_style, regime = raw.get("split_style"), raw.get("regime")
        key = _canonical([identity, boundary, route, tier, protocol, split_style, regime])
        group = groups.setdefault(key, {
            "identity": identity, "boundary": boundary, "route": route, "tier": tier,
            "protocol": protocol, "split_style": split_style, "regime": regime,
            "seen": {}, "samples": [], "explicit": False,
        })
        group["explicit"] = group["explicit"] or raw.get("kind") in {"rate", "fabric"}
        sample = {
            "work_bytes": _positive(work_bytes, "work_bytes"),
            "duration_ms": _positive(raw["duration_ms"], "duration_ms"),
            "run_id": _run_id(raw, sample_id), "repeat_index": _repeat_index(raw.get("repeat_index")),
            "source": _source(raw, sample_id, fabric=raw.get("kind") == "fabric"), "split": "train",
        }
        repeat_key = (sample["run_id"], sample["repeat_index"], sample["work_bytes"])
        if repeat_key in group["seen"] and group["seen"][repeat_key] != sample["duration_ms"]:
            raise PacketError(f"rate_repeat_conflict:{sample_id}")
        group["seen"][repeat_key] = sample["duration_ms"]
        group["samples"].append(sample)
    packets, conditions, skipped = [], [], 0
    for group in groups.values():
        if len({sample["work_bytes"] for sample in group["samples"]}) < 2:
            if group["explicit"]:
                raise PacketError("fit_rate_requires_two_distinct_training_work_sizes")
            skipped += 1
            continue
        packets.append({
            "samples": sorted(group["samples"], key=lambda s: (s["work_bytes"], s["run_id"], s["repeat_index"])),
        })
        cond = {name: group[name] for name in ("identity", "boundary", "route", "tier", "protocol", "split_style")}
        if group["regime"] is not None:
            cond["regime"] = group["regime"]
        conditions.append(cond)
    return packets, conditions, skipped


def _calibrate_samples(receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    samples, seen = [], {}
    for raw in receipts:
        key = raw.get("key")
        if key is None or raw["split"] != "train":
            continue
        sample_id = _sample_id(raw)
        if not isinstance(key, Mapping):
            raise PacketError(f"primitive_key_requires_mapping:{sample_id}")
        missing = [field for field in PRIMITIVE_KEY_FIELDS if key.get(field) is None]
        if missing:
            raise PacketError(f"primitive_key_missing:{','.join(missing)}")
        try:
            PrimitiveKey.from_mapping(key)
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            raise PacketError(f"primitive_key_invalid:{sample_id}:{error}") from error
        sample = {
            "key": deepcopy(dict(key)),
            "duration_ms": _positive(raw["duration_ms"], "duration_ms"),
            "run_id": _run_id(raw, sample_id), "repeat_index": _repeat_index(raw.get("repeat_index")),
            "source": _source(raw, sample_id), "split": "train",
        }
        if set(sample) != {"key", "duration_ms", "run_id", "repeat_index", "source", "split"}:
            raise PacketError(f"calibrate_top_level_keys_invalid:{sample_id}")
        repeat_key = (_canonical(sample["key"]), sample["run_id"], sample["repeat_index"])
        if repeat_key in seen and seen[repeat_key] != sample["duration_ms"]:
            raise PacketError(f"calibrate_repeat_conflict:{sample_id}")
        seen[repeat_key] = sample["duration_ms"]
        samples.append(sample)
    return sorted(samples, key=lambda s: (s["run_id"], s["repeat_index"], _canonical(s["key"])))


def _prediction_rows(predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for raw in predictions:
        if not isinstance(raw, Mapping):
            raise PacketError("accuracy_prediction_requires_mapping")
        point_id = str(raw.get("point_id") or raw.get("sample_id") or "")
        if not point_id:
            raise PacketError("accuracy_prediction_requires_point_id")
        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise PacketError(f"accuracy_prediction_requires_identity:{point_id}")
        norm_identity = {**deepcopy(dict(identity)), "layout": _normalize_layout(identity.get("layout"))}
        missing = [
            field for field in IDENTITY_FIELDS
            if not norm_identity.get(field) or str(norm_identity.get(field)).strip() == "declared"
        ]
        if missing:
            raise PacketError(f"accuracy_prediction_identity_missing:{point_id}:{','.join(missing)}")
        if norm_identity["level"] not in ACCURACY_LEVELS:
            raise PacketError(f"accuracy_prediction_level_invalid:{point_id}")
        row = {
            "point_id": point_id, "identity": norm_identity,
            "duration_ms": _positive(raw["duration_ms"], "prediction_duration_ms"),
        }
        if "occurrences" in raw:
            row["occurrences"] = _positive_int(raw["occurrences"], "occurrences")
        rows.append(row)
    return rows


def _accuracy_packet(receipts: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
                     training_ids: set[str]) -> dict[str, Any]:
    observations = []
    for raw in receipts:
        if raw["split"] != "heldout":
            continue
        sample_id = _sample_id(raw)
        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise PacketError(f"accuracy_observation_requires_full_identity:{sample_id}")
        identity = _accuracy_identity(raw, sample_id)
        sample = {
            "sample_id": sample_id, "identity": identity,
            "duration_ms": _positive(raw["duration_ms"], "observation_duration_ms"),
            "run_id": _run_id(raw, sample_id),
            "source": _source(raw, sample_id, fabric=raw.get("kind") == "fabric"),
            "split": "heldout",
        }
        if sample_id in training_ids:
            raise PacketError(f"accuracy_observation_reused_for_training:{sample_id}")
        observations.append(sample)
    rows = _prediction_rows(predictions)
    if len({row["point_id"] for row in rows}) != len(rows):
        raise PacketError("duplicate_accuracy_prediction_point")
    packet = {
        "training_sample_ids": sorted(training_ids),
        "predictions": sorted(rows, key=lambda row: row["point_id"]),
        "observations": sorted(observations, key=lambda row: row["sample_id"]),
    }
    assert_public_provenance(packet, "$.accuracy")
    return packet


def build_packets(receipts: Iterable[Mapping[str, Any]],
                  predictions: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Project one immutable receipt stream into the four CLI packet shapes."""
    raw_observations, receipt_count, rejected_unqualified = _observations(receipts)
    observations = _dedupe(raw_observations)
    training_ids = {_sample_id(raw) for raw in observations if raw["split"] == "train"}
    service, skipped_service = _service_packets(observations)
    rate, rate_conditions, skipped_rate = _rate_packets(observations)
    calibrate = _calibrate_samples(observations)
    accuracy = _accuracy_packet(observations, predictions, training_ids)
    for packet in service:
        assert_public_provenance(packet, "$.fit-service")
    for packet in rate:
        assert_public_provenance(packet, "$.fit-rate")
    for sample in calibrate:
        assert_public_provenance(sample, "$.calibrate")
    return {
        "fit-service": service, "fit-rate": rate, "calibrate": calibrate,
        "accuracy": accuracy, "rate_conditions": rate_conditions,
        "manifest": {
            "schema": SCHEMA, "receipts": receipt_count, "observations": len(observations),
            "rejected_unqualified_observations": rejected_unqualified,
            "fit-service_conditions": len(service), "fit-rate_conditions": len(rate),
            "calibrate_samples": len(calibrate),
            "skipped_single_point_service_groups": skipped_service,
            "skipped_single_point_rate_groups": skipped_rate,
            "training_sample_ids": len(accuracy["training_sample_ids"]),
            "heldout_observations": len(accuracy["observations"]),
            "predictions": len(accuracy["predictions"]),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_packets(bundle: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Write one condition per service/rate file and the three single packets."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files, rate_files = [], []
    for name, packets in (("fit-service", bundle["fit-service"]), ("fit-rate", bundle["fit-rate"])):
        for index, packet in enumerate(packets):
            filename = f"{name}.json" if len(packets) == 1 else f"{name}.{index:03d}.json"
            _write_json(out_dir / filename, packet)
            files.append(filename)
            if name == "fit-rate":
                rate_files.append(filename)
    with (out_dir / CALIBRATE_NAME).open("w", encoding="utf-8") as stream:
        for sample in bundle["calibrate"]:
            stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
    files.append(CALIBRATE_NAME)
    _write_json(out_dir / ACCURACY_NAME, bundle["accuracy"])
    files.append(ACCURACY_NAME)
    manifest = {
        **bundle["manifest"], "files": files,
        "rate_conditions": dict(zip(rate_files, bundle.get("rate_conditions", []))),
    }
    _write_json(out_dir / MANIFEST_NAME, manifest)
    files.append(MANIFEST_NAME)
    return manifest


def _predictions(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("predictions", value.get("points", []))
    if not isinstance(value, list):
        raise PacketError("predictions_must_be_a_list_or_contain_a_predictions_list")
    return [dict(item) for item in value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipts", nargs="+", required=True, help="One or more raw profiler/bench JSONL receipt files")
    parser.add_argument("--predictions", default=None, help="Frozen accuracy prediction JSON (list or predictions list)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        bundle = build_packets(iter_receipts(args.receipts), _predictions(args.predictions))
        manifest = write_packets(bundle, args.out)
    except (PacketError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
