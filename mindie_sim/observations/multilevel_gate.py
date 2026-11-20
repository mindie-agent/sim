"""Independent L1/L2/L3 multilevel accuracy gate and lineage enforcement.

[Role] Stream capture manifests, enforce transitive acquisition-lineage isolation
  across train/heldout and L1/L2/L3 boundaries, execute Engine-only evaluators
  with receipt checks, and produce strict sub-10% cohort summaries.
[Invariants] Lineage precedes scoring: each acquisition component has one split,
  fit training ancestors never intersect heldout ancestors, and strict 3-level
  blind acceptance requires disjoint L1/L2/L3 roots. Predictions require canonical
  Engine receipts with external_offset_ns == 0 and engine_sourced == True; L2
  never clones L3 APE or divides step by layers. Strict < threshold_pct gate
  (10.0% fails); empty cohorts emit insufficient_evidence; near-zero rung deltas
  report signed ns and two-arm errors; provenance rejects host paths and IPs.
[Timeouts & Locks] Streaming JSON/JSONL I/O (64 KiB chunks; never whole-reads
  files > 50 MiB); bounded in-memory state; zero locks or device contexts.
[Side-Effect Policy] Read-only over manifests, captures, and frozen fits.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import json, math, re, statistics
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from mindie_sim.schema.common import ModelingError

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None

DEFAULT_THRESHOLD_PCT = 10.0
TAIL_QUANTILE_P90 = 0.9
SHA256_HEX_LENGTH = 64
STREAM_WHOLE_READ_LIMIT_BYTES = 50 * 1024 * 1024
STREAM_CHUNK_BYTES = 64 * 1024
NEAR_ZERO_DELTA_MIN_RATIO = 0.05
DEFAULT_NEAR_ZERO_DELTA_NS = 500
PERCENT_SCALE = 100.0
CAPTURE_URI_SCHEME = "capture://"
VALID_LEVELS = frozenset({"L1", "L2", "L3"})
VALID_SPLITS = frozenset({"train", "heldout"})
SOURCE_REF_KEYS = frozenset({"repository", "revision", "relative_path", "source_sha256"})
MANIFEST_HEADER_KEYS = frozenset({
    "manifest_id", "schema_version", "lineage_edges",
    "source_uri", "source_sha256", "strict_three_level_blind",
})
TIME_UNIT_SCALE_NS: Mapping[str, int] = MappingProxyType({"ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000})
L1_BOUNDARY_KINDS = frozenset({"device_kernel", "native_composite", "hccl_api", "aclnn_call", "single_operator", "collective_call", "npu_event_call"})
L2_BOUNDARY_KINDS = frozenset({"fx_region", "dispatch_gmm_combine", "periodic_block", "paired_rung_delta", "subgraph_block", "npu_event_burst"})
L3_BOUNDARY_KINDS = frozenset({"model_step", "service_step", "e2e_step", "device_only_step", "host_device_step"})
FORBIDDEN_L2_BOUNDARY_KINDS = L3_BOUNDARY_KINDS | frozenset({"layer_step_quotient", "step_divided_by_layers", "l3_clone"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_UNIX_PATH_RE = re.compile(r"/(?:" + "|".join(("Users", "home", "mnt", "root")) + r")/")
_WINDOWS_DRIVE_RE = re.compile(r"(?:^|[\s\"'=:,])[A-Za-z]:\\")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CONTAINER_MARKERS = ("".join(("docker", ".io/")), "".join(("swr", ".cn-")), "".join(("@sha", "256:")))

Level = Literal["L1", "L2", "L3"]
SplitName = Literal["train", "heldout"]
ResidualStatus = Literal["ok", "missing_evidence", "invalid_evidence", "near_zero_delta", "debug_only", "counterfactual_excluded"]
CohortStatus = Literal["pass", "fail", "insufficient_evidence"]


@dataclass(frozen=True, slots=True)
class CaptureBoundary:
    raw_sha256: str
    acquisition_uid: str
    run_uid: str
    parent_digests: tuple[str, ...]
    event_ids: tuple[str, ...]
    device_ranks: tuple[int, ...]
    clock_domain: str
    start_event: str
    end_event: str
    boundary_kind: str
    unit: str
    capture_uri: str = ""
    source_ref: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True, slots=True)
class CohortKey:
    level: Level
    cohort_name: str
    sku: str
    phase: str
    mode: str
    shape_regime: str
    parallel_topology: str
    def as_id(self) -> str:
        return "|".join((self.level, self.cohort_name, self.sku, self.phase, self.mode, self.shape_regime, self.parallel_topology))
    def to_dict(self) -> dict[str, str]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class GateCase:
    case_id: str
    level: Level
    split: SplitName
    boundary: CaptureBoundary
    input_digest: str
    stack_digest: str
    implementation: str
    cohort: CohortKey
    measured_ns: int | None
    debug_cross_level: bool = False
    is_counterfactual: bool = False
    has_matched_capture: bool = True
    rung_arm_a_measured_ns: int | None = None
    rung_arm_b_measured_ns: int | None = None
    block_capture_present: bool = True
    regime_dims: tuple[tuple[str, str], ...] = ()
    payload: tuple[tuple[str, Any], ...] = ()

@dataclass(frozen=True, slots=True)
class EngineInterval:
    case_id: str
    level: Level
    boundary_kind: str
    start_event: str
    end_event: str
    start_ns: int
    end_ns: int
    predicted_ns: int
    engine_receipt: str
    engine_sourced: bool = True
    external_offset_ns: int = 0
    status: Literal["ok", "missing_evidence"] = "ok"
    missing_reason: str = ""
    source_level: str = ""
    derived_from_step_division: bool = False
    wire_bytes: int | None = None
    kernel_sequence: tuple[str, ...] = ()
    arm_a_predicted_ns: int | None = None
    arm_b_predicted_ns: int | None = None
    diagnostics: tuple[tuple[str, Any], ...] = ()

@dataclass(frozen=True, slots=True)
class Residual:
    case_id: str
    level: str
    cohort: CohortKey
    acquisition_root: str
    measured_ns: int
    predicted_ns: int
    engine_receipt: str
    signed_error: float
    absolute_percentage_error: float
    status: ResidualStatus = "ok"
    excluded_reason: str = ""
    boundary_kind: str = ""
    signed_delta_ns: int | None = None
    arm_a_signed_error: float | None = None
    arm_b_signed_error: float | None = None
    regime_dims: tuple[tuple[str, str], ...] = ()
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["signed_error"] = None if math.isnan(self.signed_error) else self.signed_error
        out["absolute_percentage_error"] = None if math.isnan(self.absolute_percentage_error) else self.absolute_percentage_error
        out["regime_dims"] = dict(self.regime_dims)
        return out

@dataclass(frozen=True, slots=True)
class FrozenFitParameter:
    parameter_id: str
    training_digests: tuple[str, ...]
    source_uri: str = ""

@dataclass(frozen=True, slots=True)
class FrozenFitManifest:
    fit_id: str
    schema_version: str
    parameters: tuple[FrozenFitParameter, ...]
    lineage_edges: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True, slots=True)
class Manifest:
    manifest_id: str
    schema_version: str
    cases: tuple[GateCase, ...]
    lineage_edges: tuple[tuple[str, str], ...] = ()
    source_uri: str = ""
    source_sha256: str = ""
    strict_three_level_blind: bool = True

@dataclass(frozen=True, slots=True)
class SplitIndex:
    case_roots: Mapping[str, str]
    component_splits: Mapping[str, str]
    train_components: frozenset[str]
    heldout_components: frozenset[str]
    heldout_roots_by_level: Mapping[str, frozenset[str]]
    debug_only_cases: frozenset[str]
    fit_ancestor_components: Mapping[str, frozenset[str]]
    def root_for(self, case_id: str) -> str:
        return self.case_roots[case_id]

@dataclass(frozen=True, slots=True)
class CohortSummary:
    cohort: CohortKey
    status: CohortStatus
    n_roots: int
    n_rows: int
    mape_pct: float | None
    medape_pct: float | None
    median_signed_pct: float | None
    p90_ape_pct: float | None
    max_ape_pct: float | None
    missing: int
    excluded_reasons: tuple[tuple[str, int], ...]
    threshold_pct: float = DEFAULT_THRESHOLD_PCT
    failed_metrics: tuple[str, ...] = ()
    tail_case_ids: tuple[str, ...] = ()
    @property
    def passed(self) -> bool:
        return self.status == "pass"
    @property
    def tail_exceeded(self) -> bool:
        return bool(self.tail_case_ids) or (self.p90_ape_pct is not None and self.p90_ape_pct >= self.threshold_pct)

@runtime_checkable
class EvaluationRunner(Protocol):
    def run_l1(self, case: GateCase) -> EngineInterval: ...
    def run_l2(self, case: GateCase) -> EngineInterval: ...
    def run_l3(self, case: GateCase) -> EngineInterval: ...


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
    def find(self, node: str) -> str:
        root = self._parent.setdefault(node, node)
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            node, self._parent[node] = self._parent[node], root
        return root
    def union(self, left: str, right: str) -> None:
        rl, rr = self.find(left), self.find(right)
        if rl != rr:
            self._parent[rr if rl < rr else rl] = rl if rl < rr else rr

def _req_str(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(reason)
    return value.strip()

def _req_sha256(value: Any, reason: str) -> str:
    text = _req_str(value, reason).lower()
    if len(text) != SHA256_HEX_LENGTH or not _SHA256_RE.fullmatch(text):
        raise ModelingError(f"{reason}:{text}")
    return text

def validate_provenance_text(text: str, *, field_name: str = "provenance") -> str:
    """Reject private host paths, IPv4 addresses, and container image digests."""
    if _PRIVATE_UNIX_PATH_RE.search(text) or _WINDOWS_DRIVE_RE.search(text):
        raise ModelingError(f"private_host_path_rejected:{field_name}")
    if _IPV4_RE.search(text):
        raise ModelingError(f"private_ip_rejected:{field_name}")
    if any(marker in text.lower() for marker in _CONTAINER_MARKERS):
        raise ModelingError(f"container_reference_rejected:{field_name}")
    return text

def sanitize_capture_uri(uri: str) -> str:
    cleaned = validate_provenance_text(_req_str(uri, "capture_uri_empty"), field_name="capture_uri")
    if not cleaned.startswith(CAPTURE_URI_SCHEME):
        raise ModelingError(f"capture_uri_scheme_invalid:{cleaned}")
    return cleaned

def sanitize_source_ref(ref: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    if not isinstance(ref, Mapping) or frozenset(ref.keys()) != SOURCE_REF_KEYS:
        raise ModelingError("source_ref_invalid")
    repo = validate_provenance_text(_req_str(ref["repository"], "source_ref_repository_empty"), field_name="source_ref.repository")
    rev = validate_provenance_text(_req_str(ref["revision"], "source_ref_revision_empty"), field_name="source_ref.revision")
    rel = validate_provenance_text(_req_str(ref["relative_path"], "source_ref_path_empty"), field_name="source_ref.relative_path")
    if rel.startswith("/") or ".." in PurePosixPath(rel).parts:
        raise ModelingError(f"source_ref_relative_path_unsafe:{rel}")
    return (("relative_path", rel), ("repository", repo), ("revision", rev),
            ("source_sha256", _req_sha256(ref["source_sha256"], "source_ref_sha256_invalid")))

def _parse_ranks(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ModelingError("device_ranks_must_be_nonempty_sequence")
    ranks = [item for item in raw if not isinstance(item, bool) and isinstance(item, int) and item >= 0]
    if len(ranks) != len(raw) or len(set(ranks)) != len(ranks):
        raise ModelingError(f"device_rank_invalid:{raw!r}")
    return tuple(ranks)

def _parse_str_tuple(raw: Any, reason: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ModelingError(reason)
    return tuple(validate_provenance_text(_req_str(x, reason), field_name=reason) for x in raw)

def _to_ns(value: Any, unit: str, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ModelingError(reason)
    return int(round(float(value) * TIME_UNIT_SCALE_NS[unit]))

def _fstr(mapping: Mapping[str, Any], key: str, cid: str, *, check_prov: bool = False) -> str:
    val = _req_str(mapping.get(key), f"{key}_missing:{cid}")
    return validate_provenance_text(val, field_name=key) if check_prov else val


def parse_gate_case(raw: Mapping[str, Any], *, allow_invalid_durations: bool = False) -> GateCase:
    """Validate and convert one raw manifest record into a ``GateCase``."""
    if not isinstance(raw, Mapping):
        raise ModelingError("gate_case_must_be_mapping")
    cid = validate_provenance_text(_req_str(raw.get("case_id"), "case_id_missing"), field_name="case_id")
    level, split = _req_str(raw.get("level"), "level_missing"), _req_str(raw.get("split"), "split_missing")
    b_raw, c_raw = raw.get("boundary"), raw.get("cohort")
    if level not in VALID_LEVELS or split not in VALID_SPLITS:
        raise ModelingError(f"case_level_or_split_invalid:{cid}:{level}:{split}")
    if not isinstance(b_raw, Mapping) or not isinstance(c_raw, Mapping):
        raise ModelingError(f"boundary_or_cohort_missing:{cid}")
    unit = _fstr(b_raw, "unit", cid).lower()
    if unit not in TIME_UNIT_SCALE_NS:
        raise ModelingError(f"unit_unsupported:{unit}")
    boundary = CaptureBoundary(
        raw_sha256=_req_sha256(b_raw.get("raw_sha256"), f"raw_sha256_invalid:{cid}"),
        acquisition_uid=_fstr(b_raw, "acquisition_uid", cid, check_prov=True),
        run_uid=_fstr(b_raw, "run_uid", cid, check_prov=True),
        parent_digests=_parse_str_tuple(b_raw.get("parent_digests"), f"parent_digests_invalid:{cid}"),
        event_ids=_parse_str_tuple(b_raw.get("event_ids"), f"event_ids_invalid:{cid}"),
        device_ranks=_parse_ranks(b_raw.get("device_ranks")),
        clock_domain=_fstr(b_raw, "clock_domain", cid, check_prov=True),
        start_event=_fstr(b_raw, "start_event", cid), end_event=_fstr(b_raw, "end_event", cid),
        boundary_kind=_fstr(b_raw, "boundary_kind", cid), unit=unit,
        capture_uri=sanitize_capture_uri(str(b_raw["capture_uri"])) if b_raw.get("capture_uri") else "",
        source_ref=sanitize_source_ref(b_raw["source_ref"]) if b_raw.get("source_ref") is not None else (),
    )
    cohort = CohortKey(
        level=level, cohort_name=_fstr(c_raw, "cohort_name", cid), sku=_fstr(c_raw, "sku", cid),  # type: ignore[arg-type]
        phase=_fstr(c_raw, "phase", cid), mode=_fstr(c_raw, "mode", cid),
        shape_regime=_fstr(c_raw, "shape_regime", cid), parallel_topology=_fstr(c_raw, "parallel_topology", cid),
    )
    arm_a = _to_ns(raw["rung_arm_a_measured_ns"], "ns", f"rung_arm_a_invalid:{cid}") if raw.get("rung_arm_a_measured_ns") is not None else None
    arm_b = _to_ns(raw["rung_arm_b_measured_ns"], "ns", f"rung_arm_b_invalid:{cid}") if raw.get("rung_arm_b_measured_ns") is not None else None
    measured_ns = (
        _to_ns(raw["measured_ns"], "ns", f"measured_ns_invalid:{cid}") if raw.get("measured_ns") is not None
        else (_to_ns(raw["measured_duration"], unit, f"measured_duration_invalid:{cid}") if raw.get("measured_duration") is not None else None)
    )
    if not allow_invalid_durations:
        if boundary.boundary_kind == "paired_rung_delta":
            if arm_a is None or arm_b is None or arm_a <= 0 or arm_b <= 0:
                raise ModelingError(f"rung_arms_must_be_positive:{cid}")
            measured_ns = (arm_b - arm_a) if measured_ns is None else measured_ns
        elif measured_ns is None or measured_ns <= 0:
            raise ModelingError(f"measured_duration_must_be_positive:{cid}")
    r_raw, p_raw = raw.get("regime_dims") or {}, raw.get("payload") or {}
    if not isinstance(r_raw, Mapping) or not isinstance(p_raw, Mapping):
        raise ModelingError(f"case_metadata_invalid:{cid}")
    return GateCase(
        case_id=cid, level=level, split=split, boundary=boundary,  # type: ignore[arg-type]
        input_digest=_fstr(raw, "input_digest", cid), stack_digest=_fstr(raw, "stack_digest", cid),
        implementation=_fstr(raw, "implementation", cid), cohort=cohort, measured_ns=measured_ns,
        debug_cross_level=bool(raw.get("debug_cross_level", False)),
        is_counterfactual=bool(raw.get("is_counterfactual", False)),
        has_matched_capture=bool(raw.get("has_matched_capture", True)),
        rung_arm_a_measured_ns=arm_a, rung_arm_b_measured_ns=arm_b,
        block_capture_present=bool(raw.get("block_capture_present", True)),
        regime_dims=tuple(sorted((str(k), validate_provenance_text(str(v), field_name=f"regime_dims.{k}")) for k, v in r_raw.items())),
        payload=tuple(sorted((str(k), validate_provenance_text(v, field_name=f"payload.{k}") if isinstance(v, str) else v) for k, v in p_raw.items())),
    )


def stream_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSON-Lines records line by line without whole-file buffering."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ModelingError(f"jsonl_parse_error:{target.name}:line={lineno}:{exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ModelingError(f"jsonl_record_not_object:{target.name}:line={lineno}")
            yield obj


def _iter_json_cases_streaming(path: Path) -> Iterator[dict[str, Any]]:
    if ijson is not None:
        with path.open("rb") as handle:
            for item in ijson.items(handle, "cases.item", use_float=True):
                if not isinstance(item, dict):
                    raise ModelingError(f"manifest_case_not_object:{path.name}")
                yield item
        return
    decoder, marker, buf, in_array = json.JSONDecoder(), '"cases"', "", False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(STREAM_CHUNK_BYTES)
            buf += chunk
            if not in_array:
                idx = buf.find(marker)
                bracket = buf.find("[", idx + len(marker)) if idx >= 0 else -1
                if bracket < 0:
                    if not chunk:
                        return
                    buf = buf[idx:] if idx >= 0 else buf[-len(marker):]
                    continue
                buf, in_array = buf[bracket + 1:], True
            pos = 0
            while pos < len(buf):
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos >= len(buf) or buf[pos] == "]":
                    if pos < len(buf):
                        return
                    buf = ""
                    break
                try:
                    item, pos = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    buf = buf[pos:]
                    break
                if not isinstance(item, dict):
                    raise ModelingError(f"manifest_case_not_object:{path.name}")
                yield item
            else:
                buf = ""
            if not chunk:
                if buf.strip() and buf.strip() != "]":
                    raise ModelingError(f"manifest_json_truncated:{path.name}")
                return

def _extract_json_header_streaming(path: Path) -> dict[str, Any]:
    if ijson is not None:
        from mindie_sim.observations.store import json_top_fields
        return json_top_fields(path, MANIFEST_HEADER_KEYS)
    decoder, header = json.JSONDecoder(), {}
    with path.open("r", encoding="utf-8") as handle:
        head = handle.read(STREAM_CHUNK_BYTES)
    for key in MANIFEST_HEADER_KEYS:
        pos = head.find(f'"{key}"')
        colon = head.find(":", pos + len(key) + 2) if pos >= 0 else -1
        if colon >= 0:
            idx = colon + 1
            while idx < len(head) and head[idx] in " \t\r\n":
                idx += 1
            try:
                header[key], _ = decoder.raw_decode(head, idx)
            except json.JSONDecodeError:
                pass
    return header

def _parse_lineage_edges(raw: Any) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ModelingError("lineage_edges_invalid")
    edges = []
    for item in raw:
        pair = (item.get("parent"), item.get("child")) if isinstance(item, Mapping) else (
            (item[0], item[1]) if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2 else None
        )
        if pair is None:
            raise ModelingError("lineage_edge_entry_invalid")
        edges.append((_req_str(pair[0], "lineage_edge_parent_missing"), _req_str(pair[1], "lineage_edge_child_missing")))
    return tuple(edges)


def stream_manifest(
    path: str | Path, *, allow_invalid_durations: bool = False,
    max_whole_read_bytes: int = STREAM_WHOLE_READ_LIMIT_BYTES,
) -> Manifest:
    """Stream and validate a JSON or JSONL gate manifest."""
    target = Path(path)
    extra_edges: list[tuple[str, str]] = []
    if target.suffix.lower() in {".jsonl", ".ndjson"}:
        header, cases = {}, []
        for row in stream_jsonl(target):
            rtype = row.get("record_type", "case")
            if rtype == "manifest_header":
                header.update({k: v for k, v in row.items() if k in MANIFEST_HEADER_KEYS})
            elif rtype == "lineage_edge":
                extra_edges.extend(_parse_lineage_edges([row]))
            else:
                cases.append(parse_gate_case(row, allow_invalid_durations=allow_invalid_durations))
    else:
        if target.stat().st_size <= max_whole_read_bytes:
            with target.open("r", encoding="utf-8") as handle:
                doc = json.load(handle)
            if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list):
                raise ModelingError(f"manifest_schema_invalid:{target.name}")
            header, raw_cases = {k: v for k, v in doc.items() if k != "cases"}, doc["cases"]
        else:
            header, raw_cases = _extract_json_header_streaming(target), _iter_json_cases_streaming(target)
        cases = [parse_gate_case(x, allow_invalid_durations=allow_invalid_durations) for x in raw_cases]
    return Manifest(
        manifest_id=_req_str(header.get("manifest_id", target.stem), "manifest_id_invalid"),
        schema_version=_req_str(header.get("schema_version", "mindie-sim.multilevel-manifest/1"), "schema_version_invalid"),
        cases=tuple(cases), lineage_edges=(*_parse_lineage_edges(header.get("lineage_edges")), *extra_edges),
        source_uri=sanitize_capture_uri(str(header["source_uri"])) if header.get("source_uri") else "",
        source_sha256=_req_sha256(header["source_sha256"], "manifest_sha256_invalid") if header.get("source_sha256") else "",
        strict_three_level_blind=bool(header.get("strict_three_level_blind", True)),
    )


def validate_lineage(manifest: Manifest, fits: FrozenFitManifest) -> SplitIndex:
    """Compute transitive acquisition lineage closure and enforce split isolation."""
    uf, seen_cases, case_anchor, acq_uids = _UnionFind(), set(), {}, set()
    for case in manifest.cases:
        if case.case_id in seen_cases:
            raise ModelingError(f"duplicate_case_id:{case.case_id}")
        seen_cases.add(case.case_id)
        b = case.boundary
        tokens = [_req_str(b.raw_sha256, f"raw_sha256_missing:{case.case_id}"),
                  _req_str(b.acquisition_uid, f"acquisition_uid_missing:{case.case_id}"),
                  _req_str(b.run_uid, f"run_uid_missing:{case.case_id}"),
                  *(t.strip() for t in (*b.parent_digests, *b.event_ids) if t and t.strip())]
        uf.find(tokens[0])
        for tok in tokens[1:]:
            uf.union(tokens[0], tok)
        case_anchor[case.case_id] = tokens[0]
        acq_uids.add(b.acquisition_uid.strip())
    for parent, child in (*manifest.lineage_edges, *fits.lineage_edges):
        uf.union(parent.strip(), child.strip())
    for param in fits.parameters:
        if param.source_uri:
            sanitize_capture_uri(param.source_uri)
        for d in param.training_digests:
            if d and d.strip():
                uf.find(d.strip())
    rep_to_acq: dict[str, list[str]] = {}
    for uid in acq_uids:
        rep_to_acq.setdefault(uf.find(uid), []).append(uid)
    canonical_root = lambda tok: min(rep_to_acq[rep]) if (rep := uf.find(tok)) in rep_to_acq else rep

    case_roots, comp_splits, comp_cases = {}, {}, {}
    train_comps, heldout_comps, debug_only = set(), set(), set()
    heldout_levels_by_root: dict[str, set[str]] = {}
    heldout_roots_by_level: dict[str, set[str]] = {"L1": set(), "L2": set(), "L3": set()}
    for case in manifest.cases:
        root = canonical_root(case_anchor[case.case_id])
        case_roots[case.case_id] = root
        comp_cases.setdefault(root, []).append(case.case_id)
        if root in comp_splits and comp_splits[root] != case.split:
            raise ModelingError(f"lineage_split_leakage:root={root}:cases={','.join(sorted(comp_cases[root]))}")
        comp_splits[root] = case.split
        if case.split == "train":
            train_comps.add(root)
        else:
            heldout_comps.add(root)
            if case.debug_cross_level:
                debug_only.add(case.case_id)
            else:
                heldout_levels_by_root.setdefault(root, set()).add(case.level)
                heldout_roots_by_level[case.level].add(root)
    fit_ancestors: dict[str, frozenset[str]] = {}
    for param in fits.parameters:
        roots = frozenset(canonical_root(d.strip()) for d in param.training_digests if d and d.strip())
        if roots & heldout_comps:
            raise ModelingError(f"lineage_fit_heldout_leakage:parameter={param.parameter_id}:roots={','.join(sorted(roots & heldout_comps))}")
        fit_ancestors[param.parameter_id] = roots
    if manifest.strict_three_level_blind:
        for root, levels in sorted(heldout_levels_by_root.items()):
            if len(levels) > 1:
                raise ModelingError(f"lineage_cross_level_root_overlap:root={root}:levels={','.join(sorted(levels))}")
    return SplitIndex(
        case_roots=MappingProxyType(case_roots), component_splits=MappingProxyType(comp_splits),
        train_components=frozenset(train_comps), heldout_components=frozenset(heldout_comps),
        heldout_roots_by_level=MappingProxyType({k: frozenset(v) for k, v in heldout_roots_by_level.items()}),
        debug_only_cases=frozenset(debug_only), fit_ancestor_components=MappingProxyType(fit_ancestors),
    )


def _validate_interval(case: GateCase, interval: EngineInterval, *, expected_level: Level) -> EngineInterval:
    if interval.level != expected_level:
        raise ModelingError(f"engine_interval_level_mismatch:{case.case_id}:{interval.level}")
    if interval.status == "missing_evidence":
        if not interval.missing_reason.strip():
            raise ModelingError(f"missing_evidence_requires_reason:{case.case_id}")
        return interval
    if not interval.engine_sourced or interval.external_offset_ns != 0:
        raise ModelingError(f"non_engine_or_offset_rejected:{case.case_id}:offset_ns={interval.external_offset_ns}")
    validate_provenance_text(_req_str(interval.engine_receipt, f"engine_receipt_missing:{case.case_id}"), field_name="engine_receipt")
    if (interval.boundary_kind != case.boundary.boundary_kind
            or interval.start_event != case.boundary.start_event or interval.end_event != case.boundary.end_event):
        raise ModelingError(f"boundary_mismatch:{case.case_id}")
    if case.boundary.boundary_kind == "paired_rung_delta":
        if (interval.arm_a_predicted_ns is None or interval.arm_b_predicted_ns is None
                or interval.arm_a_predicted_ns <= 0 or interval.arm_b_predicted_ns <= 0
                or interval.predicted_ns != interval.arm_b_predicted_ns - interval.arm_a_predicted_ns):
            raise ModelingError(f"rung_delta_prediction_invalid:{case.case_id}")
    elif interval.end_ns - interval.start_ns != interval.predicted_ns or interval.predicted_ns <= 0:
        raise ModelingError(f"interval_span_or_duration_invalid:{case.case_id}")
    return interval

def run_l1(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-1 operator/collective case on the Engine."""
    if case.level != "L1" or case.boundary.boundary_kind not in L1_BOUNDARY_KINDS:
        raise ModelingError(f"run_l1_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return _validate_interval(case, runner.run_l1(case), expected_level="L1")

def run_l2(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-2 subgraph/block/rung case on the Engine."""
    if case.level != "L2":
        raise ModelingError(f"run_l2_level_mismatch:{case.case_id}:{case.level}")
    if case.boundary.boundary_kind in FORBIDDEN_L2_BOUNDARY_KINDS:
        raise ModelingError(f"l2_derived_from_l3_forbidden:{case.case_id}:{case.boundary.boundary_kind}")
    if not case.block_capture_present or case.boundary.boundary_kind == "missing_block_capture":
        return EngineInterval(case.case_id, "L2", case.boundary.boundary_kind, case.boundary.start_event,
                              case.boundary.end_event, 0, 0, 0, "", status="missing_evidence", missing_reason="block_capture_absent")
    if case.boundary.boundary_kind not in L2_BOUNDARY_KINDS:
        raise ModelingError(f"run_l2_boundary_invalid:{case.case_id}:{case.boundary.boundary_kind}")
    interval = runner.run_l2(case)
    if interval.derived_from_step_division or interval.source_level == "L3":
        raise ModelingError(f"l2_cloned_or_divided_from_l3_forbidden:{case.case_id}")
    return _validate_interval(case, interval, expected_level="L2")

def run_l3(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-3 model/service step case on the Engine."""
    if case.level != "L3" or case.boundary.boundary_kind not in L3_BOUNDARY_KINDS:
        raise ModelingError(f"run_l3_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return _validate_interval(case, runner.run_l3(case), expected_level="L3")

def _make_residual(
    case: GateCase, root: str, *, measured_ns: int, predicted_ns: int = 0,
    receipt: str = "", signed_error: float = math.nan, status: ResidualStatus = "ok",
    excluded_reason: str = "", signed_delta_ns: int | None = None,
    arm_a_err: float | None = None, arm_b_err: float | None = None,
) -> Residual:
    return Residual(
        case.case_id, case.level, case.cohort, root, measured_ns, predicted_ns, receipt,
        signed_error, math.nan if math.isnan(signed_error) else abs(signed_error),
        status, excluded_reason, case.boundary.boundary_kind, signed_delta_ns, arm_a_err, arm_b_err, case.regime_dims,
    )

def evaluate(
    manifest: Manifest, fits: FrozenFitManifest, runner: EvaluationRunner, *,
    target_split: Literal["heldout", "train", "all"] = "heldout",
    near_zero_delta_ns: int = DEFAULT_NEAR_ZERO_DELTA_NS,
) -> Iterator[Residual]:
    """Validate lineage first, then stream exact per-case ``Residual`` records."""
    split_index = validate_lineage(manifest, fits)
    receipt_levels: dict[str, set[str]] = {}
    dispatch = {"L1": run_l1, "L2": run_l2, "L3": run_l3}
    for case in manifest.cases:
        if target_split != "all" and case.split != target_split:
            continue
        root, is_debug = split_index.root_for(case.case_id), case.case_id in split_index.debug_only_cases
        if case.is_counterfactual and not case.has_matched_capture:
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="counterfactual_excluded", excluded_reason="counterfactual_without_matched_capture")
            continue
        is_rung = case.boundary.boundary_kind == "paired_rung_delta"
        arm_a, arm_b = case.rung_arm_a_measured_ns, case.rung_arm_b_measured_ns
        if is_rung and (arm_a is None or arm_b is None or arm_a <= 0 or arm_b <= 0):
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="invalid_evidence", excluded_reason="invalid_rung_arm_duration")
            continue
        if not is_rung and (case.measured_ns is None or case.measured_ns <= 0):
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="invalid_evidence", excluded_reason="zero_or_missing_measured_duration")
            continue
        measured_ns = (case.measured_ns if case.measured_ns is not None else (arm_b - arm_a)) if is_rung else case.measured_ns  # type: ignore[operator]
        assert measured_ns is not None
        interval = dispatch[case.level](case, runner)
        if interval.status == "missing_evidence":
            yield _make_residual(case, root, measured_ns=measured_ns, status="missing_evidence", excluded_reason=interval.missing_reason)
            continue
        seen = receipt_levels.setdefault(interval.engine_receipt, set())
        seen.add(case.level)
        if len(seen) > 1:
            raise ModelingError(f"cross_level_engine_receipt_clone:{interval.engine_receipt}")
        if is_rung:
            assert arm_a is not None and arm_b is not None and interval.arm_a_predicted_ns is not None and interval.arm_b_predicted_ns is not None
            arm_a_err, arm_b_err = (interval.arm_a_predicted_ns - arm_a) / float(arm_a), (interval.arm_b_predicted_ns - arm_b) / float(arm_b)
            delta_ns = interval.predicted_ns - measured_ns
            near_zero = measured_ns == 0 or abs(measured_ns) <= max(near_zero_delta_ns, int(max(abs(arm_a), abs(arm_b)) * NEAR_ZERO_DELTA_MIN_RATIO))
            status: ResidualStatus = "debug_only" if is_debug else ("near_zero_delta" if near_zero else "ok")
            reason = "debug_cross_level_not_blind_acceptance" if is_debug else ("near_zero_rung_delta" if near_zero else "")
            yield _make_residual(case, root, measured_ns=measured_ns, predicted_ns=interval.predicted_ns, receipt=interval.engine_receipt,
                                 signed_error=math.nan if (is_debug or near_zero) else delta_ns / float(measured_ns),
                                 status=status, excluded_reason=reason, signed_delta_ns=delta_ns, arm_a_err=arm_a_err, arm_b_err=arm_b_err)
        else:
            yield _make_residual(case, root, measured_ns=measured_ns, predicted_ns=interval.predicted_ns, receipt=interval.engine_receipt,
                                 signed_error=(interval.predicted_ns - measured_ns) / float(measured_ns),
                                 status="debug_only" if is_debug else "ok",
                                 excluded_reason="debug_cross_level_not_blind_acceptance" if is_debug else "")

def summarize_cohorts(
    residuals: Iterable[Residual], *, threshold_pct: float = DEFAULT_THRESHOLD_PCT
) -> dict[CohortKey, CohortSummary]:
    """Summarize residuals per frozen cohort enforcing strict ``< threshold_pct``."""
    grouped: dict[CohortKey, list[Residual]] = {}
    for item in residuals:
        grouped.setdefault(item.cohort, []).append(item)
    summaries: dict[CohortKey, CohortSummary] = {}
    for cohort_key, rows in grouped.items():
        valid, excluded, missing = [], Counter(), 0
        for row in rows:
            if row.status == "ok" and math.isfinite(row.absolute_percentage_error):
                valid.append(row)
            else:
                missing += int(row.status == "missing_evidence")
                excluded[row.excluded_reason or row.status] += 1
        excluded_tuple = tuple(sorted(excluded.items()))
        if not valid:
            summaries[cohort_key] = CohortSummary(
                cohort_key, "insufficient_evidence", 0, 0, None, None, None, None, None,
                missing, excluded_tuple, threshold_pct, ("insufficient_evidence",),
            )
            continue
        apes_pct = sorted(r.absolute_percentage_error * PERCENT_SCALE for r in valid)
        signed_pct = [r.signed_error * PERCENT_SCALE for r in valid]
        mape_pct, medape_pct = float(statistics.fmean(apes_pct)), float(statistics.median(apes_pct))
        p90_idx = min(len(apes_pct) - 1, max(0, math.ceil(TAIL_QUANTILE_P90 * len(apes_pct)) - 1))
        tail_ids = tuple(r.case_id for r in valid if (r.absolute_percentage_error * PERCENT_SCALE) >= threshold_pct)
        failed = tuple(k for k, v in (("mape_pct", mape_pct), ("medape_pct", medape_pct)) if v >= threshold_pct)
        summaries[cohort_key] = CohortSummary(
            cohort_key, "fail" if failed else "pass", len({r.acquisition_root for r in valid}), len(valid),
            mape_pct, medape_pct, float(statistics.median(signed_pct)), float(apes_pct[p90_idx]), float(apes_pct[-1]),
            missing, excluded_tuple, threshold_pct, failed, tail_ids,
        )
    return summaries

def gate_decision(summaries: Mapping[CohortKey, CohortSummary]) -> dict[str, Any]:
    """Compute overall gate verdict without allowing global averages to hide failures."""
    if not summaries:
        return {"status": "insufficient_evidence", "cohort_count": 0, "failed_cohorts": [], "insufficient_cohorts": [], "tail_warning_cohorts": []}
    failed = sorted(k.as_id() for k, v in summaries.items() if v.status == "fail")
    insufficient = sorted(k.as_id() for k, v in summaries.items() if v.status == "insufficient_evidence")
    tail_warnings = sorted(k.as_id() for k, v in summaries.items() if v.tail_exceeded)
    status = "fail" if failed else ("insufficient_evidence" if insufficient else "pass")
    return {"status": status, "cohort_count": len(summaries), "failed_cohorts": failed, "insufficient_cohorts": insufficient, "tail_warning_cohorts": tail_warnings}


__all__ = [
    "CaptureBoundary", "CohortKey", "CohortSummary", "DEFAULT_NEAR_ZERO_DELTA_NS",
    "DEFAULT_THRESHOLD_PCT", "EngineInterval", "EvaluationRunner", "FrozenFitManifest",
    "FrozenFitParameter", "GateCase", "Manifest", "Residual", "SplitIndex", "evaluate",
    "gate_decision", "parse_gate_case", "run_l1", "run_l2", "run_l3",
    "sanitize_capture_uri", "sanitize_source_ref", "stream_jsonl", "stream_manifest",
    "summarize_cohorts", "validate_lineage", "validate_provenance_text",
]
