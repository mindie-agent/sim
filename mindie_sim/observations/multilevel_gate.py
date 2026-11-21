"""Independent L1/L2/L3 multilevel accuracy gate and lineage enforcement.

[Role] Verify streamed captures and Engine receipts, enforce lineage isolation,
  and score independent L1/L2/L3 release evidence.
[Invariants] Normalize digests and scope events by acquisition root and session.
  Reject capture mismatches and require HMAC-bound Engine receipts with zero offset.
  Release requires independent coverage, all planned cases and pointwise tolerance.
[Timeouts & Locks] Two-pass 64 KiB streaming and compact tuples. No locks or devices.
[Side-Effect Policy] Read-only over manifests, captures, and frozen fits.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
import math
from pathlib import Path, PurePosixPath
import statistics
from types import MappingProxyType
from typing import Any, Literal, Protocol, overload, runtime_checkable

from mindie_sim.observations.multilevel_drilldown import score_vertical_drilldown
from mindie_sim.observations.multilevel_stream import (
    CAPTURE_URI_SCHEME, MS_TO_NS, SHA256_HEX_LENGTH, SOURCE_REF_KEYS, STREAM_CHUNK_BYTES,
    STREAM_WHOLE_READ_LIMIT_BYTES, TIME_UNIT_SCALE_NS, VALID_CLOCK_OPERATORS,
    EngineInterval, EngineRunReceipt, _req_sha256, _req_str, _validate_clock_operator,
    bind_engine_interval, bind_engine_rung_interval, digest_file_streaming, iter_manifest_json_cases,
    normalize_lineage_token, resolve_capture_file, sanitize_capture_uri, sanitize_source_ref,
    scan_manifest_json_pass1, stream_jsonl, validate_engine_interval, validate_provenance_text,
    verify_capture_file_consistency,
)
from mindie_sim.schema.common import ModelingError

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None

DEFAULT_THRESHOLD_PCT = 10.0
DEFAULT_ATOL_NS = 2000
TAIL_QUANTILE_P90 = 0.9
NEAR_ZERO_DELTA_MIN_RATIO = 0.05
DEFAULT_NEAR_ZERO_DELTA_NS = 500
PERCENT_SCALE = 100.0
REQUIRED_LEVELS: tuple[Literal["L1", "L2", "L3"], ...] = ("L1", "L2", "L3")
VALID_LEVELS = frozenset(REQUIRED_LEVELS)
VALID_SPLITS = frozenset({"train", "heldout"})
_CALLER_PREDICTION_KEYS = frozenset({"predicted_ns", "predicted_ms", "predicted_us", "predicted_duration", "predicted_s"})
MANIFEST_HEADER_KEYS = frozenset(
    {"manifest_id", "schema_version", "lineage_edges", "source_uri", "source_sha256", "strict_three_level_blind"}
)
L1_BOUNDARY_KINDS = frozenset({"device_kernel", "native_composite", "hccl_api", "aclnn_call",
                               "single_operator", "collective_call", "npu_event_call"})
L2_BOUNDARY_KINDS = frozenset({"fx_region", "dispatch_gmm_combine", "periodic_block",
                               "paired_rung_delta", "subgraph_block", "npu_event_burst"})
L3_BOUNDARY_KINDS = frozenset({"model_step", "service_step", "e2e_step", "device_only_step", "host_device_step"})
FORBIDDEN_L2_BOUNDARY_KINDS = L3_BOUNDARY_KINDS | {"layer_step_quotient", "step_divided_by_layers", "l3_clone"}

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
    session_id: str = ""
    clock_operator: Literal["elapsed_span", "busy_union"] = "elapsed_span"


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
    clock_operator: str = "elapsed_span"
    elapsed_span_ns: int | None = None
    busy_union_ns: int | None = None
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
    cases: Sequence[GateCase]
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
    acquisition_roots: tuple[str, ...] = ()
    evaluated_ok_case_ids: tuple[str, ...] = ()
    atol_ns: float = 0.0
    pointwise_failed_count: int = 0
    @property
    def passed(self) -> bool:
        return self.status == "pass"
    @property
    def tail_exceeded(self) -> bool:
        if self.atol_ns > 0.0:
            return self.pointwise_failed_count > 0
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


class _LazyManifestCases(Sequence[GateCase]):
    """Streaming disk-backed ``GateCase`` sequence that avoids buffering cases."""

    __slots__ = ("_path", "_is_jsonl", "_allow_invalid_durations", "_count")

    def __init__(self, path: Path, *, is_jsonl: bool, allow_invalid_durations: bool, count: int) -> None:
        self._path, self._is_jsonl = path, is_jsonl
        self._allow_invalid_durations, self._count = allow_invalid_durations, count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[GateCase]:
        if self._is_jsonl:
            for row in stream_jsonl(self._path):
                if row.get("record_type", "case") not in {"manifest_header", "lineage_edge"}:
                    yield parse_gate_case(row, allow_invalid_durations=self._allow_invalid_durations)
            return
        for row in iter_manifest_json_cases(self._path, ijson_mod=ijson):
            yield parse_gate_case(row, allow_invalid_durations=self._allow_invalid_durations)

    @overload
    def __getitem__(self, index: int) -> GateCase: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[GateCase, ...]: ...

    def __getitem__(self, index: int | slice) -> GateCase | tuple[GateCase, ...]:
        if isinstance(index, slice):
            indices = range(*index.indices(self._count))
            wanted, collected, max_idx = set(indices), {}, max(indices, default=-1)
            for pos, item in enumerate(self):
                if pos in wanted:
                    collected[pos] = item
                if pos >= max_idx:
                    break
            return tuple(collected[i] for i in indices)
        target = index + self._count if index < 0 else index
        if target < 0 or target >= self._count:
            raise IndexError(index)
        for pos, item in enumerate(self):
            if pos == target:
                return item
        raise IndexError(index)


def _parse_ranks(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ModelingError("device_ranks_must_be_nonempty_sequence")
    ranks = [item for item in raw if not isinstance(item, bool) and isinstance(item, int) and item >= 0]
    if len(ranks) != len(raw) or len(set(ranks)) != len(ranks):
        raise ModelingError(f"device_rank_invalid:{raw!r}")
    return tuple(ranks)


def _parse_str_tuple(raw: Any, reason: str, *, normalize_digests: bool = False) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ModelingError(reason)
    items = [validate_provenance_text(_req_str(x, reason), field_name=reason) for x in raw]
    return tuple(normalize_lineage_token(x) if normalize_digests else x for x in items)


def _to_ns(value: Any, unit: str, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ModelingError(reason)
    return int(round(float(value) * TIME_UNIT_SCALE_NS[unit]))


def _fstr(mapping: Mapping[str, Any], key: str, cid: str, *, check_prov: bool = False) -> str:
    val = _req_str(mapping.get(key), f"{key}_missing:{cid}")
    return validate_provenance_text(val, field_name=key) if check_prov else val


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
        edges.append((
            normalize_lineage_token(_req_str(pair[0], "lineage_edge_parent_missing")),
            normalize_lineage_token(_req_str(pair[1], "lineage_edge_child_missing")),
        ))
    return tuple(edges)


def parse_gate_case(raw: Mapping[str, Any], *, allow_invalid_durations: bool = False) -> GateCase:
    """Validate and convert one raw manifest record into a ``GateCase``."""
    if not isinstance(raw, Mapping):
        raise ModelingError("gate_case_must_be_mapping")
    cid = validate_provenance_text(_req_str(raw.get("case_id"), "case_id_missing"), field_name="case_id")
    p_raw = raw.get("payload") or {}
    if not _CALLER_PREDICTION_KEYS.isdisjoint(raw) or (
        isinstance(p_raw, Mapping) and not _CALLER_PREDICTION_KEYS.isdisjoint(p_raw)
    ):
        raise ModelingError(f"caller_predicted_duration_forbidden:{cid}")
    level, split = _req_str(raw.get("level"), "level_missing"), _req_str(raw.get("split"), "split_missing")
    b_raw, c_raw = raw.get("boundary"), raw.get("cohort")
    if level not in VALID_LEVELS or split not in VALID_SPLITS:
        raise ModelingError(f"case_level_or_split_invalid:{cid}:{level}:{split}")
    if not isinstance(b_raw, Mapping) or not isinstance(c_raw, Mapping):
        raise ModelingError(f"boundary_or_cohort_missing:{cid}")
    unit = _fstr(b_raw, "unit", cid).lower()
    if unit not in TIME_UNIT_SCALE_NS:
        raise ModelingError(f"unit_unsupported:{unit}")
    sess = validate_provenance_text(_req_str(b_raw["session_id"], f"session_id_invalid:{cid}"), field_name="session_id") if b_raw.get("session_id") else ""
    boundary = CaptureBoundary(
        raw_sha256=_req_sha256(b_raw.get("raw_sha256"), f"raw_sha256_invalid:{cid}"),
        acquisition_uid=_fstr(b_raw, "acquisition_uid", cid, check_prov=True),
        run_uid=_fstr(b_raw, "run_uid", cid, check_prov=True),
        parent_digests=_parse_str_tuple(b_raw.get("parent_digests"), f"parent_digests_invalid:{cid}", normalize_digests=True),
        event_ids=_parse_str_tuple(b_raw.get("event_ids"), f"event_ids_invalid:{cid}"),
        device_ranks=_parse_ranks(b_raw.get("device_ranks")),
        clock_domain=_fstr(b_raw, "clock_domain", cid, check_prov=True),
        start_event=_fstr(b_raw, "start_event", cid), end_event=_fstr(b_raw, "end_event", cid),
        boundary_kind=_fstr(b_raw, "boundary_kind", cid), unit=unit,
        capture_uri=sanitize_capture_uri(str(b_raw["capture_uri"])) if b_raw.get("capture_uri") else "",
        source_ref=sanitize_source_ref(b_raw["source_ref"]) if b_raw.get("source_ref") is not None else (),
        session_id=sess,
        clock_operator=_validate_clock_operator(b_raw.get("clock_operator", "elapsed_span"), cid),
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
    r_raw = raw.get("regime_dims") or {}
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


def stream_manifest(
    path: str | Path, *, allow_invalid_durations: bool = False,
    max_whole_read_bytes: int = STREAM_WHOLE_READ_LIMIT_BYTES,
) -> Manifest:
    """Two-pass streaming loader for JSON/JSONL manifests without buffering cases."""
    del max_whole_read_bytes
    target, extra_edges = Path(path), []
    is_jsonl = target.suffix.lower() in {".jsonl", ".ndjson"}
    if is_jsonl:
        header, case_count = {}, 0
        for row in stream_jsonl(target):
            rtype = row.get("record_type", "case")
            if rtype == "manifest_header":
                header.update({k: v for k, v in row.items() if k in MANIFEST_HEADER_KEYS})
            elif rtype == "lineage_edge":
                extra_edges.extend(_parse_lineage_edges([row]))
            else:
                case_count += 1
    else:
        header, case_count = scan_manifest_json_pass1(target, MANIFEST_HEADER_KEYS, ijson_mod=ijson)
    return Manifest(
        manifest_id=_req_str(header.get("manifest_id", target.stem), "manifest_id_invalid"),
        schema_version=_req_str(header.get("schema_version", "mindie-sim.multilevel-manifest/1"), "schema_version_invalid"),
        cases=_LazyManifestCases(target, is_jsonl=is_jsonl, allow_invalid_durations=allow_invalid_durations, count=case_count),
        lineage_edges=(*_parse_lineage_edges(header.get("lineage_edges")), *extra_edges),
        source_uri=sanitize_capture_uri(str(header["source_uri"])) if header.get("source_uri") else "",
        source_sha256=_req_sha256(header["source_sha256"], "manifest_sha256_invalid") if header.get("source_sha256") else "",
        strict_three_level_blind=bool(header.get("strict_three_level_blind", True)),
    )


def validate_lineage(
    manifest: Manifest | str | Path, fits: FrozenFitManifest, *,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> SplitIndex:
    """Compute normalized transitive lineage closure, verify disk captures, and isolate splits."""
    resolved = stream_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    cap_root = Path(capture_root) if capture_root is not None else None
    uf, seen_cases, acq_uids = _UnionFind(), set(), set()
    compact_rows: list[tuple[str, str, str, bool, str, str, tuple[str, ...]]] = []

    for case in resolved.cases:
        if case.case_id in seen_cases:
            raise ModelingError(f"duplicate_case_id:{case.case_id}")
        seen_cases.add(case.case_id)
        b = case.boundary
        raw_sha = _req_sha256(b.raw_sha256, f"raw_sha256_invalid:{case.case_id}")
        acq_uid = _req_str(b.acquisition_uid, f"acquisition_uid_missing:{case.case_id}")
        run_uid = _req_str(b.run_uid, f"run_uid_missing:{case.case_id}")
        session_scope = (b.session_id.strip() if b.session_id else "") or run_uid
        if require_capture_files or cap_root is not None or capture_files is not None:
            cap_path = resolve_capture_file(
                b.capture_uri, b.source_ref, capture_root=cap_root,
                capture_files=capture_files, case_id=case.case_id, raw_sha256=raw_sha,
            )
            if cap_path is None:
                if require_capture_files:
                    raise ModelingError(f"capture_file_missing:{case.case_id}")
            else:
                verify_capture_file_consistency(case.case_id, b, cap_path, ijson_mod=ijson)
            if cap_root is not None and b.source_ref:
                ref_map = dict(b.source_ref)
                ref_path = cap_root / PurePosixPath(ref_map["relative_path"])
                if ref_path.is_file() and digest_file_streaming(ref_path) != _req_sha256(ref_map["source_sha256"], f"source_ref_sha256_invalid:{case.case_id}"):
                    raise ModelingError(f"capture_source_ref_digest_mismatch:{case.case_id}")
        tokens = [raw_sha, acq_uid, run_uid, *(normalize_lineage_token(t) for t in b.parent_digests if t and t.strip())]
        uf.find(tokens[0])
        for tok in tokens[1:]:
            uf.union(tokens[0], tok)
        acq_uids.add(acq_uid)
        compact_rows.append((case.case_id, case.level, case.split, case.debug_cross_level, tokens[0], session_scope, b.event_ids))

    for parent, child in (*resolved.lineage_edges, *fits.lineage_edges):
        uf.union(normalize_lineage_token(parent), normalize_lineage_token(child))
    for param in fits.parameters:
        if param.source_uri:
            sanitize_capture_uri(param.source_uri)
        for digest in param.training_digests:
            if digest and digest.strip():
                uf.find(normalize_lineage_token(digest))

    rep_to_acq: dict[str, list[str]] = {}
    for uid in acq_uids:
        rep_to_acq.setdefault(uf.find(uid), []).append(uid)
    canonical_root = lambda tok: min(rep_to_acq[rep]) if (rep := uf.find(tok)) in rep_to_acq else rep

    # Qualify local event IDs by (acquisition_root, session_scope) so distinct
    # captures reusing local IDs never erroneously merge.
    for _cid, _lvl, _spl, _dbg, anchor, session_scope, ev_ids in compact_rows:
        acq_root = canonical_root(anchor)
        for ev in ev_ids:
            if ev and ev.strip():
                uf.union(anchor, f"event::{acq_root}::{session_scope}::{ev.strip()}")
    rep_to_acq.clear()
    for uid in acq_uids:
        rep_to_acq.setdefault(uf.find(uid), []).append(uid)

    case_roots, comp_splits, comp_cases = {}, {}, {}
    train_comps, heldout_comps, debug_only = set(), set(), set()
    heldout_levels_by_root: dict[str, set[str]] = {}
    heldout_roots_by_level: dict[str, set[str]] = {"L1": set(), "L2": set(), "L3": set()}
    for cid, level, split, is_dbg, anchor, _session_scope, _ev_ids in compact_rows:
        root = canonical_root(anchor)
        case_roots[cid] = root
        comp_cases.setdefault(root, []).append(cid)
        if root in comp_splits and comp_splits[root] != split:
            raise ModelingError(f"lineage_split_leakage:root={root}:cases={','.join(sorted(comp_cases[root]))}")
        comp_splits[root] = split
        if split == "train":
            train_comps.add(root)
        else:
            heldout_comps.add(root)
            if is_dbg:
                debug_only.add(cid)
            else:
                heldout_levels_by_root.setdefault(root, set()).add(level)
                heldout_roots_by_level[level].add(root)

    fit_ancestors: dict[str, frozenset[str]] = {}
    for param in fits.parameters:
        roots = frozenset(canonical_root(normalize_lineage_token(d)) for d in param.training_digests if d and d.strip())
        if roots & heldout_comps:
            raise ModelingError(f"lineage_fit_heldout_leakage:parameter={param.parameter_id}:roots={','.join(sorted(roots & heldout_comps))}")
        fit_ancestors[param.parameter_id] = roots
    if resolved.strict_three_level_blind:
        for root, levels in sorted(heldout_levels_by_root.items()):
            if len(levels) > 1:
                raise ModelingError(f"lineage_cross_level_root_overlap:root={root}:levels={','.join(sorted(levels))}")
    return SplitIndex(
        case_roots=MappingProxyType(case_roots), component_splits=MappingProxyType(comp_splits),
        train_components=frozenset(train_comps), heldout_components=frozenset(heldout_comps),
        heldout_roots_by_level=MappingProxyType({k: frozenset(v) for k, v in heldout_roots_by_level.items()}),
        debug_only_cases=frozenset(debug_only), fit_ancestor_components=MappingProxyType(fit_ancestors),
    )


def run_l1(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-1 operator/collective case on the Engine."""
    if case.level != "L1" or case.boundary.boundary_kind not in L1_BOUNDARY_KINDS:
        raise ModelingError(f"run_l1_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return validate_engine_interval(case, runner.run_l1(case), expected_level="L1")


def run_l2(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-2 subgraph/block/rung case on the Engine."""
    if case.level != "L2":
        raise ModelingError(f"run_l2_level_mismatch:{case.case_id}:{case.level}")
    if case.boundary.boundary_kind in FORBIDDEN_L2_BOUNDARY_KINDS:
        raise ModelingError(f"l2_derived_from_l3_forbidden:{case.case_id}:{case.boundary.boundary_kind}")
    if not case.block_capture_present or case.boundary.boundary_kind == "missing_block_capture":
        return EngineInterval(case.case_id, "L2", case.boundary.boundary_kind, case.boundary.start_event,
                              case.boundary.end_event, 0, 0, 0, "", status="missing_evidence",
                              missing_reason="block_capture_absent", clock_operator=case.boundary.clock_operator)
    if case.boundary.boundary_kind not in L2_BOUNDARY_KINDS:
        raise ModelingError(f"run_l2_boundary_invalid:{case.case_id}:{case.boundary.boundary_kind}")
    interval = runner.run_l2(case)
    if interval.derived_from_step_division or interval.source_level == "L3":
        raise ModelingError(f"l2_cloned_or_divided_from_l3_forbidden:{case.case_id}")
    return validate_engine_interval(case, interval, expected_level="L2")


def run_l3(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-3 model/service step case on the Engine."""
    if case.level != "L3" or case.boundary.boundary_kind not in L3_BOUNDARY_KINDS:
        raise ModelingError(f"run_l3_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return validate_engine_interval(case, runner.run_l3(case), expected_level="L3")


def _make_residual(
    case: GateCase, root: str, *, measured_ns: int, interval: EngineInterval | None = None,
    signed_error: float = math.nan, status: ResidualStatus = "ok",
    excluded_reason: str = "", signed_delta_ns: int | None = None,
    arm_a_err: float | None = None, arm_b_err: float | None = None,
) -> Residual:
    return Residual(
        case.case_id, case.level, case.cohort, root, measured_ns,
        interval.predicted_ns if interval else 0, interval.engine_receipt if interval else "",
        signed_error, math.nan if math.isnan(signed_error) else abs(signed_error),
        status, excluded_reason, case.boundary.boundary_kind, signed_delta_ns, arm_a_err, arm_b_err, case.regime_dims,
        case.boundary.clock_operator, interval.elapsed_span_ns if interval else None, interval.busy_union_ns if interval else None,
    )


def evaluate(
    manifest: Manifest | str | Path, fits: FrozenFitManifest, runner: EvaluationRunner, *,
    target_split: Literal["heldout", "train", "all"] = "heldout",
    near_zero_delta_ns: int = DEFAULT_NEAR_ZERO_DELTA_NS,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> Iterator[Residual]:
    """Two-pass streaming gate: Pass 1 lineage/capture verification, Pass 2 Engine scoring."""
    resolved = stream_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    split_index = validate_lineage(
        resolved, fits, capture_root=capture_root,
        capture_files=capture_files, require_capture_files=require_capture_files,
    )
    receipt_levels: dict[str, set[str]] = {}
    dispatch = {"L1": run_l1, "L2": run_l2, "L3": run_l3}
    for case in resolved.cases:
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
        for rcpt_key in (interval.engine_receipt, *(f"engine_uid:{u}" for u in (interval.run_receipt.engine_uid.split("+") if interval.run_receipt else ()))):
            seen = receipt_levels.setdefault(rcpt_key, set())
            seen.add(case.level)
            if len(seen) > 1:
                raise ModelingError(f"cross_level_engine_receipt_clone:{rcpt_key}")
        if is_rung:
            assert arm_a is not None and arm_b is not None and interval.arm_a_predicted_ns is not None and interval.arm_b_predicted_ns is not None
            delta_ns = interval.predicted_ns - measured_ns
            near_zero = measured_ns == 0 or abs(measured_ns) <= max(near_zero_delta_ns, int(max(abs(arm_a), abs(arm_b)) * NEAR_ZERO_DELTA_MIN_RATIO))
            status: ResidualStatus = "debug_only" if is_debug else ("near_zero_delta" if near_zero else "ok")
            reason = "debug_cross_level_not_blind_acceptance" if is_debug else ("near_zero_rung_delta" if near_zero else "")
            yield _make_residual(
                case, root, measured_ns=measured_ns, interval=interval,
                signed_error=math.nan if (is_debug or near_zero) else delta_ns / float(measured_ns),
                status=status, excluded_reason=reason, signed_delta_ns=delta_ns,
                arm_a_err=(interval.arm_a_predicted_ns - arm_a) / float(arm_a),
                arm_b_err=(interval.arm_b_predicted_ns - arm_b) / float(arm_b),
            )
        else:
            yield _make_residual(
                case, root, measured_ns=measured_ns, interval=interval,
                signed_error=(interval.predicted_ns - measured_ns) / float(measured_ns),
                status="debug_only" if is_debug else "ok",
                excluded_reason="debug_cross_level_not_blind_acceptance" if is_debug else "",
            )


def _physical_tolerance_ns(measured_ns: int, threshold_pct: float, atol_ns: float) -> float:
    return max(threshold_pct / PERCENT_SCALE * abs(measured_ns), atol_ns)


def exceeds_physical_tolerance(
    predicted_ns: int, measured_ns: int, *, threshold_pct: float = DEFAULT_THRESHOLD_PCT, atol_ns: float = 0.0,
) -> bool:
    """Check the combined tolerance, with equality allowed when a noise floor is enabled."""
    limit = _physical_tolerance_ns(measured_ns, threshold_pct, atol_ns)
    error = abs(predicted_ns - measured_ns)
    return error > limit if atol_ns > 0.0 else error >= limit


def evaluate_vertical_drilldown(
    step_predicted_ns: int, step_measured_ns: int, components: Sequence[tuple[str, str, int, int]], *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT, atol_ns: float = DEFAULT_ATOL_NS,
) -> dict[str, Any]:
    """Score a step and its same-capture operator families and blocks."""
    exceeds = partial(exceeds_physical_tolerance, threshold_pct=threshold_pct, atol_ns=atol_ns)
    return score_vertical_drilldown(step_predicted_ns, step_measured_ns, components, exceeds)


def summarize_cohorts(
    residuals: Iterable[Residual], *, threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    strict_pointwise: bool = True, atol_ns: float = 0.0,
) -> dict[CohortKey, CohortSummary]:
    """Report raw percentages. With a noise floor, gate on fractions of each row's tolerance."""
    physical = atol_ns > 0.0
    buckets: dict[CohortKey, list[Any]] = {}
    for row in residuals:
        bucket = buckets.get(row.cohort)
        if bucket is None:
            bucket = [[], [], set(), [], set(), Counter(), [], 0]
            buckets[row.cohort] = bucket
        apes, signed, roots, tails, evaluated_ok, excluded, tolerance_ratios, _ = bucket
        ape_pct = row.absolute_percentage_error * PERCENT_SCALE
        if row.status == "ok" and math.isfinite(ape_pct):
            apes.append(ape_pct)
            signed.append(row.signed_error * PERCENT_SCALE)
            roots.add(row.acquisition_root)
            evaluated_ok.add(row.case_id)
            if physical:
                limit = _physical_tolerance_ns(row.measured_ns, threshold_pct, atol_ns)
                tolerance_ratios.append(abs(row.predicted_ns - row.measured_ns) / limit)
            exceeded = (exceeds_physical_tolerance(
                row.predicted_ns, row.measured_ns, threshold_pct=threshold_pct, atol_ns=atol_ns,
            ) if physical else ape_pct >= threshold_pct)
            if exceeded:
                tails.append(row.case_id)
        else:
            excluded[row.excluded_reason or row.status] += 1
            bucket[-1] += int(row.status == "missing_evidence")
    summaries: dict[CohortKey, CohortSummary] = {}
    for cohort_key, (apes, signed, roots, tails, evaluated_ok, excluded, ratios, missing) in buckets.items():
        excluded_tuple = tuple(sorted(excluded.items()))
        if not apes:
            summaries[cohort_key] = CohortSummary(
                cohort_key, "insufficient_evidence", 0, 0, None, None, None, None, None,
                missing, excluded_tuple, threshold_pct, ("insufficient_evidence",), atol_ns=atol_ns,
            )
            continue
        apes.sort()
        mape_pct, medape_pct = float(statistics.fmean(apes)), float(statistics.median(apes))
        p90_idx = min(len(apes) - 1, max(0, math.ceil(TAIL_QUANTILE_P90 * len(apes)) - 1))
        scores = (statistics.fmean(ratios), statistics.median(ratios)) if physical else (mape_pct, medape_pct)
        failed = tuple(name for name, score in zip(("mape_pct", "medape_pct"), scores)
                       if (score > 1.0 if physical else score >= threshold_pct))
        if strict_pointwise and tails:
            failed += ("max_ape_pct",)
        status: CohortStatus = "fail" if failed else ("insufficient_evidence" if missing else "pass")
        summaries[cohort_key] = CohortSummary(
            cohort_key, status, len(roots), len(apes),
            mape_pct, medape_pct, float(statistics.median(signed)), float(apes[p90_idx]), float(apes[-1]),
            missing, excluded_tuple, threshold_pct, failed, tuple(tails), tuple(sorted(roots)), tuple(sorted(evaluated_ok)),
            atol_ns=atol_ns, pointwise_failed_count=len(tails),
        )
    return summaries


def gate_decision(
    summaries: Mapping[CohortKey, CohortSummary], *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    atol_ns: float = 0.0,
    required_levels: Sequence[Level] = REQUIRED_LEVELS,
    require_disjoint_level_roots: bool = True,
    expected_case_ids: Iterable[str] | None = None,
    evaluated_case_ids: Iterable[str | Residual] | None = None,
    calibrated_count: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Compute the release verdict from cohort results and optional planned coverage.

    Coverage comes from scored residuals recorded in summaries. The
    evaluated_case_ids argument is retained for caller compatibility. Physical
    tolerance settings must match those used to score the cohorts.
    """
    if any((atol_ns > 0.0 or v.atol_ns > 0.0) and (atol_ns != v.atol_ns or threshold_pct != v.threshold_pct)
           for v in summaries.values()):
        raise ModelingError("cohort_physical_tolerance_mismatch")
    missing_planned_cases = []
    if expected_case_ids is not None:
        evaluated = {case_id for summary in summaries.values() for case_id in summary.evaluated_ok_case_ids}
        missing_planned_cases = sorted(set(expected_case_ids) - evaluated)
    pointwise_failures = {
        k.as_id() for k, v in summaries.items()
        if (v.pointwise_failed_count > 0 if atol_ns > 0.0 else v.max_ape_pct is not None and v.max_ape_pct >= threshold_pct)
    }
    failed = sorted(pointwise_failures | {k.as_id() for k, v in summaries.items() if v.status == "fail"})
    insufficient = sorted(k.as_id() for k, v in summaries.items() if v.status == "insufficient_evidence")
    tail_warnings = sorted(k.as_id() for k, v in summaries.items() if v.tail_exceeded)
    covered_levels: set[str] = set()
    root_levels: dict[str, set[str]] = {}
    for key, summary in summaries.items():
        if summary.n_rows > 0 and summary.n_roots > 0:
            covered_levels.add(key.level)
            for root in summary.acquisition_roots:
                root_levels.setdefault(root, set()).add(key.level)
    missing_levels = sorted(lvl for lvl in required_levels if lvl not in covered_levels)
    overlapping_roots = sorted(r for r, lvls in root_levels.items() if len(lvls) > 1)
    incomplete = (
        not summaries or insufficient or missing_levels or missing_planned_cases
        or (require_disjoint_level_roots and overlapping_roots)
        or plan_only or (calibrated_count is not None and calibrated_count <= 0)
    )
    status = "fail" if failed else ("insufficient_evidence" if incomplete else "pass")
    return {
        "status": status, "cohort_count": len(summaries),
        "covered_levels": sorted(covered_levels), "missing_levels": missing_levels,
        "overlapping_level_roots": overlapping_roots,
        "failed_cohorts": failed, "insufficient_cohorts": insufficient, "tail_warning_cohorts": tail_warnings,
        "pointwise_threshold_exceeded": sorted(pointwise_failures),
        "missing_planned_cases": missing_planned_cases, "plan_only": plan_only, "calibrated_count": calibrated_count,
    }


__all__ = [
    "CAPTURE_URI_SCHEME", "CaptureBoundary", "CohortKey", "CohortSummary",
    "DEFAULT_ATOL_NS", "DEFAULT_NEAR_ZERO_DELTA_NS", "DEFAULT_THRESHOLD_PCT", "EngineInterval",
    "EngineRunReceipt", "EvaluationRunner", "FrozenFitManifest", "FrozenFitParameter",
    "GateCase", "Manifest", "REQUIRED_LEVELS", "Residual", "SHA256_HEX_LENGTH",
    "SOURCE_REF_KEYS", "STREAM_CHUNK_BYTES", "STREAM_WHOLE_READ_LIMIT_BYTES",
    "SplitIndex", "TIME_UNIT_SCALE_NS", "VALID_CLOCK_OPERATORS", "bind_engine_interval", "bind_engine_rung_interval",
    "evaluate", "evaluate_vertical_drilldown", "exceeds_physical_tolerance", "gate_decision", "normalize_lineage_token",
    "parse_gate_case", "run_l1", "run_l2", "run_l3", "sanitize_capture_uri", "sanitize_source_ref", "stream_jsonl",
    "stream_manifest", "summarize_cohorts", "validate_lineage", "validate_provenance_text",
]
