"""Ragged request and bounded analysis-domain geometry.

[Role] Type the dynamic ``X`` input: per-request query/context ranges, explicit
  graph bucket/media/routing state and the bounded domain used for finite
  piecewise enumeration.
[Invariants] No implicit homogeneous approximation replaces a ragged vector.
  A graph bucket pads physical tokens, never valid attention pairs.  Domain
  misses raise.
[Timeouts & Locks] Pure bounded arithmetic; no I/O or locks.
[Side-Effect Policy] Immutable records only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError


def local_batch(concurrency: int, dp: int) -> int:
    """Requests per DP replica under even scheduler admission."""
    return max(1, math.ceil(max(1, int(concurrency)) / max(1, int(dp))))


def ep_batch(concurrency: int, dp: int, ep: int) -> int:
    """Total requests participating in one expert-parallel AllToAll cohort."""
    dp_n, ep_n = max(1, int(dp)), max(1, int(ep))
    if ep_n <= 1:
        return local_batch(concurrency, dp_n)
    return max(1, math.ceil(max(1, int(concurrency)) * min(dp_n, ep_n) / dp_n))


def clamp_graph_bucket(bucket: int, batch: int, dp: int, buckets: tuple[int, ...] | list[int] = ()) -> int:
    """Select a valid local graph capture bucket covering the per-DP-rank batch."""
    bucket = max(0, int(bucket))
    if bucket <= 0:
        return 0
    lb = max(1, int(batch))
    if buckets:
        valid = {int(x) for x in buckets if int(x) > 0}
        for b in sorted(valid):
            if b >= lb:
                return b
        return 0
    if bucket < lb:
        return 0
    if int(dp) <= 1:
        return bucket
    local_cap = 1 << (lb - 1).bit_length()
    return min(bucket, max(lb, local_cap))


@dataclass(frozen=True)
class AnalysisDomain:
    max_tokens: int = 1_048_576
    max_batch: int = 4096
    max_q: int = 262_144
    max_s_kv: int = 1_048_576
    max_seq: int = 1_048_576
    max_query_tiles: int = 65536
    max_kv_tiles: int = 65536
    max_work: int = 1_000_000_000_000
    max_moe_rows: int = 1_048_576
    max_index_rows: int = 262_144
    max_media_tokens: int = 65536

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"analysis_domain_invalid:{name}")

    def bound(self, key: str) -> int:
        if key not in self.__dataclass_fields__:
            raise ModelingError(f"analysis_domain_key_unknown:{key}")
        return int(getattr(self, key))

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def _normalize_query_ranges(ranges: Any, q_len: int) -> tuple[tuple[int, int], ...]:
    """Validate and canonicalise selected rows into sorted disjoint intervals.

    Merging touching intervals makes the physical selection independent of the
    caller's iteration order; the same rows always produce the same tile and
    causal-bound work.
    """
    parsed: list[tuple[int, int]] = []
    try:
        items = tuple(ranges)
    except TypeError:
        raise ModelingError(
            f"request_query_range_invalid:{ranges!r}:{q_len}") from None
    for item in items:
        try:
            start, end = item
        except (TypeError, ValueError):
            raise ModelingError(
                f"request_query_range_invalid:{item!r}:{q_len}") from None
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)):
            raise ModelingError(
                f"request_query_range_invalid:{start}:{end}:{q_len}")
        if start < 0 or start > end or end > q_len:
            raise ModelingError(
                f"request_query_range_invalid:{start}:{end}:{q_len}")
        parsed.append((start, end))
    parsed.sort()
    merged: list[tuple[int, int]] = []
    for start, end in parsed:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            if end > previous_end:
                merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class RequestGeometry:
    """One ragged request: query length, pre-iteration context and ranges."""

    q_len: int
    context: int
    phase: str = "decode"
    query_ranges: tuple[tuple[int, int], ...] = ()
    replica: int = 0
    draft_len: int = 1
    # PCP rank chunks remain part of one logical request.  owner_index is the
    # original request index, parent_query_ranges preserves its declared
    # ranges, and emit marks the chunk containing the final logical token.
    owner_index: int = -1
    parent_query_ranges: tuple[tuple[int, int], ...] = ()
    emit: bool = False
    emit_position: int = -1

    def __post_init__(self) -> None:
        for name in ("q_len", "context", "draft_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"request_geometry_invalid:{name}")
        if self.phase not in {"prefill", "decode", "draft", "draft_prefill", "verify"}:
            raise ModelingError(f"request_phase_invalid:{self.phase}")
        object.__setattr__(self, "query_ranges",
                           _normalize_query_ranges(self.query_ranges, self.q_len))
        object.__setattr__(self, "parent_query_ranges", tuple(
            (int(start), int(end)) for start, end in self.parent_query_ranges))
        object.__setattr__(self, "emit", bool(self.emit))
        if isinstance(self.emit_position, bool) or not isinstance(self.emit_position, int):
            raise ModelingError("request_geometry_invalid:emit_position")
        if self.emit_position < -1:
            raise ModelingError("request_geometry_invalid:emit_position")

    @property
    def kv_len(self) -> int:
        return self.context + self.q_len

    @property
    def selected_q_len(self) -> int:
        """Valid query rows selected by ``query_ranges`` (all rows if empty)."""
        if not self.query_ranges:
            return self.q_len
        return sum(end - start for start, end in self.query_ranges)

    def pairs(self, *, window: int = 0, causal: bool = True) -> int:
        if not causal:
            return self.selected_q_len * self.kv_len
        ranges = self.query_ranges or ((0, self.q_len),)
        total = 0
        for start, end in ranges:
            total += _range_window_sum(start, end, self.context, window)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {"q_len": self.q_len, "context": self.context, "phase": self.phase,
                "query_ranges": [list(item) for item in self.query_ranges],
                "replica": self.replica, "draft_len": self.draft_len,
                "owner_index": self.owner_index,
                "parent_query_ranges": [list(item) for item in self.parent_query_ranges],
                "emit": self.emit, "emit_position": self.emit_position}


def _range_window_sum(start: int, end: int, context: int, window: int) -> int:
    """Exact sum over query positions of valid causal key counts."""
    if end <= start:
        return 0
    if window <= 0:
        return (end - start) * context + end * (end + 1) // 2 - start * (start + 1) // 2
    split = max(start, min(end, window - context - 1))
    ramp = (split - start) * context + split * (split + 1) // 2 - start * (start + 1) // 2
    return ramp + (end - split) * window


@dataclass(frozen=True)
class WorkloadGeometry:
    """Ragged request vector and declared graph/media state for one slice."""

    requests: tuple[RequestGeometry, ...]
    graph_bucket: int = 0
    cohort_batch: int = 0
    physical_tokens: int = 0
    # Exact local operand rows selected by the source role table.  This is an
    # operand view, not a new request/capture: it may be below the logical
    # tokens or graph bucket while those dimensions remain intact.
    operand_tokens: int = 0
    media_tokens: int = 0
    routing: tuple[tuple[int, tuple[int, ...]], ...] = ()
    phase: str = ""
    local_batch: int = 0
    # Token lanes. 0 / False / "" means the case did not declare that input.
    # They never rewrite graph_bucket, physical_tokens or effective_tokens.
    tp_width: int = 0
    dense_sequence_parallel: bool = False
    graph_dispatch: str = ""
    capture_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        if not requests:
            raise ModelingError("workload_requests_empty")
        if not all(isinstance(request, RequestGeometry) for request in requests):
            raise ModelingError("workload_request_record_invalid")
        object.__setattr__(self, "requests", requests)
        for name in ("graph_bucket", "cohort_batch", "physical_tokens",
                     "operand_tokens", "media_tokens", "local_batch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"workload_geometry_invalid:{name}")
        self._normalize_lanes()
        object.__setattr__(self, "routing",
                           tuple((int(layer), tuple(counts)) for layer, counts in self.routing))
        tok = sum(r.q_len for r in requests)
        sel = sum(r.selected_q_len for r in requests)
        ssum = sum(r.kv_len for r in requests)
        smax = max(r.kv_len for r in requests)
        qmax = max(r.q_len for r in requests)
        q2 = sum(r.q_len * r.q_len for r in requests)
        captured = max(tok, self.physical_tokens or 0, self.graph_bucket or 0)
        eff = self.operand_tokens or captured
        p_c = sum(r.pairs(window=0, causal=True) for r in requests)
        p_nc = sum(r.pairs(window=0, causal=False) for r in requests)
        object.__setattr__(self, "_stats", (tok, sel, ssum, smax, qmax, q2, eff))
        object.__setattr__(self, "_for_sg_cache", {})
        object.__setattr__(self, "_bindings_cache", {
            "T": eff, "T_VALID": tok, "T_SELECTED": sel, "selected_tokens": sel,
            "B": len(requests), "EP_B": self.cohort_batch or len(requests),
            "Q": qmax, "S": smax, "S_SUM": ssum, "Q2": q2,
            "PAIRS": p_c, "PAIRS_NC": p_nc, "BUCKET": self.graph_bucket or captured,
            "MEDIA": self.media_tokens, "media_tokens": self.media_tokens,
            "tokens": eff, "valid_tokens": tok, "batch": len(requests),
            "operand_tokens": self.operand_tokens, "captured_tokens": captured,
            "q_max": qmax, "s_max": smax, "s_sum": ssum, "q2": q2,
            # Source path selector, not a timing constant: only the prefill
            # dynamic-token path executes the outbound TokenPermute launch.
            "IS_PREFILL": int(self.phase_name == "prefill"),
        })
        lanes = self._lane_record()
        object.__setattr__(self, "_lanes_cache", lanes)
        object.__setattr__(self, "_bindings_cache", {**self._bindings_cache, **lanes})

    def _normalize_lanes(self) -> None:
        width = self.tp_width
        if isinstance(width, bool) or not isinstance(width, int) or width < 0:
            raise ModelingError(f"workload_geometry_invalid:tp_width")
        if not isinstance(self.dense_sequence_parallel, bool):
            raise ModelingError("workload_geometry_invalid:dense_sequence_parallel")
        if not isinstance(self.graph_dispatch, str):
            raise ModelingError("workload_lane_dispatch_invalid")
        dispatch = self.graph_dispatch.strip().lower()
        allowed = {"", "none", "full", "full_decode_only", "piecewise"}
        if dispatch not in allowed:
            raise ModelingError(f"workload_lane_dispatch_invalid:{dispatch}")
        object.__setattr__(self, "graph_dispatch", dispatch)
        sizes: list[int] = []
        try:
            raw_sizes = tuple(self.capture_sizes or ())
        except TypeError:
            raise ModelingError("workload_capture_size_invalid") from None
        for size in raw_sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ModelingError(f"workload_capture_size_invalid:{size}")
            if size > 0:
                sizes.append(size)
        object.__setattr__(self, "capture_sizes", tuple(sizes))

    def _lane_record(self) -> dict[str, Any]:
        """Scheduled rows, capture lanes and the shard each owner must bind."""
        requests = self.requests
        scheduled = len(requests)
        valid = sum(request.q_len for request in requests)
        head = self.producing_lm_head_rows()
        tp = int(self.tp_width)
        dense = self.dense_sequence_parallel
        rounded = valid + (-valid) % tp if dense and tp > 1 else valid
        phases = {request.phase for request in requests}
        widths = {request.q_len for request in requests}
        uniform = (self.phase_name == "decode" and phases == {"decode"}
                   and len(widths) == 1 and min(widths) >= 1)
        static_mode = self.graph_dispatch in {"full", "full_decode_only"}
        static_width = 0
        if not dense:
            reason = "dense_sp_not_declared"
        elif tp < 2:
            reason = "tp_width_not_declared"
        elif not static_mode:
            reason = "graph_dispatch_not_static"
        elif not uniform:
            reason = "not_uniform_decode"
        else:
            static_width = self._capture_width(rounded, tp)
            if static_width <= 0:
                reason = "no_capture_size_covers_tp_rounded_tokens"
            elif static_width % tp != 0:
                raise ModelingError(
                    f"static_lane_not_divisible_by_tp:{static_width}:{tp}")
            else:
                reason = "static_dense_sp"
        static = reason == "static_dense_sp"
        if static:
            gathered = static_width
            embedding = static_width
        else:
            gathered = valid
            embedding = rounded if dense and tp > 1 else valid
        if dense and tp > 1 and embedding % tp == 0:
            shard = embedding // tp
        else:
            shard = valid
        return {
            "scheduled_sequences": scheduled,
            "valid_tokens": valid,
            "head_tokens": head,
            "gathered_tokens": gathered,
            "embedding_tokens": embedding,
            "mlp_tokens": embedding,
            "shard_tokens": shard,
            "tp_rounded_tokens": rounded,
            "lane_declaration": reason,
            "static_decode": static,
        }

    def _capture_width(self, rounded: int, tp: int) -> int:
        """Smallest TP-aligned capture size covering the SP round-up."""
        if not self.capture_sizes:
            return rounded
        covering = [size for size in self.capture_sizes
                    if size >= rounded and size % tp == 0]
        return min(covering) if covering else 0

    def token_lanes(self) -> dict[str, Any]:
        """The one row source for GDN, FIA, COM3 norm and the embedding span."""
        cached = getattr(self, "_lanes_cache", None)
        if not isinstance(cached, dict):
            cached = self._lane_record()
        return dict(cached)

    def operator_layout(self, kind: str, role: str = "", name: str = "",
                        family: str = "") -> "WorkloadGeometry":
        """Project one operator onto its source role's physical dense rows.

        The lane record is already source-derived by the graph declaration.
        This method chooses which existing lane an operator consumes; it does
        not invent a second request, multiply all tensors, or change the
        logical request vector.  Non-static / non-dense-SP geometries are
        returned unchanged so other models keep their ordinary effective rows.
        """
        lanes = self.token_lanes()
        if not lanes.get("static_decode"):
            return self
        base = str(name)
        if base.endswith("_QuantV2"):
            base = base[:-len("_QuantV2")]
        normish = (role in {"norm", "residual"}
                   or base in {"rmsnorm", "final_norm", "residual_add",
                               "fused:add_rmsnorm", "add_rmsnorm"}
                   or base.startswith("fused:add_rmsnorm"))
        ffnish = kind in {"F0", "F1"} or role in {
            "gate_up", "down", "activation", "projection"}
        attentionish = kind in {"A0", "A1", "A2"}
        recurrent = kind in {"A3", "A4"}
        if normish:
            lane = int(lanes["shard_tokens"])
        elif family == "collective":
            lane = int(lanes["gathered_tokens"])
        elif ffnish:
            lane = int(lanes["mlp_tokens"])
        elif attentionish:
            lane = (int(lanes["valid_tokens"])
                    if family in {"attention", "mla"} or base == "attention"
                    else int(lanes["gathered_tokens"]))
        elif recurrent:
            if (role in {"qkvz", "ba", "output"}
                    or base in {"qkvz_projection", "ba_projection", "core_zeros"}
                    or base.endswith("_projection")):
                lane = int(lanes["gathered_tokens"])
            else:
                lane = int(lanes["valid_tokens"])
        elif kind == "E0" or role == "embedding":
            lane = int(lanes["embedding_tokens"])
        elif kind == "O0":
            lane = (int(lanes["head_tokens"]) if base == "lm_head"
                    else int(lanes["shard_tokens"]))
        else:
            lane = int(lanes["valid_tokens"])
        lane = max(1, lane)
        if lane == self.effective_tokens:
            return self
        return replace(self, operand_tokens=lane)

    def physical_gdn_rows(self) -> tuple[int, int, int]:
        """Forward, decode and prefill rows the v0.30.0 copy rule sees.

        A static dense-SP decode runs q, k, v, qkv, beta, alpha, Z and the
        zero fill on the captured width. It does not add a request. Every
        other phase keeps the logical q_len partition, so a one-row view
        stays a view and a graph bucket stays off this row.
        """
        lanes = self.token_lanes()
        if lanes["static_decode"]:
            width = int(lanes["gathered_tokens"])
            return width, width, 0
        forward = decode = prefill = 0
        for request in self.requests:
            rows = int(request.q_len)
            forward += rows
            if request.phase == "decode":
                decode += rows
            elif request.phase == "prefill":
                prefill += rows
        return forward, decode, prefill

    @classmethod
    def homogeneous(cls, batch: int, q_len: int, s_kv: int, *,
                    phase: str = "decode", **kwargs: Any) -> "WorkloadGeometry":
        if batch < 1 or q_len < 1 or s_kv < q_len:
            raise ModelingError("workload_homogeneous_invalid")
        proto = RequestGeometry(q_len, s_kv - q_len, phase=phase)
        return cls((proto,) * batch, phase=phase, **kwargs)

    @property
    def batch(self) -> int:
        return len(self.requests)

    @property
    def tokens(self) -> int:
        return self._stats[0]

    @property
    def selected_tokens(self) -> int:
        """Valid query rows after applying every request's query ranges."""
        return self._stats[1]

    def producing_lm_head_rows(self) -> int:
        """Rows the LM head writes, and the logits collective must move.

        An unfinished prefill with no explicit ranges emits one terminal row
        per request. Decode, draft, verify, and explicit query ranges use the
        selected query length. Graph-bucket and physical padding are not rows.
        """
        ranged = any(request.query_ranges for request in self.requests)
        if not ranged and self.phase_name == "prefill":
            return self.batch
        return int(self.selected_tokens)

    @property
    def s_sum(self) -> int:
        return self._stats[2]

    @property
    def s_max(self) -> int:
        return self._stats[3]

    @property
    def q_max(self) -> int:
        return self._stats[4]

    @property
    def q2_sum(self) -> int:
        return self._stats[5]

    @property
    def effective_tokens(self) -> int:
        return self._stats[6]

    @property
    def phase_name(self) -> str:
        return self.phase or self.requests[0].phase

    def pairs(self, *, window: int = 0, causal: bool = True) -> int:
        return sum(request.pairs(window=window, causal=causal) for request in self.requests)

    def routing_for(self, layer: int) -> tuple[int, ...] | None:
        for key, counts in self.routing:
            if key == layer:
                return counts
        return None

    def expanded_to(self, batch: int) -> "WorkloadGeometry":
        """Expand request vector to ``batch`` while keeping cohort metadata.

        Ordinary ragged cohorts keep the historical shape guard.  PCP physical
        fragments are not independent cohort rows: each logical owner owns a
        contiguous group of early/late chunks.  Expansion repeats the real owner
        groups in source order (never uniform chunk copies), so an already-local
        EP cohort keeps both chunks unchanged.
        """
        batch = max(1, int(batch))
        owned = any(request.owner_index >= 0 for request in self.requests)
        if batch == self.batch and not owned:
            return self
        if owned:
            groups: dict[int, list[RequestGeometry]] = {}
            order: list[int] = []
            for request in self.requests:
                owner = request.owner_index
                if owner not in groups:
                    groups[owner] = []
                    order.append(owner)
                groups[owner].append(request)
            if batch < len(order):
                raise ModelingError(
                    f"workload_pcp_cohort_below_local_owners:{len(order)}->{batch}")
            else:
                reps, rem = divmod(batch, len(order))
                keep = order * reps + order[:rem]
            requests: list[RequestGeometry] = []
            for new_owner, owner in enumerate(keep):
                for request in groups[owner]:
                    requests.append(replace(request, owner_index=new_owner))
            return replace(self, requests=tuple(requests),
                           cohort_batch=batch, local_batch=batch)
        shapes = {(r.q_len, r.context, r.phase, r.query_ranges, r.draft_len) for r in self.requests}
        loc = self.local_batch or self.batch
        if len(shapes) > 1:
            if batch < self.batch or batch % self.batch != 0:
                raise ModelingError(f"workload_ragged_cohort_expansion_ambiguous:{self.batch}->{batch}")
            reps = batch // self.batch
            return replace(self, requests=tuple(replace(r) for _ in range(reps) for r in self.requests),
                           cohort_batch=batch, local_batch=loc)
        proto = self.requests[0]
        return replace(self, requests=tuple(replace(proto) for _ in range(batch)),
                       cohort_batch=batch, local_batch=loc)

    def for_subgraph(self, kind: str) -> "WorkloadGeometry":
        """Lift EP-wide subgraphs (routed MoE F3 and prefill F2) to cohort batch."""
        if kind == "F3" or (kind == "F2" and self.phase_name == "prefill"):
            if kind not in self._for_sg_cache:
                self._for_sg_cache[kind] = self.expanded_to(self.cohort_batch or self.batch)
            return self._for_sg_cache[kind]
        return self

    def bindings(self) -> dict[str, Any]:
        return dict(self._bindings_cache)

    def validate(self, domain: AnalysisDomain) -> None:
        if self.batch > domain.max_batch:
            raise ModelingError(f"analysis_domain_miss:batch:{self.batch}>{domain.max_batch}")
        if self.tokens > domain.max_tokens:
            raise ModelingError(f"analysis_domain_miss:tokens:{self.tokens}>{domain.max_tokens}")
        if self.physical_tokens > domain.max_tokens:
            raise ModelingError(
                "analysis_domain_miss:physical_tokens:"
                f"{self.physical_tokens}>{domain.max_tokens}")
        if self.effective_tokens > domain.max_tokens:
            raise ModelingError(
                f"analysis_domain_miss:effective_tokens:"
                f"{self.effective_tokens}>{domain.max_tokens}")
        if self.media_tokens > domain.max_media_tokens:
            raise ModelingError(
                f"analysis_domain_miss:media:{self.media_tokens}>{domain.max_media_tokens}")
        if self.q_max > domain.max_q:
            for request in self.requests:
                if request.q_len > domain.max_q:
                    raise ModelingError(f"analysis_domain_miss:q:{request.q_len}>{domain.max_q}")
        if self.s_max > domain.max_s_kv:
            for request in self.requests:
                if request.kv_len > domain.max_s_kv:
                    raise ModelingError(
                        f"analysis_domain_miss:s_kv:{request.kv_len}>{domain.max_s_kv}")
        if self.graph_bucket > domain.max_seq:
            raise ModelingError(f"analysis_domain_miss:bucket:{self.graph_bucket}>{domain.max_seq}")

    def to_dict(self) -> dict[str, Any]:
        return {"requests": [request.to_dict() for request in self.requests],
                "graph_bucket": self.graph_bucket,
                "physical_tokens": self.physical_tokens,
                "operand_tokens": self.operand_tokens,
                "media_tokens": self.media_tokens, "tokens": self.tokens}


def _as_map(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, list)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _switch_on(value: Any) -> bool:
    """Same string rule as dense FlashComm: ``1`` / ``true``, or a real bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    if isinstance(value, bool):
        return value
    return False


def _language_model_type(meta: Mapping[str, Any], hf: Mapping[str, Any]) -> str:
    """Bound language type. An explicit key wins, then the text config.

    The checkpoint wrapper ``qwen3_5`` is not the text model. The catalog
    nickname is not read.
    """
    declared = str(meta.get("source_model_type") or "")
    if declared:
        return declared
    nested = str(_as_map(hf.get("text_config")).get("model_type") or "")
    if nested:
        return nested
    return str(hf.get("model_type") or "")


def deployment_reject(status: Mapping[str, Any] | None, *, cell_id: str,
                      tp: int, dp: int, pp: int) -> str:
    """Scope check for the optional lane record. Not a model reject.

    A missing status is conditional. A present status that does not name
    this cell, or whose tp, dp, or pp disagree, withholds that file's
    switch and cudagraph mode. The mutable last-run cells list is not
    membership in the bound model, and a caller must not zero the forecast
    for this reason. An explicit tampered source override stays on the
    case's own ``source_reject_reason``.
    """
    if not isinstance(status, Mapping) or not status:
        return ""
    tags = {str(cell.get("tag")) for cell in status.get("cells") or ()
            if isinstance(cell, Mapping) and cell.get("tag")}
    ident = str(cell_id).split(":")[-1]
    if ident not in tags:
        return "source_cell_not_in_deployment"
    spec = _as_map(status.get("spec"))
    if int(spec.get("tp") or 0) != int(tp) or int(spec.get("dp") or 0) != int(dp):
        return "source_parallel_contradicts_deployment"
    declared_pp = spec.get("pp")
    expected_pp = declared_pp if isinstance(declared_pp, int) and not isinstance(declared_pp, bool) else 1
    if int(pp) != int(expected_pp):
        return "source_parallel_contradicts_deployment"
    return ""


def _size_list(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    sizes: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        sizes.append(item)
    return sizes


def deployment_dispatch(status: Mapping[str, Any] | None) -> str:
    """Cudagraph mode when compilation_used repeats spec.compilation_config.

    A missing or contradictory check leaves dispatch unset so static lanes
    stay conditional.
    """
    spec = _as_map(_as_map(status).get("spec"))
    comp = _as_map(spec.get("compilation_config"))
    mode = str(comp.get("cudagraph_mode") or "")
    sizes = _size_list(comp.get("cudagraph_capture_sizes"))
    used = _as_map(status).get("compilation_used")
    if isinstance(used, str):
        try:
            used = json.loads(used)
        except ValueError:
            return ""
    if not isinstance(used, Mapping) or not mode or sizes is None:
        return ""
    if str(used.get("cudagraph_mode") or "") != mode:
        return ""
    if _size_list(used.get("cudagraph_capture_sizes")) != sizes:
        return ""
    return mode


def lanes_from_bound_deployment(raw: Mapping[str, Any] | None,
                                status: Mapping[str, Any] | None, *,
                                tp: int = 0, pp: int = 1, dp: int = 1,
                                capture_sizes: tuple[int, ...] | list[int] = (),
                                cell_id: str = "") -> dict[str, Any]:
    """Lane kwargs for one bound deploy record.

    An admitted status contributes its switch and its checked cudagraph mode.
    A missing, overwritten, or history-only record leaves ``raw`` unchanged.
    A scope miss does not contribute the neighbor deployment's flags, and
    it does not discard the bound model.
    """
    if deployment_reject(status, cell_id=cell_id, tp=tp, dp=dp, pp=pp) or not isinstance(status, Mapping) or not status:
        return lanes_from_case(raw, tp=tp, pp=pp, capture_sizes=capture_sizes)
    spec = _as_map(status.get("spec"))
    merged = dict(_as_map(raw))
    runtime = _as_map(merged.get("runtime"))
    meta = _as_map(runtime.get("metadata"))
    additional = spec.get("additional_config")
    if isinstance(additional, Mapping):
        meta["additional_config"] = dict(additional)
    mode = deployment_dispatch(status)
    if mode:
        meta["captured_execution_mode"] = mode
    runtime["metadata"] = meta
    merged["runtime"] = runtime
    return lanes_from_case(merged, tp=tp, pp=pp, capture_sizes=capture_sizes)


def _dispatch_name(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"none", "off", "eager", "disabled"}:
        return "none"
    if mode in {"full", "full_decode_only", "piecewise"}:
        return mode
    if mode == "full_and_piecewise":
        return "piecewise"
    return "none"


def lanes_from_case(raw: Mapping[str, Any] | None, *, tp: int = 0, pp: int = 1,
                    capture_sizes: tuple[int, ...] | list[int] = ()) -> dict[str, Any]:
    """Kwargs for WorkloadGeometry. Only keys present on the case are returned.

    ``source_model_type`` wins when present. Otherwise the bound weight
    config's ``text_config.model_type`` is the language type, and the outer
    ``hf_config.model_type`` is last. The catalog nickname is not read.
    ``graph_capture_mode`` is not read, because its schema default is
    ``full`` even when ``graph_mode`` is ``none``.
    """
    runtime = _as_map(_as_map(raw).get("runtime"))
    meta = _as_map(runtime.get("metadata"))
    additional = _as_map(meta.get("additional_config"))
    env = _as_map(meta.get("env"))
    model = _as_map(_as_map(raw).get("model"))
    hf = _as_map(model.get("hf_config"))
    fields: dict[str, Any] = {}
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 0:
        raise ModelingError(f"workload_geometry_invalid:tp_width")
    if tp > 0:
        fields["tp_width"] = tp
    sizes = tuple(int(size) for size in capture_sizes if int(size) > 0)
    if not sizes:
        declared = runtime.get("graph_buckets") or runtime.get("cudagraph_capture_sizes")
        if isinstance(declared, (list, tuple)):
            sizes = tuple(int(size) for size in declared if int(size) > 0)
    if sizes:
        fields["capture_sizes"] = sizes
    if "captured_execution_mode" in meta:
        fields["graph_dispatch"] = _dispatch_name(meta["captured_execution_mode"])
    elif "graph_mode" in runtime:
        fields["graph_dispatch"] = _dispatch_name(runtime["graph_mode"])
    switch = None
    present = False
    if "enable_flashcomm1" in additional:
        present, switch = True, additional["enable_flashcomm1"]
    elif "VLLM_ASCEND_ENABLE_FLASHCOMM1" in meta:
        present, switch = True, meta["VLLM_ASCEND_ENABLE_FLASHCOMM1"]
    elif "VLLM_ASCEND_ENABLE_FLASHCOMM1" in env:
        present, switch = True, env["VLLM_ASCEND_ENABLE_FLASHCOMM1"]
    if present:
        model_type = _language_model_type(meta, hf)
        fields["dense_sequence_parallel"] = bool(
            _switch_on(switch) and model_type == "qwen3_5_text"
            and int(tp) > 1 and int(pp) == 1)
    return fields
