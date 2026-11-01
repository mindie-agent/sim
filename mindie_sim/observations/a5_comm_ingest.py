"""Normalize A5 campaign artifacts without promoting unconfirmed expansion.

[Role] Read embedded, in-memory or on-disk A5 campaign records into flat
  steady-state conditions, single-launch anchors and MC2 chain rows.
[Invariants] Status/canary failures raise.  Observed/profiler evidence alone
  may confirm AICPU/AIV; a request alone never does.  single=True rows never
  enter steady-state training.  Software stacks are never pooled.
[Timeouts & Locks] Bounded synchronous JSON reads; files above 50 MiB are
  delegated to the chunked binding stream.  No locks or subprocesses.
[Side-Effect Policy] Read-only over an explicit source; never launches a device
  and never writes campaign data.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from itertools import chain
from pathlib import Path
from statistics import median
from typing import Mapping
from mindie_sim.schema.common import ModelingError
from .a5_comm_golden import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU,
    EMBEDDED_A5_CAMPAIGNS, EMBEDDED_A5_MC2_CAMPAIGNS,
)
_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_CANARY_OK = frozenset({"all_ranks_matched", "ok", "passed", "matched", "match", "completed"})
_LARGE_JSON_BYTES = 50 * 1024 * 1024
_SCOPES = {"intra_node_ub": "intra_node_ub", "intra": "intra_node_ub", "intra_node": "intra_node_ub", "ub": "intra_node_ub", "cross_node_urma": "cross_node_urma", "cross_node": "cross_node_urma",
           "cross": "cross_node_urma", "urma": "cross_node_urma", "cross_node_urma_far": "cross_node_urma_far", "far": "cross_node_urma_far"}
_PAIR_PARTS = {"0->1", "0->4", "0->8", "0->24"}
_KNOWN_A5_ARTIFACTS = tuple((
    "2026-10-21/comm-rack06/a5-comm-rack06-calibration.json 2026-10-21/comm-rack08/a5-comm-rack08-calibration.json 2026-10-22/comm-w16/artifacts/a5-comm-w16-calibration.json 2026-10-22/comm-w32/artifacts/a5-comm-w32-calibration.json "
    "2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep2.json 2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep4.json 2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep2.json 2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep4.json 2026-10-22/mc2-quant/results/a5-mc2-quant-chain-validation.json 2026-10-22/mc2-quant/results/a5-mc2-quant-comm-lane.json").split())
class _Resolution(dict):
    """Mapping whose resolved fields are also readable as attributes."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
def _normalized_text(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())
def _is_a5(hardware_key, arch_family=None):
    if arch_family is not None and str(arch_family).strip().upper() != A5_ARCH_FAMILY:
        return False
    key = str(hardware_key or "").strip()
    if key in {A5_HARDWARE_KEY, A5_SKU}:
        return True
    folded = key.lower()
    return (folded in {A5_HARDWARE_KEY.lower(), A5_SKU.lower(), "a5",
                       "ascend950dt"} or folded.startswith("ascend950dt_9582"))
def _require_a5(hardware_key, arch_family=None):
    if not _is_a5(hardware_key, arch_family):
        raise ModelingError("a5_calibration_rejects_non_a5_hardware")
def _mode(value):
    text = str(value or "").strip().lower()
    if text in {"", "unspecified", "not_observed", "default", "auto", "none"}: return "unspecified"
    return ("aicpu" if text in {"aicpu", "ai_cpu", "cpu"} else "aiv"
            if text in {"aiv", "ai_vector_core", "vector", "aivector"} else text)
def _observed_mode(value):
    return text if (text := str(value or "").strip().lower()) in _EXPANSION_MODES else "not_observed"
def _profiler_modes(evidence):
    """Extract explicitly named expansion engines from profiler text."""
    modes, stack = set(), [evidence]
    while stack:
        if modes >= _EXPANSION_MODES:
            break
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                stack.extend((key, value))
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
            continue
        if item is None or item is True or item is False:
            continue
        text = _normalized_text(item)
        if "aicpu" in text:
            modes.add("aicpu")
        if "aiv" in text or "aivectorcore" in text:
            modes.add("aiv")
    return modes
def resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode, *,
                             profiler_evidence=None):
    """Resolve one expansion regime without promoting an unconfirmed request."""
    raw_req, observed = _mode(requested_expansion_mode), _observed_mode(observed_expansion_mode)
    requested = raw_req if raw_req in _EXPANSION_MODES else "unspecified"
    evidence = _profiler_modes(profiler_evidence)
    regime, conflict = "unspecified", False
    if observed in _EXPANSION_MODES:
        regime = observed
        conflict = bool(evidence and evidence != {observed})
    elif len(evidence) == 1:
        regime = next(iter(evidence))
    elif evidence: conflict = True
    confirmed = regime in _EXPANSION_MODES and not conflict
    result = {
        "requested_expansion_mode": requested, "observed_expansion_mode": observed,
        "regime": regime if confirmed else "unspecified",
        "expansion_unconfirmed": not confirmed,
        "profiler_confirmed": bool(evidence) and regime in evidence and not conflict,
        "profiler_modes": sorted(evidence), "profiler_evidence": profiler_evidence,
        "confirmed_by": ("observed" if observed in _EXPANSION_MODES and not conflict
                         else "profiler" if confirmed else None),
        "evidence_conflict": conflict,
        "requested_mismatch": (observed in _EXPANSION_MODES and requested in _EXPANSION_MODES
                               and observed != requested)}
    result["unconfirmed_tag"] = (None if confirmed else
                                 f"requested_{requested}_unconfirmed"
                                 if requested in _EXPANSION_MODES else None)
    return _Resolution(result)
def _collective(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "allreduce": "all_reduce", "all_reduce_only": "all_reduce",
        "allgather": "all_gather", "all_gather_only": "all_gather",
        "reduce_scatter_only": "reduce_scatter", "reduce_scatter_v": "reduce_scatter",
        "alltoall": "all_to_all", "all_to_all_only": "all_to_all",
        "alltoallv": "all_to_allv", "all_to_allv_only": "all_to_allv",
        "p2p": "p2p_send_recv", "p2p_only": "p2p_send_recv",
        "send_recv": "p2p_send_recv", "point_to_point": "p2p_send_recv"}
    return aliases.get(text, text)
def _dtype(value):
    text = str(value or "bf16").strip().lower().replace("_", "-")
    if text in {"fp8", "fp8e4m3", "fp8-e4m3", "e4m3", "fp8-e4m3fn"}:
        return "fp8-e4m3"
    if text in {"int8", "w8a8", "s8"}:
        return "int8"
    if text in {"bf16", "bfloat16"}:
        return "bf16"
    return text
def _explicit_scope(value):
    text = str(value or "").strip().lower()
    if text in _SCOPES:
        return _SCOPES[text]
    for token, scope in (("far", "cross_node_urma_far"), ("intra", "intra_node_ub"),
                         ("cross", "cross_node_urma"), ("urma", "cross_node_urma"),
                         ("2node", "cross_node_urma"), ("4node", "cross_node_urma")):
        if token in text:
            return scope
    return None
def _scope(value, world_size):
    return _explicit_scope(value) or ("cross_node_urma" if int(world_size) > 8 else "intra_node_ub")
def _explicit_pair(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"{int(value[0])}->{int(value[1])}"
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        source, destination = value.get("source"), value.get("destination")
        if source is not None and destination is not None:
            return f"{int(source)}->{int(destination)}"
    text = str(value or "").strip().lower()
    if text in _PAIR_PARTS:
        return text
    compact = "".join(text.replace("->", ">").replace("-", ">").replace(":", ">")
                      .replace("_", ">").split())
    if ">" not in compact:
        return None
    left, right = compact.split(">", 1)
    return f"{left}->{right}" if left.isdigit() and right.isdigit() else None
def _pair(value, collective, scope, world_size):
    if collective != "p2p_send_recv":
        return None
    if explicit := _explicit_pair(value):
        return explicit
    text = str(value or "").strip().lower()
    if "far" in text or scope == "cross_node_urma_far":
        return "0->24"
    if scope == "intra_node_ub": return "0->1"
    if "cross" in text or int(world_size) > 8:
        return "0->8"
    if "intra" in text:
        return "0->1"
    return "0->8" if scope.startswith("cross") else "0->1"
def _pair_scope(pair, world_size):
    if pair == "0->1":
        return "intra_node_ub"
    if pair == "0->24":
        return "cross_node_urma_far"
    return "cross_node_urma" if pair and int(world_size) > 8 else None
def _split(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"train", "heldout"}:
        return text
    raise ValueError("service_sample_split_requires_train_or_heldout")
def _protocol_key(value):
    if value is None:
        return None
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if isinstance(value, Mapping) else str(value))
def _same_canary(expected, observed):
    if expected is None or observed is None:
        return True
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return all(key in observed and _same_canary(value, observed[key])
                   for key, value in expected.items())
    if (isinstance(expected, (int, float)) and isinstance(observed, (int, float))
            and not isinstance(expected, bool) and not isinstance(observed, bool)):
        return float(expected) == float(observed)
    return expected == observed or str(expected) == str(observed)
def _canary_issue(value):
    if value is None or value is True:
        return None
    if value is False:
        return "canary_failed"
    if isinstance(value, str):
        return None if value.strip().lower() in _CANARY_OK else f"canary={value!r}"
    if isinstance(value, Mapping):
        expected, observed = value.get("expected"), value.get("observed")
        if expected is not None and not _same_canary(expected, observed):
            return f"canary={expected!r}!={observed!r}"
    return None
def _status_issue(document):
    for key in ("status", "state"):
        if key in document and str(document[key]).strip().lower() not in {"completed", "ok"}:
            return f"{key}={document[key]!r}"
    if any(str(status).strip().lower() not in {"completed", "ok"}
           for status in document.get("rank_status") or ()):
        return f"rank_status={document['rank_status']!r}"
    for key in ("runs", "rank_reports", "cases"):
        children = document.get(key) or ()
        for child in children if isinstance(children, (list, tuple)) else ():
            if isinstance(child, Mapping) and (issue := _status_issue(child)):
                return issue
    return None
def _validate_row(document, label):
    if (issue := _status_issue(document)):
        raise ValueError(f"a5_campaign_not_completed:{label}:{issue}")
    for key in ("canary_ok", "canary_passed"):
        if key in document and document[key] is not True:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}={document[key]!r}")
    for key in ("canary", "correctness"):
        issue = _canary_issue(document.get(key))
        if issue:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}:{issue}")
    for target in (document, document.get("machine"), document.get("hardware"), document.get("stack_binding")):
        if isinstance(target, Mapping) and any(key in target for key in ("hardware_key", "sku", "arch_family", "key", "architecture")):
            _require_a5(target.get("hardware_key") or target.get("sku") or target.get("key") or A5_HARDWARE_KEY,
                        target.get("arch_family") or target.get("architecture"))
_validate_campaign = _validate_row
def _load_json(path):
    path = Path(path)
    try:
        size = path.stat().st_size
    except TimeoutError:
        raise
    except OSError as exc:
        raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
    if size > _LARGE_JSON_BYTES:
        from .a5_comm_binding import _stream_json
        try:
            return _stream_json(path)
        except TimeoutError:
            raise
        except Exception as exc:
            raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except TimeoutError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
def _row_duration(row):
    for key in ("median_ms", "per_op_ms", "duration_ms", "ms", "per_op_ms_summary"):
        val = row.get(key)
        if isinstance(val, Mapping):
            val = val.get("median")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    if row.get("samples"):
        vals = [v for s in row["samples"] if isinstance(s, Mapping) and (v := _row_duration(s)) is not None]
        if vals: return float(median(vals))
    raw = [float(x) / 1000.0 for x in (row.get("raw_iterations_us") or ())
           if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return float(median(raw)) if raw else None
_EXPAND_KEYS = frozenset({
    "split", "run_id", "repeat_index", "single", "boundary", "observed_expansion_mode",
    "observed_execution", "requested_expansion_mode", "profiler_evidence", "stack",
    "cann", "algorithm", "protocol", "machine", "stack_binding",
    "payload_bytes", "size_bytes", "work", "dtype"})
def _selectors(document, row=None):
    row = row or {}
    row_proto, doc_proto = row.get("protocol"), document.get("protocol")
    return {
        "stack": row.get("stack", document.get("stack")),
        "cann": (row.get("cann") or (row.get("machine") or {}).get("cann") or (row.get("stack_binding") or {}).get("cann")
                 or document.get("cann") or (document.get("machine") or {}).get("cann") or (document.get("stack_binding") or {}).get("cann")),
        "algorithm": (row.get("algorithm") or (row_proto.get("algorithm") if isinstance(row_proto, Mapping) else None)
                      or document.get("algorithm") or (doc_proto.get("algorithm") if isinstance(doc_proto, Mapping) else None)),
        "protocol": row_proto if "protocol" in row else doc_proto}
def _merge_obs(parent, child):
    merged = {**parent, **child}
    if "observed_expansion_mode" in child or "observed_execution" in child:
        merged["observed_expansion_mode"] = child.get("observed_expansion_mode", child.get("observed_execution"))
    sel = _selectors(child)
    for key in ("stack", "cann", "algorithm"):
        if sel[key] is not None:
            merged[key] = sel[key]
    return merged
def _base_row(document, label, origin, collective, world, scope, pair, payload,
              duration, dtype, single, row=None, requested=None, observed=None,
              evidence=None, run_id=None, repeat_index=0, split=None):
    row = row or {}
    if (boundary := row.get("boundary", document.get("boundary"))) not in (None, "device_kernel", "resource_phase"):
        raise ModelingError("service_curve_requires_device_or_resource_phase_boundary")
    resolved = resolve_expansion_regime(
        row.get("requested_expansion_mode", requested),
        row.get("observed_expansion_mode", row.get("observed_execution", observed)),
        profiler_evidence=row.get("profiler_evidence", evidence))
    return {
        "campaign": label, "collective": collective, "world_size": int(world),
        "placement_scope": scope, "dtype": _dtype(row.get("dtype") or dtype),
        "payload_bytes": int(payload), "per_op_ms": float(duration), "single": bool(single), "pair": pair,
        **{key: resolved[key] for key in (
            "observed_expansion_mode", "requested_expansion_mode", "regime",
            "expansion_unconfirmed", "unconfirmed_tag", "profiler_confirmed")},
        "profiler_evidence": row.get("profiler_evidence", evidence),
        **_selectors(document, row), "canary": row.get("canary", row.get("correctness")),
        "run_id": run_id or label, "repeat_index": int(repeat_index or 0),
        "split": split, "source": origin, "boundary": boundary,
        **{k: row[k] for k in ("_repeats", "_weights") if k in row}}
def _rank_rows(document, origin, p2p_pair=None, default_run_id=None, default_repeat=0):
    protocol = document.get("protocol") if isinstance(document.get("protocol"), Mapping) else {}
    dtype = _dtype(document.get("dtype") or protocol.get("dtype"))
    world = document.get("world_size") or len(str(document.get("visible_devices", "")).split(","))
    placement = document.get("placement", "")
    default_scope = _explicit_scope(document.get("placement_scope"))
    observed, requested, evidence = document.get("observed_expansion_mode", document.get("observed_execution", "not_observed")), document.get("requested_expansion_mode", "unspecified"), document.get("profiler_evidence")
    rows = []
    for case in document.get("cases") or ():
        _validate_row(case, f"{origin}:{case.get('case_id')}")
        collective = _collective(case.get("collective") or case.get("mode"))
        declared_scope = (_explicit_scope(case.get("placement_scope")) or default_scope)
        scope = (declared_scope or _explicit_scope(case.get("placement"))
                 or _scope(case.get("placement") or placement, world))
        explicit = (_explicit_pair(case.get("p2p_pair")) or _explicit_pair(case.get("pair")) or _explicit_pair(document.get("p2p_pair")) or _explicit_pair(p2p_pair))
        pair = explicit if collective == "p2p_send_recv" else None
        if collective == "p2p_send_recv":
            scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
            pair = pair or _pair(case.get("p2p_scope") or protocol.get("p2p_scope"), collective, scope, world)
        run_id = (case.get("run_id") or document.get("run_id") or default_run_id
                  or Path(origin).stem or document.get("label") or "rank")
        repeat = case.get("repeat_index", document.get("repeat_index", default_repeat))
        split = _split(case.get("split", document.get("split")))
        samples = case.get("samples") or ()
        for index, sample in enumerate(samples):
            if isinstance(sample, Mapping):
                _validate_row(sample, f"{origin}:{case.get('case_id')}:sample[{index}]")
        expand = bool(samples) and any(isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample)) for sample in samples)
        if expand:
            inh_rep = "repeat_index" in case or "repeat_index" in document or default_repeat != 0
            for index, sample in enumerate(samples):
                duration, sample_payload = _row_duration(sample), sample.get("payload_bytes", sample.get("size_bytes", sample.get("work", case.get("payload_bytes", case.get("size_bytes", case.get("work"))))))
                if duration is None or sample_payload is None:
                    continue
                reps, rep = sample.get("_repeats"), sample.get("repeat_index")
                if rep is None and reps:
                    if inh_rep:
                        if len(reps[0]) > 1 or reps[1] or (reps[0] and reps[0][0][2] in (None, Ellipsis)):
                            raise ValueError("conflicting_primitive_repeat_or_split")
                        rep = int(repeat or 0)
                        sample = {**sample, "_repeats": ([[rep, rep, reps[0][0][2]]], {}) if reps[0] else reps}
                    else:
                        rep = reps[0][0][0] if reps[0] else index
                elif rep is None:
                    rep = repeat if inh_rep else index
                rows.append(_base_row(
                    document, document.get("label") or Path(origin).stem, origin, collective, world, scope, pair, sample_payload, duration, dtype,
                    sample.get("single", case.get("single", document.get("single"))), _merge_obs(case, sample), run_id=sample.get("run_id", run_id),
                    repeat_index=rep, split=_split(sample.get("split", split)), requested=requested, observed=observed, evidence=evidence))
            continue
        duration = _row_duration(case)
        payload = case.get("payload_bytes", case.get("size_bytes", case.get("work")))
        if duration is None or payload is None:
            raise ValueError(f"a5_campaign_case_has_no_median:{origin}:{case.get('case_id')}")
        rows.append(_base_row(document, document.get("label") or Path(origin).stem, origin,
                              collective, world, scope, pair, payload, duration, dtype,
                              case.get("single", document.get("single")), case, run_id=run_id,
                              repeat_index=repeat, split=split, requested=requested, observed=observed, evidence=evidence))
    return rows
def _parent_document(path, document):
    origin = str(path) if path is not None else "memory"
    stem = (path.name[:-len(".parent.json")] if path is not None and path.name.endswith(".parent.json")
            else path.stem if path is not None else str(document.get("label") or document.get("run_id") or "parent"))
    rank_files = sorted(path.parent.glob(stem + ".rank*.json")) if path is not None else ()
    reports = [(report, origin) for report in document.get("rank_reports") or () if isinstance(report, Mapping)]
    if not rank_files and not reports:
        return [(document, origin)]
    protocol, placement = document.get("protocol") or {}, document.get("placement", "")
    world = document.get("world_size") or 1
    pair = _explicit_pair(document.get("p2p_pair")) or _pair(protocol.get("p2p_scope"), "p2p_send_recv", _scope(placement, world), world)
    grouped = defaultdict(list)
    parent_run = str(document.get("run_id") or document.get("label") or stem)
    for rank_document, rank_origin in chain(((_load_json(item), str(item)) for item in rank_files), reports):
        rank_document = _merge_obs({key: document[key] for key in ("observed_expansion_mode", "observed_execution", "requested_expansion_mode", "profiler_evidence", "stack", "cann", "algorithm", "protocol", "machine", "stack_binding", "split", "boundary", "single") if key in document}, rank_document)
        _validate_campaign(rank_document, rank_origin)
        for row in _dedupe_conditions(_rank_rows(rank_document, rank_origin, pair, default_run_id=parent_run,
                                                 default_repeat=document.get("repeat_index", 0)), keep_meta=True):
            grouped[(row["collective"], row["payload_bytes"], bool(row["single"]),
                     row["split"], row["pair"], row["dtype"], row["placement_scope"],
                     row["regime"], row.get("boundary"), row["run_id"],
                     row["repeat_index"], row.get("stack"), row.get("cann"),
                     row.get("algorithm"), _protocol_key(row.get("protocol")))].append(row)
    rows = [{**{k: v for k, v in group[0].items() if k not in {"_repeats", "_weights"} or len(group) == 1},
             "campaign": document.get("label") or stem, "per_op_ms": median(row["per_op_ms"] for row in group),
             **({"_weights": [(median(row["per_op_ms"] for row in group), sum(w for _, w in group[0]["_weights"]))]}
                if len(group) > 1 and "_weights" in group[0] else {}),
             "canary": "all_ranks_matched", "source": origin}
            for key, group in sorted(grouped.items(), key=str)]
    return [({**document, "conditions": rows, "label": document.get("label") or stem}, origin)]
def _list_document(values, origin):
    if not values:
        return []
    if all(isinstance(value, Mapping) for value in values):
        keys, base = set(values[0]), {"label": Path(origin).stem, "status": "completed",
                                      "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
                                      "arch_family": A5_ARCH_FAMILY}
        if {"condition", "legs_ms"} <= keys or {"chain_ms", "legs_ms"} <= keys:
            return [({**base, "chains": values}, origin)]
        if {"payload_bytes", "per_op_ms"} <= keys or {"size_bytes", "median_ms"} <= keys:
            return [({**base, "conditions": values}, origin)]
        if any({"label", "conditions", "series", "chains", "families"} & set(value)
               for value in values):
            return [(value, origin) for value in values]
    raise ValueError(f"a5_campaign_source_list_not_a_campaign:{origin}")
def _expand_path(path):
    document = _load_json(path)
    if isinstance(document, list):
        return _list_document(document, str(path))
    if not isinstance(document, Mapping):
        raise ValueError(f"a5_campaign_source_requires_json_object:{path}")
    if (document.get("role") == "parent" or path.name.endswith(".parent.json")
            or ("rank_reports" in document and "median_ms_by_case" not in document)):
        return _parent_document(path, document)
    if document.get("role") == "rank" or ("cases" in document
                                          and "median_ms_by_case" not in document):
        label = document.get("label") or path.stem
        return [({**document, "conditions": _rank_rows(
            document, str(path), _explicit_pair(document.get("p2p_pair"))), "label": label},
            str(path))]
    return [(document, str(path))]
def _known_root(root):
    return "mindie-sim-data" in {part.lower() for part in Path(root).parts} and Path(root).name.lower() == "a5"
def _directory_documents(root):
    root = Path(root)
    if _known_root(root):
        files = [root / relative for relative in _KNOWN_A5_ARTIFACTS]
        files = [path for path in files if path.is_file()]
        if files:
            return [document for path in files for document in _expand_path(path)]
    calibration, chains, comms, parents, ranks, others = [], [], [], [], [], []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("metadata_"):
            continue
        try:
            size = path.stat().st_size
        except TimeoutError:
            raise
        except OSError as exc:
            raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
        name = path.name.lower()
        if "calibration" in name:
            calibration.append(path)
        elif "chain-validation" in name or "chain_validation" in name:
            chains.append(path)
        elif "comm-lane" in name or "comm_lane" in name:
            comms.append(path)
        elif name.endswith(".parent.json"):
            parents.append(path)
        elif ".rank" in name:
            ranks.append(path)
        elif size <= 8 * 1024 * 1024:
            others.append(path)
    files = calibration + chains + comms
    if not calibration:
        shadowed = {(path.parent, path.name[:-len(".parent.json")]) for path in parents}
        files += parents
        files += [path for path in ranks if (path.parent, path.name.split(".rank")[0]) not in shadowed]
    files = files or others
    if not files: raise ValueError(f"a5_campaign_source_has_no_json:{root}")
    return [document for path in files for document in _expand_path(path)]
def _source_documents(source):
    if source is None:
        embedded = tuple(EMBEDDED_A5_CAMPAIGNS.values()) + tuple(EMBEDDED_A5_MC2_CAMPAIGNS.values())
        return [(_campaign, _campaign.get("source")
                 or f"embedded:{_campaign.get('label', 'a5')}") for _campaign in embedded]
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            return _directory_documents(path)
        if path.is_file():
            return _expand_path(path)
        raise ValueError(f"a5_campaign_source_missing:{path}")
    if isinstance(source, Mapping):
        if source.get("role") == "parent" or ("rank_reports" in source and "median_ms_by_case" not in source):
            return _parent_document(None, source)
        if source.get("role") == "rank" or ("cases" in source and "median_ms_by_case" not in source
                                            and "conditions" not in source):
            label = source.get("label") or source.get("run_id") or "rank"
            return [({**source, "conditions": _rank_rows(
                source, "memory", _explicit_pair(source.get("p2p_pair"))),
                "label": label}, "memory")]
        if source and all(isinstance(value, Mapping)
                          and ({"label", "conditions", "series", "chains", "rank_reports", "cases"} & set(value))
                          for value in source.values()):
            return [doc for value in source.values() for doc in _source_documents(value)]
        if "legs_ms" in source and ("condition" in source or "chain_ms" in source):
            return _list_document([source], "memory")
        if ({"payload_bytes", "per_op_ms"} <= set(source)
                or {"size_bytes", "median_ms"} <= set(source)):
            return _list_document([source], "memory")
        return [(source, "memory")]
    if isinstance(source, (list, tuple)):
        if source and all(isinstance(value, Mapping) for value in source) and any(
                {"label", "conditions", "series", "chains", "families", "rank_reports", "cases"} & set(value) for value in source):
            return [doc for value in source for doc in _source_documents(value)]
        return _list_document(list(source), "memory")
    raise ValueError("a5_campaign_source_requires_path_or_mapping")
def _condition_rows(row, document, label, origin, default_dtype, default_world,
                    default_scope, default_pair, default_observed, default_requested,
                    default_evidence):
    _validate_row(row, f"{label}:{row.get('collective', row.get('mode'))}")
    collective = _collective(row.get("collective") or row.get("mode"))
    world = int(row.get("world_size") or row.get("ranks") or row.get("group_size") or default_world or 1)
    placement = row.get("placement") or document.get("placement", "")
    declared_scope = _explicit_scope(row.get("placement_scope")) or default_scope
    scope = declared_scope or _explicit_scope(placement) or _scope("", world)
    explicit = _explicit_pair(row.get("p2p_pair")) or _explicit_pair(row.get("pair"))
    pair = explicit if collective == "p2p_send_recv" else None
    if collective == "p2p_send_recv":
        scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
        pair = pair or default_pair or _pair(row.get("p2p_scope"), collective, scope, world)
    payload = row.get("payload_bytes", row.get("size_bytes", row.get("work")))
    duration = _row_duration(row)
    run_id = row.get("run_id") or document.get("run_id") or label
    repeat = row.get("repeat_index", document.get("repeat_index", 0))
    split = _split(row.get("split", document.get("split")))
    samples = row.get("samples") or ()
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            _validate_row(sample, f"{label}:sample[{index}]")
    expand = bool(samples) and any(isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample)) for sample in samples)
    if not expand and (payload is None or duration is None):
        raise ValueError(f"a5_campaign_condition_missing_payload_or_duration:{label}")
    if expand:
        result = []
        for index, sample in enumerate(samples):
            duration, sample_payload = _row_duration(sample), sample.get("payload_bytes", sample.get("size_bytes", sample.get("work", payload)))
            if duration is None or sample_payload is None:
                continue
            result.append(_base_row(
                document, label, origin, collective, world, scope, pair, sample_payload, duration, default_dtype,
                sample.get("single", row.get("single", document.get("single"))), _merge_obs(row, sample), run_id=sample.get("run_id", run_id),
                repeat_index=sample.get("repeat_index", sample["_repeats"][0][0][0] if sample.get("_repeats") and sample["_repeats"][0] else index),
                split=_split(sample.get("split", split)), requested=default_requested, observed=default_observed, evidence=default_evidence))
        return result
    return [_base_row(document, label, origin, collective, world, scope, pair, payload, duration, default_dtype, row.get("single", document.get("single")),
                      row, run_id=run_id, repeat_index=repeat, split=split, requested=default_requested, observed=default_observed, evidence=default_evidence)]
def _add_series(rows, document, label, origin, dtype, common, series, single=False):
    collective, world, scope, pair, points = series
    collective = _collective(collective)
    pair = pair or _pair("", collective, scope, world)
    for payload, duration in points:
        rows.append(_base_row(document, label, origin, collective, world, scope, pair, payload, duration, dtype, bool(single or document.get("single")), **common))
def _campaign_conditions(document, label, origin):
    protocol = document.get("protocol") if isinstance(document.get("protocol"), Mapping) else {}
    dtype, world, placement, scope = _dtype(document.get("dtype") or protocol.get("dtype")), document.get("world_size") or document.get("ranks"), document.get("placement", ""), _explicit_scope(document.get("placement_scope"))
    observed = document.get("observed_expansion_mode", document.get("observed_execution", "not_observed"))
    requested, evidence = document.get("requested_expansion_mode", "unspecified"), document.get("profiler_evidence")
    common = {"run_id": document.get("run_id") or label, "repeat_index": document.get("repeat_index", 0),
              "split": _split(document.get("split")), "requested": requested, "observed": observed, "evidence": evidence}
    rows = []
    for row in document.get("conditions") or document.get("comm_conditions") or ():
        rows.extend(_condition_rows(row, document, label, origin, dtype, world, scope, None, observed, requested, evidence))
    for series in document.get("series") or ():
        _add_series(rows, document, label, origin, dtype, common, series)
    raw_points = document.get("raw_points") or ()
    if raw_points:
        lane = {}
        for index, point in enumerate(raw_points):
            _validate_row(point, f"{label}:raw_points[{index}]")
            observed_point = point.get("observed_expansion_mode", point.get("observed_execution"))
            resolved_point = resolve_expansion_regime(
                point.get("requested_expansion_mode", requested),
                point.get("observed_expansion_mode", point.get("observed_execution", observed)),
                profiler_evidence=point.get("profiler_evidence", evidence))
            sel = _selectors(document, point)
            key = (_dtype(point.get("dtype")), int(point.get("ep")), int(point.get("link_bytes")),
                   bool(point.get("single", document.get("single", False))), _split(point.get("split", document.get("split"))), point.get("run_id"),
                   point.get("repeat_index"), observed_point, point.get("requested_expansion_mode"), point.get("algorithm"), point.get("boundary", document.get("boundary")),
                   resolved_point["regime"], resolved_point["unconfirmed_tag"], resolved_point["profiler_confirmed"], (sel["stack"], sel["cann"], sel["algorithm"], _protocol_key(sel["protocol"])))
            lane.setdefault(key, (point, []))[1].append(float(point["ms"]))
        for key, (point, values) in sorted(lane.items(), key=str):
            (lane_dtype, series_world, payload, single, split, run_id, repeat_index, observed_point, requested_point, _, _, _, _, _, _) = key
            rows.append(_base_row(
                document, label, origin, "all_to_allv", series_world, _scope("", series_world), None, payload, median(values), lane_dtype,
                single, point, run_id=run_id or common["run_id"], repeat_index=repeat_index if repeat_index is not None else common["repeat_index"],
                split=split if split is not None else common["split"], requested=requested_point if requested_point is not None else common["requested"],
                observed=observed_point if observed_point is not None else common["observed"], evidence=point.get("profiler_evidence", evidence)))
    for lane_dtype, series_world, points in document.get("all_to_allv_series") or ():
        _add_series(rows, document, label, origin, lane_dtype, common, ("all_to_allv", series_world, _scope("", series_world), None, points))
    for series in document.get("single_anchors") or ():
        _add_series(rows, document, label, origin, dtype, common, series, True)
    return rows
def _chain_row(values, campaign, origin, default_dtype, default_observed, default_requested, default_evidence, default_run_id=None, default_repeat=0):
    if isinstance(values, (list, tuple)) and len(values) == 10:
        dtype, ep, experts, tokens, dispatch, gate, swiglu, down, combine, chain = values
        legs = {"dispatch": float(dispatch), "gate_up": float(gate), "swiglu_quant" if _dtype(dtype) != "bf16" else "swiglu": float(swiglu), "down": float(down), "combine": float(combine)}
        return _chain_record(campaign, origin, _dtype(dtype), int(ep), int(experts), int(tokens), legs, float(chain), f"{_dtype(dtype)}-ep{ep}-e{experts}-g{tokens}",
                             default_observed, default_requested, default_evidence, run_id=default_run_id, repeat_index=default_repeat)
    if not isinstance(values, Mapping):
        raise ValueError(f"a5_mc2_chain_row_invalid:{campaign}")
    row_status = values.get("status", values.get("state"))
    if row_status is not None and str(row_status).strip().lower() not in {"ok", "completed"}:
        raise ValueError(f"a5_mc2_chain_row_not_completed:{campaign}:{row_status!r}")
    _validate_row(values, f"{campaign}:{values.get('condition', values.get('chain_id'))}")
    legs = {str(key): float(value) for key, value in (values.get("legs_ms") or {}).items()}
    if not legs:
        raise ValueError(f"a5_mc2_chain_row_missing_legs:{campaign}")
    chain = values.get("chain_ms", values.get("chain_p50_ms", values.get("pipeline_ms")))
    if chain is None:
        raise ValueError(f"a5_mc2_chain_row_missing_chain:{campaign}")
    dtype, swiglu = _dtype(values.get("dtype") or default_dtype), legs.get("swiglu_quant", legs.get("swiglu"))
    normalized = {"dispatch": legs.get("dispatch", 0.0), "gate_up": legs.get("gate_up", legs.get("gmm_gate_up", 0.0)), "down": legs.get("down", legs.get("gmm_down", 0.0)), "combine": legs.get("combine", 0.0)}
    normalized["swiglu_quant" if dtype != "bf16" else "swiglu"] = swiglu or 0.0
    ep = values.get("ranks") or values.get("ep") or values.get("world_size") or 1
    return _chain_record(
        campaign, origin, dtype, int(ep), values.get("experts"), values.get("global_tokens"), normalized, float(chain),
        values.get("condition") or values.get("chain_id") or campaign, values.get("observed_expansion_mode", default_observed),
        values.get("requested_expansion_mode", default_requested), values.get("profiler_evidence", default_evidence),
        run_id=values.get("run_id") or default_run_id, repeat_index=int(values.get("repeat_index") if values.get("repeat_index") is not None else default_repeat), chain_id=values.get("chain_id") or None)
def _chain_record(campaign, origin, dtype, ep, experts, tokens, legs, chain, condition, observed="not_observed", requested="unspecified", evidence=None, *, run_id=None, repeat_index=0, chain_id=None):
    total = sum(legs.values())
    if not math.isfinite(chain) or chain <= 0 or total <= 0:
        raise ValueError(f"a5_mc2_chain_row_nonpositive:{campaign}:{condition}")
    resolved = resolve_expansion_regime(requested, observed, profiler_evidence=evidence)
    result = {
        "campaign": campaign, "chain_id": chain_id or f"{campaign}:{condition}", "condition": condition, "run_id": run_id or campaign, "repeat_index": int(repeat_index or 0),
        "dtype": dtype, "ep": ep, "world_size": ep, "experts": int(experts) if experts is not None else None, "global_tokens": int(tokens) if tokens is not None else None,
        "legs_ms": legs, "chain_p50_ms": chain, "sum_leg_p50_ms": total, "overlap_ratio": chain / total, "source": origin, "profiler_evidence": evidence,
        "stages": ("quant", "all_to_allv_dispatch", "grouped_matmul", "all_to_allv_combine"),
        "stage_legs_ms": {"quant": legs.get("swiglu_quant", legs.get("swiglu", 0.0)), "all_to_allv_dispatch": legs["dispatch"], "grouped_matmul": legs["gate_up"] + legs["down"], "all_to_allv_combine": legs["combine"]}}
    for key in ("requested_expansion_mode", "observed_expansion_mode", "regime", "expansion_unconfirmed", "unconfirmed_tag", "profiler_confirmed"):
        result[key] = resolved[key]
    return result
def _bf16_chains(document, campaign, origin, observed, requested, evidence, default_run_id=None, default_repeat=0):
    by_condition = defaultdict(dict)
    for case_id, stages in (document.get("median_ms_by_case") or {}).items():
        kind = case_id.rsplit("-", 1)[-1]
        by_condition[case_id[:-(len(kind) + 1)]][kind] = stages
    rows = []
    for condition in sorted(by_condition, key=lambda value: tuple(int(part[1:]) for part in value.split("-"))):
        stages = by_condition[condition]
        def pick(kind, stage):
            values = [value for value in (stages.get(kind, {}).get(stage) or {}).values() if isinstance(value, (int, float))]
            return median(values) if values else None
        dispatch, combine, gate, down, swiglu, chain = pick("dispatch", "dispatch"), pick("combine", "combine"), pick("gmm", "gmm_gate_up"), pick("gmm", "gmm_down"), pick("gmm", "swiglu"), pick("pipeline", "pipeline")
        if None in (dispatch, combine, gate, down, swiglu, chain):
            continue
        expert_text, token_text = condition.split("-")
        legs = {"dispatch": dispatch, "gate_up": gate, "swiglu": swiglu, "down": down, "combine": combine}
        rows.append(_chain_record(campaign, origin, _dtype(document.get("dtype") or "bf16"), int(document.get("ranks_requested") or 1),
                                  int(expert_text[1:]), int(token_text[1:]), legs, chain, condition, observed, requested, evidence, run_id=default_run_id, repeat_index=default_repeat))
    return rows
def _campaign_chains(document, label, origin):
    observed = document.get("observed_expansion_mode", document.get("observed_execution", "not_observed"))
    requested, evidence = document.get("requested_expansion_mode", "unspecified"), document.get("profiler_evidence")
    default_run_id, default_repeat = document.get("run_id") or label, int(document.get("repeat_index", 0) or 0)
    if "median_ms_by_case" in document:
        return _bf16_chains(document, label, origin, observed, requested, evidence, default_run_id, default_repeat)
    values = document.get("chains") or (document.get("provenance") or {}).get("chain_validation") or document.get("chain_validation")
    if not values: return []
    return [_chain_row(value, label, origin, _dtype(document.get("dtype") or "bf16"),
                       observed, requested, evidence, default_run_id, default_repeat) for value in values]
def _dedupe_conditions(rows, keep_meta=False):
    from .a5_comm_binding import _fold_timings, _weighted_median
    grouped, weights, metas, intervals = {}, {}, {}, defaultdict(list)
    for row in rows:
        reps, wts, sp = row.pop("_repeats", None), row.pop("_weights", None), row["split"]
        eff_sp = sp or "train"
        base = (row["campaign"], row["run_id"], row["collective"], row["world_size"],
                row["placement_scope"], row["dtype"], row["payload_bytes"], row["single"], row["pair"],
                row["regime"], row.get("unconfirmed_tag"), row.get("profiler_confirmed"), row.get("stack"),
                row.get("cann"), row.get("algorithm"), _protocol_key(row.get("protocol")), row.get("boundary"))
        spans = ([(lo, hi, (ms, eff_sp)) for lo, hi, ms in reps[0]]
                 + [(r, r, (ms, eff_sp)) for r, ms in reps[1].items()]
                 if reps else [(row["repeat_index"], row["repeat_index"], (row["per_op_ms"], eff_sp))])
        if any(lo <= p_hi and p_lo <= hi and (rec[1] != p_rec[1] or (
                rec[0] is not Ellipsis and p_rec[0] is not Ellipsis
                and (rec[0] is None or p_rec[0] is None or rec[0] != p_rec[0])))
               for lo, hi, rec in spans for p_lo, p_hi, p_rec in intervals[base]):
            raise ValueError("conflicting_primitive_repeat_or_split")
        if intervals[base] and all(rec[0] is not None and rec[0] is not Ellipsis and any(
                p_rec == rec and p_lo <= lo and hi <= p_hi for p_lo, p_hi, p_rec in intervals[base])
                for lo, hi, rec in spans):
            continue
        intervals[base].extend(spans)
        c_key = base + (None, sp)
        key = c_key if reps or c_key in weights else base + (row["repeat_index"], sp)
        if reps and c_key not in weights:
            prevs = [k for k in grouped if k[:-2] == base and k[-1] == sp]
            if prevs:
                wts = [(grouped[k]["per_op_ms"], 1) for k in prevs] + list(wts or ())
                row["repeat_index"] = min([row["repeat_index"]] + [grouped[k]["repeat_index"] for k in prevs])
                for k in prevs:
                    grouped.pop(k, None)
        if key == c_key:
            cur = weights.setdefault(c_key, [])
            for val, wt in (wts or ((row["per_op_ms"], 1),)):
                _fold_timings(cur, val, wt)
            row["per_op_ms"], row["_weights"] = _weighted_median(cur), list(cur)
            if keep_meta and reps:
                m = metas.setdefault(c_key, ([], {}))
                m[0].extend(reps[0])
                m[1].update(reps[1])
                row["_repeats"] = m
            if (prev := grouped.get(c_key)) is not None:
                prev["per_op_ms"], prev["_weights"] = row["per_op_ms"], row["_weights"]
                prev["repeat_index"] = min(prev["repeat_index"], row["repeat_index"])
                continue
        grouped[key] = row
    return [grouped[k] for k in sorted(grouped, key=lambda k: str(k[:2] + (grouped[k]["repeat_index"],) + k[2:-2] + (k[-1],)))]
def _dedupe_chains(rows):
    seen = {}
    for row in rows:
        key = (row["campaign"], row["condition"], row["dtype"], row["ep"], row["experts"], row["global_tokens"],
               round(row["chain_p50_ms"], 9), row["regime"], round(float(row["sum_leg_p50_ms"]), 9),
               row.get("run_id"), row.get("repeat_index"), row.get("chain_id"),
               tuple(sorted((name, round(float(value), 9)) for name, value in (row.get("legs_ms") or {}).items())))
        seen.setdefault(key, row)
    return [seen[key] for key in sorted(seen, key=str)]
def ingest_a5_campaign_data(source=None):
    """Normalize A5 campaigns into steady-state conditions and launch anchors."""
    campaigns, conditions, anchors, chains = [], [], [], []
    for document, origin in _source_documents(source):
        label = str(document.get("label") or document.get("run_id") or document.get("campaign") or Path(origin).stem)
        if document.get("schema_version") == "mindie-mc2-launch/1" and origin not in {"memory", "embedded"}:
            label = f"{Path(origin).parent.name}:{label}"
        _validate_campaign(document, label)
        _require_a5(document.get("hardware_key") or document.get("sku") or A5_HARDWARE_KEY, document.get("arch_family"))
        campaigns.append({
            "label": label, "status": document.get("status", document.get("state", "completed")),
            "kind": document.get("kind", "calibration"), "hardware_key": document.get("hardware_key", A5_HARDWARE_KEY),
            "arch_family": document.get("arch_family", A5_ARCH_FAMILY),
            "stack": document.get("stack"), "date": document.get("date"),
            "cann": document.get("cann") or (document.get("machine") or {}).get("cann"),
            "algorithm": document.get("algorithm") or ((document.get("protocol") or {}).get("algorithm")
                                                        if isinstance(document.get("protocol"), Mapping) else None),
            "protocol": document.get("protocol"), "placement": document.get("placement"),
            "nodes": list(document.get("nodes") or ()), "dtype": _dtype(document.get("dtype")),
            "observed_expansion_mode": _observed_mode(document.get("observed_expansion_mode")), "canary": document.get("canary"),
            "source": origin})
        for row in _campaign_conditions(document, label, origin):
            (anchors if row["single"] else conditions).append(row)
        chains.extend(_campaign_chains(document, label, origin))
    conditions, anchors = _dedupe_conditions(conditions), _dedupe_conditions(anchors)
    chains = _dedupe_chains(chains)
    campaigns.sort(key=lambda item: item["label"])
    return {
        "schema": "a5-campaign-ingestion/1", "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "campaigns": campaigns,
        "conditions": conditions, "rows": conditions, "single_op_anchors": anchors,
        "mc2_chains": chains, "campaign_labels": [campaign["label"] for campaign in campaigns]}
def _coerce_ingested(ingested):
    if ingested is None:
        return ingest_a5_campaign_data()
    if isinstance(ingested, Mapping) and ingested.get("schema") == "a5-campaign-ingestion/1":
        return ingested
    return ingest_a5_campaign_data(ingested)
