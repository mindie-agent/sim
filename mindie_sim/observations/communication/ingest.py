"""Normalize communication campaign artifacts into steady-state and chain rows.

[Role] Read embedded, in-memory, or on-disk communication campaign records into
  flat steady-state conditions, single-launch anchors, and MC2 chain rows.
[Invariants] Status/canary failures raise. Observed/profiler evidence alone may
  confirm AICPU/AIV; a request alone never does. ``single=True`` rows never
  enter steady-state training. Software stacks are never pooled.
[Timeouts & Locks] Bounded synchronous JSON reads; files above 50 MiB are
  delegated to the chunked streaming projection. No locks or subprocesses.
[Side-Effect Policy] Read-only over an explicit source; never launches a device
  and never writes campaign data.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import chain
import json
import os
from pathlib import Path
from statistics import median
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .chains import (
    _campaign_chains,
    _dedupe_chains,
    _dedupe_conditions,
)
from .golden import (
    A5_ARCH_FAMILY,
    A5_HARDWARE_KEY,
    EMBEDDED_A5_CAMPAIGNS,
    EMBEDDED_A5_MC2_CAMPAIGNS,
)
from .normalization import (
    _collective,
    _dtype,
    _explicit_pair,
    _explicit_scope,
    _fallback_measurement,
    _fallback_transport_window,
    _merge_obs,
    _observed_mode,
    _pair,
    _pair_scope,
    _protocol_key,
    _require_a5,
    _scope,
    _selectors,
    _split,
    _validate_campaign,
    _validate_measurement_window,
    _validate_row,
    resolve_expansion_regime,
)
from .streaming import _stream_json

_LARGE_JSON_BYTES = 50 * 1024 * 1024
_OTHER_JSON_MAX_BYTES = 8 * 1024 * 1024
_KNOWN_A5_ARTIFACTS = tuple((
    "2026-10-21/comm-rack06/a5-comm-rack06-calibration.json "
    "2026-10-21/comm-rack08/a5-comm-rack08-calibration.json "
    "2026-10-22/comm-w16/artifacts/a5-comm-w16-calibration.json "
    "2026-10-22/comm-w32/artifacts/a5-comm-w32-calibration.json "
    "2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep2.json "
    "2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep4.json "
    "2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep2.json "
    "2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep4.json "
    "2026-10-22/mc2-quant/results/a5-mc2-quant-chain-validation.json "
    "2026-10-22/mc2-quant/results/a5-mc2-quant-comm-lane.json"
).split())
_EXPAND_KEYS = frozenset({
    "split", "run_id", "repeat_index", "single", "boundary",
    "observed_expansion_mode", "observed_execution",
    "requested_expansion_mode", "profiler_evidence", "stack",
    "cann", "algorithm", "protocol", "machine", "stack_binding",
    "payload_bytes", "size_bytes", "work", "dtype",
    "measurement", "transport_window", "direction_observed",
    "algorithm_identified",
})

__all__ = [
    "ingest_a5_campaign_data",
    "resolve_expansion_regime",
]


def _load_json(path):
    path = Path(path)
    try:
        size = path.stat().st_size
    except TimeoutError:
        raise
    except OSError as exc:
        raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
    if size > _LARGE_JSON_BYTES:
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
        vals = [v for s in row["samples"]
                if isinstance(s, Mapping) and (v := _row_duration(s)) is not None]
        if vals:
            return float(median(vals))
    raw = [float(x) / 1000.0 for x in (row.get("raw_iterations_us") or ())
           if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return float(median(raw)) if raw else None


def _base_row(document, label, origin, collective, world, scope, pair, payload,
              duration, dtype, single, row=None, requested=None, observed=None,
              evidence=None, run_id=None, repeat_index=0, split=None):
    row = row or {}
    boundary = row.get("boundary", document.get("boundary"))
    if boundary not in (None, "device_kernel", "resource_phase"):
        raise ModelingError("service_curve_requires_device_or_resource_phase_boundary")
    resolved = resolve_expansion_regime(
        row.get("requested_expansion_mode", requested),
        row.get("observed_expansion_mode", row.get("observed_execution", observed)),
        profiler_evidence=row.get("profiler_evidence", evidence))
    algorithm_identified = (
        row.get("algorithm", document.get("algorithm")) is not None)
    declared_measurement = row.get("measurement", document.get("measurement"))
    if isinstance(declared_measurement, Mapping):
        measurement = dict(declared_measurement)
        measurement.setdefault("explicit", True)
    else:
        measurement = _fallback_measurement(
            collective, boundary, algorithm_identified)
    declared_window = row.get("transport_window", document.get("transport_window"))
    window = (dict(declared_window) if isinstance(declared_window, Mapping)
              else _fallback_transport_window(measurement))
    _validate_measurement_window(measurement, window)
    return {
        "campaign": label, "collective": collective, "world_size": int(world),
        "placement_scope": scope, "dtype": _dtype(row.get("dtype") or dtype),
        "payload_bytes": int(payload), "per_op_ms": float(duration),
        "single": bool(single), "pair": pair,
        **{key: resolved[key] for key in (
            "observed_expansion_mode", "requested_expansion_mode", "regime",
            "expansion_unconfirmed", "unconfirmed_tag", "profiler_confirmed")},
        "profiler_evidence": row.get("profiler_evidence", evidence),
        **_selectors(document, row),
        "canary": row.get("canary", row.get("correctness")),
        "run_id": run_id or label, "repeat_index": int(repeat_index or 0),
        "split": split, "source": origin, "boundary": boundary,
        "measurement": measurement, "transport_window": window,
        "direction_observed": bool(measurement.get("direction_observed", False)),
        "algorithm_identified": bool(measurement.get(
            "algorithm_identified", algorithm_identified)),
        **{k: row[k] for k in ("_repeats", "_weights") if k in row},
    }


def _rank_rows(document, origin, p2p_pair=None, default_run_id=None, default_repeat=0):
    protocol = (document.get("protocol")
                if isinstance(document.get("protocol"), Mapping) else {})
    dtype = _dtype(document.get("dtype") or protocol.get("dtype"))
    world = (document.get("world_size")
             or len(str(document.get("visible_devices", "")).split(",")))
    placement = document.get("placement", "")
    default_scope = _explicit_scope(document.get("placement_scope"))
    observed = document.get(
        "observed_expansion_mode",
        document.get("observed_execution", "not_observed"))
    requested = document.get("requested_expansion_mode", "unspecified")
    evidence = document.get("profiler_evidence")
    rows = []
    for case in document.get("cases") or ():
        _validate_row(case, f"{origin}:{case.get('case_id')}")
        collective = _collective(case.get("collective") or case.get("mode"))
        declared_scope = (
            _explicit_scope(case.get("placement_scope")) or default_scope)
        scope = (declared_scope or _explicit_scope(case.get("placement"))
                 or _scope(case.get("placement") or placement, world))
        explicit = (_explicit_pair(case.get("p2p_pair"))
                    or _explicit_pair(case.get("pair"))
                    or _explicit_pair(document.get("p2p_pair"))
                    or _explicit_pair(p2p_pair))
        pair = explicit if collective == "p2p_send_recv" else None
        if collective == "p2p_send_recv":
            scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
            pair = pair or _pair(
                case.get("p2p_scope") or protocol.get("p2p_scope"),
                collective, scope, world)
        run_id = (case.get("run_id") or document.get("run_id") or default_run_id
                  or Path(origin).stem or document.get("label") or "rank")
        repeat = case.get("repeat_index", document.get("repeat_index", default_repeat))
        split = _split(case.get("split", document.get("split")))
        samples = case.get("samples") or ()
        for index, sample in enumerate(samples):
            if isinstance(sample, Mapping):
                _validate_row(sample, f"{origin}:{case.get('case_id')}:sample[{index}]")
        expand = bool(samples) and any(
            isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample))
            for sample in samples)
        if expand:
            inh_rep = ("repeat_index" in case or "repeat_index" in document
                       or default_repeat != 0)
            for index, sample in enumerate(samples):
                duration = _row_duration(sample)
                sample_payload = sample.get(
                    "payload_bytes", sample.get(
                        "size_bytes", sample.get(
                            "work", case.get(
                                "payload_bytes", case.get(
                                    "size_bytes", case.get("work"))))))
                if duration is None or sample_payload is None:
                    continue
                reps, rep = sample.get("_repeats"), sample.get("repeat_index")
                if rep is None and reps:
                    if inh_rep:
                        if (len(reps[0]) > 1 or reps[1]
                                or (reps[0] and reps[0][0][2] in (None, Ellipsis))):
                            raise ValueError("conflicting_primitive_repeat_or_split")
                        rep = int(repeat or 0)
                        sample = {
                            **sample,
                            "_repeats": ([[rep, rep, reps[0][0][2]]], {})
                            if reps[0] else reps,
                        }
                    else:
                        rep = reps[0][0][0] if reps[0] else index
                elif rep is None:
                    rep = repeat if inh_rep else index
                rows.append(_base_row(
                    document, document.get("label") or Path(origin).stem, origin,
                    collective, world, scope, pair, sample_payload, duration, dtype,
                    sample.get("single", case.get("single", document.get("single"))),
                    _merge_obs(case, sample), run_id=sample.get("run_id", run_id),
                    repeat_index=rep, split=_split(sample.get("split", split)),
                    requested=requested, observed=observed, evidence=evidence))
            continue
        duration = _row_duration(case)
        payload = case.get("payload_bytes", case.get("size_bytes", case.get("work")))
        if duration is None or payload is None:
            raise ValueError(
                f"a5_campaign_case_has_no_median:{origin}:{case.get('case_id')}")
        rows.append(_base_row(
            document, document.get("label") or Path(origin).stem, origin,
            collective, world, scope, pair, payload, duration, dtype,
            case.get("single", document.get("single")), case, run_id=run_id,
            repeat_index=repeat, split=split, requested=requested,
            observed=observed, evidence=evidence))
    return rows


def _parent_document(path, document):
    origin = str(path) if path is not None else "memory"
    stem = (path.name[:-len(".parent.json")]
            if path is not None and path.name.endswith(".parent.json")
            else path.stem if path is not None
            else str(document.get("label") or document.get("run_id") or "parent"))
    rank_files = sorted(path.parent.glob(stem + ".rank*.json")) if path is not None else ()
    reports = [(report, origin) for report in document.get("rank_reports") or ()
               if isinstance(report, Mapping)]
    if not rank_files and not reports:
        return [(document, origin)]
    protocol, placement = document.get("protocol") or {}, document.get("placement", "")
    world = document.get("world_size") or 1
    pair = _explicit_pair(document.get("p2p_pair")) or _pair(
        protocol.get("p2p_scope"), "p2p_send_recv", _scope(placement, world), world)
    grouped = defaultdict(list)
    parent_run = str(document.get("run_id") or document.get("label") or stem)
    inherited_keys = (
        "observed_expansion_mode", "observed_execution",
        "requested_expansion_mode", "profiler_evidence", "stack", "cann",
        "algorithm", "protocol", "machine", "stack_binding", "split",
        "boundary", "single",
    )
    defaults = {key: document[key] for key in inherited_keys if key in document}
    for rank_document, rank_origin in chain(
        ((_load_json(item), str(item)) for item in rank_files), reports
    ):
        rank_document = _merge_obs(defaults, rank_document)
        _validate_campaign(rank_document, rank_origin)
        rank_rows = _rank_rows(
            rank_document, rank_origin, pair, default_run_id=parent_run,
            default_repeat=document.get("repeat_index", 0))
        for row in _dedupe_conditions(rank_rows, keep_meta=True):
            grouped[(
                row["collective"], row["payload_bytes"], bool(row["single"]),
                row["split"], row["pair"], row["dtype"], row["placement_scope"],
                row["regime"], row.get("boundary"), row["run_id"],
                row["repeat_index"], row.get("stack"), row.get("cann"),
                row.get("algorithm"), _protocol_key(row.get("protocol")),
            )].append(row)
    rows = []
    for _key, group in sorted(grouped.items(), key=str):
        med = median(row["per_op_ms"] for row in group)
        base = {k: v for k, v in group[0].items()
                if k not in {"_repeats", "_weights"} or len(group) == 1}
        extra = (
            {"_weights": [(med, sum(w for _, w in group[0]["_weights"]))]}
            if len(group) > 1 and "_weights" in group[0] else {}
        )
        rows.append({
            **base,
            "campaign": document.get("label") or stem,
            "per_op_ms": med,
            **extra,
            "canary": "all_ranks_matched",
            "source": origin,
        })
    return [({**document, "conditions": rows,
              "label": document.get("label") or stem}, origin)]


def _list_document(values, origin):
    if not values:
        return []
    if all(isinstance(value, Mapping) for value in values):
        keys = set(values[0])
        base = {
            "label": Path(origin).stem, "status": "completed",
            "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
            "arch_family": A5_ARCH_FAMILY,
        }
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
    if document.get("role") == "rank" or (
        "cases" in document and "median_ms_by_case" not in document
    ):
        label = document.get("label") or path.stem
        return [({
            **document,
            "conditions": _rank_rows(
                document, str(path), _explicit_pair(document.get("p2p_pair"))),
            "label": label,
        }, str(path))]
    return [(document, str(path))]


def _known_root(root):
    return ("mindie-sim-data" in {part.lower() for part in Path(root).parts}
            and Path(root).name.lower() == "a5")


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
        elif size <= _OTHER_JSON_MAX_BYTES:
            others.append(path)
    files = calibration + chains + comms
    if not calibration:
        shadowed = {(path.parent, path.name[:-len(".parent.json")]) for path in parents}
        files += parents
        files += [path for path in ranks
                  if (path.parent, path.name.split(".rank")[0]) not in shadowed]
    files = files or others
    if not files:
        raise ValueError(f"a5_campaign_source_has_no_json:{root}")
    return [document for path in files for document in _expand_path(path)]


def _source_documents(source):
    if source is None:
        embedded = (tuple(EMBEDDED_A5_CAMPAIGNS.values())
                    + tuple(EMBEDDED_A5_MC2_CAMPAIGNS.values()))
        return [(_campaign, _campaign.get("source")
                 or f"embedded:{_campaign.get('label', 'a5')}")
                for _campaign in embedded]
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            return _directory_documents(path)
        if path.is_file():
            return _expand_path(path)
        raise ValueError(f"a5_campaign_source_missing:{path}")
    if isinstance(source, Mapping):
        if source.get("role") == "parent" or (
            "rank_reports" in source and "median_ms_by_case" not in source
        ):
            return _parent_document(None, source)
        if source.get("role") == "rank" or (
            "cases" in source and "median_ms_by_case" not in source
            and "conditions" not in source
        ):
            label = source.get("label") or source.get("run_id") or "rank"
            return [({
                **source,
                "conditions": _rank_rows(
                    source, "memory", _explicit_pair(source.get("p2p_pair"))),
                "label": label,
            }, "memory")]
        if source and all(
            isinstance(value, Mapping)
            and ({"label", "conditions", "series", "chains", "rank_reports", "cases"}
                 & set(value))
            for value in source.values()
        ):
            return [doc for value in source.values() for doc in _source_documents(value)]
        if "legs_ms" in source and ("condition" in source or "chain_ms" in source):
            return _list_document([source], "memory")
        if ({"payload_bytes", "per_op_ms"} <= set(source)
                or {"size_bytes", "median_ms"} <= set(source)):
            return _list_document([source], "memory")
        return [(source, "memory")]
    if isinstance(source, (list, tuple)):
        if (source and all(isinstance(value, Mapping) for value in source)
                and any({"label", "conditions", "series", "chains", "families",
                         "rank_reports", "cases"} & set(value) for value in source)):
            return [doc for value in source for doc in _source_documents(value)]
        return _list_document(list(source), "memory")
    raise ValueError("a5_campaign_source_requires_path_or_mapping")


def _condition_rows(row, document, label, origin, default_dtype, default_world,
                    default_scope, default_pair, default_observed, default_requested,
                    default_evidence):
    _validate_row(row, f"{label}:{row.get('collective', row.get('mode'))}")
    collective = _collective(row.get("collective") or row.get("mode"))
    world = int(row.get("world_size") or row.get("ranks")
                or row.get("group_size") or default_world or 1)
    placement = row.get("placement") or document.get("placement", "")
    declared_scope = _explicit_scope(row.get("placement_scope")) or default_scope
    scope = declared_scope or _explicit_scope(placement) or _scope("", world)
    explicit = _explicit_pair(row.get("p2p_pair")) or _explicit_pair(row.get("pair"))
    pair = explicit if collective == "p2p_send_recv" else None
    if collective == "p2p_send_recv":
        scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
        pair = pair or default_pair or _pair(
            row.get("p2p_scope"), collective, scope, world)
    payload = row.get("payload_bytes", row.get("size_bytes", row.get("work")))
    duration = _row_duration(row)
    run_id = row.get("run_id") or document.get("run_id") or label
    repeat = row.get("repeat_index", document.get("repeat_index", 0))
    split = _split(row.get("split", document.get("split")))
    samples = row.get("samples") or ()
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            _validate_row(sample, f"{label}:sample[{index}]")
    expand = bool(samples) and any(
        isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample))
        for sample in samples)
    if not expand and (payload is None or duration is None):
        raise ValueError(f"a5_campaign_condition_missing_payload_or_duration:{label}")
    if expand:
        result = []
        for index, sample in enumerate(samples):
            sample_duration = _row_duration(sample)
            sample_payload = sample.get(
                "payload_bytes", sample.get("size_bytes", sample.get("work", payload)))
            if sample_duration is None or sample_payload is None:
                continue
            rep = sample.get(
                "repeat_index",
                sample["_repeats"][0][0][0]
                if sample.get("_repeats") and sample["_repeats"][0] else index)
            result.append(_base_row(
                document, label, origin, collective, world, scope, pair,
                sample_payload, sample_duration, default_dtype,
                sample.get("single", row.get("single", document.get("single"))),
                _merge_obs(row, sample), run_id=sample.get("run_id", run_id),
                repeat_index=rep, split=_split(sample.get("split", split)),
                requested=default_requested, observed=default_observed,
                evidence=default_evidence))
        return result
    return [_base_row(
        document, label, origin, collective, world, scope, pair, payload,
        duration, default_dtype, row.get("single", document.get("single")),
        row, run_id=run_id, repeat_index=repeat, split=split,
        requested=default_requested, observed=default_observed,
        evidence=default_evidence)]


def _add_series(rows, document, label, origin, dtype, common, series, single=False):
    collective, world, scope, pair, points = series
    collective = _collective(collective)
    pair = pair or _pair("", collective, scope, world)
    for payload, duration in points:
        rows.append(_base_row(
            document, label, origin, collective, world, scope, pair, payload,
            duration, dtype, bool(single or document.get("single")), **common))


def _campaign_conditions(document, label, origin):
    protocol = (document.get("protocol")
                if isinstance(document.get("protocol"), Mapping) else {})
    dtype = _dtype(document.get("dtype") or protocol.get("dtype"))
    world = document.get("world_size") or document.get("ranks")
    scope = _explicit_scope(document.get("placement_scope"))
    observed = document.get(
        "observed_expansion_mode",
        document.get("observed_execution", "not_observed"))
    requested = document.get("requested_expansion_mode", "unspecified")
    evidence = document.get("profiler_evidence")
    common = {
        "run_id": document.get("run_id") or label,
        "repeat_index": document.get("repeat_index", 0),
        "split": _split(document.get("split")),
        "requested": requested, "observed": observed, "evidence": evidence,
    }
    rows = []
    for row in document.get("conditions") or document.get("comm_conditions") or ():
        rows.extend(_condition_rows(
            row, document, label, origin, dtype, world, scope, None,
            observed, requested, evidence))
    for series in document.get("series") or ():
        _add_series(rows, document, label, origin, dtype, common, series)
    raw_points = document.get("raw_points") or ()
    if raw_points:
        lane = {}
        for index, point in enumerate(raw_points):
            _validate_row(point, f"{label}:raw_points[{index}]")
            observed_point = point.get(
                "observed_expansion_mode", point.get("observed_execution"))
            resolved_point = resolve_expansion_regime(
                point.get("requested_expansion_mode", requested),
                point.get("observed_expansion_mode",
                          point.get("observed_execution", observed)),
                profiler_evidence=point.get("profiler_evidence", evidence))
            sel = _selectors(document, point)
            key = (
                _dtype(point.get("dtype")), int(point.get("ep")),
                int(point.get("link_bytes")),
                bool(point.get("single", document.get("single", False))),
                _split(point.get("split", document.get("split"))),
                point.get("run_id"), point.get("repeat_index"),
                observed_point, point.get("requested_expansion_mode"),
                point.get("algorithm"),
                point.get("boundary", document.get("boundary")),
                resolved_point["regime"], resolved_point["unconfirmed_tag"],
                resolved_point["profiler_confirmed"],
                (sel["stack"], sel["cann"], sel["algorithm"],
                 _protocol_key(sel["protocol"])),
            )
            lane.setdefault(key, (point, []))[1].append(float(point["ms"]))
        for key, (point, values) in sorted(lane.items(), key=str):
            (lane_dtype, series_world, payload, single, split, run_id,
             repeat_index, observed_point, requested_point,
             _, _, _, _, _, _) = key
            rows.append(_base_row(
                document, label, origin, "all_to_allv", series_world,
                _scope("", series_world), None, payload, median(values),
                lane_dtype, single, point,
                run_id=run_id or common["run_id"],
                repeat_index=(repeat_index if repeat_index is not None
                              else common["repeat_index"]),
                split=split if split is not None else common["split"],
                requested=(requested_point if requested_point is not None
                           else common["requested"]),
                observed=(observed_point if observed_point is not None
                          else common["observed"]),
                evidence=point.get("profiler_evidence", evidence)))
    for lane_dtype, series_world, points in document.get("all_to_allv_series") or ():
        _add_series(
            rows, document, label, origin, lane_dtype, common,
            ("all_to_allv", series_world, _scope("", series_world), None, points))
    for series in document.get("single_anchors") or ():
        _add_series(rows, document, label, origin, dtype, common, series, True)
    return rows


def ingest_a5_campaign_data(source=None):
    """Normalize communication campaigns into steady-state conditions and launch anchors."""
    campaigns, conditions, anchors, chains = [], [], [], []
    for document, origin in _source_documents(source):
        label = str(document.get("label") or document.get("run_id")
                    or document.get("campaign") or Path(origin).stem)
        if (document.get("schema_version") == "mindie-mc2-launch/1"
                and origin not in {"memory", "embedded"}):
            label = f"{Path(origin).parent.name}:{label}"
        _validate_campaign(document, label)
        _require_a5(
            document.get("hardware_key") or document.get("sku") or A5_HARDWARE_KEY,
            document.get("arch_family"))
        protocol = document.get("protocol")
        campaigns.append({
            "label": label,
            "status": document.get("status", document.get("state", "completed")),
            "kind": document.get("kind", "calibration"),
            "hardware_key": document.get("hardware_key", A5_HARDWARE_KEY),
            "arch_family": document.get("arch_family", A5_ARCH_FAMILY),
            "stack": document.get("stack"), "date": document.get("date"),
            "cann": document.get("cann") or (document.get("machine") or {}).get("cann"),
            "algorithm": (document.get("algorithm")
                          or (protocol.get("algorithm")
                              if isinstance(protocol, Mapping) else None)),
            "protocol": protocol, "placement": document.get("placement"),
            "nodes": list(document.get("nodes") or ()),
            "dtype": _dtype(document.get("dtype")),
            "observed_expansion_mode": _observed_mode(
                document.get("observed_expansion_mode")),
            "canary": document.get("canary"), "source": origin,
        })
        for row in _campaign_conditions(document, label, origin):
            (anchors if row["single"] else conditions).append(row)
        chains.extend(_campaign_chains(document, label, origin))
    conditions, anchors = _dedupe_conditions(conditions), _dedupe_conditions(anchors)
    chains = _dedupe_chains(chains)
    campaigns.sort(key=lambda item: item["label"])
    return {
        "schema": "a5-campaign-ingestion/1", "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "campaigns": campaigns,
        "conditions": conditions, "rows": conditions,
        "single_op_anchors": anchors, "mc2_chains": chains,
        "campaign_labels": [campaign["label"] for campaign in campaigns],
    }


def _coerce_ingested(ingested):
    if ingested is None:
        return ingest_a5_campaign_data()
    if isinstance(ingested, Mapping) and ingested.get("schema") == "a5-campaign-ingestion/1":
        return ingested
    return ingest_a5_campaign_data(ingested)
