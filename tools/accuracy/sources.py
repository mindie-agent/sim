"""Declared source registry and per-artifact readers for the accuracy dataset.

[Role] The only place that knows how a measurement artifact is shaped. Each reader turns
  one campaign file into the neutral record the assembler normalises; the registry
  declares the role, architecture, die evidence, protocol defaults and family/recipe
  vocabulary of every source.
[Invariants]
  * A source declares its *evidence*: the die rule and any protocol default cite where
    the value comes from (inventory section or a declared protocol manifest). Nothing is
    a wildcard and nothing is a hardcoded SKU constant — the hardware key is read from
    each artifact.
  * ``role`` decides ingest: raw sources become observations, derived copies and stores
    are census/verification only, so a derived copy is never counted beside its raw source.
  * Only measured, non-synthetic records survive: a ``not_a3_measured`` or dry-run
    artifact is flagged and excluded with a reason, never silently dropped.
  * Reads are format-tolerant and always bounded: a malformed file raises and the
    assembler records a read error instead of guessing.
[Timeouts & Locks] Read-only single pass per file; no device, network or subprocess work.
[Side-Effect Policy] Never writes; never opens a file for writing.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROUND1_CAMPAIGNS = ("generic83", "native48", "wave3-generic17", "wave3-attention11")

# Ready / not-ready record statuses. Anything else is excluded with the status named.
MEASURED_STATUSES = {"completed", "measured", "ok", "done"}
SYNTHETIC_FLAGS = ("synthetic", "not_a3_measured", "dry_run", "simulated")

# The native worker recipe vocabulary is shared by every native campaign; one map keeps
# the family label of a condition independent of which pilot happened to measure it.
NATIVE_FAMILY_MAP = {
    "torch_mm_preallocated": "gemm",
    "elementwise_add_reference": "add",
    "uint8_copy_rows_reference": "copy_bytes",
    "npu_rms_norm": "rmsnorm",
    "npu_swiglu": "swiglu",
    "copy_rows_reference": "copy_bytes",
    "torch_softmax": "attention_softmax",
    "npu_scaled_masked_softmax": "masked_softmax",
    "npu_grouped_matmul_list_bf16": "grouped_matmul",
    "npu_grouped_matmul_bf16": "grouped_matmul",
    "npu_dynamic_quant": "quantize",
    "npu_fusion_attention": "attention_qk",
    "npu_rotary_mul": "rope",
    "torch_attention_matmul": "attention_qk",
}

# --------------------------------------------------------------------------
# Source registry. One entry per measurement-bearing artifact. ``role`` is the
# inventory role (raw / derived / store / review); only ``role == "raw"`` rows are
# emitted as observations — a derived copy or a store is scanned for census and for
# duplicate/estimator evidence, never ingested next to its own raw source.
# ``die`` names the physical-die rule and the evidence for it; ``defaults`` are
# source-declared protocol values (with the same evidence), never wildcards.
# --------------------------------------------------------------------------
SOURCES: list[dict[str, Any]] = [
    {
        "id": "a3_measurements_root", "arch_family": "A3", "role": "raw",
        "reader": "root_cases", "files": ["a3-measurements.json"],
        "family_map": {"matmul": "gemm", "add": "add", "copy": "copy",
                       "paired_streams": "paired_cv"},
        "recipe_map": {"matmul": "torch_mm_preallocated", "add": "elementwise_add_reference",
                       "copy": "copy_rows_reference",
                       "paired_streams": "two_stream_concurrent_mm_add"},
        "lifts": {"warmup": "warmup_repeats", "iterations": "samples_per_case",
                  "trials": 1},
        "die": {"kind": "first_visible",
                "evidence": "inventory §6.1: logical NPU0 of container-visible 0,1 = physical Die0"},
        "cache_policy": "same preallocated buffers reused; no cache flush; not sustained HBM calibration",
        "outside_store": True,
    },
    {
        "id": "a3_gemm_expanded", "arch_family": "A3", "role": "raw",
        "reader": "root_cases", "files": ["a3-gemm-expanded.json"],
        "family_map": {"matmul": "gemm"},
        "recipe_map": {"matmul": "torch_mm_preallocated"},
        "lifts": {"warmup": "warmup_repeats", "iterations": "samples_per_case",
                  "trials": 1},
        "die": {"kind": "first_visible",
                "evidence": "inventory §6.1: first lease-visible logical NPU0 (Die0)"},
        "cache_policy": "same tensors reused without flushing; no sustained HBM calibration",
        "outside_store": True,
    },
    {
        "id": "a3_gemm_expanded_train_derived", "arch_family": "A3", "role": "derived",
        "ingest": False,
        "reader": "root_cases", "files": ["a3-gemm-expanded-train.json"],
        "derived_of": "a3_gemm_expanded",
        "evidence": "inventory §5.2: 48/48 raw sample lists identical to a3-gemm-expanded.json",
    },
    {
        "id": "comm_root_sio", "arch_family": "A3", "role": "raw",
        "reader": "comm_ranks", "files": ["comm-sio-v2.json"],
        "family_map": {}, "recipe_map": {},
        "die": {"kind": "all_visible", "route": "sio",
                "evidence": "inventory §6.1: visible_devices 0,1 = SIO phy0<->phy1"},
        "outside_store": True,
    },
    {
        "id": "comm_root_hccssw", "arch_family": "A3", "role": "raw",
        "reader": "comm_ranks", "files": ["comm-hccssw.json"],
        "family_map": {}, "recipe_map": {},
        "die": {"kind": "all_visible", "route": "hccssw",
                "evidence": "inventory §6.1: visible_devices 0,2 = HCCS_SW phy0<->phy2"},
        "outside_store": True,
    },
    {
        "id": "comm_root_sio_failed", "arch_family": "A3", "role": "failed",
        "reader": "comm_ranks", "files": ["comm-sio.json"],
        "evidence": "inventory §1.1 #6: comm-sio.json is a failed artifact (0 measured cases)",
    },
    {
        "id": "communication_train_derived", "arch_family": "A3", "role": "derived",
        "reader": "derived_merge", "files": ["communication-train.json"],
        "derived_of": "comm_root_sio+comm_root_hccssw",
        "evidence": "inventory §5.2: 2 runs x 2 ranks x 6 train cases = 12 distinct scenarios counted 24x",
    },
    {
        "id": "nine_model_linearity", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results",
        "files": ["nine-model-linearity-20261021/train-results.json",
                  "nine-model-linearity-20261021/heldout-results.json"],
        "family_map": {}, "recipe_map": {"mm": "torch_mm_preallocated",
                                        "add": "elementwise_add_reference",
                                        "copy": "copy_rows_reference"},
        "recipe_from": "production_result.details.kind",
        "family_from": "request.op_family",
        "lifts": {"timing_scope": "production_result.timing_scope", "trials": 1},
        "die": {"kind": "visible_devices",
                "evidence": "inventory §6.1: visible_devices=1 = physical Die1"},
    },
    {
        "id": "native_wave2", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results",
        "files": ["native-wave2-20261021/train-results.json",
                  "native-wave2-20261021/heldout-results.json"],
        "family_map": NATIVE_FAMILY_MAP,
        "recipe_from": "worker_recipe",
        "die": {"kind": "visible_devices",
                "evidence": "inventory §6.1: visible_devices=1 = physical Die1"},
        "outside_store": True,
    },
    {
        "id": "native_wave3_generic", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results",
        "files": ["native-wave3-generic-20261021/train-results.json",
                  "native-wave3-generic-20261021/heldout-results.json"],
        "family_map": NATIVE_FAMILY_MAP,
        "recipe_from": "worker_recipe",
        "die": {"kind": "visible_devices",
                "evidence": "inventory §6.1: visible_devices=1 = physical Die1"},
        "outside_store": True,
    },
    {
        "id": "native_wave3_attention", "arch_family": "A3", "role": "raw",
        "reader": "case_trials",
        "files": ["native-wave3-attention-20261021/current-train-results.json",
                  "native-wave3-attention-20261021/current-heldout-results.json"],
        "family_map": {"fusion_attention": "attention_qk", "rotary_mul": "rope",
                       "rms_norm": "rmsnorm", "swiglu": "swiglu", "copy": "copy_bytes"},
        "recipe_from": "native.observed_api",
        "recipe_map": {"torch_npu.npu_fusion_attention": "npu_fusion_attention",
                       "torch_npu.npu_rotary_mul": "npu_rotary_mul"},
        "die": {"kind": "visible_devices",
                "evidence": "inventory §6.1: visible_devices=1 = physical Die1"},
        "defaults": {"timing_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
                     "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
                     "measurement_scope": "inclusive_event_interval_with_host_gap_and_native_output_extraction",
                     "preallocated": False},
        "defaults_evidence": "declared deployed protocol contract, round-1 geometry/wave3-attention11-public-queries.json",
        "outside_store": True,
    },
    {
        "id": "native_wave3_attention_presentations", "arch_family": "A3", "role": "derived",
        "reader": "case_trials",
        "files": ["native-wave3-attention-20261021/capability-results.json",
                  "native-wave3-attention-20261021/train-results.json",
                  "native-wave3-attention-20261021/heldout-results.json"],
        "derived_of": "native_wave3_attention",
        "family_map": {"fusion_attention": "attention_qk", "rotary_mul": "rope"},
        "recipe_from": "native.observed_api",
        "recipe_map": {"torch_npu.npu_fusion_attention": "npu_fusion_attention",
                       "torch_npu.npu_rotary_mul": "npu_rotary_mul"},
        "die": {"kind": "visible_devices"},
        "defaults": {"timing_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
                     "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
                     "measurement_scope": "inclusive_event_interval_with_host_gap_and_native_output_extraction",
                     "preallocated": False},
        "evidence": "inventory §5.2: 3 presentations of the same 53 attention cases",
    },
    {
        "id": "attention_rope_remeasure", "arch_family": "A3", "role": "raw",
        "reader": "case_trials", "files": ["attention-rope-20261021/capability-results.json"],
        "family_map": {"fusion_attention": "attention_qk", "rotary_mul": "rope"},
        "recipe_from": "native.observed_api",
        "recipe_map": {"torch_npu.npu_fusion_attention": "npu_fusion_attention",
                       "torch_npu.npu_rotary_mul": "npu_rotary_mul"},
        "die": {"kind": "first_visible",
                "evidence": "inventory §6.1: same die as wave3-attention (remainder 'same die')"},
        "defaults": {"timing_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
                     "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
                     "measurement_scope": "inclusive_event_interval_with_host_gap_and_native_output_extraction",
                     "preallocated": False},
        "defaults_evidence": "declared deployed protocol contract, round-1 geometry/wave3-attention11-public-queries.json",
        "outside_store": True,
    },
    {
        "id": "graph_replay", "arch_family": "A3", "role": "raw",
        "reader": "graph_conditions",
        "files": ["graph-replay-20261021/capability-results.json",
                  "graph-replay-20261021/full-matrix-results.json"],
        "family_map": {"rms_norm": "graph_rmsnorm", "swiglu": "graph_swiglu",
                       "copy": "graph_copy"},
        "die": {"kind": "declared", "die": "phy1",
                "evidence": "inventory §6.1: Die1; the artifact attests only logical npu:0 of a 1-device lease"},
        "defaults": {"timing_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
                     "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
                     "measurement_scope": "inclusive_event_interval_with_host_gap_and_native_output_extraction",
                     "preallocated": False},
        "defaults_evidence": "graph worker three-clock protocol (capability-matrix §D)",
        "outside_store": True,
    },
    {
        "id": "comm_paired_steady", "arch_family": "A3", "role": "store",
        "ingest": True,
        "reader": "sqlite_store", "files": ["comm-paired-steady-20261021/calibration.sqlite"],
        "die": {"kind": "declared", "die": "phy2+phy3",
                "evidence": "inventory §6.1: visible 2,3 = SIO phy2<->phy3"},
        "outside_store": True,
    },
    {
        "id": "production_loop_v1", "arch_family": "A3", "role": "store",
        "ingest": True,
        "reader": "sqlite_store", "files": ["production-loop-20261021/calibration.sqlite"],
        "die": {"kind": "declared", "die": "phy1", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "production_loop_v2", "arch_family": "A3", "role": "store",
        "ingest": True,
        "reader": "sqlite_store", "files": ["production-loop-v2-20261021/calibration.sqlite"],
        "die": {"kind": "declared", "die": "phy1", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "production_loop_v3", "arch_family": "A3", "role": "store",
        "ingest": True,
        "reader": "sqlite_store", "files": ["production-loop-v3-30-20261021/calibration.sqlite"],
        "die": {"kind": "declared", "die": "phy1", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "native_next4", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results", "files": ["native-next4-20261021/results.json"],
        "family_map": NATIVE_FAMILY_MAP,
        "recipe_from": "worker_recipe",
        "die": {"kind": "visible_devices", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "native_masked_softmax_v2", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results",
        "files": ["native-masked-softmax-v2-20261021/results.json"],
        "family_map": NATIVE_FAMILY_MAP,
        "recipe_from": "worker_recipe",
        "die": {"kind": "visible_devices", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "native_worker_pilot", "arch_family": "A3", "role": "raw",
        "reader": "campaign_results",
        "files": ["native-worker-pilot-20261021/results.json"],
        "family_map": NATIVE_FAMILY_MAP,
        "recipe_from": "worker_recipe",
        "die": {"kind": "visible_devices", "evidence": "inventory §6.1: Die1"},
        "outside_store": True,
    },
    {
        "id": "generic_training_store", "arch_family": "A3", "role": "store",
        "ingest": False, "reader": "sqlite_store",
        "files": ["independent-generalization-20261021/generic-training.sqlite"],
        "derived_of": "nine_model_linearity",
        "evidence": "inventory §5.1: store is a derived ingest of the nine-model raw run (457 train rows)",
    },
    {
        "id": "nine_model_raw_records_mirror", "arch_family": "A3", "role": "derived",
        "reader": "jsonl_mirror",
        "files": ["nine-model-linearity-20261021/raw-records.jsonl"],
        "derived_of": "nine_model_linearity",
        "evidence": "inventory §5.2: raw mirror of the two nine-model results files",
    },
    {
        "id": "wave2_raw_records_mirror", "arch_family": "A3", "role": "derived",
        "reader": "jsonl_mirror", "files": ["native-wave2-20261021/raw-records.jsonl"],
        "derived_of": "native_wave2",
    },
    {
        "id": "wave3_generic_raw_records_mirror", "arch_family": "A3", "role": "derived",
        "reader": "jsonl_mirror",
        "files": ["native-wave3-generic-20261021/raw-records.jsonl"],
        "derived_of": "native_wave3_generic",
    },
]


# ------------------------------------------------------------------- helpers ---
def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ints(shape: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in shape.items():
        number = _number(value)
        result[str(key)] = int(number) if number is not None and number.is_integer() else (
            number if number is not None else value)
    return result


def _is_synthetic(document: Mapping[str, Any]) -> str | None:
    for flag in SYNTHETIC_FLAGS:
        if document.get(flag) is True:
            return f"declared_{flag}"
    if document.get("numerical_model_execution") is False and document.get("npu_initialized") is False:
        return "not_a_device_measurement"
    return None


def _split_hint(value: Any) -> str | None:
    text = str(value or "").lower()
    if text in {"heldout", "holdout", "test", "test1", "test2"}:
        return "heldout"
    if text in {"train", "training", "fit"}:
        return "train"
    return None


def _die_for(source: Mapping[str, Any], environment: Mapping[str, Any],
             record: Mapping[str, Any]) -> tuple[str, bool, str]:
    rule = source.get("die") or {}
    kind = rule.get("kind")
    visible = str(environment.get("visible_devices") or record.get("visible_devices") or "")
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if kind == "declared":
        return rule.get("die", "unknown"), False, str(rule.get("evidence") or "declared_die")
    if kind == "visible_devices" and len(devices) == 1 and devices[0].isdigit():
        return f"phy{devices[0]}", True, "environment.visible_devices"
    if kind == "first_visible" and devices:
        first = devices[0]
        if first.isdigit():
            return f"phy{first}", True, "environment.visible_devices[0]"
    if kind == "all_visible" and devices and all(part.isdigit() for part in devices):
        route = rule.get("route")
        joined = "+".join(f"phy{part}" for part in devices)
        return (f"{route}:{joined}" if route else joined), True, "environment.visible_devices"
    selected = (environment.get("device") or {}).get("selected") if isinstance(
        environment.get("device"), dict) else environment.get("device")
    if selected:
        digits = "".join(ch for ch in str(selected) if ch.isdigit())
        if digits:
            return f"logical{int(digits)}", False, "environment.device.selected (logical only)"
    return "unknown", False, "unattested"


def _device_name(environment: Mapping[str, Any]) -> str | None:
    """Resolve the SKU key from whatever the artifact attests (never a default SKU)."""
    name = environment.get("device_name")
    if isinstance(name, str) and name:
        return name
    device = environment.get("device")
    if isinstance(device, Mapping):
        for key in ("name", "device_name"):
            value = device.get(key)
            if isinstance(value, str) and value:
                return value
        facts = device.get("facts") or {}
        if isinstance(facts, Mapping):
            value = (facts.get("name") or {}).get("value") if isinstance(facts.get("name"), dict) else None
            if isinstance(value, str) and value:
                return value
        if isinstance(device.get("selected"), str) and device.get("device_name"):
            return str(device["device_name"])
    packages = environment.get("packages")
    if isinstance(packages, Mapping) and isinstance(packages.get("device_name"), str):
        return packages["device_name"]
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


def _ir(**kwargs) -> dict[str, Any]:
    base = {
        "record_id": None, "family": None, "recipe": None, "execution_kind": None,
        "worker_recipe": None, "backend": None, "precision": None, "layout": None,
        "quantization": None, "features": {}, "identity_extra": {}, "shape": None,
        "dtype": None, "itemsize": None, "hardware_key": None, "arch_family": None,
        "die": "unknown", "die_attested": False, "die_evidence": "",
        "run_id": None, "source_class": "measured", "status": "completed",
        "split_hint": None, "split_origin": None, "trials": [], "analytic_ms": None,
        "protocol_raw": {}, "record_level": {}, "defaults": {},
        "status_note": None, "flags": [],
    }
    base.update(kwargs)
    return base


def _clean_trials(trials: Iterable[Iterable[Any]]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for trial in trials:
        values = [value for value in (_number(item) for item in trial) if value is not None]
        if values:
            cleaned.append(values)
    return cleaned


def _split_flattened(samples: Sequence[float], trials: int | None,
                     trial_medians_attested: Sequence[float] | None) -> tuple[list[list[float]], bool]:
    """Recover per-trial groups from a flattened sample list, proving the split.

    The runner concatenates per-trial sample streams; the split is accepted only when
    every reconstructed chunk median equals the attested ``trial_medians_ms`` entry.
    Otherwise the condition keeps one group and the lost structure is flagged.
    """
    values = [float(value) for value in samples]
    if trials and trials > 1 and len(values) % trials == 0:
        size = len(values) // trials
        chunks = [values[i * size:(i + 1) * size] for i in range(trials)]
        if trial_medians_attested and len(trial_medians_attested) == trials:
            if all(abs(statistics.median(chunk) - float(attested)) < 1e-9
                   for chunk, attested in zip(chunks, trial_medians_attested)):
                return chunks, True
    return [values], False


def _shape_from_details(details: Mapping[str, Any]) -> dict[str, Any] | None:
    shape = details.get("shape")
    if isinstance(shape, Mapping):
        return dict(shape)
    for keys in (("rows", "row_bytes"), ("tokens", "width"), ("elements",)):
        if all(key in details for key in keys):
            return {key: details[key] for key in keys}
    if all(key in details for key in ("q", "kv", "heads")):
        return {key: details[key] for key in ("q", "kv", "heads")}
    if isinstance(details.get("geometry"), Mapping):
        return dict(details["geometry"])
    return None


def _precision(record: Mapping[str, Any], protocol: Mapping[str, Any],
               details: Mapping[str, Any]) -> dict[str, Any] | None:
    for candidate in (record.get("precision"), details.get("precision")):
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    dtype = details.get("dtype") or record.get("dtype") or protocol.get("dtype")
    if dtype:
        return {"input_dtype": str(dtype)}
    return None


def _identity_extra(family: str, details: Mapping[str, Any],
                    geometry: Mapping[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if family in {"attention_qk", "attention_softmax", "masked_softmax"}:
        for source in (geometry, details):
            for key in ("mask", "scale", "layout", "q_heads", "kv_heads", "qk_dim"):
                if key in source and key not in extra:
                    value = source[key]
                    if isinstance(value, Mapping):
                        value = value.get("kind") or value.get("convention")
                    extra[key] = value
    elif family == "rope":
        for source in (geometry, details):
            for key in ("rotary_mode", "layout", "dim", "heads"):
                if key in source and key not in extra:
                    extra[key] = source[key]
    elif family == "graph_copy" or family == "copy":
        if "itemsize" in geometry:
            extra["itemsize"] = geometry["itemsize"]
    return extra


# ------------------------------------------------------------------- readers ---
def read_root_cases(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """a3-measurements / a3-gemm-expanded: ``cases[]`` with amortised per-sample device ms."""
    document = _load_json(path)
    protocol = document.get("protocol") or {}
    environment = document.get("environment") or {}
    defaults = dict(source.get("defaults") or {})
    defaults.setdefault("timer", protocol.get("timer") or "device_event")
    defaults.setdefault("cache_policy", source.get("cache_policy"))
    defaults.setdefault("measurement_scope", protocol.get("measurement_scope"))
    lifts = source.get("lifts") or {}
    records = []
    for case in document.get("cases") or []:
        kind = case.get("kind")
        shape_mkn = case.get("shape_mkn")
        shape_list = case.get("shape")
        if shape_mkn:
            shape = {"m": shape_mkn[0], "n": shape_mkn[1], "k": shape_mkn[2]}
        elif case.get("v_shape"):
            # Two independent streams (matmul + elementwise) fed concurrently. The
            # element count is the declared vector payload; the matmul geometry is kept
            # as an identity extra because no public family work model describes the pair.
            shape = {"elements": case["v_shape"][0]}
        elif isinstance(shape_list, list) and len(shape_list) == 1:
            shape = {"elements": shape_list[0]}
        elif isinstance(shape_list, list) and len(shape_list) == 3:
            shape = {"m": shape_list[0], "n": shape_list[1], "k": shape_list[2]}
        elif isinstance(shape_list, Mapping):
            shape = dict(shape_list)
        else:
            shape = None
        samples = case.get("raw_device_ms_per_op") or [
            sample.get("device_ms") for sample in case.get("samples") or []]
        if samples and case.get("repeats_per_sample") and not case.get("raw_device_ms_per_op"):
            samples = [value / case["repeats_per_sample"] for value in samples
                       if _number(value) is not None]
        record_level = {}
        for key, field in lifts.items():
            if field in protocol:
                record_level[key] = protocol[field]
            elif isinstance(field, (int, float)):
                record_level[key] = field
        if case.get("repeats_per_sample"):
            record_level["amortization"] = case["repeats_per_sample"]
        die, attested, evidence = _die_for(source, environment, case)
        records.append(_ir(
            record_id=_record_id(source["id"], case.get("case_id")),
            family=(source.get("family_map") or {}).get(kind, kind),
            recipe=(source.get("recipe_map") or {}).get(kind, kind),
            shape=shape, dtype=protocol.get("dtype"),
            hardware_key=environment.get("device_name") or document.get("device_name"),
            arch_family=source.get("arch_family"),
            die=die, die_attested=attested, die_evidence=evidence,
            run_id=document.get("run_id"), source_class="measured",
            status=case.get("status") or "completed",
            split_hint=_split_hint(case.get("split")), split_origin="case.split",
            trials=[samples] if samples else [],
            protocol_raw=protocol, record_level=record_level, defaults=defaults,
            precision=_precision(case, protocol, {}),
            layout=protocol.get("layout"), quantization=protocol.get("quantization"),
            features={"call_path": case.get("call_path"), "kind": kind},
            analytic_ms=case.get("analytic_ms"),
        ))
    return records


def read_campaign_results(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """wave2 / wave3-generic / pilots: ``results[]`` with flattened samples + trial medians."""
    document = _load_json(path)
    protocol = document.get("protocol") or {}
    defaults = dict(source.get("defaults") or {})
    defaults.setdefault("timer", protocol.get("timer") or "device_event")
    rows = document.get("results")
    if rows is None:
        # nine-model presents ``cases[]`` whose measured payload is ``production_result``.
        rows = []
        for case in document.get("cases") or []:
            request = case.get("request") or {}
            result = dict(case.get("production_result") or {})
            result.setdefault("request", request)
            result.setdefault("measurement_id", request.get("case_id"))
            # L4: the predeclared campaign manifest owns the split label. The runner's
            # own production_result.split is an unused default ("train" even on a
            # heldout case, per split_semantics in the artifact).
            result["split"] = (case.get("campaign_split")
                               or request.get("campaign_split")
                               or result.get("split"))
            result.setdefault("worker_recipe", request.get("op_family"))
            rows.append(result)
    records = []
    for result in rows:
        details = result.get("details") or {}
        production = result.get("production_result") or {}
        if source.get("recipe_from") == "production_result.details.kind":
            details = production.get("details") or details
        record_protocol = production.get("protocol") or result.get("protocol") or protocol
        environment = (production.get("environment") or result.get("environment")
                       or document.get("environment") or {})
        campaign = result.get("campaign") or {}
        if isinstance(campaign, dict) and not environment.get("visible_devices"):
            lifted = campaign.get("visible_devices") or campaign.get("physical_npu")
            if lifted is not None:
                environment = {**environment, "visible_devices": str(lifted),
                               "device_name": environment.get("device_name")
                               or campaign.get("device_name")}
        recipe = result.get("worker_recipe")
        if source.get("recipe_from") == "production_result.details.kind":
            recipe = details.get("kind")
        family = (source.get("family_map") or {}).get(recipe) if recipe else None
        if family is None:
            family = (result.get("request") or {}).get("op_family")
        shape = _shape_from_details(details) or (result.get("request") or {}).get("shape")
        recipe_name = (source.get("recipe_map") or {}).get(recipe, recipe)
        trials_declared = result.get("trials")
        trials_count = (len(trials_declared) if isinstance(trials_declared, list)
                        else trials_declared if isinstance(trials_declared, int) else None)
        if trials_count is None and isinstance(result.get("trial_medians_ms"), list):
            trials_count = len(result["trial_medians_ms"])
        samples = result.get("raw_samples_ms") or []
        groups, recovered = _split_flattened(samples, trials_count,
                                            result.get("trial_medians_ms"))
        if trials_count is None:
            # No trial metadata: the artifact presents exactly ``len(groups)`` sample
            # groups, so that count *is* the attested trial count.
            trials_count = len(groups)
        die, attested, evidence = _die_for(source, environment, result)
        defaults_for_record = dict(defaults)
        if isinstance(details.get("preallocated"), bool):
            defaults_for_record.setdefault("preallocated", details["preallocated"])
        record_level = {name: value for name, value in
                        (("timing_scope", result.get("timing_scope")),
                         ("trials", trials_count)) if value is not None}
        flags = []
        declared_trials = record_protocol.get("trials")
        if (isinstance(declared_trials, int) and isinstance(trials_count, int)
                and trials_count != declared_trials):
            flags.append("declared_trials_mismatch")
        if document.get("not_a3_measured") is True:
            flags.append("not_a3_measured")
        if result.get("not_a3_measured") is True:
            flags.append("not_a3_measured")
        records.append(_ir(
            record_id=_record_id(source["id"], result.get("measurement_id") or result.get("request_key")),
            family=family, recipe=recipe_name,
            execution_kind=result.get("execution_kind"), worker_recipe=recipe,
            backend=result.get("backend"), shape=shape,
            dtype=details.get("dtype"),
            hardware_key=_device_name(environment),
            arch_family=source.get("arch_family"),
            die=die, die_attested=attested, die_evidence=evidence,
            run_id=result.get("run_id") or document.get("run_id"),
            source_class=result.get("source_class") or "measured",
            status=result.get("status") or "completed",
            split_hint=_split_hint(result.get("split")), split_origin="result.split",
            trials=groups,
            protocol_raw=record_protocol, record_level=record_level,
            defaults=defaults_for_record,
            precision=_precision(result, record_protocol, details),
            layout=record_protocol.get("layout") or details.get("layout"),
            quantization=record_protocol.get("quantization"),
            features={"target_match": result.get("target_match"),
                      "query_scope": (result.get("query_context") or {}).get("scope")},
            identity_extra=_identity_extra(family or "", details, details.get("geometry") or {}),
            analytic_ms=result.get("analytic_ms"),
            flags=flags if recovered or not trials_count or trials_count <= 1
            else [*flags, "trial_structure_not_recovered"],
        ))
    return records


def read_case_trials(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """wave3-attention / attention-rope: ``cases[]`` with explicit ``trials[].samples[]``."""
    document = _load_json(path)
    protocol = document.get("protocol") or {}
    defaults = dict(source.get("defaults") or {})
    defaults.setdefault("timer", "device_event")
    if isinstance(protocol.get("trials"), int):
        defaults.setdefault("trials", protocol["trials"])
    if isinstance(protocol.get("warmup_iterations"), int):
        defaults.setdefault("warmup", protocol["warmup_iterations"])
    if isinstance(protocol.get("samples_per_trial"), int):
        defaults.setdefault("iterations", protocol["samples_per_trial"])
    records = []
    for case in document.get("cases") or []:
        geometry = case.get("geometry") or {}
        native = case.get("native") or {}
        environment = document.get("environment") or {}
        operator = case.get("operator")
        recipe_raw = native.get("observed_api") or operator
        family = (source.get("family_map") or {}).get(operator, operator)
        shape = geometry.get("shape") or _shape_from_details(case.get("details") or {})
        groups = []
        for trial in case.get("trials") or []:
            groups.append([sample.get("device_interval_ms")
                           for sample in trial.get("samples") or []])
        die, attested, evidence = _die_for(source, environment, case)
        details = case.get("details") or {}
        records.append(_ir(
            record_id=_record_id(source["id"], case.get("case_id")),
            family=family,
            recipe=(source.get("recipe_map") or {}).get(recipe_raw, recipe_raw),
            execution_kind=geometry.get("execution_kind"),
            worker_recipe=recipe_raw, backend=native.get("observed_api"),
            shape=shape, dtype=geometry.get("dtype"),
            hardware_key=_device_name(environment),
            arch_family=source.get("arch_family"),
            die=die, die_attested=attested, die_evidence=evidence,
            run_id=document.get("run_id"), source_class="measured",
            status=case.get("status") or "measured",
            split_hint=_split_hint(case.get("split")), split_origin="case.split",
            trials=_clean_trials(groups),
            protocol_raw=protocol, record_level={}, defaults=defaults,
            precision={"input_dtype": geometry.get("dtype")} if geometry.get("dtype") else None,
            layout=geometry.get("layout"),
            features={"mask": (details.get("mask") or {}).get("kind")
                      if isinstance(details.get("mask"), Mapping) else geometry.get("mask"),
                      "scale": details.get("scale") or geometry.get("scale")},
            identity_extra=_identity_extra(family or "", details, geometry),
            analytic_ms=case.get("analytic_ms"),
        ))
    return records


def read_comm_ranks(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Root comm artifacts: two rank records, each holding the same scenario list."""
    document = _load_json(path)
    environment = document.get("environment") or {
        "visible_devices": document.get("visible_devices"),
        "device_name": document.get("device_name"),
        "host": document.get("host"),
    }
    defaults = {"timer": "device_event", "measurement_scope":
                "device_event around Work.wait-ordered ops; host gaps included",
                "cache_policy": "same buffers reused, no cache flush",
                "preallocated": True, "trials": 1}
    die, attested, evidence = _die_for(source, environment, {})
    records = []
    for rank in document.get("ranks") or []:
        rank_environment = rank.get("environment") or environment
        rank_protocol = rank.get("protocol") or {}
        record_level = {"trials": 1}
        if rank_protocol.get("warmup_repeats"):
            record_level["warmup"] = rank_protocol["warmup_repeats"]
        if rank_protocol.get("samples_per_case"):
            record_level["iterations"] = rank_protocol["samples_per_case"]
            record_level["amortization"] = rank_protocol.get("repetitions_per_lane")
        for case in rank.get("cases") or []:
            samples = [sample.get("active_device_span_ms") for sample in case.get("samples") or []]
            kind = str(case.get("kind") or "comm")
            family = "comm_" + (case.get("mode") or kind).replace("-", "_")
            records.append(_ir(
                record_id=_record_id(source["id"], f"{case.get('case_id')}#r{rank.get('rank')}"),
                family=family, recipe=f"two_rank_{kind}",
                execution_kind=kind, backend=case.get("collective_call"),
                shape={"bytes": case.get("tensor_payload_bytes"), "ranks": 2},
                dtype=rank_protocol.get("payload_dtype"),
                hardware_key=rank_environment.get("device_name"),
                arch_family=source.get("arch_family"),
                die=die, die_attested=attested, die_evidence=evidence,
                run_id=rank.get("run_id") or document.get("run_id"),
                source_class="measured", status=case.get("status") or "completed",
                split_hint=_split_hint(case.get("split")), split_origin="case.split",
                trials=[samples],
                protocol_raw=rank_protocol, record_level=record_level, defaults=defaults,
                layout=None, quantization=None,
                features={"rank": rank.get("rank"), "mode": case.get("mode"),
                          "rank_samples_are_correlated": True},
                analytic_ms=case.get("analytic_ms"),
            ))
    return records


def _graph_shape(operator: Any, tensors: Mapping[str, Any],
                 geometry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Declared graph geometry -> a family shape (packed SwiGLU width is unpacked)."""
    preference = ["output", "input", "packed_input", "source", "target"] if operator == "swiglu" \
        else ["input", "packed_input", "source", "target", "output"]
    for name in preference:
        tensor = tensors.get(name)
        if not isinstance(tensor, Mapping):
            continue
        shape = tensor.get("shape")
        if not isinstance(shape, list):
            continue
        if operator == "copy" and len(shape) >= 2:
            return {"rows": shape[0], "row_bytes": shape[1]}
        if operator == "copy" and len(shape) == 1:
            return {"bytes": tensor.get("nbytes") or shape[0]}
        if len(shape) >= 2:
            return {"tokens": shape[0], "width": shape[1]}
        return {"elements": shape[0]}
    total = geometry.get("total_declared_tensor_bytes")
    return {"bytes": total} if total else None


def read_graph_conditions(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    document = _load_json(path)
    protocol = document.get("protocol") or {}
    environment = document.get("environment") or {}
    defaults = dict(source.get("defaults") or {})
    defaults.setdefault("timer", "device_event")
    defaults.setdefault("trials", protocol.get("trials"))
    defaults.setdefault("warmup", protocol.get("warmup_iterations"))
    defaults.setdefault("iterations", protocol.get("samples_per_condition"))
    records = []
    operator_map = source.get("family_map") or {}
    for condition in document.get("conditions") or []:
        geometry = condition.get("geometry") or {}
        operator = geometry.get("operator") or condition.get("operator")
        mode = condition.get("mode")
        tensors = geometry.get("tensors") or {}
        shape = _graph_shape(operator, tensors, geometry)
        groups = []
        for trial in condition.get("trials") or []:
            groups.append([sample.get("device_interval_ms")
                           for sample in trial.get("samples") or []])
        die, attested, evidence = _die_for(source, environment, {})
        records.append(_ir(
            record_id=_record_id(source["id"], condition.get("condition_id")),
            family=operator_map.get(operator, operator),
            recipe=f"graph_replay:{mode}", execution_kind=mode,
            worker_recipe=(document.get("worker") or {}).get("module"),
            shape=shape or None, dtype=geometry.get("dtype"),
            hardware_key=_device_name(environment),
            arch_family=source.get("arch_family"),
            die=die, die_attested=attested, die_evidence=evidence,
            run_id=document.get("run_id"), source_class="measured",
            status=condition.get("status") or "measured",
            split_hint=None, split_origin="no_campaign_split",
            trials=_clean_trials(groups),
            protocol_raw=protocol, record_level={"trials": protocol.get("trials")},
            defaults=defaults,
            precision={"input_dtype": geometry.get("dtype")} if geometry.get("dtype") else None,
            features={"mode": mode, "n": condition.get("n"),
                      "block_invocations": (condition.get("condition_identity") or {}).get(
                          "block_invocations")},
            identity_extra={"mode": mode, "n": condition.get("n")},
        ))
    return records


def read_sqlite_store(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Calibration stores: one row per ``sample_key`` from the raw ``measurements`` table.

    The store keeps repeats but not trial membership, so the target is recovered as the
    median of the repeat groups it can prove; when the protocol declares more than one
    trial the structure is flagged unrecovered instead of being claimed as the scored
    estimator.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(measurements)")]
        if not columns:
            return []
        selected = ", ".join(columns)
        rows = connection.execute(f"SELECT {selected} FROM measurements").fetchall()
    finally:
        connection.close()
    by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        record = dict(zip(columns, row))
        by_key[record.get("sample_key") or record.get("measurement_key")].append(record)
    environment = {}
    records = []
    for key, group in by_key.items():
        first = group[0]
        try:
            protocol = json.loads(first.get("protocol_json") or "{}")
        except json.JSONDecodeError:
            protocol = {}
        details = {}
        for candidate in (first.get("details_json"), first.get("environment_json")):
            try:
                decoded = json.loads(candidate or "{}")
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict):
                if "environment" in decoded:
                    environment = decoded["environment"]
                if "protocol" in decoded:
                    protocol = decoded["protocol"] if not protocol else protocol
                details.update({k: v for k, v in decoded.items()
                                if k in {"shape", "rows", "row_bytes", "kind",
                                         "op_family", "max_ms", "min_ms", "identity"}})
        try:
            shape = json.loads(first.get("shape_json") or "{}")
        except json.JSONDecodeError:
            shape = {}
        shape = shape or details.get("shape") or {}
        family = first.get("op_family")
        protocol.pop("observed_api", None)
        trials_declared = protocol.get("trials")
        samples = [record.get("duration_ms") for record in group]
        groups, recovered = _split_flattened(samples, trials_declared if isinstance(
            trials_declared, int) else None, None)
        if not recovered:
            groups = [[float(value) for value in samples
                       if _number(value) is not None]]
        if trials_declared is None:
            trials_declared = len(groups)
        die, attested, evidence = _die_for(source, environment or {}, {})
        stored_die = first.get("die")
        if isinstance(stored_die, str) and stored_die not in {"", "unknown"}:
            die, attested, evidence = stored_die, True, "store.die"
        identity = details.get("identity") or {}
        identity_extra = {name: identity[name] for name in
                          ("requested_mode", "observed_expansion", "aiv_block_dims")
                          if name in identity}
        records.append(_ir(
            record_id=_record_id(source["id"], key),
            family=family, recipe=first.get("recipe"),
            shape=shape, dtype=first.get("dtype"),
            hardware_key=first.get("hardware_key"),
            arch_family=source.get("arch_family"),
            die=die, die_attested=attested, die_evidence=evidence,
            run_id=first.get("run_id"), source_class=first.get("source_class") or "measured",
            status="completed",
            split_hint=_split_hint(first.get("split")), split_origin="store.split",
            trials=groups,
            protocol_raw=protocol,
            record_level={"timing_scope": first.get("timing_scope")},
            defaults={},
            precision={"input_dtype": first.get("dtype")} if first.get("dtype") else None,
            layout=first.get("layout"),
            quantization=protocol.get("quantization"),
            features={"store": str(path.name)},
            identity_extra=identity_extra,
            flags=[] if recovered or not trials_declared or trials_declared <= 1
            else ["trial_structure_not_recovered"],
            analytic_ms=_number(first.get("analytic_ms")),
        ))
    return records


def read_derived_merge(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Declared derived merges are census-only: count rows, never emit observations."""
    document = _load_json(path)
    count = 0
    for run in document.get("runs") or []:
        for rank in run.get("ranks") or []:
            count += len(rank.get("cases") or [])
    return [_ir(record_id=_record_id(source["id"], "census"), family=None,
                shape=None, status="derived_census", status_note=f"derived_rows:{count}",
                run_id=document.get("run_id"))]


def read_jsonl_mirror(path: Path, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return [_ir(record_id=_record_id(source["id"], "census"), family=None, shape=None,
                status="derived_census", status_note=f"mirror_rows:{count}")]


READERS = {
    "root_cases": read_root_cases,
    "campaign_results": read_campaign_results,
    "case_trials": read_case_trials,
    "comm_ranks": read_comm_ranks,
    "graph_conditions": read_graph_conditions,
    "sqlite_store": read_sqlite_store,
    "derived_merge": read_derived_merge,
    "jsonl_mirror": read_jsonl_mirror,
}


def discover_round2_sources(root: Path, arch_family: str) -> list[dict[str, Any]]:
    """Registry entries for round-2 campaign output that is not in the static registry.

    The round-2 roots are written by other workstreams while this tool is in use, so the
    registry cannot name their files ahead of time. A file is claimed only when it parses as
    a campaign result document (``results[]`` with ``raw_samples_ms``); anything else stays
    in the discovery census and is reported as unclassified rather than guessed at.
    """
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for path in sorted(root.glob("runs/*/results.json")) + sorted(root.glob("*/results.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            continue
        if not document["results"] or "raw_samples_ms" not in (document["results"][0] or {}):
            continue
        entries.append({
            "id": f"round2_{arch_family.lower()}_{path.parent.name}",
            "arch_family": arch_family, "role": "raw", "ingest": True,
            "reader": "campaign_results",
            "files": [str(path.relative_to(root))],
            "family_map": NATIVE_FAMILY_MAP,
            "recipe_from": "worker_recipe",
            "die": {"kind": "visible_devices",
                    "evidence": "campaign.visible_devices / campaign.physical_npu"},
            "outside_store": True,
        })
    return entries
