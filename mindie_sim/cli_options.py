"""Canonical CLI argument registration and explicit case overrides.

[Role]
  Register runtime/parallel/prefix/reuse/hardware-derivation flags once and
  apply explicit values to an existing Case without overwriting declarations
  that the user omitted.

[Invariants]
  * argparse defaults are ``None`` for case-reading commands, so an omitted
    flag cannot reset a case declaration.
  * ``--kda-proj-tp`` is only an input alias; the canonical field serialized is
    ``linear_proj_tp`` and an unequal duplicate is a conflict.
  * Scalar hardware flags and a JSON derivation declaration must agree when
    both specify the same field; ``--memory-gib`` is exactly ``2**30`` bytes.
  * No flag registers a second conflicting skew redundant-expert option.

[Timeouts & Locks]
  Pure argparse/JSON metadata work; bounded local file read only for the
  declared derivation document.

[Side-Effect Policy]
  Mutates only the supplied parser and returns replacement Cases.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.runtime_features import (
    COLLECTIVE_STARTUP_MODES, HOMOGENEOUS_GRAPH_REUSE_MODES,
    MATRIX_COMPUTE_POLICIES, PREFIX_CACHE_MODES, PREFIX_READ_MODES,
    normalize_prefix_cache_key, validate_runtime_features,
)


GIB_BYTES = 2 ** 30
CLI_ONE = 1
CLI_ZERO = 0
CLI_PROMPT_TOKENS = 4096
CLI_OUTPUT_TOKENS = 128
CLI_MTP_ACCEPT = 1.0
CLI_TTFT_MS = 5000.0
CLI_TPOT_MS = 50.0
SINGLE_RECORD = 1
FIRST = 0
DEFAULT_SKU = "derived-sku"
FEATURE_KEYS = (
    "o_proj_tp", "linear_proj_tp", "lm_head_tp", "redundant_experts_per_rank",
    "collective_startup_mode", "matrix_compute_policy", "recover_ssm",
    "prefix_cache_mode", "prefix_read_mode", "prefix_cache_key",
    "homogeneous_graph_reuse",
)
HARDWARE_KEYS = (
    "aic_count", "aiv_count", "memory_bytes", "hbm_peak_bytes_per_s",
    "hbm_efficiency", "mte2_ceiling_bytes_per_s", "mte2_efficiency",
)
CASE_COMMANDS = frozenset({"sweep", "sla", "efficiency"})


def _mode_choices(values: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(values))


def add_runtime_arguments(parser: argparse.ArgumentParser, *, command: str) -> None:
    """Register the canonical Issue #34 flags on one command parser."""
    parser.add_argument("--o-proj-tp", type=int, default=None)
    parser.add_argument("--linear-proj-tp", type=int, default=None)
    parser.add_argument("--kda-proj-tp", type=int, default=None,
                        help="input alias of --linear-proj-tp")
    parser.add_argument("--lm-head-tp", type=int, default=None)
    if command != "skew":
        parser.add_argument("--redundant-experts-per-rank", type=int, default=None)
    parser.add_argument("--collective-startup-mode", default=None,
                        choices=_mode_choices(COLLECTIVE_STARTUP_MODES))
    parser.add_argument("--matrix-compute-policy", default=None,
                        choices=_mode_choices(MATRIX_COMPUTE_POLICIES))
    parser.add_argument("--recover-ssm", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--prefix-cache-mode", default=None,
                        choices=_mode_choices(PREFIX_CACHE_MODES))
    parser.add_argument("--prefix-read-mode", default=None,
                        choices=_mode_choices(PREFIX_READ_MODES))
    parser.add_argument("--prefix-cache-key", default=None)
    parser.add_argument("--homogeneous-graph-reuse", default=None,
                        choices=_mode_choices(HOMOGENEOUS_GRAPH_REUSE_MODES))
    if command in {"compose", "sweep"}:
        _add_hardware_arguments(parser)


def _add_hardware_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aic-count", type=int, default=None)
    parser.add_argument("--aiv-count", type=int, default=None)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory-bytes", type=int, default=None)
    memory.add_argument("--memory-gib", type=float, default=None)
    parser.add_argument("--hbm-peak-bytes-per-s", type=float, default=None)
    parser.add_argument("--hbm-efficiency", default=None)
    parser.add_argument("--mte2-ceiling-bytes-per-s", type=float, default=None)
    parser.add_argument("--mte2-efficiency", default=None)
    parser.add_argument("--hardware-derivation", default=None,
                        help="JSON file with one derivation declaration")


def _curve(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"{name}_invalid:{value}") from exc
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return float(parsed)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    raise ModelingError(f"{name}_invalid:{value}")


def load_hardware_derivation(path: str | Path) -> dict[str, Any]:
    """Read one declared derivation mapping from an explicit JSON file."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ModelingError("hardware_derivation_document_invalid")
    for key in ("derivations", "skus", "records"):
        records = document.get(key)
        if isinstance(records, Mapping) and len(records) == SINGLE_RECORD:
            document = next(iter(records.values()))
            break
        if isinstance(records, list) and len(records) == SINGLE_RECORD:
            document = records[FIRST]
            break
    if not isinstance(document, Mapping):
        raise ModelingError("hardware_derivation_document_invalid")
    return dict(document)


def load_hardware_derivations(path: str | Path) -> list[dict[str, Any]]:
    """Read a named derivation list for a sweep product."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, Mapping):
        records = document.get("derivations", document.get("skus"))
        if isinstance(records, Mapping):
            return [dict(record) for record in records.values()]
        if records is None:
            return [dict(document)]
        document = records
    if not isinstance(document, list) or not document:
        raise ModelingError("hardware_derivations_document_invalid")
    return [dict(record) for record in document if isinstance(record, Mapping)]


def compose_case_kwargs(namespace: argparse.Namespace) -> dict[str, Any]:
    """Build the complete compose_case keyword set from CLI arguments."""
    value = lambda name, default: (default if getattr(namespace, name, None) is None
                                   else getattr(namespace, name))
    return {
        "tp": value("tp", CLI_ONE), "dp": value("dp", CLI_ONE),
        "ep": value("ep", CLI_ONE), "pp": value("pp", CLI_ONE),
        "pcp": value("pcp", CLI_ONE), "dcp": value("dcp", CLI_ONE),
        "sp": bool(namespace.sp), "num_devices": namespace.num_devices,
        "ranks_per_node": namespace.ranks_per_node, "quant": namespace.quant,
        "layers": namespace.layers, "n_routed_experts": namespace.experts,
        "num_experts_per_tok": namespace.top_k,
        "prompt_tokens": value("prompt_tokens", CLI_PROMPT_TOKENS),
        "output_tokens": value("output_tokens", CLI_OUTPUT_TOKENS),
        "concurrency": value("concurrency", CLI_ONE),
        "prefix_tokens": value("prefix_tokens", CLI_ZERO),
        "max_num_seqs": namespace.max_num_seqs,
        "max_num_batched_tokens": namespace.max_num_batched_tokens,
        "max_model_len": namespace.max_model_len,
        "async_scheduling": value("async_scheduling", False),
        "graph_capture_mode": namespace.graph_mode or "none",
        "mtp_proposal_tokens": value("mtp_tokens", CLI_ZERO),
        "mtp_acceptance_length": value("mtp_accept", CLI_MTP_ACCEPT),
        "ttft_slo_ms": value("ttft_slo_ms", CLI_TTFT_MS),
        "tpot_slo_ms": value("tpot_slo_ms", CLI_TPOT_MS),
        "va_recommended": bool(getattr(namespace, "va_recommended", False)),
        "o_proj_tp": value("o_proj_tp", CLI_ONE),
        "linear_proj_tp": _linear_tp(namespace),
        "lm_head_tp": value("lm_head_tp", CLI_ONE),
        "redundant_experts_per_rank": value("redundant_experts_per_rank", CLI_ZERO),
        "collective_startup_mode": namespace.collective_startup_mode or "legacy",
        "matrix_compute_policy": namespace.matrix_compute_policy or "auto",
        "recover_ssm": bool(namespace.recover_ssm),
        "prefix_cache_mode": namespace.prefix_cache_mode or "private",
        "prefix_read_mode": namespace.prefix_read_mode or "auto",
        "homogeneous_graph_reuse": namespace.homogeneous_graph_reuse or "off",
        "prefix_cache_key": namespace.prefix_cache_key or "",
        "hardware_derivation": _derivation_file(namespace),
        "aic_count": namespace.aic_count, "aiv_count": namespace.aiv_count,
        "memory_bytes": _memory_bytes(namespace),
        "hbm_peak_bytes_per_s": namespace.hbm_peak_bytes_per_s,
        "hbm_efficiency": namespace.hbm_efficiency,
        "mte2_ceiling_bytes_per_s": namespace.mte2_ceiling_bytes_per_s,
        "mte2_efficiency": namespace.mte2_efficiency,
    }


def _linear_tp(namespace: argparse.Namespace) -> int:
    direct = getattr(namespace, "linear_proj_tp", None)
    alias = getattr(namespace, "kda_proj_tp", None)
    if direct is not None and alias is not None and int(direct) != int(alias):
        raise ModelingError("runtime_feature_conflict:linear_proj_tp")
    value = alias if direct is None else direct
    return CLI_ONE if value is None else int(value)


def _memory_bytes(namespace: argparse.Namespace) -> int | None:
    gib = getattr(namespace, "memory_gib", None)
    return (int(round(float(gib) * GIB_BYTES)) if gib is not None
            else namespace.memory_bytes)


def _derivation_file(namespace: argparse.Namespace) -> dict[str, Any] | None:
    path = getattr(namespace, "hardware_derivation", None)
    return load_hardware_derivation(path) if path else None


def case_overrides(namespace: argparse.Namespace) -> dict[str, Any]:
    """Return only explicitly supplied feature/hardware values."""
    overrides: dict[str, Any] = {}
    for key in FEATURE_KEYS:
        value = getattr(namespace, key, None)
        if value is not None:
            overrides[key] = value
    kda = getattr(namespace, "kda_proj_tp", None)
    if kda is not None:
        linear = overrides.get("linear_proj_tp")
        if linear is not None and int(linear) != int(kda):
            raise ModelingError("runtime_feature_conflict:linear_proj_tp")
        overrides["linear_proj_tp"] = int(kda)
    memory_bytes = getattr(namespace, "memory_bytes", None)
    memory_gib = getattr(namespace, "memory_gib", None)
    if memory_bytes is not None and memory_gib is not None:
        raise ModelingError("hardware_derivation_memory_conflict")
    if memory_gib is not None:
        overrides["memory_bytes"] = int(round(float(memory_gib) * GIB_BYTES))
    elif memory_bytes is not None:
        overrides["memory_bytes"] = int(memory_bytes)
    for key in HARDWARE_KEYS:
        value = getattr(namespace, key, None)
        if value is not None:
            overrides[key] = value
    for key in ("hbm_efficiency", "mte2_efficiency"):
        if key in overrides:
            overrides[key] = _curve(overrides[key], key)
    path = getattr(namespace, "hardware_derivation", None)
    if path:
        document = load_hardware_derivation(path)
        for key, value in document.items():
            if key in overrides and overrides[key] != value:
                raise ModelingError(f"hardware_derivation_conflict:{key}")
            overrides[key] = value
    return overrides


def apply_case_overrides(case: Case, overrides: Mapping[str, Any]) -> Case:
    """Apply explicit canonical overrides to one parsed Case."""
    parallel_values = {
        key: overrides[key] for key in (
            "o_proj_tp", "linear_proj_tp", "lm_head_tp",
            "redundant_experts_per_rank") if key in overrides
    }
    parallel = replace(case.runtime.parallel, **parallel_values) \
        if parallel_values else case.runtime.parallel
    runtime_values = {
        key: overrides[key] for key in (
            "collective_startup_mode", "matrix_compute_policy", "recover_ssm",
            "prefix_cache_mode", "prefix_read_mode", "homogeneous_graph_reuse")
        if key in overrides
    }
    runtime = replace(case.runtime, parallel=parallel, **runtime_values)
    if "prefix_cache_key" in overrides:
        key = normalize_prefix_cache_key(overrides["prefix_cache_key"])
        case = replace(case, workload=replace(
            case.workload,
            requests=tuple(replace(request, prefix_cache_key=key)
                           for request in case.workload.requests)))
    validate_runtime_features(runtime)
    from mindie_sim.hardware_derivation import derived_machine
    machine = derived_machine(case.machine, overrides)
    return replace(case, runtime=runtime, machine=machine)


def apply_case_mapping_overrides(
    value: Mapping[str, Any], overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply overrides to a raw case mapping while preserving provenance."""
    if not overrides:
        return dict(value)
    from mindie_sim.prediction.structure import parse_case

    return apply_case_overrides(parse_case(value), overrides).to_dict()
