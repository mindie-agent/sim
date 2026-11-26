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
SINGLE_RECORD = 1
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
            document = records[0]
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
    """Build the complete compose_case keyword set from CLI arguments.

    Preserves ``None`` for omitted parallel, scenario, and Domain-2 flags so
    ``compose_case`` enforces Domain-1 fail-fast, Domain-2 optimal defaults,
    and Domain-3 auto-optimal search without CLI-layer constant guessing.
    """
    value = lambda name, default: (default if getattr(namespace, name, None) is None
                                   else getattr(namespace, name))
    return {
        "tp": getattr(namespace, "tp", None),
        "dp": getattr(namespace, "dp", None),
        "ep": getattr(namespace, "ep", None),
        "pp": getattr(namespace, "pp", None),
        "pcp": getattr(namespace, "pcp", None),
        "dcp": getattr(namespace, "dcp", None),
        "sp": getattr(namespace, "sp", None),
        "num_devices": getattr(namespace, "num_devices", None),
        "ranks_per_node": getattr(namespace, "ranks_per_node", None),
        "quant": getattr(namespace, "quant", None),
        "layers": getattr(namespace, "layers", None),
        "n_routed_experts": getattr(namespace, "experts", None),
        "num_experts_per_tok": getattr(namespace, "top_k", None),
        "prompt_tokens": getattr(namespace, "prompt_tokens", None),
        "output_tokens": getattr(namespace, "output_tokens", None),
        "concurrency": getattr(namespace, "concurrency", None),
        "prefix_tokens": value("prefix_tokens", CLI_ZERO),
        "max_num_seqs": getattr(namespace, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(namespace, "max_num_batched_tokens", None),
        "max_model_len": getattr(namespace, "max_model_len", None),
        "async_scheduling": value("async_scheduling", False),
        "graph_capture_mode": getattr(namespace, "graph_mode", None),
        "mtp_proposal_tokens": getattr(namespace, "mtp_tokens", None),
        "mtp_acceptance_length": getattr(namespace, "mtp_accept", None),
        "ttft_slo_ms": getattr(namespace, "ttft_slo_ms", None),
        "tpot_slo_ms": getattr(namespace, "tpot_slo_ms", None),
        "va_recommended": getattr(namespace, "va_recommended", None),
        "raw_eager": bool(getattr(namespace, "raw_eager", False)),
        "engram_cpu_offload": getattr(namespace, "engram_cpu_offload", None),
        "multistream_overlap_shared_expert": getattr(
            namespace, "multistream_overlap_shared_expert", None
        ),
        "enable_shared_expert_dp": getattr(namespace, "enable_shared_expert_dp", None),
        "parallel": getattr(namespace, "parallel", None),
        "deployment": getattr(namespace, "deployment", None),
        "role": getattr(namespace, "role", None),
        "topology_preset": getattr(namespace, "topology_preset", None),
        "kv_cache_dtype": getattr(namespace, "kv_cache_dtype", None),
        "indexer_kv_dtype": getattr(namespace, "indexer_kv_dtype", None),
        "require_scenario": not bool(getattr(namespace, "structure_only", False)),
        "o_proj_tp": value("o_proj_tp", CLI_ONE),
        "linear_proj_tp": _linear_tp(namespace),
        "lm_head_tp": value("lm_head_tp", CLI_ONE),
        "redundant_experts_per_rank": value("redundant_experts_per_rank", CLI_ZERO),
        "collective_startup_mode": getattr(namespace, "collective_startup_mode", None),
        "matrix_compute_policy": getattr(namespace, "matrix_compute_policy", None) or "auto",
        "recover_ssm": getattr(namespace, "recover_ssm", None),
        "prefix_cache_mode": getattr(namespace, "prefix_cache_mode", None) or "private",
        "prefix_read_mode": getattr(namespace, "prefix_read_mode", None) or "auto",
        "homogeneous_graph_reuse": getattr(namespace, "homogeneous_graph_reuse", None) or "off",
        "prefix_cache_key": getattr(namespace, "prefix_cache_key", None) or "",
        "hardware_derivation": _derivation_file(namespace),
        "aic_count": getattr(namespace, "aic_count", None),
        "aiv_count": getattr(namespace, "aiv_count", None),
        "memory_bytes": _memory_bytes(namespace),
        "hbm_peak_bytes_per_s": getattr(namespace, "hbm_peak_bytes_per_s", None),
        "hbm_efficiency": getattr(namespace, "hbm_efficiency", None),
        "mte2_ceiling_bytes_per_s": getattr(namespace, "mte2_ceiling_bytes_per_s", None),
        "mte2_efficiency": getattr(namespace, "mte2_efficiency", None),
        "block_size": getattr(namespace, "block_size", None),
        "runtime_stack": (getattr(namespace, "stack_manifest", None)
                          or getattr(namespace, "stack", None)),
        "feature_track": getattr(namespace, "feature_track", None) or "shipping",
        "optimizations": tuple(getattr(namespace, "enable_opt", ()) or ()),
        "exclude_optimizations": tuple(getattr(namespace, "disable_opt", ()) or ()),
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
