"""Deterministic Case synthesizer for catalog, preset, and custom models.

[Role]
  Resolve hardware SKU aliases and synthesize complete, closed-world simulation
  ``Case`` dictionaries from catalog models, local config paths, or inline
  mappings alongside explicit parallelism, workload, runtime, and SLO inputs.

[Invariants]
  * All hardware SKUs resolve against registered Ascend profiles; unknown SKUs
    raise ``ModelingError("unknown_hardware_sku:...")``.
  * Parallelism dimensions and structural overrides are validated explicitly;
    invalid dimensions raise ``ModelingError`` or ``ValueError`` without silent
    fallback.
  * Every synthesized dictionary is verified via ``parse_case`` before return.

[Timeouts & Locks]
  Bounded local JSON config reads (< 50 MiB); no network, NPU, subprocess, or
  file locks.

[Side-Effect Policy]
  Pure in-memory case synthesis; no filesystem writes.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from mindie_sim.backends.ascend import (
    materialize as materialize_ascend,
    profiles as ascend_profiles,
)
from mindie_sim.catalog.specs import build_model_spec
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.model_catalog import (
    ARCHITECTURE_PRESET_ALIASES,
    ARCHITECTURE_PRESET_CATALOG,
    CATALOG,
    CONFIG_DIR,
    MODEL_ALIASES,
    customize_model_spec,
    load_model,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides


HARDWARE_ALIASES: dict[str, str] = {
    "a2": "Ascend910B4", "910b": "Ascend910B4", "910b4": "Ascend910B4",
    "910b1": "Ascend910B1", "910b2": "Ascend910B2", "910b3": "Ascend910B3",
    "a3": "Ascend910_9392", "910c": "Ascend910_9392", "910_9392": "Ascend910_9392",
    "910_9382": "Ascend910_9382",
    "a5": "Ascend910_9599", "950": "Ascend910_9599", "910_9599": "Ascend910_9599",
}

_EXTRA_QUANT_SPECS: dict[str, QuantSpec] = {
    "fp16": QuantSpec("fp16", "fp16", "fp16", "fp16", "fp16", "fp16", "fp16"),
    "mxfp4": QuantSpec(
        name="MXFP4", weight_format="mxfp4", activation_format="fp8_e4m3",
        expert_weight_format="mxfp4", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0, kv_dtype="bf16",
    ),
}


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _ensure_registered_profiles() -> dict[str, Any]:
    reg = ascend_profiles()
    if "Ascend910_9599" not in reg and "Ascend950DT_9582" in reg:
        reg["Ascend910_9599"] = deepcopy(reg["Ascend950DT_9582"])
    return reg


def resolve_hardware_sku(hardware: str) -> str:
    """Resolve a hardware alias or exact SKU name against registered profiles."""
    if not isinstance(hardware, str) or not hardware.strip():
        raise ModelingError(f"unknown_hardware_sku:{hardware}")
    reg = _ensure_registered_profiles()
    raw = hardware.strip()
    lower = raw.lower()
    if lower in HARDWARE_ALIASES:
        sku = HARDWARE_ALIASES[lower]
        if sku in reg:
            return sku
    if raw in reg:
        return raw
    for registered_sku in reg:
        if registered_sku.lower() == lower:
            return registered_sku
    raise ModelingError(f"unknown_hardware_sku:{hardware}")


def _read_config_json(path: Path) -> dict[str, Any]:
    target = path / "config.json" if path.is_dir() else path
    if not target.is_file():
        raise ModelingError(f"model_config_not_found:{path}")
    if target.stat().st_size > 50 * 1024**2:
        raise ValueError("metadata_input_exceeds_50_MiB")
    with target.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ModelingError(f"model_config_invalid_mapping:{path}")
    return dict(data)


def _resolve_model_spec(model: str | Path | Mapping[str, Any]) -> ModelSpec:
    if isinstance(model, Mapping):
        raw = dict(model)
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        name = str(raw.get("name") or raw.get("model_type") or "inline-model")
        return build_model_spec(name, raw)

    if isinstance(model, Path):
        raw = _read_config_json(model)
        name = str(raw.get("name") or (model.name if model.is_dir() else model.stem))
        if "layers" in raw and "hidden_size" in raw and "model_type" not in raw:
            return ModelSpec.from_mapping(raw)
        return build_model_spec(name, raw)

    if not isinstance(model, str) or not model.strip():
        raise ModelingError(f"invalid_model_identifier:{model}")

    key = model.strip()
    canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
    if canonical in CATALOG or canonical in ARCHITECTURE_PRESET_CATALOG:
        return load_model(canonical)

    candidate = Path(key)
    if candidate.exists():
        return _resolve_model_spec(candidate)

    cfg_candidate = CONFIG_DIR / (key if key.endswith(".json") else f"{key}.json")
    if cfg_candidate.is_file():
        return _resolve_model_spec(cfg_candidate)

    return load_model(key)


def _apply_quant_override(
    model_spec: ModelSpec,
    quant: str | Mapping[str, Any] | None,
    machine: Mapping[str, Any],
) -> ModelSpec:
    if quant is None:
        peaks = machine.get("device", {}).get("compute_peaks", {})
        has_fp8 = any(k in peaks for k in ("fp8", "fp8_e4m3", "mxfp4"))
        if not has_fp8 and model_spec.quant is not None:
            wf = str(model_spec.quant.weight_format)
            ewf = str(model_spec.quant.expert_weight_format)
            if "fp8" in wf or "fp8" in ewf or "mxfp4" in ewf:
                return apply_model_overrides(model_spec, {"quantization": "w8a8"})
        return model_spec

    if isinstance(quant, str):
        qkey = quant.strip().lower()
        if qkey in _EXTRA_QUANT_SPECS:
            qspec = _EXTRA_QUANT_SPECS[qkey]
            return apply_model_overrides(
                model_spec, {"quant": qspec.to_dict(), "kv_dtype": qspec.kv_dtype},
            )
        return apply_model_overrides(model_spec, {"quantization": qkey})
    if isinstance(quant, Mapping):
        qdict = dict(quant)
        return apply_model_overrides(
            model_spec,
            {"quant": qdict, "kv_dtype": str(qdict.get("kv_dtype", model_spec.kv_dtype))},
        )
    raise ModelingError(f"invalid_quant_override:{quant}")


def _resolve_max_position_embeddings(
    model: str | Path | Mapping[str, Any], model_spec: ModelSpec,
) -> int:
    for src in (
        getattr(model_spec, "max_position_embeddings", 0),
        dict(model_spec.catalog_facts).get("max_position_embeddings", 0),
        dict(model_spec.metadata).get("max_position_embeddings", 0),
    ):
        if src:
            return int(src)
    raw: Mapping[str, Any] | None = model if isinstance(model, Mapping) else None
    if isinstance(model, Path):
        raw = _read_config_json(model)
    elif isinstance(model, str):
        key = model.strip()
        canonical = MODEL_ALIASES.get(key, ARCHITECTURE_PRESET_ALIASES.get(key, key))
        entries = {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}
        cfg_path = (CONFIG_DIR / entries[canonical]) if canonical in entries else Path(key)
        if cfg_path.exists():
            raw = _read_config_json(cfg_path)
    if isinstance(raw, Mapping):
        tc = raw.get("text_config") if isinstance(raw.get("text_config"), Mapping) else {}
        val = raw.get("max_position_embeddings") or tc.get("max_position_embeddings")
        if val:
            return int(val)
    return 0


def _apply_structural_overrides(
    model_spec: ModelSpec, *, layers: int | None, n_routed_experts: int | None, num_experts_per_tok: int | None,
) -> ModelSpec:
    if layers is None and n_routed_experts is None and num_experts_per_tok is None:
        return model_spec
    return customize_model_spec(
        model_spec, layers=layers, n_routed_experts=n_routed_experts, num_experts_per_tok=num_experts_per_tok,
    )


def compose_case(
    model: str | Path | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    tp: int = 1,
    dp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    dcp: int = 1,
    sp: bool = False,
    num_devices: int | None = None,
    ranks_per_node: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    layers: int | None = None,
    n_routed_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    prompt_tokens: int = 4096,
    output_tokens: int = 128,
    concurrency: int = 1,
    prefix_tokens: int = 0,
    arrival_gap_ms: float = 0.0,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    block_size: int = 128,
    enable_chunked_prefill: bool = True,
    async_scheduling: bool = False,
    graph_capture_mode: str = "none",
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    ttft_slo_ms: float = 5000.0,
    tpot_slo_ms: float = 50.0,
) -> dict[str, Any]:
    """Synthesize and validate a complete simulation Case dictionary."""
    for name, val in (("tp", tp), ("dp", dp), ("ep", ep), ("pp", pp), ("pcp", pcp), ("dcp", dcp)):
        if int(val) <= 0:
            raise ModelingError(f"invalid_parallel_dimension:{name}={val}")
    tp, dp, ep, pp, pcp, dcp = int(tp), int(dp), int(ep), int(pp), int(pcp), int(dcp)
    if not (ep == 1 or (dp * tp) % ep == 0 or ep == dp * tp):
        raise ModelingError(f"invalid_ep_partition:ep={ep},dp={dp},tp={tp}")
    if int(prompt_tokens) <= 0 or int(output_tokens) <= 0 or int(concurrency) <= 0:
        raise ModelingError("invalid_workload_dimensions")
    if int(prefix_tokens) < 0 or int(prefix_tokens) > int(prompt_tokens):
        raise ModelingError("invalid_prefix_tokens")
    if graph_capture_mode not in {"none", "full", "piecewise"}:
        raise ModelingError(f"invalid_graph_capture_mode:{graph_capture_mode}")
    if int(mtp_proposal_tokens) < 0:
        raise ModelingError(f"invalid_mtp_proposal_tokens:{mtp_proposal_tokens}")

    required_devices = dp * tp * pp * pcp
    hw_raw = dict(hardware) if isinstance(hardware, Mapping) else {}
    sku = resolve_hardware_sku(str(hw_raw.get("sku", "a3")) if hw_raw else str(hardware))
    resolved_devices = int(
        num_devices if num_devices is not None else hw_raw.get("num_devices", required_devices)
    )
    if resolved_devices < required_devices:
        raise ModelingError(f"insufficient_num_devices:{resolved_devices}:required:{required_devices}")
    default_rpn = min(resolved_devices, 16 if sku.startswith("Ascend910_93") else 8)
    resolved_rpn = int(
        ranks_per_node if ranks_per_node is not None else hw_raw.get("ranks_per_node", default_rpn)
    )
    machine_input = {**hw_raw, "sku": sku, "num_devices": resolved_devices, "ranks_per_node": resolved_rpn}
    machine = (
        machine_input
        if "device" in machine_input and "links" in machine_input
        else materialize_ascend(machine_input)
    )

    model_spec = _resolve_model_spec(model)
    model_limit = _resolve_max_position_embeddings(model, model_spec)
    model_spec = _apply_quant_override(model_spec, quant, machine)
    model_spec = _apply_structural_overrides(
        model_spec,
        layers=layers,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if int(mtp_proposal_tokens) > 0:
        has_mtp = getattr(model_spec.auxiliary, "mtp", None) is not None
        has_dspark = getattr(model_spec.auxiliary, "dspark", None) is not None
        if not (has_mtp or has_dspark):
            raise ModelingError("draft_structure_required_for_mtp_proposal_tokens")
        if not (1.0 <= float(mtp_acceptance_length) <= float(1 + int(mtp_proposal_tokens))):
            raise ModelingError("invalid_mtp_acceptance_length")

    required_len = int(prompt_tokens) + int(output_tokens) + max(0, int(mtp_proposal_tokens))
    if max_model_len is not None:
        if int(max_model_len) < required_len:
            raise ModelingError(f"max_model_len_smaller_than_workload:{max_model_len}<{required_len}")
        resolved_max_model_len = int(max_model_len)
    else:
        if model_limit > 0 and required_len > model_limit:
            raise ModelingError(
                f"workload_exceeds_model_max_position_embeddings:{required_len}>{model_limit}"
            )
        resolved_max_model_len = max(32768, required_len)

    model_dict = _to_jsonable(model_spec)
    model_dict["n_routed_experts"] = int(model_spec.num_experts)
    model_dict["num_experts_per_tok"] = int(model_spec.top_k)

    resolved_max_seqs = int(max_num_seqs) if max_num_seqs is not None else max(int(concurrency), 16)
    resolved_max_batched = (
        int(max_num_batched_tokens)
        if max_num_batched_tokens is not None
        else max(int(prompt_tokens) * int(concurrency), 8192)
    )
    requests = [
        {
            "id": f"req-{idx}",
            "arrival_ms": float(idx * float(arrival_gap_ms)),
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "prefix_tokens": int(prefix_tokens),
        }
        for idx in range(int(concurrency))
    ]
    runtime_meta: dict[str, Any] = {"sp": bool(sp)}
    if int(mtp_proposal_tokens) > 0:
        runtime_meta["draft_mode"] = "mtp" if getattr(model_spec.auxiliary, "mtp", None) is not None else "dspark"
        runtime_meta["draft_proposal_tokens"] = int(mtp_proposal_tokens)

    runtime_dict: dict[str, Any] = {
        "parallel": {"dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": dcp, "dcp": dcp, "pcp": pcp},
        "max_num_seqs": resolved_max_seqs,
        "max_num_batched_tokens": resolved_max_batched,
        "max_model_len": resolved_max_model_len,
        "kv_block_size": int(block_size),
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "async_scheduling": bool(async_scheduling),
        "graph_mode": str(graph_capture_mode),
        "graph_capture_mode": str(graph_capture_mode),
        "mtp_acceptance_length": float(mtp_acceptance_length),
        "metadata": runtime_meta,
    }
    if graph_capture_mode == "full":
        max_cap = max(1, resolved_max_seqs * max(1, 1 + int(mtp_proposal_tokens)))
        runtime_dict["graph_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, max_cap})
    elif graph_capture_mode == "piecewise":
        runtime_dict["piecewise_capture_keys"] = sorted(
            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, resolved_max_batched}
        )
    if sp:
        from mindie_sim.prediction.sequence_parallel import DECLARATION, _validate_model
        if pp != 1 or pcp != 1 or tp <= 1 or ep < tp or (ep % tp != 0):
            raise ModelingError(
                f"sequence_parallel_invalid_topology:tp={tp},ep={ep},pp={pp},pcp={pcp}"
            )
        _validate_model(model_spec)
        runtime_dict["kernels"] = {
            "sequence_parallel": dict(DECLARATION, padding="right_to_tp")
        }

    case_dict: dict[str, Any] = {
        "model": model_dict,
        "machine": _to_jsonable(machine),
        "runtime": runtime_dict,
        "workload": {"requests": requests},
        "slo": {"ttft_ms": float(ttft_slo_ms), "tpot_ms": float(tpot_slo_ms)},
    }
    parse_case(case_dict)
    return case_dict


__all__ = ["HARDWARE_ALIASES", "compose_case", "resolve_hardware_sku"]
