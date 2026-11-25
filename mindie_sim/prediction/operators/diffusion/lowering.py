"""Diffusion operator lowering and denoising summaries.

[Role] Lower declared diffusion operators and summarize denoising work.
[Invariants] Explicit operator names and rate declarations determine lowering and summaries.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI
from mindie_sim.schema.physical_mechanisms import DAV_2201_PRESET, DAV_3510_PRESET

from mindie_sim.prediction.values.typed import _prepend_source_value_contract, typed_value

from .attention import _norm_rope_contract, _sparse_contract
from .block_ops import (
    _adalayernorm_contract,
    _eagle_ffn_contract,
    _eagle_ffn_unfused_ops,
    _laurel_contract,
    _quant_four_over_six_contract,
)
from .contracts import _input_dtype
from .registry import MINDIE_SD_VENDOR, _custom_ops, _resolve_op_name

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="csrc/plugin/register_ops.cpp",
        source_sha256="861fc97a28905b4e9c2ea39f8cd5641725b7b21f355ca8bb237446b5ae86c08a",
    ),
    SourceRef(
        repository="Ascend/MindIE-SD",
        revision="9fe772dfb90696e21b5e5d4e92ca07dffab84cf4",
        relative_path="mindiesd/layers/eagle_ffn_linear.py",
        source_sha256="9ac63b1ed1da06c35d079a132bdd52191061e8d9863d21656883f65a7adc1f84",
    ),
)

_SOC_ALIASES = {
    "a2": "Ascend910B",
    "ascend910b": "Ascend910B",
    "a3": "Ascend910_93",
    "ascend910_93": "Ascend910_93",
    "a5": "Ascend950",
    "arch35": "Ascend950",
    "ascend950": "Ascend950",
}
_AUX_OUTPUT_NAMES = frozenset({"mean", "rstd", "softmax_lse"})


def _resolve_soc_family(soc_family: str | None, device: Any | None) -> str | None:
    """Resolve canonical Ascend SoC family from Device first, falling back to kwarg."""
    dev_soc = None
    if device is not None:
        target = device.dies[0] if getattr(device, "dies", None) else device
        arch_obj = getattr(target, "compute_arch", None)
        arch_id = str(
            getattr(arch_obj, "arch_id", "") or getattr(arch_obj, "preset", "") or ""
        ).strip().lower()
        tokens = " ".join(str(getattr(target, attr, "") or "") for attr in ("arch_family", "sku", "name")).lower()
        if (arch_id in {DAV_3510_PRESET, "arch35", "a5"} or "3510" in arch_id
                or any(tok in tokens for tok in ("a5", "arch35", "dav_3510", "ascend950", "950"))):
            dev_soc = "Ascend950"
        elif any(tok in tokens for tok in ("910b", "a2")) or arch_id == "a2":
            dev_soc = "Ascend910B"
        elif (arch_id in {DAV_2201_PRESET, "arch22", "a3"} or "2201" in arch_id
              or any(tok in tokens for tok in ("910_93", "9391", "9382", "9362", "a3"))):
            dev_soc = "Ascend910_93"
    if dev_soc is not None:
        return dev_soc
    if soc_family is None:
        return None
    raw = str(soc_family).strip()
    low = raw.lower()
    if low in _SOC_ALIASES:
        return _SOC_ALIASES[low]
    if low.startswith("ascend950"):
        return "Ascend950"
    if low.startswith("ascend910_93"):
        return "Ascend910_93"
    if low.startswith("ascend910b"):
        return "Ascend910B"
    return raw


def _contract_typed_outputs(op: OpSpec, contract: dict[str, Any], resolved: str) -> tuple[dict[str, Any], ...]:
    """Build exact materialized typed output descriptors directly from contract writes."""
    entries = []
    for item in contract["writes"]:
        if item.get("kind") not in {"value", "legacy", "declared"}:
            continue
        if not item.get("materialized", True):
            continue
        name = str(item["name"])
        role = (
            "scale" if "scale" in name
            else ("auxiliary" if name in _AUX_OUTPUT_NAMES else "activation")
        )
        entries.append(typed_value(
            producer=op.op_id,
            dtype=str(item["dtype"]),
            shape=tuple(item["shape"]),
            output=name,
            role=role,
            source=f"{op.op_id}:{item.get('source') or resolved}",
            materialized=True,
            alias_of=str(item.get("alias") or ""),
            write_bytes=float(item["bytes"]),
            contract="mindie_sd_contract",
        ))
    return tuple(entries)


def lower_op(
    op: OpSpec,
    *,
    op_name: str | None = None,
    soc_family: str | None = None,
    device: Any | None = None,
    sparsity_ratio: float | None = None,
    low_rank: int | None = None,
    block_size: int | None = None,
    kv_dtype: str | None = None,
) -> OpSpec | tuple[OpSpec, ...]:
    """Lower one exact semantic occurrence into a MindIE-SD custom operator."""
    resolved = _resolve_op_name(op, op_name)
    spec = _custom_ops()[resolved]
    if op.family not in spec["families"] and op.recipe not in spec["families"]:
        raise ModelingError(f"mindie_sd_custom_op_family_mismatch:{resolved}:{op.family}")
    resolved_soc = _resolve_soc_family(soc_family, device)
    meta = dict(op.metadata)
    dtype = _input_dtype(op, meta)
    if dtype not in spec["input_dtypes"]:
        raise ModelingError(
            f"mindie_sd_custom_op_input_dtype_unsupported:{resolved}:{dtype}")
    if resolved_soc is not None and resolved_soc not in spec["supported_socs"]:
        if resolved == "eagle_ffn" and resolved_soc in spec.get("unfused_fallback_socs", ()):
            return _eagle_ffn_unfused_ops(op, meta, dtype, resolved_soc)
        raise ModelingError(f"mindie_sd_custom_op_soc_unsupported:{resolved}:{resolved_soc}")

    flops = float(op.flops)
    vec_ops = float(op.vector_ops)
    read_bytes = float(op.mte2_bytes)
    write_bytes = float(op.mte3_bytes)
    returns = tuple(op.returns)
    contract = None
    matrix_flops = op.matrix_flops
    options = {"sparsity_ratio": sparsity_ratio, "block_size": block_size,
               "low_rank": low_rank, "kv_dtype": kv_dtype}

    if spec.get("sparse"):
        contract = _sparse_contract(op, meta, dtype, returns, resolved, options)
        if contract.get("attributes", {}).get("kv_dtype") is None and spec.get("quantized_cache"):
            raise ModelingError("mindie_sd_eagle_qbsa_requires_declared_cache_dtype")
    elif resolved == "norm_rope_concat":
        contract = _norm_rope_contract(op, meta, dtype, returns)
    elif resolved == "adalayernorm":
        contract = _adalayernorm_contract(op, meta, dtype, returns, soc_family=resolved_soc)
    elif resolved == "laurel":
        if low_rank is not None:
            meta["low_rank"] = int(low_rank)
        contract = _laurel_contract(op, meta, dtype, returns)
    elif resolved == "quant_four_over_six_a5":
        contract = _quant_four_over_six_contract(op, meta, dtype, returns)
    elif resolved == "eagle_ffn":
        contract = _eagle_ffn_contract(op, meta, dtype, returns)

    output_dtype = op.dtype or dtype
    output_shape = tuple(op.local_shape)
    precision_abi = op.precision_abi
    if contract is not None:
        read_bytes = float(contract["mte2_bytes"])
        write_bytes = float(contract["mte3_bytes"])
        vec_ops = float(contract["vector_ops"])
        flops = float(contract["logical_flops"])
        matrix_flops = tuple(contract["matrix_flops"])
        meta["mindie_sd_contract"] = contract
        meta.update(contract["attributes"])
        if resolved == "laurel":
            meta["laurel_rank"] = int(contract["attributes"]["low_rank"])
        public_writes = tuple(
            item for item in contract["writes"]
            if item.get("materialized", True)
            and item.get("kind") in {"value", "legacy", "declared"}
        )
        public = next(
            (item for item in public_writes if item["name"] not in _AUX_OUTPUT_NAMES),
            public_writes[0] if public_writes else None,
        )
        if public is not None:
            output_shape = tuple(public["shape"])
            output_dtype = str(public["dtype"])
            meta["primary_output"] = str(public["name"])
            meta["output_format"] = output_dtype
            if isinstance(precision_abi, TensorPrecisionABI):
                precision_abi = replace(precision_abi, output_dtype=output_dtype)
                meta["precision_abi_signature"] = precision_abi.abi_signature
        returns = tuple(dict.fromkeys(str(item["name"]) for item in public_writes))
        typed_contract = _contract_typed_outputs(op, contract, resolved)
        if typed_contract:
            _prepend_source_value_contract(meta, op, typed_contract)

    meta.update({
        "mindie_sd_custom_op": f"mindie_sd_{resolved}",
        "native_vendor": MINDIE_SD_VENDOR,
        "native_kind": spec["native_kind"],
        "bound": spec["bound"],
        "native_implementation": {
            "framework": "mindie-sd",
            "vendor": MINDIE_SD_VENDOR,
            "native_kinds": (spec["native_kind"],),
            "op_name": f"mindie_sd_{resolved}",
            "selection": "explicit_caller_implementation",
            "soc_family": resolved_soc,
        },
    })
    return replace(
        op,
        recipe=f"mindie_sd_{resolved}",
        dtype=output_dtype,
        flops=flops,
        vector_ops=vec_ops,
        mte2_bytes=read_bytes,
        mte3_bytes=write_bytes,
        hbm_bytes=read_bytes + write_bytes,
        returns=returns,
        matrix_flops=matrix_flops,
        precision_abi=precision_abi,
        local_shape=output_shape,
        padded_shape=output_shape,
        launch_shape=output_shape,
        metadata=tuple(meta.items()),
    )


def compute_mindie_sd_effective_denoising_steps(
    infer_steps: int,
    warmup_steps: int = 2,
    cache_interval: int = 2,
    cache_hit_ratio: float | None = None,
) -> dict[str, float | int]:
    """Model DiTCache/TeaCache/TaylorSeer denoising step skipping."""
    total = int(infer_steps)
    warmup = max(0, min(int(warmup_steps), total))
    interval = max(1, int(cache_interval))
    if total <= 0:
        raise ModelingError(f"mindie_sd_invalid_infer_steps:{infer_steps}")
    remaining = max(0, total - warmup)
    if cache_hit_ratio is not None:
        hit = max(0.0, min(1.0, float(cache_hit_ratio)))
        cached = int(round(remaining * hit))
        computed = max(1, total - cached)
        cached = total - computed
    else:
        computed_after_warmup = math.ceil(remaining / interval) if remaining > 0 else 0
        computed = max(1, warmup + computed_after_warmup)
        cached = max(0, total - computed)
    ratio = float(computed) / float(total)
    return {
        "infer_steps": total,
        "warmup_steps": warmup,
        "cache_interval": interval,
        "computed_steps": computed,
        "cached_steps": cached,
        "cache_hit_ratio": float(cached) / float(total),
        "effective_step_ratio": ratio,
        "speedup_factor": float(total) / float(computed),
    }
