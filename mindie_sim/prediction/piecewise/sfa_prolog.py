"""Sparse SFA C8 MX MlaPrologV3 visible ports.

[Role] Select the consumed fused prolog and name the tensors it writes.
[Invariants] Selection is the pinned caller: sparse SFA, FP8 or INT8 cache,
  an indexer on this layer, and MX FP8 on the MLA projection category.
  Shape equality does not select. The packed cache capacity is not a port.
  Group-32 E8M0 scale bytes come from the existing MX producer contract.
[Timeouts & Locks] Pure arithmetic. No I/O, device, or locks.
[Side-Effect Policy] Returns immutable numbers. Does not edit M8 producers.
"""
from __future__ import annotations

from fractions import Fraction

from mindie_sim.prediction.mx_dynamic_quant import producer_contract
from mindie_sim.schema.common import ModelingError

SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "c09d3e20b9c45c80b34ae3781440d90904a44023",
    "caller": "vllm_ascend/attention/sfa_v1.py",
    "caller_lines": "1027-1043,1389-1509",
    "c8_gate": "vllm_ascend/ascend_config.py:928",
    "packed_width": "vllm_ascend/attention/utils.py:get_sfa_qsfa_packed_head_dim",
    "tile": 128,
    "weight_quant_mode": 3,
    "kv_cache_quant_mode": 3,
    "binary_identity": "not_established",
    "tiling": "not_reconstructed",
}

_C8_CACHE = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "int8"})
_PORTS = ("ql_nope", "q_pe", "q_c", "q_c_scale", "cache_index")
_FIELDS = _PORTS + ("packed_row", "store_row")


def packed_slot_bytes(kv_lora: int, rope: int, tile: int = 128) -> int:
    """Source packed slot: 1-byte latent, BF16 rope, FP32 scales per tile."""
    kv_lora, rope, tile = int(kv_lora), int(rope), int(tile)
    if tile < 1 or kv_lora < 1 or kv_lora % tile != 0 or rope < 1:
        raise ModelingError("sfa_prolog_packed_slot_invalid")
    return kv_lora + rope * 2 + (kv_lora // tile) * 4


def _topk(model, layer) -> int:
    mla = getattr(model, "mla", None)
    model_topk = int(getattr(mla, "index_topk", 0) or 0) if mla is not None else 0
    return int(getattr(layer, "index_topk", 0) or 0) or model_topk


def _compressed(model, layer) -> bool:
    if str(getattr(layer, "cache_kind", "dense") or "dense") != "dense":
        return True
    if int(getattr(layer, "compress_ratio", 0) or 0) > 0:
        return True
    mla = getattr(model, "mla", None)
    return mla is not None and int(getattr(mla, "compress_ratio", 0) or 0) > 0


def sfa_prolog_v3_c8_mx(model, layer, features, *, q_lora: int, kv_lora: int, rope: int) -> bool:
    """True only for the pinned SFA PROLOG_V3 C8 MX caller on this layer."""
    if int(q_lora) <= 0 or int(rope) <= 0 or int(kv_lora) <= 0:
        return False
    if int(kv_lora) % 128 != 0:
        return False
    if not bool(getattr(layer, "indexer", False)):
        return False
    if _topk(model, layer) <= 0 or _compressed(model, layer):
        return False
    kv = str(getattr(features, "kv_dtype", "") or getattr(model, "kv_dtype", "") or "").lower()
    if kv not in _C8_CACHE:
        return False
    quant = getattr(features, "quant", None) or getattr(model, "quant", None)
    if quant is None or not hasattr(quant, "linear_method"):
        return False
    return quant.linear_method("mla") == "mx_fp8"


R64_WUK = {"batch_heads": 64, "k": 192, "n": 512, "dtype": "bf16"}


def mla_prolog_v3_r64_primary(model, layer, features, *, q_lora: int,
                              kv_lora: int, rope: int, nope: int) -> bool:
    """Primary R64 SplitN prolog shape, including non-indexer layers.

    Pinned R64 caller keeps the same four-product SplitN program for all78
    layers; only query_norm_flag (indexer) changes the optional qc outputs.
    This is deliberately a narrow shape/caller selector, not a model-label
    or case selector.
    """
    if (int(q_lora), int(kv_lora), int(rope), int(nope)) != (2048, 512, 64, 192):
        return False
    heads = int(getattr(layer, "num_heads", 0) or getattr(model, "num_heads", 0) or 0)
    if heads != 64:
        return False
    meta = dict(getattr(features, "metadata", ()) or ())
    declared = str(meta.get("mx_quant_source_revision", "")).lower()
    explicit = str(meta.get("mla_prolog_source", "")).lower()
    # Shape/dtype alone is not source selection: require a witnessed R64
    # deployment or the existing explicit native source declaration.
    if declared != "r64" and explicit != "r64_split_n":
        return False
    kv = str(getattr(features, "kv_dtype", "") or getattr(model, "kv_dtype", "") or "").lower()
    if kv not in _C8_CACHE:
        return False
    quant = getattr(features, "quant", None) or getattr(model, "quant", None)
    return (quant is not None and hasattr(quant, "linear_method")
            and quant.linear_method("mla") == "mx_fp8")


def port_spec(*, heads: int, q_lora: int, kv_lora: int, rope: int, store_row: int) -> str:
    """Per-token visible bytes plus the packed slot and the unfused store row."""
    heads, q_lora = int(heads), int(q_lora)
    if heads < 1 or int(store_row) <= 0:
        raise ModelingError("sfa_prolog_port_spec_invalid")
    scale = int(producer_contract(q_lora)["scale_bytes"])
    fields = {
        "ql_nope": heads * int(kv_lora) * 2,
        "q_pe": heads * int(rope) * 2,
        "q_c": q_lora,
        "q_c_scale": scale,
        "cache_index": 8,
        "packed_row": packed_slot_bytes(kv_lora, rope),
        "store_row": int(store_row),
    }
    return ",".join(f"{name}={fields[name]}" for name in _FIELDS)


def parse_port_spec(raw: str) -> dict[str, int]:
    fields = {}
    for item in str(raw).split(","):
        key, _, value = item.partition("=")
        if key not in _FIELDS or not value:
            raise ModelingError(f"sfa_prolog_port_spec_invalid:{raw}")
        fields[key] = int(value)
    if set(fields) != set(_FIELDS) or min(fields.values()) <= 0:
        raise ModelingError(f"sfa_prolog_port_spec_invalid:{raw}")
    return fields


def visible_adjustment(instance, outputs) -> dict[str, Fraction] | None:
    """Visible port bytes and the store-traffic delta for one fused prolog.

    ``store_delta`` retargets the member store to selected packed slots.
    It is pipe work, not an output allocation. ``visible_write`` is the
    absorbed query, the rope half, and the quantized query norm.
    """
    raw = dict(getattr(instance.template, "contract", ()) or ()).get("sfa_prolog_v3")
    if not raw:
        return None
    fields = parse_port_spec(str(raw))
    traffic = Fraction(outputs.get("kv_cache", 0))
    store_row = fields["store_row"]
    if traffic % store_row != 0:
        raise ModelingError("sfa_prolog_store_traffic_not_token_aligned")
    tokens = traffic // store_row
    kept = {name: Fraction(tokens * fields[name]) for name in _PORTS}
    return {
        "kept_ql_nope": kept["ql_nope"],
        "kept_q_pe": kept["q_pe"],
        "kept_q_c": kept["q_c"],
        "kept_q_c_scale": kept["q_c_scale"],
        "kept_cache_index": kept["cache_index"],
        "store_delta": Fraction(tokens * fields["packed_row"]) - traffic,
        "index_read": kept["cache_index"],
        "visible_write": kept["ql_nope"] + kept["q_pe"] + kept["q_c"] + kept["q_c_scale"],
    }
