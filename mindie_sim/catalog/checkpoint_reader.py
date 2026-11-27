"""Streaming, runtime-agnostic on-disk checkpoint quantization reader.

[Role] Stream local ModelSlim ``quant_model_description*.json`` files (including
  ``> 50 MiB`` descriptors), parse HF ``config.json["quantization_config"]``
  metadata (``mxfp8``, ``fp8``, ``compressed-tensors``, ``awq``, ``gptq``,
  ``quark``) and pure native checkpoints into lossless L0 ``CheckpointQuantMap``
  records, and load the packaged 71+ curated checkpoint dataset read-only.
[Invariants] Strictly runtime-agnostic (zero inference-engine references);
  bounded O(unique patterns) streaming memory via ``ijson`` with stdlib chunked
  fallback; unknown non-empty quant tags raise ``ModelingError`` and never
  degrade to ``FLOAT``; analytical reads perform zero network I/O and zero
  cache writes; provenance never leaks host home directories.
[Timeouts & Locks] Pure local/packaged reads and in-memory parsing; no network
  calls, subprocesses, or locks.
[Side-Effect Policy] Strictly read-only over local files and package data;
  never creates or mutates files on disk.
"""
from __future__ import annotations

import codecs
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.catalog.types import QuantSpec, merge_source
from mindie_sim.schema.checkpoint import (
    CHECKPOINT_QUANT_MAP_SCHEMA,
    CacheQuantRule,
    CheckpointFormat,
    CheckpointOrigin,
    CheckpointProvenance,
    CheckpointQuantMap,
    LayerQuantException,
    ModelScope,
    TensorPresenceDomain,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes

try:
    import ijson as _ijson
except ImportError:  # pragma: no cover - exercised via stdlib fallback
    _ijson = None

CURATED_CHECKPOINTS_SCHEMA = "mindie-sim.curated-checkpoints/1"
CURATED_DATA_REL_DIR = "data"
CURATED_DATA_FILENAME = "curated_checkpoints.json"

DEFAULT_STREAM_CHUNK_BYTES = 65536
MAX_SINGLE_MEMBER_CHARS = 16 * 1024 * 1024
MAX_CONFIG_FILE_BYTES = 16 * 1024 * 1024
QUANT_MAP_CACHE_MAXSIZE = 64

MXFP_GROUP_SIZE = 32
MXFP_SCALE_PACK = 2
MXFP_SCALE_BYTES = 1.0
FP8_DEFAULT_BLOCK_SIZE = 128
AWQ_DEFAULT_GROUP_SIZE = 128
GPTQ_DEFAULT_GROUP_SIZE = 128
AWQ_DEFAULT_BITS = 4
GPTQ_DEFAULT_BITS = 4
INT8_BITS = 8
INT4_BITS = 4
FP32_SCALE_BYTES = 4.0
BF16_SCALE_BYTES = 2.0

_ABS_HOME_PREFIX_RE = re.compile(r"^(?:/(?:Users|home|root)/|[A-Za-z]:\\Users\\)")
_EXPERT_INDEX_RE = re.compile(r"(\.experts\.)(\d+)(\.)")
_LAYER_INDEX_RE = re.compile(
    r"(^|\.)(layers|mtp|blocks|resblocks|proj|deepstack_merger_list)\.(?P<idx>\d+)(?=\.|$)"
)
_SHARD_TOKEN_RE = re.compile(r"(?<=\.)shard_(?:\d+|\*)(?=\.|$)")
_NUMERIC_SEGMENT_RE = re.compile(r"(?<=\.)\d+(?=\.)")

_TOP_LEVEL_META_KEYS = frozenset({
    "version", "model_quant_type", "kv_cache_type", "kv_quant_type",
    "fa_quant_type", "indexer_quant_type", "group_size", "is_rot_used",
    "metadata", "optional", "quant_method", "format", "w_bit", "a_bit",
})
_AUX_PARAM_SUFFIXES = frozenset({
    "bias", "weight_scale", "weight_offset", "scale_bias",
    "input_scale", "input_offset", "deq_scale", "quant_bias", "scale_inv",
})
_KV_AUX_SUFFIXES = frozenset({"kv_cache_scale", "kv_cache_offset"})
_FA_AUX_SUFFIXES = frozenset({
    "fa_q.scale", "fa_q.offset", "fa_k.scale", "fa_k.offset",
    "fa_v.scale", "fa_v.offset",
})

KNOWN_MODELSLIM_WEIGHT_TAGS = frozenset({
    "FLOAT", "BF16", "FP16", "NONE",
    "W8A8", "W8A8_DYNAMIC", "W8A8_MIX", "W8A8_MXFP8", "W8A8_FP8",
    "W8A8_INT8", "W8A8_MXFP",
    "W4A8_DYNAMIC", "W4A8", "W4A8_MXFP", "W4A8_MXFP4", "W4A8_MXFP4_DYNAMIC",
    "W4A4_MXFP4", "W4A4_MXFP", "W4A4_DYNAMIC", "W4A4",
    "FP8_DYNAMIC", "FP8", "MXFP8", "MXFP8_DYNAMIC", "MXFP4",
    "INT8_DYNAMIC", "INT8_SYM", "INT8", "INT4",
    "W4A16_AWQ", "W4A16_GPTQ", "W8A16_GPTQ", "W4A16_COMPRESSED",
    "W4A8_COMPRESSED", "QUARK_FP8", "QUARK_MXFP4", "QUARK_INT8",
})
KNOWN_MODELSLIM_CACHE_TAGS = frozenset({
    "FAQuant", "FAKQuant", "C8", "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
    "INT8_DYNAMIC", "FP8_DYNAMIC", "MXFP8_DYNAMIC", "INT8_SYM",
})
KNOWN_MODELSLIM_TAGS_UPPER = frozenset(
    tag.upper().replace("-", "_")
    for tag in (KNOWN_MODELSLIM_WEIGHT_TAGS | KNOWN_MODELSLIM_CACHE_TAGS)
)
_SUPPORTED_HF_QUANT_METHODS = frozenset({
    "fp8", "mxfp8", "mxfp4", "compressed-tensors", "awq", "gptq", "quark", "ascend",
})

_MX_BASE: dict[str, Any] = {
    "weight_block": (1, MXFP_GROUP_SIZE),
    "expert_block": MXFP_GROUP_SIZE,
    "scale_format": "fp8_e8m0",
    "scale_bytes": MXFP_SCALE_BYTES,
    "activation_group_size": MXFP_GROUP_SIZE,
    "activation_scale_format": "fp8_e8m0",
    "activation_scale_pack": MXFP_SCALE_PACK,
    "is_quantized": True,
}
_INT_BASE: dict[str, Any] = {
    "weight_block": (1, 1),
    "expert_block": 1,
    "scale_format": "fp32",
    "scale_bytes": FP32_SCALE_BYTES,
    "activation_group_size": 0,
    "activation_scale_format": "fp32",
    "activation_scale_pack": 1,
    "is_quantized": True,
}
_INT8_RT: dict[str, Any] = {
    "runtime_weight_scale_format": "bf16",
    "runtime_weight_scale_bytes": BF16_SCALE_BYTES,
    "runtime_activation_scale_format": "fp32",
    "runtime_int8_weight_metadata": "scale_only",
}
_INT4_RT: dict[str, Any] = {
    "runtime_weight_scale_format": "fp32",
    "runtime_weight_scale_bytes": FP32_SCALE_BYTES,
    "runtime_activation_scale_format": "fp32",
    "runtime_int8_weight_metadata": "scale_offset",
}
TAG_SPEC_TABLE: dict[str, dict[str, Any]] = {
    "W8A8_MXFP8": {
        "weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3",
        "ledger_format": "MXFP8", **_MX_BASE,
    },
    "W4A8_MXFP": {
        "weight_format": "mxfp4", "activation_format": "fp8_e4m3",
        "ledger_format": "MXFP4", **_MX_BASE,
    },
    "W4A4_MXFP4": {
        "weight_format": "mxfp4", "activation_format": "fp4_e2m1",
        "ledger_format": "MXFP4", **_MX_BASE,
    },
    "W8A8_DYNAMIC": {
        "weight_format": "int8", "activation_format": "int8",
        "ledger_format": "W8A8", **_INT_BASE, **_INT8_RT,
    },
    "W8A8": {
        "weight_format": "int8", "activation_format": "int8",
        "ledger_format": "W8A8", **_INT_BASE, **_INT8_RT,
    },
    "W8A8_MIX": {
        "weight_format": "int8", "activation_format": "int8",
        "ledger_format": "W8A8", **_INT_BASE, **_INT8_RT,
    },
    "W4A8_DYNAMIC": {
        "weight_format": "int4", "activation_format": "int8",
        "ledger_format": "W4A8", **_INT_BASE, **_INT4_RT,
    },
    "W4A4_DYNAMIC": {
        "weight_format": "int4", "activation_format": "int4",
        "ledger_format": "W4A4", **_INT_BASE, **_INT4_RT,
    },
    "FP8_DYNAMIC": {
        "weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3",
        "ledger_format": "FP8",
        "weight_block": (FP8_DEFAULT_BLOCK_SIZE, FP8_DEFAULT_BLOCK_SIZE),
        "expert_block": MXFP_GROUP_SIZE,
        "scale_format": "fp32", "scale_bytes": FP32_SCALE_BYTES,
        "activation_group_size": FP8_DEFAULT_BLOCK_SIZE,
        "activation_scale_format": "fp32", "activation_scale_pack": 1,
        "is_quantized": True,
    },
    "FLOAT": {
        "weight_format": "bf16", "activation_format": "bf16",
        "ledger_format": "BF16", "weight_block": (0, 0), "expert_block": 0,
        "scale_format": "none", "scale_bytes": 0.0,
        "activation_group_size": 0, "activation_scale_format": "",
        "activation_scale_pack": 1, "is_quantized": False,
    },
}
TAG_ALIASES: dict[str, str] = {
    "W8A8_MXFP8": "W8A8_MXFP8", "MXFP8": "W8A8_MXFP8",
    "W8A8_MXFP": "W8A8_MXFP8", "MXFP8_DYNAMIC": "W8A8_MXFP8",
    "W4A8_MXFP": "W4A8_MXFP", "W4A8_MXFP4": "W4A8_MXFP",
    "MXFP4": "W4A8_MXFP", "W4A8_MXFP4_DYNAMIC": "W4A8_MXFP",
    "W4A4_MXFP4": "W4A4_MXFP4", "W4A4_MXFP": "W4A4_MXFP4",
    "MXFP4_W4A4": "W4A4_MXFP4",
    "W8A8_DYNAMIC": "W8A8_DYNAMIC", "W8A8_INT8": "W8A8_DYNAMIC",
    "W8A8": "W8A8", "W8A8_MIX": "W8A8_MIX",
    "INT8": "W8A8", "INT8_DYNAMIC": "W8A8_DYNAMIC", "INT8_SYM": "W8A8_DYNAMIC",
    "W4A8_DYNAMIC": "W4A8_DYNAMIC", "W4A8": "W4A8_DYNAMIC", "INT4": "W4A8_DYNAMIC",
    "W4A4_DYNAMIC": "W4A4_DYNAMIC", "W4A4": "W4A4_DYNAMIC",
    "FP8_DYNAMIC": "FP8_DYNAMIC", "FP8": "FP8_DYNAMIC", "W8A8_FP8": "FP8_DYNAMIC",
    "FLOAT": "FLOAT", "BF16": "FLOAT", "FP16": "FLOAT", "NONE": "FLOAT",
}
STANDARD_UNQUANTIZED_EXCLUSIONS: tuple[str, ...] = (
    "attn_res_proj", "router", "lm_head", "embedding", "norm", "mhc",
    "vision", "projector", "conv_weight_f32", "conv_weight_bf16",
)
_VISION_PREFIXES = (
    "vision_tower.", "vision_model.", "vision.", "model.visual.",
    "model.vision_tower.", "model.vision_model.", "visual.",
)
_PROJECTOR_PREFIXES = (
    "mm_projector.", "multi_modal_projector.", "aligner.",
    "patch_merge_mlp.", "vit_large_projector.",
)
_KDA_EXACT_COMPONENTS = frozenset({
    "f_a_proj", "f_b_proj", "b_proj", "g_a_proj", "g_b_proj",
    "q_conv1d", "k_conv1d", "v_conv1d",
})
_MLA_EXACT_COMPONENTS = frozenset({
    "kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj",
})
_NORM_EXACT_COMPONENTS = frozenset({
    "input_layernorm", "post_attention_layernorm", "q_a_layernorm",
    "kv_a_layernorm", "q_norm", "k_norm", "o_norm", "norm", "norm0",
    "norm1", "norm2", "final_layernorm", "q_layernorm", "k_layernorm",
    "self_attention_res_norm", "mlp_res_norm", "output_attn_res_norm",
    "routed_expert_norm", "hc_norm", "post_norm", "norm_conv",
    "norm_key", "norm_query", "enorm", "hnorm", "attn_norm", "ffn_norm",
    "pre_fc_norm_embedding", "pre_fc_norm_hidden",
})
_ATTN_RES_PROJ_COMPONENTS = frozenset({
    "self_attention_res_proj", "mlp_res_proj", "output_attn_res_proj",
})


class _HashingBinaryReader:
    """Pass-through binary stream updating a SHA-256 digest on each read."""

    def __init__(self, raw_fp: Any, hasher: Any) -> None:
        self._fp = raw_fp
        self._hasher = hasher

    def read(self, size: int = -1) -> bytes:
        data = self._fp.read(size)
        if data:
            self._hasher.update(data)
        return data


def _sanitize_checkpoint_id(raw_id: str) -> str:
    text = str(raw_id or "").strip()
    if not text:
        return "local://checkpoint"
    if _ABS_HOME_PREFIX_RE.match(text):
        path = Path(text)
        label = path.parent.name if path.is_file() and path.parent.name else path.name
        return f"local://{label or 'checkpoint'}"
    return text


def _sanitize_descriptor_path(path: Path | str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if _ABS_HOME_PREFIX_RE.match(raw) or Path(raw).is_absolute():
        return Path(raw).name
    return raw


def _validate_known_tag(raw_tag: str) -> str:
    tag = str(raw_tag or "").strip()
    if not tag:
        return ""
    upper = tag.upper().replace("-", "_")
    if upper not in KNOWN_MODELSLIM_TAGS_UPPER:
        raise ModelingError(f"unknown_checkpoint_quant_tag:{tag}")
    return tag


def normalize_quant_tag_strict(raw_tag: str) -> str:
    """Validate and canonicalize a quant tag; unknown non-empty tags raise."""
    cleaned = str(raw_tag or "").strip()
    if not cleaned:
        return "FLOAT"
    upper = cleaned.upper().replace("-", "_")
    canonical = TAG_ALIASES.get(upper)
    if canonical is not None:
        return canonical
    if upper in TAG_SPEC_TABLE:
        return upper
    raise ModelingError(f"unknown_checkpoint_quant_tag:{cleaned}")


def _extract_native_float_dtype(config: Mapping[str, Any] | None) -> str:
    if not isinstance(config, Mapping):
        return ""
    text_cfg = config.get("text_config")
    cfg = text_cfg if isinstance(text_cfg, Mapping) else config
    raw = cfg.get("torch_dtype") or cfg.get("dtype") or config.get("torch_dtype") or config.get("dtype")
    if raw is None or not str(raw).strip():
        return ""
    lower = str(raw).strip().lower().removeprefix("torch.")
    if lower in {"bfloat16", "bf16"}:
        return "bf16"
    if lower in {"float16", "fp16", "half"}:
        return "fp16"
    if lower in {"float32", "fp32", "float"}:
        return "fp32"
    return lower


def _infer_scope(tensor_key: str) -> ModelScope:
    lower = tensor_key.lower()
    if lower.startswith("mtp.") or ".mtp." in lower or ".mtp_block." in lower:
        return "draft"
    if any(lower.startswith(pref) for pref in _VISION_PREFIXES):
        return "vision"
    if (
        any(lower.startswith(pref) for pref in _PROJECTOR_PREFIXES)
        or ".merger." in lower
        or ".deepstack_merger_list." in lower
    ):
        return "projector"
    return "target"


def _split_tensor_suffix(raw_key: str) -> tuple[str, str, bool]:
    """Return ``(module_key, aux_suffix, had_weight_suffix)``."""
    for fa_suf in _FA_AUX_SUFFIXES:
        marker = "." + fa_suf
        if raw_key.endswith(marker):
            return raw_key[: -len(marker)], fa_suf, False
    head, sep, tail = raw_key.rpartition(".")
    if sep:
        if tail == "weight":
            return head, "weight", True
        if tail in _AUX_PARAM_SUFFIXES or tail in _KV_AUX_SUFFIXES:
            return head, tail, False
    return raw_key, "", False


def _extract_pattern_and_coords(module_key: str) -> tuple[str, int | None, int | None]:
    """Normalize a tensor module path and extract ``(layer_id, expert_id)``."""
    expert_id: int | None = None
    em = _EXPERT_INDEX_RE.search(module_key)
    cur = module_key
    if em is not None:
        expert_id = int(em.group(2))
        cur = _EXPERT_INDEX_RE.sub(r"\1*\3", cur, count=1)

    layer_id: int | None = None
    lm = _LAYER_INDEX_RE.search(cur)
    if lm is not None:
        layer_id = int(lm.group("idx"))
        cur = cur[: lm.start("idx")] + "*" + cur[lm.end("idx") :]

    cur = _SHARD_TOKEN_RE.sub("*", cur)
    cur = _NUMERIC_SEGMENT_RE.sub("*", cur)
    return cur, layer_id, expert_id


def encode_int_ranges(values: Sequence[int]) -> list[int | list[int]]:
    """Losslessly compress sorted non-negative integers into runs ``[lo, hi]``."""
    ordered = sorted(set(int(v) for v in values))
    if not ordered:
        return []
    runs: list[int | list[int]] = []
    start = prev = ordered[0]
    for cur in ordered[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        runs.append([start, prev] if prev > start else start)
        start = prev = cur
    runs.append([start, prev] if prev > start else start)
    return runs


def decode_int_ranges(encoded: Sequence[Any]) -> tuple[int, ...]:
    """Expand range-encoded integers back into a sorted tuple."""
    out: list[int] = []
    for item in encoded:
        if isinstance(item, list) and len(item) == 2:
            out.extend(range(int(item[0]), int(item[1]) + 1))
        else:
            out.append(int(item))
    return tuple(out)


class _ChunkedJsonScanner:
    """Bounded-memory incremental scanner over a top-level JSON object."""

    def __init__(self, raw_fp: Any, hasher: Any, chunk_size: int) -> None:
        self._fp = raw_fp
        self._hasher = hasher
        self._chunk_size = max(1, int(chunk_size))
        self._decoder = json.JSONDecoder()
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        self._buf = ""
        self._pos = 0
        self._eof = False
        self._first_chunk = True

    def _fill(self) -> bool:
        if self._eof:
            return False
        raw = self._fp.read(self._chunk_size)
        if not raw:
            tail = self._utf8.decode(b"", final=True)
            if tail:
                self._buf += tail
            self._eof = True
            return bool(tail)
        self._hasher.update(raw)
        text = self._utf8.decode(raw, final=False)
        if self._first_chunk:
            self._first_chunk = False
            text = text.removeprefix("\ufeff")
        self._buf += text
        return True

    def _skip_ws(self) -> None:
        while True:
            while self._pos < len(self._buf) and self._buf[self._pos] in " \t\r\n":
                self._pos += 1
            if self._pos < len(self._buf) or not self._fill():
                return

    def _decode_next(self) -> Any:
        while True:
            self._skip_ws()
            if self._pos >= len(self._buf):
                raise ModelingError("checkpoint_json_unexpected_eof")
            lead = self._buf[self._pos]
            try:
                val, end = self._decoder.raw_decode(self._buf, self._pos)
            except json.JSONDecodeError as exc:
                if self._eof:
                    raise ModelingError(f"checkpoint_json_malformed:{exc.msg}") from exc
                if len(self._buf) - self._pos > MAX_SINGLE_MEMBER_CHARS:
                    raise ModelingError("checkpoint_json_member_too_large") from exc
                self._fill()
                continue
            if end == len(self._buf) and not self._eof and lead not in {'"', "{", "["}:
                self._fill()
                continue
            self._pos = end
            return val

    def iter_items(self) -> Iterator[tuple[str, Any]]:
        self._skip_ws()
        if self._pos >= len(self._buf) or self._buf[self._pos] != "{":
            raise ModelingError("checkpoint_json_requires_object")
        self._pos += 1
        self._skip_ws()
        if self._pos < len(self._buf) and self._buf[self._pos] == "}":
            return
        while True:
            key = self._decode_next()
            if not isinstance(key, str):
                raise ModelingError("checkpoint_json_key_not_string")
            self._skip_ws()
            if self._pos >= len(self._buf) or self._buf[self._pos] != ":":
                raise ModelingError("checkpoint_json_expected_colon")
            self._pos += 1
            value = self._decode_next()
            yield key, value
            if self._pos >= self._chunk_size:
                self._buf = self._buf[self._pos :]
                self._pos = 0
            self._skip_ws()
            if self._pos >= len(self._buf):
                raise ModelingError("checkpoint_json_unexpected_eof")
            sep = self._buf[self._pos]
            self._pos += 1
            if sep == "}":
                return
            if sep != ",":
                raise ModelingError(f"checkpoint_json_expected_comma_or_end:{sep!r}")


def _iter_json_object_items_stdlib(
    raw_fp: Any,
    hasher: Any,
    *,
    chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
) -> Iterator[tuple[str, Any]]:
    """Stream top-level JSON object ``(key, value)`` pairs with bounded memory."""
    return _ChunkedJsonScanner(raw_fp, hasher, chunk_size).iter_items()


def iter_json_object_items(
    path: Path | str,
    *,
    chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
    force_stdlib: bool = False,
) -> tuple[Iterator[tuple[str, Any]], Any]:
    """Stream top-level JSON object members from disk with live SHA-256 hashing.

    Uses ``ijson`` when installed unless ``force_stdlib=True``; otherwise uses
    the bounded stdlib chunked decoder.  Caller must exhaust or close the
    returned iterator.
    """
    target = Path(path)
    if not target.is_file():
        raise ModelingError(f"checkpoint_file_not_found:{target.name}")
    hasher = hashlib.sha256()

    def _gen() -> Iterator[tuple[str, Any]]:
        try:
            with target.open("rb") as raw_fp:
                if _ijson is not None and not force_stdlib:
                    wrapped = _HashingBinaryReader(raw_fp, hasher)
                    try:
                        yield from _ijson.kvitems(wrapped, "")
                    except Exception as exc:
                        raise ModelingError(
                            f"checkpoint_json_stream_error:{target.name}:{type(exc).__name__}"
                        ) from exc
                    while wrapped.read(chunk_size):
                        pass
                else:
                    yield from _iter_json_object_items_stdlib(
                        raw_fp, hasher, chunk_size=chunk_size,
                    )
        except TimeoutError:
            raise
        except OSError as exc:
            raise ModelingError(
                f"checkpoint_file_unreadable:{target.name}:{type(exc).__name__}"
            ) from exc

    return _gen(), hasher


def _infer_cache_scale_dtypes(
    kv_tag: str,
    index_tag: str,
    aux_buffers: set[str],
) -> tuple[str, str, str]:
    kv_upper = kv_tag.upper()
    idx_upper = index_tag.upper()
    kv_scale = ""
    if "MXFP8" in kv_upper:
        kv_scale = "fp8_e8m0"
    elif kv_upper in {"C8", "FAQUANT", "FAKQUANT", "INT8_DYNAMIC", "FP8_DYNAMIC"}:
        kv_scale = "fp32"
    kv_offset = ""
    if any(b in aux_buffers for b in ("kv_cache_offset", "fa_k.offset", "fa_v.offset", "fa_q.offset")):
        kv_offset = "int8" if kv_upper in {"C8", "INT8_DYNAMIC"} else "fp32"
    idx_scale = ""
    if "MXFP8" in idx_upper:
        idx_scale = "fp8_e8m0"
    elif "FP8" in idx_upper:
        idx_scale = "fp32"
    elif "INT8" in idx_upper:
        idx_scale = "fp16"
    return kv_scale, idx_scale, kv_offset


def _select_dominant_kv_tag(
    layer_kv_tags: Mapping[int, str],
    top_kv: str,
    top_fa: str,
) -> str:
    if top_kv or top_fa:
        if top_kv == "FAQuant" and top_fa:
            return top_fa
        return top_kv or top_fa
    if layer_kv_tags:
        counts: dict[str, int] = defaultdict(int)
        for tag in layer_kv_tags.values():
            counts[tag] += 1
        dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if dominant == "FAQuant" and top_fa:
            return top_fa
        return dominant
    return ""


def _compile_pattern_exceptions(
    scope: ModelScope,
    pattern: str,
    tag_map: Mapping[str, set[tuple[int | None, int | None]]],
    exc_tag: str,
    coords: set[tuple[int | None, int | None]],
) -> list[LayerQuantException]:
    by_layer_exps: dict[int, set[int | None]] = defaultdict(set)
    for lid, eid in coords:
        if lid is not None:
            by_layer_exps[lid].add(eid)
    all_layer_exps = {
        lid: {eid for t_coords in tag_map.values() for l2, eid in t_coords if l2 == lid}
        for lid in by_layer_exps
    }
    full_layers = sorted(
        lid for lid, eids in by_layer_exps.items() if eids == all_layer_exps[lid]
    )
    out: list[LayerQuantException] = []
    if full_layers:
        out.append(
            LayerQuantException(
                pattern=pattern,
                scope=scope,
                layer_indices=tuple(full_layers),
                raw_tag=exc_tag,
            )
        )
    full_set = set(full_layers)
    for lid in sorted(by_layer_exps):
        if lid in full_set:
            continue
        spec_exps = tuple(sorted(e for e in by_layer_exps[lid] if e is not None))
        out.append(
            LayerQuantException(
                pattern=pattern,
                scope=scope,
                layer_indices=(lid,),
                raw_tag=exc_tag,
                expert_indices=spec_exps,
            )
        )
    return out


def _compile_rules_and_exceptions(
    pat_coords: Mapping[tuple[ModelScope, str], Mapping[str, set[tuple[int | None, int | None]]]],
) -> tuple[tuple[TensorQuantRule, ...], tuple[LayerQuantException, ...]]:
    rules: list[TensorQuantRule] = []
    exceptions: list[LayerQuantException] = []
    for (scope, pattern), tag_map in sorted(pat_coords.items()):
        experts_by_layer: dict[int, set[int]] = defaultdict(set)
        has_any_expert = False
        for coords in tag_map.values():
            for lid, eid in coords:
                if eid is not None:
                    has_any_expert = True
                    if lid is not None:
                        experts_by_layer[lid].add(eid)
        distinct_exp_sets = {frozenset(eids) for eids in experts_by_layer.values()} if has_any_expert else set()
        if len(distinct_exp_sets) > 1:
            overall_top = sorted(
                tag_map.items(),
                key=lambda kv: (-len(kv[1]), normalize_quant_tag_strict(kv[0]) == "FLOAT", kv[0]),
            )[0][0]
            layers_by_eset: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for lid, eids in sorted(experts_by_layer.items()):
                layers_by_eset[tuple(sorted(eids))].append(lid)
            for eset, group_layers in sorted(layers_by_eset.items(), key=lambda kv: kv[1][0]):
                layer_set = set(group_layers)
                sub_tag_map = {
                    t: {(l, e) for l, e in coords if l in layer_set}
                    for t, coords in tag_map.items()
                }
                sub_tag_map = {t: c for t, c in sub_tag_map.items() if c}
                ranked = sorted(
                    sub_tag_map.items(),
                    key=lambda kv: (
                        -len(kv[1]),
                        kv[0] != overall_top,
                        normalize_quant_tag_strict(kv[0]) == "FLOAT",
                        kv[0],
                    ),
                )
                presence = TensorPresenceDomain(
                    scope=scope, layer_indices=tuple(group_layers), expert_indices=eset,
                )
                rules.append(TensorQuantRule(pattern=pattern, raw_tag=ranked[0][0], presence=presence))
                for exc_tag, coords in ranked[1:]:
                    exceptions.extend(
                        _compile_pattern_exceptions(scope, pattern, sub_tag_map, exc_tag, coords)
                    )
            continue

        all_layers = sorted({lid for coords in tag_map.values() for lid, _ in coords if lid is not None})
        all_experts = sorted({eid for coords in tag_map.values() for _, eid in coords if eid is not None})
        presence = TensorPresenceDomain(
            scope=scope,
            layer_indices=tuple(all_layers),
            expert_indices=tuple(all_experts),
        )
        ranked = sorted(
            tag_map.items(),
            key=lambda kv: (-len(kv[1]), normalize_quant_tag_strict(kv[0]) == "FLOAT", kv[0]),
        )
        rules.append(TensorQuantRule(pattern=pattern, raw_tag=ranked[0][0], presence=presence))
        for exc_tag, coords in ranked[1:]:
            exceptions.extend(
                _compile_pattern_exceptions(scope, pattern, tag_map, exc_tag, coords)
            )
    return tuple(rules), tuple(exceptions)


def _build_cache_quant_rule(
    layer_kv_explicit: Mapping[int, str],
    layer_kv_aux: Mapping[int, str],
    layer_idx_explicit: Mapping[int, str],
    top_kv: str,
    top_fa: str,
    top_idx: str,
    aux_buffers: set[str],
    *,
    draft_kv_explicit: Mapping[int, str] | None = None,
    draft_kv_aux: Mapping[int, str] | None = None,
    draft_idx_explicit: Mapping[int, str] | None = None,
) -> CacheQuantRule:
    merged_kv_layers = {**layer_kv_aux, **layer_kv_explicit}
    merged_draft_kv = {**(draft_kv_aux or {}), **(draft_kv_explicit or {})}
    merged_draft_idx = dict(draft_idx_explicit or {})
    kv_tag = _select_dominant_kv_tag(merged_kv_layers, top_kv, top_fa)
    index_tag = (
        top_idx
        if top_idx
        else (
            sorted(
                ((sum(1 for v in layer_idx_explicit.values() if v == t), t) for t in set(layer_idx_explicit.values())),
                key=lambda kv: (-kv[0], kv[1]),
            )[0][1]
            if layer_idx_explicit
            else ""
        )
    )
    kv_scale, idx_scale, kv_offset = _infer_cache_scale_dtypes(kv_tag, index_tag, aux_buffers)
    has_cache = bool(
        kv_tag or index_tag or merged_kv_layers or layer_idx_explicit
        or merged_draft_kv or merged_draft_idx
    )
    return CacheQuantRule(
        kv_tag=kv_tag,
        index_tag=index_tag,
        kv_scale_dtype=kv_scale,
        index_scale_dtype=idx_scale,
        kv_offset_dtype=kv_offset,
        layer_scoped_kv=tuple(sorted(merged_kv_layers.items())),
        layer_scoped_index=tuple(sorted(layer_idx_explicit.items())),
        provenance="checkpoint" if has_cache else "native",
        draft_layer_scoped_kv=tuple(sorted(merged_draft_kv.items())),
        draft_layer_scoped_index=tuple(sorted(merged_draft_idx.items())),
    )


def _reduce_modelslim_stream(
    items: Iterator[tuple[str, Any]],
    *,
    checkpoint_id: str,
    descriptor_path: str = "",
    source_revision: str = "",
    native_float_dtype: str = "",
    hasher: Any | None = None,
) -> tuple[CheckpointQuantMap, frozenset[str]]:
    """Stream-reduce ModelSlim key/value pairs into ``(CheckpointQuantMap, non_weight_patterns)``."""
    meta_tags = {"model_quant_type": "", "kv": "", "fa": "", "idx": ""}
    aux_buffers: set[str] = set()
    layer_kv_explicit: dict[int, str] = {}
    layer_kv_aux: dict[int, str] = {}
    layer_idx_explicit: dict[int, str] = {}
    draft_kv_explicit: dict[int, str] = {}
    draft_kv_aux: dict[int, str] = {}
    draft_idx_explicit: dict[int, str] = {}
    pat_coords: dict[tuple[ModelScope, str], dict[str, set[tuple[int | None, int | None]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    weight_bearing: set[tuple[ModelScope, str]] = set()
    non_weight_only: set[tuple[ModelScope, str]] = set()

    for raw_k, raw_v in items:
        key = str(raw_k)
        if not isinstance(raw_v, str) or not raw_v.strip():
            continue
        if key in _TOP_LEVEL_META_KEYS:
            val = raw_v.strip()
            if key == "model_quant_type":
                meta_tags["model_quant_type"] = _validate_known_tag(val)
            elif key in {"kv_cache_type", "kv_quant_type"}:
                meta_tags["kv"] = _validate_known_tag(val)
            elif key == "fa_quant_type":
                meta_tags["fa"] = _validate_known_tag(val)
            elif key == "indexer_quant_type":
                meta_tags["idx"] = _validate_known_tag(val)
            continue

        tag = _validate_known_tag(raw_v.strip())
        module_key, suffix, had_weight = _split_tensor_suffix(key)
        scope = _infer_scope(key)
        if suffix and (
            scope == "target"
            or (suffix not in _KV_AUX_SUFFIXES and suffix not in _FA_AUX_SUFFIXES)
        ):
            aux_buffers.add(suffix)
        norm_pat, layer_id, expert_id = _extract_pattern_and_coords(module_key)

        if suffix in _KV_AUX_SUFFIXES or suffix in _FA_AUX_SUFFIXES:
            if scope == "draft":
                if layer_id is not None:
                    draft_kv_aux[layer_id] = tag
                continue
            if scope != "target":
                continue
            if layer_id is not None:
                layer_kv_aux[layer_id] = tag
            else:
                meta_tags["kv"] = meta_tags["kv"] or tag
            continue
        if key.endswith(".indexer.quant_type"):
            if scope == "draft":
                if layer_id is not None:
                    draft_idx_explicit[layer_id] = tag
                continue
            if scope != "target":
                continue
            if layer_id is not None:
                layer_idx_explicit[layer_id] = tag
            else:
                meta_tags["idx"] = meta_tags["idx"] or tag
            continue
        if key.endswith(".self_attn.quant_type") or key.endswith(".attn.quant_type"):
            if scope == "draft":
                if layer_id is not None:
                    draft_kv_explicit[layer_id] = tag
                continue
            if scope != "target":
                continue
            if layer_id is not None:
                layer_kv_explicit[layer_id] = tag
            else:
                meta_tags["kv"] = meta_tags["kv"] or tag
            continue
        if suffix in _AUX_PARAM_SUFFIXES:
            continue

        pat_key = (scope, norm_pat)
        pat_coords[pat_key][tag].add((layer_id, expert_id))
        (weight_bearing if had_weight else non_weight_only).add(pat_key)

    rules, exceptions = _compile_rules_and_exceptions(pat_coords)
    cache_rule = _build_cache_quant_rule(
        layer_kv_explicit, layer_kv_aux, layer_idx_explicit,
        meta_tags["kv"], meta_tags["fa"], meta_tags["idx"], aux_buffers,
        draft_kv_explicit=draft_kv_explicit,
        draft_kv_aux=draft_kv_aux,
        draft_idx_explicit=draft_idx_explicit,
    )
    provenance = CheckpointProvenance(
        checkpoint_id=_sanitize_checkpoint_id(checkpoint_id),
        source_format="modelslim_description",
        source_revision=source_revision,
        source_sha256=hasher.hexdigest() if hasher is not None else "",
        descriptor_path=_sanitize_descriptor_path(descriptor_path),
        origin="verified_checkpoint_descriptor",
    )
    qmap = CheckpointQuantMap(
        provenance=provenance,
        schema_version=CHECKPOINT_QUANT_MAP_SCHEMA,
        native_float_dtype=native_float_dtype,
        model_quant_type=meta_tags["model_quant_type"],
        rules=rules,
        layer_exceptions=exceptions,
        auxiliary_buffers=tuple(sorted(aux_buffers)),
        cache_rule=cache_rule,
    )
    non_weight_pats = frozenset(pat for _, pat in (non_weight_only - weight_bearing))
    return qmap, non_weight_pats


def stream_modelslim_description_file(
    path: Path | str,
    *,
    checkpoint_id: str = "",
    base_config: Mapping[str, Any] | None = None,
    source_revision: str = "",
    chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
    force_stdlib: bool = False,
) -> tuple[CheckpointQuantMap, frozenset[str]]:
    """Stream-parse an on-disk ModelSlim JSON file and return ``(qmap, non_weight_patterns)``."""
    target = Path(path)
    items, hasher = iter_json_object_items(
        target, chunk_size=chunk_size, force_stdlib=force_stdlib,
    )
    cid = checkpoint_id or f"local://{target.parent.name or target.stem}"
    return _reduce_modelslim_stream(
        items,
        checkpoint_id=cid,
        descriptor_path=target.name,
        source_revision=source_revision,
        native_float_dtype=_extract_native_float_dtype(base_config),
        hasher=hasher,
    )


def read_modelslim_description_file(
    path: Path | str,
    *,
    checkpoint_id: str = "",
    base_config: Mapping[str, Any] | None = None,
    source_revision: str = "",
    chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
    force_stdlib: bool = False,
) -> CheckpointQuantMap:
    """Stream-parse an on-disk ModelSlim ``quant_model_description*.json`` file."""
    qmap, _ = stream_modelslim_description_file(
        path,
        checkpoint_id=checkpoint_id,
        base_config=base_config,
        source_revision=source_revision,
        chunk_size=chunk_size,
        force_stdlib=force_stdlib,
    )
    return qmap


def read_modelslim_description_mapping(
    desc_mapping: Mapping[str, Any],
    *,
    checkpoint_id: str = "inline://modelslim",
    base_config: Mapping[str, Any] | None = None,
    source_revision: str = "",
) -> CheckpointQuantMap:
    """Parse an in-memory ModelSlim description mapping into ``CheckpointQuantMap``."""
    if not isinstance(desc_mapping, Mapping):
        raise ModelingError("quant_model_description_requires_mapping")
    expanded: list[tuple[str, Any]] = []
    for k, v in desc_mapping.items():
        key = str(k)
        upper_key = key.upper().replace("-", "_")
        if isinstance(v, (list, tuple)) and upper_key in KNOWN_MODELSLIM_TAGS_UPPER:
            for pattern in v:
                expanded.append((str(pattern), key))
        else:
            expanded.append((key, v))
    qmap, _ = _reduce_modelslim_stream(
        iter(expanded),
        checkpoint_id=checkpoint_id,
        source_revision=source_revision,
        native_float_dtype=_extract_native_float_dtype(base_config),
    )
    return qmap


def _hf_ignored_modules(qcfg: Mapping[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for field in ("ignored_layers", "modules_to_not_convert", "ignore", "exclude"):
        vals = qcfg.get(field)
        if isinstance(vals, (list, tuple)):
            for item in vals:
                text = str(item).strip()
                if text:
                    out.append(text)
    return tuple(dict.fromkeys(out))


def _hf_block_tuple(raw_block: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(raw_block, (list, tuple)) and len(raw_block) == 2:
        b0, b1 = int(raw_block[0]), int(raw_block[1])
        if b0 >= 1 and b1 >= 1:
            return (b0, b1)
    return default


def _hf_scale_dtype(raw_scale: Any, default: str) -> str:
    text = str(raw_scale or "").strip().lower()
    if text.startswith("torch."):
        text = text[len("torch."):]
    if not text:
        return default
    if "e8m0" in text:
        return "fp8_e8m0"
    if text in {"float32", "fp32"}:
        return "fp32"
    if text in {"bfloat16", "bf16"}:
        return "bf16"
    if text in {"float16", "fp16"}:
        return "fp16"
    return text


def _build_hf_mxfp8_rules(
    qcfg: Mapping[str, Any],
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    block = _hf_block_tuple(qcfg.get("weight_block_size"), (1, MXFP_GROUP_SIZE))
    scale = _hf_scale_dtype(qcfg.get("scale_fmt"), "fp8_e8m0")
    rules = (
        TensorQuantRule(
            pattern="*",
            raw_tag="W8A8_MXFP8",
            weight_dtype="fp8_e4m3",
            activation_dtype="fp8_e4m3",
            scale_dtype=scale,
            group_size=block,
            symmetric=True,
        ),
    )
    return "MXFP8", rules, CacheQuantRule()


def _build_hf_mxfp4_rules(
    qcfg: Mapping[str, Any],
    native: str,
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    block = _hf_block_tuple(qcfg.get("weight_block_size"), (1, MXFP_GROUP_SIZE))
    scale = _hf_scale_dtype(qcfg.get("scale_fmt"), "fp8_e8m0")
    act_raw = str(qcfg.get("activation_dtype") or qcfg.get("act_dtype") or "").strip().lower()
    act_dtype = "fp8_e4m3" if "fp8" in act_raw else ("fp4_e2m1" if "fp4" in act_raw else (native or "bf16"))
    rules = (
        TensorQuantRule(
            pattern="*",
            raw_tag="W4A8_MXFP",
            weight_dtype="mxfp4",
            activation_dtype=act_dtype,
            scale_dtype=scale,
            group_size=block,
            symmetric=True,
        ),
    )
    return "MXFP4", rules, CacheQuantRule()


_HF_EXPERT_DTYPE_ALIASES: dict[str, str] = {
    "fp4": "mxfp4",
    "mxfp4": "mxfp4",
    "fp4_e2m1": "mxfp4",
    "fp8": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
    "e4m3fn": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "float8_e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "e5m2": "fp8_e5m2",
    "fp8_e5m2": "fp8_e5m2",
    "float8_e5m2": "fp8_e5m2",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
}


def _normalize_hf_dtype_token(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("-", "_")
    if text.startswith("torch."):
        text = text[len("torch."):]
    return text


def _build_hf_fp8_rules(
    qcfg: Mapping[str, Any],
    base_config: Mapping[str, Any] | None,
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    fmt = _normalize_hf_dtype_token(qcfg.get("fmt") or "e4m3")
    if fmt in {"e4m3", "e4m3fn", "fp8_e4m3", "float8_e4m3", "float8_e4m3fn"}:
        weight_dtype = "fp8_e4m3"
    elif fmt in {"e5m2", "fp8_e5m2", "float8_e5m2"}:
        weight_dtype = "fp8_e5m2"
    else:
        raise ModelingError(f"unknown_checkpoint_quant_tag:fp8:{fmt}")
    block = _hf_block_tuple(
        qcfg.get("weight_block_size"), (FP8_DEFAULT_BLOCK_SIZE, FP8_DEFAULT_BLOCK_SIZE),
    )
    scale = _hf_scale_dtype(qcfg.get("scale_fmt"), "fp32")
    expert_raw = _normalize_hf_dtype_token(
        qcfg.get("expert_dtype")
        or (base_config.get("expert_dtype") if isinstance(base_config, Mapping) else "")
        or ""
    )
    rules: list[TensorQuantRule] = []
    model_qtype = "FP8_DYNAMIC"
    if expert_raw:
        expert_canon = _HF_EXPERT_DTYPE_ALIASES.get(expert_raw)
        if expert_canon is None:
            raise ModelingError(f"unknown_checkpoint_quant_tag:expert_dtype:{expert_raw}")
        if expert_canon == "mxfp4":
            model_qtype = "FP8_DYNAMIC+MXFP4_EXPERTS"
            rules.append(
                TensorQuantRule(
                    pattern="*.experts.*",
                    raw_tag="W4A8_MXFP",
                    weight_dtype="mxfp4",
                    activation_dtype=weight_dtype,
                    scale_dtype="fp8_e8m0",
                    group_size=(1, MXFP_GROUP_SIZE),
                    symmetric=True,
                )
            )
        elif expert_canon in {"bf16", "fp16", "fp32"}:
            model_qtype = f"FP8_DYNAMIC+{expert_canon.upper()}_EXPERTS"
            rules.append(
                TensorQuantRule(
                    pattern="*.experts.*",
                    raw_tag=expert_canon.upper(),
                    weight_dtype=expert_canon,
                    activation_dtype=expert_canon,
                    scale_dtype="none",
                )
            )
        elif expert_canon != weight_dtype:
            model_qtype = f"FP8_DYNAMIC+{expert_canon.upper()}_EXPERTS"
            rules.append(
                TensorQuantRule(
                    pattern="*.experts.*",
                    raw_tag=expert_canon.upper(),
                    weight_dtype=expert_canon,
                    activation_dtype=expert_canon,
                    scale_dtype=scale,
                    group_size=block,
                    symmetric=True,
                )
            )
    rules.append(
        TensorQuantRule(
            pattern="*",
            raw_tag="FP8_DYNAMIC",
            weight_dtype=weight_dtype,
            activation_dtype=weight_dtype,
            scale_dtype=scale,
            group_size=block,
            symmetric=True,
        )
    )
    return model_qtype, tuple(rules), CacheQuantRule()


def _normalize_compressed_tensors_targets(raw_targets: Any) -> tuple[str, ...]:
    if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
        return ("*",)
    out: list[str] = []
    for item in raw_targets:
        text = str(item or "").strip()
        if not text:
            continue
        if text.lower() in {"linear", "*"}:
            pat = "*"
        elif text.endswith(".weight"):
            pat = text[:-len(".weight")]
        else:
            pat = text
        if pat not in out:
            out.append(pat)
    return tuple(out) if out else ("*",)


def _build_compressed_tensors_group_rules(
    group: Mapping[str, Any],
    *,
    default_fmt: str,
    native: str,
) -> list[TensorQuantRule]:
    g_fmt = str(group.get("format") or default_fmt).strip().lower()
    weights = group.get("weights") if isinstance(group.get("weights"), Mapping) else {}
    acts = group.get("input_activations") if isinstance(group.get("input_activations"), Mapping) else {}

    w_bits = int(weights.get("num_bits") or (INT4_BITS if "4" in g_fmt else INT8_BITS))
    w_type = str(weights.get("type") or ("float" if ("fp" in g_fmt or "mxfp" in g_fmt) else "int")).lower()
    gsize = max(1, int(weights.get("group_size") or (MXFP_GROUP_SIZE if "mxfp" in g_fmt else AWQ_DEFAULT_GROUP_SIZE)))
    sym = bool(weights.get("symmetric", True))

    if "mxfp4" in g_fmt or (w_type == "float" and w_bits == INT4_BITS):
        a_dtype = "fp8_e4m3" if (acts and int(acts.get("num_bits") or INT8_BITS) == INT8_BITS) else (native or "bf16")
        raw_tag, w_dtype, scale, grp = "W4A8_MXFP", "mxfp4", "fp8_e8m0", (1, gsize)
    elif w_type == "float" and w_bits == INT8_BITS:
        is_fp8_act = (
            (
                bool(acts)
                and str(acts.get("type", "float")).lower() == "float"
                and int(acts.get("num_bits") or INT8_BITS) == INT8_BITS
            )
            if weights
            else True
        )
        raw_tag = "FP8_DYNAMIC" if is_fp8_act else "FP8"
        w_dtype = "fp8_e4m3"
        a_dtype = "fp8_e4m3" if is_fp8_act else (native or "bf16")
        scale = "fp8_e8m0" if "mxfp" in g_fmt else "fp32"
        grp = (1, gsize)
    elif w_type == "int" and w_bits == INT4_BITS:
        is_w4a8 = bool(acts) and str(acts.get("type", "")).lower() == "int" and int(acts.get("num_bits") or 0) == INT8_BITS
        raw_tag = "W4A8_COMPRESSED" if is_w4a8 else "W4A16_COMPRESSED"
        w_dtype = "int4"
        a_dtype = "int8" if is_w4a8 else (native or "bf16")
        scale = _hf_scale_dtype(weights.get("scale_dtype"), native or "bf16")
        grp = (1, gsize)
    elif w_type == "int" and w_bits == INT8_BITS:
        is_w8a8 = bool(acts) and str(acts.get("type", "")).lower() == "int" and int(acts.get("num_bits") or 0) == INT8_BITS
        if is_w8a8:
            raw_tag, w_dtype, a_dtype, scale, grp = "W8A8_DYNAMIC", "int8", "int8", "fp32", (1, 1)
        else:
            raw_tag = "W8A16_GPTQ"
            w_dtype = "int8"
            a_dtype = native or "bf16"
            scale = _hf_scale_dtype(weights.get("scale_dtype"), native or "bf16")
            grp = (1, gsize)
    else:
        raise ModelingError(f"unknown_checkpoint_quant_tag:compressed-tensors:{w_type}:{w_bits}")

    return [
        TensorQuantRule(
            pattern=pat,
            raw_tag=raw_tag,
            weight_dtype=w_dtype,
            activation_dtype=a_dtype,
            scale_dtype=scale,
            group_size=grp,
            symmetric=sym,
        )
        for pat in _normalize_compressed_tensors_targets(group.get("targets"))
    ]


def _build_hf_compressed_tensors_rules(
    qcfg: Mapping[str, Any],
    native: str,
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    fmt = str(qcfg.get("format") or "").strip().lower()
    groups = qcfg.get("config_groups")
    group_values = (
        [g for g in groups.values() if isinstance(g, Mapping)]
        if isinstance(groups, Mapping) and groups
        else [{}]
    )
    if not group_values:
        group_values = [{}]
    all_rules: list[TensorQuantRule] = []
    for group in group_values:
        all_rules.extend(
            _build_compressed_tensors_group_rules(group, default_fmt=fmt, native=native)
        )
    ordered_rules = tuple(dict.fromkeys(sorted(
        all_rules,
        key=lambda r: (
            r.pattern == "*",
            r.pattern,
            r.raw_tag,
            r.weight_dtype,
            r.activation_dtype,
            r.scale_dtype,
            r.group_size,
            bool(r.symmetric),
        ),
    )))
    default_rule = next((r for r in ordered_rules if r.pattern == "*"), ordered_rules[0])
    mtype = default_rule.raw_tag

    kv_scheme = qcfg.get("kv_cache_scheme")
    cache_rule = CacheQuantRule()
    if isinstance(kv_scheme, Mapping) and kv_scheme:
        kv_type = str(kv_scheme.get("type") or "").lower()
        kv_bits = int(kv_scheme.get("num_bits") or INT8_BITS)
        kv_tag = "FP8_DYNAMIC" if kv_type == "float" and kv_bits == INT8_BITS else "C8"
        cache_rule = CacheQuantRule(kv_tag=kv_tag, kv_scale_dtype="fp32", provenance="checkpoint")
    return mtype, ordered_rules, cache_rule


def _build_hf_awq_or_gptq_rules(
    method: str, qcfg: Mapping[str, Any], native: str,
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    default_bits = AWQ_DEFAULT_BITS if method == "awq" else GPTQ_DEFAULT_BITS
    default_group = AWQ_DEFAULT_GROUP_SIZE if method == "awq" else GPTQ_DEFAULT_GROUP_SIZE
    bits = int(qcfg.get("bits") or qcfg.get("w_bit") or default_bits)
    gsize = int(qcfg.get("group_size") or qcfg.get("q_group_size") or default_group)
    if bits not in {INT4_BITS, INT8_BITS}:
        raise ModelingError(f"unknown_checkpoint_quant_tag:{method}:bits={bits}")
    w_dtype = "int4" if bits == INT4_BITS else "int8"
    a_dtype = native or "fp16"
    sym = not bool(qcfg.get("zero_point", True)) if method == "awq" else bool(qcfg.get("sym", True))
    raw_tag = f"W{bits}A16_{method.upper()}"
    rule = TensorQuantRule(
        pattern="*",
        raw_tag=raw_tag,
        weight_dtype=w_dtype,
        activation_dtype=a_dtype,
        scale_dtype=a_dtype,
        group_size=(1, max(1, gsize)),
        symmetric=sym,
    )
    return raw_tag, (rule,), CacheQuantRule()


def _build_hf_quark_rules(
    qcfg: Mapping[str, Any], native: str,
) -> tuple[str, tuple[TensorQuantRule, ...], CacheQuantRule]:
    gcfg = qcfg.get("global_quant_config")
    gcfg = gcfg if isinstance(gcfg, Mapping) else {}
    w_cfg = gcfg.get("weight") if isinstance(gcfg.get("weight"), Mapping) else {}
    a_cfg = gcfg.get("input_tensors") if isinstance(gcfg.get("input_tensors"), Mapping) else {}
    w_raw = str(w_cfg.get("dtype") or "fp8_e4m3").strip().lower()
    a_raw = str(a_cfg.get("dtype") or w_raw).strip().lower()
    gsize = int(w_cfg.get("group_size") or (MXFP_GROUP_SIZE if "fp4" in w_raw or "mx" in w_raw else FP8_DEFAULT_BLOCK_SIZE))
    sym = bool(w_cfg.get("symmetric", True))

    if "fp4" in w_raw:
        w_dtype, a_dtype, scale, tag = "mxfp4", ("fp8_e4m3" if "fp8" in a_raw else (native or "bf16")), "fp8_e8m0", "QUARK_MXFP4"
    elif "fp8" in w_raw:
        raw_scale = w_cfg.get("scale_type") or w_cfg.get("scale_dtype")
        scale = _hf_scale_dtype(raw_scale, "fp8_e8m0" if ("mx" in w_raw or gsize == MXFP_GROUP_SIZE) else "fp32")
        w_dtype, a_dtype, tag = "fp8_e4m3", "fp8_e4m3", "QUARK_FP8"
    elif "int8" in w_raw:
        w_dtype, a_dtype, scale, tag = "int8", "int8", "fp32", "QUARK_INT8"
    else:
        raise ModelingError(f"unknown_checkpoint_quant_tag:quark:{w_raw}")

    rule = TensorQuantRule(
        pattern="*", raw_tag=tag, weight_dtype=w_dtype, activation_dtype=a_dtype,
        scale_dtype=scale, group_size=(1, max(1, gsize)), symmetric=sym,
    )
    kv_raw = str(qcfg.get("kv_cache_dtype") or "").strip().lower()
    cache_rule = CacheQuantRule()
    if kv_raw and kv_raw not in {"bf16", "fp16", "auto", "none"}:
        kv_tag = "FP8_DYNAMIC" if "fp8" in kv_raw else "C8"
        cache_rule = CacheQuantRule(kv_tag=kv_tag, kv_scale_dtype="fp32", provenance="checkpoint")
    return tag, (rule,), cache_rule


def read_hf_quantization_config(
    config_or_qcfg: Mapping[str, Any],
    *,
    checkpoint_id: str,
    base_config: Mapping[str, Any] | None = None,
    source_revision: str = "",
    source_sha256: str = "",
    descriptor_path: str = "config.json",
) -> CheckpointQuantMap:
    """Parse HF ``config.json`` / ``quantization_config`` into ``CheckpointQuantMap``."""
    if not isinstance(config_or_qcfg, Mapping):
        raise ModelingError("hf_quantization_config_requires_mapping")
    text_cfg = config_or_qcfg.get("text_config")
    nested_q = text_cfg.get("quantization_config") if isinstance(text_cfg, Mapping) else None
    if isinstance(config_or_qcfg.get("quantization_config"), Mapping) and config_or_qcfg.get("quantization_config"):
        cfg_root = config_or_qcfg
        raw_q = config_or_qcfg.get("quantization_config")
    elif isinstance(nested_q, Mapping) and nested_q:
        cfg_root = config_or_qcfg
        raw_q = nested_q
    elif "quantization_config" in config_or_qcfg:
        cfg_root = config_or_qcfg
        raw_q = config_or_qcfg.get("quantization_config")
    elif "quant_method" in config_or_qcfg:
        cfg_root = base_config if isinstance(base_config, Mapping) else {}
        raw_q = config_or_qcfg
    elif any(
        k in config_or_qcfg
        for k in ("torch_dtype", "dtype", "model_type", "architectures", "hidden_size", "text_config", "num_hidden_layers", "vocab_size")
    ):
        cfg_root = config_or_qcfg
        raw_q = None
    else:
        cfg_root = base_config if isinstance(base_config, Mapping) else {}
        raw_q = config_or_qcfg
    native = _extract_native_float_dtype(cfg_root)

    if not isinstance(raw_q, Mapping) or not raw_q:
        provenance = CheckpointProvenance(
            checkpoint_id=_sanitize_checkpoint_id(checkpoint_id),
            source_format="unquantized_native",
            source_revision=source_revision,
            source_sha256=source_sha256,
            descriptor_path=_sanitize_descriptor_path(descriptor_path),
            origin="unquantized_native",
        )
        return CheckpointQuantMap(
            provenance=provenance,
            native_float_dtype=native,
            model_quant_type="FLOAT",
            rules=(
                TensorQuantRule(
                    pattern="*", raw_tag="FLOAT",
                    weight_dtype=native, activation_dtype=native,
                ),
            ),
        )

    effective_qcfg = dict(raw_q)
    if "expert_dtype" not in effective_qcfg and isinstance(cfg_root, Mapping):
        root_text = cfg_root.get("text_config")
        outer_expert = cfg_root.get("expert_dtype")
        if not outer_expert and isinstance(root_text, Mapping):
            outer_expert = root_text.get("expert_dtype")
        if isinstance(outer_expert, str) and outer_expert.strip():
            effective_qcfg["expert_dtype"] = outer_expert.strip()

    method = str(effective_qcfg.get("quant_method") or "").strip().lower()
    if method not in _SUPPORTED_HF_QUANT_METHODS:
        raise ModelingError(f"unknown_checkpoint_quant_tag:{method or 'missing_quant_method'}")

    if method == "mxfp8":
        mtype, rules, cache_rule = _build_hf_mxfp8_rules(effective_qcfg)
    elif method == "mxfp4":
        mtype, rules, cache_rule = _build_hf_mxfp4_rules(effective_qcfg, native)
    elif method == "fp8":
        mtype, rules, cache_rule = _build_hf_fp8_rules(effective_qcfg, cfg_root)
    elif method == "compressed-tensors":
        mtype, rules, cache_rule = _build_hf_compressed_tensors_rules(effective_qcfg, native)
    elif method in {"awq", "gptq"}:
        mtype, rules, cache_rule = _build_hf_awq_or_gptq_rules(method, effective_qcfg, native)
    elif method == "quark":
        mtype, rules, cache_rule = _build_hf_quark_rules(effective_qcfg, native)
    else:
        ascend_meta_keys = {
            "quant_method", "version", "format", "config_groups",
            "ignore", "modules_to_not_convert", "optional", "group_size", "expert_dtype",
        }
        inline_items = [
            (str(k), v)
            for k, v in effective_qcfg.items()
            if isinstance(v, str) and v.strip() and str(k) not in ascend_meta_keys
        ]
        if any(k not in _TOP_LEVEL_META_KEYS for k, _ in inline_items):
            qmap, _ = _reduce_modelslim_stream(
                iter(inline_items),
                checkpoint_id=checkpoint_id,
                descriptor_path=descriptor_path,
                source_revision=source_revision,
                native_float_dtype=native,
            )
            ignored_inline = _hf_ignored_modules(effective_qcfg)
            return replace(qmap, ignored_modules=ignored_inline) if ignored_inline else qmap
        mtype = str(effective_qcfg.get("model_quant_type") or "W8A8_DYNAMIC")
        _validate_known_tag(mtype)
        spec = TAG_SPEC_TABLE[normalize_quant_tag_strict(mtype)]
        w_dt = spec["weight_format"] if spec["is_quantized"] else native
        a_dt = spec["activation_format"] if spec["is_quantized"] else native
        s_dt = spec["scale_format"] if spec["is_quantized"] else ""
        rules = (
            TensorQuantRule(
                pattern="*",
                raw_tag=mtype,
                weight_dtype=w_dt,
                activation_dtype=a_dt,
                scale_dtype=s_dt,
                group_size=spec["weight_block"],
            ),
        )
        cache_rule = CacheQuantRule()

    ignored = _hf_ignored_modules(effective_qcfg)
    sha = source_sha256 or hashlib.sha256(
        json.dumps(effective_qcfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provenance = CheckpointProvenance(
        checkpoint_id=_sanitize_checkpoint_id(checkpoint_id),
        source_format="hf_quantization_config",
        source_revision=source_revision,
        source_sha256=sha,
        descriptor_path=_sanitize_descriptor_path(descriptor_path),
        origin="hf_config_quantization",
    )
    return CheckpointQuantMap(
        provenance=provenance,
        native_float_dtype=native,
        model_quant_type=mtype,
        rules=rules,
        hf_quant_config=effective_qcfg,
        ignored_modules=ignored,
        cache_rule=cache_rule,
    )


MIN_PREFIX_CHARS = 8
MIN_PREFIX_USES = 2


def _extract_common_prefixes(patterns: Sequence[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for pat in patterns:
        dot = pat.rfind(".")
        if dot >= MIN_PREFIX_CHARS:
            counts[pat[: dot + 1]] += 1
    # Prefer longest prefixes that occur multiple times.
    candidates = [
        pref for pref, cnt in counts.items() if cnt >= MIN_PREFIX_USES
    ]
    return sorted(candidates, key=lambda p: (-len(p), p))


def _compress_pattern(pat: str, prefixes: Sequence[str]) -> str:
    for idx, pref in enumerate(prefixes):
        if pat.startswith(pref):
            return f"^{idx}/{pat[len(pref):]}"
    return pat


def _expand_pattern(encoded: str, prefixes: Sequence[str]) -> str:
    if encoded.startswith("^"):
        idx_str, _, rest = encoded[1:].partition("/")
        return prefixes[int(idx_str)] + rest
    return encoded


def serialize_curated_entry(
    qmap: CheckpointQuantMap,
    *,
    non_weight_patterns: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Serialize one ``CheckpointQuantMap`` into compact lossless JSON record form."""
    entry: dict[str, Any] = {
        "source_format": qmap.source_format,
        "origin": qmap.provenance.origin,
        "source_sha256": qmap.provenance.source_sha256,
        "descriptor_path": qmap.provenance.descriptor_path,
        "native_float_dtype": qmap.native_float_dtype,
        "model_quant_type": qmap.model_quant_type,
    }
    if qmap.provenance.source_revision:
        entry["source_revision"] = qmap.provenance.source_revision
    if qmap.auxiliary_buffers:
        entry["auxiliary_buffers"] = list(qmap.auxiliary_buffers)
    if qmap.source_format == "hf_quantization_config":
        entry["hf_quant_config"] = _unfreeze_json_value(qmap.hf_quant_config)
        cr = qmap.cache_rule
        if (
            cr.scope != "target"
            or cr.draft_layer_scoped_kv
            or cr.draft_layer_scoped_index
            or cr.layer_scoped_kv
            or cr.layer_scoped_index
        ):
            entry["cache_rule"] = _serialize_curated_cache_rule(cr)
        return entry

    all_pats = [r.pattern for r in qmap.rules] + [e.pattern for e in qmap.layer_exceptions]
    prefixes = _extract_common_prefixes(all_pats)
    if prefixes:
        entry["prefixes"] = prefixes

    grouped: dict[tuple[str, str, str, str], tuple[list[Any], list[Any], list[str], list[str]]] = {}
    for rule in qmap.rules:
        scope = rule.presence.scope if rule.presence is not None else "target"
        layers = encode_int_ranges(rule.presence.layer_indices) if rule.presence is not None else []
        experts = encode_int_ranges(rule.presence.expert_indices) if rule.presence is not None else []
        gkey = (rule.raw_tag, scope, json.dumps(layers, separators=(",", ":")), json.dumps(experts, separators=(",", ":")))
        bucket = grouped.setdefault(gkey, (layers, experts, [], []))
        enc_pat = _compress_pattern(rule.pattern, prefixes)
        if rule.pattern in non_weight_patterns:
            bucket[3].append(enc_pat)
        else:
            bucket[2].append(enc_pat)

    rule_groups: list[list[Any]] = []
    for (raw_tag, scope, _, _), (layers, experts, w_pats, nw_pats) in sorted(grouped.items()):
        row: list[Any] = [raw_tag, scope, layers, experts, sorted(w_pats)]
        if nw_pats:
            row.append(sorted(nw_pats))
        rule_groups.append(row)
    entry["rule_groups"] = rule_groups

    if qmap.layer_exceptions:
        exc_list: list[list[Any]] = []
        for exc in qmap.layer_exceptions:
            row = [
                _compress_pattern(exc.pattern, prefixes),
                exc.scope,
                encode_int_ranges(exc.layer_indices),
                exc.raw_tag,
            ]
            if exc.expert_indices:
                row.append(encode_int_ranges(exc.expert_indices))
            exc_list.append(row)
        entry["layer_exceptions"] = exc_list

    cr = qmap.cache_rule
    if (
        cr.provenance != "native" or cr.kv_tag or cr.index_tag
        or cr.layer_scoped_kv or cr.layer_scoped_index
        or cr.draft_layer_scoped_kv or cr.draft_layer_scoped_index
        or cr.scope != "target"
    ):
        entry["cache_rule"] = _serialize_curated_cache_rule(cr)
    return entry


def _serialize_curated_cache_rule(cr: CacheQuantRule) -> dict[str, Any]:
    cache_obj: dict[str, Any] = {"provenance": cr.provenance}
    if cr.scope != "target":
        cache_obj["scope"] = cr.scope
    if cr.kv_tag:
        cache_obj["kv_tag"] = cr.kv_tag
    if cr.index_tag:
        cache_obj["index_tag"] = cr.index_tag
    if cr.kv_scale_dtype:
        cache_obj["kv_scale_dtype"] = cr.kv_scale_dtype
    if cr.index_scale_dtype:
        cache_obj["index_scale_dtype"] = cr.index_scale_dtype
    if cr.kv_offset_dtype:
        cache_obj["kv_offset_dtype"] = cr.kv_offset_dtype
    if cr.layer_scoped_kv:
        cache_obj["layer_scoped_kv"] = _encode_layer_tag_map(cr.layer_scoped_kv)
    if cr.layer_scoped_index:
        cache_obj["layer_scoped_index"] = _encode_layer_tag_map(cr.layer_scoped_index)
    if cr.draft_layer_scoped_kv:
        cache_obj["draft_layer_scoped_kv"] = _encode_layer_tag_map(cr.draft_layer_scoped_kv)
    if cr.draft_layer_scoped_index:
        cache_obj["draft_layer_scoped_index"] = _encode_layer_tag_map(cr.draft_layer_scoped_index)
    return cache_obj


def _deserialize_curated_cache_rule(raw_cache: Mapping[str, Any]) -> CacheQuantRule:
    return CacheQuantRule(
        kv_tag=str(raw_cache.get("kv_tag") or ""),
        index_tag=str(raw_cache.get("index_tag") or ""),
        kv_scale_dtype=str(raw_cache.get("kv_scale_dtype") or ""),
        index_scale_dtype=str(raw_cache.get("index_scale_dtype") or ""),
        kv_offset_dtype=str(raw_cache.get("kv_offset_dtype") or ""),
        layer_scoped_kv=_decode_layer_tag_map(raw_cache.get("layer_scoped_kv") or {}),
        layer_scoped_index=_decode_layer_tag_map(raw_cache.get("layer_scoped_index") or {}),
        provenance=raw_cache.get("provenance") or "native",
        scope=raw_cache.get("scope") or "target",
        draft_layer_scoped_kv=_decode_layer_tag_map(raw_cache.get("draft_layer_scoped_kv") or {}),
        draft_layer_scoped_index=_decode_layer_tag_map(raw_cache.get("draft_layer_scoped_index") or {}),
    )


def _encode_layer_tag_map(items: Sequence[tuple[int, str]]) -> dict[str, list[int | list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for lid, tag in items:
        grouped[tag].append(lid)
    return {tag: encode_int_ranges(lids) for tag, lids in sorted(grouped.items())}


def _decode_layer_tag_map(raw: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    pairs: list[tuple[int, str]] = []
    for tag, encoded in raw.items():
        for lid in decode_int_ranges(encoded):
            pairs.append((lid, str(tag)))
    return tuple(sorted(pairs))


def _unfreeze_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {k: _unfreeze_json_value(v) for k, v in value}
        return [_unfreeze_json_value(item) for item in value]
    return value


def deserialize_curated_entry(
    checkpoint_id: str, entry: Mapping[str, Any],
) -> tuple[CheckpointQuantMap, frozenset[str]]:
    """Reconstruct an exact ``CheckpointQuantMap`` from one curated JSON entry."""
    source_format: CheckpointFormat = entry["source_format"]
    if source_format == "hf_quantization_config":
        raw_q = entry.get("hf_quant_config") or {}
        base_cfg = {"torch_dtype": entry.get("native_float_dtype") or ""} if entry.get("native_float_dtype") else None
        qmap = read_hf_quantization_config(
            raw_q,
            checkpoint_id=checkpoint_id,
            base_config=base_cfg,
            source_revision=str(entry.get("source_revision") or ""),
            source_sha256=str(entry.get("source_sha256") or ""),
            descriptor_path=str(entry.get("descriptor_path") or "config.json"),
        )
        raw_cache_hf = entry.get("cache_rule")
        if isinstance(raw_cache_hf, Mapping) and raw_cache_hf:
            qmap = replace(qmap, cache_rule=_deserialize_curated_cache_rule(raw_cache_hf))
        return qmap, frozenset()

    prefixes = tuple(str(p) for p in (entry.get("prefixes") or ()))
    rules: list[TensorQuantRule] = []
    non_weight: set[str] = set()
    for group in entry.get("rule_groups") or ():
        raw_tag, scope, layers_enc, experts_enc, w_pats = group[:5]
        nw_pats = group[5] if len(group) > 5 else ()
        validated_tag = _validate_known_tag(str(raw_tag))
        presence = TensorPresenceDomain(
            scope=scope,
            layer_indices=decode_int_ranges(layers_enc),
            expert_indices=decode_int_ranges(experts_enc),
        )
        for enc_pat in w_pats:
            rules.append(
                TensorQuantRule(
                    pattern=_expand_pattern(str(enc_pat), prefixes),
                    raw_tag=validated_tag,
                    presence=presence,
                )
            )
        for enc_pat in nw_pats:
            full_pat = _expand_pattern(str(enc_pat), prefixes)
            non_weight.add(full_pat)
            rules.append(
                TensorQuantRule(pattern=full_pat, raw_tag=validated_tag, presence=presence)
            )
    for row in entry.get("rules") or ():
        pattern, raw_tag, scope, layers_enc, experts_enc = row[:5]
        full_pat = _expand_pattern(str(pattern), prefixes)
        is_weight = bool(row[5]) if len(row) > 5 else True
        if not is_weight:
            non_weight.add(full_pat)
        rules.append(
            TensorQuantRule(
                pattern=full_pat,
                raw_tag=_validate_known_tag(str(raw_tag)),
                presence=TensorPresenceDomain(
                    scope=scope,
                    layer_indices=decode_int_ranges(layers_enc),
                    expert_indices=decode_int_ranges(experts_enc),
                ),
            )
        )
    rules.sort(key=lambda r: (r.presence.scope if r.presence else "target", r.pattern))

    exceptions: list[LayerQuantException] = []
    for row in entry.get("layer_exceptions") or ():
        pattern, scope, layers_enc, raw_tag = row[:4]
        experts_enc = row[4] if len(row) > 4 else ()
        exceptions.append(
            LayerQuantException(
                pattern=_expand_pattern(str(pattern), prefixes),
                scope=scope,
                layer_indices=decode_int_ranges(layers_enc),
                raw_tag=_validate_known_tag(str(raw_tag)),
                expert_indices=decode_int_ranges(experts_enc),
            )
        )

    raw_cache = entry.get("cache_rule") or {}
    cache_rule = _deserialize_curated_cache_rule(raw_cache)
    origin: CheckpointOrigin = entry.get("origin") or "verified_checkpoint_descriptor"
    provenance = CheckpointProvenance(
        checkpoint_id=checkpoint_id,
        source_format=source_format,
        source_revision=str(entry.get("source_revision") or ""),
        source_sha256=str(entry.get("source_sha256") or ""),
        descriptor_path=str(entry.get("descriptor_path") or ""),
        origin=origin,
    )
    qmap = CheckpointQuantMap(
        provenance=provenance,
        schema_version=CHECKPOINT_QUANT_MAP_SCHEMA,
        native_float_dtype=str(entry.get("native_float_dtype") or ""),
        model_quant_type=str(entry.get("model_quant_type") or ""),
        rules=tuple(rules),
        layer_exceptions=tuple(exceptions),
        auxiliary_buffers=tuple(str(b) for b in (entry.get("auxiliary_buffers") or ())),
        cache_rule=cache_rule,
    )
    return qmap, frozenset(non_weight)


@lru_cache(maxsize=1)
def _load_curated_store() -> tuple[dict[str, CheckpointQuantMap], dict[str, str], dict[str, frozenset[str]]]:
    data_ref = resources.files("mindie_sim.catalog").joinpath(
        CURATED_DATA_REL_DIR, CURATED_DATA_FILENAME,
    )
    raw = json.loads(data_ref.read_text(encoding="utf-8"))
    if raw.get("schema_version") != CURATED_CHECKPOINTS_SCHEMA:
        raise ModelingError(f"curated_checkpoints_schema_invalid:{raw.get('schema_version')!r}")
    by_id: dict[str, CheckpointQuantMap] = {}
    lower_index: dict[str, str] = {}
    non_weights: dict[str, frozenset[str]] = {}
    for repo_id, entry in sorted((raw.get("checkpoints") or {}).items()):
        qmap, nw = deserialize_curated_entry(repo_id, entry)
        by_id[repo_id] = qmap
        lower_index[repo_id.lower()] = repo_id
        non_weights[repo_id] = nw
    return by_id, lower_index, non_weights


def list_curated_checkpoint_ids() -> tuple[str, ...]:
    """Return all curated checkpoint repository IDs packaged with the library."""
    by_id, _, _ = _load_curated_store()
    return tuple(sorted(by_id))


def lookup_curated_checkpoint(repo_id: str) -> CheckpointQuantMap | None:
    """Pure read-only lookup of a packaged curated ``CheckpointQuantMap``."""
    text = str(repo_id or "").strip()
    if not text:
        return None
    by_id, lower_index, _ = _load_curated_store()
    canonical = lower_index.get(text.lower())
    return by_id.get(canonical) if canonical is not None else None


def curated_weight_pattern_map(repo_id: str) -> dict[str, tuple[str, ...]] | None:
    """Return legacy ``{tag: (weight_pattern, ...)}`` view derived from curated data."""
    text = str(repo_id or "").strip()
    if not text:
        return None
    by_id, lower_index, non_weights = _load_curated_store()
    canonical = lower_index.get(text.lower())
    if canonical is None:
        return None
    qmap = by_id[canonical]
    if qmap.source_format != "modelslim_description":
        return None
    nw = non_weights.get(canonical, frozenset())
    grouped: dict[str, list[str]] = defaultdict(list)
    for rule in qmap.rules:
        if rule.pattern in nw:
            continue
        pat = rule.pattern
        if pat.endswith(".ngram_embedding.*"):
            pat = pat[:-1] + "shard_*"
        grouped[rule.raw_tag].append(f"{pat}.weight")
    return {tag: tuple(sorted(pats)) for tag, pats in sorted(grouped.items())}


def read_checkpoint_quant_map(
    source: Path | str | Mapping[str, Any],
    *,
    checkpoint_id: str = "",
    base_config: Mapping[str, Any] | None = None,
    chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
    force_stdlib: bool = False,
) -> CheckpointQuantMap:
    """Pure analytical reader: local file/dir, packaged curated ID, or mapping.

    Performs ZERO network I/O and ZERO cache writes.
    """
    if isinstance(source, Mapping):
        if "quantization_config" in source or "quant_method" in source:
            return read_hf_quantization_config(
                source, checkpoint_id=checkpoint_id or "inline://hf-config",
                base_config=base_config,
            )
        return read_modelslim_description_mapping(
            source, checkpoint_id=checkpoint_id or "inline://modelslim",
            base_config=base_config,
        )

    raw_str = str(source).strip()
    if not raw_str:
        raise ModelingError("checkpoint_source_empty")
    local = Path(source).expanduser() if isinstance(source, Path) else Path(raw_str).expanduser()
    if local.exists():
        return _read_local_checkpoint_path(
            local, checkpoint_id=checkpoint_id, base_config=base_config,
            chunk_size=chunk_size, force_stdlib=force_stdlib,
        )

    curated = lookup_curated_checkpoint(raw_str)
    if curated is not None:
        return curated
    raise ModelingError(f"checkpoint_not_found_locally_or_curated:{raw_str}")


def _read_bounded_config_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    if path.stat().st_size > MAX_CONFIG_FILE_BYTES:
        raise ModelingError(f"checkpoint_config_exceeds_limit:{path.name}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelingError(f"checkpoint_config_unreadable:{path.name}:{type(exc).__name__}") from exc
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _read_local_checkpoint_path(
    local: Path,
    *,
    checkpoint_id: str,
    base_config: Mapping[str, Any] | None,
    chunk_size: int,
    force_stdlib: bool,
) -> CheckpointQuantMap:
    base_dir = local if local.is_dir() else local.parent
    cfg = dict(base_config) if isinstance(base_config, Mapping) else {}
    cfg_path = local if local.is_file() and local.name == "config.json" else base_dir / "config.json"
    if not cfg and cfg_path.is_file():
        cfg = _read_bounded_config_json(cfg_path)

    desc_path: Path | None = None
    if local.is_file() and "quant_model_description" in local.name:
        desc_path = local
    elif (base_dir / "quant_model_description.json").is_file():
        desc_path = base_dir / "quant_model_description.json"
    else:
        candidates = sorted(base_dir.glob("quant_model_description*.json"))
        if candidates:
            desc_path = candidates[0]

    cid = checkpoint_id or f"local://{base_dir.name or local.stem}"
    if desc_path is not None:
        return read_modelslim_description_file(
            desc_path, checkpoint_id=cid, base_config=cfg or None,
            chunk_size=chunk_size, force_stdlib=force_stdlib,
        )
    text_cfg = cfg.get("text_config") if isinstance(cfg, Mapping) else None
    has_inline_qcfg = bool(
        (isinstance(cfg.get("quantization_config"), Mapping) and cfg.get("quantization_config"))
        or (
            isinstance(text_cfg, Mapping)
            and isinstance(text_cfg.get("quantization_config"), Mapping)
            and text_cfg.get("quantization_config")
        )
    )
    qcfg_path: Path | None = None
    if local.is_file() and local.name == "quantize_config.json":
        qcfg_path = local
    elif not has_inline_qcfg and (base_dir / "quantize_config.json").is_file():
        qcfg_path = base_dir / "quantize_config.json"
    if qcfg_path is not None:
        try:
            raw_qcfg = json.loads(qcfg_path.read_text(encoding="utf-8"))
        except TimeoutError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelingError(
                f"checkpoint_quantize_config_invalid:{qcfg_path.name}:{type(exc).__name__}"
            ) from exc
        if isinstance(raw_qcfg, Mapping) and raw_qcfg:
            qcfg_dict = dict(raw_qcfg)
            if not str(qcfg_dict.get("quant_method") or "").strip():
                if any(k in qcfg_dict for k in ("w_bit", "q_group_size", "zero_point")):
                    qcfg_dict["quant_method"] = "awq"
                elif any(k in qcfg_dict for k in ("bits", "group_size", "sym")):
                    qcfg_dict["quant_method"] = "gptq"
            merged_cfg = dict(cfg) if isinstance(cfg, Mapping) else {}
            merged_cfg["quantization_config"] = qcfg_dict
            return read_hf_quantization_config(
                merged_cfg, checkpoint_id=cid, descriptor_path=qcfg_path.name,
            )
    return read_hf_quantization_config(
        cfg, checkpoint_id=cid, descriptor_path=cfg_path.name if cfg_path.is_file() else "config.json",
    )


def resolve_checkpoint_tensor_tag(
    qmap: CheckpointQuantMap,
    tensor_path: str,
    *,
    scope: ModelScope | None = None,
    source_native_layer_id: int | None = None,
    expert_id: int | None = None,
) -> str | None:
    """Resolve one concrete tensor path against ``CheckpointQuantMap``.

    Strips ``.weight`` when present, infers coordinates if omitted, checks
    exact ``CheckpointQuantMap.resolve_raw_tag`` precedence first, and handles
    HF ``quantization_config`` wildcard/expert rules cleanly.
    """
    raw = str(tensor_path or "").strip()
    if not raw:
        return None
    mod_path = raw[:-len(".weight")] if raw.endswith(".weight") else raw
    eff_scope: ModelScope = scope if scope is not None else _infer_scope(raw)
    _, inferred_layer, inferred_expert = _extract_pattern_and_coords(mod_path)
    eff_layer = source_native_layer_id if source_native_layer_id is not None else inferred_layer
    eff_expert = expert_id if expert_id is not None else inferred_expert

    direct = qmap.resolve_raw_tag(
        mod_path, scope=eff_scope, source_native_layer_id=eff_layer, expert_id=eff_expert,
    )
    if direct is not None:
        return direct
    if mod_path != raw:
        direct_raw = qmap.resolve_raw_tag(
            raw, scope=eff_scope, source_native_layer_id=eff_layer, expert_id=eff_expert,
        )
        if direct_raw is not None:
            return direct_raw

    if qmap.source_format == "unquantized_native":
        return "FLOAT"
    if qmap.source_format == "hf_quantization_config":
        for rule in qmap.rules:
            if rule.pattern.startswith("re:"):
                expr = rule.pattern[len("re:"):]
                if expr and (re.search(expr, mod_path) is not None or re.search(expr, raw) is not None):
                    return rule.raw_tag
        parts = set(mod_path.lower().split("."))
        if "experts" in parts:
            for rule in qmap.rules:
                if rule.pattern == "*.experts.*":
                    return rule.raw_tag
        for rule in qmap.rules:
            if rule.pattern == "*":
                return rule.raw_tag
    return None


_VISION_PARTS = frozenset({"vision_tower", "vision_model", "visual"})
_PROJECTOR_PARTS = frozenset({
    "mm_projector", "multi_modal_projector", "merger", "aligner",
    "patch_merge_mlp", "vit_large_projector", "deepstack_merger_list",
})
_EMBED_PARTS = frozenset({"embed_tokens", "word_embeddings"})
_MHC_PARTS = frozenset({
    "attn_hyper_connection", "mlp_hyper_connection", "hyper_connection_mixer",
    "attn_hc", "ffn_hc",
})
_MHC_PREFIXES = ("hc_attn_", "hc_ffn_", "hc_head_")
_PLE_PARTS = frozenset({"ple", "ple_embedding"})
_PLE_PROJ_LAST = frozenset({"key_proj", "value_proj"})
_PLE_TABLE_PARTS = frozenset({"ngram_embedding", "ple_embedding"})
_CONV_LAST_PARTS = frozenset({"conv1d", "q_conv1d", "k_conv1d", "v_conv1d"})
_ROUTER_LAST_PARTS = frozenset({"shared_expert_gate", "router_bias", "e_score_correction_bias", "gate"})
_LATENT_MOE_LAST = frozenset({"routed_expert_down_proj", "routed_expert_up_proj"})
_SHARED_EXPERT_PARTS = frozenset({"shared_experts", "shared_expert", "share_expert"})
_MOE_EXPERT_LAST_PARTS = frozenset({"gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"})
_GDN_BETA_ALPHA_LAST = frozenset({"in_proj_a", "in_proj_b", "in_proj_ba"})
_INDEXER_LAST_PARTS = frozenset({"index_q_proj", "index_k_proj", "qkv_indexer_proj"})
_INDEXER_WK_LAST = frozenset({"wk", "index_k_proj"})
_INDEXER_WQ_LAST = frozenset({"wq_b", "index_qk_proj", "index_q_proj", "wq"})
_ATTN_PARENT_PARTS = frozenset({"self_attn", "attn"})
_QKV_LAST_PARTS = frozenset({"q_proj", "k_proj", "v_proj", "qkv_proj"})
_MLP_PARENT_PARTS = frozenset({"mlp", "ffn"})
_MLP_GATE_LAST = frozenset({"gate_proj", "w1"})
_MLP_UP_LAST = frozenset({"up_proj", "w3"})
_MLP_DOWN_LAST = frozenset({"down_proj", "w2"})
_EXACT_ATTN_MLP_MAP: dict[str, str] = {
    "kv_b_proj": "mla_absorb",
    "kv_a_proj_with_mqa": "mla.kv_a",
    "wkv": "mla.kv_a",
    "q_a_proj": "mla.q_a",
    "wq_a": "mla.q_a",
    "q_b_proj": "mla.q_b",
    "wq_b": "mla.q_b",
    "wo_b": "mla.o",
    "f_a_proj": "kda.f_a",
    "f_b_proj": "kda.f_b",
    "b_proj": "kda.b",
}

_ARCH_FLAGS_ID_CACHE: OrderedDict[int, tuple[CheckpointQuantMap, tuple[bool, bool, bool]]] = OrderedDict()
_QUANT_SPEC_ID_CACHE: OrderedDict[
    tuple[int, str, int, str, str, str], tuple[CheckpointQuantMap, QuantSpec]
] = OrderedDict()


@lru_cache(maxsize=QUANT_MAP_CACHE_MAXSIZE)
def _infer_arch_flags_cached(patterns: tuple[str, ...]) -> tuple[bool, bool, bool]:
    has_mla = False
    has_kda = False
    has_gdn = False
    for pat in patterns:
        parts = set(str(pat).lower().split("."))
        if parts & _MLA_EXACT_COMPONENTS:
            has_mla = True
        if parts & _KDA_EXACT_COMPONENTS:
            has_kda = True
        if "linear_attn" in parts:
            has_gdn = True
        if has_mla and has_kda and has_gdn:
            break
    return has_mla, has_kda, has_gdn


def _arch_flags_for_checkpoint_map(qmap: CheckpointQuantMap) -> tuple[bool, bool, bool]:
    """O(1)-amortized MLA / KDA / GDN architecture flags for a frozen CheckpointQuantMap."""
    key = id(qmap)
    cached = _ARCH_FLAGS_ID_CACHE.get(key)
    if cached is not None and cached[0] is qmap:
        return cached[1]
    flags = _infer_arch_flags_cached(tuple(r.pattern for r in qmap.rules))
    _ARCH_FLAGS_ID_CACHE[key] = (qmap, flags)
    _ARCH_FLAGS_ID_CACHE.move_to_end(key)
    while len(_ARCH_FLAGS_ID_CACHE) > QUANT_MAP_CACHE_MAXSIZE:
        _ARCH_FLAGS_ID_CACHE.popitem(last=False)
    return flags


def infer_arch_flags_from_patterns(patterns: Sequence[str]) -> tuple[bool, bool, bool]:
    """Detect MLA / KDA / GDN architecture presence via exact path components."""
    tup = patterns if type(patterns) is tuple else tuple(patterns)
    try:
        return _infer_arch_flags_cached(tup)
    except TypeError:
        return _infer_arch_flags_cached(tuple(str(pat) for pat in tup))


def classify_tensor_categories(
    tensor_name: str, *, has_mla: bool, has_kda: bool,
) -> tuple[str, ...]:
    """Map one tensor key or pattern to canonical simulator categories using exact components."""
    lower = str(tensor_name or "").lower()
    raw_parts = [p for p in lower.split(".") if p and p != "weight"]
    parts = set(raw_parts)
    last = raw_parts[-1] if raw_parts else ""

    if "mtp" in parts or "mtp_block" in parts:
        return ("mtp",)
    if lower.startswith(_VISION_PREFIXES) or not parts.isdisjoint(_VISION_PARTS):
        return ("vision",)
    if not parts.isdisjoint(_PROJECTOR_PARTS):
        return ("projector",)
    if "lm_head" in parts or last == "head":
        return ("lm_head",)
    if not parts.isdisjoint(_EMBED_PARTS) or last == "embed" or ("engram" in parts and "embed" in parts):
        return ("embedding",)
    if last in _ATTN_RES_PROJ_COMPONENTS or last.endswith("_res_proj"):
        return ("attn_res_proj",)
    if not parts.isdisjoint(_MHC_PARTS) or last.startswith(_MHC_PREFIXES):
        return ("mhc",)
    if not parts.isdisjoint(_PLE_PARTS):
        if last in _PLE_PROJ_LAST:
            return ("ple_proj", "ple")
        if not parts.isdisjoint(_PLE_TABLE_PARTS):
            return ("ple_table", "ple")
        return ("ple",)
    if last in _CONV_LAST_PARTS:
        return ("conv_weight_f32",) if has_kda else ("conv_weight_bf16",)
    if last in _NORM_EXACT_COMPONENTS or last.endswith("layernorm") or last.endswith("_norm"):
        return ("norm",)
    if last in _ROUTER_LAST_PARTS or "router" in parts:
        return ("router",)
    if last in _LATENT_MOE_LAST:
        return ("latent_moe",)
    if "experts" in parts:
        return ("routed_expert",)
    if not parts.isdisjoint(_SHARED_EXPERT_PARTS):
        return ("shared_expert",)
    if "moe" in parts and last in _MOE_EXPERT_LAST_PARTS:
        return ("routed_expert",)
    if "linear_attn" in parts:
        if last == "out_proj":
            return ("gdn.o", "gdn")
        if last in _GDN_BETA_ALPHA_LAST:
            return ("gdn.beta_alpha", "gdn")
        return ("gdn.qkvz", "gdn")
    if "indexer" in parts or last in _INDEXER_LAST_PARTS:
        if last == "weights_proj":
            return ("indexer.weights",)
        if last in _INDEXER_WK_LAST:
            return ("indexer.wk",)
        if last in _INDEXER_WQ_LAST:
            return ("indexer_fp8", "indexer.wq")
        return ("indexer",)
    return _classify_attn_mlp_component(raw_parts, parts, last, has_mla=has_mla, has_kda=has_kda)


def _classify_attn_mlp_component(
    raw_parts: Sequence[str],
    parts: set[str],
    last: str,
    *,
    has_mla: bool,
    has_kda: bool,
) -> tuple[str, ...]:
    del raw_parts
    exact = _EXACT_ATTN_MLP_MAP.get(last)
    if exact is not None:
        return (exact,)
    if last == "g_proj":
        if has_mla and has_kda:
            return ("kda.g", "mla.g")
        return ("kda.g",) if has_kda else ("mla.g",)
    if last == "o_proj" and not parts.isdisjoint(_ATTN_PARENT_PARTS):
        if has_mla and has_kda:
            return ("mla.o", "kda.o")
        if has_mla:
            return ("mla.o",)
        return ("kda.o",) if has_kda else ("attn.o",)
    if last in _QKV_LAST_PARTS and not parts.isdisjoint(_ATTN_PARENT_PARTS):
        if has_kda:
            if last == "qkv_proj":
                return ("kda.q", "kda.k", "kda.v")
            return ("kda.q",) if last == "q_proj" else (("kda.k",) if last == "k_proj" else ("kda.v",))
        return ("attn.qkv",)
    if not parts.isdisjoint(_MLP_PARENT_PARTS):
        if last in _MLP_GATE_LAST:
            return ("dense_mlp.gate",)
        if last in _MLP_UP_LAST:
            return ("dense_mlp.up",)
        if last in _MLP_DOWN_LAST:
            return ("dense_mlp.down",)
        if last == "gate_up_proj":
            return ("dense_mlp.gate", "dense_mlp.up")
    return ()


def _cache_dtypes_from_rule(cache_rule: CacheQuantRule) -> tuple[str, str, str]:
    kv_u = cache_rule.kv_tag.upper()
    idx_u = cache_rule.index_tag.upper()
    if "MXFP8" in kv_u or "FP8" in kv_u:
        kv_dt = "fp8_e4m3"
    elif kv_u in {"C8", "FAQUANT", "FAKQUANT", "INT8_DYNAMIC"} or "INT8" in kv_u:
        kv_dt = "int8"
    else:
        kv_dt = ""

    if "MXFP8" in idx_u or "FP8" in idx_u:
        idx_dt = "fp8_e4m3"
    elif "INT8" in idx_u:
        idx_dt = "int8"
    else:
        idx_dt = ""

    idx_scale = cache_rule.index_scale_dtype
    if not idx_scale:
        if idx_dt in {"fp8", "fp8_e4m3"}:
            idx_scale = "fp8_e8m0"
        elif idx_dt == "int8":
            idx_scale = "fp16"
        else:
            idx_scale = "none"
    return kv_dt, idx_dt, idx_scale


def _override_fields_for_tag(tag: str, native_format: str) -> dict[str, Any]:
    spec = TAG_SPEC_TABLE[normalize_quant_tag_strict(tag)]
    if not spec["is_quantized"]:
        return {
            "weight_format": native_format,
            "activation_format": native_format,
        }
    out = {
        k: spec[k]
        for k in (
            "weight_format", "activation_format", "weight_block", "expert_block",
            "scale_format", "scale_bytes", "activation_group_size",
            "activation_scale_format", "activation_scale_pack",
        )
    }
    for rt_key in (
        "runtime_weight_scale_format", "runtime_weight_scale_bytes",
        "runtime_activation_scale_format", "runtime_int8_weight_metadata",
    ):
        if rt_key in spec:
            out[rt_key] = spec[rt_key]
    return out


def _pick_dominant_tag(tags: Sequence[str], default: str) -> str:
    counts: dict[str, int] = {}
    for tag in tags:
        norm = normalize_quant_tag_strict(tag)
        if norm != "FLOAT":
            counts[norm] = counts.get(norm, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else default


_SKIP_OVERRIDE_CATS = frozenset({
    "norm", "router", "lm_head", "embedding", "vision", "projector",
})
_MXFP8_KV_PACKET_TAG = "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"
_LOW_PRECISION_ACTIVATIONS = frozenset({"fp8_e4m3", "fp8_e5m2", "int8", "int4", "fp4_e2m1"})
_FLOAT_DTYPE_TOKENS = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})


def _override_fields_for_hf_rule(rule: TensorQuantRule, native: str) -> dict[str, Any]:
    wb = (
        (int(rule.group_size[0]), int(rule.group_size[1]))
        if len(rule.group_size) == 2
        else (1, int(rule.group_size[0]) if rule.group_size else 1)
    )
    w_fmt = rule.weight_dtype or native
    a_fmt = rule.activation_dtype or native
    s_fmt = rule.scale_dtype or ("fp8_e8m0" if w_fmt == "mxfp4" else "fp32")
    s_bytes = float(
        dtype_bytes("fp8_e4m3" if s_fmt == "fp8_e8m0" else (s_fmt if s_fmt != "none" else "fp32"))
    )
    act_is_low = a_fmt in _LOW_PRECISION_ACTIVATIONS
    act_group = int(wb[1]) if (act_is_low and w_fmt in {"fp8_e4m3", "fp8_e5m2", "mxfp4"}) else 0
    return {
        "weight_format": w_fmt,
        "activation_format": a_fmt,
        "weight_block": wb,
        "expert_block": int(wb[1]),
        "scale_format": s_fmt,
        "scale_bytes": s_bytes,
        "activation_group_size": act_group,
        "activation_scale_format": s_fmt if act_is_low else "",
        "activation_scale_pack": MXFP_SCALE_PACK if (act_is_low and s_fmt == "fp8_e8m0") else 1,
        "weight_symmetric": True if rule.symmetric is None else bool(rule.symmetric),
    }


_DENSE_MLP_SUB_CATS: tuple[str, ...] = (
    "dense_mlp.gate", "dense_mlp.up", "dense_mlp.down",
)


def _expand_simple_regex_variants(text: str) -> tuple[str, ...]:
    open_idx = text.find("(")
    close_idx = text.find(")", open_idx + 1) if open_idx >= 0 else -1
    if open_idx < 0 or close_idx < 0:
        return (text,)
    prefix, body, suffix = text[:open_idx], text[open_idx + 1:close_idx], text[close_idx + 1:]
    if any(ch in prefix or ch in suffix for ch in "()|") or "(" in body or ")" in body:
        return (text,)
    parts = [part.strip() for part in body.split("|") if part.strip()]
    return tuple(f"{prefix}{part}{suffix}" for part in parts) if parts else (text,)


def _coalesce_dense_mlp_layer_overrides(
    native_layers: dict[str, dict[str, Any]],
    *,
    keep_children: bool = False,
) -> None:
    """Coalesce identical ``dense_mlp.{gate,up,down}`` layer overrides onto ``dense_mlp``."""
    sub_set = set(_DENSE_MLP_SUB_CATS)
    for entry in native_layers.values():
        cat_map = entry.get("category_overrides")
        if not isinstance(cat_map, dict) or not sub_set.issubset(cat_map):
            continue
        first = cat_map[_DENSE_MLP_SUB_CATS[0]]
        if all(cat_map[sub] == first for sub in _DENSE_MLP_SUB_CATS[1:]):
            if not keep_children:
                for sub in _DENSE_MLP_SUB_CATS:
                    cat_map.pop(sub, None)
            cat_map["dense_mlp"] = dict(first)


def _classify_hf_target_categories(
    raw_pat: str,
    *,
    has_mla: bool,
    has_kda: bool,
    allow_lm_head: bool = False,
) -> tuple[str, ...]:
    cleaned = str(raw_pat or "").strip()
    is_regex = cleaned.startswith("re:")
    if is_regex:
        cleaned = (
            cleaned[len("re:"):]
            .replace("\\.", ".")
            .replace(".*", ".")
            .strip("^$")
        )
    variants = _expand_simple_regex_variants(cleaned) if is_regex else (cleaned,)
    skip = _SKIP_OVERRIDE_CATS - {"lm_head"} if allow_lm_head else _SKIP_OVERRIDE_CATS
    collected: list[str] = []
    for variant in variants:
        lower = variant.lower()
        if lower.endswith(".weight"):
            lower = lower[:-len(".weight")]
        raw_parts = [p for p in lower.split(".") if p]
        last = raw_parts[-1] if raw_parts else ""
        if last in {"self_attn", "attention", "attn"}:
            v_cats: tuple[str, ...] = ("mla", "mla_absorb") if has_mla else (("kda",) if has_kda else ("attn",))
        elif last == "linear_attn":
            v_cats = ("gdn",)
        elif last in {"mlp", "ffn"}:
            v_cats = ("dense_mlp", "shared_expert", "routed_expert")
        elif last == "experts":
            v_cats = ("routed_expert",)
        elif last in {"shared_experts", "shared_expert", "share_expert"}:
            v_cats = ("shared_expert",)
        else:
            v_cats = tuple(
                c for c in classify_tensor_categories(variant, has_mla=has_mla, has_kda=has_kda)
                if c not in skip
            )
        for c in v_cats:
            if c not in collected:
                collected.append(c)
    if set(_DENSE_MLP_SUB_CATS).issubset(collected):
        collapsed = [c for c in collected if c not in _DENSE_MLP_SUB_CATS]
        if "dense_mlp" not in collapsed:
            collapsed.append("dense_mlp")
        return tuple(collapsed)
    return tuple(collected)


def _extract_expert_count_hint(base_config: Mapping[str, Any] | None) -> int:
    if not isinstance(base_config, Mapping):
        return 0
    text_cfg = base_config.get("text_config")
    for cfg in (base_config, text_cfg if isinstance(text_cfg, Mapping) else {}):
        for key in ("n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts"):
            val = cfg.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                return int(val)
    return 0


def _extract_hf_pattern_coords(raw: str) -> tuple[int | None, int | None]:
    """Extract literal ``(layer_id, expert_id)`` from a literal or anchored ``re:`` HF pattern."""
    if not raw.startswith("re:"):
        _, lid, eid = _extract_pattern_and_coords(raw)
        return lid, eid
    body = raw[len("re:"):].strip()
    if body.startswith("^"):
        body = body[1:]
    if body.endswith("$"):
        body = body[:-1]
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        elif ch == "|" and depth == 0:
            return None, None
    _, lid, eid = _extract_pattern_and_coords(body.replace(r"\.", "."))
    return lid, eid


def _apply_ignored_modules_to_overrides(
    ignored_modules: Sequence[str],
    *,
    has_mla: bool,
    has_kda: bool,
    native: str,
    cat_overrides: dict[str, dict[str, Any]],
    native_layers: dict[str, dict[str, Any]],
    extra_exclusions: list[str],
    partial_by_layer_cat: dict[tuple[int, str], dict[int, tuple[tuple[str, Any], ...]]],
    expert_count_by_layer_cat: dict[tuple[int, str], int],
    expert_count_hint: int = 0,
) -> None:
    float_fields = {"weight_format": native, "activation_format": native}
    float_fkey = _freeze_fields_key(float_fields)
    for mod in ignored_modules:
        if "mtp" in mod.lower():
            extra_exclusions.append("mtp")
        if _infer_scope(mod) != "target":
            continue
        lid, eid = _extract_hf_pattern_coords(mod)
        cats = _classify_hf_target_categories(
            mod, has_mla=has_mla, has_kda=has_kda, allow_lm_head=True,
        )
        if "lm_head" in cats:
            cat_overrides.pop("lm_head", None)
            if "lm_head" not in extra_exclusions:
                extra_exclusions.append("lm_head")
            cats = tuple(c for c in cats if c != "lm_head")
        if not cats:
            continue
        if lid is not None and eid is not None:
            for cat in cats:
                key = (int(lid), cat)
                partial_by_layer_cat[key][int(eid)] = float_fkey
                expert_count_by_layer_cat[key] = max(
                    expert_count_by_layer_cat.get(key, 0), expert_count_hint, int(eid) + 1,
                )
        elif lid is not None:
            entry = native_layers.setdefault(str(int(lid)), {"category_overrides": {}})
            cat_map = entry.setdefault("category_overrides", {})
            for cat in cats:
                cat_map[cat] = dict(float_fields)
        elif eid is None:
            for cat in cats:
                cat_overrides[cat] = dict(float_fields)


def _hf_layer_and_category_overrides(
    qmap: CheckpointQuantMap,
    *,
    native: str,
    expert_count_hint: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    all_pats = tuple(r.pattern for r in qmap.rules) + tuple(qmap.ignored_modules)
    has_mla, has_kda, _ = infer_arch_flags_from_patterns(all_pats)
    cat_overrides: dict[str, dict[str, Any]] = {}
    native_layers: dict[str, dict[str, Any]] = {}
    partial_by_layer_cat: dict[tuple[int, str], dict[int, tuple[tuple[str, Any], ...]]] = defaultdict(dict)
    expert_count_by_layer_cat: dict[tuple[int, str], int] = {}
    extra_exclusions: list[str] = []

    for rule in qmap.rules:
        if rule.pattern in {"*", "*.experts.*"}:
            continue
        scope = _infer_scope(rule.pattern)
        if scope != "target":
            continue
        lid, eid = _extract_hf_pattern_coords(rule.pattern)
        cats = _classify_hf_target_categories(
            rule.pattern, has_mla=has_mla, has_kda=has_kda, allow_lm_head=True,
        )
        if not cats:
            continue
        is_float_rule = (
            str(rule.weight_dtype or "").strip().lower() in _FLOAT_DTYPE_TOKENS
            or str(rule.raw_tag or "").strip().upper() == "FLOAT"
        )
        if "lm_head" in cats and is_float_rule:
            cat_overrides.pop("lm_head", None)
            cats = tuple(c for c in cats if c != "lm_head")
            if not cats:
                continue
        fields = _override_fields_for_hf_rule(rule, native)
        if lid is not None and eid is not None:
            fkey = _freeze_fields_key(fields)
            for cat in cats:
                key = (int(lid), cat)
                partial_by_layer_cat[key][int(eid)] = fkey
                expert_count_by_layer_cat[key] = max(expert_count_by_layer_cat.get(key, 0), expert_count_hint, int(eid) + 1)
        elif lid is not None:
            entry = native_layers.setdefault(str(int(lid)), {"category_overrides": {}})
            for cat in cats:
                entry["category_overrides"][cat] = dict(fields)
        else:
            for cat in cats:
                cat_overrides[cat] = dict(fields)

    _apply_ignored_modules_to_overrides(
        qmap.ignored_modules,
        has_mla=has_mla,
        has_kda=has_kda,
        native=native,
        cat_overrides=cat_overrides,
        native_layers=native_layers,
        extra_exclusions=extra_exclusions,
        partial_by_layer_cat=partial_by_layer_cat,
        expert_count_by_layer_cat=expert_count_by_layer_cat,
        expert_count_hint=expert_count_hint,
    )
    _attach_partial_expert_overrides(native_layers, partial_by_layer_cat, expert_count_by_layer_cat)
    _coalesce_dense_mlp_layer_overrides(
        native_layers,
        keep_children=any(sub in cat_overrides for sub in _DENSE_MLP_SUB_CATS),
    )
    layer_ov = {"native_layers": native_layers} if native_layers else {}
    return cat_overrides, layer_ov, extra_exclusions


def _quant_spec_from_hf_map(
    qmap: CheckpointQuantMap,
    *,
    native: str,
    kv_override: str,
    index_override: str,
    checkpoint_url: str,
    base_config: Mapping[str, Any] | None = None,
    expert_count_hint: int | None = None,
) -> QuantSpec:
    default_rule = next((r for r in qmap.rules if r.pattern == "*"), None)
    expert_rule = next((r for r in qmap.rules if r.pattern == "*.experts.*"), None)
    target_cache_rule = qmap.cache_rule if qmap.cache_rule.scope == "target" else CacheQuantRule()
    kv_c, idx_c, idx_scale = _cache_dtypes_from_rule(target_cache_rule)
    kv_dtype = kv_override or kv_c or native
    index_dtype = index_override or idx_c or native
    if index_dtype not in {"int8", "fp8", "fp8_e4m3"}:
        idx_scale = "none"
    elif index_dtype != idx_c or not idx_c or not target_cache_rule.index_scale_dtype:
        idx_scale = "fp8_e8m0" if "fp8" in index_dtype else "fp16"

    if default_rule is not None:
        wb = (
            (int(default_rule.group_size[0]), int(default_rule.group_size[1]))
            if len(default_rule.group_size) == 2
            else (1, int(default_rule.group_size[0]) if default_rule.group_size else 1)
        )
        eb = int(expert_rule.group_size[-1]) if (expert_rule is not None and expert_rule.group_size) else int(wb[1])
        w_fmt = default_rule.weight_dtype or native
        a_fmt = default_rule.activation_dtype or native
        e_fmt = (expert_rule.weight_dtype if expert_rule is not None else w_fmt) or native
        s_fmt = default_rule.scale_dtype or ("fp8_e8m0" if w_fmt == "mxfp4" else "fp32")
        s_bytes = float(dtype_bytes("fp8_e4m3" if s_fmt == "fp8_e8m0" else (s_fmt if s_fmt != "none" else "fp32")))
        act_is_low = a_fmt in _LOW_PRECISION_ACTIVATIONS
        act_group = int(wb[1]) if (act_is_low and w_fmt in {"fp8_e4m3", "fp8_e5m2", "mxfp4"}) else 0
        act_scale = s_fmt if act_is_low else ""
        act_pack = MXFP_SCALE_PACK if (act_is_low and s_fmt == "fp8_e8m0") else 1
        w_sym = True if default_rule.symmetric is None else bool(default_rule.symmetric)
    else:
        wb = (0, 0)
        eb = int(expert_rule.group_size[-1]) if (expert_rule is not None and expert_rule.group_size) else MXFP_GROUP_SIZE
        w_fmt = native
        a_fmt = native
        e_fmt = (expert_rule.weight_dtype if expert_rule is not None else native) or native
        s_fmt = "none"
        s_bytes = 0.0
        act_group = 0
        act_scale = ""
        act_pack = 1
        w_sym = True

    eff_expert_hint = (
        _extract_expert_count_hint(base_config)
        if expert_count_hint is None else int(expert_count_hint)
    )
    overrides, layer_overrides, extra_exclusions = _hf_layer_and_category_overrides(
        qmap, native=native, expert_count_hint=eff_expert_hint,
    )
    if expert_rule is not None and "routed_expert" not in overrides:
        exp_w = str(expert_rule.weight_dtype or "").strip().lower()
        if exp_w in _FLOAT_DTYPE_TOKENS:
            overrides["routed_expert"] = {
                "weight_format": expert_rule.weight_dtype,
                "activation_format": expert_rule.activation_dtype or expert_rule.weight_dtype,
            }
        else:
            overrides["routed_expert"] = _override_fields_for_hf_rule(expert_rule, native)

    exclusions = list(STANDARD_UNQUANTIZED_EXCLUSIONS) + extra_exclusions
    if "lm_head" in overrides:
        exclusions = [e for e in exclusions if e != "lm_head"]
    source_meta = {
        "quant_method": qmap.model_quant_type.lower(),
        "checkpoint": qmap.checkpoint_id,
        "ignored_modules": str(len(qmap.ignored_modules)),
        **({"index_dtype_source": "unspecified"} if (not index_override and not idx_c) else {}),
        **({"checkpoint_url": checkpoint_url} if checkpoint_url else {}),
    }
    return QuantSpec(
        name=qmap.model_quant_type or (default_rule.raw_tag if default_rule is not None else native),
        weight_format=w_fmt,
        activation_format=a_fmt,
        expert_weight_format=e_fmt,
        weight_block=wb,
        expert_block=eb,
        scale_format=s_fmt,
        scale_bytes=s_bytes,
        kv_dtype=kv_dtype,
        index_dtype=index_dtype,
        index_scale_dtype=idx_scale,
        exclusions=tuple(dict.fromkeys(exclusions)),
        activation_group_size=act_group,
        activation_scale_format=act_scale,
        activation_scale_pack=act_pack,
        category_overrides=overrides,
        layer_quant_overrides=layer_overrides,
        native_format=native,
        weight_symmetric=w_sym,
        source=merge_source(source_meta),
        notes=(f"parsed from hf quantization_config ({qmap.checkpoint_id}): {qmap.model_quant_type}",),
    )


def _layer_arch_sets(qmap: CheckpointQuantMap) -> tuple[frozenset[int], frozenset[int]]:
    mla_layers: set[int] = set()
    kda_layers: set[int] = set()
    for rule in qmap.rules:
        if rule.presence is None or rule.presence.scope != "target" or not rule.presence.layer_indices:
            continue
        parts = set(str(rule.pattern).lower().split("."))
        if parts & _MLA_EXACT_COMPONENTS:
            mla_layers.update(rule.presence.layer_indices)
        if parts & _KDA_EXACT_COMPONENTS:
            kda_layers.update(rule.presence.layer_indices)
    return frozenset(mla_layers), frozenset(kda_layers)


def _expert_counts_by_pattern_layer(
    qmap: CheckpointQuantMap,
) -> tuple[dict[tuple[ModelScope, str, int], int], dict[tuple[ModelScope, str], int]]:
    per_layer: dict[tuple[ModelScope, str, int], int] = {}
    fallback: dict[tuple[ModelScope, str], int] = {}
    for rule in qmap.rules:
        pres = rule.presence
        if pres is None or not pres.expert_indices:
            continue
        cnt = max(len(pres.expert_indices), max(pres.expert_indices) + 1)
        pkey = (pres.scope, rule.pattern)
        fallback[pkey] = max(fallback.get(pkey, 0), cnt)
        for lid in pres.layer_indices:
            lkey = (pres.scope, rule.pattern, int(lid))
            per_layer[lkey] = max(per_layer.get(lkey, 0), cnt)
    return per_layer, fallback


def _freeze_fields_key(fields: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(fields.items()))


def _attach_partial_expert_overrides(
    native_layers: dict[str, dict[str, Any]],
    partial_by_layer_cat: Mapping[tuple[int, str], Mapping[int, tuple[tuple[str, Any], ...]]],
    expert_count_by_layer_cat: Mapping[tuple[int, str], int],
) -> None:
    for (layer_id, cat), exp_map in sorted(partial_by_layer_cat.items()):
        if not exp_map:
            continue
        grouped: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
        for eid, fkey in sorted(exp_map.items()):
            grouped[fkey].append(eid)
        entries = [
            {"expert_indices": sorted(eids), "fields": dict(fkey)}
            for fkey, eids in sorted(grouped.items(), key=lambda kv: kv[1][0])
        ]
        count = max(expert_count_by_layer_cat.get((layer_id, cat), 0), max(exp_map) + 1)
        layer_entry = native_layers.setdefault(str(int(layer_id)), {"category_overrides": {}})
        cat_entry = dict(layer_entry["category_overrides"].get(cat) or {})
        cat_entry["expert_quant_count"] = int(count)
        cat_entry["expert_quant_overrides"] = entries
        layer_entry["category_overrides"][cat] = cat_entry


def _layer_quant_overrides_from_exceptions(
    qmap: CheckpointQuantMap,
    *,
    has_mla: bool,
    has_kda: bool,
    native: str,
    mla_layers: frozenset[int] = frozenset(),
    kda_layers: frozenset[int] = frozenset(),
    category_defaults: Mapping[str, str] | None = None,
    keep_dense_mlp_children: bool = False,
) -> dict[str, Any]:
    """Project target-scope per-layer and partial-expert exceptions onto ``native_layers``."""
    per_layer_arch = bool(mla_layers or kda_layers)
    per_layer_counts, fallback_counts = _expert_counts_by_pattern_layer(qmap)
    native_layers: dict[str, dict[str, Any]] = {}
    partial_by_layer_cat: dict[tuple[int, str], dict[int, tuple[tuple[str, Any], ...]]] = defaultdict(dict)
    expert_count_by_layer_cat: dict[tuple[int, str], int] = {}

    if category_defaults:
        full_exc_lookup = {
            (exc.scope, exc.pattern, int(lid)): exc.raw_tag
            for exc in qmap.layer_exceptions
            if exc.scope == "target" and not exc.expert_indices
            for lid in exc.layer_indices
        }
        seen_coords: set[tuple[str, str, int]] = set()
        layer_cat_tags: dict[tuple[int, str], list[str]] = defaultdict(list)
        cats_memo: dict[tuple[str, bool, bool], tuple[str, ...]] = {}
        for rule in qmap.rules:
            pres = rule.presence
            if pres is None or pres.scope != "target" or not pres.layer_indices or rule.pattern.endswith(".quant_type"):
                continue
            pat = rule.pattern
            for layer_id in pres.layer_indices:
                coord = (pres.scope, pat, int(layer_id))
                seen_coords.add(coord)
                eff_tag = full_exc_lookup.get(coord, rule.raw_tag)
                l_mla = (layer_id in mla_layers) if per_layer_arch else has_mla
                l_kda = (layer_id in kda_layers) if per_layer_arch else has_kda
                ckey = (pat, l_mla, l_kda)
                cats_t = cats_memo.get(ckey)
                if cats_t is None:
                    cats_t = tuple(
                        cat for cat in classify_tensor_categories(pat, has_mla=l_mla, has_kda=l_kda)
                        if cat not in _SKIP_OVERRIDE_CATS
                    )
                    cats_memo[ckey] = cats_t
                for cat in cats_t:
                    layer_cat_tags[(int(layer_id), cat)].append(eff_tag)
        for coord, exc_tag in full_exc_lookup.items():
            if coord in seen_coords:
                continue
            _, pat, layer_id = coord
            l_mla = (layer_id in mla_layers) if per_layer_arch else has_mla
            l_kda = (layer_id in kda_layers) if per_layer_arch else has_kda
            ckey = (pat, l_mla, l_kda)
            cats_t = cats_memo.get(ckey)
            if cats_t is None:
                cats_t = tuple(
                    cat for cat in classify_tensor_categories(pat, has_mla=l_mla, has_kda=l_kda)
                    if cat not in _SKIP_OVERRIDE_CATS
                )
                cats_memo[ckey] = cats_t
            for cat in cats_t:
                layer_cat_tags[(int(layer_id), cat)].append(exc_tag)
        for (layer_id, cat), tags in sorted(layer_cat_tags.items()):
            layer_tag = _pick_category_majority_tag(tags)
            if category_defaults.get(cat) not in (None, layer_tag):
                entry = native_layers.setdefault(str(int(layer_id)), {"category_overrides": {}})
                entry["category_overrides"][cat] = _override_fields_for_tag(layer_tag, native)

    if not qmap.layer_exceptions and not native_layers:
        return {}

    for exc in qmap.layer_exceptions:
        if exc.scope != "target" or not exc.layer_indices:
            continue
        if not exc.expert_indices and category_defaults:
            continue
        fields = _override_fields_for_tag(exc.raw_tag, native)
        fkey = _freeze_fields_key(fields)
        for layer_id in exc.layer_indices:
            l_mla = (layer_id in mla_layers) if per_layer_arch else has_mla
            l_kda = (layer_id in kda_layers) if per_layer_arch else has_kda
            cats = [
                cat for cat in classify_tensor_categories(exc.pattern, has_mla=l_mla, has_kda=l_kda)
                if cat not in _SKIP_OVERRIDE_CATS
            ]
            if not cats:
                continue
            if exc.expert_indices:
                pat_exp_count = per_layer_counts.get(
                    (exc.scope, exc.pattern, int(layer_id)),
                    fallback_counts.get((exc.scope, exc.pattern), 0),
                )
                for cat in cats:
                    key = (int(layer_id), cat)
                    for eid in exc.expert_indices:
                        partial_by_layer_cat[key][int(eid)] = fkey
                    needed = max(pat_exp_count, max(exc.expert_indices) + 1)
                    expert_count_by_layer_cat[key] = max(expert_count_by_layer_cat.get(key, 0), needed)
                continue
            entry = native_layers.setdefault(str(int(layer_id)), {"category_overrides": {}})
            cat_map = entry["category_overrides"]
            for cat in cats:
                cat_map[cat] = dict(fields)

    _attach_partial_expert_overrides(native_layers, partial_by_layer_cat, expert_count_by_layer_cat)
    _coalesce_dense_mlp_layer_overrides(native_layers, keep_children=keep_dense_mlp_children)
    return {"native_layers": native_layers} if native_layers else {}


def _pick_category_majority_tag(tags: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for tag in tags:
        norm = normalize_quant_tag_strict(tag)
        counts[norm] = counts.get(norm, 0) + 1
    if not counts:
        return "FLOAT"
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0] == "FLOAT", kv[0]))[0]


def _collect_modelslim_category_tags(
    qmap: CheckpointQuantMap,
    *,
    has_mla: bool,
    has_kda: bool,
    mla_layers: frozenset[int],
    kda_layers: frozenset[int],
) -> dict[str, list[str]]:
    full_exc_lookup: dict[tuple[ModelScope, str, int], str] = {}
    for exc in qmap.layer_exceptions:
        if exc.expert_indices:
            continue
        for lid in exc.layer_indices:
            full_exc_lookup[(exc.scope, exc.pattern, int(lid))] = exc.raw_tag

    per_layer_arch = bool(mla_layers or kda_layers)
    cat_tags: dict[str, list[str]] = defaultdict(list)
    cats_memo: dict[tuple[str, bool, bool], tuple[str, ...]] = {}
    for rule in qmap.rules:
        if rule.pattern.endswith(".quant_type"):
            continue
        pat = rule.pattern
        pres = rule.presence
        if pres is not None and pres.scope == "target" and pres.layer_indices:
            for lid in pres.layer_indices:
                eff_tag = full_exc_lookup.get((pres.scope, pat, int(lid)), rule.raw_tag)
                l_mla = (lid in mla_layers) if per_layer_arch else has_mla
                l_kda = (lid in kda_layers) if per_layer_arch else has_kda
                ckey = (pat, l_mla, l_kda)
                cats_t = cats_memo.get(ckey)
                if cats_t is None:
                    cats_t = classify_tensor_categories(pat, has_mla=l_mla, has_kda=l_kda)
                    cats_memo[ckey] = cats_t
                for cat in cats_t:
                    cat_tags[cat].append(eff_tag)
            continue
        ckey = (pat, has_mla, has_kda)
        cats_t = cats_memo.get(ckey)
        if cats_t is None:
            cats_t = classify_tensor_categories(pat, has_mla=has_mla, has_kda=has_kda)
            cats_memo[ckey] = cats_t
        for cat in cats_t:
            cat_tags[cat].append(rule.raw_tag)
    return cat_tags


@lru_cache(maxsize=QUANT_MAP_CACHE_MAXSIZE)
def _quant_spec_from_checkpoint_map_cached(
    qmap: CheckpointQuantMap,
    native: str,
    expert_count_hint: int,
    kv_dtype_override: str,
    index_dtype_override: str,
    checkpoint_url: str,
) -> QuantSpec:
    """Bounded value-keyed projection over a frozen ``CheckpointQuantMap``.

    [Cohesive-Pipeline] Single-pass projection over checkpoint rules, layer exceptions, and cache tags.
    """
    if qmap.source_format == "unquantized_native":
        return QuantSpec(
            name=native, weight_format=native, activation_format=native,
            expert_weight_format=native, kv_dtype=kv_dtype_override or native,
            index_dtype=index_dtype_override or native, index_scale_dtype="none",
            exclusions=(), native_format=native,
            source=merge_source({
                "quantization": "none",
                "checkpoint": qmap.checkpoint_id,
                **({"index_dtype_source": "unspecified"} if not index_dtype_override else {}),
            }),
        )
    if qmap.source_format == "hf_quantization_config":
        return _quant_spec_from_hf_map(
            qmap, native=native, kv_override=kv_dtype_override,
            index_override=index_dtype_override, checkpoint_url=checkpoint_url,
            expert_count_hint=expert_count_hint,
        )

    has_mla, has_kda, _ = _arch_flags_for_checkpoint_map(qmap)
    mla_layers, kda_layers = _layer_arch_sets(qmap)
    cat_tags = _collect_modelslim_category_tags(
        qmap, has_mla=has_mla, has_kda=has_kda, mla_layers=mla_layers, kda_layers=kda_layers,
    )

    linear_pool = [
        t
        for c in (
            "attn.qkv", "attn.o", "mla.q_a", "mla.q_b", "mla.kv_a", "mla.o",
            "kda.q", "kda.k", "kda.v",
            "dense_mlp", *_DENSE_MLP_SUB_CATS,
            "shared_expert", "latent_moe", "indexer_fp8",
        )
        for t in cat_tags.get(c, ())
    ]
    expert_pool = cat_tags.get("routed_expert", [])
    linear_tag = _pick_dominant_tag(linear_pool, _pick_dominant_tag(expert_pool, "FLOAT"))
    expert_tag = _pick_dominant_tag(expert_pool, linear_tag)
    lin_spec, exp_spec = TAG_SPEC_TABLE[linear_tag], TAG_SPEC_TABLE[expert_tag]

    exclusions = list(STANDARD_UNQUANTIZED_EXCLUSIONS)
    if cat_tags.get("attn_res_proj") and any(
        normalize_quant_tag_strict(t) != "FLOAT" for t in cat_tags["attn_res_proj"]
    ):
        exclusions = [e for e in exclusions if e != "attn_res_proj"]
    if cat_tags.get("lm_head") and _pick_category_majority_tag(cat_tags["lm_head"]) != "FLOAT":
        exclusions = [e for e in exclusions if e != "lm_head"]
    for opt in ("mtp", "gdn", "shared_expert", "dense_mlp", "ple", "ple_proj", "ple_table"):
        if cat_tags.get(opt) and all(normalize_quant_tag_strict(t) == "FLOAT" for t in cat_tags[opt]):
            if opt not in exclusions:
                exclusions.append(opt)
    dense_all_tags = [t for c in _DENSE_MLP_SUB_CATS for t in cat_tags.get(c, ())] + cat_tags.get("dense_mlp", [])
    if dense_all_tags and all(normalize_quant_tag_strict(t) == "FLOAT" for t in dense_all_tags):
        if "dense_mlp" not in exclusions:
            exclusions.append("dense_mlp")
    attn_tags = cat_tags.get("attn.qkv", []) + cat_tags.get("attn.o", [])
    if attn_tags and all(normalize_quant_tag_strict(t) == "FLOAT" for t in attn_tags) and "attn" not in exclusions:
        exclusions.append("attn")

    overrides: dict[str, dict[str, Any]] = {}
    category_defaults: dict[str, str] = {}
    dense_sub_targets = {
        sub: _pick_category_majority_tag(cat_tags[sub])
        for sub in _DENSE_MLP_SUB_CATS
        if cat_tags.get(sub)
    }
    if dense_sub_targets and "dense_mlp" not in exclusions:
        parent_dense_tag = _pick_category_majority_tag(list(dense_sub_targets.values()))
        category_defaults["dense_mlp"] = parent_dense_tag
        if parent_dense_tag != linear_tag:
            overrides["dense_mlp"] = _override_fields_for_tag(parent_dense_tag, native)
        for sub, sub_tag in dense_sub_targets.items():
            category_defaults[sub] = sub_tag
            if all(normalize_quant_tag_strict(t) == "FLOAT" for t in cat_tags[sub]) and sub not in exclusions:
                exclusions.append(sub)
            if sub_tag != parent_dense_tag:
                overrides[sub] = _override_fields_for_tag(sub_tag, native)
    skip = set(exclusions) | (set(_SKIP_OVERRIDE_CATS) - {"lm_head"}) | set(dense_sub_targets)
    for cat, tags_here in sorted(cat_tags.items()):
        if cat in skip:
            continue
        target = _pick_category_majority_tag(tags_here)
        category_defaults[cat] = target
        needs_override = (
            (cat == "routed_expert" and (target != linear_tag or exp_spec["weight_format"] in {"mxfp4", "int4"}))
            or target != linear_tag
            or cat in {"lm_head", "indexer.wk", "indexer.weights", "indexer.wq", "indexer_fp8"}
        )
        if needs_override:
            overrides[cat] = _override_fields_for_tag(target, native)

    target_cache_rule = qmap.cache_rule if qmap.cache_rule.scope == "target" else CacheQuantRule()
    kv_c, idx_c, idx_scale = _cache_dtypes_from_rule(target_cache_rule)
    kv_dtype = kv_dtype_override or kv_c or native
    index_from_kv = (not index_dtype_override and not idx_c and kv_dtype in {"int8", "fp8", "fp8_e4m3"})
    index_unspecified = (not index_dtype_override and not idx_c and not index_from_kv)
    index_dtype = index_dtype_override or idx_c or (kv_dtype if index_from_kv else native)
    if index_dtype not in {"int8", "fp8", "fp8_e4m3"}:
        idx_scale = "none"
    elif index_dtype != idx_c or not idx_c or not target_cache_rule.index_scale_dtype:
        idx_scale = "fp8_e8m0" if index_dtype in {"fp8", "fp8_e4m3"} else "fp16"

    kv_tag_upper = target_cache_rule.kv_tag.strip().upper().replace("-", "_")
    scoped_kv_tags = {
        str(tag).strip().upper().replace("-", "_")
        for _, tag in qmap.cache_rule.layer_scoped_kv
    }
    homogeneous_mxfp8_kv = (
        kv_tag_upper == _MXFP8_KV_PACKET_TAG
        and kv_dtype in {"fp8", "fp8_e4m3"}
        and (not scoped_kv_tags or scoped_kv_tags == {_MXFP8_KV_PACKET_TAG})
    )
    kv_cache_tag = _MXFP8_KV_PACKET_TAG if homogeneous_mxfp8_kv else ""
    name = f"{linear_tag}+{expert_tag}" if (expert_tag != linear_tag and expert_tag != "FLOAT") else linear_tag
    source_meta = {
        "quant_method": name.lower(),
        "linear_tag": linear_tag,
        "expert_tag": expert_tag,
        "pattern_entries": str(len(qmap.rules)),
        "layer_exceptions": str(len(qmap.layer_exceptions)),
        **({"kv_cache_tag": kv_cache_tag} if kv_cache_tag else {}),
        **({"index_dtype_source": "unspecified"} if index_unspecified else {}),
        **({"checkpoint": qmap.checkpoint_id} if qmap.checkpoint_id else {}),
        **({"checkpoint_url": checkpoint_url} if checkpoint_url else {}),
    }
    layer_overrides = _layer_quant_overrides_from_exceptions(
        qmap, has_mla=has_mla, has_kda=has_kda, native=native,
        mla_layers=mla_layers, kda_layers=kda_layers,
        category_defaults=category_defaults,
        keep_dense_mlp_children=any(sub in overrides for sub in _DENSE_MLP_SUB_CATS),
    )
    if qmap.ignored_modules:
        ign_native = dict(layer_overrides.get("native_layers") or {})
        ign_partial: dict[tuple[int, str], dict[int, tuple[tuple[str, Any], ...]]] = defaultdict(dict)
        ign_exp_cnt: dict[tuple[int, str], int] = {}
        ign_exclusions: list[str] = []
        _apply_ignored_modules_to_overrides(
            qmap.ignored_modules,
            has_mla=has_mla,
            has_kda=has_kda,
            native=native,
            cat_overrides=overrides,
            native_layers=ign_native,
            extra_exclusions=ign_exclusions,
            partial_by_layer_cat=ign_partial,
            expert_count_by_layer_cat=ign_exp_cnt,
            expert_count_hint=expert_count_hint,
        )
        _attach_partial_expert_overrides(ign_native, ign_partial, ign_exp_cnt)
        for ex in ign_exclusions:
            if ex not in exclusions:
                exclusions.append(ex)
        if ign_native:
            layer_overrides["native_layers"] = ign_native
    if not kv_dtype_override and target_cache_rule.layer_scoped_kv:
        for lid, ltag in target_cache_rule.layer_scoped_kv:
            l_kv_c, _, _ = _cache_dtypes_from_rule(CacheQuantRule(kv_tag=ltag))
            eff_l_kv = l_kv_c or native
            if eff_l_kv != kv_dtype:
                nl = layer_overrides.setdefault("native_layers", {})
                nl.setdefault(str(int(lid)), {})["kv_dtype"] = eff_l_kv
    if not index_dtype_override and target_cache_rule.layer_scoped_index:
        for lid, ltag in target_cache_rule.layer_scoped_index:
            _, l_idx_c, l_idx_scale = _cache_dtypes_from_rule(CacheQuantRule(index_tag=ltag))
            eff_l_idx = l_idx_c or native
            if eff_l_idx != index_dtype:
                nl = layer_overrides.setdefault("native_layers", {})
                entry = nl.setdefault(str(int(lid)), {})
                entry["index_dtype"] = eff_l_idx
                entry["index_scale_dtype"] = l_idx_scale if eff_l_idx in {"int8", "fp8", "fp8_e4m3"} else "none"
    return QuantSpec(
        name=name,
        weight_format=lin_spec["weight_format"] if lin_spec["is_quantized"] else native,
        activation_format=lin_spec["activation_format"] if lin_spec["is_quantized"] else native,
        expert_weight_format=exp_spec["weight_format"] if exp_spec["is_quantized"] else native,
        weight_block=lin_spec["weight_block"],
        expert_block=exp_spec["expert_block"] or lin_spec["expert_block"] or MXFP_GROUP_SIZE,
        scale_format=lin_spec["scale_format"] if lin_spec["is_quantized"] else "fp32",
        scale_bytes=float(lin_spec["scale_bytes"] if lin_spec["is_quantized"] else FP32_SCALE_BYTES),
        kv_dtype=kv_dtype,
        index_dtype=index_dtype,
        index_scale_dtype=idx_scale,
        exclusions=tuple(exclusions),
        activation_group_size=int(lin_spec["activation_group_size"]),
        activation_scale_format=str(lin_spec["activation_scale_format"]),
        activation_scale_pack=int(lin_spec["activation_scale_pack"]),
        category_overrides=overrides,
        layer_quant_overrides=layer_overrides,
        native_format=native,
        runtime_weight_scale_format=str(lin_spec.get("runtime_weight_scale_format", "")),
        runtime_weight_scale_bytes=float(lin_spec.get("runtime_weight_scale_bytes", 0.0)),
        runtime_activation_scale_format=str(lin_spec.get("runtime_activation_scale_format", "")),
        runtime_int8_weight_metadata=str(lin_spec.get("runtime_int8_weight_metadata", "")),
        source=merge_source(source_meta),
        notes=(
            f"parsed from checkpoint map ({len(qmap.rules)} rules, "
            f"{len(qmap.layer_exceptions)} layer exceptions): "
            f"linear={linear_tag}, expert={expert_tag}, kv={kv_dtype}",
        ),
    )


def quant_spec_from_checkpoint_map(
    qmap: CheckpointQuantMap,
    *,
    base_config: Mapping[str, Any] | None = None,
    kv_dtype_override: str = "",
    index_dtype_override: str = "",
    checkpoint_url: str = "",
) -> QuantSpec:
    """Project a lossless ``CheckpointQuantMap`` into an effective ``QuantSpec``."""
    native = _extract_native_float_dtype(base_config) or qmap.native_float_dtype or "bf16"
    expert_count_hint = _extract_expert_count_hint(base_config) if base_config is not None else 0
    cache_key = (
        id(qmap),
        native,
        expert_count_hint,
        kv_dtype_override,
        index_dtype_override,
        checkpoint_url,
    )
    cached = _QUANT_SPEC_ID_CACHE.get(cache_key)
    if cached is not None and cached[0] is qmap:
        return cached[1]
    result = _quant_spec_from_checkpoint_map_cached(
        qmap,
        native,
        expert_count_hint,
        kv_dtype_override,
        index_dtype_override,
        checkpoint_url,
    )
    _QUANT_SPEC_ID_CACHE[cache_key] = (qmap, result)
    _QUANT_SPEC_ID_CACHE.move_to_end(cache_key)
    while len(_QUANT_SPEC_ID_CACHE) > QUANT_MAP_CACHE_MAXSIZE:
        _QUANT_SPEC_ID_CACHE.popitem(last=False)
    return result


__all__ = [
    "CURATED_CHECKPOINTS_SCHEMA",
    "DEFAULT_STREAM_CHUNK_BYTES",
    "KNOWN_MODELSLIM_CACHE_TAGS",
    "KNOWN_MODELSLIM_TAGS_UPPER",
    "KNOWN_MODELSLIM_WEIGHT_TAGS",
    "STANDARD_UNQUANTIZED_EXCLUSIONS",
    "TAG_ALIASES",
    "TAG_SPEC_TABLE",
    "classify_tensor_categories",
    "curated_weight_pattern_map",
    "decode_int_ranges",
    "deserialize_curated_entry",
    "encode_int_ranges",
    "infer_arch_flags_from_patterns",
    "iter_json_object_items",
    "list_curated_checkpoint_ids",
    "lookup_curated_checkpoint",
    "normalize_quant_tag_strict",
    "quant_spec_from_checkpoint_map",
    "read_checkpoint_quant_map",
    "read_hf_quantization_config",
    "read_modelslim_description_file",
    "read_modelslim_description_mapping",
    "resolve_checkpoint_tensor_tag",
    "serialize_curated_entry",
]
