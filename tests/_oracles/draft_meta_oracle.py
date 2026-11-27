"""Independent auxiliary, vision, and engram/PLE reference declarations.

[Role]
  Derive optional draft/MTP/DSpark auxiliary modes, standalone draft head
  tensors, multimodal vision encoder/projector entries, and deterministic
  engram/PLE lookup table shapes from raw HuggingFace config JSONs for
  ``catalog_meta_oracle``.

[Invariants]
  * Operates strictly on raw flattened config mappings and ``RefLayer`` records;
    zero ``mindie_sim`` runtime or catalog modules are imported.
  * Every emitted ``Entry`` carries positive ``(rows, cols, count)`` dimensions
    and explicit ``"source"`` or ``"runtime"`` ownership.
  * Deterministic 64-bit Miller-Rabin primality matches the pinned Qwen4Exp PLE
    head vocabulary prime sequence.

[Timeouts & Locks]
  Pure bounded integer arithmetic over declared layers and tables; no I/O,
  threads, subprocesses, or locks.

[Side-Effect Policy]
  Side-effect free beyond mutating caller-owned ``layers`` list when marking
  engram/PLE layers during ``_assemble_geometry``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from .cache_meta_oracle import (
    PLE_DEFAULT_CONV_KERNEL,
    PLE_DEFAULT_DIVISOR,
    RefLayer,
    _i,
)

RGB_CHANNELS = 3
DEFAULT_SPATIAL_MERGE = 2
KIMI_DEFAULT_DSPARK_LAYERS = 5
KIMI_DEFAULT_TARGET_LAYERS = (2, 23, 47, 71, 89)
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _vision_decl(v: Mapping[str, Any], *, layer_hidden: int) -> dict[str, Any]:
    model_type = str(v.get("model_type", ""))
    if "vt_hidden_size" in v:
        merge = v.get("merge_kernel_size") or [DEFAULT_SPATIAL_MERGE, DEFAULT_SPATIAL_MERGE]
        qkv_h = _i(v, "qkv_hidden_size")
        return {
            "kind": "kimi_k3",
            "hidden": _i(v, "vt_hidden_size"),
            "layers": _i(v, "vt_num_hidden_layers"),
            "intermediate": _i(v, "vt_intermediate_size"),
            "query": qkv_h,
            "qkv": RGB_CHANNELS * qkv_h,
            "patch_width": RGB_CHANNELS * _i(v, "patch_size") ** 2,
            "pos": _i(v, "init_pos_emb_height") * _i(v, "init_pos_emb_width"),
            "bias": False,
            "gated": False,
            "norm_type": "rmsnorm",
            "aligner_in": _i(v, "vt_hidden_size") * int(merge[0]) * int(merge[1]),
            "aligner_hidden": _i(v, "vt_intermediate_size"),
            "out_hidden": _i(v, "text_hidden_size", layer_hidden),
        }
    if "deepseek_v41" in model_type:
        downsample = _i(v, "downsample_ratio", 1) or 1
        hidden = _i(v, "hidden_size")
        return {
            "kind": "deepseek_v41",
            "hidden": hidden,
            "layers": _i(v, "num_hidden_layers"),
            "intermediate": _i(v, "intermediate_size"),
            "query": hidden,
            "qkv": RGB_CHANNELS * hidden,
            "patch_width": RGB_CHANNELS * _i(v, "patch_size") ** 2,
            "pos": 0,
            "bias": True,
            "gated": True,
            "norm_type": "layernorm",
            "aligner_in": hidden * downsample * downsample,
            "aligner_hidden": layer_hidden,
            "out_hidden": layer_hidden,
        }
    merge = _i(v, "spatial_merge_size", DEFAULT_SPATIAL_MERGE) or DEFAULT_SPATIAL_MERGE
    hidden = _i(v, "hidden_size")
    temporal = _i(v, "temporal_patch_size", 1) or 1
    in_channels = _i(v, "in_channels", _i(v, "in_chans", RGB_CHANNELS)) or RGB_CHANNELS
    return {
        "kind": "qwen3_5",
        "hidden": hidden,
        "layers": _i(v, "depth"),
        "intermediate": _i(v, "intermediate_size"),
        "query": hidden,
        "qkv": RGB_CHANNELS * hidden,
        "patch_width": in_channels * _i(v, "patch_size") ** 2 * temporal,
        "pos": _i(v, "num_position_embeddings"),
        "bias": True,
        "gated": False,
        "norm_type": "layernorm",
        "aligner_in": hidden * merge * merge,
        "aligner_hidden": 0,
        "out_hidden": _i(v, "out_hidden_size", layer_hidden),
    }


def _vision_entries(
    v: Mapping[str, Any], text_hidden: int, add_fn: Callable[..., None],
) -> list[Any]:
    out: list[Any] = []
    hidden, layers = int(v["hidden"]), int(v["layers"])
    inter, qkv = int(v["intermediate"]), int(v["qkv"])
    bias, norm_type = bool(v["bias"]), str(v["norm_type"])
    add_fn(out, "vision_patch", hidden, int(v["patch_width"]))
    add_fn(out, "vision_pos", max(1, int(v["pos"])), hidden)
    if bias:
        add_fn(out, "vision_patch_bias", hidden, 1)
    fc0_rows = (2 if v["gated"] else 1) * inter
    for name, rows, cols in (
        ("vision_qkv", qkv, hidden),
        ("vision_wo", hidden, int(v["query"])),
        ("vision_fc0", fc0_rows, hidden),
        ("vision_fc1", hidden, inter),
        ("vision_norm1", hidden, 1),
        ("vision_norm2", hidden, 1),
    ):
        add_fn(out, name, rows, cols, layers)
    if bias:
        for name, rows in (
            ("vision_qkv_bias", qkv),
            ("vision_wo_bias", hidden),
            ("vision_fc0_bias", fc0_rows),
            ("vision_fc1_bias", hidden),
        ):
            add_fn(out, name, rows, 1, layers)
    if norm_type == "layernorm":
        add_fn(out, "vision_norm1_bias", hidden, 1, layers)
        add_fn(out, "vision_norm2_bias", hidden, 1, layers)
    add_fn(out, "vision_final_norm", hidden, 1)
    if norm_type == "layernorm":
        add_fn(out, "vision_final_norm_bias", hidden, 1)
    aligner_in = int(v["aligner_in"])
    out_h = int(v["out_hidden"]) or text_hidden
    kind = str(v["kind"])
    if kind == "deepseek_v41":
        add_fn(out, "aligner_norm", aligner_in, 1)
        add_fn(out, "aligner_w1", out_h, aligner_in)
        add_fn(out, "aligner_w1_bias", out_h, 1)
        add_fn(out, "aligner_w2", out_h, out_h)
        add_fn(out, "aligner_w2_bias", out_h, 1)
    elif kind == "kimi_k3":
        add_fn(out, "projector_w0", int(v["aligner_hidden"]), aligner_in)
        add_fn(out, "projector_w2", out_h, int(v["aligner_hidden"]))
        add_fn(out, "projector_norm", out_h, 1)
    else:
        add_fn(out, "merger_w1", aligner_in, aligner_in)
        add_fn(out, "merger_w1_bias", aligner_in, 1)
        add_fn(out, "merger", out_h, aligner_in)
        add_fn(out, "merger_bias", out_h, 1)
    for name in ("image_start", "image_end", "image_newline"):
        add_fn(out, name, text_hidden, 1)
    return out


def _is_prime_u64(value: int) -> bool:
    if value < 2:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in _MILLER_RABIN_BASES:
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_u64(candidate):
            candidate += 2
        prime = candidate
    return prime


def _engram_decl(c: Mapping[str, Any], layers: list[RefLayer]) -> dict[str, Any] | None:
    ple_layers = [int(v) for v in c.get("ple_layer_ids", ()) or ()]
    hidden = _i(c, "hidden_size")
    hc = max((layer.hc_mult for layer in layers), default=1)
    if ple_layers:
        ngram, heads = _i(c, "ngram_size"), _i(c, "heads_per_ngram")
        embed = _i(c, "ple_embed_dim")
        ngram_heads = (ngram - 1) * heads
        width = embed // ngram_heads
        divisor = _i(c, "make_ngram_vocab_size_divisible_by", PLE_DEFAULT_DIVISOR)
        base = _i(c, "ngram_vocab_size_base")
        conv_k = _i(c, "ple_conv_kernel_size", PLE_DEFAULT_CONV_KERNEL)
        tables = []
        for index, ple_id in enumerate(ple_layers):
            total = 0
            for local_head in range(ngram_heads):
                global_head = index * ngram_heads + local_head
                total += _nth_prime_after(base - 1, global_head + 1)
            rows = ((total + divisor - 1) // divisor) * divisor
            tables.append({"id": index, "layer": ple_id - 1, "rows": rows, "width": width})
        for table in tables:
            layers[table["layer"]] = replace(layers[table["layer"]], engram=True)
        return {
            "kind": "ple", "tables": tables, "hidden": hidden, "hc_mult": hc,
            "embed": embed, "conv_k": conv_k, "max_ngram": ngram,
        }
    layer_ids = [int(v) for v in c.get("engram_layer_ids", ()) or ()]
    rows = [int(v) for v in c.get("engram_num_embeddings", ()) or ()]
    if not layer_ids:
        return None
    width, ngram = _i(c, "engram_head_dim"), _i(c, "engram_max_ngram_size")
    heads = _i(c, "engram_n_heads")
    tables = [
        {"id": index, "layer": layer, "rows": row, "width": width}
        for index, (layer, row) in enumerate(zip(layer_ids, rows))
    ]
    for table in tables:
        layers[table["layer"]] = replace(layers[table["layer"]], engram=True)
    return {
        "kind": "engram", "tables": tables, "hidden": hidden, "hc_mult": hc,
        "width": width, "columns": (ngram - 1) * heads,
    }


def _engram_entries(e: Mapping[str, Any], add_fn: Callable[..., None]) -> list[Any]:
    out: list[Any] = []
    if e.get("kind") == "ple":
        hc, h = int(e["hc_mult"]), int(e["hidden"])
        each = hc * h
        for table in e["tables"]:
            add_fn(out, f"ple_table_{table['id']}", table["rows"], table["width"])
        for _ in e["tables"]:
            add_fn(out, "ple_kv_proj", each + h, int(e["embed"]))
            add_fn(out, "ple_norm_key", each, 1)
            add_fn(out, "ple_norm_query", each, 1)
            add_fn(out, "ple_norm_conv", each, 1)
            add_fn(out, "ple_conv1d", each, int(e["conv_k"]))
        return out
    hc, h = int(e["hc_mult"]), int(e["hidden"])
    for table in e["tables"]:
        add_fn(out, f"engram_table_{table['id']}", table["rows"], table["width"])
    for _ in e["tables"]:
        add_fn(out, "engram_gate_wkv", (hc + 1) * h, int(e["columns"]) * int(e["width"]))
        add_fn(out, "engram_q", hc, h)
        add_fn(out, "engram_k", hc, h)
    return out


def _standalone_head_entries(
    model: Any, add_fn: Callable[..., None],
) -> list[Any]:
    draft, h, out = model.geometry.get("draft") or {}, model.hidden, []
    add_fn(out, "draft_input_proj", h, int(draft["input_width"]))
    add_fn(out, "draft_input_norm", h, 1)
    add_fn(out, "draft_final_norm", h, 1)
    markov = int(draft.get("markov_rank", 0))
    if markov:
        add_fn(out, "markov_embed", model.vocab, markov)
        add_fn(out, "markov_head", model.vocab, markov)
    return out


def _aux_decl(c: Mapping[str, Any], builder: str) -> dict[str, Any]:
    targets = tuple(int(v) for v in c.get("dspark_target_layer_ids", ()) or ())
    nextn = _i(c, "num_nextn_predict_layers")
    if builder in {"deepseek_v4", "deepseek_v41"}:
        layers = nextn if nextn > 0 else len(targets)
        if layers <= 0:
            return {}
        block = _i(c, "dspark_block_size", 1)
        markov = _i(c, "dspark_markov_rank")
        kind = "dspark" if (block > 1 or markov) else "mtp_dspark"
        return {
            "kind": kind, "layers": layers, "standalone": False,
            "target_layer_ids": list(targets), "block_size": block,
        }
    if builder == "kimi":
        targets = targets or KIMI_DEFAULT_TARGET_LAYERS
        return {
            "kind": "dspark", "layers": KIMI_DEFAULT_DSPARK_LAYERS,
            "standalone": False, "external": True,
            "target_layer_ids": list(targets),
            "block_size": _i(c, "dspark_block_size", KIMI_DEFAULT_DSPARK_LAYERS),
        }
    if builder == "kimi_dspark":
        return {
            "kind": "standalone_draft",
            "layers": _i(c, "num_hidden_layers"),
            "standalone": True,
        }
    if builder == "glm" and nextn > 0:
        return {"kind": "mtp", "layers": nextn, "standalone": False}
    if builder == "qwen":
        mtp = _i(c, "mtp_num_hidden_layers") or nextn
        if mtp > 0:
            return {"kind": "mtp", "layers": mtp, "standalone": False}
    if builder == "minimax_m3":
        dspark_cfg = c.get("dspark_config") if isinstance(c.get("dspark_config"), Mapping) else {}
        cfg_layers = _i(dspark_cfg, "num_hidden_layers") or _i(c, "dspark_num_hidden_layers")
        modules = cfg_layers or _i(c, "num_mtp_modules") or nextn
        if modules > 0:
            markov = (
                _i(dspark_cfg, "dspark_markov_rank")
                or _i(dspark_cfg, "markov_rank")
                or _i(c, "dspark_markov_rank")
            )
            return {"kind": "dspark" if markov > 0 else "mtp", "layers": modules, "standalone": False}
    if builder == "step5" and nextn > 0:
        return {"kind": "mtp", "layers": nextn, "standalone": False}
    return {}


__all__ = [
    "_aux_decl",
    "_engram_decl",
    "_engram_entries",
    "_standalone_head_entries",
    "_vision_decl",
    "_vision_entries",
]
