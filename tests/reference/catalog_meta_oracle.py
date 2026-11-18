"""Independent meta-device parameter oracle for the 37-model catalog.

[Role]
  Re-derive every text-backbone parameter shape from the raw model JSON alone,
  register the shapes as real ``torch.device("meta")`` parameters and compare
  that reference ledger with the simulator's declared terms.  Architecture
  formulas are ported from the pinned reference constructors (vLLM
  b65a4b4d8a25b64115135baea63373573385a8de, Transformers
  ea9eb5de74c7d9ff92d2ea2e50c0840b1453ffb2, Step
  364b3b03fed2477aa0953fe855a2c4c8c48ee3cc); no simulator module is imported.

[Invariants]
  * The only inputs are the raw config JSON and the declared manifest.
  * ``source_owned`` tensors are counted once per logical occurrence; aliases
    sharing one meta tensor are de-duplicated by ``id(tensor)``.
  * Missing/extra names, row/column/count mismatches and ownership mismatches
    are reported per model, never folded into a single total.
  * ``torch`` is imported lazily so the arithmetic ledger still works in
    environments without it; meta construction is the primary path.

[Timeouts & Locks]
  Pure integer arithmetic bounded by layers/experts; no I/O beyond reading the
  requested config and manifest files.

[Side-Effect Policy]
  Read-only; no device context, weight load, network or global mutation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataclasses import replace
from .cache_meta_oracle import (
    PLE_DEFAULT_CONV_KERNEL, PLE_DEFAULT_DIVISOR, REFERENCE_PINS, RefLayer, _i,
    build_layers, cache_layout_declaration, cache_totals, flatten_config,
)

MANIFEST_SCHEMA = "mindie-catalog-reference-manifest/1"
SCOPE_TEXT_BACKBONE = "text_backbone"
ONE, TWO = 1, 2
THREE, FOUR = 3, 4
ZERO = 0
MHC_HEAD_ROWS = 4
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def make_meta_parameters(shapes, aliases):
    import torch
    with torch.device("meta"):
        parameters = {name: torch.nn.Parameter(torch.empty(tuple(shape)), requires_grad=False)
                      for name, shape in shapes.items()}
    for alias, owner in aliases.items():
        parameters[alias] = parameters[owner]
    return parameters


def logical_numel(parameters, entries):
    """Sum source-owned logical elements, de-duplicating shared tensors by id."""
    seen = set()
    total = ZERO
    for name, count in entries:
        tensor = parameters[name]
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += int(tensor.numel()) * int(count)
    return int(total)


def _torch_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def config_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_models(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"reference_manifest_schema_unsupported:{manifest.get('schema')}")
    models = manifest.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("reference_manifest_models_missing")
    return dict(models)


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"reference_manifest_missing:{target}")
    return json.loads(target.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Reference ledger records.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    name: str
    rows: int
    cols: int
    count: int = 1
    owner: str = "source"


@dataclass
class ReferenceModel:
    key: str
    config_path: str
    config_sha256: str
    builder: str
    family: str
    scope: str
    hidden: int
    vocab: int
    tie_word_embeddings: bool
    layers: list[RefLayer]
    geometry: dict[str, Any]
    source_pins: tuple[str, ...] = REFERENCE_PINS

    def entries(self) -> list[Entry]:
        out: list[Entry] = []
        for layer in self.layers:
            out.extend(_layer_entries(self, layer))
        out.extend(_head_entries(self))
        if self.geometry.get("vision") is not None:
            out.extend(_vision_entries(self.geometry["vision"], self.hidden))
        if self.geometry.get("engram") is not None:
            out.extend(_engram_entries(self.geometry["engram"]))
        if self.geometry.get("standalone"):
            out.extend(_standalone_head_entries(self))
        return out

    def source_total(self) -> int:
        return _total(self, "source")

    def runtime_total(self) -> int:
        return _total(self, "runtime")

    def meta_parameters(self):
        shapes, names = {}, []
        for index, entry in enumerate(self.entries()):
            key = f"{entry.name}[{index}]"
            shapes[key] = (int(entry.rows), int(entry.cols))
            names.append((key, int(entry.count), entry.owner))
        return make_meta_parameters(shapes, {}), names


def _total(reference: ReferenceModel, owner: str) -> int:
    if _torch_available():
        parameters, names = reference.meta_parameters()
        return logical_numel(
            parameters, [(name, count) for name, count, kind in names if kind == owner])
    return int(sum(e.rows * e.cols * e.count for e in reference.entries()
                   if e.owner == owner))


def _add(out: list[Entry], name: str, rows: int, cols: int, count: int = 1,
         owner: str = "source") -> None:
    if int(rows) <= 0 or int(cols) <= 0 or int(count) <= 0:
        raise ValueError(f"reference_shape_invalid:{name}:{rows}x{cols}x{count}")
    out.append(Entry(name, int(rows), int(cols), int(count), owner))


# --------------------------------------------------------------------------
# Per-layer matrix shapes (ported architecture constructors).
# --------------------------------------------------------------------------
def _norm_entries(out, model, layer) -> None:
    h = model.hidden
    if layer.hc_lowrank <= 0:
        _add(out, "input_norm", h, 1)
        _add(out, "post_attn_norm", h, 1)
    if layer.attention == "mla":
        if layer.q_lora:
            _add(out, "q_a_norm", layer.q_lora, 1)
        if layer.kv_lora:
            _add(out, "kv_a_norm", layer.kv_lora, 1)
    elif layer.attention in {"gqa", "projection_only"} and layer.qk_norm:
        _add(out, "q_norm", layer.head_dim, 1)
        _add(out, "k_norm", layer.head_dim, 1)
    elif layer.attention == "gdn":
        _add(out, "gdn_norm", layer.lin_dim, 1)
    if layer.gemma4:
        _add(out, "pre_ffn_norm", h, 1)
        _add(out, "post_ffn_norm", h, 1)
        if layer.ffn == "moe":
            _add(out, "pre_experts_norm", h, 1)
            _add(out, "post_experts_norm", h, 1)
            _add(out, "router_norm", h, 1)
            _add(out, "router_scale", h, 1)
            _add(out, "expert_scale", layer.experts, 1)
    if layer.residual:
        _add(out, "attn_res_norm", h, 1)
        _add(out, "attn_res_proj", 1, h)
        _add(out, "mlp_res_norm", h, 1)
        _add(out, "mlp_res_proj", 1, h)


def _mhc_entries(out, model, layer) -> None:
    if not (layer.mhc and layer.hc_mult > 0):
        return
    hc_dim = layer.hc_mult * model.hidden
    if layer.hc_lowrank > 0:
        for part in ("attn", "ffn"):
            _add(out, f"hc_{part}_norm", hc_dim, 1)
            _add(out, f"hc_{part}_down_inject", layer.hc_lowrank + layer.hc_mult, hc_dim)
            _add(out, f"hc_{part}_up", hc_dim, layer.hc_lowrank)
        return
    mix = (TWO + layer.hc_mult) * layer.hc_mult
    for part in ("attn", "ffn"):
        _add(out, f"hc_{part}_fn", mix, hc_dim)
        _add(out, f"hc_{part}_base", mix, 1)
        _add(out, f"hc_{part}_scale", THREE, 1)


def _attention_entries(out, model, layer) -> None:
    h = model.hidden
    if layer.attention == "mla":
        qk = layer.nope + layer.rope
        if layer.variant == "latent_mqa":
            if layer.q_lora:
                _add(out, "q_a", layer.q_lora, h)
                _add(out, "q_b", layer.heads * qk, layer.q_lora)
            else:
                _add(out, "q_proj", layer.heads * qk, h)
            _add(out, "wkv", layer.kv_lora, h)
            if layer.o_groups:
                width = layer.heads * layer.head_dim // layer.o_groups
                _add(out, "wo_a", layer.o_groups * layer.o_lora, width)
                _add(out, "wo_b", h, layer.o_groups * layer.o_lora)
            else:
                _add(out, "o_proj", h, layer.heads * layer.head_dim)
            return
        if layer.q_lora:
            _add(out, "q_a", layer.q_lora, h)
            _add(out, "q_b", layer.heads * qk, layer.q_lora)
        else:
            _add(out, "q_proj", layer.heads * qk, h)
        _add(out, "kv_a", layer.kv_lora + layer.rope, h)
        _add(out, "kv_b", layer.heads * (layer.nope + layer.v_dim), layer.kv_lora)
        if layer.mla_gate:
            _add(out, "mla_g", layer.heads * layer.v_dim, h)
        if layer.o_groups:
            width = layer.heads * layer.v_dim // layer.o_groups
            _add(out, "wo_a", layer.o_groups * layer.o_lora, width)
            _add(out, "wo_b", h, layer.o_groups * layer.o_lora)
        else:
            _add(out, "o_proj", h, layer.heads * layer.v_dim)
        return
    if layer.attention in {"gqa", "projection_only"}:
        q_width = layer.heads * layer.head_dim
        if layer.qgate and not layer.gate_separate:
            q_width *= TWO
        kv_width = layer.kv_heads * layer.head_dim
        _add(out, "q_proj", q_width, h)
        _add(out, "k_proj", kv_width, h)
        if not layer.k_eq_v:
            _add(out, "v_proj", kv_width, h)
        if layer.gate_separate:
            _add(out, "g_proj", layer.heads, h)
        _add(out, "o_proj", h, layer.heads * layer.head_dim)
        return
    if layer.attention in {"kda", "mamba", "gla", "deltanet", "lightning", "retnet"}:
        d = (layer.lin_value_heads or layer.lin_heads) * layer.lin_dim
        dk = layer.lin_key_dim or layer.lin_dim
        k_dim = (layer.lin_key_heads or layer.lin_heads) * dk
        decay = (layer.lin_value_heads or layer.lin_heads) * dk
        _add(out, "kda_q", k_dim, h)
        _add(out, "kda_k", k_dim, h)
        _add(out, "kda_v", d, h)
        _add(out, "kda_b", layer.lin_value_heads or layer.lin_heads, h)
        _add(out, "kda_f_a", dk, h)
        _add(out, "kda_f_b", decay, dk)
        _add(out, "kda_q_conv", k_dim, layer.conv_k)
        _add(out, "kda_k_conv", k_dim, layer.conv_k)
        _add(out, "kda_v_conv", d, layer.conv_k)
        _add(out, "kda_A_log", layer.lin_value_heads or layer.lin_heads, 1)
        _add(out, "kda_dt_bias", decay, 1)
        _add(out, "kda_o_norm", layer.lin_dim, 1)
        _add(out, "kda_o", h, d)
        if layer.gate_rank:
            _add(out, "kda_g_a", layer.gate_rank, h)
            _add(out, "kda_g_b", d, layer.gate_rank)
        else:
            _add(out, "kda_g", d, h)
        return
    if layer.attention == "gdn":
        key_w = layer.lin_key_heads * layer.lin_key_dim
        value_w = layer.lin_value_heads * layer.lin_dim
        conv_w = layer.conv_width or (TWO * key_w + value_w)
        _add(out, "gdn_qkvz", conv_w + value_w, h)
        _add(out, "gdn_ba", TWO * layer.lin_value_heads, h)
        _add(out, "gdn_conv", conv_w, layer.conv_k)
        _add(out, "gdn_o", h, value_w)
        _add(out, "gdn_A_log", layer.lin_value_heads, 1)
        _add(out, "gdn_dt_bias", layer.lin_value_heads, 1)
        _add(out, "gdn_norm", layer.lin_dim, 1)
        return
    raise ValueError(f"reference_attention_unsupported:{layer.attention}")


def _indexer_entries(out, model, layer) -> None:
    if not (layer.owns_index_k or layer.owns_topk or layer.indexer or layer.index_query):
        return
    if layer.index_heads <= 0 or layer.index_dim <= 0:
        raise ValueError(f"reference_indexer_dims_missing:{layer.index}")
    h = model.hidden
    if layer.attention == "gqa" and layer.index_kpool > 1:
        _add(out, "index_qk_proj", (layer.index_heads + 1) * layer.index_dim, h)
        _add(out, "indexer_q_norm", layer.index_dim, 1)
        _add(out, "indexer_k_norm", layer.index_dim, 1)
        return
    if layer.owns_index_k:
        width = layer.head_dim if layer.index_input == "compressed" else h
        if layer.index_input == "hidden" and layer.compress_ratio > 1:
            expanded = layer.compressor_width * layer.index_dim
            _add(out, "indexer_wk", expanded, width)
            _add(out, "indexer_gate", expanded, width)
            _add(out, "indexer_ape", layer.compress_ratio, expanded)
            _add(out, "indexer_k_norm", layer.index_dim, 1)
        else:
            _add(out, "indexer_wk", layer.index_dim, width)
            _add(out, "indexer_k_norm", layer.index_dim, 1)
            _add(out, "indexer_k_norm_bias", layer.index_dim, 1)
            if layer.index_kpool > 1:
                _add(out, "indexer_pool_gate", layer.index_dim, width)
                _add(out, "indexer_pool_ape", layer.index_kpool, layer.index_dim)
    if layer.indexer or layer.owns_topk or layer.index_query:
        _add(out, "indexer_wq", layer.index_heads * layer.index_dim, layer.q_lora or h)
        _add(out, "indexer_weights", layer.index_heads, h)
    if layer.index_kpool > 1:
        # Head weights keep the checkpoint's BF16 representation: no fabricated
        # FP32 runtime copy exists for ``indexer.weights``.  The retained
        # analytic K/gate joint projection still prepares its own FP32 copies.
        width = layer.head_dim if layer.index_input == "compressed" else h
        if layer.owns_index_k:
            _add(out, "indexer_wk_prepared_fp32", layer.index_dim, width, owner="runtime")
            _add(out, "indexer_pool_gate_prepared_fp32", layer.index_dim, width,
                 owner="runtime")


def _compressor_entries(out, model, layer) -> None:
    if not (layer.owns_kv and layer.compress_ratio):
        return
    d = layer.head_dim * layer.compressor_width
    _add(out, "compressor_wkv", d, model.hidden)
    if layer.compress_ratio > 1:
        # A C1 layer has no gated pooling: neither ``wgate`` nor APE exists.
        _add(out, "compressor_gate", d, model.hidden)
        _add(out, "compressor_ape", layer.compress_ratio, d)
    _add(out, "compressor_norm", layer.head_dim, 1)


def _ffn_entries(out, model, layer) -> None:
    h = model.hidden
    if layer.ffn == "dense":
        _add(out, "dense_gate", layer.dense_i, h)
        _add(out, "dense_up", layer.dense_i, h)
        _add(out, "dense_down", h, layer.dense_i)
        return
    if layer.experts <= 0 or layer.moe_i <= 0:
        raise ValueError(f"reference_moe_dims_missing:{layer.index}")
    expert_in = layer.latent or h
    expert_out = layer.latent or h
    _add(out, "router", layer.experts, h)
    _add(out, "expert_gate", layer.moe_i, expert_in, layer.experts)
    _add(out, "expert_up", layer.moe_i, expert_in, layer.experts)
    _add(out, "expert_down", expert_out, layer.moe_i, layer.experts)
    if layer.router_copy:
        _add(out, "router_precast", layer.experts, h, owner="runtime")
    if layer.router_bias:
        # noaux_tc: one FP32 correction bias per routed expert per MoE layer.
        _add(out, "router_correction_bias", layer.experts, 1)
    if layer.latent:
        _add(out, "latent_down", layer.latent, h)
        _add(out, "latent_norm", layer.latent, 1)
        _add(out, "latent_up", h, layer.latent)
    shared_i = layer.shared_i or layer.shared * layer.moe_i
    if shared_i > 0:
        _add(out, "shared_gate", shared_i, h)
        _add(out, "shared_up", shared_i, h)
        _add(out, "shared_down", h, shared_i)
    if layer.shared_gate:
        _add(out, "shared_router", 1, h)

def _layer_entries(model: ReferenceModel, layer: RefLayer) -> list[Entry]:
    out: list[Entry] = []
    _norm_entries(out, model, layer)
    _mhc_entries(out, model, layer)
    _attention_entries(out, model, layer)
    _indexer_entries(out, model, layer)
    _compressor_entries(out, model, layer)
    _ffn_entries(out, model, layer)
    return out


def _head_entries(model: ReferenceModel) -> list[Entry]:
    h, out = model.hidden, []
    lowrank = any(layer.hc_lowrank > 0 for layer in model.layers)
    if not lowrank:
        _add(out, "final_norm", h, 1)
    if not model.geometry.get("standalone"):
        _add(out, "embedding", model.vocab, h)
        if not model.tie_word_embeddings:
            _add(out, "lm_head", model.vocab, h)
    if any(layer.residual for layer in model.layers):
        _add(out, "output_attn_res_norm", h, 1)
        _add(out, "output_attn_res_proj", 1, h)
    if lowrank:
        hc = max(layer.hc_mult for layer in model.layers)
        lr = max(layer.hc_lowrank for layer in model.layers)
        hc_dim = hc * h
        _add(out, "hc_head_norm", hc_dim, 1)
        _add(out, "hc_head_down", lr, hc_dim)
        _add(out, "hc_head_up", hc_dim, lr)
    elif any(layer.mhc for layer in model.layers):
        _add(out, "hc_head_fn", MHC_HEAD_ROWS, MHC_HEAD_ROWS * h)
        _add(out, "hc_head_base", MHC_HEAD_ROWS, 1)
        _add(out, "hc_head_scale", 1, 1)
    return out

def _vision_entries(v: Mapping[str, Any], text_hidden: int) -> list[Entry]:
    out: list[Entry] = []
    hidden, layers = int(v["hidden"]), int(v["layers"])
    inter, qkv = int(v["intermediate"]), int(v["qkv"])
    bias, norm_type = bool(v["bias"]), str(v["norm_type"])
    _add(out, "vision_patch", hidden, int(v["patch_width"]))
    _add(out, "vision_pos", max(1, int(v["pos"])), hidden)
    if bias:
        _add(out, "vision_patch_bias", hidden, 1)
    per = [("vision_qkv", qkv, hidden), ("vision_wo", hidden, int(v["query"])),
           ("vision_fc0", (TWO if v["gated"] else ONE) * inter, hidden),
           ("vision_fc1", hidden, inter), ("vision_norm1", hidden, 1),
           ("vision_norm2", hidden, 1)]
    for name, rows, cols in per:
        _add(out, name, rows, cols, layers)
    if bias:
        for name, rows in (("vision_qkv_bias", qkv), ("vision_wo_bias", hidden),
                           ("vision_fc0_bias", (TWO if v["gated"] else ONE) * inter),
                           ("vision_fc1_bias", hidden)):
            _add(out, name, rows, 1, layers)
    if norm_type == "layernorm":
        _add(out, "vision_norm1_bias", hidden, 1, layers)
        _add(out, "vision_norm2_bias", hidden, 1, layers)
    _add(out, "vision_final_norm", hidden, 1)
    if norm_type == "layernorm":
        _add(out, "vision_final_norm_bias", hidden, 1)
    aligner_in = int(v["aligner_in"])
    out_h = int(v["out_hidden"]) or text_hidden
    kind = str(v["kind"])
    if kind == "deepseek_v41":
        _add(out, "aligner_norm", aligner_in, 1)
        _add(out, "aligner_w1", out_h, aligner_in)
        _add(out, "aligner_w1_bias", out_h, 1)
        _add(out, "aligner_w2", out_h, out_h)
        _add(out, "aligner_w2_bias", out_h, 1)
    elif kind == "kimi_k3":
        _add(out, "projector_w0", int(v["aligner_hidden"]), aligner_in)
        _add(out, "projector_w2", out_h, int(v["aligner_hidden"]))
        _add(out, "projector_norm", out_h, 1)
    else:
        _add(out, "merger_norm", aligner_in, 1)
        _add(out, "merger_norm_bias", aligner_in, 1)
        _add(out, "merger_w1", aligner_in, aligner_in)
        _add(out, "merger_w1_bias", aligner_in, 1)
        _add(out, "merger", out_h, aligner_in)
        _add(out, "merger_bias", out_h, 1)
    for name in ("image_start", "image_end", "image_newline"):
        _add(out, name, text_hidden, 1)
    return out


def _engram_entries(e: Mapping[str, Any]) -> list[Entry]:
    out: list[Entry] = []
    if e.get("kind") == "ple":
        hc, h, each = int(e["hc_mult"]), int(e["hidden"]), int(e["hc_mult"]) * int(e["hidden"])
        for table in e["tables"]:
            _add(out, f"ple_table_{table['id']}", table["rows"], table["width"])
        for _ in e["tables"]:
            _add(out, "ple_kv_proj", each + h, int(e["embed"]))
            _add(out, "ple_norm_key", each, 1)
            _add(out, "ple_norm_query", each, 1)
            _add(out, "ple_norm_conv", each, 1)
            _add(out, "ple_conv1d", each, int(e["conv_k"]))
        return out
    hc, h = int(e["hc_mult"]), int(e["hidden"])
    for table in e["tables"]:
        _add(out, f"engram_table_{table['id']}", table["rows"], table["width"])
    for _ in e["tables"]:
        _add(out, "engram_gate_wkv", (hc + 1) * h, int(e["columns"]) * int(e["width"]))
        _add(out, "engram_q", hc, h)
        _add(out, "engram_k", hc, h)
    return out


def _standalone_head_entries(model: ReferenceModel) -> list[Entry]:
    draft, h, out = model.geometry.get("draft") or {}, model.hidden, []
    _add(out, "draft_input_proj", h, int(draft["input_width"]))
    _add(out, "draft_input_norm", h, 1)
    _add(out, "draft_final_norm", h, 1)
    markov = int(draft.get("markov_rank", 0))
    if markov:
        _add(out, "markov_embed", model.vocab, markov)
        _add(out, "markov_head", model.vocab, markov)
    return out




def _assemble_geometry(config, raw):
    c, vision_raw = flatten_config(config)
    builder, layers = raw["builder"], raw["layers"]
    vision = _vision_decl(vision_raw, layer_hidden=_i(c, "hidden_size")) if vision_raw else None
    if builder in {"standard_deepseek", "standard_kimi", "deepseek_v4", "kimi_dspark",
                   "gemma4", "minimax_m3", "step5"}:
        vision = None
    engram = _engram_decl(c, layers)
    sharing = {
        "tied_word_embeddings": bool(c.get("tie_word_embeddings", False)),
        "kv_owner_layers": [layer.index for layer in layers if layer.owns_kv],
        "shared_index_layers": [layer.index for layer in layers if layer.index_shared],
        "k_eq_v_layers": [layer.index for layer in layers if layer.k_eq_v],
        "target_aliases": (["target_embedding_alias", "target_lm_head_alias"]
                           if raw["standalone"] else []),
    }
    return {"builder": builder, "family": builder, "layers": layers,
            "vision": vision, "engram": engram, "draft": raw["draft"],
            "standalone": raw["standalone"], "auxiliary": _aux_decl(c, builder),
            "quant": _quant_decl(c, builder), "sharing": sharing,
            "hidden": _i(c, "hidden_size"), "vocab": _i(c, "vocab_size"),
            "tie_word_embeddings": bool(c.get("tie_word_embeddings", False))}


def _vision_decl(v: Mapping[str, Any], *, layer_hidden: int) -> dict[str, Any]:
    model_type = str(v.get("model_type", ""))
    if "vt_hidden_size" in v:
        merge = v.get("merge_kernel_size") or [2, 2]
        return {"kind": "kimi_k3", "hidden": _i(v, "vt_hidden_size"),
                "layers": _i(v, "vt_num_hidden_layers"),
                "intermediate": _i(v, "vt_intermediate_size"),
                "query": _i(v, "qkv_hidden_size"), "qkv": 3 * _i(v, "qkv_hidden_size"),
                "patch_width": 3 * _i(v, "patch_size") ** 2,
                "pos": _i(v, "init_pos_emb_height") * _i(v, "init_pos_emb_width"),
                "bias": False, "gated": False, "norm_type": "rmsnorm",
                "aligner_in": _i(v, "vt_hidden_size") * int(merge[0]) * int(merge[1]),
                "aligner_hidden": _i(v, "vt_intermediate_size"),
                "out_hidden": _i(v, "text_hidden_size", layer_hidden)}
    if "deepseek_v41" in model_type:
        downsample = _i(v, "downsample_ratio", 1) or 1
        hidden = _i(v, "hidden_size")
        return {"kind": "deepseek_v41", "hidden": hidden,
                "layers": _i(v, "num_hidden_layers"),
                "intermediate": _i(v, "intermediate_size"),
                "query": hidden, "qkv": 3 * hidden,
                "patch_width": 3 * _i(v, "patch_size") ** 2,
                "pos": 0, "bias": True, "gated": True, "norm_type": "layernorm",
                "aligner_in": hidden * downsample * downsample,
                "aligner_hidden": layer_hidden, "out_hidden": layer_hidden}
    merge = _i(v, "spatial_merge_size", 2) or 2
    hidden = _i(v, "hidden_size")
    return {"kind": "qwen3_5", "hidden": hidden, "layers": _i(v, "depth"),
            "intermediate": _i(v, "intermediate_size"), "query": hidden,
            "qkv": 3 * hidden,
            "patch_width": 3 * _i(v, "patch_size") ** 2 * (_i(v, "temporal_patch_size", 1) or 1),
            "pos": _i(v, "num_position_embeddings"), "bias": False, "gated": False,
            "norm_type": "layernorm", "aligner_in": hidden * merge * merge,
            "aligner_hidden": 0, "out_hidden": _i(v, "out_hidden_size", layer_hidden)}


def _is_prime_64(value: int) -> bool:
    if value < TWO:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == ZERO:
            return value == prime
    exponent, shifts = value - ONE, ZERO
    while not exponent % TWO:
        exponent //= TWO; shifts += ONE
    for base in _MILLER_RABIN_BASES:
        if base % value == ZERO:
            continue
        witness = pow(base, exponent, value)
        if witness in (ONE, value - ONE):
            continue
        for _ in range(shifts - ONE):
            witness = pow(witness, TWO, value)
            if witness == value - ONE:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + ONE
        if candidate <= TWO:
            prime = TWO; continue
        if not candidate % TWO:
            candidate += ONE
        while not _is_prime_64(candidate):
            candidate += TWO
        prime = candidate
    return prime


def _engram_decl(c: Mapping[str, Any], layers: list[RefLayer]) -> dict[str, Any] | None:
    ple_layers = [int(v) for v in c.get("ple_layer_ids", ()) or ()]
    hidden = _i(c, "hidden_size")
    hc = max((layer.hc_mult for layer in layers), default=1)
    if ple_layers:
        ngram, heads = _i(c, "ngram_size"), _i(c, "heads_per_ngram")
        embed = _i(c, "ple_embed_dim")
        ngram_heads = (ngram - ONE) * heads
        width = embed // ngram_heads
        divisor = _i(c, "make_ngram_vocab_size_divisible_by", PLE_DEFAULT_DIVISOR)
        base = _i(c, "ngram_vocab_size_base")
        conv_k = _i(c, "ple_conv_kernel_size", PLE_DEFAULT_CONV_KERNEL)
        tables = []
        for index, ple_id in enumerate(ple_layers):
            total = ZERO
            for local_head in range(ngram_heads):
                global_head = index * ngram_heads + local_head
                total += _nth_prime_after(base - ONE, global_head + ONE)
            rows = ((total + divisor - ONE) // divisor) * divisor
            tables.append({"id": index, "layer": ple_id - ONE, "rows": rows, "width": width})
        for table in tables:
            layers[table["layer"]] = _with_engram(layers[table["layer"]])
        return {"kind": "ple", "tables": tables, "hidden": hidden, "hc_mult": hc,
                "embed": embed, "conv_k": conv_k, "max_ngram": ngram}
    layer_ids = [int(v) for v in c.get("engram_layer_ids", ()) or ()]
    rows = [int(v) for v in c.get("engram_num_embeddings", ()) or ()]
    if not layer_ids:
        return None
    width, ngram = _i(c, "engram_head_dim"), _i(c, "engram_max_ngram_size")
    heads = _i(c, "engram_n_heads")
    tables = [{"id": index, "layer": layer, "rows": row, "width": width}
              for index, (layer, row) in enumerate(zip(layer_ids, rows))]
    for table in tables:
        layers[table["layer"]] = _with_engram(layers[table["layer"]])
    return {"kind": "engram", "tables": tables, "hidden": hidden, "hc_mult": hc,
            "width": width, "columns": (ngram - ONE) * heads}


def _with_engram(layer: RefLayer) -> RefLayer:
    return replace(layer, engram=True)


def _aux_decl(c: Mapping[str, Any], builder: str) -> dict[str, Any]:
    targets = tuple(int(v) for v in c.get("dspark_target_layer_ids", ()) or ())
    nextn = _i(c, "num_nextn_predict_layers")
    if builder in {"deepseek_v4", "deepseek_v41"}:
        layers = len(targets) if targets else nextn
        if layers <= 0:
            return {}
        block = _i(c, "dspark_block_size", 1)
        markov = _i(c, "dspark_markov_rank")
        # A declared multi-token DSpark block (block_size > 1 and a Markov
        # rank) is the native DSpark module, not an ordinary NextN module.
        kind = "dspark" if (block > 1 or markov) else "mtp_dspark"
        return {"kind": kind, "layers": layers, "standalone": False,
                "target_layer_ids": list(targets),
                "block_size": block}
    if builder == "kimi":
        targets = targets or (2, 23, 47, 71, 89)
        return {"kind": "dspark", "layers": 5, "standalone": False,
                "external": True, "target_layer_ids": list(targets),
                "block_size": _i(c, "dspark_block_size", 5)}
    if builder == "kimi_dspark":
        return {"kind": "standalone_draft",
                "layers": _i(c, "num_hidden_layers"), "standalone": True}
    if builder == "glm" and nextn > 0:
        return {"kind": "mtp", "layers": nextn, "standalone": False}
    mtp = _i(c, "mtp_num_hidden_layers")
    if builder == "qwen" and mtp > 0:
        return {"kind": "mtp", "layers": mtp, "standalone": False}
    if builder == "step5" and nextn > 0:
        # Step-5 owns native NextN blocks with ids num_hidden_layers .. +nextn-1.
        return {"kind": "mtp", "layers": nextn, "standalone": False}
    return {}


def _quant_decl(c: Mapping[str, Any], builder: str) -> dict[str, Any]:
    declared = c.get("quantization_config") or {}
    scheme = str(declared.get("quant_method", "none") or "none")
    native = str(c.get("torch_dtype") or c.get("dtype") or "bf16")
    index_dtype, scale = "bf16", "none"
    if builder in {"deepseek_v4", "deepseek_v41"}:
        index_dtype, scale = "int8", "fp16"
    return {"scheme": scheme, "native_dtype": native, "kv_dtype": "bf16",
            "index_dtype": index_dtype, "index_scale_dtype": scale,
            "weight_block": list(declared.get("weight_block_size") or ()),
            "expert_dtype": str(c.get("expert_dtype", ""))}


# --------------------------------------------------------------------------
# Manifest consumption and comparison.
# --------------------------------------------------------------------------
def declaration(reference: ReferenceModel) -> dict[str, Any]:
    return {
        "builder": reference.builder,
        "family": reference.family,
        "parameter_scope": reference.scope,
        "config_sha256": reference.config_sha256,
        "cache_layout": cache_layout_declaration(reference.geometry),
        "auxiliary_modes": reference.geometry["auxiliary"],
        "sharing_relations": reference.geometry["sharing"],
        "quantization": reference.geometry["quant"],
    }


def build_reference(key: str, repo_root: str | Path, entry: Mapping[str, Any]) -> ReferenceModel:
    root = Path(repo_root)
    declared = str(entry.get("config") or "")
    if not declared:
        raise ValueError(f"reference_missing_config:{key}")
    path = (root / declared).resolve()
    if not path.is_file():
        raise ValueError(f"reference_config_missing:{key}:{declared}")
    digest = config_sha256(path)
    expected = str(entry.get("config_sha256") or "")
    if not expected:
        raise ValueError(f"reference_missing_sha256:{key}")
    if digest != expected:
        raise ValueError(f"reference_sha256_mismatch:{key}:{digest}!={expected}")
    config = json.loads(path.read_text(encoding="utf-8"))
    geometry = _assemble_geometry(config, build_layers(config))
    if str(entry.get("builder") or "") != geometry["builder"]:
        raise ValueError(
            f"reference_builder_mismatch:{key}:{entry.get('builder')}!={geometry['builder']}")
    scope = str(entry.get("parameter_scope") or "")
    if scope != SCOPE_TEXT_BACKBONE:
        raise ValueError(f"reference_scope_unsupported:{key}:{scope}")
    reference = ReferenceModel(
        key=key, config_path=declared, config_sha256=digest,
        builder=geometry["builder"], family=geometry["family"], scope=scope,
        hidden=geometry["hidden"], vocab=geometry["vocab"],
        tie_word_embeddings=geometry["tie_word_embeddings"],
        layers=geometry["layers"], geometry=geometry,
    )
    derived = declaration(reference)
    for field_name in ("cache_layout", "auxiliary_modes", "sharing_relations", "quantization"):
        if field_name not in entry:
            raise ValueError(f"reference_declaration_missing:{key}:{field_name}")
        if entry[field_name] != derived[field_name]:
            raise ValueError(f"reference_declaration_mismatch:{key}:{field_name}")
    return reference

def compare_entries(reference: ReferenceModel, actual: Iterable[Sequence[Any]]) -> dict[str, Any]:
    actual_entries = [tuple(item) for item in actual]
    ref_counter = Counter((e.name, e.rows, e.cols, e.count, e.owner)
                          for e in reference.entries())
    act_counter = Counter(actual_entries)
    actual_parameters = sum(int(e[1]) * int(e[2]) * int(e[3])
                            for e in actual_entries if e[4] == "source")
    actual_runtime = sum(int(e[1]) * int(e[2]) * int(e[3])
                         for e in actual_entries if e[4] == "runtime")
    reference_parameters = _total(reference, "source")
    reference_runtime = _total(reference, "runtime")
    missing_in_actual = list((ref_counter - act_counter).elements())
    missing_in_reference = list((act_counter - ref_counter).elements())
    ref_by, act_by = defaultdict(list), defaultdict(list)
    for item in ref_counter.elements():
        ref_by[item[0]].append(item[1:])
    for item in act_counter.elements():
        act_by[item[0]].append(item[1:])
    mismatches = [{"name": name,
                   "actual": [list(v) for v in sorted(act_by[name])],
                   "reference": [list(v) for v in sorted(ref_by[name])]}
                  for name in sorted(set(ref_by) | set(act_by))
                  if sorted(ref_by[name]) != sorted(act_by[name])]
    delta = actual_parameters - reference_parameters
    runtime_delta = actual_runtime - reference_runtime
    return {
        "parameter_delta": int(delta), "actual_parameters": int(actual_parameters),
        "reference_parameters": int(reference_parameters),
        "runtime_delta": int(runtime_delta), "actual_runtime_parameters": int(actual_runtime),
        "reference_runtime_parameters": int(reference_runtime),
        "missing_in_actual": [list(v) for v in missing_in_actual],
        "missing_in_reference": [list(v) for v in missing_in_reference],
        "shape_mismatches": mismatches,
        "complete": bool(not delta and not runtime_delta and not missing_in_actual
                         and not missing_in_reference and not mismatches),
    }

def cache_delta(reference: ReferenceModel, actual_payload: float, actual_allocated: float,
                seq: int, *, tp: int = 1, cp: int = 1) -> dict[str, Any]:
    reference_payload = cache_totals(reference.geometry, seq, padded=False, tp=tp, cp=cp)
    reference_allocated = cache_totals(reference.geometry, seq, padded=True, tp=tp, cp=cp)
    return {
        "payload_delta": float(actual_payload) - float(reference_payload),
        "allocated_delta": float(actual_allocated) - float(reference_allocated),
        "actual_payload": float(actual_payload),
        "reference_payload": float(reference_payload),
        "actual_allocated": float(actual_allocated),
        "reference_allocated": float(reference_allocated),
    }


__all__ = [
    "Entry", "MANIFEST_SCHEMA", "REFERENCE_PINS", "ReferenceModel", "build_reference",
    "cache_delta", "compare_entries", "config_sha256", "declaration", "load_manifest",
    "logical_numel", "make_meta_parameters", "manifest_models",
]
