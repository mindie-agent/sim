"""Executable ``mindie-bench-subgraph-plan/1`` builder and writer-clock/geometry helpers.

[Role] Translate onboarding cliff/SKU plans into executable L1/L2/L3
  ``mindie-bench-subgraph-plan/1`` plans with exact ``ParallelSpec`` world
  sizes, batch-preserving canonical ``mindie-profiler`` operator bindings
  matched 1:1 to lowered ``mindie-sim`` recipes, layouts, KV lengths, page
  counts, per-head weight counts (``grouped_matmul``), BF16 indexer ``topk``,
  and lowered 1-layer L3 ``model_step`` DAGs; also parse
  ``run_id.<arm>.<mode>.t<trial>`` suffixes, derive ``clock_operator`` strictly
  from writer timing evidence, verify receipt/builder-reported ``actual_shape``
  (rejecting alias conflicts and ``requested_shape``-only ABI rows), and
  extract canonical executed geometry.
[Invariants]
  * Reuses ``required_ranks_for_plan`` / ``validate_parallel_topology``.
  * L3 ``model_step`` cases are lowered directly from the exact same 1-layer
    ``spine.lower(...)`` graph that ``WhiteboxSpineRunner`` replays.
  * Never accepts ``requested_shape``-only ABI receipts or conflicting shape
    aliases (e.g. ``topk=2049`` + ``sparse_count=2048``).
  * Never overwrites writer-measured ``clock_operator`` or receipt-reported
    ``actual_shape`` with the plan; fails closed before sealing.
[Timeouts & Locks] Pure in-memory plan/record helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
import math
import re
from typing import Any

from mindie_sim.backends import ascend_profiles
from mindie_sim.schema.common import ModelingError

from .contracts import VALID_CLOCK_OPERATORS

_DEFAULT_ATTENTION_HEADS = 16
_INDEXER_ATTENTION_HEADS = 32
_DEFAULT_ATTENTION_KV_HEADS = 1
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_HEAD_DIM = 128
_MLA_BLOCK_SIZE = 128
_MLA_HIDDEN_WIDTH = 2048
_MLA_BF16_BYTES = 2
_DEFAULT_HEAD_DIM = 128
_DEFAULT_HIDDEN_WIDTH = 4096
_DEFAULT_CONTEXT_FALLBACK = 2048
_L2_CONTEXT_MIN = 24576
_L3_CONTEXT_MIN = 200000
_MIN_COLLECTIVE_BYTES = 4
_DEFAULT_COLLECTIVE_BYTES = 4096
_RING_ALLREDUCE_FACTOR = 2
_MIN_DCP_PARTIALS = 2
_MIN_GEMM_SHAPE_RANK = 3
_MIN_2D_SHAPE_RANK = 2
_BLOCK_TABLE_PAIR_LEN = 2
_LOWER_CACHE_MAXSIZE = 64
_DTYPE_ELEMENT_BYTES: Mapping[str, int] = {
    "fp32": 4, "float32": 4, "int32": 4,
    "bf16": 2, "fp16": 2, "float16": 2,
    "int8": 1, "uint8": 1, "fp8_e4m3": 1, "mxfp8": 1,
}


def _dtype_element_bytes(dtype: str) -> int:
    return int(_DTYPE_ELEMENT_BYTES.get(str(dtype or "bf16").strip().lower(), _MLA_BF16_BYTES))
_ARCH35_SKU_MARKER = "950"
_ARCH35_PRESET_MARKER = "35"
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_INDEXER_SIM_MODEL = "deepseek-v4.1-flash"
_FALLBACK_LOWER_SKU = "Ascend910B4"
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_KNOWN_ARMS = frozenset({"base", "aclgraph"})
_L2_REGIME_TAGS = frozenset({"high_batch_decode_2d", "mtp_decode_2d", "chunked_prefill_2d"})
_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
_BUSY_UNION_TIMING_MODES = frozenset({
    "stream_saturated", "busy_union", "event_busy_union", "eager_device", "aclgraph_device",
})
_ELAPSED_SPAN_TIMING_MODES = frozenset({"eager", "aclgraph", "elapsed_span", "event_elapsed_span"})
_BUSY_UNION_BOUNDARIES = frozenset({"busy_union", "event_busy_union"})
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_FAMILY_BINDINGS: Mapping[str, tuple[str, str, str, str]] = {
    "rmsnorm": ("rmsnorm", "rmsnorm", "rms_norm", "aclnnRmsNorm"),
    "add_rmsnorm": ("add_rmsnorm", "add_rmsnorm", "add_rms_norm", "aclnnAddRmsNorm"),
    "layernorm": ("layernorm", "layernorm", "layer_norm", "aclnnLayerNorm"),
    "attention": ("attention", "ascend_paged_mla_decode_arch22", "absorbed_mla", "aclnnFusedInferAttentionScoreV4"),
    "sparse_indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "lightning_indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "topk": ("topk", "index_topk", "indexer_topk", "aclnnTopkV2"),
    "swiglu": ("swiglu", "swiglu", "swiglu", "aclnnSwiGlu"),
    "geglu": ("geglu", "geglu", "geglu", "aclnnGeGluV2"),
    "rotary": ("rotary", "rotary", "rotary_embedding", "aclnnApplyRotaryPosEmbV2"),
    "gemm": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "matmul": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "linear": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "grouped_gemm": ("grouped_gemm", "grouped_matmul", "grouped_matmul", "aclnnGroupedMatmulV5"),
    "routing": ("moe_init_routing", "moe_init_routing_v3", "moe_init_routing", "aclnnMoeInitRoutingV3"),
    "moe_init_routing": ("moe_init_routing", "moe_init_routing_v3", "moe_init_routing", "aclnnMoeInitRoutingV3"),
    "cast": ("cast", "cast", "dtype_cast", "aclnnCast"),
    "elementwise": ("elementwise", "residual_add", "vector_add", "aclnnAdd"),
}


def quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    """Compute linear-interpolated quantile over sorted integer ns samples."""
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return int(sorted_ns[lo])
    return int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def parse_run_id_and_arm(
    run_id: str,
    timing_mode: str = "",
    trial: Any = None,
    source: str = "",
    execution: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Strip ``.<arm>.<mode>.t<trial>`` or ``.<mode>.t<trial>`` to recover ``(base_run_id, arm)``."""
    clean = str(run_id or "run-0").strip()
    if timing_mode and trial is not None and not isinstance(trial, bool):
        suffix = f".{timing_mode}.t{int(trial)}"
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    else:
        clean = _TRIAL_SUFFIX_RE.sub("", clean)
    arm = ""
    for candidate in _KNOWN_ARMS:
        marker = f".{candidate}"
        if clean.endswith(marker):
            clean = clean[: -len(marker)]
            arm = candidate
            break
    if not arm and isinstance(execution, Mapping):
        if str(execution.get("graph") or "").strip().lower() == "aclgraph":
            arm = "aclgraph"
    if not arm and source:
        prefix = source.rsplit("/", 1)[-1].split(".", 1)[0].strip().lower()
        if prefix in _KNOWN_ARMS:
            arm = prefix
    return clean or "run-0", arm or "base"


def _int_from(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None and not isinstance(val, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def _unique_alias_int(
    mapping: Mapping[str, Any], keys: Sequence[str], *, cid: str, field: str
) -> int | None:
    """Resolve an integer across alias keys and fail closed on any conflicting values."""
    found: list[tuple[str, int]] = []
    for key in keys:
        if key not in mapping or mapping[key] is None:
            continue
        raw = mapping[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ModelingError(f"profiler_shape_alias_invalid:{cid}:{field}:{key}={raw!r}")
        found.append((key, int(raw)))
    if not found:
        return None
    distinct = {val for _k, val in found}
    if len(distinct) > 1:
        detail = ",".join(f"{k}={v}" for k, v in found)
        raise ModelingError(f"profiler_shape_alias_conflict:{cid}:{field}:{detail}")
    return found[0][1]


def extract_shape_dims(
    shape: Mapping[str, Any],
    plan_key: Mapping[str, Any],
    *,
    plan_pt: Mapping[str, Any] | None = None,
    expected_shape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract canonical integer onboarding coordinates strictly from executed shape + plan key."""
    dims: dict[str, Any] = {}
    axis = str((plan_pt or {}).get("axis") or plan_key.get("axis") or "").strip()
    merged = {**dict(plan_key), **dict(shape)}

    q_val = _int_from(shape, "q_len", "query_tokens", "q")
    if q_val is None:
        q_val = _int_from(plan_key, "q_len", "query_tokens", "q")
    dims["query_tokens"] = q_val if q_val is not None else 1

    ctx_explicit = _int_from(shape, "context_len", "context", "kv_len_prior")
    seq_len_val = _int_from(shape, "seq_len", "kv_len", "scanned_rows")
    exp_seq_len = _int_from(expected_shape or {}, "seq_len", "kv_len")
    exp_ctx_len = _int_from(expected_shape or {}, "context_len")
    if seq_len_val is not None:
        dims["kv_len"] = seq_len_val
    if ctx_explicit is not None:
        dims["context_len"] = ctx_explicit
    elif seq_len_val is not None and exp_seq_len is not None and exp_ctx_len is not None and seq_len_val == exp_seq_len:
        dims["context_len"] = exp_ctx_len
    elif seq_len_val is not None:
        dims["context_len"] = seq_len_val
    else:
        pk_ctx = _int_from(plan_key, "context_len", "seq_len", "context")
        if pk_ctx is not None:
            dims["context_len"] = pk_ctx

    tok_val = _int_from(shape, "tokens", "m", "num_tokens")
    if tok_val is None:
        tok_val = _int_from(plan_key, "tokens", "m", "num_tokens")
    if tok_val is not None:
        dims["tokens"] = tok_val

    seq_val = _int_from(shape, "sequences", "batch_size", "batch", "b")
    kv_val = _int_from(shape, "kv_heads")
    bxkv_val = _int_from(shape, "batch_x_kv_heads")
    if bxkv_val is None and seq_val is not None and kv_val is not None and axis == "batch_x_kv_heads":
        bxkv_val = seq_val * kv_val
    if bxkv_val is not None:
        dims["batch_x_kv_heads"] = bxkv_val

    if seq_val is not None:
        dims["sequences"] = seq_val
    if axis == "batch_x_kv_heads":
        dims["batch_size"] = _int_from(plan_pt or {}, "batch_size") or _int_from(plan_key, "batch_size") or 1
    elif seq_val is not None:
        dims["batch_size"] = seq_val
    else:
        dims["batch_size"] = _int_from(plan_key, "batch_size", "batch", "b", "sequences") or 1

    tk_val = _int_from(shape, "sparse_count", "top_k", "topk", "topk_items")
    if tk_val is None:
        tk_val = _int_from(plan_key, "sparse_count", "top_k", "topk", "topk_items")
    if tk_val is not None:
        dims["topk"] = tk_val

    dims["parallel_topology"] = str(merged.get("parallel_topology") or "tp1")
    return dims


def parse_topo_options(topo: str) -> dict[str, int]:
    """Parse canonical topology tokens into ``compose_options``."""
    return {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(str(topo or "tp1").lower())} or {"tp": 1}


def _decode_mla_recipe_for_sku(sku: str) -> str:
    prof = ascend_profiles().get(sku)
    preset = ""
    if isinstance(prof, Mapping):
        dev = prof.get("device")
        if isinstance(dev, Mapping):
            ca = dev.get("compute_arch")
            if isinstance(ca, Mapping) and ca.get("preset"):
                preset = str(ca["preset"]).strip()
    if _ARCH35_PRESET_MARKER in preset or _ARCH35_SKU_MARKER in str(sku).lower():
        return "ascend_paged_mla_decode_arch35"
    return "ascend_paged_mla_decode_arch22"


def assign_plan_point_level(plan_doc: Mapping[str, Any], pt: Mapping[str, Any]) -> str:
    """Deterministically assign ``L1`` / ``L2`` / ``L3`` for a plan point."""
    explicit = str(pt.get("level") or "").strip().upper()
    if explicit in _LEVEL_BOUNDARY_KINDS:
        return explicit
    if str(plan_doc.get("family") or "").strip() != "sku_suite":
        return "L1"
    if str(pt.get("split") or "").strip().lower() != "heldout":
        return "L1"
    topo = str(pt.get("parallel_topology") or "tp1").strip().lower()
    tag = str(pt.get("regime_tag") or "").strip()
    val = int(pt.get("value") or 0)
    if topo != "tp1" or (tag == "off_grid_heldout" and val >= _L3_CONTEXT_MIN):
        return "L3"
    if tag in _L2_REGIME_TAGS or (tag == "off_grid_heldout" and _L2_CONTEXT_MIN <= val < _L3_CONTEXT_MIN):
        return "L2"
    return "L1"


def _page_count(kv_len: int, block_size: int = _MLA_BLOCK_SIZE) -> int:
    return max(1, (int(kv_len) + int(block_size) - 1) // int(block_size))


@lru_cache(maxsize=_LOWER_CACHE_MAXSIZE)
def _lower_one_layer_step_ops(
    sku: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int
) -> tuple[dict[str, Any], ...]:
    """Lower a 1-layer ``deepseek-v2-lite`` step via ``default_spine()`` so L3 profiler == L3 replay."""
    from mindie_sim.compose import compose_case as compose_case_dict
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.spine import default_spine
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.stack_registry import LoweringRequest

    hw_key = sku if sku in ascend_profiles() else _FALLBACK_LOWER_SKU
    opts = parse_topo_options(topo)
    raw = compose_case_dict(model=_DEFAULT_SIM_MODEL, hardware=hw_key, **opts)
    model_d = dict(raw["model"])
    model_d["layers"] = 1
    model_d["body_layers"] = 1
    model_d["layer_plans"] = list(model_d["layer_plans"])[:1]
    case = parse_case({**raw, "model": model_d})
    prep = default_spine().lower(
        case,
        LoweringRequest(
            step=StepRequest(
                phase=phase,  # type: ignore[arg-type]
                tokens=int(q_tok),
                context=int(ctx_len),
                sequences=int(seqs),
                scope="body",
            ),
            tier="mainline",
        ),
    )
    out: list[dict[str, Any]] = []
    for op in prep.graph.executable[0].ops:
        meta = dict(op.metadata)
        nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
        pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
        out.append({
            "op_id": str(op.op_id),
            "family": str(op.family),
            "recipe": str(op.recipe),
            "dtype": str(op.dtype or "bf16"),
            "local_shape": tuple(int(x) if isinstance(x, (int, float)) else x for x in (op.local_shape or ())),
            "deps": tuple(str(d) for d in (op.deps or ())),
            "link_bytes": int(round(float(op.link_bytes or 0.0))),
            "matrix_groups": int(meta.get("matrix_groups") or 1),
            "group_axis": str(meta.get("group_axis") or ""),
            "query_layout": str(nabi.get("query_layout") or ""),
            "cache_layout": str(nabi.get("cache_layout") or ""),
            "output_layout": str(nabi.get("output_layout") or ""),
            "kv_len": int(pgeo["kv_lengths"][0]) if pgeo.get("kv_lengths") else 0,
            "page_count": int(pgeo["page_counts"][0]) if pgeo.get("page_counts") else 0,
        })
    return tuple(out)


def _lowered_attn_or_gemm_node(
    nid: str,
    op_rec: Mapping[str, Any],
    *,
    attn_shape: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    op_id = str(op_rec["op_id"])
    fam = str(op_rec["family"])
    rec = str(op_rec["recipe"])
    loc = tuple(op_rec["local_shape"])
    mg = max(1, int(op_rec["matrix_groups"]))

    if fam == "attention":
        sh = dict(attn_shape)
        if op_rec["kv_len"] > 0:
            sh["seq_len"] = int(op_rec["kv_len"])
            sh["kv_len"] = int(op_rec["kv_len"])
            sh["page_count"] = int(op_rec["page_count"]) or _page_count(int(op_rec["kv_len"]))
        elif len(loc) >= _MIN_2D_SHAPE_RANK and int(loc[1]) > 0:
            sh["seq_len"] = int(loc[1])
            sh["kv_len"] = int(loc[1])
            sh["page_count"] = _page_count(int(loc[1]))
        if op_rec["query_layout"]:
            sh["query_layout"] = op_rec["query_layout"]
            sh["input_layout"] = op_rec["query_layout"]
            sh["cache_layout"] = op_rec["cache_layout"] or "PA_BSND"
            sh["output_layout"] = op_rec["output_layout"] or "NBSD"
        node: dict[str, Any] = {
            "node_id": nid, "op_id": op_id, "kind": "operator",
            "op": "absorbed_mla", "builder": "absorbed_mla", "family": "attention",
            "recipe": rec, "implementation": "aclnnFusedInferAttentionScoreV4",
            "dtype": "bf16", "paged": True, "layout": sh.get("query_layout", "TND"),
            "shape": sh,
        }
        binding = {
            "op_id": op_id, "family": "attention", "recipe": rec, "builder": "absorbed_mla",
            "paged": True, "heads": sh["heads"], "kv_heads": 1, "latent": _MLA_LATENT_DIM,
            "rope": _MLA_ROPE_DIM, "kv_len": sh["seq_len"], "page_count": sh["page_count"],
            "query_layout": sh.get("query_layout", "TND"),
        }
        return node, binding

    if fam in {"gemm", "gate_up_projection", "down_projection"} and len(loc) >= _MIN_GEMM_SHAPE_RANK:
        m_dim, n_dim, k_dim = int(loc[0]), int(loc[1]), int(loc[2])
        if mg > 1:
            per_grp = max(1, m_dim // mg)
            node = {
                "node_id": nid, "op_id": op_id, "kind": "operator",
                "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm",
                "recipe": "grouped_matmul", "implementation": "aclnnGroupedMatmulV5",
                "dtype": "bf16", "group_list": [per_grp] * mg, "group_list_type": 1,
                "shape": {"m": m_dim, "n": n_dim, "k": k_dim, "e": mg, "matrix_groups": mg},
            }
            return node, {
                "op_id": op_id, "family": fam, "recipe": rec, "builder": "grouped_matmul",
                "m": m_dim, "n": n_dim, "k": k_dim, "matrix_groups": mg,
            }
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator",
            "op": "matmul", "builder": "matmul", "family": fam, "recipe": rec,
            "implementation": "aclnnMatmul", "dtype": "bf16",
            "shape": {"m": m_dim, "n": n_dim, "k": k_dim, "tokens": m_dim, "width": k_dim, "matrix_groups": 1},
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "matmul",
            "m": m_dim, "n": n_dim, "k": k_dim, "matrix_groups": 1,
        }
    return None


def _lowered_op_to_profiler_node(
    pid: str,
    op_rec: Mapping[str, Any],
    *,
    attn_shape: Mapping[str, Any],
    total_tokens: int,
    local_heads: int,
    topo_opts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one lowered ``OpSpec`` summary into a profiler node + node binding."""
    op_id = str(op_rec["op_id"])
    fam = str(op_rec["family"])
    rec = str(op_rec["recipe"])
    loc = tuple(op_rec["local_shape"])
    tp = max(1, int(topo_opts.get("tp", 1)))
    dcp = max(1, int(topo_opts.get("dcp", 1)))
    nid = f"{pid}:{op_id}"

    attn_or_gemm = _lowered_attn_or_gemm_node(nid, op_rec, attn_shape=attn_shape)
    if attn_or_gemm is not None:
        return attn_or_gemm

    if fam == "rmsnorm":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "rms_norm", "builder": "rms_norm",
            "family": "rmsnorm", "recipe": rec, "implementation": "aclnnRmsNorm", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "rmsnorm", "recipe": rec, "builder": "rms_norm", "width": width}

    if fam == "add_rmsnorm":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "add_rms_norm", "builder": "add_rms_norm",
            "family": "add_rmsnorm", "recipe": rec, "implementation": "aclnnAddRmsNorm", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "add_rmsnorm", "recipe": rec, "builder": "add_rms_norm", "width": width}

    if fam == "swiglu":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "swiglu", "builder": "swiglu",
            "family": "swiglu", "recipe": rec, "implementation": "aclnnSwiGlu", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "swiglu", "recipe": rec, "builder": "swiglu", "width": width}

    if fam == "elementwise" and rec == "rotary":
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "rotary_embedding", "builder": "rotary_embedding",
            "family": "elementwise", "recipe": "rotary", "implementation": "aclnnApplyRotaryPosEmbV2", "dtype": "bf16",
            "shape": {"tokens": total_tokens, "heads": local_heads, "head_dim": _MLA_ROPE_DIM, "width": _MLA_ROPE_DIM},
        }
        return node, {"op_id": op_id, "family": "elementwise", "recipe": "rotary", "builder": "rotary_embedding"}

    if fam == "kv_compress_epilog":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        op_dt = str(op_rec["dtype"] or "bf16").strip().lower()
        elem_b = _dtype_element_bytes(op_dt)
        byte_width = int(loc[-1]) if loc else (_MLA_LATENT_DIM + _MLA_ROPE_DIM) * _MLA_BF16_BYTES
        elem_width = max(1, byte_width // elem_b)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "kv_compress_epilog", "builder": "kv_compress_epilog",
            "family": "kv_compress_epilog", "recipe": rec, "implementation": "aclnnKvCompressEpilog", "dtype": op_dt,
            "shape": {"tokens": toks, "width": elem_width, "byte_width": byte_width},
        }
        return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "kv_compress_epilog", "dtype": op_dt, "width": elem_width, "byte_width": byte_width}

    if fam == "dcp_allgather":
        raw_link = int(op_rec["link_bytes"])
        send_nbytes = max(
            _MIN_COLLECTIVE_BYTES,
            (raw_link // max(1, dcp - 1)) if raw_link > 0 else _DEFAULT_COLLECTIVE_BYTES,
        )
        node = {
            "node_id": nid, "op_id": op_id, "kind": "collective", "op": "all_gather", "builder": "all_gather",
            "family": "dcp_allgather", "recipe": rec, "implementation": "hcclAllGather", "dtype": "bf16",
            "group": "dcp", "group_size": dcp, "nbytes": send_nbytes,
            "shape": {"tokens": total_tokens, "width": _MLA_LATENT_DIM + 1, "nbytes": send_nbytes, "link_bytes": raw_link},
        }
        return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "all_gather", "nbytes": send_nbytes, "group_size": dcp}

    if fam == "attention_merge":
        toks = int(loc[0]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else total_tokens
        heads = int(loc[1]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else local_heads
        hdim = int(loc[2]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else _MLA_LATENT_DIM
        merge_dt = str(op_rec["dtype"] or "fp32").strip().lower()
        partials = max(_MIN_DCP_PARTIALS, dcp)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "dcp_lse_attention_merge",
            "builder": "dcp_lse_attention_merge", "family": "attention_merge", "recipe": rec,
            "implementation": "aclnnRingAttentionUpdate", "dtype": merge_dt,
            "shape": {"tokens": toks, "heads": heads, "kv_heads": 1, "head_dim": hdim, "width": hdim, "partial_count": partials},
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "dcp_lse_attention_merge",
            "dtype": merge_dt, "partial_count": partials,
        }

    if fam == "allreduce":
        grp_axis = op_rec["group_axis"] or "tp"
        grp_sz = dcp if grp_axis == "cp" else tp
        raw_link = int(op_rec["link_bytes"])
        if raw_link > 0 and grp_sz > 1:
            payload_nbytes = max(
                _MIN_COLLECTIVE_BYTES,
                (raw_link * grp_sz) // max(1, _RING_ALLREDUCE_FACTOR * (grp_sz - 1)),
            )
        else:
            payload_nbytes = max(_MIN_COLLECTIVE_BYTES, raw_link or _DEFAULT_COLLECTIVE_BYTES)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "collective", "op": "all_reduce", "builder": "all_reduce",
            "family": "allreduce", "recipe": rec, "implementation": "hcclAllReduce", "dtype": "bf16",
            "group": grp_axis, "group_size": grp_sz, "nbytes": payload_nbytes,
            "shape": {"tokens": total_tokens, "width": _MLA_HIDDEN_WIDTH, "nbytes": payload_nbytes, "link_bytes": raw_link},
        }
        return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "all_reduce", "nbytes": payload_nbytes, "group_size": grp_sz}

    width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
    node = {
        "node_id": nid, "op_id": op_id, "kind": "operator", "op": "vector_add", "builder": "vector_add",
        "family": "elementwise", "recipe": "residual_add", "implementation": "aclnnAdd", "dtype": "bf16",
        "shape": {"tokens": total_tokens, "width": width},
    }
    return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "vector_add"}


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any],
    pt: Mapping[str, Any],
    *,
    sku: str,
) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    """Return ``(prof_fam, prof_recipe, prof_builder, impl, sim_fam, shape, sim_case)``."""
    plan_family = str(plan_doc.get("family") or "").strip() or "rmsnorm"
    axis = str(pt.get("axis") or "context_len").strip()
    val = int(pt["value"])
    batch = max(1, int(pt.get("batch_size", 1)))
    q_tok = max(1, int(pt.get("query_tokens", 1)))
    topo = str(pt.get("parallel_topology") or "tp1").strip()
    topk_val = int(pt["topk"]) if pt.get("topk") is not None else None
    is_abi_reject = str(pt.get("split") or "").strip().lower() == "abi_rejection"
    topo_opts = parse_topo_options(topo)
    tp = max(1, int(topo_opts.get("tp", 1)))
    dcp = max(1, int(topo_opts.get("dcp", 1)))
    pcp = max(1, int(topo_opts.get("pcp", 1)))
    local_mla_heads = max(1, _DEFAULT_ATTENTION_HEADS // tp)
    decode_mla_rec = _decode_mla_recipe_for_sku(sku)
    level = assign_plan_point_level(plan_doc, pt)
    custom_sim = pt.get("sim_case") or plan_doc.get("sim_case")
    is_default_mla_model = not (
        isinstance(custom_sim, Mapping)
        and custom_sim.get("compose_model")
        and str(custom_sim["compose_model"]) != _DEFAULT_SIM_MODEL
    )
    model_overrides: dict[str, Any] = {}

    if topk_val is not None or plan_family in _SPARSE_INDEXER_FAMILIES:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["indexer"]
        sim_fam, model_name = "indexer", _INDEXER_SIM_MODEL
        ctx_len = val if axis == "context_len" else max(val, _DEFAULT_CONTEXT_FALLBACK)
        kv_len = ctx_len if is_abi_reject else (ctx_len + q_tok)
        shape: dict[str, Any] = {
            "tokens": max(1, batch * q_tok), "width": _DEFAULT_HEAD_DIM,
            "heads": _INDEXER_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM,
            "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len,
            "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok,
            "query_dtype": "bf16", "cache_dtype": "bf16", "input_layout": "TND", "query_layout": "TND",
            "parallel_topology": topo,
        }
        if topk_val is not None:
            shape["top_k"] = topk_val
            shape["topk"] = topk_val
            shape["sparse_count"] = topk_val
            shape["enforce_exact_topk"] = True
            if not is_abi_reject:
                model_overrides = {"index_topk": topk_val, "index_dtype": "bf16"}
    elif (plan_family == "sku_suite" and axis in {"context_len", "batch_x_kv_heads"}) or (
        plan_family == "attention" and is_default_mla_model
    ):
        is_single_q_local = q_tok == 1 and tp == 1 and dcp == 1 and pcp == 1
        attn_rec = decode_mla_rec if is_single_q_local else "latent_attention0"
        pfam, prec, pbld, impl = ("attention", attn_rec, "absorbed_mla", "aclnnFusedInferAttentionScoreV4")
        sim_fam, model_name = "attention", _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        total_tokens = max(1, seqs * q_tok)
        kv_len = ctx_len + q_tok
        if dcp > 1:
            lowered = _lower_one_layer_step_ops(sku, topo, "prefill" if q_tok > 1 else "decode", q_tok, ctx_len, seqs)
            la_op = next((o for o in lowered if o["family"] == "attention"), None)
            if la_op and len(la_op["local_shape"]) >= _MIN_2D_SHAPE_RANK:
                kv_len = int(la_op["local_shape"][1])
        q_layout = "BNSD" if is_single_q_local else "TND"
        pages = _page_count(kv_len, _MLA_BLOCK_SIZE)
        shape = {
            "tokens": total_tokens, "sequences": seqs, "batch_size": batch,
            "query_tokens": q_tok, "q_len": q_tok,
            "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "page_count": pages,
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _MLA_LATENT_DIM, "latent": _MLA_LATENT_DIM, "kv_lora_rank": _MLA_LATENT_DIM,
            "rope": _MLA_ROPE_DIM, "qk_rope_head_dim": _MLA_ROPE_DIM,
            "block_size": _MLA_BLOCK_SIZE, "paged": True,
            "query_layout": q_layout, "input_layout": q_layout,
            "cache_layout": "PA_BSND", "output_layout": "NBSD" if is_single_q_local else "TND",
            "parallel_topology": topo,
        }
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
        if level == "L3" and plan_family == "sku_suite":
            model_overrides = {"layers": 1}
    elif plan_family == "sku_suite":
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["rmsnorm"]
        sim_fam, model_name = "rmsnorm", _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val, "width": _MLA_HIDDEN_WIDTH,
            "batch_size": batch, "sequences": batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo,
        }
    elif plan_family == "attention":
        pfam, prec, pbld, impl = ("attention", "fused_infer_attention", "fused_infer_attention", "aclnnFusedInferAttentionScoreV3")
        sim_fam, model_name = "attention", _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        kv_len = ctx_len + q_tok
        seqs = val if axis == "batch_x_kv_heads" else batch
        shape = {
            "tokens": max(1, seqs * q_tok), "sequences": seqs, "batch_size": batch,
            "query_tokens": q_tok, "q_len": q_tok,
            "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "page_count": _page_count(kv_len),
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _DEFAULT_HEAD_DIM, "block_size": _MLA_BLOCK_SIZE,
            "paged": True, "query_layout": "TND", "input_layout": "TND", "parallel_topology": topo,
        }
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
    elif plan_family in _FAMILY_BINDINGS:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS[plan_family]
        sim_fam = pfam
        model_name = _INDEXER_SIM_MODEL if sim_fam in {"indexer", "layernorm", "rotary"} else _DEFAULT_SIM_MODEL
        width = _MLA_HIDDEN_WIDTH if model_name == _DEFAULT_SIM_MODEL else _DEFAULT_HIDDEN_WIDTH
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        tok_val = val if axis == "tokens" else max(1, seqs * q_tok)
        shape = {
            "tokens": tok_val, "width": width, "m": tok_val, "n": width, "k": width,
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _DEFAULT_HEAD_DIM, "seq_len": ctx_len, "context_len": ctx_len,
            "batch_size": batch, "sequences": seqs,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }
    else:
        pfam, prec, pbld = plan_family, plan_family, ""
        impl, sim_fam, model_name = str(plan_doc.get("target_id") or plan_family), plan_family, _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val if axis == "tokens" else max(1, batch * q_tok),
            "width": _MLA_HIDDEN_WIDTH, "batch_size": batch, "sequences": batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }

    if isinstance(custom_sim, Mapping):
        sim_case = dict(custom_sim)
    else:
        sim_case = {"compose_model": model_name, "compose_hardware": sku, "compose_options": topo_opts}
        if model_overrides:
            sim_case["model_overrides"] = dict(model_overrides)
    return pfam, prec, pbld, impl, sim_fam, shape, sim_case


def _build_case_nodes_edges_and_bindings(
    pid: str,
    *,
    sku: str,
    pt: Mapping[str, Any],
    level: str,
    pfam: str,
    prec: str,
    pbld: str,
    impl: str,
    shape: Mapping[str, Any],
    topo_opts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Build exact profiler nodes, edges, per-node bindings, and timed op_ids."""
    is_paged_attn = pbld in {"absorbed_mla", "fused_infer_attention"}
    total_tokens = max(1, int(shape.get("tokens", 1)))
    tp = max(1, int(topo_opts.get("tp", 1)))
    local_heads = max(1, int(shape.get("heads", _DEFAULT_ATTENTION_HEADS // tp)))

    if level == "L1" or pbld != "absorbed_mla":
        node: dict[str, Any] = {
            "node_id": f"{pid}:op0", "kind": "operator",
            "op": pbld or prec, "family": pfam, "recipe": prec,
            "implementation": impl, "dtype": "bf16", "shape": dict(shape),
        }
        if pbld:
            node["builder"] = pbld
        if is_paged_attn:
            node["paged"] = True
        if shape.get("query_layout"):
            node["layout"] = str(shape["query_layout"])
        return [node], [], [], []

    if level == "L2":
        attn_op_id = (
            "s0.l0.paged_mla_decode"
            if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"}
            else "s0.l0.latent_attention0"
        )
        attn_node: dict[str, Any] = {
            "node_id": f"{pid}:{attn_op_id}", "op_id": attn_op_id, "kind": "operator",
            "op": pbld, "builder": pbld, "family": pfam, "recipe": prec,
            "implementation": impl, "dtype": "bf16", "paged": True,
            "layout": str(shape.get("query_layout") or "TND"), "shape": dict(shape),
        }
        v_exp_m = total_tokens * local_heads
        v_exp_node: dict[str, Any] = {
            "node_id": f"{pid}:s0.l0.v_expand", "op_id": "s0.l0.v_expand", "kind": "operator",
            "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm",
            "recipe": "grouped_matmul", "implementation": "aclnnGroupedMatmulV5",
            "dtype": "bf16", "group_list": [total_tokens] * local_heads, "group_list_type": 1,
            "shape": {
                "m": v_exp_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM,
                "e": local_heads, "matrix_groups": local_heads,
            },
        }
        edges = [{"from": f"{pid}:{attn_op_id}", "to": f"{pid}:s0.l0.v_expand", "kind": "data"}]
        node_bindings = [
            {
                "op_id": attn_op_id, "family": "attention", "recipe": prec, "builder": pbld,
                "paged": True, "heads": local_heads, "kv_heads": 1, "latent": _MLA_LATENT_DIM,
                "rope": _MLA_ROPE_DIM, "kv_len": int(shape["seq_len"]),
                "page_count": int(shape["page_count"]),
                "query_layout": str(shape.get("query_layout") or "TND"),
            },
            {
                "op_id": "s0.l0.v_expand", "family": "gemm", "recipe": "v_expand",
                "builder": "grouped_matmul", "m": v_exp_m, "n": _MLA_HEAD_DIM,
                "k": _MLA_LATENT_DIM, "matrix_groups": local_heads,
            },
        ]
        return [attn_node, v_exp_node], edges, node_bindings, [attn_op_id, "s0.l0.v_expand"]

    topo = str(pt.get("parallel_topology") or "tp1")
    q_tok = max(1, int(pt.get("query_tokens", 1)))
    ctx_len = int(shape.get("context_len", pt["value"]))
    seqs = max(1, int(shape.get("sequences", pt.get("batch_size", 1))))
    phase = "prefill" if q_tok > 1 else "decode"
    lowered = _lower_one_layer_step_ops(sku, topo, phase, q_tok, ctx_len, seqs)
    op_ids = {rec["op_id"] for rec in lowered}
    nodes: list[dict[str, Any]] = []
    edges = []
    node_bindings = []
    timed_op_ids: list[str] = []
    for rec in lowered:
        nd, nb = _lowered_op_to_profiler_node(
            pid, rec, attn_shape=shape, total_tokens=total_tokens,
            local_heads=local_heads, topo_opts=topo_opts,
        )
        nodes.append(nd)
        node_bindings.append(nb)
        timed_op_ids.append(rec["op_id"])
        for dep in rec["deps"]:
            if dep in op_ids:
                edges.append({"from": f"{pid}:{dep}", "to": f"{pid}:{rec['op_id']}", "kind": "stream"})
    return nodes, edges, node_bindings, timed_op_ids


def resolve_point_measured_binding(
    plan_doc: Mapping[str, Any],
    pt: Mapping[str, Any],
    *,
    sku: str,
) -> dict[str, Any]:
    """Build exact 1:1 replay target & per-node measured-operator binding metadata for ``pt``."""
    pfam, prec, pbld, impl, sim_fam, shape, sim_case = resolve_point_binding_and_shape(
        plan_doc, pt, sku=sku
    )
    level = assign_plan_point_level(plan_doc, pt)
    topo_opts = parse_topo_options(str(pt.get("parallel_topology") or "tp1"))
    pid = str(pt.get("point_id") or "pt")
    if sim_fam == "attention":
        target_op_id = (
            "s0.l0.paged_mla_decode"
            if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"}
            else "s0.l0.latent_attention0"
        )
    elif sim_fam == "rmsnorm":
        target_op_id = "s0.l0.norm1"
    elif sim_fam == "indexer":
        target_op_id = "s0.l2.index_quant_lightning"
    else:
        target_op_id = ""
    _nodes, _edges, node_bindings, timed_op_ids = _build_case_nodes_edges_and_bindings(
        pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl,
        shape=shape, topo_opts=topo_opts,
    )
    target_op_ids = list(timed_op_ids) if level == "L2" else []
    binding: dict[str, Any] = {
        "builder": pbld, "family": sim_fam, "recipe": prec, "implementation": impl,
        "target_op_id": target_op_id,
    }
    for key in (
        "paged", "heads", "kv_heads", "head_dim", "latent", "rope", "width",
        "kv_len", "page_count", "query_layout", "input_layout",
        "query_dtype", "cache_dtype", "sparse_count", "topk",
    ):
        if key in shape:
            binding[key] = shape[key]
    if node_bindings:
        binding["node_bindings"] = node_bindings
    if timed_op_ids:
        binding["timed_op_ids"] = timed_op_ids
    return {
        "level": level,
        "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
        "target_family": sim_fam,
        "target_op_id": target_op_id,
        "target_op_ids": target_op_ids,
        "measured_binding": binding,
        "sim_case": sim_case,
        "expected_shape": shape,
        "profiler_family": pfam,
        "profiler_recipe": prec,
        "profiler_builder": pbld,
        "implementation": impl,
    }


def build_profiler_subgraph_plan(
    plan_doc: Mapping[str, Any], *, sku: str, split_tag: str = "", acquisition_id: str = ""
) -> dict[str, Any]:
    """Translate an onboarding cliff/SKU plan into executable L1/L2/L3 ``mindie-bench-subgraph-plan/1``."""
    from .cliff_planner import required_ranks_for_plan, validate_parallel_topology

    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        out = dict(plan_doc)
        out.setdefault("plan_id", str(plan_doc.get("plan_sha256") or plan_doc.get("plan_id") or "onboard"))
        out.setdefault("plan_sha256", str(plan_doc.get("plan_sha256") or ""))
        out.setdefault("sku", sku)
        if acquisition_id:
            out["acquisition_id"] = acquisition_id
        return out

    world = required_ranks_for_plan(plan_doc)
    plan_sha = str(plan_doc["plan_sha256"])
    plan_id = f"{plan_sha}:{split_tag}" if split_tag else plan_sha
    cases: list[dict[str, Any]] = []
    for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
        for pt in plan_doc.get(split_key) or ():
            if not isinstance(pt, Mapping) or not pt.get("point_id"):
                continue
            pid = str(pt["point_id"])
            axis = str(pt.get("axis") or "context_len").strip()
            topo = str(pt.get("parallel_topology") or "tp1")
            pt_ranks = validate_parallel_topology(topo)
            topo_opts = parse_topo_options(topo)
            pfam, prec, pbld, impl, _sim_fam, shape, _sim_case = resolve_point_binding_and_shape(
                plan_doc, pt, sku=sku
            )
            level = assign_plan_point_level(plan_doc, pt)
            nodes, edges, _nb, _tids = _build_case_nodes_edges_and_bindings(
                pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl,
                shape=shape, topo_opts=topo_opts,
            )
            case_entry: dict[str, Any] = {
                "case_id": pid, "split": str(pt.get("split") or "heldout"),
                "level": level, "boundary": _LEVEL_BOUNDARY_KINDS[level],
                "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
                "archetype": str(pt.get("regime_tag") or "onboarding"),
                "stage": "s1_joint" if pt_ranks > 1 else "s0_local",
                "clock_operator": str(pt.get("clock_operator") or "elapsed_span"),
                "implementation": impl, "dtype": "bf16", "shape": dict(shape),
                "batch": int(shape.get("sequences", pt.get("batch_size", 1))),
                "plan_key": {
                    "point_id": pid, "axis": axis, "family": pfam, "recipe": prec,
                    "abi": impl, "parallel_topology": topo, "level": level,
                    "batch_size": int(pt.get("batch_size", 1)),
                },
                "nodes": nodes, "edges": edges,
            }
            if pbld in {"absorbed_mla", "fused_infer_attention"}:
                case_entry["paged"] = True
            if shape.get("query_layout"):
                case_entry["layout"] = str(shape["query_layout"])
            if world > 1:
                case_entry["group_ranks"] = list(range(pt_ranks))
            if pt_ranks > 1:
                case_entry["parallel"] = topo_opts
            cases.append(case_entry)
    plan_out: dict[str, Any] = {
        "schema": _SUBGRAPH_PLAN_SCHEMA, "plan_id": plan_id, "plan_sha256": plan_sha,
        "target_id": str(plan_doc.get("target_id") or ""), "sku": sku,
        "stage": "s1_joint" if world > 1 else "s0_local", "world_size": world,
        "execution_modes": ["eager"], "cases": cases,
    }
    if acquisition_id:
        plan_out["acquisition_id"] = acquisition_id
    return plan_out


def verify_actual_geometry_matches_plan(
    actual_shape: Mapping[str, Any],
    plan_pt: Mapping[str, Any],
    *,
    expected_shape: Mapping[str, Any],
    cid: str,
) -> None:
    """Compare receipt/builder-reported executed geometry against the plan BEFORE sealing."""
    if not actual_shape:
        return
    axis = str(plan_pt.get("axis") or "").strip()
    exp_val = int(plan_pt["value"])
    exp_q = max(1, int(plan_pt.get("query_tokens", 1)))
    exp_seqs = _int_from(expected_shape, "sequences")
    exp_tokens = _int_from(expected_shape, "tokens")

    got_q = _unique_alias_int(actual_shape, ("q_len", "query_tokens"), cid=cid, field="q_len")
    if got_q is not None and got_q != exp_q:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:q_len:expected={exp_q}:actual={got_q}")

    got_seqs = _int_from(actual_shape, "sequences")
    if exp_seqs is not None and got_seqs is not None and got_seqs != exp_seqs:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:sequences:expected={exp_seqs}:actual={got_seqs}")

    got_toks = _int_from(actual_shape, "tokens")
    if exp_tokens is not None and got_toks is not None and got_toks != exp_tokens:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:tokens:expected={exp_tokens}:actual={got_toks}")

    exp_seq_len = _int_from(expected_shape, "seq_len", "kv_len")
    got_seq_len = _int_from(actual_shape, "seq_len", "kv_len")
    if exp_seq_len is not None and got_seq_len is not None and got_seq_len != exp_seq_len:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:seq_len:expected={exp_seq_len}:actual={got_seq_len}")
    if axis == "context_len":
        got_ctx = _int_from(actual_shape, "context_len")
        if got_ctx is not None and got_ctx != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:context_len:expected={exp_val}:actual={got_ctx}")

    exp_pages = _int_from(expected_shape, "page_count")
    explicit_pages = _int_from(actual_shape, "page_count")
    bt = actual_shape.get("block_table")
    bt_pages: int | None = None
    if isinstance(bt, Sequence) and not isinstance(bt, (str, bytes)) and len(bt) >= _BLOCK_TABLE_PAIR_LEN:
        if not isinstance(bt[1], bool) and isinstance(bt[1], int):
            bt_pages = int(bt[1])
    if explicit_pages is not None and bt_pages is not None and explicit_pages != bt_pages:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:page_count_block_table_conflict"
            f":page_count={explicit_pages}:block_table_pages={bt_pages}"
        )
    for got_pages in (explicit_pages, bt_pages):
        if exp_pages is not None and got_pages is not None and got_pages != exp_pages:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:page_count:expected={exp_pages}:actual={got_pages}"
            )

    if axis == "batch_x_kv_heads":
        got_bxkv = _int_from(actual_shape, "batch_x_kv_heads")
        got_kv = _int_from(actual_shape, "kv_heads")
        if got_bxkv is not None and got_bxkv != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_bxkv}")
        if got_seqs is not None and got_kv is not None and got_seqs * got_kv != exp_val:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_seqs * got_kv}")

    got_tk = _unique_alias_int(actual_shape, ("topk", "top_k", "sparse_count"), cid=cid, field="topk")
    if plan_pt.get("topk") is not None and got_tk is not None and got_tk != int(plan_pt["topk"]):
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:top_k:expected={plan_pt['topk']}:actual={got_tk}")

    exp_layout = str(expected_shape.get("query_layout") or expected_shape.get("input_layout") or "").strip().upper()
    got_layout = str(actual_shape.get("query_layout") or actual_shape.get("input_layout") or "").strip().upper()
    if exp_layout and got_layout and got_layout != exp_layout:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:query_layout:expected={exp_layout}:actual={got_layout}")

    for field in ("kv_heads", "heads", "head_dim", "latent", "width"):
        exp_f = _int_from(expected_shape, field)
        got_f = _int_from(actual_shape, field)
        if exp_f is not None and got_f is not None and got_f != exp_f:
            raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:{field}:expected={exp_f}:actual={got_f}")


def derive_writer_clock_operator(
    group: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    cid: str,
    tmode: str,
) -> str:
    """Derive clock operator strictly from profiler writer timing evidence (never plan)."""
    explicit: set[str] = set()
    for row in group:
        exec_map = row.get("execution") if isinstance(row.get("execution"), Mapping) else {}
        ident_map = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
        ident_exec = ident_map.get("execution") if isinstance(ident_map.get("execution"), Mapping) else {}
        for cand in (
            row.get("clock_operator"), exec_map.get("clock_operator"),
            ident_map.get("clock_operator"), ident_exec.get("clock_operator"),
            summary.get("clock_operator"),
        ):
            val = str(cand or "").strip()
            if val:
                if val not in VALID_CLOCK_OPERATORS:
                    raise ModelingError(f"profiler_writer_clock_operator_invalid:{cid}:{val}")
                explicit.add(val)
    if len(explicit) > 1:
        raise ModelingError(f"profiler_writer_clock_operator_conflict:{cid}:{sorted(explicit)}")
    if explicit:
        return next(iter(explicit))
    boundary_name = str(group[0].get("boundary") or "device_kernel").strip().lower()
    mode_low = str(tmode or "").strip().lower()
    if boundary_name in _BUSY_UNION_BOUNDARIES or mode_low in _BUSY_UNION_TIMING_MODES:
        return "busy_union"
    if mode_low in _ELAPSED_SPAN_TIMING_MODES:
        return "elapsed_span"
    return "busy_union" if boundary_name == "device_kernel" else "elapsed_span"


def extract_observed_shape(
    summary: Mapping[str, Any],
    first: Mapping[str, Any],
    comp: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    *,
    allow_requested_fallback: bool = True,
) -> dict[str, Any]:
    """Return the richest non-placeholder receipt/builder-reported executed shape."""
    for rcpt in receipts:
        if isinstance(rcpt, Mapping):
            for key in ("actual_shape", "executed_shape"):
                cand = rcpt.get(key)
                if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
                    return dict(cand)
    comp_map = comp if isinstance(comp, Mapping) else {}
    writer_cands: list[Any] = [
        comp_map.get("actual_shape"), summary.get("actual_shape"), first.get("actual_shape"),
        comp_map.get("executed_shape"), summary.get("executed_shape"), first.get("executed_shape"),
    ]
    if allow_requested_fallback:
        ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
        key_map = first.get("key") if isinstance(first.get("key"), Mapping) else {}
        writer_cands.extend([
            summary.get("requested_shape"), first.get("requested_shape"),
            ident.get("shape"), key_map.get("shape"),
        ])
    for cand in writer_cands:
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            return dict(cand)
    return {}


def extract_abi_attempted_shape(
    rec: Mapping[str, Any], planned_pt: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Extract writer-reported attempted geometry for an ABI rejection; reject ``requested_shape``-only & alias conflicts."""
    cid = str(rec.get("case_id") or rec.get("point_id") or "abi").strip()
    hint = rec.get("unmodeled_hint") if isinstance(rec.get("unmodeled_hint"), Mapping) else {}
    raw: Mapping[str, Any] | None = None
    for cand in (
        rec.get("executed_shape"), rec.get("attempted_shape"), rec.get("actual_shape"),
        hint.get("executed_shape"), hint.get("actual_shape"),
    ):
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            raw = cand
            break
    if raw is None:
        return None
    out = dict(raw)
    axis = str((planned_pt or {}).get("axis") or "").strip()
    tk = _unique_alias_int(out, ("topk", "top_k", "sparse_count"), cid=cid, field="topk")
    if tk is not None:
        out["topk"] = tk
    q_val = _unique_alias_int(out, ("query_tokens", "q_len"), cid=cid, field="query_tokens")
    if q_val is not None:
        out["query_tokens"] = q_val
    ctx_val = _unique_alias_int(out, ("context_len",), cid=cid, field="context_len")
    if ctx_val is None:
        ctx_val = _unique_alias_int(out, ("seq_len",), cid=cid, field="context_len")
    if ctx_val is not None:
        out["context_len"] = ctx_val
    if axis != "batch_x_kv_heads":
        b_val = _unique_alias_int(out, ("batch_size", "sequences"), cid=cid, field="batch_size")
        if b_val is not None:
            out["batch_size"] = b_val
    return out


def select_mode_candidates(
    candidates: Sequence[tuple[str, str, str, list[Mapping[str, Any]]]],
    *,
    cid: str,
    plan_pt: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, list[Mapping[str, Any]]]]:
    """Filter multi-mode groups to the plan's clock/timing mode; fail closed on mismatch."""
    plan_clk = str((plan_pt or {}).get("clock_operator") or "").strip()
    if not plan_clk:
        return list(candidates)
    matched = [item for item in candidates if item[2] == plan_clk]
    if not matched:
        found = sorted({item[2] for item in candidates})
        raise ModelingError(
            f"clock_operator_plan_mismatch:{cid}:plan={plan_clk}:measured={','.join(found)}"
        )
    if len(matched) == 1:
        return matched
    pref_mode = "eager" if plan_clk == "elapsed_span" else "stream_saturated"
    scored = sorted(
        matched,
        key=lambda it: (0 if it[0] == "base" else 1, 0 if it[1].lower() == pref_mode else 1, it[0], it[1]),
    )
    return [scored[0]]


__all__ = [
    "assign_plan_point_level",
    "build_profiler_subgraph_plan",
    "derive_writer_clock_operator",
    "extract_abi_attempted_shape",
    "extract_observed_shape",
    "extract_shape_dims",
    "parse_run_id_and_arm",
    "parse_topo_options",
    "quantile_ns",
    "resolve_point_binding_and_shape",
    "resolve_point_measured_binding",
    "select_mode_candidates",
    "verify_actual_geometry_matches_plan",
]
