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

from collections.abc import Mapping
from typing import Any

from ._profiler_geometry import (
    _norm_timed_op_id,
    derive_writer_clock_operator,
    extract_abi_attempted_shape,
    extract_observed_shape,
    extract_shape_dims,
    parse_run_id_and_arm,
    quantile_ns,
    select_mode_candidates,
    validate_timed_nodes_sequence,
    verify_actual_geometry_matches_plan,
)
from ._profiler_lowering import (
    _MIN_2D_SHAPE_RANK,
    _MLA_HIDDEN_WIDTH,
    _decode_mla_recipe_for_sku,
    _lower_one_layer_step_ops,
    _lowered_op_to_profiler_node,
    _page_count,
    parse_topo_options,
)

_DEFAULT_ATTENTION_HEADS = 16
_INDEXER_ATTENTION_HEADS = 32
_DEFAULT_ATTENTION_KV_HEADS = 1
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_HEAD_DIM = 128
_MLA_BLOCK_SIZE = 128
_MOE_TOPK_EXPERTS = 6
_MOE_EXPERT_BANK = 64
_MOE_GATE_UP_WIDTH = 2816
_DEFAULT_HEAD_DIM = 128
_DEFAULT_HIDDEN_WIDTH = 4096
_DEFAULT_CONTEXT_FALLBACK = 2048
_L2_CONTEXT_MIN = 24576
_L3_CONTEXT_MIN = 200000
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_INDEXER_SIM_MODEL = "deepseek-v4.1-flash"
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_L2_REGIME_TAGS = frozenset({"high_batch_decode_2d", "mtp_decode_2d", "chunked_prefill_2d"})
_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
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


def _resolve_bound_family_shape(
    plan_family: str,
    *,
    axis: str,
    val: int,
    batch: int,
    q_tok: int,
    topo: str,
    topo_opts: Mapping[str, Any],
    local_mla_heads: int,
    topk_val: int | None,
) -> tuple[str, str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    pfam, prec, pbld, impl = _FAMILY_BINDINGS[plan_family]
    sim_fam = pfam
    model_name = _INDEXER_SIM_MODEL if sim_fam in {"indexer", "layernorm", "rotary"} else _DEFAULT_SIM_MODEL
    width = _MLA_HIDDEN_WIDTH if model_name == _DEFAULT_SIM_MODEL else _DEFAULT_HIDDEN_WIDTH
    seqs = val if axis == "batch_x_kv_heads" else batch
    tok_val = val if axis == "tokens" else max(1, seqs * q_tok)
    tp = max(1, int(topo_opts.get("tp", 1)))
    ep = max(1, int(topo_opts.get("ep", 1)))
    eff_opts = dict(topo_opts)
    if pbld == "matmul":
        shape: dict[str, Any] = {
            "tokens": tok_val, "m": tok_val, "n": width, "k": max(1, width // tp),
            "dtype": "bf16", "parallel_topology": topo,
        }
        eff_opts["graph_capture_mode"] = "none"
    elif pbld == "grouped_matmul":
        local_experts = max(1, _MOE_EXPERT_BANK // ep)
        shape = {
            "tokens": tok_val, "m": tok_val * _MOE_TOPK_EXPERTS,
            "n": max(1, _MOE_GATE_UP_WIDTH // tp), "k": width,
            "e": local_experts, "matrix_groups": local_experts,
            "dtype": "bf16", "parallel_topology": topo,
        }
        eff_opts["graph_capture_mode"] = "none"
    elif pbld == "rotary_embedding":
        shape = {"tokens": tok_val, "heads": local_mla_heads, "head_dim": _DEFAULT_HEAD_DIM, "parallel_topology": topo}
    elif pbld == "indexer_topk":
        shape = {"tokens": tok_val, "width": width, "top_k": topk_val or _MLA_BLOCK_SIZE, "parallel_topology": topo}
    else:
        shape = {"tokens": tok_val, "width": width, "parallel_topology": topo}
    return pfam, prec, pbld, impl, sim_fam, model_name, shape, eff_opts


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any],
    pt: Mapping[str, Any],
    *,
    sku: str,
    profile_override: Mapping[str, Any] | None = None,
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
    decode_mla_rec = _decode_mla_recipe_for_sku(sku, plan_doc=plan_doc, profile_override=profile_override)
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
            "tokens": max(1, batch * q_tok),
            "heads": _INDEXER_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM,
            "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len,
            "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok,
            "query_dtype": "bf16", "cache_dtype": "bf16", "input_layout": "TND", "query_layout": "TND",
            "parallel_topology": topo,
        }
        if topk_val is not None:
            shape["top_k"] = topk_val
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
            lowered = _lower_one_layer_step_ops(
                sku, topo, "prefill" if q_tok > 1 else "decode", q_tok, ctx_len, seqs,
                profile_override=profile_override,
            )
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
            "parallel_topology": topo,
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
        pfam, prec, pbld, impl, sim_fam, model_name, shape, topo_opts = _resolve_bound_family_shape(
            plan_family, axis=axis, val=val, batch=batch, q_tok=q_tok,
            topo=topo, topo_opts=topo_opts, local_mla_heads=local_mla_heads, topk_val=topk_val,
        )
    else:
        pfam, prec, pbld = plan_family, plan_family, ""
        impl, sim_fam, model_name = str(plan_doc.get("target_id") or plan_family), plan_family, _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val if axis == "tokens" else max(1, batch * q_tok),
            "width": _MLA_HIDDEN_WIDTH, "parallel_topology": topo,
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
                "e": local_heads, "matrix_groups": local_heads, "dtype": "bf16",
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
                "builder": "grouped_matmul", "dtype": "bf16", "quant_mode": "none",
                "m": v_exp_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM,
                "matrix_groups": local_heads,
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
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build exact 1:1 replay target & per-node measured-operator binding metadata for ``pt``."""
    from ._sku_calibration import temporary_scaffold_profile

    with temporary_scaffold_profile(sku, profile_override):
        pfam, prec, pbld, impl, sim_fam, shape, sim_case = resolve_point_binding_and_shape(
            plan_doc, pt, sku=sku, profile_override=profile_override
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
        elif pbld == "grouped_matmul":
            target_op_id = "s0.l1.expert_up"
        elif pbld == "matmul":
            target_op_id = "s0.l0.output"
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
        "dtype", "m", "n", "k", "matrix_groups",
    ):
        if key in shape:
            binding[key] = shape[key]
    if pbld in {"matmul", "grouped_matmul"} or sim_fam in {"gemm", "grouped_gemm", "gate_up_projection", "down_projection"}:
        binding.setdefault("dtype", str(shape.get("dtype") or "bf16"))
        binding.setdefault("weight_dtype", str(shape.get("weight_dtype") or binding["dtype"]))
        binding.setdefault("quant_mode", str(shape.get("quant_mode") or "none"))
        binding.setdefault("matrix_groups", int(shape.get("matrix_groups") or shape.get("e") or 1))
    if node_bindings:
        binding["node_bindings"] = node_bindings
    if timed_op_ids:
        binding["timed_op_ids"] = timed_op_ids
    if _edges:
        binding["timed_edges"] = [
            [_norm_timed_op_id(str(e["from"]), pid), _norm_timed_op_id(str(e["to"]), pid)]
            for e in _edges
            if isinstance(e, Mapping) and e.get("from") and e.get("to")
        ]
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
    plan_doc: Mapping[str, Any],
    *,
    sku: str,
    split_tag: str = "",
    acquisition_id: str = "",
    profile_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate an onboarding cliff/SKU plan into executable L1/L2/L3 ``mindie-bench-subgraph-plan/1``."""
    from ._sku_calibration import temporary_scaffold_profile
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
    with temporary_scaffold_profile(sku, profile_override):
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
                    plan_doc, pt, sku=sku, profile_override=profile_override
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
    "validate_timed_nodes_sequence",
    "verify_actual_geometry_matches_plan",
]
