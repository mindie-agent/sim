"""Before-vs-After operator alignment, L1/L2 breakdown, Sheet 4 cards & Agent Brief.

[Role] Compare lowered ``ExportBundle`` snapshots to produce:
  (1) strictly topological, scope-preserving ``(Before -> After)`` 1->1 / N->1
      fusion / eliminated / added rows for Sheet 2 (``Layer_Operator_Breakdown``),
  (2) two-level L1/L2 macro-module breakdown with explicit ``Raw / Masked /
      Exposed`` columns and 100% Step Latency conservation for Sheet 1 Block 3,
  (3) 100%-conserved Sheet 4 ``Pending_Kernel_Details`` cards pre-populated with
      concrete 3-track Ascend guidance (``Config & Env``, ``Kernel``,
      ``Framework``) and stable ``card_id`` slugs, and
  (4) turnkey Agent-native exports (``build_agent_brief``, ``export_agent_brief``,
      ``format_executive_digest_markdown``) sized ``< 8 KB``.
[Invariants] Pure functional transformations over immutable ``ExportBundle`` rows;
  zero engine execution, zero OOXML formatting, zero ``[待 Agent 补充]`` blanks,
  zero ``"4D"`` jargon, and exact conservation:
  ``Sheet 1 Exposed Δ == Sum(Sheet 2 Exposed Δ) == Sum(Sheet 4 Cards Δ)``.
[Timeouts & Locks] O(Unique Archetype Ops) in-memory passes; atomic ``os.replace``
  only inside ``export_agent_brief``; no locks or device contexts.
[Side-Effect Policy] Pure in-memory except ``export_agent_brief`` which writes
  only its caller-requested JSON path.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .kernel_catalog import classify_macro_module, strip_op_prefix
from .models import (
    AGENT_BRIEF_SCHEMA,
    DEFAULT_TOP_OPTIMIZE_KERNELS,
    GIB_BYTES,
    L1_MOD_RESIDUAL_MEM,
    MACRO_L1_HIERARCHY,
    MACRO_MOD_DRAFT_LAYERS,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_VERIFY_HEAD,
    MAX_SOURCE_REF_SUMMARY_LEN,
    OVERLAP_EXPOSED_EPS_US,
    PENDING_KIND_TO_IMPLEMENT,
    PENDING_KIND_TO_OPTIMIZE,
    US_PER_MS,
    AlignedComparisonRow,
    ExportBundle,
    MacroModuleRow,
    NewKernelBacklogItem,
    OperatorExportRow,
    PendingKernelItem,
)
from .target_envelope import build_actionable_roadmap, build_parallelism_codesign_note

#: Epsilon (ms) for treating an exposed Before->After delta as non-zero.
CONSERVATION_EPS_MS = 1e-6
#: Maximum number of compact kernel tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_KERNEL_TASKS = 6
#: Maximum number of framework tasks embedded inside ``agent_brief.json``.
BRIEF_MAX_FRAMEWORK_TASKS = 3
#: Maximum character length for compact string fields inside ``agent_brief.json``.
BRIEF_TEXT_LIMIT = 80
#: Minimum single-op gap (us) to call out gap-vs-L* in Sheet 2 notes.
ACTION_GAP_CALLOUT_US = 1.0
#: Byte budget for ``agent_brief.json`` (< 8 KB) and pretty-print threshold.
AGENT_BRIEF_MAX_BYTES = 8192
AGENT_BRIEF_PRETTY_LIMIT_BYTES = 6400

_KNOWN_FUSION_SOURCES: dict[str, tuple[str, ...]] = {
    "mla_prolog_v3": (
        "q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
        "q_norm", "q_b", "q_b.quantize", "kv_norm", "rotary", "q_rotary",
        "kv_rotary", "kv_store", "q_absorb",
    ),
    "mla_prolog": (
        "q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
        "q_norm", "q_b", "q_b.quantize", "kv_norm", "rotary", "q_rotary",
        "kv_rotary", "kv_store", "q_absorb",
    ),
    "megamoe_arch35": (
        "dispatch", "dispatch.received_quantize", "expert_up", "expert_act",
        "expert_down.quantize", "expert_down", "unpack",
    ),
    "dispatch_ffn_combine": (
        "dispatch", "dispatch.received_quantize", "expert_up", "expert_act",
        "expert_down.quantize", "expert_down", "unpack",
    ),
    "grouped_matmul_situ_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "grouped_matmul_swiglu_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "quant_lightning_indexer": ("index_query_quantize", "index_scores", "index_topk"),
    "fused_gdn_prologue": (
        "beta_contiguous", "alpha_contiguous", "core_output_zero",
        "convolution_input_contiguous", "short_convolution",
        "q_flat", "k_flat", "v_flat", "qkv_concatenate",
        "delta_gates", "decode_q_l2norm", "decode_k_l2norm",
    ),
    "fused_linear_output_epilogue": (
        "core_output_copy", "z_norm_materialize",
        "linear_output_norm_gate", "linear_output.quantize",
    ),
    "add_rmsnorm": ("residual", "norm1", "norm2", "final_norm"),
    "add_rmsnorm_quant": ("residual", "norm1", "norm2", "quantize"),
    "recurrent_scan": ("qk_l2norm", "decode_q_l2norm", "decode_k_l2norm", "delta_gates", "recurrent_scan"),
}
_MHC_SUFFIXES = ("hc_norm", "hc_down_inject", "hc_silu", "hc_up", "hc_gate_mix")
_SCOPE_PREFIXES = ("attention.", "ffn.", "shared.", "final.")
_OPTIMIZE_DESIGN_HINTS: dict[str, str] = {
    "link_bound": "切换超节点 AIV UB-Direct / HCCS One-Shot 集合通信原语与 MC2 通算融合，消除 Ring-HCCL 启动开销",
    "memory_bound": "紧凑动态 M-Tiling / L1-L2 权重驻留 / 算子融合降访存与量化压缩 (MXFP8+MXFP4)，逼近 HBM 带宽极限 L*",
    "cube_bound": "AIC 核内 M/N/K 分块与 NZ 布局对齐 / L0C Ping-Pong 双缓冲，提升 Cube AIC 利用率",
    "vector_bound": "AIV 向量算子融合与 UB 双缓冲流水编排，减少中间 HBM 回写与同步开销",
    "launch_bound": "开启 ACLGraph FULL_DECODE_ONLY 全图捕获与算子合并，消除 Host 发射气泡",
}
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _scope_of_suffix(suf: str) -> str:
    for prefix in _SCOPE_PREFIXES:
        if suf.startswith(prefix):
            return prefix
    return ""


def _op_exact_key(op: OperatorExportRow) -> str:
    return strip_op_prefix(op.op_id).lower()


def _op_canon_key(op: OperatorExportRow) -> str:
    return (op.canonical_op_key or strip_op_prefix(op.op_id)).lower()


def _fusion_absorbed_keys(after_op: OperatorExportRow) -> set[str]:
    keys = {strip_op_prefix(x).lower() for x in after_op.absorbed_source_ops if x}
    exact = _op_exact_key(after_op)
    canon = _op_canon_key(after_op)
    scope = _scope_of_suffix(exact)
    keys.update(_KNOWN_FUSION_SOURCES.get(canon, ()))
    keys.update(_KNOWN_FUSION_SOURCES.get(exact, ()))

    kern = (after_op.ascend_kernel or "").lower()
    fam = (after_op.family or "").lower()
    if "mlaprolog" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["mla_prolog_v3"])
    elif "megamoe" in kern or "dispatch_ffn_combine" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["megamoe_arch35"])
    elif "situquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_situ_quant"])
    elif "swigluquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_swiglu_quant"])
    elif "fusedgdnprologue" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["fused_gdn_prologue"])
    elif "fusedrmsnormgated" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["fused_linear_output_epilogue"])

    if fam == "mhc" or "hc_gate_mix" in exact:
        keys.update(f"{scope}{s}" for s in _MHC_SUFFIXES)

    if fam == "all_gather_matmul":
        if "shared." in exact:
            keys.add("shared.sp_gather")
        elif exact in {"gate_up", "up", "gate", "ffn.gate_up"}:
            keys.add("ffn.sp_gather")
        else:
            keys.add("attention.sp_gather")
    elif fam == "matmul_reduce_scatter":
        if "shared." in exact:
            keys.update({"shared.down", "shared.tp", "shared.tp.reduce_scatter"})
        elif "linear_output" in exact:
            keys.update({"linear_output", "linear_reduce", "linear_reduce.reduce_scatter"})
        elif exact == "output":
            keys.update({"output", "output_reduce", "output_reduce.reduce_scatter"})
        elif exact == "down":
            keys.update({"down", "tp", "tp.reduce_scatter"})
    return keys


def _build_action_note(
    relation: str,
    before_ops: Sequence[OperatorExportRow],
    after_ops: Sequence[OperatorExportRow],
) -> str:
    if relation == "N->1" and after_ops:
        a = after_ops[0]
        detail = a.kernel_action or a.framework_action or a.kernel_status
        return f"N->1 融合 ({len(before_ops)} ops -> {a.ascend_kernel}) | {detail}"
    if relation == "eliminated":
        return "优化后被融合/消除 (0 暴露耗时)"
    if relation == "added" and after_ops:
        a = after_ops[0]
        return f"优化后新增算子 ({a.logical_role}) | {a.kernel_action or a.kernel_status}"
    if after_ops:
        a = after_ops[0]
        if a.is_overlapped:
            masker = a.masked_by or "主计算流"
            return f"副流被 {masker} 100% 掩盖 (0 暴露，不计入总耗时)"
        if a.opt_pillar == "pillar3_comm":
            return f"配置+通信升级: {a.config_howto or a.kernel_status}"
        if a.opt_pillar == "pillar2_roofline":
            return f"算子调优 (目标 {a.opt_target_eta_pct:.0f}% L*): {a.kernel_action or a.kernel_status}"
        if a.needs_new_kernel and a.kernel_action:
            return f"{a.kernel_status} | {a.kernel_action}"
        if before_ops:
            b = before_ops[0]
            if b.ascend_kernel != a.ascend_kernel:
                return f"算子升级: {b.ascend_kernel} -> {a.ascend_kernel} | {a.config_howto or a.kernel_status}"
            if a.gap_us > ACTION_GAP_CALLOUT_US and a.l_star_lean_us > 0.0:
                return f"Gap vs L* ({a.l_star_lean_us:.1f}us vs {a.dur_us:.1f}us) | {a.kernel_action or a.bound_display}"
        return f"{a.bound_display} | {a.kernel_status}"
    return ""


def _scope_compatible(a_scope: str, b_scope: str) -> bool:
    if not a_scope:
        return True
    return b_scope in {"", a_scope}


def _align_archetype_topological(
    aid: str,
    title: str,
    rep: int,
    b_list: Sequence[OperatorExportRow],
    a_list: Sequence[OperatorExportRow],
) -> list[AlignedComparisonRow]:
    """Align one archetype in strict topological execution order without reordering submodules."""
    used_b: set[int] = set()
    used_a: set[int] = set()
    entries: list[tuple[float, int, AlignedComparisonRow]] = []
    anchor_b_for_a: dict[int, float] = {}

    b_exact_idx: dict[str, list[int]] = defaultdict(list)
    b_scoped_canon_idx: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for b_idx, b_op in enumerate(b_list):
        ex = _op_exact_key(b_op)
        b_exact_idx[ex].append(b_idx)
        b_scoped_canon_idx[(_scope_of_suffix(ex), b_op.submodule, _op_canon_key(b_op))].append(b_idx)

    # Pass 1: N->1 fusion matches (respecting scope prefix so attention.* never merges ffn.*).
    for a_idx, a_op in enumerate(a_list):
        a_ex = _op_exact_key(a_op)
        a_scope = _scope_of_suffix(a_ex)
        absorbed = _fusion_absorbed_keys(a_op)
        if not absorbed:
            continue
        matched_b: list[int] = []
        for b_idx, b_op in enumerate(b_list):
            if b_idx in used_b:
                continue
            b_ex = _op_exact_key(b_op)
            b_scope = _scope_of_suffix(b_ex)
            if not _scope_compatible(a_scope, b_scope):
                continue
            if b_ex in absorbed or (_op_canon_key(b_op) in absorbed and b_scope == a_scope):
                matched_b.append(b_idx)
        if not matched_b:
            continue
        has_foreign = any(_op_exact_key(b_list[i]) != a_ex for i in matched_b)
        if len(matched_b) > 1 or (has_foreign and a_ex not in b_exact_idx):
            for b_idx in b_exact_idx.get(a_ex, ()):
                if b_idx not in used_b and b_idx not in matched_b:
                    matched_b.append(b_idx)
            matched_b.sort()
            used_b.update(matched_b)
            used_a.add(a_idx)
            b_tuple = tuple(b_list[i] for i in matched_b)
            rel = "N->1" if len(b_tuple) > 1 else "1->1"
            anchor = float(matched_b[0])
            anchor_b_for_a[a_idx] = anchor
            entries.append((
                anchor, a_idx,
                AlignedComparisonRow(aid, title, rep, a_op.submodule, b_tuple, (a_op,), rel, _build_action_note(rel, b_tuple, (a_op,))),
            ))

    # Pass 2: 1->1 exact suffix match (preserves attention.attn_res_fwd vs ffn.attn_res_fwd).
    for a_idx, a_op in enumerate(a_list):
        if a_idx in used_a:
            continue
        a_ex = _op_exact_key(a_op)
        cand = next((i for i in b_exact_idx.get(a_ex, ()) if i not in used_b), None)
        if cand is None:
            key = (_scope_of_suffix(a_ex), a_op.submodule, _op_canon_key(a_op))
            cand = next((i for i in b_scoped_canon_idx.get(key, ()) if i not in used_b), None)
        if cand is not None:
            used_b.add(cand)
            used_a.add(a_idx)
            b_op = b_list[cand]
            anchor = float(cand)
            anchor_b_for_a[a_idx] = anchor
            entries.append((
                anchor, a_idx,
                AlignedComparisonRow(aid, title, rep, a_op.submodule, (b_op,), (a_op,), "1->1", _build_action_note("1->1", (b_op,), (a_op,))),
            ))

    # Pass 3: Eliminated before ops & topologically anchored added after ops.
    for b_idx, b_op in enumerate(b_list):
        if b_idx not in used_b:
            entries.append((
                float(b_idx), -1,
                AlignedComparisonRow(aid, title, rep, b_op.submodule, (b_op,), (), "eliminated", _build_action_note("eliminated", (b_op,), ())),
            ))
    last_anchor = -1.0
    for a_idx, a_op in enumerate(a_list):
        if a_idx in anchor_b_for_a:
            last_anchor = anchor_b_for_a[a_idx]
            continue
        pos = (last_anchor + 0.5) if last_anchor >= 0.0 else float(len(b_list) + a_idx)
        entries.append((
            pos, a_idx,
            AlignedComparisonRow(aid, title, rep, a_op.submodule, (), (a_op,), "added", _build_action_note("added", (), (a_op,))),
        ))

    entries.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in entries]


def align_before_after_operators(
    before_bundle: ExportBundle, after_bundle: ExportBundle,
) -> tuple[AlignedComparisonRow, ...]:
    """Align Before vs After operators in strict topological order per archetype."""
    arch_order: list[tuple[str, str, int]] = []
    seen_arch: set[str] = set()
    for src in (before_bundle, after_bundle):
        for op in src.operators:
            aid = op.archetype_id or op.layer_label
            if aid not in seen_arch:
                seen_arch.add(aid)
                arch_order.append((aid, op.archetype_title or op.layer_label, op.repeat_count))

    before_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    after_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    for op in before_bundle.operators:
        before_by_arch[op.archetype_id or op.layer_label].append(op)
    for op in after_bundle.operators:
        after_by_arch[op.archetype_id or op.layer_label].append(op)

    aligned: list[AlignedComparisonRow] = []
    for aid, title, rep in arch_order:
        aligned.extend(_align_archetype_topological(
            aid, title, rep, before_by_arch.get(aid, ()), after_by_arch.get(aid, ()),
        ))
    return tuple(aligned)


def _accumulate_macro_stats(bundle: ExportBundle) -> dict[str, dict[str, Any]]:
    arch_by_id = {a.archetype_id: a for a in bundle.archetypes}
    stats: dict[str, dict[str, Any]] = {}
    for op in bundle.operators:
        mod = classify_macro_module(op)
        bucket = stats.setdefault(mod, {
            "exposed_ms": 0.0,
            "raw_ms": 0.0,
            "lstar_ms": 0.0,
            "bound_ms": defaultdict(float),
            "masker_ms": defaultdict(float),
            "target_layer_archs": {},
            "draft_layers": 0,
            "has_head_or_embed": False,
        })
        rep = max(1, int(op.repeat_count))
        exp_ms = 0.0 if op.is_overlapped else (op.exposed_dur_us * rep / US_PER_MS)
        raw_ms = op.dur_us * rep / US_PER_MS
        masked_pos_ms = max(0.0, raw_ms - exp_ms)
        lstar_ms = 0.0 if op.is_overlapped else (op.l_star_lean_us * rep / US_PER_MS)
        bucket["exposed_ms"] += exp_ms
        bucket["raw_ms"] += raw_ms
        bucket["lstar_ms"] += lstar_ms
        if exp_ms > 0.0:
            bucket["bound_ms"][op.bound_display or op.dominant_bound] += exp_ms
        if masked_pos_ms > CONSERVATION_EPS_MS:
            bucket["masker_ms"][op.masked_by or "主计算流并发算子"] += masked_pos_ms
        arch = arch_by_id.get(op.archetype_id)
        if op.model_scope == "draft":
            if arch is not None and arch.layer_indices:
                bucket["draft_layers"] = max(bucket["draft_layers"], len(arch.layer_indices))
            else:
                bucket["has_head_or_embed"] = True
        elif arch is not None and arch.layer_indices:
            bucket["target_layer_archs"][op.archetype_id] = len(arch.layer_indices)
        else:
            bucket["has_head_or_embed"] = True
    return stats


def _format_macro_layer_label(
    mod: str,
    target_archs: Mapping[str, int],
    draft_layers: int,
    has_head: bool,
    bundle: ExportBundle,
) -> tuple[str, int]:
    t_layers = sum(target_archs.values())
    if mod == MACRO_MOD_DRAFT_LAYERS:
        d_cnt = draft_layers or bundle.total_draft_layers
        return (f"Draft {d_cnt}L", d_cnt) if d_cnt > 0 else ("Draft ×1", 1)
    if mod == MACRO_MOD_VERIFY_HEAD:
        return "Embed + Head (×1)", 1
    if t_layers > 0:
        return f"{t_layers}L", t_layers
    if has_head:
        return "×1", 1
    return "0L", 0


def _build_masking_note(
    mod: str,
    raw_aft: float,
    aft_ms: float,
    raw_bef: float,
    bef_ms: float,
    a_maskers: Mapping[str, float],
    b_maskers: Mapping[str, float],
    dom_bound: str,
) -> str:
    aft_masked = max(0.0, raw_aft - aft_ms)
    bef_masked = max(0.0, raw_bef - bef_ms)
    if aft_masked > CONSERVATION_EPS_MS:
        top_m = max(a_maskers.items(), key=lambda kv: kv[1])[0] if a_maskers else "主计算流"
        if aft_ms <= OVERLAP_EXPOSED_EPS_US:
            if mod == MACRO_MOD_MOE_SHARED:
                return f"MSO 副流被 {top_m} 100% 掩盖 (-{aft_masked:.2f}ms，不计入总耗时)"
            return f"被 {top_m} 100% 掩盖 (-{aft_masked:.2f}ms，不计入总耗时)"
        pct = 100.0 * aft_masked / max(raw_aft, CONSERVATION_EPS_MS)
        return f"{dom_bound} | 被 {top_m} 掩盖 {pct:.0f}% (-{aft_masked:.2f}ms)"
    if bef_masked > CONSERVATION_EPS_MS:
        top_m = max(b_maskers.items(), key=lambda kv: kv[1])[0] if b_maskers else "主计算流"
        return f"{dom_bound} | 优化前被 {top_m} 掩盖 -{bef_masked:.2f}ms"
    return dom_bound


def build_macro_module_breakdown(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
) -> tuple[MacroModuleRow, ...]:
    """Aggregate two-level L1/L2 Raw, Masked, and Exposed ms with 100% step conservation."""
    bef_stats = _accumulate_macro_stats(before_bundle)
    aft_bundle = after_bundle or before_bundle
    aft_stats = bef_stats if after_bundle is None else _accumulate_macro_stats(after_bundle)

    tot_bef = sum(v["exposed_ms"] for v in bef_stats.values())
    tot_aft = sum(v["exposed_ms"] for v in aft_stats.values())
    l1_by_mod = {mod: l1 for l1, mods in MACRO_L1_HIERARCHY for mod in mods}
    ordered_mods = [mod for _, mods in MACRO_L1_HIERARCHY for mod in mods]
    for extra in (*bef_stats.keys(), *aft_stats.keys()):
        if extra not in ordered_mods:
            ordered_mods.append(extra)

    rows: list[MacroModuleRow] = []
    for mod in ordered_mods:
        b_info = bef_stats.get(mod)
        a_info = aft_stats.get(mod)
        if b_info is None and a_info is None:
            continue
        bef_ms = float(b_info["exposed_ms"]) if b_info else 0.0
        aft_ms = float(a_info["exposed_ms"]) if a_info else 0.0
        raw_bef = float(b_info["raw_ms"]) if b_info else 0.0
        raw_aft = float(a_info["raw_ms"]) if a_info else 0.0
        lstar_ms = float(a_info["lstar_ms"]) if a_info else 0.0
        if raw_bef <= 0.0 and raw_aft <= 0.0:
            continue
        merged_archs = dict(b_info["target_layer_archs"]) if b_info else {}
        if a_info:
            merged_archs.update(a_info["target_layer_archs"])
        d_layers = max(
            int(b_info["draft_layers"]) if b_info else 0,
            int(a_info["draft_layers"]) if a_info else 0,
        )
        has_head = bool((b_info and b_info["has_head_or_embed"]) or (a_info and a_info["has_head_or_embed"]))
        layer_lbl, layer_cnt = _format_macro_layer_label(mod, merged_archs, d_layers, has_head, aft_bundle)
        bound_weights = a_info["bound_ms"] if (a_info and a_info["bound_ms"]) else (b_info["bound_ms"] if b_info else {})
        if mod == MACRO_MOD_MOE_SHARED and aft_ms <= OVERLAP_EXPOSED_EPS_US and raw_aft > 0.0:
            dom_bound = f"MSO Stream 1 100% 掩盖 (-{raw_aft:.3f}ms)"
        elif bound_weights:
            dom_bound = max(bound_weights.items(), key=lambda kv: kv[1])[0]
        else:
            dom_bound = "已融合 / 100% 掩盖"
        mask_note = _build_masking_note(
            mod, raw_aft, aft_ms, raw_bef, bef_ms,
            a_info["masker_ms"] if a_info else {},
            b_info["masker_ms"] if b_info else {},
            dom_bound,
        )
        rows.append(MacroModuleRow(
            module_name=mod,
            layer_calls_label=layer_lbl,
            layer_count=layer_cnt,
            before_ms=bef_ms,
            before_share_pct=(100.0 * bef_ms / tot_bef) if tot_bef > 0.0 else 0.0,
            after_ms=aft_ms,
            after_share_pct=(100.0 * aft_ms / tot_aft) if tot_aft > 0.0 else 0.0,
            saved_ms=bef_ms - aft_ms,
            lstar_ms=lstar_ms,
            dominant_bound=dom_bound,
            raw_before_ms=raw_bef,
            raw_after_ms=raw_aft,
            l1_group=l1_by_mod.get(mod, L1_MOD_RESIDUAL_MEM),
            masked_before_ms=bef_ms - raw_bef,
            masked_after_ms=aft_ms - raw_aft,
            masking_note=mask_note,
        ))
    return tuple(rows)


def _short_arch_name(title: str) -> str:
    return re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", title.split("—")[0].strip())


def _make_card_slug(raw: str, used: set[str]) -> str:
    base = _SLUG_RE.sub("_", raw.lower()).strip("_") or "kernel_card"
    slug = base[:44].rstrip("_")
    cand = slug
    idx = 2
    while cand in used:
        cand = f"{slug}_{idx}"
        idx += 1
    used.add(cand)
    return cand


def _classify_comparison_card_key(
    row: AlignedComparisonRow,
    saved_ms: float,
    *,
    is_comparison: bool,
    include_to_optimize: bool,
) -> tuple[str, str, OperatorExportRow | None] | None:
    if row.after_ops:
        a_op = row.after_ops[0]
        is_new = (
            a_op.needs_new_kernel
            or a_op.opt_pillar == "pillar1_fusion"
            or (row.relation == "N->1" and a_op.family in {"mhc", "attn_res_prefix"})
        )
        if is_new:
            return f"new::{a_op.ascend_kernel}", PENDING_KIND_TO_IMPLEMENT, a_op
        if not include_to_optimize:
            return None
        if is_comparison:
            if abs(saved_ms) <= CONSERVATION_EPS_MS:
                return None
            if a_op.is_overlapped and saved_ms > 0.0:
                return "opt::mso_stream_overlap_masking", PENDING_KIND_TO_OPTIMIZE, a_op
            return f"opt::{a_op.ascend_kernel}", PENDING_KIND_TO_OPTIMIZE, a_op
        if not a_op.is_overlapped and a_op.gap_us > 0.0:
            return f"opt::{a_op.ascend_kernel}", PENDING_KIND_TO_OPTIMIZE, a_op
        return None

    if is_comparison and include_to_optimize and row.before_ops and abs(saved_ms) > CONSERVATION_EPS_MS:
        b0 = row.before_ops[0]
        return f"opt::eliminated::{b0.submodule}", PENDING_KIND_TO_OPTIMIZE, None
    return None


def _accumulate_pending_group(
    grouped: dict[str, dict[str, Any]],
    *,
    key: str,
    category: str,
    row: AlignedComparisonRow,
    a_op: OperatorExportRow | None,
    after_bundle: ExportBundle,
    is_comparison: bool,
) -> None:
    rep = max(1, int(row.repeat_count))
    before_ms = sum(
        0.0 if b.is_overlapped else (b.exposed_dur_us * rep / US_PER_MS)
        for b in row.before_ops
    )
    after_ms = (
        0.0 if (a_op is None or a_op.is_overlapped)
        else (a_op.exposed_dur_us * rep / US_PER_MS)
    )
    lstar_ms = (
        0.0 if (a_op is None or a_op.is_overlapped)
        else (a_op.l_star_lean_us * rep / US_PER_MS)
    )
    before_lstar_ms = sum(
        0.0 if b.is_overlapped else (b.l_star_lean_us * rep / US_PER_MS)
        for b in row.before_ops
    )
    gap_ms = max(0.0, after_ms - lstar_ms)
    gain_ms = (before_ms - after_ms) if is_comparison else gap_ms

    ref_op = a_op or row.before_ops[0]
    bef_bounds = [b.bound_display or b.dominant_bound for b in row.before_ops if b.bound_display or b.dominant_bound]
    bef_bound_lbl = " / ".join(dict.fromkeys(bef_bounds)) or (ref_op.bound_display or ref_op.dominant_bound)
    is_mso_card = key == "opt::mso_stream_overlap_masking"
    is_elim_card = key.startswith("opt::eliminated::")
    card_kernel = (
        "MSO 副流异步掩盖 (Stream 1 Overlap)" if is_mso_card
        else (f"融合消除冗余算子 ({ref_op.submodule})" if is_elim_card else ref_op.ascend_kernel)
    )
    card_status = (
        "已有特效配置: MSO + MC2 加速后共享专家 100% 掩盖" if is_mso_card
        else ("优化后消除 (0 暴露)" if is_elim_card else ref_op.kernel_status)
    )

    entry = grouped.get(key)
    if entry is None:
        entry = {
            "kernel_key": key,
            "category": category,
            "target_kernel": card_kernel,
            "logical_role": ref_op.logical_role or ref_op.op_id,
            "roles": [],
            "submodule": ref_op.submodule,
            "backend": ref_op.kernel_backend or "CANN / AscendC",
            "status": card_status,
            "replaces": [],
            "sources": [],
            "archetypes": [],
            "calls": 0,
            "before_ms": 0.0,
            "after_ms": 0.0,
            "lstar_ms": 0.0,
            "before_lstar_ms": 0.0,
            "gap_ms": 0.0,
            "gain_ms": 0.0,
            "before_bound": bef_bound_lbl,
            "after_bound": "100% 掩盖 / 消除" if (is_mso_card or is_elim_card) else (ref_op.bound_display or ref_op.dominant_bound),
            "dom_key": ref_op.dominant_bound,
            "contract": ref_op.tensor_contract,
            "single_lstar_us": ref_op.l_star_lean_us,
            "single_sim_us": ref_op.dur_us,
            "die_summary": after_bundle.hardware_die_summary,
            "spec_note": ref_op.new_kernel_spec,
            "config_howto": ref_op.config_howto,
            "kernel_action": ref_op.kernel_action,
            "framework_action": ref_op.framework_action,
            "opt_pillar": ref_op.opt_pillar or ("pillar1_fusion" if category == PENDING_KIND_TO_IMPLEMENT else "pillar2_roofline"),
        }
        grouped[key] = entry
    for b_op in row.before_ops:
        entry["replaces"].append(b_op.ascend_kernel)
        if b_op.source_ref:
            entry["sources"].append(b_op.source_ref)
    if a_op is not None and a_op.source_ref:
        entry["sources"].append(a_op.source_ref)
    role_lbl = ref_op.logical_role or ref_op.op_id
    if role_lbl not in entry["roles"]:
        entry["roles"].append(role_lbl)
    entry["archetypes"].append(f"{_short_arch_name(row.archetype_title)} (×{rep})")
    entry["calls"] += rep
    entry["before_ms"] += before_ms
    entry["after_ms"] += after_ms
    entry["lstar_ms"] += lstar_ms
    entry["before_lstar_ms"] += before_lstar_ms
    entry["gap_ms"] += gap_ms
    entry["gain_ms"] += gain_ms
    if ref_op.dur_us > entry["single_sim_us"]:
        entry["single_sim_us"] = ref_op.dur_us
        entry["single_lstar_us"] = ref_op.l_star_lean_us
        entry["contract"] = ref_op.tensor_contract


def _materialize_pending_item(prio: int, entry: Mapping[str, Any], used_slugs: set[str]) -> PendingKernelItem:
    is_new = entry["category"] == PENDING_KIND_TO_IMPLEMENT
    replaces_txt = (
        " + ".join(dict.fromkeys(entry["replaces"])) or "New target kernel"
        if is_new else f"{entry['submodule']} -> {', '.join(entry['roles'][:3])}"
    )
    sources_txt = "; ".join(dict.fromkeys(entry["sources"]))[:MAX_SOURCE_REF_SUMMARY_LEN] or "-"
    arch_txt = "; ".join(dict.fromkeys(entry["archetypes"]))
    bef_ms = float(entry["before_ms"])
    aft_ms = float(entry["after_ms"])
    ls_ms = float(entry["lstar_ms"])
    b_ls = float(entry["before_lstar_ms"]) or ls_ms
    bef_att = min(100.0, 100.0 * b_ls / bef_ms) if bef_ms > 0.0 and b_ls > 0.0 else 0.0
    aft_att = min(100.0, 100.0 * ls_ms / aft_ms) if aft_ms > 0.0 and ls_ms > 0.0 else (100.0 if aft_ms <= 0.0 else 0.0)
    microarch = (
        f"Before: {entry['before_bound']} (L* 达成率 {bef_att:.1f}%) -> "
        f"Target: {entry['after_bound']} (目标 {aft_att:.1f}% L*) | "
        f"Single L*={entry['single_lstar_us']:.2f}us vs Target={entry['single_sim_us']:.2f}us | {entry['die_summary']}"
    )
    hint = _OPTIMIZE_DESIGN_HINTS.get(str(entry["dom_key"]), "Tiling / UB 双缓冲 / 访存优化")
    config_howto = str(entry["config_howto"] or "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'")
    kernel_action = str(entry["kernel_action"] or (entry["spec_note"] or hint))
    framework_action = str(entry["framework_action"] or "vllm-ascend 保持静态 shape 预分配 workspace 并接线 ACLGraph 捕获")
    accept_slot = (
        f"精度误差 <= 1e-3 (BF16/MXFP4/FP8)，端到端实测耗时 <= {aft_ms:.3f}ms "
        f"(L* 达成率 >= {max(65.0, min(90.0, aft_att)):.0f}%，相对 Before 净省 {entry['gain_ms']:.3f}ms)"
        if aft_ms > 0.0 else
        f"确认 100% 掩盖或融合消除，实际暴露耗时 = 0.000ms (净省 {entry['gain_ms']:.3f}ms)，无 ACLGraph 图断裂"
    )
    disp_kernel = f"{entry['target_kernel']} [{entry['backend']}]"
    card_id = _make_card_slug(str(entry["target_kernel"]), used_slugs)
    return PendingKernelItem(
        priority=prio,
        kernel_key=str(entry["kernel_key"]),
        target_kernel=disp_kernel,
        suggested_backend=str(entry["backend"]),
        status=str(entry["status"]),
        replaces_chain=replaces_txt,
        source_refs=sources_txt,
        archetype_summary=arch_txt,
        total_calls_per_step=int(entry["calls"]),
        expected_gain_ms=float(entry["gain_ms"]),
        tensor_contract=str(entry["contract"]),
        microarch_target=microarch,
        default_design_slot=kernel_action,
        default_acceptance_slot=accept_slot,
        category=str(entry["category"]),
        logical_role=", ".join(entry["roles"][:2]) or str(entry["logical_role"]),
        submodule=str(entry["submodule"]),
        before_bound=str(entry["before_bound"]),
        after_bound=str(entry["after_bound"]),
        before_model_ms=bef_ms,
        after_model_ms=aft_ms,
        lstar_model_ms=ls_ms,
        gap_vs_lstar_ms=float(entry["gap_ms"]),
        card_id=card_id,
        config_howto=config_howto,
        kernel_action=kernel_action,
        framework_action=framework_action,
        before_attainment_pct=round(bef_att, 1),
        target_attainment_pct=round(aft_att, 1),
        opt_pillar=str(entry["opt_pillar"]),
    )


def build_pending_kernel_details(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
    *,
    top_k_optimize: int = DEFAULT_TOP_OPTIMIZE_KERNELS,
    include_to_optimize: bool = True,
) -> tuple[PendingKernelItem, ...]:
    """Build 100%-conserved Sheet 4 cards covering To-Implement and To-Optimize savings."""
    is_comparison = any(
        row.relation != "1->1"
        or len(row.before_ops) != len(row.after_ops)
        or (
            row.before_ops and row.after_ops
            and (
                abs(row.before_ops[0].exposed_dur_us - row.after_ops[0].exposed_dur_us) > OVERLAP_EXPOSED_EPS_US
                or row.before_ops[0].ascend_kernel != row.after_ops[0].ascend_kernel
            )
        )
        for row in aligned_rows
    )
    new_groups: dict[str, dict[str, Any]] = {}
    opt_groups: dict[str, dict[str, Any]] = {}
    for row in aligned_rows:
        rep = max(1, int(row.repeat_count))
        b_ms = sum(0.0 if b.is_overlapped else (b.exposed_dur_us * rep / US_PER_MS) for b in row.before_ops)
        a_ms = sum(0.0 if a.is_overlapped else (a.exposed_dur_us * rep / US_PER_MS) for a in row.after_ops)
        saved_ms = b_ms - a_ms
        classified = _classify_comparison_card_key(
            row, saved_ms, is_comparison=is_comparison, include_to_optimize=include_to_optimize,
        )
        if classified is None:
            continue
        key, cat, a_op = classified
        target_map = new_groups if cat == PENDING_KIND_TO_IMPLEMENT else opt_groups
        _accumulate_pending_group(
            target_map, key=key, category=cat, row=row, a_op=a_op,
            after_bundle=after_bundle, is_comparison=is_comparison,
        )

    ordered_new = sorted(new_groups.values(), key=lambda x: (-x["gain_ms"], -x["calls"], x["target_kernel"]))
    if is_comparison:
        ordered_opt = sorted(opt_groups.values(), key=lambda x: (-x["gain_ms"], -x["calls"], x["kernel_key"]))
        selected = [*ordered_new, *ordered_opt]
    else:
        ordered_opt = sorted(opt_groups.values(), key=lambda x: (-x["gap_ms"], -x["calls"], x["kernel_key"]))
        selected = [*ordered_new, *ordered_opt[: max(0, int(top_k_optimize))]]

    used_slugs: set[str] = set()
    return tuple(_materialize_pending_item(prio, entry, used_slugs) for prio, entry in enumerate(selected, start=1))


def build_new_kernel_backlog(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
) -> tuple[NewKernelBacklogItem, ...]:
    """Aggregate ``[需新开发算子]`` / ``[需扩展算子 Tiling]`` items into Sheet 4 backlog."""
    return build_pending_kernel_details(aligned_rows, after_bundle, include_to_optimize=False)


def _trim_text(text: str, limit: int = BRIEF_TEXT_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else (clean[: limit - 3] + "...")


def _bundle_kpi_dict(bundle: ExportBundle) -> dict[str, Any]:
    step_ms = bundle.step_span_us / US_PER_MS
    is_decode = bundle.phase == "decode"
    acc = max(1.0, bundle.avg_accepted_tokens)
    tpot_ms = (step_ms / acc) if is_decode else step_ms
    sla = bundle.tpot_slo_ms if is_decode else bundle.ttft_slo_ms
    l_star_ms = bundle.step_l_star_lean_us / US_PER_MS
    return {
        "step_ms": round(step_ms, 3),
        "effective_tpot_ms": round(tpot_ms, 3),
        "sla_target_ms": round(sla, 2),
        "sla_verdict": "PASS" if tpot_ms <= sla else "FAIL",
        "single_die_tps": round(bundle.throughput_per_die_tokens_per_s, 2),
        "cluster_tps": round(bundle.throughput_tokens_per_s, 2),
        "l_star_ms": round(l_star_ms, 3),
        "l_star_attainment_pct": round((100.0 * l_star_ms / step_ms) if step_ms > 0.0 else 0.0, 1),
        "cube_util_pct": round(bundle.network_cube_mfu, 1),
        "vector_util_pct": round(bundle.network_vec_mfu, 1),
        "hbm_mbu_pct": round(bundle.network_mbu, 1),
        "hccl_link_util_pct": round(bundle.network_link_util, 1),
    }


def build_agent_brief(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
    *,
    generated_files: Mapping[str, str] | None = None,
    script_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact (< 8 KB) deterministic JSON brief for cold-start AI Agents."""
    after = after_bundle or before_bundle
    macro_rows = build_macro_module_breakdown(before_bundle, after_bundle)
    aligned = (
        align_before_after_operators(before_bundle, after)
        if after_bundle is not None
        else tuple(
            AlignedComparisonRow(op.archetype_id, op.archetype_title or op.layer_label, op.repeat_count, op.submodule, (op,), (op,), "1->1", "")
            for op in before_bundle.operators
        )
    )
    pending = build_pending_kernel_details(aligned, after)
    roadmap = build_actionable_roadmap(before_bundle, after, pending)
    codesign = build_parallelism_codesign_note(before_bundle, after)
    mem = after.memory_ledger

    l1_groups: list[dict[str, Any]] = []
    by_l1: dict[str, list[MacroModuleRow]] = defaultdict(list)
    for row in macro_rows:
        by_l1[row.l1_group].append(row)
    tot_aft = max(sum(r.after_ms for r in macro_rows), CONSERVATION_EPS_MS)
    for l1_name, _ in MACRO_L1_HIERARCHY:
        children = by_l1.get(l1_name)
        if not children:
            continue
        l1_bef = sum(c.before_ms for c in children)
        l1_aft = sum(c.after_ms for c in children)
        l1_groups.append({
            "l1": l1_name.split(" (", 1)[0],
            "before_ms": round(l1_bef, 3),
            "after_ms": round(l1_aft, 3),
            "share_pct": round(100.0 * l1_aft / tot_aft, 1),
            "children": [
                {
                    "l2": c.module_name.split(" (", 1)[0],
                    "raw_ms": round(c.raw_after_ms, 3),
                    "masked_ms": round(c.masked_after_ms, 3),
                    "exposed_ms": round(c.after_ms, 3),
                    "before_ms": round(c.before_ms, 3),
                    "share_pct": round(100.0 * c.after_ms / tot_aft, 1),
                }
                for c in children
            ],
        })

    cfg_items = [
        {"title": _trim_text(r.title, 48), "saved_ms": round(r.expected_gain_ms, 3), "cmd": _trim_text(r.action_detail, 110)}
        for r in roadmap if r.track == "config_and_env"
    ][:BRIEF_MAX_FRAMEWORK_TASKS]
    fw_items = [
        {"title": _trim_text(r.title, 48), "saved_ms": round(r.expected_gain_ms, 3), "work": _trim_text(r.action_detail, 80)}
        for r in roadmap if r.track == "framework_tasks"
    ][:BRIEF_MAX_FRAMEWORK_TASKS]
    kern_items = [
        {
            "card_id": item.card_id,
            "prio": f"P{item.priority}",
            "cat": "new" if item.category == PENDING_KIND_TO_IMPLEMENT else "opt",
            "kernel": _trim_text(item.target_kernel, 44),
            "saved_ms": round(item.expected_gain_ms, 3),
            "before_ms": round(item.before_model_ms, 3),
            "after_ms": round(item.after_model_ms, 3),
            "att_pct": [item.before_attainment_pct, item.target_attainment_pct],
            "summary": _trim_text(item.kernel_action or item.default_design_slot, 68),
        }
        for item in sorted(pending, key=lambda x: -x.expected_gain_ms)[:BRIEF_MAX_KERNEL_TASKS]
    ]
    override_ex = (
        {
            pending[0].card_id: {
                "config_howto": "env/CLI override",
                "kernel_design": "AscendC/Triton tiling spec",
                "framework_work": "vllm-ascend pass note",
                "acceptance": "precision & latency gate",
            },
        }
        if pending else {}
    )
    b_kpi = _bundle_kpi_dict(before_bundle)
    a_kpi = _bundle_kpi_dict(after)
    speedup = round(b_kpi["step_ms"] / a_kpi["step_ms"], 2) if a_kpi["step_ms"] > 0.0 else 1.0
    brief: dict[str, Any] = {
        "schema": AGENT_BRIEF_SCHEMA,
        "scenario_and_parallelism": {
            "model": after.model_name,
            "hardware_sku": after.hardware_sku,
            "topology_preset": after.topology_preset,
            "phase": after.phase,
            "deployment_mode": after.deployment_mode,
            "total_dies": mem.total_dies,
            "dp": mem.dp, "tp": mem.tp, "ep": mem.ep, "pp": mem.pp,
            "dcp": mem.dcp, "pcp": mem.pcp, "local_tp": mem.local_tp,
            "sequence_parallel": after.sequence_parallel,
            "prompt_tokens": after.prompt_tokens,
            "output_tokens": after.output_tokens,
            "concurrency_per_dp": mem.concurrency,
            "cluster_concurrency": mem.concurrency * mem.dp,
            "max_hbm_concurrency_per_dp": mem.max_hbm_concurrency_per_dp,
            "spec_method": after.spec_method,
            "spec_proposals": after.spec_proposals,
            "avg_accepted_tokens": after.avg_accepted_tokens,
            "weight_quant": mem.weight_quant_display or mem.weight_quant,
            "cache_dtype": mem.cache_dtype,
            "native_bf16_weight_gib_per_die": round(mem.native_bf16_weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
            "quantized_weight_gib_per_die": round(mem.weight_footprint_per_die_bytes / float(GIB_BYTES), 2),
            "weight_compression_ratio": round(mem.weight_compression_ratio, 2),
            "host_dram_offload_gib_per_die": round(mem.max_stage_host_offload_bytes / float(GIB_BYTES), 2),
            "vision_tower_included": mem.include_vision_tower,
            "fits_hbm": mem.fits_hbm,
        },
        "kpis_before_vs_after": {
            "before": b_kpi,
            "after": a_kpi,
            "saved_step_ms": round(b_kpi["step_ms"] - a_kpi["step_ms"], 3),
            "speedup": speedup,
            "sheet4_cards_total_gain_ms": round(sum(x.expected_gain_ms for x in pending), 3),
        },
        "module_hierarchy": l1_groups,
        "action_plan_3_tracks": {
            "config_and_env": cfg_items,
            "framework_tasks": fw_items,
            "kernel_tasks": kern_items,
            "parallelism_codesign_note": codesign,
            "kernel_designs_override_example": override_ex,
        },
        "generated_files": dict(generated_files or {}),
    }
    if script_audit is not None:
        raw_issues = script_audit.get("issues_found") or ()
        brief["script_audit"] = {
            "source_script": script_audit.get("source_script", ""),
            "issues_count": len(raw_issues),
            "top_issues": [
                {
                    "id": i.get("issue_id", ""),
                    "title": _trim_text(str(i.get("title", "")), 44),
                    "rec": _trim_text(str(i.get("recommended", "")), 56),
                }
                for i in list(raw_issues)[:BRIEF_MAX_FRAMEWORK_TASKS]
            ],
            "two_step_gain": script_audit.get("two_step_gain", {}),
            "optimized_script_path": script_audit.get("optimized_script_path", ""),
        }
    return brief


def export_agent_brief(brief: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically persist compact (< 8 KB) ``<prefix>_agent_brief.json`` with UTF-8 LF."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(payload.encode("utf-8")) <= AGENT_BRIEF_PRETTY_LIMIT_BYTES:
        pretty = json.dumps(brief, ensure_ascii=False, indent=1) + "\n"
        if len(pretty.encode("utf-8")) < AGENT_BRIEF_MAX_BYTES:
            payload = pretty
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except TimeoutError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def format_executive_digest_markdown(brief: Mapping[str, Any]) -> str:
    """Format a crisp ~25-line executive Markdown digest for stdout or chat."""
    sc = brief.get("scenario_and_parallelism") or {}
    kp = brief.get("kpis_before_vs_after") or {}
    bef = kp.get("before") or {}
    aft = kp.get("after") or {}
    plan = brief.get("action_plan_3_tracks") or {}
    lines = [
        f"## MindIE-Sim Executive Digest: {sc.get('model')} @ {sc.get('hardware_sku')}",
        (
            f"- **Deploy**: {sc.get('total_dies')} Dies (DP{sc.get('dp')}·TP{sc.get('tp')}·EP{sc.get('ep')}"
            f"·DCP{sc.get('dcp')}·PP{sc.get('pp')}), {sc.get('deployment_mode')}, "
            f"Prompt={sc.get('prompt_tokens')}, Concurrency={sc.get('concurrency_per_dp')}/DP "
            f"(Max HBM={sc.get('max_hbm_concurrency_per_dp')})"
        ),
        (
            f"- **Weights/Die**: {sc.get('quantized_weight_gib_per_die')} GiB "
            f"(原生 BF16: {sc.get('native_bf16_weight_gib_per_die')} GiB, 压缩 {sc.get('weight_compression_ratio')}x, "
            f"Quant={sc.get('weight_quant')}, KV={sc.get('cache_dtype')})"
        ),
        "",
        "| KPI | Before | After | Delta / Verdict |",
        "| --- | ---: | ---: | --- |",
        f"| Step Latency (ms) | {bef.get('step_ms')} | **{aft.get('step_ms')}** | -{kp.get('saved_step_ms')} ms (**{kp.get('speedup')}x**) |",
        f"| Effective TPOT (ms) | {bef.get('effective_tpot_ms')} ({bef.get('sla_verdict')}) | **{aft.get('effective_tpot_ms')} ({aft.get('sla_verdict')})** | SLA <= {aft.get('sla_target_ms')} ms |",
        f"| Single-Die / Cluster TPS | {bef.get('single_die_tps')} / {bef.get('cluster_tps')} | **{aft.get('single_die_tps')} / {aft.get('cluster_tps')}** | tok/s |",
        f"| Physical Limit Attainment (L*/Step) | {bef.get('l_star_attainment_pct')}% | **{aft.get('l_star_attainment_pct')}%** | L*={aft.get('l_star_ms')} ms |",
        f"| HBM MBU / AIC MFU / AIV / HCCL | {bef.get('hbm_mbu_pct')}% / {bef.get('cube_util_pct')}% / {bef.get('vector_util_pct')}% / {bef.get('hccl_link_util_pct')}% | **{aft.get('hbm_mbu_pct')}% / {aft.get('cube_util_pct')}% / {aft.get('vector_util_pct')}% / {aft.get('hccl_link_util_pct')}%** | Hardware Util |",
        "",
        "### 3-Track Action Plan (`Config & Env` / `Framework` / `Kernel`)",
    ]
    for cfg in (plan.get("config_and_env") or ())[:2]:
        lines.append(f"- **[Config/Env]** {cfg.get('title')}: `{cfg.get('cmd') or cfg.get('command_or_flag')}`")
    for fw in (plan.get("framework_tasks") or ())[:2]:
        lines.append(f"- **[Framework]** {fw.get('title')} (-{fw.get('saved_ms')} ms): {fw.get('work')}")
    for kt in (plan.get("kernel_tasks") or ())[:3]:
        lines.append(
            f"- **[Kernel `{kt.get('card_id')}`]** {kt.get('kernel')} "
            f"({kt.get('before_ms')}->{kt.get('after_ms')} ms, -{kt.get('saved_ms')} ms): {kt.get('summary')}"
        )
    if plan.get("parallelism_codesign_note"):
        lines.append(f"- **Co-Design**: {plan['parallelism_codesign_note']}")
    if brief.get("generated_files"):
        files_str = ", ".join(f"{k}=`{v}`" for k, v in brief["generated_files"].items())
        lines.append(f"- **Artifacts**: {files_str}")
    return "\n".join(lines) + "\n"
