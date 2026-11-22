"""Before-vs-After operator alignment, macro-module aggregation, and pending kernel details.

[Role] Compare lowered ``ExportBundle`` snapshots to produce:
  (1) aligned ``(Before -> After)`` 1->1 / N->1 fusion / added / eliminated groups
      for Sheet 2 (``Layer_Operator_Breakdown``),
  (2) full-model exposed duration, share %, saved Δ ms, ``L*``, and dominant bound
      broken down across architectural macro-modules for Sheet 1 Block 3, and
  (3) dual ``待新开发 / 扩展 (To-Implement)`` + ``重点待优化 (Top Gap To-Optimize)``
      structured kernel detail items for Sheet 4 (``Pending_Kernel_Details``).
[Invariants] Pure functional transformations over immutable ``ExportBundle`` rows;
  zero engine execution, zero OOXML formatting, and 100% exposure conservation
  across macro-modules.
[Timeouts & Locks] O(Unique Archetype Ops) in-memory passes; no locks or I/O.
[Side-Effect Policy] Zero filesystem or global-state side effects.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re
from typing import Any

from .kernel_catalog import classify_macro_module, strip_op_prefix
from .models import (
    DEFAULT_TOP_OPTIMIZE_KERNELS,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_SPEC_HEAD,
    MACRO_MODULE_CORE_ORDER,
    MACRO_MODULE_OPTIONAL_ORDER,
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
        "expert_down.quantize", "expert_down", "unpack", "join",
    ),
    "dispatch_ffn_combine": (
        "dispatch", "dispatch.received_quantize", "expert_up", "expert_act",
        "expert_down.quantize", "expert_down", "unpack", "join",
    ),
    "grouped_matmul_situ_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "grouped_matmul_swiglu_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "quant_lightning_indexer": ("index_query_quantize", "index_scores", "index_topk"),
    "matmul_reduce_scatter": (
        "output", "output_reduce", "output_reduce.reduce_scatter", "down", "tp",
        "tp.reduce_scatter", "linear_output", "linear_reduce.reduce_scatter",
    ),
    "all_gather_matmul": ("attention.sp_gather", "ffn.sp_gather", "shared.sp_gather"),
    "add_rmsnorm": ("residual", "norm1", "norm2", "final_norm"),
    "add_rmsnorm_quant": ("residual", "norm1", "norm2", "quantize"),
    "recurrent_scan": ("qk_l2norm", "decode_q_l2norm", "decode_k_l2norm", "delta_gates", "recurrent_scan"),
}
_OPTIMIZE_DESIGN_HINTS: dict[str, str] = {
    "link_bound": "异步流重叠 / MC2 通算融合 / HCCL Ring-Direct 拓扑调优与小包合并，消除暴露通信等待",
    "memory_bound": "紧凑 Tiling / L1-L2 权重驻留 / 算子融合降访存与量化压缩 (MXFP4/FP8)，逼近 HBM 带宽极限 L*",
    "cube_bound": "AIC 核内 M/N/K 分块与 NZ 布局对齐 / L0C Ping-Pong 双缓冲，提升 Cube MFU",
    "vector_bound": "AIV 向量算子融合与 UB 双缓冲流水编排，减少中间回写与同步开销",
    "launch_bound": "ACLGraph 全图捕获 / 算子合并消除 Host 核发射与小算子调度开销",
}


def _fusion_absorbed_keys(after_op: OperatorExportRow) -> set[str]:
    keys = {strip_op_prefix(x).lower() for x in after_op.absorbed_source_ops if x}
    canon = (after_op.canonical_op_key or strip_op_prefix(after_op.op_id)).lower()
    keys.update(_KNOWN_FUSION_SOURCES.get(canon, ()))
    kern = (after_op.ascend_kernel or "").lower()
    if "mlaprolog" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["mla_prolog_v3"])
    elif "megamoe" in kern or "dispatch_ffn_combine" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["megamoe_arch35"])
    elif "situquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_situ_quant"])
    elif "swigluquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_swiglu_quant"])
    return keys


def _build_action_note(
    relation: str, before_ops: Sequence[OperatorExportRow], after_ops: Sequence[OperatorExportRow],
) -> str:
    if relation == "N->1" and after_ops:
        a = after_ops[0]
        return f"N->1 Fused ({len(before_ops)} ops -> {a.ascend_kernel}); {a.kernel_status}"
    if relation == "eliminated":
        return "Eliminated by After graph/fusion pass (0 exposed ms)"
    if relation == "added" and after_ops:
        a = after_ops[0]
        return f"New op in After ({a.logical_role}); {a.kernel_status}"
    if after_ops:
        a = after_ops[0]
        if a.is_overlapped:
            return "Overlapped on MSO Stream 1 (0 exposed ms)"
        if a.needs_new_kernel and a.new_kernel_spec:
            return f"{a.kernel_status}: {a.new_kernel_spec}"
        if before_ops:
            b = before_ops[0]
            if b.ascend_kernel != a.ascend_kernel:
                return f"Kernel upgrade: {b.ascend_kernel} -> {a.ascend_kernel}"
            if a.gap_us > 1.0 and a.l_star_lean_us > 0.0:
                return f"Gap vs L* ({a.l_star_lean_us:.1f}us vs {a.dur_us:.1f}us): {a.bound_display}"
        return f"{a.bound_display} | {a.kernel_status}"
    return ""


def align_before_after_operators(
    before_bundle: ExportBundle, after_bundle: ExportBundle,
) -> tuple[AlignedComparisonRow, ...]:
    """Align Before vs After operators per ``(Archetype, Submodule)`` supporting N->1 / 1->N."""
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
        b_list = before_by_arch.get(aid, [])
        a_list = after_by_arch.get(aid, [])
        sub_order: list[str] = []
        for op in (*b_list, *a_list):
            if op.submodule not in sub_order:
                sub_order.append(op.submodule)
        for sub in sub_order:
            b_sub = [op for op in b_list if op.submodule == sub]
            a_sub = [op for op in a_list if op.submodule == sub]
            aligned.extend(_align_submodule_ops(aid, title, rep, sub, b_sub, a_sub))
    return tuple(aligned)


def _align_submodule_ops(
    aid: str, title: str, rep: int, sub: str,
    b_sub: Sequence[OperatorExportRow], a_sub: Sequence[OperatorExportRow],
) -> list[AlignedComparisonRow]:
    used_b: set[int] = set()
    used_a: set[int] = set()
    entries: list[tuple[int, AlignedComparisonRow]] = []

    b_by_key: dict[str, list[int]] = defaultdict(list)
    for idx, op in enumerate(b_sub):
        b_by_key[(op.canonical_op_key or strip_op_prefix(op.op_id)).lower()].append(idx)

    for a_idx, a_op in enumerate(a_sub):
        a_key = (a_op.canonical_op_key or strip_op_prefix(a_op.op_id)).lower()
        absorbed = _fusion_absorbed_keys(a_op)
        matched_b = [
            b_idx for b_idx, b_op in enumerate(b_sub)
            if b_idx not in used_b and (b_op.canonical_op_key or strip_op_prefix(b_op.op_id)).lower() in absorbed
            and (b_op.canonical_op_key or strip_op_prefix(b_op.op_id)).lower() != a_key
        ]
        if len(matched_b) > 1 or (len(matched_b) == 1 and a_key not in b_by_key):
            for b_idx in b_by_key.get(a_key, ()):
                if b_idx not in used_b and b_idx not in matched_b:
                    matched_b.append(b_idx)
            matched_b.sort()
            used_b.update(matched_b)
            used_a.add(a_idx)
            b_tuple = tuple(b_sub[i] for i in matched_b)
            rel = "N->1" if len(b_tuple) > 1 else "1->1"
            entries.append((matched_b[0], AlignedComparisonRow(aid, title, rep, sub, b_tuple, (a_op,), rel, _build_action_note(rel, b_tuple, (a_op,)))))

    for a_idx, a_op in enumerate(a_sub):
        if a_idx in used_a:
            continue
        a_key = (a_op.canonical_op_key or strip_op_prefix(a_op.op_id)).lower()
        cand_idx = next((i for i in b_by_key.get(a_key, ()) if i not in used_b), None)
        if cand_idx is not None:
            used_b.add(cand_idx)
            used_a.add(a_idx)
            b_op = b_sub[cand_idx]
            entries.append((cand_idx, AlignedComparisonRow(aid, title, rep, sub, (b_op,), (a_op,), "1->1", _build_action_note("1->1", (b_op,), (a_op,)))))

    for b_idx, b_op in enumerate(b_sub):
        if b_idx not in used_b:
            entries.append((b_idx, AlignedComparisonRow(aid, title, rep, sub, (b_op,), (), "eliminated", _build_action_note("eliminated", (b_op,), ()))))
    for a_idx, a_op in enumerate(a_sub):
        if a_idx not in used_a:
            entries.append((len(b_sub) + a_idx, AlignedComparisonRow(aid, title, rep, sub, (), (a_op,), "added", _build_action_note("added", (), (a_op,)))))
    entries.sort(key=lambda item: item[0])
    return [row for _, row in entries]


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
            "target_layer_archs": {},
            "draft_layers": 0,
            "has_head_or_embed": False,
        })
        rep = max(1, int(op.repeat_count))
        exp_ms = 0.0 if op.is_overlapped else (op.exposed_dur_us * rep / US_PER_MS)
        raw_ms = op.dur_us * rep / US_PER_MS
        lstar_ms = 0.0 if op.is_overlapped else (op.l_star_lean_us * rep / US_PER_MS)
        bucket["exposed_ms"] += exp_ms
        bucket["raw_ms"] += raw_ms
        bucket["lstar_ms"] += lstar_ms
        if exp_ms > 0.0:
            bucket["bound_ms"][op.bound_display or op.dominant_bound] += exp_ms
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
    if mod == MACRO_MOD_SPEC_HEAD:
        d_cnt = draft_layers or bundle.total_draft_layers
        if d_cnt > 0:
            return f"Draft {d_cnt}L + Head", d_cnt
        return "Head / Embed (×1)", 1
    if t_layers > 0:
        return f"{t_layers}L", t_layers
    if has_head:
        return "×1", 1
    return "0L", 0


def build_macro_module_breakdown(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle | None = None,
) -> tuple[MacroModuleRow, ...]:
    """Aggregate exposed full-model duration, share %, and L* across architectural modules."""
    bef_stats = _accumulate_macro_stats(before_bundle)
    aft_bundle = after_bundle or before_bundle
    aft_stats = bef_stats if after_bundle is None else _accumulate_macro_stats(after_bundle)

    tot_bef = sum(v["exposed_ms"] for v in bef_stats.values())
    tot_aft = sum(v["exposed_ms"] for v in aft_stats.values())
    ordered_mods = list(MACRO_MODULE_CORE_ORDER) + [
        m for m in MACRO_MODULE_OPTIONAL_ORDER if m in bef_stats or m in aft_stats
    ]
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
            dom_bound = f"MSO Stream 1 Overlapped ({raw_aft:.3f}ms hidden)"
        elif bound_weights:
            dom_bound = max(bound_weights.items(), key=lambda kv: kv[1])[0]
        else:
            dom_bound = "Eliminated / Overlapped"
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
        ))
    return tuple(rows)


def _short_arch_name(title: str) -> str:
    return re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", title.split("—")[0].strip())


def _accumulate_pending_group(
    grouped: dict[str, dict[str, Any]],
    *,
    key: str,
    category: str,
    row: AlignedComparisonRow,
    a_op: OperatorExportRow,
    after_bundle: ExportBundle,
) -> None:
    rep = max(1, int(row.repeat_count))
    before_ms = sum(b.exposed_dur_us * rep / US_PER_MS for b in row.before_ops)
    after_ms = 0.0 if a_op.is_overlapped else (a_op.exposed_dur_us * rep / US_PER_MS)
    lstar_ms = 0.0 if a_op.is_overlapped else (a_op.l_star_lean_us * rep / US_PER_MS)
    gap_ms = max(0.0, after_ms - lstar_ms)
    if category == PENDING_KIND_TO_IMPLEMENT:
        gain_ms = max(0.0, before_ms - after_ms)
        if gain_ms <= 0.0 and gap_ms > 0.0:
            gain_ms = gap_ms
    else:
        gain_ms = gap_ms

    bef_bounds = [b.bound_display or b.dominant_bound for b in row.before_ops if b.bound_display or b.dominant_bound]
    bef_bound_lbl = " / ".join(dict.fromkeys(bef_bounds)) or (a_op.bound_display or a_op.dominant_bound)
    entry = grouped.get(key)
    if entry is None:
        entry = {
            "kernel_key": key,
            "category": category,
            "target_kernel": a_op.ascend_kernel,
            "logical_role": a_op.logical_role or a_op.op_id,
            "submodule": a_op.submodule,
            "backend": a_op.kernel_backend,
            "status": a_op.kernel_status if category == PENDING_KIND_TO_IMPLEMENT else f"重点优化 (Gap vs L* | {a_op.kernel_status})",
            "replaces": [],
            "sources": [],
            "archetypes": [],
            "calls": 0,
            "before_ms": 0.0,
            "after_ms": 0.0,
            "lstar_ms": 0.0,
            "gap_ms": 0.0,
            "gain_ms": 0.0,
            "before_bound": bef_bound_lbl,
            "after_bound": a_op.bound_display or a_op.dominant_bound,
            "dom_key": a_op.dominant_bound,
            "contract": a_op.tensor_contract,
            "single_lstar_us": a_op.l_star_lean_us,
            "single_sim_us": a_op.dur_us,
            "die_summary": after_bundle.hardware_die_summary,
            "spec_note": a_op.new_kernel_spec,
        }
        grouped[key] = entry
    for b_op in row.before_ops:
        entry["replaces"].append(b_op.ascend_kernel)
        if b_op.source_ref:
            entry["sources"].append(b_op.source_ref)
    if a_op.source_ref:
        entry["sources"].append(a_op.source_ref)
    entry["archetypes"].append(f"{_short_arch_name(row.archetype_title)} (×{rep})")
    entry["calls"] += rep
    entry["before_ms"] += before_ms
    entry["after_ms"] += after_ms
    entry["lstar_ms"] += lstar_ms
    entry["gap_ms"] += gap_ms
    entry["gain_ms"] += gain_ms


def _materialize_pending_item(prio: int, entry: Mapping[str, Any]) -> PendingKernelItem:
    is_new = entry["category"] == PENDING_KIND_TO_IMPLEMENT
    replaces_txt = (
        " + ".join(dict.fromkeys(entry["replaces"])) or "New target kernel"
        if is_new else f"{entry['submodule']} -> {entry['logical_role']}"
    )
    sources_txt = "; ".join(dict.fromkeys(entry["sources"]))[:MAX_SOURCE_REF_SUMMARY_LEN] or "-"
    arch_txt = "; ".join(dict.fromkeys(entry["archetypes"]))
    microarch = (
        f"Before: {entry['before_bound']} -> Target: {entry['after_bound']} | "
        f"Single L*={entry['single_lstar_us']:.2f}us vs Sim={entry['single_sim_us']:.2f}us | {entry['die_summary']}"
    )
    if is_new:
        design_slot = f"[待 Agent 补充设计方案] {entry['spec_note'] or 'Tiling / L1-UB Ping-Pong / AIC-AIV 配比'}"
        accept_slot = "[待 Agent 补充验收标准] 精度误差 <= 1e-3 (BF16/MXFP4) & L* 达成率 >= 70%"
    else:
        hint = _OPTIMIZE_DESIGN_HINTS.get(str(entry["dom_key"]), "Tiling / 流水重叠 / 访存优化")
        design_slot = f"[待 Agent 补充优化方案] {hint}"
        accept_slot = f"[待 Agent 补充验收标准] 缩小 Gap vs L* ({entry['gap_ms']:.3f}ms)，逼近物理下界 L*={entry['lstar_ms']:.3f}ms"
    disp_kernel = (
        f"{entry['target_kernel']} [{entry['backend']}]"
        if is_new else f"{entry['target_kernel']} ({entry['logical_role']}) [{entry['backend']}]"
    )
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
        default_design_slot=design_slot,
        default_acceptance_slot=accept_slot,
        category=str(entry["category"]),
        logical_role=str(entry["logical_role"]),
        submodule=str(entry["submodule"]),
        before_bound=str(entry["before_bound"]),
        after_bound=str(entry["after_bound"]),
        before_model_ms=float(entry["before_ms"]),
        after_model_ms=float(entry["after_ms"]),
        lstar_model_ms=float(entry["lstar_ms"]),
        gap_vs_lstar_ms=float(entry["gap_ms"]),
    )


def build_pending_kernel_details(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
    *,
    top_k_optimize: int = DEFAULT_TOP_OPTIMIZE_KERNELS,
    include_to_optimize: bool = True,
) -> tuple[PendingKernelItem, ...]:
    """Build dual Sheet 4 items: (A) To-Implement new/extended ops + (B) Top-Gap To-Optimize."""
    new_groups: dict[str, dict[str, Any]] = {}
    opt_groups: dict[str, dict[str, Any]] = {}
    for row in aligned_rows:
        for a_op in row.after_ops:
            if a_op.needs_new_kernel:
                _accumulate_pending_group(
                    new_groups, key=a_op.ascend_kernel, category=PENDING_KIND_TO_IMPLEMENT,
                    row=row, a_op=a_op, after_bundle=after_bundle,
                )
            elif include_to_optimize and not a_op.is_overlapped and a_op.gap_us > 0.0:
                op_canon = a_op.canonical_op_key or strip_op_prefix(a_op.op_id)
                opt_key = f"{a_op.ascend_kernel}::{op_canon}"
                _accumulate_pending_group(
                    opt_groups, key=opt_key, category=PENDING_KIND_TO_OPTIMIZE,
                    row=row, a_op=a_op, after_bundle=after_bundle,
                )

    ordered_new = sorted(new_groups.values(), key=lambda x: (-x["gain_ms"], -x["calls"], x["target_kernel"]))
    ordered_opt = sorted(opt_groups.values(), key=lambda x: (-x["gap_ms"], -x["calls"], x["kernel_key"]))
    selected = [*ordered_new, *ordered_opt[: max(0, int(top_k_optimize))]]
    return tuple(_materialize_pending_item(prio, entry) for prio, entry in enumerate(selected, start=1))


def build_new_kernel_backlog(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
) -> tuple[NewKernelBacklogItem, ...]:
    """Aggregate ``[需新开发算子]`` / ``[需扩展算子 Tiling]`` items into Sheet 4 backlog."""
    return build_pending_kernel_details(aligned_rows, after_bundle, include_to_optimize=False)
