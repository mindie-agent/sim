"""Topology-preserving bipartite alignment & L1/L2 macro-module breakdown.

[Role] Own physical-layer slicing, receipt-first semantic bipartite alignment,
  dependency-preserving Kahn topological coalescing with same-submodule branch
  locality (D5a, zero ``[↑ 已并入`` split rows), L1/L2 macro-module aggregation,
  and co-domain parallelism display, while preserving public comparison imports
  re-exported from ``action_plan`` and ``agent_brief``.
[Invariants] Strictly preserves every Before and After operator occurrence once
  per physical slice, validates raw execution DAGs as acyclic, condenses only
  genuine dependency-crossing SCCs into honest ``M->N`` blocks, emits zero
  ``fused_into_anchor`` split rows, zero prose dispatch tables, and zero
  destructive mutation of producer ``needs_new_kernel`` / ABI receipts.
[Timeouts & Locks] Pure O(Unique Archetype Ops) in-memory passes; no I/O or locks.
[Side-Effect Policy] Read-only over immutable ``ExportBundle`` records.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

from .action_plan import (
    CONSERVATION_EPS_MS,
    WF_STAGE_CONFIG, WF_STAGE_FRAMEWORK, WF_STAGE_KERNEL, WF_STAGE_RESIDUAL,
    ComparisonPlan, GroupAccounting, WaterfallSummary,
    build_actionable_roadmap, build_comparison_plan,
    build_new_kernel_backlog, build_parallelism_codesign_note, build_pending_kernel_details,
    classify_waterfall_stage,
)
from .agent_brief import (
    AGENT_BRIEF_MAX_BYTES, _compact_agent_brief_to_budget, build_agent_brief,
    export_agent_brief, format_executive_digest_markdown,
)
from .bundle import format_attnres_depth_note
from .kernel_catalog import classify_macro_module, extract_semantic_atoms, strip_op_prefix
from .models import (
    BINDING_STATE_UNBOUND_FORMULA, L1_MOD_RESIDUAL_MEM, MACRO_L1_HIERARCHY,
    MACRO_MOD_DRAFT_LAYERS, MACRO_MOD_MOE_SHARED, MACRO_MOD_VERIFY_HEAD,
    MIN_DENOM_EPS, NS_PER_MS, NS_PER_US, OVERLAP_EXPOSED_EPS_US, US_PER_MS,
    AlignedComparisonRow, ComparisonRequest, ExportBundle, MacroModuleRow,
    OperatorExportRow, bundle_step_int_ns, op_dur_int_ns, op_exposed_int_ns,
    op_lstar_int_ns,
)
from .public_view import format_track_label

SLICE_MAX_EXPLICIT_LAYERS = 4
MIN_GROUP_SIZE = 2
ACTION_GAP_CALLOUT_US = 1.0
ADDED_ROW_TIE_OFFSET = 0.5
_SCOPE_PREFIXES = ("attention.", "ffn.", "shared.", "final.")
_DEPTH_NOTE_STRIP_RE = re.compile(
    r"(?:\s*\|\s*)?(?:depth-varying\b[^|]*|L\d+\s*(?:->|→)\s*L\d+，随层深[^|]*)"
)

_UNBOUND_PROJECTION_BADGE = "[未绑定实现 · 硬件物理预演]"


def format_public_track_label(bundle: ExportBundle, *, role: str = "") -> str:
    """Delegate strictly to canonical ``public_view.format_track_label`` (C-1 StackTrack)."""
    raw = str(getattr(bundle, "feature_track", "") or "").strip()
    base = format_track_label(raw) if raw else ("基线配置" if role == "before" else "优化方案")
    has_unbound = bool(getattr(bundle, "has_unbound_formula_ops", False)) or any(
        getattr(op, "binding_state", "") == BINDING_STATE_UNBOUND_FORMULA
        for op in getattr(bundle, "operators", ())
    )
    if role == "after" and has_unbound and _UNBOUND_PROJECTION_BADGE not in base:
        return f"{base} {_UNBOUND_PROJECTION_BADGE}"
    return base


def format_parallelism_codomain_summary(bundle: ExportBundle) -> tuple[str, str]:
    """Render coherent DCP<->TP co-domain banner and per-module parallel breakdown."""
    mem = bundle.memory_ledger
    sp_zh, sp_en = ("开启", "ON") if bundle.sequence_parallel else ("关闭", "OFF")
    if mem.dcp > 1 and mem.dcp == mem.tp:
        short = f"DP{mem.dp} · TP{mem.tp} (共域 DCP{mem.dcp}) · EP{mem.ep} · SP({sp_zh})"
        dcp_tag, mla_dom = f"DCP={mem.dcp} (与 TP 共域)", f"MLA: TP{mem.tp} + 共域 KV-DCP{mem.dcp}"
    elif mem.dcp > 1:
        short = f"DP{mem.dp} · TP{mem.tp} · DCP{mem.dcp} · EP{mem.ep} · SP({sp_zh})"
        dcp_tag, mla_dom = f"DCP={mem.dcp}", f"MLA: TP{mem.tp} + KV-DCP{mem.dcp}"
    else:
        short = f"DP{mem.dp} · TP{mem.tp} · EP{mem.ep} · SP({sp_zh})"
        dcp_tag, mla_dom = f"DCP={mem.dcp}", f"MLA/Attn: Local-TP{mem.local_tp}"
    lin_dom = f"Linear/Dense: TP{mem.tp}" + (" + SP" if bundle.sequence_parallel else "")
    moe_dom = f"Routed MoE: EP{mem.ep} | Shared MoE: TP{mem.shared_tp}"
    detail = f"DP={mem.dp}, TP={mem.tp}, EP={mem.ep}, PP={mem.pp}, {dcp_tag}, PCP={mem.pcp}, Local-TP={mem.local_tp}, SP={sp_en} | {mla_dom} | {lin_dom} | {moe_dom}"
    return short, detail


def _scope_of_suffix(suf: str) -> str:
    for prefix in _SCOPE_PREFIXES:
        if suf.startswith(prefix):
            return prefix
    return ""


def _op_exact_key(op: OperatorExportRow) -> str:
    return strip_op_prefix(op.op_id).lower()


def _op_occ_key(op: OperatorExportRow) -> tuple[int, int, int, str]:
    occ = op.occurrence
    return (int(occ.replica), int(occ.stage), (-1 if occ.rank is None else int(occ.rank)), str(occ.op_id))


def _op_atoms(op: OperatorExportRow) -> frozenset[str]:
    if op.semantic_atoms:
        return frozenset(op.semantic_atoms)
    return frozenset(extract_semantic_atoms(op.op_id, op.family, op.recipe, op.absorbed_source_ops))


def _scope_compatible(a_scope: str, b_scope: str) -> bool:
    return (not a_scope) or (not b_scope) or (a_scope == b_scope)


def _detect_added_glue_note(b_op: OperatorExportRow, a_op: OperatorExportRow) -> str:
    b_ex, a_fam, b_fam = _op_exact_key(b_op), a_op.family.lower(), b_op.family.lower()
    if a_fam == "all_gather_matmul" and b_fam != "all_gather_matmul":
        return "+AllGather(SP) 融合"
    if a_fam == "matmul_reduce_scatter" and b_fam != "matmul_reduce_scatter":
        return "+ReduceScatter(SP) 融合"
    extra = [
        strip_op_prefix(s) for s in a_op.absorbed_source_ops
        if strip_op_prefix(s).lower() != b_ex and not extract_semantic_atoms(s, "", "")
    ]
    return f"+{'/'.join(extra[:2])} 融合" if extra else ""


def _build_factual_action_note(
    relation: str, before_ops: Sequence[OperatorExportRow], after_ops: Sequence[OperatorExportRow], *, added_glue_note: str = "",
) -> str:
    """Emit purely factual receipt summary for Sheet 2 Col V (zero prose templates)."""
    if relation == "N->1" and after_ops:
        a_op = after_ops[0]
        parts = [f"N->1 融合 ({len(before_ops)} ops -> {a_op.ascend_kernel})"]
        if a_op.kernel_synthesis_spec is not None:
            parts.append(a_op.kernel_synthesis_spec.format_formula_block())
        elif a_op.kernel_status:
            parts.append(a_op.kernel_status)
        if a_op.stream_core_budget:
            parts.append(f"Cores={a_op.stream_core_budget}")
        return " | ".join(parts)
    if "->" in relation and relation not in {"1->1", "N->1"}:
        return f"[{relation} 子图重组块] Before {len(before_ops)} ops <-> After {len(after_ops)} ops"
    if relation == "eliminated":
        return "优化后移除 (0 暴露)"
    if relation == "added" and after_ops:
        a_op = after_ops[0]
        return f"优化后新增: {a_op.ascend_kernel} ({a_op.bound_display or a_op.dominant_bound})"
    if after_ops:
        a_op = after_ops[0]
        glue_tag = f" ({added_glue_note})" if added_glue_note else ""
        budget_tag = f" | Cores={a_op.stream_core_budget}" if a_op.stream_core_budget else ""
        if a_op.is_overlapped:
            return f"并发流被 {a_op.masked_by or '主计算流'} 100% 掩盖 (0 暴露){budget_tag}"
        if before_ops and before_ops[0].ascend_kernel != a_op.ascend_kernel:
            return f"实现变更{glue_tag}: {before_ops[0].ascend_kernel} -> {a_op.ascend_kernel}{budget_tag}"
        if a_op.gap_us > ACTION_GAP_CALLOUT_US and a_op.l_star_lean_us > 0.0:
            return f"{a_op.bound_display or a_op.dominant_bound}{glue_tag} | L*={a_op.l_star_lean_us:.2f}us vs {a_op.dur_us:.2f}us{budget_tag}"
        return f"{a_op.bound_display or a_op.dominant_bound}{glue_tag}{budget_tag}"
    return ""


def _build_side_dag(
    ops: Sequence[OperatorExportRow], side_label: str, aid: str,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build directed ``(preds, succs)`` within one slice and verify acyclicity."""
    occ_counts: dict[tuple[int, int, int, str], int] = defaultdict(int)
    opid_counts: dict[str, int] = defaultdict(int)
    for op in ops:
        occ_counts[_op_occ_key(op)] += 1
        opid_counts[op.op_id] += 1
    unique_occ = {_op_occ_key(op): i for i, op in enumerate(ops) if occ_counts[_op_occ_key(op)] == 1}
    unique_opid = {op.op_id: i for i, op in enumerate(ops) if opid_counts[op.op_id] == 1}
    preds: dict[int, set[int]] = {i: set() for i in range(len(ops))}
    succs: dict[int, set[int]] = {i: set() for i in range(len(ops))}
    for idx, op in enumerate(ops):
        for pred in op.predecessors:
            pkey = (int(pred.replica), int(pred.stage), (-1 if pred.rank is None else int(pred.rank)), str(pred.op_id))
            p_idx = unique_occ.get(pkey)
            if p_idx is None:
                p_idx = unique_opid.get(str(pred.op_id))
            if p_idx is not None and p_idx != idx:
                preds[idx].add(p_idx)
                succs[p_idx].add(idx)

    in_deg = {i: len(preds[i]) for i in range(len(ops))}
    queue = [i for i, d in in_deg.items() if d == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if visited != len(ops):
        witness = next(ops[i].op_id for i, d in in_deg.items() if d > 0)
        raise ModelingError(f"alignment_source_dag_cycle:{side_label}:{aid}:{witness}")
    return preds, succs


def _match_verified_fusions_first(
    b_list: Sequence[OperatorExportRow],
    a_list: Sequence[OperatorExportRow],
    applied_edit_receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[list[int], list[int]]], set[int], set[int]]:
    receipt_inputs_by_out: dict[str, set[str]] = defaultdict(set)
    for rec in applied_edit_receipts:
        if isinstance(rec, Mapping) and str(rec.get("status", "applied")).lower() == "applied" and str(rec.get("opcode") or "") == "FuseRegion":
            in_set = {str(x) for x in (rec.get("input_occurrences") or rec.get("affected_ops") or ()) if x}
            for out_id in (rec.get("output_occurrences") or ()):
                receipt_inputs_by_out[str(out_id)].update(in_set)

    b_by_opid: dict[str, list[int]] = defaultdict(list)
    b_by_suffix: dict[str, list[int]] = defaultdict(list)
    for bi, b_op in enumerate(b_list):
        b_by_opid[b_op.op_id].append(bi)
        b_by_suffix[_op_exact_key(b_op)].append(bi)

    pinned: list[tuple[list[int], list[int]]] = []
    claimed_b, claimed_a = set(), set()
    for ai, a_op in enumerate(a_list):
        sources: list[str] = []
        if a_op.op_id in receipt_inputs_by_out:
            sources.extend(sorted(receipt_inputs_by_out[a_op.op_id]))
        kspec = a_op.kernel_synthesis_spec
        if kspec is not None and getattr(kspec, "constituent_op_ids", ()):
            sources.extend(str(x) for x in kspec.constituent_op_ids if x)
        if len(a_op.absorbed_source_ops) >= MIN_GROUP_SIZE:
            sources.extend(str(x) for x in a_op.absorbed_source_ops if x)
        if not sources:
            continue
        a_scope = _scope_of_suffix(_op_exact_key(a_op))
        matched_b: list[int] = []
        local_seen: set[int] = set()
        for src in dict.fromkeys(sources):
            cands = [bi for bi in b_by_opid.get(src, ()) if bi not in claimed_b and bi not in local_seen]
            if not cands:
                s_key = strip_op_prefix(src).lower()
                cands = [
                    bi for bi in b_by_suffix.get(s_key, ())
                    if bi not in claimed_b and bi not in local_seen and _scope_compatible(a_scope, _scope_of_suffix(_op_exact_key(b_list[bi])))
                ]
            if cands:
                local_seen.add(cands[0])
                matched_b.append(cands[0])
        if len(matched_b) >= MIN_GROUP_SIZE or (a_op.op_id in receipt_inputs_by_out and matched_b):
            claimed_b.update(matched_b)
            claimed_a.add(ai)
            pinned.append((sorted(matched_b), [ai]))
    return pinned, claimed_b, claimed_a


def _build_bipartite_edges(
    b_list: Sequence[OperatorExportRow], a_list: Sequence[OperatorExportRow],
    claimed_b: set[int], claimed_a: set[int],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    b_atoms, a_atoms = [_op_atoms(b) for b in b_list], [_op_atoms(a) for a in a_list]
    b_keys = [_op_exact_key(b) for b in b_list]
    a_absorbed = [{strip_op_prefix(x).lower() for x in a.absorbed_source_ops if x} for a in a_list]
    b_adj: dict[int, set[int]] = defaultdict(set)
    a_adj: dict[int, set[int]] = defaultdict(set)

    for bi, b_op in enumerate(b_list):
        if bi in claimed_b:
            continue
        b_scope = _scope_of_suffix(b_keys[bi])
        for ai, a_op in enumerate(a_list):
            if ai in claimed_a or not _scope_compatible(_scope_of_suffix(_op_exact_key(a_op)), b_scope):
                continue
            if (b_atoms[bi] and (b_atoms[bi] & a_atoms[ai])) or (b_keys[bi] in a_absorbed[ai]):
                b_adj[bi].add(ai)
                a_adj[ai].add(bi)

    used_a_glue = set(claimed_a)
    for bi, b_op in enumerate(b_list):
        if bi in claimed_b or bi in b_adj or b_atoms[bi]:
            continue
        for ai, a_op in enumerate(a_list):
            if ai in a_adj or ai in used_a_glue or a_atoms[ai] or _op_exact_key(a_op) != b_keys[bi]:
                continue
            if b_op.comm_domain_tag and a_op.comm_domain_tag and b_op.comm_domain_tag != a_op.comm_domain_tag:
                continue
            b_adj[bi].add(ai)
            a_adj[ai].add(bi)
            used_a_glue.add(ai)
            break
    return b_adj, a_adj


def _extract_bipartite_components(
    n_b: int, n_a: int, b_adj: Mapping[int, set[int]], a_adj: Mapping[int, set[int]],
    claimed_b: set[int], claimed_a: set[int],
) -> list[tuple[list[int], list[int]]]:
    vis_b, vis_a = set(claimed_b), set(claimed_a)
    comps: list[tuple[list[int], list[int]]] = []
    for b_start in range(n_b):
        if b_start in vis_b:
            continue
        qb, qa, comp_b, comp_a = [b_start], [], [], []
        vis_b.add(b_start)
        while qb or qa:
            while qb:
                bi = qb.pop()
                comp_b.append(bi)
                for ai in b_adj.get(bi, ()):
                    if ai not in vis_a:
                        vis_a.add(ai)
                        qa.append(ai)
            while qa:
                ai = qa.pop()
                comp_a.append(ai)
                for bi in a_adj.get(ai, ()):
                    if bi not in vis_b:
                        vis_b.add(bi)
                        qb.append(bi)
        comps.append((sorted(comp_b), sorted(comp_a)))
    for a_start in range(n_a):
        if a_start not in vis_a:
            vis_a.add(a_start)
            comps.append(([], [a_start]))
    return comps


def _tarjan_scc(n_nodes: int, succs: Mapping[int, set[int]]) -> list[list[int]]:
    """Iterative Tarjan strongly connected components over vertices ``0..n_nodes-1``."""
    index_counter = 0
    indices, lowlink, on_stack = [-1] * n_nodes, [0] * n_nodes, [False] * n_nodes
    tarjan_stack: list[int] = []
    sccs: list[list[int]] = []
    for start in range(n_nodes):
        if indices[start] != -1:
            continue
        call_stack: list[tuple[int, Any]] = [(start, iter(sorted(succs.get(start, ()))))]
        indices[start] = lowlink[start] = index_counter
        index_counter += 1
        tarjan_stack.append(start)
        on_stack[start] = True
        while call_stack:
            v, it = call_stack[-1]
            try:
                w = next(it)
                if indices[w] == -1:
                    indices[w] = lowlink[w] = index_counter
                    index_counter += 1
                    tarjan_stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, iter(sorted(succs.get(w, ())))))
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])
            except StopIteration:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                if lowlink[v] == indices[v]:
                    comp: list[int] = []
                    while True:
                        node = tarjan_stack.pop()
                        on_stack[node] = False
                        comp.append(node)
                        if node == v:
                            break
                    sccs.append(sorted(comp))
    return sccs


def _order_members_internally(idxs: Sequence[int], side_preds: Mapping[int, set[int]]) -> list[int]:
    if len(idxs) <= 1:
        return list(idxs)
    member_set = set(idxs)
    in_deg = {i: len(side_preds[i] & member_set) for i in idxs}
    local_succ: dict[int, list[int]] = {i: [] for i in idxs}
    for i in idxs:
        for p in side_preds[i] & member_set:
            local_succ[p].append(i)
    ready = sorted(i for i in idxs if in_deg[i] == 0)
    ordered: list[int] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in local_succ[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(ordered) < len(idxs):
        ordered.extend(i for i in sorted(idxs) if i not in set(ordered))
    return ordered


def _compute_same_submodule_branches(n_verts: int, submodules: Sequence[str], edges: Mapping[int, set[int]]) -> list[int]:
    undir: dict[int, set[int]] = defaultdict(set)
    for u, vs in edges.items():
        for v in vs:
            if submodules[u] and submodules[u] == submodules[v]:
                undir[u].add(v)
                undir[v].add(u)
    branch, next_bid = [-1] * n_verts, 0
    for start in range(n_verts):
        if branch[start] != -1:
            continue
        stack, branch[start] = [start], next_bid
        while stack:
            u = stack.pop()
            for w in undir.get(u, ()):
                if branch[w] == -1:
                    branch[w] = next_bid
                    stack.append(w)
        next_bid += 1
    return branch


def _toposort_condensed_vertices(
    verts: Sequence[tuple[list[int], list[int]]], b_list: Sequence[OperatorExportRow],
    a_list: Sequence[OperatorExportRow], v_succs: Mapping[int, set[int]], v_preds: Mapping[int, set[int]],
) -> list[int]:
    """Stable Kahn topological order with same-submodule branch locality."""
    n_v = len(verts)
    submodules = [(a_list[a[0]] if a else b_list[b[0]]).submodule for b, a in verts]
    stable_keys = [(a_list[a[0]] if a else b_list[b[0]]).op_id for b, a in verts]

    a_to_b_anchor = {ai: float(b[0]) for b, a in verts if b and a for ai in a}
    added_anchor: dict[int, float] = {}
    last_anchor = -1.0
    for ai in range(len(a_list)):
        if ai in a_to_b_anchor:
            last_anchor = a_to_b_anchor[ai]
        else:
            added_anchor[ai] = (last_anchor + ADDED_ROW_TIE_OFFSET) if last_anchor >= 0.0 else float(len(b_list) + ai)

    pos_keys = [
        (float(b[0]), 0 if a else 1, a[0] if a else -1) if b
        else (added_anchor.get(a[0], float(len(b_list) + a[0])), 2, a[0])
        for b, a in verts
    ]
    branches = _compute_same_submodule_branches(n_v, submodules, v_succs)
    in_deg = {v: len(v_preds.get(v, ())) for v in range(n_v)}
    ready = {v for v, d in in_deg.items() if d == 0}
    order, cur_sub, cur_branch = [], "", -1
    while ready:
        pick = min(
            ready,
            key=lambda v: (
                0 if (cur_sub and submodules[v] == cur_sub and branches[v] == cur_branch) else 1,
                0 if (cur_sub and submodules[v] == cur_sub) else 1,
                pos_keys[v], stable_keys[v],
            ),
        )
        ready.remove(pick)
        order.append(pick)
        cur_sub, cur_branch = submodules[pick], branches[pick]
        for nxt in v_succs.get(pick, ()):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.add(nxt)
    return order


def _align_archetype_bipartite(
    aid: str, title: str, rep: int, b_list: Sequence[OperatorExportRow], a_list: Sequence[OperatorExportRow],
    *, before_ntok: float = 1.0, after_ntok: float = 1.0, before_rep: int | None = None,
    after_rep: int | None = None, applied_edit_receipts: Sequence[Mapping[str, Any]] = (),
    use_op_repeats: bool = False,
) -> list[AlignedComparisonRow]:
    """Align one slice via receipt-first bipartite matching + D5a topological coalescing."""
    del before_ntok, after_ntok
    b_w_default = max(1, int(before_rep)) if before_rep is not None else max(1, int(rep))
    a_w_default = max(1, int(after_rep)) if after_rep is not None else max(1, int(rep))

    b_preds, b_succs = _build_side_dag(b_list, "before", aid)
    a_preds, a_succs = _build_side_dag(a_list, "after", aid)
    pinned, claimed_b, claimed_a = _match_verified_fusions_first(b_list, a_list, applied_edit_receipts)
    b_adj, a_adj = _build_bipartite_edges(b_list, a_list, claimed_b, claimed_a)
    raw_groups = [*pinned, *_extract_bipartite_components(len(b_list), len(a_list), b_adj, a_adj, claimed_b, claimed_a)]
    if not raw_groups:
        return []

    b_owner = {bi: gid for gid, (bs, _) in enumerate(raw_groups) for bi in bs}
    a_owner = {ai: gid for gid, (_, as_) in enumerate(raw_groups) for ai in as_}
    q_succs: dict[int, set[int]] = defaultdict(set)
    for owner, side_succs in ((b_owner, b_succs), (a_owner, a_succs)):
        for i, succ_set in side_succs.items():
            u = owner[i]
            for j in succ_set:
                if u != owner[j]:
                    q_succs[u].add(owner[j])

    sccs = _tarjan_scc(len(raw_groups), q_succs)
    gid_to_vid: dict[int, int] = {}
    verts: list[tuple[list[int], list[int]]] = []
    for vid, comp_gids in enumerate(sccs):
        mb, ma = [], []
        for gid in comp_gids:
            gid_to_vid[gid] = vid
            mb.extend(raw_groups[gid][0])
            ma.extend(raw_groups[gid][1])
        verts.append((_order_members_internally(sorted(mb), b_preds), _order_members_internally(sorted(ma), a_preds)))

    v_succs: dict[int, set[int]] = defaultdict(set)
    v_preds: dict[int, set[int]] = defaultdict(set)
    for u_gid, succ_gids in q_succs.items():
        u_v = gid_to_vid[u_gid]
        for v_gid in succ_gids:
            v_v = gid_to_vid[v_gid]
            if u_v != v_v:
                v_succs[u_v].add(v_v)
                v_preds[v_v].add(u_v)

    rows: list[AlignedComparisonRow] = []
    for vid in _toposort_condensed_vertices(verts, b_list, a_list, v_succs, v_preds):
        b_idxs, a_idxs = verts[vid]
        b_tup, a_tup = tuple(b_list[i] for i in b_idxs), tuple(a_list[j] for j in a_idxs)
        nb, na = len(b_tup), len(a_tup)
        if use_op_repeats:
            b_w = max(1, int(b_tup[0].repeat_count)) if b_tup else 0
            a_w = max(1, int(a_tup[0].repeat_count)) if a_tup else 0
        else:
            b_w = b_w_default if b_tup else 0
            a_w = a_w_default if a_tup else 0
        row_rep = a_w if a_tup else b_w
        if nb == 1 and na == 1:
            glue = _detect_added_glue_note(b_tup[0], a_tup[0])
            note = _build_factual_action_note("1->1", b_tup, a_tup, added_glue_note=glue)
            rows.append(AlignedComparisonRow(aid, title, row_rep, a_tup[0].submodule, b_tup, a_tup, "1->1", note, added_glue_note=glue, before_repeat_count=b_w, after_repeat_count=a_w))
        elif nb >= MIN_GROUP_SIZE and na == 1:
            rows.append(AlignedComparisonRow(aid, title, row_rep, a_tup[0].submodule, b_tup, a_tup, "N->1", _build_factual_action_note("N->1", b_tup, a_tup), before_repeat_count=b_w, after_repeat_count=a_w))
        elif nb == 1 and na == 0:
            rows.append(AlignedComparisonRow(aid, title, b_w, b_tup[0].submodule, b_tup, (), "eliminated", _build_factual_action_note("eliminated", b_tup, ()), before_repeat_count=b_w, after_repeat_count=0))
        elif nb == 0 and na == 1:
            rows.append(AlignedComparisonRow(aid, title, a_w, a_tup[0].submodule, (), a_tup, "added", _build_factual_action_note("added", (), a_tup), before_repeat_count=0, after_repeat_count=a_w))
        else:
            rel = f"{nb}->{na}"
            sub = (a_tup[0] if a_tup else b_tup[0]).submodule
            rows.append(AlignedComparisonRow(aid, title, row_rep, sub, b_tup, a_tup, rel, _build_factual_action_note(rel, b_tup, a_tup), before_repeat_count=b_w if b_tup else 0, after_repeat_count=a_w if a_tup else 0))
    return rows


def _scale_slice_op_count(op: OperatorExportRow, cnt: int) -> OperatorExportRow:
    if int(op.repeat_count) <= cnt:
        return op
    dur_ns = max(0, int(round(float(op.dur_us) * float(cnt) * float(NS_PER_US))))
    exp_ns = 0 if op.is_overlapped else max(0, int(round(float(op.exposed_dur_us) * float(cnt) * float(NS_PER_US))))
    ls_ns = 0 if op.is_overlapped else max(0, int(round(float(op.l_star_lean_us) * float(cnt) * float(NS_PER_US))))
    return replace(op, repeat_count=cnt, dur_ns=dur_ns, exposed_ns=exp_ns, l_star_ns=ls_ns)


def _reaggregate_slice_ops(
    bundle: ExportBundle,
    ops: Sequence[OperatorExportRow],
    idxs: Sequence[int],
    *,
    full_arch_len: int = 0,
) -> tuple[OperatorExportRow, ...]:
    """Reaggregate depth-varying slice operators over their actual physical layer indices."""
    per_layer = getattr(bundle, "per_layer_timings", None)
    if not ops or not idxs or not isinstance(per_layer, Mapping) or not per_layer:
        return tuple(ops)
    ordered_idxs = sorted(int(x) for x in idxs)
    cnt = len(ordered_idxs)
    reagg: list[OperatorExportRow] = []
    for op in ops:
        site = strip_op_prefix(op.op_id)
        durs: list[float] = []
        durs_ns: list[int] = []
        exps_ns: list[int] = []
        lstars_ns: list[int] = []
        matched_idxs: list[int] = []
        block_size = 0
        for lyr in ordered_idxs:
            rec = per_layer.get((op.model_scope, lyr))
            if not isinstance(rec, Mapping):
                continue
            ops_map = rec.get("ops") or {}
            if site not in ops_map:
                continue
            d_val = float(ops_map[site][1])
            durs.append(d_val)
            matched_idxs.append(lyr)
            dur_ns_map = rec.get("ops_dur_ns")
            durs_ns.append(
                int(dur_ns_map[site])
                if (isinstance(dur_ns_map, Mapping) and site in dur_ns_map)
                else max(0, int(round(d_val * float(NS_PER_US))))
            )
            exp_ns_map = rec.get("ops_exposed_ns")
            exps_map = rec.get("ops_exposed")
            if isinstance(exp_ns_map, Mapping) and site in exp_ns_map:
                exps_ns.append(int(exp_ns_map[site]))
            elif isinstance(exps_map, Mapping) and site in exps_map:
                exps_ns.append(max(0, int(round(float(exps_map[site]) * float(NS_PER_US)))))
            else:
                exps_ns.append(0 if op.is_overlapped else durs_ns[-1])
            ls_ns_map = rec.get("ops_lstar_ns")
            ls_map = rec.get("ops_lstar")
            if isinstance(ls_ns_map, Mapping) and site in ls_ns_map:
                lstars_ns.append(int(ls_ns_map[site]))
            elif isinstance(ls_map, Mapping) and site in ls_map:
                lstars_ns.append(max(0, int(round(float(ls_map[site]) * float(NS_PER_US)))))
            else:
                lstars_ns.append(max(0, int(round(float(op.l_star_lean_us) * float(NS_PER_US)))))
            bs = int(rec.get("residual_block_size", 0) or 0)
            if bs > 0:
                block_size = bs
        if not durs:
            if full_arch_len > 0 and int(op.repeat_count) < full_arch_len:
                continue
            reagg.append(_scale_slice_op_count(op, cnt))
            continue
        eff_cnt = len(durs)
        if eff_cnt != cnt and not (full_arch_len > 0 and int(op.repeat_count) < full_arch_len):
            reagg.append(_scale_slice_op_count(op, cnt))
            continue
        tot_dur_ns = sum(durs_ns)
        tot_exp_ns = sum(exps_ns)
        tot_lstar_ns = sum(lstars_ns)
        denom_us = float(eff_cnt) * float(NS_PER_US)
        mean_dur = float(tot_dur_ns) / denom_us
        mean_exp = float(tot_exp_ns) / denom_us
        mean_lstar = float(tot_lstar_ns) / denom_us
        fully_hidden = mean_dur > OVERLAP_EXPOSED_EPS_US and mean_exp <= OVERLAP_EXPOSED_EPS_US
        eff = min(1.0, mean_lstar / mean_dur) if (mean_dur > 0.0 and mean_lstar > 0.0) else None
        base_contract = _DEPTH_NOTE_STRIP_RE.sub("", op.tensor_contract or "").strip(" |")
        depth_note = format_attnres_depth_note(
            op.op_id, matched_idxs, durs, block_size,
            total_target_layers=int(getattr(bundle, "total_target_layers", 0) or 0),
            family=op.family,
        )
        new_contract = (
            f"{base_contract} | {depth_note}"
            if (base_contract and depth_note)
            else (depth_note or base_contract)
        )
        reagg.append(replace(
            op,
            repeat_count=eff_cnt,
            end_us=op.start_us + mean_dur,
            dur_us=mean_dur,
            exposed_dur_us=0.0 if fully_hidden else mean_exp,
            is_overlapped=fully_hidden,
            l_star_lean_us=mean_lstar,
            gap_us=0.0 if fully_hidden else max(0.0, mean_dur - mean_lstar),
            gap_ratio=None if fully_hidden else ((mean_dur / mean_lstar) if mean_lstar > 0.0 else None),
            roofline_efficiency=eff,
            tensor_contract=new_contract,
            dur_ns=tot_dur_ns,
            exposed_ns=0 if fully_hidden else tot_exp_ns,
            l_star_ns=0 if fully_hidden else tot_lstar_ns,
        ))
    return tuple(reagg)


def _strip_trailing_repeat_tag(raw_title: str) -> str:
    """Strip trailing ``(×N, ...)`` tag including nested parenthesized counts like ``(55L)``."""
    trimmed = raw_title.strip()
    marker = " (×"
    if trimmed.endswith(")") and marker in trimmed:
        return trimmed[:trimmed.rfind(marker)].strip()
    return re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", trimmed).strip()


def _format_slice_layer_range(indices: Sequence[int]) -> str:
    ordered = sorted({int(x) for x in indices})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f"L{ordered[0]}"
    return ", ".join(f"L{x}" for x in ordered) if len(ordered) <= SLICE_MAX_EXPLICIT_LAYERS else f"L{ordered[0]}..L{ordered[-1]}"


def _index_target_layer_archetypes(
    bundle: ExportBundle, ops_by_arch: Mapping[str, Sequence[OperatorExportRow]],
) -> tuple[dict[int, tuple[str, str, tuple[OperatorExportRow, ...]]], set[str]]:
    layer_map: dict[int, tuple[str, str, tuple[OperatorExportRow, ...]]] = {}
    handled_aids: set[str] = set()
    for arch in bundle.archetypes:
        if arch.scope != "target" or not arch.layer_indices:
            continue
        ops = tuple(ops_by_arch.get(arch.archetype_id, ()))
        if not ops:
            continue
        handled_aids.add(arch.archetype_id)
        distinct_layers = {int(o.layer) for o in ops if o.layer is not None}
        max_rep = max((int(o.repeat_count) for o in ops), default=0)
        n_arch = len(arch.layer_indices)
        is_unaggregated_per_layer = (
            n_arch > 1 and max_rep == 1 and len(distinct_layers) > 1
        )
        if not is_unaggregated_per_layer and all(1 <= int(o.repeat_count) <= n_arch for o in ops):
            for phys_idx in arch.layer_indices:
                layer_map[int(phys_idx)] = (arch.archetype_id, arch.title, ops)
            continue
        by_layer: dict[int, list[OperatorExportRow]] = defaultdict(list)
        for op in ops:
            if op.layer is not None:
                by_layer[int(op.layer)].append(op)
        for phys_idx in arch.layer_indices:
            l_ops = tuple(by_layer.get(int(phys_idx), ()))
            if l_ops:
                layer_map[int(phys_idx)] = (f"{arch.archetype_id}@L{phys_idx}", arch.title, l_ops)
    return layer_map, handled_aids


def _build_target_layer_slices(
    before_bundle: ExportBundle, after_bundle: ExportBundle,
    b_layer_map: Mapping[int, tuple[str, str, tuple[OperatorExportRow, ...]]],
    a_layer_map: Mapping[int, tuple[str, str, tuple[OperatorExportRow, ...]]],
) -> list[tuple[str, str, int, int, tuple[OperatorExportRow, ...], tuple[OperatorExportRow, ...], bool]]:
    b_arch_meta = {a.archetype_id: a for a in before_bundle.archetypes}
    a_arch_meta = {a.archetype_id: a for a in after_bundle.archetypes}
    grouped: dict[tuple[str | None, str | None], list[int]] = {}
    for phys_idx in sorted(set(b_layer_map) | set(a_layer_map)):
        grouped.setdefault((b_layer_map[phys_idx][0] if phys_idx in b_layer_map else None, a_layer_map[phys_idx][0] if phys_idx in a_layer_map else None), []).append(phys_idx)

    slices, used_ids = [], set()
    for (b_key, a_key), idxs in grouped.items():
        rep = len(idxs)
        b_full_len = len(b_arch_meta[b_key].layer_indices) if (b_key is not None and b_key in b_arch_meta) else 0
        a_full_len = len(a_arch_meta[a_key].layer_indices) if (a_key is not None and a_key in a_arch_meta) else 0
        b_needs_reagg = b_full_len > 0 and rep != b_full_len
        a_needs_reagg = a_full_len > 0 and rep != a_full_len
        b_ops = (
            _reaggregate_slice_ops(before_bundle, b_layer_map[idxs[0]][2], idxs, full_arch_len=b_full_len)
            if b_needs_reagg else (b_layer_map[idxs[0]][2] if b_key is not None else ())
        )
        a_ops = (
            _reaggregate_slice_ops(after_bundle, a_layer_map[idxs[0]][2], idxs, full_arch_len=a_full_len)
            if a_needs_reagg else (a_layer_map[idxs[0]][2] if a_key is not None else ())
        )
        same_unsplit = (
            b_key is not None and b_key == a_key and b_key in b_arch_meta and b_key in a_arch_meta
            and rep == b_full_len == a_full_len
        )
        if same_unsplit:
            slice_aid, slice_title = b_key, a_arch_meta[b_key].title
        else:
            base_aid = a_key or b_key or "target_slice"
            slice_aid = base_aid if base_aid not in used_ids else f"{b_key or 'none'}__{a_key or 'none'}"
            raw_title = a_layer_map[idxs[0]][1] if a_key is not None else b_layer_map[idxs[0]][1]
            clean_title = _strip_trailing_repeat_tag(raw_title)
            rng_txt = _format_slice_layer_range(idxs)
            slice_title = f"{clean_title} (×{rep}, {rng_txt})" if rng_txt else f"{clean_title} (×{rep})"
        has_subset_repeats = any(int(o.repeat_count) < rep for o in b_ops) or any(int(o.repeat_count) < rep for o in a_ops)
        used_ids.add(slice_aid)
        slices.append((slice_aid, slice_title, rep if b_ops else 0, rep if a_ops else 0, b_ops, a_ops, same_unsplit or has_subset_repeats))
    return slices


def align_before_after_operators(before_bundle: ExportBundle, after_bundle: ExportBundle) -> tuple[AlignedComparisonRow, ...]:
    """Align Before vs After via physical layer-index slicing + D5a topological coalescing."""
    before_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    after_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    for op in before_bundle.operators:
        before_by_arch[op.archetype_id or op.layer_label].append(op)
    for op in after_bundle.operators:
        after_by_arch[op.archetype_id or op.layer_label].append(op)

    b_layer_map, b_handled = _index_target_layer_archetypes(before_bundle, before_by_arch)
    a_layer_map, a_handled = _index_target_layer_archetypes(after_bundle, after_by_arch)
    handled_target_aids = b_handled | a_handled
    target_slices = _build_target_layer_slices(before_bundle, after_bundle, b_layer_map, a_layer_map) if (b_layer_map or a_layer_map) else []

    pre_slices, post_slices, seen_arch, seen_target = [], [], set(), False
    for src in (before_bundle, after_bundle):
        for op in src.operators:
            aid = op.archetype_id or op.layer_label
            if aid in handled_target_aids:
                seen_target = True
                continue
            if aid in seen_arch:
                continue
            seen_arch.add(aid)
            b_ops, a_ops = tuple(before_by_arch.get(aid, ())), tuple(after_by_arch.get(aid, ()))
            entry = (aid, op.archetype_title or op.layer_label, max(1, int(b_ops[0].repeat_count)) if b_ops else 0, max(1, int(a_ops[0].repeat_count)) if a_ops else 0, b_ops, a_ops, True)
            (pre_slices if (not seen_target and target_slices) else post_slices).append(entry)

    b_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(after_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    receipts = tuple(after_bundle.applied_edit_receipts or ())
    aligned: list[AlignedComparisonRow] = []
    for aid, title, b_rep, a_rep, b_ops, a_ops, use_op_rep in (*pre_slices, *target_slices, *post_slices):
        aligned.extend(_align_archetype_bipartite(
            aid, title, a_rep or b_rep or 1, b_ops, a_ops,
            before_ntok=b_ntok, after_ntok=a_ntok, before_rep=b_rep or 1, after_rep=a_rep or 1,
            applied_edit_receipts=receipts,
            use_op_repeats=use_op_rep,
        ))
    return tuple(aligned)


MACRO_MOD_IDLE_WAIT = "Idle / Host Wait / Bubble (idle_or_wait)"


def _accumulate_macro_stats(bundle: ExportBundle) -> tuple[dict[str, dict[str, Any]], int, int]:
    arch_by_id = {a.archetype_id: a for a in bundle.archetypes}
    stats: dict[str, dict[str, Any]] = {}
    op_exp_total_ns = 0
    for op in bundle.operators:
        mod = classify_macro_module(op)
        bucket = stats.setdefault(mod, {
            "exposed_ns": 0, "raw_ns": 0, "lstar_ns": 0, "bound_ms": defaultdict(float),
            "masker_ms": defaultdict(float), "target_layer_indices": set(), "draft_layers": 0, "has_head_or_embed": False,
        })
        rep = max(1, int(op.repeat_count))
        exp_ns = op_exposed_int_ns(op, rep)
        raw_ns = op_dur_int_ns(op, rep)
        ls_ns = op_lstar_int_ns(op, rep)
        exp_ms = float(exp_ns) / float(NS_PER_MS)
        raw_ms = float(raw_ns) / float(NS_PER_MS)
        masked_pos_ms = max(0.0, raw_ms - exp_ms)
        bucket["exposed_ns"] += exp_ns
        bucket["raw_ns"] += raw_ns
        bucket["lstar_ns"] += ls_ns
        op_exp_total_ns += exp_ns
        if exp_ns > 0:
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
            bucket["target_layer_indices"].update(int(x) for x in arch.layer_indices)
        else:
            bucket["has_head_or_embed"] = True
    step_ns = max(bundle_step_int_ns(bundle), op_exp_total_ns)
    idle_ns = max(0, step_ns - op_exp_total_ns)
    return stats, step_ns, idle_ns


def _format_macro_layer_label(
    mod: str, target_layer_indices: set[int], draft_layers: int, has_head: bool, bundle: ExportBundle,
) -> tuple[str, int]:
    if mod == MACRO_MOD_DRAFT_LAYERS:
        d_cnt = draft_layers or bundle.total_draft_layers
        return (f"Draft {d_cnt}L", d_cnt) if d_cnt > 0 else ("Draft ×1", 1)
    if mod == MACRO_MOD_VERIFY_HEAD:
        return "Embed + Head (×1)", 1
    if target_layer_indices:
        return f"{len(target_layer_indices)}L", len(target_layer_indices)
    return ("×1", 1) if has_head else ("0L", 0)


def _build_masking_note(
    mod: str, raw_aft: float, aft_ms: float, raw_bef: float, bef_ms: float,
    a_maskers: Mapping[str, float], b_maskers: Mapping[str, float], dom_bound: str,
) -> str:
    aft_masked, bef_masked = max(0.0, raw_aft - aft_ms), max(0.0, raw_bef - bef_ms)
    if aft_masked > CONSERVATION_EPS_MS:
        top_m = max(a_maskers.items(), key=lambda kv: kv[1])[0] if a_maskers else "主计算流"
        if aft_ms <= OVERLAP_EXPOSED_EPS_US:
            prefix = "MSO 副流被" if mod == MACRO_MOD_MOE_SHARED else "被"
            return f"{prefix} {top_m} 100% 掩盖 (-{aft_masked:.2f}ms，不计入总耗时)"
        pct = 100.0 * aft_masked / max(raw_aft, CONSERVATION_EPS_MS)
        return f"{dom_bound} | 被 {top_m} 掩盖 {pct:.0f}% (-{aft_masked:.2f}ms)"
    if bef_masked > CONSERVATION_EPS_MS:
        top_m = max(b_maskers.items(), key=lambda kv: kv[1])[0] if b_maskers else "主计算流"
        return f"{dom_bound} | 优化前被 {top_m} 掩盖 -{bef_masked:.2f}ms"
    return dom_bound


def build_macro_module_breakdown(
    before_bundle: ExportBundle, after_bundle: ExportBundle | None = None,
) -> tuple[MacroModuleRow, ...]:
    """Aggregate L1/L2 Raw, Masked, Exposed ms and ``us / (tok*die)`` with 0-ns conservation."""
    bef_stats, tot_bef_ns, idle_b_ns = _accumulate_macro_stats(before_bundle)
    aft_bundle = after_bundle or before_bundle
    if after_bundle is None:
        aft_stats, tot_aft_ns, idle_a_ns = bef_stats, tot_bef_ns, idle_b_ns
    else:
        aft_stats, tot_aft_ns, idle_a_ns = _accumulate_macro_stats(after_bundle)
    b_ntok = max(float(before_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    a_ntok = max(float(aft_bundle.n_tok_per_die or 1.0), MIN_DENOM_EPS)
    tot_bef = float(tot_bef_ns) / float(NS_PER_MS)
    tot_aft = float(tot_aft_ns) / float(NS_PER_MS)
    l1_by_mod = {mod: l1 for l1, mods in MACRO_L1_HIERARCHY for mod in mods}
    ordered_mods = list(dict.fromkeys([*(mod for _, mods in MACRO_L1_HIERARCHY for mod in mods), *bef_stats.keys(), *aft_stats.keys()]))

    rows: list[MacroModuleRow] = []
    for mod in ordered_mods:
        b_info, a_info = bef_stats.get(mod), aft_stats.get(mod)
        if b_info is None and a_info is None:
            continue
        bef_ns = int(b_info["exposed_ns"]) if b_info else 0
        aft_ns = int(a_info["exposed_ns"]) if a_info else 0
        raw_bef_ns = int(b_info["raw_ns"]) if b_info else 0
        raw_aft_ns = int(a_info["raw_ns"]) if a_info else 0
        lstar_ns = int(a_info["lstar_ns"]) if a_info else 0
        if raw_bef_ns <= 0 and raw_aft_ns <= 0:
            continue
        bef_ms, aft_ms = float(bef_ns) / float(NS_PER_MS), float(aft_ns) / float(NS_PER_MS)
        raw_bef, raw_aft = float(raw_bef_ns) / float(NS_PER_MS), float(raw_aft_ns) / float(NS_PER_MS)
        lstar_ms = float(lstar_ns) / float(NS_PER_MS)
        merged_indices = (set(b_info["target_layer_indices"]) if b_info else set()) | (set(a_info["target_layer_indices"]) if a_info else set())
        d_layers = max(int(b_info["draft_layers"]) if b_info else 0, int(a_info["draft_layers"]) if a_info else 0)
        has_head = bool((b_info and b_info["has_head_or_embed"]) or (a_info and a_info["has_head_or_embed"]))
        layer_lbl, layer_cnt = _format_macro_layer_label(mod, merged_indices, d_layers, has_head, aft_bundle)
        bw = a_info["bound_ms"] if (a_info and a_info["bound_ms"]) else (b_info["bound_ms"] if b_info else {})
        if mod == MACRO_MOD_MOE_SHARED and aft_ms <= OVERLAP_EXPOSED_EPS_US and raw_aft > 0.0:
            dom_bound = f"MSO Stream 1 100% 掩盖 (-{raw_aft:.3f}ms)"
        else:
            dom_bound = max(bw.items(), key=lambda kv: kv[1])[0] if bw else "已融合 / 100% 掩盖"
        mask_note = _build_masking_note(mod, raw_aft, aft_ms, raw_bef, bef_ms, a_info["masker_ms"] if a_info else {}, b_info["masker_ms"] if b_info else {}, dom_bound)
        b_upt = float(bef_ns) / (float(NS_PER_US) * b_ntok)
        a_upt = float(aft_ns) / (float(NS_PER_US) * a_ntok)
        rows.append(MacroModuleRow(
            module_name=mod, layer_calls_label=layer_lbl, layer_count=layer_cnt,
            before_ms=bef_ms, before_share_pct=(100.0 * bef_ms / tot_bef) if tot_bef > 0.0 else 0.0,
            after_ms=aft_ms, after_share_pct=(100.0 * aft_ms / tot_aft) if tot_aft > 0.0 else 0.0,
            saved_ms=float(bef_ns - aft_ns) / float(NS_PER_MS), lstar_ms=lstar_ms, dominant_bound=dom_bound,
            raw_before_ms=raw_bef, raw_after_ms=raw_aft, l1_group=l1_by_mod.get(mod, L1_MOD_RESIDUAL_MEM),
            masked_before_ms=float(bef_ns - raw_bef_ns) / float(NS_PER_MS),
            masked_after_ms=float(aft_ns - raw_aft_ns) / float(NS_PER_MS), masking_note=mask_note,
            before_us_per_tok_die=b_upt, after_us_per_tok_die=a_upt, saved_us_per_tok_die=b_upt - a_upt,
        ))
    if idle_b_ns > 0 or idle_a_ns > 0:
        ib_ms, ia_ms = float(idle_b_ns) / float(NS_PER_MS), float(idle_a_ns) / float(NS_PER_MS)
        ib_upt = float(idle_b_ns) / (float(NS_PER_US) * b_ntok)
        ia_upt = float(idle_a_ns) / (float(NS_PER_US) * a_ntok)
        rows.append(MacroModuleRow(
            module_name=MACRO_MOD_IDLE_WAIT, layer_calls_label="Step Wall", layer_count=0,
            before_ms=ib_ms, before_share_pct=(100.0 * ib_ms / tot_bef) if tot_bef > 0.0 else 0.0,
            after_ms=ia_ms, after_share_pct=(100.0 * ia_ms / tot_aft) if tot_aft > 0.0 else 0.0,
            saved_ms=float(idle_b_ns - idle_a_ns) / float(NS_PER_MS), lstar_ms=0.0,
            dominant_bound="Host / Sync / Bubble", raw_before_ms=ib_ms, raw_after_ms=ia_ms,
            l1_group=L1_MOD_RESIDUAL_MEM, masked_before_ms=0.0, masked_after_ms=0.0,
            masking_note="Host / Sync / Bubble",
            before_us_per_tok_die=ib_upt, after_us_per_tok_die=ia_upt, saved_us_per_tok_die=ib_upt - ia_upt,
        ))
    return tuple(rows)


_CANONICAL_TRACK_RANK: dict[str, int] = {
    "eager": 0,
    "shipping": 1,
    "proposed": 2,
    "target": 3,
}
_VALID_COMPARISON_POLICIES = frozenset({"monotonic_optimization", "signed_comparison"})


def _bundle_canonical_track(bundle: ExportBundle) -> str:
    raw = str(getattr(bundle, "feature_track", "") or "").strip().lower()
    if not raw:
        return "target" if bool(getattr(bundle, "target_envelope_applied", False)) else "shipping"
    if raw not in _CANONICAL_TRACK_RANK:
        raise ModelingError(f"invalid_comparison_track:{bundle.feature_track!r}")
    return raw


def build_default_comparison_request(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    *,
    comparison_policy: str = "monotonic_optimization",
    require_model_scope: bool | None = None,
) -> ComparisonRequest:
    """Build a role-tagged ``ComparisonRequest`` binding Before and After bundle digests."""
    eff_req_model = (
        bool(require_model_scope)
        if require_model_scope is not None
        else (before_bundle.scope == "model" and after_bundle.scope == "model")
    )
    return ComparisonRequest(
        before_case_digest=str(getattr(before_bundle, "case_digest", "") or ""),
        after_case_digest=str(getattr(after_bundle, "case_digest", "") or ""),
        before_execution_key=str(getattr(before_bundle, "execution_key_sha256", "") or ""),
        after_execution_key=str(getattr(after_bundle, "execution_key_sha256", "") or ""),
        before_plan_sha256=str(getattr(before_bundle, "rewrite_plan_sha256", "") or ""),
        after_plan_sha256=str(getattr(after_bundle, "rewrite_plan_sha256", "") or ""),
        comparison_policy=comparison_policy,
        require_model_scope=eff_req_model,
    )


def assert_comparison_polarity(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    comparison_request: ComparisonRequest,
) -> None:
    """Fail-loud 3-argument role-tagged polarity & scope guard for Before vs After comparisons."""
    if not isinstance(comparison_request, ComparisonRequest):
        raise ModelingError("comparison_request_invalid:expected ComparisonRequest")
    if comparison_request.comparison_policy not in _VALID_COMPARISON_POLICIES:
        raise ModelingError(f"comparison_policy_invalid:{comparison_request.comparison_policy!r}")
    b_dig = str(getattr(before_bundle, "case_digest", "") or "")
    a_dig = str(getattr(after_bundle, "case_digest", "") or "")
    if comparison_request.before_case_digest and b_dig != comparison_request.before_case_digest:
        raise ModelingError("export_comparison_reversed_before_after:bundle_case_digest_role_mismatch")
    if comparison_request.after_case_digest and a_dig != comparison_request.after_case_digest:
        raise ModelingError("export_comparison_reversed_before_after:bundle_case_digest_role_mismatch")
    b_ek = str(getattr(before_bundle, "execution_key_sha256", "") or "")
    a_ek = str(getattr(after_bundle, "execution_key_sha256", "") or "")
    if comparison_request.before_execution_key and b_ek != comparison_request.before_execution_key:
        raise ModelingError("export_comparison_reversed_before_after:bundle_execution_key_role_mismatch")
    if comparison_request.after_execution_key and a_ek != comparison_request.after_execution_key:
        raise ModelingError("export_comparison_reversed_before_after:bundle_execution_key_role_mismatch")
    if comparison_request.before_plan_sha256 and before_bundle.rewrite_plan_sha256 != comparison_request.before_plan_sha256:
        raise ModelingError("export_comparison_reversed_before_after:bundle_plan_sha256_role_mismatch")
    if comparison_request.after_plan_sha256 and after_bundle.rewrite_plan_sha256 != comparison_request.after_plan_sha256:
        raise ModelingError("export_comparison_reversed_before_after:bundle_plan_sha256_role_mismatch")
    if comparison_request.require_model_scope and (before_bundle.scope != "model" or after_bundle.scope != "model"):
        raise ModelingError(
            f"export_comparison_requires_model_scope:before={before_bundle.scope},after={after_bundle.scope}"
        )
    if before_bundle.scope != after_bundle.scope:
        raise ModelingError(
            f"export_comparison_scope_mismatch:before={before_bundle.scope},after={after_bundle.scope}"
        )
    if comparison_request.comparison_policy == "monotonic_optimization":
        b_track = _bundle_canonical_track(before_bundle)
        a_track = _bundle_canonical_track(after_bundle)
        if before_bundle.target_envelope_applied and not after_bundle.target_envelope_applied:
            raise ModelingError("export_comparison_reversed_before_after:target_envelope_inverted")
        if _CANONICAL_TRACK_RANK[b_track] > _CANONICAL_TRACK_RANK[a_track]:
            raise ModelingError(f"export_comparison_reversed_before_after:{b_track}>{a_track}")


__all__ = [
    "AGENT_BRIEF_MAX_BYTES", "ComparisonPlan", "ComparisonRequest", "GroupAccounting",
    "WF_STAGE_CONFIG", "WF_STAGE_FRAMEWORK", "WF_STAGE_KERNEL", "WF_STAGE_RESIDUAL",
    "WaterfallSummary", "_compact_agent_brief_to_budget", "align_before_after_operators",
    "assert_comparison_polarity", "build_actionable_roadmap",
    "build_agent_brief", "build_comparison_plan", "build_default_comparison_request",
    "build_macro_module_breakdown", "build_new_kernel_backlog",
    "build_parallelism_codesign_note", "build_pending_kernel_details",
    "classify_waterfall_stage", "export_agent_brief",
    "format_executive_digest_markdown", "format_parallelism_codomain_summary",
    "format_public_track_label",
]
