"""3-Pillar ``After (expected_ideal)`` target-state optimizer & 3-track Ascend guidance.

[Role] Synthesize physically grounded post-optimization ``After`` operator graphs
  for ``expected_ideal`` export bundles on top of ``runtime.Engine`` timelines:
  - Pillar 1: Complete subgraph & cross-op fusions (``FusedGDNPrologue``,
    ``FusedRMSNormGatedQuant``, ``BlockAttnResFusedPrefix``,
    ``GroupedMatmulSituQuant`` / ``GroupedMatmulSwiGLUQuant``,
    ``RmsNormDynamicMxQuant``);
  - Pillar 2: Low-efficiency single-kernel roofline target envelopes grounded in
    certified Lean ``L*`` and SKU L0 launch minimums;
  - Pillar 3: Superpod ``CCU / AIV UB-Direct / HCCS One-Shot`` collective
    upgrades and ``MC2`` compute-communication overlap;
  - Exact DAG timeline rescheduling + Primary-Branch Interval Attribution (100%
    step-span conservation across Sheet 1, Sheet 2, Sheet 4 & ``trace_view.json``);
  - Concrete 3-track Ascend engineering guidance (``Config & Env``, ``Kernel``,
    ``Framework``) and Parallelism Co-Design callouts.
[Invariants] Strictly a top-layer export pass; never mutates caller ``Case`` or
  physics-core engine modules. Never degrades any operator duration
  (``T_after <= T_before``), derives launch/collective floors from the active
  SKU profile, and guarantees ``sum(exposed_dur_us * repeat_count) == step_span_us``.
[Timeouts & Locks] Pure O(Archetype Ops) in-memory passes; no I/O, locks, or
  device contexts.
[Side-Effect Policy] Zero filesystem, network, or global-state side effects.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.schema.case import Case

from .kernel_catalog import strip_op_prefix
from .models import (
    BACKEND_CANN,
    BACKEND_TRITON,
    BOUND_DISPLAY_LABELS,
    COLLECTIVE_FAMILIES,
    GIB_BYTES,
    KERNEL_STATUS_NEW_OP,
    MIB_BYTES,
    NS_PER_US,
    OVERLAP_EXPOSED_EPS_US,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    TRACK_KERNEL,
    US_PER_MS,
    ActionRoadmapRow,
    ExportBundle,
    LayerArchetypeSpec,
    OperatorExportRow,
    PendingKernelItem,
)

#: Achievable Ascend hardware roofline target attainment envelopes (vs Lean L*).
TARGET_ETA_HBM_STREAMING = 0.78
TARGET_ETA_MXFP4_GMM = 0.85
TARGET_ETA_SKINNY_GEMM_VEC = 0.78
TARGET_ETA_LINK_ONESHOT = 0.75
#: Residual exposed EP link share inside fused MegaMoE dispatch_ffn_combine.
MEGAMOE_LINK_RESIDUAL_SHARE = 0.65
#: Fallback SKU kernel launch minimums (us) when ``kernel_invocations`` is absent.
DEFAULT_LAUNCH_VEC_US = 1.2
DEFAULT_LAUNCH_CUBE_US = 1.5
DEFAULT_LAUNCH_NORM_US = 1.4
#: Superpod CCU/AIV one-shot collective startup floors (us) by interconnect class.
UB_ONESHOT_FLOOR_US = 3.5
HCCS_ONESHOT_FLOOR_US = 4.5
ROCE_COLLECTIVE_FLOOR_US = 12.0
#: Small/medium superpod collective payload threshold (8 MiB) for SP / one-shot notes.
UB_ONESHOT_SMALL_MSG_BYTES = 8 * MIB_BYTES
#: Minimum operator saving (us) required to tag Pillar 2/3 optimization.
TARGET_OPT_MIN_GAIN_US = 0.25
#: Minimum per-op improvement ratio (3%) to emit a tuning badge.
TARGET_OPT_MIN_RATIO = 0.03
#: Minimum TP degree and HBM headroom (GiB) to emit Attention Local-TP co-design note.
CODESIGN_MIN_TP = 4
CODESIGN_MIN_HEADROOM_GIB = 4.0
#: Maximum character length for compact ``masked_by`` labels.
MAX_MASKED_BY_LEN = 48
#: Causal timestamp epsilon (us) for detecting original same-stream overlap.
ORIG_OVERLAP_EPS_US = 1e-3
#: Minimum display saving (ms) for roadmap line items.
ROADMAP_MIN_SAVE_MS = 0.05
#: Maximum roadmap rows for Framework and Kernel summaries in Sheet 1 Block 4.
MAX_ROADMAP_FW_ITEMS = 3
MAX_ROADMAP_KERNEL_ITEMS = 5
#: Prefix length for deduplicating framework action entries.
MAX_FW_DEDUP_KEY_LEN = 40
#: Lower safety floor for target eta division.
MIN_TARGET_ETA = 0.1

_GDN_PROLOGUE_SUFFIXES = frozenset({
    "beta_contiguous", "alpha_contiguous", "core_output_zero",
    "convolution_input_contiguous", "short_convolution",
    "q_flat", "k_flat", "v_flat", "qkv_concatenate",
    "delta_gates", "decode_q_l2norm", "decode_k_l2norm",
})
_GDN_PROLOGUE_COPY_SUFFIXES = frozenset({
    "beta_contiguous", "alpha_contiguous", "core_output_zero",
    "convolution_input_contiguous", "q_flat", "k_flat", "v_flat",
    "qkv_concatenate",
})
_GDN_EPILOGUE_SUFFIXES = frozenset({
    "core_output_copy", "z_norm_materialize",
    "linear_output_norm_gate", "linear_output.quantize",
})
_MIN_GDN_PROLOGUE_COPY_COUNT = 3
_MIN_EPILOGUE_FUSION_OPS = 2


@dataclass(frozen=True, slots=True)
class SkuTargetFloors:
    """Calibrated L0 launch and superpod collective floors extracted from active SKU."""

    launch_vec_us: float
    launch_cube_us: float
    launch_norm_us: float
    oneshot_floor_us: float
    mc2_sync_us: float
    is_intra_pod: bool
    is_a5_ub: bool
    interconnect_label: str


def extract_sku_floors(case: Case) -> SkuTargetFloors:
    """Read SKU L0 kernel completion minimums and superpod link capability from ``Case``."""
    die = case.machine.dies[0]
    inv = dict(getattr(die, "kernel_invocations", ()) or ())

    def _inv_us(key: str, default: float) -> float:
        entry = inv.get(key)
        if isinstance(entry, Mapping) and entry.get("minimum_ms") is not None:
            val = float(entry["minimum_ms"]) * US_PER_MS
            if val > 0.0:
                return val
        launch = float(getattr(die, "launch_us", 0.0) or 0.0)
        return launch if launch > 0.0 else default

    launch_vec = _inv_us("vector", DEFAULT_LAUNCH_VEC_US)
    launch_cube = _inv_us("cube", DEFAULT_LAUNCH_CUBE_US)
    launch_norm = _inv_us("norm", max(launch_vec, DEFAULT_LAUNCH_NORM_US))

    sku_low = str(case.machine.sku or case.machine.name or "").lower()
    topo_opts = dict(getattr(case.machine, "topology_options", ()) or ())
    preset_low = str(topo_opts.get("topology_preset") or "").lower()
    is_a5 = any(tok in sku_low for tok in ("950", "a5", "3510", "arch35")) or "a5" in preset_low
    is_a3 = any(tok in sku_low for tok in ("910c", "910_93", "9362", "9381", "9382", "9391", "9392", "a3")) or "a3" in preset_low
    tp = max(1, int(case.runtime.parallel.tp))
    rpn = max(1, int(getattr(case.machine, "ranks_per_node", tp) or tp))
    is_pod = any(tok in preset_low for tok in ("pod", "16p", "8p")) or is_a5 or is_a3 or rpn >= tp

    if is_a5 and is_pod:
        oneshot = UB_ONESHOT_FLOOR_US
        label = "A5 UB-Direct (CCU / AIV One-Shot)"
    elif is_pod:
        oneshot = HCCS_ONESHOT_FLOOR_US
        label = "A3/A2 HCCS (AIV One-Shot)"
    else:
        oneshot = ROCE_COLLECTIVE_FLOOR_US
        label = "Inter-Node RoCE / HCCL"
    return SkuTargetFloors(
        launch_vec_us=launch_vec,
        launch_cube_us=launch_cube,
        launch_norm_us=launch_norm,
        oneshot_floor_us=oneshot,
        mc2_sync_us=oneshot,
        is_intra_pod=is_pod,
        is_a5_ub=is_a5 and is_pod,
        interconnect_label=label,
    )


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((max(0.0, a), max(a, b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        p_lo, p_hi = merged[-1]
        if lo <= p_hi:
            merged[-1] = (p_lo, max(p_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _exposed_length_outside(
    interval: tuple[float, float], cover: Sequence[tuple[float, float]],
) -> float:
    lo, hi = interval
    if hi <= lo:
        return 0.0
    cur = lo
    exposed = 0.0
    for c_lo, c_hi in cover:
        if c_hi <= cur:
            continue
        if c_lo >= hi:
            break
        if c_lo > cur:
            exposed += min(hi, c_lo) - cur
        cur = max(cur, c_hi)
        if cur >= hi:
            return 0.0
    return exposed + max(0.0, hi - cur)


def _short_primary_label(op: OperatorExportRow) -> str:
    suf = strip_op_prefix(op.op_id).lower()
    if "unpack" in suf or "megamoe" in op.recipe.lower() or "expert_up" in suf or "expert_down" in suf:
        return "MoE Routed GMM / MegaMoE"
    if "candidate_gather" in suf:
        return "DSA Candidate AllGather"
    if suf.startswith("index_") or "indexer" in op.family.lower():
        return "DSA Lightning Indexer"
    if "linear_qkvz" in suf:
        return "GDN QKVZ Projection"
    role = op.logical_role or op.ascend_kernel or suf
    return role[:MAX_MASKED_BY_LEN]


def _find_dominant_masker(
    interval: tuple[float, float],
    claimed_labeled: Sequence[tuple[float, float, str]],
) -> str:
    lo, hi = interval
    overlap_by_lbl: dict[str, float] = defaultdict(float)
    for c_lo, c_hi, lbl in claimed_labeled:
        ov = min(hi, c_hi) - max(lo, c_lo)
        if ov > OVERLAP_EXPOSED_EPS_US:
            overlap_by_lbl[lbl] += ov
    if not overlap_by_lbl:
        return "主计算流并发算子"
    ordered = sorted(overlap_by_lbl.items(), key=lambda kv: -kv[1])
    top_lbl, top_ov = ordered[0]
    if len(ordered) > 1 and ordered[1][1] > top_ov * 0.25:
        combined = f"{top_lbl} + {ordered[1][0]}"
        return combined[:MAX_MASKED_BY_LEN]
    return top_lbl


def _overlap_priority_key(op: OperatorExportRow) -> tuple[Any, ...]:
    if op.stream_index == 1:
        return (2, op.start_us, op.ordinal)
    if op.stream_index == 2:
        return (1, -op.end_us, -op.dur_us, op.ordinal)
    return (0, -(op.hbm_total_bytes + op.flops), -op.end_us, -op.dur_us, op.ordinal)


def attribute_overlap_and_archetype_spans(
    rows: Sequence[OperatorExportRow],
    archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], tuple[LayerArchetypeSpec, ...]]:
    """Attribute cross-stream/branch overlap with 100% exact step-span conservation."""
    by_arch: dict[str, list[tuple[int, OperatorExportRow]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_arch[row.archetype_id].append((idx, row))

    updated: list[OperatorExportRow] = list(rows)
    span_by_arch: dict[str, float] = {}
    for aid, indexed_ops in by_arch.items():
        all_ivs = [(r.start_us, r.end_us) for _, r in indexed_ops if r.dur_us > 0.0]
        span_by_arch[aid] = sum(hi - lo for lo, hi in _merge_intervals(all_ivs))
        ordered = sorted(indexed_ops, key=lambda item: _overlap_priority_key(item[1]))
        claimed: list[tuple[float, float]] = []
        claimed_labeled: list[tuple[float, float, str]] = []
        for idx, row in ordered:
            if row.dur_us <= 0.0:
                updated[idx] = replace(row, is_overlapped=False, exposed_dur_us=0.0, masked_by="")
                continue
            iv = (row.start_us, row.end_us)
            exp_us = _exposed_length_outside(iv, claimed)
            masked_us = max(0.0, row.dur_us - exp_us)
            fully_hidden = exp_us <= OVERLAP_EXPOSED_EPS_US
            eff_exp = 0.0 if fully_hidden else exp_us
            masker = _find_dominant_masker(iv, claimed_labeled) if masked_us > OVERLAP_EXPOSED_EPS_US else ""
            eff_gap = 0.0 if fully_hidden else max(0.0, eff_exp - row.l_star_lean_us)
            eff_ratio = None if fully_hidden else (
                (eff_exp / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None
            )
            b_disp = (
                f"Overlapped by {masker} (0 exposed)"
                if fully_hidden else row.bound_display
            )
            updated[idx] = replace(
                row,
                is_overlapped=fully_hidden,
                exposed_dur_us=eff_exp,
                masked_by=masker,
                gap_us=eff_gap,
                gap_ratio=eff_ratio,
                bound_display=b_disp,
            )
            claimed = _merge_intervals([*claimed, iv])
            claimed_labeled.append((iv[0], iv[1], _short_primary_label(row)))

    arch_updated = tuple(
        replace(arch, single_span_us=span_by_arch.get(arch.archetype_id, 0.0))
        for arch in archetypes
    )
    return updated, arch_updated


def _collapse_op_group(
    group_rows: Sequence[OperatorExportRow],
    group_demands: Sequence[tuple[int, int, int, int, int]],
    *,
    new_suffix: str,
    logical_role: str,
    ascend_kernel: str,
    kernel_backend: str,
    kernel_status: str,
    new_kernel_spec: str,
    useful_hbm_bytes: float,
    peaks: Mapping[str, Any],
    t_launch_us: float,
    eta_target: float,
    needs_new_kernel: bool = True,
) -> tuple[OperatorExportRow, tuple[int, int, int, int, int]]:
    anchor = group_rows[0]
    prefix = anchor.op_id[: len(anchor.op_id) - len(strip_op_prefix(anchor.op_id))]
    new_op_id = f"{prefix}{new_suffix}"
    absorbed = tuple(dict.fromkeys(
        s for r in group_rows
        for s in (r.absorbed_source_ops or (strip_op_prefix(r.op_id),))
    ))
    tot_flops = sum(r.flops for r in group_rows)
    tot_vec = sum(r.vector_ops for r in group_rows)
    tot_link = sum(r.link_bytes for r in group_rows)
    hbm_tot = max(0.0, useful_hbm_bytes)
    hbm_r = hbm_tot * 0.5
    hbm_w = hbm_tot * 0.5
    hbm_peak = float(peaks.get("hbm_peak_bytes_per_s") or 1.0)
    hbm_r_rate = max(float(peaks.get("hbm_read_bytes_per_s") or hbm_peak), 1.0)
    hbm_w_rate = max(float(peaks.get("hbm_write_bytes_per_s") or hbm_peak), 1.0)
    alpha = float(peaks.get("hbm_rw_alpha", 1.0) if "hbm_rw_alpha" in peaks else 1.0)
    hbm_r_ns = int((hbm_r + alpha * hbm_w) / hbm_r_rate * 1e9) if hbm_tot > 0.0 else 0
    hbm_w_ns = int((hbm_w + alpha * hbm_r) / hbm_w_rate * 1e9) if hbm_tot > 0.0 else 0
    cube_ns = sum(d[0] for d in group_demands)
    vec_ns = sum(d[1] for d in group_demands)
    link_ns = sum(d[4] for d in group_demands)
    demand = (cube_ns, vec_ns, hbm_r_ns, hbm_w_ns, link_ns)
    l_star_us = float(max(demand)) / NS_PER_US
    before_sum_us = sum(r.dur_us for r in group_rows)
    dur_us = min(before_sum_us, t_launch_us + (l_star_us / max(eta_target, MIN_TARGET_ETA)))
    hbm_ns = max(hbm_r_ns, hbm_w_ns)
    dom_bound = "memory_bound" if hbm_ns >= max(cube_ns, vec_ns, link_ns) and hbm_ns > 0 else (
        "cube_bound" if cube_ns >= max(vec_ns, link_ns) and cube_ns > 0 else "vector_bound"
    )
    group_occs = {r.occurrence for r in group_rows}
    ext_preds = tuple(dict.fromkeys(
        p for r in group_rows for p in r.predecessors if p not in group_occs
    ))
    new_occ = replace(anchor.occurrence, op_id=new_op_id)
    fused_row = replace(
        anchor,
        occurrence=new_occ,
        op_id=new_op_id,
        recipe=new_suffix,
        implementation=ascend_kernel,
        predecessors=ext_preds,
        start_us=min(r.start_us for r in group_rows),
        end_us=min(r.start_us for r in group_rows) + dur_us,
        dur_us=dur_us,
        exposed_dur_us=dur_us,
        flops=tot_flops,
        vector_ops=tot_vec,
        hbm_read_bytes=hbm_r,
        hbm_write_bytes=hbm_w,
        hbm_total_bytes=hbm_tot,
        mte2_bytes=hbm_r,
        mte3_bytes=hbm_w,
        fixpipe_bytes=0.0,
        link_bytes=tot_link,
        dominant_bound=dom_bound,
        bound_display=BOUND_DISPLAY_LABELS.get(dom_bound, dom_bound),
        l_star_lean_us=l_star_us,
        gap_us=max(0.0, dur_us - l_star_us),
        gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
        roofline_efficiency=min(1.0, l_star_us / dur_us) if dur_us > 0.0 and l_star_us > 0.0 else None,
        absorbed_source_ops=absorbed,
        logical_role=logical_role,
        ascend_kernel=ascend_kernel,
        kernel_backend=kernel_backend,
        kernel_status=kernel_status,
        needs_new_kernel=needs_new_kernel,
        new_kernel_spec=new_kernel_spec,
        canonical_op_key=new_suffix,
        opt_pillar="pillar1_fusion",
        opt_target_eta_pct=round(eta_target * 100.0, 1),
    )
    return fused_row, demand


def _rewire_preds_after_collapse(
    rows: list[OperatorExportRow], occ_remap: Mapping[Any, Any],
) -> list[OperatorExportRow]:
    if not occ_remap:
        return rows
    out: list[OperatorExportRow] = []
    for row in rows:
        new_preds = tuple(dict.fromkeys(
            occ_remap.get(p, p) for p in row.predecessors if occ_remap.get(p, p) != row.occurrence
        ))
        out.append(replace(row, predecessors=new_preds))
    return out


def _fuse_gdn_prologue_in_arch(
    a_rows: list[OperatorExportRow],
    a_dems: list[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[Any, Any]]:
    idxs = [i for i, r in enumerate(a_rows) if strip_op_prefix(r.op_id).lower() in _GDN_PROLOGUE_SUFFIXES]
    has_gate = any(strip_op_prefix(a_rows[i].op_id).lower() == "delta_gates" for i in idxs)
    copy_cnt = sum(1 for i in idxs if strip_op_prefix(a_rows[i].op_id).lower() in _GDN_PROLOGUE_COPY_SUFFIXES)
    if len(idxs) < _MIN_EPILOGUE_FUSION_OPS or (not has_gate and copy_cnt < _MIN_GDN_PROLOGUE_COPY_COUNT):
        return a_rows, a_dems, {}

    grp_rows = [a_rows[i] for i in idxs]
    grp_dems = [a_dems[i] for i in idxs]
    compute_hbm = sum(
        r.hbm_total_bytes for r in grp_rows
        if strip_op_prefix(r.op_id).lower() not in _GDN_PROLOGUE_COPY_SUFFIXES
    )
    max_qkv_copy = max((r.hbm_total_bytes for r in grp_rows if r.hbm_total_bytes > 0.0), default=0.0)
    useful_hbm = compute_hbm if compute_hbm > 0.0 else (max_qkv_copy * 2.0)
    fused, dem = _collapse_op_group(
        grp_rows, grp_dems,
        new_suffix="fused_gdn_prologue",
        logical_role="Fused GDN Prologue (Rolling Conv1d + Gating + QK-L2Norm + Strided Views)",
        ascend_kernel="FusedGDNPrologue",
        kernel_backend=BACKEND_TRITON,
        kernel_status=f"{KERNEL_STATUS_NEW_OP} [待新开发融合算子] GDN Prologue 融合 + 零拷贝 Strided View",
        new_kernel_spec=(
            "Single-pass UB-resident Triton-Ascend/AscendC kernel fusing CausalConv1d "
            "speculative rolling state update + GDN exp/softplus decay gating + Q/K L2Norm "
            "while eliminating intermediate TensorMove/Fill copies via strided views"
        ),
        useful_hbm_bytes=useful_hbm,
        peaks=peaks,
        t_launch_us=floors.launch_vec_us,
        eta_target=TARGET_ETA_HBM_STREAMING,
    )
    first_idx = idxs[0]
    idx_set = set(idxs)
    remap = {a_rows[i].occurrence: fused.occurrence for i in idxs}
    out_r: list[OperatorExportRow] = []
    out_d: list[tuple[int, int, int, int, int]] = []
    for i, (r, d) in enumerate(zip(a_rows, a_dems)):
        if i == first_idx:
            out_r.append(fused)
            out_d.append(dem)
        elif i not in idx_set:
            out_r.append(r)
            out_d.append(d)
    return _rewire_preds_after_collapse(out_r, remap), out_d, remap


def _fuse_gdn_epilogue_in_arch(
    a_rows: list[OperatorExportRow],
    a_dems: list[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[Any, Any]]:
    idxs = [i for i, r in enumerate(a_rows) if strip_op_prefix(r.op_id).lower() in _GDN_EPILOGUE_SUFFIXES]
    has_norm = any(strip_op_prefix(a_rows[i].op_id).lower() == "linear_output_norm_gate" for i in idxs)
    if not has_norm or len(idxs) < _MIN_EPILOGUE_FUSION_OPS:
        return a_rows, a_dems, {}

    grp_rows = [a_rows[i] for i in idxs]
    grp_dems = [a_dems[i] for i in idxs]
    has_quant = any(strip_op_prefix(r.op_id).lower() == "linear_output.quantize" for r in grp_rows)
    norm_row = next(r for r in grp_rows if strip_op_prefix(r.op_id).lower() == "linear_output_norm_gate")
    useful_hbm = norm_row.hbm_total_bytes
    kern = "FusedRMSNormGatedQuant" if has_quant else "FusedRMSNormGated"
    fused, dem = _collapse_op_group(
        grp_rows, grp_dems,
        new_suffix="fused_linear_output_epilogue",
        logical_role="Fused Linear-Attn Epilogue (Strided View + RMSNormGated + MXQuant)" if has_quant else "Fused Linear-Attn Epilogue (Strided View + RMSNormGated)",
        ascend_kernel=kern,
        kernel_backend=BACKEND_TRITON,
        kernel_status=f"{KERNEL_STATUS_NEW_OP} [待新开发融合算子] {kern} 消除回写拷贝",
        new_kernel_spec=(
            f"UB-resident {kern} consuming recurrent output & Z gate via strided views "
            "and emitting quantized FP8/E8M0 activations directly to O-Proj"
        ),
        useful_hbm_bytes=useful_hbm,
        peaks=peaks,
        t_launch_us=floors.launch_vec_us,
        eta_target=TARGET_ETA_HBM_STREAMING,
    )
    first_idx = idxs[0]
    idx_set = set(idxs)
    remap = {a_rows[i].occurrence: fused.occurrence for i in idxs}
    out_r: list[OperatorExportRow] = []
    out_d: list[tuple[int, int, int, int, int]] = []
    for i, (r, d) in enumerate(zip(a_rows, a_dems)):
        if i == first_idx:
            out_r.append(fused)
            out_d.append(dem)
        elif i not in idx_set:
            out_r.append(r)
            out_d.append(d)
    return _rewire_preds_after_collapse(out_r, remap), out_d, remap


def _fuse_unfused_gmm_act_in_arch(
    a_rows: list[OperatorExportRow],
    a_dems: list[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[Any, Any]]:
    by_suf = {strip_op_prefix(r.op_id).lower(): i for i, r in enumerate(a_rows)}
    if "expert_up" not in by_suf or "expert_act" not in by_suf:
        return a_rows, a_dems, {}
    idxs = [by_suf["expert_up"], by_suf["expert_act"]]
    if "expert_down.quantize" in by_suf:
        idxs.append(by_suf["expert_down.quantize"])
    idxs.sort()
    up_row = a_rows[by_suf["expert_up"]]
    act_row = a_rows[by_suf["expert_act"]]
    is_situ = act_row.family == "situ" or "situ" in act_row.recipe.lower()
    kern = "GroupedMatmulSituQuant" if is_situ else "GroupedMatmulSwiGLUQuant"
    grp_rows = [a_rows[i] for i in idxs]
    grp_dems = [a_dems[i] for i in idxs]
    fused, dem = _collapse_op_group(
        grp_rows, grp_dems,
        new_suffix="expert_up",
        logical_role=f"Fused Routed Expert GMM1 + {'SiTU' if is_situ else 'SwiGLU'} + Quant",
        ascend_kernel=kern,
        kernel_backend=BACKEND_CANN,
        kernel_status=f"{KERNEL_STATUS_NEW_OP} [待新开发融合算子] {kern}",
        new_kernel_spec=f"Fused {kern} keeping intermediate gate/up activations in L0C/UB",
        useful_hbm_bytes=up_row.hbm_total_bytes,
        peaks=peaks,
        t_launch_us=floors.launch_cube_us,
        eta_target=TARGET_ETA_MXFP4_GMM,
    )
    first_idx = idxs[0]
    idx_set = set(idxs)
    remap = {a_rows[i].occurrence: fused.occurrence for i in idxs}
    out_r: list[OperatorExportRow] = []
    out_d: list[tuple[int, int, int, int, int]] = []
    for i, (r, d) in enumerate(zip(a_rows, a_dems)):
        if i == first_idx:
            out_r.append(fused)
            out_d.append(dem)
        elif i not in idx_set:
            out_r.append(r)
            out_d.append(d)
    return _rewire_preds_after_collapse(out_r, remap), out_d, remap


def _apply_pillar1_fusions(
    rows: Sequence[OperatorExportRow],
    demands: Sequence[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    peaks: Mapping[str, Any],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]:
    by_arch: dict[str, tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]] = {}
    arch_order: list[str] = []
    for row, dem in zip(rows, demands):
        aid = row.archetype_id
        if aid not in by_arch:
            by_arch[aid] = ([], [])
            arch_order.append(aid)
        by_arch[aid][0].append(row)
        by_arch[aid][1].append(dem)

    all_rows: list[OperatorExportRow] = []
    all_dems: list[tuple[int, int, int, int, int]] = []
    global_remap: dict[Any, Any] = {}
    for aid in arch_order:
        a_rows, a_dems = by_arch[aid]
        a_rows, a_dems, rm1 = _fuse_gdn_prologue_in_arch(a_rows, a_dems, floors, peaks)
        global_remap.update(rm1)
        a_rows, a_dems, rm2 = _fuse_gdn_epilogue_in_arch(a_rows, a_dems, floors, peaks)
        global_remap.update(rm2)
        a_rows, a_dems, rm3 = _fuse_unfused_gmm_act_in_arch(a_rows, a_dems, floors, peaks)
        global_remap.update(rm3)
        all_rows.extend(a_rows)
        all_dems.extend(a_dems)
    return _rewire_preds_after_collapse(all_rows, global_remap), all_dems


def _apply_pillar2_and_pillar3(
    rows: Sequence[OperatorExportRow],
    demands: Sequence[tuple[int, int, int, int, int]],
    floors: SkuTargetFloors,
    *,
    phase: str = "decode",
) -> list[OperatorExportRow]:
    is_decode = str(phase or "decode").lower() == "decode"
    out: list[OperatorExportRow] = []
    for row, dem in zip(rows, demands):
        if row.opt_pillar == "pillar1_fusion" or row.dur_us <= 0.0:
            out.append(row)
            continue
        if row.family in {"engram_rh2d", "h2d"} or "host_uva" in row.op_id.lower():
            out.append(row)
            continue

        suf = strip_op_prefix(row.op_id).lower()
        fam = row.family.lower()
        rec = row.recipe.lower()
        kern_l = row.ascend_kernel.lower()
        l_comp = float(max(dem[0], dem[1], dem[2], dem[3])) / NS_PER_US
        l_link = float(dem[4]) / NS_PER_US

        is_mc2 = fam in {"all_gather_matmul", "matmul_reduce_scatter"}
        is_megamoe = "megamoe" in rec or "dispatch_ffn_combine" in kern_l
        is_pure_comm = (row.task_type == "HCCL" or fam in COLLECTIVE_FAMILIES) and not is_mc2 and not is_megamoe

        if is_pure_comm:
            if not floors.is_intra_pod:
                out.append(row)
                continue
            t_target = floors.oneshot_floor_us + l_link / TARGET_ETA_LINK_ONESHOT
            t_after = min(row.dur_us, t_target)
            gain = row.dur_us - t_after
            if gain >= TARGET_OPT_MIN_GAIN_US and gain >= row.dur_us * TARGET_OPT_MIN_RATIO:
                tag = "CCU/AIV UB-Direct One-Shot" if floors.is_a5_ub else "AIV HCCS One-Shot"
                is_small_sp = ("sp_gather" in suf or "sp_output" in suf) and row.link_bytes <= UB_ONESHOT_SMALL_MSG_BYTES
                status = (
                    f"[通信方案优化: 切 {tag} / 小Batch Decode 可关 SP]"
                    if (is_small_sp and is_decode) else f"[通信方案优化: 切 {tag}]"
                )
                base_k = row.ascend_kernel.split(" (", 1)[0]
                out.append(replace(
                    row,
                    dur_us=t_after,
                    exposed_dur_us=t_after,
                    ascend_kernel=f"{base_k} ({tag})",
                    kernel_status=status,
                    gap_us=max(0.0, t_after - row.l_star_lean_us),
                    gap_ratio=(t_after / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None,
                    roofline_efficiency=min(1.0, row.l_star_lean_us / t_after) if t_after > 0.0 and row.l_star_lean_us > 0.0 else None,
                    opt_pillar="pillar3_comm",
                    opt_target_eta_pct=round(TARGET_ETA_LINK_ONESHOT * 100.0, 1),
                ))
            else:
                out.append(row)
            continue

        t_launch = (
            floors.launch_cube_us if row.flops > 0.0
            else (floors.launch_norm_us if fam in {"rmsnorm", "layernorm", "add_rmsnorm", "attn_res_prefix"} else floors.launch_vec_us)
        )
        mc2_kernel = row.ascend_kernel
        if is_mc2:
            eta = TARGET_ETA_SKINNY_GEMM_VEC
            t_target = max(
                t_launch + l_comp / eta,
                floors.oneshot_floor_us + l_link / TARGET_ETA_LINK_ONESHOT,
            ) + floors.mc2_sync_us
            pillar = "pillar3_comm"
            if fam == "all_gather_matmul" and "allgathermatmul" not in kern_l:
                mc2_kernel = "aclnnAllGatherMatmulV2"
            elif fam == "matmul_reduce_scatter" and "matmulreducescatter" not in kern_l:
                mc2_kernel = "aclnnMatmulReduceScatterV2"
            status_lbl = (
                f"[通信方案优化: MC2 切 {'CCU/AIV' if floors.is_a5_ub else 'AIV'} + 目标 {int(round(eta * 100))}% L*]"
            )
        elif is_megamoe:
            eta = TARGET_ETA_MXFP4_GMM
            t_target = (
                t_launch + floors.oneshot_floor_us
                + l_comp / eta + (l_link / TARGET_ETA_LINK_ONESHOT) * MEGAMOE_LINK_RESIDUAL_SHARE
            )
            pillar = "pillar2_roofline"
            status_lbl = f"[需算子深度调优 -> 目标 {int(round(eta * 100))}% L*]"
        elif fam == "grouped_gemm":
            eta = TARGET_ETA_MXFP4_GMM
            t_target = t_launch + row.l_star_lean_us / eta
            pillar = "pillar2_roofline"
            status_lbl = f"[需算子深度调优 -> 目标 {int(round(eta * 100))}% L*]"
        elif row.dominant_bound == "memory_bound" or fam in {"indexer", "recurrent_scan", "recurrent_vector"}:
            eta = TARGET_ETA_HBM_STREAMING
            t_target = t_launch + row.l_star_lean_us / eta
            pillar = "pillar2_roofline"
            status_lbl = f"[需算子深度调优 -> 目标 {int(round(eta * 100))}% L*]"
        else:
            eta = TARGET_ETA_SKINNY_GEMM_VEC
            t_target = t_launch + row.l_star_lean_us / eta
            pillar = "pillar2_roofline"
            status_lbl = (
                "[待新开发融合算子] UB 驻留 Block-AttnRes 融合前缀"
                if fam == "attn_res_prefix"
                else f"[需算子深度调优 -> 目标 {int(round(eta * 100))}% L*]"
            )

        t_after = min(row.dur_us, t_target)
        gain = row.dur_us - t_after
        if gain >= TARGET_OPT_MIN_GAIN_US and gain >= row.dur_us * TARGET_OPT_MIN_RATIO:
            out.append(replace(
                row,
                dur_us=t_after,
                exposed_dur_us=t_after,
                ascend_kernel=mc2_kernel,
                kernel_status=status_lbl,
                gap_us=max(0.0, t_after - row.l_star_lean_us),
                gap_ratio=(t_after / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None,
                roofline_efficiency=min(1.0, row.l_star_lean_us / t_after) if t_after > 0.0 and row.l_star_lean_us > 0.0 else None,
                opt_pillar=pillar,
                opt_target_eta_pct=round(eta * 100.0, 1),
            ))
        else:
            out.append(row)
    return out


def _reschedule_archetype_timelines(rows: Sequence[OperatorExportRow]) -> list[OperatorExportRow]:
    """Recompute exact causal ``(start_us, end_us)`` preserving parallel branches."""
    by_arch: dict[str, list[tuple[int, OperatorExportRow]]] = defaultdict(list)
    arch_order: list[str] = []
    for idx, row in enumerate(rows):
        if row.archetype_id not in by_arch:
            arch_order.append(row.archetype_id)
        by_arch[row.archetype_id].append((idx, row))

    updated = list(rows)
    cursor_us = 0.0
    for aid in arch_order:
        indexed_ops = by_arch[aid]
        ordered = sorted(indexed_ops, key=lambda item: (item[1].start_us, item[1].ordinal))
        end_by_occ: dict[Any, float] = {}
        stream_history: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
        arch_origin = cursor_us
        arch_max_end = arch_origin
        for idx, row in ordered:
            pred_ready = max((end_by_occ.get(p, arch_origin) for p in row.predecessors), default=arch_origin)
            s_key = (row.stream, row.stream_index)
            stream_ready = arch_origin
            for orig_s, orig_e, new_e in stream_history[s_key]:
                if row.start_us >= orig_e - ORIG_OVERLAP_EPS_US and orig_e > orig_s:
                    stream_ready = max(stream_ready, new_e)
            new_s = max(pred_ready, stream_ready)
            new_e = new_s + row.dur_us
            end_by_occ[row.occurrence] = new_e
            arch_max_end = max(arch_max_end, new_e)
            stream_history[s_key].append((row.start_us, row.end_us, new_e))
            updated[idx] = replace(row, start_us=new_s, end_us=new_e)
        cursor_us = arch_max_end
    return updated


def build_op_action_guidance(
    op: OperatorExportRow,
    *,
    is_a5_ub: bool,
    phase: str = "decode",
    hardware_sku: str = "",
) -> tuple[str, str, str, str]:
    """Return concrete 3-track ``(config_howto, kernel_action, framework_action, badge)``."""
    suf = strip_op_prefix(op.op_id).lower()
    fam = op.family.lower()
    kern_l = op.ascend_kernel.lower()
    is_decode = phase.lower() == "decode"
    sku_low = hardware_sku.lower()
    is_a2 = (not is_a5_ub) and any(tok in sku_low for tok in ("910b", "a2"))
    nz_flag = '"enable_weight_nz_layout": true, ' if not is_a5_ub else ""
    kv_flag = "--kv-cache-dtype fp8_e4m3" if is_a5_ub else "--kv-cache-dtype int8"
    prefill_caveat = (
        " (注意: Prefill 双流 micro-batch 若出现 AIV 争抢，优先切 A5 CCU 硬件引擎)"
        if (not is_decode and is_a5_ub) else ""
    )

    if suf == "fused_gdn_prologue" or "fusedgdnprologue" in kern_l:
        return (
            "无现成开关（需新融合算子 + 框架接线）; 配合 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'",
            "[需新写算子] Triton-Ascend/AscendC 单 Kernel 融合 CausalConv1d(投机滚动状态) + GDN exp/softplus Gating + Q/K L2Norm，UB 驻留消除中间 HBM 回写",
            "[框架改造] vllm_ascend/ops/gdn.py 用 strided view 消除 beta/alpha/q/k/v/qkv_cat 共 8 次 .contiguous() TensorMove；静态预分配 conv/state 缓冲保 ACLGraph 零图断裂",
            "[算子新写 + 框架零拷贝改造]",
        )
    if suf == "fused_linear_output_epilogue" or "fusedrmsnormgated" in kern_l:
        return (
            "无现成开关（需新融合算子 + 框架接线）; 配合 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'",
            "[需新写算子] Triton-Ascend/AscendC 实现 FusedRMSNormGatedQuant，UB 内融合 RMSNorm+SigmoidGate+DynamicMxQuant",
            "[框架改造] vllm_ascend/ops/gdn.py & models/kimi_k3/kda.py 移除 core_output_copy / z_norm_materialize 拷贝并注册 CustomOp 到 ACLGraph",
            "[算子新写 + 框架 Pass 接线]",
        )
    if fam == "mhc" or ".hc_" in f".{suf}":
        return (
            "开启 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'; 无现成 CANN mHC 开关，需新融合算子",
            "[需新写算子] AscendC/Triton 融合 FusedMHCPreNormProj + FusedMHCSinkhornPost，将 [B,4*D] 驻留 UB/L1 消除 5 次 HBM 往返（目标 >=78% L*）",
            "[框架改造] vllm_ascend/models/qwen4_exp/hyper_connection.py & ops/mhc.py 替换 6 步 Eager 链路为 2 个 CustomOp，消除 .item()/动态分配",
            "[算子新写 + 框架 Pass 替换]",
        )
    if fam == "attn_res_prefix" or "attn_res" in suf:
        return (
            "开启 --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' 消除小算子 Host 发射气泡",
            "[需新写算子] Triton-Ascend/AscendC 实现 BlockAttnResFusedPrefix，单次 UB 流水完成 Norm+Score+Softmax+加权残差聚合",
            "[框架改造] vllm_ascend/models/kimi_k3/attn_res.py 注册静态 block-history 指针表并接入 ACLGraph 捕获",
            "[算子新写 + 框架图捕获适配]",
        )
    if "mlaprolog" in kern_l or suf.startswith("mla_prolog"):
        return (
            "export VLLM_ASCEND_ENABLE_MLAPO=1 && --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'",
            "[需算子调优] 优化 aclnnMlaPrologV3 在 96->128 head pad 与 MXFP8/BF16 下 L1 权重驻留与 Cube/AIV 流水配比（目标 >=78% L*）",
            "[框架改造] vllm_ascend/attention/mla_v1.py 打开全量化模式 MLAPO 路由并预排布 NZ 权重",
            "[配置开启 MLAPO + 算子调优]",
        )
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        ccu_txt = "硬件 CCU 调度 (CCU_SCHED/CCU_MS) + " if is_a5_ub else ""
        return (
            f"export HCCL_OP_EXPANSION_MODE=\"AIV\" VLLM_ASCEND_ENABLE_FUSED_MC2=1 && "
            f"--additional-config '{{\"enable_fused_mc2\": 2, \"enable_flashcomm1\": true}}'{prefill_caveat}",
            f"[需算子调优] MC2 通算融合切 {ccu_txt}AIV 展开并优化小 Batch M-tiling 掩盖通信延迟（目标 >=78% L*）",
            "[框架改造] vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py 确保线性层与 SP Gather/ReduceScatter 融合且 Decode 小 Batch 可切换关闭冗余 SP",
            "[配置开 MC2/CCU-AIV + 框架通算融合]",
        )
    if op.task_type == "HCCL" or fam in COLLECTIVE_FAMILIES:
        if is_a5_ub:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512 + A5 Arch35 硬件 CCU 调度 (CCU_SCHED/CCU_MS UB-Direct)'
            comm_kern = "[需算子调优] 启用超节点 <=8MiB 小包 AIV UB-Direct / CCU One-Shot 集合通信模板，消除 ~80-95us Host/AI_CPU 启动开销（目标 >=75% Link L*）"
        elif is_a2:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_BUFFSIZE=512 (A2 机内 HCCS / 跨机 RoCE：避免 TP 跨机、优先机内 TP<=8 + 跨机 DP/EP)'
            comm_kern = "[需算子调优] A2 机内切 AIV HCCS One-Shot 降小包启动延迟，跨机 RoCE 优先用 DP/EP 替代跨机 TP AllReduce"
        else:
            comm_env = 'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512 (A3 双 Die 超节点 HCCS)'
            comm_kern = "[需算子调优] A3 双 Die 超节点启用 <=8MiB 小包 AIV HCCS One-Shot 集合通信模板，消除 ~80-95us Host/AI_CPU 启动开销（目标 >=75% Link L*）"
        sp_tip = "；小 Batch Decode 可设 additional_config.enable_sp=false 消除负收益 SP" if "sp_" in suf else ""
        return (
            f"{comm_env}{sp_tip}{prefill_caveat}",
            comm_kern,
            "[框架改造] vllm_ascend/attention/context_parallel/mla_cp.py 将 DCP Query / DSA Candidate AllGather 下发至独立通信流并用 npu.Event 与 Indexer 异步重叠",
            "[通信切 CCU/AIV + 框架异步掩盖]" if is_a5_ub else "[通信切 AIV HCCS + 框架异步掩盖]",
        )
    if "lightning" in kern_l or fam == "indexer":
        return (
            f"export VLLM_ASCEND_ENABLE_MLAPO=1 && {kv_flag} (开启量化 KV/Indexer K-Cache 降访存)",
            "[需算子调优] 优化 aclnnQuantLightningIndexer 向量化 Index-K 预取与 UB 双缓冲（目标 >=78% HBM L*）",
            "[框架改造] vllm_ascend/attention/sfa_v1.py 保持静态 TopK-2048 缓冲并异步流水编排 Indexer 与 DCP AllGather",
            "[算子调优 + 框架流水编排]",
        )
    if fam == "grouped_gemm" or "megamoe" in op.recipe.lower() or "dispatch_ffn_combine" in kern_l:
        quant_gmm = (
            "MXFP4 group=32 GroupedMatmul / MegaMoE 紧凑动态 M-tiling + SituQuant 融合"
            if is_a5_ub else
            "W4A8 INT4 FRACTAL_NZ aclnnGroupedMatmulV4 / A3 MegaMoE + aclnnSwiGluQuant 紧凑动态 M-tiling"
        )
        return (
            f'export VLLM_ASCEND_ENABLE_FUSED_MC2=1 HCCL_BUFFSIZE_EP=2048 && '
            f'--additional-config \'{{{nz_flag}"enable_fused_mc2": 3, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true}}\'',
            f"[需算子调优] {quant_gmm}，消除稀疏专家 M-padding 浪费（目标 >=85% L*）",
            "[框架改造] vllm_ascend/ops/fused_moe/moe_comm_method.py 接线 MegaMoE + MSO Stream 1 共享专家异步掩盖",
            "[配置开 MegaMoE/MSO + 算子 Tiling 调优]",
        )
    if "recover_ssm" in suf or "recurrent" in suf or "kda" in kern_l:
        return (
            "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' (保证投机校验循环无 Host 同步气泡)",
            "[需算子调优] 优化 Persistent Recurrent KDA/GDN 核内状态驻留与按接收槽位增量写回（目标 >=78% HBM L*）",
            "[框架改造] vllm_ascend/worker/model_runner_v1.py 投机接收路径原地更新 CausalConv1d & SSM 状态槽，避免整批回滚拷贝",
            "[算子调优 + 框架原地状态提交]",
        )
    if op.is_overlapped:
        return (
            "--additional-config '{\"multistream_overlap_shared_expert\": true, \"enable_shared_expert_dp\": true}'",
            "[现成可用] 副流算子已被主计算流 100% 掩盖，无需额外算子改动",
            "[框架配置] 维持独立副流 npu.Event 同步，避免插入 host synchronize 破坏掩盖",
            "[已有特效: 副流 100% 掩盖]",
        )
    return (
        "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}' (配合 msmodelslim 生产量化)",
        f"[需算子调优] 针对 {op.bound_display or op.dominant_bound} 优化分块与双缓冲（目标 >=78% L*）",
        "[框架改造] 保持静态 shape 与预分配 workspace，避免 ACLGraph 图断裂与冗余 TensorMove",
        "[算子调优 + ACLGraph 保图]",
    )


def build_parallelism_codesign_note(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
) -> str:
    """Emit concise Parallelism Co-Design callout when high TP causes avoidable collectives."""
    mem = after_bundle.memory_ledger
    tp = int(mem.tp)
    if tp < CODESIGN_MIN_TP:
        return ""
    headroom_gib = min((st.headroom_bytes for st in mem.stage_budgets), default=0.0) / float(GIB_BYTES)
    comm_ms = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in before_bundle.operators
        if (op.task_type == "HCCL" or op.family in COLLECTIVE_FAMILIES)
        and "dispatch" not in op.op_id.lower() and "combine" not in op.op_id.lower()
    )
    after_comm_ms = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in after_bundle.operators
        if (op.task_type == "HCCL" or op.family in COLLECTIVE_FAMILIES)
        and "dispatch" not in op.op_id.lower() and "combine" not in op.op_id.lower()
    )
    if comm_ms < ROADMAP_MIN_SAVE_MS:
        return ""
    fit_hint = (
        f"当前单卡 HBM 余量 {headroom_gib:.1f} GiB 充裕，建议评估 Attention Local-TP=1/2 (Attn-DP, o_proj_tp=1/2) "
        "或小 Batch Decode 关闭 SP，可直接从架构侧消除 TP/SP 集合通信与 AllGather 启动开销"
        if headroom_gib >= CODESIGN_MIN_HEADROOM_GIB else
        f"当前单卡 HBM 余量 {headroom_gib:.1f} GiB，优先采用 QuaRot MXFP8+MXFP4 压缩权重释放 HBM 后再评估 Attn-DP (Local-TP=2) 减通信"
    )
    return (
        f"[并行与通信协同建议] TP={tp} 导致非 EP 暴露通信 Before={comm_ms:.2f}ms -> After(CCU/AIV+MC2)={after_comm_ms:.2f}ms；"
        f"{fit_hint}。"
    )


def build_actionable_roadmap(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    pending_items: Sequence[PendingKernelItem],
) -> tuple[ActionRoadmapRow, ...]:
    """Synthesize concrete 3-track Sheet 1 Block 4 roadmap directly from Before->After delta."""
    is_a5 = any(tok in after_bundle.hardware_sku.lower() for tok in ("950", "a5", "3510"))
    mem = after_bundle.memory_ledger
    rows: list[ActionRoadmapRow] = []

    comm_before = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in before_bundle.operators
        if op.task_type == "HCCL" or op.family in {"all_gather_matmul", "matmul_reduce_scatter"}
    )
    comm_after = sum(
        op.exposed_dur_us * op.repeat_count / US_PER_MS
        for op in after_bundle.operators
        if op.task_type == "HCCL" or op.family in {"all_gather_matmul", "matmul_reduce_scatter"}
    )
    comm_saved = max(0.0, comm_before - comm_after)
    if comm_saved >= ROADMAP_MIN_SAVE_MS or comm_before >= ROADMAP_MIN_SAVE_MS:
        ccu_env = (
            'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 '
            'HCCL_BUFFSIZE=512 (950DT Arch35 切硬件 CCU 调度 CCU_SCHED/CCU_MS UB-Direct)'
            if is_a5 else
            'export HCCL_OP_EXPANSION_MODE="AIV" HCCL_INTRA_PCIE_ENABLE=1 HCCL_INTRA_ROCE_ENABLE=0 HCCL_BUFFSIZE=512'
        )
        rows.append(ActionRoadmapRow(
            track=TRACK_CONFIG_ENV,
            track_label="[A] 已有特效与通信展开配置",
            title="集合通信展开切 AIV/CCU + MC2/FlashComm1 通算融合",
            expected_gain_ms=comm_saved,
            action_detail=(
                f"{ccu_env}; export VLLM_ASCEND_ENABLE_FUSED_MC2=1; "
                '--additional-config \'{"enable_fused_mc2": 2, "enable_flashcomm1": true}\''
            ),
            cross_ref="Sheet 2 通信/MC2 行 & Sheet 4",
        ))

    has_mla = any("mla" in op.submodule.lower() or "mlaprolog" in op.ascend_kernel.lower() for op in after_bundle.operators)
    has_mso = any(op.stream_index == 1 for op in after_bundle.operators)
    switch_parts = ['--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'']
    if has_mla:
        switch_parts.insert(0, "export VLLM_ASCEND_ENABLE_MLAPO=1")
    if has_mso:
        switch_parts.append('--additional-config \'{"multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true}\'')
    mso_hidden = sum((op.dur_us - op.exposed_dur_us) * op.repeat_count / US_PER_MS for op in after_bundle.operators if op.stream_index == 1)
    rows.append(ActionRoadmapRow(
        track=TRACK_CONFIG_ENV,
        track_label="[A] 已有特效与通信展开配置",
        title="开启 ACLGraph 全图捕获 + MLAPO / MSO 共享专家异步掩盖",
        expected_gain_ms=mso_hidden,
        action_detail="; ".join(switch_parts) + (f" (MSO 掩盖 {mso_hidden:.2f}ms 共享专家耗时)" if mso_hidden > 0.0 else ""),
        cross_ref="Sheet 1 Block 3 MSO 掩盖列",
    ))

    quant_tip = (
        f"当前权重={mem.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB/die (原生 BF16={mem.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB, "
        f"压缩 {mem.weight_compression_ratio:.2f}x); A5 生产建议 msmodelslim + QuaRot MXFP8+MXFP4 (--quantization ascend --kv-cache-dtype fp8_e4m3)"
        if is_a5 else
        f"当前权重={mem.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB/die (原生 BF16={mem.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB, "
        f"压缩 {mem.weight_compression_ratio:.2f}x); A2/A3 生产建议 msmodelslim W4A8_DYNAMIC(INT8 Linear+INT4 Expert)/W8A8_DYNAMIC + C8 KV "
        '--quantization ascend --kv-cache-dtype int8 --additional-config \'{"enable_weight_nz_layout": true}\''
    )
    rows.append(ActionRoadmapRow(
        track=TRACK_CONFIG_ENV,
        track_label="[A] 已有特效与通信展开配置",
        title="生产量化方案 (msmodelslim + KV Cache 量化)",
        expected_gain_ms=0.0,
        action_detail=quant_tip,
        cross_ref="Sheet 3 Memory_Quant_Calculator",
    ))

    codesign = build_parallelism_codesign_note(before_bundle, after_bundle)
    if codesign:
        rows.append(ActionRoadmapRow(
            track=TRACK_CONFIG_ENV,
            track_label="[A] 已有特效与通信展开配置",
            title="并行策略与通信协同建议 (Local-TP / SP 开关)",
            expected_gain_ms=0.0,
            action_detail=codesign,
            cross_ref="Sheet 1 Block 1 并行配置",
        ))

    fw_seen: set[str] = set()
    for item in pending_items:
        if not item.framework_action:
            continue
        fw_key = item.framework_action.split("；", 1)[0][:MAX_FW_DEDUP_KEY_LEN]
        if fw_key in fw_seen:
            continue
        fw_seen.add(fw_key)
        rows.append(ActionRoadmapRow(
            track=TRACK_FRAMEWORK,
            track_label="[B] 推理框架层改造项 (vllm-ascend)",
            title=f"接线 {item.target_kernel.split('[', 1)[0].strip()} ({item.submodule})",
            expected_gain_ms=item.expected_gain_ms,
            action_detail=item.framework_action,
            cross_ref=f"Sheet 4 Card #{item.priority}",
        ))
        if len(fw_seen) >= MAX_ROADMAP_FW_ITEMS:
            break

    for item in pending_items[:MAX_ROADMAP_KERNEL_ITEMS]:
        rows.append(ActionRoadmapRow(
            track=TRACK_KERNEL,
            track_label="[C] 算子新开发与深度调优项",
            title=f"{item.target_kernel} ({item.before_model_ms:.2f} -> {item.after_model_ms:.2f} ms)",
            expected_gain_ms=item.expected_gain_ms,
            action_detail=item.kernel_action or item.default_design_slot,
            cross_ref=f"Sheet 4 Card #{item.priority} ({item.card_id})",
        ))
    return tuple(rows)


def synthesize_target_envelope(
    rows: Sequence[OperatorExportRow],
    demands: Sequence[tuple[int, int, int, int, int]],
    archetypes: Sequence[LayerArchetypeSpec],
    case: Case,
    peaks_by_stage: Mapping[int, Mapping[str, Any]],
    *,
    phase: str = "decode",
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], tuple[LayerArchetypeSpec, ...]]:
    """Run all 3 optimization pillars, reschedule timelines, and attribute exact overlap."""
    floors = extract_sku_floors(case)
    p0 = peaks_by_stage.get(0) or next(iter(peaks_by_stage.values()), {})

    p1_rows, p1_dems = _apply_pillar1_fusions(rows, demands, floors, p0)
    p23_rows = _apply_pillar2_and_pillar3(p1_rows, p1_dems, floors, phase=phase)
    scheduled = _reschedule_archetype_timelines(p23_rows)
    attributed, arch_updated = attribute_overlap_and_archetype_spans(scheduled, archetypes)
    enriched: list[OperatorExportRow] = []
    sku_str = str(case.machine.sku or case.machine.name or "")
    for idx, op in enumerate(attributed):
        cfg, kern, fw, _ = build_op_action_guidance(
            op, is_a5_ub=floors.is_a5_ub, phase=phase, hardware_sku=sku_str,
        )
        enriched.append(replace(
            op, ordinal=idx, config_howto=cfg, kernel_action=kern, framework_action=fw,
        ))
    return enriched, p1_dems, arch_updated
