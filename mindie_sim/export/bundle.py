"""O(Unique Archetypes) export bundle evaluation and scenario composition.

[Role] Compose scenarios (``resolve_hardware_alias``, ``compose_export_case``),
  discover unique Target/Draft layer archetypes, and evaluate each archetype
  once on ``runtime.Engine`` with Lean-4 certified ``L*`` bounds, exact
  cross-stream overlap attribution, hardware efficiency (AIC MFU / AIV Util /
  HBM MBU / HCCL Link Util), 3-pillar ``expected_ideal`` target envelope, and
  Memory/Quant/PP ledgers into an immutable ``ExportBundle``.
[Invariants] Strictly O(Unique Archetypes) RAM on full multi-layer models while
  conserving full-model repeat counts and streaming replay schedules. Zero banned
  recipe class names, zero heuristic fudge factors, and exact parity between
  cached KPIs (Effective TPOT, Single-Die TPS, Cluster Total TPS) and live XLSX
  formulas.
[Timeouts & Locks] Single-pass bounded evaluation honoring
  ``ReplayOptions.deadline_mono``; no locks or device contexts.
[Side-Effect Policy] Read-only over caller cases and observation files.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
from pathlib import Path
import re
from typing import Any

from mindie_sim.catalog.npu_checkpoints import normalize_hw_family, resolve_default_checkpoint
from mindie_sim.catalog.va_speculative_rules import build_va_serving_rules
from mindie_sim.optimization.dual_bound import (
    NS_PER_SECOND,
    DagEdge,
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
    inward_floor_ns,
)
try:
    from mindie_sim.optimization.kernel_synthesis import (
        TargetOpNode,
        attribute_node_overlap,
        evaluate_macro_wall_trigger,
        extract_sku_floors,
        solve_target_graph,
    )
    _HAS_KERNEL_SYNTHESIS = True
except ImportError:  # Standalone pre-merge compatibility before optimization/kernel_synthesis lands.
    _HAS_KERNEL_SYNTHESIS = False

    @dataclass(frozen=True, slots=True)
    class _StandaloneFloors:
        is_a5_ub: bool
        is_intra_pod: bool

    @dataclass(frozen=True, slots=True)
    class _StandaloneWall:
        wall_kind: str
        tpot_ms: float
        tpot_slo_ms: float
        sla_slack_ms: float
        hbm_headroom_gib: float
        hbm_utilization_pct: float
        tradeoff_direction: str
        recommended_actions: tuple[str, ...]

        def to_dict(self) -> dict[str, Any]:
            return {
                "wall_kind": self.wall_kind,
                "tpot_ms": round(float(self.tpot_ms), 3),
                "tpot_slo_ms": round(float(self.tpot_slo_ms), 3),
                "sla_slack_ms": round(float(self.sla_slack_ms), 3),
                "hbm_headroom_gib": round(float(self.hbm_headroom_gib), 2),
                "hbm_utilization_pct": round(float(self.hbm_utilization_pct), 1),
                "tradeoff_direction": self.tradeoff_direction,
                "recommended_actions": list(self.recommended_actions),
            }

    TargetOpNode = Any  # type: ignore[misc,assignment]

    def extract_sku_floors(case: Case) -> Any:
        sku_low = str(case.machine.sku or case.machine.name or "").lower()
        topo_opts = dict(getattr(case.machine, "topology_options", ()) or ())
        preset_low = str(topo_opts.get("topology_preset") or "").lower()
        is_a5 = any(tok in sku_low for tok in ("950", "a5", "3510", "arch35")) or "a5" in preset_low
        is_pod = any(tok in preset_low for tok in ("pod", "16p", "8p")) or is_a5
        return _StandaloneFloors(is_a5_ub=(is_a5 and is_pod), is_intra_pod=is_pod)

    def evaluate_macro_wall_trigger(
        *,
        tpot_ms: float,
        tpot_slo_ms: float,
        hbm_used_bytes: float,
        hbm_usable_bytes: float,
        hbm_headroom_bytes: float,
    ) -> Any:
        slo = max(float(tpot_slo_ms), 1e-6)
        slack = slo - float(tpot_ms)
        util_pct = (100.0 * float(hbm_used_bytes) / float(hbm_usable_bytes)) if hbm_usable_bytes > 0.0 else 0.0
        headroom_gib = float(hbm_headroom_bytes) / float(1024 ** 3)
        if slack >= 3.0 and util_pct >= 88.0:
            return _StandaloneWall(
                "HBM_CAPACITY_WALL", tpot_ms, tpot_slo_ms, slack, headroom_gib, util_pct,
                "compute_for_memory_and_offload",
                (
                    "以算换存: 优先启用 MLA 矩阵吸收 / KV Cache 低比特量化 (C8/FP8) 释放单卡 HBM",
                    "内存卸载: 评估 KV Offloading / Host DRAM-SSD Prefix Cache 分级池化与 Engram Host Offload",
                ),
            )
        if float(tpot_ms) >= slo * 0.92 and headroom_gib >= 8.0:
            return _StandaloneWall(
                "LATENCY_COMPUTE_WALL", tpot_ms, tpot_slo_ms, slack, headroom_gib, util_pct,
                "memory_for_compute",
                (
                    "以存换算: 利用充裕 HBM 余量预展开高频投影权重或增大片上分块驻留",
                    "投机多槽快照: 分配 [K+1] 多槽状态缓冲并做 in-place 2D 槽位别名消除回滚拷贝",
                ),
            )
        return _StandaloneWall(
            "BALANCED", tpot_ms, tpot_slo_ms, slack, headroom_gib, util_pct, "balanced",
            ("当前延迟与 HBM 利用率处于均衡区间，优先落地已验证的片上融合与超节点通信展开方案",),
        )

    def _merge_ivs(ivs: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        ordered = sorted((max(0.0, a), max(a, b)) for a, b in ivs if b > a)
        if not ordered:
            return []
        out = [ordered[0]]
        for lo, hi in ordered[1:]:
            if lo <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], hi))
            else:
                out.append((lo, hi))
        return out

    def _exposed_outside(iv: tuple[float, float], cover: Sequence[tuple[float, float]]) -> float:
        lo, hi = iv
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

    def attribute_node_overlap(nodes: Sequence[Any]) -> tuple[list[Any], dict[str, float]]:
        by_arch: dict[str, list[tuple[int, Any]]] = defaultdict(list)
        for idx, node in enumerate(nodes):
            by_arch[node.archetype_id].append((idx, node))
        updated = list(nodes)
        spans: dict[str, float] = {}
        for aid, indexed in by_arch.items():
            spans[aid] = sum(hi - lo for lo, hi in _merge_ivs([(n.start_us, n.end_us) for _, n in indexed if n.dur_us > 0.0]))
            ordered = sorted(
                indexed,
                key=lambda item: (
                    2 if item[1].stream_index == 1 else (1 if item[1].stream_index == 2 else 0),
                    -(item[1].hbm_total_bytes + item[1].flops),
                    item[1].ordinal,
                ),
            )
            claimed: list[tuple[float, float]] = []
            claimed_lbls: list[tuple[float, float, str]] = []
            for idx, node in ordered:
                if node.dur_us <= 0.0:
                    updated[idx] = replace(node, is_overlapped=False, exposed_dur_us=0.0, masked_by="")
                    continue
                iv = (node.start_us, node.end_us)
                exp_us = _exposed_outside(iv, claimed)
                fully_hidden = exp_us <= 1e-3
                eff_exp = 0.0 if fully_hidden else exp_us
                masker = ""
                if node.dur_us - exp_us > 1e-3 and claimed_lbls:
                    best_lbl = max(
                        claimed_lbls,
                        key=lambda seg: max(0.0, min(iv[1], seg[1]) - max(iv[0], seg[0])),
                    )
                    masker = best_lbl[2]
                b_disp = f"Overlapped by {masker} (0 exposed)" if fully_hidden else node.bound_display
                updated[idx] = replace(
                    node, is_overlapped=fully_hidden, exposed_dur_us=eff_exp,
                    masked_by=masker, bound_display=b_disp,
                )
                claimed = _merge_ivs([*claimed, iv])
                claimed_lbls.append((iv[0], iv[1], (node.logical_role or node.ascend_kernel or node.op_id)[:48]))
        return updated, spans

    def solve_target_graph(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise ModelingError("kernel_synthesis_module_unavailable")
from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.efficiency import extract_hardware_peaks
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.results import (
    CaseInput,
    FusedPipelineDescriptor,
    OpOccurrence,
    ReplayOptions,
    StepEvaluation,
    StepRequest,
)
from mindie_sim.prediction.rewrite import ReplayRecipeBook
from mindie_sim.prediction.spine import default_spine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest

from .kernel_catalog import (
    classify_stream_index,
    classify_submodule,
    classify_task_type,
    extract_layer_index,
    extract_semantic_atoms,
    resolve_ascend_kernel_identity,
    resolve_comm_domain_tag,
    strip_op_prefix,
)
from .memory_ledger import compute_memory_quant_ledger, stage_layer_counts
from .models import (
    ARCHETYPE_ORDER_FALLBACK,
    BINDING_STATE_BOUND,
    BINDING_STATE_MEASURED,
    BINDING_STATE_UNBOUND_FORMULA,
    BOUND_DISPLAY_LABELS,
    CANONICAL_EXPORT_SUBMODULES,
    CERT_RESOURCE_COUNT,
    DEFAULT_EXPORT_CONTEXT_TOKENS,
    DEFAULT_GAP_RADAR_TOP_K,
    DEFAULT_MIN_BATCHED_TOKENS,
    DEFAULT_MIN_MODEL_LEN,
    DEFAULT_MODEL_LEN_HEADROOM,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    DEFAULT_TPOT_SLO_MS,
    DEFAULT_TTFT_SLO_MS,
    DEPLOYMENT_MODE_LABELS,
    EXPORT_BUNDLE_SCHEMA,
    GRAPH_DIGEST_HEX_LEN,
    HARDWARE_PRESET_ALIASES,
    KIB_BYTES,
    LAUNCH_BOUND_SHORT_US,
    LAUNCH_BOUND_WORK_SHARE,
    MAX_EXPLICIT_LAYER_LABELS,
    MIB_BYTES,
    MIN_DENOM_EPS,
    MIN_POSITIVE_SECONDS,
    MODE_MODELING,
    MODE_SIMULATION,
    NS_PER_US,
    OVERLAP_EXPOSED_EPS_US,
    TB_BYTES,
    US_PER_MS,
    US_PER_SECOND,
    ExportBundle,
    GapRadarRow,
    LayerArchetypeSpec,
    OperatorExportRow,
    ReconciledKernelGroup,
    ReconciliationSummary,
    TraceScheduleEntry,
)
from .reconcile import load_observation_bundle, reconcile_operators_with_kernels

_RECURRENT_MICRO_PATTERN = re.compile(r"^(.*)\.(recurrent_scan|chunk_kda)(\d+)$")


def resolve_hardware_alias(
    hardware: str, topology_preset: str | None = None,
) -> tuple[str, str | None]:
    """Expand intuitive CLI hardware/pod aliases into ``(sku, topology_preset)``."""
    key = str(hardware or "").strip().lower()
    if key in HARDWARE_PRESET_ALIASES:
        sku, preset = HARDWARE_PRESET_ALIASES[key]
        return sku, (topology_preset if topology_preset is not None else preset)
    return str(hardware).strip(), topology_preset


def compose_export_case(
    model: str,
    hardware: str,
    *,
    topology_preset: str | None = None,
    dp: int = 1,
    tp: int = 1,
    dcp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS,
    concurrency: int = 1,
    tpot_slo_ms: float = DEFAULT_TPOT_SLO_MS,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    feature_track: str = "merged",
    quant: str | None = None,
    kv_cache_dtype: str | None = None,
    engram_cpu_offload: bool | None = None,
    deployment_mode: str | None = None,
) -> tuple[dict[str, Any], StepRequest]:
    """Zero-boilerplate scenario composer returning ``(case_dict, StepRequest)``."""
    from mindie_sim.compose import compose_case

    sku, resolved_preset = resolve_hardware_alias(hardware, topology_preset)
    spec_toks = max(0, int(mtp_proposal_tokens))
    conc = max(1, int(concurrency))
    p_tok = max(1, int(prompt_tokens))
    o_tok = max(1, int(output_tokens))
    min_batched = max(conc * (1 + spec_toks), DEFAULT_MIN_BATCHED_TOKENS)
    model_len = max(p_tok + o_tok + DEFAULT_MODEL_LEN_HEADROOM, DEFAULT_MIN_MODEL_LEN)
    compose_kw: dict[str, Any] = {
        "tp": max(1, int(tp)),
        "dp": max(1, int(dp)),
        "ep": max(1, int(ep)),
        "pp": max(1, int(pp)),
        "dcp": max(1, int(dcp)),
        "pcp": max(1, int(pcp)),
        "prompt_tokens": p_tok,
        "output_tokens": o_tok,
        "concurrency": conc,
        "max_num_seqs": max(conc, 1),
        "max_num_batched_tokens": min_batched,
        "max_model_len": model_len,
        "mtp_proposal_tokens": spec_toks,
        "mtp_acceptance_length": float(mtp_acceptance_length),
        "tpot_slo_ms": float(tpot_slo_ms),
        "ttft_slo_ms": float(ttft_slo_ms),
        "feature_track": str(feature_track or "merged"),
    }
    if resolved_preset:
        compose_kw["topology_preset"] = resolved_preset
    resolved_quant = quant
    resolved_kv = kv_cache_dtype
    if str(quant or "").strip().lower() in {"auto_va", "auto"}:
        from mindie_sim.catalog.quant import resolve_auto_va_quant
        from mindie_sim.model_catalog import load_model

        m_spec = load_model(model)
        auto_w, auto_kv, _ = resolve_auto_va_quant(m_spec, sku, context_tokens=p_tok)
        resolved_quant = auto_w
        if resolved_kv is None:
            resolved_kv = auto_kv
    if resolved_quant is not None:
        compose_kw["quant"] = resolved_quant
    if resolved_kv is not None:
        compose_kw["kv_cache_dtype"] = resolved_kv
    if engram_cpu_offload is not None:
        compose_kw["engram_cpu_offload"] = bool(engram_cpu_offload)
    raw = compose_case(model, sku, **compose_kw)
    if deployment_mode and str(deployment_mode).strip().lower() != "auto":
        rt = dict(raw.get("runtime") or {})
        meta = dict(rt.get("metadata") or {})
        meta["deployment_mode"] = str(deployment_mode).strip()
        rt["metadata"] = meta
        raw["runtime"] = rt
    step_req = StepRequest(
        phase="decode",
        tokens=1 + spec_toks,
        context=p_tok,
        sequences=conc,
        scope="model",
    )
    return raw, step_req


_EXPORT_LAYER_QUANT_RE = re.compile(r"((?:layers|layer|blocks|h)[._])(\d+)([._]|$)")


def _layer_quant_sig(quant: Any, layer_idx: int) -> tuple[Any, ...]:
    if quant is None:
        return ()
    target = str(int(layer_idx))
    excl = sorted(
        _EXPORT_LAYER_QUANT_RE.sub(r"\1*\3", str(item))
        for item in (getattr(quant, "exclusions", ()) or ())
        if (m := _EXPORT_LAYER_QUANT_RE.search(str(item))) is not None and m.group(2) == target
    )
    raw_ov = getattr(quant, "category_overrides", ()) or ()
    ov_items = raw_ov.items() if isinstance(raw_ov, Mapping) else raw_ov
    ov = sorted(
        (
            _EXPORT_LAYER_QUANT_RE.sub(r"\1*\3", str(cat)),
            tuple(sorted(fields.items())) if isinstance(fields, Mapping) else tuple(fields),
        )
        for cat, fields in ov_items
        if (m := _EXPORT_LAYER_QUANT_RE.search(str(cat))) is not None and m.group(2) == target
    )
    return (tuple(excl), tuple(ov)) if (excl or ov) else ()


def _layer_archetype_key(
    lp: Any, stage: int, *, cand_producers: frozenset[int] = frozenset(), quant: Any = None,
) -> tuple[Any, ...]:
    idx = int(getattr(lp, "index", 0))
    kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
    idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
    topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
    cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
    idx_src = int(getattr(lp, "index_source_layer_id", -1))
    return (
        int(stage),
        str(getattr(lp, "attention", "")),
        str(getattr(lp, "ffn", "")),
        str(getattr(lp, "architecture", "")),
        bool(getattr(lp, "indexer", False)),
        str(getattr(lp, "indexer_type", "")),
        str(getattr(lp, "mla_variant", "")),
        bool(getattr(lp, "mla_output_gate", False)),
        bool(getattr(lp, "mla_use_nope", False)),
        int(getattr(lp, "num_heads", 0)),
        int(getattr(lp, "num_kv_heads", 0)),
        int(getattr(lp, "head_dim", 0)),
        int(getattr(lp, "intermediate_size", 0)),
        int(getattr(lp, "num_experts", 0)),
        int(getattr(lp, "top_k", 0)),
        int(getattr(lp, "moe_intermediate_size", 0)),
        int(getattr(lp, "num_shared_experts", 0)),
        int(getattr(lp, "shared_intermediate_size", 0)),
        int(getattr(lp, "linear_heads", 0)),
        int(getattr(lp, "linear_head_dim", 0)),
        int(getattr(lp, "expert_latent_size", 0)),
        bool(getattr(lp, "engram", False)),
        bool(getattr(lp, "hyper_connections", False)),
        int(getattr(lp, "window", 0)),
        str(getattr(lp, "cache_kind", "")),
        str(getattr(lp, "compressor_kind", "")),
        str(getattr(lp, "ffn_activation", "")),
        (kv_own >= 0, kv_own == idx),
        (idx_own >= 0, idx_own == idx),
        (topk_own >= 0, topk_own == idx),
        bool(getattr(lp, "candidate_filter", False)),
        int(getattr(lp, "candidate_topk_blocks", 0)),
        idx in cand_producers,
        cand_src >= 0,
        (idx_src >= 0, idx_src == idx),
        _layer_quant_sig(quant, idx),
    )


def _format_layer_range(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(int(x) for x in indices))
    if len(ordered) == 1:
        return f"L{ordered[0]}"
    if len(ordered) <= MAX_EXPLICIT_LAYER_LABELS:
        return ", ".join(f"L{x}" for x in ordered)
    diffs = {b - a for a, b in zip(ordered, ordered[1:])}
    if len(diffs) == 1:
        step = next(iter(diffs))
        if step == 1:
            return f"L{ordered[0]}..L{ordered[-1]}"
        return f"L{ordered[0]}, L{ordered[1]}, ..., L{ordered[-1]}"
    return f"L{ordered[0]}..L{ordered[-1]} ({len(ordered)}L)"


def _describe_layer_plan(lp: Any) -> tuple[str, str]:
    attn = str(getattr(lp, "attention", "gqa")).lower()
    ffn = str(getattr(lp, "ffn", "dense")).lower()
    indexer = bool(getattr(lp, "indexer", False))
    gated_mla = bool(getattr(lp, "mla_output_gate", False))
    latent_moe = int(getattr(lp, "expert_latent_size", 0) or 0) > 0
    engram = bool(getattr(lp, "engram", False))
    window = int(getattr(lp, "window", 0) or 0)

    if attn == "kda":
        attn_full, attn_short = "KDA", "KDA"
    elif attn == "gdn":
        attn_full, attn_short = "GDN", "GDN"
    elif attn in {"mla", "mla_dsa", "dsa"}:
        prefix = "Gated-MLA" if gated_mla else "MLA"
        attn_full = f"{prefix} + DSA (Lightning Indexer)" if indexer else prefix
        attn_short = f"{prefix}+DSA" if indexer else prefix
    elif window > 0 or attn == "swa":
        attn_full, attn_short = f"SWA (W={window})", "SWA"
    else:
        attn_full, attn_short = attn.upper(), attn.upper()
        if indexer:
            attn_full += " + Sparse Indexer"
            attn_short += "+LI"

    if ffn == "moe":
        ffn_full = "Latent MoE" if latent_moe else "MoE"
        ffn_short = "LatentMoE" if latent_moe else "MoE"
    else:
        ffn_full, ffn_short = "Dense MLP", "Dense"

    suffix = " + Engram/PLE" if engram else ""
    short_suf = "+Engram" if engram else ""
    return f"{attn_full} + {ffn_full}{suffix}", f"{attn_short}+{ffn_short}{short_suf}"


def _discover_model_archetypes(
    case: Case, *, include_draft: bool,
) -> tuple[
    tuple[LayerArchetypeSpec, ...],
    tuple[TraceScheduleEntry, ...],
    dict[int, Any],
    int,
    int,
]:
    stage_layers = stage_layer_counts(case)
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in case.model.layer_plans
        if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    groups: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()
    for lp in case.model.layer_plans:
        st = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        groups.setdefault(
            _layer_archetype_key(lp, st, cand_producers=cand_producers, quant=case.model.quant),
            [],
        ).append(lp)

    archetypes: list[LayerArchetypeSpec] = [
        LayerArchetypeSpec(
            "target_embed", "target", "Target Embedding & Pre-Process (×1)", "Embed",
            1, (), "Pre", None,
        ),
    ]
    rep_lp_by_idx: dict[int, Any] = {}
    arch_id_by_full_layer: dict[int, tuple[str, str]] = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, lps in enumerate(groups.values()):
        rep = lps[0]
        rep_lp_by_idx[idx] = rep
        full_desc, short_lbl = _describe_layer_plan(rep)
        indices = tuple(int(x.index) for x in lps)
        range_txt = _format_layer_range(indices)
        tag = letters[idx] if idx < len(letters) else str(idx)
        layer_prefix = f"Layer {indices[0]}: " if len(indices) == 1 else ""
        arch_id = f"target_arch_{idx}"
        archetypes.append(LayerArchetypeSpec(
            arch_id, "target",
            f"Target Archetype {tag} — {layer_prefix}{full_desc} (×{len(indices)}, {range_txt})",
            short_lbl, len(indices), indices, range_txt, idx,
        ))
        for full_idx in indices:
            arch_id_by_full_layer[full_idx] = (arch_id, short_lbl)

    archetypes.append(LayerArchetypeSpec(
        "target_tail", "target", "Target Final Norm & LM Head / Spec Verify (×1)",
        "LMHead/Verify", 1, (), "Post", None,
    ))

    aux = getattr(case.model, "auxiliary", None)
    dspark = getattr(aux, "dspark", None) if aux is not None else None
    mtp_spec = getattr(aux, "mtp", None) if aux is not None else None
    draft_layers = 0
    draft_label = "DSpark" if dspark is not None else "MTP"
    if include_draft and (dspark is not None or mtp_spec is not None):
        d_spec = dspark or mtp_spec
        draft_layers = max(1, int(getattr(d_spec, "layers", 1) or 1))
        d_desc = "MLA + Dense FFN" if str(getattr(d_spec, "attention_kind", "mla")).lower() == "mla" else "Draft Decoder"
        d_rng = f"L0..L{draft_layers - 1}" if draft_layers > 1 else "L0"
        d_idx_t = tuple(range(draft_layers))
        archetypes.extend([
            LayerArchetypeSpec("draft_pre", "draft", f"Draft [{draft_label}] — Context Feature Prep (×1)", f"Draft[{draft_label}]-Prep", 1, (), "Pre", None),
            LayerArchetypeSpec("draft_context_kv", "draft", f"Draft [{draft_label}] — Context KV Prolog (×{draft_layers})", f"Draft[{draft_label}]-CtxKV", draft_layers, d_idx_t, d_rng, 0),
            LayerArchetypeSpec("draft_arch_0", "draft", f"Draft [{draft_label}] — Parallel Block Draft Layer: {d_desc} (×{draft_layers})", f"Draft[{draft_label}]-Layer", draft_layers, d_idx_t, d_rng, 0),
            LayerArchetypeSpec("draft_tail", "draft", f"Draft [{draft_label}] — Final Norm, Logits & Markov Head (×1)", f"Draft[{draft_label}]-Head", 1, (), "Post", None),
        ])

    schedule: list[TraceScheduleEntry] = [
        TraceScheduleEntry("target_embed", "target", "Target Embedding & Pre-Process", None),
    ]
    for lp in case.model.layer_plans:
        a_id, short_lbl = arch_id_by_full_layer[int(lp.index)]
        schedule.append(TraceScheduleEntry(a_id, "target", f"Layer {lp.index} [{short_lbl}]", int(lp.index)))
    schedule.append(TraceScheduleEntry("target_tail", "target", "Target Final Norm & LM Head / Verify", None))
    if draft_layers > 0:
        schedule.append(TraceScheduleEntry("draft_pre", "draft", f"Draft [{draft_label}] Feature Prep", None))
        for d_idx in range(draft_layers):
            schedule.append(TraceScheduleEntry("draft_context_kv", "draft", f"Draft [{draft_label}] Context KV L{d_idx}", d_idx))
        for d_idx in range(draft_layers):
            schedule.append(TraceScheduleEntry("draft_arch_0", "draft", f"Draft [{draft_label}] Block Layer L{d_idx}", d_idx))
        schedule.append(TraceScheduleEntry("draft_tail", "draft", f"Draft [{draft_label}] Logits & Markov", None))

    return tuple(archetypes), tuple(schedule), rep_lp_by_idx, len(case.model.layer_plans), draft_layers


def _build_archetype_representative_case(
    case: Case, rep_lp_by_idx: Mapping[int, Any], *, allow_draft_dedup: bool,
) -> Case:
    old_to_new: dict[int, int] = {}
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in case.model.layer_plans
        if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    last_kv_prod = last_idx_prod = last_topk_prod = last_cand_prod = -1
    rep_list: list[Any] = []
    for new_idx, lp in sorted(rep_lp_by_idx.items()):
        old_idx = int(lp.index)
        old_to_new[old_idx] = new_idx
        kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
        idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
        topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
        cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
        idx_src = int(getattr(lp, "index_source_layer_id", -1))
        prev_kv, prev_idx, prev_topk, prev_cand = last_kv_prod, last_idx_prod, last_topk_prod, last_cand_prod

        if kv_own == old_idx:
            new_kv = last_kv_prod = new_idx
        else:
            new_kv = old_to_new.get(kv_own, prev_kv) if kv_own >= 0 else -1

        if idx_own == old_idx:
            new_idx_own = last_idx_prod = new_idx
        else:
            new_idx_own = old_to_new.get(idx_own, prev_idx) if idx_own >= 0 else -1

        if topk_own == old_idx:
            new_topk_own = last_topk_prod = new_idx
        else:
            new_topk_own = old_to_new.get(topk_own, prev_topk) if topk_own >= 0 else -1

        if old_idx in cand_producers:
            last_cand_prod = new_idx
        new_idx_src = new_idx if idx_src == old_idx else (old_to_new.get(idx_src, prev_idx) if idx_src >= 0 else -1)
        new_cand_src = old_to_new.get(cand_src, prev_cand) if cand_src >= 0 else -1
        rep_list.append(replace(
            lp, index=new_idx, kv_owner_layer_id=new_kv, index_k_owner_layer_id=new_idx_own,
            topk_owner_layer_id=new_topk_own, candidate_source_layer_id=new_cand_src,
            index_source_layer_id=new_idx_src,
        ))

    rep_plans = tuple(rep_list)
    if not rep_plans:
        return case
    aux = getattr(case.model, "auxiliary", None)
    rep_aux = aux
    if allow_draft_dedup and aux is not None and getattr(aux, "dspark", None) is not None:
        dsp = aux.dspark
        if dsp.layer_plans:
            d0 = replace(dsp.layer_plans[0], index=0, kv_owner_layer_id=0)
            clamped = tuple(min(int(lid), len(rep_plans) - 1) for lid in dsp.target_layer_ids)
            rep_aux = replace(aux, dspark=replace(dsp, layers=1, layer_plans=(d0,), target_layer_ids=clamped))
    rep_engram = getattr(case.model, "engram", None)
    if rep_engram is not None and getattr(rep_engram, "tables", ()):
        by_orig_layer = {int(t.layer_id): t for t in rep_engram.tables}
        default_tbl = rep_engram.tables[0]
        remapped = tuple(
            replace(by_orig_layer.get(int(orig_lp.index), default_tbl), layer_id=new_idx)
            for new_idx, orig_lp in sorted(rep_lp_by_idx.items())
            if bool(getattr(orig_lp, "engram", False))
        )
        if remapped:
            rep_engram = replace(rep_engram, tables=remapped)
    rep_model = replace(case.model, layers=len(rep_plans), layer_plans=rep_plans, auxiliary=rep_aux, engram=rep_engram)
    if case.runtime.parallel.pp > len(rep_plans):
        return case
    return replace(case, model=rep_model, runtime=replace(case.runtime, pp_layer_partition=()))


def _die_peaks_for_stage(case: Case, stage: int, replica: int = 0) -> dict[str, Any]:
    peaks = extract_hardware_peaks(case, stage=stage, replica=replica)
    die = peaks["die"]
    dpath = datapath_for_die(die)
    hbm_peak = float(peaks["hbm_peak_bytes_per_s"])
    return {
        **peaks,
        "hbm_read_bytes_per_s": max(float(dpath.hbm_read_bytes_per_s), hbm_peak),
        "hbm_write_bytes_per_s": max(float(dpath.hbm_write_bytes_per_s), hbm_peak),
        "hbm_rw_alpha": float(dpath.hbm_rw_alpha),
    }


def _die_cube_peak_rate(die: Any, dtype: str, default_dtype: str = "bf16") -> float:
    for candidate in (dtype, default_dtype, "bf16", "fp16"):
        if not candidate:
            continue
        try:
            return float(die.peak(str(candidate).strip().lower()))
        except ModelingError:
            continue
    return 0.0


def _op_lean_resource_demands_ns(
    *, flops: float, vector_ops: float, hbm_read: float, hbm_write: float,
    link_bytes: float, dtype: str, peaks: Mapping[str, Any],
    matrix_flops: Sequence[tuple[str, float]] = (),
) -> tuple[tuple[int, int, int, int, int], float]:
    default_dt = str(peaks.get("default_dtype") or "bf16")
    die = peaks["die"]
    cube_rate = _die_cube_peak_rate(die, dtype, default_dt)
    vec_rate = float(peaks["vector_peak_ops_per_s"] or 0.0)
    hbm_r_rate = float(peaks["hbm_read_bytes_per_s"] or 0.0)
    hbm_w_rate = float(peaks["hbm_write_bytes_per_s"] or 0.0)
    link_rate = float(peaks["link_peak_bytes_per_s"] or 0.0)
    alpha = float(peaks.get("hbm_rw_alpha", 0.0) or 0.0)

    def _demand_ns(work: float, rate: float) -> int:
        if work <= 0.0 or rate <= 0.0:
            return 0
        return inward_floor_ns(Fraction(work) * NS_PER_SECOND / Fraction(rate))

    valid_pairs = [(str(dt), float(f)) for dt, f in (matrix_flops or ()) if float(f) > 0.0]
    if valid_pairs:
        cube_frac = Fraction(0)
        total_mf = 0.0
        for dt, mf in valid_pairs:
            rate = _die_cube_peak_rate(die, dt, default_dt)
            if rate > 0.0:
                cube_frac += Fraction(mf) * NS_PER_SECOND / Fraction(rate)
                total_mf += mf
        cube_ns = inward_floor_ns(cube_frac) if cube_frac > 0 else 0
        if total_mf > 0.0 and cube_frac > 0:
            cube_rate = float(Fraction(total_mf) * NS_PER_SECOND / cube_frac)
    else:
        cube_ns = _demand_ns(flops, cube_rate)

    return (
        cube_ns,
        _demand_ns(vector_ops, vec_rate),
        _demand_ns(hbm_read + alpha * hbm_write, hbm_r_rate),
        _demand_ns(hbm_write + alpha * hbm_read, hbm_w_rate),
        _demand_ns(link_bytes, link_rate),
    ), cube_rate


def _classify_dominant_bound(
    demands_ns: tuple[int, int, int, int, int], dur_us: float, link_bytes: float,
) -> tuple[str, float | None]:
    d_cube, d_vec, d_hbm_r, d_hbm_w, d_link = demands_ns
    t_cube = float(d_cube) / NS_PER_US
    t_vec = float(d_vec) / NS_PER_US
    t_mem = float(max(d_hbm_r, d_hbm_w)) / NS_PER_US
    t_link = float(d_link) / NS_PER_US
    best_work = max(t_cube, t_vec, t_mem, t_link)
    eff = min(1.0, best_work / dur_us) if dur_us > 0.0 and best_work > 0.0 else None
    if link_bytes > 0.0 and t_link >= max(t_cube, t_vec, t_mem):
        return "link_bound", eff
    if dur_us > 0.0 and dur_us <= LAUNCH_BOUND_SHORT_US and best_work < dur_us * LAUNCH_BOUND_WORK_SHARE:
        return "launch_bound", eff
    if best_work <= 0.0:
        return "launch_bound", eff
    pairs = (("memory_bound", t_mem), ("cube_bound", t_cube), ("vector_bound", t_vec), ("link_bound", t_link))
    return max(pairs, key=lambda item: item[1])[0], eff


def _hbm_read_write_l2(spec: OpSpec | None) -> tuple[float, float, float, float, float, float, float]:
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    mte2 = float((spec.mte2_bytes if spec is not None else 0.0) or 0.0)
    mte3 = float((spec.mte3_bytes if spec is not None else 0.0) or 0.0)
    fixpipe = float(meta.get("fixpipe_bytes", 0.0) or 0.0)
    hbm_total = float(spec.hbm_bytes if spec is not None else 0.0)
    raw_write = mte3 + fixpipe
    copy_total = mte2 + raw_write
    if "l2_hit_rate" in meta and meta["l2_hit_rate"] is not None:
        l2_hit = max(0.0, min(1.0, float(meta["l2_hit_rate"])))
    elif copy_total > 0.0 and hbm_total > 0.0 and hbm_total < copy_total:
        l2_hit = max(0.0, min(1.0, 1.0 - hbm_total / copy_total))
    else:
        l2_hit = 0.0
    if copy_total > 0.0 and hbm_total > 0.0:
        scale = hbm_total / copy_total
        hbm_read, hbm_write = mte2 * scale, raw_write * scale
    elif "read_bytes" in meta:
        hbm_read = float(meta["read_bytes"] or 0.0)
        hbm_write = max(0.0, hbm_total - hbm_read)
    else:
        hbm_read, hbm_write = hbm_total, 0.0
    return hbm_read, hbm_write, hbm_total, mte2, mte3, fixpipe, l2_hit * 100.0


def _certify_step_lower_bound(
    op_rows: Sequence[OperatorExportRow],
    per_op_demands_ns: Sequence[tuple[int, int, int, int, int]],
) -> tuple[float, bool, tuple[str, ...]]:
    if not op_rows:
        return 0.0, True, ()
    res_pipes = ("cube", "vector", "hbm_read", "hbm_write", "link")
    die_coords = sorted({(row.occurrence.replica, row.occurrence.stage, int(row.occurrence.rank or 0)) for row in op_rows})
    coord_to_offset = {coord: idx * CERT_RESOURCE_COUNT for idx, coord in enumerate(die_coords)}
    totals = [0] * (len(die_coords) * CERT_RESOURCE_COUNT)
    resource_labels = [
        f"{pipe}@r{replica}.p{stage}.d{rank}"
        for replica, stage, rank in die_coords
        for pipe in res_pipes
    ]
    node_quanta: list[int] = []
    for row, demand in zip(op_rows, per_op_demands_ns):
        rep = max(1, int(row.repeat_count))
        base = coord_to_offset[(row.occurrence.replica, row.occurrence.stage, int(row.occurrence.rank or 0))]
        for idx in range(CERT_RESOURCE_COUNT):
            totals[base + idx] += demand[idx] * rep
        node_quanta.append(max(demand) * rep)

    index_by_occ = {row.occurrence: idx for idx, row in enumerate(op_rows)}
    edges: list[DagEdge] = [
        DagEdge(src=index_by_occ[pred], dst=dst_idx, wire_ns=0)
        for dst_idx, row in enumerate(op_rows)
        for pred in row.predecessors
        if pred in index_by_occ and index_by_occ[pred] < dst_idx
    ]
    prog = ResourceChoiceProgram(num_resources=len(totals), global_work=tuple(totals), site_modes=())
    dag = EventDagSpec(node_quantum_ns=tuple(node_quanta), edges=tuple(edges))
    cert, dual_report = certify_lower_bound(prog, dag)
    valid = check_optimality_certificate(prog, dag, cert)
    tight = tuple(
        resource_labels[int(tag[1:])]
        if tag.startswith("r") and tag[1:].isdigit() and int(tag[1:]) < len(resource_labels)
        else tag
        for tag in dual_report.tight_resources
    )
    return float(cert.l_star_lean_ns) / NS_PER_US, valid, tight


def _assign_op_archetype(
    op_id: str,
    spec: OpSpec | None,
    archetypes_by_id: Mapping[str, LayerArchetypeSpec],
    target_arch_by_rep_layer: Mapping[int, LayerArchetypeSpec],
) -> LayerArchetypeSpec:
    oid = str(op_id or "")
    layer_idx = extract_layer_index(oid, spec.layer if spec is not None else None)
    if oid.startswith("draft"):
        if ".context_kv" in oid:
            return archetypes_by_id.get("draft_context_kv") or next(iter(archetypes_by_id.values()))
        if ".s" in oid and layer_idx is not None:
            return archetypes_by_id.get("draft_arch_0") or next(iter(archetypes_by_id.values()))
        if any(tok in oid for tok in ("final_norm", "logits", "markov")):
            return archetypes_by_id.get("draft_tail") or next(iter(archetypes_by_id.values()))
        return archetypes_by_id.get("draft_pre") or next(iter(archetypes_by_id.values()))
    if oid.startswith("final.") or oid in {"sp_output_gather", "final_norm", "logits", "logits_gather", "accept"}:
        return archetypes_by_id["target_tail"]
    if layer_idx is not None and layer_idx in target_arch_by_rep_layer:
        return target_arch_by_rep_layer[layer_idx]
    if oid in {"embedding", "sp_shard", "iteration_input"} or (layer_idx is None and "final" not in oid):
        return archetypes_by_id["target_embed"]
    return archetypes_by_id["target_tail"]


def _collapse_micro_split_ops(
    rows: list[OperatorExportRow],
    demands: list[tuple[int, int, int, int, int]],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]:
    out_rows: list[OperatorExportRow] = []
    out_demands: list[tuple[int, int, int, int, int]] = []
    group_pos: dict[tuple[str, str], int] = {}
    for row, dem in zip(rows, demands):
        match = _RECURRENT_MICRO_PATTERN.match(row.op_id)
        if match is None:
            out_rows.append(row)
            out_demands.append(dem)
            continue
        key = (row.archetype_id, f"{match.group(1)}.{match.group(2)}")
        prev_idx = group_pos.get(key)
        if prev_idx is None:
            group_pos[key] = len(out_rows)
            out_rows.append(replace(row, op_id=f"{match.group(1)}.{match.group(2)}"))
            out_demands.append(dem)
            continue
        base = out_rows[prev_idx]
        b_dem = out_demands[prev_idx]
        merged_dem = tuple(b_dem[i] + dem[i] for i in range(CERT_RESOURCE_COUNT))
        new_dur = base.dur_us + row.dur_us
        new_l_star = float(max(merged_dem)) / NS_PER_US
        out_rows[prev_idx] = replace(
            base,
            start_us=min(base.start_us, row.start_us), end_us=max(base.end_us, row.end_us),
            dur_us=new_dur, exposed_dur_us=base.exposed_dur_us + row.exposed_dur_us,
            flops=base.flops + row.flops, vector_ops=base.vector_ops + row.vector_ops,
            hbm_read_bytes=base.hbm_read_bytes + row.hbm_read_bytes,
            hbm_write_bytes=base.hbm_write_bytes + row.hbm_write_bytes,
            hbm_total_bytes=base.hbm_total_bytes + row.hbm_total_bytes,
            mte2_bytes=base.mte2_bytes + row.mte2_bytes, mte3_bytes=base.mte3_bytes + row.mte3_bytes,
            fixpipe_bytes=base.fixpipe_bytes + row.fixpipe_bytes, link_bytes=base.link_bytes + row.link_bytes,
            l_star_lean_us=new_l_star, gap_us=new_dur - new_l_star,
            gap_ratio=(new_dur / new_l_star) if new_l_star > 0.0 else None,
        )
        out_demands[prev_idx] = merged_dem  # type: ignore[assignment]
    return out_rows, out_demands


def _fast_graph_id(ops: Sequence[OpSpec], model_name: str, phase: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}:{phase}:".encode("utf-8"))
    for op in ops:
        h.update(f"{op.op_id}|{op.family}|{op.recipe}|{op.local_shape}|".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:GRAPH_DIGEST_HEX_LEN]}"


def _extract_structure_registry_view(
    eff_case: Case, exec_graphs: Mapping[int, Sequence[OpSpec]], step_req: StepRequest,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if 0 not in exec_graphs:
        return {}, {}
    try:
        from mindie_sim.structure import export_op_structure, structure_inputs_from_graphs
        r0_graphs = {0: tuple(exec_graphs[0])}
        s_graphs, options = structure_inputs_from_graphs(
            eff_case, r0_graphs,
            phase=step_req.phase, tokens=step_req.tokens, context=step_req.context,
            sequences=step_req.sequences, scope=step_req.scope, replicas=[0],
            physical_graphs=r0_graphs,
        )
        report = export_op_structure(s_graphs, native_evidence=True, **options)
    except Exception:
        return {}, {}
    node_by_op: dict[str, dict[str, Any]] = {}
    for node in (*report.get("nodes", ()), *report.get("resource_operators", ())):
        if isinstance(node, Mapping) and node.get("op_id"):
            node_by_op[str(node["op_id"])] = dict(node)
    subgraph_by_op: dict[str, dict[str, Any]] = {}
    for sg in report.get("subgraphs", ()):
        if not isinstance(sg, Mapping):
            continue
        sg_dict = dict(sg)
        for nid in sg_dict.get("node_ids") or ():
            op_key = str(nid).split(":op:", 1)[-1] if ":op:" in str(nid) else str(nid).split(":", 1)[-1]
            subgraph_by_op.setdefault(op_key, sg_dict)
    return node_by_op, subgraph_by_op


def _extract_prepared_work_fallback(
    eff_case: Case, book: ReplayRecipeBook, ops: Sequence[OpSpec],
) -> dict[str, dict[str, float]]:
    """Query governed recipe ``book.prepare`` for ops whose ``OpSpec`` omits physical work."""
    die0 = eff_case.machine.dies[0]
    geom0 = Geometry(die0.compute_units, dict(die0.resource_units))
    out: dict[str, dict[str, float]] = {}
    for op in ops:
        if op.family == "barrier":
            continue
        if op.flops != 0 or op.vector_ops != 0 or op.hbm_bytes != 0 or op.link_bytes != 0:
            continue
        try:
            _, plan = book.prepare(op, geom0)
            work = getattr(plan, "work", None)
        except Exception:
            work = None
        if work is None:
            continue
        res = dict(getattr(work, "resources", {}) or {})
        mat = sum(float(v) for v in dict(getattr(work, "matrix", {}) or {}).values())
        hbm_b = float(res.get("hbm", 0.0) or 0.0)
        m2_b = (
            float(res.get("mte2_aiv", 0.0) or 0.0)
            + float(res.get("mte2_aic", 0.0) or 0.0)
            + float(res.get("mte2", 0.0) or 0.0)
        )
        m3_b = float(res.get("mte3", 0.0) or 0.0) + float(res.get("fixpipe", 0.0) or 0.0)
        if hbm_b <= 0.0 and (m2_b + m3_b) > 0.0:
            hbm_b = m2_b + m3_b
        vec_ops = float(res.get("vector", 0.0) or 0.0)
        link_b = float(res.get("link", 0.0) or 0.0)
        if mat > 0.0 or vec_ops > 0.0 or hbm_b > 0.0 or link_b > 0.0:
            out[op.op_id] = {
                "flops": mat,
                "vector_ops": vec_ops,
                "hbm_bytes": hbm_b,
                "mte2_bytes": m2_b,
                "mte3_bytes": m3_b,
                "link_bytes": link_b,
            }
    return out


def _evaluate_representative_step(
    spine: Any, rep_case: Case, step_req: StepRequest, replay: ReplayOptions,
) -> tuple[
    Case, list[dict[str, Any]], dict[tuple[int, str], OpSpec],
    dict[str, Sequence[Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], str,
    dict[str, dict[str, float]],
]:
    base = spine.stacks.resolve_base(rep_case, LoweringRequest(step=step_req))
    rewritten = spine.rewrite.apply(base)
    eff_case = (
        consume_target_declaration(base.case, rewritten.target_graphs)
        if rewritten.target_graphs is not None else base.case
    )
    book = ReplayRecipeBook.from_mapping(eff_case.runtime.kernels, family_registry=spine.families)
    book.graph = rewritten
    evaluation = EvaluationContext.create(eff_case, book, None)
    graphs = {entry.replica: entry.ops for entry in rewritten.executable}
    ext_inputs = (
        {"analysis_input": 0.0}
        if any("analysis_input" in op.deps for ops in graphs.values() for op in ops)
        else None
    )
    report = graph_costs(
        eff_case, graphs, recipes=book, external_inputs=ext_inputs,
        include_submission=replay.include_submission, isolated=replay.isolated,
        trace_limit=replay.event_limit, compact=False, evaluation=evaluation,
        stream_plans=rewritten.streams or None, request_views=rewritten.request_views,
        padded_tokens=step_req.padded_tokens, sequence_layout=step_req.layout,
        timeline_replicas=(0,),
    )
    op_spec_by_key = {(entry.replica, op.op_id): op for entry in rewritten.executable for op in entry.ops}
    prepared_work_by_op = _extract_prepared_work_fallback(eff_case, book, graphs.get(0, ()))
    struct_by_op, subgraph_by_op = _extract_structure_registry_view(eff_case, graphs, step_req)
    graph_id = _fast_graph_id(graphs.get(0, ()), str(eff_case.model.name), step_req.phase)
    timeline = list(report.get("timeline") or report.get("operators") or ())
    return eff_case, timeline, op_spec_by_key, {}, struct_by_op, subgraph_by_op, graph_id, prepared_work_by_op


def _timeline_to_raw_records(
    timeline: Sequence[Any], *, is_step_eval: bool, case: Case,
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
) -> list[dict[str, Any]]:
    width = case.runtime.parallel.tp * case.runtime.parallel.pcp
    records: list[dict[str, Any]] = []
    occ_by_node: dict[str, OpOccurrence] = {}
    if is_step_eval:
        for row in timeline:
            if row.occurrence.replica != 0:
                continue
            spec = op_spec_by_key.get((0, row.occurrence.op_id))
            records.append({
                "ordinal": row.ordinal, "occurrence": row.occurrence, "op_id": row.occurrence.op_id,
                "spec": spec, "family": row.family, "recipe": spec.recipe if spec is not None else "",
                "stream": row.stream, "shape": row.shape,
                "dtype": row.dtype or (spec.dtype if spec is not None else "") or case.model.compute_dtype or "bf16",
                "predecessors": row.predecessors,
                "start_us": max(0.0, float(row.start_ms or 0.0) * US_PER_MS),
                "end_us": max(0.0, float(row.end_ms or row.start_ms or 0.0) * US_PER_MS),
            })
        return records
    for row in timeline:
        if int(row.get("replica", 0)) != 0:
            continue
        oid = str(row.get("op_id", ""))
        spec = op_spec_by_key.get((0, oid))
        stage = int(spec.stage) if spec is not None else int(row.get("stage", 0))
        rank_idx = stage * width
        phys_rank = int(case.machine.dies[rank_idx].rank) if 0 <= rank_idx < len(case.machine.dies) else 0
        occ = OpOccurrence(replica=0, stage=stage, rank=phys_rank, op_id=oid)
        occ_by_node[f"r0:{oid}"] = occ
        if row.get("id"):
            occ_by_node[str(row["id"])] = occ
        start_us = max(0.0, float(row.get("start_ms", row.get("ready_ms", 0.0)) or 0.0) * US_PER_MS)
        dur_ms = float(row.get("duration_ms", 0.0) or 0.0)
        end_ms = row.get("complete_ms", row.get("end_ms"))
        end_us = max(start_us, float(end_ms) * US_PER_MS) if end_ms is not None else (start_us + max(0.0, dur_ms) * US_PER_MS)
        records.append({
            "ordinal": len(records), "occurrence": occ, "op_id": oid, "spec": spec,
            "family": str(spec.family if spec is not None else row.get("family", "")),
            "recipe": str(spec.recipe if spec is not None else row.get("recipe", "")),
            "stream": str((spec.stream if spec is not None else "") or row.get("stream", "") or "main"),
            "shape": tuple(spec.local_shape) if spec is not None else tuple(row.get("shape", ())),
            "dtype": str((spec.dtype if spec is not None else "") or row.get("dtype", "") or case.model.compute_dtype or "bf16"),
            "raw_preds": tuple(row.get("predecessors") or ()), "start_us": start_us, "end_us": end_us,
        })
    for rec in records:
        rec["predecessors"] = tuple(occ_by_node[p_id] for p_id in rec.pop("raw_preds", ()) if p_id in occ_by_node)
    return records


def _filter_barriers_and_rewire_preds(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    barrier_preds: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}
    kept: list[dict[str, Any]] = []
    for rec in records:
        dur = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        if str(rec["family"]).lower() == "barrier" and dur <= OVERLAP_EXPOSED_EPS_US:
            barrier_preds[rec["occurrence"]] = tuple(rec["predecessors"])
            continue
        rewired: list[OpOccurrence] = []
        for pred in rec["predecessors"]:
            if pred in barrier_preds:
                rewired.extend(barrier_preds[pred])
            else:
                rewired.append(pred)
        copy = dict(rec)
        copy["predecessors"] = tuple(dict.fromkeys(rewired))
        kept.append(copy)
    return kept


def _attach_action_guidance(
    rows: Sequence[OperatorExportRow], case: Case, *, phase: str,
) -> list[OperatorExportRow]:
    from .comparison import build_op_action_guidance

    floors = extract_sku_floors(case)
    sku_str = str(case.machine.sku or case.machine.name or "")
    out: list[OperatorExportRow] = []
    for row in rows:
        cfg, kern, fw, _ = build_op_action_guidance(
            row, is_a5_ub=floors.is_a5_ub, phase=phase, hardware_sku=sku_str,
        )
        out.append(replace(row, config_howto=cfg, kernel_action=kern, framework_action=fw))
    return out


def _row_to_target_node(row: OperatorExportRow, spec: OpSpec | None) -> TargetOpNode:
    if not _HAS_KERNEL_SYNTHESIS:
        return row
    return TargetOpNode(
        ordinal=row.ordinal, occurrence=row.occurrence, op_id=row.op_id,
        archetype_id=row.archetype_id, repeat_count=row.repeat_count,
        family=row.family, recipe=row.recipe, stream=row.stream, stream_index=row.stream_index,
        task_type=row.task_type, shape=row.shape, dtype=row.dtype, predecessors=row.predecessors,
        start_us=row.start_us, end_us=row.end_us, dur_us=row.dur_us,
        exposed_dur_us=row.exposed_dur_us, is_overlapped=row.is_overlapped, masked_by=row.masked_by,
        flops=row.flops, vector_ops=row.vector_ops,
        hbm_read_bytes=row.hbm_read_bytes, hbm_write_bytes=row.hbm_write_bytes,
        hbm_total_bytes=row.hbm_total_bytes, mte2_bytes=row.mte2_bytes,
        mte3_bytes=row.mte3_bytes, fixpipe_bytes=row.fixpipe_bytes, link_bytes=row.link_bytes,
        dominant_bound=row.dominant_bound, bound_display=row.bound_display,
        l_star_lean_us=row.l_star_lean_us, gap_us=row.gap_us, gap_ratio=row.gap_ratio,
        roofline_efficiency=row.roofline_efficiency, absorbed_source_ops=row.absorbed_source_ops,
        semantic_atoms=row.semantic_atoms, logical_role=row.logical_role,
        ascend_kernel=row.ascend_kernel, kernel_backend=row.kernel_backend,
        kernel_status=row.kernel_status, needs_new_kernel=row.needs_new_kernel,
        new_kernel_spec=row.new_kernel_spec, canonical_op_key=row.canonical_op_key,
        opt_pillar=row.opt_pillar,
        spec=spec, synthesis_spec=row.kernel_synthesis_spec,
    )


def _apply_node_to_row(base_row: OperatorExportRow, node: TargetOpNode) -> OperatorExportRow:
    if isinstance(node, OperatorExportRow):
        return node
    synth = getattr(node, "synthesis_spec", None)
    att_pct = (
        getattr(node, "opt_attainment_pct", None)
        if getattr(node, "opt_attainment_pct", None) is not None
        else (getattr(synth, "target_attainment_pct", None) if synth is not None else None)
    )
    b_state = str(getattr(node, "binding_state", "") or "").strip()
    if not b_state:
        b_state = (
            BINDING_STATE_UNBOUND_FORMULA
            if (getattr(node, "needs_new_kernel", False) or (synth is not None and not getattr(synth, "shipping_supported", True)))
            else base_row.binding_state
        )
    return replace(
        base_row,
        ordinal=node.ordinal, occurrence=node.occurrence, op_id=node.op_id,
        family=node.family, recipe=node.recipe, implementation=node.ascend_kernel or base_row.implementation,
        predecessors=node.predecessors, start_us=node.start_us, end_us=node.end_us,
        dur_us=node.dur_us, exposed_dur_us=node.exposed_dur_us,
        is_overlapped=node.is_overlapped, masked_by=node.masked_by,
        flops=node.flops, vector_ops=node.vector_ops,
        hbm_read_bytes=node.hbm_read_bytes, hbm_write_bytes=node.hbm_write_bytes,
        hbm_total_bytes=node.hbm_total_bytes, mte2_bytes=node.mte2_bytes,
        mte3_bytes=node.mte3_bytes, fixpipe_bytes=node.fixpipe_bytes, link_bytes=node.link_bytes,
        dominant_bound=node.dominant_bound, bound_display=node.bound_display,
        l_star_lean_us=node.l_star_lean_us, gap_us=node.gap_us, gap_ratio=node.gap_ratio,
        roofline_efficiency=node.roofline_efficiency, absorbed_source_ops=node.absorbed_source_ops,
        semantic_atoms=node.semantic_atoms, logical_role=node.logical_role,
        ascend_kernel=node.ascend_kernel, kernel_backend=node.kernel_backend,
        kernel_status=node.kernel_status, needs_new_kernel=node.needs_new_kernel,
        new_kernel_spec=node.new_kernel_spec, canonical_op_key=node.canonical_op_key,
        opt_pillar=node.opt_pillar, opt_attainment_pct=att_pct,
        kernel_synthesis_spec=synth, binding_state=b_state,
    )


def _attribute_rows_overlap(
    rows: Sequence[OperatorExportRow], archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], tuple[LayerArchetypeSpec, ...]]:
    nodes = [_row_to_target_node(r, None) for r in rows]
    attributed, spans = attribute_node_overlap(nodes)
    updated_rows = [_apply_node_to_row(r, n) for r, n in zip(rows, attributed)]
    updated_archs = tuple(replace(a, single_span_us=spans.get(a.archetype_id, 0.0)) for a in archetypes)
    return updated_rows, updated_archs


def _reorder_rows_by_archetype(
    rows: list[OperatorExportRow],
    demands: list[tuple[int, int, int, int, int]],
    archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]:
    arch_order = {arch.archetype_id: idx for idx, arch in enumerate(archetypes)}
    paired = sorted(
        zip(rows, demands),
        key=lambda item: (arch_order.get(item[0].archetype_id, ARCHETYPE_ORDER_FALLBACK), item[0].ordinal),
    )
    return (
        [replace(row, ordinal=new_ord) for new_ord, (row, _) in enumerate(paired)],
        [dem for _, dem in paired],
    )


def _compute_network_efficiency_summary(
    rows: Sequence[OperatorExportRow], step_span_us: float, peaks: Mapping[str, Any],
) -> tuple[float, float, float, float, float, dict[str, float], dict[str, dict[str, float]]]:
    tot_flops = sum(r.flops * r.repeat_count for r in rows)
    tot_vec = sum(r.vector_ops * r.repeat_count for r in rows)
    tot_hbm = sum(r.hbm_total_bytes * r.repeat_count for r in rows)
    tot_link = sum(r.link_bytes * r.repeat_count for r in rows)
    step_s = max(step_span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    default_dt = str(peaks.get("default_dtype") or "bf16")
    cube_peak = _die_cube_peak_rate(peaks["die"], default_dt, default_dt)
    vec_peak = float(peaks.get("vector_peak_ops_per_s") or 0.0)
    hbm_peak = float(peaks.get("hbm_peak_bytes_per_s") or 0.0)
    link_peak = float(peaks.get("link_peak_bytes_per_s") or 0.0)

    cube_mfu = min(100.0, 100.0 * (tot_flops / step_s) / cube_peak) if cube_peak > 0.0 else 0.0
    vec_mfu = min(100.0, 100.0 * (tot_vec / step_s) / vec_peak) if vec_peak > 0.0 else 0.0
    hbm_mbu = min(100.0, 100.0 * (tot_hbm / step_s) / hbm_peak) if hbm_peak > 0.0 else 0.0
    link_util = min(100.0, 100.0 * (tot_link / step_s) / link_peak) if link_peak > 0.0 else 0.0

    by_bound_dur: dict[str, float] = defaultdict(float)
    by_bound_cnt: dict[str, int] = defaultdict(int)
    by_bound_eff: dict[str, list[float]] = defaultdict(list)
    exposed_us = {"memory": 0.0, "cube": 0.0, "vector": 0.0, "communication": 0.0}
    total_scaled_dur = 0.0
    for r in rows:
        scaled = r.exposed_dur_us * r.repeat_count
        total_scaled_dur += scaled
        by_bound_dur[r.dominant_bound] += scaled
        by_bound_cnt[r.dominant_bound] += r.repeat_count
        if r.roofline_efficiency is not None:
            by_bound_eff[r.dominant_bound].append(r.roofline_efficiency)
        if r.dominant_bound == "memory_bound":
            exposed_us["memory"] += scaled
        elif r.dominant_bound == "cube_bound":
            exposed_us["cube"] += scaled
        elif r.dominant_bound == "vector_bound":
            exposed_us["vector"] += scaled
        elif r.dominant_bound == "link_bound":
            exposed_us["communication"] += scaled

    bottleneck_mix: dict[str, dict[str, float]] = {}
    for b_key in ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound"):
        dur_b = by_bound_dur.get(b_key, 0.0)
        effs = by_bound_eff.get(b_key, [])
        bottleneck_mix[b_key] = {
            "count": float(by_bound_cnt.get(b_key, 0)),
            "duration_us": dur_b,
            "share_pct": (100.0 * dur_b / total_scaled_dur) if total_scaled_dur > 0.0 else 0.0,
            "roofline_eff_pct": (sum(effs) / len(effs) * 100.0) if effs else 0.0,
        }
    tot_l_star = sum(r.l_star_lean_us * r.repeat_count for r in rows)
    net_roof = min(100.0, 100.0 * tot_l_star / step_span_us) if step_span_us > 0.0 else 0.0
    return cube_mfu, vec_mfu, hbm_mbu, link_util, net_roof, exposed_us, bottleneck_mix


def _finalize_ranked_rows(
    prelim_rows: Sequence[OperatorExportRow], observations_path: str | Path | None,
) -> tuple[tuple[OperatorExportRow, ...], list[int], ReconciliationSummary | None]:
    rank_order = sorted(
        range(len(prelim_rows)),
        key=lambda i: (-(prelim_rows[i].gap_us * prelim_rows[i].repeat_count), -(prelim_rows[i].gap_ratio or 0.0), prelim_rows[i].ordinal),
    )
    rank_by_idx = {idx: pos + 1 for pos, idx in enumerate(rank_order)}
    reconciled: dict[str, ReconciledKernelGroup] = {}
    reconcile_summary: ReconciliationSummary | None = None
    if observations_path is not None:
        obs_kernels, explicit_map = load_observation_bundle(observations_path)
        reconciled, reconcile_summary = reconcile_operators_with_kernels(
            prelim_rows, obs_kernels, explicit_map=explicit_map, source_label=Path(observations_path).name,
        )
    final_rows = tuple(
        replace(
            item,
            gap_radar_rank=rank_by_idx[idx],
            measured_kernels=reconciled[item.op_id].kernels if item.op_id in reconciled else (),
            measured_sum_us=reconciled[item.op_id].measured_sum_us if item.op_id in reconciled else None,
            ape_pct=reconciled[item.op_id].ape_pct if item.op_id in reconciled else None,
            binding_state=BINDING_STATE_MEASURED if item.op_id in reconciled else item.binding_state,
        )
        for idx, item in enumerate(prelim_rows)
    )
    return final_rows, rank_order, reconcile_summary


def _resolve_op_submodule(
    rec: Mapping[str, Any], struct_node: Mapping[str, Any] | None, lp: Any | None, arch_scope: str,
) -> str:
    if isinstance(struct_node, Mapping):
        raw_sub = str(struct_node.get("submodule") or "").strip()
        if raw_sub in CANONICAL_EXPORT_SUBMODULES:
            return raw_sub
    return classify_submodule(
        rec["op_id"], rec["family"], rec["recipe"],
        layer_attention=getattr(lp, "attention", "mla" if arch_scope == "draft" else ""),
        layer_ffn=getattr(lp, "ffn", "dense" if arch_scope == "draft" else ""),
    )


def _build_prelim_operator_rows(
    filtered_records: Sequence[Mapping[str, Any]],
    *,
    eff_case: Case,
    archetypes: Sequence[LayerArchetypeSpec],
    rep_lp_by_idx: Mapping[int, Any],
    use_archetype_dedup: bool,
    sources_by_op: Mapping[str, Sequence[Any]],
    struct_by_op: Mapping[str, Mapping[str, Any]],
    subgraph_by_op: Mapping[str, Mapping[str, Any]],
    prepared_work_by_op: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[int, dict[str, Any]]]:
    archetypes_by_id = {a.archetype_id: a for a in archetypes}
    target_arch_by_rep_layer = (
        {a.rep_layer_index: a for a in archetypes if a.scope == "target" and a.rep_layer_index is not None}
        if use_archetype_dedup
        else {full_idx: a for a in archetypes if a.scope == "target" for full_idx in a.layer_indices}
    )
    peaks_by_stage: dict[int, dict[str, Any]] = {}
    prelim_rows: list[OperatorExportRow] = []
    demands_ns: list[tuple[int, int, int, int, int]] = []
    seen_full_dedup: set[tuple[str, str]] = set()
    prep_map = prepared_work_by_op or {}
    for rec in filtered_records:
        occ: OpOccurrence = rec["occurrence"]
        spec: OpSpec | None = rec["spec"]
        if occ.stage not in peaks_by_stage:
            peaks_by_stage[occ.stage] = _die_peaks_for_stage(eff_case, occ.stage, 0)
        peaks = peaks_by_stage[occ.stage]
        arch = _assign_op_archetype(rec["op_id"], spec, archetypes_by_id, target_arch_by_rep_layer)
        lp = rep_lp_by_idx.get(arch.rep_layer_index) if (arch.scope == "target" and arch.rep_layer_index is not None) else None
        rep_count = arch.repeat_count if use_archetype_dedup else 1
        if use_archetype_dedup and arch.scope == "draft":
            dedup_key = (arch.archetype_id, strip_op_prefix(rec["op_id"]))
            if dedup_key in seen_full_dedup:
                continue
            seen_full_dedup.add(dedup_key)

        s_node = struct_by_op.get(rec["op_id"])
        sg_info = subgraph_by_op.get(rec["op_id"])
        layer_idx = extract_layer_index(rec["op_id"], spec.layer if spec is not None else None)
        submodule = _resolve_op_submodule(rec, s_node, lp, arch.scope)
        stream_idx = classify_stream_index(rec["stream"], rec["family"], rec["op_id"])
        dur_us = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        flops = float((spec.flops if spec is not None else 0.0) or 0.0)
        vec_ops = float((spec.vector_ops if spec is not None else 0.0) or 0.0)
        link_bytes = float((spec.link_bytes if spec is not None else 0.0) or 0.0)
        hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit = _hbm_read_write_l2(spec)
        if flops == 0.0 and vec_ops == 0.0 and hbm_tot == 0.0 and link_bytes == 0.0:
            pw = prep_map.get(rec["op_id"])
            if pw is not None:
                flops = float(pw.get("flops", 0.0))
                vec_ops = float(pw.get("vector_ops", 0.0))
                hbm_tot = float(pw.get("hbm_bytes", 0.0))
                mte2 = float(pw.get("mte2_bytes", 0.0))
                mte3 = float(pw.get("mte3_bytes", 0.0))
                copy_sum = mte2 + mte3
                hbm_r = (hbm_tot * mte2 / copy_sum) if copy_sum > 0.0 else (hbm_tot * 0.5)
                hbm_w = max(0.0, hbm_tot - hbm_r)
                link_bytes = float(pw.get("link_bytes", 0.0))
            elif dur_us > 0.0 and rec["shape"] and str(rec["family"]).lower() != "barrier":
                try:
                    elem_b = float(dtype_bytes(str(rec["dtype"] or "bf16").lower()))
                except ModelingError:
                    elem_b = 2.0
                elems = 1
                for dim in rec["shape"]:
                    elems *= max(1, int(dim))
                hbm_r = float(elems) * elem_b
                hbm_w = hbm_r
                hbm_tot = hbm_r + hbm_w
                mte2, mte3 = hbm_r, hbm_w
        op_demand, cube_peak = _op_lean_resource_demands_ns(
            flops=flops, vector_ops=vec_ops, hbm_read=hbm_r, hbm_write=hbm_w,
            link_bytes=link_bytes, dtype=rec["dtype"], peaks=peaks,
            matrix_flops=(spec.matrix_flops if spec is not None else ()),
        )
        l_star_us = float(max(op_demand)) / NS_PER_US
        dom_bound, roof_eff = _classify_dominant_bound(op_demand, dur_us, link_bytes)
        desc = resolve_ascend_kernel_identity(
            rec["op_id"], spec, rec["family"], rec["recipe"], rec["shape"], rec["dtype"],
            eff_case, lp, sources_by_op.get(rec["op_id"], ()),
            struct_node=s_node, subgraph_info=sg_info,
        )
        meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
        absorbed = tuple(str(x) for x in (meta.get("absorbed_source_ops") or meta.get("fused_from") or ()))
        sem_atoms = extract_semantic_atoms(
            rec["op_id"], rec["family"], rec["recipe"], absorbed, metadata=meta,
        )
        comm_dom = resolve_comm_domain_tag(rec["op_id"], rec["family"], rec["recipe"], meta, eff_case)
        op_binding = str(meta.get("binding_state") or "").strip() or (
            BINDING_STATE_UNBOUND_FORMULA if meta.get("unbound_formula_op") else BINDING_STATE_BOUND
        )
        demands_ns.append(op_demand)
        prelim_rows.append(OperatorExportRow(
            ordinal=len(prelim_rows), occurrence=occ, op_id=rec["op_id"],
            layer=layer_idx, layer_label=arch.title, submodule=submodule,
            family=rec["family"], recipe=rec["recipe"], implementation=desc.ascend_kernel,
            stream=rec["stream"], stream_index=stream_idx,
            task_type=classify_task_type(rec["family"], flops, vec_ops, link_bytes),
            shape=rec["shape"], dtype=rec["dtype"], predecessors=rec["predecessors"],
            start_us=float(rec["start_us"]), end_us=float(rec["end_us"]), dur_us=dur_us,
            flops=flops, vector_ops=vec_ops,
            hbm_read_bytes=hbm_r, hbm_write_bytes=hbm_w, hbm_total_bytes=hbm_tot,
            mte2_bytes=mte2, mte3_bytes=mte3, fixpipe_bytes=fixpipe,
            link_bytes=link_bytes, l2_hit_pct=l2_hit, dominant_bound=dom_bound,
            l_star_lean_us=l_star_us, gap_us=dur_us - l_star_us,
            gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
            gap_radar_rank=0, roofline_efficiency=roof_eff, absorbed_source_ops=absorbed,
            archetype_id=arch.archetype_id, archetype_title=arch.title,
            model_scope=arch.scope, repeat_count=rep_count,
            logical_role=desc.logical_role, ascend_kernel=desc.ascend_kernel,
            kernel_backend=desc.kernel_backend, kernel_status=desc.kernel_status,
            source_ref=desc.source_ref, tensor_contract=desc.tensor_contract,
            exposed_dur_us=dur_us, cube_peak_flops_per_s=cube_peak,
            bound_display=BOUND_DISPLAY_LABELS.get(dom_bound, dom_bound),
            needs_new_kernel=desc.needs_new_kernel, new_kernel_spec=desc.new_kernel_spec,
            canonical_op_key=desc.canonical_op_key,
            semantic_atoms=sem_atoms, comm_domain_tag=comm_dom,
            binding_state=op_binding,
        ))
    return prelim_rows, demands_ns, peaks_by_stage


def _resolve_deployment_mode(parsed: Case, phase: str) -> str:
    meta = dict(parsed.runtime.metadata)
    explicit = str(meta.get("deployment_mode") or meta.get("deployment") or "").strip()
    if explicit:
        low = explicit.lower()
        return DEPLOYMENT_MODE_LABELS.get(low, explicit)
    mixed = str(getattr(parsed.runtime, "mixed_mode", "none") or "none").strip().lower()
    p = parsed.runtime.parallel
    if mixed not in {"none", "off", ""}:
        return DEPLOYMENT_MODE_LABELS["colocated"]
    if p.pcp > 1 and p.cp > 1:
        return f"{DEPLOYMENT_MODE_LABELS['colocated']} (PCP+DCP)"
    prefix_mode = str(getattr(parsed.runtime, "prefix_cache_mode", "private") or "private").strip().lower()
    if prefix_mode in {"shared", "pooled"} or bool(meta.get("kv_pooled")):
        return DEPLOYMENT_MODE_LABELS["pd_pooled"]
    return DEPLOYMENT_MODE_LABELS["pd_unpooled"]


def _extract_scenario_meta(
    parsed: Case, step_req: StepRequest, final_rows: Sequence[OperatorExportRow],
) -> dict[str, Any]:
    meta = dict(parsed.runtime.metadata)
    topo_opts = dict(getattr(parsed.machine, "topology_options", ()) or ())
    topology_preset = str(topo_opts.get("topology_preset") or "")
    reqs = parsed.workload.requests
    prompt_toks = int(reqs[0].prompt_tokens) if reqs else int(step_req.context or DEFAULT_SCENARIO_PROMPT_TOKENS)
    output_toks = int(reqs[0].output_tokens) if reqs else int(DEFAULT_SCENARIO_OUTPUT_TOKENS)
    spec_tokens = int(meta.get("draft_proposal_tokens", 0) or 0)
    draft_mode = str(meta.get("draft_mode") or "").strip().lower()
    aux = getattr(parsed.model, "auxiliary", None)
    if spec_tokens > 0 or step_req.tokens > 1:
        if draft_mode == "dspark" or (aux is not None and getattr(aux, "dspark", None) is not None):
            spec_method = "DSpark"
        else:
            spec_method = "MTP"
        proposals = spec_tokens if spec_tokens > 0 else max(0, int(step_req.tokens) - 1)
    else:
        spec_method = "None"
        proposals = 0

    if step_req.phase == "decode":
        raw_accept = float(getattr(parsed.runtime, "mtp_acceptance_length", 1.0) or 1.0)
        avg_accepted = max(1.0, raw_accept) if proposals > 0 else 1.0
    else:
        avg_accepted = float(max(1, int(step_req.tokens)))

    exec_pol = meta.get("execution_policy") if isinstance(meta.get("execution_policy"), Mapping) else {}
    sp_enabled = bool(exec_pol.get("sp", False)) or any(
        "sp_gather" in r.op_id or "sp_shard" in r.op_id for r in final_rows
    )
    tpot_slo = float(getattr(parsed.slo, "tpot_ms", None) or DEFAULT_TPOT_SLO_MS)
    ttft_slo = float(getattr(parsed.slo, "ttft_ms", None) or DEFAULT_TTFT_SLO_MS)
    return {
        "topology_preset": topology_preset,
        "deployment_mode": _resolve_deployment_mode(parsed, step_req.phase),
        "prompt_tokens": prompt_toks,
        "output_tokens": output_toks,
        "tpot_slo_ms": tpot_slo,
        "ttft_slo_ms": ttft_slo,
        "spec_method": spec_method,
        "spec_proposals": proposals,
        "avg_accepted_tokens": avg_accepted,
        "sequence_parallel": sp_enabled,
    }


def build_export_bundle(
    case: CaseInput,
    *,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    include_vision_tower: bool | None = None,
    apply_preset_to_ignored: bool = False,
    apply_target_envelope: bool | None = None,
    rewrite_plan: Any = None,
    top_k_radar: int = DEFAULT_GAP_RADAR_TOP_K,
) -> ExportBundle:
    """Evaluate unique layer archetypes once on ``runtime.Engine`` into ``ExportBundle``."""
    parsed = case if isinstance(case, Case) else parse_case(dict(case))
    meta = dict(parsed.runtime.metadata)
    spec_tokens = int(meta.get("draft_proposal_tokens", 0) or 0)
    track_name = str(
        (meta.get("feature_track_report") or {}).get("feature_track")
        or meta.get("feature_track")
        or ""
    ).strip()
    step_req = request or (
        step_eval.request if step_eval is not None else StepRequest(
            phase="decode", tokens=1 + spec_tokens,
            context=max(1, min(parsed.runtime.max_model_len, DEFAULT_EXPORT_CONTEXT_TOKENS)),
            sequences=max(1, len(parsed.workload.requests)),
            scope="model" if spec_tokens > 0 else "body",
        )
    )
    seq_count = int(step_req.sequences) if isinstance(step_req.sequences, int) else max(1, len(step_req.sequences))
    include_draft = (
        step_req.scope == "model" and spec_tokens > 0 and step_req.tokens > 1
        and getattr(parsed.model, "auxiliary", None) is not None
    )
    archetypes, full_schedule, rep_lp_by_idx, total_target_layers, total_draft_layers = (
        _discover_model_archetypes(parsed, include_draft=include_draft)
    )
    use_archetype_dedup = (
        step_eval is None and observations_path is None
        and (len(rep_lp_by_idx) < len(parsed.model.layer_plans) or total_draft_layers > 1)
    )
    eval_case = (
        _build_archetype_representative_case(parsed, rep_lp_by_idx, allow_draft_dedup=True)
        if use_archetype_dedup else parsed
    )
    spine = default_spine()
    prepared_work_by_op: dict[str, dict[str, float]] = {}
    if step_eval is not None:
        prepared = spine.lower(eval_case, LoweringRequest(step=step_req))
        eff_case = prepared.case
        op_spec_by_key = {(e.replica, op.op_id): op for e in prepared.graph.executable for op in e.ops}
        exec_graphs = {e.replica: e.ops for e in prepared.graph.executable}
        struct_by_op, subgraph_by_op = _extract_structure_registry_view(eff_case, exec_graphs, step_req)
        raw_records = _timeline_to_raw_records(step_eval.operators, is_step_eval=True, case=eff_case, op_spec_by_key=op_spec_by_key)
        sources_by_op: dict[str, Sequence[Any]] = {r.occurrence.op_id: r.sources for r in prepared.operators if r.occurrence.replica == 0}
        graph_id, pipelines = step_eval.graph_id, step_eval.pipelines
    else:
        eff_case, timeline, op_spec_by_key, sources_by_op, struct_by_op, subgraph_by_op, graph_id, prepared_work_by_op = (
            _evaluate_representative_step(spine, eval_case, step_req, replay)
        )
        raw_records = _timeline_to_raw_records(timeline, is_step_eval=False, case=eff_case, op_spec_by_key=op_spec_by_key)
        pipelines = ()

    filtered_records = _filter_barriers_and_rewire_preds(raw_records)
    prelim_rows, demands_ns, peaks_by_stage = _build_prelim_operator_rows(
        filtered_records, eff_case=eff_case, archetypes=archetypes, rep_lp_by_idx=rep_lp_by_idx,
        use_archetype_dedup=use_archetype_dedup, sources_by_op=sources_by_op,
        struct_by_op=struct_by_op, subgraph_by_op=subgraph_by_op,
        prepared_work_by_op=prepared_work_by_op,
    )
    prelim_rows, demands_ns = _collapse_micro_split_ops(prelim_rows, demands_ns)
    should_synthesize = (
        apply_target_envelope
        if apply_target_envelope is not None
        else ((track_name == "expected_ideal" or rewrite_plan is not None) and step_eval is None and observations_path is None)
    )
    synthesis_specs: tuple[Any, ...] = ()
    audit_triggers: tuple[Any, ...] = ()
    plan_digest = ""
    plan_verified = True
    if should_synthesize and _HAS_KERNEL_SYNTHESIS:
        row_by_occ = {r.occurrence: r for r in prelim_rows}
        row_by_arch = {}
        for r in prelim_rows:
            row_by_arch.setdefault(r.archetype_id, r)
        scen_pre = _extract_scenario_meta(parsed, step_req, prelim_rows)
        n_tok_die = (
            float(max(1, seq_count * int(parsed.runtime.parallel.dp)))
            * float(scen_pre["avg_accepted_tokens"])
            / float(max(1, len(parsed.machine.dies)))
        )
        nodes = [_row_to_target_node(r, op_spec_by_key.get((0, r.op_id))) for r in prelim_rows]
        solved = solve_target_graph(
            nodes, demands_ns, eff_case, peaks_by_stage,
            phase=step_req.phase, rewrite_plan=rewrite_plan, n_tok_per_die=n_tok_die,
            allow_unbound_formula=True,
        )
        new_rows: list[OperatorExportRow] = []
        for node in solved.nodes:
            base_r = row_by_occ.get(node.occurrence) or row_by_arch.get(node.archetype_id) or prelim_rows[0]
            new_rows.append(_apply_node_to_row(base_r, node))
        prelim_rows = _attach_action_guidance(new_rows, eff_case, phase=step_req.phase)
        demands_ns = list(solved.demands_ns)
        archetypes = tuple(
            replace(a, single_span_us=solved.archetype_spans_us.get(a.archetype_id, 0.0))
            for a in archetypes
        )
        synthesis_specs = solved.synthesis_specs
        audit_triggers = solved.semantic_audit_triggers
        plan_digest = solved.rewrite_plan_sha256
        plan_verified = solved.verification_passed
    else:
        prelim_rows, archetypes = _attribute_rows_overlap(prelim_rows, archetypes)
        prelim_rows = _attach_action_guidance(prelim_rows, eff_case, phase=step_req.phase)

    prelim_rows, demands_ns = _reorder_rows_by_archetype(prelim_rows, demands_ns, archetypes)
    final_rows, rank_order, reconcile_summary = _finalize_ranked_rows(prelim_rows, observations_path)

    return _assemble_export_bundle(
        parsed=parsed, eff_case=eff_case, step_req=step_req, step_eval=step_eval,
        seq_count=seq_count, graph_id=graph_id, pipelines=pipelines, final_rows=final_rows,
        demands_ns=demands_ns, rank_order=rank_order, reconcile_summary=reconcile_summary,
        archetypes=archetypes, full_schedule=full_schedule,
        total_target_layers=total_target_layers, total_draft_layers=total_draft_layers,
        use_archetype_dedup=use_archetype_dedup, target_envelope_applied=should_synthesize,
        peaks_by_stage=peaks_by_stage, weight_quant=weight_quant, cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload, include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored, top_k_radar=top_k_radar,
        synthesis_specs=synthesis_specs, audit_triggers=audit_triggers,
        rewrite_plan_sha256=plan_digest, rewrite_verification_passed=plan_verified,
    )


def _build_checkpoint_and_va_summary(
    parsed: Case,
    mem_ledger: Any,
    scen: Mapping[str, Any],
    seq_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    hw_fam = normalize_hw_family(str(parsed.machine.sku or parsed.machine.name or "a5"))
    ckpt = resolve_default_checkpoint(str(parsed.model.name), hw_family=hw_fam)
    ckpt_summary = (
        {
            "repo_id": ckpt.repo_id,
            "quant_preset": ckpt.quant_preset,
            "kv_cache_dtype": ckpt.kv_cache_dtype,
            "uses_quarot": ckpt.uses_quarot,
            "modelers_url": ckpt.modelers_url,
        }
        if ckpt is not None else None
    )
    va_rules = build_va_serving_rules(
        str(parsed.model.name),
        hw_family=hw_fam,
        tp=mem_ledger.tp, dp=mem_ledger.dp, ep=mem_ledger.ep, pp=mem_ledger.pp,
        target_kv_dtype=mem_ledger.cache_dtype.lower(),
        enable_speculative=(int(scen["spec_proposals"]) > 0),
        num_speculative_tokens=(int(scen["spec_proposals"]) if int(scen["spec_proposals"]) > 0 else None),
        max_num_seqs=max(1, seq_count),
    )
    va_summary = {
        "effective_kv_cache_dtype": va_rules.effective_kv_cache_dtype,
        "speculative_enabled": va_rules.speculative_enabled,
        "speculative_method": va_rules.resolved_speculative.method if va_rules.resolved_speculative else "",
        "draft_model": va_rules.resolved_speculative.draft_model if va_rules.resolved_speculative else "",
        "draft_tp": va_rules.resolved_speculative.draft_tensor_parallel_size if va_rules.resolved_speculative else None,
        "capture_sizes": list(va_rules.compilation_config.get("cudagraph_capture_sizes") or ()),
        "hw_isa_guidance": va_rules.hw_isa_guidance,
    }
    return ckpt_summary, va_summary


def _assemble_export_bundle(
    *, parsed: Case, eff_case: Case, step_req: StepRequest, step_eval: StepEvaluation | None,
    seq_count: int, graph_id: str, pipelines: tuple[FusedPipelineDescriptor, ...],
    final_rows: tuple[OperatorExportRow, ...], demands_ns: Sequence[tuple[int, int, int, int, int]],
    rank_order: Sequence[int], reconcile_summary: ReconciliationSummary | None,
    archetypes: Sequence[LayerArchetypeSpec], full_schedule: Sequence[TraceScheduleEntry],
    total_target_layers: int, total_draft_layers: int, use_archetype_dedup: bool,
    target_envelope_applied: bool, peaks_by_stage: Mapping[int, dict[str, Any]],
    weight_quant: str | None, cache_dtype: str | None, engram_host_offload: bool | None,
    include_vision_tower: bool | None, apply_preset_to_ignored: bool, top_k_radar: int,
    synthesis_specs: tuple[Any, ...] = (), audit_triggers: tuple[Any, ...] = (),
    rewrite_plan_sha256: str = "", rewrite_verification_passed: bool = True,
) -> ExportBundle:
    exposed_total_us = sum(r.exposed_dur_us * r.repeat_count for r in final_rows)
    if use_archetype_dedup or target_envelope_applied:
        step_span_us = exposed_total_us
    elif step_eval is not None:
        step_span_us = float(step_eval.span_ms) * US_PER_MS
    else:
        step_span_us = exposed_total_us if exposed_total_us > 0.0 else max((r.end_us for r in final_rows), default=0.0)

    op_sum_us = sum(r.dur_us * r.repeat_count for r in final_rows)
    step_l_star_us, cert_valid, tight_res = _certify_step_lower_bound(final_rows, demands_ns)
    p0 = peaks_by_stage.get(0) or _die_peaks_for_stage(eff_case, 0, 0)
    cube_mfu, vec_mfu, hbm_mbu, link_util, net_roof, exposed_us, b_mix = _compute_network_efficiency_summary(
        final_rows, step_span_us, p0,
    )
    gap_radar = tuple(
        GapRadarRow(
            rank=final_rows[idx].gap_radar_rank,
            layer_label=final_rows[idx].archetype_title or final_rows[idx].layer_label,
            submodule=final_rows[idx].submodule, op_id=final_rows[idx].op_id,
            family=final_rows[idx].family,
            implementation=final_rows[idx].ascend_kernel or final_rows[idx].implementation,
            dominant_bound=final_rows[idx].bound_display or final_rows[idx].dominant_bound,
            l_star_lean_us=final_rows[idx].l_star_lean_us * final_rows[idx].repeat_count,
            pred_us=final_rows[idx].exposed_dur_us * final_rows[idx].repeat_count,
            gap_us=final_rows[idx].gap_us * final_rows[idx].repeat_count,
            gap_ratio=final_rows[idx].gap_ratio,
            share_of_step_pct=(100.0 * final_rows[idx].exposed_dur_us * final_rows[idx].repeat_count / step_span_us if step_span_us > 0.0 else 0.0),
            repeat_count=final_rows[idx].repeat_count,
            logical_role=final_rows[idx].logical_role, kernel_status=final_rows[idx].kernel_status,
        )
        for idx in rank_order[: max(1, int(top_k_radar))]
    )
    scen = _extract_scenario_meta(parsed, step_req, final_rows)
    step_s = max(step_span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    cluster_conc = max(1, seq_count * int(parsed.runtime.parallel.dp))
    tps = float(cluster_conc) * float(scen["avg_accepted_tokens"]) / step_s
    num_dies = max(1, len(parsed.machine.dies))
    n_tok_per_die = float(cluster_conc) * float(scen["avg_accepted_tokens"]) / float(num_dies)
    us_per_tok_die = step_span_us / max(n_tok_per_die, MIN_DENOM_EPS)
    default_dt = str(p0.get("default_dtype") or "bf16")
    hw_peaks_summary = {
        "cube_peak_flops_per_s": _die_cube_peak_rate(p0["die"], default_dt, default_dt),
        "vector_peak_ops_per_s": float(p0["vector_peak_ops_per_s"]),
        "hbm_peak_bytes_per_s": float(p0["hbm_peak_bytes_per_s"]),
        "mte2_peak_bytes_per_s": float(p0.get("mte2_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "mte3_peak_bytes_per_s": float(p0.get("mte3_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "fixpipe_peak_bytes_per_s": float(p0.get("fixpipe_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "link_peak_bytes_per_s": float(p0["link_peak_bytes_per_s"]),
    }
    mem_ledger = compute_memory_quant_ledger(
        parsed, phase=step_req.phase,
        seq_len=max(step_req.context + step_req.tokens, step_req.tokens),
        concurrency=seq_count, weight_quant=weight_quant,
        cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
        include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored,
    )
    eff_tpot_ms = (
        (step_span_us / US_PER_MS / max(1.0, float(scen["avg_accepted_tokens"])))
        if step_req.phase == "decode" else (step_span_us / US_PER_MS)
    )
    sla_target = float(scen["tpot_slo_ms"] if step_req.phase == "decode" else scen["ttft_slo_ms"])
    sla_slack = sla_target - eff_tpot_ms
    min_headroom = min((st.headroom_bytes for st in mem_ledger.stage_budgets), default=0.0)
    macro_wall = evaluate_macro_wall_trigger(
        tpot_ms=eff_tpot_ms,
        tpot_slo_ms=sla_target,
        hbm_used_bytes=mem_ledger.max_stage_hbm_bytes,
        hbm_usable_bytes=mem_ledger.usable_hbm_bytes,
        hbm_headroom_bytes=min_headroom,
    )
    sla_regime = (
        "SLA_VIOLATED" if eff_tpot_ms > sla_target
        else ("HBM_CAPACITY_BOUND" if macro_wall.wall_kind == "HBM_CAPACITY_WALL" else "SLA_LATENCY_BOUND")
    )
    ckpt_summary, va_summary = _build_checkpoint_and_va_summary(parsed, mem_ledger, scen, seq_count)
    d0 = parsed.machine.dies[0]
    d_bufs = dict(d0.buffers)
    aiv_cnt = int(d0.owner_count("vector") or 0)
    die_summary = (
        f"AIC={d0.aic_count}, AIV={aiv_cnt}, "
        f"L2={int(d_bufs.get('l2', 0)) // MIB_BYTES} MiB, "
        f"UB={int(d_bufs.get('ub', 0)) // KIB_BYTES} KiB, "
        f"HBM={hw_peaks_summary['hbm_peak_bytes_per_s'] / TB_BYTES:.2f} TB/s"
    )
    meta_dict = dict(parsed.runtime.metadata)
    track_name = str(
        (meta_dict.get("feature_track_report") or {}).get("feature_track")
        or meta_dict.get("feature_track")
        or ""
    ).strip()
    active_ids = {r.archetype_id for r in final_rows}
    filtered_schedule = tuple(e for e in full_schedule if e.archetype_id in active_ids) if use_archetype_dedup else ()
    has_unbound = any(r.binding_state == BINDING_STATE_UNBOUND_FORMULA for r in final_rows)
    bundle_mode = MODE_MODELING if (target_envelope_applied or track_name == "expected_ideal") else MODE_SIMULATION
    bundle_binding = (
        BINDING_STATE_UNBOUND_FORMULA if has_unbound
        else (BINDING_STATE_MEASURED if reconcile_summary is not None else BINDING_STATE_BOUND)
    )
    return ExportBundle(
        schema=EXPORT_BUNDLE_SCHEMA, model_name=str(parsed.model.name),
        hardware_sku=str(parsed.machine.sku or parsed.machine.name),
        phase=step_req.phase, scope=step_req.scope, graph_id=graph_id,
        tokens_per_seq=int(step_req.tokens), context_tokens=int(step_req.context), sequences=seq_count,
        step_span_us=step_span_us, operator_sum_us=op_sum_us,
        step_l_star_lean_us=step_l_star_us, step_certificate_valid=cert_valid,
        step_tight_resources=tight_res,
        throughput_tokens_per_s=tps, throughput_per_die_tokens_per_s=tps / float(num_dies),
        network_cube_mfu=cube_mfu, network_vec_mfu=vec_mfu,
        network_mbu=hbm_mbu, network_link_util=link_util, network_roofline_efficiency=net_roof,
        exposed_breakdown_us=exposed_us, bottleneck_mix=b_mix, hardware_peaks=hw_peaks_summary,
        operators=final_rows, pipelines=pipelines, gap_radar=gap_radar,
        memory_ledger=mem_ledger, reconciliation=reconcile_summary,
        archetypes=tuple(a for a in archetypes if a.archetype_id in active_ids),
        full_trace_schedule=filtered_schedule,
        total_target_layers=total_target_layers, total_draft_layers=total_draft_layers,
        graph_capture_mode=str(parsed.runtime.graph_capture_mode or "full"),
        feature_track=track_name, hardware_die_summary=die_summary,
        target_envelope_applied=target_envelope_applied,
        n_tok_per_die=n_tok_per_die,
        us_per_tok_die=us_per_tok_die,
        sla_regime=sla_regime,
        sla_slack_ms=sla_slack,
        kernel_synthesis_specs=synthesis_specs,
        semantic_audit_triggers=audit_triggers,
        macro_wall_trigger=macro_wall,
        rewrite_plan_sha256=rewrite_plan_sha256,
        rewrite_verification_passed=rewrite_verification_passed,
        recommended_checkpoint=ckpt_summary,
        va_serving_rules_summary=va_summary,
        analysis_mode=bundle_mode,
        binding_state=bundle_binding,
        has_unbound_formula_ops=has_unbound,
        **scen,
    )

