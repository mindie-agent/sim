"""O(Unique Archetypes) export bundle evaluation and scenario composition.

[Role] Compose scenarios (``resolve_hardware_alias``, ``compose_export_case``),
  discover unique Target/Draft layer archetypes, and evaluate each archetype
  once on ``runtime.Engine`` with certified physical lower bounds (``L*``), exact
  cross-stream overlap attribution, hardware efficiency (AIC MFU / AIV Util /
  HBM MBU / HCCL Link Util), canonical ``target`` track envelope, and
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

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields as dataclass_fields, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.catalog.npu_checkpoints import normalize_hw_family, resolve_default_checkpoint
from mindie_sim.stacks.vllm_ascend.speculative import build_va_serving_rules
from mindie_sim.optimization.causal_tail import (
    compute_acceptance_tail_closure,
    resolve_arch_family_tag,
    resolve_barrier_predecessors,
)
from mindie_sim.optimization.codesign import (
    collect_codesign_opportunities,
    evaluate_macro_wall_trigger,
)
from mindie_sim.optimization.dual_bound import (
    NS_PER_SECOND,
    inward_floor_ns,
)
from mindie_sim.optimization.ladder import solve_target_step_receipt
from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.efficiency import extract_hardware_peaks
from mindie_sim.prediction.efficiency.capacity_window import _resolve_speculative_tokens
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.layer_runs import (
    _is_target_body_op,
    discover_layer_cohorts,
    execute_native_cohort_graph,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operator_analysis import cohort_sequence_counts
from mindie_sim.prediction.results import (
    CaseInput,
    EngineStepReceipt,
    FusedPipelineDescriptor,
    OpOccurrence,
    ReplayOptions,
    StageReceipt,
    StepEvaluation,
    StepRequest,
)
from mindie_sim.prediction.rewrite import ReplayRecipeBook
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    attribute_stream0_intervals,
)
from mindie_sim.prediction.results import LoweringRequest
from mindie_sim.prediction.spine import PredictionSpine, default_spine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.primitives.loader import _table_band_p50
from mindie_sim.rewrite.oracle import prewarm_cohort_diff_ops, refine_layer_cohort_plan
from mindie_sim.rewrite.schema import load_rewrite_plan
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.scenario import ReadyStateAssumption, validate_deployment_mode
from mindie_sim.schema.stack import STACK_TRACKS

from .kernel_catalog import (
    classify_stream_index,
    classify_submodule,
    classify_task_type,
    extract_layer_index,
    extract_semantic_atoms,
    resolve_ascend_kernel_identity,
    resolve_comm_domain_tag,
    resolve_operator_display_dtype,
    strip_op_prefix,
)
from .memory_ledger import compute_memory_quant_ledger
from .models import (
    ARCHETYPE_ORDER_FALLBACK,
    BINDING_STATE_BOUND,
    BINDING_STATE_MEASURED,
    BINDING_STATE_UNBOUND_FORMULA,
    BOUND_DISPLAY_LABELS,
    CANONICAL_EXPORT_SUBMODULES,
    CERT_RESOURCE_COUNT,
    DEFAULT_EXPORT_TARGET_TRIALS,
    DEFAULT_GAP_RADAR_TOP_K,
    DEFAULT_MIN_BATCHED_TOKENS,
    DEFAULT_MIN_MODEL_LEN,
    DEFAULT_MODEL_LEN_HEADROOM,
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    DEPLOYMENT_MODE_LABELS,
    EXPORT_BUNDLE_SCHEMA,
    GRAPH_DIGEST_HEX_LEN,
    KERNEL_STATUS_NEW_OP,
    KIB_BYTES,
    LAUNCH_BOUND_SHORT_US,
    LAUNCH_BOUND_WORK_SHARE,
    MAX_EXPLICIT_LAYER_LABELS,
    MIB_BYTES,
    MIN_DENOM_EPS,
    MIN_POSITIVE_SECONDS,
    MODE_MODELING,
    MODE_SIMULATION,
    NS_PER_MS,
    NS_PER_US,
    OVERLAP_EXPOSED_EPS_US,
    SUB_ATTN_RES,
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
    op_dur_int_ns,
    op_exposed_int_ns,
    op_lstar_int_ns,
)
from .reconcile import load_observation_bundle, reconcile_operators_with_kernels

_RECURRENT_MICRO_PATTERN = re.compile(r"^(.*)\.(recurrent_scan|chunk_kda)(\d+)$")
_RECOVER_SSM_MICRO_PATTERN = re.compile(r"^(?:(s\d+)\.)?l\d+\.(recover_ssm_commit\w*)$")
_RECOVER_SSM_KERNEL_NAME = "RecoverSSMStateCommit"
_RECOVER_SSM_ROLE_CLEAN_RE = re.compile(r"\s*\(×\d+\s*layers?\)\s*$")
_OP_ROW_FIELDS = frozenset(f.name for f in dataclass_fields(OperatorExportRow))
_BUNDLE_FIELDS = frozenset(f.name for f in dataclass_fields(ExportBundle))
_MIN_FUSED_PIPELINE_MEMBERS = 2
_REWRITE_FORMULA_PREFIX = "rewrite_formula:"
_HALF_SPLIT_SHARE = 0.5


def _filter_dataclass_kwargs(allowed: frozenset[str], extra: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in extra.items() if k in allowed}


def resolve_hardware_alias(
    hardware: str, topology_preset: str | None = None,
) -> tuple[str, str | None]:
    """Expand CLI hardware/pod aliases via single-source ``compose.inputs``."""
    from mindie_sim.compose.inputs import (
        HARDWARE_ALIASES,
        HARDWARE_TOPOLOGY_ALIASES,
        _resolve_hardware,
    )

    raw = str(hardware or "").strip()
    key = raw.lower()
    if key in HARDWARE_TOPOLOGY_ALIASES:
        sku, preset = HARDWARE_TOPOLOGY_ALIASES[key]
        return sku, (topology_preset if topology_preset is not None else preset)
    if key in HARDWARE_ALIASES:
        return HARDWARE_ALIASES[key], topology_preset
    sku, _backend, alias_preset = _resolve_hardware(raw)
    return sku, (topology_preset if topology_preset is not None else (alias_preset or None))


_CERT_RESOURCE_NAMES: tuple[str, ...] = (
    "cube", "vector", "hbm_read", "hbm_write", "link",
)
_VALID_EXPORT_PHASES = frozenset({"decode", "prefill"})


def _resolve_case_stack_track(parsed: Case) -> str:
    """Project the resolved runtime stack track (including ``raw_eager=True``) onto canonical StackTrack."""
    meta = dict(parsed.runtime.metadata) if parsed.runtime.metadata else {}
    ep = meta.get("execution_policy") if isinstance(meta.get("execution_policy"), Mapping) else {}
    sp = meta.get("single_spine") if isinstance(meta.get("single_spine"), Mapping) else {}
    return str(
        (meta.get("feature_track_report") or {}).get("feature_track")
        or meta.get("feature_track")
        or sp.get("stack_track")
        or ("eager" if (bool(meta.get("raw_eager")) or bool(ep.get("raw_eager"))) else "")
    ).strip()


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
    tpot_slo_ms: float | None = None,
    ttft_slo_ms: float | None = None,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    feature_track: str = "shipping",
    quant: str | None = None,
    kv_cache_dtype: str | None = None,
    indexer_kv_dtype: str | None = None,
    engram_cpu_offload: bool | None = None,
    deployment_mode: str | None = None,
    runtime_stack: str | Path | Mapping[str, Any] | None = None,
    phase: str | None = None,
    ready_state_source: str = "pre_populated_hbm",
) -> tuple[dict[str, Any], StepRequest]:
    """Compose a 4-mode validated scenario returning ``(case_dict, StepRequest)``."""
    from mindie_sim.compose import compose_case

    canonical_track = str(feature_track or "").strip().lower()
    if canonical_track not in STACK_TRACKS:
        raise ModelingError(
            f"invalid_stack_track:{feature_track!r}:expected one of {sorted(STACK_TRACKS)}"
        )

    validated_mode: str | None = None
    if deployment_mode is not None:
        validated_mode = validate_deployment_mode(deployment_mode)
    req_phase = str(phase or "").strip().lower()
    if req_phase and req_phase not in _VALID_EXPORT_PHASES:
        raise ModelingError(f"invalid_export_phase:{phase!r}")

    if validated_mode == "pd_decode":
        if ttft_slo_ms is not None:
            raise ModelingError("pd_decode_forbids_ttft_slo_ms")
        if req_phase == "prefill":
            raise ModelingError("pd_decode_forbids_prefill_phase")
        eff_phase = "decode"
    elif validated_mode == "pd_prefill":
        if tpot_slo_ms is not None:
            raise ModelingError("pd_prefill_forbids_tpot_slo_ms")
        if req_phase == "decode":
            raise ModelingError("pd_prefill_forbids_decode_phase")
        eff_phase = "prefill"
    else:
        eff_phase = req_phase or "decode"

    sku, resolved_preset = resolve_hardware_alias(hardware, topology_preset)
    spec_toks = max(0, int(mtp_proposal_tokens))
    conc = max(1, int(concurrency))
    p_tok = max(1, int(prompt_tokens))
    o_tok = max(1, int(output_tokens))
    min_batched = max(
        p_tok * conc if eff_phase == "prefill" else conc * (1 + spec_toks),
        DEFAULT_MIN_BATCHED_TOKENS,
    )
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
        "_stack_phase": eff_phase,
        "feature_track": canonical_track,
    }
    if tpot_slo_ms is not None:
        compose_kw["tpot_slo_ms"] = float(tpot_slo_ms)
    if ttft_slo_ms is not None:
        compose_kw["ttft_slo_ms"] = float(ttft_slo_ms)
    if resolved_preset:
        compose_kw["topology_preset"] = resolved_preset
    if runtime_stack is not None:
        compose_kw["runtime_stack"] = runtime_stack
    if quant is not None:
        compose_kw["quant"] = quant
    if kv_cache_dtype is not None:
        compose_kw["kv_cache_dtype"] = kv_cache_dtype
    if indexer_kv_dtype is not None:
        compose_kw["indexer_kv_dtype"] = indexer_kv_dtype
    if engram_cpu_offload is not None:
        compose_kw["engram_cpu_offload"] = bool(engram_cpu_offload)
    if validated_mode is not None:
        compose_kw["deployment_mode"] = validated_mode

    case_dict = compose_case(model=model, hardware=sku, **compose_kw)
    rt_meta = case_dict.setdefault("runtime", {}).setdefault("metadata", {})
    rt_meta["feature_track"] = canonical_track
    if isinstance(rt_meta.get("feature_track_report"), dict):
        rt_meta["feature_track_report"]["feature_track"] = canonical_track
    if validated_mode is not None:
        rt_meta["deployment_mode"] = validated_mode
    if validated_mode == "pd_decode":
        ready = ReadyStateAssumption(
            source=ready_state_source,  # type: ignore[arg-type]
            context_tokens=p_tok,
            layout="paged_kv",
        )
        rt_meta["ready_state_assumption"] = {
            "source": ready.source,
            "context_tokens": ready.context_tokens,
            "layout": ready.layout,
        }

    if eff_phase == "prefill":
        step_req = StepRequest(phase="prefill", tokens=p_tok, context=0, sequences=conc, scope="model")
    else:
        step_req = StepRequest(
            phase="decode", tokens=1 + spec_toks, context=p_tok, sequences=conc, scope="model",
        )
    return case_dict, step_req


DEPTH_VARY_REL_THRESHOLD = 0.01


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
    cohort_plan = discover_layer_cohorts(case)
    archetypes: list[LayerArchetypeSpec] = [
        LayerArchetypeSpec(
            "target_embed", "target", "Target Embedding & Pre-Process (×1)", "Embed",
            1, (), "Pre", None,
        ),
    ]
    rep_lp_by_idx: dict[int, Any] = {}
    arch_id_by_full_layer: dict[int, tuple[str, str]] = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for cohort in cohort_plan.cohorts:
        idx = cohort.cohort_index
        rep = cohort.representative_plan
        rep_lp_by_idx[cohort.representative_layer] = rep
        full_desc, short_lbl = _describe_layer_plan(rep)
        indices = cohort.layer_indices
        range_txt = _format_layer_range(indices)
        tag = letters[idx] if idx < len(letters) else str(idx)
        layer_prefix = f"Layer {indices[0]}: " if len(indices) == 1 else ""
        arch_id = f"target_arch_{idx}"
        archetypes.append(LayerArchetypeSpec(
            arch_id, "target",
            f"Target Archetype {tag} — {layer_prefix}{full_desc} (×{len(indices)}, {range_txt})",
            short_lbl, len(indices), indices, range_txt, cohort.representative_layer,
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


def _die_cube_peak_rate(die: Any, dtype: str, default_dtype: str = "") -> float:
    tok = str(dtype or default_dtype or "").strip().lower()
    if not tok:
        raise ModelingError("missing_cube_peak_dtype")
    return float(die.peak(tok))


def _op_lean_resource_demands_ns(
    *, flops: float, vector_ops: float, hbm_read: float, hbm_write: float,
    link_bytes: float, dtype: str, peaks: Mapping[str, Any],
    matrix_flops: Sequence[tuple[str, float]] = (),
) -> tuple[tuple[int, int, int, int, int], float]:
    default_dt = str(peaks.get("default_dtype") or "").strip()
    die = peaks["die"]
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
        cube_rate = 0.0
        for dt, mf in valid_pairs:
            rate = _die_cube_peak_rate(die, dt, default_dt)
            if rate > 0.0:
                cube_frac += Fraction(mf) * NS_PER_SECOND / Fraction(rate)
                total_mf += mf
        cube_ns = inward_floor_ns(cube_frac) if cube_frac > 0 else 0
        if total_mf > 0.0 and cube_frac > 0:
            cube_rate = float(Fraction(total_mf) * NS_PER_SECOND / cube_frac)
        elif default_dt:
            cube_rate = _die_cube_peak_rate(die, default_dt, "")
    elif flops > 0.0:
        cube_rate = _die_cube_peak_rate(die, dtype, default_dt)
        cube_ns = _demand_ns(flops, cube_rate)
    else:
        cube_rate = _die_cube_peak_rate(die, default_dt, "") if default_dt else 0.0
        cube_ns = 0

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
    res_work = meta.get("resource_work") if isinstance(meta.get("resource_work"), Mapping) else {}
    mte2 = (
        float((spec.mte2_bytes if spec is not None else 0.0) or 0.0)
        + float(res_work.get("mte2_aic", 0.0) or 0.0)
        + float(res_work.get("mte2_aiv", 0.0) or 0.0)
    )
    mte3 = float((spec.mte3_bytes if spec is not None else 0.0) or 0.0)
    fixpipe = float(meta.get("fixpipe_bytes", 0.0) or 0.0) + float(res_work.get("fixpipe", 0.0) or 0.0)
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
    *,
    step_span_ns: int | None = None,
) -> tuple[int, bool, tuple[str, ...]]:
    """Certify exact integer-ns physical lower bound ``L*`` strictly on an unfolded DAG."""
    if not op_rows or not per_op_demands_ns:
        return 0, True, ()
    if len(op_rows) != len(per_op_demands_ns):
        raise ModelingError(
            f"certify_row_demand_length_mismatch:{len(op_rows)}!={len(per_op_demands_ns)}"
        )
    if any(int(row.repeat_count) != 1 for row in op_rows):
        raise ModelingError(
            "cannot_certify_folded_display_graph: step lower bound must be certified on unfolded rows"
        )
    res_sums = [0] * CERT_RESOURCE_COUNT
    op_min_by_occ: dict[OpOccurrence, int] = {}
    preds_by_occ: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}
    for row, dem in zip(op_rows, per_op_demands_ns):
        for idx in range(CERT_RESOURCE_COUNT):
            res_sums[idx] += int(dem[idx])
        op_min_by_occ[row.occurrence] = max(0, int(max(dem)))
        preds_by_occ[row.occurrence] = row.predecessors

    cp_cache: dict[OpOccurrence, int] = {}
    visiting: set[OpOccurrence] = set()
    for occ in op_min_by_occ:
        if occ in cp_cache:
            continue
        stack: list[tuple[OpOccurrence, bool]] = [(occ, False)]
        while stack:
            node, expanded = stack.pop()
            if node in cp_cache:
                continue
            if expanded:
                visiting.discard(node)
                best_pred = max((cp_cache[p] for p in preds_by_occ.get(node, ()) if p in cp_cache), default=0)
                cp_cache[node] = op_min_by_occ[node] + best_pred
                continue
            if node in visiting:
                cp_cache[node] = op_min_by_occ[node]
                continue
            visiting.add(node)
            stack.append((node, True))
            for pred in preds_by_occ.get(node, ()):
                if pred in op_min_by_occ and pred not in cp_cache:
                    stack.append((pred, False))

    dag_cp_ns = max(cp_cache.values(), default=0)
    max_res_ns = max(res_sums, default=0)
    l_star_ns = max(max_res_ns, dag_cp_ns)
    tight: list[str] = []
    if l_star_ns > 0:
        for idx, total in enumerate(res_sums):
            if total == l_star_ns:
                tight.append(_CERT_RESOURCE_NAMES[idx])
        if dag_cp_ns == l_star_ns:
            tight.append("critical_path")
    if step_span_ns is not None:
        cert_valid = l_star_ns <= int(step_span_ns)
    else:
        min_s = min(int(round(r.start_us * NS_PER_US)) for r in op_rows)
        max_e = max(int(round(r.end_us * NS_PER_US)) for r in op_rows)
        cert_valid = l_star_ns <= max(0, max_e - min_s)
    return int(l_star_ns), bool(cert_valid), tuple(tight)


def _assign_op_archetype(
    op_id: str,
    spec: OpSpec | None,
    archetypes_by_id: Mapping[str, LayerArchetypeSpec],
    target_arch_by_rep_layer: Mapping[int, LayerArchetypeSpec],
    *,
    occurrence: OpOccurrence | None = None,
    tail_closure: frozenset[OpOccurrence] = frozenset(),
    absorbed_source_ops: Sequence[str] = (),
) -> LayerArchetypeSpec:
    if occurrence is not None and occurrence in tail_closure:
        return archetypes_by_id["target_tail"]
    oid = str(op_id or "")
    layer_idx = extract_layer_index(oid, spec.layer if spec is not None else None)
    if layer_idx is None and absorbed_source_ops:
        for src in absorbed_source_ops:
            src_str = str(src or "")
            if src_str.startswith("draft"):
                oid = src_str
                layer_idx = extract_layer_index(src_str, None)
                break
            src_layer = extract_layer_index(src_str, None)
            if src_layer is not None:
                layer_idx = src_layer
                break
    if oid.startswith("draft"):
        if ".context_kv" in oid:
            return archetypes_by_id.get("draft_context_kv") or next(iter(archetypes_by_id.values()))
        if ".s" in oid and layer_idx is not None:
            return archetypes_by_id.get("draft_arch_0") or next(iter(archetypes_by_id.values()))
        if any(tok in oid for tok in ("final_norm", "logits", "markov", "select_hidden", ".final.")):
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
    """Group micro-split recurrent scan & tail RecoverSSM commit rows after pricing & DAG attribution."""
    out_rows: list[OperatorExportRow] = []
    out_demands: list[tuple[int, int, int, int, int]] = []
    group_pos: dict[tuple[str, str], int] = {}
    group_counts: dict[tuple[str, str], int] = {}
    group_is_recover: dict[tuple[str, str], bool] = {}
    group_preds: dict[tuple[str, str], list[OpOccurrence]] = {}
    renamed: dict[str, str] = {}
    for row, dem in zip(rows, demands):
        match = _RECURRENT_MICRO_PATTERN.match(row.op_id)
        is_recover_ssm = False
        if match is None and row.archetype_id == "target_tail":
            match = _RECOVER_SSM_MICRO_PATTERN.match(row.op_id)
            is_recover_ssm = match is not None
        if match is None:
            out_rows.append(row)
            out_demands.append(dem)
            continue
        collapsed_id = (
            f"{match.group(1)}.{match.group(2)}"
            if match.group(1) else str(match.group(2))
        )
        key = (row.archetype_id, collapsed_id)
        renamed[row.op_id] = collapsed_id
        prev_idx = group_pos.get(key)
        if prev_idx is None:
            group_pos[key] = len(out_rows)
            group_counts[key] = 1
            group_is_recover[key] = is_recover_ssm
            group_preds[key] = list(row.predecessors)
            row_rep = max(1, int(row.repeat_count))
            first_dur_ns = op_dur_int_ns(row, row_rep)
            first_exp_ns = op_exposed_int_ns(row, row_rep)
            first_ls_ns = int(max(dem)) if max(dem) > 0 else op_lstar_int_ns(row, row_rep)
            out_rows.append(replace(
                row,
                occurrence=replace(row.occurrence, op_id=collapsed_id),
                op_id=collapsed_id,
                layer=None if is_recover_ssm else row.layer,
                dur_ns=first_dur_ns,
                exposed_ns=first_exp_ns,
                l_star_ns=first_ls_ns,
            ))
            out_demands.append(dem)
            continue
        group_counts[key] += 1
        group_preds[key].extend(row.predecessors)
        base = out_rows[prev_idx]
        b_dem = out_demands[prev_idx]
        base_rep = max(1, int(base.repeat_count))
        row_rep = max(1, int(row.repeat_count))
        merged_dem = tuple(b_dem[i] + dem[i] for i in range(CERT_RESOURCE_COUNT))
        new_dur_ns = int(base.dur_ns) + op_dur_int_ns(row, row_rep)
        new_exposed_ns = int(base.exposed_ns) + op_exposed_int_ns(row, row_rep)
        new_l_star_ns = int(max(merged_dem))
        denom_us = float(base_rep) * NS_PER_US
        new_dur = float(new_dur_ns) / denom_us if new_dur_ns > 0 else (base.dur_us + row.dur_us)
        new_exposed = float(new_exposed_ns) / denom_us if new_exposed_ns > 0 else (base.exposed_dur_us + row.exposed_dur_us)
        new_l_star = float(new_l_star_ns) / denom_us
        out_rows[prev_idx] = replace(
            base,
            start_us=min(base.start_us, row.start_us), end_us=max(base.end_us, row.end_us),
            dur_us=new_dur, exposed_dur_us=new_exposed,
            dur_ns=new_dur_ns, exposed_ns=new_exposed_ns, l_star_ns=new_l_star_ns,
            is_overlapped=(new_exposed_ns == 0 and new_dur_ns > 0),
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
    if renamed:
        for key, pos in group_pos.items():
            cnt = max(1, int(group_counts.get(key, 1)))
            row_cur = out_rows[pos]
            if group_is_recover.get(key, False):
                inv = 1.0 / float(cnt)
                per_call_dem = tuple(int(out_demands[pos][k]) // cnt for k in range(CERT_RESOURCE_COUNT))
                per_call_dur = float(row_cur.dur_ns) / (float(cnt) * NS_PER_US) if row_cur.dur_ns > 0 else row_cur.dur_us * inv
                per_call_exp = float(row_cur.exposed_ns) / (float(cnt) * NS_PER_US) if row_cur.exposed_ns > 0 else row_cur.exposed_dur_us * inv
                per_call_lstar = float(row_cur.l_star_ns) / (float(cnt) * NS_PER_US) if row_cur.l_star_ns > 0 else float(max(per_call_dem)) / NS_PER_US
                fully_hidden = int(row_cur.exposed_ns) == 0 and int(row_cur.dur_ns) > 0
                eff = (
                    min(1.0, per_call_lstar / per_call_dur)
                    if (per_call_dur > 0.0 and per_call_lstar > 0.0) else None
                )
                clean_role = _RECOVER_SSM_ROLE_CLEAN_RE.sub("", row_cur.logical_role or "").strip()
                out_demands[pos] = per_call_dem  # type: ignore[assignment]
                out_rows[pos] = replace(
                    row_cur,
                    repeat_count=cnt,
                    end_us=row_cur.start_us + per_call_dur,
                    dur_us=per_call_dur,
                    exposed_dur_us=0.0 if fully_hidden else per_call_exp,
                    is_overlapped=fully_hidden,
                    flops=row_cur.flops * inv,
                    vector_ops=row_cur.vector_ops * inv,
                    hbm_read_bytes=row_cur.hbm_read_bytes * inv,
                    hbm_write_bytes=row_cur.hbm_write_bytes * inv,
                    hbm_total_bytes=row_cur.hbm_total_bytes * inv,
                    mte2_bytes=row_cur.mte2_bytes * inv,
                    mte3_bytes=row_cur.mte3_bytes * inv,
                    fixpipe_bytes=row_cur.fixpipe_bytes * inv,
                    link_bytes=row_cur.link_bytes * inv,
                    l_star_lean_us=per_call_lstar,
                    gap_us=0.0 if fully_hidden else max(0.0, per_call_dur - per_call_lstar),
                    gap_ratio=None if fully_hidden else ((per_call_dur / per_call_lstar) if per_call_lstar > 0.0 else None),
                    roofline_efficiency=eff,
                    ascend_kernel=_RECOVER_SSM_KERNEL_NAME,
                    logical_role=clean_role or row_cur.logical_role,
                    predecessors=tuple(group_preds[key]),
                )
            else:
                out_rows[pos] = replace(row_cur, predecessors=tuple(group_preds[key]))
        out_rows = [
            replace(
                r,
                predecessors=tuple(dict.fromkeys(
                    replace(p, op_id=renamed.get(p.op_id, p.op_id))
                    for p in r.predecessors
                    if renamed.get(p.op_id, p.op_id) != r.op_id
                )),
            )
            if (any(p.op_id in renamed for p in r.predecessors) or (r.archetype_id, r.op_id) in group_pos)
            else r
            for r in out_rows
        ]
    return out_rows, out_demands


def _fast_graph_id(
    graphs_or_ops: Mapping[int, Sequence[OpSpec]] | Sequence[OpSpec],
    model_name: str,
    phase: str,
) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}:{phase}:".encode("utf-8"))
    if isinstance(graphs_or_ops, Mapping):
        for rep, ops in sorted(graphs_or_ops.items()):
            h.update(f"r{int(rep)}:".encode("utf-8"))
            for op in ops:
                h.update(f"{op.op_id}|{op.family}|{op.recipe}|{op.local_shape}|{op.dtype}|".encode("utf-8"))
    else:
        for op in graphs_or_ops:
            h.update(f"{op.op_id}|{op.family}|{op.recipe}|{op.local_shape}|{op.dtype}|".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:GRAPH_DIGEST_HEX_LEN]}"


def _extract_structure_registry_view(
    eff_case: Case,
    exec_graphs: Mapping[int, Sequence[OpSpec]],
    step_req: StepRequest,
    *,
    semantic_graphs: Mapping[int, Sequence[OpSpec]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if 0 not in exec_graphs:
        return {}, {}
    from mindie_sim.structure import export_op_structure, structure_inputs_from_graphs
    r0_phys = {0: tuple(exec_graphs[0])}
    sem_graphs: Any = (
        semantic_graphs
        if (semantic_graphs is not None and 0 in semantic_graphs)
        else r0_phys
    )
    s_graphs, options = structure_inputs_from_graphs(
        eff_case, sem_graphs,
        phase=step_req.phase, tokens=step_req.tokens, context=step_req.context,
        sequences=step_req.sequences, scope=step_req.scope, replicas=[0],
        physical_graphs=r0_phys,
    )
    report = export_op_structure(s_graphs, native_evidence=True, **options)
    native_by_node_id: dict[str, dict[str, Any]] = {}
    for call in report.get("native_calls") or ():
        if isinstance(call, Mapping) and call.get("node_id"):
            native_by_node_id[str(call["node_id"])] = dict(call)
    node_by_op: dict[str, dict[str, Any]] = {}
    for node in (*report.get("nodes", ()), *report.get("resource_operators", ())):
        if isinstance(node, Mapping) and node.get("op_id"):
            nd = dict(node)
            nid = str(nd.get("node_id") or "")
            if nid and nid in native_by_node_id:
                nd["native_call"] = native_by_node_id[nid]
            node_by_op[str(nd["op_id"])] = nd
    subgraph_by_op: dict[str, dict[str, Any]] = {}
    for sg in report.get("subgraphs", ()):
        if not isinstance(sg, Mapping):
            continue
        sg_dict = dict(sg)
        for nid in sg_dict.get("node_ids") or ():
            op_key = str(nid).split(":op:", 1)[-1] if ":op:" in str(nid) else str(nid).split(":", 1)[-1]
            subgraph_by_op.setdefault(op_key, sg_dict)
    return node_by_op, subgraph_by_op


def _resolved_execution_dtype(spec: OpSpec | None) -> str:
    """Read resolved execution Cube/compute precision from a prepared ``OpSpec``."""
    if spec is None:
        return ""
    meta = dict(spec.metadata) if spec.metadata else {}
    exec_dt = str(meta.get("execution_compute_format") or "").strip()
    if exec_dt:
        return exec_dt
    return str(spec.dtype or "").strip()


def _collect_persisted_prepared_ops(
    evaluation: Any,
    report: Mapping[str, Any] | None = None,
) -> dict[tuple[int, str], OpSpec]:
    """Read persisted prepared ``OpSpec`` entries from ``EvaluationContext`` or report."""
    persisted: dict[tuple[int, str], OpSpec] = {}
    candidates = (
        getattr(evaluation, "prepared_ops", None),
        getattr(evaluation, "_prepared_ops", None),
        getattr(evaluation, "resolved_ops", None),
        (report.get("prepared_ops") if isinstance(report, Mapping) else None),
    )
    for mapping in candidates:
        if not isinstance(mapping, Mapping) or not mapping:
            continue
        for key, val in mapping.items():
            if not isinstance(val, OpSpec):
                continue
            if isinstance(key, tuple) and len(key) == 2:
                persisted[(int(key[0]), str(key[1]))] = val
            elif isinstance(key, str):
                persisted[(0, key)] = val
        if persisted:
            break
    return persisted


def _stage_devices_for_replica(case: Case, stage: int, replica: int = 0) -> tuple[Any, ...]:
    p = case.runtime.parallel
    width = max(1, int(p.tp) * int(p.pcp))
    start = (int(replica) * max(1, int(p.pp)) + int(stage)) * width
    dies = tuple(case.machine.dies[start:start + width])
    return dies if dies else (case.machine.dies[0],)


def _prepare_op_spec_via_book(
    phys_op: OpSpec,
    *,
    case: Case,
    book: ReplayRecipeBook,
    stage_dies: Sequence[Any],
    stage_geom: Geometry,
) -> OpSpec:
    from mindie_sim.prediction.collectives.waves import startup_waves_recipe
    from mindie_sim.prediction.operators.matrix.matrix_precision import resolve_matrix_precision

    prep_op, plan = book.prepare(phys_op, stage_geom)
    plan = startup_waves_recipe(prep_op, plan, case.runtime.collective_startup_mode)
    return resolve_matrix_precision(
        prep_op, devices=stage_dies,
        policy=case.runtime.matrix_compute_policy, recipe=plan,
    )


def _resolve_prepared_op_specs(
    entries: Sequence[Any],
    evaluation: Any,
    *,
    case: Case | None = None,
    book: ReplayRecipeBook | None = None,
    report: Mapping[str, Any] | None = None,
    request_views: Sequence[Any] = (),
    padded_tokens: int = 0,
    sequence_layout: str = "",
    reduced_r0: Sequence[OpSpec] | None = None,
    op_rep_map: Mapping[str, str] | None = None,
) -> dict[tuple[int, str], OpSpec]:
    """Populate ``op_spec_by_key`` from prepared execution ``OpSpec``s on replica 0."""
    op_spec_by_key: dict[tuple[int, str], OpSpec] = {
        (int(entry.replica), op.op_id): op for entry in entries for op in entry.ops
    }
    persisted = _collect_persisted_prepared_ops(evaluation, report)
    op_spec_by_key.update(persisted)
    prepare_fn = getattr(evaluation, "prepare", None)
    eff_case = case if case is not None else getattr(evaluation, "case", None)
    if not callable(prepare_fn) and (book is None or eff_case is None):
        return op_spec_by_key
    stage_cache: dict[int, tuple[tuple[Any, ...], Geometry]] = {}
    op_memo: dict[tuple[Any, ...], tuple[tuple[Any, ...], OpSpec]] = {}
    req_tuple = tuple(request_views or ())
    pad_tok = int(padded_tokens or 0)
    seq_layout = str(sequence_layout or "")
    rep_map = op_rep_map or {}

    def _prepare_one(phys_op: OpSpec) -> OpSpec:
        if phys_op.family == "barrier" or phys_op.source_boundary == "call":
            return phys_op
        if callable(prepare_fn):
            prep_op, _plan = prepare_fn(phys_op, 0)
            return prep_op if isinstance(prep_op, OpSpec) else phys_op
        assert eff_case is not None and book is not None
        st = int(phys_op.stage)
        if st not in stage_cache:
            s_dies = _stage_devices_for_replica(eff_case, st, 0)
            s_geom = Geometry.from_devices(
                s_dies, links=eff_case.machine.links,
                transfer_channels=eff_case.machine.transfer_channels,
            )
            if req_tuple or pad_tok or seq_layout:
                s_geom = replace(
                    s_geom, requests=req_tuple,
                    padded_tokens=pad_tok, layout=seq_layout,
                )
            stage_cache[st] = (s_dies, s_geom)
        s_dies, s_geom = stage_cache[st]
        memo_key = (
            st, phys_op.family, phys_op.recipe, phys_op.phase, phys_op.dtype,
            phys_op.layout, phys_op.local_shape, phys_op.padded_shape,
            phys_op.flops, phys_op.matrix_flops, phys_op.mte2_bytes,
            phys_op.mte3_bytes, phys_op.vector_ops, phys_op.hbm_bytes,
            phys_op.link_bytes, phys_op.source_boundary,
        )
        cached = op_memo.get(memo_key)
        if cached is not None and cached[0] == phys_op.metadata:
            return replace(
                cached[1], op_id=phys_op.op_id, layer=phys_op.layer,
                deps=phys_op.deps, requests=phys_op.requests,
                stream=phys_op.stream, notes=phys_op.notes,
            )
        resolved = _prepare_op_spec_via_book(
            phys_op, case=eff_case, book=book,
            stage_dies=s_dies, stage_geom=s_geom,
        )
        op_memo[memo_key] = (phys_op.metadata, resolved)
        return resolved

    if reduced_r0 is not None and rep_map:
        rep_phys = tuple(book.expand(reduced_r0)) if book is not None else tuple(reduced_r0)
        for phys_op in rep_phys:
            key = (0, phys_op.op_id)
            if key not in persisted:
                op_spec_by_key[key] = _prepare_one(phys_op)
        for oid, rep_oid in rep_map.items():
            key = (0, oid)
            if key in persisted:
                continue
            orig = op_spec_by_key.get(key)
            rep_spec = op_spec_by_key.get((0, rep_oid))
            if orig is not None and rep_spec is not None and (
                orig.family == rep_spec.family
                and orig.recipe == rep_spec.recipe
                and orig.local_shape == rep_spec.local_shape
                and orig.flops == rep_spec.flops
                and orig.hbm_bytes == rep_spec.hbm_bytes
            ):
                op_spec_by_key[key] = replace(
                    rep_spec, op_id=orig.op_id, layer=orig.layer,
                    deps=orig.deps, requests=orig.requests,
                    stream=orig.stream, notes=orig.notes,
                )
            elif orig is not None:
                op_spec_by_key[key] = _prepare_one(orig)
        return op_spec_by_key

    for entry in entries:
        if int(entry.replica) != 0:
            continue
        phys_ops = tuple(book.expand(entry.ops)) if book is not None else tuple(entry.ops)
        for phys_op in phys_ops:
            key = (0, phys_op.op_id)
            if key not in persisted:
                op_spec_by_key[key] = _prepare_one(phys_op)
    return op_spec_by_key


def _evaluate_representative_step(
    spine: Any, rep_case: Case, step_req: StepRequest, replay: ReplayOptions,
    *, primitive_table: Any = None, streams: tuple[Any, ...] = (),
) -> tuple[
    Case, list[dict[str, Any]], dict[tuple[int, str], OpSpec],
    dict[str, Sequence[Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], str,
    dict[str, dict[str, float]], Any, float,
]:
    base = spine.stacks.resolve_base(rep_case, LoweringRequest(step=step_req))
    rewritten = spine.rewrite.apply(base, streams=tuple(streams))
    eff_case = (
        consume_target_declaration(base.case, rewritten.target_graphs)
        if rewritten.target_graphs is not None else base.case
    )
    book = ReplayRecipeBook.from_mapping(eff_case.runtime.kernels, family_registry=spine.families)
    book.graph = rewritten
    evaluation = EvaluationContext.create(eff_case, book, None, primitive_table=primitive_table)
    rebound_primitive_table = False if primitive_table is False else evaluation.bindings.primitive_table
    graphs = {entry.replica: entry.ops for entry in rewritten.executable}
    ext_inputs = (
        {"analysis_input": 0.0}
        if any("analysis_input" in op.deps for ops in graphs.values() for op in ops)
        else None
    )
    eff_streams = tuple(rewritten.streams or ()) or tuple(streams)
    refined_cohort = refine_layer_cohort_plan(eff_case, graphs[0]) if 0 in graphs else None
    if refined_cohort is not None and 0 in graphs:
        prewarm_cohort_diff_ops(eff_case, graphs[0], refined_cohort, evaluation, book)
    report, reduced_graphs, op_rep_map = execute_native_cohort_graph(
        eff_case, graphs, cohort_plan=refined_cohort, recipes=book, external_inputs=ext_inputs,
        include_submission=replay.include_submission, isolated=replay.isolated,
        trace_limit=replay.event_limit, compact=False, evaluation=evaluation,
        stream_plans=eff_streams or None, request_views=rewritten.request_views,
        padded_tokens=step_req.padded_tokens, sequence_layout=step_req.layout,
        timeline_replicas=(0,),
    )
    op_spec_by_key = _resolve_prepared_op_specs(
        rewritten.executable, evaluation,
        case=eff_case, book=book, report=report,
        request_views=rewritten.request_views,
        padded_tokens=step_req.padded_tokens,
        sequence_layout=step_req.layout,
        reduced_r0=(reduced_graphs[0] if (reduced_graphs is not None and 0 in reduced_graphs) else None),
        op_rep_map=op_rep_map,
    )
    prepared_work_by_op: dict[str, dict[str, float]] = {}
    if reduced_graphs is not None and 0 in reduced_graphs:
        rep_r0 = reduced_graphs[0]
        kept_layers = {int(op.layer) for op in rep_r0 if int(op.layer) >= 0}
        sem_src = rewritten.target_graphs
        if sem_src is not None and 0 in sem_src:
            sem_filtered = {
                0: tuple(
                    op for op in sem_src[0]
                    if int(op.layer) < 0 or int(op.layer) in kept_layers
                    or op.op_id.startswith(("draft", "final."))
                )
            }
        else:
            sem_filtered = None
        struct_by_op, subgraph_by_op = _extract_structure_registry_view(
            eff_case, reduced_graphs, step_req, semantic_graphs=sem_filtered,
        )
        for oid, rep_oid in op_rep_map.items():
            if rep_oid in struct_by_op and oid not in struct_by_op:
                struct_by_op[oid] = struct_by_op[rep_oid]
            if rep_oid in subgraph_by_op and oid not in subgraph_by_op:
                subgraph_by_op[oid] = subgraph_by_op[rep_oid]
    else:
        struct_by_op, subgraph_by_op = _extract_structure_registry_view(
            eff_case, graphs, step_req, semantic_graphs=rewritten.target_graphs,
        )
    graph_id = _fast_graph_id(graphs, str(eff_case.model.name), step_req.phase)
    timeline = list(report.get("timeline") or report.get("operators") or ())
    report_span_ms = float(report.get("duration_ms", 0.0) or 0.0)
    return (
        eff_case, timeline, op_spec_by_key, {}, struct_by_op, subgraph_by_op,
        graph_id, prepared_work_by_op, rebound_primitive_table, report_span_ms,
    )


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
                "dtype": _resolved_execution_dtype(spec) or row.dtype or case.model.compute_dtype or "",
                "predecessors": row.predecessors,
                "work": tuple(getattr(row, "work", ()) or ()),
                "implementation": str(getattr(row, "implementation", "") or ""),
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
            "dtype": str(_resolved_execution_dtype(spec) or row.get("dtype", "") or case.model.compute_dtype or ""),
            "raw_preds": tuple(row.get("predecessors") or ()), "start_us": start_us, "end_us": end_us,
        })
    for rec in records:
        rec["predecessors"] = tuple(occ_by_node[p_id] for p_id in rec.pop("raw_preds", ()) if p_id in occ_by_node)
    return records


def _filter_barriers_and_rewire_preds(
    records: Sequence[Mapping[str, Any]],
    *,
    protected_occurrences: frozenset[OpOccurrence] = frozenset(),
) -> list[dict[str, Any]]:
    return resolve_barrier_predecessors(
        records,
        protected_occurrences=protected_occurrences,
        barrier_eps_us=OVERLAP_EXPOSED_EPS_US,
    )


def _attribute_full_bundle_stream0(
    rows: Sequence[OperatorExportRow],
    archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], tuple[LayerArchetypeSpec, ...], int, int]:
    """Run conserved integer-ns ``ExposedObserver`` attribution over full-depth native rows."""
    if not rows:
        return [], tuple(archetypes), 0, 0
    occ_to_id = {r.occurrence: str(i) for i, r in enumerate(rows)}
    intervals: list[ObservedInterval] = []
    preds: dict[str, tuple[str, ...]] = {}
    min_ns: int | None = None
    max_ns = 0
    for idx, row in enumerate(rows):
        s_ns = max(0, int(round(float(row.start_us) * NS_PER_US)))
        e_ns = max(s_ns, int(round(float(row.end_us) * NS_PER_US)))
        min_ns = s_ns if min_ns is None else min(min_ns, s_ns)
        max_ns = max(max_ns, e_ns)
        oid = str(idx)
        intervals.append(ObservedInterval(
            occurrence_id=oid, stream=row.stream, stream_index=row.stream_index,
            start_ns=s_ns, end_ns=e_ns, ordinal=row.ordinal,
            label=row.logical_role or row.ascend_kernel or row.op_id,
            replica=int(row.occurrence.replica),
        ))
        preds[oid] = tuple(occ_to_id[p] for p in row.predecessors if p in occ_to_id)
    start_i = min_ns if min_ns is not None else 0
    step_span_ns = max(0, max_ns - start_i)
    res = attribute_stream0_intervals(intervals, start_ns=start_i, terminal_ns=max_ns, causal_preds=preds)
    spans_ns: dict[str, int] = defaultdict(int)
    out_rows: list[OperatorExportRow] = []
    total_exposed_ns = 0
    for idx, row in enumerate(rows):
        oid = str(idx)
        dur_ns = max(0, int(row.dur_ns) if int(row.dur_ns) > 0 else int(round(float(row.dur_us) * NS_PER_US)))
        exp_ns = max(0, int(res.exposed_ns_by_op.get(oid, 0)))
        if dur_ns > 0:
            exp_ns = min(dur_ns, exp_ns)
        total_exposed_ns += exp_ns
        exp_us = float(exp_ns) / NS_PER_US
        fully_hidden = dur_ns > 0 and exp_ns == 0
        masker = res.masked_by.get(oid, "") if (dur_ns > exp_ns) else ""
        b_disp = f"Overlapped by {masker} (0 exposed)" if (fully_hidden and masker) else row.bound_display
        out_rows.append(replace(
            row,
            dur_ns=dur_ns,
            exposed_ns=exp_ns,
            exposed_dur_us=exp_us,
            is_overlapped=fully_hidden,
            masked_by=masker,
            gap_us=0.0 if fully_hidden else max(0.0, row.dur_us - row.l_star_lean_us),
            gap_ratio=None if fully_hidden else ((row.dur_us / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None),
            bound_display=b_disp,
        ))
        spans_ns[row.archetype_id] += exp_ns
    idle_or_wait_ns = max(0, step_span_ns - total_exposed_ns)
    updated_archs = tuple(
        replace(
            a,
            single_span_us=float(spans_ns.get(a.archetype_id, 0)) / (NS_PER_US * float(max(1, a.repeat_count))),
        )
        for a in archetypes
    )
    return out_rows, updated_archs, step_span_ns, idle_or_wait_ns


def _is_attnres_op(op_id: str, family: str = "") -> bool:
    low = f"{op_id} {family}".lower()
    return any(tok in low for tok in ("attn_res", "attnres", "self_attention_res", "mlp_res"))


def format_attnres_depth_note(
    op_id: str,
    layer_indices: Sequence[int],
    durs_us: Sequence[float],
    block_size: int,
    *,
    total_target_layers: int = 0,
    family: str = "",
) -> str:
    """Format exact physical layer depth & ``valid_blocks`` progression note."""
    low = str(op_id or "").lower()
    is_attnres = _is_attnres_op(low, family)
    if is_attnres and block_size > 0 and low.startswith("final."):
        n_layers = int(total_target_layers) or (max(layer_indices, default=0) + 1)
        v_final = (n_layers + int(block_size) - 1) // int(block_size)
        return f"final={n_layers}，valid_blocks={v_final}，Repeats=1"
    if not layer_indices or not durs_us:
        return ""
    pairs = sorted(zip(layer_indices, durs_us), key=lambda kv: kv[0])
    first_l, first_d = int(pairs[0][0]), float(pairs[0][1])
    last_l, last_d = int(pairs[-1][0]), float(pairs[-1][1])
    if is_attnres and block_size > 0:
        bs = int(block_size)
        if "mlp" in low or "ffn" in low:
            v_min = (first_l // bs) + 1
            v_max = (last_l // bs) + 1
        else:
            v_min = (first_l + bs - 1) // bs
            v_max = (last_l + bs - 1) // bs
        return (
            f"L{first_l} -> L{last_l}，随层深 valid_blocks={v_min}..{v_max} 递增: "
            f"{first_d:.2f} us -> {last_d:.2f} us"
        )
    min_d, max_d = min(durs_us), max(durs_us)
    if min_d > 0.0 and (max_d - min_d) / min_d > DEPTH_VARY_REL_THRESHOLD:
        return f"L{first_l} -> L{last_l}，随层深递增: {first_d:.2f} us -> {last_d:.2f} us"
    return ""


_DRAFT_STEP_PREFIX_RE = re.compile(r"^draft(\d+)\.")
_REWRITE_FORMULA_MIN_PARTS = 3


def _collect_draft_proposal_steps(rows: Sequence[OperatorExportRow]) -> tuple[int, ...]:
    steps: set[int] = set()
    for row in rows:
        if row.model_scope == "draft":
            m_step = _DRAFT_STEP_PREFIX_RE.match(row.op_id)
            if m_step is not None:
                steps.add(int(m_step.group(1)))
    return tuple(sorted(steps))


def _per_layer_timing_key(
    row: OperatorExportRow,
    multi_draft_step: bool,
) -> tuple[str, int] | None:
    if row.archetype_id in {"target_embed", "target_tail"}:
        return (row.archetype_id, -1)
    if row.model_scope == "draft":
        m_step = _DRAFT_STEP_PREFIX_RE.match(row.op_id)
        step_suffix = f"_step{m_step.group(1)}" if (multi_draft_step and m_step is not None) else ""
        if row.archetype_id == "draft_context_kv" and row.layer is not None and row.layer >= 0:
            return (f"draft_context_kv{step_suffix}", int(row.layer))
        if row.archetype_id == "draft_arch_0" and row.layer is not None and row.layer >= 0:
            return (f"draft{step_suffix}", int(row.layer))
        if row.archetype_id in {"draft_pre", "draft_tail"}:
            return (f"{row.archetype_id}{step_suffix}", -1)
        return None
    if row.layer is not None and row.layer >= 0:
        return (row.model_scope, int(row.layer))
    return None


def _micro_collapsed_site(row: OperatorExportRow) -> str:
    match = _RECURRENT_MICRO_PATTERN.match(row.op_id)
    if match is None and row.archetype_id == "target_tail":
        match = _RECOVER_SSM_MICRO_PATTERN.match(row.op_id)
    if match is None:
        return ""
    collapsed_id = f"{match.group(1)}.{match.group(2)}" if match.group(1) else str(match.group(2))
    return strip_op_prefix(collapsed_id)


def _extract_per_layer_timings(
    rows: Sequence[OperatorExportRow],
    layer_plans: Sequence[Any] = (),
) -> dict[tuple[str, int], dict[str, Any]]:
    """Capture exact native per-layer and per-proposal physical timings for trace replay."""
    block_by_layer = {
        int(lp.index): int(getattr(lp, "residual_block_size", 0) or 0)
        for lp in layer_plans
    }
    multi_draft_step = len(_collect_draft_proposal_steps(rows)) > 1
    by_layer: dict[tuple[str, int], list[OperatorExportRow]] = defaultdict(list)
    for row in rows:
        key = _per_layer_timing_key(row, multi_draft_step)
        if key is not None:
            by_layer[key].append(row)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, l_rows in by_layer.items():
        min_s = min(r.start_us for r in l_rows)
        max_e = max(r.end_us for r in l_rows)
        op_map: dict[str, tuple[float, float]] = {}
        op_exp_map: dict[str, float] = {}
        op_lstar_map: dict[str, float] = {}
        op_dur_ns_map: dict[str, int] = {}
        op_exp_ns_map: dict[str, int] = {}
        op_lstar_ns_map: dict[str, int] = {}
        micro_groups: dict[str, list[OperatorExportRow]] = defaultdict(list)
        for r in l_rows:
            sites = [strip_op_prefix(r.op_id)]
            if r.op_id.startswith("rewrite:") and r.absorbed_source_ops:
                sites.append(
                    "rewrite_fused:" + "+".join(strip_op_prefix(src) for src in r.absorbed_source_ops)
                )
            micro_site = _micro_collapsed_site(r)
            if micro_site:
                micro_groups[micro_site].append(r)
            dur_u = max(0.0, r.dur_us)
            exp_u = 0.0 if r.is_overlapped else max(0.0, r.exposed_dur_us)
            ls_u = 0.0 if r.is_overlapped else max(0.0, r.l_star_lean_us)
            dur_ns_v = int(r.dur_ns) if int(r.dur_ns) > 0 else max(0, int(round(dur_u * NS_PER_US)))
            exp_ns_v = (
                0 if r.is_overlapped
                else (int(r.exposed_ns) if int(r.exposed_ns) > 0 else max(0, int(round(exp_u * NS_PER_US))))
            )
            ls_ns_v = (
                0 if r.is_overlapped
                else (int(r.l_star_ns) if int(r.l_star_ns) > 0 else max(0, int(round(ls_u * NS_PER_US))))
            )
            for site in sites:
                if site not in op_map:
                    op_map[site] = (max(0.0, r.start_us - min_s), dur_u)
                    op_exp_map[site] = exp_u
                    op_lstar_map[site] = ls_u
                    op_dur_ns_map[site] = dur_ns_v
                    op_exp_ns_map[site] = exp_ns_v
                    op_lstar_ns_map[site] = ls_ns_v
        for micro_site, m_rows in micro_groups.items():
            if micro_site not in op_map and m_rows:
                m_start = min(mr.start_us for mr in m_rows)
                m_avg_dur = sum(max(0.0, mr.dur_us) for mr in m_rows) / float(len(m_rows))
                op_map[micro_site] = (max(0.0, m_start - min_s), m_avg_dur)
        out[key] = {
            "start_us": min_s,
            "end_us": max_e,
            "span_us": max(0.0, max_e - min_s),
            "ops": op_map,
            "ops_exposed": op_exp_map,
            "ops_lstar": op_lstar_map,
            "ops_dur_ns": op_dur_ns_map,
            "ops_exposed_ns": op_exp_ns_map,
            "ops_lstar_ns": op_lstar_ns_map,
            "residual_block_size": block_by_layer.get(key[1], 0) if key[0] == "target" else 0,
        }
    return out


def _aggregate_bundle_cohort_rows(
    rows: Sequence[OperatorExportRow],
    demands_ns: Sequence[tuple[int, int, int, int, int]],
    archetypes: Sequence[LayerArchetypeSpec],
    *,
    layer_plans: Sequence[Any] = (),
    total_target_layers: int = 0,
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], tuple[LayerArchetypeSpec, ...]]:
    """Read-only display folding preserving exact integer-ns group totals (never re-certifies)."""
    if not rows:
        return [], [], tuple(archetypes)
    block_by_layer = {
        int(lp.index): int(getattr(lp, "residual_block_size", 0) or 0)
        for lp in layer_plans
    }
    groups: OrderedDict[tuple[str, str], list[tuple[OperatorExportRow, tuple[int, int, int, int, int]]]] = OrderedDict()
    for row, dem in zip(rows, demands_ns):
        if row.op_id.startswith("rewrite:") and row.absorbed_source_ops:
            site = "rewrite_fused:" + "+".join(
                strip_op_prefix(src) for src in row.absorbed_source_ops
            )
        else:
            site = strip_op_prefix(row.op_id)
        groups.setdefault((row.archetype_id, site), []).append((row, dem))

    folded_rows: list[OperatorExportRow] = []
    folded_demands: list[tuple[int, int, int, int, int]] = []
    arch_exposed_ns: dict[str, int] = defaultdict(int)
    for (arch_id, _site), members in groups.items():
        rep = members[0][0]
        cnt = sum(max(1, int(m[0].repeat_count)) for m in members)
        tot_dur_ns = sum(int(m[0].dur_ns) for m in members)
        tot_exp_ns = sum(int(m[0].exposed_ns) for m in members)
        tot_lstar_ns = sum(int(m[0].l_star_ns) for m in members)
        sum_dem = tuple(sum(int(m[1][k]) for m in members) for k in range(CERT_RESOURCE_COUNT))
        inv = 1.0 / float(max(1, cnt))
        avg_dur_us = float(tot_dur_ns) * inv / NS_PER_US
        avg_exp_us = float(tot_exp_ns) * inv / NS_PER_US
        avg_lstar_us = float(tot_lstar_ns) * inv / NS_PER_US
        fully_hidden = tot_dur_ns > 0 and tot_exp_ns == 0
        layer_idxs = [int(m[0].layer) for m in members if m[0].layer is not None]
        durs_us = [float(m[0].dur_us) for m in members if m[0].layer is not None]
        blk_sz = block_by_layer.get(layer_idxs[0], 0) if layer_idxs else 0
        depth_note = format_attnres_depth_note(
            rep.op_id, layer_idxs, durs_us, blk_sz,
            total_target_layers=total_target_layers, family=rep.family,
        )
        role_txt = rep.logical_role
        if depth_note and depth_note not in role_txt:
            role_txt = f"{role_txt} [{depth_note}]" if role_txt else depth_note
        eff = min(1.0, avg_lstar_us / avg_dur_us) if (avg_dur_us > 0.0 and avg_lstar_us > 0.0) else rep.roofline_efficiency
        folded_rows.append(replace(
            rep,
            ordinal=len(folded_rows),
            repeat_count=cnt,
            dur_us=avg_dur_us,
            exposed_dur_us=0.0 if fully_hidden else avg_exp_us,
            dur_ns=tot_dur_ns,
            exposed_ns=tot_exp_ns,
            l_star_ns=tot_lstar_ns,
            l_star_lean_us=avg_lstar_us,
            gap_us=0.0 if fully_hidden else max(0.0, avg_dur_us - avg_lstar_us),
            gap_ratio=None if fully_hidden else ((avg_dur_us / avg_lstar_us) if avg_lstar_us > 0.0 else None),
            roofline_efficiency=eff,
            is_overlapped=fully_hidden,
            flops=sum(float(m[0].flops) for m in members) * inv,
            vector_ops=sum(float(m[0].vector_ops) for m in members) * inv,
            hbm_read_bytes=sum(float(m[0].hbm_read_bytes) for m in members) * inv,
            hbm_write_bytes=sum(float(m[0].hbm_write_bytes) for m in members) * inv,
            hbm_total_bytes=sum(float(m[0].hbm_total_bytes) for m in members) * inv,
            mte2_bytes=sum(float(m[0].mte2_bytes) for m in members) * inv,
            mte3_bytes=sum(float(m[0].mte3_bytes) for m in members) * inv,
            fixpipe_bytes=sum(float(m[0].fixpipe_bytes) for m in members) * inv,
            link_bytes=sum(float(m[0].link_bytes) for m in members) * inv,
            logical_role=role_txt,
        ))
        folded_demands.append(sum_dem)  # type: ignore[arg-type]
        arch_exposed_ns[arch_id] += tot_exp_ns

    updated_archs = tuple(
        replace(
            a,
            single_span_us=float(arch_exposed_ns.get(a.archetype_id, 0)) / (NS_PER_US * float(max(1, a.repeat_count))),
        )
        for a in archetypes
    )
    return folded_rows, folded_demands, updated_archs


def _reconcile_draft_schedule_repeats(
    archetypes: Sequence[LayerArchetypeSpec],
    full_schedule: Sequence[TraceScheduleEntry],
    rows: Sequence[OperatorExportRow],
    *,
    draft_steps: Sequence[int] = (),
) -> tuple[tuple[LayerArchetypeSpec, ...], tuple[TraceScheduleEntry, ...]]:
    """Align multi-step MTP draft archetype repeat counts & per-proposal execution order."""
    if len(draft_steps) > 1:
        active_draft_aids = {r.archetype_id for r in rows if r.model_scope == "draft"}
        non_draft_prefix: list[TraceScheduleEntry] = []
        draft_template: list[TraceScheduleEntry] = []
        non_draft_suffix: list[TraceScheduleEntry] = []
        seen_draft = False
        for entry in full_schedule:
            if entry.scope == "draft":
                seen_draft = True
                if entry.archetype_id in active_draft_aids:
                    draft_template.append(entry)
            elif seen_draft:
                non_draft_suffix.append(entry)
            else:
                non_draft_prefix.append(entry)
        expanded_draft = [
            TraceScheduleEntry(e.archetype_id, e.scope, f"{e.label} [Step {step_i}]", e.full_layer_index)
            for step_i in draft_steps
            for e in draft_template
        ]
        sched_counts = Counter(e.archetype_id for e in expanded_draft)
        updated_archs: list[LayerArchetypeSpec] = []
        for arch in archetypes:
            cnt = sched_counts.get(arch.archetype_id, 0)
            if arch.scope == "draft" and cnt > arch.repeat_count:
                new_title = re.sub(r"\(×\d+\)", f"(×{cnt})", arch.title)
                arch = replace(arch, repeat_count=cnt, title=new_title)
            updated_archs.append(arch)
        return tuple(updated_archs), tuple([*non_draft_prefix, *expanded_draft, *non_draft_suffix])

    max_rep_by_arch: dict[str, int] = {}
    for row in rows:
        aid = row.archetype_id
        max_rep_by_arch[aid] = max(max_rep_by_arch.get(aid, 0), max(1, int(row.repeat_count)))
    updated_archs = []
    expanded_targets: dict[str, int] = {}
    for arch in archetypes:
        if arch.scope == "draft" and arch.archetype_id in {"draft_arch_0", "draft_context_kv"}:
            actual = max_rep_by_arch.get(arch.archetype_id, arch.repeat_count)
            if actual > arch.repeat_count:
                new_title = re.sub(r"\(×\d+\)", f"(×{actual})", arch.title)
                arch = replace(arch, repeat_count=actual, title=new_title)
                expanded_targets[arch.archetype_id] = actual
        updated_archs.append(arch)
    if not expanded_targets:
        return tuple(updated_archs), tuple(full_schedule)
    expanded_sched: list[TraceScheduleEntry] = []
    idx = 0
    sched_list = list(full_schedule)
    while idx < len(sched_list):
        entry = sched_list[idx]
        target_rep = expanded_targets.get(entry.archetype_id, 0)
        if target_rep > 0:
            block = [entry]
            scan = idx + 1
            while scan < len(sched_list) and sched_list[scan].archetype_id == entry.archetype_id:
                block.append(sched_list[scan])
                scan += 1
            if target_rep > len(block) and target_rep % len(block) == 0:
                steps = target_rep // len(block)
                for step_i in range(steps):
                    for blk_e in block:
                        lbl = f"{blk_e.label} [Step {step_i}]" if steps > 1 else blk_e.label
                        expanded_sched.append(TraceScheduleEntry(blk_e.archetype_id, blk_e.scope, lbl, blk_e.full_layer_index))
                idx = scan
                continue
        expanded_sched.append(entry)
        idx += 1
    return tuple(updated_archs), tuple(expanded_sched)


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
    default_dt = str(peaks.get("default_dtype") or "").strip()
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


def _extract_work_items(
    work_items: Sequence[Any],
) -> tuple[float, tuple[tuple[str, float], ...], float, float, float, float, float]:
    """Read exact resource work totals from ``OperatorSequenceRow.work``."""
    flops = vec_ops = hbm_tot = mte2 = mte3 = link_bytes = 0.0
    m_flops: list[tuple[str, float]] = []
    for item in work_items:
        res = str(getattr(item, "resource", "") or "").strip().lower()
        amt = float(getattr(item, "amount", 0.0) or 0.0)
        if amt <= 0.0:
            continue
        if res == "cube":
            flops += amt
            dt = str(getattr(item, "dtype", "") or "").strip()
            if dt:
                m_flops.append((dt, amt))
        elif res == "vector":
            vec_ops += amt
        elif res == "hbm":
            hbm_tot += amt
        elif res == "mte2":
            mte2 += amt
        elif res == "mte3":
            mte3 += amt
        elif res == "link":
            link_bytes += amt
    return flops, tuple(m_flops), vec_ops, hbm_tot, mte2, mte3, link_bytes


def _inherit_fused_submodule(
    absorbed: Sequence[str],
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
    struct_by_op: Mapping[str, Mapping[str, Any]],
    lp: Any | None,
    arch_scope: str,
) -> str:
    """Resolve fused op submodule read-only from its absorbed source operators."""
    fallback = ""
    for src in absorbed:
        s_spec = op_spec_by_key.get((0, src))
        s_node = struct_by_op.get(src)
        s_rec = {
            "op_id": src,
            "family": s_spec.family if s_spec is not None else "",
            "recipe": s_spec.recipe if s_spec is not None else "",
        }
        cand = _resolve_op_submodule(s_rec, s_node, lp, arch_scope)
        if not fallback:
            fallback = cand
        if cand != SUB_ATTN_RES:
            return cand
    return fallback


def _pipeline_map_from_eval(step_eval: StepEvaluation | None) -> dict[str, FusedPipelineDescriptor]:
    if step_eval is None:
        return {}
    return {
        p.owner.op_id: p
        for p in (getattr(step_eval, "pipelines", ()) or ())
        if p.owner.replica == 0 and p.absorbed_source_op_ids
    }


def _resolve_row_work_metrics(
    rec: Mapping[str, Any],
    spec: OpSpec | None,
    prep_map: Mapping[str, Mapping[str, float]],
) -> tuple[float, Sequence[tuple[str, float]], float, float, float, float, float, float, float, float, float]:
    """Resolve physical work totals from ``OpSpec``, prepared map, or ``OperatorSequenceRow.work``."""
    flops = float((spec.flops if spec is not None else 0.0) or 0.0)
    matrix_flops: Sequence[tuple[str, float]] = (
        spec.matrix_flops if (spec is not None and spec.matrix_flops) else ()
    )
    vec_ops = float((spec.vector_ops if spec is not None else 0.0) or 0.0)
    link_bytes = float((spec.link_bytes if spec is not None else 0.0) or 0.0)
    hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit = _hbm_read_write_l2(spec)
    if flops == 0.0 and vec_ops == 0.0 and hbm_tot == 0.0 and link_bytes == 0.0:
        pw = prep_map.get(rec["op_id"])
        if pw is not None:
            flops = float(pw.get("flops", 0.0))
            if not matrix_flops and pw.get("matrix_flops"):
                matrix_flops = tuple(pw["matrix_flops"])
            vec_ops = float(pw.get("vector_ops", 0.0))
            hbm_tot = float(pw.get("hbm_bytes", 0.0))
            mte2 = float(pw.get("mte2_bytes", 0.0))
            mte3 = float(pw.get("mte3_bytes", 0.0))
            link_bytes = float(pw.get("link_bytes", 0.0))
        elif rec.get("work"):
            flops, matrix_flops, vec_ops, hbm_tot, mte2, mte3, link_bytes = _extract_work_items(rec["work"])
        copy_sum = mte2 + mte3
        hbm_r = (hbm_tot * mte2 / copy_sum) if copy_sum > 0.0 else (hbm_tot * _HALF_SPLIT_SHARE)
        hbm_w = max(0.0, hbm_tot - hbm_r)
    return flops, matrix_flops, vec_ops, link_bytes, hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit


def _native_binding_from_pipeline(pipe: FusedPipelineDescriptor | None) -> dict[str, Any] | None:
    """Extract verified native kernel boundary metadata from a finalized ``FusedPipelineDescriptor``."""
    if pipe is None:
        return None
    impl = str(pipe.implementation or "").strip()
    if not impl or impl.startswith((_REWRITE_FORMULA_PREFIX, "rewrite_")):
        return None
    src = pipe.sources[0] if getattr(pipe, "sources", None) else None
    low = impl.lower()
    if low.startswith("aclnn"):
        return {
            "symbol": impl, "backend": "cann_aclnn", "readiness": "shipping",
            "source": src, "preserve_symbol": True,
        }
    if low.startswith("hccl"):
        return {
            "symbol": impl, "backend": "hccl_mc2", "readiness": "shipping",
            "source": src, "preserve_symbol": True,
        }
    if low.startswith("catlass."):
        return {"symbol": impl, "backend": "catlass", "source": src, "preserve_symbol": True}
    if low.startswith("dsl."):
        return {"symbol": impl, "backend": "tilelang", "source": src, "preserve_symbol": True}
    if ":" not in impl:
        return {"symbol": impl, "source": src, "preserve_symbol": True}
    return None


def _resolve_row_fusion_binding(
    rec: Mapping[str, Any],
    desc: Any,
    pipe: FusedPipelineDescriptor | None,
    meta: Mapping[str, Any],
    absorbed: Sequence[str],
) -> tuple[str, str, str, bool, str, bool, str]:
    """Project read-only binding state and kernel identity for fused rewrite pipelines."""
    pipe_impl = str(pipe.implementation or "") if pipe is not None else ""
    is_unbound_fused = bool(
        pipe_impl.startswith(_REWRITE_FORMULA_PREFIX)
        or (bool(meta.get("target_rewrite_fused")) and not desc.is_shipping_abi)
    )
    is_unbound_ext = bool(
        meta.get("abi_variant_extension")
        and desc.needs_new_kernel
        and not desc.is_shipping_abi
    )
    op_binding = str(meta.get("binding_state") or "").strip() or (
        BINDING_STATE_UNBOUND_FORMULA if (is_unbound_fused or is_unbound_ext) else BINDING_STATE_BOUND
    )
    ascend_kernel = desc.ascend_kernel
    kernel_status = desc.kernel_status
    needs_new_kernel = desc.needs_new_kernel
    new_kernel_spec = desc.new_kernel_spec
    is_shipping_abi = bool(desc.is_shipping_abi)
    opt_pillar = ""
    if is_unbound_fused:
        if pipe_impl.startswith(_REWRITE_FORMULA_PREFIX):
            parts = pipe_impl.split(":")
            cand = parts[1] if len(parts) >= _REWRITE_FORMULA_MIN_PARTS else parts[-1]
            ascend_kernel = cand or ascend_kernel
        elif ":" in pipe_impl:
            ascend_kernel = pipe_impl.rsplit(":", 1)[-1] or ascend_kernel
        kernel_status = KERNEL_STATUS_NEW_OP
        needs_new_kernel = True
        is_shipping_abi = False
        opt_pillar = "pillar1_fusion"
        if not new_kernel_spec and absorbed:
            new_kernel_spec = f"Fused {len(absorbed)} ops: {', '.join(strip_op_prefix(s) for s in absorbed)}"
    elif pipe is not None and rec["family"] == "rewrite_fused":
        opt_pillar = "pillar1_fusion"
    return op_binding, ascend_kernel, kernel_status, needs_new_kernel, new_kernel_spec, is_shipping_abi, opt_pillar


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
    tail_closure: frozenset[OpOccurrence] = frozenset(),
    pipeline_by_op: Mapping[str, FusedPipelineDescriptor] = (),
    op_spec_by_key: Mapping[tuple[int, str], OpSpec] = (),
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[int, dict[str, Any]]]:
    archetypes_by_id = {a.archetype_id: a for a in archetypes}
    target_arch_by_rep_layer = {
        full_idx: a for a in archetypes if a.scope == "target" for full_idx in a.layer_indices
    }
    peaks_by_stage: dict[int, dict[str, Any]] = {}
    prelim_rows: list[OperatorExportRow] = []
    demands_ns: list[tuple[int, int, int, int, int]] = []
    prep_map = prepared_work_by_op or {}
    pipe_map = dict(pipeline_by_op)
    del use_archetype_dedup
    for rec in filtered_records:
        occ: OpOccurrence = rec["occurrence"]
        spec: OpSpec | None = rec["spec"]
        pipe = pipe_map.get(rec["op_id"])
        meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
        absorbed = tuple(dict.fromkeys((
            *(str(x) for x in (pipe.absorbed_source_op_ids if pipe is not None else ())),
            *(str(x) for x in (meta.get("absorbed_source_ops") or meta.get("fused_from") or ())),
        )))
        if occ.stage not in peaks_by_stage:
            peaks_by_stage[occ.stage] = _die_peaks_for_stage(eff_case, occ.stage, 0)
        peaks = peaks_by_stage[occ.stage]
        arch = _assign_op_archetype(
            rec["op_id"], spec, archetypes_by_id, target_arch_by_rep_layer,
            occurrence=occ, tail_closure=tail_closure, absorbed_source_ops=absorbed,
        )
        lp = rep_lp_by_idx.get(arch.rep_layer_index) if (arch.scope == "target" and arch.rep_layer_index is not None) else None
        s_node = struct_by_op.get(rec["op_id"])
        sg_info = subgraph_by_op.get(rec["op_id"])
        layer_idx = extract_layer_index(rec["op_id"], spec.layer if spec is not None else None)
        if layer_idx is None and absorbed:
            for src in absorbed:
                src_spec = op_spec_by_key.get((0, src))
                layer_idx = extract_layer_index(src, src_spec.layer if src_spec is not None else None)
                if layer_idx is not None:
                    break
        submodule = (
            _inherit_fused_submodule(absorbed, op_spec_by_key, struct_by_op, lp, arch.scope)
            if (spec is None and absorbed)
            else _resolve_op_submodule(rec, s_node, lp, arch.scope)
        )
        stream_idx = classify_stream_index(rec["stream"], rec["family"], rec["op_id"])
        dur_us = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        flops, matrix_flops, vec_ops, link_bytes, hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit = (
            _resolve_row_work_metrics(rec, spec, prep_map)
        )
        op_demand, cube_peak = _op_lean_resource_demands_ns(
            flops=flops, vector_ops=vec_ops, hbm_read=hbm_r, hbm_write=hbm_w,
            link_bytes=link_bytes, dtype=rec["dtype"], peaks=peaks,
            matrix_flops=matrix_flops,
        )
        l_star_us = float(max(op_demand)) / NS_PER_US
        dom_bound, roof_eff = _classify_dominant_bound(op_demand, dur_us, link_bytes)
        desc = resolve_ascend_kernel_identity(
            rec["op_id"], spec, rec["family"], rec["recipe"], rec["shape"], rec["dtype"],
            eff_case, lp, sources_by_op.get(rec["op_id"], ()),
            struct_node=s_node, subgraph_info=sg_info,
            native_binding=_native_binding_from_pipeline(pipe) if spec is None else None,
        )
        sem_atoms = extract_semantic_atoms(
            rec["op_id"], rec["family"], rec["recipe"], absorbed, metadata=meta,
        )
        comm_dom = resolve_comm_domain_tag(rec["op_id"], rec["family"], rec["recipe"], meta, eff_case)
        op_binding, ascend_kernel, kernel_status, needs_new_kernel, new_kernel_spec, is_shipping_abi, opt_pillar = (
            _resolve_row_fusion_binding(rec, desc, pipe, meta, absorbed)
        )
        demands_ns.append(op_demand)
        row_dur_ns = max(0, int(round(dur_us * NS_PER_US)))
        row_l_star_ns = max(0, int(max(op_demand)))
        row_extras = _filter_dataclass_kwargs(
            _OP_ROW_FIELDS,
            {
                "is_shipping_abi": is_shipping_abi,
                "dur_ns": row_dur_ns,
                "exposed_ns": row_dur_ns,
                "l_star_ns": row_l_star_ns,
                "opt_pillar": opt_pillar,
            },
        )
        prelim_rows.append(OperatorExportRow(
            ordinal=len(prelim_rows), occurrence=occ, op_id=rec["op_id"],
            layer=layer_idx, layer_label=arch.title, submodule=submodule,
            family=rec["family"], recipe=rec["recipe"], implementation=ascend_kernel,
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
            model_scope=arch.scope, repeat_count=1,
            logical_role=desc.logical_role, ascend_kernel=ascend_kernel,
            kernel_backend=desc.kernel_backend, kernel_status=kernel_status,
            source_ref=desc.source_ref, tensor_contract=desc.tensor_contract,
            exposed_dur_us=dur_us, cube_peak_flops_per_s=cube_peak,
            bound_display=BOUND_DISPLAY_LABELS.get(dom_bound, dom_bound),
            needs_new_kernel=needs_new_kernel, new_kernel_spec=new_kernel_spec,
            canonical_op_key=desc.canonical_op_key,
            semantic_atoms=sem_atoms, comm_domain_tag=comm_dom,
            binding_state=op_binding,
            **row_extras,
        ))
    return prelim_rows, demands_ns, peaks_by_stage


def _resolve_deployment_mode(parsed: Case, phase: str) -> str:
    meta = dict(parsed.runtime.metadata)
    spine = meta.get("single_spine") or {}
    explicit = str(
        meta.get("deployment_mode")
        or spine.get("deployment_mode")
        or meta.get("deployment_role")
        or meta.get("deployment")
        or ""
    ).strip()
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
    # Only a decode query width implies draft proposals; a prefill prompt width never does.
    inferred = max(0, int(step_req.tokens) - 1) if step_req.phase == "decode" else 0
    proposals = spec_tokens if spec_tokens > 0 else inferred
    if step_req.phase == "decode":
        q_width, avg_accepted = _resolve_speculative_tokens(
            parsed, output_budget=output_toks, tokens_per_seq=int(step_req.tokens),
            emitted_tokens_per_seq=None,
        )
        proposals = min(proposals, q_width - 1)
    else:
        avg_accepted = float(max(1, int(step_req.tokens)))
    if proposals > 0:
        if draft_mode == "dspark" or (aux is not None and getattr(aux, "dspark", None) is not None):
            spec_method = "DSpark"
        else:
            spec_method = "MTP"
    else:
        spec_method = "None"

    exec_pol = meta.get("execution_policy") if isinstance(meta.get("execution_policy"), Mapping) else {}
    sp_enabled = bool(exec_pol.get("sp", False)) or any(
        "sp_gather" in r.op_id or "sp_shard" in r.op_id for r in final_rows
    )
    raw_tpot = getattr(parsed.slo, "tpot_ms", None)
    tpot_slo = float(raw_tpot) if raw_tpot is not None else None
    raw_ttft = getattr(parsed.slo, "ttft_ms", None)
    ttft_slo = float(raw_ttft) if raw_ttft is not None else None
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


def _normalize_bundle_sequences(
    sequences: int | Sequence[int], dp: int,
) -> tuple[int, int]:
    """Return ``(per_replica_peak_concurrency, cluster_concurrency)`` via ``cohort_sequence_counts``."""
    dp_n = max(1, int(dp))
    if isinstance(sequences, int):
        per_dp = max(1, int(sequences))
        return per_dp, max(1, per_dp * dp_n)
    counts = cohort_sequence_counts(sequences, range(dp_n))
    return max(1, max(counts.values(), default=1)), max(1, sum(counts.values()))


def _occurrence_stage_label(occ: OpOccurrence) -> str:
    rank = -1 if occ.rank is None else int(occ.rank)
    return f"r{int(occ.replica)}.p{int(occ.stage)}.d{rank}:{occ.op_id}"


def _build_stage_waterfall_entries(receipt: EngineStepReceipt) -> tuple[dict[str, Any], ...]:
    """Project canonical 4-stage ``EngineStepReceipt`` into exact int-ns telescoping entries."""
    entries: list[dict[str, Any]] = []
    prev_span_ns: int | None = None
    for idx, stage in enumerate(receipt.stages):
        raw_span_ns = int(stage.span_ns)
        if idx == 0 or stage.status == "committed":
            eff_span_ns = raw_span_ns
        else:
            eff_span_ns = int(prev_span_ns) if prev_span_ns is not None else raw_span_ns
        delta_ns = 0 if prev_span_ns is None else int(prev_span_ns - eff_span_ns)
        prev_span_ns = eff_span_ns
        entries.append({
            "stage_id": stage.stage_id,
            "stack_track": stage.stack_track,
            "status": stage.status,
            "span_ns": eff_span_ns,
            "delta_ns": delta_ns,
            "span_ms": float(eff_span_ns) / float(NS_PER_MS),
            "delta_ms": float(delta_ns) / float(NS_PER_MS),
            "l_star_ns": int(stage.l_star_ns),
            "l_star_valid": bool(stage.l_star_valid),
            "idle_or_wait_ns": int(stage.idle_or_wait_ns),
            "plan_sha256": stage.plan_sha256,
            "execution_key_sha256": stage.execution_key_sha256,
            "rejection_reason": stage.rejection_reason,
        })
    return tuple(entries)


def _extract_applied_edit_receipts(
    ext_plan: Any,
    *,
    verified: bool,
    pipelines: Sequence[FusedPipelineDescriptor] = (),
) -> tuple[dict[str, Any], ...]:
    """Serialize verified ``RewritePlan`` edits and fused pipelines into receipt dicts."""
    if not verified:
        return ()
    receipts: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    if ext_plan is not None:
        for edit in getattr(ext_plan, "edits", ()) or ():
            body = getattr(edit, "operation", None) or getattr(edit, "body", None)
            opcode = type(body).__name__ if body is not None else ""
            edit_id = str(getattr(edit, "edit_id", "") or "")
            rec: dict[str, Any] = {
                "edit_id": edit_id,
                "opcode": opcode,
                "status": "applied",
                "verified": True,
            }
            if hasattr(body, "members"):
                rec["members"] = [str(getattr(m, "id", m)) for m in (getattr(body, "members", ()) or ())]
            if hasattr(body, "fusion_tier"):
                rec["fusion_tier"] = str(getattr(body, "fusion_tier", "") or "")
            if hasattr(body, "kernel_boundaries"):
                rec["abi_ids"] = [
                    str(getattr(kb, "abi_id", "") or "")
                    for kb in (getattr(body, "kernel_boundaries", ()) or ())
                    if getattr(kb, "abi_id", "")
                ]
            if edit_id:
                seen_outputs.add(f"rewrite:{edit_id}")
            receipts.append(rec)
    for pipe in pipelines:
        if pipe.owner.replica != 0 or len(pipe.absorbed_source_op_ids) < _MIN_FUSED_PIPELINE_MEMBERS:
            continue
        oid = str(pipe.owner.op_id)
        if oid in seen_outputs:
            continue
        seen_outputs.add(oid)
        receipts.append({
            "edit_id": oid,
            "opcode": "FuseRegion",
            "status": "applied",
            "verified": True,
            "output_occurrences": [oid],
            "members": [str(m) for m in pipe.absorbed_source_op_ids],
        })
    return tuple(receipts)


def _apply_stage_exposed_ns(
    rows: Sequence[OperatorExportRow],
    archetypes: Sequence[LayerArchetypeSpec],
    stage: StageReceipt,
) -> tuple[list[OperatorExportRow], tuple[LayerArchetypeSpec, ...], int, int]:
    """Apply exact integer-ns exposure from a canonical ``StageReceipt`` onto unfolded rows."""
    occ_exp = {str(k): max(0, int(v)) for k, v in stage.exposed_ns_by_occ}
    spans_ns: dict[str, int] = defaultdict(int)
    out_rows: list[OperatorExportRow] = []
    total_exp_ns = 0
    for row in rows:
        lbl = _occurrence_stage_label(row.occurrence)
        dur_ns = max(0, int(row.dur_ns) if int(row.dur_ns) > 0 else int(round(float(row.dur_us) * NS_PER_US)))
        exp_ns = occ_exp.get(lbl, int(row.exposed_ns))
        if dur_ns > 0:
            exp_ns = min(dur_ns, max(0, exp_ns))
        total_exp_ns += exp_ns
        fully_hidden = dur_ns > 0 and exp_ns == 0
        out_rows.append(replace(
            row,
            dur_ns=dur_ns,
            exposed_ns=exp_ns,
            exposed_dur_us=float(exp_ns) / NS_PER_US,
            is_overlapped=fully_hidden,
            gap_us=0.0 if fully_hidden else max(0.0, row.dur_us - row.l_star_lean_us),
            gap_ratio=None if fully_hidden else ((row.dur_us / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None),
        ))
        spans_ns[row.archetype_id] += exp_ns
    step_span_ns = max(total_exp_ns, int(stage.span_ns))
    idle_or_wait_ns = max(0, step_span_ns - total_exp_ns)
    updated_archs = tuple(
        replace(
            a,
            single_span_us=float(spans_ns.get(a.archetype_id, 0)) / (NS_PER_US * float(max(1, a.repeat_count))),
        )
        for a in archetypes
    )
    return out_rows, updated_archs, step_span_ns, idle_or_wait_ns


def _select_final_receipt_stage(receipt: EngineStepReceipt) -> tuple[Any, Any, bool]:
    """Select the last committed waterfall stage and whether Stage 3 reverted to baseline."""
    stage0 = receipt.stage_for("before_merged")
    stage1 = receipt.stage_for("stage1_ops_config")
    stage2 = receipt.stage_for("stage2_unmerged_pr")
    stage3 = receipt.stage_for("stage3_target_after")
    if stage3 is not None and stage3.status == "committed":
        final_stage = stage3
    elif stage2 is not None and stage2.status == "committed":
        final_stage = stage2
    elif stage1 is not None and stage1.status == "committed":
        final_stage = stage1
    elif stage0 is not None:
        final_stage = stage0
    else:
        final_stage = receipt.stages[-1]
    reverted = bool(
        stage3 is None
        or stage3.status != "committed"
        or (stage0 is not None and stage3.plan_sha256 == stage0.plan_sha256)
    )
    return stage0, final_stage, reverted


def _project_stage_step_eval_rows(
    eff_spine: Any,
    parsed: Case,
    step_req: StepRequest,
    final_stage: Any,
    *,
    archetypes: Sequence[LayerArchetypeSpec],
    rep_lp_by_idx: Mapping[int, Any],
    rebound_primitive_table: Any,
    streams: tuple[Any, ...],
) -> tuple[
    list[OperatorExportRow],
    list[tuple[int, int, int, int, int]],
    tuple[LayerArchetypeSpec, ...],
    int,
    int,
    dict[tuple[int, str], OpSpec],
    Case,
    dict[int, dict[str, Any]],
    str,
    tuple[FusedPipelineDescriptor, ...],
    Any,
]:
    (
        st_case, st_raw_records, st_spec_map, st_sources,
        st_struct, st_subgraph, st_gid, st_pipes, st_pt,
    ) = _resolve_bundle_step_eval(
        eff_spine, parsed, step_req, final_stage.step_eval,
        primitive_table=rebound_primitive_table, streams=streams,
    )
    st_filtered, st_tail = _prepare_bundle_records(
        st_raw_records, archetypes=archetypes, use_archetype_dedup=False,
    )
    st_rows, st_demands, st_peaks = _build_prelim_operator_rows(
        st_filtered, eff_case=st_case, archetypes=archetypes, rep_lp_by_idx=rep_lp_by_idx,
        use_archetype_dedup=False, sources_by_op=st_sources,
        struct_by_op=st_struct, subgraph_by_op=st_subgraph,
        prepared_work_by_op=None, tail_closure=st_tail,
        pipeline_by_op=_pipeline_map_from_eval(final_stage.step_eval),
        op_spec_by_key=st_spec_map,
    )
    st_rows, st_archs, _, _ = _attribute_full_bundle_stream0(st_rows, archetypes)
    out_rows, out_archs, out_span_ns, out_idle_ns = _apply_stage_exposed_ns(
        st_rows, st_archs, final_stage,
    )
    return (
        out_rows, st_demands, out_archs, out_span_ns, out_idle_ns,
        st_spec_map, st_case, st_peaks, st_gid, st_pipes, st_pt,
    )


def _apply_target_envelope_to_rows(
    prelim_rows: list[OperatorExportRow],
    demands_ns: list[tuple[int, int, int, int, int]],
    archetypes: Sequence[LayerArchetypeSpec],
    *,
    parsed: Case,
    eff_case: Case,
    step_req: StepRequest,
    cluster_conc: int,
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
    peaks_by_stage: Mapping[int, dict[str, Any]],
    ext_plan: Any,
    rebound_primitive_table: Any,
    allow_unbound_formula: bool,
    spine: Any = None,
    replay: ReplayOptions = ReplayOptions(),
    rep_lp_by_idx: Mapping[int, Any] = (),
    streams: tuple[Any, ...] = (),
    target_max_trials: int = DEFAULT_EXPORT_TARGET_TRIALS,
    deferred_prelim: bool = False,
    use_archetype_dedup: bool = False,
) -> tuple[Any, ...]:
    """Consume canonical 4-stage ``EngineStepReceipt`` from ``W3a`` ladder."""
    del cluster_conc
    eff_spine = spine if spine is not None else default_spine()
    pt_arg = None if rebound_primitive_table is False else rebound_primitive_table
    receipt = solve_target_step_receipt(
        eff_case,
        step_req,
        spine=None,
        replay=replay,
        primitive_table=pt_arg,
        stack_track="target",
        comparison_policy="monotonic_optimization",
        rewrite_plan=ext_plan,
        max_trials=max(1, int(target_max_trials)),
        allow_unbound_formula=bool(allow_unbound_formula),
    )
    stage0, final_stage, reverted_to_native_baseline = _select_final_receipt_stage(receipt)
    stage_waterfall = _build_stage_waterfall_entries(receipt)
    plan_digest = str(final_stage.plan_sha256 or "")
    plan_verified = bool(final_stage.status == "committed" and final_stage.l_star_valid)
    stage_pipelines = tuple(getattr(final_stage.step_eval, "pipelines", ()) or ()) if final_stage.step_eval is not None else ()
    applied_receipts = _extract_applied_edit_receipts(
        None if reverted_to_native_baseline else ext_plan,
        verified=plan_verified,
        pipelines=stage_pipelines,
    )

    out_rows = list(prelim_rows)
    out_demands = list(demands_ns)
    out_archs = tuple(archetypes)
    out_spec_map: dict[tuple[int, str], OpSpec] = dict(op_spec_by_key)
    out_case = eff_case
    out_peaks: dict[int, dict[str, Any]] = dict(peaks_by_stage)
    out_gid = ""
    out_pipes: tuple[FusedPipelineDescriptor, ...] = ()
    out_pt = rebound_primitive_table
    out_span_ns = int(final_stage.span_ns)
    out_idle_ns = int(final_stage.idle_or_wait_ns)
    out_l_star_ns = int(final_stage.l_star_ns)
    out_l_star_valid = bool(final_stage.l_star_valid)

    has_improved_eval = (
        final_stage.step_eval is not None
        and (stage0 is None or final_stage.plan_sha256 != stage0.plan_sha256 or final_stage.span_ns < stage0.span_ns)
    )
    if (has_improved_eval or deferred_prelim) and final_stage.step_eval is not None:
        (
            out_rows, out_demands, out_archs, out_span_ns, out_idle_ns,
            out_spec_map, st_case, st_peaks, st_gid, st_pipes, st_pt,
        ) = _project_stage_step_eval_rows(
            eff_spine, parsed, step_req, final_stage,
            archetypes=archetypes, rep_lp_by_idx=rep_lp_by_idx,
            rebound_primitive_table=rebound_primitive_table, streams=streams,
        )
        if deferred_prelim:
            out_case, out_peaks, out_gid, out_pipes, out_pt = (
                st_case, st_peaks, st_gid, st_pipes, st_pt,
            )
    else:
        if deferred_prelim and not out_rows:
            (
                out_case, out_rows, out_demands, out_peaks, out_archs_list,
                out_spec_map, out_gid, out_pipes, out_pt, _, _,
            ) = _evaluate_bundle_prelim_rows(
                eff_spine, parsed, step_req, None, replay,
                archetypes=list(archetypes), rep_lp_by_idx=rep_lp_by_idx,
                use_archetype_dedup=use_archetype_dedup,
                primitive_table=rebound_primitive_table, streams=streams,
            )
            out_archs = tuple(out_archs_list)
        out_rows, out_archs, out_span_ns, out_idle_ns = _apply_stage_exposed_ns(
            out_rows, out_archs, final_stage,
        )

    co_opps = collect_codesign_opportunities(
        out_case,
        step_request=step_req,
        step_receipt=receipt,
        primitive_table=(None if out_pt is False else out_pt),
    )
    return (
        out_rows, out_demands, out_archs, (), (), plan_digest, plan_verified,
        applied_receipts, "", {}, co_opps, stage_waterfall, receipt,
        reverted_to_native_baseline, out_span_ns, out_idle_ns,
        out_l_star_ns, out_l_star_valid, out_spec_map, out_case,
        out_peaks, out_gid, out_pipes, out_pt,
    )


def _extract_cohort_structure_view(
    eff_case: Case,
    exec_graphs: Mapping[int, Sequence[OpSpec]],
    step_req: StepRequest,
    *,
    semantic_graphs: Mapping[int, Sequence[OpSpec]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract structure provenance over representative cohort layers and expand to peers."""
    if 0 not in exec_graphs:
        return {}, {}
    ops0 = tuple(exec_graphs[0])
    refined = refine_layer_cohort_plan(eff_case, ops0)
    by_layer: dict[int, list[OpSpec]] = defaultdict(list)
    for op in ops0:
        if _is_target_body_op(op):
            by_layer[int(op.layer)].append(op)
    op_rep_map: dict[str, str] = {}
    skipped_ids: set[str] = set()
    kept_layers: set[int] = set()
    for cohort in refined.cohorts:
        rep_lyr = int(cohort.representative_layer)
        kept_layers.add(rep_lyr)
        rep_ops = by_layer.get(rep_lyr, ())
        rep_skel = tuple((op.family, op.recipe, op.stream) for op in rep_ops)
        for lyr in cohort.layer_indices:
            lyr_i = int(lyr)
            if lyr_i == rep_lyr:
                continue
            l_ops = by_layer.get(lyr_i, ())
            if tuple((op.family, op.recipe, op.stream) for op in l_ops) == rep_skel:
                for op_r, op_l in zip(rep_ops, l_ops):
                    op_rep_map[op_l.op_id] = op_r.op_id
                    skipped_ids.add(op_l.op_id)
            else:
                kept_layers.add(lyr_i)
    if not skipped_ids:
        return _extract_structure_registry_view(
            eff_case, exec_graphs, step_req, semantic_graphs=semantic_graphs,
        )
    r0_filtered = tuple(op for op in ops0 if op.op_id not in skipped_ids)
    sem_filtered: Mapping[int, Sequence[OpSpec]] | None = None
    if semantic_graphs is not None and 0 in semantic_graphs:
        sem_filtered = {
            0: tuple(
                op for op in semantic_graphs[0]
                if int(op.layer) < 0 or int(op.layer) in kept_layers
                or op.op_id.startswith(("draft", "final."))
            )
        }
    struct_by_op, subgraph_by_op = _extract_structure_registry_view(
        eff_case, {0: r0_filtered}, step_req, semantic_graphs=sem_filtered,
    )
    for oid, rep_oid in op_rep_map.items():
        if rep_oid in struct_by_op and oid not in struct_by_op:
            struct_by_op[oid] = struct_by_op[rep_oid]
        if rep_oid in subgraph_by_op and oid not in subgraph_by_op:
            subgraph_by_op[oid] = subgraph_by_op[rep_oid]
    return struct_by_op, subgraph_by_op


def _resolve_bundle_step_eval(
    spine: Any,
    eval_case: Case,
    step_req: StepRequest,
    step_eval: StepEvaluation,
    *,
    primitive_table: Any,
    streams: tuple[Any, ...],
) -> tuple[
    Case, list[dict[str, Any]], dict[tuple[int, str], OpSpec],
    dict[str, Sequence[Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]],
    str, tuple[FusedPipelineDescriptor, ...], Any,
]:
    eff_streams = tuple(streams) if streams else tuple(getattr(step_eval, "streams", None) or ())
    eff_pt_arg = primitive_table if primitive_table is not None else getattr(step_eval, "primitive_table", None)
    if type(spine) is PredictionSpine:
        base = spine.stacks.resolve_base(eval_case, LoweringRequest(step=step_req))
        rewritten = spine.rewrite.apply(base, streams=eff_streams)
        eff_case = (
            consume_target_declaration(base.case, rewritten.target_graphs)
            if rewritten.target_graphs is not None else base.case
        )
        book = ReplayRecipeBook.from_mapping(eff_case.runtime.kernels, family_registry=spine.families)
        book.graph = rewritten
        r0_entries = tuple(entry for entry in rewritten.executable if int(entry.replica) == 0)
        if r0_entries:
            r0_streams = tuple(plan for plan in rewritten.streams if int(plan.rank) == 0)
            r0_owners = spine._bind_stream_plans(eff_case, r0_streams, frozenset({0}))
            r0_resolved = spine._resolve_governed_operators(
                eff_case, replace(rewritten, executable=r0_entries), book, r0_owners,
            )
            sources_by_op: dict[str, Sequence[Any]] = {r.occurrence.op_id: r.sources for r in r0_resolved}
        else:
            sources_by_op = {}
    else:
        prepared = spine.lower(eval_case, LoweringRequest(step=step_req), streams=eff_streams)
        rewritten = prepared.graph
        eff_case = prepared.case
        book = ReplayRecipeBook.from_mapping(eff_case.runtime.kernels, family_registry=spine.families)
        book.graph = rewritten
        sources_by_op = {
            r.occurrence.op_id: r.sources for r in prepared.operators if r.occurrence.replica == 0
        }
    eval_ctx = EvaluationContext.create(
        eff_case, book, None,
        primitive_table=(False if eff_pt_arg is False else eff_pt_arg),
    )
    if eff_pt_arg is False:
        rebound_pt: Any = False
    elif eff_pt_arg is not None:
        rebound_pt = eff_pt_arg
    else:
        rebound_pt = eval_ctx.bindings.primitive_table
    op_spec_by_key = _resolve_prepared_op_specs(
        rewritten.executable, eval_ctx,
        case=eff_case, book=book,
        request_views=getattr(rewritten, "request_views", ()),
        padded_tokens=step_req.padded_tokens,
        sequence_layout=step_req.layout,
    )
    exec_graphs = {e.replica: e.ops for e in rewritten.executable}
    struct_by_op, subgraph_by_op = _extract_cohort_structure_view(
        eff_case, exec_graphs, step_req, semantic_graphs=rewritten.target_graphs,
    )
    raw_records = _timeline_to_raw_records(
        step_eval.operators, is_step_eval=True, case=eff_case, op_spec_by_key=op_spec_by_key,
    )
    return (
        eff_case, raw_records, op_spec_by_key, sources_by_op,
        struct_by_op, subgraph_by_op, step_eval.graph_id, step_eval.pipelines, rebound_pt,
    )


def _prepare_bundle_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    archetypes: Sequence[LayerArchetypeSpec],
    use_archetype_dedup: bool,
) -> tuple[list[dict[str, Any]], frozenset[OpOccurrence]]:
    del archetypes, use_archetype_dedup
    tail_closure = compute_acceptance_tail_closure(raw_records)
    filtered = _filter_barriers_and_rewire_preds(
        raw_records, protected_occurrences=tail_closure,
    )
    return filtered, tail_closure


def _apply_display_dtypes(
    rows: Sequence[OperatorExportRow],
    *,
    eff_case: Case,
    archetypes: Sequence[LayerArchetypeSpec],
    rep_lp_by_idx: Mapping[int, Any],
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
) -> list[OperatorExportRow]:
    """Surface precision ABI display tags (e.g. ``mxfp4``) after Engine & target synthesis."""
    arch_by_id = {a.archetype_id: a for a in archetypes}
    out: list[OperatorExportRow] = []
    for row in rows:
        arch = arch_by_id.get(row.archetype_id)
        lp = (
            rep_lp_by_idx.get(arch.rep_layer_index)
            if (arch is not None and arch.scope == "target" and arch.rep_layer_index is not None)
            else None
        )
        spec = op_spec_by_key.get((0, row.op_id))
        disp_dt = resolve_operator_display_dtype(
            row.op_id, spec, row.family, row.recipe, row.dtype, eff_case, lp,
            absorbed_source_ops=row.absorbed_source_ops,
            op_spec_by_key=op_spec_by_key,
        )
        out.append(replace(row, dtype=disp_dt) if disp_dt != row.dtype else row)
    return out


def _resolve_bundle_step_context(
    case: CaseInput,
    *,
    request: StepRequest | None,
    step_eval: StepEvaluation | None,
    apply_target_envelope: bool | None,
    rewrite_plan: Any,
) -> tuple[Case, str, StepRequest, int, int, Any, bool, bool]:
    parsed = case if isinstance(case, Case) else parse_case(dict(case))
    meta = dict(parsed.runtime.metadata)
    spec_tokens = int(meta.get("draft_proposal_tokens", 0) or 0)
    track_name = _resolve_case_stack_track(parsed)
    step_req = request if request is not None else (step_eval.request if step_eval is not None else None)
    if step_req is None:
        raise ModelingError(
            "missing_explicit_step_request: build_export_bundle requires explicit request or step_eval "
            "(silent decode StepRequest fabrication with 1024 context was deleted)."
        )
    dp_width = int(parsed.runtime.parallel.dp)
    if step_req.requests:
        # Explicit RequestViews own the per-replica batch, matching the graph builder's precedence.
        active = max(1, sum(1 for view in step_req.requests if not view.padding))
        seq_count, cluster_conc = active, active * max(1, dp_width)
    else:
        seq_count, cluster_conc = _normalize_bundle_sequences(step_req.sequences, dp_width)
    if rewrite_plan is not None and apply_target_envelope is False:
        raise ModelingError("rewrite_plan_conflicts_with_apply_target_envelope_false")
    ext_plan = load_rewrite_plan(rewrite_plan) if rewrite_plan is not None else None
    has_explicit_edits = ext_plan is not None and bool(getattr(ext_plan, "edits", ()))
    include_draft = (
        step_req.scope == "model" and spec_tokens > 0 and step_req.tokens > 1
        and getattr(parsed.model, "auxiliary", None) is not None
    )
    return parsed, track_name, step_req, seq_count, cluster_conc, ext_plan, has_explicit_edits, include_draft


def _evaluate_bundle_prelim_rows(
    spine: Any,
    eval_case: Case,
    step_req: StepRequest,
    step_eval: StepEvaluation | None,
    replay: ReplayOptions,
    *,
    archetypes: list[LayerArchetypeSpec],
    rep_lp_by_idx: Mapping[int, int],
    use_archetype_dedup: bool,
    primitive_table: Any,
    streams: tuple[Any, ...],
) -> tuple[
    Case,
    list[OperatorExportRow],
    list[tuple[int, int, int, int, int]],
    dict[int, dict[str, Any]],
    list[LayerArchetypeSpec],
    dict[str, Any],
    str,
    tuple[FusedPipelineDescriptor, ...],
    Any,
    int,
    int,
]:
    prepared_work_by_op: dict[str, dict[str, float]] = {}
    report_span_ms = 0.0
    if step_eval is not None:
        (
            eff_case, raw_records, op_spec_by_key, sources_by_op,
            struct_by_op, subgraph_by_op, graph_id, pipelines, rebound_primitive_table,
        ) = _resolve_bundle_step_eval(
            spine, eval_case, step_req, step_eval,
            primitive_table=primitive_table, streams=streams,
        )
    else:
        (
            eff_case, timeline, op_spec_by_key, sources_by_op,
            struct_by_op, subgraph_by_op, graph_id, prepared_work_by_op,
            rebound_primitive_table, report_span_ms,
        ) = _evaluate_representative_step(
            spine, eval_case, step_req, replay,
            primitive_table=primitive_table, streams=streams,
        )
        raw_records = _timeline_to_raw_records(
            timeline, is_step_eval=False, case=eff_case, op_spec_by_key=op_spec_by_key,
        )
        pipelines = ()
    filtered_records, tail_closure = _prepare_bundle_records(
        raw_records, archetypes=archetypes, use_archetype_dedup=use_archetype_dedup,
    )
    prelim_rows, demands_ns, peaks_by_stage = _build_prelim_operator_rows(
        filtered_records, eff_case=eff_case, archetypes=archetypes, rep_lp_by_idx=rep_lp_by_idx,
        use_archetype_dedup=use_archetype_dedup, sources_by_op=sources_by_op,
        struct_by_op=struct_by_op, subgraph_by_op=subgraph_by_op,
        prepared_work_by_op=prepared_work_by_op, tail_closure=tail_closure,
        pipeline_by_op=_pipeline_map_from_eval(step_eval),
        op_spec_by_key=op_spec_by_key,
    )
    prelim_rows, archetypes, unfolded_span_ns, unfolded_idle_ns = _attribute_full_bundle_stream0(
        prelim_rows, archetypes,
    )
    full_span_ms = float(step_eval.span_ms) if step_eval is not None else float(report_span_ms)
    if full_span_ms > 0.0:
        eval_span_ns = max(0, int(round(full_span_ms * NS_PER_MS)))
        if eval_span_ns > unfolded_span_ns:
            unfolded_idle_ns += eval_span_ns - unfolded_span_ns
            unfolded_span_ns = eval_span_ns
    return (
        eff_case, prelim_rows, demands_ns, peaks_by_stage, archetypes,
        op_spec_by_key, graph_id, pipelines, rebound_primitive_table,
        unfolded_span_ns, unfolded_idle_ns,
    )


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
    target_max_trials: int = DEFAULT_EXPORT_TARGET_TRIALS,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    top_k_radar: int = DEFAULT_GAP_RADAR_TOP_K,
    primitive_table: Any = None,
    streams: tuple[Any, ...] = (),
    handoff_tpot_ms: float | None = None,
    deployment_total_dies: int | None = None,
    pd_pool_summary: Mapping[str, Any] | None = None,
    serve_script_supplied: bool = False,
) -> ExportBundle:
    """Evaluate unique layer archetypes once on ``runtime.Engine`` into ``ExportBundle``."""
    parsed, track_name, step_req, seq_count, cluster_conc, ext_plan, has_explicit_edits, include_draft = (
        _resolve_bundle_step_context(
            case, request=request, step_eval=step_eval,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
        )
    )
    archetypes, full_schedule, rep_lp_by_idx, total_target_layers, total_draft_layers = (
        _discover_model_archetypes(parsed, include_draft=include_draft)
    )
    use_archetype_dedup = (
        step_eval is None and observations_path is None and not has_explicit_edits
        and (len(rep_lp_by_idx) < len(parsed.model.layer_plans) or total_draft_layers > 1)
    )
    spine = default_spine()
    should_synthesize = True if rewrite_plan is not None else (
        apply_target_envelope if apply_target_envelope is not None
        else (track_name == "target" and step_eval is None and observations_path is None)
    )
    defer_prelim = should_synthesize and step_eval is None and observations_path is None and not streams
    if defer_prelim:
        eff_case = parsed
        prelim_rows: list[OperatorExportRow] = []
        demands_ns: list[tuple[int, int, int, int, int]] = []
        peaks_by_stage: dict[int, dict[str, Any]] = {}
        op_spec_by_key: dict[tuple[int, str], OpSpec] = {}
        graph_id, pipelines = "", ()
        rebound_primitive_table: Any = (
            False if primitive_table is False
            else EvaluationContext.create(parsed, None, None, primitive_table=primitive_table).bindings.primitive_table
        )
        unfolded_span_ns, unfolded_idle_ns = 0, 0
        step_l_star_ns, step_cert_valid, step_tight_res = 0, True, ""
    else:
        (
            eff_case, prelim_rows, demands_ns, peaks_by_stage, archetypes,
            op_spec_by_key, graph_id, pipelines, rebound_primitive_table,
            unfolded_span_ns, unfolded_idle_ns,
        ) = _evaluate_bundle_prelim_rows(
            spine, parsed, step_req, step_eval, replay,
            archetypes=archetypes, rep_lp_by_idx=rep_lp_by_idx,
            use_archetype_dedup=use_archetype_dedup,
            primitive_table=primitive_table, streams=streams,
        )
        step_l_star_ns, step_cert_valid, step_tight_res = _certify_step_lower_bound(
            prelim_rows, demands_ns, step_span_ns=unfolded_span_ns,
        )
    synthesis_specs: tuple[Any, ...] = ()
    audit_triggers: tuple[Any, ...] = ()
    plan_digest, plan_verified = "", False
    applied_edit_receipts: tuple[dict[str, Any], ...] = ()
    stream_core_budget_summary, moe_route_summary = "", {}
    co_design_opportunities: tuple[Any, ...] = ()
    stage_waterfall: tuple[dict[str, Any], ...] = ()
    engine_step_receipt: EngineStepReceipt | None = None
    reverted_to_native_baseline = False
    if should_synthesize:
        (
            prelim_rows, demands_ns, archetypes,
            synthesis_specs, audit_triggers, plan_digest, plan_verified,
            applied_edit_receipts, stream_core_budget_summary, moe_route_summary,
            co_design_opportunities, stage_waterfall,
            engine_step_receipt, reverted_to_native_baseline,
            unfolded_span_ns, unfolded_idle_ns, step_l_star_ns, step_cert_valid,
            op_spec_by_key,
            st_eff_case, st_peaks, st_gid, st_pipes, st_pt,
        ) = _apply_target_envelope_to_rows(
            prelim_rows, demands_ns, archetypes, parsed=parsed, eff_case=eff_case,
            step_req=step_req, cluster_conc=cluster_conc, op_spec_by_key=op_spec_by_key,
            peaks_by_stage=peaks_by_stage, ext_plan=ext_plan,
            rebound_primitive_table=rebound_primitive_table, allow_unbound_formula=allow_unbound_formula,
            spine=spine, replay=replay, rep_lp_by_idx=rep_lp_by_idx, streams=streams,
            target_max_trials=target_max_trials,
            deferred_prelim=defer_prelim, use_archetype_dedup=use_archetype_dedup,
        )
        if defer_prelim:
            eff_case, peaks_by_stage, graph_id, pipelines, rebound_primitive_table = (
                st_eff_case, st_peaks, st_gid, st_pipes, st_pt,
            )
        unfolded_l_star_check, _, step_tight_res = _certify_step_lower_bound(
            prelim_rows, demands_ns, step_span_ns=unfolded_span_ns,
        )
        if step_l_star_ns <= 0 and unfolded_l_star_check > 0:
            step_l_star_ns = unfolded_l_star_check

    # Capture per-layer timings strictly on unfolded rows BEFORE read-only display folding.
    draft_steps = _collect_draft_proposal_steps(prelim_rows)
    per_layer_timings = _extract_per_layer_timings(prelim_rows, parsed.model.layer_plans)
    if use_archetype_dedup:
        prelim_rows, demands_ns, archetypes = _aggregate_bundle_cohort_rows(
            prelim_rows, demands_ns, archetypes,
            layer_plans=parsed.model.layer_plans,
            total_target_layers=total_target_layers,
        )

    # Display grouping and precision ABI display tags belong strictly after target synthesis.
    prelim_rows = _apply_display_dtypes(
        prelim_rows, eff_case=eff_case, archetypes=archetypes,
        rep_lp_by_idx=rep_lp_by_idx, op_spec_by_key=op_spec_by_key,
    )
    prelim_rows, demands_ns = _collapse_micro_split_ops(prelim_rows, demands_ns)
    archetypes, full_schedule = _reconcile_draft_schedule_repeats(
        archetypes, full_schedule, prelim_rows, draft_steps=draft_steps,
    )
    prelim_rows, demands_ns = _reorder_rows_by_archetype(prelim_rows, demands_ns, archetypes)
    final_rows, rank_order, reconcile_summary = _finalize_ranked_rows(prelim_rows, observations_path)

    assembled = _assemble_export_bundle(
        parsed=parsed, eff_case=eff_case, step_req=step_req, step_eval=step_eval,
        seq_count=seq_count, cluster_conc=cluster_conc,
        graph_id=graph_id, pipelines=pipelines, final_rows=final_rows,
        demands_ns=demands_ns, rank_order=rank_order, reconcile_summary=reconcile_summary,
        archetypes=archetypes, full_schedule=full_schedule,
        total_target_layers=total_target_layers, total_draft_layers=total_draft_layers,
        use_archetype_dedup=use_archetype_dedup, target_envelope_applied=should_synthesize,
        peaks_by_stage=peaks_by_stage, weight_quant=weight_quant, cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload, include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored, top_k_radar=top_k_radar,
        synthesis_specs=synthesis_specs, audit_triggers=audit_triggers,
        rewrite_plan_sha256=plan_digest, rewrite_verification_passed=plan_verified,
        allow_unbound_formula=allow_unbound_formula,
        handoff_tpot_ms=handoff_tpot_ms,
        deployment_total_dies=deployment_total_dies,
        pd_pool_summary=pd_pool_summary,
        rebound_primitive_table=rebound_primitive_table,
        applied_edit_receipts=applied_edit_receipts,
        stream_core_budget_summary=stream_core_budget_summary,
        moe_route_summary=moe_route_summary,
        serve_script_supplied=bool(serve_script_supplied),
        per_layer_timings=per_layer_timings,
        co_design_opportunities=co_design_opportunities,
        stage_waterfall=stage_waterfall,
        unfolded_step_span_ns=unfolded_span_ns,
        unfolded_idle_ns=unfolded_idle_ns,
        unfolded_l_star_ns=step_l_star_ns,
        unfolded_cert_valid=step_cert_valid,
        unfolded_tight_res=step_tight_res,
        engine_step_receipt=engine_step_receipt,
        reverted_to_native_baseline=reverted_to_native_baseline,
    )
    return assembled


def _build_checkpoint_and_va_summary(
    parsed: Case,
    mem_ledger: Any,
    scen: Mapping[str, Any],
    seq_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    hw_fam = normalize_hw_family(str(parsed.machine.sku or parsed.machine.name or resolve_arch_family_tag(parsed)))
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


def _build_bundle_memory_and_sla(
    parsed: Case,
    *,
    step_req: StepRequest,
    seq_count: int,
    step_span_us: float,
    scen: Mapping[str, Any],
    weight_quant: str | None,
    cache_dtype: str | None,
    engram_host_offload: bool | None,
    include_vision_tower: bool | None,
    apply_preset_to_ignored: bool,
    handoff_tpot_ms: float | None,
) -> tuple[Any, float, float, str, Any]:
    declared_out = (
        max((int(getattr(req, "output_tokens", 0) or 0) for req in parsed.workload.requests), default=0)
        if parsed.workload.requests else 0
    )
    req_ctx = int(step_req.context)
    if req_ctx <= 0 and step_req.requests:
        req_ctx = max((int(getattr(v, "context", 0) or 0) for v in step_req.requests), default=0)
    horizon_tokens = (
        max(int(step_req.tokens), declared_out)
        if step_req.phase == "decode" else int(step_req.tokens)
    )
    ledger_seq_len = max(req_ctx + horizon_tokens, int(step_req.tokens))
    mem_ledger = compute_memory_quant_ledger(
        parsed, phase=step_req.phase,
        seq_len=ledger_seq_len,
        concurrency=seq_count, weight_quant=weight_quant,
        cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
        include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored,
    )
    meta_dict = dict(parsed.runtime.metadata)
    eff_handoff = (
        max(0.0, float(handoff_tpot_ms if handoff_tpot_ms is not None else (meta_dict.get("handoff_tpot_ms", 0.0) or 0.0)))
        if step_req.phase == "decode" else 0.0
    )
    eff_tpot_ms = (
        (step_span_us / US_PER_MS / max(1.0, float(scen["avg_accepted_tokens"]))) + eff_handoff
        if step_req.phase == "decode" else (step_span_us / US_PER_MS)
    )
    raw_sla_target = scen["tpot_slo_ms"] if step_req.phase == "decode" else scen["ttft_slo_ms"]
    min_headroom = min((st.headroom_bytes for st in mem_ledger.stage_budgets), default=0.0)
    if raw_sla_target is None or float(raw_sla_target) <= 0.0:
        sla_regime = "HBM_CAPACITY_EXCEEDED" if not bool(mem_ledger.fits_hbm) else "UNCONSTRAINED"
        return mem_ledger, eff_handoff, 0.0, sla_regime, None
    macro_wall = evaluate_macro_wall_trigger(
        tpot_ms=eff_tpot_ms,
        tpot_slo_ms=float(raw_sla_target),
        hbm_used_bytes=float(mem_ledger.max_stage_hbm_bytes),
        hbm_usable_bytes=float(mem_ledger.usable_hbm_bytes),
        hbm_headroom_bytes=float(min_headroom),
    )
    sla_slack = float(macro_wall.sla_slack_ms)
    if not bool(mem_ledger.fits_hbm):
        sla_regime = "HBM_CAPACITY_EXCEEDED"
    elif sla_slack < 0.0:
        sla_regime = "SLA_VIOLATED"
    elif macro_wall.wall_kind == "HBM_CAPACITY_WALL":
        sla_regime = "HBM_CAPACITY_BOUND"
    else:
        sla_regime = "SLA_LATENCY_BOUND"
    return mem_ledger, eff_handoff, sla_slack, sla_regime, macro_wall


def _resolve_graph_capture_mode(parsed: Case) -> str:
    explicit = str(getattr(parsed.runtime, "graph_capture_mode", "") or "").strip()
    if explicit:
        return explicit
    meta = dict(parsed.runtime.metadata)
    comp = meta.get("compilation_config")
    if isinstance(comp, Mapping):
        cg_mode = str(comp.get("cudagraph_mode") or comp.get("mode") or "").strip()
        if cg_mode:
            return cg_mode
    return "none"


def _build_effective_config_dict(
    parsed: Case,
    mem_ledger: Any,
    scen: Mapping[str, Any],
    *,
    graph_capture_mode: str,
    track_name: str,
    stream_core_budget_summary: str,
    moe_route_summary: Mapping[str, Any],
    serve_script_supplied: bool,
) -> dict[str, Any]:
    p = parsed.runtime.parallel
    meta = dict(parsed.runtime.metadata)
    return {
        "hardware_sku": str(parsed.machine.sku or parsed.machine.name),
        "arch_family": resolve_arch_family_tag(parsed),
        "tp": int(p.tp),
        "dp": int(p.dp),
        "ep": int(p.ep),
        "pp": int(p.pp),
        "dcp": int(p.cp),
        "pcp": int(p.pcp),
        "graph_capture_mode": graph_capture_mode,
        "feature_track": track_name,
        "weight_quant": str(getattr(mem_ledger, "weight_quant", "") or ""),
        "cache_dtype": str(getattr(mem_ledger, "cache_dtype", "") or ""),
        "spec_method": str(scen.get("spec_method", "None")),
        "spec_proposals": int(scen.get("spec_proposals", 0) or 0),
        "multistream_overlap_shared_expert": bool(
            moe_route_summary.get("multistream_overlap_shared_expert")
            if moe_route_summary else meta.get("multistream_overlap_shared_expert", False)
        ),
        "enable_fused_mc2": int(moe_route_summary.get("enable_fused_mc2", 0) or 0) if moe_route_summary else 0,
        "moe_route": dict(moe_route_summary) if moe_route_summary else {},
        "stream_core_budget": stream_core_budget_summary,
        "serve_script_supplied": bool(serve_script_supplied),
    }


def _build_bundle_gap_radar(
    final_rows: Sequence[OperatorExportRow],
    rank_order: Sequence[int],
    *,
    step_span_us: float,
    top_k_radar: int,
) -> tuple[GapRadarRow, ...]:
    return tuple(
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


def _build_hw_peaks_and_die_summary(
    parsed: Case, p0: Mapping[str, Any],
) -> tuple[dict[str, float], str]:
    default_dt = str(p0.get("default_dtype") or "").strip()
    hw_peaks_summary = {
        "cube_peak_flops_per_s": _die_cube_peak_rate(p0["die"], default_dt, default_dt),
        "vector_peak_ops_per_s": float(p0["vector_peak_ops_per_s"]),
        "hbm_peak_bytes_per_s": float(p0["hbm_peak_bytes_per_s"]),
        "mte2_peak_bytes_per_s": float(p0.get("mte2_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "mte3_peak_bytes_per_s": float(p0.get("mte3_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "fixpipe_peak_bytes_per_s": float(p0.get("fixpipe_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "link_peak_bytes_per_s": float(p0["link_peak_bytes_per_s"]),
    }
    d0 = parsed.machine.dies[0]
    d_bufs = dict(d0.buffers)
    die_summary = (
        f"AIC={d0.aic_count}, AIV={int(d0.owner_count('vector') or 0)}, "
        f"L2={int(d_bufs.get('l2', 0)) // MIB_BYTES} MiB, "
        f"UB={int(d_bufs.get('ub', 0)) // KIB_BYTES} KiB, "
        f"HBM={hw_peaks_summary['hbm_peak_bytes_per_s'] / TB_BYTES:.2f} TB/s"
    )
    return hw_peaks_summary, die_summary


def _resolve_calibrated_launch_us(parsed: Case, primitive_table: Any) -> tuple[float, float, float]:
    """Read calibrated STARS/host launch durations (us) from PrimitiveTable or runtime.submission."""
    sub = getattr(parsed.runtime, "submission", None)
    sub_kernel_us = max(0.0, float(getattr(sub, "kernel_launch_ms", 0.0) or 0.0) * US_PER_MS) if sub is not None else 0.0
    sub_graph_us = max(0.0, float(getattr(sub, "graph_launch_ms", 0.0) or 0.0) * US_PER_MS) if sub is not None else 0.0
    pt_host_us = 0.0
    pt_issue_us = 0.0
    pt_h2d_us = 0.0
    if primitive_table is not None and primitive_table is not False and hasattr(primitive_table, "cell"):
        host_ns = _table_band_p50(primitive_table, "P15_rts_stars", "host_submit_ns")
        issue_ns = _table_band_p50(primitive_table, "P15_rts_stars", "issue_interval_ns")
        h2d_ns = _table_band_p50(primitive_table, "P20_host_dma_sync", "pinned_dma_startup_ns")
        if host_ns is not None and host_ns > 0.0:
            pt_host_us = float(host_ns) / NS_PER_US
        if issue_ns is not None and issue_ns > 0.0:
            pt_issue_us = float(issue_ns) / NS_PER_US
        if h2d_ns is not None and h2d_ns > 0.0:
            pt_h2d_us = float(h2d_ns) / NS_PER_US
    host_launch_us = sub_kernel_us if sub_kernel_us > 0.0 else (pt_host_us if pt_host_us > 0.0 else pt_issue_us)
    graph_launch_us = sub_graph_us if sub_graph_us > 0.0 else (pt_issue_us if pt_issue_us > 0.0 else host_launch_us)
    h2d_startup_us = pt_h2d_us if pt_h2d_us > 0.0 else host_launch_us
    return host_launch_us, graph_launch_us, h2d_startup_us


def _canonical_step_request_digest_fields(step_req: StepRequest) -> dict[str, Any]:
    raw_seqs = step_req.sequences
    if isinstance(raw_seqs, Mapping):
        norm_seqs: Any = sorted((int(k), int(v)) for k, v in raw_seqs.items())
    elif isinstance(raw_seqs, (tuple, list)):
        norm_seqs = [int(v) for v in raw_seqs]
    else:
        norm_seqs = int(raw_seqs)
    norm_reqs = [
        [
            str(r.request_id), int(r.q_len), int(r.context), bool(r.samples), str(r.phase),
            int(r.prefix_tokens), int(r.priority), bool(r.padding), str(r.semantic_phase),
            str(r.execution_mode), bool(r.media_encoded), bool(r.draft_verify),
            bool(r.draft_prepare), int(r.draft_proposal_tokens),
            int(r.draft_accepted_tokens), int(r.draft_advanced_tokens), int(r.prompt_tokens),
        ]
        for r in (step_req.requests or ())
    ]
    return {
        "phase": str(step_req.phase),
        "scope": str(step_req.scope),
        "tokens": int(step_req.tokens),
        "context": int(step_req.context),
        "sequences": norm_seqs,
        "media_tokens_per_sequence": (
            int(step_req.media_tokens_per_sequence)
            if step_req.media_tokens_per_sequence is not None else None
        ),
        "draft_prepare": bool(step_req.draft_prepare),
        "requests": norm_reqs,
        "padded_tokens": int(step_req.padded_tokens),
        "layout": str(step_req.layout),
    }


def _compute_bundle_case_digest(
    parsed: Case,
    *,
    step_req: StepRequest,
    graph_id: str,
    track_name: str,
    weight_quant: str,
    cache_dtype: str,
    target_envelope_applied: bool,
    rewrite_plan_sha256: str,
) -> str:
    p = parsed.runtime.parallel
    payload = {
        "model": str(parsed.model.name),
        "sku": str(parsed.machine.sku or parsed.machine.name),
        **_canonical_step_request_digest_fields(step_req),
        "graph_id": str(graph_id),
        "track": str(track_name),
        "parallel": [int(p.dp), int(p.tp), int(p.ep), int(p.pp), int(p.cp), int(p.pcp)],
        "weight_quant": str(weight_quant),
        "cache_dtype": str(cache_dtype),
        "target_envelope": bool(target_envelope_applied),
        "plan_sha256": str(rewrite_plan_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_bundle_step_accounting(
    final_rows: tuple[OperatorExportRow, ...],
    *,
    step_eval: StepEvaluation | None,
    unfolded_step_span_ns: int,
    unfolded_l_star_ns: int,
    unfolded_cert_valid: bool,
) -> tuple[int, int, float, float, int, float, bool]:
    sum_row_exposed_ns = sum(max(0, int(r.exposed_ns)) for r in final_rows)
    if unfolded_step_span_ns > 0:
        step_span_ns = max(int(unfolded_step_span_ns), sum_row_exposed_ns)
    elif step_eval is not None:
        step_span_ns = max(int(round(float(step_eval.span_ms) * NS_PER_MS)), sum_row_exposed_ns)
    else:
        min_start = min((r.start_us for r in final_rows), default=0.0)
        max_end = max((r.end_us for r in final_rows), default=0.0)
        step_span_ns = max(int(round(max(0.0, max_end - min_start) * NS_PER_US)), sum_row_exposed_ns)
    idle_or_wait_ns = max(0, step_span_ns - sum_row_exposed_ns)
    step_span_us = float(step_span_ns) / NS_PER_US
    op_sum_us = (
        float(sum(int(r.dur_ns) for r in final_rows)) / NS_PER_US
        if any(r.dur_ns > 0 for r in final_rows)
        else sum(r.dur_us * r.repeat_count for r in final_rows)
    )
    step_l_star_ns = max(0, int(unfolded_l_star_ns))
    step_l_star_us = float(step_l_star_ns) / NS_PER_US
    cert_valid = bool(unfolded_cert_valid and step_l_star_ns <= step_span_ns)
    return step_span_ns, idle_or_wait_ns, step_span_us, op_sum_us, step_l_star_ns, step_l_star_us, cert_valid


def _resolve_bundle_identity_and_extras(
    parsed: Case,
    *,
    step_req: StepRequest,
    graph_id: str,
    track_name: str,
    mem_ledger: Any,
    scen: Mapping[str, Any],
    graph_mode: str,
    stream_core_budget_summary: str,
    moe_route_summary: Mapping[str, Any],
    serve_script_supplied: bool,
    rebound_primitive_table: Any,
    target_envelope_applied: bool,
    rewrite_plan_sha256: str,
    applied_edit_receipts: tuple[dict[str, Any], ...],
    per_layer_timings: Mapping[tuple[str, int], Mapping[str, Any]],
    co_design_opportunities: tuple[Any, ...],
    stage_waterfall: tuple[dict[str, Any], ...],
    step_span_ns: int,
    idle_or_wait_ns: int,
    step_l_star_ns: int,
    engine_step_receipt: EngineStepReceipt | None,
    reverted_to_native_baseline: bool,
) -> dict[str, Any]:
    eff_cfg = _build_effective_config_dict(
        parsed, mem_ledger, scen, graph_capture_mode=graph_mode, track_name=track_name,
        stream_core_budget_summary=stream_core_budget_summary,
        moe_route_summary=moe_route_summary if isinstance(moe_route_summary, Mapping) else {},
        serve_script_supplied=serve_script_supplied,
    )
    host_launch_us, graph_launch_us, h2d_startup_us = _resolve_calibrated_launch_us(
        parsed, rebound_primitive_table,
    )
    case_digest = _compute_bundle_case_digest(
        parsed,
        step_req=step_req,
        graph_id=graph_id,
        track_name=track_name,
        weight_quant=str(getattr(mem_ledger, "weight_quant", "") or ""),
        cache_dtype=str(getattr(mem_ledger, "cache_dtype", "") or ""),
        target_envelope_applied=target_envelope_applied,
        rewrite_plan_sha256=rewrite_plan_sha256,
    )
    exec_key = ""
    if engine_step_receipt is not None:
        for st in reversed(engine_step_receipt.stages):
            if st.status == "committed" and st.execution_key_sha256:
                exec_key = str(st.execution_key_sha256)
                break
    if not exec_key:
        exec_key = hashlib.sha256(f"{case_digest}:{graph_id}:{step_span_ns}".encode("utf-8")).hexdigest()
    return _filter_dataclass_kwargs(
        _BUNDLE_FIELDS,
        {
            "applied_edit_receipts": applied_edit_receipts,
            "effective_config": eff_cfg,
            "serve_script_supplied": bool(serve_script_supplied),
            "per_layer_timings": dict(per_layer_timings) if isinstance(per_layer_timings, Mapping) else {},
            "co_design_opportunities": tuple(co_design_opportunities),
            "stage_waterfall": tuple(stage_waterfall),
            "step_span_ns": step_span_ns,
            "idle_or_wait_ns": idle_or_wait_ns,
            "step_l_star_ns": step_l_star_ns,
            "case_digest": case_digest,
            "execution_key_sha256": exec_key,
            "engine_step_receipt": engine_step_receipt,
            "reverted_to_native_baseline": bool(reverted_to_native_baseline),
            "host_launch_us": host_launch_us,
            "graph_launch_us": graph_launch_us,
            "h2d_startup_us": h2d_startup_us,
        },
    )


def _assemble_export_bundle(
    *, parsed: Case, eff_case: Case, step_req: StepRequest, step_eval: StepEvaluation | None,
    seq_count: int, cluster_conc: int, graph_id: str, pipelines: tuple[FusedPipelineDescriptor, ...],
    final_rows: tuple[OperatorExportRow, ...], demands_ns: Sequence[tuple[int, int, int, int, int]],
    rank_order: Sequence[int], reconcile_summary: ReconciliationSummary | None,
    archetypes: Sequence[LayerArchetypeSpec], full_schedule: Sequence[TraceScheduleEntry],
    total_target_layers: int, total_draft_layers: int, use_archetype_dedup: bool,
    target_envelope_applied: bool, peaks_by_stage: Mapping[int, dict[str, Any]],
    weight_quant: str | None, cache_dtype: str | None, engram_host_offload: bool | None,
    include_vision_tower: bool | None, apply_preset_to_ignored: bool, top_k_radar: int,
    synthesis_specs: tuple[Any, ...] = (), audit_triggers: tuple[Any, ...] = (),
    rewrite_plan_sha256: str = "", rewrite_verification_passed: bool = False,
    allow_unbound_formula: bool = True,
    handoff_tpot_ms: float | None = None,
    deployment_total_dies: int | None = None,
    pd_pool_summary: Mapping[str, Any] | None = None,
    rebound_primitive_table: Any = None,
    applied_edit_receipts: tuple[dict[str, Any], ...] = (),
    stream_core_budget_summary: str = "",
    moe_route_summary: Mapping[str, Any] = (),
    serve_script_supplied: bool = False,
    per_layer_timings: Mapping[tuple[str, int], Mapping[str, Any]] = (),
    co_design_opportunities: tuple[Any, ...] = (),
    stage_waterfall: tuple[dict[str, Any], ...] = (),
    unfolded_step_span_ns: int = 0,
    unfolded_idle_ns: int = 0,
    unfolded_l_star_ns: int = 0,
    unfolded_cert_valid: bool = True,
    unfolded_tight_res: tuple[str, ...] = (),
    engine_step_receipt: EngineStepReceipt | None = None,
    reverted_to_native_baseline: bool = False,
) -> ExportBundle:
    del demands_ns, unfolded_idle_ns
    (
        step_span_ns, idle_or_wait_ns, step_span_us, op_sum_us,
        step_l_star_ns, step_l_star_us, cert_valid,
    ) = _resolve_bundle_step_accounting(
        final_rows, step_eval=step_eval, unfolded_step_span_ns=unfolded_step_span_ns,
        unfolded_l_star_ns=unfolded_l_star_ns, unfolded_cert_valid=unfolded_cert_valid,
    )
    tight_res = tuple(unfolded_tight_res)
    p0 = peaks_by_stage.get(0) or _die_peaks_for_stage(eff_case, 0, 0)
    cube_mfu, vec_mfu, hbm_mbu, link_util, net_roof, exposed_us, b_mix = _compute_network_efficiency_summary(
        final_rows, step_span_us, p0,
    )
    gap_radar = _build_bundle_gap_radar(
        final_rows, rank_order, step_span_us=step_span_us, top_k_radar=top_k_radar,
    )
    scen = _extract_scenario_meta(parsed, step_req, final_rows)
    step_s = max(step_span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    eff_cluster_conc = max(1, int(cluster_conc))
    tps = float(eff_cluster_conc) * float(scen["avg_accepted_tokens"]) / step_s
    num_dies = max(1, len(parsed.machine.dies))
    n_tok_per_die = float(eff_cluster_conc) * float(scen["avg_accepted_tokens"]) / float(num_dies)
    us_per_tok_die = step_span_us / max(n_tok_per_die, MIN_DENOM_EPS)
    hw_peaks_summary, die_summary = _build_hw_peaks_and_die_summary(parsed, p0)
    mem_ledger, eff_handoff, sla_slack, sla_regime, macro_wall = _build_bundle_memory_and_sla(
        parsed, step_req=step_req, seq_count=seq_count, step_span_us=step_span_us,
        scen=scen, weight_quant=weight_quant, cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload, include_vision_tower=include_vision_tower,
        apply_preset_to_ignored=apply_preset_to_ignored, handoff_tpot_ms=handoff_tpot_ms,
    )
    meta_dict = dict(parsed.runtime.metadata)
    ckpt_summary, va_summary = _build_checkpoint_and_va_summary(parsed, mem_ledger, scen, seq_count)
    track_name = _resolve_case_stack_track(parsed)
    active_ids = {r.archetype_id for r in final_rows}
    filtered_schedule = tuple(e for e in full_schedule if e.archetype_id in active_ids) if use_archetype_dedup else ()
    has_unbound = any(r.binding_state == BINDING_STATE_UNBOUND_FORMULA for r in final_rows)
    bundle_mode = MODE_MODELING if ((target_envelope_applied and allow_unbound_formula) or track_name == "target") else MODE_SIMULATION
    has_measured_match = reconcile_summary is not None and int(reconcile_summary.matched_sim_ops) > 0
    bundle_binding = (
        BINDING_STATE_UNBOUND_FORMULA if has_unbound
        else (BINDING_STATE_MEASURED if has_measured_match else BINDING_STATE_BOUND)
    )
    raw_pd_pool = pd_pool_summary if pd_pool_summary is not None else meta_dict.get("pd_pool_summary")
    eff_pd_pool = dict(raw_pd_pool) if isinstance(raw_pd_pool, Mapping) else None
    if eff_pd_pool:
        scen["deployment_mode"] = DEPLOYMENT_MODE_LABELS["pd_disaggregated"]
    raw_dep_dies = (
        deployment_total_dies if deployment_total_dies is not None
        else (eff_pd_pool.get("used_dies") if eff_pd_pool else meta_dict.get("deployment_total_dies"))
    )
    eff_dep_dies = max(0, int(raw_dep_dies or 0))
    graph_mode = _resolve_graph_capture_mode(parsed)
    bundle_extras = _resolve_bundle_identity_and_extras(
        parsed,
        step_req=step_req, graph_id=graph_id, track_name=track_name,
        mem_ledger=mem_ledger, scen=scen, graph_mode=graph_mode,
        stream_core_budget_summary=stream_core_budget_summary,
        moe_route_summary=moe_route_summary, serve_script_supplied=serve_script_supplied,
        rebound_primitive_table=rebound_primitive_table,
        target_envelope_applied=target_envelope_applied,
        rewrite_plan_sha256=rewrite_plan_sha256,
        applied_edit_receipts=applied_edit_receipts,
        per_layer_timings=per_layer_timings,
        co_design_opportunities=co_design_opportunities,
        stage_waterfall=stage_waterfall,
        step_span_ns=step_span_ns, idle_or_wait_ns=idle_or_wait_ns,
        step_l_star_ns=step_l_star_ns, engine_step_receipt=engine_step_receipt,
        reverted_to_native_baseline=reverted_to_native_baseline,
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
        graph_capture_mode=graph_mode,
        feature_track=track_name, hardware_die_summary=die_summary,
        target_envelope_applied=target_envelope_applied,
        n_tok_per_die=n_tok_per_die, us_per_tok_die=us_per_tok_die,
        sla_regime=sla_regime, sla_slack_ms=sla_slack,
        kernel_synthesis_specs=synthesis_specs, semantic_audit_triggers=audit_triggers,
        macro_wall_trigger=macro_wall, rewrite_plan_sha256=rewrite_plan_sha256,
        rewrite_verification_passed=rewrite_verification_passed,
        recommended_checkpoint=ckpt_summary, va_serving_rules_summary=va_summary,
        analysis_mode=bundle_mode, binding_state=bundle_binding,
        has_unbound_formula_ops=has_unbound, handoff_tpot_ms=eff_handoff,
        cluster_concurrency=eff_cluster_conc, deployment_total_dies=eff_dep_dies,
        pd_pool_summary=eff_pd_pool, calibration_basis=None,
        **scen, **bundle_extras,
    )

