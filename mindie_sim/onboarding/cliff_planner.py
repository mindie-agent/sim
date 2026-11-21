"""Deterministic disjoint train/off-grid cliff bench planner and white-box scorer.

[Role] Synthesize strictly disjoint ``train`` anchor and ``heldout`` off-grid
  cliff benchmark plans for both new SKUs (512..400K + L1/L2/L3) and new
  operators (tile ± 1, cores * tile ± 1, topk ± 1, L2 knee ± 1, 512..400K),
  validate acquisition-lineage separation, and score pure white-box predictions
  against measured silicon ``p50`` timings.
[Invariants] ``train`` and ``heldout`` never share context points or normalized
  acquisition lineage roots; ``L1 != L2 != L3`` heldout roots stay pairwise
  disjoint; ``service_regimes`` crutches are unconditionally rejected; all
  scored residuals use fractional ``signed_error`` / ``absolute_percentage_error``
  matching ``multilevel_gate.Residual`` and ``scripts/diagnose_tail_cluster.py``.
[Timeouts & Locks] Pure deterministic planning and in-memory scoring; no locks,
  subprocesses, or device contexts.
[Side-Effect Policy] Read-only over inputs; returns new immutable plans and
  serializable reports.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import math
import statistics
from typing import Any

from mindie_sim.observations.multilevel_gate import (
    CohortKey,
    Residual,
    normalize_lineage_token,
)
from mindie_sim.schema.common import ModelingError

from .contracts import (
    ANCHOR_CONTEXT_GRID,
    CLIFF_PLAN_SCHEMA,
    COLD_RING_L2_MULTIPLIER,
    DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    DISPLAY_ROUND_DIGITS,
    NS_PER_MS,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    PERCENT_SCALE,
    SHORT_DIGEST_LEN,
    CliffBenchPlan,
    CliffPointSpec,
    OperatorCliffParameters,
    canonical_json_bytes,
    compute_receipt_sha256,
)

TAIL_QUANTILE_P90 = 0.9
WAVE_TAIL_MAX_TOKENS = 4097
SPARSE_PHASE_CHANGE_TOKEN = 2049
L2_KNEE_MIN_TOKENS = 24576
L2_KNEE_MAX_TOKENS = 98304

STANDARD_SEQUENCE_OPERATORS: tuple[str, ...] = (
    "fused_infer_attention",
    "paged_mla",
    "c8_sfa",
    "lightning_indexer_topk",
    "dcp_lse_attention_merge",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
    "chunk_kda",
    "recurrent_kda",
    "matmul",
    "grouped_matmul",
    "mhc_combine",
    "all_reduce",
)

STANDARD_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "single_operator",
    "L2": "subgraph_block",
    "L3": "model_step",
}

_FORBIDDEN_CRUTCH_KEYS = frozenset({"service_regimes", "service_regime_override"})


def _lineage_root(tag: str, seed_payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(seed_payload)).hexdigest()[:SHORT_DIGEST_LEN]
    return f"acq-{tag}-{digest}"


def _cliff_regime_for_context(seq_len: int) -> tuple[str, str]:
    if seq_len == SPARSE_PHASE_CHANGE_TOKEN:
        return (
            "dsa_topk_phase_change",
            "Exact transition across dsa_index_topk=2048 dense-to-sparse phase boundary.",
        )
    if seq_len <= WAVE_TAIL_MAX_TOKENS:
        return (
            "partial_wave_tail_block",
            "Partial-wave / split_kv +1 tail-block wave-quantization cliff.",
        )
    if L2_KNEE_MIN_TOKENS <= seq_len <= L2_KNEE_MAX_TOKENS:
        return (
            "l2_residency_knee",
            "L2 cache residency knee where KV/page-table working set spills to cold HBM.",
        )
    return (
        "ultra_long_sort_mrg_dcp",
        "Ultra-long-context Lightning Indexer + Sort32/MrgSort multi-pass merge & DCP regime.",
    )


def synthesize_operator_cliff_plan(
    params: OperatorCliffParameters,
    *,
    sku: str,
    campaign_id: str = "onboard-op",
    include_context_grids: bool = True,
) -> CliffBenchPlan:
    """Synthesize a strictly disjoint train + off-grid cliff plan for an operator."""
    if not sku or not sku.strip():
        raise ModelingError("cliff_plan_sku_missing")
    tile = params.tile_dim
    wave = params.core_count * tile

    heldout_raw: list[CliffPointSpec] = [
        CliffPointSpec(
            f"tile-minus-1-{tile - 1}", "heldout", "boundary_cliff", "tile_m", tile - 1,
            "tile_tail_underfill", f"tile - 1 ({tile - 1}) stresses partial-tile padding and masking.",
        ),
        CliffPointSpec(
            f"tile-plus-1-{tile + 1}", "heldout", "boundary_cliff", "tile_m", tile + 1,
            "tile_tail_spill", f"tile + 1 ({tile + 1}) forces an extra tail tile beyond {tile}.",
        ),
        CliffPointSpec(
            f"wave-minus-1-{wave - 1}", "heldout", "boundary_cliff", "wave_m", wave - 1,
            "wave_under_occupancy", f"cores * tile - 1 ({wave - 1}) stresses last-core partial load.",
        ),
        CliffPointSpec(
            f"wave-plus-1-{wave + 1}", "heldout", "boundary_cliff", "wave_m", wave + 1,
            "wave_quantization_cliff", f"cores * tile + 1 ({wave + 1}) adds a 1-item tail wave across {params.core_count} cores.",
        ),
    ]
    train_raw: list[CliffPointSpec] = [
        CliffPointSpec(
            f"tile-exact-{tile}", "train", "anchor", "tile_m", tile,
            "tile_aligned_anchor", f"Exact single-tile anchor ({tile}).",
        ),
        CliffPointSpec(
            f"wave-exact-{wave}", "train", "anchor", "wave_m", wave,
            "full_wave_anchor", f"Exact full-wave anchor ({params.core_count} * {tile}).",
        ),
    ]

    if params.topk_threshold is not None:
        tk = params.topk_threshold
        train_raw.append(
            CliffPointSpec(
                f"topk-exact-{tk}", "train", "anchor", "topk", tk,
                "topk_dense_anchor", f"Exact dense/sparse topk boundary anchor ({tk}).",
            )
        )
        heldout_raw.extend((
            CliffPointSpec(
                f"topk-minus-1-{tk - 1}", "heldout", "boundary_cliff", "topk", tk - 1,
                "topk_below_threshold", f"topk - 1 ({tk - 1}) stays just inside dense/fast selection.",
            ),
            CliffPointSpec(
                f"topk-plus-1-{tk + 1}", "heldout", "boundary_cliff", "topk", tk + 1,
                "topk_above_threshold", f"topk + 1 ({tk + 1}) triggers sparse/multi-pass selection.",
            ),
        ))

    if params.bytes_per_token > 0:
        l2_knee = max(2, params.l2_bytes // params.bytes_per_token)
        train_raw.append(
            CliffPointSpec(
                f"l2-knee-anchor-{l2_knee}", "train", "anchor", "l2_tokens", l2_knee,
                "l2_capacity_anchor", f"Exact L2 residency boundary ({l2_knee} tokens).",
            )
        )
        heldout_raw.extend((
            CliffPointSpec(
                f"l2-knee-minus-1-{l2_knee - 1}", "heldout", "boundary_cliff", "l2_tokens", l2_knee - 1,
                "l2_warm_boundary", f"l2_knee - 1 ({l2_knee - 1}) stays warm in L2.",
            ),
            CliffPointSpec(
                f"l2-knee-plus-1-{l2_knee + 1}", "heldout", "boundary_cliff", "l2_tokens", l2_knee + 1,
                "l2_cold_spill_cliff", f"l2_knee + 1 ({l2_knee + 1}) spills working set to cold HBM.",
            ),
        ))

    if include_context_grids:
        for seq_len in ANCHOR_CONTEXT_GRID:
            if params.context_min <= seq_len <= params.context_max:
                train_raw.append(
                    CliffPointSpec(
                        f"ctx-anchor-{seq_len}", "train", "anchor", "context_len", seq_len,
                        "anchor_context_grid", f"Canonical power-of-2 / 400K anchor context ({seq_len}).",
                    )
                )
        for seq_len in OFF_GRID_CLIFF_HELDOUT_GRID:
            if params.context_min <= seq_len <= params.context_max:
                regime_tag, rationale = _cliff_regime_for_context(seq_len)
                heldout_raw.append(
                    CliffPointSpec(
                        f"ctx-cliff-{seq_len}", "heldout", "off_grid_cliff", "context_len", seq_len,
                        regime_tag, rationale,
                    )
                )

    heldout_seen: set[tuple[str, int]] = set()
    heldout_dedup: list[CliffPointSpec] = []
    for pt in heldout_raw:
        key = (pt.axis, pt.value)
        if key not in heldout_seen:
            heldout_seen.add(key)
            heldout_dedup.append(pt)

    train_seen: set[tuple[str, int]] = set()
    train_dedup: list[CliffPointSpec] = []
    for pt in train_raw:
        key = (pt.axis, pt.value)
        if key not in heldout_seen and key not in train_seen:
            train_seen.add(key)
            train_dedup.append(pt)

    plan_id = f"{campaign_id}:{params.operator_id}:{sku}"
    train_root = _lineage_root("train", {"plan_id": plan_id, "split": "train"})
    heldout_root = _lineage_root("heldout", {"plan_id": plan_id, "split": "heldout"})
    draft = {
        "schema": CLIFF_PLAN_SCHEMA,
        "plan_id": plan_id,
        "target_id": params.operator_id,
        "sku": sku,
        "train_lineage_root": train_root,
        "heldout_lineage_root": heldout_root,
        "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "train_points": [pt.to_dict() for pt in train_dedup],
        "heldout_points": [pt.to_dict() for pt in heldout_dedup],
    }
    digest = compute_receipt_sha256(draft, digest_field="plan_sha256")
    return CliffBenchPlan(
        schema=CLIFF_PLAN_SCHEMA,
        plan_id=plan_id,
        target_id=params.operator_id,
        sku=sku,
        train_lineage_root=train_root,
        heldout_lineage_root=heldout_root,
        cold_ring_l2_multiplier=COLD_RING_L2_MULTIPLIER,
        train_points=tuple(train_dedup),
        heldout_points=tuple(heldout_dedup),
        plan_sha256=digest,
    )


def synthesize_sku_512_400k_plan(
    sku: str,
    *,
    l2_bytes: int,
    aic_count: int,
    aiv_count: int,
    operators: Sequence[str] = STANDARD_SEQUENCE_OPERATORS,
    campaign_id: str = "onboard-sku-512-400k",
) -> dict[str, Any]:
    """Build a complete disjoint Anchor + Off-Grid Cliff + L1/L2/L3 bench plan for a SKU."""
    if not sku or not sku.strip():
        raise ModelingError("sku_plan_name_missing")
    if l2_bytes < 1 or aic_count < 1 or aiv_count < 1:
        raise ModelingError("sku_plan_hardware_dimensions_invalid")
    train_root = _lineage_root("train", {"sku": sku, "campaign_id": campaign_id, "split": "train"})
    level_heldout_roots = {
        level: _lineage_root(
            f"heldout-{level.lower()}",
            {"sku": sku, "campaign_id": campaign_id, "split": "heldout", "level": level},
        )
        for level in ("L1", "L2", "L3")
    }
    cases: list[dict[str, Any]] = []
    for op_name in operators:
        for seq_len in ANCHOR_CONTEXT_GRID:
            cases.append({
                "case_id": f"{sku}:{op_name}:anchor:{seq_len}",
                "level": "L1",
                "split": "train",
                "grid_kind": "anchor",
                "operator": op_name,
                "context_len": seq_len,
                "acquisition_root": train_root,
                "boundary_kind": STANDARD_LEVEL_BOUNDARY_KINDS["L1"],
            })
        for seq_len in OFF_GRID_CLIFF_HELDOUT_GRID:
            regime_tag, _rationale = _cliff_regime_for_context(seq_len)
            cases.append({
                "case_id": f"{sku}:{op_name}:cliff:{seq_len}",
                "level": "L1",
                "split": "heldout",
                "grid_kind": "off_grid_cliff",
                "operator": op_name,
                "context_len": seq_len,
                "regime_tag": regime_tag,
                "acquisition_root": level_heldout_roots["L1"],
                "boundary_kind": STANDARD_LEVEL_BOUNDARY_KINDS["L1"],
            })
    for level in ("L2", "L3"):
        for seq_len in OFF_GRID_CLIFF_HELDOUT_GRID:
            regime_tag, _rationale = _cliff_regime_for_context(seq_len)
            cases.append({
                "case_id": f"{sku}:{level.lower()}:cliff:{seq_len}",
                "level": level,
                "split": "heldout",
                "grid_kind": "off_grid_cliff",
                "operator": "decoder_block" if level == "L2" else "full_model_step",
                "context_len": seq_len,
                "regime_tag": regime_tag,
                "acquisition_root": level_heldout_roots[level],
                "boundary_kind": STANDARD_LEVEL_BOUNDARY_KINDS[level],
            })
    payload = {
        "schema": CLIFF_PLAN_SCHEMA,
        "sku": sku,
        "campaign_id": campaign_id,
        "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "cold_ring_min_bytes": COLD_RING_L2_MULTIPLIER * l2_bytes,
        "hardware": {"l2_bytes": l2_bytes, "aic_count": aic_count, "aiv_count": aiv_count},
        "anchor_grid": list(ANCHOR_CONTEXT_GRID),
        "off_grid_cliff_heldout_grid": list(OFF_GRID_CLIFF_HELDOUT_GRID),
        "train_lineage_root": train_root,
        "heldout_lineage_roots_by_level": level_heldout_roots,
        "operators": list(operators),
        "case_count": len(cases),
        "cases": cases,
    }
    payload["plan_sha256"] = compute_receipt_sha256(payload, digest_field="plan_sha256")
    return payload


def validate_disjoint_lineage(
    train_roots: Iterable[str],
    heldout_roots: Iterable[str],
    *,
    level_roots: Mapping[str, Iterable[str]] | None = None,
) -> None:
    """Enforce strictly disjoint normalized acquisition roots across train/heldout and L1/L2/L3."""
    norm_train = {
        normalize_lineage_token(tok) for tok in train_roots if isinstance(tok, str) and tok.strip()
    }
    norm_heldout = {
        normalize_lineage_token(tok) for tok in heldout_roots if isinstance(tok, str) and tok.strip()
    }
    leaked = sorted(norm_train & norm_heldout)
    if leaked:
        raise ModelingError(f"onboarding_lineage_split_leakage:roots={','.join(leaked)}")
    if level_roots:
        seen_by_root: dict[str, set[str]] = {}
        for level, roots in level_roots.items():
            for raw_tok in roots:
                if not isinstance(raw_tok, str) or not raw_tok.strip():
                    continue
                tok = normalize_lineage_token(raw_tok)
                if tok in norm_train:
                    raise ModelingError(
                        f"onboarding_lineage_split_leakage:level={level}:root={tok}"
                    )
                seen_by_root.setdefault(tok, set()).add(str(level))
        for tok, levels in sorted(seen_by_root.items()):
            if len(levels) > 1:
                raise ModelingError(
                    f"onboarding_cross_level_root_overlap:root={tok}:levels={','.join(sorted(levels))}"
                )


def _reject_service_regimes(obj: Any) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) in _FORBIDDEN_CRUTCH_KEYS and value:
                raise ModelingError(f"service_regimes_crutch_forbidden:{key}")
            _reject_service_regimes(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _reject_service_regimes(item)


def _summarize_apes(apes_pct: Sequence[float], signed_pct: Sequence[float]) -> dict[str, Any]:
    if not apes_pct:
        return {
            "count": 0,
            "mape_pct": None,
            "medape_pct": None,
            "median_signed_pct": None,
            "p90_ape_pct": None,
            "max_ape_pct": None,
        }
    ordered = sorted(apes_pct)
    p90_idx = min(len(ordered) - 1, max(0, math.ceil(TAIL_QUANTILE_P90 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mape_pct": round(float(statistics.fmean(ordered)), DISPLAY_ROUND_DIGITS),
        "medape_pct": round(float(statistics.median(ordered)), DISPLAY_ROUND_DIGITS),
        "median_signed_pct": round(float(statistics.median(signed_pct)), DISPLAY_ROUND_DIGITS),
        "p90_ape_pct": round(float(ordered[p90_idx]), DISPLAY_ROUND_DIGITS),
        "max_ape_pct": round(float(ordered[-1]), DISPLAY_ROUND_DIGITS),
    }


def _to_int_ns(value: Any, label: str) -> int:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"observation_duration_invalid:{label}")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ModelingError(f"observation_duration_nonpositive:{label}")
    return int(round(fval))


def _predict_whitebox_from_case(row: Mapping[str, Any]) -> int | None:
    sim_case = row.get("sim_case")
    if not isinstance(sim_case, Mapping):
        return None
    _reject_service_regimes(sim_case)
    from mindie_sim.api import evaluate_step
    from mindie_sim.prediction.results import StepRequest

    req = StepRequest(
        phase=str(row.get("phase") or "decode"),  # type: ignore[arg-type]
        tokens=int(row.get("tokens") or 1),
        context=int(row.get("context_len") or 0),
        sequences=int(row.get("sequences") or 1),
        scope=str(row.get("scope") or "body"),  # type: ignore[arg-type]
    )
    step = evaluate_step(sim_case, req)
    return int(round(float(step.span_ms) * NS_PER_MS))


def score_whitebox_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    sku: str,
    threshold_pct: float = DEFAULT_HELDOUT_APE_THRESHOLD_PCT,
    predictor: Callable[[Mapping[str, Any]], int] | None = None,
    external_train_roots: Iterable[str] = (),
) -> dict[str, Any]:
    """Score pure white-box predictions vs silicon p50 and emit multilevel/diagnosis residuals."""
    _reject_service_regimes(observations)
    train_roots: set[str] = set(external_train_roots)
    heldout_roots: set[str] = set()
    level_roots: dict[str, set[str]] = {}
    residuals: list[dict[str, Any]] = []
    heldout_apes: list[float] = []
    heldout_signed: list[float] = []
    train_apes: list[float] = []
    train_signed: list[float] = []
    failing_cliffs: list[dict[str, Any]] = []
    seen_anchor_ctx: set[int] = set()
    seen_cliff_ctx: set[int] = set()

    for idx, row in enumerate(observations):
        cid = str(row.get("case_id") or f"obs-{idx}")
        split = str(row.get("split") or "").strip().lower()
        if split not in ("train", "heldout"):
            raise ModelingError(f"observation_split_invalid:{cid}:{split}")
        level = str(row.get("level") or "L1").strip().upper()
        if level not in ("L1", "L2", "L3"):
            raise ModelingError(f"observation_level_invalid:{cid}:{level}")
        root = str(row.get("acquisition_root") or row.get("acquisition_uid") or "").strip()
        if not root:
            raise ModelingError(f"observation_lineage_root_missing:{cid}")
        if split == "train":
            train_roots.add(root)
        else:
            heldout_roots.add(root)
            level_roots.setdefault(level, set()).add(root)

        measured_ns = _to_int_ns(
            row.get("measured_p50_ns", row.get("measured_ns")), f"{cid}:measured"
        )
        raw_pred = row.get("predicted_ns")
        if raw_pred is None and predictor is not None:
            raw_pred = predictor(row)
        if raw_pred is None:
            raw_pred = _predict_whitebox_from_case(row)
        predicted_ns = _to_int_ns(raw_pred, f"{cid}:predicted")

        signed = (predicted_ns - measured_ns) / float(measured_ns)
        ape_ratio = abs(signed)
        ape_pct = ape_ratio * PERCENT_SCALE
        grid_kind = str(row.get("grid_kind") or ("anchor" if split == "train" else "off_grid_cliff"))
        ctx_len = row.get("context_len")
        if isinstance(ctx_len, int) and not isinstance(ctx_len, bool):
            if split == "train":
                seen_anchor_ctx.add(ctx_len)
            else:
                seen_cliff_ctx.add(ctx_len)

        cohort = CohortKey(
            level=level,  # type: ignore[arg-type]
            cohort_name=str(row.get("cohort_name") or row.get("operator") or "onboarding"),
            sku=str(row.get("sku") or sku),
            phase=str(row.get("phase") or "decode"),
            mode=str(row.get("mode") or "eager"),
            shape_regime=str(row.get("regime_tag") or grid_kind),
            parallel_topology=str(row.get("parallel_topology") or "tp1_dp1"),
        )
        boundary_kind = str(row.get("boundary_kind") or STANDARD_LEVEL_BOUNDARY_KINDS[level])
        receipt_id = str(row.get("engine_receipt") or f"whitebox:{level}:{cid}")
        regime_dims = tuple(
            sorted((str(k), str(v)) for k, v in dict(row.get("regime_dims") or {}).items())
        )
        res_obj = Residual(
            case_id=cid,
            level=level,
            cohort=cohort,
            acquisition_root=root,
            measured_ns=measured_ns,
            predicted_ns=predicted_ns,
            engine_receipt=receipt_id,
            signed_error=signed,
            absolute_percentage_error=ape_ratio,
            status="ok",
            boundary_kind=boundary_kind,
            regime_dims=regime_dims,
        )
        res_dict = res_obj.to_dict()
        res_dict["split"] = split
        res_dict["grid_kind"] = grid_kind
        res_dict["ape_pct"] = round(ape_pct, DISPLAY_ROUND_DIGITS)
        residuals.append(res_dict)

        if split == "train":
            train_apes.append(ape_pct)
            train_signed.append(signed * PERCENT_SCALE)
        else:
            heldout_apes.append(ape_pct)
            heldout_signed.append(signed * PERCENT_SCALE)
            if ape_pct >= threshold_pct:
                failing_cliffs.append({
                    "case_id": cid,
                    "level": level,
                    "operator": cohort.cohort_name,
                    "grid_kind": grid_kind,
                    "context_len": ctx_len,
                    "measured_ns": measured_ns,
                    "predicted_ns": predicted_ns,
                    "ape_pct": round(ape_pct, DISPLAY_ROUND_DIGITS),
                    "acquisition_root": root,
                })

    validate_disjoint_lineage(train_roots, heldout_roots, level_roots=level_roots)
    heldout_summary = _summarize_apes(heldout_apes, heldout_signed)
    train_summary = _summarize_apes(train_apes, train_signed)
    passed = bool(heldout_apes) and not failing_cliffs
    return {
        "sku": sku,
        "threshold_pct": threshold_pct,
        "passed": passed,
        "ok": passed,
        "train_roots": sorted(train_roots),
        "heldout_roots": sorted(heldout_roots),
        "heldout_roots_by_level": {lvl: sorted(roots) for lvl, roots in sorted(level_roots.items())},
        "train_summary": train_summary,
        "heldout_summary": heldout_summary,
        "covered_anchor_contexts": sorted(seen_anchor_ctx),
        "covered_cliff_contexts": sorted(seen_cliff_ctx),
        "failing_cliff_count": len(failing_cliffs),
        "failing_cliffs": failing_cliffs,
        "residuals": residuals,
    }


__all__ = [
    "STANDARD_LEVEL_BOUNDARY_KINDS",
    "STANDARD_SEQUENCE_OPERATORS",
    "score_whitebox_observations",
    "synthesize_operator_cliff_plan",
    "synthesize_sku_512_400k_plan",
    "validate_disjoint_lineage",
]
