#!/usr/bin/env python3
"""Produce 3-level CANN 9.3.0 A5 silicon calibration & heldout accuracy report.

[Role] The cohesive producer owns report assembly from capture census to verdict.
  Evaluate ``mindie_sim`` across disjoint ``train`` and ``heldout`` CANN 9.3.0
  silicon captures on ``Ascend950DT_9582`` and ``Ascend950PR_958b`` across
  Level 1 (single operators & 2P/8P UB-Mesh collectives), Level 2 (matching
  captured layer/subgraph boundaries & rung deltas), and Level 3 (end-to-end
  ``vllm-ascend@405617bae`` model steps).

[Invariants]
  * ``train`` and ``heldout`` splits are driven by frozen acquisition manifest
    lineage (never index order within one run); any acquisition root shared
    across splits fails closed.
  * Level-2 evaluation scores strictly against the matching captured layer or
    subgraph boundary (never full-body prediction or L3 step division).
  * Every row is scored against the shared physical tolerance before display
    rounding or recorded as blocked.
  * Source provenance hashes live ``mindie_sim/prediction/execution/*.py``
    (sorted) and emits strictly ``capture://`` URIs with zero private paths.

[Timeouts & Locks]
  Bounded in-process or ``ProcessPoolExecutor`` evaluation; zero device context
  or locks.

[Side-Effect Policy]
  Reads declared JSON captures and writes only the target ``--output`` JSON.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from mindie_sim import compose_case, graph_costs, parse_case, target_graphs
from mindie_sim.catalog.ops import comm_op, gemm_op
from mindie_sim.observations.multilevel import (
    TIME_UNIT_SCALE_NS,
    digest_file_streaming,
    evaluate_vertical_drilldown,
    exceeds_physical_tolerance,
    normalize_lineage_token,
    sanitize_capture_uri,
)
from mindie_sim.prediction.api import RecipeBook
from mindie_sim.prediction.call_builders import build_aclnn_call
from mindie_sim.prediction.efficiency import capacity_feedback
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.primitives import load_primitives
from mindie_sim.prediction.operators import (
    attention as bsa,
    hyper_connections as hc,
    mla_prolog as prolog4,
    sparse_indexer as scorer,
)
from mindie_sim.prediction.operators.attention.attention_decode import RECIPE_ID, StagedGQADecode
from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.operators.linear_attention.gdn_chunk32 import OUTPUT, PREPARE, STATE
from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mindie_sim.observations.campaign_reconcile import nearest_rank_p90  # noqa: E402

PERCENT_SCALE = 100.0
L1_MEDIAN_APE_LIMIT_PCT = 8.0
L1_P90_APE_LIMIT_PCT = 12.0
L1_CLIFF_RTOL = 0.15
L1_CLIFF_ATOL_NS = 3_000
DISPLAY_ROUND_DIGITS = 4
SAMPLE_RECORD_LIMIT = 40
SPLITS = ("train", "heldout", "unassigned")
STATIC_TRACKED_FILES = (
    "mindie_sim/backends/profiles/ascend.json",
    "mindie_sim/prediction/operators/attention/attention_decode.py",
    "mindie_sim/prediction/operators/grouped_moe/expected_grouped.py",
    "mindie_sim/prediction/operators/linear_attention/gdn_chunk32.py",
)
TRACKED_OPERATOR_PACKAGES = (
    Path("mindie_sim") / "prediction" / "operators" / "attention",
    Path("mindie_sim") / "prediction" / "operators" / "hyper_connections",
    Path("mindie_sim") / "prediction" / "operators" / "linear_attention",
    Path("mindie_sim") / "prediction" / "operators" / "mla_prolog",
    Path("mindie_sim") / "prediction" / "operators" / "sparse_indexer",
)
EXECUTION_PACKAGE_REL = Path("mindie_sim") / "prediction" / "execution"
DEFAULT_SPLIT_MANIFEST_NAMES = ("acquisition_split_manifest.json", "split_manifest.json")
MODEL_PHASES = (
    ("decode_B8_S512", "decode", 1, 511, 8, "decode_B8_S512_step_p50_ms", "decode_B8_S512_per_layer_us"),
    ("prefill_B4_S512", "prefill", 512, 0, 4, "prefill_B4_S512_device_p50_ms", "prefill_B4_S512_per_layer_us"),
)


def _tracked_source_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash static tracked modules plus live sorted execution & operator package ``*.py``."""
    exec_dir = root / EXECUTION_PACKAGE_REL
    exec_files = sorted(p.relative_to(root).as_posix() for p in exec_dir.glob("*.py"))
    if not exec_files:
        raise FileNotFoundError(f"execution_package_sources_missing:{exec_dir}")
    op_files: list[str] = []
    for pkg_rel in TRACKED_OPERATOR_PACKAGES:
        pkg_dir = root / pkg_rel
        members = sorted(p.relative_to(root).as_posix() for p in pkg_dir.glob("*.py"))
        if not members:
            raise FileNotFoundError(f"operator_package_sources_missing:{pkg_dir}")
        op_files.extend(members)
    tracked = sorted({*STATIC_TRACKED_FILES, *exec_files, *op_files})
    return {rel: digest_file_streaming(root / rel) for rel in tracked}


class _LineageSplitResolver:
    """Resolve and enforce disjoint train/heldout splits from frozen acquisition lineage."""

    def __init__(self, manifest_doc: Mapping[str, Any] | None = None) -> None:
        self._parent: dict[str, str] = {}
        self._declared_splits: dict[str, str] = {}
        self._observed_splits: dict[str, str] = {}
        if manifest_doc:
            self._load_manifest(manifest_doc)

    def _find(self, node: str) -> str:
        tok = normalize_lineage_token(node)
        root = self._parent.setdefault(tok, tok)
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[tok] != root:
            tok, self._parent[tok] = self._parent[tok], root
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        self._parent[drop] = keep
        for store, err in (
            (self._declared_splits, "producer_manifest_split_conflict"),
            (self._observed_splits, "producer_lineage_split_leakage"),
        ):
            if drop in store:
                sp = store.pop(drop)
                if keep in store and store[keep] != sp:
                    raise ModelingError(f"{err}:root={keep}:{store[keep]}:{sp}")
                store[keep] = sp

    def _assign_declared(self, token: str, split: str) -> None:
        if split not in ("train", "heldout"):
            return
        root = self._find(token)
        prev = self._declared_splits.get(root)
        if prev is not None and prev != split:
            raise ModelingError(f"producer_manifest_split_conflict:root={root}:{prev}:{split}")
        self._declared_splits[root] = split

    def _load_manifest(self, doc: Mapping[str, Any]) -> None:
        for edge in doc.get("lineage_edges") or ():
            if isinstance(edge, Mapping):
                p, c = edge.get("parent"), edge.get("child")
            elif isinstance(edge, Sequence) and not isinstance(edge, (str, bytes)) and len(edge) == 2:
                p, c = edge[0], edge[1]
            else:
                continue
            if p and c:
                self._union(str(p), str(c))
        for entry in doc.get("acquisitions") or ():
            if not isinstance(entry, Mapping):
                continue
            tokens = [
                str(t) for t in (
                    entry.get("acquisition_uid"), entry.get("acquisition_root"),
                    entry.get("raw_sha256"), entry.get("run_id"),
                    *(entry.get("parent_digests") or ()), *(entry.get("run_ids") or ()),
                ) if t and str(t).strip()
            ]
            if not tokens:
                continue
            for tok in tokens[1:]:
                self._union(tokens[0], tok)
            if sp := str(entry.get("split") or "").strip():
                self._assign_declared(tokens[0], sp)
        for key, sp in (doc.get("splits") or {}).items():
            if key and sp:
                self._assign_declared(str(key), str(sp).strip())
        for sp_name, field in (("train", "train_acquisitions"), ("heldout", "heldout_acquisitions")):
            for tok in doc.get(field) or ():
                if tok and str(tok).strip():
                    self._assign_declared(str(tok), sp_name)

    def resolve(
        self,
        record: Mapping[str, Any],
        *,
        file_doc: Mapping[str, Any],
        fallback_acq: str,
        allow_record_split: bool = True,
    ) -> tuple[str, str]:
        """Return ``(split, canonical_acquisition_root)`` enforcing single-split lineage."""
        tokens: list[str] = []
        for src in (record, file_doc):
            for field in ("acquisition_uid", "acquisition_root", "raw_sha256", "run_uid"):
                if val := src.get(field):
                    tokens.append(str(val))
            for parent in src.get("parent_digests") or ():
                if parent:
                    tokens.append(str(parent))
        if rid := record.get("run_id"):
            tokens.append(str(rid))
        if not tokens:
            if not fallback_acq:
                return "", ""
            tokens.append(fallback_acq)
        anchor = tokens[0]
        for tok in tokens[1:]:
            self._union(anchor, tok)
        root = self._find(anchor)
        resolved_split = self._declared_splits.get(root, "")
        if not resolved_split and allow_record_split:
            cand = str(record.get("split") or file_doc.get("split") or "").strip()
            if cand in ("train", "heldout"):
                resolved_split = cand
        if resolved_split:
            prev = self._observed_splits.get(root)
            if prev is not None and prev != resolved_split:
                raise ModelingError(f"producer_lineage_split_leakage:root={root}:{prev}:{resolved_split}")
            self._observed_splits[root] = resolved_split
        return resolved_split, root

    def overlap_count(self) -> int:
        # Components are checked fail-closed during resolve(); re-verify post-union.
        by_root: dict[str, set[str]] = {}
        for tok, sp in self._observed_splits.items():
            by_root.setdefault(self._find(tok), set()).add(sp)
        return sum(1 for splits in by_root.values() if len(splits) > 1)


def _load_split_manifest(captures_dir: Path, split_manifest: Path | None) -> _LineageSplitResolver:
    target = split_manifest
    if target is None:
        for name in DEFAULT_SPLIT_MANIFEST_NAMES:
            cand = captures_dir / name
            if cand.is_file():
                target = cand
                break
    if target is None:
        return _LineageSplitResolver(None)
    return _LineageSplitResolver(json.loads(target.read_text(encoding="utf-8")))


def _timing_fields(predicted: float, measured: float, *, unit: str = "us") -> dict[str, Any]:
    if not math.isfinite(measured) or measured <= 0:
        raise ModelingError("measured_duration_invalid")
    if not math.isfinite(predicted) or predicted <= 0:
        raise ModelingError("predicted_duration_invalid")
    ape = abs(predicted - measured) / measured
    scale = TIME_UNIT_SCALE_NS[unit]
    pred_ns = round(predicted * scale)
    meas_ns = round(measured * scale)
    cliff_limit_ns = max(L1_CLIFF_RTOL * float(meas_ns), float(L1_CLIFF_ATOL_NS))
    return {
        "status": "ok", "ape": ape, "ape_pct": round(ape * PERCENT_SCALE, 2),
        "pointwise_pass": not exceeds_physical_tolerance(pred_ns, meas_ns),
        "l1_cliff_pass": abs(pred_ns - meas_ns) <= cliff_limit_ns,
    }


def _stats(rows: Sequence[Mapping[str, Any]], *, level: str = "L2") -> dict[str, Any]:
    """Summarize rows using Two-Tier law for L1 and strict 10%/2us for L2/L3."""
    ok_rows = [row for row in rows if row["status"] == "ok"]
    apes = sorted(row["ape"] for row in ok_rows)
    blocked = len(rows) - len(ok_rows)
    strict_passed = sum(bool(row["pointwise_pass"]) for row in ok_rows)
    raw_medape_pct = (statistics.median(apes) * PERCENT_SCALE) if apes else None
    raw_p90_ape_pct = (nearest_rank_p90(apes) * PERCENT_SCALE) if apes else None
    medape_pct = round(raw_medape_pct, DISPLAY_ROUND_DIGITS) if raw_medape_pct is not None else None
    p90_ape_pct = round(raw_p90_ape_pct, DISPLAY_ROUND_DIGITS) if raw_p90_ape_pct is not None else None
    max_ape_pct = round(apes[-1] * PERCENT_SCALE, DISPLAY_ROUND_DIGITS) if apes else None
    mape_pct = round(statistics.fmean(apes) * PERCENT_SCALE, DISPLAY_ROUND_DIGITS) if apes else None
    if level == "L1":
        cliff_passed = sum(bool(row.get("l1_cliff_pass", row["pointwise_pass"])) for row in ok_rows)
        cliff_failed = len(apes) - cliff_passed
        median_ok = raw_medape_pct is not None and raw_medape_pct <= L1_MEDIAN_APE_LIMIT_PCT
        p90_ok = raw_p90_ape_pct is not None and raw_p90_ape_pct <= L1_P90_APE_LIMIT_PCT
        cliff_ok = cliff_failed == 0
        status = (
            "fail" if (apes and not (median_ok and p90_ok and cliff_ok))
            else ("insufficient_evidence" if blocked or not apes else "pass")
        )
        return {
            "status": status, "gate_tier": "L1_two_tier",
            "total": len(rows), "count": len(apes), "passed": cliff_passed, "failed": cliff_failed,
            "strict_10pct_passed": strict_passed, "explicitly_blocked": blocked,
            "mape_pct": mape_pct, "medape_pct": medape_pct, "p90_ape_pct": p90_ape_pct,
            "max_ape_pct": max_ape_pct,
            "cliff_pass_rate": round(cliff_passed / len(apes), DISPLAY_ROUND_DIGITS) if apes else None,
            "two_tier_thresholds": {
                "median_ape_limit_pct": L1_MEDIAN_APE_LIMIT_PCT,
                "p90_ape_limit_pct": L1_P90_APE_LIMIT_PCT,
                "cliff_rtol_pct": round(L1_CLIFF_RTOL * PERCENT_SCALE, 2),
                "cliff_atol_us": round(L1_CLIFF_ATOL_NS / TIME_UNIT_SCALE_NS["us"], 2),
            },
            "two_tier_checks": {"median_pass": median_ok, "p90_pass": p90_ok, "cliff_pass": cliff_ok},
        }
    failed = len(apes) - strict_passed
    return {
        "status": "fail" if failed else ("insufficient_evidence" if blocked or not apes else "pass"),
        "gate_tier": "strict_10pct_2us",
        "total": len(rows), "count": len(apes), "passed": strict_passed, "failed": failed,
        "explicitly_blocked": blocked, "mape_pct": mape_pct,
        "medape_pct": medape_pct, "p90_ape_pct": p90_ape_pct, "max_ape_pct": max_ape_pct,
    }


def _record_status(record: Mapping[str, Any], measured: float, split: str) -> str:
    if split not in ("train", "heldout"):
        return "acquisition_split_unassigned"
    if record.get("status", "ok") != "ok":
        return "capture_row_unsuccessful"
    if not math.isfinite(measured) or measured <= 0:
        return "measured_duration_invalid"
    return ""


def _operator_row(resolver, r, doc, *, sku, name, family, fallback, uri, measured, predict):
    split, root = resolver.resolve(r, file_doc=doc, fallback_acq=fallback)
    row = {"id": f"{sku}:{name}", "family": family, "sku": sku, "acquisition_root": root,
           "capture_uri": uri, "measured_us": round(measured, 3) if math.isfinite(measured) else None}
    reason = _record_status(r, measured, split)
    if not reason:
        try:
            resolved_family, predicted = predict()
            row.update(_timing_fields(predicted, measured))
            return split, {**row, "family": resolved_family, "predicted_us": round(predicted, 3)}
        except ModelingError as exc:
            reason = exc.reason
        except (KeyError, IndexError, TypeError, ValueError):
            reason = "unsupported_capture_geometry"
    return split or "unassigned", {**row, "status": "blocked", "reason": reason, "predicted_us": None}


_PRIMITIVE_CACHE: dict[str, Any] = {}


def _calibrated_evaluation(cs, recipes: RecipeBook | None = None) -> EvaluationContext:
    sku = str(getattr(getattr(cs, "machine", None), "sku", "") or "").strip()
    if sku and sku not in _PRIMITIVE_CACHE:
        try:
            _PRIMITIVE_CACHE[sku] = load_primitives(sku)
        except (ModelingError, FileNotFoundError, KeyError, ValueError):
            _PRIMITIVE_CACHE[sku] = None
    pt = _PRIMITIVE_CACHE.get(sku)
    book = recipes if recipes is not None else RecipeBook.from_mapping(cs.runtime.kernels)
    return EvaluationContext.create(cs, book, primitive_table=pt)


def _eval_collective(cs_comm, coll: str, world_size: int, payload_bytes: int) -> float:
    w = max(2, int(world_size))
    if coll == "AllReduce":
        op = comm_op(
            op_id=f"allreduce_w{w}_{payload_bytes}", deps=(), stage=0, layer=0, requests=("r0",),
            family="allreduce", phase="prefill", recipe="ring_allreduce",
            link_bytes=2.0 * (w - 1) / w * payload_bytes, shape=(1, max(1, payload_bytes // 2)),
            metadata=(("group_axis", "tp"), ("collective", "all_reduce"), ("steps", 2 * (w - 1))),
        )
    else:
        op = comm_op(
            op_id=f"alltoall_w{w}_{payload_bytes}", deps=(), stage=0, layer=0, requests=("r0",),
            family="alltoall", phase="prefill", recipe="expert_dispatch",
            link_bytes=(w - 1) / w * payload_bytes, shape=(1, max(1, payload_bytes // 2)),
            metadata=(("group_axis", "tp"), ("collective", "all_to_all")),
        )
    rep = graph_costs(cs_comm, {0: [op]}, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs_comm))
    return float(rep["duration_ms"]) * 1000.0


def _eval_planned(cs, op: OpSpec, builder_fn) -> float:
    rb = RecipeBook(recipes={op.recipe: PlannedRecipe(builder_fn)})
    rep = graph_costs(cs, {0: [op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs, rb))
    return float(rep["duration_ms"]) * 1000.0


_HC_VARIANT_RECIPES = {
    "hc_pre_v3": "npu_hc_pre_v3_arch35",
    "hc_pre_v2": "npu_hc_pre_v2_arch35",
    "hc_post": "npu_hc_post_arch35",
    "mhc_post_collapse": "mhc_post_collapse_norm_arch35",
    "text_gated_residual": "text_gated_residual_arch35",
}


def _make_hc_op(
    *, variant: str, op_id: str, rows: int, hidden_size: int,
    hc_mult: int = 4, hc_lowrank: int = 16,
) -> OpSpec:
    kind = str(variant).strip().lower()
    recipe = _HC_VARIANT_RECIPES[kind]
    shape = (max(1, int(rows)), max(1, int(hc_mult)), max(1, int(hidden_size)))
    meta = (
        ("hc_geometry", {
            "variant": kind, "rows": shape[0], "hidden_size": shape[2],
            "hc_mult": shape[1], "hc_lowrank": max(1, int(hc_lowrank)),
        }),
    )
    return OpSpec(
        op_id=op_id, deps=(), family="mhc", recipe=recipe, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", source_boundary="kernel", metadata=meta,
    )


def _eval_fia(cs, B: int, S: int, Hq: int, Hkv: int, D: int, seq_kv: int, *, window: int = 0) -> float:
    gqa_meta = {
        "requests": tuple((S, max(0, seq_kv - S)) for _ in range(B)),
        "query_heads": Hq, "kv_heads": Hkv, "head_dim": D,
        "query_dtype": "bf16", "kv_dtype": "bf16", "causal": True, "window": window, "cp": 1,
    }
    shape = (B * S, seq_kv, Hq, D)
    op = OpSpec(
        op_id="fia_probe", family="attention", recipe=RECIPE_ID, phase="decode", stage=0, layer=0,
        requests=tuple(f"r{i}" for i in range(B)), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape, dtype="bf16", metadata=(("gqa_decode", gqa_meta),),
    )
    cores = int(cs.machine.dies[0].compute_units)
    key_tile = 32 if D >= 512 else (64 if D >= 256 else 128)
    rb = RecipeBook(recipes={RECIPE_ID: StagedGQADecode(key_tile=key_tile, target_cores=cores, kv_mapping="fai_host_arch35")})
    return float(graph_costs(cs, {0: [op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs, rb))["duration_ms"]) * 1000.0


def _eval_gdn_chunk32(cs, T: int, Hk: int, Hv: int, K: int, V: int) -> float:
    params = {"lengths": (T,), "heads": Hv, "key_heads": Hk, "dk": K, "dv": V, "chunk": 32}
    p_op = PREPARE.make_op(params)
    s_op = replace(STATE.make_op(params), deps=(p_op.op_id,))
    o_op = replace(OUTPUT.make_op(params), deps=(p_op.op_id, s_op.op_id))
    rb = RecipeBook(recipes={p_op.recipe: PlannedRecipe(PREPARE.build), s_op.recipe: PlannedRecipe(STATE.build), o_op.recipe: PlannedRecipe(OUTPUT.build)})
    return float(graph_costs(cs, {0: [p_op, s_op, o_op]}, recipes=rb, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs, rb))["duration_ms"]) * 1000.0


def _single_operator(cs_bf, cs_mx, gmm_rb, dom, name, sh) -> tuple[str, float]:
    if cs_bf is None or cs_mx is None:
        raise ModelingError("single_operator_sku_unsupported")
    if dom == "domain2_bf16_gemm":
        cd = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": sh[0], "n": sh[1], "k": sh[2]})
        return "gemm_qmm", float(graph_costs(cs_bf, {0: cd.ops}, recipes=cd.recipes, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs_bf, cd.recipes))["duration_ms"]) * 1000.0
    if dom == "domain2_mxfp8_qmm":
        op = gemm_op(op_id=name, deps=(), stage=0, layer=0, requests=("r0",), m=sh[0], n=sh[1], k=sh[2], quant=cs_mx.model.quant, category="linear_param", phase="prefill")
        return "gemm_qmm", float(graph_costs(cs_mx, {0: [op]}, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs_mx))["duration_ms"]) * 1000.0
    if dom == "domain3_moe_gmm" and ("uniform" in name or "zero_tail" in name):
        act_e = sh["E_local"] // 2 if "zero_tail" in name else sh["E_local"]
        q_spec = replace(cs_mx.model.quant.for_category("expert"), weight_format="mxfp4") if "MXFP4" in name else cs_mx.model.quant
        meta = (("compute_format", "fp8_e4m3"), ("matrix_mapping_domain", "aggregate_experts"), ("routing_basis", "probability_moments"), ("expected_active_experts", float(act_e)), ("expected_rows", float(sh["M_total"])), ("matrix_groups", act_e))
        op = gemm_op(op_id=name, deps=(), stage=0, layer=0, requests=("r0",), m=sh["M_total"], n=sh["N"], k=sh["K"], quant=q_spec, category="expert", phase="prefill", recipe="aclnn_grouped_matmul", family="grouped_gemm", activation_dtype="fp8_e4m3", weight_count=act_e, metadata=meta)
        return "moe_gmm", float(graph_costs(cs_mx, {0: [op]}, recipes=gmm_rb, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs_mx, gmm_rb))["duration_ms"]) * 1000.0
    if dom == "domain4_fia" and "fia_v1" in name:
        return "fia_decode", _eval_fia(cs_bf, sh["B"], sh["S"], sh["Hq"], sh["Hkv"], sh["D"], sh["seq_kv"], window=sh.get("swa", 0))
    if dom == "probe1_c8_sfa":
        op = bsa.make_kv_quant_sfa_c8_op(op_id=name, query_tokens=sh["B"] * sh["S"], num_heads=sh["H_q"], kv_lora_rank=512, rope_dim=64, selected_tokens=sh["topk"], batch_size=sh["B"], shared_prefix=("shared" in name or sh.get("idx_mode") == "s6_same_idx"))
        return "c8_kv_quant_sfa", _eval_planned(cs_bf, op, bsa.build)
    if dom == "domain4_gbsa":
        blk = sh["block_shape"][-1]
        op = bsa.make_gbsa_op(op_id=name, query_tokens=sh["B"] * sh["S_q"], num_heads=sh["Hq"], num_kv_heads=sh["Hkv"], head_dim=sh["hd"], selected_tokens=blk * sh["topk_blocks"], block_size=blk, batch_size=sh["B"], shared_prefix=False)
        return "gbsa", _eval_planned(cs_bf, op, bsa.build)
    if dom == "domain4_gbsa_sparse_gqa":
        B, S, Hq, Hkv, hd, blk, nblk = sh
        op = bsa.make_gbsa_op(op_id=name, query_tokens=B * S, num_heads=Hq, num_kv_heads=Hkv, head_dim=hd, selected_tokens=blk * nblk, block_size=blk, batch_size=B, shared_prefix=True)
        return "gbsa", _eval_planned(cs_bf, op, bsa.build)
    if dom == "domain4_indexer":
        op = scorer.make_step5_region_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["ctx"], index_heads=sh["H"], index_dim=sh["D"], region_block_size=sh["region_size"], topk_regions=sh["topk"], batch_size=sh["B"]) if "step5" in name else scorer.make_msa_index_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["ctx"], index_heads=sh["Hq"], index_dim=sh["D"], return_choice=0, batch_size=sh["B"])
        return "sparse_index_scorer", _eval_planned(cs_bf, op, scorer.build)
    if dom in ("probe2_mhc_arch35", "probe2_mhc", "probe2_qwen4exp_hc"):
        if "sinkhorn" in name.lower():
            raise ModelingError("standalone_sinkhorn_geometry_unsupported")
        M, S_hc, C, lr = (sh[0], sh[1], sh[2], 16) if isinstance(sh, list) else (sh["M"], sh.get("hc_mult", sh.get("channels", 4)), sh.get("C", sh.get("H", 2560)), sh.get("low_rank", 16))
        var = "text_gated_residual" if "text_gated" in name else ("hc_post" if "post" in name else "hc_pre_v3")
        return "mhc_and_gated_residual", _eval_planned(cs_bf, _make_hc_op(variant=var, op_id=name, rows=M, hidden_size=C, hc_mult=S_hc, hc_lowrank=lr), hc.build)
    if dom == "probe3_mla_prolog":
        M, D_hid, ql, H = sh
        return "mla_prolog_v3_v4", _eval_planned(cs_bf, prolog4.make_op(op_id=name, tokens=M, hidden_size=D_hid, num_heads=H, q_lora_rank=ql, weight_quant_mode=0, kv_cache_quant_mode=0), prolog4.build)
    if dom == "domain5_gdn_chunks" and "chunk32" in name:
        return "gdn_chunk32", _eval_gdn_chunk32(cs_bf, sh["T"], sh["Hk"], sh["Hv"], sh["K"], sh["V"])
    raise ModelingError("single_operator_variant_unsupported")


def _sequence_operator(cs_dt, dom, name, sh) -> tuple[str, float]:
    if dom == "shard0_fia_decode" and sh.get("kv_dtype", "bf16") == "bf16":
        return "fia_decode", _eval_fia(cs_dt, sh["B"], sh["S"], sh["Hq"], sh["Hkv"], sh["D"], sh["seq_kv"], window=sh.get("swa_window", 0))
    if dom == "shard1_c8_sfa":
        return "c8_kv_quant_sfa", _eval_planned(cs_dt, bsa.make_kv_quant_sfa_c8_op(op_id=name, query_tokens=sh["B"] * sh["S"], num_heads=sh["Hq"], kv_lora_rank=512, rope_dim=64, selected_tokens=sh["topk"], batch_size=sh["B"], shared_prefix=False), bsa.build)
    if dom == "shard2_minimax_msa":
        return "sparse_index_scorer", _eval_planned(cs_dt, scorer.make_msa_index_score_op(op_id=name, query_tokens=sh["B"] * sh["S"], context_tokens=sh["seq_kv"], index_heads=sh["Hq"], index_dim=sh["D"], return_choice=1, batch_size=sh["B"]), scorer.build)
    if dom == "shard2_step5_region_score":
        return "sparse_index_scorer", _eval_planned(cs_dt, scorer.make_step5_region_score_op(op_id=name, query_tokens=sh["B"], context_tokens=sh["seq_kv"], index_heads=sh["H"], index_dim=sh["D"], region_block_size=sh["region_size"], topk_regions=sh["topk"], batch_size=sh["B"]), scorer.build)
    if dom == "shard3_gdn_chunk32":
        return "gdn_chunk32", _eval_gdn_chunk32(cs_dt, sh["T"], sh["Hk"], sh["Hv"], sh["D"], sh["D"])
    if dom == "shard3_mhc":
        return "mhc_and_gated_residual", _eval_planned(cs_dt, _make_hc_op(variant="hc_post" if "post" in name else "hc_pre_v3", op_id=name, rows=sh["M"], hidden_size=sh["C"], hc_mult=sh["hc_mult"]), hc.build)
    if dom in ("shard3_mla_prolog_v3", "shard3_mla_prolog_v4"):
        return "mla_prolog_v3_v4", _eval_planned(cs_dt, prolog4.make_op(op_id=name, tokens=sh["M"], hidden_size=sh["D_hid"], num_heads=sh["H"], q_lora_rank=sh["q_lora"], weight_quant_mode=3 if "v4" in dom else 0, kv_cache_quant_mode=1 if "v4" in dom else 0), prolog4.build)
    if dom == "shard3_text_gated_residual":
        return "mhc_and_gated_residual", _eval_planned(cs_dt, _make_hc_op(variant="text_gated_residual", op_id=name, rows=sh["M"], hidden_size=sh["H"], hc_mult=sh["channels"], hc_lowrank=sh["low_rank"]), hc.build)
    raise ModelingError("sequence_operator_variant_unsupported")


def _hotspot_operator(cs, record, name, shape) -> tuple[str, float]:
    if record.get("group") != "Roofline_Cube" or "bf16" not in name:
        raise ModelingError("hotspot_operator_variant_unsupported")
    call = build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": shape["M"], "n": shape.get("N", 4096), "k": shape.get("K", 4096)})
    report = graph_costs(cs, {0: call.ops}, recipes=call.recipes, include_submission=False,
                         isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs, call.recipes))
    return "gemm_qmm", float(report["duration_ms"]) * 1000.0


def _collect_collective_rows(doc, collect):
    comm_cases = {}

    def predict(sku, coll, world_size, payload):
        if payload <= 0 or coll not in ("AllReduce", "AllToAll"):
            raise ModelingError("collective_geometry_unsupported")
        key = sku, world_size
        if key not in comm_cases:
            comm_cases[key] = parse_case(compose_case("qwen3.8-27b", hardware=sku, tp=world_size, ep=1,
                                                    num_devices=world_size, layers=1, quant="none"))
        return "ub_mesh_collectives", _eval_collective(comm_cases[key], coll, world_size, payload)

    for row in doc.get("collective_measurements", ()):
        payload, coll = int(row.get("bytes", 0)), row.get("collective", "")
        world_size, sku = int(row.get("world_size", 2)), row.get("sku", "Ascend950DT_9582")
        name = f"{coll}_w{world_size}_{payload}"
        capture = row.get("source_ref")
        uri = (sanitize_capture_uri(capture) if isinstance(capture, str) and capture.startswith("capture://")
               else f"capture://cann93-a5-microbench/{sku}/ub_mesh/{name}")
        collect(row, doc, sku=sku, name=name, family="ub_mesh_collectives", fallback=f"d1:comm:{sku}:{name}",
                uri=uri,
                measured=_measurement(row, "eager_p50_us", "eager_p50_us"), predict=partial(predict, sku, coll, world_size, payload))


def _measurement(row: Mapping[str, Any], field: str, fallback: str = "cold_p50_us") -> float:
    metrics = row.get("metrics") or {}
    try:
        return float(metrics.get(field) or metrics.get(fallback) or 0)
    except (TypeError, ValueError):
        return math.nan


def evaluate_level1(captures_dir: Path, resolver: _LineageSplitResolver) -> dict[str, Any]:
    skus = ("Ascend950DT_9582", "Ascend950PR_958b")
    cases_bf = {s: parse_case(compose_case("qwen3.8-27b", hardware=s, tp=1, ep=1, num_devices=1, layers=1, quant="none")) for s in skus}
    cases_mx = {s: parse_case(compose_case("qwen3.8-27b", hardware=s, tp=1, ep=1, num_devices=1, layers=1, quant="w8a8_mxfp8")) for s in skus}
    gmm_rb = RecipeBook(recipes={"aclnn_grouped_matmul": ExpectedGroupedMatmul(SteadyMatmul())})
    d1 = json.loads((captures_dir / "cann93_single_op_measurements.json").read_text(encoding="utf-8"))
    d2 = json.loads((captures_dir / "cann93_seq512_to_400k_measurements.json").read_text(encoding="utf-8"))
    d4 = json.loads((captures_dir / "task1_kimi_k31_silicon_hotspots.json").read_text(encoding="utf-8"))
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    def collect(*args, **kwargs):
        split, row = _operator_row(resolver, *args, **kwargs)
        records[split].append(row)

    for sku, rows in (d1.get("single_op_measurements") or {}).items():
        for index, row in enumerate(rows):
            dom, name, shape = row.get("domain", ""), row.get("op_name") or f"row-{index}", row.get("shape")
            field = "rts_stream_saturated_us" if dom in ("domain2_bf16_gemm", "domain2_mxfp8_qmm", "domain3_moe_gmm") else "cold_p50_us"
            collect(row, d1, sku=sku, name=name, family=dom, fallback=f"d1:{sku}:{dom}:{name}",
                    uri=f"capture://cann93-a5-microbench/{sku}/{dom}/{name}", measured=_measurement(row, field),
                    predict=partial(_single_operator, cases_bf.get(sku), cases_mx.get(sku), gmm_rb, dom, name, shape))
    _collect_collective_rows(d1, collect)

    sku = skus[0]
    cs_dt = cases_bf[sku]
    sequence_root = str(d2.get("acquisition_uid") or d2.get("run_uid")
                        or "d2:" + digest_file_streaming(captures_dir / "cann93_seq512_to_400k_measurements.json"))
    for index, row in enumerate(d2.get("records", ())):
        dom, name, shape = row.get("domain", ""), row.get("op_name") or f"row-{index}", row.get("shape")
        collect(row, d2, sku=sku, name=name, family=dom, fallback=sequence_root,
                uri=f"capture://cann93-a5-seq512-400k/{sku}/{dom}/{name}", measured=_measurement(row, "cold_p50_us"),
                predict=partial(_sequence_operator, cs_dt, dom, name, shape))
    for index, row in enumerate(d4.get("records", ())):
        name, shape = row.get("op_name") or f"row-{index}", row.get("shape") or {}
        collect(row, d4, sku=sku, name=f"kimi_{name}", family="gemm_qmm", fallback="d4:kimi_hotspots",
                uri=f"capture://cann93-a5-kimi-k31-hotspots/{sku}/Roofline_Cube/{name}", measured=_measurement(row, "cold_p50_us"),
                predict=partial(_hotspot_operator, cs_dt, row, name, shape))

    out = {}
    for split, rows in records.items():
        by_family = {}
        for row in rows:
            by_family.setdefault(row["family"], []).append(row)
        report_rows = [_display_row(row) for row in rows]
        families = {f: _stats(v, level="L1") for f, v in sorted(by_family.items())}
        summary = _stats(rows, level="L1")
        all_families_pass = bool(families and all(fs.get("status") == "pass" for fs in families.values()))
        summary["all_families_pass"] = all_families_pass
        if summary.get("status") == "pass" and not all_families_pass:
            summary["status"] = "fail"
        out[split] = {
            "summary": summary,
            "families": families,
            "ids": [row["id"] for row in rows], "records": report_rows,
            "sample_records": report_rows[:SAMPLE_RECORD_LIMIT],
        }
    return out


def _display_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "ape"}


def _capture_duration(raw: Any) -> float:
    try:
        duration = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    return duration if math.isfinite(duration) and duration > 0 else 0.0


def _eval_l2_boundary(
    cs, r: Mapping[str, Any], *, ph: str, tok: int, ctx: int, seqs: int,
    measured: float, l3_uri: str, l3_acq_root: str, l2_split: str, l2_acq_root: str,
) -> dict[str, Any]:
    """Score L2 strictly against its own independent capture, root, and split."""
    indep_l2_uri = str(r.get("independent_l2_capture_uri") or "").strip()
    if (not indep_l2_uri or indep_l2_uri == l3_uri
            or not l2_acq_root or l2_acq_root == l3_acq_root
            or l2_split not in ("train", "heldout")
            or measured <= 0):
        return {
            "status": "missing_evidence", "reason": "independent_l2_block_capture_absent",
            "split": l2_split if l2_split in ("train", "heldout") else "",
            "acquisition_root": l2_acq_root if l2_acq_root != l3_acq_root else "",
            "capture_uri": "", "boundary_kind": "missing_block_capture", "l2_layers": 0,
            "measured_block_us": None, "predicted_block_us": None, "ape": None, "ape_pct": None,
        }
    layer_indices = r.get("l2_layer_indices")
    if layer_indices is None and r.get("l2_layer_index") is not None:
        layer_indices = [int(r["l2_layer_index"])]
    if layer_indices is not None:
        selected_plans = tuple(cs.model.layer_plans[int(i)] for i in layer_indices)
        cs_l2 = replace(cs, model=replace(cs.model, layer_plans=selected_plans))
        l2_layers = len(selected_plans)
        l2_graphs = target_graphs(cs_l2, phase=ph, tokens=tok, context=ctx, sequences=seqs, scope="body")
    else:
        raw_layers = r.get("l2_num_layers", r.get("l2_layers"))
        if raw_layers is not None:
            l2_layers = max(1, min(len(cs.model.layer_plans), int(raw_layers)))
        else:
            l2_layers = 1
        cs_l2 = cs
        l2_graphs = target_graphs(cs, phase=ph, tokens=tok, context=ctx, sequences=seqs, scope="body", layer_limit=l2_layers)

    if op_families := r.get("l2_op_families"):
        allowed = {str(f) for f in op_families}
        filtered: dict[int, list[OpSpec]] = {}
        for rep_id, ops in l2_graphs.items():
            kept = [op for op in ops if op.family in allowed]
            kept_ids = {op.op_id for op in kept}
            filtered[rep_id] = [replace(op, deps=tuple(d for d in op.deps if d in kept_ids)) for op in kept]
        l2_graphs = filtered

    l2_rep = graph_costs(cs_l2, l2_graphs, external_inputs={"analysis_input": 0}, include_submission=False, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs_l2))
    l2_pred_us = float(l2_rep["duration_ms"]) * 1000.0
    return {
        **_timing_fields(l2_pred_us, measured),
        "split": l2_split,
        "acquisition_root": l2_acq_root,
        "capture_uri": indep_l2_uri,
        "boundary_kind": str(r.get("l2_boundary_kind") or "periodic_block"),
        "l2_layers": l2_layers,
        "measured_block_us": round(measured, 2),
        "predicted_block_us": round(l2_pred_us, 2),
    }


def _eval_model_run(task: tuple[dict[str, Any], str, str, str, str, str]) -> dict[str, Any]:
    r, resolved_split, acq_root, l2_split, l2_acq_root, l3_blocked = task
    rid, mod, rung, sku, L = r["run_id"], r["model_id"], r["rung"], r.get("sku", "Ascend950DT_9582"), int(r["num_layers"])
    rc = r.get("resolved_compilation_and_ascend_config") or {}
    pt = rc.get("parallel_topology") or {}
    dp = int(r.get("dp") or pt.get("data_parallel_size") or 1)
    tp = int(r.get("tp") or pt.get("tensor_parallel_size") or 2)
    pcp = int(r.get("pcp") or pt.get("prefill_context_parallel_size") or 1)
    dcp = int(r.get("dcp") or pt.get("decode_context_parallel_size") or 1)
    world = dp * tp * max(1, pcp)
    ep = world if bool(r.get("ep") if r.get("ep") is not None else pt.get("enable_expert_parallel", False)) else 1
    track = (
        "proposed" if ("unmerged_pr" in rung or "proposed" in rung)
        else ("eager" if ("raw_eager" in rung or "eager" in rung) else "shipping")
    )
    if not l3_blocked and track == "eager":
        l3_blocked = "raw_eager_step_boundary_unsupported"
    fx = r.get("fx_compilation_summary") or {}
    pre_ops = (fx.get("pre_split_full_graph_summary") or {}).get("op_counts") or {}
    orig_L = int(r.get("orig_layers") or L)
    n_unq = int(pre_ops.get("vllm.unquantized_gemm", 0)) + int(pre_ops.get("vllm.unquantized_gemm.default", 0))
    n_qmm = int(pre_ops.get("npu.npu_quant_matmul", 0)) + int(pre_ops.get("npu.npu_quant_matmul.default", 0))
    n_mxq = int(pre_ops.get("npu.npu_dynamic_mx_quant", 0)) + int(pre_ops.get("npu.npu_dynamic_mx_quant.default", 0))
    mso = bool(rc.get("multistream_overlap_shared_expert", False))
    is_hybrid_60 = n_unq > 0 and n_mxq > 0 and orig_L >= 60
    if (n_unq > 0 and n_qmm == 0) or (
        n_unq > 2 * n_qmm and orig_L <= 48
        and capacity_feedback(compose_case(mod, hardware=sku, dp=dp, tp=tp, ep=ep, num_devices=world, layers=L, quant="none"))["is_vram_feasible"]
    ):
        quant_arg = "none"
    elif is_hybrid_60:
        quant_arg = "w8a8_mxfp8"
    else:
        quant_arg = None
    cg_mode = str(rc.get("cudagraph_mode") or "FULL_AND_PIECEWISE").upper()
    cg_max = max(rc.get("cudagraph_capture_sizes") or [8]) if cg_mode != "NONE" else 0
    t, fx_sub = r.get("step_and_layer_timings") or {}, fx.get("num_compiled_fx_subgraphs")
    l3_uri = str(r.get("capture_uri") or f"capture://cann93-a5-vllm-subgraphs/20261119/{rid}")
    raw_l2_timings = r.get("independent_l2_timings") or r.get("l2_timings")
    l2_timings = raw_l2_timings if isinstance(raw_l2_timings, Mapping) else {}
    scalar_l2 = r.get("independent_l2_measured_us", r.get("l2_measured_block_us"))
    has_l2_capture = bool(r.get("independent_l2_capture_uri"))

    phases: dict[str, Any] = {}
    for ptag, ph, tok, ctx, seqs, mkey, lkey in MODEL_PHASES:
        meas = _capture_duration(t.get(mkey))
        step_reason = l3_blocked or ("step_duration_invalid" if not meas else "")
        phase_l2_measured = _capture_duration(l2_timings.get(lkey, scalar_l2))
        if step_reason:
            phases[ptag] = {"phase": ph, "status": "blocked", "reason": step_reason}
            if not (has_l2_capture and (lkey in l2_timings or scalar_l2 is not None and (lkey in t or mkey in t))):
                continue
        sched_toks = tok * seqs
        cap_mode = (
            "none" if (cg_mode == "NONE" or track == "eager")
            else ("piecewise" if (cg_mode == "PIECEWISE" or "piecewise" in rung or sched_toks > cg_max) else "full")
        )
        use_fmoe = mso and is_hybrid_60 and ph == "prefill"
        sp_arg = None if (track != "eager" and (ep == 1 or use_fmoe or "vllm.sequence_parallel_chunk_impl" in pre_ops)) else False
        excl = ([] if use_fmoe else (["cann_megamoe_fusion"] if rc.get("mega_moe_max_tokens", 0) == 0 else [])) + (["dynamic_mc2_dispatch", "tp_mc2_overlap"] if rc.get("enable_fused_mc2", 1) == 0 else [])
        raw = compose_case(
            mod, hardware=sku, feature_track=track, _stack_phase=ph, dp=dp, tp=tp, ep=ep, dcp=dcp, pcp=pcp,
            sp=sp_arg, num_devices=world, layers=L, quant=quant_arg, graph_capture_mode=cap_mode,
            enable_npugraph_ex=bool(rc.get("enable_npugraph_ex", False)) if cap_mode == "full" else False,
            multistream_overlap_shared_expert=mso,
            moe_comm_mode=None if use_fmoe else ("all2all" if rc.get("enable_fused_mc2", 1) == 0 else None),
            exclude_optimizations=tuple(excl),
        )
        if cap_mode == "piecewise":
            raw["runtime"]["mixed_mode"] = "piecewise"
            raw["runtime"]["splitting_ops"] = ["attention", "attention_qk", "attention_pv", "attention_softmax", "attention_merge", "mla", "gdn", "gdn_prepare", "gdn_state", "gdn_output", "recurrent_vector", "kda", "kda_recurrent"]
        cs = parse_case(raw)
        l2_block = _eval_l2_boundary(
            cs, r, ph=ph, tok=tok, ctx=ctx, seqs=seqs, measured=phase_l2_measured,
            l3_uri=l3_uri, l3_acq_root=acq_root,
            l2_split=l2_split, l2_acq_root=l2_acq_root,
        )
        phase_result = phases.setdefault(ptag, {"phase": ph})
        phase_result.update(l2_block=l2_block, fx_post_split_subgraphs=fx_sub, predicted_replays=None)
        if step_reason:
            continue
        try:
            rep = graph_costs(cs, target_graphs(cs, phase=ph, tokens=tok, context=ctx, sequences=seqs, scope="model"), external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, fast_span_only=True, evaluation=_calibrated_evaluation(cs))
            pred_ms = float(rep["duration_ms"])
            phase_result.update(_timing_fields(pred_ms, meas, unit="ms"))
        except ModelingError as exc:
            phase_result.update(status="blocked", reason=exc.reason)
            continue
        scale = TIME_UNIT_SCALE_NS["ms"]
        # This capture export supplies step timings and independent component
        # timings. It contains no component census measured inside the step.
        drilldown = evaluate_vertical_drilldown(round(pred_ms * scale), round(meas * scale), ())
        phase_result.update(
            cap_mode=cap_mode, measured_ms=round(meas, 3), predicted_ms=round(pred_ms, 3),
            predicted_replays=0 if cap_mode == "none" else max(1, int(rep["execution"]["submission"]["graph_replays"]) // max(1, dp)),
            vertical_drilldown=drilldown,
        )
    unscoped_l2 = None
    if has_l2_capture and scalar_l2 is not None and not any("l2_block" in result for result in phases.values()):
        unscoped_l2 = {
            "id": f"{rid}:layer_block", "run_id": rid, "model_id": mod, "rung": rung, "phase": None,
            "acquisition_root": l2_acq_root, "capture_uri": str(r["independent_l2_capture_uri"]),
            "boundary_kind": str(r.get("l2_boundary_kind") or "periodic_block"), "l2_layers": None,
            "status": "missing_evidence", "reason": "independent_l2_phase_absent",
            "measured_block_us": _capture_duration(scalar_l2) or None, "predicted_block_us": None,
            "ape_pct": None, "pointwise_pass": None,
        }
    return {
        "run_id": rid, "model_id": mod, "rung": rung, "layers": L,
        "split": resolved_split, "acquisition_root": acq_root, "l3_blocked": l3_blocked,
        "topology": f"dp{dp}_tp{tp}_ep{ep}_pcp{pcp}_dcp{dcp}",
        "capture_uri": l3_uri, "phases": phases, "unscoped_l2": unscoped_l2, "l2_split": l2_split,
    }


def build_report(
    captures_dir: Path, *, split_manifest: Path | None = None, workers: int = 1, root: Path = ROOT,
) -> dict[str, Any]:
    """Assemble the complete 3-level accuracy report with lineage-enforced splits."""
    t0 = time.perf_counter()
    resolver = _load_split_manifest(captures_dir, split_manifest)
    l1 = evaluate_level1(captures_dir, resolver)
    d3 = json.loads((captures_dir / "cann93_model_subgraphs_and_perf.json").read_text(encoding="utf-8"))
    run_tasks: list[tuple[dict[str, Any], str, str, str, str, str]] = []
    l2: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    l3: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for mv in (d3.get("models") or {}).values():
        for r in (mv.get("runs") or {}).values():
            sp, acq_root = resolver.resolve(r, file_doc=d3, fallback_acq=str(r.get("acquisition_uid") or r["run_id"]))
            reason = _record_status(r, 1, sp)
            if not reason:
                if r.get("status") != "ok":
                    reason = "capture_row_unsuccessful"
                elif r.get("rung") in ("raw_eager", "eager") or "raw_eager" in str(r.get("rung") or ""):
                    reason = "raw_eager_step_boundary_unsupported"
            if reason:
                l3[sp or "unassigned"].append({
                    "id": r["run_id"], "run_id": r["run_id"], "acquisition_root": acq_root,
                    "status": "blocked", "reason": reason,
                })
            l2_probe = {
                "acquisition_uid": r.get("l2_acquisition_uid"),
                "acquisition_root": r.get("l2_acquisition_root"),
                "raw_sha256": r.get("l2_raw_sha256"),
                "parent_digests": r.get("l2_parent_digests"),
                "split": r.get("l2_split"),
            }
            l2_sp, l2_root = resolver.resolve(l2_probe, file_doc={}, fallback_acq="")
            if not reason or r.get("independent_l2_capture_uri"):
                run_tasks.append((r, sp or "unassigned", acq_root, l2_sp, l2_root, reason))
    if workers > 1 and len(run_tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(run_tasks))) as pool:
            run_results = list(pool.map(_eval_model_run, run_tasks))
    else:
        run_results = [_eval_model_run(task) for task in run_tasks]

    fx_matches = 0
    for rr in run_results:
        sp = rr["split"]
        if unscoped_l2 := rr["unscoped_l2"]:
            l2[rr["l2_split"] or "unassigned"].append(unscoped_l2)
        for ptag, pv in rr["phases"].items():
            if blk := pv.get("l2_block"):
                l2_bucket = blk["split"] if blk["split"] in ("train", "heldout") else sp
                l2[l2_bucket].append({
                    **{key: value for key, value in blk.items() if key != "split"},
                    "id": f"{rr['run_id']}:{ptag}:layer_block", "run_id": rr["run_id"],
                    "model_id": rr["model_id"], "rung": rr["rung"], "phase": pv["phase"],
                    "fx_post_split_subgraphs": pv["fx_post_split_subgraphs"], "predicted_replays": pv["predicted_replays"],
                    "pointwise_pass": blk.get("pointwise_pass"),
                })
            if pv["status"] == "blocked":
                if not rr["l3_blocked"]:
                    l3[sp].append({"phase": pv["phase"], "status": "blocked", "reason": pv["reason"],
                                   "id": f"{rr['run_id']}:{ptag}", "run_id": rr["run_id"],
                                   "acquisition_root": rr["acquisition_root"], "capture_uri": rr["capture_uri"]})
                continue
            if pv["cap_mode"] == "piecewise" and pv["fx_post_split_subgraphs"] is not None and int(pv["fx_post_split_subgraphs"]) == pv["predicted_replays"]:
                fx_matches += 1
            l3[sp].append({
                "id": f"{rr['run_id']}:{ptag}", "run_id": rr["run_id"], "model_id": rr["model_id"], "rung": rr["rung"],
                "acquisition_root": rr["acquisition_root"], "topology": rr["topology"], "layers": rr["layers"],
                "phase": pv["phase"], "measured_ms": pv["measured_ms"], "predicted_ms": pv["predicted_ms"],
                "ape_pct": pv["ape_pct"], "ape": pv["ape"], "capture_uri": rr["capture_uri"],
                "status": pv["status"], "pointwise_pass": pv["pointwise_pass"],
                "vertical_drilldown": pv["vertical_drilldown"],
            })

    acq_overlap = resolver.overlap_count()
    if acq_overlap:
        raise ModelingError(f"producer_lineage_split_leakage:overlap={acq_overlap}")
    return {
        "schema": "mindie-sim.a5-cann93-multilevel-accuracy/1",
        "campaign": "a5-cann93-regimes-20261120",
        "source_sha256": _tracked_source_hashes(root),
        "disjoint_split_verification": {
            "level1_overlap": len(set(l1["train"]["ids"]) & set(l1["heldout"]["ids"])),
            "level2_overlap": len({x["id"] for x in l2["train"]} & {x["id"] for x in l2["heldout"]}),
            "level3_overlap": len({x["id"] for x in l3["train"]} & {x["id"] for x in l3["heldout"]}),
            "acquisition_root_overlap": acq_overlap,
            "exact_fx_subgraph_matches": fx_matches,
        },
        "level1_operators": {sp: {k: v for k, v in l1[sp].items() if k != "ids"} for sp in SPLITS},
        "level2_subgraphs_and_blocks": {
            sp: {
                "summary": {
                    **_stats(l2[sp], level="L2"),
                    "missing_evidence": sum(1 for x in l2[sp] if x.get("status") == "missing_evidence"),
                },
                "records": [{k: v for k, v in x.items() if k != "ape"} for x in l2[sp]],
            }
            for sp in SPLITS
        },
        "level3_model_steps": {
            sp: {"summary": _stats(l3[sp], level="L3"), "records": [{k: v for k, v in x.items() if k != "ape"} for x in l3[sp]]}
            for sp in SPLITS
        },
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, default=Path(os.environ.get("MINDIE_SIM_CAPTURES_DIR", "/ci/silicon_data")))
    parser.add_argument("--split-manifest", type=Path, default=None, help="Frozen acquisition split manifest JSON")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(args.captures_dir, split_manifest=args.split_manifest, workers=max(1, int(args.workers)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "l1_train": report["level1_operators"]["train"]["summary"], "l1_heldout": report["level1_operators"]["heldout"]["summary"],
        "l2_train": report["level2_subgraphs_and_blocks"]["train"]["summary"], "l2_heldout": report["level2_subgraphs_and_blocks"]["heldout"]["summary"],
        "l3_train": report["level3_model_steps"]["train"]["summary"], "l3_heldout": report["level3_model_steps"]["heldout"]["summary"],
        "fx_matches": report["disjoint_split_verification"]["exact_fx_subgraph_matches"], "wall_seconds": report["wall_seconds"],
    }, indent=2))
    levels = (report["level1_operators"], report["level2_subgraphs_and_blocks"], report["level3_model_steps"])
    l1_heldout_families = report["level1_operators"]["heldout"].get("families") or {}
    l1_families_failed = (
        not l1_heldout_families
        or any(fam_stats.get("status") != "pass" for fam_stats in l1_heldout_families.values())
    )
    return int(
        any(level["heldout"]["summary"]["status"] != "pass" or level["unassigned"]["summary"]["total"]
            for level in levels)
        or l1_families_failed
        or any(row.get("vertical_drilldown", {}).get("passed") is not True
               for row in report["level3_model_steps"]["heldout"]["records"])
    )


if __name__ == "__main__":
    sys.exit(main())
