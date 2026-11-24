"""Strict allowlist-only public view projectors for ``mindie_sim.export``.

[Role] Enforce a single closed allowlist boundary between internal simulator
  records and all human/agent-facing export surfaces (``action_plan``,
  ``agent_brief.json`` bundle & mapping overloads, Markdown digest, and CLI
  receipts):
  - Fixed-order scenario & configuration projection covering ONLY user-facing
    serving, parallelism, quantization, speculative, and runtime fields;
  - Internal track enum mapper (``merged`` / ``unmerged_pr`` /
    ``expected_ideal`` / ``raw_eager`` -> clean public labels);
  - Strict allowlist projectors for ``recommended_checkpoint``,
    ``serving_rules``, ``pd_pool``, ``artifacts``, ``two_step_gain``,
    ``script_audit``, ``action_plan_3_tracks``, ``codesign_opportunities``
    (``MegaFusionBound``, ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``),
    and the full ``agent_brief`` payload.
[Invariants] Zero arbitrary dictionary passthrough, zero ``str(dict_obj)``
  leakage (e.g. ``{'audit': 'canary'}``), zero raw internal track enums, zero
  ``feature_track`` / ``moe_route`` / ``solver_feasible`` keys, zero non-finite
  floats, and zero internal-only DSL backend labels.
[Timeouts & Locks] Pure O(N) in-memory projection; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Read-only over caller inputs; returns freshly constructed
  JSON-safe plain Python structures.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

from .models import (
    AGENT_BRIEF_SCHEMA,
    BACKEND_ASCENDC,
    BACKEND_CANN,
    BACKEND_CATLASS,
    BACKEND_HCCL_MC2,
    BACKEND_TILELANG,
    BACKEND_TORCH_NPU,
    BACKEND_TRITON,
    DEPLOYMENT_MODE_LABELS,
    STRUCTURE_BACKEND_MAP,
    TRACK_CONFIG_ENV,
    TRACK_FRAMEWORK,
    TRACK_KERNEL,
    ExportBundle,
)

DEFAULT_TOP_ISSUES_LIMIT = 3
DEFAULT_ISSUE_TITLE_LIMIT = 40
DEFAULT_MAX_CAPTURE_SIZES = 64
DEFAULT_MAX_CODESIGN_OPPS = 8
DEFAULT_MAX_REFS = 4
DEFAULT_MAX_OPS = 8
FLOAT_DIFF_TOL = 1e-6

#: Canonical public display labels for internal feature/stack track enums.
TRACK_DISPLAY_LABELS_ZH: Mapping[str, str] = {
    "merged": "当前栈基线",
    "shipping": "当前栈基线",
    "unmerged_pr": "提案版本",
    "proposed": "提案版本",
    "expected_ideal": "目标架构预演",
    "target": "目标架构预演",
    "raw_eager": "Eager 基线",
    "eager": "Eager 基线",
}
TRACK_DISPLAY_LABELS_EN: Mapping[str, str] = {
    "merged": "Shipping",
    "shipping": "Shipping",
    "unmerged_pr": "Proposed changes",
    "proposed": "Proposed changes",
    "expected_ideal": "Target projection",
    "target": "Target projection",
    "raw_eager": "Eager baseline",
    "eager": "Eager baseline",
}

#: Canonical 3-track action plan labels shared across Sheet 1, Sheet 4, and agent_brief.
THREE_TRACK_LABELS: Mapping[str, str] = {
    TRACK_CONFIG_ENV: "[1] 配置与并行",
    TRACK_FRAMEWORK: "[2] 框架图与多流改造",
    TRACK_KERNEL: "[3] 新算子开发与 Tiling 调优",
}

#: Strict allowlists for public configuration, parallel axes, and artifact keys.
PUBLIC_CONFIG_FIELD_ORDER: tuple[str, ...] = (
    "hardware_sku",
    "deployment_mode",
    "pd_transfer_mode",
    "phase",
    "prompt_tokens",
    "output_tokens",
    "concurrency_per_dp",
    "cluster_concurrency",
    "dp",
    "tp",
    "ep",
    "pp",
    "pcp",
    "dcp",
    "local_tp",
    "shared_tp",
    "sp",
    "graph_mode",
    "weight_quant",
    "kv_cache_dtype",
    "indexer_kv_dtype",
    "speculative_method",
    "speculative_proposals",
    "speculative_accepted",
    "engram_offload",
    "enable_fused_mc2",
    "multistream_overlap_shared_expert",
)
PUBLIC_EFFECTIVE_CONFIG_ALIASES: Mapping[str, str] = {
    "hardware_sku": "hardware_sku",
    "deployment_mode": "deployment_mode",
    "pd_transfer_mode": "pd_transfer_mode",
    "phase": "phase",
    "prompt_tokens": "prompt_tokens",
    "output_tokens": "output_tokens",
    "concurrency_per_dp": "concurrency_per_dp",
    "cluster_concurrency": "cluster_concurrency",
    "dp": "dp",
    "tp": "tp",
    "ep": "ep",
    "pp": "pp",
    "pcp": "pcp",
    "dcp": "dcp",
    "local_tp": "local_tp",
    "shared_tp": "shared_tp",
    "sp": "sp",
    "sequence_parallel": "sp",
    "graph_mode": "graph_mode",
    "graph_capture_mode": "graph_mode",
    "weight_quant": "weight_quant",
    "kv_cache_dtype": "kv_cache_dtype",
    "cache_dtype": "kv_cache_dtype",
    "indexer_kv_dtype": "indexer_kv_dtype",
    "speculative_method": "speculative_method",
    "spec_method": "speculative_method",
    "speculative_proposals": "speculative_proposals",
    "spec_proposals": "speculative_proposals",
    "speculative_accepted": "speculative_accepted",
    "avg_accepted_tokens": "speculative_accepted",
    "engram_offload": "engram_offload",
    "engram_host_offload": "engram_offload",
    "enable_fused_mc2": "enable_fused_mc2",
    "multistream_overlap_shared_expert": "multistream_overlap_shared_expert",
}
_CONFIG_INT_KEYS = frozenset({
    "prompt_tokens", "output_tokens", "concurrency_per_dp", "cluster_concurrency",
    "dp", "tp", "ep", "pp", "pcp", "dcp", "local_tp", "shared_tp",
    "speculative_proposals", "enable_fused_mc2",
})
_CONFIG_BOOL_KEYS = frozenset({
    "sp", "engram_offload", "multistream_overlap_shared_expert",
})
_CONFIG_FLOAT_KEYS = frozenset({"speculative_accepted"})
_CONFIG_STR_KEYS = frozenset({
    "hardware_sku", "deployment_mode", "pd_transfer_mode", "phase",
    "graph_mode", "weight_quant", "kv_cache_dtype", "indexer_kv_dtype",
    "speculative_method",
})

PUBLIC_PARALLEL_AXES: tuple[str, ...] = (
    "dp", "tp", "ep", "pp", "dcp", "pcp", "cp", "local_tp", "shared_tp",
)
PUBLIC_ARTIFACT_KEYS: tuple[str, ...] = (
    "comparison_xlsx", "trace_view_json", "agent_brief_json", "optimized_serve_sh",
)
PUBLIC_CHECKPOINT_KEYS: tuple[str, ...] = (
    "repo_id", "quant_preset", "kv_cache_dtype", "uses_quarot", "modelers_url",
)
PUBLIC_SERVING_RULE_KEYS: tuple[str, ...] = (
    "effective_kv_cache_dtype", "speculative_enabled", "speculative_method",
    "draft_model", "draft_tp", "capture_sizes",
)
PUBLIC_PD_POOL_KEYS: tuple[str, ...] = (
    "pd_ratio", "pd_transfer_mode", "prefill_instances", "decode_instances",
    "prefill_used_dies", "decode_used_dies", "used_dies",
    "prefill_parallel", "decode_parallel",
    "handoff_tpot_ms", "prefill_kv_rps", "e2e_total_tps",
)
PUBLIC_STEP1_GAIN_KEYS: tuple[str, ...] = (
    "label", "tuned_evaluated", "original_fits_hbm", "before_step_ms",
    "before_effective_tpot_ms", "tuned_step_ms", "saved_ms",
    "tuned_effective_tpot_ms", "tuned_single_die_tps", "issues_fixed_count",
)
PUBLIC_STEP2_GAIN_KEYS: tuple[str, ...] = (
    "label", "original_fits_hbm", "before_step_ms", "after_step_ms",
    "saved_ms", "saved_from_tuned_ms", "after_effective_tpot_ms",
    "after_single_die_tps", "speedup_vs_original",
)
PUBLIC_CODESIGN_KINDS = frozenset({
    "mega_fusion", "space_time_tradeoff", "cross_hw_offload_multistream", "tiler_tune",
})
PUBLIC_BACKENDS = frozenset({
    BACKEND_CANN, BACKEND_ASCENDC, BACKEND_TILELANG, BACKEND_CATLASS,
    BACKEND_TRITON, BACKEND_TORCH_NPU, BACKEND_HCCL_MC2,
})
_KERNEL_BINDING_BACKEND_MAP: Mapping[str, str] = {
    "cann": BACKEND_CANN,
    "cann aclnn/ascendc": BACKEND_CANN,
    "ascendc": BACKEND_ASCENDC,
    "tilelang": BACKEND_TILELANG,
    "catlass": BACKEND_CATLASS,
    "triton": BACKEND_TRITON,
    "triton-ascend": BACKEND_TRITON,
    "triton_ascend": BACKEND_TRITON,
    "hccl/mc2": BACKEND_HCCL_MC2,
    "hccl / mc2": BACKEND_HCCL_MC2,
    "hccl": BACKEND_HCCL_MC2,
    "mc2": BACKEND_HCCL_MC2,
    "pytorch-npu": BACKEND_TORCH_NPU,
    "torch_npu": BACKEND_TORCH_NPU,
    "torch_npu custom op": BACKEND_TORCH_NPU,
    "host": BACKEND_TORCH_NPU,
    "logical": BACKEND_CANN,
}
PUBLIC_PIPE_NAMES = frozenset({
    "cube", "vector", "aic", "aiv", "hbm", "hbm_read", "hbm_write",
    "mte1", "mte2", "mte3", "fixpipe", "link", "hccl", "hccs", "comm",
    "host", "launch", "memory", "compute", "sram", "ub", "l1", "l0",
    "vec_alu", "vec_sfu", "vec_reduce",
})

_INTERNAL_TRACK_WORD_RE = re.compile(r"\b(expected_ideal|unmerged_pr|raw_eager|merged)\b")
_INTERNAL_KEY_WORD_RE = re.compile(r"\b(feature_track|moe_route|solver_feasible)\b")
_LEGACY_DSL_REPO_RE = re.compile(r"(?i)\bcann/[a-z0-9_]+-dsl\b")
_LEGACY_DSL_TOKEN_RE = re.compile(r"(?i)\bcann[_\-]?bot(?:[_\-]?dsl)?\b")
_PY_DICT_LITERAL_RE = re.compile(r"\{[^{}]*['\"][^'\"]+['\"]\s*:[^{}]*\}")


def _is_strict_int(val: Any) -> bool:
    return type(val) is int


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _coerce_strict_int(val: Any) -> int | None:
    return int(val) if _is_strict_int(val) else None


def _coerce_strict_bool(val: Any) -> bool | None:
    return val if isinstance(val, bool) else None


def _coerce_finite_float(val: Any) -> float | None:
    return float(val) if _is_finite_number(val) else None


def _is_container(val: Any) -> bool:
    return isinstance(val, (Mapping, list, tuple, set, frozenset, bytes, bytearray))


def format_track_label(raw_track: Any, *, lang: str = "zh") -> str:
    """Map internal track enum to a clean public label (never emitting raw enum tokens)."""
    table = TRACK_DISPLAY_LABELS_EN if str(lang).strip().lower() == "en" else TRACK_DISPLAY_LABELS_ZH
    default_lbl = table["merged"]
    if not isinstance(raw_track, str):
        return default_lbl
    key = raw_track.strip().lower()
    if key in table:
        return table[key]
    if raw_track.strip() in table.values():
        return raw_track.strip()
    return default_lbl


def normalize_public_backend(raw_backend: Any, *, is_gemm: bool = False) -> str:
    """Map any backend token onto the public open-source Ascend backend taxonomy."""
    if not isinstance(raw_backend, str) or not raw_backend.strip():
        return BACKEND_CATLASS if is_gemm else BACKEND_TILELANG
    raw = raw_backend.strip()
    if raw in PUBLIC_BACKENDS:
        return raw
    low = raw.lower()
    if low in _KERNEL_BINDING_BACKEND_MAP:
        return _KERNEL_BINDING_BACKEND_MAP[low]
    if low in STRUCTURE_BACKEND_MAP:
        return STRUCTURE_BACKEND_MAP[low]
    if low.endswith(("_dsl", "-dsl")) or low == "dsl" or _LEGACY_DSL_TOKEN_RE.search(low):
        return BACKEND_CATLASS if is_gemm else BACKEND_TILELANG
    return BACKEND_CATLASS if is_gemm else BACKEND_TILELANG


def sanitize_public_text(val: Any) -> str:
    """Project a scalar into clean public text, rejecting dicts/containers and internal tokens."""
    if val is None or _is_container(val):
        return ""
    if isinstance(val, float) and not math.isfinite(val):
        return ""
    text = str(val)
    text = _PY_DICT_LITERAL_RE.sub("", text)
    text = _LEGACY_DSL_REPO_RE.sub("Ascend/tilelang-ascend", text)
    text = _LEGACY_DSL_TOKEN_RE.sub(BACKEND_TILELANG, text)
    text = _INTERNAL_TRACK_WORD_RE.sub(lambda m: TRACK_DISPLAY_LABELS_ZH[m.group(1)], text)
    text = _INTERNAL_KEY_WORD_RE.sub("", text)
    return " ".join(text.split())


def sanitize_source_ref(raw_ref: Any) -> str:
    """Sanitize repository/path source references into clean open-source paths."""
    if not isinstance(raw_ref, str) or not raw_ref.strip():
        return ""
    parts: list[str] = []
    for chunk in raw_ref.split(";"):
        cleaned = _LEGACY_DSL_REPO_RE.sub("Ascend/tilelang-ascend", chunk.strip())
        cleaned = _LEGACY_DSL_TOKEN_RE.sub("tilelang-ascend", cleaned)
        cleaned = _PY_DICT_LITERAL_RE.sub("", cleaned).strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return "; ".join(parts)


def _trim(text: str, limit: int) -> str:
    clean = sanitize_public_text(text)
    if limit <= 0 or len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def _coerce_config_value(canon_key: str, raw_val: Any) -> Any:
    if raw_val is None or _is_container(raw_val):
        return None
    if canon_key in _CONFIG_INT_KEYS:
        return _coerce_strict_int(raw_val)
    if canon_key in _CONFIG_BOOL_KEYS:
        return _coerce_strict_bool(raw_val)
    if canon_key in _CONFIG_FLOAT_KEYS:
        return _coerce_finite_float(raw_val)
    if canon_key in _CONFIG_STR_KEYS:
        if not isinstance(raw_val, str):
            return None
        cleaned = sanitize_public_text(raw_val)
        return cleaned if cleaned else None
    return None


def project_public_effective_config(bundle_or_mapping: Any) -> dict[str, Any]:
    """Return ONLY allowlisted, type-checked user-facing fields from ``effective_config``."""
    raw = (
        bundle_or_mapping
        if isinstance(bundle_or_mapping, Mapping)
        else getattr(bundle_or_mapping, "effective_config", None)
    )
    if not isinstance(raw, Mapping):
        return {}
    staged: dict[str, Any] = {}
    moe_route = raw.get("moe_route")
    if isinstance(moe_route, Mapping):
        mc2_val = _coerce_strict_int(moe_route.get("enable_fused_mc2"))
        if mc2_val is not None:
            staged["enable_fused_mc2"] = mc2_val
        mso_val = _coerce_strict_bool(moe_route.get("multistream_overlap_shared_expert"))
        if mso_val is not None:
            staged["multistream_overlap_shared_expert"] = mso_val
    for raw_key, raw_val in raw.items():
        if not isinstance(raw_key, str):
            continue
        canon_key = PUBLIC_EFFECTIVE_CONFIG_ALIASES.get(raw_key)
        if canon_key is None:
            continue
        coerced = _coerce_config_value(canon_key, raw_val)
        if coerced is not None:
            staged[canon_key] = coerced
    return {k: staged[k] for k in PUBLIC_CONFIG_FIELD_ORDER if k in staged}


def project_public_scenario_config(bundle: ExportBundle) -> dict[str, Any]:
    """Project canonical typed scenario & runtime config in fixed field order."""
    mem = bundle.memory_ledger
    eff = project_public_effective_config(bundle)
    pd_pool = project_pd_pool(bundle.pd_pool_summary)
    dep_raw = sanitize_public_text(bundle.deployment_mode)
    dep_label = DEPLOYMENT_MODE_LABELS.get(dep_raw, dep_raw)
    pd_xfer = str((pd_pool or {}).get("pd_transfer_mode") or eff.get("pd_transfer_mode") or "")
    cluster_c = int(bundle.cluster_concurrency or (mem.concurrency * max(1, mem.dp)))
    return {
        "hardware_sku": sanitize_public_text(bundle.hardware_sku),
        "deployment_mode": dep_label,
        "pd_transfer_mode": sanitize_public_text(pd_xfer),
        "phase": sanitize_public_text(bundle.phase),
        "prompt_tokens": int(bundle.prompt_tokens),
        "output_tokens": int(bundle.output_tokens),
        "concurrency_per_dp": int(mem.concurrency),
        "cluster_concurrency": cluster_c,
        "dp": int(mem.dp),
        "tp": int(mem.tp),
        "ep": int(mem.ep),
        "pp": int(mem.pp),
        "pcp": int(mem.pcp),
        "dcp": int(mem.dcp),
        "local_tp": int(mem.local_tp),
        "shared_tp": int(mem.shared_tp),
        "sp": bool(bundle.sequence_parallel),
        "graph_mode": sanitize_public_text(bundle.graph_capture_mode or eff.get("graph_mode") or ""),
        "weight_quant": sanitize_public_text(mem.weight_quant_display or mem.weight_quant),
        "kv_cache_dtype": sanitize_public_text(mem.cache_dtype),
        "indexer_kv_dtype": sanitize_public_text(eff.get("indexer_kv_dtype") or ""),
        "speculative_method": sanitize_public_text(bundle.spec_method or "None"),
        "speculative_proposals": int(bundle.spec_proposals),
        "speculative_accepted": float(bundle.avg_accepted_tokens),
        "engram_offload": bool(mem.engram_host_offload),
        "enable_fused_mc2": int(eff.get("enable_fused_mc2", 0)),
        "multistream_overlap_shared_expert": bool(eff.get("multistream_overlap_shared_expert", False)),
    }


def _same_public_config_values(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in PUBLIC_CONFIG_FIELD_ORDER:
        lv, rv = left.get(key), right.get(key)
        if key in _CONFIG_FLOAT_KEYS:
            if abs(float(lv or 0.0) - float(rv or 0.0)) > FLOAT_DIFF_TOL:
                return False
        elif lv != rv:
            return False
    return True


def is_same_public_config(before: ExportBundle, after: ExportBundle) -> bool:
    """Compare two bundles strictly on their allowlisted typed public configs."""
    return _same_public_config_values(
        project_public_scenario_config(before),
        project_public_scenario_config(after),
    )


def diff_public_scenario_configs(before: ExportBundle, after: ExportBundle) -> list[tuple[str, str, str]]:
    """Compare typed allowlisted configs and return fixed-order factual public diffs."""
    b_raw_eff = before.effective_config if isinstance(before.effective_config, Mapping) else {}
    a_raw_eff = after.effective_config if isinstance(after.effective_config, Mapping) else {}
    if b_raw_eff.get("solver_feasible") is False or a_raw_eff.get("solver_feasible") is False:
        return []
    bc = project_public_scenario_config(before)
    ac = project_public_scenario_config(after)
    diffs: list[tuple[str, str, str]] = []
    if bc["hardware_sku"] != ac["hardware_sku"]:
        diffs.append(("硬件型号 (Hardware SKU)", bc["hardware_sku"] or "—", ac["hardware_sku"] or "—"))
    if bc["deployment_mode"] != ac["deployment_mode"] and (bc["deployment_mode"] or ac["deployment_mode"]):
        diffs.append(("部署形态 (Deployment Mode)", bc["deployment_mode"] or "—", ac["deployment_mode"] or "—"))
    bp = project_pd_pool(before.pd_pool_summary) or {}
    ap = project_pd_pool(after.pd_pool_summary) or {}
    if (bp.get("pd_ratio"), bp.get("pd_transfer_mode"), bp.get("used_dies")) != (
        ap.get("pd_ratio"), ap.get("pd_transfer_mode"), ap.get("used_dies")
    ) and (bp or ap):
        b_txt = f"{bp.get('pd_ratio', '—')} ({bp.get('pd_transfer_mode', '—')}, {bp.get('used_dies', before.memory_ledger.total_dies)} dies)" if bp else "Colocated"
        a_txt = f"{ap.get('pd_ratio', '—')} ({ap.get('pd_transfer_mode', '—')}, {ap.get('used_dies', after.memory_ledger.total_dies)} dies)" if ap else "Colocated"
        diffs.append(("PD 池化配比 (PD Pool)", b_txt, a_txt))
    elif bc["pd_transfer_mode"] != ac["pd_transfer_mode"]:
        diffs.append(("PD 传输模式 (PD Transfer)", bc["pd_transfer_mode"] or "—", ac["pd_transfer_mode"] or "—"))
    if bc["phase"] != ac["phase"]:
        diffs.append(("推理阶段 (Phase)", bc["phase"] or "—", ac["phase"] or "—"))
    if (bc["prompt_tokens"], bc["output_tokens"]) != (ac["prompt_tokens"], ac["output_tokens"]):
        diffs.append((
            "序列长度 (Prompt / Output)",
            f"Prompt={bc['prompt_tokens']}, Output={bc['output_tokens']}",
            f"Prompt={ac['prompt_tokens']}, Output={ac['output_tokens']}",
        ))
    par_axes = (
        ("DP", bc["dp"], ac["dp"]), ("TP", bc["tp"], ac["tp"]), ("EP", bc["ep"], ac["ep"]),
        ("PP", bc["pp"], ac["pp"]), ("DCP", bc["dcp"], ac["dcp"]), ("PCP", bc["pcp"], ac["pcp"]),
        ("Local-TP", bc["local_tp"], ac["local_tp"]), ("Shared-TP", bc["shared_tp"], ac["shared_tp"]),
        ("SP", "ON" if bc["sp"] else "OFF", "ON" if ac["sp"] else "OFF"),
    )
    changed_axes = [(n, str(bv), str(av)) for n, bv, av in par_axes if bv != av]
    if changed_axes:
        diffs.append((
            "并行切分策略 (Parallelism)",
            ", ".join(f"{n}={bv}" for n, bv, _ in changed_axes),
            ", ".join(f"{n}={av}" for n, _, av in changed_axes),
        ))
    if (bc["concurrency_per_dp"], bc["cluster_concurrency"]) != (ac["concurrency_per_dp"], ac["cluster_concurrency"]):
        diffs.append((
            "并发批量 (Batch / Concurrency)",
            f"B/DP={bc['concurrency_per_dp']}, Cluster={bc['cluster_concurrency']}",
            f"B/DP={ac['concurrency_per_dp']}, Cluster={ac['cluster_concurrency']}",
        ))
    if (
        bc["speculative_method"] != ac["speculative_method"]
        or bc["speculative_proposals"] != ac["speculative_proposals"]
        or abs(bc["speculative_accepted"] - ac["speculative_accepted"]) > FLOAT_DIFF_TOL
    ):
        diffs.append((
            "投机解码配置 (Speculative)",
            f"{bc['speculative_method']} (K={bc['speculative_proposals']}, acc={bc['speculative_accepted']:.2f})",
            f"{ac['speculative_method']} (K={ac['speculative_proposals']}, acc={ac['speculative_accepted']:.2f})",
        ))
    if (
        bc["weight_quant"] != ac["weight_quant"]
        or bc["kv_cache_dtype"] != ac["kv_cache_dtype"]
        or bc["indexer_kv_dtype"] != ac["indexer_kv_dtype"]
        or bc["engram_offload"] != ac["engram_offload"]
    ):
        b_idx = f", Indexer={bc['indexer_kv_dtype']}" if bc["indexer_kv_dtype"] else ""
        a_idx = f", Indexer={ac['indexer_kv_dtype']}" if ac["indexer_kv_dtype"] else ""
        diffs.append((
            "量化与缓存配置 (Quant / Cache)",
            f"Weight={bc['weight_quant']}, KV={bc['kv_cache_dtype']}{b_idx}, Offload={'ON' if bc['engram_offload'] else 'OFF'}",
            f"Weight={ac['weight_quant']}, KV={ac['kv_cache_dtype']}{a_idx}, Offload={'ON' if ac['engram_offload'] else 'OFF'}",
        ))
    if bc["graph_mode"] != ac["graph_mode"]:
        diffs.append(("图捕获模式 (Graph Capture)", bc["graph_mode"] or "—", ac["graph_mode"] or "—"))
    if bc["enable_fused_mc2"] != ac["enable_fused_mc2"]:
        diffs.append(("MC2 融合等级 (Fused MC2)", str(bc["enable_fused_mc2"]), str(ac["enable_fused_mc2"])))
    if bc["multistream_overlap_shared_expert"] != ac["multistream_overlap_shared_expert"]:
        diffs.append((
            "共享专家多流掩盖 (MSO)",
            "ON" if bc["multistream_overlap_shared_expert"] else "OFF",
            "ON" if ac["multistream_overlap_shared_expert"] else "OFF",
        ))
    return diffs


def project_recommended_checkpoint(raw: Any) -> dict[str, Any] | None:
    """Allowlist-project recommended checkpoint metadata."""
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in PUBLIC_CHECKPOINT_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if key == "uses_quarot":
            bval = _coerce_strict_bool(val)
            if bval is not None:
                out[key] = bval
        elif isinstance(val, str):
            out[key] = sanitize_public_text(val)
    return out or None


def project_serving_rules(raw: Any) -> dict[str, Any] | None:
    """Allowlist-project serving rules (dropping hw_isa_guidance and unknown keys)."""
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in PUBLIC_SERVING_RULE_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if key == "speculative_enabled":
            bval = _coerce_strict_bool(val)
            if bval is not None:
                out[key] = bval
        elif key == "draft_tp":
            if val is None:
                out[key] = None
            else:
                ival = _coerce_strict_int(val)
                if ival is not None:
                    out[key] = ival
        elif key == "capture_sizes":
            if isinstance(val, (list, tuple)):
                out[key] = [int(x) for x in val if _is_strict_int(x) and x > 0][:DEFAULT_MAX_CAPTURE_SIZES]
        elif isinstance(val, str):
            out[key] = sanitize_public_text(val)
    return out or None


def project_parallel_axes(raw: Any) -> dict[str, int]:
    """Allowlist-project typed parallel axes dictionary."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, int] = {}
    for axis in PUBLIC_PARALLEL_AXES:
        ival = _coerce_strict_int(raw.get(axis))
        if ival is not None and ival > 0:
            out[axis] = ival
    return out


def project_pd_pool(raw: Any) -> dict[str, Any] | None:
    """Allowlist-project PD pool summary including typed prefill/decode parallel maps."""
    if not isinstance(raw, Mapping):
        return None
    int_keys = frozenset({
        "prefill_instances", "decode_instances",
        "prefill_used_dies", "decode_used_dies", "used_dies",
    })
    float_keys = frozenset({"handoff_tpot_ms", "prefill_kv_rps", "e2e_total_tps"})
    out: dict[str, Any] = {}
    for key in PUBLIC_PD_POOL_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if key in {"prefill_parallel", "decode_parallel"}:
            axes = project_parallel_axes(val)
            if axes:
                out[key] = axes
        elif key in int_keys:
            ival = _coerce_strict_int(val)
            if ival is not None:
                out[key] = ival
        elif key in float_keys:
            fval = _coerce_finite_float(val)
            if fval is not None:
                out[key] = fval
        elif isinstance(val, str):
            cleaned = sanitize_public_text(val)
            if cleaned:
                out[key] = cleaned
    return out or None


def project_artifacts(raw: Any, *, basenames_only: bool = False) -> dict[str, str]:
    """Allowlist-project generated artifact paths (only the 4 public artifact keys)."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key in PUBLIC_ARTIFACT_KEYS:
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        cleaned = sanitize_public_text(val)
        if not cleaned:
            continue
        out[key] = Path(cleaned).name if basenames_only else cleaned
    return out


def _project_gain_stage(raw: Any, allowed_keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    bool_keys = frozenset({"tuned_evaluated", "original_fits_hbm"})
    int_keys = frozenset({"issues_fixed_count"})
    out: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in raw:
            continue
        val = raw[key]
        if key == "label":
            if isinstance(val, str):
                out[key] = sanitize_public_text(val)
        elif key in bool_keys:
            bval = _coerce_strict_bool(val)
            if bval is not None:
                out[key] = bval
        elif key in int_keys:
            ival = _coerce_strict_int(val)
            if ival is not None:
                out[key] = ival
        elif val is None:
            out[key] = None
        else:
            fval = _coerce_finite_float(val)
            if fval is not None:
                out[key] = fval
    return out


def project_two_step_gain(raw: Any) -> dict[str, Any]:
    """Allowlist-project two-step script audit gain KPIs."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    if "step1_immediate_zero_code" in raw:
        s1 = _project_gain_stage(raw.get("step1_immediate_zero_code"), PUBLIC_STEP1_GAIN_KEYS)
        if s1:
            out["step1_immediate_zero_code"] = s1
    if "step2_deep_kernel_and_framework" in raw:
        s2 = _project_gain_stage(raw.get("step2_deep_kernel_and_framework"), PUBLIC_STEP2_GAIN_KEYS)
        if s2:
            out["step2_deep_kernel_and_framework"] = s2
    return out


def project_script_audit(
    raw: Any,
    *,
    max_top_issues: int = DEFAULT_TOP_ISSUES_LIMIT,
    title_limit: int = DEFAULT_ISSUE_TITLE_LIMIT,
) -> dict[str, Any] | None:
    """Allowlist-project script audit report for ``agent_brief.json``."""
    if not isinstance(raw, Mapping):
        return None
    raw_issues = raw.get("issues_found")
    raw_top = raw.get("top_issues")
    top_issues: list[dict[str, str]] = []
    if isinstance(raw_issues, Sequence) and not isinstance(raw_issues, (str, bytes)):
        issues_count = sum(1 for item in raw_issues if isinstance(item, Mapping))
        for item in raw_issues:
            if not isinstance(item, Mapping):
                continue
            top_issues.append({
                "id": sanitize_public_text(item.get("issue_id") or item.get("id") or ""),
                "title": _trim(str(item.get("title") or ""), title_limit),
            })
            if len(top_issues) >= max_top_issues:
                break
    else:
        issues_count = _coerce_strict_int(raw.get("issues_count")) or 0
        if isinstance(raw_top, Sequence) and not isinstance(raw_top, (str, bytes)):
            for item in raw_top:
                if not isinstance(item, Mapping):
                    continue
                top_issues.append({
                    "id": sanitize_public_text(item.get("id") or item.get("issue_id") or ""),
                    "title": _trim(str(item.get("title") or ""), title_limit),
                })
                if len(top_issues) >= max_top_issues:
                    break
    out: dict[str, Any] = {
        "source_script": sanitize_public_text(raw.get("source_script") or ""),
        "issues_count": issues_count,
        "top_issues": top_issues,
        "two_step_gain": project_two_step_gain(raw.get("two_step_gain")),
        "optimized_script_path": sanitize_public_text(raw.get("optimized_script_path") or ""),
    }
    if isinstance(raw.get("deployment_mode_hint"), str) and raw.get("deployment_mode_hint"):
        out["deployment_mode_hint"] = sanitize_public_text(raw["deployment_mode_hint"])
    if isinstance(raw.get("stack_id"), str) and raw.get("stack_id"):
        out["stack_id"] = sanitize_public_text(raw["stack_id"])
    return out


def _get_field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _project_mega_fusion(raw: Any) -> dict[str, Any] | None:
    if raw is None or not (isinstance(raw, Mapping) or hasattr(raw, "region_op_ids")):
        return None
    ops_raw = _get_field(raw, "region_op_ids")
    ops = [
        sanitize_public_text(x)
        for x in (ops_raw if isinstance(ops_raw, Sequence) and not isinstance(ops_raw, (str, bytes)) else ())
        if isinstance(x, str) and sanitize_public_text(x)
    ][:DEFAULT_MAX_OPS]
    out: dict[str, Any] = {"region_op_ids": ops}
    for key in (
        "exposed_before_ms", "boundary_lower_bound_ms", "tiled_engine_ms",
        "gain_ceiling_ms", "gain_expected_ms",
        "tilelang_simt_gain_ms", "tilelang_simd_or_ascendc_gain_ms",
    ):
        fval = _coerce_finite_float(_get_field(raw, key))
        if fval is not None:
            out[key] = fval
    for key in ("eliminated_gm_bytes", "ub_live_bytes", "ub_capacity_bytes"):
        ival = _coerce_strict_int(_get_field(raw, key))
        if ival is not None:
            out[key] = ival
    ub_warn = _get_field(raw, "ub_wall_warning")
    if isinstance(ub_warn, str) and ub_warn.strip():
        out["ub_wall_warning"] = sanitize_public_text(ub_warn)
    return out


def _project_space_time(raw: Any) -> dict[str, Any] | None:
    if raw is None or not (isinstance(raw, Mapping) or hasattr(raw, "mechanism")):
        return None
    out: dict[str, Any] = {}
    for key in ("direction", "mechanism"):
        val = _get_field(raw, key)
        if isinstance(val, str) and val.strip():
            out[key] = sanitize_public_text(val)
    for key in ("delta_hbm_bytes", "free_hbm_headroom_bytes", "delta_concurrency_per_dp"):
        ival = _coerce_strict_int(_get_field(raw, key))
        if ival is not None:
            out[key] = ival
    for key in ("delta_step_ms", "delta_tps_per_die"):
        fval = _coerce_finite_float(_get_field(raw, key))
        if fval is not None:
            out[key] = fval
    feas = _coerce_strict_bool(_get_field(raw, "feasible_under_hbm_and_sla"))
    if feas is not None:
        out["feasible_under_hbm_and_sla"] = feas
    return out or None


def _project_cross_hw(raw: Any) -> dict[str, Any] | None:
    if raw is None or not (isinstance(raw, Mapping) or hasattr(raw, "target_component")):
        return None
    out: dict[str, Any] = {}
    for key in ("target_component", "source_tier", "transfer_engine"):
        val = _get_field(raw, key)
        if isinstance(val, str) and val.strip():
            out[key] = sanitize_public_text(val)
    for key in ("compute_hide_window_ms", "transfer_ms", "net_exposed_ms", "delta_tps_per_die"):
        fval = _coerce_finite_float(_get_field(raw, key))
        if fval is not None:
            out[key] = fval
    for key in ("freed_hbm_bytes", "delta_concurrency_per_dp"):
        ival = _coerce_strict_int(_get_field(raw, key))
        if ival is not None:
            out[key] = ival
    return out or None


def project_codesign_opportunities(
    raw_items: Any, *, max_items: int = DEFAULT_MAX_CODESIGN_OPPS,
) -> list[dict[str, Any]]:
    """Allowlist-project ``CoDesignOpportunity`` records (MegaFusion, SpaceTime, CrossHw)."""
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    projected: list[dict[str, Any]] = []
    for item in raw_items:
        if len(projected) >= max_items:
            break
        kind = sanitize_public_text(_get_field(item, "kind"))
        if kind not in PUBLIC_CODESIGN_KINDS:
            continue
        raw_refs = _get_field(item, "evidence_refs")
        refs = [
            sanitize_source_ref(r)
            for r in (raw_refs if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, (str, bytes)) else ())
            if isinstance(r, str) and sanitize_source_ref(r)
        ][:DEFAULT_MAX_REFS]
        rec: dict[str, Any] = {
            "opportunity_id": sanitize_public_text(_get_field(item, "opportunity_id")),
            "kind": kind,
            "track": sanitize_public_text(_get_field(item, "track")),
            "readiness": sanitize_public_text(_get_field(item, "readiness")),
            "title": sanitize_public_text(_get_field(item, "title")),
            "evidence_refs": refs,
        }
        mf = _project_mega_fusion(_get_field(item, "mega_fusion"))
        st = _project_space_time(_get_field(item, "space_time"))
        xhw = _project_cross_hw(_get_field(item, "cross_hw"))
        if mf is not None:
            rec["mega_fusion"] = mf
        if st is not None:
            rec["space_time"] = st
        if xhw is not None:
            rec["cross_hw"] = xhw
        projected.append(rec)
    return projected


def _project_kpi_column(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    str_keys = ("analysis_mode", "binding_state", "sla_verdict", "sla_regime")
    bool_keys = ("has_unbound_formula_ops",)
    float_keys = (
        "step_ms", "handoff_tpot_ms", "effective_tpot_ms", "n_tok_per_die",
        "us_per_tok_die", "single_die_tps", "cluster_tps", "l_star_ms",
        "l_star_attainment_pct", "cube_util_pct", "vector_util_pct",
        "hbm_mbu_pct", "hccl_link_util_pct",
    )
    nullable_float_keys = ("sla_target_ms", "sla_slack_ms")
    out: dict[str, Any] = {}
    for key in str_keys:
        if isinstance(raw.get(key), str):
            out[key] = sanitize_public_text(raw[key])
    if "track_label" in raw and isinstance(raw["track_label"], str):
        out["track_label"] = format_track_label(raw["track_label"])
    elif "feature_track" in raw and isinstance(raw["feature_track"], str):
        out["track_label"] = format_track_label(raw["feature_track"])
    for key in bool_keys:
        bval = _coerce_strict_bool(raw.get(key))
        if bval is not None:
            out[key] = bval
    for key in float_keys:
        fval = _coerce_finite_float(raw.get(key))
        if fval is not None:
            out[key] = fval
    for key in nullable_float_keys:
        if key not in raw:
            continue
        if raw[key] is None:
            out[key] = None
        else:
            fval = _coerce_finite_float(raw[key])
            if fval is not None:
                out[key] = fval
    return out


def _project_scenario_block(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    str_keys = (
        "model", "hardware_sku", "topology_preset", "phase", "deployment_mode",
        "parallelism_co_domain", "module_parallel_domains", "spec_method",
        "weight_quant", "cache_dtype", "indexer_kv_dtype",
        "limiting_constraint", "outcome_status",
    )
    int_keys = (
        "total_dies", "dp", "tp", "ep", "pp", "dcp", "pcp", "local_tp", "shared_tp",
        "prompt_tokens", "output_tokens", "concurrency_per_dp", "cluster_concurrency",
        "max_hbm_concurrency_per_dp", "spec_proposals", "physical_b_vram_max",
        "configured_max_num_seqs",
    )
    nullable_int_keys = ("requested_concurrency",)
    float_keys = (
        "avg_accepted_tokens", "native_bf16_weight_gib_per_die",
        "quantized_weight_gib_per_die", "weight_compression_ratio",
    )
    nullable_float_keys = ("tpot_slo_ms", "ttft_slo_ms")
    bool_keys = ("sequence_parallel", "fits_hbm", "engram_offload")
    out: dict[str, Any] = {}
    for key in str_keys:
        if key in raw and isinstance(raw[key], str):
            out[key] = sanitize_public_text(raw[key])
    if isinstance(raw.get("track_label"), str):
        out["track_label"] = format_track_label(raw["track_label"])
    for key in int_keys:
        ival = _coerce_strict_int(raw.get(key))
        if ival is not None:
            out[key] = ival
    for key in nullable_int_keys:
        if key in raw:
            out[key] = None if raw[key] is None else _coerce_strict_int(raw[key])
    for key in float_keys:
        fval = _coerce_finite_float(raw.get(key))
        if fval is not None:
            out[key] = fval
    for key in nullable_float_keys:
        if key in raw:
            out[key] = None if raw[key] is None else _coerce_finite_float(raw[key])
    for key in bool_keys:
        bval = _coerce_strict_bool(raw.get(key))
        if bval is not None:
            out[key] = bval
    if "recommended_checkpoint" in raw:
        out["recommended_checkpoint"] = project_recommended_checkpoint(raw.get("recommended_checkpoint"))
    if "pd_pool" in raw:
        pd_val = project_pd_pool(raw.get("pd_pool"))
        if pd_val is not None:
            out["pd_pool"] = pd_val
    if isinstance(raw.get("calibration_basis"), Mapping):
        cb_raw = raw["calibration_basis"]
        cb_out: dict[str, Any] = {}
        for ck in ("mode", "pt_sku", "cann"):
            if isinstance(cb_raw.get(ck), str):
                cb_out[ck] = sanitize_public_text(cb_raw[ck])
        cal_b = _coerce_strict_bool(cb_raw.get("calibrated"))
        if cal_b is not None:
            cb_out["calibrated"] = cal_b
        if cb_out:
            out["calibration_basis"] = cb_out
    return out


def _project_kpis_block(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    if isinstance(raw.get("before"), Mapping):
        out["before"] = _project_kpi_column(raw["before"])
    if isinstance(raw.get("after"), Mapping):
        out["after"] = _project_kpi_column(raw["after"])
    for key in (
        "saved_step_ms", "saved_us_per_tok_die", "speedup",
        "sheet4_cards_total_gain_ms", "sheet4_cards_total_gain_us_per_tok_die",
    ):
        fval = _coerce_finite_float(raw.get(key))
        if fval is not None:
            out[key] = fval
    if isinstance(raw.get("waterfall"), Mapping):
        wf_keys = (
            "config_saved_ms", "new_kernel_saved_ms", "tune_kernel_saved_ms",
            "residual_saved_ms", "reconciliation_error_ms",
            "config_saved_us_per_tok_die", "new_kernel_saved_us_per_tok_die",
            "tune_kernel_saved_us_per_tok_die", "residual_saved_us_per_tok_die",
            "reconciliation_error_us_per_tok_die",
        )
        out["waterfall"] = {
            k: float(raw["waterfall"][k])
            for k in wf_keys
            if _is_finite_number(raw["waterfall"].get(k))
        }
    return out


def _project_action_plan_block(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    cfg_out: list[dict[str, Any]] = []
    for item in (raw.get("config_and_env") or ()):
        if not isinstance(item, Mapping):
            continue
        rec: dict[str, Any] = {
            "title": sanitize_public_text(item.get("title") or ""),
            "before": sanitize_public_text(item.get("before") or ""),
            "after": sanitize_public_text(item.get("after") or ""),
        }
        if isinstance(item.get("cmd"), str) and item["cmd"].strip():
            rec["cmd"] = sanitize_public_text(item["cmd"])
        cfg_out.append(rec)
    fw_out: list[dict[str, Any]] = []
    for item in (raw.get("framework_tasks") or ()):
        if not isinstance(item, Mapping):
            continue
        fw_out.append({
            "title": sanitize_public_text(item.get("title") or ""),
            "saved_ms": _coerce_finite_float(item.get("saved_ms")) or 0.0,
            "work": sanitize_public_text(item.get("work") or ""),
        })
    kern_out: list[dict[str, Any]] = []
    for item in (raw.get("kernel_tasks") or ()):
        if not isinstance(item, Mapping):
            continue
        krec: dict[str, Any] = {
            "card_id": sanitize_public_text(item.get("card_id") or ""),
            "prio": sanitize_public_text(item.get("prio") or ""),
            "cat": "new" if str(item.get("cat") or "").strip() == "new" else "tune",
            "kernel": sanitize_public_text(item.get("kernel") or ""),
            "submodule": sanitize_public_text(item.get("submodule") or ""),
            "saved_ms": _coerce_finite_float(item.get("saved_ms")) or 0.0,
            "saved_us_per_tok_die": _coerce_finite_float(item.get("saved_us_per_tok_die")) or 0.0,
            "before_ms": _coerce_finite_float(item.get("before_ms")) or 0.0,
            "after_ms": _coerce_finite_float(item.get("after_ms")) or 0.0,
            "binding": sanitize_public_text(item.get("binding") or ""),
            "summary": sanitize_public_text(item.get("summary") or ""),
        }
        if isinstance(item.get("backend"), str) and item["backend"].strip():
            krec["backend"] = normalize_public_backend(item["backend"])
        kern_out.append(krec)
    return {
        "track_labels": dict(THREE_TRACK_LABELS),
        "config_and_env": cfg_out,
        "framework_tasks": fw_out,
        "kernel_tasks": kern_out,
        "parallelism_codesign_note": sanitize_public_text(raw.get("parallelism_codesign_note") or ""),
    }


def _project_module_hierarchy(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    groups: list[dict[str, Any]] = []
    for grp in raw:
        if not isinstance(grp, Mapping):
            continue
        entry: dict[str, Any] = {
            "l1": sanitize_public_text(grp.get("l1") or ""),
            "before_ms": _coerce_finite_float(grp.get("before_ms")) or 0.0,
            "after_ms": _coerce_finite_float(grp.get("after_ms")) or 0.0,
            "share_pct": _coerce_finite_float(grp.get("share_pct")) or 0.0,
        }
        if isinstance(grp.get("children"), Sequence) and not isinstance(grp["children"], (str, bytes)):
            entry["children"] = [
                {
                    "l2": sanitize_public_text(ch.get("l2") or ""),
                    "exposed_ms": _coerce_finite_float(ch.get("exposed_ms")) or 0.0,
                    "before_ms": _coerce_finite_float(ch.get("before_ms")) or 0.0,
                    "saved_us_per_tok_die": _coerce_finite_float(ch.get("saved_us_per_tok_die")) or 0.0,
                }
                for ch in grp["children"]
                if isinstance(ch, Mapping)
            ]
        groups.append(entry)
    return groups


def _project_synthesis_and_triggers(raw: Mapping[str, Any], out: dict[str, Any]) -> None:
    if isinstance(raw.get("kernel_synthesis_specs"), Sequence) and not isinstance(raw["kernel_synthesis_specs"], (str, bytes)):
        specs_out: list[dict[str, Any]] = []
        for sp in raw["kernel_synthesis_specs"]:
            if not isinstance(sp, Mapping):
                continue
            srec: dict[str, Any] = {
                "kernel": sanitize_public_text(sp.get("kernel") or ""),
                "tier": sanitize_public_text(sp.get("tier") or ""),
                "sram_ok": bool(_coerce_strict_bool(sp.get("sram_ok"))),
                "ub_kb": sanitize_public_text(sp.get("ub_kb") or ""),
                "math": sanitize_public_text(sp.get("math") or ""),
            }
            if isinstance(sp.get("backend"), str) and sp["backend"].strip():
                srec["backend"] = normalize_public_backend(sp["backend"])
            specs_out.append(srec)
        out["kernel_synthesis_specs"] = specs_out
    if isinstance(raw.get("semantic_audit_triggers"), Sequence) and not isinstance(raw["semantic_audit_triggers"], (str, bytes)):
        tr_out: list[dict[str, Any]] = []
        for tr in raw["semantic_audit_triggers"]:
            if not isinstance(tr, Mapping):
                continue
            ops_raw = tr.get("ops")
            refs_raw = tr.get("refs")
            tr_out.append({
                "id": sanitize_public_text(tr.get("id") or ""),
                "kind": sanitize_public_text(tr.get("kind") or ""),
                "ops": [
                    sanitize_public_text(o)
                    for o in (ops_raw if isinstance(ops_raw, Sequence) and not isinstance(ops_raw, (str, bytes)) else ())
                    if isinstance(o, str) and sanitize_public_text(o)
                ],
                "gain_us_tok_die": _coerce_finite_float(tr.get("gain_us_tok_die")) or 0.0,
                "reason": sanitize_public_text(tr.get("reason") or ""),
                "refs": [
                    sanitize_source_ref(r)
                    for r in (refs_raw if isinstance(refs_raw, Sequence) and not isinstance(refs_raw, (str, bytes)) else ())
                    if isinstance(r, str) and sanitize_source_ref(r)
                ],
            })
        out["semantic_audit_triggers"] = tr_out


def _project_infeasible_outcome(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in ("status", "deployment_mode", "limiting_constraint"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            out[key] = sanitize_public_text(raw[key])
    for key in ("physical_b_vram_max", "configured_max_num_seqs", "feasible_b_per_dp"):
        ival = _coerce_strict_int(raw.get(key))
        if ival is not None:
            out[key] = ival
    if "requested_concurrency" in raw:
        out["requested_concurrency"] = (
            None if raw["requested_concurrency"] is None else _coerce_strict_int(raw["requested_concurrency"])
        )
    return out or None


def project_agent_brief_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the strict allowlist schema on ANY ``agent_brief`` mapping (bundle, overload, digest)."""
    if not isinstance(raw, Mapping):
        return {"schema": AGENT_BRIEF_SCHEMA}
    out: dict[str, Any] = {
        "schema": sanitize_public_text(raw.get("schema") or AGENT_BRIEF_SCHEMA) or AGENT_BRIEF_SCHEMA,
    }
    if isinstance(raw.get("summary"), str):
        out["summary"] = sanitize_public_text(raw["summary"])
    if isinstance(raw.get("scenario_and_parallelism"), Mapping):
        out["scenario_and_parallelism"] = _project_scenario_block(raw["scenario_and_parallelism"])
    if isinstance(raw.get("kpis_before_vs_after"), Mapping):
        out["kpis_before_vs_after"] = _project_kpis_block(raw["kpis_before_vs_after"])
    if "module_hierarchy" in raw:
        out["module_hierarchy"] = _project_module_hierarchy(raw.get("module_hierarchy"))
    if isinstance(raw.get("action_plan_3_tracks"), Mapping):
        out["action_plan_3_tracks"] = _project_action_plan_block(raw["action_plan_3_tracks"])
    if "codesign_opportunities" in raw:
        out["codesign_opportunities"] = project_codesign_opportunities(raw.get("codesign_opportunities"))
    _project_synthesis_and_triggers(raw, out)
    if "macro_wall_trigger" in raw:
        mw = raw.get("macro_wall_trigger")
        if isinstance(mw, Mapping):
            mw_out: dict[str, Any] = {}
            if isinstance(mw.get("wall_kind"), str):
                mw_out["wall_kind"] = sanitize_public_text(mw["wall_kind"])
            for fk in ("tpot_ms", "tpot_slo_ms", "sla_slack_ms", "hbm_headroom_gib", "hbm_utilization_pct"):
                fval = _coerce_finite_float(mw.get(fk))
                if fval is not None:
                    mw_out[fk] = fval
            out["macro_wall_trigger"] = mw_out or None
        else:
            out["macro_wall_trigger"] = None
    if "va_serving_rules" in raw or "serving_rules" in raw:
        src_rules = raw.get("va_serving_rules") if "va_serving_rules" in raw else raw.get("serving_rules")
        out["va_serving_rules"] = project_serving_rules(src_rules)
    if "generated_files" in raw or "artifacts" in raw:
        src_art = raw.get("generated_files") if "generated_files" in raw else raw.get("artifacts")
        out["generated_files"] = project_artifacts(src_art)
    if raw.get("script_audit") is not None:
        sa = project_script_audit(raw.get("script_audit"))
        if sa is not None:
            out["script_audit"] = sa
    if raw.get("infeasible_outcome") is not None:
        inf = _project_infeasible_outcome(raw.get("infeasible_outcome"))
        if inf is not None:
            out["infeasible_outcome"] = inf
    return out


__all__ = [
    "PUBLIC_ARTIFACT_KEYS",
    "PUBLIC_BACKENDS",
    "PUBLIC_CHECKPOINT_KEYS",
    "PUBLIC_CODESIGN_KINDS",
    "PUBLIC_CONFIG_FIELD_ORDER",
    "PUBLIC_PARALLEL_AXES",
    "PUBLIC_PD_POOL_KEYS",
    "PUBLIC_PIPE_NAMES",
    "PUBLIC_SERVING_RULE_KEYS",
    "THREE_TRACK_LABELS",
    "TRACK_DISPLAY_LABELS_EN",
    "TRACK_DISPLAY_LABELS_ZH",
    "diff_public_scenario_configs",
    "format_track_label",
    "is_same_public_config",
    "normalize_public_backend",
    "project_agent_brief_mapping",
    "project_artifacts",
    "project_codesign_opportunities",
    "project_parallel_axes",
    "project_pd_pool",
    "project_public_effective_config",
    "project_public_scenario_config",
    "project_recommended_checkpoint",
    "project_script_audit",
    "project_serving_rules",
    "project_two_step_gain",
    "sanitize_public_text",
    "sanitize_source_ref",
]
