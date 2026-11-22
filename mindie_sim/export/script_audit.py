"""Safe static launch-script parser, diagnostic auditor, and tuned-script generator.

[Role] Support ``mindie-sim report --from-script <serve.sh>`` workflows without
  ever executing user shell scripts:
  1. ``parse_launch_script(path_or_text)`` extracts environment variables,
     ``vllm serve`` / ``api_server`` flags, JSON config flags, device counts,
     and catalog/local model + quantization hints via stdlib ``shlex``.
  2. ``audit_launch_script(...)`` diagnoses concrete Ascend serving bottlenecks
     (unquantized BF16 weights/KV, ``--enforce-eager``, ``HCCL_OP_EXPANSION_MODE``,
     missing ``MLAPO``/``MC2``/``MSO``/``NZ`` layout, high-TP collectives) and
     quantifies Two-Step Gain (Immediate 0-Code Config Gain vs Deep Target Gain).
  3. ``render_optimized_serve_sh(...)`` emits a ready-to-run, hardware-matched
     ``<prefix>_optimized_serve.sh`` (A5 MXFP8/MXFP4+FP8 KV vs A2/A3 W4A8/W8A8+C8 KV).
[Invariants] Strictly static parsing via ``shlex`` (ZERO ``subprocess``, ``eval``,
  or shell execution). Enforces strict A2/A3 INT8/INT4 vs A5 MXFP hardware-ISA
  separation and atomic same-directory UTF-8 LF file writes.
[Timeouts & Locks] Pure O(script length) in-memory parsing; no locks or network I/O.
[Side-Effect Policy] Read-only except ``render_optimized_serve_sh`` when
  ``output_path`` is explicitly supplied.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any

from mindie_sim.catalog.quant import LONG_CONTEXT_KV_QUANT_THRESHOLD

from .models import GIB_BYTES, US_PER_MS, ExportBundle

#: Schema tag for launch-script audit reports embedded in ``agent_brief.json``.
SCRIPT_AUDIT_SCHEMA = "mindie-sim.script-audit/1"
#: Threshold for flagging high-TP collective overhead when HBM headroom permits Attn-DP.
HIGH_TP_AUDIT_THRESHOLD = 4
#: Default HCCL buffer size (MiB) in generated Ascend launch scripts.
DEFAULT_HCCL_BUFFSIZE_MIB = 512
DEFAULT_HCCL_EP_BUFFSIZE_MIB = 2048
#: Maximum file size (bytes) read from a user script or local config.json.
MAX_STATIC_SCRIPT_BYTES = 262144

_ENV_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_KNOWN_ENV_KEYS = frozenset({
    "HCCL_OP_EXPANSION_MODE", "VLLM_ASCEND_ENABLE_MLAPO", "VLLM_ASCEND_ENABLE_FUSED_MC2",
    "HCCL_BUFFSIZE", "HCCL_BUFFSIZE_EP", "HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE",
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "PYTORCH_NPU_ALLOC_CONF", "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_USE_V1", "TASK_QUEUE_ENABLE",
})
_QUANT_BASENAME_TOKENS: tuple[tuple[str, str], ...] = (
    ("w4a8-mxfp", "mxfp8_mxfp4"),
    ("w4a8_mxfp", "mxfp8_mxfp4"),
    ("mxfp8_mxfp4", "mxfp8_mxfp4"),
    ("mxfp4", "mxfp4"),
    ("mxfp8", "mxfp8"),
    ("w4a8c8", "w4a8"),
    ("w4a8", "w4a8"),
    ("w8a8c8", "w8a8"),
    ("w8a8", "w8a8"),
    ("fp8", "fp8"),
    ("bf16", "bf16"),
)
_KNOWN_CATALOG_HINTS: tuple[tuple[str, str], ...] = (
    ("kimi-k3.1", "kimi-k3.1"),
    ("kimi_k3.1", "kimi-k3.1"),
    ("kimi-k3", "kimi-k3"),
    ("kimi-k2.6", "kimi-k2.6"),
    ("kimi-k2.5", "kimi-k2.5"),
    ("kimi-k2", "kimi-k2"),
    ("qwen3.8-flash", "qwen3.8-flash-next"),
    ("qwen3.8-2.4t", "qwen3.8-2.4t-a95b"),
    ("qwen3.8-27b", "qwen3.8-27b"),
    ("qwen3.6-35b", "qwen3.6-35b-a3b"),
    ("qwen3.5-397b", "qwen3.5-397b-a17b"),
    ("qwen3.5-122b", "qwen3.5-122b-a10b"),
    ("qwen3.5-35b", "qwen3.5-35b-a3b"),
    ("qwen3-next-80b", "qwen3-next-80b-a3b"),
    ("qwen3-235b", "qwen3-235b-a22b"),
    ("qwen3-32b", "qwen3-32b"),
    ("deepseek-v4-pro", "deepseek-v4-pro"),
    ("deepseek-v4.1-flash", "deepseek-v4.1-flash"),
    ("deepseek-v4-flash", "deepseek-v4-flash"),
    ("deepseek-v3.2", "deepseek-v3.2"),
    ("deepseek-v3", "deepseek-v3"),
    ("deepseek-r1", "deepseek-r1"),
    ("glm-5.3-flash", "glm-5.3-flash"),
    ("glm-5.3", "glm-5.3"),
    ("glm-5.2", "glm-5.2"),
    ("glm-5.1", "glm-5.1"),
    ("glm-5", "glm-5"),
)


@dataclass(frozen=True, slots=True)
class ParsedLaunchScript:
    """Structured fields statically extracted from a user shell launch script."""

    source_label: str
    model_arg: str = ""
    inferred_catalog_model: str = ""
    local_config_path: str = ""
    inferred_quant: str = ""
    quantization_flag: str = ""
    kv_cache_dtype: str = ""
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep_enabled: bool = False
    dcp: int = 1
    inferred_num_devices: int = 0
    max_model_len: int = 0
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    block_size: int = 0
    gpu_memory_utilization: float = 0.0
    enforce_eager: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)
    compilation_config: dict[str, Any] = field(default_factory=dict)
    speculative_config: dict[str, Any] = field(default_factory=dict)
    additional_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "model_arg": self.model_arg,
            "inferred_catalog_model": self.inferred_catalog_model,
            "local_config_path": self.local_config_path,
            "inferred_quant": self.inferred_quant,
            "quantization_flag": self.quantization_flag,
            "kv_cache_dtype": self.kv_cache_dtype,
            "tp": self.tp, "dp": self.dp, "pp": self.pp,
            "ep_enabled": self.ep_enabled, "dcp": self.dcp,
            "inferred_num_devices": self.inferred_num_devices,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "env_vars": dict(self.env_vars),
        }


@dataclass(frozen=True, slots=True)
class ScriptAuditIssue:
    """One concrete bottleneck diagnosed in the user's original launch script."""

    issue_id: str
    severity: str
    title: str
    current_value: str
    recommended_value: str
    impact_note: str

    def to_dict(self) -> dict[str, str]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity,
            "title": self.title,
            "current": self.current_value,
            "recommended": self.recommended_value,
            "impact": self.impact_note,
        }


@dataclass(frozen=True, slots=True)
class ScriptAuditReport:
    """Two-step diagnostic audit comparing Original Script -> Tuned Script -> Deep After."""

    schema: str
    parsed: ParsedLaunchScript
    issues: tuple[ScriptAuditIssue, ...]
    immediate_step_summary: dict[str, Any]
    deep_step_summary: dict[str, Any]
    optimized_script_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_script": self.parsed.source_label,
            "issues_found": [i.to_dict() for i in self.issues],
            "two_step_gain": {
                "step1_immediate_zero_code": dict(self.immediate_step_summary),
                "step2_deep_kernel_and_framework": dict(self.deep_step_summary),
            },
            "optimized_script_path": self.optimized_script_path,
        }


def _strip_comments_and_join_continuations(raw_text: str) -> list[str]:
    logical_lines: list[str] = []
    buf = ""
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if buf:
                logical_lines.append(buf)
                buf = ""
            continue
        if stripped.endswith("\\"):
            buf = (buf + " " + stripped[:-1].strip()).strip()
        else:
            buf = (buf + " " + stripped).strip()
            logical_lines.append(buf)
            buf = ""
    if buf:
        logical_lines.append(buf)
    return logical_lines


def _safe_json_dict(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    except Exception:
        return {}


def _safe_int(raw: str, default: int = 0) -> int:
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _safe_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _infer_model_and_quant_from_arg(model_arg: str) -> tuple[str, str, str]:
    """Infer ``(catalog_model, local_config_path, quant_hint)`` without executing code."""
    if not model_arg:
        return "", "", ""
    local_cfg = ""
    quant_hint = ""
    candidate_path = Path(model_arg)
    if candidate_path.exists():
        cfg_file = candidate_path / "config.json" if candidate_path.is_dir() else candidate_path
        if cfg_file.is_file() and cfg_file.name.endswith(".json"):
            local_cfg = str(cfg_file)
            q_desc = cfg_file.parent / "quant_model_description.json"
            if q_desc.is_file():
                try:
                    raw_desc = json.loads(q_desc.read_text(encoding="utf-8")[:MAX_STATIC_SCRIPT_BYTES])
                    vals = " ".join(str(v).upper() for v in raw_desc.values()) if isinstance(raw_desc, Mapping) else ""
                    if "W4A8_MXFP" in vals or "MXFP4" in vals:
                        quant_hint = "mxfp8_mxfp4"
                    elif "W8A8_MXFP8" in vals:
                        quant_hint = "mxfp8"
                    elif "W4A8" in vals:
                        quant_hint = "w4a8"
                    elif "W8A8" in vals:
                        quant_hint = "w8a8"
                except Exception:
                    pass

    base_low = Path(model_arg.rstrip("/")).name.lower()
    if not quant_hint:
        for tok, preset in _QUANT_BASENAME_TOKENS:
            if tok in base_low:
                quant_hint = preset
                break
    catalog_model = ""
    for needle, canonical in _KNOWN_CATALOG_HINTS:
        if needle in base_low:
            catalog_model = canonical
            break
    return catalog_model, local_cfg, quant_hint


def _parse_vllm_tokens(tokens: list[str], state: dict[str, Any]) -> None:
    """Parse ``vllm serve`` / ``api_server`` CLI flags from a tokenized command."""
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "serve" and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            state["model_arg"] = tokens[idx + 1]
            idx += 2
            continue
        if "=" in tok and tok.startswith("--"):
            flag, val = tok.split("=", 1)
            nxt = val
            step = 1
        else:
            flag = tok
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            step = 2

        if flag in {"--model", "-m"}:
            state["model_arg"] = nxt
        elif flag in {"-tp", "--tensor-parallel-size"}:
            state["tp"] = max(1, _safe_int(nxt, 1))
        elif flag in {"-dp", "--data-parallel-size"}:
            state["dp"] = max(1, _safe_int(nxt, 1))
        elif flag in {"-pp", "--pipeline-parallel-size"}:
            state["pp"] = max(1, _safe_int(nxt, 1))
        elif flag in {"--dcp-size", "--cp-size", "--decode-context-parallel-size"}:
            state["dcp"] = max(1, _safe_int(nxt, 1))
        elif flag == "--enable-expert-parallel":
            state["ep_enabled"] = True
            step = 1
        elif flag == "--enforce-eager":
            state["enforce_eager"] = True
            step = 1
        elif flag == "--max-model-len":
            state["max_model_len"] = _safe_int(nxt, 0)
        elif flag == "--max-num-seqs":
            state["max_num_seqs"] = _safe_int(nxt, 0)
        elif flag == "--max-num-batched-tokens":
            state["max_num_batched_tokens"] = _safe_int(nxt, 0)
        elif flag == "--block-size":
            state["block_size"] = _safe_int(nxt, 0)
        elif flag == "--gpu-memory-utilization":
            state["gpu_memory_utilization"] = _safe_float(nxt, 0.0)
        elif flag == "--quantization":
            state["quantization_flag"] = nxt
        elif flag == "--kv-cache-dtype":
            state["kv_cache_dtype"] = nxt
        elif flag == "--compilation-config":
            state["compilation_config"] = _safe_json_dict(nxt)
        elif flag == "--speculative-config":
            state["speculative_config"] = _safe_json_dict(nxt)
        elif flag == "--additional-config":
            state["additional_config"] = _safe_json_dict(nxt)
        else:
            step = 1
        idx += step


def parse_launch_script(path_or_text: str | Path) -> ParsedLaunchScript:
    """Safely parse a shell launch script or inline command without executing it."""
    if isinstance(path_or_text, Path) or (
        isinstance(path_or_text, str) and "\n" not in path_or_text and Path(path_or_text).is_file()
    ):
        src_path = Path(path_or_text)
        raw_text = src_path.read_text(encoding="utf-8")[:MAX_STATIC_SCRIPT_BYTES]
        src_label = src_path.name
    else:
        raw_text = str(path_or_text or "")[:MAX_STATIC_SCRIPT_BYTES]
        src_label = "inline_command"

    env_vars: dict[str, str] = {}
    state: dict[str, Any] = {
        "model_arg": "", "quantization_flag": "", "kv_cache_dtype": "",
        "tp": 1, "dp": 1, "pp": 1, "ep_enabled": False, "dcp": 1,
        "max_model_len": 0, "max_num_seqs": 0, "max_num_batched_tokens": 0,
        "block_size": 0, "gpu_memory_utilization": 0.0, "enforce_eager": False,
        "compilation_config": {}, "speculative_config": {}, "additional_config": {},
    }
    for line in _strip_comments_and_join_continuations(raw_text):
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        if not tokens:
            continue
        pos = 0
        if tokens[0] == "export":
            pos = 1
        while pos < len(tokens):
            match = _ENV_ASSIGN_RE.match(tokens[pos])
            if match is None:
                break
            k, v = match.group(1), match.group(2)
            if k in _KNOWN_ENV_KEYS or k.startswith(("HCCL_", "VLLM_", "ASCEND_", "OMP_", "PYTORCH_NPU")):
                env_vars[k] = v
            pos += 1
        if pos < len(tokens):
            _parse_vllm_tokens(tokens[pos:], state)

    vis = env_vars.get("ASCEND_RT_VISIBLE_DEVICES", "")
    inferred_devs = len([x for x in vis.split(",") if x.strip()]) if vis else 0
    if inferred_devs <= 0:
        inferred_devs = max(1, int(state["dp"]) * int(state["tp"]) * int(state["pp"]))
    cat_model, local_cfg, q_hint = _infer_model_and_quant_from_arg(str(state["model_arg"]))
    if not q_hint and str(state["quantization_flag"]).lower() in {"bf16", "none", ""}:
        q_hint = "bf16"
    return ParsedLaunchScript(
        source_label=src_label,
        model_arg=str(state["model_arg"]),
        inferred_catalog_model=cat_model,
        local_config_path=local_cfg,
        inferred_quant=q_hint,
        quantization_flag=str(state["quantization_flag"]),
        kv_cache_dtype=str(state["kv_cache_dtype"]),
        tp=int(state["tp"]),
        dp=int(state["dp"]),
        pp=int(state["pp"]),
        ep_enabled=bool(state["ep_enabled"]),
        dcp=int(state["dcp"]),
        inferred_num_devices=inferred_devs,
        max_model_len=int(state["max_model_len"]),
        max_num_seqs=int(state["max_num_seqs"]),
        max_num_batched_tokens=int(state["max_num_batched_tokens"]),
        block_size=int(state["block_size"]),
        gpu_memory_utilization=float(state["gpu_memory_utilization"]),
        enforce_eager=bool(state["enforce_eager"]),
        env_vars=env_vars,
        compilation_config=dict(state["compilation_config"]),
        speculative_config=dict(state["speculative_config"]),
        additional_config=dict(state["additional_config"]),
    )


def _is_a5_sku(hardware_sku: str) -> bool:
    low = str(hardware_sku or "").lower()
    return any(tok in low for tok in ("950", "a5", "3510"))


def audit_launch_script(
    parsed: ParsedLaunchScript,
    *,
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
    tuned_bundle: ExportBundle | None = None,
    optimized_script_path: str = "",
) -> ScriptAuditReport:
    """Audit a parsed user launch script and compute Two-Step Gain."""
    is_a5 = _is_a5_sku(after_bundle.hardware_sku)
    mem_b = before_bundle.memory_ledger
    mem_a = after_bundle.memory_ledger
    issues: list[ScriptAuditIssue] = []

    if not parsed.quantization_flag and parsed.inferred_quant in {"", "bf16"}:
        rec_q = (
            "--quantization ascend (msmodelslim QuaRot MXFP8+MXFP4)"
            if is_a5 else "--quantization ascend (msmodelslim W4A8_DYNAMIC / W8A8_DYNAMIC)"
        )
        issues.append(ScriptAuditIssue(
            "missing_quantization_bf16", "HIGH",
            "未启用生产量化 (原生 BF16 显存浪费/带宽受限)",
            "未设置 --quantization (BF16)", rec_q,
            f"原生 BF16 单卡权重={mem_a.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB -> "
            f"量化后={mem_a.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB (压缩 {mem_a.weight_compression_ratio:.2f}x)",
        ))

    ctx = max(parsed.max_model_len, before_bundle.context_tokens)
    if ctx >= LONG_CONTEXT_KV_QUANT_THRESHOLD and parsed.kv_cache_dtype.lower() in {"", "auto", "bf16", "fp16"}:
        rec_kv = "--kv-cache-dtype fp8_e4m3" if is_a5 else "--kv-cache-dtype int8 (C8)"
        issues.append(ScriptAuditIssue(
            "missing_kv_quant", "HIGH",
            "长序列未启用 KV Cache 量化",
            parsed.kv_cache_dtype or "auto (BF16)", rec_kv,
            "长上下文 (>=32K) 启用量化 KV 可降 ~50% KV HBM 占用与注意力访存带宽",
        ))

    if parsed.enforce_eager or str(parsed.compilation_config.get("cudagraph_mode", "")).upper() != "FULL_DECODE_ONLY":
        issues.append(ScriptAuditIssue(
            "missing_aclgraph_full_decode", "HIGH",
            "未启用 ACLGraph FULL_DECODE_ONLY 图捕获",
            "--enforce-eager" if parsed.enforce_eager else str(parsed.compilation_config or "unset"),
            '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'',
            "移除 --enforce-eager 并开启 FULL_DECODE_ONLY 消除 Host 逐算子发射气泡",
        ))

    hccl_mode = parsed.env_vars.get("HCCL_OP_EXPANSION_MODE", "").upper()
    if hccl_mode not in {"AIV", "CCU"}:
        rec_mode = 'HCCL_OP_EXPANSION_MODE="AIV" (+ A5 CCU 硬件调度)' if is_a5 else 'HCCL_OP_EXPANSION_MODE="AIV"'
        issues.append(ScriptAuditIssue(
            "hccl_expansion_mode_suboptimal", "HIGH",
            "通信展开模式未切换至 AIV / CCU",
            hccl_mode or "unset (默认 AI_CPU/HOST)", rec_mode,
            "避免小包通信 ~80-95us AI_CPU/Host 启动惩罚，提升超节点小包吞吐",
        ))

    has_mla = any("mla" in op.submodule.lower() for op in after_bundle.operators)
    if has_mla and parsed.env_vars.get("VLLM_ASCEND_ENABLE_MLAPO", "0") != "1":
        issues.append(ScriptAuditIssue(
            "missing_mlapo", "MEDIUM",
            "MLA 模型未开启 MLAPO 前置融合",
            parsed.env_vars.get("VLLM_ASCEND_ENABLE_MLAPO", "unset"),
            "export VLLM_ASCEND_ENABLE_MLAPO=1",
            "将 Q/KV-A + RMSNorm + RoPE + Absorb + KV Write 融合为单算子 aclnnMlaPrologV3",
        ))

    if not parsed.additional_config.get("multistream_overlap_shared_expert") and any(
        op.stream_index == 1 for op in after_bundle.operators
    ):
        issues.append(ScriptAuditIssue(
            "missing_mso_and_mc2", "MEDIUM",
            "未启用 MSO 共享专家异步掩盖与 MC2 通算融合",
            str(parsed.additional_config or "unset"),
            'export VLLM_ASCEND_ENABLE_FUSED_MC2=1 && --additional-config \'{"enable_fused_mc2": 2, "multistream_overlap_shared_expert": true}\'',
            "用主计算流 MoE Routed GMM 异步 100% 掩盖共享专家耗时",
        ))

    if (not is_a5) and not parsed.additional_config.get("enable_weight_nz_layout"):
        issues.append(ScriptAuditIssue(
            "missing_nz_layout_a23", "MEDIUM",
            "A2/A3 未开启 FRACTAL_NZ 权重排布",
            "enable_weight_nz_layout unset",
            '--additional-config \'{"enable_weight_nz_layout": true}\'',
            "A2/A3 INT8/INT4 GroupedMatmulV4 需 NZ 排布获得最优 Cube 访存效率",
        ))

    if parsed.tp >= HIGH_TP_AUDIT_THRESHOLD and parsed.tp > mem_a.tp:
        issues.append(ScriptAuditIssue(
            "suboptimal_high_tp", "MEDIUM",
            "TP 切分过细导致跨卡小包通信偏重",
            f"DP={parsed.dp}, TP={parsed.tp}",
            f"DP={mem_a.dp}, TP={mem_a.tp}, EP={mem_a.ep}",
            "降低 TP 并提升 DP/EP 可直接减少 Attention/Norm 跨卡集合通信并提高集群吞吐",
        ))

    if not mem_b.fits_hbm:
        issues.append(ScriptAuditIssue(
            "hbm_oom_in_original_script", "CRITICAL",
            "原脚本配置存在 HBM OOM 风险",
            f"Peak HBM={mem_b.max_stage_hbm_bytes / GIB_BYTES:.1f} GiB > Usable={mem_b.usable_hbm_bytes / GIB_BYTES:.1f} GiB",
            f"调优后 Peak HBM={mem_a.max_stage_hbm_bytes / GIB_BYTES:.1f} GiB (Fit={mem_a.fits_hbm})",
            "通过生产量化 + KV 量化 + 合理并行切分消除 OOM 并释放并发空间",
        ))

    b_step = before_bundle.step_span_us / US_PER_MS
    t_ref = tuned_bundle or before_bundle
    t_step = t_ref.step_span_us / US_PER_MS
    a_step = after_bundle.step_span_us / US_PER_MS
    acc = max(1.0, after_bundle.avg_accepted_tokens)
    immediate = {
        "label": "1. 立即可得收益 (0 算子开发 — 仅应用调优后脚本 + va 量化)",
        "before_step_ms": round(b_step, 3),
        "tuned_step_ms": round(t_step, 3),
        "saved_ms": round(max(0.0, b_step - t_step), 3),
        "tuned_effective_tpot_ms": round(t_step / acc, 3),
        "tuned_single_die_tps": round(t_ref.throughput_per_die_tokens_per_s, 2),
        "issues_fixed_count": len(issues),
    }
    deep = {
        "label": "2. 深度攻关终态收益 (After + 算子融合/Roofline 调优/框架改造)",
        "before_step_ms": round(b_step, 3),
        "after_step_ms": round(a_step, 3),
        "saved_ms": round(max(0.0, b_step - a_step), 3),
        "after_effective_tpot_ms": round(a_step / acc, 3),
        "after_single_die_tps": round(after_bundle.throughput_per_die_tokens_per_s, 2),
        "speedup_vs_original": round(b_step / a_step, 2) if a_step > 0.0 else 1.0,
    }
    return ScriptAuditReport(
        schema=SCRIPT_AUDIT_SCHEMA,
        parsed=parsed,
        issues=tuple(issues),
        immediate_step_summary=immediate,
        deep_step_summary=deep,
        optimized_script_path=optimized_script_path,
    )


def render_optimized_serve_sh(
    parsed: ParsedLaunchScript,
    *,
    after_bundle: ExportBundle,
    output_path: str | Path | None = None,
) -> str:
    """Render (and optionally atomically save) a hardware-matched ``optimized_serve.sh``."""
    mem = after_bundle.memory_ledger
    is_a5 = _is_a5_sku(after_bundle.hardware_sku)
    is_a2 = (not is_a5) and any(tok in after_bundle.hardware_sku.lower() for tok in ("910b", "a2"))
    has_mla = any("mla" in op.submodule.lower() for op in after_bundle.operators)
    has_mso = any(op.stream_index == 1 for op in after_bundle.operators)

    kv_flag = "fp8_e4m3" if is_a5 else ("int8" if mem.cache_dtype in {"C8", "FP8"} else "auto")
    add_cfg: dict[str, Any] = dict(parsed.additional_config)
    add_cfg["enable_fused_mc2"] = 3 if mem.ep >= 8 else 2
    add_cfg["enable_flashcomm1"] = True
    if has_mso:
        add_cfg["multistream_overlap_shared_expert"] = True
        add_cfg["enable_shared_expert_dp"] = True
    if not is_a5:
        add_cfg["enable_weight_nz_layout"] = True

    comp_cfg = dict(parsed.compilation_config)
    comp_cfg["cudagraph_mode"] = "FULL_DECODE_ONLY"
    model_target = parsed.model_arg or f"/weights/{after_bundle.model_name}"

    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by mindie-sim launch-script audit & tuning",
        f"# Model: {after_bundle.model_name} | Hardware: {after_bundle.hardware_sku} | "
        f"Parallel: DP={mem.dp}, TP={mem.tp}, EP={mem.ep}, PP={mem.pp}, DCP={mem.dcp}",
        "set -euo pipefail",
        "",
        'export HCCL_OP_EXPANSION_MODE="AIV"',
        f"export HCCL_BUFFSIZE={DEFAULT_HCCL_BUFFSIZE_MIB}",
        f"export HCCL_BUFFSIZE_EP={DEFAULT_HCCL_EP_BUFFSIZE_MIB}",
        "export VLLM_ASCEND_ENABLE_FUSED_MC2=1",
    ]
    if not is_a2:
        lines.append("export HCCL_INTRA_PCIE_ENABLE=1")
        lines.append("export HCCL_INTRA_ROCE_ENABLE=0")
    if has_mla:
        lines.append("export VLLM_ASCEND_ENABLE_MLAPO=1")
    lines.append('export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"')
    if "ASCEND_RT_VISIBLE_DEVICES" in parsed.env_vars:
        lines.append(f'export ASCEND_RT_VISIBLE_DEVICES="{parsed.env_vars["ASCEND_RT_VISIBLE_DEVICES"]}"')
    lines.extend([
        "",
        f'vllm serve "{model_target}" \\',
        f"  --tensor-parallel-size {mem.tp} \\",
        f"  --data-parallel-size {mem.dp} \\",
    ])
    if mem.pp > 1:
        lines.append(f"  --pipeline-parallel-size {mem.pp} \\")
    if mem.ep > 1:
        lines.append("  --enable-expert-parallel \\")
    if mem.dcp > 1:
        lines.append(f"  --decode-context-parallel-size {mem.dcp} \\")
    lines.extend([
        "  --quantization ascend \\",
        f"  --kv-cache-dtype {kv_flag} \\",
        f"  --max-model-len {max(parsed.max_model_len, mem.seq_len)} \\",
        f"  --max-num-seqs {max(parsed.max_num_seqs, mem.concurrency)} \\",
        f"  --gpu-memory-utilization {mem.gpu_memory_utilization:.2f} \\",
        f"  --compilation-config '{json.dumps(comp_cfg, ensure_ascii=False)}' \\",
        f"  --additional-config '{json.dumps(add_cfg, ensure_ascii=False)}'",
        "",
    ])
    script_text = "\n".join(lines)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(script_text)
            os.replace(tmp_path, target)
        except TimeoutError:
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    return script_text
