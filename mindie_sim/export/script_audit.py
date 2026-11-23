"""Safe static launch-script parser, diagnostic auditor, and tuned-script generator.

[Role] Support ``mindie-sim report --from-script <serve.sh>`` workflows without
  ever executing user shell scripts:
  1. ``parse_launch_script(path_or_text)`` extracts env vars, ``vllm serve`` /
     ``api_server`` flags, JSON config flags, device counts, and catalog/local
     model + quantization hints via stdlib ``shlex``.
  2. ``audit_launch_script(...)`` compares original effective settings against
     the tuned shipping bundle with strict per-generation hardware isolation
     (A5 CCU + UB-Direct vs A3 HCCS AIV vs A2 8P HCCS AIV + RoCE), respecting
     ``enable_mlapo=True`` defaults and the MC2/MSO mutex, and computes signed
     Two-Step Gain (Original -> Tuned Shipping -> Target).
  3. ``render_optimized_serve_sh(...)`` emits a ready-to-run, hardware-matched
     ``<prefix>_optimized_serve.sh`` strictly from the tuned shipping bundle.
[Invariants] Strictly static parsing via ``shlex`` (ZERO ``subprocess``, ``eval``,
  or shell execution). Strict A5 vs A3 vs A2 isolation: zero ``CCU`` tokens on
  A2/A3, zero ``HCCL_INTRA_PCIE_ENABLE`` or blanket MXFP ``FRACTAL_NZ`` on A5,
  zero mixed ``AIV``/``CCU`` labels, and atomic UTF-8 LF file writes.
[Timeouts & Locks] Pure O(script length) in-memory parsing; no locks or network I/O.
[Side-Effect Policy] Read-only except ``render_optimized_serve_sh`` when
  ``output_path`` is explicitly supplied.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any

from mindie_sim.catalog.npu_checkpoints import (
    msmodelslim_quarot_guidance,
    resolve_default_checkpoint,
    validate_quant_for_hardware,
)
from mindie_sim.catalog.quant import LONG_CONTEXT_KV_QUANT_THRESHOLD
from mindie_sim.stacks.vllm_ascend.speculative import (
    build_va_serving_rules,
    validate_aclgraph_capture_sizes,
)

from .models import GIB_BYTES, US_PER_MS, ExportBundle

SCRIPT_AUDIT_SCHEMA = "mindie-sim.script-audit/1"
HIGH_TP_AUDIT_THRESHOLD = 4
DEFAULT_HCCL_BUFFSIZE_MIB = 512
DEFAULT_HCCL_EP_BUFFSIZE_MIB = 2048
MAX_STATIC_SCRIPT_BYTES = 262144
DEFAULT_FUSED_MC2_LEVEL = 2
AUDIT_MS_DECIMALS = 3
AUDIT_RATE_DECIMALS = 2

_ENV_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_KNOWN_ENV_KEYS = frozenset({
    "HCCL_OP_EXPANSION_MODE", "VLLM_ASCEND_ENABLE_MLAPO", "VLLM_ASCEND_ENABLE_FUSED_MC2",
    "HCCL_BUFFSIZE", "HCCL_BUFFSIZE_EP", "HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE",
    "OMP_NUM_THREADS", "OMP_PROC_BIND", "PYTORCH_NPU_ALLOC_CONF", "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_USE_V1", "TASK_QUEUE_ENABLE",
})
_QUANT_BASENAME_TOKENS: tuple[tuple[str, str], ...] = (
    ("w4a8-mxfp", "mxfp8_mxfp4"), ("w4a8_mxfp", "mxfp8_mxfp4"), ("mxfp8_mxfp4", "mxfp8_mxfp4"),
    ("mxfp4", "mxfp4"), ("mxfp8", "mxfp8"), ("w4a8c8", "w4a8"), ("w4a8", "w4a8"),
    ("w8a8c8", "w8a8"), ("w8a8", "w8a8"), ("fp8", "fp8"), ("bf16", "bf16"),
)
_A5_SKU_TOKENS = ("950", "a5", "910_95", "9582", "9581", "9579", "958b", "9599", "dav_3510", "3510", "arch35")
_A3_SKU_TOKENS = ("910_93", "910c", "a3", "9392", "9382", "arch22", "dav_22")
_A2_SKU_TOKENS = ("910b", "a2", "310p", "300i", "arch20", "dav_20")
_A5_VALID_HCCL_MODES = frozenset({"CCU_SCHED", "CCU_MS", "CCU"})
_A5_DEFAULT_HCCL_MODE = "CCU_SCHED"
_A23_VALID_HCCL_MODES = frozenset({"AIV"})
_A23_DEFAULT_HCCL_MODE = "AIV"
_MLA_FAMILIES = frozenset({"mla", "mla_prolog", "paged_mla", "flash_mla", "sparse_mla"})
_FUSED_MC2_KERNEL_TOKENS = ("megamoe", "dispatch_ffn_combine", "fused_mc2")
_UNQUANTIZED_FORMATS = frozenset({"", "AUTO", "BF16", "FP16", "FLOAT", "NONE"})
_INTEGER_WEIGHT_TOKENS = ("W8A8", "W4A8", "W4A4", "INT8", "INT4")
_A5_MXFP_TOKENS = ("MXFP", "FP8", "FP4")
_FALSE_ENV_TOKENS = frozenset({"0", "false", "no", "off"})


@lru_cache(maxsize=1)
def _dynamic_catalog_hints() -> tuple[tuple[str, str], ...]:
    from mindie_sim.model_catalog import CATALOG, MODEL_ALIASES

    pairs: dict[str, str] = {}
    for canonical in CATALOG:
        low = str(canonical).strip().lower()
        pairs[low] = canonical
        pairs[low.replace("-", "_")] = canonical
    for alias, target in MODEL_ALIASES.items():
        low = str(alias).strip().lower()
        pairs.setdefault(low, str(target))
        pairs.setdefault(low.replace("-", "_"), str(target))
    return tuple(sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0])))


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
            "source_label": self.source_label, "model_arg": self.model_arg,
            "inferred_catalog_model": self.inferred_catalog_model,
            "local_config_path": self.local_config_path,
            "inferred_quant": self.inferred_quant, "quantization_flag": self.quantization_flag,
            "kv_cache_dtype": self.kv_cache_dtype,
            "tp": self.tp, "dp": self.dp, "pp": self.pp, "ep_enabled": self.ep_enabled, "dcp": self.dcp,
            "inferred_num_devices": self.inferred_num_devices, "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs, "enforce_eager": self.enforce_eager,
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
            "issue_id": self.issue_id, "severity": self.severity, "title": self.title,
            "current": self.current_value, "recommended": self.recommended_value, "impact": self.impact_note,
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
            logical_lines.append((buf + " " + stripped).strip())
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
    if not model_arg:
        return "", "", ""
    local_cfg, quant_hint = "", ""
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
                    for needle, qname in (("W4A8_MXFP", "mxfp8_mxfp4"), ("MXFP4", "mxfp8_mxfp4"), ("W8A8_MXFP8", "mxfp8"), ("W4A8", "w4a8"), ("W8A8", "w8a8")):
                        if needle in vals:
                            quant_hint = qname
                            break
                except TimeoutError:
                    raise
                except (OSError, ValueError):
                    pass

    base_low = Path(model_arg.rstrip("/")).name.lower()
    if not quant_hint:
        for tok, preset in _QUANT_BASENAME_TOKENS:
            if tok in base_low:
                quant_hint = preset
                break
    catalog_model, norm_base = "", base_low.replace("_", "-")
    for needle, canonical in _dynamic_catalog_hints():
        if needle in base_low or needle in norm_base:
            catalog_model = canonical
            break
    return catalog_model, local_cfg, quant_hint


def _parse_vllm_tokens(tokens: list[str], state: dict[str, Any]) -> None:
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "serve" and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            state["model_arg"] = tokens[idx + 1]
            idx += 2
            continue
        if "=" in tok and tok.startswith("--"):
            flag, nxt, step = (*tok.split("=", 1), 1)
        else:
            flag, nxt, step = tok, (tokens[idx + 1] if idx + 1 < len(tokens) else ""), 2

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
            state["ep_enabled"], step = True, 1
        elif flag == "--enforce-eager":
            state["enforce_eager"], step = True, 1
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
        raw_text, src_label = str(path_or_text or "")[:MAX_STATIC_SCRIPT_BYTES], "inline_command"

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
        pos = 1 if tokens[0] == "export" else 0
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
        source_label=src_label, model_arg=str(state["model_arg"]), inferred_catalog_model=cat_model,
        local_config_path=local_cfg, inferred_quant=q_hint, quantization_flag=str(state["quantization_flag"]),
        kv_cache_dtype=str(state["kv_cache_dtype"]), tp=int(state["tp"]), dp=int(state["dp"]), pp=int(state["pp"]),
        ep_enabled=bool(state["ep_enabled"]), dcp=int(state["dcp"]), inferred_num_devices=inferred_devs,
        max_model_len=int(state["max_model_len"]), max_num_seqs=int(state["max_num_seqs"]),
        max_num_batched_tokens=int(state["max_num_batched_tokens"]), block_size=int(state["block_size"]),
        gpu_memory_utilization=float(state["gpu_memory_utilization"]), enforce_eager=bool(state["enforce_eager"]),
        env_vars=env_vars, compilation_config=dict(state["compilation_config"]),
        speculative_config=dict(state["speculative_config"]), additional_config=dict(state["additional_config"]),
    )


def _resolve_audit_hw_family(hardware_sku: str) -> str:
    low = str(hardware_sku or "").strip().lower()
    if any(tok in low for tok in _A5_SKU_TOKENS):
        return "a5"
    if any(tok in low for tok in _A3_SKU_TOKENS):
        return "a3"
    if any(tok in low for tok in _A2_SKU_TOKENS):
        return "a2"
    return "unknown"


def _bundle_has_mla(bundle: ExportBundle) -> bool:
    return any(op.family.lower() in _MLA_FAMILIES or "mla_prolog" in op.ascend_kernel.lower() for op in bundle.operators)


def _bundle_moe_overlap_mode(bundle: ExportBundle) -> tuple[str, int]:
    eff = bundle.effective_config if isinstance(bundle.effective_config, Mapping) else {}
    raw_mc2 = eff.get("enable_fused_mc2")
    if isinstance(raw_mc2, int) and not isinstance(raw_mc2, bool) and raw_mc2 > 0:
        return ("mc2", int(raw_mc2))
    if bool(eff.get("multistream_overlap_shared_expert")):
        return ("mso", 0)
    if any(any(t in op.ascend_kernel.lower() or t in op.family.lower() for t in _FUSED_MC2_KERNEL_TOKENS) for op in bundle.operators):
        return ("mc2", DEFAULT_FUSED_MC2_LEVEL)
    if any(op.stream_index == 1 for op in bundle.operators):
        return ("mso", 0)
    return ("none", 0)


def _bundle_effective_tpot_ms(bundle: ExportBundle) -> float:
    return (float(bundle.step_span_us) / US_PER_MS) / max(1.0, float(bundle.avg_accepted_tokens)) + float(bundle.handoff_tpot_ms or 0.0)


def _bundle_weight_format(bundle: ExportBundle) -> str:
    mem = bundle.memory_ledger
    return str(mem.weight_quant_display or mem.weight_quant or "")


def _is_integer_quant_format(weight_format: str) -> bool:
    return any(tok in str(weight_format or "").upper() for tok in _INTEGER_WEIGHT_TOKENS)


def _is_mxfp_or_fp8_format(weight_format: str) -> bool:
    return any(tok in str(weight_format or "").upper() for tok in _A5_MXFP_TOKENS)


def _resolve_target_hccl_mode(hw_fam: str, ref_bundle: ExportBundle) -> str:
    eff = ref_bundle.effective_config if isinstance(ref_bundle.effective_config, Mapping) else {}
    declared = str(eff.get("hccl_op_expansion_mode") or "").strip().upper()
    if hw_fam == "a5":
        return declared if declared in _A5_VALID_HCCL_MODES else _A5_DEFAULT_HCCL_MODE
    return _A23_DEFAULT_HCCL_MODE if hw_fam in {"a3", "a2"} else declared


def _audit_isa_and_speculative_rules(
    parsed: ParsedLaunchScript, *, ref_bundle: ExportBundle, hw_fam: str, quarot_info: Mapping[str, Any] | None,
) -> list[ScriptAuditIssue]:
    mem_r, found = ref_bundle.memory_ledger, []
    if hw_fam in {"a2", "a3", "a5"}:
        isa_warns = validate_quant_for_hardware(
            parsed.inferred_quant or parsed.quantization_flag, hw_fam, kv_cache_dtype=parsed.kv_cache_dtype or "bf16",
        )
        if isa_warns:
            found.append(ScriptAuditIssue(
                "hw_isa_quant_incompatible", "CRITICAL", f"硬件 ISA 不支持所选量化/KV 格式 ({hw_fam.upper()})",
                f"quant={parsed.inferred_quant or parsed.quantization_flag}, kv={parsed.kv_cache_dtype or 'auto'}",
                str((quarot_info or {}).get("target_recipe", _bundle_weight_format(ref_bundle))), "; ".join(isa_warns),
            ))
    spec_k = int(parsed.speculative_config.get("num_speculative_tokens") or ref_bundle.spec_proposals or 0)
    raw_cap_sizes = parsed.compilation_config.get("cudagraph_capture_sizes")
    if isinstance(raw_cap_sizes, list) and spec_k > 0:
        cap_warns = validate_aclgraph_capture_sizes(
            [int(x) for x in raw_cap_sizes if isinstance(x, int)], num_speculative_tokens=spec_k,
            cudagraph_mode=str(parsed.compilation_config.get("cudagraph_mode") or "FULL_DECODE_ONLY"),
        )
        if cap_warns:
            found.append(ScriptAuditIssue(
                "aclgraph_capture_size_not_k_plus_1", "HIGH",
                f"ACLGraph capture_sizes 非 (K+1)={spec_k + 1} 的整数倍",
                str(raw_cap_sizes), f"设置为 n * {spec_k + 1} 倍数", cap_warns[0],
            ))
    if hw_fam in {"a2", "a3", "a5"}:
        va_rules = build_va_serving_rules(
            ref_bundle.model_name, hw_family=hw_fam, tp=mem_r.tp, dp=mem_r.dp, ep=mem_r.ep, pp=mem_r.pp,
            target_kv_dtype=mem_r.cache_dtype.lower(), enable_speculative=(spec_k > 0),
            num_speculative_tokens=spec_k if spec_k > 0 else None, max_num_seqs=max(1, mem_r.concurrency),
        )
        if parsed.speculative_config and va_rules.resolved_speculative is not None:
            req_dtp, got_dtp = va_rules.resolved_speculative.draft_tensor_parallel_size, parsed.speculative_config.get("draft_tensor_parallel_size")
            if req_dtp is not None and got_dtp is not None and int(got_dtp) != int(req_dtp):
                found.append(ScriptAuditIssue(
                    "draft_tp_policy_mismatch", "HIGH",
                    f"投机草稿 draft_tensor_parallel_size 与 {va_rules.resolved_speculative.method} 规则冲突",
                    f"draft_tensor_parallel_size={got_dtp}", f"draft_tensor_parallel_size={req_dtp}",
                    f"vLLM Ascend 规则要求 {va_rules.resolved_speculative.method} 使用 draft_tp={req_dtp}",
                ))
    return found


def _audit_comm_and_layout_rules(parsed: ParsedLaunchScript, *, ref_bundle: ExportBundle, hw_fam: str) -> list[ScriptAuditIssue]:
    issues: list[ScriptAuditIssue] = []
    mem_r = ref_bundle.memory_ledger
    hccl_mode = parsed.env_vars.get("HCCL_OP_EXPANSION_MODE", "").strip().upper()
    target_mode = _resolve_target_hccl_mode(hw_fam, ref_bundle)

    if hw_fam == "a5":
        if hccl_mode not in _A5_VALID_HCCL_MODES:
            issues.append(ScriptAuditIssue(
                "hccl_expansion_mode_suboptimal", "HIGH", "A5 通信展开模式未切换至硬件 CCU 调度",
                hccl_mode or "unset (默认 AI_CPU/HOST)", f'HCCL_OP_EXPANSION_MODE="{target_mode}"',
                "A5 UB-Direct 通信应启用硬件 CCU 调度以降低小包通信启动开销",
            ))
        legacy_keys = [k for k in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE") if k in parsed.env_vars]
        if legacy_keys:
            issues.append(ScriptAuditIssue(
                "a5_legacy_transport_override", "MEDIUM", "A5 脚本包含遗留 PCIe/RoCE 机内环境变量",
                ", ".join(legacy_keys), "移除 HCCL_INTRA_PCIE_ENABLE / HCCL_INTRA_ROCE_ENABLE",
                "A5 使用 UB-Direct 互连，不应设置遗留 PCIe/RoCE 覆盖开关",
            ))
        if _is_mxfp_or_fp8_format(_bundle_weight_format(ref_bundle)) and parsed.additional_config.get("enable_weight_nz_layout"):
            issues.append(ScriptAuditIssue(
                "a5_mxfp_nz_layout_incompatible", "MEDIUM", "A5 MXFP/FP8 权重不应强制开启全局 NZ 排布",
                "enable_weight_nz_layout=true", "移除 enable_weight_nz_layout",
                "A5 MXFP 排布由具体算子 ABI 决定，不应沿用旧架构全局开关",
            ))
    elif hw_fam in {"a3", "a2"}:
        check_hccl = (hw_fam == "a2") or (ref_bundle.phase == "decode")
        if check_hccl and hccl_mode not in _A23_VALID_HCCL_MODES:
            cur_val = "该设备不支持所选通信展开模式" if (hccl_mode and "CCU" in hccl_mode) else (hccl_mode or "unset (默认 AI_CPU/HOST)")
            title = "A3 Decode 通信展开模式未切换至 HCCS AIV" if hw_fam == "a3" else "A2 通信展开模式未切换至 8P HCCS AIV"
            note = "A3 HCCS Decode 启用 AIV 展开可降低小包通信 AI_CPU/Host 启动惩罚" if hw_fam == "a3" else "A2 8P HCCS 域启用 AIV 展开可降低小包通信 AI_CPU/Host 启动惩罚"
            issues.append(ScriptAuditIssue("hccl_expansion_mode_suboptimal", "HIGH", title, cur_val, f'HCCL_OP_EXPANSION_MODE="{target_mode}"', note))
        if _is_integer_quant_format(_bundle_weight_format(ref_bundle)) and not parsed.additional_config.get("enable_weight_nz_layout"):
            gen = hw_fam.upper()
            issues.append(ScriptAuditIssue(
                "missing_nz_layout_a23", "MEDIUM", f"{gen} 整数量化未开启 FRACTAL_NZ 权重排布",
                "enable_weight_nz_layout unset", '--additional-config \'{"enable_weight_nz_layout": true}\'',
                f"{gen} INT8/INT4 权重算子需 FRACTAL_NZ 排布获得最优 Cube 访存效率",
            ))
    return issues


def _audit_runtime_mechanism_rules(parsed: ParsedLaunchScript, *, ref_bundle: ExportBundle) -> list[ScriptAuditIssue]:
    issues: list[ScriptAuditIssue] = []
    if ref_bundle.phase == "decode" and (
        parsed.enforce_eager or str(parsed.compilation_config.get("cudagraph_mode", "")).upper() != "FULL_DECODE_ONLY"
    ):
        issues.append(ScriptAuditIssue(
            "missing_aclgraph_full_decode", "HIGH", "未启用 ACLGraph FULL_DECODE_ONLY 图捕获",
            "--enforce-eager" if parsed.enforce_eager else str(parsed.compilation_config or "unset"),
            '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'',
            "移除 --enforce-eager 并开启 FULL_DECODE_ONLY 消除 Decode 逐算子发射气泡",
        ))
    if _bundle_has_mla(ref_bundle):
        raw_mlapo = parsed.env_vars.get("VLLM_ASCEND_ENABLE_MLAPO")
        if (raw_mlapo is not None and raw_mlapo.strip().lower() in _FALSE_ENV_TOKENS) or parsed.additional_config.get("enable_mlapo") is False:
            issues.append(ScriptAuditIssue(
                "missing_mlapo", "MEDIUM", "MLA 模型显式禁用了 MLAPO 前置融合",
                f"VLLM_ASCEND_ENABLE_MLAPO={raw_mlapo}" if raw_mlapo is not None else "enable_mlapo=false",
                "恢复默认开启或设置 export VLLM_ASCEND_ENABLE_MLAPO=1",
                "vLLM Ascend 默认开启 MLAPO；显式关闭会拆散 MLA Prolog 融合算子",
            ))
    user_mc2_val = parsed.additional_config.get("enable_fused_mc2")
    user_has_mc2 = isinstance(user_mc2_val, int) and not isinstance(user_mc2_val, bool) and user_mc2_val > 0
    user_has_mso = bool(parsed.additional_config.get("multistream_overlap_shared_expert"))
    tuned_mode, tuned_mc2_level = _bundle_moe_overlap_mode(ref_bundle)
    if user_has_mc2 and user_has_mso:
        rec = f'--additional-config \'{{"enable_fused_mc2": {tuned_mc2_level}}}\'' if tuned_mode == "mc2" else '--additional-config \'{"multistream_overlap_shared_expert": true}\''
        issues.append(ScriptAuditIssue(
            "mc2_mso_mutex_conflict", "HIGH", "enable_fused_mc2 与 multistream_overlap_shared_expert 互斥冲突",
            f"enable_fused_mc2={user_mc2_val}, multistream_overlap_shared_expert=true", rec,
            "vLLM Ascend 上游规则禁止同时开启融合 MC2 与共享专家多流掩盖，须二选一",
        ))
    elif tuned_mode == "mc2" and not user_has_mc2:
        issues.append(ScriptAuditIssue(
            "missing_fused_mc2", "MEDIUM", "未启用与调优配置一致的 MC2 通算融合",
            str(parsed.additional_config or "unset"),
            f'export VLLM_ASCEND_ENABLE_FUSED_MC2=1 && --additional-config \'{{"enable_fused_mc2": {tuned_mc2_level}}}\'',
            "与调优后 shipping 配置对齐，启用合法的 MC2 融合等级（不与 MSO 混用）",
        ))
    elif tuned_mode == "mso" and not user_has_mso:
        issues.append(ScriptAuditIssue(
            "missing_mso_shared_expert", "MEDIUM", "未启用共享专家多流异步掩盖 (MSO)",
            str(parsed.additional_config or "unset"), '--additional-config \'{"multistream_overlap_shared_expert": true}\'',
            "在独立多流路径下用主计算流 MoE 计算掩盖共享专家执行（不与融合 MC2 混用）",
        ))
    return issues


def audit_launch_script(
    parsed: ParsedLaunchScript, *, before_bundle: ExportBundle, after_bundle: ExportBundle,
    tuned_bundle: ExportBundle | None = None, optimized_script_path: str = "",
) -> ScriptAuditReport:
    """Audit a parsed user launch script against tuned shipping facts and compute signed Two-Step Gain."""
    ref_bundle = tuned_bundle if tuned_bundle is not None else after_bundle
    hw_fam = _resolve_audit_hw_family(ref_bundle.hardware_sku)
    ckpt = resolve_default_checkpoint(ref_bundle.model_name, hw_family=hw_fam) if hw_fam in {"a2", "a3", "a5"} else None
    quarot_info = msmodelslim_quarot_guidance(hw_fam) if hw_fam in {"a2", "a3", "a5"} else None
    mem_b, mem_r = before_bundle.memory_ledger, ref_bundle.memory_ledger

    issues = _audit_isa_and_speculative_rules(parsed, ref_bundle=ref_bundle, hw_fam=hw_fam, quarot_info=quarot_info)
    w_fmt = _bundle_weight_format(ref_bundle)
    if not parsed.quantization_flag and parsed.inferred_quant in {"", "bf16"} and w_fmt.upper() not in _UNQUANTIZED_FORMATS:
        ckpt_hint = f" | 推荐权重: {ckpt.repo_id}" if ckpt is not None else ""
        recipe = str((quarot_info or {}).get("target_recipe", w_fmt))
        issues.append(ScriptAuditIssue(
            "missing_quantization_bf16", "HIGH", "未启用生产量化 (原生 BF16 显存占用与带宽压力高)",
            "未设置 --quantization (BF16)", f"--quantization ascend ({recipe}){ckpt_hint}",
            f"原生 BF16 单卡权重={mem_r.native_bf16_weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB -> "
            f"量化后={mem_r.weight_footprint_per_die_bytes / GIB_BYTES:.1f} GiB (压缩 {mem_r.weight_compression_ratio:.2f}x)",
        ))
    ctx = max(parsed.max_model_len, before_bundle.context_tokens)
    if (
        mem_r.cache_dtype.upper() not in _UNQUANTIZED_FORMATS
        and parsed.kv_cache_dtype.strip().upper() in _UNQUANTIZED_FORMATS
        and (ctx >= LONG_CONTEXT_KV_QUANT_THRESHOLD or mem_r.cache_dtype.upper() in {"FP8", "FP8_E4M3", "C8", "INT8"})
    ):
        rec_kv = "--kv-cache-dtype fp8_e4m3" if hw_fam == "a5" else "--kv-cache-dtype int8 (C8)"
        issues.append(ScriptAuditIssue(
            "missing_kv_quant", "HIGH", "未启用与调优配置一致的 KV Cache 量化",
            parsed.kv_cache_dtype or "auto (BF16)", rec_kv,
            f"调优配置采用 {mem_r.cache_dtype} KV Cache 以压缩显存并降低注意力访存带宽",
        ))
    issues.extend(_audit_runtime_mechanism_rules(parsed, ref_bundle=ref_bundle))
    issues.extend(_audit_comm_and_layout_rules(parsed, ref_bundle=ref_bundle, hw_fam=hw_fam))
    if parsed.tp >= HIGH_TP_AUDIT_THRESHOLD and parsed.tp > mem_r.tp:
        issues.append(ScriptAuditIssue(
            "suboptimal_high_tp", "MEDIUM", "TP 切分过细导致跨卡小包通信偏重",
            f"DP={parsed.dp}, TP={parsed.tp}", f"DP={mem_r.dp}, TP={mem_r.tp}, EP={mem_r.ep}",
            "降低 TP 并提升 DP/EP 可直接减少 Attention/Norm 跨卡集合通信并提高集群吞吐",
        ))
    if not mem_b.fits_hbm:
        issues.append(ScriptAuditIssue(
            "hbm_oom_in_original_script", "CRITICAL", "原脚本配置存在 HBM OOM 风险",
            f"Peak HBM={mem_b.max_stage_hbm_bytes / GIB_BYTES:.1f} GiB > Usable={mem_b.usable_hbm_bytes / GIB_BYTES:.1f} GiB",
            f"调优后 Peak HBM={mem_r.max_stage_hbm_bytes / GIB_BYTES:.1f} GiB (Fit={mem_r.fits_hbm})",
            "通过生产量化、KV 量化与可行并行切分消除 HBM OOM 并释放并发空间",
        ))

    b_step, a_step = float(before_bundle.step_span_us) / US_PER_MS, float(after_bundle.step_span_us) / US_PER_MS
    b_tpot, a_tpot = _bundle_effective_tpot_ms(before_bundle), _bundle_effective_tpot_ms(after_bundle)
    orig_fits, tuned_evaluated = bool(mem_b.fits_hbm), tuned_bundle is not None
    if tuned_bundle is not None:
        t_step, t_tpot = float(tuned_bundle.step_span_us) / US_PER_MS, _bundle_effective_tpot_ms(tuned_bundle)
        imm_saved_ms = round(b_step - t_step, AUDIT_MS_DECIMALS) if orig_fits else None
        deep_from_tuned_ms = round(t_step - a_step, AUDIT_MS_DECIMALS)
        t_tps = round(float(tuned_bundle.throughput_per_die_tokens_per_s), AUDIT_RATE_DECIMALS)
    else:
        t_step = t_tpot = imm_saved_ms = deep_from_tuned_ms = t_tps = None

    immediate = {
        "label": "1. 立即可得收益 (0 算子开发 — 仅应用调优后 shipping 脚本)",
        "tuned_evaluated": tuned_evaluated, "original_fits_hbm": orig_fits,
        "before_step_ms": round(b_step, AUDIT_MS_DECIMALS), "before_effective_tpot_ms": round(b_tpot, AUDIT_MS_DECIMALS),
        "tuned_step_ms": round(t_step, AUDIT_MS_DECIMALS) if t_step is not None else None,
        "saved_ms": imm_saved_ms,
        "tuned_effective_tpot_ms": round(t_tpot, AUDIT_MS_DECIMALS) if t_tpot is not None else None,
        "tuned_single_die_tps": t_tps, "issues_fixed_count": len(issues),
    }
    deep = {
        "label": "2. 深度攻关终态收益 (Target 算子融合/调优/框架改造)",
        "original_fits_hbm": orig_fits, "before_step_ms": round(b_step, AUDIT_MS_DECIMALS),
        "after_step_ms": round(a_step, AUDIT_MS_DECIMALS),
        "saved_ms": round(b_step - a_step, AUDIT_MS_DECIMALS) if orig_fits else None,
        "saved_from_tuned_ms": deep_from_tuned_ms,
        "after_effective_tpot_ms": round(a_tpot, AUDIT_MS_DECIMALS),
        "after_single_die_tps": round(float(after_bundle.throughput_per_die_tokens_per_s), AUDIT_RATE_DECIMALS),
        "speedup_vs_original": round(b_step / a_step, AUDIT_RATE_DECIMALS) if (orig_fits and a_step > 0.0) else None,
    }
    return ScriptAuditReport(
        schema=SCRIPT_AUDIT_SCHEMA, parsed=parsed, issues=tuple(issues),
        immediate_step_summary=immediate, deep_step_summary=deep, optimized_script_path=optimized_script_path,
    )


def render_optimized_serve_sh(
    parsed: ParsedLaunchScript, *, after_bundle: ExportBundle,
    tuned_bundle: ExportBundle | None = None, output_path: str | Path | None = None,
) -> str:
    """Render (and optionally atomically save) a hardware-matched ``optimized_serve.sh``."""
    ref = tuned_bundle if tuned_bundle is not None else after_bundle
    mem, hw_fam = ref.memory_ledger, _resolve_audit_hw_family(ref.hardware_sku)
    has_mla, (overlap_mode, mc2_level) = _bundle_has_mla(ref), _bundle_moe_overlap_mode(ref)
    ckpt = resolve_default_checkpoint(ref.model_name, hw_family=hw_fam) if hw_fam in {"a2", "a3", "a5"} else None
    va_rules = (
        build_va_serving_rules(
            ref.model_name, hw_family=hw_fam, tp=mem.tp, dp=mem.dp, ep=mem.ep, pp=mem.pp,
            target_kv_dtype=mem.cache_dtype.lower(), enable_speculative=(ref.spec_proposals > 0),
            num_speculative_tokens=(ref.spec_proposals if ref.spec_proposals > 0 else None),
            max_num_seqs=max(parsed.max_num_seqs, mem.concurrency),
        ) if hw_fam in {"a2", "a3", "a5"} else None
    )
    kv_flag = (
        ("fp8_e4m3" if mem.cache_dtype.upper() in {"FP8", "FP8_E4M3"} else "auto") if hw_fam == "a5"
        else ("int8" if mem.cache_dtype.upper() in {"C8", "INT8"} else "auto") if hw_fam in {"a2", "a3"} else "auto"
    )
    add_cfg: dict[str, Any] = dict(parsed.additional_config)
    if va_rules is not None:
        add_cfg.update(va_rules.additional_config)
    for k in ("enable_fused_mc2", "multistream_overlap_shared_expert", "enable_shared_expert_dp", "enable_weight_nz_layout"):
        add_cfg.pop(k, None)
    if overlap_mode == "mc2":
        add_cfg["enable_fused_mc2"] = int(mc2_level)
    elif overlap_mode == "mso":
        add_cfg["multistream_overlap_shared_expert"] = True
        add_cfg["enable_shared_expert_dp"] = True
    if hw_fam in {"a2", "a3"} and _is_integer_quant_format(_bundle_weight_format(ref)):
        add_cfg["enable_weight_nz_layout"] = True

    comp_cfg = dict(parsed.compilation_config)
    if va_rules is not None:
        comp_cfg.update(va_rules.compilation_config)
    if ref.phase == "decode":
        comp_cfg["cudagraph_mode"] = "FULL_DECODE_ONLY"
    model_target = parsed.model_arg or f"/weights/{ref.model_name}"
    hccl_mode = _resolve_target_hccl_mode(hw_fam, ref)

    lines = [
        "#!/usr/bin/env bash", "# Auto-generated by mindie-sim launch-script audit & tuning",
        f"# Model: {ref.model_name} | Hardware: {ref.hardware_sku} | Parallel: DP={mem.dp}, TP={mem.tp}, EP={mem.ep}, PP={mem.pp}, DCP={mem.dcp}",
    ]
    if ckpt is not None:
        lines.append(f"# Recommended Checkpoint: {ckpt.repo_id} ({ckpt.quant_preset})")
    lines.extend(["set -euo pipefail", ""])
    if hccl_mode and (hw_fam in {"a5", "a2"} or (hw_fam == "a3" and ref.phase == "decode")):
        lines.append(f'export HCCL_OP_EXPANSION_MODE="{hccl_mode}"')
    lines.extend([f"export HCCL_BUFFSIZE={DEFAULT_HCCL_BUFFSIZE_MIB}", f"export HCCL_BUFFSIZE_EP={DEFAULT_HCCL_EP_BUFFSIZE_MIB}"])
    if overlap_mode == "mc2":
        lines.append("export VLLM_ASCEND_ENABLE_FUSED_MC2=1")
    if has_mla:
        lines.append("export VLLM_ASCEND_ENABLE_MLAPO=1")
    lines.append('export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"')
    if "ASCEND_RT_VISIBLE_DEVICES" in parsed.env_vars:
        lines.append(f'export ASCEND_RT_VISIBLE_DEVICES="{parsed.env_vars["ASCEND_RT_VISIBLE_DEVICES"]}"')
    lines.extend(["", f'vllm serve "{model_target}" \\', f"  --tensor-parallel-size {mem.tp} \\", f"  --data-parallel-size {mem.dp} \\"])
    if mem.pp > 1:
        lines.append(f"  --pipeline-parallel-size {mem.pp} \\")
    if mem.ep > 1:
        lines.append("  --enable-expert-parallel \\")
    if mem.dcp > 1:
        lines.append(f"  --decode-context-parallel-size {mem.dcp} \\")
    if va_rules is not None and va_rules.speculative_enabled and va_rules.speculative_config:
        lines.append(f"  --speculative-config '{json.dumps(va_rules.speculative_config, ensure_ascii=False)}' \\")
    if _bundle_weight_format(ref).upper() not in _UNQUANTIZED_FORMATS:
        lines.append("  --quantization ascend \\")
    lines.extend([
        f"  --kv-cache-dtype {kv_flag} \\",
        f"  --max-model-len {max(parsed.max_model_len, mem.seq_len)} \\",
        f"  --max-num-seqs {max(parsed.max_num_seqs, mem.concurrency)} \\",
        f"  --gpu-memory-utilization {mem.gpu_memory_utilization:.2f} \\",
        f"  --compilation-config '{json.dumps(comp_cfg, ensure_ascii=False)}' \\",
        f"  --additional-config '{json.dumps(add_cfg, ensure_ascii=False)}'", "",
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
