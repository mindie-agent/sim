"""vLLM-Ascend static launch-script codec (parse, audit, render).

[Role] Implement ``LaunchConfigCodec`` for ``vllm-ascend``:
  1. ``parse_launch_config(source)`` statically decodes ``vllm serve`` shell
     scripts or CLI commands via ``shlex`` into ``LaunchConfigIngress`` without
     ever executing user code or guessing ambiguous deployment modes.
  2. ``audit_launch_config(ingress, before_facts, tuned_facts)`` diagnoses ISA,
     quantization, KV cache, ACLGraph ``n*(K+1)``, draft TP, MLAPO, MC2/MSO
     mutex, A5 CCU vs A2/A3 AIV HCCL isolation, and HBM OOM bottlenecks.
  3. ``render_launch_config(tuned_facts)`` emits a clean, hardware-isolated
     ``vllm serve`` bash script that round-trips through ``parse_launch_config``.
[Invariants] Strictly static ``shlex`` parsing (zero ``subprocess``/``eval``).
  Zero imports from ``mindie_sim.export`` or ``mindie_sim.compose`` (AG010).
  ``deployment_mode_hint`` is populated ONLY when ``--kv-transfer-config``
  (``kv_role``) or explicit ``--deployment-mode`` unambiguously declares it;
  otherwise ``None`` so Core refuses silent guessing.
[Timeouts & Locks] Pure O(script length) bounded read (<= 256 KiB); no locks or
  network I/O.
[Side-Effect Policy] Read-only over input paths; returns immutable records and
  plain dicts/strings.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any

from mindie_sim.catalog.checkpoint_reader import (
    lookup_curated_checkpoint,
    read_checkpoint_quant_map,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.scenario import DEPLOYMENT_MODES
from mindie_sim.schema.stack import LaunchConfigIngress

from .profiles import (
    VLLM_ASCEND_STACK_ID,
    default_checkpoint_repo,
    normalize_canonical_model,
    normalize_hw_family,
    shipping_collective_policy,
    validate_vllm_ascend_quant_isa,
)
from .speculative import (
    FULL_GRAPH_CUDAGRAPH_MODES,
    WEIGHT_NZ_MODE_DISABLED,
    WEIGHT_NZ_MODE_QUANT,
    build_va_serving_rules,
    validate_aclgraph_capture_sizes,
)

SCRIPT_AUDIT_SCHEMA = "mindie-sim.script-audit/1"
MAX_STATIC_SCRIPT_BYTES = 262144
LONG_CONTEXT_KV_QUANT_THRESHOLD = 32768
HIGH_TP_AUDIT_THRESHOLD = 4
DEFAULT_HCCL_BUFFSIZE_MIB = 512
DEFAULT_HCCL_EP_BUFFSIZE_MIB = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_AUDIT_MAX_MODEL_LEN = 32768
DEFAULT_AUDIT_MAX_NUM_SEQS = 64
DEFAULT_PD_KV_CONNECTOR = "MooncakeConnectorV2"
GIB_BYTES = 1024 * 1024 * 1024
US_PER_MS = 1000.0
AUDIT_MS_DECIMALS = 3
AUDIT_RATE_DECIMALS = 2

_ENV_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_SHELL_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_HUMAN_INT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kKmMgGtT])$")
_HUMAN_INT_SCALES: dict[str, int] = {
    "k": 1000,
    "m": 1000 * 1000,
    "g": 1000 * 1000 * 1000,
    "t": 1000 * 1000 * 1000 * 1000,
    "K": 1024,
    "M": 1024 * 1024,
    "G": 1024 * 1024 * 1024,
    "T": 1024 * 1024 * 1024 * 1024,
}
_KNOWN_ENV_KEYS = frozenset({
    "HCCL_OP_EXPANSION_MODE",
    "VLLM_ASCEND_ENABLE_MLAPO",
    "VLLM_ASCEND_ENABLE_FUSED_MC2",
    "HCCL_BUFFSIZE",
    "HCCL_BUFFSIZE_EP",
    "HCCL_INTRA_PCIE_ENABLE",
    "HCCL_INTRA_ROCE_ENABLE",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "PYTORCH_NPU_ALLOC_CONF",
    "ASCEND_RT_VISIBLE_DEVICES",
    "VLLM_USE_V1",
    "TASK_QUEUE_ENABLE",
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
_A5_VALID_HCCL_MODES = frozenset({"CCU_SCHED", "CCU_MS", "CCU"})
_A5_DEFAULT_HCCL_MODE = "CCU_SCHED"
_A23_VALID_HCCL_MODES = frozenset({"AIV"})
_A23_DEFAULT_HCCL_MODE = "AIV"
_UNQUANTIZED_FORMATS = frozenset({"", "AUTO", "BF16", "FP16", "FLOAT", "NONE"})
_INTEGER_WEIGHT_TOKENS = ("W8A8", "W4A8", "W4A4", "INT8", "INT4")
_A5_MXFP_TOKENS = ("MXFP", "FP8", "FP4")
_FALSE_ENV_TOKENS = frozenset({"0", "false", "no", "off"})
_DOUBLE_QUOTE_ESCAPES = frozenset({'"', "\\", "$", "`"})
_DOTTED_CONFIG_PREFIXES: tuple[tuple[str, str], ...] = (
    ("--additional-config.", "additional_config"),
    ("--compilation-config.", "compilation_config"),
    ("--speculative-config.", "speculative_config"),
    ("--kv-transfer-config.", "kv_transfer_config"),
)
_NON_PD_KV_CONNECTORS = frozenset({
    "offloadingconnector",
    "ascendoffloadingconnector",
    "preemptoffloadconnector",
    "preemptoffloadconnectorv1",
    "ascendstoreconnector",
    "mooncakeconnectorstorev1",
    "ucmconnector",
    "ucmconnectorv1",
    "hisparseconnector",
    "hisparse",
    "lmcacheconnector",
    "lmcacheconnectorv1",
})

_KV_ROLE_TO_DEPLOYMENT_MODE: dict[str, str] = {
    "kv_producer": "pd_prefill",
    "kv_consumer": "pd_decode",
    "kv_both": "pd_disaggregated",
}


def _split_shell_commands(raw_text: str) -> list[list[list[tuple[str, str]]]]:
    """Tokenize static shell text into commands of ``(segment, quote_kind)`` pieces."""
    commands: list[list[list[tuple[str, str]]]] = []
    cmd: list[list[tuple[str, str]]] = []
    tok: list[tuple[str, str]] = []
    buf: list[str] = []
    tok_started = False
    in_single = False
    in_double = False
    idx = 0
    length = len(raw_text)

    def flush_seg(quote_kind: str) -> None:
        if buf:
            tok.append(("".join(buf), quote_kind))
            buf.clear()

    def flush_tok() -> None:
        nonlocal tok_started
        flush_seg("single" if in_single else ("double" if in_double else "none"))
        if tok_started:
            cmd.append(list(tok))
            tok.clear()
            tok_started = False

    def flush_cmd() -> None:
        flush_tok()
        if cmd:
            commands.append(list(cmd))
            cmd.clear()

    while idx < length:
        ch = raw_text[idx]
        if in_single:
            if ch == "'":
                flush_seg("single")
                in_single = False
            else:
                buf.append(ch)
            idx += 1
            continue
        if in_double:
            if ch == '"':
                flush_seg("double")
                in_double = False
                idx += 1
            elif ch == "\\" and idx + 1 < length:
                nxt = raw_text[idx + 1]
                if nxt == "\n":
                    idx += 2
                elif nxt in _DOUBLE_QUOTE_ESCAPES:
                    buf.append(nxt)
                    idx += 2
                else:
                    buf.append(ch)
                    idx += 1
            else:
                buf.append(ch)
                idx += 1
            continue
        if ch == "\\" and idx + 1 < length:
            nxt = raw_text[idx + 1]
            if nxt == "\n":
                idx += 2
                continue
            if nxt == "\r" and idx + 2 < length and raw_text[idx + 2] == "\n":
                idx += 3
                continue
            tok_started = True
            buf.append(nxt)
            idx += 2
            continue
        if ch == "#" and not tok_started:
            while idx < length and raw_text[idx] != "\n":
                idx += 1
            continue
        if ch == "'":
            flush_seg("none")
            in_single = True
            tok_started = True
            idx += 1
            continue
        if ch == '"':
            flush_seg("none")
            in_double = True
            tok_started = True
            idx += 1
            continue
        if ch in {"\n", ";"}:
            flush_cmd()
            idx += 1
            continue
        if ch in {" ", "\t", "\r"}:
            flush_tok()
            idx += 1
            continue
        if ch in {"&", "|"} and idx + 1 < length and raw_text[idx + 1] == ch:
            flush_cmd()
            idx += 2
            continue
        tok_started = True
        buf.append(ch)
        idx += 1

    flush_cmd()
    return commands


def _expand_command_tokens(
    raw_cmd: list[list[tuple[str, str]]],
    shell_vars: Mapping[str, str],
    unresolved_vars: list[str],
) -> list[str]:
    tokens: list[str] = []
    for raw_tok in raw_cmd:
        pieces: list[str] = []
        all_unquoted = True
        expanded_any = False
        for seg, quote_kind in raw_tok:
            if quote_kind != "none":
                all_unquoted = False
            if quote_kind == "single" or "$" not in seg:
                pieces.append(seg)
                continue

            def _replace(match: re.Match[str]) -> str:
                nonlocal expanded_any
                name = match.group(1) or match.group(2) or ""
                if name in shell_vars:
                    expanded_any = True
                    return shell_vars[name]
                if name and name not in unresolved_vars:
                    unresolved_vars.append(name)
                return match.group(0)

            pieces.append(_SHELL_VAR_RE.sub(_replace, seg))
        joined = "".join(pieces)
        if all_unquoted and expanded_any:
            tokens.extend(joined.split())
        else:
            tokens.append(joined)
    return tokens


def _parse_dotted_scalar(raw: str) -> Any:
    text = str(raw).strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def _set_nested_config(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    cursor = target
    for key in parts[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _safe_json_dict(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    except ValueError:
        return {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _parse_human_int(raw: Any, default: int = 0) -> int:
    """Parse decimal ints and vLLM human-readable suffixes (``K``=1024, ``k``=1000, etc.)."""
    if isinstance(raw, bool):
        return default
    text = str(raw if raw is not None else "").strip().replace("_", "")
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        pass
    match = _HUMAN_INT_RE.fullmatch(text)
    if match is None:
        return default
    num_text, suffix = match.group(1), match.group(2)
    scale = _HUMAN_INT_SCALES[suffix]
    try:
        return int(Decimal(num_text) * scale)
    except (InvalidOperation, ValueError):
        return default


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _canonical_quant_from_qmap(qmap: Any) -> tuple[str, str]:
    raw_type = str(getattr(qmap, "model_quant_type", "") or "").strip()
    if not raw_type:
        return "", ""
    upper = raw_type.upper()
    rule_tags = {str(r.raw_tag or "").upper() for r in getattr(qmap, "rules", ()) or ()}
    if "MXFP4" in upper or any("MXFP4" in t for t in rule_tags):
        if upper == "W4A4_MXFP4" or ("W4A4_MXFP4" in rule_tags and not any("MXFP8" in t or "FP8" in t for t in rule_tags)):
            return "w4a4_mxfp4", raw_type
        return "mxfp8_mxfp4", raw_type
    if "W4A8_MXFP" in upper or any("W4A8_MXFP" in t for t in rule_tags):
        return "mxfp8_mxfp4", raw_type
    if "MXFP8" in upper or any("MXFP8" in t for t in rule_tags):
        return "mxfp8", raw_type
    if "FP8" in upper or any("FP8" in t for t in rule_tags):
        return "fp8", raw_type
    if "W4A8" in upper or any("W4A8" in t for t in rule_tags):
        return "w4a8", raw_type
    if upper == "W8A8_DYNAMIC":
        return "w8a8_dynamic", raw_type
    if "W8A8" in upper or any("W8A8" in t for t in rule_tags):
        return "w8a8", raw_type
    if upper in {"FLOAT", "BF16", "FP16"}:
        return "bf16", raw_type
    return raw_type.lower(), raw_type


def _infer_quant_hint(model_arg: str, quantization_flag: str) -> tuple[str, str]:
    if model_arg:
        curated = lookup_curated_checkpoint(model_arg)
        if curated is not None and curated.model_quant_type:
            return _canonical_quant_from_qmap(curated)
        local_path = Path(model_arg)
        if local_path.exists():
            try:
                local_qmap = read_checkpoint_quant_map(local_path)
            except TimeoutError:
                raise
            except (ModelingError, OSError):
                local_qmap = None
            if local_qmap is not None and local_qmap.model_quant_type:
                return _canonical_quant_from_qmap(local_qmap)
    base_low = Path(model_arg.rstrip("/")).name.lower() if model_arg else ""
    for tok, preset in _QUANT_BASENAME_TOKENS:
        if tok in base_low:
            return preset, ""
    q_low = str(quantization_flag or "").strip().lower()
    if q_low and q_low not in {"none", "auto"}:
        return q_low, ""
    return "", ""


def _parse_vllm_tokens(tokens: list[str], state: dict[str, Any], origins: dict[str, str]) -> None:
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "serve" and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            state["model_arg"] = tokens[idx + 1]
            origins["model_ref"] = "launch_script"
            idx += 2
            continue
        if "=" in tok and tok.startswith("-"):
            flag, nxt, step = (*tok.split("=", 1), 1)
        else:
            flag, nxt, step = tok, (tokens[idx + 1] if idx + 1 < len(tokens) else ""), 2

        dotted_handled = False
        for prefix, cfg_key in _DOTTED_CONFIG_PREFIXES:
            if flag.startswith(prefix):
                sub_key = flag[len(prefix):]
                _set_nested_config(state[cfg_key], sub_key, _parse_dotted_scalar(nxt))
                origins[cfg_key] = "launch_script"
                dotted_handled = True
                break
        if dotted_handled:
            idx += step
            continue

        if flag in {"--model", "-m"}:
            state["model_arg"] = nxt
            origins["model_ref"] = "launch_script"
        elif flag in {"-tp", "--tensor-parallel-size"}:
            state["tp"] = max(1, _safe_int(nxt, 1))
            state["has_parallel_flags"] = True
            origins["parallel.tp"] = "launch_script"
        elif flag in {"-dp", "--data-parallel-size"}:
            state["dp"] = max(1, _safe_int(nxt, 1))
            state["has_parallel_flags"] = True
            origins["parallel.dp"] = "launch_script"
        elif flag in {"-pp", "--pipeline-parallel-size"}:
            state["pp"] = max(1, _safe_int(nxt, 1))
            state["has_parallel_flags"] = True
            origins["parallel.pp"] = "launch_script"
        elif flag in {"-dcp", "--dcp-size", "--cp-size", "--decode-context-parallel-size"}:
            state["dcp"] = max(1, _safe_int(nxt, 1))
            state["has_parallel_flags"] = True
            origins["parallel.cp"] = "launch_script"
        elif flag in {"-pcp", "--pcp-size", "--prefill-context-parallel-size"}:
            state["pcp"] = max(1, _safe_int(nxt, 1))
            state["has_parallel_flags"] = True
            origins["parallel.pcp"] = "launch_script"
        elif flag == "--nnodes":
            state["nnodes"] = max(1, _safe_int(nxt, 1))
            origins["runtime_limits.nnodes"] = "launch_script"
        elif flag == "--node-rank":
            state["node_rank"] = max(0, _safe_int(nxt, 0))
        elif flag in {"-ep", "--enable-expert-parallel"}:
            state["ep_enabled"] = (nxt.strip().lower() not in _FALSE_ENV_TOKENS) if "=" in tok else True
            step = 1
            state["has_parallel_flags"] = True
            origins["parallel.ep"] = "launch_script"
        elif flag == "--no-enable-expert-parallel":
            state["ep_enabled"], step = False, 1
            state["has_parallel_flags"] = True
            origins["parallel.ep"] = "launch_script"
        elif flag == "--enforce-eager":
            state["enforce_eager"], step = True, 1
            origins["enforce_eager"] = "launch_script"
        elif flag == "--use-replayssm":
            state["use_replayssm"], step = True, 1
            origins["use_replayssm"] = "launch_script"
        elif flag == "--enable-ep-weight-filter":
            state["enable_ep_weight_filter"], step = True, 1
        elif flag == "--no-enable-prefix-caching":
            state["enable_prefix_caching"], step = False, 1
        elif flag == "--max-model-len":
            state["max_model_len"] = _parse_human_int(nxt, 0)
            origins["runtime_limits.max_model_len"] = "launch_script"
        elif flag == "--max-num-seqs":
            state["max_num_seqs"] = _safe_int(nxt, 0)
            origins["runtime_limits.max_num_seqs"] = "launch_script"
        elif flag == "--max-num-batched-tokens":
            state["max_num_batched_tokens"] = _parse_human_int(nxt, 0)
            origins["runtime_limits.max_num_batched_tokens"] = "launch_script"
        elif flag == "--block-size":
            state["block_size"] = _safe_int(nxt, 0)
            origins["runtime_limits.block_size"] = "launch_script"
        elif flag == "--gpu-memory-utilization":
            state["gpu_memory_utilization"] = _safe_float(nxt, 0.0)
            origins["runtime_limits.gpu_memory_utilization"] = "launch_script"
        elif flag in {"-q", "--quantization"}:
            state["quantization_flag"] = nxt
            origins["quantization"] = "launch_script"
        elif flag == "--kv-cache-dtype":
            state["kv_cache_dtype"] = nxt
            origins["kv_cache_dtype"] = "launch_script"
        elif flag == "--safetensors-load-strategy":
            state["safetensors_load_strategy"] = nxt
        elif flag == "--mm-processor-cache-gb":
            state["mm_processor_cache_gb"] = _safe_int(nxt, 0)
        elif flag == "--deployment-mode":
            state["explicit_deployment_mode"] = str(nxt).strip().lower()
            origins["deployment_mode"] = "launch_script"
        elif flag == "--compilation-config":
            state["compilation_config"].update(_safe_json_dict(nxt))
            origins["compilation_config"] = "launch_script"
        elif flag == "--speculative-config":
            state["speculative_config"].update(_safe_json_dict(nxt))
            origins["speculative_config"] = "launch_script"
        elif flag == "--additional-config":
            state["additional_config"].update(_safe_json_dict(nxt))
            origins["additional_config"] = "launch_script"
        elif flag == "--kv-transfer-config":
            state["kv_transfer_config"].update(_safe_json_dict(nxt))
            origins["kv_transfer_config"] = "launch_script"
        else:
            step = 1
        idx += step


def _detect_explicit_deployment_mode(state: Mapping[str, Any]) -> str | None:
    """Return deployment mode ONLY when explicitly declared (zero silent guessing)."""
    explicit = str(state.get("explicit_deployment_mode") or "").strip().lower()
    if explicit in DEPLOYMENT_MODES:
        return explicit
    kv_xfer = state.get("kv_transfer_config")
    if isinstance(kv_xfer, Mapping) and kv_xfer:
        connector = str(kv_xfer.get("kv_connector") or "").strip().lower()
        if connector in _NON_PD_KV_CONNECTORS:
            return None
        kv_role = str(kv_xfer.get("kv_role") or "").strip().lower()
        if kv_role in _KV_ROLE_TO_DEPLOYMENT_MODE:
            return _KV_ROLE_TO_DEPLOYMENT_MODE[kv_role]
    return None


def _is_integer_quant(weight_fmt: str) -> bool:
    return any(tok in str(weight_fmt or "").upper() for tok in _INTEGER_WEIGHT_TOKENS)


def _is_mxfp_or_fp8(weight_fmt: str) -> bool:
    return any(tok in str(weight_fmt or "").upper() for tok in _A5_MXFP_TOKENS)


def _target_hccl_mode(hw_fam: str, tuned_facts: Mapping[str, Any]) -> str:
    declared = str(tuned_facts.get("hccl_op_expansion_mode") or "").strip().upper()
    if hw_fam == "a5":
        return declared if declared in _A5_VALID_HCCL_MODES else _A5_DEFAULT_HCCL_MODE
    if hw_fam in {"a3", "a2"}:
        return _A23_DEFAULT_HCCL_MODE
    return declared


def _make_issue(
    issue_id: str, severity: str, title: str, current: str, recommended: str, impact: str,
) -> dict[str, str]:
    return {
        "issue_id": issue_id,
        "severity": severity,
        "title": title,
        "current": current,
        "recommended": recommended,
        "impact": impact,
    }


def _unfreeze_nested(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {k: _unfreeze_nested(v) for k, v in value}
        return [_unfreeze_nested(item) for item in value]
    return value


class VllmAscendLaunchCodec:
    """Static ``LaunchConfigCodec`` implementation for ``vllm-ascend``."""

    def parse_launch_config(self, source: Path | str) -> LaunchConfigIngress:
        """Statically decode a ``vllm serve`` shell script or CLI string."""
        if isinstance(source, Path) or (
            isinstance(source, str) and "\n" not in source and Path(source).is_file()
        ):
            src_path = Path(source)
            raw_text = src_path.read_text(encoding="utf-8")[:MAX_STATIC_SCRIPT_BYTES]
            src_label = src_path.name
        else:
            raw_text = str(source or "")[:MAX_STATIC_SCRIPT_BYTES]
            src_label = "inline_command"

        env_vars: dict[str, str] = {}
        shell_vars: dict[str, str] = {}
        unresolved_vars: list[str] = []
        origins: dict[str, str] = {}
        state: dict[str, Any] = {
            "model_arg": "",
            "quantization_flag": "",
            "kv_cache_dtype": "",
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "ep_enabled": False,
            "dcp": 1,
            "pcp": 1,
            "nnodes": 1,
            "node_rank": 0,
            "has_parallel_flags": False,
            "max_model_len": 0,
            "max_num_seqs": 0,
            "max_num_batched_tokens": 0,
            "block_size": 0,
            "gpu_memory_utilization": 0.0,
            "enforce_eager": False,
            "use_replayssm": False,
            "enable_ep_weight_filter": False,
            "enable_prefix_caching": True,
            "safetensors_load_strategy": "",
            "mm_processor_cache_gb": None,
            "explicit_deployment_mode": "",
            "compilation_config": {},
            "speculative_config": {},
            "additional_config": {},
            "kv_transfer_config": {},
        }
        for raw_cmd in _split_shell_commands(raw_text):
            tokens = _expand_command_tokens(raw_cmd, shell_vars, unresolved_vars)
            if not tokens:
                continue
            pos = 1 if tokens[0] == "export" else 0
            while pos < len(tokens):
                match = _ENV_ASSIGN_RE.match(tokens[pos])
                if match is None:
                    break
                key, val = match.group(1), match.group(2)
                shell_vars[key] = val
                if key in _KNOWN_ENV_KEYS or key.startswith(("HCCL_", "VLLM_", "ASCEND_", "OMP_", "PYTORCH_NPU")):
                    env_vars[key] = val
                    origins[f"env.{key}"] = "launch_script"
                pos += 1
            if pos < len(tokens):
                _parse_vllm_tokens(tokens[pos:], state, origins)

        tp, dp, pp, dcp, pcp = (
            int(state["tp"]),
            int(state["dp"]),
            int(state["pp"]),
            int(state["dcp"]),
            int(state["pcp"]),
        )
        ep_size = (dp * tp * pcp) if state["ep_enabled"] else 1
        parallel = ParallelSpec(dp=dp, tp=tp, pp=pp, ep=ep_size, cp=dcp, pcp=pcp)
        model_arg = str(state["model_arg"])
        quant_hint, ckpt_quant_type = _infer_quant_hint(model_arg, str(state["quantization_flag"]))
        deploy_hint = _detect_explicit_deployment_mode(state)
        if deploy_hint is not None:
            origins["deployment_mode_hint"] = "launch_script"
        if unresolved_vars:
            origins["unresolved_shell_vars"] = "launch_script"

        vis = env_vars.get("ASCEND_RT_VISIBLE_DEVICES", "")
        local_vis = len([x for x in vis.split(",") if x.strip()]) if vis else 0
        nnodes = max(1, int(state["nnodes"]))
        world_size = parallel.world_size
        inferred_devs = (
            world_size
            if (state["has_parallel_flags"] or nnodes > 1 or local_vis <= 0)
            else local_vis
        )
        runtime_limits: list[tuple[str, Any]] = [
            ("inferred_num_devices", inferred_devs),
        ]
        if local_vis > 0:
            runtime_limits.append(("local_visible_devices", local_vis))
        if nnodes > 1:
            runtime_limits.append(("nnodes", nnodes))
        for limit_key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "block_size"):
            if int(state[limit_key]) > 0:
                runtime_limits.append((limit_key, int(state[limit_key])))
        if float(state["gpu_memory_utilization"]) > 0.0:
            runtime_limits.append(("gpu_memory_utilization", float(state["gpu_memory_utilization"])))

        raw_flags: list[tuple[str, Any]] = [
            ("inferred_catalog_model", normalize_canonical_model(model_arg)),
            ("inferred_quant", quant_hint),
            ("quantization", str(state["quantization_flag"])),
            ("kv_cache_dtype", str(state["kv_cache_dtype"])),
            ("enforce_eager", bool(state["enforce_eager"])),
            ("use_replayssm", bool(state["use_replayssm"])),
            ("enable_ep_weight_filter", bool(state["enable_ep_weight_filter"])),
            ("enable_prefix_caching", bool(state["enable_prefix_caching"])),
            ("safetensors_load_strategy", str(state["safetensors_load_strategy"])),
            ("env_vars", dict(env_vars)),
            ("compilation_config", dict(state["compilation_config"])),
            ("speculative_config", dict(state["speculative_config"])),
            ("additional_config", dict(state["additional_config"])),
            ("kv_transfer_config", dict(state["kv_transfer_config"])),
        ]
        if state["mm_processor_cache_gb"] is not None:
            raw_flags.append(("mm_processor_cache_gb", int(state["mm_processor_cache_gb"])))
        if ckpt_quant_type:
            raw_flags.append(("checkpoint_quant_type", ckpt_quant_type))
        if unresolved_vars:
            raw_flags.append(("unresolved_shell_vars", tuple(unresolved_vars)))

        return LaunchConfigIngress(
            stack_id=VLLM_ASCEND_STACK_ID,
            source_label=src_label,
            model_ref=model_arg,
            checkpoint_hint=ckpt_quant_type or quant_hint,
            parallel=parallel,
            deployment_mode_hint=deploy_hint,
            runtime_limits=tuple(runtime_limits),
            raw_stack_flags=tuple(raw_flags),
            field_origins=tuple(sorted(origins.items())),
        )

    def _audit_isa_and_spec_rules(
        self,
        *,
        issues: list[dict[str, str]],
        ingress: LaunchConfigIngress,
        flags: Mapping[str, Any],
        comp_cfg: Mapping[str, Any],
        spec_cfg: Mapping[str, Any],
        hw_fam: str,
        model_name: str,
        w_fmt: str,
        tuned_kv: str,
        tuned_facts: Mapping[str, Any],
    ) -> str:
        if hw_fam in {"a2", "a3", "a5"}:
            isa_warns = validate_vllm_ascend_quant_isa(
                str(flags.get("inferred_quant") or flags.get("quantization") or ""),
                hw_fam,
                kv_cache_dtype=str(flags.get("kv_cache_dtype") or "bf16"),
            )
            if isa_warns:
                issues.append(_make_issue(
                    "hw_isa_quant_incompatible", "CRITICAL",
                    f"Hardware ISA does not support selected quant/KV ({hw_fam.upper()})",
                    f"quant={flags.get('inferred_quant')}, kv={flags.get('kv_cache_dtype') or 'auto'}",
                    w_fmt or tuned_kv, "; ".join(isa_warns),
                ))

        orig_spec_k = _safe_int(spec_cfg.get("num_speculative_tokens"), 0)
        raw_cap = comp_cfg.get("cudagraph_capture_sizes")
        cg_mode = str(comp_cfg.get("cudagraph_mode") or "FULL_DECODE_ONLY").upper()
        if isinstance(raw_cap, list) and orig_spec_k > 0:
            cap_warns = validate_aclgraph_capture_sizes(
                [int(x) for x in raw_cap if isinstance(x, int)],
                num_speculative_tokens=orig_spec_k,
                cudagraph_mode=cg_mode,
            )
            if cap_warns:
                issues.append(_make_issue(
                    "aclgraph_capture_size_not_k_plus_1", "HIGH",
                    f"ACLGraph capture sizes are not multiples of (K+1)={orig_spec_k + 1}",
                    str(raw_cap), f"Use multiples of {orig_spec_k + 1}", cap_warns[0],
                ))

        if hw_fam in {"a2", "a3", "a5"} and spec_cfg and orig_spec_k > 0:
            orig_kv = str(flags.get("kv_cache_dtype") or "bf16").strip().lower()
            if hw_fam in {"a2", "a3"} and orig_kv in {"fp8", "fp8_e4m3", "mxfp8"}:
                orig_kv = "bf16"
            orig_limits = {k: _unfreeze_nested(v) for k, v in ingress.runtime_limits}
            orig_max_seqs = max(1, _safe_int(orig_limits.get("max_num_seqs"), DEFAULT_AUDIT_MAX_NUM_SEQS))
            va_rules = build_va_serving_rules(
                model_name, hw_family=hw_fam,
                tp=max(1, ingress.parallel.tp),
                dp=max(1, ingress.parallel.dp),
                ep=max(1, ingress.parallel.ep),
                pp=max(1, ingress.parallel.pp),
                dcp=max(1, ingress.parallel.cp),
                pcp=max(1, ingress.parallel.pcp),
                target_kv_dtype=orig_kv, enable_speculative=True,
                spec_method=str(spec_cfg.get("method") or "") or None,
                num_speculative_tokens=orig_spec_k,
                draft_model=str(spec_cfg.get("model") or "") or None,
                max_num_seqs=orig_max_seqs,
            )
            if va_rules.resolved_speculative is not None:
                req_dtp = va_rules.resolved_speculative.draft_tensor_parallel_size
                got_dtp = spec_cfg.get("draft_tensor_parallel_size")
                if req_dtp is not None and got_dtp is not None and int(got_dtp) != int(req_dtp):
                    issues.append(_make_issue(
                        "draft_tp_policy_mismatch", "HIGH",
                        f"draft_tensor_parallel_size conflicts with {va_rules.resolved_speculative.method} rule",
                        f"draft_tensor_parallel_size={got_dtp}", f"draft_tensor_parallel_size={req_dtp}",
                        f"vLLM-Ascend requires draft_tp={req_dtp}",
                    ))
        del tuned_facts
        return cg_mode

    def audit_launch_config(
        self,
        ingress: LaunchConfigIngress,
        before_facts: Mapping[str, Any],
        tuned_facts: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Audit decoded launch ingress against before/tuned runtime facts."""
        flags = {k: _unfreeze_nested(v) for k, v in ingress.raw_stack_flags}
        limits = {k: _unfreeze_nested(v) for k, v in ingress.runtime_limits}
        env_vars: dict[str, str] = dict(flags.get("env_vars") or {})
        comp_cfg: dict[str, Any] = dict(flags.get("compilation_config") or {})
        spec_cfg: dict[str, Any] = dict(flags.get("speculative_config") or {})
        add_cfg: dict[str, Any] = dict(flags.get("additional_config") or {})

        hw_fam = normalize_hw_family(str(tuned_facts.get("hw_family") or tuned_facts.get("hardware_sku") or "a5"))
        model_name = str(tuned_facts.get("model_name") or flags.get("inferred_catalog_model") or ingress.model_ref)
        phase = str(tuned_facts.get("phase") or "decode").strip().lower()
        w_fmt = str(tuned_facts.get("weight_format") or "")
        tuned_kv = str(tuned_facts.get("kv_cache_dtype") or "bf16")
        issues: list[dict[str, str]] = []

        cg_mode = self._audit_isa_and_spec_rules(
            issues=issues, ingress=ingress, flags=flags, comp_cfg=comp_cfg,
            spec_cfg=spec_cfg, hw_fam=hw_fam, model_name=model_name,
            w_fmt=w_fmt, tuned_kv=tuned_kv, tuned_facts=tuned_facts,
        )

        if (
            not flags.get("quantization")
            and str(flags.get("inferred_quant") or "bf16").lower() in {"", "bf16"}
            and w_fmt.upper() not in _UNQUANTIZED_FORMATS
        ):
            ckpt = default_checkpoint_repo(model_name, hw_fam)
            hint = f" ({ckpt})" if ckpt else ""
            issues.append(_make_issue(
                "missing_quantization_bf16", "HIGH",
                "Production quantization disabled (native BF16 weights)",
                "unset (--quantization)", f"--quantization ascend{hint}",
                f"Tuned config uses {w_fmt} weights for HBM and bandwidth reduction",
            ))

        ctx = max(_safe_int(limits.get("max_model_len"), 0), _safe_int(before_facts.get("context_tokens"), 0))
        if (
            tuned_kv.upper() not in _UNQUANTIZED_FORMATS
            and str(flags.get("kv_cache_dtype") or "").strip().upper() in _UNQUANTIZED_FORMATS
            and (ctx >= LONG_CONTEXT_KV_QUANT_THRESHOLD or tuned_kv.upper() in {"FP8", "FP8_E4M3", "C8", "INT8"})
        ):
            rec_kv = "--kv-cache-dtype fp8_e4m3" if hw_fam == "a5" else "--kv-cache-dtype int8"
            issues.append(_make_issue(
                "missing_kv_quant", "HIGH", "KV cache quantization disabled",
                str(flags.get("kv_cache_dtype") or "auto"), rec_kv,
                f"Tuned config uses {tuned_kv} KV cache",
            ))

        if phase == "decode" and (bool(flags.get("enforce_eager")) or cg_mode not in FULL_GRAPH_CUDAGRAPH_MODES):
            issues.append(_make_issue(
                "missing_aclgraph_full_decode", "HIGH",
                "ACLGraph full decode capture not enabled",
                "--enforce-eager" if flags.get("enforce_eager") else str(comp_cfg or "unset"),
                '--compilation-config \'{"cudagraph_mode": "FULL_DECODE_ONLY"}\'',
                "Enable FULL_DECODE_ONLY or FULL_AND_PIECEWISE to eliminate decode launch bubbles",
            ))

        if bool(tuned_facts.get("has_mla")) and add_cfg.get("enable_mlapo") is False:
            issues.append(_make_issue(
                "missing_mlapo", "MEDIUM", "MLAPO explicitly disabled on MLA model",
                "enable_mlapo=false",
                '--additional-config \'{"enable_mlapo": true}\'',
                "Restore default MLA Prolog fusion",
            ))

        self._audit_overlap_and_hw_isolation(
            issues=issues, env_vars=env_vars, add_cfg=add_cfg, hw_fam=hw_fam,
            phase=phase, w_fmt=w_fmt, tuned_facts=tuned_facts,
        )

        tuned_tp = max(1, _safe_int(tuned_facts.get("tp"), ingress.parallel.tp))
        if ingress.parallel.tp >= HIGH_TP_AUDIT_THRESHOLD and ingress.parallel.tp > tuned_tp:
            issues.append(_make_issue(
                "suboptimal_high_tp", "MEDIUM", "Excessive TP increases cross-die small collectives",
                f"DP={ingress.parallel.dp}, TP={ingress.parallel.tp}",
                f"DP={tuned_facts.get('dp', ingress.parallel.dp)}, TP={tuned_tp}",
                "Lower TP and widen DP/EP to reduce collective overhead",
            ))

        orig_fits = bool(before_facts.get("fits_hbm", True))
        if not orig_fits:
            issues.append(_make_issue(
                "hbm_oom_in_original_script", "CRITICAL", "Original launch script exceeds usable HBM",
                f"Peak={_safe_float(before_facts.get('peak_hbm_bytes')) / GIB_BYTES:.1f} GiB",
                f"Tuned Fit={bool(tuned_facts.get('fits_hbm', True))}",
                "Apply quantization and feasible parallel sharding to eliminate HBM OOM",
            ))

        b_step = _safe_float(before_facts.get("step_span_us")) / US_PER_MS
        t_step = _safe_float(tuned_facts.get("step_span_us")) / US_PER_MS
        saved = round(b_step - t_step, AUDIT_MS_DECIMALS) if (orig_fits and t_step > 0.0) else None
        return {
            "schema": SCRIPT_AUDIT_SCHEMA,
            "stack_id": VLLM_ASCEND_STACK_ID,
            "source_script": ingress.source_label,
            "deployment_mode_hint": ingress.deployment_mode_hint,
            "issues_found": issues,
            "two_step_gain": {
                "step1_immediate_zero_code": {
                    "original_fits_hbm": orig_fits,
                    "before_step_ms": round(b_step, AUDIT_MS_DECIMALS),
                    "tuned_step_ms": round(t_step, AUDIT_MS_DECIMALS) if t_step > 0.0 else None,
                    "saved_ms": saved,
                    "issues_fixed_count": len(issues),
                },
            },
        }

    def _audit_overlap_and_hw_isolation(
        self,
        *,
        issues: list[dict[str, str]],
        env_vars: Mapping[str, str],
        add_cfg: Mapping[str, Any],
        hw_fam: str,
        phase: str,
        w_fmt: str,
        tuned_facts: Mapping[str, Any],
    ) -> None:
        user_mc2 = add_cfg.get("enable_fused_mc2")
        has_mc2 = isinstance(user_mc2, int) and not isinstance(user_mc2, bool) and user_mc2 > 0
        has_mso = bool(add_cfg.get("multistream_overlap_shared_expert"))
        if has_mc2 and has_mso:
            issues.append(_make_issue(
                "mc2_mso_mutex_conflict", "HIGH",
                "enable_fused_mc2 and multistream_overlap_shared_expert are mutually exclusive",
                f"enable_fused_mc2={user_mc2}, multistream_overlap_shared_expert=true",
                "Select either enable_fused_mc2 OR multistream_overlap_shared_expert",
                "Upstream vLLM-Ascend forbids combining fused MC2 and MSO shared-expert overlap",
            ))

        hccl_mode = str(env_vars.get("HCCL_OP_EXPANSION_MODE") or "").strip().upper()
        target_hccl = _target_hccl_mode(hw_fam, tuned_facts)
        if hw_fam == "a5":
            if hccl_mode not in _A5_VALID_HCCL_MODES:
                issues.append(_make_issue(
                    "hccl_expansion_mode_suboptimal", "HIGH",
                    "A5 collective expansion mode is not hardware CCU",
                    hccl_mode or "unset", f'HCCL_OP_EXPANSION_MODE="{target_hccl}"',
                    "Enable CCU scheduling on A5 UB-Direct collectives",
                ))
            legacy = [k for k in ("HCCL_INTRA_PCIE_ENABLE", "HCCL_INTRA_ROCE_ENABLE") if k in env_vars]
            if legacy:
                issues.append(_make_issue(
                    "a5_legacy_transport_override", "MEDIUM",
                    "A5 script includes legacy PCIe/RoCE env overrides",
                    ", ".join(legacy), "Remove legacy PCIe/RoCE env vars",
                    "A5 uses native UB-Direct interconnect",
                ))
            nz_val = add_cfg.get("weight_nz_mode")
            if _is_mxfp_or_fp8(w_fmt) and (
                add_cfg.get("enable_weight_nz_layout")
                or (nz_val is not None and nz_val != WEIGHT_NZ_MODE_DISABLED)
            ):
                issues.append(_make_issue(
                    "a5_mxfp_nz_layout_incompatible", "MEDIUM",
                    "A5 MXFP/FP8 weights should not force global NZ layout",
                    f"weight_nz_mode={nz_val}" if nz_val is not None else "enable_weight_nz_layout=true",
                    "Remove weight_nz_mode override on A5",
                    "MXFP layouts are operator-ABI driven on A5",
                ))
        elif hw_fam in {"a3", "a2"}:
            if (hw_fam == "a2" or phase == "decode") and hccl_mode not in _A23_VALID_HCCL_MODES:
                issues.append(_make_issue(
                    "hccl_expansion_mode_suboptimal", "HIGH",
                    f"{hw_fam.upper()} collective expansion mode is not AIV",
                    hccl_mode or "unset", f'HCCL_OP_EXPANSION_MODE="{target_hccl}"',
                    "A2/A3 HCCS requires AIV expansion (never CCU)",
                ))
            if _is_integer_quant(w_fmt) and (
                add_cfg.get("weight_nz_mode") == WEIGHT_NZ_MODE_DISABLED
                or add_cfg.get("enable_weight_nz_layout") is False
            ):
                issues.append(_make_issue(
                    "missing_nz_layout_a23", "MEDIUM",
                    f"{hw_fam.upper()} integer quantization disables FRACTAL_NZ layout",
                    "weight_nz_mode=0",
                    f'--additional-config \'{{"weight_nz_mode": {WEIGHT_NZ_MODE_QUANT}}}\'',
                    "Enable FRACTAL_NZ weight layout for A2/A3 INT8/INT4 GEMMs",
                ))

    def _resolve_pd_transfer_config(
        self, role: str, phase: str, tuned_facts: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if role == "pd_prefill":
            kv_role = "kv_producer"
        elif role == "pd_decode":
            kv_role = "kv_consumer"
        elif role == "pd_disaggregated":
            kv_role = "kv_producer" if phase == "prefill" else "kv_consumer"
        else:
            return None
        raw_xfer = tuned_facts.get("kv_transfer_config")
        kv_xfer: dict[str, Any] = dict(raw_xfer) if isinstance(raw_xfer, Mapping) else {}
        connector = str(kv_xfer.get("kv_connector") or tuned_facts.get("kv_connector") or "").strip()
        if not connector or connector == "AscendHcclConnector":
            connector = DEFAULT_PD_KV_CONNECTOR
        kv_xfer["kv_connector"] = connector
        kv_xfer["kv_role"] = kv_role
        return kv_xfer

    def render_launch_config(self, tuned_facts: Mapping[str, Any]) -> str:
        """Render a clean, hardware-isolated ``vllm serve`` bash script."""
        model_name = str(tuned_facts.get("model_name") or tuned_facts.get("model_ref") or "model")
        model_target = str(tuned_facts.get("model_ref") or f"/weights/{model_name}")
        hw_sku = str(tuned_facts.get("hardware_sku") or tuned_facts.get("hw_family") or "a5")
        hw_fam = normalize_hw_family(hw_sku)
        role = str(tuned_facts.get("role") or tuned_facts.get("deployment_mode") or "colocated").strip().lower()
        phase = str(tuned_facts.get("phase") or ("prefill" if role == "pd_prefill" else "decode")).strip().lower()

        tp = max(1, _safe_int(tuned_facts.get("tp"), 1))
        dp = max(1, _safe_int(tuned_facts.get("dp"), 1))
        pp = max(1, _safe_int(tuned_facts.get("pp"), 1))
        ep = max(1, _safe_int(tuned_facts.get("ep"), 1))
        dcp = max(1, _safe_int(tuned_facts.get("dcp") or tuned_facts.get("cp"), 1))
        pcp = max(1, _safe_int(tuned_facts.get("pcp"), 1))
        full_ep = dp * tp * pcp
        raw_etp = tuned_facts.get("expert_tp")
        if raw_etp is None:
            raw_etp = tuned_facts.get("etp")
        if raw_etp is not None:
            eff_etp = -1 if isinstance(raw_etp, bool) else _safe_int(raw_etp, -1)
        else:
            eff_etp = ParallelSpec(dp=dp, tp=tp, pp=pp, ep=ep, cp=dcp, pcp=pcp).etp
        expected_etp = tp if ep == 1 else 1
        if (ep != 1 and ep != full_ep) or eff_etp != expected_etp:
            raise ModelingError(
                f"vllm_ascend_partial_ep_unrepresentable_in_cli:"
                f"ep={ep},etp={eff_etp},dp={dp},tp={tp},pcp={pcp},"
                f"supported_ep=(1,{full_ep}),expected_etp={expected_etp}"
            )
        max_len = max(1, _parse_human_int(tuned_facts.get("max_model_len") or tuned_facts.get("seq_len"), DEFAULT_AUDIT_MAX_MODEL_LEN))
        max_seqs = max(1, _safe_int(tuned_facts.get("max_num_seqs") or tuned_facts.get("concurrency"), DEFAULT_AUDIT_MAX_NUM_SEQS))
        max_batched = _parse_human_int(tuned_facts.get("max_num_batched_tokens"), 0)
        block_size = _safe_int(tuned_facts.get("block_size") or tuned_facts.get("kv_block_size"), 0)
        enforce_eager = bool(tuned_facts.get("enforce_eager", False))
        use_replayssm = bool(tuned_facts.get("use_replayssm", False))
        gpu_util = _safe_float(tuned_facts.get("gpu_memory_utilization"), DEFAULT_GPU_MEMORY_UTILIZATION)
        if gpu_util <= 0.0:
            gpu_util = DEFAULT_GPU_MEMORY_UTILIZATION

        w_fmt = str(tuned_facts.get("weight_format") or "")
        kv_raw = str(tuned_facts.get("kv_cache_dtype") or "auto").strip().lower()
        spec_k = _safe_int(tuned_facts.get("spec_proposals"), 0)
        cg_mode = "NONE" if enforce_eager else str(tuned_facts.get("cudagraph_mode") or "FULL_DECODE_ONLY").upper()
        is_pd = role in {"pd_prefill", "pd_decode", "pd_disaggregated"}
        va_pd_role = ("prefill" if phase == "prefill" else "decode") if role == "pd_disaggregated" else role
        explicit_nz_mode = tuned_facts.get("weight_nz_mode")
        va_rules = (
            build_va_serving_rules(
                model_name, hw_family=hw_fam, tp=tp, dp=dp, ep=ep, pp=pp, dcp=dcp, pcp=pcp,
                target_kv_dtype=kv_raw, enable_speculative=(spec_k > 0),
                spec_method=str(tuned_facts.get("spec_method") or "") or None,
                num_speculative_tokens=spec_k if spec_k > 0 else None,
                draft_model=str(tuned_facts.get("draft_model") or "") or None,
                cudagraph_mode=cg_mode, max_num_seqs=max_seqs,
                pd_disaggregated=is_pd, pd_role=va_pd_role,
                weight_nz_mode=explicit_nz_mode,
                enable_replayssm=use_replayssm,
            ) if hw_fam in {"a2", "a3", "a5"} else None
        )

        eff_kv = va_rules.effective_kv_cache_dtype if va_rules is not None else kv_raw
        if hw_fam == "a5":
            if eff_kv == "mxfp8":
                kv_flag = "mxfp8"
            elif eff_kv in {"fp8", "fp8_e4m3"}:
                kv_flag = "fp8_e4m3"
            else:
                kv_flag = "auto"
        elif hw_fam in {"a2", "a3"}:
            kv_flag = "int8" if eff_kv in {"c8", "int8"} else "auto"
        else:
            kv_flag = eff_kv or "auto"

        add_cfg: dict[str, Any] = dict(va_rules.additional_config) if va_rules is not None else {}
        add_cfg.pop("enable_weight_nz_layout", None)
        mc2_level = _safe_int(tuned_facts.get("enable_fused_mc2"), 0)
        mso_on = bool(tuned_facts.get("multistream_overlap_shared_expert", False)) and mc2_level <= 0
        if mc2_level > 0:
            add_cfg["enable_fused_mc2"] = mc2_level
        elif mso_on:
            add_cfg["multistream_overlap_shared_expert"] = True
        if explicit_nz_mode is None:
            if hw_fam in {"a2", "a3"} and _is_integer_quant(w_fmt):
                add_cfg["weight_nz_mode"] = WEIGHT_NZ_MODE_QUANT
            elif hw_fam == "a5":
                add_cfg.pop("weight_nz_mode", None)
        if bool(tuned_facts.get("has_mla")):
            add_cfg["enable_mlapo"] = (role != "pd_prefill")

        comp_cfg = {"cudagraph_mode": "NONE"} if enforce_eager else (
            dict(va_rules.compilation_config) if va_rules is not None else {"cudagraph_mode": cg_mode}
        )
        coll = shipping_collective_policy(hw_fam, canonical=model_name, role=role)
        hccl_mode = str(tuned_facts.get("hccl_op_expansion_mode") or (coll or {}).get("process_env_mode") or "")
        if hw_fam in {"a2", "a3"} and "CCU" in hccl_mode.upper():
            hccl_mode = _A23_DEFAULT_HCCL_MODE

        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated by mindie-sim vllm-ascend launch codec",
            f"# Model: {model_name} | Hardware: {hw_sku} | Parallel: DP={dp}, TP={tp}, EP={ep}, PP={pp}, DCP={dcp}, PCP={pcp}",
            "set -euo pipefail",
            "",
        ]
        if hccl_mode and (hw_fam in {"a5", "a2"} or (hw_fam == "a3" and phase == "decode")):
            lines.append(f'export HCCL_OP_EXPANSION_MODE="{hccl_mode}"')
        lines.extend([
            f"export HCCL_BUFFSIZE={DEFAULT_HCCL_BUFFSIZE_MIB}",
            f"export HCCL_BUFFSIZE_EP={DEFAULT_HCCL_EP_BUFFSIZE_MIB}",
            'export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"',
            "",
            f'vllm serve "{model_target}" \\',
            f"  --tensor-parallel-size {tp} \\",
            f"  --data-parallel-size {dp} \\",
        ])
        if pp > 1:
            lines.append(f"  --pipeline-parallel-size {pp} \\")
        if ep > 1:
            lines.append("  --enable-expert-parallel \\")
        if dcp > 1:
            lines.append(f"  --decode-context-parallel-size {dcp} \\")
        if pcp > 1:
            lines.append(f"  --prefill-context-parallel-size {pcp} \\")
        if role == "pd_disaggregated":
            lines.append("  --deployment-mode pd_disaggregated \\")
        kv_xfer = self._resolve_pd_transfer_config(role, phase, tuned_facts)
        if kv_xfer is not None:
            lines.append(f"  --kv-transfer-config '{json.dumps(kv_xfer, separators=(',', ':'))}' \\")
        if va_rules is not None and va_rules.speculative_enabled and va_rules.speculative_config:
            lines.append(f"  --speculative-config '{json.dumps(va_rules.speculative_config, separators=(',', ':'))}' \\")
        no_prefix = (
            tuned_facts.get("enable_prefix_caching") is False
            or (va_rules is not None and "--no-enable-prefix-caching" in va_rules.cli_flags)
        )
        if no_prefix:
            lines.append("  --no-enable-prefix-caching \\")
        if use_replayssm:
            lines.append("  --use-replayssm \\")
        if enforce_eager:
            lines.append("  --enforce-eager \\")
        if w_fmt.upper() not in _UNQUANTIZED_FORMATS:
            lines.append("  --quantization ascend \\")
        lines.extend([
            f"  --kv-cache-dtype {kv_flag} \\",
            f"  --max-model-len {max_len} \\",
            f"  --max-num-seqs {max_seqs} \\",
        ])
        if max_batched > 0:
            lines.append(f"  --max-num-batched-tokens {max_batched} \\")
        if block_size > 0:
            lines.append(f"  --block-size {block_size} \\")
        lines.extend([
            f"  --gpu-memory-utilization {gpu_util:g} \\",
            f"  --compilation-config '{json.dumps(comp_cfg, separators=(',', ':'))}' \\",
            f"  --additional-config '{json.dumps(add_cfg, separators=(',', ':'))}'",
            "",
        ])
        return "\n".join(lines)


__all__ = [
    "SCRIPT_AUDIT_SCHEMA",
    "VllmAscendLaunchCodec",
]
