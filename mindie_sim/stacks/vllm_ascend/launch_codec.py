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
import json
from pathlib import Path
import re
import shlex
from typing import Any

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
GIB_BYTES = 1024 * 1024 * 1024
US_PER_MS = 1000.0
AUDIT_MS_DECIMALS = 3
AUDIT_RATE_DECIMALS = 2

_ENV_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
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

_KV_ROLE_TO_DEPLOYMENT_MODE: dict[str, str] = {
    "kv_producer": "pd_prefill",
    "kv_consumer": "pd_decode",
    "kv_both": "pd_disaggregated",
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
    except ValueError:
        return {}


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _infer_quant_hint(model_arg: str, quantization_flag: str) -> str:
    base_low = Path(model_arg.rstrip("/")).name.lower() if model_arg else ""
    for tok, preset in _QUANT_BASENAME_TOKENS:
        if tok in base_low:
            return preset
    q_low = str(quantization_flag or "").strip().lower()
    if q_low in {"", "bf16", "none"}:
        return "bf16"
    return q_low


def _parse_vllm_tokens(tokens: list[str], state: dict[str, Any], origins: dict[str, str]) -> None:
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "serve" and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            state["model_arg"] = tokens[idx + 1]
            origins["model_ref"] = "launch_script"
            idx += 2
            continue
        if "=" in tok and tok.startswith("--"):
            flag, nxt, step = (*tok.split("=", 1), 1)
        else:
            flag, nxt, step = tok, (tokens[idx + 1] if idx + 1 < len(tokens) else ""), 2

        if flag in {"--model", "-m"}:
            state["model_arg"] = nxt
            origins["model_ref"] = "launch_script"
        elif flag in {"-tp", "--tensor-parallel-size"}:
            state["tp"] = max(1, _safe_int(nxt, 1))
            origins["parallel.tp"] = "launch_script"
        elif flag in {"-dp", "--data-parallel-size"}:
            state["dp"] = max(1, _safe_int(nxt, 1))
            origins["parallel.dp"] = "launch_script"
        elif flag in {"-pp", "--pipeline-parallel-size"}:
            state["pp"] = max(1, _safe_int(nxt, 1))
            origins["parallel.pp"] = "launch_script"
        elif flag in {"--dcp-size", "--cp-size", "--decode-context-parallel-size"}:
            state["dcp"] = max(1, _safe_int(nxt, 1))
            origins["parallel.cp"] = "launch_script"
        elif flag in {"--pcp-size", "--prefill-context-parallel-size"}:
            state["pcp"] = max(1, _safe_int(nxt, 1))
            origins["parallel.pcp"] = "launch_script"
        elif flag == "--enable-expert-parallel":
            state["ep_enabled"], step = True, 1
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
            state["max_model_len"] = _safe_int(nxt, 0)
            origins["runtime_limits.max_model_len"] = "launch_script"
        elif flag == "--max-num-seqs":
            state["max_num_seqs"] = _safe_int(nxt, 0)
            origins["runtime_limits.max_num_seqs"] = "launch_script"
        elif flag == "--max-num-batched-tokens":
            state["max_num_batched_tokens"] = _safe_int(nxt, 0)
            origins["runtime_limits.max_num_batched_tokens"] = "launch_script"
        elif flag == "--block-size":
            state["block_size"] = _safe_int(nxt, 0)
            origins["runtime_limits.block_size"] = "launch_script"
        elif flag == "--gpu-memory-utilization":
            state["gpu_memory_utilization"] = _safe_float(nxt, 0.0)
            origins["runtime_limits.gpu_memory_utilization"] = "launch_script"
        elif flag == "--quantization":
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
            state["compilation_config"] = _safe_json_dict(nxt)
            origins["compilation_config"] = "launch_script"
        elif flag == "--speculative-config":
            state["speculative_config"] = _safe_json_dict(nxt)
            origins["speculative_config"] = "launch_script"
        elif flag == "--additional-config":
            state["additional_config"] = _safe_json_dict(nxt)
            origins["additional_config"] = "launch_script"
        elif flag == "--kv-transfer-config":
            state["kv_transfer_config"] = _safe_json_dict(nxt)
            origins["kv_transfer_config"] = "launch_script"
        else:
            step = 1
        idx += step


def _detect_explicit_deployment_mode(state: Mapping[str, Any]) -> str | None:
    """Return deployment mode ONLY when explicitly declared (zero silent guessing)."""
    kv_xfer = state.get("kv_transfer_config")
    if isinstance(kv_xfer, Mapping) and kv_xfer:
        kv_role = str(kv_xfer.get("kv_role") or "").strip().lower()
        if kv_role in _KV_ROLE_TO_DEPLOYMENT_MODE:
            return _KV_ROLE_TO_DEPLOYMENT_MODE[kv_role]
    explicit = str(state.get("explicit_deployment_mode") or "").strip().lower()
    if explicit in DEPLOYMENT_MODES:
        return explicit
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
                key, val = match.group(1), match.group(2)
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
        quant_hint = _infer_quant_hint(model_arg, str(state["quantization_flag"]))
        deploy_hint = _detect_explicit_deployment_mode(state)
        if deploy_hint is not None:
            origins["deployment_mode_hint"] = "launch_script"

        vis = env_vars.get("ASCEND_RT_VISIBLE_DEVICES", "")
        inferred_devs = len([x for x in vis.split(",") if x.strip()]) if vis else (dp * tp * pp * pcp)
        runtime_limits: list[tuple[str, Any]] = [
            ("inferred_num_devices", inferred_devs),
        ]
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

        return LaunchConfigIngress(
            stack_id=VLLM_ASCEND_STACK_ID,
            source_label=src_label,
            model_ref=model_arg,
            checkpoint_hint=quant_hint,
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

        spec_k = _safe_int(spec_cfg.get("num_speculative_tokens") or tuned_facts.get("spec_proposals"), 0)
        raw_cap = comp_cfg.get("cudagraph_capture_sizes")
        cg_mode = str(comp_cfg.get("cudagraph_mode") or "FULL_DECODE_ONLY").upper()
        if isinstance(raw_cap, list) and spec_k > 0:
            cap_warns = validate_aclgraph_capture_sizes(
                [int(x) for x in raw_cap if isinstance(x, int)],
                num_speculative_tokens=spec_k,
                cudagraph_mode=cg_mode,
            )
            if cap_warns:
                issues.append(_make_issue(
                    "aclgraph_capture_size_not_k_plus_1", "HIGH",
                    f"ACLGraph capture sizes are not multiples of (K+1)={spec_k + 1}",
                    str(raw_cap), f"Use multiples of {spec_k + 1}", cap_warns[0],
                ))

        if hw_fam in {"a2", "a3", "a5"} and spec_cfg and spec_k > 0:
            va_rules = build_va_serving_rules(
                model_name, hw_family=hw_fam,
                tp=max(1, _safe_int(tuned_facts.get("tp"), ingress.parallel.tp)),
                dp=max(1, _safe_int(tuned_facts.get("dp"), ingress.parallel.dp)),
                ep=max(1, _safe_int(tuned_facts.get("ep"), ingress.parallel.ep)),
                pp=max(1, _safe_int(tuned_facts.get("pp"), ingress.parallel.pp)),
                target_kv_dtype=tuned_kv.lower(), enable_speculative=True,
                spec_method=str(spec_cfg.get("method") or "") or None,
                num_speculative_tokens=spec_k,
                max_num_seqs=max(1, _safe_int(tuned_facts.get("concurrency"), DEFAULT_AUDIT_MAX_NUM_SEQS)),
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

        if bool(tuned_facts.get("has_mla")):
            raw_mlapo = env_vars.get("VLLM_ASCEND_ENABLE_MLAPO")
            if (raw_mlapo is not None and raw_mlapo.strip().lower() in _FALSE_ENV_TOKENS) or add_cfg.get("enable_mlapo") is False:
                issues.append(_make_issue(
                    "missing_mlapo", "MEDIUM", "MLAPO explicitly disabled on MLA model",
                    f"VLLM_ASCEND_ENABLE_MLAPO={raw_mlapo}" if raw_mlapo is not None else "enable_mlapo=false",
                    "export VLLM_ASCEND_ENABLE_MLAPO=1", "Restore default MLA Prolog fusion",
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
            if _is_mxfp_or_fp8(w_fmt) and add_cfg.get("enable_weight_nz_layout"):
                issues.append(_make_issue(
                    "a5_mxfp_nz_layout_incompatible", "MEDIUM",
                    "A5 MXFP/FP8 weights should not force global NZ layout",
                    "enable_weight_nz_layout=true", "Remove enable_weight_nz_layout",
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
            if _is_integer_quant(w_fmt) and not add_cfg.get("enable_weight_nz_layout"):
                issues.append(_make_issue(
                    "missing_nz_layout_a23", "MEDIUM",
                    f"{hw_fam.upper()} integer quantization missing FRACTAL_NZ layout",
                    "enable_weight_nz_layout unset", '--additional-config \'{"enable_weight_nz_layout": true}\'',
                    "Enable FRACTAL_NZ weight layout for A2/A3 INT8/INT4 GEMMs",
                ))

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
        max_len = max(1, _safe_int(tuned_facts.get("max_model_len") or tuned_facts.get("seq_len"), DEFAULT_AUDIT_MAX_MODEL_LEN))
        max_seqs = max(1, _safe_int(tuned_facts.get("max_num_seqs") or tuned_facts.get("concurrency"), DEFAULT_AUDIT_MAX_NUM_SEQS))
        gpu_util = _safe_float(tuned_facts.get("gpu_memory_utilization"), DEFAULT_GPU_MEMORY_UTILIZATION)
        if gpu_util <= 0.0:
            gpu_util = DEFAULT_GPU_MEMORY_UTILIZATION

        w_fmt = str(tuned_facts.get("weight_format") or "")
        kv_raw = str(tuned_facts.get("kv_cache_dtype") or "auto").strip().lower()
        if hw_fam == "a5":
            kv_flag = "fp8_e4m3" if kv_raw in {"fp8", "fp8_e4m3", "mxfp8"} else "auto"
        elif hw_fam in {"a2", "a3"}:
            kv_flag = "int8" if kv_raw in {"c8", "int8"} else "auto"
        else:
            kv_flag = kv_raw or "auto"

        spec_k = _safe_int(tuned_facts.get("spec_proposals"), 0)
        cg_mode = str(tuned_facts.get("cudagraph_mode") or "FULL_DECODE_ONLY").upper()
        va_rules = (
            build_va_serving_rules(
                model_name, hw_family=hw_fam, tp=tp, dp=dp, ep=ep, pp=pp,
                target_kv_dtype=kv_flag, enable_speculative=(spec_k > 0),
                spec_method=str(tuned_facts.get("spec_method") or "") or None,
                num_speculative_tokens=spec_k if spec_k > 0 else None,
                cudagraph_mode=cg_mode, max_num_seqs=max_seqs,
                pd_disaggregated=(role in {"pd_prefill", "pd_decode", "pd_disaggregated"}),
                pd_role=role,
            ) if hw_fam in {"a2", "a3", "a5"} else None
        )

        add_cfg: dict[str, Any] = dict(va_rules.additional_config) if va_rules is not None else {}
        mc2_level = _safe_int(tuned_facts.get("enable_fused_mc2"), 0)
        mso_on = bool(tuned_facts.get("multistream_overlap_shared_expert", False)) and mc2_level <= 0
        if mc2_level > 0:
            add_cfg["enable_fused_mc2"] = mc2_level
        elif mso_on:
            add_cfg["multistream_overlap_shared_expert"] = True
        if hw_fam in {"a2", "a3"} and _is_integer_quant(w_fmt):
            add_cfg["enable_weight_nz_layout"] = True
        elif hw_fam == "a5":
            add_cfg.pop("enable_weight_nz_layout", None)

        comp_cfg = dict(va_rules.compilation_config) if va_rules is not None else {"cudagraph_mode": cg_mode}
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
        ])
        if mc2_level > 0:
            lines.append("export VLLM_ASCEND_ENABLE_FUSED_MC2=1")
        if bool(tuned_facts.get("has_mla")) and role != "pd_prefill":
            lines.append("export VLLM_ASCEND_ENABLE_MLAPO=1")
        lines.extend([
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
        if role == "pd_prefill":
            kv_xfer = {"kv_connector": "AscendHcclConnector", "kv_role": "kv_producer"}
            lines.append(f"  --kv-transfer-config '{json.dumps(kv_xfer, separators=(',', ':'))}' \\")
        elif role == "pd_decode":
            kv_xfer = {"kv_connector": "AscendHcclConnector", "kv_role": "kv_consumer"}
            lines.append(f"  --kv-transfer-config '{json.dumps(kv_xfer, separators=(',', ':'))}' \\")
        if va_rules is not None and va_rules.speculative_enabled and va_rules.speculative_config:
            lines.append(f"  --speculative-config '{json.dumps(va_rules.speculative_config, separators=(',', ':'))}' \\")
        if w_fmt.upper() not in _UNQUANTIZED_FORMATS:
            lines.append("  --quantization ascend \\")
        lines.extend([
            f"  --kv-cache-dtype {kv_flag} \\",
            f"  --max-model-len {max_len} \\",
            f"  --max-num-seqs {max_seqs} \\",
            f"  --gpu-memory-utilization {gpu_util:.2f} \\",
            f"  --compilation-config '{json.dumps(comp_cfg, separators=(',', ':'))}' \\",
            f"  --additional-config '{json.dumps(add_cfg, separators=(',', ':'))}'",
            "",
        ])
        return "\n".join(lines)


__all__ = [
    "SCRIPT_AUDIT_SCHEMA",
    "VllmAscendLaunchCodec",
]
