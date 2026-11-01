"""Deterministic case ingestion from exported Cases, Wave 5 captures and C7 profiles.

[Role]
  Normalize heterogeneous profiling artifacts into one validated simulation
  Case while preserving measured observations only as calibration metadata.

[Invariants]
  * Observed latencies are metadata, never forward-prediction inputs.
  * Profile values are explicit; CLI overrides win over captured values.
  * Unknown model/capture directories fail explicitly instead of guessing a
    silent default.

[Timeouts & Locks]
  Bounded local JSON/text reads (< 50 MiB each); no network, NPU, or locks.

[Side-Effect Policy]
  Read-only; every returned case is a fresh dictionary.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json, re, shlex
from pathlib import Path
from typing import Any
from .advise_util import _PROFILE_MAX_BYTES, _candidate_mappings, _deep_values, _first_int, _first_number, _first_text, _has_measured_observation, _sanitize_case
from .compose import compose_case, resolve_hardware_sku
from .model_catalog import ARCHITECTURE_PRESET_ALIASES, ARCHITECTURE_PRESET_CATALOG, CATALOG, MODEL_ALIASES
from .prediction.structure import parse_case
from .profile_observed import _extract_observed
from .schema.case import Case as CaseSpec, Software
from .schema.common import ModelingError
_PROVENANCE_SCHEMA = "mindie-sim.profile-provenance/1"
_MODEL_PATTERNS = (
    (("dsv2lite", "deepseek-v2-lite", "deepseek_v2_lite"), "deepseek-v2-lite"),
    (("dsv41", "deepseek-v4.1", "v4.1-flash"), "deepseek-v4.1-flash"),
    (("dsv410t", "deepseek-v4-10t"), "deepseek-v4-10t"),
    (("dsv4f", "dsv4-flash", "deepseek-v4-flash"), "deepseek-v4-flash-0731"),
    (("dsv4pro", "deepseek-v4-pro"), "deepseek-v4-pro-0813"),
    (("dsv32", "dsv3.2", "v3.2-prune", "v32prune"), "deepseek-v3.2-prune-3l"),
    (("glm53-flash", "glm-5.3-flash", "glm_5_3_flash", "glm53_flash"), "glm-5.3-flash"),
    (("glm53", "glm-5.3", "glm_5_3"), "glm-5.3"),
    (("kimi-k3-dspark", "kimi_k3_dspark"), "kimi-k3-dspark"),
    (("kimi-k3", "kimi_k3", "kimi"), "kimi-k3"),
    (("minimax",), "minimax-m3"),
    (("longcat",), "longcat-flash-lite"),
    (("bailing",), "bailing-v3"),
    (("qwen3.8-2.4t", "qwen3_8_2_4t", "qwen38-2.4t", "2.4t-a95b"), "qwen3.8-2.4t-a95b"),
    (("qwen38", "qwen3.8", "qwen3_8"), "qwen3.8-27b"),
    (("qwen36", "qwen3.6", "qwen3_6"), "qwen3.6-27b"),
    (("qwen35", "qwen3.5", "qwen3_5"), "qwen3.5-35b-a3b"),
    (("qwen3-30b", "qwen30ba3b", "qwen3_30b"), "qwen3-30b-a3b"),
)
def _read_json(path: Path) -> Any:
    if path.stat().st_size > _PROFILE_MAX_BYTES: raise ValueError(f"metadata_input_exceeds_50_MiB:{path}")
    with path.open(encoding="utf-8") as stream: return json.load(stream)
def _read_text(path: Path) -> str:
    if path.stat().st_size > _PROFILE_MAX_BYTES: raise ValueError(f"metadata_input_exceeds_50_MiB:{path}")
    return path.read_text(encoding="utf-8", errors="replace")
def _source_digest(path: Path | None, data: Any = None) -> str | None:
    import hashlib
    digest = hashlib.sha256()
    def _feed(file_path: Path, rel: str) -> None:
        digest.update(f"{rel}:{file_path.stat().st_size}:".encode("utf-8"))
        with file_path.open("rb") as stream:
            while chunk := stream.read(65536): digest.update(chunk)
    if isinstance(data, Mapping) and data.get("_consumed_paths"):
        for p_str in data["_consumed_paths"]:
            p = Path(p_str)
            if p.is_file(): _feed(p, p.name)
        return digest.hexdigest()
    if path is not None and path.is_dir():
        items = [p for p in sorted(path.iterdir()) if p.is_file()] + [p for d in sorted(path.glob("rank_*")) if d.is_dir() for p in sorted(d.iterdir()) if p.is_file()]
        for item in items: _feed(item, str(item.relative_to(path)))
        return digest.hexdigest()
    if path is not None and path.is_file():
        _feed(path, path.name)
        return digest.hexdigest()
    if data is not None:
        digest.update(json.dumps(data, sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()
    return None
def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values
def _cmd_flags(text: str) -> dict[str, Any]:
    try: tokens = shlex.split(text)
    except ValueError: tokens = text.split()
    flags: dict[str, Any] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"): continue
        name, _, inline = token.partition("=")
        value: Any = inline if inline else (tokens[index + 1] if index + 1 < len(tokens) and not tokens[index + 1].startswith("-") else True)
        flags[name] = value
    return flags
def _cmd_graph_mode(flags: Mapping[str, Any]) -> str | None:
    if "--enforce-eager" in flags: return "none"
    raw = flags.get("--compilation-config") or flags.get("--compilation_config") or flags.get("--graph-mode") or flags.get("--aclgraph")
    if raw is None: return None
    upper = (json.dumps(raw) if isinstance(raw, Mapping) else str(raw or "")).upper()
    if "FULL_DECODE_ONLY" in upper or "FULL_AND_PIECEWISE" in upper or '"FULL"' in upper or "ACLGRAPH" in upper or upper == "FULL": return "full"
    if "PIECEWISE" in upper: return "piecewise"
    if "EAGER" in upper or "NONE" in upper: return "none"
    return None
def _canonical_model(name: str | None) -> str | None:
    if not name: return None
    raw, lowered = str(name).strip(), str(name).strip().lower()
    for candidate in (raw, lowered):
        if candidate in CATALOG or candidate in ARCHITECTURE_PRESET_CATALOG: return candidate
        for table in (MODEL_ALIASES, ARCHITECTURE_PRESET_ALIASES):
            for alias, target in table.items():
                if alias.lower() == lowered: return target
    sub_pairs = sorted(
        [(k.lower(), k) for k in (*CATALOG, *ARCHITECTURE_PRESET_CATALOG)]
        + [(a.lower(), t) for tbl in (MODEL_ALIASES, ARCHITECTURE_PRESET_ALIASES) for a, t in tbl.items()],
        key=lambda pair: len(pair[0]), reverse=True,
    )
    for needle, target in sub_pairs:
        if len(needle) >= 5 and needle in lowered: return target
    return None


def _infer_model(facts: Mapping[str, Any]) -> str | Path:
    candidates = [facts.get("model_name"), facts.get("weight_path"), facts.get("tag"),
                  facts.get("source_path")]
    for raw in candidates:
        if not raw:
            continue
        text = str(raw)
        direct = _canonical_model(text)
        if direct:
            return direct
        lowered = text.lower()
        for needles, target in _MODEL_PATTERNS:
            if any(needle in lowered for needle in needles):
                if target == "qwen3.5-35b-a3b" and not facts.get("experts"):
                    return "qwen3.5-27b"
                return target
        candidate = Path(text)
        if candidate.suffix == ".json" and candidate.is_file():
            return candidate
    raise ModelingError(f"profile_model_not_recognized:{facts.get('model_name') or facts.get('tag')}")


def _infer_hardware(facts: Mapping[str, Any]) -> str:
    for key in ("hardware", "npu_type", "tag", "host"):
        if not (raw := facts.get(key)): continue
        text, lowered = str(raw).strip(), str(raw).strip().lower()
        try: return resolve_hardware_sku(text)
        except ModelingError: pass
        if "a5" in lowered or "910_95" in lowered or "950" in lowered: return "a5"
        if "a3" in lowered or "910_93" in lowered or "910c" in lowered: return "a3"
        if "a2" in lowered or "910b" in lowered: return "a2"
        if key in {"hardware", "npu_type"}: raise ModelingError(f"profile_hardware_not_recognized:{text}")
    return "a3"

def _gather_facts(data: Any) -> dict[str, Any]:
    nodes, facts = _candidate_mappings(data), {}
    for field, keys in (("model_name", ("model_name", "model", "served_model_name", "model_id")),
                        ("npu_type", ("npu_type", "hardware", "sku", "chip")),
                        ("tag", ("tag", "cell", "cell_tag")), ("host", ("host",)),
                        ("weight_path", ("weight_path", "checkpoint", "model_path")),
                        ("quant", ("quantization", "quant", "quant_method")), ("mode", ("mode",))):
        if value := _first_text(data, keys): facts[field] = value
    for field, keys in (("layers", ("num_hidden_layers", "num_layers", "layers")),
                        ("experts", ("num_routed_experts", "n_routed_experts", "num_experts")),
                        ("top_k", ("num_experts_per_tok", "top_k")),
                        ("max_model_len", ("max_model_len", "max_model_length")),
                        ("max_num_seqs", ("max_num_seqs",)), ("max_num_batched_tokens", ("max_num_batched_tokens",)),
                        ("block_size", ("block_size", "kv_block_size"))):
        if (value := _first_int(data, keys)) is not None: facts[field] = value
    for field, keys in (("gpu_memory_utilization", ("gpu_memory_utilization",)),
                        ("async_scheduling", ("async_scheduling",)),
                        ("enable_chunked_prefill", ("enable_chunked_prefill",)), ("sp", ("sp", "sequence_parallel"))):
        for value in _deep_values(data, keys):
            if isinstance(value, bool): facts[field] = value; break
            if isinstance(value, (int, float)): facts[field] = float(value); break
            if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
                facts[field] = value.strip().lower() in {"true", "1"}; break
    for key in ("tp", "dp", "ep", "pp", "pcp", "dcp", "cp"):
        val = next((int(h[key]) for node in nodes
                    for h in (node, node.get("parallelism"), node.get("parallel"), node.get("scene"))
                    if isinstance(h, Mapping) and isinstance(h.get(key), int) and not isinstance(h.get(key), bool)), None)
        if val is not None: facts[key] = val
    if "dcp" not in facts and "cp" in facts: facts["dcp"] = facts["cp"]
    for key in ("input_len", "prompt_tokens", "context_len", "seq_len", "context_tokens",
                "prefix_len", "prefix_tokens", "shared_tokens", "output_len",
                "output_tokens", "max_tokens", "completion_tokens", "decode_steps",
                "decode_len", "concurrency", "batch_size"):
        if (value := _first_int(data, (key,))) is not None: facts.setdefault(key, value)
    if raw_gm := _first_text(data, ("graph_capture_mode", "graph_mode", "execution_mode", "aclgraph_mode")):
        gm_low = raw_gm.strip().lower()
        if "full" in gm_low: facts.setdefault("graph_mode", "full")
        elif "piecewise" in gm_low: facts.setdefault("graph_mode", "piecewise")
        elif gm_low in {"none", "eager", "off", "disabled"}: facts.setdefault("graph_mode", "none")
    for val in _deep_values(data, ("is_eager", "enforce_eager")):
        if val is True or (isinstance(val, str) and val.strip().lower() in {"true", "1"}):
            facts.setdefault("graph_mode", "none"); break
    for node in nodes:
        workload = node.get("workload")
        if isinstance(workload, Mapping) and isinstance(workload.get("mode"), str):
            facts["mode"] = workload["mode"]
            break
    env = data.get("meta_env") if isinstance(data, Mapping) else None
    if isinstance(env, Mapping):
        for key, field, coercion in (("MODEL_NAME", "model_name", str), ("NPU_TYPE", "hardware", str),
                                     ("TP", "tp", int), ("DP", "dp", int), ("EP", "ep", int),
                                     ("NUM_LAYERS", "layers", int), ("NUM_EXPERTS", "experts", int),
                                     ("TOP_K", "top_k", int), ("INPUT_LEN", "input_len", int),
                                     ("PREFIX_LEN", "prefix_len", int), ("OUTPUT_LEN", "output_len", int),
                                     ("CONCURRENCY", "concurrency", int), ("QUANTIZATION", "quant", str),
                                     ("QUANT", "quant", str), ("MODE", "mode", str)):
            if key in env:
                try:
                    facts.setdefault(field, coercion(env[key]))
                except (TypeError, ValueError):
                    pass
    flags = data.get("cmd_flags") if isinstance(data, Mapping) else None
    if isinstance(flags, Mapping):
        mapping = {
            "--tensor-parallel-size": ("tp", int), "--data-parallel-size": ("dp", int),
            "--expert-parallel-size": ("ep", int), "--pipeline-parallel-size": ("pp", int),
            "--max-model-len": ("max_model_len", int), "--max-num-seqs": ("max_num_seqs", int),
            "--max-num-batched-tokens": ("max_num_batched_tokens", int),
            "--block-size": ("block_size", int),
            "--gpu-memory-utilization": ("gpu_memory_utilization", float),
            "--quantization": ("quant", str),
            "--dtype": ("dtype", str),
            "--served-model-name": ("model_name", str),
            "--model": ("model_name", str),
        }
        for flag, (field, coercion) in mapping.items():
            if flag in flags and flags[flag] is not True:
                try:
                    facts.setdefault(field, coercion(flags[flag]))
                except (TypeError, ValueError):
                    pass
        if "--enable-expert-parallel" in flags and "ep" not in facts:
            facts["ep"] = int(facts.get("tp", 1) or 1)
        if "--async-scheduling" in flags:
            facts["async_scheduling"] = True
        if "--enable-chunked-prefill" in flags:
            facts["enable_chunked_prefill"] = True
        cmd_gm = _cmd_graph_mode(flags)
        if cmd_gm is not None:
            facts.setdefault("graph_mode", cmd_gm)
    spec_k = _first_int(data, ("spec_tokens", "spec_k", "num_speculative_tokens", "mtp_proposal_tokens", "draft_proposal_tokens"))
    if spec_k is None and isinstance(env, Mapping):
        for ek in ("SPEC_K", "NUM_SPECULATIVE_TOKENS", "MTP_PROPOSAL_TOKENS"):
            if ek in env:
                try: spec_k = int(env[ek]); break
                except (TypeError, ValueError): pass
    if spec_k is not None and spec_k >= 0:
        facts["mtp_proposal_tokens"] = spec_k
    acc_v = _first_number(data, ("mtp_acceptance_length", "acceptance_length"))
    if acc_v is None and isinstance(env, Mapping) and "MTP_ACCEPTANCE_LENGTH" in env:
        try: acc_v = float(env["MTP_ACCEPTANCE_LENGTH"])
        except (TypeError, ValueError): pass
    if acc_v is not None and acc_v >= 1.0:
        facts["mtp_acceptance_length"] = acc_v
    text = " ".join(str(value) for key, value in facts.items() if key in {"tag", "model_name", "weight_path"})
    match = re.search(r"_L(\d+)_E(\d+)_K(\d+)", text)
    if match:
        facts.setdefault("layers", int(match.group(1)))
        facts.setdefault("experts", int(match.group(2)))
        facts.setdefault("top_k", int(match.group(3)))
    return facts


def _validate_case_envelope(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get("structure_variant"):
        raise ModelingError("structure_variant_retired:use_model_layer_plans_or_layer_overrides")
    allowed = {"machine", "model", "workload", "runtime", "slo", "schema", "structure_variant",
               "_source_kind", "_files", "_consumed_paths", "_profile_dir", "_rank_files",
               "observed_profile", "observed", "profiler_provenance"}
    unknown = {str(k) for k in raw if not str(k).startswith("observed_")} - allowed
    if unknown:
        raise ModelingError("unknown_field:case." + ",".join(sorted(unknown)))
    return {name: deepcopy(raw[name]) for name in ("machine", "model", "workload", "runtime", "slo") if name in raw}


def _unwrap_case(data: Any, depth: int = 0) -> dict[str, Any] | None:
    if not isinstance(data, Mapping) or depth > 2: return None
    for wrapper in ("capture", "provenance", "summary_document", "attribution_document"):
        if isinstance(nw := data.get(wrapper), Mapping) and (found := _unwrap_case(nw, depth + 1)): return found
    for key in ("case", "sim_case", "case_document"):
        if isinstance(nested := data.get(key), Mapping) and {"machine", "model", "workload"} <= set(nested):
            return _validate_case_envelope(nested)
    return _validate_case_envelope(data) if {"machine", "model", "workload"} <= set(data) else None

def _ingredients_from_data(data: Any, path: Path | None, model_override: Any = None,
                           hardware_override: Any = None, delegated: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = _unwrap_case(data)
    if base is not None:
        return {"base_case": base, "source_kind": "exported_case", "facts": _gather_facts(data)}
    facts = _gather_facts(data)
    facts.setdefault("source_path", str(path) if path else None)
    _m = lambda v: dict(v) if isinstance(v, Mapping) else {}
    del_rt, del_mod, del_hw = _m(_m(delegated).get("runtime")), _m(_m(delegated).get("model")), _m(_m(delegated).get("machine")).get("sku")
    del_par, del_reqs = _m(del_rt.get("parallel")), _m(_m(delegated).get("workload")).get("requests")
    del_req0 = _m(del_reqs[0]) if isinstance(del_reqs, list) and del_reqs else {}
    quant = facts.get("quant")
    if quant and str(quant).lower() in {"ascend", "auto", "none"}:
        quant = None
    if quant is None:
        w = f"{facts.get('model_name') or ''} {facts.get('weight_path') or ''} {facts.get('tag') or ''}".lower()
        quant = ("w8a8_mxfp8" if ("w8a8" in w and "mxfp8" in w) else "mxfp8" if "mxfp8" in w
                 else "w8a8" if "w8a8" in w else "w4a8" if "w4a8" in w else "fp8" if "fp8" in w
                 else del_mod.get("quantization"))
    model_val = model_override
    if model_val is None:
        try:
            model_val = _infer_model(facts)
        except ModelingError:
            model_val = del_mod.get("catalog_key") or del_mod.get("name")
            if not model_val:
                raise
    hw_val = hardware_override if hardware_override is not None else (
        _infer_hardware(facts) if any(facts.get(k) for k in ("hardware", "npu_type", "tag", "host")) else (del_hw or "a3"))
    return {
        "facts": facts, "source_kind": "profile", "model": model_val,
        "hardware": hw_val,
        "tp": int(facts.get("tp") or del_par.get("tp") or 1), "dp": int(facts.get("dp") or del_par.get("dp") or 1),
        "ep": int(facts.get("ep") or del_par.get("ep") or 1), "pp": int(facts.get("pp") or del_par.get("pp") or 1),
        "pcp": int(facts.get("pcp") or del_par.get("pcp") or 1),
        "dcp": int(facts.get("dcp") or facts.get("cp") or del_par.get("dcp") or del_par.get("cp") or 1),
        "sp": bool(facts.get("sp", False)),
        "layers": facts.get("layers") if facts.get("layers") is not None else del_mod.get("layers"),
        "experts": facts.get("experts") if facts.get("experts") is not None else (del_mod.get("num_experts") or del_mod.get("n_routed_experts") or None),
        "top_k": facts.get("top_k") if facts.get("top_k") is not None else (del_mod.get("top_k") or del_mod.get("num_experts_per_tok") or None),
        "prompt_tokens": int(facts.get("input_len") or facts.get("prompt_tokens") or facts.get("context_len")
                             or facts.get("seq_len") or facts.get("context_tokens") or del_req0.get("prompt_tokens") or 4096),
        "prefix_tokens": int(next((facts[k] for k in ("prefix_len", "prefix_tokens", "shared_tokens") if facts.get(k) is not None), del_req0.get("prefix_tokens") or 0)),
        "output_tokens": int(facts.get("output_len") or facts.get("output_tokens") or facts.get("max_tokens")
                             or facts.get("completion_tokens") or facts.get("decode_steps") or facts.get("decode_len") or del_req0.get("output_tokens") or 128),
        "concurrency": int(facts.get("concurrency") or facts.get("batch_size") or (len(del_reqs) if isinstance(del_reqs, list) and del_reqs else 0) or 1),
        "quant": quant, "graph_mode": facts.get("graph_mode") or del_rt.get("graph_capture_mode") or del_rt.get("graph_mode") or "none",
        "max_model_len": facts.get("max_model_len") or del_rt.get("max_model_len"),
        "max_num_seqs": facts.get("max_num_seqs") or del_rt.get("max_num_seqs"),
        "max_num_batched_tokens": facts.get("max_num_batched_tokens") or del_rt.get("max_num_batched_tokens"),
        "gpu_memory_utilization": facts.get("gpu_memory_utilization") if facts.get("gpu_memory_utilization") is not None else del_rt.get("gpu_memory_utilization"),
        "block_size": facts.get("block_size") or del_rt.get("kv_block_size"),
        "enable_chunked_prefill": bool(facts.get("enable_chunked_prefill", del_rt.get("enable_chunked_prefill", True))),
        "async_scheduling": bool(facts.get("async_scheduling", del_rt.get("async_scheduling", False))),
        "mtp_proposal_tokens": int(facts.get("mtp_proposal_tokens") if facts.get("mtp_proposal_tokens") is not None else (_m(del_rt.get("metadata")).get("draft_proposal_tokens") or _m(del_rt.get("metadata")).get("spec_tokens") or 0)),
        "mtp_acceptance_length": float(facts.get("mtp_acceptance_length") if facts.get("mtp_acceptance_length") is not None else (del_rt.get("mtp_acceptance_length") or 1.0)),
        "phase": facts.get("phase") or ("prefill" if str(facts.get("mode", "")).startswith("prefill") else "decode"),
    }

def _load_directory(path: Path) -> tuple[Any, list[str]]:
    files = sorted(item.name for item in path.iterdir() if item.is_file())
    data: dict[str, Any] = {"_profile_dir": str(path), "_files": files}
    def _opt_env_cmd() -> None:
        if (path / "meta.env").is_file(): data["meta_env"] = _parse_env(_read_text(path / "meta.env"))
        if (path / "cmd.sh").is_file(): data["cmd_flags"] = _cmd_flags(_read_text(path / "cmd.sh"))
    if (path / "run_meta.json").is_file():
        data["run_meta"] = _read_json(path / "run_meta.json")
        if (path / "benchmark_summary.json").is_file(): data["benchmark_summary"] = _read_json(path / "benchmark_summary.json")
        _opt_env_cmd(); data["_source_kind"] = "wave5_capture"; return data, files
    if (path / "capture.json").is_file() or (path / "provenance.json").is_file():
        if (path / "capture.json").is_file(): data["capture"] = _read_json(path / "capture.json")
        if (path / "provenance.json").is_file(): data["provenance"] = _read_json(path / "provenance.json")
        data["_rank_files"] = [str(item) for rank_dir in sorted(path.glob("rank_*"))
                               for item in (rank_dir / "op_statistic.csv", rank_dir / "kernel_details.csv") if item.is_file()]
        data["_source_kind"] = "profiler_capture"; return data, files
    if (path / "summary.json").is_file() or (path / "attribution.json").is_file():
        if (path / "summary.json").is_file(): data["summary_document"] = _read_json(path / "summary.json")
        if (path / "attribution.json").is_file(): data["attribution_document"] = _read_json(path / "attribution.json")
        data["_source_kind"] = "c7_profile"; return data, files
    if (path / "benchmark_summary.json").is_file():
        data["benchmark_summary"] = _read_json(path / "benchmark_summary.json")
        _opt_env_cmd(); data["_source_kind"] = "wave5_benchmark"; return data, files
    cases = sorted(path.glob("*.case.json")) or sorted(path.glob("*.json"))
    if cases:
        data["case_document"] = _read_json(cases[0]); data["_source_kind"] = "profile_case"; return data, files
    if (path / "meta.env").is_file() or (path / "cmd.sh").is_file():
        _opt_env_cmd(); data["_source_kind"] = "wave5_capture"; return data, files
    raise ModelingError(f"profile_directory_unrecognized:{path}")

def _read_source(source: str | Path | Mapping[str, Any] | CaseSpec) -> tuple[Any, Path | None, list[str]]:
    if isinstance(source, CaseSpec): return source.to_dict(), None, []
    if isinstance(source, Mapping): return dict(source), None, []
    path = Path(source)
    if not path.exists(): raise ModelingError(f"profile_not_found:{path}")
    if path.is_dir():
        data, files = _load_directory(path)
        return data, path, files
    suffix = path.suffix.lower()
    if suffix == ".json": data = _read_json(path)
    elif suffix in {".env", ".sh", ".txt"}:
        text = _read_text(path)
        data = {"meta_env": _parse_env(text)} if suffix == ".env" else {"cmd_text": text, "cmd_flags": _cmd_flags(text)}
    else: raise ModelingError(f"profile_file_unsupported:{path}")
    if not isinstance(data, dict): data = {"case_document": data}
    data.setdefault("_source_kind", "profile_file")
    if isinstance(data, Mapping) and {"machine", "model", "workload"} <= set(data):
        data["_source_kind"] = "exported_case"
    parent, consumed = path.parent, [path]
    if data.get("_source_kind") != "exported_case":
        sibling_json = {"run_meta.json": "run_meta", "benchmark_summary.json": "benchmark_summary",
                        "summary.json": "summary_document", "attribution.json": "attribution_document",
                        "capture.json": "capture", "provenance.json": "provenance"}
        for name, target_key in sibling_json.items():
            candidate = parent / name
            if candidate.is_file() and candidate != path:
                data.setdefault(target_key, _read_json(candidate)); consumed.append(candidate)
        if (parent / "meta.env").is_file() and (parent / "meta.env") != path:
            data.setdefault("meta_env", _parse_env(_read_text(parent / "meta.env"))); consumed.append(parent / "meta.env")
        if (parent / "cmd.sh").is_file() and (parent / "cmd.sh") != path:
            data.setdefault("cmd_flags", _cmd_flags(_read_text(parent / "cmd.sh"))); consumed.append(parent / "cmd.sh")
        for csv_name in ("op_statistic.csv", "kernel_details.csv"):
            if (parent / csv_name).is_file() and (parent / csv_name) != path: consumed.append(parent / csv_name)
    files = [p.name for p in consumed]
    data["_files"], data["_consumed_paths"] = files, [str(p) for p in consumed]
    return data, path, files


def _load_observed_evidence(data: Any, path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = _extract_observed(data, path=path)
    provenance = {
        "schema": _PROVENANCE_SCHEMA,
        "source_kind": str(data.get("_source_kind", "profile")) if isinstance(data, Mapping) else "profile",
        "source_path": str(path) if path else None, "source_sha256": _source_digest(path, data),
        "files": list(data.get("_files", ())) if isinstance(data, Mapping) else [],
    }
    if isinstance(data, Mapping) and data.get("_profile_dir"):
        provenance["profile_directory"] = str(data["_profile_dir"])
    return observed, provenance


def _compose_from_ingredients(ingredients: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(ingredients)
    pick = lambda name, default: default if overrides.get(name) is None else overrides[name]
    prompt_tokens = int(pick("prompt_tokens", values.get("prompt_tokens", 4096)))
    prefix_tokens = min(int(pick("prefix_tokens", values.get("prefix_tokens", 0))), prompt_tokens)
    output_tokens = int(pick("output_tokens", values.get("output_tokens", 128)))
    concurrency = int(pick("concurrency", values.get("concurrency", 1)))
    graph_mode = str(pick("graph_mode", values.get("graph_mode", "none")) or "none")
    if graph_mode not in {"none", "full", "piecewise"}: raise ValueError(f"invalid_graph_mode:{graph_mode}")
    case = compose_case(
        model=pick("model", values["model"]), hardware=pick("hardware", values["hardware"]),
        tp=int(pick("tp", values.get("tp", 1))), dp=int(pick("dp", values.get("dp", 1))),
        ep=int(pick("ep", values.get("ep", 1))), pp=int(pick("pp", values.get("pp", 1))),
        pcp=int(pick("pcp", values.get("pcp", 1))), dcp=int(pick("dcp", values.get("dcp", 1))),
        sp=bool(pick("sp", values.get("sp", False))),
        num_devices=pick("num_devices", values.get("num_devices")),
        ranks_per_node=pick("ranks_per_node", values.get("ranks_per_node")),
        layers=pick("layers", values.get("layers")),
        n_routed_experts=pick("experts", values.get("experts")),
        num_experts_per_tok=pick("top_k", values.get("top_k")),
        quant=pick("quant", values.get("quant")),
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        concurrency=concurrency, prefix_tokens=prefix_tokens,
        max_num_seqs=pick("max_num_seqs", values.get("max_num_seqs")),
        max_num_batched_tokens=pick("max_num_batched_tokens", values.get("max_num_batched_tokens")),
        max_model_len=pick("max_model_len", values.get("max_model_len")),
        block_size=int(pick("block_size", values.get("block_size", 128)) or 128),
        enable_chunked_prefill=bool(pick("enable_chunked_prefill", values.get("enable_chunked_prefill", True))),
        async_scheduling=bool(pick("async_scheduling", values.get("async_scheduling", False))),
        graph_capture_mode=graph_mode,
        mtp_proposal_tokens=int(pick("mtp_proposal_tokens", values.get("mtp_proposal_tokens", 0)) or 0),
        mtp_acceptance_length=float(pick("mtp_acceptance_length", values.get("mtp_acceptance_length", 1.0)) or 1.0),
        ttft_slo_ms=float(pick("ttft_slo_ms", 5000.0)), tpot_slo_ms=float(pick("tpot_slo_ms", 50.0)),
    )
    if (gmu := pick("gpu_memory_utilization", values.get("gpu_memory_utilization"))) is not None:
        case["runtime"]["gpu_memory_utilization"] = float(gmu)
    return case


def _apply_graph_keys(runtime: dict[str, Any], mode: str) -> None:
    max_seqs, max_batched = max(1, int(runtime.get("max_num_seqs", 1))), max(1, int(runtime.get("max_num_batched_tokens", 8192)))
    if mode == "full": runtime["graph_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, max_seqs})
    elif mode == "piecewise": runtime["piecewise_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, max_batched})


def _normalized_requests(workload: Any) -> list[dict[str, Any]]:
    if not isinstance(workload, Mapping): return []
    reqs = workload.get("requests")
    if isinstance(reqs, (list, tuple)) and reqs: return [deepcopy(dict(r)) for r in reqs if isinstance(r, Mapping)]
    try:
        from .schema.model import WorkloadSpec
        return [r.to_dict() for r in WorkloadSpec.from_mapping(workload).requests]
    except Exception: return []


def _merge_requests(requests: Any, overrides: Mapping[str, Any], runtime: dict[str, Any]) -> list[dict[str, Any]]:
    reqs = list(requests or ())
    if not any(overrides.get(k) is not None for k in ("prompt_tokens", "output_tokens", "prefix_tokens", "concurrency")):
        return [deepcopy(dict(r)) for r in reqs]
    count = int(overrides["concurrency"]) if overrides.get("concurrency") is not None else (len(reqs) or 1)
    tmpl, out_reqs, used_ids = (dict(reqs[0]) if reqs else {}), [], set()
    for idx in range(count):
        req = deepcopy(dict(reqs[idx])) if idx < len(reqs) else deepcopy(tmpl)
        raw_id = (req.get("id") or req.get("request_id")) if idx < len(reqs) else None
        rid, k = (str(raw_id) if raw_id and str(raw_id) not in used_ids else f"req-{idx}"), idx
        while rid in used_ids:
            k += 1; rid = f"req-{k}"
        used_ids.add(rid)
        req["id"] = rid
        if "request_id" in req: req["request_id"] = rid
        req["arrival_ms"] = float(req.get("arrival_ms", 0.0))
        p_tok = int(overrides["prompt_tokens"]) if overrides.get("prompt_tokens") is not None else int(req.get("prompt_tokens", 4096))
        o_tok = int(overrides["output_tokens"]) if overrides.get("output_tokens") is not None else int(req.get("output_tokens", 128))
        pref = int(overrides["prefix_tokens"]) if overrides.get("prefix_tokens") is not None else int(req.get("prefix_tokens", 0))
        req.update(prompt_tokens=p_tok, output_tokens=o_tok, prefix_tokens=min(pref, p_tok))
        out_reqs.append(req)
    runtime["max_model_len"] = max(int(runtime.get("max_model_len", 32768)), max((r["prompt_tokens"] + r["output_tokens"] for r in out_reqs), default=4096))
    runtime["max_num_seqs"] = max(int(runtime.get("max_num_seqs", 256)), count)
    runtime["max_num_batched_tokens"] = max(int(runtime.get("max_num_batched_tokens", 8192)), sum(r["prompt_tokens"] for r in out_reqs))
    return out_reqs


_SPEC_META_KEYS = ("draft_mode", "draft_proposal_tokens", "spec_tokens", "speculative", "draft_accept_tokens", "draft_accept_trace")


def _derive_from_base_case(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    _pick = lambda key, default: default if overrides.get(key) is None else overrides[key]
    structural = any(overrides.get(key) is not None for key in (
        "model", "hardware", "tp", "dp", "ep", "pp", "pcp", "dcp", "layers", "experts",
        "top_k", "quant", "num_devices", "ranks_per_node", "sp"))
    if not structural:
        case = deepcopy(dict(base))
        runtime = case.setdefault("runtime", {})
        requests = _normalized_requests(case.get("workload"))
        if any(overrides.get(k) is not None for k in ("prompt_tokens", "output_tokens", "prefix_tokens", "concurrency")):
            case["workload"] = {"requests": _merge_requests(requests, overrides, runtime)}
        graph_mode = overrides.get("graph_mode")
        if graph_mode is not None:
            if graph_mode not in {"none", "full", "piecewise"}: raise ValueError(f"invalid_graph_mode:{graph_mode}")
            runtime["graph_mode"] = runtime["graph_capture_mode"] = graph_mode
            _apply_graph_keys(runtime, graph_mode)
            meta = dict(runtime.get("metadata") or {})
            meta["execution_mode"] = "aclgraph_full" if graph_mode == "full" else "piecewise" if graph_mode == "piecewise" else "eager"
            runtime["metadata"] = meta
        if overrides.get("async_scheduling") is not None: runtime["async_scheduling"] = bool(overrides["async_scheduling"])
        if overrides.get("enable_chunked_prefill") is not None: runtime["enable_chunked_prefill"] = bool(overrides["enable_chunked_prefill"])
        if overrides.get("gpu_memory_utilization") is not None: runtime["gpu_memory_utilization"] = float(overrides["gpu_memory_utilization"])
        for key in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
            if overrides.get(key) is not None: runtime[key] = int(overrides[key])
        if overrides.get("block_size") is not None: runtime["kv_block_size"] = int(overrides["block_size"])
        if overrides.get("ttft_slo_ms") is not None or overrides.get("tpot_slo_ms") is not None:
            slo = case.setdefault("slo", {})
            slo["ttft_ms"] = float(_pick("ttft_slo_ms", slo.get("ttft_ms", 5000.0)))
            slo["tpot_ms"] = float(_pick("tpot_slo_ms", slo.get("tpot_ms", 50.0)))
        if overrides.get("mtp_proposal_tokens") is not None or overrides.get("mtp_acceptance_length") is not None:
            meta = dict(runtime.get("metadata") or {})
            prop = int(_pick("mtp_proposal_tokens", meta.get("draft_proposal_tokens", 0) or 0))
            if prop <= 0:
                for dk in _SPEC_META_KEYS: meta.pop(dk, None)
                runtime["mtp_acceptance_length"] = 1.0
                for rk in ("mtp_acceptance_by_output_tokens", "dspark"): runtime.pop(rk, None)
            else:
                meta["draft_proposal_tokens"] = prop
                meta.setdefault("draft_mode", "mtp")
                runtime["mtp_acceptance_length"] = float(_pick("mtp_acceptance_length", runtime.get("mtp_acceptance_length", 1.0)))
            runtime["metadata"] = meta
        parse_case(case)
        return case

    runtime = base.get("runtime", {})
    raw_meta = dict(runtime.get("metadata") or {})
    parallel = Software.parallel_mapping(runtime) if isinstance(runtime, Mapping) else {}
    requests = _normalized_requests(base.get("workload"))
    first = requests[0] if requests else {}
    model, slo, machine = base.get("model", {}), base.get("slo", {}), base.get("machine", {})
    graph_mode = str(_pick("graph_mode", runtime.get("graph_capture_mode") or runtime.get("graph_mode") or "none"))
    if graph_mode not in {"none", "full", "piecewise"}: graph_mode = "none"
    prompt = int(_pick("prompt_tokens", first.get("prompt_tokens") if first.get("prompt_tokens") is not None else 4096))
    prefix = min(int(_pick("prefix_tokens", first.get("prefix_tokens", 0) if first else 0)), prompt)
    base_quant = model.get("quantization") or (model.get("quant", {}).get("name") if isinstance(model.get("quant"), Mapping) else model.get("quant"))
    tp, dp, ep = int(_pick("tp", parallel.get("tp", 1))), int(_pick("dp", parallel.get("dp", 1))), int(_pick("ep", parallel.get("ep", 1)))
    pp, pcp = int(_pick("pp", parallel.get("pp", 1))), int(_pick("pcp", parallel.get("pcp", 1)))
    dcp = int(_pick("dcp", parallel.get("dcp") or parallel.get("cp", 1)))
    req_devs = max(tp * dp * pp * pcp, ep * pp, dp * dcp * pp)
    base_devs = len(machine.get("devices") or ()) if isinstance(machine, Mapping) else 0
    if not base_devs and isinstance(machine, Mapping) and isinstance(machine.get("num_devices"), int): base_devs = int(machine["num_devices"])
    num_devices = overrides.get("num_devices") if overrides.get("num_devices") is not None else (base_devs if base_devs >= req_devs else None)
    base_rpn = machine.get("ranks_per_node") if isinstance(machine, Mapping) else None
    eff_devs = int(num_devices or req_devs)
    ranks_per_node = overrides.get("ranks_per_node") if overrides.get("ranks_per_node") is not None else (
        int(base_rpn) if isinstance(base_rpn, int) and base_rpn > 0 and (eff_devs <= base_rpn or eff_devs % base_rpn == 0) else None)
    mtp_prop = int(_pick("mtp_proposal_tokens", raw_meta.get("draft_proposal_tokens", 0) or 0))
    mtp_acc = 1.0 if mtp_prop <= 0 else float(_pick("mtp_acceptance_length", runtime.get("mtp_acceptance_length", 1.0) or 1.0))
    raw_hw = _pick("hardware", (machine.get("sku") or machine.get("name") or "a3") if isinstance(machine, Mapping) else "a3")
    custom_hw = False
    if overrides.get("hardware") is None and isinstance(machine, Mapping):
        try: resolve_hardware_sku(str(raw_hw))
        except ModelingError: raw_hw, custom_hw = "a3", True
    case = compose_case(
        model=_pick("model", model), hardware=raw_hw,
        tp=tp, dp=dp, ep=ep, pp=pp, pcp=pcp, dcp=dcp,
        sp=bool(_pick("sp", raw_meta.get("sp", False))),
        num_devices=num_devices, ranks_per_node=ranks_per_node,
        layers=_pick("layers", model.get("layers")),
        n_routed_experts=_pick("experts", model.get("num_experts") or model.get("n_routed_experts") or None),
        num_experts_per_tok=_pick("top_k", model.get("top_k") or model.get("num_experts_per_tok") or None),
        quant=_pick("quant", base_quant if overrides.get("model") is None else None),
        prompt_tokens=prompt,
        output_tokens=int(_pick("output_tokens", first.get("output_tokens") if first.get("output_tokens") is not None else 128)),
        concurrency=int(_pick("concurrency", len(requests) or 1)),
        prefix_tokens=prefix,
        max_num_seqs=_pick("max_num_seqs", runtime.get("max_num_seqs")),
        max_num_batched_tokens=_pick("max_num_batched_tokens", runtime.get("max_num_batched_tokens")),
        max_model_len=_pick("max_model_len", runtime.get("max_model_len")),
        block_size=int(_pick("block_size", runtime.get("kv_block_size", 128))),
        enable_chunked_prefill=bool(_pick("enable_chunked_prefill", runtime.get("enable_chunked_prefill", True))),
        async_scheduling=bool(_pick("async_scheduling", runtime.get("async_scheduling", False))),
        graph_capture_mode=graph_mode,
        mtp_proposal_tokens=mtp_prop, mtp_acceptance_length=mtp_acc,
        ttft_slo_ms=float(_pick("ttft_slo_ms", slo.get("ttft_ms", 5000.0))),
        tpot_slo_ms=float(_pick("tpot_slo_ms", slo.get("tpot_ms", 50.0))),
    )
    if requests:
        case["workload"]["requests"] = _merge_requests(requests, overrides, case["runtime"])
    if overrides.get("quant") is None and overrides.get("model") is None:
        for qk in ("quant", "quantization", "weight_dtype", "activation_dtype", "kv_dtype", "compute_dtype"):
            if qk in model: case["model"][qk] = deepcopy(model[qk])
    recomputed = {"parallel", "metadata", "max_num_seqs", "max_num_batched_tokens", "max_model_len",
                  "kv_block_size", "enable_chunked_prefill", "async_scheduling", "graph_mode",
                  "graph_capture_mode", "graph_capture_keys", "piecewise_capture_keys",
                  "mtp_acceptance_length", "gpu_memory_utilization", "kernels", "pp_layer_partition",
                  "dp", "tp", "ep", "pp", "cp", "dcp", "pcp", "cp_plan",
                  "prefill_context_parallel_size", "context_parallel_plan"} | (
                      {"mtp_acceptance_by_output_tokens", "dspark"} if mtp_prop <= 0 else set())
    for key, val in runtime.items():
        if key not in recomputed: case["runtime"][key] = deepcopy(val)
    if overrides.get("layers") is None and overrides.get("pp") is None and "pp_layer_partition" in runtime:
        case["runtime"]["pp_layer_partition"] = deepcopy(runtime["pp_layer_partition"])
    if not any(overrides.get(k) is not None for k in ("tp", "dp", "ep", "pp", "pcp", "dcp")) and parallel:
        case["runtime"]["parallel"] = deepcopy(parallel)
    elif overrides.get("dcp") is None and overrides.get("pcp") is None and "cp_plan" in parallel and (dcp > 1 or pcp > 1):
        case["runtime"]["parallel"]["cp_plan"] = parallel["cp_plan"]
    if overrides.get("graph_mode") is None and overrides.get("max_num_seqs") is None and overrides.get("max_num_batched_tokens") is None:
        for gk in ("graph_mode", "graph_capture_mode", "graph_capture_keys", "piecewise_capture_keys", "graph_buckets"):
            if gk in runtime: case["runtime"][gk] = deepcopy(runtime[gk])
    if overrides.get("hardware") is None:
        new_sp = case["runtime"].get("kernels", {}).get("sequence_parallel")
        if "kernels" in runtime:
            merged_k = deepcopy(dict(runtime["kernels"]))
            if overrides.get("sp") is not None:
                if new_sp is not None: merged_k["sequence_parallel"] = deepcopy(new_sp)
                else: merged_k.pop("sequence_parallel", None)
            case["runtime"]["kernels"] = merged_k
        elif custom_hw:
            case["runtime"]["kernels"] = {"sequence_parallel": deepcopy(new_sp)} if (overrides.get("sp") and new_sp) else {}
        if isinstance(machine, Mapping):
            if custom_hw or (overrides.get("num_devices") is None and overrides.get("ranks_per_node") is None and base_devs == eff_devs):
                m_copy = deepcopy(dict(machine))
                if custom_hw and base_devs != eff_devs:
                    if isinstance(m_copy.get("devices"), list) and m_copy["devices"]:
                        d0 = m_copy["devices"][0]
                        rpn = int(ranks_per_node or m_copy.get("ranks_per_node") or eff_devs)
                        dpp = int(m_copy.get("dies_per_package") or 1)
                        m_copy["devices"] = [{**deepcopy(d0), "rank": i, "node": i // rpn, "package": i // dpp} for i in range(eff_devs)]
                        for lnk in m_copy.get("links") or ():
                            if isinstance(lnk, dict) and "members" in lnk: lnk["members"] = list(range(eff_devs))
                    if "num_devices" in m_copy: m_copy["num_devices"] = eff_devs
                if ranks_per_node is not None and "ranks_per_node" in m_copy:
                    m_copy["ranks_per_node"] = int(ranks_per_node)
                case["machine"] = m_copy
            elif "device" in machine:
                case["machine"]["device"] = deepcopy(machine["device"])
    if _pick("gpu_memory_utilization", runtime.get("gpu_memory_utilization")) is not None:
        case["runtime"]["gpu_memory_utilization"] = float(_pick("gpu_memory_utilization", runtime.get("gpu_memory_utilization")))
    meta = dict(case["runtime"].get("metadata") or {})
    for key, val in raw_meta.items():
        if mtp_prop <= 0 and key in _SPEC_META_KEYS: continue
        if key == "execution_mode" and overrides.get("graph_mode") is not None: continue
        if key not in meta or (key == "sp" and overrides.get("sp") is None): meta[key] = deepcopy(val)
    if overrides.get("graph_mode") is not None:
        meta["execution_mode"] = "aclgraph_full" if graph_mode == "full" else "piecewise" if graph_mode == "piecewise" else "eager"
    if meta: case["runtime"]["metadata"] = meta
    return case


def _merge_observed_profiles(base: Any, extra: Any) -> dict[str, Any]:
    merged = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(extra, Mapping): return merged
    for key, value in extra.items():
        if value is None: merged.setdefault(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_observed_profiles(merged[key], value)
        else: merged[key] = value
    if merged and "schema" not in merged: merged["schema"] = "mindie-sim.observed-profile/1"
    return merged


def _attach_profile_metadata(case: dict[str, Any], observed: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    runtime = case.setdefault("runtime", {})
    meta = dict(runtime.get("metadata") or {})
    if merged_observed := _merge_observed_profiles(meta.get("observed_profile"), observed):
        meta["observed_profile"] = merged_observed
    existing_provenance = meta.get("profiler_provenance")
    meta["profiler_provenance"] = {**(dict(existing_provenance) if isinstance(existing_provenance, Mapping) else {}), **dict(provenance)}
    runtime["metadata"] = meta


def _delegated_export(source: str | Path | Mapping[str, Any]) -> dict[str, Any] | None:
    try: from mindie_profiler.sim_bridge import export_sim_case
    except ImportError: return None
    exported = export_sim_case(deepcopy(dict(source)) if isinstance(source, Mapping) else source)
    if isinstance(exported, CaseSpec): return exported.to_dict()
    if isinstance(exported, Mapping) and {"machine", "model", "workload"} <= set(exported): return deepcopy(dict(exported))
    if isinstance(exported, (str, Path)):
        data, _, _ = _read_source(exported)
        if isinstance(data, Mapping) and {"machine", "model", "workload"} <= set(data): return deepcopy(dict(data))
    if exported is None: return None
    raise ModelingError("sim_bridge_export_returned_unsupported_payload")


def load_case_from_profile(source: str | Path | Mapping[str, Any] | None = None, *,
                           from_profile: str | Path | Mapping[str, Any] | None = None,
                           **overrides: Any) -> dict[str, Any]:
    """Load a validated Case from an exported Case, Wave 5 capture, or C7 profile."""
    source = from_profile if source is None else source
    if source is None: raise ModelingError("profile_source_required")
    active = {k: v for k, v in overrides.items() if v is not None
              and k not in {"output", "command", "max_interventions", "from_profile", "case"}}
    delegated = None
    try: delegated = _delegated_export(source)
    except TimeoutError: raise
    except (ImportError, ValueError, ModelingError, RuntimeError, KeyError, TypeError): delegated = None
    data, path, files = {}, None, []
    try: data, path, files = _read_source(source)
    except ModelingError:
        if delegated is None: raise
    if data and isinstance(data, Mapping) and "_source_kind" not in data:
        data = dict(data, _source_kind="exported_case")
    delegated_meta = dict(delegated.get("runtime", {}).get("metadata") or {}) if delegated else {}
    observed, provenance, builtin_case, builtin_error, _ingredients = {}, {}, None, None, {}
    if data:
        try:
            observed, provenance = _load_observed_evidence(data, path)
            provenance["files"] = files
            _ingredients = _ingredients_from_data(data, path, model_override=active.get("model"),
                                                  hardware_override=active.get("hardware"), delegated=delegated)
            if _ingredients.get("base_case") is not None:
                builtin_case = (_derive_from_base_case(_ingredients["base_case"], active) if active
                                else dict(_ingredients["base_case"]))
            else:
                builtin_case = _compose_from_ingredients(_ingredients, active)
        except (ValueError, ModelingError) as exc:
            builtin_case, builtin_error = None, exc
            if delegated is not None and _unwrap_case(data) is None:
                try: builtin_case = _derive_from_base_case(delegated, active) if active else deepcopy(dict(delegated))
                except (ValueError, ModelingError) as del_exc: builtin_error = del_exc
    if builtin_case is None and (delegated is None or _unwrap_case(data) is not None):
        raise builtin_error or ModelingError(f"profile_case_synthesis_failed:{source}")
    delegated_observed = dict(delegated_meta.get("observed_profile") or {})
    if not _has_measured_observation(delegated_observed): delegated_observed = {}
    if builtin_case is None:
        case = _derive_from_base_case(delegated, active) if active else dict(delegated)
        observed = _merge_observed_profiles(delegated_observed, observed)
        provenance = {"schema": _PROVENANCE_SCHEMA, "source_kind": "delegated_sim_bridge",
                      "source_path": str(source) if not isinstance(source, Mapping) else None,
                      "source_sha256": None, "files": files,
                      **(dict(delegated_meta.get("profiler_provenance") or {}))}
    else:
        case = builtin_case
        runtime_meta = dict(case.setdefault("runtime", {}).get("metadata") or {})
        skip_meta = {"observed_profile", "profiler_provenance"} | (
            set(_SPEC_META_KEYS) if active.get("mtp_proposal_tokens") is not None and int(active["mtp_proposal_tokens"]) <= 0 else set()) | (
            {"execution_mode"} if active.get("graph_mode") is not None else set())
        for key, value in delegated_meta.items():
            if key not in skip_meta: runtime_meta.setdefault(key, value)
        case["runtime"]["metadata"] = runtime_meta
        observed = _merge_observed_profiles(delegated_observed, observed)
        provenance = {**(dict(delegated_meta.get("profiler_provenance") or {})), **provenance}
    if observed and _has_measured_observation(observed):
        src_c = _unwrap_case(data) or delegated
        if isinstance(src_c, Mapping):
            sm, sr, shw = dict(src_c.get("model") or {}), dict(src_c.get("runtime") or {}), dict(src_c.get("machine") or {})
            sp, sreqs, smeta = dict(sr.get("parallel") or {}), _normalized_requests(src_c.get("workload")), dict(sr.get("metadata") or {})
            sexec = smeta.get("execution_mode") or sr.get("graph_capture_mode") or sr.get("graph_mode")
            for ok, ov in (("model", sm.get("catalog_key") or sm.get("name")), ("layers", sm.get("layers")),
                           ("experts", sm.get("num_experts") or sm.get("n_routed_experts")),
                           ("top_k", sm.get("top_k") or sm.get("num_experts_per_tok")),
                           ("quantization", sm.get("quantization") or (sm.get("quant", {}).get("name") if isinstance(sm.get("quant"), Mapping) else None)),
                           ("hardware", shw.get("sku") or shw.get("name")), ("tp", sp.get("tp")), ("dp", sp.get("dp")), ("ep", sp.get("ep")), ("pp", sp.get("pp")),
                           ("concurrency", len(sreqs) if sreqs else None), ("context", sreqs[0].get("prompt_tokens") if sreqs else None),
                           ("execution_mode", sexec)):
                if ov is not None: observed.setdefault(ok, ov)
        elif _ingredients.get("graph_mode"):
            observed.setdefault("execution_mode", _ingredients.get("graph_mode"))
    case = _sanitize_case(case)
    _attach_profile_metadata(case, observed, provenance)
    parse_case(case)
    return case


__all__ = ["load_case_from_profile"]
