"""Guarded counterfactual catalog and deterministic intervention ranking.

[Role]
  Build fresh Case mutations across execution, HCCL expansion, MoE communication,
  shared-expert overlap, KV quantization, speculative decoding, and async
  scheduling; evaluate each through the causal engine and attach guardrails.

[Invariants]
  * Candidate Cases are deep copies; caller records remain untouched.
  * Every timing delta comes from ``analyze_network_efficiency`` on a parsed
    mutated Case; speculative verification uses the multi-token verify step.
  * Guardrail identifiers are stable public strings and never fabricated from
    model names.

[Timeouts & Locks]
  Bounded in-process graph evaluation only; no I/O, subprocess, or locks.

[Side-Effect Policy]
  Pure in-memory candidate construction.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

from .advise_core import (
    _Config,
    _analyze,
    _decode_step,
    _emitted_tokens,
    _is_a3_or_a5,
    _memory_facts,
    _proposal_tokens,
    _should_run_graph,
    _sla_metrics,
)
from .prediction.structure import parse_case
from .schema.case import Case as CaseSpec
from .schema.common import ModelingError


def _guardrail(identifier: str, message: str, *, severity: str = "warning",
               metric: str | None = None, value: Any = None,
               threshold: Any = None) -> dict[str, Any]:
    return {"id": identifier, "severity": severity, "blocking": severity in {"block", "error"},
            "message": message, "metric": metric, "value": value, "threshold": threshold}


def _merge_guardrails(target: list[dict[str, Any]], extra: list[dict[str, Any]]) -> None:
    existing = {item.get("id") for item in target}
    for item in extra:
        if item.get("id") not in existing:
            target.append(item)
            existing.add(item.get("id"))


def _runtime_meta(case: Mapping[str, Any]) -> dict[str, Any]:
    raw = case.get("runtime", {}).get("metadata") or {}
    return dict(raw) if isinstance(raw, Mapping) else dict(raw)


def _set_meta(case: dict[str, Any], key: str, value: Any) -> None:
    meta = _runtime_meta(case)
    meta[key] = deepcopy(value)
    case.setdefault("runtime", {})["metadata"] = meta


def _apply_graph_mode(case: dict[str, Any], mode: str) -> None:
    runtime = case.setdefault("runtime", {})
    runtime["graph_mode"] = mode
    runtime["graph_capture_mode"] = mode
    max_seqs = max(1, int(runtime.get("max_num_seqs", 1)))
    max_batched = max(1, int(runtime.get("max_num_batched_tokens", 8192)))
    if mode == "full":
        runtime["graph_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, max_seqs})
    else:
        runtime["piecewise_capture_keys"] = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                                    1024, 2048, 4096, max_batched})


def _execution_mutations(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime = base.get("runtime", {})
    graph_mode = str(runtime.get("graph_mode", "auto") or "auto").lower()
    current = ("none" if graph_mode in {"none", "off", "eager", "disabled"}
               else str(runtime.get("graph_capture_mode") or graph_mode or "none").lower())
    if current == "full" and not (runtime.get("graph_capture_keys") or runtime.get("graph_buckets")):
        current = "none"
    if current == "piecewise" and not runtime.get("piecewise_capture_keys"):
        current = "none"
    records = []
    for mode in ("full", "piecewise"):
        already = current == mode
        records.append({
            "intervention_id": "execution_mode.aclgraph_full" if mode == "full"
            else "execution_mode.piecewise",
            "category": "execution_mode", "variant": mode,
            "title": f"eager -> {mode} captured execution",
            "applicable": not already,
            "reason": "execution_mode_already_selected" if already else None,
            "mutations": {"execution_mode": mode, "runtime.graph_mode": mode,
                          "runtime.graph_capture_mode": mode},
            "apply": None if already else lambda case, mode=mode: _apply_graph_mode(case, mode),
        })
    return records


def _memory_guardrails(case: Mapping[str, Any], cfg: _Config) -> list[dict[str, Any]]:
    parsed = parse_case(case)
    memory, available, hbm, kv, workspace = _memory_facts(parsed, cfg)
    graph_pool = float(memory.policy.graph_pool_bytes or 0.0)
    capture = max(graph_pool, 2.0 * workspace + 0.02 * hbm)
    headroom = available - kv * cfg.sequences
    if capture > max(0.0, headroom):
        return [_guardrail("aclgraph_capture_memory_exceeds_hbm_headroom",
                           "ACLGraph capture pool exceeds remaining KV headroom",
                           severity="block", metric="capture_vs_headroom_bytes",
                           value=round(capture, 3), threshold=round(max(0.0, headroom), 3))]
    return []


def _hccl_mutations(base: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime = base.get("runtime", {})
    kernels = runtime.get("kernels") or {}
    declared = dict(kernels.get("hccl") or {}) if isinstance(kernels, Mapping) else {}
    comm = float(baseline.get("analysis", {}).get("breakdown_by_kind", {})
                 .get("communication", {}).get("total_ms", 0.0) or 0.0)
    parallel = runtime.get("parallel", {})
    axes = [axis for axis in ("tp", "etp", "pcp") if int(parallel.get(axis, 1) or 1) >= 2]
    if not declared and comm <= 0.0:
        return [{"intervention_id": "hccl_op_expansion_mode.aiv", "category": "hccl_op_expansion_mode",
                 "title": "AICPU -> AIV HCCL expansion", "applicable": False,
                 "reason": "no_communication_or_hccl_expansion_present", "mutations": {}, "apply": None}]
    if not declared and not axes:
        return [{"intervention_id": "hccl_op_expansion_mode.aiv", "category": "hccl_op_expansion_mode",
                 "title": "AICPU -> AIV HCCL expansion", "applicable": False,
                 "reason": "no_collective_group_axis_present", "mutations": {}, "apply": None}]
    machine = base.get("machine", {})
    device = (machine.get("devices") or [machine.get("device") or {}])[0]
    resources = device.get("resource_units", {}) if isinstance(device, Mapping) else {}
    block_limit = max(2, int(resources.get("vector", 0) or device.get("compute_units", 1) or 1))
    max_rank = max([int(parallel.get(axis, 1) or 1) for axis in axes] or [1])
    mutation: dict[str, Any] = {}
    for collective, raw in declared.items():
        if isinstance(raw, Mapping) and str(raw.get("expansion", "")).startswith("AIC"):
            mutation[collective] = {**deepcopy(dict(raw)), "expansion": "AIV",
                                    "block_limit": int(raw.get("block_limit") or block_limit)}
    if not mutation and axes:
        mutation["allreduce"] = {"expansion": "AIV",
                                 "template": "mesh_oneshot" if max_rank <= 4 else "mesh_twoshot",
                                 "axes": axes, "block_limit": block_limit}

    def apply(case: dict[str, Any]) -> None:
        runtime = case.setdefault("runtime", {})
        current = dict(runtime.get("kernels") or {})
        hccl = dict(current.get("hccl") or {})
        for collective, raw in mutation.items():
            hccl[collective] = {**deepcopy(dict(raw)), "axes": raw.get("axes") or axes or ["tp"]}
        current["hccl"] = hccl
        runtime["kernels"] = current

    return [{"intervention_id": "hccl_op_expansion_mode.aiv", "category": "hccl_op_expansion_mode",
             "variant": "aiv", "title": "AICPU -> AIV collective expansion",
             "applicable": True, "reason": None,
             "mutations": {"hccl_op_expansion_mode": "aiv",
                           "runtime.kernels.hccl": mutation}, "apply": apply}]


def _moe_mutations(base: Mapping[str, Any], case_spec: CaseSpec) -> list[dict[str, Any]]:
    has_moe = bool(int(getattr(case_spec.model, "num_experts", 0) or 0)) or any(
        int(getattr(layer, "num_experts", 0) or 0) > 0 or getattr(layer, "ffn", "") == "moe"
        for layer in case_spec.model.layer_plans)
    if not has_moe:
        return [{"intervention_id": "moe_comm_fusion.unavailable", "category": "moe_comm_fusion",
                 "variant": "unavailable", "title": "MoE communication fusion",
                 "applicable": False, "reason": "no_routed_moe", "mutations": {}, "apply": None}]
    records = []
    if str(case_spec.runtime.moe_recipe) != "mc2":
        records.append({"intervention_id": "moe_comm_fusion.fused_mc2", "category": "moe_comm_fusion",
                        "variant": "enable_fused_mc2", "title": "alltoall -> fused MC2",
                        "applicable": True, "reason": None,
                        "mutations": {"enable_fused_mc2": True, "runtime.moe_recipe": "mc2"},
                        "apply": lambda case: case.setdefault("runtime", {}).__setitem__("moe_recipe", "mc2")})
    p = case_spec.runtime.parallel
    if p.ep == 1 and p.dp == 1 and p.cp == 1 and p.pcp == 1:
        records.append({"intervention_id": "moe_comm_fusion.tp_join_reduce", "category": "moe_comm_fusion",
                        "variant": "all2all_backend", "all2all_backend": "tp_join_reduce",
                        "title": "alltoall -> tp_join_reduce", "applicable": True, "reason": None,
                        "mutations": {"all2all_backend": "tp_join_reduce",
                                      "runtime.moe_recipe": "tp_join_reduce"},
                        "apply": lambda case: case.setdefault("runtime", {}).__setitem__(
                            "moe_recipe", "tp_join_reduce")})
    else:
        records.append({"intervention_id": "moe_comm_fusion.deepep_high_throughput",
                        "category": "moe_comm_fusion", "variant": "all2all_backend",
                        "all2all_backend": "deepep_high_throughput",
                        "title": "DeepEP high-throughput AllToAll backend", "applicable": True,
                        "reason": None, "mutations": {"all2all_backend": "deepep_high_throughput",
                                                       "runtime.deepep_high_throughput": True},
                        "apply": lambda case: case.setdefault("runtime", {}).__setitem__(
                            "deepep_high_throughput", True)})
    return records


def _shared_expert_mutations(case_spec: CaseSpec) -> list[dict[str, Any]]:
    shared = int(getattr(case_spec.model, "num_shared_experts", 0) or 0) or any(
        int(getattr(layer, "num_shared_experts", 0) or 0) > 0 for layer in case_spec.model.layer_plans)
    if not shared or case_spec.runtime.multistream_overlap_shared_expert:
        return [{"intervention_id": "multistream_overlap_shared_expert.unavailable",
                 "category": "multistream_overlap_shared_expert", "variant": "unavailable",
                 "title": "Shared-expert overlap", "applicable": False,
                 "reason": "no_shared_experts" if not shared else "already_enabled",
                 "mutations": {}, "apply": None}]
    return [{"intervention_id": "multistream_overlap_shared_expert.enable",
             "category": "multistream_overlap_shared_expert", "variant": "enable",
             "title": "Overlap shared-expert projections with routed dispatch",
             "applicable": True, "reason": None,
             "mutations": {"multistream_overlap_shared_expert": True,
                           "runtime.multistream_overlap_shared_expert": True},
             "apply": lambda case: case.setdefault("runtime", {}).__setitem__(
                 "multistream_overlap_shared_expert", True)}]


def _apply_kv_quant(case: dict[str, Any], target: str) -> None:
    model = case.setdefault("model", {})
    model["kv_dtype"] = target
    if isinstance(model.get("quant"), Mapping):
        model["quant"] = {**dict(model["quant"]), "kv_dtype": target}


def _kv_quant_mutations(case_spec: CaseSpec, baseline: Mapping[str, Any],
                        cfg: _Config) -> list[dict[str, Any]]:
    quant_kv = getattr(getattr(case_spec.model, "quant", None), "kv_dtype", None)
    current = str(quant_kv or case_spec.model.kv_dtype or "bf16").lower()
    if current not in {"bf16", "bfloat16", "fp16", "float16"}:
        return [{"intervention_id": "kv_quant.unavailable", "category": "kv_quant",
                 "variant": "unavailable", "title": "Low-bit KV cache",
                 "applicable": False, "reason": "kv_cache_already_low_bit",
                 "mutations": {}, "apply": None}]
    supports_fp8 = any("fp8" in str(key).lower() for key in dict(case_spec.machine.dies[0].compute_peaks))
    target = "fp8_e4m3" if supports_fp8 else "int8"
    has_mla = (getattr(case_spec.model, "mla", None) is not None
               or any(str(getattr(layer, "attention", "")) in {"mla", "mqa", "mqsmla"}
                      for layer in case_spec.model.layer_plans))
    attention = float(baseline["attribution"]["stage"].get("attention_share", 0.0) or 0.0)
    kv = float(baseline["attribution"]["capacity"].get("kv_bytes_per_seq", 0.0) or 0.0)
    available = float(baseline["attribution"]["capacity"].get(
        "available_kv_capacity_bytes_per_die", 1.0) or 1.0)
    if kv <= 0.0 or (attention < 0.10 and kv < 0.05 * max(available, 1.0)
                     and baseline["dominant_bottleneck"] != "memory_kv_bound"):
        return [{"intervention_id": "kv_quant.unavailable", "category": "kv_quant",
                 "variant": "unavailable", "title": "Low-bit KV cache",
                 "applicable": False, "reason": "kv_cache_not_a_bottleneck",
                 "mutations": {}, "apply": None}]
    return [{"intervention_id": f"kv_quant.{target}", "category": "kv_quant",
             "variant": target, "title": f"KV cache {current} -> {target}",
             "applicable": True, "reason": None,
             "mutations": {"kv_quant": target, "model.kv_dtype": target,
                           "model.quant.kv_dtype": target},
             "apply": lambda case, target=target: _apply_kv_quant(case, target),
             "accuracy_risk": bool(cfg.context >= 131072 or has_mla)}]


def _speculative_mutations(case_spec: CaseSpec, baseline: Mapping[str, Any],
                           cfg: _Config) -> list[dict[str, Any]]:
    auxiliary = case_spec.model.auxiliary
    mtp = getattr(auxiliary, "mtp", None) if auxiliary is not None else None
    dspark = getattr(auxiliary, "dspark", None) if auxiliary is not None else None
    mode = "mtp" if mtp is not None else "dspark" if dspark is not None else None
    if mode == "mtp":
        proposal = int(getattr(mtp, "proposal_tokens", 0) or 0)
    elif mode == "dspark":
        try:
            proposal = int(dspark.step_spec().draft_len)
        except (AttributeError, TypeError, ValueError):
            proposal = 2
    else:
        proposal = 2
    proposal = max(1, proposal)
    acceptance = min(1.8, max(1.6, 1.0 + 0.4 * proposal))
    guardrails = []
    already = _proposal_tokens(case_spec, cfg.output_tokens) > 0 and cfg.tokens > 1
    applicable = (not already and cfg.phase == "decode" and cfg.sequences <= 16
                  and baseline["dominant_bottleneck"] in {"memory_kv_bound", "launch_host_bound"})
    if mode is None:
        guardrails.append(_guardrail("requires_draft_checkpoint_weights",
                                     "Speculative verification requires MTP/DSpark draft checkpoint weights",
                                     severity="block", metric="draft_structure", value="absent",
                                     threshold="mtp_or_dspark"))
        applicable = False
    if cfg.sequences > 16 or baseline["dominant_bottleneck"] == "compute_bound":
        guardrails.append(_guardrail("speculative_compute_bound_at_high_batch",
                                     "Speculative verification is compute-bound at high batch",
                                     metric="sequences", value=cfg.sequences, threshold=16))
        applicable = False
    if mode is None:
        return [{"intervention_id": "speculative_mode.unavailable", "category": "speculative_mode",
                 "title": "Speculative decode (draft weights unavailable)", "applicable": False,
                 "reason": "requires_draft_checkpoint_weights", "guardrails": guardrails,
                 "mutations": {}, "apply": None}]

    def apply(case: dict[str, Any], mode=mode, proposal=proposal, acceptance=acceptance) -> None:
        runtime = case.setdefault("runtime", {})
        runtime["mtp_acceptance_length"] = max(
            float(runtime.get("mtp_acceptance_length", 1.0) or 1.0), acceptance)
        _set_meta(case, "draft_mode", mode)
        _set_meta(case, "draft_proposal_tokens", proposal)

    return [{"intervention_id": f"speculative_mode.{mode}", "category": "speculative_mode",
             "variant": mode, "title": f"{mode} speculative verify", "applicable": applicable,
             "reason": ("speculative_mode_already_enabled" if already
                        else None if applicable else "not_memory_or_launch_bound_at_low_batch"),
             "guardrails": guardrails, "mutations": {"speculative_mode": mode,
                                                       "runtime.metadata.draft_mode": mode,
                                                       "runtime.metadata.draft_proposal_tokens": proposal},
             "apply": None if already else apply, "proposal": proposal, "acceptance": acceptance}]


def _async_mutations(case_spec: CaseSpec, baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    if case_spec.runtime.async_scheduling:
        return [{"intervention_id": "async_scheduling.unavailable", "category": "async_scheduling",
                 "variant": "unavailable", "title": "Async host scheduling", "applicable": False,
                 "reason": "async_scheduling_already_enabled", "mutations": {}, "apply": None,
                 "guardrails": []}]
    guardrails = []
    if baseline["b_vram_max_async"] < baseline["b_vram_max_sync"]:
        guardrails.append(_guardrail("async_scheduling_delayed_kv_release_reduces_max_batch",
                                     "Delayed KV release lowers async maximum batch",
                                     severity="warning", metric="b_vram_max_async",
                                     value=int(baseline["b_vram_max_async"]),
                                     threshold=int(baseline["b_vram_max_sync"])))
    return [{"intervention_id": "async_scheduling.enable", "category": "async_scheduling",
             "variant": "enable", "title": "Overlap host scheduling with device execution",
             "applicable": True,
             "reason": None, "guardrails": guardrails,
             "mutations": {"async_scheduling": True, "runtime.async_scheduling": True},
             "apply": lambda case: case.setdefault("runtime", {}).__setitem__("async_scheduling", True)}]


def candidate_specs(case_spec: CaseSpec, baseline: Mapping[str, Any], cfg: _Config,
                    base: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = _execution_mutations(base)
    specs.extend(_hccl_mutations(base, baseline))
    specs.extend(_moe_mutations(base, case_spec))
    specs.extend(_shared_expert_mutations(case_spec))
    specs.extend(_kv_quant_mutations(case_spec, baseline, cfg))
    specs.extend(_speculative_mutations(case_spec, baseline, cfg))
    specs.extend(_async_mutations(case_spec, baseline))
    return specs


def _pre_guardrails(record: Mapping[str, Any], case_spec: CaseSpec,
                    baseline: Mapping[str, Any], cfg: _Config) -> list[dict[str, Any]]:
    guardrails: list[dict[str, Any]] = []
    category = record.get("category")
    if category == "hccl_op_expansion_mode" and _is_a3_or_a5(case_spec):
        vector_share = float(baseline["attribution"]["efficiency"].get(
            "breakdown_by_kind", {}).get("vector", {}).get("share_of_operator_sum", 0.0) or 0.0)
        vector_mfu = float(baseline["attribution"]["efficiency"].get("vec_mfu", 0.0) or 0.0)
        if vector_share >= 0.20 or vector_mfu >= 0.35:
            guardrails.append(_guardrail("aiv_expansion_vector_contention_risk",
                                         "AIV collective expansion contends with heavy Vector work",
                                         severity="warning", metric="vector_share",
                                         value=round(vector_share, 9), threshold=0.20))
    elif category == "moe_comm_fusion" and record.get("intervention_id", "").endswith(
            "deepep_high_throughput"):
        p = case_spec.runtime.parallel
        if len(case_spec.machine.dies) > int(case_spec.machine.ranks_per_node):
            guardrails.append(_guardrail("cross_node_all2all_requires_deepep_transport",
                                         "Cross-node AllToAll requires the DeepEP transport path",
                                         severity="warning", metric="cross_node", value=True,
                                         threshold=True))
        if case_spec.runtime.graph_capture_mode in {"full", "piecewise"}:
            guardrails.append(_guardrail("full_graph_incompatible_with_deepep_high_throughput",
                                         "DeepEP high-throughput mode disables graph capture",
                                         severity="warning", metric="graph_capture_mode",
                                         value=case_spec.runtime.graph_capture_mode,
                                         threshold="none"))
    elif (category == "multistream_overlap_shared_expert"
          and record.get("intervention_id", "").endswith(".enable")):
        mbu = float(baseline["attribution"]["efficiency"].get("mbu", 0.0) or 0.0)
        if mbu >= 0.75:
            guardrails.append(_guardrail("shared_expert_mte_bandwidth_contention",
                                         "Shared-expert overlap contends for near-saturated MTE/HBM bandwidth",
                                         severity="warning", metric="mbu", value=round(mbu, 9),
                                         threshold=0.75))
    elif (category == "kv_quant" and record.get("accuracy_risk")
          and record.get("intervention_id", "") != "kv_quant.unavailable"):
        guardrails.append(_guardrail("kv_quant_calibration_accuracy_guardrail",
                                     "Low-bit KV cache requires calibration for long context or MLA",
                                     severity="warning", metric="context", value=cfg.context,
                                     threshold=131072))
    elif category == "speculative_mode":
        guardrails.extend(deepcopy(list(record.get("guardrails", ()))))
    elif category == "async_scheduling":
        guardrails.extend(deepcopy(list(record.get("guardrails", ()))))
    return guardrails


def _evaluate_intervention(spec: Mapping[str, Any], base: Mapping[str, Any],
                           baseline: Mapping[str, Any], case_spec: CaseSpec,
                           cfg: _Config) -> dict[str, Any]:
    record = {key: deepcopy(value) for key, value in spec.items()
              if key not in {"apply"}}
    record.setdefault("guardrails", [])
    record.setdefault("applicable", True)
    record.setdefault("reason", None)
    record["tpot_improvement_pct"] = 0.0
    record["sla_goodput_gain_pct"] = 0.0
    record["ttft_improvement_pct"] = 0.0
    record["step_improvement_pct"] = 0.0
    record["applicable_and_recommended"] = False
    _merge_guardrails(record["guardrails"], _pre_guardrails(record, case_spec, baseline, cfg))
    record["risk"] = ("high" if any(item.get("blocking") for item in record["guardrails"])
                      else "medium" if record["guardrails"] else "low")
    if not record["applicable"] or spec.get("apply") is None:
        return record
    case = deepcopy(dict(base))
    try:
        spec["apply"](case)
        parsed = parse_case(case)
    except (ValueError, ModelingError) as exc:
        record["applicable"] = False
        record["reason"] = f"mutation_not_selectable:{exc}"
        record["risk"] = ("high" if any(item.get("blocking") for item in record["guardrails"])
                          else "medium" if record["guardrails"] else "low")
        return record
    eval_cfg = replace(cfg, tokens=1 + int(spec.get("proposal", 0) or 0)) if record[
        "category"] == "speculative_mode" else cfg
    run_graph = _should_run_graph(parsed, eval_cfg.context)
    try:
        analysis = _analyze(parsed, eval_cfg, run_graph=run_graph)
        step = float(analysis["graph_duration_ms"])
        emitted = _emitted_tokens(parsed, eval_cfg)
        tpot = step / emitted if eval_cfg.phase == "decode" else _decode_step(
            parsed, eval_cfg, sequences=eval_cfg.sequences) / emitted
        if eval_cfg.phase == "prefill":
            ttft = step
        else:
            prefill_tokens = max(1, eval_cfg.context - eval_cfg.prefix_tokens)
            prefill = _analyze(parsed, eval_cfg, run_graph=True, phase="prefill",
                               tokens=prefill_tokens, context=eval_cfg.prefix_tokens,
                               sequences=eval_cfg.sequences)
            ttft = float(prefill["graph_duration_ms"])
        sla = _sla_metrics(parsed, eval_cfg, step_ms=step, ttft_ms=ttft)
    except (ValueError, ModelingError) as exc:
        record["applicable"] = False
        record["reason"] = f"counterfactual_not_selectable:{exc}"
        record["risk"] = ("high" if any(item.get("blocking") for item in record["guardrails"])
                          else "medium" if record["guardrails"] else "low")
        return record
    record["predicted"] = {
        "step_duration_ms": step, "tpot_p50_ms": tpot, "ttft_p50_ms": ttft,
        "sla_goodput_tok_s": sla["sla_goodput_tok_s"],
        "b_vram_max_sync": sla["b_vram_max_sync"],
        "b_vram_max_async": sla["b_vram_max_async"],
    }
    record["tpot_improvement_pct"] = round(
        (float(baseline["tpot_p50_ms"]) - tpot) / max(float(baseline["tpot_p50_ms"]), 1e-12) * 100.0, 9)
    record["step_improvement_pct"] = round(
        (float(baseline["step_duration_ms"]) - step) / max(float(baseline["step_duration_ms"]), 1e-12) * 100.0, 9)
    record["ttft_improvement_pct"] = round(
        (float(baseline["ttft_p50_ms"]) - ttft) / max(float(baseline["ttft_p50_ms"]), 1e-12) * 100.0, 9)
    base_tps = float(baseline["sla_goodput_tok_s"])
    record["sla_goodput_gain_pct"] = round(
        ((float(sla["sla_goodput_tok_s"]) / base_tps) - 1.0) * 100.0, 9) if base_tps > 0.0 else 0.0
    record["b_vram_max_delta"] = int(sla["b_vram_max_sync"]) - int(baseline["b_vram_max_sync"])
    eff_b_vram = int(sla["b_vram_max_async"] if parsed.runtime.async_scheduling else sla["b_vram_max_sync"])
    if eff_b_vram < eval_cfg.sequences:
        _merge_guardrails(record["guardrails"], [_guardrail(
            "candidate_exceeds_vram_capacity",
            "Candidate configuration exceeds available HBM KV capacity at requested concurrency",
            severity="block", metric="b_vram_max", value=eff_b_vram, threshold=eval_cfg.sequences)])
    if record["category"] == "execution_mode":
        _merge_guardrails(record["guardrails"], _memory_guardrails(case, eval_cfg))
        if parsed.runtime.deepep_high_throughput:
            _merge_guardrails(record["guardrails"], [_guardrail(
                "deepep_high_throughput_incompatible_with_graph_capture",
                "DeepEP high-throughput mode disables graph capture", severity="block",
                metric="deepep_high_throughput", value=True, threshold=False)])
    blocking = any(item.get("blocking") for item in record["guardrails"])
    improvement = (record["tpot_improvement_pct"] > 0.0
                   or record["sla_goodput_gain_pct"] > 0.0
                   or record["ttft_improvement_pct"] > 0.0)
    record["applicable_and_recommended"] = bool(record["applicable"] and not blocking and improvement)
    record["risk"] = "high" if blocking else "medium" if record["guardrails"] else "low"
    return record


def rank_key(record: Mapping[str, Any]) -> tuple:
    return (
        0 if record.get("applicable_and_recommended") else 1,
        -float(record.get("tpot_improvement_pct", 0.0) or 0.0),
        -float(record.get("sla_goodput_gain_pct", 0.0) or 0.0),
        -float(record.get("ttft_improvement_pct", 0.0) or 0.0),
        str(record.get("intervention_id", "")),
    )


__all__ = ["candidate_specs", "rank_key"]
