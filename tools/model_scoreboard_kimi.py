"""Reproduce descriptor-conditioned Kimi native-family comparisons.

[Role] Stream observations, execute independent scenarios, then compare totals.
[Invariants] Prediction workers receive descriptors and declarations, never
  measured durations or occurrence counts. Capture activity is not step latency.
  Unknown KDA launch stages never acquire a fabricated complete-call residual.
[Timeouts & Locks] One worker at a time; at most 30 s each and 90 s in total.
  Capture parsing is streaming; metadata reads are bounded to 50 MiB.
[Side-Effect Policy] No devices or calibration; write only requested artifacts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

if __package__:
    from .kimi_call_scopes import SITU_INPUT, kda_stage_ledger, situ_reference
    from .model_convergence import _runtime_identity, worker_usage
else:
    from kimi_call_scopes import SITU_INPUT, kda_stage_ledger, situ_reference
    from model_convergence import _runtime_identity, worker_usage


MODES = ("mla", "grouped", "kda", "situ")
KINDS = {"mla": "MlaPrologV3", "grouped": "GroupedMatmul", "kda": "ChunkKdaFwd",
         "situ": "GroupedMatmulSituQuant"}
DESCRIPTORS = (
    "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
    "input_formats", "output_formats",
)
QUANT = {
    "name": "descriptor_conditioned_mxfp8",
    "weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3",
    "expert_weight_format": "mxfp4", "weight_block": [1, 32], "expert_block": 32,
    "scale_format": "fp8_e8m0", "scale_bytes": 1, "activation_group_size": 32,
    "activation_scale_format": "fp8_e8m0", "activation_scale_pack": 2,
    "kv_dtype": "bf16", "exclusions": ["mla_absorb"],
}
COMMON_LIMITS = [
    "Shape/dtype conditioning is explicit; durations never select the recipe, rates, work, or parameters.",
    "Activity sum over retained call count is not a model-step span or additive critical-path contribution.",
    "Independent installed source/binary and native mapping remain unknown; no accuracy certification.",
]


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError(f"kimi_metadata_exceeds_50_mib:{path}")
    return json.loads(path.read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source(path, *, deadline=None):
    path, digest = Path(path).resolve(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("kimi_source_hash_budget_exceeded")
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def producer():
    return {"driver": source(__file__), "argv": sys.argv[1:]}


def observe(capture, *, max_rows=1000000, timeout=25, max_groups=4096):
    """Keep scalar durations for selected bounded rows, never the raw capture."""
    from mindie_sim.observations import kernels
    from mindie_profiler import normalized, sources_csv

    started = time.monotonic()
    identity = source(capture, deadline=started + timeout)
    groups, counts, total = {}, Counter(), 0
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("kimi_observation_budget_exceeded")
    for row in kernels.kernel_rows(capture, max_rows=max_rows, timeout_s=remaining,
                                   include_zero_duration=True):
        total += 1
        if row.kind not in KINDS.values():
            continue
        descriptors = {key: getattr(row, key) for key in DESCRIPTORS}
        key = json.dumps([row.kind, descriptors], sort_keys=True)
        if key not in groups:
            if len(groups) >= max_groups:
                raise ValueError("kimi_descriptor_group_limit_exceeded")
            groups[key] = {
                "id": f"group{len(groups)}", "kind": row.kind,
                "descriptors": descriptors, "first_source_line": row.line,
                "first_name": row.name, "durations_ns": [],
            }
        groups[key]["durations_ns"].append(row.duration_ns)
        counts[row.kind] += 1
    rows = []
    for group in groups.values():
        durations = group.pop("durations_ns")
        group.update(
            count=len(durations), activity_sum_ms=sum(durations) / 1e6,
            mean_ms=statistics.mean(durations) / 1e6,
            p50_ms=statistics.median(durations) / 1e6,
            min_ms=min(durations) / 1e6, max_ms=max(durations) / 1e6,
        )
        rows.append(group)
    return {
        "schema": "kimi_native_family_descriptor_aggregate/1",
        "source": identity, "raw_rows": total, "counts": dict(counts),
        "groups": sorted(rows, key=lambda row: -row["activity_sum_ms"]),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_mib": worker_usage().get("peak_rss_mib"),
        "producer": producer(),
        "parser_sources": [source(module.__file__)
                           for module in (kernels, normalized, sources_csv)],
        "limits": {"max_rows": max_rows, "timeout_seconds": timeout,
                   "max_descriptor_groups": max_groups},
        "limitations": [
            "Whole mixed capture; descriptor equivalence is not layer or complete-call occurrence binding.",
            "ChunkKdaFwd native launch stage is not inferred from common descriptors.",
            "No timings populate prediction inputs.",
        ],
    }


def prediction_packet(mode, observation, case, case_source, *, context, kda_tokens,
                      matrix_recipe="steady_matmul", kda_implementation=None):
    """Extract only tensor declarations; discard timings and occurrence counts."""
    return {
        "mode": mode, "case": case, "case_source": case_source,
        "quant": QUANT, "context": context, "kda_tokens": kda_tokens,
        "matrix_recipe": matrix_recipe,
        "kda_implementation": kda_implementation,
        "groups": [{"id": row["id"], "kind": row["kind"],
                    "descriptors": row["descriptors"]}
                   for row in observation["groups"] if row["kind"] == KINDS[mode]],
    }


def grouped_predictions(case, groups, quant, *, matrix_recipe):
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.prediction.operator_analysis import operator_costs, physical_id
    from mindie_sim.prediction.program import RecipeBook

    ops, contracts = [], []
    for group in groups:
        desc = group["descriptors"]
        m, k = desc["input_shapes"][0]
        n, count = desc["output_shapes"][0][-1], desc["input_shapes"][1][0]
        weight = desc["input_dtypes"][1]
        if (desc["input_dtypes"][0] != "DT_FLOAT8_E4M3FN"
                or desc["output_dtypes"][0] != "DT_BF16"
                or weight not in {"DT_FLOAT4_E2M1", "DT_FLOAT8_E4M3FN"}):
            raise ValueError(f"unsupported_grouped_operand_contract:{group['id']}")
        fmt = "mxfp4" if weight == "DT_FLOAT4_E2M1" else "fp8_e4m3"
        op = gemm_op(
            op_id=group["id"], deps=(), stage=0, layer=-1, requests=(),
            m=m, n=n, k=k, quant=replace(quant, weight_format=fmt, expert_weight_format=fmt),
            category="standalone_matrix", phase="descriptor_conditioned", family="gemm",
            recipe="matrix_envelope" if count > 1 else "matrix_mapping",
            weight_count=count, activation_dtype="fp8_e4m3", output_dtype="bf16",
        )
        ops.append(op)
        contracts.append((count, fmt))
    book = RecipeBook.from_mapping({"recipes": {
        "matrix_envelope": {"kind": "envelope"},
        "matrix_mapping": {"kind": matrix_recipe},
    }})
    costs = {row["physical_id"]: row
             for row in operator_costs(case, ops, recipes=book)["operators"]}
    rows = []
    for group, op, (count, fmt) in zip(groups, ops, contracts):
        cost = costs[physical_id(op)]
        rows.append({
            "id": group["id"], "native_kind": group["kind"],
            "scenario_descriptors": group["descriptors"],
            "prediction": {
                "per_call_ms": cost["duration_ms"], "program": cost["program"],
                "boundary": "generic_matrix_resource_reference", "matrix_groups": count,
                "compute_dtype": op.dtype, "basis": cost.get("basis"),
                "logical_work": {key: getattr(op, key)
                                 for key in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes")},
                "operand_contract": dict(op.metadata), "serviced": cost["serviced"],
            },
            "comparison_level": "descriptor_conditioned_matrix_family_reference",
            "status": "conditional_reference",
            "limitations": ([
                f"Group list values are absent; the aggregate envelope prices total FLOPs and all {count} weight matrices without inventing equal rows or a native grouped schedule.",
                "MXFP4 uses the existing explicit BF16 dequant/accumulate resource path; native FP8xFP4 instruction mapping is not reconstructed.",
            ] if count > 1 else [
                f"One-group native GroupedMatmul uses the explicit generic {matrix_recipe} mapping.",
            ]) + COMMON_LIMITS,
        })
    return rows


def compact_execution(execution):
    result = {key: execution[key] for key in (
        "completed_tasks", "live_tasks", "live_programs", "peak_live_tasks",
        "work", "rank_model")}
    result["serviced"] = {key: value for key, value in execution["serviced"].items() if value}
    return result


def situ_predictions(case, groups, quant):
    """Execute each descriptor's complete gate-up/SiTU/output-quantizer DAG."""
    from mindie_sim.prediction.graph_analysis import graph_costs

    rows = []
    for group in groups:
        reference = situ_reference(case, group["descriptors"], quant)
        ops = reference["ops"]
        cost = graph_costs(
            case, ops, recipes=reference["recipes"],
            external_inputs={SITU_INPUT: 0}, isolated=False,
            include_submission=False, trace_limit=0)
        execution = compact_execution(cost["execution"])
        rows.append({
            "id": group["id"], "native_kind": group["kind"],
            "scenario_descriptors": group["descriptors"],
            "prediction": {
                "per_call_ms": cost["duration_ms"],
                "boundary": "declared_gate_up_situ_output_quantization_dag",
                "boundary_ledger": reference["ledger"],
                "logical_work": {key: math.fsum(getattr(op, key) for op in ops)
                                 for key in ("flops", "vector_ops", "hbm_bytes",
                                             "mte2_bytes", "mte3_bytes")},
                "serviced": execution["serviced"], "execution": execution,
            },
            "comparison_level": "descriptor_conditioned_situ_semantic_reference",
            "status": "conditional_reference",
            "limitations": reference["ledger"]["limitations"] + COMMON_LIMITS,
        })
    return rows


def kda_stage_predictions(case, ops, recipes, ledger):
    """Price disjoint semantic projections; historical launch stages stay unknown."""
    from mindie_sim.prediction.graph_analysis import graph_costs

    prefixes = {
        "GATE_PREPARE": ("delta_gates", "chunk_products", "triangular_solve"),
        "POST_WU": ("wy_transform",), "FWD_H": ("recurrent_scan",),
        "FINALIZE": ("recurrent_output",),
        "FULL": tuple(op.recipe for op in ops),
    }
    rows, selected_ids = [], []
    for stage in ledger["stage_ledger"]:
        selected = tuple(op for op in ops if (
            dict(op.metadata).get("source_phase") == stage["name"]
            or (op.family == "recurrent_scan" if stage["name"] == "FWD_H"
                else op.recipe.startswith(prefixes[stage["name"]]))))
        local = {op.op_id for op in selected}
        external = {dep: 0 for op in selected for dep in op.deps if dep not in local}
        cost = graph_costs(case, {0: selected}, recipes=recipes, external_inputs=external,
                           include_submission=False, isolated=False, trace_limit=0)
        work = {field: sum(getattr(op, field) for op in selected) for field in (
            "flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")}
        source_selected = any(op.source_boundary for op in selected)
        rows.append({
            "stage": stage["name"], "semantic_member_ids": sorted(local),
            "semantic_work": work, "predicted_semantic_projection_ms": cost["duration_ms"],
            "boundary": (f"source_selected_{stage['name']}_phase; external_inputs_ready"
                         if source_selected else
                         "isolated_semantic_projection; external_inputs_ready"),
            "observed_stage_ms": None, "observed_stage_launches": None,
            "cumulative_gap_ms": None,
            "native_mapping_status": ("source_phase_not_capture_binary_identity"
                                     if source_selected else stage["mapping_status"]),
            "source_recognized_product_flops": (
                [{"name": "selected_source_matrix_work", "dtype": dtype,
                  "flops": sum(dict(op.matrix_flops).get(dtype, 0) for op in selected)}
                 for dtype in sorted({dtype for op in selected for dtype, _ in op.matrix_flops})]
                if source_selected else stage["recognized_product_flops"]),
            "source_unresolved_work": (
                [dict(op.metadata)["timing_approximation"] for op in selected
                 if "timing_approximation" in dict(op.metadata)]
                or ["flag/queue/synchronization duration", "internal row pipeline overlap",
                    "odd V1 tail AIV imbalance", "L1-tail-fill service duration"]
                if source_selected else stage["unresolved_work"]),
            "execution": compact_execution(cost["execution"]),
        })
        selected_ids.extend(local)
    if sorted(selected_ids) != sorted(op.op_id for op in ops):
        raise ValueError("kda_stage_projection_must_partition_complete_semantic_core")
    return rows


def semantic_prediction(case, group, quant, *, mode, context, kda_tokens, matrix_recipe,
                        kda_implementation=None):
    from mindie_sim.prediction.attention import AttentionPolicy
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.mla import mla
    from mindie_sim.prediction.native_fusion import _groups
    from mindie_sim.prediction.operator_analysis import operator_costs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.prediction.recurrent import linear
    from mindie_sim.prediction.transforms import lower_implementations
    from mindie_sim.schema.workload import RequestView

    is_mla = mode == "mla"
    kernels = deepcopy(dict(case.runtime.kernels))
    lowerings = kernels.setdefault("lowerings", {})
    if is_mla:
        lowerings.update(mla_prolog="vllm_ascend_mla_prolog_v3", native_fusion={
            "implementation": "declared_native_fusion",
            "options": {"roles": ["mla_prolog"], "traffic_profile": "preserve_workspaces"},
        })
    elif kda_implementation is not None:
        lowerings["kda"] = kda_implementation
    else:
        lowerings.setdefault("kda", "vllm_ascend_kda_in_kernel_gates")
    kernels.setdefault("programs", {})["matmul"] = {"kind": matrix_recipe}
    case = replace(case, runtime=replace(case.runtime, kernels=kernels),
                   model=replace(case.model, quant=quant) if is_mla else case.model)
    plan = next(plan for plan in case.model.layer_plans if plan.attention == mode)
    tokens, phase = (1, "decode") if is_mla else (kda_tokens, "prefill")
    descriptors = group["descriptors"]
    shape = descriptors["input_shapes"][0]
    if (is_mla and shape != [1, 1, case.model.hidden_size]
            or not is_mla and (len(shape) != 4 or shape[:2] != [1, tokens])):
        raise ValueError(f"semantic_probe_descriptor_geometry_mismatch:{group['id']}")
    if is_mla:
        heads = plan.num_heads // case.runtime.parallel.tp
        expected = [
            [case.model.hidden_size, plan.q_lora_rank],
            [plan.q_lora_rank, heads * (plan.qk_nope_head_dim + plan.qk_rope_head_dim)],
            [heads, plan.qk_nope_head_dim, plan.kv_lora_rank],
            [case.model.hidden_size, plan.kv_lora_rank + plan.qk_rope_head_dim],
        ]
        if (descriptors["input_shapes"][1:5] != expected
                or descriptors["input_dtypes"][:5] != [
                    "DT_FLOAT8_E4M3FN", "DT_FLOAT8_E4M3FN", "DT_FLOAT8_E4M3FN",
                    "DT_BF16", "DT_FLOAT8_E4M3FN"]):
            raise ValueError(f"mla_projection_contract_differs_from_scenario:{group['id']}")
    view = RequestView("one", tokens, context if is_mla else 0, False, phase,
                       draft_prepare=False)
    kwargs = dict(
        prefix="probe", plan=plan, views=(view,), stage=0, deps="input",
        model=case.model, parallel=case.runtime.parallel, quant=case.model.quant,
        mode=phase, padded_tokens=tokens, runtime=case.runtime, policy=AttentionPolicy(),
    )
    book = RecipeBook.from_mapping(kernels)
    if is_mla:
        result = mla(**kwargs, cache_ids={}, topk_ids={}, index_cache_ids={},
                     candidate_ids={}, stage_of=lambda _: 0, cp_interleave_size=128)
        ops = lower_implementations(result.ops, case.runtime, case.model.activation_dtype)
        op = next(op for op in ops if dict(op.metadata).get("fusion_role") == "mla_prolog")
        cost = operator_costs(case, [op], recipes=book)["operators"][0]
        prediction = {
            "per_call_ms": cost["duration_ms"], "program": cost["program"],
            "boundary": "declared_native_resource_phase_dag", "basis": cost.get("basis"),
            "fusion_boundary": dict(op.metadata)["fusion_boundary"],
        }
        limits = [
            "FP8 projection operands/E8M0 group32 scales are descriptor-conditioned scenario inputs, not independent checkpoint-precision evidence; catalog dimensions are retained.",
            "Preserve-workspaces resource DAG does not claim native tiling/fusion traffic or binary identity.",
            f"Context {context} is an explicit declaration; the prolog does not scan historical KV and no per-step context is recovered.",
            "Native backing cache shape is not used as valid-token work.",
        ] + COMMON_LIMITS
    else:
        result = linear(**kwargs)
        declaration, ops = next((decl, ops) for decl, ops in _groups(result.ops)
                                if decl["role"] == "kda_chunk")
        local = {op.op_id for op in ops}
        external = {dep: 0 for op in ops for dep in op.deps if dep not in local}
        cost = graph_costs(case, {0: tuple(ops)}, recipes=book, external_inputs=external,
                           include_submission=False, isolated=False, trace_limit=0)
        ledger = kda_stage_ledger(descriptors, chunk_size=64)
        prediction = {
            "per_call_ms": cost["duration_ms"], "boundary": "complete_declared_kda_semantic_core",
            "operators": [(op.recipe, op.family) for op in ops],
            "declared_subgraph": declaration, "execution": cost["execution"],
            "stage_ledger": ledger,
            "stage_predictions": kda_stage_predictions(case, ops, book, ledger),
            "kda_implementation": lowerings["kda"],
        }
        limits = [
            "Native launches share a tensor descriptor but the stage scalar is absent; complete semantic call cost cannot be compared with launch Mean/P50 or multiplied by launch count.",
            "This is a complete in-kernel-gate KDA core reference under the explicit selected wrapper.",
        ]
    return {
        "id": group["id"], "native_kind": group["kind"],
        "scenario_descriptors": group["descriptors"], "prediction": prediction,
        "comparison_level": ("descriptor_conditioned_native_family_reference" if is_mla
                             else "complete_semantic_core_vs_unresolved_native_launch_stage"),
        "status": "conditional_reference" if is_mla else "not_comparable",
        "limitations": [
            f"Generic constituent matrices use the explicit {matrix_recipe} recipe; native source recipes retain precedence.",
        ] + limits,
    }


def predict_worker(packet):
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.prediction.structure import parse_case

    started, before = time.monotonic(), _runtime_identity()
    mode, groups = packet["mode"], packet["groups"]
    case, quant = parse_case(packet["case"]), QuantSpec(**packet["quant"])
    if not groups or mode not in {"grouped", "situ"} and len(groups) != 1:
        raise ValueError(f"semantic_reference_requires_descriptor_population:{mode}")
    matrix_recipe = packet["matrix_recipe"]
    if mode == "grouped":
        rows = grouped_predictions(case, groups, quant, matrix_recipe=matrix_recipe)
    elif mode == "situ":
        rows = situ_predictions(case, groups, quant)
    else:
        rows = [semantic_prediction(case, groups[0], quant, mode=mode,
                                    context=packet["context"], kda_tokens=packet["kda_tokens"],
                                    matrix_recipe=matrix_recipe,
                                    kda_implementation=packet.get("kda_implementation"))]
    return {
        "schema": "kimi_family_prediction_reference/2", "mode": mode,
        "status": "completed", "case_source": packet["case_source"],
        "scenario": packet, "runtime_before": before, "runtime_after": _runtime_identity(),
        "semantic_helper_source": source(situ_reference.__code__.co_filename),
        "rows": rows, "producer": producer(), "wall_seconds": time.monotonic() - started,
        "peak_rss_mib": worker_usage().get("peak_rss_mib"),
    }


def predict(observed_path, case_path, directory, *, context, kda_tokens,
            timeout=30, total_timeout=90, matrix_recipe="steady_matmul",
            kda_implementation=None):
    started = time.monotonic()
    observation, case = read_json(observed_path), read_json(case_path)
    case_source, directory = source(case_path), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    present = {row["kind"] for row in observation["groups"]}
    for mode in (mode for mode in MODES if KINDS[mode] in present):
        packet = prediction_packet(mode, observation, case, case_source,
                                   context=context, kda_tokens=kda_tokens,
                                   matrix_recipe=matrix_recipe,
                                   kda_implementation=kda_implementation)
        input_path = directory / f"input-{mode}.json"
        output_path = directory / f"prediction-{mode}.json"
        write_json(input_path, packet)
        remaining = total_timeout - (time.monotonic() - started)
        if remaining <= 0:
            result = {"mode": mode, "status": "failed", "reason": "total_prediction_budget_exhausted"}
        else:
            try:
                child = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "worker",
                     str(input_path), "-o", str(output_path)],
                    timeout=min(timeout, remaining), capture_output=True, text=True,
                )
                result = (read_json(output_path) if not child.returncode else {
                    "mode": mode, "status": "failed", "returncode": child.returncode,
                    "error": child.stderr[-4000:],
                })
            except subprocess.TimeoutExpired:
                result = {"mode": mode, "status": "failed", "reason": "prediction_timeout"}
        result["input_source"] = source(input_path)
        result["orchestrator"] = producer()
        write_json(output_path, result)
        results.append({**source(output_path), "mode": mode, "status": result["status"]})
    index = {
        "schema": "kimi_prediction_campaign/1", "results": results,
        "observation_source": source(observed_path), "producer": producer(),
        "wall_seconds": time.monotonic() - started,
        "budgets": {"per_probe_seconds": timeout, "total_seconds": total_timeout},
        "matrix_recipe": matrix_recipe,
        "kda_implementation_override": kda_implementation,
        "status": "completed" if all(row["status"] == "completed" for row in results) else "partial",
    }
    write_json(directory / "prediction-index.json", index)
    return index


def difference(predicted, observed):
    delta = predicted - observed
    return {
        "signed_gap_ms": delta, "absolute_gap_ms": abs(delta),
        "signed_gap_percent": 100 * (predicted / observed - 1) if observed else None,
        "absolute_gap_percent": 100 * abs(predicted / observed - 1) if observed else None,
    }


def byte_check(descriptors, actual):
    """Independent scalar check against the retained tensor inventories."""
    m, k = descriptors["input_shapes"][0]
    n = descriptors["output_shapes"][0][-1]
    fp4 = descriptors["input_dtypes"][1] == "DT_FLOAT4_E2M1"
    weight = math.prod(descriptors["input_shapes"][1]) * (0.5 if fp4 else 1)
    scales = math.prod(descriptors["input_shapes"][5 if fp4 else 3])
    activation_scales = math.prod(descriptors["input_shapes"][8])
    expected = {
        "flops": 2 * m * n * k, "mte2_bytes": m * k + weight + scales + activation_scales,
        "mte3_bytes": m * n * 2, "hbm_bytes": m * k + weight + scales + activation_scales + m * n * 2,
    }
    if not all(math.isclose(actual[key], value, rel_tol=1e-12)
               for key, value in expected.items()):
        raise ValueError("grouped_descriptor_bytes_or_flops_differ")
    return {
        "equal": True, "packed_weight_payload_bytes": weight, "weight_scale_bytes": scales,
        "activation_scale_bytes": activation_scales, "input_payload_bytes": m * k,
        "output_payload_bytes": m * n * 2,
        "scope": "semantic inventory; selected mapping may duplicate private operand copies",
    }


def comparison_row(original, observation, mode):
    prediction, count = original["prediction"], observation["count"]
    descriptors = original.get("scenario_descriptors", original.get("observed", {}).get("descriptors"))
    if descriptors != observation["descriptors"] or original["native_kind"] != observation["kind"]:
        raise ValueError(f"prediction_observation_descriptor_mismatch:{original['id']}")
    comparable = original["status"] == "conditional_reference"
    value = prediction["per_call_ms"]
    row = {
        "id": original["id"], "family": original["native_kind"], "mode": mode,
        "status": original["status"], "comparison_level": original["comparison_level"],
        "occurrences": count, "observed_mean_ms": observation["mean_ms"],
        "observed_p50_ms": observation["p50_ms"], "observed_sum_ms": observation["activity_sum_ms"],
        "predicted_ms": value, "predicted_sum_ms": value * count if comparable else None,
        "assumptions": original["limitations"], "descriptors": observation["descriptors"],
        "first_source_line": observation["first_source_line"], "first_native_name": observation["first_name"],
        "prediction_boundary": prediction["boundary"], "runtime_key": mode,
        "mean_error": difference(value, observation["mean_ms"]) if comparable else None,
        "p50_error": difference(value, observation["p50_ms"]) if comparable else None,
        "sum_error": difference(value * count, observation["activity_sum_ms"]) if comparable else None,
        "observation_range_ms": {"min": observation["min_ms"], "max": observation["max_ms"]},
    }
    for key in ("program", "matrix_groups", "compute_dtype", "logical_work",
                "operand_contract", "serviced", "fusion_boundary",
                "boundary_ledger", "stage_ledger", "stage_predictions", "kda_implementation"):
        if key in prediction:
            row[key] = prediction[key]
    mappings = (prediction.get("basis") or {}).get("matrix_mappings", [])
    if mappings:
        row["mapping_choices"] = [{key: value for key, value in mapping.items() if key != "candidates"}
                                  for mapping in mappings]
    if mode == "grouped":
        row["descriptor_byte_check"] = byte_check(row["descriptors"], row["logical_work"])
    if mode == "kda":
        row["semantic_reference"] = {key: prediction[key] for key in ("operators", "declared_subgraph")}
    if mode in {"kda", "situ"}:
        row["execution"] = compact_execution(prediction["execution"])
    return row


def totals(rows, **identity):
    count = sum(row["occurrences"] for row in rows)
    observed = sum(row["observed_sum_ms"] for row in rows)
    valid = all(row["predicted_sum_ms"] is not None for row in rows)
    predicted = sum(row["predicted_sum_ms"] for row in rows) if valid else None
    return {
        **identity, "descriptor_groups": len(rows), "occurrences": count,
        "observed_sum_ms": observed, "predicted_sum_ms": predicted,
        "observed_mean_ms": observed / count, "predicted_weighted_mean_ms": predicted / count if valid else None,
        "observed_p50_ms": rows[0]["observed_p50_ms"] if len(rows) == 1 else None,
        "p50_basis": "exact single descriptor population" if len(rows) == 1 else "not reconstructed from descriptor medians",
        "sum_error": difference(predicted, observed) if valid else None,
        "descriptor_absolute_residual_sum_ms": (
            math.fsum(row["sum_error"]["absolute_gap_ms"] for row in rows) if valid else None),
        "status": "conditional_reference" if valid else "not_comparable",
        "boundary": "sum_of_native_activity_over_whole_mixed_capture; not_step_span",
    }


def assemble(observed_path, prediction_paths, *, strict_paths=()):
    observation = read_json(observed_path)
    observations = {row["id"]: row for row in observation["groups"]}
    rows, inputs, snapshots, failures = [], [], {}, []
    for path in prediction_paths:
        report = read_json(path)
        mode = report["mode"]
        if mode in snapshots:
            raise ValueError(f"duplicate_prediction_mode:{mode}")
        inputs.append({
            **source(path), "mode": mode,
            "case_source": report.get("case_source"),
            "semantic_helper_source": report.get("semantic_helper_source"),
            "scenario_input_source": report.get("input_source"),
            "original_producer": report.get("producer", {
                "status": "not_embedded", "note": "Legacy temporary-driver output; new assembly did not execute it.",
            }),
        })
        if report["status"] != "completed":
            failures.append(report)
            snapshots[mode] = {"status": report["status"]}
            continue
        snapshots[mode] = {
            "before": report["runtime_before"], "after": report["runtime_after"],
            "stable_within_probe": report["runtime_before"] == report["runtime_after"],
            "wall_seconds": report["wall_seconds"], "peak_rss_mib": report["peak_rss_mib"],
            "matrix_recipe": report.get("scenario", {}).get("matrix_recipe", "not_recorded"),
        }
        rows.extend(comparison_row(row, observations[row["id"]], mode) for row in report["rows"])
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_prediction_descriptor_identity")
    missing = sorted(observations.keys() - set(ids))
    families = [totals([row for row in rows if row["family"] == name], family=name)
                for name in KINDS.values() if any(row["family"] == name for row in rows)]
    domains = [totals([row for row in rows if row.get("matrix_groups") == count], matrix_groups=count)
               for count in sorted({row["matrix_groups"] for row in rows if "matrix_groups" in row})]
    strict = []
    for path in strict_paths:
        report = read_json(path)
        strict.append({**source(path), **{key: report[key] for key in (
            "distinct_priced_programs", "status_counts", "mapping_coverage")}})
    return {
        "schema": "mindie-sim.kimi-family-scoreboard/2",
        "status": "partial" if missing or failures else "completed_with_explicit_conditional_references",
        "model": "kimi-k3", "capture": {
            **observation["source"], "raw_rows": observation["raw_rows"],
            "selected_counts": observation["counts"], "scope": "whole_mixed_capture",
            "complete_step_boundary": "not_established",
        },
        "observation_aggregation": {**source(observed_path), **{key: observation[key] for key in (
            "wall_seconds", "peak_rss_mib", "parser_sources")}},
        "prediction_inputs": inputs, "runtime_snapshots": snapshots,
        "assembly_producer": producer(),
        "provenance_policy": "Assembly reads retained predictions without rerunning them; each input producer remains independent.",
        "runtime_scope": "Per-mode source snapshots; not automatically one clean installed-wheel revision.",
        "strict_cohort_evidence": strict, "family_totals": families,
        "grouped_reference_domains": domains, "rows": rows, "failures": failures,
        "coverage": {
            "modeled_descriptor_groups": sum(row["status"] == "conditional_reference" for row in rows),
            "unresolved_native_stage_descriptor_groups": sum(row["status"] == "not_comparable" for row in rows),
            "missing_descriptor_ids": missing,
            "grouped_arithmetic_byte_and_flop_checks_passed": sum("descriptor_byte_check" in row for row in rows),
        },
        "limitations": [
            "The standalone precision declaration is descriptor-conditioned and is not equivalent to the main catalog case.",
            "Activity totals are not step spans, layer bindings, or additive critical-path contributions.",
            "Native stage and mapping uncertainty remains visible; it never calibrates rates or disables analytic prediction.",
            "Selected matrix recipes are recorded per mode; the grouped envelope does not infer expert row histograms.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    observed = commands.add_parser("observe", help="Stream selected native descriptor populations.")
    observed.add_argument("capture")
    observed.add_argument("-o", "--output", required=True)
    observed.add_argument("--max-rows", type=int, default=1000000)
    observed.add_argument("--timeout", type=float, default=25)
    predicted = commands.add_parser("predict", help="Run bounded independent family scenario workers.")
    predicted.add_argument("--observed", required=True)
    predicted.add_argument("--case", required=True)
    predicted.add_argument("--output-dir", required=True)
    predicted.add_argument("--context", type=int, default=32768)
    predicted.add_argument("--kda-tokens", type=int, default=8192)
    predicted.add_argument("--timeout", type=float, default=30)
    predicted.add_argument("--total-timeout", type=float, default=90)
    predicted.add_argument("--matrix-recipe", choices=("steady_matmul", "phase_matmul"),
                           default="steady_matmul")
    predicted.add_argument("--kda-implementation", choices=(
        "vllm_ascend_kda_in_kernel_gates", "vllm_ascend_kda_chunk_state",
        "vllm_ascend_kda_chunk_prepare_state"),
        help="Override KDA explicitly; otherwise inherit the case or retained semantic default.")
    assembled = commands.add_parser("assemble", help="Compare retained predictions without rerunning them.")
    assembled.add_argument("--observed", required=True)
    assembled.add_argument("--predictions", nargs="+", required=True)
    assembled.add_argument("--strict-cohort", action="append", default=[])
    assembled.add_argument("-o", "--output", required=True)
    worker = commands.add_parser("worker", help="Internal worker invoked by the bounded predict command.")
    worker.add_argument("input")
    worker.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "observe":
        if not 0 < args.timeout <= 30 or not 0 < args.max_rows <= 1000000:
            parser.error("observe requires timeout <=30s and max_rows <=1000000")
        report = observe(args.capture, max_rows=args.max_rows, timeout=args.timeout)
    elif args.command == "predict":
        if not 0 < args.timeout <= 30 or not 0 < args.total_timeout <= 90:
            parser.error("predict requires per-probe <=30s and total <=90s")
        report = predict(args.observed, args.case, args.output_dir, context=args.context,
                         kda_tokens=args.kda_tokens, timeout=args.timeout,
                         total_timeout=args.total_timeout, matrix_recipe=args.matrix_recipe,
                         kda_implementation=args.kda_implementation)
        return 0 if report["status"] == "completed" else 2
    elif args.command == "assemble":
        report = assemble(args.observed, args.predictions, strict_paths=args.strict_cohort)
    else:
        report = predict_worker(read_json(args.input))
    write_json(args.output, report)
    return 2 if report.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
