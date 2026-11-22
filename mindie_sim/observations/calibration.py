"""Observed call alignment, operator bindings and calibration evaluation.

[Role] Align observed kernel captures to predicted operator calls and bind
  GDN, grouped-MoE, activation, and general calibration profiles.
[Invariants] Zero silent skipping of unaligned calls.
[Timeouts & Locks] In-memory alignment; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from statistics import median

from mindie_sim.prediction.call_builders import call_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from .captures import iter_conditions
from .comparison import call_observations


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _statistics(samples):
    if not samples:
        return None
    values = [sample["duration_ms"] for sample in samples]
    trials = defaultdict(list)
    for sample in samples:
        trial = sample.get("trial", sample.get("metadata", {}).get("trial", 0))
        trials[sample.get("run_id"), trial].append(sample["duration_ms"])
    return {
        "sample_count": len(values), "trial_count": len(trials),
        "sample_median_ms": median(values),
        "median_of_trial_medians_ms": median(median(v) for v in trials.values()),
        "minimum_ms": min(values), "maximum_ms": max(values),
    }


def _capture_record(condition):
    samples = condition["samples"]
    return {
        "condition_id": condition["condition_id"], "source": condition["source"],
        "identity": condition["identity"],
        "boundary": condition["identity"].get("boundary"),
        "capture_status": condition["status"], "metadata": condition["metadata"],
        "family": condition["metadata"].get("variant", "grouped_gemm"),
        "canary": condition["metadata"].get("canary"),
        "sample_roles": dict(Counter(s["split"] for s in samples)),
        "run_ids": sorted({s["run_id"] for s in samples}),
        "statistics": _statistics(samples),
    }


def _gdn_record(observation):
    return {
        "condition_id": observation["observation_id"], "source": observation["source"],
        "identity": observation["identity"], "boundary": observation["boundary"],
        "capture_status": observation["status"],
        "family": observation["identity"]["conditions"]["role"],
        "metadata": {key: observation[key] for key in
                     ("producer", "capture_environment", "model_geometry_label",
                      "reported_statistics", "failure") if key in observation},
        "canary": observation["canary"],
        "sample_roles": {observation["split"]: len(observation["samples"])},
        "run_ids": [observation["source_sha256"]],
        "statistics": observation["statistics"],
    }


def _activation(condition):
    from .calibration import bind_activation_condition
    binding = bind_activation_condition(condition)
    return (binding.call, binding.ledger,
            {"invocations": binding.invocations, "mode": binding.mode})


def _gdn(observation):
    from .calibration import bind_observation
    binding = bind_observation(observation)
    if binding["status"] != "bound":
        raise ModelingError("; ".join(binding["mismatches"]))
    call = binding["call"]
    return call, {**call.ledger, "unresolved": list(call.limitations)}, {
        "invocations": 1, "mode": "eager"}


def _grouped(condition):
    from .calibration import bind_grouped_condition
    binding = bind_grouped_condition(condition)
    return binding.call, binding.ledger, {
        "invocations": binding.invocations, "mode": binding.mode}


# Adapters preserve original schemas; builders remain independently callable.
ADAPTERS = {
    "activation_calls": (iter_conditions, _capture_record, _activation),
    "grouped_calls": (iter_conditions, _capture_record, _grouped),
    "historical-call-protocol/1": (call_observations, _gdn_record, _gdn),
}


def _diagnostics(rows):
    groups = defaultdict(list)
    for row in rows:
        for comparison in row["comparisons"]:
            if "relative_difference" in comparison and comparison["relative_difference"] is not None:
                groups[row["family"], row["boundary"], comparison["scenario"]].append(comparison)
    return [{
        "family": family, "observation_boundary": boundary, "scenario": scenario,
        "conditions": len(values),
        "median_relative_difference": median(v["relative_difference"] for v in values),
        "minimum_relative_difference": min(v["relative_difference"] for v in values),
        "maximum_relative_difference": max(v["relative_difference"] for v in values),
        "interpretation": "descriptive_condition_distribution_not_workload_accuracy",
    } for (family, boundary, scenario), values in sorted(groups.items())]


def compare_calls(case, protocol, *, root=".", include_submission=False,
                  submission=None, max_predictions=512, max_conditions=4096):
    """Report conditional event/graph differences without fitting either side.

    The optional submitted scenario uses declared host queue parameters. It is
    an alternative scenario, not a confidence bound on the original event.
    Unsupported callback semantics remain rows with their original observations.
    """
    case = parse_case(case)
    case_sha256 = _hash(case.to_dict())
    adapter = protocol.get("adapter", protocol.get("schema"))
    if adapter not in ADAPTERS:
        raise ModelingError(f"call_alignment_protocol_unsupported:{adapter}")
    if max_predictions < 1 or max_conditions < 1:
        raise ModelingError("call_alignment_requires_positive_budgets")
    reader, record, bind = ADAPTERS[adapter]
    rows, predictions, cache = [], [], {}
    scenarios = [("device", False)] + ([("submitted", True)] if include_submission else [])
    failures, evaluations = 0, 0
    for observation in reader(protocol, root=root):
        if len(rows) >= max_conditions:
            raise ModelingError("call_alignment_condition_budget_exceeded")
        row = record(observation)
        row.update(status="unsupported", comparisons=[])
        rows.append(row)
        # A failed capture does not prevent a declaration from being simulated.
        # Only missing/unimplemented arguments prevent this adapter binding.
        try:
            call, ledger, schedule = bind(observation)
        except ValueError as exc:
            row["reason"] = str(exc)
            continue
        row["alignment"] = ledger
        row["status"] = "predicted"
        for scenario, with_submission in scenarios:
            key = _hash({"case_sha256": case_sha256, "identity": row["identity"],
                         "ledger": ledger, "schedule": schedule, "scenario": scenario,
                         "submission": dict(submission or {})})
            if key not in cache:
                if evaluations >= max_predictions:
                    cache[key] = {"status": "failed", "reason": "prediction_budget_exceeded"}
                else:
                    evaluations += 1
                    try:
                        result = call_costs(
                            case, call.ops, recipes=call.recipes, **schedule,
                            include_submission=with_submission, submission=submission,
                        )
                        prediction = {
                            "prediction_id": key, "scenario": scenario,
                            "status": "modeled", **result,
                        }
                        predictions.append(prediction)
                        cache[key] = prediction
                    except ValueError as exc:
                        cache[key] = {"status": "failed", "reason": str(exc)}
            prediction = cache[key]
            comparison = {
                "scenario": scenario, "status": prediction["status"],
                "strict_identity_equivalence": False,
                "observation_boundary": row["boundary"],
            }
            if prediction["status"] == "failed":
                comparison["reason"] = prediction["reason"]
                row["status"] = "prediction_failed"
                failures += 1
            else:
                comparison.update(
                    prediction_id=key, prediction_boundary=prediction["boundary"],
                    predicted_duration_ms=prediction["duration_ms"])
                statistics = row["statistics"]
                if statistics:
                    observed = statistics["sample_median_ms"]
                    delta = prediction["duration_ms"] - observed
                    comparison.update(
                        reference_statistic="sample_median_ms",
                        observed_duration_ms=observed, signed_difference_ms=delta,
                        absolute_difference_ms=abs(delta),
                        relative_difference=delta / observed if observed > 0 else None,
                        interpretation="conditional_boundary_difference",
                    )
                else:
                    comparison["interpretation"] = "prediction_without_timing_samples"
            row["comparisons"].append(comparison)
    statuses = Counter(row["status"] for row in rows)
    roles = Counter()
    for row in rows:
        roles.update(row["sample_roles"])
    return {
        "schema": "mindie-sim.call-alignment/1",
        "status": "completed" if statuses["predicted"] == len(rows) else "partial",
        "adapter": adapter, "case_sha256": case_sha256,
        "machine": case.machine.to_dict(),
        "protocol": protocol,
        "scenario_declarations": {
            "include_submission": include_submission, "submission": dict(submission or {}),
            "note": "Hardware launch service remains from the supplied case; no residual fitting.",
        },
        "summary": {
            "conditions": len(rows), "condition_statuses": dict(statuses),
            "samples": sum(roles.values()), "sample_roles": dict(roles),
            "unique_predictions": len(predictions), "prediction_evaluations": evaluations,
            "failed_condition_scenarios": failures,
        },
        "validation": {
            "status": "unassessed", "strict_pairs": 0,
            "reason": "Declared device/submitted graph spans retain a different boundary from raw events.",
        },
        "interpretation": [
            "Complete callback dataflow is aligned only within each binder's declared domain.",
            "Source references do not identify the historical loaded native binary.",
            "Host allocator, internal dispatch and event-enqueue gaps are not inferred from residuals.",
            "Device and submitted scenarios are declarations, not statistical lower/upper bounds.",
            "Original canaries, sample roles and event spans remain unchanged.",
            "Conditional differences diagnose mechanisms; no tolerance or accuracy pass is assigned.",
        ],
        "diagnostics": _diagnostics(rows),
        "predictions": predictions, "conditions": rows,
    }


from copy import deepcopy
from dataclasses import dataclass

from mindie_sim.prediction.call_builders import (
    ACTIVATION_CALLS, ActivationCall, ActivationCallSpec, build_activation_call,
)
from mindie_sim.schema.common import ModelingError


PRODUCER_SHA256 = "ea76c64cd935e2d35c849f957f2179701d81d72c951a738a430260de4b251136"
PRODUCER_PATH = (
    "repository://mindie-agent/.workstreams/composability/scripts/bench_activation.py"
)


@dataclass(frozen=True)
class ActivationBinding:
    call: ActivationCall
    invocations: int
    mode: str
    target_identity: dict
    condition_id: str
    ledger: dict


def bind_activation_condition(condition) -> ActivationBinding:
    """Return a conditional boundary binding, not an accuracy prediction row."""
    identity = condition["identity"]
    implementation = identity["implementation"]
    if implementation.get("producer_sha256") != PRODUCER_SHA256:
        raise ModelingError("activation_binding_requires_audited_producer")
    variant = implementation["variant"]
    spec = ActivationCallSpec(
        variant, identity["shape"]["tokens"], identity["shape"]["intermediate"],
        dtype=identity["dtype"], phase=identity["phase"],
    )
    declaration = ACTIVATION_CALLS[variant]
    if (implementation.get("api") != list(declaration["api"])
            or identity["layout"] != declaration["layout"]):
        raise ModelingError("activation_binding_api_or_layout_differs_from_producer")
    execution = identity["execution"]
    invocations = execution["invocations"]
    if not isinstance(invocations, int) or isinstance(invocations, bool) or not 1 <= invocations <= 100:
        raise ModelingError("activation_binding_invocations_outside_producer_protocol")
    burst = identity["boundary"] == "npu_event_burst"
    if identity["boundary"] not in {"npu_event_call", "npu_event_burst"} or (
            not burst and invocations != 1):
        raise ModelingError("activation_binding_unknown_event_protocol")
    expected_level = "subgraph" if burst or len(declaration["api"]) > 1 else "operator"
    expected_execution = {
        "mode": "eager", "timer": "npu_events",
        "input_values": "zeros", "output_lifetime": "discard_after_call",
        "warmup_calls": 3 if burst else 10,
        "timer_interleaving": "event_calls_event" if burst else "event_call_event",
    }
    if (identity["level"] != expected_level or any(
            execution.get(name) != value for name, value in expected_execution.items())):
        raise ModelingError("activation_binding_execution_differs_from_producer")
    call = build_activation_call(spec)
    ledger = {
        "binding": "retained_activation_api_closure/1",
        "producer": {"path": PRODUCER_PATH, "sha256": PRODUCER_SHA256},
        "timed_region": {
            "starts": "event_record_before_first_api",
            "ends": "event_record_after_last_api",
            "input_allocation_and_zero_fill": "outside",
            "warmup": "outside",
            "output_allocation_and_destruction": "inside_api_closure",
            "invocations": invocations,
            "api_calls_per_invocation": call.coverage["api_calls"],
            "serial_same_device_stream": True,
        },
        "known_structure": call.coverage,
        "unresolved": call.coverage["unresolved"],
        "conditional_comparison_only": True,
        "strict_identity_equivalence": False,
        "observation_role": "unchanged",
        "measured_values_consumed": False,
        "duration_basis": "separately_supplied_hardware_resources_and_call_schedule",
    }
    return ActivationBinding(
        call, invocations, "eager", deepcopy(identity), condition["condition_id"], ledger,
    )


from mindie_sim.prediction.operators.linear_attention.call_gdn import make_call


_PRODUCER_SHA256 = "1662a58957250fed9043593b62a9c5b558783850c1016f66367cb5737cb43063"
_IMPLEMENTATIONS = {
    "gating": "vllm_ascend.ops.triton.fused_gdn_gating.fused_gdn_gating_patch",
    "conv_prefill": "fla_npu.ops.ascendc.causal_conv1d_fn",
    "scan_prefill": "fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd",
}
_CONTRACTS = {
    "gating": {"a_dtype": "fp32", "b_dtype": "fp32", "a_log_dtype": "fp32",
               "dt_bias_dtype": "fp32"},
    "conv_prefill": {"activation": "silu", "bias": True, "initial_state": False,
                     "input_dtype": "bf16", "sequence_lengths": "batch equal-length sequences, tokens per sequence",
                     "state_cache_rows": "batch+1", "weight_layout": "[kernel,width]"},
    "scan_prefill": {"chunk_size": 64, "input_dtype": "bf16", "gate_dtype": "fp32",
                     "beta_dtype": "bf16", "state_dtype": "fp32",
                     "state_v_first": True, "output_final_state": True,
                     "use_qk_l2norm_in_kernel": True, "layout": "BSND",
                     "use_exp2": True, "disable_recompute": True,
                     "allow_neg_eigval": False},
}
_UNSUPPORTED = {
    "conv_update": [
        "native causal convolution runMode=1 with max_query_len=1 and caller-provided out is not implemented",
        "the prefill FN planner cannot substitute for update state movement",
    ],
    "scan_decode": [
        "complete callback includes q and k l2norm calls",
        "actual_seq_lengths is constructed and copied to the device inside every timed callback",
        "the native recurrence and its stable wrapper are not represented by the analytical serving recurrence alone",
    ],
}


def bind_observation(observation):
    """Return a call spec for compatible arguments, never a timing prediction."""
    identity = observation["identity"]
    condition, contract = identity["conditions"], identity["call_contract"]
    role = condition["role"]
    mismatches = list(_UNSUPPORTED.get(role, ()))
    if identity.get("producer_sha256") != _PRODUCER_SHA256:
        mismatches.append("producer source differs from the reviewed callback argument semantics")
    if role not in _IMPLEMENTATIONS and not mismatches:
        mismatches.append(f"unrecognized historical role:{role}")
    if role in _IMPLEMENTATIONS:
        if identity["implementation"] != _IMPLEMENTATIONS[role]:
            mismatches.append("implementation does not select the reviewed public callable")
        if condition.get("impl") not in (None, _IMPLEMENTATIONS[role]):
            mismatches.append("recorded condition names a different public callable")
        for field, expected in _CONTRACTS[role].items():
            if contract.get(field) != expected:
                mismatches.append(f"call_argument:{field}:expected={expected!r}:actual={contract.get(field)!r}")
        if role in ("gating", "conv_prefill"):
            dtype = "a fp32 / b fp32" if role == "gating" else "bf16"
            if condition.get("dtype") != dtype:
                mismatches.append(f"recorded_dtype:{condition.get('dtype')!r}:expected={dtype!r}")
        if role == "conv_prefill" and condition.get("layout") != "x[T_total,width] varlen":
            mismatches.append("recorded convolution layout is not the producer's contiguous varlen input")
        if role == "scan_prefill":
            batch, tokens = condition.get("batch"), condition.get("tokens")
            heads, keys = condition.get("v_heads"), condition.get("k_heads")
            dk, dv = condition.get("k_dim"), condition.get("v_dim")
            chunk = contract["chunk_size"]
            if condition.get("chunk") != chunk:
                mismatches.append(f"call_argument:chunk:expected={chunk!r}:actual={condition.get('chunk')!r}")
            if condition.get("phase") != "prefill":
                mismatches.append(f"recorded_phase:{condition.get('phase')!r}:expected='prefill'")
            if condition.get("dtype") not in (None, contract["input_dtype"]):
                mismatches.append(f"recorded_dtype:{condition.get('dtype')!r}:expected={contract['input_dtype']!r}")
            if condition.get("layout") not in (None, contract["layout"]):
                mismatches.append(f"recorded_layout:{condition.get('layout')!r}:expected={contract['layout']!r}")
            for name, value in (("batch", batch), ("tokens", tokens), ("k_heads", keys),
                                ("v_heads", heads), ("k_dim", dk), ("v_dim", dv)):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    mismatches.append(f"recorded_geometry:{name}:{value!r}")
            if not mismatches:
                total = batch * tokens
                chunks = (total + chunk - 1) // chunk
                state_bytes = batch * heads * dv * dk * 4
                if condition.get("chunks") != chunks:
                    mismatches.append(f"derived_chunks:{condition.get('chunks')!r}:expected={chunks}")
                if condition.get("state_bytes") != state_bytes:
                    mismatches.append(f"derived_state_bytes:{condition.get('state_bytes')!r}:expected={state_bytes}")
    result = {
        "status": "semantic_mismatch" if mismatches else "bound",
        "role": role, "observation_id": observation["observation_id"],
        "observation_boundary": observation["boundary"], "mismatches": mismatches,
        "call": None,
        "comparison_limit": "Device work is not relabeled as an NPU-event interval; host/event gaps remain separate.",
    }
    if mismatches:
        return result
    params = {"batch": condition["batch"], "tokens": condition["tokens"],
              "phase": condition["phase"]}
    if role == "gating":
        params.update(rows=params["batch"] * params["tokens"], heads=condition["heads"])
    elif role == "conv_prefill":
        params.update(conv_width=condition["conv_width"], kernel=condition["kernel"])
    else:
        contract = _CONTRACTS[role]
        params.update(k_heads=condition["k_heads"], v_heads=condition["v_heads"],
                      k_dim=condition["k_dim"], v_dim=condition["v_dim"],
                      chunk=contract["chunk_size"], input_dtype=contract["input_dtype"],
                      gate_dtype=contract["gate_dtype"], beta_dtype=contract["beta_dtype"],
                      state_dtype=contract["state_dtype"],
                      state_v_first=contract["state_v_first"],
                      output_final_state=contract["output_final_state"],
                      use_qk_l2norm_in_kernel=contract["use_qk_l2norm_in_kernel"],
                      layout=contract["layout"], use_exp2=contract["use_exp2"],
                      disable_recompute=contract["disable_recompute"],
                      allow_neg_eigval=contract["allow_neg_eigval"])
    result["call"] = make_call(role, {**params, **_CONTRACTS[role]})
    return result


from copy import deepcopy
from dataclasses import dataclass

from mindie_sim.prediction.operators.grouped_moe.call_grouped import (
    API, NZ_LAYOUT, PRODUCER_SHA256, QuantizedGroupedCall,
    build_quantized_grouped_call,
)
from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class GroupedBinding:
    call: QuantizedGroupedCall
    invocations: int
    mode: str
    target_identity: dict
    condition_id: str
    ledger: dict


def _expected_contract(experts):
    return {
        "api": API,
        "x": "int8 [tokens, k]",
        "weight": "int8 [E, k, n] FRACTAL_NZ",
        "scale": "float32 [E, n] per-channel",
        "per_token_scale": "float32 [tokens]",
        "split_item": 2,
        "group_type": 0,
        "group_list_type": 1,
        "group_list": f"int64 counts over E={experts} groups",
        "out_dtype": "bfloat16",
    }


def _expected_execution(mode, invocations):
    if mode == "eager" and invocations == 1:
        return {
            "mode": "eager", "timer": "npu_events", "invocations": 1,
            "warmup_calls": 10, "activation_quant": "outside_timed_region",
            "output_lifetime": "discard_after_call",
            "capture_warmup_calls": 0, "dependency": "serial",
        }
    if mode == "graph" and invocations in {1, 8}:
        return {
            "mode": "graph", "timer": "npu_events", "invocations": invocations,
            "warmup_calls": 10, "activation_quant": "outside_timed_region",
            "output_lifetime": "captured_static_buffers",
            "capture_warmup_calls": 3, "dependency": "serial",
        }
    raise ModelingError("grouped_binding_unknown_execution_protocol")


def _identity_contract(identity):
    """Return validated tensors/contract without reading observation samples."""
    implementation = identity.get("implementation")
    if not isinstance(implementation, dict):
        raise ModelingError("grouped_binding_requires_implementation_identity")
    if implementation.get("producer_sha256") != PRODUCER_SHA256:
        raise ModelingError("grouped_binding_requires_audited_producer")
    if implementation.get("api") != API:
        raise ModelingError("grouped_binding_api_differs_from_producer")
    contract = implementation.get("contract")
    if not isinstance(contract, dict):
        raise ModelingError("grouped_binding_requires_complete_api_contract")

    shape = identity.get("shape")
    if not isinstance(shape, dict) or set(shape) != {
            "tokens", "k", "n", "experts", "expert_counts"}:
        raise ModelingError("grouped_binding_requires_exact_shape_contract")
    counts = shape["expert_counts"]
    if (isinstance(counts, (str, bytes)) or not isinstance(counts, (list, tuple))
            or len(counts) != shape["experts"]
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in counts)):
        raise ModelingError("grouped_binding_requires_explicit_integer_expert_counts")
    if sum(counts) != shape["tokens"]:
        raise ModelingError("grouped_binding_counts_must_sum_to_logical_rows")
    if contract != _expected_contract(shape["experts"]):
        raise ModelingError("grouped_binding_api_contract_differs_from_producer")

    expected_dtype = {
        "input": "int8", "weight": "int8", "scale": "float32",
        "per_token_scale": "float32", "output": "bfloat16",
    }
    expected_layout = {"input": "ND", "weight": NZ_LAYOUT, "output": "ND"}
    if identity.get("dtype") != expected_dtype:
        raise ModelingError("grouped_binding_dtype_contract_differs_from_producer")
    if identity.get("layout") != expected_layout:
        raise ModelingError("grouped_binding_layout_contract_differs_from_producer")
    if identity.get("phase") != "primitive":
        raise ModelingError("grouped_binding_phase_differs_from_producer")
    return shape, counts, contract


def bind_grouped_condition(condition):
    """Bind one retained condition; never consume timings, canary or split."""
    identity = condition.get("identity")
    if not isinstance(identity, dict):
        raise ModelingError("grouped_binding_requires_preserved_identity")
    shape, counts, contract = _identity_contract(identity)

    execution = identity.get("execution")
    if not isinstance(execution, dict):
        raise ModelingError("grouped_binding_requires_execution_protocol")
    mode = execution.get("mode")
    invocations = execution.get("invocations")
    expected = _expected_execution(mode, invocations)
    if execution != expected:
        raise ModelingError("grouped_binding_execution_differs_from_producer")

    level = "subgraph" if mode == "graph" else "operator"
    boundary = ("npu_event_graph_replay" if mode == "graph" else "npu_event_call")
    if identity.get("level") != level or identity.get("boundary") != boundary:
        raise ModelingError("grouped_binding_event_boundary_differs_from_producer")

    call = build_quantized_grouped_call({
        "expert_rows": tuple(counts),
        "k": shape["k"],
        "n": shape["n"],
        "phase": "primitive",
    })
    ledger = deepcopy(call.ledger)
    ledger.update({
        "binding": "retained_grouped_int8_fractal_nz_call/1",
        "target_identity": deepcopy(identity),
        "condition_id": condition.get("condition_id"),
        "timed_protocol": {
            "mode": mode,
            "invocations": invocations,
            "whole_replay_events": mode == "graph",
            "sample_division": False,
            "serial_calls": True,
        },
        "read_scope": {
            "geometry": True,
            "dtype": True,
            "layout": True,
            "api_contract": True,
            "execution": True,
            "provenance": True,
            "duration": False,
            "canary": False,
            "sample_split": False,
        },
        "modeled_api_arguments": dict(contract),
        "conditional_comparison_only": True,
        "strict_identity_equivalence": False,
        "observation_role": "unchanged",
        "duration_basis": "separately_supplied_engine_resources_and_call_schedule",
    })
    return GroupedBinding(
        call, invocations, mode, deepcopy(identity), condition.get("condition_id"), ledger,
    )


from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math

from .resource_services import (
    RESOURCE_SERVICE_BINDING_SCHEMA,
    NormalizedResourceService,
    identity_digest,
    normalize_resource_service,
)


_CONDITION_MISSING = object()


def _condition_value(context, path):
    value = context
    for part in str(path).split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return _CONDITION_MISSING
            value = value[part]
        elif (isinstance(value, (list, tuple)) and part.isdigit()
              and int(part) < len(value)):
            value = value[int(part)]
        else:
            return _CONDITION_MISSING
    return value


def _condition_matches(actual, expected, path=""):
    if str(path).endswith("image_digest"):
        return identity_digest(actual) == identity_digest(expected)
    if str(path) == "dtype":
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, Mapping) and set(expected) <= {"min", "max"}:
        try:
            number = float(actual)
            if "min" in expected and number < float(expected["min"]):
                return False
            if "max" in expected and number > float(expected["max"]):
                return False
            return True
        except (TypeError, ValueError):
            return False
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)):
        return math.isclose(float(actual), float(expected),
                            rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return (actual.keys() == expected.keys()
                and all(_condition_matches(actual[key], expected[key], path)
                        for key in expected))
    if (isinstance(actual, (list, tuple))
            and isinstance(expected, (list, tuple))):
        return (len(actual) == len(expected)
                and all(_condition_matches(a, b, path)
                        for a, b in zip(actual, expected)))
    return actual == expected


def service_conditions_match(service, context):
    """Check only the conditions the service actually declares.

    Extra keys in a broader physical contract (unrelated rates, modes, owners or
    environment fields) are not identity.  A declared rate/mode/unit/environment
    condition must still match, so a real change in the depended-on supply is
    rejected.  ``conditions`` may use dotted paths such as
    ``resource_units.mte2_aiv`` or ``rates.hbm``.
    """
    if not isinstance(context, Mapping):
        raise ValueError("service_condition_context_requires_mapping")
    conditions = getattr(service, "conditions", None)
    if conditions is None and isinstance(service, Mapping):
        conditions = service.get("conditions")
    if conditions is None:
        raise ValueError("service_condition_service_requires_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("service_condition_conditions_requires_mapping")
    checks = []
    for path, expected in conditions.items():
        actual = _condition_value(context, path)
        present = actual is not _CONDITION_MISSING
        matched = present and _condition_matches(actual, expected, path)
        checks.append({
            "path": str(path),
            "expected": deepcopy(expected),
            "actual": deepcopy(actual) if present else None,
            "present": present,
            "matched": matched,
            "optional": False,
        })
    optional = getattr(service, "optional_conditions", None)
    if optional is None and isinstance(service, Mapping):
        optional = service.get("optional_conditions")
    if optional:
        if not isinstance(optional, Mapping):
            raise ValueError("service_optional_conditions_requires_mapping")
        for path, expected in optional.items():
            actual = _condition_value(context, path)
            present = actual is not _CONDITION_MISSING
            if not present or actual is None:
                checks.append({
                    "path": str(path), "expected": deepcopy(expected),
                    "actual": None, "present": present, "matched": True,
                    "optional": True,
                })
                continue
            matched = _condition_matches(actual, expected, path)
            checks.append({
                "path": str(path), "expected": deepcopy(expected),
                "actual": deepcopy(actual), "present": True,
                "matched": matched, "optional": True,
            })
    return {
        "applies": all(check["matched"] for check in checks),
        "checks": checks,
        "policy": ("only_declared_condition_paths; unrelated physical "
                   "contract keys are ignored"),
    }


def _condition_failure_reason(service, result):
    failed = [check for check in result["checks"] if not check["matched"]]
    if not failed:
        return "conditions_matched"
    parts = []
    for check in failed:
        actual = check["actual"] if check["present"] else "<missing>"
        parts.append(f"{check['path']}={actual!r}!={check['expected']!r}")
    return service.service_id + ":" + ";".join(parts)


def select_resource_service_candidate(services, context, *, service_id=None):
    """Narrow by optional ID, then apply the one condition matcher to all.

    Explicit ID, single candidate and automatic selection all pass through the
    same condition check.  A missing ID, missing condition, value mismatch or
    ambiguity raises a specific error; there is no nominal fallback.
    """
    if not isinstance(context, Mapping):
        raise ValueError("resource_service_context_requires_mapping")
    candidates = list(services or ())
    if service_id is not None:
        candidates = [item for item in candidates
                      if item.service_id == service_id]
        if not candidates:
            raise ValueError("resource_service_id_not_found:" + str(service_id))
    checked = []
    selected = []
    for service in candidates:
        result = service_conditions_match(service, context)
        checked.append((service, result))
        if result["applies"]:
            selected.append(service)
    if len(selected) == 1:
        return selected[0]
    if not selected:
        reasons = ";".join(
            _condition_failure_reason(service, result)
            for service, result in checked)
        raise ValueError("resource_service_conditions_have_no_match:" + reasons)
    raise ValueError("resource_service_conditions_are_ambiguous:"
                     + ",".join(item.service_id for item in selected))


def resource_service_binding(service, *, resource_role=None, phase=None,
                             work_unit=None, conditions=None):
    """Generate the exact T04/T14 binding for one normalized service.

    The binding names the service, the actual physical resource role, the
    declared work unit and the phase.  If the service declares applicability
    conditions, a real placement context must be supplied and match.  The
    produced mapping is the one consumed by ``Execution._context`` /
    ``_apply_resource_service_program``; it is not a whole-call duration.
    """
    if not isinstance(service, NormalizedResourceService):
        service = normalize_resource_service(service)
    if service.scope != "independent_resource" or service.kind != "resource_rate":
        raise ValueError("resource_service_binding_requires_independent_rate")
    role = str(resource_role or service.resource or "")
    if role not in service.resources:
        raise ValueError("resource_service_binding_role_not_declared:" + role)
    unit = str(work_unit or service.work_unit)
    if unit != service.work_unit:
        raise ValueError(
            "resource_service_binding_work_unit_mismatch:"
            + service.service_id + ":" + unit + ":" + service.work_unit)
    selected_phase = str(phase or service.phase)
    if service.phase and selected_phase != service.phase:
        raise ValueError(
            "resource_service_binding_phase_mismatch:"
            + service.service_id + ":" + selected_phase + ":" + service.phase)
    if service.conditions:
        if conditions is None:
            raise ValueError(
                "resource_service_binding_requires_conditions_context:"
                + service.service_id)
        if not service_conditions_match(service, conditions)["applies"]:
            raise ValueError(
                "resource_service_binding_conditions_do_not_match:"
                + service.service_id)
    source = deepcopy(service.provenance.get("source")
                      or service.provenance.get("sources"))
    return {
        "schema": RESOURCE_SERVICE_BINDING_SCHEMA,
        "service_id": service.service_id,
        "model_id": service.model_id,
        "resource_role": role,
        "work_unit": unit,
        "phase": selected_phase,
        "rate_scope": service.rate_scope,
        "active_units": service.active_units,
        "fixed_us": service.fixed_us,
        "effective_rate": service.effective_rate,
        "formula": service.formula,
        "conditions": deepcopy(service.conditions),
        "optional_conditions": deepcopy(service.optional_conditions),
        "conditions_checked": bool(service.conditions or service.optional_conditions),
        "boundary": service.boundary,
        "resource": service.resource,
        "resources": list(service.resources),
        "startup_us": service.startup_us,
        "response_us": service.response_us,
        "work_scope": service.rate_scope,
        "aggregate_rate": service.aggregate_rate,
        "per_owner_rate": service.per_owner_rate,
        "identification": deepcopy(service.identification),
        "rate_representation": deepcopy(service.rate_representation),
        "normalization": {**deepcopy(service.normalization),
                          "applied_once": True},
        "startup_once": True,
        "fixed_once": True,
        "formula": service.formula,
        "source": source,
        "production_prediction": True,
        "legacy": False,
    }


def attach_operator_resource_service_binding(op, binding):
    """Attach a previously selected binding to an OpSpec without re-selecting."""
    if not isinstance(binding, Mapping):
        raise ValueError("resource_service_binding_requires_mapping")
    metadata = [(key, value) for key, value in tuple(op.metadata)
                if key != "resource_service"]
    previous = dict(op.metadata).get("resource_service")
    if previous is not None and previous != dict(binding):
        raise ValueError("resource_service_binding_already_present")
    metadata.append(("resource_service", dict(binding)))
    return replace(op, metadata=tuple(metadata))


def bind_operator_resource_service(op, service, *, resource_role=None,
                                   phase=None, work_unit=None, conditions=None):
    """Return a new OpSpec carrying the exact T04/T14 resource-service binding.

    The actual selected producer calls this once after it has chosen the
    program/phase/context; T14 can pass the returned op to the common
    Execution.  The binding is never inferred later from a model label or a
    whole-call observation.
    """
    binding = resource_service_binding(
        service, resource_role=resource_role, phase=phase,
        work_unit=work_unit, conditions=conditions)
    binding.update({
        "op_id": getattr(op, "op_id", None),
        "op_family": getattr(op, "family", None),
        "op_stage": getattr(op, "stage", None),
        "op_phase": getattr(op, "phase", None),
    })
    return attach_operator_resource_service_binding(op, binding)


def bind_a5_local_service(record, *, domain, op=None, resource_role=None,
                          phase=None, include_lineage=False):
    """Select an accepted A5COMM local service and expose one T04 binding.

    The A5COMM applicability/presence/lineage functions remain the single
    authority.  This wrapper only turns the already-normalized service and its
    measured domain into the fields T04/T09/PW-FAMILIES/T15 consume.
    """
    from mindie_sim.observations.communication import (
        a5_local_service_applicability_key, a5_local_service_lineage_key,
        require_a5_local_service_applicability,
        require_a5_local_service_domain,
    )
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    if include_lineage:
        require_a5_local_service_domain(record, domain)
    else:
        require_a5_local_service_applicability(record, domain)
    service = normalize_resource_service(record["service"])
    binding = resource_service_binding(
        service, resource_role=resource_role or service.resource,
        phase=phase or service.phase, work_unit=service.work_unit,
        conditions=domain)
    provenance = dict(service.provenance)
    binding.update({
        "service_schema": record.get("schema"),
        "service_semantics": record.get("service_semantics"),
        "domain_key": deepcopy(record.get("domain_key") or {}),
        "applicability_key": a5_local_service_applicability_key(domain),
        "lineage_key": a5_local_service_lineage_key(domain),
        "evidence_gaps": list(record.get("evidence_gaps") or ()),
        "heldout_max_abs_relative_error": record.get(
            "heldout_max_abs_relative_error"),
        "link_transit_identified": bool(
            provenance.get("link_transit_identified", False)),
        "hop_scaling": provenance.get("hop_scaling", "not_derived"),
        "shared_domain": deepcopy(provenance.get("shared_domain")),
        "transport_window": deepcopy(provenance.get("transport_window")),
        "direction": service.conditions.get("direction"),
        "actual_phase": getattr(op, "phase", None) if op is not None else None,
        "physical_resource_role": (resource_role or service.resource),
    })
    if record.get("source") is not None:
        binding["source"] = deepcopy(record.get("source"))
    if op is None:
        return binding
    binding.update({
        "op_id": getattr(op, "op_id", None),
        "op_family": getattr(op, "family", None),
        "op_stage": getattr(op, "stage", None),
        "op_phase": getattr(op, "phase", None),
    })
    return attach_operator_resource_service_binding(op, binding)


def bind_a5_local_service_operator(record, op, *, domain, **kwargs):
    return bind_a5_local_service(record, domain=domain, op=op, **kwargs)


from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from mindie_sim.prediction.program.primitives import PrimitiveKey
from .resource_services import (
    RESOURCE_SERVICE_BINDING_SCHEMA,
    RESOURCE_SERVICE_PROFILE_SCHEMA,
    RESOURCE_SERVICE_SCHEMA,
    NormalizedResourceService,
    c7_applicability,
    identity_digest,
    normalize_resource_service,
)

_A5_REEXPORTS = frozenset({
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU",
    "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "A5CommCalibrationBundle", "bind_service_curve", "build_a5_comm_recipe",
    "build_a5_followup_manifest", "collect_native_service", "compute_a5_contention_slowdown",
    "evaluate_a5_mc2_chains", "fit_a5_service_curves", "ingest_a5_campaign_data",
    "resolve_expansion_regime", "to_service_regime",
})


def __getattr__(name: str):
    if name in _A5_REEXPORTS:
        from . import communication as _comm
        return getattr(_comm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

PROFILER_SUPPLY_SCHEMA = "mindie-profiler-measured-supply/1"




def import_profiler(catalogue, store, bindings, *, max_samples=1000):
    from .captures import ProfilerCatalogue

    imported = 0
    with ProfilerCatalogue(catalogue) as source:
        for binding in bindings:
            key = PrimitiveKey.from_mapping(binding["key"])
            rows = list(source.samples(binding["variant"], limit=max_samples + 1))
            if len(rows) > max_samples:
                raise ValueError("profiler_binding_exceeds_sample_budget:select_a_capture_subset")
            if not rows:
                raise ValueError("profiler_binding_has_no_raw_events")
            for row in rows:
                identity = row["source"] + ("::" + row["member"] if row["member"] else "")
                store.record(
                    key, duration_ms=row["duration_ms"],
                    run_id=binding["run_id"] + ":" + identity, repeat_index=row["row"],
                    source=f"{identity}#row={row['row']}", split=binding["split"])
                imported += 1
    return {"imported_samples": imported, "inventory": store.inventory()}


def _measured_supply_payload(source):
    if isinstance(source, dict):
        payload = source
        if payload.get("schema_version") == PROFILER_SUPPLY_SCHEMA:
            return payload
        enrichment = payload.get("enrichment") if isinstance(payload.get("enrichment"), dict) else {}
        pmu = enrichment.get("pmu_observations") if isinstance(enrichment.get("pmu_observations"), dict) else {}
        supply = pmu.get("measured_supply")
        if isinstance(supply, dict):
            return supply
        raise ValueError("profiler_report_has_no_measured_supply")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("profiler_supply_source_missing:" + str(path))
    if path.stat().st_size <= 50 * 1024 ** 2:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("schema_version") == PROFILER_SUPPLY_SCHEMA:
                return data
            enrichment = data.get("enrichment") if isinstance(data.get("enrichment"), dict) else {}
            pmu = enrichment.get("pmu_observations") if isinstance(enrichment.get("pmu_observations"), dict) else {}
            if isinstance(pmu.get("measured_supply"), dict):
                return pmu["measured_supply"]
    import ijson

    with path.open("rb") as stream:
        for item in ijson.items(
                stream, "enrichment.pmu_observations.measured_supply",
                use_float=True):
            if isinstance(item, dict):
                return item
    # A compact export may contain only the measured-supply object itself.
    with path.open("rb") as stream:
        for item in ijson.items(stream, "measured_supply", use_float=True):
            if isinstance(item, dict):
                return item
    raise ValueError("profiler_report_has_no_measured_supply")


def _profiler_source_description(source):
    if isinstance(source, Mapping):
        return {"kind": "in_memory",
                "schema_version": source.get("schema_version")}
    path = Path(source).expanduser().resolve()
    return {"kind": "file", "path": str(path), "bytes": path.stat().st_size}


def import_profiler_supply(source, *, recipe=None, kernel=None, hardware=None,
                           owners=None, required=True, require_hardware=True):
    """Return the normal service-regime rule from one measured dual-load pair.

    ``source`` is a C7 report path/mapping or an already-extracted
    ``mindie-profiler-measured-supply/1`` mapping.  The function never reads a
    target duration as a service cost; it copies only the measured pipe rates,
    active-owner counts and declared work domains.
    """
    if require_hardware and not hardware:
        raise ValueError("profiler_supply_requires_explicit_hardware_identity")
    supply = _measured_supply_payload(source)
    resources = deepcopy(supply.get("resource_services") or ())
    if supply.get("status") != "measured_pair" or not isinstance(supply.get("pair"), dict):
        if resources:
            return {"service_regimes": [], "supply": None,
                    "status": supply.get("status"),
                    "source": _profiler_source_description(source),
                    "source_identity": dict(supply.get("source_identity") or {}),
                    "resource_services": resources}
        if required:
            reasons = [item.get("code") for item in supply.get("limitations", ())
                       if isinstance(item, dict)]
            raise ValueError("profiler_measured_supply_not_available:"
                             + ",".join(str(code) for code in reasons if code))
        return {"service_regimes": [], "supply": supply, "status": supply.get("status"),
                "source": _profiler_source_description(source),
                "resource_services": []}
    pair = deepcopy(supply["pair"])
    rule = pair.get("service_regime")
    if not isinstance(rule, dict):
        raise ValueError("profiler_measured_supply_pair_requires_service_regime")
    match = dict(rule.get("match") or {})
    match.pop("kernel", None)
    match.pop("recipe", None)
    if recipe:
        match["recipe"] = str(recipe)
    elif kernel:
        match["kernel"] = str(kernel)
    elif not match.get("kernel"):
        match["kernel"] = str(pair.get("identity", {}).get("native_type") or "")
    if hardware:
        match["hardware"] = str(hardware)
    rule["match"] = match
    # Validate with the same public response parser the runtime uses.
    from mindie_sim.prediction.response_functions import ResponseFunction

    coordinates = rule.get("coordinates") or {}
    if not isinstance(coordinates, dict) or not coordinates:
        raise ValueError("profiler_supply_regime_requires_coordinates")
    if owners is not None:
        if isinstance(owners, bool) or not isinstance(owners, int) or owners <= 0:
            raise ValueError("profiler_supply_owners_must_be_positive_integer")
        owner_coordinate = coordinates.get("owners")
        owner_coordinate = dict(owner_coordinate) if isinstance(owner_coordinate, dict) else {
            "path": "metadata.service_owners"}
        owner_coordinate["default"] = owners
        coordinates = {**coordinates, "owners": owner_coordinate}
        rule["coordinates"] = coordinates
        rule["provenance"] = {
            **dict(rule.get("provenance") or {}),
            "declared_owner_default": owners,
            "declared_owner_basis": "caller_selected_implementation_active_owners",
        }
    ResponseFunction.from_mapping(rule.get("response"), coordinates)
    return {
        "service_regimes": [rule],
        "supply": pair,
        "status": "measured_pair",
        "source": _profiler_source_description(source),
        "source_identity": dict(supply.get("source_identity") or {}),
        "resource_services": resources,
        "source_schema": supply.get("schema_version"),
        "scope": supply.get("scope"),
    }


def _normalize_profiler_resource_services(resources, *, hardware, source,
                                          source_identity):
    normalized = []
    for raw in resources or ():
        if not isinstance(raw, Mapping):
            raise ValueError("profiler_resource_service_requires_mapping")
        rate = raw.get("rate_bytes_per_s")
        if rate is None:
            continue
        owners = raw.get("owners")
        if isinstance(owners, bool) or not isinstance(owners, int) or owners < 1:
            raise ValueError("profiler_resource_service_requires_positive_owners")
        resource = str(raw.get("resource") or "").strip()
        if not resource:
            raise ValueError("profiler_resource_service_requires_resource")
        declared_hardware = str(raw.get("hardware") or hardware or "").strip()
        if not declared_hardware:
            raise ValueError("profiler_resource_service_requires_hardware")
        hardware_gap = bool(
            hardware and raw.get("hardware")
            and str(raw["hardware"]) != str(hardware))
        c7_record = ({**dict(raw), "hardware": declared_hardware}
                     if not raw.get("hardware") else dict(raw))
        applicability = c7_applicability(c7_record)
        conditions = dict(applicability["conditions"])
        optional_conditions = dict(applicability["optional_conditions"])
        geometry = dict(raw.get("geometry_binding") or {})
        source_sha = str(raw.get("source_sha256") or "")
        identifier = (f"profiler:{resource}:"
                      f"{raw.get('shape_class') or 'default'}:"
                      f"{source_sha[:12] or 'unhashed'}")
        source_scope = str(raw.get("scope") or "")
        effective_rate = float(rate)
        rate_conversion = "per_owner_rate_preserved_critical_owner_work"
        model = normalize_resource_service({
            "id": identifier,
            "scope": "independent_resource",
            "resources": [resource],
            "phase": str(raw.get("phase") or "prefill"),
            "work_unit": "bytes",
            "boundary": "resource_phase",
            "startup_us": 0,
            "response_us": 0,
            "active_units": owners,
            "rate_scope": "per_owner",
            "supply": {"kind": "absolute_rate", "rate": effective_rate},
            "conditions": conditions,
            "optional_conditions": optional_conditions,
            "identification": {
                "status": "identified",
                "resource_identified": True,
                "rate_source": "measured_pmu",
                "source_identity": deepcopy(
                    raw.get("source_identity") or source_identity or {}),
                **({"train_work": [int(raw["measured_critical_bytes"])]}
                   if raw.get("measured_critical_bytes") else {}),
            },
            "provenance": {
                "kind": "profiler_resource_service",
                "source": deepcopy(source),
                "source_sha256": source_sha,
                "source_rate_bytes_per_s": float(rate),
                "source_rate_scope": source_scope,
                "rate_conversion": rate_conversion,
                "source_measured_critical_bytes": raw.get("measured_critical_bytes"),
                "source_identity": deepcopy(
                    raw.get("source_identity") or source_identity or {}),
                "geometry_binding": geometry,
                "c7_work_domain": deepcopy(applicability["work_domain"]),
                "not_elapsed_rate": bool(
                    dict(raw.get("provenance") or {}).get("not_elapsed_rate")),
                "selection_policy": "c7_applicability_match",
                "requested_hardware": str(hardware) if hardware else None,
                "applicability_gap": (
                    "hardware_mismatch" if hardware_gap else None),
                "production_prediction": True,
            },
        })
        normalized.append(model)
    return tuple(normalized)


def _legacy_diagnostic_comparison(rules, normalized):
    rows = []
    normalized_ids = [item.service_id for item in normalized]
    for rule in rules:
        response = dict(rule.get("response") or {})
        rows.append({
            "rule_id": rule.get("id"),
            "response_kind": response.get("kind", "affine"),
            "legacy_fixed_us": response.get("fixed_us"),
            "legacy_rate_terms": deepcopy(response.get("terms") or response.get("loads") or ()),
            "normalized_service_ids": normalized_ids,
            "relation": "legacy_diagnostic_only_not_used_for_production",
            "difference_is_diagnostic": True,
        })
    return {
        "rules": rows,
        "normalized_parameters": [{
            "service_id": item.service_id,
            "resource": item.resource,
            "work_unit": item.work_unit,
            "rate_scope": item.rate_scope,
            "active_units": item.active_units,
            "effective_rate": item.effective_rate,
            "fixed_us": item.fixed_us,
        } for item in normalized],
        "production_prediction": False,
    }


def apply_profiler_supply(case, source, *, recipe=None, kernel=None,
                          hardware=None, owners=None, required=True,
                          require_hardware=True, legacy_diagnostics=False):
    """Attach normalized C7 resource services; legacy rules stay diagnostic.

    Ordinary prediction never receives the imported ``service_regimes``.
    Those rules are preserved under ``runtime.metadata`` with their source and
    ``production_prediction=false``.  Only an explicit
    ``legacy_diagnostics=True`` replay materializes them into
    ``runtime.kernels.service_regimes``.
    """
    if not isinstance(case, dict):
        raise ValueError("profiler_supply_case_requires_mapping")
    imported = import_profiler_supply(
        source, recipe=recipe, kernel=kernel, hardware=hardware,
        owners=owners, required=required, require_hardware=require_hardware)
    if not imported["service_regimes"] and not imported.get("resource_services"):
        return deepcopy(case)
    result = deepcopy(case)
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("profiler_supply_case_runtime_requires_mapping")
    metadata = runtime.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("profiler_supply_case_runtime_metadata_requires_mapping")
    source_desc = deepcopy(imported.get("source") or {})
    source_identity = deepcopy(imported.get("source_identity") or {})
    pair = imported.get("supply") or {}
    normalized = _normalize_profiler_resource_services(
        imported.get("resource_services") or (),
        hardware=hardware, source=source_desc, source_identity=source_identity)
    legacy_rules = deepcopy(imported.get("service_regimes") or [])
    for rule in legacy_rules:
        rule["production_prediction"] = False
        rule["diagnostic_only"] = True
        rule["diagnostic_source"] = deepcopy(source_desc)
        rule["source_identity"] = deepcopy(source_identity)
    execution = (dict(legacy_rules[0].get("execution") or {})
                 if legacy_rules else {})
    if normalized:
        profile = {
            "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
            # Top-level hardware is the source/profile identity, not the
            # caller's requested placement.  A mismatched placement is an
            # explicit applicability gap checked at selection time.
            "hardware": (str(normalized[0].conditions.get("hardware") or "") or None),
            "services": [item.to_dict() for item in normalized],
            "source": source_desc,
            "provenance": {
                "kind": "profiler_measured_supply",
                "source_identity": source_identity,
                "production_prediction": True,
            },
        }
        existing = metadata.get("resource_service_profile")
        if existing is None:
            metadata["resource_service_profile"] = profile
        else:
            if not isinstance(existing, dict):
                raise ValueError("profiler_supply_existing_profile_requires_mapping")
            merged = deepcopy(existing)
            services = list(merged.get("services") or ())
            have = {item.get("id") for item in services if isinstance(item, Mapping)}
            services.extend(item for item in profile["services"]
                            if item.get("id") not in have)
            merged["services"] = services
            metadata["resource_service_profile"] = merged
    metadata["profiler_measured_supply"] = {
        "schema_version": imported.get("source_schema"),
        "scope": imported.get("scope"),
        "source": source_desc,
        "identity": deepcopy(pair.get("identity")),
        "service_ratio": deepcopy(pair.get("service_ratio")),
        "declared_owner_default": owners,
        "source_identity": source_identity,
        "resource_services": deepcopy(imported.get("resource_services") or ()),
        "normalized_resource_services": [item.to_dict() for item in normalized],
        "legacy_service_regimes": legacy_rules,
        "legacy_production_prediction": False,
        "legacy_diagnostics_replay": bool(legacy_diagnostics),
        "diagnostic_comparison": _legacy_diagnostic_comparison(
            legacy_rules, normalized),
        "applicability_gaps": [
            {"service_id": item.service_id,
             "gap": item.provenance.get("applicability_gap")}
            for item in normalized
            if item.provenance.get("applicability_gap")],
        "mac_execution": execution or None,
        "mac_execution_production_prediction": False,
    }
    if legacy_diagnostics:
        kernels = runtime.setdefault("kernels", {})
        if not isinstance(kernels, dict):
            raise ValueError("profiler_supply_case_kernels_requires_mapping")
        regimes = list(kernels.get("service_regimes") or ())
        # The runtime parser accepts the original rule schema; our
        # production_prediction/diagnostic markers stay only in metadata.
        regimes.extend(deepcopy(imported.get("service_regimes") or ()))
        kernels["service_regimes"] = regimes
        metadata["legacy_diagnostics"] = True
    return result

_RESOURCE_PROFILE_MAX_BYTES = 50 * 1024 ** 2


def _resource_profile_payload(source, *, source_kind=None):
    if isinstance(source, Mapping):
        payload = deepcopy(dict(source))
        origin = {
            "kind": source_kind or "in_memory",
            "label": str(payload.get("name") or "resource_service_profile"),
        }
        return payload, origin
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("resource_profile_source_missing:" + str(path))
    size = path.stat().st_size
    if size > _RESOURCE_PROFILE_MAX_BYTES:
        raise ValueError("resource_profile_source_exceeds_50_mib")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("resource_profile_source_requires_utf8_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("resource_profile_source_requires_object")
    return payload, {
        "kind": source_kind or "file",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": size,
    }


def _profile_origin(source):
    """Return a preserved loaded-profile origin, never a caller request."""
    if not isinstance(source, Mapping):
        return None
    if source.get("schema") != RESOURCE_SERVICE_PROFILE_SCHEMA:
        return None
    origin = source.get("source")
    if not isinstance(origin, Mapping):
        return None
    kind = origin.get("kind")
    if kind not in {"file", "builtin", "in_memory", "user"}:
        return None
    if kind == "file" and not (origin.get("path") and origin.get("sha256")):
        return None
    return deepcopy(dict(origin))


def load_resource_profile(source, *, nominal_rates=None, hardware=None,
                          source_kind=None):
    """Load one built-in or user resource-service profile through one path.

    ``source`` is a mapping (built-in profile) or a JSON file path (user
    profile).  Exactly the same normalization runs for both.  Every declared
    multiplier resolves against an explicit nominal rate; an incompatible or
    incomplete profile raises instead of falling back to a nominal service.
    """
    preserved_origin = (_profile_origin(source)
                        if source_kind is None else None)
    raw, origin = _resource_profile_payload(source, source_kind=source_kind)
    if preserved_origin is not None:
        origin = preserved_origin
    schema = raw.get("schema")
    if schema == RESOURCE_SERVICE_PROFILE_SCHEMA:
        services = raw.get("services")
        profile_hardware = raw.get("hardware")
        profile_scope = raw.get("scope") or "resource_and_phase_services"
        profile_provenance = raw.get("provenance") or {}
        profile_source = raw.get("source")
    elif schema == RESOURCE_SERVICE_SCHEMA:
        services = [raw]
        profile_hardware = None
        profile_scope = "single_resource_service"
        profile_provenance = raw.get("provenance") or {}
        profile_source = None
    else:
        raise ValueError("resource_profile_schema_unknown")
    if not isinstance(services, (list, tuple)) or not services:
        raise ValueError("resource_profile_requires_at_least_one_service")
    if not isinstance(profile_provenance, Mapping):
        raise ValueError("resource_profile_provenance_requires_mapping")
    try:
        profile_provenance = json.loads(json.dumps(
            dict(profile_provenance), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("resource_profile_provenance_requires_json_values") from exc
    if hardware and profile_hardware and str(profile_hardware) != str(hardware):
        raise ValueError("resource_profile_hardware_mismatch")
    effective_hardware = str(hardware or profile_hardware or "") or None
    normalized = []
    identifiers = set()
    for index, service in enumerate(services):
        if not isinstance(service, Mapping):
            raise ValueError(f"resource_profile_service_{index}_requires_mapping")
        item = deepcopy(dict(service))
        conditions = item.get("conditions") or {}
        if not isinstance(conditions, Mapping):
            raise ValueError("resource_profile_service_conditions_requires_mapping")
        conditions = dict(conditions)
        if effective_hardware and not conditions.get("hardware"):
            conditions["hardware"] = effective_hardware
        item["conditions"] = conditions
        provenance = item.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("resource_profile_service_provenance_requires_mapping")
        provenance = dict(provenance)
        provenance.setdefault("profile_source", origin)
        item["provenance"] = provenance
        model = normalize_resource_service(
            item, nominal_rates=nominal_rates, hardware=effective_hardware)
        if model.service_id in identifiers:
            raise ValueError("resource_profile_requires_distinct_service_ids")
        identifiers.add(model.service_id)
        normalized.append(model)
    formulas = {model.service_id: model.formula for model in normalized}
    return {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "source": origin,
        "profile_source": deepcopy(profile_source),
        "hardware": effective_hardware,
        "scope": profile_scope,
        "provenance": deepcopy(profile_provenance),
        "services": [model.to_dict() for model in normalized],
        "formulas": formulas,
        "service_ids": [model.service_id for model in normalized],
        "normalization": {
            "applied_once": True,
            "service_count": len(normalized),
            "nominal_rates_supplied": nominal_rates is not None,
        },
    }


def export_resource_profile(profile, *, path=None):
    """Export normalized parameters and formulas; write only when asked."""
    if not isinstance(profile, Mapping):
        raise ValueError("resource_profile_export_requires_mapping")
    schema = profile.get("schema")
    if schema == RESOURCE_SERVICE_PROFILE_SCHEMA:
        payload = deepcopy(dict(profile))
        services = payload.get("services")
        if not isinstance(services, (list, tuple)) or not services:
            raise ValueError("resource_profile_export_requires_services")
        models = [normalize_resource_service(service) for service in services]
        if len({model.service_id for model in models}) != len(models):
            raise ValueError("resource_profile_requires_distinct_service_ids")
        payload["services"] = [model.to_dict() for model in models]
        payload["formulas"] = {model.service_id: model.formula for model in models}
        payload["service_ids"] = [model.service_id for model in models]
        payload["normalization"] = {
            **dict(payload.get("normalization") or {}),
            "applied_once": True,
            "service_count": len(models),
        }
    elif schema == RESOURCE_SERVICE_SCHEMA:
        model = normalize_resource_service(profile)
        payload = {
            "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
            "source": {"kind": "in_memory", "label": model.service_id},
            "services": [model.to_dict()],
            "formulas": {model.service_id: model.formula},
            "service_ids": [model.service_id],
            "normalization": {"applied_once": True, "service_count": 1},
        }
    else:
        raise ValueError("resource_profile_export_requires_loaded_or_service_schema")
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resource_profile_export_requires_json_values") from exc
    if path is None:
        return payload
    target = Path(path).expanduser().resolve()
    data = (text + "\n").encode("utf-8")
    target.write_bytes(data)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "profile": payload,
    }


def apply_resource_profile(case, source, *, nominal_rates=None, hardware=None):
    """Attach a normalized profile to a new Case metadata handoff.

    In-memory and file profiles use the identical load/normalization path, so
    top-level hardware propagation and service-condition conflicts behave the
    same.  An already-loaded profile keeps its original source identity.  The
    function does not rewrite recipes, service regimes or hardware rates; a
    later consumer owns how a service enters its program.
    """
    if not isinstance(case, Mapping):
        raise ValueError("resource_profile_case_requires_mapping")
    profile = load_resource_profile(
        source, nominal_rates=nominal_rates, hardware=hardware)
    result = deepcopy(dict(case))
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("resource_profile_case_runtime_requires_mapping")
    metadata = runtime.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("resource_profile_case_runtime_metadata_requires_mapping")
    metadata["resource_service_profile"] = deepcopy(profile)
    metadata["resource_service_profile_source"] = deepcopy(profile.get("source"))
    metadata["resource_service_profile_hardware"] = profile.get("hardware")
    return result
