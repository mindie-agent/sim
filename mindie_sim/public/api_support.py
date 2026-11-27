"""Shared provider dispatch, gap classification, and report checks for ``public.api``.

[Role] Hold the input normalization, extension provider selection, typed
  ``CapabilityGap`` wrapping, and legacy report schema validators used by
  ``mindie_sim/public/api.py``.
[Invariants] Selected providers never fall back to core on failure. Report
  validators preserve exact schema identifiers and finite non-negative metrics.
[Timeouts & Locks] Pure in-memory helpers; no I/O or locks.
[Side-Effect Policy] Never mutates caller mappings, profiles, or contexts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import math
from typing import Any

from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec

from .contracts import CORE_PROVIDER, ExtensionContext
from .environment import ExecutionEnvironment
from .errors import CapabilityGap, ExtensionError
from .profiles import UserProfile

_GAP_UNMODELED = (
    "unsupported", "unmodeled", "not_modeled", "not_implemented",
    "native_compute_mode", "unknown_compute", "unknown_quant",
    "quantization_not", "abi_not", "abi_spec_not_declared",
    "representation_not",
)
_GAP_INFEASIBLE = (
    "exceeds", "exceeded", "capacity", "infeasible", "cannot_fit",
    "does_not_fit", "insufficient_capacity", "over_budget",
)
_GAP_FIELDS = (
    "missing", "requires", "require_", "ambiguous", "unknown_field",
    "not_declared", "undeclared", "invalid",
)
_ACLNN_DECLARATION_KEYS = (
    "api", "symbol", "aclnn_symbol", "entry", "arguments", "tensors", "attrs",
)
_ACLNN_PROTOCOL_KEYS = frozenset({
    "include_submission", "input_ready_ms", "invocations", "max_operators",
    "mode", "recipes", "replica", "submission", "trace_limit",
})


def _environment(value: Any) -> ExecutionEnvironment | None:
    if isinstance(value, ExecutionEnvironment):
        return value
    if isinstance(value, Hardware):
        return ExecutionEnvironment.from_hardware(value)
    if isinstance(value, Mapping) and (
            ("model" not in value and "workload" not in value)
            or (value.get("model") is None and value.get("workload") is None)):
        machine = value.get("machine") or value.get("hardware")
        if machine is None and "device" in value:
            machine = value
        if machine is not None:
            return ExecutionEnvironment.from_hardware(
                machine, runtime=value.get("runtime"),
                parallel=value.get("parallel"))
    return None


def _profile(value: Any) -> UserProfile | None:
    if value is None:
        return None
    if not isinstance(value, UserProfile):
        raise ExtensionError("user_profile_requires_load_user_profile")
    return value


def _profile_environment(case: Any, ops: Any = None) -> ExecutionEnvironment:
    """Bind a case onto the unified ``ExecutionEnvironment`` for profile pricing."""
    del ops
    parsed = parse_case(case)
    return ExecutionEnvironment.from_hardware(
        parsed.machine, runtime=parsed.runtime,
    )


def _op_sequence(value: Any) -> tuple[OpSpec, ...]:
    if isinstance(value, OpSpec):
        return (value,)
    if isinstance(value, (str, bytes, bytearray)):
        raise ExtensionError("operator_requires_op_spec_or_sequence")
    try:
        return tuple(value)
    except TypeError as exc:
        raise ExtensionError("operator_requires_op_spec_or_sequence") from exc


def _ops(value: Any) -> tuple[OpSpec, ...] | dict[int, tuple[OpSpec, ...]]:
    if isinstance(value, OpSpec):
        return (value,)
    if isinstance(value, Mapping):
        mapped: dict[int, tuple[OpSpec, ...]] = {}
        for rank, rank_ops in value.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ExtensionError("operator_replica_key_invalid:" + str(rank))
            mapped[rank] = _op_sequence(rank_ops)
        return mapped
    return _op_sequence(value)


def _gap_kind(reason: str) -> str:
    """Classify a concrete modeling failure without hiding its raw reason."""
    lowered = reason.lower()
    if any(token in lowered for token in _GAP_UNMODELED):
        return "unmodeled_capability"
    if any(token in lowered for token in _GAP_INFEASIBLE):
        return "infeasible"
    if any(token in lowered for token in _GAP_FIELDS):
        return "insufficient_fields"
    return "invalid_input"


def _machine_reference(environment: ExecutionEnvironment | None) -> Any:
    if environment is None:
        return None
    machine = environment.machine
    return {
        "sku": getattr(machine, "sku", ""),
        "backend": getattr(machine, "backend", ""),
        "num_devices": getattr(machine, "num_devices", None),
    }


def _ops_replay(ops: Any) -> Any:
    try:
        if isinstance(ops, Mapping):
            return {
                str(rank): [
                    item.to_dict() if hasattr(item, "to_dict") else {"type": type(item).__name__}
                    for item in _op_sequence(rank_ops)
                ]
                for rank, rank_ops in ops.items()
            }
        values = _op_sequence(ops)
        return [
            value.to_dict() if hasattr(value, "to_dict") else {"type": type(value).__name__}
            for value in values
        ]
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001 - replay metadata must never mask the gap
        return {"type": type(ops).__name__}


def _case_replay(case: Any) -> dict[str, Any]:
    if isinstance(case, Mapping):
        model = case.get("model") if isinstance(case.get("model"), Mapping) else {}
        workload = case.get("workload") if isinstance(
            case.get("workload"), Mapping) else {}
        requests = workload.get("requests") if isinstance(
            workload.get("requests"), Sequence) else ()
        return {
            "name": case.get("name"),
            "model_catalog_key": model.get("catalog_key"),
            "model_quantization": model.get("quantization"),
            "requests": len(requests),
            "request_geometry": [
                {"prompt_tokens": row.get("prompt_tokens"),
                 "output_tokens": row.get("output_tokens")}
                for row in requests[:8] if isinstance(row, Mapping)],
            "geometry": {key: case.get(key) for key in
                         ("phase", "tokens", "context", "sequences") if key in case},
        }
    name = getattr(case, "name", None)
    model = getattr(case, "model", None)
    return {
        "name": name,
        "model_catalog_key": getattr(model, "catalog_key", None),
        "type": type(case).__name__,
    }


def _core(capability: str, entry: str, call: Any, *,
          affected_results: Sequence[str],
          minimal_replay_input: Any) -> Any:
    """Run one existing core entry; a concrete failure becomes a typed gap."""
    try:
        return call()
    except CapabilityGap:
        raise
    except ExtensionError:
        raise
    except ModelingError as exc:
        reason = str(getattr(exc, "reason", exc))
        raise CapabilityGap(
            capability=capability, entry=entry, kind=_gap_kind(reason),
            code=reason.split(":", 1)[0] or "modeling_error",
            message=reason, affected_results=affected_results,
            minimal_replay_input=minimal_replay_input,
            fallback="not_attempted", cause=exc) from exc


def _context(value: Any) -> ExtensionContext | None:
    if value is None:
        return None
    if not isinstance(value, ExtensionContext):
        raise ExtensionError("extension_context_requires_load_extensions_result")
    return value


def _provider(
    capability: str, provider: str | None, extensions: ExtensionContext | None,
):
    if provider == CORE_PROVIDER:
        return None
    if extensions is None:
        if provider is not None:
            raise ExtensionError("extension_provider_requires_loaded_context:"
                                 + capability + ":" + str(provider))
        return None
    if provider is not None:
        return extensions.provider(capability, provider)
    return extensions.select(capability)


def _handler(
    capability: str, provider: str | None, extensions: ExtensionContext | None,
):
    selected = _provider(capability, provider, extensions)
    return selected.factory if selected is not None else None


def _accepts_context(factory: Any) -> bool:
    try:
        return "extensions" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _invoke(factory: Any, extension_context: ExtensionContext | None, /,
            **kwargs: Any) -> Any:
    if extension_context is not None and _accepts_context(factory):
        kwargs["extensions"] = extension_context
    return factory(**kwargs)


def _build_value(value: Any) -> tuple[Any, dict[str, Any]]:
    """Parse a builder result into ops plus only the fields it actually gave."""
    if isinstance(value, OpSpec):
        return value, {}
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], {"recipes": value[1]}
    if not isinstance(value, Mapping):
        raise ExtensionError("extension_builder_result_requires_ops_or_mapping")
    ops = value.get("ops", value.get("operators"))
    if ops is None:
        raise ExtensionError("extension_builder_result_missing_ops")
    options: dict[str, Any] = {}
    if "recipes" in value:
        recipes = value["recipes"]
        if isinstance(recipes, Mapping):
            from mindie_sim.prediction.program import RecipeBook
            recipes = RecipeBook.from_mapping(recipes)
        options["recipes"] = recipes
    for key in ("external_inputs", "execution", "input_ready_ms"):
        if key in value:
            options[key] = value[key]
    if "replica" in value:
        replica = value["replica"]
        if not isinstance(replica, int) or isinstance(replica, bool) or replica < 0:
            raise ExtensionError("extension_builder_replica_invalid")
        options["replica"] = replica
    return ops, options


def _merge_builder(value: Any, kwargs: Mapping[str, Any]
                   ) -> tuple[Any, dict[str, Any]]:
    ops, builder = _build_value(value)
    merged = dict(kwargs)
    if isinstance(value, Mapping) and "consumed_kwargs" in value:
        consumed = value["consumed_kwargs"]
        if (isinstance(consumed, (Sequence, set, frozenset))
                and not isinstance(consumed, (str, bytes, bytearray))):
            for key in consumed:
                merged.pop(key, None)
    merged.update(builder)
    return ops, merged


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0)


def _report(value: Any, capability: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtensionError("extension_result_not_mapping:" + capability)
    return value


def _require_schema(report: Mapping[str, Any], capability: str, schema: str) -> None:
    if report.get("schema") != schema:
        raise ExtensionError("extension_result_schema_invalid:" + capability
                             + ":requires:" + schema)


def _check_operator(report: Any) -> Mapping[str, Any]:
    report = _report(report, "operator")
    _require_schema(report, "operator", "mindie-sim.operators/1")
    if not _number(report.get("operator_sum_ms")):
        raise ExtensionError("extension_result_missing_metric:operator:operator_sum_ms")
    return report


def _check_graph(report: Any, capability: str = "graph",
                 schema: str = "mindie-sim.graph/1") -> Mapping[str, Any]:
    report = _report(report, capability)
    _require_schema(report, capability, schema)
    if not _number(report.get("duration_ms")):
        raise ExtensionError("extension_result_missing_metric:"
                             + capability + ":duration_ms")
    return report


def _check_model(report: Any) -> Mapping[str, Any]:
    report = _report(report, "model")
    _require_schema(report, "model", "mindie-sim.body-forward/1")
    if not isinstance(report.get("status"), str) or not report["status"]:
        raise ExtensionError("extension_result_missing_status:model")
    value = report.get("body_forward_ms")
    if isinstance(value, Mapping):
        if not value or not all(_number(item) for item in value.values()):
            raise ExtensionError("extension_result_missing_metric:model:body_forward_ms")
    elif not _number(value):
        raise ExtensionError("extension_result_missing_metric:model:body_forward_ms")
    return report


def _check_service(report: Any) -> Mapping[str, Any]:
    report = _report(report, "service")
    _require_schema(report, "service", "mindie-sim.predict/2")
    if not isinstance(report.get("status"), str) or not report["status"]:
        raise ExtensionError("extension_result_missing_status:service")
    if not isinstance(report.get("metrics"), Mapping):
        raise ExtensionError("extension_result_missing_metric:service:metrics")
    return report


def _aclnn_definition(value: Any) -> Any:
    """Return a built aclnn ``CallDefinition``, or ``None`` for OpSpec input."""
    from mindie_sim.prediction.call_builders import CallDefinition, build_call
    if isinstance(value, CallDefinition):
        return value
    if not isinstance(value, Mapping):
        return None
    if "builder" in value:
        if value.get("builder") != "aclnn":
            raise ExtensionError("aclnn_call_requires_aclnn_builder")
        return build_call(value)
    if any(key in value for key in _ACLNN_DECLARATION_KEYS):
        return build_call({"builder": "aclnn", "arguments": dict(value)})
    return None
