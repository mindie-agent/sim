"""Thin public calls for already-supported independent simulation levels.

[Role]
  Forward operator, native-call, graph, body-model and service requests to the
  existing business modules.  Loaded extensions may supply one explicit
  provider per capability, but the report contract is unchanged.

[Invariants]
  * No fallback after a selected provider fails: an error is returned as the
    provider's error, never replaced by a cheaper envelope or the core path.
  * ``provider="core"`` is the explicit bypass.  An unqualified request uses
    a declared default or the only provider; several unnamed providers are an
    ambiguity error.  With no provider for the capability the core module runs.
  * Model/service calls own their scope; a provider cannot silently reinterpret
    a body request as serving or vice versa.
  * Provider reports must retain the existing schema and requested metric.
  * A provider receives the context only when it declares an ``extensions``
    parameter; ordinary ``**kwargs`` pass-through must not inherit it.

[Timeouts & Locks]
  Delegates the complete bounded work to the selected existing implementation.

[Side-Effect Policy]
  No I/O, device context or registry mutation; inputs are not modified here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import math
from typing import Any

from mindie_sim.prediction.api import predict
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.body_defaults import TargetOperators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec

from .contracts import CORE_PROVIDER, ExtensionContext
from .environment import ExecutionEnvironment
from .errors import CapabilityGap, ExtensionError
from .profiles import UserProfile


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


def _ops(value: Any) -> tuple[OpSpec, ...]:
    if isinstance(value, OpSpec):
        return (value,)
    try:
        return tuple(value)
    except TypeError as exc:
        raise ExtensionError("operator_requires_op_spec_or_sequence") from exc


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
        values = _ops(ops)
    except TimeoutError:
        raise
    except Exception:  # noqa: BLE001 - replay metadata must never mask the gap
        return {"type": type(ops).__name__}
    return [value.to_dict() for value in values]


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
    """Run one existing core entry; a concrete failure becomes a typed gap.

    Extension declaration/contract errors keep their own type.  No fallback is
    attempted after a modeling failure: the caller receives the exact reason,
    the result fields it invalidates and a minimal input that can be replayed.
    """
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


def _legacy_profiled(case: Any, ops: Any, profile: UserProfile | None,
                     service_id: str | None):
    if profile is None:
        return ops, None
    if isinstance(ops, Mapping):
        raise ExtensionError("user_profile_supports_sequence_ops_only")
    parsed = parse_case(case)
    environment = ExecutionEnvironment.from_hardware(
        parsed.machine, runtime=parsed.runtime)
    probe = environment.execution(recipes=None)
    normalized, recipes = environment._profiled(
        _ops(ops), profile, service_id, probe)
    if isinstance(ops, TargetOperators):
        normalized = TargetOperators(
            normalized, kernels=ops.effective_kernels,
            selection=ops.default_selection)
    return normalized, recipes


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
    # Positional-only: request keywords such as model geometry ``context=0``
    # must not collide with the extension context parameter.
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
    """Merge caller and builder options once; builder fields win only if given."""
    ops, builder = _build_value(value)
    merged = dict(kwargs)
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


def simulate_operator(case: Any, ops: Any, *, extensions: Any = None,
                      provider: str | None = None,
                      profile: Any = None, service_id: str | None = None,
                      **kwargs: Any) -> dict[str, Any]:
    """Price isolated operators from a Case or declared hardware environment.

    Hardware/``ExecutionEnvironment`` input needs no model, request or workload:
    the existing Engine executes the declared OpSpec work.  A full legacy Case
    remains accepted through ``operator_costs``.  A user profile is a T06
    resource/phase service, never a whole-operator duration lookup.
    """
    context = _context(extensions)
    selected = _provider("operator", provider, context)
    environment = _environment(case)
    selected_profile = _profile(profile)
    replay = {"entry": "simulate_operator", "ops": _ops_replay(ops),
              "environment": _machine_reference(environment)}
    affected = ("operator_sum_ms", "timeline", "operators")
    profile_kwargs = {}
    if selected_profile is not None:
        profile_kwargs["profile"] = selected_profile
    if service_id is not None:
        profile_kwargs["service_id"] = service_id
    if selected is not None and selected.purpose == "builder":
        built = _invoke(
            selected.factory, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs)
        built_ops, merged = _merge_builder(built, kwargs)
        if environment is not None:
            return _core(
                "operator", "simulate_operator", lambda: environment.operator_report(
                    _ops(built_ops), profile=selected_profile,
                    service_id=service_id,
                    input_ready_ms=merged.get("input_ready_ms", 0),
                    recipes=merged.get("recipes"),
                    replica=merged.get("replica", 0)),
                affected_results=affected, minimal_replay_input=replay)
        normalized, recipes = _legacy_profiled(
            case, built_ops, selected_profile, service_id)
        if merged.get("recipes") is not None:
            recipes = merged["recipes"]
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        return _core(
            "operator", "simulate_operator", lambda: operator_costs(
                case, normalized if recipes is not None else built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    handler = selected.factory if selected is not None else None
    if handler is not None:
        return dict(_check_operator(_invoke(
            handler, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs)))
    if environment is not None:
        input_ready_ms = kwargs.pop("input_ready_ms", 0)
        recipes = kwargs.pop("recipes", None)
        replica = kwargs.pop("replica", 0)
        if kwargs:
            raise ExtensionError(
                "standalone_operator_unsupported_keywords:"
                + ",".join(sorted(kwargs)))
        return _core(
            "operator", "simulate_operator", lambda: environment.operator_report(
                _ops(ops), profile=selected_profile, service_id=service_id,
                input_ready_ms=input_ready_ms, recipes=recipes, replica=replica),
            affected_results=affected, minimal_replay_input=replay)
    normalized, recipes = _legacy_profiled(case, ops, selected_profile, service_id)
    if recipes is not None:
        return _core(
            "operator", "simulate_operator",
            lambda: operator_costs(case, normalized, recipes=recipes, **kwargs),
            affected_results=affected, minimal_replay_input=replay)
    return _core(
        "operator", "simulate_operator", lambda: operator_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_native_call(case: Any, ops: Any, *, extensions: Any = None,
                         provider: str | None = None, profile: Any = None,
                         service_id: str | None = None,
                         **kwargs: Any) -> dict[str, Any]:
    """Compose an already-declared native call through the ordinary engine.

    The core implementation is ``mindie_sim.call_costs`` for a full Case, or
    the declared-work execution context when no model/workload exists.  A
    selected ``native`` provider receives the same boundary; failures are not
    downgraded to a semantic-only approximation or to a whole-call target.
    """
    context = _context(extensions)
    selected = _provider("native", provider, context)
    environment = _environment(case)
    selected_profile = _profile(profile)
    replay = {"entry": "simulate_native_call", "ops": _ops_replay(ops),
              "environment": _machine_reference(environment)}
    affected = ("duration_ms", "timeline", "physical_operators")
    profile_kwargs = {}
    if selected_profile is not None:
        profile_kwargs["profile"] = selected_profile
    if service_id is not None:
        profile_kwargs["service_id"] = service_id
    if selected is not None and selected.purpose == "builder":
        built = _invoke(
            selected.factory, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs)
        built_ops, merged = _merge_builder(built, kwargs)
        if environment is not None:
            options = dict(merged)
            input_ready_ms = options.pop("input_ready_ms", 0)
            recipes = options.pop("recipes", None)
            replica = options.pop("replica", 0)
            options.pop("execution", None)
            outside = set(options) - {"invocations", "mode", "max_operators"}
            if outside:
                raise ExtensionError(
                    "standalone_native_unsupported_keywords:"
                    + ",".join(sorted(outside)))
            return _core(
                "native", "simulate_native_call",
                lambda: environment.call_report(
                    _ops(built_ops), profile=selected_profile,
                    service_id=service_id, input_ready_ms=input_ready_ms,
                    recipes=recipes, replica=replica, **options),
                affected_results=affected, minimal_replay_input=replay)
        normalized, recipes = _legacy_profiled(
            case, built_ops, selected_profile, service_id)
        if merged.get("recipes") is not None:
            recipes = merged["recipes"]
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        return _core(
            "native", "simulate_native_call", lambda: call_costs(
                case, normalized if recipes is not None else built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    handler = selected.factory if selected is not None else None
    if handler is not None:
        return dict(_check_graph(_invoke(
            handler, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs),
            capability="native", schema="mindie-sim.call-costs/1"))
    if environment is not None:
        input_ready_ms = kwargs.pop("input_ready_ms", 0)
        recipes = kwargs.pop("recipes", None)
        replica = kwargs.pop("replica", 0)
        outside = set(kwargs) - {"invocations", "mode", "max_operators"}
        if outside:
            raise ExtensionError(
                "standalone_native_unsupported_keywords:"
                + ",".join(sorted(outside)))
        return _core(
            "native", "simulate_native_call", lambda: environment.call_report(
                _ops(ops), profile=selected_profile, service_id=service_id,
                input_ready_ms=input_ready_ms, recipes=recipes,
                replica=replica, **kwargs),
            affected_results=affected, minimal_replay_input=replay)
    normalized, recipes = _legacy_profiled(case, ops, selected_profile, service_id)
    if recipes is not None:
        return _core(
            "native", "simulate_native_call",
            lambda: call_costs(case, normalized, recipes=recipes, **kwargs),
            affected_results=affected, minimal_replay_input=replay)
    return _core(
        "native", "simulate_native_call", lambda: call_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_graph(case: Any, ops: Any, *, extensions: Any = None,
                   provider: str | None = None, profile: Any = None,
                   service_id: str | None = None,
                   **kwargs: Any) -> dict[str, Any]:
    """Compose a declared dependency graph on shared device resources.

    A hardware/``ExecutionEnvironment`` input executes declared OpSpec nodes
    without a model/request Case.  A full legacy Case still uses
    ``graph_costs``.  An extension supplies a selected ``graph`` provider,
    never a second scheduler.
    """
    context = _context(extensions)
    selected = _provider("graph", provider, context)
    environment = _environment(case)
    selected_profile = _profile(profile)
    replay = {"entry": "simulate_graph", "ops": _ops_replay(ops),
              "environment": _machine_reference(environment)}
    affected = ("duration_ms", "timeline", "operator_occurrences")
    profile_kwargs = {}
    if selected_profile is not None:
        profile_kwargs["profile"] = selected_profile
    if service_id is not None:
        profile_kwargs["service_id"] = service_id
    if selected is not None and selected.purpose == "builder":
        built = _invoke(
            selected.factory, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs)
        built_ops, merged = _merge_builder(built, kwargs)
        if environment is not None:
            return _core(
                "graph", "simulate_graph", lambda: environment.graph_report(
                    _ops(built_ops), profile=selected_profile,
                    service_id=service_id,
                    input_ready_ms=merged.get("input_ready_ms", 0),
                    external_inputs=merged.get("external_inputs"),
                    execution=merged.get("execution"),
                    replica=merged.get("replica", 0),
                    recipes=merged.get("recipes")),
                affected_results=affected, minimal_replay_input=replay)
        normalized, recipes = _legacy_profiled(
            case, built_ops, selected_profile, service_id)
        if merged.get("recipes") is not None:
            recipes = merged["recipes"]
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        call_kwargs.pop("replica", None)
        return _core(
            "graph", "simulate_graph", lambda: graph_costs(
                case, normalized if recipes is not None else built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    handler = selected.factory if selected is not None else None
    if handler is not None:
        return dict(_check_graph(_invoke(
            handler, context, case=environment or case, ops=ops,
            **profile_kwargs, **kwargs)))
    if environment is not None:
        input_ready_ms = kwargs.pop("input_ready_ms", 0)
        external_inputs = kwargs.pop("external_inputs", None)
        execution = kwargs.pop("execution", None)
        replica = kwargs.pop("replica", 0)
        recipes = kwargs.pop("recipes", None)
        if kwargs:
            raise ExtensionError(
                "standalone_graph_unsupported_keywords:"
                + ",".join(sorted(kwargs)))
        return _core(
            "graph", "simulate_graph", lambda: environment.graph_report(
                _ops(ops), profile=selected_profile, service_id=service_id,
                input_ready_ms=input_ready_ms, external_inputs=external_inputs,
                execution=execution, replica=replica, recipes=recipes),
            affected_results=affected, minimal_replay_input=replay)
    normalized, recipes = _legacy_profiled(case, ops, selected_profile, service_id)
    if recipes is not None:
        return _core(
            "graph", "simulate_graph",
            lambda: graph_costs(case, normalized, recipes=recipes, **kwargs),
            affected_results=affected, minimal_replay_input=replay)
    return _core(
        "graph", "simulate_graph", lambda: graph_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_model(case: Any, *, extensions: Any = None,
                   provider: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate the independent target-body forward program.

    The core implementation is ``predict(..., scope="body")``.  The scope is
    owned here so a provider cannot turn this entry into a serving request.
    """
    if "scope" in kwargs:
        raise ExtensionError("simulate_model_owns_scope")
    context = _context(extensions)
    selected = _provider("model", provider, context)
    replay = {"entry": "simulate_model", "case": _case_replay(case),
              "geometry": {key: kwargs[key] for key in
                           ("phase", "tokens", "context", "sequences",
                            "event_limit") if key in kwargs}}
    affected = ("body_forward_ms", "timeline", "execution")
    handler = selected.factory if selected is not None else None
    if selected is not None and selected.purpose == "builder":
        built = _invoke(handler, context, case=case, **kwargs)
        if isinstance(built, Mapping) and "case" in built:
            if "recipes" in built:
                kwargs["recipes"] = built["recipes"]
            case = built["case"]
        elif built is not None:
            case = built
        else:
            raise ExtensionError("extension_model_builder_missing_case")
        handler = None
    if handler is None:
        if isinstance(case, Mapping) and "model" in case \
                and "workload" not in case:
            from mindie_sim.prediction.structure import parse_model
            case = parse_model(case)
        return _core(
            "model", "simulate_model",
            lambda: predict(case, scope="body", **kwargs),
            affected_results=affected, minimal_replay_input=replay)
    return dict(_check_model(_invoke(
        handler, context, case=case, scope="body", **kwargs)))


def simulate_service(case: Any, *, extensions: Any = None,
                     provider: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate the independent serving request and capacity path.

    The core implementation is ``predict`` with serving scope.  A ``service``
    provider must return the same committed-event metrics contract.
    """
    if "scope" in kwargs:
        raise ExtensionError("simulate_service_owns_scope")
    context = _context(extensions)
    selected = _provider("service", provider, context)
    replay = {"entry": "simulate_service", "case": _case_replay(case),
              "geometry": {key: kwargs[key] for key in
                           ("event_limit", "max_model_steps", "trace_limit")
                           if key in kwargs}}
    affected = ("metrics", "requests", "execution")
    handler = selected.factory if selected is not None else None
    if selected is not None and selected.purpose == "builder":
        built = _invoke(handler, context, case=case, **kwargs)
        if isinstance(built, Mapping) and "case" in built:
            if "recipes" in built:
                kwargs["recipes"] = built["recipes"]
            case = built["case"]
        elif built is not None:
            case = built
        else:
            raise ExtensionError("extension_service_builder_missing_case")
        handler = None
    if handler is None:
        result = _core(
            "service", "simulate_service",
            lambda: predict(case, **kwargs),
            affected_results=affected, minimal_replay_input=replay)
        result["production_entry"] = "simulate_service"
        return result
    result = dict(_check_service(_invoke(
        handler, context, case=case, scope="serving", **kwargs)))
    result["production_entry"] = "simulate_service"
    return result


__all__ = [
    "simulate_operator", "simulate_native_call",
    "simulate_graph", "simulate_model", "simulate_service",
]
