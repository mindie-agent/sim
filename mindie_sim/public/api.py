"""Thin public calls for independent simulation levels plus canonical typed SDK.

[Role]
  Forward operator, native-call, graph, body-model and service requests to the
  existing business modules while re-exporting the seven canonical typed SDK
  functions from ``mindie_sim.api`` as identical function objects.
[Invariants]
  * No fallback after a selected provider fails: an error is returned as the
    provider's error, never replaced by a cheaper envelope or the core path.
  * ``provider="core"`` is the explicit bypass.  An unqualified request uses
    a declared default or the only provider; several unnamed providers are an
    ambiguity error.  With no provider for the capability the core module runs.
  * Model/service calls own their scope; a provider cannot silently reinterpret
    a body request as serving or vice versa.
  * Provider reports retain the existing schema and requested metric.
[Timeouts & Locks]
  Delegates the complete bounded work to the selected existing implementation.
[Side-Effect Policy]
  No I/O, device context or registry mutation; inputs are not modified here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.api import (
    compose_case,
    evaluate_service,
    evaluate_step,
    evaluate_three_tiers,
    export_structure,
    solve_sla,
    sweep_topology,
)
from mindie_sim.prediction.api import predict
from mindie_sim.prediction.call_builders import call_costs
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs

from .api_support import (
    _ACLNN_DECLARATION_KEYS,
    _ACLNN_PROTOCOL_KEYS,
    _accepts_context,
    _aclnn_definition,
    _build_value,
    _case_replay,
    _check_graph,
    _check_model,
    _check_operator,
    _check_service,
    _context,
    _core,
    _environment,
    _gap_kind,
    _handler,
    _invoke,
    _machine_reference,
    _merge_builder,
    _number,
    _ops,
    _ops_replay,
    _profile,
    _provider,
    _report,
    _require_schema,
)
from .errors import ExtensionError
from .formula_sdk import (
    FormulaKernelEvaluation,
    FormulaKernelSpec,
    FormulaTensorSpec,
    bind_kernel_implementation,
    enforce_simulation_binding_guard,
    evaluate_formula_kernel,
    model_target_ceiling,
    simulate_formula_kernel,
)


def simulate_operator(case: Any, ops: Any, *, extensions: Any = None,
                      provider: str | None = None,
                      profile: Any = None, service_id: str | None = None,
                      **kwargs: Any) -> dict[str, Any]:
    """Price isolated operators from a Case or declared hardware environment."""
    allow_unbound = bool(kwargs.pop("allow_unbound_formula", False))
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
        enforce_simulation_binding_guard(
            built_ops, capability="operator", entry="simulate_operator",
            allow_unbound_formula=allow_unbound, has_bound_provider=False,
        )
        if environment is not None:
            return _core(
                "operator", "simulate_operator", lambda: environment.operator_report(
                    _ops(built_ops), profile=selected_profile,
                    service_id=service_id,
                    input_ready_ms=merged.get("input_ready_ms", 0),
                    recipes=merged.get("recipes"),
                    replica=merged.get("replica", 0)),
                affected_results=affected, minimal_replay_input=replay)
        if selected_profile is not None:
            raise NotImplementedError(
                "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
                "in public/api.py:simulate_operator (S7-SG-05)"
            )
        recipes = merged.get("recipes")
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        return _core(
            "operator", "simulate_operator", lambda: operator_costs(
                case, built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    enforce_simulation_binding_guard(
        ops, capability="operator", entry="simulate_operator",
        allow_unbound_formula=allow_unbound, has_bound_provider=(selected is not None),
    )
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
    if selected_profile is not None:
        raise NotImplementedError(
            "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
            "in public/api.py:simulate_operator (S7-SG-05)"
        )
    return _core(
        "operator", "simulate_operator", lambda: operator_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_native_call(case: Any, ops: Any, *, extensions: Any = None,
                         provider: str | None = None, profile: Any = None,
                         service_id: str | None = None,
                         **kwargs: Any) -> dict[str, Any]:
    """Compose an already-declared native call through the ordinary engine."""
    allow_unbound = bool(kwargs.pop("allow_unbound_formula", False))
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
        enforce_simulation_binding_guard(
            built_ops, capability="native", entry="simulate_native_call",
            allow_unbound_formula=allow_unbound, has_bound_provider=False,
        )
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
        if selected_profile is not None:
            raise NotImplementedError(
                "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
                "in public/api.py:simulate_native_call (S7-SG-05)"
            )
        recipes = merged.get("recipes")
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        return _core(
            "native", "simulate_native_call", lambda: call_costs(
                case, built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    enforce_simulation_binding_guard(
        ops, capability="native", entry="simulate_native_call",
        allow_unbound_formula=allow_unbound, has_bound_provider=(selected is not None),
    )
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
    if selected_profile is not None:
        raise NotImplementedError(
            "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
            "in public/api.py:simulate_native_call (S7-SG-05)"
        )
    return _core(
        "native", "simulate_native_call", lambda: call_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_aclnn_call(case: Any, call: Any = None, *, ops: Any = None,
                        declaration: Any = None, api: str | None = None,
                        arguments: Any = None, tensors: Any = None,
                        attrs: Any = None, extensions: Any = None,
                        provider: str | None = None, profile: Any = None,
                        service_id: str | None = None,
                        **kwargs: Any) -> dict[str, Any]:
    """Price one standalone aclnn ABI invocation on the canonical call path."""
    if kwargs.pop("allow_unbound_formula", False):
        raise ExtensionError("simulation_mode_forbids_unbound_formula_ops:simulate_aclnn_call")
    if isinstance(call, str):
        if api is not None:
            raise ExtensionError("aclnn_call_api_and_symbol_ambiguous")
        api, call = call, None
    if ops is not None:
        if call is not None:
            raise ExtensionError("aclnn_call_ops_and_call_ambiguous")
        call = ops
    if declaration is not None:
        if call is not None:
            raise ExtensionError("aclnn_call_declaration_and_call_ambiguous")
        call = declaration
    if call is None:
        if api is None:
            raise ExtensionError("aclnn_call_requires_api_or_declaration")
        if not isinstance(arguments, Mapping) and arguments is not None:
            raise ExtensionError("aclnn_call_arguments_requires_mapping")
        payload = dict(arguments or {})
        payload.update({key: kwargs.pop(key) for key in list(kwargs)
                        if key not in _ACLNN_PROTOCOL_KEYS})
        payload["api"] = api
        if tensors is not None:
            payload["tensors"] = tensors
        if attrs is not None:
            payload["attrs"] = attrs
        definition = _aclnn_definition(payload)
    else:
        if api is not None or arguments is not None or tensors is not None \
                or attrs is not None:
            raise ExtensionError("aclnn_call_argument_source_ambiguous")
        definition = _aclnn_definition(call)
    if definition is not None:
        call = definition.ops
        kwargs.setdefault("recipes", definition.recipes)
    return simulate_native_call(
        case, call, extensions=extensions, provider=provider, profile=profile,
        service_id=service_id, **kwargs)


def simulate_graph(case: Any, ops: Any, *, extensions: Any = None,
                   provider: str | None = None, profile: Any = None,
                   service_id: str | None = None,
                   **kwargs: Any) -> dict[str, Any]:
    """Compose a declared dependency graph on shared device resources."""
    allow_unbound = bool(kwargs.pop("allow_unbound_formula", False))
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
        enforce_simulation_binding_guard(
            built_ops, capability="graph", entry="simulate_graph",
            allow_unbound_formula=allow_unbound, has_bound_provider=False,
        )
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
        if selected_profile is not None:
            raise NotImplementedError(
                "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
                "in public/api.py:simulate_graph (S7-SG-05)"
            )
        recipes = merged.get("recipes")
        call_kwargs = dict(merged)
        call_kwargs.pop("recipes", None)
        call_kwargs.pop("execution", None)
        call_kwargs.pop("replica", None)
        return _core(
            "graph", "simulate_graph", lambda: graph_costs(
                case, built_ops,
                **({"recipes": recipes} if recipes is not None else {}),
                **call_kwargs),
            affected_results=affected, minimal_replay_input=replay)
    enforce_simulation_binding_guard(
        ops, capability="graph", entry="simulate_graph",
        allow_unbound_formula=allow_unbound, has_bound_provider=(selected is not None),
    )
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
    if selected_profile is not None:
        raise NotImplementedError(
            "LEGACY_PURGED: wire unified ExecutionEnvironment profile binding "
            "in public/api.py:simulate_graph (S7-SG-05)"
        )
    return _core(
        "graph", "simulate_graph", lambda: graph_costs(case, ops, **kwargs),
        affected_results=affected, minimal_replay_input=replay)


def simulate_model(case: Any, *, extensions: Any = None,
                   provider: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate the independent target-body forward program."""
    if kwargs.pop("allow_unbound_formula", False):
        raise ExtensionError("simulation_mode_forbids_unbound_formula_ops:simulate_model")
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
    """Evaluate the independent serving request and capacity path."""
    if kwargs.pop("allow_unbound_formula", False):
        raise ExtensionError("simulation_mode_forbids_unbound_formula_ops:simulate_service")
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
    "compose_case", "evaluate_step", "evaluate_three_tiers",
    "evaluate_service", "solve_sla", "sweep_topology", "export_structure",
    "simulate_operator", "simulate_native_call", "simulate_aclnn_call",
    "simulate_graph", "simulate_model", "simulate_service",
    "FormulaTensorSpec", "FormulaKernelSpec", "FormulaKernelEvaluation",
    "evaluate_formula_kernel", "simulate_formula_kernel",
    "bind_kernel_implementation", "model_target_ceiling",
]
