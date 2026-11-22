"""Two-Layer Operator SDK and Simulation-vs-Modeling honesty boundary.

[Role] Expose the canonical two-layer operator workflow and enforce the strict
  separation between Layer-2 Simulation Mode (``allow_unbound_formula=False``)
  and Layer-1 Formula-Driven Hardware Modeling Mode (``allow_unbound_formula=True``):
  - Layer 1 (Modeling Mode ONLY): ``FormulaTensorSpec``, ``FormulaKernelSpec``,
    ``FormulaKernelEvaluation``, ``evaluate_formula_kernel`` /
    ``simulate_formula_kernel``, and ``model_target_ceiling``;
  - Layer 2 (Simulation Mode Binding): ``bind_kernel_implementation`` promoting
    a formula specification or custom operator into a real ``Extension`` /
    ``Provider`` bound to a concrete implementation, recipe, or ``aclnn`` call.
[Invariants]
  * Simulation Mode (``simulate_operator``, ``simulate_native_call``,
    ``simulate_aclnn_call``, ``simulate_graph``, ``simulate_model``,
    ``simulate_service``) hard-locks ``allow_unbound_formula=False`` and fails
    fast with ``CapabilityGap(code="unbound_operator_implementation")`` whenever
    an unbound ``FormulaKernelSpec`` or unbound formula ``OpSpec`` is passed
    without a bound Layer-2 provider.
  * Layer-1 formula projections ALWAYS stamp ``binding_state="unbound_formula_projection"``
    and ``has_unbound_formula_ops=True``, and compute latencies purely from
    hardware ``datapath_for_die`` SRAM/pipe limits (zero heuristic multipliers).
  * ``bind_kernel_implementation`` rejects empty/placeholder bindings so Layer 2
    cannot be bypassed without an executable implementation, recipe, or ABI call.
[Timeouts & Locks] Pure bounded in-memory arithmetic; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Pure functional builders and evaluations; zero global
  registry mutation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import re
from typing import Any

from mindie_sim.optimization.kernel_synthesis import (
    FormulaKernelEvaluation,
    FormulaKernelSpec,
    FormulaTensorSpec,
    evaluate_formula_kernel as _canonical_evaluate_formula_kernel,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.hardware import Device, Hardware, parse_hardware
from mindie_sim.schema.operators import OpSpec

from .contracts import Extension, Provider
from .environment import ExecutionEnvironment
from .errors import CapabilityGap, ExtensionError

BINDING_STATE_BOUND = "bound_implementation"
BINDING_STATE_UNBOUND_FORMULA = "unbound_formula_projection"
MODE_MODELING = "modeling"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FULL_HARDWARE_KEYS = ("device", "devices", "dies", "links")


def _resolve_hardware_target(
    hardware: Any,
) -> Case | Hardware | Device | DatapathCapabilitySpec:
    """Resolve ``ExecutionEnvironment`` / SKU string / mapping into canonical hardware input."""
    if isinstance(hardware, (Case, Hardware, Device, DatapathCapabilitySpec)):
        return hardware
    if isinstance(hardware, ExecutionEnvironment):
        return hardware.machine
    if isinstance(hardware, Mapping):
        keys = hardware.keys()
        if {"machine", "model", "workload"} <= keys or {"hardware", "model"} <= keys:
            from mindie_sim.prediction.structure import parse_case
            return parse_case(dict(hardware))
        hardware = hardware.get("hardware", hardware.get("machine", hardware))
        if isinstance(hardware, Mapping) and any(k in hardware for k in _FULL_HARDWARE_KEYS):
            return parse_hardware(dict(hardware))
    if isinstance(hardware, (str, Mapping)):
        # SKU strings / shorthand mappings reuse compose's single machine materializer.
        from mindie_sim.compose import _COMPOSE_OPTION_GROUPS, _materialize_machine
        options = dict.fromkeys(_COMPOSE_OPTION_GROUPS["machine_options"])
        return parse_hardware(_materialize_machine(hardware, required_devices=1, **options)[2])
    raise ExtensionError("formula_kernel_hardware_invalid:" + type(hardware).__name__)


def evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Evaluate a Layer-1 ``FormulaKernelSpec`` directly against hardware SRAM & pipes."""
    if not isinstance(spec, FormulaKernelSpec):
        raise ExtensionError("formula_kernel_requires_FormulaKernelSpec")
    target = _resolve_hardware_target(hardware)
    return _canonical_evaluate_formula_kernel(
        spec, target, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def simulate_formula_kernel(
    spec_or_hardware: Any,
    hardware_or_spec: Any,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Layer-1 Modeling Mode entrypoint accepting ``(spec, hardware)`` or ``(hardware, spec)``."""
    if isinstance(spec_or_hardware, FormulaKernelSpec):
        spec, hardware = spec_or_hardware, hardware_or_spec
    elif isinstance(hardware_or_spec, FormulaKernelSpec):
        hardware, spec = spec_or_hardware, hardware_or_spec
    else:
        raise ExtensionError("simulate_formula_kernel_requires_FormulaKernelSpec")
    return evaluate_formula_kernel(
        spec, hardware, primitive_table=primitive_table,
        active_aic=active_aic, active_aiv=active_aiv,
    )


def _iter_ops_items(ops: Any) -> tuple[Any, ...]:
    if ops is None:
        return ()
    if isinstance(ops, Mapping) and ops and all(isinstance(rank, int) for rank in ops):
        # Rank-indexed graph sets (``{rank: ops}``) are inspected op by op.
        return tuple(op for group in ops.values() for op in _iter_ops_items(group))
    if isinstance(ops, (FormulaKernelSpec, OpSpec, Mapping)):
        return (ops,)
    if isinstance(ops, Sequence) and not isinstance(ops, (str, bytes, bytearray)):
        return tuple(ops)
    return (ops,)


def _is_unbound_op_item(item: Any) -> str | None:
    """Return offending kernel/op name if ``item`` represents an unbound Layer-1 formula op."""
    if isinstance(item, FormulaKernelSpec):
        return item.name or "formula_kernel"
    if isinstance(item, OpSpec):
        meta = dict(item.metadata)
        if (
            str(meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(meta.get("unbound_formula_op"))
        ):
            return item.op_id or item.family
        return None
    if isinstance(item, Mapping):
        meta = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        if (
            str(item.get("binding_state") or meta.get("binding_state") or "") == BINDING_STATE_UNBOUND_FORMULA
            or bool(item.get("unbound_formula_op") or meta.get("unbound_formula_op"))
        ):
            return str(item.get("op_id") or item.get("name") or "unbound_op")
    return None


def enforce_simulation_binding_guard(
    ops: Any,
    *,
    capability: str,
    entry: str,
    allow_unbound_formula: bool = False,
    has_bound_provider: bool = False,
) -> None:
    """Hard-lock Simulation Mode (``allow_unbound_formula=False``) against unbound Layer-1 ops."""
    if allow_unbound_formula:
        raise ExtensionError(f"simulation_mode_forbids_unbound_formula_ops:{entry}")
    if has_bound_provider:
        return
    for item in _iter_ops_items(ops):
        offender = _is_unbound_op_item(item)
        if offender is not None:
            raise CapabilityGap(
                capability=capability,
                entry=entry,
                kind="unmodeled_capability",
                code="unbound_operator_implementation",
                message=(
                    f"unbound_operator_implementation:{offender}: Simulation Mode "
                    "(allow_unbound_formula=False) requires a Layer-2 bound implementation "
                    "registered via bind_kernel_implementation(); for Layer-1 formula-driven "
                    "ceiling projection use simulate_formula_kernel() or model_target_ceiling()."
                ),
                affected_results=("operator_sum_ms", "duration_ms", "timeline"),
                minimal_replay_input={"entry": entry, "unbound_op": offender},
                fallback="not_attempted",
            )


def _strip_unbound_metadata(op: OpSpec, *, kernel_name: str) -> OpSpec:
    meta = dict(op.metadata)
    meta.pop("unbound_formula_op", None)
    meta["binding_state"] = BINDING_STATE_BOUND
    meta.setdefault("bound_kernel_name", kernel_name)
    return replace(op, metadata=meta)


def bind_kernel_implementation(
    spec_or_name: FormulaKernelSpec | str,
    *,
    implementation: Callable[..., Any] | Mapping[str, Any] | None = None,
    extension_id: str | None = None,
    version: str = "1.0.0",
    provider_name: str | None = None,
    capability: str = "operator",
    recipe: Any = None,
    aclnn_api: str | None = None,
    aclnn_symbol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    default: bool = True,
) -> Extension:
    """Layer-2 SDK: bind a formula kernel or custom op to a real ``Extension`` / ``Provider``.

    The returned immutable ``Extension`` can be loaded via ``load_extensions(extensions=[ext])``
    and passed to ``simulate_operator`` / ``simulate_native_call`` / ``simulate_graph`` in
    strict Simulation Mode (``allow_unbound_formula=False``).
    """
    kernel_name = spec_or_name.name if isinstance(spec_or_name, FormulaKernelSpec) else str(spec_or_name or "").strip()
    if not kernel_name:
        raise ExtensionError("bind_kernel_implementation_requires_kernel_name")
    if implementation is None and recipe is None and not (aclnn_api or aclnn_symbol):
        raise ExtensionError(f"bind_kernel_implementation_requires_real_binding:{kernel_name}")

    slug = _SLUG_RE.sub("_", kernel_name.lower()).strip("_") or "custom_kernel"
    ext_id = extension_id or f"bound_{slug}"
    prov_name = provider_name or f"bound_{slug}"
    binding_meta: dict[str, Any] = {
        "binding_state": BINDING_STATE_BOUND,
        "bound_kernel_name": kernel_name,
        "layer": "layer2_bound_implementation",
    }
    if isinstance(spec_or_name, FormulaKernelSpec):
        binding_meta["equations"] = list(spec_or_name.equations)
        binding_meta["abi_signature_id"] = spec_or_name.abi_signature_id or f"bound::{slug}"
    if aclnn_api or aclnn_symbol:
        binding_meta["aclnn_api"] = str(aclnn_api or aclnn_symbol)
    if metadata:
        binding_meta.update(dict(metadata))

    if callable(implementation):
        purpose = "report" if capability in {"operator", "native", "graph"} and recipe is None and not (aclnn_api or aclnn_symbol) else "builder"

        def _callable_factory(*, case: Any, ops: Any, **kwargs: Any) -> Any:
            return implementation(case=case, ops=ops, **kwargs)

        provider = Provider(
            capability=capability,
            name=prov_name,
            factory=_callable_factory,
            purpose=purpose,
            version=version,
            default=default,
            description=f"Layer-2 bound implementation for {kernel_name}",
            metadata=binding_meta,
        )
        return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)

    def _builder_factory(*, case: Any, ops: Any, **kwargs: Any) -> Mapping[str, Any]:
        del case, kwargs
        if isinstance(implementation, Mapping):
            out = dict(implementation)
            if recipe is not None and "recipes" not in out:
                out["recipes"] = recipe
            raw_ops = out.get("ops", out.get("operators", ops))
            out["ops"] = tuple(
                _strip_unbound_metadata(item, kernel_name=kernel_name) if isinstance(item, OpSpec) else item
                for item in _iter_ops_items(raw_ops)
            )
            return out
        if aclnn_api or aclnn_symbol:
            from mindie_sim.prediction.call_builders import build_call
            built = build_call({"builder": "aclnn", "arguments": {"api": str(aclnn_api or aclnn_symbol)}})
            bound_ops = tuple(_strip_unbound_metadata(op, kernel_name=kernel_name) for op in built.ops)
            return {"ops": bound_ops, "recipes": recipe or built.recipes}
        converted: list[OpSpec] = []
        for item in _iter_ops_items(ops):
            if isinstance(item, OpSpec):
                converted.append(_strip_unbound_metadata(item, kernel_name=kernel_name))
            else:
                raise ExtensionError(f"bind_kernel_implementation_requires_op_spec_for_recipe:{kernel_name}")
        return {"ops": tuple(converted), "recipes": recipe}

    provider = Provider(
        capability=capability,
        name=prov_name,
        factory=_builder_factory,
        purpose="builder",
        version=version,
        default=default,
        description=f"Layer-2 recipe/ABI binding for {kernel_name}",
        metadata=binding_meta,
    )
    return Extension(id=ext_id, version=version, providers=(provider,), metadata=binding_meta)


def model_target_ceiling(
    model: Any,
    hardware: Any = "a5",
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """First-class Modeling Mode entrypoint (``allow_unbound_formula=True``).

    Independently solves ``Before (Parallel0, B0)`` (current-stack simulation)
    vs ``After (Parallel1*, B1*)`` (target ceiling modeling with unimplemented
    algebraic kernel synthesis) under the declared SLA and HBM budget.
    """
    from mindie_sim.solve import solve_deployment

    call_kwargs = dict(kwargs)
    call_kwargs["mode"] = MODE_MODELING
    result = solve_deployment(model, hardware, **call_kwargs)
    result["analysis_mode"] = MODE_MODELING
    result["allow_unbound_formula"] = True
    return result


__all__ = [
    "FormulaKernelEvaluation",
    "FormulaKernelSpec",
    "FormulaTensorSpec",
    "bind_kernel_implementation",
    "enforce_simulation_binding_guard",
    "evaluate_formula_kernel",
    "model_target_ceiling",
    "simulate_formula_kernel",
]
