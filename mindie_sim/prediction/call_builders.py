"""Standalone call declarations, independent of historical capture adapters.

[Role] Resolve extensible call builders for the public CLI.
[Invariants] Builders consume arguments only; no measurements or model labels.
[Timeouts & Locks] Finite geometry construction; no I/O, locks or device work.
[Side-Effect Policy] Returns a graph, recipe book and interpretation ledger.
"""
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class CallDefinition:
    ops: tuple
    recipes: object
    ledger: dict
    limitations: tuple


def _activation(arguments):
    from .calls_activation import ActivationCallSpec, build_activation_call
    call = build_activation_call(ActivationCallSpec(**arguments))
    return CallDefinition(call.ops, call.recipes, call.coverage,
                          tuple(call.coverage["unresolved"]))


def _gdn(arguments):
    from .call_gdn import make_call
    call = make_call(arguments["role"], {key: value for key, value in arguments.items()
                                        if key != "role"})
    return CallDefinition(call.ops, call.recipes, call.ledger, call.limitations)


def _quantized_grouped(arguments):
    from .call_grouped import build_quantized_grouped_call
    call = build_quantized_grouped_call(arguments)
    return CallDefinition(call.ops, call.recipes, call.ledger, call.limitations)


def _quantized_grouped_source(arguments):
    from .call_grouped_source import build_quantized_grouped_source_call
    return build_quantized_grouped_source_call(arguments)


CALL_BUILDERS = {
    "activation": _activation, "gdn": _gdn, "quantized_grouped": _quantized_grouped,
    "quantized_grouped_source": _quantized_grouped_source,
}


def build_call(declaration):
    """Build one callback; repetition/submission belongs to ``call_costs``."""
    name = declaration["builder"]
    if name not in CALL_BUILDERS:
        raise ModelingError(f"call_builder_unsupported:{name}")
    return CALL_BUILDERS[name](declaration["arguments"])
