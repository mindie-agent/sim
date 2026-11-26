"""Compatibility exports for implementation selection and typed graph helpers.

[Role] Keep the original import and patch points available after splitting the
  implementation registry, typed values, consumer uses, rewrites and audit.
[Invariants] Legacy names share their original objects with the owning modules.
  Setting or deleting a legacy attribute reaches the modules that use it.
[Timeouts & Locks] Local imports only. No subprocesses or locks.
[Side-Effect Policy] Re-export local helpers. Legacy patches update their
  imported bindings. Importing this module starts no device work.
"""
from dataclasses import dataclass as dataclass, replace as replace
from importlib import import_module as import_module
import math as math
import sys as _sys
from types import ModuleType as _ModuleType

from mindie_sim.schema.common import ModelingError as ModelingError, dtype_bytes as dtype_bytes
from mindie_sim.schema.operators import FieldDisposition as FieldDisposition
from mindie_sim.schema.tensors import TensorLayout as TensorLayout, TensorValue as TensorValue

import mindie_sim.prediction.program.implementation_audit as _audit
import mindie_sim.prediction.program.implementation_registry as _registry
from .values import typed as _typed
from .values import rewrite as _rewrite
from .values import uses as _uses
from .program.implementation_audit import (
    _audit_outputs as _audit_outputs,
    append_audit as append_audit,
    audit_boundary as audit_boundary,
    audit_records as audit_records,
    capture_fusion_overlap as capture_fusion_overlap,
    capture_partition as capture_partition,
    field_disposition as field_disposition,
    input_ports as input_ports,
    member_contracts as member_contracts,
    nested_boundary as nested_boundary,
    replaced_occurrences as replaced_occurrences,
    semantic_attrs as semantic_attrs,
)
from .program.implementation_registry import (
    IMPLEMENTATIONS as IMPLEMENTATIONS,
    Implementation as Implementation,
    Selection as Selection,
    implementation as implementation,
    lower_operators as lower_operators,
    plan_record as plan_record,
    recipes as recipes,
    select as select,
)
from .values.typed import (
    _contiguous_strides as _contiguous_strides,
    _declared_output_geometry as _declared_output_geometry,
    _json_identity as _json_identity,
    _normalize_output as _normalize_output,
    _primary_output_name as _primary_output_name,
    _row_geometry as _row_geometry,
    _row_shape as _row_shape,
    first_value as first_value,
    merge_typed_roles as merge_typed_roles,
    produced_values as produced_values,
    typed_outputs as typed_outputs,
    typed_roles as typed_roles,
    typed_value as typed_value,
)
from .values.rewrite import (
    _SINGLE_DEP as _SINGLE_DEP,
    _move_access_slots as _move_access_slots,
    add_input_ports as add_input_ports,
    attach_quantized_ports as attach_quantized_ports,
    preserve_graph as preserve_graph,
    rewire_replacement as rewire_replacement,
    rewire_replacements as rewire_replacements,
    rewire_value_graph as rewire_value_graph,
    set_input_ports as set_input_ports,
)
from .values.uses import (
    _FIRST_SLOT as _FIRST_SLOT,
    _LAST_ENTRY as _LAST_ENTRY,
    _PRODUCER_FIELDS as _PRODUCER_FIELDS,
    _access_bytes as _access_bytes,
    _op_producer_refs as _op_producer_refs,
    _select_named_value as _select_named_value,
    consumer_index as consumer_index,
    consumer_value_uses as consumer_value_uses,
    external_value_plan as external_value_plan,
    producer_input_values as producer_input_values,
)

# Legacy monkeypatches used to modify the globals seen by these helpers. Keep
# the same attribute patch points while normal calls use direct local bindings.
_namespaces = tuple(vars(module) for module in (_audit, _registry, _typed, _rewrite, _uses))
_patch_namespaces = {
    name: tuple(namespace for namespace in _namespaces if name in namespace)
    for name in tuple(globals())
    if not name.startswith("__")
}


class _ImplementationFacade(_ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for namespace in _patch_namespaces.get(name, ()):
            namespace[name] = value

    def __delattr__(self, name):
        super().__delattr__(name)
        for namespace in _patch_namespaces.get(name, ()):
            namespace.pop(name, None)


_sys.modules[__name__].__class__ = _ImplementationFacade
