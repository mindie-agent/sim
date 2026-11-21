"""Project declared operator records into typed per-value structure facts.

[Role] Preserve existing typed outputs, source/target views, transforms, scale
  associations, semantic attrs and state facts before falling back to the
  compact comparison tuple.
[Invariants] A comparison tuple adds only fields it declares. A lone native
  name is a correspondence, not a selection. Missing layout/shape/ABI becomes a
  named gap; measured shapes never gate recognition.
[Timeouts & Locks] Bounded metadata arithmetic; no I/O, locks or device access.
[Side-Effect Policy] Pure records; no profiler, capture or timing dependency.
"""
import math
import re

from mindie_sim.schema.common import dtype_bytes as _dtype_bytes

from .records import (
    _QUANT_DTYPES,
    _TENSOR_LAYOUTS,
    _common_values,
    _is_tensor,
    _is_tensor_seq,
    _matrix_values,
    _named_tensors,
    _nbytes,
    _scan_tensors,
    _shape,
    _tensor_record,
    _typed_record_value,
    _value,
)
from .layouts import (
    _EXPLICIT_LAYOUT_KINDS,
    _SEMANTIC_ATTR_KEYS,
    _attention_values,
    _audit_inputs,
    _audit_projection,
    _input_ports,
    _layout_hint,
    _layout_value,
    _metadata_boundary,
    _postprocess_compact_layout,
    _producer_inputs,
    _recurrent_values,
    _semantic_attrs,
    _source_layout_producer,
    _source_view_value,
    _state_facts,
    _swiglu_values,
    _tensor_copy_values,
    _view_values,
)
from .projection import (
    typed_values,
    _apply_input_ports,
    _contract_gap,
    _declared_metadata_values,
    _fill_from_produced,
    _merge_produced_outputs,
    _merge_values,
    _quantization,
    _scale_details,
    _scale_values,
)


__all__ = [
    "math",
    "re",
    "typed_values",
]
