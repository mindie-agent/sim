"""Standalone call declarations, independent of historical capture adapters.

[Role] Resolve extensible call builders for the public CLI, including the
  canonical single-invocation ``aclnn`` ABI declaration.
[Invariants] Builders consume arguments only; no measurements or model labels.
  An ``aclnn`` symbol lowers to one declared physical operator and the physical
  recipe class it names; it never becomes a piecewise replay sidecar, a second
  engine or a duration table.
[Timeouts & Locks] Finite geometry construction; no I/O, locks or device work.
[Side-Effect Policy] Returns a graph, recipe book and interpretation ledger.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math as math

from mindie_sim.schema.common import ModelingError, dtype_bytes as dtype_bytes

from .call_builders_operators import (
    ACLNN_ABI_SCHEMA as ACLNN_ABI_SCHEMA,
    ACLNN_ATTENTION_DTYPES as ACLNN_ATTENTION_DTYPES,
    ACLNN_ATTENTION_KIND as ACLNN_ATTENTION_KIND,
    ACLNN_ATTENTION_KV_DTYPES as ACLNN_ATTENTION_KV_DTYPES,
    ACLNN_ATTENTION_QUERY_DTYPES as ACLNN_ATTENTION_QUERY_DTYPES,
    ACLNN_DEFAULT_DTYPE as ACLNN_DEFAULT_DTYPE,
    ACLNN_GROUPED_KIND as ACLNN_GROUPED_KIND,
    ACLNN_MATRIX_DTYPES as ACLNN_MATRIX_DTYPES,
    ACLNN_PHASES as ACLNN_PHASES,
    ACLNN_VECTOR_DTYPES as ACLNN_VECTOR_DTYPES,
    ACLNN_VECTOR_KIND as ACLNN_VECTOR_KIND,
    ADD_RMSNORM_OPS_PER_ELEM as ADD_RMSNORM_OPS_PER_ELEM,
    ATTENTION_SOFTMAX_OPS_PER_PAIR as ATTENTION_SOFTMAX_OPS_PER_PAIR,
    ATTN_BATCHED_RANK as ATTN_BATCHED_RANK,
    ATTN_UNBATCHED_RANK as ATTN_UNBATCHED_RANK,
    DYNAMIC_QUANT_OPS_PER_ELEM as DYNAMIC_QUANT_OPS_PER_ELEM,
    FLOPS_PER_FMA as FLOPS_PER_FMA,
    MATMUL_SHAPE_DIMS as MATMUL_SHAPE_DIMS,
    MATRIX_MIN_RANK as MATRIX_MIN_RANK,
    RMSNORM_OPS_PER_ELEM as RMSNORM_OPS_PER_ELEM,
    _attention_geometry as _attention_geometry,
    _attention_op as _attention_op,
    _boolean as _boolean,
    _build_matrix as _build_matrix,
    _build_vector as _build_vector,
    _dtype as _dtype,
    _expert_rows as _expert_rows,
    _first_integer as _first_integer,
    _leading_product as _leading_product,
    _mapping as _mapping,
    _matrix_geometry as _matrix_geometry,
    _matrix_metadata as _matrix_metadata,
    _nonnegative_integer as _nonnegative_integer,
    _optional_tensor_dtype as _optional_tensor_dtype,
    _phase as _phase,
    _positive_integer as _positive_integer,
    _quant_spec as _quant_spec,
    _shape as _shape,
    _tensor as _tensor,
    _tensor_dtype as _tensor_dtype,
    _text as _text,
    _vector_geometry as _vector_geometry,
    _vector_work as _vector_work,
)


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


# ---------------------------------------------------------------------------
# Canonical standalone aclnn ABI declarations
# ---------------------------------------------------------------------------
#
# The ABI symbol selects exactly one declared physical operator.  The recipe
# field selects the physical mechanism class (never a measured duration and
# never a piecewise replay plan); the ordinary ``RecipeBook``/Engine path prices
# the declared work.
ACLNN_NAMESPACE = "aclnn"
ACLNN_MATRIX_KIND = "matrix"

#: Entry prefix -> (kind, recipe, family, mechanism class name).  The longest
#: matching prefix wins so ``aclnnMatmul`` cannot swallow another ABI.
ACLNN_ABI_REGISTRY = {
    "aclnnMatmul": (ACLNN_MATRIX_KIND, "aclnn_matmul", "gemm", "SteadyMatmul"),
    "QuantBatchMatmulV3": (ACLNN_MATRIX_KIND, "aclnn_quant_batch_matmul_v3",
                           "gemm", "SteadyMatmul"),
    "aclnnQuantBatchMatmulV3": (ACLNN_MATRIX_KIND, "aclnn_quant_batch_matmul_v3",
                                "gemm", "SteadyMatmul"),
    "aclnnGroupedMatmul": (ACLNN_GROUPED_KIND, "aclnn_grouped_matmul",
                           "grouped_gemm", "ExpectedGroupedMatmul"),
    "aclnnFlashAttentionScore": (ACLNN_ATTENTION_KIND,
                                 "aclnn_flash_attention_score", "attention",
                                 "TiledFlashAndPagedAttentionRecipe"),
    "aclnnFusedInferAttentionScore": (ACLNN_ATTENTION_KIND,
                                      "aclnn_fused_infer_attention_score",
                                      "attention",
                                      "TiledFlashAndPagedAttentionRecipe"),
    "aclnnRmsNorm": (ACLNN_VECTOR_KIND, "aclnn_rms_norm", "rmsnorm",
                     "AIVTiledVectorAndGatherRecipe"),
    "aclnnAddRmsNorm": (ACLNN_VECTOR_KIND, "aclnn_add_rms_norm", "add_rmsnorm",
                        "AIVTiledVectorAndGatherRecipe"),
    "aclnnSwiGlu": (ACLNN_VECTOR_KIND, "aclnn_swiglu", "swiglu",
                    "AIVTiledVectorAndGatherRecipe"),
    "aclnnDynamicQuant": (ACLNN_VECTOR_KIND, "aclnn_dynamic_quant", "quantize",
                          "AIVTiledVectorAndGatherRecipe"),
    "aclnnInplaceCopy": (ACLNN_VECTOR_KIND, "aclnn_inplace_copy", "tensor_copy",
                         "AIVTiledVectorAndGatherRecipe"),
}


def _string_list(value, reason):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModelingError(reason)
    return tuple(str(item) for item in value)


def _symbol(arguments):
    raw = arguments.get("api", arguments.get("symbol"))
    if raw is None:
        raw = arguments.get("aclnn_symbol", arguments.get("entry"))
    if isinstance(raw, Mapping):
        raw = (raw.get("entry") or raw.get("aclnn_symbol")
               or raw.get("symbol"))
    symbol = _text(raw, "aclnn_call_requires_api_symbol")
    if symbol.startswith(ACLNN_NAMESPACE + "."):
        symbol = symbol.split(".", 1)[1]
    return symbol


def resolve_aclnn_abi(symbol):
    """Resolve one ABI entry prefix; longest declared prefix wins."""
    symbol = _text(symbol, "aclnn_call_requires_api_symbol")
    if symbol in ACLNN_ABI_REGISTRY:
        return symbol, ACLNN_ABI_REGISTRY[symbol]
    matches = [entry for entry in ACLNN_ABI_REGISTRY
               if symbol.startswith(entry)]
    if not matches:
        raise ModelingError(f"aclnn_call_api_unsupported:{symbol}")
    entry = max(matches, key=len)
    return entry, ACLNN_ABI_REGISTRY[entry]


def _source(arguments):
    raw = arguments.get("source")
    if raw is None:
        return None
    return _mapping(raw, "aclnn_call_source_requires_mapping")


def _attach_abi(op, arguments, symbol, kind):
    """Carry the ABI identity on the ordinary native-call metadata path."""
    from .native_binding import declare_native_call
    typed = {
        "dtype": op.dtype or None,
        "layout": op.layout or "contiguous",
        "shapes": {"local": list(op.local_shape)},
        "attrs": {"aclnn_kind": kind, "aclnn_symbol": symbol},
    }
    return declare_native_call(
        op, api={"namespace": ACLNN_NAMESPACE, "entry": symbol,
                 "aclnn_symbol": symbol, "version": "1"},
        source=_source(arguments), typed=typed,
        kernels=arguments.get("kernels"))


def _recipes(kind, recipe):
    from .attention_pipeline import TiledFlashAndPagedAttentionRecipe
    from .expected_grouped import ExpectedGroupedMatmul
    from .matrix_steady import SteadyMatmul
    from .program import RecipeBook
    from .vector_pipeline import AIVTiledVectorAndGatherRecipe
    if kind == ACLNN_MATRIX_KIND:
        selected = SteadyMatmul()
    elif kind == ACLNN_GROUPED_KIND:
        selected = ExpectedGroupedMatmul(SteadyMatmul())
    elif kind == ACLNN_ATTENTION_KIND:
        selected = TiledFlashAndPagedAttentionRecipe()
    elif kind == ACLNN_VECTOR_KIND:
        selected = AIVTiledVectorAndGatherRecipe()
    else:  # pragma: no cover - resolved registry entries are exhaustive
        raise ModelingError(f"aclnn_call_kind_unsupported:{kind}")
    return RecipeBook(recipes={recipe: selected}), selected


def build_aclnn_call(arguments):
    """Lower one standalone aclnn ABI invocation to one physical operator.

    The symbol resolves through ``ACLNN_ABI_REGISTRY``; the returned recipe
    book names the physical mechanism class.  There is no piecewise replay
    sidecar and no second timing engine here.
    """
    arguments = _mapping(arguments, "aclnn_call_arguments_requires_mapping")
    symbol = _symbol(arguments)
    entry, (kind, recipe, family, mechanism) = resolve_aclnn_abi(symbol)
    if kind == ACLNN_ATTENTION_KIND:
        op = _attention_op(arguments, symbol, recipe)
    elif kind == ACLNN_VECTOR_KIND:
        op = _build_vector(arguments, symbol, family, recipe, family)
    else:
        op, _groups = _build_matrix(arguments, symbol, kind, recipe, family)
    op = _attach_abi(op, arguments, symbol, kind)
    recipes, selected = _recipes(kind, recipe)
    ledger = {
        "schema": ACLNN_ABI_SCHEMA,
        "api": symbol,
        "resolved_abi": entry,
        "kind": kind,
        "operator": {"op_id": op.op_id, "family": op.family,
                     "recipe": op.recipe, "dtype": op.dtype,
                     "local_shape": list(op.local_shape)},
        "physical_recipe": {"class": type(selected).__name__,
                            "mechanism": mechanism},
        "work": {"flops": op.flops, "vector_ops": op.vector_ops,
                 "hbm_bytes": op.hbm_bytes, "mte2_bytes": op.mte2_bytes,
                 "mte3_bytes": op.mte3_bytes},
        "timed_region": {
            "call": symbol, "starts": "before_abi_dispatch",
            "ends": "after_abi_dispatch_returns_output",
            "input_allocation_and_initialization": "outside",
        },
        "unresolved": [
            "identified_historical_compiled_kernel_binary",
            "native_api_host_dispatch_and_output_allocator_cost",
            "internal_kernel_tiling_and_dispatch",
            "graph_replay_launch_gap_and_runtime_scheduling",
        ],
    }
    if arguments.get("notes") is not None:
        ledger["notes"] = list(_string_list(
            arguments["notes"], "aclnn_call_notes_requires_sequence"))
    limitations = tuple(ledger["unresolved"])
    if arguments.get("limitations") is not None:
        limitations += _string_list(
            arguments["limitations"], "aclnn_call_limitations_requires_sequence")
    return CallDefinition((op,), recipes, ledger, limitations)


def _aclnn(arguments):
    return build_aclnn_call(arguments)


CALL_BUILDERS = {
    "activation": _activation, "gdn": _gdn, "quantized_grouped": _quantized_grouped,
    "quantized_grouped_source": _quantized_grouped_source, "aclnn": _aclnn,
}


def build_call(declaration):
    """Build one callback; repetition/submission belongs to ``call_costs``."""
    declaration = _mapping(declaration, "call_declaration_requires_mapping")
    name = declaration.get("builder")
    if not isinstance(name, str) or name not in CALL_BUILDERS:
        raise ModelingError(f"call_builder_unsupported:{name}")
    arguments = declaration.get("arguments", {})
    return CALL_BUILDERS[name](_mapping(arguments, "call_arguments_requires_mapping"))


__all__ = [
    "ACLNN_ABI_REGISTRY", "ACLNN_ABI_SCHEMA", "CALL_BUILDERS", "CallDefinition",
    "build_aclnn_call", "build_call", "resolve_aclnn_abi",
]
