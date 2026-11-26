"""Operator call builders, activation calls and direct call-cost evaluation.

[Role] Construct standalone operator call graphs and evaluate call costs
  through the canonical runtime Engine.
[Invariants] Preserves exact tensor shapes, dtypes, and operator binding contracts.
[Timeouts & Locks] Pure simulation; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError
from .graph_analysis import graph_costs
from .program import RecipeBook
from .structure import parse_case

_CALL_OPERATOR_BUDGET = 8192


def call_costs(case, ops, *, recipes=None, invocations=1, mode="eager",
               include_submission=False, submission=None, input_ready_ms=0,
               max_operators=_CALL_OPERATOR_BUDGET, trace_limit=0):
    """Compose timed invocations, returning a device/submitted graph boundary.

    ``graph`` means one replay containing all invocations; ``eager`` submits
    each expanded kernel/API node separately. This models the declared queue,
    not Python allocator behavior, event-record overhead or unknown dispatch.
    """
    if (not isinstance(invocations, int) or isinstance(invocations, bool)
            or invocations < 1 or max_operators < 1):
        raise ModelingError("call_requires_positive_invocations_and_operator_budget")
    if mode not in {"eager", "graph"}:
        raise ModelingError("call_mode_requires_eager_or_graph")
    submission = dict(submission or {})
    outside = submission.keys() - {"graph_replay_ms"}
    if outside:
        raise ModelingError("call_submission_outside_callback_boundary:" + ",".join(sorted(outside)))
    replay = submission.get("graph_replay_ms")
    if replay is not None and (mode != "graph" or not math.isfinite(replay) or replay < 0):
        raise ModelingError("call_replay_service_requires_graph_mode_and_nonnegative_finite_time")
    case = parse_case(case)
    case = replace(case, runtime=replace(
        case.runtime, submission=dict(submission or {}), async_scheduling=False))
    book = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    single = tuple(book.expand(tuple(ops)))
    if len(single) * invocations > max_operators:
        raise ModelingError("call_expanded_operator_budget_exceeded")
    identifiers = {op.op_id for op in single}
    if len(identifiers) != len(single):
        raise ModelingError("call_duplicate_operator_id")
    if any(set(op.deps) - identifiers for op in single):
        raise ModelingError("call_requires_complete_callback_dependencies")
    repeated_list = []
    stream_tail: dict[tuple[int, str], str] = {}
    for index in range(invocations):
        for op in single:
            new_id = f"call{index}/{op.op_id}"
            new_deps = [f"call{index}/{dep}" for dep in op.deps]
            stream_key = (op.stage, op.stream or "main")
            prev_on_stream = stream_tail.get(stream_key)
            if prev_on_stream is not None and prev_on_stream not in new_deps:
                new_deps.append(prev_on_stream)
            stream_tail[stream_key] = new_id
            repeated_list.append(replace(op, op_id=new_id, deps=tuple(new_deps)))
    repeated = tuple(repeated_list)
    report = graph_costs(
        case, repeated, recipes=book, input_ready_ms=input_ready_ms,
        include_submission=include_submission,
        replay_scope="full" if mode == "graph" else "none",
        isolated=False, trace_limit=trace_limit,
    )
    return {
        "schema": "mindie-sim.call-costs/1",
        "boundary": report["boundary"], "duration_ms": report["duration_ms"],
        "input_ready_ms": input_ready_ms, "complete_ms": report["complete_ms"],
        "protocol": {"invocations": invocations, "mode": mode,
                     "include_submission": include_submission,
                     "submission": dict(submission or {})},
        "single_call": {
            "expanded_operators": len(single),
            "families": dict(Counter(op.family for op in single)),
            "source_boundaries": dict(Counter(op.source_boundary or "analytical"
                                              for op in single)),
        },
        "operator_occurrences": report["operator_occurrences"],
        "execution": report["execution"], "resources": report["resources"],
        "parameters": report["parameters"], "events": report["events"],
        "assumptions": [
            "A graph protocol is one replay containing every declared invocation.",
            "Eager nodes queue on their declared streams; host submission can overlap device work.",
            "Repeated calls preserve per-stream order; no implicit cross-stream callback fence is added.",
            "Input preparation and warmup outside the timed callback are excluded.",
            "Undeclared allocator, Python dispatch and event-record costs are not inferred.",
            "Device/submitted graph boundaries are not relabeled as observed event spans.",
        ],
    }


from dataclasses import dataclass

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .program import RecipeBook


# API topology and expression work, independent of measured event durations.
# SiLU x/(1+exp(-x)): negate, exp, add, divide. Tanh GELU uses two
# multiplications for x^3, three scalar products, two additions, tanh and
# the final product. All count as nominal vector-lane operations here.
ACTIVATION_CALLS = {
    "swiglu_fused": {
        "api": ("torch_npu.npu_swiglu(dim=-1)",),
        "layout": "contiguous_gate_first",
        "steps": (("swiglu", 5, 2),),
    },
    "swiglu_separate": {
        "api": ("torch.nn.functional.silu", "Tensor.mul"),
        "layout": "separate_contiguous_gate_up",
        "steps": (("silu", 4, 1), ("multiply", 1, 2)),
    },
    "geglu_separate": {
        "api": ("torch.nn.functional.gelu(approximate=tanh)", "Tensor.mul"),
        "layout": "separate_contiguous_gate_up",
        "steps": (("gelu_tanh", 9, 1), ("multiply", 1, 2)),
    },
    "silu_standalone": {
        "api": ("torch.nn.functional.silu",),
        "layout": "contiguous",
        "steps": (("silu", 4, 1),),
    },
    "gelu_standalone": {
        "api": ("torch.nn.functional.gelu(approximate=tanh)",),
        "layout": "contiguous",
        "steps": (("gelu_tanh", 9, 1),),
    },
}


@dataclass(frozen=True)
class ActivationCallSpec:
    variant: str
    tokens: int
    intermediate: int
    dtype: str = "bf16"
    phase: str = "prefill"

    def __post_init__(self):
        if self.variant not in ACTIVATION_CALLS:
            raise ModelingError(f"activation_call_variant_unsupported:{self.variant}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (self.tokens, self.intermediate)):
            raise ModelingError("activation_call_requires_positive_integer_tensor_shape")
        if self.dtype not in {"bf16", "bfloat16"}:
            raise ModelingError(f"activation_call_dtype_unsupported:{self.dtype}")
        if self.phase not in {"prefill", "decode"}:
            raise ModelingError(f"activation_call_phase_unsupported:{self.phase}")


@dataclass(frozen=True)
class ActivationCall:
    spec: ActivationCallSpec
    ops: tuple[OpSpec, ...]
    recipes: RecipeBook
    coverage: dict


def build_activation_call(spec: ActivationCallSpec) -> ActivationCall:
    """Construct one complete closure; the common call runner owns repetition."""
    declaration = ACTIVATION_CALLS[spec.variant]
    elements = spec.tokens * spec.intermediate
    tensor_bytes = elements * 2
    shape = (spec.tokens, spec.intermediate)
    ops = []
    for ordinal, (operation, lane_ops, inputs) in enumerate(declaration["steps"]):
        ops.append(vector_op(
            op_id=f"activation_api{ordinal}", deps=(ops[-1].op_id,) if ops else (),
            stage=0, layer=0, requests=(), elements=elements,
            ops_per_element=lane_ops, dtype="bf16", family=operation, phase=spec.phase,
            recipe=f"analytic_api_{operation}", shape=shape,
            read_bytes_override=inputs * tensor_bytes, write_bytes_override=tensor_bytes,
            metadata=(
                ("api", declaration["api"][ordinal]),
                ("call_variant", spec.variant),
                ("call_layout", declaration["layout"]),
                ("arithmetic_basis", "nominal_expression_lane_operations"),
                ("native_kernel_dispatch", "unresolved"),
                ("materialized_input", "activation_temporary" if ordinal else "resident_operand"),
            ),
        ))
    separate = len(ops) == 2
    coverage = {
        "model": "analytic_activation_call/1",
        "abstraction": "api_resource_envelope",
        "variant": spec.variant,
        "phase": spec.phase,
        "shape": {"tokens": spec.tokens, "intermediate": spec.intermediate},
        "dtype": "bf16",
        "layout": declaration["layout"],
        "api_sequence": list(declaration["api"]),
        "api_calls": len(ops),
        "native_kernel_count": None,
        "input_residency": "allocated_and_initialized_before_timed_call",
        "intermediate": (
            {"shape": list(shape), "dtype": "bf16", "bytes": tensor_bytes,
             "producer": ops[0].op_id, "consumer": ops[1].op_id,
             "lifetime": "activation_return_through_multiply_read"}
            if separate else None),
        "output": {"shape": list(shape), "dtype": "bf16", "bytes": tensor_bytes},
        "hbm_bytes": sum(op.hbm_bytes for op in ops),
        "vector_lane_operations": sum(op.vector_ops for op in ops),
        "unresolved": [
            "loaded_kernel_binary",
            "native_api_internal_dispatch_and_kernel_count",
            "instruction_class_service_for_exp_divide_and_tanh",
            "cache_residency_and_backend_tile_mapping",
            "host_api_allocator_and_event_dispatch_gaps",
        ],
        "interpretation": (
            "Complete declared API dataflow; analytical device service estimate, "
            "not an identified native kernel implementation or event duration."
        ),
    }
    return ActivationCall(spec, tuple(ops), RecipeBook(), coverage)


from collections.abc import Mapping, Sequence
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes

ACLNN_ABI_SCHEMA = "mindie-sim.aclnn-abi/1"
ACLNN_DEFAULT_DTYPE = "bf16"
ACLNN_GROUPED_KIND = "grouped"
ACLNN_ATTENTION_KIND = "attention"
ACLNN_VECTOR_KIND = "vector"

ACLNN_PHASES = frozenset({"prefill", "decode", "mixed", "primitive"})
ACLNN_MATRIX_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16",
                                 "fp32", "float32", "int8", "w8a8",
                                 "fp8", "fp8_e4m3", "fp8_e5m2",
                                 "int4", "w4a8", "mxfp4"})
ACLNN_VECTOR_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16",
                                 "fp32", "float32"})
ACLNN_ATTENTION_QUERY_DTYPES = ACLNN_VECTOR_DTYPES | frozenset({
    "fp8", "fp8_e4m3", "fp8_e5m2", "int8",
})
ACLNN_ATTENTION_KV_DTYPES = ACLNN_ATTENTION_QUERY_DTYPES | frozenset({
    "int4", "mxfp4",
})
MATRIX_MIN_RANK = 2
MATMUL_SHAPE_DIMS = 3
ATTN_UNBATCHED_RANK = 3
ATTN_BATCHED_RANK = 4
FLOPS_PER_FMA = 2
ATTENTION_SOFTMAX_OPS_PER_PAIR = 5
RMSNORM_OPS_PER_ELEM = 5
ADD_RMSNORM_OPS_PER_ELEM = 6
SWIGLU_OPS_PER_ELEM = 5
SWIGLU_SPLIT_PARTS = 2
DYNAMIC_QUANT_OPS_PER_ELEM = 4
CAST_OPS_PER_ELEM = 1.0
GROUP_LIST_TYPE_CUMULATIVE = 0
GROUP_LIST_TYPE_COUNTS = 1
GROUP_LIST_TYPE_INDEX_COUNT = 2
GROUP_LIST_PAIR_LEN = 2
GROUP_TYPE_M = 0
GROUP_TYPE_K = 2
SPARSE_MODE_DEFAULT_MASK = 0
SPARSE_MODE_ALL_MASK = 1
SPARSE_MODE_LEFT_UP_CAUSAL = 2
SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3
ATTN_PAGED_CACHE_RANK = 3
ATTN_PAGED_4D_CACHE_RANK = 4
ACLNN_ATTENTION_LAYOUTS = frozenset({"BNSD", "BSND", "BNSD_NBSD", "BSH", "SBH", "TND"})
ACLNN_ATTN_MASK_KEYS = ("atten_mask", "attenMaskOptional", "attn_mask", "mask")
ACLNN_ATTN_BLOCK_TABLE_KEYS = ("block_table", "block_tables")
ACLNN_ATTN_SEQLEN_Q_KEYS = (
    "actual_seq_lengths", "actual_seq_qlen", "actual_seq_len_q", "cu_seqlens_q",
)
ACLNN_ATTN_SEQLEN_KV_KEYS = (
    "actual_seq_lengths_kv", "actual_seq_kvlen", "actual_seq_len_kv", "cu_seqlens_k",
)
ACLNN_DTYPE_CANONICAL = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}


def _mapping(value, reason):
    if not isinstance(value, Mapping):
        raise ModelingError(reason)
    return dict(value)


def _text(value, reason):
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(reason)
    return value.strip()


def _positive_integer(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(reason)
    return value


def _nonnegative_integer(value, reason):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(reason)
    return value


def _dtype(value, allowed, reason):
    dtype = _text(value or ACLNN_DEFAULT_DTYPE, reason).lower()
    if dtype not in allowed:
        raise ModelingError(f"{reason}:{dtype}")
    dtype_bytes(dtype)
    return dtype


def _boolean(value, reason):
    if not isinstance(value, bool):
        raise ModelingError(reason)
    return value


def _phase(value):
    phase = _text(value or "primitive", "aclnn_call_phase_invalid")
    if phase not in ACLNN_PHASES:
        raise ModelingError(f"aclnn_call_phase_unsupported:{phase}")
    return phase


def _shape(value, name):
    if isinstance(value, Mapping):
        value = value.get("shape")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ModelingError(f"aclnn_call_tensor_shape_invalid:{name}")
    shape = []
    for item in value:
        shape.append(_positive_integer(item, f"aclnn_call_tensor_shape_invalid:{name}"))
    return tuple(shape)


def _tensor(arguments, *names):
    tensors = arguments.get("tensors")
    if not isinstance(tensors, Mapping):
        return None, None
    for name in names:
        raw = tensors.get(name)
        if raw is None:
            continue
        return name, _shape(raw, name)
    return None, None


def _canonical_dtype(dtype: str) -> str:
    return ACLNN_DTYPE_CANONICAL.get(dtype, dtype)


def _leading_batch_shape(shape):
    return shape[:-2] if len(shape) > 2 else ()


def _broadcast_batch_shape(left, right):
    max_len = max(len(left), len(right))
    left_pad = (1,) * (max_len - len(left)) + left
    right_pad = (1,) * (max_len - len(right)) + right
    result = []
    for a, b in zip(left_pad, right_pad):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ModelingError("aclnn_matmul_weight_batch_mismatch")
    return tuple(result)


def _parse_indexed_group_list(raw):
    pairs = []
    for item in raw:
        if (isinstance(item, (str, bytes))
                or not isinstance(item, Sequence)
                or len(item) != GROUP_LIST_PAIR_LEN):
            raise ModelingError("aclnn_grouped_matmul_expert_rows_invalid")
        expert_idx = _nonnegative_integer(
            item[0], "aclnn_grouped_matmul_expert_rows_invalid")
        count = _nonnegative_integer(
            item[1], "aclnn_grouped_matmul_expert_rows_invalid")
        pairs.append((expert_idx, count))
    return tuple(count for _, count in pairs)


def _expert_rows_and_semantics(arguments):
    attrs = arguments.get("attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    if arguments.get("expert_rows") is not None:
        raw = arguments["expert_rows"]
        explicit_direct = True
    elif arguments.get("group_list") is not None:
        raw = arguments["group_list"]
        explicit_direct = False
    else:
        return None, None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ModelingError("aclnn_grouped_matmul_expert_rows_invalid")
    if explicit_direct:
        values = tuple(_nonnegative_integer(
            value, "aclnn_grouped_matmul_expert_rows_invalid") for value in raw)
        counts, semantics = values, "direct_counts"
    else:
        raw_type = arguments.get("group_list_type", attrs.get("group_list_type"))
        if raw_type is not None:
            list_type = _nonnegative_integer(
                raw_type, "aclnn_grouped_matmul_group_list_type_invalid")
            if list_type not in (
                GROUP_LIST_TYPE_CUMULATIVE,
                GROUP_LIST_TYPE_COUNTS,
                GROUP_LIST_TYPE_INDEX_COUNT,
            ):
                raise ModelingError("aclnn_grouped_matmul_group_list_type_invalid")
        else:
            list_type = None
        if list_type == GROUP_LIST_TYPE_INDEX_COUNT:
            counts, semantics = _parse_indexed_group_list(raw), "indexed_counts"
        else:
            values = tuple(_nonnegative_integer(
                value, "aclnn_grouped_matmul_expert_rows_invalid") for value in raw)
            monotonic = len(values) > 1 and all(
                values[idx] >= values[idx - 1] for idx in range(1, len(values))
            )
            if list_type == GROUP_LIST_TYPE_CUMULATIVE:
                if not monotonic and len(values) > 1:
                    raise ModelingError("aclnn_grouped_matmul_group_list_non_monotonic")
                counts = tuple(
                    values[0] if idx == 0 else values[idx] - values[idx - 1]
                    for idx in range(len(values))
                )
                semantics = "cumulative_offsets"
            elif list_type == GROUP_LIST_TYPE_COUNTS or not monotonic:
                counts, semantics = values, "direct_counts"
            else:
                counts = tuple(
                    values[0] if idx == 0 else values[idx] - values[idx - 1]
                    for idx in range(len(values))
                )
                semantics = "cumulative_offsets"
    if not any(counts):
        raise ModelingError("aclnn_grouped_matmul_requires_active_expert")
    return counts, semantics


def _expert_rows(arguments):
    rows, _ = _expert_rows_and_semantics(arguments)
    return rows


def _grouped_matmul_axis(arguments) -> int:
    attrs = arguments.get("attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    raw = arguments.get("group_type", attrs.get("group_type", GROUP_TYPE_M))
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ModelingError("aclnn_grouped_matmul_group_type_invalid")
    return raw


def _matrix_geometry(arguments, kind):
    """Return ``(m, n, k, groups, group_type)`` for one standalone matrix ABI."""
    attrs = arguments.get("attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    _, x_shape = _tensor(arguments, "x", "input", "activation", "a")
    _, w_shape = _tensor(arguments, "weight", "w", "b")
    if x_shape is None and arguments.get("x_shape") is not None:
        x_shape = _shape(arguments["x_shape"], "x_shape")
    if w_shape is None and arguments.get("weight_shape") is not None:
        w_shape = _shape(arguments["weight_shape"], "weight_shape")
    transpose_a = bool(arguments.get("transpose_a", attrs.get("transpose_a", False)))
    transpose_b = bool(arguments.get("transpose_b", attrs.get("transpose_b", False)))
    x_batch, m, k = (), None, None
    shape = arguments.get("shape")
    if x_shape is None and shape is not None:
        shape = _shape(shape, "shape")
        if len(shape) != MATMUL_SHAPE_DIMS:
            raise ModelingError("aclnn_matmul_shape_requires_m_n_k")
        m, n, k = shape
        shape = None
    else:
        n = None
    if x_shape is not None:
        if len(x_shape) < MATRIX_MIN_RANK:
            raise ModelingError("aclnn_matmul_requires_rank_at_least_two_activation")
        x_batch = _leading_batch_shape(x_shape)
        m, k = (x_shape[-1], x_shape[-2]) if transpose_a else x_shape[-2:]
    w_batch = ()
    weight_count = 1
    if w_shape is not None:
        if len(w_shape) < MATRIX_MIN_RANK:
            raise ModelingError("aclnn_matmul_requires_rank_at_least_two_weight")
        w_batch = _leading_batch_shape(w_shape)
        weight_count = math.prod(w_batch) if w_batch else 1
        first, second = w_shape[-2:]
        n = first if transpose_b else second
        weight_k = second if transpose_b else first
        if k is not None and weight_k != k:
            raise ModelingError("aclnn_matmul_weight_k_dimension_mismatch")
        k = weight_k if k is None else k
    group_type = _grouped_matmul_axis(arguments) if kind == ACLNN_GROUPED_KIND else GROUP_TYPE_M
    rows = _expert_rows(arguments) if kind == ACLNN_GROUPED_KIND else None
    if rows is not None and group_type == GROUP_TYPE_K:
        k_grouped = sum(rows)
        if k is not None and k_grouped > k:
            raise ModelingError("aclnn_grouped_matmul_k_exceeds_operand")
        k = k_grouped
        groups = sum(1 for value in rows if value)
        batch = math.prod(x_batch) if x_batch else 1
    elif rows is not None:
        m, groups = sum(rows), sum(1 for value in rows if value)
        batch = 1
    elif kind == ACLNN_GROUPED_KIND:
        batch = math.prod(x_batch) if x_batch else 1
        groups = _positive_integer(
            arguments.get("groups", weight_count),
            "aclnn_grouped_matmul_requires_positive_groups")
        if weight_count not in (1, groups):
            raise ModelingError("aclnn_grouped_matmul_weight_group_mismatch")
    else:
        out_batch = _broadcast_batch_shape(x_batch, w_batch)
        batch = math.prod(out_batch) if out_batch else 1
        groups = max(1, weight_count)
    m = _positive_integer(arguments.get("m", m),
                          "aclnn_matmul_requires_positive_m") * batch
    n = _positive_integer(arguments.get("n", n),
                          "aclnn_matmul_requires_positive_n")
    k = _positive_integer(arguments.get("k", k),
                          "aclnn_matmul_requires_positive_k")
    return m, n, k, groups, group_type


def _default_matrix_output_dtype(activation: str, weight: str) -> str:
    floats = [_canonical_dtype(d) for d in (activation, weight) if d in ACLNN_VECTOR_DTYPES]
    if not floats:
        return ACLNN_DEFAULT_DTYPE
    if "fp32" in floats:
        return "fp32"
    if all(d == "fp16" for d in floats):
        return "fp16"
    return ACLNN_DEFAULT_DTYPE


def _quant_spec(symbol, kind, arguments):
    from mindie_sim.catalog.types import QuantSpec
    quantized = kind == ACLNN_GROUPED_KIND or "QuantBatchMatmul" in symbol
    activation = _dtype(
        arguments.get("activation_dtype") or arguments.get("dtype")
        or _optional_tensor_dtype(
            arguments, ("x", "input", "a"), ACLNN_MATRIX_DTYPES,
            "aclnn_call_activation_dtype_unsupported")
        or ("int8" if quantized else ACLNN_DEFAULT_DTYPE),
        ACLNN_MATRIX_DTYPES, "aclnn_call_activation_dtype_unsupported")
    default_weight = ("int8" if quantized and activation == "int8"
                      else activation)
    weight = _dtype(
        arguments.get("weight_dtype")
        or _optional_tensor_dtype(
            arguments, ("weight", "w", "b"), ACLNN_MATRIX_DTYPES,
            "aclnn_call_weight_dtype_unsupported")
        or default_weight,
        ACLNN_MATRIX_DTYPES, "aclnn_call_weight_dtype_unsupported")
    output = _dtype(
        arguments.get("output_dtype")
        or _optional_tensor_dtype(
            arguments, ("output", "out", "y"), ACLNN_VECTOR_DTYPES,
            "aclnn_call_output_dtype_unsupported")
        or _default_matrix_output_dtype(activation, weight),
        ACLNN_VECTOR_DTYPES, "aclnn_call_output_dtype_unsupported")
    return QuantSpec(
        name="aclnn_grouped_matmul" if kind == ACLNN_GROUPED_KIND
        else "aclnn_quant_batch_matmul_v3" if quantized else "aclnn_matmul",
        weight_format=weight, activation_format=activation,
        output_format=output), activation


def _matrix_metadata(symbol, kind, arguments, m, n, k, groups, group_type):
    metadata = [("api", symbol), ("aclnn_symbol", symbol),
                ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
                ("aclnn_call_kind", kind),
                ("aclnn_geometry", {"m": m, "n": n, "k": k,
                                    "matrix_groups": groups,
                                    "group_type": group_type}),
                ("program_kind", "matmul")]
    if kind == ACLNN_GROUPED_KIND:
        metadata.extend((
            ("matrix_mapping_domain", "aggregate_experts"),
            ("routing_basis", "probability_moments"),
            ("expected_active_experts", float(groups)),
            ("expected_rows", float(m)),
            ("matrix_groups", groups),
            ("group_type", group_type),
        ))
        rows, semantics = _expert_rows_and_semantics(arguments)
        if rows is not None:
            metadata.extend((("aclnn_group_list", rows),
                             ("group_list_semantics", semantics)))
    return tuple(metadata)


def _build_matrix(arguments, symbol, kind, recipe, family):
    from dataclasses import replace
    from mindie_sim.catalog.ops import gemm_op
    m, n, k, groups, group_type = _matrix_geometry(arguments, kind)
    quant, activation = _quant_spec(symbol, kind, arguments)
    metadata = _matrix_metadata(symbol, kind, arguments, m, n, k, groups, group_type)
    # K-axis grouped matmul reads X[M, K_total] and W[K_total, N] once while
    # emitting one [M, N] output matrix per active group.
    weight_banks = 1 if group_type == GROUP_TYPE_K else groups
    op = gemm_op(
        op_id="aclnn_call", deps=(), stage=0, layer=0, requests=(),
        m=m, n=n, k=k, quant=quant, category="linear",
        phase=_phase(arguments.get("phase")), recipe=recipe, family=family,
        weight_count=weight_banks, activation_dtype=activation, metadata=metadata)
    if group_type == GROUP_TYPE_K:
        out_bytes = float(groups * m * n) * dtype_bytes(quant.output_format)
        op = replace(op, mte3_bytes=out_bytes, hbm_bytes=op.mte2_bytes + out_bytes)
    return op, groups


def _optional_tensor_dtype(arguments, names, allowed, reason):
    tensors = arguments.get("tensors")
    if isinstance(tensors, Mapping):
        for name in names:
            raw = tensors.get(name)
            if isinstance(raw, Mapping) and raw.get("dtype"):
                return _dtype(raw["dtype"], allowed, reason)
    return None


def _tensor_dtype(arguments, names, fallback):
    tensors = arguments.get("tensors")
    if isinstance(tensors, Mapping):
        for name in names:
            raw = tensors.get(name)
            if isinstance(raw, Mapping) and raw.get("dtype"):
                return _dtype(raw["dtype"], ACLNN_VECTOR_DTYPES,
                              "aclnn_call_tensor_dtype_unsupported")
    return _dtype(arguments.get("dtype", fallback), ACLNN_VECTOR_DTYPES,
                  "aclnn_call_dtype_unsupported")


def _vector_tensor_shape(arguments):
    name, shape = _tensor(arguments, "x", "input", "src", "self", "dst", "output")
    if shape is not None:
        return name, shape
    for key in ("x_shape", "shape"):
        if arguments.get(key) is not None:
            return key, _shape(arguments[key], key)
    return None, None


def _vector_geometry(arguments, *, kind=""):
    _, shape = _vector_tensor_shape(arguments)
    if shape is not None:
        if kind in {"rmsnorm", "add_rmsnorm"}:
            _, gamma_shape = _tensor(arguments, "gamma", "weight")
            if gamma_shape is not None:
                norm_rank = len(gamma_shape)
                if norm_rank > len(shape) or shape[-norm_rank:] != gamma_shape:
                    raise ModelingError("aclnn_rmsnorm_gamma_shape_mismatch")
                default_rows = math.prod(shape[:-norm_rank]) if len(shape) > norm_rank else 1
                default_cols = math.prod(gamma_shape)
            else:
                default_rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
                default_cols = shape[-1]
        else:
            default_rows = math.prod(shape[:-1]) if len(shape) > 1 else 1
            default_cols = shape[-1]
        rows = arguments.get("rows", arguments.get("tokens", default_rows))
        columns = arguments.get(
            "columns", arguments.get(
                "cols", arguments.get(
                    "hidden", arguments.get("width", default_cols))))
        return _positive_integer(rows, "aclnn_vector_requires_positive_rows"), \
            _positive_integer(columns, "aclnn_vector_requires_positive_columns")
    if "elements" in arguments:
        elements = _positive_integer(arguments["elements"],
                                     "aclnn_vector_requires_positive_elements")
        return 1, elements
    return (_positive_integer(arguments.get("rows", arguments.get("tokens")),
                              "aclnn_vector_requires_positive_rows"),
            _positive_integer(arguments.get(
                "columns", arguments.get("cols", arguments.get(
                    "hidden", arguments.get("width")))),
                "aclnn_vector_requires_positive_columns"))


def _swiglu_geometry(arguments):
    attrs = arguments.get("attrs")
    attrs = attrs if isinstance(attrs, Mapping) else {}
    tensor_name, raw_shape = _tensor(arguments, "x", "input")
    if raw_shape is None:
        for key in ("x_shape", "shape"):
            if arguments.get(key) is not None:
                tensor_name, raw_shape = key, _shape(arguments[key], key)
                break
    if raw_shape is not None:
        raw_dim = attrs.get("dim", arguments.get("dim", -1))
        if isinstance(raw_dim, bool) or not isinstance(raw_dim, int):
            raise ModelingError("aclnn_swiglu_split_axis_invalid")
        rank = len(raw_shape)
        axis = raw_dim + rank if raw_dim < 0 else raw_dim
        if axis < 0 or axis >= rank or raw_shape[axis] % SWIGLU_SPLIT_PARTS != 0:
            raise ModelingError("aclnn_swiglu_split_axis_invalid")
        out_shape = tuple(
            dim // SWIGLU_SPLIT_PARTS if idx == axis else dim
            for idx, dim in enumerate(raw_shape)
        )
        out_rows = math.prod(out_shape[:-1]) if len(out_shape) > 1 else 1
        out_cols = out_shape[-1]
        return math.prod(raw_shape), math.prod(out_shape), (out_rows, out_cols)
    rows, columns = _vector_geometry(arguments, kind="swiglu")
    out_elements = rows * columns
    return SWIGLU_SPLIT_PARTS * out_elements, out_elements, (rows, columns)


def _broadcast_compatible(src_shape, dst_shape) -> bool:
    if len(src_shape) > len(dst_shape):
        return False
    offset = len(dst_shape) - len(src_shape)
    return all(
        src_dim == dst_shape[offset + idx] or src_dim == 1
        for idx, src_dim in enumerate(src_shape)
    )


def _tensor_copy_work(arguments):
    dtype = _tensor_dtype(arguments, ("src", "x", "input", "self"),
                          ACLNN_DEFAULT_DTYPE)
    dst_dtype = _optional_tensor_dtype(
        arguments, ("self", "dst", "output", "out"), ACLNN_VECTOR_DTYPES,
        "aclnn_inplace_copy_output_dtype_unsupported")
    target = _dtype(arguments.get("output_dtype") or dst_dtype or dtype,
                    ACLNN_VECTOR_DTYPES,
                    "aclnn_inplace_copy_output_dtype_unsupported")
    _, src_shape = _tensor(arguments, "src", "x", "input")
    _, dst_shape = _tensor(arguments, "self", "dst", "output", "out")
    if src_shape is not None and dst_shape is not None:
        if not _broadcast_compatible(src_shape, dst_shape):
            raise ModelingError("aclnn_inplace_copy_broadcast_mismatch")
        src_elements = math.prod(src_shape)
        dst_elements = math.prod(dst_shape)
        rows = math.prod(dst_shape[:-1]) if len(dst_shape) > 1 else 1
        columns = dst_shape[-1]
    else:
        rows, columns = _vector_geometry(arguments, kind="tensor_copy")
        src_elements = dst_elements = rows * columns
    read = src_elements * dtype_bytes(dtype)
    write = dst_elements * dtype_bytes(target)
    cast_ops = (CAST_OPS_PER_ELEM
                if _canonical_dtype(dtype) != _canonical_dtype(target) else 0.0)
    return dtype, dst_elements, cast_ops, read, write, (rows, columns), {
        "input_format": dtype, "output_format": target,
        "copy_semantics": "inplace_copy_into_destination"}


def _vector_work(kind, arguments):
    if kind == "swiglu":
        dtype = _tensor_dtype(arguments, ("x", "input", "gate"), ACLNN_DEFAULT_DTYPE)
        in_elements, out_elements, out_shape = _swiglu_geometry(arguments)
        act_bytes = dtype_bytes(dtype)
        read = in_elements * act_bytes
        write = out_elements * act_bytes
        return dtype, out_elements, SWIGLU_OPS_PER_ELEM, read, write, out_shape, {
            "activation_kind": "swiglu", "gate_layout": "gate_first"}
    if kind == "tensor_copy":
        return _tensor_copy_work(arguments)
    rows, columns = _vector_geometry(arguments, kind=kind)
    elements = rows * columns
    if kind == "rmsnorm":
        dtype = _tensor_dtype(arguments, ("x", "input", "gamma"), ACLNN_DEFAULT_DTYPE)
        gamma_dtype = _optional_tensor_dtype(
            arguments, ("gamma", "weight"), ACLNN_VECTOR_DTYPES,
            "aclnn_call_tensor_dtype_unsupported") or dtype
        act_bytes = dtype_bytes(dtype)
        gamma_bytes = columns * dtype_bytes(gamma_dtype)
        read = elements * act_bytes + gamma_bytes
        write = elements * act_bytes + rows * dtype_bytes("fp32")
        return dtype, elements, RMSNORM_OPS_PER_ELEM, read, write, (rows, columns), {
            "reduction_width": columns, "norm_has_weight": True,
            "norm_gamma_dtype": gamma_dtype}
    if kind == "add_rmsnorm":
        dtype = _tensor_dtype(arguments, ("x", "residual", "gamma"),
                              ACLNN_DEFAULT_DTYPE)
        gamma_dtype = _optional_tensor_dtype(
            arguments, ("gamma", "weight"), ACLNN_VECTOR_DTYPES,
            "aclnn_call_tensor_dtype_unsupported") or dtype
        residual_dtype = _optional_tensor_dtype(
            arguments, ("residual",), ACLNN_VECTOR_DTYPES,
            "aclnn_call_tensor_dtype_unsupported") or dtype
        act_bytes = dtype_bytes(dtype)
        res_bytes = dtype_bytes(residual_dtype)
        read = elements * (act_bytes + res_bytes) + columns * dtype_bytes(gamma_dtype)
        write = elements * (act_bytes + res_bytes) + rows * dtype_bytes("fp32")
        return dtype, elements, ADD_RMSNORM_OPS_PER_ELEM, read, write, (rows, columns), {
            "reduction_width": columns, "norm_has_weight": True,
            "norm_has_residual": True, "norm_gamma_dtype": gamma_dtype}
    if kind == "quantize":
        dtype = _tensor_dtype(arguments, ("x", "input"), ACLNN_DEFAULT_DTYPE)
        output = _dtype(arguments.get("output_dtype", "int8"),
                        frozenset({"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}),
                        "aclnn_dynamic_quant_output_dtype_unsupported")
        read = elements * dtype_bytes(dtype)
        write = elements * dtype_bytes(output) + rows * dtype_bytes("fp32")
        return dtype, elements, DYNAMIC_QUANT_OPS_PER_ELEM, read, write, (rows, columns), {
            "input_format": dtype, "output_format": output,
            "activation_scale_format": "fp32", "activation_scale_shape": (rows,)}
    raise ModelingError(f"aclnn_vector_kind_unsupported:{kind}")


def _build_vector(arguments, symbol, vector_kind, recipe, family):
    from mindie_sim.catalog.ops import vector_op
    dtype, elements, ops_per_element, read, write, shape, extra = _vector_work(
        vector_kind, arguments)
    metadata = (
        ("api", symbol), ("aclnn_symbol", symbol),
        ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
        ("aclnn_call_kind", ACLNN_VECTOR_KIND),
        *tuple(sorted(extra.items())),
    )
    op = vector_op(
        op_id="aclnn_call", deps=(), stage=0, layer=0, requests=(),
        elements=elements, ops_per_element=ops_per_element, dtype=dtype,
        family=family, phase=_phase(arguments.get("phase")), recipe=recipe,
        shape=shape, read_bytes_override=float(read),
        write_bytes_override=float(write), metadata=metadata)
    return op


def _first_integer(arguments, *names, default=None):
    for name in names:
        if arguments.get(name) is not None:
            return arguments[name]
    return default


def _decode_attention_tensor(shape, layout, *, role, expected_heads=None, head_dim=None):
    """Decode ``(batch, heads, tokens, dim)`` honoring CANN ``input_layout``."""
    if layout is None:
        if len(shape) == ATTN_BATCHED_RANK:
            return shape
        if len(shape) == ATTN_UNBATCHED_RANK:
            return (1, *shape)
        raise ModelingError(f"aclnn_attention_{role}_requires_rank_three_or_four")
    if layout in {"BNSD", "BNSD_NBSD"}:
        if len(shape) != ATTN_BATCHED_RANK:
            raise ModelingError(f"aclnn_attention_{role}_layout_rank_mismatch:{layout}")
        return shape
    if layout == "BSND":
        if len(shape) != ATTN_BATCHED_RANK:
            raise ModelingError(f"aclnn_attention_{role}_layout_rank_mismatch:{layout}")
        batch, tokens, heads, dim = shape
        return batch, heads, tokens, dim
    if layout == "TND":
        if len(shape) != ATTN_UNBATCHED_RANK:
            raise ModelingError(f"aclnn_attention_{role}_layout_rank_mismatch:{layout}")
        tokens, heads, dim = shape
        return 1, heads, tokens, dim
    if layout in {"BSH", "SBH"}:
        if len(shape) != ATTN_UNBATCHED_RANK:
            raise ModelingError(f"aclnn_attention_{role}_layout_rank_mismatch:{layout}")
        batch, tokens, hidden = shape if layout == "BSH" else (shape[1], shape[0], shape[2])
        if expected_heads is not None:
            if hidden % expected_heads != 0:
                raise ModelingError(f"aclnn_attention_{role}_hidden_not_divisible_by_heads")
            return batch, expected_heads, tokens, hidden // expected_heads
        if head_dim is not None and hidden % head_dim == 0:
            return batch, hidden // head_dim, tokens, head_dim
        raise ModelingError(f"aclnn_attention_{role}_requires_head_num_for_{layout}")
    raise ModelingError(f"aclnn_attention_layout_unsupported:{layout}")


def _extract_block_table_shape(arguments, attrs):
    tensors = arguments.get("tensors")
    for source in (tensors, attrs, arguments):
        if not isinstance(source, Mapping):
            continue
        for key in ACLNN_ATTN_BLOCK_TABLE_KEYS:
            raw = source.get(key)
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                if raw.get("shape") is not None:
                    return _shape(raw["shape"], key)
                vals = raw.get("values")
                if isinstance(vals, Sequence) and vals and isinstance(vals[0], Sequence):
                    return (len(vals), len(vals[0]))
                return (1, 1)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
                if isinstance(raw[0], Sequence) and not isinstance(raw[0], (str, bytes)):
                    return (len(raw), len(raw[0]))
                return (1, len(raw))
    return None


def _decode_paged_cache_tensor(
    shape, *, role, head_dim, expected_kv_heads, declared_block_size, has_block_table,
):
    """Decode vLLM-Ascend 3D packed ``[pages, block, kv_heads*D]`` or 4D paged cache."""
    if len(shape) == ATTN_PAGED_CACHE_RANK:
        num_pages, page_size, packed_width = shape
        if not has_block_table and (declared_block_size is None or page_size != declared_block_size):
            return None
        if declared_block_size is not None and page_size != declared_block_size:
            raise ModelingError(f"aclnn_attention_{role}_paged_block_size_mismatch")
        if role == "key":
            if expected_kv_heads is not None:
                if packed_width != expected_kv_heads * head_dim:
                    raise ModelingError("aclnn_attention_key_geometry_mismatch")
                kv_heads, dim = expected_kv_heads, head_dim
            elif packed_width % head_dim == 0:
                kv_heads, dim = packed_width // head_dim, head_dim
            else:
                raise ModelingError("aclnn_attention_key_geometry_mismatch")
        else:
            if expected_kv_heads is None or packed_width % expected_kv_heads != 0:
                raise ModelingError("aclnn_attention_value_geometry_mismatch")
            kv_heads, dim = expected_kv_heads, packed_width // expected_kv_heads
        return num_pages, page_size, kv_heads, dim
    if len(shape) == ATTN_PAGED_4D_CACHE_RANK and has_block_table:
        num_pages, page_size, kv_heads, dim = shape
        if declared_block_size is not None and page_size != declared_block_size:
            raise ModelingError(f"aclnn_attention_{role}_paged_block_size_mismatch")
        if role == "key" and dim != head_dim:
            raise ModelingError("aclnn_attention_key_geometry_mismatch")
        if expected_kv_heads is not None and kv_heads != expected_kv_heads:
            raise ModelingError(f"aclnn_attention_{role}_geometry_mismatch")
        return num_pages, page_size, kv_heads, dim
    return None


def _extract_int_sequence(arguments, attrs, keys):
    tensors = arguments.get("tensors")
    for source in (attrs, arguments, tensors):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            raw = source.get(key)
            if raw is None:
                continue
            if isinstance(raw, Mapping):
                raw = raw.get("values", raw.get("data"))
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
                continue
            return tuple(_nonnegative_integer(v, f"aclnn_attention_{key}_invalid") for v in raw)
    return None


def _difference_cumulative(values, reason):
    if any(values[idx] < values[idx - 1] for idx in range(1, len(values))):
        raise ModelingError(reason)
    return tuple(
        values[0] if idx == 0 else values[idx] - values[idx - 1]
        for idx in range(len(values))
    )


def _attention_request_pairs(arguments, attrs, *, layout, paged, batch, query_tokens, key_tokens):
    """Resolve per-request ``((q_len, kv_len), ...)`` pairs for packed TND or batched calls."""
    if layout != "TND":
        return None
    raw_q = _extract_int_sequence(arguments, attrs, ACLNN_ATTN_SEQLEN_Q_KEYS)
    raw_kv = _extract_int_sequence(arguments, attrs, ACLNN_ATTN_SEQLEN_KV_KEYS)
    if raw_q is None and raw_kv is None:
        return None
    if raw_q is None:
        raw_q = raw_kv
    if raw_kv is None:
        raw_kv = raw_q
    if len(raw_q) != len(raw_kv):
        raise ModelingError("aclnn_attention_seqlen_count_mismatch")
    q_lens = _difference_cumulative(raw_q, "aclnn_attention_seqlen_q_non_monotonic")
    if sum(q_lens) != query_tokens:
        raise ModelingError("aclnn_attention_seqlen_q_total_mismatch")
    if paged:
        kv_lens = raw_kv
    else:
        kv_lens = _difference_cumulative(raw_kv, "aclnn_attention_seqlen_kv_non_monotonic")
        if sum(kv_lens) != key_tokens:
            raise ModelingError("aclnn_attention_seqlen_kv_total_mismatch")
    return tuple(zip(q_lens, kv_lens))


def _visible_pairs(query_tokens: int, key_tokens: int, *, causal: bool, top_left_causal: bool) -> int:
    if not causal:
        return query_tokens * key_tokens
    visible = min(query_tokens, key_tokens)
    if top_left_causal:
        return visible * (visible + 1) // 2 + (query_tokens - visible) * key_tokens
    return visible * key_tokens - visible * (visible - 1) // 2


def _attention_geometry(arguments, attrs):
    """[Cohesive-Pipeline] Compute unified attention tensor and KV geometry for one layer call."""
    raw_layout = attrs.get("input_layout", arguments.get("input_layout"))
    layout = None
    if raw_layout is not None:
        layout = _text(raw_layout, "aclnn_attention_layout_invalid").upper()
        if layout not in ACLNN_ATTENTION_LAYOUTS:
            raise ModelingError(f"aclnn_attention_layout_unsupported:{layout}")
    declared_heads = _first_integer(attrs, "head_num", "num_heads", "heads")
    if declared_heads is None:
        declared_heads = _first_integer(arguments, "head_num", "num_heads", "heads")
    if declared_heads is not None:
        declared_heads = _positive_integer(
            declared_heads, "aclnn_attention_requires_positive_heads")
    declared_kv_heads = _first_integer(attrs, "kv_heads", "num_key_value_heads")
    if declared_kv_heads is None:
        declared_kv_heads = _first_integer(arguments, "kv_heads", "num_key_value_heads")
    if declared_kv_heads is not None:
        declared_kv_heads = _positive_integer(
            declared_kv_heads, "aclnn_attention_requires_positive_kv_heads")
    raw_block_size = arguments.get("block_size", attrs.get("block_size"))
    declared_block_size = (
        _positive_integer(raw_block_size, "aclnn_attention_block_size_invalid")
        if raw_block_size is not None else None
    )
    block_table_shape = _extract_block_table_shape(arguments, attrs)
    has_block_table = block_table_shape is not None
    paged = declared_block_size is not None or has_block_table
    tensors = arguments.get("tensors")
    query = None
    if isinstance(tensors, Mapping):
        for name in ("query", "q", "query_shape", "q_shape"):
            raw = tensors.get(name)
            if raw is not None:
                query = _shape(raw, name)
                break
    if query is None:
        for name in ("query_shape", "q_shape", "shape"):
            if arguments.get(name) is not None:
                query = _shape(arguments[name], name)
                break
    if query is None:
        batch = _positive_integer(arguments.get("batch", 1),
                                  "aclnn_attention_requires_positive_batch")
        heads = _positive_integer(declared_heads,
                                  "aclnn_attention_requires_positive_heads")
        query_tokens = _positive_integer(
            _first_integer(arguments, "query_tokens", "q_len", "seq_len", "query_len"),
            "aclnn_attention_requires_positive_query_tokens")
        key_tokens = _positive_integer(
            _first_integer(arguments, "key_tokens", "kv_len", "kv_seq_len",
                           default=query_tokens),
            "aclnn_attention_requires_positive_key_tokens")
        head_dim = _positive_integer(
            _first_integer(arguments, "head_dim", "d"),
            "aclnn_attention_requires_positive_head_dim")
        v_head_dim = _positive_integer(
            _first_integer(arguments, "v_head_dim", "value_head_dim",
                           default=head_dim),
            "aclnn_attention_requires_positive_v_head_dim")
        kv_heads = _positive_integer(
            declared_kv_heads if declared_kv_heads is not None else heads,
            "aclnn_attention_requires_positive_kv_heads")
        return (batch, heads, query_tokens, key_tokens, head_dim, v_head_dim,
                kv_heads, layout, paged, declared_block_size or 0)
    batch, heads, query_tokens, head_dim = _decode_attention_tensor(
        query, layout, role="query", expected_heads=declared_heads)
    if declared_heads is not None and heads != declared_heads:
        raise ModelingError("aclnn_attention_head_num_mismatch")
    key_tokens = _first_integer(arguments, "key_tokens", "kv_len",
                                "kv_seq_len", default=None)
    kv_heads = declared_kv_heads
    block_size = declared_block_size or 0
    key_shape = None
    if isinstance(tensors, Mapping):
        for name in ("key", "k", "key_shape", "kv_shape"):
            raw = tensors.get(name)
            if raw is not None:
                key_shape = _shape(raw, name)
                break
    if key_shape is None:
        for name in ("key_shape", "kv_shape"):
            if arguments.get(name) is not None:
                key_shape = _shape(arguments[name], name)
                break
    paged_cache_layout = False
    if key_shape is not None:
        decoded_paged = (
            _decode_paged_cache_tensor(
                key_shape, role="key", head_dim=head_dim,
                expected_kv_heads=declared_kv_heads,
                declared_block_size=declared_block_size,
                has_block_table=has_block_table,
            )
            if paged else None
        )
        if decoded_paged is not None:
            paged_cache_layout = True
            num_pages, page_size, shape_heads, _ = decoded_paged
            block_size = declared_block_size or page_size
            kv_heads = shape_heads if kv_heads is None else kv_heads
            if key_tokens is None:
                if block_table_shape is not None:
                    key_tokens = block_table_shape[-1] * page_size
                else:
                    key_tokens = num_pages * page_size
        else:
            key_batch, shape_heads, shape_tokens, shape_dim = _decode_attention_tensor(
                key_shape, layout, role="key",
                expected_heads=declared_kv_heads, head_dim=head_dim)
            if key_batch != batch or shape_dim != head_dim:
                raise ModelingError("aclnn_attention_key_geometry_mismatch")
            key_tokens = shape_tokens if key_tokens is None else key_tokens
            kv_heads = shape_heads if kv_heads is None else kv_heads
    value_shape = None
    if isinstance(tensors, Mapping):
        for name in ("value", "v", "value_shape", "v_shape"):
            raw = tensors.get(name)
            if raw is not None:
                value_shape = _shape(raw, name)
                break
    if value_shape is None:
        for name in ("value_shape", "v_shape"):
            if arguments.get(name) is not None:
                value_shape = _shape(arguments[name], name)
                break
    v_head_dim = _first_integer(arguments, "v_head_dim", "value_head_dim",
                               default=None)
    if value_shape is not None:
        decoded_v_paged = (
            _decode_paged_cache_tensor(
                value_shape, role="value", head_dim=head_dim,
                expected_kv_heads=kv_heads or declared_kv_heads,
                declared_block_size=block_size or declared_block_size,
                has_block_table=has_block_table or paged_cache_layout,
            )
            if paged_cache_layout else None
        )
        if decoded_v_paged is not None:
            _, _, value_heads, shape_v_dim = decoded_v_paged
            kv_heads = value_heads if kv_heads is None else kv_heads
            v_head_dim = shape_v_dim
        else:
            value_batch, value_heads, value_tokens, shape_v_dim = _decode_attention_tensor(
                value_shape, layout, role="value",
                expected_heads=kv_heads or declared_kv_heads, head_dim=head_dim)
            if value_batch != batch:
                raise ModelingError("aclnn_attention_value_geometry_mismatch")
            key_tokens = value_tokens if key_tokens is None else key_tokens
            if value_tokens != key_tokens:
                raise ModelingError("aclnn_attention_value_geometry_mismatch")
            kv_heads = value_heads if kv_heads is None else kv_heads
            if value_heads != kv_heads:
                raise ModelingError("aclnn_attention_value_geometry_mismatch")
            v_head_dim = shape_v_dim
    key_tokens = _positive_integer(
        query_tokens if key_tokens is None else key_tokens,
        "aclnn_attention_requires_positive_key_tokens")
    kv_heads = _positive_integer(
        heads if kv_heads is None else kv_heads,
        "aclnn_attention_requires_positive_kv_heads")
    v_head_dim = _positive_integer(
        head_dim if v_head_dim is None else v_head_dim,
        "aclnn_attention_requires_positive_v_head_dim")
    if batch < 1 or heads < 1 or query_tokens < 1 or head_dim < 1:
        raise ModelingError("aclnn_attention_requires_positive_geometry")
    return (batch, heads, query_tokens, key_tokens, head_dim, v_head_dim,
            kv_heads, layout, paged, block_size)


def _iter_scalar_leaves(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _iter_scalar_leaves(item)
    else:
        yield value


def _mask_entry(arguments, attrs):
    tensors = arguments.get("tensors")
    for source in (tensors, attrs, arguments):
        if isinstance(source, Mapping):
            for key in ACLNN_ATTN_MASK_KEYS:
                if source.get(key) is not None:
                    return source[key]
    return None


def _attention_has_mask(arguments, attrs) -> bool:
    if _mask_entry(arguments, attrs) is not None:
        return True
    return bool(attrs.get("has_atten_mask") or arguments.get("has_atten_mask"))


def _mask_is_all_unmasked(arguments, attrs) -> bool:
    entry = _mask_entry(arguments, attrs)
    if entry is None:
        return True
    if isinstance(entry, Mapping):
        if entry.get("all_false") is True or entry.get("mask_kind") == "none":
            return True
        vals = entry.get("values", entry.get("data"))
        if vals is not None:
            leaves = list(_iter_scalar_leaves(vals))
            return bool(leaves) and all(not bool(item) for item in leaves)
        return False
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        leaves = list(_iter_scalar_leaves(entry))
        return bool(leaves) and all(not bool(item) for item in leaves)
    return False


def _attention_op(arguments, symbol, recipe):
    from mindie_sim.schema.operators import OpSpec
    attrs = arguments.get("attrs")
    attrs = (_mapping(attrs, "aclnn_attention_attrs_requires_mapping")
             if attrs is not None else {})
    (batch, heads, query_tokens, key_tokens, head_dim, v_head_dim, kv_heads,
     layout, paged, block_size) = _attention_geometry(arguments, attrs)
    sparse_mode = attrs.get("sparse_mode", arguments.get("sparse_mode"))
    sparse_mode = (_nonnegative_integer(sparse_mode,
                                        "aclnn_attention_sparse_mode_invalid")
                   if sparse_mode is not None else None)
    has_mask = _attention_has_mask(arguments, attrs)
    if "causal" in attrs:
        causal = _boolean(attrs["causal"], "aclnn_attention_causal_requires_boolean")
    elif "causal" in arguments:
        causal = _boolean(arguments["causal"], "aclnn_attention_causal_requires_boolean")
    elif sparse_mode in (SPARSE_MODE_LEFT_UP_CAUSAL, SPARSE_MODE_RIGHT_DOWN_CAUSAL):
        causal = True
    elif sparse_mode == SPARSE_MODE_ALL_MASK:
        causal = False
    elif not has_mask or _mask_is_all_unmasked(arguments, attrs):
        causal = False
    else:
        causal = True
    top_left_causal = sparse_mode in (SPARSE_MODE_DEFAULT_MASK, SPARSE_MODE_LEFT_UP_CAUSAL)
    req_pairs = _attention_request_pairs(
        arguments, attrs, layout=layout, paged=paged, batch=batch,
        query_tokens=query_tokens, key_tokens=key_tokens)
    if req_pairs is not None:
        total_pairs = sum(
            _visible_pairs(q_i, k_i, causal=causal, top_left_causal=top_left_causal)
            for q_i, k_i in req_pairs
        )
        effective_batch = len(req_pairs)
        total_kv_tokens = sum(k_i for _, k_i in req_pairs)
    else:
        total_pairs = batch * _visible_pairs(
            query_tokens, key_tokens, causal=causal, top_left_causal=top_left_causal)
        effective_batch = batch
        total_kv_tokens = batch * key_tokens
    query_dtype = _dtype(
        arguments.get("query_dtype")
        or _optional_tensor_dtype(
            arguments, ("query", "q"), ACLNN_ATTENTION_QUERY_DTYPES,
            "aclnn_attention_query_dtype_unsupported")
        or ACLNN_DEFAULT_DTYPE,
        ACLNN_ATTENTION_QUERY_DTYPES, "aclnn_attention_query_dtype_unsupported")
    key_dtype = _dtype(
        arguments.get("key_dtype") or arguments.get("kv_dtype")
        or _optional_tensor_dtype(
            arguments, ("key", "k"), ACLNN_ATTENTION_KV_DTYPES,
            "aclnn_attention_kv_dtype_unsupported")
        or query_dtype,
        ACLNN_ATTENTION_KV_DTYPES, "aclnn_attention_kv_dtype_unsupported")
    value_dtype = _dtype(
        arguments.get("value_dtype") or arguments.get("kv_dtype")
        or _optional_tensor_dtype(
            arguments, ("value", "v"), ACLNN_ATTENTION_KV_DTYPES,
            "aclnn_attention_kv_dtype_unsupported")
        or key_dtype,
        ACLNN_ATTENTION_KV_DTYPES, "aclnn_attention_kv_dtype_unsupported")
    output_dtype = _dtype(
        arguments.get("output_dtype")
        or _optional_tensor_dtype(
            arguments, ("output", "out"), ACLNN_ATTENTION_QUERY_DTYPES,
            "aclnn_attention_output_dtype_unsupported")
        or query_dtype,
        ACLNN_ATTENTION_QUERY_DTYPES, "aclnn_attention_output_dtype_unsupported")
    query_bytes = batch * heads * query_tokens * head_dim * dtype_bytes(query_dtype)
    key_bytes = total_kv_tokens * kv_heads * head_dim * dtype_bytes(key_dtype)
    value_bytes = total_kv_tokens * kv_heads * v_head_dim * dtype_bytes(value_dtype)
    kv_bytes = key_bytes + value_bytes
    output_bytes = batch * heads * query_tokens * v_head_dim * dtype_bytes(output_dtype)
    flops = float(FLOPS_PER_FMA * heads * total_pairs * (head_dim + v_head_dim))
    vector_ops = float(ATTENTION_SOFTMAX_OPS_PER_PAIR * heads * total_pairs)
    if layout == "TND":
        view_q = [query_tokens, heads, head_dim]
        view_k = [key_tokens, kv_heads, head_dim]
        view_v = [key_tokens, kv_heads, v_head_dim]
        view_o = [query_tokens, heads, v_head_dim]
    elif layout == "BNSD_NBSD":
        view_q = [batch, heads, query_tokens, head_dim]
        view_k = [batch, kv_heads, key_tokens, head_dim]
        view_v = [batch, kv_heads, key_tokens, v_head_dim]
        view_o = [heads, batch, query_tokens, v_head_dim]
    else:
        view_q = [batch, heads, query_tokens, head_dim]
        view_k = [batch, kv_heads, key_tokens, head_dim]
        view_v = [batch, kv_heads, key_tokens, v_head_dim]
        view_o = [batch, heads, query_tokens, v_head_dim]
    view = {
        "kind": "paged_gqa" if paged else "dense_gqa",
        "logical_layout": "TND",
        "kv_storage": "paged" if paged else "contiguous",
        "query_shape": view_q,
        "key_shape": view_k,
        "value_shape": view_v,
        "output_shape": view_o,
        "query_dtype": query_dtype, "key_dtype": key_dtype,
        "value_dtype": value_dtype, "output_dtype": output_dtype,
        "key_tail_shape": [block_size or key_tokens, kv_heads * head_dim],
        "value_tail_shape": [block_size or key_tokens, kv_heads * v_head_dim],
        "batch_count": effective_batch, "causal": causal,
    }
    if paged:
        view["page_size"] = block_size
    metadata = (
        ("api", symbol), ("aclnn_symbol", symbol),
        ("aclnn_abi_schema", ACLNN_ABI_SCHEMA),
        ("aclnn_call_kind", ACLNN_ATTENTION_KIND),
        ("attention_tensor_view", view),
        ("attention_geometry", {
            "batch": effective_batch, "heads": heads, "kv_heads": kv_heads,
            "query_tokens": query_tokens, "key_tokens": key_tokens,
            "head_dim": head_dim, "v_head_dim": v_head_dim,
            "causal": causal, "paged": paged, "block_size": block_size or None,
            "sparse_mode": sparse_mode, "top_left_causal": top_left_causal,
            "input_layout": layout,
            "query_dtype": query_dtype, "key_dtype": key_dtype,
            "value_dtype": value_dtype, "output_dtype": output_dtype}),
    )
    return OpSpec(
        op_id="aclnn_call", family="attention", phase=_phase(arguments.get("phase")),
        stage=0, layer=0, requests=(),
        semantic_shape=(batch, heads, query_tokens, head_dim),
        local_shape=(batch, heads, query_tokens, head_dim),
        padded_shape=(batch, heads, query_tokens, head_dim),
        launch_shape=(batch, heads, query_tokens, head_dim),
        flops=flops, vector_ops=vector_ops,
        hbm_bytes=query_bytes + kv_bytes + output_bytes,
        mte2_bytes=query_bytes + kv_bytes, mte3_bytes=output_bytes,
        dtype=query_dtype, recipe=recipe, metadata=metadata)


from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math as math

from mindie_sim.schema.common import ModelingError, dtype_bytes as dtype_bytes



@dataclass(frozen=True)
class CallDefinition:
    ops: tuple
    recipes: object
    ledger: dict
    limitations: tuple


def _activation(arguments):
    from .call_builders import ActivationCallSpec, build_activation_call
    call = build_activation_call(ActivationCallSpec(**arguments))
    return CallDefinition(call.ops, call.recipes, call.coverage,
                          tuple(call.coverage["unresolved"]))


def _gdn(arguments):
    from mindie_sim.prediction.operators.linear_attention.call_gdn import make_call
    call = make_call(arguments["role"], {key: value for key, value in arguments.items()
                                        if key != "role"})
    return CallDefinition(call.ops, call.recipes, call.ledger, call.limitations)


def _quantized_grouped(arguments):
    from mindie_sim.prediction.operators.grouped_moe.call_grouped import build_quantized_grouped_call
    call = build_quantized_grouped_call(arguments)
    return CallDefinition(call.ops, call.recipes, call.ledger, call.limitations)


def _quantized_grouped_source(arguments):
    from mindie_sim.prediction.operators.grouped_moe.call_grouped_source import build_quantized_grouped_source_call
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

#: Exact ABI symbol -> (kind, recipe, family, mechanism class name).
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

#: Explicit versioned/layout aliases of the same operator semantics (zero prefix matching).
ACLNN_ABI_ALIASES = {
    "aclnnMatmulV2": "aclnnMatmul",
    "aclnnMatmulV3": "aclnnMatmul",
    "aclnnMatmulWeightNz": "aclnnMatmul",
    "QuantBatchMatmulV4": "QuantBatchMatmulV3",
    "aclnnQuantMatmulWeightNz": "aclnnQuantBatchMatmulV3",
    "aclnnGroupedMatmulV2": "aclnnGroupedMatmul",
    "aclnnGroupedMatmulV3": "aclnnGroupedMatmul",
    "aclnnGroupedMatmulV4": "aclnnGroupedMatmul",
    "aclnnGroupedMatmulV5": "aclnnGroupedMatmul",
    "aclnnGroupedMatmulWeightNz": "aclnnGroupedMatmul",
    "aclnnFlashAttentionScoreV2": "aclnnFlashAttentionScore",
    "aclnnFusedInferAttentionScoreV2": "aclnnFusedInferAttentionScore",
    "aclnnFusedInferAttentionScoreV3": "aclnnFusedInferAttentionScore",
    "aclnnFusedInferAttentionScoreV4": "aclnnFusedInferAttentionScore",
    "aclnnRmsNormV2": "aclnnRmsNorm",
    "aclnnAddRmsNormV2": "aclnnAddRmsNorm",
    "aclnnDynamicQuantV2": "aclnnDynamicQuant",
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
    """Resolve one exact ABI symbol or explicit same-semantics alias."""
    symbol = _text(symbol, "aclnn_call_requires_api_symbol")
    if symbol in ACLNN_ABI_REGISTRY:
        return symbol, ACLNN_ABI_REGISTRY[symbol]
    target = ACLNN_ABI_ALIASES.get(symbol)
    if target is not None:
        return target, ACLNN_ABI_REGISTRY[target]
    raise ModelingError(f"aclnn_call_api_unsupported:{symbol}")


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
    from mindie_sim.prediction.operators.attention.attention_pipeline import TiledFlashAndPagedAttentionRecipe
    from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
    from .program import RecipeBook
    from mindie_sim.prediction.operators.vector.vector_pipeline import AIVTiledVectorAndGatherRecipe
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
