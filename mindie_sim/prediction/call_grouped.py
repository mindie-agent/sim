"""Complete INT8/FRACTAL_NZ grouped-matmul API callback.

[Role] Declare the retained ``torch_npu.npu_grouped_matmul`` W8A8 dynamic
  callback and bind its equal-row expert geometry to the common matrix core.
[Invariants] Expert counts are explicit direct counts. Nonempty counts are
  equal, empty experts retain their ids and read no weight payload. Logical M
  is their sum. INT8 payloads, FP32 scales, INT32 accumulation and BF16 output
  remain separate. One API/kernel boundary spans all active groups.
[Timeouts & Locks] Pure bounded geometry; no I/O, device work or locks.
[Side-Effect Policy] Returns immutable descriptors, a RecipeBook and a ledger.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import MappedMatmul
from .mechanism import Work
from .program import KernelProgram, RecipeBook, Stage


API = "torch_npu.npu_grouped_matmul"
PRODUCER_PATH = (
    "/Users/maoxx241/code/mindie-agent/.workstreams/hp-serving/qwen35-grouped/"
    "scripts/probe_grouped_graph.py"
)
PRODUCER_SHA256 = "e04e38ccf8a6f27be886ad7db9856fa97491c1cf9d41bd11cf3efa170edb8e5a"
NZ_DOC_PATH = (
    "/Users/maoxx241/code/vllm-ascend/csrc/gmm/"
    "grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h"
)
NZ_DOC_SHA256 = "08e07b078ca73626df3a135bff230c9b7b61108b235e0f98f22420f4c94f9733"
NZ_TILE_K = 16
NZ_TILE_N = 32
NZ_LAYOUT = "FRACTAL_NZ"
FP32_BYTES = 4
INT64_BYTES = 8
BF16_BYTES = 2
INT8_BYTES = 1
EPILOGUE_CONTRACT = "int32_scale_per_token_per_channel_to_bf16/1"
RECIPE_NAME = "quantized_grouped_int8_nz"
ALLOWED_ARGUMENTS = {
    "expert_rows", "k", "n", "phase", "split_item", "group_type",
    "group_list_type", "output_dtype", "weight_layout", "x_dtype",
    "weight_dtype", "scale_dtype", "per_token_scale_dtype", "bias",
}


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(f"quantized_grouped_{name}_requires_integer_at_least_{minimum}")
    return value


def _fixed(arguments, name, expected):
    value = arguments.get(name, expected)
    if value != expected:
        raise ModelingError(f"quantized_grouped_api_argument_unsupported:{name}={value!r}")
    return value


@dataclass(frozen=True)
class QuantizedGroupedSpec:
    expert_rows: tuple
    k: int
    n: int
    phase: str = "primitive"
    split_item: int = 2
    group_type: int = 0
    group_list_type: int = 1
    output_dtype: str = "bfloat16"
    weight_layout: str = NZ_LAYOUT
    x_dtype: str = "int8"
    weight_dtype: str = "int8"
    scale_dtype: str = "float32"
    per_token_scale_dtype: str = "float32"
    bias: None = None

    def __post_init__(self):
        if not self.expert_rows:
            raise ModelingError("quantized_grouped_expert_rows_must_not_be_empty")
        rows = tuple(_integer(value, "expert_row_count", minimum=0)
                     for value in self.expert_rows)
        positive = {value for value in rows if value}
        if not positive:
            raise ModelingError(
                "quantized_grouped_requires_at_least_one_nonempty_expert")
        if len(positive) > 1:
            raise ModelingError(
                "quantized_grouped_requires_uniform_nonempty_expert_rows")
        object.__setattr__(self, "expert_rows", rows)
        _integer(self.k, "k")
        _integer(self.n, "n")
        if self.k % NZ_TILE_K or self.n % NZ_TILE_N:
            raise ModelingError(
                "quantized_grouped_fractal_nz_requires_whole_tiles:"
                f"k%{NZ_TILE_K}=0,n%{NZ_TILE_N}=0")
        if self.phase not in {"primitive", "prefill", "decode", "mixed"}:
            raise ModelingError(f"quantized_grouped_phase_unsupported:{self.phase}")
        fixed = (
            ("split_item", 2), ("group_type", 0), ("group_list_type", 1),
            ("weight_layout", NZ_LAYOUT), ("x_dtype", "int8"),
            ("weight_dtype", "int8"), ("scale_dtype", "float32"),
            ("per_token_scale_dtype", "float32"),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise ModelingError(
                    f"quantized_grouped_branch_unsupported:{name}={getattr(self, name)!r}")
        if self.output_dtype not in {"bf16", "bfloat16"}:
            raise ModelingError(
                f"quantized_grouped_output_dtype_unsupported:{self.output_dtype}")
        if self.bias is not None:
            raise ModelingError("quantized_grouped_bias_must_be_none")
        object.__setattr__(self, "output_dtype", "bfloat16")

    @property
    def experts(self):
        return len(self.expert_rows)

    @property
    def active_ids(self):
        return tuple(index for index, rows in enumerate(self.expert_rows) if rows)

    @property
    def empty_ids(self):
        return tuple(index for index, rows in enumerate(self.expert_rows) if not rows)

    @property
    def groups(self):
        return len(self.active_ids)

    @property
    def tokens_per_expert(self):
        return next((rows for rows in self.expert_rows if rows), 0)

    @property
    def rows(self):
        return sum(self.expert_rows)

    @property
    def nz_storage_shape(self):
        return (self.experts, self.n // NZ_TILE_N, self.k // NZ_TILE_K,
                NZ_TILE_K, NZ_TILE_N)


@dataclass(frozen=True)
class QuantizedGroupedCall:
    spec: QuantizedGroupedSpec
    ops: tuple[OpSpec, ...]
    recipes: RecipeBook
    ledger: dict
    limitations: tuple


def parse_arguments(arguments):
    """Parse the narrow public builder arguments, refusing every other branch."""
    if not isinstance(arguments, dict):
        raise ModelingError("quantized_grouped_arguments_must_be_mapping")
    unknown = set(arguments) - ALLOWED_ARGUMENTS
    if unknown:
        raise ModelingError(
            "quantized_grouped_arguments_unsupported:" + ",".join(sorted(unknown)))
    if "expert_rows" not in arguments or "k" not in arguments or "n" not in arguments:
        raise ModelingError("quantized_grouped_requires_expert_rows_k_and_n")
    rows = arguments["expert_rows"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        raise ModelingError("quantized_grouped_expert_rows_must_be_sequence")
    fixed = (
        ("split_item", 2), ("group_type", 0), ("group_list_type", 1),
        ("weight_layout", NZ_LAYOUT), ("x_dtype", "int8"),
        ("weight_dtype", "int8"), ("scale_dtype", "float32"),
        ("per_token_scale_dtype", "float32"),
    )
    for name, expected in fixed:
        _fixed(arguments, name, expected)
    output_dtype = arguments.get("output_dtype", "bfloat16")
    if output_dtype not in {"bf16", "bfloat16"}:
        raise ModelingError(
            f"quantized_grouped_output_dtype_unsupported:{output_dtype!r}")
    if arguments.get("bias", None) is not None:
        raise ModelingError("quantized_grouped_bias_must_be_none")
    phase = arguments.get("phase", "primitive")
    return QuantizedGroupedSpec(
        tuple(rows), arguments["k"], arguments["n"], phase=phase,
        split_item=2, group_type=0, group_list_type=1,
        output_dtype=output_dtype, weight_layout=NZ_LAYOUT,
        x_dtype="int8", weight_dtype="int8", scale_dtype="float32",
        per_token_scale_dtype="float32", bias=None,
    )


def _epilogue_record(spec):
    vector_ops = 4 * spec.rows * spec.n
    return {
        "contract": EPILOGUE_CONTRACT,
        "vector_ops": vector_ops,
        "operations_per_output": 4,
        "operations": (
            "read_int32_accumulator",
            "cast_int32_to_fp32",
            "multiply_fp32_per_channel_scale",
            "multiply_fp32_per_token_scale",
            "cast_bf16",
            "store_bf16_output",
        ),
        "accumulator_dtype": "int32",
        "accumulator_bytes": 4 * spec.rows * spec.n,
        "output_dtype": "bfloat16",
        "output_bytes": BF16_BYTES * spec.rows * spec.n,
        "ordering": "after_complete_int32_accumulation_before_output_store",
        "service_status": "shared_matrix_final_k_epilogue",
    }


def _geometry(spec):
    groups = spec.groups
    rows = spec.rows
    counts = INT64_BYTES * spec.experts
    payload = INT8_BYTES * rows * spec.k
    activation_scales = FP32_BYTES * rows
    weight_payload = INT8_BYTES * groups * spec.k * spec.n
    weight_scales = FP32_BYTES * groups * spec.n
    packed_weight = weight_payload + weight_scales
    output = BF16_BYTES * rows * spec.n
    return {
        "experts": spec.experts,
        "expert_rows": spec.expert_rows,
        "active_expert_ids": spec.active_ids,
        "empty_expert_ids": spec.empty_ids,
        "active_groups": groups,
        "tokens_per_active_expert": spec.tokens_per_expert,
        "logical_m": rows,
        "n": spec.n,
        "k": spec.k,
        "flops": 2 * rows * spec.n * spec.k,
        "activation_payload_bytes": payload,
        "activation_scale_bytes": activation_scales,
        "weight_payload_bytes": weight_payload,
        "weight_scale_bytes": weight_scales,
        "packed_weight_bytes": packed_weight,
        "output_bytes": output,
        "group_list_scan_bytes": counts,
        "weight_bank_residency_bytes": INT8_BYTES * spec.experts * spec.k * spec.n,
        "weight_scale_bank_residency_bytes": FP32_BYTES * spec.experts * spec.n,
        "mte2_bytes": payload + activation_scales + packed_weight,
        "mte3_bytes": output,
        "hbm_bytes": payload + activation_scales + packed_weight + counts + output,
    }


def _metadata(spec, geometry, epilogue):
    return (
        ("api", API),
        ("api_call", 1),
        ("kernel_boundary", "one_grouped_matmul_api_call"),
        ("implementation", {
            "api": API,
            "native_dispatch": "unresolved",
            "weight_layout": NZ_LAYOUT,
            "quantization": "int8_dynamic_per_token_per_channel",
            "compute_precision": "int8",
            "accumulator_precision": "int32",
            "output_precision": "bfloat16",
        }),
        ("producer_path", PRODUCER_PATH),
        ("producer_sha256", PRODUCER_SHA256),
        ("expert_rows", spec.expert_rows),
        ("active_expert_ids", spec.active_ids),
        ("empty_expert_ids", spec.empty_ids),
        ("active_expert_groups", spec.groups),
        ("tokens_per_active_expert", spec.tokens_per_expert),
        ("matrix_groups", spec.groups),
        ("group_list_type", 1),
        ("group_list_semantics", "direct_counts_not_cumulative_offsets"),
        ("group_list_dtype", "int64"),
        ("group_list_length", spec.experts),
        ("group_list_scan_bytes", geometry["group_list_scan_bytes"]),
        ("group_list_traffic_policy", "one_shared_argument_fetch"),
        ("split_item", 2),
        ("group_type", 0),
        ("bias", "none"),
        ("x_dtype", "int8"),
        ("weight_dtype", "int8"),
        ("scale_dtype", "float32"),
        ("per_token_scale_dtype", "float32"),
        ("activation_format", "int8"),
        ("weight_format", "int8"),
        ("output_format", "bfloat16"),
        ("compute_format", "int8"),
        ("accumulator_format", "int32"),
        ("activation_payload_bytes", geometry["activation_payload_bytes"]),
        ("activation_shape", (spec.rows, spec.k)),
        ("activation_scale_bytes", geometry["activation_scale_bytes"]),
        ("activation_scale_shape", (spec.rows,)),
        ("weight_payload_bytes", geometry["weight_payload_bytes"]),
        ("weight_shape", (spec.experts, spec.k, spec.n)),
        ("weight_scale_bytes", geometry["weight_scale_bytes"]),
        ("weight_scale_shape", (spec.experts, spec.n)),
        ("packed_weight_bytes", geometry["packed_weight_bytes"]),
        ("output_bytes", geometry["output_bytes"]),
        ("output_shape", (spec.rows, spec.n)),
        ("weight_bank_residency_bytes", geometry["weight_bank_residency_bytes"]),
        ("weight_scale_bank_residency_bytes",
         geometry["weight_scale_bank_residency_bytes"]),
        ("weight_bank_residency_policy",
         "declared_full_bank_separate_from_per_invocation_active_payload"),
        ("nz_layout", NZ_LAYOUT),
        ("nz_tile_kn", (NZ_TILE_K, NZ_TILE_N)),
        ("nz_logical_storage_shape", (spec.experts, spec.k, spec.n)),
        ("nz_storage_shape", spec.nz_storage_shape),
        ("nz_whole_tile_bytes_identity", True),
        ("nz_internal_tiled_access", "unresolved"),
        ("matrix_epilogue", {
            "kind": "row_column_scale",
            "operations_per_output": epilogue["operations_per_output"],
            "row_scale_bytes": geometry["activation_scale_bytes"],
            "column_scale_bytes": geometry["weight_scale_bytes"],
        }),
        ("quantization_boundary", "activation_quantization_outside_timed_call"),
        ("weight_conversion_boundary", "fractal_nz_conversion_outside_timed_call"),
        ("program_kind", "matmul"),
        ("coverage_basis", "analytic_tensor_dataflow_and_shared_matrix_program"),
    )


def _ledger(spec, geometry, epilogue):
    return {
        "model": "analytic_grouped_int8_fractal_nz_call/1",
        "coverage_basis": "analytic_tensor_dataflow_and_shared_matrix_program",
        "identified_historical_compiled_kernel": None,
        "producer": {"path": PRODUCER_PATH, "sha256": PRODUCER_SHA256},
        "layout_source": {"path": NZ_DOC_PATH, "sha256": NZ_DOC_SHA256},
        "api": {
            "name": API,
            "x": "int8 [M,K] ND",
            "weight": f"int8 [E,K,N] {NZ_LAYOUT}",
            "scale": "float32 [E,N] per-channel",
            "bias": None,
            "per_token_scale": "float32 [M]",
            "split_item": 2,
            "group_type": 0,
            "group_list": "int64 direct counts [E]",
            "group_list_type": 1,
            "output_dtype": "bfloat16",
            "outputs": 1,
        },
        "geometry": geometry,
        "quantization": {
            "activation": "int8_dynamic_per_token",
            "weight": "int8_per_channel",
            "compute": "int8",
            "accumulation": "int32",
            "output": "bfloat16",
            "activation_quantization_timed": False,
            "weight_nz_conversion_timed": False,
        },
        "matrix": {
            "recipe": RECIPE_NAME,
            "core_recipe": "mapped_matmul",
            "matrix_groups": spec.groups,
            "equal_rows_per_group": spec.tokens_per_expert,
            "one_api_kernel_boundary": True,
            "global_dense_collapse": False,
            "epilogue": epilogue,
        },
        "group_list_fetch": {
            "bytes": geometry["group_list_scan_bytes"],
            "policy": "one_shared_argument_fetch_before_grouped_compute",
            "scope": "analytic_hbm_argument_traffic",
            "per_core_scalar_instruction_schedule": "unresolved",
            "included_in_matrix_operands": False,
        },
        "timed_region": {
            "call": API,
            "starts": "before_python_api_call",
            "ends": "after_python_api_call_returns_output_0",
            "input_allocation_and_initialization": "outside",
            "activation_quantization": "outside",
            "weight_nz_conversion": "outside",
            "count_tensor_construction": "outside",
            "output_allocation": "inside_api_closure_unresolved",
            "api_host_dispatch": "inside_api_closure_unresolved",
            "runtime": "selected_case_device_and_submission_not_capture_timing",
        },
        "unresolved": [
            "identified_historical_compiled_kernel_binary_and_internal_dispatch",
            "native_api_host_dispatch_and_output_allocator_cost",
            "fractal_nz_internal_tiled_load_order_and_alignment",
            "scalar_group_list_and_epilogue_instruction_schedule",
            "native_accumulator_to_vector_storage_transfer_port_and_work",
            "graph_replay_launch_gap_and_runtime_scheduling",
        ],
        "interpretation": (
            "Complete retained API dataflow with exact declared tensor arithmetic; "
            "not an identified historical compiled kernel or measured duration."
        ),
    }


def build_quantized_grouped_call(arguments):
    """Build the one-boundary callback for the retained INT8/FRACTAL_NZ branch."""
    spec = parse_arguments(arguments)
    geometry = _geometry(spec)
    epilogue = _epilogue_record(spec)
    metadata = _metadata(spec, geometry, epilogue)
    op = OpSpec(
        op_id="grouped_matmul",
        family="grouped_gemm",
        phase=spec.phase,
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(spec.rows, spec.n, spec.k),
        local_shape=(spec.rows, spec.n, spec.k),
        padded_shape=(spec.rows, spec.n, spec.k),
        launch_shape=(spec.rows, spec.n, spec.k),
        flops=geometry["flops"],
        hbm_bytes=geometry["hbm_bytes"],
        mte2_bytes=geometry["mte2_bytes"],
        mte3_bytes=geometry["mte3_bytes"],
        vector_ops=epilogue["vector_ops"],
        deps=(),
        recipe=RECIPE_NAME,
        coverage="analytic_tensor_dataflow",
        metadata=metadata,
        notes=(
            "one API callback covers all active expert matrices",
            "expert groups are independent equal-row matrix domains",
            "group list contains direct counts and preserves expert ids",
            "full weight bank residency is distinct from active payload",
            "fused epilogue vector work executes after final K in the common core",
        ),
        dtype="int8",
        layout=NZ_LAYOUT,
    )
    MatrixEpilogue.from_op(op)
    ledger = _ledger(spec, geometry, epilogue)
    limitations = tuple(ledger["unresolved"])
    recipes = RecipeBook({RECIPE_NAME: _GroupedMappedMatmul()})
    return QuantizedGroupedCall(spec, (op,), recipes, ledger, limitations)


class _GroupedMappedMatmul(MappedMatmul):
    """Fetch the count argument without adding it to matrix operand traffic."""

    def lower(self, context):
        op = context.op
        count_bytes = dict(op.metadata)["group_list_scan_bytes"]
        matrix_context = context.for_phase(
            Work(
                matrix={op.dtype: op.flops},
                resources={
                    "hbm": op.hbm_bytes - count_bytes,
                    "mte2": op.mte2_bytes,
                    "mte3": op.mte3_bytes,
                    "vector": op.vector_ops,
                },
            ),
            kernel="quantized_grouped_matrix",
        )
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("group_counts", {context.resource_names["hbm"]: count_bytes},
                  after=("setup",)),
            Stage("matrix", after=("group_counts",),
                  program=super().lower(matrix_context)),
        ), minimum_ms=context.minimum_ms)


__all__ = [
    "API", "EPILOGUE_CONTRACT", "NZ_DOC_PATH", "NZ_DOC_SHA256",
    "NZ_LAYOUT", "NZ_TILE_K", "NZ_TILE_N", "PRODUCER_PATH", "PRODUCER_SHA256",
    "QuantizedGroupedCall", "QuantizedGroupedSpec", "build_quantized_grouped_call",
    "parse_arguments",
]
