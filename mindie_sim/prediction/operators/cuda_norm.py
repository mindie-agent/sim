"""Source-derived plans for the CUDA RMSNorm kernels.

[Role] Reconstruct CUDA host dispatch, typed work and finite CTA programs.
[Invariants] Source work is independent of measurements and model labels.
  Per-warp loops preserve read/compute/write dependencies; CUB shuffles use
  active-lane predicates and thread0 combines warp aggregates sequentially.
  Each CTA owns one compute unit; HBM is shared across units. Typed source
  counts require explicit service rates, never generic-vector equivalence.
  Declared smem/shared pools reserve the 136B source payload for the whole CTA;
  absent pools mean unconstrained capacity. Compiled sizeof/register counts,
  compiler instruction issue and multi-CTA occupancy remain unresolved.
[Timeouts & Locks] Finite virtual execution; O(owners * warps * stages) live
  tasks, one lazy row stream per owner, no row-count-sized retained history.
[Side-Effect Policy] No CUDA/torch, file I/O, measurements or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, Work
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from mindie_sim.prediction.ownership import unit_group
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

SOURCE_PATH = "vllm-v0.25.1/csrc/libtorch_stable/layernorm_kernels.cu"
SOURCE_SHA256 = "faaf442f9a520eb7d636acac06b7b52a1765da9b5db13d0d818ff2e1e2537a82"
SOURCE_BYTES = 15657
SOURCE_SCOPE = "current_checkout_vllm_v0.25.1_not_historical_capture"

CCCL_TAG = "v3.0.0"
CCCL_REVISION = "e944297b0b51a8e9c0f1b30425370c31c1c0e0a0"
CCCL_SOURCES = (
    ("block_reduce.cuh", "63b196aa7d6434b487a406bf4792b5cbaf12b504ea1b6f16302fc7393a47e3b1",
     23921, "cub/cub/block/block_reduce.cuh"),
    ("block_reduce_warp_reductions.cuh",
     "894f3a93fbec0c675fe550228ccfe511b825afcb827696fc6a4570e4a0ac6f32",
     9552, "cub/cub/block/specializations/block_reduce_warp_reductions.cuh"),
    ("warp_reduce_shfl.cuh",
     "1bf8a57652fa1bb2d1b23a0e7b34958ded66074854cf67085c2fc65b0544cb61",
     22020, "cub/cub/warp/specializations/warp_reduce_shfl.cuh"),
)
# Supporting headers read as source data, not manifest entries.
SUPPORTING_SOURCES = (
    ("vllm-v0.25.1/csrc/libtorch_stable/type_convert.cuh",
     "9274a7464c53de92976f6b5a19cc548b315f5d85e68b825515a6ff1fd73be4ca",
     "29-97,115-138,178-194"),
    ("vllm-v0.25.1/csrc/libtorch_stable/quantization/vectorization_utils.cuh",
     "991d501211c90a6a5c88d1062723b04c73ca1418d84f8aa07e255bd97128dff8",
     "21-105,130-193"),
    ("vllm-v0.25.1/csrc/libtorch_stable/quantization/vectorization.cuh",
     "6b28cf0cfe9e948c989274adbfa46c6cdec393bf245efb49678213eca15171e7",
     "13-15"),
    ("vllm-v0.25.1/csrc/libtorch_stable/cub_helpers.h",
     "91bd3c5f862387c475806a13245f0353369439a4e83962d2798e820314c48c31",
     "3-12"),
    ("vllm-v0.25.1/csrc/libtorch_stable/dispatch_utils.h",
     "7ec1bb7b43cee1e9d8e8150c799618f53e37b7060b061f628049f454d6272e76",
     "29-31,123-151,153-170"),
    ("vllm-v0.25.1/csrc/core/batch_invariant.hpp",
     "0f7d5937bf059a1fe2a13d8710667d3ec826ba0bf221c0137e477773ad6049bd",
     "1-24"),
)

IMPLEMENTATION = "vllm_cuda_rms_norm"
RMS_NORM = "rms_norm"
FUSED_ADD_RMS_NORM = "fused_add_rms_norm"
KERNELS = (RMS_NORM, FUSED_ADD_RMS_NORM)
SERVICE_ROLES = {
    **dict.fromkeys((
        "fp32_add", "fp32_mul", "fp32_div", "fp32_rsqrt", "warp_shuffle_calls",
        "bf16_add", "fp16_add", "cvt_bf16_fp32", "cvt_fp32_bf16",
        "cvt_fp16_fp32", "cvt_fp32_fp16", "cvt_int32_fp32",
    ), "vector"),
    "shared_read_bytes": "shared_memory",
    "shared_write_bytes": "shared_memory",
}
_DEVICE_KERNEL = {
    RMS_NORM: "vllm::rms_norm_kernel",
    FUSED_ADD_RMS_NORM: "vllm::fused_add_rms_norm_kernel",
}
_KERNEL_TEMPLATE = {
    RMS_NORM: "rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS, HasWeight>",
    FUSED_ADD_RMS_NORM: "fused_add_rms_norm_kernel<scalar_t, width, HasWeight>",
}
_RECIPE = {kernel: "cuda_norm." + kernel for kernel in KERNELS}

_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
_DTYPE_ALIASES = {
    "bfloat16": "bf16", "bfloat": "bf16",
    "float16": "fp16", "half": "fp16",
    "float32": "fp32", "float": "fp32",
}

# Source lines 252 and 329: (num_tokens < 256) ? 1024 : 256.
_BLOCK_THRESHOLD_ROWS = 256
_MAX_BLOCK_SMALL = 1024
_MAX_BLOCK_LARGE = 256
# CUDA warp width and CUB's Log2<32> shuffle steps for WarpReduceShfl<float,32>.
_CUB_WARP_THREADS = 32
_CUB_WARP_STEPS = 5
_CUB_BLOCK_TEMPLATE_THREADS = 1024
_CUB_ALGORITHM = "cub::BLOCK_REDUCE_WARP_REDUCTIONS"
# Source line 334: the fused specialization is hard-coded to vector width 8.
_FUSED_VECTOR_WIDTH = 8
# Source lines 28, 68-70, 120, 141-143, 186, 197-199.
_SHARED_S_VARIANCE_BYTES = 4
_CUB_WARP_AGGREGATE_BYTES = _CUB_WARP_THREADS * 4  # T warp_aggregates[WARPS]
_CUB_BLOCK_PREFIX_BYTES = 4
_CUB_TEMP_PAYLOAD_BYTES = _CUB_WARP_AGGREGATE_BYTES + _CUB_BLOCK_PREFIX_BYTES
_SHARED_PAYLOAD_BYTES = _CUB_TEMP_PAYLOAD_BYTES + _SHARED_S_VARIANCE_BYTES

_PARAMS = frozenset({
    "rows", "width", "dtype", "kernel", "has_weight", "aligned",
    "batch_invariant", "input_stride",
})


@dataclass(frozen=True)
class Dispatch:
    """Launch geometry and reduction structure selected by the source."""

    kernel: str
    rows: int
    width: int
    num_dims: int
    dtype: str
    dtype_bytes: int
    grid: int
    block: int
    warps: int
    full_warps: int
    partial_warp_lanes: int
    max_block_size: int
    vector_width: int
    vectorized: bool
    vector_path: str
    has_weight: bool
    aligned: bool
    batch_invariant: bool
    input_stride: int
    warp_shuffle_steps: int
    warp_lane_adds: int
    cross_warp_adds: int
    warp_aggregate_writes: int
    warp_aggregate_reads: int

    @property
    def device_kernel(self) -> str:
        return _DEVICE_KERNEL[self.kernel]

    @property
    def kernel_template(self) -> str:
        return _KERNEL_TEMPLATE[self.kernel]

    @property
    def active_lanes(self) -> int:
        return self.block


def _canonical_dtype(value: object) -> str:
    key = str(value).strip().lower()
    key = _DTYPE_ALIASES.get(key, key)
    if key not in _DTYPE_BYTES:
        raise ModelingError(f"cuda_norm_dtype_unsupported:{value}")
    return key


def _positive_int(value: object, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(reason)
    return int(value)


def _boolean(value: object, reason: str) -> bool:
    if not isinstance(value, bool):
        raise ModelingError(reason)
    return value


def _max_block_size(rows: int) -> int:
    return _MAX_BLOCK_SMALL if rows < _BLOCK_THRESHOLD_ROWS else _MAX_BLOCK_LARGE


def _warp_step_adds(active_lanes: int) -> int:
    """PTX predicate: only shuffled source lanes inside the segment add."""
    if active_lanes < 1:
        return 0
    return sum(max(0, active_lanes - (1 << step))
               for step in range(_CUB_WARP_STEPS))


def _reduction_structure(block: int) -> tuple[int, int, int, int]:
    """CUB lines244-251 use full/partial warps; lines141-146 select successors."""
    full_warps, partial_lanes = divmod(block, _CUB_WARP_THREADS)
    warps = full_warps + (1 if partial_lanes else 0)
    warp_adds = full_warps * _warp_step_adds(_CUB_WARP_THREADS)
    if partial_lanes:
        warp_adds += _warp_step_adds(partial_lanes)
    cross_warp_adds = sum(1 for warp in range(1, _CUB_WARP_THREADS)
                          if warp * _CUB_WARP_THREADS < block)
    return warps, warp_adds, cross_warp_adds, _CUB_WARP_STEPS * warps


def source_dispatch(
    *,
    rows: int,
    width: int,
    dtype: str,
    kernel: str = RMS_NORM,
    has_weight: bool = True,
    aligned: bool = True,
    batch_invariant: bool = False,
    input_stride: int | None = None,
    num_dims: int = 2,
) -> Dispatch:
    """Reconstruct the source launch and reduction structure without a device."""
    rows = _positive_int(rows, "cuda_norm_rows_invalid")
    width = _positive_int(width, "cuda_norm_width_invalid")
    dtype = _canonical_dtype(dtype)
    if kernel not in KERNELS:
        raise ModelingError(f"cuda_norm_kernel_unknown:{kernel}")
    has_weight = _boolean(has_weight, "cuda_norm_has_weight_invalid")
    aligned = _boolean(aligned, "cuda_norm_aligned_invalid")
    batch_invariant = _boolean(batch_invariant, "cuda_norm_batch_invariant_invalid")
    if num_dims not in (2, 3, 4):
        # Source dispatch_utils.h:153-170 accepts rank 2, 3 or 4 only.
        raise ModelingError(f"cuda_norm_rank_unsupported:{num_dims}")
    stride = width if input_stride is None else _positive_int(
        input_stride, "cuda_norm_input_stride_invalid")
    max_block = _max_block_size(rows)
    dtype_bytes = _DTYPE_BYTES[dtype]

    if kernel == RMS_NORM:
        # Source263-267; gcd guarantees the selected vector divides width.
        vector_width = math.gcd(16 // dtype_bytes, width)
        block = min(width // vector_width, max_block)
        vectorized = vector_width > 1
        alignment_bytes = vector_width * dtype_bytes
        if vector_width == 1:
            vector_path = "scalar_width_one"
        elif not aligned:
            vector_path = "runtime_input_alignment"
        elif (stride * dtype_bytes) % alignment_bytes == 0:
            vector_path = "fully_vectorized_rows"
        else:
            vector_path = "row_prefix_vector_tail"
    else:
        # Source329-361 gates the width-8 body; generic width-0 handles tails.
        block = min(width, max_block)
        vectorized = (
            aligned
            and not batch_invariant
            and width % _FUSED_VECTOR_WIDTH == 0
            and stride % _FUSED_VECTOR_WIDTH == 0
        )
        vector_width = _FUSED_VECTOR_WIDTH if vectorized else 0
        vector_path = "fused_width8" if vectorized else "fused_generic_width0"

    warps, warp_adds, cross_warp_adds, shuffle_steps = _reduction_structure(block)
    full_warps, partial_lanes = divmod(block, _CUB_WARP_THREADS)
    return Dispatch(
        kernel=kernel, rows=rows, width=width, num_dims=num_dims, dtype=dtype,
        dtype_bytes=dtype_bytes, grid=rows, block=block, warps=warps,
        full_warps=full_warps, partial_warp_lanes=partial_lanes,
        max_block_size=max_block, vector_width=vector_width,
        vectorized=vectorized, vector_path=vector_path, has_weight=has_weight,
        aligned=aligned, batch_invariant=batch_invariant, input_stride=stride,
        warp_shuffle_steps=shuffle_steps, warp_lane_adds=warp_adds,
        cross_warp_adds=cross_warp_adds, warp_aggregate_writes=warps,
        warp_aggregate_reads=max(0, warps - 1),
    )


def _reduction_metadata(dispatch: Dispatch) -> dict:
    return {
        "algorithm": _CUB_ALGORITHM,
        "template": f"cub::BlockReduce<float, {_CUB_BLOCK_TEMPLATE_THREADS}>",
        "pinned_cccl_tag": CCCL_TAG,
        "pinned_cccl_revision": CCCL_REVISION,
        "cccl_role": "declared_source_implementation_not_observed_toolchain",
        "template_warps": _CUB_WARP_THREADS,
        "logical_warp_size": _CUB_WARP_THREADS,
        "warp_threads": _CUB_WARP_THREADS,
        "warp_reduce_specialization": "cub::WarpReduceShfl<float, 32>",
        "warp_steps": _CUB_WARP_STEPS,
        "warp_shuffle_steps": dispatch.warp_shuffle_steps,
        "warp_lane_adds": dispatch.warp_lane_adds,
        "active_warps": dispatch.warps,
        "full_warps": dispatch.full_warps,
        "partial_warp_lanes": dispatch.partial_warp_lanes,
        "num_valid": dispatch.block,
        "full_tile": dispatch.block >= _CUB_BLOCK_TEMPLATE_THREADS,
        "cross_warp_adds": dispatch.cross_warp_adds,
        "cross_warp_chain": "thread0_sequential_over_lane0_aggregates",
        "cross_warp_shuffle_steps": 0,
        "successor_predicate": "warp * 32 < num_valid",
        "warp_aggregate_writes": dispatch.warp_aggregate_writes,
        "warp_aggregate_reads": dispatch.warp_aggregate_reads,
        "barriers": 2,
        "active_lane_shuffle_calls": dispatch.block * _CUB_WARP_STEPS,
        "warp_issue_cost": "source_warp_calls_separate_from_predicated_lane_adds",
        "compiled_instruction_count": "unverified",
        "register_count": "unresolved",
        "dependency_latency": "unresolved",
        "compiled_occupancy": "not_declared_by_source",
    }


def _shared_memory_metadata() -> dict:
    return {
        "s_variance_bytes": _SHARED_S_VARIANCE_BYTES,
        "cub_warp_aggregates_bytes": _CUB_WARP_AGGREGATE_BYTES,
        "cub_block_prefix_bytes": _CUB_BLOCK_PREFIX_BYTES,
        "cub_temp_payload_bytes": _CUB_TEMP_PAYLOAD_BYTES,
        "declared_payload_bytes": _SHARED_PAYLOAD_BYTES,
        "template_warp_slots": _CUB_WARP_THREADS,
        "cub_warp_reduce_temp_array": "NullType[32]; size/padding unresolved",
        "exact_cub_temp_size": "unresolved",
        "exact_size_reason": (
            "WarpReduceShfl::TempStorage is NullType and its array/padding "
            "definitions are outside the supplied CCCL closure; compiled "
            "sizeof(BlockReduce<float,1024>::TempStorage) was not observed"
        ),
        "lifetime": "CTA admission through last output store",
        "residency": "owner_bound_smem_then_shared_pool_else_unconstrained",
        "size_basis": "source_payload_lower_bound_not_compiled_sizeof",
        "service_basis": "source_float_accesses_without_compiler_hoisting_credit",
    }


def _source_validity(dispatch: Dispatch) -> dict:
    """RMS second-pass vec_n_t requires aligned bases and every visited row."""
    if (dispatch.kernel == RMS_NORM and dispatch.vector_width > 1
            and (not dispatch.aligned or (dispatch.rows > 1
                 and dispatch.input_stride % dispatch.vector_width != 0))):
        return {
            "valid": False,
            "reason": "standalone second-pass vec_n_t row pointer misaligned",
            "source_lines": "78-81,83; vectorization.cuh:14",
            "input_stride": dispatch.input_stride,
            "vector_width": dispatch.vector_width,
            "disposition": (
                "reported, not silently executed: the selected host dispatch "
                "still launches the kernel, but CUDA defines no result for "
                "this row alignment"
            ),
        }
    return {
        "valid": True,
        "reason": "",
        "source_lines": "",
        "input_stride": dispatch.input_stride,
        "vector_width": dispatch.vector_width,
    }


def _dispatch_metadata(dispatch: Dispatch) -> dict:
    """The complete source dispatch retained on every selected path."""
    validity = _source_validity(dispatch)
    return {
        "implementation": IMPLEMENTATION,
        "pattern": "cuda_rms_norm",
        "kernel": dispatch.kernel,
        "kernel_function": dispatch.device_kernel,
        "device_kernel": dispatch.device_kernel,
        "kernel_template": dispatch.kernel_template,
        "host_function": f"vllm::{dispatch.kernel}",
        "host_entry": dispatch.kernel,
        "grid_blocks": dispatch.grid,
        "grid": [dispatch.grid],
        "block_threads": dispatch.block,
        "block": dispatch.block,
        "warps_per_block": dispatch.warps,
        "warp_threads": _CUB_WARP_THREADS,
        "max_block_size": dispatch.max_block_size,
        "block_threshold_rows": _BLOCK_THRESHOLD_ROWS,
        "block_rule": (
            "grid=num_tokens; max_block=(rows<256?1024:256); "
            "rms block=min(hidden/gcd(16/sizeof,hidden),max_block); "
            "fused block=min(hidden,max_block)"
        ),
        "vector_width": dispatch.vector_width,
        "vec_width": dispatch.vector_width,
        "vectorized": dispatch.vectorized,
        "vector_path": dispatch.vector_path,
        "vector_rule": (
            "rms VEC_SIZE=gcd(16/sizeof,hidden); fused width=8 iff "
            "aligned and not batch_invariant and hidden%8==0 and "
            "input_stride%8==0 else width=0"
        ),
        "num_dims": dispatch.num_dims,
        "rows": dispatch.rows,
        "width": dispatch.width,
        "dtype": dispatch.dtype,
        "dtype_bytes": dispatch.dtype_bytes,
        "input_stride": dispatch.input_stride,
        "has_weight": dispatch.has_weight,
        "aligned": dispatch.aligned,
        "batch_invariant": dispatch.batch_invariant,
        "residual_fusion": dispatch.kernel == FUSED_ADD_RMS_NORM,
        "source_execution_valid": validity["valid"],
        "source_validity": validity,
        "cccl_tag": CCCL_TAG,
        "cccl_revision": CCCL_REVISION,
        "cccl_role": "pinned_source_implementation_not_observed_toolchain",
        "reduction": _reduction_metadata(dispatch),
        "shared_memory": _shared_memory_metadata(),
        "source": {
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "bytes": SOURCE_BYTES,
            "scope": SOURCE_SCOPE,
        },
        "cccl": {
            "tag": CCCL_TAG,
            "revision": CCCL_REVISION,
            "sources": [
                {"name": name, "sha256": digest, "bytes": size, "path": path}
                for name, digest, size, path in CCCL_SOURCES
            ],
            "role": "pinned_source_implementation_not_observed_toolchain",
        },
        "compiled_branch_assumptions": {
            "cuda_version": ">=12.0 (type_convert.cuh:66-97 enables Half/BFloat16)",
            "cuda_arch": ">=800 for bfloat16 (type_convert.cuh:82-97)",
            "cub_add_op": "CUB_VERSION>=200800 selects cuda::std::plus<>",
            "rocm_branch": "not_selected",
        },
        "source_closure": [
            {"path": path, "sha256": digest, "lines": lines}
            for path, digest, lines in SUPPORTING_SOURCES
        ],
    }


def make_op(params: dict) -> OpSpec:
    """Build the explicit geometry adapter for one campaign case."""
    if not isinstance(params, Mapping):
        raise ModelingError("cuda_norm_params_must_be_mapping")
    unknown = set(params) - _PARAMS
    if unknown:
        raise ModelingError("cuda_norm_unknown_parameter:" + ",".join(sorted(unknown)))
    rows = _positive_int(params.get("rows"), "cuda_norm_rows_invalid")
    width = _positive_int(params.get("width"), "cuda_norm_width_invalid")
    dtype = _canonical_dtype(params.get("dtype"))
    kernel = str(params.get("kernel", RMS_NORM))
    dispatch = source_dispatch(
        rows=rows,
        width=width,
        dtype=dtype,
        kernel=kernel,
        has_weight=params.get("has_weight", True),
        aligned=params.get("aligned", True),
        batch_invariant=params.get("batch_invariant", False),
        input_stride=params.get("input_stride", width),
    )
    dispatch_dict = _dispatch_metadata(dispatch)
    implementation = {
        "kernel": dispatch.kernel,
        "device_kernel": dispatch.device_kernel,
        "block_threads": dispatch.block,
        "max_block_size": dispatch.max_block_size,
        "vector_width": dispatch.vector_width,
        "vectorized": dispatch.vectorized,
        "vector_path": dispatch.vector_path,
        "has_weight": dispatch.has_weight,
        "aligned": dispatch.aligned,
        "batch_invariant": dispatch.batch_invariant,
        "reduction_algorithm": _CUB_ALGORITHM,
        "cccl_revision": CCCL_REVISION,
    }
    metadata = (
        ("kernel", dispatch.kernel),
        ("device_kernel", dispatch.device_kernel),
        ("source", SOURCE_PATH),
        ("source_sha256", SOURCE_SHA256),
        ("source_bytes", SOURCE_BYTES),
        ("source_revision", SOURCE_SCOPE),
        ("source_lines",
         "rms_norm:14-100,219-289;fused_add:106-173,178-215,291-363"),
        ("cccl_tag", CCCL_TAG),
        ("cccl_revision", CCCL_REVISION),
        ("cccl_role", "pinned_source_implementation_not_observed_toolchain"),
        ("dispatch", dispatch_dict),
        ("grid_blocks", dispatch.grid),
        ("block_threads", dispatch.block),
        ("warps_per_block", dispatch.warps),
        ("warp_threads", _CUB_WARP_THREADS),
        ("max_block_size", dispatch.max_block_size),
        ("vector_width", dispatch.vector_width),
        ("vectorized", dispatch.vectorized),
        ("vector_path", dispatch.vector_path),
        ("input_stride", dispatch.input_stride),
        ("has_weight", dispatch.has_weight),
        ("aligned", dispatch.aligned),
        ("batch_invariant", dispatch.batch_invariant),
        ("residual_fusion", dispatch.kernel == FUSED_ADD_RMS_NORM),
        ("reduction", _CUB_ALGORITHM),
        ("warp_steps", _CUB_WARP_STEPS),
        ("warp_lane_adds", dispatch.warp_lane_adds),
        ("cross_warp_adds", dispatch.cross_warp_adds),
        ("cross_warp_chain", "thread0_sequential_over_lane0_aggregates"),
        ("shared_s_variance_bytes", _SHARED_S_VARIANCE_BYTES),
        ("reduction_temp_bytes", _CUB_TEMP_PAYLOAD_BYTES),
        ("shared_cub_temp_payload_bytes", _CUB_TEMP_PAYLOAD_BYTES),
        ("shared_declared_payload_bytes", _SHARED_PAYLOAD_BYTES),
        ("shared_exact_temp_size", "unresolved"),
        ("norm_epsilon", "caller_parameter"),
        ("cache_reuse_credit", "none"),
        ("implementation", implementation),
    )
    notes = (
        "source_lines=rms_norm:14-100,219-289;fused_add:106-215,291-363",
        "block_dispatch=source,not_calibrated",
        "reduction=cub_warp_shfl_predicated_adds+thread0_sequential_aggregate",
        "hbm_bytes=source_level_global_element_traffic_without_cache_credit",
        "cccl=v3.0.0_e944297b_pinned_source_implementation_not_toolchain_identity",
    )
    return OpSpec(
        op_id=f"cuda_norm.{dispatch.kernel}.{rows}x{width}.{dtype}",
        family="rmsnorm",
        phase="prefill",
        stage=0,
        layer=0,
        requests=(),
        semantic_shape=(rows, width),
        local_shape=(rows, width),
        padded_shape=(rows, width),
        launch_shape=(dispatch.grid, dispatch.block),
        dtype=dtype,
        layout="contiguous",
        recipe=_RECIPE[dispatch.kernel], source_boundary="kernel",
        returns=("residual",) if dispatch.kernel == FUSED_ADD_RMS_NORM else (),
        metadata=metadata,
        notes=notes,
    )


def _pass_work(d: Dispatch, elements: int, second: bool) -> tuple[Work, ...]:
    """Source values, not compiled instruction counts; casts retain directions."""
    fused = d.kernel == FUSED_ADD_RMS_NORM
    size, weight = d.dtype_bytes, int(d.has_weight)
    compute = []
    if second:
        read = {"hbm": elements * size * (1 + weight),
                "shared_read_bytes": elements * 4}
        write = {"hbm": elements * size}
        if d.dtype != "fp32":
            compute.append({f"cvt_{d.dtype}_fp32": elements * (1 + weight)})
        compute.append({"fp32_mul": elements})
        if weight:
            compute.append({"fp32_mul": elements})
        if d.dtype != "fp32":
            compute.append({f"cvt_fp32_{d.dtype}": elements})
    else:
        read = {"hbm": elements * size * (2 if fused else 1)}
        if fused:
            compute.append({d.dtype + "_add": elements})
        if d.dtype != "fp32":
            compute.append({f"cvt_{d.dtype}_fp32": elements})
        compute.extend(({"fp32_mul": elements}, {"fp32_add": elements}))
        if fused and d.vectorized:
            # sum_squares has a fresh local sum; variance += adds once more.
            compute.append({"fp32_add": elements // d.vector_width})
        write = {"hbm": elements * size} if fused else {}
    return tuple(Work(resources=work) for work in (read, *compute, write))


def _row_work(d: Dispatch) -> Work:
    total = Work(resources={
        "warp_shuffle_calls": d.warp_shuffle_steps,
        "fp32_add": d.warp_lane_adds + d.cross_warp_adds + 1,
        "cvt_int32_fp32": 1, "fp32_div": 1, "fp32_rsqrt": 1,
        "shared_write_bytes": 4 * (d.warps + 1),
        "shared_read_bytes": 4 * d.cross_warp_adds,
    })
    for second in (False, True):
        for work in _pass_work(d, d.width, second):
            total += work
    return total


def _cta_program(d: Dispatch, context: RecipeContext, placement) -> KernelProgram:
    """Warps can advance independently until the two source block barriers."""
    vector = d.vector_width if d.vector_width > 0 else 1
    chunks = d.width // vector

    def loop(warp, second):
        first_thread = warp * 32
        count = max(0, (chunks - 1 - first_thread) // d.block + 1)

        def tile(iteration):
            start = first_thread + iteration * d.block
            elements = min(32, max(0, chunks - start), d.block - first_thread) * vector
            work = _pass_work(d, elements, second)
            names = ("read", *(f"compute{i}" for i in range(len(work) - 2)), "write")
            return tuple(Stage(name, context.bind(part),
                               after=(names[index - 1],) if index else (),
                               placement=placement)
                         for index, (name, part) in enumerate(zip(names, work)))
        return PipelineProgram(count, 1, tile)

    def first_warp(warp):
        stages = [Stage("elements", program=loop(warp, False))]
        valid = min(32, d.block - warp * 32)
        tail = "elements"
        for step in range(_CUB_WARP_STEPS):
            shuffle, add = f"shuffle{step}", f"add{step}"
            stages.append(Stage(shuffle, context.bind(Work(resources={
                "warp_shuffle_calls": 1})), after=(tail,), placement=placement))
            stages.append(Stage(add, context.bind(Work(resources={
                "fp32_add": max(0, valid - (1 << step))})),
                after=(shuffle,), placement=placement))
            tail = add
        stages.append(Stage("aggregate_store", context.bind(Work(resources={
            "shared_write_bytes": 4})), after=(tail,), placement=placement))
        return KernelProgram(tuple(stages))

    stages = [Stage("first_pass", program=ParallelProgram(tuple(
        first_warp(warp) for warp in range(d.warps))))]
    tail = "first_pass"
    for successor in range(1, d.warps):
        read, add = f"aggregate_read{successor}", f"aggregate_add{successor}"
        stages.append(Stage(read, context.bind(Work(resources={
            "shared_read_bytes": 4})), after=(tail,), placement=placement))
        stages.append(Stage(add, context.bind(Work(resources={
            "fp32_add": 1})), after=(read,), placement=placement))
        tail = add
    for name, work in (
        ("width_float", {"cvt_int32_fp32": 1}),
        ("divide", {"fp32_div": 1}), ("epsilon", {"fp32_add": 1}),
        ("rsqrt", {"fp32_rsqrt": 1}), ("variance_store", {"shared_write_bytes": 4}),
    ):
        stages.append(Stage(name, context.bind(Work(resources=work)),
                            after=(tail,), placement=placement))
        tail = name
    stages.append(Stage("second_pass", after=(tail,), program=ParallelProgram(
        tuple(loop(warp, True) for warp in range(d.warps)))))
    return KernelProgram(tuple(stages))


def _validate_dispatch(metadata: Mapping, dispatch: Dispatch) -> None:
    """Reject a plan whose retained dispatch disagrees with the source rule."""
    raw = metadata.get("dispatch")
    expected = {
        "kernel": dispatch.kernel,
        "host_function": f"vllm::{dispatch.kernel}",
        "device_kernel": dispatch.device_kernel,
        "kernel_function": dispatch.device_kernel,
        "grid_blocks": dispatch.grid,
        "grid": [dispatch.grid],
        "block_threads": dispatch.block,
        "block": dispatch.block,
        "warps_per_block": dispatch.warps,
        "max_block_size": dispatch.max_block_size,
        "vector_width": dispatch.vector_width,
        "vec_width": dispatch.vector_width,
        "vectorized": dispatch.vectorized,
        "vector_path": dispatch.vector_path,
        "rows": dispatch.rows,
        "width": dispatch.width,
        "dtype": dispatch.dtype,
        "dtype_bytes": dispatch.dtype_bytes,
        "input_stride": dispatch.input_stride,
        "has_weight": dispatch.has_weight,
        "aligned": dispatch.aligned,
        "batch_invariant": dispatch.batch_invariant,
        "num_dims": dispatch.num_dims,
        "residual_fusion": dispatch.kernel == FUSED_ADD_RMS_NORM,
    }
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ModelingError("cuda_norm_dispatch_metadata_invalid")
        for key, value in expected.items():
            if key not in raw:
                continue
            actual = raw[key]
            if key == "dtype":
                try:
                    actual = _canonical_dtype(actual)
                except ModelingError as error:
                    raise ModelingError(
                        f"cuda_norm_dispatch_metadata_mismatch:dtype") from error
            if actual != value:
                raise ModelingError(f"cuda_norm_dispatch_metadata_mismatch:{key}")
    # Legacy/top-level mirrors must also agree when present.
    for key in ("kernel", "device_kernel", "grid_blocks", "block_threads",
                "warps_per_block", "max_block_size", "vector_width",
                "vectorized", "has_weight", "aligned", "batch_invariant",
                "input_stride"):
        if key in metadata and key in expected and metadata[key] != expected[key]:
            raise ModelingError(f"cuda_norm_dispatch_metadata_mismatch:{key}")


def _require_owner_geometry(owners: int, units: int, role: str) -> None:
    if units < 1:
        raise ModelingError(f"cuda_norm_owner_units_invalid:{role}:{units}")
    compatible = (
        (units >= owners and units % owners == 0)
        or (units < owners and owners % units == 0)
    )
    if not compatible:
        raise ModelingError(
            f"cuda_norm_owner_geometry_incompatible:{role}:{owners}:{units}")


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Declare per-owner source CTAs and their total physical work."""
    shape = tuple(int(value) for value in op.local_shape)
    if len(shape) < 2 or any(value < 1 for value in shape):
        raise ModelingError("cuda_norm_local_shape_invalid")
    if len(shape) > 4:
        # Source dispatch_utils.h:153-170 supports rank 2, 3 and 4 only.
        raise ModelingError("cuda_norm_rank_unsupported")
    width = shape[-1]
    rows = math.prod(shape[:-1])
    metadata = dict(op.metadata)

    raw_dispatch = metadata.get("dispatch")
    raw_dispatch = raw_dispatch if isinstance(raw_dispatch, Mapping) else {}
    kernel = raw_dispatch.get("kernel", metadata.get("kernel"))
    if kernel is None:
        kernel = FUSED_ADD_RMS_NORM if "fused" in op.recipe else RMS_NORM
    dispatch = source_dispatch(
        rows=rows,
        width=width,
        dtype=op.dtype,
        kernel=kernel,
        has_weight=metadata.get("has_weight", raw_dispatch.get("has_weight", True)),
        aligned=metadata.get("aligned", raw_dispatch.get("aligned", True)),
        batch_invariant=metadata.get(
            "batch_invariant", raw_dispatch.get("batch_invariant", False)),
        input_stride=metadata.get(
            "input_stride", raw_dispatch.get("input_stride", width)),
        num_dims=len(shape),
    )
    _validate_dispatch(metadata, dispatch)
    if not _source_validity(dispatch)["valid"]:
        raise ModelingError("cuda_norm_source_requires_aligned_vector_rows")

    units = int(geometry.compute_units)
    if units < 1:
        raise ModelingError("cuda_norm_compute_units_invalid")
    vector_units = int(geometry.resource_units.get("vector", units))
    _require_owner_geometry(units, vector_units, "vector")

    total = _row_work(dispatch).scaled(rows)
    owners = units
    pool = next((name for name in ("smem", "shared") if name in geometry.buffers), None)

    def program(context: RecipeContext):
        if int(context.core_count) != owners:
            raise ModelingError(
                f"cuda_norm_owner_count_mismatch:{owners}:{context.core_count}")
        _require_owner_geometry(
            owners, int(context.resource_units.get("vector", owners)), "vector")

        def owner_stream(owner: int):
            # One resident CTA per owner is explicit; compiled occupancy is unknown.
            count = (rows - 1 - owner) // owners + 1 if owner < rows else 0
            if count <= 0:
                return None
            placement = context.owned(owner, owners, SERVICE_ROLES)
            shared = ()
            if pool is not None:
                shared = (context.allocate("cta_shared", pool, _SHARED_PAYLOAD_BYTES,
                          unit_group(owner, owners,
                                     context.pool_units.get(pool, owners))),)

            def tile(_index: int):
                return (
                    Stage("cta", acquire=shared, free=shared,
                          program=_cta_program(dispatch, context, placement)),
                )

            return PipelineProgram(count, 1, tile)

        return ParallelProgram(tuple(
            stream for stream in (owner_stream(owner) for owner in range(owners))
            if stream is not None))

    return OperatorPlan(total, program)


__all__ = [
    "CCCL_REVISION", "CCCL_SOURCES", "CCCL_TAG", "Dispatch",
    "FUSED_ADD_RMS_NORM", "IMPLEMENTATION", "KERNELS", "RMS_NORM",
    "SERVICE_ROLES", "SOURCE_PATH", "SOURCE_SHA256", "SUPPORTING_SOURCES", "build", "make_op",
    "source_dispatch",
]
