"""Source-phase model of CUDA dynamic per-token symmetric INT8 quantization.

[Role] Lower the no-AZP vLLM kernel and pinned CUB float-max reduction.
[Invariants] Source lane work, warp calls and bytes have distinct service rates.
  CUB max is predicated; scale and INT8 stores depend on their producers.
  A CTA owns one compute unit; global memory remains shared. Pointer alignment
  is explicit even for contiguous tensors. No compiler latency is invented.
[Timeouts & Locks] One lazy row pipeline per active owner; bounded in row count.
[Side-Effect Policy] Pure metadata/program construction; no device or source I/O.
"""
from __future__ import annotations

from math import gcd, prod

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..mechanism import Geometry, OperatorPlan, Work
from ..ownership import unit_group
from ..program import ParallelProgram, PipelineProgram, Stage

IMPLEMENTATION = "cuda_dynamic_int8_quant_symmetric_no_azp"
RECIPE = "cuda_dynamic_int8_quant"
KERNEL = "dynamic_scaled_int8_quant_kernel"
SHARED_BYTES = 48
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
SERVICE_ROLES = {
    **dict.fromkeys(("fp32_abs", "fp32_max", "fp32_div", "fp32_mul",
                     "fp32_compare", "fp32_select", "warp_shuffle_calls",
                     "cvt_bf16_fp32", "cvt_fp16_fp32", "cvt_fp32_int8"), "vector"),
    "shared_read_bytes": "shared_memory", "shared_write_bytes": "shared_memory",
}
SOURCES = (
    ("vllm-v0.25.1/csrc/libtorch_stable/quantization/w8a8/int8/scaled_quant.cu",
     "e6fc36b6ddc925e40e0a7a84543dd29da456c19672e0daf28f8287d2e3645243",
     "10-37,142-178,300-324"),
    ("vllm-v0.25.1/csrc/libtorch_stable/quantization/vectorization_utils.cuh",
     "991d501211c90a6a5c88d1062723b04c73ca1418d84f8aa07e255bd97128dff8",
     "21-105,130-193"),
    ("cccl-v3.0.0/block_reduce_warp_reductions.cuh",
     "894f3a93fbec0c675fe550228ccfe511b825afcb827696fc6a4570e4a0ac6f32",
     "94-110,137-197,238-254"),
    ("cccl-v3.0.0/warp_reduce_shfl.cuh",
     "1bf8a57652fa1bb2d1b23a0e7b34958ded66074854cf67085c2fc65b0544cb61",
     "448-458,528-577"),
)


def cub_inventory(block):
    """CUB float ReduceStep: every lane shuffles; only valid peers apply max."""
    warps = tuple(min(32, block - start) for start in range(0, block, 32))
    steps = tuple({
        "offset": offset, "warp_shuffle_calls": len(warps),
        "shuffle_active_lanes": block,
        "max_active_lanes": sum(max(0, lanes - offset) for lanes in warps),
    } for offset in (1, 2, 4, 8, 16))
    return {
        "block_threads": block, "template_threads": 256,
        "warp_shuffle_calls": 5 * len(warps), "shuffle_active_lanes": 5 * block,
        "max_active_lanes": sum(step["max_active_lanes"] for step in steps),
        "thread0_max_calls": len(warps) - 1,
        "warp_aggregate_stores": len(warps), "steps": steps,
    }


def _read_traversal(address, width, size):
    prefix = min((-address % (16 * size)) // size, width)
    packs, tail = divmod(width - prefix, 16)
    return prefix, packs, tail, False


def _quant_traversal(address, output_address, width, size):
    prefix, packs, tail, _ = _read_traversal(address, width, size)
    if (output_address + prefix) % 16:
        return 0, 0, width, True
    return prefix, packs, tail, False


def traversal_summary(rows, width, dtype, input_byte_offset=0, output_byte_offset=0):
    """Enumerate at most16 distinct row-alignment states, including storage offsets."""
    size = DTYPE_BYTES[dtype]
    period = 16 // gcd(width, 16)
    quotient, remainder = divmod(rows, period)
    result = {name: dict.fromkeys(("fast_rows", "scalar_rows", "prefix_elements",
                                   "vector_packs", "tail_elements", "vector_elements"), 0)
              for name in ("read", "quant")}
    for index in range(min(rows, period)):
        count = quotient + (index < remainder)
        address = input_byte_offset + index * width * size
        output = output_byte_offset + index * width
        paths = (_read_traversal(address, width, size),
                 _quant_traversal(address, output, width, size))
        for name, (prefix, packs, tail, scalar) in zip(result, paths):
            entry = result[name]
            entry["fast_rows"] += count * (not scalar and not prefix and not tail)
            entry["scalar_rows"] += count * scalar
            entry["prefix_elements"] += count * prefix
            entry["vector_packs"] += count * packs
            entry["tail_elements"] += count * tail
            entry["vector_elements"] += count * packs * 16
    return result


def source_phases(width, dtype, *, zero_absmax=False):
    """Source-data dependencies; these service envelopes are not a SASS schedule."""
    block = min(width, 256)
    cub = cub_inventory(block)
    read = width * DTYPE_BYTES[dtype]
    phases = []

    def add(name, **resources):
        phases.append((name, Work(resources={key: value for key, value in resources.items()
                                             if value})))

    add("read_absmax_input", hbm=read, mte2=read)
    if dtype != "fp32":
        add("convert_absmax_input", **{f"cvt_{dtype}_fp32": width})
    add("absolute", fp32_abs=width)
    add("thread_max", fp32_max=width)
    for step in cub["steps"]:
        add(f"shuffle_{step['offset']}", warp_shuffle_calls=step["warp_shuffle_calls"])
        add(f"warp_max_{step['offset']}", fp32_max=step["max_active_lanes"])
    add("share_warp_max", shared_write_bytes=4 * cub["warp_aggregate_stores"])
    add("cub_barrier")
    for warp in range(1, cub["warp_aggregate_stores"]):
        add(f"read_warp_{warp}", shared_read_bytes=4)
        add(f"combine_warp_{warp}", fp32_max=1)
    add("publish_absmax", shared_write_bytes=4)
    add("read_scale_absmax", shared_read_bytes=4)
    add("compute_scale", fp32_div=1)
    add("store_scale", hbm=4, mte3=4)
    add("absmax_barrier")
    add("read_inverse_predicate", shared_read_bytes=4 * block)
    add("test_absmax_zero", fp32_compare=block)
    if not zero_absmax:
        add("read_inverse_absmax", shared_read_bytes=4 * block)
        add("inverse", fp32_div=block)
    add("select_inverse", fp32_select=block)
    add("read_quant_input", hbm=read, mte2=read)
    if dtype != "fp32":
        add("convert_quant_input", **{f"cvt_{dtype}_fp32": width})
    add("multiply_inverse", fp32_mul=width)
    add("convert_int8", cvt_fp32_int8=width)
    add("store_int8", hbm=width, mte3=width)
    return tuple(phases)


def _total(phases):
    return sum((work for _, work in phases), Work())


def _geometry(op):
    if not op.local_shape or any(value < 1 or int(value) != value for value in op.local_shape):
        raise ModelingError("cuda_dynamic_int8_quant_requires_positive_shape")
    if op.dtype not in DTYPE_BYTES or op.layout != "contiguous":
        raise ModelingError("cuda_dynamic_int8_quant_requires_contiguous_fp32_fp16_or_bf16")
    return prod(op.local_shape[:-1]), op.local_shape[-1], op.dtype


def _dispatch(rows, width, dtype, input_offset, output_offset, zero_absmax):
    return {
        "implementation": IMPLEMENTATION, "host_entry": "dynamic_scaled_int8_quant",
        "kernel_variant": KERNEL, "source_branch": "dynamic_scaled_int8_quant_no_azp",
        "azp_present": False, "input_dtype": dtype, "output_dtype": "int8",
        "scale_dtype": "fp32", "grid_blocks": rows, "block_threads": min(width, 256),
        "vector_width": 16, "input_byte_offset": input_offset,
        "output_byte_offset": output_offset, "zero_absmax": zero_absmax,
    }


def make_op(params):
    allowed = {"rows", "width", "dtype", "input_byte_offset", "output_byte_offset",
               "zero_absmax", "implementation", "azp", "block_threads", "vector_width"}
    if params.keys() - allowed:
        raise ModelingError(f"cuda_dynamic_int8_quant_unknown_params:{sorted(params.keys() - allowed)}")
    rows, width = params["rows"], params["width"]
    if any(type(value) is not int or value < 1 for value in (rows, width)):
        raise ModelingError("cuda_dynamic_int8_quant_requires_positive_rows_and_width")
    dtype = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        params["dtype"], params["dtype"])
    if dtype not in DTYPE_BYTES:
        raise ModelingError("cuda_dynamic_int8_quant_input_dtype_not_selected")
    if params.get("implementation", IMPLEMENTATION) != IMPLEMENTATION or params.get("azp"):
        raise ModelingError("cuda_dynamic_int8_quant_requires_symmetric_no_azp_branch")
    if (params.get("block_threads", min(width, 256)) != min(width, 256)
            or params.get("vector_width", 16) != 16):
        raise ModelingError("cuda_dynamic_int8_quant_launch_does_not_match_source")
    input_offset, output_offset = (params.get(name, 0) for name in
                                   ("input_byte_offset", "output_byte_offset"))
    if (any(type(value) is not int or value < 0 for value in (input_offset, output_offset))
            or input_offset % DTYPE_BYTES[dtype]):
        raise ModelingError("cuda_dynamic_int8_quant_requires_element_aligned_byte_offsets")
    zero = params.get("zero_absmax", False)
    if type(zero) is not bool:
        raise ModelingError("cuda_dynamic_int8_quant_zero_absmax_requires_declared_boolean")
    dispatch = _dispatch(rows, width, dtype, input_offset, output_offset, zero)
    phases = source_phases(width, dtype, zero_absmax=zero)
    work = _total(phases).scaled(rows)
    metadata = {
        "dispatch": dispatch, "implementation": IMPLEMENTATION,
        "input_byte_offset": input_offset, "output_byte_offset": output_offset,
        "zero_absmax": zero, "cub": cub_inventory(min(width, 256)),
        "traversal": traversal_summary(rows, width, dtype, input_offset, output_offset),
        "source_phases_per_row": {name: dict(work.resources) for name, work in phases},
        "shared_temp_bytes": SHARED_BYTES,
        "shared_capacity_scope": "source_layout_estimate_not_compiled_size; unconstrained_if_pool_absent",
        "service_model": "typed_source_work_with_explicit_rates; not_compiled_issue_or_latency",
        "source_payload_assumption": "shared_lvalue_accesses_without_compiler_register_reuse",
        "memory_assumption": "two_logical_input_passes_no_cache_reuse_or_sector_model",
        "cta_schedule": "one_resident_row_per_owner_source_phase_envelope",
        "barriers_per_row": 2, "barrier_latency": "not_modeled",
        "source_identity": "declared_vllm_checkout_and_cccl_v3.0.0_not_captured_toolchain",
        "sources": [{"path": path, "sha256": sha, "lines": lines} for path, sha, lines in SOURCES],
    }
    op = OpSpec(
        "cuda_dynamic_int8_quant", "quantize", "prefill", 0, 0, (),
        (rows, width), (rows, width), (rows, width), (rows, min(width, 256)),
        recipe=RECIPE, source_boundary="kernel", dtype=dtype, layout="contiguous", returns=("activation_scales",),
        metadata=tuple(sorted(metadata.items())),
    )
    return work.apply(op)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    rows, width, dtype = _geometry(op)
    meta = dict(op.metadata)
    zero = meta.get("zero_absmax", False)
    expected = _dispatch(rows, width, dtype, meta.get("input_byte_offset", 0),
                         meta.get("output_byte_offset", 0), zero)
    if meta.get("dispatch", expected) != expected:
        raise ModelingError("cuda_dynamic_int8_quant_dispatch_does_not_match_source")
    owners = geometry.compute_units
    for role in ("vector", "shared_memory"):
        unit_group(0, owners, geometry.resource_units.get(role, owners))
    pool = next((name for name in ("smem", "shared") if name in geometry.buffers), None)
    if pool:
        if geometry.buffer_units.get(pool, owners) != owners:
            raise ModelingError("cuda_dynamic_int8_quant_shared_pool_requires_compute_owners")
        if geometry.buffers[pool] < SHARED_BYTES:
            raise ModelingError("cuda_dynamic_int8_quant_shared_memory_capacity_exceeded")
    phases = source_phases(width, dtype, zero_absmax=zero)
    work = _total(phases).scaled(rows)

    def program(context):
        programs = []
        for owner in range(min(rows, owners)):
            allocation = ((context.allocate("cub_shared", pool, SHARED_BYTES, (owner,)),)
                          if pool else ())
            owned = context.owned(owner, owners, tuple(SERVICE_ROLES) + ("mte2", "mte3"))
            stages = tuple(Stage(
                name, context.bind(work),
                after=(phases[index - 1][0],) if index else (),
                acquire=allocation if index == 0 else (),
                free=allocation if index == len(phases) - 1 else (),
                placement=owned,
            ) for index, (name, work) in enumerate(phases))
            count = (rows - 1 - owner) // owners + 1
            programs.append(PipelineProgram(count, 1, lambda _, stages=stages: stages))
        return ParallelProgram(tuple(programs))

    return OperatorPlan(work, program)
