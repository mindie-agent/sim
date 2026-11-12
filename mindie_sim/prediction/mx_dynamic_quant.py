"""Raw DynamicMxQuant producer ABI for ordinary MX32 projections.

[Role] Declare the BF16 -> E4M3 + E8M0 group-32 producer used before an
  ordinary QuantBatchMatmulV3 whose runtime method is mx_fp8.
[Invariants] The group is 32 values. The producer returns a packed E4M3
  payload and a separate E8M0 exponent scale; scales are never folded into
  the payload width or re-derived from a header label. The device virtual
  instruction sequence is the source's max-exponent/scale/clamp-cast path,
  not a generic QuantV2 or a fused SwiGLU quantizer.
[Timeouts & Locks] Constant-time arithmetic; no I/O, device context or locks.
[Side-Effect Policy] Pure work/contract descriptions only.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError

SOURCE_R64 = {
    "repository": "cann",
    "image": "vllm-ascend:v0.32.0rc-a5-openeuler-cann9.2.1-20261101-r64",
    "image_sha256": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
    "cann": "9.2.1",
    "path": "dynamic_mx_quant/op_kernel/dynamic_mx_quant.cpp",
    "sha256": "ffe5efb4d41480b7af43841722064682d3bc6e3418f8092853a8d9b1db941785",
    "switch_lines": "88-97",
    "kernel": "arch35/dynamic_mx_quant_tail_axis_fp8.h",
    "kernel_sha256": "8c0d95fe8da060b082de2e0a86384a3a284180e6bd5e1e4fb8d0f94853d4b1b6",
    "struct_sha256": "85ac600ee78aef56b6379af3ab3bfc3b7b9515c86c3777e9ac290a144351c5ce",
    "quant_python_sha256": "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79",
    "quant_utils_sha256": "459625b94d258b8eefb5aefca3525d6f6f64d86097be7593548af19f0a254d85",
    "branch": "tail_axis_block32_fp8",
    "kernel_class": "DynamicMxQuantTailAxisFP8",
    "selected_key": 33,
    "key_basis": "MAX_EXPONENT scaleAlg0 from r64 get_dynamic_mx_quant_scale_alg; keys 33/65 for alg0/1",
    "scope": "actual_glm51_r64_deployment_source",
}

SOURCE_955F = {
    "repository": "cann/ops-nn",
    "revision": "955f8b33985b6d72989ebff341bb9c2c564a51bc",
    "path": "quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis_fp8.h",
    "sha256": "31cf4eb673be3cbe38601a90b210e88cc2ed48692c2d84ce8e5cc8bf5c19ad03",
    "kernel": "DynamicMxQuantTailAxisFP8",
    "selected_key": 33,
    "scope": "old_955f_source_variant_not_r64",
}

SOURCE_UNPINNED = {
    "repository": "unpinned",
    "kernel": "DynamicMxQuantSourceUnpinned",
    "scope": "generic ABI; source revision not witnessed",
    "selected_key": None,
}

SOURCES = {"r64": SOURCE_R64, "955f": SOURCE_955F, "unpinned": SOURCE_UNPINNED}
SOURCE = SOURCE_UNPINNED


_GROUP = 32
_VECTOR_LANES_BF16 = 128
_VECTOR_BLOCK = 2 * _VECTOR_LANES_BF16  # DIST_DINTLV loads two 128-lane vectors
# Tail-axis ComputeMaxExp loop loads two 128-lane BF16 vectors, applies two
# And, one pairwise Max and one ReduceMaxWithDataBlock.  ComputeData loads two
# vectors, two Multiply, one Interleave, two Cast and two packed stores.
_MAX_EXP_OPS_PER_256 = 6
_DATA_OPS_PER_256 = 9
_SCALE_OPS_PER_GROUP = 15


def _clamp(width: int, lane: int) -> int:
    return min(int(width), int(lane))


def _phase_lane_ops(width: int, fp16: bool) -> tuple[int, int]:
    """Masked lane occupancy for one 2x128 source loop.

    Widths are the source register widths: DINTLV loads up to 256 BF16 lanes,
    BF16 arithmetic 128, FP32 casts 64, packed uint8 adds/stores 256, and the
    E8M0 scale store one byte per group.  A tail loop is masked to its true
    width; it is not charged as a second full 256-lane loop.
    """
    if fp16:
        max_exp = (
            _clamp(width, 256) + 2 * _clamp(width, 128)
            + 2 * _clamp(width, 128) + 2 * _clamp(width, 64)
            + 2 * _clamp(width, 128) + 2 * _clamp(width, 128)
            + _clamp(width, 128) + _clamp(width, 128)
            + (width + 31) // 32
        )
        data = (
            2 * _clamp(width, 256)
            + 4 * _clamp(width, 64) + _clamp(width, 64)
            + 4 * _clamp(width, 128)
            + 4 * _clamp(width, 64)
            + 3 * _clamp(width, 256) + _clamp(width, 256)
        )
    else:
        max_exp = (
            _clamp(width, 256) + 2 * _clamp(width, 128)
            + _clamp(width, 128) + _clamp(width, 128)
            + (width + 31) // 32
        )
        data = (
            2 * _clamp(width, 256)
            + 2 * _clamp(width, 128) + 4 * _clamp(width, 64)
            + 4 * _clamp(width, 64)
            + 3 * _clamp(width, 256) + _clamp(width, 256)
        )
    return max_exp, data


def group32_work(width: int, input_dtype: str = "bf16") -> dict[str, int | str | bool]:
    """Register-call and masked-lane ledger for r64 tail-axis FP8, alg0.

    Calls are declared Reg:: instructions.  Lane occupancy uses the source's
    native register widths, not ``calls * 128``.  This is not a hardware issue
    cycle count; the hardware slot service must be applied separately.
    """
    width = int(width)
    if width < 1:
        raise ModelingError("mx_dynamic_quant_requires_positive_width")
    dtype = str(input_dtype or "bf16").lower()
    if dtype not in {"bf16", "bfloat16", "fp16", "float16"}:
        raise ModelingError("mx_dynamic_quant_requires_bf16_or_fp16_input")
    fp16 = dtype in {"fp16", "float16"}
    loops = (width + _VECTOR_BLOCK - 1) // _VECTOR_BLOCK
    groups = (width + _GROUP - 1) // _GROUP
    padded_groups = (groups + 1) // 2 * 2
    scale_loops = (padded_groups + _VECTOR_LANES_BF16 - 1) // _VECTOR_LANES_BF16
    max_calls_per_loop = 14 if fp16 else 6
    data_calls_per_loop = 19 if fp16 else 16
    scale_calls = 15 * scale_loops
    max_lane_ops = 0
    data_lane_ops = 0
    for index in range(loops):
        lane_width = _VECTOR_BLOCK if index < loops - 1 else width - index * _VECTOR_BLOCK
        mx, dx = _phase_lane_ops(lane_width, fp16)
        max_lane_ops += mx
        data_lane_ops += dx
    scale_lane_ops = 15 * min(padded_groups, scale_loops * _VECTOR_LANES_BF16)
    lane_ops = max_lane_ops + data_lane_ops + scale_lane_ops
    return {
        "register_calls": {
            "max_exp": max_calls_per_loop * loops,
            "scale": scale_calls,
            "data": data_calls_per_loop * loops,
            "total": (max_calls_per_loop + data_calls_per_loop) * loops + scale_calls,
        },
        "lane_ops": {"max_exp": max_lane_ops, "scale": scale_lane_ops,
                     "data": data_lane_ops, "total": lane_ops},
        "calls_per_full_loop": {"max_exp": max_calls_per_loop,
                                 "scale": 15, "data": data_calls_per_loop},
        "full_loops": loops - 1 if width % _VECTOR_BLOCK else loops,
        "tail_lanes": width % _VECTOR_BLOCK,
        "groups": groups,
        "padded_groups": padded_groups,
        "scale_loops": scale_loops,
        "lanes_per_full_loop": _VECTOR_BLOCK,
        "input_dtype": "fp16" if fp16 else "bf16",
        "algorithm": "ocp_alg0",
        "work_kind": "register_calls_and_masked_lanes",
        "issue_cycles_known": False,
        "owners": None,
        "owner_basis": "not_derived_from_trace",
        "use_channels": ("vector", "ub_store"),
    }



def producer_contract(width: int, *, input_dtype: str = "bf16",
                      payload_format: str = "fp8_e4m3",
                      scale_alg: int = 0,
                      source_revision: str = "unpinned") -> dict[str, int | str | tuple | None]:
    """Exact materialization and consumer view for one logical row.

    FP8 supports scaleAlg 0/1 only; 955f rejects alg2 for FP8.  The E8M0
    tensor has ceil(ceil(width/32)/2) * 2 lanes, so an odd group count carries
    one extra padded lane.  The payload keeps true width; tail reduction
    padding is not written as payload data.
    """
    width = int(width)
    dtype = str(input_dtype or "bf16").lower()
    alg = int(scale_alg)
    if payload_format != "fp8_e4m3":
        raise ModelingError("mx_dynamic_quant_requires_fp8_e4m3_payload")
    if alg not in {0, 1, 2}:
        raise ModelingError("mx_dynamic_quant_scale_alg_unsupported")
    if alg == 2:
        raise ModelingError("mx_dynamic_quant_scale_alg_2_rejected_for_fp8")
    selected_revision = str(source_revision) if str(source_revision) in SOURCES else "unpinned"
    source = SOURCES[selected_revision]
    work = group32_work(width, dtype) if alg == 0 else None
    payload_bytes = width
    groups = (width + _GROUP - 1) // _GROUP
    padded = (groups + 1) // 2 * 2
    scale_bytes = padded
    key = {0: 33, 1: 65}[alg]
    return {
        "input_format": "fp16" if dtype in {"fp16", "float16"} else "bf16",
        "input_bytes": width * 2,
        "output_format": payload_format,
        "payload_bytes": payload_bytes,
        "scale_format": "fp8_e8m0",
        "scale_bytes": scale_bytes,
        "scale_shape": (padded // 2, 2),
        "write_bytes": payload_bytes + scale_bytes,
        "quant_op": "DynamicMxQuant",
        "vector_ops": (work["lane_ops"]["total"] if work else None),
        "work": work,
        "work_known": work is not None,
        "branch": "tail_axis_block32_fp8",
        "tiling_key": key,
        "kernel": "DynamicMxQuantTailAxisFP8",
        "source": source,
        "source_revision": selected_revision,
        "source_certified": selected_revision in {"r64", "955f"},
    }
