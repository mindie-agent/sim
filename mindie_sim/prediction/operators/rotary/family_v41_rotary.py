"""Explicit partial-RoPE native boundaries for shared-KV attention callers.

[Role] Declare independent Q, KV and post-attention inverse rotations using the
  existing graph builder and resource engine.
[Invariants] Only the rotary slice is read/written in place; NoPE is untouched.
  Each declaration is one invocation. A Q/KV aggregate must be replaced, never
  retained alongside these nodes. Negative sine is supplied by the caller.
  Coefficient sharing across a token's heads is an explicit analytic assumption,
  not proof of native tiling, cache hits or captured library identity.
[Timeouts & Locks] Constant arithmetic; no I/O, locks or device contexts.
[Side-Effect Policy] Appends one immutable operator to the caller's builder.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage


TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
IMPLEMENTATION = "v41_partial_rotary"
NATIVE_KIND = "InplacePartialRotaryMul"
VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_REVISION,
        relative_path="csrc/attention/inplace_partial_rotary_mul/op_host/rope_regbase_tiling_a_and_b.cpp",
        source_sha256="c460ce395d075ef4a95a7d16ed855b601cdd77c1bbeefde5d9a135e3b749191a",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_REVISION,
        relative_path="csrc/attention/inplace_partial_rotary_mul/op_kernel/rotary_position_embedding_reg_a_and_b_mixed.h",
        source_sha256="4af72e46cfdfc1be5cf4a55f4e8d10a8725ef3e38966b67b9fb9d27e9fa883ef",
    ),
)
_BF16_ELEMENT_BYTES = 2
_FP32_COEFFICIENT_BYTES = 4
_COS_SIN_TENSORS = 2
_UB_ALIGNMENT_BYTES = 32
_SBND_UB_BUFFER_FACTOR = 4
_A_AND_B_X_BUFFER_FACTOR = 4
_A_AND_B_COS_SIN_BUFFER_FACTOR = 2
_A_AND_B_MAX_COPY_BLOCK_COUNT = 4095
_A_AND_B_MAX_UB_FACTOR = _A_AND_B_MAX_COPY_BLOCK_COUNT // 2
_A_AND_B_BROADCAST_UB_BYTES = 1024
_A_AND_B_NO_BROADCAST_UB_BYTES = 1536
_A_AND_B_UB_QUEUE_COUNT = 4
SOURCE = {
    "caller": {
        "repository": "vllm-ascend",
        "path": "vllm_ascend/attention/dsa_v41.py",
        "sha256": "56d3ce83098babf585289d3e078d1efff0442547f61fb0955a11f0b42bfc6ba3",
        "forward_q_kv_lines": [283, 305],
        "inverse_output_lines": [596, 611],
    },
    "slice_kernel": {
        "repository": "cann-recipes-infer",
        "path": "ops/ascendc/src/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_split_bsn_mixed.h",
        "sha256": "303cbd132566519a9cede7cc13156c62ea7462be26d71222425638bf0997b457",
        "copy_lines": [145, 216], "compute_lines": [284, 316],
    },
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def recipes():
    return {name: PlannedRecipe(_build) for name in
            ("q_rotary", "kv_rotary", "output_inverse_rotary")}


def _partition_windows(total: int, ub_factor: int) -> tuple[int, ...]:
    windows = []
    left = total
    while left > 0:
        part = min(left, ub_factor)
        windows.append(part)
        left -= part
    return tuple(windows) or (1,)


def _owner_partition(tokens: int, heads: int, domain: int) -> dict[str, int]:
    """Single owner-partition helper shared by declaration and execution."""
    eff_domain = max(1, int(domain))
    if tokens > 1 and heads > 1:
        block_factor_bs = max(1, -(-tokens // eff_domain))
        block_num_bs = max(1, -(-tokens // block_factor_bs))
        if tokens <= eff_domain // 2:
            block_num_n = max(1, eff_domain // block_num_bs)
            block_factor_n = max(1, -(-heads // block_num_n))
            block_num_n = max(1, -(-heads // block_factor_n))
        else:
            block_num_n = 1
            block_factor_n = heads
        return {
            "block_factor": block_factor_bs,
            "block_num_bs": block_num_bs,
            "block_factor_n": block_factor_n,
            "block_num_n": block_num_n,
            "owners": block_num_bs * block_num_n,
            "coeff_groups": block_num_n,
        }
    if tokens > 1:
        block_factor = max(1, -(-tokens // eff_domain))
        owners = max(1, -(-tokens // block_factor))
        return {
            "block_factor": block_factor,
            "block_num_bs": owners,
            "block_factor_n": heads,
            "block_num_n": 1,
            "owners": owners,
            "coeff_groups": 1,
        }
    merged = tokens * heads
    block_factor = max(1, -(-merged // eff_domain))
    owners = max(1, -(-merged // block_factor))
    return {
        "block_factor": block_factor,
        "block_num_bs": 1,
        "block_factor_n": block_factor,
        "block_num_n": owners,
        "owners": owners,
        "coeff_groups": max(1, owners),
    }


def _rotary_width(op) -> tuple[int, int, int, int]:
    tokens, heads, head_dim = (int(v) for v in op.local_shape)
    view = dict(dict(op.metadata).get("native_tensor_view") or {})
    declared = dict(dict(op.metadata).get("partial_rotary_work") or {})
    rotary = int(declared.get("rotated_elements", 0) or 0)
    rotary = (rotary // max(1, tokens * heads)) if rotary else int(
        view.get("partial_slice", (0, head_dim))[1] - view.get("partial_slice", (0, head_dim))[0])
    return tokens, heads, head_dim, rotary


def issued_mte2_bytes(op, *, domain: int) -> int:
    """Compute physical GM->UB MTE2 issue bytes for a target AIV core count."""
    tokens, heads, _head_dim, rotary = _rotary_width(op)
    part = _owner_partition(tokens, heads, domain)
    tensor_bytes = tokens * heads * rotary * _BF16_ELEMENT_BYTES
    coeff_bytes = (_COS_SIN_TENSORS * tokens * part["coeff_groups"] * rotary
                   * _FP32_COEFFICIENT_BYTES)
    return int(tensor_bytes + coeff_bytes)


def _host_tiling(op, domain, ub_size):
    """Retained Ascend950 RegBase dispatch for the measured V41 shapes.

    Decode: X[1,1,H,D], C[1,1,1,64] -> AAndB BROADCAST/NO_BROADCAST.
    Prefill: X[T,1,H,D] -> AB/SBND for Q/inverse; KV is AAndB NO_BROADCAST.
    The host tiler constants are source and machine geometry, not a case name.
    """
    tokens, heads, head_dim, rotary = _rotary_width(op)
    block = _UB_ALIGNMENT_BYTES
    d_size_x = ((rotary * _BF16_ELEMENT_BYTES + block - 1) // block) * block
    d_size_cos = ((rotary * _FP32_COEFFICIENT_BYTES + block - 1) // block) * block
    part = _owner_partition(tokens, heads, domain)
    if tokens > 1 and heads > 1:
        block_factor_bs = part["block_factor"]
        block_num_bs = part["block_num_bs"]
        block_factor_n = part["block_factor_n"]
        block_num_n = part["block_num_n"]
        owners = part["owners"]
        base = ((ub_size // _SBND_UB_BUFFER_FACTOR) // block) * block // d_size_x
        effective = max(1, (d_size_cos + d_size_x - 1) // d_size_x)
        if base < 1 + effective:
            raise ModelingError(f"partial_rotary_ub_insufficient:{rotary}:{ub_size}")
        ub_n = max(1, min(block_factor_n, base - effective))
        ub_bs = max(1, min(base // (ub_n + effective), block_factor_bs))
        bs_windows = _partition_windows(block_factor_bs, ub_bs)
        n_windows = _partition_windows(block_factor_n, ub_n)
        windows = tuple(b_part * n_part for b_part in bs_windows for n_part in n_windows)
        return {
            "class": "RopeRegBaseTilingClassAB", "layout": "SBND",
            "tokens": tokens, "heads": heads, "head_dim": head_dim,
            "rotary_width": rotary, "merged_b": tokens,
            "core_num": domain, "block_factor": block_factor_bs,
            "block_num_bs": block_num_bs, "block_factor_n": block_factor_n,
            "block_num_n": block_num_n, "owners": owners,
            "coeff_groups": part["coeff_groups"],
            "ub_factor_n": ub_n, "ub_factor_bs": ub_bs,
            "ub_windows_bs": bs_windows, "ub_windows_n": n_windows,
            "ub_loops": len(windows), "ub_windows": windows,
            "ub_bytes": ub_size,
            "ub_bytes_owner": _SBND_UB_BUFFER_FACTOR * ub_bs * (ub_n * d_size_x + d_size_cos),
            "d_size_x": d_size_x, "d_size_cos": d_size_cos,
            "implemented": True,
        }
    if tokens > 1:
        block_factor = part["block_factor"]
        owners = part["owners"]
        unit_ub = _A_AND_B_X_BUFFER_FACTOR * (d_size_x + d_size_cos)
        num_avail = ub_size // unit_ub
        if num_avail < 1:
            raise ModelingError(f"partial_rotary_ub_insufficient:{rotary}:{ub_size}")
        ub_factor = min(block_factor, num_avail, _A_AND_B_MAX_UB_FACTOR)
        windows = _partition_windows(block_factor, ub_factor)
        return {
            "class": "RopeRegBaseTilingClassAAndB", "layout": "NO_BROADCAST",
            "tokens": tokens, "heads": heads, "head_dim": head_dim,
            "rotary_width": rotary, "merged_b": tokens,
            "core_num": domain, "block_factor": block_factor,
            "owners": owners, "coeff_groups": part["coeff_groups"],
            "ub_factor": ub_factor,
            "ub_loops": len(windows), "ub_windows": windows,
            "ub_bytes": ub_size, "ub_bytes_owner": unit_ub * ub_factor,
            "d_size_x": d_size_x, "d_size_cos": d_size_cos,
            "implemented": True,
        }
    broadcast = heads > 1
    merged = tokens * heads
    block_factor = part["block_factor"]
    owners = part["owners"]
    if broadcast:
        cos_ub = _A_AND_B_COS_SIN_BUFFER_FACTOR * d_size_cos
        x_unit_ub = _A_AND_B_X_BUFFER_FACTOR * d_size_x
        avail = ub_size - cos_ub
        if avail < x_unit_ub:
            raise ModelingError(f"partial_rotary_ub_insufficient:{rotary}:{ub_size}")
        ub_factor = min(block_factor, avail // x_unit_ub, _A_AND_B_MAX_UB_FACTOR)
        ub_bytes_owner = x_unit_ub * ub_factor + cos_ub
    else:
        unit_ub = _A_AND_B_X_BUFFER_FACTOR * (d_size_x + d_size_cos)
        num_avail = ub_size // unit_ub
        if num_avail < 1:
            raise ModelingError(f"partial_rotary_ub_insufficient:{rotary}:{ub_size}")
        ub_factor = min(block_factor, num_avail, _A_AND_B_MAX_UB_FACTOR)
        ub_bytes_owner = unit_ub * ub_factor
    windows = _partition_windows(block_factor, ub_factor)
    return {
        "class": "RopeRegBaseTilingClassAAndB",
        "layout": "BROADCAST_BSN" if broadcast else "NO_BROADCAST",
        "tokens": tokens, "heads": heads, "head_dim": head_dim,
        "rotary_width": rotary, "merged_b": merged,
        "core_num": domain, "block_factor": block_factor,
        "owners": owners, "coeff_groups": part["coeff_groups"],
        "ub_factor": ub_factor, "ub_loops": len(windows),
        "ub_windows": windows, "ub_bytes": ub_size,
        "ub_bytes_owner": ub_bytes_owner,
        "d_size_x": d_size_x, "d_size_cos": d_size_cos,
        "implemented": True,
    }


def _cycle_work(op):
    """Native FP32-coefficient slice stage counts per source ComputeCastFp32."""
    tokens, heads, head_dim = (int(v) for v in op.local_shape)
    view = dict(dict(op.metadata).get("native_tensor_view") or {})
    declared = dict(dict(op.metadata).get("partial_rotary_work") or {})
    rotary = int(declared.get("rotated_elements", 0) or 0)
    rotary = (rotary // max(1, tokens * heads)) if rotary else int(
        view.get("partial_slice", (0, head_dim))[1] - view.get("partial_slice", (0, head_dim))[0])
    elements = tokens * heads * rotary
    coeff = 2 * tokens * rotary * _FP32_COEFFICIENT_BYTES
    operations = {
        "cast_x": elements,
        "multiply": 2 * elements, "gather": elements,
        "interleaved_inversion": elements, "add": elements,
        "cast_out": elements,
        "coefficient_broadcast": 2 * tokens * max(0, heads - 1) * rotary,
        "gather_offset_setup": tokens * heads,
    }
    return elements, coeff, operations


def _build(op, geometry):
    domain = max(1, int(dict(geometry.resource_units).get("vector", 1)))
    if "ub" not in dict(geometry.buffers):
        raise ModelingError("partial_rotary_requires_actual_ub_geometry")
    ub_size = int(dict(geometry.buffers)["ub"])
    tiling = _host_tiling(op, domain, ub_size)
    elements, coeff_bytes, operations = _cycle_work(op)
    tensor_bytes = elements * _BF16_ELEMENT_BYTES
    coefficient_once = (_COS_SIN_TENSORS * tiling["tokens"] * tiling["rotary_width"]
                        * _FP32_COEFFICIENT_BYTES)
    total_vector = sum(operations.values())
    coeff_groups = max(1, int(tiling.get("coeff_groups", 1)))
    mte2_issue = (tensor_bytes + _COS_SIN_TENSORS * tiling["tokens"] * coeff_groups
                  * tiling["rotary_width"] * _FP32_COEFFICIENT_BYTES)
    owners = max(1, int(tiling["owners"]))
    windows = tuple(int(v) for v in tiling.get("ub_windows", (tiling.get("ub_factor", 1),))) or (1,)
    loops = len(windows)
    if not tiling.get("implemented", True):
        def analytic_lower(context):
            return KernelProgram((Stage("partial_slice_analytic", context.work),))
        return OperatorPlan(Work(resources={
            "hbm": op.hbm_bytes, "mte2_aiv": op.mte2_bytes,
            "mte3": op.mte3_bytes, "vector": total_vector,
        }), analytic_lower, {
            "execution_abstraction": "analytic_partial_rotary_fallback",
            "execution_class": "vector",
            "owner_schedule": "unimplemented_regbase_ab_tiling",
            "host_tiler": tiling,
            "issued_mte2_aiv_bytes": mte2_issue,
            "issued_mte3_bytes": op.mte3_bytes,
            "hbm_unique_bytes": op.hbm_bytes,
            "native_BlockDim_not_selected": True,
        })
    def lower(context):
        physical = dict(context.work)
        roles = ("vector", "mte2_aiv", "mte3")
        live_owners = owners
        bs_windows = tuple(int(v) for v in tiling.get("ub_windows_bs", ()))
        n_windows = tuple(int(v) for v in tiling.get("ub_windows_n", ()))
        sbnd_loop = bool(bs_windows and n_windows)
        if sbnd_loop:
            bs_total = max(1, sum(bs_windows))
            n_total = max(1, sum(n_windows))
            iter_specs = [
                (b_part / bs_total, n_part / n_total, n_idx == 0)
                for b_part in bs_windows
                for n_idx, n_part in enumerate(n_windows)
            ]
        else:
            windows = tuple(int(v) for v in tiling.get("ub_windows", (1,))) or (1,)
            window_total = max(1, sum(windows))
            iter_specs = [(window / window_total, 1.0, True) for window in windows]
        loops = len(iter_specs)
        ub_pool = context.pool_names.get("ub") or next(
            (name for name in context.pool_names if name == "ub"), None)
        coeff_mte2_total = (
            _COS_SIN_TENSORS * tiling["tokens"] * coeff_groups * tiling["rotary_width"]
            * _FP32_COEFFICIENT_BYTES
        )
        programs = []
        for owner in range(live_owners):
            placement = context.owned(owner, domain, roles)
            acquire = free = ()
            if ub_pool:
                ub_bytes = int(tiling.get("ub_bytes_owner", 1024))
                alloc = context.allocate(f"ub:{owner}", "ub", ub_bytes, (owner,))
                acquire = (alloc,)
                free = (alloc,)
            stages = []
            previous = ()
            for index, (bs_share, n_share, load_coeff) in enumerate(iter_specs):
                share = bs_share * n_share
                load = {}
                compute = {}
                store = {}
                for key, amount in physical.items():
                    if not amount:
                        continue
                    if key.startswith("vector"):
                        compute[key] = amount * share / live_owners
                    elif key.startswith("mte3"):
                        store[key] = amount * share / live_owners
                    elif key.startswith("memory"):
                        input_hbm = tensor_bytes + coefficient_once
                        out_hbm = tensor_bytes
                        scale = amount / max(1, input_hbm + out_hbm)
                        coeff_hbm = (coefficient_once * scale * bs_share / live_owners) if load_coeff else 0.0
                        load[key] = (tensor_bytes * scale * share / live_owners) + coeff_hbm
                        store[key] = (out_hbm * scale) * share / live_owners
                    elif sbnd_loop and key.startswith("mte2"):
                        scale = amount / max(1, tensor_bytes + coeff_mte2_total)
                        coeff_mte2 = (coeff_mte2_total * scale * bs_share / live_owners) if load_coeff else 0.0
                        load[key] = (tensor_bytes * scale * share / live_owners) + coeff_mte2
                    else:
                        load[key] = amount * share / live_owners
                vkey = next((key for key in physical if key.startswith("vector")), "")
                if vkey and vkey not in compute:
                    compute[vkey] = total_vector * share / live_owners
                load_name = f"ub_window_{index}_copy_in"
                compute_name = f"ub_window_{index}_compute"
                store_name = f"ub_window_{index}_copy_out"
                stages.append(Stage(load_name, load, after=previous,
                                    placement=placement,
                                    acquire=acquire if index == 0 else ()))
                stages.append(Stage(compute_name, compute, after=(load_name,),
                                    placement=placement))
                stages.append(Stage(store_name, store,
                                    after=(compute_name,), placement=placement,
                                    free=free if index == loops - 1 else ()))
                previous = (store_name,)
            programs.append(KernelProgram(tuple(stages)))
        return ParallelProgram(tuple(programs))

    return OperatorPlan(Work(resources={
        "hbm": op.hbm_bytes, "mte2_aiv": mte2_issue,
        "mte3": op.mte3_bytes, "vector": total_vector,
    }), lower, {
        "execution_abstraction": "regbase_aandb_owner_ub_windows",
        "execution_class": "vector",
        "owner_schedule": "regbase_blockNumB_owners",
        "host_tiler": tiling,
        "issued_mte2_aiv_bytes": mte2_issue,
        "issued_mte3_bytes": op.mte3_bytes,
        "hbm_unique_bytes": op.hbm_bytes,
        "cycle_owner": {
            "fp32_coefficient_path": True,
            "cast_x": 1,
            "multiply": 2, "gather": 1, "interleaved_inversion": 1,
            "add": 1, "cast_out": 1,
            "vector_lane_operations": operations,
        },
        "native_BlockDim_not_selected": True,
    })


def validate(plan, *, fused_prolog):
    if (plan.mla_variant != "latent_mqa" or plan.cache_kind != "window_compressed"
            or not plan.qk_rope_head_dim or fused_prolog):
        raise ModelingError("partial_rotary_caller_requires_unfused_shared_kv_mqa")


def declare(builder, *, name, deps, tokens, heads, head_dim, rotary_width,
            inverse=False, vector_cores: int | None = None):
    """Append one BF16/FP32-coefficient in-place partial interleaved rotation.

    The caller selects this mechanism explicitly and supplies the full active
    tensor view. For Q/KV use independent dependencies and names ``q_rotary`` /
    ``kv_rotary``; output ``output_inverse_rotary`` depends on attention results.
    The shared vector resource counts semantic lanes, not ISA cycles: two
    multiplies, one add, two casts, one permutation and half-lane sign change.
    UB coefficient broadcast lanes are retained separately from HBM traffic.
    """
    dtype = builder.model.activation_dtype
    if (dtype != "bf16" or tokens < 0 or heads < 1 or head_dim < 1
            or rotary_width < 2 or rotary_width > head_dim or rotary_width % 2):
        raise ModelingError("partial_rotary_requires_bf16_even_slice_inside_head")
    elements = tokens * heads * rotary_width
    tensor_bytes = elements * dtype_bytes(dtype)
    # Without a bound target core count, declare the hardware-independent
    # compulsory minimum (1 coefficient load per token) so OpSpec.mte2_bytes
    # remains a sound lower bound across all AIV core counts; when target
    # vector_cores is given, derive exact groups via _owner_partition.
    groups = (
        _owner_partition(tokens, heads, vector_cores)["coeff_groups"]
        if vector_cores is not None
        else 1
    )
    coefficient_bytes = (_COS_SIN_TENSORS * tokens * groups * rotary_width
                         * dtype_bytes("fp32"))
    operations = {
        "multiply": 2 * elements, "add": elements,
        "cast": 2 * elements, "interleave_permutation": elements,
        "odd_lane_sign_change": elements // 2,
        "coefficient_broadcast": 2 * tokens * max(0, heads - 1) * rotary_width,
    }
    view = {
        "kind": "inplace_partial_rotary", "native_kind": NATIVE_KIND,
        "shape": (tokens, 1, heads, head_dim),
        "coefficient_shape": (tokens, 1, 1, rotary_width),
        "dtype": dtype, "coefficient_dtype": "fp32", "layout": "ND",
        "partial_slice": (head_dim - rotary_width, head_dim),
        "rotary_mode": "interleave", "inverse": bool(inverse),
        "sine_sign": "caller_negated" if inverse else "caller_forward",
    }
    metadata = (
        ("native_tensor_view", view), ("native_source", SOURCE),
        ("partial_rotary_work", {
            "rotated_elements": elements, "untouched_nope_elements": tokens * heads * (head_dim - rotary_width),
            "input_slice_bytes": tensor_bytes, "output_slice_bytes": tensor_bytes,
            "coefficient_bytes": coefficient_bytes, "vector_lane_operations": operations,
            "coefficient_load_scope": "once_per_token_shared_across_heads",
            "coefficient_sharing_is_analytic_assumption": False,
            "native_owner_split_and_reload_not_selected": False,
            "ub_bytes_per_owner": (_A_AND_B_BROADCAST_UB_BYTES if heads > 1
                                   else _A_AND_B_NO_BROADCAST_UB_BYTES),
            "ub_queue_count": _A_AND_B_UB_QUEUE_COUNT,
            "cos_sin_double_buffer": (1 if heads > 1 else 2),
            "q_in_out_double_buffer": 2,
            "ub_permutation_and_broadcast_are_vector_lane_work_not_hbm": True,
            "caller_negated_sine_preparation_outside_this_invocation": bool(inverse),
        }),
        ("implementation_assumption", "source_partial_slice_analytic_resource_work"),
        ("native_implementation", {
            "native_kind": NATIVE_KIND, "selection": IMPLEMENTATION,
            "conditions": {"scope": "one_inplace_partial_rotation",
                           "native_launch_count": 1, "capture_scalar_flags_proven": False},
        }),
    )
    result = builder.work(
        name, deps, (tokens, heads, head_dim), family="rotary", dtype=dtype,
        vector=sum(operations.values()), read=tensor_bytes + coefficient_bytes,
        write=tensor_bytes, metadata=metadata,
    )
    # Preserve the caller's invocation across generic fusion passes. This
    # boundary declares one kernel; its timing remains analytic resource work.
    # GM issue (MTE2) includes each owner's private cos/sin copy; HBM is the
    # unique tensor traffic plus one coefficient fetch, not a DRAM miss per
    # repeated broadcast owner.
    coefficient_once = 2 * tokens * rotary_width * dtype_bytes("fp32")
    builder.ops[-1] = replace(
        builder.ops[-1], source_boundary="kernel",
        hbm_bytes=2 * tensor_bytes + coefficient_once,
        mte2_bytes=tensor_bytes + coefficient_bytes,
        mte3_bytes=tensor_bytes,
    )
    return result
