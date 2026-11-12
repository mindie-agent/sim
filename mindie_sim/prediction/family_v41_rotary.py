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

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, Stage


IMPLEMENTATION = "v41_partial_rotary"
NATIVE_KIND = "InplacePartialRotaryMul"
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


def _host_tiling(op, domain, ub_size):
    """Retained Ascend950 RegBase dispatch for the measured V41 shapes.

    Decode: X[1,1,H,D], C[1,1,1,64] -> AAndB BROADCAST/NO_BROADCAST.
    Prefill: X[T,1,H,D] -> AB/SBND for Q/inverse; KV is AAndB NO_BROADCAST.
    The host tiler constants are source and machine geometry, not a case name.
    """
    tokens, heads, head_dim = (int(v) for v in op.local_shape)
    view = dict(dict(op.metadata).get("native_tensor_view") or {})
    declared = dict(dict(op.metadata).get("partial_rotary_work") or {})
    rotary = int(declared.get("rotated_elements", 0) or 0)
    rotary = (rotary // max(1, tokens * heads)) if rotary else int(
        view.get("partial_slice", (0, head_dim))[1] - view.get("partial_slice", (0, head_dim))[0])
    block = 32
    if tokens > 1 and heads > 1:
        b, s_, n = tokens, 1, heads
        cosb = tokens
        bs = b * s_
        if cosb == 1:
            bs = s_
            n = b * n
        block_factor_bs = max(1, -(-bs // domain))
        block_num_bs = max(1, -(-bs // block_factor_bs))
        if bs <= domain // 2:
            block_num_n = max(1, domain // block_num_bs)
            block_factor_n = max(1, -(-n // block_num_n))
            block_num_n = max(1, -(-n // block_factor_n))
        else:
            block_num_n = 1
            block_factor_n = n
        owners = block_num_bs * block_num_n
        d_size_x = ((rotary * 2 + block - 1) // block) * block
        d_size_cos = ((rotary * 4 + block - 1) // block) * block
        base = ((ub_size // 4) // block) * block // d_size_x
        effective = max(1, (d_size_cos + d_size_x - 1) // d_size_x)
        ub_n = max(1, min(block_factor_n, base - effective))
        ub_bs = max(1, min(base // (ub_n + effective), block_factor_bs))
        windows = []
        left = block_factor_bs
        while left > 0:
            part = min(left, ub_bs)
            windows.append(part)
            left -= part
        return {
            "class": "RopeRegBaseTilingClassAB", "layout": "SBND",
            "tokens": tokens, "heads": heads, "head_dim": head_dim,
            "rotary_width": rotary, "merged_b": bs,
            "core_num": domain, "block_factor": block_factor_bs,
            "block_num_bs": block_num_bs, "block_factor_n": block_factor_n,
            "block_num_n": block_num_n, "owners": owners,
            "ub_factor_n": ub_n, "ub_factor_bs": ub_bs,
            "ub_loops": len(windows), "ub_windows": tuple(windows),
            "ub_bytes": ub_size,
            "ub_bytes_owner": 4 * ub_bs * (ub_n * d_size_x + d_size_cos),
            "d_size_x": d_size_x, "d_size_cos": d_size_cos,
            "implemented": True,
        }
    if tokens > 1:
        block_factor = max(1, -(-tokens // domain))
        owners = max(1, -(-tokens // block_factor))
        return {
            "class": "RopeRegBaseTilingClassAAndB", "layout": "NO_BROADCAST",
            "tokens": tokens, "heads": heads, "head_dim": head_dim,
            "rotary_width": rotary, "merged_b": tokens,
            "core_num": domain, "block_factor": block_factor,
            "owners": owners, "ub_factor": block_factor,
            "ub_loops": 1, "ub_windows": (block_factor,),
            "ub_bytes": ub_size, "ub_bytes_owner": 1536 * block_factor,
            "d_size_x": 128, "d_size_cos": 256,
            "implemented": True,
        }
    broadcast = heads > 1
    merged = tokens * heads
    block_factor = max(1, -(-merged // domain))
    owners = max(1, -(-merged // block_factor))
    return {
        "class": "RopeRegBaseTilingClassAAndB",
        "layout": "BROADCAST_BSN" if broadcast else "NO_BROADCAST",
        "tokens": tokens, "heads": heads, "head_dim": head_dim,
        "rotary_width": rotary, "merged_b": merged,
        "core_num": domain, "block_factor": block_factor,
        "owners": owners, "ub_factor": 1, "ub_loops": 1,
        "ub_windows": (1,), "ub_bytes": ub_size,
        "ub_bytes_owner": (1024 if broadcast else 1536) * block_factor,
        "d_size_x": 128, "d_size_cos": 256,
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
    coeff = 2 * tokens * rotary * 4
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
    tensor_bytes = elements * 2
    coefficient_once = 2 * tiling["tokens"] * tiling["rotary_width"] * 4
    total_vector = sum(operations.values())
    if tiling.get("tokens", 0) <= 1:
        coeff_groups = max(1, int(tiling.get("owners", 1)))
    elif tiling.get("heads", 1) > 1:
        coeff_groups = max(1, int(tiling.get("block_num_n", 1)))
    else:
        coeff_groups = 1
    mte2_issue = tensor_bytes + 2 * tiling["tokens"] * coeff_groups * tiling["rotary_width"] * 4
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
        windows = tuple(int(v) for v in tiling.get("ub_windows", (1,))) or (1,)
        loops = len(windows)
        window_total = max(1, sum(windows))
        ub_pool = context.pool_names.get("ub") or next(
            (name for name in context.pool_names if name == "ub"), None)
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
            for index, window in enumerate(windows):
                share = window / window_total
                load = {}
                compute = {}
                store = {}
                for key, amount in physical.items():
                    if not amount:
                        continue
                    value = amount * share / live_owners
                    if key.startswith("vector"):
                        compute[key] = value
                    elif key.startswith("mte3"):
                        store[key] = value
                    elif key.startswith("memory"):
                        # HBM unique input is copy-in; output is copy-out.
                        input_hbm = tensor_bytes + coefficient_once
                        out_hbm = tensor_bytes
                        scale = amount / max(1, input_hbm + out_hbm)
                        load[key] = (input_hbm * scale) * share / live_owners
                        store[key] = (out_hbm * scale) * share / live_owners
                    else:
                        load[key] = value
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


def declare(builder, *, name, deps, tokens, heads, head_dim, rotary_width, inverse=False):
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
    # BROADCAST_BSN: each source owner performs its private CopyInCosAndSin;
    # NO_BROADCAST (KV) has one owner. This is GM issue bytes, not DRAM misses.
    if tokens > 1 and heads > 1:
        # AB N-group owners each reload the per-token coefficient row.
        factor = max(1, -(-tokens // 64))
        groups = max(1, -(-tokens // factor))
        if tokens <= 32:
            n_blocks = max(1, 64 // groups)
            n_factor = max(1, -(-heads // n_blocks))
            groups = max(1, -(-heads // n_factor))
        else:
            groups = 1
    elif tokens <= 1:
        groups = heads
    else:
        groups = 1
    coefficient_bytes = (2 * tokens * groups * rotary_width
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
            "ub_bytes_per_owner": (1024 if heads > 1 else 1536),
            "ub_queue_count": (4 if heads > 1 else 4),
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
