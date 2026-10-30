"""CUDA MoE alignment input geometry and source dispatch.

[Role] Normalize the source caller into geometry and work counters.
[Invariants] Integer histograms, sentinel padding, source dispatch and expert IDs remain exact.
  No durations, hardware-rate fits or runtime source reads enter normalization.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Immutable declarations; no device or external I/O.
"""
from __future__ import annotations

from typing import Any, Mapping

_IMPLEMENTATION = "cuda_moe_align_block_size"
_WRAPPER_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/vllm/model_executor/layers/"
    "fused_moe/moe_align_block_size.py"
)
_CUDA_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/csrc/libtorch_stable/moe/"
    "moe_align_sum_kernels.cu"
)
_MATH_PATH = "/Users/maoxx241/code/vllm-v0.25.1/vllm/utils/math_utils.py"

_WARP_SIZE = 32
_SMALL_NUMEL_LIMIT = 1024
_SMALL_EXPERT_LIMIT = 64
_FILL_THREADS = 256
_SORT_THREADS = 256
_SORT_BLOCK_LIMIT = 65535
_ALIGN_THREADS = 1024
_INT32_BYTES = 4
_PIPE_ROLES = ("vector", "mte2", "mte3")
_SHARED_POOL = "shared"

#: Source dispatch accepts integral tensors; the routing values themselves are
#: interpreted as ``int expert_id`` by the kernel.
_INTEGRAL_BYTES = {
    "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
}
_DEFAULT_TOPK_DTYPE = "int32"

_PHASE_IDS = {
    "small_batch_expert": "align",
    "large_align": "align",
    "large_sort": "sort",
}

_SOURCE_IDENTITY = {
    "scope": "current_checkout_manifest_matched_not_historical_capture_image",
    "files": (
        {
            "path": _WRAPPER_PATH,
            "role": "python_wrapper",
            "sha256": "c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b",
            "bytes": 8339,
            "selected_lines": "74-103",
        },
        {
            "path": _CUDA_PATH,
            "role": "cuda_align_and_sort_kernels",
            "sha256": "e2bdb00824895cf7f43569159910ca369db3d7ad1da08c51cff2325d03d2e68b",
            "bytes": 40399,
            "selected_lines": (
                "85-185,187-293,295-324,326-351,453-466,582-680"),
        },
    ),
    "helper": {
        "path": _MATH_PATH,
        "role": "round_up_helper_outside_manifest",
        "sha256": "23d9d5af62ee43dc55fd647822265f5e20b813bf89e5510130065932c7868fa6",
        "bytes": 904,
        "selected_lines": "20-22",
    },
}

_SOURCE_LINES = {
    "wrapper_buffer_sizing": "moe_align_block_size.py:74-88",
    "wrapper_device_call": "moe_align_block_size.py:90-98",
    "wrapper_expert_map_gather": "moe_align_block_size.py:100-103",
    "host_dispatch_and_launch": "moe_align_sum_kernels.cu:582-680",
    "small_batch_threshold": "moe_align_sum_kernels.cu:613-614",
    "small_batch_launch": "moe_align_sum_kernels.cu:616-637",
    "small_batch_kernel_entry": "moe_align_sum_kernels.cu:453-466",
    "small_batch_device_body": "moe_align_sum_kernels.cu:187-293",
    "small_batch_fill_barriers": "moe_align_sum_kernels.cu:205-216",
    "small_batch_first_worker_barrier": "moe_align_sum_kernels.cu:240",
    "small_batch_expert_prefix": "moe_align_sum_kernels.cu:266-270",
    "small_batch_residual_fill": "moe_align_sum_kernels.cu:272-276",
    "small_batch_scatter": "moe_align_sum_kernels.cu:278-292",
    "two_kernel_launch": "moe_align_sum_kernels.cu:638-678",
    "align_device_body": "moe_align_sum_kernels.cu:85-185",
    "align_init_cta": "moe_align_sum_kernels.cu:103-112",
    "align_expert_ids": "moe_align_sum_kernels.cu:172-177",
    "align_residual_fill": "moe_align_sum_kernels.cu:179-185",
    "sort_device_body": "moe_align_sum_kernels.cu:295-324",
    "align_kernel": "moe_align_sum_kernels.cu:326-340",
    "sort_kernel": "moe_align_sum_kernels.cu:342-351",
}

_UNRESOLVED = (
    "cub::BlockScan<int32_t,1024>::ExclusiveSum shuffle, instruction and "
    "synchronization cost is not declared by the supplied sources; the scan stage "
    "declares only its cumsum traffic and no scan-library instruction rate is fabricated",
    "shared-memory atomics, bank conflicts, barriers and general integer instruction "
    "issue are not separately priced",
    "the runtime integer width of topk_ids is not fixed by the supplied sources; int32 "
    "is the default and an alternate declared width changes read bytes",
    "the large align kernel skips topk ids at or above num_experts, but the small "
    "histogram/scatter path has no such guard; the planner requires every id in "
    "[0,num_experts) for both launch branches rather than model an out-of-bounds read",
    "expert_map local ids are required to be -1 or in [0,num_experts); the source "
    "does not validate a map that would index outside the per-expert counters",
    "round_up() is imported from vllm/utils/math_utils.py, outside the manifest; the "
    "current checkout uses ((x + y - 1) // y) * y for positive integers",
    "the engine has no global-workspace capacity pool; tensor allocation sizes are "
    "reported but only traffic is charged and no on-chip Allocation is claimed",
    "dynamic shared memory is modelled as an Allocation only when the hardware "
    "declares a 'shared' buffer pool; otherwise its size is reported without a "
    "capacity check",
    "the sort grid is assigned one CTA per physical unit in waves; this occupancy-1 "
    "assumption is structural and is not a measured occupancy or efficiency value",
    "mte2/mte3 copy units are per-owner resources while HBM remains globally shared; "
    "on unified-memory hardware the copy roles are absent and only HBM is charged",
    "no historical moe_align capture row, dispatch or timing was supplied; no capture "
    "match or accuracy claim is made",
)

_SELECTION_LIMITATIONS = (
    "the wrapper post-gather expert_map[expert_ids] path (expert_map present with "
    "ignore_invalid_experts=False, wrapper lines 100-103) is not supported: it is a "
    "PyTorch/ATen advanced-indexing CUDA kernel whose launch, grid, CTA layout and "
    "copy/vector ownership are outside the supplied native source.  make_op rejects "
    "that input with requires_native_layout instead of leaving an unowned global "
    "stage; the native filtered path (ignore_invalid_experts=True, CUDA lines "
    "133-137/231-235/311-315) is supported",
)


def _source_identity() -> dict[str, Any]:
    """Return a per-op copy so metadata never aliases mutable module globals."""
    return {
        **{key: value for key, value in _SOURCE_IDENTITY.items()
           if key not in {"files", "helper"}},
        "files": tuple(dict(item) for item in _SOURCE_IDENTITY["files"]),
        "helper": dict(_SOURCE_IDENTITY["helper"]),
    }


def _require_int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{reason}:{value!r}")
    if value < minimum:
        raise ValueError(f"{reason}:{value!r}")
    return value


def _require_bool(value: Any, reason: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{reason}:{value!r}")
    return value


def _require_sequence(value: Any, reason: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{reason}:{value!r}")
    return tuple(value)


def _select_implementation(params: Mapping[str, Any]) -> str:
    implementation = params.get("implementation", _IMPLEMENTATION)
    if implementation != _IMPLEMENTATION:
        raise ValueError(f"unsupported_moe_align_implementation:{implementation!r}")
    return implementation


def _parse_expert_map(params: Mapping[str, Any], num_experts: int) -> tuple[int, ...] | None:
    raw = params.get("expert_map")
    if raw is None:
        return None
    values = _require_sequence(raw, "expert_map_must_be_sequence_or_none")
    if len(values) != num_experts:
        raise ValueError(
            f"expert_map_length_must_match_num_experts:{len(values)}!={num_experts}")
    result = []
    for value in values:
        local = _require_int(value, "expert_map_values_must_be_ints_in_[-1,num_experts)",
                             minimum=-1)
        if local >= num_experts:
            raise ValueError(
                f"expert_map_local_id_must_be_below_num_experts:{local}>={num_experts}")
        result.append(local)
    return tuple(result)


def _round_up(value: int, alignment: int) -> int:
    """Source helper: ((x + y - 1) // y) * y for positive y."""
    return (value + alignment - 1) // alignment * alignment


def _analyze(params: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact wrapper sizes, per-expert histogram and launch geometry."""
    if not isinstance(params, Mapping):
        raise ValueError("moe_align_params_must_be_mapping")
    _select_implementation(params)
    pad_sorted_ids = _require_bool(
        params.get("pad_sorted_ids", False), "pad_sorted_ids_must_be_bool")
    ignore_invalid_experts = _require_bool(
        params.get("ignore_invalid_experts", False),
        "ignore_invalid_experts_must_be_bool")
    topk_ids_dtype = params.get("topk_ids_dtype", _DEFAULT_TOPK_DTYPE)
    if topk_ids_dtype not in _INTEGRAL_BYTES:
        raise ValueError(f"unsupported_topk_ids_dtype:{topk_ids_dtype!r}")
    topk_element_bytes = _INTEGRAL_BYTES[topk_ids_dtype]

    num_experts = _require_int(
        params.get("num_experts"), "num_experts_must_be_positive_int", minimum=1)
    block_size = _require_int(
        params.get("block_size"), "block_size_must_be_positive_int", minimum=1)
    padded_num_experts = ((num_experts + _WARP_SIZE - 1) // _WARP_SIZE) * _WARP_SIZE
    # Host line 599 rejects this before either launch branch is selected.
    if padded_num_experts >= 1024:
        raise ValueError(
            "moe_align_padded_num_experts_must_be_below_1024:"
            f"{padded_num_experts}>=1024")

    rows = _require_sequence(params.get("topk_ids"), "topk_ids_must_be_2d_sequence")
    if not rows:
        raise ValueError("topk_ids_must_contain_at_least_one_token")
    top_k = len(_require_sequence(rows[0], "topk_ids_row_must_be_sequence"))
    if top_k < 1:
        raise ValueError("topk_ids_top_k_must_be_positive")
    expert_map = _parse_expert_map(params, num_experts)
    if expert_map is not None and not ignore_invalid_experts:
        # Wrapper lines 100-103 run ``expert_ids = expert_map[expert_ids]`` as
        # a separate ATen advanced-indexing CUDA kernel.  That launch, grid,
        # CTA layout and copy/vector ownership are outside the supplied native
        # source, so an executable plan would be an unowned fallback.
        raise ValueError(
            "moe_align_post_gather_not_supported:requires_native_layout:"
            "expert_map_with_ignore_invalid_experts_false")
    has_expert_map_kernel_arg = expert_map is not None
    expert_ids_post_gather = False

    tokens = len(rows)
    numel = tokens * top_k
    small_batch_expert_mode = (
        numel < _SMALL_NUMEL_LIMIT and num_experts <= _SMALL_EXPERT_LIMIT)
    sort_ctas = 0 if small_batch_expert_mode else min(
        (numel + _SORT_THREADS - 1) // _SORT_THREADS, _SORT_BLOCK_LIMIT)
    sort_cta_ids = [0] * sort_ctas
    sort_cta_valid = [0] * sort_ctas

    counts = [0] * num_experts
    valid_entries = 0
    for token, row in enumerate(rows):
        row = _require_sequence(row, f"topk_ids_row_{token}_must_be_sequence")
        if len(row) != top_k:
            raise ValueError(
                f"topk_ids_rows_must_have_constant_top_k:{token}:{len(row)}!={top_k}")
        for column, value in enumerate(row):
            expert_id = _require_int(
                value, f"topk_ids_value_must_be_nonnegative_int_at_token_{token}")
            if expert_id >= num_experts:
                raise ValueError(
                    f"topk_ids_value_must_be_below_num_experts_at_token_{token}:"
                    f"{expert_id}>={num_experts}")
            flat_index = token * top_k + column
            if sort_ctas:
                sort_cta_ids[(flat_index // _SORT_THREADS) % sort_ctas] += 1
            if has_expert_map_kernel_arg:
                local = expert_map[expert_id]  # type: ignore[index]
                if local == -1:
                    continue
                counts[local] += 1
            else:
                counts[expert_id] += 1
            valid_entries += 1
            if sort_ctas:
                sort_cta_valid[(flat_index // _SORT_THREADS) % sort_ctas] += 1

    padded_per_expert = tuple(
        (count + block_size - 1) // block_size * block_size for count in counts)
    cumsum = [0]
    for padded in padded_per_expert:
        cumsum.append(cumsum[-1] + padded)
    total_padded = cumsum[-1]
    used_blocks = total_padded // block_size

    max_tokens_padded = numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_tokens_padded = _round_up(max_tokens_padded, block_size)
    if numel < num_experts:
        max_tokens_padded = min(numel * block_size, max_tokens_padded)
    max_m_blocks = (max_tokens_padded + block_size - 1) // block_size
    if total_padded > max_tokens_padded:
        raise ValueError(
            "moe_align_padded_rows_exceed_declared_buffer:"
            f"{total_padded}>{max_tokens_padded}")

    tail_blocks = max_m_blocks - used_blocks
    # Lines 272-276 are outside if (tid < num_experts): every worker thread
    # with tid in [0, stride) writes its residue, so the entire tail is
    # covered.  The same is true for the large-kernel loop at lines 179-184
    # with a 1024-thread stride.
    residual_writers = max(num_experts, _WARP_SIZE) if small_batch_expert_mode else _ALIGN_THREADS
    tail_blocks_written = tail_blocks
    expert_ids_blocks_written = used_blocks + tail_blocks_written

    small_threads = _FILL_THREADS + max(num_experts, _WARP_SIZE)
    small_shared_bytes = (
        (max(num_experts, _WARP_SIZE) + 1) * num_experts + (num_experts + 1)
    ) * _INT32_BYTES
    big_shared_bytes = padded_num_experts * _INT32_BYTES
    if small_batch_expert_mode:
        path = "small_batch_expert"
        symbol = "vllm::moe::moe_align_block_size_small_batch_expert_kernel"
        sort_symbol = None
        kernel_phases = ["small_batch_expert"]
        align_ctas = 1
        align_shared_bytes = small_shared_bytes
        sort_grid_value: tuple[int, int] | None = None
    else:
        path = "two_kernel_sort"
        symbol = "vllm::moe::moe_align_block_size_kernel"
        sort_symbol = "vllm::moe::count_and_sort_expert_tokens_kernel"
        kernel_phases = ["large_align", "large_sort"]
        align_ctas = 2
        align_shared_bytes = big_shared_bytes
        sort_grid_value = (1, sort_ctas)

    workspace_bytes = {
        "sorted_token_ids": max_tokens_padded * _INT32_BYTES,
        "expert_ids": max_m_blocks * _INT32_BYTES,
        "num_tokens_post_pad": _INT32_BYTES,
        "cumsum_buffer": (
            0 if small_batch_expert_mode else (num_experts + 1) * _INT32_BYTES),
    }
    return {
        "tokens": tokens,
        "top_k": top_k,
        "numel": numel,
        "num_experts": num_experts,
        "block_size": block_size,
        "pad_sorted_ids": pad_sorted_ids,
        "ignore_invalid_experts": ignore_invalid_experts,
        "expert_map_present": expert_map is not None,
        "expert_map": expert_map,
        "has_expert_map_kernel_arg": has_expert_map_kernel_arg,
        "expert_ids_post_gather": expert_ids_post_gather,
        "post_gather_tail_value": None,
        "topk_ids_dtype": topk_ids_dtype,
        "topk_element_bytes": topk_element_bytes,
        "valid_entries": valid_entries,
        "counts": tuple(counts),
        "padded_per_expert": padded_per_expert,
        "cumsum": tuple(cumsum),
        "total_padded": total_padded,
        "used_blocks": used_blocks,
        "max_tokens_padded": max_tokens_padded,
        "max_m_blocks": max_m_blocks,
        "tail_blocks": tail_blocks,
        "tail_blocks_written": tail_blocks_written,
        "residual_writers": residual_writers,
        "expert_ids_blocks_written": expert_ids_blocks_written,
        "sentinel": numel,
        "padded_num_experts": padded_num_experts,
        "small_batch_expert_mode": small_batch_expert_mode,
        "path": path,
        "symbol": symbol,
        "sort_symbol": sort_symbol,
        "kernel_phases": tuple(kernel_phases),
        "host_submissions": len(kernel_phases),
        "align_ctas": align_ctas,
        "align_shared_bytes": align_shared_bytes,
        "small_threads": small_threads if small_batch_expert_mode else None,
        "small_fill_threads": _FILL_THREADS if small_batch_expert_mode else None,
        "small_shared_bytes": small_shared_bytes if small_batch_expert_mode else None,
        "big_threads": _ALIGN_THREADS if not small_batch_expert_mode else None,
        "big_shared_bytes": big_shared_bytes if not small_batch_expert_mode else None,
        "sort_block_threads": _SORT_THREADS if not small_batch_expert_mode else None,
        "sort_ctas": sort_ctas or None,
        "sort_grid": sort_grid_value,
        "sort_cta_ids": tuple(sort_cta_ids),
        "sort_cta_valid": tuple(sort_cta_valid),
        "scan_lanes": _ALIGN_THREADS if not small_batch_expert_mode else None,
        "workspace_bytes": workspace_bytes,
    }
