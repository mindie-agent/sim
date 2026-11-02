"""HiSparse / Ascend Sparse KV Offloading & memfabric_hybrid physical modeling.

[Role]
Model NPU HBM vs. Host DRAM capacity partitioning, memfabric_hybrid acc_offload
GVA DMA kernels (sparse_copy, kv_exchange_copy, entry_gather), on-device Indexer
slot-map/LRU management (LIM), and decode/prefill sparse KV offloading across
full and shared Indexer layers (e.g., GLM-5.x, DeepSeek-V3.2/V4).

[Invariants]
- Indexer K cache (index_k) stays device-resident in NPU HBM across full sequence
  length while main MLA KV (kv_lora_rank + qk_rope_head_dim) is offloaded to the
  shared MemFabric Host DRAM pool (1 copy per DP replica across TP ranks).
- Shared Indexer layers (topk_owner_layer_id >= 0 and not owns_topk) skip both
  Indexer scoring and LIM slot-map/LRU management, reusing the owner layer's
  topk indices and miss transfer plan.
- Short decode sequences (context < topk_buffer_size, request_state == -3) incur
  zero H2D miss copy; long decode sequences transfer only miss_tokens = ceil(topk * miss_rate).
- State Honesty: Distinguishes native A5 verified kernels (memfabric_hybrid
  acc_offload, npu_lightning_indexer, npu_sparse_flash_attention) from A2/A3-only
  prebuilt ACLNN fused binaries (fused_lightning_indexer_manage, copy_sfa).

[Timeouts & Locks]
Purely analytical/discrete-event graph construction; O(layers + requests), no I/O.

[Side-Effect Policy]
Constructs immutable model/runtime metadata, OpSpec nodes, and Engine programs.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import math
from typing import Any, Mapping

from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from .engine import Allocation
from .mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .ownership import unit_group
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

IMPLEMENTATION = "vllm_ascend_sparse_kv_offload"
HISPARSE_ALIAS = "hisparse_kv_offload"

MEMFABRIC_SOURCE = {
    "repository": "https://gitcode.com/Ascend/memfabric_hybrid",
    "version": "1.3.0",
    "kernels": (
        "acc_offload_sparse_copy.h (OffloadMoeSparseCopyKernel / OffloadKVSparseCopyKernel)",
        "acc_offload_kv_exchange.h (OffloadKVExchangeKernel, 2 AIV cores blockDim-2..blockDim-1)",
        "acc_offload_entry_gather.h (OffloadEntryGatherKernel, 120 KiB ping-pong UB)",
    ),
    "caller": "vllm_ascend/attention/sfa_kv_offload.py & sparse_kv_offload_manager.py",
}

# Calibrated on live Ascend950DT_9582 (A5, 32 AIC / 64 AIV, memfabric_hybrid 1.3.0):
# - sparse_copy D2H: floor 20.8 us, setup 13.5 us, marginal GVA BW 108.9 GB/s
# - sparse_copy H2D: floor 20.6 us, setup 14.5 us, marginal GVA BW 95.1 GB/s
# - kv_exchange H2D (2 AIVs 62..63): floor 20.5 us, setup 9.8 us, marginal GVA BW 87.9 GB/s
# - kv_exchange D2H (2 AIVs 62..63): floor 20.5 us, setup 8.5 us, marginal GVA BW 99.6 GB/s
# - entry_gather H2D (64 AIVs, 1024B row): floor 21.1 us, setup 14.8 us, marginal GVA BW 106.8 GB/s
A5_GVA_CALIBRATION: dict[str, dict[str, float]] = {
    "sparse_copy_d2h": {"minimum_ms": 0.0208, "setup_ms": 0.0135, "gva_bw_bytes_per_ms": 108.9e6, "reference_uni_bw_gbps": 100.15},
    "sparse_copy_h2d": {"minimum_ms": 0.0206, "setup_ms": 0.0145, "gva_bw_bytes_per_ms": 95.1e6, "reference_uni_bw_gbps": 100.01},
    "kv_exchange_h2d": {"minimum_ms": 0.0205, "setup_ms": 0.0098, "gva_bw_bytes_per_ms": 87.9e6, "reference_uni_bw_gbps": 100.01},
    "kv_exchange_d2h": {"minimum_ms": 0.0205, "setup_ms": 0.0085, "gva_bw_bytes_per_ms": 99.6e6, "reference_uni_bw_gbps": 100.15},
    "entry_gather": {"minimum_ms": 0.0211, "setup_ms": 0.0148, "gva_bw_bytes_per_ms": 106.8e6, "reference_uni_bw_gbps": 100.01},
}

DEFAULT_OPTIONS: dict[str, Any] = {
    "dram_size_per_dp_gb": 128.0, "topk_buffer_size": 2048, "miss_rate": 0.15,
    "use_fused_copy_sfa": True, "use_fused_overlap": False, "keep_device_kv_cache": False,
    "store_index_k_on_host": False, "layerwise_kv_offload": False, "num_shared_layers": 2,
    "soc_family": None, "page_size": 128,
}


def copy_sfa_pool_capacity(max_num_seqs: int) -> int:
    """Match vllm-ascend SparseKvOffloadManager._compute_copy_sfa_pool_capacity."""
    n = max(4, int(max_num_seqs))
    return 1 << (n - 1).bit_length()


def _resolve_options(raw_options: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = {**DEFAULT_OPTIONS, **(raw_options or {})}
    if opts["dram_size_per_dp_gb"] <= 0:
        raise ModelingError("sparse_kv_offload_requires_positive_dram_size_per_dp_gb")
    if int(opts["topk_buffer_size"]) <= 0:
        raise ModelingError("sparse_kv_offload_requires_positive_topk_buffer_size")
    miss_rate = float(opts["miss_rate"])
    if not math.isfinite(miss_rate) or miss_rate < 0.0 or miss_rate > 1.0:
        raise ModelingError("sparse_kv_offload_miss_rate_must_be_in_0_1")
    if int(opts["num_shared_layers"]) < 1:
        raise ModelingError("sparse_kv_offload_num_shared_layers_must_be_positive")
    if int(opts["page_size"]) < 1:
        raise ModelingError("sparse_kv_offload_page_size_must_be_positive")
    if opts["use_fused_overlap"]:
        opts["use_fused_copy_sfa"] = False
    return opts


def _channel_bw_scale(channel_bw_gbps: float | None, profile_key: str) -> float:
    ref = A5_GVA_CALIBRATION[profile_key]["reference_uni_bw_gbps"]
    if channel_bw_gbps is None or channel_bw_gbps <= 0:
        return 1.0
    return float(channel_bw_gbps) / ref


def _machine_channel_bw(machine: Any, channel: str) -> float | None:
    channels = dict(getattr(machine, "transfer_channels", ()) or ())
    entry = dict(channels.get(channel, ()) or ())
    val = entry.get("effective_uni_bw_gbps")
    return float(val) if isinstance(val, (int, float)) and val > 0 else None


def sparse_kv_offload_memory_budget(
    model: Any,
    runtime: Any,
    machine: Any,
    workload: Any = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive NPU HBM vs. MemFabric Host DRAM capacity and static buffer budgets."""
    opts = _resolve_options(options)
    mla_layers = tuple(p for p in model.layer_plans if p.attention == "mla")
    indexed_layers = tuple(p for p in mla_layers if p.index_topk > 0)
    layerwise = bool(opts["layerwise_kv_offload"])
    if not indexed_layers and not layerwise:
        raise ModelingError("sparse_kv_offload_requires_indexed_mla_or_layerwise_offload")

    offload_layers = tuple(p.index for p in (mla_layers if layerwise else indexed_layers))
    full_indexer_layers = tuple(p.index for p in indexed_layers if p.owns_index_k)
    shared_indexer_layers = tuple(
        p.index for p in indexed_layers if not p.owns_index_k and p.topk_owner_layer_id >= 0
    )

    sample_plan = (indexed_layers or mla_layers)[0]
    kv_dtype = (model.quant.kv_dtype if model.quant else model.kv_dtype) or "bf16"
    kv_elements = sample_plan.cache_row_elements or (
        sample_plan.kv_lora_rank + sample_plan.qk_rope_head_dim
    )
    kv_row_bytes = int(kv_elements * dtype_bytes(kv_dtype))
    idx_row_bytes = (
        int(index_row_bytes(model, sample_plan.index_dim))
        if sample_plan.index_dim > 0 else 0
    )

    block_size = int(runtime.kv_block_size or opts["page_size"])
    max_seqs = int(runtime.max_num_seqs)
    max_model_len = int(runtime.max_model_len)
    max_blocks_per_req = math.ceil(max_model_len / block_size)
    topk_buffer_size = int(opts["topk_buffer_size"])

    pool_cap = copy_sfa_pool_capacity(max_seqs)
    if layerwise:
        topk_rows = 0
        buffer_width = 0
        topk_buffer_bytes_per_layer = 0
        lru_bytes_per_layer = 0
        slot_map_bytes_per_layer = 0
    elif opts["use_fused_copy_sfa"]:
        topk_rows = 2 * pool_cap
        buffer_width = topk_buffer_size + 2 * block_size
        topk_buffer_bytes_per_layer = topk_rows * buffer_width * kv_row_bytes
        lru_bytes_per_layer = pool_cap * topk_buffer_size * 4
        slot_map_bytes_per_layer = pool_cap * max_blocks_per_req * block_size * 4
    else:
        decode_width = max(1, int(math.ceil(getattr(runtime, "mtp_acceptance_length", 1.0))))
        topk_rows = min(int(runtime.max_num_batched_tokens), max_seqs * decode_width)
        buffer_width = topk_buffer_size
        topk_buffer_bytes_per_layer = topk_rows * buffer_width * kv_row_bytes
        lru_bytes_per_layer = 0
        slot_map_bytes_per_layer = 0

    static_npu_bytes_per_offload_layer = (
        topk_buffer_bytes_per_layer + lru_bytes_per_layer + slot_map_bytes_per_layer
    )
    p = runtime.parallel
    counts = runtime.pp_layer_partition or tuple(
        model.layers // p.pp + int(stage < model.layers % p.pp) for stage in range(p.pp)
    )
    layer_stage = tuple(stage for stage, count in enumerate(counts) for _ in range(count))
    static_npu_bytes_per_stage = [0.0] * p.pp
    for layer_idx in offload_layers:
        static_npu_bytes_per_stage[layer_stage[layer_idx]] += float(static_npu_bytes_per_offload_layer)

    # MemFabric Scene.SHARED Host DRAM pool per DP replica (shared across TP ranks).
    dram_pool_bytes_per_dp = int(float(opts["dram_size_per_dp_gb"]) * (1024 ** 3))
    dram_alignment_reserve_bytes = len(offload_layers) * 3 * 2 * 1024 * 1024
    usable_dram_bytes_per_dp = max(0, dram_pool_bytes_per_dp - dram_alignment_reserve_bytes)
    usable_dram_per_layer = usable_dram_bytes_per_dp // max(1, len(offload_layers))
    dram_block_bytes = block_size * kv_row_bytes
    dram_limit_blocks = max(0, usable_dram_per_layer // max(1, dram_block_bytes) - 1)
    dram_max_tokens_per_dp = dram_limit_blocks * block_size

    workload_blocks = max_seqs * max_blocks_per_req
    if workload is not None and getattr(workload, "requests", ()):
        req_blocks = sum(
            math.ceil((r.prompt_tokens + r.output_tokens) / block_size)
            for r in workload.requests
        )
        workload_blocks = max(workload_blocks, req_blocks)

    independent_layers = model.layers - len(offload_layers)
    num_shared_layers = int(opts["num_shared_layers"])
    active_npu_kv_layers = (
        independent_layers + min(num_shared_layers, len(offload_layers))
        if layerwise else (model.layers if opts["keep_device_kv_cache"] else 0)
    )

    arch_family = getattr(machine, "arch_family", "generic")
    soc_family = opts["soc_family"] or (
        "Ascend950" if arch_family == "A5" else "Ascend910_93" if arch_family == "A3" else "Ascend910B"
    )
    h2d_bw_gbps = _machine_channel_bw(machine, "h2d") or (100.01 if arch_family == "A5" else 50.0)
    d2h_bw_gbps = _machine_channel_bw(machine, "d2h") or (100.15 if arch_family == "A5" else 50.0)

    hardware_support = {
        "soc_family": soc_family,
        "arch_family": arch_family,
        "memfabric_acc_offload": "native_verified_on_a5" if arch_family == "A5" else "native_supported",
        "npu_lightning_indexer": "native_verified_on_a5" if arch_family == "A5" else "native_supported",
        "npu_sparse_flash_attention": "native_verified_on_a5" if arch_family == "A5" else "native_supported",
        "fused_lightning_indexer_manage": (
            "a2_a3_aclnn_binary_analytical_on_a5" if arch_family == "A5" else "native_aclnn_supported"
        ),
        "copy_sfa_attention": (
            "a2_a3_aclnn_binary_analytical_on_a5" if arch_family == "A5" else "native_aclnn_supported"
        ),
        "fused_sparse_attention_overlap": "native_ascendc_source_available",
    }

    return {
        "implementation": IMPLEMENTATION, "options": opts, "offload_layers": offload_layers,
        "full_indexer_layers": full_indexer_layers, "shared_indexer_layers": shared_indexer_layers,
        "kv_row_bytes": kv_row_bytes, "index_row_bytes": idx_row_bytes, "block_size": block_size,
        "topk_buffer_size": topk_buffer_size, "copy_sfa_pool_capacity": pool_cap,
        "topk_buffer_rows": topk_rows, "topk_buffer_width": buffer_width,
        "topk_buffer_bytes_per_layer": topk_buffer_bytes_per_layer,
        "lru_table_bytes_per_layer": lru_bytes_per_layer,
        "slot_map_bytes_per_layer": slot_map_bytes_per_layer,
        "static_npu_bytes_per_offload_layer": static_npu_bytes_per_offload_layer,
        "static_npu_bytes_per_stage_rank": tuple(static_npu_bytes_per_stage),
        "dram_pool_bytes_per_dp": dram_pool_bytes_per_dp,
        "dram_alignment_reserve_bytes": dram_alignment_reserve_bytes,
        "usable_dram_bytes_per_dp": usable_dram_bytes_per_dp,
        "usable_dram_bytes_per_layer": usable_dram_per_layer,
        "dram_block_bytes": dram_block_bytes, "dram_limit_blocks": dram_limit_blocks,
        "dram_max_tokens_per_dp": dram_max_tokens_per_dp,
        "workload_limit_blocks": workload_blocks, "active_npu_kv_layers": active_npu_kv_layers,
        "h2d_effective_bw_gbps": h2d_bw_gbps, "d2h_effective_bw_gbps": d2h_bw_gbps,
        "hardware_support": hardware_support, "source": MEMFABRIC_SOURCE,
    }


def prepare_case(model: Any, runtime: Any, machine: Any, workload: Any, binding: Any):
    """Attach sparse KV offload memory/execution contracts to model and runtime."""
    budget = sparse_kv_offload_memory_budget(
        model, runtime, machine, workload, binding.options
    )
    opts = budget["options"]
    offload_layers = budget["offload_layers"]

    # Configure CachePages groups so Memory(case) reflects NPU HBM residency:
    # - When keep_device_kv_cache=False (Decode offload), main mla_kv has 0 dynamic
    #   per-request pages in NPU HBM (hot topk_buffers are reserved statically),
    #   while index_k stays full-sequence in NPU HBM!
    # - When layerwise_kv_offload=True (Prefill layerwise offload), only
    #   active_npu_kv_layers retain NPU HBM mla_kv residency simultaneously.
    memory_cfg = dict(runtime.memory)
    rt_meta_dict = dict(runtime.metadata)
    base_groups = rt_meta_dict.get(
        "sparse_kv_offload_base_page_groups",
        tuple(memory_cfg.get("cache_page_groups", ())),
    )
    existing_groups = list(base_groups)
    if not opts["keep_device_kv_cache"] and not opts["layerwise_kv_offload"]:
        existing_groups.append({
            "layers": offload_layers,
            "tokens_per_page": budget["block_size"],
            "bytes_per_layer_page": budget["dram_block_bytes"],
            "components": ("mla_kv",),
            "resident_tokens": 0,
        })
    elif opts["layerwise_kv_offload"]:
        kept = max(0, int(opts["num_shared_layers"]))
        offloaded_zero = offload_layers[kept:]
        if offloaded_zero:
            existing_groups.append({
                "layers": offloaded_zero,
                "tokens_per_page": budget["block_size"],
                "bytes_per_layer_page": budget["dram_block_bytes"],
                "components": ("mla_kv",),
                "resident_tokens": 0,
            })
    memory_cfg["cache_page_groups"] = tuple(existing_groups)

    model_meta = {**dict(model.metadata), "sparse_kv_offload": budget}
    runtime_meta = {
        **rt_meta_dict,
        "sparse_kv_offload": budget,
        "sparse_kv_offload_base_page_groups": base_groups,
    }
    return (
        replace(model, metadata=tuple(sorted(model_meta.items()))),
        replace(runtime, memory=memory_cfg, metadata=tuple(sorted(runtime_meta.items()))),
    )


def strip_case(model: Any, runtime: Any):
    """Remove derived sparse_kv_offload metadata when unselected on case reuse."""
    rt_meta_dict = dict(runtime.metadata)
    memory_cfg = dict(runtime.memory)
    if "sparse_kv_offload_base_page_groups" in rt_meta_dict:
        memory_cfg["cache_page_groups"] = rt_meta_dict["sparse_kv_offload_base_page_groups"]
    model_meta = tuple((k, v) for k, v in model.metadata if k != "sparse_kv_offload")
    runtime_meta = tuple(
        (k, v) for k, v in runtime.metadata
        if k not in {"sparse_kv_offload", "sparse_kv_offload_base_page_groups"}
    )
    return replace(model, metadata=model_meta), replace(runtime, memory=memory_cfg, metadata=runtime_meta)


def _build_memfabric_sparse_copy(op: OpSpec, geometry: Geometry, *, direction: str) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("memfabric_call", {}))
    payload_bytes = float(spec.get("payload_bytes", op.hbm_bytes))
    entries = max(1, int(spec.get("entries", max(1, op.local_shape[0] if op.local_shape else 1))))
    aiv_units = int(geometry.resource_units.get("vector", max(1, geometry.compute_units * 2)))
    ub_capacity = int(geometry.buffers.get("ub", 196608))
    # memfabric_hybrid acc_offload_sparse_copy.h:
    # entries < aiv_units -> OffloadMoeSparseCopyKernel; entries >= aiv_units -> OffloadKVSparseCopyKernel.
    kernel_mode = "OffloadMoeSparseCopyKernel" if entries < aiv_units else "OffloadKVSparseCopyKernel"
    active_owners = min(aiv_units, entries)
    ub_per_owner = min(90112, ub_capacity)

    profile_key = "sparse_copy_d2h" if direction == "d2h" else "sparse_copy_h2d"
    cal = A5_GVA_CALIBRATION[profile_key]
    bw_gbps = spec.get("channel_bw_gbps")
    scale = _channel_bw_scale(bw_gbps, profile_key)
    gva_bw_bytes_per_ms = cal["gva_bw_bytes_per_ms"] * scale
    setup_ms = cal["setup_ms"]
    minimum_ms = cal["minimum_ms"]

    if direction == "d2h":
        work = Work(resources={"hbm": payload_bytes, "mte2_aiv": payload_bytes, "mte3": payload_bytes})
        gva_role = "mte3"
    else:
        work = Work(resources={"hbm": payload_bytes, "mte2_aiv": payload_bytes, "mte3": payload_bytes})
        gva_role = "mte2_aiv"

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        per_owner = payload_bytes / active_owners
        gva_name = context.resource_names.get(gva_role)
        per_owner_cap = {gva_name: gva_bw_bytes_per_ms / active_owners} if gva_name else {}

        def owner_prog(owner: int):
            owned = context.owned(owner, aiv_units, ("vector", "mte2_aiv", "mte3"))
            alloc = (
                (Allocation("mf_sparse_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            if direction == "d2h":
                dma_w = context.bind(Work(resources={
                    "hbm": per_owner, "mte2_aiv": per_owner, "mte3": per_owner,
                }))
            else:
                dma_w = context.bind(Work(resources={
                    "mte2_aiv": per_owner, "mte3": per_owner, "hbm": per_owner,
                }))
            stages = (
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_dma", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_dma",), free=alloc),
            )
            return KernelProgram(stages)

        return KernelProgram(
            (
                Stage("gva_aperture_setup", latency=setup_ms),
                Stage("aiv_dma", after=("gva_aperture_setup",),
                      program=ParallelProgram(tuple(owner_prog(i) for i in range(active_owners)))),
            ),
            minimum_ms=minimum_ms,
        )

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": kernel_mode,
        "direction": direction,
        "entries": entries,
        "active_aiv_owners": active_owners,
        "physical_aiv_units": aiv_units,
        "ub_bytes_per_owner": ub_per_owner,
        "gva_bw_gbps": gva_bw_bytes_per_ms / 1e6,
        "gva_setup_us": setup_ms * 1000.0,
        "gva_floor_us": minimum_ms * 1000.0,
    })


def _build_memfabric_kv_exchange(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("memfabric_call", {}))
    direction = str(spec.get("direction", "d2h")).lower()
    payload_bytes = float(spec.get("payload_bytes", op.hbm_bytes))
    pages = max(1, int(spec.get("pages", 1)))
    aiv_units = int(geometry.resource_units.get("vector", max(2, geometry.compute_units * 2)))
    ub_capacity = int(geometry.buffers.get("ub", 196608))
    # acc_offload_kv_exchange.h: only the last 2 AIV cores (subBlockNum-2, subBlockNum-1) execute!
    active_owners = min(2, aiv_units)
    owner_indices = tuple(range(max(0, aiv_units - active_owners), aiv_units))
    ub_per_owner = min(90112, ub_capacity)

    profile_key = "kv_exchange_d2h" if direction == "d2h" else "kv_exchange_h2d"
    cal = A5_GVA_CALIBRATION[profile_key]
    scale = _channel_bw_scale(spec.get("channel_bw_gbps"), profile_key)
    gva_bw_bytes_per_ms = cal["gva_bw_bytes_per_ms"] * scale
    setup_ms = cal["setup_ms"]
    minimum_ms = cal["minimum_ms"]

    work = Work(resources={"hbm": payload_bytes, "mte2_aiv": payload_bytes, "mte3": payload_bytes})
    gva_role = "mte3" if direction == "d2h" else "mte2_aiv"

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        per_owner = payload_bytes / active_owners
        gva_name = context.resource_names.get(gva_role)
        per_owner_cap = {gva_name: gva_bw_bytes_per_ms / active_owners} if gva_name else {}

        def owner_prog(phys_owner: int):
            owned = context.owned(phys_owner, aiv_units, ("vector", "mte2_aiv", "mte3"))
            alloc = (
                (Allocation("mf_kv_ex_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(phys_owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            dma_w = context.bind(Work(resources={
                "hbm": per_owner, "mte2_aiv": per_owner, "mte3": per_owner,
            }))
            return KernelProgram((
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_dma", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_dma",), free=alloc),
            ))

        return KernelProgram(
            (
                Stage("gva_aperture_setup", latency=setup_ms),
                Stage("aiv_tail2_dma", after=("gva_aperture_setup",),
                      program=ParallelProgram(tuple(owner_prog(idx) for idx in owner_indices))),
            ),
            minimum_ms=minimum_ms,
        )

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": "OffloadKVExchangeKernel",
        "direction": direction,
        "pages": pages,
        "active_aiv_owners": owner_indices,
        "physical_aiv_units": aiv_units,
        "ub_bytes_per_owner": ub_per_owner,
        "gva_bw_gbps": gva_bw_bytes_per_ms / 1e6,
    })


def _build_memfabric_entry_gather(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("memfabric_call", {}))
    payload_bytes = float(spec.get("payload_bytes", op.hbm_bytes))
    rows = max(1, int(spec.get("rows", max(1, op.local_shape[0] if op.local_shape else 1))))
    aiv_units = int(geometry.resource_units.get("vector", max(1, geometry.compute_units * 2)))
    ub_capacity = int(geometry.buffers.get("ub", 196608))
    # acc_offload_entry_gather.h: 120 KiB ping-pong UB on >=128 KiB UB, else 88 KiB.
    ub_per_owner = 122880 if ub_capacity >= 131072 else min(90112, ub_capacity)
    active_owners = min(aiv_units, rows)

    cal = A5_GVA_CALIBRATION["entry_gather"]
    scale = _channel_bw_scale(spec.get("channel_bw_gbps"), "entry_gather")
    gva_bw_bytes_per_ms = cal["gva_bw_bytes_per_ms"] * scale

    work = Work(resources={"hbm": payload_bytes, "mte2_aiv": payload_bytes, "mte3": payload_bytes})

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        per_owner = payload_bytes / active_owners
        gva_name = context.resource_names.get("mte2_aiv")
        per_owner_cap = {gva_name: gva_bw_bytes_per_ms / active_owners} if gva_name else {}

        def owner_prog(owner: int):
            owned = context.owned(owner, aiv_units, ("vector", "mte2_aiv", "mte3"))
            alloc = (
                (Allocation("mf_entry_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            dma_w = context.bind(Work(resources={
                "mte2_aiv": per_owner, "mte3": per_owner, "hbm": per_owner,
            }))
            return KernelProgram((
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_gather", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_gather",), free=alloc),
            ))

        return KernelProgram(
            (
                Stage("gva_aperture_setup", latency=cal["setup_ms"]),
                Stage("aiv_gather", after=("gva_aperture_setup",),
                      program=ParallelProgram(tuple(owner_prog(i) for i in range(active_owners)))),
            ),
            minimum_ms=cal["minimum_ms"],
        )

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": "OffloadEntryGatherKernel",
        "rows": rows,
        "active_aiv_owners": active_owners,
        "ub_bytes_per_owner": ub_per_owner,
        "gva_bw_gbps": gva_bw_bytes_per_ms / 1e6,
    })


def _build_lim_plan(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    work = Work(
        matrix={op.dtype: op.flops} if op.flops else {},
        resources={
            "hbm": op.hbm_bytes,
            "mte2_aiv": op.mte2_bytes,
            "mte3": op.mte3_bytes,
            "vector": op.vector_ops,
        },
    )

    def program(context):
        return KernelProgram((Stage("lim_slot_map_lru", context.bind(work)),))

    return OperatorPlan(work, program, metadata={
        "execution_class": "vector",
        "native_kernel": "npu_fused_lightning_indexer_manage",
    })


def _build_overlapped_sfa_plan(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("sparse_offload_sfa", {}))
    miss_bytes = float(spec.get("miss_h2d_bytes", 0.0))
    attn_hbm_bytes = max(0.0, float(op.hbm_bytes) - miss_bytes)
    dtype = op.dtype or "bf16"
    cal = A5_GVA_CALIBRATION["sparse_copy_h2d"]
    scale = _channel_bw_scale(spec.get("channel_bw_gbps"), "sparse_copy_h2d")
    gva_bw_bytes_per_ms = cal["gva_bw_bytes_per_ms"] * scale

    work = Work(
        matrix={dtype: op.flops} if op.flops else {},
        resources={
            "hbm": attn_hbm_bytes + miss_bytes,
            "mte2_aic": attn_hbm_bytes,
            "mte2_aiv": miss_bytes,
            "mte3": float(op.mte3_bytes) + miss_bytes,
            "vector": float(op.vector_ops),
        },
    )

    def program(context):
        aic_work = Work(
            matrix={dtype: op.flops} if op.flops else {},
            resources={"hbm": attn_hbm_bytes, "mte2_aic": attn_hbm_bytes,
                       "mte3": float(op.mte3_bytes), "vector": float(op.vector_ops)},
        )
        aic_prog = KernelProgram((Stage("sfa_aic", context.bind(aic_work)),))
        if miss_bytes <= 0:
            return aic_prog
        gva_name = context.resource_names.get("mte2_aiv")
        cap = {gva_name: gva_bw_bytes_per_ms} if gva_name else {}
        aiv_prog = KernelProgram(
            (
                Stage("gva_setup", latency=cal["setup_ms"]),
                Stage("gva_h2d_gather", context.bind(Work(resources={"mte2_aiv": miss_bytes})),
                      after=("gva_setup",), caps=cap),
                Stage("topk_buffer_write", context.bind(Work(resources={"hbm": miss_bytes, "mte3": miss_bytes})),
                      after=("gva_h2d_gather",)),
            ),
            minimum_ms=cal["minimum_ms"],
        )
        return ParallelProgram((aiv_prog, aic_prog))

    return OperatorPlan(work, program, metadata={
        "execution_class": "mixed_aiv_aic_overlap",
        "miss_h2d_bytes": miss_bytes,
        "attn_hbm_bytes": attn_hbm_bytes,
        "gva_h2d_bw_gbps": gva_bw_bytes_per_ms / 1e6,
    })


def recipes() -> dict[str, PlannedRecipe]:
    return {
        "memfabric_sparse_copy_d2h": PlannedRecipe(partial(_build_memfabric_sparse_copy, direction="d2h")),
        "memfabric_sparse_copy_h2d": PlannedRecipe(partial(_build_memfabric_sparse_copy, direction="h2d")),
        "memfabric_kv_exchange": PlannedRecipe(_build_memfabric_kv_exchange),
        "memfabric_entry_gather": PlannedRecipe(_build_memfabric_entry_gather),
        "fused_lightning_indexer_manage": PlannedRecipe(_build_lim_plan),
        "copy_sfa_attention": PlannedRecipe(_build_overlapped_sfa_plan),
        "fused_sparse_attention_overlap": PlannedRecipe(_build_overlapped_sfa_plan),
    }


def lower_prefill_offload(
    b: Any, *, plan: Any, views: tuple[Any, ...], store_id: str, budget: Mapping[str, Any]
) -> str:
    """Emit side-stream MemFabric 2-AIV kv_exchange_copy for layerwise prefill offload."""
    opts = budget["options"]
    if not opts.get("layerwise_kv_offload") or plan.index not in budget["offload_layers"]:
        return store_id
    valid_tokens = sum(v.q_len for v in views)
    page_size = int(budget["block_size"])
    pages = max(1, math.ceil(valid_tokens / page_size))
    payload_bytes = float(valid_tokens * budget["kv_row_bytes"])
    op_id = b.work(
        "memfabric_kv_exchange_d2h", (store_id,), (pages, page_size, plan.cache_row_elements),
        read=payload_bytes, write=payload_bytes, family="kv_offload_exchange", dtype=b.model.kv_dtype,
        metadata=(
            ("kernel", "OffloadKVExchangeKernel"), ("implementation", IMPLEMENTATION),
            ("memfabric_call", {"direction": "d2h", "pages": pages, "page_size": page_size,
                                "payload_bytes": payload_bytes, "channel_bw_gbps": budget["d2h_effective_bw_gbps"]}),
            ("hardware_support", budget["hardware_support"]),
        ),
    )
    b.ops[-1] = replace(b.ops[-1], recipe="memfabric_kv_exchange", stream="offload", source_boundary="kernel")
    return op_id


def lower_decode_offload(
    b: Any, *, plan: Any, views: tuple[Any, ...], store_id: str, topk_id: str | None, budget: Mapping[str, Any]
) -> tuple[str, str | None, dict[str, Any]]:
    """Lower Decode Sparse KV Offload (D2H new-token writeback + LIM + H2D miss copy)."""
    opts = budget["options"]
    valid_tokens = sum(v.q_len for v in views)
    kv_row_bytes = int(budget["kv_row_bytes"])
    topk_buffer_size = int(budget["topk_buffer_size"])
    miss_rate = float(opts["miss_rate"])

    # 1. New decode tokens D2H writeback to shared MemFabric Host DRAM (2 entries/token: k_nope + k_pe).
    d2h_bytes = float(valid_tokens * kv_row_bytes)
    d2h_entries = max(1, 2 * valid_tokens)
    d2h_id = b.work(
        "memfabric_sparse_copy_d2h", (store_id,), (valid_tokens, plan.cache_row_elements),
        read=d2h_bytes, write=d2h_bytes, family="kv_offload_d2h", dtype=b.model.kv_dtype,
        metadata=(
            ("kernel", "OffloadKVSparseCopyKernel" if d2h_entries >= 64 else "OffloadMoeSparseCopyKernel"),
            ("implementation", IMPLEMENTATION),
            ("memfabric_call", {"direction": "d2h", "entries": d2h_entries, "tokens": valid_tokens,
                                "payload_bytes": d2h_bytes, "channel_bw_gbps": budget["d2h_effective_bw_gbps"]}),
            ("hardware_support", budget["hardware_support"]),
        ),
    )
    b.ops[-1] = replace(b.ops[-1], recipe="memfabric_sparse_copy_d2h", source_boundary="kernel")

    # 2. Classify request states (-3 short, -1 first long step, -2 steady long decode).
    short_reqs, long_reqs, total_miss_tokens = 0, 0, 0
    request_states = []
    for v in views:
        if v.context < topk_buffer_size:
            short_reqs += 1
            request_states.append(-3)
        elif v.context < topk_buffer_size + v.q_len:
            long_reqs += 1
            request_states.append(-1)
            total_miss_tokens += min(v.context, plan.index_topk or topk_buffer_size)
        else:
            long_reqs += 1
            request_states.append(-2)
            active_topk = min(v.context, plan.index_topk or topk_buffer_size)
            total_miss_tokens += int(math.ceil(active_topk * miss_rate))

    is_shared_indexer_layer = (not plan.indexer and not plan.owns_topk and plan.topk_owner_layer_id >= 0)
    lim_id = topk_id

    # 3. On full Indexer layers with long requests, emit LIM slot-map lookup & LRU update.
    #    Shared Indexer layers (GLM-5.x "shared" layers) skip LIM and reuse owner's miss_plan!
    if long_reqs > 0 and topk_id is not None and not is_shared_indexer_layer and opts["use_fused_copy_sfa"]:
        topk_width = plan.index_topk or topk_buffer_size
        lim_read = float(long_reqs * topk_width * (4 + 4 + 4))
        lim_write = float(long_reqs * topk_width * (4 + 8))
        lim_vec = float(long_reqs * topk_width * 6)
        lim_id = b.work(
            "fused_lightning_indexer_manage", (topk_id,), (long_reqs, topk_width),
            vector=lim_vec, read=lim_read, write=lim_write, family="indexer_slot_manage", dtype="int32",
            metadata=(
                ("kernel", "npu_fused_lightning_indexer_manage"), ("implementation", IMPLEMENTATION),
                ("request_states", tuple(request_states)), ("long_requests", long_reqs),
                ("short_requests", short_reqs), ("miss_tokens", total_miss_tokens),
                ("skip_topk", False), ("hardware_support", budget["hardware_support"]),
            ),
        )
        b.ops[-1] = replace(b.ops[-1], recipe="fused_lightning_indexer_manage", source_boundary="kernel")

    # 4. H2D miss token gather from MemFabric Host DRAM into NPU topk_buffers.
    miss_h2d_bytes = float(total_miss_tokens * kv_row_bytes)
    h2d_id = d2h_id
    if total_miss_tokens > 0 and not opts["use_fused_overlap"]:
        h2d_entries = 2 * total_miss_tokens
        deps = tuple(dict.fromkeys(x for x in (d2h_id, lim_id) if x))
        h2d_id = b.work(
            "memfabric_sparse_copy_h2d", deps, (total_miss_tokens, plan.cache_row_elements),
            read=miss_h2d_bytes, write=miss_h2d_bytes, family="kv_offload_h2d", dtype=b.model.kv_dtype,
            metadata=(
                ("kernel", "OffloadKVSparseCopyKernel" if h2d_entries >= 64 else "OffloadMoeSparseCopyKernel"),
                ("implementation", IMPLEMENTATION),
                ("memfabric_call", {"direction": "h2d", "entries": h2d_entries, "tokens": total_miss_tokens,
                                    "payload_bytes": miss_h2d_bytes, "channel_bw_gbps": budget["h2d_effective_bw_gbps"]}),
                ("request_states", tuple(request_states)), ("shared_indexer_reuse", is_shared_indexer_layer),
                ("skip_topk", is_shared_indexer_layer), ("hardware_support", budget["hardware_support"]),
            ),
        )
        b.ops[-1] = replace(b.ops[-1], recipe="memfabric_sparse_copy_h2d", source_boundary="kernel")

    info = {
        "request_states": tuple(request_states), "short_requests": short_reqs,
        "long_requests": long_reqs, "miss_tokens": total_miss_tokens,
        "miss_h2d_bytes": miss_h2d_bytes, "d2h_writeback_bytes": d2h_bytes,
        "shared_indexer_reuse": is_shared_indexer_layer, "skip_topk": is_shared_indexer_layer,
        "use_fused_overlap": bool(opts["use_fused_overlap"]),
        "use_fused_copy_sfa": bool(opts["use_fused_copy_sfa"]),
        "channel_bw_gbps": budget["h2d_effective_bw_gbps"],
        "hardware_support": budget["hardware_support"],
    }
    return h2d_id, lim_id, info


def lower_op(op: OpSpec, **options: Any) -> OpSpec:
    """Operator-level lowering hook for standalone memfabric/sparse_kv_offload recipes."""
    opts = _resolve_options(options)
    meta = dict(op.metadata)
    recipe_name = op.recipe if op.recipe in recipes() else op.family
    if recipe_name not in recipes():
        recipe_name = "memfabric_sparse_copy_h2d"
    call = dict(meta.get("memfabric_call", {}))
    call.setdefault("payload_bytes", float(op.hbm_bytes))
    call.setdefault("entries", max(1, int(op.local_shape[0]) * 2 if op.local_shape else 2))
    meta["memfabric_call"] = call
    meta["sparse_kv_offload_options"] = opts
    return replace(op, recipe=recipe_name, source_boundary="kernel", metadata=tuple(meta.items()))


__all__ = [
    "A5_GVA_CALIBRATION", "HISPARSE_ALIAS", "IMPLEMENTATION", "MEMFABRIC_SOURCE",
    "copy_sfa_pool_capacity", "lower_decode_offload", "lower_op", "lower_prefill_offload",
    "prepare_case", "recipes", "sparse_kv_offload_memory_budget", "strip_case",
]
