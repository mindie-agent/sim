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

from mindie_sim.catalog.cache import index_row_bytes, packed_region_row_bytes
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.ownership import read_role, unit_group
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage


BASELINE_OFFLOAD_UB_BYTES = 90112
DEVICE_SLOT_INDEX_BYTES = 4
GIB_EXPONENT = 3
HOST_BUFFER_POOLS = 3
HOST_SLOT_INDEX_BYTES = 4
KV_COPY_MIN_ENTRIES = 64
LIMIT_SCAN_OPS_PER_ENTRY = 6
MIN_OFFLOAD_POOL_REQUESTS = 4
OFFLOAD_ADDRESS_PAIR_BYTES = 8
OFFLOAD_INACTIVE_REQUEST = -3
OFFLOAD_INDEX_ELEMENT_BYTES = 4
TOPK_INDEX_BYTES = 4
WIDE_OFFLOAD_MIN_UB_BYTES = 131072
WIDE_OFFLOAD_UB_BYTES = 122880
NS_TO_MS = 1e-6
US_TO_MS = 1e-3
BYTES_PER_S_TO_GBPS = 1e-9
GBPS_TO_BYTES_PER_MS = 1e6

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

DEFAULT_OPTIONS: dict[str, Any] = {
    "dram_size_per_dp_gb": 128.0, "topk_buffer_size": 2048, "miss_rate": 0.15,
    "use_fused_copy_sfa": True, "use_fused_overlap": False, "keep_device_kv_cache": False,
    "store_index_k_on_host": False, "layerwise_kv_offload": False, "num_shared_layers": 2,
    "soc_family": None, "page_size": 128,
}


def copy_sfa_pool_capacity(max_num_seqs: int) -> int:
    """Match vllm-ascend SparseKvOffloadManager._compute_copy_sfa_pool_capacity."""
    n = max(MIN_OFFLOAD_POOL_REQUESTS, int(max_num_seqs))
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


def _machine_channel_spec(machine: Any, channel: str) -> tuple[float | None, float | None]:
    channels = dict(getattr(machine, "transfer_channels", ()) or ())
    entry = dict(channels.get(channel, ()) or ())
    bw = entry.get("effective_uni_bw_gbps")
    alpha = entry.get("alpha_us")
    bw_val = float(bw) if isinstance(bw, (int, float)) and bw > 0 else None
    alpha_val = float(alpha) if isinstance(alpha, (int, float)) and alpha >= 0 else None
    return bw_val, alpha_val


def _primitive_host_dma(table: Any, direction: str) -> tuple[float | None, float]:
    """Read calibrated host DMA bandwidth (bytes/ms) and setup latency (ms) from PrimitiveTable."""
    if table is None:
        return None, 0.0
    primitives = getattr(table, "primitives", ())
    bw_bytes_per_ms: float | None = None
    setup_ms = 0.0
    if "P20_host_dma_sync" in primitives:
        metrics = table.cell("P20_host_dma_sync").metrics
        bw_key = "host_d2h_bytes_per_s" if direction == "d2h" else "host_h2d_bytes_per_s"
        bw_band = metrics.get(bw_key)
        if bw_band is not None and bw_band.p50 is not None and bw_band.p50 > 0.0:
            bw_bytes_per_ms = float(bw_band.p50) * US_TO_MS
        setup_band = metrics.get("pinned_dma_startup_ns")
        if setup_band is None and direction == "d2h":
            setup_band = metrics.get("eager_d2h_sync_ns")
        if setup_band is not None and setup_band.p50 is not None and setup_band.p50 > 0.0:
            setup_ms = float(setup_band.p50) * NS_TO_MS
    return bw_bytes_per_ms, setup_ms


def _resolve_host_dma_channel(context: Any, spec: Mapping[str, Any], direction: str) -> tuple[float | None, float]:
    """Resolve host DMA bandwidth (bytes/ms) and setup latency (ms) from transfer_channels / PrimitiveTable."""
    table = getattr(context, "primitive_table", None)
    prim_bw, prim_setup = _primitive_host_dma(table, direction)
    raw_bw = spec.get("channel_bw_gbps")
    bw_bytes_per_ms = (
        float(raw_bw) * GBPS_TO_BYTES_PER_MS
        if isinstance(raw_bw, (int, float)) and raw_bw > 0
        else prim_bw
    )
    raw_alpha = spec.get("channel_alpha_us")
    setup_ms = (
        float(raw_alpha) * US_TO_MS
        if isinstance(raw_alpha, (int, float)) and raw_alpha >= 0
        else prim_setup
    )
    return bw_bytes_per_ms, setup_ms


def _explicit_aiv_units(geometry: Geometry) -> int:
    units = (geometry.resource_units or {}).get("vector")
    if units is None or int(units) <= 0:
        raise ModelingError("sparse_kv_offload_requires_explicit_vector_units")
    return int(units)


def sparse_kv_offload_memory_budget(
    model: Any,
    runtime: Any,
    machine: Any,
    workload: Any = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive NPU HBM vs. MemFabric Host DRAM capacity and static buffer budgets."""
    opts = _resolve_options(options)
    mla_layers = tuple(
        p for p in model.layer_plans
        if p.attention == "mla" and str(getattr(p, "mla_variant", "") or "") == "latent_mla"
    )
    indexed_layers = tuple(p for p in mla_layers if p.index_topk > 0)
    layerwise = bool(opts["layerwise_kv_offload"])
    if not indexed_layers and not layerwise:
        raise ModelingError("sparse_kv_offload_requires_indexed_mla_or_layerwise_offload")

    offload_plans = mla_layers if layerwise else indexed_layers
    offload_layers = tuple(p.index for p in offload_plans)
    full_indexer_layers = tuple(p.index for p in indexed_layers if p.owns_index_k)
    shared_indexer_layers = tuple(
        p.index for p in indexed_layers if not p.owns_index_k and p.topk_owner_layer_id >= 0
    )

    kv_row_bytes_by_layer: dict[int, int] = {}
    index_row_bytes_by_layer: dict[int, int] = {}
    for layer_plan in offload_plans:
        kv_elements = layer_plan.cache_row_elements or (
            layer_plan.kv_lora_rank + layer_plan.qk_rope_head_dim
        )
        kv_row_bytes_by_layer[layer_plan.index] = int(
            packed_region_row_bytes(model, "mla_kv", kv_elements)
        )
        index_row_bytes_by_layer[layer_plan.index] = (
            int(index_row_bytes(model, layer_plan.index_dim))
            if layer_plan.index_dim > 0 else 0
        )
    kv_row_bytes = kv_row_bytes_by_layer[offload_layers[0]]
    idx_row_bytes = index_row_bytes_by_layer[offload_layers[0]]

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
        lru_bytes_per_layer = pool_cap * topk_buffer_size * OFFLOAD_INDEX_ELEMENT_BYTES
        slot_map_bytes_per_layer = pool_cap * max_blocks_per_req * block_size * OFFLOAD_INDEX_ELEMENT_BYTES
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
        layer_topk = 0 if layerwise else (topk_rows * buffer_width * kv_row_bytes_by_layer[layer_idx])
        static_npu_bytes_per_stage[layer_stage[layer_idx]] += float(
            layer_topk + lru_bytes_per_layer + slot_map_bytes_per_layer
        )

    # MemFabric Scene.SHARED Host DRAM pool per DP replica (shared across TP ranks).
    dram_pool_bytes_per_dp = int(float(opts["dram_size_per_dp_gb"]) * (1024 ** GIB_EXPONENT))
    dram_alignment_reserve_bytes = len(offload_layers) * HOST_BUFFER_POOLS * 2 * 1024 * 1024
    usable_dram_bytes_per_dp = max(0, dram_pool_bytes_per_dp - dram_alignment_reserve_bytes)
    usable_dram_per_layer = usable_dram_bytes_per_dp // max(1, len(offload_layers))
    dram_block_bytes = block_size * kv_row_bytes
    total_dram_block_bytes = sum(block_size * kv_row_bytes_by_layer[idx] for idx in offload_layers)
    dram_limit_blocks = max(0, usable_dram_bytes_per_dp // max(1, total_dram_block_bytes) - 1)
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
    h2d_bw_gbps, h2d_alpha_us = _machine_channel_spec(machine, "h2d")
    d2h_bw_gbps, d2h_alpha_us = _machine_channel_spec(machine, "d2h")
    if h2d_bw_gbps is None or d2h_bw_gbps is None:
        raise ModelingError("sparse_kv_offload_host_channel_bw_missing")

    # Step 2: verify exact kernel availability via backends.abi_registry.ProbedAbiRegistry once W1 merges.
    hardware_support = {
        "soc_family": soc_family,
        "arch_family": arch_family,
        "abi_verification": "deferred_to_abi_registry",
        "memfabric_acc_offload": "host_gva_dma_modeled",
        "npu_lightning_indexer": "native_indexer_modeled",
        "npu_sparse_flash_attention": "native_sfa_modeled",
        "fused_lightning_indexer_manage": (
            "arch22_aclnn_binary" if arch_family in {"A2", "A3"} else "analytical_model"
        ),
        "copy_sfa_attention": (
            "arch22_aclnn_binary" if arch_family in {"A2", "A3"} else "analytical_model"
        ),
        "fused_sparse_attention_overlap": "ascendc_overlap_modeled",
    }

    return {
        "implementation": IMPLEMENTATION, "options": opts, "offload_layers": offload_layers,
        "full_indexer_layers": full_indexer_layers, "shared_indexer_layers": shared_indexer_layers,
        "kv_row_bytes": kv_row_bytes, "index_row_bytes": idx_row_bytes,
        "kv_row_bytes_by_layer": kv_row_bytes_by_layer,
        "index_row_bytes_by_layer": index_row_bytes_by_layer,
        "block_size": block_size,
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
        "h2d_alpha_us": h2d_alpha_us, "d2h_alpha_us": d2h_alpha_us,
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
    row_by_layer = budget.get("kv_row_bytes_by_layer") or {}
    target_zero_layers = (
        offload_layers
        if (not opts["keep_device_kv_cache"] and not opts["layerwise_kv_offload"])
        else (offload_layers[max(0, int(opts["num_shared_layers"])):] if opts["layerwise_kv_offload"] else ())
    )
    grouped_by_page_bytes: dict[int, list[int]] = {}
    for layer_idx in target_zero_layers:
        page_b = budget["block_size"] * int(row_by_layer.get(layer_idx, budget["kv_row_bytes"]))
        grouped_by_page_bytes.setdefault(page_b, []).append(layer_idx)
    for page_b, group_layers in grouped_by_page_bytes.items():
        existing_groups.append({
            "layers": tuple(group_layers),
            "tokens_per_page": budget["block_size"],
            "bytes_per_layer_page": page_b,
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
    aiv_units = _explicit_aiv_units(geometry)
    if "ub" not in geometry.buffers or int(geometry.buffers["ub"]) <= 0:
        raise ModelingError("sparse_kv_offload_requires_explicit_ub_bytes")
    ub_capacity = int(geometry.buffers["ub"])
    # memfabric_hybrid acc_offload_sparse_copy.h:
    # entries < aiv_units -> OffloadMoeSparseCopyKernel; entries >= aiv_units -> OffloadKVSparseCopyKernel.
    kernel_mode = "OffloadMoeSparseCopyKernel" if entries < aiv_units else "OffloadKVSparseCopyKernel"
    active_owners = min(aiv_units, entries)
    ub_per_owner = min(BASELINE_OFFLOAD_UB_BYTES, ub_capacity)

    rpipe = read_role(geometry.resource_units or {}, aiv=0)
    work = Work(resources={"hbm": payload_bytes, rpipe: payload_bytes, "mte3": payload_bytes})

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        pipe = read_role(context.resource_names, aiv=0)
        eff_gva = "mte3" if direction == "d2h" else pipe
        per_owner = payload_bytes / active_owners
        gva_bw_bytes_per_ms, setup_ms = _resolve_host_dma_channel(context, spec, direction)
        gva_name = context.resource_names.get(eff_gva)
        per_owner_cap = (
            {gva_name: gva_bw_bytes_per_ms / active_owners}
            if (gva_name and gva_bw_bytes_per_ms and gva_bw_bytes_per_ms > 0)
            else {}
        )

        def owner_prog(owner: int):
            owned = context.owned(owner, aiv_units, ("vector", pipe, "mte3"))
            alloc = (
                (Allocation("mf_sparse_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            dma_w = context.bind(Work(resources={
                "hbm": per_owner, pipe: per_owner, "mte3": per_owner,
            }))
            stages = (
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_dma", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_dma",), free=alloc),
            )
            return KernelProgram(stages)

        return KernelProgram((
            Stage("gva_aperture_setup", latency=setup_ms),
            Stage("aiv_dma", after=("gva_aperture_setup",),
                  program=ParallelProgram(tuple(owner_prog(i) for i in range(active_owners)))),
        ))

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": kernel_mode,
        "direction": direction,
        "entries": entries,
        "active_aiv_owners": active_owners,
        "physical_aiv_units": aiv_units,
        "ub_bytes_per_owner": ub_per_owner,
    })


def _build_memfabric_kv_exchange(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("memfabric_call", {}))
    direction = str(spec.get("direction", "d2h")).lower()
    payload_bytes = float(spec.get("payload_bytes", op.hbm_bytes))
    pages = max(1, int(spec.get("pages", 1)))
    aiv_units = _explicit_aiv_units(geometry)
    if "ub" not in geometry.buffers or int(geometry.buffers["ub"]) <= 0:
        raise ModelingError("sparse_kv_offload_requires_explicit_ub_bytes")
    ub_capacity = int(geometry.buffers["ub"])
    # acc_offload_kv_exchange.h: only the last 2 AIV cores (subBlockNum-2, subBlockNum-1) execute!
    active_owners = min(2, aiv_units)
    owner_indices = tuple(range(max(0, aiv_units - active_owners), aiv_units))
    ub_per_owner = min(BASELINE_OFFLOAD_UB_BYTES, ub_capacity)

    rpipe = read_role(geometry.resource_units or {}, aiv=0)
    work = Work(resources={"hbm": payload_bytes, rpipe: payload_bytes, "mte3": payload_bytes})

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        pipe = read_role(context.resource_names, aiv=0)
        eff_gva = "mte3" if direction == "d2h" else pipe
        per_owner = payload_bytes / active_owners
        gva_bw_bytes_per_ms, setup_ms = _resolve_host_dma_channel(context, spec, direction)
        gva_name = context.resource_names.get(eff_gva)
        per_owner_cap = (
            {gva_name: gva_bw_bytes_per_ms / active_owners}
            if (gva_name and gva_bw_bytes_per_ms and gva_bw_bytes_per_ms > 0)
            else {}
        )

        def owner_prog(phys_owner: int):
            owned = context.owned(phys_owner, aiv_units, ("vector", pipe, "mte3"))
            alloc = (
                (Allocation("mf_kv_ex_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(phys_owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            dma_w = context.bind(Work(resources={
                "hbm": per_owner, pipe: per_owner, "mte3": per_owner,
            }))
            return KernelProgram((
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_dma", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_dma",), free=alloc),
            ))

        return KernelProgram((
            Stage("gva_aperture_setup", latency=setup_ms),
            Stage("aiv_tail2_dma", after=("gva_aperture_setup",),
                  program=ParallelProgram(tuple(owner_prog(idx) for idx in owner_indices))),
        ))

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": "OffloadKVExchangeKernel",
        "direction": direction,
        "pages": pages,
        "active_aiv_owners": owner_indices,
        "physical_aiv_units": aiv_units,
        "ub_bytes_per_owner": ub_per_owner,
    })


def _build_memfabric_entry_gather(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    meta = dict(op.metadata)
    spec = dict(meta.get("memfabric_call", {}))
    payload_bytes = float(spec.get("payload_bytes", op.hbm_bytes))
    rows = max(1, int(spec.get("rows", max(1, op.local_shape[0] if op.local_shape else 1))))
    aiv_units = _explicit_aiv_units(geometry)
    if "ub" not in geometry.buffers or int(geometry.buffers["ub"]) <= 0:
        raise ModelingError("sparse_kv_offload_requires_explicit_ub_bytes")
    ub_capacity = int(geometry.buffers["ub"])
    # acc_offload_entry_gather.h: 120 KiB ping-pong UB on >=128 KiB UB, else 88 KiB.
    ub_per_owner = WIDE_OFFLOAD_UB_BYTES if ub_capacity >= WIDE_OFFLOAD_MIN_UB_BYTES else min(BASELINE_OFFLOAD_UB_BYTES, ub_capacity)
    active_owners = min(aiv_units, rows)

    rpipe = read_role(geometry.resource_units or {}, aiv=0)
    work = Work(resources={"hbm": payload_bytes, rpipe: payload_bytes, "mte3": payload_bytes})

    def program(context):
        if payload_bytes <= 0:
            return KernelProgram(())
        pipe = read_role(context.resource_names, aiv=0)
        per_owner = payload_bytes / active_owners
        gva_bw_bytes_per_ms, setup_ms = _resolve_host_dma_channel(context, spec, "h2d")
        gva_name = context.resource_names.get(pipe)
        per_owner_cap = (
            {gva_name: gva_bw_bytes_per_ms / active_owners}
            if (gva_name and gva_bw_bytes_per_ms and gva_bw_bytes_per_ms > 0)
            else {}
        )

        def owner_prog(owner: int):
            owned = context.owned(owner, aiv_units, ("vector", pipe, "mte3"))
            alloc = (
                (Allocation("mf_entry_ub", context.pool_names["ub"], ub_per_owner,
                            unit_group(owner, aiv_units, context.pool_units.get("ub", aiv_units))),)
                if "ub" in context.pool_names else ()
            )
            dma_w = context.bind(Work(resources={
                pipe: per_owner, "mte3": per_owner, "hbm": per_owner,
            }))
            return KernelProgram((
                Stage("ub_acquire", acquire=alloc),
                Stage("double_buffered_gather", dma_w, after=("ub_acquire",),
                      caps=per_owner_cap, placement=owned),
                Stage("ub_release", after=("double_buffered_gather",), free=alloc),
            ))

        return KernelProgram((
            Stage("gva_aperture_setup", latency=setup_ms),
            Stage("aiv_gather", after=("gva_aperture_setup",),
                  program=ParallelProgram(tuple(owner_prog(i) for i in range(active_owners)))),
        ))

    return OperatorPlan(work, program, metadata={
        "execution_class": "memfabric_acc_offload",
        "kernel_mode": "OffloadEntryGatherKernel",
        "rows": rows,
        "active_aiv_owners": active_owners,
        "ub_bytes_per_owner": ub_per_owner,
    })


def _build_lim_plan(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    rpipe = read_role(geometry.resource_units or {}, aiv=0)
    work = Work(
        matrix={op.dtype: op.flops} if op.flops else {},
        resources={
            "hbm": op.hbm_bytes,
            rpipe: op.mte2_bytes,
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
    if not op.dtype:
        raise ModelingError("sparse_kv_offload_op_dtype_required")
    dtype = op.dtype
    aic_pipe = read_role(geometry.resource_units or {})
    aiv_pipe = read_role(geometry.resource_units or {}, aiv=0)

    res: dict[str, float] = {
        "hbm": attn_hbm_bytes + miss_bytes,
        "mte3": float(op.mte3_bytes) + miss_bytes,
        "vector": float(op.vector_ops),
    }
    if aic_pipe == aiv_pipe:
        res[aic_pipe] = attn_hbm_bytes + miss_bytes
    else:
        res[aic_pipe] = attn_hbm_bytes
        res[aiv_pipe] = miss_bytes
    work = Work(matrix={dtype: op.flops} if op.flops else {}, resources=res)

    def program(context):
        c_pipe = read_role(context.resource_names)
        v_pipe = read_role(context.resource_names, aiv=0)
        aic_work = Work(
            matrix={dtype: op.flops} if op.flops else {},
            resources={"hbm": attn_hbm_bytes, c_pipe: attn_hbm_bytes,
                       "mte3": float(op.mte3_bytes), "vector": float(op.vector_ops)},
        )
        aic_prog = KernelProgram((Stage("sfa_aic", context.bind(aic_work)),))
        if miss_bytes <= 0:
            return aic_prog
        gva_bw_bytes_per_ms, setup_ms = _resolve_host_dma_channel(context, spec, "h2d")
        gva_name = context.resource_names.get(v_pipe)
        cap = (
            {gva_name: gva_bw_bytes_per_ms}
            if (gva_name and gva_bw_bytes_per_ms and gva_bw_bytes_per_ms > 0)
            else {}
        )
        aiv_prog = KernelProgram((
            Stage("gva_setup", latency=setup_ms),
            Stage("gva_h2d_gather", context.bind(Work(resources={v_pipe: miss_bytes})),
                  after=("gva_setup",), caps=cap),
            Stage("topk_buffer_write", context.bind(Work(resources={"hbm": miss_bytes, "mte3": miss_bytes})),
                  after=("gva_h2d_gather",)),
        ))
        return ParallelProgram((aiv_prog, aic_prog))

    return OperatorPlan(work, program, metadata={
        "execution_class": "mixed_aiv_aic_overlap",
        "miss_h2d_bytes": miss_bytes,
        "attn_hbm_bytes": attn_hbm_bytes,
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
    layer_kv_row_bytes = int((budget.get("kv_row_bytes_by_layer") or {}).get(plan.index, budget["kv_row_bytes"]))
    payload_bytes = float(valid_tokens * layer_kv_row_bytes)
    call_meta: dict[str, Any] = {
        "direction": "d2h", "pages": pages, "page_size": page_size,
        "payload_bytes": payload_bytes, "channel_bw_gbps": budget["d2h_effective_bw_gbps"],
    }
    if budget.get("d2h_alpha_us") is not None:
        call_meta["channel_alpha_us"] = budget["d2h_alpha_us"]
    op_id = b.work(
        "memfabric_kv_exchange_d2h", (store_id,), (pages, page_size, plan.cache_row_elements),
        read=payload_bytes, write=payload_bytes, family="kv_offload_exchange", dtype=b.model.kv_dtype,
        metadata=(
            ("kernel", "OffloadKVExchangeKernel"), ("implementation", IMPLEMENTATION),
            ("memfabric_call", call_meta),
            ("hardware_support", budget["hardware_support"]),
        ),
    )
    b.ops[-1] = replace(b.ops[-1], hbm_bytes=payload_bytes, recipe="memfabric_kv_exchange", stream="offload", source_boundary="kernel")
    return op_id


def lower_decode_offload(
    b: Any, *, plan: Any, views: tuple[Any, ...], store_id: str, topk_id: str | None, budget: Mapping[str, Any]
) -> tuple[str, str | None, dict[str, Any]]:
    """Lower Decode Sparse KV Offload (D2H new-token writeback + LIM + H2D miss copy)."""
    opts = budget["options"]
    valid_tokens = sum(v.q_len for v in views)
    kv_row_bytes = int((budget.get("kv_row_bytes_by_layer") or {}).get(plan.index, budget["kv_row_bytes"]))
    topk_buffer_size = int(budget["topk_buffer_size"])
    miss_rate = float(opts["miss_rate"])

    # 1. New decode tokens D2H writeback to shared MemFabric Host DRAM (2 entries/token: k_nope + k_pe).
    d2h_bytes = float(valid_tokens * kv_row_bytes)
    d2h_entries = max(1, 2 * valid_tokens)
    d2h_call: dict[str, Any] = {
        "direction": "d2h", "entries": d2h_entries, "tokens": valid_tokens,
        "payload_bytes": d2h_bytes, "channel_bw_gbps": budget["d2h_effective_bw_gbps"],
    }
    if budget.get("d2h_alpha_us") is not None:
        d2h_call["channel_alpha_us"] = budget["d2h_alpha_us"]
    d2h_id = b.work(
        "memfabric_sparse_copy_d2h", (store_id,), (valid_tokens, plan.cache_row_elements),
        read=d2h_bytes, write=d2h_bytes, family="kv_offload_d2h", dtype=b.model.kv_dtype,
        metadata=(
            ("kernel", "OffloadKVSparseCopyKernel" if d2h_entries >= KV_COPY_MIN_ENTRIES else "OffloadMoeSparseCopyKernel"),
            ("implementation", IMPLEMENTATION),
            ("memfabric_call", d2h_call),
            ("hardware_support", budget["hardware_support"]),
        ),
    )
    b.ops[-1] = replace(b.ops[-1], hbm_bytes=d2h_bytes, recipe="memfabric_sparse_copy_d2h", source_boundary="kernel")

    # 2. Classify request states (-3 short, -1 first long step, -2 steady long decode).
    short_reqs, long_reqs, total_miss_tokens = 0, 0, 0
    request_states = []
    for v in views:
        if v.context < topk_buffer_size:
            short_reqs += 1
            request_states.append(OFFLOAD_INACTIVE_REQUEST)
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
        lim_read = float(long_reqs * topk_width * (TOPK_INDEX_BYTES + DEVICE_SLOT_INDEX_BYTES + HOST_SLOT_INDEX_BYTES))
        lim_write = float(long_reqs * topk_width * (TOPK_INDEX_BYTES + OFFLOAD_ADDRESS_PAIR_BYTES))
        lim_vec = float(long_reqs * topk_width * LIMIT_SCAN_OPS_PER_ENTRY)
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
        h2d_call: dict[str, Any] = {
            "direction": "h2d", "entries": h2d_entries, "tokens": total_miss_tokens,
            "payload_bytes": miss_h2d_bytes, "channel_bw_gbps": budget["h2d_effective_bw_gbps"],
        }
        if budget.get("h2d_alpha_us") is not None:
            h2d_call["channel_alpha_us"] = budget["h2d_alpha_us"]
        h2d_id = b.work(
            "memfabric_sparse_copy_h2d", deps, (total_miss_tokens, plan.cache_row_elements),
            read=miss_h2d_bytes, write=miss_h2d_bytes, family="kv_offload_h2d", dtype=b.model.kv_dtype,
            metadata=(
                ("kernel", "OffloadKVSparseCopyKernel" if h2d_entries >= KV_COPY_MIN_ENTRIES else "OffloadMoeSparseCopyKernel"),
                ("implementation", IMPLEMENTATION),
                ("memfabric_call", h2d_call),
                ("request_states", tuple(request_states)), ("shared_indexer_reuse", is_shared_indexer_layer),
                ("skip_topk", is_shared_indexer_layer), ("hardware_support", budget["hardware_support"]),
            ),
        )
        b.ops[-1] = replace(b.ops[-1], hbm_bytes=miss_h2d_bytes, recipe="memfabric_sparse_copy_h2d", source_boundary="kernel")

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
        raise ModelingError(f"unsupported_sparse_kv_offload_recipe:{recipe_name}")
    call = dict(meta.get("memfabric_call", {}))
    call.setdefault("payload_bytes", float(op.hbm_bytes))
    call.setdefault("entries", max(1, int(op.local_shape[0]) * 2 if op.local_shape else 2))
    meta["memfabric_call"] = call
    meta["sparse_kv_offload_options"] = opts
    return replace(op, recipe=recipe_name, source_boundary="kernel", metadata=tuple(meta.items()))


__all__ = [
    "HISPARSE_ALIAS", "IMPLEMENTATION", "MEMFABRIC_SOURCE",
    "copy_sfa_pool_capacity", "lower_decode_offload", "lower_op", "lower_prefill_offload",
    "prepare_case", "recipes", "sparse_kv_offload_memory_budget", "strip_case",
]
