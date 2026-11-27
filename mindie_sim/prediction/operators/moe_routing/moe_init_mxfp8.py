"""Finite source-phase MoE initialization with MXFP8 gather quantization.

[Role] Model the one-core sort/count/index and full-row MX gather branch.
[Invariants] All experts are local; valid assignments are not invented from a
  histogram. Source gather-copy quantizes a token once per owner partition,
  retains its outputs through all top-k stores, and publishes input scales.
[Timeouts & Locks] Bounded owner descriptors and two-row lazy windows; no I/O.
[Side-Effect Policy] Pure plans on the common Engine; no measurements or locks.
"""
from math import ceil
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.instructions import PrimitiveInventory
from mindie_sim.prediction.mechanism import OperatorPlan, Work
from .moe_init_instructions import INDEX_BYTES, SORT_BLOCK_ROWS, diagnostics, source_operations
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_gather_mxfp8_quant.h",
        source_sha256="34b49b8e82da39bfe59dc4e80c46fc3eea96f93ae16ade54b913c0cfea5df813",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_sort_one_core_arch35.h",
        source_sha256="920534d82fd33eb308a4b3493e978b598e5bfba1da51f6700794a12197e94709",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_expert_tokens_count_arch35.h",
        source_sha256="d0dfd8f977596357f37fddd1ea871789657b128e9c0d4bda03badbbd99ce7077",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_row_idx_gather_arch35.h",
        source_sha256="cfc375ee5bb0a5f9ddb845ba349ea09e3ed64ada80c3e947e6ff3fc25ce6d353",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="moe/moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.cpp",
        source_sha256="1507153599d4ccae0cc056dc5b1e4f4849c02cf97893736f3d6913a43b52385d",
    ),
)

RECIPE = "ascend_moe_init_routing_v3_mxfp8"
MX_SCALE_GROUP_SIZE = 32
MAX_COUNT_MODE_EXPERTS = 10240
_ARCH35_IDS = frozenset({"npu_arch_3510", "dav_3510"})
_BF16_BYTES = int(dtype_bytes("bf16"))
_PUBLISHED_COUNT_BYTES = int(dtype_bytes("int64"))
_UB_ALIGNMENT_BYTES = 32
_SIMT_CACHE_BYTES = 64 * 1024
_SORT_INPUT_ALIGNMENT_ROWS = 8
_SORT_UB_LOAD_LANES = 64
_SORT_UB_BYTES_PER_ROW = 32
_SORT_MAX_BLOCKS = 255
_SORT_LANE_OPS_PER_INPUT_ROW = 6
_SORT_FIXED_LANE_OPS = 64
_SORT_OUTPUT_FIELDS_PER_ROW = 2
_MX_SCALE_OPS_PER_GROUP = 21
_QUANT_CONSTANT_REGISTERS = 11
_CONSTANT_REGISTER_LANES = 128
_QUANT_VF_WIDTH = 2 * _CONSTANT_REGISTER_LANES
_QUANT_DATA_OPS_PER_ELEMENT = 5
_QUANT_PACKED_TEMP_BUFFERS = 4
_QUANT_BF16_UB_READ_PASSES = 2
_QUANT_SCALE_INTERMEDIATE_BUFFERS = 2
_QUANT_SCALE_INTERMEDIATE_BYTES = 2
_GATHER_COPY_TOPK_THRESHOLD = 3

SOURCE = {
    "repository": "https://gitcode.com/cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "root": "moe/moe_init_routing_v3/",
    "sha256": {
        "op_kernel/arch35/moe_v3_gather_mxfp8_quant.h":
            "34b49b8e82da39bfe59dc4e80c46fc3eea96f93ae16ade54b913c0cfea5df813",
        "op_kernel/arch35/moe_v3_sort_one_core_arch35.h":
            "920534d82fd33eb308a4b3493e978b598e5bfba1da51f6700794a12197e94709",
        "op_kernel/arch35/moe_v3_expert_tokens_count_arch35.h":
            "d0dfd8f977596357f37fddd1ea871789657b128e9c0d4bda03badbbd99ce7077",
        "op_kernel/arch35/moe_v3_row_idx_gather_arch35.h":
            "cfc375ee5bb0a5f9ddb845ba349ea09e3ed64ada80c3e947e6ff3fc25ce6d353",
        "op_host/moe_init_routing_v3_tiling_arch35.cpp":
            "1507153599d4ccae0cc056dc5b1e4f4849c02cf97893736f3d6913a43b52385d"},
    "capture_binary_identity": "not_established",
}


def aligned(n, width=_UB_ALIGNMENT_BYTES):
    return ceil(n / width) * width


def mxfp8_scale_cols(width: int) -> int:
    """CANN ``MoeInitRoutingV3`` quant mode 3 scale width: ``CeilAlign(CeilDiv(width, 32), 2)``."""
    valid_groups = ceil(width / MX_SCALE_GROUP_SIZE)
    return ceil(valid_groups / 2) * 2


def row_runs(assignments, topk, units, owner):
    """Actual source assignment intervals; a token split across owners reloads."""
    per_owner = ceil(assignments / units)
    start, end = owner * per_owner, min(assignments, (owner + 1) * per_owner)
    for token in range(start // topk, ceil(end / topk)):
        yield token, min(end, (token + 1) * topk) - max(start, token * topk)


def quant_vector(width):
    """Known lane expressions in the BF16, unclamped E4M3 source branch.

    Sort/SIMT/scalar instruction throughput and register-port issue are kept
    separate in diagnostics; this is the existing elementary lane envelope.
    CANN executes unmasked 2-register (256-BF16) vector loops in both the amax
    and data passes (``moe_v3_gather_mxfp8_quant.h:642``).
    """
    vf_width = aligned(width, _QUANT_VF_WIDTH)
    vf_scales = vf_width // MX_SCALE_GROUP_SIZE
    scales = mxfp8_scale_cols(width)
    # amax: abs mask + pair max + block max = W_vf + W_vf/2 + (W_vf/2 - W_vf/32).
    # RoundScale: 2 initialization + 19 expressions per scale; ten live
    # full-register constants plus amax's abs mask (three unused constants).
    # Data: multiply, BF16 interleave, BF16->FP32, FP32 interleave, FP32->FP8.
    return ((2 * vf_width - vf_scales) + _MX_SCALE_OPS_PER_GROUP * scales
            + _QUANT_CONSTANT_REGISTERS * _CONSTANT_REGISTER_LANES
            + _QUANT_DATA_OPS_PER_ELEMENT * vf_width)


def _sort_program(stage, init, assignment_ids_read, sort_services, sort_input_rows, sort_vector, sort_out):
    sort_stages = [stage("zero_count_workspace", init, 0),
                   stage("read_ids", assignment_ids_read, 0, after=("zero_count_workspace",))]
    if any(work.resources for work in sort_services.values()):
        # Keep the known lane envelope once, split at the actual macro
        # boundary. Whole-sort services require size-specific provenance;
        # blocks are a normalization coordinate, not independent sorts.
        post = Work(resources={"vector": 2 * sort_input_rows})
        pre = Work(resources={
            **sort_vector.resources,
            "vector": sort_vector.resources["vector"] - post.resources["vector"],
        })
        sort_stages.append(stage("sort_pre_lane_work", pre, 0, after=("read_ids",)))
        prior = "sort_pre_lane_work"
        for name, work in sort_services.items():
            if work.resources:
                key = "source_" + name + "_macro"
                sort_stages.append(stage(key, work, 0, after=(prior,)))
                prior = key
        sort_stages.append(stage("sort_post_lane_work", post, 0, after=(prior,)))
        prior = "sort_post_lane_work"
    else:
        sort_stages.append(stage("sort_known_lane_work", sort_vector, 0, after=("read_ids",)))
        prior = "sort_known_lane_work"
    sort_stages.append(stage("publish_sorted_ids", sort_out, 0, after=(prior,)))
    sorting = KernelProgram(tuple(sort_stages))
    return sorting


def _count_reduction_program(stage, primitive_work, experts, count_reduce):
    scalar_stages=[]
    previous='cast_i64'
    if primitive_work.resources:
        if 'pipe_barrier' in primitive_work.resources:
            scalar_stages.append(stage('count_vector_barrier',
                Work(resources={'pipe_barrier':1}),0,after=(previous,)))
            previous='count_vector_barrier'
        def scalar_iteration(index):
            chain = []
            deps = ()
            for name,role in (('get_count','scalar_ub_access'),('add_count','scalar_alu_issue')):
                if role in primitive_work.resources:
                    chain.append(stage(name,Work(resources={role:1}),0,after=deps))
                    deps=(name,)
            return tuple(chain)
        if any(role in primitive_work.resources for role in ('scalar_ub_access','scalar_alu_issue')):
            scalar_stages.append(Stage('scalar_count_sum',after=(previous,),
                program=PipelineProgram(experts,1,scalar_iteration,interior_fold=False)))
            previous='scalar_count_sum'
        if 'scalar_ub_access' in primitive_work.resources:
            scalar_stages.append(stage('store_total_count',
                Work(resources={'scalar_ub_access':1}),0,after=(previous,)))
            previous='store_total_count'
    reduction = KernelProgram((
        stage("read_counts", count_reduce[0], 0),
        stage("cast_i64", count_reduce[1], 0, after=("read_counts",)),
        *scalar_stages,
        stage("publish_counts_and_total", count_reduce[2], 0, after=(previous,)),
    ))
    return reduction


def _gather_quantize_owner(
    owner, stage, memory, assignments, per_owner, topk, use_gather_copy,
    row_load, row_quant, row_write,
):
    start = owner * per_owner
    end = min(assignments, start + per_owner)
    if use_gather_copy:
        first, last = start // topk, (end - 1) // topk
        row_count = last - first + 1
        def row(index):
            token = first + index
            copies = min(end, (token + 1) * topk) - max(start, token * topk)
            return (stage("read", row_load, owner, previous=("read",)),
                    stage("MX_quantize", row_quant, owner, after=("read",), previous=("MX_quantize",)),
                    stage("publish_data_and_scales", row_write.scaled(copies), owner,
                          after=("MX_quantize",), previous=("publish_data_and_scales",)))
    else:
        row_count = end - start
        def row(index):
            return (stage("read", row_load, owner, previous=("read",)),
                    stage("MX_quantize", row_quant, owner, after=("read",), previous=("MX_quantize",)),
                    stage("publish_data_and_scales", row_write, owner,
                          after=("MX_quantize",), previous=("publish_data_and_scales",)))
    return KernelProgram((
        stage("read_inverse_ids", memory(read=INDEX_BYTES * (end - start)), owner),
        Stage("rows", after=("read_inverse_ids",), program=PipelineProgram(row_count, 2, row)),
    ))



def _validate_arch35_mxfp8(geometry) -> None:
    """Require Arch35 (``npu_arch_3510``) and native E4M3 quantization support."""
    datapath = getattr(geometry, "datapath", None)
    ca = (
        getattr(datapath, "compute_arch", None)
        if datapath is not None
        else getattr(geometry, "compute_arch", None)
    )
    arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
    if arch_id not in _ARCH35_IDS:
        raise ModelingError(
            f"moe_init_mxfp8_requires_arch35_npu_arch_3510:{arch_id or 'unknown'}"
        )
    declared = ca if hasattr(ca, "supports_native") else getattr(ca, "declared", None)
    if declared is not None and hasattr(declared, "supports_native") and not declared.supports_native("matrix", "fp8_e4m3"):
        raise ModelingError("moe_init_mxfp8_requires_native_fp8_e4m3")


def _resolve_stream_aiv(geometry) -> tuple[int, int, tuple[int, ...], str, tuple[int, ...]]:
    """Resolve ``(die_units, effective_units, aiv_pool, isolation, shared_pool)``."""
    die_units = int(geometry.resource_units.get("vector", 0))
    owners = getattr(geometry, "stream_owners", None)
    if owners is None:
        return die_units, die_units, tuple(range(max(0, die_units))), "full", ()
    aiv_limit = int(owners.aiv_limit)
    if aiv_limit < 1:
        raise ModelingError("moe_init_mxfp8_requires_positive_stream_aiv_budget")
    units = min(die_units, aiv_limit)
    if owners.isolation == "disjoint":
        pool = owners.aiv if owners.aiv else tuple(range(units))
        units = min(units, len(pool))
        return die_units, units, tuple(pool[:units]), "disjoint", ()
    shared_pool = tuple(owners.pool_aiv) if owners.pool_aiv else ()
    if shared_pool:
        units = min(units, len(shared_pool))
        pool = shared_pool[:units]
    else:
        pool = tuple(range(units))
    return die_units, units, tuple(pool), "shared_cap", shared_pool


def _build_moe_init_program(
    context, *, die_units, units, aiv_pool, isolation, shared_pool, active, assignments,
    per_owner, topk, experts, use_gather_copy, memory, init, assignment_ids_read,
    sort_services, sort_input_rows, sort_vector, sort_ub_work, sort_out, count_parts,
    count_services, primitive_work, operation_work, count_reduce, inverse_parts,
    inverse_services, idle_index, row_load, row_quant, row_write, ub_bytes,
):
    def owned(owner):
        phys_owner = aiv_pool[owner] if owner < len(aiv_pool) else owner
        return context.owned(phys_owner, die_units, (
            "vector", "mte2_aiv", "mte3", *sort_ub_work.resources,
            *primitive_work.resources, *operation_work.resources))

    def stage(name, work, owner, **kwargs):
        return Stage(name, context.bind(work), placement=owned(owner), **kwargs)

    sorting = _sort_program(
        stage, init, assignment_ids_read, sort_services, sort_input_rows, sort_vector, sort_out)
    counting = ParallelProgram(tuple(KernelProgram((
        stage("init", parts[0], owner),
        stage("SIMT_count_memory", parts[1] + count_services[owner], owner, after=("init",)),
        stage("atomic_count_payload", parts[2], owner, after=("SIMT_count_memory",)),
    )) for owner, parts in enumerate(count_parts)))
    reduction = _count_reduction_program(stage, primitive_work, experts, count_reduce)
    if use_gather_copy:
        inversion = ParallelProgram(tuple(
            KernelProgram((stage("SIMT_invert_memory", work + inverse_services[owner], owner),))
            for owner, work in enumerate(inverse_parts)))
        tail_stages = (
            Stage("inverse_ids", after=("reduce_counts",), program=inversion),
            Stage("idle_total_reads", context.bind(idle_index), after=("reduce_counts",)),
            Stage("gather_quantize", after=("inverse_ids", "idle_total_reads"),
                  program=ParallelProgram(tuple(
                      _gather_quantize_owner(
                          owner, stage, memory, assignments, per_owner, topk,
                          use_gather_copy, row_load, row_quant, row_write,
                      ) for owner in range(active)))),
        )
    else:
        # ScatterCopyExpandedXandMXQuant (K <= 3) has no cross-core SyncAll after
        # RowIdxGather: each AIV reads the INT32 total count immediately after its
        # own inversion while idle AIVs issue their two count reads after reduction.
        owner_scatter_read = context.bind(memory(direct=INDEX_BYTES))
        scatter_owners = ParallelProgram(tuple(
            KernelProgram((
                stage("SIMT_invert_memory", inverse_parts[owner] + inverse_services[owner], owner),
                Stage("scatter_total_count_read", owner_scatter_read, after=("SIMT_invert_memory",)),
                Stage("gather_quantize", after=("scatter_total_count_read",),
                      program=_gather_quantize_owner(
                          owner, stage, memory, assignments, per_owner, topk,
                          use_gather_copy, row_load, row_quant, row_write,
                      )),
            )) for owner in range(active)))
        idle_scatter = memory(direct=INDEX_BYTES * (units - active))
        tail_stages = (
            Stage("inverse_and_scatter_quantize", after=("reduce_counts",), program=scatter_owners),
            Stage("idle_total_reads", context.bind(idle_index), after=("reduce_counts",)),
            *((Stage("idle_scatter_total_reads", context.bind(idle_scatter), after=("reduce_counts",)),)
              if idle_scatter.resources else ()),
        )
    phases = KernelProgram((
        Stage("sort", program=sorting),
        Stage("count", after=("sort",), program=counting),
        Stage("reduce_counts", after=("count",), program=reduction),
        *tail_stages,
    ))
    if isolation == "disjoint":
        ub_alloc_bytes, ub_owners = ub_bytes, aiv_pool
    elif isolation == "shared_cap":
        # Shared-cap lanes do not own fixed physical AIV indices; spread the
        # lane's total UB footprint evenly across its shared AIV pool so
        # independent launches conserve aggregate UB without pinning unit 0.
        ub_owners = shared_pool or tuple(range(context.pool_units.get("ub", die_units)))
        ub_alloc_bytes = ub_bytes * (float(units) / float(max(1, len(ub_owners))))
    else:
        ub_alloc_bytes = ub_bytes
        ub_owners = tuple(range(context.pool_units.get("ub", units)))
    allocation = (Allocation("source_ub", context.pool_names["ub"], ub_alloc_bytes, ub_owners),)
    return KernelProgram((Stage("reserve", acquire=allocation),
                          Stage("source", after=("reserve",), program=phases),
                          Stage("release", after=("source",), free=allocation)))


def _ub_traffic_contract(assignments, quantized_rows, width, scales, declared_roles):
    """Bind aligned B32 sort UB accesses to UB ports and record BF16/UINT16 UB traffic."""
    sort_b32_read = aligned(assignments, _SORT_UB_LOAD_LANES) * INDEX_BYTES
    sort_b32_write = aligned(assignments, _SORT_INPUT_ALIGNMENT_ROWS) * INDEX_BYTES
    sort_ub_counts = PrimitiveInventory({
        "ub_read_bytes": sort_b32_read,
        "ub_write_bytes": sort_b32_write,
    })
    sort_ub_work, _ = sort_ub_counts.partition(declared_roles)
    vf_width = aligned(width, _QUANT_VF_WIDTH)
    quant_bf16_read = _QUANT_BF16_UB_READ_PASSES * _BF16_BYTES * vf_width * quantized_rows
    quant_output_write = (width + scales) * quantized_rows
    quant_scale_intermediate = (
        _QUANT_SCALE_INTERMEDIATE_BUFFERS * _QUANT_SCALE_INTERMEDIATE_BYTES * scales * quantized_rows
    )
    metadata = {
        "sort_b32_read_bytes": sort_b32_read,
        "sort_b32_write_bytes": sort_b32_write,
        "quant_bf16_read_bytes": quant_bf16_read,
        "quant_output_write_bytes": quant_output_write,
        "quant_scale_intermediate_read_bytes": quant_scale_intermediate,
        "quant_scale_intermediate_write_bytes": quant_scale_intermediate,
        "total_ub_read_bytes": sort_b32_read + quant_bf16_read + quant_scale_intermediate,
        "total_ub_write_bytes": sort_b32_write + quant_output_write + quant_scale_intermediate,
        "b32_port_services": sort_ub_counts.to_dict(declared_roles),
        "bf16_read_boundary": "amax_and_quant_two_pass_bf16_and_u16_scale_ub_unpriced_on_b32_port",
    }
    return sort_ub_work, metadata


def build(op, geometry):
    _validate_arch35_mxfp8(geometry)
    cfg = dict(op.metadata)["moe_init_config"]
    tokens, topk, width, experts = (cfg[k] for k in ("tokens", "topk", "width", "experts"))
    if experts > MAX_COUNT_MODE_EXPERTS:
        raise ModelingError(
            f"moe_init_mxfp8_count_mode_experts_exceed_limit:{experts}:{MAX_COUNT_MODE_EXPERTS}"
        )
    assignments = tokens * topk
    die_units, units, aiv_pool, isolation, shared_pool = _resolve_stream_aiv(geometry)
    ub = geometry.buffers.get("ub", 0)
    available = ub - _SIMT_CACHE_BYTES
    sort_capacity_blocks = min(available // _SORT_UB_BYTES_PER_ROW // SORT_BLOCK_ROWS,
                               _SORT_MAX_BLOCKS)
    if (min(tokens, topk, width, experts, units) < 1
            or assignments > sort_capacity_blocks * SORT_BLOCK_ROWS):
        raise ModelingError("moe_init_mxfp8_requires_full_range_K_gt3_onecore_sort")
    if any(role not in geometry.resource_units for role in ("mte2_aiv", "mte3")):
        raise ModelingError("moe_init_mxfp8_requires_AIV_read_write_pipes")
    per_owner = ceil(assignments / units)
    active = ceil(assignments / per_owner)
    scales = mxfp8_scale_cols(width)
    sort_rows = aligned(assignments, SORT_BLOCK_ROWS)
    sort_input_rows = aligned(assignments, _SORT_INPUT_ALIGNMENT_ROWS)
    packed_temp_bytes = _QUANT_PACKED_TEMP_BUFFERS * aligned(ceil(width / _QUANT_PACKED_TEMP_BUFFERS))
    gather_row_bytes = aligned(_BF16_BYTES * width) + aligned(scales) + packed_temp_bytes
    max_indices = (available // 2 - (gather_row_bytes + 2 * aligned(2 * scales))) // INDEX_BYTES
    if max_indices < per_owner:
        raise ModelingError("moe_init_mxfp8_requires_single_index_chunk_and_full_column_tile")
    gather_ub = (_SIMT_CACHE_BYTES + 2 * (aligned(per_owner * INDEX_BYTES) + gather_row_bytes)
                 + 2 * aligned(2 * scales))
    count_ub = (_SIMT_CACHE_BYTES + aligned(per_owner * INDEX_BYTES)
                + 2 * aligned(experts * INDEX_BYTES) + aligned(experts * _PUBLISHED_COUNT_BYTES)
                + _UB_ALIGNMENT_BYTES)
    sort_ub = _SIMT_CACHE_BYTES + _SORT_UB_BYTES_PER_ROW * sort_rows
    if max(sort_ub, count_ub, gather_ub) > ub:
        raise ModelingError("moe_init_mxfp8_source_UB_capacity")

    def memory(read=0, write=0, *, direct=0, vector=0):
        return Work(resources={"hbm": read + write + direct, "mte2_aiv": read,
                               "mte3": write, "vector": vector})
    init = memory(write=INDEX_BYTES * experts, vector=experts)
    assignment_ids_read = memory(read=INDEX_BYTES * assignments)
    sort_vector = Work(resources={
        "vector": sort_rows + _SORT_LANE_OPS_PER_INPUT_ROW * sort_input_rows
        + _SORT_FIXED_LANE_OPS + (sort_rows - assignments),
    })
    sort_out = memory(write=_SORT_OUTPUT_FIELDS_PER_ROW * INDEX_BYTES * assignments)
    sort_ops, count_ops, inverse_ops = source_operations(assignments, per_owner, active, experts)
    sort_services = {key: value.partition(geometry.resource_units)[0] for key,value in sort_ops.items()}
    count_services = tuple(value.partition(geometry.resource_units)[0] for value in count_ops)
    inverse_services = tuple(value.partition(geometry.resource_units)[0] for value in inverse_ops)
    operation_work = Work()
    for value in (*sort_services.values(), *count_services, *inverse_services):
        operation_work += value
    count_parts = []
    for owner, count_op in enumerate(count_ops):
        rows = min(per_owner, assignments - owner * per_owner)
        # Count_mode, dropless, row_idx_type=GATHER: init output inverse IDs;
        # two SIMT passes each load current and adjacent IDs. No DMA alias.
        count_parts.append((
            memory(write=INDEX_BYTES * rows, vector=rows + experts),
            memory(direct=INDEX_BYTES * count_op.counts["source_simt_gm_load32"]),
            memory(write=INDEX_BYTES * experts),  # atomic-add payload; cache RMW unknown
        ))
    count_reduce = (memory(read=INDEX_BYTES * experts),
                    Work(resources={"vector": experts}),
                    memory(write=_PUBLISHED_COUNT_BYTES * experts + INDEX_BYTES))
    # Dropless full-range COUNT_MODE: owner0 visits all INT32 UB counts and
    # accumulates them serially, then SetValue publishes one total-count cell.
    # These are source operations, not loop/branch/ISA instruction counts.
    primitive_counts=PrimitiveInventory({'scalar_ub_access':experts+1,
        'scalar_alu_issue':experts,'pipe_barrier':1})
    primitive_work,_=primitive_counts.partition(geometry.resource_units)
    inverse_parts = [
        memory(direct=INDEX_BYTES * (inv.counts["source_simt_gm_load32"]
                                     + inv.counts["source_simt_gm_store32"] + 1))
        for inv in inverse_ops
    ]
    # Every core reads the total count during RowIdxGather::Init, including
    # idle cores, before the active-owner guard.
    idle_index = memory(direct=INDEX_BYTES * (units - active))
    # Source tiling sets useGatherCopy = ((expertEnd - expertStart) * k > expertNum * 3).
    # For full expert range (expertEnd - expertStart == expertNum), useGatherCopy is (topk > 3).
    # When topk <= 3, ScatterCopyExpandedXandMXQuant loads and quantizes each sorted
    # assignment row individually without cross-topk input row deduplication, and
    # every launched AIV reads the INT32 total count prior to the active-core guard.
    use_gather_copy = topk > _GATHER_COPY_TOPK_THRESHOLD
    scatter_count_read = Work() if use_gather_copy else memory(direct=INDEX_BYTES * units)
    quantized_rows = (
        sum(sum(1 for _ in row_runs(assignments, topk, units, owner)) for owner in range(active))
        if use_gather_copy else assignments
    )
    sort_ub_work, ub_traffic = _ub_traffic_contract(
        assignments, quantized_rows, width, scales, geometry.resource_units,
    )
    sort_vector = sort_vector + sort_ub_work
    row_load, row_quant = memory(read=width * _BF16_BYTES), Work(resources={"vector": quant_vector(width)})
    row_write = memory(write=width + scales)
    total = init + assignment_ids_read + sort_vector + sort_out + idle_index + scatter_count_read
    for parts in count_parts:
        for part in parts:
            total += part
    for part in (*count_reduce, *inverse_parts):
        total += part
    total += assignment_ids_read  # The gather phase reads one inverse ID per assignment.
    total += (row_load + row_quant).scaled(quantized_rows) + row_write.scaled(assignments)
    total += primitive_work + operation_work

    ub_bytes = max(sort_ub, count_ub, gather_ub)

    def program(context):
        return _build_moe_init_program(
            context, die_units=die_units, units=units, aiv_pool=aiv_pool,
            isolation=isolation, shared_pool=shared_pool, active=active,
            assignments=assignments, per_owner=per_owner, topk=topk, experts=experts,
            use_gather_copy=use_gather_copy, memory=memory, init=init,
            assignment_ids_read=assignment_ids_read, sort_services=sort_services,
            sort_input_rows=sort_input_rows, sort_vector=sort_vector,
            sort_ub_work=sort_ub_work, sort_out=sort_out, count_parts=count_parts,
            count_services=count_services, primitive_work=primitive_work,
            operation_work=operation_work, count_reduce=count_reduce,
            inverse_parts=inverse_parts, inverse_services=inverse_services,
            idle_index=idle_index, row_load=row_load, row_quant=row_quant,
            row_write=row_write, ub_bytes=ub_bytes)
    source_branch = (
        "onecore_sort_GATHER_count_mode_full_range_useGatherCopy"
        if use_gather_copy else "onecore_sort_GATHER_count_mode_full_range_scatterCopy"
    )
    return OperatorPlan(total, program, metadata={
        "execution_class": "vector", "source": SOURCE, "active_aiv": active,
        "assignments": assignments, "quantized_input_rows": quantized_rows,
        "output_data_bytes": assignments * width, "output_scale_bytes": assignments * scales,
        "source_branch": source_branch,
        "sort_blocks32": sort_rows // SORT_BLOCK_ROWS,
        "ub_traffic": ub_traffic,
        "primitive_services":primitive_counts.to_dict(geometry.resource_units),
        "source_operation_services": diagnostics(sort_ops,count_ops,inverse_ops,
            experts=experts,declared_roles=geometry.resource_units),
        "primitive_count_boundary":{
            'owner':0,'source':'expertCountCompute COUNT_MODE:309-338',
            'scalar_ub_access':{'GetValue_INT32':experts,'SetValue_INT32':1},
            'scalar_alu_issue':{'source_total_accumulation_adds':experts},
            'pipe_barrier':{'source_PIPE_V_calls':1},
            'excluded':'unknown compiled loop/branch/address ISA and barrier flag expansion'},
        "unpriced": ("Sort_Concat_Extract_specialized_issue", "SIMT_scalar_address_control",
                     "SyncAll_flag_latency", "atomic_count_cache_RMW", "BF16_UB_register_ports"),
        "schedule_limitations": "two_whole_row_slots_conservative_vs_separate_input_output_queues",
    })
