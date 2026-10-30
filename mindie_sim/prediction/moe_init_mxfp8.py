"""Finite source-phase MoE initialization with MXFP8 gather quantization.

[Role] Model the one-core sort/count/index and full-row MX gather branch.
[Invariants] All experts are local; valid assignments are not invented from a
  histogram. Source gather-copy quantizes a token once per owner partition,
  retains its outputs through all top-k stores, and publishes input scales.
[Timeouts & Locks] Bounded owner descriptors and two-row lazy windows; no I/O.
[Side-Effect Policy] Pure plans on the common Engine; no measurements or locks.
"""
from math import ceil

from mindie_sim.schema.common import ModelingError
from .engine import Allocation
from .instructions import PrimitiveInventory
from .mechanism import OperatorPlan, Work
from .moe_init_instructions import source_operations, diagnostics
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage

RECIPE = "ascend_moe_init_routing_v3_mxfp8"
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


def aligned(n, width=32):
    return ceil(n / width) * width


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
    """
    scales = width // 32
    # amax: abs mask + pair max + block max = W + W/2 +(W/2-W/32).
    # RoundScale: 2 initialization + 19 expressions per scale; ten live
    # full-register constants plus amax's abs mask (three unused constants).
    # Data: multiply, BF16 interleave, BF16->FP32, FP32 interleave, FP32->FP8.
    return (2 * width - scales) + 21 * scales + 11 * 128 + 5 * width


def build(op, geometry):
    cfg = dict(op.metadata)["moe_init_config"]
    tokens, topk, width, experts = (cfg[k] for k in ("tokens", "topk", "width", "experts"))
    assignments = tokens * topk
    units = geometry.resource_units.get("vector", 0)
    ub = geometry.buffers.get("ub", 0)
    available = ub - 65536  # host reserves SIMT DCache before every tiling
    if (min(tokens, topk, width, experts, units) < 1 or topk <= 3 or width % 256
            or assignments > min(available // 32 // 32 * 32, 32 * 255)):
        raise ModelingError("moe_init_mxfp8_requires_full_range_K_gt3_onecore_sort_full_256_columns")
    if any(role not in geometry.resource_units for role in ("mte2_aiv", "mte3")):
        raise ModelingError("moe_init_mxfp8_requires_AIV_read_write_pipes")
    per_owner = ceil(assignments / units)
    active = ceil(assignments / per_owner)
    scales = width // 32
    sort_rows = aligned(assignments)
    sort_input_rows = aligned(assignments, 8)
    max_indices = (available // 2 - (aligned(2 * width) + 2 * aligned(2 * scales)
                                    + aligned(scales) + 4 * aligned(width // 4))) // 4
    if max_indices < per_owner:
        raise ModelingError("moe_init_mxfp8_requires_single_index_chunk_and_full_column_tile")
    gather_ub = (65536 + 2 * (aligned(per_owner * 4) + aligned(width * 2)
                              + 4 * aligned(width // 4) + aligned(scales))
                 + 2 * aligned(scales * 2))
    count_ub = 65536 + aligned(per_owner * 4) + 2 * aligned(experts * 4) + aligned(experts * 8) + 32
    sort_ub = 65536 + 32 * sort_rows
    if max(sort_ub, count_ub, gather_ub) > ub:
        raise ModelingError("moe_init_mxfp8_source_UB_capacity")

    def memory(read=0, write=0, *, direct=0, vector=0):
        return Work(resources={"hbm": read + write + direct, "mte2_aiv": read,
                               "mte3": write, "vector": vector})
    init = memory(write=4 * experts, vector=experts)
    sort_in = memory(read=4 * assignments)
    sort_vector = Work(resources={"vector": sort_rows + 6 * sort_input_rows + 64
                                  + (sort_rows - assignments if assignments % 32 else 0)})
    sort_out = memory(write=8 * assignments)
    sort_ops, count_ops, inverse_ops = source_operations(assignments, per_owner, active, experts)
    sort_services = {key: value.partition(geometry.resource_units)[0] for key,value in sort_ops.items()}
    count_services = tuple(value.partition(geometry.resource_units)[0] for value in count_ops)
    inverse_services = tuple(value.partition(geometry.resource_units)[0] for value in inverse_ops)
    operation_work = Work()
    for value in (*sort_services.values(), *count_services, *inverse_services):
        operation_work += value
    count_parts = []
    for owner in range(active):
        rows = min(per_owner, assignments - owner * per_owner)
        # Count_mode, dropless, row_idx_type=GATHER: init output inverse IDs;
        # two SIMT passes each load current and adjacent IDs. No DMA alias.
        count_parts.append((
            memory(write=4 * rows, vector=rows + experts),
            memory(direct=4 * (4 * rows - 2)),
            memory(write=4 * experts),  # atomic-add payload; cache RMW unknown
        ))
    count_reduce = (memory(read=4 * experts),
                    Work(resources={"vector": experts}), memory(write=8 * experts + 4))
    # Dropless full-range COUNT_MODE: owner0 visits all INT32 UB counts and
    # accumulates them serially, then SetValue publishes one total-count cell.
    # These are source operations, not loop/branch/ISA instruction counts.
    primitive_counts=PrimitiveInventory({'scalar_ub_access':experts+1,
        'scalar_alu_issue':experts,'pipe_barrier':1})
    primitive_work,_=primitive_counts.partition(geometry.resource_units)
    inverse_parts = [memory(direct=8 * min(per_owner, assignments - owner * per_owner) + 4)
                     for owner in range(active)]
    # Every core reads the total count during RowIdxGather::Init, including
    # idle cores, before the active-owner guard.
    idle_index = memory(direct=4 * (units - active))
    quantized_rows = sum(sum(1 for _ in row_runs(assignments, topk, units, owner))
                         for owner in range(active))
    row_load, row_quant = memory(read=width * 2), Work(resources={"vector": quant_vector(width)})
    row_write = memory(write=width + scales)
    total = init + sort_in + sort_vector + sort_out + idle_index
    for parts in count_parts:
        for part in parts:
            total += part
    for part in (*count_reduce, *inverse_parts):
        total += part
    total += memory(read=4 * assignments)
    total += (row_load + row_quant).scaled(quantized_rows) + row_write.scaled(assignments)
    total += primitive_work + operation_work

    def program(context):
        def owned(owner):
            return context.owned(owner, units,
                ("vector", "mte2_aiv", "mte3",*primitive_work.resources,*operation_work.resources))

        def stage(name, work, owner, **kwargs):
            return Stage(name, context.bind(work), placement=owned(owner), **kwargs)

        sort_stages = [stage("zero_count_workspace", init, 0),
                       stage("read_ids", sort_in, 0, after=("zero_count_workspace",))]
        if any(work.resources for work in sort_services.values()):
            # Keep the known lane envelope once, split at the actual macro
            # boundary. Whole-sort services require size-specific provenance;
            # blocks are a normalization coordinate, not independent sorts.
            post = Work(resources={"vector": 2 * sort_input_rows})
            pre = Work(resources={"vector": sort_vector.resources["vector"] - post.resources["vector"]})
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
        counting = ParallelProgram(tuple(KernelProgram((
            stage("init", parts[0], owner),
            stage("SIMT_count_memory", parts[1] + count_services[owner], owner, after=("init",)),
            stage("atomic_count_payload", parts[2], owner, after=("SIMT_count_memory",)),
        )) for owner, parts in enumerate(count_parts)))
        scalar_stages=[]
        previous='cast_i64'
        if primitive_work.resources:
            if 'pipe_barrier' in primitive_work.resources:
                scalar_stages.append(stage('count_vector_barrier',
                    Work(resources={'pipe_barrier':1}),0,after=(previous,)))
                previous='count_vector_barrier'
            def scalar_iteration(index):
                chain=[];deps=()
                for name,role in (('get_count','scalar_ub_access'),('add_count','scalar_alu_issue')):
                    if role in primitive_work.resources:
                        chain.append(stage(name,Work(resources={role:1}),0,after=deps))
                        deps=(name,)
                return tuple(chain)
            if any(role in primitive_work.resources for role in ('scalar_ub_access','scalar_alu_issue')):
                scalar_stages.append(Stage('scalar_count_sum',after=(previous,),
                    program=PipelineProgram(experts,1,scalar_iteration)))
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
        inversion = ParallelProgram(tuple(KernelProgram((stage("SIMT_invert_memory", work + inverse_services[owner], owner),))
                                           for owner, work in enumerate(inverse_parts)))

        def gather(owner):
            start = owner * per_owner
            end = min(assignments, start + per_owner)
            first, last = start // topk, (end - 1) // topk
            def row(index):
                token = first + index
                copies = min(end, (token + 1) * topk) - max(start, token * topk)
                return (stage("read", row_load, owner, previous=("read",)),
                        stage("MX_quantize", row_quant, owner, after=("read",), previous=("MX_quantize",)),
                        stage("publish_data_and_scales", row_write.scaled(copies), owner,
                              after=("MX_quantize",), previous=("publish_data_and_scales",)))
            return KernelProgram((
                stage("read_inverse_ids", memory(read=4 * (end - start)), owner),
                Stage("rows", after=("read_inverse_ids",), program=PipelineProgram(last - first + 1, 2, row)),
            ))
        phases = KernelProgram((
            Stage("sort", program=sorting),
            Stage("count", after=("sort",), program=counting),
            Stage("reduce_counts", after=("count",), program=reduction),
            Stage("inverse_ids", after=("reduce_counts",), program=inversion),
            Stage("idle_total_reads", context.bind(idle_index), after=("reduce_counts",)),
            Stage("gather_quantize", after=("inverse_ids", "idle_total_reads"),
                  program=ParallelProgram(tuple(gather(owner) for owner in range(active)))),
        ))
        # InitBuffer runs on every launched AIV, including non-sort owners.
        # Keep the kernel's maximum arena live once; phases reuse it.
        allocation = (Allocation("source_ub", context.pool_names["ub"],
                                 max(sort_ub, count_ub, gather_ub),
                                 tuple(range(context.pool_units.get("ub", units)))),)
        return KernelProgram((Stage("reserve", acquire=allocation),
                              Stage("source", after=("reserve",), program=phases),
                              Stage("release", after=("source",), free=allocation)))
    return OperatorPlan(total, program, metadata={
        "execution_class": "vector", "source": SOURCE, "active_aiv": active,
        "assignments": assignments, "quantized_input_rows": quantized_rows,
        "output_data_bytes": assignments * width, "output_scale_bytes": assignments * scales,
        "source_branch": "onecore_sort_GATHER_count_mode_full_range_useGatherCopy",
        "sort_blocks32": sort_rows // 32,
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
                     "SyncAll_flag_latency", "atomic_count_cache_RMW", "UB_register_ports"),
        "schedule_limitations": "two_whole_row_slots_conservative_vs_separate_input_output_queues",
    })
