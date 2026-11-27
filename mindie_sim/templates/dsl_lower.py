"""Executable SRAM, GM and synchronization schedules for TileLang/AscendC DSL ABIs.

[Role] Lower each admitted K3 specialization onto shared engine stages.
[Invariants] GM scratch remains visible in the ledger. Direct routes carry
  local copy work and credit dependencies independently of GM transactions.
[Timeouts & Locks] Finite task schedules without waits on external state.
[Side-Effect Policy] Uses CPU records and the caller's optional admission view.
"""
from __future__ import annotations

from collections import Counter

from mindie_sim.prediction.program.stages import Stage
from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import DEFAULT_INST_BLOCK_DIM
from mindie_sim.schema.common import ModelingError

from .contracts import SemanticWorkItem, ceil_div
from .dsl_catalog import (
    ATTN_RES, ATTN_RES_VL, BF16_BYTES, COMMIT_KDA, FLASH_KDA, FP32_BYTES, INT32_BYTES,
    KDA_CHUNK, KDA_CONV_HISTORY_KEEP, KDA_CONV_STREAMS, KDA_CONV_TAPS, KDA_HEAD_DIM, KV_ROW_BYTES_CMP, KV_ROW_BYTES_ORI,
    MEGA_KDA, MLA_CACHE_NZ_LANE_BYTES, MLA_COLUMN_TILE, MLA_PROLOG, MLA_QA_PARTIALS, MLA_QUANTIZED_QUERY,
    MLA_WKB_N_CHUNKS, MQSMLA, MQSMLA_HEADS, MQSMLA_KV_DEPTH, MQSMLA_KV_TILE, QSLI,
    QSLI_CANDIDATE_BLOCK_SIZE, QSLI_CANDIDATE_CAPACITY, QSLI_HISTOGRAM_BINS, QSLI_MATRIX_DTYPE,
    QSLI_METADATA_BYTES, QSLI_PACKED_D, QSLI_SCALE_ROW_BYTES, QSLI_STAGING_DEPTH, QSLI_TILE_M, QSLI_TILE_N,
    QSLI_TRUNK, QkvQuantMode, commit_rows, describe_dsl_kernel,
    dsl_problem, flash_kda_lane_tasks, mega_conv_stride_elems, mega_workspace_sizes,
    mla_projection_widths, mla_workspaces, mqsmla_tile_m, mqsmla_tile_partitions, qsli_workspace_sizes,
    replay_pool_capacity, replay_pool_strides,
)
from .fuse_region import Lowering, ring_credit

DSL_SUBBLOCK_LIMIT = 2
KDA_STAGED_INVERSE_PRODUCTS = 5
KDA_PERSISTENT_INVERSE_PRODUCTS = 4
KDA_INPUT_VECTORS = 4
KDA_STATE_UPDATE_OPS = 3
KDA_RECURRENT_STATE_OPS = 7
KDA_NORM_VECTORS = 2
MEGA_OUT_TILE_N = 256
MEGA_OUT_WAVE_ROWS = 128
MLA_CACHE_INDEX_BYTES = 8
MLA_FRONT_ACTUAL_ROWS_MAX = 64
MLA_WKB_M_TAIL_ROWS = 16
MLA_SINGLE_AIV_STRIDE_SPLITS = 2
ATTN_RES_MERGE_OPS = 5
ATTN_RES_NORM_OPS = 4
ATTN_RES_SCORE_OPS = 5
QSLI_BLOCKS_PER_TILE = QSLI_TILE_N // QSLI_CANDIDATE_BLOCK_SIZE
QSLI_BLOCKS_PER_AIV = QSLI_BLOCKS_PER_TILE // DSL_SUBBLOCK_LIMIT
MQSMLA_KV_HALF = MQSMLA_KV_TILE // DSL_SUBBLOCK_LIMIT
MQSMLA_KV_SUB_SLICE = MQSMLA_KV_HALF // DSL_SUBBLOCK_LIMIT
MQSMLA_FD_SLOTS_PER_CORE = 2
MQSMLA_ONLINE_SCORE_OPS = 4
MQSMLA_ONLINE_VALUE_OPS = 3


class _DslLowering:
    def __init__(self, plan, spec, region_id):
        self.plan = plan
        self.spec = spec
        self.region_id = region_id
        self.logical_cube = Counter()
        self.pairs = dict(plan.datapath.aic_aiv_pairs)

    def stage(self, name, after, **metadata):
        return self.plan.add(Stage(name, after=tuple(after), metadata=metadata))

    def access(self, name, tensor, amount, *, after=(), owner=0, write=False, private=False,
               cube=False, offset=0, working_set=None, bypass=False, version=None, runs=None, obligation=True):
        pipe = ("fixpipe" if cube else "mte3") if write else ("mte2_aic" if cube else "mte2_aiv")
        allocation = f"{self.region_id}:{tensor}" if private else tensor
        return self.plan.access(
            name, allocation, amount, pipe=pipe, after=after, owner=owner, offset=offset,
            direction="write" if write else "read", private=private, working_set=working_set,
            bypass=bypass, version=version, runs=runs, obligation=obligation,
        )

    def spill(self, name, tensor, size, after, *, owner=0, src_owner=None, dst_owner=None,
              cube_src=False, cube_dst=False, offset=0, working_set=None, bypass=False):
        written = self.access(
            name + ":write", tensor, size, after=after, owner=owner if src_owner is None else src_owner,
            write=True, private=True, cube=cube_src, offset=offset, working_set=working_set, bypass=bypass,
        )
        return self.access(name + ":read", tensor, size, after=(written,), owner=owner if dst_owner is None else dst_owner,
                           private=True, cube=cube_dst, offset=offset, working_set=working_set)

    def vector(self, name, ops, after, owner=0, role="vec_alu", **metadata):
        return self.plan.work(name, role, ops, after=after, owner=owner, metadata=metadata)

    def cube(self, name, logical, after, owner=0, *, dtype="bf16", issued=None,
             operands=None, explicit_mte1=False):
        if operands is None:
            if not explicit_mte1:
                raise ModelingError(f"dsl_cube_missing_mte1_operands:{name}")
        else:
            mte1 = tuple(
                self.local(f"{name}:l0{bank}", "mte1", size, after, owner)
                for bank, size in zip(("a", "b"), operands, strict=True) if size > 0
            )
            if mte1:
                after = mte1
        self.logical_cube[dtype] += logical
        return self.plan.work(name, "cube", logical if issued is None else issued, dtype=dtype, after=after,
                              owner=owner, metadata={"logical_flops": logical})

    def local(self, name, pipe, amount, after, owner):
        return self.plan.work(name, pipe, amount, after=after, owner=owner)

    def paired(self, core):
        owners = tuple(v for v in self.pairs.get(core, ()) if v < self.spec.aiv_budget)
        if not owners:
            raise ModelingError(f"dsl_paired_owner_unavailable:{core}")
        return owners

    def paired_subblocks(self, core):
        owners = self.paired(core)
        if len(owners) < DSL_SUBBLOCK_LIMIT:
            raise ModelingError(f"dsl_two_paired_subblocks_required:{core}")
        return owners[:DSL_SUBBLOCK_LIMIT]

    def launch(self, p):
        ready = ()
        if self.spec.uses_aicpu_metadata:
            ready = (self.plan.launch("aicpu_launch"),)
            ready = (self.plan.metadata(p.metadata_ms, after=ready),)
        if self.spec.has_eager_d2h_readback:
            if p.eager_d2h_bytes < 0 or p.eager_d2h_ms < 0:
                raise ModelingError("flash_kda_eager_readback_invalid")
            work = {}
            if p.eager_d2h_bytes:
                key = "d2h"
                if self.plan.context is not None:
                    key = self.plan.context.resource_names.get("d2h")
                    if key is None and not p.eager_d2h_ms:
                        raise ModelingError("flash_kda_d2h_resource_required")
                if key is not None:
                    work[key] = p.eager_d2h_bytes
            ready = (self.plan.add(Stage(
                "metadata_eager_d2h", work, after=ready, latency=p.eager_d2h_ms,
                metadata={"readback_bytes": p.eager_d2h_bytes},
            )),)
        return self.plan.launch(after=ready)

    def semantic(self):
        work = [
            SemanticWorkItem(f"{self.region_id}:matrix:{dt}", "matrix_mac", dt, ops, ("cube",),
                             (self.spec.template_variant,))
            for dt, ops in self.logical_cube.items() if ops
        ]
        kinds = {"vec_alu": "vector_alu", "vec_sfu": "vector_sfu", "vec_reduce": "vector_reduce",
                 "sort_topk": "sort_topk"}
        work.extend(SemanticWorkItem(
            self.region_id + ":" + pipe, kinds[pipe], "fp32", int(ops),
            ("aiv_simd",), (self.spec.template_variant,),
        ) for pipe, ops in self.plan.vector.items() if ops)
        return work


def _read_ring(ring, generation, ready, consumer):
    return lower_generation(
        ring, generation, "write", ("consume",), producer_extra_after=(ready,),
        consumer_extra_after={"consume": (consumer,)},
    ).publish


def _flash_state_subblocks(lower, prefix, suffix, tensor, base_offset, subblocks, after, *,
                           write=False, private=False, bypass=False, version=None, obligation=True):
    cols = KDA_HEAD_DIM // DSL_SUBBLOCK_LIMIT
    row_bytes = cols * FP32_BYTES
    stride = KDA_HEAD_DIM * FP32_BYTES
    return tuple(
        lower.access(
            f"{prefix}:sub{sub}:{suffix}", tensor, KDA_HEAD_DIM * row_bytes, owner=lane, after=after,
            runs=(AddressRun(base_offset + sub * row_bytes, KDA_HEAD_DIM, row_bytes, stride),),
            write=write, private=private, bypass=bypass, version=version, obligation=obligation,
        )
        for sub, lane in enumerate(subblocks)
    )


def _flash_norm_exchange(lower, name, core, subblocks, head, query_head, row, tail, deps):
    d = KDA_HEAD_DIM
    sub_cols = d // DSL_SUBBLOCK_LIMIT
    sub_bytes = sub_cols * BF16_BYTES
    row_stride = d * BF16_BYTES
    beta_load = lower.access(f"{name}:beta", f"kda:beta:{head}", tail * FP32_BYTES,
                             owner=subblocks[0], after=deps, offset=row * FP32_BYTES)
    exchange_bytes = KDA_NORM_VECTORS * KDA_CHUNK * FP32_BYTES
    exchange_ws = lower.spec.aic_budget * DSL_SUBBLOCK_LIMIT * exchange_bytes
    per_sub_loads, ex_writes = [], []
    for sub, lane in enumerate(subblocks):
        sub_loads = [beta_load]
        for tensor, tensor_head in (("q", query_head), ("k", query_head), ("v", head), ("g", head)):
            run = AddressRun(row * row_stride + sub * sub_bytes, tail, sub_bytes, row_stride)
            sub_loads.append(lower.access(
                f"{name}:sub{sub}:{tensor}", f"kda:{tensor}:{tensor_head}", tail * sub_bytes,
                owner=lane, after=deps, runs=(run,),
            ))
        per_sub_loads.append(sub_loads)
        partial = lower.vector(
            f"{name}:sub{sub}:qk_partial_norm", KDA_NORM_VECTORS * tail * sub_cols * 2,
            (sub_loads[1], sub_loads[2]), lane, "vec_reduce",
        )
        ex_writes.append(lower.access(
            f"{name}:sub{sub}:norm_exchange_write", "kda_qk_norm_exchange", exchange_bytes,
            after=(partial,), owner=lane, write=True, private=True,
            offset=(core * DSL_SUBBLOCK_LIMIT + sub) * exchange_bytes, working_set=exchange_ws,
        ))
    moves_subs = []
    for sub, lane in enumerate(subblocks):
        peer = 1 - sub
        ex_read = lower.access(
            f"{name}:sub{sub}:norm_exchange_read", "kda_qk_norm_exchange", exchange_bytes,
            after=(ex_writes[peer],), owner=lane, private=True,
            offset=(core * DSL_SUBBLOCK_LIMIT + peer) * exchange_bytes, working_set=exchange_ws,
        )
        norm = lower.vector(
            f"{name}:sub{sub}:norm_gate", tail * sub_cols * KDA_INPUT_VECTORS,
            (*per_sub_loads[sub], ex_read), lane,
        )
        moves_subs.append(lower.local(
            f"{name}:sub{sub}:operands_to_l1", "mte3", KDA_INPUT_VECTORS * tail * sub_bytes, (norm,), lane,
        ))
    return tuple(moves_subs)


def _lower_flash(lower, p, ready):
    """[Cohesive-Pipeline] Lower one FlashAttention DSL template into engine stages."""
    plan, spec = lower.plan, lower.spec
    d = KDA_HEAD_DIM
    state_bytes = d * d * FP32_BYTES
    staged = spec.template_variant == "staged"
    inv_products = KDA_STAGED_INVERSE_PRODUCTS if staged else KDA_PERSISTENT_INVERSE_PRODUCTS
    buffers, lane_last, lane_generation, prior_chunk = {}, {}, Counter(), {}
    previous_round = (ready,)
    token_base, chunk_base = [0], [0]
    for length in p.lengths:
        token_base.append(token_base[-1] + length)
        chunk_base.append(chunk_base[-1] + ceil_div(length, KDA_CHUNK))
    for round_tasks in flash_kda_lane_tasks(p, spec, datapath=lower.plan.datapath):
        scheduled = {(batch, head, chunk) for batch, head, chunk, _, _ in round_tasks} if staged else set()
        for batch, head, chunk, core, slot in round_tasks:
            subblocks = lower.paired_subblocks(core)
            sub_cols = d // DSL_SUBBLOCK_LIMIT
            vector = subblocks[0]
            name = f"kda:{batch}:{head}:{chunk}"
            if not staged and core not in buffers:
                buffers[core] = RingBuffer(die=plan.die, owner=core, pool="l1", depth=2,
                                           allocation=f"kda:state_l1:{core}")
            ring = buffers.get(core)
            generation = lane_generation[core]
            lane_generation[core] += 1
            seq = (batch, head)
            state_offset = (batch * p.value_heads + head) * state_bytes
            h_offset = (chunk_base[batch] + chunk) * state_bytes
            h_tensor = f"kda:h:{head}"
            private_h = not p.save_chunk_states
            tail = min(KDA_CHUNK, p.lengths[batch] - chunk * KDA_CHUNK)
            phys_m = tail if staged else KDA_CHUNK
            row = token_base[batch] + chunk * KDA_CHUNK
            deps = tuple(dict.fromkeys((*previous_round, lane_last.get(core, ready))))
            query_head = head // (p.value_heads // p.query_heads)
            moves_subs = _flash_norm_exchange(lower, name, core, subblocks, head, query_head, row, tail, deps)
            scores = lower.cube(
                name + ":kkt_qkt", 2 * 2 * tail * tail * d, moves_subs, core,
                issued=2 * 2 * phys_m * phys_m * d,
                operands=(2 * phys_m * d * BF16_BYTES, phys_m * d * BF16_BYTES),
            )
            score_ub = lower.local(name + ":scores_to_ub", "fixpipe", 2 * phys_m * phys_m * FP32_BYTES, (scores,), core)
            inv_bytes = inv_products * KDA_CHUNK * KDA_CHUNK * BF16_BYTES
            inverse = lower.cube(
                name + ":wy_inverse", 2 * inv_products * tail * tail * tail, (score_ub,), core,
                dtype="fp16", issued=2 * inv_products * KDA_CHUNK * KDA_CHUNK * KDA_CHUNK,
                operands=(inv_bytes, inv_bytes),
            )
            wy = lower.cube(
                name + ":kcd_u", 2 * 2 * tail * tail * d, (inverse,), core,
                issued=2 * 2 * phys_m * phys_m * d,
                operands=(phys_m * phys_m * BF16_BYTES, 2 * phys_m * d * BF16_BYTES),
            )
            saved = []
            matrix_offset = row * KDA_CHUNK * BF16_BYTES
            matrix_bytes = tail * KDA_CHUNK * BF16_BYTES
            if p.save_matrices:
                for tensor in ("Akk", "Aqk"):
                    saved.append(lower.access(
                        f"{name}:save_{tensor}", f"kda:{tensor}:{head}", matrix_bytes,
                        after=(wy,), offset=matrix_offset, owner=vector, write=True, bypass=True,
                    ))
            if p.save_kernel_caches:
                for tensor, element in (("qg", BF16_BYTES), ("g_cumsum", FP32_BYTES)):
                    saved.append(lower.access(
                        f"{name}:save_{tensor}", f"kda:{tensor}:{head}", tail * d * element,
                        after=(wy,), offset=row * d * element, owner=vector, write=True, bypass=True,
                    ))
            entry = prior_chunk.get(seq, (ready,))
            continued = staged and chunk > 0 and (*seq, chunk - 1) not in scheduled
            if chunk == 0 and p.has_initial_state:
                entry = _flash_state_subblocks(lower, name, "initial_state", "kda:initial_state",
                                               state_offset, subblocks, deps)
            elif continued:
                entry = _flash_state_subblocks(
                    lower, name, "reload_h", h_tensor, h_offset, subblocks, (*deps, *entry),
                    version=int(not private_h), private=private_h, bypass=True, obligation=False,
                )
            if staged:
                q_size = tail * d * BF16_BYTES
                gamma_size = d * FP32_BYTES
                q = lower.spill(name + ":q_decayed", "Q_decayed", q_size, (wy,), src_owner=vector,
                                dst_owner=core, cube_dst=True, offset=slot * KDA_CHUNK * d * BF16_BYTES)
                gamma = lower.spill(name + ":gamma", "gamma_C", gamma_size, (wy,), owner=vector,
                                    offset=slot * gamma_size)
                wk = tuple(lower.spill(
                    f"{name}:{tensor}", f"{tensor}:{head}", q_size, (wy,), owner=core, cube_src=True,
                    cube_dst=True, offset=row * d * BF16_BYTES, working_set=token_base[-1] * d * BF16_BYTES,
                ) for tensor in ("w", "u"))
                if p.save_matrices:
                    aqk = lower.access(
                        name + ":Aqk:read", f"kda:Aqk:{head}", matrix_bytes, after=(saved[1],),
                        owner=core, cube=True, offset=matrix_offset, version=1, obligation=False,
                    )
                else:
                    aqk = lower.spill(name + ":Aqk", f"Aqk:{head}", matrix_bytes, (wy,), owner=core,
                                      cube_src=True, cube_dst=True, offset=matrix_offset)
                state_deps = (q, gamma, *wk, aqk, *entry)
            else:
                state_deps = (wy, *entry, ring_credit(ring, generation))
            state_subs = tuple(
                lower.local(f"{name}:sub{sub}:state_to_l1", "mte3", d * sub_cols * BF16_BYTES, state_deps, lane)
                for sub, lane in enumerate(subblocks)
            )
            state = lower.stage(name + ":state_to_l1", state_subs)
            if ring is not None:
                state = _read_ring(ring, generation, state, name + ":state_products")
            inter = lower.cube(
                name + ":state_products", 2 * 2 * tail * d * d, (state,), core,
                issued=2 * 2 * phys_m * d * d,
                operands=(2 * phys_m * d * BF16_BYTES, d * d * BF16_BYTES),
            )
            delta = lower.cube(
                name + ":delta", 2 * tail * d * d, (inter,), core,
                issued=2 * phys_m * d * d,
                operands=(phys_m * d * BF16_BYTES, phys_m * d * BF16_BYTES),
            )
            delta_ub = lower.local(name + ":delta_to_ub", "fixpipe", state_bytes, (delta,), core)
            update_deps = (delta_ub,)
            if (staged or p.save_chunk_states) and not continued:
                update_deps += _flash_state_subblocks(
                    lower, name, "save_h", h_tensor, h_offset, subblocks, (state,),
                    write=True, bypass=True, private=private_h,
                )
            update_subs = tuple(
                lower.vector(f"{name}:sub{sub}:fp32_state_update", d * sub_cols * 2, update_deps, lane)
                for sub, lane in enumerate(subblocks)
            )
            update = lower.stage(name + ":fp32_state_update", update_subs)
            intra = lower.cube(
                name + ":intra", 2 * tail * tail * d, (inter,), core,
                issued=2 * phys_m * phys_m * d,
                operands=(phys_m * phys_m * BF16_BYTES, phys_m * d * BF16_BYTES),
            )
            output = lower.access(name + ":output", f"kda:head:{head}:output", tail * d * BF16_BYTES,
                                  offset=row * d * BF16_BYTES, after=(intra,), cube=True, owner=core, write=True)
            epilogue = (output, update, *saved)
            final = chunk + 1 == ceil_div(p.lengths[batch], KDA_CHUNK)
            carry = (update,)
            if final and p.return_final_state:
                epilogue += _flash_state_subblocks(
                    lower, name, "final_state", "kda:final_state", state_offset, subblocks, (update,), write=True,
                )
            elif staged and not final and (*seq, chunk + 1) not in scheduled:
                carry = _flash_state_subblocks(
                    lower, name, "carry_h", h_tensor, h_offset + state_bytes, subblocks, (update,),
                    private=private_h, write=True, bypass=True,
                )
                epilogue += carry
            prior_chunk[seq] = carry
            lane_last[core] = lower.stage(name + ":done", epilogue)
        if staged:
            previous_round = tuple(lane_last.values())
    for ring in buffers.values():
        plan.stages.extend(ring.to_stages())


def _mega_output_projection(lower, total_rows, in_width, hidden_size, ws_q_bytes, publish, ready):
    spec = lower.spec
    col_tiles = ceil_div(hidden_size, MEGA_OUT_TILE_N)
    core_prev = {}
    job_idx = 0
    for row0 in range(0, total_rows, MEGA_OUT_WAVE_ROWS):
        wave_rows = min(MEGA_OUT_WAVE_ROWS, total_rows - row0)
        wave_a_bytes = wave_rows * in_width * BF16_BYTES
        for col in range(col_tiles):
            col0 = col * MEGA_OUT_TILE_N
            col_w = min(MEGA_OUT_TILE_N, hidden_size - col0)
            core = col % spec.aic_budget
            prev = (core_prev[core],) if core in core_prev else ()
            job_name = f"mega:{job_idx}"
            operand = lower.access(
                job_name + ":read_gated_workspace", "Q", wave_a_bytes, cube=True, owner=core,
                after=(publish, *prev), offset=row0 * in_width * BF16_BYTES, private=True, working_set=ws_q_bytes,
            )
            w_bytes = in_width * col_w * BF16_BYTES
            weight = lower.access(
                job_name + ":out_weight", "mega:w_out", w_bytes, cube=True, owner=core,
                after=(ready, *prev), offset=col0 * in_width * BF16_BYTES,
            )
            result = lower.cube(
                job_name + ":output_projection", 2 * wave_rows * in_width * col_w, (operand, weight), core,
                operands=(wave_a_bytes, w_bytes),
            )
            out_run = AddressRun(
                (row0 * hidden_size + col0) * BF16_BYTES, wave_rows, col_w * BF16_BYTES, hidden_size * BF16_BYTES,
            )
            core_prev[core] = lower.access(
                job_name + ":output", "mega:output", wave_rows * col_w * BF16_BYTES,
                after=(result,), cube=True, owner=core, write=True, runs=(out_run,),
            )
            job_idx += 1


def _lower_mega(lower, p, ready):
    spec = lower.spec
    if p.num_heads < 1 or p.num_heads % p.tp != 0 or p.hidden_size < 1:
        raise ModelingError("mega_kda_geometry_invalid")
    h = p.num_heads // p.tp
    hidden_size = p.hidden_size
    d, batch = KDA_HEAD_DIM, p.batch * p.tp
    workspaces = mega_workspace_sizes(p)
    capacity = replay_pool_capacity(p.replay_capacity, p.sequence)
    u_stride, k_stride, decay_stride, checkpoint_stride = replay_pool_strides(p, h, capacity)
    if not p.unfused_launches or len(set(p.unfused_launches)) != len(p.unfused_launches):
        raise ModelingError("mega_kda_unfused_launch_baseline_invalid")
    ready = lower.stage("mega:unfused_baseline", (ready,), saved_launches=len(p.unfused_launches) - 1)
    conv_width = KDA_CONV_STREAMS * h * d
    conv_read_bytes = (KDA_CONV_TAPS - 1) * conv_width * BF16_BYTES
    conv_write_bytes = (KDA_CONV_HISTORY_KEEP + p.sequence) * conv_width * BF16_BYTES
    conv_stride_bytes = mega_conv_stride_elems(p) * BF16_BYTES
    taps = lower.access("mega:conv_weight", "mega:w_conv", KDA_CONV_TAPS * conv_width * BF16_BYTES, after=(ready,))
    a_log_loads = tuple(
        lower.access(f"mega:{lane}:a_log", "mega:A_log", h * FP32_BYTES, owner=lane, after=(ready,))
        for lane in range(spec.aiv_budget)
    )
    dt_bias_loads = tuple(
        lower.access(f"mega:{lane}:dt_bias", "mega:dt_bias", h * d * FP32_BYTES, owner=lane, after=(ready,))
        for lane in range(spec.aiv_budget)
    )
    out_norm_loads = tuple(
        lower.access(f"mega:{lane}:output_norm_weight", "mega:output_norm_weight", d * FP32_BYTES, owner=lane, after=(ready,))
        for lane in range(spec.aiv_budget)
    )
    lane_last, handoffs = {}, []
    for tile in range(batch):
        core, vector = tile % spec.aic_budget, tile % spec.aiv_budget
        slot = p.state_indices[tile] if p.state_indices else tile
        name = f"mega:{tile}"
        prior = (lane_last.get(core, ready),)
        hidden = lower.access(name + ":hidden", "hidden_states", p.sequence * hidden_size * BF16_BYTES,
                              cube=True, owner=core, after=prior, offset=tile * p.sequence * hidden_size * BF16_BYTES)
        indices = lower.access(name + ":state_index", "state_indices", INT32_BYTES,
                               owner=vector, after=prior, offset=tile * INT32_BYTES)
        conv_accepted = lower.access(name + ":conv_accepted", "mega:conv_num_accepted_tokens", INT32_BYTES,
                                     owner=vector, after=prior, offset=tile * INT32_BYTES)
        accepted = p.conv_accepted[tile] if p.conv_accepted else 0
        hist_off_bytes = max(accepted - 1, 0) * conv_width * BF16_BYTES
        conv_history = lower.access(name + ":conv_read", "mega:conv_state", conv_read_bytes, owner=vector,
                                    after=(*prior, indices, conv_accepted), offset=slot * conv_stride_bytes + hist_off_bytes)
        fa_weight = lower.access(name + ":fa_weight", "mega:w_fa", hidden_size * d * BF16_BYTES,
                                 cube=True, owner=core, after=(ready,))
        fa = lower.cube(
            name + ":fa_projection", 2 * p.sequence * hidden_size * d, (hidden, fa_weight), core,
            operands=(p.sequence * hidden_size * BF16_BYTES, hidden_size * d * BF16_BYTES),
        )
        decay_input = lower.spill(
            name + ":fa", "fa", p.sequence * d * BF16_BYTES, (fa,),
            src_owner=core, dst_owner=core, offset=tile * p.sequence * d * BF16_BYTES, working_set=workspaces["fa"],
            cube_src=True, cube_dst=True,
        )
        projected = {}
        for tensor in ("Q", "K", "V", "decay", "beta", "output_gate"):
            width = 1 if tensor == "beta" else d
            in_width = d if tensor == "decay" else hidden_size
            weight_bytes = in_width * h * width * BF16_BYTES
            load = lower.access(f"{name}:{tensor}:weight", f"mega:w_{tensor}", weight_bytes,
                                cube=True, owner=core, after=(ready,))
            operand = decay_input if tensor == "decay" else hidden
            projection = lower.cube(
                f"{name}:{tensor}:projection", 2 * p.sequence * in_width * h * width, (operand, load), core,
                operands=(p.sequence * in_width * BF16_BYTES, weight_bytes),
            )
            amount = p.sequence * h * width * BF16_BYTES
            if tensor in {"Q", "K", "V"}:
                vector_dep = lower.vector(f"{name}:{tensor}:causal_conv", 2 * KDA_CONV_TAPS * p.sequence * h * d,
                                          (projection, conv_history, taps), vector)
            else:
                vector_dep = lower.vector(f"{name}:{tensor}:prepare", p.sequence * h * width, (projection,), vector)
            projected[tensor] = lower.spill(
                f"{name}:{tensor}:gm", tensor, amount, (vector_dep,), owner=vector,
                offset=tile * (workspaces[tensor] // batch), working_set=workspaces[tensor],
            )
        conv_out = lower.access(name + ":conv_write", "mega:conv_state", conv_write_bytes, owner=vector,
                                after=tuple(projected[t] for t in ("Q", "K", "V")),
                                offset=slot * conv_stride_bytes, write=True)
        state_size = h * d * d * FP32_BYTES
        state = lower.access(name + ":state_read", "checkpoint", state_size,
                             owner=vector, after=(indices,), offset=slot * checkpoint_stride * FP32_BYTES)
        rec = lower.vector(
            name + ":recurrent", p.sequence * h * (d * d * KDA_RECURRENT_STATE_OPS + 2 * d),
            (state, a_log_loads[vector], dt_bias_loads[vector],
             *(dep for tensor, dep in projected.items() if tensor != "output_gate")),
            vector,
        )
        records = []
        for (tensor, element), pool_stride in zip(_REPLAY_DTYPES, (u_stride, k_stride, decay_stride), strict=True):
            row_bytes = d * element
            stride = capacity * row_bytes
            for token in range(p.sequence):
                offset = slot * pool_stride * element + token * row_bytes
                records.append(lower.access(f"{name}:{tensor}:{token}:write", tensor, h * row_bytes, after=(rec,),
                                            owner=vector, write=True, runs=(AddressRun(offset, h, row_bytes, stride),)))
        gated = lower.vector(name + ":gate", p.sequence * h * d,
                             (rec, projected["output_gate"], out_norm_loads[vector]), vector)
        payload = p.sequence * h * d * BF16_BYTES
        handoff = lower.access(name + ":gate_workspace", "Q", payload, owner=vector, after=(gated,),
                               offset=tile * payload, private=True, write=True, working_set=workspaces["Q"])
        handoffs.append(handoff)
        lane_last[core] = lower.stage(name + ":vector_done", (conv_out, handoff, *records))
    publish = lower.stage("mega:publish_vector_grid_to_cube", handoffs, gm_workspace_handoff=True)
    _mega_output_projection(lower, batch * p.sequence, h * d, hidden_size, workspaces["Q"], publish, ready)


_REPLAY_DTYPES = (("replay_u", FP32_BYTES), ("replay_k", BF16_BYTES), ("replay_decay", FP32_BYTES))


def _lower_commit(lower, p, ready):
    d = KDA_HEAD_DIM
    capacity = replay_pool_capacity(p.replay_capacity, max(p.accepted, default=0))
    u_stride, k_stride, decay_stride, checkpoint_stride = replay_pool_strides(p, p.heads, capacity)
    for lane, items in enumerate(commit_rows(p, lower.spec.aiv_budget)):
        previous = ready
        for item, row, rows in items:
            batch, head = divmod(item, p.heads)
            accepted = p.accepted[batch]
            name = f"commit:{item}:{row}"
            previous = lower.access(name + ":accepted_read", "accepted", INT32_BYTES, owner=lane,
                                    offset=batch * INT32_BYTES, after=(previous,))
            if not accepted:
                continue
            slot = p.state_indices[batch] if p.state_indices else batch
            state_offset = (slot * checkpoint_stride + (head * d + row) * d) * FP32_BYTES
            state_size = rows * d * FP32_BYTES
            item_ready = lower.access(name + ":state_index", "state_indices", INT32_BYTES, owner=lane,
                                      after=(previous,), offset=batch * INT32_BYTES)
            previous = lower.access(name + ":checkpoint_read", "checkpoint", state_size, owner=lane,
                                    offset=state_offset, after=(item_ready,))
            rings = tuple(
                RingBuffer(die=lower.plan.die, owner=lane, pool="ub", depth=2, allocation=f"{name}:{tensor}",
                           slot_bytes=d * element if lower.plan.context is not None else 0)
                for tensor, element in _REPLAY_DTYPES
            )
            for token in range(accepted):
                updated = f"{name}:token{token}:update"
                publish = []
                for ring, (tensor, element), pool_stride, payload, col in zip(
                    rings, _REPLAY_DTYPES, (u_stride, k_stride, decay_stride), (rows, d, d), (row, 0, 0), strict=True,
                ):
                    offset = (slot * pool_stride + (head * capacity + token) * d + col) * element
                    read = lower.access(
                        f"{name}:token{token}:{tensor}:read", tensor,
                        payload * element, owner=lane, after=(item_ready, ring_credit(ring, token)), offset=offset, version=1,
                    )
                    publish.append(_read_ring(ring, token, read, updated))
                previous = lower.vector(updated, KDA_STATE_UPDATE_OPS * rows * d,
                                        (previous, *publish), lane)
            previous = lower.access(name + ":checkpoint_write", "checkpoint", state_size,
                                    owner=lane, after=(previous,), offset=state_offset, write=True)
            for ring in rings:
                lower.plan.add_ring(ring, after=(item_ready,))


def _dsl_issued_cube_flops(datapath, m: int, n: int, k: int, *, dtype: str = "bf16") -> int:
    """Pad cube matrix dimensions to hardware ``subtile_wave`` instruction granules."""
    native = next((wave for wave in datapath.subtile_wave if wave.native_mode == dtype), None)
    bm, bn, bk = (
        (native.independently_issuable_m, native.tile_n, native.tile_k)
        if native is not None
        else (DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM)
    )
    return 2 * (ceil_div(m, bm) * bm) * (ceil_div(n, bn) * bn) * (ceil_div(k, bk) * bk)


def _mla_issued_m_rows(
    datapath, rows: int, bm: int, *, actual_rows: bool = False, wkb_tail: bool = False, dtype: str = "bf16",
) -> int:
    """Compute physical M rows executed by an MLA projection across its launcher M tiles."""
    native = next((wave for wave in datapath.subtile_wave if wave.native_mode == dtype), None)
    subtile_m = native.independently_issuable_m if native is not None else DEFAULT_INST_BLOCK_DIM
    m_tiles = ceil_div(rows, bm)
    if actual_rows:
        return sum(
            ceil_div(min(bm, rows - idx * bm), subtile_m) * subtile_m
            for idx in range(m_tiles)
        )
    if wkb_tail and bm > MLA_WKB_M_TAIL_ROWS:
        return sum(
            ceil_div(
                MLA_WKB_M_TAIL_ROWS if idx * bm + MLA_WKB_M_TAIL_ROWS >= rows else bm,
                subtile_m,
            ) * subtile_m
            for idx in range(m_tiles)
        )
    return m_tiles * ceil_div(bm, subtile_m) * subtile_m


def _mla_emit_fp8_descale_writes(lower, name: str, start: int, rows: int, heads: int, after) -> tuple[str, ...]:
    """Emit per-token/head 4 B FP32 ``descale_q`` writes across participating AIVs."""
    items = rows * heads
    active_aivs = min(items, lower.spec.aiv_budget)
    base_item = start * heads
    writes: list[str] = []
    for lane in range(active_aivs):
        count = ceil_div(items - lane, active_aivs)
        if active_aivs > 1:
            runs = (AddressRun((base_item + lane) * FP32_BYTES, count, FP32_BYTES, active_aivs * FP32_BYTES),)
        elif count > 1:
            even_count = ceil_div(count, MLA_SINGLE_AIV_STRIDE_SPLITS)
            odd_count = count - even_count
            stride = MLA_SINGLE_AIV_STRIDE_SPLITS * FP32_BYTES
            runs = (
                AddressRun(base_item * FP32_BYTES, even_count, FP32_BYTES, stride),
                AddressRun((base_item + 1) * FP32_BYTES, odd_count, FP32_BYTES, stride),
            )
        else:
            runs = (AddressRun((base_item + lane) * FP32_BYTES, 1, FP32_BYTES, FP32_BYTES),)
        writes.append(lower.access(
            f"{name}:descale_q:{lane}", "mla:descale_q", count * FP32_BYTES,
            after=after, write=True, owner=lane, runs=runs,
        ))
    return tuple(writes)


def _mla_project_groups(lower, prefix, proj_name, weight_id, rows, in_width, total_out, group_size,
                        after, ready, a_spec, *, bm, actual_rows=False, per_core_a_reuse=False):
    a_tensor, a_offset, a_private, a_ws, a_version, a_obligation = a_spec
    aic = lower.spec.aic_budget
    core_prev, core_a, out = {}, {}, []
    a_full_bytes = rows * in_width * BF16_BYTES
    issued_m = _mla_issued_m_rows(lower.plan.datapath, rows, bm, actual_rows=actual_rows)
    for group, col0 in enumerate(range(0, total_out, group_size)):
        g_out = min(group_size, total_out - col0)
        core = group % aic
        prev = (core_prev[core],) if core in core_prev else ()
        if not per_core_a_reuse or core not in core_a:
            a_dep = lower.access(
                f"{prefix}:g{group}:{proj_name}:a_l1", a_tensor, a_full_bytes,
                after=(*after, *prev), offset=a_offset, cube=True, owner=core,
                private=a_private, working_set=a_ws, version=a_version, obligation=a_obligation,
            )
            a_l0_bytes = a_full_bytes
            core_a[core] = a_dep
        else:
            a_dep = core_a[core]
            a_l0_bytes = 0
        w_bytes = in_width * g_out * BF16_BYTES
        w_dep = lower.access(
            f"{prefix}:g{group}:{proj_name}:weight", weight_id, w_bytes,
            after=(ready, *prev), offset=col0 * in_width * BF16_BYTES, cube=True, owner=core,
        )
        stage = lower.cube(
            f"{prefix}:g{group}:{proj_name}", 2 * rows * in_width * g_out, (a_dep, w_dep, *prev),
            core, issued=_dsl_issued_cube_flops(lower.plan.datapath, issued_m, g_out, in_width),
            operands=(a_l0_bytes, w_bytes),
        )
        core_prev[core] = stage
        out.append((core, stage, col0, g_out))
    return tuple(out)


def _mla_project_wkb(lower, prefix, proj_name, rows, heads, q_nope, head_dim, padded_qp, latent,
                     wkb_grouped, after, ready, *, bm, ws_q_proj_bytes=None):
    aic = lower.spec.aic_budget
    chunks = 1 if wkb_grouped else MLA_WKB_N_CHUNKS
    chunk_cols = latent // chunks
    core_prev, stages = {}, []
    total_tasks = heads * chunks
    issued_m = _mla_issued_m_rows(lower.plan.datapath, rows, bm, wkb_tail=not wkb_grouped)
    for task in range(total_tasks):
        n_chunk = task % chunks
        head = task // chunks
        core = task % aic
        prev = (core_prev[core],) if core in core_prev else ()
        a_bytes = rows * q_nope * BF16_BYTES
        if ws_q_proj_bytes is not None:
            a_run = AddressRun(head * head_dim * BF16_BYTES, rows, q_nope * BF16_BYTES, padded_qp * BF16_BYTES)
            a_dep = lower.access(
                f"{prefix}:t{task}:{proj_name}:a_l1", "ws_q_proj", a_bytes,
                after=(*after, *prev), cube=True, owner=core, private=True,
                working_set=ws_q_proj_bytes, runs=(a_run,),
            )
            cube_after = (a_dep, *prev)
        else:
            cube_after = (*after, *prev)
        w_bytes = q_nope * chunk_cols * BF16_BYTES
        w_run = AddressRun(
            (head * q_nope * latent + n_chunk * chunk_cols) * BF16_BYTES,
            q_nope, chunk_cols * BF16_BYTES, latent * BF16_BYTES,
        )
        w_dep = lower.access(
            f"{prefix}:t{task}:{proj_name}:weight", "wkb", w_bytes,
            after=(ready, *prev), cube=True, owner=core, runs=(w_run,),
        )
        stage = lower.cube(
            f"{prefix}:t{task}:{proj_name}", 2 * rows * q_nope * chunk_cols, (*cube_after, w_dep),
            core, issued=_dsl_issued_cube_flops(lower.plan.datapath, issued_m, chunk_cols, q_nope),
            operands=(a_bytes, w_bytes),
        )
        core_prev[core] = stage
        stages.append(stage)
    return tuple(stages)


def _mla_spill_groups(lower, prefix, tensor, ws_rows, row_pitch_cols, elem_bytes, groups, dst_owner, *,
                      cube_dst=False, working_set=None, emit_read=True):
    stride_bytes = row_pitch_cols * elem_bytes
    writes = []
    for idx, (core, stage, col0, g_out) in enumerate(groups):
        row_bytes = g_out * elem_bytes
        run = AddressRun(col0 * elem_bytes, ws_rows, row_bytes, stride_bytes)
        writes.append(lower.access(
            f"{prefix}:g{idx}:write", tensor, ws_rows * row_bytes, after=(stage,), owner=core,
            write=True, private=True, cube=True, working_set=working_set, runs=(run,),
        ))
    if not emit_read:
        return tuple(writes)
    return lower.access(
        prefix + ":read", tensor, ws_rows * stride_bytes, after=tuple(writes), owner=dst_owner,
        private=True, cube=cube_dst, working_set=working_set,
    )


def _mla_kva_owner(lower, core):
    owners = tuple(v for v in lower.pairs.get(core, ()) if v < lower.spec.aiv_budget)
    return owners[0] if owners else core % lower.spec.aiv_budget


def _lower_mla(lower, p, ready):
    """[Cohesive-Pipeline] Lower one DSL MLA kernel specification into tiled compute and memory stages."""
    spec = lower.spec
    mode = p.quant_mode
    workspace, _, scratch_rows, recompute = mla_workspaces(p, spec.aic_budget)
    head_dim = p.q_nope + p.rope
    qp = p.heads * head_dim
    padded_qa = ceil_div(p.q_lora, MLA_COLUMN_TILE) * MLA_COLUMN_TILE
    padded_kva = ceil_div(p.kvr, MLA_COLUMN_TILE) * MLA_COLUMN_TILE
    padded_qp = ceil_div(qp, MLA_COLUMN_TILE) * MLA_COLUMN_TILE
    qa_partials = workspace["ws_qa"] != scratch_rows * padded_qa * BF16_BYTES
    qa_ws_rows = scratch_rows * (MLA_QA_PARTIALS if qa_partials else 1)
    qa_elem = FP32_BYTES if qa_partials else BF16_BYTES
    latent = p.kvr - p.rope
    if latent < 1:
        raise ModelingError("mla_prolog_latent_width_invalid")
    cache_row = p.cache_row_bytes
    quant_rows = []
    maxima_by_core = {core: [] for core in range(spec.aic_budget)}
    previous = (ready,)
    for chunk, start in enumerate(range(0, p.tokens, scratch_rows)):
        rows = min(scratch_rows, p.tokens - start)
        name = f"mla:{chunk}"
        core, vector = chunk % spec.aic_budget, chunk % spec.aiv_budget
        wqa_w, wkva_w, wqb_w, qb_grouped, wkb_grouped, bm = mla_projection_widths(p, spec.aic_budget, rows)
        direct_front = (
            p.weight_format == "nz" and mode == QkvQuantMode.A16C16 and bm <= MLA_FRONT_ACTUAL_ROWS_MAX
        )
        if "ws_x_pad" in workspace:
            x_raw = lower.access(name + ":x", "mla:x", rows * p.dim * BF16_BYTES, after=previous,
                                 offset=start * p.dim * BF16_BYTES, owner=vector)
            x_pad = lower.access(name + ":pad:write", "ws_x_pad", workspace["ws_x_pad"], after=(x_raw,),
                                 owner=vector, write=True, private=True, working_set=workspace["ws_x_pad"])
            front_after = (x_pad,)
            front_a = ("ws_x_pad", 0, True, workspace["ws_x_pad"], None, True)
        else:
            front_after = previous
            front_a = ("mla:x", start * p.dim * BF16_BYTES, False, None, None, True)
        qa_groups = _mla_project_groups(lower, name, "wqa", "wqa", rows, p.dim, p.q_lora, wqa_w,
                                        front_after, ready, front_a, bm=bm,
                                        actual_rows=direct_front and not qa_partials)
        kva_groups = _mla_project_groups(lower, name, "wkva", "wkva", rows, p.dim, p.kvr, wkva_w,
                                         front_after, ready, front_a, bm=bm, actual_rows=direct_front)
        qa_read = _mla_spill_groups(lower, name + ":ws_qa", "ws_qa", qa_ws_rows, padded_qa, qa_elem, qa_groups,
                                    vector, working_set=workspace["ws_qa"])
        kval = _mla_spill_groups(lower, name + ":ws_kva", "ws_kva", scratch_rows, padded_kva, BF16_BYTES, kva_groups,
                                 vector, working_set=workspace["ws_kva"])
        norm_qa_loads = tuple(
            lower.access(f"{name}:norm_weight_qa:{lane}:load", "mla:norm_weight_qa", p.q_lora * FP32_BYTES,
                         owner=lane, after=(ready,))
            for lane in range(min(rows, spec.aiv_budget))
        )
        norm_kva_loads = tuple(
            lower.access(f"{name}:norm_weight_kva:{c}:load", "mla:norm_weight_kva", latent * FP32_BYTES,
                         owner=_mla_kva_owner(lower, c), after=(ready,))
            for c in range(min(rows, spec.aic_budget))
        )
        norm = lower.vector(name + ":q_rms", rows * p.q_lora * ATTN_RES_NORM_OPS, (qa_read, *norm_qa_loads), vector)
        qa_bytes = rows * p.q_lora * BF16_BYTES
        if p.return_qr:
            norm = lower.access(name + ":qr", "mla:qr", qa_bytes, after=(norm,), owner=vector,
                                write=True, offset=start * p.q_lora * BF16_BYTES)
            qb_a = ("mla:qr", start * p.q_lora * BF16_BYTES, False, None, 1, False)
        elif workspace["ws_qb_aux"]:
            norm = lower.access(name + ":ws_qb_aux:write", "ws_qb_aux", qa_bytes, after=(norm,), owner=vector,
                                write=True, private=True, working_set=workspace["ws_qb_aux"])
            qb_a = ("ws_qb_aux", 0, True, workspace["ws_qb_aux"], None, True)
        else:
            qb_a = ("ws_qa", 0, True, workspace["ws_qa"], None, True)
        kv_norm = lower.vector(name + ":kv_norm_rope", rows * p.kvr * ATTN_RES_NORM_OPS, (kval, *norm_kva_loads), vector)
        index = lower.access(name + ":cache_index", "mla:cache_index", rows * MLA_CACHE_INDEX_BYTES,
                             after=(kv_norm,), owner=vector, offset=start * MLA_CACHE_INDEX_BYTES)
        slots = range(start, start + rows) if p.cache_index is None else p.cache_index[start:start + rows]
        slots = tuple(slot for slot in slots if slot >= 0)
        lane = MLA_CACHE_NZ_LANE_BYTES
        if p.cache_layout == "PA_BSND":
            runs = tuple(AddressRun(slot * cache_row, 1, cache_row, cache_row) for slot in slots)
        else:
            runs = tuple(AddressRun(page * p.cache_page_size * cache_row + row * lane, cache_row // lane,
                                    lane, p.cache_page_size * lane)
                         for page, row in (divmod(slot, p.cache_page_size) for slot in slots))
        cache = (lower.access(name + ":kv_cache", "mla:kv_cache", len(slots) * cache_row,
                              after=(index,), owner=vector, write=True, runs=runs) if slots else index)
        qb_groups = _mla_project_groups(lower, name, "wqb", "wqb", rows, p.q_lora, qp, wqb_w,
                                        (norm, cache), ready, qb_a, bm=bm, per_core_a_reuse=qb_grouped)
        proj_writes = _mla_spill_groups(lower, name + ":ws_q_proj", "ws_q_proj", scratch_rows, padded_qp, BF16_BYTES,
                                        qb_groups, core, cube_dst=True, working_set=workspace["ws_q_proj"], emit_read=False)
        kb_stages = _mla_project_wkb(lower, name, "wkb", rows, p.heads, p.q_nope, head_dim, padded_qp, latent,
                                     wkb_grouped, proj_writes, ready, bm=bm, ws_q_proj_bytes=workspace["ws_q_proj"])
        q_elements = rows * p.heads * p.kvr
        rope = lower.vector(name + ":q_rope", rows * p.heads * p.rope * ATTN_RES_NORM_OPS, kb_stages, vector)
        if mode in MLA_QUANTIZED_QUERY:
            size = q_elements * FP32_BYTES
            offset = 0 if mode == QkvQuantMode.FP8_TENSOR or recompute else start * p.heads * p.kvr * FP32_BYTES
            q_value = lower.access(name + ":ws_q_fp_write", "ws_q_fp", size, after=(rope,),
                                   owner=vector, offset=offset, private=True, write=True, working_set=workspace["ws_q_fp"])
            if mode == QkvQuantMode.HiF8:
                active = min(q_elements, spec.aic_budget)
                base, tail = divmod(q_elements, active)
                for worker in range(active):
                    w_lane = worker % spec.aiv_budget
                    maxima_by_core[worker].append(lower.vector(
                        f"{name}:partial_max:{worker}", base + (worker < tail), (q_value,), w_lane, "vec_reduce",
                    ))
                quant_rows.append((name, core, vector, start, rows, q_elements, offset, size, wqb_w, qb_grouped, wkb_grouped, bm))
                previous = tuple(values[-1] for values in maxima_by_core.values() if values)
                continue
            q_value = lower.access(name + ":ws_q_fp_read", "ws_q_fp", size, after=(q_value,),
                                   owner=vector, private=True, working_set=workspace["ws_q_fp"])
            rope = lower.vector(name + ":q_quant", q_elements * 2, (q_value,), vector)
        elem = 1 if mode == QkvQuantMode.FP8_TENSOR else BF16_BYTES
        output = lower.access(name + ":q_output", "mla:q", q_elements * elem, after=(rope,), write=True,
                              offset=start * p.heads * p.kvr * elem, owner=vector)
        if mode == QkvQuantMode.FP8_TENSOR:
            descales = _mla_emit_fp8_descale_writes(lower, name, start, rows, p.heads, (rope,))
            previous = (output, *descales)
        else:
            previous = (output,)
    if not quant_rows:
        return
    max_events = []
    for worker, events in maxima_by_core.items():
        if not events:
            continue
        w_lane = worker % spec.aiv_budget
        maximum = lower.vector(f"mla:merge_core_max:{worker}", len(events) - 1, events, w_lane, "vec_reduce")
        max_events.append(lower.access(
            f"mla:ws_amax_write:{worker}", "ws_amax", FP32_BYTES, after=(maximum,), owner=w_lane,
            private=True, write=True, offset=worker * FP32_BYTES,
        ))
    barrier = lower.stage("q_global_quant_launch_barrier", max_events, launch_wide=True)
    maxima = lower.access("q_global_quant_amax_read", "ws_amax", workspace["ws_amax"],
                          after=(barrier,), private=True)
    lower.access(
        "mla:descale_q", "mla:descale_q", FP32_BYTES,
        after=(maxima,), owner=quant_rows[0][2], write=True, offset=0,
    )
    for name, core, vector, start, rows, elements, offset, size, wqb_w, qb_grouped, wkb_grouped, bm in quant_rows:
        after = (maxima,)
        if recompute:
            re_a = (
                "mla:qr" if p.return_qr else "ws_qb_aux",
                start * p.q_lora * BF16_BYTES, not p.return_qr, None, 1 if p.return_qr else None, not p.return_qr,
            )
            qb_re = _mla_project_groups(lower, name, "wqb_recompute", "wqb", rows, p.q_lora, qp, wqb_w,
                                        after, ready, re_a, bm=bm, per_core_a_reuse=qb_grouped)
            after = _mla_project_wkb(lower, name, "wkb_recompute", rows, p.heads, p.q_nope, head_dim, padded_qp, latent,
                                     wkb_grouped, tuple(stage for _, stage, _, _ in qb_re), ready, bm=bm)
        else:
            after = (lower.access(name + ":ws_q_fp_read", "ws_q_fp", size, after=after, private=True,
                                  offset=offset, owner=vector, working_set=workspace["ws_q_fp"]),)
        quant = lower.vector(name + ":q_global_quant", elements * 2, after, vector)
        lower.access(name + ":q_output", "mla:q", elements, after=(quant,), owner=vector, write=True,
                     offset=start * p.heads * p.kvr)


def _lower_attn_res(lower, p, ready):
    spec = lower.spec
    d = p.hidden_size
    rows = ceil_div(p.tokens, spec.aiv_budget)
    tile = spec.tile_shape[0]
    depth = 1 if rows <= tile else 2
    ready = lower.stage("attn_res:inplace_contract", (ready,),
                        saved_logical_bytes=2 * p.tokens * d * BF16_BYTES)
    for lane, start in enumerate(range(0, p.tokens, rows)):
        previous = ready
        weight = lower.access(f"attn_res:{lane}:weights", "attn_res:query_gamma", d * (FP32_BYTES + BF16_BYTES),
                              owner=lane, after=(ready,))
        for first in range(start, min(start + rows, p.tokens), tile):
            name = f"attn_res:{first}"
            count = min(tile, start + rows - first, p.tokens - first)
            elems = count * d
            a = lower.access(name + ":block", "attn_res:block", elems * BF16_BYTES,
                             owner=lane, after=(previous,), offset=first * d * BF16_BYTES)
            b = lower.access(name + ":partial", "attn_res:partial_block", elems * FP32_BYTES,
                             owner=lane, after=(previous,), offset=first * d * FP32_BYTES)
            numerator = lower.access(name + ":numerator", "attn_res:numerator", elems * FP32_BYTES,
                                     owner=lane, after=(previous,), offset=first * d * FP32_BYTES)
            stat = tuple(lower.access(
                f"{name}:{tensor}", f"attn_res:{tensor}", count * FP32_BYTES,
                owner=lane, after=(previous,), offset=first * FP32_BYTES,
            ) for tensor in ("logit_max", "exp_sum"))
            scores = lower.vector(name + ":score_dot_and_squares", elems * ATTN_RES_SCORE_OPS, (weight, a, b), lane)
            reduce = lower.vector(name + ":score_reduce", 2 * elems, (scores,), lane, "vec_reduce")
            score_rms = lower.vector(name + ":score_sqrt", count, (reduce,), lane, "vec_sfu")
            score = lower.vector(name + ":score_divide", count, (score_rms,), lane)
            exp = lower.vector(name + ":merge_exp", 2 * count * ATTN_RES_VL, (score, *stat), lane, "vec_sfu")
            merge = lower.vector(name + ":merge_round_bf16", elems * ATTN_RES_MERGE_OPS,
                                 (scores, numerator, exp), lane, buffer_depth=depth, inplace=True)
            partial = lower.access(name + ":inplace_partial", "attn_res:partial_block", elems * FP32_BYTES,
                                   after=(merge,), owner=lane, write=True, offset=first * d * FP32_BYTES)
            reduce = lower.vector(name + ":sum_squares", elems * 2, (merge,), lane, "vec_reduce")
            root = lower.vector(name + ":norm_sqrt", count, (reduce,), lane, "vec_sfu")
            rms = lower.vector(name + ":rms_norm", elems * ATTN_RES_NORM_OPS, (root,), lane)
            out = lower.access(name + ":output", "attn_res:y", elems * BF16_BYTES, owner=lane, after=(rms,),
                               write=True, offset=first * d * BF16_BYTES)
            previous = lower.stage(name + ":done", (partial, out))


def _qsli_block_base(block_id, page_table, pa_block_size, q_payload):
    block_bytes = QSLI_CANDIDATE_BLOCK_SIZE * q_payload
    blocks_per_page = pa_block_size // QSLI_CANDIDATE_BLOCK_SIZE
    page, within = divmod(block_id, blocks_per_page)
    physical = page_table[page] if page_table else page
    return physical * pa_block_size * q_payload + within * block_bytes


def _qsli_half_gather_runs(block_ids, page_table, pa_block_size, half_first_block, valid_blocks, q_payload):
    block_bytes = QSLI_CANDIDATE_BLOCK_SIZE * q_payload
    half_end = half_first_block + QSLI_BLOCKS_PER_AIV
    valid_lo = min(half_first_block, valid_blocks)
    valid_hi = min(half_end, valid_blocks)
    runs = []
    for blk in range(valid_lo, valid_hi):
        logical = block_ids[blk] if block_ids else blk
        base = _qsli_block_base(logical, page_table, pa_block_size, q_payload)
        runs.append(AddressRun(base, 1, block_bytes, block_bytes))
    pad_blocks = QSLI_BLOCKS_PER_AIV - (valid_hi - valid_lo)
    if pad_blocks > 0:
        last_logical = block_ids[valid_blocks - 1] if block_ids else valid_blocks - 1
        last_base = _qsli_block_base(last_logical, page_table, pa_block_size, q_payload)
        runs.append(AddressRun(last_base, pad_blocks, block_bytes, 0))
    return tuple(runs)


def _qsli_gather_tile(lower, p, n, tile_idx, total_blocks, block_ids, page_table, pair, q_payload, after):
    tile_first_block = tile_idx * QSLI_BLOCKS_PER_TILE
    half_bytes = QSLI_BLOCKS_PER_AIV * QSLI_CANDIDATE_BLOCK_SIZE * q_payload
    gathers = []
    for half, lane in enumerate(pair):
        half_first = tile_first_block + half * QSLI_BLOCKS_PER_AIV
        runs = _qsli_half_gather_runs(block_ids, page_table, p.pa_block_size, half_first, total_blocks, q_payload)
        gathers.append(lower.access(
            f"{n}:sub{half}:gather_pa", "qsli:key", half_bytes, owner=lane, after=after, runs=runs,
        ))
    return tuple(gathers)


def _lower_qsli(lower, p, ready):
    """[Cohesive-Pipeline] Lower one Quantized Sparse Lightning Indexer DSL template into stages."""
    spec = lower.spec
    workspace = qsli_workspace_sizes(p, spec.aic_budget)
    meta = lower.access("qsli:metadata_load", "qsli:metadata", QSLI_METADATA_BYTES, after=(ready,), private=True)
    q_payload = QSLI_PACKED_D + QSLI_SCALE_ROW_BYTES
    total_blocks = ceil_div(p.candidates, QSLI_CANDIDATE_BLOCK_SIZE)
    total_tiles = ceil_div(p.candidates, QSLI_TILE_N)
    cand_row_pitch = max(QSLI_CANDIDATE_CAPACITY, total_blocks)
    core_done: dict[int, tuple[str, ...]] = {}
    partial_slot_free: dict[int, tuple[str, ...]] = {}
    owner_done: dict[int, tuple[str, ...]] = {}
    for query in range(p.query_rows):
        owner = (
            lower.paired_subblocks(query % spec.aic_budget)[0]
            if (p.candidates and p.split_parts == 1)
            else query % spec.aiv_budget
        )
        block_ids = p.block_ids(query)
        page_table = p.page_table(query)
        parts = []
        for split in range(p.split_parts if p.candidates else 0):
            core = (query * p.split_parts + split) % spec.aic_budget
            core_after = core_done.get(core, (meta,))
            pair = lower.paired_subblocks(core)
            name = f"qsli:{query}:{split}"
            tile_begin = total_tiles * split // p.split_parts
            tile_end = total_tiles * (split + 1) // p.split_parts
            first = tile_begin * QSLI_TILE_N
            end = min(p.candidates, tile_end * QSLI_TILE_N)
            first_block = first // QSLI_CANDIDATE_BLOCK_SIZE
            end_block = ceil_div(end, QSLI_CANDIDATE_BLOCK_SIZE)
            cand_len = lower.access(
                name + ":candidate_block_length", "qsli:candidate_block_length", INT32_BYTES,
                owner=pair[0], after=core_after, offset=query * INT32_BYTES,
            )
            cand_idx = lower.access(
                name + ":candidate_block_indices", "qsli:candidate_block_indices",
                (end_block - first_block) * INT32_BYTES, owner=pair[0], after=(cand_len,),
                offset=(query * cand_row_pitch + first_block) * INT32_BYTES,
            )
            gather_base_after = (cand_idx,)
            if page_table:
                blocks_per_page = p.pa_block_size // QSLI_CANDIDATE_BLOCK_SIZE
                pages = tuple(dict.fromkeys(
                    (block_ids[b] if block_ids else b) // blocks_per_page for b in range(first_block, end_block)
                ))
                bt_runs = tuple(AddressRun(pg * INT32_BYTES, 1, INT32_BYTES, INT32_BYTES) for pg in pages)
                gather_base_after = (lower.access(
                    name + ":block_table", "qsli:block_table", len(bt_runs) * INT32_BYTES,
                    owner=pair[0], after=(cand_idx,), runs=bt_runs,
                ),)
            weights = tuple(
                lower.access(
                    f"{name}:sub{sub}:weights", "qsli:weights", QSLI_TILE_M * FP32_BYTES,
                    owner=lane, after=core_after, offset=query * QSLI_TILE_M * FP32_BYTES,
                )
                for sub, lane in enumerate(pair)
            )
            q = lower.access(name + ":query", "qsli:query", QSLI_TILE_M * q_payload, cube=True, owner=core,
                             after=core_after, offset=query * QSLI_TILE_M * q_payload)
            q_l0 = lower.local(name + ":query_l0a", "mte1", QSLI_TILE_M * q_payload, (q,), core)
            k_l1, l0b, l0c, staging = (
                RingBuffer(die=lower.plan.die, owner=core, pool=pool, depth=depth, allocation=f"{name}:{tag}")
                for tag, pool, depth in (("k_l1", "l1", 2), ("l0b", "l0b", 2),
                                         ("l0c", "l0c", 2), ("staging", "gm", QSLI_STAGING_DEPTH))
            )
            scores = []
            for tile, tile_idx in enumerate(range(tile_begin, tile_end)):
                pos = tile_idx * QSLI_TILE_N
                real = min(QSLI_TILE_N, end - pos)
                n = f"{name}:tile{tile}"
                gathers = _qsli_gather_tile(
                    lower, p, n, tile_idx, total_blocks, block_ids, page_table, pair, q_payload,
                    (*gather_base_after, ring_credit(staging, tile)),
                )
                staging_slot = (core * QSLI_STAGING_DEPTH + tile % QSLI_STAGING_DEPTH) * QSLI_TILE_N
                packed = lower.spill(n + ":publish_packed_k", "staging_key", QSLI_TILE_N * QSLI_PACKED_D, gathers,
                                     src_owner=pair[0], dst_owner=core, cube_dst=True,
                                     offset=staging_slot * QSLI_PACKED_D, working_set=workspace["staging_key"])
                scale = lower.spill(n + ":publish_scale", "staging_scale", QSLI_TILE_N * QSLI_SCALE_ROW_BYTES, gathers,
                                    src_owner=pair[1], dst_owner=core, cube_dst=True,
                                    offset=staging_slot * QSLI_SCALE_ROW_BYTES, working_set=workspace["staging_scale"])
                staged = lower.stage(n + ":staged_key_scale", (packed, scale, ring_credit(k_l1, tile)))
                staged = _read_ring(staging, tile, staged, n + ":l0_operands")
                refill = _read_ring(k_l1, tile, staged, n + ":l0_operands")
                k_l0 = lower.local(n + ":l0_operands", "mte1", QSLI_TILE_N * q_payload,
                                   (refill, ring_credit(l0b, tile)), core)
                operand = _read_ring(l0b, tile, k_l0, n + ":QsliCube")
                logical = 2 * QSLI_TILE_M * real * KDA_HEAD_DIM
                score = lower.cube(
                    n + ":QsliCube", logical, (q_l0, operand, ring_credit(l0c, tile)), core,
                    dtype=QSLI_MATRIX_DTYPE, issued=2 * QSLI_TILE_M * QSLI_TILE_N * KDA_HEAD_DIM, explicit_mte1=True,
                )
                score = _read_ring(l0c, tile, score, n + ":fix_done")
                fixes = []
                for sub, lane in enumerate(pair):
                    count = (real + (sub == 0)) // 2
                    fix = lower.plan.work(f"{n}:fix_subblock{sub}", "fixpipe", QSLI_TILE_M * count * FP32_BYTES,
                                          after=(score,), owner=core, metadata={"dst_subblock": sub})
                    fixes.append(fix)
                    scores.append(lower.vector(
                        f"{n}:score_reduce{sub}", QSLI_TILE_M * count, (fix, weights[sub]), lane, "vec_reduce",
                    ))
                lower.stage(n + ":fix_done", fixes)
            for ring in (k_l1, l0b, l0c, staging):
                lower.plan.add_ring(ring, after=core_after)
            counts = end - first
            score_bytes = counts * BF16_BYTES
            published = lower.spill(name + ":select_keys", "key_workspace", score_bytes,
                                    scores or (q,), owner=pair[0], offset=(core * p.candidates + first) * BF16_BYTES,
                                    working_set=workspace["key_workspace"])
            selected = published
            for trunk, size_start in enumerate(range(0, counts, QSLI_TRUNK)):
                size = min(QSLI_TRUNK, counts - size_start)
                high = lower.vector(f"{name}:trunk{trunk}:high_histogram", size + QSLI_HISTOGRAM_BINS,
                                    (selected,), pair[0], "sort_topk", bins=QSLI_HISTOGRAM_BINS)
                low = lower.vector(f"{name}:trunk{trunk}:masked_low_histogram", size + QSLI_HISTOGRAM_BINS,
                                   (high,), pair[0], "sort_topk", bins=QSLI_HISTOGRAM_BINS,
                                   counter_bytes=BF16_BYTES, low_byte_masked=True)
                selected = lower.vector(f"{name}:trunk{trunk}:threshold_ties", size, (low,), pair[0], "sort_topk")
            partial_size = min(p.topk, counts)
            pos_bytes = partial_size * INT32_BYTES
            positions = lower.spill(name + ":positions", "selected_position_workspace", pos_bytes, (selected,),
                                    owner=pair[0], offset=core * p.topk * INT32_BYTES)
            part = (positions,)
            if p.split_parts > 1:
                write_after = (positions, *partial_slot_free.get(core, ()))
                for tensor, elem in (("partial_indices", INT32_BYTES), ("partial_values", BF16_BYTES)):
                    part += (lower.access(f"{name}:{tensor}", tensor, partial_size * elem, owner=pair[0],
                                          after=write_after, private=True, write=True, offset=core * p.topk * elem),)
            core_done[core] = part
            parts.append((core, partial_size, part))
        if p.candidates:
            deps = tuple(dep for _, _, group in parts for dep in group)
        else:
            deps = (lower.access(
                f"qsli:{query}:candidate_block_length", "qsli:candidate_block_length", INT32_BYTES,
                owner=owner, after=owner_done.get(owner, (meta,)), offset=query * INT32_BYTES,
            ),)
        if p.split_parts > 1:
            barrier = lower.stage(f"qsli:{query}:ld_barrier", deps, launch_wide=True)
            reads = []
            for core, count, _ in parts:
                core_reads = tuple(
                    lower.access(f"qsli:{query}:{core}:{tensor}_read", tensor, count * elem, owner=owner,
                                 after=(barrier,), private=True, offset=core * p.topk * elem)
                    for tensor, elem in (("partial_indices", INT32_BYTES), ("partial_values", BF16_BYTES))
                )
                partial_slot_free[core] = core_reads
                reads.extend(core_reads)
            merge_after = tuple(reads) + owner_done.get(owner, ())
            deps = (lower.vector(f"qsli:{query}:LdMergeStage", sum(part[1] for part in parts),
                                 merge_after, owner, "sort_topk"),)
        elif p.candidates and owner in owner_done:
            deps = (*deps, *owner_done[owner])
        if p.topk > p.candidates:
            deps = (lower.vector(f"qsli:{query}:pad_topk", 2 * (p.topk - p.candidates), deps, owner,
                                 index=-1, value="-inf"),)
        owner_done[owner] = tuple(lower.access(
            f"qsli:{query}:{tensor}", f"qsli:{tensor}", p.topk * elem, owner=owner, after=deps,
            write=True, offset=query * p.topk * elem,
        ) for tensor, elem in (("indices", INT32_BYTES), ("values", BF16_BYTES)))


def _mqsmla_head_slices(heads, tile_m, subblocks):
    sub_tile_m = tile_m // DSL_SUBBLOCK_LIMIT
    return tuple(
        (sub, lane, min(heads, sub * sub_tile_m), min(heads, (sub + 1) * sub_tile_m) - min(heads, sub * sub_tile_m))
        for sub, lane in enumerate(subblocks)
    )


def _mqsmla_fd_runs(aic_budget, slot_id, h0, h_sub, d):
    total_slots = aic_budget * MQSMLA_FD_SLOTS_PER_CORE
    fd_ms_base = total_slots * MQSMLA_HEADS * d * FP32_BYTES
    o_base = (slot_id * MQSMLA_HEADS + h0) * d * FP32_BYTES
    o_bytes = h_sub * d * FP32_BYTES
    ms_slot_base = fd_ms_base + slot_id * DSL_SUBBLOCK_LIMIT * MQSMLA_HEADS * FP32_BYTES
    stat_bytes = h_sub * FP32_BYTES
    return (
        AddressRun(o_base, 1, o_bytes, o_bytes),
        AddressRun(ms_slot_base + h0 * FP32_BYTES, 1, stat_bytes, stat_bytes),
        AddressRun(ms_slot_base + (MQSMLA_HEADS + h0) * FP32_BYTES, 1, stat_bytes, stat_bytes),
    )


def _mqsmla_kv_tile(lower, p, ring, name, tile_idx, query, bank, local_pos, rows, sparse, table, subblocks, ready):
    d = p.head_dim
    bank_width = p.n_ori if bank == "ori" else p.n_cmp
    width = KV_ROW_BYTES_ORI if bank == "ori" else KV_ROW_BYTES_CMP
    sparse_base = local_pos if bank == "ori" else p.n_ori + local_pos
    last_sparse = sparse[sparse_base + rows - 1] if sparse else local_pos + rows - 1
    last_page, last_row = divmod(last_sparse, p.page_size)
    last_base = ((table[last_page] if table else last_page) * p.page_size + last_row) * width
    kv_subs = []
    for sub, lane in enumerate(subblocks):
        idx_runs, kv_runs, seen_pages = [], [], []
        for half in range(DSL_SUBBLOCK_LIMIT):
            slice_lo = half * MQSMLA_KV_HALF + sub * MQSMLA_KV_SUB_SLICE
            slice_hi = slice_lo + MQSMLA_KV_SUB_SLICE
            valid_lo = min(slice_lo, rows)
            valid_hi = min(slice_hi, rows)
            valid_count = valid_hi - valid_lo
            if valid_count > 0:
                idx_runs.append(AddressRun(
                    (query * bank_width + local_pos + valid_lo) * INT32_BYTES,
                    1, valid_count * INT32_BYTES, valid_count * INT32_BYTES,
                ))
                for off in range(valid_lo, valid_hi):
                    idx = sparse[sparse_base + off] if sparse else local_pos + off
                    page, row = divmod(idx, p.page_size)
                    seen_pages.append(page)
                    kv_runs.append(AddressRun(((table[page] if table else page) * p.page_size + row) * width,
                                              1, width, width))
            pad_count = MQSMLA_KV_SUB_SLICE - valid_count
            if pad_count > 0:
                idx_runs.append(AddressRun(
                    (query * bank_width + local_pos + rows - 1) * INT32_BYTES, pad_count, INT32_BYTES, 0,
                ))
                seen_pages.append(last_page)
                kv_runs.append(AddressRun(last_base, pad_count, width, 0))
        idx_dep = lower.access(
            f"{name}:sub{sub}:{bank}:indices", f"mqsmla:{bank}:indices", MQSMLA_KV_HALF * INT32_BYTES,
            owner=lane, after=(ready,), runs=tuple(idx_runs),
        )
        words = tuple(AddressRun(page * INT32_BYTES, 1, INT32_BYTES, INT32_BYTES) for page in dict.fromkeys(seen_pages))
        bt_dep = lower.access(
            f"{name}:sub{sub}:{bank}:block_table", f"mqsmla:{bank}:block_table",
            len(words) * INT32_BYTES, after=(idx_dep,), owner=lane, runs=words,
        )
        indexed = lower.vector(f"{name}:sub{sub}:{bank}:pa_addresses", MQSMLA_KV_HALF, (bt_dep,), lane)
        kv_read = lower.access(
            f"{name}:sub{sub}:{bank}", f"mqsmla:{bank}", MQSMLA_KV_HALF * width, owner=lane,
            after=(indexed, ring_credit(ring, tile_idx)), runs=tuple(kv_runs),
        )
        dq = lower.vector(f"{name}:sub{sub}:dequant_kv", MQSMLA_KV_HALF * d, (kv_read,), lane)
        kv_subs.append(lower.local(f"{name}:sub{sub}:ub_to_l1", "mte3", MQSMLA_KV_HALF * d * BF16_BYTES, (dq,), lane))
    return lower.stage(name + ":ub_to_l1", kv_subs)


def _mqsmla_tile_step(lower, p, ring, name, tile_idx, query, tile_spec, is_replay, reset_acc,
                      sparse, ori_table, cmp_table, core, subblocks, head_slices, tile_m, issued_tile,
                      q, state, prev_scores, ready):
    bank, local_pos, rows = tile_spec
    h, d = p.heads, p.head_dim
    sub_tile_m = tile_m // DSL_SUBBLOCK_LIMIT
    n = f"{name}:kv:{tile_idx}"
    table = ori_table if bank == "ori" else cmp_table
    kv = _mqsmla_kv_tile(lower, p, ring, n, tile_idx, query, bank, local_pos, rows, sparse, table, subblocks, ready)
    publish = _read_ring(ring, tile_idx, kv, n + ":pv")
    logical_tile = 0 if is_replay else 2 * h * rows * d
    qk = lower.cube(
        n + ":qk", logical_tile, (q, publish, *prev_scores), core, issued=issued_tile,
        operands=(tile_m * d * BF16_BYTES, MQSMLA_KV_TILE * d * BF16_BYTES),
    )
    score_fixes, prob_subs, sms = [], [], []
    for sub, lane, _, h_sub in head_slices:
        fix = lower.local(f"{n}:sub{sub}:scores", "fixpipe", sub_tile_m * MQSMLA_KV_TILE * FP32_BYTES, (qk,), core)
        score_fixes.append(fix)
        prev_sub = (state[sub],) if len(state) == len(head_slices) else state
        sm_ops = max(1, h_sub) * rows * MQSMLA_ONLINE_SCORE_OPS
        sm = lower.vector(f"{n}:sub{sub}:online_softmax", sm_ops, (*prev_sub, fix), lane,
                          replay=is_replay, reset_accumulator=reset_acc)
        sms.append(sm)
        prob_subs.append(lower.local(f"{n}:sub{sub}:p_l1", "mte3", sub_tile_m * MQSMLA_KV_TILE * BF16_BYTES, (sm,), lane))
    pv = lower.cube(
        n + ":pv", logical_tile, (publish, *prob_subs), core, issued=issued_tile,
        operands=(tile_m * MQSMLA_KV_TILE * BF16_BYTES, MQSMLA_KV_TILE * d * BF16_BYTES),
    )
    next_state = []
    for sub, lane, _, h_sub in head_slices:
        pv_fix = lower.local(f"{n}:sub{sub}:pv_to_ub", "fixpipe", sub_tile_m * d * FP32_BYTES, (pv,), core)
        acc_ops = max(1, h_sub) * d * MQSMLA_ONLINE_VALUE_OPS
        next_state.append(lower.vector(f"{n}:sub{sub}:accumulate", acc_ops, (pv_fix, sms[sub]), lane,
                                       replay=is_replay, reset_accumulator=reset_acc))
    return tuple(next_state), tuple(score_fixes)


def _lower_mqsmla(lower, p, ready):
    spec = lower.spec
    h, d = p.heads, p.head_dim
    tile_m = mqsmla_tile_m(h)
    issued_tile = 2 * tile_m * MQSMLA_KV_TILE * d
    partitions = mqsmla_tile_partitions(p)
    all_tiles = tuple(tile for pt in partitions for tile in pt)
    part_bounds, cursor = [], 0
    for pt in partitions:
        part_bounds.append((cursor, cursor + len(pt)))
        cursor += len(pt)
    deferred, partial_publications = [], []
    ori_table, cmp_table = p.block_tables
    for query in range(p.queries):
        sparse = p.sparse_rows(query)
        partials = []
        for part, (part_start, part_end) in enumerate(part_bounds):
            core = (query * p.split_parts + part) % spec.aic_budget
            subblocks = lower.paired_subblocks(core)
            head_slices = _mqsmla_head_slices(h, tile_m, subblocks)
            name = f"mqsmla:{query}:{part}"
            q = lower.access(name + ":q", "mqsmla:q", h * d * BF16_BYTES, after=(ready,), owner=core, cube=True,
                             offset=query * h * d * BF16_BYTES)
            sink_tensor = "mqsmla:sink" if p.has_sink else "sink_sentinel"
            sink_suffix = "sink" if p.has_sink else "sink_sentinel"
            sinks = tuple(
                lower.access(
                    f"{name}:sub{sub}:{sink_suffix}", sink_tensor, h_sub * FP32_BYTES,
                    after=(ready,), owner=lane, offset=h0 * FP32_BYTES, private=not p.has_sink,
                )
                for sub, lane, h0, h_sub in head_slices if h_sub > 0
            )
            state, prev_scores = (q, *sinks), ()
            ring = RingBuffer(die=lower.plan.die, owner=core, pool="l1", allocation=name + ":kv_l1", depth=MQSMLA_KV_DEPTH)
            run_tiles = all_tiles[:part_end] if (p.split_parts > 1 and part_end > part_start) else all_tiles[part_start:part_end]
            for i, tile_spec in enumerate(run_tiles):
                is_replay = i < part_start
                reset_acc = part > 0 and i == part_start
                state, prev_scores = _mqsmla_tile_step(
                    lower, p, ring, name, i, query, tile_spec, is_replay, reset_acc,
                    sparse, ori_table, cmp_table, core, subblocks, head_slices, tile_m, issued_tile,
                    q, state, prev_scores, ready,
                )
            lower.plan.stages.extend(ring.to_stages())
            if p.split_parts > 1:
                slot = (query * p.split_parts + part) // spec.aic_budget
                slot_id = core * MQSMLA_FD_SLOTS_PER_CORE + slot
                part_writes = []
                for sub, lane, h0, h_sub in head_slices:
                    if h_sub <= 0:
                        continue
                    sub_bytes = (h_sub * d + 2 * h_sub) * FP32_BYTES
                    runs = _mqsmla_fd_runs(spec.aic_budget, slot_id, h0, h_sub, d)
                    after_sub = (state[sub],) if len(state) == len(head_slices) else state
                    part_writes.append(lower.access(
                        f"{name}:sub{sub}:fd_partial_write", "fd_workspace", sub_bytes, after=after_sub,
                        owner=lane, private=True, write=True, working_set=spec.workspace_bytes_total, runs=runs,
                    ))
                partials.append((tuple(part_writes), slot_id))
                partial_publications.extend(part_writes)
            else:
                partials.append((state, core))
        deferred.append(tuple(partials))
    barrier = lower.stage("mqsmla:fd_barrier", partial_publications, launch_wide=True) if partial_publications else None
    for query, partials in enumerate(deferred):
        reduce_core = query % spec.aic_budget if barrier is not None else partials[0][1]
        subblocks = lower.paired_subblocks(reduce_core)
        head_slices = _mqsmla_head_slices(h, tile_m, subblocks)
        for sub, lane, h0, h_sub in head_slices:
            if h_sub <= 0:
                continue
            if barrier is not None:
                sub_bytes = (h_sub * d + 2 * h_sub) * FP32_BYTES
                reads = tuple(
                    lower.access(
                        f"mqsmla:{query}:sub{sub}:fd_partial_read:{part}", "fd_workspace", sub_bytes,
                        after=(barrier,), owner=lane, private=True, working_set=spec.workspace_bytes_total,
                        runs=_mqsmla_fd_runs(spec.aic_budget, slot_id, h0, h_sub, d),
                    )
                    for part, (_, slot_id) in enumerate(partials)
                )
                done = (lower.vector(
                    f"mqsmla:{query}:sub{sub}:fd_reduce", p.split_parts * (h_sub * d + 2 * h_sub),
                    reads, lane, "vec_reduce",
                ),)
            else:
                unsplit_state = partials[0][0]
                done = (unsplit_state[sub],) if len(unsplit_state) == len(head_slices) else unsplit_state
            lower.access(
                f"mqsmla:{query}:sub{sub}:output", "mqsmla:output", h_sub * d * BF16_BYTES, after=done,
                owner=lane, write=True, offset=(query * h + h0) * d * BF16_BYTES,
            )


_LOWER = {FLASH_KDA: _lower_flash, MEGA_KDA: _lower_mega, COMMIT_KDA: _lower_commit, MLA_PROLOG: _lower_mla,
          ATTN_RES: _lower_attn_res, QSLI: _lower_qsli, MQSMLA: _lower_mqsmla}


def lower_dsl_spec(spec, problem, datapath, *, context=None, memory_view=None, die=0,
                   live_admission=False, region_id=None):
    """Verify and execute the caller's selected K3 specialization."""
    name, p = dsl_problem(spec.kernel_id, problem)
    expected, solution = describe_dsl_kernel(
        spec.kernel_id, p, datapath, aic_budget=spec.aic_budget, aiv_budget=spec.aiv_budget,
        memory_view=memory_view, die=die,
    )
    if expected != spec:
        raise ModelingError(f"dsl_spec_does_not_match_problem:{spec.kernel_id}")
    plan = Lowering(datapath, context=context, memory_view=memory_view, die=die, live_admission=live_admission)
    rid = region_id or name
    if name in {ATTN_RES, MLA_PROLOG} and p.tokens == 0:
        return plan.result("K3", rid, (rid,), spec, (), solution, launches=0)
    lower = _DslLowering(plan, spec, rid)
    ready = lower.launch(p)
    _LOWER[name](lower, p, ready)
    return plan.result("K3", rid, (rid,), spec, lower.semantic(), solution, launches=1 + spec.uses_aicpu_metadata)
