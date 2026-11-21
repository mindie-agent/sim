"""Executable SRAM, GM and synchronization schedules for the DSL reference ABIs.

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
from mindie_sim.schema.common import ModelingError

from .contracts import SemanticWorkItem, ceil_div
from .dsl_catalog import (
    ATTN_RES, ATTN_RES_VL, BF16_BYTES, COMMIT_KDA, FLASH_KDA, FP32_BYTES, INT32_BYTES,
    KDA_CHUNK, KDA_HEAD_DIM, KDA_HIDDEN_SIZE, KDA_MODEL_HEADS, KV_ROW_BYTES_CMP, KV_ROW_BYTES_ORI,
    MEGA_KDA, MLA_CACHE_NZ_LANE_BYTES, MLA_PROLOG, MLA_QUANTIZED_QUERY, MQSMLA, MQSMLA_FD_BYTES_PER_CORE, MQSMLA_FD_SLOT_BYTES,
    MQSMLA_KV_DEPTH, MQSMLA_KV_TILE, QSLI, QSLI_HISTOGRAM_BINS, QSLI_METADATA_BYTES,
    QSLI_PACKED_D, QSLI_SCALE_ROW_BYTES, QSLI_STAGING_DEPTH, QSLI_TILE_M, QSLI_TILE_N,
    QSLI_TRUNK, QkvQuantMode, commit_rows, describe_dsl_kernel,
    dsl_problem, flash_kda_lane_tasks, mega_workspace_sizes,
    mla_workspaces, qsli_workspace_sizes, replay_pool_capacity, replay_pool_strides,
)
from .fuse_region import Lowering, ring_credit

KDA_INVERSE_PRODUCTS = 5
KDA_INPUT_VECTORS = 4
KDA_STATE_UPDATE_OPS = 3
KDA_RECURRENT_STATE_OPS = 7
KDA_CONV_TAPS = 3
MLA_CACHE_INDEX_BYTES = 8
ATTN_RES_MERGE_OPS = 5
ATTN_RES_NORM_OPS = 4
ATTN_RES_SCORE_OPS = 5
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

    def cube(self, name, logical, after, owner=0, *, dtype="bf16", issued=None, operands=None):
        if operands is not None:
            after = tuple(self.local(f"{name}:l0{bank}", "mte1", size, after, owner)
                          for bank, size in zip(("a", "b"), operands))
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


def _lower_flash(lower, p, ready):
    plan, spec = lower.plan, lower.spec
    d = KDA_HEAD_DIM
    state_bytes = d * d * FP32_BYTES
    staged = spec.template_variant == "staged"
    buffers, lane_last, lane_generation, prior_chunk = {}, {}, Counter(), {}
    previous_round = (ready,)
    token_base, chunk_base = [0], [0]
    for length in p.lengths:
        token_base.append(token_base[-1] + length)
        chunk_base.append(chunk_base[-1] + ceil_div(length, KDA_CHUNK))
    for round_tasks in flash_kda_lane_tasks(p, spec):
        scheduled = {(batch, head, chunk) for batch, head, chunk, _, _ in round_tasks} if staged else set()
        for batch, head, chunk, core, slot in round_tasks:
            vector = lower.paired(core)[0]
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
            row = token_base[batch] + chunk * KDA_CHUNK
            deps = tuple(dict.fromkeys((*previous_round, lane_last.get(core, ready))))
            query_head = head // (p.value_heads // p.query_heads)
            inputs = (("q", query_head, d * BF16_BYTES), ("k", query_head, d * BF16_BYTES),
                      ("v", head, d * BF16_BYTES), ("g", head, d * BF16_BYTES), ("beta", head, FP32_BYTES))
            loads = tuple(lower.access(
                f"{name}:{tensor}", f"kda:{tensor}:{tensor_head}", tail * row_bytes,
                owner=vector, after=deps, offset=row * row_bytes,
            ) for tensor, tensor_head, row_bytes in inputs)
            norm = lower.vector(name + ":norm_gate", tail * d * KDA_INPUT_VECTORS, loads, vector)
            moves = lower.local(name + ":operands_to_l1", "mte3", KDA_INPUT_VECTORS * tail * d * BF16_BYTES,
                                (norm,), vector)
            scores = lower.cube(name + ":kkt_qkt", 2 * 2 * tail * tail * d, (moves,), core)
            score_ub = lower.local(name + ":scores_to_ub", "fixpipe", 2 * tail * tail * FP32_BYTES, (scores,), core)
            inverse = lower.cube(
                name + ":wy_inverse", 2 * KDA_INVERSE_PRODUCTS * tail * tail * tail, (score_ub,), core,
                dtype="fp16", issued=2 * KDA_INVERSE_PRODUCTS * KDA_CHUNK * KDA_CHUNK * KDA_CHUNK,
            )
            wy = lower.cube(name + ":kcd_u", 2 * 2 * tail * tail * d, (inverse,), core)
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
            entry = prior_chunk.get(seq, ready)
            continued = staged and chunk > 0 and (*seq, chunk - 1) not in scheduled
            if chunk == 0 and p.has_initial_state:
                entry = lower.access(name + ":initial_state", "kda:initial_state", state_bytes,
                                     owner=vector, after=deps, offset=state_offset)
            elif continued:
                entry = lower.access(name + ":reload_h", h_tensor, state_bytes, owner=vector, after=deps,
                                     offset=h_offset, version=int(not private_h), private=private_h,
                                     bypass=True, obligation=False)
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
                deps = (q, gamma, *wk, aqk, entry)
            else:
                deps = (wy, entry, ring_credit(ring, generation))
            state = lower.local(name + ":state_to_l1", "mte3", d * d * BF16_BYTES, deps, vector)
            if ring is not None:
                state = _read_ring(ring, generation, state, name + ":state_products")
            inter = lower.cube(name + ":state_products", 2 * 2 * tail * d * d, (state,), core)
            delta = lower.cube(name + ":delta", 2 * tail * d * d, (inter,), core)
            delta_ub = lower.local(name + ":delta_to_ub", "fixpipe", state_bytes, (delta,), core)
            update_deps = (delta_ub,)
            if (staged or p.save_chunk_states) and not continued:
                update_deps += (lower.access(name + ":save_h", h_tensor, state_bytes,
                                             after=(state,), owner=vector, offset=h_offset, write=True, bypass=True,
                                             private=private_h),)
            update = lower.vector(name + ":fp32_state_update", d * d * 2, update_deps, vector)
            intra = lower.cube(name + ":intra", 2 * tail * tail * d, (inter,), core)
            output = lower.access(name + ":output", f"kda:head:{head}:output", tail * d * BF16_BYTES,
                                  offset=row * d * BF16_BYTES, after=(intra,), cube=True, owner=core, write=True)
            epilogue = (output, update, *saved)
            final = chunk + 1 == ceil_div(p.lengths[batch], KDA_CHUNK)
            if final and p.return_final_state:
                epilogue += (lower.access(name + ":final_state", "kda:final_state", state_bytes,
                                          after=(update,), owner=vector, offset=state_offset, write=True),)
            elif staged and not final and (*seq, chunk + 1) not in scheduled:
                update = lower.access(name + ":carry_h", h_tensor, state_bytes, after=(update,), owner=vector,
                                      offset=h_offset + state_bytes, private=private_h, write=True, bypass=True)
                epilogue += (update,)
            prior_chunk[seq] = update
            lane_last[core] = lower.stage(name + ":done", epilogue)
        if staged:
            previous_round = tuple(lane_last.values())
    for ring in buffers.values():
        plan.stages.extend(ring.to_stages())


def _lower_mega(lower, p, ready):
    spec = lower.spec
    h = KDA_MODEL_HEADS // p.tp
    d, batch = KDA_HEAD_DIM, p.batch * p.tp
    workspaces = mega_workspace_sizes(p)
    capacity = replay_pool_capacity(p.replay_capacity, p.sequence)
    u_stride, k_stride, decay_stride, checkpoint_stride = replay_pool_strides(p, h, capacity)
    if not p.unfused_launches or len(set(p.unfused_launches)) != len(p.unfused_launches):
        raise ModelingError("mega_kda_unfused_launch_baseline_invalid")
    ready = lower.stage("mega:unfused_baseline", (ready,), saved_launches=len(p.unfused_launches) - 1)
    conv_width = (2 + 1) * h * d
    taps = lower.access("mega:conv_weight", "mega:w_conv", KDA_CONV_TAPS * conv_width * BF16_BYTES, after=(ready,))
    lane_last = {}
    for tile in range(batch):
        core, vector = tile % spec.aic_budget, tile % spec.aiv_budget
        slot = p.state_indices[tile] if p.state_indices else tile
        name = f"mega:{tile}"
        prior = (lane_last.get(core, ready),)
        hidden = lower.access(name + ":hidden", "hidden_states", p.sequence * KDA_HIDDEN_SIZE * BF16_BYTES,
                              cube=True, owner=core, after=prior, offset=tile * p.sequence * KDA_HIDDEN_SIZE * BF16_BYTES)
        conv_bytes = (KDA_CONV_TAPS - 1) * conv_width * BF16_BYTES
        conv_history = lower.access(name + ":conv_read", "mega:conv_state", conv_bytes, owner=vector, after=prior,
                                    offset=slot * (p.sequence + KDA_CONV_TAPS - 1) * conv_width * BF16_BYTES)
        decay_weight = lower.access(name + ":decay_a_weight", "mega:w_decay_a",
                                    KDA_HIDDEN_SIZE * d * BF16_BYTES, cube=True, owner=core, after=(ready,))
        decay_a = lower.cube(name + ":decay_a", 2 * p.sequence * KDA_HIDDEN_SIZE * d, (hidden, decay_weight), core)
        decay_input = lower.spill(
            name + ":fa_workspace", "fa", p.sequence * d * BF16_BYTES, (decay_a,), owner=core,
            offset=tile * p.sequence * d * BF16_BYTES, working_set=workspaces["fa"],
            cube_src=True, cube_dst=True,
        )
        projected = {}
        for tensor in ("Q", "K", "V", "decay", "beta", "output_gate"):
            width = 1 if tensor == "beta" else d
            in_width = d if tensor == "decay" else KDA_HIDDEN_SIZE
            weight_bytes = in_width * h * width * BF16_BYTES
            load = lower.access(f"{name}:{tensor}:weight", f"mega:w_{tensor}", weight_bytes,
                                cube=True, owner=core, after=(ready,))
            operand = decay_input if tensor == "decay" else hidden
            projection = lower.cube(f"{name}:{tensor}:projection", 2 * p.sequence * in_width * h * width,
                                    (operand, load), core)
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
        conv_stride = conv_bytes + p.sequence * conv_width * BF16_BYTES
        conv_out = lower.access(name + ":conv_write", "mega:conv_state", conv_stride, owner=vector,
                                after=tuple(projected[t] for t in ("Q", "K", "V")), offset=slot * conv_stride, write=True)
        state_size = h * d * d * FP32_BYTES
        indices = lower.access(name + ":state_index", "state_indices", INT32_BYTES,
                               owner=vector, after=prior, offset=tile * INT32_BYTES)
        state = lower.access(name + ":state_read", "checkpoint", state_size,
                             owner=vector, after=(indices,), offset=slot * checkpoint_stride * FP32_BYTES)
        rec = lower.vector(name + ":recurrent", p.sequence * h * (d * d * KDA_RECURRENT_STATE_OPS + 2 * d),
                           (state, *(dep for tensor, dep in projected.items() if tensor != "output_gate")), vector)
        records = []
        for (tensor, element), pool_stride in zip(_REPLAY_DTYPES, (u_stride, k_stride, decay_stride), strict=True):
            row_bytes = d * element
            stride = capacity * row_bytes
            for token in range(p.sequence):
                offset = slot * pool_stride * element + token * row_bytes
                records.append(lower.access(f"{name}:{tensor}:{token}:write", tensor, h * row_bytes, after=(rec,),
                                            owner=vector, write=True, runs=(AddressRun(offset, h, row_bytes, stride),)))
        gated = lower.vector(name + ":gate", p.sequence * h * d, (rec, projected["output_gate"]), vector)
        payload = p.sequence * h * d * BF16_BYTES
        handoff = lower.access(name + ":gate_workspace", "Q", payload, owner=vector, after=(gated,),
                               offset=tile * payload, private=True, write=True, working_set=workspaces["Q"])
        publish = lower.stage(name + ":publish_vector_grid_to_cube", (handoff,), gm_workspace_handoff=True)
        operand = lower.access(name + ":read_gated_workspace", "Q", payload, cube=True, owner=core,
                               after=(publish,), offset=tile * payload, private=True, working_set=workspaces["Q"])
        weight = lower.access(name + ":out_weight", "mega:w_out", h * d * KDA_HIDDEN_SIZE * BF16_BYTES,
                              cube=True, owner=core, after=(ready,))
        result = lower.cube(name + ":output_projection", 2 * p.sequence * h * d * KDA_HIDDEN_SIZE,
                            (operand, weight), core)
        output = p.sequence * KDA_HIDDEN_SIZE * BF16_BYTES
        out = lower.access(name + ":output", "mega:output", output, after=(result,), cube=True, owner=core,
                           write=True, offset=tile * output)
        lane_last[core] = lower.stage(name + ":done", (conv_out, out, *records))


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


def _lower_mla(lower, p, ready):
    spec = lower.spec
    mode = p.quant_mode
    workspace, _, scratch_rows, recompute = mla_workspaces(p, spec.aic_budget)
    qp = p.heads * (p.q_nope + p.rope)
    latent = p.kvr - p.rope
    if latent < 1:
        raise ModelingError("mla_prolog_latent_width_invalid")
    weights = {"wqa": p.dim * p.q_lora, "wkva": p.dim * p.kvr, "wqb": p.q_lora * qp,
               "wkb": p.heads * p.q_nope * latent}
    inputs = {"wqa": p.dim, "wkva": p.dim, "wqb": p.q_lora, "wkb": p.heads * p.q_nope}
    weight_dep = {
        weight: lower.access(weight + ":load", weight, elements * BF16_BYTES, after=(ready,), cube=True)
        for weight, elements in weights.items()
    }

    def project(name, weight, rows, after, core):
        operands = (rows * inputs[weight] * BF16_BYTES, weights[weight] * BF16_BYTES)
        return lower.cube(name, 2 * rows * weights[weight], (*after, weight_dep[weight]), core, operands=operands)

    cache_row = p.cache_row_bytes
    quant_rows = []
    maxima_by_core = {core: [] for core in range(spec.aic_budget)}
    previous = (ready,)
    for chunk, start in enumerate(range(0, p.tokens, scratch_rows)):
        rows = min(scratch_rows, p.tokens - start)
        name = f"mla:{chunk}"
        core, vector = chunk % spec.aic_budget, chunk % spec.aiv_budget
        x = lower.access(name + ":x", "mla:x", rows * p.dim * BF16_BYTES, after=previous,
                         offset=start * p.dim * BF16_BYTES, cube=True, owner=core)
        if "ws_x_pad" in workspace:
            x = lower.spill(name + ":pad", "ws_x_pad", workspace["ws_x_pad"], (x,), src_owner=vector,
                            dst_owner=core, cube_dst=True, working_set=workspace["ws_x_pad"])
        qa = project(name + ":wqa", "wqa", rows, (x,), core)
        kva = project(name + ":wkva", "wkva", rows, (x,), core)
        qa_read = lower.spill(name + ":ws_qa", "ws_qa", workspace["ws_qa"], (qa,),
                              src_owner=core, dst_owner=vector, cube_src=True, working_set=workspace["ws_qa"])
        kval = lower.spill(name + ":ws_kva", "ws_kva", workspace["ws_kva"], (kva,),
                          src_owner=core, dst_owner=vector, cube_src=True, working_set=workspace["ws_kva"])
        norm = lower.vector(name + ":q_rms", rows * p.q_lora * ATTN_RES_NORM_OPS, (qa_read,), vector)
        qa_bytes = rows * p.q_lora * BF16_BYTES
        if p.return_qr:
            norm = lower.access(name + ":qr", "mla:qr", qa_bytes, after=(norm,), owner=vector,
                                write=True, offset=start * p.q_lora * BF16_BYTES)
            norm = lower.access(name + ":qr_for_wqb", "mla:qr", qa_bytes, after=(norm,), owner=core,
                                cube=True, offset=start * p.q_lora * BF16_BYTES, version=1, obligation=False)
        elif workspace["ws_qb_aux"]:
            norm = lower.spill(name + ":ws_qb_aux", "ws_qb_aux", qa_bytes, (norm,), src_owner=vector,
                               dst_owner=core, cube_dst=True, working_set=workspace["ws_qb_aux"])
        kv_norm = lower.vector(name + ":kv_norm_rope", rows * p.kvr * ATTN_RES_NORM_OPS, (kval,), vector)
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
        qb = project(name + ":wqb", "wqb", rows, (norm, cache), core)
        proj = lower.spill(name + ":ws_q_proj", "ws_q_proj", workspace["ws_q_proj"], (qb,),
                           owner=core, cube_src=True, cube_dst=True, working_set=workspace["ws_q_proj"])
        kb = project(name + ":wkb", "wkb", rows, (proj,), core)
        q_elements = rows * p.heads * p.kvr
        rope = lower.vector(name + ":q_rope", rows * p.heads * p.rope * ATTN_RES_NORM_OPS, (kb,), vector)
        if mode in MLA_QUANTIZED_QUERY:
            size = q_elements * FP32_BYTES
            offset = 0 if mode == QkvQuantMode.FP8_TENSOR or recompute else start * p.heads * p.kvr * FP32_BYTES
            q_value = lower.access(name + ":ws_q_fp_write", "ws_q_fp", size, after=(rope,),
                                   owner=vector, offset=offset, private=True, write=True, working_set=workspace["ws_q_fp"])
            if mode == QkvQuantMode.HiF8:
                active = min(q_elements, spec.aic_budget)
                base, tail = divmod(q_elements, active)
                for worker in range(active):
                    lane = worker % spec.aiv_budget
                    maxima_by_core[worker].append(lower.vector(
                        f"{name}:partial_max:{worker}", base + (worker < tail), (q_value,), lane, "vec_reduce",
                    ))
                quant_rows.append((name, core, vector, start, rows, q_elements, offset, size))
                previous = tuple(values[-1] for values in maxima_by_core.values() if values)
                continue
            q_value = lower.access(name + ":ws_q_fp_read", "ws_q_fp", size, after=(q_value,),
                                   owner=vector, private=True, working_set=workspace["ws_q_fp"])
            rope = lower.vector(name + ":q_quant", q_elements * 2, (q_value,), vector)
        elem = 1 if mode == QkvQuantMode.FP8_TENSOR else BF16_BYTES
        output = lower.access(name + ":q_output", "mla:q", q_elements * elem, after=(rope,), write=True,
                              offset=start * p.heads * p.kvr * elem, owner=vector)
        previous = (output,)
    if not quant_rows:
        return
    max_events = []
    for worker, events in maxima_by_core.items():
        if not events:
            continue
        lane = worker % spec.aiv_budget
        maximum = lower.vector(f"mla:merge_core_max:{worker}", len(events) - 1, events, lane, "vec_reduce")
        max_events.append(lower.access(
            f"mla:ws_amax_write:{worker}", "ws_amax", FP32_BYTES, after=(maximum,), owner=lane,
            private=True, write=True, offset=worker * FP32_BYTES,
        ))
    barrier = lower.stage("q_global_quant_launch_barrier", max_events, launch_wide=True)
    maxima = lower.access("q_global_quant_amax_read", "ws_amax", workspace["ws_amax"],
                          after=(barrier,), private=True)
    for name, core, vector, start, rows, elements, offset, size in quant_rows:
        after = (maxima,)
        if recompute:
            qr = lower.access(name + ":recompute_qr", "mla:qr" if p.return_qr else "ws_qb_aux",
                              rows * p.q_lora * BF16_BYTES, after=after, cube=True, owner=core,
                              offset=start * p.q_lora * BF16_BYTES, private=not p.return_qr)
            qb = project(name + ":wqb_recompute", "wqb", rows, (qr,), core)
            kb = project(name + ":wkb_recompute", "wkb", rows, (qb,), core)
            after = (kb,)
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


def _lower_qsli(lower, p, ready):
    spec = lower.spec
    workspace = qsli_workspace_sizes(p, spec.aic_budget)
    meta = lower.access("qsli:metadata_load", "qsli:metadata", QSLI_METADATA_BYTES, after=(ready,), private=True)
    q_payload = QSLI_PACKED_D + QSLI_SCALE_ROW_BYTES
    query_ready = (meta,)
    for query in range(p.query_rows):
        owner = query % spec.aiv_budget
        parts = []
        for split in range(p.split_parts if p.candidates else 0):
            core = (query * p.split_parts + split) % spec.aic_budget
            pair = lower.paired(core)
            name = f"qsli:{query}:{split}"
            first = p.candidates * split // p.split_parts
            end = p.candidates * (split + 1) // p.split_parts
            q = lower.access(name + ":query", "qsli:query", QSLI_TILE_M * q_payload, cube=True, owner=core,
                             after=query_ready, offset=query * QSLI_TILE_M * q_payload)
            q_l0 = lower.local(name + ":query_l0a", "mte1", QSLI_TILE_M * q_payload, (q,), core)
            k_l1, l0b, l0c, staging = (
                RingBuffer(die=lower.plan.die, owner=core, pool=pool, depth=depth, allocation=f"{name}:{tag}")
                for tag, pool, depth in (("k_l1", "l1", 2), ("l0b", "l0b", 2),
                                         ("l0c", "l0c", 2), ("staging", "gm", QSLI_STAGING_DEPTH))
            )
            scores = []
            for tile, pos in enumerate(range(first, end, QSLI_TILE_N)):
                real = min(QSLI_TILE_N, end - pos)
                n = f"{name}:tile{tile}"
                gather = lower.access(n + ":gather_pa", "qsli:key", real * q_payload, owner=pair[0],
                                      after=(*query_ready, ring_credit(staging, tile)), offset=pos * q_payload)
                staging_slot = (core * QSLI_STAGING_DEPTH + tile % QSLI_STAGING_DEPTH) * QSLI_TILE_N
                packed = lower.spill(n + ":publish_packed_k", "staging_key", real * QSLI_PACKED_D, (gather,),
                                     src_owner=pair[0], dst_owner=core, cube_dst=True,
                                     offset=staging_slot * QSLI_PACKED_D, working_set=workspace["staging_key"])
                scale = lower.spill(n + ":publish_scale", "staging_scale", real * QSLI_SCALE_ROW_BYTES, (gather,),
                                    src_owner=pair[-1], dst_owner=core, cube_dst=True,
                                    offset=staging_slot * QSLI_SCALE_ROW_BYTES, working_set=workspace["staging_scale"])
                staged = lower.stage(n + ":staged_key_scale", (packed, scale, ring_credit(k_l1, tile)))
                staged = _read_ring(staging, tile, staged, n + ":l0_operands")
                refill = _read_ring(k_l1, tile, staged, n + ":l0_operands")
                k_l0 = lower.local(n + ":l0_operands", "mte1", real * q_payload,
                                   (refill, ring_credit(l0b, tile)), core)
                operand = _read_ring(l0b, tile, k_l0, n + ":QsliCube")
                logical = 2 * QSLI_TILE_M * real * KDA_HEAD_DIM
                score = lower.cube(n + ":QsliCube", logical, (q_l0, operand, ring_credit(l0c, tile)), core,
                                   dtype="mxfp4", issued=2 * QSLI_TILE_M * QSLI_TILE_N * KDA_HEAD_DIM)
                score = _read_ring(l0c, tile, score, n + ":fix_done")
                fixes = []
                for sub, lane in enumerate(pair[:2]):
                    count = (real + (sub == 0)) // 2
                    fix = lower.plan.work(f"{n}:fix_subblock{sub}", "fixpipe", QSLI_TILE_M * count * FP32_BYTES,
                                          after=(score,), owner=core, metadata={"dst_subblock": sub})
                    fixes.append(fix)
                    scores.append(lower.vector(f"{n}:score_reduce{sub}", QSLI_TILE_M * count, (fix,), lane, "vec_reduce"))
                lower.stage(n + ":fix_done", fixes)
            for ring in (k_l1, l0b, l0c, staging):
                lower.plan.add_ring(ring, after=query_ready)
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
                for tensor, elem in (("partial_indices", INT32_BYTES), ("partial_values", BF16_BYTES)):
                    part += (lower.access(f"{name}:{tensor}", tensor, partial_size * elem, owner=pair[0],
                                          after=(positions,), private=True, write=True, offset=core * p.topk * elem),)
            parts.append((core, partial_size, part))
        deps = tuple(dep for _, _, group in parts for dep in group) if p.candidates else query_ready
        if p.split_parts > 1:
            barrier = lower.stage(f"qsli:{query}:ld_barrier", deps, launch_wide=True)
            reads = [
                lower.access(f"qsli:{query}:{core}:{tensor}_read", tensor, count * elem, owner=owner,
                             after=(barrier,), private=True, offset=core * p.topk * elem)
                for core, count, _ in parts
                for tensor, elem in (("partial_indices", INT32_BYTES), ("partial_values", BF16_BYTES))
            ]
            deps = (lower.vector(f"qsli:{query}:LdMergeStage", sum(part[1] for part in parts),
                                 tuple(reads), owner, "sort_topk"),)
        if p.topk > p.candidates:
            deps = (lower.vector(f"qsli:{query}:pad_topk", 2 * (p.topk - p.candidates), deps, owner,
                                 index=-1, value="-inf"),)
        query_ready = tuple(lower.access(
            f"qsli:{query}:{tensor}", f"qsli:{tensor}", p.topk * elem, owner=owner, after=deps,
            write=True, offset=query * p.topk * elem,
        ) for tensor, elem in (("indices", INT32_BYTES), ("values", BF16_BYTES)))


def _lower_mqsmla(lower, p, ready):
    spec = lower.spec
    total = p.n_ori + p.n_cmp
    h, d = p.heads, p.head_dim
    sizes = p.part_lengths or tuple(
        total * (part + 1) // p.split_parts - total * part // p.split_parts
        for part in range(p.split_parts)
    )
    deferred = []
    partial_publications = []
    ori_table, cmp_table = p.block_tables
    for query in range(p.queries):
        sparse = p.sparse_rows(query)
        start, partials = 0, []
        for part, size in enumerate(sizes):
            core = (query * p.split_parts + part) % spec.aic_budget
            vector = lower.paired(core)[0]
            name = f"mqsmla:{query}:{part}"
            q = lower.access(name + ":q", "mqsmla:q", h * d * BF16_BYTES, after=(ready,), owner=core, cube=True,
                             offset=query * h * d * BF16_BYTES)
            state = (q, lower.access(name + ":sink", "mqsmla:sink", h * FP32_BYTES, after=(ready,), owner=vector))
            ring = RingBuffer(die=lower.plan.die, owner=core, pool="l1", allocation=name + ":kv_l1", depth=MQSMLA_KV_DEPTH)
            for i, pos in enumerate(range(start, start + size, MQSMLA_KV_TILE)):
                n = f"{name}:kv:{i}"
                end = min(pos + MQSMLA_KV_TILE, start + size)
                rows = end - pos
                ori, cmp = max(0, min(end, p.n_ori) - pos), max(0, end - max(pos, p.n_ori))
                reads = []
                for bank, count, base, width, table in (
                    ("ori", ori, pos, KV_ROW_BYTES_ORI, ori_table),
                    ("cmp", cmp, max(pos, p.n_ori), KV_ROW_BYTES_CMP, cmp_table),
                ):
                    if not count:
                        continue
                    first = base if bank == "ori" else base - p.n_ori
                    indices = sparse[base:base + count] if sparse else range(first, first + count)
                    deps = (lower.access(f"{n}:{bank}:indices", f"mqsmla:{bank}:indices", count * INT32_BYTES,
                                         owner=vector, after=(ready,), offset=(query * total + base) * INT32_BYTES),)
                    pages = tuple(divmod(index, p.page_size) for index in indices)
                    words = tuple(AddressRun(page * INT32_BYTES, 1, INT32_BYTES, INT32_BYTES)
                                  for page in dict.fromkeys(page for page, _ in pages))
                    deps = (lower.access(f"{n}:{bank}:block_table", f"mqsmla:{bank}:block_table",
                                         len(words) * INT32_BYTES, after=deps, owner=vector, runs=words),)
                    runs = tuple(AddressRun(((table[page] if table else page) * p.page_size + row) * width,
                                            1, width, width) for page, row in pages)
                    indexed = lower.vector(f"{n}:{bank}:pa_addresses", count, deps, vector)
                    reads.append(lower.access(f"{n}:{bank}", f"mqsmla:{bank}", count * width, owner=vector,
                                              after=(indexed, ring_credit(ring, i)), runs=runs))
                dq = lower.vector(n + ":dequant_kv", rows * d, reads, vector)
                kv = lower.local(n + ":ub_to_l1", "mte3", rows * d * BF16_BYTES, (dq,), vector)
                publish = _read_ring(ring, i, kv, n + ":pv")
                qk = lower.cube(n + ":qk", 2 * h * rows * d, (q, publish), core)
                fix = lower.local(n + ":scores", "fixpipe", h * rows * FP32_BYTES, (qk,), core)
                sm = lower.vector(n + ":online_softmax", h * rows * MQSMLA_ONLINE_SCORE_OPS, (*state, fix), vector)
                probs = lower.local(n + ":p_l1", "mte3", h * rows * BF16_BYTES, (sm,), vector)
                pv = lower.cube(n + ":pv", 2 * h * rows * d, (publish, probs), core)
                state = (lower.vector(n + ":accumulate", h * d * MQSMLA_ONLINE_VALUE_OPS, (pv,), vector),)
            lower.plan.stages.extend(ring.to_stages())
            if p.split_parts > 1:
                slot = (query * p.split_parts + part) // spec.aic_budget
                offset = core * MQSMLA_FD_BYTES_PER_CORE + slot * MQSMLA_FD_SLOT_BYTES
                partial = lower.access(name + ":fd_partial_write", "fd_workspace", MQSMLA_FD_SLOT_BYTES,
                                       after=state, owner=vector, offset=offset, private=True, write=True,
                                       working_set=spec.workspace_bytes_total)
                partials.append((partial, offset))
                partial_publications.append(partial)
            else:
                partials.append((lower.stage(name + ":done", state) if len(state) > 1 else state[-1], 0))
            start += size
        deferred.append(tuple(partials))
    barrier = lower.stage("mqsmla:fd_barrier", partial_publications, launch_wide=True) if partial_publications else None
    for query, partials in enumerate(deferred):
        owner = query % spec.aiv_budget
        done = tuple(dep for dep, _ in partials)
        if barrier is not None:
            reads = tuple(lower.access(
                f"mqsmla:{query}:fd_partial_read:{part}", "fd_workspace", MQSMLA_FD_SLOT_BYTES,
                after=(barrier,), owner=owner, offset=offset, private=True, working_set=spec.workspace_bytes_total,
            ) for part, (_, offset) in enumerate(partials))
            done = (lower.vector(f"mqsmla:{query}:fd_reduce", p.split_parts * (h * d + 2 * h),
                                 reads, owner, "vec_reduce"),)
        lower.access(f"mqsmla:{query}:output", "mqsmla:output", h * d * BF16_BYTES, after=done,
                     owner=owner, write=True, offset=query * h * d * BF16_BYTES)


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
    if name == ATTN_RES and p.tokens == 0:
        return plan.result("K3", rid, (rid,), spec, (), solution, launches=0)
    lower = _DslLowering(plan, spec, rid)
    ready = lower.launch(p)
    _LOWER[name](lower, p, ready)
    return plan.result("K3", rid, (rid,), spec, lower.semantic(), solution, launches=1 + spec.uses_aicpu_metadata)
