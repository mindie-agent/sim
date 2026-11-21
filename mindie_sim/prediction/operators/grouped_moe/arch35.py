"""Arch35 grouped matmul, MX activation and wave MegaMoE.

[Role] Bind the paired FixPipe-to-UB producers, AIV epilogues and wave order.
[Invariants] MX scales are E8M0 per 32 elements. AIC-to-AIV transfers consume
  the direct route and never charge a GM accumulator write or read.
[Timeouts & Locks] Finite wave and owner work; no I/O or locks.
[Side-Effect Policy] Build immutable plans without querying device state.
Upstream-ABI: aclnnMegaMoe
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ..contracts import ArchStrategy, OperatorPlanningContext
from .common import (
    ACTIVATION_OPS, FRACTAL_M_DIM, FRACTAL_N_DIM, GEMM_MODES, INT32_HANDOFF_BYTES_PER_ELEM,
    MX_DTYPES, MX_SCALE_GROUP_SIZE, PER_TOKEN_SCALE_BYTES, QUANT_OPS_PER_ELEMENT, ROUTED_MODES,
    SPARSE_GROUP_LIST_TYPE,
    StageDemand, activation_stages, capability_gap, contract_fields,
    blaze_tasks, ceil_div, element_bytes, fractal_issues, fractal_k0, make_plan, matrix_schedule,
    matrix_stages, tensor_bytes, validate_contract,
)

from .lowering import DualBatchMoEPipeline, routing_stages, validate_transport_work

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel_arch35.h",
        source_sha256="ec37a6ab8f9568ff25985fdb71e37e2d21551acd140430a9931dac45b94e868d",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_mxquant.h",
        source_sha256="7c3dee59db38f4f3d89b8ce721634e457827838842e19cfe5d6634d0c174c60f",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h",
        source_sha256="799f7caf064cb129177529d70210bc8cbc94e2d314bc08cb6b13924b2f13c54e",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="gmm/common/cgmct/epilogue/cgmct_block_epilogue_swiglu_mx_quant.h",
        source_sha256="b98ff86940e80d42a6e33f087f71fa39324909f5f3e772e08a9a89ff56e846ee",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp",
        source_sha256="a6aec88582ccaa2de389ba5f483a2b56b7897f8fe92ad220960b08513ec01a35",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="af03b86b9c5cf197cef7338a1e750881fcee426b",
        relative_path="mc2/mega_moe/op_kernel/arch35/mega_moe_wave_a8w8.h",
        source_sha256="ebb1b7326d655c1d281eed8b835f44bf46b86fbb73d2a7e77642966f49f80297",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp",
        source_sha256="84ac2b05ccfb7366de7706e4ae4c92c7bbec7b544cc4b12de75537538e7d2bc3",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h",
        source_sha256="5626ba254394928a42c70e31907cb7269c99b9ea7759ba57e7ef0a0f6b07ee2f",
    ),
    SourceRef(
        repository="cann/ops-tensor",
        revision="c7798ac7b1b53a73a4a69eac2ca0b405393e09b0",
        relative_path="include/blaze/gemm/block/block_scheduler_grouped_matmul.h",
        source_sha256="3ee3e474c99195ef1ee86b18d59a3083096ab591e24c118e4790bb57664310b8",
    ),
)

ARCH_ID = ArchId("npu_arch_3510")
GMM2_LAG_MIN_TOKEN_NUM = 4096
MEGA_MOE_WAVE_TILE = 256
GMM1_WAVE_N_TILES_PER_CORE = 4
FP4_DTYPES = frozenset({"fp4", "fp4_e2m1", "mxfp4"})
BYTE_DTYPES = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "int8", "hif8"})
AIV_INPUT_DTYPES = frozenset({"int32", "fp16", "bf16"})
MX_AIVS_PER_AIC = 2
MX_ABI = "aclnnGroupedMatmulSwigluMxQuant"
FLOAT_EPILOGUE_DTYPES = frozenset({"bf16", "fp16", "fp32"})
DEQUANT_GROUP_CUT_MIN_BINS = 32
DEQUANT_SPARSE_CUT_MIN_BINS = 64
DEQUANT_GROUP_CUT_MAX_AVG_ROWS = 16
DEQUANT_FULL_LOAD_MAX_WIDTH = 5120
DEQUANT_UB_RESERVE_BYTES = 1024
UB_BLOCK_BYTES = 32


def _nz_tail_alignment(contract):
    return fractal_k0(contract.matrix_weight_dtype) if contract.padded_weight else 1


def _specialization(op, ctx, contract):
    a, b = contract.dtype_a, contract.dtype_b
    if a == b == contract.output_dtype == "bf16":
        return "MegaMoeA16W16Wave"
    if a in FP4_DTYPES and b in FP4_DTYPES:
        return "MegaMoeA4W4Wave"
    if a in BYTE_DTYPES and b in FP4_DTYPES:
        return "MegaMoeA8W4Wave"
    if a in BYTE_DTYPES and b in BYTE_DTYPES:
        return "MegaMoeA8W8Wave"
    raise capability_gap(op, ctx, f"arch35_mega_moe_operand_pair:{a}:{b}")


def _validate(op, ctx, contract):
    mode = contract.mode
    if mode != "dequant_swiglu_quant" and ctx.effective_aic_cores < 1:
        raise capability_gap(op, ctx, "aic_owners")
    if mode != "dequant_swiglu_quant" and mode != "gmm_swiglu_quant" and not ctx.datapath.has_path("l0c", "gm"):
        raise capability_gap(op, ctx, "l0c_to_gm_fixpipe")
    if mode not in GEMM_MODES and ctx.effective_aiv_cores < 1:
        raise capability_gap(op, ctx, "aiv_owners")
    requested = contract.abi
    if requested == "aclnnMoeGroupedMatmulSwigluQuant":
        raise capability_gap(op, ctx, "arch35_xllm_int32_gm_abi")
    if mode in ROUTED_MODES | {"gmm_swiglu_quant"} and not ctx.datapath.has_l0c_to_ub:
        raise capability_gap(op, ctx, "l0c_to_ub_fixpipe")
    if mode in ROUTED_MODES:
        _specialization(op, ctx, contract)
    if mode in ROUTED_MODES | {"gmm_swiglu_quant"}:
        ctx.datapath.route("l0c", "ub")
    if mode == "gmm_swiglu_quant" and (
        contract.dtype_a not in MX_DTYPES or contract.dtype_b not in MX_DTYPES
        or contract.activation_divisor != 2
    ):
        raise capability_gap(op, ctx, "arch35_paired_mx_operands")
    if mode == "dequant_swiglu_quant" and contract.handoff_dtype not in AIV_INPUT_DTYPES:
        raise capability_gap(op, ctx, f"arch35_dequant_input:{contract.handoff_dtype}")


def _fit_ub(contract, ctx):
    """Reserve the largest AIV share of a handoff and leave room for DBO credits."""
    capacity = ctx.datapath.usable_bytes("ub")
    if contract.enable_dbo:
        capacity -= 2 * min(capacity // 2, contract.top_k * INT32_HANDOFF_BYTES_PER_ELEM)
    accum_bytes = element_bytes(contract.activation_divisor, contract.handoff_dtype)
    columns = min(contract.intermediate, contract.base_n)
    # Reduce M before N so independently stored MX scales remain group aligned.
    if columns < contract.intermediate and contract.quant_dtype in MX_DTYPES:
        columns = columns // MX_SCALE_GROUP_SIZE * MX_SCALE_GROUP_SIZE
    if not columns:
        raise ModelingError("grouped_moe_arch35_ub_tile_width_invalid")
    rows = min(contract.base_m, max(contract.group_rows, default=1) or 1)
    consumers = MX_AIVS_PER_AIC if contract.mode == "gmm_swiglu_quant" else 1
    limit = consumers * (capacity // (accum_bytes * columns))
    if rows > limit:
        rows = limit // FRACTAL_M_DIM * FRACTAL_M_DIM
    if not rows:
        raise ModelingError("grouped_moe_arch35_ub_tile_does_not_fit")
    return replace(contract, base_m=rows, base_n=columns), ceil_div(rows, consumers) * columns * accum_bytes


def _gated_stages(contract, ctx, name, after, metadata, *, start_core=0, owner_ready=None):
    """Lower paired MMADs with a UB credit and an AIV acknowledgement per tile."""
    tiled, slot_bytes = _fit_ub(contract, ctx)
    n_alignment = MX_SCALE_GROUP_SIZE if contract.quant_dtype in MX_DTYPES else 1
    schedule = matrix_schedule(tiled, ctx, contract.group_rows, contract.intermediate,
                               contract.k, start_core=start_core, n_tail_alignment=n_alignment)
    gmm, activation = name + "gmm1", name + "swiglu_quant"
    gmm_ends, stores, stages = [], [], []
    pairs = dict(ctx.datapath.aic_aiv_pairs)
    previous = {} if owner_ready is None else owner_ready
    for group in schedule.groups:
        split_group = group if tiled.tail_split else replace(group, tail_tiles=0)
        for task, rows, columns in blaze_tasks(
            split_group, contract.intermediate, tiled.base_m, tiled.base_n,
            tiled.m_tail_cnt, tiled.n_tail_cnt, n_alignment,
        ):
            owner = (group.start_core + task) % ctx.effective_aic_cores
            paired = pairs.get(owner, ())
            if contract.mode == "gmm_swiglu_quant":
                if len(paired) != MX_AIVS_PER_AIC or max(paired) >= ctx.effective_aiv_cores:
                    raise ModelingError("grouped_moe_arch35_mx_requires_paired_aivs")
                consumers = paired
            else:
                consumers = (paired[0] if paired else owner * ctx.effective_aiv_cores // ctx.effective_aic_cores,)
            key = f"e{group.expert}:t{task}"
            deps = (*after, *previous.get(owner, ()))
            body, fixpipe, done = _paired_tile(contract, gmm, activation, key, deps, rows, columns,
                                               owner, consumers, slot_bytes, metadata)
            stages.extend(body)
            gmm_ends.append(fixpipe)
            stores.extend(done)
            previous[owner] = done
    divisor = contract.activation_divisor
    stages.append(StageDemand(gmm, tuple(gmm_ends) or after, metadata=(*metadata,
        ("useful_flops", divisor * schedule.useful_flops), ("issued_fractals", divisor * schedule.issued_fractals),
        ("max_waves", schedule.max_waves), ("wave_tail_imbalance", schedule.wave_tail_imbalance),
    )))
    if contract.quant_dtype not in MX_DTYPES | FLOAT_EPILOGUE_DTYPES and contract.m:
        scale_bytes = contract.m * PER_TOKEN_SCALE_BYTES
        scales = StageDemand(activation + ":scales", tuple(stores), Work(resources={
            "hbm": scale_bytes, "l2": scale_bytes, "mte3": scale_bytes,
        }), metadata=metadata)
        stages.append(scales)
        stores = [scales.name]
    stages.append(StageDemand(activation, (gmm, *stores), metadata=metadata))
    return stages, schedule


def _paired_tile(contract, gmm, activation, key, after, rows, columns,
                 owner, consumers, slot_bytes, metadata):
    divisor = contract.activation_divisor
    input_scaled = contract.dtype_a == "int8" and contract.mode in ROUTED_MODES
    weight_k, weight_n = contract.k, columns
    if contract.padded_weight:
        k0 = fractal_k0(contract.dtype_b)
        k_align, n_align = (k0, FRACTAL_N_DIM) if contract.transpose_weight else (FRACTAL_M_DIM, k0)
        weight_k = ceil_div(weight_k, k_align) * k_align
        weight_n = ceil_div(weight_n, n_align) * n_align
    operand_bytes = divisor * (
        tensor_bytes(rows, contract.k, contract.dtype_a, scaled=input_scaled)
        + tensor_bytes(weight_n, weight_k, contract.dtype_b)
    )
    _, issued = fractal_issues(rows, columns, contract.k, contract.compute_dtype)
    fix_bytes = rows * element_bytes(columns * divisor, contract.handoff_dtype)
    quantized = contract.quant_dtype not in FLOAT_EPILOGUE_DTYPES
    ops_per_row = columns * (
        ACTIVATION_OPS[contract.activation] + (QUANT_OPS_PER_ELEMENT if quantized else 0)
    )
    base = f"{gmm}:{key}"
    read, cube, flag = (f"{base}:{phase}" for phase in ("read", "cube", "fixpipe"))
    stages = [
        StageDemand(read, after, Work(resources={
            "hbm": operand_bytes, "l2": operand_bytes, "mte2_aic": operand_bytes,
        }), "aic", owner, metadata),
        StageDemand(cube, (read,), Work({contract.compute_dtype: divisor * issued},
                                       {"mte1": operand_bytes}), "aic", owner, metadata),
    ]
    credits, done, epilogue = [], [], []
    rows_per_aiv = ceil_div(rows, len(consumers))
    for subblock, aiv in enumerate(consumers):
        prefix = f"{activation}:{key}:aiv{aiv}"
        credit, vector, store, sync = (f"{prefix}:{phase}" for phase in ("credit", "vector", "store", "sync"))
        active = max(0, min(rows_per_aiv, rows - subblock * rows_per_aiv))
        slot = (f"{gmm}:aic{owner}:handoff:aiv{aiv}", slot_bytes)
        tag = (*metadata, ("aiv_consumer", aiv), ("subblock", subblock), ("rows", active))
        if active:
            stages.append(StageDemand(credit, after, kind="aiv", owner=aiv, metadata=tag, ub_acquire=slot))
            credits.append(credit)
            output = tensor_bytes(active, columns, contract.quant_dtype)
            epilogue.extend((
                StageDemand(vector, (flag,), Work(resources={"vector": active * ops_per_row}), "aiv", aiv, tag),
                StageDemand(store, (vector,), Work(resources={
                    "hbm": output, "l2": output, "mte3": output,
                }), "aiv", aiv, tag),
            ))
        # The idle subblock still acknowledges both gate/up buffers to the AIC.
        epilogue.append(StageDemand(sync, (store if active else flag,), kind="aiv", owner=aiv,
                                    metadata=tag, ub_release=slot if active else None))
        done.append(sync)
    # Join the independent UB grants before publishing one accumulator tile.
    admitted = f"{base}:admitted"
    stages.extend((
        StageDemand(admitted, tuple(credits), metadata=metadata),
        StageDemand(flag, (cube, admitted), Work(resources={"fixpipe": fix_bytes}), "aic", owner, metadata),
        *epilogue,
    ))
    return stages, flag, tuple(done)


def _wave_histograms(contract, aic_cores):
    """Cut expert rows at the common GMM1 M-group budget for a wave."""
    m_group = 1 if contract.wave_rows else MEGA_MOE_WAVE_TILE
    groups_per_wave = contract.wave_rows or max(
        ceil_div(aic_cores * GMM1_WAVE_N_TILES_PER_CORE, ceil_div(contract.intermediate, MEGA_MOE_WAVE_TILE)),
        ceil_div(aic_cores, ceil_div(contract.k, MEGA_MOE_WAVE_TILE)),
    )
    batch = [0] * len(contract.group_rows)
    used = 0
    for expert, rows in enumerate(contract.group_rows):
        while rows:
            count = min(rows, (groups_per_wave - used) * m_group)
            batch[expert] += count
            rows -= count
            used += ceil_div(count, m_group)
            if used == groups_per_wave:
                yield tuple(batch)
                batch = [0] * len(batch)
                used = 0
    if used:
        yield tuple(batch)


def _routed_stages(ctx, specialization, contract, name, after, metadata, transport_builder):
    dispatch, combine = name + "dispatch", name + "combine"
    cohort = tuple((key, value) for key, value in metadata if key in {"cohort_id", "slot"})
    wave_count = sum(1 for _ in _wave_histograms(contract, ctx.effective_aic_cores))
    shared = contract.shared_intermediate and contract.tokens
    if not wave_count:
        stages = routing_stages(dispatch, after, contract, metadata, transport_builder)
        deps = (dispatch,)
        if shared:
            deps += (_shared_stages(stages, ctx, contract, name, after, cohort),)
        return stages + routing_stages(combine, deps, contract, metadata, transport_builder)
    stages, dispatch_ends = [], []
    last_dispatch = after
    owner_ready = {}
    gmm1_core = gmm2_core = 0
    gmm1_ends, activation_ends, waves = [], [], []
    offset = 0
    for wave, rows in enumerate(_wave_histograms(contract, ctx.effective_aic_cores)):
        prefix = name if wave_count == 1 else f"{name}wave{wave}_"
        end = offset + sum(rows)
        amounts = contract.transport_bytes
        if wave_count > 1:
            # A directed builder owns the exact per-wave wire cut. An
            # aggregate byte declaration only provides a batch total.
            amounts = (None, None) if transport_builder else tuple(
                None if amount is None else amount * end // contract.m - amount * offset // contract.m
                for amount in amounts
            )
        tokens = contract.tokens * end // contract.m - contract.tokens * offset // contract.m
        active = replace(contract, group_rows=rows, m=end - offset, tokens=tokens, transport_bytes=amounts)
        route_metadata = (*metadata, ("wave", wave), ("wave_row_start", offset), ("wave_row_end", end),
                          ("wave_expert_rows", rows))
        offset = end
        ready = prefix + "dispatch"
        stages.extend(routing_stages(ready, last_dispatch, active, route_metadata, transport_builder,
                                     read_counts=wave == 0))
        last_dispatch = (ready,)
        dispatch_ends.append(ready)
        wave_metadata = (*cohort, ("wave", wave))
        body, schedule = _gated_stages(
            active, ctx, prefix, (ready,), wave_metadata, start_core=gmm1_core, owner_ready=owner_ready,
        )
        gmm1_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        stages.extend(body)
        gmm1_ends.append(prefix + "gmm1")
        activation_ends.append(prefix + "swiglu_quant")
        waves.append((prefix, active, route_metadata))
    gmm2_ends, wave_combines = [], []
    lag = specialization == "MegaMoeA8W8Wave" and contract.tokens >= GMM2_LAG_MIN_TOKEN_NUM
    for wave, (prefix, active, route_metadata) in enumerate(waves):
        deps = (activation_ends[wave], *gmm2_ends[-1:])
        if lag and wave + 1 < wave_count:
            deps += (gmm1_ends[wave + 1],)
        schedule = matrix_schedule(
            contract, ctx, active.group_rows, contract.k, contract.intermediate, start_core=gmm2_core,
            n_tail_alignment=_nz_tail_alignment(contract),
        )
        gmm2_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        gmm2_ends.append(prefix + "gmm2")
        stages.extend(matrix_stages(
            gmm2_ends[-1], deps, contract, schedule, contract.intermediate,
            contract.output_dtype, input_dtype=contract.quant_dtype,
            input_scaled=contract.quant_dtype == "int8", metadata=(*cohort, ("wave", wave)),
        ))
        if wave_count > 1:
            wave_combines.append(prefix + "combine")
            stages.extend(routing_stages(wave_combines[-1], (gmm2_ends[-1],), active,
                                         route_metadata, transport_builder,
                                         publish_output=False, read_counts=False))
    if wave_count == 1:
        combine_after = (name + "gmm2",)
    else:
        if transport_builder:
            validate_transport_work(stages, contract.transport_bytes)
        for phase, deps in (("dispatch", dispatch_ends), ("gmm1", gmm1_ends),
                            ("swiglu_quant", activation_ends), ("gmm2", gmm2_ends)):
            stages.append(StageDemand(name + phase, tuple(deps), metadata=metadata))
        combine_after = (name + "gmm2", *wave_combines)
        # Reverse each wave's routes before the single local weighted reduction.
        contract = replace(contract, m=0, transport_bytes=(0, 0))
        transport_builder = None
    if shared:
        combine_after += (_shared_stages(stages, ctx, contract, name, after, cohort),)
    stages.extend(routing_stages(combine, combine_after, contract, metadata, transport_builder))
    return stages


def _shared_stages(stages, ctx, contract, name, after, metadata):
    """One local shared expert with independent weights and the same token view."""
    compute = contract.shared_compute_dtype
    input_dtype = contract.dtype_a if compute == contract.compute_dtype else ("int8" if compute == "w4a8" else compute)
    handoff = "int32" if compute in {"int8", "w4a8"} else (
        "fp32" if compute in MX_DTYPES else contract.output_dtype
    )
    shared = replace(
        contract, group_rows=(contract.tokens,), m=contract.tokens,
        n=contract.activation_divisor * contract.shared_intermediate,
        compute_dtype=compute, dtype_a=input_dtype, dtype_b=contract.shared_weight_dtype,
        quant_dtype=input_dtype, handoff_dtype=handoff,
    )
    prefix = name + "shared_"
    body, _ = _gated_stages(shared, ctx, prefix, after, metadata)
    stages.extend(body)
    schedule = matrix_schedule(shared, ctx, shared.group_rows, shared.k, shared.intermediate,
                               n_tail_alignment=_nz_tail_alignment(shared))
    end = prefix + "gmm2"
    stages.extend(matrix_stages(
        end, (prefix + "swiglu_quant",), shared, schedule, shared.intermediate,
        shared.output_dtype, input_dtype=shared.quant_dtype,
        input_scaled=shared.quant_dtype == "int8", metadata=metadata,
    ))
    return end


def _int32_dequant_stages(contract, ctx, fields):
    name = "dequant_swiglu_quant"
    if not contract.m:
        return [StageDemand(name)]
    rows, width = contract.group_rows, contract.intermediate
    allocated = contract.allocated_rows
    channels = element_bytes(contract.n, "fp32")
    max_owners = min(ctx.effective_aiv_cores, allocated)
    sparse = fields.get("group_list_type") == SPARSE_GROUP_LIST_TYPE
    min_groups = DEQUANT_SPARSE_CUT_MIN_BINS if sparse else DEQUANT_GROUP_CUT_MIN_BINS
    special = (width <= DEQUANT_FULL_LOAD_MAX_WIDTH and len(rows) >= min_groups
               and allocated // len(rows) <= DEQUANT_GROUP_CUT_MAX_AVG_ROWS
               and (sparse or not (fields.get("has_bias") or fields.get("has_quant_scale"))))
    owners = max_owners
    if not special and width <= DEQUANT_FULL_LOAD_MAX_WIDTH:
        # Full-load CANN tiling uses two row slots for x, activation scales
        # and y, plus the FP32 vector scratch and the weight-scale planes.
        aligned = ceil_div(element_bytes(width, "fp32"), UB_BLOCK_BYTES) * UB_BLOCK_BYTES
        y_aligned = ceil_div(width, UB_BLOCK_BYTES) * UB_BLOCK_BYTES
        row_bytes = 2 * (2 * aligned + UB_BLOCK_BYTES + y_aligned) + UB_BLOCK_BYTES + aligned
        remaining = ctx.datapath.usable_bytes("ub") - DEQUANT_UB_RESERVE_BYTES - 2 * aligned - aligned
        tile_rows = remaining // row_bytes
        if tile_rows < 1:
            raise ModelingError("grouped_moe_dequant_full_load_does_not_fit")
        owners = min(owners, ceil_div(allocated, tile_rows))
    stages, ends, offset = [], {}, 0
    for group, count in enumerate(rows):
        if not count:
            continue
        step = count if special else ceil_div(count, owners)
        for part, start in enumerate(range(0, count, step)):
            owner = group % owners if special else part
            size = min(step, count - start)
            base = f"{name}:group{group}:aiv{owner}"
            after = (ends[owner],) if owner in ends else ()
            read, vector, store = (f"{base}:{phase}" for phase in ("read", "vector", "store"))
            tag = (("group", group), ("row_start", offset + start), ("rows", size),
                   ("weight_scale_bytes", channels), ("special_core_cut", special))
            row_scale = PER_TOKEN_SCALE_BYTES if fields.get("has_activation_scale", True) else 0
            inputs = size * (element_bytes(contract.n, "int32") + row_scale) + channels
            output = tensor_bytes(size, width, contract.quant_dtype, scaled=True)
            ops = size * width * (ACTIVATION_OPS[contract.activation] + QUANT_OPS_PER_ELEMENT)
            stages.extend((
                StageDemand(read, after, Work(resources={"hbm": inputs, "l2": inputs, "mte2_aiv": inputs}),
                            "aiv", owner, tag),
                StageDemand(vector, (read,), Work(resources={"vector": ops}), "aiv", owner, tag),
                StageDemand(store, (vector,), Work(resources={"hbm": output, "l2": output, "mte3": output}),
                            "aiv", owner, tag),
            ))
            ends[owner] = store
        offset += count
    return [*stages, StageDemand(name, tuple(ends.values()))]


@dataclass(frozen=True, slots=True)
class _Arch35Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = "aclnnMegaMoe"
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            _validate(op, ctx, validate_contract(op, ctx))
        except (ValueError, TypeError):
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        if ctx.compute_arch.arch_id != self.arch_id:
            raise capability_gap(op, ctx, "arch35_compute_identity")
        contract = validate_contract(op, ctx)
        _validate(op, ctx, contract)
        mode = contract.mode
        metadata = {"mode": mode, "enable_dbo": contract.enable_dbo}
        if mode in GEMM_MODES:
            schedule = matrix_schedule(contract, ctx, contract.group_rows, contract.n, contract.k,
                                       n_tail_alignment=_nz_tail_alignment(contract))
            stages = matrix_stages("gmm", (), contract, schedule, contract.k, contract.output_dtype)
            metadata.update(useful_flops=schedule.useful_flops, issued_fractals=schedule.issued_fractals,
                            max_waves=schedule.max_waves, wave_tail_imbalance=schedule.wave_tail_imbalance)
            implementation = "aclnnGroupedMatmul"
        elif mode == "dequant_swiglu_quant":
            stages = (_int32_dequant_stages(contract, ctx, contract_fields(op)) if contract.handoff_dtype == "int32"
                      else activation_stages("dequant_swiglu_quant", (), ctx, contract, contract.group_rows))
            implementation = "aclnnDequantSwigluQuant"
        elif mode == "gmm_swiglu_quant":
            contract = replace(contract, handoff_dtype="fp32")
            stages, schedule = _gated_stages(contract, ctx, "", (), ())
            metadata.update(useful_flops=2 * schedule.useful_flops,
                            issued_fractals=2 * schedule.issued_fractals,
                            gm_handoff_eliminated_bytes=2 * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM)
            implementation = contract.abi or MX_ABI
        else:
            specialization = _specialization(op, ctx, contract)
            emit = partial(_routed_stages, ctx, specialization)
            transport_builder = contract_fields(op).get("cohort_transport_builder")
            if contract.dbo_split is not None:
                stages = DualBatchMoEPipeline(contract.dbo_split).stages(
                    contract, ctx, emit, transport_builder=transport_builder
                )
            else:
                stages = emit(contract, "", (), (), transport_builder) if (
                    contract.m or contract.tokens or any(contract.transport_bytes) or transport_builder
                ) else []
            metadata["wave_specialization"] = specialization
            metadata["gm_handoff_eliminated_bytes"] = 2 * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM
            implementation = self.abi
        return make_plan(implementation, ctx, stages, self.sources, metadata=metadata)


STRATEGY: ArchStrategy = _Arch35Strategy()
