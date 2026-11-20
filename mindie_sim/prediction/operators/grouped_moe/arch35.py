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
    DualBatchMoEPipeline, StageDemand, activation_stages, capability_gap,
    blaze_tasks, ceil_div, element_bytes, fractal_issues, fractal_k0, make_plan, matrix_schedule,
    matrix_stages, routing_stages, tensor_bytes, validate_contract,
)

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h",
        source_sha256="799f7caf064cb129177529d70210bc8cbc94e2d314bc08cb6b13924b2f13c54e",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp",
        source_sha256="a6aec88582ccaa2de389ba5f483a2b56b7897f8fe92ad220960b08513ec01a35",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
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
    """Reserve a paired output tile per owner and leave room for DBO routing credits."""
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
    limit = capacity // (accum_bytes * columns)
    if rows > limit:
        rows = limit // FRACTAL_M_DIM * FRACTAL_M_DIM
    if not rows:
        raise ModelingError("grouped_moe_arch35_ub_tile_does_not_fit")
    return replace(contract, base_m=rows, base_n=columns), rows * columns * accum_bytes


def _gated_stages(contract, ctx, name, after, metadata, *, start_core=0):
    """Lower paired MMADs with a UB credit and an AIV acknowledgement per tile."""
    tiled, slot_bytes = _fit_ub(contract, ctx)
    n_alignment = MX_SCALE_GROUP_SIZE if contract.quant_dtype in MX_DTYPES else 1
    schedule = matrix_schedule(tiled, ctx, contract.group_rows, contract.intermediate,
                               contract.k, start_core=start_core, n_tail_alignment=n_alignment)
    gmm, activation = name + "gmm1", name + "swiglu_quant"
    gmm_ends, stores, stages = [], [], []
    previous = {}
    for group in schedule.groups:
        split_group = group if tiled.tail_split else replace(group, tail_tiles=0)
        for task, rows, columns in blaze_tasks(
            split_group, contract.intermediate, tiled.base_m, tiled.base_n,
            tiled.m_tail_cnt, tiled.n_tail_cnt, n_alignment,
        ):
            owner = (group.start_core + task) % ctx.effective_aic_cores
            aiv = owner * ctx.effective_aiv_cores // ctx.effective_aic_cores
            key = f"e{group.expert}:t{task}"
            deps = previous.get(owner, after)
            body = _paired_tile(contract, gmm, activation, key, deps, rows, columns,
                                owner, aiv, slot_bytes, metadata)
            stages.extend(body)
            gmm_ends.append(body[-3].name)
            stores.append(body[-1].name)
            previous[owner] = (body[-2].name,)
    divisor = contract.activation_divisor
    stages.append(StageDemand(gmm, tuple(gmm_ends) or after, metadata=(*metadata,
        ("useful_flops", divisor * schedule.useful_flops), ("issued_fractals", divisor * schedule.issued_fractals),
        ("max_waves", schedule.max_waves), ("wave_tail_imbalance", schedule.wave_tail_imbalance),
    )))
    if contract.quant_dtype not in MX_DTYPES | {"bf16", "fp16"} and contract.m:
        scale_bytes = contract.m * PER_TOKEN_SCALE_BYTES
        scales = StageDemand(activation + ":scales", tuple(stores), Work(resources={
            "hbm": scale_bytes, "l2": scale_bytes, "mte3": scale_bytes,
        }), metadata=metadata)
        stages.append(scales)
        stores = [scales.name]
    stages.append(StageDemand(activation, (gmm, *stores), metadata=metadata))
    return stages, schedule


def _paired_tile(contract, gmm, activation, key, after, rows, columns,
                 owner, aiv, slot_bytes, metadata):
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
    quantized = contract.quant_dtype not in {"bf16", "fp16"}
    output = tensor_bytes(rows, columns, contract.quant_dtype)
    ops = rows * columns * (
        ACTIVATION_OPS[contract.activation] + (QUANT_OPS_PER_ELEMENT if quantized else 0)
    )
    base = f"{gmm}:{key}"
    read, cube, credit, flag = (f"{base}:{phase}" for phase in ("read", "cube", "credit", "fixpipe"))
    vector, store = f"{activation}:{key}:vector", f"{activation}:{key}:store"
    slot = (f"{gmm}:aic{owner}:handoff", slot_bytes)
    return (
        StageDemand(read, after, Work(resources={
            "hbm": operand_bytes, "l2": operand_bytes, "mte2_aic": operand_bytes,
        }), "aic", owner, metadata),
        StageDemand(cube, (read,), Work({contract.compute_dtype: divisor * issued},
                                       {"mte1": operand_bytes}), "aic", owner, metadata),
        StageDemand(credit, after, kind="aiv", owner=aiv, metadata=metadata, ub_acquire=slot),
        StageDemand(flag, (cube, credit), Work(resources={"fixpipe": fix_bytes}), "aic", owner, metadata),
        StageDemand(vector, (flag,), Work(resources={"vector": ops}), "aiv", aiv, metadata, ub_release=slot),
        StageDemand(store, (vector,), Work(resources={
            "hbm": output, "l2": output, "mte3": output,
        }), "aiv", aiv, metadata),
    )


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
    stages = routing_stages(dispatch, after, contract, metadata, transport_builder)
    cohort = tuple((key, value) for key, value in metadata if key in {"cohort_id", "slot"})
    wave_count = sum(1 for _ in _wave_histograms(contract, ctx.effective_aic_cores))
    last_activation = (dispatch,)
    gmm1_core = gmm2_core = 0
    gmm1_ends, activation_ends = [], []
    for wave, rows in enumerate(_wave_histograms(contract, ctx.effective_aic_cores)):
        prefix = name if wave_count == 1 else f"{name}wave{wave}_"
        active = replace(contract, group_rows=rows, m=sum(rows))
        wave_metadata = (*cohort, ("wave", wave))
        body, schedule = _gated_stages(
            active, ctx, prefix, last_activation, wave_metadata, start_core=gmm1_core
        )
        gmm1_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        stages.extend(body)
        gmm1_ends.append(prefix + "gmm1")
        last_activation = (prefix + "swiglu_quant",)
        activation_ends.extend(last_activation)
    gmm2_ends = []
    lag = specialization == "MegaMoeA8W8Wave" and contract.tokens >= GMM2_LAG_MIN_TOKEN_NUM
    for wave, rows in enumerate(_wave_histograms(contract, ctx.effective_aic_cores)):
        prefix = name if wave_count == 1 else f"{name}wave{wave}_"
        deps = (activation_ends[wave], *gmm2_ends[-1:])
        if lag and wave + 1 < wave_count:
            deps += (gmm1_ends[wave + 1],)
        schedule = matrix_schedule(
            contract, ctx, rows, contract.k, contract.intermediate, start_core=gmm2_core
        )
        gmm2_core = schedule.groups[-1].end_block % ctx.effective_aic_cores
        gmm2_ends.append(prefix + "gmm2")
        stages.extend(matrix_stages(
            gmm2_ends[-1], deps, contract, schedule, contract.intermediate,
            contract.output_dtype, input_dtype=contract.quant_dtype,
            input_scaled=contract.quant_dtype == "int8", metadata=(*cohort, ("wave", wave)),
        ))
    if wave_count != 1:
        previous = dispatch
        for phase, deps in (("gmm1", gmm1_ends), ("swiglu_quant", activation_ends), ("gmm2", gmm2_ends)):
            stages.append(StageDemand(name + phase, (previous, *deps), metadata=metadata))
            previous = name + phase
    combine_after = (name + "gmm2",)
    if contract.shared_intermediate:
        combine_after += (_shared_stages(stages, ctx, contract, name, after, cohort),)
    stages.extend(routing_stages(combine, combine_after, contract, metadata, transport_builder))
    return stages


def _shared_stages(stages, ctx, contract, name, after, metadata):
    """One local shared expert with independent weights and the same token view."""
    shared = replace(
        contract, group_rows=(contract.tokens,), m=contract.tokens,
        n=contract.activation_divisor * contract.shared_intermediate,
    )
    prefix = name + "shared_"
    body, _ = _gated_stages(shared, ctx, prefix, after, metadata)
    stages.extend(body)
    schedule = matrix_schedule(shared, ctx, shared.group_rows, shared.k, shared.intermediate)
    end = prefix + "gmm2"
    stages.extend(matrix_stages(
        end, (prefix + "swiglu_quant",), shared, schedule, shared.intermediate,
        shared.output_dtype, input_dtype=shared.quant_dtype, metadata=metadata,
    ))
    return end


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
            schedule = matrix_schedule(contract, ctx, contract.group_rows, contract.n, contract.k)
            stages = matrix_stages("gmm", (), contract, schedule, contract.k, contract.output_dtype)
            metadata.update(useful_flops=schedule.useful_flops, issued_fractals=schedule.issued_fractals,
                            max_waves=schedule.max_waves, wave_tail_imbalance=schedule.wave_tail_imbalance)
            implementation = "aclnnGroupedMatmul"
        elif mode == "dequant_swiglu_quant":
            stages = activation_stages("dequant_swiglu_quant", (), ctx, contract, contract.group_rows)
            implementation = "aclnnDequantSwigluQuant"
        elif mode == "gmm_swiglu_quant":
            stages, schedule = _gated_stages(contract, ctx, "", (), ())
            metadata.update(useful_flops=2 * schedule.useful_flops,
                            issued_fractals=2 * schedule.issued_fractals,
                            gm_handoff_eliminated_bytes=2 * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM)
            implementation = "aclnnGroupedMatmulSwigluMxQuant"
        else:
            specialization = _specialization(op, ctx, contract)
            emit = partial(_routed_stages, ctx, specialization)
            transport_builder = dict(op.metadata).get("cohort_transport_builder")
            if contract.dbo_split is not None:
                stages = DualBatchMoEPipeline(contract.dbo_split).stages(
                    contract, ctx, emit, transport_builder=transport_builder
                )
            else:
                stages = emit(contract, "", (), (), transport_builder) if contract.m or contract.shared_intermediate else []
            metadata["wave_specialization"] = specialization
            metadata["gm_handoff_eliminated_bytes"] = 2 * contract.m * contract.n * INT32_HANDOFF_BYTES_PER_ELEM
            implementation = self.abi
        return make_plan(implementation, ctx, stages, self.sources, metadata=metadata)


STRATEGY: ArchStrategy = _Arch35Strategy()
