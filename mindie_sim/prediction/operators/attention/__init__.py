"""Public entrypoint for governed Ascend attention.

[Role] Bind structural attention contracts to the declared ISA and source ABI.
[Invariants] A source ABI selects exactly one strategy. Legacy sparse callers
  use the same planner and retain their recipe identities.
[Timeouts & Locks] Bounded source dispatch without I/O or locks.
[Side-Effect Policy] Pure operator, planning and registration records.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import ArchId, ComputeArchIdentity, DatapathCapabilitySpec, GENERIC_ARCH_ID, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...collectives import PairwiseCollective
from ...mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from ...program import KernelProgram, Stage
from ...results import OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from . import arch22, arch35
from ._cp_tiling import (
    HISTORY_DECODE_SITE, HISTORY_PREFILL_SITE, OUTPUT_LSE_EXCHANGE_RECIPE, cp_decode_suffix,
    cp_operators, history_decode_contract, history_output_shape, with_collectives,
)
from .common import (
    ARCH22, ARCH35, BF16_BYTES, C8_ABI, C8_LATENT_DIM, C8_ROPE_DIM, C8_V1_ABI, C8_VLLM_ABI, C8_VLLM_ROW_BYTES,
    FIA_ABI, FIA_TILE, GBSA_ABI, MINDIE_ABIS, MINDIE_V3_ABI, NOPE_ABI, QUANT_MODE_BF16, QUANT_MODE_FP8_C8,
    QUANT_MODE_INT8_C8, QUANT_MODE_MINDIE_FP8, QUANT_MODE_MXFP4_CX, QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP8,
    QUANT_MODE_TQ4, SFA_V1_ABI, SOFTMAX_OPS, SUPPORTED_QUANT_MODES, TQ4_ROW_BYTES, AttentionContract,
    ContextParallelPlan, FiaPanelPlan, bsh_kv_access_plan, c8_gather_accounting, describe, lower_cp_collectives,
    shipping_fia_panel_plan, solve_wide_head_tiling, unmodeled, with_sequence_context,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = arch35.SOURCES
FIA_RECIPE = "fused_infer_attention_score"
GBSA_RECIPE = "generic_block_sparse_attention_arch35"
KV_QUANT_SFA_C8_RECIPE = "kv_quant_sfa_c8_arch35"
SPARSE_FLASH_MLA_NOPE_RECIPE = "sparse_flash_mla_nope_arch35"
RECIPE_BY_ABI = {
    FIA_ABI: FIA_RECIPE, GBSA_ABI: GBSA_RECIPE, C8_ABI: KV_QUANT_SFA_C8_RECIPE,
    C8_V1_ABI: C8_V1_ABI, C8_VLLM_ABI: C8_VLLM_ABI, SFA_V1_ABI: SFA_V1_ABI,
    NOPE_ABI: SPARSE_FLASH_MLA_NOPE_RECIPE, **{abi: abi for abi in MINDIE_ABIS},
}
ABI_BY_RECIPE = {recipe: abi for abi, recipe in RECIPE_BY_ABI.items()}
STRATEGIES = {ARCH22: arch22.STRATEGY, ARCH35: arch35.STRATEGY}
DEFAULT_GBSA_BLOCK_SIZE = 8
DEFAULT_GBSA_KV_LAYOUT = "PA_BLK_BSND"
DEFAULT_SFA_SELECTED_TOKENS = 2048
FP8_ELEMENT_BYTES = 1.0
INT8_ELEMENT_BYTES = 1.0
MXFP4_ELEMENT_BYTES = 0.5
C8_SCALE_BYTES_PER_ROW = 16


def _is_gbsa_fp8_quant(contract: AttentionContract) -> bool:
    return contract.abi == GBSA_ABI and contract.quant_mode in {QUANT_MODE_FP8_C8, QUANT_MODE_MXFP8}


def _merged_interval_len(intervals: list[tuple[int, int]]) -> int:
    """Count distinct integer positions across possibly overlapping [lo, hi) spans."""
    ordered = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    if not ordered:
        return 0
    total = 0
    cur_lo, cur_hi = ordered[0]
    for lo, hi in ordered[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    return total + (cur_hi - cur_lo)


def _contract_io_seq_vis(contract: AttentionContract) -> list[int]:
    seq_vis: list[int] = []
    for idx, (q, kv) in enumerate(contract.sequences):
        if q <= 0 or kv <= 0:
            continue
        selected = kv if contract.selected_tokens is None else contract.selected_tokens
        spans = (
            contract.sequence_query_spans[idx]
            if idx < len(contract.sequence_query_spans) else ()
        )
        eff_kv = (
            min(kv, max((hi for _, hi in spans), default=0))
            if (contract.causal and spans) else kv
        )
        if contract.window_size:
            if spans:
                if contract.causal:
                    clipped = [
                        (max(0, lo + 1 - contract.window_size), min(eff_kv, hi))
                        for lo, hi in spans
                    ]
                else:
                    radius = contract.window_size - 1
                    clipped = [
                        (max(0, lo - radius), min(kv, hi + radius))
                        for lo, hi in spans
                    ]
                win_vis = min(eff_kv, _merged_interval_len(clipped))
            else:
                win_vis = contract.window_size + max(0, q - 1)
            seq_vis.append(min(eff_kv, selected, win_vis))
        else:
            seq_vis.append(min(eff_kv, selected))
    return seq_vis


def _minimum_contract_io(contract: AttentionContract, dtype: str) -> tuple[float, float, float, float]:
    """Conservative physical lower bound on (hbm, mte2, mte3, vector_ops) for L* soundness."""
    out_elem = float(dtype_bytes(dtype))
    if contract.abi == MINDIE_V3_ABI and contract.quant_mode in {QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX}:
        q_elem = kv_elem = MXFP4_ELEMENT_BYTES
    elif contract.quant_mode == QUANT_MODE_MXFP8 or _is_gbsa_fp8_quant(contract) or (
        contract.abi in MINDIE_ABIS and contract.quant_mode == QUANT_MODE_MINDIE_FP8
    ):
        q_elem = kv_elem = FP8_ELEMENT_BYTES
    elif contract.abi == GBSA_ABI and contract.quant_mode == QUANT_MODE_INT8_C8:
        q_elem = out_elem
        kv_elem = INT8_ELEMENT_BYTES
    else:
        q_elem = kv_elem = out_elem
    q_bytes = float(contract.query_tokens * contract.num_heads * contract.head_dim) * q_elem
    o_bytes = float(contract.query_tokens * contract.num_heads * contract.v_head_dim) * out_elem
    row_counts = tuple(contract.row_counts())
    vector_ops = float(SOFTMAX_OPS * contract.num_heads * sum(row_counts))
    if contract.selection_lists:
        unique_rows = len({addr for rows in contract.selection_lists for addr in rows})
    elif contract.sequences:
        seq_vis = _contract_io_seq_vis(contract)
        unique_rows = (max(seq_vis, default=0)
                       if (contract.shared_prefix or contract.block_table)
                       else sum(seq_vis))
    elif row_counts:
        unique_rows = max(row_counts)
    else:
        unique_rows = 0
    if contract.abi in {C8_ABI, C8_V1_ABI} and contract.quant_mode == QUANT_MODE_TQ4:
        kv_row_b = float(TQ4_ROW_BYTES)
    elif contract.abi in {C8_ABI, C8_V1_ABI, C8_VLLM_ABI}:
        rope_dim = max(0, contract.head_dim - contract.v_head_dim)
        kv_row_b = float(contract.v_head_dim + C8_SCALE_BYTES_PER_ROW + rope_dim * BF16_BYTES)
    elif contract.abi in {NOPE_ABI, SFA_V1_ABI} and contract.num_kv_heads == 1:
        kv_row_b = float(contract.head_dim) * kv_elem
    else:
        kv_row_b = float(contract.num_kv_heads * (contract.head_dim + contract.v_head_dim)) * kv_elem
    kv_read = float(unique_rows) * kv_row_b
    # Warm external L2 backing (kv_tensor_id) may absorb KV HBM reads; Q/O remain live.
    kv_hbm = 0.0 if contract.kv_tensor_id else kv_read
    return q_bytes + kv_hbm + o_bytes, q_bytes + kv_read, o_bytes, vector_ops


def _contract(op: OpSpec, requested_abi: str | None) -> AttentionContract:
    meta = dict(op.metadata)
    raw = meta.get("attention_contract")
    if isinstance(raw, AttentionContract):
        contract = raw
    elif isinstance(raw, Mapping):
        values = dict(raw)
        if values.get("cp_plan"):
            values["cp_plan"] = tuple(p if isinstance(p, ContextParallelPlan) else ContextParallelPlan(**p)
                                      for p in values["cp_plan"])
        if isinstance(values.get("fia_template"), Mapping):
            template = dict(values["fia_template"])
            for key in ("k_panels", "v_panels"):
                template[key] = tuple(template[key])
            values["fia_template"] = FiaPanelPlan(**template)
        contract = AttentionContract(**values)
    else:
        raise unmodeled("attention_contract_required", "Bind rank-local Q, KV and selection geometry")
    bindings = (requested_abi, meta.get("native_kind"), ABI_BY_RECIPE.get(op.recipe))
    if any(abi is not None and abi != contract.abi for abi in bindings):
        raise unmodeled("attention_abi_conflict", f"Bind one consistent ABI for {op.op_id}")
    if contract.phase != op.phase:
        raise ModelingError("attention_op_phase_mismatch")
    return history_decode_contract(contract)


def _planning_context(op: OpSpec, geometry: Geometry) -> OperatorPlanningContext:
    datapath = geometry.datapath
    if datapath is None:
        arch = geometry.compute_arch
        if arch is None:
            raise unmodeled("attention_datapath_required", "Bind explicit compute architecture and datapath")
        aics = geometry.compute_units
        aivs = geometry.resource_units.get("vector", aics)
        if aics < 1 or aivs < 1:
            raise unmodeled("attention_owner_pairs", "Bind positive AIC and AIV budgets")
        ratio, remainder = divmod(aivs, aics)
        pairs = tuple((aic, tuple(range(aic * ratio, (aic + 1) * ratio))) for aic in range(aics))
        datapath = DatapathCapabilitySpec(
            ComputeArchIdentity(ArchId(arch.preset) if arch.preset else GENERIC_ARCH_ID, arch),
            aics, aivs, pairs if not remainder else (),
            tuple(arch.paths), frozenset(), (), None, buffers=tuple(geometry.buffers.items()),
        )
        geometry = replace(geometry, datapath=datapath)
    return OperatorPlanningContext(
        OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id), geometry, datapath,
        stream_owners=geometry.stream_owners,
    )


def build(op: OpSpec, ctx: OperatorPlanningContext | Geometry, /) -> OperatorPlan:
    """Build the source-bound attention kernel at either planning entrypoint."""
    if op.recipe == RING_UPDATE_RECIPE:
        return build_ring_update(op, ctx)
    if op.recipe == GBSA_METADATA_RECIPE:
        return build_gbsa_metadata(op, ctx.geometry if isinstance(ctx, OperatorPlanningContext) else ctx)
    if isinstance(ctx, Geometry):
        ctx = _planning_context(op, ctx)
    ctx = with_sequence_context(op, ctx)
    contract = _contract(op, ctx.requested_abi)
    strategy = STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise unmodeled("attention_arch_unsupported", f"Bind attention for {ctx.compute_arch.arch_id}")
    if not op.dtype:
        raise ModelingError("attention_op_dtype_required")
    normalized = replace(op, dtype=op.dtype, metadata=tuple(sorted({
        **dict(op.metadata), "attention_contract": contract, "native_kind": contract.abi,
    }.items())))
    plan = strategy.build(normalized, ctx)
    if contract.cp_plan:
        caller_files = {("common_cp.py", binding.stack_revision) for binding in contract.cp_plan}
        caller_files.update((binding.caller_site.split(".", 1)[0] + ".py", binding.stack_revision)
                            for binding in contract.cp_plan)
        caller_files.update(("dcp_a2a.py", binding.stack_revision) for binding in contract.cp_plan
                            if binding.caller_site in {HISTORY_DECODE_SITE, HISTORY_PREFILL_SITE})
        sources = tuple(ref for ref in arch35.CP_SOURCES
                        if (ref.relative_path.rsplit("/", 1)[-1], ref.revision) in caller_files)
        plan = with_collectives(
            plan,
            cp_operators(normalized, contract.cp_plan, restored_query_tokens=contract.query_tokens),
            cp_decode_suffix(normalized, contract),
            sources,
        )
    return plan


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    """Return the kernel program and occurrence-bound pipeline projection."""
    if op.recipe == RING_UPDATE_RECIPE:
        plan = build_ring_update(op, ctx)
        return ResolvedOperator(ctx.occurrence, op, plan, RING_UPDATE_RECIPE, RING_UPDATE_SOURCES, None)
    if op.recipe == GBSA_METADATA_RECIPE:
        plan = build_gbsa_metadata(op, ctx.geometry)
        return ResolvedOperator(ctx.occurrence, op, plan, GBSA_METADATA_RECIPE, (), None)
    plan = build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(ctx.occurrence, op, plan, descriptor.implementation, descriptor.sources, descriptor)


def register_family() -> OperatorFamilySpec:
    """Declare the governed family's exact owned recipe IDs."""
    return OperatorFamilySpec(
        "attention",
        frozenset({"attention", "mla", "attention_metadata"}),
        frozenset((*ABI_BY_RECIPE, RING_UPDATE_RECIPE, GBSA_METADATA_RECIPE)),
        resolve_operator_plan,
    )


def _kernels(op: OpSpec) -> tuple[OpSpec, ...]:
    contract = _contract(op, None)
    gathers = cp_operators(op, contract.cp_plan or (), restored_query_tokens=contract.query_tokens)
    kernel_shape = (contract.query_tokens, contract.num_heads, contract.v_head_dim)
    kernel = replace(
        op, op_id="attention", deps=(gathers[-1].op_id,) if gathers else (), source_boundary="kernel",
        semantic_shape=kernel_shape, local_shape=kernel_shape, padded_shape=kernel_shape, launch_shape=kernel_shape,
        returns=tuple(dict.fromkeys((*op.returns, "lse"))) if contract.return_lse else op.returns,
        metadata=tuple({**dict(op.metadata), "attention_contract": asdict(replace(contract, cp_plan=None))}.items()),
    )
    return (*gathers, kernel, *cp_decode_suffix(op, contract))


GBSA_METADATA_RECIPE = "generic_block_sparse_attention_metadata"
GBSA_METADATA_TOTAL_WORDS = 1024
GBSA_METADATA_WORD_BYTES = 4
GBSA_METADATA_OUTPUT_BYTES = GBSA_METADATA_TOTAL_WORDS * GBSA_METADATA_WORD_BYTES
GBSA_METADATA_SERVICE = "aicpu_metadata_call"


def build_gbsa_metadata(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """AI_CPU GenericBlockSparseAttentionMetadata plan (CANN fixed 1024 int32 words = 4,096 B)."""
    meta = dict(op.metadata or ())
    spec = dict(meta.get("metadata_call") or {})
    read_bytes = int(spec.get("input_read_bytes", int(round(op.mte2_bytes or 0))))
    write_bytes = int(spec.get("output_write_bytes", GBSA_METADATA_OUTPUT_BYTES))
    read = Work(resources={"hbm": read_bytes})
    write = Work(resources={"hbm": write_bytes})
    priced = GBSA_METADATA_SERVICE in geometry.resource_units
    controller = Work(resources={GBSA_METADATA_SERVICE: 1}) if priced else Work()

    def program(context):
        return KernelProgram((
            Stage("inputs", context.bind(read)),
            Stage(
                "controller",
                context.bind(controller),
                after=("inputs",),
                placement={context.resource_names[GBSA_METADATA_SERVICE]: (0,)} if priced else {},
            ),
            Stage("metadata_output", context.bind(write), after=("controller",)),
        ))

    return OperatorPlan(
        read + controller + write,
        program,
        metadata={
            "execution_class": "aicpu",
            "count_boundary": "one_source_native_call_not_ISA",
            "output_words": GBSA_METADATA_TOTAL_WORDS,
            "output_bytes": write_bytes,
            "priced_service_work": {GBSA_METADATA_SERVICE: 1} if priced else {},
            "unpriced_service_work": {} if priced else {GBSA_METADATA_SERVICE: 1},
        },
    )


def recipes() -> dict[str, PlannedRecipe | PairwiseCollective]:
    """Expose exact recipe bindings through the existing source-recipe entrypoint."""
    planner = PlannedRecipe(build, _kernels)
    return {
        OUTPUT_LSE_EXCHANGE_RECIPE: PairwiseCollective(),
        RING_UPDATE_RECIPE: PlannedRecipe(build_ring_update),
        GBSA_METADATA_RECIPE: PlannedRecipe(build_gbsa_metadata),
        **{recipe: planner for recipe in ABI_BY_RECIPE},
    }


def make_attention_op(
    contract: AttentionContract, *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0,
    layer: int = 0, requests: tuple[str, ...] = (), dtype: str = "bf16",
) -> OpSpec:
    """Create a JSON-compatible source kernel declaration from the typed contract."""
    return_lse = contract.return_lse
    contract = history_decode_contract(contract)
    shape = history_output_shape(contract)
    family = "mla" if contract.abi in {C8_ABI, C8_V1_ABI, C8_VLLM_ABI, NOPE_ABI} else "attention"
    matrix_dtype = dtype
    if contract.quant_mode == QUANT_MODE_MXFP8 or _is_gbsa_fp8_quant(contract) or (
        contract.abi in MINDIE_ABIS and contract.quant_mode == QUANT_MODE_MINDIE_FP8
    ):
        matrix_dtype = "fp8_e4m3"
    if contract.abi == MINDIE_V3_ABI and contract.quant_mode in {QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX}:
        matrix_dtype = "mxfp4"
    hbm_bytes, mte2_bytes, mte3_bytes, vector_ops = _minimum_contract_io(contract, dtype)
    return OpSpec(
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        family=family, phase=contract.phase, recipe=RECIPE_BY_ABI[contract.abi],
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype=dtype, layout=contract.kv_layout,
        source_boundary="call" if contract.cp_plan else "kernel",
        flops=contract.useful_flops, matrix_flops=((matrix_dtype, contract.useful_flops),),
        vector_ops=vector_ops, hbm_bytes=hbm_bytes, mte2_bytes=mte2_bytes, mte3_bytes=mte3_bytes,
        returns=("lse",) if return_lse else (),
        metadata=(("attention_contract", asdict(contract)), ("native_kind", contract.abi)),
    )


def make_gbsa_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int, num_heads: int,
    num_kv_heads: int, head_dim: int, v_head_dim: int | None = None, selected_tokens: int,
    context_tokens: int | None = None, block_size: int = DEFAULT_GBSA_BLOCK_SIZE,
    kv_layout: str = DEFAULT_GBSA_KV_LAYOUT,
    quant_mode: int = QUANT_MODE_BF16, ssmax: bool = False, metadata_reuse: bool = False,
    batch_size: int | None = None, shared_prefix: bool = False,
) -> OpSpec:
    """Bind the legacy GBSA parameters to the governed ABI."""
    batch = (len(requests) or query_tokens) if batch_size is None else batch_size
    sequences = ()
    if context_tokens is not None and query_tokens > 0 and batch > 0:
        groups = min(query_tokens, batch)
        quotient, remainder = divmod(query_tokens, groups)
        sequences = tuple((quotient + (rank < remainder), context_tokens)
                          for rank in range(groups))
    contract = AttentionContract(
        query_tokens, num_heads, num_kv_heads, head_dim, head_dim if v_head_dim is None else v_head_dim,
        selected_tokens=selected_tokens, phase=phase, abi=GBSA_ABI, kv_layout=kv_layout,
        block_size=block_size, quant_mode=quant_mode, ssmax=ssmax, metadata_reuse=metadata_reuse,
        batch_size=batch, shared_prefix=shared_prefix, sequences=sequences,
    )
    return make_attention_op(contract, op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests)


def make_kv_quant_sfa_c8_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int, num_heads: int,
    kv_lora_rank: int = C8_LATENT_DIM, rope_dim: int | None = None,
    selected_tokens: int = DEFAULT_SFA_SELECTED_TOKENS,
    precompute_scale_addr: bool = True, nope_bf16: bool = False, batch_size: int | None = None,
    shared_prefix: bool = False, quant_mode: int = QUANT_MODE_FP8_C8,
    selection_lists: tuple[tuple[int, ...], ...] = (), kv_tensor_id: str = "", kv_version: int = 0,
) -> OpSpec:
    """Bind the legacy C8 or NoPE parameters to their distinct kernel ABI."""
    eff_rope = (0 if nope_bf16 else C8_ROPE_DIM) if rope_dim is None else rope_dim
    abi = NOPE_ABI if nope_bf16 else (C8_VLLM_ABI if eff_rope == 0 else C8_ABI)
    contract = AttentionContract(
        query_tokens, num_heads, 1, kv_lora_rank + eff_rope, kv_lora_rank, selected_tokens=selected_tokens,
        phase=phase, abi=abi, kv_layout="PA_BLK_BSND",
        quant_mode=QUANT_MODE_BF16 if nope_bf16 else quant_mode, precompute_scale_addr=precompute_scale_addr,
        batch_size=(len(requests) or query_tokens) if batch_size is None else batch_size,
        shared_prefix=shared_prefix, selection_lists=selection_lists, kv_tensor_id=kv_tensor_id, kv_version=kv_version,
    )
    return make_attention_op(contract, op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests)


from ..vector.ring_update import (
    RING_UPDATE_ABI, RING_UPDATE_RECIPE, RING_UPDATE_SOURCES, RingUpdateContract,
    build_ring_update, make_ring_update_op,
)

__all__ = [
    "SOURCES", "TARGET_BACKEND", "C8_VLLM_ABI",
    "build", "resolve_operator_plan", "register_family", "AttentionContract", "ContextParallelPlan",
    "c8_gather_accounting", "bsh_kv_access_plan", "solve_wide_head_tiling", "shipping_fia_panel_plan",
    "lower_cp_collectives", "make_gbsa_op", "make_kv_quant_sfa_c8_op", "make_attention_op",
    "RING_UPDATE_ABI", "RING_UPDATE_RECIPE", "RingUpdateContract", "build_ring_update", "make_ring_update_op",
]
