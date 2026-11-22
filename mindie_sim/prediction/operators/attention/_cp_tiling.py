"""Source-bound context parallel attention collectives.

[Role] Bind caller tensor shapes and lower their collective dependencies.
[Invariants] Each collective uses its input tensor shape and declared group.
  History decode exchanges dtype-specific rows of packed output and FP32 LSE.
[Timeouts & Locks] Finite graph construction without locking.
[Side-Effect Policy] Pure declarations on the shared execution graph.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from mindie_sim.catalog.context_parallel import merge_update_elements
from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Work
from ...program import KernelProgram, Stage
from ...results import PipelinePhase, WorkItem

MERGE_OPS_PER_ELEM = 3.0
PACK_LSE_OPS = 4.0

VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
VLLM_MLA_REVISION = "7a7ed073c481d5e3a2fc2194c5b879efb2d373c0"
VLLM_STACK_REVISION = "405617bae227fe5b097ff163efaa60bb7f22c698"

CACHE_PUBLICATION = "cache_publication"
HISTORY_ATTENTION = "history_attention"
COMPLETE_KV_SPARSE_PREFILL = "complete_kv_sparse_prefill"
CACHE_GQA_TENSORS = ("cache_key", "cache_value")
CACHE_MLA_TENSORS = ("kv_no_split", "cos", "sin")
CACHE_GQA_SITE = "attention_v1._reshape_and_cache_pcp"
CACHE_MLA_SITE = "mla_v1.exec_kv_prefill"
HISTORY_PREFILL_SITE = "attention_cp._prefill_query_all_gather"
HISTORY_DECODE_SITE = "attention_cp._forward_decode_dcp"
COMPLETE_KV_SITE = "sfa_cp._execute_sparse_flash_attention_process"
OUTPUT_LSE_EXCHANGE_RECIPE = "attention_cp_output_lse_exchange"
DCP_NARROW_LSE_LANES = 4
CP_CALLERS = {
    CACHE_GQA_SITE: CACHE_PUBLICATION,
    CACHE_MLA_SITE: CACHE_PUBLICATION,
    HISTORY_PREFILL_SITE: HISTORY_ATTENTION,
    HISTORY_DECODE_SITE: HISTORY_ATTENTION,
    COMPLETE_KV_SITE: COMPLETE_KV_SPARSE_PREFILL,
}


@dataclass(frozen=True, slots=True)
class ContextParallelPlan:
    """Pinned caller and per-rank tensor shapes for one collective boundary.

    Tensor entries are (name, shape, element_bytes). Shapes include rank-local
    padding. MLA publication receives the already-sliced local tensors.
    Distinct head ranks bind the TP subgroup when DCP spans PCP copies.
    """

    stack_revision: str
    caller_site: str
    phase: str
    case: str
    tensors: tuple[tuple[str, tuple[int, ...], int], ...]
    pcp_size: int = 1
    dcp_size: int = 1
    decode_tokens: int = 0
    pcp_decode_sharded: bool = False
    distinct_head_ranks: int | None = None

    def __post_init__(self) -> None:
        tensors = tuple((name, tuple(shape), size) for name, shape, size in self.tensors)
        object.__setattr__(self, "tensors", tensors)
        # vLLM-Ascend removed GQA DCP in #21737. Bind history decode to its
        # verified caller revisions.
        revisions = {
            CACHE_MLA_SITE: (VLLM_REVISION, VLLM_MLA_REVISION),
            HISTORY_DECODE_SITE: (VLLM_REVISION, VLLM_STACK_REVISION),
        }.get(self.caller_site, (VLLM_REVISION,))
        if self.stack_revision not in revisions or CP_CALLERS.get(self.caller_site) != self.case:
            from .common import unmodeled
            raise unmodeled("attention_cp_caller", "Bind a verified vLLM-Ascend revision and caller site")
        if self.phase not in {"prefill", "decode"} or min(self.pcp_size, self.dcp_size) < 1:
            raise ModelingError("attention_cp_phase_or_group_invalid")
        if self.decode_tokens < 0 or not tensors or len({t[0] for t in tensors}) != len(tensors):
            raise ModelingError("attention_cp_tensor_inventory_invalid")
        if any(not name or not shape or min(shape) < 0 or size < 1 for name, shape, size in tensors):
            raise ModelingError("attention_cp_tensor_shape_invalid")
        if self.case == CACHE_PUBLICATION:
            expected = CACHE_GQA_TENSORS if self.caller_site == CACHE_GQA_SITE else CACHE_MLA_TENSORS
            if (tuple(name for name, _, _ in tensors) != expected
                    or len({shape[0] for _, shape, _ in tensors}) != 1
                    or (self.caller_site == CACHE_GQA_SITE and not self.pcp_decode_sharded
                        and tensors[0][1][0] < self.decode_tokens)
                    or (self.caller_site == CACHE_MLA_SITE and self.phase != "prefill")):
                raise ModelingError("attention_cp_cache_tensor_inventory_invalid")
        elif self.case == HISTORY_ATTENTION:
            site_phase = "prefill" if self.caller_site == HISTORY_PREFILL_SITE else "decode"
            if self.phase != site_phase or any(len(shape) < 2 for _, shape, _ in tensors):
                raise ModelingError("attention_cp_history_tensor_shape_invalid")
        elif self.phase != "prefill":
            raise ModelingError("attention_cp_complete_kv_requires_prefill")
        if self.distinct_head_ranks is not None and not 1 <= self.distinct_head_ranks <= self.dcp_size:
            raise ModelingError("attention_cp_distinct_head_ranks_invalid")

    @property
    def return_lse(self) -> bool:
        return self.case == HISTORY_ATTENTION

    @property
    def output_merge(self) -> bool:
        return self.case == HISTORY_ATTENTION


def _history_head_group(plan):
    if plan.pcp_size == 1 or plan.dcp_size == 1:
        return "dcp", plan.dcp_size
    if plan.dcp_size == plan.pcp_size:
        return "dcp", 1
    if plan.distinct_head_ranks is None or plan.pcp_size * plan.distinct_head_ranks != plan.dcp_size:
        from .common import unmodeled
        raise unmodeled("attention_cp_distinct_heads", "Bind the distinct TP head subgroup")
    return "tp", plan.distinct_head_ranks


def _history_decode(contract):
    return next((p for p in contract.cp_plan or () if p.caller_site == HISTORY_DECODE_SITE), None)


def history_decode_contract(contract):
    binding = _history_decode(contract)
    if binding is None:
        return contract
    _, ranks = _history_head_group(binding)
    name, shape, _ = binding.tensors[0]
    if (len(binding.tensors) != 1 or name != "query" or len(shape) != 3 or shape[0] != contract.query_tokens
            or shape[-1] != contract.head_dim or not shape[1]):
        raise ModelingError("attention_cp_history_decode_query_shape_invalid")
    heads = shape[1] * ranks
    if contract.num_heads not in {shape[1], heads}:
        raise ModelingError("attention_cp_history_decode_head_count_invalid")
    return replace(contract, num_heads=heads, return_lse=True)


def lower_cp_collectives(
    plan: ContextParallelPlan, *, op_id: str = "attention_cp", deps: tuple[str, ...] = (),
    stage: int = 0, layer: int = 0, requests: tuple[str, ...] = (),
) -> tuple[OpSpec, ...]:
    """Emit topology-bound all-gathers with tensor-level ownership metadata."""
    case = plan.case
    tensors = plan.tensors
    replicated = 0
    if case == CACHE_PUBLICATION:
        replicated = 0 if plan.pcp_decode_sharded else plan.decode_tokens
        if plan.caller_site == CACHE_GQA_SITE:
            tensors = tuple((name, (shape[0] - replicated, *shape[1:]), size)
                            for name, shape, size in tensors)
        steps = (("pcp", 0, plan.pcp_size, "cache_writer", "slot_mapping"),)
    elif case == HISTORY_ATTENTION:
        head_group, head_ranks = _history_head_group(plan)
        steps = ((head_group, 1, head_ranks, "history_attention", ""),)
        if plan.phase == "prefill":
            steps = (("pcp", 0, plan.pcp_size, "history_query", "query_restore"), *steps)
    else:
        steps = (("dcp", 0, plan.dcp_size, "local_sparse_prefill", ""),)

    ops = []
    for group, axis, ranks, consumer, restore in steps:
        local_bytes = sum(prod(shape) * size for _, shape, size in tensors)
        if ranks == 1 or local_bytes == 0:
            continue
        keys = tuple((plan.stack_revision, plan.caller_site, plan.phase, group, name, axis, consumer)
                     for name, _, _ in tensors)
        out_shapes = tuple((name, tuple(dim * ranks if i == axis else dim for i, dim in enumerate(shape)))
                           for name, shape, _ in tensors)
        op = comm_op(
            op_id=f"{op_id}.{plan.caller_site}.{case}.{group}.{axis}", deps=deps, stage=stage, layer=layer,
            requests=requests, family="allgather", phase=plan.phase, recipe="ring_allgather",
            link_bytes=local_bytes * (ranks - 1), shape=(local_bytes * ranks,),
            metadata=(("group_axis", "cp" if group == "dcp" else group), ("steps", ranks - 1), ("attention_cp_keys", keys),
                      ("input_tensors", tuple((name, shape) for name, shape, _ in tensors)),
                      ("output_tensors", out_shapes), ("restore", restore),
                      ("return_lse", plan.return_lse), ("output_merge", plan.output_merge),
                      ("replicated_decode_tokens", replicated)),
        )
        ops.append(op)
        deps = (op.op_id,)
        tensors = tuple((name, shape, size) for (_, _, size), (name, shape) in zip(tensors, out_shapes))
    return tuple(ops)


def cp_operators(op: OpSpec, bindings: tuple[ContextParallelPlan, ...]) -> tuple[OpSpec, ...]:
    collectives = []
    deps = ()
    for binding in bindings:
        gathered = lower_cp_collectives(binding, op_id=f"gather{len(collectives)}", deps=deps,
                                        stage=op.stage, layer=op.layer, requests=op.requests)
        if gathered:
            collectives.extend(gathered)
            deps = (gathered[-1].op_id,)
    return tuple(collectives)


def cp_decode_suffix(op, contract):
    binding = _history_decode(contract)
    if binding is None or binding.dcp_size == 1:
        return ()
    group, head_ranks = _history_head_group(binding)
    q, heads, width = contract.query_tokens, contract.num_heads // head_ranks, contract.v_head_dim
    dtype = op.dtype or "bf16"
    if dtype in {"bf16", "bfloat16", "fp16", "float16"}:
        lse_lanes = DCP_NARROW_LSE_LANES
    elif dtype in {"fp32", "float32"}:
        lse_lanes = 1
    else:
        from .common import unmodeled
        raise unmodeled("attention_dcp_output_lse_dtype", f"Bind a supported DCP output dtype for {dtype}")
    element_bytes = dtype_bytes(dtype)
    out_bytes = q * heads * width * element_bytes
    packed_shape = (q, heads, width + lse_lanes)
    payload = prod(packed_shape) * element_bytes
    tensors = (("output", (q, heads, width)), ("lse", (q, heads, 1)))
    payload_metadata = (("payload_tensors", tensors), ("packed_shape", packed_shape))
    arguments = dict(stage=op.stage, layer=op.layer, requests=op.requests, phase=op.phase)
    pack = vector_op(
        op_id="output_lse_pack", deps=("attention",), **arguments, family="elementwise", recipe="dcp_output_pack",
        shape=(q, heads * head_ranks, packed_shape[-1]), dtype=dtype,
        elements=q * heads * head_ranks, ops_per_element=PACK_LSE_OPS,
        read_bytes_override=(out_bytes + q * heads * dtype_bytes("fp32")) * head_ranks,
        write_bytes_override=payload * head_ranks,
    )
    ops = [pack]
    deps = (pack.op_id,)
    if head_ranks > 1:
        ops.append(comm_op(
            op_id="output_lse_exchange", deps=deps, **arguments, family="alltoall", recipe=OUTPUT_LSE_EXCHANGE_RECIPE,
            link_bytes=payload * (head_ranks - 1), shape=packed_shape,
            metadata=(("group_axis", "cp" if group == "dcp" else group), ("collective", "alltoall"),
                      ("source", "dcp_a2a_fused"), *payload_metadata),
        ))
        deps = (ops[-1].op_id,)
    if binding.pcp_size > 1:
        ops.append(comm_op(
            op_id="output_lse_pcp_gather", deps=deps, **arguments, family="allgather", recipe="ring_allgather",
            link_bytes=payload * head_ranks * (binding.pcp_size - 1),
            shape=(binding.dcp_size, *packed_shape),
            metadata=(("group_axis", "pcp"), ("steps", binding.pcp_size - 1), *payload_metadata),
        ))
        deps = (ops[-1].op_id,)
    ops.append(vector_op(
        op_id="output_lse_merge", deps=deps, **arguments, family="attention_merge", recipe="dcp_lse_merge",
        shape=(q, heads, width), dtype="fp32", semantic_family="dcp_logsumexp_attention_update",
        elements=int(merge_update_elements(q, heads, width, binding.dcp_size)), ops_per_element=MERGE_OPS_PER_ELEM,
        read_bytes_override=payload * binding.dcp_size, write_bytes_override=out_bytes,
        metadata=(("partial_reduction", "logsumexp"), *payload_metadata),
    ))
    return tuple(ops)


def _cp_work(op):
    if op.link_bytes:
        return Work(resources={"link": op.link_bytes})
    return Work(resources={"hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes,
                           "vector": op.vector_ops})


@dataclass(frozen=True, slots=True)
class _CollectiveAttention:
    prefix: tuple[OpSpec, ...]
    consumer: object
    suffix: tuple[OpSpec, ...]

    def __call__(self, context):
        # A directly lowered rank-local plan takes a bound link. Graph execution
        # expands the same collectives through the existing topology recipes.
        def stage(op):
            work = _cp_work(op)
            group = dict(op.metadata).get("group_axis")
            if group in context.resource_names:
                work = Work(resources={group: op.link_bytes})
            return Stage(op.op_id, context.bind(work), after=op.deps)

        prefix = tuple(stage(op) for op in self.prefix)
        consumer = Stage("attention", after=(prefix[-1].name,) if prefix else (), program=self.consumer(context))
        return KernelProgram((*prefix, consumer, *(stage(op) for op in self.suffix)))


def with_collectives(plan, collectives, suffix, sources):
    spec = plan.pipeline_spec
    spec = replace(spec, sources=(*spec.sources, *sources))
    if not (collectives or suffix):
        return replace(plan, pipeline_spec=spec)
    prefix = tuple(PipelinePhase(op.op_id, op.deps, (WorkItem("link", "bytes", op.link_bytes),))
                   for op in collectives)
    consumer = tuple(replace(phase, after=(prefix[-1].phase_id,)) if prefix and not phase.after else phase
                     for phase in spec.phases)
    consumed = {dep for phase in consumer for dep in phase.after}
    terminals = tuple(phase.phase_id for phase in consumer if phase.phase_id not in consumed)
    work = plan.work
    post = []
    for op in (*collectives, *suffix):
        work += _cp_work(op)
    for op in suffix:
        deps = terminals if op.deps == ("attention",) else op.deps
        items = tuple(WorkItem(role, "op" if role == "vector" else "bytes", amount)
                      for role, amount in _cp_work(op).resources.items() if amount)
        post.append(PipelinePhase(op.op_id, deps, items))
    return replace(
        plan, program=_CollectiveAttention(collectives, plan.program, suffix), work=work,
        pipeline_spec=replace(spec, phases=(*prefix, *consumer, *post)),
    )
