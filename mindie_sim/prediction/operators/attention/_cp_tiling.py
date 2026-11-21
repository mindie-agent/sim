"""Source-bound context parallel attention collectives.

[Role] Bind caller tensor shapes and lower their collective dependencies.
[Invariants] Each gather uses its input tensor shape and declared group.
[Timeouts & Locks] Finite graph construction without locking.
[Side-Effect Policy] Pure declarations on the shared execution graph.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from mindie_sim.catalog.ops import comm_op
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Work
from ...program import KernelProgram, Stage
from ...results import PipelinePhase, WorkItem

VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
VLLM_MLA_REVISION = "7a7ed073c481d5e3a2fc2194c5b879efb2d373c0"

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
        revisions = (VLLM_REVISION, VLLM_MLA_REVISION) if self.caller_site == CACHE_MLA_SITE else (VLLM_REVISION,)
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
        head_group, head_ranks = "dcp", plan.dcp_size
        if plan.pcp_size > 1 and plan.dcp_size > 1:
            if plan.dcp_size == plan.pcp_size:
                head_ranks = 1
            else:
                head_group, head_ranks = "tp", plan.distinct_head_ranks
                if head_ranks is None or plan.pcp_size * head_ranks != plan.dcp_size:
                    from .common import unmodeled
                    raise unmodeled("attention_cp_distinct_heads", "Bind the distinct TP head subgroup")
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


@dataclass(frozen=True, slots=True)
class _CollectivePrefix:
    collectives: tuple[OpSpec, ...]
    consumer: object

    def __call__(self, context):
        # A directly lowered rank-local plan takes a bound link. Graph execution
        # expands the same collectives through the existing topology recipes.
        stages = []
        for op in self.collectives:
            group = dict(op.metadata)["group_axis"]
            role = group if group in context.resource_names else "link"
            demand = context.bind(Work(resources={role: op.link_bytes}))
            stages.append(Stage(op.op_id, demand, after=op.deps))
        stages.append(Stage("attention", after=(stages[-1].name,), program=self.consumer(context)))
        return KernelProgram(tuple(stages))


def with_collectives(plan, collectives, sources):
    spec = plan.pipeline_spec
    spec = replace(spec, sources=(*spec.sources, *sources))
    if not collectives:
        return replace(plan, pipeline_spec=spec)
    prefix = tuple(PipelinePhase(op.op_id, op.deps, (WorkItem("link", "bytes", op.link_bytes),))
                   for op in collectives)
    consumer = tuple(replace(phase, after=(prefix[-1].phase_id,)) if not phase.after else phase
                     for phase in spec.phases)
    return replace(
        plan, program=_CollectivePrefix(collectives, plan.program),
        work=plan.work + Work(resources={"link": sum(op.link_bytes for op in collectives)}),
        pipeline_spec=replace(spec, phases=(*prefix, *consumer)),
    )
