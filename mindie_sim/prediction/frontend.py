"""Model structure to operator graph, with pluggable semantic builders.

[Role] Compose model blocks, media, speculative work and software graph capture.
[Invariants] There is one target body per verify pass. Dependencies express
  actual producers and consumers. A model name and a calibration row cannot
  choose a code path; semantic kinds dispatch through explicit registries.
[Timeouts & Locks] O(active layer graph), no iteration history, device or I/O.
[Side-Effect Policy] Creates immutable operator graphs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.engram import lower_engram
from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, lower_ple
from mindie_sim.catalog.media import lower_media
from mindie_sim.catalog.ops import barrier_op, comm_op, gemm_op, vector_op
from mindie_sim.catalog.context_parallel import pcp_rank_chunk_ranges, split_pcp_pairs
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.subgraphs import (
    EventPortSpec,
    PortSet,
    StatePortSpec,
    TensorPortSpec,
)
from mindie_sim.schema.workload import selected_mode
from .attention import ATTENTION_BUILDERS, AttentionPolicy
from .blocks import RESIDUAL_BUILDERS, lowrank_hyper_final_mixer, norm
from .feedforward import FFN_BUILDERS, RoutingPolicy
from .frontend_features import FeatureConfig
from .graph import GraphPolicy, annotate_geometry
from .routing_realizations import RoutingRealization, select_realization
from .structure import normalize_model, stage_map
from . import family_attnres, sequence_parallel
from .implementation_selection import (
    resolve as resolve_implementations,
    LoweringRegion,
    lower_regions,
    selection_delivery,
    annotate as annotate_selections,
)
from .recover_ssm_commit import bind_recover_ssm_commits, is_recover_ssm_intent
from .recover_ssm_work import recover_ssm_enabled
from .recurrent_kda_helpers import strip_excluded_native_kda


ZERO = 0
ONE = 1
_ENV_MODEL = 0
_ENV_RUNTIME = 1
_ENV_MACHINE = 2
_ENV_REPLICA = 3
_ENV_FEATURES = 4
ZERO_FLOAT = 0.0
FINAL_LAYER = -1
MIN_DBO_MICROBATCHES = 2


def _partition_global_tokens(shared_global, region_rows):
    """Apportion one cohort's routed rows across microbatches without loss.

    A per-region ``round`` loses a row whenever every partition is below one
    half (``[1, 1]`` of five rows rounds to ``[2, 2]``).  The physical routing
    ledger is conserved, so the integer quotas are the floor of the exact
    proportions and the remainder goes to the largest fractional parts.
    """
    rows = tuple(max(ZERO, int(value)) for value in region_rows)
    if shared_global is None or not rows or sum(rows) <= ZERO:
        return tuple(None for _ in rows)
    total = float(sum(rows))
    target = int(round(float(shared_global)))
    quotas = [target * float(value) / total for value in rows]
    partitions = [int(math.floor(quota)) for quota in quotas]
    remainder = max(ZERO, target - sum(partitions))
    order = sorted(range(len(rows)),
                   key=lambda index: (-(quotas[index] - partitions[index]), index))
    for index in order[:remainder]:
        partitions[index] += ONE
    return tuple(partitions)


def _cohort_source_rows(views):
    """Routed source rows a view set owns on one physical EP replica."""
    return sum(int(view.q_len) + int(view.draft_proposal_tokens) for view in views)


def _scope(ops, block, role="body"):
    """The composing builder owns scope; operator names do not identify it."""
    return [replace(op, metadata=tuple({
        **dict(op.metadata), "structure_block": block, "structure_role": role,
    }.items())) for op in ops]


# Service-only fragments. A pure-model assembly removes these from the selected
# program; the generic execution graph still prices them as ordinary nodes.
SERVICE_FAMILIES = frozenset({"sampling", "acceptance", "token_commit"})
_TRIGGERS = {
    "sampling": "per_invocation",
    "acceptance": "per_invocation_when_proposals",
    "token_commit": "per_request_per_invocation",
}


def _without_service_fragments(regions):
    """Remove service-only ops from ordered regions and reconnect consumers.

    Dependencies are rewired to the removed op's own predecessors, so a pure
    model keeps the same shared work and readiness edges without inventing a
    layer or a zero-work service marker. Regions are processed in execution
    order because draft/commit boundaries can cross a region edge.
    """
    aliases = {}
    result = []
    for region in regions:
        kept = []
        for op in region.ops:
            deps = tuple(dict.fromkeys(
                parent for dep in op.deps for parent in aliases.get(dep, (dep,))))
            if op.family in SERVICE_FAMILIES and op.family != "acceptance":
                aliases[op.op_id] = deps
                continue
            kept.append(replace(op, deps=deps))
        result.append(replace(region, ops=tuple(kept)))
    return tuple(result)


def _strip_recover_intents(ops):
    """Remove deferred commit intents and reconnect their consumers.

    A body/model assembly prices only forward work; the post-acceptance state
    publication belongs to the service assembly, so a leftover deferred intent
    must never make a pure body depend on a sampling event.
    """
    intents = {op.op_id: op for op in ops if is_recover_ssm_intent(op)}
    if not intents:
        return tuple(ops)

    def resolve(dep, seen):
        if dep not in intents or dep in seen:
            return (dep,)
        parents = tuple(
            parent for parent in intents[dep].deps
            if parent != dep)
        return tuple(
            item for parent in parents
            for item in resolve(parent, seen | {dep}))

    result = []
    for op in ops:
        if op.op_id in intents:
            continue
        deps = tuple(dict.fromkeys(
            parent for dep in op.deps for parent in resolve(dep, frozenset())))
        result.append(replace(op, deps=deps) if deps != op.deps else op)
    return tuple(result)


def _declared_format(op, *keys):
    """Read this operator's own representation; never inherit model dtype."""
    metadata = dict(op.metadata)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _ports_for_work(work):
    """Typed boundary ports for one executable model program.

    A barrier has readiness but no tensor payload: it gets an event port only.
    A tensor port uses that boundary's declared input/output format. If the
    operator does not declare one, the port keeps an empty dtype explicitly
    rather than substituting the model activation dtype.
    """
    consumed = {dep for op in work.ops for dep in op.deps}
    tensor_in, tensor_out, state_out = [], [], []
    event_in, event_out = [], []
    for op in work.ops:
        payload = op.family != "barrier"
        input_dtype = _declared_format(op, "input_format", "activation_format")
        output_dtype = _declared_format(op, "output_format", "output_dtype")
        if not op.deps:
            if payload:
                tensor_in.append(TensorPortSpec(
                    name="input." + op.op_id, direction="in",
                    representation=op.recipe or op.family, dtype=input_dtype,
                    scope="cohort", readiness="external"))
            event_in.append(EventPortSpec(
                name="ready." + op.op_id, direction="in", event="ready",
                scope="cohort", readiness="external"))
        if op.op_id not in consumed and op.family not in {
                "barrier", "sampling", "acceptance", "token_commit"}:
            tensor_out.append(TensorPortSpec(
                name="output." + op.op_id, direction="out",
                representation=op.recipe or op.family, dtype=output_dtype,
                scope="request", readiness="producer_complete"))
            event_out.append(EventPortSpec(
                name="complete." + op.op_id, direction="out", event="complete",
                scope="request", readiness="producer_complete"))
        if op.family == "token_commit":
            event_out.append(EventPortSpec(
                name="commit." + op.op_id, direction="out", event="commit",
                scope="request", readiness="producer_complete"))
        elif op.family in {"sampling", "acceptance"}:
            event_out.append(EventPortSpec(
                name=("sample." if op.family == "sampling" else "accept.") + op.op_id,
                direction="out", event="complete", scope="request",
                readiness="producer_complete"))
    if any(op.family == "token_commit" for op in work.ops):
        state_out.append(StatePortSpec(
            name="state.token_committed", direction="out",
            representation="token_position", scope="request",
            readiness="producer_complete"))
    return PortSet(
        tensor_in=tuple(tensor_in), tensor_out=tuple(tensor_out),
        state_out=tuple(state_out), event_in=tuple(event_in),
        event_out=tuple(event_out),
    )


ANALYTICAL_SELECTIONS = "analytical_implementations"


def mark_analytical_runtime(runtime, selections):
    """Retain policy-resolved analytical decisions on the immutable runtime.

    A policy may remove a selector after choosing its analytical fallback.
    Downstream source-default resolvers must see that decision instead of
    treating the absent binding as permission to install a native source.
    """
    names = {str(record.get("selector"))
             for record in (selections or ())
             if record.get("implementation") == "analytical"}
    names.discard("")
    if not names:
        return runtime
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    previous = tuple(metadata.get(ANALYTICAL_SELECTIONS, ()) or ())
    metadata[ANALYTICAL_SELECTIONS] = tuple(sorted(set(previous) | names))
    return replace(runtime, metadata=tuple(metadata.items()))


def program_attached_to(program, ops):
    """Reattach one program to its final cohort-bound operator occurrences."""
    work = replace(program.work, ops=tuple(ops))
    counts = Counter(op.family for op in work.ops)
    triggers = tuple((family, counts[family], _TRIGGERS[family])
                     for family in sorted(SERVICE_FAMILIES) if counts[family])
    return replace(program, work=work, ports=_ports_for_work(work),
                   triggers=triggers)


@dataclass(frozen=True)
class ModelProgram:
    """Executable model work plus its typed ports and occurrence triggers.

    ``work`` stays the same ``IterationWork`` consumed by ``Execution.add``;
    the ports/triggers describe assembly only and never add device work.
    """

    work: IterationWork
    ports: PortSet
    triggers: tuple[tuple[str, int, str], ...]
    assembly: str
    layer_scope: str

    def to_dict(self, *, include_ops=False):
        return {
            "assembly": self.assembly,
            "layer_scope": self.layer_scope,
            "valid_tokens": self.work.valid_tokens,
            "padded_tokens": self.work.padded_tokens,
            "mode": self.work.mode,
            "graph_bucket": self.work.graph_bucket,
            "replay_scope": self.work.replay_scope,
            "ports": self.ports.to_dict(),
            "unbound_tensor_ports": [
                port.name for port in (*self.ports.tensor_in, *self.ports.tensor_out)
                if not port.dtype],
            "triggers": [
                {"family": family, "occurrences": count, "trigger": trigger}
                for family, count, trigger in self.triggers
            ],
            **({"ops": [op.to_dict() for op in self.work.ops]}
               if include_ops else {}),
        }


class Frontend:
    def __init__(self, *, attention=None, ffn=None, residuals=None, graph=None,
                 attention_policy=None, routing=None, features=None,
                 resource_fusion=True):
        self.features = features
        self.resource_fusion = bool(resource_fusion)
        self.attention = {**ATTENTION_BUILDERS, **(attention or {})}
        self.ffn = {**FFN_BUILDERS, **(ffn or {})}
        self.residuals = {**RESIDUAL_BUILDERS, **(residuals or {})}
        self.graph = graph or GraphPolicy()
        self.attention_policy = attention_policy or AttentionPolicy()
        self.routing = routing or RoutingPolicy()
        self._prev_work_ops: tuple | None = None
        self._prev_arch_state: tuple | None = None
        self._prev_call_env: tuple | None = None
        self._prev_homo_env: tuple | None = None

    def body(self, model, runtime, views, *, prefix="", ready="", tokens=None,
             mode="prefill", global_tokens=None, stages=None, replica=ZERO,
             features=None, source_rows_by_replica=None,
             routing_global_tokens=None, routing_views=None,
             cohort_views_by_replica=None,
             physical_rank=None, query_position=None,
             include_recover_commits: bool = False, machine=None,
             dbo_partition: int | None = None, layer_limit: int | None = None,
             topk_scope=None, reuse_index_selection: bool = False):
        p = runtime.parallel
        if layer_limit is not None:
            limit = max(ONE, int(layer_limit))
            model = replace(model, layer_plans=tuple(model.layer_plans[:limit]))
        features = self.features if features is None else features
        # An explicit mechanism exclusion owns every selector it can install,
        # including the structural body default resolved before this call.
        runtime = strip_excluded_native_kda(runtime)
        # DBO splits the same EP collective into microbatches, so every EP
        # peer must meet the split threshold.  Letting only the peers with two
        # views split changes the collective cohort and breaks the inverse
        # traffic rendezvous for an otherwise valid batch.
        cohort_rows = tuple(max(ZERO, int(value))
                            for value in (source_rows_by_replica or ()))
        # Only an EP collective needs one split decision across peers.  A pure
        # DP replica set has no cross-replica rendezvous, so DBO may split an
        # active replica on its own.
        ep_cohort = int(getattr(runtime.parallel, "ep", ONE) or ONE) > ONE
        dbo_cohort_ready = (not ep_cohort or not cohort_rows
                            or min(cohort_rows) >= MIN_DBO_MICROBATCHES)
        if (runtime.enable_dbo and dbo_partition is None and len(views) > ONE
                and dbo_cohort_ready):
            from .overlap_plan import MicrobatchPolicy, build_microbatch_regions
            from .physical_recipe_registry import COLLECTIVE_FAMILIES
            regions = build_microbatch_regions(
                views, policy=MicrobatchPolicy(count=MIN_DBO_MICROBATCHES))
            if len(regions) > ONE and reuse_index_selection:
                # A shared TopK selection is one per proposal step; splitting
                # that step into independent microbatches has no single owner.
                raise ModelingError("mtp_shared_index_selection_requires_single_microbatch")
            if len(regions) > ONE:
                merged, previous_partition_attention, sinks = [], None, []
                partition_requests = tuple(view.request_id for view in views)
                shared_global = (routing_global_tokens
                                 if routing_global_tokens is not None
                                 else global_tokens)
                region_rows = tuple(
                    sum(int(view.q_len) + int(view.draft_proposal_tokens)
                        for view in region.views) for region in regions)
                partition_globals = _partition_global_tokens(
                    shared_global, region_rows)
                peer_regions = (
                    tuple(tuple(build_microbatch_regions(
                        peer_views, policy=MicrobatchPolicy(
                            count=MIN_DBO_MICROBATCHES)))
                          for peer_views in cohort_views_by_replica)
                    if cohort_views_by_replica is not None else None)
                for region, region_rows_value, partition_global in zip(
                        regions, region_rows, partition_globals):
                    region_ids = {view.request_id for view in region.views}
                    partition_routing = (
                        tuple(view for view in routing_views
                              if view.request_id in region_ids) or None
                        if routing_views is not None else None)
                    partition_source_rows = source_rows_by_replica
                    if peer_regions is not None:
                        parts = tuple(
                            tuple(view for peer_region in peer_plan
                                  if peer_region.index == region.index
                                  for view in peer_region.views)
                            for peer_plan in peer_regions)
                        partition_source_rows = tuple(
                            _cohort_source_rows(part) for part in parts)
                        partition_routing = tuple(
                            view for part in parts for view in part) or None
                        partition_global = sum(partition_source_rows)
                    elif cohort_rows:
                        count = len(regions)
                        partition_source_rows = tuple(
                            region_rows_value
                            if peer == replica and sum(region_rows) == cohort_rows[peer]
                            else (cohort_rows[peer] + count - ONE - region.index) // count
                            for peer in range(len(cohort_rows)))
                        partition_global = sum(partition_source_rows)
                    part_ops, part_last = self.body(
                        model, runtime, region.views,
                        prefix=f"{prefix}dbo{region.index}.",
                        ready=ready,
                        tokens=region_rows_value,
                        mode=mode,
                        global_tokens=(partition_global
                                       if partition_global is not None
                                       else global_tokens),
                        stages=stages,
                        replica=replica, features=features,
                        source_rows_by_replica=partition_source_rows,
                        routing_global_tokens=partition_global,
                        routing_views=partition_routing,
                        physical_rank=physical_rank,
                        query_position=query_position,
                        include_recover_commits=include_recover_commits,
                        machine=machine, dbo_partition=region.index)
                    first_attention = None
                    for op in part_ops:
                        collective = op.family in COLLECTIVE_FAMILIES
                        auxiliary = (op.stream
                                     if op.stream and op.stream != region.stream
                                     else "")
                        stream = (f"dbo{region.index}:{auxiliary}" if auxiliary
                                  else region.comm_stream if collective
                                  else region.stream)
                        metadata = dict(op.metadata)
                        metadata.update({
                            "dbo_microbatch": region.index,
                            "dbo_microbatch_count": len(regions),
                            "dbo_compute_stream": region.stream,
                        })
                        if auxiliary:
                            metadata["dbo_auxiliary_stream"] = auxiliary
                        if collective:
                            metadata["collective_rendezvous"] = (
                                f"dbo{region.index}:{op.op_id}")
                        if op.family == "attention" and first_attention is None:
                            first_attention = op.op_id
                        merged.append(replace(
                            op, stream=stream,
                            metadata=tuple(sorted(metadata.items()))))
                    if previous_partition_attention and first_attention:
                        position = next(
                            index for index, op in enumerate(merged)
                            if op.op_id == first_attention)
                        merged[position] = replace(
                            merged[position],
                            deps=tuple(dict.fromkeys(
                                (*merged[position].deps,
                                 previous_partition_attention))))
                    previous_partition_attention = (
                        first_attention or previous_partition_attention)
                    sinks.append(part_last)
                # Every partition's residual/FFN sink must finish before the
                # shared final norm, logits and token commit.  Returning only
                # the last partition's tail lets the commit overtake partition
                # zero and understates the DBO step.
                joined = barrier_op(
                    op_id=prefix + "dbo.join", deps=tuple(dict.fromkeys(sinks)),
                    stage=stages[ZERO] if stages else ZERO, layer=-1,
                    requests=partition_requests, phase=mode,
                    recipe="dbo_partition_join", shape=(len(partition_requests),))
                merged.append(joined)
                return merged, joined.op_id
        requests = tuple(view.request_id for view in views)
        tokens = sum(view.q_len for view in views) if tokens is None else tokens
        stages = stages or stage_map(model, runtime)
        ops, previous = [], ready
        sp = sequence_parallel.select(runtime, model, views, tokens=tokens,
                                      global_tokens=global_tokens)
        if sp:
            if runtime.moe_recipe != "alltoall":
                raise ValueError("sequence_parallel_requires_unchunked_alltoall")
            tokens, global_tokens = sp.tokens_local, sp.cohort_tokens
            if tokens:
                shard = sequence_parallel.input_shard(
                    plan=sp, op_id=prefix + "sp_shard", deps=(previous,),
                    stage=stages[0] if stages else 0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(shard)
                previous = shard.op_id
        caches, indices, topks, candidates = {}, {}, {}, {}
        if topk_scope is not None:
            # Reuse a TopK selection produced by an earlier proposal step.
            topks = topk_scope
        residual_sources = []
        attnres = (family_attnres.selection(
            runtime, model=model, phase=mode)
            if (tokens and mode not in {"draft", "draft_prefill"}) else None)
        attn_state = None
        if attnres is not None:
            if len(set(stages)) != 1 and runtime.parallel.pp == 1:
                raise ModelingError("attn_res_source_caller_requires_pp1")
            attn_state = family_attnres.AttnResPrefixState.start(
                model=model, runtime=runtime, ready=previous, tokens=tokens,
                mode=mode, requests=requests,
                optimize_prefill=attnres["optimize_prefill"])
        for index, layer in enumerate(model.layer_plans):
            stage = stages[index]
            name = f"{prefix}s{stage}.l{index}"
            common = dict(prefix=name, plan=layer, stage=stage, requests=requests,
                          tokens=tokens, model=model, mode=mode,
                          residual_sources=tuple(residual_sources),
                          features=features,
                          source_rows_by_replica=source_rows_by_replica)
            residual = self.residuals["lowrank_hyper" if getattr(layer, "hc_lowrank", ZERO) > ZERO else
                                      "hyper" if layer.hyper_connections else
                                      "attention" if layer.residual_block_size else "plain"]
            if tokens:
                if attn_state is not None:
                    prepared = attn_state.prepare(
                        prefix=name + ".attention", layer=layer, stage=stage)
                    ops.extend(_scope(prepared.ops, "attention", "residual"))
                    hidden = prepared.hidden_id
                else:
                    if layer.engram and is_qwen4_exp_ple(model.engram):
                        extra, previous = lower_ple(
                            prefix=name, plan=layer, stage=stage, deps=previous,
                            model=model, parallel=p, quant=model.quant, mode=mode,
                            padded_tokens=tokens, views=views, runtime=runtime)
                        ops.extend(extra)
                    pre, hidden, saved = residual.before(
                        {**common, "prefix": name + ".attention", "structure_block": "attention"}, previous)
                    ops.extend(_scope(pre, "attention", "residual"))
                    if getattr(layer, "hc_lowrank", ZERO) <= ZERO:
                        normalization = norm(
                            name=name + ".norm1", deps=(hidden,), stage=stage, layer=index,
                            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                        )
                        ops.extend(_scope((normalization,), "attention"))
                        hidden = normalization.op_id
                attention_ready = hidden
                if sp:
                    gathered = sequence_parallel.allgather(
                        plan=sp, op_id=name + ".attention.sp_gather",
                        deps=(attention_ready,), stage=stage, layer=index,
                        requests=requests, phase=mode, width=model.hidden_size,
                        dtype=model.activation_dtype)
                    ops.extend(_scope((gathered,), "attention"))
                    attention_ready = gathered.op_id
                    if sp.physical_global != sp.attention_tokens:
                        sliced = sequence_parallel.attention_input(
                            plan=sp, op_id=name + ".attention.sp_input_view",
                            deps=(attention_ready,), stage=stage, layer=index,
                            requests=requests, phase=mode, width=model.hidden_size,
                            dtype=model.activation_dtype)
                        ops.extend(_scope((sliced,), "attention"))
                        attention_ready = sliced.op_id
                selection_kwargs = (
                    {"reuse_index_selection": True, "shared_topk": topks}
                    if reuse_index_selection and layer.attention == "mla" else {})
                attention = self.attention[layer.attention](
                    prefix=name, plan=layer, views=views, stage=stage, deps=attention_ready,
                    model=model, parallel=p, quant=model.quant, mode=mode,
                    padded_tokens=sp.attention_tokens if sp else tokens,
                    cache_ids=caches, topk_ids=topks, index_cache_ids=indices,
                    candidate_ids=candidates, stage_of=lambda layer: stages[layer],
                    cp_interleave_size=runtime.kv_block_size, runtime=runtime,
                    policy=self.attention_policy, machine=machine,
                    **selection_kwargs,
                )
                if sp:
                    attention_ops, attention_last = sequence_parallel.close_output(
                        attention.ops, attention.last, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype, input_rows=sp.attention_tokens)
                    attention = replace(attention, ops=attention_ops, last=attention_last)
                ops.extend(_scope(attention.ops, "attention"))
                for mapping, result in ((caches, attention.cache_store_id), (indices, attention.index_store_id),
                                        (topks, attention.topk_id), (candidates, attention.candidate_id)):
                    if result:
                        mapping[layer.index] = result
                if attention.index_store_id:
                    caches["index", layer.index] = attention.index_store_id
                if attn_state is not None:
                    post = attn_state.attention_added(
                        prefix=name + ".ffn", layer=layer, stage=stage,
                        attention_id=attention.last)
                    ops.extend(_scope(post.ops, "ffn", "residual"))
                else:
                    post, previous = residual.after({**common, "prefix": name + ".attention",
                                                    "structure_block": "attention"},
                                                    attention.last, saved)
                    ops.extend(_scope(post, "attention", "residual"))
                if layer.engram and not is_qwen4_exp_ple(model.engram):
                    if attn_state is not None:
                        raise ModelingError("attn_res_selected_with_engram_unsupported")
                    extra, previous = lower_engram(
                        prefix=name, plan=layer, stage=stage, deps=previous,
                        model=model, parallel=p, quant=model.quant, mode=mode,
                        padded_tokens=tokens, views=views, runtime=runtime,
                    )
                    ops.extend(extra)
                if attn_state is not None:
                    hidden = post.hidden_id
                else:
                    pre, hidden, saved = residual.before(
                        {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous)
                    ops.extend(_scope(pre, "ffn", "residual"))
                    if getattr(layer, "hc_lowrank", ZERO) <= ZERO:
                        normalization = norm(
                            name=name + ".norm2", deps=(hidden,), stage=stage, layer=index,
                            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                        )
                        ops.extend(_scope((normalization,), "ffn"))
                        hidden = normalization.op_id
            else:
                hidden = previous
            if tokens or layer.ffn == "moe":
                routing_tokens = (routing_global_tokens
                                  if routing_global_tokens is not None
                                  else global_tokens)
                # explicit token ledgers match genuine serving request views or
                # an optional target-graph correlation passed by the caller.
                realization_views = routing_views if routing_views is not None else views
                realization = (select_realization(
                    runtime, layer, phase=mode, tokens=tokens,
                    valid_tokens=(sp.valid_per_tp[0] if sp else
                                  sum(view.logical_q_len for view in views)),
                    global_tokens=routing_tokens,
                    token_shards=(p.tp if sp and (routing_tokens is None
                                                  or routing_tokens >= tokens * p.tp)
                                  else 1),
                    valid_source_tokens=sp.valid_global if sp else None, replica=replica,
                    views=realization_views,
                    physical_rank=physical_rank,
                    query_position=query_position)
                    if layer.ffn == "moe" else None)
                ffn_tokens = tokens
                if sp and layer.ffn == "dense":
                    gathered = sequence_parallel.allgather(
                        plan=sp, op_id=name + ".ffn.sp_gather", deps=(hidden,),
                        stage=stage, layer=index, requests=requests, phase=mode,
                        width=model.hidden_size, dtype=model.activation_dtype)
                    ops.extend(_scope((gathered,), "ffn"))
                    hidden, ffn_tokens = gathered.op_id, sp.tokens_global
                ffn_ops, previous = self.ffn[layer.ffn](
                    **{**common, "tokens": ffn_tokens},
                    deps=hidden, parallel=p, global_tokens=global_tokens,
                    routing=self.routing, routing_realization=realization,
                    moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
                    overlap_shared=bool(runtime.multistream_overlap_shared_expert),
                    runtime=runtime, sequence_plan=sp,
                    ep_owner=(int(physical_rank) % int(p.ep)
                              if physical_rank is not None and int(p.ep) > ZERO
                              else None),
                )
                if sp and layer.ffn == "dense":
                    ffn_ops, previous = sequence_parallel.close_output(
                        ffn_ops, previous, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype)
                ops.extend(_scope(ffn_ops, "ffn"))
            if tokens:
                if attn_state is not None:
                    attn_state.defer_mlp(previous)
                else:
                    post, previous = residual.after(
                        {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous, saved)
                    ops.extend(_scope(post, "ffn", "residual"))
            if attn_state is None and layer.residual_block_size and (index + 1) % layer.residual_block_size == 0:
                residual_sources.append(previous)
            if index + 1 < len(stages) and stages[index + 1] != stage:
                transfer = comm_op(
                    op_id=name + ".pp", deps=(previous,), stage=stage, layer=index,
                    requests=requests, family="pp_p2p", phase=mode, recipe="pipeline_send",
                    link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype)
                               * max(1, layer.hc_mult), shape=(tokens, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])),
                )
                ops.append(transfer)
                previous = transfer.op_id
                if attn_state is not None:
                    if attn_state.pending_mlp_id is not None:
                        attn_state.pending_mlp_id = previous
                        ptx = comm_op(
                            op_id=f"{name}.prefix_pp", deps=(attn_state.prefix_id,),
                            stage=stage, layer=index, requests=requests, family="pp_p2p",
                            phase=mode, recipe="pipeline_residual_send",
                            link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                            shape=(tokens, model.hidden_size),
                            metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                        ops.append(ptx)
                        attn_state.prefix_id = ptx.op_id
                    else:
                        attn_state.prefix_id = previous
                    moved_banks = list(attn_state.bank_ids)
                    for slot_idx, bank_id in enumerate(moved_banks):
                        if bank_id is not None:
                            btx = comm_op(
                                op_id=f"{name}.residual_pp{slot_idx}", deps=(bank_id,),
                                stage=stage, layer=index, requests=requests, family="pp_p2p",
                                phase=mode, recipe="pipeline_residual_send",
                                link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                                shape=(tokens, model.hidden_size),
                                metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                            ops.append(btx)
                            moved_banks[slot_idx] = btx.op_id
                    attn_state.bank_ids = tuple(moved_banks)
                moved = []
                for source_index, source in enumerate(residual_sources):
                    transfer = comm_op(
                        op_id=f"{name}.residual_pp{source_index}", deps=(source,),
                        stage=stage, layer=index, requests=requests, family="pp_p2p",
                        phase=mode, recipe="pipeline_residual_send",
                        link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                        shape=(tokens, model.hidden_size),
                        metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                    ops.append(transfer)
                    moved.append(transfer.op_id)
                residual_sources = moved
        if attn_state is not None:
            last = model.layer_plans[-1]
            final = attn_state.final(prefix=prefix + "final", last_layer=last,
                                     stage=stages[last.index])
            ops.extend(_scope(final.ops, "ffn", "residual"))
            previous = final.hidden_id
        if sp and tokens:
            gathered = sequence_parallel.allgather(
                plan=sp, op_id=prefix + "sp_output_gather", deps=(previous,),
                stage=stages[-1] if stages else 0, layer=-1, requests=requests, phase=mode,
                width=model.hidden_size, dtype=model.activation_dtype)
            ops.append(gathered)
            previous = gathered.op_id
            if sp.physical_global > sp.input_rows:
                trimmed = sequence_parallel.output_trim(
                    plan=sp, op_id=prefix + "sp_output_trim", deps=(previous,),
                    stage=stages[-1] if stages else 0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(trimmed)
                previous = trimmed.op_id
        from .sparse_mla_metadata import before_forward
        ops = before_forward(ops, ready)
        if not include_recover_commits:
            ops = _strip_recover_intents(ops)
        return ops, previous

    @staticmethod
    def final_mixer(model, ready, *, prefix, mode, stage, tokens, requests):
        """Emit the Qwen4Exp stream collapse, or the ordinary final norm."""
        if any(getattr(plan, "hc_lowrank", ZERO) > ZERO for plan in model.layer_plans):
            return lowrank_hyper_final_mixer(
                prefix=prefix + "final", deps=ready, stage=stage,
                requests=requests, tokens=tokens, model=model, mode=mode)
        final = norm(name=prefix + "final_norm", deps=(ready,), stage=stage,
                     layer=FINAL_LAYER, requests=requests, tokens=tokens,
                     width=model.hidden_size, model=model, mode=mode)
        return [final], final.op_id

    def logits(self, model, runtime, views, ready, *, prefix="", mode="decode", stage=None, sample=True,
               input_tokens=None, normalized=False):
        p = runtime.parallel
        stage = p.pp - 1 if stage is None else stage
        sampled = tuple(view for view in views if view.samples)
        rows = sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        input_tokens = sum(view.q_len for view in views) if input_tokens is None else input_tokens
        requests = tuple(view.request_id for view in sampled)
        ops, last = [], ready
        if not normalized and runtime.final_norm_scope == "all" and input_tokens:
            finals, last = self.final_mixer(model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=input_tokens, requests=tuple(view.request_id for view in views))
            ops.extend(finals)
        if not rows:
            return ops, last
        if rows != input_tokens:
            select = self.select_hidden(model, views, last, prefix=prefix, mode=mode, stage=stage)
            ops.append(select)
            last = select.op_id
        if not normalized and runtime.final_norm_scope == "selected":
            finals, last = self.final_mixer(model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=rows, requests=requests)
            ops.extend(finals)
        logits = gemm_op(
            op_id=prefix + "logits", deps=(last,), stage=stage, layer=-1, requests=requests,
            m=rows, n=math.ceil(model.vocab_size / p.tp), k=model.hidden_size, quant=model.quant,
            category="lm_head", phase=mode, family="logits", recipe="vocab_logits",
        )
        ops.append(logits)
        last = logits.op_id
        if p.tp > 1:
            gather = comm_op(
                op_id=prefix + "logits_gather", deps=(last,), stage=stage, layer=-1, requests=requests,
                family="allgather", phase=mode, recipe="ring_allgather",
                link_bytes=rows * model.vocab_size * 2 * (p.tp - 1) / p.tp,
                shape=(rows, model.vocab_size),
                metadata=(("group_axis", "tp"), ("steps", p.tp - 1)),
            )
            ops.append(gather)
            last = gather.op_id
        if not sample:
            return ops, last
        sample = vector_op(
            op_id=prefix + "sample", deps=(last,), stage=stage, layer=-1, requests=requests,
            elements=rows * model.vocab_size, ops_per_element=5, dtype="fp32",
            family="sampling", phase=mode, recipe="sample_logits", shape=(rows, model.vocab_size),
        )
        ops.append(sample)
        return ops, sample.op_id

    @staticmethod
    def select_hidden(model, views, ready, *, prefix="", mode="decode", stage=0):
        sampled = tuple(view for view in views if view.samples)
        rows = sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        payload = rows * model.hidden_size * dtype_bytes(model.activation_dtype)
        return vector_op(
            op_id=prefix + "select_hidden", deps=(ready,), stage=stage, layer=-1,
            requests=tuple(view.request_id for view in sampled), elements=rows * model.hidden_size,
            ops_per_element=0, dtype=model.activation_dtype, family="tensor_select", phase=mode,
            recipe="select_hidden", shape=(rows, model.hidden_size),
            read_bytes_override=payload + rows * 8, write_bytes_override=payload,
            metadata=(("selection", "all_verify_rows_or_last_prefill_row"),))

    @staticmethod
    def partition_views(views, runtime, padded, *, model=None):
        degree = runtime.parallel.pcp
        if degree == ONE or not any(v.phase == "prefill" for v in views):
            return views, padded, False
        rep_win = ZERO
        plans = getattr(model, "layer_plans", ()) if model is not None else ()
        attn_plans = [
            lp for lp in plans
            if lp.attention in {"gqa", "mla"} and lp.cache_kind != "recurrent_unmodeled"
        ]
        if attn_plans and all(
            int(lp.window) > ZERO and not lp.compress_ratio and not lp.index_topk
            for lp in attn_plans
        ):
            rep_win = max(int(lp.window) for lp in attn_plans)
        costs = [ZERO_FLOAT] * degree
        for view in views:
            if view.phase == "prefill":
                split = split_pcp_pairs(view.q_len, view.context, degree, window=rep_win)
                costs = [old + current for old, current in zip(costs, split.per_rank)]
        rank = max(range(degree), key=costs.__getitem__)
        ranges = [pcp_rank_chunk_ranges(view.q_len, degree, rank) for view in views]
        local = tuple(replace(view, q_len=sum(end - start for start, end in parts),
                              global_q_len=view.q_len, query_ranges=parts)
                      if view.phase == "prefill" else view for view, parts in zip(views, ranges))
        return local, sum(view.q_len for view in local), True

    @staticmethod
    def pcp_cohort_tokens(views, local_views, local_tokens, runtime, global_tokens=None, global_views=None):
        p = runtime.parallel
        if global_tokens is None and not getattr(p, "ep_spans_pcp", False):
            return None
        pcp_in_ep = min(p.pcp, max(1, p.ep // p.tp)) if getattr(p, "ep_spans_pcp", False) else 1
        gviews = global_views if global_views is not None else views
        expand = global_views is not None and (
            global_tokens is None or sum(v.q_len for v in gviews) < global_tokens)
        q_of = lambda v: v.q_len + (v.draft_proposal_tokens if expand else 0)
        dec_g = sum(q_of(v) for v in gviews if v.phase != "prefill")
        pf_views = tuple(v for v in gviews if v.phase == "prefill")
        pf_unpadded = sum(q_of(v) for v in pf_views)
        rank_totals = [
            sum(sum(e - s for s, e in pcp_rank_chunk_ranges(q_of(v), p.pcp, r))
                for v in pf_views)
            for r in range(p.pcp)
        ]
        pf_local_sum = max(
            (sum(rank_totals[g:g + pcp_in_ep]) for g in range(0, p.pcp, pcp_in_ep)),
            default=0,
        )
        pad_extra = max(0, (global_tokens - dec_g - pf_unpadded) if global_tokens is not None else 0)
        return max(local_tokens, dec_g + pf_local_sum + (pad_extra * pcp_in_ep + p.pcp - 1) // p.pcp)

    def program(self, plan, model, runtime, machine, *, assembly="service", **options):
        """Return executable model work with typed ports and occurrence triggers.

        ``assembly="model"`` selects only the forward plan and its necessary
        implementation support. ``assembly="service"`` keeps sampling,
        acceptance and token commit as declared service fragments.
        """
        work = self(plan, model, runtime, machine, assembly=assembly, **options)
        counts = Counter(op.family for op in work.ops)
        triggers = tuple((family, counts[family], _TRIGGERS[family])
                         for family in sorted(SERVICE_FAMILIES) if counts[family])
        return ModelProgram(
            work=work,
            ports=_ports_for_work(work),
            triggers=triggers,
            assembly=assembly,
            layer_scope=("per_model_layer" if any(op.layer >= 0 for op in work.ops)
                         else "none"),
        )

    def __call__(self, plan, model, runtime, machine, *, global_tokens=None, global_views=None,
                 resource_fusion=None, replica=ZERO, source_rows_by_replica=None,
                 routing_global_tokens=None, routing_views=None,
                 cohort_views_by_replica=None,
                 physical_rank=None, query_position=None, assembly="service",
                 dispatch_binding=None, dispatch_fast_step: bool = False):
        # ``dispatch_binding`` is the T14 PP/EP microbatch envelope (original
        # slot/version/replica). The ordinary producer keeps using the views in
        # ``plan``; a cached template owner can read this envelope without
        # inferring a slot from local list positions.
        if assembly not in {"model", "service"}:
            raise ModelingError(f"assembly_scope_unknown:{assembly}")
        if resource_fusion is None:
            resource_fusion = self.resource_fusion
        s_env = self._prev_call_env
        if (s_env is not None and s_env[_ENV_MODEL] is model
                and (s_env[_ENV_RUNTIME] is runtime
                     or s_env[_ENV_RUNTIME] == runtime)
                and s_env[_ENV_MACHINE] is machine
                and s_env[_ENV_REPLICA] == replica
                and s_env[_ENV_FEATURES] is self.features):
            _, _, _, _, _, model, auto_rank, features = s_env
            if physical_rank is None:
                physical_rank = auto_rank
        else:
            raw_model = model
            model = normalize_model(model)
            auto_rank = None
            try:
                from .topology import Topology
                auto_rank = Topology(machine, runtime.parallel).ranks(
                    replica, 0)[0]
            except Exception:
                auto_rank = None
            features = (self.features if self.features is not None else
                        FeatureConfig.from_runtime(runtime, model, machine=machine))
            self._prev_call_env = (
                raw_model, runtime, machine, replica, self.features,
                model, auto_rank, features,
            )
            if physical_rank is None:
                physical_rank = auto_rank
        # An empty local query set can still receive expert rows. Its execution
        # phase belongs to the active cohort, not a fictitious "idle" kernel.
        mode = selected_mode(plan if plan.requests else replace(
            plan, requests=tuple(global_views or ())))
        if assembly == "model":
            from math import floor
            mean = float(runtime.mtp_acceptance_length)
            plan = replace(plan, requests=tuple(
                replace(view,
                        draft_advanced_tokens=min(
                            view.q_len + view.draft_proposal_tokens,
                            max(ONE, floor((index + ONE) * mean) - floor(index * mean))))
                if view.draft_proposal_tokens and not view.draft_advanced_tokens
                else view for index, view in enumerate(plan.requests)))
        views = tuple(replace(view, q_len=view.q_len + view.draft_proposal_tokens)
                      for view in plan.requests)
        valid = sum(view.logical_q_len for view in views)
        physical = sum(view.q_len for view in views)
        bucket, scope = self.graph.select(views, runtime) if views else (None, "none")
        padded = max(physical, bucket or 0)
        declared_runtime = runtime
        runtime, selections = resolve_implementations(
            runtime, views, padded_tokens=padded, replay_scope=scope,
            global_views=global_views)
        runtime = mark_analytical_runtime(runtime, selections)
        # A speculative decode leaves state publication to the service
        # acceptance boundary. The homogeneous fast path splices one layer
        # template across copies, which would lose those deferred intents.
        deferred_commits = bool(recover_ssm_enabled(runtime)) and any(
            view.phase == "decode" and view.q_len > ONE
            and not getattr(view, "padding", False) for view in views)
        requests = tuple(view.request_id for view in views)
        ops = []
        entry = barrier_op(
            op_id="iteration_input", deps=(), stage=0, layer=-1, requests=requests, phase=mode,
            recipe="input_ready", shape=(valid, model.hidden_size),
        )
        previous = entry.op_id
        before, after = [], []
        if any(view.draft_proposal_tokens for view in (global_views or plan.requests)):
            from .speculation import draft_program
            before, previous = draft_program(
                self, model, declared_runtime, plan.requests, previous,
                global_views=global_views, replica=replica, replay_scope=scope,
                machine=machine)
        media_counts = {str(request): int(count) for request, count in
                        _media_counts(runtime).items()}
        for index, view in enumerate(views):
            if view.media_encoded or view.phase != "prefill" or not media_counts.get(view.request_id):
                continue
            media = lower_media(
                model=model, runtime=runtime, parallel=runtime.parallel, quant=model.quant,
                mode=mode, views=(view,), media_tokens_override=media_counts[view.request_id],
            )
            rename = {op.op_id: f"media{index}:{op.op_id}" for op in media.ops}
            ops.extend(replace(op, op_id=rename[op.op_id],
                               deps=tuple(rename[dep] for dep in op.deps) or (previous,))
                       for op in media.ops)
            previous = rename[media.last]
        local_views, local_tokens, pcp_prefill = self.partition_views(
            views, runtime, padded, model=model)
        if valid:
            embedding = vector_op(
                op_id="embedding", deps=(previous,), stage=0, layer=-1, requests=requests,
                elements=local_tokens * model.hidden_size, ops_per_element=1,
                dtype=model.activation_dtype, family="embedding", phase=mode, recipe="embedding",
            )
            ops.append(embedding)
            previous = embedding.op_id
        pcp_cohort = runtime.parallel.pcp > 1 and any(
            v.phase == "prefill" for v in (global_views or views))
        body_global_tokens = (
            self.pcp_cohort_tokens(views, local_views, local_tokens, runtime,
                                   global_tokens=global_tokens, global_views=global_views)
            if pcp_cohort else global_tokens
        )
        eff_routing_views = routing_views if routing_views is not None else views
        eff_query_pos = (
            query_position if query_position is not None
            else next((v.context for v in (routing_views or views) if not v.padding), None)
        )
        preparation_views = tuple(view for view in plan.requests if view.phase == "prefill"
                                  and view.draft_prepare)
        global_preparation = tuple(view for view in (global_views or plan.requests)
                                   if view.phase == "prefill" and view.draft_prepare)
        has_after_draft = bool(
            dict(runtime.metadata).get("draft_mode")
            and (preparation_views or global_views and global_preparation)
        )
        fast_result = None if deferred_commits else self._try_lower_homogeneous(
            entry=entry, pre_ops=tuple(ops), body_ready=previous,
            plan=plan, model=model, runtime=runtime,
            views=views, local_views=local_views,
            valid=valid, padded=padded, local_tokens=local_tokens,
            bucket=bucket, scope=scope, mode=mode, requests=requests,
            selections=selections, resource_fusion=resource_fusion,
            assembly=assembly, replica=replica, features=features,
            body_global_tokens=body_global_tokens,
            source_rows_by_replica=source_rows_by_replica,
            routing_global_tokens=routing_global_tokens,
            routing_views=eff_routing_views,
            physical_rank=physical_rank,
            query_position=eff_query_pos,
            before=before, has_after_draft=has_after_draft,
            pcp_prefill=pcp_prefill,
            dispatch_fast_step=dispatch_fast_step,
            machine=machine,
        )
        if fast_result is not None:
            return fast_result
        body, previous = self.body(model, runtime, local_views, ready=previous, tokens=local_tokens,
                                   mode=mode, global_tokens=body_global_tokens, replica=replica,
                                   features=features,
                                   source_rows_by_replica=source_rows_by_replica,
                                   routing_global_tokens=routing_global_tokens,
                                   routing_views=eff_routing_views,
                                   cohort_views_by_replica=cohort_views_by_replica,
                                   physical_rank=physical_rank,
                                   query_position=eff_query_pos,
                                   include_recover_commits=(assembly == "service" or any(
                                       view.draft_proposal_tokens for view in plan.requests)),
                                   machine=machine)
        ops.extend(body)
        if pcp_prefill:
            if runtime.final_norm_scope == "all":
                finals, previous = self.final_mixer(model, previous, prefix="", mode=mode,
                    stage=runtime.parallel.pp + FINAL_LAYER, tokens=local_tokens, requests=requests)
                ops.extend(finals)
            select = self.select_hidden(model, views, previous, mode=mode, stage=runtime.parallel.pp - 1)
            ops.append(select)
            previous = select.op_id
            rows = sum(v.q_len if v.phase == "decode" else 1 for v in views if v.samples)
            gather = comm_op(
                op_id="pcp_hidden_gather", deps=(previous,), stage=runtime.parallel.pp - 1,
                layer=-1, requests=requests, family="allgather", phase=mode,
                recipe="pcp_hidden_gather", link_bytes=rows * model.hidden_size * dtype_bytes(model.activation_dtype)
                * (runtime.parallel.pcp - 1) / runtime.parallel.pcp,
                shape=(rows, model.hidden_size), metadata=(("group_axis", "pcp"),))
            ops.append(gather)
            previous = gather.op_id
        heads, previous = self.logits(
            model, runtime, views, previous, mode=mode,
            input_tokens=rows if pcp_prefill else padded,
            normalized=pcp_prefill and runtime.final_norm_scope == "all")
        ops.extend(heads)
        proposal_count = sum(view.draft_proposal_tokens for view in plan.requests)
        if proposal_count:
            accept = vector_op(
                op_id="accept", deps=(previous,), stage=runtime.parallel.pp - 1, layer=-1,
                requests=requests, elements=proposal_count, ops_per_element=6,
                dtype="fp32", family="acceptance", phase=mode, recipe="speculative_acceptance",
            )
            ops.append(accept)
            previous = accept.op_id
        committed = barrier_op(
            op_id="token_commit", deps=(previous,), stage=runtime.parallel.pp - 1, layer=-1,
            requests=tuple(view.request_id for view in views if view.samples),
            phase=mode, recipe="token_commit", shape=(len(requests),),
        )
        ops.append(replace(committed, family="token_commit"))
        if deferred_commits:
            intents = tuple(op for op in ops if is_recover_ssm_intent(op))
            if intents and assembly == "model" and not proposal_count:
                ops = list(_strip_recover_intents(ops))
            elif intents:
                spec_views = tuple(
                    view for view in views
                    if view.phase == "decode" and view.q_len > ONE
                    and not bool(getattr(view, "padding", False)))
                anchor = next(
                    (op for op in reversed(ops)
                     if op.family in {"acceptance", "sampling"}), None)
                if anchor is None:
                    raise ModelingError(
                        "recover_ssm_commit_requires_acceptance_boundary")
                kept = [op for op in ops if not is_recover_ssm_intent(op)]
                bound = bind_recover_ssm_commits(
                    intents, views=spec_views,
                    acceptance_events={view.request_id: anchor.op_id
                                       for view in spec_views},
                    state_bindings=None, runtime=runtime)
                position = next(
                    index for index, op in enumerate(kept)
                    if op.op_id == anchor.op_id) + ONE
                kept[position:position] = bound
                commit_ids = tuple(op.op_id for op in bound)
                ops = [
                    replace(op, deps=tuple(dict.fromkeys(
                        (*op.deps, *commit_ids))))
                    if op.family == "token_commit" else op
                    for op in kept
                ]
        if has_after_draft:
            from .speculation import draft_program
            after, _ = draft_program(
                self, model, declared_runtime, preparation_views, committed.op_id,
                global_views=global_preparation if global_views else None, prefill=True,
                replica=replica, replay_scope=scope)
        # A capture descriptor is distinct from each kernel's active/sliced
        # tensor. Preserve both so native responses cannot alias two buckets
        # merely because their logical attention shapes are equal.
        ops = annotate_geometry(ops, active_rows=valid, physical_rows=padded)
        regions = (*before, LoweringRegion(tuple(ops), runtime, model.activation_dtype,
                                          local_tokens, selections), *after)
        if assembly == "model":
            # Scope selection happens before lowering so service-only nodes do
            # not enter native fusion, capture or implementation decisions.
            regions = _without_service_fragments(regions)
        ops, selections = lower_regions(
            regions, graph=self.graph, replay_scope=scope, runtime=runtime,
            prefix=(entry,), resource_fusion=resource_fusion)
        ops = self._reuse_identical_ops(ops)
        return IterationWork(
            ops=ops,
            plan=plan, stage_count=runtime.parallel.pp,
            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
            recipe="structural", replay_scope=scope,
            implementation_selections=selections,
        )

    def _reuse_identical_ops(self, ops: tuple) -> tuple:
        prev = self._prev_work_ops
        if prev is not None and len(prev) == len(ops):
            reused = tuple(
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev, ops)
            )
            self._prev_work_ops = reused
            return reused
        self._prev_work_ops = ops
        return ops

    def _try_lower_homogeneous(
        self, *, entry, pre_ops, body_ready, plan, model, runtime,
        views, local_views, valid, padded, local_tokens, bucket, scope,
        mode, requests, selections, resource_fusion, assembly, replica,
        features, body_global_tokens, source_rows_by_replica,
        routing_global_tokens, routing_views, physical_rank,
        query_position, before, has_after_draft, pcp_prefill,
        dispatch_fast_step: bool = False, machine=None,
    ):
        layers = model.layer_plans
        if (len(layers) <= 2 or local_tokens <= 0 or scope != "none"
                or before or has_after_draft or pcp_prefill
                or runtime.enable_dbo or runtime.parallel.pp != 1):
            return None
        fusions_cfg = runtime.kernels.get("fusions") or ()
        if "residual_norm" in fusions_cfg:
            return None
        rmeta = dict(runtime.metadata)
        if "sparse_mla_metadata_groups" in rmeta:
            return None
        p = runtime.parallel
        routing_tokens = (
            routing_global_tokens if routing_global_tokens is not None
            else body_global_tokens
        )
        valid_local = sum(v.logical_q_len for v in local_views)
        t_shards = 1
        has_explicit_real = "realizations" in runtime.routing
        env_views = (
            routing_views if has_explicit_real
            else tuple((v.q_len, v.logical_q_len, bool(getattr(v, "padding", False)))
                       for v in local_views)
        )
        static_env_key = (
            mode, local_tokens, valid_local, body_global_tokens,
            routing_tokens, replica, physical_rank,
            query_position if has_explicit_real else None,
            env_views,
        )
        prev_env = self._prev_homo_env
        if (prev_env is not None and prev_env[0] is model
                and (prev_env[1] is runtime or prev_env[1] == runtime)
                and prev_env[2] == static_env_key):
            if not prev_env[3]:
                return None
            arch_for_layer, unique_archs, realizations = prev_env[4]
        else:
            for lyr in layers:
                if (lyr.engram or lyr.hyper_connections or lyr.residual_block_size
                        or lyr.kv_owner_layer_id not in (-1, lyr.index)
                        or lyr.index_k_owner_layer_id not in (-1, lyr.index)
                        or lyr.topk_owner_layer_id not in (-1, lyr.index)
                        or lyr.candidate_source_layer_id not in (-1, lyr.index)
                        or lyr.index_source_layer_id not in (-1, lyr.index)
                        or lyr.native_layer_id not in (-1, lyr.index)):
                    self._prev_homo_env = (model, runtime, static_env_key, False, None)
                    return None
            if sequence_parallel.select(
                    runtime, model, local_views, tokens=local_tokens,
                    global_tokens=body_global_tokens) is not None:
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            if (mode not in {"draft", "draft_prefill"}
                    and family_attnres.selection(runtime, model=model, phase=mode) is not None):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            stages = stage_map(model, runtime)
            if any(s != 0 for s in stages):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None

            arch_for_layer_list = []
            unique_archs = {}
            realizations_list = []
            for idx, lyr in enumerate(layers):
                real = (
                    select_realization(
                        runtime, lyr, phase=mode, tokens=local_tokens,
                        valid_tokens=valid_local, global_tokens=routing_tokens,
                        token_shards=t_shards, valid_source_tokens=None,
                        replica=replica, views=routing_views,
                        physical_rank=physical_rank, query_position=query_position,
                    )
                    if lyr.ffn == "moe" else None
                )
                realizations_list.append(real)
                norm_real = (
                    replace(real, layer=0)
                    if isinstance(real, RoutingRealization) else real
                )
                norm_lyr = replace(
                    lyr,
                    index=0,
                    kv_owner_layer_id=(-1 if lyr.kv_owner_layer_id == -1 else 0),
                    index_k_owner_layer_id=(-1 if lyr.index_k_owner_layer_id == -1 else 0),
                    topk_owner_layer_id=(-1 if lyr.topk_owner_layer_id == -1 else 0),
                    candidate_source_layer_id=(-1 if lyr.candidate_source_layer_id == -1 else 0),
                    index_source_layer_id=(-1 if lyr.index_source_layer_id == -1 else 0),
                    native_layer_id=(-1 if lyr.native_layer_id == -1 else 0),
                )
                akey = (norm_lyr, norm_real)
                arch_for_layer_list.append(akey)
                if akey not in unique_archs:
                    unique_archs[akey] = idx

            if len(unique_archs) >= len(layers):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            arch_for_layer = tuple(arch_for_layer_list)
            realizations = tuple(realizations_list)
            self._prev_homo_env = (
                model, runtime, static_env_key, True,
                (arch_for_layer, unique_archs, realizations),
            )

        from .execution import meta_scan_flags, op_meta_key_from_metadata, seed_meta_id_key
        decode_sig = (
            (
                tuple((v.request_id, v.q_len, v.logical_q_len, v.samples,
                       v.draft_proposal_tokens, bool(getattr(v, "padding", False)))
                      for v in views),
                tuple((v.request_id, v.q_len, v.logical_q_len,
                       bool(getattr(v, "padding", False)))
                      for v in local_views),
                resource_fusion,
                assembly,
                selections,
            )
            if (mode == "decode" and local_views
                and all(v.phase == "decode" and v.context > 0 for v in local_views))
            else None
        )
        prev_state = self._prev_arch_state
        can_reuse_prev = (
            prev_state is not None
            and prev_state[0] == arch_for_layer
            and prev_state[1] == body_ready
        )
        reused_archs = {}
        if (dispatch_fast_step and can_reuse_prev and decode_sig is not None
                and prev_state[7] == decode_sig):
            prev_tmpls_fast = prev_state[2]
            for akey in unique_archs:
                if akey[0].attention in {"gdn", "kda", "projection_only", "unmodeled"}:
                    p_entry = prev_tmpls_fast.get(akey)
                    if p_entry is not None and not any(
                            meta_scan_flags(op.metadata)[2] for op in p_entry[4]):
                        reused_archs[akey] = p_entry
            if len(reused_archs) == len(unique_archs) and self._prev_work_ops is not None:
                prev_sel_c = prev_state[6]
                return IterationWork(
                    ops=self._prev_work_ops,
                    plan=plan, stage_count=runtime.parallel.pp,
                    valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                    recipe="structural", replay_scope=scope,
                    implementation_selections=(prev_sel_c[2] if prev_sel_c is not None else ()),
                )
            if not selections and prev_state[8]:
                prev_raw_pre, layer_offsets, prev_attn_splits, prev_final_ops = prev_state[8]
                active_fast = [
                    rep_idx for akey, rep_idx in unique_archs.items()
                    if akey not in reused_archs
                ]
                if (self._prev_work_ops is prev_final_ops
                        and pre_ops == prev_raw_pre
                        and all(arch_for_layer[r] in prev_attn_splits for r in active_fast)):
                    local_requests = tuple(v.request_id for v in local_views)
                    residual = self.residuals["plain"]
                    caches, indices_map, topks, candidates = {}, {}, {}, {}
                    splice_ok = True
                    new_arch_templates = dict(prev_tmpls_fast)
                    new_attn_splits = dict(prev_attn_splits)
                    changed_by_arch = {}
                    for rep_idx in active_fast:
                        lyr = layers[rep_idx]
                        akey = arch_for_layer[rep_idx]
                        in_op, expected_done_id, prev_tmpl_attn = prev_attn_splits[akey]
                        in_id = in_op.op_id
                        name = f"s0.l{rep_idx}"
                        common = dict(
                            prefix=name, plan=lyr, stage=0, requests=local_requests,
                            tokens=local_tokens, model=model, mode=mode,
                            residual_sources=(), features=features,
                            source_rows_by_replica=source_rows_by_replica,
                        )
                        pre_r, hidden, saved = residual.before(
                            {**common, "prefix": name + ".attention", "structure_block": "attention"},
                            in_id,
                        )
                        norm1 = norm(
                            name=name + ".norm1", deps=(hidden,), stage=0, layer=rep_idx,
                            requests=local_requests, tokens=local_tokens,
                            width=model.hidden_size, model=model, mode=mode,
                        )
                        attn = self.attention[lyr.attention](
                            prefix=name, plan=lyr, views=local_views, stage=ZERO,
                            deps=norm1.op_id, model=model, parallel=p, quant=model.quant,
                            mode=mode, padded_tokens=local_tokens,
                            cache_ids=caches, topk_ids=topks, index_cache_ids=indices_map,
                            candidate_ids=candidates, stage_of=lambda _l: ZERO,
                            cp_interleave_size=runtime.kv_block_size, runtime=runtime,
                            policy=self.attention_policy, machine=machine,
                        )
                        post_a, done_id = residual.after(
                            {**common, "prefix": name + ".attention", "structure_block": "attention"},
                            attn.last, saved,
                        )
                        attn_raw = [
                            *_scope(pre_r, "attention", "residual"),
                            *_scope((norm1,), "attention"),
                            *_scope(attn.ops, "attention"),
                            *_scope(post_a, "attention", "residual"),
                        ]
                        if (done_id != expected_done_id
                                or any(dict(op.metadata).get("metadata_schedule") == "pre_forward"
                                       for op in attn_raw)):
                            splice_ok = False
                            break
                        attn_ann = annotate_geometry(attn_raw, active_rows=valid, physical_rows=padded)
                        attn_regs = (LoweringRegion(tuple(attn_ann), runtime, model.activation_dtype,
                                                    local_tokens, ()),)
                        if assembly == "model":
                            attn_regs = _without_service_fragments(attn_regs)
                        solo_attn_full, _ = lower_regions(
                            attn_regs, graph=self.graph, replay_scope=scope, runtime=runtime,
                            prefix=(in_op,), resource_fusion=resource_fusion,
                        )
                        solo_attn = (
                            solo_attn_full[1:]
                            if (solo_attn_full and solo_attn_full[0].op_id == in_id)
                            else solo_attn_full
                        )
                        if (len(solo_attn) != len(prev_tmpl_attn)
                                or not solo_attn
                                or solo_attn[-1].op_id != expected_done_id
                                or any(op.layer != rep_idx for op in solo_attn)):
                            splice_ok = False
                            break
                        changed_pos = tuple(
                            pos for pos in range(len(solo_attn))
                            if solo_attn[pos] != prev_tmpl_attn[pos]
                        )
                        (p_rep_idx, p_pfx, p_src_prev,
                         p_out_sfx, p_tmpl_ops, p_meta_plans) = prev_tmpls_fast[akey]
                        if changed_pos:
                            merged_attn = tuple(
                                prev_tmpl_attn[pos] if solo_attn[pos] == prev_tmpl_attn[pos] else solo_attn[pos]
                                for pos in range(len(solo_attn))
                            )
                            new_tmpl_ops = merged_attn + p_tmpl_ops[len(solo_attn):]
                            new_arch_templates[akey] = (
                                p_rep_idx, p_pfx, p_src_prev, p_out_sfx, new_tmpl_ops, p_meta_plans,
                            )
                            new_attn_splits[akey] = (in_op, expected_done_id, merged_attn)
                            changed_by_arch[akey] = (
                                p_rep_idx, p_pfx, p_src_prev, p_out_sfx,
                                merged_attn, changed_pos, p_meta_plans,
                            )
                    if splice_ok:
                        if not changed_by_arch:
                            return IterationWork(
                                ops=prev_final_ops,
                                plan=plan, stage_count=runtime.parallel.pp,
                                valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                                recipe="structural", replay_scope=scope,
                                implementation_selections=(),
                            )
                        out_ops = list(prev_final_ops)
                        new_layers_ops = list(prev_state[3])
                        cur_prev = body_ready
                        step_meta_keys_fast: dict[tuple[Any, int], Any] = {}
                        for idx in range(len(layers)):
                            akey = arch_for_layer[idx]
                            ch = changed_by_arch.get(akey)
                            if ch is None:
                                cur_prev = f"s0.l{idx}." + new_arch_templates[akey][3]
                                continue
                            (rep_idx, src_pfx, src_prev, out_sfx,
                             solo_attn, changed_pos, meta_plans) = ch
                            dst_pfx = f"s0.l{idx}."
                            base_off = layer_offsets[idx]
                            lyr_list = list(new_layers_ops[idx])
                            for pos in changed_pos:
                                tmpl_op = solo_attn[pos]
                                if idx == rep_idx and cur_prev == src_prev:
                                    retargeted = tmpl_op
                                else:
                                    new_deps = tuple(
                                        cur_prev if d == src_prev
                                        else (d.replace(src_pfx, dst_pfx) if src_pfx in d else d)
                                        for d in tmpl_op.deps
                                    )
                                    mplan = meta_plans[pos]
                                    if mplan is False:
                                        mplan = _compile_meta_plan(
                                            tmpl_op.metadata, src_pfx, rep_idx, src_prev)
                                        meta_plans[pos] = mplan
                                    if mplan is None:
                                        new_meta = tmpl_op.metadata
                                    else:
                                        new_meta = _apply_meta_plan(
                                            tmpl_op.metadata, mplan, src_pfx, dst_pfx,
                                            idx, cur_prev,
                                        )
                                        if idx > 0:
                                            mk_slot = (akey, pos)
                                            mkey = step_meta_keys_fast.get(mk_slot)
                                            if mkey is None:
                                                if rep_idx > 0 or not _plan_uses_prev_id(mplan):
                                                    mkey = op_meta_key_from_metadata(tmpl_op.metadata)
                                                else:
                                                    mkey = op_meta_key_from_metadata(new_meta)
                                                step_meta_keys_fast[mk_slot] = mkey
                                            seed_meta_id_key(new_meta, mkey)
                                    retargeted = replace(
                                        tmpl_op,
                                        op_id=tmpl_op.op_id.replace(src_pfx, dst_pfx),
                                        deps=new_deps,
                                        layer=idx,
                                        metadata=new_meta,
                                    )
                                out_ops[base_off + pos] = retargeted
                                lyr_list[pos] = retargeted
                            new_layers_ops[idx] = tuple(lyr_list)
                            cur_prev = dst_pfx + out_sfx
                        final_ops = tuple(out_ops)
                        self._prev_work_ops = final_ops
                        self._prev_arch_state = (
                            arch_for_layer, body_ready, new_arch_templates,
                            tuple(new_layers_ops), prev_state[4], prev_state[5],
                            None, decode_sig,
                            (prev_raw_pre, layer_offsets, new_attn_splits, final_ops),
                        )
                        return IterationWork(
                            ops=final_ops,
                            plan=plan, stage_count=runtime.parallel.pp,
                            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                            recipe="structural", replay_scope=scope,
                            implementation_selections=(),
                        )

        # Build reduced graph containing pre_ops + active representative layers + post_ops
        red_ops = list(pre_ops)
        cur_prev = body_ready
        arch_input_id = {}
        arch_output_id = {}
        raw_attn_by_arch = {}
        active_rep_indices = [
            rep_idx for akey, rep_idx in unique_archs.items()
            if akey not in reused_archs
        ]
        local_requests = tuple(v.request_id for v in local_views)
        residual = self.residuals["plain"]
        caches, indices_map, topks, candidates = {}, {}, {}, {}

        for rep_idx in active_rep_indices:
            lyr = layers[rep_idx]
            akey = arch_for_layer[rep_idx]
            in_id = cur_prev
            arch_input_id[akey] = in_id
            name = f"s0.l{rep_idx}"
            common = dict(
                prefix=name, plan=lyr, stage=0, requests=local_requests,
                tokens=local_tokens, model=model, mode=mode,
                residual_sources=(), features=features,
                source_rows_by_replica=source_rows_by_replica,
            )
            pre_r, hidden, saved = residual.before(
                {**common, "prefix": name + ".attention", "structure_block": "attention"},
                cur_prev,
            )
            norm1 = norm(
                name=name + ".norm1", deps=(hidden,), stage=0, layer=rep_idx,
                requests=local_requests, tokens=local_tokens,
                width=model.hidden_size, model=model, mode=mode,
            )
            attn = self.attention[lyr.attention](
                prefix=name, plan=lyr, views=local_views, stage=0,
                deps=norm1.op_id, model=model, parallel=p, quant=model.quant,
                mode=mode, padded_tokens=local_tokens,
                cache_ids=caches, topk_ids=topks, index_cache_ids=indices_map,
                candidate_ids=candidates, stage_of=lambda _l: 0,
                cp_interleave_size=runtime.kv_block_size, runtime=runtime,
                policy=self.attention_policy, machine=machine,
            )
            post_a, cur_prev = residual.after(
                {**common, "prefix": name + ".attention", "structure_block": "attention"},
                attn.last, saved,
            )
            attn_sub_raw = [
                *_scope(pre_r, "attention", "residual"),
                *_scope((norm1,), "attention"),
                *_scope(attn.ops, "attention"),
                *_scope(post_a, "attention", "residual"),
            ]
            if lyr.attention not in {"gdn", "kda", "projection_only", "unmodeled"}:
                raw_attn_by_arch[akey] = (in_id, attn_sub_raw, cur_prev)
            red_ops.extend(attn_sub_raw)
            pre_f, hidden, saved = residual.before(
                {**common, "prefix": name + ".ffn", "structure_block": "ffn"},
                cur_prev,
            )
            red_ops.extend(_scope(pre_f, "ffn", "residual"))
            norm2 = norm(
                name=name + ".norm2", deps=(hidden,), stage=0, layer=rep_idx,
                requests=local_requests, tokens=local_tokens,
                width=model.hidden_size, model=model, mode=mode,
            )
            red_ops.extend(_scope((norm2,), "ffn"))
            ffn_ops, cur_prev = self.ffn[lyr.ffn](
                **common, deps=norm2.op_id, parallel=p,
                global_tokens=body_global_tokens, routing=self.routing,
                routing_realization=realizations[rep_idx],
                moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
                overlap_shared=bool(runtime.multistream_overlap_shared_expert),
                runtime=runtime, sequence_plan=None,
                ep_owner=(int(physical_rank) % int(p.ep)
                          if physical_rank is not None and int(p.ep) > ZERO
                          else None),
            )
            red_ops.extend(_scope(ffn_ops, "ffn"))
            post_f, cur_prev = residual.after(
                {**common, "prefix": name + ".ffn", "structure_block": "ffn"},
                cur_prev, saved,
            )
            red_ops.extend(_scope(post_f, "ffn", "residual"))
            arch_output_id[akey] = cur_prev

        if any(dict(op.metadata).get("metadata_schedule") == "pre_forward" for op in red_ops):
            return None

        red_body_last = cur_prev
        heads, post_prev = self.logits(
            model, runtime, views, red_body_last, mode=mode,
            input_tokens=padded, normalized=False,
        )
        red_ops.extend(heads)
        proposal_count = sum(view.draft_proposal_tokens for view in plan.requests)
        if proposal_count:
            accept = vector_op(
                op_id="accept", deps=(post_prev,), stage=0, layer=-1,
                requests=requests, elements=proposal_count, ops_per_element=6,
                dtype="fp32", family="acceptance", phase=mode,
                recipe="speculative_acceptance",
            )
            red_ops.append(accept)
            post_prev = accept.op_id
        committed = barrier_op(
            op_id="token_commit", deps=(post_prev,), stage=0, layer=-1,
            requests=tuple(view.request_id for view in views if view.samples),
            phase=mode, recipe="token_commit", shape=(len(requests),),
        )
        red_ops.append(replace(committed, family="token_commit"))

        red_ops = annotate_geometry(red_ops, active_rows=valid, physical_rows=padded)
        regions = (LoweringRegion(tuple(red_ops), runtime, model.activation_dtype,
                                  local_tokens, ()),)
        if assembly == "model":
            regions = _without_service_fragments(regions)
        lowered_red, _ = lower_regions(
            regions, graph=self.graph, replay_scope=scope, runtime=runtime,
            prefix=(entry,), resource_fusion=resource_fusion,
        )

        # Partition lowered_red into pre_lowered, per-rep-layer templates, and post_lowered
        lowered_pre = []
        lowered_post = []
        lowered_by_rep: dict[int, list] = {r: [] for r in active_rep_indices}
        seen_body = False
        for op in lowered_red:
            if op.layer >= 0:
                seen_body = True
                if op.layer not in lowered_by_rep:
                    return None
                lowered_by_rep[op.layer].append(op)
            elif not seen_body:
                lowered_pre.append(op)
            else:
                lowered_post.append(op)

        arch_templates = dict(reused_archs)
        for akey, rep_idx in unique_archs.items():
            if akey in reused_archs:
                continue
            tmpl_ops = tuple(lowered_by_rep[rep_idx])
            if not tmpl_ops:
                return None
            rep_pfx = f"s0.l{rep_idx}."
            src_prev_id = arch_input_id[akey]
            out_id = arch_output_id[akey]
            # Confirm the archetype's terminal output op_id survived lowering
            tmpl_ids = {op.op_id for op in tmpl_ops}
            if out_id not in tmpl_ids:
                out_id = tmpl_ops[-1].op_id
            if not out_id.startswith(rep_pfx):
                return None
            arch_templates[akey] = (
                rep_idx, rep_pfx, src_prev_id,
                out_id[len(rep_pfx):], tmpl_ops, [False] * len(tmpl_ops),
            )

        # Verify lowered_red's last body output matches the last active rep's output
        if active_rep_indices:
            last_rep_akey = arch_for_layer[active_rep_indices[-1]]
            _, last_rep_pfx, _, last_out_sfx, _, _ = arch_templates[last_rep_akey]
            red_last_lowered_id = last_rep_pfx + last_out_sfx
        else:
            red_last_lowered_id = body_ready

        unchanged_by_arch = {}
        prev_layers_ops = None
        prev_sel_cache = None
        if can_reuse_prev:
            (_, _, prev_tmpls, prev_layers_ops,
             prev_pre_ops, prev_post_ops, prev_sel_cache, _, _) = prev_state
            for akey, (_, _, _, out_sfx, tmpl_ops, _) in arch_templates.items():
                if akey in reused_archs:
                    unchanged_by_arch[akey] = (True,) * len(tmpl_ops)
                    continue
                p_entry = prev_tmpls.get(akey)
                if p_entry is None or p_entry[3] != out_sfx or len(p_entry[4]) != len(tmpl_ops):
                    can_reuse_prev = False
                    prev_sel_cache = None
                    break
                unchanged_by_arch[akey] = tuple(
                    a == b for a, b in zip(tmpl_ops, p_entry[4])
                )

        if can_reuse_prev and len(lowered_pre) == len(prev_pre_ops):
            lowered_pre = [
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev_pre_ops, lowered_pre)
            ]

        expanded = list(lowered_pre)
        cur_layers_ops = []
        layer_offsets_list = []
        cur_prev = body_ready
        step_meta_keys: dict[tuple[Any, int], Any] = {}
        for idx in range(len(layers)):
            layer_offsets_list.append(len(expanded))
            akey = arch_for_layer[idx]
            rep_idx, src_pfx, src_prev, out_sfx, tmpl_ops, meta_plans = arch_templates[akey]
            dst_pfx = f"s0.l{idx}."
            unchanged = unchanged_by_arch.get(akey) if can_reuse_prev else None
            if unchanged is not None and all(unchanged):
                lyr_ops = prev_layers_ops[idx]
                expanded.extend(lyr_ops)
                cur_layers_ops.append(lyr_ops)
            elif idx == rep_idx and cur_prev == src_prev and unchanged is None:
                expanded.extend(tmpl_ops)
                cur_layers_ops.append(tmpl_ops)
            else:
                prev_lyr = prev_layers_ops[idx] if unchanged is not None else None
                built_lyr = []
                for pos, tmpl_op in enumerate(tmpl_ops):
                    if unchanged is not None and unchanged[pos]:
                        built_lyr.append(prev_lyr[pos])
                        continue
                    if idx == rep_idx and cur_prev == src_prev:
                        built_lyr.append(tmpl_op)
                        continue
                    new_deps = tuple(
                        cur_prev if d == src_prev
                        else (d.replace(src_pfx, dst_pfx) if src_pfx in d else d)
                        for d in tmpl_op.deps
                    )
                    mplan = meta_plans[pos]
                    if mplan is False:
                        mplan = _compile_meta_plan(
                            tmpl_op.metadata, src_pfx, rep_idx, src_prev)
                        meta_plans[pos] = mplan
                    if mplan is None:
                        new_meta = tmpl_op.metadata
                    else:
                        new_meta = _apply_meta_plan(
                            tmpl_op.metadata, mplan, src_pfx, dst_pfx,
                            idx, cur_prev,
                        )
                        if idx > 0:
                            mk_slot = (akey, pos)
                            mkey = step_meta_keys.get(mk_slot)
                            if mkey is None:
                                if rep_idx > 0 or not _plan_uses_prev_id(mplan):
                                    mkey = op_meta_key_from_metadata(tmpl_op.metadata)
                                else:
                                    mkey = op_meta_key_from_metadata(new_meta)
                                step_meta_keys[mk_slot] = mkey
                            seed_meta_id_key(new_meta, mkey)
                    built_lyr.append(replace(
                        tmpl_op,
                        op_id=tmpl_op.op_id.replace(src_pfx, dst_pfx),
                        deps=new_deps,
                        layer=idx,
                        metadata=new_meta,
                    ))
                lyr_tup = tuple(built_lyr)
                expanded.extend(lyr_tup)
                cur_layers_ops.append(lyr_tup)
            cur_prev = dst_pfx + out_sfx

        built_post = []
        if cur_prev == red_last_lowered_id:
            built_post = list(lowered_post)
        else:
            for post_op in lowered_post:
                new_deps = tuple(
                    cur_prev if d == red_last_lowered_id else d
                    for d in post_op.deps
                )
                new_meta = _retarget_id_only(
                    post_op.metadata, red_last_lowered_id, cur_prev)
                if new_deps != post_op.deps or new_meta is not post_op.metadata:
                    built_post.append(replace(post_op, deps=new_deps, metadata=new_meta))
                else:
                    built_post.append(post_op)
        if can_reuse_prev and len(built_post) == len(prev_post_ops):
            built_post = [
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev_post_ops, built_post)
            ]
        expanded.extend(built_post)

        raw_final_ops = tuple(expanded)
        split_pack = None
        if selections:
            reused_sel = False
            if can_reuse_prev and prev_sel_cache is not None:
                prev_sel_in, prev_raw_ops, prev_final_sels, prev_ann_first = prev_sel_cache
                if (prev_sel_in == selections
                        and len(prev_raw_ops) == len(raw_final_ops)
                        and not any(
                            _selection_matches_any(p_op, selections)
                            or _selection_matches_any(n_op, selections)
                            for p_op, n_op in zip(prev_raw_ops, raw_final_ops)
                            if p_op is not n_op)):
                    final_selections = prev_final_sels
                    if raw_final_ops and raw_final_ops[0] is prev_raw_ops[0]:
                        final_ops = (prev_ann_first, *raw_final_ops[1:])
                    else:
                        final_ops = annotate_selections(raw_final_ops, final_selections)
                    self._prev_work_ops = final_ops
                    reused_sel = True
            if not reused_sel:
                final_selections = selection_delivery(raw_final_ops, selections)
                final_ops = annotate_selections(raw_final_ops, final_selections)
                final_ops = self._reuse_identical_ops(final_ops)
            sel_cache = (
                selections, raw_final_ops, final_selections,
                final_ops[0] if final_ops else None,
            )
        else:
            final_selections = ()
            final_ops = raw_final_ops
            self._prev_work_ops = final_ops
            sel_cache = None
            if (decode_sig is not None
                    and not (prev_state is not None
                             and prev_state[7] == decode_sig
                             and prev_state[8] is False)):
                attn_split_info = {}
                prev_splits_old = (
                    prev_state[8][2]
                    if (prev_state is not None and prev_state[8])
                    else {}
                )
                for akey in reused_archs:
                    if akey in prev_splits_old:
                        attn_split_info[akey] = prev_splits_old[akey]
                op_by_id_ann = {entry.op_id: entry}
                for rop in red_ops:
                    op_by_id_ann[rop.op_id] = rop
                for lop in lowered_red:
                    op_by_id_ann[lop.op_id] = lop
                all_verified = True
                for akey, (in_id, attn_sub_raw, attn_done_id) in raw_attn_by_arch.items():
                    in_op = op_by_id_ann.get(in_id)
                    if in_op is None:
                        all_verified = False
                        break
                    tmpl_ops = arch_templates[akey][4]
                    split_k = -1
                    for k_idx, top in enumerate(tmpl_ops):
                        if top.op_id == attn_done_id:
                            split_k = k_idx
                            break
                    if split_k < 0:
                        all_verified = False
                        break
                    tmpl_attn = tmpl_ops[:split_k + 1]
                    tmpl_ffn = tmpl_ops[split_k + 1:]
                    internal_attn_ids = {top.op_id for top in tmpl_attn} - {attn_done_id}
                    cross_dep = any(
                        (d in internal_attn_ids) for fop in tmpl_ffn for d in fop.deps
                    )
                    if not cross_dep and internal_attn_ids:
                        for fop in tmpl_ffn:
                            if fop.metadata:
                                s_meta = str(fop.metadata)
                                if any(aid in s_meta for aid in internal_attn_ids):
                                    cross_dep = True
                                    break
                    if cross_dep:
                        all_verified = False
                        break
                    attn_ann = annotate_geometry(attn_sub_raw, active_rows=valid, physical_rows=padded)
                    attn_regs = (LoweringRegion(tuple(attn_ann), runtime, model.activation_dtype,
                                                local_tokens, ()),)
                    if assembly == "model":
                        attn_regs = _without_service_fragments(attn_regs)
                    solo_attn_full, _ = lower_regions(
                        attn_regs, graph=self.graph, replay_scope=scope, runtime=runtime,
                        prefix=(in_op,), resource_fusion=resource_fusion,
                    )
                    solo_attn = (
                        solo_attn_full[1:]
                        if (solo_attn_full and solo_attn_full[0].op_id == in_id)
                        else solo_attn_full
                    )
                    if solo_attn == tmpl_attn:
                        attn_split_info[akey] = (in_op, attn_done_id, tmpl_attn)
                    else:
                        all_verified = False
                        break
                split_pack = (
                    (pre_ops, tuple(layer_offsets_list), attn_split_info, final_ops)
                    if all_verified else False
                )
            elif decode_sig is not None:
                split_pack = False
        self._prev_arch_state = (
            arch_for_layer, body_ready, arch_templates,
            tuple(cur_layers_ops), tuple(lowered_pre), tuple(built_post),
            sel_cache, decode_sig, split_pack,
        )
        return IterationWork(
            ops=final_ops,
            plan=plan, stage_count=runtime.parallel.pp,
            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
            recipe="structural", replay_scope=scope,
            implementation_selections=final_selections,
        )


def _selection_matches_any(op, records) -> bool:
    rec_name = op.recipe
    impl = dict(op.metadata).get("implementation") if op.metadata else None
    for record in records:
        sel_impl = (record.get("selected") or {}).get("implementation")
        if sel_impl and impl == sel_impl:
            return True
        if rec_name and rec_name in ((record.get("plan") or {}).get("recipes") or ()):
            return True
    return False


def _plan_uses_prev_id(plan) -> bool:
    if plan == 0:
        return True
    if isinstance(plan, int):
        return False
    kind = plan[0]
    if kind in (2, 3, 5):
        return any(_plan_uses_prev_id(sub) for _, sub in plan[1])
    return any(
        (kp == 0) or (vp is not None and _plan_uses_prev_id(vp))
        for _, kp, vp in plan[1]
    )


def _compile_meta_plan(val, src_pfx: str, src_layer: int, src_prev: str):
    """Compile a pruned subtree retargeting plan touching only layer-dependent nodes."""
    if isinstance(val, str):
        if val == src_prev:
            return 0
        if src_pfx in val:
            return 1
        return None
    if isinstance(val, tuple):
        edits = []
        for idx, item in enumerate(val):
            sub = _compile_meta_plan(item, src_pfx, src_layer, src_prev)
            if sub is not None:
                edits.append((idx, sub))
        return (2, tuple(edits)) if edits else None
    if isinstance(val, list):
        edits = []
        for idx, item in enumerate(val):
            sub = _compile_meta_plan(item, src_pfx, src_layer, src_prev)
            if sub is not None:
                edits.append((idx, sub))
        return (3, tuple(edits)) if edits else None
    if isinstance(val, dict):
        key_edits = False
        val_edits = []
        kv_items = []
        for k, v in val.items():
            kp = None
            if isinstance(k, str):
                if k == src_prev:
                    kp = 0
                elif src_pfx in k:
                    kp = 1
            if k == "layer" and type(v) is int and v == src_layer:
                vp = 4
            else:
                vp = _compile_meta_plan(v, src_pfx, src_layer, src_prev)
            if kp is not None:
                key_edits = True
            if vp is not None:
                val_edits.append((k, vp))
            kv_items.append((k, kp, vp))
        if not key_edits and not val_edits:
            return None
        if not key_edits:
            return (5, tuple(val_edits))
        return (6, tuple(kv_items))
    return None


def _apply_meta_plan(val, plan, src_pfx: str, dst_pfx: str,
                     dst_layer: int, dst_prev: str):
    if plan == 0:
        return dst_prev
    if plan == 1:
        return val.replace(src_pfx, dst_pfx)
    if plan == 4:
        return dst_layer
    kind = plan[0]
    if kind == 2:
        out = list(val)
        for idx, sub in plan[1]:
            out[idx] = _apply_meta_plan(
                val[idx], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return tuple(out)
    if kind == 3:
        out = list(val)
        for idx, sub in plan[1]:
            out[idx] = _apply_meta_plan(
                val[idx], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return out
    if kind == 5:
        out = val.copy()
        for k, sub in plan[1]:
            out[k] = _apply_meta_plan(
                val[k], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return out
    out = {}
    for k, kp, vp in plan[1]:
        nk = (dst_prev if kp == 0 else k.replace(src_pfx, dst_pfx)) if kp is not None else k
        nv = (_apply_meta_plan(val[k], vp, src_pfx, dst_pfx, dst_layer, dst_prev)
              if vp is not None else val[k])
        out[nk] = nv
    return out


def _retarget_id_only(val, old_id: str, new_id: str):
    if isinstance(val, str):
        return new_id if val == old_id else val
    if isinstance(val, tuple):
        changed = False
        out = []
        for item in val:
            new_item = _retarget_id_only(item, old_id, new_id)
            if new_item is not item:
                changed = True
            out.append(new_item)
        return tuple(out) if changed else val
    if isinstance(val, list):
        changed = False
        out = []
        for item in val:
            new_item = _retarget_id_only(item, old_id, new_id)
            if new_item is not item:
                changed = True
            out.append(new_item)
        return out if changed else val
    if isinstance(val, dict):
        changed = False
        out = {}
        for k, v in val.items():
            nk = new_id if k == old_id else k
            nv = _retarget_id_only(v, old_id, new_id)
            if nk is not k or nv is not v:
                changed = True
            out[nk] = nv
        return out if changed else val
    return val


def _media_counts(runtime):
    import json

    raw = dict(runtime.metadata).get("media_tokens_by_request", "{}")
    return json.loads(raw) if isinstance(raw, str) else raw
