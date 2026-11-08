"""[Role] Layer-3/4 model & framework piecewise assembly.
[Invariants] Deterministic subgraph evaluation; bounded cache; explicit state threading.
[Timeouts & Locks] Pure in-memory arithmetic; no locks or blocking waits.
[Side-Effect Policy] Side-effect free; immutable records, no file/network/device I/O.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import LayerPlan, ModelSpec, RequestSpec
from mindie_sim.prediction.piecewise.compiler import (CompiledSubgraph, ImplementationPolicy, OperatorPiecewiseCompiler,
    PipeSupply)
from mindie_sim.prediction.piecewise.composition import (ContentionEnvelope, SerialResult, advance_state,
    contention_envelope, evaluate_serial_split, p2p_transfer_cost)
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.subgraphs.builders_attention import gemma4_model_spec, _gemma_config, _gemma_global
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig, InstantiatedModel, LayerInstances, instantiate_model
from mindie_sim.prediction.subgraphs.registry import TemplateRegistry
from mindie_sim.prediction.subgraphs.state import (CompressionState, MoeRoutingState, ResidualSlotState,
    SharedKvContract, StateView)
_ACTIVE_KINDS = frozenset({"A0", "A1", "A2", "A3", "A4", "K0", "K1", "I0", "I1", "F1", "F2", "F3", "D0"})
_ATTENTION = frozenset({"attention", "mla"})
_FAST_FIELDS = ( "attention", "ffn", "architecture", "compress_ratio", "indexer", "window",
    "num_heads", "num_kv_heads", "head_dim", "intermediate_size", "num_experts",
    "top_k", "moe_intermediate_size", "num_shared_experts", "shared_intermediate_size",
    "shared_expert_gate", "linear_heads", "linear_head_dim", "linear_key_heads",
    "linear_key_head_dim", "linear_value_heads", "conv_kernel", "qk_norm",
    "q_output_gate", "mla_variant", "mla_output_gate", "expert_latent_size",
    "o_groups", "o_lora_rank", "residual_block_size", "ffn_activation", "cache_kind", "compressor_kind", )
def _frac(value: Any) -> Fraction: return value if isinstance(value, Fraction) else Fraction(str(value))
def _ms(value: Any) -> float: return float(value)
@dataclass(frozen=True)
class OperatorRow:
    layer: int; kind: str; role: str; cost: Any
    padded_tokens: int; active_tokens: int; geometry: str = "active"
    @property
    def duration_ms(self) -> Fraction:
        return self.cost.duration_ms
    def to_dict(self) -> dict:
        return {"layer": self.layer, "kind": self.kind, "role": self.role,
                "name": self.cost.name, "family": self.cost.family, "duration_ms": str(self.cost.duration_ms),
                "dominant_pipe": self.cost.dominant_pipe, "padded_tokens": self.padded_tokens,
                "active_tokens": self.active_tokens, "geometry": self.geometry, "fused": self.cost.family == "fused"}
@dataclass(frozen=True)
class SubgraphPrediction:
    instance_id: str; kind: str; variant: str; role: str
    duration_ms: Fraction; rows: tuple[OperatorRow, ...]
    fusion_results: tuple[Any, ...] = ()
    def to_dict(self) -> dict:
        return {"instance_id": self.instance_id, "kind": self.kind, "variant": self.variant, "role": self.role,
                "duration_ms": str(self.duration_ms), "operators": [row.to_dict() for row in self.rows],
                "fusions": [item.to_dict() for item in self.fusion_results]}
@dataclass(frozen=True)
class LayerPrediction:
    layer: int
    stage: int
    duration_ms: Fraction
    subgraphs: tuple[SubgraphPrediction, ...]
    state_in: StateView
    state_out: StateView
    group: int = -1
    reused: bool = False
    contention: ContentionEnvelope | None = None
    operator_duration_ms: Fraction = Fraction(0)
    @property
    def rows(self) -> tuple[OperatorRow, ...]:
        return tuple(row for item in self.subgraphs for row in item.rows)
    def to_dict(self) -> dict:
        return {"layer": self.layer, "stage": self.stage, "duration_ms": str(self.duration_ms),
                "operator_duration_ms": str(self.operator_duration_ms), "group": self.group, "reused": self.reused,
                "subgraphs": [item.to_dict() for item in self.subgraphs],
                "contention": self.contention.to_dict() if self.contention else None}
@dataclass(frozen=True)
class ModelPrediction:
    layers: tuple[LayerPrediction, ...]
    globals: tuple[SubgraphPrediction, ...]
    total_ms: Fraction
    state_out: StateView
    active: WorkloadGeometry
    padded: WorkloadGeometry | None
    supply: PipeSupply
    evaluation_count: int = 0
    template_groups: int = 0
    @property
    def rows(self) -> tuple[OperatorRow, ...]:
        return (tuple(row for layer in self.layers for row in layer.rows)
                + tuple(row for item in self.globals for row in item.rows))
    def bottlenecks(self, limit: int = 8) -> tuple[tuple[str, float], ...]:
        families, pipes = defaultdict(Fraction), defaultdict(Fraction)
        for row in self.rows:
            families[row.cost.family] += row.cost.duration_ms
            for pipe, value in row.cost.times:
                pipes[pipe] += value
        merged = {f"family:{key}": value for key, value in families.items()}
        merged.update({f"pipe:{key}": value for key, value in pipes.items()})
        ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        return tuple((key, _ms(value)) for key, value in ordered[:limit])
    def to_dict(self) -> dict:
        return {"total_ms": str(self.total_ms), "active_tokens": self.active.tokens,
                "padded_tokens": (self.padded.effective_tokens if self.padded else self.active.tokens),
                "state": self.state_out.to_dict(), "layers": [item.to_dict() for item in self.layers],
                "globals": [item.to_dict() for item in self.globals], "bottlenecks": dict(self.bottlenecks()),
                "evaluation_count": self.evaluation_count, "template_groups": self.template_groups}
@dataclass(frozen=True)
class LayerTemplateGroup:
    group_id: int; key: tuple; indices: tuple[int, ...]; representative: int
@dataclass(frozen=True)
class _CachedLayer:
    duration_ms: Fraction; subgraphs: tuple[SubgraphPrediction, ...]
    contention: ContentionEnvelope | None; operator_duration_ms: Fraction
class ModelPipeline:
    """A compiled model walk with layer-template aggregation and state ports."""
    def __init__(self, model: ModelSpec, features: FeatureConfig, supply: PipeSupply,
                 policy: ImplementationPolicy | None = None, *, registry: TemplateRegistry | None = None):
        if not isinstance(supply, PipeSupply):
            raise ModelingError("model_pipeline_requires_pipe_supply")
        self.model, self.features, self.supply = model, features, supply
        self.policy = policy or ImplementationPolicy()
        self.instantiated: InstantiatedModel = instantiate_model( model, features, registry=registry)
        self.compiler = OperatorPiecewiseCompiler(supply, self.policy)
        self._compiled: dict[tuple, CompiledSubgraph] = {}
        self._cache: dict[tuple, _CachedLayer] = {}
        self.groups = self._build_groups()
        self._group_of = {layer: group.group_id for group in self.groups for layer in group.indices}
    def _subgraph(self, instance: Any) -> CompiledSubgraph:
        key = (instance.performance_key(), instance.layer)
        if key not in self._compiled:
            self._compiled[key] = self.compiler.compile(instance)
        return self._compiled[key]
    def _layer_key(self, layer: LayerInstances) -> tuple:
        plan = self.model.layer_plans[layer.layer]
        structural = tuple(getattr(plan, name, None) for name in _FAST_FIELDS)
        slots = tuple((slot.kind, slot.variant, slot.role) for slot in layer.slots)
        owner = ("owner" if plan.owns_long_kv or plan.kv_owner_layer_id == plan.index
                 else f"consumer:{plan.kv_owner_layer_id}")
        return (structural, slots, layer.stage, owner, tuple(item.fingerprint() for item in layer.subgraphs))
    def _build_groups(self) -> tuple[LayerTemplateGroup, ...]:
        buckets: dict[tuple, list[int]] = {}
        for layer in self.instantiated.layers:
            buckets.setdefault(self._layer_key(layer), []).append(layer.layer)
        return tuple(LayerTemplateGroup(index, key, tuple(indices), indices[0])
                     for index, (key, indices) in enumerate(buckets.items()))
    @property
    def globals(self) -> tuple[Any, ...]:
        return self.instantiated.globals
    @property
    def draft_subgraph(self) -> Any | None:
        return next((item for item in self.globals if item.kind == "D0"), None)
    def shared_kv_contracts(self) -> tuple[SharedKvContract, ...]:
        owners: dict[int, list[int]] = defaultdict(list)
        tp = max(1, self.features.tp)
        kv_bytes = dtype_bytes(self.features.kv_dtype)
        row_bytes = {}
        for plan in self.model.layer_plans:
            if plan.attention == "mla" or plan.cache_kind == "window_compressed":
                elements = plan.cache_row_elements
            else:
                heads = int(plan.num_kv_heads or self.model.num_kv_heads)
                elements = max(1, math.ceil(heads / tp)) * 2 * int( plan.head_dim or self.model.head_dim)
            row_bytes[plan.index] = int(elements * kv_bytes)
        stage_of = self.instantiated.stage_of
        for plan in self.model.layer_plans:
            owner = int(plan.kv_owner_layer_id)
            if 0 <= owner < plan.index and plan.attention in {"gqa", "mla"}:
                owners[owner].append(plan.index)
        contracts = []
        for owner, consumers in sorted(owners.items()):
            cross = any(stage_of[consumer] != stage_of[owner] for consumer in consumers)
            contracts.append(SharedKvContract( owner, tuple(consumers), row_bytes[owner], cross_stage=cross,
                transfer_bytes=row_bytes[owner] if cross else 0))
        return tuple(contracts)
    @staticmethod
    def _selected_rows(active: WorkloadGeometry, emit_mask: Sequence[bool] | None = None ) -> WorkloadGeometry | None:
        """Selected logits rows: one per completing request, all verify rows."""
        requests = []
        for index, request in enumerate(active.requests):
            if request.q_len <= 0:
                continue
            if emit_mask is not None and index >= len(emit_mask):
                continue
            if emit_mask is not None and not emit_mask[index]:
                continue
            if request.phase == "verify":
                q_len, context = request.q_len, request.context
            else:
                q_len, context = 1, request.context + request.q_len - 1
            requests.append(RequestGeometry( q_len, context, phase=request.phase, replica=request.replica,
                draft_len=request.draft_len))
        if not requests:
            return None
        return WorkloadGeometry(tuple(requests), media_tokens=active.media_tokens, phase=active.phase_name)

    def _geometry_for(self, kind: str, operator: Any, active: WorkloadGeometry, padded: WorkloadGeometry | None,
                      selected: WorkloadGeometry | None = None ) -> tuple[WorkloadGeometry, str]:
        if kind == "O0":
            chosen = selected or active
            if operator.name == "hidden_select":
                return chosen, "selected"
            if operator.name == "final_norm" and self.features.final_norm_scope == "all":
                return active, "active"
            return chosen, "selected"
        if padded is None:
            return active, "active"
        if kind in {"A0", "A1"}:
            return ((active, "active") if operator.family in _ATTENTION else (padded, "padded"))
        if kind in {"A2", "A3", "A4"}:
            return ((active, "active") if operator.family in {"attention", "mla", "recurrent", "kv_access"}
                    else (padded, "padded"))
        if kind in _ACTIVE_KINDS:
            return active, "active"
        return padded, "padded"
    def _prediction(self, compiled: CompiledSubgraph, result: SerialResult,
                    active: WorkloadGeometry, padded: WorkloadGeometry | None,
                    selected: WorkloadGeometry | None = None) -> SubgraphPrediction:
        rows = []
        for cost in result.costs:
            if cost.family == "fused":
                fusion = next((item for item in result.fusion_results if f"fused:{item.name}" == cost.name), None)
                first = (compiled.operator(fusion.semantic[0].name) if fusion else compiled.operators[0])
                used, geometry = self._geometry_for( compiled.instance.kind, first, active, padded, selected)
            else:
                used, geometry = self._geometry_for( compiled.instance.kind, compiled.operator(cost.name),
                    active, padded, selected)
            rows.append(OperatorRow( compiled.instance.layer, compiled.instance.kind,
                str(dict(compiled.instance.metadata).get("role", "")), cost,
                used.effective_tokens, active.tokens, geometry))
        return SubgraphPrediction( compiled.instance.instance_id, compiled.instance.kind, compiled.instance.variant,
            str(dict(compiled.instance.metadata).get("role", "")),
            result.total_ms, tuple(rows), result.fusion_results)
    def _run(self, compiled: Iterable[CompiledSubgraph], state: StateView,
             active: WorkloadGeometry, padded: WorkloadGeometry | None,
             fusion: bool, selected: WorkloadGeometry | None = None,
             skip: Sequence[str] = ()) -> tuple[tuple[SubgraphPrediction, ...], StateView, Fraction]:
        current, predictions, duration = state, [], Fraction(0)
        self._last_contention: ContentionEnvelope | None = None
        a1_stages, idx_stages, a1_ms, idx_ms = [], [], Fraction(0), Fraction(0)
        for item in compiled:
            names = (tuple(contract.name for contract in item.instance.template.fusions) if fusion else ())
            result = evaluate_serial_split( item, active, current,
                geometry_for=lambda operator, kind=item.instance.kind:
                    self._geometry_for(kind, operator, active, padded, selected)[0], fusion=names, skip=skip)
            if result.contention is not None:
                self._last_contention = result.contention
            if self.features.multistream_attention_indexer and item.instance.kind == "A1":
                prolog_names = {"q_down", "q_norm", "q_up", "q_direct", "kv_down", "kv_norm", "rotary", "kv_store", "fused:mla_prolog"}
                prolog = [c for c in result.costs if c.name in prolog_names]
                for c in prolog:
                    a1_stages.extend([list(s) for s in c.stage_work] if c.stage_work else ([list(c.work)] if c.work else []))
                    a1_ms += c.duration_ms
            elif self.features.multistream_attention_indexer and item.instance.kind in {"I0", "I1"}:
                idx_stages.extend(self._stages(result)); idx_ms += result.total_ms
            predictions.append(self._prediction(item, result, active, padded, selected))
            current, duration = result.state, duration + result.total_ms
        if a1_stages and idx_stages:
            self._last_contention = contention_envelope({"attention": a1_stages, "indexer": idx_stages}, self.supply)
            duration += self._last_contention.makespan_ms - (a1_ms + idx_ms)
        return tuple(predictions), current, duration
    @staticmethod
    def _stages(result: SerialResult) -> list[list[tuple[str, Fraction]]]:
        stages = []
        for cost in result.costs:
            if cost.stage_work:
                stages.extend([list(stage) for stage in cost.stage_work])
            elif cost.work:
                stages.append(list(cost.work))
        return stages
    def _evaluate_layer(self, layer: LayerInstances, state: StateView,
                        active: WorkloadGeometry, padded: WorkloadGeometry | None,
                        fusion: bool, selected: WorkloadGeometry | None = None ) -> _CachedLayer:
        compiled = tuple(self._subgraph(item) for item in layer.subgraphs)
        indices = [index for index, item in enumerate(compiled) if item.instance.kind in {"F2", "F3", "F4"}]
        can_fork = (self.features.multistream_shared_expert and any(item.instance.kind == "F4" for item in compiled)
                    and any(item.instance.kind in {"F2", "F3"} for item in compiled))
        if not can_fork:
            predictions, _, duration = self._run(compiled, state, active, padded, fusion, selected)
            return _CachedLayer(duration, predictions, self._last_contention, duration)
        first, last = min(indices), max(indices)
        prefix, suffix = compiled[:first], compiled[last + 1:]
        selected = compiled[first:last + 1]
        def is_join(item) -> bool:
            return (item.instance.kind == "F4" and str(dict(item.instance.metadata).get("role", "")) == "join")
        joins = tuple(item for item in selected if is_join(item))
        shared = tuple(item for item in selected if item.instance.kind == "F4" and not is_join(item))
        routed = tuple(item for item in selected if item.instance.kind in {"F2", "F3"})
        branches = {}
        if shared:
            branches["shared"] = shared
        if routed:
            branches["routed"] = routed
        prefix_rows, current, prefix_ms = self._run(prefix, state, active, padded, fusion, selected)
        base = current
        branch_rows, work, branch_states = [], {}, {}
        for name, items in branches.items():
            branch_state = base
            work[name] = []
            for item in items:
                result = evaluate_serial_split( item, active, branch_state,
                    geometry_for=lambda operator, kind=item.instance.kind:
                        self._geometry_for(kind, operator, active, padded, selected)[0],
                    fusion=(tuple(contract.name for contract in item.instance.template.fusions) if fusion else ()))
                work[name].extend(self._stages(result))
                branch_rows.append(self._prediction(item, result, active, padded, selected))
                branch_state = result.state
            branch_states[name] = branch_state
        envelope = contention_envelope(work, self.supply)
        current = (branch_states.get("routed") or branch_states.get("shared") or base)
        join_rows, join_ms = [], Fraction(0)
        for item in joins:
            result = evaluate_serial_split( item, active, current,
                geometry_for=lambda operator, kind=item.instance.kind:
                    self._geometry_for(kind, operator, active, padded, selected)[0],
                fusion=(tuple(contract.name for contract in item.instance.template.fusions) if fusion else ()))
            join_rows.append(self._prediction(item, result, active, padded, selected))
            join_ms += result.total_ms
            current = result.state
        suffix_rows, current, suffix_ms = self._run(suffix, current, active, padded, fusion, selected)
        duration = prefix_ms + envelope.makespan_ms + join_ms + suffix_ms
        return _CachedLayer(duration, prefix_rows + tuple(branch_rows)
                            + tuple(join_rows) + suffix_rows, envelope, duration)
    def _state_digest(self, state: StateView, layer: LayerInstances) -> tuple:
        plan = self.model.layer_plans[layer.layer]
        index = layer.layer
        compression = tuple(sorted( (key, value.ratio, value.consumed, value.rows, value.remainder)
            for key, value in state.compression if key == index))
        compression_requests = tuple(sorted( ((key[0], str(key[1])), value.ratio, value.consumed, value.rows,
             value.remainder) for key, value in state.compression_requests if key[0] == index))
        routing = tuple(sorted( (key, value.counts, value.top_k, value.valid_tokens, value.local_expert_ids)
            for key, value in state.routing if key == index))
        owner = int(plan.kv_owner_layer_id)
        selection = tuple(sorted(key for key, _ in state.selection_ready if key == index))
        cross_stage = tuple(sorted((key, value) for key, value in state.cross_stage_ready
                                   if key[0] == owner))
        return (compression, compression_requests, routing, (state.residual.block_size, state.residual.retained),
                (state.acceptance.accepted, state.acceptance.committed,
                 state.acceptance.rejected, state.acceptance.commits),
                state.is_kv_ready(owner) if owner >= 0 else None, selection, cross_stage)
    @staticmethod
    def _rebase(subgraphs: tuple[SubgraphPrediction, ...], layer: int) -> tuple[SubgraphPrediction, ...]:
        return tuple(replace(item, rows=tuple(replace(row, layer=layer) for row in item.rows)) for item in subgraphs)

    def _advance_layer_state(self, layer: LayerInstances, state: StateView, active: WorkloadGeometry,
                             request_ids: Sequence[Any] | None = None,
                             commit_tokens: Sequence[int] | None = None) -> StateView:
        for instance in layer.subgraphs:
            if instance.kind == "K1":
                ratio = max(1, int(instance.operator("compress").theta.find( "ratio", 1) or 1))
                records = ((request_ids[index] if request_ids is not None and index < len(request_ids) else index,
                            request.context,
                            int(commit_tokens[index]) if commit_tokens is not None and index < len(commit_tokens)
                            else request.q_len) for index, request in enumerate(active.requests))
                state, _ = state.update_compression(layer.layer, ratio, records)
                committed = [record for (owner, _), record in state.compression_requests
                             if owner == layer.layer]
                if committed:
                    # The owner cursor is the furthest committed request, so a
                    # rolled-back speculative tail cannot leave it ahead.
                    cursor = max(committed, key=lambda record: (record.consumed, record.rows))
                    values = dict(state.compression)
                    values[layer.layer] = cursor
                    state = replace(state, compression=tuple(sorted(values.items())))
            else:
                state = advance_state(self._subgraph(instance), state, active)
        return state

    def forward(self, geometry: WorkloadGeometry, state: StateView | None = None, *,
                padded_geometry: WorkloadGeometry | None = None, fusion: bool = True, aggregate: bool = True,
                include_draft: bool = False, request_ids: Sequence[Any] | None = None,
                emit_mask: Sequence[bool] | None = None,
                acceptance: tuple[int, int] | None = None,
                commit_tokens: Sequence[int] | None = None) -> ModelPrediction:
        self._cache.clear()
        current, active, padded = state or StateView(), geometry, padded_geometry
        if current.residual.retained or current.index_ready or current.selection_ready:
            current = replace(current, residual=ResidualSlotState(current.residual.block_size, ()),
                              index_ready=(), selection_ready=())
        if self.features.ep > 1:
            routing = dict(active.routing)
            local = max(1, math.ceil(self.model.num_experts / self.features.ep))
            experts = max(1, self.model.num_experts)
            total_rows = active.for_subgraph("F3").tokens * max(1, self.model.top_k)
            base, extra = divmod(total_rows, experts)
            uniform = tuple(base + (1 if index < extra else 0) for index in range(experts))[::self.features.ep][:local]
            uniform = uniform + (0,) * (local - len(uniform))
            for layer in self.instantiated.layers:
                if (layer.layer not in routing and any(item.kind == "F3" for item in layer.subgraphs)):
                    routing[layer.layer] = uniform
            active = replace(active, routing=tuple(sorted(routing.items())))
        selected = self._selected_rows(active, emit_mask)
        rows_in, current, total = self._run( tuple(self._subgraph(item) for item in self.globals
                  if item.kind == "E0" or (item.kind == "V0"
                                           and active.media_tokens > 0)), current, active, padded, fusion)
        layers, evaluations = [], 0
        for layer in self.instantiated.layers:
            block = int(self.model.layer_plans[layer.layer].residual_block_size or 0)
            if block and current.residual.block_size != block:
                current = replace(current, residual=ResidualSlotState( block, current.residual.retained))
            key = (self._group_of[layer.layer], geometry_key(active),
                   geometry_key(padded), self._state_digest(current, layer),
                   tuple((k, tuple(v)) for k, v in active.routing if k == layer.layer), bool(fusion), bool(aggregate))
            cached = self._cache.get(key); reused = cached is not None
            if cached is None:
                cached = self._evaluate_layer(layer, current, active, padded, fusion, selected)
                if aggregate:
                    self._cache[key] = cached
                evaluations += 1
            next_state = self._advance_layer_state(layer, current, active, request_ids, commit_tokens)
            layers.append(LayerPrediction( layer.layer, layer.stage, cached.duration_ms,
                self._rebase(cached.subgraphs, layer.layer), current, next_state, self._group_of[layer.layer], reused,
                cached.contention, cached.operator_duration_ms))
            current, total = next_state, total + cached.duration_ms
        if self.features.pp > 1:
            for index in range(len(layers) - 1):
                if layers[index].stage >= layers[index + 1].stage:
                    continue
                transferred = p2p_transfer_cost( layers[index].layer, active.tokens, self.model.hidden_size,
                    dtype_bytes(self.features.activation_format), self.supply)
                row = OperatorRow(layers[index].layer, "PP", "transfer",
                                  transferred, active.tokens, active.tokens, "active")
                subgraph = SubgraphPrediction( f"PP:stage_transfer:L{layers[index].layer}", "PP", "p2p",
                    "transfer", transferred.duration_ms, (row,))
                layers[index] = replace( layers[index],
                    duration_ms=layers[index].duration_ms + transferred.duration_ms,
                    operator_duration_ms=(layers[index].operator_duration_ms + transferred.duration_ms),
                    subgraphs=layers[index].subgraphs + (subgraph,))
                total += transferred.duration_ms
        skip = (("lm_head", "logits_gather", "sampling") if selected is None else ())
        if selected is None and self.features.final_norm_scope == "selected":
            skip += ("hidden_select", "final_norm")
        rows_out, current, out_ms = self._run( tuple(self._subgraph(item) for item in self.globals
                  if item.kind == "O0" or (include_draft and item.kind == "D0")),
            current, active, padded, fusion, selected, skip)
        if acceptance is not None:
            current = current.commit(int(acceptance[0]), verified=int(acceptance[1]))
        return ModelPrediction(tuple(layers), rows_in + rows_out, total + out_ms,
                               current, active, padded, self.supply, evaluations, len(self.groups))
def geometry_key(geometry: WorkloadGeometry | None) -> tuple:
    if geometry is None:
        return ()
    return (geometry.tokens, geometry.effective_tokens, geometry.graph_bucket, geometry.cohort_batch,
            geometry.physical_tokens, geometry.media_tokens, geometry.phase_name,
            tuple((request.q_len, request.context, request.phase,
                   request.query_ranges, request.replica, request.draft_len) for request in geometry.requests),
            tuple((layer, tuple(counts)) for layer, counts in geometry.routing))
def assemble_model(model: ModelSpec | Mapping[str, Any], *, supply: PipeSupply, features: FeatureConfig | None = None,
                   policy: ImplementationPolicy | None = None,
                   registry: TemplateRegistry | None = None) -> ModelPipeline:
    """Zero-parameter entry: config -> templates -> compiled model walk."""
    if isinstance(model, Mapping):
        family = str(model.get("model_type", "")).lower()
        spec = (gemma4_model_spec(model) if family.startswith("gemma") else ModelSpec.from_mapping(model))
    elif isinstance(model, ModelSpec):
        spec = model
    else:
        raise ModelingError("model_pipeline_requires_model_spec_or_mapping")
    return ModelPipeline(spec, features or FeatureConfig.from_model(spec), supply, policy, registry=registry)
@dataclass(frozen=True)
class FrameworkConfig:
    supply: PipeSupply
    policy: ImplementationPolicy = field(default_factory=ImplementationPolicy)
    graph_buckets: tuple[int, ...] = ()
    max_batch: int = 8
    max_prefill_tokens: int = 0
    mtp_draft_len: int = 0
    mtp_accept: int = 0
    routing: tuple[tuple[int, tuple[int, ...]], ...] = ()
    graph_replay_ms: Fraction = Fraction(0)
    max_steps: int = 4096
    max_retained_steps: int = 256
    def __post_init__(self) -> None:
        if self.max_batch < 1 or self.max_steps < 1 or self.max_prefill_tokens < 0:
            raise ModelingError("framework_scheduler_geometry_invalid")
        if not 0 <= self.mtp_accept <= self.mtp_draft_len:
            raise ModelingError("framework_mtp_accept_invalid")
        if self.graph_replay_ms < 0:
            raise ModelingError("framework_replay_invalid")
        object.__setattr__(self, "graph_buckets", tuple(sorted({int(value) for value in self.graph_buckets})))
@dataclass
class _RunRequest:
    spec: RequestSpec
    prefilled: int = 0
    generated: int = 0
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_at: float | None = None
    replica: int = 0
    itl_sum: float = 0.0
    itl_count: int = 0
@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    ttft_ms: float
    tpot_ms: float
    itl_ms: float
    completion_ms: float
    prompt_tokens: int
    output_tokens: int
@dataclass(frozen=True)
class FrameworkPrediction:
    body_duration_ms: float
    ttft_ms: float
    tpot_ms: float
    itl_ms: float
    requests: tuple[RequestMetrics, ...]
    steps: tuple[dict, ...]
    bottlenecks: tuple[tuple[str, float], ...]
    padding: dict
    sync: dict
    mtp: dict | None
    def to_dict(self) -> dict:
        return {"body_duration_ms": self.body_duration_ms, "ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms,
                "itl_ms": self.itl_ms, "requests": [item.__dict__ for item in self.requests],
                "steps": list(self.steps), "bottlenecks": dict(self.bottlenecks),
                "padding": dict(self.padding), "sync": dict(self.sync), "mtp": dict(self.mtp) if self.mtp else None}
def _coerce_requests(requests: Iterable[Any]) -> tuple[RequestSpec, ...]:
    result = tuple(value if isinstance(value, RequestSpec) else RequestSpec.from_mapping(value, index)
                   for index, value in enumerate(requests))
    if not result:
        raise ModelingError("framework_requests_empty")
    return result
def _bucket_for(tokens: int, buckets: Sequence[int]) -> int:
    for value in buckets:
        if tokens <= value:
            return value
    return buckets[-1] if buckets else 0
def _padding(active: WorkloadGeometry, buckets: Sequence[int]) -> WorkloadGeometry | None:
    if not buckets:
        return None
    bucket = _bucket_for(active.tokens, buckets)
    if bucket <= active.tokens:
        return None
    return WorkloadGeometry(active.requests, graph_bucket=bucket, phase=active.phase_name)
def _ep_skew(pipeline: ModelPipeline, active: WorkloadGeometry, config: FrameworkConfig) -> tuple[float, float]:
    """Per-rank complete layer envelopes, then one join maximum per layer (rank-permutation invariant)."""
    if pipeline.features.ep <= 1:
        return 0.0, 0.0
    local = max(1, math.ceil(pipeline.model.num_experts / pipeline.features.ep))
    size = max(1, pipeline.model.num_experts)
    base, extra = divmod(active.for_subgraph("F3").tokens * max(1, pipeline.model.top_k), size)
    uniform = tuple(base + (1 if index < extra else 0) for index in range(size))
    layers = pipeline.instantiated.layers
    kv = tuple((layer.layer, "kv_cache") for layer in layers)
    index = tuple((layer.layer, "index") for layer in layers)
    selection = tuple((layer.layer, "selection") for layer in layers)
    compression = tuple((plan.index, CompressionState.start(int(plan.compress_ratio), int(plan.compress_ratio)))
                        for plan in pipeline.model.layer_plans if int(plan.compress_ratio or 0) > 1)
    seed = StateView(kv_ready=kv, index_ready=index, selection_ready=selection, compression=compression)
    layer_scores: list[list[Fraction]] = []
    for layer in layers:
        if not any(item.kind == "F3" for item in layer.subgraphs):
            continue
        counts = next((tuple(values) for key, values in config.routing if key == layer.layer), None) or uniform
        row = []
        for rank in range(pipeline.features.ep):
            shard = counts[rank * local:(rank + 1) * local]
            geometry = replace(active, routing=((layer.layer, shard),))
            row.append(pipeline._evaluate_layer(layer, seed, geometry, None, False).duration_ms)
        layer_scores.append(row)
    critical = sum((max(row) for row in layer_scores), Fraction(0))
    rank0 = sum((row[0] for row in layer_scores), Fraction(0))
    skew = sum((max(row) - min(row) for row in layer_scores), Fraction(0))
    return _ms(critical - rank0), _ms(skew)
def _draft_cost(pipeline: ModelPipeline, geometry: WorkloadGeometry, state: StateView,
                *, proposals: int | None = None) -> tuple[Fraction, Fraction]:
    draft, draft_ms, commit_ms = pipeline.draft_subgraph, Fraction(0), Fraction(0)
    if draft is None: return draft_ms, commit_ms
    heads = next((op for op in draft.operators if op.name == "draft_heads"), None)
    proposals = max(1, int(proposals or (heads.theta.find("draft_len", 1) if heads else 1) or 1))
    noncausal = bool(int(heads.theta.find("noncausal_block", 0) or 0)) if heads else (draft.template.variant == "dspark")
    body = tuple(item for item in draft.operators if item.name not in {"target_verify", "accept_commit"})
    commit = tuple(item for item in draft.operators if item.name == "accept_commit")
    steps = ((proposals, proposals),) if noncausal else tuple((1, step) for step in range(proposals))
    for q_mult, ctx_off in steps:
        step_geom = WorkloadGeometry(
            tuple(RequestGeometry(max(1, request.q_len) * q_mult, request.context + ctx_off,
                                  phase="draft", replica=request.replica)
                  for request in geometry.requests), phase="draft")
        for operator in body:
            draft_ms += pipeline.compiler.compile_operator(operator, kind="D0").evaluate(step_geom, state).duration_ms
    for operator in commit:
        commit_ms += pipeline.compiler.compile_operator(operator, kind="D0").evaluate(geometry, state).duration_ms
    return draft_ms, commit_ms
def predict_framework(model: ModelSpec | ModelPipeline | Mapping[str, Any],
                      requests: Iterable[Any], config: FrameworkConfig, *, features: FeatureConfig | None = None,
                      policy: ImplementationPolicy | None = None) -> FrameworkPrediction:
    """Zero-parameter serving entry from structure, ragged requests and config."""
    pipeline = (model if isinstance(model, ModelPipeline) else
                assemble_model(model, supply=config.supply, features=features, policy=policy or config.policy))
    dp = max(1, pipeline.features.dp)
    work = [_RunRequest(item, prefilled=min(item.prefix_tokens, item.prompt_tokens - 1), replica=index % dp)
            for index, item in enumerate(_coerce_requests(requests))]
    pending = deque(sorted(work, key=lambda item: (item.spec.arrival_ms, -item.spec.priority)))
    queues = [ (deque(), deque()) for _ in range(dp) ]  # running, prefilling per replica
    replica_now = [0.0 for _ in range(dp)]
    replica_states = [StateView() for _ in range(dp)]
    step_index = 0
    records, family_ms, pipe_ms = [], defaultdict(float), defaultdict(float)
    padded_rows = attention_rows = moe_rows = padding_waste = decode_cnt = all_cnt = max_act_tok = max_pad_tok = 0
    dp_skew_total = ep_skew_total = decode_body_sum = all_body_sum = 0.0
    mtp_draft = mtp_target = mtp_commit = 0.0
    mtp_steps = mtp_accepted = mtp_verified = 0
    if config.mtp_draft_len and pipeline.draft_subgraph is None:
        raise ModelingError("mtp_requires_draft_subgraph")
    local_experts = max(1, math.ceil(pipeline.model.num_experts / pipeline.features.ep))
    buckets = config.graph_buckets or pipeline.features.graph_buckets
    def localized(rank: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return tuple((int(layer), tuple(counts[rank * local_experts: (rank + 1) * local_experts]))
                     for layer, counts in config.routing)
    def take_pending(rank: int, limit: float) -> _RunRequest | None:
        for candidate in pending:
            if candidate.replica == rank and candidate.spec.arrival_ms <= limit + 1e-12:
                pending.remove(candidate)
                return candidate
        return None
    while pending or any(running or prefilling for running, prefilling in queues):
        if step_index >= config.max_steps:
            raise ModelingError("framework_max_steps_exceeded")
        round_runs: list[dict] = []
        round_preds: dict[int, Any] = {}
        for rank in range(dp):
            running, prefilling = queues[rank]
            start = replica_now[rank]
            admitted = take_pending(rank, start)
            while admitted is not None:
                prefilling.append(admitted)
                admitted = take_pending(rank, start)
            items = [(run, "decode", 1) for run in running]
            slots = max(1, config.max_batch) - len(items)
            while prefilling and slots > 0:
                run = prefilling.popleft()
                remaining = run.spec.prompt_tokens - run.prefilled
                chunk = remaining if not config.max_prefill_tokens else min(remaining, config.max_prefill_tokens)
                items.append((run, "prefill", max(1, chunk)))
                slots -= 1
            if not items:
                future = [run.spec.arrival_ms for run in pending if run.replica == rank]
                if future and not running and not prefilling:
                    replica_now[rank] = max(start, min(future))
                continue
            kinds = {kind for _, kind, _ in items}
            phase = next(iter(kinds)) if len(kinds) == 1 else "mixed"
            requests_geom = [RequestGeometry(
                chunk, run.prefilled if kind == "prefill" else run.spec.prompt_tokens + max(0, run.generated - 1),
                phase=kind, replica=rank) for run, kind, chunk in items]
            active = WorkloadGeometry(tuple(requests_geom), phase=phase)
            mtp = (config.mtp_draft_len > 0 and all(kind == "decode" for _, kind, _ in items))
            if mtp:
                declared = pipeline.draft_subgraph.operator("target_verify").theta.find("verify_tokens", 0) or 0
                verify_rows = max(1 + config.mtp_draft_len, int(declared))
                active = WorkloadGeometry(tuple(RequestGeometry(
                    verify_rows, request.context, phase="verify", replica=rank) for request in active.requests),
                    phase="verify")
            routing = localized(rank)
            if routing:
                active = WorkloadGeometry(active.requests, routing=routing, phase=active.phase_name)
            ids = tuple(run.spec.request_id for run, _, _ in items)
            emit = tuple(kind == "decode" or (kind == "prefill" and run.prefilled + chunk >= run.spec.prompt_tokens)
                         for run, kind, chunk in items)
            accepted, commits = 0, None
            if mtp:
                commits = tuple(min(config.mtp_accept + 1, run.spec.output_tokens - run.generated)
                                for run, _, _ in items)
                accepted = sum(min(config.mtp_accept, max(0, commit - 1)) for commit in commits)
            padded = _padding(active, buckets)
            prediction = pipeline.forward(active, replica_states[rank], padded_geometry=padded,
                request_ids=ids, emit_mask=emit, commit_tokens=commits,
                acceptance=((accepted, len(items) * config.mtp_draft_len) if mtp else None))
            round_preds[rank] = prediction
            round_runs.append(dict(rank=rank, start=start, items=items, active=active, padded=padded,
                                   phase=phase, mtp=mtp, prediction=prediction, commits=commits))
        if not round_runs:
            continue
        prediction = max(round_preds.values(), key=lambda item: item.total_ms)
        dp_skew = (max(item.total_ms for item in round_preds.values())
                   - min(item.total_ms for item in round_preds.values()))
        reference_ms = (pipeline.forward(prediction.active, StateView()).total_ms
                        if prediction.padded is not None else prediction.total_ms)
        ep_delta, ep_skew = _ep_skew(pipeline, prediction.active, config)
        ep_delta, ep_skew = _frac(ep_delta), _frac(ep_skew)
        draft_ms, commit_ms = (_draft_cost(pipeline, WorkloadGeometry(
                tuple(RequestGeometry(1, request.context, phase="draft", replica=request.replica)
                      for request in prediction.active.requests), phase="draft"),
            replica_states[0], proposals=config.mtp_draft_len)
            if any(item["mtp"] for item in round_runs) else (Fraction(0), Fraction(0)))
        overhead = max(Fraction(0), ep_delta) + draft_ms + commit_ms + _frac(config.graph_replay_ms)
        body_local = {rank: item.total_ms + overhead for rank, item in round_preds.items()}
        body = max(body_local.values())
        for row in prediction.rows:
            family_ms[row.cost.family] += _ms(row.cost.duration_ms)
            for pipe, value in row.cost.times:
                pipe_ms[pipe] += _ms(value)
        if prediction.padded:
            padded_rows += sum(row.padded_tokens for row in prediction.rows if row.geometry == "padded")
            padding_waste += max(0.0, _ms(prediction.total_ms - reference_ms))
        attention_rows += sum(row.active_tokens for row in prediction.rows if row.cost.family in _ATTENTION)
        moe_rows += sum(int(row.cost.get("moe_rows", 0)) for row in prediction.rows if row.cost.family == "grouped_gemm")
        completed, accepted_total, mtp_verified_round = [], 0, 0
        active_tokens = padded_tokens = 0
        ids_all: list[str] = []
        for entry in round_runs:
            rank, start, items = entry["rank"], entry["start"], entry["items"]
            running, _ = queues[rank]
            local_end = start + _ms(body_local[rank])
            active_tokens += entry["active"].tokens
            padded_tokens += entry["padded"].effective_tokens if entry["padded"] else entry["active"].tokens
            ids_all.extend(run.spec.request_id for run, _, _ in items)
            mtp_verified_round += len(items) * config.mtp_draft_len if entry["mtp"] else 0
            for index, (run, kind, chunk) in enumerate(items):
                if kind == "prefill":
                    run.prefilled += chunk
                    if run.prefilled >= run.spec.prompt_tokens:
                        run.first_token_at = run.last_token_at = local_end
                        run.generated = 1
                        if run.spec.output_tokens <= 1:
                            run.completion_at = local_end
                            completed.append(run)
                        else:
                            running.append(run)
                    else:
                        queues[rank][1].appendleft(run)
                else:
                    committed = (entry["commits"][index] if entry["mtp"] else 1)
                    accepted = committed - 1 if entry["mtp"] else 0
                    if run.last_token_at is not None:
                        run.itl_sum += max(0.0, local_end - run.last_token_at)
                        run.itl_count += max(1, committed)
                    accepted_total += accepted
                    run.generated += committed
                    run.last_token_at = local_end
                    if run.generated >= run.spec.output_tokens:
                        run.completion_at = local_end
                        completed.append(run)
            replica_now[rank] = local_end
        for run in completed:
            if run in queues[run.replica][0]:
                queues[run.replica][0].remove(run)
        for rank, item in round_preds.items():
            state = item.state_out
            for run in completed:
                if run.replica == rank:
                    state = state.retire_compression(run.spec.request_id)
            replica_states[rank] = state
        if any(item["mtp"] for item in round_runs):
            mtp_steps += 1
            mtp_accepted += accepted_total
            mtp_verified += mtp_verified_round
            mtp_draft += _ms(draft_ms)
            mtp_target += _ms(prediction.total_ms)
            mtp_commit += _ms(commit_ms)
        dp_skew_total += _ms(dp_skew)
        ep_skew_total += _ms(ep_skew)
        phase = next(iter({item["phase"] for item in round_runs})) if len(
            {item["phase"] for item in round_runs}) == 1 else "mixed"
        body_val = _ms(body)
        all_body_sum += body_val; all_cnt += 1; max_act_tok = max(max_act_tok, active_tokens); max_pad_tok = max(max_pad_tok, padded_tokens)
        if phase == "decode": decode_body_sum += body_val; decode_cnt += 1
        if len(records) < max(1, int(config.max_retained_steps)):
            records.append({"index": int(step_index), "kind": phase, "batch": sum(len(item["items"]) for item in round_runs),
                            "active_tokens": active_tokens, "padded_tokens": padded_tokens,
                            "body_ms": body_val, "sync_wait_ms": _ms(dp_skew + ep_skew),
                            "request_ids": tuple(ids_all)})
        step_index += 1
    metrics = []
    for run in work:
        if run.first_token_at is None or run.completion_at is None:
            raise ModelingError(f"request_not_completed:{run.spec.request_id}")
        metrics.append(RequestMetrics( run.spec.request_id, run.first_token_at - run.spec.arrival_ms,
            (run.completion_at - run.first_token_at) / max(1, run.spec.output_tokens - 1),
            run.itl_sum / run.itl_count if run.itl_count else 0.0, run.completion_at - run.spec.arrival_ms,
            run.spec.prompt_tokens, run.spec.output_tokens))
    body = (decode_body_sum / decode_cnt if decode_cnt else all_body_sum / max(1, all_cnt))
    bottlenecks = sorted({**{f"family:{key}": value for key, value in family_ms.items()},
                          **{f"pipe:{key}": value for key, value in pipe_ms.items()}}.items(),
                         key=lambda item: (-item[1], item[0]))
    padding = {"bucket": max_pad_tok, "active_tokens": max_act_tok, "padded_tokens": max_pad_tok,
               "operator_padded_rows": padded_rows, "attention_active_rows": attention_rows,
               "moe_active_rows": moe_rows, "padding_waste_ms": padding_waste}
    sync = {"dp_ranks": dp, "ep_ranks": pipeline.features.ep,
            "dp_skew_ms": dp_skew_total, "ep_skew_ms": ep_skew_total, "sync_wait_ms": dp_skew_total + ep_skew_total,
            "source": "dp_request_skew+ep_routing_skew"}
    mtp_report = ({"draft_len": config.mtp_draft_len, "accepted": mtp_accepted,
                   "verified": mtp_verified, "draft_ms": mtp_draft, "target_ms": mtp_target,
                   "commit_ms": mtp_commit, "steps": mtp_steps,
                   "total_ms": mtp_draft + mtp_target + mtp_commit,
                   "acceptance_factor": mtp_accepted / max(1, mtp_verified)}
                  if config.mtp_draft_len else None)
    return FrameworkPrediction( body, sum(item.ttft_ms for item in metrics) / len(metrics),
        sum(item.tpot_ms for item in metrics) / len(metrics),
        sum(item.itl_ms for item in metrics) / len(metrics), tuple(metrics),
        tuple(records), tuple(bottlenecks[:12]), padding, sync, mtp_report)
