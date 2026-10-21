"""Request-level framework simulation over the shared-resource engine.

[Role]
  Reproduce a declared vLLM-like token-budget scheduler (running first, then
  waiting; chunked prefill; one in-flight iteration), lower each iteration plan
  to shape/work DAGs, execute them on the resource engine, and advance request
  state from resource completions.  Produce TTFT/ITL/TPOT/E2E, throughput,
  SLO goodput and iteration/shape/recipe traces.

[Invariants]
  * A partial prompt chunk never produces a first token.  TTFT is recorded only
    when the prompt is fully computed and a sample is actually executed.
  * Token budgets are consumed by both prefill chunks and decode tokens.  A
    second chunk of one token can be in a mixed batch filled by waiting
    prefill; the target request waits for that whole batch.
  * KV capacity and graph buckets are explicit.  Graph padding adds physical
    work but never valid output tokens.
  * The scheduler uses one in-flight iteration per DP replica in this version;
    this is reported as a support boundary, not hidden behind a queue model.
  * DP replicas are independent device groups (no cross-DP collective assumed);
    wall time is the maximum replica completion, not a product of TP*EP*DP.
  * No measured field is synthesized.  If a request cannot be served under the
    declared capacity, it is reported unserved with a reason.
  * A supplied fitted physical artifact is one machine's calibration: a foreign
    hardware key is refused, and every op carries the artifact's own answer (or
    its refusal reason).  The measurement timing scope is a declared input
    (``device_event_warm`` / ``device_event_warm_batched`` / ...), never inferred
    from a shape, because each scope carries a different launch intercept.

[Timeouts & Locks]
  Deterministic virtual time only.  Iteration guard prevents infinite loops;
  no wall-clock sleeps, threads, locks or device calls.

[Side-Effect Policy]
  No filesystem, network, provider import or device mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Sequence

from .a3_cost import (
    LARGE_OPERATOR_FAMILIES,
    PHYSICAL_CALIBRATED, PHYSICAL_LOW_CONFIDENCE, PHYSICAL_UNMODELED,
    CalibrationStore, PredictiveCostModel, communication_placement_for_machine,
    physical_calibration_provenance,
)
from .a3_resources import (
    ResourceEngine, ResourceSpec, add_costed_ops, build_resource_specs,
    fusion_pool_capacities, plan_replica_links, select_link_for_ranks,
)
from .a3_types import A3InputError, CaseSpec, ModelSpec, ParallelSpec, RuntimeSpec, SloSpec
from .a3_async import (
    add_prepare_tasks, prepare_residency_report, prepare_summary,
)
from .a3_predict_global import predict_global_replicas
from .a3_request import (
    runtime_with_draft_next_counts,
    RequestState, TokenLinearCostBackend, _backend_duration, with_media_runtime,
)
from .a3_metrics import _merge_replicas, _slo_goodput, _percentile
from .a3_workload import (
    IterationPlan, MemoryPlan, RequestView, derive_iteration_work, fits_workload,
    memory_plan, select_recipe,
)

MAX_ITERATIONS = 100_000
#: Timing scope the analytic path has always declared.  A caller may override it
#: (``predict(..., timing_scope=...)`` or ``--timing-scope``), and the scope is
#: then handed unchanged to the fitted physical layer: ``device_event_warm``
#: (per-call interval) and ``device_event_warm_batched`` (kernel service inside a
#: batched graph) carry *different* launch intercepts and are never pooled.
DEFAULT_TIMING_SCOPE = "device_event_warm"


@dataclass(frozen=True)
class _PredictiveCase(CaseSpec):
    """A case carrying the per-run physical-calibration context.

    ``CaseSpec`` has no field for a fitted artifact and lives in
    ``a3_types.py`` (outside this work item), while ``predict_global_replicas``
    builds every ``_ReplicaRunner`` from the case it is given.  The context
    therefore rides on the case as two additive, non-serialized fields: nothing
    in ``to_plan``/``semantic_payload`` reads them, so the case's public shape
    and signature are unchanged.
    """

    physical_calibration: Any = None
    timing_scope: str = ""


def _predictive_case(
    case: CaseSpec, *, physical_calibration: Any, timing_scope: str,
) -> _PredictiveCase:
    if isinstance(case, _PredictiveCase):
        return replace(
            case, physical_calibration=physical_calibration,
            timing_scope=timing_scope,
        )
    values = {item.name: getattr(case, item.name) for item in fields(CaseSpec)}
    return _PredictiveCase(
        **values, physical_calibration=physical_calibration,
        timing_scope=timing_scope,
    )


def _machine_key_for(case: CaseSpec) -> str:
    return str(
        case.machine.hardware_key or case.machine.sku or case.machine.name or ""
    ).strip()


def _refuse_foreign_physical_calibration(
    case: CaseSpec, physical_calibration: Any,
) -> None:
    """A fitted artifact is one machine's calibration; never retarget silently."""
    if physical_calibration is None:
        return
    artifact_key = str(getattr(physical_calibration, "hardware_key", "") or "")
    machine_key = _machine_key_for(case)
    if artifact_key and machine_key and artifact_key != machine_key:
        raise A3InputError(
            f"physical_calibration_hardware_key_conflict:{artifact_key}!={machine_key}"
        )


def _physical_calibration_summary(
    physical_calibration: Any, *, timing_scope: str, replica_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Artifact provenance plus how the fitted layer was actually used."""
    provenance = physical_calibration_provenance(
        physical_calibration, timing_scope=timing_scope,
    )
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    transfers: dict[str, int] = {}
    for result in replica_results:
        for trace in result.get("traces", ()) or ():
            for op in trace.get("ops", ()) or ():
                record = op.get("physical")
                if not isinstance(record, Mapping):
                    continue
                status = str(record.get("status") or "")
                counts[status] = counts.get(status, 0) + 1
                if status != PHYSICAL_CALIBRATED:
                    reason = str(record.get("reason") or "")
                    key = f"{status}:{reason}"
                    reasons[key] = reasons.get(key, 0) + 1
                source = record.get("transfer_from")
                if status == PHYSICAL_CALIBRATED and isinstance(source, Mapping):
                    target = dict(record.get("transfer_to") or {})
                    key = (
                        f"{source.get('family')}->{target.get('family')}/"
                        f"{target.get('recipe')}"
                    )
                    transfers[key] = transfers.get(key, 0) + 1
    provenance["applied_op_count"] = counts.get(PHYSICAL_CALIBRATED, 0)
    provenance["low_confidence_op_count"] = counts.get(PHYSICAL_LOW_CONFIDENCE, 0)
    provenance["unmodeled_op_count"] = counts.get(PHYSICAL_UNMODELED, 0)
    provenance["unmodeled_reasons"] = dict(sorted(reasons.items()))
    # A transferred row is a declared approximation, not the op's own fit: it is
    # counted separately so a reader can see how much of the answer is transfer.
    provenance["applied_transfer_op_count"] = sum(transfers.values())
    provenance["transfer_counts"] = dict(sorted(transfers.items()))
    provenance.update(_wiring_summary(replica_results))
    return provenance


def _wiring_summary(
    replica_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Large-operator wiring status of the whole prediction.

    Every trace op is classified into the closed wiring vocabulary
    (``physical_calibrated`` / ``physical_transfer`` / ``mc2_overlap_unverified``
    / ``mc2_overlap_calibrated`` / ``unmodeled``) and the large-operator families
    are counted separately, so "the big operators are wired" is visible without
    walking the traces.  The MC2 path is counted on its own: a model with MoE must
    show it.
    """
    wiring: dict[str, int] = {}
    large: dict[str, int] = {}
    by_family: dict[str, int] = {}
    scope_aliases: dict[str, int] = {}
    mc2_ops = 0
    modeled_ms = 0.0
    applied_ms = 0.0
    large_ms = 0.0
    large_applied_ms = 0.0
    for result in replica_results:
        for trace in result.get("traces", ()) or ():
            for op in trace.get("ops", ()) or ():
                family = str(op.get("family") or "")
                mc2 = op.get("mc2") if isinstance(op.get("mc2"), Mapping) else None
                record = op.get("physical")
                if mc2 is not None:
                    status = str(mc2.get("wiring_status") or "")
                    mc2_ops += 1
                elif isinstance(record, Mapping):
                    status = str(dict(record.get("wiring") or {}).get("status") or "")
                else:
                    status = ""
                if not status:
                    status = PHYSICAL_UNMODELED
                wiring[status] = wiring.get(status, 0) + 1
                alias = dict(
                    (record or {}).get("scope_alias") or {}
                ) if isinstance(record, Mapping) else {}
                if alias:
                    key = f"{alias.get('from')}->{alias.get('to')}"
                    scope_aliases[key] = scope_aliases.get(key, 0) + 1
                estimated = float(op.get("estimated_ms") or 0.0)
                modeled_ms += estimated
                applied = bool(isinstance(record, Mapping) and record.get("applied"))
                if applied:
                    applied_ms += estimated
                if family in LARGE_OPERATOR_FAMILIES:
                    large[status] = large.get(status, 0) + 1
                    by_family[family] = by_family.get(family, 0) + 1
                    large_ms += estimated
                    if applied:
                        large_applied_ms += estimated
    return {
        "wiring_counts": dict(sorted(wiring.items())),
        "large_operator_wiring_counts": dict(sorted(large.items())),
        "large_operator_family_counts": dict(sorted(by_family.items())),
        "mc2_op_count": mc2_ops,
        # Ops the artifact answered through the declared timing-scope family (a
        # ``device_event_warm`` case reaching a ``device_event_warm_operator``
        # row).  Reported per pair: a scope alias is a protocol sibling, not an
        # exact measurement, and a reader has to see how much of the prediction
        # rests on it.
        "scope_alias_counts": dict(sorted(scope_aliases.items())),
        # The share the whole-network prediction rests on: how much of the modelled
        # duration came from a fitted row rather than the analytic formula.
        "modeled_duration_ms": modeled_ms,
        "physical_applied_duration_ms": applied_ms,
        "physical_applied_duration_share": (
            applied_ms / modeled_ms if modeled_ms else 0.0
        ),
        "large_operator_duration_ms": large_ms,
        "large_operator_applied_duration_share": (
            large_applied_ms / large_ms if large_ms else 0.0
        ),
    }




def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _calibration_protocol(case: CaseSpec) -> dict[str, Any] | None:
    import json as _json

    metadata = dict(case.runtime.metadata or ())
    raw = metadata.get("calibration_protocol") or metadata.get("measurement_protocol")
    if raw:
        try:
            parsed = _json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, _json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    # A partial runtime graph/eager mode is not a measured physical protocol.
    # Exact calibration requires the caller to declare the full protocol.
    return None


def _fusion_core_kwargs(case: CaseSpec) -> dict[str, Any]:
    die = case.machine.dies[0]
    aic = int(getattr(die, "aic_count", 0) or 0)
    aiv = int(getattr(die, "aiv_count", 0) or 0)
    if aic <= 0 or aiv <= 0:
        return {}
    return {
        "core_count": aic,
        "per_core_rates": {
            "cube": case.machine.peak_flops_for(case.model.compute_dtype) / aic,
            "vector": die.vector_ops_per_s / aiv,
            "mte2": die.mte_bw_bytes_per_s / 1000.0,
            "mte3": die.mte_bw_bytes_per_s / 1000.0,
            "hbm": die.hbm_bw_bytes_per_s / 1000.0,
        },
        "core_base": 0,
        "worker_count": max(1, int(case.runtime.fusion_worker_count)),
    }




class _ReplicaRunner:
    def __init__(
        self, case: CaseSpec, states: Sequence[RequestState], replica_index: int,
        memory: MemoryPlan, cost_backend: Any = None,
        replica_link: Mapping[str, Any] | None = None,
    ) -> None:
        self.case = case
        self.states = list(states)
        for state in self.states:
            state.configure_draft(case.runtime, case.model)
        self.replica_index = replica_index
        self.memory = memory
        self.cost_backend = cost_backend
        self.now = 0.0
        self.iteration = 0
        self.allocated_blocks = sum(state.kv_blocks for state in self.states)
        self.traces: list[dict[str, Any]] = []
        self._last_draft_next: tuple[str, ...] = ()
        # A collective runs on a physical fabric.  It is derived from the
        # machine's declared transfer channels for this replica's rank group --
        # not guessed from a link name -- so a measured comm curve is looked up
        # for the fabric the traffic actually uses.  An unresolvable topology
        # stays empty and the cost model reports the comm calibration as
        # explicitly unmodeled instead of borrowing another fabric's curve.
        self.communication_placement = communication_placement_for_machine(
            case.machine, link=(replica_link or {}).get("link"),
            ranks=(replica_link or {}).get("ranks"),
        )
        self.timing_scope = str(
            getattr(case, "timing_scope", "") or DEFAULT_TIMING_SCOPE
        )
        self.physical_calibration = getattr(case, "physical_calibration", None)
        self.cost_model = PredictiveCostModel(
            case.machine,
            calibration=CalibrationStore.from_mapping(case.calibration),
            compute_dtype=case.model.compute_dtype,
            weight_dtype=case.model.weight_dtype,
            activation_dtype=case.model.activation_dtype,
            quantization=case.model.quantization,
            timing_scope=self.timing_scope,
            communication_calibration=case.communication_calibration,
            communication_placement=self.communication_placement,
            communication_ranks=case.runtime.parallel.tp,
            communication_backend=case.runtime.communication_backend,
            communication_observed_backend=case.runtime.communication_observed_backend,
            communication_route=case.runtime.communication_route,
            communication_timing_completion=case.runtime.communication_timing_completion,
            communication_requested_aiv_cores=(
                case.runtime.communication_requested_aiv_cores
            ),
            communication_observed_aiv_cores=(
                case.runtime.communication_observed_aiv_cores
            ),
            communication_algorithm_regime=case.runtime.communication_algorithm_regime,
            communication_topology_class=case.runtime.communication_topology_class,
            communication_in_place=case.runtime.communication_in_place,
            cv_calibration=case.cv_calibration,
            cv_query=(
                tuple(case.runtime.cv_query)
                if case.runtime.cv_query is not None else None
            ),
            curve_artifact=case.curve_artifact,
            curve_allow_primitive_transfer=(
                case.runtime.curve_allow_primitive_transfer
            ),
            graph_mode=case.runtime.graph_mode,
            # Graph replay amortizes the host submit across a bucket.  The ratio is
            # a measured protocol property, so it is declared by the case
            # (``runtime.metadata.launch_amortization``) instead of invented here;
            # eager per-op submission stays 1.0.
            launch_amortization=float(
                dict(case.runtime.metadata).get("launch_amortization", 1.0) or 1.0
            ),
            calibration_protocol=_calibration_protocol(case),
            physical_calibration=self.physical_calibration,
        )
        link_scale = float((replica_link or {}).get("capacity_scale", 1.0))
        link = (replica_link or {}).get("link")
        ranks = (replica_link or {}).get("ranks")
        self.resources, self.stage_map = build_resource_specs(
            case.machine, case.runtime.parallel.pp,
            link_capacity_scale=link_scale, compute_dtype=case.model.compute_dtype,
            link=link, ranks=ranks,
        )
        # One persistent timeline per replica.  A step runs only until its
        # token-critical tasks complete; background preparation may remain
        # active and be drained by a later step.
        self.engine = ResourceEngine(self.resources)
        self.backend_engine = ResourceEngine(
            {"simulated_time": ResourceSpec("simulated_time", 1.0, "ms")}
        )
        self._prepared: set[str] = set()
        self._prepare_ids: list[str] = []
        owned = {state.request.request_id for state in self.states}
        startup = tuple(
            item for item in case.runtime.prepare
            if item.known_at != "previous_forward"
            and (item.request_id is None or item.request_id in owned)
        )
        self._forward_prepare_ids: list[str] = []
        self._sample_prepare_ids: list[str] = []
        if startup:
            self._prepare_ids.extend(add_prepare_tasks(
                self.engine, startup, self.stage_map, self.states,
                now_ms=0.0, prefix=f"dp{self.replica_index}", added=self._prepared,
            ))
            self._forward_prepare_ids.extend(
                task for task, spec in zip(self._prepare_ids[-len(startup):], startup)
                if spec.consumer == "forward"
            )
            self._sample_prepare_ids.extend(
                task for task, spec in zip(self._prepare_ids[-len(startup):], startup)
                if spec.consumer == "sample"
            )
        self._max_iterations = MAX_ITERATIONS

    def _waiting(self) -> list[RequestState]:
        return [state for state in self.states if not state.finished and not state.started]

    def _running(self) -> list[RequestState]:
        return [state for state in self.states if not state.finished and state.started]

    def _free_blocks(self) -> int:
        return max(0, self.memory.kv_blocks - self.allocated_blocks)

    def _can_allocate(self, state: RequestState, q_len: int) -> bool:
        target_tokens = state.computed + q_len
        target_blocks = _ceil_div(max(target_tokens, state.kv_blocks * self.case.runtime.kv_block_size), self.case.runtime.kv_block_size)
        return target_blocks - state.kv_blocks <= self._free_blocks()

    def _allocate(self, state: RequestState, q_len: int) -> None:
        target_tokens = state.computed + q_len
        target_blocks = _ceil_div(max(target_tokens, state.kv_blocks * self.case.runtime.kv_block_size), self.case.runtime.kv_block_size)
        extra = max(0, target_blocks - state.kv_blocks)
        if extra > self._free_blocks():
            raise A3InputError("kv_allocation_exceeded_free_blocks")
        state.kv_blocks += extra
        self.allocated_blocks += extra

    def _release(self, state: RequestState) -> None:
        self.allocated_blocks = max(0, self.allocated_blocks - state.kv_blocks)
        state.kv_blocks = 0

    def _prompt_chunk(self, state: RequestState, budget: int, input_budget: int | None) -> int:
        remaining = state.prompt - state.computed
        if remaining <= 0:
            return 0
        if not self.case.runtime.enable_chunked_prefill:
            if remaining > budget:
                return 0
            if input_budget is not None and remaining > input_budget:
                return 0
            return remaining
        q = min(remaining, max(0, budget))
        if input_budget is not None:
            q = min(q, max(0, input_budget))
        threshold = self.case.runtime.long_prefill_token_threshold
        if threshold > 0:
            q = min(q, threshold)
        return max(0, q)

    def _settle_draft_kv(self, state: RequestState) -> None:
        block = max(1, int(self.case.runtime.kv_block_size))
        desired = _ceil_div(max(0, state.computed), block)
        if state.kv_blocks > desired:
            drop = state.kv_blocks - desired
            state.kv_blocks = desired
            self.allocated_blocks = max(0, self.allocated_blocks - drop)

    def _commit_views(
        self, selected: Sequence[RequestState], views: Sequence[RequestView],
        end_ms: float,
    ) -> None:
        for state, view in zip(selected, views):
            state.commit_view(view, end_ms)
            if bool(getattr(view, "draft_verify", False)):
                self._settle_draft_kv(state)

    def _schedule(self) -> tuple[list[RequestView], list[RequestState]]:
        runtime = self.case.runtime
        budget = runtime.max_num_batched_tokens
        input_budget = runtime.max_num_input_tokens
        views: list[RequestView] = []
        selected: list[RequestState] = []
        # Running requests first, preserving their admission order.
        for state in self._running():
            if budget <= 0:
                break
            if state.prompt_done:
                q = 1
                remaining_output = max(0, state.output - state.generated)
                scheduled_proposal = 0
                if state.draft_mode and remaining_output > 1:
                    scheduled_proposal = min(
                        state.draft_proposal_tokens,
                        max(0, int(budget) - q),
                        max(0, remaining_output - 1),
                    )
                draft_verify = scheduled_proposal > 0
                proposal = scheduled_proposal
                accepted = (
                    min(proposal, state.peek_draft_accept())
                    if draft_verify else 0
                )
                q_alloc = max(q, proposal + 1) if draft_verify else q
                budget_charge = q + proposal if draft_verify else q
                if not self._can_allocate(state, q_alloc):
                    state.unserved_reason = "kv_capacity_blocked_decode"
                    continue
                views.append(RequestView(
                    state.request.request_id, q, state.computed, True, "decode",
                    state.request.prefix_tokens, state.request.priority,
                    semantic_phase="decode", execution_mode="decode",
                    media_encoded=state.media_encoded,
                    draft_verify=draft_verify,
                    draft_proposal_tokens=proposal,
                    draft_accepted_tokens=accepted,
                ))
                selected.append(state)
                self._allocate(state, q_alloc)
                budget -= budget_charge
            else:
                q = self._prompt_chunk(state, budget, input_budget)
                if q <= 0:
                    continue
                if not self._can_allocate(state, q):
                    state.unserved_reason = "kv_capacity_blocked_prefill"
                    continue
                completed = state.computed + q >= state.prompt
                execution = "decode" if q == 1 else "prefill"
                views.append(RequestView(
                    state.request.request_id, q, state.computed, completed, execution,
                    state.request.prefix_tokens, state.request.priority,
                    semantic_phase="prefill", execution_mode=execution,
                    media_encoded=state.media_encoded, draft_verify=False,
                ))
                selected.append(state)
                self._allocate(state, q)
                budget -= q
                if input_budget is not None:
                    input_budget -= q
            if len(views) >= runtime.max_num_seqs:
                return views, selected
        waiting = sorted(
            (state for state in self._waiting() if state.request.arrival_ms <= self.now),
            key=lambda state: (-state.request.priority, state.request.arrival_ms, state.request.request_id),
        )
        for state in waiting:
            if budget <= 0 or len(views) >= runtime.max_num_seqs:
                break
            remaining = state.prompt - state.computed
            draft_verify = False
            proposal = 0
            accepted = 0
            if remaining <= 0:
                q = 1
                semantic = "decode"
                execution = "decode"
                completed = True
                remaining_output = max(0, state.output - state.generated)
                scheduled_proposal = 0
                if state.draft_mode and remaining_output > 1:
                    scheduled_proposal = min(
                        state.draft_proposal_tokens,
                        max(0, int(budget) - q),
                        max(0, remaining_output - 1),
                    )
                draft_verify = scheduled_proposal > 0
                proposal = scheduled_proposal
                accepted = (
                    min(proposal, state.peek_draft_accept())
                    if draft_verify else 0
                )
            else:
                q = self._prompt_chunk(state, budget, input_budget)
                if q <= 0:
                    continue
                semantic = "prefill"
                execution = "decode" if q == 1 else "prefill"
                completed = state.computed + q >= state.prompt
            q_alloc = max(q, proposal + 1) if draft_verify else q
            budget_charge = q + proposal if draft_verify else q
            if not self._can_allocate(state, q_alloc):
                state.unserved_reason = "kv_capacity_blocked_admission"
                continue
            views.append(RequestView(
                state.request.request_id, q, state.computed, completed, execution,
                state.request.prefix_tokens, state.request.priority,
                semantic_phase=semantic, execution_mode=execution,
                media_encoded=state.media_encoded,
                draft_verify=draft_verify,
                draft_proposal_tokens=proposal,
                draft_accepted_tokens=accepted,
            ))
            selected.append(state)
            state.started = True
            self._allocate(state, q_alloc)
            budget -= budget_charge
            if input_budget is not None and remaining > 0:
                input_budget -= q
        return views, selected

    def _ops_trace(self, work: Any, costed: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        cost_by_id = {item.op.op_id: item for item in (costed or ())}
        result = []
        for op in work.ops:
            cost = cost_by_id.get(op.op_id)
            result.append({
                "op_id": op.op_id, "family": op.family, "stage": op.stage,
                "layer": op.layer, "recipe": op.recipe, "phase": op.phase,
                "deps": list(op.deps),
                "semantic_shape": list(op.semantic_shape),
                "local_shape": list(op.local_shape),
                "padded_shape": list(op.padded_shape),
                "launch_shape": list(op.launch_shape),
                "flops": op.flops, "hbm_bytes": op.hbm_bytes,
                "link_bytes": op.link_bytes,
                "dtype": op.dtype,
                "layout": op.layout,
                "shape_features": self.cost_model.shape_features(op),
                "features": self.cost_model.calibration_features(),
                "competition": {},
                "analytic_ms": cost.analytic_ms if cost is not None else None,
                "estimated_ms": cost.estimated_ms if cost is not None else None,
                # The additive decomposition of ``estimated_ms``: the dispatch
                # intercept and the kernel service it precedes.  A whole-network
                # reader needs both to separate launch-bound from
                # bandwidth-bound iterations.
                "launch_ms": cost.launch_ms if cost is not None else None,
                "service_ms": cost.service_ms if cost is not None else None,
                "calibration_status": cost.calibration.status if cost is not None else "injected_backend",
                "calibration_evidence": (
                    dict(cost.calibration.evidence)
                    if cost is not None and cost.calibration.evidence else None
                ),
                "curve_result": (
                    dict(cost.curve_result)
                    if cost is not None and cost.curve_result else None
                ),
                "physical": (
                    dict(cost.physical)
                    if cost is not None and cost.physical else None
                ),
                # MC2 chunk provenance: dispatch/combine bytes and lane, the
                # grouped-matmul compute term, both overlap bounds and whether
                # the overlap is measured.  ``None`` for every non-MC2 op.
                "mc2": (
                    dict(cost.mc2) if cost is not None and cost.mc2 else None
                ),
                # A grouped matmul pays one dispatch intercept per tensor-list
                # launch (128-expert cap); the count is reported, not folded in.
                "launch_detail": (
                    dict(cost.launch_detail)
                    if cost is not None and cost.launch_detail else None
                ),
            })
        return result

    def schedule_next(self) -> dict[str, Any] | None:
        """Prepare one iteration without executing or mutating token state."""
        if all(state.finished for state in self.states):
            return None
        if self.iteration >= self._max_iterations:
            for state in self.states:
                if not state.finished:
                    state.unserved_reason = "iteration_limit"
            return None
        running = self._running()
        waiting_now = [state for state in self._waiting() if state.request.arrival_ms <= self.now]
        if not running and not waiting_now:
            future = [state.request.arrival_ms for state in self._waiting()]
            if not future:
                return None
            self.now = min(future)
        views, selected = self._schedule()
        if not views:
            unfinished = [state for state in self.states if not state.finished]
            future = [state.request.arrival_ms for state in unfinished
                      if state.request.arrival_ms > self.now and not state.started]
            if not self._running() and future:
                return {"wait_until": min(future)}
            for state in unfinished:
                state.unserved_reason = state.unserved_reason or "scheduler_no_progress"
            return None
        plan = IterationPlan(tuple(views), iteration=self.iteration)
        step_runtime = runtime_with_draft_next_counts(
            self.case.runtime, tuple(views), self.states,
        )
        work = derive_iteration_work(
            plan, self.case.model, step_runtime, self.case.machine,
            self.case.runtime.parallel,
        )
        from .a3_fusion import apply_fusion

        fusion = apply_fusion(
            work, enabled=self.case.runtime.fusion,
            scratch_capacity_bytes=(
                None if fusion_pool_capacities(self.case.machine.dies[0])
                else (
                    self.case.machine.dies[0].l2_bytes
                    if self.case.machine.dies[0].l2_bytes > 0 else 0
                )
            ),
            pool_capacities=fusion_pool_capacities(self.case.machine.dies[0]),
            **_fusion_core_kwargs(self.case),
            cube_flops_per_s=self.case.machine.peak_flops_for(
                self.case.model.compute_dtype
            ),
            vector_ops_per_s=self.case.machine.dies[0].vector_ops_per_s,
            model_features={
                "weight_dtype": self.case.model.weight_dtype,
                "quantization": self.case.model.quantization,
            },
        )
        work = fusion.work
        costed = self.cost_model.cost_iteration(work) if self.cost_backend is None else None
        return {
            "views": tuple(views), "selected": tuple(selected),
            "work": work, "fusion": fusion, "costed": costed,
            "iteration": self.iteration,
            "start_ms": self.now,
            "prefix": f"dp{self.replica_index}.it{self.iteration}",
        }

    def complete_step(
        self, step: Mapping[str, Any], end_ms: float,
        *, events: Sequence[Mapping[str, Any]] = (),
        background_remaining: Sequence[str] = (),
        deadline_slack: Mapping[str, float] | None = None,
    ) -> None:
        views = step["views"]
        selected = step["selected"]
        end = max(float(end_ms), self.now)
        self._commit_views(selected, views, end)
        for state in selected:
            if state.finished:
                self._release(state)
                from .a3_async import release_prepare_residency
                release_prepare_residency(
                    self.engine,
                    request_id=state.request.request_id,
                    now_ms=end,
                )
        work = step["work"]
        self.traces.append({
            "iteration": step["iteration"], "replica": self.replica_index,
            "start_ms": step["start_ms"], "end_ms": end,
            "duration_ms": end - step["start_ms"],
            "valid_tokens": work.valid_tokens, "padded_tokens": work.padded_tokens,
            "scheduled_tokens": sum(
                view.q_len + (view.draft_proposal_tokens if view.draft_verify else 0)
                for view in views
            ),
            "speculative_query_tokens": sum(
                view.draft_proposal_tokens if view.draft_verify else 0
                for view in views
            ),
            "padding_tokens": work.padded_tokens - work.valid_tokens,
            "graph_bucket": work.graph_bucket, "mode": work.mode,
            "recipe": work.recipe,
            "recipe_guards": [dict(item) for item in work.recipe_guards],
            "requests": [view.to_dict() for view in views],
            "routing": work.routing.to_dict() if work.routing is not None else None,
            "fusion": step["fusion"].to_dict() if self.case.runtime.fusion else None,
            "op_count": len(work.ops),
            "ops": self._ops_trace(work, step["costed"]),
            "events": list(events),
            "background_remaining": list(background_remaining),
            "deadline_slack_ms": dict(deadline_slack or {}),
        })
        self.now = end
        self.iteration += 1

    def run(self) -> dict[str, Any]:
        while True:
            if all(state.finished for state in self.states):
                break
            if self.iteration >= self._max_iterations:
                for state in self.states:
                    if not state.finished:
                        state.unserved_reason = "iteration_limit"
                break
            running = self._running()
            waiting_now = [state for state in self._waiting() if state.request.arrival_ms <= self.now]
            if not running and not waiting_now:
                future = [state.request.arrival_ms for state in self._waiting()]
                if not future:
                    break
                self.now = min(future)
            views, selected = self._schedule()
            if not views:
                unfinished = [state for state in self.states if not state.finished]
                future = [state.request.arrival_ms for state in unfinished
                          if state.request.arrival_ms > self.now and not state.started]
                if not self._running() and future:
                    self.now = min(future)
                    continue
                for state in unfinished:
                    state.unserved_reason = state.unserved_reason or "scheduler_no_progress"
                break
            previous = tuple(
                item for item in self.case.runtime.prepare
                if item.known_at == "previous_forward"
            )
            if previous and self.iteration > 0:
                new_ids = add_prepare_tasks(
                    self.engine, previous, self.stage_map, self.states,
                    now_ms=self.now, prefix=f"dp{self.replica_index}",
                    added=self._prepared,
                )
                self._prepare_ids.extend(new_ids)
                self._forward_prepare_ids.extend(
                    task for task, spec in zip(new_ids, previous)
                    if spec.consumer == "forward"
                )
            plan = IterationPlan(tuple(views), iteration=self.iteration)
            step_runtime = runtime_with_draft_next_counts(
                self.case.runtime, tuple(views), self.states,
            )
            work = derive_iteration_work(
                plan, self.case.model, step_runtime, self.case.machine,
                self.case.runtime.parallel,
            )
            from .a3_fusion import apply_fusion

            fusion = apply_fusion(
                work, enabled=self.case.runtime.fusion,
                scratch_capacity_bytes=(
                    self.case.machine.dies[0].l2_bytes
                    if self.case.machine.dies[0].l2_bytes > 0 else 0
                ),
                cube_flops_per_s=self.case.machine.peak_flops_for(
                    self.case.model.compute_dtype
                ),
                vector_ops_per_s=self.case.machine.dies[0].vector_ops_per_s,
                model_features={
                    "weight_dtype": self.case.model.weight_dtype,
                    "quantization": self.case.model.quantization,
                },
            )
            work = fusion.work
            prefix = f"dp{self.replica_index}.it{self.iteration}"
            costed: list[Any] | None = None
            if self.cost_backend is not None:
                duration = _backend_duration(self.cost_backend, work, self.case)
                task_id = f"{prefix}:backend"
                self.backend_engine.add_task(
                    task_id, {"simulated_time": duration},
                    dependencies=tuple(self._forward_prepare_ids),
                    release_ms=self.now,
                )
                result = self.backend_engine.run(
                    start_ms=self.now, until=(task_id,), compact=True,
                )
                targets = (task_id,)
            else:
                costed = self.cost_model.cost_iteration(work)
                draft_deps = (
                    {"draft.prev_proposals_ready": self._last_draft_next}
                    if self._last_draft_next else None
                )
                id_map = add_costed_ops(
                    self.engine, costed, self.stage_map, prefix=prefix,
                    start_ms=self.now,
                    external_deps=tuple(self._forward_prepare_ids),
                    family_external_deps={
                        "sampling": tuple(self._sample_prepare_ids),
                    } if self._sample_prepare_ids else None,
                    op_external_deps=draft_deps,
                    core_counts={
                        "aic": self.case.machine.dies[0].aic_count,
                        "aiv": self.case.machine.dies[0].aiv_count,
                    },
                )
                self._last_draft_next = tuple(
                    id_map[item.op.op_id] for item in costed
                    if item.op.op_id.startswith("draft.next.")
                    and item.op.op_id in id_map
                )
                targets = tuple(id_map.values())
                result = self.engine.run(
                    start_ms=self.now, until=targets, compact=True,
                )
            target_times = [
                result["completed_at_ms"][task_id] for task_id in targets
                if task_id in result["completed_at_ms"]
            ]
            end = max(target_times, default=self.now)
            if end <= self.now:
                end = self.now + 1e-9
            # Resource completion gates token visibility.  A partial prefill
            # chunk has samples=False and therefore cannot advance TTFT.
            self._commit_views(selected, views, end)
            for state in selected:
                if state.finished:
                    self._release(state)
            self.traces.append({
                "iteration": self.iteration,
                "replica": self.replica_index,
                "start_ms": self.now,
                "end_ms": end,
                "duration_ms": end - self.now,
                "valid_tokens": work.valid_tokens,
                "padded_tokens": work.padded_tokens,
                "scheduled_tokens": sum(
                    view.q_len + (
                        view.draft_proposal_tokens if view.draft_verify else 0
                    )
                    for view in views
                ),
                "speculative_query_tokens": sum(
                    view.draft_proposal_tokens if view.draft_verify else 0
                    for view in views
                ),
                "padding_tokens": work.padded_tokens - work.valid_tokens,
                "graph_bucket": work.graph_bucket,
                "mode": work.mode,
                "recipe": work.recipe,
                "recipe_guards": [dict(item) for item in work.recipe_guards],
                "requests": [view.to_dict() for view in views],
                "routing": work.routing.to_dict() if work.routing is not None else None,
                "fusion": fusion.to_dict() if self.case.runtime.fusion else None,
                "op_count": len(work.ops),
                "ops": self._ops_trace(work, costed),
                "events": result["trace"],
                "background_remaining": result.get("remaining_tasks", []),
                "deadline_slack_ms": result.get("deadline_slack_ms", {}),
            })
            self.now = end
            self.iteration += 1
            for state in self.states:
                if not state.finished and not state.started and state.request.arrival_ms <= self.now:
                    # Waiting requests become visible only on the next scheduling
                    # turn; this is intentional and matches a virtual ingress.
                    pass
        served = [state for state in self.states if state.finished]
        unserved = [state for state in self.states if not state.finished]
        drain = self.engine.run(start_ms=self.now)
        token_end = max(
            (state.finish_ms or self.now for state in self.states), default=self.now,
        )
        return {
            "replica": self.replica_index,
            "makespan_ms": max((state.finish_ms or self.now for state in self.states), default=0.0),
            "iterations": self.iteration,
            "served": len(served),
            "unserved": len(unserved),
            "requests": [state.to_dict() for state in self.states],
            "traces": self.traces,
            "prepare": prepare_summary(
                drain, token_end, prep_task_ids=tuple(self._prepare_ids),
                residency=prepare_residency_report(self.engine),
            ),
            "unserved_reasons": {
                state.request.request_id: state.unserved_reason
                for state in unserved
            },
        }



def _oom_guidance(case: CaseSpec, memory: MemoryPlan) -> dict[str, Any]:
    parallel = case.runtime.parallel
    model = case.model
    world = len(case.machine.dies)
    candidates = []
    tps = sorted({1, 2, 4, 8, parallel.tp, model.num_heads})
    tps = [tp for tp in tps if 1 <= tp <= world and model.num_heads % tp == 0]
    pps = sorted({1, 2, 4, 8, parallel.pp, model.layers})
    pps = [pp for pp in pps if 1 <= pp <= world and model.layers % pp == 0]
    for tp in tps:
        for pp in pps:
            if tp * pp > world:
                continue
            max_dp = max(1, world // (tp * pp))
            for dp in sorted({1, max_dp, min(2, max_dp)}):
                if dp * tp * pp > world:
                    continue
                eps = [1]
                if model.num_experts:
                    domain = dp * tp
                    eps = sorted({1, domain, min(2, domain), min(4, domain),
                                  min(8, domain), model.num_experts})
                    eps = [ep for ep in eps if 1 <= ep <= domain and domain % ep == 0
                           and model.num_experts % ep == 0]
                for ep in eps:
                    candidate = ParallelSpec(dp=dp, tp=tp, ep=ep, pp=pp)
                    try:
                        candidate.validate(model)
                        if candidate.world_size > world:
                            continue
                        runtime = CaseSpec(
                            machine=case.machine, model=model, workload=case.workload,
                            runtime=__import__("dataclasses").replace(
                                case.runtime, parallel=candidate,
                            ),
                        ).runtime
                        plan = memory_plan(model, runtime, case.machine)
                        if plan.status == "fits":
                            candidates.append({
                                "dp": dp, "tp": tp, "ep": ep, "pp": pp,
                                "world_size": candidate.world_size,
                                "weights_bytes": plan.weights_bytes,
                                "kv_capacity_tokens": plan.kv_capacity_tokens,
                            })
                    except A3InputError:
                        continue
    if candidates:
        candidates.sort(key=lambda item: (item["world_size"], item["weights_bytes"]))
        return {
            "status": "oom",
            "reason": "current_parallelism_does_not_fit",
            "suggested_parallel": candidates[:8],
            "requires_more_devices": False,
        }
    per_die = case.machine.dies[0]
    usable = max(1.0, per_die.hbm_bytes * case.runtime.gpu_memory_utilization * 0.75)
    required = max(1, math.ceil(model.weight_bytes(ParallelSpec()) / usable))
    return {
        "status": "oom",
        "reason": "machine_capacity_insufficient_for_model_at_any_legal_parallelism",
        "required_world_size_estimate": required,
        "requires_more_devices": True,
        "suggested_parallel": None,
        "note": "increase declared device count / EP domain; the architecture was not downsized",
    }



def predict(
    case: CaseSpec | Mapping[str, Any],
    *,
    calibration: Any = None,
    cost_backend: Any = None,
    physical_calibration: Any = None,
    timing_scope: str | None = None,
) -> dict[str, Any]:
    """Run the predictive path for one declared case.

    ``physical_calibration`` is a loaded
    :class:`mindie_sim.a3_physical_calibration.PhysicalCalibration` (or
    ``None``).  It must belong to this case's machine: a foreign hardware key is
    refused rather than retargeted.  ``timing_scope`` selects the measurement
    protocol the analytic *and* physical layers price against; it defaults to
    the case's declared scope, then to ``device_event_warm``.
    """
    if isinstance(case, Mapping):
        case = CaseSpec.from_mapping(case)
    if not isinstance(case, CaseSpec):
        raise A3InputError("case_invalid")
    case = with_media_runtime(case)
    if physical_calibration is None:
        physical_calibration = getattr(case, "physical_calibration", None)
    _refuse_foreign_physical_calibration(case, physical_calibration)
    effective_scope = str(
        timing_scope
        or getattr(case, "timing_scope", "")
        or dict(case.runtime.metadata or ()).get("timing_scope")
        or DEFAULT_TIMING_SCOPE
    )
    case = _predictive_case(
        case, physical_calibration=physical_calibration,
        timing_scope=effective_scope,
    )
    memory = memory_plan(case.model, case.runtime, case.machine)
    if memory.status != "fits":
        return {
                "status": "infeasible",
            "prediction_kind": "predictive_operational",
            "metrics": None,
            "reason": "memory_infeasible",
            "memory": memory.to_dict(),
            "oom_guidance": _oom_guidance(case, memory),
            "requests": [],
            "traces": [],
            "timing_scope": effective_scope,
            "physical_calibration": _physical_calibration_summary(
                physical_calibration, timing_scope=effective_scope,
                replica_results=(),
            ),
            "plan": case.to_plan(),
        }
    if calibration is not None:
        case = replace(case, calibration=calibration)
    dp = case.runtime.parallel.dp
    states_by_replica: list[list[RequestState]] = [[] for _ in range(dp)]
    for index, request in enumerate(case.workload.requests):
        replica = index % dp
        states_by_replica[replica].append(RequestState(request))
    parallel = case.runtime.parallel
    if parallel.ep_spans_dp and parallel.dp > 1:
        world_ranks = tuple(range(parallel.world_size))
        link = select_link_for_ranks(case.machine, world_ranks)
        replica_links = [
            {
                "replica": index, "ranks": world_ranks, "link": link,
                "link_name": link.name if link else None,
                "share_count": parallel.dp,
                "capacity_scale": 1.0 / parallel.dp,
            }
            for index in range(parallel.dp)
        ]
    else:
        active_replicas = [
            index for index, states in enumerate(states_by_replica) if states
        ]
        replica_links = plan_replica_links(
            case.machine, parallel, active_replicas=active_replicas,
        )
    link_by_replica = {item["replica"]: item for item in replica_links}
    active_indices = [index for index, states in enumerate(states_by_replica) if states]
    try:
        replica_results = predict_global_replicas(
            case, states_by_replica, memory, replica_links,
            cost_backend=cost_backend,
        )
    except A3InputError as exc:
        if not str(exc).startswith("prepare_capacity_"):
            raise
        return {
            "status": "infeasible",
            "prediction_kind": "predictive_operational",
            "reason": str(exc),
            "metrics": None,
            "memory": memory.to_dict(),
            "requests": [],
            "traces": [],
            "timing_scope": effective_scope,
            "physical_calibration": _physical_calibration_summary(
                physical_calibration, timing_scope=effective_scope,
                replica_results=(),
            ),
            "plan": case.to_plan(),
        }
    arbitration_rounds: list[dict[str, Any]] = []
    scales: dict[int, float] = {index: 1.0 for index in active_indices}
    if not replica_results:
        raise A3InputError("workload_empty")
    if not replica_results:
        raise A3InputError("workload_empty")
    merged = _merge_replicas(replica_results)
    status = "modeled" if merged["unserved_requests"] == 0 else "partial"
    goodput = _slo_goodput(merged["requests"], case.slo)
    capacity_tokens = sum(request.prompt_tokens + request.output_tokens
                          for request in case.workload.requests)
    return {
        "status": status,
        "prediction_kind": "predictive_operational",
        "notes": [
            "predictive answers expected performance for this machine/workload/config",
            "one in-flight iteration per DP replica; asynchronous multi-batch overlap is not claimed",
            "DP replicas have disjoint local compute; a shared physical link is arbitrated over live transfers",
            "all durations are analytic/calibrated simulator output, not measured NPU timings",
        ],
        "metrics": merged["metrics"],
        "slo_goodput": goodput,
        "served_requests": merged["served_requests"],
        "unserved_requests": merged["unserved_requests"],
        "output_tokens": merged["output_tokens"],
        "workspace_tokens": capacity_tokens,
        "memory": memory.to_dict(),
        "iterations": sum(result["iterations"] for result in replica_results),
        "replicas": [
            {key: value for key, value in result.items() if key not in {"traces", "requests"}}
            for result in replica_results
        ],
        "resource_topology": {
            "ep_spans_dp": case.runtime.parallel.ep_spans_dp,
            "replica_links": [
                {
                    "replica": item["replica"], "ranks": list(item["ranks"]),
                    "link": item["link_name"], "share_count": item["share_count"],
                    "capacity_scale": item["capacity_scale"],
                }
                for item in replica_links
            ],
            "note": "routes are chosen by rank membership; shared links use one capacity budget",
        },
        "prepare": {
            "replicas": [
                {"replica": result["replica"], **result.get("prepare", {})}
                for result in replica_results
            ],
            "model_derived_gaps": [
                gap for result in replica_results
                for gap in result.get("model_prepare_gaps", ())
            ],
            "note": "background preparation/drain is reported separately from token-ready metrics",
        },
        "requests": merged["requests"],
        "traces": merged["traces"],
        "timing_scope": effective_scope,
        "physical_calibration": _physical_calibration_summary(
            physical_calibration, timing_scope=effective_scope,
            replica_results=replica_results,
        ),
        "link_arbitration": {
            "rounds": arbitration_rounds,
            "final_scales": {
                str(index): 1.0 for index in active_indices
            },
            "dynamic_reclaim": True,
            "static_active_replica_share": False,
            "one_global_event_timeline": True,
        },
        "plan": case.to_plan(),
    }


run_predictive = predict


__all__ = [
    "MAX_ITERATIONS", "RequestState", "TokenLinearCostBackend", "predict",
    "run_predictive",
]
