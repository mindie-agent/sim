"""Serial, fork/join, fusion and state composition of compiled operators.

[Role]
  Compose guarded operator cost functions into subgraph costs.  A single-stream
  post-fusion kernel chain is an exact sum when no external wait is declared.
  Overlapping branches run through the shared ``Engine`` so HBM/Cube/Vector
  contention changes the makespan; taking the maximum of isolated scalars is
  deliberately not used.  Native fusion lowers semantic groups to physical
  kernels and removes only declared internal materialisations.

[Invariants]
  * ``serial.total_ms == sum(operator durations) + declared readiness gaps``.
    A late input adds its real positive gap, never a fixed overlap discount.
  * Fork/join contention is executed on the one Engine: shared pipes are not
    duplicated and disjoint pipes may overlap.
  * Fusion keeps every declared external output; only contract-listed
    internal materialisation bytes and repeated startup are removed.
  * Pipeline recurrence keeps finite fill/drain and slot limits; one tile is
    ``load+compute(+store)`` rather than a steady-state ``max``.
  * State effects are applied in order; a consumer before its producer raises.

[Timeouts & Locks]
  Pure finite arithmetic and bounded Engine events.  No waits or locks.

[Side-Effect Policy]
  Returns immutable results; Engine calls are caller-owned virtual execution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.subgraphs import SubgraphInstance
from .compiler import ( CompiledOperator, CompiledSubgraph, OperatorCost, PipeSupply, RequestGeometry,
    SubgraphEvaluation, WorkloadGeometry, _PIPE_UNITS, _apply_operator_state, )
from .expressions import Expr, as_fraction
from .guards import PipeDemand

@dataclass(frozen=True)
class SerialResult:
    costs: tuple[OperatorCost, ...]; total_ms: Fraction; state: Any
    gaps_ms: tuple[Fraction, ...] = (); fused: tuple[str, ...] = ()
    readiness: tuple[tuple[str, Fraction], ...] = ()
    fusion_results: tuple["FusionResult", ...] = ()
    contention: "ContentionEnvelope | None" = None
    @property
    def gap_ms(self) -> Fraction: return sum(self.gaps_ms, Fraction(0))
    def by_name(self, name: str) -> OperatorCost:
        for cost in self.costs:
            if cost.name == name: return cost
        raise ModelingError(f"serial_cost_missing:{name}")
    def to_dict(self) -> dict[str, Any]:
        return { "total_ms": str(self.total_ms), "gap_ms": str(self.gap_ms), "fused": list(self.fused),
            "fusion_results": [item.to_dict() for item in self.fusion_results], "operators": [cost.to_dict() for cost in self.costs], }

@dataclass(frozen=True)
class FusionResult:
    name: str; semantic: tuple[OperatorCost, ...]; fused: OperatorCost
    removed: tuple[tuple[str, Fraction], ...]; kept: tuple[tuple[str, Fraction], ...]
    startup_count: int = 1; isolated_ms: Fraction = Fraction(0)
    @property
    def removed_bytes(self) -> Fraction: return sum((value for _, value in self.removed), Fraction(0))
    def to_dict(self) -> dict[str, Any]:
        return { "name": self.name, "members": [cost.name for cost in self.semantic], "fused": self.fused.to_dict(),
            "removed": {key: str(value) for key, value in self.removed},
            "kept": {key: str(value) for key, value in self.kept}, "startup_count": self.startup_count,
            "isolated_ms": str(self.isolated_ms), }

@dataclass(frozen=True)
class ForkJoinResult:
    prefix_ms: Fraction; join_ms: Fraction; isolated_max_ms: Fraction
    contention_ms: Fraction; engine_ms: Fraction
    branches: tuple[tuple[str, Fraction], ...]; resources: tuple[tuple[str, Fraction], ...] = ()
    @property
    def total_ms(self) -> Fraction: return self.prefix_ms + self.contention_ms + self.join_ms
    @property
    def rejected_isolated_max(self) -> bool: return self.contention_ms != self.isolated_max_ms
    def to_dict(self) -> dict[str, Any]:
        return { "prefix_ms": str(self.prefix_ms), "join_ms": str(self.join_ms),
            "isolated_max_ms": str(self.isolated_max_ms), "contention_ms": str(self.contention_ms),
            "engine_ms": str(self.engine_ms), "total_ms": str(self.total_ms),
            "branches": {name: str(value) for name, value in self.branches},
            "resources": {name: str(value) for name, value in self.resources}, "rejected_isolated_max": self.rejected_isolated_max, }

@dataclass(frozen=True)
class ContentionEnvelope:
    """Shared-capacity envelope for concurrently ready branches.

    ``isolated_max_ms`` is reported only as a rejected counterfactual.  The
    contract is the one-Engine makespan over the union of resource work plus
    the per-resource lower bound ``sum(work_p)/rate_p``.  Two HBM-only streams
    therefore add their bytes; disjoint Cube/Link streams may overlap.
    """
    makespan_ms: Fraction; isolated_max_ms: Fraction; isolated_sum_ms: Fraction
    resource_work: tuple[tuple[str, Fraction], ...]; resource_bounds: tuple[tuple[str, Fraction], ...]
    shared_pipes: tuple[str, ...]; disjoint_pipes: tuple[str, ...]; branches: tuple[tuple[str, Fraction], ...]
    @property
    def additivity_gap_ms(self) -> Fraction: return self.makespan_ms - self.isolated_sum_ms
    @property
    def lower_bound_ms(self) -> Fraction: return max((value for _, value in self.resource_bounds), default=Fraction(0))
    @property
    def rejected_isolated_max(self) -> bool: return self.makespan_ms != self.isolated_max_ms
    def to_dict(self) -> dict[str, Any]:
        return { "makespan_ms": str(self.makespan_ms), "isolated_max_ms": str(self.isolated_max_ms),
            "isolated_sum_ms": str(self.isolated_sum_ms), "lower_bound_ms": str(self.lower_bound_ms),
            "additivity_gap_ms": str(self.additivity_gap_ms), "resource_work": {key: str(value) for key, value in self.resource_work},
            "resource_bounds": {key: str(value) for key, value in self.resource_bounds},
            "shared_pipes": list(self.shared_pipes), "disjoint_pipes": list(self.disjoint_pipes),
            "branches": {name: str(value) for name, value in self.branches}, "rejected_isolated_max": self.rejected_isolated_max, }

@dataclass(frozen=True)
class ClosedForm:
    formula: str; certified: bool = True; reason: str = ""
    def to_dict(self) -> dict[str, Any]:
        return {"formula": self.formula, "certified": self.certified, "reason": self.reason}

@dataclass(frozen=True)
class EngineResult:
    makespan_ms: Fraction; branches: tuple[tuple[str, Fraction], ...]
    resources: tuple[tuple[str, Fraction], ...]
    summary: tuple[tuple[str, Any], ...] = ()
def _resource_units(pipe: str) -> str:
    return _PIPE_UNITS.get(pipe, "work")

def run_engine(branches: Mapping[str, Sequence[Sequence[PipeDemand]]], supply: PipeSupply, env: Mapping[str, Any]) -> EngineResult:
    """Execute symbolic branch stage chains on one shared-resource Engine."""
    from .expressions import evaluate
    bound = {name: tuple(tuple((demand.pipe, evaluate(demand.work, env)) for demand in stage) for stage in stages)
             for name, stages in branches.items()}
    return run_engine_work(bound, supply)

def run_engine_work(branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]], supply: PipeSupply) -> EngineResult:
    """Engine execution from already-bound ``(pipe, work)`` stage chains."""
    for stages in branches.values():
        for stage in stages:
            for pipe, amount in stage:
                val = float(amount)
                if not math.isfinite(val) or val < 0:
                    raise ModelingError(f"engine_work_negative_or_nonfinite:{pipe}:{amount}")
    pipes = sorted({pipe for stages in branches.values() for stage in stages for pipe, amount in stage if amount})
    resources = tuple(Resource(pipe, supply.effective(pipe), _resource_units(pipe), "fair") for pipe in pipes)
    engine = Engine(resources)
    events: dict[str, Any] = {}
    for name, stages in branches.items():
        previous = None
        for index, stage in enumerate(stages):
            work: dict[str, float] = {}
            for pipe, amount in stage:
                if amount > 0:
                    work[pipe] = work.get(pipe, 0.0) + float(amount)
            event = engine.submit(f"{name}:{index}", work=work, after=(previous,) if previous is not None else ())
            previous = event
        if previous is None:
            previous = engine.submit(f"{name}:empty", work={})
        events[name] = previous
    engine.advance()
    makespan = Fraction(str(engine.now))
    completions = tuple(sorted( (name, Fraction(str(event.time if event.time is not None else engine.now)))
        for name, event in events.items()))
    summary = tuple(sorted(engine.summary().items()))
    return EngineResult(makespan, completions, tuple(sorted((key, Fraction(str(value))) for key, value in engine.busy_ms.items())), summary)

def contention_envelope( branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]], supply: PipeSupply, ) -> ContentionEnvelope:
    """Compose concurrently ready branches through one shared-capacity engine."""
    result = run_engine_work(branches, supply)
    branch_totals: dict[str, Fraction] = {}
    work: dict[str, Fraction] = {}
    users: dict[str, set[str]] = {}
    for name, stages in branches.items():
        total = Fraction(0)
        for stage in stages:
            coalesced: dict[str, Fraction] = {}
            for pipe, amount in stage:
                if amount > 0:
                    coalesced[pipe] = coalesced.get(pipe, Fraction(0)) + amount
            stage_time = max((amount / as_fraction(supply.effective(pipe)) for pipe, amount in coalesced.items()), default=Fraction(0))
            total += stage_time
            for pipe, amount in coalesced.items():
                work[pipe] = work.get(pipe, Fraction(0)) + amount
                users.setdefault(pipe, set()).add(name)
        branch_totals[name] = total
    bounds = tuple(sorted((pipe, amount / supply.effective(pipe)) for pipe, amount in work.items()))
    shared = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) > 1))
    disjoint = tuple(sorted(pipe for pipe, owners in users.items() if len(owners) == 1))
    isolated_max = max(branch_totals.values(), default=Fraction(0))
    return ContentionEnvelope( result.makespan_ms, isolated_max, sum(branch_totals.values(), Fraction(0)),
        tuple(sorted(work.items())), bounds, shared, disjoint, tuple(sorted(branch_totals.items())))

def pipeline_makespan(stage_ms: Sequence[Fraction | float], tiles: int, slots: int) -> Fraction:
    """Finite two-level pipeline recurrence with explicit buffer slots and O(stages + slots) memory."""
    durations = tuple(as_fraction(value) for value in stage_ms)
    if not durations:
        raise ModelingError("pipeline_stages_empty")
    if tiles < 1:
        raise ModelingError("pipeline_tiles_invalid")
    if slots < 1:
        raise ModelingError("pipeline_slots_invalid")
    if any(value < 0 for value in durations):
        raise ModelingError("pipeline_stage_time_invalid")
    slots = min(slots, tiles)
    prev_end = [Fraction(0) for _ in durations]
    slot_free = [Fraction(0) for _ in range(slots)]
    for tile in range(tiles):
        slot = tile % slots
        start = slot_free[slot] if slots < tiles else Fraction(0)
        for index, duration in enumerate(durations):
            if tile:
                start = max(start, prev_end[index])
            prev_end[index] = start + duration
            start = prev_end[index]
        slot_free[slot] = prev_end[-1]
    return max(prev_end)

def _subtract_stage(stages: list[dict[str, Fraction]], pipe: str, amount: Fraction, *, reverse: bool = False) -> Fraction:
    """Subtract up to ``amount`` of one pipe, returning the amount removed."""
    if amount <= 0:
        return Fraction(0)
    remaining = amount
    order = range(len(stages) - 1, -1, -1) if reverse else range(len(stages))
    for index in order:
        value = stages[index].get(pipe, Fraction(0))
        if value <= 0:
            continue
        take = min(value, remaining)
        stages[index][pipe] = value - take
        remaining -= take
        if remaining <= 0:
            break
    return amount - remaining

_FUSION_ALIASES = { "residual": ("residual_add_output", "residual_output", "residual"),
    "norm_output": ("rmsnorm_output", "norm_output"),
    "kv": ("kv_store_output", "kv_cache", "kv_norm_output", "rotary_output"), "kv_cache": ("kv_store_output", "kv_cache", "kv_norm_output"),
    "query": ("q_activation", "q_direct_output", "qk_norm_output", "q_projection_output", "q_lora_activation"),
    "key": ("qk_norm_output", "rotary_output", "kv_norm_output", "kv_cache"),
    "value": ("v_expand_output", "mla_attention_output", "attention_output", "rotary_output"),
    "down_input": ("activation_intermediate", "ffn_activation"),
    "mla_attention_output": ("mla_attention_output", "attention_output"),
    "attention_output": ("attention_output", "mla_attention_output"), }

def _narrow_keep_bytes(name, instance, members):
    """Size rstd and scales from their real shape, not a full activation."""
    owners = {"rstd": {"rmsnorm"}, "scale": {"rmsnorm", "activation_quant"},
              "activation_scale": {"activation", "shared_activation"}}[name]
    derived = {cost.name: dict(cost.derived) for cost in members}
    for operator in instance.operators:
        if operator.name not in owners:
            continue
        cols = int(operator.theta.find("elements", 0) or 0)
        total = as_fraction(derived.get(operator.name, {}).get("elements", 0))
        if cols <= 0 or total <= 0:
            continue
        rows = total / cols
        return rows * (4 if name == "rstd" else math.ceil(cols / 32))
    return Fraction(0)

def _fusion_rates(members: Sequence[OperatorCost]) -> dict[str, Fraction]:
    """Recover exact supply from evaluated work/time counters; never treat FLOPs/bytes as time."""
    rates: dict[str, Fraction] = {}
    for cost in members:
        work, times = dict(cost.work), dict(cost.times)
        for pipe in set(work) | set(times):
            amount, elapsed = as_fraction(work.get(pipe, 0)), as_fraction(times.get(pipe, 0))
            if amount < 0 or elapsed < 0 or (amount == 0) != (elapsed == 0):
                raise ModelingError("fusion_supply_required")
            if amount == 0:
                continue
            rate = amount / elapsed
            if pipe in rates and rates[pipe] != rate:
                raise ModelingError("fusion_supply_required")
            rates[pipe] = rate
    if not rates:
        raise ModelingError("fusion_supply_required")
    return rates

def _qkv_slices(instance: SubgraphInstance, contract_name: str, outputs: Mapping[str, Fraction]):
    """Declared Q/K/V slice sizes for the fused A0 projection."""
    if contract_name != "qkv_norm_rope":
        return None
    raw = dict(instance.template.contract).get("qkv_slices")
    if not raw:
        return None
    try:
        q_elems, gate_elems, k_elems, v_elems, total_elems, act_bytes = (
            Fraction(str(value)) for value in str(raw).split(","))
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"fusion_qkv_slices_invalid:{raw}") from exc
    if min(q_elems, total_elems, act_bytes) <= 0 or min(gate_elems, k_elems, v_elems) < 0 or total_elems != q_elems + gate_elems + k_elems + v_elems:
        raise ModelingError(f"fusion_qkv_slices_invalid:{raw}")
    produced = as_fraction(outputs.get("qkv_activation", 0) or outputs.get("q_projection_output", 0))
    if produced <= 0:
        return None
    tokens = produced / (total_elems * act_bytes)
    scale = tokens * act_bytes
    return {"tokens": tokens, "query": q_elems * scale, "gate": gate_elems * scale,
            "key": k_elems * scale, "value": v_elems * scale, "stored_value": v_elems * scale,
            "qk": (q_elems + k_elems) * scale, "qk_input": (q_elems + k_elems) * scale, "produced": produced}

def lower_fusion(instance: SubgraphInstance, costs: Sequence[OperatorCost],
                 contract_name: str, *, supply: PipeSupply | None = None) -> FusionResult:
    """Lower one declared native-fusion group to a physical kernel."""
    contract = next((item for item in instance.template.fusions
                     if item.name == contract_name), None)
    if contract is None:
        raise ModelingError(f"fusion_contract_missing:{instance.instance_id}:{contract_name}")
    by_name = {cost.name: cost for cost in costs}
    missing = [name for name in contract.members if name not in by_name]
    if missing:
        raise ModelingError(f"fusion_members_missing:{contract_name}:{sorted(missing)}")
    members = [by_name[name] for name in contract.members]
    rates = _fusion_rates(members) if supply is None else None
    streams = {operator.stream for operator in instance.operators if operator.name in contract.members}
    if len(streams) > 1:
        raise ModelingError(f"fusion_crosses_stream:{contract_name}:{sorted(streams)}")
    if contract.stream not in streams and streams:
        raise ModelingError(f"fusion_stream_mismatch:{contract_name}")

    outputs: dict[str, Fraction] = {}
    producer: dict[str, int] = {}
    for index, cost in enumerate(members):
        for name, value in (*cost.materialization, *cost.internal):
            outputs[name] = outputs.get(name, Fraction(0)) + value
            if value > 0:
                producer.setdefault(name, index)
    slices = _qkv_slices(instance, contract_name, outputs)

    kept_names: dict[str, str | None] = {}
    external: set[str] = set()
    for name in contract.keeps:
        if slices is not None and name in {"query", "key", "value", "gate", "kv"}:
            kept_names[name] = None
            continue
        candidates = _FUSION_ALIASES.get(name, (name,))
        chosen = next((candidate for candidate in candidates if outputs.get(candidate, Fraction(0)) > 0), None)
        kept_names[name] = chosen
        if chosen is not None:
            external.add(chosen)
        if name in outputs:
            external.add(name)
    for name, _ in (*members[-1].materialization, *members[-1].internal):
        external.add(name)
    removed_names = [name for name in contract.removes if name in outputs]
    write_removed = {name for name in removed_names if name not in external}

    kept: dict[str, Fraction] = {}
    for cost in members:
        for name, value in cost.materialization:
            if name not in write_removed:
                kept[name] = kept.get(name, Fraction(0)) + value
    for name in contract.keeps:
        if name in kept or (slices is not None and name in {"query", "key", "value", "gate", "kv"}):
            continue
        if name in {"rstd", "scale", "activation_scale"}:
            kept[name] = _narrow_keep_bytes(name, instance, members)
            continue
        chosen = kept_names[name]
        kept[name] = outputs.get(chosen, Fraction(0)) if chosen is not None else Fraction(0)
    if slices is not None:
        for stale in ("rotary_output", "qk_norm_output", "qkv_activation"):
            kept.pop(stale, None)
        kept["query"] = slices["query"]
        kept["key"] = slices["key"]
        kept["value"] = slices["stored_value"]
        if slices["gate"] > 0:
            kept["gate"] = slices["gate"]
        kept["kv"] = slices["key"] + slices["stored_value"]

    stages_by_member: list[list[dict[str, Fraction]]] = []
    removed: dict[str, Fraction] = {}
    kept_startup = False
    has_cube_startup = any(
        len(cost.stage_work) > 1 and len(cost.stage_work[-1]) == 1 and "cube" in dict(cost.stage_work[-1])
        for cost in members
    )
    for cost in members:
        base = ([dict(stage) for stage in cost.stage_work] if cost.stage_work else [dict(cost.work)])
        if contract.startup in {"single", 1} and len(base) > 1 and len(base[-1]) == 1:
            if kept_startup or (has_cube_startup and "cube" not in base[-1]):
                base = base[:-1]
            else:
                kept_startup = True
        stages_by_member.append(base)
    for name in removed_names:
        owner = producer.get(name)
        if owner is None:
            continue
        produced = sum((value for key, value in (*members[owner].materialization, *members[owner].internal) if key == name), Fraction(0))
        if produced <= 0:
            continue
        removed_here = Fraction(0)
        if slices is not None and name in {"qkv_activation", "q_projection_output"}:
            # Only the projection's intermediate Q/K write disappears; the
            # gate and V slices stay stored, and norm/rotary republish Q/K.
            # HBM is paired with the MTE bytes actually removed so a narrower
            # consumer cannot eat the producer's weight traffic or its own
            # surviving store.
            wrote = _subtract_stage(stages_by_member[owner], "mte3", slices["qk_input"], reverse=True)
            _subtract_stage(stages_by_member[owner], "hbm", wrote, reverse=True)
            removed_here += wrote
            if owner < len(members) - 1:
                read = _subtract_stage(stages_by_member[owner + 1], "mte2", produced)
                _subtract_stage(stages_by_member[owner + 1], "hbm", read)
                removed_here += read
            removed[name] = removed.get(name, Fraction(0)) + removed_here
            continue
        if name in write_removed:
            wrote = _subtract_stage(stages_by_member[owner], "mte3", produced, reverse=True)
            _subtract_stage(stages_by_member[owner], "hbm", wrote, reverse=True)
            removed_here += wrote
        if owner < len(members) - 1:
            read = _subtract_stage(stages_by_member[owner + 1], "mte2", produced)
            _subtract_stage(stages_by_member[owner + 1], "hbm", read)
            removed_here += read
        if removed_here > 0:
            removed[name] = removed.get(name, Fraction(0)) + removed_here

    flat: list[tuple[tuple[str, Fraction], ...]] = []
    work: dict[str, Fraction] = {}
    for stages in stages_by_member:
        for stage in stages:
            entries = tuple(sorted((pipe, value) for pipe, value in stage.items() if value > 0))
            flat.append(entries)
            for pipe, value in entries:
                work[pipe] = work.get(pipe, Fraction(0)) + value
    def rate(pipe: str) -> Fraction:
        if supply is not None:
            return Fraction(str(supply.effective(pipe)))
        return rates[pipe]

    stage_times: list[Fraction] = []
    times: dict[str, Fraction] = {}
    peak: dict[str, Fraction] = {}
    for entries in flat:
        stage_time = max((value / rate(pipe) for pipe, value in entries), default=Fraction(0))
        stage_times.append(stage_time)
        for pipe, value in entries:
            elapsed = value / rate(pipe)
            times[pipe] = times.get(pipe, Fraction(0)) + elapsed
            peak[pipe] = max(peak.get(pipe, Fraction(0)), elapsed)
    duration = sum(stage_times, Fraction(0))
    removed_total = sum(removed.values(), Fraction(0))
    kept_total = sum(kept.values(), Fraction(0))
    dominant = max(peak, key=peak.get) if peak else ""
    tied = tuple(pipe for pipe, value in peak.items() if peak and value == max(peak.values()))
    material = (tuple(sorted((name, value) for name, value in kept.items()
                             if name in {"query", "key", "value", "gate"} and value > 0))
                if slices is not None else tuple(sorted(kept.items())))
    derived = { "fused": Fraction(1), "physical_kernels": Fraction(1), "kept_outputs": Fraction(max(1, len(kept))),
        "removed_bytes": removed_total, "kept_bytes": kept_total, "member_stages": Fraction(len(flat)), }
    fused = OperatorCost( name=f"fused:{contract_name}", family="fused", kind=instance.kind, segment_index=-1, regime=(), guard_cells=(),
        work=tuple(sorted(work.items())), times=tuple(sorted(times.items())), stage_times=tuple(stage_times), duration_ms=duration,
        stage_work=tuple(flat), dominant_pipe=dominant, tied_pipes=tied, materialization=material, internal=(),
        residency_bytes=sum((cost.residency_bytes for cost in members), Fraction(0)),
        derived=tuple(sorted(derived.items())), formula=f"fused({contract_name})", fidelity="fused_physical_kernel", )
    return FusionResult(contract_name, tuple(members), fused, tuple(sorted(removed.items())), tuple(sorted(kept.items())), contract.startup,
                        sum((as_fraction(cost.duration_ms) for cost in members), Fraction(0)))

def _declared_effects(operator: CompiledOperator) -> tuple[tuple[str, str], ...]:
    raw = operator.operator.theta.find("state_effects", None)
    if not raw:
        return ()
    if isinstance(raw, (tuple, list)):
        return tuple((str(item[0]), str(item[1])) for item in raw
                     if isinstance(item, (tuple, list)) and len(item) == 2)
    result = []
    for item in str(raw).split(","):
        parts = item.strip().split(":", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            result.append((parts[0], parts[1]))
    return tuple(result)

def _check_requirements(state: Any, operator: CompiledOperator,
                        geometry: WorkloadGeometry | None = None) -> None:
    theta = operator.operator.theta
    kv_owner = theta.find("require_kv", None)
    if kv_owner is not None:
        state.require_kv(int(kv_owner))
    index_owner = theta.find("require_index", None)
    if index_owner is not None:
        state.require_index(int(index_owner))
    selection_owner = theta.find("require_selection", None)
    if selection_owner is not None:
        state.require_selection(int(selection_owner))
    ratio = theta.find("require_compression", None)
    if ratio is not None and geometry is not None and int(ratio) > 0:
        owner = int(theta.find("compression_owner", operator.layer) or 0)
        # Only a committed span that crossed a ratio boundary must have a
        # published row.  A rejected speculative tail can leave consumed below
        # the next boundary without invalidating earlier cache rows.
        record = state.compression_state(owner, int(ratio))
        if record.consumed >= int(ratio) and record.rows < 1:
            raise ModelingError(
                f"state_port_not_ready:compressed_kv:layer={owner}:rows=0")

def _apply_declared_effects(state: Any, operator: CompiledOperator, instance: SubgraphInstance) -> Any:
    for effect, value in _declared_effects(operator):
        if effect == "index" and value == "index_store":
            owner = operator.operator.theta.find("owner_layer", instance.layer)
            state = state.mark_index(int(owner) if int(owner) >= 0 else instance.layer)
        elif effect == "index" and value == "index_read":
            state = state.require_index(instance.layer)
            state = state
        elif effect == "kv" and value == "kv_store":
            state = state.mark_kv(instance.layer)
        elif effect == "kv" and value == "kv_read":
            state.require_kv(instance.layer)
        elif effect == "selection" and value in {"index_select", "select"}:
            owner = operator.operator.theta.find("owner_layer", instance.layer)
            state = state.mark_selection(int(owner) if int(owner) >= 0 else instance.layer)
        elif effect == "cross_stage" and value == "transfer":
            owner = int(operator.operator.theta.find("owner_layer", instance.layer) or 0)
            for stage in operator.operator.theta.find("destination_stages", ()) or ():
                state = state.mark_cross_stage(owner, int(stage))
    return state

def advance_state(compiled: CompiledSubgraph, state: Any, geometry: WorkloadGeometry) -> Any:
    """Apply one subgraph's declared write effects without re-pricing it."""
    current = state if state is not None else _empty_state()
    for operator in compiled.operators:
        _check_requirements(current, operator, geometry)
        current = _apply_operator_state(current, operator, None, compiled.instance, geometry)
        current = _apply_declared_effects(current, operator, compiled.instance)
    return current

def evaluate_serial_split( compiled: CompiledSubgraph, geometry: WorkloadGeometry, state: Any = None, *,
    geometry_for: Callable[[CompiledOperator], WorkloadGeometry] | None = None, fusion: Sequence[str] = (),
    readiness: Mapping[str, Fraction | float] | None = None, skip: Sequence[str] = (), ) -> SerialResult:
    """Evaluate one subgraph with per-operator geometry and ordered state."""
    current = state if state is not None else _empty_state()
    costs: list[OperatorCost] = []
    skipped = frozenset(skip)
    for operator in compiled.operators:
        if operator.name in skipped:
            continue
        local = geometry_for(operator) if geometry_for is not None else geometry
        _check_requirements(current, operator, local)
        cost = operator.evaluate(local, current)
        costs.append(cost)
        current = _apply_operator_state(current, operator, cost, compiled.instance, local)
        current = _apply_declared_effects(current, operator, compiled.instance)
    fused_names: list[str] = []
    fusion_results: list[FusionResult] = []
    waits = {key: as_fraction(value) for key, value in (readiness or {}).items()}
    fused_ready: dict[str, Fraction] = {}
    for name in fusion:
        result = lower_fusion(compiled.instance, costs, name, supply=compiled.supply)
        fusion_results.append(result)
        members = {cost.name for cost in result.semantic}
        member_ready = max((waits.get(cost.name, Fraction(0)) for cost in result.semantic), default=Fraction(0))
        fused_ready[result.fused.name] = max(member_ready, waits.get(result.fused.name, Fraction(0)))
        replaced: list[OperatorCost] = []
        for cost in costs:
            if cost.name == result.semantic[0].name:
                replaced.append(result.fused)
            elif cost.name not in members:
                replaced.append(cost)
        costs = replaced
        fused_names.append(name)
    gaps, completion, envelope = [], Fraction(0), None
    streams = {op.name: op.operator.stream for op in compiled.operators}
    if not waits and "aux" in streams.values():
        fork_ops = {"qkvz_projection", "qkv_projection", "ba_projection", "gate_projection", "conv", "prepare",
                    "window_attention", "compressed_attention"}
        # A fused kernel replaces its members in place and keeps their stream.
        # Dropping it onto the serial tail would undo the overlap and could
        # make the fused schedule slower than the unfused one.
        fork_names = set(fork_ops)
        for result in fusion_results:
            member_streams = {streams.get(cost.name, "main") for cost in result.semantic}
            streams[result.fused.name] = next(iter(member_streams)) if len(member_streams) == 1 else "main"
            if any(cost.name in fork_ops for cost in result.semantic):
                fork_names.add(result.fused.name)
        branches: dict[str, list[list[tuple[str, Fraction]]]] = {"main": [], "aux": []}
        for cost in costs:
            if cost.name in fork_names:
                target = branches["aux" if streams.get(cost.name) == "aux" else "main"]
                target.extend([list(s) for s in cost.stage_work] if cost.stage_work else ([list(cost.work)] if cost.work else []))
            else:
                completion += cost.duration_ms
            gaps.append(Fraction(0))
        envelope = contention_envelope(branches, compiled.supply)
        completion += envelope.makespan_ms
    else:
        for cost in costs:
            ready = max(waits.get(cost.name, Fraction(0)), fused_ready.get(cost.name, Fraction(0)))
            gaps.append(max(Fraction(0), ready - completion))
            completion = max(completion, ready) + cost.duration_ms
    effective = {**waits, **fused_ready}
    return SerialResult(tuple(costs), completion, current, tuple(gaps),
                        tuple(fused_names), tuple(sorted(effective.items())), tuple(fusion_results), envelope)

def evaluate_subgraphs_padded( compiled: Iterable[CompiledSubgraph], active: WorkloadGeometry, padded: WorkloadGeometry | None = None, *,
    active_kinds: Iterable[str] = ("A0", "A1", "A2", "A3", "A4", "K0", "K1", "I0", "I1", "F1", "F2", "F3", "D0"),
    fusion: bool = True, state: Any = None, ) -> tuple[SerialResult, ...]:
    """Sequential subgraph composition with capture-padding separation.

    Attention/MoE/transport families keep their real active rows by default;
    matrix/elementwise/copy families receive the padded bucket geometry when
    the caller supplied one.  Attention family operators inside A0/A1 are also
    active even though the containing macro exists in the padded set.
    """
    active_set = frozenset(active_kinds)
    current = state if state is not None else _empty_state()
    results = []
    for compiled in compiled:
        kind = compiled.instance.kind

        def geometry_for(operator, kind=kind):
            if padded is None:
                return active
            if kind in active_set:
                return active
            if kind in {"A0", "A1"} and operator.family in {"attention", "mla"}:
                return active
            return padded

        names = tuple(item.name for item in compiled.instance.template.fusions) if fusion else ()
        result = evaluate_serial_split(compiled, active, current, geometry_for=geometry_for, fusion=names)
        current = result.state
        results.append(result)
    return tuple(results)

class SubgraphCostFunction:
    """Analytical facade over one compiled subgraph instance."""
    def __init__(self, compiled: CompiledSubgraph):
        if not isinstance(compiled, CompiledSubgraph):
            raise ModelingError("cost_function_requires_compiled_subgraph")
        self.compiled = compiled
        self.instance = compiled.instance
        self.supply = compiled.supply
        self.policy = compiled.policy
    def evaluate_serial(self, geometry: WorkloadGeometry, state: Any = None, *, fusion: Sequence[str] = (),
                        readiness: Mapping[str, Fraction | float] | None = None ) -> SerialResult:
        state = state or _empty_state()
        base = self.compiled.evaluate(geometry, state)
        costs = list(base.operators)
        fused_names: list[str] = []
        fusion_results: list[FusionResult] = []
        readiness = {key: as_fraction(value) for key, value in (readiness or {}).items()}
        fused_ready: dict[str, Fraction] = {}
        for name in fusion:
            result = lower_fusion(self.instance, costs, name, supply=self.supply)
            fusion_results.append(result)
            members = {cost.name for cost in result.semantic}
            member_ready = max((readiness.get(cost.name, Fraction(0)) for cost in result.semantic), default=Fraction(0))
            fused_ready[result.fused.name] = max( member_ready, readiness.get(result.fused.name, Fraction(0)))
            replaced: list[OperatorCost] = []
            for cost in costs:
                if cost.name == result.semantic[0].name:
                    replaced.append(result.fused)
                elif cost.name not in members:
                    replaced.append(cost)
            costs = replaced
            fused_names.append(name)
        gaps: list[Fraction] = []
        completion = Fraction(0)
        ordered = []
        for cost in costs:
            ready = max(readiness.get(cost.name, Fraction(0)), fused_ready.get(cost.name, Fraction(0)))
            gap = max(Fraction(0), ready - completion)
            start = max(completion, ready)
            completion = start + cost.duration_ms
            gaps.append(gap)
            ordered.append(cost)
        effective = dict(readiness)
        effective.update(fused_ready)
        return SerialResult(tuple(ordered), completion, base.state_out, tuple(gaps), tuple(fused_names),
                            tuple(sorted(effective.items())), tuple(fusion_results))
    def evaluate_isolated(self, geometry: WorkloadGeometry, state: Any = None) -> SerialResult:
        return self.evaluate_serial(geometry, state)
    def evaluate_fork_join(self, geometry: WorkloadGeometry, branches: Mapping[str, Sequence[OperatorCost]],
                           *, prefix_ms: Fraction | float = 0, join_ms: Fraction | float = 0, state: Any = None) -> ForkJoinResult:
        state = state or _empty_state()
        costs = {name: tuple(values) for name, values in branches.items()}
        stage_work = {}
        isolated = {}
        for name, operators in costs.items():
            stages: list[list[tuple[str, Fraction]]] = []
            for cost in operators:
                if cost.stage_work:
                    stages.extend([[(pipe, value) for pipe, value in stage] for stage in cost.stage_work])
                else:
                    stages.append([(pipe, value) for pipe, value in cost.work])
            stage_work[name] = stages
            isolated[name] = sum((cost.duration_ms for cost in operators), Fraction(0))
        engine_result = run_engine_work(stage_work, self.supply)
        prefix, join = as_fraction(prefix_ms), as_fraction(join_ms)
        return ForkJoinResult( prefix, join, max(isolated.values()) if isolated else Fraction(0),
            engine_result.makespan_ms, engine_result.makespan_ms, engine_result.branches, engine_result.resources)
    def evaluate_fork_join_stages( self, branches: Mapping[str, Sequence[Sequence[tuple[str, Fraction]]]],
        *, prefix_ms: Fraction | float = 0, join_ms: Fraction | float = 0, ) -> ForkJoinResult:
        engine_result = run_engine_work(branches, self.supply)
        isolated = {}
        for name, stages in branches.items():
            isolated[name] = sum( (max((value / self.supply.effective(pipe) for pipe, value in stage),
                     default=Fraction(0)) for stage in stages), Fraction(0))
        return ForkJoinResult( as_fraction(prefix_ms), as_fraction(join_ms),
            max(isolated.values()) if isolated else Fraction(0), engine_result.makespan_ms, engine_result.makespan_ms,
            engine_result.branches, engine_result.resources)
    def explain(self, geometry: WorkloadGeometry, state: Any = None):
        from .diagnostics import explain
        return explain(self, geometry, state)
    def closed_form(self) -> ClosedForm:
        stages = [stage for operator in self.compiled.operators for stage in operator.program.stages]
        if any(len(operator.program.stages) > 1 for operator in self.compiled.operators):
            return ClosedForm("", False, "multi_stage_operator_requires_engine")
        if any(operator.axes for operator in self.compiled.operators):
            return ClosedForm( "sum_m max_p W_{m,p}(X,Z)/R_p ; guards lazy", True, "piecewise_guarded_serial_chain")
        return ClosedForm("sum_m max_p W_{m,p}/R_p", True, "single_stage_serial_chain")
    def to_iteration_work(self, geometry: WorkloadGeometry, state: Any = None) -> tuple[OpSpec, ...]:
        """Lower the same evaluated work to the existing iteration contract."""
        state = state or _empty_state()
        evaluation = self.compiled.evaluate(geometry, state)
        records = []
        for cost in evaluation.operators:
            mte2 = cost.pipe_work("mte2")
            mte3 = cost.pipe_work("mte3")
            vector = cost.pipe_work("vector")
            flops = cost.get("flops") or cost.get("logical_flops") or cost.pipe_work("cube")
            hbm = cost.pipe_work("hbm") or (mte2 + mte3)
            records.append(OpSpec( op_id=f"{self.instance.instance_id}:{cost.name}",
                family=cost.family, phase=geometry.phase_name, stage=self.instance.stage, layer=self.instance.layer,
                requests=tuple(f"r{index}" for index in range(geometry.batch)), semantic_shape=(geometry.effective_tokens,),
                local_shape=(geometry.tokens,), padded_shape=(geometry.effective_tokens,),
                launch_shape=(geometry.effective_tokens,), flops=float(flops), hbm_bytes=float(hbm),
                mte2_bytes=float(mte2), mte3_bytes=float(mte3),
                vector_ops=float(vector), recipe=cost.formula[:200] or cost.family, dtype="bf16", layout="contiguous",
                metadata=(("piecewise_segment", cost.segment_index), ("piecewise_regime", cost.regime),
                          ("dominant_pipe", cost.dominant_pipe)), ))
        return tuple(records)
    def to_engine_work(self, geometry: WorkloadGeometry, state: Any = None) -> dict[str, list[list[tuple[str, Fraction]]]]:
        state = state or _empty_state()
        evaluation = self.compiled.evaluate(geometry, state)
        stages: list[list[tuple[str, Fraction]]] = []
        for cost in evaluation.operators:
            if cost.stage_work:
                stages.extend([[ (pipe, value) for pipe, value in stage] for stage in cost.stage_work])
            elif cost.work:
                stages.append([(pipe, value) for pipe, value in cost.work])
        return {self.instance.instance_id: stages}

def p2p_transfer_cost(layer: int, tokens: int, width: int, act_bytes: float, supply: PipeSupply) -> OperatorCost:
    """Hidden-state point-to-point at an actual pipeline stage cut.

    This is a directed copy, not a collective: no ring expansion is applied
    and the operation is not classified as AllReduce.
    """
    payload = Fraction(tokens * width * float(act_bytes))
    stage_work = ((("link", payload), ("mte2", payload)), (("hbm", 2 * payload), ("mte3", payload)))
    rates = {pipe: Fraction(str(supply.effective(pipe))) for pipe in ("link", "mte2", "mte3", "hbm")}
    times = {"link": payload / rates["link"], "mte2": payload / rates["mte2"], "mte3": payload / rates["mte3"],
             "hbm": 2 * payload / rates["hbm"]}
    stage_times = tuple(max(times[pipe] for pipe, _ in stage) for stage in stage_work)
    work: dict[str, Fraction] = {}
    for stage in stage_work:
        for pipe, value in stage:
            work[pipe] = work.get(pipe, Fraction(0)) + value
    duration = sum(stage_times, Fraction(0))
    peak = max(times.values())
    return OperatorCost( name="stage_transfer", family="p2p", kind="PP", segment_index=-1,
        regime=(), guard_cells=(), work=tuple(sorted(work.items())),
        times=tuple(sorted(times.items())), stage_times=stage_times, duration_ms=duration, stage_work=stage_work,
        dominant_pipe=max(times, key=times.get), tied_pipes=tuple(pipe for pipe, value in times.items() if value == peak),
        materialization=(), internal=(), residency_bytes=Fraction(0),
        derived=(("payload_bytes", payload), ("p2p", Fraction(1)), ("destination_stage_delta", Fraction(1))),
        formula="pipeline_send;p2p", fidelity="isolated_device_time")

def compose_sequence(subgraphs: Iterable[CompiledSubgraph], geometry: WorkloadGeometry, state: Any = None,
                     *, readiness: Mapping[str, Fraction | float] | None = None ) -> SerialResult:
    """Compose subgraphs in model order, carrying one explicit state view."""
    state, costs, gaps, completion = state or _empty_state(), [], [], Fraction(0)
    readiness = {key: as_fraction(value) for key, value in (readiness or {}).items()}
    for subgraph in subgraphs:
        evaluation = subgraph.evaluate(geometry, state)
        state = evaluation.state_out
        for cost in evaluation.operators:
            ready = readiness.get(cost.name, Fraction(0))
            gaps.append(max(Fraction(0), ready - completion))
            completion = max(completion, ready) + cost.duration_ms
            costs.append(cost)
    return SerialResult(tuple(costs), completion, state, tuple(gaps))
def _empty_state():
    from mindie_sim.prediction.subgraphs.state import StateView
    return StateView()
