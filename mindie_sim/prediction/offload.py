"""Hierarchical expert/KV cache DDR-SSD offload with prefetch masking.

[Role]
  Price host-DDR and SSD tiering for routed MoE experts and KV cache, and
  expose how much of each transfer hides behind attention, shared-expert,
  routed-FFN, and full layer-stack work. The module is an analysis layer over
  the shared causal engine; it never loads weights or touches devices.

[Invariants]
  * Disabled policies reproduce the baseline step exactly: zero transfer,
    zero exposed time, and zero resident-byte movement.
  * Expert bytes are derived from catalog weight terms per stage rank, so TP,
    EP, ETP, PP and quantization keep their existing ownership boundaries.
  * Each transfer tier is charged its own bandwidth, utilization, and latency;
    MTP hit rates discount redundant bytes, never the fixed per-layer latency.
  * Prefetch masks can only hide work, never create negative exposed time, and
    the v1 -> v4 masks are monotone by construction.
  * All ratios and bandwidths are validated before arithmetic is performed.

[Timeouts & Locks]
  Pure in-memory arithmetic and one bounded operator evaluation; no I/O,
  subprocesses, sleeps, device contexts, or locks.

[Side-Effect Policy]
  Inputs are never mutated; every entry point returns a new plain dictionary.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any
import weakref

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec

_PREFETCH_VERSIONS = ("v1", "v2", "v3", "v4")
_ATTENTION_FAMILIES = frozenset({
    "attention", "qkv_projection", "output_projection", "rotary", "cache_store",
    "gdn_gating", "short_convolution", "recurrent_vector",
})
_FFN_FAMILIES = frozenset({
    "grouped_gemm", "moe_ffn", "gate_up_projection", "down_projection", "swiglu",
    "situ", "router", "topk", "routing", "alltoall", "alltoallv", "projection",
})
_ATTENTION_RECIPES = frozenset({
    "q_a", "q_b", "kv_a", "kv_b", "qkv", "qkv_projection", "linear_qkvz",
    "linear_output", "q_proj", "kv_proj", "q_absorb", "v_expand", "q_flat",
    "k_flat", "v_flat", "qkv_concatenate", "qkv_split_materialize",
})
_ATTENTION_NORMS = frozenset({"q_norm", "kv_norm", "qk_norm", "q_a_norm", "kv_a_norm"})
_ATTENTION_COMMS = frozenset({"linear_reduce", "catalog_tp_allreduce", "output_reduce"})
_FFN_TOKENS = (
    "expert", "grouped", "moe", "ffn", "router", "route_", "topk", "gate_up",
    "dense_down", "dense_gate_up", "swiglu", "situ", "latent", "shared",
)


@dataclass(frozen=True)
class OffloadPolicy:
    """Validated host-DDR/SSD offload and prefetch policy."""

    expert_offload_enabled: bool = False
    force_expert_offload: bool = False
    expert_ddr_ratio: float = 0.0
    ddr_bw_gbps: float = 200.0
    ddr_bw_util: float = 0.70
    ddr_latency_us: float = 5.0
    ssd_bw_gbps: float = 14.0
    ssd_bw_util: float = 0.70
    ssd_latency_us: float = 80.0
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0
    kvc_prefetch_version: str = "v3"
    mtp_ffn_hit_rate: float = 0.0
    mtp_kvc_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("expert_offload_enabled", "force_expert_offload"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"offload_policy_boolean_required:{name}")
        for name in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio",
                     "mtp_ffn_hit_rate", "mtp_kvc_hit_rate"):
            _ratio(name, getattr(self, name))
        for name in ("ddr_bw_util", "ssd_bw_util"):
            _utilization(name, getattr(self, name))
        for name in ("ddr_bw_gbps", "ssd_bw_gbps"):
            _positive(name, getattr(self, name))
        for name in ("ddr_latency_us", "ssd_latency_us"):
            _nonnegative(name, getattr(self, name))
        if self.kvc_prefetch_version not in _PREFETCH_VERSIONS:
            raise ValueError(f"offload_policy_invalid_prefetch_version:{self.kvc_prefetch_version}")
        if self.kvc_ddr_ratio + self.kvc_ssd_ratio > 1.0 + 1e-12:
            raise ValueError("offload_policy_kvc_ratios_exceed_unity")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "OffloadPolicy":
        """Build a policy with strict ratio, bandwidth, and enum validation."""
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if not isinstance(mapping, Mapping):
            raise ValueError("offload_policy_mapping_required")
        unknown = set(mapping) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("offload_policy_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**dict(mapping))


def _ratio(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    return result


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"offload_policy_must_be_positive:{name}:{value}")
    return float(value)


def _utilization(name: str, value: Any) -> float:
    result = _positive(name, value)
    if result > 1.0:
        raise ValueError(f"offload_policy_invalid_utilization:{name}:{value}")
    return result


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"offload_policy_must_be_nonnegative:{name}:{value}")
    return float(value)


def _resolve_policy(policy: OffloadPolicy | Mapping[str, Any] | None) -> OffloadPolicy:
    if policy is None:
        return OffloadPolicy()
    if isinstance(policy, OffloadPolicy):
        return policy
    return OffloadPolicy.from_mapping(policy)


def _moe_geometry(parsed: CaseSpec, memory: Memory, batch_tokens: int) -> dict[str, Any]:
    """Derive routed-expert bytes from the same weight terms that own HBM."""
    parallel = parsed.runtime.parallel
    quant = parsed.model.quant or QuantSpec("bf16")
    stage_routed = [0.0] * max(1, parallel.pp)
    layers: list[dict[str, Any]] = []
    for plan in parsed.model.layer_plans:
        if plan.ffn != "moe" or int(plan.num_experts) <= 0 or int(plan.moe_intermediate_size) <= 0:
            continue
        routed_bytes = float(sum(
            term_bytes(term, quant, parallel, parsed.machine.ranks_per_node)
            for term in layer_weight_terms(plan, parsed.model) if term.category == "routed_expert"))
        local_experts = max(1, (int(plan.num_experts) + max(1, int(parallel.ep)) - 1) // max(1, int(parallel.ep)))
        stage = memory.layer_stage[plan.index] if plan.index < len(memory.layer_stage) else 0
        stage_routed[stage] += routed_bytes
        layers.append({
            "layer": int(plan.index), "stage": int(stage),
            "num_experts": int(plan.num_experts), "top_k": int(plan.top_k),
            "local_experts_per_rank": int(local_experts),
            "one_expert_bytes_per_rank": float(routed_bytes / local_experts),
            "routed_expert_bytes_per_rank": float(routed_bytes),
            "active_experts_per_rank": _active_experts_per_rank(
                int(plan.num_experts), int(plan.top_k), batch_tokens, parallel.ep),
            "ddr_bytes": 0.0,
        })
    return {"layers": layers, "stage_routed_bytes": stage_routed}


def _expected_active_experts(num_experts: int, batch_tokens: int, top_k: int) -> float:
    if num_experts <= 0 or batch_tokens <= 0 or top_k <= 0:
        return 0.0
    k = min(int(num_experts), int(top_k))
    if k >= num_experts:
        return float(num_experts)
    miss_per_token = 1.0 - float(k) / float(num_experts)
    expected = float(num_experts) * (1.0 - (miss_per_token ** float(batch_tokens)))
    return min(float(num_experts), max(0.0, expected))


def _active_experts_per_rank(num_experts: int, top_k: int, batch_tokens: int, ep: int) -> float:
    return _expected_active_experts(num_experts, batch_tokens, top_k) / float(max(1, ep))


def resolve_case_offload_policy(case: CaseSpec, memory_policy: Any = None) -> OffloadPolicy | None:
    """Resolve active offload policy from case metadata and memory policy."""
    raw_meta = dict(case.runtime.metadata).get("offload", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    merged: dict[str, Any] = dict(raw_meta)
    mem_cfg = dict(case.runtime.memory or {})
    for key in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio"):
        if mem_cfg.get(key, 0.0):
            merged[key] = mem_cfg[key]
        if memory_policy is not None and getattr(memory_policy, key, 0.0):
            merged[key] = getattr(memory_policy, key)
    if not merged:
        return None
    policy = OffloadPolicy.from_mapping(merged)
    if (policy.expert_ddr_ratio <= 0.0 and policy.kvc_ddr_ratio <= 0.0
            and policy.kvc_ssd_ratio <= 0.0 and not policy.force_expert_offload):
        return None
    return policy


def offload_resources(policy: OffloadPolicy | None, namespace: str, dp: int, pp: int) -> list[Any]:
    """Build host DDR and SSD transfer resources for Execution when offload is active."""
    if policy is None:
        return []
    from mindie_sim.prediction.engine import Resource
    ddr_rate = policy.ddr_bw_gbps * policy.ddr_bw_util * 1e6
    ssd_rate = policy.ssd_bw_gbps * policy.ssd_bw_util * 1e6
    resources = []
    for replica in range(max(1, dp)):
        for stage in range(max(1, pp)):
            resources.append(Resource(f"{namespace}offload_ddr@r{replica}:s{stage}",
                                      ddr_rate, "bytes", "fifo"))
            resources.append(Resource(f"{namespace}offload_ssd@r{replica}:s{stage}",
                                      ssd_rate, "bytes", "fifo"))
    return resources


def _submit_offload_transfer(
    engine: Any, name: str, work: Mapping[str, float], *,
    after: tuple[Any, ...], latency_us: float, owner: str,
) -> Any:
    startup_ms = float(latency_us) / 1000.0
    if startup_ms > 0.0:
        start = engine.submit(f"{name}:startup", after=after, latency=startup_ms, owner=owner)
        return engine.submit(name, work, after=(start,), owner=owner)
    return engine.submit(name, work, after=after, owner=owner)


def is_routed_expert_entry_op(op: Any) -> bool:
    if getattr(op, "layer", -1) < 0:
        return False
    meta = dict(getattr(op, "metadata", ()) or ())
    if getattr(op, "family", "") == "grouped_gemm" or getattr(op, "recipe", "") in {
        "grouped_gate_up", "balanced_grouped_gate_up", "grouped_expert_gemm",
    }:
        return True
    if meta.get("semantic_recipe") in {"grouped_gate_up", "balanced_grouped_gate_up"}:
        return True
    if meta.get("weight_category") == "routed_expert":
        return True
    return any(
        isinstance(r, Mapping) and (
            r.get("family") == "grouped_gemm"
            or r.get("recipe") in {"grouped_gate_up", "balanced_grouped_gate_up"}
        )
        for r in meta.get("fusion_semantic_recipes", ())
    )


def is_kv_or_state_entry_op(op: Any) -> bool:
    if getattr(op, "layer", -1) < 0:
        return False
    # A draft producer can share a numeric layer index with the target.  It
    # must not enter the target layer fetch/writeback path; T04 schedules its
    # aggregate writeback separately with schedule_draft_kv_writeback.
    if dict(getattr(op, "metadata", ()) or ()).get("producer_scope") == "draft":
        return False
    return getattr(op, "family", "") in {
        "attention", "attention_online", "mla", "mla_decode",
        "compressed_attention", "dsa", "indexer", "recurrent_scan",
        "recurrent_vector", "gdn_state", "gdn_recurrent", "kda",
        "kda_recurrent", "short_convolution", "cache_store",
    } or getattr(op, "recipe", "") in {"short_convolution", "causal_conv1d_update"}


def _resolve_active_experts_from_op(op: Any) -> float | None:
    if op is None:
        return None
    op_meta = dict(getattr(op, "metadata", ()) or ())
    sem_meta = op_meta.get("semantic_metadata")
    sources = (op_meta, dict(sem_meta) if isinstance(sem_meta, Mapping) else {})
    for meta in sources:
        val = meta.get("expected_active_experts", meta.get("matrix_groups"))
        if val is not None:
            return float(val)
        rows = meta.get("matrix_group_rows")
        if isinstance(rows, (list, tuple)) and rows:
            return float(sum(1 for r in rows if r > 0))
        realization = meta.get("routing_realization")
        if isinstance(realization, Mapping):
            if realization.get("active") is not None:
                return float(realization["active"])
            r_rows = realization.get("local_expert_rows", realization.get("expert_rows"))
            if isinstance(r_rows, (list, tuple)) and r_rows:
                return float(sum(1 for r in r_rows if r > 0))
        gmm = meta.get("grouped_gemm")
        if isinstance(gmm, Mapping):
            g_rows = gmm.get("expert_rows")
            if isinstance(g_rows, (list, tuple)) and g_rows:
                return float(sum(1 for r in g_rows if r > 0))
    return None


def _offload_memory_for(execution: Any) -> Memory:
    """The one Memory shared with the service Scheduler when available."""
    memory = getattr(execution, "memory", None)
    if memory is None:
        memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
    execution._offload_memory = memory
    return memory


def schedule_expert_layer_offload(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    valid_tokens: int, input_ready: Any, op: Any = None,
) -> Any | None:
    """Submit per-MoE-layer DDR expert fetch contended on the stage FIFO DDR bus."""
    policy = getattr(execution, "offload_policy", None)
    active_override = _resolve_active_experts_from_op(op)
    if policy is None or policy.expert_ddr_ratio <= 0.0 or (valid_tokens <= 0 and not active_override):
        return None
    memory = _offload_memory_for(execution)
    moe = _moe_geometry(execution.case, memory, max(1, int(valid_tokens)))
    layer_info = next((item for item in moe["layers"] if item["layer"] == layer), None)
    if layer_info is None:
        return None
    active_experts = float(active_override) if active_override is not None else float(layer_info["active_experts_per_rank"])
    ddr_bytes = float(
        active_experts * layer_info["one_expert_bytes_per_rank"]
        * policy.expert_ddr_ratio * (1.0 - policy.mtp_ffn_hit_rate)
    )
    if ddr_bytes <= 0.0:
        return None
    ddr_res = f"{execution.namespace}offload_ddr@r{replica}:s{stage}"
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    return _submit_offload_transfer(
        execution.engine,
        f"{prefix}:s{stage}.l{layer}:expert_ddr_offload",
        {ddr_res: ddr_bytes, hbm_res: ddr_bytes},
        after=(input_ready,),
        latency_us=policy.ddr_latency_us,
        owner=f"replica:{replica}",
    )


def _normalize_span(value) -> tuple[tuple[int, int], ...]:
    """Stable hashable range identity for mapping and dataclass inputs."""
    if not value:
        return ()
    normalized = []
    for item in value:
        if isinstance(item, (str, bytes)):
            raise ValueError("state_access_span_range_invalid")
        pair = tuple(item)
        if len(pair) != 2:
            raise ValueError("state_access_span_range_requires_pair")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end < start:
            raise ValueError("state_access_span_range_invalid")
        normalized.append((start, end))
    return tuple(normalized)


@dataclass(frozen=True)
class StateAccess:
    """Instance binding of one real state read/write/alias to an occurrence.

    This is a private projection over existing OpSpec/TensorValue/StateEffect
    data, not a new public IR.  ``bytes`` is the actual issued transfer volume
    for this direction/span; ``read_bytes``/``write_bytes`` are retained for
    callers that carry one record per direction.
    """

    occurrence: str
    producer: str
    request: str
    stage: int
    layer: int
    direction: str
    position: str = "cache"
    output: str = ""
    storage: str = ""
    state_version: int | None = None
    span: tuple[tuple[int, int], ...] = ()
    rank: int | None = None
    scope: str = "target"
    bytes: float = 0.0
    source: str = ""
    reader: str = ""
    grant: str = ""
    decision: str = ""
    last_use: str = ""

    def __post_init__(self):
        object.__setattr__(self, "span", _normalize_span(self.span))
        if self.direction not in {"read", "write", "alias"}:
            raise ValueError(f"state_access_direction_invalid:{self.direction}")
        if not self.occurrence or not self.producer:
            raise ValueError("state_access_requires_occurrence_and_producer")
        if isinstance(self.bytes, bool) or not math.isfinite(float(self.bytes)) or float(self.bytes) < 0:
            raise ValueError("state_access_bytes_invalid")
        object.__setattr__(self, "bytes", float(self.bytes))

    @property
    def key(self) -> tuple:
        # One outer occurrence can carry several physical writers.  The
        # producer and stage are part of the physical access identity so two
        # different stores under the same call are not collapsed into one
        # issued write.
        return (self.occurrence, self.producer, self.stage, self.output,
                self.storage, self.state_version, self.span, self.direction,
                self.rank, self.request, self.scope)

    def to_dict(self) -> dict:
        return {
            "occurrence": self.occurrence, "producer": self.producer,
            "request": self.request, "stage": self.stage, "layer": self.layer,
            "direction": self.direction, "position": self.position,
            "output": self.output, "storage": self.storage,
            "state_version": self.state_version, "span": self.span,
            "rank": self.rank, "scope": self.scope, "bytes": self.bytes,
            "source": self.source, "reader": self.reader, "grant": self.grant,
            "decision": self.decision, "last_use": self.last_use,
        }


def _as_access(value: Any) -> StateAccess:
    if isinstance(value, StateAccess):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        if "bytes" not in data:
            data["bytes"] = data.get("write_bytes", data.get("read_bytes", 0.0))
        data.pop("read_bytes", None)
        data.pop("write_bytes", None)
        data.setdefault("position", "cache")
        data.setdefault("output", "")
        data.setdefault("storage", "")
        data.setdefault("scope", "target")
        data["span"] = _normalize_span(data.get("span", ()))
        data.setdefault("rank", None)
        data.setdefault("state_version", None)
        data.setdefault("source", "")
        data.setdefault("reader", "")
        data.setdefault("grant", "")
        data.setdefault("decision", "")
        data.setdefault("last_use", "")
        return StateAccess(**data)
    raise TypeError("state_access_mapping_or_StateAccess_required")


def _as_accesses(values) -> tuple[StateAccess, ...]:
    return tuple(_as_access(value) for value in (values or ()))


def _unique_accesses(accesses) -> tuple[StateAccess, ...]:
    seen = set()
    result = []
    for access in accesses:
        if access.key in seen:
            continue
        seen.add(access.key)
        result.append(access)
    return tuple(result)


def _access_registry(execution: Any, work: Any) -> dict:
    """Issued-access registry for one actual IterationWork invocation.

    The same work shares layer/draft/fallback accounting; a later batch has a
    new work object and is charged independently even for the same request or
    op name.  A weak reference prevents the registry from owning the batch.
    """
    batches = getattr(execution, "_offload_access_batches", None)
    if batches is None:
        batches = {}
        execution._offload_access_batches = batches
    for key, (reference, _registry) in list(batches.items()):
        if reference() is None:
            batches.pop(key, None)
    entry = batches.get(id(work))
    if entry is not None and entry[0]() is work:
        return entry[1]
    registry: dict = {}
    try:
        reference = weakref.ref(work)
    except TypeError:
        reference = lambda: work
    batches[id(work)] = (reference, registry)
    return registry


def _accesses_for_work(execution: Any, work: Any) -> tuple[StateAccess, ...]:
    """One actual access set for all four paths of this IterationWork."""
    bindings = getattr(execution, "_offload_actual_accesses", None)
    if bindings is None:
        bindings = {}
        execution._offload_actual_accesses = bindings
    for key, (reference, _accesses) in list(bindings.items()):
        if reference() is None:
            bindings.pop(key, None)
    entry = bindings.get(id(work))
    if entry is not None and entry[0]() is work:
        return entry[1]
    accesses = _accesses_from_ops(work)
    try:
        reference = weakref.ref(work)
    except TypeError:
        reference = lambda: work
    bindings[id(work)] = (reference, accesses)
    return accesses


def release_offload_access(execution: Any, work: Any) -> int:
    """Drop one batch's issued registry and actual Event bindings.

    This is the explicit batch-end hook T04 calls after real last use.  Event
    history must not accumulate with the number of iterations.
    """
    batches = getattr(execution, "_offload_access_batches", None) or {}
    entry = batches.pop(id(work), None)
    bindings = getattr(execution, "_offload_actual_accesses", None)
    if bindings is not None:
        bindings.pop(id(work), None)
    event_bindings = getattr(execution, "_offload_access_events", None)
    if event_bindings is not None:
        event_bindings.pop(id(work), None)
    return len(entry[1]) if entry is not None else 0


def _event_table(source: Any) -> dict:
    """Normalize actual access->Event bindings without inspecting Event.name.

    Accepted shapes are an ``access key``/``StateAccess``/``(producer,
    output)``/``producer`` mapping, or an iterable of rows carrying ``event``
    plus those identity fields.  This is the same physical identity used by
    ``StateAccess.key``; display names are irrelevant.
    """
    table = {"key": {}, "producer_output": {}, "producer": {}}
    if source is None:
        return table
    if isinstance(source, Mapping):
        items = source.items()
    else:
        items = ((None, row) for row in source)
    for key, value in items:
        if key is None:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            event = row.get("event")
            if event is None:
                continue
            access = row.get("access")
            if isinstance(access, StateAccess):
                table["key"][access.key] = event
                continue
            identity = row.get("key")
            if isinstance(identity, (tuple, list)) and identity:
                table["key"][tuple(identity)] = event
                continue
            producer = str(row.get("producer") or "")
            output = str(row.get("output") or "")
            occurrence = str(row.get("occurrence") or "")
            if producer and output:
                table["producer_output"][(producer, output)] = event
            elif producer:
                table["producer"][producer] = event
            elif occurrence:
                table["producer"][occurrence] = event
            continue
        if value is None:
            continue
        if isinstance(key, StateAccess):
            table["key"][key.key] = value
        elif isinstance(key, (tuple, list)):
            identity = tuple(key)
            if (len(identity) == 2
                    and all(isinstance(item, str) for item in identity)):
                table["producer_output"][identity] = value
            else:
                table["key"][identity] = value
        elif isinstance(key, str):
            table["producer"][key] = value
    return table


def _merge_event_tables(*tables) -> dict:
    merged = {"key": {}, "producer_output": {}, "producer": {}}
    for table in tables:
        if not table:
            continue
        for field in merged:
            merged[field].update(table.get(field, {}))
    return merged


def _has_event_bindings(table: Mapping | None) -> bool:
    return bool(table and any(table.get(field) for field in
                              ("key", "producer_output", "producer")))


def _event_for_access(access: StateAccess, table: Mapping | None) -> Any:
    """Resolve the actual producer.output Event for one physical access."""
    if not table:
        return None
    event = table.get("key", {}).get(access.key)
    if event is not None:
        return event
    event = table.get("producer_output", {}).get(
        (access.producer, access.output))
    if event is not None:
        return event
    return table.get("producer", {}).get(access.producer)


def bind_offload_access_events(execution: Any, work: Any, bindings: Any) -> int:
    """Bind the same batch's actual producer.output Events once.

    All four schedule_* paths consume this table through their optional
    ``access_events`` argument or this explicit per-work registration.  A
    later real output can be added before the unissued write is submitted.
    """
    events = getattr(execution, "_offload_access_events", None)
    if events is None:
        events = {}
        execution._offload_access_events = events
    for key, (reference, _table) in list(events.items()):
        if reference() is None:
            events.pop(key, None)
    current = _work_event_table(execution, work)
    table = _merge_event_tables(current, _event_table(bindings))
    if not _has_event_bindings(table):
        return 0
    try:
        reference = weakref.ref(work)
    except TypeError:
        reference = lambda: work
    events[id(work)] = (reference, table)
    return sum(len(rows) for rows in table.values())


def _work_event_table(execution: Any, work: Any) -> dict:
    events = getattr(execution, "_offload_access_events", None)
    if not events:
        return {}
    entry = events.get(id(work))
    if entry is None or entry[0]() is not work:
        return {}
    return entry[1]


def _registered_events(execution: Any, work: Any, accesses) -> tuple[Any, ...]:
    registry = _access_registry(execution, work)
    events = []
    for access in _unique_accesses(accesses):
        events.extend(registry.get(access.key, ()))
    return tuple(dict.fromkeys(events))


def _register_access_events(execution: Any, work: Any, accesses, events) -> None:
    if not events:
        return
    registry = _access_registry(execution, work)
    for access in _unique_accesses(accesses):
        registry[access.key] = tuple(events)



def _access_bytes(accesses, direction: str) -> float:
    return float(sum(access.bytes for access in _unique_accesses(accesses)
                     if access.direction == direction))


def _physical_producer_output(meta: Mapping, occurrence: str,
                              default_output: str = "") -> tuple[str, str]:
    """Read the producer/output port carried by the selected op.

    Only producer-owned fields are consumed.  A generic metadata key named
    ``producer`` is not guessed to be a state port; explicit lowering owners
    supply ``state_producer``/``state_output`` or ``state_producer_output``.
    When none is present the selected occurrence itself is the writer.
    """
    raw = meta.get("state_producer_output")
    producer = ""
    output = ""
    if isinstance(raw, Mapping):
        producer = str(raw.get("producer") or "")
        output = str(raw.get("output") or "")
    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
        producer, output = str(raw[0] or ""), str(raw[1] or "")
    producer = producer or str(
        meta.get("state_producer") or meta.get("cache_producer") or "")
    output = output or str(
        meta.get("state_output") or meta.get("cache_output") or "")
    return producer or occurrence, output or default_output


def _accesses_from_ops(work: Any, physical_ops=None) -> tuple[StateAccess, ...]:
    """Compatibility binding from real selected op metadata only.

    It never walks the model layer list, a token total or a capacity delta.
    Missing state-byte/output/storage bindings stay absent; the lowering owner
    must supply them through the explicit ``state_accesses`` argument.
    """
    ops = (tuple(physical_ops) if physical_ops is not None
           else tuple(getattr(work, "ops", ()) or ()))
    if not ops:
        return ()
    plan = getattr(work, "plan", None)
    padding_ids = {str(view.request_id) for view in
                   (getattr(plan, "requests", ()) or ())
                   if getattr(view, "padding", False)}
    result = []
    for op in ops:
        meta = dict(getattr(op, "metadata", ()) or ())
        stage = int(getattr(op, "stage", 0))
        layer = int(getattr(op, "layer", -1))
        raw_requests = tuple(getattr(op, "requests", ()) or ("*",))
        requests = tuple(request for request in raw_requests
                         if str(request) not in padding_ids)
        if not requests:
            continue
        scope = "draft" if meta.get("producer_scope") == "draft" else "target"
        occurrence = str(getattr(op, "op_id", ""))
        kind = str(meta.get("cache_kind", ""))
        producer, bound_output = _physical_producer_output(
            meta, occurrence, kind)
        if kind == "index_pool_tail":
            retained = int(meta.get("retained_rows", 0) or 0)
            width = int(meta.get("index_width", 0) or 0)
            write = float(meta.get("write_bytes", 0.0) or 0.0)
            if write <= 0 and retained and width:
                write = float(retained * width * 2 * 4)
            if write <= 0:
                write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
            if write > 0:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=requests[0] if requests else "*", stage=stage,
                    layer=layer, direction="write", position="tail",
                    output=bound_output or kind, storage=kind, bytes=write,
                    span=((0, retained),) if retained else (),
                    scope=scope, source=producer,
                ))
            continue
        if bool(meta.get("shared_owner", False)):
            continue
        if kind and str(getattr(op, "family", "")) == "cache_store":
            write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
            if write <= 0:
                write = float(getattr(op, "hbm_bytes", 0.0) or 0.0) / 2.0
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="write", position="cache",
                    output=bound_output or kind, storage=kind,
                    bytes=write / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
            continue
        contract = meta.get("gdn_state_contract")
        if contract:
            # The original-slot contract is the physical access authority.
            # Aggregate state_read/write_bytes are validated against a
            # per-slot formula when the producer declared one; a conv contract
            # that only declares state_write_bytes explicitly is split across
            # its real active slots instead of being dropped.
            state_dtype = str(contract.get("state_dtype")
                              or meta.get("state_dtype") or "fp32")
            heads = int(meta.get("heads", contract.get("heads", 0)) or 0)
            dk = int(meta.get("dk", contract.get("dk", 0)) or 0)
            dv = int(meta.get("dv", contract.get("dv", 0)) or 0)
            element = float(meta.get("state_element_bytes", 0.0) or 0.0)
            if not element:
                from mindie_sim.schema.common import dtype_bytes
                element = float(dtype_bytes(state_dtype))
            state_bytes = float(heads * dk * dv) * element
            read_versions = tuple(contract.get("read_versions", ()))
            checkpoints = tuple(contract.get("checkpoints", ()))
            targets = tuple(contract.get("targets", ()))
            publish_versions = tuple(contract.get("publish_versions", ()))
            slot_rows = []
            for slot, request in enumerate(raw_requests):
                if str(request) in padding_ids:
                    continue
                slot_rows.append({
                    "slot": slot, "request": request,
                    "target": (targets[slot] if slot < len(targets)
                               else (bound_output or "state")),
                    "read_version": (read_versions[slot]
                                     if slot < len(read_versions) else None),
                    "publish_version": (publish_versions[slot]
                                        if slot < len(publish_versions)
                                        else None),
                    "checkpoint": (int(checkpoints[slot])
                                   if slot < len(checkpoints) else 0),
                })
            for row in slot_rows:
                row["read"] = (state_bytes
                               if state_bytes > 0.0
                               and row["read_version"] is not None else 0.0)
                row["write"] = (state_bytes * row["checkpoint"]
                                if state_bytes > 0.0 else 0.0)
            aggregate_read = float(meta.get("state_read_bytes", 0.0) or 0.0)
            aggregate_write = float(meta.get("state_write_bytes", 0.0) or 0.0)
            if state_bytes > 0.0:
                contract_read = sum(row["read"] for row in slot_rows)
                contract_write = sum(row["write"] for row in slot_rows)
                for direction, aggregate, total in (
                        ("read", aggregate_read, contract_read),
                        ("write", aggregate_write, contract_write)):
                    if aggregate > 0.0 and abs(aggregate - total) > 1e-9:
                        raise ValueError(
                            f"state_access_contract_total_mismatch:{direction}:"
                            f"{aggregate}:{total}")
            else:
                if slot_rows:
                    if aggregate_read > 0.0:
                        for row in slot_rows:
                            row["read"] = aggregate_read / len(slot_rows)
                    if aggregate_write > 0.0:
                        for row in slot_rows:
                            row["write"] = aggregate_write / len(slot_rows)
            for row in slot_rows:
                if row["read"] > 0.0:
                    result.append(StateAccess(
                        occurrence=occurrence, producer=producer,
                        request=row["request"], stage=stage, layer=layer,
                        direction="read", position="state",
                        output=row["target"],
                        storage=str(contract.get("state_layout", "state")),
                        state_version=row["read_version"], span=((0, 1),),
                        scope=scope, bytes=row["read"], source=producer))
                if row["write"] > 0.0:
                    result.append(StateAccess(
                        occurrence=occurrence, producer=producer,
                        request=row["request"], stage=stage, layer=layer,
                        direction="write", position="state",
                        output=row["target"],
                        storage=str(contract.get("state_layout", "state")),
                        state_version=row["publish_version"],
                        span=((0, 1),), scope=scope, bytes=row["write"],
                        source=producer, decision=producer))
            continue
            continue
        read = float(meta.get("state_read_bytes", 0.0) or 0.0)
        write = float(meta.get("state_write_bytes", 0.0) or 0.0)
        if read > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="read", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_read_versions") or (None,))[0]
                    if isinstance(meta.get("state_read_versions"), (tuple, list)) else None,
                    bytes=read / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
        if write > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="write", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_commit_versions") or (None,))[0]
                    if isinstance(meta.get("state_commit_versions"), (tuple, list)) else None,
                    bytes=write / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
    return tuple(result)



def _submit_kv_tier_transfers(
    execution: Any, stem: str, replica: int, stage: int, kv_bytes: float,
    anchor: tuple[Any, ...], *, discount_hit: bool,
) -> tuple[Any, ...]:
    if kv_bytes <= 0.0:
        return ()
    policy = execution.offload_policy
    scale = (1.0 - policy.mtp_kvc_hit_rate) if discount_hit else 1.0
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    events = []
    ddr_bytes = kv_bytes * policy.kvc_ddr_ratio * scale
    if ddr_bytes > 0.0:
        ddr_res = f"{execution.namespace}offload_ddr@r{replica}:s{stage}"
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ddr", {ddr_res: ddr_bytes, hbm_res: ddr_bytes},
            after=anchor, latency_us=policy.ddr_latency_us, owner=f"replica:{replica}"))
    ssd_bytes = kv_bytes * policy.kvc_ssd_ratio * scale
    if ssd_bytes > 0.0:
        ssd_res = f"{execution.namespace}offload_ssd@r{replica}:s{stage}"
        events.append(_submit_offload_transfer(
            execution.engine, f"{stem}_ssd", {ssd_res: ssd_bytes, hbm_res: ssd_bytes},
            after=anchor, latency_us=policy.ssd_latency_us, owner=f"replica:{replica}"))
    return tuple(events)


_STATE_WRITER_FAMILIES = frozenset({
    "cache_store", "state_scatter", "state", "kv_store", "gdn_state",
    "gdn_recurrent", "recurrent_vector", "recurrent_scan",
    "short_convolution",
})


def _writer_occurrences(physical_ops) -> set[str]:
    """Ops whose own completion can be the actual state/output writer."""
    writers: set[str] = set()
    for op in physical_ops or ():
        occurrence = str(getattr(op, "op_id", ""))
        meta = dict(getattr(op, "metadata", ()) or ())
        if (str(getattr(op, "family", "")) in _STATE_WRITER_FAMILIES
                or meta.get("cache_kind")
                or float(meta.get("state_write_bytes", 0.0) or 0.0) > 0.0
                or "state_accesses" in meta
                or "memory_accesses" in meta):
            writers.add(occurrence)
    return writers


def _writeback_groups(accesses, producer_done: Any, table: Mapping | None, *,
                      explicit: bool, error: str, batch_outer: bool = False):
    """Group pending writes by their actual producer.output Event.

    A single outer completion is reused only for an explicitly selected
    single-producer write set, or when the selected occurrence itself is the
    declared producer.  Multiple producers without per-access bindings stay
    unresolved; they are not sent on one early Event.
    """
    producers = {access.producer for access in accesses}
    if (batch_outer and explicit and producer_done is not None
            and not _has_event_bindings(table)):
        # The documented draft aggregate writeback is one batch-level write
        # after the last draft op; its producer handle is the batch completion,
        # not one component label.  Per-output bindings still take precedence.
        groups: dict[tuple[Any, int], list[StateAccess]] = {}
        for access in accesses:
            groups.setdefault((producer_done, access.stage), []).append(access)
        return groups
    allow_outer = (
        producer_done is not None and len(producers) == 1
        and (explicit or all(access.occurrence == access.producer
                             for access in accesses))
    )
    groups: dict[tuple[Any, int], list[StateAccess]] = {}
    unresolved: list[StateAccess] = []
    for access in accesses:
        event = _event_for_access(access, table)
        if event is None and allow_outer:
            event = producer_done
        if event is None:
            unresolved.append(access)
            continue
        groups.setdefault((event, access.stage), []).append(access)
    if (unresolved and len(producers) > 1
            and not _has_event_bindings(table)):
        raise ValueError(error)
    return groups


def _submit_writeback_groups(execution: Any, stem: str, replica: int,
                             work: Any, groups) -> tuple[Any, ...]:
    """Submit one tier copy set per actual producer Event and register it."""
    if not groups:
        return ()
    multiple = len(groups) > 1
    events: list[Any] = []
    for index, ((event, stage), group) in enumerate(sorted(
            groups.items(), key=lambda item: (item[0][1], repr(item[0][0])))):
        write_bytes = sum(access.bytes for access in group)
        if write_bytes <= 0.0:
            continue
        name = stem if not multiple else f"{stem}:{index}"
        submitted = _submit_kv_tier_transfers(
            execution, name, replica, stage, write_bytes, (event,),
            discount_hit=False)
        _register_access_events(execution, work, group, submitted)
        events.extend(submitted)
    return tuple(events)


def schedule_kv_layer_offload(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, input_ready: Any, consumer_deps: tuple[Any, ...],
    layer_entries: Mapping[int, Any], *, prev_kv_done: Any = None,
    state_accesses: Any = None, source_ready: Any = None,
    grant_ready: Any = None, decision: Any = None,
) -> tuple[Any, ...]:
    """Submit per-layer KV cache DDR/SSD fetches before the layer's attention consumer."""
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    memory = _offload_memory_for(execution)
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "read" and access.stage == stage
                and (state_accesses is not None or access.scope == "target")
                and (access.layer == layer or access.layer < 0)]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    kv_bytes = _access_bytes(pending, "read")
    if kv_bytes <= 0.0:
        return existing
    mandatory = tuple(item for item in (source_ready, grant_ready, decision)
                      if item is not None)
    if policy.kvc_prefetch_version == "v1":
        anchor = (input_ready,) if mandatory else (tuple(consumer_deps) or (input_ready,))
    elif policy.kvc_prefetch_version == "v2":
        anchor = (layer_entries.get(layer, input_ready),)
    elif policy.kvc_prefetch_version == "v3":
        anchor = (layer_entries.get(layer - 1, input_ready),)
    else:
        anchor = (input_ready,)
    anchor = tuple(dict.fromkeys((*anchor, *mandatory)))
    if prev_kv_done is not None:
        anchor = tuple(dict.fromkeys((*anchor, prev_kv_done)))
    events = _submit_kv_tier_transfers(
        execution, f"{prefix}:s{stage}.l{layer}:kv_offload", replica, stage,
        kv_bytes, anchor, discount_hit=True)
    _register_access_events(execution, work, pending, events)
    return tuple(dict.fromkeys((*existing, *events)))


def schedule_kv_layer_writeback(
    execution: Any, prefix: str, replica: int, stage: int, layer: int,
    work: Any, producer_done: Any, *, state_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Submit per-layer KV/state writebacks to DDR/SSD after their writers.

    Each pending physical write uses the same batch's actual
    producer.output->Event binding.  A single outer completion is reused only
    for one explicitly selected producer.  A write whose real output Event is
    not available is left unissued and unregistered for a later supplement.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "write" and access.stage == stage
                and access.scope == "target"
                and (access.layer == layer or access.layer < 0)]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    if not pending:
        return existing
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    groups = _writeback_groups(
        pending, producer_done, table,
        explicit=state_accesses is not None,
        error="state_writeback_requires_one_producer_event_per_call")
    events = _submit_writeback_groups(
        execution, f"{prefix}:s{stage}.l{layer}:kv_writeback", replica,
        work, groups)
    return tuple(dict.fromkeys((*existing, *events)))


def schedule_draft_kv_writeback(
    execution: Any, prefix: str, replica: int, stage: int,
    work: Any, producer_done: Any, *, state_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Write actual draft-materialized rows after the draft producer completes.

    This is a separate producer identity from the target layer with the same
    numeric index.  T04 should call it once per executed batch after the last
    draft op completes, not once per target layer.  ``proposal=0`` and a
    disabled prefill preparation produce no record and no event.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    accesses = _unique_accesses(_as_accesses(state_accesses) if state_accesses is not None
                                else _accesses_for_work(execution, work))
    selected = [access for access in accesses
                if access.direction == "write" and access.stage == stage
                and access.scope == "draft"]
    registry = _access_registry(execution, work)
    existing = _registered_events(execution, work, selected)
    pending = [access for access in selected if access.key not in registry]
    if not pending:
        return existing
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    groups = _writeback_groups(
        pending, producer_done, table,
        explicit=state_accesses is not None,
        error="draft_writeback_requires_one_producer_event_per_call",
        batch_outer=True)
    events = _submit_writeback_groups(
        execution, f"{prefix}:s{stage}:draft_kv_writeback", replica,
        work, groups)
    return tuple(dict.fromkeys((*existing, *events)))



def schedule_kv_iteration_offload(
    execution: Any, prefix: str, replica: int, work: Any, input_ready: Any,
    physical_ops: tuple[Any, ...] | list[Any], ids: Mapping[str, Any],
    scheduled_layers: set[tuple[int, int]] | None = None, *,
    state_accesses: Any = None, scheduled_accesses: Any = None,
    access_events: Any = None,
) -> tuple[Any, ...]:
    """Submit fallback writebacks for real, still-unissued physical writes.

    The fallback uses the same per-work issued registry and the same actual
    access->producer.output Event bindings as the layer/draft paths.  It never
    substitutes an unrelated outer occurrence ID or matches Event.name.
    """
    policy = getattr(execution, "offload_policy", None)
    if (policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0
            or not work.plan.requests):
        return ()
    registry = _access_registry(execution, work)
    if state_accesses is not None:
        accesses = _unique_accesses(_as_accesses(state_accesses))
    else:
        base = _accesses_for_work(execution, work)
        allowed = {str(getattr(op, "op_id", "")) for op in physical_ops}
        accesses = tuple(access for access in base
                         if access.occurrence in allowed)
    table = _merge_event_tables(
        _work_event_table(execution, work), _event_table(access_events))
    submitted = {str(getattr(op, "op_id", "")) for op in physical_ops}
    writers = _writer_occurrences(physical_ops)
    done = set(scheduled_accesses or ()) | set(registry)
    pending = [access for access in accesses
               if access.key not in done and access.direction == "write"]
    pp = max(1, int(execution.case.runtime.parallel.pp))
    groups: dict[tuple[Any, int], list[StateAccess]] = {}
    for access in pending:
        event = _event_for_access(access, table)
        if event is None and access.producer in ids:
            event = ids.get(access.producer)
        if event is None and access.producer not in submitted:
            # The declared producer is a port not submitted as its own op.
            # Only the occurrence that is itself a real state writer can stand
            # for that output; a generic call/outer ID cannot.
            if (access.occurrence == access.producer
                    or access.occurrence in writers):
                event = ids.get(access.occurrence)
        if event is None:
            # Real unissued write without its own producer.output completion:
            # preserve it for the supplement, do not emit or register it.
            continue
        stage = min(max(0, access.stage), pp - 1)
        groups.setdefault((event, stage), []).append(access)

    events: list[Any] = []
    multiple = len(groups) > 1
    for index, ((producer_event, stage), group) in enumerate(sorted(
            groups.items(), key=lambda item: (
                item[0][1], repr(item[0][0])))):
        write_bytes = sum(access.bytes for access in group)
        if write_bytes <= 0.0:
            continue
        stem = f"{prefix}:s{stage}:kv_writeback"
        if multiple:
            stem = f"{stem}:{index}"
        submitted_events = _submit_kv_tier_transfers(
            execution, stem, replica, stage, write_bytes, (producer_event,),
            discount_hit=False)
        _register_access_events(execution, work, group, submitted_events)
        events.extend(submitted_events)
    return tuple(events)



def project_state_tiers(
    accesses: Any, policy: OffloadPolicy, *, expected_bytes: float | None = None,
) -> tuple[dict, ...]:
    """Project bound accesses into per-occurrence issued tier bytes.

    The returned records preserve request/rank/occurrence/output/storage/
    version/span/direction and the source/reader/grant/decision/last-use
    references.  No model layer or whole-request total is consulted.
    """
    bound = _unique_accesses(_as_accesses(accesses))
    if expected_bytes is not None:
        if (isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, (int, float))
                or not math.isfinite(float(expected_bytes))
                or float(expected_bytes) < 0.0):
            raise ValueError("state_access_total_bytes_invalid")
        if abs(float(expected_bytes) - sum(access.bytes for access in bound)) > 1e-9:
            raise ValueError("state_access_total_bytes_mismatch")
    records = []
    for access in bound:
        read = access.direction == "read"
        scale = (1.0 - policy.mtp_kvc_hit_rate) if read else 1.0
        ddr = access.bytes * policy.kvc_ddr_ratio * scale
        ssd = access.bytes * policy.kvc_ssd_ratio * scale
        records.append({
            **access.to_dict(),
            "ddr_bytes": ddr, "ssd_bytes": ssd, "hbm_bytes": ddr + ssd,
        })
    return tuple(records)


__all__ = [
    "OffloadPolicy",
    "resolve_case_offload_policy", "offload_resources", "schedule_expert_layer_offload",
    "schedule_kv_layer_offload", "schedule_kv_layer_writeback",
    "schedule_draft_kv_writeback", "schedule_kv_iteration_offload",
    "StateAccess", "project_state_tiers", "release_offload_access",
    "bind_offload_access_events",
]
