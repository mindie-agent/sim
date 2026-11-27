"""Scoped routed-expert receipts for native grouped operators.

[Role] Project declared rank and layer routing data into the receipt shared
  by the two expert matrix multiplies.
[Invariants] Local expert counts and received rows retain their original scope.
  Capacity is separate from valid rows, and unknown ownership stays symbolic.
[Timeouts & Locks] Bounded by the declared routing input, with no I/O or locks.
[Side-Effect Policy] Returns immutable receipts from the supplied routing data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_VLLM_ASCEND_REVISION = "175ef4d49c2072f9233619636f0309f4f88b4e2c"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/token_dispatcher.py",
        source_sha256="6126304014959e8c52d87284ccac078f7622a6d72c7a7fc8be25929dfe644cfb",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision=_VLLM_ASCEND_REVISION,
        relative_path="vllm_ascend/ops/fused_moe/moe_mlp.py",
        source_sha256="6b5f4f6d3259cb14e0613a6f7b1c2592331f143a14cf7240dd9d49a0a57dd53b",
    ),
)


@dataclass(frozen=True)
class ExpertReceipt:
    """One current-rank routed-expert receipt shared by native GMM1/GMM2.

    ``counts``/``group_list`` are local-expert values; ``received_rows`` is the
    actual local activation R_rank only when an explicit scoped source declared
    it; ``upper_bound`` is the separately declared capacity.  A global count
    vector without the current rank's local ids stays symbolic, never sliced.
    """

    layer: int
    rank_index: int | None
    ep: int
    local_expert_ids: tuple[int, ...] | None
    counts: tuple[int, ...] | None
    group_list: tuple[int, ...] | None
    valid_rows: int | None
    received_rows: int | None
    upper_bound: int | None
    activation_dtype: str
    scale_dtype: str
    group_list_dtype: str
    basis: str
    symbolic: bool
    ownership_known: bool
    input_group_list: tuple[int, ...] | None = None
    input_group_list_type: int | None = None
    gmm1_group_list: tuple[int, ...] | None = None
    gmm1_group_list_type: int | None = 0
    gmm2_group_list: tuple[int, ...] | None = None
    gmm2_group_list_type: int | None = None
    gmm1_shape_complete: bool = False
    gmm2_shape_complete: bool = False
    conflicts: tuple[str, ...] = ()

    @property
    def shape_complete(self) -> bool:
        return self.gmm1_shape_complete and self.gmm2_shape_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rank_index": self.rank_index,
            "ep": self.ep,
            "local_expert_ids": (list(self.local_expert_ids)
                                 if self.local_expert_ids is not None else None),
            "counts": list(self.counts) if self.counts is not None else None,
            "group_list": (list(self.group_list)
                           if self.group_list is not None else None),
            "group_list_type": 0,
            "canonical_counts": list(self.counts) if self.counts is not None else None,
            "canonical_group_list": (list(self.gmm1_group_list)
                                     if self.gmm1_group_list is not None else None),
            "canonical_group_list_type": 0,
            "input_group_list": (list(self.input_group_list)
                                 if self.input_group_list is not None else None),
            "input_group_list_type": self.input_group_list_type,
            "input_group_list_layout": ("per_expert"
                                        if self.input_group_list_type == 1 else
                                        "cumulative"
                                        if self.input_group_list_type == 0 else None),
            "gmm1_group_list": (list(self.gmm1_group_list)
                                if self.gmm1_group_list is not None else None),
            "gmm1_group_list_type": self.gmm1_group_list_type,
            "gmm2_group_list": (list(self.gmm2_group_list)
                                if self.gmm2_group_list is not None else None),
            "gmm2_group_list_type": self.gmm2_group_list_type,
            "gmm1_shape_complete": self.gmm1_shape_complete,
            "gmm2_shape_complete": self.gmm2_shape_complete,
            "valid_rows": self.valid_rows,
            "received_rows": self.received_rows,
            "upper_bound": self.upper_bound,
            "activation_dtype": self.activation_dtype,
            "scale_dtype": self.scale_dtype,
            "group_list_dtype": self.group_list_dtype,
            "basis": self.basis,
            "symbolic": self.symbolic,
            "ownership_known": self.ownership_known,
            "shape_complete": self.shape_complete,
            "conflicts": list(self.conflicts),
        }


def _receipt_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _receipt_ints(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(item < 0 for item in values):
        return None
    return values


def _receipt_vector(raw: Any, layer: int,
                    rank: int | None) -> tuple[int, ...] | None:
    """Current-layer vector: flat tuple or layer/rank-keyed mapping."""
    if isinstance(raw, Mapping):
        if layer in raw:
            return _receipt_ints(raw[layer])
        if rank is not None and rank in raw:
            return _receipt_ints(raw[rank])
        return None
    return _receipt_ints(raw)


def _receipt_source_type(value: Any) -> int | None:
    if isinstance(value, str):
        name = value.strip().lower()
        if name in {"0", "cumulative", "prefix"}:
            return 0
        if name in {"1", "per_expert", "counts"}:
            return 1
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return value
    return None


def _receipt_ids(raw: Any, layer: int, rank: int | None) -> tuple[int, ...] | None:
    """Current-layer ids: flat tuple or layer/rank-keyed mapping, never a scalar."""
    if isinstance(raw, Mapping):
        if layer in raw:
            return _receipt_ints(raw[layer])
        if rank is not None and rank in raw:
            return _receipt_ints(raw[rank])
        return None
    return _receipt_ints(raw)


def _receipt_layer_value(raw: Any, layer: int,
                         rank: int | None = None) -> Any:
    """Resolve a per-layer (or rank->per-layer) explicit source value."""
    if isinstance(raw, Mapping):
        if layer in raw:
            value = raw[layer]
            if (isinstance(value, Mapping) and rank is not None
                    and rank in value):
                return value[rank]
            if (isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    and rank is not None and 0 <= rank < len(value)
                    and isinstance(value[rank], int)
                    and not isinstance(value[rank], bool)):
                return value[rank]
            return value
        if rank is not None and rank in raw:
            value = raw[rank]
            if isinstance(value, Mapping):
                return value.get(layer)
            if (isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes))
                    and 0 <= layer < len(value)):
                return value[layer]
        return None
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if 0 <= layer < len(raw):
            return raw[layer]
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _resolve_local_routing(
    state, geometry, explicit, layer, local_experts, experts, ep, conflicts,
):
    record = None
    if state is not None:
        routing_state = getattr(state, "routing_state", None)
        if callable(routing_state):
            try:
                record = routing_state(layer)
            except ModelingError:
                record = None
        elif hasattr(state, "counts"):
            record = state
    state_counts = _receipt_ints(getattr(record, "counts", None))
    state_ids = _receipt_ints(getattr(record, "local_expert_ids", None))

    geometry_counts = None
    routing_for = getattr(geometry, "routing_for", None)
    if callable(routing_for):
        try:
            geometry_counts = _receipt_ints(routing_for(layer))
        except ModelingError:
            geometry_counts = None

    rank = _receipt_int(explicit.get("rank_index",
                                     explicit.get("current_rank")))
    explicit_ids = _receipt_ids(explicit.get("local_expert_ids"), layer, rank)
    if explicit_ids is not None and state_ids is not None \
            and tuple(explicit_ids) != tuple(state_ids):
        conflicts.append("state_explicit_local_expert_ids_disagree")
    local_ids = explicit_ids if explicit_ids is not None else state_ids

    if local_experts < 1:
        local_experts = (len(local_ids) if local_ids is not None
                         else len(state_counts) if state_counts is not None
                         else len(geometry_counts) if geometry_counts is not None else 0)
    if experts < 1:
        experts = (len(geometry_counts) if geometry_counts is not None
                   else len(state_counts) if state_counts is not None else local_experts)
    if ep < 1:
        ep = (experts // local_experts
              if local_experts > 0 and experts % local_experts == 0 else 0)
    if local_ids is not None and len(local_ids) != local_experts:
        conflicts.append("local_expert_ids_length_mismatch")
        local_ids = None

    if state_counts is not None and geometry_counts is not None:
        compared_geometry = geometry_counts
        if (len(state_counts) == local_experts
                and len(geometry_counts) == experts
                and local_ids is not None
                and not any(index >= len(geometry_counts) for index in local_ids)):
            compared_geometry = tuple(geometry_counts[index] for index in local_ids)
        if tuple(state_counts) != tuple(compared_geometry):
            conflicts.append("state_geometry_counts_disagree")
    counts = state_counts if state_counts is not None else geometry_counts
    basis = ("state_routing" if state_counts is not None
             else "geometry_routing" if geometry_counts is not None
             else "unknown")
    if "state_geometry_counts_disagree" in conflicts:
        counts = None
        basis = "state_geometry_conflict"

    local_counts = None
    ownership_known = False
    if counts is not None and local_experts > 0:
        if basis == "state_routing" and len(counts) == local_experts:
            local_counts = counts
            ownership_known = local_ids is not None
        elif len(counts) == experts:
            if local_ids is not None:
                if any(index >= len(counts) for index in local_ids):
                    conflicts.append("local_expert_ids_out_of_range")
                else:
                    local_counts = tuple(counts[index] for index in local_ids)
                    ownership_known = True
            else:
                conflicts.append("global_counts_without_local_expert_ids")
        elif len(counts) == local_experts:
            if local_ids is not None:
                local_counts = counts
                ownership_known = True
            else:
                conflicts.append("geometry_local_counts_without_local_expert_ids")
        else:
            conflicts.append("routing_count_length_mismatch")
    return local_ids, rank, ep, local_counts, ownership_known, basis


def _build_expert_receipt(
    *,
    layer: int,
    geometry: Any = None,
    state: Any = None,
    local_experts: int = 0,
    experts: int = 0,
    ep: int = 0,
    explicit: Mapping[str, Any] | None = None,
    activation_dtype: str = "",
    scale_dtype: str = "",
    group_list_dtype: str = "int64",
    expected_scope: str = "",
) -> ExpertReceipt:
    """Project the one current-rank receipt used by GMM1 and GMM2.

    Counts come from ``StateView.routing_state(layer)``/``MoeRoutingState`` when
    present, else ``WorkloadGeometry.routing_for(layer)``.  A global vector is
    accepted only with the current rank's explicit ``local_expert_ids``; it is
    never sliced by rank or stride.  ``received_rows`` and ``upper_bound`` come
    only from the explicit scoped source and must not contradict the counts.
    """
    explicit = dict(explicit or {})
    layer = int(layer)
    conflicts: list[str] = []
    explicit_layer = _receipt_int(explicit.get("layer"))
    if explicit_layer is not None and explicit_layer != layer:
        conflicts.append("explicit_layer_mismatch")
    scope = str(explicit.get("scope") or "")
    if expected_scope and scope and scope != expected_scope:
        conflicts.append("explicit_scope_mismatch")

    local_ids, rank, ep, local_counts, ownership_known, basis = _resolve_local_routing(
        state, geometry, explicit, layer, local_experts, experts, ep, conflicts,
    )

    (
        source_type, valid, gmm1_group_list, gmm1_group_list_type,
        input_group_list, gmm2_group_list, gmm2_group_list_type,
    ) = _resolve_group_lists(explicit, layer, rank, local_counts, conflicts)

    received = _receipt_int(_receipt_layer_value(
        explicit.get("received_rows"), layer, rank))
    capacity = _receipt_int(_receipt_layer_value(
        explicit.get("capacity_rows", explicit.get("capacity")), layer, rank))
    if received is not None and valid is not None and received < valid:
        conflicts.append("received_rows_below_valid_rows")
        received = None
    if capacity is not None and (
            (received is not None and capacity < received)
            or (valid is not None and capacity < valid)):
        conflicts.append("capacity_below_received_or_valid_rows")
        capacity = None
    if "explicit_layer_mismatch" in conflicts or "explicit_scope_mismatch" in conflicts:
        local_counts = None
        gmm1_group_list = None
        gmm1_group_list_type = None
        input_group_list = None
        gmm2_group_list = None
        gmm2_group_list_type = None
        valid = None
        received = None
        capacity = None

    gmm1_complete = (received is not None and gmm1_group_list is not None
                     and not conflicts)
    gmm2_complete = (received is not None and gmm2_group_list is not None
                     and gmm2_group_list_type is not None and not conflicts)
    if conflicts and (gmm1_group_list is None or gmm2_group_list is None):
        basis = conflicts[0]
    elif gmm1_group_list is None:
        basis = "group_list_not_declared"
    elif gmm2_group_list is None or gmm2_group_list_type is None:
        basis = "input_group_list_type_not_declared"
    return ExpertReceipt(
        layer=layer, rank_index=rank, ep=ep,
        local_expert_ids=local_ids,
        counts=local_counts,
        group_list=gmm1_group_list,
        valid_rows=valid,
        received_rows=received,
        upper_bound=capacity,
        input_group_list=input_group_list,
        input_group_list_type=source_type,
        gmm1_group_list=gmm1_group_list,
        gmm1_group_list_type=gmm1_group_list_type,
        gmm2_group_list=gmm2_group_list,
        gmm2_group_list_type=gmm2_group_list_type,
        gmm1_shape_complete=gmm1_complete,
        gmm2_shape_complete=gmm2_complete,
        activation_dtype=str(activation_dtype or ""),
        scale_dtype=str(scale_dtype or ""),
        group_list_dtype=str(group_list_dtype or "int64"),
        basis=basis,
        symbolic=not (gmm1_complete and gmm2_complete),
        ownership_known=ownership_known,
        conflicts=tuple(dict.fromkeys(conflicts)),
    )


def _resolve_group_lists(explicit, layer, rank, local_counts, conflicts):
    source_type_raw = explicit.get(
        "input_group_list_type",
        explicit.get("moe_input_group_list_type",
                     explicit.get("moe_group_list_type",
                                  explicit.get("group_list_type"))))
    source_type = _receipt_source_type(source_type_raw)
    if source_type_raw is not None and source_type is None:
        conflicts.append("input_group_list_type_invalid")
    source_vector = _receipt_vector(
        explicit.get("input_group_list",
                     explicit.get("moe_input_group_list")), layer, rank)

    canonical_prefix = None
    valid = None
    if local_counts is not None:
        running = 0
        values: list[int] = []
        for value in local_counts:
            running += int(value)
            values.append(running)
        canonical_prefix = tuple(values)
        valid = canonical_prefix[-1]

    gmm1_group_list = canonical_prefix
    gmm1_group_list_type = 0 if gmm1_group_list is not None else None
    input_group_list = source_vector
    gmm2_group_list = None
    gmm2_group_list_type = None
    if local_counts is not None:
        if source_type == 1:
            if source_vector is None:
                source_vector = canonical_prefix if canonical_prefix is not None else None
                # type 1 input is the per-expert counts, not the prefix
                source_vector = local_counts
            if tuple(source_vector) != tuple(local_counts):
                conflicts.append("input_per_expert_group_list_disagrees")
            else:
                input_group_list = local_counts
                gmm2_group_list = local_counts
                gmm2_group_list_type = 1
        elif source_type == 0:
            if source_vector is None:
                source_vector = canonical_prefix
            if canonical_prefix is not None and (
                    len(source_vector) != len(canonical_prefix)
                    or any(source_vector[index] > source_vector[index + 1]
                           for index in range(len(source_vector) - 1))
                    or any(source_vector[index] - (source_vector[index - 1]
                                                   if index else 0)
                           != local_counts[index]
                           for index in range(len(local_counts)))):
                conflicts.append("input_cumulative_group_list_disagrees")
            else:
                input_group_list = source_vector
                gmm2_group_list = source_vector
                gmm2_group_list_type = 0
        else:
            if source_vector is not None:
                input_group_list = source_vector
                gmm2_group_list = source_vector
                gmm2_group_list_type = None
    return (
        source_type, valid, gmm1_group_list, gmm1_group_list_type,
        input_group_list, gmm2_group_list, gmm2_group_list_type,
    )



def _compiled_receipt_terms(compiled: Any) -> dict[str, Any] | None:
    """Extract only the shared GMM1/GMM2 source facts from one compiled node."""
    operator = getattr(compiled, "operator", None)
    theta = getattr(operator, "theta", None)
    if theta is None:
        return None
    extras = dict(getattr(theta, "extras", {}) or ())
    quant = dict(getattr(theta, "quant", {}) or ())
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())

    def pick(*keys: str) -> Any:
        for key in keys:
            for source in (extras, meta):
                if key in source and source[key] is not None:
                    return source[key]
        return None

    local = _receipt_int(pick("E_local", "local")) or 0
    experts = _receipt_int(pick("E")) or 0
    if local < 1:
        ids = _receipt_ids(pick("moe_local_expert_ids"),
                           int(getattr(compiled, "layer", 0) or 0),
                           _receipt_int(pick("moe_rank_index", "rank_index")))
        local = len(ids) if ids is not None else 0
    if experts < 1:
        experts = local
    ep = (experts // local
          if local > 0 and experts > 0 and experts % local == 0 else 0)
    explicit = {
        "layer": int(getattr(compiled, "layer", 0) or 0),
        "rank_index": pick("moe_rank_index", "rank_index"),
        "local_expert_ids": pick("moe_local_expert_ids"),
        "received_rows": pick("moe_received_rows", "dispatch_rows",
                              "received_rows", "router_received_rows"),
        "capacity_rows": pick("moe_capacity_rows", "capacity_rows"),
        "input_group_list": pick("moe_input_group_list", "input_group_list"),
        "input_group_list_type": pick(
            "moe_input_group_list_type", "input_group_list_type",
            "moe_group_list_type"),
        "group_list_type": pick("moe_group_list_type"),
        "scope": pick("moe_route_scope", "native_scope"),
    }
    return {
        "local_experts": local, "experts": experts, "ep": ep,
        "explicit": explicit,
        "activation_dtype": str(quant.get("output_format")
                                or quant.get("act_format") or ""),
        "scale_dtype": str(quant.get("weight_scale_format")
                           or quant.get("x_scale_format")
                           or quant.get("weight_scale_storage_format") or ""),
    }


def project_expert_receipt(
    compiled: Any,
    geometry: Any,
    *,
    state: Any = None,
    activation_dtype: str | None = None,
    scale_dtype: str | None = None,
    expected_scope: str = "",
) -> dict[str, Any] | None:
    """Public GMM1/GMM2 receipt: one R_rank, group values and upper bound.

    Returns the shared dictionary (no private schema marker) or ``None`` for a
    compiled node that is not the routed-expert grouped source.  Counts come
    from ``StateView.routing_state(layer).counts``/``MoeRoutingState`` when
    present, else ``WorkloadGeometry.routing_for(layer)``.  A global vector is
    never sliced; it needs the current rank's explicit ``local_expert_ids``.
    """
    terms = _compiled_receipt_terms(compiled)
    if terms is None:
        return None
    receipt = _build_expert_receipt(
        layer=int(getattr(compiled, "layer", 0) or 0),
        geometry=geometry, state=state,
        local_experts=terms["local_experts"], experts=terms["experts"],
        ep=terms["ep"], explicit=terms["explicit"],
        activation_dtype=(terms["activation_dtype"]
                          if activation_dtype is None else activation_dtype),
        scale_dtype=(terms["scale_dtype"]
                     if scale_dtype is None else scale_dtype),
        group_list_dtype="int64", expected_scope=expected_scope)
    unresolved = list(receipt.conflicts)
    if receipt.received_rows is None:
        unresolved.append("received_rows_not_declared")
    if receipt.gmm1_group_list is None:
        unresolved.append(receipt.basis)
    if receipt.input_group_list is None:
        unresolved.append("input_group_list_not_declared")
    if receipt.input_group_list_type is None:
        unresolved.append("input_group_list_type_not_declared")
    if receipt.gmm2_group_list is None:
        unresolved.append("gmm2_group_list_not_declared")
    if receipt.gmm2_group_list_type is None:
        unresolved.append("gmm2_group_list_type_not_declared")
    if not receipt.ownership_known:
        unresolved.append("local_expert_ids_not_declared")
    if receipt.upper_bound is None:
        unresolved.append("upper_bound_not_declared")
    return {
        "layer": receipt.layer,
        "rank": receipt.rank_index,
        "ep": receipt.ep,
        "received_rows": receipt.received_rows,
        "input_rows": receipt.received_rows,
        "group_list": (list(receipt.group_list)
                       if receipt.group_list is not None else None),
        "group_list_layout": "cumulative",
        "group_list_type": 0,
        "canonical_counts": (list(receipt.counts)
                             if receipt.counts is not None else None),
        "canonical_group_list": (list(receipt.gmm1_group_list)
                                 if receipt.gmm1_group_list is not None else None),
        "canonical_group_list_type": 0,
        "input_group_list": (list(receipt.input_group_list)
                             if receipt.input_group_list is not None else None),
        "input_group_list_type": receipt.input_group_list_type,
        "input_group_list_layout": (
            "per_expert" if receipt.input_group_list_type == 1 else
            "cumulative" if receipt.input_group_list_type == 0 else None),
        "gmm1_group_list": (list(receipt.gmm1_group_list)
                            if receipt.gmm1_group_list is not None else None),
        "gmm1_group_list_type": receipt.gmm1_group_list_type,
        "gmm2_group_list": (list(receipt.gmm2_group_list)
                            if receipt.gmm2_group_list is not None else None),
        "gmm2_group_list_type": receipt.gmm2_group_list_type,
        "gmm1_shape_complete": receipt.gmm1_shape_complete,
        "gmm2_shape_complete": receipt.gmm2_shape_complete,
        "local_expert_ids": (list(receipt.local_expert_ids)
                             if receipt.local_expert_ids is not None else None),
        "expert_rows": (list(receipt.counts)
                        if receipt.counts is not None else None),
        "capacity_rows": receipt.upper_bound,
        "capacity_basis": receipt.basis,
        "activation_dtype": receipt.activation_dtype,
        "scale_dtype": receipt.scale_dtype,
        "group_list_dtype": receipt.group_list_dtype,
        "basis": receipt.basis,
        "symbolic": receipt.symbolic,
        "ownership_known": receipt.ownership_known,
        "shape_complete": receipt.shape_complete,
        "complete": False,
        "valid_rows": receipt.valid_rows,
        "unresolved": list(dict.fromkeys(unresolved)),
    }
