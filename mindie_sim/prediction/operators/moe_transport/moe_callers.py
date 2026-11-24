"""MoE caller policies and the explicit local-expert reduce caller.

[Role] Declare expected routing and the vllm-ascend ALLGATHER dp=1 caller branch.
[Invariants] The local expert bank filters replicated rows; it owns no exchange,
  token slice, transport quantization or MC2 lowering. Valid routing rows and
  eager allocated capacity stay separate. Explicit integer counts are preserved;
  probability moments remain expected values. Provenance is metadata only.
[Timeouts & Locks] O(experts + tokens) validation per layer; no I/O or locks.
[Side-Effect Policy] Immutable policies; constructs no work outside the builder.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import (
    EPLB_VARIANCE_DAMPING,
    ORDER_STAT_M2_Z,
    ModelingError,
    dtype_bytes,
    normal_max_order_z,
    normal_max_order_z as _normal_max_order_z,
)
from .moe_callers_receipts import (
    ExpertReceipt,
    _build_expert_receipt,
    _compiled_receipt_terms,
    _receipt_ids,
    _receipt_int,
    _receipt_ints,
    _receipt_layer_value,
    _receipt_source_type,
    _receipt_vector,
    project_expert_receipt,
)


LOCAL_EXPERT_REDUCE = "local_expert_reduce"
EXPLICIT_TOKEN_ROUTING = "explicit_token_routing_integer"
_DECLARED_LEDGER_BASES = {
    "forced_balanced_integer_counts", "declared_integer_counts",
    EXPLICIT_TOKEN_ROUTING,
}
_FLOAT_EXPERT_FORMATS = {"bf16": "bf16", "bfloat16": "bf16",
                         "fp16": "fp16", "float16": "fp16"}
_W8A8_FORMATS = {"int8": "int8", "w8a8": "int8"}
_FULL_EAGER_GROUPED = "full_eager_allocation"
_ROUTING_INDEX_BYTES = int(dtype_bytes("int32"))
_RAW_EXPERT_COUNT_BYTES = int(dtype_bytes("int32"))
_CAST_EXPERT_COUNT_BYTES = int(dtype_bytes("int64"))
_STRAGGLER_ORDER_EPS = 1e-12


def _eplb_damping_active(parallel, runtime=None) -> bool:
    """True when redundant EPLB or force-EPLB dampens EP rank variance."""
    if int(getattr(parallel, "redundant_experts_per_rank", 0) or 0) > 0:
        return True
    if runtime is None:
        return False
    if getattr(runtime, "force_eplb", False) or getattr(runtime, "enable_eplb", False):
        return True
    kernels = getattr(runtime, "kernels", None) or {}
    if isinstance(kernels, Mapping) and kernels.get("force_eplb_source"):
        return True
    routing = getattr(runtime, "routing", None) or {}
    if isinstance(routing, Mapping) and (
        routing.get("force_eplb")
        or routing.get("enable_eplb")
        or routing.get("eplb")
        or routing.get("expert_slot_map")
    ):
        return True
    meta = dict(getattr(runtime, "metadata", ()) or ())
    if meta.get("force_eplb") or meta.get("enable_eplb"):
        return True
    additional = meta.get("additional_config")
    if isinstance(additional, Mapping):
        if additional.get("force_eplb") or additional.get("enable_eplb"):
            return True
        eplb_cfg = additional.get("eplb_config")
        if isinstance(eplb_cfg, Mapping) and (
            eplb_cfg.get("dynamic_eplb")
            or int(eplb_cfg.get("num_redundant_experts", 0) or 0) > 0
        ):
            return True
    return False


def _straggler_rank_moments(
    *,
    tokens: int,
    top_k: int,
    num_experts: int,
    ep: int,
    total_rows: float,
    total_active: float,
    mean_rows: float,
    local_active: Sequence[float],
    local_expert_count: int,
    eplb_damped: bool,
    pool_experts: int | None = None,
) -> tuple[float, float]:
    """Closed-form O(E) order-statistic slowest EP rank rows and active banks."""
    if tokens <= 0 or top_k <= 0 or total_rows <= 0.0 or local_expert_count <= 0:
        return 0.0, 0.0
    if ep <= 1:
        return mean_rows, math.fsum(local_active)
    pool = max(1, int(pool_experts if pool_experts is not None else num_experts))
    variance_scale = EPLB_VARIANCE_DAMPING if eplb_damped else 1.0
    rank_share = min(1.0, max(0.0, mean_rows / max(1.0, total_rows)))
    fpc = max(0.0, 1.0 - float(top_k - 1) / float(max(1, pool - 1)))
    sigma_rows = math.sqrt(max(
        0.0,
        total_rows * rank_share * (1.0 - rank_share) * fpc * variance_scale,
    ))
    active_ranks = min(ep, max(1, math.ceil(total_rows - _STRAGGLER_ORDER_EPS)))
    z_order = normal_max_order_z(active_ranks)
    row_cap = float(tokens * min(top_k, local_expert_count))
    if total_rows < 1.0:
        # Exact indicator lower bound when zero_experts shrinks N_eff < 1.0:
        # E[max_r R_r] >= P(at least one real assignment) = 1 - (1 - p_real)^(tokens * top_k).
        n_draws = max(1, int(tokens) * int(top_k))
        p_real = min(1.0, max(0.0, total_rows / float(n_draws)))
        p_any = 1.0 - (1.0 - p_real) ** n_draws
        row_lower = max(mean_rows, p_any)
        row_upper = min(total_rows, row_cap)
        r_max = min(row_upper, max(row_lower, mean_rows + z_order * sigma_rows))
    else:
        row_lower = float(math.ceil(total_rows / float(ep) - _STRAGGLER_ORDER_EPS))
        row_upper = min(float(math.ceil(total_rows - _STRAGGLER_ORDER_EPS)), row_cap)
        r_max = float(min(
            int(row_upper),
            max(int(row_lower), int(round(mean_rows + z_order * sigma_rows))),
        ))
    mu_act = math.fsum(local_active)
    var_act = (
        math.fsum(a * (1.0 - a) for a in local_active)
        * (1.0 - 1.0 / float(max(2, ep)))
        * variance_scale
    )
    sigma_act = math.sqrt(max(0.0, var_act))
    if total_active < 1.0:
        n_draws = max(1, int(tokens) * int(top_k))
        p_real = min(1.0, max(0.0, total_rows / float(n_draws)))
        p_any = 1.0 - (1.0 - p_real) ** n_draws
        act_lower = min(r_max, max(mu_act, p_any))
    else:
        act_lower = max(1.0, float(math.ceil(total_active / float(ep) - _STRAGGLER_ORDER_EPS)))
    a_max = min(
        float(local_expert_count),
        r_max,
        max(act_lower, mu_act + z_order * sigma_act),
    )
    return r_max, a_max


def _category_format(quant, category):
    return quant.for_category(category).format_for(category)


def _expert_path(model):
    """Float experts stay on the original caller. W8A8 is a separate legal path.

    Routed int8 or w8a8 with a floating activation is accepted when the shared
    expert is that same activation float or the same int8. Shared int8 beside
    floating routed experts, mixed floats, and every other quantized format
    stay rejected. The recipe still has to be selected explicitly.
    """
    activation = _FLOAT_EXPERT_FORMATS.get(str(model.activation_dtype).strip().lower())
    if activation is None:
        raise ModelingError(
            f"local_expert_reduce_requires_floating_activation_dtype:{model.activation_dtype}")
    routed_shown = _category_format(model.quant, "routed_expert")
    shared_shown = _category_format(model.quant, "shared_expert")
    routed_f = _FLOAT_EXPERT_FORMATS.get(str(routed_shown).strip().lower())
    shared_f = _FLOAT_EXPERT_FORMATS.get(str(shared_shown).strip().lower())
    routed_q = _W8A8_FORMATS.get(str(routed_shown).strip().lower())
    shared_q = _W8A8_FORMATS.get(str(shared_shown).strip().lower())
    if routed_f is not None and shared_f is not None:
        if routed_f != activation:
            raise ModelingError(
                "local_expert_reduce_rejects_mixed_expert_format:"
                f"routed_expert:{routed_f}!={activation}")
        if shared_f != activation:
            raise ModelingError(
                "local_expert_reduce_rejects_mixed_expert_format:"
                f"shared_expert:{shared_f}!={activation}")
        return "float"
    if routed_q == "int8" and (shared_f == activation or shared_q == "int8"):
        return "w8a8"
    if routed_f is None:
        raise ModelingError(
            "local_expert_reduce_rejects_quantized_expert_format:"
            f"routed_expert:{routed_shown}")
    raise ModelingError(
        "local_expert_reduce_rejects_quantized_expert_format:"
        f"shared_expert:{shared_shown}")


# Source identities describe the selected branch. They are never a timing input.
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "committed",
    "branch": ("MoECommType.ALLGATHER, dp=1, ep=tp, etp=1, cp=pcp=1, "
               "is_sequence_parallel=False, no routed output transform"),
    "files": {
        "vllm_ascend/ops/fused_moe/token_dispatcher.py": (
            "6126304014959e8c52d87284ccac078f7622a6d72c7a7fc8be25929dfe644cfb",
            (358, 461), "TokenDispatcherWithAllGather"),
        "vllm_ascend/ops/fused_moe/prepare_finalize.py": (
            "88560c839172f5b433753c796d17a5d41278204f33b2e9ca9d4b2af9ccbe16d8",
            (318, 529), "PrepareAndFinalizeWithAllGather"),
        "vllm_ascend/ops/fused_moe/fused_moe.py": (
            "2def0f6ba0320be66fa8aaac7ea6fe20b2abb00fb1d8a5793d98f407cb996627",
            (109, 200), "AscendMoERunner._fused_output_is_reduced"),
        "vllm_ascend/ops/fused_moe/routed_experts.py": (
            "c5d0bce5c8ef7ceebc0711d45d919a21dca7ea5c3a36822a60977a8120a67006",
            (496, 609), "AscendRoutedExperts.forward_impl"),
        "vllm_ascend/ops/fused_moe/shared_experts.py": (
            "564c010c94e217fba40a1fc8464cd2809c139b678896c89d8f950163cdd78e4e",
            (150, 228), "AscendSharedExperts.parallel_mode"),
        "vllm_ascend/ops/fused_moe/moe_comm_method.py": (
            "b785ade12a81349d01d4806c95ab17d37537db9548b3aae492135c8f4318fe39",
            (165, 218), "AllGatherCommImpl"),
        "vllm_ascend/ops/fused_moe/moe_mlp.py": (
            "6b5f4f6d3259cb14e0613a6f7b1c2592331f143a14cf7240dd9d49a0a57dd53b",
            (170, 372), "quant_apply_mlp"),
    },
    "final_tp_reduce_owner": "vllm_ascend/ops/fused_moe/fused_moe.py:185-197",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


@dataclass(frozen=True)
class RoutingPolicy:
    tile_rows: int = 1

    def expert_moments(self, model, layer, tokens: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Expected per-expert rows and active probabilities for one layer."""
        probabilities = model.expert_routing or (1 / layer.num_experts,) * layer.num_experts
        if len(probabilities) != layer.num_experts:
            raise ValueError("routing_probabilities_do_not_match_layer_experts")
        if (any(value < 0 or not math.isfinite(value) for value in probabilities)
                or sum(value > 0 for value in probabilities) < layer.top_k):
            raise ValueError("routing_support_must_cover_top_k")
        # Weighted sampling without replacement: saturate certain selections,
        # then redistribute remaining probability mass. Sum inclusions = top-k.
        remaining, available = float(layer.top_k), math.fsum(probabilities)
        inclusions = [0.0] * layer.num_experts
        for index in sorted(range(layer.num_experts), key=lambda i: probabilities[i], reverse=True):
            probability = probabilities[index]
            inclusion = min(1.0, remaining * probability / available) if available else 0
            inclusions[index] = inclusion
            remaining -= inclusion
            available -= probability
        rows, active = [], []
        for inclusion in inclusions:
            rows.append(tokens * inclusion)
            active.append(-math.expm1(tokens * math.log1p(-inclusion))
                          if 0 < inclusion < 1 else float(tokens > 0 and inclusion > 0))
        return tuple(rows), tuple(active)

    def local(
        self, model, layer, parallel, tokens: int, *,
        owner=None, runtime=None, eplb_damped: bool | None = None,
        zero_experts: int = 0,
    ) -> tuple[float, float]:
        rows, active = self.expert_moments(model, layer, tokens)
        num_experts = len(rows)
        pool_experts = num_experts + max(0, int(zero_experts))
        if zero_experts > 0 and num_experts > 0:
            real_share = float(num_experts) / float(pool_experts)
            rows = tuple(value * real_share for value in rows)
            scaled_active = []
            for value in rows:
                inclusion = (value / float(tokens)) if tokens > 0 else 0.0
                scaled_active.append(
                    -math.expm1(float(tokens) * math.log1p(-inclusion))
                    if 0.0 < inclusion < 1.0
                    else float(tokens > 0 and inclusion > 0.0)
                )
            active = tuple(scaled_active)
        ep = int(parallel.ep)
        damped = (
            _eplb_damping_active(parallel, runtime)
            if eplb_damped is None else bool(eplb_damped)
        )
        total_rows = (
            math.fsum(rows)
            if zero_experts > 0
            else float(max(0, int(tokens)) * int(layer.top_k))
        )
        if ep > num_experts and num_experts > 0:
            def copies(expert):
                return 1 + (ep - 1 - expert) // num_experts
            rank_rows = [rows[r % num_experts] / copies(r % num_experts) for r in range(ep)]
            slowest = max(range(ep), key=rank_rows.__getitem__) if owner is None else owner
            expert_idx = slowest % num_experts
            replicas = copies(expert_idx)
            inclusion = (rows[expert_idx] / tokens) if tokens > 0 else 0.0
            rep_tokens = tokens / replicas
            act_val = (-math.expm1(rep_tokens * math.log1p(-inclusion))
                       if 0 < inclusion < 1 else float(rep_tokens > 0 and inclusion > 0))
            if owner is not None or tokens <= 0 or ep <= 1:
                return rank_rows[slowest], act_val
            return _straggler_rank_moments(
                tokens=tokens,
                top_k=int(layer.top_k),
                num_experts=num_experts,
                ep=ep,
                total_rows=total_rows,
                total_active=math.fsum(active),
                mean_rows=rank_rows[slowest],
                local_active=(act_val,),
                local_expert_count=1,
                eplb_damped=damped,
                pool_experts=pool_experts,
            )
        rank_rows = [math.fsum(rows[index::ep]) for index in range(ep)]
        slowest = max(range(ep), key=rank_rows.__getitem__) if owner is None else owner
        local_active = active[slowest::ep]
        if owner is not None or tokens <= 0 or ep <= 1:
            return rank_rows[slowest], math.fsum(local_active)
        return _straggler_rank_moments(
            tokens=tokens,
            top_k=int(layer.top_k),
            num_experts=num_experts,
            ep=ep,
            total_rows=total_rows,
            total_active=math.fsum(active),
            mean_rows=rank_rows[slowest],
            local_active=local_active,
            local_expert_count=len(local_active),
            eplb_damped=damped,
            pool_experts=pool_experts,
        )


@dataclass(frozen=True)
class LocalBankRouting:
    """One native contiguous expert bank projected from a full expert ledger.

    ``expert_rows`` is the original global ledger. ``local_expert_rows`` is the
    selected contiguous bank's valid rows. ``local_weight_expert_ids`` address
    the local weight tensor; ``representative_expert_ids`` are the physical
    global ids. These are deliberately distinct: source lowering must not use
    global rows or global weight ids for a local bank.
    """

    expert_rows: tuple[int, ...]
    local_expert_rows: tuple[int, ...]
    rank_rows: tuple[int, ...]
    representative_rank: int
    basis: str
    ep: int
    parent: dict
    topk_ids: tuple[tuple[int, ...], ...] | None = None

    @property
    def rows(self):
        return sum(self.local_expert_rows)

    @property
    def active(self):
        return sum(value > 0 for value in self.local_expert_rows)

    @property
    def bank_size(self) -> int:
        return len(self.local_expert_rows)

    @property
    def local_weight_expert_ids(self) -> tuple[int, ...]:
        """Indices into the selected contiguous weight bank, never global ids."""
        return tuple(range(self.bank_size))

    @property
    def representative_expert_ids(self) -> tuple[int, ...]:
        """Physical global expert ids represented by this contiguous bank."""
        start = self.representative_rank * self.bank_size
        return tuple(range(start, start + self.bank_size))

    @property
    def valid_local_tokens(self) -> int:
        """Source-owned logical rows retained by the underlying route."""
        value = self.parent.get("valid_local_tokens")
        return int(value) if type(value) is int and value >= 0 else self.rows

    def payload_matrix(self, row_bytes, *, reverse=False):
        """Exact route cells when the parent ledger published them."""
        cells = self.parent.get("peer_assignments",
                                self.parent.get("peer_assignment_rows"))
        if cells is None:
            raise ModelingError(
                "local_bank_routing_has_no_explicit_peer_assignment_ledger")
        size = len(cells)
        return tuple(tuple(0 if source == destination else int(row_bytes) * (
            cells[destination][source] if reverse else cells[source][destination])
            for destination in range(size)) for source in range(size))

    def __post_init__(self):
        bank = len(self.local_expert_rows)
        if bank == 0 or len(self.expert_rows) != bank * self.ep:
            raise ModelingError(
                "local_bank_routing_global_ledger_must_have_ep_contiguous_banks")
        if len(self.rank_rows) != self.ep:
            raise ModelingError("local_bank_routing_rank_rows_must_match_ep")
        if not 0 <= self.representative_rank < self.ep:
            raise ModelingError("local_bank_routing_representative_rank_out_of_range")
        if any(type(value) is not int or value < 0 for value in self.expert_rows):
            raise ModelingError("local_bank_routing_global_rows_must_be_nonnegative_integers")
        if any(type(value) is not int or value < 0 for value in self.local_expert_rows):
            raise ModelingError("local_bank_routing_local_rows_must_be_nonnegative_integers")
        start = self.representative_rank * bank
        if tuple(self.expert_rows[start:start + bank]) != tuple(self.local_expert_rows):
            raise ModelingError("local_bank_routing_local_rows_must_be_global_bank_slice")
        if tuple(sum(self.expert_rows[rank * bank:(rank + 1) * bank])
                 for rank in range(self.ep)) != tuple(self.rank_rows):
            raise ModelingError("local_bank_routing_rank_rows_must_match_global_ledger")

    def to_dict(self):
        bank = self.bank_size
        payload = dict(self.parent)
        active = [value for value in self.local_expert_rows if value > 0]
        physical_ids = self.representative_expert_ids
        local_weight_ids = self.local_weight_expert_ids
        explicit = self.basis == EXPLICIT_TOKEN_ROUTING
        payload.update(
            basis=self.basis,
            expert_placement="contiguous_ep",
            expert_rows_basis=("explicit_token_route" if explicit
                               else "global_expert_ledger"),
            local_expert_rows_basis=("explicit_physical_bank" if explicit
                                     else "contiguous_ep_bank"),
            physical_expert_ids_basis="global_expert_id",
            expert_rows=self.expert_rows,
            global_expert_rows=self.expert_rows,
            local_expert_rows=self.local_expert_rows,
            rank_assignment_rows=self.rank_rows,
            representative_expert_ids=physical_ids,
            local_weight_expert_ids=local_weight_ids,
            representative_rank=self.representative_rank,
            represented_ep_ranks=((self.representative_rank,) if explicit
                                  else tuple(range(self.ep))),
            rank_timing_assumption=("explicit_physical_rank" if explicit
                                    else "heaviest_contiguous_bank_upper_bound"),
            row_classes=tuple(sorted(Counter(self.local_expert_rows).items())),
            matrix_groups=self.active,
            rows_per_active_expert=(
                active[0] if active and len(set(active)) == 1 else None),
            source_partitions=1,
            communication_assumption="none_local_expert_bank_filtering",
        )
        payload["token_identities"] = (
            "declared" if self.topk_ids is not None else "not_declared")
        if self.topk_ids is not None:
            payload["topk_ids"] = self.topk_ids
        return payload


@dataclass(frozen=True)
class LocalCallerInput:
    """Valid local rows, active experts and the contiguous-bank metadata."""

    local_rows: float
    active: float
    routing_realization: LocalBankRouting | None
    routing_metadata: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class LocalExpertReduce:
    """Replicated full-token input filtered by one local contiguous expert bank."""

    tokens: int
    top_k: int
    experts: int
    ep: int
    source_partitions: int = 1
    exchange: bool = False
    reduce_after_join: bool = True
    #: W8A8 eager allocation is a witness; matrix m stays the local bank rows.
    full_buffer_grouped: bool = False

    def __post_init__(self):
        if self.ep < 1 or self.experts % self.ep:
            raise ModelingError("local_expert_reduce_requires_divisible_expert_banks")

    @property
    def allocated_rows(self) -> int:
        # Eager MoeInitRoutingV3 allocates q*top_k rows before local filtering.
        return self.tokens * self.top_k

    @property
    def local_bank(self) -> int:
        return math.ceil(self.experts / self.ep)

    def activation_rows(self) -> int:
        # Native SwiGlu consumes the complete allocated output buffer, including
        # local-bank slots that received no valid assignment.
        return self.allocated_rows

    def _metadata(self, *, basis, projected, expert_rows, bank_rows, rank_rows,
                  rank, local_rows, active):
        explicit = basis == EXPLICIT_TOKEN_ROUTING
        metadata = {
            "moe_caller": LOCAL_EXPERT_REDUCE,
            "moe_caller_branch": "vllm_ascend_allgather_dp1_ep_tp",
            "local_expert_bank_filtering": True,
            "local_expert_bank_size": self.local_bank,
            "expert_placement": "contiguous_ep",
            "exchange": "none",
            "source_partitions": self.source_partitions,
            "allocated_routing_rows": self.allocated_rows,
            "activation_capacity_rows": self.allocated_rows,
            "routing_weighting": "router_weights_masked_for_nonlocal_experts",
            "payload_read_basis": "defined_valid_local_payload_slots",
            "rank_assignment_rows": tuple(rank_rows),
            "local_expert_rows": tuple(bank_rows),
            "representative_rank": rank,
            "representative_expert_ids": tuple(
                range(rank * self.local_bank, (rank + 1) * self.local_bank)),
            "represented_ep_ranks": ((rank,) if explicit
                                     else tuple(range(self.ep))),
            "rank_timing_assumption": ("explicit_physical_rank" if explicit
                                       else "heaviest_contiguous_bank_upper_bound"),
            "expert_rows_basis": ("explicit_token_route" if explicit
                                  else "global_expert_ledger"),
            "local_expert_rows_basis": ("explicit_physical_bank" if explicit
                                        else "contiguous_ep_bank"),
        }
        if projected is None:
            # Expected fractions are not an observed or invented histogram.
            metadata.update(
                routing_basis="probability_moments",
                expected_rows=local_rows,
                expected_active_experts=active,
                valid_local_rows=local_rows,
                expected_expert_rows=tuple(expert_rows),
                histogram_counts=None,
            )
        else:
            metadata.update(
                routing_basis=basis,
                routing_realization=projected.to_dict(),
                valid_local_rows=local_rows,
                active_local_experts=active,
                histogram_counts=tuple(bank_rows),
                global_expert_rows=tuple(expert_rows),
            )
        if self.full_buffer_grouped:
            # The eager router allocates q*top_k slots, but the native grouped
            # GEMM consumes only the local bank's group-list rows. Keep the
            # active matrix rows and the larger allocation as separate facts;
            # never turn rank totals or the allocation into per-expert counts.
            metadata.update(
                grouped_gemm_rows=_FULL_EAGER_GROUPED,
                matrix_active_rows=local_rows,
                matrix_capacity_rows=self.allocated_rows,
                matrix_capacity_basis="eager_routing_allocation_not_activated_rows",
            )
        return tuple(metadata.items())

    def prepare(self, routing, model, layer, parallel, tokens, realization, *, runtime=None):
        bank = self.local_bank
        if realization is None:
            moments = getattr(routing, "expert_moments", None)
            if moments is None:
                raise ModelingError("local_expert_reduce_requires_expert_moment_policy")
            expert_rows, expert_active = moments(model, layer, tokens)
            if len(expert_rows) != self.experts or len(expert_active) != self.experts:
                raise ModelingError("local_expert_reduce_expert_moment_width_mismatch")
            rank_rows = tuple(math.fsum(expert_rows[rank * bank:(rank + 1) * bank])
                              for rank in range(self.ep))
            representative = max(range(self.ep), key=rank_rows.__getitem__)
            bank_rows = tuple(expert_rows[representative * bank:(representative + 1) * bank])
            bank_active = tuple(
                expert_active[representative * bank:(representative + 1) * bank])
            if tokens > 0 and self.ep > 1:
                local_rows, active = _straggler_rank_moments(
                    tokens=tokens,
                    top_k=int(layer.top_k),
                    num_experts=self.experts,
                    ep=self.ep,
                    total_rows=float(tokens * int(layer.top_k)),
                    total_active=math.fsum(expert_active),
                    mean_rows=rank_rows[representative],
                    local_active=bank_active,
                    local_expert_count=len(bank_active),
                    eplb_damped=_eplb_damping_active(parallel, runtime),
                )
            else:
                local_rows = rank_rows[representative]
                active = math.fsum(bank_active)
            if self.allocated_rows > 0 and local_rows <= 0:
                raise ModelingError("local_expert_reduce_requires_nonempty_representative_bank")
            metadata = self._metadata(
                basis="probability_moments", projected=None, expert_rows=expert_rows,
                bank_rows=bank_rows, rank_rows=rank_rows, rank=representative,
                local_rows=local_rows, active=active)
            return LocalCallerInput(local_rows, active, None, metadata)
        rows = tuple(getattr(realization, "expert_rows", ()))
        if len(rows) != self.experts:
            raise ModelingError("local_expert_reduce_explicit_ledger_must_match_all_experts")
        if any(type(value) is not int or value < 0 for value in rows):
            raise ModelingError("local_expert_reduce_explicit_ledger_requires_nonnegative_integers")
        if sum(rows) != self.allocated_rows:
            raise ModelingError("local_expert_reduce_explicit_ledger_must_conserve_topk_assignments")
        topk_ids = getattr(realization, "topk_ids", None)
        if topk_ids is not None and len(topk_ids) != self.tokens:
            raise ModelingError("local_expert_reduce_topk_ids_must_match_tokens")
        rank_rows = tuple(sum(rows[rank * bank:(rank + 1) * bank])
                          for rank in range(self.ep))
        if getattr(realization, "basis", "") == EXPLICIT_TOKEN_ROUTING:
            # The explicit ledger was bound to this physical rank; its bank is
            # the actual work, not the heaviest bank of a statistical bound.
            representative = getattr(realization, "representative_rank", None)
            if type(representative) is not int or not 0 <= representative < self.ep:
                raise ModelingError(
                    "local_expert_reduce_explicit_representative_rank_invalid")
        else:
            representative = max(range(self.ep), key=rank_rows.__getitem__)
        bank_rows = rows[representative * bank:(representative + 1) * bank]
        local_rows = sum(bank_rows)
        active = sum(value > 0 for value in bank_rows)
        explicit = getattr(realization, "basis", "") == EXPLICIT_TOKEN_ROUTING
        if not explicit and self.allocated_rows > 0 and local_rows <= 0:
            raise ModelingError("local_expert_reduce_requires_nonempty_representative_bank")
        projected = LocalBankRouting(
            expert_rows=rows, local_expert_rows=bank_rows, rank_rows=rank_rows,
            representative_rank=representative, basis=realization.basis,
            ep=self.ep, parent=dict(realization.to_dict()), topk_ids=topk_ids)
        metadata = self._metadata(
            basis=realization.basis, projected=projected, expert_rows=rows,
            bank_rows=bank_rows, rank_rows=rank_rows, rank=representative,
            local_rows=local_rows, active=active)
        return LocalCallerInput(local_rows, active, projected, metadata)

    def pack_work(self, *, rows, latent, local_rows, weight_bytes, act_bytes):
        allocation = self.allocated_rows
        bank = self.local_bank
        read = (rows * latent * act_bytes + allocation * (2 * _ROUTING_INDEX_BYTES + weight_bytes)
                + bank * _RAW_EXPERT_COUNT_BYTES)
        write = (local_rows * latent * act_bytes + allocation * _ROUTING_INDEX_BYTES
                 + allocation * weight_bytes + bank * (_RAW_EXPERT_COUNT_BYTES + _CAST_EXPERT_COUNT_BYTES))
        return read, write, 2 * allocation + bank

    def pack_metadata(self, local_rows):
        return (
            ("assignment_scan_rows", self.allocated_rows),
            ("expert_mask_rows", self.allocated_rows),
            ("topk_weight_mask_rows", self.allocated_rows),
            ("payload_write_rows", local_rows),
            ("local_count_rows", self.local_bank),
            ("count_cast_dtype", "int64"),
        )

    def unpack_work(self, *, tokens, latent, local_rows, weight_bytes, act_bytes):
        allocation = self.allocated_rows
        read = local_rows * latent * act_bytes + allocation * (_ROUTING_INDEX_BYTES + weight_bytes)
        write = tokens * latent * act_bytes
        return read, write, 2 * tokens * latent * self.top_k + allocation

    def unpack_metadata(self, local_rows, weight_dtype):
        return (
            ("assignment_scan_rows", self.allocated_rows),
            ("payload_read_rows", local_rows),
            ("payload_slot_basis", "defined_valid_local_payload_slots"),
            ("inverse_index_dtype", "int32"),
            ("topk_weight_dtype", weight_dtype),
        )

    def tag_activation(self, op, local_rows: float):
        return replace(op, metadata=op.metadata + (
            ("moe_caller", LOCAL_EXPERT_REDUCE),
            ("valid_local_rows", local_rows),
            ("allocated_routing_rows", self.allocated_rows),
            ("activation_rows", self.allocated_rows),
            ("activation_capacity_basis", "full_eager_init_routing_allocation"),
        ))

    def tag_reduce(self, op):
        return replace(op, metadata=op.metadata + (
            ("moe_caller", LOCAL_EXPERT_REDUCE),
            ("reduction_scope", "routed_and_shared_joined_output"),
            ("shared_weight_layout", "tp_sharded"),
            ("shared_reduction_before_join", False),
            ("routed_reduction_before_join", False),
        ))


def select(runtime, plan, parallel, model, tokens: int, *,
           global_tokens=None, sequence_plan=None, routing_realization=None):
    if getattr(runtime, "moe_recipe", "alltoall") != LOCAL_EXPERT_REDUCE:
        raise ModelingError("local_expert_reduce_requires_explicit_moe_recipe")
    kernels = getattr(runtime, "kernels", None) or {}
    if "moe_transport" in kernels:
        raise ModelingError("local_expert_reduce_rejects_transport_declaration")
    if "moe_token_partition" in kernels:
        raise ModelingError("local_expert_reduce_rejects_token_partition_declaration")
    if "native_moe_protocol" in kernels:
        raise ModelingError("local_expert_reduce_rejects_native_mc2_declaration")
    sp = kernels.get("sequence_parallel")
    if sp is not None:
        layout = sp.get("shared_expert_layout") if isinstance(sp, dict) else None
        if layout not in (None, "tp_sharded", "TENSOR_PARALLEL"):
            raise ModelingError(
                "local_expert_reduce_rejects_shared_expert_data_parallel_layout")
        raise ModelingError("local_expert_reduce_rejects_sequence_parallel_declaration")
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    layouts = {None, "tp_sharded", "tensor_parallel", "TENSOR_PARALLEL"}
    for key in ("shared_expert_layout", "shared_expert_parallel_mode"):
        if metadata.get(key) not in layouts:
            raise ModelingError(
                "local_expert_reduce_rejects_shared_expert_data_parallel_layout")
    if metadata.get("enable_shared_expert_dp"):
        raise ModelingError("local_expert_reduce_rejects_shared_expert_data_parallel_layout")
    if metadata.get("enable_deepep") or metadata.get("moe_dispatch_quantization"):
        raise ModelingError("local_expert_reduce_rejects_dispatch_transport_metadata")
    if parallel.dp != 1:
        raise ModelingError(f"local_expert_reduce_requires_dp1:{parallel.dp}")
    if parallel.ep < 2 or parallel.ep != parallel.tp:
        raise ModelingError(
            f"local_expert_reduce_requires_ep_equals_tp:{parallel.ep}!={parallel.tp}")
    if parallel.etp != 1:
        raise ModelingError(f"local_expert_reduce_requires_etp1:{parallel.etp}")
    if plan.num_experts % parallel.ep:
        raise ModelingError(
            f"local_expert_reduce_requires_divisible_expert_banks:"
            f"{plan.num_experts}%{parallel.ep}")
    # PCP/PP do not change the replicated token view inside one EP=TP group.
    # CP still shards the KV context this caller does not model.
    if parallel.cp != 1 or parallel.pcp < 1 or parallel.pp < 1:
        raise ModelingError(
            f"local_expert_reduce_requires_cp_pcp1:{parallel.cp},{parallel.pcp}")
    if sequence_plan is not None:
        raise ModelingError("local_expert_reduce_rejects_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("local_expert_reduce_rejects_routed_output_transform")
    if global_tokens not in (None, tokens):
        raise ModelingError("local_expert_reduce_requires_replicated_dp1_tokens")
    path = _expert_path(model)
    # The full grouped-matmul buffer is an explicit W8A8 witness. A float
    # caller that does not declare it keeps the heaviest contiguous bank.
    full_buffer = (
        path == "w8a8"
        and metadata.get("moe_grouped_gemm_rows") == _FULL_EAGER_GROUPED
    )
    policy = LocalExpertReduce(tokens=tokens, top_k=plan.top_k,
                               experts=plan.num_experts, ep=parallel.ep,
                               full_buffer_grouped=full_buffer)
    if routing_realization is not None:
        basis = getattr(routing_realization, "basis", "")
        if basis == "source_force_eplb_masked_integer_counts":
            raise ModelingError("local_expert_reduce_rejects_source_force_eplb_assignments")
        if basis not in _DECLARED_LEDGER_BASES:
            raise ModelingError("local_expert_reduce_requires_declared_integer_ledger")
        if basis == EXPLICIT_TOKEN_ROUTING:
            # The local-expert-reduce caller consumes one contiguous weight
            # bank. An exact route may carry arbitrary source owners, but its
            # logical->physical placement must still be that same bank order;
            # a non-contiguous map belongs to the grouped/exchange path.
            bank = plan.num_experts // parallel.ep
            expected = tuple((rank, slot) for rank in range(parallel.ep)
                             for slot in range(bank))
            if tuple(getattr(routing_realization, "expert_placement", ())) != expected:
                raise ModelingError(
                    "local_expert_reduce_requires_contiguous_physical_bank")
        rows = tuple(getattr(routing_realization, "expert_rows", ()))
        if (len(rows) != plan.num_experts
                or any(type(row) is not int or row < 0 for row in rows)):
            raise ModelingError("local_expert_reduce_explicit_ledger_must_match_all_experts")
        if sum(rows) != policy.allocated_rows:
            raise ModelingError("local_expert_reduce_explicit_ledger_must_conserve_topk_assignments")
        declared_tokens = getattr(routing_realization, "tokens", None)
        if declared_tokens is not None and declared_tokens != tokens:
            raise ModelingError("local_expert_reduce_realization_tokens_must_match_source_rows")
        topk_ids = getattr(routing_realization, "topk_ids", None)
        if topk_ids is not None and len(topk_ids) != tokens:
            raise ModelingError("local_expert_reduce_topk_ids_must_match_tokens")
        if getattr(routing_realization, "ep", parallel.ep) != parallel.ep:
            raise ModelingError("local_expert_reduce_realization_ep_must_match_parallel_ep")
    return policy


def tp_join_reduce(*, plan, parallel, model, **kwargs):
    """TP-sharded experts and shared output reduce once after linear combine."""
    from mindie_sim.prediction.feedforward import alltoall

    # PP stages the same token rows; PCP has already split them. The deferred
    # TP reduce stays inside the TP group either way.
    if (parallel.ep != 1 or parallel.dp != 1 or parallel.cp != 1
            or parallel.pcp < 1 or parallel.pp < 1):
        raise ModelingError("tp_join_reduce_requires_tp_only_non_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("tp_join_reduce_cannot_defer_across_routed_output_transform")
    # This explicit recipe declares TP-sharded shared weights and replicated
    # router/shared gates. Their linear combine commutes with the final sum.
    return alltoall(plan=plan, parallel=parallel, model=model,
                    reduce_after_join=True, **kwargs)


def local_expert_reduce_recipe(**kwargs):
    """Validate the explicit branch, then compose the ordinary MoE builder."""
    from mindie_sim.prediction.feedforward import alltoall

    policy = select(
        kwargs.get("runtime"), kwargs["plan"], kwargs["parallel"], kwargs["model"],
        kwargs["tokens"], global_tokens=kwargs.get("global_tokens"),
        sequence_plan=kwargs.get("sequence_plan"),
        routing_realization=kwargs.get("routing_realization"),
    )
    kwargs.pop("chunks", None)
    kwargs["caller"] = policy
    return alltoall(**kwargs)
