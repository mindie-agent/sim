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

from mindie_sim.schema.common import ModelingError


LOCAL_EXPERT_REDUCE = "local_expert_reduce"
_FLOAT_EXPERT_FORMATS = {"bf16": "bf16", "bfloat16": "bf16",
                         "fp16": "fp16", "float16": "fp16"}


def _effective_expert_format(quant, category):
    """Resolve the category selection used by the ordinary matrix builders."""
    return _FLOAT_EXPERT_FORMATS.get(
        str(quant.for_category(category).format_for(category)).strip().lower())


def _require_floating_expert_paths(model):
    activation = _FLOAT_EXPERT_FORMATS.get(str(model.activation_dtype).strip().lower())
    if activation is None:
        raise ModelingError(
            f"local_expert_reduce_requires_floating_activation_dtype:{model.activation_dtype}")
    for category in ("routed_expert", "shared_expert"):
        fmt = _effective_expert_format(model.quant, category)
        if fmt is None:
            raise ModelingError(
                "local_expert_reduce_rejects_quantized_expert_format:"
                f"{category}:{model.quant.for_category(category).format_for(category)}")
        if fmt != activation:
            raise ModelingError(
                "local_expert_reduce_rejects_mixed_expert_format:"
                f"{category}:{fmt}!={activation}")


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

    def local(self, model, layer, parallel, tokens: int) -> tuple[float, float]:
        rows, active = self.expert_moments(model, layer, tokens)
        rank_rows = [math.fsum(rows[index::parallel.ep]) for index in range(parallel.ep)]
        slowest = max(range(parallel.ep), key=rank_rows.__getitem__)
        return rank_rows[slowest], math.fsum(active[slowest::parallel.ep])


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
        payload.update(
            basis=self.basis,
            expert_placement="contiguous_ep",
            expert_rows_basis="global_expert_ledger",
            local_expert_rows_basis="contiguous_ep_bank",
            physical_expert_ids_basis="global_expert_id",
            expert_rows=self.expert_rows,
            global_expert_rows=self.expert_rows,
            local_expert_rows=self.local_expert_rows,
            rank_assignment_rows=self.rank_rows,
            representative_expert_ids=physical_ids,
            local_weight_expert_ids=local_weight_ids,
            representative_rank=self.representative_rank,
            represented_ep_ranks=tuple(range(self.ep)),
            rank_timing_assumption="heaviest_contiguous_bank_upper_bound",
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
            "represented_ep_ranks": tuple(range(self.ep)),
            "rank_timing_assumption": "heaviest_contiguous_bank_upper_bound",
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
        return tuple(metadata.items())

    def prepare(self, routing, model, layer, parallel, tokens, realization):
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
            local_rows = rank_rows[representative]
            active = math.fsum(
                expert_active[representative * bank:(representative + 1) * bank])
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
        representative = max(range(self.ep), key=rank_rows.__getitem__)
        bank_rows = rows[representative * bank:(representative + 1) * bank]
        local_rows = sum(bank_rows)
        active = sum(value > 0 for value in bank_rows)
        if self.allocated_rows > 0 and local_rows <= 0:
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
        read = (rows * latent * act_bytes + allocation * (8 + weight_bytes) + bank * 4)
        write = (local_rows * latent * act_bytes + allocation * 4
                 + allocation * weight_bytes + bank * 12)
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
        read = local_rows * latent * act_bytes + allocation * (4 + weight_bytes)
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
    if parallel.cp != 1 or parallel.pcp != 1:
        raise ModelingError(
            f"local_expert_reduce_requires_cp_pcp1:{parallel.cp},{parallel.pcp}")
    if sequence_plan is not None:
        raise ModelingError("local_expert_reduce_rejects_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("local_expert_reduce_rejects_routed_output_transform")
    if global_tokens not in (None, tokens):
        raise ModelingError("local_expert_reduce_requires_replicated_dp1_tokens")
    _require_floating_expert_paths(model)
    policy = LocalExpertReduce(tokens=tokens, top_k=plan.top_k,
                               experts=plan.num_experts, ep=parallel.ep)
    if routing_realization is not None:
        basis = getattr(routing_realization, "basis", "")
        if basis == "source_force_eplb_masked_integer_counts":
            raise ModelingError("local_expert_reduce_rejects_source_force_eplb_assignments")
        if basis not in {"forced_balanced_integer_counts", "declared_integer_counts"}:
            raise ModelingError("local_expert_reduce_requires_declared_integer_ledger")
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
    from .feedforward import alltoall

    if parallel.ep != 1 or parallel.dp != 1 or parallel.cp != 1 or parallel.pcp != 1:
        raise ModelingError("tp_join_reduce_requires_tp_only_non_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("tp_join_reduce_cannot_defer_across_routed_output_transform")
    # This explicit recipe declares TP-sharded shared weights and replicated
    # router/shared gates. Their linear combine commutes with the final sum.
    return alltoall(plan=plan, parallel=parallel, model=model,
                    reduce_after_join=True, **kwargs)


def local_expert_reduce_recipe(**kwargs):
    """Validate the explicit branch, then compose the ordinary MoE builder."""
    from .feedforward import alltoall

    policy = select(
        kwargs.get("runtime"), kwargs["plan"], kwargs["parallel"], kwargs["model"],
        kwargs["tokens"], global_tokens=kwargs.get("global_tokens"),
        sequence_plan=kwargs.get("sequence_plan"),
        routing_realization=kwargs.get("routing_realization"),
    )
    kwargs.pop("chunks", None)
    kwargs["caller"] = policy
    return alltoall(**kwargs)
