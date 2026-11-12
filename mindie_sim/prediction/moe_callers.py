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

from mindie_sim.schema.common import ModelingError


LOCAL_EXPERT_REDUCE = "local_expert_reduce"
_FLOAT_EXPERT_FORMATS = {"bf16": "bf16", "bfloat16": "bf16",
                         "fp16": "fp16", "float16": "fp16"}
_W8A8_FORMATS = {"int8": "int8", "w8a8": "int8"}
_FULL_EAGER_GROUPED = "full_eager_allocation"


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

    def local(self, model, layer, parallel, tokens: int) -> tuple[float, float]:
        rows, active = self.expert_moments(model, layer, tokens)
        if parallel.ep > len(rows) and len(rows) > 0:
            replicas = parallel.ep // len(rows)
            rank_rows = [rows[index % len(rows)] / replicas for index in range(parallel.ep)]
            slowest = max(range(parallel.ep), key=rank_rows.__getitem__)
            expert_idx = slowest % len(rows)
            inclusion = (rows[expert_idx] / tokens) if tokens > 0 else 0.0
            rep_tokens = tokens / replicas
            act_val = (-math.expm1(rep_tokens * math.log1p(-inclusion))
                       if 0 < inclusion < 1 else float(rep_tokens > 0 and inclusion > 0))
            return rank_rows[slowest], act_val
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

    if (state_counts is not None and geometry_counts is not None
            and tuple(state_counts) != tuple(geometry_counts)):
        conflicts.append("state_geometry_counts_disagree")
    counts = state_counts if state_counts is not None else geometry_counts
    basis = ("state_routing" if state_counts is not None
             else "geometry_routing" if geometry_counts is not None
             else "unknown")
    if "state_geometry_counts_disagree" in conflicts:
        counts = None
        basis = "state_geometry_conflict"

    explicit_ids = _receipt_ids(
        explicit.get("local_expert_ids"), layer,
        _receipt_int(explicit.get("rank_index",
                                  explicit.get("current_rank"))))
    if explicit_ids is not None and state_ids is not None \
            and tuple(explicit_ids) != tuple(state_ids):
        conflicts.append("state_explicit_local_expert_ids_disagree")
    local_ids = explicit_ids if explicit_ids is not None else state_ids
    rank = _receipt_int(explicit.get("rank_index",
                                     explicit.get("current_rank")))

    if local_experts < 1:
        local_experts = (len(local_ids) if local_ids is not None
                         else len(state_counts) if state_counts is not None
                         else len(counts) if counts is not None else 0)
    if experts < 1:
        experts = len(counts) if counts is not None else local_experts
    if ep < 1:
        ep = (experts // local_experts
              if local_experts > 0 and experts % local_experts == 0 else 0)
    if local_ids is not None and len(local_ids) != local_experts:
        conflicts.append("local_expert_ids_length_mismatch")
        local_ids = None

    local_counts = None
    ownership_known = False
    if counts is not None and local_experts > 0:
        if len(counts) == local_experts:
            if basis == "state_routing":
                local_counts = counts
                ownership_known = local_ids is not None
            elif local_ids is not None:
                local_counts = counts
                ownership_known = True
            else:
                conflicts.append("geometry_local_counts_without_local_expert_ids")
        elif len(counts) == experts:
            if local_ids is not None:
                if any(index >= len(counts) for index in local_ids):
                    conflicts.append("local_expert_ids_out_of_range")
                else:
                    local_counts = tuple(counts[index] for index in local_ids)
                    ownership_known = True
            else:
                conflicts.append("global_counts_without_local_expert_ids")
        else:
            conflicts.append("routing_count_length_mismatch")

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
        canonical_prefix = None
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
