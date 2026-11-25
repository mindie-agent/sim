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
from typing import Any, Literal, Mapping, Sequence

from mindie_sim.catalog.expert_slots import expert_slot_plan
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.prediction.operators.moe_routing.expert_slot_work import physical_slot_work
from mindie_sim.schema.common import (
    EPLB_VARIANCE_DAMPING,
    ORDER_STAT_M2_Z,
    ModelingError,
    dtype_bytes,
    normal_max_order_z,
    normal_max_order_z as _normal_max_order_z,
)
from mindie_sim.schema.datapath import SourceRef
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

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"


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
_FP32_BYTES = int(dtype_bytes("fp32"))
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


def _zero_expert_count(layer, model) -> int:
    """Count zero/identity MoE slots competing in routing."""
    facts = dict(getattr(model, "catalog_facts", ()) or {})
    return max(0, int(
        getattr(layer, "zero_expert_num", 0)
        or getattr(model, "zero_expert_num", 0)
        or facts.get("zero_expert_num", 0)
        or 0
    ))


def _apply_zero_expert_share(
    rows: Sequence[float], active: Sequence[float], tokens: int, zero_experts: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Project real-expert rows and active probabilities under zero-expert slots."""
    num_experts = len(rows)
    if zero_experts <= 0 or num_experts <= 0:
        return tuple(rows), tuple(active)
    real_share = float(num_experts) / float(num_experts + int(zero_experts))
    scaled_rows = tuple(value * real_share for value in rows)
    scaled_active = []
    for value in scaled_rows:
        inclusion = (value / float(tokens)) if tokens > 0 else 0.0
        scaled_active.append(
            -math.expm1(float(tokens) * math.log1p(-inclusion))
            if 0.0 < inclusion < 1.0
            else float(tokens > 0 and inclusion > 0.0)
        )
    return scaled_rows, tuple(scaled_active)


def _hypergeometric_any_real_prob(
    pool: int, real_experts: int, top_k: int, tokens: int,
) -> float:
    """Exact P(>=1 real expert drawn across tokens independent top-k draws)."""
    if tokens <= 0 or top_k <= 0 or real_experts <= 0 or pool <= 0:
        return 0.0
    zero_pool = max(0, int(pool) - int(real_experts))
    if top_k > zero_pool:
        return 1.0
    p_miss_token = math.comb(zero_pool, top_k) / math.comb(int(pool), top_k)
    if p_miss_token <= 0.0:
        return 1.0
    if p_miss_token >= 1.0:
        return 0.0
    return -math.expm1(float(tokens) * math.log(p_miss_token))


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
    local_slot_probs: Sequence[float] | None = None,
    slot_copies: Sequence[int] | None = None,
    total_slots: int | None = None,
    uncertain_global_mass: float | None = None,
    active_rank_count: int | None = None,
) -> tuple[float, float]:
    """Closed-form O(E) order-statistic slowest EP rank rows and active banks."""
    if tokens <= 0 or top_k <= 0 or total_rows <= 0.0 or local_expert_count <= 0:
        return 0.0, 0.0
    if ep <= 1:
        return mean_rows, math.fsum(local_active)
    pool = max(1, int(pool_experts if pool_experts is not None else num_experts))
    fpc_pool = max(pool, int(total_slots or 0))
    variance_scale = EPLB_VARIANCE_DAMPING if eplb_damped else 1.0
    rank_share = min(1.0, max(0.0, mean_rows / max(1.0, total_rows)))
    fpc = max(0.0, 1.0 - float(top_k - 1) / float(max(1, fpc_pool - 1)))
    hypergeom_var = total_rows * rank_share * (1.0 - rank_share) * fpc * variance_scale
    if local_slot_probs is not None:
        copies_seq = (
            slot_copies if slot_copies is not None else (1,) * len(local_slot_probs)
        )
        reachable_local = sum(1 for p in local_slot_probs if p > _STRAGGLER_ORDER_EPS)
        replicated_var = 0.0
        uncertain_local = 0.0
        uncertain_bernoulli = 0.0
        for p, copy_count in zip(local_slot_probs, copies_seq):
            if p <= _STRAGGLER_ORDER_EPS:
                continue
            if int(copy_count) > 1:
                replicated_var += p * (1.0 - p)
            elif p < 1.0 - _STRAGGLER_ORDER_EPS:
                uncertain_local += p
                uncertain_bernoulli += p * (1.0 - p)
        if uncertain_global_mass is not None and uncertain_global_mass > _STRAGGLER_ORDER_EPS:
            share_uncertain = min(1.0, max(0.0, uncertain_local / uncertain_global_mass))
            uncertain_var = min(
                uncertain_bernoulli,
                uncertain_local * (1.0 - share_uncertain),
            )
        else:
            uncertain_var = uncertain_bernoulli
        slot_var = float(tokens) * (replicated_var + uncertain_var) * variance_scale
        row_var = min(hypergeom_var, slot_var)
    else:
        reachable_local = local_expert_count
        row_var = hypergeom_var
    if reachable_local <= 0:
        return 0.0, 0.0
    sigma_rows = math.sqrt(max(0.0, row_var))
    rank_cap = max(1, min(ep, int(active_rank_count))) if active_rank_count is not None else ep
    active_ranks = min(rank_cap, max(1, math.ceil(total_rows - _STRAGGLER_ORDER_EPS)))
    z_order = normal_max_order_z(active_ranks)
    row_cap = float(tokens * min(top_k, reachable_local))
    row_upper = min(total_rows, row_cap)
    is_integer_total = math.isclose(
        total_rows, round(total_rows), rel_tol=0.0, abs_tol=_STRAGGLER_ORDER_EPS,
    )
    if total_rows < 1.0:
        # Exact indicator lower bound when zero_experts shrinks N_eff < 1.0:
        # E[max_r R_r] >= P(at least one real assignment) under without-replacement top-k.
        p_any = _hypergeometric_any_real_prob(pool, num_experts, top_k, tokens)
        row_lower = max(mean_rows, p_any)
        r_max = min(row_upper, max(row_lower, mean_rows + z_order * sigma_rows))
    elif not is_integer_total:
        row_lower = max(mean_rows, total_rows / float(ep))
        r_max = min(row_upper, max(row_lower, mean_rows + z_order * sigma_rows))
    else:
        row_lower = float(math.ceil(total_rows / float(ep) - _STRAGGLER_ORDER_EPS))
        r_max = float(min(
            int(round(row_upper)),
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
        p_any = _hypergeometric_any_real_prob(pool, num_experts, top_k, tokens)
        act_lower = min(r_max, max(mu_act, p_any))
    else:
        act_lower = max(1.0, float(math.ceil(total_active / float(ep) - _STRAGGLER_ORDER_EPS)))
    a_max = min(
        float(reachable_local),
        r_max,
        max(act_lower, mu_act + z_order * sigma_act),
    )
    return r_max, a_max


def _category_format(quant, category):
    return quant.for_category(category).format_for(category)


def _expert_path(model, plan=None):
    """Float experts stay on the original caller. W8A8 is a separate legal path.

    Routed W8A8 with a floating activation is accepted when any present shared
    expert is that same activation float or W8A8. W8A16, shared W8A8 beside
    floating routed experts, mixed floats, and every other quantized format
    stay rejected. The recipe still has to be selected explicitly.
    """
    activation = _FLOAT_EXPERT_FORMATS.get(str(model.activation_dtype).strip().lower())
    if activation is None:
        raise ModelingError(
            f"local_expert_reduce_requires_floating_activation_dtype:{model.activation_dtype}")
    quant = projection_quant(model.quant, plan) if plan is not None else model.quant
    has_shared = plan is None or int(getattr(plan, "num_shared_experts", 0) or 0) > 0
    routed_shown = _category_format(quant, "routed_expert")
    routed_f = _FLOAT_EXPERT_FORMATS.get(str(routed_shown).strip().lower())
    routed_w8a8 = quant.linear_method("routed_expert") == "w8a8"
    if not has_shared:
        if routed_f is not None:
            if routed_f != activation:
                raise ModelingError(
                    "local_expert_reduce_rejects_mixed_expert_format:"
                    f"routed_expert:{routed_f}!={activation}")
            return "float"
        if routed_w8a8:
            return "w8a8"
        raise ModelingError(
            "local_expert_reduce_rejects_quantized_expert_format:"
            f"routed_expert:{routed_shown}")
    shared_shown = _category_format(quant, "shared_expert")
    shared_f = _FLOAT_EXPERT_FORMATS.get(str(shared_shown).strip().lower())
    shared_w8a8 = quant.linear_method("shared_expert") == "w8a8"
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
    if routed_w8a8 and (shared_f == activation or shared_w8a8):
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


SOURCES: tuple[SourceRef, ...] = tuple(
    SourceRef(
        repository=SOURCE["repository"],
        revision=SOURCE["revision"],
        relative_path=rel_path,
        source_sha256=sha256,
    )
    for rel_path, (sha256, _lines, _symbol) in SOURCE["files"].items()
)


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
        # Weighted sampling without replacement: exact two-phase water-filling.
        # Phase 1 peels off strictly saturated experts using exact integer
        # remaining slots ((remaining_int - 1) * p_i >= tail_after) to avoid
        # floating-point absorption; Phase 2 distributes remaining slots
        # proportionally over the tail without iterative subtraction.
        ordered = sorted(range(layer.num_experts), key=lambda i: probabilities[i], reverse=True)
        positive = [index for index in ordered if probabilities[index] > 0.0]
        inclusions = [0.0] * layer.num_experts
        if len(positive) <= layer.top_k:
            for index in positive:
                inclusions[index] = 1.0
        else:
            suffix = [0.0] * (len(positive) + 1)
            for pos in range(len(positive) - 1, -1, -1):
                suffix[pos] = math.fsum(
                    probabilities[positive[j]] for j in range(pos, len(positive))
                )
            remaining_int = int(layer.top_k)
            split = len(positive)
            for pos, index in enumerate(positive):
                if remaining_int <= 0:
                    break
                tail_after = suffix[pos + 1]
                if float(remaining_int - 1) * probabilities[index] >= tail_after:
                    inclusions[index] = 1.0
                    remaining_int -= 1
                else:
                    split = pos
                    break
            if remaining_int > 0 and split < len(positive):
                denom = suffix[split]
                if denom > 0.0:
                    for pos in range(split, len(positive)):
                        index = positive[pos]
                        inclusions[index] = min(
                            1.0, float(remaining_int) * (probabilities[index] / denom),
                        )
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
        rows, active = _apply_zero_expert_share(rows, active, tokens, zero_experts)
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
        if tokens <= 0 or num_experts <= 0:
            return 0.0, 0.0
        logical_inclusions = tuple(
            min(1.0, max(0.0, value / float(tokens))) for value in rows
        )
        slot_plan = expert_slot_plan(model, layer, parallel)
        slot_work = physical_slot_work(
            slot_plan=slot_plan,
            logical_inclusions=logical_inclusions,
            global_tokens=int(tokens),
        )
        rank_rows = slot_work.expected_rows_by_rank
        slowest = max(range(ep), key=rank_rows.__getitem__) if owner is None else int(owner)
        local_active = slot_work.active_probabilities[slowest]
        if owner is not None or ep <= 1:
            return float(rank_rows[slowest]), float(math.fsum(local_active))
        copies = slot_plan.copies_per_expert()
        local_ids = slot_plan.expert_ids_by_rank[slowest]
        local_slot_probs = tuple(
            logical_inclusions[expert] / float(max(1, copies[expert]))
            for expert in local_ids
        )
        local_slot_copies = tuple(max(1, int(copies[expert])) for expert in local_ids)
        total_slots = sum(len(ids) for ids in slot_plan.expert_ids_by_rank)
        uncertain_global_mass = math.fsum(
            logical_inclusions[e]
            for e in range(num_experts)
            if copies.get(e, 1) == 1 and _STRAGGLER_ORDER_EPS < logical_inclusions[e] < 1.0 - _STRAGGLER_ORDER_EPS
        )
        active_rank_count = sum(1 for r in rank_rows if r > _STRAGGLER_ORDER_EPS)
        total_active = math.fsum(
            math.fsum(probs) for probs in slot_work.active_probabilities
        )
        return _straggler_rank_moments(
            tokens=tokens,
            top_k=int(layer.top_k),
            num_experts=num_experts,
            ep=ep,
            total_rows=total_rows,
            total_active=total_active,
            mean_rows=float(rank_rows[slowest]),
            local_active=local_active,
            local_expert_count=len(local_active),
            eplb_damped=damped,
            pool_experts=pool_experts,
            local_slot_probs=local_slot_probs,
            slot_copies=local_slot_copies,
            total_slots=total_slots,
            uncertain_global_mass=uncertain_global_mass,
            active_rank_count=active_rank_count,
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
    #: Exact count of tokens whose slot-0 top-k expert belongs to this local bank.
    slot0_active_tokens: int | None = None

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
            zero_experts = _zero_expert_count(layer, model)
            expert_rows, expert_active = _apply_zero_expert_share(
                expert_rows, expert_active, tokens, zero_experts,
            )
            rank_rows = tuple(math.fsum(expert_rows[rank * bank:(rank + 1) * bank])
                              for rank in range(self.ep))
            representative = max(range(self.ep), key=rank_rows.__getitem__)
            bank_rows = tuple(expert_rows[representative * bank:(representative + 1) * bank])
            bank_active = tuple(
                expert_active[representative * bank:(representative + 1) * bank])
            if tokens > 0 and self.ep > 1:
                local_slot_probs = tuple(
                    min(1.0, max(0.0, value / float(tokens))) for value in bank_rows
                )
                local_rows, active = _straggler_rank_moments(
                    tokens=tokens,
                    top_k=int(layer.top_k),
                    num_experts=self.experts,
                    ep=self.ep,
                    total_rows=math.fsum(expert_rows),
                    total_active=math.fsum(expert_active),
                    mean_rows=rank_rows[representative],
                    local_active=bank_active,
                    local_expert_count=len(bank_active),
                    eplb_damped=_eplb_damping_active(parallel, runtime),
                    pool_experts=self.experts + zero_experts,
                    local_slot_probs=local_slot_probs,
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
        # CANN KernelMoeTokenUnpermute (moe_token_unpermute.h:249-358):
        # masked slot-0 tokens zero-fill (Duplicate), active slot-0 tokens scale
        # without Add (CalFirstToken), later active slots scale+accumulate
        # (CalToken), and non-FP32 rows cast on load and output store.
        if self.slot0_active_tokens is not None:
            first_active = min(float(tokens), float(local_rows), float(self.slot0_active_tokens))
        else:
            first_active = (
                min(float(tokens), float(local_rows) / float(self.top_k))
                if self.top_k > 0 else 0.0
            )
        zero_fill_ops = max(0.0, float(tokens) - first_active) * latent
        mul_ops = float(local_rows) * latent
        add_ops = max(0.0, float(local_rows) - first_active) * latent
        cast_ops = (float(local_rows) + float(tokens)) * latent if act_bytes < _FP32_BYTES else 0.0
        return read, write, zero_fill_ops + mul_ops + add_ops + cast_ops + allocation

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
    # CP and PCP shard context/prompt tokens across collectives not emitted by this caller.
    if parallel.cp != 1 or parallel.pcp != 1 or parallel.pp < 1:
        raise ModelingError(
            f"local_expert_reduce_requires_cp_pcp1:{parallel.cp},{parallel.pcp}")
    if sequence_plan is not None:
        raise ModelingError("local_expert_reduce_rejects_sequence_parallel")
    if plan.expert_latent_size not in (0, None, model.hidden_size):
        raise ModelingError("local_expert_reduce_rejects_routed_output_transform")
    if global_tokens not in (None, tokens):
        raise ModelingError("local_expert_reduce_requires_replicated_dp1_tokens")
    path = _expert_path(model, plan)
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
        if topk_ids is not None:
            bank = plan.num_experts // parallel.ep
            if basis == EXPLICIT_TOKEN_ROUTING:
                rep = getattr(routing_realization, "representative_rank", None)
                rep_valid = type(rep) is int and 0 <= rep < parallel.ep
            else:
                rank_totals = tuple(
                    sum(rows[rank * bank:(rank + 1) * bank]) for rank in range(parallel.ep)
                )
                rep = max(range(parallel.ep), key=rank_totals.__getitem__)
                rep_valid = True
            if rep_valid:
                lo, hi = rep * bank, (rep + 1) * bank
                slot0_active = sum(1 for slots in topk_ids if slots and lo <= int(slots[0]) < hi)
                policy = replace(policy, slot0_active_tokens=slot0_active)
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
