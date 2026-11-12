"""Source-selected MoE communication method from the bound deployment.

[Role]
  Project the retained vLLM-ascend hardware MoE policy, the configured
  ``enable_fused_mc2`` flag, the largest graph capture size and the runtime
  token budget into the one communication method a forward selects.  The same
  projection is shared by the ordinary piecewise F1/F2/F3 families and by the
  test-time recipe helper; it is input selection, not a timing formula.

[Invariants]
  * ``set_mc2_tokens_capacity`` is reproduced with its real branches: an
    explicit prefill-MC2 budget, the largest capture size, or the decode
    request budget.  The fused-MC2 per-rank limit is 512; MegaMoe uses 16384
    only when the platform declares that capability.
  * The A5 ``CAPACITY_AND_WORLD_SIZE`` selector, the A3
    ``FUSED_OR_CAPACITY`` selector and the A2
    ``CAPACITY_AND_EXPERT_DENSITY`` selector keep their source conditions.
  * Missing deployment declarations leave the caller's declared
    ``moe_recipe`` untouched.  No model name, phase name or measured latency
    selects a method.
  * Local tokens are the pre-expansion request rows, not the EP cohort total.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O, device work, locks or background tasks.

[Side-Effect Policy]
  Immutable records only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError

ALLGATHER = "allgather"
MC2 = "mc2"
ALLTOALL = "alltoall"
FUSED_MC2 = "fused_mc2"

CAPACITY_AND_EXPERT_DENSITY = "capacity_and_expert_density"
FUSED_OR_CAPACITY = "fused_or_capacity"
CAPACITY_AND_WORLD_SIZE = "capacity_and_world_size"

_MEGA_MOE_TOKENS_PER_RANK_LIMIT = 16384
_DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT = 512
_MC2_TOKENS_PER_RANK_LIMIT = 512

_ARCH_POLICY = {
    "A2": CAPACITY_AND_EXPERT_DENSITY,
    "A3": FUSED_OR_CAPACITY,
    "A5": CAPACITY_AND_WORLD_SIZE,
}

# The declaration strings are stable input identities, not paths or labels.
DYNAMIC = "dynamic"
ALWAYS_MC2 = "always_mc2"


def _metadata(features: Any) -> dict[str, Any]:
    return dict(getattr(features, "metadata", ()) or ())


def _additional(meta: Mapping[str, Any]) -> dict[str, Any]:
    raw = meta.get("additional_config")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _positive_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return int(default)
    return int(value)


def _capture_sizes(values: Any) -> tuple[int, ...]:
    try:
        raw = tuple(int(item) for item in (values or ()))
    except (TypeError, ValueError):
        return ()
    return tuple(item for item in raw if item > 0)


def source_mc2_tokens_capacity(*, max_num_reqs: int, uniform_decode_query_len: int,
                               capture_sizes: tuple[int, ...], tp: int,
                               enable_prefill_mc2: bool,
                               max_num_batched_tokens: int,
                               enable_fused_mc2: bool,
                               mega_moe_supported: bool) -> int:
    """Exact source ``set_mc2_tokens_capacity`` reduction.

    This is the capacity of the shared MC2/MegaMoe reservation, not a generic
    hardware default.  It is used by the selector and remains inspectable even
    when the selected method is AllToAll.
    """
    if tp < 1:
        raise ModelingError("mc2_tokens_capacity_tp_invalid")
    if max_num_reqs < 0 or uniform_decode_query_len < 0:
        raise ModelingError("mc2_tokens_capacity_request_invalid")
    if enable_prefill_mc2 or mega_moe_supported:
        # The second disjunct is the source's "MegaMoe and not a
        # decode-only node" branch.  The adapter has no separate decode-only
        # declaration; a caller that sets the mega capability can supply the
        # batched budget.  It never changes the ordinary path.
        if max_num_batched_tokens < 1:
            raise ModelingError("mc2_tokens_capacity_prefill_budget_missing")
        max_tokens = int(max_num_batched_tokens)
    elif capture_sizes:
        max_tokens = max(capture_sizes)
    else:
        max_tokens = max_num_reqs * uniform_decode_query_len
    per_rank = (max_tokens + tp - 1) // tp
    if enable_fused_mc2:
        per_rank = min(per_rank, _MEGA_MOE_TOKENS_PER_RANK_LIMIT
                       if mega_moe_supported else
                       _DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT)
    else:
        per_rank = min(per_rank, _MC2_TOKENS_PER_RANK_LIMIT)
    return per_rank * tp


@dataclass(frozen=True)
class MoeCommConfig:
    """One immutable source-selection input for the ordinary MoE families."""

    mode: str
    policy: str
    capacity: int
    world_size: int
    tp: int
    top_k: int
    experts_per_device: int
    enable_fused_mc2: bool = False
    mega_moe_supported: bool = False
    enable_prefill_mc2: bool = False
    max_num_batched_tokens: int = 0
    max_num_reqs: int = 0
    uniform_decode_query_len: int = 1
    capture_sizes: tuple[int, ...] = ()
    lora: bool = False
    is_draft_model: bool = False
    draft_quant_supported: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {DYNAMIC, ALWAYS_MC2}:
            raise ModelingError(f"moe_comm_mode_invalid:{self.mode}")
        allowed = {CAPACITY_AND_EXPERT_DENSITY, FUSED_OR_CAPACITY,
                   CAPACITY_AND_WORLD_SIZE, ""}
        if self.policy not in allowed:
            raise ModelingError(f"moe_comm_policy_invalid:{self.policy}")
        for name in ("capacity", "world_size", "tp", "top_k",
                     "experts_per_device", "max_num_batched_tokens",
                     "max_num_reqs", "uniform_decode_query_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"moe_comm_value_invalid:{name}")
        if self.tp < 1 or self.world_size < 1 or self.top_k < 1:
            raise ModelingError("moe_comm_positive_axes_required")
        object.__setattr__(self, "capture_sizes", tuple(self.capture_sizes))

    def select(self, num_tokens: int) -> str:
        """Selected source method for one forward's local token count."""
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 0:
            raise ModelingError("moe_comm_num_tokens_invalid")
        if self.mode == ALWAYS_MC2:
            return MC2
        if self.world_size <= 1:
            return ALLGATHER
        if self.lora:
            return ALLTOALL
        if self.policy == CAPACITY_AND_EXPERT_DENSITY:
            if (self.experts_per_device <= 24 and self.world_size >= 16
                    and num_tokens <= self.capacity):
                return MC2
            return ALLGATHER
        if self.policy == FUSED_OR_CAPACITY:
            if self.mega_moe_supported:
                return FUSED_MC2
            if self.enable_fused_mc2 and self.world_size <= 32:
                return FUSED_MC2
            if num_tokens <= self.capacity:
                return MC2
            return ALLTOALL
        if self.policy == CAPACITY_AND_WORLD_SIZE:
            if self.enable_fused_mc2 and self.mega_moe_supported:
                if (self.is_draft_model and not self.draft_quant_supported):
                    # Source fall-through for an unsupported MTP draft quant.
                    capacity = min(self.capacity,
                                   _MC2_TOKENS_PER_RANK_LIMIT * self.tp)
                else:
                    return FUSED_MC2
            else:
                capacity = self.capacity
            if num_tokens <= capacity and self.world_size > 1:
                return MC2
            if self.world_size <= self.top_k:
                return ALLGATHER
            return ALLTOALL
        # No source policy declared: the caller's static recipe remains in
        # force.  ``capacity`` is still published for diagnostics.
        return ALLTOALL

    def to_items(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("mode", self.mode), ("policy", self.policy),
            ("capacity", self.capacity), ("world_size", self.world_size),
            ("tp", self.tp), ("top_k", self.top_k),
            ("experts_per_device", self.experts_per_device),
            ("enable_fused_mc2", int(self.enable_fused_mc2)),
            ("mega_moe_supported", int(self.mega_moe_supported)),
            ("enable_prefill_mc2", int(self.enable_prefill_mc2)),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("max_num_reqs", self.max_num_reqs),
            ("uniform_decode_query_len", self.uniform_decode_query_len),
            ("capture_sizes", self.capture_sizes),
            ("lora", int(self.lora)),
            ("is_draft_model", int(self.is_draft_model)),
            ("draft_quant_supported", int(self.draft_quant_supported)),
        )

    @classmethod
    def from_items(cls, raw: Any) -> "MoeCommConfig | None":
        if raw is None:
            return None
        try:
            values = dict(raw)
            return cls(
                mode=str(values["mode"]), policy=str(values.get("policy", "")),
                capacity=int(values["capacity"]), world_size=int(values["world_size"]),
                tp=int(values["tp"]), top_k=int(values["top_k"]),
                experts_per_device=int(values["experts_per_device"]),
                enable_fused_mc2=bool(int(values.get("enable_fused_mc2", 0))),
                mega_moe_supported=bool(int(values.get("mega_moe_supported", 0))),
                enable_prefill_mc2=bool(int(values.get("enable_prefill_mc2", 0))),
                max_num_batched_tokens=int(values.get("max_num_batched_tokens", 0)),
                max_num_reqs=int(values.get("max_num_reqs", 0)),
                uniform_decode_query_len=int(values.get("uniform_decode_query_len", 1)),
                capture_sizes=tuple(int(v) for v in (values.get("capture_sizes") or ())),
                lora=bool(int(values.get("lora", 0))),
                is_draft_model=bool(int(values.get("is_draft_model", 0))),
                draft_quant_supported=bool(int(values.get("draft_quant_supported", 1))),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelingError(f"moe_comm_config_invalid:{error}") from None

    @classmethod
    def from_features(cls, features: Any, *, ep: int, top_k: int,
                      experts: int, mode: str = "") -> "MoeCommConfig | None":
        """Build the selector input from the compiled feature record.

        The ordinary case carries the deployment's ``additional_config`` and
        capture sizes.  A record without the fused-MC2 declaration keeps the
        historical static ``moe_recipe``; this is not a silent new switch.
        """
        meta = _metadata(features)
        additional = _additional(meta)
        enable_fused = bool(int(additional.get("enable_fused_mc2", 0) or 0))
        declared_fused = "enable_fused_mc2" in additional
        declared_mode = str(meta.get("moe_comm_mode", "") or "")
        arch = str(meta.get("arch_family", "") or "").strip().upper()
        mega = bool(int(
            additional.get("enable_mega_moe", 0) or
            meta.get("moe_mega_moe_supported", 0) or 0))
        provenance = bool(meta.get("source_lowerings") or meta.get("image") or
                          meta.get("image_id") or meta.get("moe_comm_mode"))
        if not mode and declared_mode in {DYNAMIC, ALWAYS_MC2}:
            mode = declared_mode
        if not mode:
            # The A5 source selector chooses ordinary MC2 at either fused flag
            # when the token count is under the same capacity crossover.  The
            # presence of the deployment's fused declaration (not its value)
            # is the evidence that this A5 policy/EP path is loaded.  Unknown
            # deployments without that declaration keep their static recipe.
            if (arch == "A5" and int(ep) > 1 and not mega
                    and (declared_fused or provenance)):
                mode = DYNAMIC
        if not mode:
            return None
        policy = str(meta.get("moe_comm_policy", "") or "").strip().lower()
        if not policy:
            policy = _ARCH_POLICY.get(arch, "")
        capture_sizes = _capture_sizes(getattr(features, "graph_buckets", ()))
        max_num_reqs = _positive_int(meta.get("max_num_seqs"), 0)
        uniform = _positive_int(meta.get("uniform_decode_query_len"), 1) or 1
        max_batched = _positive_int(meta.get("max_num_batched_tokens"), 0)
        enable_prefill = bool(int(additional.get("enable_prefill_mc2", 0) or 0))
        tp = max(1, int(getattr(features, "tp", 1) or 1))
        world = max(1, int(ep))
        capacity = source_mc2_tokens_capacity(
            max_num_reqs=max_num_reqs, uniform_decode_query_len=uniform,
            capture_sizes=capture_sizes, tp=tp,
            enable_prefill_mc2=enable_prefill,
            max_num_batched_tokens=max_batched,
            enable_fused_mc2=enable_fused or mode == ALWAYS_MC2,
            mega_moe_supported=mega)
        return cls(
            mode=mode, policy=policy, capacity=capacity, world_size=world,
            tp=tp, top_k=max(1, int(top_k)), experts_per_device=max(1, int(experts) // world),
            enable_fused_mc2=enable_fused, mega_moe_supported=mega,
            enable_prefill_mc2=enable_prefill,
            max_num_batched_tokens=max_batched, max_num_reqs=max_num_reqs,
            uniform_decode_query_len=uniform, capture_sizes=capture_sizes,
            lora=bool(int(meta.get("lora", 0) or 0)),
            is_draft_model=bool(int(meta.get("is_draft_model", 0) or 0)),
            draft_quant_supported=bool(int(meta.get("draft_quant_supported", 1) or 1)),
        )

    def for_static_mc2(self) -> "MoeCommConfig":
        return replace(self, mode=ALWAYS_MC2)


def local_tokens(geometry: Any) -> int:
    """This rank's pre-cohort request rows; never the EP-expanded total."""
    local_batch = int(getattr(geometry, "local_batch", 0) or 0) or int(
        getattr(geometry, "batch", 0) or 0)
    return sum(int(request.q_len) for request in geometry.requests[:local_batch])


def selection_bindings(config: MoeCommConfig | None, geometry: Any) -> dict[str, Any]:
    """Environment used by the ordinary family ``active`` guards."""
    if config is None:
        return {}
    tokens = local_tokens(geometry)
    selected = config.select(tokens)
    return {
        "MOE_LOCAL_TOKENS": tokens,
        "MOE_COMM_CAPACITY": config.capacity,
        "MOE_COMM_SELECTED_ALLTOALL": int(selected == ALLTOALL),
        "MOE_COMM_SELECTED_MC2": int(selected == MC2),
        "MOE_COMM_SELECTED_FUSED": int(selected == FUSED_MC2),
        "MOE_COMM_SELECTED_ALLGATHER": int(selected == ALLGATHER),
    }

