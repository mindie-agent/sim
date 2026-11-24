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
_DENSITY_MC2_MAX_EXPERTS_PER_DEVICE = 24
_DENSITY_MC2_MIN_WORLD_SIZE = 16
_FUSED_OR_CAPACITY_MAX_FUSED_WORLD_SIZE = 32
_FUSED_MC2_MODE_MEGAMOE_ROUTED = 2
_FUSED_MC2_MODE_MEGAMOE_SHARED = 3
NO_DRAFT_TOKENS = 0
BASE_DECODE_QUERY_LEN = 1

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
    fused_mc2_level: int = 0
    arch_family: str = ""
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
        if isinstance(self.fused_mc2_level, bool) or not isinstance(self.fused_mc2_level, int) or self.fused_mc2_level < 0:
            raise ModelingError("moe_comm_value_invalid:fused_mc2_level")
        object.__setattr__(self, "capture_sizes", tuple(self.capture_sizes))

    def select(self, num_tokens: int, *, phase: str = "") -> str:
        """Selected source method for one forward's TP-global DP-group token count."""
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 0:
            raise ModelingError("moe_comm_num_tokens_invalid")
        if self.mode == ALWAYS_MC2:
            return MC2
        if self.world_size <= 1:
            return ALLGATHER
        if self.lora:
            return ALLTOALL
        allow_plain_mc2 = phase != "prefill" or self.enable_prefill_mc2
        if self.policy == CAPACITY_AND_EXPERT_DENSITY:
            if (allow_plain_mc2
                    and self.experts_per_device <= _DENSITY_MC2_MAX_EXPERTS_PER_DEVICE
                    and self.world_size >= _DENSITY_MC2_MIN_WORLD_SIZE
                    and num_tokens <= self.capacity):
                return MC2
            return ALLGATHER
        if self.policy == FUSED_OR_CAPACITY:
            if self.mega_moe_supported and allow_plain_mc2:
                return FUSED_MC2
            if (allow_plain_mc2
                    and self.enable_fused_mc2
                    and self.world_size <= _FUSED_OR_CAPACITY_MAX_FUSED_WORLD_SIZE):
                return FUSED_MC2
            if allow_plain_mc2 and num_tokens <= self.capacity:
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
            if allow_plain_mc2 and num_tokens <= capacity and self.world_size > 1:
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
            ("fused_mc2_level", int(self.fused_mc2_level)),
            ("arch_family", str(self.arch_family)),
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
                fused_mc2_level=int(values.get("fused_mc2_level", int(bool(int(values.get("enable_fused_mc2", 0)))))),
                arch_family=str(values.get("arch_family", "") or ""),
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
        raw_fused = int(additional.get("enable_fused_mc2", 0) or 0)
        enable_fused = bool(raw_fused)
        declared_fused = "enable_fused_mc2" in additional
        declared_mode = str(meta.get("moe_comm_mode", "") or "")
        arch = str(meta.get("arch_family", "") or "").strip().upper()
        # Mode 3 (shared-expert MegaMoE) requires A5; on A3 clamp effective fused level to mode 2 if mega supported.
        effective_fused_level = raw_fused
        if arch != "A5" and effective_fused_level >= _FUSED_MC2_MODE_MEGAMOE_SHARED:
            effective_fused_level = _FUSED_MC2_MODE_MEGAMOE_ROUTED
        mega = bool(int(
            additional.get("enable_mega_moe", 0) or
            meta.get("moe_mega_moe_supported", 0) or 0))
        if arch == "A3" and effective_fused_level == _FUSED_MC2_MODE_MEGAMOE_ROUTED:
            mega = True
        if arch == "A5" and effective_fused_level >= _FUSED_MC2_MODE_MEGAMOE_ROUTED:
            mega = True
        provenance = bool(meta.get("source_lowerings") or meta.get("moe_comm_mode"))
        if not mode and declared_mode in {DYNAMIC, ALWAYS_MC2}:
            mode = declared_mode
        if not mode:
            if (arch in {"A2", "A3", "A5"} and int(ep) > 1
                    and (declared_fused or mega or provenance)):
                mode = DYNAMIC
        if not mode:
            return None
        policy = str(meta.get("moe_comm_policy", "") or "").strip().lower()
        if not policy:
            policy = _ARCH_POLICY.get(arch, "")
        capture_sizes = _capture_sizes(getattr(features, "graph_buckets", ()))
        max_num_reqs = _positive_int(meta.get("max_num_seqs"), 0)
        draft = _positive_int(meta.get("draft_proposal_tokens"), NO_DRAFT_TOKENS)
        uniform = _positive_int(
            meta.get("uniform_decode_query_len"),
            BASE_DECODE_QUERY_LEN + draft) or BASE_DECODE_QUERY_LEN
        max_batched = _positive_int(meta.get("max_num_batched_tokens"), 0)
        enable_prefill = bool(int(additional.get("enable_prefill_mc2", 0) or 0))
        if mega and not enable_prefill and max_batched < 1:
            max_batched = (
                max(capture_sizes) if capture_sizes
                else max(1, max_num_reqs * uniform)
            )
        tp = max(1, int(getattr(features, "tp", 1) or 1))
        world = max(1, int(ep))
        capacity = source_mc2_tokens_capacity(
            max_num_reqs=max_num_reqs, uniform_decode_query_len=uniform,
            capture_sizes=capture_sizes, tp=tp,
            enable_prefill_mc2=enable_prefill,
            max_num_batched_tokens=max_batched,
            enable_fused_mc2=enable_fused or mode == ALWAYS_MC2,
            mega_moe_supported=mega)
        raw_draft_quant = meta.get("draft_quant_supported")
        draft_quant_ok = True if raw_draft_quant is None else bool(int(raw_draft_quant))
        return cls(
            mode=mode, policy=policy, capacity=capacity, world_size=world,
            tp=tp, top_k=max(1, int(top_k)), experts_per_device=max(1, int(experts) // world),
            enable_fused_mc2=enable_fused,
            fused_mc2_level=effective_fused_level,
            arch_family=arch,
            mega_moe_supported=mega,
            enable_prefill_mc2=enable_prefill,
            max_num_batched_tokens=max_batched, max_num_reqs=max_num_reqs,
            uniform_decode_query_len=uniform, capture_sizes=capture_sizes,
            lora=bool(int(meta.get("lora", 0) or 0)),
            is_draft_model=bool(int(meta.get("is_draft_model", 0) or 0)),
            draft_quant_supported=draft_quant_ok,
        )

    def for_static_mc2(self) -> "MoeCommConfig":
        return replace(self, mode=ALWAYS_MC2)

