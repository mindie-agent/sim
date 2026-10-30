"""Primitive units and input parsing.

[Role] Common numerical contracts independent of device vendors.
[Invariants] Units and declared precision are explicit.
[Timeouts & Locks] Pure arithmetic and validation.
[Side-Effect Policy] No I/O or mutable global state.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from mindie_sim.schema.common import ModelingError, _int, _mapping, _unknown
from typing import Any, Mapping

_CP_PLAN_ALIASES = {
    "auto": "auto",
    "": "auto",
    "dcp": "dcp",
    "decode_context_parallel": "dcp",
    "kv_context_shard_lse_merge": "dcp",
    "pcp": "pcp",
    "prefill_context_parallel": "pcp",
    "pcp_query_split_kv_gather": "pcp",
    "dsa_cp": "dsa_cp",
    "dsa-cp": "dsa_cp",
    "dsa_cp_sharded_topk_lse_merge": "dsa_cp",
    "none": "none",
}

def _normalise_cp_plan(value: Any, cp: int, pcp: int) -> str:
    """Resolve and validate the declared plan name against the two degrees."""
    raw = str(value if value is not None else "auto").strip().lower()
    plan = _CP_PLAN_ALIASES.get(raw)
    if plan is None:
        raise ModelingError(f"cp_plan_unsupported:{raw}")
    if plan == "auto":
        if int(pcp) > 1:
            plan = "pcp"
        elif int(cp) > 1:
            plan = "dcp"
        else:
            plan = "none"
    if int(pcp) > 1 and plan != "pcp":
        raise ModelingError(f"pcp_requires_pcp_plan:{plan}")
    if int(pcp) <= 1 and plan == "pcp":
        raise ModelingError("pcp_plan_requires_pcp")
    if plan in {"dcp", "dsa_cp"} and int(cp) <= 1:
        raise ModelingError(f"{plan}_plan_requires_cp")
    return plan

@dataclass(frozen=True)
class ParallelSpec:
    """Declared parallel decomposition of one serving replica.

    ``cp`` is DCP (upstream vLLM ``decode_context_parallel_size``): it shards
    the contiguous KV context inside the **TP group** and adds no world axis.
    ``dcp`` is the legacy spelling and remains a parse alias for ``cp`` only.

    ``pcp`` is PCP (``prefill_context_parallel_size``): it adds a real rank axis
    (``world_size = dp*pp*tp*pcp``), splits prefill query tokens across the PCP
    group and gathers K/V.  It is not an alias of ``cp``.

    ``cp_plan`` names the execution strategy: ``dcp`` (KV shard + replicated
    indexer + LSE merge), ``dsa_cp`` (sharded sparse indexer TopK + global merge
    + LSE merge) or ``pcp`` (prefill query split + K/V gather, replicated
    decode).  ``auto`` resolves from the declared degrees and is stored resolved.
    """

    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    cp: int = 1
    dcp: int = 1
    pcp: int = 1
    cp_plan: str = "auto"

    def __post_init__(self) -> None:
        cp, dcp = int(self.cp), int(self.dcp)
        if cp < 1 or dcp < 1:
            raise ModelingError("cp_invalid")
        if cp != 1 and dcp != 1 and cp != dcp:
            raise ModelingError(f"cp_dcp_conflict:{cp}!={dcp}")
        value = cp if cp != 1 else dcp
        if cp != value:
            object.__setattr__(self, "cp", value)
        if dcp != value:
            object.__setattr__(self, "dcp", value)
        pcp = int(self.pcp)
        if pcp < 1:
            raise ModelingError("pcp_invalid")
        if pcp != self.pcp:
            object.__setattr__(self, "pcp", pcp)
        plan = _normalise_cp_plan(self.cp_plan, value, pcp)
        if plan != self.cp_plan:
            object.__setattr__(self, "cp_plan", plan)

    @classmethod
    def from_mapping(cls, value: Any, prefix: str = "parallel") -> "ParallelSpec":
        raw = _mapping(value, f"{prefix}_invalid")
        allowed = {
            "dp", "tp", "ep", "pp", "cp", "dcp", "pcp",
            "prefill_context_parallel_size", "cp_plan",
            "context_parallel_plan", "world_size",
        }
        _unknown(raw, allowed, prefix)
        cp = (
            _int(raw["cp"], f"{prefix}.cp_invalid", minimum=1)
            if raw.get("cp") is not None else None
        )
        dcp = (
            _int(raw["dcp"], f"{prefix}.dcp_invalid", minimum=1)
            if raw.get("dcp") is not None else None
        )
        if cp is not None and dcp is not None and cp != dcp:
            raise ModelingError(f"{prefix}.cp_dcp_conflict:{cp}!={dcp}")
        cp = cp if cp is not None else (dcp if dcp is not None else 1)
        pcp = _int(
            raw.get("pcp", raw.get("prefill_context_parallel_size", 1)),
            f"{prefix}.pcp_invalid", minimum=1,
        )
        plan_raw = raw.get("cp_plan", raw.get("context_parallel_plan", "auto"))
        return cls(
            dp=_int(raw.get("dp", 1), f"{prefix}.dp_invalid", minimum=1),
            tp=_int(raw.get("tp", 1), f"{prefix}.tp_invalid", minimum=1),
            ep=_int(raw.get("ep", 1), f"{prefix}.ep_invalid", minimum=1),
            pp=_int(raw.get("pp", 1), f"{prefix}.pp_invalid", minimum=1),
            cp=cp, dcp=cp, pcp=pcp, cp_plan=str(plan_raw),
        )

    @property
    def world_size(self) -> int:
        # PCP is a real rank axis (DP * PP * PCP * TP); DCP is not.
        return self.dp * self.pp * self.tp * self.pcp

    @property
    def ep_domain(self) -> int:
        return self.dp * self.tp

    @property
    def ep_spans_dp(self) -> bool:
        return self.ep > self.tp

    @property
    def etp(self) -> int:
        # Expert tensor-parallel shard width.  EP groups live in the DP*TP
        # domain; when EP is smaller than TP the remaining TP factor may shard
        # each expert.  EP itself is never added as another world axis.
        if self.ep <= 1:
            return self.tp
        if self.ep <= self.tp and self.tp % self.ep == 0:
            return self.tp // self.ep
        return 1

    def validate(
        self, model: "ModelSpec", *,
        layer_partition: Sequence[int] | None = None,
    ) -> None:
        """Check the topology against the model.

        ``layer_partition`` is an explicit per-stage physical-layer split (for
        example ``PPPlan.partition``).  A declared partition replaces the
        implicit even rule, so a complete uneven split (K3 PP2 ``47/46``) is
        legal while a broken one is refused.  Callers without a plan keep the
        even ``layers % pp`` requirement unchanged.
        """
        if self.ep > self.ep_domain:
            raise ModelingError(f"ep_exceeds_dp_tp_domain:{self.ep}>{self.ep_domain}")
        if self.ep_domain % self.ep != 0:
            raise ModelingError(f"ep_not_divisor_of_dp_tp_domain:{self.ep_domain}%{self.ep}")
        if self.ep > 1 and model.num_experts and model.num_experts % self.ep != 0:
            raise ModelingError(f"experts_not_divisible_by_ep:{model.num_experts}%{self.ep}")
        if layer_partition is None:
            if model.layers % self.pp != 0:
                raise ModelingError(
                    f"layers_not_divisible_by_pp:{model.layers}%{self.pp}"
                )
        else:
            counts = tuple(int(value) for value in layer_partition)
            if len(counts) != self.pp:
                raise ModelingError(f"pp_partition_count:{len(counts)}!={self.pp}")
            if sum(counts) != model.layers:
                raise ModelingError(
                    f"pp_partition_layers:{sum(counts)}!={model.layers}"
                )
            if any(count < 1 for count in counts):
                raise ModelingError(f"pp_partition_empty_stage:{list(counts)}")
        if model.num_heads % self.tp != 0:
            raise ModelingError(f"heads_not_divisible_by_tp:{model.num_heads}%{self.tp}")
        if self.ep > 1 and self.ep <= self.tp and self.tp % self.ep != 0:
            raise ModelingError(f"ep_not_divisor_of_tp:{self.ep}%{self.tp}")
        if self.cp > 1 or self.pcp > 1:
            # DCP reuses TP devices (``tp % cp == 0``); PCP is a separate rank
            # axis with its own prefill schedule.  The plan decides which guards
            # apply; an unsupported combination is refused, never dropped.
            from mindie_sim.catalog.context_parallel import validate_cp_plan

            validate_cp_plan(model, self)
        if self.ep > 1 and not model.num_experts:
            raise ModelingError("ep_requires_moe_model")

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("dp", "tp", "ep", "pp", "cp", "pcp", "cp_plan")}
