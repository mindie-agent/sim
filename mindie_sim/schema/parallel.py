"""Primitive units and input parsing.

[Role] Common numerical contracts independent of device vendors.
[Invariants] Units and declared precision are explicit.
[Timeouts & Locks] Pure arithmetic and validation.
[Side-Effect Policy] No I/O or mutable global state.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Mapping, NewType, Sequence
from mindie_sim.schema.common import ModelingError, _int, _mapping, _unknown
from mindie_sim.schema.runtime_features import (
    LOCAL_PARALLEL_DEFAULTS, LOCAL_TP_AXIS_NAMES, PARALLEL_FEATURE_NAMES,
    normalize_parallel_features,
)

if TYPE_CHECKING:
    from mindie_sim.schema.model import ModelSpec


GlobalRank = NewType("GlobalRank", int)
ReplicaLocalRank = NewType("ReplicaLocalRank", int)
DieIndex = NewType("DieIndex", int)


@dataclass(frozen=True, slots=True)
class RankCoord:
    """Strong-typed multi-axis physical and logical rank coordinate."""

    global_rank: GlobalRank
    die_index: DieIndex
    replica: int
    stage: int
    local_rank: ReplicaLocalRank
    tp_lane: int
    pcp_lane: int
    dcp_lane: int
    ep_rank: int

def factor_dcp_axes(cp: int, *, pcp: int, tp: int, dp: int) -> tuple[int, int, int]:
    """Canonical 3-axis DCP factorization ``(pcp_factor, tp_gather, dp_width)``."""
    cp = max(1, int(cp))
    if cp <= 1:
        return (1, 1, 1)
    pcp = max(1, int(pcp))
    tp = max(1, int(tp))
    pcp_factor = math.gcd(cp, pcp) if pcp > 1 else 1
    rem = cp // pcp_factor
    tp_gather = math.gcd(rem, tp)
    dp_width = rem // tp_gather
    return (pcp_factor, tp_gather, dp_width)


def local_query_heads(num_heads: int, tp: int) -> int:
    """Per-rank query heads; H must be divisible by TP (no padding)."""
    heads, width = int(num_heads), max(1, int(tp))
    if heads <= 0 or heads % width != 0:
        raise ModelingError(f"attention_query_heads_not_divisible:{heads}%{width}")
    return heads // width


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    """Per-rank KV heads with explicit replication when K < TP."""
    heads, width = int(num_kv_heads), max(1, int(tp))
    if heads <= 0:
        raise ModelingError(f"attention_kv_heads_invalid:{heads}")
    if heads >= width:
        if heads % width != 0:
            raise ModelingError(f"attention_kv_heads_not_divisible:{heads}%{width}")
        return heads // width
    if width % heads != 0:
        raise ModelingError(
            f"attention_kv_head_replication_indivisible:{width}%{heads}")
    return 1



_CP_PLAN_ALIASES = {
    "auto": "auto",
    "": "auto",
    "dcp": "dcp",
    "decode_context_parallel": "dcp",
    "kv_context_shard_lse_merge": "dcp",
    "pcp": "pcp",
    "prefill_context_parallel": "pcp",
    "pcp_query_split_kv_gather": "pcp",
    "pcp_dcp": "pcp_dcp",
    "pcp+dcp": "pcp_dcp",
    "pcp_and_dcp": "pcp_dcp",
    "dcp+pcp": "pcp_dcp",
    "prefill_pcp_decode_dcp": "pcp_dcp",
    "pcp_prefill_dcp_decode": "pcp_dcp",
    "dsa_cp": "dsa_cp",
    "dsa-cp": "dsa_cp",
    "dsa_cp_query_token_sp": "dsa_cp",
    "dsa_cp_sp_full_kv": "dsa_cp",
    "none": "none",
}

#: Dual-phase plan: PCP owns the prefill query shard and K/V gather, DCP owns
#: the decode KV context shard and LSE merge.  The two degrees are orthogonal
#: (PCP is a rank axis, DCP reuses the TP group), so both can be enabled.
CP_PLAN_PCP_DCP = "pcp_dcp"
CP_PLAN_KINDS = ("none", "dcp", "pcp", "pcp_dcp", "dsa_cp")
#: Upstream vLLM / vLLM-Ascend axis spellings accepted beside the short names.
PARALLEL_AXIS_ALIASES = {
    "tensor_parallel_size": "tp",
    "data_parallel_size": "dp",
    "pipeline_parallel_size": "pp",
    "expert_parallel_size": "ep",
    "decode_context_parallel_size": "cp",
    "prefill_context_parallel_size": "pcp",
}
#: Non-axis parallel switches accepted in the same mapping.
PARALLEL_SWITCHES = frozenset({
    "enable_expert_parallel", "prefill_cp_plan", "decode_cp_plan",
})
_PHASE_PLAN_ALIASES = {
    "": "", "none": "none", "auto": "auto",
    "pcp": "pcp", "prefill_context_parallel": "pcp",
    "dcp": "dcp", "decode_context_parallel": "dcp",
    "dsa_cp": "dsa_cp", "dsa-cp": "dsa_cp",
}

def _normalise_cp_plan(value: Any, cp: int, pcp: int, tp: int) -> str:
    """Resolve and validate the declared plan name against the two degrees."""
    raw = str(value if value is not None else "auto").strip().lower()
    plan = _CP_PLAN_ALIASES.get(raw)
    if plan is None:
        raise ModelingError(f"cp_plan_unsupported:{raw}")
    if plan == "auto":
        # Both degrees declared: PCP owns prefill, DCP owns decode.  This is
        # the orthogonal dual-phase plan, not a conflict.
        if int(pcp) > 1 and int(cp) > 1:
            plan = CP_PLAN_PCP_DCP
        elif int(pcp) > 1:
            plan = "pcp"
        elif int(cp) > 1:
            plan = "dcp"
        else:
            plan = "none"
    if plan == "dsa_cp":
        # DSA-CP is query-token SP over the TP group; it is neither DCP nor
        # PCP, and both exclusions are reported with their own reason.
        if cp > 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_dcp")
        if pcp > 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_pcp")
        if tp <= 1:
            raise ModelingError("dsa_cp_query_token_sp_requires_tp")
        return plan
    if pcp > 1 and plan not in {"pcp", CP_PLAN_PCP_DCP}:
        raise ModelingError(f"pcp_requires_pcp_plan:{plan}")
    if pcp <= 1 and plan in {"pcp", CP_PLAN_PCP_DCP}:
        raise ModelingError("pcp_plan_requires_pcp")
    if cp > 1 and plan not in {"dcp", CP_PLAN_PCP_DCP}:
        raise ModelingError(f"cp_requires_dcp_plan:{plan}")
    if plan == "dcp" and cp <= 1:
        raise ModelingError("dcp_plan_requires_cp")
    if plan == CP_PLAN_PCP_DCP and cp <= 1:
        raise ModelingError("pcp_dcp_plan_requires_cp")
    return plan


_CP_PHASES = {
    "pcp": ("pcp", "none"),
    "dcp": ("none", "dcp"),
    CP_PLAN_PCP_DCP: ("pcp", "dcp"),
    "dsa_cp": ("dsa_cp", "none"),
    "none": ("none", "none"),
}


def _plan_from_phases(prefill: str, decode: str) -> str:
    pair = (prefill or "none", decode or "none")
    for plan, phases in _CP_PHASES.items():
        if phases == pair:
            return plan
    raise ModelingError(f"cp_phase_plans_unsupported:{prefill}+{decode}")


def _normalise_phase_plan(value: Any, field: str) -> str:
    raw = str(value if value is not None else "").strip().lower()
    plan = _PHASE_PLAN_ALIASES.get(raw)
    if plan is None or plan == "auto":
        raise ModelingError(f"{field}_unsupported:{raw}")
    return plan


def _axis_int(raw: Mapping[str, Any], name: str, alias: str,
              prefix: str) -> int | None:
    """One axis declared twice must agree; a silent pick would hide a conflict."""
    direct = raw.get(name)
    aliased = raw.get(alias) if alias else None
    if direct is not None and aliased is not None:
        if _int(direct, f"{prefix}.{name}_invalid", minimum=1) != _int(
                aliased, f"{prefix}.{alias}_invalid", minimum=1):
            raise ModelingError(f"{prefix}.{name}_conflict:{direct}!={aliased}")
    value = direct if direct is not None else aliased
    if value is None:
        return None
    return _int(value, f"{prefix}.{name}_invalid", minimum=1)


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
    indexer + LSE merge), ``pcp`` (prefill query split + K/V gather, replicated
    decode), ``dsa_cp`` (query-token SP over the TP group with a
    full/replicated KV/index cache, local query TopK and a token all-gather
    output) or ``pcp_dcp`` (both degrees declared: PCP owns prefill, DCP owns
    decode).  ``auto`` resolves from the declared degrees and is stored
    resolved; with ``pcp > 1`` **and** ``cp > 1`` it resolves to ``pcp_dcp``.

    ``prefill_cp_plan``/``decode_cp_plan`` expose the resolved phase strategy.
    They may also be declared independently (``pcp`` + ``dcp``), in which case
    they must agree with any explicitly declared ``cp_plan``.
    """

    dp: int = 1
    tp: int = 1
    ep: int = 1
    pp: int = 1
    cp: int = 1
    dcp: int = 1
    pcp: int = 1
    cp_plan: str = "auto"
    #: Raw vLLM-Ascend ``additional_config.enable_dsa_cp`` declaration.  It is
    #: the source switch; it resolves to the ``dsa_cp`` query-token SP plan.
    expert_tp: int | None = None
    expert_dp: int | None = None
    pcp_lane_policy: str = "park_extra_lanes"
    rank_table: tuple[tuple[int, ...], ...] | None = None
    enable_dsa_cp: bool = False
    #: Node-local dense-subgraph TP widths.  They partition ranks inside one
    #: node/stage/PCP lane without adding a world axis; global ``tp`` remains
    #: the attention tensor-parallel degree.
    q_proj_tp: int = 1
    o_proj_tp: int = 1
    dcp_linear_tp: int = 1
    mlp_tp: int = 1
    shared_expert_tp: int = 1
    linear_proj_tp: int = 1
    lm_head_tp: int = 1
    #: Extra resident physical routed-expert slots per EP owner rank.
    redundant_experts_per_rank: int = 0
    #: External data-parallel replica multiplier outside one serving engine's
    #: internal DP/PP/PCP/TP rank mesh (``world_size = external_dp*dp*pp*pcp*tp``).
    external_dp: int = 1

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "ep", "pp", "cp", "dcp", "pcp", "external_dp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"{name}_invalid:{value}")
        cp, dcp = self.cp, self.dcp
        if cp != 1 and dcp != 1 and cp != dcp:
            raise ModelingError(f"cp_dcp_conflict:{cp}!={dcp}")
        value = cp if cp != 1 else dcp
        if cp != value:
            object.__setattr__(self, "cp", value)
        if dcp != value:
            object.__setattr__(self, "dcp", value)
        plan = _normalise_cp_plan(self.cp_plan, value, self.pcp, self.tp)
        enable_dsa_cp = bool(self.enable_dsa_cp)
        if enable_dsa_cp:
            # The source switch is authoritative; auto/none resolves to the
            # query-token SP plan, and a conflicting DCP/PCP plan is refused.
            if plan not in {"dsa_cp", "none"}:
                raise ModelingError(
                    f"enable_dsa_cp_conflicts_with_plan:{plan}")
            if plan == "none":
                if value > 1:
                    raise ModelingError("dsa_cp_query_token_sp_excludes_dcp")
                if int(self.tp) <= 1:
                    raise ModelingError("dsa_cp_query_token_sp_requires_tp")
                plan = "dsa_cp"
        elif plan == "dsa_cp":
            enable_dsa_cp = True
        object.__setattr__(self, "enable_dsa_cp", enable_dsa_cp)
        if plan != self.cp_plan:
            object.__setattr__(self, "cp_plan", plan)
        policy = str(self.pcp_lane_policy or "park_extra_lanes").strip().lower()
        if policy not in {"park_extra_lanes", "replica_lanes", "shard_decode_requests"}:
            raise ModelingError(f"pcp_lane_policy_unsupported:{policy}")
        if policy != self.pcp_lane_policy:
            object.__setattr__(self, "pcp_lane_policy", policy)
        for name in ("expert_tp", "expert_dp"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ModelingError(f"{name}_invalid:{value}")
        for name in (*LOCAL_TP_AXIS_NAMES, "redundant_experts_per_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelingError(f"{name}_invalid:{value}")
        if any(getattr(self, name) < 1 for name in LOCAL_TP_AXIS_NAMES):
            raise ModelingError("local_projection_tp_must_be_positive")
        if self.dcp_linear_tp > 1:
            if self.q_proj_tp > 1 and self.q_proj_tp != self.dcp_linear_tp:
                raise ModelingError(
                    f"local_tp_conflict:dcp_linear_tp={self.dcp_linear_tp},q_proj_tp={self.q_proj_tp}"
                )
            if self.o_proj_tp > 1 and self.o_proj_tp != self.dcp_linear_tp:
                raise ModelingError(
                    f"local_tp_conflict:dcp_linear_tp={self.dcp_linear_tp},o_proj_tp={self.o_proj_tp}"
                )
        table = self.rank_table
        if table is not None:
            if not isinstance(table, tuple) or not table or not all(
                isinstance(group, tuple) and group
                and all(isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0 for rank in group)
                for group in table
            ):
                raise ModelingError("rank_table_invalid")
            flat = tuple(rank for group in table for rank in group)
            if len(set(flat)) != len(flat):
                raise ModelingError("rank_table_duplicate_rank")


    @property
    def prefill_cp_plan(self) -> str:
        """Resolved prefill strategy (``pcp``/``dsa_cp``/``none``)."""
        return _CP_PHASES[self.cp_plan][0]

    @property
    def decode_cp_plan(self) -> str:
        """Resolved decode strategy (``dcp``/``none``)."""
        return _CP_PHASES[self.cp_plan][1]

    @classmethod
    def from_mapping(cls, value: Any, prefix: str = "parallel") -> "ParallelSpec":
        raw = _mapping(value, f"{prefix}_invalid")
        # Upstream vLLM / vLLM-Ascend spellings are first-class aliases; a
        # repeated declaration must agree instead of one value silently
        # winning.
        allowed = {
            "dp", "tp", "ep", "pp", "cp", "dcp", "pcp",
            "prefill_context_parallel_size", "cp_plan",
            "context_parallel_plan", "world_size", "enable_dsa_cp",
            "tensor_parallel_size", "data_parallel_size",
            "pipeline_parallel_size", "expert_parallel_size",
            "decode_context_parallel_size", "enable_expert_parallel",
            "prefill_cp_plan", "decode_cp_plan",
            "expert_tp", "expert_dp", "pcp_lane_policy", "rank_table",
            "q_proj_tp", "o_proj_tp", "dcp_linear_tp", "mlp_tp",
            "shared_expert_tp", "linear_proj_tp", "kda_proj_tp", "lm_head_tp",
            "redundant_experts_per_rank",
            "external_dp", "external_data_parallel_size",
        }
        # ``prefill_cp_plan``/``decode_cp_plan`` are declaration aliases for
        # ``cp_plan``; they are resolved into the single plan name above and
        # exposed as properties, so ``replace(spec, cp_plan=...)`` stays sound.
        _unknown(raw, allowed, prefix)
        tp = _axis_int(raw, "tp", "tensor_parallel_size", prefix)
        dp = _axis_int(raw, "dp", "data_parallel_size", prefix)
        pp = _axis_int(raw, "pp", "pipeline_parallel_size", prefix)
        external_dp = _axis_int(raw, "external_dp", "external_data_parallel_size", prefix)
        tp = 1 if tp is None else tp
        dp = 1 if dp is None else dp
        pp = 1 if pp is None else pp
        external_dp = 1 if external_dp is None else external_dp
        pcp = _axis_int(raw, "pcp", "prefill_context_parallel_size", prefix) or 1
        ep = _axis_int(raw, "ep", "expert_parallel_size", prefix)
        enable_ep = raw.get("enable_expert_parallel")
        if enable_ep is not None and not isinstance(enable_ep, bool):
            raise ModelingError(f"{prefix}.enable_expert_parallel_invalid")
        if enable_ep is True:
            if ep is not None and ep == 1:
                raise ModelingError(
                    f"{prefix}.expert_parallel_conflict:ep=1,enable_expert_parallel=True")
            # vLLM enables EP over the attention DP*PCP*TP domain by default.
            ep = ep if ep is not None else tp * pcp * dp
        elif enable_ep is False and ep is not None and ep > 1:
            raise ModelingError(
                f"{prefix}.expert_parallel_conflict:ep={ep},enable_expert_parallel=False")
        ep = 1 if ep is None else ep
        cp = _int(raw["cp"], f"{prefix}.cp_invalid", minimum=1) \
            if raw.get("cp") is not None else None
        cp_alias = raw.get("decode_context_parallel_size")
        if cp_alias is not None:
            alias_value = _int(cp_alias, f"{prefix}.cp_invalid", minimum=1)
            if cp is not None and cp != alias_value:
                raise ModelingError(
                    f"{prefix}.cp_conflict:{cp}!={alias_value}")
            cp = alias_value
        dcp = (
            _int(raw["dcp"], f"{prefix}.dcp_invalid", minimum=1)
            if raw.get("dcp") is not None else None
        )
        if cp is not None and dcp is not None and cp != dcp:
            raise ModelingError(f"{prefix}.cp_dcp_conflict:{cp}!={dcp}")
        cp = cp if cp is not None else (dcp if dcp is not None else 1)
        plan_raw = raw.get("cp_plan", raw.get("context_parallel_plan", "auto"))
        if "cp_plan" in raw and "context_parallel_plan" in raw:
            if (_normalise_cp_plan(raw["cp_plan"], cp, pcp, tp)
                    != _normalise_cp_plan(raw["context_parallel_plan"], cp, pcp, tp)):
                raise ModelingError(f"{prefix}.cp_plan_conflict")
        declared_prefill = _normalise_phase_plan(
            raw.get("prefill_cp_plan", ""), "prefill_cp_plan")
        declared_decode = _normalise_phase_plan(
            raw.get("decode_cp_plan", ""), "decode_cp_plan")
        if declared_prefill or declared_decode:
            expected = _plan_from_phases(declared_prefill, declared_decode)
            raw_plan = str(plan_raw if plan_raw is not None else "auto").strip().lower()
            if (raw_plan not in {"", "auto"}
                    and _CP_PLAN_ALIASES.get(raw_plan) != expected):
                raise ModelingError(
                    f"cp_plan_conflicts_with_phase_plans:{raw_plan}!={expected}")
            plan_raw = expected
        enable_dsa_cp = raw.get("enable_dsa_cp", False)
        if not isinstance(enable_dsa_cp, bool):
            raise ModelingError(f"{prefix}.enable_dsa_cp_invalid")
        rank_table = None
        if raw.get("rank_table") is not None:
            raw_table = raw["rank_table"]
            if not isinstance(raw_table, (list, tuple)) or not raw_table:
                raise ModelingError(f"{prefix}.rank_table_invalid")
            groups = []
            for group in raw_table:
                if not isinstance(group, (list, tuple)) or not group:
                    raise ModelingError(f"{prefix}.rank_table_invalid")
                row = []
                for rank in group:
                    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                        raise ModelingError(f"{prefix}.rank_table_fractional_or_invalid:{rank}")
                    row.append(rank)
                groups.append(tuple(row))
            rank_table = tuple(groups)
        features = normalize_parallel_features({
            key: raw[key] for key in PARALLEL_FEATURE_NAMES if key in raw})
        spec = cls(
            dp=dp, tp=tp, ep=ep, pp=pp,
            cp=cp, dcp=cp, pcp=pcp, cp_plan=str(plan_raw),
            enable_dsa_cp=enable_dsa_cp,
            expert_tp=raw.get("expert_tp"), expert_dp=raw.get("expert_dp"),
            pcp_lane_policy=str(raw.get("pcp_lane_policy", "park_extra_lanes")),
            rank_table=rank_table,
            external_dp=external_dp,
            **{key: features[key] for key in LOCAL_PARALLEL_DEFAULTS
               if key in features},
        )
        # ``world_size`` is derived, but when a source declares it the value
        # must agree with the axes.  Silently dropping it would accept a
        # topology that contradicts its own deployment record.
        if raw.get("world_size") is not None:
            declared = _int(raw["world_size"], f"{prefix}.world_size_invalid", minimum=1)
            if declared != spec.world_size:
                raise ModelingError(
                    f"{prefix}.world_size_conflict:{declared}!={spec.world_size}")
        return spec

    @property
    def world_size(self) -> int:
        # World size = external_dp * dp * pp * pcp * tp. DCP and EP reuse
        # existing ranks and never multiply into world_size.
        return self.external_dp * self.dp * self.pp * self.pcp * self.tp

    @property
    def lane_plane(self) -> int:
        """Ranks inside one ``(pp, pcp_lane)`` coordinate: ``DP * TP``."""
        return self.dp * self.tp

    @property
    def ep_domain(self) -> int:
        if self.rank_table is None:
            if self.pcp > 1 and self.ep * self.etp > self.lane_plane:
                return self.dp * self.pcp * self.tp
            return self.lane_plane
        total = len({rank for group in self.rank_table for rank in group})
        if self.pp > 1 and total % self.pp == 0 and total // self.pp >= self.ep:
            return total // self.pp
        return total

    @property
    def active_pcp_lanes(self) -> int:
        return self.pcp if self.pcp_lane_policy == "replica_lanes" else 1

    @property
    def pcp_shard_decode_requests(self) -> bool:
        return self.pcp_lane_policy == "shard_decode_requests"

    @property
    def parked_pcp_lanes(self) -> int:
        return self.pcp - self.active_pcp_lanes

    @property
    def active_replicas(self) -> int:
        """Independent decode replicas per PP stage."""
        return self.dp * self.active_pcp_lanes

    @property
    def effective_q_proj_tp(self) -> int:
        return self.dcp_linear_tp if self.dcp_linear_tp > 1 else self.q_proj_tp

    @property
    def effective_o_proj_tp(self) -> int:
        return self.dcp_linear_tp if self.dcp_linear_tp > 1 else self.o_proj_tp

    @property
    def local_tp_widths(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in LOCAL_TP_AXIS_NAMES)

    @property
    def has_local_tp(self) -> bool:
        return any(width > 1 for width in self.local_tp_widths)

    @property
    def dcp_pcp_factor(self) -> int:
        """DCP factor absorbed along the PCP rank axis."""
        return factor_dcp_axes(self.cp, pcp=self.pcp, tp=self.tp, dp=self.dp)[0]

    @property
    def dcp_tp_gather(self) -> int:
        """DCP factor absorbed across the intra-replica TP group."""
        return factor_dcp_axes(self.cp, pcp=self.pcp, tp=self.tp, dp=self.dp)[1]

    @property
    def dcp_dp_width(self) -> int:
        """DCP factor absorbed across DP replicas when ``cp > tp * pcp``."""
        return factor_dcp_axes(self.cp, pcp=self.pcp, tp=self.tp, dp=self.dp)[2]

    @property
    def cp_spans_dp(self) -> bool:
        return self.dcp_dp_width > 1

    @property
    def ep_spans_pcp(self) -> bool:
        if self.pcp > 1:
            ep_tp_width = max(1, self.tp // self.etp)
            return self.ep > self.dp * ep_tp_width
        return False

    @property
    def ep_spans_dp(self) -> bool:
        ep_tp_width = max(1, self.tp // self.etp)
        return (self.ep > ep_tp_width * self.pcp) if self.ep_spans_pcp else (self.ep > ep_tp_width)

    @property
    def ep_dp_width(self) -> int:
        if not self.ep_spans_dp:
            return 1
        ep_tp_width = max(1, self.tp // self.etp)
        return self.ep // (ep_tp_width * self.pcp) if self.ep_spans_pcp else self.ep // ep_tp_width

    @property
    def etp(self) -> int:
        # Expert tensor-parallel shard width.  An explicit ``expert_tp`` wins;
        # otherwise EP groups live in the DP*TP lane plane and the remaining TP
        # factor may shard each expert.  EP is never another world axis.
        if self.expert_tp is not None:
            return int(self.expert_tp)
        if self.ep <= 1:
            return self.tp
        if self.ep <= self.tp and self.tp % self.ep == 0:
            return self.tp // self.ep
        return 1

    def validate_expert_placement(self) -> None:
        """Check that EP groups and tensor shards factor the declared rank plane."""
        if self.rank_table is not None:
            flat = tuple(rank for group in self.rank_table for rank in group)
            groups = {len(group) for group in self.rank_table}
            if max(groups, default=0) > self.world_size:
                raise ModelingError("rank_table_group_exceeds_world")
            if len(flat) > self.world_size:
                raise ModelingError(
                    f"rank_table_ranks_exceed_world:{len(flat)}>{self.world_size}"
                )
            if groups != {self.ep}:
                raise ModelingError(f"rank_table_ep_membership_mismatch:{sorted(groups)}!={self.ep}")
        plane = self.ep_domain
        expert_tp = self.etp
        expert_dp = self.expert_dp if self.expert_dp is not None else plane // (self.ep * expert_tp)
        if self.ep > plane:
            raise ModelingError(f"ep_exceeds_lane_plane:{self.ep}>{plane}")
        if expert_tp < 1 or expert_dp < 1 or self.ep * expert_tp * expert_dp != plane:
            raise ModelingError(
                f"ep_expert_factorization_invalid:"
                f"ep={self.ep},expert_tp={expert_tp},expert_dp={expert_dp},plane={plane}")
        if self.tp % expert_tp != 0:
            raise ModelingError(f"expert_tp_not_divisor_of_tp:{expert_tp}!|{self.tp}")
        ep_tp_width = max(1, self.tp // self.etp)
        if not self.ep_spans_pcp:
            if self.ep > 1 and self.ep <= ep_tp_width and ep_tp_width % self.ep != 0:
                raise ModelingError(f"ep_not_divisor_of_tp:{self.ep}%{ep_tp_width}")
            if self.rank_table is None and self.ep > ep_tp_width and self.ep % ep_tp_width != 0:
                raise ModelingError(f"ep_not_multiple_of_tp:{self.ep}%{ep_tp_width}")
        else:
            local_domain = ep_tp_width * self.pcp
            if self.ep <= local_domain and local_domain % self.ep != 0:
                raise ModelingError(f"ep_not_divisor_of_tp_pcp:{local_domain}%{self.ep}")
            if self.ep > local_domain and self.ep % local_domain != 0:
                raise ModelingError(f"ep_not_multiple_of_tp_pcp:{self.ep}%{local_domain}")

    def ep_group_coords(
        self,
        replica: int,
        stage: int = 0,
        lane: ReplicaLocalRank = ReplicaLocalRank(0),
    ) -> tuple[tuple[int, int, ReplicaLocalRank], ...]:
        """Return ``(replica, stage, local_rank)`` coordinates for one EP group."""
        if self.rank_table is not None:
            raise ModelingError("ep_group_coords_requires_implicit_placement")
        total_replicas = self.external_dp * self.dp
        width = self.tp * self.pcp
        lane_int = int(lane)
        if not (0 <= replica < total_replicas and 0 <= stage < self.pp):
            raise ModelingError(f"placement_coordinate_unknown:{replica}:{stage}")
        if not 0 <= lane_int < width:
            raise ModelingError(f"placement_lane_unknown:{lane_int}")
        self.validate_expert_placement()
        if not (self.ep_spans_dp or self.ep_spans_pcp):
            start = (lane_int // self.ep) * self.ep
            return tuple(
                (replica, stage, ReplicaLocalRank(idx))
                for idx in range(start, start + self.ep)
            )
        ep_dp_w = max(1, self.ep_dp_width)
        ep_tp_w = max(1, self.tp // self.etp)
        if self.ep_spans_pcp:
            ep_local = self.ep // ep_dp_w
            tp_span = math.gcd(ep_local, ep_tp_w)
            copy_count = ep_local // tp_span
        else:
            tp_span = ep_tp_w
            copy_count = 1
        copy, tp_lane = divmod(lane_int, self.tp)
        offset = (tp_lane // tp_span) * tp_span
        ext_base = (replica // self.dp) * self.dp
        first_rep = ext_base + ((replica - ext_base) // ep_dp_w) * ep_dp_w
        first_copy = (copy // copy_count) * copy_count
        return tuple(
            (rep, stage, ReplicaLocalRank(c * self.tp + t))
            for rep in range(first_rep, first_rep + ep_dp_w)
            for c in range(first_copy, first_copy + copy_count)
            for t in range(offset, offset + tp_span)
        )

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
        self.validate_expert_placement()
        if self.cp > 1:
            max_domain = self.dp * self.tp * self.pcp
            if self.cp > max_domain:
                raise ModelingError(
                    f"cp_exceeds_parallel_domain:cp={self.cp}>dp*tp*pcp={max_domain}"
                )
            pcp_factor = self.dcp_pcp_factor
            rem = self.cp // pcp_factor
            tp_gather = self.dcp_tp_gather
            dp_width = self.dcp_dp_width
            if (self.pcp % pcp_factor != 0 or self.cp % pcp_factor != 0
                    or self.tp % tp_gather != 0 or rem % tp_gather != 0
                    or self.dp % dp_width != 0
                    or pcp_factor * tp_gather * dp_width != self.cp):
                raise ModelingError(
                    f"dcp_factorization_invalid:dcp={self.cp},pcp={self.pcp},"
                    f"tp={self.tp},dp={self.dp}"
                )
        # ``E % EP != 0`` is a legal placement.  The physical slot plan owns the
        # ceil(E/EP) base slots plus automatic padding copies, so a nondivisible
        # expert population never drops a logical expert and never needs a
        # global divisibility rejection here.
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
        if self.cp > 1 or self.pcp > 1 or self.cp_plan == "dsa_cp":
            # DCP reuses TP devices (``tp % cp == 0``); PCP is a separate rank
            # axis with its own prefill schedule; DSA-CP is query-token SP over
            # the TP group with ``cp=1``.  The plan decides which guards apply;
            # an unsupported combination is refused, never dropped.
            from mindie_sim.catalog.context_parallel import validate_cp_plan

            validate_cp_plan(model, self)
        if self.ep > 1 and not model.num_experts:
            raise ModelingError("ep_requires_moe_model")

    def to_dict(self) -> dict[str, Any]:
        out = {key: getattr(self, key) for key in (
            "dp", "tp", "ep", "pp", "cp", "pcp", "cp_plan",
            "prefill_cp_plan", "decode_cp_plan", "enable_dsa_cp",
            "expert_tp", "expert_dp", "pcp_lane_policy",
            "o_proj_tp", "linear_proj_tp", "lm_head_tp",
            "redundant_experts_per_rank")}
        for key in ("q_proj_tp", "dcp_linear_tp", "mlp_tp", "shared_expert_tp"):
            val = getattr(self, key)
            if val != 1:
                out[key] = val
        if self.external_dp != 1:
            out["external_dp"] = self.external_dp
        if self.rank_table is not None:
            out["rank_table"] = [list(group) for group in self.rank_table]
        return out
