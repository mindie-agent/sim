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
from mindie_sim.schema.runtime_features import (
    LOCAL_PARALLEL_DEFAULTS, ONE, PARALLEL_FEATURE_NAMES, ZERO,
    normalize_parallel_features,
)
from typing import Any, Mapping

def local_query_heads(num_heads: int, tp: int) -> int:
    """Per-rank query heads; H must be divisible by TP (no padding)."""
    heads, width = int(num_heads), max(ONE_VALUE, int(tp))
    if heads <= ZERO_VALUE or heads % width != ZERO_VALUE:
        raise ModelingError(f"attention_query_heads_not_divisible:{heads}%{width}")
    return heads // width


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    """Per-rank KV heads with explicit replication when K < TP."""
    heads, width = int(num_kv_heads), max(ONE_VALUE, int(tp))
    if heads <= ZERO_VALUE:
        raise ModelingError(f"attention_kv_heads_invalid:{heads}")
    if heads >= width:
        if heads % width != ZERO_VALUE:
            raise ModelingError(f"attention_kv_heads_not_divisible:{heads}%{width}")
        return heads // width
    if width % heads != ZERO_VALUE:
        raise ModelingError(
            f"attention_kv_head_replication_indivisible:{width}%{heads}")
    return ONE_VALUE


ZERO_VALUE = 0
ONE_VALUE = 1

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
        if int(cp) > 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_dcp")
        if int(pcp) > 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_pcp")
        if int(tp) <= 1:
            raise ModelingError("dsa_cp_query_token_sp_requires_tp")
        return plan
    if int(pcp) > 1 and plan not in {"pcp", CP_PLAN_PCP_DCP}:
        raise ModelingError(f"pcp_requires_pcp_plan:{plan}")
    if int(pcp) <= 1 and plan in {"pcp", CP_PLAN_PCP_DCP}:
        raise ModelingError("pcp_plan_requires_pcp")
    if plan == "dcp" and int(cp) <= 1:
        raise ModelingError("dcp_plan_requires_cp")
    if plan == CP_PLAN_PCP_DCP and int(cp) <= 1:
        raise ModelingError("pcp_dcp_plan_requires_cp")
    return plan


def _phase_plans(plan: str) -> tuple[str, str]:
    """Resolved (prefill, decode) strategy for one declared plan."""
    return {
        "pcp": ("pcp", "none"),
        "dcp": ("none", "dcp"),
        CP_PLAN_PCP_DCP: ("pcp", "dcp"),
        # DSA-CP is a prefill query-token plan; decode stays ordinary.
        "dsa_cp": ("dsa_cp", "none"),
        "none": ("none", "none"),
    }[plan]


def _plan_from_phases(prefill: str, decode: str) -> str:
    """Derive the single plan name from independently declared phase plans."""
    pair = (prefill, decode)
    if pair == ("pcp", "dcp"):
        return CP_PLAN_PCP_DCP
    if pair == ("pcp", "none") or pair == ("pcp", ""):
        return "pcp"
    if pair == ("none", "dcp") or pair == ("", "dcp"):
        return "dcp"
    if pair in {("none", "none"), ("", "")}:
        return "none"
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
    o_proj_tp: int = ONE
    linear_proj_tp: int = ONE
    lm_head_tp: int = ONE
    #: Extra resident physical routed-expert slots per EP owner rank.
    redundant_experts_per_rank: int = ZERO


    def __post_init__(self) -> None:
        for name in ("dp", "tp", "ep", "pp", "cp", "dcp", "pcp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelingError(f"{name}_invalid:{value}")
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
        plan = _normalise_cp_plan(self.cp_plan, value, pcp, int(self.tp))
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
        if policy not in {"park_extra_lanes", "replica_lanes"}:
            raise ModelingError(f"pcp_lane_policy_unsupported:{policy}")
        if policy != self.pcp_lane_policy:
            object.__setattr__(self, "pcp_lane_policy", policy)
        for name in ("expert_tp", "expert_dp"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ModelingError(f"{name}_invalid:{value}")
        for name in ("o_proj_tp", "linear_proj_tp", "lm_head_tp",
                     "redundant_experts_per_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < ZERO:
                raise ModelingError(f"{name}_invalid:{value}")
        if self.o_proj_tp < ONE or self.linear_proj_tp < ONE or self.lm_head_tp < ONE:
            raise ModelingError("local_projection_tp_must_be_positive")
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
        return _phase_plans(self.cp_plan)[0]

    @property
    def decode_cp_plan(self) -> str:
        """Resolved decode strategy (``dcp``/``none``)."""
        return _phase_plans(self.cp_plan)[1]

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
            "o_proj_tp", "linear_proj_tp", "kda_proj_tp", "lm_head_tp",
            "redundant_experts_per_rank",
        }
        # ``prefill_cp_plan``/``decode_cp_plan`` are declaration aliases for
        # ``cp_plan``; they are resolved into the single plan name above and
        # exposed as properties, so ``replace(spec, cp_plan=...)`` stays sound.
        _unknown(raw, allowed, prefix)
        tp = _axis_int(raw, "tp", "tensor_parallel_size", prefix)
        dp = _axis_int(raw, "dp", "data_parallel_size", prefix)
        pp = _axis_int(raw, "pp", "pipeline_parallel_size", prefix)
        tp = 1 if tp is None else tp
        dp = 1 if dp is None else dp
        pp = 1 if pp is None else pp
        ep = _axis_int(raw, "ep", "expert_parallel_size", prefix)
        enable_ep = raw.get("enable_expert_parallel")
        if enable_ep is not None and not isinstance(enable_ep, bool):
            raise ModelingError(f"{prefix}.enable_expert_parallel_invalid")
        if enable_ep is True:
            if ep is not None and ep == 1:
                raise ModelingError(
                    f"{prefix}.expert_parallel_conflict:ep=1,enable_expert_parallel=True")
            # vLLM enables EP over the attention DP*TP domain by default.
            ep = ep if ep is not None else tp * dp
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
        pcp = _int(
            raw.get("pcp", raw.get("prefill_context_parallel_size", 1)),
            f"{prefix}.pcp_invalid", minimum=1,
        )
        plan_raw = raw.get("cp_plan", raw.get("context_parallel_plan", "auto"))
        declared_prefill = _normalise_phase_plan(
            raw.get("prefill_cp_plan", ""), "prefill_cp_plan")
        declared_decode = _normalise_phase_plan(
            raw.get("decode_cp_plan", ""), "decode_cp_plan")
        # ``to_dict`` exports the resolved phase properties.  A round-tripped
        # default spec therefore carries ``("none", "none")`` even when a
        # caller subsequently declares ``pcp``/``cp``; those two values are
        # the serialized default, not a phase declaration, so let the explicit
        # plan and degree axes remain authoritative.
        if (declared_prefill, declared_decode) == ("none", "none"):
            declared_prefill = declared_decode = ""
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
        # PCP is a real rank axis (DP * PP * PCP * TP); DCP is not.
        return self.dp * self.pp * self.tp * self.pcp

    @property
    def lane_plane(self) -> int:
        """Ranks inside one ``(pp, pcp_lane)`` coordinate: ``DP * TP``."""
        return self.dp * self.tp

    @property
    def ep_domain(self) -> int:
        # Homogeneous EP is confined to the lane plane.  The legacy PCP-spanning
        # reading survives only for an explicitly tabulated placement.
        return len({rank for group in self.rank_table for rank in group}) \
            if self.rank_table is not None else self.lane_plane

    @property
    def active_pcp_lanes(self) -> int:
        return self.pcp if self.pcp_lane_policy == "replica_lanes" else 1

    @property
    def parked_pcp_lanes(self) -> int:
        return self.pcp - self.active_pcp_lanes

    @property
    def active_replicas(self) -> int:
        """Independent decode replicas per PP stage."""
        return self.dp * self.active_pcp_lanes

    @property
    def ep_spans_pcp(self) -> bool:
        # A homogeneous group cannot cross a PCP lane boundary.  Only an
        # explicit rank table can declare a broader EP domain.
        if self.rank_table is not None and self.pcp > 1:
            return self.ep > self.dp * self.tp
        return False

    @property
    def ep_spans_dp(self) -> bool:
        return (self.ep > self.tp * self.pcp) if self.ep_spans_pcp else (self.ep > self.tp)

    @property
    def ep_dp_width(self) -> int:
        if not self.ep_spans_dp:
            return 1
        return self.ep // (self.tp * self.pcp) if self.ep_spans_pcp else self.ep // self.tp

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
        if self.rank_table is None:
            plane = self.lane_plane
            expert_tp = self.etp
            expert_dp = self.expert_dp if self.expert_dp is not None else (
                plane // (self.ep * expert_tp) if expert_tp > 0
                and plane % (self.ep * expert_tp) == 0 else 0)
            if self.ep > plane:
                raise ModelingError(
                    f"ep_exceeds_lane_plane:{self.ep}>{plane};"
                    "ep_across_pcp_requires_explicit_rank_table")
            if expert_tp < 1 or expert_dp < 1 or self.ep * expert_tp * expert_dp != plane:
                raise ModelingError(
                    f"ep_expert_factorization_invalid:"
                    f"ep={self.ep},expert_tp={expert_tp},expert_dp={expert_dp},plane={plane}")
            if self.tp % expert_tp != 0:
                raise ModelingError(f"expert_tp_not_divisor_of_tp:{expert_tp}!|{self.tp}")
        else:
            expected = self.lane_plane
            groups = {len(group) for group in self.rank_table}
            if max(groups, default=0) > self.world_size:
                raise ModelingError("rank_table_group_exceeds_world")
            if not groups <= {self.ep} and self.ep > 1:
                raise ModelingError(f"rank_table_ep_membership_mismatch:{sorted(groups)}!={self.ep}")
            del expected
        if self.cp > 1:
            if self.cp > self.tp or self.tp % self.cp != 0:
                raise ModelingError(f"dcp_subdivides_tp:{self.cp}!|{self.tp}")
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
        if not self.ep_spans_pcp:
            if self.ep > 1 and self.ep <= self.tp and self.tp % self.ep != 0:
                raise ModelingError(f"ep_not_divisor_of_tp:{self.ep}%{self.tp}")
        else:
            local_domain = self.tp * self.pcp
            if self.ep <= local_domain and local_domain % self.ep != 0:
                raise ModelingError(f"ep_not_divisor_of_tp_pcp:{local_domain}%{self.ep}")
            if self.ep > local_domain and self.ep % local_domain != 0:
                raise ModelingError(f"ep_not_multiple_of_tp_pcp:{self.ep}%{local_domain}")
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
        if self.rank_table is not None:
            out["rank_table"] = [list(group) for group in self.rank_table]
        return out
