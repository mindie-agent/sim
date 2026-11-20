"""Structural parallel arms, expert-slot banks, routing ledgers and split rules.

[Role]
  Derive primary and multi-axis parallel policies (``tp_x_ep``, ``dp_x_ep``,
  ``dcp_x_ep``, ``pcp_x_ep``, ``pcp_x_dcp``, ``pcp_x_tp``), expert-bank sizing
  and integer routing ledgers aligned with ``expert_slot_plan``, layer
  signatures, model archetypes and deterministic train/heldout split freezes.
[Invariants]
  * ``parallel_policy`` returns the deterministic primary grid used by model
    summaries and baseline arms.
  * Every multi-axis arm obeys ``num_heads % tp == 0``, KDA
    ``heads % (tp*pcp) == 0``, GQA KV-head DCP domain divisibility, and
    ``mindie_sim.schema.parallel`` padded expert-slot legality
    (``E % EP != 0`` is permitted via ``ceil(E / EP)`` base slots).
  * Collective ``group_size`` resolves from the operator's declared
    ``group_axis`` (``tp``, ``ep``, ``pcp``, ``cp``/``dcp``, ``dp``, ``pp``),
    never blindly ``tp``.
  * ``assign_split`` freezes train/heldout solely from declared geometry.
[Timeouts & Locks]
  Pure bounded integer arithmetic; O(layers + experts) per model, zero locks.
[Side-Effect Policy]
  Stateless pure functions returning plain JSON-serializable dictionaries.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import accumulate
import json
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.expert_slots import expert_slot_plan

DEFAULT_TP, ALT_TP_QUARTER, ALT_TP_WIDE = 8, 4, 16
DENSE_EP, LOCAL_MOE_EP, WIDE_MOE_EP, TRILLION_MOE_EP = 1, 8, 16, 64
DEFAULT_CP_DEGREE, WIDE_MLA_DCP_DEGREE = 2, 4
MAX_MULTI_AXIS_REP_LAYERS = 32
TRILLION_PARAMETER_COUNT = 1_000_000_000_000
MIB = 1024 * 1024
ROUTED_EXPERT_MATRIX_FACTOR = 3
SEQ_512, SEQ_2K, SEQ_8K, SEQ_16K = 512, 2048, 8192, 16384
SEQ_32K, SEQ_128K, SEQ_256K, SEQ_400K = 32768, 131072, 262144, 409600
MAX_EXPORT_MODEL_LEN = 425984
PREFILL_BASE_TOKENS, PREFILL_CHUNK_CAP = 128, 2048
BATCH_4, BATCH_8, BATCH_16 = 4, 8, 16
SEQUENCE_LADDER_TOKENS = (SEQ_512, SEQ_2K, SEQ_8K, SEQ_32K, SEQ_128K, SEQ_256K, SEQ_400K)
HELDOUT_LONG_KV_LENGTHS = frozenset({SEQ_16K, SEQ_256K, SEQ_400K})
ZERO_EXPERT_FRACTIONS, ROUTING_DISTRIBUTIONS = (0.0, 0.25, 0.5, 0.75), ("uniform", "zipf_s1.2")
HELDOUT_MATRIX_ROWS, HELDOUT_BATCH, HELDOUT_KV_LENGTH = frozenset({3, 12, 96, 384, 1536, 3072}), 16, 16384
HELDOUT_ROUTING_DISTRIBUTION, HELDOUT_ROUTING_ZERO_FRACTION = "zipf_s1.2", 0.5
HELDOUT_FABRIC_BYTES = frozenset({1 << 15, 1 << 19, 1 << 23, 1 << 27})
EXPERT_BYTES_PER_ELEMENT = {
    "bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "int8": 1.0,
    "mxfp4": 0.5, "fp4": 0.5, "int4": 0.5, "w4a8": 0.5, "w4a16": 0.5,
    "compressed_tensors_w4a16": 0.5, "mxfp4_routed_only": 0.5,
}
ARCHETYPES = (
    (1, "dense_gqa", "qwen3-32b"),
    (2, "moe_gqa", "qwen3-30b-a3b qwen3-235b-a22b"),
    (3, "gemma_swa_full", "gemma-4-31b gemma-4-26b-a4b"),
    (4, "sparse_indexed_gqa_moe", "minimax-m3 minimax-m3.1-flash step-5-preview"),
    (5, "absorbed_mla_moe", "deepseek-v2-lite deepseek-v3 deepseek-r1 deepseek-v3.2 deepseek-v3.2-prune-3l kimi-k2 kimi-k2-thinking kimi-k2.5 kimi-k2.6"),
    (6, "mla_dsa_pooled_indexer", "glm-5.1 glm-5.2 glm-5.3"),
    (7, "compressed_mla_v4", "deepseek-v4-10t deepseek-v4-pro-0813 deepseek-v4-flash-0731 deepseek-v4.1-flash"),
    (8, "kda_mla_dsa_moe", "glm-5.3-flash kimi-k3 kimi-k3.1"),
    (9, "qwen_hybrid_gdn_gqa", "qwen3-next-80b-a3b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b qwen3.5-397b-a17b qwen3.6-27b qwen3.6-35b-a3b qwen3.8-27b qwen3.8-2.4t-a95b"),
    (10, "qwen4exp_and_drafts", "qwen3.8-flash-next kimi-k3-dspark kimi-k3-dspark-block5"),
)
ARCHETYPE_BY_MODEL = {model_id: (number, name) for number, name, ids in ARCHETYPES for model_id in ids.split()}
GEOMETRY_ARMS: dict[str, dict[str, Any]] = {
    "decode": {"phase": "decode", "tokens": 1, "context": SEQ_2K, "sequences": BATCH_8},
    "decode_b16": {"phase": "decode", "tokens": 1, "context": SEQ_2K, "sequences": BATCH_16},
    "decode_kv512": {"phase": "decode", "tokens": 1, "context": SEQ_512, "sequences": BATCH_8},
    "decode_kv8192": {"phase": "decode", "tokens": 1, "context": SEQ_8K, "sequences": BATCH_8},
    "decode_kv16384": {"phase": "decode", "tokens": 1, "context": SEQ_16K, "sequences": BATCH_8},
    "decode_kv32768": {"phase": "decode", "tokens": 1, "context": SEQ_32K, "sequences": BATCH_4},
    "decode_kv131072": {"phase": "decode", "tokens": 1, "context": SEQ_128K, "sequences": 2},
    "decode_kv262144": {"phase": "decode", "tokens": 1, "context": SEQ_256K, "sequences": 1},
    "decode_kv409600": {"phase": "decode", "tokens": 1, "context": SEQ_400K, "sequences": 1},
    "prefill": {"phase": "prefill", "tokens": PREFILL_BASE_TOKENS, "context": 0, "sequences": 1},
    "prefill_s512": {"phase": "prefill", "tokens": SEQ_512, "context": 0, "sequences": 1},
    "prefill_s2048": {"phase": "prefill", "tokens": SEQ_2K, "context": 0, "sequences": 1},
    "prefill_s8192": {"phase": "prefill", "tokens": PREFILL_CHUNK_CAP, "context": SEQ_8K - PREFILL_CHUNK_CAP, "sequences": 1},
    "prefill_s32768": {"phase": "prefill", "tokens": PREFILL_CHUNK_CAP, "context": SEQ_32K - PREFILL_CHUNK_CAP, "sequences": 1},
    "prefill_s131072": {"phase": "prefill", "tokens": PREFILL_CHUNK_CAP, "context": SEQ_128K - PREFILL_CHUNK_CAP, "sequences": 1},
    "prefill_s262144": {"phase": "prefill", "tokens": PREFILL_CHUNK_CAP, "context": SEQ_256K - PREFILL_CHUNK_CAP, "sequences": 1},
    "prefill_s409600": {"phase": "prefill", "tokens": PREFILL_CHUNK_CAP, "context": SEQ_400K - PREFILL_CHUNK_CAP, "sequences": 1},
}
DECODE_LADDER_PROBES = (
    "decode_b16", "decode_kv512", "decode_kv8192", "decode_kv16384",
    "decode_kv32768", "decode_kv131072", "decode_kv262144", "decode_kv409600",
)
PREFILL_LADDER_PROBES = (
    "prefill_s512", "prefill_s2048", "prefill_s8192",
    "prefill_s32768", "prefill_s131072", "prefill_s262144", "prefill_s409600",
)
HELDOUT_GEOMETRY_PROBES = (*DECODE_LADDER_PROBES, *PREFILL_LADDER_PROBES)


class PlanError(RuntimeError):
    """A catalog/geometry/routing condition that would make a plan lie."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def short_digest(value: Any, prefix: str = "") -> str:
    return prefix + hashlib.sha256(canonical_json(value).encode()).hexdigest()[:16]


def probe_geometries_for(geometries: Sequence[str], *, enabled: bool = True) -> tuple[str, ...]:
    """Return sequence-ladder probe geometries matching requested phases."""
    if not enabled:
        return ()
    phases = {GEOMETRY_ARMS[g]["phase"] for g in geometries if g in GEOMETRY_ARMS}
    out: list[str] = []
    if "decode" in phases:
        out.extend(DECODE_LADDER_PROBES)
    if "prefill" in phases:
        out.extend(PREFILL_LADDER_PROBES)
    return tuple(out)


def parameter_count(model: Any) -> float:
    value = getattr(model, "parameter_count", 0)
    return float(value() if callable(value) else value)


def parallel_policy(model: Any) -> dict[str, Any]:
    """Declare the primary virtual rank grid used for local replay compilation."""
    if int(model.num_experts) <= 0:
        ep, dp, tier = DENSE_EP, 1, "dense"
    elif parameter_count(model) >= TRILLION_PARAMETER_COUNT:
        ep, dp, tier = TRILLION_MOE_EP, TRILLION_MOE_EP // DEFAULT_TP, "trillion_parameter_moe"
    else:
        ep, dp, tier = LOCAL_MOE_EP, 1, "local_moe"
    return {
        "tp": DEFAULT_TP, "dp": dp, "ep": ep, "pp": 1, "pcp": 1, "dcp": 1,
        "sp": None, "num_devices": dp * DEFAULT_TP, "tier": tier,
        "virtual_rank_zero": ep > DEFAULT_TP,
    }


def _make_arm(*, tp: int, dp: int, ep: int, pcp: int = 1, dcp: int = 1, tier: str) -> dict[str, Any]:
    num_devices = dp * tp * pcp
    return {
        "tp": tp, "dp": dp, "ep": ep, "pp": 1, "pcp": pcp, "dcp": dcp,
        "sp": None, "num_devices": num_devices, "tier": tier,
        "virtual_rank_zero": ep > DEFAULT_TP or num_devices > DEFAULT_TP,
    }


def _heads_divisible(plans: Sequence[Any], tp: int, pcp: int = 1) -> bool:
    for plan in plans:
        heads = int(getattr(plan, "num_heads", 0) or 0)
        if heads > 0 and heads % tp != 0:
            return False
        if str(getattr(plan, "attention", "")) == "kda" and heads > 0 and heads % (tp * pcp) != 0:
            return False
    return True


def _gqa_dcp_compatible(plans: Sequence[Any], tp: int, dcp: int, pcp: int = 1) -> bool:
    if dcp <= 1:
        return True
    if pcp <= 1 and (tp <= 1 or tp % dcp != 0):
        return False
    if pcp > 1 and dcp not in {pcp, tp * pcp}:
        return False
    for plan in plans:
        if str(getattr(plan, "attention", "")) != "gqa":
            continue
        if pcp > 1 and dcp != pcp:
            return False
        kv = max(1, int(getattr(plan, "num_kv_heads", 1) or 1))
        domain = (tp // kv if pcp <= 1 else ((tp // kv) * pcp if tp >= kv else (pcp if kv % tp == 0 else 0)))
        if domain < dcp or domain % dcp != 0:
            return False
    return True


def _has_dsa_indexer(plans: Sequence[Any]) -> bool:
    return any(str(getattr(plan, "attention", "")) == "mla" and bool(getattr(plan, "indexer", False)) for plan in plans)


def _ep_slot_legal(experts: int, ep: int) -> bool:
    """True when ``expert_slot_plan`` can host ``experts`` across ``ep`` ranks."""
    return experts > 0 and 1 <= ep <= experts


def multi_axis_parallel_arms(
    model: Any, *, geometries: Sequence[str], tracks: Sequence[str], probe_ladder: bool = False,
) -> list[tuple[str | None, dict[str, Any], list[tuple[str, str]]]]:
    """Return ``(axis_tag, parallel_dict, [(track, geometry_name), ...])`` arms."""
    primary = parallel_policy(model)
    aux = getattr(model, "auxiliary", None)
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if (aux is not None and getattr(aux, "standalone_draft", False)) or not plans:
        return [(None, primary, [])]
    experts, tier, ep_base = int(getattr(model, "num_experts", 0) or 0), str(primary["tier"]), int(primary["ep"])
    has_merged, has_unmerged = "merged" in tracks, "unmerged_pr" in tracks
    base_geoms = [g for g in geometries if g in {"prefill", "decode"}]
    if not has_merged or not base_geoms:
        return [(None, primary, [])]
    all_geoms = [*base_geoms, *probe_geometries_for(base_geoms, enabled=probe_ladder)]
    merged_both = [("merged", g) for g in all_geoms]
    has_gdn = any(str(getattr(p, "attention", "")) == "gdn" for p in plans)
    pcp_pairs = ([("unmerged_pr", g) for g in all_geoms] if (has_gdn and has_unmerged)
                 else ([] if has_gdn else list(merged_both)))
    decode_only = [("merged", g) for g in all_geoms if GEOMETRY_ARMS[g]["phase"] == "decode"]
    if has_unmerged and _has_dsa_indexer(plans) and "decode" in base_geoms:
        decode_only = [*decode_only, ("unmerged_pr", "decode")]
    arms: list[tuple[str | None, dict[str, Any], list[tuple[str, str]]]] = [(None, primary, [])]
    if experts > 0:
        if _ep_slot_legal(experts, ep_base) and _heads_divisible(plans, 1, 1):
            arms.append(("dp_x_ep", _make_arm(tp=1, dp=ep_base, ep=ep_base, tier=tier), list(merged_both)))
        if decode_only:
            for cand_tp in (DEFAULT_TP, ALT_TP_WIDE):
                cand_ep = max(ep_base, cand_tp)
                if (_ep_slot_legal(experts, cand_ep) and _heads_divisible(plans, cand_tp, 1)
                        and _gqa_dcp_compatible(plans, cand_tp, DEFAULT_CP_DEGREE, 1)):
                    arms.append(("dcp_x_ep", _make_arm(tp=cand_tp, dp=cand_ep // cand_tp, ep=cand_ep,
                                                       dcp=DEFAULT_CP_DEGREE, tier=tier), list(decode_only)))
                    break
        if pcp_pairs:
            for cand_tp in (ALT_TP_QUARTER, DEFAULT_TP):
                cand_ep = max(ep_base, cand_tp * DEFAULT_CP_DEGREE)
                if _ep_slot_legal(experts, cand_ep) and _heads_divisible(plans, cand_tp, DEFAULT_CP_DEGREE):
                    arms.append(("pcp_x_ep", _make_arm(tp=cand_tp, dp=cand_ep // (cand_tp * DEFAULT_CP_DEGREE),
                                                       ep=cand_ep, pcp=DEFAULT_CP_DEGREE, tier=tier), list(pcp_pairs)))
                    break
            has_gqa = any(str(getattr(p, "attention", "")) == "gqa" for p in plans)
            cands = (((DEFAULT_CP_DEGREE, WIDE_MLA_DCP_DEGREE), (ALT_TP_QUARTER, DEFAULT_CP_DEGREE))
                     if not has_gqa else ((ALT_TP_QUARTER, DEFAULT_CP_DEGREE), (DEFAULT_TP, DEFAULT_CP_DEGREE)))
            for cand_tp, cand_dcp in cands:
                cand_ep = max(ep_base, cand_tp * DEFAULT_CP_DEGREE)
                if (_ep_slot_legal(experts, cand_ep) and _heads_divisible(plans, cand_tp, DEFAULT_CP_DEGREE)
                        and _gqa_dcp_compatible(plans, cand_tp, cand_dcp, DEFAULT_CP_DEGREE)):
                    arms.append(("pcp_x_dcp", _make_arm(tp=cand_tp, dp=cand_ep // (cand_tp * DEFAULT_CP_DEGREE),
                                                        ep=cand_ep, pcp=DEFAULT_CP_DEGREE, dcp=cand_dcp, tier=tier), list(pcp_pairs)))
                    break
    elif pcp_pairs:
        for cand_tp in (ALT_TP_QUARTER, DEFAULT_TP):
            if _heads_divisible(plans, cand_tp, DEFAULT_CP_DEGREE):
                arms.append(("pcp_x_tp", _make_arm(tp=cand_tp, dp=1, ep=DENSE_EP,
                                                   pcp=DEFAULT_CP_DEGREE, tier=tier), list(pcp_pairs)))
                break
        for cand_tp in (ALT_TP_QUARTER, DEFAULT_TP):
            if (_heads_divisible(plans, cand_tp, DEFAULT_CP_DEGREE)
                    and _gqa_dcp_compatible(plans, cand_tp, DEFAULT_CP_DEGREE, DEFAULT_CP_DEGREE)):
                arms.append(("pcp_x_dcp", _make_arm(tp=cand_tp, dp=1, ep=DENSE_EP, pcp=DEFAULT_CP_DEGREE,
                                                    dcp=DEFAULT_CP_DEGREE, tier=tier), list(pcp_pairs)))
                break
    return arms


def representative_layer_depth(model: Any, signatures: Sequence[Mapping[str, Any]]) -> int:
    """Smallest prefix of ``model.layers`` covering every body layer signature."""
    total = int(model.layers)
    if total <= 2:
        return total
    needed = {0, 1}
    for record in signatures:
        if record.get("scope") != "body":
            continue
        ids = record.get("layer_ids") or ()
        if ids and int(ids[0]) < MAX_MULTI_AXIS_REP_LAYERS:
            needed.add(int(ids[0]))
            if int(ids[0]) == 0 and len(ids) > 1 and int(ids[1]) < MAX_MULTI_AXIS_REP_LAYERS:
                needed.add(int(ids[1]))
    return min(total, max(needed) + 1)


def collective_group_size(metadata: Mapping[str, Any], parallel: Mapping[str, Any]) -> int:
    """Resolve the physical group size of a collective from its ``group_axis``."""
    explicit = int(metadata.get("group_size") or metadata.get("participants") or 0)
    if explicit > 0:
        return explicit
    axis = str(metadata.get("group_axis") or "tp")
    tp, pcp = int(parallel.get("tp") or 1), int(parallel.get("pcp") or 1)
    dcp = int(parallel.get("dcp") or parallel.get("cp") or 1)
    ep, dp, pp = int(parallel.get("ep") or 1), int(parallel.get("dp") or 1), int(parallel.get("pp") or 1)
    raw_etp = parallel.get("expert_tp")
    etp = (int(raw_etp) if raw_etp is not None
           else (tp if ep <= 1 else (tp // ep if ep <= tp and tp % ep == 0 else 1)))
    axis_sizes = {
        "tp": tp, "ep": ep, "pcp": pcp, "cp": dcp, "dcp": dcp, "dp": dp, "pp": pp,
        "etp": etp,
    }
    return max(1, int(axis_sizes.get(axis, tp)))


def model_archetype(model_id: str, model: Any) -> dict[str, Any]:
    """Return the reporting-only ten-family label, never a kernel selector."""
    known = ARCHETYPE_BY_MODEL.get(model_id)
    if known:
        return {"archetype_id": known[0], "name": known[1], "basis": "catalog_membership"}
    plans, aux = tuple(model.layer_plans), getattr(model, "auxiliary", None)
    if aux is not None and aux.standalone_draft:
        return {"archetype_id": 10, "name": "qwen4exp_and_drafts", "basis": "standalone_draft"}
    if model.attention == "gdn":
        return {"archetype_id": 9, "name": "qwen_hybrid_gdn_gqa", "basis": "structural_gdn"}
    if model.attention == "mla" and any(p.compress_ratio or p.window for p in plans):
        return {"archetype_id": 7, "name": "compressed_mla_v4", "basis": "compressed_mla"}
    if model.attention == "mla" and any(p.indexer or p.indexer_type not in {"", "none"} for p in plans):
        return {"archetype_id": 6, "name": "mla_dsa_pooled_indexer", "basis": "indexed_mla"}
    if model.attention == "mla":
        return {"archetype_id": 5, "name": "absorbed_mla_moe", "basis": "absorbed_mla"}
    if any(p.indexer or p.candidate_filter for p in plans):
        return {"archetype_id": 4, "name": "sparse_indexed_gqa_moe", "basis": "indexed_gqa"}
    if model.quant.name.startswith("bf16_gemma") or any(p.k_eq_v for p in plans):
        return {"archetype_id": 3, "name": "gemma_swa_full", "basis": "gemma_storage"}
    if model.num_experts > 0:
        return {"archetype_id": 2, "name": "moe_gqa", "basis": "moe_gqa"}
    return {"archetype_id": 1, "name": "dense_gqa", "basis": "dense_gqa"}


def slot_plan_for(experts: int, ep: int, *, redundant: int = 0) -> Any:
    """Resolve an ``ExpertSlotPlan`` for ``experts`` across ``ep`` ranks."""
    if ep <= 0:
        raise PlanError(f"expert_slot_ep_invalid:{ep}")
    return expert_slot_plan(
        None, SimpleNamespace(num_experts=int(experts), index=0),
        SimpleNamespace(ep=int(ep), redundant_experts_per_rank=int(redundant)),
    )


def expert_bank_bytes(model: Any, parallel: Mapping[str, Any], *, slots: Any = None) -> dict[str, Any] | None:
    """Resident routed-expert bank aligned with ``expert_slot_plan`` padded/redundant slots."""
    experts = int(model.num_experts)
    if experts <= 0:
        return None
    ep = int(parallel["ep"])
    redundant = int(parallel.get("redundant_experts_per_rank") or 0)
    resolved = slots if slots is not None else slot_plan_for(experts, ep, redundant=redundant)
    local_base = int(resolved.base_slots_by_rank[0])
    local_extra = int(resolved.redundant_slots_by_rank[0])
    local_total = len(resolved.expert_ids_by_rank[0])
    padded_slots = ep * local_base - experts
    intermediate = int(model.moe_intermediate_size) or max(
        (int(plan.moe_intermediate_size) for plan in model.layer_plans), default=0,
    )
    fmt = str(model.quant.expert_weight_format)
    if fmt not in EXPERT_BYTES_PER_ELEMENT:
        raise PlanError(f"unknown_expert_weight_format:{fmt}")
    width = EXPERT_BYTES_PER_ELEMENT[fmt]
    elements = local_total * ROUTED_EXPERT_MATRIX_FACTOR * int(model.hidden_size) * intermediate
    return {
        "local_experts": local_total, "base_local_experts": local_base,
        "redundant_local_experts": local_extra,
        "local_weight_expert_ids": list(resolved.expert_ids_by_rank[0]),
        "padded_expert_slots": padded_slots, "expert_count": experts, "ep": ep,
        "expert_weight_format": fmt, "bytes_per_element": width,
        "matrices_per_expert": ROUTED_EXPERT_MATRIX_FACTOR,
        "resident_expert_bank_bytes": int(elements * width),
        "resident_expert_bank_mib": round(elements * width / MIB, 3),
        "scope": "three_routed_matrices_before_scales_and_workspace",
        "basis": "expert_slot_plan_declared_format",
    }


def _normalized_layer(plan: Any, scope: str) -> dict[str, Any]:
    drop = {"index", "native_layer_id", "kv_owner_layer_id", "index_k_owner_layer_id",
            "topk_owner_layer_id", "candidate_source_layer_id", "index_source_layer_id"}
    return {**{k: v for k, v in asdict(plan).items() if k not in drop}, "scope": scope}


def layer_signatures(model: Any) -> list[dict[str, Any]]:
    """Group every declared body/draft layer by its normalized structure."""
    plans = [(plan, "body") for plan in model.layer_plans]
    aux = getattr(model, "auxiliary", None)
    if aux is not None and not aux.standalone_draft:
        for name in ("mtp", "dspark"):
            if (draft := getattr(aux, name, None)) is not None:
                plans.extend((plan, name) for plan in draft.layer_plans)
    groups: dict[str, dict[str, Any]] = {}
    for plan, scope in plans:
        payload = _normalized_layer(plan, scope)
        rec = groups.setdefault(canonical_json(payload), {
            "signature_id": short_digest(payload, "layer-"), "scope": scope, "layer_ids": [], "fields": payload,
        })
        rec["layer_ids"].append(int(plan.index))
    for rec in groups.values():
        rec["layer_ids"].sort()
        rec["occurrences"] = len(rec["layer_ids"])
    return sorted(groups.values(), key=lambda item: (item["scope"], item["layer_ids"]))


def assign_split(*, matrix_rows=None, batch=None, kv_length=None, routing_distribution=None,
                 zero_expert_fraction=None, fabric_bytes=None, real_weights=False, pr_overlay=False):
    """Freeze train/heldout from geometry only, before any measurement."""
    reasons = [reason for condition, reason in (
        (matrix_rows is not None and int(matrix_rows) in HELDOUT_MATRIX_ROWS, f"matrix_rows:{matrix_rows}"),
        (batch is not None and int(batch) == HELDOUT_BATCH, f"batch:{batch}"),
        (kv_length is not None and (int(kv_length) == HELDOUT_KV_LENGTH or int(kv_length) in HELDOUT_LONG_KV_LENGTHS), f"kv_length:{kv_length}"),
        (routing_distribution == HELDOUT_ROUTING_DISTRIBUTION and zero_expert_fraction is not None
         and math.isclose(float(zero_expert_fraction), HELDOUT_ROUTING_ZERO_FRACTION),
         f"routing:{HELDOUT_ROUTING_DISTRIBUTION}@{HELDOUT_ROUTING_ZERO_FRACTION}"),
        (fabric_bytes is not None and int(fabric_bytes) in HELDOUT_FABRIC_BYTES, f"fabric_bytes:{fabric_bytes}"),
        (real_weights, "real_weights:true"), (pr_overlay, "pr_overlay:true")) if condition]
    return ("heldout" if reasons else "train"), tuple(reasons)


def split_fields(*, matrix_rows=None, batch=None, kv_length=None, routing_distribution=None,
                 zero_expert_fraction=None, fabric_bytes=None, real_weights=False, pr_overlay=False):
    split, reasons = assign_split(
        matrix_rows=matrix_rows, batch=batch, kv_length=kv_length,
        routing_distribution=routing_distribution, zero_expert_fraction=zero_expert_fraction,
        fabric_bytes=fabric_bytes, real_weights=real_weights, pr_overlay=pr_overlay,
    )
    return {
        "split": split, "split_reasons": list(reasons),
        "split_axes": {
            "matrix_rows": matrix_rows, "batch": batch, "kv_length": kv_length,
            "routing_distribution": routing_distribution, "zero_expert_fraction": zero_expert_fraction,
            "fabric_bytes": fabric_bytes, "real_weights": bool(real_weights), "pr_overlay": bool(pr_overlay),
        },
    }


def positive_experts(experts: int, zero_fraction: float) -> int:
    return max(1, min(int(experts), int(round(int(experts) * (1.0 - float(zero_fraction))))))


def _uniform_counts(experts: int, assignments: int, zero_fraction: float) -> list[int]:
    positive = positive_experts(experts, zero_fraction)
    base, remainder = divmod(int(assignments), positive)
    return [base + (1 if index < remainder else 0) for index in range(positive)] + [0] * (int(experts) - positive)


def _zipf_counts(experts: int, assignments: int, zero_fraction: float, tokens: int) -> list[int] | None:
    positive = positive_experts(experts, zero_fraction)
    if assignments < positive:
        return None
    weights = [1.0 / math.pow(rank + 1, 1.2) for rank in range(positive)]
    quotas = [assignments * weight / sum(weights) for weight in weights]
    counts = [max(1, int(math.floor(quota))) for quota in quotas]
    while sum(counts) > assignments:
        donor = max(range(positive), key=lambda index: (counts[index], -index))
        if counts[donor] <= 1:
            return None
        counts[donor] -= 1
    fractional = sorted(range(positive), key=lambda index: (-(quotas[index] - math.floor(quotas[index])), index))
    for cursor in range(assignments - sum(counts)):
        counts[fractional[cursor % positive]] += 1
    rows = [min(value, tokens) for value in counts] + [0] * (experts - positive)
    excess = assignments - sum(rows)
    while excess > 0:
        for expert in sorted(range(positive), key=lambda index: (rows[index], index)):
            if excess <= 0:
                break
            if rows[expert] < tokens:
                rows[expert] += 1
                excess -= 1
    return rows


def routing_counts(distribution: str, experts: int, tokens: int, top_k: int, zero_fraction: float) -> list[int] | None:
    return (_uniform_counts(experts, tokens * top_k, zero_fraction) if distribution == "uniform"
            else _zipf_counts(experts, tokens * top_k, zero_fraction, tokens))


def topk_ids_for_counts(expert_rows: Sequence[int], tokens: int, top_k: int) -> list[list[int]]:
    remaining, result = [int(value) for value in expert_rows], []
    for _token in range(tokens):
        chosen = sorted(range(len(remaining)), key=lambda expert: (-remaining[expert], expert))[:top_k]
        if any(remaining[expert] <= 0 for expert in chosen):
            raise PlanError("routing_arm_requires_one_reserved_row_per_positive_expert")
        for expert in chosen:
            remaining[expert] -= 1
        result.append(chosen)
    if any(remaining):
        raise PlanError("routing_arm_histogram_not_conserved")
    return result


def explicit_ledger(
    expert_rows: Sequence[int], topk_ids: Sequence[Sequence[int]], ep: int,
    *, redundant: int = 0, slots: Any = None,
) -> dict[str, Any]:
    """Build exact per-rank integer routing ledger aligned with ``expert_slot_plan``."""
    experts = len(expert_rows)
    if ep <= 0 or experts <= 0:
        raise PlanError(f"explicit_ledger_invalid_geometry:{experts}:{ep}")
    if any(type(v) is not int or v < 0 for v in expert_rows):
        raise PlanError("explicit_ledger_expert_rows_require_nonnegative_integers")
    declared = [int(v) for v in expert_rows]
    observed = [0] * experts
    for ids in topk_ids:
        for raw_id in ids:
            if type(raw_id) is not int or not (0 <= raw_id < experts):
                raise PlanError(f"explicit_ledger_topk_expert_out_of_range:{raw_id}:experts={experts}")
            observed[raw_id] += 1
    if observed != declared:
        raise PlanError(f"explicit_ledger_topk_histogram_mismatch:declared={declared},observed={observed}")
    resolved = slots if slots is not None else slot_plan_for(experts, ep, redundant=redundant)
    if int(resolved.logical_experts) != experts or int(resolved.ranks) != ep:
        raise PlanError(f"explicit_ledger_slot_plan_mismatch:{resolved.logical_experts}:{resolved.ranks}")
    base = int(resolved.base_slots_by_rank[0])
    rank_rows = [[0] * len(resolved.expert_ids_by_rank[r]) for r in range(ep)]
    primary_slot = {
        expert: (expert // base, resolved.expert_ids_by_rank[expert // base].index(expert))
        for expert in range(experts)
    }
    for ids in topk_ids:
        for expert in ids:
            owner_rank, slot_index = primary_slot[int(expert)]
            rank_rows[owner_rank][slot_index] += 1
    local_rows = list(rank_rows[0])
    return {
        "basis": "explicit_token_routing_integer", "expert_rows": declared,
        "rank_expert_rows": rank_rows, "rank_assignment_rows": [sum(row) for row in rank_rows],
        "local_expert_rows": local_rows, "local_weight_expert_ids": list(resolved.expert_ids_by_rank[0]),
        "redundant_local_experts": int(resolved.redundant_slots_by_rank[0]),
        "padded_expert_slots": ep * base - experts,
        "topk_ids": [list(row) for row in topk_ids], "group_list": local_rows,
        "group_list_cumulative": list(accumulate(local_rows)), "group_list_type": 1, "m": sum(local_rows),
    }


def compact_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in ledger.items()
            if key not in {"topk_ids", "rank_expert_rows", "expert_rows"}}


def routing_block(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    realization = metadata.get("routing_realization")
    source = realization if isinstance(realization, Mapping) else {}
    group = list(source.get("local_expert_rows") or metadata.get("matrix_group_rows") or ())
    if not group:
        return None
    return {
        "basis": source.get("basis", "declared_integer_counts"), "top_k": metadata.get("top_k"),
        "num_experts": metadata.get("num_experts"), "ep": metadata.get("ep_degree"),
        "local_expert_rows": list(source.get("local_expert_rows") or ()),
        "rank_assignment_rows": list(source.get("rank_assignment_rows") or ()),
        "group_list": group, "expected_rows_per_expert_diagnostic": source.get("expected_rows_per_expert"),
        "group_list_cumulative": list(accumulate(group)), "m": sum(int(value) for value in group),
        "group_list_type": 1,
    }


def validate_routing_payload(payload: Mapping[str, Any]) -> None:
    routing = payload.get("routing")
    if not routing:
        return
    group = routing.get("group_list") or ()
    if not group or any(type(value) is not int or value < 0 for value in group):
        raise PlanError(f"routing_group_list_requires_nonnegative_integers:{payload['case_id']}")
    if sum(group) != int(routing.get("m", -1)):
        raise PlanError(f"routing_group_list_must_sum_to_m:{payload['case_id']}")
    if routing.get("group_list_cumulative") != list(accumulate(group)):
        raise PlanError(f"routing_cumulative_ends_must_match_counts:{payload['case_id']}")
    if int(routing.get("group_list_type", 0)) != 1:
        raise PlanError(f"routing_group_list_type_must_be_counts:{payload['case_id']}")
    for row in routing.get("rank_expert_rows") or ():
        if any(type(value) is not int or value < 0 for value in row):
            raise PlanError(f"routing_rank_rows_require_integers:{payload['case_id']}")
    if routing.get("rank_expert_rows") and sum(routing["rank_expert_rows"][0]) != sum(group):
        raise PlanError(f"routing_rank_zero_must_sum_to_m:{payload['case_id']}")
    expected = routing.get("expected_rows_per_expert_diagnostic")
    if expected is not None and float(expected) != int(expected) and str(routing.get("basis", "")).startswith("declared"):
        raise PlanError(f"fractional_expected_expert_count_in_replay:{payload['case_id']}")


def occurrences_for_layers(model_id: str, archetype: Mapping[str, Any], arm: str, layers: Any) -> list[dict[str, Any]]:
    return [{"model_id": model_id, "archetype_id": archetype["archetype_id"], "layer_idx": layer,
             "scope": "body" if layer >= 0 else "global", "arm": arm, "replica": 0, "count": count}
            for layer, count in sorted(Counter(layers).items())]


def merge_occurrences(existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in (*existing, *incoming):
        key = (item["model_id"], item["archetype_id"], item["layer_idx"],
               item.get("scope", "body"), item.get("arm", ""), item.get("replica", 0))
        record = table.setdefault(key, {**item, "count": 0})
        record["count"] += int(item.get("count", 1))
    return sorted(table.values(), key=lambda i: (
        i["model_id"], i["archetype_id"], i["layer_idx"], i.get("scope", "body"), i.get("arm", ""), i.get("replica", 0)))


def merge_case(cases: dict[str, dict[str, Any]], case: Mapping[str, Any]) -> str:
    case_id = str(case["case_id"])
    existing = cases.get(case_id)
    merged = merge_occurrences(existing["occurrences"] if existing else (), case["occurrences"])
    if existing is not None:
        if existing["payload"]["split"] != case["payload"]["split"]:
            raise PlanError(f"same_case_requires_conflicting_splits:{case_id}")
        existing["occurrences"], existing["occurrence_count"] = merged, sum(item["count"] for item in merged)
        return case_id
    cases[case_id] = {"case_id": case_id, "payload": case["payload"], "occurrences": merged,
                      "occurrence_count": sum(item["count"] for item in merged)}
    return case_id


def add_case(cases: dict[str, dict[str, Any]], key: Any, payload: Mapping[str, Any], occurrences: Any) -> str:
    case_id = short_digest(key, "case-")
    record = {**payload, "case_id": case_id}
    validate_routing_payload(record)
    return merge_case(cases, {"case_id": case_id, "payload": record, "occurrences": merge_occurrences((), list(occurrences))})


def service_identity(case: Mapping[str, Any]) -> dict[str, Any]:
    work = case.get("physical_work") or {}
    domains = [(res, unit, work[key]) for res, unit, key in (
        ("hbm", "bytes", "hbm_bytes"), ("cube", "flops", "flops"),
        ("vector", "ops", "vector_ops"), ("link", "bytes", "link_bytes"),
    ) if work.get(key)]
    if domains:
        return {"resource": domains[0][0], "work_unit": domains[0][1], "work": domains[0][2],
                "active_units": 1, "boundary": case["timing"]["boundary"], "mixed_domains": len(domains) > 1}
    return {"resource": "latency", "work_unit": "launches", "work": max(1, int(work.get("launches") or 0)),
            "active_units": 1, "boundary": case["timing"]["boundary"], "mixed_domains": False}


def case_identities(case: Mapping[str, Any], sku: str, hardware: Mapping[str, Any]) -> dict[str, Any]:
    node = case["nodes"][0]
    phase = str(case.get("phase") or node.get("phase") or "decode")
    mode = str(case.get("execution_mode") or ("aclgraph" if phase in {"decode", "draft"} else "eager"))
    execution = {"mode": mode, "stage": str(case["stage"])}
    if case["kind"] == "subgraph":
        level, shape = "subgraph", {"member_count": len(case["nodes"]), "matrix_rows": case.get("matrix_rows")}
        implementation = ",".join(sorted({i["implementation"] for i in case["nodes"]}))[:240]
        kernel = ",".join(sorted({i["kernel"] for i in case["nodes"]}))[:240]
        dtype, layout = str(node.get("dtype") or "fp32"), "contiguous"
    else:
        level, shape = "operator", deepcopy(node["shape"])
        dtype, layout = str(node.get("dtype") or "fp32"), str(node.get("layout") or "contiguous")
        implementation, kernel = node["implementation"], node["kernel"]
    boundary = str(case["timing"]["boundary"])
    primitive = {"hardware": {"sku": sku, "architecture": hardware["architecture"]},
                 "implementation": implementation, "kernel": kernel, "dtype": dtype, "shape": shape, "scope": boundary}
    return {
        "primitive_key": primitive,
        "accuracy_identity": {"level": level, "boundary": boundary, "hardware": sku, "phase": phase,
                              "implementation": implementation, "shape": shape, "dtype": dtype,
                              "layout": layout, "execution": execution},
        "service": service_identity(case),
    }


def reverse_index_cases(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for case in cases.values():
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in case["occurrences"]:
            key = (item["model_id"], int(item["archetype_id"]), item.get("scope", "body"), item.get("arm", ""), int(item.get("replica", 0)))
            record = groups.setdefault(key, {
                "model_id": item["model_id"], "scope": item.get("scope", "body"), "arm": item.get("arm", ""),
                "archetype_id": int(item["archetype_id"]), "replica": int(item.get("replica", 0)), "layers": [], "counts": [],
            })
            record["layers"].append(int(item["layer_idx"]))
            record["counts"].append(int(item.get("count", 1)))
        index[case["case_id"]] = sorted(groups.values(), key=lambda i: (i["model_id"], i["archetype_id"], i["scope"], i["arm"], i["replica"]))
    for record in index.values():
        for item in record:
            item["count"] = sum(item["counts"])
    return index


def compact_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = record.get("fields") or {}
    return {
        "signature_id": record["signature_id"], "scope": record["scope"],
        "layer_ids": list(record["layer_ids"]), "occurrences": record["occurrences"],
        **{k: fields[k] for k in ("attention", "window", "compress_ratio", "engram", "num_experts") if k in fields},
    }


def compact_model(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("model_id", "archetype", "parallel", "parameter_count", "layers", "attention", "ffn",
            "num_experts", "top_k", "quantization", "expert_bank", "layer_coverage", "compiled_arms", "proposal_tokens")
    return {**{key: record[key] for key in keys}, "layer_signatures": [compact_signature(item) for item in record["layer_signatures"]]}


def layer_coverage(model: Any, signatures: Sequence[Mapping[str, Any]], draft_layers: set[int], *, has_decode_arm: bool = True) -> dict[str, Any]:
    body_layers = {layer for record in signatures if record["scope"] == "body" for layer in record["layer_ids"]}
    body_complete = sorted(body_layers) == list(range(int(model.layers)))
    drafts, aux = {}, getattr(model, "auxiliary", None)
    active = ("mtp" if aux is not None and getattr(aux, "mtp", None) is not None
              else ("dspark" if aux is not None and getattr(aux, "dspark", None) is not None else None))
    if aux is not None and not aux.standalone_draft:
        for name in ("mtp", "dspark"):
            if (draft := getattr(aux, name, None)) is None:
                continue
            declared = int(draft.layers)
            covered = len(draft_layers) if (name == active and has_decode_arm) else 0
            entry: dict[str, Any] = {"declared_layers": declared, "covered_layers": covered, "complete": covered == declared}
            if not has_decode_arm:
                entry.update(declared_only=True, reason="prefill_only_geometry")
            elif name != active:
                entry.update(declared_only=True, reason="attached_draft_has_no_decode_proposal_flag")
            drafts[name] = entry
    return {"body": {"declared_layers": int(model.layers), "covered_layers": len(body_layers), "complete": body_complete},
            "drafts": drafts, "full_depth": body_complete and all(i["complete"] or i.get("declared_only") for i in drafts.values())}


def register_routing_case(cases: dict[str, dict[str, Any]], representative: Mapping[str, Any],
                          ledger: Mapping[str, Any], *, m: int, distribution: str,
                          zero_expert_fraction: float | None, kind: str, arm: str) -> None:
    base, axes = representative["payload"], representative["payload"]["split_axes"]
    split = split_fields(matrix_rows=m, batch=axes.get("batch"), kv_length=axes.get("kv_length"),
                         routing_distribution=distribution, zero_expert_fraction=zero_expert_fraction)
    nodes = deepcopy(base["nodes"])
    for node in nodes:
        if "m" in node["shape"]:
            node["shape"]["m"] = int(m)
        if "tokens" in node["shape"]:
            node["shape"]["tokens"] = int(m)
        if node["builder"] in {"grouped_matmul", "grouped_matmul_swiglu_quant"}:
            node["group_list"] = list(ledger["group_list"])
    stage = "s1_joint" if all(node["kind"] == "collective" for node in nodes) else "s0_local"
    drop = {"routing", "split", "split_reasons", "split_axes", "case_id", "sku", "identities",
            "source_case_id", "arm", "kind", "nodes", "edges", "timed_boundary", "stage"}
    payload = {
        **{k: v for k, v in base.items() if k not in drop},
        "kind": kind, "matrix_rows": int(m), "routing": compact_ledger(ledger),
        "source_case_id": representative["case_id"], "arm": arm, "nodes": nodes, "edges": [],
        "timed_boundary": {"include": [node["node_id"] for node in nodes]}, "stage": stage, **split,
    }
    add_case(cases, ("case", kind, base["phase"], base["model_id"], canonical_json({
        "m": m, "distribution": distribution, "zero": zero_expert_fraction, "source": representative["case_id"],
    })), payload, representative["occurrences"])


def add_routing_arms(structural: Mapping[str, Any]) -> None:
    representatives: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case in structural["cases"].values():
        payload, routing = case["payload"], case["payload"].get("routing") or {}
        if payload.get("kind") == "operator" and routing.get("local_expert_rows"):
            representatives.setdefault((payload["model_id"], str(payload.get("recipe") or payload.get("family"))), case)
    for representative in representatives.values():
        base = representative["payload"]
        record = structural["models"][base["model_id"]]
        par = base.get("parallel") or record["parallel"]
        ep, redundant = int(par["ep"]), int(par.get("redundant_experts_per_rank") or 0)
        experts = int(record["expert_bank"]["expert_count"])
        top_k = int(base["routing"].get("top_k") or record["top_k"])
        slots = slot_plan_for(experts, ep, redundant=redundant)
        local_experts, m_local = len(slots.expert_ids_by_rank[0]), int(base.get("matrix_rows") or 0)
        for distribution in ROUTING_DISTRIBUTIONS:
            for fraction in ZERO_EXPERT_FRACTIONS:
                positive = positive_experts(experts, fraction)
                tokens = max(m_local, int(math.ceil(positive / max(1, top_k))))
                rows = routing_counts(distribution, experts, tokens, top_k, fraction)
                if rows is None:
                    structural["diagnostics"]["routing_arms_skipped"].append({
                        "model_id": base["model_id"], "distribution": distribution,
                        "zero_expert_fraction": fraction, "reason": "fewer_assignments_than_positive_experts",
                    })
                    continue
                ledger = explicit_ledger(rows, topk_ids_for_counts(rows, tokens, top_k), ep, slots=slots)
                ledger.update({"distribution": distribution, "zero_expert_fraction": fraction,
                               "tokens": tokens, "top_k": top_k, "num_experts": experts, "ep": ep})
                register_routing_case(structural["cases"], representative, ledger, m=ledger["m"],
                                      distribution=distribution, zero_expert_fraction=fraction, kind="routing_arm", arm="routing_arm")
        target_m = min(HELDOUT_MATRIX_ROWS, key=lambda value: abs(value - HELDOUT_BATCH ** 2))
        base_rows, remainder = divmod(target_m, local_experts)
        rows = [base_rows + (1 if index < remainder else 0) for index in range(local_experts)]
        ledger = {
            "basis": "forced_balanced_local_integer_counts", "expert_rows": None,
            "rank_expert_rows": [rows], "rank_assignment_rows": [sum(rows)],
            "local_expert_rows": rows, "local_weight_expert_ids": list(slots.expert_ids_by_rank[0]),
            "group_list": rows, "group_list_cumulative": list(accumulate(rows)), "group_list_type": 1,
            "m": target_m, "distribution": "balanced", "zero_expert_fraction": None,
            "tokens": None, "top_k": top_k, "num_experts": experts, "ep": ep,
        }
        register_routing_case(structural["cases"], representative, ledger, m=target_m,
                              distribution="balanced", zero_expert_fraction=None, kind="matrix_ladder", arm="matrix_ladder")


__all__ = [
    "ARCHETYPES", "ARCHETYPE_BY_MODEL", "DEFAULT_TP", "DENSE_EP", "EXPERT_BYTES_PER_ELEMENT",
    "GEOMETRY_ARMS", "HELDOUT_BATCH", "HELDOUT_FABRIC_BYTES", "HELDOUT_GEOMETRY_PROBES",
    "HELDOUT_KV_LENGTH", "HELDOUT_LONG_KV_LENGTHS", "HELDOUT_MATRIX_ROWS",
    "HELDOUT_ROUTING_DISTRIBUTION", "HELDOUT_ROUTING_ZERO_FRACTION", "LOCAL_MOE_EP",
    "MAX_EXPORT_MODEL_LEN", "MIB", "PlanError", "ROUTED_EXPERT_MATRIX_FACTOR",
    "ROUTING_DISTRIBUTIONS", "SEQUENCE_LADDER_TOKENS", "TRILLION_MOE_EP",
    "TRILLION_PARAMETER_COUNT", "ZERO_EXPERT_FRACTIONS", "add_case", "add_routing_arms",
    "assign_split", "canonical_json", "case_identities", "collective_group_size", "compact_ledger",
    "compact_model", "compact_signature", "expert_bank_bytes", "explicit_ledger",
    "layer_coverage", "layer_signatures", "merge_case", "merge_occurrences",
    "model_archetype", "multi_axis_parallel_arms", "occurrences_for_layers",
    "parallel_policy", "parameter_count", "positive_experts", "probe_geometries_for",
    "register_routing_case", "representative_layer_depth", "reverse_index_cases", "routing_block",
    "routing_counts", "service_identity", "short_digest", "slot_plan_for", "split_fields",
    "topk_ids_for_counts", "validate_routing_payload",
]
