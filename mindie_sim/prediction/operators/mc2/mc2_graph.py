"""Select the ordinary MoE graph's native communication protocol.

[Role] Resolve explicit or runtime-selected communication for the EP group.
[Invariants] A whole EP group selects one protocol. The MC2 operator package
  owns its validation, recipe registration, and graph transformation.
[Timeouts & Locks] Finite in-memory dispatch, with no locks or external work.
[Side-Effect Policy] No hardware or source reads.
"""
from .moe_comm_selection import FUSED_MC2, MC2, MoeCommConfig
from mindie_sim.prediction.operators import mc2

NETWORK = mc2.NATIVE_PROTOCOL_NETWORK
EXPLICIT_ROUTING_LOWERINGS = frozenset(
    {"route_pack", "route_unpack", "moe_init_routing"})


def select(kernels):
    return mc2.select_native_protocol(kernels)


def recipes(kernels):
    return mc2.native_protocol_recipes(kernels)


def select_common(runtime, features=None, *, tokens, top_k, experts, ep, phase=""):
    """Explicit native declaration first, otherwise the common selector.

    The ordinary path and the piecewise families consume the same
    ``MoeCommConfig`` projection of the actual deployment feature record.  No
    model name, measured trace, or local heuristic selects MC2.
    """
    kernels = getattr(runtime, "kernels", {}) or {}
    explicit = select(kernels)
    if explicit is not None:
        return explicit
    if EXPLICIT_ROUTING_LOWERINGS & set(kernels.get("lowerings") or {}):
        # An explicitly lowered routing/init kernel owns the expert input
        # representation; automatic MC2 selection must not replace it.
        return None
    rmeta = dict(getattr(runtime, "metadata", ()) or ())
    radditional = rmeta.get("additional_config")
    radditional = dict(radditional) if isinstance(radditional, dict) else {}
    has_runtime_moe_comm = bool(
        "enable_fused_mc2" in radditional
        or "enable_mega_moe" in radditional
        or rmeta.get("moe_mega_moe_supported")
        or rmeta.get("moe_comm_mode")
        or rmeta.get("image")
        or rmeta.get("image_id")
    )
    if (features is None
            or not has_runtime_moe_comm
            or kernels.get("moe_transport")
            or bool({"expert_dispatch", "expert_combine"} & set(kernels.get("recipes") or ()))
            or getattr(runtime, "moe_recipe", "alltoall") != "alltoall"):
        return None
    config = MoeCommConfig.from_features(
        features, ep=ep, top_k=top_k, experts=experts)
    if config is None:
        return None
    chosen = config.select(max(0, int(tokens)), phase=phase)
    if chosen not in {MC2, FUSED_MC2}:
        return None
    if (chosen == MC2 and phase == "prefill"
            and not bool(radditional.get("enable_prefill_mc2"))
            and not config.enable_prefill_mc2):
        return None
    fused_level = int(radditional.get("enable_fused_mc2", 0) or 0)
    arch_family = str(
        config.arch_family or rmeta.get("arch_family") or ""
    ).strip().upper()
    return mc2.common_native_protocol(
        chosen, fused_level, config.to_items(), arch_family=arch_family,
    )


def lower(ops, *, runtime, prefix, tokens, top_k, width, experts, ep,
          source_partitions, received_assignments, dtype, source_routing=None,
          valid_tokens=None, features=None, source_rows_by_replica=None,
          destination_rows_by_rank=None, phase="", tp_scale=1):
    # ``dynamic`` selects one method for the whole EP group.  ``MoeCommConfig``
    # capacity is TP-global (``per_rank * tp``), so SP-sharded per-rank rows
    # scale by ``tp_scale`` to compare in consistent DP-group token units.
    raw_selection_tokens = (max(int(value) for value in source_rows_by_replica)
                            if source_rows_by_replica else int(tokens))
    selection_tokens = raw_selection_tokens * max(1, int(tp_scale))
    selected = select_common(
        runtime, features, tokens=selection_tokens, top_k=top_k,
        experts=experts, ep=ep, phase=phase)
    if selected is None:
        return ops
    return mc2.lower_native_protocol(
        ops, selected=selected, runtime=runtime, prefix=prefix, tokens=tokens,
        top_k=top_k, width=width, experts=experts, ep=ep,
        source_partitions=source_partitions, received_assignments=received_assignments,
        dtype=dtype, source_routing=source_routing, valid_tokens=valid_tokens,
        source_rows_by_replica=source_rows_by_replica,
        destination_rows_by_rank=destination_rows_by_rank)
