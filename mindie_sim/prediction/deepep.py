"""DeepEP two-stage deduplicated communication and shared-expert overlap.

[Role] Price DeepEP's inter-node group exchange plus intra-node scatter against
  a flat AllToAll ledger, expose the EPLB load ratio, and run the causal engine
  with the shared-expert stream enabled or disabled.
[Invariants] Node-level dispatch deduplicates a token to one remote delivery;
  intra-node traffic counts each of its top-k assignments. Link work is
  capacity times bytes, and startup latency is charged once per collective
  stage. Shared-expert overlap changes scheduling only: total FLOPs and HBM
  bytes are conserved. Unsupported topology or ratio declarations raise.
[Timeouts & Locks] Pure arithmetic; the causal-engine calls are bounded by the
  supplied case and start no device work, I/O, locks or remote execution.
[Side-Effect Policy] Returns immutable reports; never mutates the input case.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

from mindie_sim.schema.common import ModelingError, dtype_bytes


def _validate_eplb(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("eplb_peak_to_avg_invalid") from exc
    if not math.isfinite(result) or result < 1.0:
        raise ModelingError("eplb_peak_to_avg_must_be_at_least_one")
    return result


def _validate_overlap(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_overlap_ratio_invalid") from exc
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ModelingError("deepep_overlap_ratio_must_be_in_0_1")
    return result


def deepep_group_volume(tokens: float, hidden_size: int, top_k: int, *,
                        ep: int, ranks_per_node: int, dtype_bytes_val: float,
                        eplb_peak_to_avg: float = 1.0) -> float:
    """Inter-node group volume with one delivery per token and remote node."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    nodes = max(1, math.ceil(ep / max(1, ranks_per_node)))
    if nodes <= 1 or ep <= 1 or tokens <= 0:
        return 0.0
    node_hit_prob = min(float(top_k) / float(nodes), 1.0)
    remote_nodes = float(nodes - 1)
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * node_hit_prob * remote_nodes * ratio)


def deepep_intra_volume(tokens: float, hidden_size: int, top_k: int, *,
                        ep: int, ranks_per_node: int, dtype_bytes_val: float,
                        eplb_peak_to_avg: float = 1.0) -> float:
    """Intra-node scatter volume: every top-k assignment stays local."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    intra_ep = min(max(1, int(ep)), max(1, int(ranks_per_node)))
    if intra_ep <= 1 or tokens <= 0:
        return 0.0
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * float(top_k) * (float(intra_ep - 1) / float(intra_ep)) * ratio)


def standard_alltoall_volume(tokens: float, hidden_size: int, top_k: int, *,
                             ep: int, dtype_bytes_val: float,
                             eplb_peak_to_avg: float = 1.0) -> float:
    """Flat AllToAll assignment volume with a uniform remote fraction."""
    ratio = _validate_eplb(eplb_peak_to_avg)
    if ep <= 1 or tokens <= 0:
        return 0.0
    return (float(tokens) * float(hidden_size) * float(dtype_bytes_val)
            * float(top_k) * (float(ep - 1) / float(ep)) * ratio)


def resolve_exchange_options(runtime, transport, *, layer_scope: str = "body"):
    """Resolve EPLB/DeepEP options independent of a payload selector."""
    kernels = getattr(runtime, "kernels", None) or {}
    cfg = kernels.get("moe_transport") or {}
    if not isinstance(cfg, Mapping):
        raise ModelingError("moe_transport_requires_mapping")
    scopes = cfg.get("layer_scopes")
    if scopes is not None:
        if (not isinstance(scopes, (list, tuple)) or not scopes
                or any(scope not in {"body", "draft"} for scope in scopes)):
            raise ModelingError("moe_transport_layer_scopes_require_declared_model_scopes")
        if layer_scope not in scopes:
            return {"eplb": 1.0, "deepep": False, "overlap": 0.0, "ranks_per_node": 0}
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    eplb = (transport.eplb_peak_to_avg if transport is not None else
            cfg.get("eplb_peak_to_avg", cfg.get("ep_ratio",
                    metadata.get("eplb_peak_to_avg", 1.0))))
    overlap = cfg.get("deepep_overlap_ratio",
                      metadata.get("deepep_overlap_ratio", 0.0))
    ranks = cfg.get("ranks_per_node",
                    metadata.get("deepep_ranks_per_node", 0))
    try:
        ranks = int(ranks or 0)
    except (TypeError, ValueError) as exc:
        raise ModelingError("deepep_ranks_per_node_invalid") from exc
    if ranks < 0:
        raise ModelingError("deepep_ranks_per_node_invalid")
    return {
        "eplb": _validate_eplb(eplb),
        "deepep": cfg.get("mode") == "deepep" or bool(metadata.get("enable_deepep", False)),
        "overlap": _validate_overlap(overlap),
        "ranks_per_node": ranks,
    }


def deepep_exchange_factor(*, options, ep, top_k, tokens, hidden, dtype_bytes_val):
    """DeepEP group+intra deduplication ratio over a flat AllToAll (bytes conserved)."""
    if not options["deepep"]:
        return 1.0
    ranks_per_node = options["ranks_per_node"] or ep
    if ep <= ranks_per_node:
        return 1.0
    eff_tokens = max(1.0, float(tokens))
    common = dict(ep=ep, ranks_per_node=ranks_per_node,
                  dtype_bytes_val=dtype_bytes_val, eplb_peak_to_avg=1.0)
    group = deepep_group_volume(eff_tokens, hidden, top_k, **common)
    intra = deepep_intra_volume(eff_tokens, hidden, top_k, **common)
    standard = standard_alltoall_volume(
        eff_tokens, hidden, top_k, ep=ep, dtype_bytes_val=dtype_bytes_val,
        eplb_peak_to_avg=1.0)
    return (group + intra) / max(standard, 1e-12)


def _deepep_peer_traffic_matrix(total_bytes: float, *, ep: int,
                                ranks_per_node: int, top_k: int) -> dict[str, Any]:
    rpn = max(1, min(int(ep), int(ranks_per_node)))
    nodes = max(1, math.ceil(ep / rpn))
    w_intra = float(top_k) * (float(rpn - 1) / float(rpn)) if rpn > 1 else 0.0
    w_group = min(float(top_k) / float(nodes), 1.0) * float(nodes - 1) if nodes > 1 else 0.0
    w_total = max(w_intra + w_group, 1e-12)
    intra_share = w_intra / w_total
    inter_share = 1.0 - intra_share if w_intra + w_group > 0 else 0.0
    intra_row = float(total_bytes) * intra_share
    inter_row = float(total_bytes) - intra_row
    return {
        "kind": "two_tier",
        "ranks_per_node": int(rpn),
        "intra_share": intra_share,
        "inter_share": inter_share,
        "intra_row_bytes": intra_row,
        "inter_row_bytes": inter_row,
        "direction": "forward",
    }


_UNSCALED_MATRIX_KEYS = frozenset({"ranks_per_node", "intra_share", "inter_share"})


def _scale_matrix(value, factor):
    if isinstance(value, Mapping):
        return {
            key: (item if key in _UNSCALED_MATRIX_KEYS else _scale_matrix(item, factor))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return type(value)(_scale_matrix(item, factor) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * factor
    return value


def scale_exchange_operator(op, factor, options, *, ep: int = 1,
                            top_k: int = 1, is_combine: bool = False):
    """Scale token payload traffic and retain EPLB/DeepEP provenance."""
    if factor == 1.0 and not options["deepep"] and options["eplb"] == 1.0:
        return op
    scaled_bytes = op.link_bytes * factor
    metadata = dict(op.metadata)
    if "peer_traffic" in metadata:
        metadata["peer_traffic"] = _scale_matrix(metadata["peer_traffic"], factor)
    elif (options["deepep"] and not is_combine
          and options["ranks_per_node"] > 0 and ep > options["ranks_per_node"]
          and ep % options["ranks_per_node"] == 0):
        metadata["peer_traffic"] = _deepep_peer_traffic_matrix(
            scaled_bytes, ep=ep, ranks_per_node=options["ranks_per_node"], top_k=top_k)
    metadata["eplb_peak_to_avg"] = options["eplb"]
    if options["deepep"]:
        metadata["deepep_mode"] = True
        metadata["deepep_overlap_ratio"] = options["overlap"]
        metadata["deepep_dedup_scale"] = factor / max(options["eplb"], 1e-12)
    return replace(op, link_bytes=scaled_bytes,
                   metadata=tuple(metadata.items()))


def _scale_op_portion(op, ratio: float, *, suffix: str = "", stream: str | None = None,
                      deps: tuple[str, ...] | None = None):
    meta = dict(op.metadata)
    if "peer_traffic" in meta:
        meta["peer_traffic"] = _scale_matrix(meta["peer_traffic"], ratio)
    if suffix:
        if "collective_id" in meta:
            meta["collective_id"] = str(meta["collective_id"]) + suffix
        if "reverse_of_collective" in meta:
            meta["reverse_of_collective"] = str(meta["reverse_of_collective"]) + suffix
        meta["deepep_overlap_branch"] = True
    return replace(
        op,
        op_id=op.op_id + suffix,
        recipe=(op.recipe + suffix) if suffix else op.recipe,
        deps=op.deps if deps is None else deps,
        stream=op.stream if stream is None else stream,
        link_bytes=op.link_bytes * ratio,
        metadata=tuple(meta.items()),
    )


def split_deepep_dispatch_overlap(op, options):
    """Split dispatch into critical-path and concurrent-stream ops conserving bytes."""
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0:
        return (op,), None
    main_op = _scale_op_portion(op, 1.0 - overlap)
    over_op = _scale_op_portion(
        op, overlap, suffix="_overlap", stream="deepep_comm_stream",
        deps=(main_op.op_id,),
    )
    return (main_op, over_op), over_op.op_id


def split_deepep_combine_overlap(op, options, *, overlap_deps: tuple[str, ...],
                                 main_deps: tuple[str, ...]):
    """Split combine into concurrent-stream and critical-path ops conserving bytes."""
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0:
        return (op,), op.op_id
    over_op = _scale_op_portion(
        op, overlap, suffix="_overlap", stream="deepep_comm_stream",
        deps=overlap_deps or op.deps,
    )
    main_op = _scale_op_portion(
        op, 1.0 - overlap, deps=tuple(dict.fromkeys((*main_deps, over_op.op_id))),
    )
    return (over_op, main_op), main_op.op_id


def split_deepep_up_overlap(up, options, *, dispatch_overlap_id: str | None,
                            allow_split: bool = True):
    """Split expert_up by token fraction so remaining tokens wait for dispatch_overlap."""
    overlap = float(options.get("overlap", 0.0))
    orig_meta = dict(up.metadata)
    if (not options.get("deepep") or overlap <= 0.0 or dispatch_overlap_id is None
            or len(up.local_shape) < 3):
        return (up,), up.op_id
    m, n, k = up.local_shape[-3:]
    if (not allow_split or m < 2 or "matrix_group_rows" in orig_meta
            or "moe_transport_kind" in orig_meta):
        gated = replace(up, deps=tuple(dict.fromkeys((*up.deps, dispatch_overlap_id))))
        return (gated,), gated.op_id
    m1 = max(1, min(m - 1, round(m * (1.0 - overlap))))
    m2 = max(1, m - m1)
    shapes = ((m1, n, k), (m2, n, k))
    fracs = (1.0 - overlap, overlap)
    orig_expected = float(orig_meta.get("expected_rows", m))
    parts = []
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        meta = dict(up.metadata)
        meta["expected_rows"] = orig_expected * frac
        if "activation_scale_bytes" in meta:
            meta["activation_scale_bytes"] = float(meta["activation_scale_bytes"]) * frac
        if "activation_scale_shape" in meta and meta["activation_scale_shape"]:
            meta["activation_scale_shape"] = (shape[0], *tuple(meta["activation_scale_shape"])[1:])
        if idx == 1:
            meta["deepep_overlap_branch"] = True
        for key in ("matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes", "weight_bytes"):
            if key in meta and isinstance(meta[key], (int, float)):
                meta[key] = float(meta[key]) * frac
        if "matrix_epilogue" in meta:
            ep_m = dict(meta["matrix_epilogue"])
            ep_m["row_scale_bytes"] = float(ep_m.get("row_scale_bytes", 0.0)) * frac
            ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * frac
            ep_m["operations_per_output"] = (up.vector_ops * frac) / (shape[0] * shape[1])
            meta["matrix_epilogue"] = ep_m
        parts.append(replace(
            up,
            op_id=up.op_id if idx == 0 else up.op_id + "_overlap",
            recipe=up.recipe,
            deps=up.deps if idx == 0 else (up.op_id, dispatch_overlap_id),
            local_shape=shape,
            semantic_shape=shape if len(up.semantic_shape) == 3 else up.semantic_shape,
            launch_shape=shape if len(up.launch_shape) == 3 else up.launch_shape,
            padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
            flops=up.flops * frac,
            matrix_flops=tuple((dt, val * frac) for dt, val in up.matrix_flops),
            hbm_bytes=up.hbm_bytes * frac,
            vector_ops=up.vector_ops * frac,
            mte2_bytes=up.mte2_bytes * frac,
            mte3_bytes=up.mte3_bytes * frac,
            metadata=tuple(meta.items()),
        ))
    return (parts[0], parts[1]), parts[1].op_id


def split_deepep_down_overlap(down, options):
    """Split expert_down along output columns N so combine_overlap sends complete outputs."""
    overlap = float(options.get("overlap", 0.0))
    if not options.get("deepep") or overlap <= 0.0 or len(down.local_shape) < 3:
        return (down,), down.op_id, down.op_id
    m, n, k = down.local_shape[-3:]
    if n < 2:
        return (down,), down.op_id, down.op_id
    raw_n1 = max(1, min(n - 1, round(n * overlap)))
    n1 = max(64, min(n - 64, (raw_n1 // 64) * 64)) if (n >= 128 and n % 64 == 0) else raw_n1
    n2 = max(1, n - n1)
    shapes = ((m, n1, k), (m, n2, k))
    fracs = (n1 / n, n2 / n)
    parts = []
    for idx, (shape, frac) in enumerate(zip(shapes, fracs)):
        meta = dict(down.metadata)
        if idx == 1:
            meta.pop("input_conversion_subgraphs", None)
            meta["activation_producer"] = "explicit_quantization"
            meta["deepep_overlap_branch"] = True
        for key in ("matrix_weight_bytes", "packed_weight_bytes", "weight_scale_bytes", "weight_bytes"):
            if key in meta and isinstance(meta[key], (int, float)):
                meta[key] = float(meta[key]) * frac
        if "matrix_epilogue" in meta:
            ep_m = dict(meta["matrix_epilogue"])
            ep_m["row_scale_bytes"] = float(ep_m.get("row_scale_bytes", 0.0)) * frac
            ep_m["column_scale_bytes"] = float(ep_m.get("column_scale_bytes", 0.0)) * frac
            meta["matrix_epilogue"] = ep_m
        parts.append(replace(
            down,
            op_id=down.op_id if idx == 0 else down.op_id + "_overlap",
            recipe=down.recipe,
            deps=down.deps if idx == 0 else (down.op_id,),
            local_shape=shape,
            semantic_shape=shape if len(down.semantic_shape) == 3 else down.semantic_shape,
            launch_shape=shape if len(down.launch_shape) == 3 else down.launch_shape,
            padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
            flops=down.flops * frac,
            matrix_flops=tuple((dt, val * frac) for dt, val in down.matrix_flops),
            hbm_bytes=down.hbm_bytes * frac,
            vector_ops=down.vector_ops * frac,
            mte2_bytes=down.mte2_bytes * frac,
            mte3_bytes=down.mte3_bytes * frac,
            metadata=tuple(meta.items()),
        ))
    return (parts[0], parts[1]), parts[0].op_id, parts[1].op_id


def _link_summary(link):
    return {
        "name": link.name,
        "channel": link.channel,
        "capacity_bytes_per_ms": float(link.capacity_bytes_per_ms),
        "startup_ms": float(link.propagation_us) / 1000.0,
    }


def _link_profile(machine, ranks_per_node):
    """Choose declared local and fabric links for the two DeepEP stages."""
    links = tuple(machine.links)
    if not links:
        raise ModelingError("deepep_requires_declared_links")
    intra, inter = [], []
    for link in links:
        label = f"{link.name} {link.channel}".lower()
        if any(marker in label for marker in
               ("inter", "rdma", "roce", "scaleout", "cross", "fabric")):
            inter.append(link)
        elif any(marker in label for marker in
                 ("intra", "d2d", "ub", "hccs", "sio", "package")):
            intra.append(link)
        elif ranks_per_node > 1 and len(set(link.members)) <= ranks_per_node:
            intra.append(link)
        else:
            inter.append(link)
    if not intra:
        intra = inter[:1] or links[:1]
    if not inter:
        inter = intra[:1] or links[:1]
    local = max(intra, key=lambda link: link.capacity_bytes_per_ms)
    fabric = min(inter, key=lambda link: link.capacity_bytes_per_ms)
    return {"intra": _link_summary(local), "inter": _link_summary(fabric)}


def _transfer_ms(byte_count, link):
    if byte_count <= 0:
        return 0.0
    return byte_count / float(link["capacity_bytes_per_ms"]) + float(link["startup_ms"])


def _resolve_dispatch_dtype(case, dispatch_dtype):
    if dispatch_dtype is not None:
        return str(dispatch_dtype).strip().lower()
    cfg = dict(case.runtime.kernels.get("moe_transport") or {})
    return str(cfg.get("dispatch_dtype", cfg.get("payload_dtype", "bf16"))).strip().lower()


def analyze_deepep_collectives(
    case, *, phase: str = "decode", tokens: int = 1, sequences: int = 32,
    deepep_overlap_ratio: float = 0.0, eplb_peak_to_avg: float = 1.0,
    dispatch_dtype: str | None = None, combine_dtype: str = "bf16",
) -> dict[str, Any]:
    """Compare flat AllToAll and two-stage DeepEP dispatch/combine work."""
    from mindie_sim.prediction.structure import parse_case

    parsed = parse_case(case)
    eplb = _validate_eplb(eplb_peak_to_avg)
    overlap = _validate_overlap(deepep_overlap_ratio)
    ep = max(1, int(parsed.runtime.parallel.ep))
    ranks_per_node = max(1, int(parsed.machine.ranks_per_node))
    nodes = max(1, math.ceil(ep / ranks_per_node))
    total_tokens = max(0.0, float(tokens) * float(sequences))
    dispatch_name = _resolve_dispatch_dtype(parsed, dispatch_dtype)
    combine_name = str(combine_dtype).strip().lower()
    dispatch_bytes = dtype_bytes(dispatch_name)
    combine_bytes = dtype_bytes(combine_name)
    profile = _link_profile(parsed.machine, ranks_per_node)
    layers = [layer for layer in parsed.model.layer_plans if layer.ffn == "moe"]
    standard_link = profile["inter"] if nodes > 1 else profile["intra"]
    intra_link, group_link = profile["intra"], profile["inter"]
    hidden = parsed.model.hidden_size

    per_layer = []
    for layer in layers:
        top_k = int(layer.top_k)
        dispatch_flat = standard_alltoall_volume(
            total_tokens, hidden, top_k, ep=ep, dtype_bytes_val=dispatch_bytes,
            eplb_peak_to_avg=eplb)
        combine_flat = standard_alltoall_volume(
            total_tokens, hidden, top_k, ep=ep, dtype_bytes_val=combine_bytes,
            eplb_peak_to_avg=eplb)
        dispatch_group = deepep_group_volume(
            total_tokens, hidden, top_k, ep=ep, ranks_per_node=ranks_per_node,
            dtype_bytes_val=dispatch_bytes, eplb_peak_to_avg=eplb)
        dispatch_intra = deepep_intra_volume(
            total_tokens, hidden, top_k, ep=ep, ranks_per_node=ranks_per_node,
            dtype_bytes_val=dispatch_bytes, eplb_peak_to_avg=eplb)
        combine_intra = deepep_intra_volume(
            total_tokens, hidden, top_k, ep=ep, ranks_per_node=ranks_per_node,
            dtype_bytes_val=combine_bytes, eplb_peak_to_avg=eplb)
        combine_group = deepep_group_volume(
            total_tokens, hidden, top_k, ep=ep, ranks_per_node=ranks_per_node,
            dtype_bytes_val=combine_bytes, eplb_peak_to_avg=eplb)
        row = {
            "layer": int(layer.index),
            "top_k": top_k,
            "standard_dispatch_bytes": dispatch_flat,
            "standard_combine_bytes": combine_flat,
            "deepep_dispatch_group_bytes": dispatch_group,
            "deepep_dispatch_intra_bytes": dispatch_intra,
            "deepep_combine_intra_bytes": combine_intra,
            "deepep_combine_group_bytes": combine_group,
            "standard_dispatch_ms": _transfer_ms(dispatch_flat, standard_link),
            "standard_combine_ms": _transfer_ms(combine_flat, standard_link),
            "deepep_dispatch_group_ms": _transfer_ms(dispatch_group, group_link),
            "deepep_dispatch_intra_ms": _transfer_ms(dispatch_intra, intra_link),
            "deepep_combine_intra_ms": _transfer_ms(combine_intra, intra_link),
            "deepep_combine_group_ms": _transfer_ms(combine_group, group_link),
        }
        per_layer.append(row)

    def total(key):
        return math.fsum(row[key] for row in per_layer)

    standard_dispatch_ms = total("standard_dispatch_ms")
    standard_combine_ms = total("standard_combine_ms")
    standard_total = standard_dispatch_ms + standard_combine_ms
    group_ms = total("deepep_dispatch_group_ms")
    dispatch_intra_ms = total("deepep_dispatch_intra_ms")
    combine_intra_ms = total("deepep_combine_intra_ms")
    combine_group_ms = total("deepep_combine_group_ms")
    raw_deepep = group_ms + dispatch_intra_ms + combine_intra_ms + combine_group_ms
    deepep_total = raw_deepep * (1.0 - overlap)
    saving = (1.0 - deepep_total / max(standard_total, 1e-12)
              if standard_total > 0 else 0.0)
    return {
        "schema": "mindie-sim.deepep-collectives/1",
        "phase": phase, "tokens": int(tokens), "sequences": int(sequences),
        "total_tokens": total_tokens, "ep": ep,
        "ranks_per_node": ranks_per_node, "nodes": nodes,
        "dispatch_dtype": dispatch_name, "combine_dtype": combine_name,
        "eplb_peak_to_avg": eplb, "deepep_overlap_ratio": overlap,
        "moe_layers": len(layers), "per_layer": per_layer,
        "links": profile,
        "standard_dispatch_bytes": total("standard_dispatch_bytes"),
        "standard_combine_bytes": total("standard_combine_bytes"),
        "deepep_dispatch_group_bytes": total("deepep_dispatch_group_bytes"),
        "deepep_dispatch_intra_bytes": total("deepep_dispatch_intra_bytes"),
        "deepep_combine_intra_bytes": total("deepep_combine_intra_bytes"),
        "deepep_combine_group_bytes": total("deepep_combine_group_bytes"),
        "standard_dispatch_ms": standard_dispatch_ms,
        "standard_combine_ms": standard_combine_ms,
        "standard_total_comm_ms": standard_total,
        "deepep_dispatch_group_ms": group_ms,
        "deepep_dispatch_intra_ms": dispatch_intra_ms,
        "deepep_combine_intra_ms": combine_intra_ms,
        "deepep_combine_group_ms": combine_group_ms,
        "deepep_unoverlapped_comm_ms": raw_deepep,
        "deepep_total_comm_ms": deepep_total,
        "comm_saving_ratio": saving,
    }


def analyze_shared_expert_overlap(
    case, *, phase: str = "decode", tokens: int = 1, context: int = 4096,
    sequences: int = 16, scope: str = "body",
) -> dict[str, Any]:
    """Evaluate the causal engine with and without the shared-expert stream."""
    from mindie_sim.prediction.efficiency import analyze_network_efficiency
    from mindie_sim.prediction.structure import parse_case

    parsed = parse_case(case)

    def evaluate(enabled):
        metadata = {**dict(parsed.runtime.metadata),
                    "overlap_shared_expert": bool(enabled)}
        runtime = replace(parsed.runtime, metadata=tuple(sorted(metadata.items())))
        report = analyze_network_efficiency(
            replace(parsed, runtime=runtime), phase=phase, tokens=tokens,
            context=context, sequences=sequences, scope=scope)
        return report

    baseline = evaluate(False)
    overlapped = evaluate(True)
    baseline_ms = float(baseline["graph_duration_ms"])
    overlapped_ms = float(overlapped["graph_duration_ms"])
    baseline_flops = float(baseline["total_flops"])
    overlapped_flops = float(overlapped["total_flops"])
    baseline_hbm = float(baseline["total_hbm_bytes"])
    overlapped_hbm = float(overlapped["total_hbm_bytes"])
    return {
        "schema": "mindie-sim.shared-expert-overlap/1",
        "phase": phase, "tokens": int(tokens), "context": int(context),
        "sequences": int(sequences), "scope": scope,
        "baseline_step_ms": baseline_ms,
        "overlapped_step_ms": overlapped_ms,
        "saved_ms": baseline_ms - overlapped_ms,
        "speedup_ratio": baseline_ms / max(overlapped_ms, 1e-12),
        "baseline_operator_sum_ms": float(baseline["operator_sum_ms"]),
        "overlapped_operator_sum_ms": float(overlapped["operator_sum_ms"]),
        "baseline_total_flops": baseline_flops,
        "overlapped_total_flops": overlapped_flops,
        "baseline_total_hbm_bytes": baseline_hbm,
        "overlapped_total_hbm_bytes": overlapped_hbm,
        "flops_conserved": math.isclose(
            baseline_flops, overlapped_flops, rel_tol=1e-12, abs_tol=0.0),
        "hbm_bytes_conserved": math.isclose(
            baseline_hbm, overlapped_hbm, rel_tol=1e-12, abs_tol=0.0),
    }


__all__ = [
    "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
    "resolve_exchange_options", "deepep_exchange_factor",
    "scale_exchange_operator", "analyze_deepep_collectives",
    "analyze_shared_expert_overlap",
]
