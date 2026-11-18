"""Physical mHC boundary fusion/overlap and attention/recurrent lowering passes.

[Role]
  Rewrite semantic operator DAGs for mHC sublayer boundary fusion and stream
  overlap, QKV RMSNorm+MRoPE split fusion, in-kernel KDA L2Norm, and GDN BA
  projection overlap.

[Invariants]
  * Every pass preserves total mathematical FLOPs and non-eliminated external
    HBM bytes; only genuinely fused intermediate HBM round-trips and launches
    disappear.
  * Secondary-stream placement (``mhc_aux``, ``gdn_proj_aux``) never deletes
    work; ``Engine`` arbitrates shared AIC/AIV/HBM contention dynamically.
  * Passes match structural operator families, recipes, and dependency edges,
    never model-name strings.

[Timeouts & Locks]
  Linear O(|V| + |E|) passes over one active layer graph; no I/O or locks.

[Side-Effect Policy]
  Pure functional transforms returning new immutable ``OpSpec`` sequences.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math

from mindie_sim.schema.common import dtype_bytes

from .implementations import (
    input_ports, rewire_replacements, set_input_ports,
)

MHC_DEFAULT_CUTOFF_ROWS = 72
NO_PORTS: dict = {}
COEFFICIENT_FUSION_RECIPE = "hc_coefficients_fused"
COEFFICIENT_FUSION_DTYPE = "fp32"
#: Declared native output ports of ``split_qkv_rmsnorm_mrope``.
QKV_PORTS = ("query", "key", "value", "gate")


def users_map(ops):
    users = defaultdict(list)
    for op in ops:
        for dep in op.deps:
            users[dep].append(op.op_id)
    return users


def merge_meta(*ops, extra=()):
    merged = {}
    origins = []
    for op in ops:
        meta = dict(op.metadata)
        merged.update(meta)
        for prev in meta.get("fused_origin_scopes", ()):
            if prev not in origins:
                origins.append(prev)
        origins.append({
            "op_id": op.op_id, "layer": op.layer, "stage": op.stage,
            "block": meta.get("structure_block"), "role": meta.get("structure_role"),
        })
    merged["fused_origin_scopes"] = tuple(origins)
    merged.update(dict(extra))
    return tuple(merged.items())


def mhc_boundary_fusion_overlap(ops, source_dtype, *, max_rows=0,
                                require_captured=False):
    """Fuse ``hc_post + hc_pre_mix + rmsnorm`` and overlap ``HcPreCoefficients``.

    Physical mechanism (vLLM-Ascend PR #22142 & #20824, ROCm vLLM #59802):
    1) Standard ``HyperConnection`` (``hc_post_mix -> hc_pre_mix -> rmsnorm``):
       * One boundary kernel produces updated multi-stream residual ``R_k`` and
         normalized sublayer input ``norm_k``, eliminating ``hc_pre_mix``'s
         ``[T, hc*H]`` HBM read and the ``[T, H]`` HBM write+read into ``norm``.
       * Coefficient chain ``hc_values_fp32 -> hc_project -> hc_inv_rms ->
         hc_affine -> hc_sinkhorn`` depends on ``R_k`` (not blocking Attention
         or MoE) and executes on ``stream="mhc_aux"`` when graph-captured,
         joining at the sublayer's ``hc_post_mix``.  When ``max_rows == 0`` the
         closed chain collapses into one resident ``hc_coefficients_fused``
         call, removing only its internal HBM round-trip and its extra launches.
    2) ``LowRankHyperConnection`` (``hc_lowrank_down_inject -> hc_silu -> hc_up``):
       * ``hc_gate_mix`` consumes gate columns from ``hc_lowrank_down_inject``
         directly so Attention/MoE starts immediately on ``main``, while
         ``hc_silu -> hc_up`` runs on ``stream="mhc_aux"`` and joins at
         ``hc_combine`` (``hc_post``). Consecutive ``hc_combine -> hc_norm``
         boundaries fuse their ``[T, hc*H]`` HBM round-trip when ``max_rows == 0``.
    """
    del source_dtype
    by_id = {op.op_id: op for op in ops}
    users = users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    stream_rewrites = {}

    for op in ops:
        meta = dict(op.metadata)
        captured = bool(meta.get("graph_captured", False))
        if require_captured and not captured:
            continue
        active_rows = int(meta.get("graph_active_query_rows")
                          or (op.local_shape[0] if op.local_shape else 0))
        if max_rows > 0 and active_rows > max_rows:
            continue

        # Branch A: HyperConnection sublayer boundary
        if op.recipe == "hc_pre_mix" and len(op.deps) == 2:
            ready_id, sinkhorn_id = op.deps[0], op.deps[1]
            sinkhorn = by_id.get(sinkhorn_id)
            affine = by_id.get(sinkhorn.deps[0]) if sinkhorn and sinkhorn.deps else None
            if sinkhorn is None or affine is None or len(affine.deps) < 2:
                continue
            proj = by_id.get(affine.deps[0])
            inv_rms = by_id.get(affine.deps[1])
            vals = by_id.get(proj.deps[0]) if proj and proj.deps else None
            if None in (proj, inv_rms, vals) or vals.deps != (ready_id,):
                continue

            # Decouple coefficient chain onto auxiliary stream when captured:
            # Attention/MoE input only needs collapsed pre_mix + norm; post/comb
            # coefficients are consumed at this sublayer's hc_post_mix.
            coeff_stream = "mhc_aux" if captured else (op.stream or "")
            for coeff_op in (vals, proj, inv_rms, affine, sinkhorn):
                stream_rewrites[coeff_op.op_id] = coeff_stream

            mix_users = users[op.op_id]
            norm_op = by_id.get(mix_users[0]) if len(mix_users) == 1 else None
            can_fuse_norm = (
                norm_op is not None
                and norm_op.family in {"rmsnorm", "rmsnorm_quant"}
                and norm_op.stage == op.stage
                and (norm_op.stream or "") == (op.stream or "")
            )
            prev_post = by_id.get(ready_id)
            # The Sinkhorn pre-gates are computed from the residual that the
            # preceding ``hc_post`` produces.  A single fused node cannot both
            # produce that residual and wait for coefficients derived from it,
            # so the post boundary stays separate and the fused norm keeps the
            # coefficient producer as a real dependency.
            if can_fuse_norm:
                # Fuse ``hc_pre_mix + norm``; the residual producer and the
                # coefficient producer are both required inputs.
                gamma_bytes = float(dict(norm_op.metadata).get("norm_gamma_runtime_bytes", 0.0) or 0.0)
                fused_read = float(op.mte2_bytes) + gamma_bytes
                fused_write = float(norm_op.mte3_bytes)
                fused_vec = float(op.vector_ops) + float(norm_op.vector_ops)
                # Only the post/comb Sinkhorn iterations overlap the sublayer;
                # the pre-gate producer (``affine``) must finish before the
                # fused norm reads its mixing gates.  When ``max_rows == 0``
                # collapses the closed chain into ``sinkhorn``, ``aliases``
                # rewires ``affine`` to that resident region automatically.
                fused_deps = tuple(dict.fromkeys((ready_id, affine.op_id)))
                fused = replace(
                    norm_op,
                    deps=fused_deps,
                    vector_ops=fused_vec,
                    mte2_bytes=fused_read,
                    mte3_bytes=fused_write,
                    hbm_bytes=fused_read + fused_write,
                    metadata=merge_meta(op, norm_op, extra=(
                        ("fused_from", ("hc_pre_mix", norm_op.family)),
                        ("mhc_coeff_stream", coeff_stream or "main"),
                        ("mhc_post_boundary_separate",
                         prev_post.recipe == "hc_post_mix"
                         if prev_post is not None else False),
                    )),
                )
                replacements[norm_op.op_id] = fused
                removed.add(op.op_id)
                aliases[op.op_id] = norm_op.op_id

            if max_rows == 0:
                coeff_ops = (vals, proj, inv_rms, affine, sinkhorn)
                vals_users = set(users[vals.op_id])
                if (vals_users <= {proj.op_id, inv_rms.op_id}
                        and {proj.op_id, inv_rms.op_id} <= vals_users
                        and users[proj.op_id] == [affine.op_id]
                        and users[inv_rms.op_id] == [affine.op_id]
                        and users[affine.op_id] == [sinkhorn.op_id]
                        and not any(item.plan_required or item.op_id in removed
                                    for item in coeff_ops)):
                    # One resident coefficient region: the only removable
                    # traffic is the internal 3V + 2P + 2R + 2A round-trip.
                    proj_weight_bytes = max(
                        0.0, float(proj.mte2_bytes) - float(vals.mte3_bytes))
                    affine_param_bytes = max(
                        0.0, float(affine.mte2_bytes) - float(proj.mte3_bytes)
                        - float(inv_rms.mte3_bytes))
                    coeff_read = (float(vals.mte2_bytes) + proj_weight_bytes
                                  + affine_param_bytes)
                    coeff_write = float(sinkhorn.mte3_bytes)
                    fused_coeff = replace(
                        sinkhorn,
                        deps=vals.deps,
                        stream=coeff_stream,
                        family=proj.family,
                        recipe=COEFFICIENT_FUSION_RECIPE,
                        dtype=COEFFICIENT_FUSION_DTYPE,
                        returns=sinkhorn.returns,
                        # The resident region keeps the projection's declared
                        # matrix geometry; only its internal round-trips vanish.
                        semantic_shape=proj.semantic_shape,
                        local_shape=proj.local_shape,
                        padded_shape=proj.padded_shape,
                        launch_shape=proj.launch_shape,
                        layout=proj.layout,
                        flops=sum(float(item.flops) for item in coeff_ops),
                        matrix_flops=proj.matrix_flops,
                        vector_ops=sum(float(item.vector_ops)
                                       for item in coeff_ops),
                        mte2_bytes=coeff_read,
                        mte3_bytes=coeff_write,
                        hbm_bytes=coeff_read + coeff_write,
                        metadata=merge_meta(*coeff_ops, extra=(
                            ("fused_from", (
                                "hc_values_fp32", "hc_projection",
                                "hc_inverse_rms", "hc_normalize_scale_bias_gate",
                                "hc_sinkhorn")),
                            ("mhc_coefficients_resident_region", True),
                            ("mhc_coeff_stream", coeff_stream or "main"),
                        )),
                    )
                    replacements[sinkhorn.op_id] = fused_coeff
                    for item in (vals, proj, inv_rms, affine):
                        removed.add(item.op_id)
                        aliases[item.op_id] = sinkhorn.op_id

        # Branch B: Qwen4Exp LowRankHyperConnection (PR #20824 + boundary norm fusion)
        elif op.recipe == "hc_gate_mix" and len(op.deps) == 2:
            norm_id, up_id = op.deps[0], op.deps[1]
            hc_norm = by_id.get(norm_id)
            hc_up = by_id.get(up_id)
            hc_silu = by_id.get(hc_up.deps[0]) if hc_up and hc_up.deps else None
            hc_down = by_id.get(hc_silu.deps[0]) if hc_silu and hc_silu.deps else None
            if None in (hc_norm, hc_up, hc_silu, hc_down):
                continue
            aux_stream = "mhc_aux" if captured else (op.stream or "")
            stream_rewrites[hc_silu.op_id] = aux_stream
            stream_rewrites[hc_up.op_id] = aux_stream
            # Preserve the declared ``hc_up -> hc_gate_mix`` (injection) and
            # ``hc_down -> hc_combine`` (gate-slice) data edges and byte reads
            # when placing the low-rank injection branch onto ``mhc_aux``.
            replacements[op.op_id] = replace(
                op,
                deps=(norm_id, hc_up.op_id),
                metadata=merge_meta(op, extra=(
                    ("mhc_lowrank_injection_stream", aux_stream or "main"),
                    ("mhc_lowrank_injection_producer", hc_up.op_id),
                )),
            )
            # In expected_ideal (max_rows == 0), fuse preceding hc_combine + grouped_gemma_rmsnorm
            prev_comb = by_id.get(hc_norm.deps[0]) if len(hc_norm.deps) == 1 else None
            if (max_rows == 0 and prev_comb is not None and prev_comb.recipe == "hc_combine"
                    and prev_comb.stage == hc_norm.stage and prev_comb.op_id not in removed):
                base_comb = replacements.get(prev_comb.op_id, prev_comb)
                gamma_b = max(0.0, float(hc_norm.mte2_bytes) - float(base_comb.mte3_bytes))
                f_read = float(base_comb.mte2_bytes) + gamma_b
                f_write = float(base_comb.mte3_bytes) + float(hc_norm.mte3_bytes)
                replacements[hc_norm.op_id] = replace(
                    hc_norm,
                    deps=base_comb.deps,
                    family="mhc",
                    recipe="hc_combine_rmsnorm",
                    vector_ops=float(base_comb.vector_ops) + float(hc_norm.vector_ops),
                    mte2_bytes=f_read,
                    mte3_bytes=f_write,
                    hbm_bytes=f_read + f_write,
                    metadata=merge_meta(base_comb, hc_norm, extra=(
                        ("fused_from", ("hc_combine", "grouped_gemma_rmsnorm")),
                    )),
                )
                replacements.pop(prev_comb.op_id, None)
                removed.add(prev_comb.op_id)
                aliases[prev_comb.op_id] = hc_norm.op_id

    if not replacements and not stream_rewrites:
        return list(ops)
    updated = [
        replace(op, stream=stream_rewrites[op.op_id])
        if op.op_id in stream_rewrites and op.op_id not in replacements else op
        for op in ops
    ]
    if not replacements:
        return updated
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        updated, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )


def _qkv_declared_view(members, key):
    """The group declaration's tensor view, including all four output shapes."""
    for member in members:
        for declaration in dict(member.metadata).get("structure_subgraphs", ()):
            if declaration.get("key") == key:
                return dict(declaration.get("tensor_view") or {})
    return {}


def _qkv_source_map(ops, member_ids):
    """Applied sources each external reader used before the group is folded."""
    result = {}
    for op in ops:
        used = []
        for entry in input_ports(op):
            source = entry.get("source_op") or entry.get("op")
            if source in member_ids:
                used.append(source)
        for dep in op.deps:
            if dep in member_ids:
                used.append(dep)
        if used:
            result[op.op_id] = tuple(dict.fromkeys(used))
    return result


def _qkv_port_for(op, source, member_ports):
    # Attention consumes the query plane; the cache store consumes key/value.
    # The fused anchor's member port is the key plane, so resolving members
    # first would mis-bind the query output to a key consumer edge.
    if op.family == "attention":
        return "query"
    if source in member_ports:
        return member_ports[source]
    return "key"


def _bind_qkv_output_ports(ops, anchor_id, member_ports, sources):
    """Name the fused producer's port every external reader actually consumes."""
    result = []
    for op in ops:
        used = sources.get(op.op_id)
        if not used or op.op_id == anchor_id:
            result.append(op)
            continue
        existing = [dict(entry) for entry in input_ports(op)
                    if entry.get("op") != anchor_id]
        mapped = []
        for source in used:
            entry = {"op": anchor_id,
                     "output": _qkv_port_for(op, source, member_ports),
                     "source_op": source}
            if entry not in mapped:
                mapped.append(entry)
        result.append(set_input_ports(op, tuple(existing + mapped)))
    return result


def qkv_norm_rope_fusion(ops, source_dtype):
    """Fuse ``split_qkv_rmsnorm_mrope`` (QK norm + RoPE + split copies) into 1 op."""
    del source_dtype
    by_id = {op.op_id: op for op in ops}
    groups = defaultdict(list)
    for op in ops:
        for key in dict(op.metadata).get("structure_subgraph_memberships", ()):
            if str(key).endswith(".norm_rope"):
                groups[str(key)].append(op)
    if not groups:
        return list(ops)
    replacements, removed, aliases, ports = {}, set(), {}, {}
    group_ports: dict[str, tuple[dict[str, str], dict[str, tuple[str, ...]]]] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        member_ids = {m.op_id for m in members}
        anchor = members[-1]
        view = _qkv_declared_view(members, key)
        shapes = {name: tuple(shape) for name, shape in
                  dict(view.get("output_shapes") or {}).items() if name in QKV_PORTS}
        if not shapes:
            continue
        dtype = str(view.get("dtype") or anchor.dtype)
        ext_deps = tuple(dict.fromkeys(
            dep for m in members for dep in m.deps if dep not in member_ids
        ))
        qkv_id = ext_deps[0] if ext_deps else ""
        qkv_op = by_id.get(qkv_id)
        input_read = (float(qkv_op.mte3_bytes) if qkv_op and qkv_op.mte3_bytes > 0
                      else max(float(m.mte2_bytes) for m in members))
        gamma_read = sum(float(dict(m.metadata).get("norm_gamma_runtime_bytes", 0.0) or 0.0)
                         for m in members)
        total_read = input_read + gamma_read
        # All four declared outputs are written, including the normalized,
        # unrotated tails of a partial-RoPE Q/K projection.
        total_write = sum(math.prod(shape) * dtype_bytes(dtype)
                          for shape in shapes.values())
        total_vec = sum(float(m.vector_ops) for m in members)
        primary = shapes.get("query") or next(iter(shapes.values()))
        extra = [("fusion_group_key", key), ("primary_output", "query"),
                 ("output_shape", primary),
                 ("qkv_output_shapes", tuple(shapes.items()))]
        extra.extend((f"{name}_shape", shape) for name, shape in shapes.items())
        extra.extend((f"{name}_dtype", dtype) for name in shapes)
        replacements[anchor.op_id] = replace(
            anchor,
            deps=ext_deps,
            family="rotary",
            recipe="split_qkv_rmsnorm_mrope_fused",
            semantic_shape=primary,
            local_shape=primary,
            padded_shape=primary,
            launch_shape=primary,
            vector_ops=total_vec,
            mte2_bytes=total_read,
            mte3_bytes=total_write,
            hbm_bytes=total_read + total_write,
            returns=tuple(name for name in shapes if name != "query"),
            metadata=merge_meta(*members, extra=extra),
        )
        member_ports = {anchor.op_id: "key"}
        for m in members:
            source_output = str(dict(m.metadata).get("source_output") or "")
            if source_output in QKV_PORTS and m.op_id != anchor.op_id:
                member_ports[m.op_id] = source_output
            if m.op_id != anchor.op_id:
                removed.add(m.op_id)
                aliases[m.op_id] = anchor.op_id
        ports[anchor.op_id] = {"output": "key"}
        for old, name in member_ports.items():
            if old != anchor.op_id:
                ports[old] = {"output": name}
        group_ports[anchor.op_id] = (
            member_ports, _qkv_source_map(ops, member_ids))
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    lowered = rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=ports,
    )
    for anchor_id, replacement in replacements.items():
        if replacement.recipe != "split_qkv_rmsnorm_mrope_fused":
            continue
        member_ports, sources = group_ports[anchor_id]
        lowered = _bind_qkv_output_ports(
            lowered, anchor_id, member_ports, sources)
    return lowered


def kda_in_kernel_l2norm(ops, source_dtype):
    """Fold standalone ``qk_l2norm`` into KDA prefill ``chunk_products`` (PR #22048)."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for op in ops:
        if op.recipe != "qk_l2norm":
            continue
        consumers = [by_id[u] for u in users.get(op.op_id, ()) if u in by_id]
        recurrent_families = {"recurrent_local", "recurrent_scan", "recurrent_output"}
        if not consumers or not all(c.family in recurrent_families for c in consumers):
            continue
        products = [c for c in consumers
                    if str(c.recipe).startswith("chunk_products")]
        if products and dict(op.metadata).get("batch_chunk_norm"):
            product_indices = {
                str(product.recipe)[len("chunk_products"):] for product in products
            }
            decode_scans = [
                c for c in consumers
                if str(c.recipe).startswith("recurrent_scan")
                and "_seg" not in str(c.recipe)
                and str(c.recipe)[len("recurrent_scan"):] not in product_indices
            ]
            targets = [*products, *decode_scans]
            rows = sum(target.local_shape[0] for target in targets)
            if rows:
                for target in targets:
                    share = op.vector_ops * target.local_shape[0] / rows
                    replacements[target.op_id] = replace(
                        target, vector_ops=target.vector_ops + share,
                        metadata=merge_meta(op, target, extra=(("in_kernel_qk_l2norm", True),)))
                removed.add(op.op_id)
                aliases[op.op_id] = op.deps[0]
            continue
        if len(products) != 1:
            continue
        first_prod = products[0]
        seq_suffix = str(first_prod.recipe)[len("chunk_products"):]
        allowed_exact = {
            f"chunk_products{seq_suffix}",
            f"recurrent_scan{seq_suffix}",
            f"recurrent_output{seq_suffix}",
            f"pcp_remote_initial_state{seq_suffix}",
        }
        seg_prefix = f"recurrent_scan{seq_suffix}_seg"
        if any(
            (str(consumer.recipe) not in allowed_exact
             and not str(consumer.recipe).startswith(seg_prefix))
            or consumer.phase != first_prod.phase
            or consumer.requests != first_prod.requests
            for consumer in consumers
        ):
            continue
        # Fold L2Norm vector ops into chunk_products0; Q and K are already read there!
        replacements[first_prod.op_id] = replace(
            first_prod,
            vector_ops=float(first_prod.vector_ops) + float(op.vector_ops),
            metadata=merge_meta(op, first_prod, extra=(("in_kernel_qk_l2norm", True),)),
        )
        removed.add(op.op_id)
        aliases[op.op_id] = op.deps[0] if op.deps else first_prod.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )


def gdn_projection_overlap(ops, source_dtype):
    """Place ``linear_beta_alpha`` on ``gdn_proj_aux`` stream during captured decode (PR #21037)."""
    del source_dtype
    result = []
    for op in ops:
        meta = dict(op.metadata)
        if (op.recipe == "linear_beta_alpha"
                and op.phase == "decode"
                and bool(meta.get("graph_captured", False))
                and not op.stream):
            result.append(replace(op, stream="gdn_proj_aux"))
        else:
            result.append(op)
    return result
