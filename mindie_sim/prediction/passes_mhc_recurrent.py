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

from .implementations import rewire_replacements

MHC_DEFAULT_CUTOFF_ROWS = 72
NO_PORTS: dict = {}


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
         joining at the sublayer's ``hc_post_mix``.
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
            can_fuse_post = (
                prev_post is not None
                and prev_post.recipe == "hc_post_mix"
                and prev_post.stage == op.stage
                and prev_post.op_id not in removed
            )

            if can_fuse_post and can_fuse_norm:
                # Eliminate hc_mix read of [T, hc*H] residual AND [T, H] write+read into norm.
                # Retain coefficient scalar read (T * hc * 4) and norm gamma read.
                gamma_bytes = float(dict(norm_op.metadata).get("norm_gamma_runtime_bytes", 0.0) or 0.0)
                tokens = int(op.local_shape[0]) if op.local_shape else active_rows
                coeff_pre_bytes = max(0.0, float(op.mte2_bytes) - float(prev_post.mte3_bytes))
                fused_read = float(prev_post.mte2_bytes) + coeff_pre_bytes + gamma_bytes
                fused_write = float(prev_post.mte3_bytes) + float(norm_op.mte3_bytes)
                fused_vec = float(prev_post.vector_ops) + float(op.vector_ops) + float(norm_op.vector_ops)
                target_family = "add_rmsnorm_quant" if norm_op.family == "rmsnorm_quant" else "mhc"
                target_recipe = ("mhc_boundary_rmsnorm_quant"
                                 if norm_op.family == "rmsnorm_quant" else "mhc_boundary_rmsnorm")
                fused = replace(
                    norm_op,
                    deps=prev_post.deps,
                    family=target_family,
                    recipe=target_recipe,
                    vector_ops=fused_vec,
                    mte2_bytes=fused_read,
                    mte3_bytes=fused_write,
                    hbm_bytes=fused_read + fused_write,
                    returns=tuple(dict.fromkeys((*norm_op.returns, "residual"))),
                    metadata=merge_meta(prev_post, op, norm_op, extra=(
                        ("fused_from", ("hc_post_mix", "hc_pre_mix", norm_op.family)),
                        ("mhc_coeff_stream", coeff_stream or "main"),
                        ("mhc_active_rows", tokens),
                    )),
                )
                replacements[norm_op.op_id] = fused
                removed.update({prev_post.op_id, op.op_id})
                aliases[prev_post.op_id] = norm_op.op_id
                aliases[op.op_id] = norm_op.op_id
            elif can_fuse_norm:
                # First layer l0.attention (no preceding hc_post): fuse hc_pre_mix + norm,
                # removing sinkhorn dependency from blocking main-stream Attention!
                gamma_bytes = float(dict(norm_op.metadata).get("norm_gamma_runtime_bytes", 0.0) or 0.0)
                fused_read = float(op.mte2_bytes) + gamma_bytes
                fused_write = float(norm_op.mte3_bytes)
                fused_vec = float(op.vector_ops) + float(norm_op.vector_ops)
                fused = replace(
                    norm_op,
                    deps=(ready_id,),
                    vector_ops=fused_vec,
                    mte2_bytes=fused_read,
                    mte3_bytes=fused_write,
                    hbm_bytes=fused_read + fused_write,
                    metadata=merge_meta(op, norm_op, extra=(
                        ("fused_from", ("hc_pre_mix", norm_op.family)),
                        ("mhc_coeff_stream", coeff_stream or "main"),
                    )),
                )
                replacements[norm_op.op_id] = fused
                removed.add(op.op_id)
                aliases[op.op_id] = norm_op.op_id

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
            # Gate mix reads hc_norm + gate slice from hc_down directly; hc_up joins at hc_combine.
            # The exact parametric read is the declared mix read minus the injection
            # write moved to the combine: zero hardcoded column padding.
            gate_only_read = max(
                0.0, float(op.mte2_bytes) - float(hc_up.mte3_bytes),
            )
            replacements[op.op_id] = replace(
                op,
                deps=(norm_id, hc_down.op_id),
                mte2_bytes=min(float(op.mte2_bytes), gate_only_read),
                hbm_bytes=min(float(op.mte2_bytes), gate_only_read) + float(op.mte3_bytes),
                metadata=merge_meta(op, extra=(("mhc_lowrank_injection_stream", aux_stream or "main"),)),
            )
            # Rewire downstream hc_combine (hc_post) to wait for hc_up injection completion
            for combine_id in users.get(hc_down.op_id, ()):
                comb_op = by_id.get(combine_id)
                if comb_op is not None and comb_op.recipe == "hc_combine":
                    new_deps = tuple(hc_up.op_id if d == hc_down.op_id else d for d in comb_op.deps)
                    replacements[comb_op.op_id] = replace(comb_op, deps=new_deps)
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
    replacements, removed, aliases = {}, set(), {}
    users = users_map(ops)
    for key, members in groups.items():
        if len(members) < 2:
            continue
        member_ids = {m.op_id for m in members}
        anchor = members[-1]
        ext_deps = tuple(dict.fromkeys(
            dep for m in members for dep in m.deps if dep not in member_ids
        ))
        # Projected QKV is read once; all split outputs (Q_rope, K_rope, V, gate) written once.
        qkv_id = ext_deps[0] if ext_deps else ""
        qkv_op = by_id.get(qkv_id)
        input_read = (float(qkv_op.mte3_bytes) if qkv_op and qkv_op.mte3_bytes > 0
                      else max(float(m.mte2_bytes) for m in members))
        gamma_read = sum(float(dict(m.metadata).get("norm_gamma_runtime_bytes", 0.0) or 0.0)
                         for m in members)
        total_read = input_read + gamma_read
        # Keep only externally consumed member writes (eliminate q_norm -> rope intermediate)
        total_write = sum(
            float(m.mte3_bytes) for m in members
            if any(u not in member_ids for u in users.get(m.op_id, ())) or m is anchor
        )
        total_vec = sum(float(m.vector_ops) for m in members)
        replacements[anchor.op_id] = replace(
            anchor,
            deps=ext_deps,
            family="rotary",
            recipe="split_qkv_rmsnorm_mrope_fused",
            vector_ops=total_vec,
            mte2_bytes=total_read,
            mte3_bytes=total_write,
            hbm_bytes=total_read + total_write,
            metadata=merge_meta(*members, extra=(("fusion_group_key", key),)),
        )
        for m in members:
            if m.op_id != anchor.op_id:
                removed.add(m.op_id)
                aliases[m.op_id] = anchor.op_id
    if not replacements:
        return list(ops)
    full_aliases = {old: old for old in replacements}
    full_aliases.update(aliases)
    return rewire_replacements(
        ops, replacements=replacements, aliases=full_aliases,
        removed=removed, ports=NO_PORTS,
    )


def kda_in_kernel_l2norm(ops, source_dtype):
    """Fold standalone ``qk_l2norm`` into KDA prefill ``chunk_products`` (PR #22048)."""
    del source_dtype
    by_id, users = {op.op_id: op for op in ops}, users_map(ops)
    replacements, removed, aliases = {}, set(), {}
    for op in ops:
        if op.recipe != "qk_l2norm":
            continue
        consumers = [by_id[u] for u in users.get(op.op_id, ()) if u in by_id]
        if not consumers or not all(
            c.family in {"recurrent_local", "recurrent_scan", "recurrent_output"}
            for c in consumers
        ):
            continue
        first_prod = next((c for c in consumers if str(c.recipe).startswith("chunk_products")), None)
        if first_prod is None:
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
