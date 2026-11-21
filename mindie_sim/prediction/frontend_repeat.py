"""Homogeneous layer template lowering and reuse.

[Role] Lower representative layers and reuse compatible operator and metadata templates.
[Invariants] Each occurrence keeps its layer identity and causal predecessors.
[Timeouts & Locks] Retains the latest graph and templates. No locks or I/O.
[Side-Effect Policy] Updates frontend caches and seeds execution metadata keys.
"""
from __future__ import annotations

from . import frontend as _frontend
from .frontend_body import _build_representative_body, _representative_attention
from .frontend_iteration import _append_service_tail, _finalize_repeated_selections


_META_LIST_EDITS = 3
_META_LAYER_INDEX = 4
_META_DICT_VALUE_EDITS = 5
_META_DICT_KEY_AND_VALUE_EDITS = 6


class _FrontendRepeat:
    def _try_lower_homogeneous(
        self, *, entry, pre_ops, body_ready, plan, model, runtime,
        views, local_views, valid, padded, local_tokens, bucket, scope,
        mode, requests, selections, resource_fusion, assembly, replica,
        features, body_global_tokens, source_rows_by_replica,
        routing_global_tokens, routing_views, physical_rank,
        query_position, before, has_after_draft, pcp_prefill,
        dispatch_fast_step: bool = False, machine=None,
    ):
        layers = model.layer_plans
        if (len(layers) <= 2 or local_tokens <= 0 or scope != "none"
                or before or has_after_draft or pcp_prefill
                or runtime.parallel.pp != 1):
            return None
        fusions_cfg = runtime.kernels.get("fusions") or ()
        if "residual_norm" in fusions_cfg:
            return None
        rmeta = dict(runtime.metadata)
        if "sparse_mla_metadata_groups" in rmeta:
            return None
        p = runtime.parallel
        routing_tokens = (
            routing_global_tokens if routing_global_tokens is not None
            else body_global_tokens
        )
        valid_local = sum(v.logical_q_len for v in local_views)
        t_shards = 1
        has_explicit_real = "realizations" in runtime.routing
        env_views = (
            routing_views if has_explicit_real
            else tuple((v.q_len, v.logical_q_len, bool(getattr(v, "padding", False)))
                       for v in local_views)
        )
        static_env_key = (
            mode, local_tokens, valid_local, body_global_tokens,
            routing_tokens, replica, physical_rank,
            query_position if has_explicit_real else None,
            env_views,
        )
        prev_env = self._prev_homo_env
        if (prev_env is not None and prev_env[0] is model
                and (prev_env[1] is runtime or prev_env[1] == runtime)
                and prev_env[2] == static_env_key):
            if not prev_env[3]:
                return None
            arch_for_layer, unique_archs, realizations = prev_env[4]
        else:
            for lyr in layers:
                if (lyr.engram or lyr.hyper_connections or lyr.residual_block_size
                        or lyr.kv_owner_layer_id not in (-1, lyr.index)
                        or lyr.index_k_owner_layer_id not in (-1, lyr.index)
                        or lyr.topk_owner_layer_id not in (-1, lyr.index)
                        or lyr.candidate_source_layer_id not in (-1, lyr.index)
                        or lyr.index_source_layer_id not in (-1, lyr.index)
                        or lyr.native_layer_id not in (-1, lyr.index)):
                    self._prev_homo_env = (model, runtime, static_env_key, False, None)
                    return None
            if _frontend.sequence_parallel.select(
                    runtime, model, local_views, tokens=local_tokens,
                    global_tokens=body_global_tokens) is not None:
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            if (mode not in {"draft", "draft_prefill"}
                    and _frontend.family_attnres.selection(runtime, model=model, phase=mode) is not None):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            stages = _frontend.stage_map(model, runtime)
            if any(s != 0 for s in stages):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None

            arch_for_layer_list = []
            unique_archs = {}
            realizations_list = []
            for idx, lyr in enumerate(layers):
                real = (
                    _frontend.select_realization(
                        runtime, lyr, phase=mode, tokens=local_tokens,
                        valid_tokens=valid_local, global_tokens=routing_tokens,
                        token_shards=t_shards, valid_source_tokens=None,
                        replica=replica, views=routing_views,
                        physical_rank=physical_rank, query_position=query_position,
                    )
                    if lyr.ffn == "moe" else None
                )
                realizations_list.append(real)
                norm_real = (
                    _frontend.replace(real, layer=0)
                    if isinstance(real, _frontend.RoutingRealization) else real
                )
                norm_lyr = _frontend.replace(
                    lyr,
                    index=0,
                    kv_owner_layer_id=(-1 if lyr.kv_owner_layer_id == -1 else 0),
                    index_k_owner_layer_id=(-1 if lyr.index_k_owner_layer_id == -1 else 0),
                    topk_owner_layer_id=(-1 if lyr.topk_owner_layer_id == -1 else 0),
                    candidate_source_layer_id=(-1 if lyr.candidate_source_layer_id == -1 else 0),
                    index_source_layer_id=(-1 if lyr.index_source_layer_id == -1 else 0),
                    native_layer_id=(-1 if lyr.native_layer_id == -1 else 0),
                )
                akey = (norm_lyr, norm_real)
                arch_for_layer_list.append(akey)
                if akey not in unique_archs:
                    unique_archs[akey] = idx

            if len(unique_archs) >= len(layers):
                self._prev_homo_env = (model, runtime, static_env_key, False, None)
                return None
            arch_for_layer = tuple(arch_for_layer_list)
            realizations = tuple(realizations_list)
            self._prev_homo_env = (
                model, runtime, static_env_key, True,
                (arch_for_layer, unique_archs, realizations),
            )

        from .execution import meta_scan_flags, op_meta_key_from_metadata, seed_meta_id_key
        metadata_key_helpers = (op_meta_key_from_metadata, seed_meta_id_key)
        decode_sig = (
            (
                tuple((v.request_id, v.q_len, v.logical_q_len, v.samples,
                       v.draft_proposal_tokens, bool(getattr(v, "padding", False)))
                      for v in views),
                tuple((v.request_id, v.q_len, v.logical_q_len,
                       bool(getattr(v, "padding", False)))
                      for v in local_views),
                resource_fusion,
                assembly,
                selections,
            )
            if (mode == "decode" and local_views
                and all(v.phase == "decode" and v.context > 0 for v in local_views))
            else None
        )
        prev_state = self._prev_arch_state
        can_reuse_prev = (
            prev_state is not None
            and prev_state[0] == arch_for_layer
            and prev_state[1] == body_ready
        )
        reused_archs = {}
        if (dispatch_fast_step and can_reuse_prev and decode_sig is not None
                and prev_state[7] == decode_sig):
            prev_tmpls_fast = prev_state[2]
            for akey in unique_archs:
                if akey[0].attention in {"gdn", "kda", "projection_only", "unmodeled"}:
                    p_entry = prev_tmpls_fast.get(akey)
                    if p_entry is not None and not any(
                            meta_scan_flags(op.metadata)[2] for op in p_entry[4]):
                        reused_archs[akey] = p_entry
            if len(reused_archs) == len(unique_archs) and self._prev_work_ops is not None:
                prev_sel_c = prev_state[6]
                return _frontend.IterationWork(
                    ops=self._prev_work_ops,
                    plan=plan, stage_count=runtime.parallel.pp,
                    valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                    recipe="structural", replay_scope=scope,
                    implementation_selections=(prev_sel_c[2] if prev_sel_c is not None else ()),
                )
            if not selections and prev_state[8]:
                prev_raw_pre, layer_offsets, prev_attn_splits, prev_final_ops = prev_state[8]
                active_fast = [
                    rep_idx for akey, rep_idx in unique_archs.items()
                    if akey not in reused_archs
                ]
                if (self._prev_work_ops is prev_final_ops
                        and pre_ops == prev_raw_pre
                        and all(arch_for_layer[r] in prev_attn_splits for r in active_fast)):
                    local_requests = tuple(v.request_id for v in local_views)
                    residual = self.residuals["plain"]
                    attention_state = ({}, {}, {}, {})
                    splice_ok = True
                    new_arch_templates = dict(prev_tmpls_fast)
                    new_attn_splits = dict(prev_attn_splits)
                    changed_by_arch = {}
                    for rep_idx in active_fast:
                        lyr = layers[rep_idx]
                        akey = arch_for_layer[rep_idx]
                        in_op, expected_done_id, prev_tmpl_attn = prev_attn_splits[akey]
                        in_id = in_op.op_id
                        attn_raw, done_id, _ = _representative_attention(
                            self, residual, layer=lyr, rep_idx=rep_idx, local_views=local_views,
                            local_requests=local_requests, local_tokens=local_tokens, mode=mode,
                            model=model, runtime=runtime, parallel=p, features=features,
                            source_rows_by_replica=source_rows_by_replica, machine=machine,
                            ready=in_id, attention_state=attention_state,
                        )
                        if (done_id != expected_done_id
                                or any(dict(op.metadata).get("metadata_schedule") == "pre_forward"
                                       for op in attn_raw)):
                            splice_ok = False
                            break
                        solo_attn = _lower_representative_attention(
                            self, attn_raw, in_op=in_op, model=model, runtime=runtime,
                            valid=valid, padded=padded, local_tokens=local_tokens,
                            assembly=assembly, scope=scope, resource_fusion=resource_fusion,
                        )
                        if (len(solo_attn) != len(prev_tmpl_attn)
                                or not solo_attn
                                or solo_attn[-1].op_id != expected_done_id
                                or any(op.layer != rep_idx for op in solo_attn)):
                            splice_ok = False
                            break
                        changed_pos = tuple(
                            pos for pos in range(len(solo_attn))
                            if solo_attn[pos] != prev_tmpl_attn[pos]
                        )
                        (p_rep_idx, p_pfx, p_src_prev,
                         p_out_sfx, p_tmpl_ops, p_meta_plans) = prev_tmpls_fast[akey]
                        if changed_pos:
                            merged_attn = tuple(
                                prev_tmpl_attn[pos] if solo_attn[pos] == prev_tmpl_attn[pos] else solo_attn[pos]
                                for pos in range(len(solo_attn))
                            )
                            new_tmpl_ops = merged_attn + p_tmpl_ops[len(solo_attn):]
                            new_arch_templates[akey] = (
                                p_rep_idx, p_pfx, p_src_prev, p_out_sfx, new_tmpl_ops, p_meta_plans,
                            )
                            new_attn_splits[akey] = (in_op, expected_done_id, merged_attn)
                            changed_by_arch[akey] = (
                                p_rep_idx, p_pfx, p_src_prev, p_out_sfx,
                                merged_attn, changed_pos, p_meta_plans,
                            )
                    if splice_ok:
                        if not changed_by_arch:
                            return _frontend.IterationWork(
                                ops=prev_final_ops,
                                plan=plan, stage_count=runtime.parallel.pp,
                                valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                                recipe="structural", replay_scope=scope,
                                implementation_selections=(),
                            )
                        out_ops = list(prev_final_ops)
                        new_layers_ops = list(prev_state[3])
                        cur_prev = body_ready
                        step_meta_keys_fast = {}
                        for idx in range(len(layers)):
                            akey = arch_for_layer[idx]
                            ch = changed_by_arch.get(akey)
                            if ch is None:
                                cur_prev = f"s0.l{idx}." + new_arch_templates[akey][3]
                                continue
                            (rep_idx, src_pfx, src_prev, out_sfx,
                             solo_attn, changed_pos, meta_plans) = ch
                            dst_pfx = f"s0.l{idx}."
                            base_off = layer_offsets[idx]
                            lyr_list = list(new_layers_ops[idx])
                            for pos in changed_pos:
                                tmpl_op = solo_attn[pos]
                                if idx == rep_idx and cur_prev == src_prev:
                                    retargeted = tmpl_op
                                else:
                                    retargeted = _retarget_operator(
                                        tmpl_op, akey=akey, pos=pos, meta_plans=meta_plans,
                                        src_pfx=src_pfx, src_layer=rep_idx, src_prev=src_prev,
                                        dst_pfx=dst_pfx, dst_layer=idx, dst_prev=cur_prev,
                                        step_meta_keys=step_meta_keys_fast, metadata_key_helpers=metadata_key_helpers,
                                    )
                                out_ops[base_off + pos] = retargeted
                                lyr_list[pos] = retargeted
                            new_layers_ops[idx] = tuple(lyr_list)
                            cur_prev = dst_pfx + out_sfx
                        final_ops = tuple(out_ops)
                        self._prev_work_ops = final_ops
                        self._prev_arch_state = (
                            arch_for_layer, body_ready, new_arch_templates,
                            tuple(new_layers_ops), prev_state[4], prev_state[5],
                            None, decode_sig,
                            (prev_raw_pre, layer_offsets, new_attn_splits, final_ops),
                        )
                        return _frontend.IterationWork(
                            ops=final_ops,
                            plan=plan, stage_count=runtime.parallel.pp,
                            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
                            recipe="structural", replay_scope=scope,
                            implementation_selections=(),
                        )

        # Lower the graph with one representative per active archetype.
        red_ops = list(pre_ops)
        active_rep_indices = [
            rep_idx for akey, rep_idx in unique_archs.items()
            if akey not in reused_archs
        ]
        cur_prev, arch_input_id, arch_output_id, raw_attn_by_arch = _build_representative_body(
            self, red_ops, active_rep_indices=active_rep_indices, layers=layers,
            arch_for_layer=arch_for_layer, body_ready=body_ready, runtime=runtime,
            local_views=local_views, local_tokens=local_tokens, model=model, mode=mode,
            features=features, realizations=realizations, source_rows_by_replica=source_rows_by_replica,
            body_global_tokens=body_global_tokens, physical_rank=physical_rank, machine=machine,
            parallel=p,
        )

        if any(dict(op.metadata).get("metadata_schedule") == "pre_forward" for op in red_ops):
            return None

        red_body_last = cur_prev
        heads, post_prev = self.logits(
            model, runtime, views, red_body_last, mode=mode,
            input_tokens=padded, normalized=False,
        )
        red_ops.extend(heads)
        _append_service_tail(red_ops, plan, views, post_prev, stage=0, mode=mode, requests=requests)

        red_ops = _frontend.annotate_geometry(red_ops, active_rows=valid, physical_rows=padded)
        regions = (_frontend.LoweringRegion(tuple(red_ops), runtime, model.activation_dtype,
                                  local_tokens, ()),)
        if assembly == "model":
            regions = _frontend._without_service_fragments(regions)
        lowered_red, _ = _frontend.lower_regions(
            regions, graph=self.graph, replay_scope=scope, runtime=runtime,
            prefix=(entry,), resource_fusion=resource_fusion,
        )

        # Partition lowered_red into pre_lowered, per-rep-layer templates, and post_lowered
        lowered_pre = []
        lowered_post = []
        lowered_by_rep: dict[int, list] = {r: [] for r in active_rep_indices}
        seen_body = False
        for op in lowered_red:
            if op.layer >= 0:
                seen_body = True
                if op.layer not in lowered_by_rep:
                    return None
                lowered_by_rep[op.layer].append(op)
            elif not seen_body:
                lowered_pre.append(op)
            else:
                lowered_post.append(op)

        arch_templates = dict(reused_archs)
        for akey, rep_idx in unique_archs.items():
            if akey in reused_archs:
                continue
            tmpl_ops = tuple(lowered_by_rep[rep_idx])
            if not tmpl_ops:
                return None
            rep_pfx = f"s0.l{rep_idx}."
            src_prev_id = arch_input_id[akey]
            out_id = arch_output_id[akey]
            # Confirm the archetype's terminal output op_id survived lowering
            tmpl_ids = {op.op_id for op in tmpl_ops}
            if out_id not in tmpl_ids:
                out_id = tmpl_ops[-1].op_id
            if not out_id.startswith(rep_pfx):
                return None
            arch_templates[akey] = (
                rep_idx, rep_pfx, src_prev_id,
                out_id[len(rep_pfx):], tmpl_ops, [False] * len(tmpl_ops),
            )

        # Verify lowered_red's last body output matches the last active rep's output
        if active_rep_indices:
            last_rep_akey = arch_for_layer[active_rep_indices[-1]]
            _, last_rep_pfx, _, last_out_sfx, _, _ = arch_templates[last_rep_akey]
            red_last_lowered_id = last_rep_pfx + last_out_sfx
        else:
            red_last_lowered_id = body_ready

        unchanged_by_arch = {}
        prev_layers_ops = None
        prev_sel_cache = None
        if can_reuse_prev:
            (_, _, prev_tmpls, prev_layers_ops,
             prev_pre_ops, prev_post_ops, prev_sel_cache, _, _) = prev_state
            for akey, (_, _, _, out_sfx, tmpl_ops, _) in arch_templates.items():
                if akey in reused_archs:
                    unchanged_by_arch[akey] = (True,) * len(tmpl_ops)
                    continue
                p_entry = prev_tmpls.get(akey)
                if p_entry is None or p_entry[3] != out_sfx or len(p_entry[4]) != len(tmpl_ops):
                    can_reuse_prev = False
                    prev_sel_cache = None
                    break
                unchanged_by_arch[akey] = tuple(
                    a == b for a, b in zip(tmpl_ops, p_entry[4])
                )

        if can_reuse_prev and len(lowered_pre) == len(prev_pre_ops):
            lowered_pre = [
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev_pre_ops, lowered_pre)
            ]

        expanded = list(lowered_pre)
        cur_layers_ops = []
        layer_offsets_list = []
        cur_prev = body_ready
        step_meta_keys = {}
        for idx in range(len(layers)):
            layer_offsets_list.append(len(expanded))
            akey = arch_for_layer[idx]
            rep_idx, src_pfx, src_prev, out_sfx, tmpl_ops, meta_plans = arch_templates[akey]
            dst_pfx = f"s0.l{idx}."
            unchanged = unchanged_by_arch.get(akey) if can_reuse_prev else None
            if unchanged is not None and all(unchanged):
                lyr_ops = prev_layers_ops[idx]
                expanded.extend(lyr_ops)
                cur_layers_ops.append(lyr_ops)
            elif idx == rep_idx and cur_prev == src_prev and unchanged is None:
                expanded.extend(tmpl_ops)
                cur_layers_ops.append(tmpl_ops)
            else:
                prev_lyr = prev_layers_ops[idx] if unchanged is not None else None
                built_lyr = []
                for pos, tmpl_op in enumerate(tmpl_ops):
                    if unchanged is not None and unchanged[pos]:
                        built_lyr.append(prev_lyr[pos])
                        continue
                    if idx == rep_idx and cur_prev == src_prev:
                        built_lyr.append(tmpl_op)
                        continue
                    built_lyr.append(_retarget_operator(
                        tmpl_op, akey=akey, pos=pos, meta_plans=meta_plans,
                        src_pfx=src_pfx, src_layer=rep_idx, src_prev=src_prev,
                        dst_pfx=dst_pfx, dst_layer=idx, dst_prev=cur_prev,
                        step_meta_keys=step_meta_keys, metadata_key_helpers=metadata_key_helpers,
                    ))
                lyr_tup = tuple(built_lyr)
                expanded.extend(lyr_tup)
                cur_layers_ops.append(lyr_tup)
            cur_prev = dst_pfx + out_sfx

        built_post = []
        if cur_prev == red_last_lowered_id:
            built_post = list(lowered_post)
        else:
            for post_op in lowered_post:
                new_deps = tuple(
                    cur_prev if d == red_last_lowered_id else d
                    for d in post_op.deps
                )
                new_meta = _frontend._retarget_id_only(
                    post_op.metadata, red_last_lowered_id, cur_prev)
                if new_deps != post_op.deps or new_meta is not post_op.metadata:
                    built_post.append(_frontend.replace(post_op, deps=new_deps, metadata=new_meta))
                else:
                    built_post.append(post_op)
        if can_reuse_prev and len(built_post) == len(prev_post_ops):
            built_post = [
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev_post_ops, built_post)
            ]
        expanded.extend(built_post)

        raw_final_ops = tuple(expanded)
        split_pack = None
        if selections:
            final_ops, final_selections, sel_cache = _finalize_repeated_selections(
                self, raw_final_ops, selections, prev_sel_cache if can_reuse_prev else None)
        else:
            final_selections = ()
            final_ops = raw_final_ops
            self._prev_work_ops = final_ops
            sel_cache = None
            if (decode_sig is not None
                    and not (prev_state is not None
                             and prev_state[7] == decode_sig
                             and prev_state[8] is False)):
                attn_split_info = {}
                prev_splits_old = (
                    prev_state[8][2]
                    if (prev_state is not None and prev_state[8])
                    else {}
                )
                for akey in reused_archs:
                    if akey in prev_splits_old:
                        attn_split_info[akey] = prev_splits_old[akey]
                op_by_id_ann = {entry.op_id: entry}
                for rop in red_ops:
                    op_by_id_ann[rop.op_id] = rop
                for lop in lowered_red:
                    op_by_id_ann[lop.op_id] = lop
                all_verified = True
                for akey, (in_id, attn_sub_raw, attn_done_id) in raw_attn_by_arch.items():
                    in_op = op_by_id_ann.get(in_id)
                    if in_op is None:
                        all_verified = False
                        break
                    tmpl_ops = arch_templates[akey][4]
                    split_k = -1
                    for k_idx, top in enumerate(tmpl_ops):
                        if top.op_id == attn_done_id:
                            split_k = k_idx
                            break
                    if split_k < 0:
                        all_verified = False
                        break
                    tmpl_attn = tmpl_ops[:split_k + 1]
                    tmpl_ffn = tmpl_ops[split_k + 1:]
                    internal_attn_ids = {top.op_id for top in tmpl_attn} - {attn_done_id}
                    cross_dep = any(
                        (d in internal_attn_ids) for fop in tmpl_ffn for d in fop.deps
                    )
                    if not cross_dep and internal_attn_ids:
                        for fop in tmpl_ffn:
                            if fop.metadata:
                                s_meta = str(fop.metadata)
                                if any(aid in s_meta for aid in internal_attn_ids):
                                    cross_dep = True
                                    break
                    if cross_dep:
                        all_verified = False
                        break
                    solo_attn = _lower_representative_attention(
                        self, attn_sub_raw, in_op=in_op, model=model, runtime=runtime,
                        valid=valid, padded=padded, local_tokens=local_tokens,
                        assembly=assembly, scope=scope, resource_fusion=resource_fusion,
                    )
                    if solo_attn == tmpl_attn:
                        attn_split_info[akey] = (in_op, attn_done_id, tmpl_attn)
                    else:
                        all_verified = False
                        break
                split_pack = (
                    (pre_ops, tuple(layer_offsets_list), attn_split_info, final_ops)
                    if all_verified else False
                )
            elif decode_sig is not None:
                split_pack = False
        self._prev_arch_state = (
            arch_for_layer, body_ready, arch_templates,
            tuple(cur_layers_ops), tuple(lowered_pre), tuple(built_post),
            sel_cache, decode_sig, split_pack,
        )
        return _frontend.IterationWork(
            ops=final_ops,
            plan=plan, stage_count=runtime.parallel.pp,
            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
            recipe="structural", replay_scope=scope,
            implementation_selections=final_selections,
        )


def _lower_representative_attention(
    frontend, ops, *, in_op, model, runtime, valid, padded, local_tokens,
    assembly, scope, resource_fusion,
):
    annotated = _frontend.annotate_geometry(ops, active_rows=valid, physical_rows=padded)
    regions = (_frontend.LoweringRegion(tuple(annotated), runtime, model.activation_dtype, local_tokens, ()),)
    if assembly == "model":
        regions = _frontend._without_service_fragments(regions)
    lowered, _ = _frontend.lower_regions(
        regions, graph=frontend.graph, replay_scope=scope, runtime=runtime,
        prefix=(in_op,), resource_fusion=resource_fusion,
    )
    return lowered[1:] if lowered and lowered[0].op_id == in_op.op_id else lowered


def _retarget_operator(
    op, *, akey, pos, meta_plans, src_pfx, src_layer, src_prev, dst_pfx,
    dst_layer, dst_prev, step_meta_keys, metadata_key_helpers,
):
    op_meta_key_from_metadata, seed_meta_id_key = metadata_key_helpers
    deps = tuple(
        dst_prev if dep == src_prev else dep.replace(src_pfx, dst_pfx) if src_pfx in dep else dep
        for dep in op.deps
    )
    plan = meta_plans[pos]
    if plan is False:
        plan = _frontend._compile_meta_plan(op.metadata, src_pfx, src_layer, src_prev)
        meta_plans[pos] = plan
    metadata = op.metadata
    if plan is not None:
        metadata = _frontend._apply_meta_plan(metadata, plan, src_pfx, dst_pfx, dst_layer, dst_prev)
        if dst_layer > 0:
            key_slot = (akey, pos)
            key = step_meta_keys.get(key_slot)
            if key is None:
                key = op_meta_key_from_metadata(
                    op.metadata if src_layer > 0 or not _frontend._plan_uses_prev_id(plan) else metadata)
                step_meta_keys[key_slot] = key
            seed_meta_id_key(metadata, key)
    return _frontend.replace(op, op_id=op.op_id.replace(src_pfx, dst_pfx), deps=deps,
                   layer=dst_layer, metadata=metadata)


def _plan_uses_prev_id(plan) -> bool:
    if plan == 0:
        return True
    if isinstance(plan, int):
        return False
    kind = plan[0]
    if kind in (2, _META_LIST_EDITS, _META_DICT_VALUE_EDITS):
        return any(_frontend._plan_uses_prev_id(sub) for _, sub in plan[1])
    return any(
        (kp == 0) or (vp is not None and _frontend._plan_uses_prev_id(vp))
        for _, kp, vp in plan[1]
    )


def _compile_meta_plan(val, src_pfx: str, src_layer: int, src_prev: str):
    """Compile a pruned subtree retargeting plan touching only layer-dependent nodes."""
    if isinstance(val, str):
        if val == src_prev:
            return 0
        if src_pfx in val:
            return 1
        return None
    if isinstance(val, tuple):
        edits = []
        for idx, item in enumerate(val):
            sub = _frontend._compile_meta_plan(item, src_pfx, src_layer, src_prev)
            if sub is not None:
                edits.append((idx, sub))
        return (2, tuple(edits)) if edits else None
    if isinstance(val, list):
        edits = []
        for idx, item in enumerate(val):
            sub = _frontend._compile_meta_plan(item, src_pfx, src_layer, src_prev)
            if sub is not None:
                edits.append((idx, sub))
        return (_META_LIST_EDITS, tuple(edits)) if edits else None
    if isinstance(val, dict):
        key_edits = False
        val_edits = []
        kv_items = []
        for k, v in val.items():
            kp = None
            if isinstance(k, str):
                if k == src_prev:
                    kp = 0
                elif src_pfx in k:
                    kp = 1
            if k == "layer" and type(v) is int and v == src_layer:
                vp = _META_LAYER_INDEX
            else:
                vp = _frontend._compile_meta_plan(v, src_pfx, src_layer, src_prev)
            if kp is not None:
                key_edits = True
            if vp is not None:
                val_edits.append((k, vp))
            kv_items.append((k, kp, vp))
        if not key_edits and not val_edits:
            return None
        if not key_edits:
            return (_META_DICT_VALUE_EDITS, tuple(val_edits))
        return (_META_DICT_KEY_AND_VALUE_EDITS, tuple(kv_items))
    return None


def _apply_meta_plan(val, plan, src_pfx: str, dst_pfx: str,
                     dst_layer: int, dst_prev: str):
    if plan == 0:
        return dst_prev
    if plan == 1:
        return val.replace(src_pfx, dst_pfx)
    if plan == _META_LAYER_INDEX:
        return dst_layer
    kind = plan[0]
    if kind == 2:
        out = list(val)
        for idx, sub in plan[1]:
            out[idx] = _frontend._apply_meta_plan(
                val[idx], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return tuple(out)
    if kind == _META_LIST_EDITS:
        out = list(val)
        for idx, sub in plan[1]:
            out[idx] = _frontend._apply_meta_plan(
                val[idx], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return out
    if kind == _META_DICT_VALUE_EDITS:
        out = val.copy()
        for k, sub in plan[1]:
            out[k] = _frontend._apply_meta_plan(
                val[k], sub, src_pfx, dst_pfx, dst_layer, dst_prev)
        return out
    out = {}
    for k, kp, vp in plan[1]:
        nk = (dst_prev if kp == 0 else k.replace(src_pfx, dst_pfx)) if kp is not None else k
        nv = (_frontend._apply_meta_plan(val[k], vp, src_pfx, dst_pfx, dst_layer, dst_prev)
              if vp is not None else val[k])
        out[nk] = nv
    return out


def _retarget_id_only(val, old_id: str, new_id: str):
    if isinstance(val, str):
        return new_id if val == old_id else val
    if isinstance(val, tuple):
        changed = False
        out = []
        for item in val:
            new_item = _frontend._retarget_id_only(item, old_id, new_id)
            if new_item is not item:
                changed = True
            out.append(new_item)
        return tuple(out) if changed else val
    if isinstance(val, list):
        changed = False
        out = []
        for item in val:
            new_item = _frontend._retarget_id_only(item, old_id, new_id)
            if new_item is not item:
                changed = True
            out.append(new_item)
        return out if changed else val
    if isinstance(val, dict):
        changed = False
        out = {}
        for k, v in val.items():
            nk = new_id if k == old_id else k
            nv = _frontend._retarget_id_only(v, old_id, new_id)
            if nk is not k or nv is not v:
                changed = True
            out[nk] = nv
        return out if changed else val
    return val
