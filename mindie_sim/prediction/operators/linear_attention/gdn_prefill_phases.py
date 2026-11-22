"""Extracted state-slot resolution and causal-conv phases for Arch35 GDN prefill.

[Role]
  Resolve dynamic/bound recurrent-state slot indices and lower both PCP-segmented
  and mixed (decode+prefill) causal-convolution stages for ``gdn_prefill``.

[Invariants]
  * Padding slots retain physical row positions in ``qsl`` while reading and
    writing zero persistent state rows.
  * PCP segmented query ranges advance local conv history per range and emit
    explicit ``state_transfer`` collectives across gaps.

[Timeouts & Locks]
  O(sequences + query segments) metadata construction; no I/O or locks.

[Side-Effect Policy]
  Appends operators only to the caller's ``Builder``.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.prediction.operators.causal_conv import native_causal_conv_prefill
from mindie_sim.prediction.operators.linear_attention import gdn_recurrent


def state_slot_inputs(runtime: Any, ordered: tuple[Any, ...]):
    """Resolve index presence, optional values and optional capacity."""
    metadata = dict(runtime.metadata)
    capacity = metadata.get("gdn_state_rows", metadata.get("gdn_state_capacity"))
    if capacity is not None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("gdn_prefill_state_capacity_invalid")
    if all(view.padding for view in ordered):
        return ((0,) * len(ordered), capacity, "padding_null", 0)
    raw = metadata.get("gdn_prefill_state_indices", metadata.get("gdn_state_indices"))
    fixture = bool(metadata.get("gdn_state_uniform_fixture", False))
    if raw is None and not fixture:
        active = sum(1 for view in ordered if not view.padding)
        if capacity is not None and capacity < active:
            raise ValueError("gdn_prefill_state_capacity_below_active_slots")
        return (tuple(0 if view.padding else None for view in ordered),
                capacity, "dynamic_caller_index", 0)
    if raw == "uniform" or raw is None:
        indices = tuple(range(1, len(ordered) + 1))
        source = "uniform_fixture"
    elif isinstance(raw, dict):
        try:
            indices = tuple(
                0 if view.padding else (
                    None if raw[view.request_id] is None else int(raw[view.request_id])
                )
                for view in ordered
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("gdn_prefill_state_indices_missing_request")
        source = "caller_metadata"
    elif isinstance(raw, (list, tuple)):
        if len(raw) != len(ordered):
            raise ValueError("gdn_prefill_state_indices_length_mismatch")
        try:
            indices = tuple(
                0 if view.padding else (None if value is None else int(value))
                for view, value in zip(ordered, raw)
            )
        except (TypeError, ValueError):
            raise ValueError("gdn_prefill_state_indices_must_be_integers")
        source = "caller_metadata"
    else:
        raise ValueError("gdn_prefill_state_indices_invalid")
    known = tuple(index for index in indices if index is not None and index != 0)
    if any(index < 1 for index in known):
        raise ValueError("gdn_prefill_state_indices_must_be_positive")
    if capacity is not None and max(known, default=0) >= capacity:
        raise ValueError("gdn_prefill_state_index_exceeds_capacity")
    return tuple(indices), capacity, source, 0


def _initial_conv_transfer(b, conv_input, *, view, plan, width, pcp, first_start, version, view_index=0):
    remote_initial = None
    if pcp > 1 and first_start > 0:
        op_name = "pcp_remote_conv_state" if view_index == 0 else f"pcp_remote_conv_state_{view_index}"
        remote_initial = b.work(
            op_name, (conv_input.producer,), (0,),
            family="state_transfer",
            metadata=(
                ("status", "pending_external_rank"),
                ("producer_scope", "external_rank"),
                ("completion_owner", "T09/COMM-LEAVES"),
                ("source_implementation", "pcp1_only"),
                ("request", view.request_id),
                ("from_version", view.context),
                ("to_version", version),
                ("state_value_output", gdn_recurrent.tensor_port(
                    TensorValue(
                        "", plan.conv_state_dtype or "bf16",
                        TensorLayout.contiguous((1, plan.conv_kernel - 1, width)),
                        binding="external",
                        binding_name=f"pcp_peer:{view.request_id}:conv@{version}",
                        state_version=version,
                    ),
                    ready="remote_transfer", role="state_input",
                    owner="external_rank",
                )),
            ),
        )
    return remote_initial


def _segment_remote_ports(*, plan, width, version, remote_initial, exchange, previous_seg):
    remote_port = (
        gdn_recurrent.tensor_port(
            TensorValue(
                str(remote_initial), plan.conv_state_dtype or "bf16",
                TensorLayout.contiguous((1, plan.conv_kernel - 1, width)),
                state_version=version,
            ),
            ready=str(remote_initial), role="remote_input",
            owner="external_rank",
        )
        if remote_initial is not None else None
    )
    remote_ports = (
        (remote_port,) if remote_port else (
            (gdn_recurrent.tensor_port(
                TensorValue(
                    str(exchange), plan.conv_state_dtype or "bf16",
                    TensorLayout.contiguous((1, plan.conv_kernel - 1, width)),
                    state_version=version,
                ),
                ready=str(exchange), role="remote_input",
                owner="external_rank",
            ),)
            if previous_seg is not None else ()
        )
    )
    return remote_ports


def _lower_segmented_conv(b, tensors, *, prefix, plan, parallel, prefill, conv_input,
                          conv_params, conv_meta, cache_indices, state_rows, index_source,
                          chunk, heads, width, actual):
    parts, offset = [], 0
    segment_index = 0
    for view_index, view in enumerate(prefill):
        previous_seg = None
        ranges = tuple(view.query_ranges)
        first_start = int(ranges[0][0]) if ranges else 0
        global_end = view.context + (view.global_q_len or view.q_len)
        version = view.context + first_start
        remote_initial = _initial_conv_transfer(
            b, conv_input, view=view, plan=plan, width=width, pcp=parallel.pcp,
            first_start=first_start, version=version, view_index=view_index)
        for range_index, (start, end) in enumerate(ranges):
            length = int(end) - int(start)
            if length <= 0:
                continue
            seg_input = conv_input.narrow(0, offset, length)
            seg_params = {
                **conv_params,
                "lengths": (length,),
                "cache_indices": (
                    cache_indices[view_index] if cache_indices is not None else None,
                ),
                "has_initial_state": (version > 0 and not view.padding,),
            }
            seg_op = native_causal_conv_prefill.make_op(seg_params)
            deps = [seg_input.producer]
            if range_index == 0 and remote_initial is not None:
                deps.append(remote_initial)
            exchange = None
            if previous_seg is not None:
                exchange = b.communicate(
                    f"pcp_conv_tail_gather{segment_index}", (conv_input.producer,),
                    max(0, plan.conv_kernel - 1) * width * 2 * (parallel.pcp - 1),
                    "allgather", "pcp",
                    state_scope="conv_history",
                    request=view.request_id,
                    from_version=version,
                    to_version=view.context + int(start),
                    source="vllm_ascend.ops.gdn",
                )
                deps.append(exchange)
            segment_context = view.context + int(start)
            segment_view = SimpleNamespace(
                request_id=f"{view.request_id}#convseg{segment_index}",
                q_len=length, context=segment_context, global_q_len=length,
                query_ranges=(), padding=False, phase="prefill",
                draft_accepted_tokens=0, draft_advanced_tokens=0,
            )
            seg_contract, seg_effects = gdn_recurrent.state_contract(
                views=(segment_view,), phase="prefill",
                state_dtype=plan.recurrent_state_dtype,
                conv_state_dtype=plan.conv_state_dtype,
                conv_kernel=plan.conv_kernel, target_prefix=prefix,
                chunk=chunk, initial_state=(version > 0,),
                direction="conv", target_kind="conv",
                global_lengths=(length,),
                slot_indices=((cache_indices[view_index]
                               if cache_indices is not None else None),),
                state_capacity=state_rows, slot_source=index_source,
                remote_publish=parallel.pcp > 1,
            )
            seg_contract = {
                **seg_contract,
                "request_id": view.request_id,
                "global_versions": (global_end,),
                "publish_versions": (None,),
                "publish_pending": "remote_producer" if parallel.pcp > 1 else "",
            }
            seg_contract, _ = gdn_recurrent.bind_state_ports(
                seg_contract, "", heads=heads, head_shards=1)
            meta = dict(conv_meta)
            seg_index_value = (None if cache_indices is None
                               else cache_indices[view_index])
            remote_ports = _segment_remote_ports(
                plan=plan, width=width, version=version, remote_initial=remote_initial,
                exchange=exchange, previous_seg=previous_seg)
            meta.update({
                "state_segment_index": segment_index,
                "state_segment_range": (int(start), int(end)),
                "state_segment_local_offset": offset,
                "state_segment_from_version": segment_context,
                "state_segment_to_version": segment_context + length,
                "state_segment_global_version": global_end,
                "state_value_inputs": (gdn_recurrent.tensor_port(
                    seg_input, ready=seg_input.producer, role="conv_input",
                    owner=seg_input.producer),),
                "state_remote_inputs": remote_ports,
                "state_ready": None,
                "cache_indices": (seg_index_value,),
                "cache_index_present": cache_indices is not None,
                "cache_index_bound": (cache_indices is not None
                                      and seg_index_value is not None),
                "state_rows": state_rows,
                "state_rows_bound": state_rows is not None,
            })
            seg_op = replace(seg_op, metadata=tuple(sorted({
                **dict(seg_op.metadata), **meta}.items())))
            seg = b.add(replace(
                seg_op,
                op_id=prefix + f".short_convolution_seg{segment_index}",
                deps=tuple(dict.fromkeys(deps)), **b.common,
            ))
            seg_contract, _ = gdn_recurrent.bind_state_ports(
                seg_contract, seg, heads=heads, head_shards=1)
            b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
                **dict(b.ops[-1].metadata),
                "state_ready": seg,
                "gdn_state_contract": seg_contract,
                "gdn_state_effects": seg_effects,
                "state_value_outputs": seg_contract["value_outputs"],
                "state_value_inputs": (gdn_recurrent.tensor_port(
                    seg_input, ready=seg_input.producer, role="conv_input",
                    owner=seg_input.producer),),
            }.items())))
            parts.append(tensors.reshape(
                tensors.result(seg, (length, width)), (length * width,),
                name=f"conv_seg_flat{segment_index}"))
            previous_seg = seg
            version += length
            offset += length
            segment_index += 1
    combined = tensors.concatenate_flat(parts, "pcp_conv_concatenate")
    return tensors.reshape(combined, (actual, width), name="pcp_conv_output").producer


def lower_conv_stage(
    *,
    builder: Any,
    tensors: Any,
    prefix: str,
    plan: Any,
    parallel: Any,
    ordered: tuple[Any, ...],
    prefill: tuple[Any, ...],
    decode: tuple[Any, ...],
    conv_input: TensorValue,
    conv_params: dict[str, Any],
    conv_op: Any,
    conv_meta: dict[str, Any],
    mixed_contract: dict[str, Any],
    mixed_conv: tuple[Any, ...],
    cache_indices: Any,
    state_rows: Any,
    index_source: str,
    chunk: int,
    heads: int,
    width: int,
    actual: int,
) -> str:
    """Lower PCP-segmented or mixed (decode+prefill) causal convolution."""
    b = builder
    pcp_ranges = (
        parallel.pcp > 1
        and not decode
        and all(not view.padding for view in prefill)
        and any(len(view.query_ranges) > 1 for view in prefill)
    )
    if pcp_ranges:
        return _lower_segmented_conv(
            b, tensors, prefix=prefix, plan=plan, parallel=parallel, prefill=prefill,
            conv_input=conv_input, conv_params=conv_params, conv_meta=conv_meta,
            cache_indices=cache_indices, state_rows=state_rows, index_source=index_source,
            chunk=chunk, heads=heads, width=width, actual=actual)


    remote_inputs = [None] * len(ordered)
    conv_deps = [conv_input.producer]
    for index, view in enumerate(ordered):
        if view.phase != "prefill":
            continue
        ranges = tuple(view.query_ranges)
        first_start = int(ranges[0][0]) if ranges else 0
        if parallel.pcp <= 1 or first_start <= 0:
            continue
        remote = b.work(
            "pcp_remote_conv_state", (conv_input.producer,), (0,),
            family="state_transfer",
            metadata=(
                ("status", "pending_external_rank"),
                ("producer_scope", "external_rank"),
                ("completion_owner", "T09/COMM-LEAVES"),
                ("source_implementation", "pcp1_only"),
                ("request", view.request_id),
                ("from_version", view.context),
                ("to_version", view.context + first_start),
            ),
        )
        remote_inputs[index] = gdn_recurrent.tensor_port(
            TensorValue(
                "", plan.conv_state_dtype or "bf16",
                TensorLayout.contiguous((1, plan.conv_kernel - 1, width)),
                binding="external",
                binding_name=f"pcp_peer:{view.request_id}:conv@{view.context + first_start}",
                state_version=view.context + first_start,
            ),
            ready=str(remote), role="remote_input", owner="external_rank",
        )
        conv_deps.append(remote)
    has_token = any(d == "token" for d in mixed_contract["slot_directions"])
    mixed_contract = {
        **mixed_contract,
        "candidate_storage": "history_ring+query_rows" if has_token else "",
        "candidate_recovery": (
            "accepted_prefix_from_conv_history_and_query_inputs" if has_token else ""
        ),
    }
    mixed_contract, _ = gdn_recurrent.bind_state_ports(
        mixed_contract, "", heads=heads, head_shards=1,
        remote_input=tuple(remote_inputs))
    conv_op = replace(conv_op, metadata=tuple(sorted({
        **dict(conv_op.metadata), **conv_meta,
        "state_remote_inputs": tuple(remote_inputs),
        "gdn_state_contract": mixed_contract,
        "gdn_state_effects": mixed_conv,
    }.items())))
    conv = b.add(replace(conv_op, op_id=prefix + ".short_convolution",
                         deps=tuple(dict.fromkeys(conv_deps)), **b.common))
    mixed_contract, _ = gdn_recurrent.bind_state_ports(
        mixed_contract, conv, heads=heads, head_shards=1,
        remote_input=tuple(remote_inputs))
    b.ops[-1] = replace(b.ops[-1], metadata=tuple(sorted({
        **dict(b.ops[-1].metadata),
        "state_ready": conv,
        "state_remote_inputs": tuple(remote_inputs),
        "state_value_inputs": mixed_contract.get("value_inputs"),
        "state_value_outputs": mixed_contract.get("value_outputs"),
        "state_candidate_stores": mixed_contract.get("candidate_stores"),
        "state_candidate_storage": {
            "kind": mixed_contract.get("candidate_storage", ""),
            "recovery": mixed_contract.get("candidate_recovery", ""),
            "history_slots": conv_params.get("state_history_slots"),
            "kernel": plan.conv_kernel,
            "query_rows": tuple(v.q_len for v in ordered if v.phase == "decode"),
            "op": conv,
        },
        "gdn_state_contract": mixed_contract,
        "gdn_state_effects": mixed_conv,
    }.items())))
    if parallel.pcp > 1:
        conv = b.communicate(
            "pcp_conv_state_step", (conv,),
            len(prefill) * max(0, plan.conv_kernel - 1) * width * 2
            * (2 * parallel.pcp - 1) / parallel.pcp,
            "allgather", "pcp")
    return conv
