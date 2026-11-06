"""MC2 capacity, masked routing, native scale traffic and causal execution."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.family_mc2_unfused import IMPLEMENTATION, METADATA, lower
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_matrix_contracts import matrix_comparison_scope


def case(*, selected=True, masked=True):
    kernels = {
        "moe_token_partition": {"kind": "vllm_ascend_mc2_tp_slice", "padding": "right_to_tp"},
        "native_moe_protocol": {"kind": "arch35_fullmesh", "max_inflight_per_rank": 4,
                                "dispatch_quant": "per_token_int8"},
        "lowerings": {},
    }
    if selected:
        kernels["lowerings"]["grouped_int8_caller"] = {
            "implementation": IMPLEMENTATION, "options": {"global_bs": 0}}
    if masked:
        kernels["force_eplb_source"] = {"kind": "vllm_ascend_mc2_tp", "global_bs": 0}
    return {
        "machine": {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 4},
        "model": {"catalog_key": "qwen3.5-35b-a3b", "quantization": "w8a8_dynamic", "layers": 1},
        "runtime": {"tp": 4, "ep": 4, "routing": {"policy": "balanced"},
                    "graph_mode": "auto", "graph_capture_mode": "full",
                    "graph_capture_keys": [4, 8, 12, 16, 24, 32], "kernels": kernels},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4096, "output_tokens": 8}]},
    }


def graph(raw, sequences=7, scope="body"):
    return target_graphs(raw, phase="decode", tokens=4, sequences=sequences,
                         context=4096, scope=scope)[0]


def source_histogram(actual, physical):
    """Enumerate the pinned force-EPLB cursor and active-prefix mask independently."""
    local, ep, top_k, experts = physical // 4, 4, 8, 256
    counts = [0] * experts
    for rank in range(ep):
        for token in range(local):
            if rank * local + token >= actual:
                continue
            for choice in range(top_k):
                cursor = (local * top_k + 1) * rank + token * top_k + choice
                expert = (cursor % ep) * (experts // ep) + (cursor // ep) % (experts // ep)
                counts[expert] += 1
    return tuple(counts[:64])


@pytest.mark.parametrize("sequences,physical", [(1, 4), (4, 16), (7, 32), (8, 32)])
def test_unfused_storage_scans_capacity_without_inflating_valid_gemm(sequences, physical):
    raw = case(); original = deepcopy(raw)
    ops = {o.op_id: o for o in graph(raw, sequences)}
    counts = source_histogram(sequences * 4, physical)
    valid, active, capacity = sum(counts), sum(n > 0 for n in counts), physical * 8
    gate, down, activation, quant, cast = (ops["s0.l0." + key] for key in (
        "expert_up", "expert_down", "expert_act", "expert_down.quantize", "expert_up.weight_scale_cast"))
    assert raw == original
    for matrix, n, k in ((gate, 1024, 2048), (down, 2048, 512)):
        meta = dict(matrix.metadata)
        assert matrix.local_shape == matrix.launch_shape == (valid, n, k)
        assert matrix.padded_shape == (capacity, n, k)
        assert tuple(meta["matrix_group_rows"]) == counts
        assert matrix.flops == 2 * valid * n * k
        assert matrix.mte2_bytes == valid * k + active * n * (k + 2) + valid * 4
        assert matrix.mte3_bytes == valid * n * 2
        assert meta["weight_scale_format"] == "bf16"
        assert meta["matrix_epilogue"]["column_scale_bytes"] == active * n * 2
        assert meta[METADATA]["graph_active_query_rows"] == sequences * 4
        assert meta[METADATA]["graph_physical_query_rows"] == physical
    assert cast.local_shape == (64, 1024)
    assert (cast.mte2_bytes, cast.mte3_bytes) == (64 * 1024 * 4, 64 * 1024 * 2)
    assert cast.deps == ("s0.l0.dispatch",)
    assert cast.op_id in gate.deps
    for pointwise in (activation, quant):
        assert pointwise.local_shape == (valid, 512)
        assert pointwise.padded_shape == pointwise.launch_shape == (capacity, 512)
    assert (activation.mte2_bytes, activation.mte3_bytes) == (capacity * 1024 * 2, capacity * 512 * 2)
    assert activation.vector_ops == capacity * 512 * 4
    assert (quant.mte2_bytes, quant.mte3_bytes) == (capacity * 512 * 2, capacity * 512 + capacity * 4)
    assert quant.vector_ops == capacity * (4 * 512 + 2)
    assert dict(quant.metadata)["activation_scale_shape"] == (capacity,)


@pytest.mark.parametrize("scope", ["body", "model"])
def test_q28_physical32_requires_mask_and_keeps_valid_group_rows56(scope):
    with pytest.raises(ValueError, match="padded_call_requires_explicit_source_mask_ledger"):
        graph(case(masked=False), scope=scope)
    ops = {o.op_id: o for o in graph(case(), scope=scope)}
    gate = ops["s0.l0.expert_up"]; info = dict(gate.metadata)[METADATA]
    assert gate.local_shape[0] == info["valid_rows"] == 56
    assert info["capacity_rows"] == 256
    assert (info["graph_active_query_rows"], info["graph_physical_query_rows"]) == (28, 32)
    ledger = dict(gate.metadata)["routing_realization"]
    assert ledger["valid_tokens_per_rank"] == (8, 8, 8, 4)
    assert sum(ledger["global_expert_rows"]) == 28 * 8
    assert ops["s0.l0.router"].local_shape[0] == 32
    assert ops["s0.l0.topk"].local_shape[0] == 8
    payload = next(o for o in ops.values() if o.op_id == "s0.l0.dispatch.payload_network")
    assert dict(payload.metadata)["source_force_eplb"]["valid_global_tokens"] == 28


def test_native_capture_allocation_matches_export_without_claiming_group_values():
    capture = json.loads((Path(__file__).parent / "fixtures/mc2_unfused/a5_qwen35_q28_physical32.json").read_text())
    ops = {o.op_id: o for o in graph(case())}
    native = {n["abi"]["type"]: n["abi"] for n in capture["native"]}
    assert native["SwiGlu"]["input_shapes"] == '"256,1024"'
    assert predicted_contract(ops["s0.l0.expert_act"].to_dict()) == ("swiglu", (256, 512), "bf16")
    quant_contract = predicted_contract(ops["s0.l0.expert_down.quantize"].to_dict())
    assert quant_contract[2:] == (((256, 512), "bf16", "ND"), ((256, 512), "int8", "ND"), ((256,), "fp32", "ND"))
    gate_contract = predicted_contract(ops["s0.l0.expert_up"].to_dict())
    assert gate_contract[:2] == ("grouped_matrix", (64, 256, 1024, 2048))
    assert gate_contract[-2:] == ((256,), "fp32")
    scope = matrix_comparison_scope(ops["s0.l0.expert_up"].to_dict())
    assert "valid_grouped_rows_not_proven_by_tensor_capacity" in scope["not_compared"]
    cast_contract = predicted_contract(ops["s0.l0.expert_up.weight_scale_cast"].to_dict())
    assert cast_contract == ("native_tensor_abi", "Cast", (((64, 1024), "fp32", "ND"),),
                             (((64, 1024), "bf16", "ND"),))


def test_undeclared_path_preserves_old_work_and_selected_rewrite_is_idempotent():
    raw = case(selected=False)
    old = graph(raw); old_by_id = {o.op_id: o for o in old}
    assert old_by_id["s0.l0.expert_act"].local_shape == (56, 512)
    assert old_by_id["s0.l0.expert_act"].mte2_bytes == 56 * 1024 * 2
    assert not any(o.recipe == "expert_weight_scale_cast" for o in old)
    selected = case(); ops = graph(selected)
    assert lower(ops, selected["runtime"]["kernels"]["lowerings"]) == ops
    assert sum(o.flops for o in ops) == sum(o.flops for o in old)


@pytest.mark.parametrize("global_bs", [None, False, 32])
def test_allocation_policy_must_be_explicit_uniform_global_bs_zero(global_bs):
    raw = case()
    raw["runtime"]["kernels"]["lowerings"]["grouped_int8_caller"]["options"] = (
        {} if global_bs is None else {"global_bs": global_bs})
    with pytest.raises(ValueError, match="explicit_global_bs0"):
        graph(raw)


def test_unfused_contract_rejects_fused_selection_and_foreign_receiver():
    raw = case()
    raw["runtime"]["kernels"]["lowerings"]["native_fusion"] = {
        "implementation": "declared_native_fusion", "options": {"roles": ["grouped_gate_up_activation_quant"]}}
    with pytest.raises(ValueError, match="cannot_select_grouped_fusion"):
        graph(raw)
    ops = graph(case(selected=False))
    changed = tuple(replace(o, metadata=tuple({**dict(o.metadata), "activation_scale_producer": "unrelated"}.items()))
                    if o.op_id == "s0.l0.expert_up" else o for o in ops)
    with pytest.raises(ValueError, match="owned_mc2_int8_receiver"):
        lower(changed, case()["runtime"]["kernels"]["lowerings"])


def test_cast_and_allocated_pointwise_wait_for_received_inputs():
    raw = case(); ops = graph(raw)
    selected = [o for o in ops if o.op_id in {"s0.l0.expert_up.weight_scale_cast", "s0.l0.expert_up",
                "s0.l0.expert_act", "s0.l0.expert_down.quantize", "s0.l0.expert_down"}]
    result = graph_costs(raw, {0: selected}, input_ready_ms=7,
                         external_inputs={"s0.l0.dispatch": 7})
    times = {r["op_id"]: r for r in result["timeline"]}
    assert min(r["ready_ms"] for r in times.values()) >= 7
    for before, after in zip(selected, selected[1:]):
        assert times[after.op_id]["ready_ms"] >= times[before.op_id]["complete_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["execution"]["work"]["flops"] == sum(o.flops for o in selected)
